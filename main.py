import os
import time
import numpy as np  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from segment_anything.build_sam import sam_model_registry
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
from dataset import build_dataset
from config import get_args_parser, setup_output_dirs
from utils import generate_click_prompt, random_box, l2_regularisation, elbo
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def get_optimizer(args, model):
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    return optimizer

def get_scheduler(args, optimizer):
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=0
        )
    return scheduler

def prepare_prompts(images, labels, args, devices):
    """
    Prepare prompts based on configuration
    """
    batched_input = []
    
    point_prompt = None
    box_prompt = None
    
    if args.prompt_type in ['point', 'both']:
        point_prompt = generate_click_prompt(
            labels,
            num_points=args.num_points,
            pt_label=args.point_label
        )
        
    if args.prompt_type in ['box', 'both']:
        box_prompt = random_box(
            labels,
            box_num=args.num_boxes,
            std=args.box_noise_std,
            max_pixel=args.box_noise_max
        ).to(devices)
    if args.dataset == 'refuge':
        original_size = (args.img_size, args.img_size)
    elif args.dataset == 'lidc':
        original_size = (128, 128)
    for i in range(images.size(0)):
        input_dict = {
            "image": images[i],
            "masks": labels[i],
            "original_size": original_size,
        }
        
        if point_prompt is not None:
            input_dict["point_coords"] = point_prompt['point_coords']
            input_dict["point_labels"] = point_prompt['point_labels']
            
        if box_prompt is not None:
            input_dict["boxes"] = box_prompt
            
        batched_input.append(input_dict)
        
    return batched_input

def train_one_epoch(model, train_loader, optimizer, scheduler, args, device, epoch, writer):
    model.train()
    total_loss = 0
    total_reg_loss = 0
    total_rec_loss = 0
    train_step = 0
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        if args.dataset == 'refuge':
            labels = batch['random_cup_label'].to(device)
        elif args.dataset == 'lidc':
            labels = batch['random_label'].to(device)
        
        batched_input = prepare_prompts(images, labels, args, device)
        
        outputs_tuple = model(
            batched_input=batched_input,
            multimask_output=False
        )
        outputs_list = outputs_tuple[0]
        masks = outputs_list[0]['masks'].float()
        # Calculate individual losses
        reg_loss = (l2_regularisation(model.Probabilistic_model.prior) + 
                   l2_regularisation(model.Probabilistic_model.posterior))
        
        # Total loss
        loss = elbo(masks, labels, outputs_tuple[1], beta=args.beta) + args.reg_weight * reg_loss
        loss = loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reconstruction_loss = elbo(masks, labels, outputs_tuple[1], beta=args.beta)  
        # Accumulate losses
        total_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_rec_loss += reconstruction_loss.item()
        
        # Log to tensorboard every N steps
        if batch_idx % 10 == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), step)
            writer.add_scalar('Train/RegLoss', reg_loss.item(), step)
            writer.add_scalar('Train/ReconstructionLoss', reconstruction_loss.item(), step)
        
        train_step += 1
        print(f"\rStep: {train_step}/{len(train_loader)}, Loss: {loss.item():.4f}", end='')
    
    # Calculate average losses
    avg_loss = total_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)
    avg_rec_loss = total_rec_loss / len(train_loader)
    scheduler.step()
    # Log epoch averages
    writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
    writer.add_scalar('Train/EpochRegLoss', avg_reg_loss, epoch)
    writer.add_scalar('Train/EpochReconLoss', avg_rec_loss, epoch)
    
    return avg_loss

@torch.no_grad()
def validate(model, val_loader, args, device, epoch, writer):
    model.eval()
    total_dice = 0
    total_loss = 0
    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean",
        get_not_nans=False,
        ignore_empty=False
    )
    
    for batch in val_loader:
        images = batch['image'].to(device)
        if args.mode == 'train':
            if args.dataset == 'refuge':
                labels = batch['random_cup_label'].to(device)
            elif args.dataset == 'lidc':
                labels = batch['random_label'].to(device)
        elif args.mode == 'val':
            if args.dataset == 'refuge':
                labels = batch['majorityvote_cup_label'].to(device)
            elif args.dataset == 'lidc':
                labels = batch['majorityvote_label'].to(device)
        batched_input = prepare_prompts(images, labels, args, device)
        
        outputs_tuple = model(
            batched_input=batched_input,
            multimask_output=False
        )
        outputs_list = outputs_tuple[0]
        masks = outputs_list[0]['masks'].float()
        
        # Calculate validation loss
        loss = elbo(masks, labels, outputs_tuple[1], beta=args.beta)
        total_loss += loss.item()
        
        masks = torch.sigmoid(masks)
        binary_masks = (masks > 0.5).float()
        dice_score = dice_metric(binary_masks, labels).to(device)
        total_dice += dice_score[0].item()
        
        # Optional: Log sample predictions periodically
        if writer is not None and epoch % 5 == 0:  # Every 5 epochs
            writer.add_images('Val/Images', images[:4], epoch)  # Log first 4 images
            writer.add_images('Val/TrueMasks', labels[:4], epoch)
            writer.add_images('Val/PredMasks', binary_masks[:4], epoch)
    
    avg_dice = total_dice / len(val_loader)
    avg_loss = total_loss / len(val_loader)
    
    if writer is not None:
        writer.add_scalar('Val/DiceScore', avg_dice, epoch)
        writer.add_scalar('Val/Loss', avg_loss, epoch)
    
    return avg_loss, avg_dice

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup output directories
    output_dirs = setup_output_dirs(args)
    exp_dir = output_dirs['exp_dir']
    ckpt_dir = output_dirs['ckpt_dir']
    log_dir = output_dirs['log_dir']
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\nExperiment directory: {exp_dir}")
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    print(f"Tensorboard logs will be saved to: {log_dir}")
    
    # Save configuration
    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
    
    # Log hyperparameters
    writer.add_text('Hyperparameters/Dataset', args.dataset)
    writer.add_text('Hyperparameters/PromptType', args.prompt_type)
    writer.add_text('Hyperparameters/ModelType', args.model_type)
    writer.add_text('Hyperparameters/BatchSize', str(args.batch_size))
    writer.add_text('Hyperparameters/LearningRate', str(args.lr))
    writer.add_text('Hyperparameters/Optimizer', args.opt)
    writer.add_text('Hyperparameters/Scheduler', args.scheduler)
    
    # Build datasets and dataloaders
    train_dataset, val_dataset, test_dataset = build_dataset(args)
    
    if args.mode == 'train':
        if args.dataset == 'refuge':
            train_dataset = ConcatDataset([train_dataset, val_dataset])
        elif args.dataset == 'lidc':
            train_dataset = train_dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=args.pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.pin_memory
        )
        
        # Build model
        model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, args=args)
        model = model.to(device)
        
        # Freeze parameters except adapters
        for name, param in model.named_parameters():
            if 'image_encoder.adapters' in name:
                param.requires_grad = True
            elif 'sample_reconstruct' in name:
                param.requires_grad = True
            elif 'Probabilistic_model' in name:
                param.requires_grad = True
            elif 'condition' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Setup training
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)
        best_loss = 0.7
        no_improve = 0
        
        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, args, device, epoch, writer)
            
            # Validate
            val_loss, _ = validate(model, test_loader, args, device, epoch, writer)
            
            # Log learning rate
            writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], epoch)
            
            print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                
                # Save different versions of checkpoints
                checkpoints = {
                    'best_model.pth': {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    },
                    f'checkpoint_epoch_{epoch}.pth': {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                }
                
                for filename, checkpoint in checkpoints.items():
                    save_path = os.path.join(ckpt_dir, filename)
                    torch.save(checkpoint, save_path)
                    print(f"Saved checkpoint to: {save_path}")
            
            # Early stopping
            if best_loss < 0.6 and val_loss >= best_loss:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        writer.close()
            
    else:  # Validation mode
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.pin_memory
        )
        
        # Load model
        model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, args=args)
        if args.resume:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from: {args.resume}")
        model = model.to(device)
        model.eval()
        # Run validation
        final_dice = 0
        num_runs = 10
        print("Running validation...")
        for i in range(num_runs):
            _, val_dice = validate(model, test_loader, args, device, i, None)  # No writer needed for validation
            print(f"Run {i+1}/{num_runs} - Dice Score: {val_dice:.4f}")
            final_dice += val_dice
        
        final_dice = final_dice / num_runs
        print(f"\nFinal Average Dice Score: {final_dice:.4f}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)