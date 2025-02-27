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
from utils import generate_click_prompt, random_box, l2_regularisation, elbo, iou, generalized_energy_distance
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
    Prepare prompts based on configuration.
    The function uses the provided labels (e.g. the random mask)
    to generate point prompts and box prompts.
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
        # For our modified refuge dataset, the key is now 'random_mask_label'
        if args.dataset == 'refuge':
            labels = batch['random_mask_label'].to(device)
        elif args.dataset == 'lidc':
            labels = batch['random_label'].to(device)

        batched_input = prepare_prompts(images, labels, args, device)

        outputs_tuple = model(
            batched_input=batched_input,
            multimask_output=False
        )
        outputs_list = outputs_tuple[0]
        masks = outputs_list[0]['masks'].float()

        reg_loss = (l2_regularisation(model.Probabilistic_model.prior) +
                    l2_regularisation(model.Probabilistic_model.posterior))
        loss = elbo(masks, labels, outputs_tuple[1], beta=args.beta) + args.reg_weight * reg_loss
        loss = loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reconstruction_loss = elbo(masks, labels, outputs_tuple[1], beta=args.beta)
        total_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_rec_loss += reconstruction_loss.item()

        if batch_idx % 10 == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), step)
            writer.add_scalar('Train/RegLoss', reg_loss.item(), step)
            writer.add_scalar('Train/ReconstructionLoss', reconstruction_loss.item(), step)

        train_step += 1
        print(f"\rStep: {train_step}/{len(train_loader)}, Loss: {loss.item():.4f}", end='')

    avg_loss = total_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)
    avg_rec_loss = total_rec_loss / len(train_loader)
    scheduler.step()

    writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
    writer.add_scalar('Train/EpochRegLoss', avg_reg_loss, epoch)
    writer.add_scalar('Train/EpochReconLoss', avg_rec_loss, epoch)

    return avg_loss


@torch.no_grad()
def validate(model, val_loader, args, device, epoch, writer):
    model.eval()
    total_dice = 0
    total_loss = 0
    total_iou = 0
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=False)

    for batch in val_loader:
        images = batch['image'].to(device)
        # For validation, use the majority vote label.
        if args.mode == 'train':
            if args.dataset == 'refuge':
                labels = batch['random_mask_label'].to(device)
            elif args.dataset == 'lidc':
                labels = batch['random_label'].to(device)
        elif args.mode == 'val':
            if args.dataset == 'refuge':
                labels = batch['majorityvote_label'].to(device)
            elif args.dataset == 'lidc':
                labels = batch['majorityvote_label'].to(device)

        batched_input = prepare_prompts(images, labels, args, device)
        outputs_tuple = model(batched_input=batched_input, multimask_output=False)
        outputs_list = outputs_tuple[0]
        masks = outputs_list[0]['masks'].float()

        loss = elbo(masks, labels, outputs_tuple[1], beta=args.beta)
        total_loss += loss.item()

        masks = torch.sigmoid(masks)
        binary_masks = (masks > 0.5).float()

        dice_score = dice_metric(binary_masks, labels).to(device)
        total_dice += dice_score[0].item()

        intersection = torch.logical_and(binary_masks.bool(), labels.bool())
        union = torch.logical_or(binary_masks.bool(), labels.bool())
        intersection_sum = intersection.sum(dim=(1, 2, 3))
        union_sum = union.sum(dim=(1, 2, 3))
        iou_per_image = (intersection_sum + 1e-8) / (union_sum + 1e-8)
        mean_iou_batch = iou_per_image.mean()
        total_iou += mean_iou_batch.item()

        if writer is not None and epoch % 5 == 0:
            writer.add_images('Val/Images', images[:4], epoch)
            writer.add_images('Val/TrueMasks', labels[:4], epoch)
            writer.add_images('Val/PredMasks', binary_masks[:4], epoch)

    avg_dice = total_dice / len(val_loader)
    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    if writer is not None:
        writer.add_scalar('Val/DiceScore', avg_dice, epoch)
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/IoU', avg_iou, epoch)

    return avg_loss, avg_dice, avg_iou


@torch.no_grad()
def validate_ged(model, val_loader, args, device, epoch, writer, num_samples=10):
    model.eval()
    total_ged = 0

    for batch_idx, batch in enumerate(val_loader):
        images = batch['image'].to(device)
        all_masks = batch['all_masks'].to(device)

        batch_predictions_list = []
        for _ in range(num_samples):
            if args.dataset == 'refuge':
                labels = batch['majorityvote_label'].to(device)
            elif args.dataset == 'lidc':
                labels = batch['majorityvote_label'].to(device)

            batched_input = prepare_prompts(images, labels, args, device)
            outputs_tuple = model(batched_input=batched_input, multimask_output=False)
            masks = torch.sigmoid(outputs_tuple[0][0]['masks'])
            batch_predictions_list.append(masks)  # shape: (B,1,H,W)

        preds_stacked = torch.stack(batch_predictions_list, dim=0)
        preds_stacked = preds_stacked.permute(1, 0, 2, 3, 4).squeeze(2)
        batch_ged = generalized_energy_distance(
            labels=all_masks,
            preds=preds_stacked,
            thresh=0.5,
            num_classes=2
        )
        total_ged += batch_ged
    avg_ged = total_ged / len(val_loader)

    if writer is not None:
        writer.add_scalar('Val/GED_Samples', avg_ged, epoch)

    return avg_ged


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dirs = setup_output_dirs(args)
    exp_dir = output_dirs['exp_dir']
    ckpt_dir = output_dirs['ckpt_dir']
    log_dir = output_dirs['log_dir']

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
        elif args.dataset == 'lidc':
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                drop_last=True,
                pin_memory=args.pin_memory
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                drop_last=True,
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

        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)
        best_loss = 0.7
        no_improve = 0

        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, args, device, epoch, writer)
            val_loss, _, _ = validate(model, test_loader, args, device, epoch, writer)

            writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], epoch)

            print(f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
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

            if best_loss < 0.6 and val_loss >= best_loss:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        writer.close()

    else:
        if args.dataset == 'refuge':
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=args.pin_memory
            )
        elif args.dataset == 'lidc':
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                drop_last=True,
                pin_memory=args.pin_memory
            )

        model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, args=args)
        if args.resume:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from: {args.resume}")
        model = model.to(device)
        model.eval()
        final_dice = 0
        final_iou = 0
        final_ged = 0
        num_runs = 10
        print("Running validation...")
        for i in range(num_runs):
            _, val_dice, val_iou = validate(model, test_loader, args, device, i, writer)
            val_ged = validate_ged(model, test_loader, args, device, i, writer, num_samples=16)
            print(f"Run {i + 1}/{num_runs}")
            print(f"  - Dice Score: {val_dice:.4f}")
            print(f"  - IoU Score: {val_iou:.4f}")
            print(f"  - GED Score: {val_ged:.4f}")
            final_dice += val_dice
            final_iou += val_iou
            final_ged += val_ged

        final_dice = final_dice / num_runs
        final_iou = final_iou / num_runs
        final_ged = final_ged / num_runs

        print("\nFinal Results:")
        print(f"Average Dice Score: {final_dice:.4f}")
        print(f"Average IoU Score: {final_iou:.4f}")
        print(f"Average GED Score: {final_ged:.4f}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
