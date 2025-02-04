import argparse
import os
import datetime

def get_args_parser():
    parser = argparse.ArgumentParser('UA-SAM')
    
    # Runtime parameters
    parser.add_argument('--mode', default='train', type=str,
                        choices=['train', 'val'],
                        help='running mode: train or validation')
    
    # Directory parameters
    parser.add_argument('--output_root', default='./runs', type=str,
                        help='root output directory')
    parser.add_argument('--experiment_name', default='experiment', type=str,
                        help='experiment name')
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str,
                        help='directory name for checkpoints')
    parser.add_argument('--log_dir', default='logs', type=str,
                        help='directory name for tensorboard logs')
    parser.add_argument('--dataset_path', default='/local/scratch/v_jiaying_zhou/uncertainty_adapter/segment-anything-main/data/data_lidc-001.pickle', type=str,
                        help='dataset path')
    parser.add_argument('--checkpoint', default='./pretrain/sam_vit_b_01ec64.pth', type=str,
                        help='path to SAM checkpoint')
    parser.add_argument('--resume', default='./your_checkpoint', type=str,
                    help='path to checkpoint for validation')
    parser.add_argument('--split_ratio', default=(0.8, 0.2), type=tuple,
                        help='train-validation split ratio')
    
    # Model parameters
    parser.add_argument('--model_type', default='vit_b', type=str,
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type')
    parser.add_argument('--latent_dim', default=6, type=int,
                        help='latent dimension')
    
    # Dataset parameters
    parser.add_argument('--dataset', default='refuge', type=str,
                        choices=['refuge', 'lidc'],
                        help='dataset type')
    parser.add_argument('--img_size', default=512, type=int,
                        help='input image size')
    
    # Training parameters
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--num_workers', default=32, type=int,
                        help='number of workers for dataloader')
    parser.add_argument('--pin_memory', default=True, type=bool,
                        help='pin memory for faster GPU training')
    
    # Optimization parameters
    parser.add_argument('--opt', default='adam', type=str,
                        choices=['adam', 'adamw'])
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--scheduler', default='step', type=str,
                        choices=['step', 'cosine'])
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)
    
    # Loss parameters
    parser.add_argument('--beta', default=1e-0, type=float,
                        help='beta parameter for ELBO loss')
    parser.add_argument('--reg_weight', default=1e-5, type=float,
                        help='weight for L2 regularization')
    
    # Early stopping
    parser.add_argument('--patience', default=15, type=int,
                        help='patience epochs for early stopping')
    
    # Prompt parameters
    parser.add_argument('--prompt_type', default='point', type=str,
                        choices=['point', 'box', 'both'],
                        help='type of prompt to use')
    parser.add_argument('--num_points', default=1, type=int,
                        help='number of point prompts')
    parser.add_argument('--point_label', default=1, type=int,
                        help='label for point prompts')
    parser.add_argument('--num_boxes', default=1, type=int,
                        help='number of box prompts')
    parser.add_argument('--box_noise_std', default=0.1, type=float,
                        help='standard deviation of box noise')
    parser.add_argument('--box_noise_max', default=5, type=int,
                        help='maximum box noise in pixels')
    
    return parser

def setup_output_dirs(args):
    """Setup output directory structure"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.experiment_name}_{timestamp}"
    exp_dir = os.path.join(args.output_root, exp_name)
    
    # Create subdirectories
    ckpt_dir = os.path.join(exp_dir, args.checkpoint_dir)
    log_dir = os.path.join(exp_dir, args.log_dir)
    
    # Create directories if they don't exist
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # For validation mode, find the latest valid checkpoint
    if args.mode == 'val':
        if not args.resume:
            experiments = sorted([d for d in os.listdir(args.output_root) 
                                if os.path.isdir(os.path.join(args.output_root, d))
                                and d.startswith('experiment_')])
            
            valid_checkpoint = None
            for exp in reversed(experiments):
                checkpoint_path = os.path.join(args.output_root, exp, 
                                             args.checkpoint_dir, 'best_model.pth')
                if os.path.exists(checkpoint_path):
                    valid_checkpoint = checkpoint_path
                    args.resume = valid_checkpoint
                    print(f"Using checkpoint from experiment {exp}: {args.resume}")
                    break
            
            if valid_checkpoint is None:
                raise FileNotFoundError("No valid checkpoint found in any experiment directory")
        elif not os.path.exists(args.resume):
            raise FileNotFoundError(f"Specified checkpoint does not exist: {args.resume}")
    
    return {
        'exp_dir': exp_dir,
        'ckpt_dir': ckpt_dir,
        'log_dir': log_dir
    }

def get_default_args():
    return get_args_parser().parse_args([])