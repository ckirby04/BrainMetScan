"""
Training script for brain metastasis segmentation U-Net
Optimized for consumer GPUs with mixed precision training
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

from dataset import BrainMetDataset, get_train_val_split
from unet import LightweightUNet3D, CombinedLoss, EnhancedCombinedLoss, count_parameters


def dice_coefficient(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-8)

    return dice.item()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None, gradient_clip=None):
    """Train for one epoch

    Args:
        scheduler: Optional scheduler for per-batch stepping (e.g., OneCycleLR)
        gradient_clip: Optional gradient clipping max norm
    """
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        # Step scheduler per batch if provided (e.g., OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        # Metrics
        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(loader, desc="Validation")
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_dice


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load checkpoint and restore all training state"""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore scheduler if present
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore scaler if present
    if 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Get starting epoch and best metrics
    start_epoch = checkpoint.get('epoch', 0)
    best_dice = checkpoint.get('best_dice', checkpoint.get('val_dice', 0))

    # Load training history if present
    history = checkpoint.get('history', {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    })

    print(f"Resumed from epoch {start_epoch}")
    print(f"Best validation Dice so far: {best_dice:.4f}")

    return start_epoch, best_dice, history


def train(args):
    """Main training function"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")

    # Train/val split
    train_cases, val_cases = get_train_val_split(
        args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Create separate train and val datasets with different augmentation settings
    train_dataset = BrainMetDataset(
        data_dir=args.data_dir,
        sequences=['t1_pre', 't1_gd', 'flair', 'bravo'],
        patch_size=tuple(args.patch_size),
        metadata_path=args.metadata_path,
        augment=args.use_augmentation,  # Enable augmentation for training
        augmentation_prob=args.augmentation_prob
    )

    val_dataset = BrainMetDataset(
        data_dir=args.data_dir,
        sequences=['t1_pre', 't1_gd', 'flair', 'bravo'],
        patch_size=tuple(args.patch_size),
        metadata_path=args.metadata_path,
        augment=False  # No augmentation for validation
    )

    # Get indices for train/val
    case_to_idx = {case.name: idx for idx, case in enumerate(train_dataset.cases)}
    train_indices = [case_to_idx[case.name] for case in train_cases]
    val_indices = [case_to_idx[case.name] for case in val_cases]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=args.base_channels,
        depth=args.depth,
        dropout_p=args.dropout,
        use_attention=args.use_attention,
        use_residual=args.use_residual
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Attention gates: {args.use_attention}")
    print(f"Residual connections: {args.use_residual}")

    # Loss and optimizer
    if args.loss_type == 'enhanced_combined':
        criterion = EnhancedCombinedLoss(
            dice_weight=args.dice_weight,
            focal_tversky_weight=args.focal_tversky_weight,
            bce_weight=args.bce_weight
        )
        print(f"Using EnhancedCombinedLoss (Dice={args.dice_weight}, FocalTversky={args.focal_tversky_weight}, BCE={args.bce_weight})")
    else:
        criterion = CombinedLoss(
            dice_weight=args.dice_weight,
            bce_weight=args.bce_weight
        )
        print(f"Using CombinedLoss (Dice={args.dice_weight}, BCE={args.bce_weight})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    if args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=25.0,    # initial_lr = max_lr/25
            final_div_factor=1e4  # min_lr = max_lr/1e4
        )
        step_per_batch = True
        print(f"Using OneCycleLR scheduler (max_lr={args.lr}, warmup=10%)")
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        step_per_batch = False
        print(f"Using CosineAnnealingLR scheduler")
    else:  # reduce_plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        step_per_batch = False
        print(f"Using ReduceLROnPlateau scheduler")

    # Mixed precision scaler
    scaler = GradScaler()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }

    if args.resume:
        start_epoch, best_dice, history = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )

    # Training loop
    if args.resume:
        print(f"\nResuming training from epoch {start_epoch + 1} to {args.epochs}...\n")
    else:
        print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Train (with per-batch scheduler if OneCycleLR)
        if step_per_batch:
            train_loss, train_dice = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                scheduler=scheduler,
                gradient_clip=args.gradient_clip
            )
        else:
            train_loss, train_dice = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                scheduler=None,
                gradient_clip=args.gradient_clip
            )

        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler per epoch if not OneCycleLR
        if not step_per_batch:
            if args.scheduler == 'reduce_plateau':
                scheduler.step(val_dice)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print(f"LR: {current_lr:.6f}\n")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state_dict': scaler.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
                'best_dice': best_dice,
                'history': history,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"[SAVED] Best model (Dice: {val_dice:.4f})\n")

        # Save checkpoint every N epochs
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state_dict': scaler.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
                'best_dice': best_dice,
                'history': history,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"[SAVED] Checkpoint at epoch {epoch}\n")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brain metastasis segmentation model")

    # Data
    parser.add_argument('--data_dir', type=str, default='../../train',
                        help='Path to training data directory')
    parser.add_argument('--metadata_path', type=str, default='../../metadata.csv',
                        help='Path to metadata CSV')
    parser.add_argument('--output_dir', type=str, default='../../../models',
                        help='Output directory for models')

    # Model
    parser.add_argument('--base_channels', type=int, default=16,
                        help='Base number of channels in U-Net')
    parser.add_argument('--depth', type=int, default=3,
                        help='Depth of U-Net')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--use_attention', action='store_true',
                        help='Use attention gates in decoder')
    parser.add_argument('--use_residual', action='store_true',
                        help='Use residual connections in conv blocks')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (reduce if OOM)')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[96, 96, 96],
                        help='3D patch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (max_lr for OneCycleLR)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--scheduler', type=str, default='reduce_plateau',
                        choices=['onecycle', 'cosine', 'reduce_plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--gradient_clip', type=float, default=None,
                        help='Gradient clipping max norm (None = no clipping)')

    # Augmentation
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--augmentation_prob', type=float, default=0.3,
                        help='Probability for each augmentation')

    # Loss
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['combined', 'enhanced_combined'],
                        help='Loss function type')
    parser.add_argument('--dice_weight', type=float, default=0.7,
                        help='Weight for Dice loss')
    parser.add_argument('--bce_weight', type=float, default=0.3,
                        help='Weight for BCE loss')
    parser.add_argument('--focal_tversky_weight', type=float, default=0.4,
                        help='Weight for Focal Tversky loss (enhanced_combined only)')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., ../../../models/best_model.pth)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
