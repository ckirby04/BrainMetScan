"""
Conservative Multi-scale Training with Size-Stratified Performance Monitoring

Trains a model optimized for both small and large lesions, with detailed
tracking of performance across different lesion size categories.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import nibabel as nib
import yaml
import argparse

sys.path.append('src/segmentation')
from unet import LightweightUNet3D
from dataset import BrainMetDataset, get_train_val_split

# Simple combined loss for small lesion focus
class BalancedSmallLesionLoss(nn.Module):
    """Balanced loss that helps small lesions without ignoring large ones"""
    def __init__(self, small_threshold=500, small_weight=3.0):
        super().__init__()
        self.small_threshold = small_threshold
        self.small_weight = small_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def dice_loss(self, pred, target, smooth=1.0):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, pred, target):
        # Calculate base losses
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target).mean()

        # Check if this is a small lesion
        lesion_size = target.sum().item()

        if lesion_size < self.small_threshold and lesion_size > 0:
            # Apply weighting for small lesions
            weight = self.small_weight
        else:
            weight = 1.0

        # Combined weighted loss
        total_loss = weight * (0.5 * dice + 0.5 * bce)
        return total_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for improved recall on small lesions

    Alpha controls weight of false negatives (missed lesions)
    Beta controls weight of false positives (over-prediction)

    For better tiny lesion detection: alpha < beta (favors recall)
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # Weight for false negatives
        self.beta = beta    # Weight for false positives
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)

        # True Positives, False Positives, False Negatives
        TP = (pred_sigmoid * target).sum()
        FP = (pred_sigmoid * (1 - target)).sum()
        FN = ((1 - pred_sigmoid) * target).sum()

        # Tversky index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        # Return Tversky loss (1 - Tversky index)
        return 1 - tversky_index


def get_lesion_size(case_dir):
    """Get the lesion size for a case"""
    seg_path = case_dir / 'seg.nii.gz'
    if not seg_path.exists():
        return 0
    mask = nib.load(str(seg_path)).get_fdata()
    return int(np.sum(mask > 0))


def categorize_by_size(dataset, indices):
    """Categorize cases by lesion size"""
    categories = {
        'tiny': [],      # <500 voxels
        'medium': [],    # 500-5000 voxels
        'large': []      # >5000 voxels
    }

    for idx in indices:
        case_dir = dataset.cases[idx]  # Already a Path object
        size = get_lesion_size(case_dir)

        if size < 500:
            categories['tiny'].append(idx)
        elif size < 5000:
            categories['medium'].append(idx)
        else:
            categories['large'].append(idx)

    return categories


@torch.no_grad()
def evaluate_by_size(model, dataset, indices_by_size, device, patch_size=(96, 96, 96)):
    """Evaluate model performance separately for each size category"""
    model.eval()

    results = {}

    for size_cat, indices in indices_by_size.items():
        if len(indices) == 0:
            continue

        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)

        dice_scores = []

        for batch in loader:
            images, mask, case_ids = batch
            images = images.to(device)
            mask = mask.to(device)

            # Simple forward pass (not sliding window for speed)
            pred = model(images)
            pred_sigmoid = torch.sigmoid(pred)

            # Calculate Dice
            pred_binary = (pred_sigmoid > 0.5).float()
            intersection = (pred_binary * mask).sum()
            dice = (2. * intersection) / (pred_binary.sum() + mask.sum() + 1e-8)

            dice_scores.append(dice.item())

        # Calculate statistics
        dice_scores = np.array(dice_scores)
        results[size_cat] = {
            'n_cases': len(indices),
            'mean_dice': float(np.mean(dice_scores)),
            'median_dice': float(np.median(dice_scores)),
            'success_rate': float(np.mean(dice_scores > 0.5) * 100)
        }

    return results


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    dice_scores = []

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images, mask, case_ids = batch
        images = images.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(images)
                loss = criterion(pred, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(images)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()

        # Calculate Dice for monitoring
        with torch.no_grad():
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            intersection = (pred_binary * mask).sum()
            dice = (2. * intersection) / (pred_binary.sum() + mask.sum() + 1e-8)
            dice_scores.append(dice.item())

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})

    return total_loss / len(train_loader), np.mean(dice_scores)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    dice_scores = []

    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        images, mask, case_ids = batch
        images = images.to(device)
        mask = mask.to(device)

        pred = model(images)
        loss = criterion(pred, mask)

        # Calculate Dice
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > 0.5).float()
        intersection = (pred_binary * mask).sum()
        dice = (2. * intersection) / (pred_binary.sum() + mask.sum() + 1e-8)

        total_loss += loss.item()
        dice_scores.append(dice.item())
        pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})

    return total_loss / len(val_loader), np.mean(dice_scores)


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

    # Restore scaler if present (for mixed precision training)
    if 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Get starting epoch and best metrics
    start_epoch = checkpoint.get('epoch', 0)
    best_val_dice = checkpoint.get('best_val_dice', checkpoint.get('val_dice', 0))

    # Load training history if present
    history = checkpoint.get('history', {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_by_size': [],
        'lr': []
    })

    print(f"Resumed from epoch {start_epoch}")
    print(f"Best validation Dice so far: {best_val_dice:.4f}")

    return start_epoch, best_val_dice, history


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train brain metastasis segmentation model')
    parser.add_argument('--config', type=str, default='config_small_lesion_focus.yaml',
                        help='Path to YAML config file (default: config_small_lesion_focus.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., models_multiscale/best_model.pth)')
    args = parser.parse_args()

    # Load configuration from YAML file
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        # Default configuration (fallback)
        config = {
            'data_dir': 'data/train',
            'output_dir': 'models_multiscale',
            'base_channels': 16,
            'depth': 3,
            'dropout': 0.1,
            'use_residual': True,
            'patch_size': [96, 96, 96],
            'batch_size': 2,
            'epochs': 75,
            'lr': 0.001,
            'small_lesion_threshold': 500,
            'small_lesion_weight': 3.0,
            'val_split': 0.15,
            'num_workers': 4,
            'save_freq': 10
        }

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONSERVATIVE MULTISCALE TRAINING WITH SIZE MONITORING")
    print("=" * 80)
    print(f"\nDevice: {device}")
    print(f"Output directory: {output_dir}")
    print(f"\nSmall lesion threshold: {config['small_lesion_threshold']} voxels")
    print(f"Small lesion weight: {config['small_lesion_weight']}x")
    print(f"Patch size: {config['patch_size']}")
    print("=" * 80)

    # Create dataset
    print("\nLoading dataset...")
    full_dataset = BrainMetDataset(
        data_dir=config['data_dir'],
        patch_size=tuple(config['patch_size'])
    )

    # Train/val split
    train_indices, val_indices = get_train_val_split(
        len(full_dataset),
        val_split=config['val_split'],
        seed=42
    )

    print(f"Total cases: {len(full_dataset)}")
    print(f"Training cases: {len(train_indices)}")
    print(f"Validation cases: {len(val_indices)}")

    # Categorize by size
    print("\nCategorizing cases by lesion size...")
    train_by_size = categorize_by_size(full_dataset, train_indices)
    val_by_size = categorize_by_size(full_dataset, val_indices)

    print("\nTraining set distribution:")
    for size_cat, indices in train_by_size.items():
        print(f"  {size_cat.capitalize()}: {len(indices)} cases")

    print("\nValidation set distribution:")
    for size_cat, indices in val_by_size.items():
        print(f"  {size_cat.capitalize()}: {len(indices)} cases")

    # Create data loaders
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Create model
    print("\nCreating model...")
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config['base_channels'],
        depth=config['depth'],
        dropout_p=config['dropout'],
        use_residual=config['use_residual'],
        use_attention=False
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer - choose based on config
    loss_type = config.get('loss_type', 'small_lesion')

    if loss_type == 'tversky':
        criterion = TverskyLoss(
            alpha=config.get('tversky_alpha', 0.3),
            beta=config.get('tversky_beta', 0.7)
        )
        print(f"Using Tversky Loss (alpha={config.get('tversky_alpha', 0.3)}, beta={config.get('tversky_beta', 0.7)})")
    else:
        criterion = BalancedSmallLesionLoss(
            small_threshold=config['small_lesion_threshold'],
            small_weight=config['small_lesion_weight']
        )
        print(f"Using Balanced Small Lesion Loss (threshold={config['small_lesion_threshold']}, weight={config['small_lesion_weight']})")
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_dice = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_by_size': [],  # Track performance by size category
        'lr': []
    }

    if args.resume:
        start_epoch, best_val_dice, history = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        best_epoch = start_epoch

    # Training loop
    print("\n" + "=" * 80)
    if args.resume:
        print(f"RESUMING TRAINING FROM EPOCH {start_epoch + 1}")
    else:
        print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 80)

        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Evaluate by size category every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\nEvaluating by lesion size...")
            val_size_metrics = evaluate_by_size(
                model, full_dataset, val_by_size, device,
                patch_size=tuple(config['patch_size'])
            )
            history['val_by_size'].append({
                'epoch': epoch + 1,
                'metrics': val_size_metrics
            })

            print("\nValidation Performance by Size:")
            print("-" * 80)
            for size_cat, metrics in val_size_metrics.items():
                print(f"{size_cat.capitalize()}:")
                print(f"  Cases: {metrics['n_cases']}")
                print(f"  Mean Dice: {metrics['mean_dice']:.4f}")
                print(f"  Success Rate: {metrics['success_rate']:.1f}%")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'val_dice': val_dice,
                'val_loss': val_loss,
                'best_val_dice': best_val_dice,
                'history': history,
                'config': config
            }, output_dir / 'best_model.pth')

            print(f"  ✓ New best model saved! (Dice: {val_dice:.4f})")

        # Save checkpoint periodically
        if (epoch + 1) % config['save_freq'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'val_dice': val_dice,
                'val_loss': val_loss,
                'best_val_dice': best_val_dice,
                'history': history,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pth')
            print(f"  ✓ Checkpoint saved at epoch {epoch + 1}")

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nBest model from epoch {best_epoch} (Val Dice: {best_val_dice:.4f})")

    # Final size-stratified evaluation
    print("\nFinal validation performance by lesion size:")
    final_metrics = evaluate_by_size(
        model, full_dataset, val_by_size, device,
        patch_size=tuple(config['patch_size'])
    )

    print("\n" + "=" * 80)
    for size_cat, metrics in final_metrics.items():
        print(f"\n{size_cat.upper()} LESIONS:")
        print(f"  Cases: {metrics['n_cases']}")
        print(f"  Mean Dice: {metrics['mean_dice']:.4f}")
        print(f"  Median Dice: {metrics['median_dice']:.4f}")
        print(f"  Success Rate: {metrics['success_rate']:.1f}%")

    # Save final metrics
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_val_dice': best_val_dice,
            'final_metrics_by_size': final_metrics,
            'config': config
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")
    print(f"Training history saved to: {output_dir / 'history.json'}")
    print(f"Final metrics saved to: {output_dir / 'final_metrics.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
