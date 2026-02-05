"""
Tiny Lesion Training Script
============================
Trains a model with small patches optimized for tiny lesion detection (<500 voxels).
Supports multiple patch sizes (48^3, 32^3, 24^3) for experimentation.

Usage:
    python scripts/train_tiny_lesion.py  # Default 48^3 config
    python scripts/train_tiny_lesion.py --config configs/config_tiny_lesion.yaml  # 48^3 patches
    python scripts/train_tiny_lesion.py --config configs/config_tiny_lesion_24patch.yaml  # 24^3 patches
    python scripts/train_tiny_lesion.py --evaluate-only  # Just evaluate existing model
    python scripts/train_tiny_lesion.py --resume  # Resume from checkpoint
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from segmentation.unet import LightweightUNet3D
from segmentation.dataset import BrainMetDataset, get_train_val_split
from segmentation.advanced_losses import TverskyLoss, SmallLesionOptimizedLoss, FocalLoss
from segmentation.augmentation import AugmentationPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sequences for Superset
SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> nn.Module:
    """Create model from config"""
    return LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config.get('base_channels', 24),
        depth=config.get('depth', 3),
        use_attention=config.get('use_attention', True),
        use_residual=config.get('use_residual', True),
        dropout_p=config.get('dropout', 0.15)
    )


class TinyLesionLoss(nn.Module):
    """
    Combined loss specifically optimized for tiny lesion detection.
    Heavily penalizes false negatives (missed lesions).
    """
    def __init__(self, tversky_alpha=0.2, tversky_beta=0.8, focal_gamma=3.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal = FocalLoss(alpha=0.8, gamma=focal_gamma)

    def forward(self, pred, target):
        # Weighted combination favoring recall
        tversky_loss = self.tversky(pred, target)
        focal_loss = self.focal(pred, target)
        return 0.7 * tversky_loss + 0.3 * focal_loss


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice coefficient"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.item()


def compute_lesion_volume(mask: torch.Tensor) -> int:
    """Compute lesion volume in voxels"""
    return int(mask.sum().item())


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images, masks, _ = batch
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate_with_size_stratification(model, dataloader, criterion, device, threshold=0.5):
    """
    Validate model with size-stratified metrics.
    Returns overall metrics and per-size-category metrics.
    """
    model.eval()

    # Size categories (in voxels)
    SIZE_BINS = {
        'tiny': (0, 500),
        'small': (500, 2000),
        'medium': (2000, 5000),
        'large': (5000, float('inf'))
    }

    results = {cat: {'dice_scores': [], 'count': 0} for cat in SIZE_BINS}
    all_dice_scores = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images, masks, case_ids = batch
            images = images.to(device)
            masks = masks.to(device)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

            # Compute per-sample metrics
            for i in range(images.size(0)):
                pred = outputs[i:i+1]
                target = masks[i:i+1]

                dice = compute_dice(pred, target, threshold)
                volume = compute_lesion_volume(target)

                all_dice_scores.append(dice)

                # Categorize by size
                for cat, (low, high) in SIZE_BINS.items():
                    if low <= volume < high:
                        results[cat]['dice_scores'].append(dice)
                        results[cat]['count'] += 1
                        break

    # Compute summary statistics
    summary = {
        'overall': {
            'mean_dice': np.mean(all_dice_scores) if all_dice_scores else 0,
            'median_dice': np.median(all_dice_scores) if all_dice_scores else 0,
            'loss': total_loss / num_batches if num_batches > 0 else 0,
            'count': len(all_dice_scores)
        }
    }

    for cat in SIZE_BINS:
        scores = results[cat]['dice_scores']
        summary[cat] = {
            'mean_dice': np.mean(scores) if scores else 0,
            'median_dice': np.median(scores) if scores else 0,
            'count': results[cat]['count'],
            'success_rate': (np.array(scores) > 0.5).mean() * 100 if scores else 0
        }

    return summary


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, best_dice, config):
    """Save training checkpoint"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_dice': best_dice,
        'config': config
    }, path)


def save_training_state(output_dir: Path, model_name: str, epoch: int, best_dice: float,
                        metrics: dict, completed: bool = False):
    """Save training state for monitoring progress"""
    state_path = output_dir / f"{model_name}_state.json"
    state = {
        'model_name': model_name,
        'epoch': epoch,
        'best_weighted_dice': best_dice,
        'latest_metrics': metrics,
        'completed': completed,
        'timestamp': datetime.now().isoformat()
    }
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def load_training_state(output_dir: Path, model_name: str) -> dict:
    """Load training state if it exists"""
    state_path = output_dir / f"{model_name}_state.json"
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Train tiny lesion detection model')
    parser.add_argument('--config', type=str,
                        default='configs/config_tiny_lesion.yaml',
                        help='Path to config file')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate existing model')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    args = parser.parse_args()

    # Setup
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / args.config
    config = load_config(config_path)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Paths
    train_dir = base_dir / config['data_dir']
    test_dir = base_dir / config.get('test_dir', config['data_dir'].replace('train', 'test'))
    output_dir = base_dir / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.get('model_name', 'tiny_lesion_48patch')
    checkpoint_path = output_dir / f"{model_name}_best.pth"
    resume_checkpoint_path = output_dir / f"{model_name}_resume.pth"

    # TensorBoard
    log_dir = base_dir / 'logs' / 'tensorboard' / f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=str(log_dir))

    logger.info("=" * 60)
    logger.info("TINY LESION MODEL TRAINING")
    logger.info(f"Patch size: {config['patch_size']}")
    logger.info(f"Target: lesions < {config.get('small_lesion_threshold', 500)} voxels")
    logger.info("=" * 60)

    # Create model
    model = create_model(config)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    criterion = TinyLesionLoss(
        tversky_alpha=config.get('tversky_alpha', 0.2),
        tversky_beta=config.get('tversky_beta', 0.8),
        focal_gamma=config.get('focal_gamma', 3.0)
    )

    # Dataset
    patch_size = tuple(config['patch_size'])
    target_size_cfg = config.get('target_size')
    target_size = tuple(target_size_cfg) if target_size_cfg else None  # None = no resize (preprocessed data)

    # Create full dataset
    full_dataset = BrainMetDataset(
        data_dir=str(train_dir),
        sequences=SEQUENCES,
        patch_size=patch_size,
        target_size=target_size,
        transform=AugmentationPipeline(
            augmentation_probability=config.get('augmentation_prob', 0.4)
        ) if config.get('use_augmentation', True) else None
    )

    # Train/val split
    train_indices, val_indices = get_train_val_split(
        len(full_dataset),
        val_split=config.get('val_split', 0.15)
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # DataLoaders
    num_workers = config.get('num_workers', 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    if args.evaluate_only:
        # Load and evaluate existing model
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {checkpoint_path}")

            # Evaluate on test set
            test_dataset = BrainMetDataset(
                data_dir=str(test_dir),
                sequences=SEQUENCES,
                patch_size=patch_size,
                target_size=target_size
            )
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

            logger.info("\nEvaluating on test set...")
            results = validate_with_size_stratification(model, test_loader, criterion, device)

            logger.info("\n" + "=" * 60)
            logger.info("TEST SET RESULTS (Size-Stratified)")
            logger.info("=" * 60)
            logger.info(f"Overall: {results['overall']['mean_dice']*100:.1f}% Dice")
            for cat in ['tiny', 'small', 'medium', 'large']:
                r = results[cat]
                logger.info(f"  {cat.upper():8s}: {r['mean_dice']*100:5.1f}% Dice | "
                           f"{r['success_rate']:5.1f}% success | n={r['count']}")
        else:
            logger.error(f"No checkpoint found at {checkpoint_path}")
        return

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 1e-4)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('epochs', 100)
    )

    scaler = GradScaler('cuda')

    # Training loop
    best_dice = 0
    patience_counter = 0
    patience = config.get('patience', 25)
    start_epoch = 1

    # Resume from checkpoint if requested
    if args.resume and resume_checkpoint_path.exists():
        logger.info(f"Resuming from {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0)
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, best_dice={best_dice:.4f}")

    for epoch in range(start_epoch, config.get('epochs', 100) + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)

        # Validate with size stratification
        val_results = validate_with_size_stratification(model, val_loader, criterion, device)

        scheduler.step()

        # Log metrics
        overall_dice = val_results['overall']['mean_dice']
        tiny_dice = val_results['tiny']['mean_dice']

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/overall', overall_dice, epoch)
        writer.add_scalar('Dice/tiny', tiny_dice, epoch)
        writer.add_scalar('Dice/small', val_results['small']['mean_dice'], epoch)
        writer.add_scalar('Dice/medium', val_results['medium']['mean_dice'], epoch)
        writer.add_scalar('Dice/large', val_results['large']['mean_dice'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        logger.info(f"Epoch {epoch}: Loss={train_loss:.4f} | "
                   f"Dice: Overall={overall_dice*100:.1f}% Tiny={tiny_dice*100:.1f}%")

        # Save best model (prioritize tiny lesion performance)
        # Use weighted score: 60% tiny + 40% overall
        weighted_score = 0.6 * tiny_dice + 0.4 * overall_dice

        if weighted_score > best_dice:
            best_dice = weighted_score
            patience_counter = 0
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler,
                epoch, best_dice, config
            )
            logger.info(f"  -> New best model saved! (weighted={weighted_score*100:.1f}%)")
        else:
            patience_counter += 1

        # Always save resume checkpoint and training state (for monitoring)
        save_checkpoint(
            resume_checkpoint_path, model, optimizer, scheduler, scaler,
            epoch, best_dice, config
        )
        save_training_state(
            output_dir, model_name, epoch, best_dice,
            {
                'train_loss': train_loss,
                'overall_dice': overall_dice,
                'tiny_dice': tiny_dice,
                'small_dice': val_results['small']['mean_dice'],
                'medium_dice': val_results['medium']['mean_dice'],
                'large_dice': val_results['large']['mean_dice'],
            }
        )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    writer.close()

    # Mark training as complete
    save_training_state(output_dir, model_name, epoch, best_dice, {}, completed=True)

    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    # Load best model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test dataset
    test_dataset = BrainMetDataset(
        data_dir=str(test_dir),
        sequences=SEQUENCES,
        patch_size=patch_size,
        target_size=target_size
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    test_results = validate_with_size_stratification(model, test_loader, criterion, device)

    logger.info(f"Overall: {test_results['overall']['mean_dice']*100:.1f}% Dice")
    for cat in ['tiny', 'small', 'medium', 'large']:
        r = test_results[cat]
        logger.info(f"  {cat.upper():8s}: {r['mean_dice']*100:5.1f}% Dice | "
                   f"{r['success_rate']:5.1f}% success | n={r['count']}")

    # Save results
    results_path = output_dir / f"{model_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'test_results': test_results,
            'best_weighted_score': best_dice
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
