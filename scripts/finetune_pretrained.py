"""
Fine-tune Pretrained Model and Compare to Original
===================================================
Fine-tunes the pretrained exp1_8patch model on labeled data and
compares performance to the original model trained from scratch.

Usage:
    python scripts/finetune_pretrained.py
    python scripts/finetune_pretrained.py --epochs 50
    python scripts/finetune_pretrained.py --resume  # Resume if interrupted
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from segmentation.unet import LightweightUNet3D
from segmentation.dataset import BrainMetDataset
from segmentation.advanced_losses import TverskyLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2 * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)


def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate model on validation set."""
    model.eval()
    total_dice = 0
    n_batches = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images, masks = images.to(device), masks.to(device)

            with autocast('cuda'):
                outputs = model(images)

            dice = compute_dice(outputs, masks, threshold)
            total_dice += dice.item()
            n_batches += 1

    return total_dice / n_batches


def detect_architecture(checkpoint):
    """Auto-detect model architecture from checkpoint weights."""
    state = checkpoint['model_state_dict']

    # Detect base_channels from first conv layer
    first_conv = state['inc.conv1.weight']
    base_channels = first_conv.shape[0]

    # Detect if attention is used
    use_attention = any('attention' in k for k in state.keys())

    # Detect if residual connections are used
    use_residual = any('residual' in k for k in state.keys())

    return {
        'base_channels': base_channels,
        'use_attention': use_attention,
        'use_residual': use_residual
    }


def load_model(model_path, device):
    """Load a model from checkpoint with auto-detected architecture."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Auto-detect architecture
    arch = detect_architecture(checkpoint)
    logger.info(f"  Detected architecture: base_channels={arch['base_channels']}, "
                f"attention={arch['use_attention']}, residual={arch['use_residual']}")

    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=arch['base_channels'],
        use_attention=arch['use_attention'],
        use_residual=arch['use_residual']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint, arch


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, masks, _ in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dice = compute_dice(outputs, masks).item()
        total_loss += loss.item()
        total_dice += dice
        n_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})

    return total_loss / n_batches, total_dice / n_batches


def main():
    parser = argparse.ArgumentParser(description='Fine-tune pretrained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')

    args = parser.parse_args()

    # Paths
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / 'model'
    superset_dir = project_dir.parent / 'Superset'
    train_dir = superset_dir / 'full' / 'train'

    pretrained_path = model_dir / 'exp1_8patch_pretrained.pth'
    original_path = model_dir / 'exp1_8patch_best.pth'
    output_path = model_dir / 'exp1_8patch_pretrained_finetuned.pth'
    state_path = model_dir / 'finetune_state.pth'

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Verify pretrained model exists
    if not pretrained_path.exists():
        logger.error(f"Pretrained model not found: {pretrained_path}")
        return

    # ==========================================================================
    # STEP 1: Evaluate original model (baseline)
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Evaluate Original Model (Baseline)")
    logger.info("=" * 60)

    # Load dataset
    patch_size = 8
    try:
        dataset = BrainMetDataset(
            data_dir=str(train_dir),
            sequences=['t1_pre', 't1_gd', 'flair', 't2'],
            patch_size=(patch_size, patch_size, patch_size),
            target_size=(128, 128, 128)
        )
    except:
        dataset = BrainMetDataset(
            data_dir=str(train_dir),
            sequences=['t1_pre', 't1_gd', 'flair', 'bravo'],
            patch_size=(patch_size, patch_size, patch_size),
            target_size=(128, 128, 128)
        )

    # Split
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    logger.info(f"Dataset: {n_train} train, {n_val} val")

    # Evaluate original model
    original_dice = None
    original_arch = None
    if original_path.exists():
        logger.info(f"Loading original model: {original_path}")
        original_model, orig_ckpt, original_arch = load_model(original_path, device)
        original_dice = evaluate_model(original_model, val_loader, device)
        logger.info(f"Original model validation Dice: {original_dice:.4f}")
        del original_model
        torch.cuda.empty_cache()
    else:
        logger.warning("Original model not found, skipping baseline evaluation")

    # ==========================================================================
    # STEP 2: Fine-tune pretrained model
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Fine-tune Pretrained Model")
    logger.info("=" * 60)

    # Load pretrained model
    logger.info(f"Loading pretrained model: {pretrained_path}")
    model, pretrain_ckpt, pretrain_arch = load_model(pretrained_path, device)
    logger.info(f"Pretrained at epoch {pretrain_ckpt.get('epoch', '?')}, loss {pretrain_ckpt.get('loss', '?'):.6f}")

    # Warn if architectures differ
    if original_arch and pretrain_arch:
        if original_arch['base_channels'] != pretrain_arch['base_channels']:
            logger.warning("=" * 60)
            logger.warning("ARCHITECTURE MISMATCH DETECTED!")
            logger.warning(f"  Original model: base_channels={original_arch['base_channels']}")
            logger.warning(f"  Pretrained model: base_channels={pretrain_arch['base_channels']}")
            logger.warning("  Comparison may not be apples-to-apples!")
            logger.warning("=" * 60)

    # Evaluate pretrained model BEFORE fine-tuning
    pretrain_dice_before = evaluate_model(model, val_loader, device)
    logger.info(f"Pretrained model Dice (before fine-tuning): {pretrain_dice_before:.4f}")

    # Setup training
    criterion = TverskyLoss(alpha=0.3, beta=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    # Resume if requested
    start_epoch = 1
    best_dice = 0
    history = {'train_loss': [], 'train_dice': [], 'val_dice': [], 'lr': []}

    if args.resume and state_path.exists():
        logger.info(f"Resuming from {state_path}")
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        start_epoch = state['epoch'] + 1
        best_dice = state['best_dice']
        history = state.get('history', history)
        logger.info(f"Resumed from epoch {start_epoch-1}, best_dice {best_dice:.4f}")

    # Training loop
    logger.info(f"\nStarting fine-tuning from epoch {start_epoch} to {args.epochs}")
    logger.info("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_dice = evaluate_model(model, val_loader, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)

        # Log progress
        improvement = ""
        if original_dice:
            diff = val_dice - original_dice
            improvement = f" | vs Original: {diff:+.4f}"

        logger.info(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, "
                   f"Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}{improvement}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': best_dice,
                'original_dice': original_dice,
                'improvement': best_dice - original_dice if original_dice else None
            }, output_path)
            logger.info(f"  *** New best model saved (Dice={best_dice:.4f}) ***")

        # Save state for resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_dice': best_dice,
            'history': history
        }, state_path)

        # Cleanup
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================================================
    # STEP 3: Final Comparison
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Show architecture info
    if original_arch and pretrain_arch:
        print(f"\nArchitecture comparison:")
        print(f"  Original:   base_channels={original_arch['base_channels']}")
        print(f"  Pretrained: base_channels={pretrain_arch['base_channels']}")
        if original_arch['base_channels'] != pretrain_arch['base_channels']:
            print("  ⚠ Different architectures - comparison is informative but not exact")
        print()

    print(f"{'Model':<45} {'Val Dice':>10}")
    print("-" * 70)

    if original_dice:
        arch_note = f" (base={original_arch['base_channels']})" if original_arch else ""
        print(f"{'Original (trained from scratch)' + arch_note:<45} {original_dice:>10.4f}")

    arch_note = f" (base={pretrain_arch['base_channels']})" if pretrain_arch else ""
    print(f"{'Pretrained (before fine-tuning)' + arch_note:<45} {pretrain_dice_before:>10.4f}")
    print(f"{'Pretrained + Fine-tuned (best)' + arch_note:<45} {best_dice:>10.4f}")

    print("-" * 70)

    if original_dice:
        diff = best_dice - original_dice
        same_arch = original_arch and pretrain_arch and \
                    original_arch['base_channels'] == pretrain_arch['base_channels']

        if diff > 0.01:
            print(f"{'IMPROVEMENT over original:':<45} {diff:>+10.4f} ✓")
            if same_arch:
                print("\n>>> PRETRAINING WAS VALUABLE! <<<")
            else:
                print("\n>>> Pretrained model is better (but different architecture)")
        elif diff < -0.01:
            print(f"{'DIFFERENCE from original:':<45} {diff:>+10.4f}")
            print("\n>>> Pretraining did not help (or needs more fine-tuning)")
        else:
            print(f"{'DIFFERENCE from original:':<45} {diff:>+10.4f}")
            print("\n>>> Roughly equivalent performance")

    print("=" * 70)

    # Save final results
    results = {
        'timestamp': datetime.now().isoformat(),
        'original_dice': original_dice,
        'original_architecture': original_arch,
        'pretrained_before_finetune': pretrain_dice_before,
        'pretrained_after_finetune': best_dice,
        'pretrained_architecture': pretrain_arch,
        'improvement': best_dice - original_dice if original_dice else None,
        'same_architecture': original_arch and pretrain_arch and \
                             original_arch['base_channels'] == pretrain_arch['base_channels'],
        'epochs_trained': args.epochs,
        'history': history
    }

    results_path = model_dir / 'finetune_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Best model saved to: {output_path}")


if __name__ == '__main__':
    main()
