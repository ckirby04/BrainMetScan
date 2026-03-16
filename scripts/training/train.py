"""
Standard training script with leaderboard tracking.
Uses DataLoader with workers - slower but memory efficient.

Usage:
    python scripts/train.py --patch-size 16
    python scripts/train.py --patch-size 48
    python scripts/train.py --patch-size 32 --batch-size 8 --epochs 150
    python scripts/train.py --leaderboard
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import json
from datetime import datetime

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D
from segmentation.leaderboard import Leaderboard


def get_batch_size_for_patch(patch_size: int, gpu_memory_gb: float = 8.0) -> int:
    """Estimate optimal batch size based on patch size and GPU memory."""
    memory_per_sample = {
        16: 0.05,
        24: 0.15,
        32: 0.35,
        48: 0.8,
        64: 1.5,
        96: 4.0,
    }
    mem = memory_per_sample.get(patch_size, patch_size ** 3 / 16 ** 3 * 0.05)
    batch_size = int(gpu_memory_gb * 0.7 / mem)
    return max(1, min(batch_size, 64))


def main():
    parser = argparse.ArgumentParser(description='Train with leaderboard tracking')
    parser.add_argument('--patch-size', type=int, default=48, help='Patch size (16, 24, 32, 48, 64, 96)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto if not specified)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--leaderboard', action='store_true', help='Just show leaderboard')
    args = parser.parse_args()

    # Paths
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    model_dir.mkdir(exist_ok=True)

    # Initialize leaderboard
    leaderboard = Leaderboard(str(model_dir / "leaderboard.json"))

    if args.leaderboard:
        leaderboard.print_summary()
        return

    # Config
    PATCH_SIZE = (args.patch_size, args.patch_size, args.patch_size)
    EPOCHS = args.epochs
    LR = args.lr
    NUM_WORKERS = args.num_workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        BATCH_SIZE = args.batch_size or get_batch_size_for_patch(args.patch_size, gpu_mem)
    else:
        BATCH_SIZE = args.batch_size or 4
        gpu_mem = 0

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_mem:.1f} GB")

    # Load dataset
    print(f"\nLoading dataset from {data_dir}")
    full_dataset = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=PATCH_SIZE,
        target_size=None,
        transform=None
    )

    # Train/val split (85/15)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Model
    model = LightweightUNet3D(in_channels=4, out_channels=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    model_name = "tiny_lesion"
    model_path = model_dir / f"{model_name}_{args.patch_size}patch_best.pth"
    state_path = model_dir / f"{model_name}_{args.patch_size}patch_state.json"

    print(f"\n{'='*70}")
    print(f"TRAINING: {args.patch_size}³ patches | {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR}")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    best_val_dice = 0
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for img, mask, _ in train_pbar:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for img, mask, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False):
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                val_loss += criterion(out, mask).item()

                # Dice
                pred = (torch.sigmoid(out) > 0.5).float()
                intersection = (pred * mask).sum()
                dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
                val_dice += dice.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step()

        # Save best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice
            }, model_path)

        # Update leaderboard
        leaderboard.update(
            model_name=model_name,
            patch_size=args.patch_size,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_dice=val_dice,
            model_path=str(model_path) if is_best else None,
            is_best=is_best
        )

        # Save state
        with open(state_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'epochs_total': EPOCHS,
                'patch_size': args.patch_size,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'best_val_loss': best_val_loss,
                'best_val_dice': best_val_dice,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{EPOCHS}: train={train_loss:.4f}, val={val_loss:.4f}, "
              f"dice={val_dice:.3f}, best={best_val_dice:.3f} ({epoch_time:.1f}s)")

    total_time = (time.time() - train_start) / 60

    print(f"\n{'='*70}")
    print(f"Training complete in {total_time:.1f} min")
    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Model saved: {model_path}")
    print(f"{'='*70}")

    # Show leaderboard
    print("\n")
    leaderboard.print_summary()


if __name__ == '__main__':
    main()
