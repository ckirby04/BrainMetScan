"""
Fast training with pre-loading and leaderboard tracking.
Supports any patch size and automatically tracks performance.

Usage:
    python scripts/train_fast.py --patch-size 16
    python scripts/train_fast.py --patch-size 48
    python scripts/train_fast.py --patch-size 32 --batch-size 32 --epochs 200
    python scripts/train_fast.py --leaderboard  # View leaderboard only
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import time
import json
from datetime import datetime

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D
from segmentation.leaderboard import Leaderboard


def compute_size_stratified_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Compute dice scores stratified by lesion size."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    # Size bins in voxels
    SIZE_BINS = {
        'tiny': (0, 500),
        'small': (500, 2000),
        'medium': (2000, 5000),
        'large': (5000, float('inf'))
    }

    results = {}
    batch_size = target.shape[0]

    for name, (low, high) in SIZE_BINS.items():
        dice_scores = []
        for i in range(batch_size):
            mask = target[i]
            volume = mask.sum().item()

            if low <= volume < high:
                p = pred_binary[i]
                t = mask
                intersection = (p * t).sum()
                dice = (2 * intersection + 1e-6) / (p.sum() + t.sum() + 1e-6)
                dice_scores.append(dice.item())

        if dice_scores:
            results[name] = sum(dice_scores) / len(dice_scores)
        else:
            results[name] = None

    # Overall dice
    intersection = (pred_binary * target).sum()
    overall_dice = (2 * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)
    results['overall'] = overall_dice.item()

    return results


def get_batch_size_for_patch(patch_size: int, gpu_memory_gb: float = 8.0) -> int:
    """Estimate optimal batch size based on patch size and GPU memory."""
    # Rough estimates based on 3D U-Net memory usage
    memory_per_sample = {
        16: 0.05,   # ~50 MB per sample
        24: 0.15,   # ~150 MB per sample
        32: 0.35,   # ~350 MB per sample
        48: 0.8,    # ~800 MB per sample
        64: 1.5,    # ~1.5 GB per sample
        96: 4.0,    # ~4 GB per sample
    }

    mem = memory_per_sample.get(patch_size, patch_size ** 3 / 16 ** 3 * 0.05)
    batch_size = int(gpu_memory_gb * 0.7 / mem)  # Use 70% of GPU memory
    return max(1, min(batch_size, 128))


def main():
    parser = argparse.ArgumentParser(description='Fast training with leaderboard')
    parser.add_argument('--patch-size', type=int, default=48, help='Patch size (16, 24, 32, 48, 64, 96)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto if not specified)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
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

    # Auto batch size based on GPU memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        BATCH_SIZE = args.batch_size or get_batch_size_for_patch(args.patch_size, gpu_mem)
    else:
        BATCH_SIZE = args.batch_size or 4

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_mem:.1f} GB")

    # Load dataset
    print(f"\nLoading dataset from {data_dir}")
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=PATCH_SIZE,
        target_size=None,
        transform=None
    )

    # Pre-load ALL samples into RAM
    print(f"\nPre-loading {len(ds)} samples into RAM...")
    print(f"Patch size: {args.patch_size}³")
    start = time.time()

    import gc

    all_images = []
    all_masks = []
    for i in tqdm(range(len(ds))):
        img, mask, _ = ds[i]
        all_images.append(img)
        all_masks.append(mask)

        # Force garbage collection every 50 samples to prevent memory buildup
        if i % 50 == 0:
            gc.collect()

    images_tensor = torch.stack(all_images)
    masks_tensor = torch.stack(all_masks)
    del all_images, all_masks
    gc.collect()

    img_size_mb = images_tensor.element_size() * images_tensor.numel() / 1e6
    mask_size_mb = masks_tensor.element_size() * masks_tensor.numel() / 1e6
    print(f"Pre-loaded in {(time.time()-start)/60:.1f} min")
    print(f"Images: {images_tensor.shape} ({img_size_mb:.1f} MB)")
    print(f"Masks: {masks_tensor.shape} ({mask_size_mb:.1f} MB)")

    # Train/val split (85/15)
    cached_ds = TensorDataset(images_tensor, masks_tensor)
    train_size = int(0.85 * len(cached_ds))
    val_size = len(cached_ds) - train_size
    train_ds, val_ds = random_split(cached_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Batches/epoch: {len(train_loader)}")
    print(f"Batch size: {BATCH_SIZE}")

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
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate with size-stratified metrics
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                val_loss += criterion(out, mask).item()
                all_preds.append(out.cpu())
                all_targets.append(mask.cpu())

        val_loss /= len(val_loader)

        # Compute size-stratified dice
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        dice_scores = compute_size_stratified_dice(all_preds, all_targets)

        val_dice = dice_scores['overall']
        tiny_dice = dice_scores.get('tiny')
        small_dice = dice_scores.get('small')
        medium_dice = dice_scores.get('medium')
        large_dice = dice_scores.get('large')

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
                'val_dice': val_dice,
                'dice_scores': dice_scores
            }, model_path)

        # Update leaderboard
        leaderboard.update(
            model_name=model_name,
            patch_size=args.patch_size,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_dice=val_dice,
            tiny_dice=tiny_dice,
            small_dice=small_dice,
            medium_dice=medium_dice,
            large_dice=large_dice,
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
                'dice_scores': dice_scores,
                'best_val_loss': best_val_loss,
                'best_val_dice': best_val_dice,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        epoch_time = time.time() - epoch_start

        # Print progress
        if epoch % 10 == 0 or epoch <= 5:
            tiny_str = f"tiny={tiny_dice:.3f}" if tiny_dice else "tiny=N/A"
            print(f"Epoch {epoch:3d}/{EPOCHS}: loss={train_loss:.4f}, dice={val_dice:.3f}, "
                  f"{tiny_str}, best={best_val_dice:.3f} ({epoch_time:.2f}s)")

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
