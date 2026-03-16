"""
Train 16³ patch model with chunked pre-loading.
Optimized for tiny lesion detection.

Usage:
    python scripts/train_16patch.py
    python scripts/train_16patch.py --epochs 200 --lr 0.0005
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
import gc
from datetime import datetime

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D
from segmentation.leaderboard import Leaderboard


def compute_size_stratified_dice(pred, target, threshold=0.5):
    """Compute dice scores stratified by lesion size."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()

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

        results[name] = sum(dice_scores) / len(dice_scores) if dice_scores else None

    # Overall dice
    intersection = (pred_binary * target).sum()
    results['overall'] = ((2 * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)).item()

    return results


def load_samples_chunked(ds, chunk_size=30):
    """Load samples in chunks to avoid memory fragmentation."""
    n = len(ds)
    all_images = []
    all_masks = []
    num_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_idx, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        print(f"  Loading chunk {chunk_idx+1}/{num_chunks} ({start+1}-{end}/{n})...", end='\r')

        chunk_images = []
        chunk_masks = []

        for i in range(start, end):
            img, mask, _ = ds[i]
            chunk_images.append(img)
            chunk_masks.append(mask)

        chunk_img_tensor = torch.stack(chunk_images)
        chunk_mask_tensor = torch.stack(chunk_masks)

        all_images.append(chunk_img_tensor)
        all_masks.append(chunk_mask_tensor)

        del chunk_images, chunk_masks
        gc.collect()

    print(f"  Loaded {n} samples in {num_chunks} chunks.        ")

    images_tensor = torch.cat(all_images, dim=0)
    masks_tensor = torch.cat(all_masks, dim=0)

    del all_images, all_masks
    gc.collect()

    return images_tensor, masks_tensor


def main():
    parser = argparse.ArgumentParser(description='Train 16³ patch model')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--chunk-size', type=int, default=30, help='Samples per chunk during loading')
    args = parser.parse_args()

    # Fixed config for 16³ patches
    PATCH_SIZE = (16, 16, 16)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr

    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    model_dir.mkdir(exist_ok=True)

    leaderboard = Leaderboard(str(model_dir / "leaderboard.json"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"16³ PATCH TRAINING - Tiny Lesion Detection")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    print(f"\nLoading dataset from {data_dir}")
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=PATCH_SIZE,
        target_size=None,
        transform=None
    )

    print(f"\nPre-loading {len(ds)} samples (chunk_size={args.chunk_size})...")
    start = time.time()
    images_tensor, masks_tensor = load_samples_chunked(ds, chunk_size=args.chunk_size)

    img_mb = images_tensor.element_size() * images_tensor.numel() / 1e6
    mask_mb = masks_tensor.element_size() * masks_tensor.numel() / 1e6
    print(f"Pre-loaded in {(time.time()-start)/60:.1f} min")
    print(f"Images: {images_tensor.shape} ({img_mb:.1f} MB)")
    print(f"Masks: {masks_tensor.shape} ({mask_mb:.1f} MB)")

    # Train/val split
    cached_ds = TensorDataset(images_tensor, masks_tensor)
    train_size = int(0.85 * len(cached_ds))
    val_size = len(cached_ds) - train_size
    train_ds, val_ds = random_split(cached_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Batch size: {BATCH_SIZE}, Batches/epoch: {len(train_loader)}")

    model = LightweightUNet3D(in_channels=4, out_channels=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    model_name = "tiny_lesion"
    model_path = model_dir / f"{model_name}_16patch_best.pth"
    state_path = model_dir / f"{model_name}_16patch_state.json"

    print(f"\n{'='*70}")
    print(f"TRAINING: 16³ patches | {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR}")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    best_val_dice = 0
    best_tiny_dice = 0
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [Train]",
                          leave=False, ncols=100)
        for img, mask in train_pbar:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        train_loss /= len(train_loader)

        # Validate with size-stratified metrics
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [Val]  ",
                        leave=False, ncols=100)
        with torch.no_grad():
            for img, mask in val_pbar:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                batch_loss = criterion(out, mask).item()
                val_loss += batch_loss
                all_preds.append(out.cpu())
                all_targets.append(mask.cpu())
                val_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

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

        # Track best tiny dice separately
        if tiny_dice and tiny_dice > best_tiny_dice:
            best_tiny_dice = tiny_dice

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

        leaderboard.update(
            model_name=model_name,
            patch_size=16,
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

        with open(state_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'epochs_total': EPOCHS,
                'patch_size': 16,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'tiny_dice': tiny_dice,
                'small_dice': small_dice,
                'medium_dice': medium_dice,
                'large_dice': large_dice,
                'best_val_loss': best_val_loss,
                'best_val_dice': best_val_dice,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        epoch_time = time.time() - epoch_start
        best_marker = " *" if is_best else ""
        tiny_str = f"tiny={tiny_dice:.3f}" if tiny_dice else "tiny=N/A"
        print(f"Epoch {epoch:3d}/{EPOCHS}: loss={train_loss:.4f}, dice={val_dice:.3f}, "
              f"{tiny_str}, best={best_val_dice:.3f} ({epoch_time:.2f}s){best_marker}")

    total_time = (time.time() - train_start) / 60
    print(f"\n{'='*70}")
    print(f"Training complete in {total_time:.1f} min")
    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Best tiny dice: {best_tiny_dice:.4f}")
    print(f"Model saved: {model_path}")
    print(f"{'='*70}\n")

    leaderboard.print_summary()


if __name__ == '__main__':
    main()
