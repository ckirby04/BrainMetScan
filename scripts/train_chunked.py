"""
Chunked pre-loading training - memory efficient version.
Loads data in chunks to avoid memory fragmentation issues.

Usage:
    python scripts/train_chunked.py --patch-size 16
    python scripts/train_chunked.py --patch-size 48
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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


def get_batch_size_for_patch(patch_size: int, gpu_memory_gb: float = 8.0) -> int:
    memory_per_sample = {16: 0.05, 24: 0.15, 32: 0.35, 48: 0.8, 64: 1.5, 96: 4.0}
    mem = memory_per_sample.get(patch_size, patch_size ** 3 / 16 ** 3 * 0.05)
    batch_size = int(gpu_memory_gb * 0.7 / mem)
    return max(1, min(batch_size, 64))


def load_samples_chunked(ds, chunk_size=50):
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
    parser = argparse.ArgumentParser(description='Chunked pre-loading training')
    parser.add_argument('--patch-size', type=int, default=48, help='Patch size')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--chunk-size', type=int, default=50, help='Samples per chunk during loading')
    parser.add_argument('--leaderboard', action='store_true', help='Just show leaderboard')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    model_dir.mkdir(exist_ok=True)

    leaderboard = Leaderboard(str(model_dir / "leaderboard.json"))

    if args.leaderboard:
        leaderboard.print_summary()
        return

    PATCH_SIZE = (args.patch_size, args.patch_size, args.patch_size)
    EPOCHS = args.epochs
    LR = args.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        BATCH_SIZE = args.batch_size or get_batch_size_for_patch(args.patch_size, gpu_mem)
    else:
        BATCH_SIZE = args.batch_size or 4
        gpu_mem = 0

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    print(f"\nLoading dataset from {data_dir}")
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=PATCH_SIZE,
        target_size=None,
        transform=None
    )

    print(f"\nPre-loading {len(ds)} samples in chunks of {args.chunk_size}...")
    start = time.time()

    images_tensor, masks_tensor = load_samples_chunked(ds, chunk_size=args.chunk_size)

    img_mb = images_tensor.element_size() * images_tensor.numel() / 1e6
    mask_mb = masks_tensor.element_size() * masks_tensor.numel() / 1e6
    print(f"\nPre-loaded in {(time.time()-start)/60:.1f} min")
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

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Batches/epoch: {len(train_loader)}")
    print(f"Batch size: {BATCH_SIZE}")

    model = LightweightUNet3D(in_channels=4, out_channels=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

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

        # Validate
        model.eval()
        val_loss = 0
        val_dice = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [Val]  ",
                        leave=False, ncols=100)
        with torch.no_grad():
            for img, mask in val_pbar:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                batch_loss = criterion(out, mask).item()
                val_loss += batch_loss

                pred = (torch.sigmoid(out) > 0.5).float()
                intersection = (pred * mask).sum()
                dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
                val_dice += dice.item()
                val_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'dice': f'{dice.item():.3f}'})

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step()

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
        best_marker = " *NEW BEST*" if is_best else ""
        print(f"Epoch {epoch:3d}/{EPOCHS}: train={train_loss:.4f}, val={val_loss:.4f}, "
              f"dice={val_dice:.3f}, best={best_val_dice:.3f} ({epoch_time:.2f}s){best_marker}")

    total_time = (time.time() - train_start) / 60
    print(f"\n{'='*70}")
    print(f"Training complete in {total_time:.1f} min")
    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Model saved: {model_path}")
    print(f"{'='*70}\n")

    leaderboard.print_summary()


if __name__ == '__main__':
    main()
