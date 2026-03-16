"""
Fast 16-patch training with pre-loading.
Pre-loads all patches into RAM for maximum GPU utilization.

Usage:
    python scripts/train_16patch_fast.py
"""

import sys
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


def main():
    # Config
    BATCH_SIZE = 64
    EPOCHS = 150
    LR = 0.001
    PATCH_SIZE = (16, 16, 16)

    # Paths
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    model_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    start = time.time()
    all_images = []
    all_masks = []
    for i in tqdm(range(len(ds))):
        img, mask, _ = ds[i]
        all_images.append(img)
        all_masks.append(mask)

    images_tensor = torch.stack(all_images)
    masks_tensor = torch.stack(all_masks)
    del all_images, all_masks

    print(f"Pre-loaded in {(time.time()-start)/60:.1f} min")
    print(f"Images: {images_tensor.shape} ({images_tensor.element_size() * images_tensor.numel() / 1e6:.1f} MB)")

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

    # Model
    model = LightweightUNet3D(in_channels=4, out_channels=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    best_val_loss = float('inf')
    best_val_dice = 0
    model_path = model_dir / "tiny_lesion_16patch_best.pth"
    state_path = model_dir / "tiny_lesion_16patch_state.json"

    print(f"\n{'='*60}")
    print(f"Starting training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"{'='*60}\n")

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

        # Validate
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                val_loss += criterion(out, mask).item()

                # Dice score
                pred = (torch.sigmoid(out) > 0.5).float()
                intersection = (pred * mask).sum()
                dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
                val_dice += dice.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice
            }, model_path)

        # Save state for monitoring
        with open(state_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'epochs_total': EPOCHS,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'best_val_loss': best_val_loss,
                'best_val_dice': best_val_dice,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        epoch_time = time.time() - epoch_start
        if epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}/{EPOCHS}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"dice={val_dice:.3f}, best={best_val_loss:.4f} ({epoch_time:.2f}s)")

    total_time = (time.time() - train_start) / 60
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time:.1f} min")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Model saved: {model_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
