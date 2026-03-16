"""
Improved training script with:
- Data augmentation (spatial + intensity)
- Combined loss (Dice + Focal + Tversky)
- Multiple patches per volume
- Leaderboard tracking

Usage:
    python scripts/train_improved.py --patch-size 16
    python scripts/train_improved.py --patch-size 48 --patches-per-volume 10
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import time
import json
import gc
from datetime import datetime

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D
from segmentation.leaderboard import Leaderboard


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss - focuses on hard examples."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class TverskyLoss(nn.Module):
    """Tversky loss - controls FP vs FN trade-off."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for FP
        self.beta = beta    # Weight for FN (higher = penalize missed lesions more)
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Combined loss optimized for tiny lesion detection."""
    def __init__(self, dice_weight=0.4, focal_weight=0.3, tversky_weight=0.3,
                 focal_alpha=0.75, focal_gamma=2.0, tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, pred, target):
        return (self.dice_weight * self.dice(pred, target) +
                self.focal_weight * self.focal(pred, target) +
                self.tversky_weight * self.tversky(pred, target))


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class Augmentation3D:
    """3D augmentation pipeline for medical images."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def random_flip(self, image, mask):
        """Random flip along each axis."""
        for axis in [1, 2, 3]:  # Skip channel axis
            if np.random.rand() < 0.5:
                image = torch.flip(image, [axis])
                mask = torch.flip(mask, [axis])
        return image, mask

    def random_rotate90(self, image, mask):
        """Random 90-degree rotation in xy plane."""
        k = np.random.randint(0, 4)
        if k > 0:
            image = torch.rot90(image, k, [2, 3])
            mask = torch.rot90(mask, k, [2, 3])
        return image, mask

    def random_intensity(self, image, mask):
        """Random intensity augmentation."""
        # Brightness
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            image = image * factor

        # Contrast
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Gaussian noise
        if np.random.rand() < 0.2:
            noise = torch.randn_like(image) * 0.05
            image = image + noise

        return image, mask

    def __call__(self, image, mask):
        """Apply augmentations."""
        if np.random.rand() > self.prob:
            return image, mask

        image, mask = self.random_flip(image, mask)
        image, mask = self.random_rotate90(image, mask)
        image, mask = self.random_intensity(image, mask)

        return image, mask


# =============================================================================
# METRICS
# =============================================================================

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

    intersection = (pred_binary * target).sum()
    results['overall'] = ((2 * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)).item()

    return results


def compute_sensitivity_specificity(pred, target, threshold=0.5):
    """Compute sensitivity and specificity."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    tp = ((pred_binary == 1) & (target == 1)).sum().float()
    tn = ((pred_binary == 0) & (target == 0)).sum().float()
    fp = ((pred_binary == 1) & (target == 0)).sum().float()
    fn = ((pred_binary == 0) & (target == 1)).sum().float()

    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)

    return sensitivity.item(), specificity.item()


# =============================================================================
# DATA LOADING
# =============================================================================

def get_batch_size_for_patch(patch_size, gpu_memory_gb=8.0):
    """Estimate optimal batch size based on patch size."""
    base_memory = 0.05
    scale = (patch_size / 16) ** 3
    mem_per_sample = base_memory * scale
    batch_size = int(gpu_memory_gb * 0.6 / mem_per_sample)
    return max(2, min(batch_size, 64))


def load_samples_with_multiple_patches(ds, patches_per_volume=5, chunk_size=30):
    """Load multiple patches per volume for more training data."""
    n = len(ds)
    all_images = []
    all_masks = []
    num_chunks = (n + chunk_size - 1) // chunk_size

    total_patches = n * patches_per_volume
    print(f"  Extracting {patches_per_volume} patches per volume ({total_patches} total)...")

    for chunk_idx, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        print(f"  Chunk {chunk_idx+1}/{num_chunks} ({start+1}-{end}/{n})...", end='\r')

        chunk_images = []
        chunk_masks = []

        for i in range(start, end):
            # Extract multiple patches from each volume
            for _ in range(patches_per_volume):
                img, mask, _ = ds[i]
                chunk_images.append(img)
                chunk_masks.append(mask)

        chunk_img_tensor = torch.stack(chunk_images)
        chunk_mask_tensor = torch.stack(chunk_masks)

        all_images.append(chunk_img_tensor)
        all_masks.append(chunk_mask_tensor)

        del chunk_images, chunk_masks
        gc.collect()

    print(f"  Loaded {total_patches} patches from {n} volumes.        ")

    images_tensor = torch.cat(all_images, dim=0)
    masks_tensor = torch.cat(all_masks, dim=0)

    del all_images, all_masks
    gc.collect()

    return images_tensor, masks_tensor


# =============================================================================
# AUGMENTED DATASET
# =============================================================================

class AugmentedTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with on-the-fly augmentation."""

    def __init__(self, images, masks, augment=True, aug_prob=0.5):
        self.images = images
        self.masks = masks
        self.augment = augment
        self.augmentation = Augmentation3D(prob=aug_prob) if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].clone()
        mask = self.masks[idx].clone()

        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        return image, mask


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs} [Train]", leave=False, ncols=100)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, mask)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device, epoch, epochs):
    """Validate for one epoch with size-stratified metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs} [Val]  ", leave=False, ncols=100)
    with torch.no_grad():
        for img, mask in pbar:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            loss = criterion(out, mask)
            total_loss += loss.item()
            all_preds.append(out.cpu())
            all_targets.append(mask.cpu())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    dice_scores = compute_size_stratified_dice(all_preds, all_targets)
    sensitivity, specificity = compute_sensitivity_specificity(all_preds, all_targets)

    return total_loss / len(loader), dice_scores, sensitivity, specificity


def train_model(
    patch_size,
    epochs=250,
    patches_per_volume=5,
    batch_size=None,
    lr=0.001,
    data_dir=None,
    model_dir=None,
    leaderboard=None,
    device=None,
    gpu_mem=8.0
):
    """Train a model with improved settings."""

    PATCH_SIZE = (patch_size, patch_size, patch_size)
    BATCH_SIZE = batch_size or get_batch_size_for_patch(patch_size, gpu_mem)

    # Load dataset
    print(f"Loading dataset...")
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=PATCH_SIZE,
        target_size=None,
        transform=None
    )

    # Pre-load with multiple patches per volume
    print(f"Pre-loading with {patches_per_volume} patches per volume...")
    start = time.time()
    images_tensor, masks_tensor = load_samples_with_multiple_patches(
        ds, patches_per_volume=patches_per_volume, chunk_size=30
    )

    img_mb = images_tensor.element_size() * images_tensor.numel() / 1e6
    print(f"Pre-loaded in {(time.time()-start)/60:.1f} min ({img_mb:.1f} MB)")
    print(f"Total samples: {len(images_tensor)}")

    # Train/val split
    n_samples = len(images_tensor)
    train_size = int(0.85 * n_samples)
    val_size = n_samples - train_size

    indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(42))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create datasets with augmentation for training only
    train_dataset = AugmentedTensorDataset(
        images_tensor[train_indices],
        masks_tensor[train_indices],
        augment=True,
        aug_prob=0.5
    )
    val_dataset = AugmentedTensorDataset(
        images_tensor[val_indices],
        masks_tensor[val_indices],
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Batch: {BATCH_SIZE}, Batches/epoch: {len(train_loader)}")

    # Model
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        use_attention=True,
        use_residual=True
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # Combined loss
    criterion = CombinedLoss(
        dice_weight=0.4,
        focal_weight=0.3,
        tversky_weight=0.3,
        focal_alpha=0.75,
        focal_gamma=2.0,
        tversky_alpha=0.3,
        tversky_beta=0.7
    )

    model_name = "improved"
    model_path = model_dir / f"{model_name}_{patch_size}patch_best.pth"
    state_path = model_dir / f"{model_name}_{patch_size}patch_state.json"

    print(f"\n{'='*70}")
    print(f"TRAINING: {patch_size}³ patches | {epochs} epochs | batch={BATCH_SIZE}")
    print(f"Loss: Dice(0.4) + Focal(0.3) + Tversky(0.3)")
    print(f"Augmentation: ON | Patches/volume: {patches_per_volume}")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    best_val_dice = 0
    best_tiny_dice = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)

        # Validate
        val_loss, dice_scores, sensitivity, specificity = validate_epoch(
            model, val_loader, criterion, device, epoch, epochs
        )

        scheduler.step()

        val_dice = dice_scores['overall']
        tiny_dice = dice_scores.get('tiny')
        small_dice = dice_scores.get('small')
        medium_dice = dice_scores.get('medium')
        large_dice = dice_scores.get('large')

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
                'dice_scores': dice_scores,
                'sensitivity': sensitivity,
                'specificity': specificity
            }, model_path)

        if leaderboard:
            leaderboard.update(
                model_name=model_name,
                patch_size=patch_size,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_dice=val_dice,
                tiny_dice=tiny_dice,
                small_dice=small_dice,
                medium_dice=medium_dice,
                large_dice=large_dice,
                sensitivity=sensitivity,
                specificity=specificity,
                model_path=str(model_path) if is_best else None,
                is_best=is_best
            )

        with open(state_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'epochs_total': epochs,
                'patch_size': patch_size,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'tiny_dice': tiny_dice,
                'small_dice': small_dice,
                'medium_dice': medium_dice,
                'large_dice': large_dice,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'best_val_loss': best_val_loss,
                'best_val_dice': best_val_dice,
                'best_tiny_dice': best_tiny_dice,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        epoch_time = time.time() - epoch_start
        best_marker = " *" if is_best else ""
        tiny_str = f"tiny={tiny_dice:.3f}" if tiny_dice else "tiny=N/A"
        print(f"Epoch {epoch:3d}/{epochs}: loss={train_loss:.4f}, dice={val_dice:.3f}, "
              f"{tiny_str}, sens={sensitivity:.3f} ({epoch_time:.2f}s){best_marker}")

    total_time = (time.time() - train_start) / 60

    print(f"\n{'='*70}")
    print(f"{patch_size}³ COMPLETE in {total_time:.1f} min")
    print(f"Best dice: {best_val_dice:.4f}, Best tiny: {best_tiny_dice:.4f}")
    print(f"{'='*70}")

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    del images_tensor, masks_tensor, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_dice, best_tiny_dice


def main():
    parser = argparse.ArgumentParser(description='Improved training with augmentation and combined loss')
    parser.add_argument('--patch-size', type=int, default=48, help='Patch size')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--patches-per-volume', type=int, default=5, help='Patches to extract per volume')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto if not specified)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--leaderboard', action='store_true', help='Just show leaderboard')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    model_dir.mkdir(exist_ok=True)

    leaderboard = Leaderboard(str(model_dir / "leaderboard.json"))

    if args.leaderboard:
        leaderboard.print_summary()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_mem = 8.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n{'#'*70}")
    print(f"# IMPROVED TRAINING")
    print(f"# Patch size: {args.patch_size}³")
    print(f"# Epochs: {args.epochs}")
    print(f"# Patches per volume: {args.patches_per_volume}")
    print(f"# Device: {device}")
    if torch.cuda.is_available():
        print(f"# GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    print(f"{'#'*70}")

    best_dice, best_tiny = train_model(
        patch_size=args.patch_size,
        epochs=args.epochs,
        patches_per_volume=args.patches_per_volume,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=data_dir,
        model_dir=model_dir,
        leaderboard=leaderboard,
        device=device,
        gpu_mem=gpu_mem
    )

    print("\n")
    leaderboard.print_summary()


if __name__ == '__main__':
    main()
