"""
Overnight Training Experiments
==============================
Runs multiple training experiments to improve lesion detection.

Current Status:
- Voxel sensitivity: 95.5% achievable (threshold 0.05)
- Lesion sensitivity: 71.9% max (need 90%+)
- Missing ~28% of lesions (mostly tiny)

Experiments:
1. Ultra-tiny patches (8x8x8) - catch smallest lesions
2. 16x16x16 patches - bridge gap between 12 and 24
3. Aggressive FN penalty (Tversky alpha=0.2, beta=0.8)
4. Higher lesion weights in loss
5. Heavy augmentation training
6. Deep ensemble (more models)
7. Final ensemble with all new models

Estimated time: ~7-8 hours total
"""

import sys
import os
import gc
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class FocalLoss(nn.Module):
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
    """Tversky loss - alpha < beta penalizes false negatives more."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight (higher = penalize missing lesions more)
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class SizeWeightedDiceLoss(nn.Module):
    """Weight loss by lesion size - penalize missing small lesions more."""
    def __init__(self, tiny_weight=4.0, small_weight=2.0, smooth=1e-6):
        super().__init__()
        self.tiny_weight = tiny_weight
        self.small_weight = small_weight
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)

        # Compute base dice
        intersection = (pred_sig * target).sum()
        base_dice = (2 * intersection + self.smooth) / (pred_sig.sum() + target.sum() + self.smooth)

        # Weight by target size (smaller targets get higher weight)
        target_size = target.sum()
        if target_size < 500:
            weight = self.tiny_weight
        elif target_size < 2000:
            weight = self.small_weight
        else:
            weight = 1.0

        return weight * (1 - base_dice)


class CombinedLossV2(nn.Module):
    """Enhanced combined loss with configurable FN penalty."""
    def __init__(self, tversky_alpha=0.3, tversky_beta=0.7, focal_alpha=0.75):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=2.0)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, pred, target):
        return 0.3 * self.dice(pred, target) + \
               0.3 * self.focal(pred, target) + \
               0.4 * self.tversky(pred, target)


class AggressiveFNLoss(nn.Module):
    """Very aggressive false negative penalty for maximum sensitivity."""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=0.1, beta=0.9)  # 9x FN penalty
        self.focal = FocalLoss(alpha=0.9, gamma=3.0)     # High alpha for positives

    def forward(self, pred, target):
        return 0.2 * self.dice(pred, target) + \
               0.5 * self.tversky(pred, target) + \
               0.3 * self.focal(pred, target)


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class Augmentation3D:
    """3D augmentation for medical images.

    Expects tensors of shape [C, D, H, W] (4D, no batch).
    """

    def __init__(self, flip_prob=0.5, rotate_prob=0.5, intensity_prob=0.3,
                 noise_prob=0.2, scale_prob=0.2):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.intensity_prob = intensity_prob
        self.noise_prob = noise_prob
        self.scale_prob = scale_prob

    def __call__(self, image, mask):
        # image shape: [C, D, H, W], mask shape: [1, D, H, W]

        # Random flips along spatial axes (1, 2, 3 for D, H, W)
        if np.random.random() < self.flip_prob:
            axis = np.random.randint(1, 4)  # axes 1,2,3 (spatial)
            image = torch.flip(image, [axis])
            mask = torch.flip(mask, [axis])

        # Random 90-degree rotations on spatial plane pairs
        if np.random.random() < self.rotate_prob:
            k = np.random.randint(1, 4)
            # Rotate in D-H, D-W, or H-W planes (axes 1-2, 1-3, or 2-3)
            dims = [(1, 2), (1, 3), (2, 3)][np.random.randint(3)]
            image = torch.rot90(image, k, dims)
            mask = torch.rot90(mask, k, dims)

        # Intensity augmentation
        if np.random.random() < self.intensity_prob:
            # Brightness
            image = image + (torch.rand(1).item() - 0.5) * 0.2
            # Contrast
            image = image * (0.8 + torch.rand(1).item() * 0.4)
            image = torch.clamp(image, 0, 1)

        # Gaussian noise
        if np.random.random() < self.noise_prob:
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0, 1)

        return image, mask


class HeavyAugmentation3D(Augmentation3D):
    """More aggressive augmentation."""
    def __init__(self):
        super().__init__(
            flip_prob=0.7,
            rotate_prob=0.7,
            intensity_prob=0.5,
            noise_prob=0.4,
            scale_prob=0.3
        )


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(pred, target, threshold=0.5):
    """Compute all metrics including lesion-wise."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    # Voxel-wise
    tp = ((pred_binary == 1) & (target == 1)).sum().float()
    tn = ((pred_binary == 0) & (target == 0)).sum().float()
    fp = ((pred_binary == 1) & (target == 0)).sum().float()
    fn = ((pred_binary == 0) & (target == 1)).sum().float()

    sensitivity = (tp / (tp + fn + 1e-6)).item()
    specificity = (tn / (tn + fp + 1e-6)).item()
    precision = (tp / (tp + fp + 1e-6)).item()

    intersection = (pred_binary * target).sum()
    dice = ((2 * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)).item()

    # Size-stratified dice
    size_dice = compute_size_stratified_dice(pred, target, threshold)

    # Lesion-wise (sample a subset for speed)
    lesion_metrics = compute_lesion_metrics_fast(pred_binary, target)

    return {
        'dice': dice,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        **size_dice,
        **lesion_metrics
    }


def compute_size_stratified_dice(pred, target, threshold=0.5):
    """Compute dice by lesion size."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    SIZE_BINS = {
        'tiny': (0, 500),
        'small': (500, 2000),
        'medium': (2000, 5000),
        'large': (5000, float('inf'))
    }

    results = {}
    pred_np = pred_binary.cpu().numpy()
    target_np = target.cpu().numpy()

    for size_name, (min_size, max_size) in SIZE_BINS.items():
        dice_scores = []

        for b in range(target_np.shape[0]):
            labeled, n_lesions = ndimage.label(target_np[b, 0])

            for i in range(1, n_lesions + 1):
                lesion_mask = (labeled == i)
                lesion_size = lesion_mask.sum()

                if min_size <= lesion_size < max_size:
                    pred_region = pred_np[b, 0] * lesion_mask
                    intersection = (pred_region * lesion_mask).sum()
                    dice = (2 * intersection) / (pred_region.sum() + lesion_mask.sum() + 1e-6)
                    dice_scores.append(dice)

        results[f'{size_name}_dice'] = np.mean(dice_scores) if dice_scores else 0.0

    return results


def compute_lesion_metrics_fast(pred_binary, target, max_samples=50):
    """Fast lesion-wise metrics on subset."""
    pred_np = pred_binary.cpu().numpy().astype(np.uint8)
    target_np = target.cpu().numpy().astype(np.uint8)

    n_samples = min(pred_np.shape[0], max_samples)

    total_true = 0
    total_pred = 0
    total_detected = 0
    total_tp = 0

    for b in range(n_samples):
        pred_labels, n_pred = ndimage.label(pred_np[b, 0])
        target_labels, n_target = ndimage.label(target_np[b, 0])

        total_pred += n_pred
        total_true += n_target

        for i in range(1, n_target + 1):
            mask = (target_labels == i)
            if (pred_np[b, 0] * mask).sum() > 0.3 * mask.sum():
                total_detected += 1

        for i in range(1, n_pred + 1):
            mask = (pred_labels == i)
            if (target_np[b, 0] * mask).sum() > 0.3 * mask.sum():
                total_tp += 1

    lesion_sens = total_detected / (total_true + 1e-6)
    lesion_prec = total_tp / (total_pred + 1e-6)
    lesion_f1 = 2 * lesion_sens * lesion_prec / (lesion_sens + lesion_prec + 1e-6)

    return {
        'lesion_sensitivity': lesion_sens,
        'lesion_precision': lesion_prec,
        'lesion_f1': lesion_f1,
        'n_true_lesions': total_true,
        'n_detected': total_detected
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_chunked(data_dir, patch_size, n_patches=3, max_volumes=None):
    """Load data with chunked approach to prevent OOM."""
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=(patch_size, patch_size, patch_size),
        target_size=None,
        transform=None
    )

    n_volumes = len(ds) if max_volumes is None else min(len(ds), max_volumes)

    images, masks = [], []
    print(f"Loading {n_volumes} volumes x {n_patches} patches...")

    for i in range(n_volumes):
        try:
            for _ in range(n_patches):
                img, mask, _ = ds[i]
                images.append(img)
                masks.append(mask)
        except Exception as e:
            print(f"  Warning: Failed to load volume {i}: {e}")
            continue

        # Aggressive garbage collection every 20 volumes
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i+1}/{n_volumes} volumes ({len(images)} patches)")
            gc.collect()

    print(f"  Total patches loaded: {len(images)}")
    return torch.stack(images), torch.stack(masks)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_experiment(
    name,
    patch_size,
    loss_fn,
    augmentation=None,
    epochs=150,
    batch_size=64,
    lr=1e-3,
    n_patches=3,
    max_volumes=None,
    use_amp=True
):
    """Train a single experiment."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print(f"Patch size: {patch_size}, Epochs: {epochs}, Batch: {batch_size}")
    print("=" * 70)

    start_time = time.time()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    images, masks = load_data_chunked(data_dir, patch_size, n_patches, max_volumes)

    # Train/val split
    n = len(images)
    indices = torch.randperm(n)
    train_idx = indices[:int(0.85 * n)]
    val_idx = indices[int(0.85 * n):]

    train_images, train_masks = images[train_idx], masks[train_idx]
    val_images, val_masks = images[val_idx], masks[val_idx]

    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Clear memory
    del images, masks
    gc.collect()
    torch.cuda.empty_cache()

    # Model
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        use_attention=True,
        use_residual=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if use_amp else None

    # Training loop
    best_lesion_f1 = 0
    best_metrics = {}

    train_loader = DataLoader(
        TensorDataset(train_images, train_masks),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_img, batch_mask in train_loader:
            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device)

            # Apply augmentation
            if augmentation is not None:
                aug_imgs, aug_masks = [], []
                for i in range(batch_img.shape[0]):
                    img, mask = augmentation(batch_img[i], batch_mask[i])
                    aug_imgs.append(img)
                    aug_masks.append(mask)
                batch_img = torch.stack(aug_imgs)
                batch_mask = torch.stack(aug_masks)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    pred = model(batch_img)
                    loss = loss_fn(pred, batch_mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(batch_img)
                loss = loss_fn(pred, batch_mask)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loader = DataLoader(
                    TensorDataset(val_images, val_masks),
                    batch_size=batch_size,
                    shuffle=False
                )

                all_preds, all_targets = [], []
                val_loss = 0

                for batch_img, batch_mask in val_loader:
                    batch_img = batch_img.to(device)
                    batch_mask = batch_mask.to(device)

                    with autocast():
                        pred = model(batch_img)
                        val_loss += loss_fn(pred, batch_mask).item()

                    all_preds.append(pred.cpu())
                    all_targets.append(batch_mask.cpu())

                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)

                metrics = compute_metrics(all_preds, all_targets)
                val_loss /= len(val_loader)

                # Check for best
                if metrics['lesion_f1'] > best_lesion_f1:
                    best_lesion_f1 = metrics['lesion_f1']
                    best_metrics = metrics.copy()

                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'metrics': metrics
                    }, model_dir / f"{name}_best.pth")

                print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                      f"Dice: {metrics['dice']*100:.1f}% | Sens: {metrics['sensitivity']*100:.1f}% | "
                      f"L-Sens: {metrics['lesion_sensitivity']*100:.1f}% | L-F1: {metrics['lesion_f1']*100:.1f}%")

    # Save final state
    elapsed = time.time() - start_time

    state = {
        'experiment': name,
        'patch_size': patch_size,
        'epochs': epochs,
        'elapsed_minutes': elapsed / 60,
        'best_metrics': best_metrics,
        'final_metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    with open(model_dir / f"{name}_state.json", 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\nCompleted {name} in {elapsed/60:.1f} minutes")
    print(f"Best Lesion F1: {best_lesion_f1*100:.1f}%")

    # Cleanup
    del model, train_images, train_masks, val_images, val_masks
    gc.collect()
    torch.cuda.empty_cache()

    return best_metrics


# ============================================================================
# ENSEMBLE EVALUATION
# ============================================================================

def evaluate_ensemble(model_names, patch_sizes, data_dir, model_dir, device):
    """Evaluate ensemble of models."""
    print("\n" + "=" * 70)
    print("EVALUATING ENSEMBLE")
    print("=" * 70)

    # Load validation data at largest patch size
    max_patch = max(patch_sizes)
    images, masks = load_data_chunked(data_dir, max_patch, n_patches=2, max_volumes=100)

    # Get predictions from each model
    all_preds = []

    for name, patch_size in zip(model_names, patch_sizes):
        model_path = model_dir / f"{name}_best.pth"
        if not model_path.exists():
            print(f"  Skipping {name} - not found")
            continue

        print(f"  Loading {name}...")
        model = LightweightUNet3D(in_channels=4, out_channels=1, use_attention=True, use_residual=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        preds = []
        loader = DataLoader(TensorDataset(images), batch_size=8, shuffle=False)

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)

                # Resize if needed
                if patch_size != max_patch:
                    batch_resized = F.interpolate(batch, size=(patch_size,)*3, mode='trilinear', align_corners=False)
                    pred = model(batch_resized)
                    pred = F.interpolate(pred, size=(max_patch,)*3, mode='trilinear', align_corners=False)
                else:
                    pred = model(batch)

                preds.append(torch.sigmoid(pred).cpu())

        all_preds.append(torch.cat(preds))

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if not all_preds:
        print("No models found!")
        return {}

    # Average predictions
    ensemble_pred = torch.stack(all_preds).mean(dim=0)

    # Evaluate at multiple thresholds
    print("\nEnsemble Results:")
    print("-" * 60)

    best_result = None
    for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]:
        metrics = compute_metrics(ensemble_pred, masks, threshold=thresh)
        print(f"  Thresh {thresh:.2f}: Dice={metrics['dice']*100:.1f}%, "
              f"Sens={metrics['sensitivity']*100:.1f}%, "
              f"L-Sens={metrics['lesion_sensitivity']*100:.1f}%, "
              f"L-F1={metrics['lesion_f1']*100:.1f}%")

        if best_result is None or metrics['lesion_f1'] > best_result['lesion_f1']:
            best_result = metrics
            best_result['threshold'] = thresh

    return best_result


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    print("=" * 70)
    print("OVERNIGHT TRAINING EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated completion: {(datetime.now() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    output_dir = project_dir / "outputs"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    # =========================================================================
    # EXPERIMENT 1: Ultra-tiny patches (8x8x8)
    # Goal: Catch the smallest lesions that larger patches miss
    # =========================================================================
    try:
        results['exp1_8patch'] = train_experiment(
            name='exp1_8patch',
            patch_size=8,
            loss_fn=AggressiveFNLoss(),
            augmentation=Augmentation3D(),
            epochs=200,
            batch_size=128,
            n_patches=5,
            max_volumes=50  # Safe for 32GB RAM
        )
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
        results['exp1_8patch'] = {'error': str(e)}

    # =========================================================================
    # EXPERIMENT 2: 16x16x16 patches with aggressive FN penalty
    # Goal: Bridge gap between 12 and 24, focus on small lesions
    # =========================================================================
    try:
        results['exp2_16patch_aggressive'] = train_experiment(
            name='exp2_16patch_aggressive',
            patch_size=16,
            loss_fn=AggressiveFNLoss(),
            augmentation=Augmentation3D(),
            epochs=200,
            batch_size=96,
            n_patches=4,
            max_volumes=50  # Safe for 32GB RAM
        )
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
        results['exp2_16patch_aggressive'] = {'error': str(e)}

    # =========================================================================
    # EXPERIMENT 3: 12-patch with maximum FN penalty
    # Goal: Retrain 12-patch with even more aggressive loss
    # =========================================================================
    try:
        results['exp3_12patch_maxfn'] = train_experiment(
            name='exp3_12patch_maxfn',
            patch_size=12,
            loss_fn=AggressiveFNLoss(),
            augmentation=HeavyAugmentation3D(),
            epochs=200,
            batch_size=96,
            n_patches=5,
            max_volumes=50  # Safe for 32GB RAM
        )
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
        results['exp3_12patch_maxfn'] = {'error': str(e)}

    # =========================================================================
    # EXPERIMENT 4: 24-patch with heavy augmentation
    # Goal: More robust 24-patch model
    # =========================================================================
    try:
        results['exp4_24patch_heavy'] = train_experiment(
            name='exp4_24patch_heavy',
            patch_size=24,
            loss_fn=CombinedLossV2(tversky_alpha=0.2, tversky_beta=0.8),
            augmentation=HeavyAugmentation3D(),
            epochs=200,
            batch_size=64,
            n_patches=4,
            max_volumes=50  # Safe for 32GB RAM
        )
    except Exception as e:
        print(f"Experiment 4 failed: {e}")
        results['exp4_24patch_heavy'] = {'error': str(e)}

    # =========================================================================
    # EXPERIMENT 5: 36-patch balanced
    # Goal: Good context model for medium/large lesions
    # =========================================================================
    try:
        results['exp5_36patch_balanced'] = train_experiment(
            name='exp5_36patch_balanced',
            patch_size=36,
            loss_fn=CombinedLossV2(tversky_alpha=0.25, tversky_beta=0.75),
            augmentation=Augmentation3D(),
            epochs=150,
            batch_size=48,
            n_patches=3,
            max_volumes=50  # Safe for 32GB RAM
        )
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
        results['exp5_36patch_balanced'] = {'error': str(e)}

    # =========================================================================
    # EXPERIMENT 6: 6-patch ultra-micro (if time permits)
    # Goal: Catch sub-voxel lesions
    # =========================================================================
    try:
        results['exp6_6patch'] = train_experiment(
            name='exp6_6patch',
            patch_size=6,
            loss_fn=AggressiveFNLoss(),
            augmentation=Augmentation3D(),
            epochs=150,
            batch_size=256,
            n_patches=6,
            max_volumes=50  # Safe for 32GB RAM
        )
    except Exception as e:
        print(f"Experiment 6 failed: {e}")
        results['exp6_6patch'] = {'error': str(e)}

    # =========================================================================
    # FINAL: Evaluate combined ensemble
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL ENSEMBLE EVALUATION")
    print("=" * 70)

    # Combine new models with existing best models
    all_models = [
        ('exp1_8patch', 8),
        ('exp2_16patch_aggressive', 16),
        ('exp3_12patch_maxfn', 12),
        ('exp4_24patch_heavy', 24),
        ('exp5_36patch_balanced', 36),
        ('exp6_6patch', 6),
        ('improved_12patch', 12),  # Existing
        ('improved_24patch', 24),  # Existing
        ('improved_36patch', 36),  # Existing
    ]

    # Filter to existing models
    existing_models = [(n, p) for n, p in all_models if (model_dir / f"{n}_best.pth").exists()]

    if existing_models:
        model_names = [n for n, p in existing_models]
        patch_sizes = [p for n, p in existing_models]

        ensemble_result = evaluate_ensemble(model_names, patch_sizes, data_dir, model_dir, device)
        results['final_ensemble'] = ensemble_result

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================
    summary = {
        'start_time': datetime.now().isoformat(),
        'experiments': results,
        'models_trained': [k for k, v in results.items() if 'error' not in v]
    }

    with open(output_dir / 'overnight_experiments_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("OVERNIGHT EXPERIMENTS COMPLETE")
    print("=" * 70)

    print("\nResults Summary:")
    for name, result in results.items():
        if 'error' in result:
            print(f"  {name}: FAILED - {result['error'][:50]}")
        elif 'lesion_f1' in result:
            print(f"  {name}: L-F1={result['lesion_f1']*100:.1f}%, "
                  f"L-Sens={result['lesion_sensitivity']*100:.1f}%")
        elif 'lesion_f1' in result.get('best_metrics', {}):
            m = result['best_metrics']
            print(f"  {name}: L-F1={m['lesion_f1']*100:.1f}%, "
                  f"L-Sens={m['lesion_sensitivity']*100:.1f}%")

    print(f"\nSummary saved to: {output_dir / 'overnight_experiments_summary.json'}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
