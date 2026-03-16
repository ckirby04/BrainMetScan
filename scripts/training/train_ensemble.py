"""
Ensemble Model Training Script

Trains a learnable ensemble that combines multiple patch-size models
to maximize overall dice, sensitivity, specificity, and lesion-wise F1.

Based on training results:
- 12³: Best tiny dice (78.0%)
- 24³: Best overall (88.3%)
- 36³: Best sensitivity (90.0%)

Usage:
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --epochs 100
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import time
import json
import gc
from datetime import datetime

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D
from segmentation.leaderboard import Leaderboard


# =============================================================================
# ENSEMBLE ARCHITECTURE
# =============================================================================

class EnsembleModel(nn.Module):
    """
    Learnable ensemble that combines predictions from multiple models.

    Strategies:
    1. Learned weighted fusion
    2. Attention-based fusion
    3. Size-aware routing
    """

    def __init__(self, models, fusion_type='attention'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.fusion_type = fusion_type

        # Freeze base models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        if fusion_type == 'weighted':
            # Simple learned weights
            self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)

        elif fusion_type == 'attention':
            # Attention-based fusion - learns to weight based on input
            self.attention = nn.Sequential(
                nn.Conv3d(self.n_models, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(16, self.n_models, kernel_size=1),
                nn.Softmax(dim=1)
            )

        elif fusion_type == 'confidence':
            # Confidence-weighted fusion
            self.confidence_net = nn.Sequential(
                nn.Conv3d(self.n_models, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(16, self.n_models, kernel_size=1),
                nn.Sigmoid()
            )

        # Refinement layer
        self.refine = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, inputs_list):
        """
        Forward pass.

        Args:
            inputs_list: List of input tensors, one for each model's patch size
                        Each tensor shape: (B, C, H, W, D) where H,W,D match model's patch size

        Returns:
            Fused prediction (B, 1, H, W, D) at the largest patch size
        """
        # Get predictions from each model
        predictions = []
        with torch.no_grad():
            for model, x in zip(self.models, inputs_list):
                pred = model(x)
                predictions.append(pred)

        # Resize all predictions to match the largest
        target_size = predictions[-1].shape[2:]  # Use largest model's output size

        resized_preds = []
        for pred in predictions:
            if pred.shape[2:] != target_size:
                pred = F.interpolate(pred, size=target_size, mode='trilinear', align_corners=False)
            resized_preds.append(pred)

        # Stack predictions: (B, n_models, H, W, D)
        stacked = torch.cat(resized_preds, dim=1)

        # Fuse predictions
        if self.fusion_type == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            fused = sum(w * p for w, p in zip(weights, resized_preds))

        elif self.fusion_type == 'attention':
            attention_weights = self.attention(stacked)  # (B, n_models, H, W, D)
            fused = (stacked * attention_weights).sum(dim=1, keepdim=True)

        elif self.fusion_type == 'confidence':
            confidence = self.confidence_net(stacked)
            fused = (stacked * confidence).sum(dim=1, keepdim=True) / (confidence.sum(dim=1, keepdim=True) + 1e-6)

        else:
            # Simple average
            fused = stacked.mean(dim=1, keepdim=True)

        # Refinement
        fused = fused + self.refine(fused)

        return fused


class MultiScaleEnsemble(nn.Module):
    """
    Ensemble that processes input at multiple scales and combines results.
    Uses pre-trained models for each patch size.
    """

    def __init__(self, model_paths, patch_sizes, device='cuda'):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.device = device

        # Load pre-trained models
        self.models = nn.ModuleList()
        for path in model_paths:
            model = LightweightUNet3D(in_channels=4, out_channels=1, use_attention=True, use_residual=True)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.models.append(model)

        self.n_models = len(self.models)

        # Learnable fusion weights (one per model, can be size-dependent)
        self.fusion_weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)

        # Size-aware routing network
        self.size_router = nn.Sequential(
            nn.Linear(self.n_models, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_models),
            nn.Softmax(dim=-1)
        )

        # Confidence estimation per model
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(8, 1, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_models)
        ])

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 1, 3, padding=1)
        )

    def forward(self, x, return_individual=False):
        """
        Forward pass with multi-scale processing.

        Args:
            x: Input tensor (B, C, H, W, D) - should be at largest patch size
            return_individual: If True, also return individual model predictions
        """
        B = x.shape[0]
        target_size = x.shape[2:]

        # Get predictions from each model at its native scale
        predictions = []
        confidences = []

        for i, (model, patch_size) in enumerate(zip(self.models, self.patch_sizes)):
            # Resize input to model's patch size
            if patch_size != target_size[0]:
                x_resized = F.interpolate(x, size=(patch_size, patch_size, patch_size),
                                         mode='trilinear', align_corners=False)
            else:
                x_resized = x

            # Get prediction
            with torch.no_grad():
                pred = model(x_resized)

            # Resize prediction back to target size
            if patch_size != target_size[0]:
                pred = F.interpolate(pred, size=target_size, mode='trilinear', align_corners=False)

            # Compute confidence
            conf = self.confidence_heads[i](pred)

            predictions.append(pred)
            confidences.append(conf)

        # Stack predictions and confidences
        pred_stack = torch.stack(predictions, dim=1)  # (B, n_models, 1, H, W, D)
        conf_stack = torch.stack(confidences, dim=1)  # (B, n_models, 1, H, W, D)

        # Compute fusion weights
        base_weights = F.softmax(self.fusion_weights, dim=0)  # (n_models,)

        # Confidence-weighted fusion
        weighted_conf = conf_stack * base_weights.view(1, -1, 1, 1, 1, 1)
        weighted_conf = weighted_conf / (weighted_conf.sum(dim=1, keepdim=True) + 1e-6)

        # Fuse predictions
        fused = (pred_stack * weighted_conf).sum(dim=1)  # (B, 1, H, W, D)

        # Refinement
        fused = fused + 0.1 * self.refine(fused)

        if return_individual:
            return fused, predictions, confidences
        return fused


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * intersection + self.smooth) / (union + self.smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


class EnsembleLoss(nn.Module):
    """Combined loss optimized for ensemble training."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)  # Penalize FN more

    def forward(self, pred, target):
        return 0.4 * self.dice(pred, target) + \
               0.3 * self.focal(pred, target) + \
               0.3 * self.tversky(pred, target)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(pred, target, threshold=0.5):
    """Compute comprehensive metrics including lesion-wise F1."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    # Voxel-wise metrics
    tp = ((pred_binary == 1) & (target == 1)).sum().float()
    tn = ((pred_binary == 0) & (target == 0)).sum().float()
    fp = ((pred_binary == 1) & (target == 0)).sum().float()
    fn = ((pred_binary == 0) & (target == 1)).sum().float()

    sensitivity = (tp / (tp + fn + 1e-6)).item()
    specificity = (tn / (tn + fp + 1e-6)).item()
    precision = (tp / (tp + fp + 1e-6)).item()

    # Dice
    intersection = (pred_binary * target).sum()
    dice = ((2 * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)).item()

    # Voxel-wise F1
    voxel_f1 = (2 * precision * sensitivity) / (precision + sensitivity + 1e-6)

    return {
        'dice': dice,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'voxel_f1': voxel_f1
    }


def compute_lesion_wise_metrics(pred, target, threshold=0.5):
    """
    Compute lesion-wise detection metrics using connected components.

    Returns:
        lesion_sensitivity: Fraction of true lesions detected
        lesion_precision: Fraction of predicted lesions that are true
        lesion_f1: Harmonic mean of lesion sensitivity and precision
    """
    pred_np = (torch.sigmoid(pred) > threshold).cpu().numpy().astype(np.int32)
    target_np = target.cpu().numpy().astype(np.int32)

    total_true_lesions = 0
    total_pred_lesions = 0
    total_detected = 0
    total_true_positives = 0

    for b in range(pred_np.shape[0]):
        # Get connected components
        pred_labels, n_pred = ndimage.label(pred_np[b, 0])
        target_labels, n_target = ndimage.label(target_np[b, 0])

        total_pred_lesions += n_pred
        total_true_lesions += n_target

        # Check detection: a true lesion is detected if >50% overlap with any prediction
        for i in range(1, n_target + 1):
            true_mask = (target_labels == i)
            overlap = (pred_np[b, 0] * true_mask).sum()
            if overlap > 0.5 * true_mask.sum():
                total_detected += 1

        # Check true positives: a prediction is TP if >50% overlap with any true lesion
        for i in range(1, n_pred + 1):
            pred_mask = (pred_labels == i)
            overlap = (target_np[b, 0] * pred_mask).sum()
            if overlap > 0.5 * pred_mask.sum():
                total_true_positives += 1

    lesion_sensitivity = total_detected / (total_true_lesions + 1e-6)
    lesion_precision = total_true_positives / (total_pred_lesions + 1e-6)
    lesion_f1 = 2 * lesion_sensitivity * lesion_precision / (lesion_sensitivity + lesion_precision + 1e-6)

    return {
        'lesion_sensitivity': lesion_sensitivity,
        'lesion_precision': lesion_precision,
        'lesion_f1': lesion_f1,
        'n_true_lesions': total_true_lesions,
        'n_pred_lesions': total_pred_lesions,
        'n_detected': total_detected
    }


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

    return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_for_ensemble(data_dir, target_patch_size=64, patches_per_volume=5, chunk_size=30):
    """Load data at the largest patch size for ensemble training."""

    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=(target_patch_size, target_patch_size, target_patch_size),
        target_size=None,
        transform=None
    )

    print(f"Loading {len(ds)} volumes with {patches_per_volume} patches each...")

    all_images = []
    all_masks = []
    n = len(ds)
    num_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_idx, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        print(f"  Chunk {chunk_idx+1}/{num_chunks}...", end='\r')

        chunk_images = []
        chunk_masks = []

        for i in range(start, end):
            for _ in range(patches_per_volume):
                img, mask, _ = ds[i]
                chunk_images.append(img)
                chunk_masks.append(mask)

        all_images.append(torch.stack(chunk_images))
        all_masks.append(torch.stack(chunk_masks))

        gc.collect()

    print(f"  Loaded {n * patches_per_volume} patches.              ")

    images = torch.cat(all_images, dim=0)
    masks = torch.cat(all_masks, dim=0)

    return images, masks


# =============================================================================
# TRAINING
# =============================================================================

def train_ensemble(args):
    """Train the ensemble model."""

    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model paths - use the best models from training
    model_configs = [
        (12, model_dir / "improved_12patch_best.pth"),
        (24, model_dir / "improved_24patch_best.pth"),
        (36, model_dir / "improved_36patch_best.pth"),
    ]

    # Filter to existing models
    available_models = [(ps, p) for ps, p in model_configs if p.exists()]

    if len(available_models) < 2:
        print("Error: Need at least 2 trained models for ensemble")
        print("Available models:")
        for ps, p in model_configs:
            status = "✓" if p.exists() else "✗"
            print(f"  {status} {ps}³: {p}")
        return

    patch_sizes = [ps for ps, _ in available_models]
    model_paths = [p for _, p in available_models]
    target_patch_size = max(patch_sizes)

    print(f"\nEnsemble configuration:")
    print(f"  Models: {patch_sizes}")
    print(f"  Target patch size: {target_patch_size}³")

    # Load data
    print(f"\nLoading training data...")
    images, masks = load_data_for_ensemble(
        data_dir,
        target_patch_size=target_patch_size,
        patches_per_volume=args.patches_per_volume
    )
    print(f"Data shape: {images.shape}")

    # Train/val split
    n = len(images)
    train_size = int(0.85 * n)
    val_size = n - train_size

    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = TensorDataset(images[train_idx], masks[train_idx])
    val_dataset = TensorDataset(images[val_idx], masks[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create ensemble
    print(f"\nCreating ensemble model...")
    ensemble = MultiScaleEnsemble(model_paths, patch_sizes, device=device).to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, ensemble.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    criterion = EnsembleLoss()

    # Training
    model_path = model_dir / "ensemble_best.pth"
    state_path = model_dir / "ensemble_state.json"

    print(f"\n{'='*70}")
    print(f"ENSEMBLE TRAINING")
    print(f"Models: {patch_sizes}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*70}\n")

    best_val_dice = 0
    best_metrics = {}
    train_start = time.time()

    leaderboard = Leaderboard(str(model_dir / "leaderboard.json"))

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        ensemble.train()
        train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs} [Train]", leave=False, ncols=100)
        for img, mask in train_pbar:
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            pred = ensemble(img)
            loss = criterion(pred, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Validate
        ensemble.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{args.epochs} [Val]  ", leave=False, ncols=100)
        with torch.no_grad():
            for img, mask in val_pbar:
                img, mask = img.to(device), mask.to(device)
                pred = ensemble(img)
                val_loss += criterion(pred, mask).item()
                all_preds.append(pred.cpu())
                all_targets.append(mask.cpu())

        val_loss /= len(val_loader)

        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        voxel_metrics = compute_metrics(all_preds, all_targets)
        lesion_metrics = compute_lesion_wise_metrics(all_preds, all_targets)
        size_dice = compute_size_stratified_dice(all_preds, all_targets)

        scheduler.step()

        # Combined metrics
        metrics = {
            **voxel_metrics,
            **lesion_metrics,
            **{f'{k}_dice': v for k, v in size_dice.items() if v is not None}
        }

        # Save best
        is_best = voxel_metrics['dice'] > best_val_dice
        if is_best:
            best_val_dice = voxel_metrics['dice']
            best_metrics = metrics.copy()

            torch.save({
                'epoch': epoch,
                'model_state_dict': ensemble.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'patch_sizes': patch_sizes,
                'model_paths': [str(p) for p in model_paths]
            }, model_path)

        # Update leaderboard
        leaderboard.update(
            model_name="ensemble",
            patch_size=0,  # 0 indicates ensemble
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_dice=voxel_metrics['dice'],
            tiny_dice=size_dice.get('tiny'),
            small_dice=size_dice.get('small'),
            medium_dice=size_dice.get('medium'),
            large_dice=size_dice.get('large'),
            sensitivity=voxel_metrics['sensitivity'],
            specificity=voxel_metrics['specificity'],
            model_path=str(model_path) if is_best else None,
            is_best=is_best
        )

        # Save state
        with open(state_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'epochs_total': args.epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics,
                'best_metrics': best_metrics,
                'patch_sizes': patch_sizes,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        # Print progress
        epoch_time = time.time() - epoch_start
        best_marker = " *" if is_best else ""
        tiny_str = f"tiny={size_dice.get('tiny', 0):.3f}" if size_dice.get('tiny') else "tiny=N/A"

        print(f"Epoch {epoch:3d}/{args.epochs}: dice={voxel_metrics['dice']:.3f}, {tiny_str}, "
              f"sens={voxel_metrics['sensitivity']:.3f}, lesion_f1={lesion_metrics['lesion_f1']:.3f} "
              f"({epoch_time:.1f}s){best_marker}")

    # Final summary
    total_time = (time.time() - train_start) / 60

    print(f"\n{'='*70}")
    print(f"ENSEMBLE TRAINING COMPLETE")
    print(f"Total time: {total_time:.1f} min")
    print(f"{'='*70}")
    print(f"\nBest Metrics:")
    print(f"  Overall Dice:      {best_metrics.get('dice', 0)*100:.1f}%")
    print(f"  Tiny Dice:         {best_metrics.get('tiny_dice', 0)*100:.1f}%" if best_metrics.get('tiny_dice') else "  Tiny Dice:         N/A")
    print(f"  Small Dice:        {best_metrics.get('small_dice', 0)*100:.1f}%" if best_metrics.get('small_dice') else "  Small Dice:        N/A")
    print(f"  Medium Dice:       {best_metrics.get('medium_dice', 0)*100:.1f}%" if best_metrics.get('medium_dice') else "  Medium Dice:       N/A")
    print(f"  Large Dice:        {best_metrics.get('large_dice', 0)*100:.1f}%" if best_metrics.get('large_dice') else "  Large Dice:        N/A")
    print(f"  Sensitivity:       {best_metrics.get('sensitivity', 0)*100:.1f}%")
    print(f"  Specificity:       {best_metrics.get('specificity', 0)*100:.1f}%")
    print(f"  Lesion F1:         {best_metrics.get('lesion_f1', 0)*100:.1f}%")
    print(f"  Lesion Sensitivity:{best_metrics.get('lesion_sensitivity', 0)*100:.1f}%")
    print(f"  Lesion Precision:  {best_metrics.get('lesion_precision', 0)*100:.1f}%")
    print(f"\nModel saved: {model_path}")

    # Show leaderboard
    print("\n")
    leaderboard.print_summary()

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description='Train ensemble model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patches-per-volume', type=int, default=3, help='Patches per volume')
    args = parser.parse_args()

    train_ensemble(args)


if __name__ == '__main__':
    main()
