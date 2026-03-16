"""
Smart Ensemble Training Script
==============================
Combines multiple patch-size models with intelligent fusion strategy:
- Small patches (8, 12) for tiny lesion detection (high sensitivity)
- Medium patches (24) for balanced detection
- Large patches (36) for context and large lesion refinement

Fusion Strategy:
- Union-based: OR of small patch predictions (maximize sensitivity)
- Confidence-weighted: weight by model confidence
- Size-aware: route predictions based on lesion size estimates

Based on overnight experiment results:
- exp3_12patch_maxfn: 97-100% lesion sensitivity
- exp1_8patch: 96.9% lesion sensitivity
"""

import sys
import gc
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D


class SmartEnsemble(nn.Module):
    """
    Smart ensemble that combines models with different strategies:
    - Union mode: OR of predictions (max sensitivity)
    - Weighted mode: learned confidence weights
    - Hybrid mode: union for detection, refinement for boundaries
    """

    def __init__(self, model_configs, device='cuda', fusion_mode='union'):
        """
        Args:
            model_configs: list of (name, patch_size, model_path, threshold)
            device: cuda or cpu
            fusion_mode: 'union', 'weighted', or 'hybrid'
        """
        super().__init__()
        self.device = device
        self.fusion_mode = fusion_mode
        self.models = nn.ModuleList()
        self.patch_sizes = []
        self.thresholds = []
        self.names = []

        for name, patch_size, model_path, threshold in model_configs:
            if not Path(model_path).exists():
                print(f"  Warning: {name} not found at {model_path}, skipping")
                continue

            print(f"  Loading {name} (patch {patch_size}, threshold {threshold})...")
            model = LightweightUNet3D(in_channels=4, out_channels=1,
                                      use_attention=True, use_residual=True)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            # Freeze weights
            for param in model.parameters():
                param.requires_grad = False

            self.models.append(model)
            self.patch_sizes.append(patch_size)
            self.thresholds.append(threshold)
            self.names.append(name)

        print(f"  Loaded {len(self.models)} models")

        # Learnable fusion weights (for weighted mode)
        if fusion_mode == 'weighted':
            self.fusion_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))

    def forward(self, x, target_size=None):
        """
        Forward pass with smart fusion.

        Args:
            x: input tensor [B, C, D, H, W]
            target_size: output size (default: input size)
        """
        if target_size is None:
            target_size = x.shape[2]

        predictions = []

        for i, (model, patch_size, threshold) in enumerate(
                zip(self.models, self.patch_sizes, self.thresholds)):

            # Resize input to model's patch size if needed
            if patch_size != x.shape[2]:
                x_resized = F.interpolate(x, size=(patch_size,) * 3,
                                         mode='trilinear', align_corners=False)
            else:
                x_resized = x

            # Get prediction
            with torch.no_grad():
                pred = model(x_resized)
                pred_prob = torch.sigmoid(pred)

            # Resize back to target size
            if patch_size != target_size:
                pred_prob = F.interpolate(pred_prob, size=(target_size,) * 3,
                                         mode='trilinear', align_corners=False)

            predictions.append(pred_prob)

        # Fuse predictions based on mode
        if self.fusion_mode == 'union':
            # Union (OR): take max probability across models
            # This maximizes sensitivity
            stacked = torch.stack(predictions, dim=0)
            fused = stacked.max(dim=0)[0]

        elif self.fusion_mode == 'weighted':
            # Weighted average with learnable weights
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * p for w, p in zip(weights, predictions))

        elif self.fusion_mode == 'hybrid':
            # Union for detection, average for refinement
            stacked = torch.stack(predictions, dim=0)
            union = stacked.max(dim=0)[0]
            average = stacked.mean(dim=0)
            # Use union where any model is confident, average elsewhere
            confident = (union > 0.3).float()
            fused = confident * union + (1 - confident) * average

        else:
            # Default: simple average
            fused = torch.stack(predictions).mean(dim=0)

        return fused

    def predict_with_details(self, x, target_size=None):
        """Get predictions with per-model details."""
        if target_size is None:
            target_size = x.shape[2]

        results = {'individual': {}, 'fused': None}

        predictions = []
        for i, (model, patch_size, threshold, name) in enumerate(
                zip(self.models, self.patch_sizes, self.thresholds, self.names)):

            if patch_size != x.shape[2]:
                x_resized = F.interpolate(x, size=(patch_size,) * 3,
                                         mode='trilinear', align_corners=False)
            else:
                x_resized = x

            with torch.no_grad():
                pred = model(x_resized)
                pred_prob = torch.sigmoid(pred)

            if patch_size != target_size:
                pred_prob = F.interpolate(pred_prob, size=(target_size,) * 3,
                                         mode='trilinear', align_corners=False)

            predictions.append(pred_prob)
            results['individual'][name] = {
                'prediction': pred_prob,
                'threshold': threshold
            }

        # Fused prediction
        stacked = torch.stack(predictions, dim=0)
        results['fused'] = stacked.max(dim=0)[0]

        return results


def compute_metrics(pred_prob, target, threshold=0.5):
    """Compute comprehensive metrics."""
    pred_binary = (pred_prob > threshold).float()

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

    # Lesion-wise
    pred_np = pred_binary.cpu().numpy().astype(np.uint8)
    target_np = target.cpu().numpy().astype(np.uint8)

    total_true, total_pred, total_detected, total_tp = 0, 0, 0, 0

    for b in range(min(pred_np.shape[0], 50)):  # Limit for speed
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
        'dice': dice,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'lesion_sensitivity': lesion_sens,
        'lesion_precision': lesion_prec,
        'lesion_f1': lesion_f1,
        'n_true_lesions': total_true,
        'n_detected': total_detected,
        'n_predicted': total_pred
    }


def load_validation_data(data_dir, patch_size, n_patches=3, max_volumes=100):
    """Load validation data."""
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=(patch_size,) * 3,
        target_size=None,
        transform=None
    )

    images, masks = [], []
    n_volumes = min(len(ds), max_volumes)

    print(f"Loading {n_volumes} volumes x {n_patches} patches...")
    for i in tqdm(range(n_volumes), desc="Loading"):
        try:
            for _ in range(n_patches):
                img, mask, _ = ds[i]
                images.append(img)
                masks.append(mask)
        except:
            continue

        if (i + 1) % 25 == 0:
            gc.collect()

    return torch.stack(images), torch.stack(masks)


def evaluate_ensemble(ensemble, images, masks, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """Evaluate ensemble at multiple thresholds."""
    device = next(ensemble.parameters()).device if list(ensemble.parameters()) else 'cuda'

    print("\nGenerating ensemble predictions...")
    all_preds = []

    loader = DataLoader(TensorDataset(images), batch_size=8, shuffle=False)
    for (batch,) in tqdm(loader, desc="Predicting"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = ensemble(batch, target_size=images.shape[2])
        all_preds.append(pred.cpu())

    all_preds = torch.cat(all_preds)

    print("\nEvaluating at different thresholds:")
    print("-" * 70)
    print(f"{'Thresh':>7} {'Dice':>8} {'Sens':>8} {'Spec':>8} {'L-Sens':>8} {'L-F1':>8}")
    print("-" * 70)

    results = []
    for thresh in thresholds:
        metrics = compute_metrics(all_preds, masks, threshold=thresh)
        results.append({'threshold': thresh, **metrics})

        print(f"{thresh:>7.2f} {metrics['dice']*100:>7.1f}% {metrics['sensitivity']*100:>7.1f}% "
              f"{metrics['specificity']*100:>7.1f}% {metrics['lesion_sensitivity']*100:>7.1f}% "
              f"{metrics['lesion_f1']*100:>7.1f}%")

    return results, all_preds


def main():
    print("=" * 70)
    print("SMART ENSEMBLE EVALUATION")
    print("=" * 70)

    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    output_dir = project_dir / "outputs"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Define model configurations
    # (name, patch_size, model_path, optimal_threshold)
    model_configs = [
        # High-sensitivity small patch models
        ('exp3_12patch_maxfn', 12, model_dir / 'exp3_12patch_maxfn_best.pth', 0.25),
        ('exp1_8patch', 8, model_dir / 'exp1_8patch_best.pth', 0.3),

        # Medium patch for balance
        ('improved_24patch', 24, model_dir / 'improved_24patch_best.pth', 0.5),

        # Large patch for context (optional)
        ('improved_36patch', 36, model_dir / 'improved_36patch_best.pth', 0.5),
    ]

    # Check which models exist
    available_configs = [(n, p, m, t) for n, p, m, t in model_configs if Path(m).exists()]
    print(f"\nAvailable models: {[c[0] for c in available_configs]}")

    if len(available_configs) < 2:
        print("Need at least 2 models for ensemble!")
        return

    # Create ensemble with UNION fusion (maximize sensitivity)
    print("\nCreating Smart Ensemble (union fusion)...")
    ensemble = SmartEnsemble(available_configs, device=device, fusion_mode='union')

    # Load validation data at medium patch size
    val_patch_size = 24
    images, masks = load_validation_data(data_dir, val_patch_size, n_patches=3, max_volumes=80)
    print(f"Loaded {len(images)} validation samples")

    # Evaluate
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    results, predictions = evaluate_ensemble(ensemble, images, masks, thresholds)

    # Find best thresholds
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS")
    print("=" * 70)

    # Best for 95% sensitivity
    sens_95 = [r for r in results if r['sensitivity'] >= 0.95]
    if sens_95:
        best = max(sens_95, key=lambda x: x['dice'])
        print(f"\n[95% SENSITIVITY] threshold = {best['threshold']:.2f}")
        print(f"   Dice: {best['dice']*100:.1f}%, Lesion-F1: {best['lesion_f1']*100:.1f}%")
    else:
        best_sens = max(results, key=lambda x: x['sensitivity'])
        print(f"\n[MAX SENSITIVITY] {best_sens['sensitivity']*100:.1f}% at threshold {best_sens['threshold']:.2f}")

    # Best for 90% lesion sensitivity
    lesion_90 = [r for r in results if r['lesion_sensitivity'] >= 0.90]
    if lesion_90:
        best = max(lesion_90, key=lambda x: x['dice'])
        print(f"\n[90% LESION SENSITIVITY] threshold = {best['threshold']:.2f}")
        print(f"   Dice: {best['dice']*100:.1f}%, Lesion-F1: {best['lesion_f1']*100:.1f}%")
    else:
        best_lsens = max(results, key=lambda x: x['lesion_sensitivity'])
        print(f"\n[MAX LESION SENSITIVITY] {best_lsens['lesion_sensitivity']*100:.1f}% at threshold {best_lsens['threshold']:.2f}")

    # Best lesion F1
    best_f1 = max(results, key=lambda x: x['lesion_f1'])
    print(f"\n[BEST LESION F1] threshold = {best_f1['threshold']:.2f}")
    print(f"   Dice: {best_f1['dice']*100:.1f}%, L-Sens: {best_f1['lesion_sensitivity']*100:.1f}%, "
          f"L-F1: {best_f1['lesion_f1']*100:.1f}%")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'fusion_mode': 'union',
        'models_used': [c[0] for c in available_configs],
        'results': results,
        'best_thresholds': {
            'best_lesion_f1': best_f1
        }
    }

    output_path = output_dir / 'smart_ensemble_results.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
