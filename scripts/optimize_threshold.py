"""
Threshold Optimization Script

Finds optimal detection threshold to maximize sensitivity while maintaining
acceptable specificity. Tests both ensemble and individual models.

Usage:
    python scripts/optimize_threshold.py
    python scripts/optimize_threshold.py --model ensemble
    python scripts/optimize_threshold.py --model improved_12patch
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import json
import gc
import matplotlib.pyplot as plt

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D


def load_data(data_dir, patch_size, n_patches=3, max_samples=150):
    """Load validation data (memory-efficient version).

    Args:
        data_dir: Path to data
        patch_size: Size of patches to extract
        n_patches: Patches per volume
        max_samples: Maximum total samples to load (prevents OOM)
    """
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=(patch_size, patch_size, patch_size),
        target_size=None,
        transform=None
    )

    images, masks = [], []
    n_volumes = min(len(ds), max_samples // n_patches)

    print(f"Loading {n_volumes} volumes × {n_patches} patches = {n_volumes * n_patches} samples")

    for i in tqdm(range(n_volumes), desc="Loading data"):
        for _ in range(n_patches):
            img, mask, _ = ds[i]
            images.append(img)
            masks.append(mask)

        # Garbage collect every 50 volumes
        if i > 0 and i % 50 == 0:
            gc.collect()

    return torch.stack(images), torch.stack(masks)


def compute_metrics_at_threshold(pred_probs, targets, threshold, batch_size=32):
    """Compute all metrics at a specific threshold (memory-efficient)."""
    n_samples = pred_probs.shape[0]

    # Compute voxel-wise metrics incrementally
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0
    intersection_total, pred_sum, target_sum = 0, 0, 0

    total_true_lesions = 0
    total_pred_lesions = 0
    total_detected = 0
    total_true_positives = 0

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        pred_batch = pred_probs[start_idx:end_idx]
        target_batch = targets[start_idx:end_idx]

        pred_binary = (pred_batch > threshold).float()

        # Voxel-wise stats
        tp_total += ((pred_binary == 1) & (target_batch == 1)).sum().item()
        tn_total += ((pred_binary == 0) & (target_batch == 0)).sum().item()
        fp_total += ((pred_binary == 1) & (target_batch == 0)).sum().item()
        fn_total += ((pred_binary == 0) & (target_batch == 1)).sum().item()
        intersection_total += (pred_binary * target_batch).sum().item()
        pred_sum += pred_binary.sum().item()
        target_sum += target_batch.sum().item()

        # Lesion-wise (process one sample at a time to save memory)
        pred_np = pred_binary.cpu().numpy().astype(np.uint8)
        target_np = target_batch.cpu().numpy().astype(np.uint8)

        for b in range(pred_np.shape[0]):
            pred_labels, n_pred = ndimage.label(pred_np[b, 0])
            target_labels, n_target = ndimage.label(target_np[b, 0])

            total_pred_lesions += n_pred
            total_true_lesions += n_target

            for i in range(1, n_target + 1):
                true_mask = (target_labels == i)
                overlap = (pred_np[b, 0] * true_mask).sum()
                if overlap > 0.3 * true_mask.sum():
                    total_detected += 1

            for i in range(1, n_pred + 1):
                pred_mask = (pred_labels == i)
                overlap = (target_np[b, 0] * pred_mask).sum()
                if overlap > 0.3 * pred_mask.sum():
                    total_true_positives += 1

        del pred_binary, pred_np, target_np

    sensitivity = tp_total / (tp_total + fn_total + 1e-6)
    specificity = tn_total / (tn_total + fp_total + 1e-6)
    precision = tp_total / (tp_total + fp_total + 1e-6)
    dice = (2 * intersection_total + 1e-6) / (pred_sum + target_sum + 1e-6)

    lesion_sens = total_detected / (total_true_lesions + 1e-6)
    lesion_prec = total_true_positives / (total_pred_lesions + 1e-6)
    lesion_f1 = 2 * lesion_sens * lesion_prec / (lesion_sens + lesion_prec + 1e-6)

    return {
        'threshold': threshold,
        'dice': dice,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'lesion_sensitivity': lesion_sens,
        'lesion_precision': lesion_prec,
        'lesion_f1': lesion_f1,
        'n_true_lesions': total_true_lesions,
        'n_detected': total_detected,
        'n_predicted': total_pred_lesions
    }


def get_predictions(model, images, device, batch_size=8):
    """Get model predictions."""
    model.eval()
    all_preds = []

    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Getting predictions"):
            batch = batch.to(device)
            pred = model(batch)
            pred_probs = torch.sigmoid(pred)
            all_preds.append(pred_probs.cpu())

    return torch.cat(all_preds)


def get_ensemble_predictions(model_paths, patch_sizes, images, device, batch_size=4):
    """Get ensemble predictions by averaging multiple models."""
    all_preds = []

    for model_path, patch_size in zip(model_paths, patch_sizes):
        print(f"  Loading {patch_size}³ model...")
        model = LightweightUNet3D(in_channels=4, out_channels=1, use_attention=True, use_residual=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        preds = []
        loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (batch,) in tqdm(loader, desc=f"  {patch_size}³ predictions", leave=False):
                batch = batch.to(device)

                # Resize to model's patch size if needed
                target_size = images.shape[2]
                if patch_size != target_size:
                    batch_resized = F.interpolate(batch, size=(patch_size, patch_size, patch_size),
                                                  mode='trilinear', align_corners=False)
                    pred = model(batch_resized)
                    pred = F.interpolate(pred, size=(target_size, target_size, target_size),
                                        mode='trilinear', align_corners=False)
                else:
                    pred = model(batch)

                pred_probs = torch.sigmoid(pred)
                preds.append(pred_probs.cpu())

        all_preds.append(torch.cat(preds))

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Average predictions
    stacked = torch.stack(all_preds)
    return stacked.mean(dim=0)


def optimize_threshold(pred_probs, targets, thresholds):
    """Test multiple thresholds and find optimal."""
    results = []

    print("\nTesting thresholds...")
    for thresh in tqdm(thresholds):
        metrics = compute_metrics_at_threshold(pred_probs, targets, thresh)
        results.append(metrics)

    return results


def find_optimal_thresholds(results):
    """Find optimal thresholds for different objectives."""
    # Best for sensitivity >= 95%
    sens_95 = [r for r in results if r['sensitivity'] >= 0.95]
    best_sens_95 = max(sens_95, key=lambda x: x['dice']) if sens_95 else None

    # Best for lesion sensitivity >= 90%
    lesion_90 = [r for r in results if r['lesion_sensitivity'] >= 0.90]
    best_lesion_90 = max(lesion_90, key=lambda x: x['dice']) if lesion_90 else None

    # Best dice overall
    best_dice = max(results, key=lambda x: x['dice'])

    # Best lesion F1
    best_lesion_f1 = max(results, key=lambda x: x['lesion_f1'])

    # Best balanced (maximize sensitivity + specificity + lesion_f1)
    best_balanced = max(results, key=lambda x: x['sensitivity'] + x['specificity'] + x['lesion_f1'])

    return {
        'best_for_95_sensitivity': best_sens_95,
        'best_for_90_lesion_sensitivity': best_lesion_90,
        'best_dice': best_dice,
        'best_lesion_f1': best_lesion_f1,
        'best_balanced': best_balanced
    }


def plot_threshold_curves(results, output_path):
    """Plot threshold vs metrics curves."""
    thresholds = [r['threshold'] for r in results]
    dice = [r['dice'] for r in results]
    sensitivity = [r['sensitivity'] for r in results]
    specificity = [r['specificity'] for r in results]
    lesion_sens = [r['lesion_sensitivity'] for r in results]
    lesion_f1 = [r['lesion_f1'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Threshold Optimization Results', fontsize=14, fontweight='bold')

    # Plot 1: Sensitivity vs Specificity
    ax1 = axes[0, 0]
    ax1.plot(thresholds, sensitivity, 'b-', linewidth=2, label='Sensitivity')
    ax1.plot(thresholds, specificity, 'g-', linewidth=2, label='Specificity')
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Target')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Voxel-wise Sensitivity vs Specificity')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)

    # Plot 2: Dice vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(thresholds, dice, 'purple', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Dice Score vs Threshold')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1)

    # Plot 3: Lesion-wise metrics
    ax3 = axes[1, 0]
    ax3.plot(thresholds, lesion_sens, 'b-', linewidth=2, label='Lesion Sensitivity')
    ax3.plot(thresholds, lesion_f1, 'orange', linewidth=2, label='Lesion F1')
    ax3.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='90% Target')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title('Lesion-wise Metrics')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.05)

    # Plot 4: ROC-like curve
    ax4 = axes[1, 1]
    ax4.plot([1-s for s in specificity], sensitivity, 'b-', linewidth=2)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax4.set_xlabel('False Positive Rate (1 - Specificity)')
    ax4.set_ylabel('True Positive Rate (Sensitivity)')
    ax4.set_title('ROC Curve')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize detection threshold')
    parser.add_argument('--model', type=str, default='ensemble',
                        help='Model to optimize: ensemble, improved_12patch, improved_24patch, etc.')
    parser.add_argument('--patch-size', type=int, default=36,
                        help='Patch size for data loading (use largest model patch size)')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data (limited to prevent OOM)
    print(f"\nLoading validation data (patch size: {args.patch_size}³)...")
    val_images, val_masks = load_data(data_dir, args.patch_size, n_patches=3, max_samples=300)
    print(f"Loaded {len(val_images)} validation samples")

    # Force garbage collection before predictions
    gc.collect()
    torch.cuda.empty_cache()

    # Get predictions
    print(f"\nGetting predictions for: {args.model}")

    if args.model == 'ensemble':
        # Use multiple models
        model_configs = [
            (12, model_dir / "improved_12patch_best.pth"),
            (24, model_dir / "improved_24patch_best.pth"),
            (36, model_dir / "improved_36patch_best.pth"),
        ]
        available = [(ps, p) for ps, p in model_configs if p.exists()]

        if len(available) < 2:
            print("Not enough models for ensemble. Using best available single model.")
            args.model = 'improved_24patch'
        else:
            patch_sizes = [ps for ps, _ in available]
            model_paths = [p for _, p in available]
            pred_probs = get_ensemble_predictions(model_paths, patch_sizes, val_images, device)

    if args.model != 'ensemble':
        # Single model
        model_path = model_dir / f"{args.model}_best.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return

        model = LightweightUNet3D(in_channels=4, out_channels=1, use_attention=True, use_residual=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        pred_probs = get_predictions(model, val_images, device)

    # Test thresholds
    thresholds = np.arange(0.05, 0.95, 0.05).tolist()
    results = optimize_threshold(pred_probs, val_masks, thresholds)

    # Find optimal thresholds
    optimal = find_optimal_thresholds(results)

    # Print results
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*70)

    print("\n[ALL THRESHOLDS]")
    print("-"*70)
    print(f"{'Thresh':>7} {'Dice':>8} {'Sens':>8} {'Spec':>8} {'L-Sens':>8} {'L-F1':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['threshold']:>7.2f} {r['dice']*100:>7.1f}% {r['sensitivity']*100:>7.1f}% "
              f"{r['specificity']*100:>7.1f}% {r['lesion_sensitivity']*100:>7.1f}% {r['lesion_f1']*100:>7.1f}%")

    print("\n" + "="*70)
    print("OPTIMAL THRESHOLDS")
    print("="*70)

    if optimal['best_for_95_sensitivity']:
        r = optimal['best_for_95_sensitivity']
        print(f"\n[TARGET] For >=95% Sensitivity: threshold = {r['threshold']:.2f}")
        print(f"   Dice: {r['dice']*100:.1f}%, Sens: {r['sensitivity']*100:.1f}%, "
              f"Spec: {r['specificity']*100:.1f}%, Lesion-F1: {r['lesion_f1']*100:.1f}%")
    else:
        print("\n[WARNING]  Could not achieve 95% sensitivity at any threshold")
        # Find threshold for max sensitivity
        best_sens = max(results, key=lambda x: x['sensitivity'])
        print(f"   Max sensitivity: {best_sens['sensitivity']*100:.1f}% at threshold {best_sens['threshold']:.2f}")

    if optimal['best_for_90_lesion_sensitivity']:
        r = optimal['best_for_90_lesion_sensitivity']
        print(f"\n[LESION] For >=90% Lesion Sensitivity: threshold = {r['threshold']:.2f}")
        print(f"   Dice: {r['dice']*100:.1f}%, Sens: {r['sensitivity']*100:.1f}%, "
              f"Spec: {r['specificity']*100:.1f}%, Lesion-F1: {r['lesion_f1']*100:.1f}%")
    else:
        print("\n[WARNING]  Could not achieve 90% lesion sensitivity at any threshold")
        best_lsens = max(results, key=lambda x: x['lesion_sensitivity'])
        print(f"   Max lesion sensitivity: {best_lsens['lesion_sensitivity']*100:.1f}% at threshold {best_lsens['threshold']:.2f}")

    r = optimal['best_dice']
    print(f"\n[BEST] Best Dice: threshold = {r['threshold']:.2f}")
    print(f"   Dice: {r['dice']*100:.1f}%, Sens: {r['sensitivity']*100:.1f}%, "
          f"Spec: {r['specificity']*100:.1f}%, Lesion-F1: {r['lesion_f1']*100:.1f}%")

    r = optimal['best_lesion_f1']
    print(f"\n[WINNER] Best Lesion F1: threshold = {r['threshold']:.2f}")
    print(f"   Dice: {r['dice']*100:.1f}%, Sens: {r['sensitivity']*100:.1f}%, "
          f"Spec: {r['specificity']*100:.1f}%, Lesion-F1: {r['lesion_f1']*100:.1f}%")

    r = optimal['best_balanced']
    print(f"\n[BALANCED]  Best Balanced: threshold = {r['threshold']:.2f}")
    print(f"   Dice: {r['dice']*100:.1f}%, Sens: {r['sensitivity']*100:.1f}%, "
          f"Spec: {r['specificity']*100:.1f}%, Lesion-F1: {r['lesion_f1']*100:.1f}%")

    # Save results
    results_path = output_dir / f"threshold_optimization_{args.model}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'model': args.model,
            'all_results': results,
            'optimal': {k: v for k, v in optimal.items() if v is not None}
        }, f, indent=2)
    print(f"\n[SAVED] Results saved: {results_path}")

    # Plot
    plot_path = output_dir / f"threshold_curves_{args.model}.png"
    plot_threshold_curves(results, plot_path)

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
