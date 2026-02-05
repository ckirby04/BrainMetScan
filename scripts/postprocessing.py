"""
Post-Processing Script for Brain Metastasis Segmentation
=========================================================
Tests different post-processing strategies to reduce false positives
while maintaining high sensitivity.

Strategies:
1. Morphological opening - removes small noise
2. Size filtering - removes tiny connected components
3. Morphological closing - fills small holes
4. Confidence thresholding - only keep high-confidence regions
5. Combined pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing, binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import json
from datetime import datetime

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D


def load_ensemble_predictions(data_dir, model_dir, device, max_volumes=80, patch_size=24):
    """Load data and get ensemble predictions."""

    # Load models
    models = []
    patch_sizes = []
    names = ['exp3_12patch_maxfn', 'exp1_8patch', 'improved_24patch', 'improved_36patch']
    patches = [12, 8, 24, 36]

    print("Loading models...")
    for name, ps in zip(names, patches):
        path = model_dir / f'{name}_best.pth'
        if path.exists():
            model = LightweightUNet3D(in_channels=4, out_channels=1, use_attention=True, use_residual=True)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
            patch_sizes.append(ps)
            print(f'  Loaded {name}')

    # Load data
    print("Loading validation data...")
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=(patch_size,) * 3,
        target_size=None,
        transform=None
    )

    images, masks = [], []
    for i in tqdm(range(min(max_volumes, len(ds))), desc='Loading'):
        for _ in range(3):
            img, mask, _ = ds[i]
            images.append(img)
            masks.append(mask)
        if (i + 1) % 25 == 0:
            gc.collect()

    images = torch.stack(images)
    masks = torch.stack(masks)
    print(f'Loaded {len(images)} samples')

    # Get ensemble predictions
    print("Getting ensemble predictions...")
    all_preds = []
    loader = DataLoader(TensorDataset(images), batch_size=8, shuffle=False)

    for (batch,) in tqdm(loader, desc='Predicting'):
        batch = batch.to(device)
        preds = []
        for model, ps in zip(models, patch_sizes):
            if ps != patch_size:
                x = F.interpolate(batch, size=(ps,) * 3, mode='trilinear', align_corners=False)
                with torch.no_grad():
                    p = torch.sigmoid(model(x))
                p = F.interpolate(p, size=(patch_size,) * 3, mode='trilinear', align_corners=False)
            else:
                with torch.no_grad():
                    p = torch.sigmoid(model(batch))
            preds.append(p)
        ensemble_pred = torch.stack(preds).max(dim=0)[0]
        all_preds.append(ensemble_pred.cpu())

    return torch.cat(all_preds), masks


def compute_metrics(pred_binary, target):
    """Compute metrics from binary predictions."""
    pred = pred_binary.float()
    tgt = target.float()

    TP = ((pred == 1) & (tgt == 1)).sum().item()
    TN = ((pred == 0) & (tgt == 0)).sum().item()
    FP = ((pred == 1) & (tgt == 0)).sum().item()
    FN = ((pred == 0) & (tgt == 1)).sum().item()

    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-6)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'dice': dice,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }


def compute_lesion_metrics(pred_binary, target, max_samples=50):
    """Compute lesion-wise metrics."""
    pred_np = pred_binary.numpy().astype(np.uint8)
    target_np = target.numpy().astype(np.uint8)

    total_true, total_pred, total_detected, total_tp = 0, 0, 0, 0

    for b in range(min(pred_np.shape[0], max_samples)):
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
        'n_true': total_true,
        'n_detected': total_detected,
        'n_predicted': total_pred
    }


# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================

def remove_small_components(pred_binary, min_size=10):
    """Remove connected components smaller than min_size voxels."""
    pred_np = pred_binary.numpy()
    result = np.zeros_like(pred_np)

    for b in range(pred_np.shape[0]):
        labeled, n_components = ndimage.label(pred_np[b, 0])
        for i in range(1, n_components + 1):
            component = (labeled == i)
            if component.sum() >= min_size:
                result[b, 0][component] = 1

    return torch.from_numpy(result)


def morphological_opening(pred_binary, structure_size=2):
    """Apply morphological opening (erosion then dilation)."""
    pred_np = pred_binary.numpy()
    structure = ndimage.generate_binary_structure(3, 1)

    # Dilate structure for larger effect
    if structure_size > 1:
        structure = ndimage.iterate_structure(structure, structure_size)

    result = np.zeros_like(pred_np)
    for b in range(pred_np.shape[0]):
        result[b, 0] = binary_opening(pred_np[b, 0], structure=structure)

    return torch.from_numpy(result.astype(np.float32))


def morphological_closing(pred_binary, structure_size=2):
    """Apply morphological closing (dilation then erosion)."""
    pred_np = pred_binary.numpy()
    structure = ndimage.generate_binary_structure(3, 1)

    if structure_size > 1:
        structure = ndimage.iterate_structure(structure, structure_size)

    result = np.zeros_like(pred_np)
    for b in range(pred_np.shape[0]):
        result[b, 0] = binary_closing(pred_np[b, 0], structure=structure)

    return torch.from_numpy(result.astype(np.float32))


def confidence_filter(pred_probs, threshold_low=0.5, threshold_high=0.7):
    """
    Two-stage confidence filtering:
    - Keep regions where max probability > threshold_high
    - For threshold_low < prob < threshold_high, only keep if connected to high-confidence region
    """
    high_conf = (pred_probs > threshold_high).float()
    low_conf = (pred_probs > threshold_low).float()

    result = high_conf.clone()

    # For each sample, dilate high-confidence regions and intersect with low-confidence
    pred_np = result.numpy()
    low_np = low_conf.numpy()

    structure = ndimage.generate_binary_structure(3, 1)
    structure = ndimage.iterate_structure(structure, 2)

    for b in range(pred_np.shape[0]):
        dilated = binary_dilation(pred_np[b, 0], structure=structure, iterations=2)
        # Keep low-confidence voxels that are near high-confidence regions
        connected_low = dilated & low_np[b, 0].astype(bool)
        pred_np[b, 0] = pred_np[b, 0] | connected_low

    return torch.from_numpy(pred_np.astype(np.float32))


def full_postprocessing_pipeline(pred_probs, threshold=0.5, min_size=15, opening_size=1, closing_size=1):
    """
    Full post-processing pipeline:
    1. Threshold predictions
    2. Morphological opening (remove noise)
    3. Remove small components
    4. Morphological closing (fill holes)
    """
    # 1. Threshold
    pred_binary = (pred_probs > threshold).float()

    # 2. Opening (remove noise)
    if opening_size > 0:
        pred_binary = morphological_opening(pred_binary, structure_size=opening_size)

    # 3. Remove small components
    if min_size > 0:
        pred_binary = remove_small_components(pred_binary, min_size=min_size)

    # 4. Closing (fill holes)
    if closing_size > 0:
        pred_binary = morphological_closing(pred_binary, structure_size=closing_size)

    return pred_binary


def evaluate_postprocessing(pred_probs, masks, name, postprocess_fn, **kwargs):
    """Evaluate a post-processing strategy."""
    pred_binary = postprocess_fn(pred_probs, **kwargs)

    voxel_metrics = compute_metrics(pred_binary, masks)
    lesion_metrics = compute_lesion_metrics(pred_binary, masks)

    return {
        'name': name,
        'params': kwargs,
        **voxel_metrics,
        **lesion_metrics
    }


def main():
    print("=" * 70)
    print("POST-PROCESSING EVALUATION")
    print("=" * 70)

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    output_dir = project_dir / "outputs"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load predictions
    pred_probs, masks = load_ensemble_predictions(data_dir, model_dir, device)

    # Test different strategies
    results = []

    print("\n" + "=" * 70)
    print("TESTING POST-PROCESSING STRATEGIES")
    print("=" * 70)

    # Baseline (no post-processing)
    for thresh in [0.5, 0.6]:
        print(f"\nBaseline (threshold={thresh})...")
        r = evaluate_postprocessing(pred_probs, masks, f'baseline_t{thresh}',
                                    lambda x, t=thresh: (x > t).float())
        results.append(r)
        print(f"  Sens: {r['sensitivity']*100:.1f}%, Spec: {r['specificity']*100:.1f}%, "
              f"Dice: {r['dice']*100:.1f}%, L-Sens: {r['lesion_sensitivity']*100:.1f}%")

    # Size filtering only
    for min_size in [5, 10, 20, 30, 50]:
        print(f"\nSize filter (min_size={min_size})...")
        r = evaluate_postprocessing(pred_probs, masks, f'size_filter_{min_size}',
                                    lambda x, ms=min_size: remove_small_components((x > 0.5).float(), min_size=ms))
        results.append(r)
        print(f"  Sens: {r['sensitivity']*100:.1f}%, Spec: {r['specificity']*100:.1f}%, "
              f"Dice: {r['dice']*100:.1f}%, L-Sens: {r['lesion_sensitivity']*100:.1f}%, FP: {r['FP']:,}")

    # Morphological opening
    for size in [1, 2]:
        print(f"\nMorphological opening (size={size})...")
        r = evaluate_postprocessing(pred_probs, masks, f'opening_{size}',
                                    lambda x, s=size: morphological_opening((x > 0.5).float(), structure_size=s))
        results.append(r)
        print(f"  Sens: {r['sensitivity']*100:.1f}%, Spec: {r['specificity']*100:.1f}%, "
              f"Dice: {r['dice']*100:.1f}%, L-Sens: {r['lesion_sensitivity']*100:.1f}%, FP: {r['FP']:,}")

    # Combined pipeline with different parameters
    print("\n" + "-" * 70)
    print("COMBINED PIPELINES")
    print("-" * 70)

    pipeline_configs = [
        {'threshold': 0.5, 'min_size': 10, 'opening_size': 1, 'closing_size': 1},
        {'threshold': 0.5, 'min_size': 20, 'opening_size': 1, 'closing_size': 1},
        {'threshold': 0.5, 'min_size': 15, 'opening_size': 1, 'closing_size': 0},
        {'threshold': 0.6, 'min_size': 10, 'opening_size': 1, 'closing_size': 1},
        {'threshold': 0.6, 'min_size': 20, 'opening_size': 1, 'closing_size': 1},
        {'threshold': 0.55, 'min_size': 15, 'opening_size': 1, 'closing_size': 1},
    ]

    for config in pipeline_configs:
        name = f"pipeline_t{config['threshold']}_s{config['min_size']}_o{config['opening_size']}"
        print(f"\n{name}...")
        r = evaluate_postprocessing(pred_probs, masks, name,
                                    full_postprocessing_pipeline, **config)
        results.append(r)
        print(f"  Sens: {r['sensitivity']*100:.1f}%, Spec: {r['specificity']*100:.1f}%, "
              f"Dice: {r['dice']*100:.1f}%, Prec: {r['precision']*100:.1f}%")
        print(f"  L-Sens: {r['lesion_sensitivity']*100:.1f}%, L-Prec: {r['lesion_precision']*100:.1f}%, "
              f"L-F1: {r['lesion_f1']*100:.1f}%")
        print(f"  FP: {r['FP']:,} -> reduced by {(results[0]['FP'] - r['FP']):,}")

    # Find best configurations
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)

    # Best sensitivity >= 95%
    sens_95 = [r for r in results if r['sensitivity'] >= 0.95]
    if sens_95:
        best = max(sens_95, key=lambda x: x['dice'])
        print(f"\nBest for >=95% sensitivity: {best['name']}")
        print(f"  Sens: {best['sensitivity']*100:.1f}%, Dice: {best['dice']*100:.1f}%, "
              f"L-F1: {best['lesion_f1']*100:.1f}%")

    # Best lesion F1
    best_lf1 = max(results, key=lambda x: x['lesion_f1'])
    print(f"\nBest Lesion F1: {best_lf1['name']}")
    print(f"  L-F1: {best_lf1['lesion_f1']*100:.1f}%, Sens: {best_lf1['sensitivity']*100:.1f}%, "
          f"L-Sens: {best_lf1['lesion_sensitivity']*100:.1f}%")

    # Best dice
    best_dice = max(results, key=lambda x: x['dice'])
    print(f"\nBest Dice: {best_dice['name']}")
    print(f"  Dice: {best_dice['dice']*100:.1f}%, Sens: {best_dice['sensitivity']*100:.1f}%")

    # Save results
    save_results = []
    for r in results:
        save_r = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                  for k, v in r.items()}
        save_results.append(save_r)

    output_path = output_dir / 'postprocessing_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': save_results
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Create comparison plot
    print("\nGenerating comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Sensitivity vs Specificity trade-off
    ax1 = axes[0]
    for r in results:
        marker = 'o' if 'pipeline' in r['name'] else 's'
        color = 'blue' if 'pipeline' in r['name'] else 'gray'
        ax1.scatter(1 - r['specificity'], r['sensitivity'], marker=marker, c=color, s=80, alpha=0.7)

    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% Sensitivity Target')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=11)
    ax1.set_title('ROC-style: Sensitivity vs FPR', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Dice vs Lesion F1
    ax2 = axes[1]
    for r in results:
        marker = 'o' if 'pipeline' in r['name'] else 's'
        color = 'green' if 'pipeline' in r['name'] else 'gray'
        ax2.scatter(r['dice'], r['lesion_f1'], marker=marker, c=color, s=80, alpha=0.7)

    ax2.set_xlabel('Dice Score', fontsize=11)
    ax2.set_ylabel('Lesion F1 Score', fontsize=11)
    ax2.set_title('Dice vs Lesion F1', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'postprocessing_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {plot_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
