"""
Unified model comparison pipeline.

Compares nnU-Net vs custom ensemble on identical validation cases using
the same metrics: voxel Dice, surface Dice, lesion F1, HD95, per-size breakdown.

Works with:
  - nnU-Net validation predictions (NIfTI from fold_0/validation/)
  - Custom ensemble predictions (from stacking cache .npz files)
  - Any combination of the above

Usage:
    python scripts/compare_models.py                        # Compare all available models
    python scripts/compare_models.py --nnunet-only          # Only evaluate nnU-Net
    python scripts/compare_models.py --ensemble-only        # Only evaluate ensemble
    python scripts/compare_models.py --fold 0               # Specific nnU-Net fold
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import (label as ndimage_label, distance_transform_edt,
                            binary_dilation, binary_erosion, generate_binary_structure)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# nnU-Net paths
NNUNET_BASE = ROOT / 'nnUNet' / 'nnUNet_results' / 'Dataset001_BrainMets'
NNUNET_TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

# Custom ensemble paths
CACHE_DIR = ROOT / 'model' / 'stacking_cache_v4'
TOP3 = ['exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']
ALL_MODELS = ['exp1_8patch', 'exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']
ENSEMBLE_THRESHOLD = 0.40


# =============================================================================
# METRICS (same as lesionwise_eval.py for consistency)
# =============================================================================

def voxel_dice(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin > 0)).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-8)


def voxel_sensitivity(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin > 0)).sum()
    return float(tp) / float(tp + fn + 1e-8)


def voxel_precision(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
    return float(tp) / float(tp + fp + 1e-8)


def lesionwise_detection(pred_bin, gt_bin, min_overlap=1):
    gt_labeled, n_gt = ndimage_label(gt_bin)
    pred_labeled, n_pred = ndimage_label(pred_bin)

    gt_detected = set()
    pred_matched = set()

    for i in range(1, n_gt + 1):
        gt_mask = (gt_labeled == i)
        overlapping = pred_labeled[gt_mask]
        overlapping = overlapping[overlapping > 0]
        if len(overlapping) >= min_overlap:
            gt_detected.add(i)
            for p in np.unique(overlapping):
                pred_matched.add(p)

    tp = len(gt_detected)
    fn = n_gt - tp
    fp = n_pred - len(pred_matched)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Missed lesion sizes
    missed_sizes = []
    for i in range(1, n_gt + 1):
        if i not in gt_detected:
            missed_sizes.append(int((gt_labeled == i).sum()))

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_gt': n_gt, 'n_pred': n_pred,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'missed_sizes': missed_sizes,
    }


def compute_surface_dice_hd95(pred_bin, gt_bin, tolerance=2.0):
    """Compute surface Dice and HD95 in a single pass."""
    pred_any = pred_bin.sum() > 0
    gt_any = gt_bin.sum() > 0

    if not pred_any and not gt_any:
        return {'surface_dice': 1.0, 'hd95': 0.0}
    if not pred_any or not gt_any:
        return {'surface_dice': 0.0, 'hd95': float('inf')}

    struct = generate_binary_structure(3, 1)
    pred_bool = pred_bin.astype(bool)
    gt_bool = gt_bin.astype(bool)

    pred_surface = pred_bool & ~binary_erosion(pred_bool, structure=struct, iterations=1)
    gt_surface = gt_bool & ~binary_erosion(gt_bool, structure=struct, iterations=1)

    n_pred = int(pred_surface.sum())
    n_gt = int(gt_surface.sum())
    if n_pred == 0 or n_gt == 0:
        return {'surface_dice': 0.0, 'hd95': float('inf')}

    gt_dist = distance_transform_edt(~gt_surface)
    pred_dist = distance_transform_edt(~pred_surface)

    pred_to_gt = gt_dist[pred_surface]
    gt_to_pred = pred_dist[gt_surface]

    # Surface Dice
    pred_within = (pred_to_gt <= tolerance).sum()
    gt_within = (gt_to_pred <= tolerance).sum()
    surface_dice = float(pred_within + gt_within) / float(n_pred + n_gt + 1e-8)

    # HD95
    all_dists = np.concatenate([pred_to_gt, gt_to_pred])
    hd95 = float(np.percentile(all_dists, 95))

    return {'surface_dice': surface_dice, 'hd95': hd95}


def evaluate_case(pred_bin, gt_bin):
    """Compute all metrics for a single case."""
    vd = voxel_dice(pred_bin, gt_bin)
    vs = voxel_sensitivity(pred_bin, gt_bin)
    vp = voxel_precision(pred_bin, gt_bin)
    det = lesionwise_detection(pred_bin, gt_bin)
    surf = compute_surface_dice_hd95(pred_bin, gt_bin, tolerance=2.0)

    return {
        'voxel_dice': vd,
        'voxel_sensitivity': vs,
        'voxel_precision': vp,
        'lesion_f1': det['f1'],
        'lesion_recall': det['recall'],
        'lesion_precision': det['precision'],
        'n_gt_lesions': det['n_gt'],
        'n_pred_lesions': det['n_pred'],
        'lesion_tp': det['tp'],
        'lesion_fp': det['fp'],
        'lesion_fn': det['fn'],
        'surface_dice_2': surf['surface_dice'],
        'hd95': surf['hd95'],
        'lesion_voxels': int(gt_bin.sum()),
        'pred_voxels': int(pred_bin.sum()),
        'missed_sizes': det['missed_sizes'],
    }


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_nnunet_predictions(fold=0):
    """Load nnU-Net validation predictions and ground truth.

    Returns dict of {case_id: {'pred': binary_array, 'gt': binary_array}}.
    """
    import nibabel as nib

    fold_dir = NNUNET_BASE / NNUNET_TRAINER / f'fold_{fold}'
    val_dir = fold_dir / 'validation'

    if not val_dir.exists():
        print(f"  nnU-Net validation dir not found: {val_dir}")
        return {}

    # Load the split to know which cases are validation
    splits_file = ROOT / 'nnUNet' / 'nnUNet_preprocessed' / 'Dataset001_BrainMets' / 'splits_final.json'
    if splits_file.exists():
        with open(splits_file) as f:
            splits = json.load(f)
        val_keys = splits[fold]['val'] if fold < len(splits) else []
    else:
        val_keys = []

    # Find prediction files
    pred_files = sorted(val_dir.glob('*.nii.gz'))
    if not pred_files:
        print(f"  No prediction files found in {val_dir}")
        return {}

    # Load ground truth labels
    labels_dir = ROOT / 'nnUNet' / 'nnUNet_raw' / 'Dataset001_BrainMets' / 'labelsTr'

    results = {}
    for pred_file in pred_files:
        case_name = pred_file.name.replace('.nii.gz', '')

        # Load prediction
        pred_nii = nib.load(str(pred_file))
        pred = np.asarray(pred_nii.dataobj, dtype=np.float32)
        pred_bin = (pred > 0.5).astype(np.float32)

        # Load ground truth
        gt_file = labels_dir / pred_file.name
        if not gt_file.exists():
            continue
        gt_nii = nib.load(str(gt_file))
        gt = np.asarray(gt_nii.dataobj, dtype=np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)

        # Map nnU-Net case name back to original case ID
        # nnU-Net format: BrainMets_0000 -> need to map back
        case_id = nnunet_to_case_id(case_name)
        results[case_id] = {'pred': pred_bin, 'gt': gt_bin}

    return results


def nnunet_to_case_id(nnunet_name):
    """Map nnU-Net prediction filename back to original case ID.

    Our setup_nnunet.py preserves original case IDs (e.g., UCSF_100001A),
    so the prediction file is already named after the case.
    """
    return nnunet_name


def load_ensemble_predictions(val_case_ids, threshold=ENSEMBLE_THRESHOLD):
    """Load custom ensemble predictions from stacking cache.

    Returns dict of {case_id: {'pred': binary_array, 'gt': binary_array}}.
    """
    results = {}
    for case_id in val_case_ids:
        cache_file = CACHE_DIR / f'{case_id}.npz'
        if not cache_file.exists():
            continue

        data = np.load(cache_file)
        mask = data['mask'].astype(np.float32)

        # Top-3 average ensemble
        prob = np.mean([data[m].astype(np.float32) for m in TOP3], axis=0)
        pred_bin = (prob > threshold).astype(np.float32)

        results[case_id] = {'pred': pred_bin, 'gt': mask}

    return results


def get_val_case_ids():
    """Get validation case IDs using same split as training (seed=42, 15%)."""
    cached = sorted([f.stem for f in CACHE_DIR.glob('*.npz')])
    if not cached:
        return []
    random.seed(42)
    shuffled = cached.copy()
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * 0.15)
    return shuffled[:n_val]


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_results(case_metrics):
    """Aggregate per-case metrics into summary statistics."""
    if not case_metrics:
        return {}

    metric_keys = ['voxel_dice', 'voxel_sensitivity', 'voxel_precision',
                   'lesion_f1', 'lesion_recall', 'lesion_precision', 'surface_dice_2']

    summary = {'n_cases': len(case_metrics)}

    for key in metric_keys:
        values = [c[key] for c in case_metrics]
        summary[key] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)),
        }

    # HD95
    hd_values = [c['hd95'] for c in case_metrics if c['hd95'] != float('inf')]
    summary['hd95'] = {
        'mean': float(np.mean(hd_values)) if hd_values else float('inf'),
        'median': float(np.median(hd_values)) if hd_values else float('inf'),
        'valid': len(hd_values),
        'inf_cases': len(case_metrics) - len(hd_values),
    }

    # Lesion detection totals
    total_tp = sum(c['lesion_tp'] for c in case_metrics)
    total_fp = sum(c['lesion_fp'] for c in case_metrics)
    total_fn = sum(c['lesion_fn'] for c in case_metrics)
    summary['lesion_detection'] = {
        'total_gt': sum(c['n_gt_lesions'] for c in case_metrics),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'overall_recall': float(total_tp / (total_tp + total_fn + 1e-8)),
        'overall_precision': float(total_tp / (total_tp + total_fp + 1e-8)),
    }

    # Missed lesion sizes
    all_missed = []
    for c in case_metrics:
        all_missed.extend(c.get('missed_sizes', []))
    if all_missed:
        summary['missed_lesions'] = {
            'total': len(all_missed),
            'median_size': float(np.median(all_missed)),
            '<50_vox': sum(1 for s in all_missed if s < 50),
            '50-200': sum(1 for s in all_missed if 50 <= s < 200),
            '>200': sum(1 for s in all_missed if s >= 200),
        }

    # Size buckets
    buckets = [
        ('tiny (<100)', 0, 100), ('small (100-500)', 100, 500),
        ('medium (500-5k)', 500, 5000), ('large (5k-20k)', 5000, 20000),
        ('huge (>20k)', 20000, 1e9),
    ]
    summary['by_size'] = {}
    for label, lo, hi in buckets:
        bucket = [c for c in case_metrics if lo <= c['lesion_voxels'] < hi]
        if bucket:
            summary['by_size'][label] = {
                'n': len(bucket),
                'dice': float(np.mean([c['voxel_dice'] for c in bucket])),
                'lesion_f1': float(np.mean([c['lesion_f1'] for c in bucket])),
                'surface_dice_2': float(np.mean([c['surface_dice_2'] for c in bucket])),
            }

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified model comparison pipeline")
    parser.add_argument('--fold', type=int, default=0, help="nnU-Net fold (default: 0)")
    parser.add_argument('--nnunet-only', action='store_true', help="Only evaluate nnU-Net")
    parser.add_argument('--ensemble-only', action='store_true', help="Only evaluate ensemble")
    parser.add_argument('--threshold', type=float, default=ENSEMBLE_THRESHOLD,
                        help=f"Ensemble threshold (default: {ENSEMBLE_THRESHOLD})")
    args = parser.parse_args()

    print("=" * 70)
    print("  Unified Model Comparison Pipeline")
    print("=" * 70)

    t0 = time.time()
    all_results = {}

    # =========================================================================
    # Load nnU-Net predictions
    # =========================================================================
    if not args.ensemble_only:
        print(f"\n--- Loading nnU-Net predictions (fold {args.fold}) ---")
        nnunet_data = load_nnunet_predictions(fold=args.fold)

        if nnunet_data:
            print(f"  Loaded {len(nnunet_data)} nnU-Net validation cases")

            nnunet_metrics = []
            for case_id, data in sorted(nnunet_data.items()):
                m = evaluate_case(data['pred'], data['gt'])
                m['case_id'] = case_id
                nnunet_metrics.append(m)

            all_results['nnunet'] = {
                'summary': aggregate_results(nnunet_metrics),
                'per_case': nnunet_metrics,
            }
        else:
            print("  No nnU-Net predictions found (training may still be running)")

    # =========================================================================
    # Load ensemble predictions
    # =========================================================================
    if not args.nnunet_only:
        print(f"\n--- Loading ensemble predictions (top3_avg @ {args.threshold}) ---")
        val_ids = get_val_case_ids()

        if val_ids:
            ensemble_data = load_ensemble_predictions(val_ids, threshold=args.threshold)
            print(f"  Loaded {len(ensemble_data)} ensemble validation cases")

            ensemble_metrics = []
            for case_id, data in sorted(ensemble_data.items()):
                m = evaluate_case(data['pred'], data['gt'])
                m['case_id'] = case_id
                ensemble_metrics.append(m)

            all_results['ensemble_top3'] = {
                'summary': aggregate_results(ensemble_metrics),
                'per_case': ensemble_metrics,
            }
        else:
            print("  No ensemble cache found")

    # =========================================================================
    # Print comparison table
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 70}")

    headers = ['Metric'] + list(all_results.keys())
    metric_keys = ['voxel_dice', 'voxel_sensitivity', 'voxel_precision',
                   'lesion_f1', 'surface_dice_2']

    # Header
    print(f"\n  {'Metric':<25}", end='')
    for name in all_results:
        print(f"  {name:<20}", end='')
    print()
    print(f"  {'-'*25}", end='')
    for _ in all_results:
        print(f"  {'-'*20}", end='')
    print()

    # Metric rows (mean +/- std)
    for key in metric_keys:
        print(f"  {key:<25}", end='')
        for name, data in all_results.items():
            s = data['summary'].get(key, {})
            if s:
                print(f"  {s['mean']:.4f} +/- {s['std']:.4f}  ", end='')
            else:
                print(f"  {'N/A':<20}", end='')
        print()

    # HD95
    print(f"  {'hd95 (median)':<25}", end='')
    for name, data in all_results.items():
        h = data['summary'].get('hd95', {})
        if h:
            print(f"  {h['median']:.2f}{'':>14}", end='')
        else:
            print(f"  {'N/A':<20}", end='')
    print()

    # Lesion detection
    print(f"\n  {'Lesion Detection':<25}", end='')
    for name in all_results:
        print(f"  {name:<20}", end='')
    print()
    print(f"  {'-'*25}", end='')
    for _ in all_results:
        print(f"  {'-'*20}", end='')
    print()

    for det_key in ['overall_recall', 'overall_precision']:
        print(f"  {det_key:<25}", end='')
        for name, data in all_results.items():
            ld = data['summary'].get('lesion_detection', {})
            print(f"  {ld.get(det_key, 0):.4f}{'':>14}", end='')
        print()

    # Size breakdown
    print(f"\n  {'Dice by Size':<25}", end='')
    for name in all_results:
        print(f"  {name:<20}", end='')
    print()
    print(f"  {'-'*25}", end='')
    for _ in all_results:
        print(f"  {'-'*20}", end='')
    print()

    size_labels = ['tiny (<100)', 'small (100-500)', 'medium (500-5k)',
                   'large (5k-20k)', 'huge (>20k)']
    for label in size_labels:
        print(f"  {label:<25}", end='')
        for name, data in all_results.items():
            by_size = data['summary'].get('by_size', {})
            b = by_size.get(label)
            if b:
                print(f"  {b['dice']:.4f} (n={b['n']:<3})     ", end='')
            else:
                print(f"  {'---':<20}", end='')
        print()

    # =========================================================================
    # Head-to-head on overlapping cases (if both available)
    # =========================================================================
    if len(all_results) >= 2 and 'nnunet' in all_results and 'ensemble_top3' in all_results:
        nn_cases = {c['case_id']: c for c in all_results['nnunet']['per_case']}
        ens_cases = {c['case_id']: c for c in all_results['ensemble_top3']['per_case']}
        overlap = set(nn_cases.keys()) & set(ens_cases.keys())

        if overlap:
            print(f"\n{'=' * 70}")
            print(f"  HEAD-TO-HEAD ({len(overlap)} overlapping cases)")
            print(f"{'=' * 70}")

            nn_wins, ens_wins, ties = 0, 0, 0
            for cid in overlap:
                diff = nn_cases[cid]['voxel_dice'] - ens_cases[cid]['voxel_dice']
                if diff > 0.01:
                    nn_wins += 1
                elif diff < -0.01:
                    ens_wins += 1
                else:
                    ties += 1

            nn_mean = np.mean([nn_cases[c]['voxel_dice'] for c in overlap])
            ens_mean = np.mean([ens_cases[c]['voxel_dice'] for c in overlap])

            print(f"  nnU-Net wins: {nn_wins}")
            print(f"  Ensemble wins: {ens_wins}")
            print(f"  Ties (+/-0.01): {ties}")
            print(f"  nnU-Net mean Dice: {nn_mean:.4f}")
            print(f"  Ensemble mean Dice: {ens_mean:.4f}")
            print(f"  Delta (nn - ens): {nn_mean - ens_mean:+.4f}")

    # =========================================================================
    # Save results
    # =========================================================================
    # Strip non-serializable data
    save_data = {}
    for name, data in all_results.items():
        save_data[name] = {
            'summary': data['summary'],
            'per_case': [{k: v for k, v in c.items() if k != 'missed_sizes'}
                         for c in data['per_case']],
        }

    out_path = ROOT / 'model' / 'model_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()
