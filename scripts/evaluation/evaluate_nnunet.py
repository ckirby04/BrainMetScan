"""
Full lesion-wise evaluation for nnU-Net predictions.

Loads nnU-Net validation predictions (NIfTI files from fold_X/validation/) and
ground truth masks, computes ALL metrics identical to lesionwise_eval.py for
apples-to-apples comparison with the custom ensemble.

Metrics computed per case:
  1. Voxel Dice (standard)
  2. Voxel Sensitivity & Precision
  3. Lesion-wise F1 (detection-level)
  4. Per-lesion Dice (matched GT<->pred pairs)
  5. Surface Dice / NSD at tolerance 1, 2, 3 voxels
  6. Hausdorff Distance 95th percentile
  7. Boundary-relaxed Dice (dilate GT by 1, 2, 3 voxels)
  8. Missed lesion analysis by size

Usage:
    python scripts/evaluate_nnunet.py                    # Evaluate fold 0 (raw + postprocessed if available)
    python scripts/evaluate_nnunet.py --fold 0           # Specific fold
    python scripts/evaluate_nnunet.py --postprocessed    # Only evaluate postprocessed predictions
    python scripts/evaluate_nnunet.py --trainer "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (label as ndimage_label, distance_transform_edt,
                            binary_dilation, binary_erosion, generate_binary_structure)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'model'

# nnU-Net paths
NNUNET_BASE = ROOT / 'nnUNet' / 'nnUNet_results' / 'Dataset001_BrainMets'
DEFAULT_TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
LABELS_DIR = ROOT / 'nnUNet' / 'nnUNet_raw' / 'Dataset001_BrainMets' / 'labelsTr'


# =============================================================================
# METRICS (identical to lesionwise_eval.py)
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


def lesionwise_f1(pred_bin, gt_bin, min_overlap=1):
    """Lesion-level detection F1."""
    gt_labeled, n_gt = ndimage_label(gt_bin)
    pred_labeled, n_pred = ndimage_label(pred_bin)

    gt_detected = set()
    pred_matched = set()

    for i in range(1, n_gt + 1):
        gt_mask = (gt_labeled == i)
        overlapping_preds = pred_labeled[gt_mask]
        overlapping_preds = overlapping_preds[overlapping_preds > 0]
        if len(overlapping_preds) >= min_overlap:
            gt_detected.add(i)
            for p in np.unique(overlapping_preds):
                pred_matched.add(p)

    tp = len(gt_detected)
    fn = n_gt - tp
    fp = n_pred - len(pred_matched)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    gt_lesion_sizes = []
    gt_lesion_detected = []
    for i in range(1, n_gt + 1):
        size = int((gt_labeled == i).sum())
        gt_lesion_sizes.append(size)
        gt_lesion_detected.append(i in gt_detected)

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_gt_lesions': n_gt, 'n_pred_lesions': n_pred,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'gt_lesion_sizes': gt_lesion_sizes,
        'gt_lesion_detected': gt_lesion_detected,
    }


def per_lesion_dice(pred_bin, gt_bin):
    """Dice computed per GT lesion, then averaged. Missed lesions = 0."""
    gt_labeled, n_gt = ndimage_label(gt_bin)
    pred_labeled, n_pred = ndimage_label(pred_bin)

    if n_gt == 0:
        return {'mean_dice': 1.0 if n_pred == 0 else 0.0, 'per_lesion': [], 'n_lesions': 0}

    lesion_dices = []
    for i in range(1, n_gt + 1):
        gt_mask = (gt_labeled == i)
        gt_size = int(gt_mask.sum())

        overlapping = pred_labeled[gt_mask]
        unique_preds = np.unique(overlapping[overlapping > 0])

        if len(unique_preds) == 0:
            lesion_dices.append({'size': gt_size, 'dice': 0.0, 'matched': False})
            continue

        merged_pred = np.zeros_like(pred_bin, dtype=bool)
        for p in unique_preds:
            merged_pred |= (pred_labeled == p)

        tp = (gt_mask & merged_pred).sum()
        fp = (merged_pred & ~gt_mask).sum()
        fn = (gt_mask & ~merged_pred).sum()
        dice = float(2 * tp) / float(2 * tp + fp + fn + 1e-8)

        lesion_dices.append({'size': gt_size, 'dice': dice, 'matched': True})

    mean_dice = np.mean([l['dice'] for l in lesion_dices])
    return {'mean_dice': float(mean_dice), 'per_lesion': lesion_dices, 'n_lesions': n_gt}


def compute_surface_metrics(pred_bin, gt_bin):
    """
    Compute ALL surface-based metrics in a single pass.

    Returns surface_dice at tolerances 1/2/3, hausdorff_95, and relaxed_dice at 1/2/3.
    Only computes EDT twice total (instead of 8+ times).
    """
    result = {
        'surface_dice_1': 0.0, 'surface_dice_2': 0.0, 'surface_dice_3': 0.0,
        'hausdorff_95': float('inf'),
        'relaxed_dice_1': 0.0, 'relaxed_dice_2': 0.0, 'relaxed_dice_3': 0.0,
    }

    pred_any = pred_bin.sum() > 0
    gt_any = gt_bin.sum() > 0

    if not pred_any and not gt_any:
        result.update({k: 1.0 for k in result})
        result['hausdorff_95'] = 0.0
        return result

    struct = generate_binary_structure(3, 1)

    # --- Relaxed Dice (dilate GT, then standard dice) ---
    gt_bool = gt_bin.astype(bool)
    gt_d1 = binary_dilation(gt_bool, structure=struct, iterations=1)
    gt_d2 = binary_dilation(gt_d1, structure=struct, iterations=1)
    gt_d3 = binary_dilation(gt_d2, structure=struct, iterations=1)
    result['relaxed_dice_1'] = voxel_dice(pred_bin, gt_d1.astype(np.float32))
    result['relaxed_dice_2'] = voxel_dice(pred_bin, gt_d2.astype(np.float32))
    result['relaxed_dice_3'] = voxel_dice(pred_bin, gt_d3.astype(np.float32))

    if not pred_any or not gt_any:
        return result

    # --- Surfaces (computed once) ---
    pred_bool = pred_bin.astype(bool)
    pred_eroded = binary_erosion(pred_bool, structure=struct, iterations=1)
    pred_surface = pred_bool & ~pred_eroded

    gt_eroded = binary_erosion(gt_bool, structure=struct, iterations=1)
    gt_surface = gt_bool & ~gt_eroded

    n_pred_surface = int(pred_surface.sum())
    n_gt_surface = int(gt_surface.sum())

    if n_pred_surface == 0 or n_gt_surface == 0:
        return result

    # --- EDT (the expensive part — only computed TWICE total) ---
    gt_dist = distance_transform_edt(~gt_surface)
    pred_dist = distance_transform_edt(~pred_surface)

    # Surface Dice at multiple tolerances (reuses same EDT)
    pred_to_gt_dists = gt_dist[pred_surface]
    gt_to_pred_dists = pred_dist[gt_surface]

    for tol in [1.0, 2.0, 3.0]:
        pred_within = (pred_to_gt_dists <= tol).sum()
        gt_within = (gt_to_pred_dists <= tol).sum()
        nsd = float(pred_within + gt_within) / float(n_pred_surface + n_gt_surface + 1e-8)
        result[f'surface_dice_{int(tol)}'] = nsd

    # Hausdorff 95 (reuses same distance arrays)
    all_distances = np.concatenate([pred_to_gt_dists, gt_to_pred_dists])
    result['hausdorff_95'] = float(np.percentile(all_distances, 95))

    return result


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions(pred_dir, labels_dir):
    """Load prediction NIfTIs and matching ground truth.

    Returns list of (case_id, pred_bin, gt_bin) tuples.
    """
    pred_files = sorted(pred_dir.glob('*.nii.gz'))
    if not pred_files:
        print(f"  No prediction files found in {pred_dir}")
        return []

    cases = []
    skipped = 0
    for pred_file in pred_files:
        case_id = pred_file.name.replace('.nii.gz', '')

        # Load ground truth
        gt_file = labels_dir / pred_file.name
        if not gt_file.exists():
            skipped += 1
            continue

        pred_nii = nib.load(str(pred_file))
        pred = np.asarray(pred_nii.dataobj, dtype=np.float32)
        pred_bin = (pred > 0.5).astype(np.float32)

        gt_nii = nib.load(str(gt_file))
        gt = np.asarray(gt_nii.dataobj, dtype=np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)

        cases.append((case_id, pred_bin, gt_bin))

    if skipped:
        print(f"  Skipped {skipped} predictions (no matching GT)")

    return cases


# =============================================================================
# EVALUATE A SET OF CASES
# =============================================================================

def evaluate_cases(cases, label):
    """Run full evaluation on loaded cases. Returns per-case metrics and summary."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {label} ({len(cases)} cases)")
    print(f"{'='*70}")

    case_metrics = []

    for i, (case_id, pred_bin, gt_bin) in enumerate(cases):
        t_case = time.time()

        # Voxel-level metrics
        vd = voxel_dice(pred_bin, gt_bin)
        vs = voxel_sensitivity(pred_bin, gt_bin)
        vp = voxel_precision(pred_bin, gt_bin)

        # Lesion detection
        lf1 = lesionwise_f1(pred_bin, gt_bin)

        # Per-lesion Dice
        pld = per_lesion_dice(pred_bin, gt_bin)

        # Surface/boundary metrics in one pass
        surf = compute_surface_metrics(pred_bin, gt_bin)

        elapsed_case = time.time() - t_case
        if (i + 1) % 10 == 0 or (i + 1) == len(cases) or (i + 1) == 1:
            print(f"  [{i+1}/{len(cases)}] {case_id} — dice={vd:.3f}, "
                  f"lesF1={lf1['f1']:.3f}, surfD2={surf['surface_dice_2']:.3f} ({elapsed_case:.1f}s)")

        case_metrics.append({
            'case_id': case_id,
            'lesion_voxels': int(gt_bin.sum()),
            'pred_voxels': int(pred_bin.sum()),
            'n_gt_lesions': lf1['n_gt_lesions'],
            'n_pred_lesions': lf1['n_pred_lesions'],
            # Voxel-level
            'voxel_dice': vd,
            'voxel_sensitivity': vs,
            'voxel_precision': vp,
            # Lesion detection
            'lesion_f1': lf1['f1'],
            'lesion_recall': lf1['recall'],
            'lesion_precision': lf1['precision'],
            'lesion_tp': lf1['tp'],
            'lesion_fp': lf1['fp'],
            'lesion_fn': lf1['fn'],
            # Per-lesion quality
            'per_lesion_dice': pld['mean_dice'],
            # Surface metrics
            'surface_dice_1': surf['surface_dice_1'],
            'surface_dice_2': surf['surface_dice_2'],
            'surface_dice_3': surf['surface_dice_3'],
            # Hausdorff
            'hausdorff_95': surf['hausdorff_95'],
            # Relaxed dice
            'relaxed_dice_1': surf['relaxed_dice_1'],
            'relaxed_dice_2': surf['relaxed_dice_2'],
            'relaxed_dice_3': surf['relaxed_dice_3'],
            # Missed lesion info
            'missed_lesion_sizes': [
                s for s, d in zip(lf1['gt_lesion_sizes'], lf1['gt_lesion_detected']) if not d
            ],
        })

    # =========================================================================
    # Aggregate
    # =========================================================================
    valid = [c for c in case_metrics if c['lesion_voxels'] > 0]
    n = len(valid)

    summary = {'n_cases': n, 'n_total': len(case_metrics), 'label': label}

    # Per-metric stats
    metric_keys = [
        'voxel_dice', 'voxel_sensitivity', 'voxel_precision',
        'lesion_f1', 'lesion_recall', 'lesion_precision',
        'per_lesion_dice',
        'surface_dice_1', 'surface_dice_2', 'surface_dice_3',
        'relaxed_dice_1', 'relaxed_dice_2', 'relaxed_dice_3',
    ]

    for key in metric_keys:
        values = [c[key] for c in valid]
        summary[key] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)),
        }

    # HD95
    hd_values = [c['hausdorff_95'] for c in valid if c['hausdorff_95'] != float('inf')]
    summary['hausdorff_95'] = {
        'mean': float(np.mean(hd_values)) if hd_values else float('inf'),
        'median': float(np.median(hd_values)) if hd_values else float('inf'),
        'p95': float(np.percentile(hd_values, 95)) if hd_values else float('inf'),
        'valid_cases': len(hd_values),
        'inf_cases': n - len(hd_values),
    }

    # Lesion detection totals
    total_tp = sum(c['lesion_tp'] for c in valid)
    total_fp = sum(c['lesion_fp'] for c in valid)
    total_fn = sum(c['lesion_fn'] for c in valid)
    total_gt = sum(c['n_gt_lesions'] for c in valid)
    total_pred = sum(c['n_pred_lesions'] for c in valid)
    summary['lesion_detection'] = {
        'total_gt_lesions': total_gt,
        'total_pred_lesions': total_pred,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'overall_precision': float(total_tp / (total_tp + total_fp + 1e-8)),
        'overall_recall': float(total_tp / (total_tp + total_fn + 1e-8)),
        'overall_f1': float(2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)),
    }

    # Missed lesion analysis
    all_missed = []
    for c in valid:
        all_missed.extend(c['missed_lesion_sizes'])

    if all_missed:
        summary['missed_lesions'] = {
            'total': len(all_missed),
            'mean_size': float(np.mean(all_missed)),
            'median_size': float(np.median(all_missed)),
            'min_size': int(min(all_missed)),
            'max_size': int(max(all_missed)),
            'by_size': {
                '<50 vox': sum(1 for s in all_missed if s < 50),
                '50-200': sum(1 for s in all_missed if 50 <= s < 200),
                '200-1000': sum(1 for s in all_missed if 200 <= s < 1000),
                '1000-5000': sum(1 for s in all_missed if 1000 <= s < 5000),
                '>5000': sum(1 for s in all_missed if s >= 5000),
            }
        }
    else:
        summary['missed_lesions'] = {'total': 0}

    # Dice by lesion size bucket
    buckets = [
        ('tiny (<100)', 0, 100),
        ('small (100-500)', 100, 500),
        ('medium (500-5k)', 500, 5000),
        ('large (5k-20k)', 5000, 20000),
        ('huge (>20k)', 20000, 1e9),
    ]
    summary['dice_by_size'] = {}
    for bucket_label, lo, hi in buckets:
        bucket_cases = [c for c in valid if lo <= c['lesion_voxels'] < hi]
        if bucket_cases:
            summary['dice_by_size'][bucket_label] = {
                'n': len(bucket_cases),
                'voxel_dice': float(np.mean([c['voxel_dice'] for c in bucket_cases])),
                'per_lesion_dice': float(np.mean([c['per_lesion_dice'] for c in bucket_cases])),
                'surface_dice_2': float(np.mean([c['surface_dice_2'] for c in bucket_cases])),
                'lesion_f1': float(np.mean([c['lesion_f1'] for c in bucket_cases])),
                'relaxed_dice_2': float(np.mean([c['relaxed_dice_2'] for c in bucket_cases])),
            }

    # =========================================================================
    # Print results
    # =========================================================================
    print(f"\n  === {label} ({n} cases with foreground) ===\n")

    print(f"  {'Metric':<25} {'Mean':<10} {'Median':<10} {'Std':<10} {'25th':<10} {'75th':<10}")
    print(f"  {'-'*75}")
    for key in metric_keys:
        s = summary[key]
        print(f"  {key:<25} {s['mean']:<10.4f} {s['median']:<10.4f} "
              f"{s['std']:<10.4f} {s['p25']:<10.4f} {s['p75']:<10.4f}")

    h = summary['hausdorff_95']
    print(f"  {'hausdorff_95':<25} {h['mean']:<10.2f} {h['median']:<10.2f} "
          f"{'':10} {'':10} (inf={h['inf_cases']})")

    ld = summary['lesion_detection']
    print(f"\n  Lesion detection: {ld['total_tp']}/{ld['total_gt_lesions']} detected "
          f"(recall={ld['overall_recall']:.3f}), "
          f"{ld['total_fp']} FP (prec={ld['overall_precision']:.3f}), "
          f"F1={ld['overall_f1']:.3f}")

    ml = summary['missed_lesions']
    if ml['total'] > 0:
        print(f"  Missed {ml['total']} lesions: median={ml['median_size']:.0f} vox, "
              f"range=[{ml['min_size']}, {ml['max_size']}]")
        by = ml['by_size']
        print(f"    By size: <50={by['<50 vox']}, 50-200={by['50-200']}, "
              f"200-1k={by['200-1000']}, 1k-5k={by['1000-5000']}, >5k={by['>5000']}")

    print(f"\n  Dice by lesion size:")
    print(f"  {'Bucket':<20} {'N':<5} {'Voxel':<8} {'PerLes':<8} {'SurfD2':<8} {'LesF1':<8} {'RelxD2':<8}")
    print(f"  {'-'*65}")
    for bucket_label, _, _ in buckets:
        if bucket_label in summary['dice_by_size']:
            b = summary['dice_by_size'][bucket_label]
            print(f"  {bucket_label:<20} {b['n']:<5} {b['voxel_dice']:<8.4f} "
                  f"{b['per_lesion_dice']:<8.4f} {b['surface_dice_2']:<8.4f} "
                  f"{b['lesion_f1']:<8.4f} {b['relaxed_dice_2']:<8.4f}")

    return summary, case_metrics


def print_comparison(nnunet_summary, ensemble_data):
    """Print side-by-side comparison with ensemble results."""
    print(f"\n{'='*70}")
    print("  COMPARISON: nnU-Net vs Custom Ensemble")
    print(f"{'='*70}")

    # Use val_set from ensemble data for comparison
    ens_summary = ensemble_data.get('val_set', {}).get('summary', {})
    if not ens_summary:
        print("  No ensemble val_set data found for comparison")
        return

    compare_keys = [
        'voxel_dice', 'voxel_sensitivity', 'voxel_precision',
        'lesion_f1', 'lesion_recall', 'lesion_precision',
        'per_lesion_dice',
        'surface_dice_1', 'surface_dice_2', 'surface_dice_3',
        'relaxed_dice_1', 'relaxed_dice_2', 'relaxed_dice_3',
    ]

    print(f"\n  {'Metric':<25} {'nnU-Net':<12} {'Ensemble':<12} {'Delta':<10}")
    print(f"  {'-'*60}")
    for key in compare_keys:
        nn_val = nnunet_summary.get(key, {}).get('mean', 0)
        ens_val = ens_summary.get(key, {}).get('mean', 0)
        delta = nn_val - ens_val
        marker = ' ***' if abs(delta) > 0.02 else ''
        print(f"  {key:<25} {nn_val:<12.4f} {ens_val:<12.4f} {delta:+.4f}{marker}")

    # HD95
    nn_hd = nnunet_summary.get('hausdorff_95', {}).get('mean', float('inf'))
    ens_hd = ens_summary.get('hausdorff_95', {}).get('mean', float('inf'))
    if nn_hd != float('inf') and ens_hd != float('inf'):
        print(f"  {'hausdorff_95':<25} {nn_hd:<12.2f} {ens_hd:<12.2f} {nn_hd - ens_hd:+.2f}")

    # Lesion detection
    nn_ld = nnunet_summary.get('lesion_detection', {})
    ens_ld = ens_summary.get('lesion_detection', {})
    print(f"\n  Lesion Detection:")
    print(f"    {'Metric':<20} {'nnU-Net':<15} {'Ensemble':<15}")
    print(f"    {'-'*50}")
    for det_key in ['overall_f1', 'overall_recall', 'overall_precision']:
        nn_v = nn_ld.get(det_key, 0)
        ens_v = ens_ld.get(det_key, 0)
        print(f"    {det_key:<20} {nn_v:<15.4f} {ens_v:<15.4f}")
    print(f"    {'total_gt_lesions':<20} {nn_ld.get('total_gt_lesions', 0):<15} "
          f"{ens_ld.get('total_gt_lesions', 0):<15}")
    print(f"    {'total_tp':<20} {nn_ld.get('total_tp', 0):<15} "
          f"{ens_ld.get('total_tp', 0):<15}")
    print(f"    {'total_fn':<20} {nn_ld.get('total_fn', 0):<15} "
          f"{ens_ld.get('total_fn', 0):<15}")

    # Size buckets
    nn_sizes = nnunet_summary.get('dice_by_size', {})
    ens_sizes = ens_summary.get('dice_by_size', {})
    if nn_sizes or ens_sizes:
        print(f"\n  Dice by Size:")
        all_buckets = ['tiny (<100)', 'small (100-500)', 'medium (500-5k)',
                       'large (5k-20k)', 'huge (>20k)']
        print(f"    {'Bucket':<20} {'nnU-Net':<12} {'Ensemble':<12} {'Delta':<10}")
        print(f"    {'-'*55}")
        for b in all_buckets:
            nn_d = nn_sizes.get(b, {}).get('voxel_dice', None)
            ens_d = ens_sizes.get(b, {}).get('voxel_dice', None)
            nn_str = f"{nn_d:.4f}" if nn_d is not None else "---"
            ens_str = f"{ens_d:.4f}" if ens_d is not None else "---"
            delta_str = f"{nn_d - ens_d:+.4f}" if nn_d is not None and ens_d is not None else "---"
            print(f"    {b:<20} {nn_str:<12} {ens_str:<12} {delta_str:<10}")


def print_multi_comparison(results_dict, ensemble_data=None):
    """Print comparison across multiple nnU-Net variants and optionally ensemble."""
    print(f"\n{'='*70}")
    print("  FULL COMPARISON TABLE")
    print(f"{'='*70}")

    # Gather all summaries
    summaries = {}
    for key, data in results_dict.items():
        summaries[key] = data['summary']

    if ensemble_data:
        ens_summary = ensemble_data.get('val_set', {}).get('summary', {})
        if ens_summary:
            summaries['ensemble'] = ens_summary

    if not summaries:
        return

    # Header
    names = list(summaries.keys())
    col_width = max(14, max(len(n) for n in names) + 2)

    print(f"\n  {'Metric':<25}", end='')
    for name in names:
        print(f"  {name:<{col_width}}", end='')
    print()
    print(f"  {'-'*25}", end='')
    for _ in names:
        print(f"  {'-'*col_width}", end='')
    print()

    compare_keys = [
        'voxel_dice', 'voxel_sensitivity', 'voxel_precision',
        'lesion_f1', 'per_lesion_dice',
        'surface_dice_2', 'relaxed_dice_2',
    ]

    for key in compare_keys:
        print(f"  {key:<25}", end='')
        for name in names:
            val = summaries[name].get(key, {}).get('mean', None)
            if val is not None:
                print(f"  {val:<{col_width}.4f}", end='')
            else:
                print(f"  {'---':<{col_width}}", end='')
        print()

    # HD95
    print(f"  {'hausdorff_95':<25}", end='')
    for name in names:
        val = summaries[name].get('hausdorff_95', {}).get('median', None)
        if val is not None and val != float('inf'):
            print(f"  {val:<{col_width}.2f}", end='')
        else:
            print(f"  {'inf':<{col_width}}", end='')
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full nnU-Net evaluation (lesionwise_eval format)")
    parser.add_argument('--fold', type=int, default=0, help="nnU-Net fold (default: 0)")
    parser.add_argument('--trainer', type=str, default=DEFAULT_TRAINER,
                        help=f"Trainer directory name (default: {DEFAULT_TRAINER})")
    parser.add_argument('--postprocessed', action='store_true',
                        help="Only evaluate postprocessed predictions")
    parser.add_argument('--no-compare', action='store_true',
                        help="Skip comparison with ensemble results")
    args = parser.parse_args()

    print("=" * 70)
    print("  Full nnU-Net Evaluation (lesionwise_eval format)")
    print("=" * 70)

    t0 = time.time()

    trainer_dir = NNUNET_BASE / args.trainer
    fold_dir = trainer_dir / f'fold_{args.fold}'

    if not fold_dir.exists():
        print(f"\nERROR: Fold directory not found: {fold_dir}")
        sys.exit(1)

    # Determine short label for this trainer
    if 'ResEnc' in args.trainer:
        trainer_label = 'resenc'
    else:
        trainer_label = 'standard'

    results = {}

    # =========================================================================
    # Evaluate raw predictions
    # =========================================================================
    if not args.postprocessed:
        val_dir = fold_dir / 'validation'
        if val_dir.exists():
            print(f"\n--- Loading raw predictions from {val_dir} ---")
            cases = load_predictions(val_dir, LABELS_DIR)
            print(f"  Loaded {len(cases)} cases")

            if cases:
                key = f'nnunet_fold{args.fold}'
                if trainer_label != 'standard':
                    key = f'nnunet_{trainer_label}_fold{args.fold}'
                summary, case_metrics = evaluate_cases(cases, f"nnU-Net {trainer_label} fold {args.fold} (raw)")
                results[key] = {
                    'summary': summary,
                    'per_case': [{k: v for k, v in c.items() if k != 'missed_lesion_sizes'}
                                 for c in case_metrics],
                }
        else:
            print(f"\n  Raw validation dir not found: {val_dir}")

    # =========================================================================
    # Evaluate postprocessed predictions (if they exist)
    # =========================================================================
    # Check multiple possible locations for postprocessed predictions
    pp_candidates = [
        fold_dir / 'validation' / 'postprocessed',
        trainer_dir / f'crossval_results_folds_{args.fold}' / 'postprocessed',
        trainer_dir / f'crossval_results_folds_{args.fold}',
    ]

    pp_dir = None
    for candidate in pp_candidates:
        if candidate.exists() and list(candidate.glob('*.nii.gz')):
            pp_dir = candidate
            break

    if pp_dir:
        print(f"\n--- Loading postprocessed predictions from {pp_dir} ---")
        pp_cases = load_predictions(pp_dir, LABELS_DIR)
        print(f"  Loaded {len(pp_cases)} postprocessed cases")

        if pp_cases:
            key = f'nnunet_fold{args.fold}_postprocessed'
            if trainer_label != 'standard':
                key = f'nnunet_{trainer_label}_fold{args.fold}_postprocessed'
            pp_summary, pp_case_metrics = evaluate_cases(
                pp_cases, f"nnU-Net {trainer_label} fold {args.fold} (postprocessed)")
            results[key] = {
                'summary': pp_summary,
                'per_case': [{k: v for k, v in c.items() if k != 'missed_lesion_sizes'}
                             for c in pp_case_metrics],
            }
    elif args.postprocessed:
        print(f"\n  No postprocessed predictions found. Checked:")
        for c in pp_candidates:
            print(f"    {c}")
        print("  Run nnU-Net postprocessing first (see Task 2 in plan).")

    # =========================================================================
    # Worst cases
    # =========================================================================
    for key, data in results.items():
        per_case = data['per_case']
        valid = [c for c in per_case if c.get('lesion_voxels', 0) > 0]
        if valid:
            print(f"\n{'='*70}")
            print(f"  WORST 10 CASES — {key}")
            print(f"{'='*70}")
            print(f"  {'Case':<25} {'VoxDice':<9} {'LesF1':<8} {'PerLes':<8} {'SurfD2':<8} {'HD95':<8} {'Vox':<8}")
            print(f"  {'-'*75}")
            sorted_cases = sorted(valid, key=lambda c: c['voxel_dice'])
            for c in sorted_cases[:10]:
                hd = f"{c['hausdorff_95']:.1f}" if c['hausdorff_95'] != float('inf') else "inf"
                print(f"  {c['case_id']:<25} {c['voxel_dice']:<9.4f} {c['lesion_f1']:<8.4f} "
                      f"{c['per_lesion_dice']:<8.4f} {c['surface_dice_2']:<8.4f} "
                      f"{hd:<8} {c['lesion_voxels']:<8}")

    # =========================================================================
    # Comparison with ensemble
    # =========================================================================
    if not args.no_compare:
        ensemble_path = OUTPUT_DIR / 'lesionwise_evaluation.json'
        ensemble_data = None
        if ensemble_path.exists():
            with open(ensemble_path) as f:
                ensemble_data = json.load(f)

        if ensemble_data and results:
            # Print pairwise comparison for the first (raw) result
            first_key = list(results.keys())[0]
            print_comparison(results[first_key]['summary'], ensemble_data)

        # Print full comparison table
        if len(results) > 1 or ensemble_data:
            print_multi_comparison(results, ensemble_data)

    # =========================================================================
    # Save results
    # =========================================================================
    out_path = OUTPUT_DIR / 'nnunet_evaluation.json'

    # Merge with existing results if file exists
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} minutes")
    print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()
