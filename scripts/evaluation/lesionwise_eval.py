"""
Full lesion-wise and boundary-tolerant evaluation on ALL cached cases.

Evaluates the top3_avg ensemble on:
  - Val set only (84 cases, never seen during base model training)
  - All cases (566 total, includes train split — optimistically biased)

Metrics computed per case:
  1. Voxel Dice (standard)
  2. Lesion-wise F1 (detection-level)
  3. Per-lesion Dice (matched GT↔pred pairs)
  4. Surface Dice / NSD at tolerance 1, 2, 3 voxels
  5. Hausdorff Distance 95th percentile
  6. Boundary-relaxed Dice (dilate GT by 1, 2, 3 voxels)
  7. Sensitivity & Precision (voxel-level)

Usage:
    python scripts/lesionwise_eval.py
    python scripts/lesionwise_eval.py --tolerance 2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import (distance_transform_edt, binary_dilation,
                            binary_erosion, generate_binary_structure)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'model' / 'stacking_cache_v4'
OUTPUT_DIR = ROOT / 'model'
ALL_MODELS = ['exp1_8patch', 'exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']
TOP3 = ['exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']
THRESHOLD = 0.40  # Optimized for top3_avg


def get_splits():
    """Return (val_case_ids, all_case_ids) using same split as training."""
    all_cases = sorted([f.stem for f in CACHE_DIR.glob('*.npz')])
    random.seed(42)
    cases_shuffled = all_cases.copy()
    random.shuffle(cases_shuffled)
    n_val = int(len(cases_shuffled) * 0.15)
    val_cases = set(cases_shuffled[:n_val])
    return val_cases, all_cases


def load_case(case_id):
    data = np.load(CACHE_DIR / f'{case_id}.npz')
    mask = data['mask'].astype(np.float32)
    preds = {name: data[name].astype(np.float32) for name in ALL_MODELS}
    return preds, mask


# =============================================================================
# METRICS
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


def get_surfaces(binary_mask):
    """Extract surface voxels from a binary mask."""
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(binary_mask.astype(bool), structure=struct, iterations=1)
    return binary_mask.astype(bool) & ~eroded


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
    # Compute incrementally: dilate 1 → check, dilate 1 more → check, etc.
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
# EVALUATE A SET OF CASES
# =============================================================================

def evaluate_cases(case_ids, label):
    """Run full evaluation on a list of case IDs. Returns per-case metrics and summary."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: top3_avg @ {THRESHOLD} — {label} ({len(case_ids)} cases)")
    print(f"{'='*70}")

    case_metrics = []

    for i, case_id in enumerate(case_ids):
        t_case = time.time()
        preds, mask = load_case(case_id)

        # Top-3 average
        prob = np.mean([preds[m] for m in TOP3], axis=0)
        pred_bin = (prob > THRESHOLD).astype(np.float32)

        # Fast metrics first
        vd = voxel_dice(pred_bin, mask)
        vs = voxel_sensitivity(pred_bin, mask)
        vp = voxel_precision(pred_bin, mask)
        lf1 = lesionwise_f1(pred_bin, mask)
        pld = per_lesion_dice(pred_bin, mask)

        # All surface/boundary metrics in ONE pass (2 EDTs instead of 8)
        surf = compute_surface_metrics(pred_bin, mask)

        elapsed_case = time.time() - t_case
        if (i + 1) % 10 == 0 or (i + 1) == len(case_ids) or (i + 1) == 1:
            print(f"  [{i+1}/{len(case_ids)}] {case_id} — dice={vd:.3f}, "
                  f"lesF1={lf1['f1']:.3f}, surfD2={surf['surface_dice_2']:.3f} ({elapsed_case:.1f}s)")

        case_metrics.append({
            'case_id': case_id,
            'lesion_voxels': int(mask.sum()),
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
            # Surface metrics (from single-pass computation)
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

        # Progress is already printed above in the per-case loop

    # =========================================================================
    # Aggregate
    # =========================================================================
    valid = [c for c in case_metrics if c['lesion_voxels'] > 0]
    n = len(valid)

    summary = {'n_cases': n, 'n_total': len(case_ids), 'threshold': THRESHOLD, 'label': label}

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
        sorted_v = sorted(values)
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
    print(f"\n  === {label} — top3_avg @ {THRESHOLD} ({n} cases with foreground) ===\n")

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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full lesion-wise evaluation")
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help='Default surface dice tolerance (default: 2.0)')
    args = parser.parse_args()

    print("=" * 70)
    print("  Full Lesion-Wise Evaluation — top3_avg ensemble")
    print("=" * 70)

    val_case_set, all_case_ids = get_splits()
    val_case_ids = [c for c in all_case_ids if c in val_case_set]
    train_case_ids = [c for c in all_case_ids if c not in val_case_set]

    print(f"Total cached cases: {len(all_case_ids)}")
    print(f"Val split: {len(val_case_ids)} cases (unseen by base models)")
    print(f"Train split: {len(train_case_ids)} cases (seen during training — biased)")
    print(f"Method: top3_avg (exp3_12patch, improved_24patch, improved_36patch)")
    print(f"Threshold: {THRESHOLD}")

    t0 = time.time()

    # Evaluate val set
    val_summary, val_cases = evaluate_cases(val_case_ids, "VAL SET (unseen)")

    # Evaluate all cases
    all_summary, all_cases = evaluate_cases(all_case_ids, "ALL CASES (train+val)")

    # =========================================================================
    # Side-by-side comparison
    # =========================================================================
    print(f"\n{'='*70}")
    print("  SIDE-BY-SIDE: Val vs All")
    print(f"{'='*70}")

    compare_keys = [
        'voxel_dice', 'voxel_sensitivity', 'voxel_precision',
        'lesion_f1', 'lesion_recall', 'lesion_precision',
        'per_lesion_dice',
        'surface_dice_1', 'surface_dice_2', 'surface_dice_3',
        'relaxed_dice_1', 'relaxed_dice_2', 'relaxed_dice_3',
    ]

    print(f"\n  {'Metric':<25} {'Val Mean':<12} {'All Mean':<12} {'Delta':<10}")
    print(f"  {'-'*60}")
    for key in compare_keys:
        v = val_summary[key]['mean']
        a = all_summary[key]['mean']
        delta = a - v
        print(f"  {key:<25} {v:<12.4f} {a:<12.4f} {delta:+.4f}")

    vh = val_summary['hausdorff_95']
    ah = all_summary['hausdorff_95']
    print(f"  {'hausdorff_95':<25} {vh['mean']:<12.2f} {ah['mean']:<12.2f} {ah['mean']-vh['mean']:+.2f}")

    vld = val_summary['lesion_detection']
    ald = all_summary['lesion_detection']
    print(f"\n  Lesion F1:  Val={vld['overall_f1']:.3f}  All={ald['overall_f1']:.3f}")
    print(f"  Recall:     Val={vld['overall_recall']:.3f}  All={ald['overall_recall']:.3f}")
    print(f"  Precision:  Val={vld['overall_precision']:.3f}  All={ald['overall_precision']:.3f}")

    # =========================================================================
    # Worst cases across all
    # =========================================================================
    print(f"\n{'='*70}")
    print("  WORST 15 CASES (All, by voxel dice)")
    print(f"{'='*70}")
    print(f"  {'Case':<25} {'VoxDice':<9} {'LesF1':<8} {'PerLes':<8} {'SurfD2':<8} {'HD95':<8} {'Vox':<8} {'Split'}")
    print(f"  {'-'*85}")

    all_sorted = sorted(all_cases, key=lambda c: c['voxel_dice'])
    for c in all_sorted[:15]:
        split = "VAL" if c['case_id'] in val_case_set else "train"
        hd = f"{c['hausdorff_95']:.1f}" if c['hausdorff_95'] != float('inf') else "inf"
        print(f"  {c['case_id']:<25} {c['voxel_dice']:<9.4f} {c['lesion_f1']:<8.4f} "
              f"{c['per_lesion_dice']:<8.4f} {c['surface_dice_2']:<8.4f} "
              f"{hd:<8} {c['lesion_voxels']:<8} {split}")

    # =========================================================================
    # Save
    # =========================================================================
    save_data = {
        'val_set': {
            'summary': val_summary,
            'per_case': [{k: v for k, v in c.items() if k != 'missed_lesion_sizes'}
                         for c in val_cases],
        },
        'all_cases': {
            'summary': all_summary,
            'per_case': [{k: v for k, v in c.items() if k != 'missed_lesion_sizes'}
                         for c in all_cases],
        },
    }

    out_path = OUTPUT_DIR / 'lesionwise_evaluation.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} minutes")
    print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()
