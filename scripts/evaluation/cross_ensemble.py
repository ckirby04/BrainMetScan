"""
Cross-model ensemble evaluation: Lightweight ensemble + nnU-Net multi-fold.

Loads predictions from both systems and evaluates multiple fusion strategies
on validation cases. CPU-only (no GPU needed).

Data sources:
  - Lightweight: model/stacking_cache_v4/{case_id}.npz (top3 model probs)
  - nnU-Net probs: model/nnunet_probs/fold_X/{case_id}.npz (softmax probs)
  - nnU-Net binary: nnUNet/.../fold_X/validation/{case_id}.nii.gz (fallback)

Ensemble methods evaluated:
  1. lw_only      — lightweight top3_avg baseline
  2. nn_only      — nnU-Net baseline
  3. cross_avg    — weighted average of both probabilities
  4. cross_union  — positive if either system detects
  5. cross_intersection — positive if both detect
  6. cross_size_conditional — nnU-Net + small lightweight-only detections
  7. cross_confidence — weight by confidence (distance from 0.5)
  8. cross_optimized — scipy.optimize over [w_lw, w_nn, threshold]

Usage:
    python scripts/cross_ensemble.py                     # Full eval
    python scripts/cross_ensemble.py --predict           # Generate test predictions
    python scripts/cross_ensemble.py --nnunet-binary     # Use binary masks (no probs)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import (distance_transform_edt, binary_dilation,
                            binary_erosion, generate_binary_structure)
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'model'

# Lightweight ensemble
CACHE_DIR = ROOT / 'model' / 'stacking_cache_v4'
TOP3 = ['exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']

# nnU-Net
NNUNET_PROBS_DIR = ROOT / 'model' / 'nnunet_probs'
NNUNET_RESULTS_BASE = (ROOT / 'nnUNet' / 'nnUNet_results' / 'Dataset001_BrainMets'
                       / 'nnUNetTrainer__nnUNetPlans__3d_fullres')
LABELS_DIR = ROOT / 'nnUNet' / 'nnUNet_raw' / 'Dataset001_BrainMets' / 'labelsTr'
SPLITS_FILE = (ROOT / 'nnUNet' / 'nnUNet_preprocessed'
               / 'Dataset001_BrainMets' / 'splits_final.json')


# =============================================================================
# METRICS (same as evaluate_nnunet.py / lesionwise_eval.py)
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

    gt_sizes = []
    gt_det = []
    for i in range(1, n_gt + 1):
        gt_sizes.append(int((gt_labeled == i).sum()))
        gt_det.append(i in gt_detected)

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_gt': n_gt, 'n_pred': n_pred,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'gt_sizes': gt_sizes,
        'gt_detected': gt_det,
    }


def compute_surface_dice(pred_bin, gt_bin, tolerance=2.0):
    """Surface Dice at given tolerance. Returns 1.0 for both-empty."""
    pred_any = pred_bin.sum() > 0
    gt_any = gt_bin.sum() > 0
    if not pred_any and not gt_any:
        return 1.0
    if not pred_any or not gt_any:
        return 0.0

    struct = generate_binary_structure(3, 1)
    pred_bool = pred_bin.astype(bool)
    gt_bool = gt_bin.astype(bool)

    pred_surface = pred_bool & ~binary_erosion(pred_bool, structure=struct)
    gt_surface = gt_bool & ~binary_erosion(gt_bool, structure=struct)

    n_pred_s = int(pred_surface.sum())
    n_gt_s = int(gt_surface.sum())
    if n_pred_s == 0 or n_gt_s == 0:
        return 0.0

    gt_dist = distance_transform_edt(~gt_surface)
    pred_dist = distance_transform_edt(~pred_surface)

    pred_within = (gt_dist[pred_surface] <= tolerance).sum()
    gt_within = (pred_dist[gt_surface] <= tolerance).sum()

    return float(pred_within + gt_within) / float(n_pred_s + n_gt_s + 1e-8)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_val_cases():
    """Return val case IDs using same 15%/seed=42 split as top3_ensemble.py."""
    cached_cases = sorted([f.stem for f in CACHE_DIR.glob('*.npz')])
    random.seed(42)
    cases_shuffled = cached_cases.copy()
    random.shuffle(cases_shuffled)
    n_val = int(len(cases_shuffled) * 0.15)
    return cases_shuffled[:n_val]


def load_splits():
    """Load nnU-Net 5-fold split definitions."""
    with open(SPLITS_FILE) as f:
        return json.load(f)


def build_case_to_fold_map(splits):
    """Map each case_id -> which nnU-Net fold had it as validation (unseen)."""
    case_to_fold = {}
    for fold_idx, split in enumerate(splits):
        for case_id in split['val']:
            case_to_fold[case_id] = fold_idx
    return case_to_fold


def load_lw_probs(case_id):
    """Load lightweight ensemble probabilities for one case.

    Returns top3_avg probability map (float32).
    """
    path = CACHE_DIR / f'{case_id}.npz'
    if not path.exists():
        return None
    data = np.load(path)
    probs = [data[m].astype(np.float32) for m in TOP3]
    return np.mean(probs, axis=0)


def load_nn_probs(case_id, fold_idx, use_binary=False):
    """Load nnU-Net probability for one case.

    Tries .npz prob file first, falls back to binary .nii.gz.
    Returns float32 probability map.
    """
    if not use_binary:
        npz_path = NNUNET_PROBS_DIR / f'fold_{fold_idx}' / f'{case_id}.npz'
        if npz_path.exists():
            data = np.load(npz_path)
            # Foreground probability = channel 1
            # Transpose (2,1,0) to convert from nnU-Net internal axis order to NIfTI RAS
            return data['probabilities'][1].transpose(2, 1, 0).astype(np.float32)

    # Binary fallback
    nifti_path = NNUNET_RESULTS_BASE / f'fold_{fold_idx}' / 'validation' / f'{case_id}.nii.gz'
    if nifti_path.exists():
        img = nib.load(str(nifti_path))
        return np.asarray(img.dataobj, dtype=np.float32)

    return None


def load_gt(case_id):
    """Load ground truth segmentation mask."""
    gt_path = LABELS_DIR / f'{case_id}.nii.gz'
    if not gt_path.exists():
        return None
    img = nib.load(str(gt_path))
    return (np.asarray(img.dataobj, dtype=np.float32) > 0.5).astype(np.float32)


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================

def method_lw_only(lw_prob, nn_prob, threshold):
    """Lightweight-only baseline."""
    return (lw_prob > threshold).astype(np.float32)


def method_nn_only(lw_prob, nn_prob, threshold):
    """nnU-Net-only baseline."""
    return (nn_prob > threshold).astype(np.float32)


def method_cross_avg(lw_prob, nn_prob, threshold, weight=0.5):
    """Weighted average: w * lw + (1-w) * nn."""
    combined = weight * lw_prob + (1.0 - weight) * nn_prob
    return (combined > threshold).astype(np.float32)


def method_cross_union(lw_prob, nn_prob, threshold):
    """Union: positive if either system detects at threshold."""
    lw_bin = lw_prob > threshold
    nn_bin = nn_prob > threshold
    return (lw_bin | nn_bin).astype(np.float32)


def method_cross_intersection(lw_prob, nn_prob, threshold):
    """Intersection: positive if both systems detect at threshold."""
    lw_bin = lw_prob > threshold
    nn_bin = nn_prob > threshold
    return (lw_bin & nn_bin).astype(np.float32)


def method_cross_size_conditional(lw_prob, nn_prob, threshold, size_thresh=200):
    """nnU-Net mask + small lightweight-only detections that nnU-Net missed.

    Uses nnU-Net as base, then adds connected components from lightweight
    that don't overlap with nnU-Net predictions and are below size_thresh.
    This captures tiny lesions that lightweight detects but nnU-Net misses.
    """
    nn_bin = (nn_prob > threshold).astype(np.float32)
    lw_bin = (lw_prob > threshold).astype(np.float32)

    # Find lightweight-only detections (not in nnU-Net)
    lw_only = lw_bin * (1.0 - nn_bin)

    if lw_only.sum() == 0:
        return nn_bin

    # Get connected components of lightweight-only detections
    labeled, n_cc = ndimage_label(lw_only > 0.5)
    result = nn_bin.copy()

    for i in range(1, n_cc + 1):
        cc = (labeled == i)
        cc_size = cc.sum()
        if cc_size <= size_thresh:
            result[cc] = 1.0

    return result


def method_cross_confidence(lw_prob, nn_prob, threshold):
    """Weight each voxel by confidence (distance from 0.5)."""
    lw_conf = np.abs(lw_prob - 0.5)
    nn_conf = np.abs(nn_prob - 0.5)
    total_conf = lw_conf + nn_conf + 1e-8
    combined = (lw_conf * lw_prob + nn_conf * nn_prob) / total_conf
    return (combined > threshold).astype(np.float32)


# =============================================================================
# EVALUATION
# =============================================================================

def fast_threshold_sweep(prob_fn, val_data, thresholds, **kwargs):
    """Fast threshold sweep using histogram binning.

    prob_fn(lw, nn, **kwargs) -> probability map (no thresholding).
    Evaluates all thresholds simultaneously per case via histogram bins.
    """
    n_bins = 10000
    n_thresh = len(thresholds)
    thresh_bins = np.minimum((np.asarray(thresholds) * n_bins).astype(int),
                             n_bins - 1)

    dice_sums = np.zeros(n_thresh, dtype=np.float64)
    n_valid = 0

    for case_id, lw, nn, gt in val_data:
        gt_mask = gt > 0
        n_fg = int(gt_mask.sum())
        if n_fg == 0:
            continue
        n_valid += 1

        prob = prob_fn(lw, nn, **kwargs)

        # Histogram foreground and total probabilities
        fg_hist = np.histogram(prob[gt_mask], bins=n_bins, range=(0.0, 1.0))[0]
        all_hist = np.histogram(prob.ravel(), bins=n_bins, range=(0.0, 1.0))[0]
        bg_hist = all_hist - fg_hist

        # Cumulative from right: count of voxels >= each bin
        fg_above = np.cumsum(fg_hist[::-1])[::-1]
        bg_above = np.cumsum(bg_hist[::-1])[::-1]

        # Dice for all thresholds at once
        tp = fg_above[thresh_bins].astype(np.float64)
        fp = bg_above[thresh_bins].astype(np.float64)
        fn = n_fg - tp
        dice_vals = np.where(
            (tp == 0) & (fp == 0) & (fn == 0),
            1.0,
            2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
        )
        dice_sums += dice_vals

    if n_valid == 0:
        return 0.0, {'mean_dice': 0, 'threshold': 0, 'n_cases': 0}

    mean_dices = dice_sums / n_valid
    best_idx = int(np.argmax(mean_dices))

    return float(thresholds[best_idx]), {
        'mean_dice': float(mean_dices[best_idx]),
        'threshold': float(thresholds[best_idx]),
        'n_cases': n_valid,
    }


def evaluate_method(method_fn, val_data, thresholds, **kwargs):
    """Sweep thresholds for a method (slow path for CC-based methods)."""
    best_dice = -1
    best_thresh = 0
    best_metrics = None

    for thresh in thresholds:
        dices = []
        for case_id, lw_prob, nn_prob, gt in val_data:
            if gt.sum() == 0:
                continue
            pred_bin = method_fn(lw_prob, nn_prob, thresh, **kwargs)
            dices.append(voxel_dice(pred_bin, gt))

        if dices:
            mean_dice = np.mean(dices)
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_thresh = thresh
                best_metrics = {
                    'mean_dice': float(mean_dice),
                    'median_dice': float(np.median(dices)),
                    'threshold': float(thresh),
                    'n_cases': len(dices),
                }

    return best_thresh, best_metrics


# Probability-only functions (no thresholding) for fast_threshold_sweep
def prob_lw(lw, nn, **kw):
    return lw

def prob_nn(lw, nn, **kw):
    return nn

def prob_avg(lw, nn, weight=0.5, **kw):
    return weight * lw + (1.0 - weight) * nn

def prob_union(lw, nn, **kw):
    return np.maximum(lw, nn)

def prob_inter(lw, nn, **kw):
    return np.minimum(lw, nn)

def prob_conf(lw, nn, **kw):
    lw_c = np.abs(lw - 0.5)
    nn_c = np.abs(nn - 0.5)
    return (lw_c * lw + nn_c * nn) / (lw_c + nn_c + 1e-8)


def full_evaluate(method_fn, val_data, threshold, compute_sd=False, **kwargs):
    """Full evaluation at a specific threshold. Returns detailed metrics."""
    results = []
    for i, (case_id, lw_prob, nn_prob, gt) in enumerate(val_data):
        if gt.sum() == 0:
            continue

        pred_bin = method_fn(lw_prob, nn_prob, threshold, **kwargs)
        vd = voxel_dice(pred_bin, gt)
        vs = voxel_sensitivity(pred_bin, gt)
        vp = voxel_precision(pred_bin, gt)
        lf = lesionwise_f1(pred_bin, gt)
        sd2 = compute_surface_dice(pred_bin, gt, tolerance=2.0) if compute_sd else 0.0

        results.append({
            'case_id': case_id,
            'lesion_voxels': int(gt.sum()),
            'voxel_dice': vd,
            'voxel_sensitivity': vs,
            'voxel_precision': vp,
            'lesion_f1': lf['f1'],
            'lesion_recall': lf['recall'],
            'lesion_precision': lf['precision'],
            'lesion_tp': lf['tp'],
            'lesion_fp': lf['fp'],
            'lesion_fn': lf['fn'],
            'n_gt': lf['n_gt'],
            'n_pred': lf['n_pred'],
            'surface_dice_2': sd2,
            'gt_sizes': lf['gt_sizes'],
            'gt_detected': lf['gt_detected'],
        })

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(val_data)}]...")

    return results


def summarize(results, label):
    """Aggregate per-case metrics into a summary."""
    n = len(results)
    if n == 0:
        return {'label': label, 'n_cases': 0}

    summary = {'label': label, 'n_cases': n}

    for key in ['voxel_dice', 'voxel_sensitivity', 'voxel_precision',
                'lesion_f1', 'lesion_recall', 'lesion_precision', 'surface_dice_2']:
        values = [r[key] for r in results]
        summary[key] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
        }

    # Lesion detection totals
    total_tp = sum(r['lesion_tp'] for r in results)
    total_fp = sum(r['lesion_fp'] for r in results)
    total_fn = sum(r['lesion_fn'] for r in results)
    total_gt = sum(r['n_gt'] for r in results)
    summary['lesion_detection'] = {
        'total_gt': total_gt,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'overall_recall': float(total_tp / (total_tp + total_fn + 1e-8)),
        'overall_precision': float(total_tp / (total_tp + total_fp + 1e-8)),
        'overall_f1': float(2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)),
    }

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
        bucket = [r for r in results if lo <= r['lesion_voxels'] < hi]
        if bucket:
            summary['dice_by_size'][bucket_label] = {
                'n': len(bucket),
                'mean_dice': float(np.mean([r['voxel_dice'] for r in bucket])),
            }

    # Missed lesion analysis
    all_missed = []
    for r in results:
        for size, detected in zip(r['gt_sizes'], r['gt_detected']):
            if not detected:
                all_missed.append(size)
    summary['missed_lesions'] = {
        'total': len(all_missed),
        'by_size': {
            '<50': sum(1 for s in all_missed if s < 50),
            '50-200': sum(1 for s in all_missed if 50 <= s < 200),
            '200-1k': sum(1 for s in all_missed if 200 <= s < 1000),
            '>1k': sum(1 for s in all_missed if s >= 1000),
        }
    } if all_missed else {'total': 0}

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model ensemble: lightweight + nnU-Net")
    parser.add_argument('--predict', action='store_true',
                        help="Generate test predictions with best method")
    parser.add_argument('--nnunet-binary', action='store_true',
                        help="Use binary nnU-Net masks (skip probability files)")
    parser.add_argument('--quick', action='store_true',
                        help="Quick mode: coarser threshold sweep (0.05 steps)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Cross-Model Ensemble: Lightweight + nnU-Net")
    print("=" * 70)

    t0 = time.time()

    # Load splits and build case-to-fold mapping
    splits = load_splits()
    case_to_fold = build_case_to_fold_map(splits)

    # Get val cases (same split as top3_ensemble.py)
    val_cases = get_val_cases()
    print(f"Val cases: {len(val_cases)}")

    # Check which folds are available
    available_folds = set()
    for fold_idx in range(5):
        fold_dir = NNUNET_PROBS_DIR / f'fold_{fold_idx}'
        binary_dir = NNUNET_RESULTS_BASE / f'fold_{fold_idx}' / 'validation'
        if fold_dir.exists() or binary_dir.exists():
            available_folds.add(fold_idx)
    print(f"Available nnU-Net folds: {sorted(available_folds)}")

    # =========================================================================
    # Load validation data (streaming, case-by-case)
    # =========================================================================
    print(f"\n  Loading validation data...")
    val_data = []  # list of (case_id, lw_prob, nn_prob, gt)
    n_lw_only = 0
    n_nn_only = 0
    n_both = 0
    n_skipped = 0

    for i, case_id in enumerate(val_cases):
        # Load lightweight probs
        lw_prob = load_lw_probs(case_id)
        if lw_prob is None:
            n_skipped += 1
            continue

        # Load ground truth
        gt = load_gt(case_id)
        if gt is None:
            n_skipped += 1
            continue

        # Find which nnU-Net fold has this case as validation
        fold_idx = case_to_fold.get(case_id)
        nn_prob = None
        if fold_idx is not None and fold_idx in available_folds:
            nn_prob = load_nn_probs(case_id, fold_idx, use_binary=args.nnunet_binary)

        if nn_prob is not None:
            # Ensure shapes match
            if lw_prob.shape != nn_prob.shape:
                print(f"    WARNING: Shape mismatch for {case_id}: "
                      f"lw={lw_prob.shape} nn={nn_prob.shape}, skipping")
                n_skipped += 1
                continue
            n_both += 1
        else:
            n_lw_only += 1

        val_data.append((case_id, lw_prob, nn_prob, gt))

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(val_cases)}] loaded...")

    print(f"  Loaded: {n_both} with both systems, "
          f"{n_lw_only} lightweight-only, {n_skipped} skipped")

    # Separate cases with both predictions vs lightweight-only
    both_data = [(c, lw, nn, gt) for c, lw, nn, gt in val_data if nn is not None]
    all_data_with_placeholder = []
    for c, lw, nn, gt in val_data:
        if nn is None:
            # Use zeros as placeholder for nnU-Net (won't affect lw_only baseline)
            nn = np.zeros_like(lw)
        all_data_with_placeholder.append((c, lw, nn, gt))

    if not both_data:
        print("\nERROR: No cases found with both lightweight and nnU-Net predictions!")
        print("Run 'python scripts/nnunet_probs.py --mode val' first, "
              "or use --nnunet-binary flag.")
        sys.exit(1)

    print(f"\n  Cross-model evaluation on {len(both_data)} cases with both predictions")

    # =========================================================================
    # Threshold sweep
    # =========================================================================
    if args.quick:
        thresholds = np.arange(0.20, 0.65, 0.05)
    else:
        thresholds = np.arange(0.20, 0.65, 0.01)

    print(f"\n{'='*70}")
    print(f"  STEP 1: Threshold Sweep ({len(thresholds)} thresholds)")
    print(f"{'='*70}")

    all_results = {}

    # --- Baselines (fast histogram sweep) ---
    print("\n  Baselines:")
    t_method = time.time()
    _, lw_best = fast_threshold_sweep(prob_lw, both_data, thresholds)
    all_results['lw_only'] = lw_best
    print(f"    lw_only:  dice={lw_best['mean_dice']:.4f} "
          f"@ {lw_best['threshold']:.2f}  ({time.time()-t_method:.0f}s)")

    t_method = time.time()
    _, nn_best = fast_threshold_sweep(prob_nn, both_data, thresholds)
    all_results['nn_only'] = nn_best
    print(f"    nn_only:  dice={nn_best['mean_dice']:.4f} "
          f"@ {nn_best['threshold']:.2f}  ({time.time()-t_method:.0f}s)")

    # --- Cross-model methods (fast) ---
    print("\n  Cross-model methods:")

    # Cross avg with weight sweep
    t_method = time.time()
    best_avg_dice = -1
    best_avg_w = 0.5
    best_avg_result = None
    for w in np.arange(0.1, 0.95, 0.05):
        _, result = fast_threshold_sweep(
            prob_avg, both_data, thresholds, weight=w)
        if result and result['mean_dice'] > best_avg_dice:
            best_avg_dice = result['mean_dice']
            best_avg_w = w
            best_avg_result = result
    if best_avg_result:
        best_avg_result['weight'] = float(best_avg_w)
        all_results['cross_avg'] = best_avg_result
        print(f"    cross_avg: dice={best_avg_dice:.4f} "
              f"@ w={best_avg_w:.2f}, t={best_avg_result['threshold']:.2f}"
              f"  ({time.time()-t_method:.0f}s)")

    # Union = max(lw, nn) > threshold
    t_method = time.time()
    _, union_result = fast_threshold_sweep(prob_union, both_data, thresholds)
    all_results['cross_union'] = union_result
    print(f"    cross_union: dice={union_result['mean_dice']:.4f} "
          f"@ {union_result['threshold']:.2f}  ({time.time()-t_method:.0f}s)")

    # Intersection = min(lw, nn) > threshold
    t_method = time.time()
    _, inter_result = fast_threshold_sweep(prob_inter, both_data, thresholds)
    all_results['cross_intersection'] = inter_result
    print(f"    cross_intersection: dice={inter_result['mean_dice']:.4f} "
          f"@ {inter_result['threshold']:.2f}  ({time.time()-t_method:.0f}s)")

    # Size-conditional (uses CC — very slow, only evaluate at best nn threshold)
    t_method = time.time()
    nn_thresh = nn_best['threshold']
    best_sc_dice = -1
    best_sc_size = 200
    best_sc_result = None
    sc_thresholds = np.array([nn_thresh])  # Just evaluate at best nn threshold
    for size_t in [100, 500, 2000]:
        _, result = evaluate_method(
            method_cross_size_conditional, both_data, sc_thresholds,
            size_thresh=size_t)
        if result and result['mean_dice'] > best_sc_dice:
            best_sc_dice = result['mean_dice']
            best_sc_size = size_t
            best_sc_result = result
    if best_sc_result:
        best_sc_result['size_threshold'] = best_sc_size
        all_results['cross_size_conditional'] = best_sc_result
        print(f"    cross_size_cond: dice={best_sc_dice:.4f} "
              f"@ size={best_sc_size}, t={best_sc_result['threshold']:.2f}"
              f"  ({time.time()-t_method:.0f}s)")

    # Confidence-weighted
    t_method = time.time()
    _, conf_result = fast_threshold_sweep(prob_conf, both_data, thresholds)
    all_results['cross_confidence'] = conf_result
    print(f"    cross_confidence: dice={conf_result['mean_dice']:.4f} "
          f"@ {conf_result['threshold']:.2f}  ({time.time()-t_method:.0f}s)")

    # =========================================================================
    # Scipy optimization
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  STEP 2: Scipy Optimization")
    print(f"{'='*70}")

    # Subsample cases for faster optimization (25 random cases)
    opt_all = [(lw, nn, gt > 0, int((gt > 0).sum()))
               for _, lw, nn, gt in both_data if gt.sum() > 0]
    rng = np.random.RandomState(42)
    opt_idx = rng.choice(len(opt_all), size=min(25, len(opt_all)), replace=False)
    opt_cases = [opt_all[i] for i in opt_idx]
    print(f"  Optimizing on {len(opt_cases)} subsampled cases...")

    def neg_mean_dice(params):
        w_lw, w_nn, thresh = params
        w_lw = max(0.01, min(0.99, w_lw))
        w_nn = max(0.01, min(0.99, w_nn))
        thresh = max(0.1, min(0.8, thresh))
        total = w_lw + w_nn
        w_lw, w_nn = w_lw / total, w_nn / total

        dices = []
        for lw, nn, gt_mask, n_fg in opt_cases:
            pred = (w_lw * lw + w_nn * nn) > thresh
            tp = int((pred & gt_mask).sum())
            fp = int(pred.sum()) - tp
            fn = n_fg - tp
            if tp == 0 and fp == 0 and fn == 0:
                dices.append(1.0)
            else:
                dices.append(float(2 * tp) / float(2 * tp + fp + fn + 1e-8))
        return -np.mean(dices)

    # Start from best cross_avg result
    starting_points = [
        [best_avg_w, 1.0 - best_avg_w,
         best_avg_result['threshold'] if best_avg_result else 0.40],
        [0.3, 0.7, 0.35],
        [0.5, 0.5, 0.40],
    ]

    best_opt = None
    for i, x0 in enumerate(starting_points):
        t_opt = time.time()
        result = minimize(
            neg_mean_dice, x0,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 0.005, 'fatol': 0.0005},
        )
        print(f"    Start {i+1}: dice={-result.fun:.4f} "
              f"({result.nfev} evals, {time.time()-t_opt:.0f}s)")
        if best_opt is None or result.fun < best_opt.fun:
            best_opt = result

    opt_w_lw = best_opt.x[0]
    opt_w_nn = best_opt.x[1]
    total = opt_w_lw + opt_w_nn
    opt_w_lw /= total
    opt_w_nn /= total
    opt_thresh = best_opt.x[2]

    print(f"\n  Optimized weights: lw={opt_w_lw:.4f}, nn={opt_w_nn:.4f}")
    print(f"  Optimized threshold: {opt_thresh:.4f}")
    print(f"  Subsampled Dice: {-best_opt.fun:.4f}")

    # Validate optimized params on ALL cases using fast sweep at nearby thresholds
    opt_thresholds = np.array([opt_thresh])
    _, opt_full = fast_threshold_sweep(
        prob_avg, both_data, opt_thresholds, weight=opt_w_lw)
    opt_dice = opt_full['mean_dice']
    print(f"  Full-dataset Dice: {opt_dice:.4f}")

    all_results['cross_optimized'] = {
        'mean_dice': float(opt_dice),
        'threshold': float(opt_thresh),
        'weight_lw': float(opt_w_lw),
        'weight_nn': float(opt_w_nn),
        'n_cases': len(opt_all),
    }

    # =========================================================================
    # Full evaluation of top methods
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  STEP 3: Full Evaluation of Top Methods")
    print(f"{'='*70}")

    # Sort by dice, pick top 2 + baselines (limit to avoid hour-long evals)
    ranked = sorted(all_results.items(), key=lambda x: -x[1]['mean_dice'])
    top_methods = []
    for name, info in ranked[:2]:
        top_methods.append(name)
    # Always include baselines
    for baseline in ['lw_only', 'nn_only']:
        if baseline not in top_methods:
            top_methods.append(baseline)

    full_results = {}

    for method_name in top_methods:
        info = all_results[method_name]
        thresh = info['threshold']

        # Build the method function with the right kwargs
        if method_name == 'lw_only':
            fn = method_lw_only
            kw = {}
        elif method_name == 'nn_only':
            fn = method_nn_only
            kw = {}
        elif method_name == 'cross_avg':
            fn = method_cross_avg
            kw = {'weight': info.get('weight', 0.5)}
        elif method_name == 'cross_union':
            fn = method_cross_union
            kw = {}
        elif method_name == 'cross_intersection':
            fn = method_cross_intersection
            kw = {}
        elif method_name == 'cross_size_conditional':
            fn = method_cross_size_conditional
            kw = {'size_thresh': info.get('size_threshold', 200)}
        elif method_name == 'cross_confidence':
            fn = method_cross_confidence
            kw = {}
        elif method_name == 'cross_optimized':
            w_lw = info['weight_lw']
            w_nn = info['weight_nn']
            fn = method_cross_avg
            kw = {'weight': w_lw}
            # cross_optimized is just cross_avg with optimized weight
        else:
            continue

        t_eval = time.time()
        print(f"\n  Evaluating: {method_name} @ {thresh:.2f}...")
        per_case = full_evaluate(fn, both_data, thresh, compute_sd=False, **kw)
        print(f"    Done ({time.time()-t_eval:.0f}s)")
        summary = summarize(per_case, method_name)
        summary['threshold'] = thresh
        summary.update({k: v for k, v in info.items()
                        if k not in ['mean_dice', 'threshold', 'n_cases']})

        full_results[method_name] = {
            'summary': summary,
            'per_case': [{k: v for k, v in r.items()
                          if k not in ['gt_sizes', 'gt_detected']}
                         for r in per_case],
        }

    # =========================================================================
    # Print comparison table
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  RESULTS COMPARISON ({len(both_data)} cases)")
    print(f"{'='*70}")

    # Header
    header_metrics = ['voxel_dice', 'voxel_sensitivity', 'voxel_precision',
                      'lesion_f1', 'surface_dice_2']
    print(f"\n  {'Method':<25} {'Thresh':<7}", end='')
    for m in header_metrics:
        short = m.replace('voxel_', '').replace('lesion_', 'les_').replace('surface_dice_', 'SD')
        print(f" {short:<8}", end='')
    print()
    print(f"  {'-'*75}")

    for name, info in ranked:
        thresh = info['threshold']
        if name in full_results:
            s = full_results[name]['summary']
            print(f"  {name:<25} {thresh:<7.2f}", end='')
            for m in header_metrics:
                val = s.get(m, {}).get('mean', info.get('mean_dice', 0))
                print(f" {val:<8.4f}", end='')
            # Marker for best
            if name == ranked[0][0]:
                print(" <-- BEST", end='')
            print()
        else:
            print(f"  {name:<25} {thresh:<7.2f} {info['mean_dice']:<8.4f}")

    # Lesion detection details
    print(f"\n  Lesion Detection:")
    print(f"  {'Method':<25} {'Recall':<9} {'Precision':<11} {'F1':<8} {'TP':<6} {'FP':<6} {'FN':<6}")
    print(f"  {'-'*70}")
    for name in top_methods:
        if name not in full_results:
            continue
        ld = full_results[name]['summary'].get('lesion_detection', {})
        print(f"  {name:<25} {ld.get('overall_recall',0):<9.4f} "
              f"{ld.get('overall_precision',0):<11.4f} "
              f"{ld.get('overall_f1',0):<8.4f} "
              f"{ld.get('total_tp',0):<6} {ld.get('total_fp',0):<6} "
              f"{ld.get('total_fn',0):<6}")

    # Dice by size
    print(f"\n  Dice by Lesion Size:")
    buckets = ['tiny (<100)', 'small (100-500)', 'medium (500-5k)',
               'large (5k-20k)', 'huge (>20k)']
    print(f"  {'Bucket':<18}", end='')
    for name in top_methods[:4]:
        print(f" {name[:12]:<13}", end='')
    print()
    print(f"  {'-'*70}")
    for bucket in buckets:
        print(f"  {bucket:<18}", end='')
        for name in top_methods[:4]:
            if name not in full_results:
                print(f" {'---':<13}", end='')
                continue
            bs = full_results[name]['summary'].get('dice_by_size', {})
            val = bs.get(bucket, {}).get('mean_dice')
            if val is not None:
                print(f" {val:<13.4f}", end='')
            else:
                print(f" {'---':<13}", end='')
        print()

    # =========================================================================
    # Improvement analysis
    # =========================================================================
    best_name = ranked[0][0]
    best_dice = ranked[0][1]['mean_dice']
    lw_dice = all_results['lw_only']['mean_dice']
    nn_dice = all_results['nn_only']['mean_dice']

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Best method: {best_name}")
    print(f"  Best Dice: {best_dice:.4f}")
    print(f"  vs lightweight: {best_dice - lw_dice:+.4f}")
    print(f"  vs nnU-Net:     {best_dice - nn_dice:+.4f}")
    if best_dice > lw_dice and best_dice > nn_dice:
        print(f"  >> Cross-model ensemble IMPROVES over both baselines!")
    elif best_dice > max(lw_dice, nn_dice):
        print(f"  >> Cross-model ensemble improves over the best baseline")
    else:
        print(f"  >> No improvement from cross-model fusion on these cases")

    # =========================================================================
    # Save results
    # =========================================================================
    save_data = {
        'config': {
            'n_val_cases': len(val_cases),
            'n_both_systems': len(both_data),
            'n_lw_only': n_lw_only,
            'available_folds': sorted(available_folds),
            'nnunet_binary': args.nnunet_binary,
        },
        'method_ranking': [
            {'method': name, **info} for name, info in ranked
        ],
        'full_results': {
            name: data['summary'] for name, data in full_results.items()
        },
        'per_case': {
            name: data['per_case'] for name, data in full_results.items()
        },
    }

    out_path = OUTPUT_DIR / 'cross_ensemble_results.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved results to: {out_path}")

    # =========================================================================
    # Generate test predictions (if --predict)
    # =========================================================================
    if args.predict:
        print(f"\n{'='*70}")
        print(f"  GENERATING TEST PREDICTIONS")
        print(f"{'='*70}")
        generate_test_predictions(
            best_name, all_results[best_name], args.nnunet_binary)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} minutes")


def generate_test_predictions(method_name, method_info, use_binary):
    """Generate test set predictions using the best ensemble method."""
    test_output = ROOT / 'outputs' / 'cross_ensemble'
    test_output.mkdir(parents=True, exist_ok=True)

    # Determine method parameters
    thresh = method_info['threshold']
    weight = method_info.get('weight', method_info.get('weight_lw', 0.5))

    print(f"  Method: {method_name}")
    print(f"  Threshold: {thresh:.4f}")
    if 'weight' in method_info or 'weight_lw' in method_info:
        print(f"  Weight: {weight:.4f}")

    # Load test case IDs from stacking cache (test cases have no masks)
    lw_test_dir = CACHE_DIR
    nn_test_dir = NNUNET_PROBS_DIR / 'test'

    # Get test case list from nnU-Net test images
    images_ts = ROOT / 'nnUNet' / 'nnUNet_raw' / 'Dataset001_BrainMets' / 'imagesTs'
    test_files = sorted(images_ts.glob('*_0000.nii.gz'))
    test_cases = [f.name.replace('_0000.nii.gz', '') for f in test_files]

    if not test_cases:
        print("  ERROR: No test cases found")
        return

    print(f"  Test cases: {len(test_cases)}")

    # Get a reference NIfTI for affine/header
    ref_nifti = nib.load(str(test_files[0]))
    ref_affine = ref_nifti.affine
    ref_header = ref_nifti.header

    n_saved = 0
    n_lw_only = 0
    n_skipped = 0

    for case_id in test_cases:
        # Load lightweight probs (test cases are in stacking cache too)
        lw_path = lw_test_dir / f'{case_id}.npz'
        lw_prob = None
        if lw_path.exists():
            data = np.load(lw_path)
            probs = [data[m].astype(np.float32) for m in TOP3
                     if m in data]
            if probs:
                lw_prob = np.mean(probs, axis=0)

        # Load nnU-Net probs
        nn_prob = None
        nn_path = nn_test_dir / f'{case_id}.npz'
        if nn_path.exists():
            data = np.load(nn_path)
            nn_prob = data['probabilities'][1].astype(np.float32)

        if lw_prob is None and nn_prob is None:
            n_skipped += 1
            continue

        # Apply ensemble method
        if lw_prob is not None and nn_prob is not None:
            if method_name in ('cross_avg', 'cross_optimized'):
                combined = weight * lw_prob + (1.0 - weight) * nn_prob
                pred_bin = (combined > thresh).astype(np.uint8)
            elif method_name == 'cross_union':
                pred_bin = ((lw_prob > thresh) | (nn_prob > thresh)).astype(np.uint8)
            elif method_name == 'cross_intersection':
                pred_bin = ((lw_prob > thresh) & (nn_prob > thresh)).astype(np.uint8)
            elif method_name == 'cross_size_conditional':
                size_t = method_info.get('size_threshold', 200)
                pred_bin = method_cross_size_conditional(
                    lw_prob, nn_prob, thresh, size_thresh=size_t).astype(np.uint8)
            elif method_name == 'cross_confidence':
                pred_bin = method_cross_confidence(
                    lw_prob, nn_prob, thresh).astype(np.uint8)
            elif method_name == 'nn_only':
                pred_bin = (nn_prob > thresh).astype(np.uint8)
            else:  # lw_only or fallback
                pred_bin = (lw_prob > thresh).astype(np.uint8)
        elif lw_prob is not None:
            pred_bin = (lw_prob > thresh).astype(np.uint8)
            n_lw_only += 1
        else:
            pred_bin = (nn_prob > thresh).astype(np.uint8)

        # Save as NIfTI
        # Load the actual test case NIfTI for correct affine
        case_ref = images_ts / f'{case_id}_0000.nii.gz'
        if case_ref.exists():
            case_nii = nib.load(str(case_ref))
            affine = case_nii.affine
        else:
            affine = ref_affine

        out_nii = nib.Nifti1Image(pred_bin, affine)
        out_path = test_output / f'{case_id}.nii.gz'
        nib.save(out_nii, str(out_path))
        n_saved += 1

    print(f"  Saved {n_saved} predictions to {test_output}")
    if n_lw_only > 0:
        print(f"  ({n_lw_only} cases used lightweight-only — no nnU-Net test probs)")
    if n_skipped > 0:
        print(f"  ({n_skipped} cases skipped — no predictions available)")


if __name__ == '__main__':
    main()
