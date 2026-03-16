"""
Per-size-category threshold optimization and post-processing analysis (CPU only).

Finds optimal thresholds per lesion size bucket to maximize detection
of small metastases while maintaining precision on large ones.

Also tests morphological post-processing: connected component filtering,
hole filling, and conditional filtering based on prediction confidence.

Usage:
    python scripts/threshold_analysis.py
    python scripts/threshold_analysis.py --granularity 0.01
    python scripts/threshold_analysis.py --output model/threshold_analysis.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import (label as ndimage_label, binary_fill_holes,
                            binary_dilation, generate_binary_structure)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'model' / 'stacking_cache_v4'
ALL_MODELS = ['exp1_8patch', 'exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']
TOP3 = ['exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']


def get_val_data():
    """Load all validation cases into memory."""
    cached = sorted([f.stem for f in CACHE_DIR.glob('*.npz')])
    if not cached:
        print("No cached predictions found!")
        return [], []

    random.seed(42)
    shuffled = cached.copy()
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * 0.15)
    val_ids = shuffled[:n_val]

    val_data = []
    for case_id in val_ids:
        data = np.load(CACHE_DIR / f'{case_id}.npz')
        mask = data['mask'].astype(np.float32)
        if mask.sum() == 0:
            continue

        preds = {name: data[name].astype(np.float32) for name in ALL_MODELS}
        prob = np.mean([preds[m] for m in TOP3], axis=0)
        lesion_voxels = int(mask.sum())

        val_data.append({
            'case_id': case_id,
            'prob': prob,
            'mask': mask,
            'preds': preds,
            'lesion_voxels': lesion_voxels,
        })

    return val_data, val_ids


def dice_score(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin > 0)).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-8)


def lesion_recall(pred_bin, gt_bin):
    """Fraction of GT lesions that overlap with at least 1 predicted voxel."""
    gt_labeled, n_gt = ndimage_label(gt_bin)
    if n_gt == 0:
        return 1.0
    detected = 0
    for i in range(1, n_gt + 1):
        gt_mask = (gt_labeled == i)
        if (pred_bin[gt_mask] > 0).any():
            detected += 1
    return detected / n_gt


def postprocess(binary_mask, min_size):
    """Remove connected components smaller than min_size."""
    labeled, n = ndimage_label(binary_mask)
    if n == 0:
        return binary_mask
    result = np.zeros_like(binary_mask)
    for i in range(1, n + 1):
        comp = (labeled == i)
        if comp.sum() >= min_size:
            result[comp] = 1
    return result


# =============================================================================
# ANALYSIS 1: Global threshold sweep (fine-grained)
# =============================================================================

def global_threshold_sweep(val_data, granularity=0.01):
    """Fine-grained global threshold sweep."""
    print(f"\n{'='*60}")
    print("  ANALYSIS 1: Global Threshold Sweep")
    print(f"{'='*60}")

    thresholds = np.arange(0.10, 0.80, granularity)

    results = {}
    for t in thresholds:
        dices = []
        recalls = []
        for case in val_data:
            pred_bin = (case['prob'] > t).astype(np.float32)
            dices.append(dice_score(pred_bin, case['mask']))
            recalls.append(lesion_recall(pred_bin, case['mask']))
        results[float(t)] = {
            'mean_dice': float(np.mean(dices)),
            'median_dice': float(np.median(dices)),
            'mean_lesion_recall': float(np.mean(recalls)),
        }

    # Find optimal
    best_t = max(results, key=lambda t: results[t]['mean_dice'])
    best = results[best_t]

    print(f"\n  Best threshold: {best_t:.3f}")
    print(f"  Mean Dice:          {best['mean_dice']:.4f}")
    print(f"  Median Dice:        {best['median_dice']:.4f}")
    print(f"  Mean lesion recall: {best['mean_lesion_recall']:.4f}")

    # Show recall-dice tradeoff at key thresholds
    print(f"\n  {'Threshold':<12} {'Mean Dice':<12} {'Median Dice':<14} {'Lesion Recall':<14}")
    print(f"  {'-'*52}")
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        if t in results:
            r = results[t]
            marker = " <-- best" if abs(t - best_t) < granularity else ""
            print(f"  {t:<12.2f} {r['mean_dice']:<12.4f} {r['median_dice']:<14.4f} "
                  f"{r['mean_lesion_recall']:<14.4f}{marker}")

    return results, best_t


# =============================================================================
# ANALYSIS 2: Per-size threshold optimization
# =============================================================================

def per_size_threshold_sweep(val_data, granularity=0.02):
    """Find optimal threshold per lesion size bucket."""
    print(f"\n{'='*60}")
    print("  ANALYSIS 2: Per-Size Threshold Optimization")
    print(f"{'='*60}")

    buckets = [
        ('tiny (<100)', 0, 100),
        ('small (100-500)', 100, 500),
        ('medium (500-5k)', 500, 5000),
        ('large (5k-20k)', 5000, 20000),
        ('huge (>20k)', 20000, 1e9),
    ]

    thresholds = np.arange(0.10, 0.70, granularity)
    results = {}

    print(f"\n  {'Size Bucket':<20} {'N':<5} {'Best Thresh':<12} {'Mean Dice':<12} {'Lesion Recall':<14}")
    print(f"  {'-'*65}")

    for label, lo, hi in buckets:
        bucket_cases = [c for c in val_data if lo <= c['lesion_voxels'] < hi]
        if not bucket_cases:
            continue

        best_t, best_dice = 0.4, 0
        for t in thresholds:
            dices = [dice_score((c['prob'] > t).astype(np.float32), c['mask'])
                     for c in bucket_cases]
            mean_d = np.mean(dices)
            if mean_d > best_dice:
                best_dice = mean_d
                best_t = t

        # Get recall at best threshold
        recalls = [lesion_recall((c['prob'] > best_t).astype(np.float32), c['mask'])
                   for c in bucket_cases]
        mean_recall = np.mean(recalls)

        results[label] = {
            'n': len(bucket_cases),
            'best_threshold': float(best_t),
            'mean_dice': float(best_dice),
            'mean_lesion_recall': float(mean_recall),
        }

        print(f"  {label:<20} {len(bucket_cases):<5} {best_t:<12.3f} "
              f"{best_dice:<12.4f} {mean_recall:<14.4f}")

    return results


# =============================================================================
# ANALYSIS 3: Post-processing experiments
# =============================================================================

def postprocessing_sweep(val_data, base_threshold=0.40):
    """Test various post-processing strategies."""
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 3: Post-Processing (base threshold={base_threshold})")
    print(f"{'='*60}")

    struct = generate_binary_structure(3, 1)

    strategies = {
        'none': lambda x: x,
        'cc_5': lambda x: postprocess(x, min_size=5),
        'cc_10': lambda x: postprocess(x, min_size=10),
        'cc_20': lambda x: postprocess(x, min_size=20),
        'cc_50': lambda x: postprocess(x, min_size=50),
        'cc_100': lambda x: postprocess(x, min_size=100),
        'fill_holes': lambda x: binary_fill_holes(x).astype(np.float32),
        'cc10_fill': lambda x: binary_fill_holes(postprocess(x, min_size=10)).astype(np.float32),
        'dilate_1': lambda x: binary_dilation(x.astype(bool), structure=struct, iterations=1).astype(np.float32),
        'cc10_dilate': lambda x: binary_dilation(
            postprocess(x, min_size=10).astype(bool), structure=struct, iterations=1
        ).astype(np.float32),
    }

    print(f"\n  {'Strategy':<20} {'Mean Dice':<12} {'Median Dice':<14} {'Lesion Recall':<14} {'Delta Dice':<12}")
    print(f"  {'-'*72}")

    baseline_dice = None
    results = {}

    for name, pp_fn in strategies.items():
        dices = []
        recalls = []
        for case in val_data:
            pred_bin = (case['prob'] > base_threshold).astype(np.float32)
            processed = pp_fn(pred_bin)
            dices.append(dice_score(processed, case['mask']))
            recalls.append(lesion_recall(processed, case['mask']))

        mean_d = np.mean(dices)
        median_d = np.median(dices)
        mean_r = np.mean(recalls)

        if baseline_dice is None:
            baseline_dice = mean_d
        delta = mean_d - baseline_dice

        results[name] = {
            'mean_dice': float(mean_d),
            'median_dice': float(median_d),
            'mean_lesion_recall': float(mean_r),
            'delta_dice': float(delta),
        }

        print(f"  {name:<20} {mean_d:<12.4f} {median_d:<14.4f} {mean_r:<14.4f} {delta:+.4f}")

    return results


# =============================================================================
# ANALYSIS 4: Model agreement analysis
# =============================================================================

def model_agreement_analysis(val_data):
    """Analyze where models agree/disagree and how that correlates with errors."""
    print(f"\n{'='*60}")
    print("  ANALYSIS 4: Model Agreement vs Accuracy")
    print(f"{'='*60}")

    agreement_bins = {
        'all_4_agree_pos': {'dice': [], 'count': 0},
        '3_of_4_pos': {'dice': [], 'count': 0},
        '2_of_4_pos': {'dice': [], 'count': 0},
        '1_of_4_pos': {'dice': [], 'count': 0},
        'all_4_agree_neg': {'dice': [], 'count': 0},
    }

    for case in val_data:
        # Count how many models predict positive at their optimal thresholds
        model_thresholds = {
            'exp1_8patch': 0.3, 'exp3_12patch_maxfn': 0.25,
            'improved_24patch': 0.5, 'improved_36patch': 0.5,
        }

        # Per-voxel agreement
        agreement_map = np.zeros_like(case['mask'])
        for name, thresh in model_thresholds.items():
            agreement_map += (case['preds'][name] > thresh).astype(np.float32)

        mask = case['mask']
        # Analyze: what's the dice in regions of different agreement levels?
        for n_agree in [4, 3, 2, 1, 0]:
            region = (agreement_map == n_agree)
            if not region.any():
                continue

            # What fraction of this agreement region is actually foreground?
            fg_in_region = (mask[region] > 0).sum()
            total_in_region = region.sum()
            fg_frac = fg_in_region / total_in_region if total_in_region > 0 else 0

            key = f"{'all_4_agree_pos' if n_agree == 4 else f'{n_agree}_of_4_pos' if n_agree > 0 else 'all_4_agree_neg'}"
            if key in agreement_bins:
                agreement_bins[key]['count'] += int(total_in_region)

    # Also compute: what threshold gives best dice for regions of low agreement?
    print(f"\n  When models disagree, lower thresholds might help catch small lesions.")
    print(f"  Analyzing ensemble probability distribution in missed lesion regions...\n")

    missed_probs = []
    detected_probs = []

    for case in val_data:
        pred_bin = (case['prob'] > 0.40).astype(np.float32)
        gt_labeled, n_gt = ndimage_label(case['mask'])

        for i in range(1, n_gt + 1):
            gt_mask = (gt_labeled == i)
            lesion_prob = case['prob'][gt_mask].mean()
            lesion_size = gt_mask.sum()

            if (pred_bin[gt_mask] > 0).any():
                detected_probs.append((lesion_prob, lesion_size))
            else:
                missed_probs.append((lesion_prob, lesion_size))

    if missed_probs:
        missed_p = [p for p, _ in missed_probs]
        missed_s = [s for _, s in missed_probs]
        detected_p = [p for p, _ in detected_probs]

        print(f"  Detected lesions ({len(detected_probs)}):")
        print(f"    Mean ensemble prob: {np.mean(detected_p):.4f}")
        print(f"    Median: {np.median(detected_p):.4f}")

        print(f"\n  Missed lesions ({len(missed_probs)}):")
        print(f"    Mean ensemble prob: {np.mean(missed_p):.4f}")
        print(f"    Median: {np.median(missed_p):.4f}")
        print(f"    Median size: {np.median(missed_s):.0f} voxels")

        # What fraction of missed lesions could be recovered at lower thresholds?
        for recovery_thresh in [0.30, 0.25, 0.20, 0.15, 0.10]:
            recoverable = sum(1 for p in missed_p if p > recovery_thresh)
            print(f"    Recoverable at thresh={recovery_thresh}: "
                  f"{recoverable}/{len(missed_probs)} ({100*recoverable/len(missed_probs):.1f}%)")

    results = {
        'n_missed': len(missed_probs),
        'n_detected': len(detected_probs),
        'missed_mean_prob': float(np.mean([p for p, _ in missed_probs])) if missed_probs else 0,
        'detected_mean_prob': float(np.mean([p for p, _ in detected_probs])) if detected_probs else 0,
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Threshold and post-processing analysis")
    parser.add_argument('--granularity', type=float, default=0.01,
                        help="Threshold sweep step size (default: 0.01)")
    parser.add_argument('--output', type=str, default='model/threshold_analysis.json',
                        help="Output path for results")
    args = parser.parse_args()

    print("=" * 60)
    print("  Threshold & Post-Processing Analysis (CPU only)")
    print("=" * 60)

    t0 = time.time()
    val_data, val_ids = get_val_data()
    print(f"Loaded {len(val_data)} validation cases with foreground")

    if not val_data:
        print("No data to analyze!")
        return

    # Run all analyses
    sweep_results, best_global_t = global_threshold_sweep(val_data, args.granularity)
    size_results = per_size_threshold_sweep(val_data)
    pp_results = postprocessing_sweep(val_data, base_threshold=best_global_t)
    agreement_results = model_agreement_analysis(val_data)

    # Save everything
    save_data = {
        'global_sweep': {
            'best_threshold': best_global_t,
            'best_dice': sweep_results[best_global_t]['mean_dice'],
            'key_thresholds': {
                str(t): sweep_results[t] for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                if t in sweep_results
            },
        },
        'per_size_thresholds': size_results,
        'post_processing': pp_results,
        'model_agreement': agreement_results,
    }

    out_path = ROOT / args.output
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Analysis complete in {elapsed:.1f}s")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
