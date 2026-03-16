"""
Dedicated top-3 ensemble analysis (CPU only).

Deep analysis of top-3 model combinations (excluding exp1_8patch which scores 0.585).
Tests fine-grained threshold sweeps, weight optimization, post-processing, and
per-case comparison vs 4-model average.

Run alongside TTA (GPU) and analysis script (CPU).

Usage:
    python scripts/top3_ensemble.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import binary_fill_holes
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'model' / 'stacking_cache_v4'
OUTPUT_DIR = ROOT / 'model'
ALL_MODELS = ['exp1_8patch', 'exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']
TOP3 = ['exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']


def get_val_cases():
    cached_cases = sorted([f.stem for f in CACHE_DIR.glob('*.npz')])
    random.seed(42)
    cases_shuffled = cached_cases.copy()
    random.shuffle(cases_shuffled)
    n_val = int(len(cases_shuffled) * 0.15)
    return cases_shuffled[:n_val]


def load_case(case_id):
    data = np.load(CACHE_DIR / f'{case_id}.npz')
    mask = data['mask'].astype(np.float32)
    preds = {name: data[name].astype(np.float32) for name in ALL_MODELS}
    return preds, mask


def dice_score(pred_binary, target):
    pred = pred_binary.flatten()
    tgt = target.flatten()
    tp = ((pred == 1) & (tgt == 1)).sum()
    fp = ((pred == 1) & (tgt == 0)).sum()
    fn = ((pred == 0) & (tgt == 1)).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-8)


def postprocess(binary_mask, min_size=20):
    labeled, n = ndimage_label(binary_mask)
    if n == 0:
        return binary_mask
    result = np.zeros_like(binary_mask)
    for i in range(1, n + 1):
        comp = (labeled == i)
        if comp.sum() >= min_size:
            result[comp] = 1
    return result


def main():
    print("=" * 70)
    print("  Top-3 Ensemble Deep Analysis (CPU only)")
    print("=" * 70)

    val_cases = get_val_cases()
    print(f"Validation cases: {len(val_cases)}")
    t0 = time.time()

    # =========================================================================
    # STEP 1: Fine-grained threshold sweep (0.01 resolution)
    # =========================================================================
    print(f"\n{'='*60}")
    print("  STEP 1: Fine-Grained Threshold Sweep")
    print(f"{'='*60}")

    thresholds = np.arange(0.10, 0.70, 0.01)
    methods = {
        'avg_4': lambda p: np.mean([p[m] for m in ALL_MODELS], axis=0),
        'top3_avg': lambda p: np.mean([p[m] for m in TOP3], axis=0),
        'top3_max': lambda p: np.max([p[m] for m in TOP3], axis=0),
        'top2_24_36': lambda p: np.mean([p['improved_24patch'], p['improved_36patch']], axis=0),
        'top2_24_12': lambda p: np.mean([p['improved_24patch'], p['exp3_12patch_maxfn']], axis=0),
        'top2_36_12': lambda p: np.mean([p['improved_36patch'], p['exp3_12patch_maxfn']], axis=0),
    }

    # method -> threshold -> list of dice
    sweep_results = {m: {t: [] for t in thresholds} for m in methods}

    for i, case_id in enumerate(val_cases):
        preds, mask = load_case(case_id)
        if mask.sum() == 0:
            continue

        for method_name, method_fn in methods.items():
            prob = method_fn(preds)
            for t in thresholds:
                pred_bin = (prob > t).astype(np.float32)
                d = dice_score(pred_bin, mask)
                sweep_results[method_name][t].append(d)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(val_cases)}] cases...")

    print(f"\n{'Method':<20} {'Best Thresh':<12} {'Mean Dice':<10} {'Median':<10}")
    print("-" * 55)

    best_configs = {}
    for method_name in methods:
        best_t, best_dice = 0, 0
        for t in thresholds:
            dices = sweep_results[method_name][t]
            if not dices:
                continue
            mean_d = np.mean(dices)
            if mean_d > best_dice:
                best_dice = mean_d
                best_t = t
        median_d = float(np.median(sweep_results[method_name][best_t]))
        best_configs[method_name] = {'threshold': float(best_t), 'dice': float(best_dice), 'median': median_d}
        print(f"{method_name:<20} {best_t:<12.2f} {best_dice:<10.4f} {median_d:<10.4f}")

    # =========================================================================
    # STEP 2: Weight optimization for top-3
    # =========================================================================
    print(f"\n{'='*60}")
    print("  STEP 2: Weight Optimization (Scipy)")
    print(f"{'='*60}")

    # Preload all val case predictions for fast optimization
    print("  Loading all val cases into memory...")
    val_data = []
    for case_id in val_cases:
        preds, mask = load_case(case_id)
        if mask.sum() == 0:
            continue
        val_data.append((case_id, preds, mask))
    print(f"  Loaded {len(val_data)} cases with foreground")

    def neg_mean_dice(params):
        """Negative mean dice for scipy minimization. params = [w1, w2, w3, threshold]."""
        w = np.array(params[:3])
        w = np.abs(w) / np.abs(w).sum()  # normalize to sum to 1
        threshold = params[3]

        dices = []
        for case_id, preds, mask in val_data:
            prob = (w[0] * preds[TOP3[0]] + w[1] * preds[TOP3[1]] + w[2] * preds[TOP3[2]])
            pred_bin = (prob > threshold).astype(np.float32)
            d = dice_score(pred_bin, mask)
            dices.append(d)
        return -np.mean(dices)

    # Try multiple starting points
    best_result = None
    starting_points = [
        [1/3, 1/3, 1/3, 0.35],
        [0.25, 0.40, 0.35, 0.35],
        [0.20, 0.45, 0.35, 0.40],
        [0.30, 0.35, 0.35, 0.30],
        [0.15, 0.50, 0.35, 0.35],
        [0.33, 0.33, 0.33, 0.40],
    ]

    for x0 in starting_points:
        result = minimize(
            neg_mean_dice, x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 0.001, 'fatol': 0.0001}
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    opt_w = np.abs(best_result.x[:3])
    opt_w = opt_w / opt_w.sum()
    opt_t = best_result.x[3]
    opt_dice = -best_result.fun

    print(f"\n  Optimal weights:")
    for i, m in enumerate(TOP3):
        print(f"    {m}: {opt_w[i]:.4f}")
    print(f"  Optimal threshold: {opt_t:.4f}")
    print(f"  Optimized Dice: {opt_dice:.4f}")

    # Compare to equal weights
    equal_dice = -neg_mean_dice([1/3, 1/3, 1/3, opt_t])
    print(f"  Equal-weight Dice (same thresh): {equal_dice:.4f}")
    print(f"  Improvement from weighting: {opt_dice - equal_dice:+.4f}")

    # =========================================================================
    # STEP 3: Post-processing sweep on best methods
    # =========================================================================
    print(f"\n{'='*60}")
    print("  STEP 3: Post-Processing on Best Methods")
    print(f"{'='*60}")

    pp_configs = [
        ('none', lambda x: x),
        ('cc_10', lambda x: postprocess(x, min_size=10)),
        ('cc_20', lambda x: postprocess(x, min_size=20)),
        ('cc_50', lambda x: postprocess(x, min_size=50)),
        ('cc_100', lambda x: postprocess(x, min_size=100)),
        ('cc_200', lambda x: postprocess(x, min_size=200)),
        ('cc20_fill', lambda x: binary_fill_holes(postprocess(x, min_size=20)).astype(np.float32)),
        ('cc50_fill', lambda x: binary_fill_holes(postprocess(x, min_size=50)).astype(np.float32)),
    ]

    # Test post-processing on top3_avg, optimized weighted, and avg_4
    test_methods = {
        'top3_avg': {
            'fn': lambda p: np.mean([p[m] for m in TOP3], axis=0),
            'threshold': best_configs['top3_avg']['threshold'],
        },
        'top3_optimized': {
            'fn': lambda p: opt_w[0]*p[TOP3[0]] + opt_w[1]*p[TOP3[1]] + opt_w[2]*p[TOP3[2]],
            'threshold': opt_t,
        },
        'avg_4': {
            'fn': lambda p: np.mean([p[m] for m in ALL_MODELS], axis=0),
            'threshold': best_configs['avg_4']['threshold'],
        },
    }

    print(f"\n{'Method':<22} {'Post-Proc':<12} {'Mean Dice':<10} {'Median':<10} {'Delta':<8}")
    print("-" * 65)

    pp_results = {}
    for method_name, cfg in test_methods.items():
        pp_results[method_name] = {}
        baseline_dice = None

        for pp_name, pp_fn in pp_configs:
            dices = []
            for case_id, preds, mask in val_data:
                prob = cfg['fn'](preds)
                pred_bin = (prob > cfg['threshold']).astype(np.float32)
                processed = pp_fn(pred_bin)
                d = dice_score(processed, mask)
                dices.append(d)

            mean_d = np.mean(dices)
            median_d = np.median(dices)
            if baseline_dice is None:
                baseline_dice = mean_d
            delta = mean_d - baseline_dice

            pp_results[method_name][pp_name] = float(mean_d)
            print(f"{method_name:<22} {pp_name:<12} {mean_d:<10.4f} {median_d:<10.4f} {delta:+.4f}")
        print()

    # =========================================================================
    # STEP 4: Per-case comparison (top3 vs avg_4 vs best individual)
    # =========================================================================
    print(f"{'='*60}")
    print("  STEP 4: Per-Case Comparison")
    print(f"{'='*60}")

    t3_thresh = best_configs['top3_avg']['threshold']
    a4_thresh = best_configs['avg_4']['threshold']

    per_case = []
    for case_id, preds, mask in val_data:
        top3_prob = np.mean([preds[m] for m in TOP3], axis=0)
        avg4_prob = np.mean([preds[m] for m in ALL_MODELS], axis=0)
        opt_prob = opt_w[0]*preds[TOP3[0]] + opt_w[1]*preds[TOP3[1]] + opt_w[2]*preds[TOP3[2]]

        d_top3 = dice_score((top3_prob > t3_thresh).astype(np.float32), mask)
        d_avg4 = dice_score((avg4_prob > a4_thresh).astype(np.float32), mask)
        d_opt = dice_score((opt_prob > opt_t).astype(np.float32), mask)

        # Best individual model
        best_indiv_d = 0
        best_indiv_m = ''
        for m in ALL_MODELS:
            for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
                d = dice_score((preds[m] > t).astype(np.float32), mask)
                if d > best_indiv_d:
                    best_indiv_d = d
                    best_indiv_m = m

        per_case.append({
            'case_id': case_id,
            'lesion_voxels': int(mask.sum()),
            'top3_avg': d_top3,
            'avg_4': d_avg4,
            'optimized': d_opt,
            'best_individual': best_indiv_d,
            'best_individual_model': best_indiv_m,
            'top3_minus_avg4': d_top3 - d_avg4,
        })

    # Summary stats
    top3_wins = sum(1 for c in per_case if c['top3_minus_avg4'] > 0.01)
    avg4_wins = sum(1 for c in per_case if c['top3_minus_avg4'] < -0.01)
    ties = len(per_case) - top3_wins - avg4_wins

    print(f"\n  Top-3 wins: {top3_wins} cases")
    print(f"  Avg-4 wins: {avg4_wins} cases")
    print(f"  Ties (±0.01): {ties} cases")
    print(f"\n  Mean delta (top3 - avg4): {np.mean([c['top3_minus_avg4'] for c in per_case]):+.4f}")

    # Cases where top-3 helps most
    print(f"\n  Top 10 cases where TOP-3 beats AVG-4:")
    for c in sorted(per_case, key=lambda x: -x['top3_minus_avg4'])[:10]:
        print(f"    {c['case_id']}: top3={c['top3_avg']:.4f} avg4={c['avg_4']:.4f} "
              f"delta={c['top3_minus_avg4']:+.4f} vox={c['lesion_voxels']}")

    # Cases where avg-4 is better
    print(f"\n  Top 10 cases where AVG-4 beats TOP-3:")
    for c in sorted(per_case, key=lambda x: x['top3_minus_avg4'])[:10]:
        print(f"    {c['case_id']}: top3={c['top3_avg']:.4f} avg4={c['avg_4']:.4f} "
              f"delta={c['top3_minus_avg4']:+.4f} vox={c['lesion_voxels']}")

    # =========================================================================
    # STEP 5: Final recommendation
    # =========================================================================
    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")

    # Collect all results
    all_results = {
        'avg_4 (baseline)': best_configs['avg_4']['dice'],
        'top3_avg': best_configs['top3_avg']['dice'],
        'top3_optimized_weighted': opt_dice,
    }

    # Add best post-processed variants
    for method_name, pp_dict in pp_results.items():
        best_pp = max(pp_dict.items(), key=lambda x: x[1])
        if best_pp[0] != 'none':
            all_results[f'{method_name}+{best_pp[0]}'] = best_pp[1]

    print(f"\n  {'Method':<35} {'Mean Dice':<10}")
    print("  " + "-" * 45)
    for method, dice in sorted(all_results.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if dice == max(all_results.values()) else ""
        print(f"  {method:<35} {dice:<10.4f}{marker}")

    # =========================================================================
    # Save everything
    # =========================================================================
    save_data = {
        'threshold_sweep': best_configs,
        'optimized_weights': {
            'weights': {TOP3[i]: float(opt_w[i]) for i in range(3)},
            'threshold': float(opt_t),
            'dice': float(opt_dice),
        },
        'post_processing': pp_results,
        'per_case_comparison': {
            'top3_wins': top3_wins,
            'avg4_wins': avg4_wins,
            'ties': ties,
            'mean_delta': float(np.mean([c['top3_minus_avg4'] for c in per_case])),
        },
        'per_case': per_case,
        'final_ranking': {k: float(v) for k, v in sorted(all_results.items(), key=lambda x: -x[1])},
    }

    out_path = OUTPUT_DIR / 'top3_ensemble_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} minutes")
    print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()
