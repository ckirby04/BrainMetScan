"""
Run improvements #1-3: threshold sweep, postprocessing tuning, and hard example analysis.
Generates stacking probability maps, sweeps thresholds + min_component sizes,
and compares against current results.
"""

import json
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.ndimage import label as ndimage_label
from torch.amp import autocast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "model" / "stacking_cache_v4"

# ── Stacking model ──────────────────────────────────────────────────
class StackingClassifier(nn.Module):
    def __init__(self, in_channels=8, mid_channels=32):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm3d(mid_channels), nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm3d(mid_channels), nn.ReLU(inplace=True), nn.Dropout3d(0.1),
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1), nn.BatchNorm3d(mid_channels))
        self.block2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm3d(mid_channels), nn.ReLU(inplace=True), nn.Dropout3d(0.1),
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1), nn.BatchNorm3d(mid_channels))
        self.head = nn.Conv3d(mid_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.entry(x)
        x = self.relu(x + self.block1(x))
        x = self.relu(x + self.block2(x))
        return self.head(x)


MODEL_NAMES = [
    "exp1_8patch", "exp3_12patch_maxfn", "improved_24patch",
    "improved_36patch", "nnunet", "nnunet_2d",
]


def sliding_window_inference(mdl, volume, patch_size, dev, overlap=0.5):
    mdl.eval()
    C, H, W, D = volume.shape
    p = patch_size
    stride = max(int(p * (1 - overlap)), 1)
    pad_h = (p - H % p) % p if H % stride != 0 else 0
    pad_w = (p - W % p) % p if W % stride != 0 else 0
    pad_d = (p - D % p) % p if D % stride != 0 else 0
    orig_H, orig_W, orig_D = H, W, D
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode="constant")
        C, H, W, D = volume.shape
    output = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)
    coords = [(h, w, d)
              for h in range(0, H - p + 1, stride)
              for w in range(0, W - p + 1, stride)
              for d in range(0, D - p + 1, stride)]
    with torch.no_grad():
        for i in range(0, len(coords), 8):
            batch_coords = coords[i:i + 8]
            patches = [volume[:, h:h+p, w:w+p, d:d+p] for h, w, d in batch_coords]
            batch = torch.from_numpy(np.stack(patches)).float().to(dev)
            with autocast("cuda"):
                out = mdl(batch)
                preds = torch.sigmoid(out).cpu().numpy()
            for j, (h, w, d) in enumerate(batch_coords):
                output[h:h+p, w:w+p, d:d+p] += preds[j, 0]
                count[h:h+p, w:w+p, d:d+p] += 1
    output = output / np.maximum(count, 1)
    return output[:orig_H, :orig_W, :orig_D]


def build_features(cache_file):
    data = np.load(cache_file)
    mask = data["mask"]
    preds = np.stack([data[n].astype(np.float32) for n in MODEL_NAMES], axis=0)
    var = preds.var(axis=0, keepdims=True)
    rng = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)
    features = np.concatenate([preds, var, rng], axis=0)
    return features, mask


def dice(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin > 0)).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-8)


def apply_postprocess(prob_map, threshold, min_component):
    pred_bin = (prob_map > threshold).astype(np.uint8)
    if min_component > 0:
        labeled, n = ndimage_label(pred_bin)
        cleaned = np.zeros_like(pred_bin)
        for j in range(1, n + 1):
            comp = (labeled == j)
            if comp.sum() >= min_component:
                cleaned[comp] = 1
        pred_bin = cleaned
    return pred_bin


def main():
    # Load current results
    with open(ROOT / "model" / "stacking_v4_results_per_case.json") as f:
        per_case = json.load(f)
    val_cases = list(per_case.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load stacking model
    ckpt = torch.load(ROOT / "model" / "stacking_v4_classifier.pth",
                      map_location=device, weights_only=False)
    model = StackingClassifier(in_channels=8).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Generate stacking probability maps (save to disk) ──────────
    prob_dir = ROOT / "model" / "stacking_probmaps_tmp"
    prob_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Generating stacking probability maps for all val cases...")
    print("=" * 70)

    valid_cases = []
    for i, case_id in enumerate(val_cases):
        cache_file = CACHE / f"{case_id}.npz"
        if not cache_file.exists():
            continue
        prob_file = prob_dir / f"{case_id}_prob.npy"
        if not prob_file.exists():
            features, mask = build_features(cache_file)
            prob = sliding_window_inference(model, features, 32, device, overlap=0.5)
            np.save(prob_file, prob.astype(np.float16))  # float16 to save space
            del features, prob
        valid_cases.append(case_id)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(val_cases)}]")
    print(f"  Done: {len(valid_cases)} cases\n")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # ── #1: Threshold + min_component sweep (one case at a time) ──
    print("=" * 70)
    print("  #1: THRESHOLD + POST-PROCESSING SWEEP")
    print("=" * 70)

    thresholds = np.arange(0.05, 0.96, 0.05)
    min_components = [0, 3, 5, 10, 15, 20, 30, 50]
    n_configs = len(thresholds) * len(min_components)

    # Accumulate per-config dice scores: config_idx -> list of dices
    config_dices = {i: [] for i in range(n_configs)}

    for ci, case_id in enumerate(valid_cases):
        prob = np.load(prob_dir / f"{case_id}_prob.npy").astype(np.float32)
        data = np.load(CACHE / f"{case_id}.npz")
        mask = data["mask"]

        idx = 0
        for mc in min_components:
            for t in thresholds:
                pred_bin = apply_postprocess(prob, t, mc)
                config_dices[idx].append(dice(pred_bin, mask))
                idx += 1

        del prob, mask, data
        if (ci + 1) % 20 == 0:
            print(f"  Sweep progress: [{ci+1}/{len(valid_cases)}]")

    # Find best config
    best_mean = 0
    best_median = 0
    best_config = None
    results_grid = []

    idx = 0
    for mc in min_components:
        for t in thresholds:
            dices = config_dices[idx]
            mean_d = np.mean(dices)
            median_d = np.median(dices)
            results_grid.append((t, mc, mean_d, median_d))

            if median_d > best_median or (median_d == best_median and mean_d > best_mean):
                best_median = median_d
                best_mean = mean_d
                best_config = (t, mc, mean_d, median_d)
            idx += 1

    print(f"\nBest config: threshold={best_config[0]:.2f}, min_component={best_config[1]}")
    print(f"  Mean Dice:   {best_config[2]:.4f}")
    print(f"  Median Dice: {best_config[3]:.4f}")

    # Top 10
    results_grid.sort(key=lambda x: (-x[3], -x[2]))
    print(f"\nTop 10 configs by median Dice:")
    print(f"  {'Thresh':<8} {'MinComp':<10} {'Mean':<10} {'Median':<10}")
    print(f"  " + "-" * 38)
    for t, mc, mean_d, med_d in results_grid[:10]:
        marker = " <-- current" if abs(t - 0.90) < 0.01 and mc == 20 else ""
        print(f"  {t:<8.2f} {mc:<10} {mean_d:<10.4f} {med_d:<10.4f}{marker}")

    # Where does current config rank?
    current = [r for r in results_grid if abs(r[0] - 0.90) < 0.01 and r[1] == 20]
    if current:
        rank = results_grid.index(current[0]) + 1
        print(f"\nCurrent config (t=0.90, mc=20) ranks #{rank}/{len(results_grid)}")
        print(f"  Mean: {current[0][2]:.4f}, Median: {current[0][3]:.4f}")

    # ── Per-case improvement with best config ───────────────────────
    print("\n" + "=" * 70)
    print(f"  PER-CASE: best config (t={best_config[0]:.2f}, mc={best_config[1]}) vs current")
    print("=" * 70)

    best_t, best_mc = best_config[0], best_config[1]
    improved = []
    degraded = []
    new_per_case = {}

    for case_id in valid_cases:
        prob = np.load(prob_dir / f"{case_id}_prob.npy").astype(np.float32)
        data = np.load(CACHE / f"{case_id}.npz")
        mask = data["mask"]
        old_dice_val = per_case[case_id]["dice"]
        pred_bin = apply_postprocess(prob, best_t, best_mc)
        new_d = dice(pred_bin, mask)
        new_per_case[case_id] = new_d
        delta = new_d - old_dice_val
        if delta > 0.02:
            improved.append((case_id, old_dice_val, new_d, delta))
        elif delta < -0.02:
            degraded.append((case_id, old_dice_val, new_d, delta))
        del prob, mask, data

    improved.sort(key=lambda x: -x[3])
    degraded.sort(key=lambda x: x[3])

    print(f"\nImproved (>2% gain): {len(improved)} cases")
    for case_id, old, new, delta in improved[:15]:
        print(f"  {case_id:<25} {old:.4f} -> {new:.4f}  (+{delta:.4f})")

    print(f"\nDegraded (>2% loss): {len(degraded)} cases")
    for case_id, old, new, delta in degraded[:15]:
        print(f"  {case_id:<25} {old:.4f} -> {new:.4f}  ({delta:.4f})")

    # Distribution comparison
    old_dices = np.array([per_case[c]["dice"] for c in valid_cases])
    new_dices = np.array([new_per_case[c] for c in valid_cases])

    print(f"\n{'Metric':<20} {'Current (t=0.9,mc=20)':<25} {'Best config':<25}")
    print("-" * 70)
    print(f"{'Mean':<20} {np.mean(old_dices):<25.4f} {np.mean(new_dices):<25.4f}")
    print(f"{'Median':<20} {np.median(old_dices):<25.4f} {np.median(new_dices):<25.4f}")
    print(f"{'25th pct':<20} {np.percentile(old_dices, 25):<25.4f} {np.percentile(new_dices, 25):<25.4f}")
    print(f"{'75th pct':<20} {np.percentile(old_dices, 75):<25.4f} {np.percentile(new_dices, 75):<25.4f}")
    print(f"{'Cases < 0.5':<20} {(old_dices < 0.5).sum():<25} {(new_dices < 0.5).sum():<25}")
    print(f"{'Cases > 0.8':<20} {(old_dices > 0.8).sum():<25} {(new_dices > 0.8).sum():<25}")
    print(f"{'Cases > 0.85':<20} {(old_dices > 0.85).sum():<25} {(new_dices > 0.85).sum():<25}")

    # ── #2: Hard example analysis ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  #2: HARD EXAMPLE ANALYSIS — cases that still fail with best config")
    print("=" * 70)

    still_bad = [(c, new_per_case[c]) for c in valid_cases if new_per_case[c] < 0.6]
    still_bad.sort(key=lambda x: x[1])

    print(f"\nCases still below 0.6 Dice: {len(still_bad)}")
    for case_id, d in still_bad:
        # Check if any individual model does better
        data = np.load(CACHE / f"{case_id}.npz")
        mask = data["mask"]
        best_model = ""
        best_model_dice = 0
        for mn in MODEL_NAMES:
            pred = data[mn].astype(np.float32)
            for t_try in [0.1, 0.2, 0.3, 0.5]:
                md = dice((pred > t_try).astype(np.uint8), mask)
                if md > best_model_dice:
                    best_model_dice = md
                    best_model = f"{mn}@{t_try}"
        print(f"  {case_id:<25} stacking={d:.4f}  best_individual={best_model_dice:.4f} ({best_model})")

    # ── #3: Simple ensemble of raw nnU-Net 3D probs (all 5 folds) ──
    print("\n" + "=" * 70)
    print("  #3: RAW nnU-NET 3D 5-FOLD ENSEMBLE (bypass stacking)")
    print("=" * 70)

    # The nnunet probs in cache are already from proper CV folds.
    # Let's also check: what if we just average nnunet_3d + nnunet_2d?
    nn3d_dices = []
    nn2d_dices = []
    nn_avg_dices = []

    for case_id in valid_cases:
        data = np.load(CACHE / f"{case_id}.npz")
        mask = data["mask"]

        nn3d = data["nnunet"].astype(np.float32)
        nn2d = data["nnunet_2d"].astype(np.float32)
        nn_avg = (nn3d + nn2d) / 2.0

        # Sweep a few thresholds for each
        best_3d = max(dice((nn3d > t).astype(np.uint8), mask) for t in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
        best_2d = max(dice((nn2d > t).astype(np.uint8), mask) for t in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
        best_avg = max(dice((nn_avg > t).astype(np.uint8), mask) for t in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5])

        nn3d_dices.append(best_3d)
        nn2d_dices.append(best_2d)
        nn_avg_dices.append(best_avg)

    print(f"\n  {'Method':<30} {'Mean':<10} {'Median':<10}")
    print(f"  " + "-" * 50)
    print(f"  {'nnU-Net 3D (best thresh)':<30} {np.mean(nn3d_dices):<10.4f} {np.median(nn3d_dices):<10.4f}")
    print(f"  {'nnU-Net 2D (best thresh)':<30} {np.mean(nn2d_dices):<10.4f} {np.median(nn2d_dices):<10.4f}")
    print(f"  {'nnU-Net 3D+2D avg':<30} {np.mean(nn_avg_dices):<10.4f} {np.median(nn_avg_dices):<10.4f}")
    print(f"  {'Stacking (current t=0.9)':<30} {np.mean(old_dices):<10.4f} {np.median(old_dices):<10.4f}")
    print(f"  {'Stacking (best config)':<30} {np.mean(new_dices):<10.4f} {np.median(new_dices):<10.4f}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Best threshold: {best_t:.2f} (was 0.90)")
    print(f"  Best min_component: {best_mc} (was 20)")
    print(f"  Median Dice: {np.median(old_dices):.4f} -> {np.median(new_dices):.4f} "
          f"(+{np.median(new_dices) - np.median(old_dices):.4f})")
    print(f"  Mean Dice:   {np.mean(old_dices):.4f} -> {np.mean(new_dices):.4f} "
          f"(+{np.mean(new_dices) - np.mean(old_dices):.4f})")
    print(f"  Target: 0.85 median (human inter-rater)")
    print()


if __name__ == "__main__":
    main()
