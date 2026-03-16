"""
TTA (Test-Time Augmentation) evaluation for v2 models.

Runs 8-flip TTA on all 4 v2 models, caches results, then evaluates
ensemble methods (avg, top-3, weighted, etc.) at full volume.

This script uses the GPU — run alongside CPU-only scripts.

Usage:
    python scripts/tta_evaluation.py
    python scripts/tta_evaluation.py --models top3     # Only top-3 models
    python scripts/tta_evaluation.py --force            # Regenerate all predictions
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from segmentation.unet import LightweightUNet3D

MODEL_DIR = ROOT / 'model'
DATA_DIR = ROOT / 'data' / 'preprocessed_256' / 'train'
CACHE_DIR = ROOT / 'model' / 'stacking_cache_v4_tta'

V2_BASE_MODELS = {
    'exp1_8patch': {'patch_size': 32},
    'exp3_12patch_maxfn': {'patch_size': 48},
    'improved_24patch': {'patch_size': 64},
    'improved_36patch': {'patch_size': 96},
}

TOP3_MODELS = ['exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']


# =============================================================================
# INFERENCE
# =============================================================================

def sliding_window_inference(model, volume, patch_size, device, overlap=0.5):
    """Sliding window inference on a single volume."""
    model.eval()
    p = patch_size
    C, H, W, D = volume.shape

    # Pad if needed
    pad_H = max(p - H, 0)
    pad_W = max(p - W, 0)
    pad_D = max(p - D, 0)
    if pad_H > 0 or pad_W > 0 or pad_D > 0:
        volume = np.pad(volume, ((0, 0), (0, pad_H), (0, pad_W), (0, pad_D)), mode='constant')

    orig_H, orig_W, orig_D = H, W, D
    _, H, W, D = volume.shape

    stride = max(1, int(p * (1 - overlap)))
    output = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)

    # Batch size based on patch size
    batch_sizes = {32: 32, 48: 16, 64: 8, 96: 4}
    batch_size = batch_sizes.get(p, 4)

    coords = []
    for h in range(0, H - p + 1, stride):
        for w in range(0, W - p + 1, stride):
            for d in range(0, D - p + 1, stride):
                coords.append((h, w, d))

    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            patches = []
            for h, w, d in batch_coords:
                patches.append(volume[:, h:h+p, w:w+p, d:d+p])

            batch = torch.from_numpy(np.stack(patches)).float().to(device)
            with autocast('cuda'):
                out = model(batch)
                if isinstance(out, tuple):
                    out = out[0]
                preds = torch.sigmoid(out).cpu().numpy()

            for j, (h, w, d) in enumerate(batch_coords):
                output[h:h+p, w:w+p, d:d+p] += preds[j, 0]
                count[h:h+p, w:w+p, d:d+p] += 1

    output = output / np.maximum(count, 1)
    return output[:orig_H, :orig_W, :orig_D]


def tta_sliding_window_inference(model, volume, patch_size, device, overlap=0.5):
    """Run inference with 8 flip TTA combinations."""
    flip_axes_combos = [
        [], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3],
    ]

    all_preds = []
    for flip_axes in flip_axes_combos:
        vol = volume.copy()
        for ax in flip_axes:
            vol = np.flip(vol, axis=ax).copy()

        pred = sliding_window_inference(model, vol, patch_size, device, overlap=overlap)

        for ax in flip_axes:
            pred = np.flip(pred, axis=ax - 1).copy()

        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


# =============================================================================
# VOLUME LOADING
# =============================================================================

def load_volume(case_dir):
    """Load 4-channel volume at native 256³ with z-score normalization."""
    sequences = ['t1_pre', 't1_gd', 'flair', 't2']
    channels = []
    for seq in sequences:
        for ext in ['.npz', '.npy', '.nii.gz']:
            p = case_dir / f'{seq}{ext}'
            if p.exists():
                if ext == '.npz':
                    data = np.load(str(p))['data'].astype(np.float32)
                elif ext == '.npy':
                    data = np.load(str(p)).astype(np.float32)
                else:
                    import nibabel as nib
                    data = nib.load(str(p)).get_fdata().astype(np.float32)
                # Z-score normalize (same as train_stacking.py)
                mean, std = data.mean(), data.std()
                data = (data - mean) / (std + 1e-6)
                channels.append(data)
                break
    if len(channels) != 4:
        return None
    return np.stack(channels, axis=0)


def load_mask(case_dir):
    """Load segmentation mask."""
    for ext in ['.npz', '.npy', '.nii.gz']:
        p = case_dir / f'seg{ext}'
        if p.exists():
            if ext == '.npz':
                return np.load(str(p))['data'].astype(np.float32)
            elif ext == '.npy':
                return np.load(str(p)).astype(np.float32)
            else:
                import nibabel as nib
                return nib.load(str(p)).get_fdata().astype(np.float32)
    return None


# =============================================================================
# METRICS
# =============================================================================

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
    from scipy.ndimage import label as ndimage_label
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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TTA evaluation for v2 models")
    parser.add_argument('--models', type=str, default='all',
                        choices=['all', 'top3'],
                        help='Which models to run TTA on (default: all)')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate all predictions even if cached')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Sliding window overlap (default: 0.5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_names = TOP3_MODELS if args.models == 'top3' else list(V2_BASE_MODELS.keys())
    print(f"Models: {model_names}")

    # Get val cases (same split as training)
    valid_prefixes = ('Mets_', 'UCSF_', 'BraTS_', 'Yale_', 'BMS_')
    all_cases = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith(valid_prefixes)
    ])
    all_case_ids = [d.name for d in all_cases]

    random.seed(42)
    cases_shuffled = all_case_ids.copy()
    random.shuffle(cases_shuffled)
    n_val = int(len(cases_shuffled) * 0.15)
    val_case_ids = cases_shuffled[:n_val]
    print(f"Validation cases: {len(val_case_ids)}")

    # Create cache dir
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    models = {}
    for name in model_names:
        config = V2_BASE_MODELS[name]
        model_path = MODEL_DIR / f'v2_{name}_finetuned.pth'
        if not model_path.exists():
            print(f"WARNING: {model_path} not found, skipping {name}")
            continue

        model = LightweightUNet3D(
            in_channels=4, out_channels=1,
            base_channels=20, use_attention=True, use_residual=True,
            deep_supervision=True
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models[name] = (model, config['patch_size'])
        print(f"  Loaded v2_{name} (Dice={checkpoint.get('dice', 'N/A')})")

    # =========================================================================
    # STEP 1: Generate TTA predictions for val cases
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STEP 1: TTA Inference (8 flips × {len(models)} models × {len(val_case_ids)} cases)")
    print(f"{'='*60}")

    case_dirs = {d.name: d for d in all_cases}

    for i, case_id in enumerate(val_case_ids):
        cache_file = CACHE_DIR / f'{case_id}.npz'
        if cache_file.exists() and not args.force:
            continue

        case_dir = case_dirs.get(case_id)
        if case_dir is None:
            continue

        print(f"\n[{i+1}/{len(val_case_ids)}] {case_id}")
        volume = load_volume(case_dir)
        mask = load_mask(case_dir)
        if volume is None or mask is None:
            continue

        preds = {}
        for name, (model, patch_size) in models.items():
            t0 = time.time()
            prob_map = tta_sliding_window_inference(
                model, volume, patch_size, device, overlap=args.overlap
            )
            elapsed = time.time() - t0
            print(f"  {name} (patch={patch_size}): {elapsed:.1f}s")
            preds[name] = prob_map.astype(np.float16)

        np.savez_compressed(
            cache_file,
            mask=mask.astype(np.uint8),
            **preds
        )

    # =========================================================================
    # STEP 2: Evaluate ensemble methods
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STEP 2: Full-Volume Evaluation with TTA Predictions")
    print(f"{'='*60}")

    thresholds_to_try = np.arange(0.1, 0.91, 0.05)

    # Methods to evaluate
    method_names = ['avg_all', 'top3_avg', 'top3_weighted', 'top2_avg',
                    'max_all', 'top3_max']
    # Add individual models
    method_names += [f'individual_{m}' for m in model_names]

    # method -> threshold -> list of (case_id, dice)
    results = {m: {t: [] for t in thresholds_to_try} for m in method_names}

    for case_id in tqdm(val_case_ids, desc="Evaluating"):
        cache_file = CACHE_DIR / f'{case_id}.npz'
        if not cache_file.exists():
            continue

        data = np.load(cache_file)
        mask = data['mask'].astype(np.float32)
        if mask.sum() == 0:
            continue

        preds = {name: data[name].astype(np.float32) for name in model_names if name in data}
        if len(preds) == 0:
            continue

        # Build probability maps for each method
        prob_maps = {}
        prob_maps['avg_all'] = np.mean([preds[m] for m in model_names if m in preds], axis=0)

        top3_available = [m for m in TOP3_MODELS if m in preds]
        if top3_available:
            prob_maps['top3_avg'] = np.mean([preds[m] for m in top3_available], axis=0)
            prob_maps['top3_max'] = np.max([preds[m] for m in top3_available], axis=0)
            prob_maps['top3_weighted'] = (
                0.6971 * preds.get('exp3_12patch_maxfn', np.zeros_like(mask)) +
                0.7310 * preds.get('improved_24patch', np.zeros_like(mask)) +
                0.7286 * preds.get('improved_36patch', np.zeros_like(mask))
            ) / (0.6971 + 0.7310 + 0.7286)

        top2 = ['improved_24patch', 'improved_36patch']
        top2_available = [m for m in top2 if m in preds]
        if len(top2_available) == 2:
            prob_maps['top2_avg'] = np.mean([preds[m] for m in top2_available], axis=0)

        prob_maps['max_all'] = np.max([preds[m] for m in model_names if m in preds], axis=0)

        for m in model_names:
            if m in preds:
                prob_maps[f'individual_{m}'] = preds[m]

        # Sweep thresholds
        for method_name, prob_map in prob_maps.items():
            if method_name not in results:
                continue
            for t in thresholds_to_try:
                pred_bin = (prob_map > t).astype(np.float32)
                d = dice_score(pred_bin, mask)
                results[method_name][t].append((case_id, d))

    # Find best threshold per method
    print(f"\n{'Method':<30} {'Best Thresh':<12} {'Mean Dice':<10} {'Median Dice':<12}")
    print("-" * 65)

    summary = {}
    for method_name in method_names:
        best_t, best_dice = 0, 0
        for t in thresholds_to_try:
            dices = [d for _, d in results[method_name][t]]
            if not dices:
                continue
            mean_d = np.mean(dices)
            if mean_d > best_dice:
                best_dice = mean_d
                best_t = t

        dices_at_best = [d for _, d in results[method_name][best_t]]
        median_d = float(np.median(dices_at_best)) if dices_at_best else 0

        summary[method_name] = {
            'best_threshold': float(best_t),
            'mean_dice': float(best_dice),
            'median_dice': median_d,
        }

    for method_name, s in sorted(summary.items(), key=lambda x: -x[1]['mean_dice']):
        print(f"{method_name:<30} {s['best_threshold']:<12.2f} {s['mean_dice']:<10.4f} {s['median_dice']:<12.4f}")

    # Also test best method + post-processing
    print(f"\n--- Top 3 methods + cc_20 post-processing ---")
    top3_methods = sorted(summary.keys(), key=lambda m: -summary[m]['mean_dice'])[:3]
    for method_name in top3_methods:
        t = summary[method_name]['best_threshold']
        dices_pp = []
        for case_id in val_case_ids:
            cache_file = CACHE_DIR / f'{case_id}.npz'
            if not cache_file.exists():
                continue
            data = np.load(cache_file)
            mask = data['mask'].astype(np.float32)
            if mask.sum() == 0:
                continue
            preds = {name: data[name].astype(np.float32) for name in model_names if name in data}

            if method_name == 'avg_all':
                prob = np.mean([preds[m] for m in model_names if m in preds], axis=0)
            elif method_name == 'top3_avg':
                prob = np.mean([preds[m] for m in TOP3_MODELS if m in preds], axis=0)
            elif method_name == 'top3_weighted':
                prob = (0.6971*preds.get('exp3_12patch_maxfn', np.zeros_like(mask)) +
                        0.7310*preds.get('improved_24patch', np.zeros_like(mask)) +
                        0.7286*preds.get('improved_36patch', np.zeros_like(mask))) / (0.6971+0.7310+0.7286)
            else:
                continue

            pred_bin = (prob > t).astype(np.float32)
            processed = postprocess(pred_bin, min_size=20)
            d = dice_score(processed, mask)
            dices_pp.append(d)

        if dices_pp:
            mean_pp = np.mean(dices_pp)
            delta = mean_pp - summary[method_name]['mean_dice']
            print(f"  {method_name:<30} Dice={mean_pp:.4f} ({delta:+.4f} vs raw)")

    # Compare TTA vs non-TTA
    print(f"\n--- TTA vs Non-TTA Comparison ---")
    non_tta_cache = ROOT / 'model' / 'stacking_cache_v4'
    if non_tta_cache.exists():
        for method_label, compute_fn_name in [('avg_all', 'avg'), ('top3_avg', 'top3')]:
            tta_dice = summary.get(method_label, {}).get('mean_dice', 0)
            tta_thresh = summary.get(method_label, {}).get('best_threshold', 0.5)

            # Compute non-TTA equivalent
            non_tta_dices = []
            for case_id in val_case_ids:
                ntta_file = non_tta_cache / f'{case_id}.npz'
                if not ntta_file.exists():
                    continue
                data = np.load(ntta_file)
                mask = data['mask'].astype(np.float32)
                if mask.sum() == 0:
                    continue
                preds = {name: data[name].astype(np.float32) for name in model_names if name in data}
                if compute_fn_name == 'avg':
                    prob = np.mean([preds[m] for m in model_names if m in preds], axis=0)
                else:
                    prob = np.mean([preds[m] for m in TOP3_MODELS if m in preds], axis=0)
                pred_bin = (prob > tta_thresh).astype(np.float32)
                d = dice_score(pred_bin, mask)
                non_tta_dices.append(d)

            if non_tta_dices:
                non_tta_mean = np.mean(non_tta_dices)
                delta = tta_dice - non_tta_mean
                print(f"  {method_label}: TTA={tta_dice:.4f} vs NoTTA={non_tta_mean:.4f} (delta={delta:+.4f})")

    # Save results
    out_path = MODEL_DIR / 'tta_evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
