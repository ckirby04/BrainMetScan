#!/usr/bin/env python
"""
Precompute stacking classifier inference for all demo cases.

Loads base model predictions from stacking_cache_v4, runs the stacking
classifier, and saves results to outputs/demo_cache as compressed .npz files.
The demo loads these on startup so analysis is instant.

Also computes Dice scores and saves a manifest (manifest.json) so
the demo only shows the top-performing cases.

Usage:
    python scripts/precompute_demo.py
    python scripts/precompute_demo.py --force   # re-run even if cached
    python scripts/precompute_demo.py --top 25  # keep top 25 instead of 10
"""

import sys
import time
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.stacking import (
    StackingClassifier, load_stacking_model, run_stacking_inference,
    build_stacking_features, STACKING_MODEL_NAMES, STACKING_THRESHOLD,
    DISPLAY_NAMES,
)
from demo.app import (
    get_available_cases, load_case_data,
    DATA_DIR, MODEL_DIR, DEVICE,
)

CACHE_DIR = PROJECT_ROOT / "outputs" / "demo_cache"
STACKING_CACHE_DIR = PROJECT_ROOT / "model" / "stacking_cache_v4"


def compute_dice(prediction, ground_truth, threshold=STACKING_THRESHOLD):
    """Compute Dice score between prediction and ground truth."""
    pred_binary = (prediction > threshold).astype(np.float32)
    gt_binary = (ground_truth > 0.5).astype(np.float32)
    intersection = np.sum(pred_binary * gt_binary)
    return float((2 * intersection) / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8))


def precompute_single(stacking_model, case_name, force=False):
    """Run stacking inference for one case and save to disk. Returns Dice score."""
    cache_path = CACHE_DIR / f"{case_name}_ensemble.npz"
    stacking_cache_path = STACKING_CACHE_DIR / f"{case_name}.npz"

    if not stacking_cache_path.exists():
        print(f"  [skip] {case_name} (no stacking cache)")
        return None

    try:
        sequences, ground_truth = load_case_data(case_name)
    except Exception as e:
        print(f"  [error] {case_name}: could not load data - {e}")
        return None

    # Check if already cached
    if cache_path.exists() and not force:
        data = np.load(str(cache_path))
        fused = data['fused'].astype(np.float32)
        dice = compute_dice(fused, ground_truth)
        print(f"  [skip] {case_name} (cached, Dice={dice:.4f})")
        return dice

    target_size = ground_truth.shape  # e.g. (256, 256, 256)

    t0 = time.time()
    result = run_stacking_inference(
        stacking_cache_path, stacking_model, DEVICE,
        target_size=target_size,
    )
    elapsed = time.time() - t0

    # Compute Dice
    dice = compute_dice(result['fused'], ground_truth)

    # Save: fused + individual predictions + agreement map
    save_dict = {
        'fused': result['fused'].astype(np.float16),
        'agreement': result['agreement'].astype(np.float16),
    }
    for model_name, pred in result['individual'].items():
        save_dict[f'individual_{model_name}'] = pred.astype(np.float16)

    np.savez_compressed(str(cache_path), **save_dict)

    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"  [done] {case_name}  ({elapsed:.1f}s, {size_mb:.1f} MB, Dice={dice:.4f})")
    return dice


def main():
    parser = argparse.ArgumentParser(description="Precompute demo inferences (stacking)")
    parser.add_argument('--force', action='store_true', help='Re-run even if cached')
    parser.add_argument('--top', type=int, default=25, help='Number of top cases to keep (default: 25)')
    args = parser.parse_args()

    print("=" * 60)
    print("Precomputing stacking classifier inference for demo cases")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_DIR}")
    print(f"Cache:  {CACHE_DIR}")
    print(f"Stacking cache: {STACKING_CACHE_DIR}")
    print()

    # Load stacking classifier
    print("Loading stacking classifier...")
    stacking_model = load_stacking_model(MODEL_DIR, DEVICE)
    if stacking_model is None:
        print("ERROR: stacking_v4_classifier.pth not found!")
        sys.exit(1)
    print(f"  Models: {STACKING_MODEL_NAMES}")
    print(f"  Threshold: {STACKING_THRESHOLD}")
    print()

    # Get cases that have stacking cache AND exist in data dir
    stacking_case_names = set()
    if STACKING_CACHE_DIR.exists():
        for npz_path in STACKING_CACHE_DIR.glob("*.npz"):
            stacking_case_names.add(npz_path.stem)

    # Get all data cases
    all_data_cases = []
    if DATA_DIR.exists():
        for case_dir in sorted(DATA_DIR.iterdir()):
            if case_dir.is_dir() and case_dir.name in stacking_case_names:
                seg_file = case_dir / 'seg.nii.gz'
                if seg_file.exists():
                    all_data_cases.append(case_dir.name)

    print(f"Found {len(all_data_cases)} cases with stacking cache + data\n")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Run inference and collect Dice scores
    scores = {}
    t_total = time.time()

    for i, case_name in enumerate(all_data_cases, 1):
        print(f"[{i}/{len(all_data_cases)}] {case_name}")
        dice = precompute_single(stacking_model, case_name, force=args.force)
        if dice is not None:
            scores[case_name] = dice

    elapsed_total = time.time() - t_total

    # Rank by Dice and pick top N
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cases = ranked[:args.top]

    manifest = {
        'version': 3,
        'model_type': 'stacking',
        'top_cases': [{'case_name': name, 'dice_score': score} for name, score in top_cases],
        'all_scores': {name: score for name, score in ranked},
        'num_total': len(all_data_cases),
        'num_selected': len(top_cases),
        'ensemble_models': list(STACKING_MODEL_NAMES),
        'threshold': STACKING_THRESHOLD,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    manifest_path = CACHE_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 60)
    print(f"Done in {elapsed_total:.0f}s")
    print(f"  Total cases:    {len(all_data_cases)}")
    print(f"  Successful:     {len(scores)}")
    print(f"  Top {args.top} by Dice:")
    for name, score in top_cases:
        print(f"    {name}: {score:.4f}")
    print(f"  Manifest:       {manifest_path}")
    print(f"  Cache:          {CACHE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
