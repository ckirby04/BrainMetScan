"""
Train multiple patch sizes sequentially with improved settings.
Uses augmentation, combined loss, and multiple patches per volume.

Trains 12³, 24³, 36³, 48³, and 64³ patches for 250 epochs each.

Usage:
    python scripts/train_all_patches.py
    python scripts/train_all_patches.py --epochs 150
    python scripts/train_all_patches.py --start-from 36
    python scripts/train_all_patches.py --patches-per-volume 10
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import time
import gc

from segmentation.leaderboard import Leaderboard

# Import the improved training function
from train_improved import train_model


def main():
    parser = argparse.ArgumentParser(description='Train all patch sizes with improved settings')
    parser.add_argument('--epochs', type=int, default=250, help='Epochs per patch size')
    parser.add_argument('--patches-per-volume', type=int, default=5, help='Patches to extract per volume')
    parser.add_argument('--start-from', type=int, default=None, help='Start from this patch size (skip earlier)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    PATCH_SIZES = [12, 24, 36, 48, 64]
    EPOCHS = args.epochs
    PATCHES_PER_VOLUME = args.patches_per_volume

    # Skip patch sizes if --start-from specified
    if args.start_from:
        PATCH_SIZES = [p for p in PATCH_SIZES if p >= args.start_from]

    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data" / "preprocessed_256" / "train"
    model_dir = project_dir / "model"
    model_dir.mkdir(exist_ok=True)

    leaderboard = Leaderboard(str(model_dir / "leaderboard.json"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_mem = 8.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n{'#'*70}")
    print(f"# MULTI-PATCH IMPROVED TRAINING")
    print(f"#")
    print(f"# Patch sizes: {PATCH_SIZES}")
    print(f"# Epochs per size: {EPOCHS}")
    print(f"# Patches per volume: {PATCHES_PER_VOLUME}")
    print(f"# Device: {device}")
    if torch.cuda.is_available():
        print(f"# GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    print(f"#")
    print(f"# Improvements enabled:")
    print(f"#   - Data augmentation (flip, rotate, intensity)")
    print(f"#   - Combined loss (Dice + Focal + Tversky)")
    print(f"#   - Multiple patches per volume ({PATCHES_PER_VOLUME}x data)")
    print(f"#   - Attention gates + residual connections")
    print(f"#   - Gradient clipping")
    print(f"{'#'*70}")

    results = {}
    total_start = time.time()

    for i, patch_size in enumerate(PATCH_SIZES):
        print(f"\n\n{'#'*70}")
        print(f"# TRAINING {i+1}/{len(PATCH_SIZES)}: {patch_size}³ PATCHES")
        print(f"{'#'*70}\n")

        best_dice, best_tiny = train_model(
            patch_size=patch_size,
            epochs=EPOCHS,
            patches_per_volume=PATCHES_PER_VOLUME,
            lr=args.lr,
            data_dir=data_dir,
            model_dir=model_dir,
            leaderboard=leaderboard,
            device=device,
            gpu_mem=gpu_mem
        )

        results[patch_size] = {'dice': best_dice, 'tiny': best_tiny}

        # Force cleanup between models
        gc.collect()
        torch.cuda.empty_cache()

        # Print progress
        elapsed = (time.time() - total_start) / 60
        remaining_patches = len(PATCH_SIZES) - (i + 1)
        est_remaining = elapsed / (i + 1) * remaining_patches if i > 0 else 0

        print(f"\n[Progress] Completed {i+1}/{len(PATCH_SIZES)} patch sizes")
        print(f"[Progress] Elapsed: {elapsed:.1f} min, Est. remaining: {est_remaining:.1f} min")

    total_time = (time.time() - total_start) / 60

    # Final summary
    print(f"\n\n{'#'*70}")
    print(f"# ALL TRAINING COMPLETE")
    print(f"# Total time: {total_time:.1f} min ({total_time/60:.1f} hours)")
    print(f"{'#'*70}\n")

    print("=" * 60)
    print("RESULTS BY PATCH SIZE")
    print("=" * 60)
    print(f"{'Patch':<10} {'Best Dice':>12} {'Best Tiny':>12}")
    print("-" * 60)
    for ps in PATCH_SIZES:
        r = results[ps]
        dice_str = f"{r['dice']*100:.1f}%" if r['dice'] else "N/A"
        tiny_str = f"{r['tiny']*100:.1f}%" if r['tiny'] else "N/A"
        print(f"{ps}³".ljust(10) + dice_str.rjust(12) + tiny_str.rjust(12))
    print("=" * 60)

    # Find best models
    best_overall = max(results.items(), key=lambda x: x[1]['dice'] or 0)
    best_tiny = max(results.items(), key=lambda x: x[1]['tiny'] or 0)

    print(f"\nBest overall dice: {best_overall[0]}³ ({best_overall[1]['dice']*100:.1f}%)")
    print(f"Best tiny dice: {best_tiny[0]}³ ({best_tiny[1]['tiny']*100:.1f}%)")

    print("\n")
    leaderboard.print_summary()

    # Save final results
    import json
    results_path = model_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'patch_sizes': PATCH_SIZES,
            'epochs': EPOCHS,
            'patches_per_volume': PATCHES_PER_VOLUME,
            'total_time_min': total_time,
            'results': {str(k): v for k, v in results.items()},
            'best_overall': {'patch': best_overall[0], 'dice': best_overall[1]['dice']},
            'best_tiny': {'patch': best_tiny[0], 'dice': best_tiny[1]['tiny']}
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
