"""
Generate test predictions using the stacking ensemble.

Steps:
  1. Generate base model predictions for test cases (cached)
  2. Add nnU-Net test predictions (multi-fold ensemble)
  3. Run stacking inference and save NIfTI outputs

Usage:
    python scripts/predict_test.py
    python scripts/predict_test.py --method stacking --threshold 0.9
"""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Import from train_stacking
sys.path.insert(0, str(ROOT / "scripts"))
from train_stacking import (
    generate_predictions, build_stacking_features, sliding_window_inference,
    postprocess_prediction, StackingClassifier,
    NNUNET_PROBS_DIR,
)


def load_stacking_model(model_path, device):
    """Load trained stacking classifier."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    in_channels = checkpoint['in_channels']
    model = StackingClassifier(in_channels=in_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  Loaded stacking model: {checkpoint.get('dice', 0):.4f} val Dice, "
          f"epoch {checkpoint.get('epoch', '?')}")
    print(f"  Models: {checkpoint['model_names']}")
    return model, checkpoint['model_names']


def add_nnunet_test_probs(cache_dir):
    """Add nnU-Net multi-fold test predictions to cache.

    For test cases, nnU-Net uses all 5 folds averaged (built-in ensemble).
    If probs aren't generated yet, falls back to binary predictions.
    """
    # Check for test probs
    test_probs_dir = NNUNET_PROBS_DIR / 'test'

    # Try binary predictions from any fold as fallback
    nnunet_results = (ROOT / 'nnUNet' / 'nnUNet_results' / 'Dataset001_BrainMets'
                      / 'nnUNetTrainer__nnUNetPlans__3d_fullres')

    cache_dir = Path(cache_dir)
    updated = 0
    skipped = 0

    for cache_file in sorted(cache_dir.glob('*.npz')):
        case_id = cache_file.stem
        try:
            existing = np.load(cache_file)
            if 'nnunet' in existing:
                skipped += 1
                continue

            # Try prob file first
            pred = None
            if test_probs_dir.exists():
                prob_path = test_probs_dir / f'{case_id}.npz'
                if prob_path.exists():
                    prob_data = np.load(prob_path)
                    pred = prob_data['probabilities'][1].transpose(2, 1, 0).astype(np.float32)

            # Fallback: binary prediction from validation dirs
            if pred is None:
                for fold in range(5):
                    bin_path = nnunet_results / f'fold_{fold}' / 'validation' / f'{case_id}.nii.gz'
                    if bin_path.exists():
                        pred = np.asarray(nib.load(str(bin_path)).dataobj, dtype=np.float32)
                        break

            if pred is None:
                continue

            mask_shape = existing['mask'].shape
            if pred.shape != mask_shape:
                from scipy.ndimage import zoom
                pred = zoom(pred, [t / s for t, s in zip(mask_shape, pred.shape)], order=1)

            save_dict = {k: existing[k] for k in existing.files}
            save_dict['nnunet'] = pred.astype(np.float16)
            np.savez_compressed(cache_file, **save_dict)
            updated += 1
        except Exception as e:
            print(f"  WARNING: {case_id} failed ({e})")
            continue

    print(f"  nnU-Net test preds: {updated} added, {skipped} already cached")
    return updated + skipped


def main():
    parser = argparse.ArgumentParser(description="Generate test predictions")
    parser.add_argument('--method', type=str, default='stacking',
                        help="Prediction method (default: stacking)")
    parser.add_argument('--threshold', type=float, default=None,
                        help="Override threshold (default: from results JSON)")
    parser.add_argument('--min-component', type=int, default=20,
                        help="Min connected component size (default: 20)")
    parser.add_argument('--stacking-patch', type=int, default=32)
    parser.add_argument('--stacking-overlap', type=float, default=0.5)
    parser.add_argument('--output-dir', type=str, default='outputs/test_predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("  TEST PREDICTION GENERATION")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load stacking model
    model_path = ROOT / 'model' / 'stacking_v4_classifier.pth'
    stacking_model, model_names = load_stacking_model(model_path, device)

    # Get threshold from results
    if args.threshold is not None:
        threshold = args.threshold
    else:
        results_path = ROOT / 'model' / 'stacking_v4_results.json'
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            threshold = results.get('_thresholds', {}).get(args.method, 0.5)
        else:
            threshold = 0.9  # Default for stacking
    print(f"  Method: {args.method}, threshold: {threshold}")

    # =========================================================================
    # Step 1: Generate base model predictions for test cases
    # =========================================================================
    print(f"\n{'='*60}")
    print("  Step 1: Base model predictions for test cases")
    print(f"{'='*60}")

    test_data_dir = str(ROOT / 'data' / 'preprocessed_256' / 'test')
    test_cache_dir = ROOT / 'model' / 'stacking_cache_v4_test'

    base_models = [m for m in model_names if m != 'nnunet']
    generate_predictions(
        test_data_dir, test_cache_dir, device,
        selected_models=base_models, overlap=0.5, tta=False, v2=True
    )

    # =========================================================================
    # Step 2: Add nnU-Net predictions
    # =========================================================================
    if 'nnunet' in model_names:
        print(f"\n{'='*60}")
        print("  Step 2: Add nnU-Net predictions")
        print(f"{'='*60}")
        add_nnunet_test_probs(test_cache_dir)

    # =========================================================================
    # Step 3: Generate predictions
    # =========================================================================
    print(f"\n{'='*60}")
    print("  Step 3: Stacking inference on test cases")
    print(f"{'='*60}")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get reference NIfTI for affine
    test_images = ROOT / 'nnUNet' / 'nnUNet_raw' / 'Dataset001_BrainMets' / 'imagesTs'

    test_files = sorted(test_cache_dir.glob('*.npz'))
    print(f"  Test cases: {len(test_files)}")

    n_saved = 0
    n_skipped = 0

    for cache_file in tqdm(test_files, desc="Predicting"):
        case_id = cache_file.stem
        try:
            features, preds, _ = build_stacking_features(cache_file, model_names)
        except Exception as e:
            print(f"  WARNING: {case_id} skipped ({e})")
            n_skipped += 1
            continue

        if args.method == 'stacking':
            prob_map = sliding_window_inference(
                stacking_model, features, args.stacking_patch, device,
                overlap=args.stacking_overlap
            )
        elif args.method == 'simple_average':
            prob_map = preds.mean(axis=0)
        elif args.method == 'nnunet' and 'nnunet' in model_names:
            idx = model_names.index('nnunet')
            prob_map = preds[idx]
        else:
            prob_map = preds.mean(axis=0)

        # Threshold + postprocess
        pred_bin = (prob_map > threshold).astype(np.uint8)
        if args.min_component > 0:
            pred_bin = postprocess_prediction(pred_bin, min_size=args.min_component).astype(np.uint8)

        # Get affine from original test NIfTI
        ref_path = test_images / f'{case_id}_0000.nii.gz'
        if ref_path.exists():
            affine = nib.load(str(ref_path)).affine
        else:
            affine = np.eye(4)

        # Save
        out_nii = nib.Nifti1Image(pred_bin, affine)
        out_path = output_dir / f'{case_id}.nii.gz'
        nib.save(out_nii, str(out_path))
        n_saved += 1

    print(f"\n  Saved {n_saved} predictions to {output_dir}")
    if n_skipped > 0:
        print(f"  ({n_skipped} skipped)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Method: {args.method}")
    print(f"  Threshold: {threshold}")
    print(f"  Min component: {args.min_component}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
