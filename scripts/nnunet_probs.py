"""
Regenerate nnU-Net predictions with probability output.

Uses nnUNetPredictor to re-run inference on validation (per-fold) or test
cases with save_probabilities=True, producing .npz files with softmax
probability maps needed for cross-model ensemble fusion.

Validation mode (--mode val):
  For each completed fold, loads that fold's checkpoint and predicts on that
  fold's held-out validation cases. This avoids data leakage — each case is
  only predicted by the fold that never saw it during training.

Test mode (--mode test):
  Uses all available folds together (nnU-Net's built-in multi-fold averaging)
  to predict on test cases.

Output:
  model/nnunet_probs/fold_X/{case_id}.npz   (val mode)
  model/nnunet_probs/test/{case_id}.npz      (test mode)

Each .npz contains key 'probabilities' with shape [2, D, H, W]:
  channel 0 = background, channel 1 = foreground (metastasis)

Usage:
    python scripts/nnunet_probs.py --mode val                 # All available folds
    python scripts/nnunet_probs.py --mode val --folds 0 1     # Specific folds
    python scripts/nnunet_probs.py --mode test                # Test set
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# nnU-Net directories
NNUNET_BASE = ROOT / 'nnUNet'
NNUNET_RAW = NNUNET_BASE / 'nnUNet_raw'
NNUNET_PREPROCESSED = NNUNET_BASE / 'nnUNet_preprocessed'
NNUNET_RESULTS = NNUNET_BASE / 'nnUNet_results'

DATASET_NAME = 'Dataset001_BrainMets'
DEFAULT_TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

IMAGES_TR = NNUNET_RAW / DATASET_NAME / 'imagesTr'
IMAGES_TS = NNUNET_RAW / DATASET_NAME / 'imagesTs'
SPLITS_FILE = NNUNET_PREPROCESSED / DATASET_NAME / 'splits_final.json'

DEFAULT_OUTPUT_BASE = ROOT / 'model' / 'nnunet_probs'

CHANNELS = ['0000', '0001', '0002', '0003']


def set_nnunet_env():
    """Set nnU-Net environment variables."""
    os.environ['nnUNet_raw'] = str(NNUNET_RAW)
    os.environ['nnUNet_preprocessed'] = str(NNUNET_PREPROCESSED)
    os.environ['nnUNet_results'] = str(NNUNET_RESULTS)


def load_splits():
    """Load nnU-Net splits_final.json."""
    with open(SPLITS_FILE) as f:
        return json.load(f)


def get_available_folds(trainer_dir):
    """Return list of fold indices that have checkpoint_final.pth."""
    available = []
    for fold_idx in range(5):
        checkpoint = trainer_dir / f'fold_{fold_idx}' / 'checkpoint_final.pth'
        if checkpoint.exists():
            available.append(fold_idx)
    return available


def get_case_input_files(case_id, images_dir):
    """Get list of channel files for a case (nnU-Net format)."""
    files = []
    for ch in CHANNELS:
        f = images_dir / f'{case_id}_{ch}.nii.gz'
        if not f.exists():
            return None
        files.append(str(f))
    return files


def predict_val_fold(predictor_cls, trainer_dir, fold_idx, splits, output_base):
    """Run prediction for one fold's validation cases."""
    fold_output = output_base / f'fold_{fold_idx}'
    fold_output.mkdir(parents=True, exist_ok=True)

    # Get this fold's validation case IDs
    val_cases = splits[fold_idx]['val']
    print(f"\n  Fold {fold_idx}: {len(val_cases)} validation cases")

    # Check which cases still need prediction
    cases_to_predict = []
    input_lists = []
    output_files = []
    for case_id in val_cases:
        out_npz = fold_output / f'{case_id}.npz'
        if out_npz.exists():
            continue
        input_files = get_case_input_files(case_id, IMAGES_TR)
        if input_files is None:
            print(f"    WARNING: Missing input files for {case_id}, skipping")
            continue
        cases_to_predict.append(case_id)
        input_lists.append(input_files)
        output_files.append(str(fold_output / f'{case_id}'))

    already_done = len(val_cases) - len(cases_to_predict)
    if already_done > 0:
        print(f"    {already_done} cases already have probabilities, skipping")

    if not cases_to_predict:
        print(f"    All cases already predicted for fold {fold_idx}")
        return len(val_cases)

    print(f"    Predicting {len(cases_to_predict)} cases...")

    # Initialize predictor for this fold
    predictor = predictor_cls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=_get_device(),
        verbose=False,
        verbose_preprocessing=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(trainer_dir),
        use_folds=(fold_idx,),
        checkpoint_name='checkpoint_final.pth',
    )

    # Run prediction with probabilities
    t0 = time.time()
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_lists,
        output_folder_or_list_of_truncated_output_files=output_files,
        save_probabilities=True,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    elapsed = time.time() - t0
    print(f"    Fold {fold_idx} done in {elapsed/60:.1f} min "
          f"({elapsed/len(cases_to_predict):.1f}s/case)")

    # Clean up the .nii.gz files (we only need .npz)
    for case_id in cases_to_predict:
        nifti = fold_output / f'{case_id}.nii.gz'
        if nifti.exists():
            nifti.unlink()

    return len(val_cases)


def predict_test(predictor_cls, trainer_dir, available_folds, output_base):
    """Run prediction on test set using all available folds."""
    test_output = output_base / 'test'
    test_output.mkdir(parents=True, exist_ok=True)

    # Get test case IDs
    test_files = sorted(IMAGES_TS.glob('*_0000.nii.gz'))
    test_cases = [f.name.replace('_0000.nii.gz', '') for f in test_files]
    print(f"\n  Test set: {len(test_cases)} cases, using folds {available_folds}")

    # Check which cases still need prediction
    cases_to_predict = []
    input_lists = []
    output_files = []
    for case_id in test_cases:
        out_npz = test_output / f'{case_id}.npz'
        if out_npz.exists():
            continue
        input_files = get_case_input_files(case_id, IMAGES_TS)
        if input_files is None:
            print(f"    WARNING: Missing input files for {case_id}, skipping")
            continue
        cases_to_predict.append(case_id)
        input_lists.append(input_files)
        output_files.append(str(test_output / f'{case_id}'))

    already_done = len(test_cases) - len(cases_to_predict)
    if already_done > 0:
        print(f"    {already_done} cases already have probabilities, skipping")

    if not cases_to_predict:
        print(f"    All test cases already predicted")
        return

    print(f"    Predicting {len(cases_to_predict)} cases...")

    # Initialize predictor with all available folds (multi-fold ensemble)
    predictor = predictor_cls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=_get_device(),
        verbose=False,
        verbose_preprocessing=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(trainer_dir),
        use_folds=tuple(available_folds),
        checkpoint_name='checkpoint_final.pth',
    )

    t0 = time.time()
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_lists,
        output_folder_or_list_of_truncated_output_files=output_files,
        save_probabilities=True,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    elapsed = time.time() - t0
    print(f"    Test predictions done in {elapsed/60:.1f} min")

    # Clean up .nii.gz files
    for case_id in cases_to_predict:
        nifti = test_output / f'{case_id}.nii.gz'
        if nifti.exists():
            nifti.unlink()


def _get_device():
    """Get best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def verify_outputs(mode, folds_used, splits, output_base):
    """Verify .npz files have correct format."""
    print(f"\n  Verifying outputs...")
    errors = 0

    if mode == 'val':
        for fold_idx in folds_used:
            fold_dir = output_base / f'fold_{fold_idx}'
            val_cases = splits[fold_idx]['val']
            for case_id in val_cases:
                npz_path = fold_dir / f'{case_id}.npz'
                if not npz_path.exists():
                    continue
                try:
                    data = np.load(npz_path)
                    if 'probabilities' not in data:
                        print(f"    ERROR: {npz_path} missing 'probabilities' key")
                        errors += 1
                        continue
                    shape = data['probabilities'].shape
                    if len(shape) != 4 or shape[0] != 2:
                        print(f"    ERROR: {npz_path} unexpected shape {shape}")
                        errors += 1
                except Exception as e:
                    print(f"    ERROR: {npz_path}: {e}")
                    errors += 1
    else:
        test_dir = output_base / 'test'
        for npz_path in sorted(test_dir.glob('*.npz')):
            try:
                data = np.load(npz_path)
                if 'probabilities' not in data:
                    print(f"    ERROR: {npz_path} missing 'probabilities' key")
                    errors += 1
                    continue
                shape = data['probabilities'].shape
                if len(shape) != 4 or shape[0] != 2:
                    print(f"    ERROR: {npz_path} unexpected shape {shape}")
                    errors += 1
            except Exception as e:
                print(f"    ERROR: {npz_path}: {e}")
                errors += 1

    if errors == 0:
        print(f"    All outputs verified OK")
    else:
        print(f"    {errors} errors found!")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate nnU-Net predictions with probability output")
    parser.add_argument('--mode', choices=['val', 'test'], default='val',
                        help="val: per-fold validation, test: test set (default: val)")
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help="Specific folds to process (default: all available)")
    parser.add_argument('--trainer', type=str, default=DEFAULT_TRAINER,
                        help=f"Trainer dir name (default: {DEFAULT_TRAINER})")
    parser.add_argument('--no-verify', action='store_true',
                        help="Skip output verification")
    args = parser.parse_args()

    print("=" * 70)
    print("  nnU-Net Probability Prediction Generator")
    print("=" * 70)

    set_nnunet_env()

    trainer_dir = NNUNET_RESULTS / DATASET_NAME / args.trainer
    if not trainer_dir.exists():
        print(f"\nERROR: Trainer directory not found: {trainer_dir}")
        sys.exit(1)

    # Determine output directory based on trainer
    # Non-default trainers get their own output subdirectory (e.g. nnunet_probs_resencm)
    if 'ResEncUNetM' in args.trainer:
        output_base = ROOT / 'model' / 'nnunet_probs_resencm'
    elif args.trainer != DEFAULT_TRAINER:
        # Generic fallback: strip trainer prefix/suffix for a clean dir name
        suffix = args.trainer.replace('nnUNetTrainer__', '').replace('__3d_fullres', '')
        output_base = ROOT / 'model' / f'nnunet_probs_{suffix}'
    else:
        output_base = DEFAULT_OUTPUT_BASE

    print(f"Trainer: {args.trainer}")
    print(f"Output: {output_base}")

    # Check available folds
    available = get_available_folds(trainer_dir)
    print(f"Available folds with checkpoints: {available}")

    if not available:
        print("ERROR: No completed folds found!")
        sys.exit(1)

    # Filter to requested folds
    if args.folds is not None:
        missing = [f for f in args.folds if f not in available]
        if missing:
            print(f"WARNING: Folds {missing} not available, skipping")
        folds_to_use = [f for f in args.folds if f in available]
    else:
        folds_to_use = available

    if not folds_to_use:
        print("ERROR: No folds to process!")
        sys.exit(1)

    # Load splits
    splits = load_splits()
    print(f"Loaded splits: {len(splits)} folds, "
          f"{len(splits[0]['val'])} val cases in fold 0")

    # Import predictor (late import to avoid slow startup when just checking args)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    output_base.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.mode == 'val':
        print(f"\nMode: VALIDATION — predicting per-fold held-out cases")
        print(f"Folds: {folds_to_use}")
        total_cases = 0
        for fold_idx in folds_to_use:
            n = predict_val_fold(nnUNetPredictor, trainer_dir, fold_idx, splits,
                                 output_base)
            total_cases += n
        print(f"\n  Total: {total_cases} validation cases across {len(folds_to_use)} folds")

    elif args.mode == 'test':
        print(f"\nMode: TEST — multi-fold ensemble prediction")
        predict_test(nnUNetPredictor, trainer_dir, folds_to_use, output_base)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} minutes")

    # Verify
    if not args.no_verify:
        verify_outputs(args.mode, folds_to_use, splits, output_base)

    print(f"  Output directory: {output_base}")


if __name__ == '__main__':
    main()
