"""
Train nnU-Net ResEncM on Dataset001_BrainMets.

Uses the Python API (run_training) since CLI entry points are not on PATH.
Requires ResEncM plans to be generated first via plan_resencm.py.

Results stored at:
  nnUNet_results/Dataset001_BrainMets/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_X/

Expected timing on RTX 5060 Ti (16 GB):
  - ~120-150s/epoch x 1000 epochs = ~35-42 hours per fold
  - VRAM usage: ~10-14 GB (fits comfortably in 16 GB)

Usage:
    python scripts/train_resencm.py --fold 0              # Train fold 0
    python scripts/train_resencm.py --fold 0 --continue   # Resume training
    python scripts/train_resencm.py --fold 0 --val-only   # Only run validation
    python scripts/train_resencm.py --fold 0 --val-best   # Validate with best checkpoint
    python scripts/train_resencm.py --fold 0 --export-val-probs  # Export validation probabilities
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# nnU-Net directories
NNUNET_BASE = ROOT / 'nnUNet'
NNUNET_RAW = NNUNET_BASE / 'nnUNet_raw'
NNUNET_PREPROCESSED = NNUNET_BASE / 'nnUNet_preprocessed'
NNUNET_RESULTS = NNUNET_BASE / 'nnUNet_results'

DATASET_ID = 1
DATASET_NAME = 'Dataset001_BrainMets'
PLANS_IDENTIFIER = 'nnUNetResEncUNetMPlans'
CONFIGURATION = '3d_fullres'


def set_nnunet_env():
    """Set nnU-Net environment variables."""
    os.environ['nnUNet_raw'] = str(NNUNET_RAW)
    os.environ['nnUNet_preprocessed'] = str(NNUNET_PREPROCESSED)
    os.environ['nnUNet_results'] = str(NNUNET_RESULTS)
    # Windows optimization: limit data augmentation processes
    # Keep DA workers low to avoid RAM exhaustion (each worker loads full volumes)
    os.environ['nnUNet_n_proc_DA'] = os.environ.get('nnUNet_n_proc_DA', '2')
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


def main():
    parser = argparse.ArgumentParser(
        description="Train nnU-Net ResEncM on brain metastasis dataset")
    parser.add_argument('--fold', type=int, required=True,
                        help="Fold to train (0-4)")
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help="Resume training from checkpoint")
    parser.add_argument('--val-only', action='store_true',
                        help="Only run validation (skip training)")
    parser.add_argument('--val-best', action='store_true',
                        help="Use best checkpoint for validation (instead of final)")
    parser.add_argument('--export-val-probs', action='store_true',
                        help="Export validation probabilities as .npz")
    parser.add_argument('--num-gpus', type=int, default=1,
                        help="Number of GPUs (default: 1)")
    args = parser.parse_args()

    print("=" * 70)
    print("  nnU-Net ResEncM Training")
    print("=" * 70)

    set_nnunet_env()

    # Verify plans exist
    plans_file = NNUNET_PREPROCESSED / DATASET_NAME / f'{PLANS_IDENTIFIER}.json'
    if not plans_file.exists():
        print(f"\nERROR: ResEncM plans not found: {plans_file}")
        print("Run plan_resencm.py first:")
        print("  python scripts/plan_resencm.py")
        sys.exit(1)

    # Expected output directory
    trainer_dir = (NNUNET_RESULTS / DATASET_NAME /
                   f'nnUNetTrainer__{PLANS_IDENTIFIER}__{CONFIGURATION}')
    fold_dir = trainer_dir / f'fold_{args.fold}'

    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Plans: {PLANS_IDENTIFIER}")
    print(f"  Configuration: {CONFIGURATION}")
    print(f"  Fold: {args.fold}")
    print(f"  Continue: {args.continue_training}")
    print(f"  Val only: {args.val_only}")
    print(f"  Val best: {args.val_best}")
    print(f"  Export val probs: {args.export_val_probs}")
    print(f"  Output dir: {fold_dir}")
    print(f"  nnUNet_n_proc_DA: {os.environ.get('nnUNet_n_proc_DA', 'default')}")

    # Check for existing checkpoint
    checkpoint_final = fold_dir / 'checkpoint_final.pth'
    checkpoint_latest = fold_dir / 'checkpoint_latest.pth'
    if checkpoint_final.exists() and not args.val_only and not args.continue_training:
        print(f"\n  WARNING: checkpoint_final.pth already exists!")
        print(f"  Use --continue to resume or --val-only to just validate.")
        response = input("  Overwrite and retrain? [y/N]: ").strip().lower()
        if response != 'y':
            print("  Aborted.")
            sys.exit(0)
    elif checkpoint_latest.exists() and not args.continue_training and not args.val_only:
        print(f"\n  NOTE: checkpoint_latest.pth found. Use --continue to resume.")

    # GPU info
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {gpu_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
    else:
        print("\n  WARNING: No CUDA GPU detected! Training will be very slow.")

    # Import and run training
    print("\n  Starting training...")
    t0 = time.time()

    from nnunetv2.run.run_training import run_training

    run_training(
        dataset_name_or_id=DATASET_NAME,
        configuration=CONFIGURATION,
        fold=args.fold,
        trainer_class_name='nnUNetTrainer',
        plans_identifier=PLANS_IDENTIFIER,
        num_gpus=args.num_gpus,
        export_validation_probabilities=args.export_val_probs,
        continue_training=args.continue_training,
        only_run_validation=args.val_only,
        val_with_best=args.val_best,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    )

    elapsed = time.time() - t0
    hours = elapsed / 3600
    print(f"\n{'='*70}")
    print(f"  Training completed in {hours:.1f} hours ({elapsed/60:.0f} minutes)")
    print(f"  Output: {fold_dir}")

    if checkpoint_final.exists():
        size_mb = checkpoint_final.stat().st_size / 1e6
        print(f"  checkpoint_final.pth: {size_mb:.0f} MB")

    print(f"\n  Next steps:")
    print(f"    Evaluate:  python scripts/evaluate_nnunet.py "
          f"--trainer nnUNetTrainer__{PLANS_IDENTIFIER}__{CONFIGURATION}")
    if args.fold == 0:
        print(f"    Next fold: python scripts/train_resencm.py --fold 1")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
