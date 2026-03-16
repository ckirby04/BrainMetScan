"""
Generate nnU-Net ResEncM experiment plans for Dataset001_BrainMets.

Uses nnUNetPlannerResEncM to create nnUNetResEncUNetMPlans.json in the
preprocessed directory. No re-preprocessing is needed — ResEncM reuses
the existing nnUNetPlans_3d_fullres preprocessed data.

The gpu_memory_target_in_gb is set to 16 to leverage the full RTX 5060 Ti
VRAM, enabling larger patches/batches than the default 8 GB target.

Usage:
    python scripts/plan_resencm.py
    python scripts/plan_resencm.py --gpu-mem 12   # Conservative VRAM target
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# nnU-Net directories
NNUNET_BASE = ROOT / 'nnUNet'
NNUNET_RAW = NNUNET_BASE / 'nnUNet_raw'
NNUNET_PREPROCESSED = NNUNET_BASE / 'nnUNet_preprocessed'
NNUNET_RESULTS = NNUNET_BASE / 'nnUNet_results'

DATASET_ID = 1
DATASET_NAME = 'Dataset001_BrainMets'


def set_nnunet_env():
    """Set nnU-Net environment variables."""
    os.environ['nnUNet_raw'] = str(NNUNET_RAW)
    os.environ['nnUNet_preprocessed'] = str(NNUNET_PREPROCESSED)
    os.environ['nnUNet_results'] = str(NNUNET_RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="Generate nnU-Net ResEncM experiment plans")
    parser.add_argument('--gpu-mem', type=float, default=16,
                        help="GPU memory target in GB (default: 16 for 5060 Ti)")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing plans without prompting")
    args = parser.parse_args()

    print("=" * 70)
    print("  nnU-Net ResEncM Experiment Planner")
    print("=" * 70)

    set_nnunet_env()

    # Verify dataset exists
    dataset_dir = NNUNET_PREPROCESSED / DATASET_NAME
    if not dataset_dir.exists():
        print(f"\nERROR: Preprocessed dataset not found: {dataset_dir}")
        print("Run nnU-Net preprocessing first.")
        sys.exit(1)

    existing_plans = dataset_dir / 'nnUNetPlans.json'
    if not existing_plans.exists():
        print(f"\nERROR: Standard plans not found: {existing_plans}")
        print("Run standard nnU-Net planning first.")
        sys.exit(1)

    print(f"  Dataset: {DATASET_NAME}")
    print(f"  GPU memory target: {args.gpu_mem} GB")
    print(f"  Preprocessed dir: {dataset_dir}")

    # Check if ResEncM plans already exist
    resencm_plans = dataset_dir / 'nnUNetResEncUNetMPlans.json'
    if resencm_plans.exists() and not args.force:
        print(f"\n  WARNING: ResEncM plans already exist: {resencm_plans}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # Import and run planner
    print("\n  Importing nnU-Net ResEncM planner...")
    from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
        nnUNetPlannerResEncM,
    )

    print(f"  Creating planner with gpu_memory_target={args.gpu_mem} GB...")
    planner = nnUNetPlannerResEncM(
        dataset_name_or_id=DATASET_NAME,
        gpu_memory_target_in_gb=args.gpu_mem,
    )

    print("  Running experiment planning...")
    plans = planner.plan_experiment()

    # Verify output
    if resencm_plans.exists():
        import json
        with open(resencm_plans) as f:
            plan_data = json.load(f)

        configs = plan_data.get('configurations', {})
        print(f"\n  Plans saved: {resencm_plans}")
        print(f"  Plans identifier: {planner.plans_identifier}")
        print(f"  Configurations: {list(configs.keys())}")

        if '3d_fullres' in configs:
            cfg = configs['3d_fullres']
            print(f"\n  3d_fullres configuration:")
            print(f"    Patch size: {cfg.get('patch_size', 'N/A')}")
            print(f"    Batch size: {cfg.get('batch_size', 'N/A')}")
            print(f"    Architecture: {cfg.get('architecture', {}).get('network_class_name', 'N/A')}")
            arch = cfg.get('architecture', {})
            n_stages = len(arch.get('n_blocks_per_stage', []))
            print(f"    Stages: {n_stages}")
            print(f"    Blocks per stage: {arch.get('n_blocks_per_stage', 'N/A')}")
            print(f"    Features per stage: {arch.get('features_per_stage', 'N/A')}")
    else:
        print(f"\n  ERROR: Plans file was not created at {resencm_plans}")
        sys.exit(1)

    print(f"\n  Next step: Train fold 0 with:")
    print(f"    python scripts/train_resencm.py --fold 0")


if __name__ == '__main__':
    main()
