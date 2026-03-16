"""
End-to-end nnU-Net 2D pipeline: train all folds, generate probs, retrain stacking.

Runs everything sequentially so you can monitor progress in one terminal.
Resumable — skips completed folds and existing probability files automatically.

Fold 0 uses the default nnUNetTrainer (1000 epochs, already in progress).
Folds 1-4 use nnUNetTrainer_500ep (500 epochs, ~12 hrs/fold instead of ~24).
Both save to the same results directory so nnU-Net treats them as one model.

Usage:
    python scripts/run_2d_pipeline.py              # Full pipeline
    python scripts/run_2d_pipeline.py --skip-train  # Skip training, just probs + stacking
    python scripts/run_2d_pipeline.py --skip-probs   # Skip probs, just stacking
    python scripts/run_2d_pipeline.py --epochs 500   # Set epoch count for remaining folds
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT / "scripts"

# nnU-Net env
NNUNET_DIR = PROJECT / "nnUNet"
ENV = os.environ.copy()
ENV["nnUNet_raw"] = str(NNUNET_DIR / "nnUNet_raw")
ENV["nnUNet_preprocessed"] = str(NNUNET_DIR / "nnUNet_preprocessed")
ENV["nnUNet_results"] = str(NNUNET_DIR / "nnUNet_results")
ENV.setdefault("nnUNet_n_proc_DA", "12")

# Force CUDA device ordering to match nvidia-smi (PCI bus order)
ENV["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Default: train on GPU 1 (3070 Ti), keep GPU 0 (5060 Ti) free for gaming
# Override with --gpu 0 if your display is on the 3070 Ti
DEFAULT_GPU = 1

DATASET = "Dataset001_BrainMets"
CONFIG = "2d"

# Default trainer dir (where fold 0 is already training)
DEFAULT_TRAINER = "nnUNetTrainer"
SHORT_TRAINER = "nnUNetTrainer_500ep"

DEFAULT_TRAINER_DIR = (NNUNET_DIR / "nnUNet_results" / DATASET
                       / f"{DEFAULT_TRAINER}__nnUNetPlans__{CONFIG}")
SHORT_TRAINER_DIR = (NNUNET_DIR / "nnUNet_results" / DATASET
                     / f"{SHORT_TRAINER}__nnUNetPlans__{CONFIG}")


def banner(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n")


def fold_done(fold, trainer_dir):
    d = trainer_dir / f"fold_{fold}"
    return (d / "checkpoint_final.pth").exists()


def fold_validated(fold, trainer_dir):
    d = trainer_dir / f"fold_{fold}"
    return (d / "validation" / "summary.json").exists()


def run(cmd, desc):
    """Run a command, streaming output live. Returns True on success."""
    print(f"  >> {desc}")
    print(f"  >> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=ENV)
    if result.returncode != 0:
        print(f"\n  !! FAILED (exit code {result.returncode}): {desc}")
        return False
    return True


# =========================================================================
# PHASE 1: Train all 5 folds
# =========================================================================
def train_all_folds(num_epochs):
    banner(f"PHASE 1: nnU-Net 2D Training (5 folds, {num_epochs} epochs)")

    # Decide trainer per fold
    use_short = num_epochs <= 500

    for fold in range(5):
        # Check both trainer dirs for completion
        if fold_done(fold, DEFAULT_TRAINER_DIR) or fold_done(fold, SHORT_TRAINER_DIR):
            print(f"  Fold {fold}: already complete, skipping")
            continue

        # Fold 0 is already in progress with default trainer — let it finish
        fold_0_in_progress = (
            fold == 0
            and (DEFAULT_TRAINER_DIR / "fold_0" / "checkpoint_latest.pth").exists()
        )

        if fold_0_in_progress:
            trainer = DEFAULT_TRAINER
            trainer_dir = DEFAULT_TRAINER_DIR
        elif use_short:
            trainer = SHORT_TRAINER
            trainer_dir = SHORT_TRAINER_DIR
        else:
            trainer = DEFAULT_TRAINER
            trainer_dir = DEFAULT_TRAINER_DIR

        # Check if checkpoint exists (resume)
        fold_dir = trainer_dir / f"fold_{fold}"
        resuming = (fold_dir / "checkpoint_latest.pth").exists()
        action = "Resuming" if resuming else "Starting"
        ep_label = "1000ep" if trainer == DEFAULT_TRAINER else f"{num_epochs}ep"

        banner(f"FOLD {fold}/4 — {action} ({ep_label}, {trainer})")
        t0 = time.time()

        cmd = [sys.executable, "-m", "nnunetv2.run.run_training",
               DATASET, CONFIG, str(fold), "-tr", trainer]
        if resuming:
            cmd.append("--c")

        ok = run(cmd, f"Train 2D fold {fold}")
        elapsed = (time.time() - t0) / 3600
        print(f"\n  Fold {fold} took {elapsed:.1f} hours")

        if not ok:
            print(f"  Stopping — fold {fold} failed.")
            return False

    # Consolidate: copy short trainer fold results into default trainer dir
    # so that prob generation and evaluation see all 5 folds in one place
    if use_short and SHORT_TRAINER_DIR.exists():
        banner("Consolidating fold results into single trainer directory")
        DEFAULT_TRAINER_DIR.mkdir(parents=True, exist_ok=True)

        # Copy plans and dataset files
        for fname in ["plans.json", "dataset.json", "dataset_fingerprint.json"]:
            src = SHORT_TRAINER_DIR / fname
            dst = DEFAULT_TRAINER_DIR / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        for fold in range(5):
            src_fold = SHORT_TRAINER_DIR / f"fold_{fold}"
            dst_fold = DEFAULT_TRAINER_DIR / f"fold_{fold}"
            if src_fold.exists() and not dst_fold.exists():
                print(f"  Copying fold_{fold} from {SHORT_TRAINER} -> {DEFAULT_TRAINER}")
                shutil.copytree(src_fold, dst_fold)

    # Summary
    banner("PHASE 1 COMPLETE")
    for fold in range(5):
        done_default = fold_done(fold, DEFAULT_TRAINER_DIR)
        done_short = fold_done(fold, SHORT_TRAINER_DIR)
        status = "DONE" if (done_default or done_short) else "MISSING"
        print(f"  Fold {fold}: {status}")
    print()
    return True


# =========================================================================
# PHASE 2: Generate probability maps
# =========================================================================
def generate_probs():
    banner("PHASE 2: Generate 2D Probability Maps")

    # Use the default trainer dir (which has all folds consolidated)
    trainer_name = f"{DEFAULT_TRAINER}__nnUNetPlans__{CONFIG}"
    cmd = [
        sys.executable, str(SCRIPTS / "nnunet_probs.py"),
        "--mode", "val",
        "--trainer", trainer_name,
    ]
    return run(cmd, "Generate validation probabilities for all folds")


# =========================================================================
# PHASE 3: Retrain stacking with 2D
# =========================================================================
def retrain_stacking():
    banner("PHASE 3: Retrain Stacking Classifier (with nnU-Net 3D + 2D)")
    cmd = [
        sys.executable, str(SCRIPTS / "train_stacking.py"),
        "--v2",
        "--include-nnunet",
        "--include-2d",
        "--epochs", "100",
        "--stacking-patch", "32",
        "--stacking-overlap", "0.5",
    ]
    return run(cmd, "Train stacking meta-learner with 2D added")


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Full nnU-Net 2D pipeline")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, go straight to prob generation")
    parser.add_argument("--skip-probs", action="store_true",
                        help="Skip training + probs, go straight to stacking")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Epochs for new folds (default: 500). Fold 0 in progress keeps 1000.")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU,
                        help=f"GPU index to train on (default: {DEFAULT_GPU}). "
                        "Uses nvidia-smi ordering (PCI bus ID).")
    args = parser.parse_args()

    ENV["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    t_start = time.time()

    banner("nnU-Net 2D Pipeline")
    print(f"  Phase 1: Train 5 folds ({args.epochs} epochs/fold)")
    print("  Phase 2: Generate probability maps (~30-60 min)")
    print("  Phase 3: Retrain stacking classifier (~15 min)")
    print(f"  Training GPU: nvidia-smi index {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")
    print(f"\n  GPU: ", end="")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"{torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        else:
            print("No CUDA GPU detected!")
    except ImportError:
        print("PyTorch not available")

    # Phase 1
    if not args.skip_train and not args.skip_probs:
        if not train_all_folds(args.epochs):
            print("\n  Pipeline stopped at Phase 1.")
            sys.exit(1)
    else:
        print("\n  Skipping Phase 1 (training)")

    # Phase 2
    if not args.skip_probs:
        if not generate_probs():
            print("\n  Pipeline stopped at Phase 2.")
            sys.exit(1)
    else:
        print("\n  Skipping Phase 2 (probability generation)")

    # Phase 3
    if not retrain_stacking():
        print("\n  Pipeline stopped at Phase 3.")
        sys.exit(1)

    total_hours = (time.time() - t_start) / 3600
    banner(f"PIPELINE COMPLETE — {total_hours:.1f} hours total")
    print("  Next steps:")
    print("    python scripts/compare_models.py   # Compare all models")
    print("    python scripts/train_2d.py --eval   # Evaluate 2D folds individually")
    print()


if __name__ == "__main__":
    main()
