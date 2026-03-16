"""
Overnight training: run 2 nnU-Net 2D folds in parallel, one per GPU.

GPU 0 (5060 Ti) + GPU 1 (3070 Ti) each train a separate fold simultaneously.
After both finish, starts the next pair. Then generates probs and retrains stacking.

Resumable — detects completed/in-progress folds automatically.

Usage:
    python scripts/train_overnight.py          # Train folds 1-4 in pairs, then probs+stacking
    python scripts/train_overnight.py --status  # Check progress
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT / "scripts"
NNUNET_DIR = PROJECT / "nnUNet"

DATASET = "Dataset001_BrainMets"
CONFIG = "2d"
TRAINER = "nnUNetTrainer_500ep"
DEFAULT_TRAINER = "nnUNetTrainer"

# Both trainer dirs (fold 0 used default, folds 1-4 use 500ep)
DEFAULT_TRAINER_DIR = (NNUNET_DIR / "nnUNet_results" / DATASET
                       / f"{DEFAULT_TRAINER}__nnUNetPlans__{CONFIG}")
SHORT_TRAINER_DIR = (NNUNET_DIR / "nnUNet_results" / DATASET
                     / f"{TRAINER}__nnUNetPlans__{CONFIG}")


def get_env():
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_DIR / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(NNUNET_DIR / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(NNUNET_DIR / "nnUNet_results")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Reduce DA workers per fold (2 folds sharing CPU = need fewer each)
    env["nnUNet_n_proc_DA"] = "6"
    return env


def fold_done(fold):
    """Check if fold is complete in either trainer dir."""
    for d in [DEFAULT_TRAINER_DIR, SHORT_TRAINER_DIR]:
        f = d / f"fold_{fold}"
        if (f / "checkpoint_final.pth").exists():
            return True
    return False


def fold_resumable(fold):
    """Check if fold has a checkpoint to resume from."""
    f = SHORT_TRAINER_DIR / f"fold_{fold}"
    return (f / "checkpoint_latest.pth").exists()


def get_epoch(fold):
    """Get current epoch from the latest training log."""
    for trainer_dir in [SHORT_TRAINER_DIR, DEFAULT_TRAINER_DIR]:
        fold_dir = trainer_dir / f"fold_{fold}"
        logs = sorted(fold_dir.glob("training_log_*.txt"))
        if not logs:
            continue
        try:
            text = logs[-1].read_text()
            for line in reversed(text.splitlines()):
                if line.strip().startswith("Epoch ") and "time" not in line:
                    parts = line.strip().split()
                    return int(parts[1])
        except Exception:
            pass
    return None


def print_status():
    print("\n" + "=" * 60)
    print("  Overnight Training Status")
    print("=" * 60)
    print(f"\n  {'Fold':<6} {'Status':<14} {'Epoch':<12} {'Trainer'}")
    print("  " + "-" * 50)
    for fold in range(5):
        if fold_done(fold):
            ep = "done"
            trainer = DEFAULT_TRAINER if fold == 0 else TRAINER
            status = "COMPLETE"
        elif fold_resumable(fold):
            ep_num = get_epoch(fold)
            ep = f"~{ep_num}/500" if ep_num else "?"
            trainer = TRAINER
            status = "IN PROGRESS"
        else:
            ep = "-"
            trainer = TRAINER if fold > 0 else DEFAULT_TRAINER
            status = "NOT STARTED"
        print(f"  {fold:<6} {status:<14} {ep:<12} {trainer}")
    print()


def train_pair(fold_a, fold_b):
    """Train two folds in parallel, one per GPU."""
    env = get_env()
    procs = {}

    for gpu, fold in [(0, fold_a), (1, fold_b)]:
        if fold is None or fold_done(fold):
            continue

        gpu_name = "5060 Ti" if gpu == 0 else "3070 Ti"
        resuming = fold_resumable(fold)
        action = "Resuming" if resuming else "Starting"
        print(f"  GPU {gpu} ({gpu_name}): {action} fold {fold}")

        fold_env = env.copy()
        fold_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_file = PROJECT / f"fold{fold}_log.txt"
        log_fh = open(log_file, "w")

        cmd = [sys.executable, "-m", "nnunetv2.run.run_training",
               DATASET, CONFIG, str(fold), "-tr", TRAINER]
        if resuming:
            cmd.append("--c")

        proc = subprocess.Popen(cmd, env=fold_env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs[fold] = (proc, log_fh, gpu, log_file)

    if not procs:
        print("  All folds in this pair already complete!")
        return True

    print(f"\n  Monitor logs with:")
    for fold, (_, _, gpu, log_file) in procs.items():
        print(f"    tail -f {log_file.name}    # fold {fold} (GPU {gpu})")
    print()

    # Wait for both to finish
    try:
        while procs:
            for fold in list(procs.keys()):
                proc, log_fh, gpu, log_file = procs[fold]
                ret = proc.poll()
                if ret is not None:
                    log_fh.close()
                    gpu_name = "5060 Ti" if gpu == 0 else "3070 Ti"
                    if ret == 0:
                        print(f"  Fold {fold} (GPU {gpu}, {gpu_name}) finished successfully!")
                    else:
                        print(f"  WARNING: Fold {fold} (GPU {gpu}) exited with code {ret}")
                        print(f"  Check {log_file} for details")
                    del procs[fold]
            if procs:
                time.sleep(30)
    except KeyboardInterrupt:
        print("\n  Interrupted — stopping training processes...")
        for fold, (proc, log_fh, _, _) in procs.items():
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_fh.close()
        print("  Stopped. Progress is saved — resume anytime.")
        return False

    return True


def consolidate_folds():
    """Copy 500ep fold results into default trainer dir for unified prob generation."""
    if not SHORT_TRAINER_DIR.exists():
        return

    DEFAULT_TRAINER_DIR.mkdir(parents=True, exist_ok=True)

    import shutil
    for fname in ["plans.json", "dataset.json", "dataset_fingerprint.json"]:
        src = SHORT_TRAINER_DIR / fname
        dst = DEFAULT_TRAINER_DIR / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    for fold in range(5):
        src = SHORT_TRAINER_DIR / f"fold_{fold}"
        dst = DEFAULT_TRAINER_DIR / f"fold_{fold}"
        if src.exists() and not dst.exists():
            print(f"  Consolidating fold_{fold} -> {DEFAULT_TRAINER}__nnUNetPlans__2d/")
            shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Overnight dual-GPU training")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--skip-stacking", action="store_true",
                        help="Stop after training, skip probs + stacking")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    t_start = time.time()

    print("\n" + "=" * 60)
    print("  OVERNIGHT DUAL-GPU TRAINING")
    print("  GPU 0 (5060 Ti) + GPU 1 (3070 Ti)")
    print("=" * 60)

    try:
        import torch
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i}: {torch.cuda.get_device_name(i)}")
    except Exception:
        pass

    print_status()

    # Phase 1: folds 1+2 in parallel
    remaining = [f for f in [1, 2, 3, 4] if not fold_done(f)]
    if len(remaining) == 0:
        print("  All folds complete!")
    else:
        # Train in pairs
        pairs = []
        for i in range(0, len(remaining), 2):
            a = remaining[i]
            b = remaining[i + 1] if i + 1 < len(remaining) else None
            pairs.append((a, b))

        for pair_idx, (fold_a, fold_b) in enumerate(pairs):
            pair_label = f"{fold_a}" + (f"+{fold_b}" if fold_b else "")
            print(f"\n{'='*60}")
            print(f"  PAIR {pair_idx+1}/{len(pairs)}: Folds {pair_label}")
            print(f"{'='*60}\n")

            t0 = time.time()
            ok = train_pair(fold_a, fold_b)
            elapsed = (time.time() - t0) / 3600
            print(f"  Pair took {elapsed:.1f} hours")

            if not ok:
                print("  Stopping due to interruption/failure.")
                print_status()
                return

    print_status()

    if args.skip_stacking:
        print("  Skipping probs + stacking (--skip-stacking)")
        return

    # Phase 2: consolidate + generate probs + retrain stacking
    all_done = all(fold_done(f) for f in range(5))
    if all_done:
        print("\n" + "=" * 60)
        print("  All folds complete! Running probs + stacking...")
        print("=" * 60)

        consolidate_folds()

        env = get_env()
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use 5060 Ti for inference

        print("\n  Generating 2D probability maps...")
        trainer_name = f"{DEFAULT_TRAINER}__nnUNetPlans__{CONFIG}"
        subprocess.run([
            sys.executable, str(SCRIPTS / "nnunet_probs.py"),
            "--mode", "val", "--trainer", trainer_name,
        ], env=env)

        print("\n  Retraining stacking classifier...")
        subprocess.run([
            sys.executable, str(SCRIPTS / "train_stacking.py"),
            "--v2", "--include-nnunet", "--include-2d",
            "--epochs", "100", "--stacking-patch", "32", "--stacking-overlap", "0.5",
        ], env=env)
    else:
        print("  Not all folds finished — run again to complete remaining folds.")

    total_hours = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Session complete — {total_hours:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
