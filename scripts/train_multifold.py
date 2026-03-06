"""
nnU-Net multi-fold training script.

Trains folds 1-4 sequentially (fold 0 already complete).
Resumable — detects which folds are done and skips them.
Uses subprocess to launch training (avoids Windows spawn/multiprocessing hangs).
Auto-restarts when epoch times degrade (Windows worker degradation workaround).

Usage:
    python scripts/train_multifold.py                  # Train all remaining folds
    python scripts/train_multifold.py --folds 1 2      # Train specific folds
    python scripts/train_multifold.py --status          # Check fold status
    python scripts/train_multifold.py --eval            # Evaluate all completed folds
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
NNUNET_DIR = PROJECT / "nnUNet"
RAW_DIR = NNUNET_DIR / "nnUNet_raw"
PREPROCESSED_DIR = NNUNET_DIR / "nnUNet_preprocessed"
RESULTS_DIR = NNUNET_DIR / "nnUNet_results"
TRAINER_DIR = RESULTS_DIR / "Dataset001_BrainMets" / "nnUNetTrainer__nnUNetPlans__3d_fullres"

DATASET_ID = 1
CONFIGURATION = "3d_fullres"
ALL_FOLDS = [0, 1, 2, 3, 4]

# Auto-restart when epoch time exceeds this multiple of the baseline
SLOWDOWN_THRESHOLD = 2.0  # restart if epoch takes >2x the baseline
WARMUP_EPOCHS = 5         # ignore first N epochs (data loading warmup)


def get_env():
    """Get environment with nnU-Net variables set."""
    env = os.environ.copy()
    env["nnUNet_raw"] = str(RAW_DIR)
    env["nnUNet_preprocessed"] = str(PREPROCESSED_DIR)
    env["nnUNet_results"] = str(RESULTS_DIR)
    env.setdefault("nnUNet_n_proc_DA", "12")
    return env


def get_fold_status(fold: int) -> dict:
    """Check training status for a fold."""
    fold_dir = TRAINER_DIR / f"fold_{fold}"
    status = {
        "fold": fold,
        "exists": fold_dir.exists(),
        "has_final": (fold_dir / "checkpoint_final.pth").exists(),
        "has_best": (fold_dir / "checkpoint_best.pth").exists(),
        "has_validation": (fold_dir / "validation" / "summary.json").exists(),
        "epoch": None,
    }

    # Try to read current epoch from debug.json
    debug_file = fold_dir / "debug.json"
    if debug_file.exists():
        try:
            with open(debug_file) as f:
                debug = json.load(f)
            status["epoch"] = debug.get("current_epoch", None)
        except (json.JSONDecodeError, KeyError):
            pass

    # Check training logs for latest epoch
    if status["epoch"] is None:
        log_files = sorted(fold_dir.glob("training_log_*.txt"))
        if log_files:
            try:
                text = log_files[-1].read_text()
                for line in reversed(text.splitlines()):
                    if line.startswith("Epoch "):
                        status["epoch"] = int(line.split()[1])
                        break
            except Exception:
                pass

    status["complete"] = status["has_final"] and status["has_validation"]
    return status


def get_latest_log(fold: int) -> Path | None:
    """Get the most recent training log file for a fold."""
    fold_dir = TRAINER_DIR / f"fold_{fold}"
    logs = sorted(fold_dir.glob("training_log_*.txt"))
    return logs[-1] if logs else None


def parse_epoch_times(log_path: Path, after_line: int = 0) -> list[tuple[int, float]]:
    """Parse (epoch_number, epoch_time_seconds) from a training log."""
    results = []
    try:
        lines = log_path.read_text().splitlines()
        current_epoch = None
        for i, line in enumerate(lines):
            if i < after_line:
                continue
            m = re.search(r"Epoch (\d+)\s*$", line)
            if m:
                current_epoch = int(m.group(1))
            m = re.search(r"Epoch time: ([\d.]+) s", line)
            if m and current_epoch is not None:
                results.append((current_epoch, float(m.group(1))))
                current_epoch = None
    except Exception:
        pass
    return results


def print_status():
    """Print status of all folds."""
    print("\n" + "=" * 60)
    print("  nnU-Net Multi-Fold Training Status")
    print("=" * 60)
    print(f"\n  {'Fold':<6} {'Status':<14} {'Epoch':<10} {'Checkpoint':<14} {'Validated'}")
    print("  " + "-" * 58)

    all_complete = True
    for fold in ALL_FOLDS:
        s = get_fold_status(fold)
        if s["complete"]:
            status_str = "COMPLETE"
        elif s["has_final"]:
            status_str = "NEEDS EVAL"
        elif s["exists"]:
            status_str = "IN PROGRESS"
            all_complete = False
        else:
            status_str = "NOT STARTED"
            all_complete = False

        epoch_str = "1000/1000" if s["complete"] else (
            f"{s['epoch']}/1000" if s["epoch"] is not None else "-")
        ckpt_str = "final+best" if s["has_final"] and s["has_best"] else (
            "best" if s["has_best"] else ("partial" if s["exists"] else "-"))

        print(f"  {fold:<6} {status_str:<14} {epoch_str:<10} {ckpt_str:<14} {'yes' if s['has_validation'] else 'no'}")

    print()
    if all_complete:
        print("  All folds complete! Ready for ensemble inference.")
    print()


def train_fold(fold: int):
    """Train a single fold with auto-restart on degradation."""
    s = get_fold_status(fold)

    if s["complete"]:
        print(f"\n  Fold {fold} already complete. Skipping.")
        return True

    if s["has_final"]:
        print(f"\n  Fold {fold} has checkpoint_final.pth but no validation. Running validation...")
        return run_validation(fold)

    restart_count = 0
    while True:
        s = get_fold_status(fold)
        resuming = s["exists"] and (s["has_best"] or s["epoch"] is not None)
        action = "Resuming" if resuming else "Starting"
        epoch_info = f" from epoch ~{s['epoch']}" if s.get("epoch") else ""

        if restart_count > 0:
            print(f"\n  [Auto-restart #{restart_count}] {action} fold {fold}{epoch_info}")
        else:
            print(f"\n{'=' * 60}")
            print(f"  {action} fold {fold}{epoch_info}")
            print(f"  Auto-restart enabled (threshold: {SLOWDOWN_THRESHOLD}x baseline)")
            print(f"{'=' * 60}\n")

        cmd = [
            sys.executable, "-m", "nnunetv2.run.run_training",
            str(DATASET_ID), CONFIGURATION, str(fold),
        ]
        if resuming:
            cmd.append("--c")

        # Launch training subprocess
        proc = subprocess.Popen(cmd, env=get_env())

        # Monitor for degradation
        exit_reason = monitor_training(proc, fold)

        if exit_reason == "complete":
            print(f"\n  Fold {fold} training complete!")
            return True
        elif exit_reason == "degraded":
            restart_count += 1
            print(f"\n  Epoch time degraded — restarting with fresh workers...")
            continue
        elif exit_reason == "failed":
            print(f"\n  ERROR: Fold {fold} training failed")
            return False
        elif exit_reason == "interrupted":
            print(f"\n  Training interrupted by user")
            return False


def monitor_training(proc: subprocess.Popen, fold: int) -> str:
    """Monitor training log for epoch time degradation.

    Returns: 'complete', 'degraded', 'failed', or 'interrupted'
    """
    log_path = None
    last_line_count = 0
    baseline_time = None
    consecutive_slow = 0
    epochs_seen = 0

    try:
        while proc.poll() is None:
            time.sleep(30)  # check every 30 seconds

            # Find latest log
            if log_path is None or not log_path.exists():
                log_path = get_latest_log(fold)
                if log_path is None:
                    continue

            # Parse new epoch times
            epoch_times = parse_epoch_times(log_path, after_line=last_line_count)
            if not epoch_times:
                continue

            # Update line count to avoid re-parsing
            try:
                last_line_count = len(log_path.read_text().splitlines())
            except Exception:
                pass

            for epoch_num, epoch_sec in epoch_times:
                epochs_seen += 1

                # Skip warmup epochs
                if epochs_seen <= WARMUP_EPOCHS:
                    continue

                # Establish baseline from first post-warmup epochs
                if baseline_time is None:
                    baseline_time = epoch_sec
                    print(f"  [Monitor] Baseline epoch time: {baseline_time:.0f}s")
                    continue

                # Update baseline with exponential moving average (slow)
                baseline_time = 0.95 * baseline_time + 0.05 * min(epoch_sec, baseline_time * 1.2)

                # Check for degradation
                ratio = epoch_sec / baseline_time
                if ratio > SLOWDOWN_THRESHOLD:
                    consecutive_slow += 1
                    if consecutive_slow >= 3:  # 3 consecutive slow epochs = restart
                        print(f"  [Monitor] Epoch {epoch_num}: {epoch_sec:.0f}s "
                              f"({ratio:.1f}x baseline {baseline_time:.0f}s) — "
                              f"degraded for {consecutive_slow} epochs")
                        # Kill the training process
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        return "degraded"
                else:
                    consecutive_slow = 0

    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return "interrupted"

    # Process exited on its own
    if proc.returncode == 0:
        return "complete"
    else:
        return "failed"


def run_validation(fold: int):
    """Run validation only for a completed fold via subprocess."""
    print(f"  Running validation for fold {fold}...")
    cmd = [
        sys.executable, "-m", "nnunetv2.run.run_training",
        str(DATASET_ID), CONFIGURATION, str(fold),
        "--val",
    ]
    result = subprocess.run(cmd, env=get_env())
    return result.returncode == 0


def evaluate_all():
    """Run evaluate_nnunet.py for each completed fold."""
    eval_script = PROJECT / "scripts" / "evaluate_nnunet.py"
    if not eval_script.exists():
        print("  ERROR: scripts/evaluate_nnunet.py not found")
        return

    completed = [f for f in ALL_FOLDS if get_fold_status(f)["complete"]]
    if not completed:
        print("  No completed folds to evaluate.")
        return

    print(f"\n  Evaluating {len(completed)} completed fold(s): {completed}")
    for fold in completed:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating fold {fold}")
        print(f"{'=' * 60}")
        subprocess.run([sys.executable, str(eval_script), "--fold", str(fold)], env=get_env())


def main():
    parser = argparse.ArgumentParser(description="nnU-Net multi-fold training")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Which folds to train (default: 1 2 3 4)")
    parser.add_argument("--status", action="store_true",
                        help="Show fold status and exit")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate all completed folds and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.eval:
        evaluate_all()
        return

    # Training mode
    print_status()

    folds_to_train = [f for f in args.folds if not get_fold_status(f)["complete"]]
    if not folds_to_train:
        print("  All requested folds are complete!")
        evaluate_all()
        return

    print(f"  Will train folds: {folds_to_train}")
    total_hours = len(folds_to_train) * 27
    print(f"  Estimated total time: ~{total_hours} hours ({total_hours / 24:.1f} days)\n")

    for i, fold in enumerate(folds_to_train):
        print(f"\n  [{i + 1}/{len(folds_to_train)}] Training fold {fold}...")
        t0 = time.time()
        success = train_fold(fold)
        elapsed = (time.time() - t0) / 3600
        print(f"  Fold {fold} took {elapsed:.1f} hours")
        if not success:
            print(f"  Stopping due to fold {fold} failure.")
            break

    print("\n" + "=" * 60)
    print("  Training session complete!")
    print("=" * 60)
    print_status()


if __name__ == "__main__":
    main()
