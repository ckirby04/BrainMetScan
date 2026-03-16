"""
Parallel Training Launcher
==========================
Launches both training strategies in parallel:
- Strategy A (Curriculum Learning) on GPU 0
- Strategy B (Pretraining + Fine-tuning) on GPU 1

If only one GPU is available, runs strategies sequentially.

Usage:
    python scripts/launch_parallel_training.py
    python scripts/launch_parallel_training.py --sequential  # Force sequential
"""

import subprocess
import sys
import os
import torch
import argparse
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_available_gpus():
    """Get number of available GPUs"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def launch_strategy(strategy: str, gpu: int, log_file: Path, show_output: bool = True):
    """Launch a training strategy as a subprocess"""
    script_dir = Path(__file__).parent
    train_script = script_dir / 'train_superset.py'

    cmd = [
        sys.executable,
        str(train_script),
        '--strategy', strategy,
        '--gpu', str(gpu),
    ]

    logger.info(f"Launching Strategy {strategy} on GPU {gpu}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Command: {' '.join(cmd)}")

    if show_output:
        # Show output in real-time while also logging to file
        log_handle = open(log_file, 'w')
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(script_dir.parent),
            bufsize=1,
            universal_newlines=True
        )
        return process, log_handle
    else:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(script_dir.parent)
            )
        return process, None


def run_with_live_output(process, log_handle):
    """Stream process output to both console and log file"""
    try:
        for line in process.stdout:
            print(line, end='')  # Print to console
            if log_handle:
                log_handle.write(line)
                log_handle.flush()
        process.wait()
    finally:
        if log_handle:
            log_handle.close()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description='Launch parallel training')
    parser.add_argument('--sequential', action='store_true',
                        help='Force sequential execution')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    logs_dir = base_dir / 'logs' / 'parallel_training'
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    num_gpus = get_available_gpus()
    logger.info(f"Available GPUs: {num_gpus}")

    if num_gpus >= 2 and not args.sequential:
        # Parallel execution
        logger.info("=" * 60)
        logger.info("PARALLEL TRAINING MODE")
        logger.info("Strategy A (Curriculum) on GPU 0")
        logger.info("Strategy B (Pretrain+Finetune) on GPU 1")
        logger.info("=" * 60)

        log_a = logs_dir / f'strategy_a_{timestamp}.log'
        log_b = logs_dir / f'strategy_b_{timestamp}.log'

        proc_a = launch_strategy('A', 0, log_a)
        proc_b = launch_strategy('B', 1, log_b)

        logger.info("\nBoth strategies launched. Waiting for completion...")
        logger.info(f"Monitor Strategy A: tail -f {log_a}")
        logger.info(f"Monitor Strategy B: tail -f {log_b}")

        # Wait for both to complete
        proc_a.wait()
        proc_b.wait()

        logger.info("\n" + "=" * 60)
        logger.info("BOTH STRATEGIES COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Strategy A exit code: {proc_a.returncode}")
        logger.info(f"Strategy B exit code: {proc_b.returncode}")

    else:
        # Sequential execution
        logger.info("=" * 60)
        logger.info("SEQUENTIAL TRAINING MODE")
        logger.info(f"Running on GPU 0 (or CPU)")
        logger.info("=" * 60)

        gpu = 0 if num_gpus > 0 else -1

        # Strategy A first
        log_a = logs_dir / f'strategy_a_{timestamp}.log'
        logger.info("\n[1/2] Running Strategy A: Curriculum Learning")
        logger.info("-" * 60)
        proc_a, log_handle_a = launch_strategy('A', gpu, log_a, show_output=True)
        exit_code_a = run_with_live_output(proc_a, log_handle_a)
        logger.info("-" * 60)
        logger.info(f"Strategy A complete (exit code: {exit_code_a})")

        # Strategy B second
        log_b = logs_dir / f'strategy_b_{timestamp}.log'
        logger.info("\n[2/2] Running Strategy B: Pretraining + Fine-tuning")
        logger.info("-" * 60)
        proc_b, log_handle_b = launch_strategy('B', gpu, log_b, show_output=True)
        exit_code_b = run_with_live_output(proc_b, log_handle_b)
        logger.info("-" * 60)
        logger.info(f"Strategy B complete (exit code: {exit_code_b})")

        logger.info("\n" + "=" * 60)
        logger.info("ALL TRAINING COMPLETE")
        logger.info("=" * 60)

    # Summary
    logger.info("\nNext steps:")
    logger.info("1. Check model/curriculum_final.pth (Strategy A result)")
    logger.info("2. Check model/pretrain_final.pth (Strategy B result)")
    logger.info("3. Run evaluation: python evaluate_model.py --model model/curriculum_final.pth")
    logger.info("4. Consider ensemble of both models for best results")


if __name__ == '__main__':
    main()
