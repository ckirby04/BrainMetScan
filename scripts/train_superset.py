"""
Superset Training Runner
========================
Implements the dual training strategy for brain metastasis segmentation:
- Strategy A: Curriculum Learning (supervised)
- Strategy B: Pretraining + Fine-tuning (transfer learning)

Can run strategies in parallel on separate GPUs or sequentially on single GPU.

Usage:
    python scripts/train_superset.py --strategy A      # Curriculum only
    python scripts/train_superset.py --strategy B      # Pretrain+finetune only
    python scripts/train_superset.py --strategy both   # Run sequentially
    python scripts/train_superset.py --strategy both --parallel  # Parallel (multi-GPU)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import logging
from tqdm import tqdm
import json
import shutil
from torch.utils.tensorboard import SummaryWriter

# Training state file for resume functionality
TRAINING_STATE_FILE = "model/training_state.json"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from segmentation.unet import LightweightUNet3D
from segmentation.dataset import BrainMetDataset
from segmentation.advanced_losses import (
    SmallLesionOptimizedLoss, TverskyLoss, ComboLoss, get_loss_function
)
from segmentation.augmentation import AugmentationPipeline, ValidationAugmentation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Superset uses t2 instead of bravo (matches UCSF dataset structure)
SUPERSET_SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']

# Target volume size for batching (all volumes resized to this)
TARGET_SIZE = (128, 128, 128)


def load_training_state(base_dir: Path) -> dict:
    """Load training state from file if it exists"""
    state_path = base_dir / TRAINING_STATE_FILE
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return None


def save_training_state(base_dir: Path, strategy: str, phase: int, epoch: int,
                        best_dice: float, completed: bool = False):
    """Save current training state for resume capability"""
    state_path = base_dir / TRAINING_STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        'strategy': strategy,
        'phase': phase,
        'epoch': epoch,
        'best_dice': best_dice,
        'completed': completed,
        'timestamp': datetime.now().isoformat()
    }

    # Write to temp file first, then rename for atomicity
    temp_path = state_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(state, f, indent=2)
    shutil.move(str(temp_path), str(state_path))

    logger.debug(f"Saved training state: strategy={strategy}, phase={phase}, epoch={epoch}")


def get_resume_checkpoint_path(base_dir: Path, config: dict, strategy: str, phase: int) -> Path:
    """Get the checkpoint path for resuming a specific phase"""
    strategy_config = config[f'strategy_{strategy.lower()}']
    phase_config = strategy_config[f'phase_{phase}']
    return base_dir / phase_config['checkpoint'].replace('.pth', '_resume.pth')


def save_resume_checkpoint(path: Path, model, optimizer, scheduler, scaler,
                           epoch: int, best_dice: float):
    """Save checkpoint with full state for resuming mid-phase"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_dice': best_dice
    }, path)


def load_resume_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    """Load checkpoint and restore full training state"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint.get('best_dice', 0)


class CurriculumSampler:
    """Implements curriculum learning by gradually including harder samples"""

    def __init__(self, metadata_df: pd.DataFrame, quality_column: str,
                 initial_percentile: float, final_percentile: float,
                 total_epochs: int, schedule_epochs: int):
        self.metadata = metadata_df.sort_values(quality_column, ascending=False)
        self.quality_column = quality_column
        self.initial_percentile = initial_percentile
        self.final_percentile = final_percentile
        self.total_epochs = total_epochs
        self.schedule_epochs = schedule_epochs

    def get_sample_weights(self, epoch: int) -> np.ndarray:
        """Get sampling weights for current epoch"""
        # Calculate current percentile threshold
        if epoch >= self.schedule_epochs:
            current_percentile = self.final_percentile
        else:
            progress = epoch / self.schedule_epochs
            current_percentile = self.initial_percentile - \
                                 progress * (self.initial_percentile - self.final_percentile)

        # Get quality threshold
        threshold = np.percentile(self.metadata[self.quality_column], current_percentile)

        # Create weights (1.0 for included, 0.0 for excluded)
        weights = (self.metadata[self.quality_column] >= threshold).astype(float)

        # Small weight for excluded to maintain some diversity
        weights[weights == 0] = 0.1

        return weights.values

    def get_case_ids(self, epoch: int) -> list:
        """Get case IDs for current epoch based on curriculum"""
        weights = self.get_sample_weights(epoch)
        # Return cases with weight >= 0.5 (i.e., included in curriculum)
        return self.metadata[weights >= 0.5]['case_id'].tolist()


class DifficultyWeightedSampler:
    """Weights sampling by difficulty (inverse of model performance)"""

    def __init__(self, metadata_df: pd.DataFrame, difficulty_multiplier: float = 5.0):
        self.metadata = metadata_df
        self.difficulty_multiplier = difficulty_multiplier
        self.case_difficulties = {}

    def update_difficulties(self, case_performances: dict):
        """Update difficulties based on model performance"""
        for case_id, dice in case_performances.items():
            # Lower dice = higher difficulty = higher sampling weight
            self.case_difficulties[case_id] = 1.0 - dice

    def get_sample_weights(self) -> np.ndarray:
        """Get sampling weights based on difficulty"""
        weights = []
        for case_id in self.metadata['case_id']:
            if case_id in self.case_difficulties:
                difficulty = self.case_difficulties[case_id]
                # Exponential weighting for hard cases
                weight = 1.0 + self.difficulty_multiplier * difficulty
            else:
                weight = 1.0  # Default weight for unseen cases
            weights.append(weight)

        weights = np.array(weights)
        return weights / weights.sum()  # Normalize


def create_loss_function(config: dict) -> nn.Module:
    """Create loss function from config"""
    loss_type = config.get('type', 'combo')

    if loss_type == 'small_lesion':
        # Use SmallLesionOptimizedLoss which is designed for tiny lesion detection
        return SmallLesionOptimizedLoss(
            tversky_alpha=0.3,
            tversky_beta=0.7,
            tversky_gamma=1.5,
            sensitivity_r=0.75
        )
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=config.get('alpha', 0.7),
            beta=config.get('beta', 0.3)
        )
    elif loss_type == 'combo':
        return ComboLoss(
            alpha=config.get('dice_weight', 0.5),
            beta=config.get('bce_weight', 0.5)
        )
    else:
        return SmallLesionOptimizedLoss()


def create_model(config: dict) -> nn.Module:
    """Create model from config"""
    return LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config.get('base_channels', 20),
        depth=config.get('depth', 3),
        use_attention=config.get('attention', False),
        dropout_p=config.get('dropout', 0.1)
    )


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Dataset returns (images, masks, case_ids) tuple
        images, masks, _ = batch
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            # Check if loss function expects 2 or 3 arguments
            try:
                loss = criterion(outputs, masks)
            except TypeError:
                loss = criterion(outputs, masks, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    case_performances = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Dataset returns (images, masks, case_ids) tuple
            images, masks, case_ids = batch
            images = images.to(device)
            masks = masks.to(device)

            with autocast():
                outputs = model(images)
                try:
                    loss = criterion(outputs, masks)
                except TypeError:
                    loss = criterion(outputs, masks, masks)

            # Compute Dice
            preds = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (preds * masks).sum(dim=(1, 2, 3, 4))
            union = preds.sum(dim=(1, 2, 3, 4)) + masks.sum(dim=(1, 2, 3, 4))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)

            # Store per-case performance
            for i, case_id in enumerate(case_ids):
                if case_id:
                    case_performances[case_id] = dice[i].item()

            total_loss += loss.item()
            total_dice += dice.mean().item()
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'case_performances': case_performances
    }


def run_curriculum_learning(config: dict, base_dir: Path, device: torch.device,
                            resume_state: dict = None):
    """Run Strategy A: Curriculum Learning with resume support and TensorBoard logging"""
    logger.info("=" * 60)
    logger.info("STRATEGY A: CURRICULUM LEARNING")
    logger.info("=" * 60)

    strategy_config = config['strategy_a']

    # Initialize TensorBoard writer
    log_dir = base_dir / 'logs' / 'tensorboard' / f'strategy_a_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs: {log_dir}")

    # Determine resume point
    start_phase = 1
    start_epoch = 1
    if resume_state and resume_state.get('strategy') == 'A' and not resume_state.get('completed'):
        start_phase = resume_state.get('phase', 1)
        start_epoch = resume_state.get('epoch', 0) + 1  # Start from next epoch
        logger.info(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")

    # Load metadata for curriculum
    metadata_path = base_dir / config['data']['metadata']
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        logger.warning("Metadata not found, running without curriculum sampling")
        metadata = None

    # Global step counter for TensorBoard
    global_step = 0

    # =========================================================================
    # Phase 1: High-quality bootstrap
    # =========================================================================
    if start_phase <= 1:
        logger.info("\n[Phase 1] High-Quality Bootstrap")
        phase1_config = strategy_config['phase_1']

        model = create_model(phase1_config['model'])
        model = model.to(device)

        criterion = create_loss_function(phase1_config['loss'])
        optimizer = optim.AdamW(model.parameters(), lr=phase1_config['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase1_config['epochs']
        )
        scaler = GradScaler()

        # Dataset
        train_dir = base_dir / config['data']['high_quality_train']
        train_dataset = BrainMetDataset(
            data_dir=str(train_dir),
            sequences=SUPERSET_SEQUENCES,
            target_size=TARGET_SIZE,
            transform=AugmentationPipeline(augmentation_probability=phase1_config['augmentation']['probability'])
        )
        train_loader = DataLoader(
            train_dataset, batch_size=phase1_config['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True
        )

        # Resume checkpoint if needed
        resume_ckpt_path = get_resume_checkpoint_path(base_dir, config, 'A', 1)
        best_dice = 0
        phase1_start_epoch = start_epoch if start_phase == 1 else 1

        if start_phase == 1 and start_epoch > 1 and resume_ckpt_path.exists():
            phase1_start_epoch, best_dice = load_resume_checkpoint(
                resume_ckpt_path, model, optimizer, scheduler, scaler, device
            )
            phase1_start_epoch += 1  # Start from next epoch
            logger.info(f"Resumed Phase 1 from epoch {phase1_start_epoch}, best_dice={best_dice:.4f}")

        for epoch in range(phase1_start_epoch, phase1_config['epochs'] + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
            val_results = validate(model, train_loader, criterion, device)

            scheduler.step()
            global_step += 1

            # TensorBoard logging
            writer.add_scalar('Phase1/train_loss', train_loss, epoch)
            writer.add_scalar('Phase1/val_dice', val_results['dice'], epoch)
            writer.add_scalar('Phase1/val_loss', val_results['loss'], epoch)
            writer.add_scalar('Phase1/learning_rate', scheduler.get_last_lr()[0], epoch)

            logger.info(f"Phase 1 Epoch {epoch}/{phase1_config['epochs']}: "
                       f"Loss={train_loss:.4f}, Dice={val_results['dice']:.4f}")

            # Save best model
            if val_results['dice'] > best_dice:
                best_dice = val_results['dice']
                checkpoint_path = base_dir / phase1_config['checkpoint']
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': best_dice
                }, checkpoint_path)
                logger.info(f"Saved best Phase 1 model (Dice: {best_dice:.4f})")

            # Save resume checkpoint and training state after each epoch
            save_resume_checkpoint(resume_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_dice)
            save_training_state(base_dir, 'A', 1, epoch, best_dice)

        logger.info(f"Phase 1 complete. Best Dice: {best_dice:.4f}")

    # =========================================================================
    # Phase 2: Full dataset with curriculum
    # =========================================================================
    if start_phase <= 2:
        logger.info("\n[Phase 2] Full Dataset with Curriculum Sampling")
        phase1_config = strategy_config['phase_1']
        phase2_config = strategy_config['phase_2']

        # Create model if starting from phase 2
        if start_phase > 1:
            model = create_model(phase1_config['model'])
            model = model.to(device)

        # Load phase 1 checkpoint if starting fresh for phase 2
        if start_phase == 2 and start_epoch == 1:
            checkpoint = torch.load(base_dir / phase1_config['checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded Phase 1 checkpoint for Phase 2")

        optimizer = optim.AdamW(model.parameters(), lr=phase2_config['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase2_config['epochs']
        )
        criterion = create_loss_function(phase2_config['loss'])
        scaler = GradScaler()

        # Full dataset
        full_train_dir = base_dir / config['data']['full_train']
        full_dataset = BrainMetDataset(
            data_dir=str(full_train_dir),
            sequences=SUPERSET_SEQUENCES,
            target_size=TARGET_SIZE,
            transform=AugmentationPipeline(augmentation_probability=phase2_config['augmentation']['probability'])
        )

        # Curriculum sampler
        if metadata is not None and phase2_config.get('curriculum', {}).get('enabled', False):
            curr_config = phase2_config['curriculum']
            curriculum = CurriculumSampler(
                metadata, curr_config['quality_score_column'],
                curr_config['initial_percentile'], curr_config['final_percentile'],
                phase2_config['epochs'], curr_config['schedule_epochs']
            )
        else:
            curriculum = None

        # Resume checkpoint if needed
        resume_ckpt_path = get_resume_checkpoint_path(base_dir, config, 'A', 2)
        best_dice = 0
        phase2_start_epoch = start_epoch if start_phase == 2 else 1

        if start_phase == 2 and start_epoch > 1 and resume_ckpt_path.exists():
            phase2_start_epoch, best_dice = load_resume_checkpoint(
                resume_ckpt_path, model, optimizer, scheduler, scaler, device
            )
            phase2_start_epoch += 1
            logger.info(f"Resumed Phase 2 from epoch {phase2_start_epoch}, best_dice={best_dice:.4f}")

        for epoch in range(phase2_start_epoch, phase2_config['epochs'] + 1):
            # Update curriculum weights
            if curriculum:
                weights = curriculum.get_sample_weights(epoch)
                sampler = WeightedRandomSampler(weights, len(weights))
                train_loader = DataLoader(
                    full_dataset, batch_size=phase2_config['batch_size'],
                    sampler=sampler, num_workers=4, pin_memory=True
                )
            else:
                train_loader = DataLoader(
                    full_dataset, batch_size=phase2_config['batch_size'],
                    shuffle=True, num_workers=4, pin_memory=True
                )

            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
            val_results = validate(model, train_loader, criterion, device)

            scheduler.step()
            global_step += 1

            # TensorBoard logging
            writer.add_scalar('Phase2/train_loss', train_loss, epoch)
            writer.add_scalar('Phase2/val_dice', val_results['dice'], epoch)
            writer.add_scalar('Phase2/val_loss', val_results['loss'], epoch)
            writer.add_scalar('Phase2/learning_rate', scheduler.get_last_lr()[0], epoch)

            logger.info(f"Phase 2 Epoch {epoch}/{phase2_config['epochs']}: "
                       f"Loss={train_loss:.4f}, Dice={val_results['dice']:.4f}")

            if val_results['dice'] > best_dice:
                best_dice = val_results['dice']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'dice': best_dice
                }, base_dir / phase2_config['checkpoint'])
                logger.info(f"Saved best Phase 2 model (Dice: {best_dice:.4f})")

            # Save resume checkpoint and training state
            save_resume_checkpoint(resume_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_dice)
            save_training_state(base_dir, 'A', 2, epoch, best_dice)

        logger.info(f"Phase 2 complete. Best Dice: {best_dice:.4f}")

    # =========================================================================
    # Phase 3: Hard sample focus
    # =========================================================================
    logger.info("\n[Phase 3] Hard Sample Focus")
    phase2_config = strategy_config['phase_2']
    phase3_config = strategy_config['phase_3']

    # Create model if starting from phase 3
    if start_phase > 2:
        phase1_config = strategy_config['phase_1']
        model = create_model(phase1_config['model'])
        model = model.to(device)

    # Load phase 2 checkpoint if starting fresh for phase 3
    if start_phase == 3 and start_epoch == 1:
        checkpoint = torch.load(base_dir / phase2_config['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded Phase 2 checkpoint for Phase 3")

    optimizer = optim.AdamW(model.parameters(), lr=phase3_config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase3_config['epochs']
    )
    criterion = create_loss_function(phase3_config['loss'])
    scaler = GradScaler()

    # Full dataset (reuse if already loaded)
    if start_phase > 2:
        full_train_dir = base_dir / config['data']['full_train']
        full_dataset = BrainMetDataset(
            data_dir=str(full_train_dir),
            sequences=SUPERSET_SEQUENCES,
            target_size=TARGET_SIZE,
            transform=AugmentationPipeline(augmentation_probability=phase3_config.get('augmentation', {}).get('probability', 0.5))
        )

    # Default train_loader for difficulty sampler initialization
    train_loader = DataLoader(
        full_dataset, batch_size=phase3_config['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True
    )

    # Difficulty sampler
    if metadata is not None:
        difficulty_sampler = DifficultyWeightedSampler(
            metadata, phase3_config['sampling']['difficulty_multiplier']
        )
        # Initialize with current model performance
        val_results = validate(model, train_loader, criterion, device)
        difficulty_sampler.update_difficulties(val_results['case_performances'])
    else:
        difficulty_sampler = None

    # Resume checkpoint if needed
    resume_ckpt_path = get_resume_checkpoint_path(base_dir, config, 'A', 3)
    best_dice = 0
    phase3_start_epoch = start_epoch if start_phase == 3 else 1

    if start_phase == 3 and start_epoch > 1 and resume_ckpt_path.exists():
        phase3_start_epoch, best_dice = load_resume_checkpoint(
            resume_ckpt_path, model, optimizer, scheduler, scaler, device
        )
        phase3_start_epoch += 1
        logger.info(f"Resumed Phase 3 from epoch {phase3_start_epoch}, best_dice={best_dice:.4f}")

    for epoch in range(phase3_start_epoch, phase3_config['epochs'] + 1):
        if difficulty_sampler:
            weights = difficulty_sampler.get_sample_weights()
            sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(
                full_dataset, batch_size=phase3_config['batch_size'],
                sampler=sampler, num_workers=4, pin_memory=True
            )

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_results = validate(model, train_loader, criterion, device)

        # Update difficulty weights periodically
        if difficulty_sampler and epoch % 5 == 0:
            difficulty_sampler.update_difficulties(val_results['case_performances'])

        scheduler.step()
        global_step += 1

        # TensorBoard logging
        writer.add_scalar('Phase3/train_loss', train_loss, epoch)
        writer.add_scalar('Phase3/val_dice', val_results['dice'], epoch)
        writer.add_scalar('Phase3/val_loss', val_results['loss'], epoch)
        writer.add_scalar('Phase3/learning_rate', scheduler.get_last_lr()[0], epoch)

        logger.info(f"Phase 3 Epoch {epoch}/{phase3_config['epochs']}: "
                   f"Loss={train_loss:.4f}, Dice={val_results['dice']:.4f}")

        if val_results['dice'] > best_dice:
            best_dice = val_results['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dice': best_dice
            }, base_dir / phase3_config['checkpoint'])
            logger.info(f"Saved best Phase 3 model (Dice: {best_dice:.4f})")

        # Save resume checkpoint and training state
        save_resume_checkpoint(resume_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_dice)
        save_training_state(base_dir, 'A', 3, epoch, best_dice)

    # Mark strategy A as complete
    save_training_state(base_dir, 'A', 3, phase3_config['epochs'], best_dice, completed=True)

    # Close TensorBoard writer
    writer.close()

    logger.info(f"\nCurriculum Learning Complete! Final Dice: {best_dice:.4f}")
    return best_dice


def run_pretraining_finetuning(config: dict, base_dir: Path, device: torch.device,
                               resume_state: dict = None):
    """Run Strategy B: Pretraining + Fine-tuning with resume support and TensorBoard logging"""
    logger.info("=" * 60)
    logger.info("STRATEGY B: PRETRAINING + FINE-TUNING")
    logger.info("=" * 60)

    strategy_config = config['strategy_b']

    # Initialize TensorBoard writer
    log_dir = base_dir / 'logs' / 'tensorboard' / f'strategy_b_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs: {log_dir}")

    # Determine resume point
    start_phase = 1
    start_epoch = 1
    if resume_state and resume_state.get('strategy') == 'B' and not resume_state.get('completed'):
        start_phase = resume_state.get('phase', 1)
        start_epoch = resume_state.get('epoch', 0) + 1
        logger.info(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")

    # =========================================================================
    # Phase 1: Self-supervised pretraining
    # =========================================================================
    phase1_config = strategy_config['phase_1']
    model = None

    if start_phase <= 1:
        logger.info("\n[Phase 1] Self-Supervised Pretraining on Yale Data")

        pretraining_dir = base_dir / config['data']['pretraining']

        if not pretraining_dir.exists() or len(list(pretraining_dir.iterdir())) == 0:
            logger.warning("Pretraining data not found, skipping pretraining phase")
            model = create_model(phase1_config['model'])
        else:
            model = create_model(phase1_config['model'])
            model = model.to(device)

            pretrain_dataset = BrainMetDataset(
                data_dir=str(pretraining_dir),
                sequences=SUPERSET_SEQUENCES,
                target_size=TARGET_SIZE,
                transform=AugmentationPipeline(augmentation_probability=phase1_config['augmentation']['probability'])
            )

            if len(pretrain_dataset) > 0:
                pretrain_loader = DataLoader(
                    pretrain_dataset, batch_size=phase1_config['batch_size'],
                    shuffle=True, num_workers=4, pin_memory=True
                )

                optimizer = optim.AdamW(model.parameters(), lr=phase1_config['learning_rate'])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=phase1_config['epochs']
                )
                scaler = GradScaler()
                reconstruction_criterion = nn.MSELoss()

                logger.info(f"Pretraining on {len(pretrain_dataset)} cases")

                # Resume checkpoint if needed
                resume_ckpt_path = get_resume_checkpoint_path(base_dir, config, 'B', 1)
                phase1_start_epoch = start_epoch if start_phase == 1 else 1
                best_loss = float('inf')

                if start_phase == 1 and start_epoch > 1 and resume_ckpt_path.exists():
                    phase1_start_epoch, _ = load_resume_checkpoint(
                        resume_ckpt_path, model, optimizer, scheduler, scaler, device
                    )
                    phase1_start_epoch += 1
                    logger.info(f"Resumed Phase 1 from epoch {phase1_start_epoch}")

                for epoch in range(phase1_start_epoch, phase1_config['epochs'] + 1):
                    model.train()
                    total_loss = 0

                    for batch in tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch}"):
                        images, _ = batch
                        images = images.to(device)

                        optimizer.zero_grad()

                        mask_ratio = phase1_config['pretraining'].get('mask_ratio', 0.4)
                        rand_mask = torch.rand_like(images) > mask_ratio
                        masked_images = images * rand_mask

                        with autocast():
                            outputs = model(masked_images)
                            loss = reconstruction_criterion(
                                outputs * (~rand_mask).float(),
                                images * (~rand_mask).float()
                            )

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        total_loss += loss.item()

                    scheduler.step()
                    avg_loss = total_loss / len(pretrain_loader)

                    # TensorBoard logging
                    writer.add_scalar('Phase1_Pretrain/loss', avg_loss, epoch)
                    writer.add_scalar('Phase1_Pretrain/learning_rate', scheduler.get_last_lr()[0], epoch)

                    logger.info(f"Pretrain Epoch {epoch}/{phase1_config['epochs']}: Loss={avg_loss:.4f}")

                    # Save best and resume checkpoints
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'loss': best_loss
                        }, base_dir / phase1_config['checkpoint'])

                    save_resume_checkpoint(resume_ckpt_path, model, optimizer, scheduler, scaler, epoch, 0)
                    save_training_state(base_dir, 'B', 1, epoch, 0)

                logger.info("Saved pretrained encoder")

    # =========================================================================
    # Phase 2: Supervised fine-tuning
    # =========================================================================
    if start_phase <= 2:
        logger.info("\n[Phase 2] Supervised Fine-tuning")
        phase2_config = strategy_config['phase_2']

        # Create model if needed
        if model is None:
            model = create_model(phase1_config['model'])

        # Load pretrained weights if starting phase 2 fresh
        if start_phase == 2 and start_epoch == 1:
            pretrained_path = base_dir / phase1_config['checkpoint']
            if pretrained_path.exists():
                checkpoint = torch.load(pretrained_path, map_location=device)
                if phase2_config.get('load_encoder_only', False):
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                                      if 'encoder' in k}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded pretrained weights")

        model = model.to(device)

        # Full training dataset
        full_train_dir = base_dir / config['data']['full_train']
        train_dataset = BrainMetDataset(
            data_dir=str(full_train_dir),
            sequences=SUPERSET_SEQUENCES,
            target_size=TARGET_SIZE,
            transform=AugmentationPipeline(augmentation_probability=phase2_config['augmentation']['probability'])
        )
        train_loader = DataLoader(
            train_dataset, batch_size=phase2_config['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True
        )

        criterion = create_loss_function(phase2_config['loss'])
        optimizer = optim.AdamW(model.parameters(), lr=phase2_config['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase2_config['epochs']
        )
        scaler = GradScaler()

        # Resume checkpoint if needed
        resume_ckpt_path = get_resume_checkpoint_path(base_dir, config, 'B', 2)
        best_dice = 0
        phase2_start_epoch = start_epoch if start_phase == 2 else 1

        if start_phase == 2 and start_epoch > 1 and resume_ckpt_path.exists():
            phase2_start_epoch, best_dice = load_resume_checkpoint(
                resume_ckpt_path, model, optimizer, scheduler, scaler, device
            )
            phase2_start_epoch += 1
            logger.info(f"Resumed Phase 2 from epoch {phase2_start_epoch}, best_dice={best_dice:.4f}")

        for epoch in range(phase2_start_epoch, phase2_config['epochs'] + 1):
            # Gradual unfreezing
            if phase2_config.get('unfreezing', {}).get('enabled', False):
                freeze_epochs = phase2_config['unfreezing'].get('freeze_encoder_epochs', 10)
                if epoch <= freeze_epochs:
                    for name, param in model.named_parameters():
                        if 'encoder' in name:
                            param.requires_grad = False
                else:
                    for param in model.parameters():
                        param.requires_grad = True

            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
            val_results = validate(model, train_loader, criterion, device)

            scheduler.step()

            # TensorBoard logging
            writer.add_scalar('Phase2_Finetune/train_loss', train_loss, epoch)
            writer.add_scalar('Phase2_Finetune/val_dice', val_results['dice'], epoch)
            writer.add_scalar('Phase2_Finetune/val_loss', val_results['loss'], epoch)
            writer.add_scalar('Phase2_Finetune/learning_rate', scheduler.get_last_lr()[0], epoch)

            logger.info(f"Finetune Epoch {epoch}/{phase2_config['epochs']}: "
                       f"Loss={train_loss:.4f}, Dice={val_results['dice']:.4f}")

            if val_results['dice'] > best_dice:
                best_dice = val_results['dice']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'dice': best_dice
                }, base_dir / phase2_config['checkpoint'])
                logger.info(f"Saved best Phase 2 model (Dice: {best_dice:.4f})")

            save_resume_checkpoint(resume_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_dice)
            save_training_state(base_dir, 'B', 2, epoch, best_dice)

        logger.info(f"Phase 2 complete. Best Dice: {best_dice:.4f}")

    # =========================================================================
    # Phase 3: Hard sample refinement
    # =========================================================================
    logger.info("\n[Phase 3] Hard Sample Refinement")
    phase2_config = strategy_config['phase_2']
    phase3_config = strategy_config['phase_3']

    # Create model if needed
    if model is None or start_phase > 2:
        model = create_model(phase1_config['model'])
        model = model.to(device)

    # Load phase 2 checkpoint if starting phase 3 fresh
    if start_phase == 3 and start_epoch == 1:
        checkpoint = torch.load(base_dir / phase2_config['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded Phase 2 checkpoint for Phase 3")

    # Ensure dataset is loaded
    if start_phase > 2:
        full_train_dir = base_dir / config['data']['full_train']
        train_dataset = BrainMetDataset(
            data_dir=str(full_train_dir),
            sequences=SUPERSET_SEQUENCES,
            target_size=TARGET_SIZE,
            transform=AugmentationPipeline(augmentation_probability=phase3_config.get('augmentation', {}).get('probability', 0.5))
        )
        train_loader = DataLoader(
            train_dataset, batch_size=phase3_config['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True
        )

    optimizer = optim.AdamW(model.parameters(), lr=phase3_config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase3_config['epochs']
    )
    criterion = create_loss_function(phase3_config['loss'])
    scaler = GradScaler()

    # Resume checkpoint if needed
    resume_ckpt_path = get_resume_checkpoint_path(base_dir, config, 'B', 3)
    best_dice = 0
    phase3_start_epoch = start_epoch if start_phase == 3 else 1

    if start_phase == 3 and start_epoch > 1 and resume_ckpt_path.exists():
        phase3_start_epoch, best_dice = load_resume_checkpoint(
            resume_ckpt_path, model, optimizer, scheduler, scaler, device
        )
        phase3_start_epoch += 1
        logger.info(f"Resumed Phase 3 from epoch {phase3_start_epoch}, best_dice={best_dice:.4f}")

    for epoch in range(phase3_start_epoch, phase3_config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_results = validate(model, train_loader, criterion, device)

        scheduler.step()

        # TensorBoard logging
        writer.add_scalar('Phase3_Refine/train_loss', train_loss, epoch)
        writer.add_scalar('Phase3_Refine/val_dice', val_results['dice'], epoch)
        writer.add_scalar('Phase3_Refine/val_loss', val_results['loss'], epoch)
        writer.add_scalar('Phase3_Refine/learning_rate', scheduler.get_last_lr()[0], epoch)

        logger.info(f"Refinement Epoch {epoch}/{phase3_config['epochs']}: "
                   f"Loss={train_loss:.4f}, Dice={val_results['dice']:.4f}")

        if val_results['dice'] > best_dice:
            best_dice = val_results['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dice': best_dice
            }, base_dir / phase3_config['checkpoint'])
            logger.info(f"Saved best Phase 3 model (Dice: {best_dice:.4f})")

        save_resume_checkpoint(resume_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_dice)
        save_training_state(base_dir, 'B', 3, epoch, best_dice)

    # Mark strategy B as complete
    save_training_state(base_dir, 'B', 3, phase3_config['epochs'], best_dice, completed=True)

    # Close TensorBoard writer
    writer.close()

    logger.info(f"\nPretraining+Finetuning Complete! Final Dice: {best_dice:.4f}")
    return best_dice


def main():
    parser = argparse.ArgumentParser(description='Train on Superset')
    parser.add_argument('--strategy', type=str, choices=['A', 'B', 'both'],
                        default='both', help='Training strategy to run')
    parser.add_argument('--config', type=str, default='configs/training_strategy.yaml',
                        help='Path to training config')
    parser.add_argument('--parallel', action='store_true',
                        help='Run strategies in parallel (requires 2 GPUs)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignoring any saved training state')

    args = parser.parse_args()

    # Load config
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Check for existing training state (auto-resume unless --no-resume)
    resume_state = None
    if not args.no_resume:
        resume_state = load_training_state(base_dir)
        if resume_state:
            if resume_state.get('completed'):
                logger.info("Previous training completed. Starting fresh.")
                logger.info(f"  (Previous: Strategy {resume_state['strategy']}, "
                           f"Best Dice: {resume_state.get('best_dice', 'N/A')})")
                resume_state = None
            else:
                logger.info("=" * 70)
                logger.info("RESUMING INTERRUPTED TRAINING")
                logger.info("=" * 70)
                logger.info(f"  Strategy: {resume_state['strategy']}")
                logger.info(f"  Phase: {resume_state['phase']}")
                logger.info(f"  Epoch: {resume_state['epoch']}")
                logger.info(f"  Best Dice: {resume_state.get('best_dice', 'N/A')}")
                logger.info(f"  Last saved: {resume_state.get('timestamp', 'Unknown')}")
                logger.info("=" * 70)

                # Override strategy to match resume state
                if resume_state['strategy'] == 'A':
                    args.strategy = 'A'
                elif resume_state['strategy'] == 'B':
                    args.strategy = 'B'

    results = {}

    if args.strategy in ['A', 'both']:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING STRATEGY A: CURRICULUM LEARNING")
        logger.info("=" * 70)
        # Pass resume_state only if it's for strategy A
        state_for_a = resume_state if resume_state and resume_state.get('strategy') == 'A' else None
        results['strategy_a'] = run_curriculum_learning(config, base_dir, device, state_for_a)

    if args.strategy in ['B', 'both']:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING STRATEGY B: PRETRAINING + FINE-TUNING")
        logger.info("=" * 70)
        # Pass resume_state only if it's for strategy B
        state_for_b = resume_state if resume_state and resume_state.get('strategy') == 'B' else None
        results['strategy_b'] = run_pretraining_finetuning(config, base_dir, device, state_for_b)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 70)
    for strategy, dice in results.items():
        logger.info(f"{strategy}: Final Dice = {dice:.4f}")

    # Save results
    results_path = base_dir / 'model' / 'training_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
