"""
Multi-Scale Self-Supervised Pretraining for Ensemble (Resumable Version)
=========================================================================
Features:
- Skips corrupted files automatically and caches the list
- Pausable/resumable training - stop anytime, resume from last completed epoch
- Saves full training state after each epoch

Usage:
    # Start fresh
    python scripts/pretrain_ensemble_resumable.py --phase pretrain

    # Resume from checkpoint (automatic)
    python scripts/pretrain_ensemble_resumable.py --phase pretrain --resume

    # Start fresh (ignore existing checkpoints)
    python scripts/pretrain_ensemble_resumable.py --phase pretrain --no-resume
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
import argparse
import logging
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from segmentation.unet import LightweightUNet3D
from segmentation.dataset import BrainMetDataset
from segmentation.advanced_losses import TverskyLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENSEMBLE CONFIGURATION - matches your current best models
# =============================================================================
ENSEMBLE_CONFIGS = {
    'exp1_8patch': {
        'patch_size': 8,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'optimal_threshold': 0.3,
        'pretrain_epochs': 30,
        'finetune_epochs': 100,
    },
    'exp3_12patch_maxfn': {
        'patch_size': 12,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'optimal_threshold': 0.25,
        'pretrain_epochs': 30,
        'finetune_epochs': 100,
    },
    'improved_24patch': {
        'patch_size': 24,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'optimal_threshold': 0.5,
        'pretrain_epochs': 30,
        'finetune_epochs': 100,
    },
    'improved_36patch': {
        'patch_size': 36,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'optimal_threshold': 0.5,
        'pretrain_epochs': 30,
        'finetune_epochs': 100,
    },
}

SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']


# =============================================================================
# CORRUPTED FILE TRACKER
# =============================================================================
class CorruptedFileTracker:
    """Tracks and persists list of corrupted files to skip."""

    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self.corrupted_files = set()
        self.load()

    def load(self):
        """Load corrupted files list from cache."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    self.corrupted_files = set(json.load(f))
                logger.info(f"Loaded {len(self.corrupted_files)} known corrupted files from cache")
            except:
                self.corrupted_files = set()

    def save(self):
        """Save corrupted files list to cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(list(self.corrupted_files), f, indent=2)

    def add(self, filepath):
        """Add a corrupted file to the list."""
        filepath_str = str(filepath)
        if filepath_str not in self.corrupted_files:
            self.corrupted_files.add(filepath_str)
            self.save()
            logger.info(f"Added to corrupted files list: {filepath}")

    def is_corrupted(self, filepath):
        """Check if a file is known to be corrupted."""
        return str(filepath) in self.corrupted_files


# =============================================================================
# DATASET FOR PRETRAINING (with corruption handling)
# =============================================================================
class RobustPretrainingDataset(Dataset):
    """
    Dataset for self-supervised pretraining on unlabeled data.
    - Handles missing modalities by using zeros
    - Tracks and skips corrupted files
    - Validates files on first load
    """

    def __init__(self, data_dir, sequences=SEQUENCES, patch_size=24,
                 target_size=(128, 128, 128), min_modalities=3,
                 corrupted_tracker=None, validate_on_init=True):
        self.data_dir = Path(data_dir)
        self.sequences = sequences
        self.patch_size = patch_size
        self.target_size = target_size
        self.min_modalities = min_modalities
        self.corrupted_tracker = corrupted_tracker or CorruptedFileTracker(
            self.data_dir.parent / 'corrupted_files.json'
        )

        # Find valid cases
        self.cases = []
        self.case_modalities = {}  # Cache which modalities each case has

        all_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        logger.info(f"Scanning {len(all_dirs)} directories for valid cases...")

        for case_dir in tqdm(all_dirs, desc="Validating cases"):
            available, valid_modalities = self._validate_case(case_dir, validate_on_init)
            if available >= min_modalities:
                self.cases.append(case_dir)
                self.case_modalities[str(case_dir)] = valid_modalities

        logger.info(f"Found {len(self.cases)} valid cases (>= {min_modalities} modalities)")
        logger.info(f"Skipped {len(self.corrupted_tracker.corrupted_files)} known corrupted files")

    def _validate_case(self, case_dir, full_validate=False):
        """Validate a case and return count of valid modalities."""
        valid_modalities = []

        for seq in self.sequences:
            filepath = case_dir / f"{seq}.nii.gz"

            if not filepath.exists():
                continue

            # Skip known corrupted files
            if self.corrupted_tracker.is_corrupted(filepath):
                continue

            # Full validation: try to actually load the file
            if full_validate:
                try:
                    # Quick validation - just try to load header
                    nib.load(str(filepath))
                    valid_modalities.append(seq)
                except Exception as e:
                    self.corrupted_tracker.add(filepath)
            else:
                valid_modalities.append(seq)

        return len(valid_modalities), valid_modalities

    def _load_nifti_safe(self, path):
        """Safely load NIfTI file, return None if corrupted."""
        try:
            data = nib.load(str(path)).get_fdata().astype(np.float32)
            return data
        except Exception as e:
            self.corrupted_tracker.add(path)
            return None

    def _normalize(self, img):
        """Z-score normalization."""
        mean = img.mean()
        std = img.std()
        if std > 0:
            img = (img - mean) / std
        return img

    def _resize_volume(self, img):
        """Resize to target size."""
        from scipy.ndimage import zoom
        if self.target_size is None:
            return img
        factors = [t / c for t, c in zip(self.target_size, img.shape)]
        return zoom(img, factors, order=1, mode='nearest')

    def _extract_patch(self, images):
        """Extract random patch from volume."""
        C, H, W, D = images.shape
        ph = pw = pd = self.patch_size

        # Pad if needed
        if H < ph or W < pw or D < pd:
            pad_h = max(0, ph - H)
            pad_w = max(0, pw - W)
            pad_d = max(0, pd - D)
            images = np.pad(images, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            C, H, W, D = images.shape

        h_start = np.random.randint(0, max(1, H - ph + 1))
        w_start = np.random.randint(0, max(1, W - pw + 1))
        d_start = np.random.randint(0, max(1, D - pd + 1))

        return images[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        valid_modalities = self.case_modalities.get(str(case_dir), self.sequences)

        images = []
        for seq in self.sequences:
            path = case_dir / f"{seq}.nii.gz"

            if seq in valid_modalities and path.exists() and not self.corrupted_tracker.is_corrupted(path):
                img = self._load_nifti_safe(path)
                if img is not None:
                    img = self._resize_volume(img)
                    img = self._normalize(img)
                    images.append(img)
                else:
                    images.append(np.zeros(self.target_size, dtype=np.float32))
            else:
                images.append(np.zeros(self.target_size, dtype=np.float32))

        images = np.stack(images, axis=0)
        images = self._extract_patch(images)

        return torch.from_numpy(images).float()


# =============================================================================
# MASKED RECONSTRUCTION TASK
# =============================================================================
class MaskedReconstruction(nn.Module):
    """Masked reconstruction pretraining task with patch-based masking."""

    def __init__(self, mask_ratio=0.4, patch_mask_size=4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_mask_size = patch_mask_size

    def create_mask(self, shape, device):
        """Create block-wise mask."""
        B, C, H, W, D = shape
        p = self.patch_mask_size

        nh, nw, nd = max(1, H // p), max(1, W // p), max(1, D // p)
        n_patches = nh * nw * nd
        n_mask = max(1, int(n_patches * self.mask_ratio))

        masks = []
        for _ in range(B):
            mask_idx = np.random.choice(n_patches, min(n_mask, n_patches), replace=False)
            mask = torch.ones(H, W, D, device=device)

            for idx in mask_idx:
                i = (idx // (nw * nd)) * p
                j = ((idx % (nw * nd)) // nd) * p
                k = (idx % nd) * p
                i_end = min(i + p, H)
                j_end = min(j + p, W)
                k_end = min(k + p, D)
                mask[i:i_end, j:j_end, k:k_end] = 0

            masks.append(mask)

        return torch.stack(masks).unsqueeze(1).expand(-1, C, -1, -1, -1)

    def forward(self, model, images, criterion):
        """Forward pass for masked reconstruction."""
        device = images.device
        mask = self.create_mask(images.shape, device)

        masked_images = images * mask
        reconstructed = model(masked_images)

        inv_mask = 1 - mask[:, :1, :, :, :]
        loss = criterion(reconstructed * inv_mask, images[:, :1, :, :, :] * inv_mask)

        return loss


# =============================================================================
# TRAINING STATE MANAGER (for pause/resume)
# =============================================================================
class TrainingStateManager:
    """Manages training state for pause/resume functionality."""

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def get_state_path(self, model_name, phase='pretrain'):
        return self.state_dir / f'{model_name}_{phase}_state.pth'

    def save_state(self, model_name, phase, epoch, model, optimizer, scheduler,
                   scaler, best_loss, history):
        """Save complete training state."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }

        state_path = self.get_state_path(model_name, phase)
        torch.save(state, state_path)
        logger.info(f"Saved training state: epoch {epoch}, loss {best_loss:.4f}")

    def load_state(self, model_name, phase, model, optimizer, scheduler, scaler, device):
        """Load training state if exists."""
        state_path = self.get_state_path(model_name, phase)

        if not state_path.exists():
            return None

        logger.info(f"Loading training state from {state_path}")
        state = torch.load(state_path, map_location=device)

        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])

        logger.info(f"Resumed from epoch {state['epoch']}, best_loss {state['best_loss']:.4f}")
        logger.info(f"Last saved: {state.get('timestamp', 'unknown')}")

        return state

    def get_completed_models(self, phase='pretrain'):
        """Get list of models that completed training."""
        completed = []
        for model_name in ENSEMBLE_CONFIGS.keys():
            done_marker = self.state_dir / f'{model_name}_{phase}_DONE'
            if done_marker.exists():
                completed.append(model_name)
        return completed

    def mark_complete(self, model_name, phase='pretrain'):
        """Mark a model as complete."""
        done_marker = self.state_dir / f'{model_name}_{phase}_DONE'
        done_marker.touch()
        logger.info(f"Marked {model_name} {phase} as COMPLETE")


# =============================================================================
# PRETRAINING FUNCTION (RESUMABLE)
# =============================================================================
def pretrain_model(model_name, config, pretraining_dir, output_dir, device,
                   state_manager, corrupted_tracker, epochs=None, batch_size=4,
                   lr=0.001, resume=True):
    """Pretrain a single model with pause/resume support."""

    patch_size = config['patch_size']
    epochs = epochs or config['pretrain_epochs']

    logger.info(f"\n{'='*60}")
    logger.info(f"PRETRAINING: {model_name} (patch_size={patch_size})")
    logger.info(f"{'='*60}")

    # Create model
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config['base_channels'],
        use_attention=config['use_attention'],
        use_residual=config['use_residual']
    ).to(device)

    # Dataset with corruption handling
    dataset = RobustPretrainingDataset(
        data_dir=pretraining_dir,
        sequences=SEQUENCES,
        patch_size=patch_size,
        target_size=(128, 128, 128),
        min_modalities=3,
        corrupted_tracker=corrupted_tracker,
        validate_on_init=False  # Skip full validation for speed (use cached)
    )

    if len(dataset) == 0:
        logger.error("No valid pretraining cases found!")
        return None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"Pretraining on {len(dataset)} cases, {len(loader)} batches/epoch")

    # Pretraining task
    masked_recon = MaskedReconstruction(mask_ratio=0.4, patch_mask_size=max(2, patch_size // 4))
    recon_criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    # Try to resume
    start_epoch = 1
    best_loss = float('inf')
    history = {'loss': [], 'lr': []}

    if resume:
        state = state_manager.load_state(model_name, 'pretrain', model, optimizer,
                                          scheduler, scaler, device)
        if state:
            start_epoch = state['epoch'] + 1
            best_loss = state['best_loss']
            history = state.get('history', history)

            if start_epoch > epochs:
                logger.info(f"Training already complete ({start_epoch-1}/{epochs} epochs)")
                state_manager.mark_complete(model_name, 'pretrain')
                return output_dir / f'{model_name}_pretrained.pth'

    # Best model checkpoint path
    checkpoint_path = output_dir / f'{model_name}_pretrained.pth'

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                loss = masked_recon(model, images, recon_criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        history['loss'].append(avg_loss)
        history['lr'].append(current_lr)

        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={current_lr:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"  Saved best model (loss={best_loss:.4f})")

        # Save training state after EVERY epoch (for resume)
        state_manager.save_state(
            model_name, 'pretrain', epoch, model, optimizer,
            scheduler, scaler, best_loss, history
        )

        logger.info(f"  [Checkpoint saved - safe to stop]")

        # Cleanup
        if epoch % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Mark as complete
    state_manager.mark_complete(model_name, 'pretrain')

    logger.info(f"Pretraining complete. Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")

    return checkpoint_path


# =============================================================================
# FINE-TUNING FUNCTION (RESUMABLE)
# =============================================================================
def finetune_model(model_name, config, pretrained_path, train_dir, output_dir,
                   device, state_manager, epochs=None, batch_size=2, lr=0.0003,
                   resume=True):
    """Fine-tune a pretrained model with pause/resume support."""

    patch_size = config['patch_size']
    epochs = epochs or config['finetune_epochs']

    logger.info(f"\n{'='*60}")
    logger.info(f"FINE-TUNING: {model_name} (patch_size={patch_size})")
    logger.info(f"{'='*60}")

    # Create model
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config['base_channels'],
        use_attention=config['use_attention'],
        use_residual=config['use_residual']
    ).to(device)

    # Dataset
    try:
        dataset = BrainMetDataset(
            data_dir=str(train_dir),
            sequences=['t1_pre', 't1_gd', 'flair', 't2'],
            patch_size=(patch_size, patch_size, patch_size),
            target_size=(128, 128, 128)
        )
    except:
        dataset = BrainMetDataset(
            data_dir=str(train_dir),
            sequences=['t1_pre', 't1_gd', 'flair', 'bravo'],
            patch_size=(patch_size, patch_size, patch_size),
            target_size=(128, 128, 128)
        )

    # Split
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    logger.info(f"Training: {n_train} cases, Validation: {n_val} cases")

    # Loss
    criterion = TverskyLoss(alpha=0.3, beta=0.7)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    # Try to resume or load pretrained
    start_epoch = 1
    best_dice = 0
    history = {'train_loss': [], 'val_dice': [], 'lr': []}

    if resume:
        state = state_manager.load_state(model_name, 'finetune', model, optimizer,
                                          scheduler, scaler, device)
        if state:
            start_epoch = state['epoch'] + 1
            best_dice = state['best_loss']  # Note: using best_loss field for dice
            history = state.get('history', history)

            if start_epoch > epochs:
                logger.info(f"Fine-tuning already complete")
                state_manager.mark_complete(model_name, 'finetune')
                return output_dir / f'{model_name}_pretrained_finetuned_best.pth', best_dice
        elif pretrained_path and Path(pretrained_path).exists():
            # Load pretrained weights
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint_path = output_dir / f'{model_name}_pretrained_finetuned_best.pth'

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        # Train
        model.train()
        train_loss = 0

        for images, masks, _ in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_dice = 0
        n_val_batches = 0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)

                with autocast('cuda'):
                    outputs = model(images)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum()
                dice = (2 * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)
                val_dice += dice.item()
                n_val_batches += 1

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_dice = val_dice / n_val_batches
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(avg_train_loss)
        history['val_dice'].append(avg_val_dice)
        history['lr'].append(current_lr)

        logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Dice={avg_val_dice:.4f}")

        # Save best
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': best_dice,
                'config': config
            }, checkpoint_path)
            logger.info(f"  Saved best model (Dice={best_dice:.4f})")

        # Save state after every epoch
        state_manager.save_state(
            model_name, 'finetune', epoch, model, optimizer,
            scheduler, scaler, best_dice, history
        )

        logger.info(f"  [Checkpoint saved - safe to stop]")

        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    state_manager.mark_complete(model_name, 'finetune')

    logger.info(f"Fine-tuning complete. Best Dice: {best_dice:.4f}")
    return checkpoint_path, best_dice


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-scale ensemble pretraining (resumable)')
    parser.add_argument('--phase', type=str, choices=['pretrain', 'finetune', 'both'],
                        default='both', help='Training phase')
    parser.add_argument('--models', type=str, nargs='+',
                        default=list(ENSEMBLE_CONFIGS.keys()),
                        help='Which models to train')
    parser.add_argument('--pretrain-epochs', type=int, default=None,
                        help='Override pretrain epochs')
    parser.add_argument('--finetune-epochs', type=int, default=None,
                        help='Override finetune epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignore existing checkpoints')

    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    # Paths
    project_dir = Path(__file__).parent.parent.parent
    superset_dir = project_dir.parent / 'Superset'
    pretraining_dir = superset_dir / 'pretraining'
    train_dir = superset_dir / 'full' / 'train'
    output_dir = project_dir / 'model'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Resume mode: {args.resume}")

    # Initialize managers
    state_manager = TrainingStateManager(output_dir / 'training_states')
    corrupted_tracker = CorruptedFileTracker(superset_dir / 'corrupted_files.json')

    # Show completed models
    if args.resume:
        completed_pretrain = state_manager.get_completed_models('pretrain')
        completed_finetune = state_manager.get_completed_models('finetune')
        if completed_pretrain:
            logger.info(f"Already completed pretraining: {completed_pretrain}")
        if completed_finetune:
            logger.info(f"Already completed fine-tuning: {completed_finetune}")

    # Track results
    results = {'pretrain': {}, 'finetune': {}}

    # Process each model
    for model_name in args.models:
        if model_name not in ENSEMBLE_CONFIGS:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        config = ENSEMBLE_CONFIGS[model_name]
        pretrained_path = None

        # Phase 1: Pretrain
        if args.phase in ['pretrain', 'both']:
            # Skip if already complete
            if args.resume and model_name in state_manager.get_completed_models('pretrain'):
                logger.info(f"Skipping {model_name} pretrain (already complete)")
                pretrained_path = output_dir / f'{model_name}_pretrained.pth'
            else:
                pretrained_path = pretrain_model(
                    model_name=model_name,
                    config=config,
                    pretraining_dir=pretraining_dir,
                    output_dir=output_dir,
                    device=device,
                    state_manager=state_manager,
                    corrupted_tracker=corrupted_tracker,
                    epochs=args.pretrain_epochs,
                    batch_size=args.batch_size,
                    resume=args.resume
                )
            results['pretrain'][model_name] = str(pretrained_path) if pretrained_path else None

        # Phase 2: Fine-tune
        if args.phase in ['finetune', 'both']:
            if args.resume and model_name in state_manager.get_completed_models('finetune'):
                logger.info(f"Skipping {model_name} finetune (already complete)")
            else:
                if pretrained_path is None:
                    pretrained_path = output_dir / f'{model_name}_pretrained.pth'

                finetuned_path, best_dice = finetune_model(
                    model_name=model_name,
                    config=config,
                    pretrained_path=pretrained_path,
                    train_dir=train_dir,
                    output_dir=output_dir,
                    device=device,
                    state_manager=state_manager,
                    epochs=args.finetune_epochs,
                    batch_size=max(1, args.batch_size // 2),
                    resume=args.resume
                )
                results['finetune'][model_name] = {
                    'path': str(finetuned_path),
                    'best_dice': best_dice
                }

        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    summary_path = output_dir / 'pretrain_ensemble_results.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'phase': args.phase,
            'models': args.models,
            'results': results
        }, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("SESSION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {summary_path}")
    logger.info("You can resume anytime with: python scripts/pretrain_ensemble_resumable.py --resume")


if __name__ == '__main__':
    main()
