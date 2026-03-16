"""
Multi-Scale Self-Supervised Pretraining for Ensemble
=====================================================
Pretrains each model in the ensemble architecture separately on Yale data,
then fine-tunes on labeled data.

Ensemble models (from current best):
- 8x8x8 patches (exp1_8patch) - ultra-small lesions
- 12x12x12 patches (exp3_12patch_maxfn) - tiny lesions
- 24x24x24 patches (improved_24patch) - balanced
- 36x36x36 patches (improved_36patch) - large lesions + context

Usage:
    python scripts/pretrain_ensemble.py --phase pretrain
    python scripts/pretrain_ensemble.py --phase finetune
    python scripts/pretrain_ensemble.py --phase both
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from segmentation.unet import LightweightUNet3D
from segmentation.dataset import BrainMetDataset
from segmentation.advanced_losses import SmallLesionOptimizedLoss, TverskyLoss

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

# Sequences to use
SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']


# =============================================================================
# DATASET FOR PRETRAINING (handles missing modalities gracefully)
# =============================================================================
class PretrainingDataset(Dataset):
    """
    Dataset for self-supervised pretraining on unlabeled data.
    Handles missing modalities by skipping incomplete cases.
    """

    def __init__(self, data_dir, sequences=SEQUENCES, patch_size=24,
                 target_size=(128, 128, 128), min_modalities=3):
        self.data_dir = Path(data_dir)
        self.sequences = sequences
        self.patch_size = patch_size
        self.target_size = target_size
        self.min_modalities = min_modalities

        # Find valid cases (with enough modalities)
        self.cases = []
        all_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        logger.info(f"Scanning {len(all_dirs)} directories for valid cases...")
        for case_dir in tqdm(all_dirs, desc="Validating cases"):
            available = self._count_modalities(case_dir)
            if available >= min_modalities:
                self.cases.append(case_dir)

        logger.info(f"Found {len(self.cases)} valid cases (>= {min_modalities} modalities)")

    def _count_modalities(self, case_dir):
        """Count available modalities for a case."""
        count = 0
        for seq in self.sequences:
            if (case_dir / f"{seq}.nii.gz").exists():
                count += 1
        return count

    def _load_nifti(self, path):
        """Load NIfTI file."""
        try:
            return nib.load(str(path)).get_fdata().astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    def _normalize(self, img):
        """Z-score normalization."""
        mean = img.mean()
        std = img.std()
        if std > 0:
            img = (img - mean) / std
        return img

    def _resize_volume(self, img):
        """Resize to target size using scipy zoom."""
        from scipy.ndimage import zoom
        if self.target_size is None:
            return img
        factors = [t / c for t, c in zip(self.target_size, img.shape)]
        return zoom(img, factors, order=1, mode='nearest')

    def _extract_patch(self, images):
        """Extract random patch from volume."""
        C, H, W, D = images.shape
        ph = pw = pd = self.patch_size

        # Ensure volume is large enough
        if H < ph or W < pw or D < pd:
            # Pad if needed
            pad_h = max(0, ph - H)
            pad_w = max(0, pw - W)
            pad_d = max(0, pd - D)
            images = np.pad(images, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            C, H, W, D = images.shape

        # Random start coordinates
        h_start = np.random.randint(0, H - ph + 1)
        w_start = np.random.randint(0, W - pw + 1)
        d_start = np.random.randint(0, D - pd + 1)

        return images[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]

        # Load available modalities
        images = []
        for seq in self.sequences:
            path = case_dir / f"{seq}.nii.gz"
            if path.exists():
                img = self._load_nifti(path)
                if img is not None:
                    img = self._resize_volume(img)
                    img = self._normalize(img)
                    images.append(img)
                else:
                    # Use zeros for failed loads
                    images.append(np.zeros(self.target_size, dtype=np.float32))
            else:
                # Use zeros for missing modalities
                images.append(np.zeros(self.target_size, dtype=np.float32))

        # Stack to (C, H, W, D)
        images = np.stack(images, axis=0)

        # Extract patch
        images = self._extract_patch(images)

        return torch.from_numpy(images).float()


# =============================================================================
# PRETRAINING TASKS
# =============================================================================
class MaskedReconstruction(nn.Module):
    """
    Masked reconstruction pretraining task.
    Masks random patches and trains model to reconstruct them.
    """

    def __init__(self, mask_ratio=0.4, patch_mask_size=4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_mask_size = patch_mask_size  # Mask in blocks, not random voxels

    def create_mask(self, shape, device):
        """Create block-wise mask."""
        B, C, H, W, D = shape
        p = self.patch_mask_size

        # Number of patches in each dimension
        nh, nw, nd = H // p, W // p, D // p

        # Create patch-level mask
        n_patches = nh * nw * nd
        n_mask = int(n_patches * self.mask_ratio)

        masks = []
        for _ in range(B):
            # Random patch indices to mask
            mask_idx = np.random.choice(n_patches, n_mask, replace=False)

            # Create full-resolution mask
            mask = torch.ones(H, W, D, device=device)
            for idx in mask_idx:
                i = (idx // (nw * nd)) * p
                j = ((idx % (nw * nd)) // nd) * p
                k = (idx % nd) * p
                mask[i:i+p, j:j+p, k:k+p] = 0

            masks.append(mask)

        # Stack and expand for channels
        return torch.stack(masks).unsqueeze(1).expand(-1, C, -1, -1, -1)

    def forward(self, model, images, criterion):
        """
        Forward pass for masked reconstruction.
        Returns loss for reconstructing masked regions.
        """
        device = images.device
        mask = self.create_mask(images.shape, device)

        # Mask input
        masked_images = images * mask

        # Reconstruct
        reconstructed = model(masked_images)

        # Loss only on masked regions
        inv_mask = 1 - mask[:, :1, :, :, :]  # Single channel mask
        loss = criterion(reconstructed * inv_mask, images[:, :1, :, :, :] * inv_mask)

        return loss


class ContrastiveLearning(nn.Module):
    """
    Simple contrastive learning - different augmentations of same image should be similar.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def augment(self, images):
        """Apply random augmentations."""
        # Random flip
        if np.random.rand() > 0.5:
            images = torch.flip(images, dims=[2])
        if np.random.rand() > 0.5:
            images = torch.flip(images, dims=[3])
        if np.random.rand() > 0.5:
            images = torch.flip(images, dims=[4])

        # Random intensity shift
        shift = (torch.rand(1, device=images.device) - 0.5) * 0.2
        images = images + shift

        return images

    def forward(self, model, images):
        """
        Compute contrastive loss between two augmented views.
        """
        # Create two augmented views
        view1 = self.augment(images.clone())
        view2 = self.augment(images.clone())

        # Get encoder features (use model output and global pool)
        with autocast():
            feat1 = model(view1)
            feat2 = model(view2)

        # Global average pooling to get feature vectors
        feat1 = feat1.mean(dim=[2, 3, 4])  # (B, 1)
        feat2 = feat2.mean(dim=[2, 3, 4])  # (B, 1)

        # Normalize
        feat1 = nn.functional.normalize(feat1, dim=1)
        feat2 = nn.functional.normalize(feat2, dim=1)

        # Similarity matrix
        sim = torch.mm(feat1, feat2.t()) / self.temperature

        # Contrastive loss (diagonal should be high)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = nn.functional.cross_entropy(sim, labels)

        return loss


# =============================================================================
# PRETRAINING FUNCTION
# =============================================================================
def pretrain_model(model_name, config, pretraining_dir, output_dir, device,
                   epochs=None, batch_size=4, lr=0.001):
    """Pretrain a single model on Yale data."""

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

    # Dataset
    dataset = PretrainingDataset(
        data_dir=pretraining_dir,
        sequences=SEQUENCES,
        patch_size=patch_size,
        target_size=(128, 128, 128),
        min_modalities=3  # Require at least 3 of 4 modalities
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
    scaler = GradScaler()

    # Training loop
    best_loss = float('inf')
    checkpoint_path = output_dir / f'{model_name}_pretrained.pth'

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)

            optimizer.zero_grad()

            with autocast():
                loss = masked_recon(model, images, recon_criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / len(loader)

        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"  Saved best checkpoint (loss={best_loss:.4f})")

        # Periodic cleanup
        if epoch % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"Pretraining complete. Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")

    return checkpoint_path


# =============================================================================
# FINE-TUNING FUNCTION
# =============================================================================
def finetune_model(model_name, config, pretrained_path, train_dir, output_dir,
                   device, epochs=None, batch_size=2, lr=0.0003):
    """Fine-tune a pretrained model on labeled data."""

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

    # Load pretrained weights if available
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("No pretrained weights found, training from scratch")

    # Dataset - use Superset full training data
    # Try t2 first (Superset naming), fall back to bravo (BrainMetShare naming)
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

    # Split into train/val
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    logger.info(f"Training: {n_train} cases, Validation: {n_val} cases")

    # Loss - Tversky for high sensitivity
    criterion = TverskyLoss(alpha=0.3, beta=0.7)

    # Optimizer with lower LR for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    # Training loop
    best_dice = 0
    checkpoint_path = output_dir / f'{model_name}_pretrained_finetuned_best.pth'

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0

        for images, masks, _ in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            with autocast():
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

                with autocast():
                    outputs = model(images)

                # Compute Dice
                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum()
                dice = (2 * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)
                val_dice += dice.item()
                n_val_batches += 1

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_dice = val_dice / n_val_batches

        logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Dice={avg_val_dice:.4f}")

        # Save best
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_dice,
                'config': config
            }, checkpoint_path)
            logger.info(f"  Saved best checkpoint (Dice={best_dice:.4f})")

        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"Fine-tuning complete. Best Dice: {best_dice:.4f}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")

    return checkpoint_path, best_dice


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-scale ensemble pretraining')
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

    args = parser.parse_args()

    # Paths
    project_dir = Path(__file__).parent.parent.parent
    superset_dir = project_dir.parent / 'Superset'
    pretraining_dir = superset_dir / 'pretraining'
    train_dir = superset_dir / 'full' / 'train'
    output_dir = project_dir / 'model'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Verify paths
    if args.phase in ['pretrain', 'both']:
        if not pretraining_dir.exists():
            logger.error(f"Pretraining directory not found: {pretraining_dir}")
            return
        logger.info(f"Pretraining data: {pretraining_dir}")

    if args.phase in ['finetune', 'both']:
        if not train_dir.exists():
            logger.error(f"Training directory not found: {train_dir}")
            return
        logger.info(f"Fine-tuning data: {train_dir}")

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
            pretrained_path = pretrain_model(
                model_name=model_name,
                config=config,
                pretraining_dir=pretraining_dir,
                output_dir=output_dir,
                device=device,
                epochs=args.pretrain_epochs,
                batch_size=args.batch_size
            )
            results['pretrain'][model_name] = str(pretrained_path) if pretrained_path else None

        # Phase 2: Fine-tune
        if args.phase in ['finetune', 'both']:
            # If only fine-tuning, look for existing pretrained checkpoint
            if pretrained_path is None:
                pretrained_path = output_dir / f'{model_name}_pretrained.pth'

            finetuned_path, best_dice = finetune_model(
                model_name=model_name,
                config=config,
                pretrained_path=pretrained_path,
                train_dir=train_dir,
                output_dir=output_dir,
                device=device,
                epochs=args.finetune_epochs,
                batch_size=max(1, args.batch_size // 2)  # Smaller batch for fine-tuning
            )
            results['finetune'][model_name] = {
                'path': str(finetuned_path),
                'best_dice': best_dice
            }

        # Cleanup between models
        gc.collect()
        torch.cuda.empty_cache()

    # Save results summary
    summary_path = output_dir / 'pretrain_ensemble_results.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'phase': args.phase,
            'models': args.models,
            'results': results
        }, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {summary_path}")

    if results['finetune']:
        logger.info("\nFine-tuned model performance:")
        for model_name, info in results['finetune'].items():
            logger.info(f"  {model_name}: Dice={info['best_dice']:.4f}")


if __name__ == '__main__':
    main()
