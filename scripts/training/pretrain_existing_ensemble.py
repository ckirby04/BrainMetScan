"""
Pretrain Models Matching Existing Ensemble Architecture
=======================================================
Two modes:
1. FROM_SCRATCH: Pretrain new models with same architecture, then fine-tune
2. CONTINUE: Continue training existing models with self-supervised task (risky)

Your ensemble uses: base_channels=16, attention=True, residual=True

Usage:
    # Pretrain from scratch (recommended)
    python scripts/pretrain_existing_ensemble.py --mode scratch

    # Continue training existing models (experimental)
    python scripts/pretrain_existing_ensemble.py --mode continue

    # Single model only
    python scripts/pretrain_existing_ensemble.py --models exp1_8patch

    # Resume if interrupted
    python scripts/pretrain_existing_ensemble.py --resume
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
from torch.amp import autocast, GradScaler
import numpy as np
import nibabel as nib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from segmentation.unet import LightweightUNet3D
from segmentation.dataset import BrainMetDataset
from segmentation.advanced_losses import TverskyLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# YOUR EXACT ENSEMBLE CONFIGURATION
# =============================================================================
ENSEMBLE_MODELS = {
    'exp1_8patch': {
        'patch_size': 8,
        'original_checkpoint': 'exp1_8patch_best.pth',
        'epochs_trained': 150,
    },
    'exp3_12patch_maxfn': {
        'patch_size': 12,
        'original_checkpoint': 'exp3_12patch_maxfn_best.pth',
        'epochs_trained': 90,
    },
    'improved_24patch': {
        'patch_size': 24,
        'original_checkpoint': 'improved_24patch_best.pth',
        'epochs_trained': 230,
    },
    'improved_36patch': {
        'patch_size': 36,
        'original_checkpoint': 'improved_36patch_best.pth',
        'epochs_trained': 244,
    },
}

# Matching your existing architecture exactly
MODEL_ARCHITECTURE = {
    'base_channels': 16,
    'use_attention': True,
    'use_residual': True,
}

SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']


# =============================================================================
# ROBUST PRETRAINING DATASET
# =============================================================================
class PretrainingDataset(Dataset):
    """Dataset for self-supervised pretraining with corruption handling."""

    def __init__(self, data_dir, patch_size=24, target_size=(128, 128, 128),
                 corrupted_cache=None):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.target_size = target_size

        # Load or create corrupted files cache
        self.corrupted_cache = corrupted_cache or self.data_dir.parent / 'corrupted_files.json'
        self.corrupted_files = self._load_corrupted_cache()

        # Find valid cases
        self.cases = []
        all_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        for case_dir in tqdm(all_dirs, desc="Validating cases"):
            valid_count = sum(1 for seq in SEQUENCES
                           if (case_dir / f"{seq}.nii.gz").exists()
                           and str(case_dir / f"{seq}.nii.gz") not in self.corrupted_files)
            if valid_count >= 3:
                self.cases.append(case_dir)

        logger.info(f"Found {len(self.cases)} valid pretraining cases")

    def _load_corrupted_cache(self):
        if Path(self.corrupted_cache).exists():
            with open(self.corrupted_cache) as f:
                return set(json.load(f))
        return set()

    def _save_corrupted(self, path):
        self.corrupted_files.add(str(path))
        with open(self.corrupted_cache, 'w') as f:
            json.dump(list(self.corrupted_files), f)

    def _load_nifti(self, path):
        try:
            return nib.load(str(path)).get_fdata().astype(np.float32)
        except:
            self._save_corrupted(path)
            return None

    def _normalize(self, img):
        mean, std = img.mean(), img.std()
        return (img - mean) / (std + 1e-8)

    def _resize(self, img):
        from scipy.ndimage import zoom
        factors = [t / c for t, c in zip(self.target_size, img.shape)]
        return zoom(img, factors, order=1)

    def _extract_patch(self, images):
        C, H, W, D = images.shape
        p = self.patch_size

        if H < p or W < p or D < p:
            pad = [(0, 0)] + [(0, max(0, p - s)) for s in [H, W, D]]
            images = np.pad(images, pad, mode='constant')
            C, H, W, D = images.shape

        h = np.random.randint(0, H - p + 1)
        w = np.random.randint(0, W - p + 1)
        d = np.random.randint(0, D - p + 1)

        return images[:, h:h+p, w:w+p, d:d+p]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]

        images = []
        for seq in SEQUENCES:
            path = case_dir / f"{seq}.nii.gz"
            if path.exists() and str(path) not in self.corrupted_files:
                img = self._load_nifti(path)
                if img is not None:
                    img = self._resize(img)
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
# MASKED RECONSTRUCTION
# =============================================================================
class MaskedReconstruction(nn.Module):
    def __init__(self, mask_ratio=0.4, block_size=4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.block_size = block_size

    def forward(self, model, images, criterion):
        B, C, H, W, D = images.shape
        device = images.device
        p = self.block_size

        # Create block mask
        nh, nw, nd = H // p, W // p, D // p
        n_blocks = nh * nw * nd
        n_mask = int(n_blocks * self.mask_ratio)

        masks = []
        for _ in range(B):
            mask = torch.ones(H, W, D, device=device)
            indices = np.random.choice(n_blocks, n_mask, replace=False)
            for idx in indices:
                i = (idx // (nw * nd)) * p
                j = ((idx % (nw * nd)) // nd) * p
                k = (idx % nd) * p
                mask[i:i+p, j:j+p, k:k+p] = 0
            masks.append(mask)

        mask = torch.stack(masks).unsqueeze(1).expand(-1, C, -1, -1, -1)

        masked_input = images * mask
        output = model(masked_input)

        inv_mask = 1 - mask[:, :1]
        loss = criterion(output * inv_mask, images[:, :1] * inv_mask)

        return loss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def create_model(load_from=None, device='cuda'):
    """Create model with exact ensemble architecture."""
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=MODEL_ARCHITECTURE['base_channels'],
        use_attention=MODEL_ARCHITECTURE['use_attention'],
        use_residual=MODEL_ARCHITECTURE['use_residual']
    ).to(device)

    if load_from and Path(load_from).exists():
        ckpt = torch.load(load_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded weights from {load_from}")

    return model


def pretrain(model_name, config, pretraining_dir, output_dir, device,
             epochs=30, batch_size=4, lr=0.001, resume=False, mode='scratch',
             model_dir=None):
    """Pretrain a model."""

    patch_size = config['patch_size']

    logger.info(f"\n{'='*60}")
    logger.info(f"PRETRAINING: {model_name} (patch={patch_size}, mode={mode})")
    logger.info(f"Architecture: base_channels={MODEL_ARCHITECTURE['base_channels']}")
    logger.info(f"{'='*60}")

    # Create or load model
    if mode == 'continue' and model_dir:
        original_path = model_dir / config['original_checkpoint']
        model = create_model(load_from=original_path, device=device)
        logger.info(f"Continuing from existing model: {original_path}")
    else:
        model = create_model(device=device)
        logger.info("Training from scratch")

    # Dataset
    dataset = PretrainingDataset(pretraining_dir, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)

    # Training setup
    masked_recon = MaskedReconstruction(mask_ratio=0.4, block_size=max(2, patch_size // 4))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    # Checkpoint paths
    state_path = output_dir / f'{model_name}_pretrain_state.pth'
    best_path = output_dir / f'{model_name}_pretrained_v2.pth'

    # Resume
    start_epoch = 1
    best_loss = float('inf')
    history = []

    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        start_epoch = state['epoch'] + 1
        best_loss = state['best_loss']
        history = state.get('history', [])
        logger.info(f"Resumed from epoch {start_epoch-1}")

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for images in pbar:
            images = images.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                loss = masked_recon(model, images, criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)

        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'config': config,
                'architecture': MODEL_ARCHITECTURE,
                'mode': mode
            }, best_path)
            logger.info(f"  Saved best (loss={best_loss:.4f})")

        # Save state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'history': history
        }, state_path)

        if epoch % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"Pretraining complete. Best loss: {best_loss:.4f}")
    return best_path


def finetune(model_name, config, pretrained_path, train_dir, output_dir,
             original_path, device, epochs=50, batch_size=2, lr=0.0003, resume=False):
    """Fine-tune pretrained model and compare to original."""

    patch_size = config['patch_size']

    logger.info(f"\n{'='*60}")
    logger.info(f"FINE-TUNING: {model_name}")
    logger.info(f"{'='*60}")

    # Load pretrained model
    model = create_model(load_from=pretrained_path, device=device)

    # Dataset
    try:
        dataset = BrainMetDataset(
            data_dir=str(train_dir),
            sequences=['t1_pre', 't1_gd', 'flair', 't2'],
            patch_size=(patch_size,) * 3,
            target_size=(128, 128, 128)
        )
    except:
        dataset = BrainMetDataset(
            data_dir=str(train_dir),
            sequences=['t1_pre', 't1_gd', 'flair', 'bravo'],
            patch_size=(patch_size,) * 3,
            target_size=(128, 128, 128)
        )

    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    # Evaluate original for comparison
    original_dice = None
    if original_path.exists():
        orig_model = create_model(load_from=original_path, device=device)
        orig_model.eval()
        total_dice = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = orig_model(images)
                pred = (torch.sigmoid(outputs) > 0.5).float()
                dice = (2 * (pred * masks).sum() + 1e-6) / (pred.sum() + masks.sum() + 1e-6)
                total_dice += dice.item()
        original_dice = total_dice / len(val_loader)
        logger.info(f"Original model Dice: {original_dice:.4f}")
        del orig_model
        torch.cuda.empty_cache()

    # Training setup
    criterion = TverskyLoss(alpha=0.3, beta=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    # Paths
    state_path = output_dir / f'{model_name}_finetune_state.pth'
    best_path = output_dir / f'{model_name}_pretrained_finetuned_v2.pth'

    # Resume
    start_epoch = 1
    best_dice = 0

    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state['epoch'] + 1
        best_dice = state['best_dice']
        logger.info(f"Resumed from epoch {start_epoch-1}")

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0

        for images, masks, _ in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                pred = (torch.sigmoid(outputs) > 0.5).float()
                dice = (2 * (pred * masks).sum() + 1e-6) / (pred.sum() + masks.sum() + 1e-6)
                val_dice += dice.item()

        val_dice /= len(val_loader)
        scheduler.step()

        diff = f" (vs orig: {val_dice - original_dice:+.4f})" if original_dice else ""
        logger.info(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, Dice={val_dice:.4f}{diff}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': best_dice,
                'original_dice': original_dice,
                'improvement': best_dice - original_dice if original_dice else None
            }, best_path)
            logger.info(f"  *** New best: {best_dice:.4f} ***")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice
        }, state_path)

    # Final summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    if original_dice:
        print(f"Original model:              {original_dice:.4f}")
    print(f"Pretrained + Fine-tuned:     {best_dice:.4f}")
    if original_dice:
        diff = best_dice - original_dice
        symbol = "✓" if diff > 0 else "✗"
        print(f"Improvement:                 {diff:+.4f} {symbol}")
    print(f"{'='*60}\n")

    return best_path, best_dice, original_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['scratch', 'continue'], default='scratch',
                       help='scratch=new model, continue=from existing weights')
    parser.add_argument('--models', nargs='+', default=list(ENSEMBLE_MODELS.keys()),
                       help='Which models to train')
    parser.add_argument('--phase', choices=['pretrain', 'finetune', 'both'], default='both')
    parser.add_argument('--pretrain-epochs', type=int, default=30)
    parser.add_argument('--finetune-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    # Paths
    project_dir = Path(__file__).parent.parent.parent
    model_dir = project_dir / 'model'
    superset_dir = project_dir.parent / 'Superset'
    pretraining_dir = superset_dir / 'pretraining'
    train_dir = superset_dir / 'full' / 'train'
    output_dir = model_dir

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Models: {args.models}")

    results = {}

    for model_name in args.models:
        if model_name not in ENSEMBLE_MODELS:
            logger.warning(f"Unknown model: {model_name}")
            continue

        config = ENSEMBLE_MODELS[model_name]

        # Pretrain
        pretrained_path = None
        if args.phase in ['pretrain', 'both']:
            pretrained_path = pretrain(
                model_name, config, pretraining_dir, output_dir, device,
                epochs=args.pretrain_epochs, batch_size=args.batch_size,
                resume=args.resume, mode=args.mode, model_dir=model_dir
            )

        # Fine-tune
        if args.phase in ['finetune', 'both']:
            if pretrained_path is None:
                pretrained_path = output_dir / f'{model_name}_pretrained_v2.pth'

            original_path = model_dir / config['original_checkpoint']

            best_path, best_dice, orig_dice = finetune(
                model_name, config, pretrained_path, train_dir, output_dir,
                original_path, device, epochs=args.finetune_epochs,
                batch_size=max(1, args.batch_size // 2), resume=args.resume
            )

            results[model_name] = {
                'original_dice': orig_dice,
                'finetuned_dice': best_dice,
                'improvement': best_dice - orig_dice if orig_dice else None
            }

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY - ALL MODELS")
        print("=" * 70)
        print(f"{'Model':<25} {'Original':>12} {'Fine-tuned':>12} {'Δ':>10}")
        print("-" * 70)

        for name, r in results.items():
            orig = f"{r['original_dice']:.4f}" if r['original_dice'] else "N/A"
            ft = f"{r['finetuned_dice']:.4f}"
            diff = f"{r['improvement']:+.4f}" if r['improvement'] else "N/A"
            print(f"{name:<25} {orig:>12} {ft:>12} {diff:>10}")

        print("=" * 70)

        # Save results
        with open(output_dir / 'ensemble_pretrain_results.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
