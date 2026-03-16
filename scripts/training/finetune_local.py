"""
Local fine-tuning script for brain metastasis ensemble models.
Runs on local GPU (RTX 3070 Ti / 8 GB VRAM).

Usage:
    python scripts/finetune_local.py
    python scripts/finetune_local.py --model improved_36patch   # Single model
    python scripts/finetune_local.py --epochs 30                # Override epochs
    python scripts/finetune_local.py --data-dir data/train      # Use original 105 cases

Fully resumable - safe to Ctrl+C and restart anytime.
"""

import gc
import json
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.unet import LightweightUNet3D
from segmentation.dataset import BrainMetDataset

# =============================================================================
# CONFIGURATION
# =============================================================================
ENSEMBLE_CONFIGS = {
    'exp1_8patch': {
        'patch_size': 8,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'finetune_epochs': 50,
        'optimal_threshold': 0.3,
    },
    'exp3_12patch_maxfn': {
        'patch_size': 12,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'finetune_epochs': 50,
        'optimal_threshold': 0.25,
    },
    'improved_24patch': {
        'patch_size': 24,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'finetune_epochs': 50,
        'optimal_threshold': 0.5,
    },
    'improved_36patch': {
        'patch_size': 36,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'finetune_epochs': 50,
        'optimal_threshold': 0.5,
    },
}

BATCH_SIZE = 2
LR = 0.0003

# =============================================================================
# LOSSES
# =============================================================================
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

    def focal_loss(self, pred, target, alpha=0.75, gamma=2.0):
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt) ** gamma * bce).mean()

    def forward(self, pred, target):
        return 0.7 * self.tversky(pred, target) + 0.3 * self.focal_loss(pred, target)

# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    tp = ((pred_binary == 1) & (target == 1)).sum().float()
    tn = ((pred_binary == 0) & (target == 0)).sum().float()
    fp = ((pred_binary == 1) & (target == 0)).sum().float()
    fn = ((pred_binary == 0) & (target == 1)).sum().float()

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)

    return {
        'dice': dice.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
    }

# =============================================================================
# FIND BEST STARTING WEIGHTS
# =============================================================================
def find_starting_weights(model_name, model_dir):
    """Find best available starting weights: pretrained > finetuned > best."""
    candidates = [
        model_dir / f'{model_name}_pretrained.pth',
        model_dir / f'{model_name}_finetuned.pth',
        model_dir / f'{model_name}_best.pth',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

# =============================================================================
# DETECT SEQUENCES
# =============================================================================
def detect_sequences(data_dir):
    """Auto-detect whether data uses t2 or bravo."""
    data_path = Path(data_dir)
    case_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case directories found in {data_dir}")

    test_case = case_dirs[0]
    if (test_case / 't2.nii.gz').exists():
        return ['t1_pre', 't1_gd', 'flair', 't2']
    elif (test_case / 'bravo.nii.gz').exists():
        return ['t1_pre', 't1_gd', 'flair', 'bravo']
    else:
        raise RuntimeError(f"Cannot detect sequences in {test_case}")

# =============================================================================
# FINE-TUNE ONE MODEL
# =============================================================================
def finetune_model(model_name, config, args):
    patch_size = config['patch_size']
    epochs = args.epochs or config['finetune_epochs']
    threshold = config['optimal_threshold']

    model_dir = ROOT / 'model'
    state_dir = model_dir / 'training_states'
    state_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"FINE-TUNING: {model_name} (patch={patch_size}, epochs={epochs})")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Check if already complete
    done_marker = state_dir / f'{model_name}_finetune_DONE'
    if done_marker.exists() and not args.force:
        print(f"Already complete - skipping (use --force to retrain)")
        return model_dir / f'{model_name}_finetuned.pth'

    # Create model
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config['base_channels'],
        use_attention=config['use_attention'],
        use_residual=config['use_residual']
    ).to(device)

    # Load starting weights
    weights_path = find_starting_weights(model_name, model_dir)
    if weights_path:
        print(f"Loading weights from: {weights_path.name}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: No starting weights found - training from scratch")

    # Dataset
    data_dir = str(ROOT / args.data_dir)
    sequences = detect_sequences(data_dir)
    print(f"Data: {data_dir}")
    print(f"Sequences: {sequences}")

    dataset = BrainMetDataset(
        data_dir=data_dir,
        sequences=sequences,
        patch_size=(patch_size, patch_size, patch_size),
        target_size=(128, 128, 128)
    )

    # Train/val split (85/15)
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    print(f"Train: {n_train}, Val: {n_val}, Batches/epoch: {len(train_loader)}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    # Resume from checkpoint
    start_epoch = 1
    best_dice = 0
    history = {'train_loss': [], 'val_dice': [], 'val_sens': [], 'val_spec': [], 'lr': []}

    state_path = state_dir / f'{model_name}_finetune_state.pth'
    if state_path.exists() and not args.force:
        print(f"Resuming from checkpoint...")
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        start_epoch = state['epoch'] + 1
        best_dice = state['best_dice']
        history = state.get('history', history)
        print(f"  Resumed at epoch {start_epoch}, best Dice={best_dice:.4f}")

        if start_epoch > epochs:
            print("Training already complete")
            done_marker.touch()
            return model_dir / f'{model_name}_finetuned.pth'

    checkpoint_path = model_dir / f'{model_name}_finetuned.pth'

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs}")
        for images, masks, _ in pbar:
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
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # --- VALIDATE ---
        model.eval()
        val_metrics = {'dice': 0, 'sensitivity': 0, 'specificity': 0}
        n_val_batches = 0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                with autocast('cuda'):
                    outputs = model(images)
                metrics = compute_metrics(outputs, masks, threshold=threshold)
                for k in val_metrics:
                    val_metrics[k] += metrics[k]
                n_val_batches += 1

        scheduler.step()

        # Average
        avg_loss = train_loss / max(n_batches, 1)
        for k in val_metrics:
            val_metrics[k] /= max(n_val_batches, 1)

        history['train_loss'].append(avg_loss)
        history['val_dice'].append(val_metrics['dice'])
        history['val_sens'].append(val_metrics['sensitivity'])
        history['val_spec'].append(val_metrics['specificity'])
        history['lr'].append(scheduler.get_last_lr()[0])

        improved = val_metrics['dice'] > best_dice
        marker = " *NEW BEST*" if improved else ""

        print(f"  Loss={avg_loss:.4f}  Dice={val_metrics['dice']:.4f}  "
              f"Sens={val_metrics['sensitivity']:.4f}  Spec={val_metrics['specificity']:.4f}{marker}")

        # Save best model
        if improved:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dice': best_dice,
                'sensitivity': val_metrics['sensitivity'],
                'specificity': val_metrics['specificity'],
                'config': config
            }, checkpoint_path)

        # Save resume state every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_dice': best_dice,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }, state_path)

        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    done_marker.touch()
    print(f"\nComplete! Best Dice: {best_dice:.4f} -> {checkpoint_path.name}")
    return checkpoint_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Local fine-tuning for brain met ensemble")
    parser.add_argument('--model', type=str, default=None,
                        help="Fine-tune specific model (e.g. improved_36patch)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument('--data-dir', type=str, default='data/preprocessed_256/train',
                        help="Training data directory (relative to project root)")
    parser.add_argument('--force', action='store_true',
                        help="Retrain from scratch (ignore existing checkpoints)")
    args = parser.parse_args()

    print("="*60)
    print("LOCAL FINE-TUNING")
    print("="*60)

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected - training will be very slow")

    # Select models
    if args.model:
        if args.model not in ENSEMBLE_CONFIGS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(ENSEMBLE_CONFIGS.keys())}")
            sys.exit(1)
        models_to_train = {args.model: ENSEMBLE_CONFIGS[args.model]}
    else:
        models_to_train = ENSEMBLE_CONFIGS

    print(f"Models: {list(models_to_train.keys())}")
    print(f"Data: {args.data_dir}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LR}")

    # Train
    results = {}
    for model_name, config in models_to_train.items():
        path = finetune_model(model_name, config, args)
        results[model_name] = str(path) if path else None
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for name, path in results.items():
        state_path = ROOT / 'model' / 'training_states' / f'{name}_finetune_state.pth'
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            print(f"  {name}: Dice={state['best_dice']:.4f} (epoch {state['epoch']})")
        else:
            print(f"  {name}: {'Done' if path else 'Failed'}")


if __name__ == '__main__':
    main()
