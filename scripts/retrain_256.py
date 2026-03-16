"""
Retrain ensemble models at native 256^3 resolution with all improvements:
- Deep supervision (auxiliary decoder heads)
- Boundary loss (signed distance maps)
- Data augmentation (MONAI pipeline)
- Small-lesion oversampling (WeightedRandomSampler)

Usage:
    python scripts/retrain_256.py                          # All 4 models, 200 epochs
    python scripts/retrain_256.py --model exp1_8patch      # Single model
    python scripts/retrain_256.py --epochs 100             # Override epochs
    python scripts/retrain_256.py --no-augment             # Disable augmentation
    python scripts/retrain_256.py --no-boundary            # Disable boundary loss
    python scripts/retrain_256.py --force                  # Restart from scratch

Fully resumable - safe to Ctrl+C and restart anytime.
"""

import gc
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.unet import LightweightUNet3D, BoundaryLoss
from segmentation.dataset import BrainMetDataset

# =============================================================================
# CONFIGURATION — patch sizes scaled for 256^3
# =============================================================================
ENSEMBLE_CONFIGS = {
    'exp1_8patch': {
        'patch_size': 32,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'optimal_threshold': 0.3,
    },
    'exp3_12patch_maxfn': {
        'patch_size': 48,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'optimal_threshold': 0.25,
    },
    'improved_24patch': {
        'patch_size': 64,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'optimal_threshold': 0.5,
    },
    'improved_36patch': {
        'patch_size': 96,
        'base_channels': 20,
        'use_attention': True,
        'use_residual': True,
        'optimal_threshold': 0.5,
    },
}

BATCH_SIZE = 2
LR = 0.0003
DEFAULT_EPOCHS = 200
BOUNDARY_RAMP_EPOCHS = 50  # epochs to ramp boundary loss weight from 0 to 1

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


class CombinedSegLoss(nn.Module):
    """0.7 * Tversky + 0.3 * Focal"""
    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

    def focal_loss(self, pred, target, alpha=0.75, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt) ** gamma * bce).mean()

    def forward(self, pred, target):
        return 0.7 * self.tversky(pred, target) + 0.3 * self.focal_loss(pred, target)


# =============================================================================
# DEEP SUPERVISION LOSS
# =============================================================================
def deep_supervision_loss(main_out, aux_outs, target, criterion, weights=(0.5, 0.3, 0.1)):
    """Compute weighted loss across main output + auxiliary decoder outputs."""
    loss = criterion(main_out, target)
    for aux, w in zip(aux_outs, weights):
        aux_resized = F.interpolate(aux, size=target.shape[2:], mode='trilinear', align_corners=False)
        loss = loss + w * criterion(aux_resized, target)
    return loss


# =============================================================================
# DISTANCE MAP DATASET WRAPPER
# =============================================================================
class DistanceMapDataset(Dataset):
    """Wraps BrainMetDataset to add signed distance map computation."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        images, mask, case_id = self.base[idx]
        dist_map = self._compute_dist_map(mask[0].numpy())
        return images, mask, torch.from_numpy(dist_map).unsqueeze(0).float(), case_id

    @staticmethod
    def _compute_dist_map(binary_mask):
        """Compute normalized signed distance map from binary mask."""
        posdis = distance_transform_edt(binary_mask == 0)
        negdis = distance_transform_edt(binary_mask == 1)
        dist = posdis - negdis
        max_val = np.abs(dist).max()
        if max_val > 0:
            dist = dist / max_val
        return dist.astype(np.float32)


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
# LESION SIZE COMPUTATION
# =============================================================================
def compute_lesion_sizes(dataset):
    """Load masks and count foreground voxels for oversampling weights."""
    import nibabel as nib
    sizes = []
    for case_dir in dataset.cases:
        for ext in ['seg.npz', 'seg.npy', 'seg.nii.gz']:
            p = case_dir / ext
            if p.exists():
                if ext.endswith('.npz'):
                    m = np.load(str(p))['data']
                elif ext.endswith('.npy'):
                    m = np.load(str(p))
                else:
                    m = nib.load(str(p)).get_fdata()
                sizes.append(int((m > 0).sum()))
                break
        else:
            sizes.append(1)  # fallback
    return sizes


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
    if (test_case / 't2.nii.gz').exists() or (test_case / 't2.npz').exists():
        return ['t1_pre', 't1_gd', 'flair', 't2']
    elif (test_case / 'bravo.nii.gz').exists() or (test_case / 'bravo.npz').exists():
        return ['t1_pre', 't1_gd', 'flair', 'bravo']
    else:
        raise RuntimeError(f"Cannot detect sequences in {test_case}")


# =============================================================================
# FIND PRETRAINED WEIGHTS
# =============================================================================
def find_pretrained_weights(model_name, model_dir):
    """Find pretrained weights (self-supervised phase)."""
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
# TRAIN ONE MODEL
# =============================================================================
def train_model(model_name, config, args):
    patch_size = config['patch_size']
    epochs = args.epochs
    threshold = config['optimal_threshold']
    use_boundary = not args.no_boundary
    use_augment = not args.no_augment

    model_dir = ROOT / 'model'
    state_dir = model_dir / 'training_states'
    state_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"RETRAIN v2: {model_name} (patch={patch_size}, epochs={epochs})")
    print(f"Device: {device}")
    print(f"Augmentation: {use_augment}, Boundary loss: {use_boundary}")
    print(f"Deep supervision: True")
    print(f"{'='*60}")

    # Check if already complete
    done_marker = state_dir / f'v2_{model_name}_finetune_DONE'
    if done_marker.exists() and not args.force:
        print(f"Already complete - skipping (use --force to retrain)")
        return model_dir / f'v2_{model_name}_finetuned.pth'

    # Create model with deep supervision
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config['base_channels'],
        use_attention=config['use_attention'],
        use_residual=config['use_residual'],
        deep_supervision=True
    ).to(device)

    # Load pretrained weights (ignore missing ds_heads keys)
    weights_path = find_pretrained_weights(model_name, model_dir)
    if weights_path:
        print(f"Loading pretrained weights from: {weights_path.name}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        # Filter out keys that don't exist in current model (strict=False handles new ds_heads)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded (strict=False for new deep supervision heads)")
    else:
        print("WARNING: No pretrained weights found - training from scratch")

    # Dataset — native 256^3, no resize
    data_dir = str(ROOT / args.data_dir)
    sequences = detect_sequences(data_dir)
    print(f"Data: {data_dir}")
    print(f"Sequences: {sequences}")

    dataset = BrainMetDataset(
        data_dir=data_dir,
        sequences=sequences,
        patch_size=(patch_size, patch_size, patch_size),
        target_size=None,  # Native 256^3, skip resize
        augment=use_augment,
    )

    # Train/val split (85/15) — use same seed as original for consistency
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Small-lesion oversampling
    print("Computing lesion sizes for oversampling...")
    all_lesion_sizes = compute_lesion_sizes(dataset)
    train_lesion_sizes = [all_lesion_sizes[i] for i in train_indices]
    sample_weights = [1.0 / np.sqrt(s + 1) for s in train_lesion_sizes]

    # Print oversampling stats
    min_size = min(train_lesion_sizes)
    max_size = max(train_lesion_sizes)
    min_weight = min(sample_weights)
    max_weight = max(sample_weights)
    print(f"  Lesion sizes: min={min_size}, max={max_size}")
    print(f"  Sampling weights: min={min_weight:.4f}, max={max_weight:.4f} (ratio={max_weight/max(min_weight,1e-8):.1f}x)")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_subset),
        replacement=True
    )

    # Wrap with distance map dataset if using boundary loss
    if use_boundary:
        train_ds = DistanceMapDataset(train_subset)
        val_ds = DistanceMapDataset(val_subset)
    else:
        train_ds = train_subset
        val_ds = val_subset

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    print(f"Train: {n_train}, Val: {n_val}, Batches/epoch: {len(train_loader)}")

    # Loss, optimizer, scheduler
    criterion = CombinedSegLoss()
    boundary_criterion = BoundaryLoss() if use_boundary else None
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    # Resume from checkpoint
    start_epoch = 1
    best_dice = 0
    history = {'train_loss': [], 'val_dice': [], 'val_sens': [], 'val_spec': [], 'lr': []}

    state_path = state_dir / f'v2_{model_name}_finetune_state.pth'
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
            return model_dir / f'v2_{model_name}_finetuned.pth'

    checkpoint_path = model_dir / f'v2_{model_name}_finetuned.pth'

    # TensorBoard
    log_dir = ROOT / 'logs' / f'v2_{model_name}'
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard: {log_dir}")

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        # Boundary weight ramp: 0 -> 1 over first BOUNDARY_RAMP_EPOCHS
        boundary_weight = min(1.0, epoch / BOUNDARY_RAMP_EPOCHS) if use_boundary else 0.0

        # --- TRAIN ---
        model.train()
        train_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs}")
        for batch in pbar:
            if use_boundary:
                images, masks, dist_maps, _ = batch
                images, masks, dist_maps = images.to(device), masks.to(device), dist_maps.to(device)
            else:
                images, masks, _ = batch
                images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                main_out, aux_outs = model(images)

                # Deep supervision loss
                loss = deep_supervision_loss(main_out, aux_outs, masks, criterion)

                # Boundary loss (on main output only)
                if use_boundary and boundary_weight > 0:
                    b_loss = boundary_criterion(main_out, dist_maps)
                    loss = loss + boundary_weight * b_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'bw': f'{boundary_weight:.2f}'})

        # --- VALIDATE ---
        model.eval()
        val_metrics = {'dice': 0, 'sensitivity': 0, 'specificity': 0}
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if use_boundary:
                    images, masks, _, _ = batch
                else:
                    images, masks, _ = batch
                images, masks = images.to(device), masks.to(device)

                with autocast('cuda'):
                    main_out, _ = model(images)

                metrics = compute_metrics(main_out, masks, threshold=threshold)
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

        # TensorBoard logging
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
        writer.add_scalar('Sensitivity/val', val_metrics['sensitivity'], epoch)
        writer.add_scalar('Specificity/val', val_metrics['specificity'], epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('BoundaryWeight', boundary_weight, epoch)

        improved = val_metrics['dice'] > best_dice
        marker = " *NEW BEST*" if improved else ""

        print(f"  Loss={avg_loss:.4f}  Dice={val_metrics['dice']:.4f}  "
              f"Sens={val_metrics['sensitivity']:.4f}  Spec={val_metrics['specificity']:.4f}  "
              f"BW={boundary_weight:.2f}{marker}")

        # Save best model
        if improved:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dice': best_dice,
                'sensitivity': val_metrics['sensitivity'],
                'specificity': val_metrics['specificity'],
                'config': config,
                'deep_supervision': True,
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

    writer.close()
    done_marker.touch()
    print(f"\nComplete! Best Dice: {best_dice:.4f} -> {checkpoint_path.name}")
    return checkpoint_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Retrain ensemble at 256^3 with all improvements")
    parser.add_argument('--model', type=str, default=None,
                        help="Train specific model (e.g. exp1_8patch)")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument('--data-dir', type=str, default='data/preprocessed_256/train',
                        help="Training data directory (relative to project root)")
    parser.add_argument('--no-augment', action='store_true',
                        help="Disable data augmentation")
    parser.add_argument('--no-boundary', action='store_true',
                        help="Disable boundary loss")
    parser.add_argument('--force', action='store_true',
                        help="Restart from scratch (ignore existing checkpoints)")
    args = parser.parse_args()

    print("=" * 60)
    print("RETRAIN v2 — 256^3 + Deep Supervision + Boundary + Oversampling")
    print("=" * 60)

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
    print(f"Epochs: {args.epochs}, Batch size: {BATCH_SIZE}, LR: {LR}")
    print(f"Augmentation: {not args.no_augment}, Boundary loss: {not args.no_boundary}")

    # Train
    results = {}
    for model_name, config in models_to_train.items():
        path = train_model(model_name, config, args)
        results[model_name] = str(path) if path else None
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, path in results.items():
        state_path = ROOT / 'model' / 'training_states' / f'v2_{name}_finetune_state.pth'
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            print(f"  {name}: Dice={state['best_dice']:.4f} (epoch {state['epoch']})")
        else:
            print(f"  {name}: {'Done' if path else 'Failed'}")


if __name__ == '__main__':
    main()
