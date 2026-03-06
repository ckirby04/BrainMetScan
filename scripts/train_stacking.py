"""
Stacking classifier for brain metastasis ensemble (v3).
Trains a lightweight 3D CNN meta-learner on base model predictions.

v3 improvements over v2:
    - Sliding window stacking inference (closes patch->volume gap)
    - Training distribution fix (configurable foreground ratio)
    - Test-time augmentation (TTA) for base model predictions
    - Failure case analysis (worst cases + distribution stats)

Usage:
    python scripts/train_stacking.py --models improved_24patch,improved_36patch \
        --regen-overlap 0.5 --min-component 20 --cache-dir stacking_cache_v3 \
        --tta --fg-ratio 0.7 --patch-size 64 --stacking-patch 32 --stacking-overlap 0.5 \
        --force-predict

Steps:
    1. Loads selected fine-tuned base models
    2. Generates predictions (sliding window with optional TTA)
    3. Builds stacking features (N predictions + variance + range)
    4. Trains lightweight CNN meta-learner
    5. Evaluates with post-processing: stacking vs individual models vs ensembles
    6. Failure case analysis

Resumable - caches base model predictions to avoid regenerating.
"""

import gc
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, label as ndimage_label
from tqdm import tqdm

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.unet import LightweightUNet3D

# =============================================================================
# BASE MODEL CONFIGS
# =============================================================================
BASE_MODELS = {
    'exp1_8patch': {'patch_size': 8, 'threshold': 0.3},
    'exp3_12patch_maxfn': {'patch_size': 12, 'threshold': 0.25},
    'improved_24patch': {'patch_size': 24, 'threshold': 0.5},
    'improved_36patch': {'patch_size': 36, 'threshold': 0.5},
}

V2_BASE_MODELS = {
    'exp1_8patch': {'patch_size': 32, 'threshold': 0.3},
    'exp3_12patch_maxfn': {'patch_size': 48, 'threshold': 0.25},
    'improved_24patch': {'patch_size': 64, 'threshold': 0.5},
    'improved_36patch': {'patch_size': 96, 'threshold': 0.5},
}

TARGET_SIZE = (128, 128, 128)

# nnU-Net integration paths
NNUNET_BASE = ROOT / 'nnUNet' / 'nnUNet_results' / 'Dataset001_BrainMets'
NNUNET_TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
NNUNET_PROBS_DIR = ROOT / 'model' / 'nnunet_probs'
RESENCM_PROBS_DIR = ROOT / 'model' / 'nnunet_probs_resencm'
NNUNET_SPLITS = (ROOT / 'nnUNet' / 'nnUNet_preprocessed'
                 / 'Dataset001_BrainMets' / 'splits_final.json')

# =============================================================================
# STACKING CLASSIFIER
# =============================================================================
class StackingClassifier(nn.Module):
    """
    3D CNN meta-learner with residual connections.
    Input: 6 channels (4 model predictions + variance + max-min range)
    Output: 1 channel (final segmentation logits)
    ~25K trainable parameters
    """
    def __init__(self, in_channels=6, mid_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
        )
        self.head = nn.Conv3d(mid_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.entry(x)
        x = self.relu(x + self.block1(x))
        x = self.relu(x + self.block2(x))
        return self.head(x)

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
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt) ** gamma * bce).mean()

    def forward(self, pred, target):
        return 0.7 * self.tversky(pred, target) + 0.3 * self.focal_loss(pred, target)

# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(pred_binary, target):
    """Compute metrics from binary prediction and target."""
    pred = pred_binary.flatten()
    target = target.flatten()

    tp = ((pred == 1) & (target == 1)).sum().float()
    tn = ((pred == 0) & (target == 0)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)

    return {
        'dice': dice.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'precision': precision.item(),
    }


def compute_relaxed_dice(pred_binary_np, target_np, tolerance=2):
    """Relaxed Dice: dilate GT by `tolerance` voxels, then compute standard Dice.

    Predictions within `tolerance` voxels of the true boundary count as correct,
    accounting for noisy label boundaries.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure
    gt_bool = target_np.astype(bool)
    pred_bool = pred_binary_np.astype(bool)
    if not gt_bool.any() and not pred_bool.any():
        return 1.0
    struct = generate_binary_structure(3, 1)
    gt_dilated = binary_dilation(gt_bool, structure=struct, iterations=tolerance)
    tp = (pred_bool & gt_dilated).sum()
    fp = (pred_bool & ~gt_dilated).sum()
    fn = (gt_bool & ~pred_bool).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-8)

# =============================================================================
# POST-PROCESSING
# =============================================================================
def postprocess_prediction(binary_mask, min_size=20):
    """Remove connected components smaller than min_size voxels."""
    labeled, n_components = ndimage_label(binary_mask)
    if n_components == 0:
        return binary_mask
    result = np.zeros_like(binary_mask)
    for i in range(1, n_components + 1):
        component = (labeled == i)
        if component.sum() >= min_size:
            result[component] = 1
    return result

# =============================================================================
# VOLUME LOADING
# =============================================================================
def detect_sequences(data_dir):
    data_path = Path(data_dir)
    case_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    test_case = case_dirs[0]
    if (test_case / 't2.nii.gz').exists():
        return ['t1_pre', 't1_gd', 'flair', 't2']
    return ['t1_pre', 't1_gd', 'flair', 'bravo']


def load_volume(case_dir, sequences, target_size=TARGET_SIZE):
    """Load and preprocess a full volume. If target_size is None, use native resolution."""
    images = []
    for seq in sequences:
        data = None
        for ext in ['.npz', '.npy', '.nii.gz']:
            path = case_dir / f"{seq}{ext}"
            if path.exists():
                if ext == '.npz':
                    data = np.load(str(path))['data'].astype(np.float32)
                elif ext == '.npy':
                    data = np.load(str(path)).astype(np.float32)
                else:
                    data = nib.load(str(path)).get_fdata().astype(np.float32)
                break

        if data is not None:
            if target_size is not None:
                factors = [t / s for t, s in zip(target_size, data.shape)]
                data = zoom(data, factors, order=1)
            mean, std = data.mean(), data.std()
            data = (data - mean) / (std + 1e-6)
            images.append(data)
        else:
            shape = target_size if target_size is not None else (256, 256, 256)
            images.append(np.zeros(shape, dtype=np.float32))
    return np.stack(images, axis=0)


def load_mask(case_dir, target_size=TARGET_SIZE):
    """Load ground truth mask. If target_size is None, use native resolution."""
    for mask_name in ['seg.npz', 'seg.npy', 'seg.nii.gz', 'mask.nii.gz', 'label.nii.gz']:
        path = case_dir / mask_name
        if path.exists():
            if mask_name.endswith('.npz'):
                data = np.load(str(path))['data'].astype(np.float32)
            elif mask_name.endswith('.npy'):
                data = np.load(str(path)).astype(np.float32)
            else:
                data = nib.load(str(path)).get_fdata().astype(np.float32)
            if target_size is not None:
                factors = [t / s for t, s in zip(target_size, data.shape)]
                data = zoom(data, factors, order=0)
            return (data > 0.5).astype(np.float32)
    return None

# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================
def sliding_window_inference(model, volume, patch_size, device, overlap=0.25):
    """
    Run inference on full volume using sliding window with overlap.
    Returns probability map (not thresholded).

    Dynamically scales batch size based on patch size to maximize GPU throughput.
    Small patches (8, 12) use very large batches since they use minimal VRAM.
    """
    model.eval()
    C, H, W, D = volume.shape
    p = patch_size
    stride = max(int(p * (1 - overlap)), 1)

    # Dynamic batch size based on patch size and VRAM
    if p <= 8:
        batch_size = 512
    elif p <= 12:
        batch_size = 256
    elif p <= 24:
        batch_size = 64
    elif p <= 48:
        batch_size = 16
    elif p <= 64:
        batch_size = 8
    elif p <= 96:
        batch_size = 4
    else:
        batch_size = 2

    # Pad volume if needed
    pad_h = (p - H % p) % p if H % stride != 0 else 0
    pad_w = (p - W % p) % p if W % stride != 0 else 0
    pad_d = (p - D % p) % p if D % stride != 0 else 0

    orig_H, orig_W, orig_D = H, W, D

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
        C, H, W, D = volume.shape

    output = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)

    coords = []
    for h in range(0, H - p + 1, stride):
        for w in range(0, W - p + 1, stride):
            for d in range(0, D - p + 1, stride):
                coords.append((h, w, d))

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for h, w, d in batch_coords:
                patches.append(volume[:, h:h+p, w:w+p, d:d+p])

            batch = torch.from_numpy(np.stack(patches)).float().to(device)
            with autocast('cuda'):
                out = model(batch)
                if isinstance(out, tuple):
                    out = out[0]  # main output (deep supervision)
                preds = torch.sigmoid(out).cpu().numpy()

            for j, (h, w, d) in enumerate(batch_coords):
                output[h:h+p, w:w+p, d:d+p] += preds[j, 0]
                count[h:h+p, w:w+p, d:d+p] += 1

    output = output / np.maximum(count, 1)
    return output[:orig_H, :orig_W, :orig_D]


def tta_sliding_window_inference(model, volume, patch_size, device, overlap=0.25):
    """
    Run inference with test-time augmentation (8 flip combinations).
    Averages predictions over all flip combinations for smoother probability maps.

    Args:
        model: The model to run inference with
        volume: Input volume of shape (C, H, W, D)
        patch_size: Patch size for sliding window
        device: torch device
        overlap: Overlap fraction for sliding window

    Returns:
        Averaged probability map of shape (H, W, D)
    """
    # 8 flip combinations on spatial axes (1,2,3) of (C,H,W,D) volume
    flip_axes_combos = [
        [],        # no flip
        [1],       # flip H
        [2],       # flip W
        [3],       # flip D
        [1, 2],    # flip HW
        [1, 3],    # flip HD
        [2, 3],    # flip WD
        [1, 2, 3], # flip HWD
    ]

    all_preds = []
    for flip_axes in flip_axes_combos:
        # Flip input volume along spatial axes
        vol = volume.copy()
        for ax in flip_axes:
            vol = np.flip(vol, axis=ax).copy()

        # Run inference
        pred = sliding_window_inference(model, vol, patch_size, device, overlap=overlap)

        # Flip prediction back (pred is H,W,D so axes are 0,1,2)
        for ax in flip_axes:
            pred = np.flip(pred, axis=ax - 1).copy()

        all_preds.append(pred)

    return np.mean(all_preds, axis=0)

# =============================================================================
# GENERATE BASE MODEL PREDICTIONS
# =============================================================================
def generate_predictions(data_dir, cache_dir, device, selected_models=None, overlap=0.5, tta=False,
                         v2=False):
    """
    Generate predictions from selected base models on all cases.
    Caches results to disk to avoid regenerating.
    When tta=True, uses test-time augmentation (8 flip combinations).
    When v2=True, uses v2 model configs (256³ native, deep supervision).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sequences = detect_sequences(data_dir)
    data_path = Path(data_dir)

    valid_prefixes = ('Mets_', 'UCSF_', 'BraTS_', 'Yale_', 'BMS_')
    all_cases = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and d.name.startswith(valid_prefixes)
    ])

    model_dir = ROOT / 'model'
    base_configs = V2_BASE_MODELS if v2 else BASE_MODELS
    target_size = None if v2 else TARGET_SIZE

    # Filter to selected models
    models_config = base_configs
    if selected_models:
        models_config = {k: v for k, v in base_configs.items() if k in selected_models}

    # Load base models
    models = {}
    for name, config in models_config.items():
        prefix = 'v2_' if v2 else ''
        model_path = model_dir / f'{prefix}{name}_finetuned.pth'
        if not model_path.exists():
            print(f"WARNING: {model_path} not found, skipping {name}")
            continue

        model = LightweightUNet3D(
            in_channels=4, out_channels=1,
            base_channels=20, use_attention=True, use_residual=True,
            deep_supervision=v2
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models[name] = (model, config['patch_size'])
        print(f"  Loaded {prefix}{name} (Dice={checkpoint.get('dice', 'N/A'):.4f})")

    infer_fn_name = "TTA sliding window (8 flips)" if tta else "sliding window"
    res_label = "256³ native" if v2 else "128³"
    print(f"\nGenerating predictions for {len(all_cases)} cases with {len(models)} models "
          f"(overlap={overlap}, method={infer_fn_name}, resolution={res_label})...")

    for case_dir in tqdm(all_cases, desc="Predicting"):
        case_id = case_dir.name
        cache_file = cache_dir / f'{case_id}.npz'

        # Skip if already cached
        if cache_file.exists():
            continue

        # Load volume at appropriate resolution
        volume = load_volume(case_dir, sequences, target_size=target_size)
        mask = load_mask(case_dir, target_size=target_size)
        if mask is None:
            continue

        # Get predictions from each model
        preds = {}
        for name, (model, patch_size) in models.items():
            if tta:
                prob_map = tta_sliding_window_inference(model, volume, patch_size, device, overlap=overlap)
            else:
                prob_map = sliding_window_inference(model, volume, patch_size, device, overlap=overlap)
            preds[name] = prob_map.astype(np.float16)  # Save as float16 to reduce cache size

        # Save predictions + mask
        np.savez_compressed(
            cache_file,
            mask=mask.astype(np.uint8),
            **preds
        )

        # Free model memory between cases
        gc.collect()
        torch.cuda.empty_cache()

    # Unload base models to free VRAM
    del models
    gc.collect()
    torch.cuda.empty_cache()

    return all_cases

# =============================================================================
# nnU-Net PREDICTION INTEGRATION
# =============================================================================
def load_nnunet_predictions_into_cache(cache_dir, data_dir, target_size, fold=0):
    """
    Load nnU-Net validation predictions and add them to the stacking cache.
    Resizes nnU-Net predictions (native space) to match cache resolution.

    Returns list of case_ids that have nnU-Net predictions.
    """
    fold_dir = NNUNET_BASE / NNUNET_TRAINER / f'fold_{fold}'
    val_dir = fold_dir / 'validation'

    if not val_dir.exists():
        print(f"  nnU-Net validation dir not found: {val_dir}")
        return []

    pred_files = sorted(val_dir.glob('*.nii.gz'))
    if not pred_files:
        print(f"  No nnU-Net prediction files found in {val_dir}")
        return []

    updated = 0
    for pred_file in pred_files:
        # Our setup preserves original case IDs, so pred filename = case_id
        case_id = pred_file.stem
        cache_file = Path(cache_dir) / f'{case_id}.npz'

        if not cache_file.exists():
            continue

        # Check if nnunet prediction already in cache
        existing = np.load(cache_file)
        if 'nnunet' in existing:
            continue

        # Load nnU-Net prediction
        pred_nii = nib.load(str(pred_file))
        pred = np.asarray(pred_nii.dataobj, dtype=np.float32)

        # Resize to cache resolution if needed
        if target_size is not None:
            pred_resized = zoom(pred, [t / s for t, s in zip(target_size, pred.shape)], order=1)
        else:
            # Check if shapes match
            mask_shape = existing['mask'].shape
            if pred.shape != mask_shape:
                pred_resized = zoom(pred, [t / s for t, s in zip(mask_shape, pred.shape)], order=1)
            else:
                pred_resized = pred

        # Re-save cache with nnunet prediction added
        save_dict = {k: existing[k] for k in existing.files}
        save_dict['nnunet'] = pred_resized.astype(np.float16)
        np.savez_compressed(cache_file, **save_dict)
        updated += 1

    print(f"  Added nnU-Net predictions to {updated} cache files")
    return [f.stem for f in pred_files]


def load_nnunet_probs_all_folds(cache_dir):
    """
    Load nnU-Net probability predictions from all 5 folds into stacking cache.

    Uses model/nnunet_probs/fold_X/{case_id}.npz files (softmax probs).
    Each case is loaded from the fold where it was unseen (proper CV).
    Probabilities are transposed from nnU-Net internal axis order to NIfTI RAS.

    Returns list of case_ids that have nnU-Net predictions.
    """
    import json as _json

    if not NNUNET_SPLITS.exists():
        print(f"  nnU-Net splits file not found: {NNUNET_SPLITS}")
        return []

    with open(NNUNET_SPLITS) as f:
        splits = _json.load(f)

    # Map each case -> which fold had it as validation
    case_to_fold = {}
    for fold_idx, split in enumerate(splits):
        for case_id in split['val']:
            case_to_fold[case_id] = fold_idx

    updated = 0
    skipped = 0

    for case_id, fold_idx in case_to_fold.items():
        cache_file = Path(cache_dir) / f'{case_id}.npz'
        if not cache_file.exists():
            continue

        try:
            # Check if nnunet prediction already in cache
            existing = np.load(cache_file)
            if 'nnunet' in existing:
                skipped += 1
                continue

            # Load probability from the correct fold
            prob_path = NNUNET_PROBS_DIR / f'fold_{fold_idx}' / f'{case_id}.npz'
            if not prob_path.exists():
                continue

            prob_data = np.load(prob_path)
            # Foreground probability = channel 1, transpose to NIfTI RAS
            pred = prob_data['probabilities'][1].transpose(2, 1, 0).astype(np.float32)

            # Resize to cache resolution if needed
            mask_shape = existing['mask'].shape
            if pred.shape != mask_shape:
                pred_resized = zoom(pred, [t / s for t, s in zip(mask_shape, pred.shape)], order=1)
            else:
                pred_resized = pred

            # Re-save cache with nnunet prediction added
            save_dict = {k: existing[k] for k in existing.files}
            save_dict['nnunet'] = pred_resized.astype(np.float16)
            np.savez_compressed(cache_file, **save_dict)
            updated += 1
        except Exception as e:
            print(f"    WARNING: error on {case_id}, skipping ({e})")
            continue

        if (updated + skipped) % 100 == 0:
            print(f"    [{updated + skipped}/{len(case_to_fold)}] "
                  f"{updated} updated, {skipped} already cached...")

    print(f"  Added nnU-Net probs (all folds) to {updated} cache files "
          f"({skipped} already cached)")
    return list(case_to_fold.keys())


def load_resencm_probs_all_folds(cache_dir):
    """
    Load ResEncM probability predictions from all 5 folds into stacking cache.

    Same pattern as load_nnunet_probs_all_folds but reads from
    model/nnunet_probs_resencm/fold_X/{case_id}.npz.

    Returns list of case_ids that have ResEncM predictions.
    """
    import json as _json

    if not NNUNET_SPLITS.exists():
        print(f"  nnU-Net splits file not found: {NNUNET_SPLITS}")
        return []

    if not RESENCM_PROBS_DIR.exists():
        print(f"  ResEncM probs dir not found: {RESENCM_PROBS_DIR}")
        print("  Run: python scripts/nnunet_probs.py --mode val "
              "--trainer nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
        return []

    with open(NNUNET_SPLITS) as f:
        splits = _json.load(f)

    # Map each case -> which fold had it as validation
    case_to_fold = {}
    for fold_idx, split in enumerate(splits):
        for case_id in split['val']:
            case_to_fold[case_id] = fold_idx

    updated = 0
    skipped = 0

    for case_id, fold_idx in case_to_fold.items():
        cache_file = Path(cache_dir) / f'{case_id}.npz'
        if not cache_file.exists():
            continue

        try:
            existing = np.load(cache_file)
            if 'resencm' in existing:
                skipped += 1
                continue

            prob_path = RESENCM_PROBS_DIR / f'fold_{fold_idx}' / f'{case_id}.npz'
            if not prob_path.exists():
                continue

            prob_data = np.load(prob_path)
            # Foreground probability = channel 1, transpose to NIfTI RAS
            pred = prob_data['probabilities'][1].transpose(2, 1, 0).astype(np.float32)

            # Resize to cache resolution if needed
            mask_shape = existing['mask'].shape
            if pred.shape != mask_shape:
                pred_resized = zoom(pred, [t / s for t, s in zip(mask_shape, pred.shape)], order=1)
            else:
                pred_resized = pred

            save_dict = {k: existing[k] for k in existing.files}
            save_dict['resencm'] = pred_resized.astype(np.float16)
            np.savez_compressed(cache_file, **save_dict)
            updated += 1
        except Exception as e:
            print(f"    WARNING: error on {case_id}, skipping ({e})")
            continue

        if (updated + skipped) % 100 == 0:
            print(f"    [{updated + skipped}/{len(case_to_fold)}] "
                  f"{updated} updated, {skipped} already cached...")

    print(f"  Added ResEncM probs (all folds) to {updated} cache files "
          f"({skipped} already cached)")
    return list(case_to_fold.keys())


# =============================================================================
# STACKING DATASET
# =============================================================================
class StackingDataset(Dataset):
    """
    Dataset that loads cached base model predictions and builds stacking features.
    Extracts random patches for memory-efficient training.
    """
    def __init__(self, case_ids, cache_dir, model_names, patch_size=32, fg_ratio=0.7):
        self.case_ids = case_ids
        self.cache_dir = Path(cache_dir)
        self.model_names = model_names
        self.patch_size = patch_size
        self.fg_ratio = fg_ratio

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        cache_file = self.cache_dir / f'{case_id}.npz'
        data = np.load(cache_file)

        mask = data['mask'].astype(np.float32)  # (H, W, D)

        # Stack model predictions
        preds = []
        for name in self.model_names:
            preds.append(data[name].astype(np.float32))
        preds = np.stack(preds, axis=0)  # (4, H, W, D)

        # Disagreement features
        variance = preds.var(axis=0, keepdims=True)       # (1, H, W, D)
        range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)  # (1, H, W, D)

        # Full feature stack: 6 channels
        features = np.concatenate([preds, variance, range_map], axis=0)  # (6, H, W, D)

        # Extract random patch (biased toward foreground)
        p = self.patch_size
        C, H, W, D = features.shape

        foreground = np.where(mask > 0)
        if len(foreground[0]) > 0 and np.random.rand() < self.fg_ratio:
            idx_fg = np.random.randint(len(foreground[0]))
            ch, cw, cd = foreground[0][idx_fg], foreground[1][idx_fg], foreground[2][idx_fg]
        else:
            ch = np.random.randint(p // 2, max(H - p // 2, p // 2 + 1))
            cw = np.random.randint(p // 2, max(W - p // 2, p // 2 + 1))
            cd = np.random.randint(p // 2, max(D - p // 2, p // 2 + 1))

        h_start = max(0, min(ch - p // 2, H - p))
        w_start = max(0, min(cw - p // 2, W - p))
        d_start = max(0, min(cd - p // 2, D - p))

        feat_patch = features[:, h_start:h_start+p, w_start:w_start+p, d_start:d_start+p]
        mask_patch = mask[h_start:h_start+p, w_start:w_start+p, d_start:d_start+p]
        mask_patch = mask_patch[np.newaxis]  # (1, p, p, p)

        return (
            torch.from_numpy(feat_patch).float(),
            torch.from_numpy(mask_patch).float(),
        )

# =============================================================================
# FULL-VOLUME STACKING FEATURES
# =============================================================================
def build_stacking_features(cache_file, model_names):
    """Build full-volume stacking features from cached predictions."""
    data = np.load(cache_file)
    mask = data['mask']

    preds = []
    for name in model_names:
        preds.append(data[name])
    preds = np.stack(preds, axis=0)

    variance = preds.var(axis=0, keepdims=True)
    range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)

    features = np.concatenate([preds, variance, range_map], axis=0)
    return features, preds, mask

# =============================================================================
# EVALUATION
# =============================================================================
def get_prob_maps(case_id, cache_dir, model_names, stacking_model, device,
                  stacking_patch_size=32, stacking_overlap=0.5):
    """Get probability maps for all methods from a single case."""
    cache_file = Path(cache_dir) / f'{case_id}.npz'
    if not cache_file.exists():
        return None, None

    features, preds, mask = build_stacking_features(cache_file, model_names)

    prob_maps = {}
    # Individual models
    for i, name in enumerate(model_names):
        prob_maps[name] = preds[i]
    # Simple ensembles
    prob_maps['simple_average'] = preds.mean(axis=0)
    prob_maps['max_fusion'] = preds.max(axis=0)
    # Stacking - sliding window inference (matches training patch distribution)
    prob_maps['stacking'] = sliding_window_inference(
        stacking_model, features, stacking_patch_size, device, overlap=stacking_overlap
    )

    return prob_maps, mask


def tune_thresholds(tune_case_ids, cache_dir, model_names, stacking_model, device,
                    min_component_size=0, stacking_patch_size=32, stacking_overlap=0.5):
    """
    Find optimal threshold for each method by sweeping on tuning set.
    Returns dict of {method_name: best_threshold}.
    """
    thresholds_to_try = np.arange(0.1, 0.95, 0.05)
    stacking_model.eval()

    # Collect probability maps for all tuning cases
    all_prob_maps = {}  # method -> list of (prob_map, mask)
    method_names = None

    for case_id in tqdm(tune_case_ids, desc="Threshold tuning"):
        prob_maps, mask = get_prob_maps(
            case_id, cache_dir, model_names, stacking_model, device,
            stacking_patch_size=stacking_patch_size, stacking_overlap=stacking_overlap
        )
        if prob_maps is None:
            continue

        if method_names is None:
            method_names = list(prob_maps.keys())
            all_prob_maps = {m: [] for m in method_names}

        for m in method_names:
            all_prob_maps[m].append((prob_maps[m], mask))

    # Sweep thresholds
    best_thresholds = {}
    print(f"\n{'Method':<25} {'Best Thresh':<14} {'Dice @ Best':<12}")
    print("-" * 55)

    for method_name in method_names:
        best_dice = 0
        best_t = 0.5

        for t in thresholds_to_try:
            dices = []
            for prob_map, mask in all_prob_maps[method_name]:
                pred_binary = (prob_map > t).astype(np.float32)
                if min_component_size > 0:
                    pred_binary = postprocess_prediction(pred_binary, min_size=min_component_size)
                mask_t = torch.from_numpy(mask).float()
                pred_t = torch.from_numpy(pred_binary).float()
                m = compute_metrics(pred_t, mask_t)
                dices.append(m['dice'])

            avg_dice = np.mean(dices)
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_t = t

        best_thresholds[method_name] = float(best_t)
        print(f"{method_name:<25} {best_t:<14.2f} {best_dice:<12.4f}")

    return best_thresholds


def evaluate_all_methods(val_case_ids, cache_dir, model_names, stacking_model, device,
                         thresholds, min_component_size=0,
                         stacking_patch_size=32, stacking_overlap=0.5):
    """Compare stacking vs individual models vs simple ensembles using tuned thresholds.
    Returns (results, all_metrics) where all_metrics tracks per-case results."""
    stacking_model.eval()

    all_metrics = {m: [] for m in thresholds}

    for case_id in tqdm(val_case_ids, desc="Evaluating"):
        prob_maps, mask = get_prob_maps(
            case_id, cache_dir, model_names, stacking_model, device,
            stacking_patch_size=stacking_patch_size, stacking_overlap=stacking_overlap
        )
        if prob_maps is None:
            continue

        mask_t = torch.from_numpy(mask).float()

        for method_name, threshold in thresholds.items():
            pred_binary = (prob_maps[method_name] > threshold).astype(np.float32)
            if min_component_size > 0:
                pred_binary = postprocess_prediction(pred_binary, min_size=min_component_size)
            pred_t = torch.from_numpy(pred_binary).float()
            metrics = compute_metrics(pred_t, mask_t)
            metrics['relaxed_dice_2'] = compute_relaxed_dice(pred_binary, mask, tolerance=2)
            all_metrics[method_name].append((case_id, metrics))

    # Aggregate
    results = {}
    for method_name, case_metrics in all_metrics.items():
        if not case_metrics:
            continue
        results[method_name] = {
            'dice': np.mean([m['dice'] for _, m in case_metrics]),
            'sensitivity': np.mean([m['sensitivity'] for _, m in case_metrics]),
            'specificity': np.mean([m['specificity'] for _, m in case_metrics]),
            'precision': np.mean([m['precision'] for _, m in case_metrics]),
            'relaxed_dice_2': np.mean([m['relaxed_dice_2'] for _, m in case_metrics]),
            'std_dice': np.std([m['dice'] for _, m in case_metrics]),
            'n_cases': len(case_metrics),
            'threshold': thresholds[method_name],
        }

    return results, all_metrics


def analyze_failures(all_metrics, best_method_name, results_path):
    """Analyze worst performing cases and print distribution statistics."""
    case_metrics = all_metrics.get(best_method_name, [])
    if not case_metrics:
        print("No cases to analyze.")
        return

    # Sort by dice (ascending = worst first)
    sorted_cases = sorted(case_metrics, key=lambda x: x[1]['dice'])
    dices = [m['dice'] for _, m in case_metrics]

    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS: {best_method_name}")
    print(f"{'='*60}")

    # Worst 10 cases
    print(f"\nWorst 10 cases by Dice:")
    print(f"{'Case ID':<30} {'Dice':<10} {'RelD2':<10} {'Sens':<10} {'Prec':<10}")
    print("-" * 70)
    for case_id, metrics in sorted_cases[:10]:
        print(f"{case_id:<30} {metrics['dice']:.4f}     {metrics.get('relaxed_dice_2', 0):.4f}     "
              f"{metrics['sensitivity']:.4f}     {metrics['precision']:.4f}")

    # Distribution stats
    dices_arr = np.array(dices)
    print(f"\nDice distribution ({len(dices)} cases):")
    print(f"  Min:    {np.min(dices_arr):.4f}")
    print(f"  25th:   {np.percentile(dices_arr, 25):.4f}")
    print(f"  Median: {np.median(dices_arr):.4f}")
    print(f"  75th:   {np.percentile(dices_arr, 75):.4f}")
    print(f"  Max:    {np.max(dices_arr):.4f}")
    print(f"  Mean:   {np.mean(dices_arr):.4f} +/- {np.std(dices_arr):.4f}")

    # Save per-case metrics
    per_case = {}
    for case_id, metrics in case_metrics:
        per_case[case_id] = {k: float(v) for k, v in metrics.items()}

    # Derive per-case filename from results path (e.g. stacking_v4_results.json -> stacking_v4_results_per_case.json)
    per_case_path = results_path.parent / results_path.name.replace('_results.json', '_results_per_case.json')
    with open(per_case_path, 'w') as f:
        json.dump(per_case, f, indent=2)
    print(f"\nPer-case metrics saved to: {per_case_path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train stacking classifier (v3)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patch-size', type=int, default=64,
                        help="Patch size for stacking training (default: 64)")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default='data/preprocessed_256/train')
    parser.add_argument('--force-predict', action='store_true',
                        help="Regenerate base model predictions (ignore cache)")
    parser.add_argument('--models', type=str, default='improved_24patch,improved_36patch',
                        help="Comma-separated model names (default: improved_24patch,improved_36patch)")
    parser.add_argument('--min-component', type=int, default=20,
                        help="Min connected component size in voxels (default: 20)")
    parser.add_argument('--regen-overlap', type=float, default=0.5,
                        help="Overlap for regenerating predictions (default: 0.5)")
    parser.add_argument('--cache-dir', type=str, default='stacking_cache_v3',
                        help="Cache directory name under model/ (default: stacking_cache_v3)")
    parser.add_argument('--fg-ratio', type=float, default=0.7,
                        help="Foreground sampling ratio during training (default: 0.7)")
    parser.add_argument('--tta', action='store_true',
                        help="Use test-time augmentation (8 flips) for base model predictions")
    parser.add_argument('--stacking-patch', type=int, default=32,
                        help="Patch size for stacking sliding window inference (default: 32)")
    parser.add_argument('--stacking-overlap', type=float, default=0.5,
                        help="Overlap for stacking sliding window inference (default: 0.5)")
    parser.add_argument('--v2', action='store_true',
                        help="Use v2 models (256³ native, deep supervision)")
    parser.add_argument('--include-nnunet', action='store_true',
                        help="Include nnU-Net predictions as additional stacking input channel")
    parser.add_argument('--nnunet-fold', type=int, default=0,
                        help="nnU-Net fold to use for binary predictions (default: 0)")
    parser.add_argument('--nnunet-all-folds', action='store_true',
                        help="Use nnU-Net probability predictions from all 5 CV folds")
    parser.add_argument('--include-resencm', action='store_true',
                        help="Include ResEncM predictions as additional stacking input channel")
    args = parser.parse_args()

    # Apply v2 defaults if --v2 flag is set
    if args.v2:
        if args.models == 'improved_24patch,improved_36patch':
            args.models = 'exp1_8patch,exp3_12patch_maxfn,improved_24patch,improved_36patch'
        if args.cache_dir == 'stacking_cache_v3':
            args.cache_dir = 'stacking_cache_v4'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = ROOT / 'model'
    cache_dir = ROOT / 'model' / args.cache_dir
    state_dir = model_dir / 'training_states'

    # Parse selected models
    model_names = [m.strip() for m in args.models.split(',')]
    in_channels = len(model_names) + 2  # N predictions + variance + range

    version = "v4 (256³ v2 models)" if args.v2 else "v3"
    print("=" * 60)
    print(f"STACKING CLASSIFIER TRAINING ({version})")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"V2 models: {args.v2}")
    print(f"Models: {model_names}")
    print(f"Input channels: {in_channels} ({len(model_names)} preds + variance + range)")
    print(f"Overlap: {args.regen_overlap}")
    print(f"Min component size: {args.min_component}")
    print(f"Cache dir: {cache_dir}")
    print(f"TTA: {args.tta}")
    print(f"FG ratio: {args.fg_ratio}")
    print(f"Training patch size: {args.patch_size}")
    print(f"Stacking inference patch: {args.stacking_patch}, overlap: {args.stacking_overlap}")

    data_dir = str(ROOT / args.data_dir)

    # =========================================================================
    # STEP 1: Generate base model predictions (cached)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Generate base model predictions")
    print("=" * 60)

    if args.force_predict:
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    all_cases = generate_predictions(
        data_dir, cache_dir, device,
        selected_models=model_names, overlap=args.regen_overlap,
        tta=args.tta, v2=args.v2
    )

    # Filter to cases that have cached predictions
    cached_cases = []
    for case_dir in all_cases:
        cache_file = cache_dir / f'{case_dir.name}.npz'
        if cache_file.exists():
            cached_cases.append(case_dir.name)

    print(f"\nCached predictions: {len(cached_cases)} cases")

    # Optional: add nnU-Net predictions to cache
    if args.nnunet_all_folds or args.include_nnunet:
        if args.nnunet_all_folds:
            print("\n  Adding nnU-Net prob predictions (all 5 folds) to stacking cache...")
            nn_cases = load_nnunet_probs_all_folds(cache_dir)
        else:
            print("\n  Adding nnU-Net predictions to stacking cache...")
            target_size_nnunet = None if args.v2 else TARGET_SIZE
            nn_cases = load_nnunet_predictions_into_cache(
                cache_dir, data_dir, target_size_nnunet, fold=args.nnunet_fold
            )
        if nn_cases:
            model_names.append('nnunet')
            in_channels = len(model_names) + 2
            # Filter to only cases that have nnunet in cache
            valid = []
            for cid in cached_cases:
                try:
                    d = np.load(cache_dir / f'{cid}.npz')
                    if 'nnunet' in d:
                        valid.append(cid)
                except Exception:
                    pass
            dropped = len(cached_cases) - len(valid)
            if dropped > 0:
                print(f"  Dropped {dropped} cases missing nnunet predictions")
            cached_cases = valid
            print(f"  Updated: {len(model_names)} models, {in_channels} input channels, "
                  f"{len(cached_cases)} cases")
        else:
            print("  No nnU-Net predictions available - proceeding without")

    # Optional: add ResEncM predictions to cache
    if args.include_resencm:
        print("\n  Adding ResEncM prob predictions (all 5 folds) to stacking cache...")
        resencm_cases = load_resencm_probs_all_folds(cache_dir)
        if resencm_cases:
            model_names.append('resencm')
            in_channels = len(model_names) + 2
            # Filter to only cases that have resencm in cache
            valid = []
            for cid in cached_cases:
                try:
                    d = np.load(cache_dir / f'{cid}.npz')
                    if 'resencm' in d:
                        valid.append(cid)
                except Exception:
                    pass
            dropped = len(cached_cases) - len(valid)
            if dropped > 0:
                print(f"  Dropped {dropped} cases missing resencm predictions")
            cached_cases = valid
            print(f"  Updated: {len(model_names)} models, {in_channels} input channels, "
                  f"{len(cached_cases)} cases")
        else:
            print("  No ResEncM predictions available - proceeding without")

    # =========================================================================
    # STEP 2: Train/val split (same seed as fine-tuning)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Train/val split")
    print("=" * 60)

    import random
    random.seed(42)
    cases_shuffled = cached_cases.copy()
    random.shuffle(cases_shuffled)

    n_val = int(len(cases_shuffled) * 0.15)
    val_cases = cases_shuffled[:n_val]
    train_cases = cases_shuffled[n_val:]

    # Further split train into stacking-train and stacking-val
    # to avoid training stacking on the same data base models saw
    n_stack_val = int(len(train_cases) * 0.15)
    stack_val_cases = train_cases[:n_stack_val]
    stack_train_cases = train_cases[n_stack_val:]

    print(f"Base model val set: {n_val} cases (held out)")
    print(f"Stacking train: {len(stack_train_cases)} cases")
    print(f"Stacking val: {len(stack_val_cases)} cases")
    print(f"Final eval: {n_val} cases (never seen by any model)")

    # =========================================================================
    # STEP 3: Train stacking classifier
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Train stacking classifier")
    print("=" * 60)

    train_ds = StackingDataset(stack_train_cases, cache_dir, model_names,
                               patch_size=args.patch_size, fg_ratio=args.fg_ratio)
    val_ds = StackingDataset(stack_val_cases, cache_dir, model_names,
                             patch_size=args.patch_size, fg_ratio=args.fg_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    stacking_model = StackingClassifier(in_channels=in_channels).to(device)
    n_params = sum(p.numel() for p in stacking_model.parameters())
    print(f"Stacking model: {n_params:,} parameters ({in_channels} input channels)")

    criterion = CombinedLoss()
    optimizer = optim.AdamW(stacking_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    # Resume if checkpoint exists and config matches
    stacking_tag = 'v4' if args.v2 else 'v3'
    stacking_state_path = state_dir / f'stacking_{stacking_tag}_finetune_state.pth'
    stacking_best_path = model_dir / f'stacking_{stacking_tag}_classifier.pth'
    start_epoch = 1
    best_dice = 0
    history = {'train_loss': [], 'val_dice': [], 'val_sens': [], 'val_spec': []}

    current_config = {
        'in_channels': in_channels,
        'model_names': model_names,
        'fg_ratio': args.fg_ratio,
        'patch_size': args.patch_size,
        'tta': args.tta,
    }

    if stacking_state_path.exists():
        state = torch.load(stacking_state_path, map_location=device, weights_only=False)
        saved_config = {
            'in_channels': state.get('in_channels', 6),
            'model_names': state.get('model_names', []),
            'fg_ratio': state.get('fg_ratio', 0.9),
            'patch_size': state.get('patch_size', 32),
            'tta': state.get('tta', False),
        }
        if saved_config == current_config:
            print("Resuming from checkpoint...")
            stacking_model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            scaler.load_state_dict(state['scaler_state_dict'])
            start_epoch = state['epoch'] + 1
            best_dice = state['best_dice']
            history = state.get('history', history)
            print(f"  Resumed at epoch {start_epoch}, best Dice={best_dice:.4f}")
        else:
            print(f"Config mismatch (saved={saved_config}, need={current_config})")
            print("  Starting fresh training...")
            stacking_state_path.unlink()
            if stacking_best_path.exists():
                stacking_best_path.unlink()

    patience = 30
    no_improve_count = 0

    for epoch in range(start_epoch, args.epochs + 1):
        # --- TRAIN ---
        stacking_model.train()
        train_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[stacking] Epoch {epoch}/{args.epochs}")
        for features, masks in pbar:
            features, masks = features.to(device), masks.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = stacking_model(features)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(stacking_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # --- VALIDATE (patch-based) ---
        stacking_model.eval()
        val_dice_sum = 0
        val_sens_sum = 0
        val_spec_sum = 0
        n_val_batches = 0

        with torch.no_grad():
            for features, masks in val_loader:
                features, masks = features.to(device), masks.to(device)
                with autocast('cuda'):
                    outputs = stacking_model(features)

                pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                m = compute_metrics(pred_binary, masks)
                val_dice_sum += m['dice']
                val_sens_sum += m['sensitivity']
                val_spec_sum += m['specificity']
                n_val_batches += 1

        scheduler.step()

        avg_loss = train_loss / max(n_batches, 1)
        avg_dice = val_dice_sum / max(n_val_batches, 1)
        avg_sens = val_sens_sum / max(n_val_batches, 1)
        avg_spec = val_spec_sum / max(n_val_batches, 1)

        history['train_loss'].append(avg_loss)
        history['val_dice'].append(avg_dice)
        history['val_sens'].append(avg_sens)
        history['val_spec'].append(avg_spec)

        improved = avg_dice > best_dice
        marker = " *NEW BEST*" if improved else ""

        print(f"  Loss={avg_loss:.4f}  Dice={avg_dice:.4f}  "
              f"Sens={avg_sens:.4f}  Spec={avg_spec:.4f}{marker}")

        if improved:
            best_dice = avg_dice
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': stacking_model.state_dict(),
                'dice': best_dice,
                'n_params': n_params,
                'in_channels': in_channels,
                'model_names': model_names,
                'fg_ratio': args.fg_ratio,
                'patch_size': args.patch_size,
                'tta': args.tta,
            }, stacking_best_path)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n  Early stopping: no improvement for {patience} epochs")
                break

        # Save resume state
        torch.save({
            'epoch': epoch,
            'model_state_dict': stacking_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_dice': best_dice,
            'history': history,
            'in_channels': in_channels,
            'model_names': model_names,
            'fg_ratio': args.fg_ratio,
            'patch_size': args.patch_size,
            'tta': args.tta,
            'timestamp': datetime.now().isoformat(),
        }, stacking_state_path)

    print(f"\nStacking training complete! Best patch Dice: {best_dice:.4f}")

    # =========================================================================
    # STEP 4: Threshold tuning on stacking-val set
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"STEP 4: Threshold tuning (post-processing min_component={args.min_component})")
    print("=" * 60)

    # Load best stacking model
    if stacking_best_path.exists():
        checkpoint = torch.load(stacking_best_path, map_location=device, weights_only=False)
        stacking_model.load_state_dict(checkpoint['model_state_dict'])
    stacking_model.eval()

    # Tune thresholds on stacking-val set (not the final eval set)
    best_thresholds = tune_thresholds(
        stack_val_cases, cache_dir, model_names, stacking_model, device,
        min_component_size=args.min_component,
        stacking_patch_size=args.stacking_patch,
        stacking_overlap=args.stacking_overlap
    )

    # =========================================================================
    # STEP 5: Full-volume evaluation with tuned thresholds
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Full-volume evaluation (tuned thresholds + post-processing)")
    print("=" * 60)

    results, all_metrics = evaluate_all_methods(
        val_cases, cache_dir, model_names, stacking_model, device, best_thresholds,
        min_component_size=args.min_component,
        stacking_patch_size=args.stacking_patch,
        stacking_overlap=args.stacking_overlap
    )

    # Print results table
    print(f"\n{'Method':<25} {'Dice':<10} {'RelDice2':<10} {'Sens':<10} {'Prec':<10} {'Thresh':<8}")
    print("-" * 80)

    for method_name, metrics in sorted(results.items(), key=lambda x: -x[1]['dice']):
        print(f"{method_name:<25} {metrics['dice']:.4f}     {metrics['relaxed_dice_2']:.4f}     "
              f"{metrics['sensitivity']:.4f}     {metrics['precision']:.4f}     "
              f"{metrics['threshold']:.2f}")

    # Save results
    results_tag = 'v4' if args.v2 else 'v3'
    results_path = model_dir / f'stacking_{results_tag}_results.json'
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: float(vv) for kk, vv in v.items()}
    json_results['_thresholds'] = {k: float(v) for k, v in best_thresholds.items()}
    json_results['_config'] = {
        'models': model_names,
        'in_channels': in_channels,
        'overlap': args.regen_overlap,
        'min_component_size': args.min_component,
        'cache_dir': args.cache_dir,
        'fg_ratio': args.fg_ratio,
        'tta': args.tta,
        'patch_size': args.patch_size,
        'stacking_patch': args.stacking_patch,
        'stacking_overlap': args.stacking_overlap,
    }

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Highlight winner
    best_method = max(
        [(k, v) for k, v in results.items() if not k.startswith('_')],
        key=lambda x: x[1]['dice']
    )
    print(f"\nBest method: {best_method[0]} (Dice={best_method[1]['dice']:.4f}, "
          f"threshold={best_thresholds[best_method[0]]:.2f})")

    # =========================================================================
    # STEP 6: Failure case analysis
    # =========================================================================
    analyze_failures(all_metrics, best_method[0], results_path)


if __name__ == '__main__':
    main()
