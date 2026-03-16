"""
External Validation — Run stacking ensemble inference + full evaluation on PRETREAT.

Pipeline:
  1. Run all 4 custom base models at 128³ (patch sizes 8, 12, 24, 36)
  2. Run nnU-Net 3D + 2D inference (if available)
  3. Build stacking features (6 predictions + variance + range = 8ch)
  4. Run stacking meta-learner (v4) at patch_size=32, threshold=0.9
  5. Upsample to 256³, postprocess, evaluate against ground truth
  6. Also evaluates top-3 simple average for comparison

Usage:
    # Full pipeline (stacking + top3 comparison):
    python scripts/eval_external_validation.py

    # Skip inference, only evaluate existing predictions:
    python scripts/eval_external_validation.py --eval-only

    # Custom paths:
    python scripts/eval_external_validation.py \
        --data-dir data/external_validation/test \
        --output-dir results/external_validation
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import (distance_transform_edt, binary_dilation,
                            binary_erosion, generate_binary_structure, zoom)
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "segmentation"))

from unet import LightweightUNet3D
from stacking import (
    StackingClassifier, STACKING_IN_CHANNELS,
    STACKING_PATCH_SIZE, STACKING_OVERLAP, STACKING_THRESHOLD,
)

# ─── Config ──────────────────────────────────────────────────────────────────

SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']

# Base models for stacking (all 4 custom models)
BASE_MODELS = {
    'exp1_8patch':        {'checkpoint': 'exp1_8patch_finetuned.pth',        'patch_size': 8},
    'exp3_12patch_maxfn': {'checkpoint': 'exp3_12patch_maxfn_finetuned.pth', 'patch_size': 12},
    'improved_24patch':   {'checkpoint': 'improved_24patch_finetuned.pth',   'patch_size': 24},
    'improved_36patch':   {'checkpoint': 'improved_36patch_finetuned.pth',   'patch_size': 36},
}

# Top-3 for simple average comparison
TOP3_MODELS = ['exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']
TOP3_THRESHOLD = 0.40

# Stacking operates at 128³
STACKING_TARGET_SIZE = (128, 128, 128)

# Final eval at 256³
EVAL_TARGET_SIZE = (256, 256, 256)

MIN_LESION_SIZE = 20


# ─── Volume loading ─────────────────────────────────────────────────────────

def load_volume(patient_dir, target_size):
    """Load 4-channel MRI, z-score normalize, resize to target_size."""
    images = []
    affine = None

    for seq in SEQUENCES:
        nii_path = Path(patient_dir) / f'{seq}.nii.gz'
        if not nii_path.exists():
            images.append(np.zeros(target_size, dtype=np.float32))
            continue

        nii = nib.load(str(nii_path))
        img = np.asarray(nii.dataobj, dtype=np.float32)

        if affine is None:
            affine = nii.affine

        # Resize to target
        if img.shape != target_size:
            factors = [t / s for t, s in zip(target_size, img.shape)]
            img = zoom(img, factors, order=1)

        # Z-score normalize
        mean, std = img.mean(), img.std()
        if std > 0:
            img = (img - mean) / std

        images.append(img)

    return np.stack(images, axis=0), affine


def load_mask(patient_dir, target_size):
    """Load ground truth mask, resize to target_size."""
    seg_path = Path(patient_dir) / 'seg.nii.gz'
    if not seg_path.exists():
        return None

    data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
    if data.shape != target_size:
        factors = [t / s for t, s in zip(target_size, data.shape)]
        data = zoom(data, factors, order=0)
    return (data > 0.5).astype(np.float32)


# ─── Sliding window inference ───────────────────────────────────────────────

@torch.no_grad()
def sliding_window_inference(model, volume, patch_size, device, overlap=0.5):
    """Sliding window inference with dynamic batch size. Returns prob map."""
    model.eval()
    C, H, W, D = volume.shape
    p = patch_size
    stride = max(int(p * (1 - overlap)), 1)

    if p <= 8:
        batch_size = 512
    elif p <= 12:
        batch_size = 256
    elif p <= 24:
        batch_size = 64
    else:
        batch_size = 32

    # Pad if needed
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

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        patches = [volume[:, h:h+p, w:w+p, d:d+p] for h, w, d in batch_coords]

        batch = torch.from_numpy(np.stack(patches)).float().to(device)
        if device.type == 'cuda':
            with autocast('cuda'):
                out = model(batch)
                if isinstance(out, tuple):
                    out = out[0]
                preds = torch.sigmoid(out).cpu().numpy()
        else:
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            preds = torch.sigmoid(out).cpu().numpy()

        for j, (h, w, d) in enumerate(batch_coords):
            output[h:h+p, w:w+p, d:d+p] += preds[j, 0]
            count[h:h+p, w:w+p, d:d+p] += 1

    output = output / np.maximum(count, 1)
    return output[:orig_H, :orig_W, :orig_D]


# ─── Postprocessing ─────────────────────────────────────────────────────────

def postprocess(binary_mask, min_size=MIN_LESION_SIZE):
    """Remove connected components smaller than min_size voxels."""
    labeled, n = ndimage_label(binary_mask)
    result = np.zeros_like(binary_mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_size:
            result[labeled == i] = 1
    return result


# ─── nnU-Net inference ───────────────────────────────────────────────────────

def run_nnunet_inference(data_dir, output_dir, config_name, device):
    """
    Run nnU-Net inference using nnUNetv2_predict CLI.
    Returns path to predictions dir, or None if nnU-Net is not available.
    """
    nnunet_raw = ROOT / 'nnUNet' / 'nnUNet_raw' / 'Dataset002_BrainMetsExtVal'
    images_ts = nnunet_raw / 'imagesTs'

    if not images_ts.exists():
        print(f"    nnU-Net imagesTs not found at {images_ts}")
        return None

    pred_dir = Path(output_dir) / f'nnunet_{config_name}_preds'
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = list(pred_dir.glob('*.nii.gz'))
    if len(existing) >= 10:
        print(f"    Using existing nnU-Net {config_name} predictions ({len(existing)} files)")
        return pred_dir

    import subprocess

    nnunet_results = ROOT / 'nnUNet' / 'nnUNet_results' / 'Dataset001_BrainMets'
    trainer_dir = nnunet_results / f'nnUNetTrainer__nnUNetPlans__{config_name}'

    if not trainer_dir.exists():
        print(f"    nnU-Net trainer not found: {trainer_dir}")
        return None

    cmd = [
        sys.executable, '-m', 'nnunetv2.inference.predict_from_raw_data',
        '-i', str(images_ts),
        '-o', str(pred_dir),
        '-d', '001',
        '-c', config_name,
        '-tr', 'nnUNetTrainer',
        '--disable_tta',
    ]

    print(f"    Running nnU-Net {config_name} inference...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            print(f"    nnU-Net {config_name} failed: {result.stderr[:200]}")
            return None
        print(f"    nnU-Net {config_name} complete")
        return pred_dir
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    nnU-Net {config_name} unavailable: {e}")
        return None


def load_nnunet_prediction(pred_dir, case_id, target_size):
    """Load a single nnU-Net prediction and resize to target_size."""
    pred_path = Path(pred_dir) / f'{case_id}.nii.gz'
    if not pred_path.exists():
        return None

    data = nib.load(str(pred_path)).get_fdata().astype(np.float32)
    if data.shape != target_size:
        factors = [t / s for t, s in zip(target_size, data.shape)]
        data = zoom(data, factors, order=1)
    return data


# ─── Full stacking inference pipeline ────────────────────────────────────────

def run_stacking_pipeline(data_dir, output_dir, model_dir, device):
    """
    Full stacking pipeline:
      1. Load all 4 base models, generate predictions at 128³
      2. Attempt nnU-Net 3D + 2D inference
      3. Build stacking features, run meta-learner
      4. Also generate top-3 average predictions for comparison
      5. Upsample to 256³, save predictions
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / 'stacking_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Discover cases ───────────────────────────────────────────
    patient_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / 't1_gd.nii.gz').exists()
    ])
    print(f"\n  Found {len(patient_dirs)} cases")

    # ── Step 2: Load base models ─────────────────────────────────────────
    print(f"\n  Loading 4 base models...")
    models = {}
    for name, config in BASE_MODELS.items():
        path = model_dir / config['checkpoint']
        if not path.exists():
            print(f"    WARNING: {path} not found, skipping {name}")
            continue

        model = LightweightUNet3D(
            in_channels=4, out_channels=1,
            base_channels=20, use_attention=True, use_residual=True,
        ).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models[name] = (model, config['patch_size'])
        print(f"    Loaded {name} (patch={config['patch_size']})")

    # ── Step 3: Generate base model predictions at 128³ ──────────────────
    print(f"\n  Generating base model predictions at {STACKING_TARGET_SIZE[0]}³...")

    for patient_dir in tqdm(patient_dirs, desc="Base predictions"):
        case_id = patient_dir.name
        cache_file = cache_dir / f'{case_id}.npz'

        if cache_file.exists():
            continue

        try:
            volume, _ = load_volume(patient_dir, STACKING_TARGET_SIZE)
            mask = load_mask(patient_dir, STACKING_TARGET_SIZE)
            if mask is None:
                mask = np.zeros(STACKING_TARGET_SIZE, dtype=np.float32)

            preds = {}
            for name, (model, patch_size) in models.items():
                prob_map = sliding_window_inference(
                    model, volume, patch_size, device, overlap=0.5
                )
                preds[name] = prob_map.astype(np.float16)

            np.savez_compressed(cache_file, mask=mask, **preds)

        except Exception as e:
            tqdm.write(f"  ERROR {case_id}: {e}")

    # Free base models
    del models
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 4: nnU-Net predictions ──────────────────────────────────────
    print(f"\n  Attempting nnU-Net inference...")

    nnunet_3d_dir = run_nnunet_inference(data_dir, output_dir, '3d_fullres', device)
    nnunet_2d_dir = run_nnunet_inference(data_dir, output_dir, '2d', device)

    # Add nnU-Net predictions to cache
    has_nnunet_3d = False
    has_nnunet_2d = False

    if nnunet_3d_dir or nnunet_2d_dir:
        print(f"  Adding nnU-Net predictions to stacking cache...")
        for cache_file in sorted(cache_dir.glob('*.npz')):
            case_id = cache_file.stem
            existing = np.load(cache_file)

            needs_update = False
            save_dict = {k: existing[k] for k in existing.files}

            if nnunet_3d_dir and 'nnunet' not in existing.files:
                pred = load_nnunet_prediction(nnunet_3d_dir, case_id, STACKING_TARGET_SIZE)
                if pred is not None:
                    save_dict['nnunet'] = pred.astype(np.float16)
                    needs_update = True
                    has_nnunet_3d = True

            if nnunet_2d_dir and 'nnunet_2d' not in existing.files:
                pred = load_nnunet_prediction(nnunet_2d_dir, case_id, STACKING_TARGET_SIZE)
                if pred is not None:
                    save_dict['nnunet_2d'] = pred.astype(np.float16)
                    needs_update = True
                    has_nnunet_2d = True

            if needs_update:
                np.savez_compressed(cache_file, **save_dict)

    # ── Step 5: Determine available model names for stacking ─────────────
    # Check first cache file to see what's available
    sample_cache = list(cache_dir.glob('*.npz'))[0]
    sample_data = np.load(sample_cache)
    available_models = [k for k in sample_data.files if k != 'mask']
    print(f"\n  Available models in cache: {available_models}")

    # Full stacking requires all 6 models (4 custom + nnunet + nnunet_2d)
    stacking_model_names = ['exp1_8patch', 'exp3_12patch_maxfn', 'improved_24patch',
                            'improved_36patch', 'nnunet', 'nnunet_2d']
    can_run_stacking = all(m in available_models for m in stacking_model_names)

    # If missing nnU-Net, try with just 4 custom models + variance + range = 6ch
    if not can_run_stacking:
        missing = [m for m in stacking_model_names if m not in available_models]
        print(f"  WARNING: Missing models for full stacking: {missing}")
        print(f"  Will use available models: {available_models}")
        stacking_model_names = [m for m in stacking_model_names if m in available_models]

    in_channels = len(stacking_model_names) + 2  # predictions + variance + range

    # ── Step 6: Load stacking classifier ─────────────────────────────────
    stacking_checkpoint = model_dir / 'stacking_v4_classifier.pth'
    stacking_model = None

    if stacking_checkpoint.exists():
        cp = torch.load(stacking_checkpoint, map_location=device, weights_only=False)
        expected_channels = cp['in_channels']

        if in_channels == expected_channels:
            stacking_model = StackingClassifier(in_channels=expected_channels).to(device)
            stacking_model.load_state_dict(cp['model_state_dict'])
            stacking_model.eval()
            print(f"  Loaded stacking v4 classifier (in_channels={expected_channels}, "
                  f"dice={cp.get('dice', 0):.4f})")
        else:
            print(f"  WARNING: Stacking model expects {expected_channels} channels "
                  f"but only {in_channels} available. Falling back to simple average.")
    else:
        print(f"  WARNING: Stacking checkpoint not found at {stacking_checkpoint}")

    # ── Step 7: Run stacking + top3 inference ────────────────────────────
    print(f"\n  Running inference on {len(patient_dirs)} cases...")

    for cache_file in tqdm(sorted(cache_dir.glob('*.npz')), desc="Stacking inference"):
        case_id = cache_file.stem

        # Skip if both predictions exist
        stacking_pred_path = output_dir / f'{case_id}_pred.nii.gz'
        top3_pred_path = output_dir / f'{case_id}_top3_pred.nii.gz'
        if stacking_pred_path.exists() and top3_pred_path.exists():
            continue

        try:
            data = np.load(cache_file)
            mask_128 = data['mask']

            # Get affine from original data for saving
            patient_dir = data_dir / case_id
            if patient_dir.exists():
                ref_nii = nib.load(str(patient_dir / 't1_gd.nii.gz'))
                affine = ref_nii.affine
            else:
                affine = np.eye(4)

            # ── Stacking prediction ──────────────────────────────────────
            if stacking_model is not None and not stacking_pred_path.exists():
                preds = np.stack([data[m].astype(np.float32) for m in stacking_model_names], axis=0)
                variance = preds.var(axis=0, keepdims=True)
                range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)
                features = np.concatenate([preds, variance, range_map], axis=0)

                prob_128 = sliding_window_inference(
                    stacking_model, features, STACKING_PATCH_SIZE, device,
                    overlap=STACKING_OVERLAP
                )

                # Upsample to 256³
                factors = [t / s for t, s in zip(EVAL_TARGET_SIZE, prob_128.shape)]
                prob_256 = zoom(prob_128, factors, order=1)

                # Save probability map
                nib.save(nib.Nifti1Image(prob_256, affine),
                         str(output_dir / f'{case_id}_stacking_prob.nii.gz'))

                # Threshold + postprocess
                seg = (prob_256 >= STACKING_THRESHOLD).astype(np.uint8)
                seg = postprocess(seg).astype(np.uint8)
                nib.save(nib.Nifti1Image(seg, affine), str(stacking_pred_path))

            # ── Top-3 average prediction ─────────────────────────────────
            if not top3_pred_path.exists():
                top3_preds = [data[m].astype(np.float32) for m in TOP3_MODELS
                              if m in data.files]
                if top3_preds:
                    prob_128 = np.mean(top3_preds, axis=0)

                    factors = [t / s for t, s in zip(EVAL_TARGET_SIZE, prob_128.shape)]
                    prob_256 = zoom(prob_128, factors, order=1)

                    nib.save(nib.Nifti1Image(prob_256, affine),
                             str(output_dir / f'{case_id}_top3_prob.nii.gz'))

                    seg = (prob_256 >= TOP3_THRESHOLD).astype(np.uint8)
                    seg = postprocess(seg).astype(np.uint8)
                    nib.save(nib.Nifti1Image(seg, affine), str(top3_pred_path))

        except Exception as e:
            tqdm.write(f"  ERROR {case_id}: {e}")

    # Free stacking model
    if stacking_model is not None:
        del stacking_model
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ─── Metrics (same as lesionwise_eval.py) ────────────────────────────────────

def voxel_dice(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin > 0)).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-8)


def voxel_sensitivity(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin > 0)).sum()
    return float(tp) / float(tp + fn + 1e-8)


def voxel_precision(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
    return float(tp) / float(tp + fp + 1e-8)


def lesionwise_f1(pred_bin, gt_bin, min_overlap=1):
    gt_labeled, n_gt = ndimage_label(gt_bin)
    pred_labeled, n_pred = ndimage_label(pred_bin)

    gt_detected = set()
    pred_matched = set()

    for i in range(1, n_gt + 1):
        gt_mask = (gt_labeled == i)
        overlapping_preds = pred_labeled[gt_mask]
        overlapping_preds = overlapping_preds[overlapping_preds > 0]
        if len(overlapping_preds) >= min_overlap:
            gt_detected.add(i)
            for p in np.unique(overlapping_preds):
                pred_matched.add(p)

    tp = len(gt_detected)
    fn = n_gt - tp
    fp = n_pred - len(pred_matched)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    gt_lesion_sizes = []
    gt_lesion_detected = []
    for i in range(1, n_gt + 1):
        size = int((gt_labeled == i).sum())
        gt_lesion_sizes.append(size)
        gt_lesion_detected.append(i in gt_detected)

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_gt_lesions': n_gt, 'n_pred_lesions': n_pred,
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'gt_lesion_sizes': gt_lesion_sizes,
        'gt_lesion_detected': gt_lesion_detected,
    }


def per_lesion_dice(pred_bin, gt_bin):
    gt_labeled, n_gt = ndimage_label(gt_bin)
    pred_labeled, n_pred = ndimage_label(pred_bin)

    if n_gt == 0:
        return {'mean_dice': 1.0 if n_pred == 0 else 0.0, 'per_lesion': [], 'n_lesions': 0}

    lesion_dices = []
    for i in range(1, n_gt + 1):
        gt_mask = (gt_labeled == i)
        gt_size = int(gt_mask.sum())

        overlapping = pred_labeled[gt_mask]
        unique_preds = np.unique(overlapping[overlapping > 0])

        if len(unique_preds) == 0:
            lesion_dices.append({'size': gt_size, 'dice': 0.0, 'matched': False})
            continue

        merged_pred = np.zeros_like(pred_bin, dtype=bool)
        for p in unique_preds:
            merged_pred |= (pred_labeled == p)

        tp = (gt_mask & merged_pred).sum()
        fp = (merged_pred & ~gt_mask).sum()
        fn = (gt_mask & ~merged_pred).sum()
        dice = float(2 * tp) / float(2 * tp + fp + fn + 1e-8)
        lesion_dices.append({'size': gt_size, 'dice': dice, 'matched': True})

    mean_dice = np.mean([l['dice'] for l in lesion_dices])
    return {'mean_dice': float(mean_dice), 'per_lesion': lesion_dices, 'n_lesions': n_gt}


def compute_surface_metrics(pred_bin, gt_bin):
    result = {
        'surface_dice_1': 0.0, 'surface_dice_2': 0.0, 'surface_dice_3': 0.0,
        'hausdorff_95': float('inf'),
        'relaxed_dice_1': 0.0, 'relaxed_dice_2': 0.0, 'relaxed_dice_3': 0.0,
    }

    pred_any = pred_bin.sum() > 0
    gt_any = gt_bin.sum() > 0

    if not pred_any and not gt_any:
        result.update({k: 1.0 for k in result})
        result['hausdorff_95'] = 0.0
        return result

    struct = generate_binary_structure(3, 1)

    gt_bool = gt_bin.astype(bool)
    gt_d1 = binary_dilation(gt_bool, structure=struct, iterations=1)
    gt_d2 = binary_dilation(gt_d1, structure=struct, iterations=1)
    gt_d3 = binary_dilation(gt_d2, structure=struct, iterations=1)
    result['relaxed_dice_1'] = voxel_dice(pred_bin, gt_d1.astype(np.float32))
    result['relaxed_dice_2'] = voxel_dice(pred_bin, gt_d2.astype(np.float32))
    result['relaxed_dice_3'] = voxel_dice(pred_bin, gt_d3.astype(np.float32))

    if not pred_any or not gt_any:
        return result

    pred_bool = pred_bin.astype(bool)
    pred_eroded = binary_erosion(pred_bool, structure=struct, iterations=1)
    pred_surface = pred_bool & ~pred_eroded

    gt_eroded = binary_erosion(gt_bool, structure=struct, iterations=1)
    gt_surface = gt_bool & ~gt_eroded

    n_pred_surface = int(pred_surface.sum())
    n_gt_surface = int(gt_surface.sum())

    if n_pred_surface == 0 or n_gt_surface == 0:
        return result

    gt_dist = distance_transform_edt(~gt_surface)
    pred_dist = distance_transform_edt(~pred_surface)

    pred_to_gt_dists = gt_dist[pred_surface]
    gt_to_pred_dists = pred_dist[gt_surface]

    for tol in [1.0, 2.0, 3.0]:
        pred_within = (pred_to_gt_dists <= tol).sum()
        gt_within = (gt_to_pred_dists <= tol).sum()
        nsd = float(pred_within + gt_within) / float(n_pred_surface + n_gt_surface + 1e-8)
        result[f'surface_dice_{int(tol)}'] = nsd

    all_distances = np.concatenate([pred_to_gt_dists, gt_to_pred_dists])
    result['hausdorff_95'] = float(np.percentile(all_distances, 95))

    return result


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_predictions(data_dir, pred_dir, pred_suffix='_pred.nii.gz'):
    """Evaluate predictions against ground truth at 256³."""
    data_dir = Path(data_dir)
    pred_dir = Path(pred_dir)

    case_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / 'seg.nii.gz').exists()
    ])

    case_metrics = []
    skipped = 0

    for case_dir in tqdm(case_dirs, desc="Evaluating"):
        case_id = case_dir.name

        pred_path = pred_dir / f'{case_id}{pred_suffix}'
        if not pred_path.exists():
            skipped += 1
            continue

        # Load at 256³ (already preprocessed to this size)
        gt = nib.load(str(case_dir / 'seg.nii.gz')).get_fdata()
        gt_bin = (gt > 0).astype(np.float32)

        pred = nib.load(str(pred_path)).get_fdata()
        pred_bin = (pred > 0).astype(np.float32)

        vd = voxel_dice(pred_bin, gt_bin)
        vs = voxel_sensitivity(pred_bin, gt_bin)
        vp = voxel_precision(pred_bin, gt_bin)
        lf1 = lesionwise_f1(pred_bin, gt_bin)
        pld = per_lesion_dice(pred_bin, gt_bin)
        surf = compute_surface_metrics(pred_bin, gt_bin)

        case_metrics.append({
            'case_id': case_id,
            'lesion_voxels': int(gt_bin.sum()),
            'pred_voxels': int(pred_bin.sum()),
            'n_gt_lesions': lf1['n_gt_lesions'],
            'n_pred_lesions': lf1['n_pred_lesions'],
            'voxel_dice': vd,
            'voxel_sensitivity': vs,
            'voxel_precision': vp,
            'lesion_f1': lf1['f1'],
            'lesion_recall': lf1['recall'],
            'lesion_precision': lf1['precision'],
            'lesion_tp': lf1['tp'],
            'lesion_fp': lf1['fp'],
            'lesion_fn': lf1['fn'],
            'per_lesion_dice': pld['mean_dice'],
            'surface_dice_1': surf['surface_dice_1'],
            'surface_dice_2': surf['surface_dice_2'],
            'surface_dice_3': surf['surface_dice_3'],
            'hausdorff_95': surf['hausdorff_95'],
            'relaxed_dice_1': surf['relaxed_dice_1'],
            'relaxed_dice_2': surf['relaxed_dice_2'],
            'relaxed_dice_3': surf['relaxed_dice_3'],
            'missed_lesion_sizes': [
                s for s, d in zip(lf1['gt_lesion_sizes'], lf1['gt_lesion_detected']) if not d
            ],
        })

    if skipped:
        print(f"  Skipped {skipped} cases (no prediction found)")

    return case_metrics


def build_summary(case_metrics, label, threshold):
    """Aggregate per-case metrics into summary statistics."""
    valid = [c for c in case_metrics if c['lesion_voxels'] > 0]
    n = len(valid)

    if n == 0:
        return {'n_cases': 0, 'n_total': len(case_metrics), 'threshold': threshold, 'label': label}

    summary = {'n_cases': n, 'n_total': len(case_metrics), 'threshold': threshold, 'label': label}

    metric_keys = [
        'voxel_dice', 'voxel_sensitivity', 'voxel_precision',
        'lesion_f1', 'lesion_recall', 'lesion_precision',
        'per_lesion_dice',
        'surface_dice_1', 'surface_dice_2', 'surface_dice_3',
        'relaxed_dice_1', 'relaxed_dice_2', 'relaxed_dice_3',
    ]

    for key in metric_keys:
        values = [c[key] for c in valid]
        summary[key] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)),
        }

    hd_values = [c['hausdorff_95'] for c in valid if c['hausdorff_95'] != float('inf')]
    summary['hausdorff_95'] = {
        'mean': float(np.mean(hd_values)) if hd_values else float('inf'),
        'median': float(np.median(hd_values)) if hd_values else float('inf'),
        'p95': float(np.percentile(hd_values, 95)) if hd_values else float('inf'),
        'valid_cases': len(hd_values),
        'inf_cases': n - len(hd_values),
    }

    total_tp = sum(c['lesion_tp'] for c in valid)
    total_fp = sum(c['lesion_fp'] for c in valid)
    total_fn = sum(c['lesion_fn'] for c in valid)
    total_gt = sum(c['n_gt_lesions'] for c in valid)
    total_pred = sum(c['n_pred_lesions'] for c in valid)
    summary['lesion_detection'] = {
        'total_gt_lesions': total_gt,
        'total_pred_lesions': total_pred,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'overall_precision': float(total_tp / (total_tp + total_fp + 1e-8)),
        'overall_recall': float(total_tp / (total_tp + total_fn + 1e-8)),
        'overall_f1': float(2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)),
    }

    all_missed = []
    for c in valid:
        all_missed.extend(c['missed_lesion_sizes'])

    if all_missed:
        summary['missed_lesions'] = {
            'total': len(all_missed),
            'mean_size': float(np.mean(all_missed)),
            'median_size': float(np.median(all_missed)),
            'min_size': int(min(all_missed)),
            'max_size': int(max(all_missed)),
            'by_size': {
                '<50 vox': sum(1 for s in all_missed if s < 50),
                '50-200': sum(1 for s in all_missed if 50 <= s < 200),
                '200-1000': sum(1 for s in all_missed if 200 <= s < 1000),
                '1000-5000': sum(1 for s in all_missed if 1000 <= s < 5000),
                '>5000': sum(1 for s in all_missed if s >= 5000),
            }
        }
    else:
        summary['missed_lesions'] = {'total': 0}

    buckets = [
        ('tiny (<100)', 0, 100),
        ('small (100-500)', 100, 500),
        ('medium (500-5k)', 500, 5000),
        ('large (5k-20k)', 5000, 20000),
        ('huge (>20k)', 20000, 1e9),
    ]
    summary['dice_by_size'] = {}
    for bucket_label, lo, hi in buckets:
        bucket_cases = [c for c in valid if lo <= c['lesion_voxels'] < hi]
        if bucket_cases:
            summary['dice_by_size'][bucket_label] = {
                'n': len(bucket_cases),
                'voxel_dice': float(np.mean([c['voxel_dice'] for c in bucket_cases])),
                'per_lesion_dice': float(np.mean([c['per_lesion_dice'] for c in bucket_cases])),
                'surface_dice_2': float(np.mean([c['surface_dice_2'] for c in bucket_cases])),
                'lesion_f1': float(np.mean([c['lesion_f1'] for c in bucket_cases])),
                'relaxed_dice_2': float(np.mean([c['relaxed_dice_2'] for c in bucket_cases])),
            }

    return summary


# ─── Printing ────────────────────────────────────────────────────────────────

def print_method_results(summary, case_metrics):
    """Print results for a single method."""
    label = summary['label']
    n = summary['n_cases']
    threshold = summary['threshold']

    metric_keys = [
        'voxel_dice', 'voxel_sensitivity', 'voxel_precision',
        'lesion_f1', 'lesion_recall', 'lesion_precision',
        'per_lesion_dice',
        'surface_dice_1', 'surface_dice_2', 'surface_dice_3',
        'relaxed_dice_1', 'relaxed_dice_2', 'relaxed_dice_3',
    ]

    print(f"\n  === {label} @ {threshold} ({n} cases) ===\n")
    print(f"  {'Metric':<25} {'Mean':<10} {'Median':<10} {'Std':<10} {'25th':<10} {'75th':<10}")
    print(f"  {'-'*75}")
    for key in metric_keys:
        if key not in summary:
            continue
        s = summary[key]
        print(f"  {key:<25} {s['mean']:<10.4f} {s['median']:<10.4f} "
              f"{s['std']:<10.4f} {s['p25']:<10.4f} {s['p75']:<10.4f}")

    h = summary.get('hausdorff_95', {})
    if h:
        print(f"  {'hausdorff_95':<25} {h['mean']:<10.2f} {h['median']:<10.2f} "
              f"{'':10} {'':10} (inf={h.get('inf_cases', 0)})")

    ld = summary.get('lesion_detection', {})
    if ld:
        print(f"\n  Lesion detection: {ld['total_tp']}/{ld['total_gt_lesions']} detected "
              f"(recall={ld['overall_recall']:.3f}), "
              f"{ld['total_fp']} FP (prec={ld['overall_precision']:.3f}), "
              f"F1={ld['overall_f1']:.3f}")

    ml = summary.get('missed_lesions', {})
    if ml.get('total', 0) > 0:
        print(f"  Missed {ml['total']} lesions: median={ml['median_size']:.0f} vox, "
              f"range=[{ml['min_size']}, {ml['max_size']}]")
        by = ml['by_size']
        print(f"    By size: <50={by['<50 vox']}, 50-200={by['50-200']}, "
              f"200-1k={by['200-1000']}, 1k-5k={by['1000-5000']}, >5k={by['>5000']}")

    buckets = ['tiny (<100)', 'small (100-500)', 'medium (500-5k)', 'large (5k-20k)', 'huge (>20k)']
    if summary.get('dice_by_size'):
        print(f"\n  Dice by lesion size:")
        print(f"  {'Bucket':<20} {'N':<5} {'Voxel':<8} {'PerLes':<8} {'SurfD2':<8} {'LesF1':<8} {'RelxD2':<8}")
        print(f"  {'-'*65}")
        for bucket_label in buckets:
            if bucket_label in summary['dice_by_size']:
                b = summary['dice_by_size'][bucket_label]
                print(f"  {bucket_label:<20} {b['n']:<5} {b['voxel_dice']:<8.4f} "
                      f"{b['per_lesion_dice']:<8.4f} {b['surface_dice_2']:<8.4f} "
                      f"{b['lesion_f1']:<8.4f} {b['relaxed_dice_2']:<8.4f}")


def print_comparison(stacking_summary, top3_summary, internal_path=None):
    """Print side-by-side comparison: stacking vs top3 vs internal."""
    metric_keys = [
        'voxel_dice', 'voxel_sensitivity', 'voxel_precision',
        'lesion_f1', 'lesion_recall', 'lesion_precision',
        'per_lesion_dice', 'surface_dice_2', 'relaxed_dice_2',
    ]

    # Load internal results if available
    int_summary = None
    if internal_path and Path(internal_path).exists():
        with open(internal_path) as f:
            internal = json.load(f)
        int_summary = internal.get('val_set', internal.get('all_cases', {})).get('summary', {})

    print(f"\n{'='*70}")
    print("  METHOD COMPARISON")
    print(f"{'='*70}")

    header = f"  {'Metric':<25} {'Stacking':<12} {'Top3-Avg':<12}"
    if int_summary:
        header += f" {'Internal':<12}"
    print(header)
    print(f"  {'-'*70}")

    for key in metric_keys:
        s_val = stacking_summary.get(key, {}).get('mean', 0) if stacking_summary.get('n_cases', 0) > 0 else 0
        t_val = top3_summary.get(key, {}).get('mean', 0) if top3_summary.get('n_cases', 0) > 0 else 0
        line = f"  {key:<25} {s_val:<12.4f} {t_val:<12.4f}"
        if int_summary and key in int_summary:
            i_val = int_summary[key]['mean'] if isinstance(int_summary[key], dict) else int_summary[key]
            line += f" {i_val:<12.4f}"
        print(line)

    # HD95
    s_hd = stacking_summary.get('hausdorff_95', {}).get('mean', float('inf'))
    t_hd = top3_summary.get('hausdorff_95', {}).get('mean', float('inf'))
    line = f"  {'hausdorff_95':<25} {s_hd:<12.2f} {t_hd:<12.2f}"
    if int_summary and 'hausdorff_95' in int_summary:
        i_hd = int_summary['hausdorff_95']['mean'] if isinstance(int_summary['hausdorff_95'], dict) else int_summary['hausdorff_95']
        line += f" {i_hd:<12.2f}"
    print(line)

    # Lesion detection
    print()
    for label, summary in [('Stacking', stacking_summary), ('Top3-Avg', top3_summary)]:
        ld = summary.get('lesion_detection', {})
        if ld:
            print(f"  {label}: F1={ld['overall_f1']:.3f} "
                  f"(Recall={ld['overall_recall']:.3f}, Prec={ld['overall_precision']:.3f})")
    if int_summary and 'lesion_detection' in int_summary:
        ild = int_summary['lesion_detection']
        print(f"  Internal: F1={ild['overall_f1']:.3f} "
              f"(Recall={ild['overall_recall']:.3f}, Prec={ild['overall_precision']:.3f})")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="External validation with stacking ensemble on PRETREAT dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline:
  python scripts/eval_external_validation.py

  # Evaluate existing predictions only:
  python scripts/eval_external_validation.py --eval-only

  # Custom threshold:
  python scripts/eval_external_validation.py --stacking-threshold 0.85
        """
    )
    parser.add_argument('--data-dir', type=str, default='data/external_validation/test',
                        help='Directory with preprocessed PRETREAT cases')
    parser.add_argument('--output-dir', type=str, default='results/external_validation',
                        help='Directory for predictions and results')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Model weights directory (default: model/)')
    parser.add_argument('--stacking-threshold', type=float, default=STACKING_THRESHOLD,
                        help=f'Stacking threshold (default: {STACKING_THRESHOLD})')
    parser.add_argument('--top3-threshold', type=float, default=TOP3_THRESHOLD,
                        help=f'Top-3 average threshold (default: {TOP3_THRESHOLD})')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip inference, only evaluate existing predictions')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    model_dir = Path(args.model_dir) if args.model_dir else ROOT / 'model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  BrainMetScan — External Validation (PRETREAT-METSTOBRAIN-MASKS)")
    print("  Method: Stacking Ensemble (v4) + Top-3 Average comparison")
    print("=" * 70)
    print(f"  Device:              {device}")
    if torch.cuda.is_available():
        print(f"  GPU:                 {torch.cuda.get_device_name(0)}")
    print(f"  Data:                {data_dir}")
    print(f"  Output:              {output_dir}")
    print(f"  Stacking threshold:  {args.stacking_threshold}")
    print(f"  Top-3 threshold:     {args.top3_threshold}")

    stacking_thresh = args.stacking_threshold
    top3_thresh = args.top3_threshold

    t0 = time.time()

    # ─── Step 1: Inference ───────────────────────────────────────────────
    if not args.eval_only:
        print(f"\n{'='*70}")
        print("  STEP 1: STACKING INFERENCE PIPELINE")
        print(f"{'='*70}")

        run_stacking_pipeline(data_dir, output_dir, model_dir, device)

    # ─── Step 2: Evaluate stacking ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 2: EVALUATION — STACKING ENSEMBLE")
    print(f"{'='*70}")

    stacking_metrics = evaluate_predictions(data_dir, output_dir, '_pred.nii.gz')
    stacking_summary = build_summary(stacking_metrics, "STACKING ENSEMBLE (ext)", stacking_thresh)

    if stacking_summary['n_cases'] > 0:
        print_method_results(stacking_summary, stacking_metrics)
    else:
        print("  No stacking predictions found — skipping")

    # ─── Step 3: Evaluate top-3 average ──────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 3: EVALUATION — TOP-3 AVERAGE")
    print(f"{'='*70}")

    top3_metrics = evaluate_predictions(data_dir, output_dir, '_top3_pred.nii.gz')
    top3_summary = build_summary(top3_metrics, "TOP-3 AVERAGE (ext)", top3_thresh)

    if top3_summary['n_cases'] > 0:
        print_method_results(top3_summary, top3_metrics)
    else:
        print("  No top-3 predictions found — skipping")

    # ─── Step 4: Comparison ──────────────────────────────────────────────
    internal_path = ROOT / 'model' / 'lesionwise_evaluation.json'
    print_comparison(stacking_summary, top3_summary, internal_path)

    # ─── Step 5: Save ────────────────────────────────────────────────────
    save_data = {
        'dataset': 'PRETREAT-METSTOBRAIN-MASKS',
        'source': 'https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/',
        'stacking': {
            'summary': stacking_summary,
            'per_case': [{k: v for k, v in c.items() if k != 'missed_lesion_sizes'}
                         for c in stacking_metrics],
        },
        'top3_average': {
            'summary': top3_summary,
            'per_case': [{k: v for k, v in c.items() if k != 'missed_lesion_sizes'}
                         for c in top3_metrics],
        },
    }

    results_path = output_dir / 'external_validation_results.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  COMPLETE — {elapsed/60:.1f} minutes total")
    print(f"  Results: {results_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
