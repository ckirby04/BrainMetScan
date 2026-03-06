"""
Full-volume validation for v2 retrained models.
Runs sliding window inference at native 256³ on the validation set.
Reports per-case Dice and flags previously zero-dice cases.

Usage:
    python scripts/validate_fullvol.py
    python scripts/validate_fullvol.py --model exp1_8patch
    python scripts/validate_fullvol.py --threshold 0.3
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.amp import autocast
from tqdm import tqdm
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.unet import LightweightUNet3D

# Previously zero-dice cases (from stacking v3 analysis)
PREV_ZERO_DICE = [
    'UCSF_100238A', 'UCSF_100414B', 'UCSF_100238D', 'UCSF_100318A',
    'UCSF_100209A', 'BMS_Mets_238', 'UCSF_100364A', 'UCSF_100108D',
    'UCSF_100224A', 'UCSF_100317A'
]

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


def detect_sequences(data_dir):
    data_path = Path(data_dir)
    case_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    test_case = case_dirs[0]
    if (test_case / 't2.nii.gz').exists() or (test_case / 't2.npz').exists():
        return ['t1_pre', 't1_gd', 'flair', 't2']
    elif (test_case / 'bravo.nii.gz').exists() or (test_case / 'bravo.npz').exists():
        return ['t1_pre', 't1_gd', 'flair', 'bravo']
    else:
        raise RuntimeError(f"Cannot detect sequences in {test_case}")


def load_volume(case_dir, sequences):
    """Load a full case as (C, H, W, D) float32 numpy array, no resize."""
    images = []
    for seq in sequences:
        for ext in ['.npz', '.npy', '.nii.gz']:
            p = case_dir / f"{seq}{ext}"
            if p.exists():
                if ext == '.npz':
                    img = np.load(str(p))['data'].astype(np.float32)
                elif ext == '.npy':
                    img = np.load(str(p)).astype(np.float32)
                else:
                    img = np.asarray(nib.load(str(p)).dataobj, dtype=np.float32)
                break
        else:
            raise FileNotFoundError(f"Missing {seq} for {case_dir.name}")
        # Z-score normalize
        mean, std = img.mean(), img.std()
        if std > 0:
            img = (img - mean) / std
        images.append(img)
    return np.stack(images, axis=0)


def load_mask(case_dir):
    """Load segmentation mask as binary (H, W, D) float32."""
    for ext in ['seg.npz', 'seg.npy', 'seg.nii.gz']:
        p = case_dir / ext
        if p.exists():
            if ext.endswith('.npz'):
                m = np.load(str(p))['data']
            elif ext.endswith('.npy'):
                m = np.load(str(p))
            else:
                m = np.asarray(nib.load(str(p)).dataobj, dtype=np.float32)
            return (m > 0).astype(np.float32)
    return None


def sliding_window_inference(model, volume, patch_size, device, overlap=0.5):
    """
    Run sliding window inference on a full volume.

    Args:
        model: trained model
        volume: (C, H, W, D) numpy array
        patch_size: int, cubic patch size
        device: torch device
        overlap: fraction of overlap between patches

    Returns:
        (H, W, D) probability map
    """
    C, H, W, D = volume.shape
    ps = patch_size
    stride = max(1, int(ps * (1 - overlap)))

    # Output accumulator
    prob_sum = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)

    # Generate patch positions
    h_starts = list(range(0, max(H - ps, 0) + 1, stride))
    w_starts = list(range(0, max(W - ps, 0) + 1, stride))
    d_starts = list(range(0, max(D - ps, 0) + 1, stride))

    # Ensure we cover the edges
    if len(h_starts) == 0 or h_starts[-1] + ps < H:
        h_starts.append(max(0, H - ps))
    if len(w_starts) == 0 or w_starts[-1] + ps < W:
        w_starts.append(max(0, W - ps))
    if len(d_starts) == 0 or d_starts[-1] + ps < D:
        d_starts.append(max(0, D - ps))

    # Deduplicate
    h_starts = sorted(set(h_starts))
    w_starts = sorted(set(w_starts))
    d_starts = sorted(set(d_starts))

    total_patches = len(h_starts) * len(w_starts) * len(d_starts)

    vol_tensor = torch.from_numpy(volume).float()

    patch_count = 0
    for h in h_starts:
        for w in w_starts:
            for d in d_starts:
                patch = vol_tensor[:, h:h+ps, w:w+ps, d:d+ps].unsqueeze(0).to(device)

                with autocast('cuda'):
                    output = model(patch)
                    if isinstance(output, tuple):
                        output = output[0]  # main output only

                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                prob_sum[h:h+ps, w:w+ps, d:d+ps] += prob
                count[h:h+ps, w:w+ps, d:d+ps] += 1
                patch_count += 1

    # Average overlapping regions
    count = np.maximum(count, 1)
    return prob_sum / count


def compute_dice(pred_binary, target):
    """Compute Dice score between binary arrays."""
    tp = ((pred_binary == 1) & (target == 1)).sum()
    fp = ((pred_binary == 1) & (target == 0)).sum()
    fn = ((pred_binary == 0) & (target == 1)).sum()
    if tp + fp + fn == 0:
        return 1.0  # Both empty = perfect
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-8)


def get_val_cases(data_dir):
    """Get validation case directories using same split as training."""
    data_path = Path(data_dir)
    valid_prefixes = ('Mets_', 'UCSF_', 'BraTS_', 'Yale_', 'BMS_')
    cases = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith(valid_prefixes)])

    # Same split as retrain_256.py
    n_val = int(len(cases) * 0.15)
    n_train = len(cases) - n_val
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(cases), generator=generator).tolist()
    val_indices = indices[n_train:]

    val_cases = [cases[i] for i in val_indices]
    print(f"Total cases: {len(cases)}, Val cases: {len(val_cases)}")
    return val_cases


def validate_model(model_name, config, val_cases, sequences, device, threshold_override=None):
    """Run full-volume validation for one model."""
    model_dir = ROOT / 'model'
    checkpoint_path = model_dir / f'v2_{model_name}_finetuned.pth'

    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None

    # Load model
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config['base_channels'],
        use_attention=config['use_attention'],
        use_residual=config['use_residual'],
        deep_supervision=True
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    train_dice = checkpoint.get('dice', 'N/A')
    print(f"\n  Loaded {checkpoint_path.name} (train best dice: {train_dice})")

    patch_size = config['patch_size']
    threshold = threshold_override if threshold_override is not None else config['optimal_threshold']
    print(f"  Patch size: {patch_size}, Threshold: {threshold}, Overlap: 0.5")

    results = {}
    all_dice = []

    for case_dir in tqdm(val_cases, desc=f"  {model_name}"):
        case_id = case_dir.name

        # Load volume and mask
        volume = load_volume(case_dir, sequences)
        mask = load_mask(case_dir)
        if mask is None:
            continue

        # Sliding window inference
        with torch.no_grad():
            prob_map = sliding_window_inference(model, volume, patch_size, device, overlap=0.5)

        # Threshold
        pred_binary = (prob_map > threshold).astype(np.float32)

        # Compute metrics
        dice = compute_dice(pred_binary, mask)
        tp = ((pred_binary == 1) & (mask == 1)).sum()
        fn = ((pred_binary == 0) & (mask == 1)).sum()
        fp = ((pred_binary == 1) & (mask == 0)).sum()
        sensitivity = float(tp) / float(tp + fn + 1e-8)
        precision = float(tp) / float(tp + fp + 1e-8)
        lesion_size = int(mask.sum())

        results[case_id] = {
            'dice': dice,
            'sensitivity': sensitivity,
            'precision': precision,
            'lesion_voxels': lesion_size,
            'pred_voxels': int(pred_binary.sum()),
            'was_zero_dice': case_id in PREV_ZERO_DICE,
        }
        all_dice.append(dice)

    return results, all_dice


def main():
    parser = argparse.ArgumentParser(description="Full-volume validation for v2 models")
    parser.add_argument('--model', type=str, default=None,
                        help="Validate specific model")
    parser.add_argument('--threshold', type=float, default=None,
                        help="Override threshold (default: use model's optimal)")
    parser.add_argument('--data-dir', type=str, default='data/preprocessed_256/train',
                        help="Data directory")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = str(ROOT / args.data_dir)
    sequences = detect_sequences(data_dir)
    val_cases = get_val_cases(data_dir)

    # Select models
    if args.model:
        models = {args.model: ENSEMBLE_CONFIGS[args.model]}
    else:
        # Only validate models that have finished training
        models = {}
        for name, cfg in ENSEMBLE_CONFIGS.items():
            done = (ROOT / 'model' / 'training_states' / f'v2_{name}_finetune_DONE').exists()
            ckpt = (ROOT / 'model' / f'v2_{name}_finetuned.pth').exists()
            if done and ckpt:
                models[name] = cfg

    if not models:
        print("No finished models found!")
        return

    print(f"Models to validate: {list(models.keys())}")

    all_results = {}

    for model_name, config in models.items():
        print(f"\n{'='*60}")
        print(f"FULL-VOLUME VALIDATION: {model_name}")
        print(f"{'='*60}")

        results, all_dice = validate_model(
            model_name, config, val_cases, sequences, device, args.threshold
        )

        if results is None:
            continue

        all_results[model_name] = results

        # Summary stats
        dice_arr = np.array(all_dice)
        print(f"\n  --- {model_name} Summary ---")
        print(f"  Cases: {len(all_dice)}")
        print(f"  Mean Dice:   {dice_arr.mean():.4f} +/- {dice_arr.std():.4f}")
        print(f"  Median Dice: {np.median(dice_arr):.4f}")
        print(f"  Min:  {dice_arr.min():.4f}   Max: {dice_arr.max():.4f}")
        print(f"  25th: {np.percentile(dice_arr, 25):.4f}   75th: {np.percentile(dice_arr, 75):.4f}")

        # Zero-dice count
        n_zero = sum(1 for d in all_dice if d == 0)
        print(f"  Zero-dice cases: {n_zero}")

        # Previously zero-dice cases
        print(f"\n  --- Previously Zero-Dice Cases ---")
        for case_id in PREV_ZERO_DICE:
            if case_id in results:
                r = results[case_id]
                status = "FIXED" if r['dice'] > 0 else "STILL ZERO"
                print(f"  {case_id}: dice={r['dice']:.4f}  sens={r['sensitivity']:.4f}  "
                      f"lesion={r['lesion_voxels']}  pred={r['pred_voxels']}  [{status}]")
            else:
                print(f"  {case_id}: not in val set")

        # Bottom 10
        sorted_results = sorted(results.items(), key=lambda x: x[1]['dice'])
        print(f"\n  --- Bottom 10 Cases ---")
        for case_id, r in sorted_results[:10]:
            flag = " *PREV_ZERO*" if r['was_zero_dice'] else ""
            print(f"  {case_id}: dice={r['dice']:.4f}  sens={r['sensitivity']:.4f}  "
                  f"lesion={r['lesion_voxels']}  pred={r['pred_voxels']}{flag}")

        # Top 10
        print(f"\n  --- Top 10 Cases ---")
        for case_id, r in sorted_results[-10:]:
            print(f"  {case_id}: dice={r['dice']:.4f}  sens={r['sensitivity']:.4f}  "
                  f"lesion={r['lesion_voxels']}  pred={r['pred_voxels']}")

        # Free GPU memory
        torch.cuda.empty_cache()

    # Save results
    output_path = ROOT / 'model' / 'v2_fullvol_validation.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
