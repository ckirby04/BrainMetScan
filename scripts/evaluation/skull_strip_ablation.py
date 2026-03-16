"""
Skull-stripping ablation study.

Tests whether the model is robust to non-skull-stripped input by:
1. Running on original skull-stripped data (baseline)
2. Adding synthetic tissue outside the brain to simulate non-skull-stripped MRI
3. Comparing Dice scores

Uses a subset of internal training cases (with ground truth masks).
"""

import sys
import json
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from scipy.ndimage import zoom, binary_dilation, generate_binary_structure, label as ndimage_label
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "segmentation"))

from unet import LightweightUNet3D
from stacking import sliding_window_inference

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = ROOT / "data" / "preprocessed_256" / "train"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "results" / "skull_strip_ablation"
SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']
TARGET_SIZE = (128, 128, 128)

TOP3_MODELS = {
    'exp3_12patch_maxfn': {'checkpoint': 'exp3_12patch_maxfn_finetuned.pth', 'patch_size': 12},
    'improved_24patch':   {'checkpoint': 'improved_24patch_finetuned.pth',   'patch_size': 24},
    'improved_36patch':   {'checkpoint': 'improved_36patch_finetuned.pth',   'patch_size': 36},
}
TOP3_THRESHOLD = 0.40
MAX_CASES = 30  # Enough for statistical comparison


# ============================================================================
# HELPERS
# ============================================================================

def load_volume(case_dir, target_size):
    """Load and resize 4-channel MRI volume."""
    channels = []
    for seq in SEQUENCES:
        path = case_dir / f"{seq}.nii.gz"
        nii = nib.load(str(path))
        data = np.asarray(nii.dataobj, dtype=np.float32)
        factors = [t / s for t, s in zip(target_size, data.shape)]
        data = zoom(data, factors, order=1)
        # Z-score normalize
        mean, std = data.mean(), data.std()
        if std > 0:
            data = (data - mean) / std
        channels.append(data)
    return np.stack(channels, axis=0)


def load_mask(case_dir, target_size):
    """Load and resize binary mask."""
    path = case_dir / "seg.nii.gz"
    nii = nib.load(str(path))
    mask = np.asarray(nii.dataobj, dtype=np.float32)
    mask = (mask > 0).astype(np.float32)
    # Resize to eval size (256^3)
    return mask


def create_brain_mask(volume):
    """
    Create approximate brain mask from skull-stripped data.
    Brain region = where any channel has non-zero signal.
    """
    # Use the mean across channels
    mean_vol = np.abs(volume).mean(axis=0)
    # Threshold at a low value (skull-stripped data has ~0 outside brain)
    mask = (mean_vol > 0.05).astype(np.float32)
    # Dilate slightly to get clean boundary
    struct = generate_binary_structure(3, 1)
    mask = binary_dilation(mask, struct, iterations=2).astype(np.float32)
    return mask


def add_synthetic_tissue(volume, brain_mask):
    """
    Add synthetic tissue outside the brain mask to simulate non-skull-stripped MRI.

    Strategy: fill non-brain regions with realistic-ish tissue signal:
    - Gaussian noise with mean/std derived from brain surface voxels
    - Smoothed to look like scalp/skull tissue rather than random noise
    """
    result = volume.copy()
    C = volume.shape[0]

    for c in range(C):
        chan = volume[c]
        # Get brain surface values (edge of brain mask)
        struct = generate_binary_structure(3, 1)
        eroded = binary_dilation(brain_mask == 0, struct, iterations=1).astype(bool) & (brain_mask > 0)
        if eroded.sum() > 0:
            surface_vals = chan[eroded]
            surf_mean = surface_vals.mean()
            surf_std = max(surface_vals.std(), 0.1)
        else:
            surf_mean = 0.0
            surf_std = 0.3

        # Generate tissue-like signal outside brain
        non_brain = brain_mask == 0
        noise = np.random.normal(surf_mean * 0.6, surf_std * 0.8, size=chan.shape).astype(np.float32)

        # Smooth the noise to make it look more tissue-like
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(noise, sigma=2.0)

        result[c][non_brain] = noise[non_brain]

    return result


def dice_score(pred, gt):
    """Compute Dice coefficient."""
    intersection = (pred * gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return (2.0 * intersection / total).item() if isinstance(intersection, torch.Tensor) else 2.0 * intersection / total


def run_top3_inference(volume, models, device):
    """Run top-3 ensemble inference on a volume."""
    probs = []
    for name, (model, patch_size) in models.items():
        prob = sliding_window_inference(model, volume, patch_size, device, overlap=0.25)
        probs.append(prob)
    avg_prob = np.mean(probs, axis=0)
    return avg_prob


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load models
    print("\nLoading top-3 models...")
    models = {}
    for name, info in TOP3_MODELS.items():
        ckpt_path = MODEL_DIR / info['checkpoint']
        model = LightweightUNet3D(
            in_channels=4, out_channels=1,
            base_channels=20, use_attention=True, use_residual=True,
        ).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        model.eval()
        models[name] = (model, info['patch_size'])
        print(f"  Loaded {name} (patch={info['patch_size']})")

    # Find cases with tumors
    cases = sorted([d for d in DATA_DIR.iterdir()
                    if d.is_dir() and (d / 'seg.nii.gz').exists()])

    # Filter to cases that actually have tumors
    tumor_cases = []
    for case_dir in cases:
        mask = nib.load(str(case_dir / 'seg.nii.gz')).get_fdata()
        if mask.sum() > 0:
            tumor_cases.append(case_dir)
        if len(tumor_cases) >= MAX_CASES:
            break

    print(f"\nUsing {len(tumor_cases)} cases with tumors")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_stripped = []
    results_unstripped = []

    for case_dir in tqdm(tumor_cases, desc="Ablation"):
        case_id = case_dir.name

        # Load volume at model resolution (128^3)
        volume = load_volume(case_dir, TARGET_SIZE)

        # Load GT mask at 256^3 then resize to 128^3 for comparison
        gt_mask_256 = load_mask(case_dir, (256, 256, 256))
        factors = [t / s for t, s in zip(TARGET_SIZE, gt_mask_256.shape)]
        gt_mask = zoom(gt_mask_256, factors, order=0)
        gt_mask = (gt_mask > 0.5).astype(np.float32)

        # --- Condition 1: Skull-stripped (original) ---
        prob_stripped = run_top3_inference(volume, models, device)
        pred_stripped = (prob_stripped > TOP3_THRESHOLD).astype(np.float32)
        dice_stripped = dice_score(pred_stripped, gt_mask)

        # --- Condition 2: Non-skull-stripped (synthetic tissue added) ---
        brain_mask = create_brain_mask(volume)
        volume_unstripped = add_synthetic_tissue(volume, brain_mask)

        prob_unstripped = run_top3_inference(volume_unstripped, models, device)
        pred_unstripped = (prob_unstripped > TOP3_THRESHOLD).astype(np.float32)
        dice_unstripped = dice_score(pred_unstripped, gt_mask)

        results_stripped.append({'case': case_id, 'dice': float(dice_stripped)})
        results_unstripped.append({'case': case_id, 'dice': float(dice_unstripped)})

        tqdm.write(f"  {case_id}: stripped={dice_stripped:.3f}  unstripped={dice_unstripped:.3f}  "
                   f"delta={dice_unstripped - dice_stripped:+.3f}")

    # Summary
    dices_s = [r['dice'] for r in results_stripped]
    dices_u = [r['dice'] for r in results_unstripped]
    deltas = [u - s for s, u in zip(dices_s, dices_u)]

    print(f"\n{'='*60}")
    print(f"  SKULL-STRIPPING ABLATION RESULTS ({len(tumor_cases)} cases)")
    print(f"{'='*60}")
    print(f"  Skull-stripped (original):     Dice = {np.mean(dices_s):.4f} +/- {np.std(dices_s):.4f}")
    print(f"  Non-skull-stripped (synthetic): Dice = {np.mean(dices_u):.4f} +/- {np.std(dices_u):.4f}")
    print(f"  Mean delta:                    {np.mean(deltas):+.4f}")
    print(f"  Median delta:                  {np.median(deltas):+.4f}")
    print(f"  Cases where unstripped worse:  {sum(1 for d in deltas if d < -0.01)}/{len(deltas)}")
    print(f"  Cases where unstripped better: {sum(1 for d in deltas if d > 0.01)}/{len(deltas)}")
    print(f"  Cases within +/-0.01:          {sum(1 for d in deltas if abs(d) <= 0.01)}/{len(deltas)}")

    # Statistical test
    from scipy.stats import wilcoxon
    if len(deltas) >= 10:
        stat, pval = wilcoxon(dices_s, dices_u)
        print(f"\n  Wilcoxon signed-rank test: p = {pval:.4f}")
        if pval < 0.05:
            print(f"  --> Significant difference (p < 0.05)")
        else:
            print(f"  --> No significant difference (p >= 0.05)")

    # Save results
    report = {
        'n_cases': len(tumor_cases),
        'threshold': TOP3_THRESHOLD,
        'target_size': list(TARGET_SIZE),
        'skull_stripped': {
            'mean_dice': float(np.mean(dices_s)),
            'std_dice': float(np.std(dices_s)),
            'median_dice': float(np.median(dices_s)),
        },
        'non_skull_stripped': {
            'mean_dice': float(np.mean(dices_u)),
            'std_dice': float(np.std(dices_u)),
            'median_dice': float(np.median(dices_u)),
        },
        'delta': {
            'mean': float(np.mean(deltas)),
            'median': float(np.median(deltas)),
            'cases_worse': sum(1 for d in deltas if d < -0.01),
            'cases_better': sum(1 for d in deltas if d > 0.01),
            'cases_neutral': sum(1 for d in deltas if abs(d) <= 0.01),
        },
        'per_case': [
            {'case': s['case'], 'stripped': s['dice'], 'unstripped': u['dice'], 'delta': u['dice'] - s['dice']}
            for s, u in zip(results_stripped, results_unstripped)
        ],
    }

    report_path = OUTPUT_DIR / "skull_strip_ablation_results.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved: {report_path}")


if __name__ == '__main__':
    main()
