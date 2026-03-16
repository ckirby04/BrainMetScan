"""
Run brain metastasis segmentation inference on new patient data.

Loads the top-3 custom ensemble models, runs sliding window inference on each,
averages their probability maps, and saves a binary NIfTI segmentation mask.

Usage:
    python scripts/run_inference.py --input path/to/patient_dir --output path/to/output_dir
    python scripts/run_inference.py --input path/to/patient_dir --output out/ --threshold 0.35
    python scripts/run_inference.py --input path/to/patients/ --output out/  # batch mode

Each patient directory must contain:
    t1_pre.nii.gz, t1_gd.nii.gz, flair.nii.gz, t2.nii.gz

Output:
    {output_dir}/{patient_id}_seg.nii.gz  — binary segmentation mask
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "segmentation"))

from unet import LightweightUNet3D

# Models to ensemble (top 3 performers)
ENSEMBLE_MODELS = [
    'exp3_12patch_maxfn_finetuned.pth',
    'improved_24patch_finetuned.pth',
    'improved_36patch_finetuned.pth',
]

SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']


def load_models(model_dir, device):
    """Load ensemble models from model directory."""
    models = []
    for name in ENSEMBLE_MODELS:
        path = model_dir / name
        if not path.exists():
            print(f"  WARNING: Model not found: {path}")
            continue

        model = LightweightUNet3D(
            in_channels=4, out_channels=1,
            base_channels=20, depth=3,
            use_attention=True, use_residual=True,
        ).to(device)

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        models.append(model)
        print(f"  Loaded {name}")

    if not models:
        print("ERROR: No models found. Download weights first.")
        print("  See: https://github.com/ckirby04/BrainMetScan/releases")
        sys.exit(1)

    return models


def load_patient(patient_dir):
    """Load and normalize 4-channel MRI from a patient directory."""
    patient_dir = Path(patient_dir)
    images = []
    affine = None

    for seq in SEQUENCES:
        nii_path = patient_dir / f'{seq}.nii.gz'
        if not nii_path.exists():
            raise FileNotFoundError(
                f"Missing {seq}.nii.gz in {patient_dir}. "
                f"Expected files: {', '.join(s + '.nii.gz' for s in SEQUENCES)}"
            )

        nii = nib.load(str(nii_path))
        img = np.asarray(nii.dataobj, dtype=np.float32)

        if affine is None:
            affine = nii.affine

        # Z-score normalize
        mean, std = img.mean(), img.std()
        if std > 0:
            img = (img - mean) / std

        images.append(img)

    volume = np.stack(images, axis=0)  # (4, H, W, D)
    return volume, affine


@torch.no_grad()
def sliding_window_predict(models, volume, device, patch_size=96, overlap=0.5):
    """Run ensemble sliding window inference on a 4-channel volume."""
    C, H, W, D = volume.shape
    stride = int(patch_size * (1 - overlap))

    prob_sum = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)

    h_starts = list(range(0, max(H - patch_size, 0) + 1, stride))
    w_starts = list(range(0, max(W - patch_size, 0) + 1, stride))
    d_starts = list(range(0, max(D - patch_size, 0) + 1, stride))

    # Ensure we cover the edges
    if h_starts[-1] + patch_size < H:
        h_starts.append(H - patch_size)
    if w_starts[-1] + patch_size < W:
        w_starts.append(W - patch_size)
    if d_starts[-1] + patch_size < D:
        d_starts.append(D - patch_size)

    total = len(h_starts) * len(w_starts) * len(d_starts)
    done = 0

    for hs in h_starts:
        for ws in w_starts:
            for ds in d_starts:
                patch = volume[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size]

                # Pad if patch is smaller than expected (edge cases)
                ph, pw, pd = patch.shape[1:]
                if ph < patch_size or pw < patch_size or pd < patch_size:
                    padded = np.zeros((C, patch_size, patch_size, patch_size), dtype=np.float32)
                    padded[:, :ph, :pw, :pd] = patch
                    patch = padded

                x = torch.from_numpy(patch).unsqueeze(0).to(device)  # (1, 4, P, P, P)

                # Average across ensemble models
                ensemble_prob = torch.zeros(1, 1, patch_size, patch_size, patch_size, device=device)
                for model in models:
                    out = model(x)
                    if isinstance(out, tuple):
                        out = out[0]  # deep supervision returns (main, aux)
                    ensemble_prob += torch.sigmoid(out)
                ensemble_prob /= len(models)

                prob = ensemble_prob[0, 0, :ph, :pw, :pd].cpu().numpy()
                prob_sum[hs:hs+ph, ws:ws+pw, ds:ds+pd] += prob
                count[hs:hs+ph, ws:ws+pw, ds:ds+pd] += 1

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"    [{done}/{total}] patches processed", end='\r')

    print()
    # Average overlapping regions
    count = np.maximum(count, 1)
    return prob_sum / count


def process_patient(patient_dir, output_dir, models, device, threshold, patch_size):
    """Process a single patient: load, predict, save."""
    patient_id = Path(patient_dir).name
    output_path = Path(output_dir) / f'{patient_id}_seg.nii.gz'

    if output_path.exists():
        print(f"  {patient_id}: already exists, skipping")
        return

    print(f"  {patient_id}: loading...")
    volume, affine = load_patient(patient_dir)
    print(f"  {patient_id}: volume shape {volume.shape[1:]}, running inference...")

    t0 = time.time()
    prob_map = sliding_window_predict(models, volume, device, patch_size=patch_size)
    elapsed = time.time() - t0

    # Threshold to binary
    seg = (prob_map >= threshold).astype(np.uint8)

    # Save
    nii_out = nib.Nifti1Image(seg, affine)
    nib.save(nii_out, str(output_path))

    n_voxels = seg.sum()
    print(f"  {patient_id}: {n_voxels} voxels segmented in {elapsed:.1f}s -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run brain metastasis segmentation on patient MRI data")
    parser.add_argument('--input', required=True,
                        help="Patient directory or parent directory containing multiple patients")
    parser.add_argument('--output', required=True,
                        help="Output directory for segmentation masks")
    parser.add_argument('--threshold', type=float, default=0.4,
                        help="Probability threshold for binary segmentation (default: 0.4)")
    parser.add_argument('--patch-size', type=int, default=96,
                        help="Sliding window patch size (default: 96)")
    parser.add_argument('--model-dir', type=str, default=None,
                        help="Model weights directory (default: model/)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = Path(args.model_dir) if args.model_dir else ROOT / 'model'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BrainMetScan — Ensemble Inference")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Patch size: {args.patch_size}")

    # Load models
    print(f"\n  Loading ensemble models from {model_dir}...")
    models = load_models(model_dir, device)
    print(f"  Loaded {len(models)} models")

    # Determine patient directories
    input_path = Path(args.input)
    if (input_path / f'{SEQUENCES[0]}.nii.gz').exists():
        # Single patient directory
        patients = [input_path]
    else:
        # Parent directory containing multiple patients
        patients = sorted([
            d for d in input_path.iterdir()
            if d.is_dir() and (d / f'{SEQUENCES[0]}.nii.gz').exists()
        ])

    if not patients:
        print(f"\nERROR: No patient directories found in {input_path}")
        print(f"Expected directories containing: {', '.join(s + '.nii.gz' for s in SEQUENCES)}")
        sys.exit(1)

    print(f"\n  Processing {len(patients)} patient(s)...")
    t0 = time.time()

    for patient_dir in patients:
        process_patient(patient_dir, output_dir, models, device,
                        args.threshold, args.patch_size)

    elapsed = time.time() - t0
    print(f"\n  Done. {len(patients)} patient(s) processed in {elapsed:.0f}s")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
