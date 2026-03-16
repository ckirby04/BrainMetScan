"""
Preprocess all MRI volumes to 256³ resolution.

This script resizes all training and test data once, so training doesn't
need to resize on-the-fly (which causes memory issues with DataLoader workers).

Usage:
    python scripts/preprocess_256.py

Output:
    data/preprocessed_256/train/  - Preprocessed training data
    data/preprocessed_256/test/   - Preprocessed test data
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import argparse

# Target resolution
TARGET_SIZE = (256, 256, 256)

# Sequences to process
SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']


def load_nifti(path: Path) -> tuple:
    """Load NIfTI file and return data + affine"""
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj, dtype=np.float32)
    return data, nii.affine


def resize_volume(img: np.ndarray, target_size: tuple, is_mask: bool = False) -> np.ndarray:
    """Resize 3D volume to target size"""
    current_shape = img.shape
    zoom_factors = [t / c for t, c in zip(target_size, current_shape)]
    order = 0 if is_mask else 1  # Nearest for masks, linear for images
    return zoom(img, zoom_factors, order=order, mode='nearest')


def normalize(img: np.ndarray) -> np.ndarray:
    """Z-score normalization (memory-efficient)"""
    n = img.size
    mean = img.sum() / n

    # Compute std without large temporary arrays
    sum_sq = 0.0
    chunk_size = 1024 * 1024
    flat = img.ravel()
    for i in range(0, n, chunk_size):
        chunk = flat[i:i + chunk_size]
        sum_sq += (chunk * chunk).sum()
    variance = (sum_sq / n) - (mean * mean)
    std = np.sqrt(max(variance, 0))

    if std > 0:
        img = (img - mean) / std
    return img


def process_case(case_dir: Path, output_dir: Path, sequences: list, has_mask: bool = True):
    """Process a single case: resize all sequences (normalization done during training)"""
    case_id = case_dir.name
    output_case_dir = output_dir / case_id
    output_case_dir.mkdir(parents=True, exist_ok=True)

    # Process each sequence
    for seq in sequences:
        input_path = case_dir / f"{seq}.nii.gz"
        output_path = output_case_dir / f"{seq}.nii.gz"

        if not input_path.exists():
            print(f"  Warning: Missing {seq} for {case_id}")
            continue

        if output_path.exists():
            continue  # Skip if already processed

        # Load and resize only (normalization happens during training for consistency)
        data, affine = load_nifti(input_path)
        data = resize_volume(data, TARGET_SIZE, is_mask=False)
        # Note: NOT normalizing here - dataset handles normalization with memory-efficient method

        # Save as NIfTI (preserves compatibility with existing code)
        # Update affine to reflect new voxel size
        new_affine = affine.copy()
        for i in range(3):
            scale = data.shape[i] / (affine[i, i] if affine[i, i] != 0 else 1)
            if scale != 0:
                new_affine[i, i] = affine[i, i] * (data.shape[i] / TARGET_SIZE[i]) if TARGET_SIZE[i] != 0 else affine[i, i]

        nii_out = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(nii_out, str(output_path))

    # Process segmentation mask if available
    if has_mask:
        mask_input = case_dir / "seg.nii.gz"
        mask_output = output_case_dir / "seg.nii.gz"

        if mask_input.exists() and not mask_output.exists():
            data, affine = load_nifti(mask_input)
            data = resize_volume(data, TARGET_SIZE, is_mask=True)
            data = (data > 0).astype(np.float32)  # Ensure binary

            nii_out = nib.Nifti1Image(data, affine)
            nib.save(nii_out, str(mask_output))


def main():
    parser = argparse.ArgumentParser(description='Preprocess MRI data to 256³')
    parser.add_argument('--input-dir', type=str, default='../Superset/full',
                        help='Input data directory containing train/ and test/')
    parser.add_argument('--output-dir', type=str, default='data/preprocessed_256',
                        help='Output directory for preprocessed data')
    parser.add_argument('--target-size', type=int, default=256,
                        help='Target resolution (default: 256)')
    args = parser.parse_args()

    global TARGET_SIZE
    TARGET_SIZE = (args.target_size, args.target_size, args.target_size)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Preprocessing MRI data to {TARGET_SIZE[0]}³ resolution")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Process training data
    train_input = input_dir / 'train'
    train_output = output_dir / 'train'

    if train_input.exists():
        valid_prefixes = ('Mets_', 'UCSF_', 'BraTS_', 'Yale_', 'BMS_', 'PRETREAT_')
        train_cases = sorted([d for d in train_input.iterdir()
                              if d.is_dir() and d.name.startswith(valid_prefixes)])

        print(f"Processing {len(train_cases)} training cases...")
        for case_dir in tqdm(train_cases, desc="Training"):
            process_case(case_dir, train_output, SEQUENCES, has_mask=True)

    # Process test data
    test_input = input_dir / 'test'
    test_output = output_dir / 'test'

    if test_input.exists():
        valid_prefixes = ('Mets_', 'UCSF_', 'BraTS_', 'Yale_', 'BMS_', 'PRETREAT_')
        test_cases = sorted([d for d in test_input.iterdir()
                             if d.is_dir() and d.name.startswith(valid_prefixes)])

        print(f"\nProcessing {len(test_cases)} test cases...")
        for case_dir in tqdm(test_cases, desc="Test"):
            process_case(case_dir, test_output, SEQUENCES, has_mask=True)

    print("\nPreprocessing complete!")
    print(f"Preprocessed data saved to: {output_dir}")
    print()
    print("To use preprocessed data, update your config:")
    print(f'  data_dir: "{output_dir}/train"')
    print(f'  test_dir: "{output_dir}/test"')
    print("  target_size: null  # Already resized, no need to resize again")


if __name__ == '__main__':
    main()
