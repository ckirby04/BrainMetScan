"""
Convert preprocessed NIfTI files to numpy format for faster loading.

Usage:
    python scripts/convert_to_numpy.py

This converts data/preprocessed_256/ to data/preprocessed_256_npy/
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import os


def convert_case(args):
    """Convert a single case from NIfTI to compressed numpy."""
    case_dir, output_dir, is_train = args
    case_name = case_dir.name
    out_case_dir = output_dir / case_name
    out_case_dir.mkdir(parents=True, exist_ok=True)

    # Files to convert
    if is_train:
        files = ['t1_pre', 't1_gd', 'flair', 't2', 'seg']
    else:
        files = ['t1_pre', 't1_gd', 'flair', 't2']

    for fname in files:
        nii_path = case_dir / f"{fname}.nii.gz"
        npz_path = out_case_dir / f"{fname}.npz"

        if nii_path.exists() and not npz_path.exists():
            # Load NIfTI and save as compressed numpy
            nii = nib.load(str(nii_path))
            data = np.asarray(nii.dataobj, dtype=np.float32)
            np.savez_compressed(npz_path, data=data)

    return case_name


def main():
    project_dir = Path(__file__).parent.parent.parent
    input_dir = project_dir / "data" / "preprocessed_256"
    output_dir = project_dir / "data" / "preprocessed_256_npy"

    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        print("Run scripts/preprocess_256.py first")
        return

    print(f"Converting NIfTI to numpy...")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process train and test
    for split in ['train', 'test']:
        split_input = input_dir / split
        split_output = output_dir / split
        split_output.mkdir(parents=True, exist_ok=True)

        if not split_input.exists():
            print(f"Skipping {split} (not found)")
            continue

        cases = [d for d in split_input.iterdir() if d.is_dir()]
        print(f"\n{split}: {len(cases)} cases")

        is_train = (split == 'train')
        args_list = [(case, split_output, is_train) for case in cases]

        # Use parallel processing for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(
                executor.map(convert_case, args_list),
                total=len(cases),
                desc=f"Converting {split}"
            ))

    # Calculate sizes
    nii_size = sum(f.stat().st_size for f in input_dir.rglob("*.nii.gz")) / (1024**3)
    npz_size = sum(f.stat().st_size for f in output_dir.rglob("*.npz")) / (1024**3)

    print(f"\nDone!")
    print(f"NIfTI size: {nii_size:.1f} GB")
    print(f"NumPy compressed size: {npz_size:.1f} GB")
    print(f"\nTo use numpy data, update config:")
    print(f"  data_dir: 'data/preprocessed_256_npy/train'")
    print(f"  test_dir: 'data/preprocessed_256_npy/test'")


if __name__ == "__main__":
    main()
