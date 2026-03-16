"""
Download and preprocess the PROTEAS brain metastasis dataset from Zenodo.

Dataset: "Diagnostic and follow-up MRIs, CTs, Radiotherapy and Radiomics data of Brain Metastases"
DOI: 10.5281/zenodo.17253793
40 patients, 185 imaging studies, 65 brain metastases with 3-region segmentation.

Usage:
    # Step 1: Download all patient zips
    python download_proteas.py --download

    # Step 2: Explore structure of first patient (to understand naming)
    python download_proteas.py --explore

    # Step 3: Preprocess into pipeline format (256^3, binary masks)
    python download_proteas.py --preprocess
"""

import json
import os
import sys
import zipfile
import shutil
import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_JSON = PROJECT_ROOT.parent / "17253793.json"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "proteas_raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "external_validation_proteas" / "test"
TARGET_SIZE = (256, 256, 256)

# Sequence name mapping: PROTEAS naming -> our pipeline naming
# Will be populated after exploring the dataset structure
SEQUENCE_MAP = {}


# ============================================================================
# DOWNLOAD
# ============================================================================

def load_metadata():
    """Load Zenodo metadata JSON."""
    with open(METADATA_JSON, 'r') as f:
        return json.load(f)


def download_file(url, dest_path, expected_md5=None):
    """Download a file with progress and optional MD5 verification."""
    import urllib.request

    if dest_path.exists():
        # Verify existing file if MD5 provided
        if expected_md5:
            md5 = hashlib.md5()
            with open(dest_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5.update(chunk)
            if md5.hexdigest() == expected_md5:
                print(f"  [SKIP] {dest_path.name} (already downloaded, MD5 OK)")
                return True
            else:
                print(f"  [RE-DOWNLOAD] {dest_path.name} (MD5 mismatch)")
        else:
            print(f"  [SKIP] {dest_path.name} (already exists)")
            return True

    print(f"  Downloading {dest_path.name}...")
    try:
        urllib.request.urlretrieve(url, str(dest_path))
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {dest_path.name}: {e}")
        return False


def download_all():
    """Download all patient zip files from Zenodo."""
    metadata = load_metadata()
    entries = metadata['files']['entries']

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Sort entries to download patients in order
    patient_files = sorted(
        [(k, v) for k, v in entries.items() if k.endswith('.zip')],
        key=lambda x: x[0]
    )
    other_files = [(k, v) for k, v in entries.items() if not k.endswith('.zip')]

    total_size = sum(v['size'] for _, v in patient_files + other_files)
    print(f"Total files: {len(patient_files)} patient zips + {len(other_files)} metadata files")
    print(f"Total size: {total_size / 1e9:.1f} GB\n")

    # Download patient zips
    for filename, info in patient_files:
        url = info['links']['content']
        dest = DOWNLOAD_DIR / filename
        expected_md5 = info['checksum'].replace('md5:', '') if 'checksum' in info else None
        download_file(url, dest, expected_md5)

    # Download metadata files
    for filename, info in other_files:
        url = info['links']['content']
        dest = DOWNLOAD_DIR / filename
        download_file(url, dest)

    print(f"\nDownload complete. Files saved to: {DOWNLOAD_DIR}")


# ============================================================================
# EXPLORE
# ============================================================================

def explore_structure():
    """Explore the internal structure of downloaded zip files."""
    zips = sorted(DOWNLOAD_DIR.glob("P*.zip"))
    if not zips:
        print("No zip files found. Run --download first.")
        return

    print(f"Found {len(zips)} patient zip files\n")

    # Explore first 3 patients in detail
    for zip_path in zips[:3]:
        print(f"=== {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB) ===")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = sorted(zf.namelist())
            for name in names:
                info = zf.getinfo(name)
                if not name.endswith('/'):
                    print(f"  {name} ({info.file_size / 1e6:.1f} MB)")
                else:
                    print(f"  {name}")
        print()

    # Summary: collect all unique file suffixes/patterns across all patients
    print("=== File patterns across all patients ===")
    all_suffixes = set()
    patient_sequences = {}
    for zip_path in zips:
        patient_id = zip_path.stem
        with zipfile.ZipFile(zip_path, 'r') as zf:
            niftis = [n for n in zf.namelist() if n.endswith('.nii') or n.endswith('.nii.gz')]
            patient_sequences[patient_id] = niftis
            for n in niftis:
                # Get the filename part after any directory prefix
                basename = os.path.basename(n)
                all_suffixes.add(basename)

    print(f"\nUnique NIfTI filenames across all patients:")
    for s in sorted(all_suffixes):
        count = sum(1 for seqs in patient_sequences.values() if any(s in n for n in seqs))
        print(f"  {s} ({count}/{len(zips)} patients)")


# ============================================================================
# PREPROCESS
# ============================================================================

def resize_volume(img, target_size, is_mask=False):
    """Resize 3D volume to target size."""
    factors = [t / c for t, c in zip(target_size, img.shape)]
    order = 0 if is_mask else 1
    return zoom(img, factors, order=order, mode='nearest')


def normalize_volume(img):
    """Z-score normalization."""
    mean = img.mean()
    std = img.std()
    if std > 0:
        img = (img - mean) / std
    return img


def process_patient(zip_path, output_dir, sequence_map, target_size):
    """
    Extract, identify sequences, and preprocess a single patient.

    Returns dict with patient info or None on failure.
    """
    patient_id = zip_path.stem
    extract_dir = zip_path.parent / f"_tmp_{patient_id}"

    try:
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Find all NIfTI files
        niftis = list(extract_dir.rglob("*.nii")) + list(extract_dir.rglob("*.nii.gz"))

        if not niftis:
            print(f"  [{patient_id}] No NIfTI files found, skipping")
            return None

        # Identify sequences by filename patterns
        found_sequences = {}
        found_mask = None

        for nii_path in niftis:
            basename = nii_path.name.lower()

            # Check for segmentation mask
            if any(kw in basename for kw in ['seg', 'mask', 'label', 'msk', 'roi']):
                found_mask = nii_path
                continue

            # Match to our sequence names using the map
            for our_name, patterns in sequence_map.items():
                if any(p in basename for p in patterns):
                    found_sequences[our_name] = nii_path
                    break

        # Report what was found
        seq_names = list(found_sequences.keys())
        has_mask = found_mask is not None

        if not has_mask:
            print(f"  [{patient_id}] No segmentation mask found, skipping")
            return None

        if len(found_sequences) < 2:
            print(f"  [{patient_id}] Only {len(found_sequences)} sequences found ({seq_names}), skipping")
            return None

        # Create output directory
        case_dir = output_dir / f"PROTEAS_{patient_id}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # Process and save each sequence
        for seq_name, nii_path in found_sequences.items():
            nii = nib.load(str(nii_path))
            data = np.asarray(nii.dataobj, dtype=np.float32)
            data = resize_volume(data, target_size, is_mask=False)
            # Save as NIfTI (no normalization here - done at training time)
            out_nii = nib.Nifti1Image(data, np.eye(4))
            nib.save(out_nii, str(case_dir / f"{seq_name}.nii.gz"))

        # Process mask - merge multi-class to binary
        nii = nib.load(str(found_mask))
        mask = np.asarray(nii.dataobj, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)  # Binary: any label -> 1
        mask = resize_volume(mask, target_size, is_mask=True)
        out_nii = nib.Nifti1Image(mask, np.eye(4))
        nib.save(out_nii, str(case_dir / "seg.nii.gz"))

        met_count = int(mask.sum() > 0)
        print(f"  [{patient_id}] OK: {seq_names}, mask={'yes' if has_mask else 'no'}, "
              f"voxels={int(mask.sum())}")

        return {
            'patient_id': patient_id,
            'sequences': seq_names,
            'has_mask': True,
            'mask_voxels': int(mask.sum()),
        }

    except Exception as e:
        print(f"  [{patient_id}] ERROR: {e}")
        return None

    finally:
        # Clean up temp extraction
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)


def preprocess_all(sequence_map):
    """Preprocess all patients."""
    zips = sorted(DOWNLOAD_DIR.glob("P*.zip"))
    if not zips:
        print("No zip files found. Run --download first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(zips)} patients...")
    print(f"Sequence mapping: {sequence_map}")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Output: {OUTPUT_DIR}\n")

    results = []
    for zip_path in zips:
        result = process_patient(zip_path, OUTPUT_DIR, sequence_map, TARGET_SIZE)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Processed: {len(results)}/{len(zips)} patients")
    if results:
        all_seqs = set()
        for r in results:
            all_seqs.update(r['sequences'])
        print(f"Sequences found: {sorted(all_seqs)}")

        # Count patients with each sequence
        for seq in sorted(all_seqs):
            count = sum(1 for r in results if seq in r['sequences'])
            print(f"  {seq}: {count}/{len(results)} patients")

    # Save processing report
    report_path = OUTPUT_DIR.parent / "processing_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'total_zips': len(zips),
            'processed': len(results),
            'target_size': list(TARGET_SIZE),
            'sequence_map': {k: v for k, v in sequence_map.items()},
            'patients': results,
        }, f, indent=2)
    print(f"\nReport saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess PROTEAS dataset")
    parser.add_argument('--download', action='store_true', help='Download all patient zips from Zenodo')
    parser.add_argument('--explore', action='store_true', help='Explore internal structure of zip files')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess into pipeline format')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')

    args = parser.parse_args()

    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)

    if not any([args.download, args.explore, args.preprocess]):
        # Default: show help
        parser.print_help()
        print("\nRecommended workflow:")
        print("  1. python download_proteas.py --download")
        print("  2. python download_proteas.py --explore")
        print("  3. Update SEQUENCE_MAP based on explore output")
        print("  4. python download_proteas.py --preprocess")
        return

    if args.download:
        download_all()

    if args.explore:
        explore_structure()

    if args.preprocess:
        # You MUST run --explore first to determine the correct sequence map.
        # Update this map based on the explore output.
        sequence_map = {
            # our_name: [list of filename patterns to match]
            # These are placeholders - update after running --explore
            't1_pre': ['t1_pre', 't1w_pre', 't1_native', 't1w.nii'],
            't1_gd': ['t1_gd', 't1_post', 't1_ce', 't1c', 'contrast', 't1_gado'],
            'flair': ['flair'],
            't2': ['t2w', 't2.nii', 't2_'],
        }
        preprocess_all(sequence_map)


if __name__ == '__main__':
    main()
