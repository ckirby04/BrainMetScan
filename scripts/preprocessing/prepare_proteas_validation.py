"""
Download and preprocess PROTEAS brain metastasis dataset for external validation.

Dataset: University of Cyprus / Bank of Cyprus Oncology Center (PROTEAS project)
DOI: 10.5281/zenodo.17253793
40 patients, BraTS-format NIfTI (skull-stripped, SRI24 atlas, 240x240x155 @ 1mm)
3-region segmentation: enhancing (3), edema (2), necrotic (1)

Pipeline: download zips → extract BraTS baseline → rename sequences → binary mask → resize 256³

Usage:
    python prepare_proteas_validation.py
"""

import json
import os
import sys
import zipfile
import shutil
import hashlib
import urllib.request
from pathlib import Path

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

# BraTS filename → pipeline filename
SEQUENCE_MAP = {
    't1.nii.gz': 't1_pre',
    't1c.nii.gz': 't1_gd',
    'fla.nii.gz': 'flair',
    't2.nii.gz': 't2',
}


# ============================================================================
# DOWNLOAD
# ============================================================================

def download_all_zips():
    """Download all patient zip files from Zenodo."""
    with open(METADATA_JSON, 'r') as f:
        metadata = json.load(f)

    entries = metadata['files']['entries']
    patient_zips = sorted(
        [(k, v) for k, v in entries.items() if k.endswith('.zip')],
        key=lambda x: x[0]
    )

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    total_size = sum(v['size'] for _, v in patient_zips)
    print(f"Downloading {len(patient_zips)} patient zips ({total_size / 1e9:.1f} GB)")
    print(f"Destination: {DOWNLOAD_DIR}\n")

    for i, (filename, info) in enumerate(patient_zips):
        dest = DOWNLOAD_DIR / filename
        url = info['links']['content']
        expected_md5 = info.get('checksum', '').replace('md5:', '')
        size_mb = info['size'] / 1e6

        # Skip if already downloaded with correct MD5
        if dest.exists() and expected_md5:
            md5 = hashlib.md5()
            with open(dest, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5.update(chunk)
            if md5.hexdigest() == expected_md5:
                print(f"[{i+1}/{len(patient_zips)}] {filename} ({size_mb:.0f} MB) — SKIP (MD5 OK)")
                continue

        print(f"[{i+1}/{len(patient_zips)}] {filename} ({size_mb:.0f} MB) — downloading...")
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print("\nDownload complete.")


# ============================================================================
# PREPROCESS
# ============================================================================

def resize_volume(img, target_size, is_mask=False):
    """Resize 3D volume to target size."""
    factors = [t / c for t, c in zip(target_size, img.shape[:3])]
    order = 0 if is_mask else 1
    return zoom(img, factors, order=order, mode='nearest')


def process_patient(zip_path):
    """
    Extract BraTS baseline + mask from a patient zip, preprocess to 256³.
    Returns patient info dict or None on failure.
    """
    patient_id = zip_path.stem  # e.g., "P01"
    case_dir = OUTPUT_DIR / f"PROTEAS_{patient_id}"

    # Skip if already processed
    if case_dir.exists() and (case_dir / "seg.nii.gz").exists():
        existing_seqs = [f.stem for f in case_dir.glob("*.nii.gz") if f.stem != 'seg']
        print(f"  [{patient_id}] SKIP (already processed, seqs: {existing_seqs})")
        return {'patient_id': patient_id, 'sequences': existing_seqs, 'status': 'cached'}

    tmp_dir = DOWNLOAD_DIR / f"_tmp_{patient_id}"
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_names = zf.namelist()

            # Find BraTS baseline files
            brats_baseline = [n for n in all_names
                              if '/BraTS/baseline/' in n and n.endswith('.nii.gz')]
            # Find tumor mask for baseline
            mask_candidates = [n for n in all_names
                               if 'tumor_mask_baseline' in n and n.endswith('.nii.gz')]

            if not brats_baseline:
                print(f"  [{patient_id}] No BraTS/baseline/ found, skipping")
                return None

            if not mask_candidates:
                print(f"  [{patient_id}] No baseline tumor mask found, skipping")
                return None

            # Extract needed files
            for f in brats_baseline + mask_candidates:
                zf.extract(f, str(tmp_dir))

        # Process sequences
        case_dir.mkdir(parents=True, exist_ok=True)
        found_seqs = []

        baseline_dir = tmp_dir / patient_id / "BraTS" / "baseline"
        for brats_name, pipeline_name in SEQUENCE_MAP.items():
            src = baseline_dir / brats_name
            if not src.exists():
                continue

            nii = nib.load(str(src))
            data = np.asarray(nii.dataobj, dtype=np.float32)

            # Handle 4D data (some files may have trailing dimension)
            if data.ndim == 4:
                data = data[:, :, :, 0]

            data = resize_volume(data, TARGET_SIZE, is_mask=False)
            out_nii = nib.Nifti1Image(data, np.eye(4))
            nib.save(out_nii, str(case_dir / f"{pipeline_name}.nii.gz"))
            found_seqs.append(pipeline_name)

        # Process mask — merge multi-class to binary
        mask_path = tmp_dir / mask_candidates[0]
        nii = nib.load(str(mask_path))
        mask = np.asarray(nii.dataobj, dtype=np.float32)
        if mask.ndim == 4:
            mask = mask[:, :, :, 0]
        mask = (mask > 0).astype(np.float32)
        mask = resize_volume(mask, TARGET_SIZE, is_mask=True)
        out_nii = nib.Nifti1Image(mask, np.eye(4))
        nib.save(out_nii, str(case_dir / "seg.nii.gz"))

        tumor_voxels = int(mask.sum())
        print(f"  [{patient_id}] OK: {found_seqs}, tumor_voxels={tumor_voxels}")

        return {
            'patient_id': patient_id,
            'sequences': found_seqs,
            'tumor_voxels': tumor_voxels,
            'status': 'processed',
        }

    except Exception as e:
        print(f"  [{patient_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def preprocess_all():
    """Preprocess all downloaded patient zips."""
    zips = sorted(DOWNLOAD_DIR.glob("P*.zip"))
    if not zips:
        print("No zip files found. Downloading first...")
        download_all_zips()
        zips = sorted(DOWNLOAD_DIR.glob("P*.zip"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nPreprocessing {len(zips)} patients -> {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Sequence map: {SEQUENCE_MAP}\n")

    results = []
    for zip_path in zips:
        result = process_patient(zip_path)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Successfully processed: {len(results)}/{len(zips)} patients")

    if results:
        for seq in ['t1_pre', 't1_gd', 'flair', 't2']:
            count = sum(1 for r in results if seq in r['sequences'])
            print(f"  {seq}: {count}/{len(results)} patients")

        has_tumor = sum(1 for r in results if r.get('tumor_voxels', 0) > 0)
        print(f"  Cases with tumor: {has_tumor}/{len(results)}")

    # Save report
    report = {
        'dataset': 'PROTEAS',
        'source': 'BraTS-format (skull-stripped, SRI24 atlas, 240x240x155)',
        'target_size': list(TARGET_SIZE),
        'total_zips': len(zips),
        'processed': len(results),
        'patients': results,
    }
    report_path = OUTPUT_DIR.parent / "processing_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-only', action='store_true', help='Only download, no preprocessing')
    args = parser.parse_args()

    if args.download_only:
        download_all_zips()
    else:
        # Download if needed, then preprocess
        zips = sorted(DOWNLOAD_DIR.glob("P*.zip")) if DOWNLOAD_DIR.exists() else []
        if len(zips) < 40:
            print("=== Step 1: Download ===")
            download_all_zips()
        print("\n=== Step 2: Preprocess ===")
        preprocess_all()
