"""
Prepare PRETREAT-METSTOBRAIN-MASKS dataset as external validation set.

Downloads (or processes already-downloaded) data from TCIA and converts it
to match the pipeline's expected format:
  - 4-channel NIfTI: t1_pre.nii.gz, t1_gd.nii.gz, flair.nii.gz, t2.nii.gz
  - Binary segmentation mask: seg.nii.gz
  - Resized to 256³
  - Case prefix: PRETREAT_

The PRETREAT dataset contains 200 patients with:
  - T1, T1 post-gadolinium, T2, FLAIR (co-registered, 1mm iso, skull-stripped)
  - Multi-class segmentation (enhancing=1, necrotic=2, edema=3) → merged to binary

Data source: https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/

Usage:
    # Step 1: Download raw data (requires tcia_utils)
    python scripts/prepare_pretreat_validation.py --download --output-dir "Raw Data/PRETREAT"

    # Step 2: Process downloaded data into pipeline format
    python scripts/prepare_pretreat_validation.py --input-dir "Raw Data/PRETREAT" --output-dir data/external_validation

    # Or do both at once:
    python scripts/prepare_pretreat_validation.py --download --output-dir data/external_validation

    # Also set up nnU-Net format:
    python scripts/prepare_pretreat_validation.py --input-dir "Raw Data/PRETREAT" --output-dir data/external_validation --setup-nnunet
"""

import os
import sys
import json
import shutil
import argparse
import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

TARGET_SIZE = (256, 256, 256)
CASE_PREFIX = "PRETREAT_"
SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']

# Known naming patterns for TCIA brain metastasis datasets.
# The script auto-detects which pattern matches the downloaded data.
SEQUENCE_PATTERNS = {
    't1_pre': [
        '-t1n.nii.gz', '_t1n.nii.gz', 't1n.nii.gz',   # BraTS-METS style (T1 native)
        't1.nii.gz', 't1_pre.nii.gz', 't1w.nii.gz',
        '_t1.nii.gz', '_T1.nii.gz', '-t1.nii.gz',
        'T1.nii.gz', 'T1w.nii.gz', 'T1_pre.nii.gz',
    ],
    't1_gd': [
        '-t1c.nii.gz', '_t1c.nii.gz', 't1c.nii.gz',   # BraTS-METS style (T1 contrast)
        't1ce.nii.gz', 't1_gd.nii.gz',
        't1_post.nii.gz', 't1post.nii.gz', 't1_ce.nii.gz',
        '_t1ce.nii.gz', '_T1c.nii.gz',
        '_T1CE.nii.gz', '_T1post.nii.gz',
        'T1c.nii.gz', 'T1CE.nii.gz', 'T1_Gd.nii.gz',
        'T1post.nii.gz', 'T1_post.nii.gz', 'T1_ce.nii.gz',
        't1_contrast.nii.gz', 'T1_contrast.nii.gz',
    ],
    'flair': [
        '-t2f.nii.gz', '_t2f.nii.gz', 't2f.nii.gz',   # BraTS-METS style (T2-FLAIR)
        'flair.nii.gz', 'FLAIR.nii.gz', 'Flair.nii.gz',
        '_flair.nii.gz', '_FLAIR.nii.gz', '-flair.nii.gz',
        't2_flair.nii.gz', 'T2_FLAIR.nii.gz', 'T2FLAIR.nii.gz',
    ],
    't2': [
        '-t2w.nii.gz', '_t2w.nii.gz', 't2w.nii.gz',   # BraTS-METS style (T2 weighted)
        't2.nii.gz', 'T2.nii.gz', 'T2w.nii.gz',
        '_t2.nii.gz', '_T2.nii.gz', '-t2.nii.gz',
        'T2_weighted.nii.gz',
    ],
    'seg': [
        '-seg.nii.gz', '_seg.nii.gz',                   # BraTS-METS style
        'seg.nii.gz', 'mask.nii.gz', 'segmentation.nii.gz',
        '_mask.nii.gz',
        'labels.nii.gz', '_labels.nii.gz',
        'tumor_seg.nii.gz', 'tumor_mask.nii.gz',
    ],
}

# ─── Download ────────────────────────────────────────────────────────────────

def download_from_tcia(download_dir: Path):
    """Download PRETREAT-METSTOBRAIN-MASKS from TCIA using tcia_utils."""
    try:
        from tcia_utils import nbia
    except ImportError:
        logger.error(
            "tcia_utils not installed. Install with:\n"
            "  pip install tcia_utils\n\n"
            "Alternatively, download manually:\n"
            "  1. Go to https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/\n"
            "  2. Click 'Download' and use NBIA Data Retriever\n"
            "  3. Extract to a folder and run this script with --input-dir <folder>"
        )
        sys.exit(1)

    download_dir.mkdir(parents=True, exist_ok=True)
    collection = "PRETREAT-METSTOBRAIN-MASKS"

    logger.info(f"Querying TCIA for collection: {collection}")

    try:
        # Get all series in the collection
        series = nbia.getSeries(collection=collection)
        if series is None or len(series) == 0:
            logger.error(
                f"No series found for collection '{collection}'.\n"
                "The collection may require a data usage agreement.\n"
                "Please download manually from:\n"
                "  https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/"
            )
            sys.exit(1)

        logger.info(f"Found {len(series)} series to download")
        logger.info(f"Downloading to: {download_dir}")

        # Download all series
        nbia.downloadSeries(
            series,
            path=str(download_dir),
            input_type="list"
        )

        logger.info("Download complete!")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(
            "\nManual download instructions:\n"
            "  1. Visit: https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/\n"
            "  2. Download using NBIA Data Retriever or the web interface\n"
            "  3. Extract NIfTI files to a folder\n"
            "  4. Re-run: python scripts/prepare_pretreat_validation.py --input-dir <folder> --output-dir data/external_validation"
        )
        sys.exit(1)


# ─── Auto-detection ──────────────────────────────────────────────────────────

def detect_naming_convention(patient_dir: Path) -> dict:
    """Auto-detect which naming convention is used for NIfTI files in a patient dir."""
    all_niftis = list(patient_dir.rglob("*.nii.gz"))
    if not all_niftis:
        all_niftis = list(patient_dir.rglob("*.nii"))

    filenames_lower = {f.name.lower(): f for f in all_niftis}
    detected = {}

    for seq, patterns in SEQUENCE_PATTERNS.items():
        for pattern in patterns:
            # Direct match
            if pattern.lower() in filenames_lower:
                detected[seq] = filenames_lower[pattern.lower()]
                break
            # Suffix match (e.g., "PatientID_t1c.nii.gz")
            for fname_lower, fpath in filenames_lower.items():
                if fname_lower.endswith(pattern.lower()):
                    if seq not in detected:
                        detected[seq] = fpath
                        break

    return detected


def discover_patients(input_dir: Path) -> list:
    """
    Discover patient directories in the downloaded data.
    Handles multiple possible directory structures from TCIA downloads.
    """
    patients = []

    # Strategy 1: Direct patient folders with NIfTI files
    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue
        niftis = list(d.glob("*.nii.gz")) + list(d.glob("*.nii"))
        if niftis:
            patients.append(d)

    # Strategy 2: Nested structure (TCIA download format)
    if not patients:
        for d in sorted(input_dir.rglob("*")):
            if not d.is_dir():
                continue
            niftis = list(d.glob("*.nii.gz")) + list(d.glob("*.nii"))
            if len(niftis) >= 2:  # At least 2 NIfTI files (1 image + 1 mask)
                # Only add leaf directories (no child dirs with NIfTIs)
                child_dirs_with_nifti = [
                    c for c in d.iterdir()
                    if c.is_dir() and (list(c.glob("*.nii.gz")) or list(c.glob("*.nii")))
                ]
                if not child_dirs_with_nifti:
                    patients.append(d)

    if not patients:
        logger.error(
            f"No patient directories with NIfTI files found in {input_dir}\n"
            "Expected structure:\n"
            "  input_dir/\n"
            "    patient_001/\n"
            "      *_t1.nii.gz, *_t1c.nii.gz, *_flair.nii.gz, *_t2.nii.gz, *_seg.nii.gz\n"
            "    patient_002/\n"
            "      ..."
        )
        sys.exit(1)

    return patients


# ─── Processing ──────────────────────────────────────────────────────────────

def load_nifti(path: Path) -> tuple:
    """Load NIfTI file, return (data, affine)."""
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj, dtype=np.float32)
    return data, nii.affine


def resize_volume(img: np.ndarray, target_size: tuple, is_mask: bool = False) -> np.ndarray:
    """Resize 3D volume to target size."""
    zoom_factors = [t / c for t, c in zip(target_size, img.shape)]
    order = 0 if is_mask else 1
    return zoom(img, zoom_factors, order=order, mode='nearest')


def merge_to_binary(mask: np.ndarray) -> np.ndarray:
    """
    Merge multi-class segmentation to binary (any tumor = 1).

    PRETREAT labels:
      0 = background
      1 = enhancing/core tumor
      2 = necrotic core
      3 = peritumoral edema (whole tumor boundary on FLAIR)

    All non-zero labels → 1 (metastasis present).
    """
    return (mask > 0).astype(np.float32)


def process_patient(
    patient_dir: Path,
    output_dir: Path,
    case_id: str,
    detected_files: dict,
    target_size: tuple = TARGET_SIZE,
    skip_existing: bool = True,
) -> dict:
    """
    Process a single patient: map sequences, merge masks, resize to 256³.

    Returns dict with processing stats.
    """
    output_case_dir = output_dir / f"{CASE_PREFIX}{case_id}"
    stats = {"case_id": case_id, "sequences_found": [], "sequences_missing": [], "status": "ok"}

    if skip_existing and output_case_dir.exists():
        expected = [f"{s}.nii.gz" for s in SEQUENCES] + ["seg.nii.gz"]
        if all((output_case_dir / f).exists() for f in expected):
            stats["status"] = "skipped (already processed)"
            return stats

    output_case_dir.mkdir(parents=True, exist_ok=True)

    # Process image sequences
    for seq in SEQUENCES:
        output_path = output_case_dir / f"{seq}.nii.gz"

        if seq not in detected_files:
            stats["sequences_missing"].append(seq)
            # Create zero volume as placeholder if sequence is missing
            logger.warning(f"  {case_id}: Missing {seq}, creating zero-filled placeholder")
            zero_vol = np.zeros(target_size, dtype=np.float32)
            nii_out = nib.Nifti1Image(zero_vol, np.eye(4))
            nib.save(nii_out, str(output_path))
            continue

        stats["sequences_found"].append(seq)
        src_path = detected_files[seq]

        try:
            data, affine = load_nifti(src_path)
            if data.shape != target_size:
                data = resize_volume(data, target_size, is_mask=False)

            nii_out = nib.Nifti1Image(data.astype(np.float32), affine)
            nib.save(nii_out, str(output_path))
        except Exception as e:
            logger.error(f"  {case_id}: Failed to process {seq} from {src_path}: {e}")
            stats["status"] = "error"

    # Process segmentation mask
    seg_output = output_case_dir / "seg.nii.gz"
    if 'seg' in detected_files:
        try:
            mask_data, affine = load_nifti(detected_files['seg'])

            # Check if multi-class and merge to binary
            unique_labels = np.unique(mask_data)
            if len(unique_labels) > 2:
                logger.info(f"  {case_id}: Multi-class mask (labels: {unique_labels}) → binary")

            mask_data = merge_to_binary(mask_data)

            if mask_data.shape != target_size:
                mask_data = resize_volume(mask_data, target_size, is_mask=True)

            # Ensure binary after resize
            mask_data = (mask_data > 0.5).astype(np.float32)

            num_lesion_voxels = int(mask_data.sum())
            stats["lesion_voxels"] = num_lesion_voxels

            nii_out = nib.Nifti1Image(mask_data, affine)
            nib.save(nii_out, str(seg_output))

        except Exception as e:
            logger.error(f"  {case_id}: Failed to process segmentation: {e}")
            stats["status"] = "error"
    else:
        stats["sequences_missing"].append("seg")
        logger.warning(f"  {case_id}: No segmentation mask found")

    return stats


# ─── nnU-Net setup ───────────────────────────────────────────────────────────

def setup_nnunet_format(processed_dir: Path, nnunet_dir: Path):
    """
    Set up external validation data in nnU-Net format for inference.

    Creates:
      nnunet_dir/Dataset002_BrainMetsExtVal/
        imagesTs/    (test images for inference)
        labelsTs/    (ground truth for evaluation)
        dataset.json
    """
    dataset_name = "Dataset002_BrainMetsExtVal"
    dataset_dir = nnunet_dir / "nnUNet_raw" / dataset_name
    images_dir = dataset_dir / "imagesTs"
    labels_dir = dataset_dir / "labelsTs"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    channel_map = {
        "t1_pre": "0000",
        "t1_gd": "0001",
        "flair": "0002",
        "t2": "0003",
    }

    case_count = 0
    case_dirs = sorted([
        d for d in processed_dir.iterdir()
        if d.is_dir() and d.name.startswith(CASE_PREFIX)
    ])

    for case_dir in case_dirs:
        case_id = case_dir.name

        # Copy image channels
        all_found = True
        for seq_name, channel_id in channel_map.items():
            src = case_dir / f"{seq_name}.nii.gz"
            dst = images_dir / f"{case_id}_{channel_id}.nii.gz"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                all_found = False

        # Copy segmentation as label
        seg_src = case_dir / "seg.nii.gz"
        seg_dst = labels_dir / f"{case_id}.nii.gz"
        if seg_src.exists():
            shutil.copy2(seg_src, seg_dst)

        if all_found:
            case_count += 1

    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0000": "T1_pre",
            "0001": "T1_Gd",
            "0002": "FLAIR",
            "0003": "T2"
        },
        "labels": {
            "background": 0,
            "metastasis": 1
        },
        "numTraining": 0,
        "numTest": case_count,
        "file_ending": ".nii.gz",
        "description": "PRETREAT-METSTOBRAIN-MASKS external validation set (TCIA)",
        "reference": "https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/",
    }

    with open(dataset_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)

    logger.info(f"nnU-Net format ready: {dataset_dir}")
    logger.info(f"  {case_count} cases in imagesTs/labelsTs")
    logger.info(f"  Run inference with:")
    logger.info(f"    nnUNetv2_predict -i {images_dir} -o <output_dir> -d 001 -c 3d_fullres")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare PRETREAT-METSTOBRAIN-MASKS as external validation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process:
  python scripts/prepare_pretreat_validation.py --download --output-dir data/external_validation

  # Process already-downloaded data:
  python scripts/prepare_pretreat_validation.py --input-dir "Raw Data/PRETREAT" --output-dir data/external_validation

  # Also create nnU-Net format:
  python scripts/prepare_pretreat_validation.py --input-dir "Raw Data/PRETREAT" --output-dir data/external_validation --setup-nnunet
        """
    )
    parser.add_argument('--download', action='store_true',
                        help='Download dataset from TCIA (requires tcia_utils)')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory with downloaded PRETREAT NIfTI data')
    parser.add_argument('--output-dir', type=str, default='data/external_validation',
                        help='Output directory for processed validation data')
    parser.add_argument('--target-size', type=int, default=256,
                        help='Target resolution (default: 256)')
    parser.add_argument('--setup-nnunet', action='store_true',
                        help='Also create nnU-Net imagesTs/labelsTs format')
    parser.add_argument('--nnunet-dir', type=str, default='nnUNet',
                        help='nnU-Net base directory (default: nnUNet)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Reprocess cases even if output already exists')

    args = parser.parse_args()

    # Resolve paths relative to project root (1.30/)
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / args.output_dir

    target_size = (args.target_size, args.target_size, args.target_size)

    # Step 1: Download if requested
    if args.download:
        download_dir = output_dir / "_raw_download"
        download_from_tcia(download_dir)
        input_dir = download_dir
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute():
            input_dir = script_dir / input_dir
    else:
        parser.error("Provide --input-dir or --download")
        return

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Step 2: Discover patients
    logger.info("=" * 60)
    logger.info("PRETREAT-METSTOBRAIN-MASKS → External Validation Set")
    logger.info("=" * 60)
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Target: {target_size[0]}³")
    logger.info("")

    patients = discover_patients(input_dir)
    logger.info(f"Discovered {len(patients)} patient directories")

    # Step 3: Detect naming convention from first patient
    sample_detected = detect_naming_convention(patients[0])
    logger.info(f"Sample patient: {patients[0].name}")
    logger.info(f"Detected files:")
    for seq, path in sample_detected.items():
        logger.info(f"  {seq:8s} → {path.name}")
    logger.info("")

    # Step 4: Process all patients
    processed_dir = output_dir / "test"
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    from tqdm import tqdm

    for patient_dir in tqdm(patients, desc="Processing patients"):
        detected = detect_naming_convention(patient_dir)
        case_id = patient_dir.name

        # Clean up case ID (remove special chars, keep alphanumeric + underscore)
        case_id = case_id.replace("-", "_").replace(" ", "_")

        stats = process_patient(
            patient_dir=patient_dir,
            output_dir=processed_dir,
            case_id=case_id,
            detected_files=detected,
            target_size=target_size,
            skip_existing=not args.no_skip,
        )
        all_stats.append(stats)

    # Step 5: Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)

    successful = [s for s in all_stats if s["status"] == "ok"]
    skipped = [s for s in all_stats if "skipped" in s["status"]]
    errors = [s for s in all_stats if s["status"] == "error"]

    logger.info(f"Total patients:     {len(all_stats)}")
    logger.info(f"Successfully processed: {len(successful)}")
    logger.info(f"Skipped (existing):     {len(skipped)}")
    logger.info(f"Errors:                 {len(errors)}")

    if successful:
        # Count missing sequences
        missing_counts = defaultdict(int)
        for s in successful:
            for m in s.get("sequences_missing", []):
                missing_counts[m] += 1

        if missing_counts:
            logger.info("\nMissing sequences (zero-filled):")
            for seq, count in sorted(missing_counts.items()):
                logger.info(f"  {seq}: {count} cases")

        # Lesion statistics
        lesion_voxels = [s.get("lesion_voxels", 0) for s in successful if "lesion_voxels" in s]
        if lesion_voxels:
            logger.info(f"\nLesion volume stats (voxels at {target_size[0]}³):")
            logger.info(f"  Mean:   {np.mean(lesion_voxels):.0f}")
            logger.info(f"  Median: {np.median(lesion_voxels):.0f}")
            logger.info(f"  Min:    {np.min(lesion_voxels)}")
            logger.info(f"  Max:    {np.max(lesion_voxels)}")

    if errors:
        logger.info("\nFailed cases:")
        for s in errors:
            logger.info(f"  {s['case_id']}")

    # Save processing report
    report_path = output_dir / "processing_report.json"
    report = {
        "source": "PRETREAT-METSTOBRAIN-MASKS",
        "tcia_url": "https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/",
        "target_size": list(target_size),
        "total_patients": len(all_stats),
        "processed": len(successful),
        "skipped": len(skipped),
        "errors": len(errors),
        "case_prefix": CASE_PREFIX,
        "output_structure": {
            "test_dir": str(processed_dir),
            "format": "NIfTI (.nii.gz)",
            "sequences": SEQUENCES,
            "mask": "seg.nii.gz (binary: 0=background, 1=metastasis)",
        },
        "cases": all_stats,
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved: {report_path}")

    # Step 6: nnU-Net format (optional)
    if args.setup_nnunet:
        logger.info("\n" + "=" * 60)
        logger.info("Setting up nnU-Net format...")
        nnunet_dir = script_dir / args.nnunet_dir
        setup_nnunet_format(processed_dir, nnunet_dir)

    # Final usage instructions
    logger.info("\n" + "=" * 60)
    logger.info("READY FOR EVALUATION")
    logger.info("=" * 60)
    logger.info(f"\nCustom ensemble inference:")
    logger.info(f"  python scripts/run_inference.py \\")
    logger.info(f"    --data-dir {processed_dir} \\")
    logger.info(f"    --output-dir results/external_validation")
    logger.info(f"\nEvaluate predictions:")
    logger.info(f"  python scripts/evaluate_model.py \\")
    logger.info(f"    --pred-dir results/external_validation \\")
    logger.info(f"    --gt-dir {processed_dir} \\")
    logger.info(f"    --prefix {CASE_PREFIX}")
    if args.setup_nnunet:
        nnunet_images = script_dir / args.nnunet_dir / "nnUNet_raw" / "Dataset002_BrainMetsExtVal" / "imagesTs"
        logger.info(f"\nnnU-Net inference:")
        logger.info(f"  nnUNetv2_predict -i {nnunet_images} -o <output> -d 001 -c 3d_fullres")
    logger.info("")


if __name__ == '__main__':
    main()
