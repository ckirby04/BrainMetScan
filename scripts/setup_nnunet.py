"""
Setup nnU-Net v2 for brain metastasis segmentation.

Converts our preprocessed_256 data into nnU-Net's expected directory structure
using hard links (no disk space wasted).

Usage:
    pip install nnunetv2
    python scripts/setup_nnunet.py

Then follow the printed instructions for training.
"""

import json
import os
import sys
from pathlib import Path

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_256" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "preprocessed_256" / "test"

# nnU-Net directories (inside project to keep things organized)
NNUNET_BASE = PROJECT_ROOT / "nnUNet"
NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"

DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_BrainMets"

# Channel mapping: index -> (filename, display_name)
CHANNELS = {
    "0000": ("t1_pre", "T1_pre"),
    "0001": ("t1_gd", "T1_Gd"),
    "0002": ("flair", "FLAIR"),
    "0003": ("t2", "T2"),
}


def link_or_copy(src: Path, dst: Path):
    """Create hard link (no extra disk space). Falls back to copy."""
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        import shutil
        shutil.copy2(str(src), str(dst))


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # Create directory structure
    dataset_dir = NNUNET_RAW / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"

    for d in [NNUNET_PREPROCESSED, NNUNET_RESULTS, images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all training cases
    cases = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])
    print(f"Found {len(cases)} training cases")

    # Link training data
    n_linked = 0
    n_skipped = 0
    for case_dir in cases:
        case_id = case_dir.name
        seg_file = case_dir / "seg.nii.gz"

        if not seg_file.exists():
            print(f"  WARNING: No seg.nii.gz for {case_id}, skipping")
            n_skipped += 1
            continue

        # Link each channel
        all_channels_found = True
        for suffix, (filename, _) in CHANNELS.items():
            src = case_dir / f"{filename}.nii.gz"
            if not src.exists():
                print(f"  WARNING: Missing {filename}.nii.gz for {case_id}")
                all_channels_found = False
                break

        if not all_channels_found:
            n_skipped += 1
            continue

        # All files exist, create links
        for suffix, (filename, _) in CHANNELS.items():
            src = case_dir / f"{filename}.nii.gz"
            dst = images_tr / f"{case_id}_{suffix}.nii.gz"
            link_or_copy(src, dst)

        # Link segmentation
        dst_seg = labels_tr / f"{case_id}.nii.gz"
        link_or_copy(seg_file, dst_seg)
        n_linked += 1

        if n_linked % 50 == 0:
            print(f"  Linked {n_linked}/{len(cases)} cases...")

    print(f"Linked {n_linked} training cases ({n_skipped} skipped)")

    # Link test data (if exists)
    n_test = 0
    if TEST_DIR.exists():
        test_cases = sorted([p for p in TEST_DIR.iterdir() if p.is_dir()])
        for case_dir in test_cases:
            case_id = case_dir.name
            for suffix, (filename, _) in CHANNELS.items():
                src = case_dir / f"{filename}.nii.gz"
                if src.exists():
                    dst = images_ts / f"{case_id}_{suffix}.nii.gz"
                    link_or_copy(src, dst)
            n_test += 1
        print(f"Linked {n_test} test cases")

    # Create dataset.json
    dataset_json = {
        "channel_names": {k: v[1] for k, v in CHANNELS.items()},
        "labels": {
            "background": 0,
            "metastasis": 1,
        },
        "numTraining": n_linked,
        "file_ending": ".nii.gz",
    }

    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Wrote {json_path}")

    # Print next steps
    print("\n" + "=" * 60)
    print("nnU-Net setup complete!")
    print("=" * 60)

    env_raw = str(NNUNET_RAW)
    env_pre = str(NNUNET_PREPROCESSED)
    env_res = str(NNUNET_RESULTS)

    print(f"""
STEP 1: Set environment variables (add to your shell profile):

    export nnUNet_raw="{env_raw}"
    export nnUNet_preprocessed="{env_pre}"
    export nnUNet_results="{env_res}"

STEP 2: Plan and preprocess (CPU, ~30-60 min):

    nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity

STEP 3: Train fold 0 baseline (GPU, ~24-48 hrs):

    nnUNetv2_train {DATASET_ID} 3d_fullres 0

    If OOM, try 3d_lowres instead:
    nnUNetv2_train {DATASET_ID} 3d_lowres 0

STEP 4 (optional): Train all 5 folds for full cross-validation:

    for fold in 0 1 2 3 4; do
        nnUNetv2_train {DATASET_ID} 3d_fullres $fold
    done

STEP 5: Find best configuration:

    nnUNetv2_find_best_configuration {DATASET_ID} -c 3d_fullres

STEP 6: Predict test cases:

    nnUNetv2_predict -i "{env_raw}/{DATASET_NAME}/imagesTs" -o "{env_res}/test_predictions" -d {DATASET_ID} -c 3d_fullres -f 0
""")


if __name__ == "__main__":
    main()
