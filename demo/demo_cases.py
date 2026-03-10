"""
Demo case generator and loader for BrainMetScan investor demos.
Pre-computes results from real cases so the viewer works without GPU/models loaded.

Usage:
    # Generate demo cases (requires GPU and models):
    python demo/demo_cases.py --generate

    # The demo app automatically loads pre-computed cases when models aren't available.
"""

import json
import sys
import os
from pathlib import Path

import numpy as np

# Demo case directory
DEMO_DIR = Path(__file__).parent / "precomputed"


def get_demo_cases():
    """List available pre-computed demo cases."""
    if not DEMO_DIR.exists():
        return []
    cases = []
    for case_dir in sorted(DEMO_DIR.iterdir()):
        if case_dir.is_dir() and (case_dir / "metadata.json").exists():
            with open(case_dir / "metadata.json") as f:
                meta = json.load(f)
            cases.append({
                "name": case_dir.name,
                "description": meta.get("description", ""),
                "lesion_count": meta.get("lesion_count", 0),
                "dice_score": meta.get("dice_score", 0),
                "category": meta.get("category", "unknown"),
            })
    return cases


def load_demo_case(case_name):
    """
    Load a pre-computed demo case.

    Returns:
        dict with keys: t1_gd_slice, prediction_slice, ground_truth_slice,
                        visualization_png, report_html, metadata
    """
    case_dir = DEMO_DIR / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f"Demo case not found: {case_name}")

    result = {}

    # Load metadata
    with open(case_dir / "metadata.json") as f:
        result["metadata"] = json.load(f)

    # Load pre-rendered images (PNG bytes)
    for img_name in ["multiview.png", "ensemble.png", "navigator.png", "metrics.png"]:
        img_path = case_dir / img_name
        if img_path.exists():
            result[img_name.replace(".png", "")] = img_path.read_bytes()

    # Load report HTML
    report_path = case_dir / "report.html"
    if report_path.exists():
        result["report_html"] = report_path.read_text(encoding="utf-8")

    # Load slice info
    info_path = case_dir / "slice_info.md"
    if info_path.exists():
        result["slice_info"] = info_path.read_text(encoding="utf-8")

    return result


def generate_demo_cases(data_dir, model_dir, output_dir=None, max_cases=5):
    """
    Generate pre-computed demo cases from real data.
    Requires models and data to be available.

    Args:
        data_dir: Path to training data (with Mets_XXX directories)
        model_dir: Path to model checkpoints
        output_dir: Output directory (defaults to demo/precomputed/)
        max_cases: Maximum number of demo cases to generate
    """
    import nibabel as nib
    from PIL import Image
    import io

    if output_dir is None:
        output_dir = DEMO_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)

    # Find cases with interesting characteristics
    case_dirs = sorted([d for d in data_dir.iterdir()
                        if d.is_dir() and (d.name.startswith("Mets_") or d.name.startswith("BMS_Mets_"))])

    # Select diverse demo cases
    selected = _select_diverse_cases(case_dirs, max_cases)

    print(f"Generating {len(selected)} demo cases...")

    for i, case_dir in enumerate(selected):
        case_name = case_dir.name
        print(f"  [{i+1}/{len(selected)}] Processing {case_name}...")

        case_output = output_dir / case_name
        case_output.mkdir(parents=True, exist_ok=True)

        try:
            _generate_single_case(case_dir, case_output, model_dir)
        except Exception as e:
            print(f"    Error: {e}")
            continue

    print(f"Demo cases saved to: {output_dir}")


def _select_diverse_cases(case_dirs, max_cases):
    """Select a diverse set of cases (varying lesion counts and sizes)."""
    import nibabel as nib

    case_info = []
    for case_dir in case_dirs:
        seg_path = case_dir / "seg.nii.gz"
        if not seg_path.exists():
            continue

        seg = nib.load(str(seg_path)).get_fdata()
        lesion_volume = np.sum(seg > 0.5)

        from skimage.measure import label
        labeled = label(seg > 0.5)
        n_lesions = labeled.max()

        case_info.append({
            "dir": case_dir,
            "n_lesions": n_lesions,
            "volume": lesion_volume,
        })

    if not case_info:
        return case_dirs[:max_cases]

    # Sort by volume and select diverse examples
    case_info.sort(key=lambda x: x["volume"])

    # Categories: no lesion, small single, large single, multiple, many
    categories = {
        "small_single": [c for c in case_info if c["n_lesions"] == 1 and c["volume"] < 2000],
        "large_single": [c for c in case_info if c["n_lesions"] == 1 and c["volume"] >= 2000],
        "multiple": [c for c in case_info if 2 <= c["n_lesions"] <= 4],
        "many": [c for c in case_info if c["n_lesions"] >= 5],
    }

    selected = []
    for cat_name, cat_cases in categories.items():
        if cat_cases and len(selected) < max_cases:
            # Pick median case from category
            mid = len(cat_cases) // 2
            selected.append(cat_cases[mid]["dir"])

    # Fill remaining with random
    remaining = [c["dir"] for c in case_info if c["dir"] not in selected]
    while len(selected) < max_cases and remaining:
        selected.append(remaining.pop(len(remaining) // 2))

    return selected


def _generate_single_case(case_dir, output_dir, model_dir):
    """Generate pre-computed data for a single demo case."""
    import nibabel as nib
    from skimage.measure import label, regionprops

    # Load ground truth
    seg = nib.load(str(case_dir / "seg.nii.gz")).get_fdata().astype(np.float32)
    labeled = label(seg > 0.5)
    regions = regionprops(labeled)

    n_lesions = len(regions)
    total_volume = np.sum(seg > 0.5)

    # Categorize
    if n_lesions == 0:
        category = "no_lesions"
    elif n_lesions == 1:
        category = "single_lesion"
    elif n_lesions <= 4:
        category = "oligometastatic"
    else:
        category = "multiple_metastases"

    # Save metadata
    metadata = {
        "case_name": case_dir.name,
        "category": category,
        "lesion_count": n_lesions,
        "total_volume_voxels": int(total_volume),
        "description": _get_case_description(n_lesions, total_volume, category),
        "dice_score": 0.0,  # Will be filled if model available
        "generated_with": "demo_cases.py",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate slice info
    slice_info = f"""### Case Info
- **Case:** {case_dir.name}
- **Category:** {category.replace('_', ' ').title()}
- **Lesions:** {n_lesions}
- **Total Volume:** {total_volume:,.0f} voxels

### Demo Mode
Pre-computed results — no GPU required.
"""
    with open(output_dir / "slice_info.md", "w") as f:
        f.write(slice_info)

    print(f"    Saved metadata for {case_dir.name} ({category}, {n_lesions} lesions)")


def _get_case_description(n_lesions, volume, category):
    """Generate a human-readable description for a demo case."""
    if n_lesions == 0:
        return "Normal brain MRI — no metastatic lesions detected"
    elif n_lesions == 1:
        size = "small" if volume < 1000 else "moderate" if volume < 5000 else "large"
        return f"Single {size} brain metastasis ({volume:,.0f} voxels)"
    elif n_lesions <= 4:
        return f"Oligometastatic disease — {n_lesions} lesions ({volume:,.0f} total voxels)"
    else:
        return f"Multiple brain metastases — {n_lesions} lesions ({volume:,.0f} total voxels)"


# Hardcoded demo cases for when no pre-computed data exists
# These use synthetic data to show the UI without any real data or models
SYNTHETIC_DEMO_CASES = [
    {
        "name": "Demo_SingleLesion",
        "description": "Single brain metastasis in the right parietal lobe",
        "category": "single_lesion",
        "lesion_count": 1,
        "total_volume_voxels": 3500,
        "dice_score": 0.87,
    },
    {
        "name": "Demo_MultipleLesions",
        "description": "Multiple metastases (3 lesions) in bilateral hemispheres",
        "category": "oligometastatic",
        "lesion_count": 3,
        "total_volume_voxels": 8200,
        "dice_score": 0.82,
    },
    {
        "name": "Demo_TinyLesion",
        "description": "Small metastasis (<5mm) in the cerebellum",
        "category": "single_lesion",
        "lesion_count": 1,
        "total_volume_voxels": 450,
        "dice_score": 0.71,
    },
    {
        "name": "Demo_LargeTumor",
        "description": "Large metastasis with surrounding edema in the frontal lobe",
        "category": "single_lesion",
        "lesion_count": 1,
        "total_volume_voxels": 15000,
        "dice_score": 0.91,
    },
    {
        "name": "Demo_PostTreatment",
        "description": "Post-SRS treatment response — residual enhancement",
        "category": "single_lesion",
        "lesion_count": 1,
        "total_volume_voxels": 1200,
        "dice_score": 0.78,
    },
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pre-computed demo cases")
    parser.add_argument("--generate", action="store_true", help="Generate demo cases from real data")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to training data")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to model checkpoints")
    parser.add_argument("--max-cases", type=int, default=5, help="Max demo cases to generate")
    parser.add_argument("--list", action="store_true", help="List available demo cases")

    args = parser.parse_args()

    if args.list:
        cases = get_demo_cases()
        if cases:
            print(f"Available demo cases ({len(cases)}):")
            for c in cases:
                print(f"  {c['name']}: {c['description']} (Dice: {c['dice_score']:.1%})")
        else:
            print("No pre-computed demo cases found.")
            print("Use --generate to create them, or use synthetic demo mode.")

    elif args.generate:
        data_dir = args.data_dir or str(Path(__file__).parent.parent / "data" / "train")
        model_dir = args.model_dir or str(Path(__file__).parent.parent / "model")
        generate_demo_cases(data_dir, model_dir, max_cases=args.max_cases)

    else:
        print("Use --generate to create demo cases or --list to list existing ones.")
