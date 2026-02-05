"""
Create a zip file containing only the missing files.

Usage:
    python scripts/create_missing_zip.py missing_files.txt

The missing_files.txt should contain paths like:
    train/CASE_NAME/t1_pre.nii.gz
    train/CASE_NAME/flair.nii.gz
    ...

This script reads from data/preprocessed_256/ and creates missing_files.zip
"""

import zipfile
import sys
from pathlib import Path


def create_missing_zip(missing_list_path: str):
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data" / "preprocessed_256"
    output_zip = project_dir / "missing_files.zip"

    # Read missing files list
    missing_list = Path(missing_list_path)
    if not missing_list.exists():
        print(f"Error: {missing_list_path} not found")
        sys.exit(1)

    with open(missing_list, 'r') as f:
        missing_files = [line.strip() for line in f if line.strip()]

    print(f"Found {len(missing_files)} missing files to package")

    # Create zip
    added = 0
    errors = []

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel_path in missing_files:
            src_path = data_dir / rel_path
            if src_path.exists():
                zf.write(src_path, rel_path)
                added += 1
                if added % 50 == 0:
                    print(f"  Added {added}/{len(missing_files)} files...")
            else:
                errors.append(rel_path)

    # Report
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"\nCreated: {output_zip}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Files added: {added}")

    if errors:
        print(f"\nWarning: {len(errors)} files not found locally:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print("\nUpload missing_files.zip to Colab and run the extraction cell.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_missing_zip.py missing_files.txt")
        sys.exit(1)

    create_missing_zip(sys.argv[1])
