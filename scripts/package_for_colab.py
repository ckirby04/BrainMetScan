"""
Package code for Colab upload.

Creates brainMetShare_code.zip containing only the necessary files.

Usage:
    python scripts/package_for_colab.py
"""

import zipfile
import os
from pathlib import Path

def package_for_colab():
    project_dir = Path(__file__).parent.parent
    output_zip = project_dir / "brainMetShare_code.zip"

    # Files/folders to include
    include_patterns = [
        "src/**/*.py",
        "scripts/*.py",
        "configs/*.yaml",
    ]

    # Specific files to include
    include_files = [
        "src/segmentation/__init__.py",
    ]

    # Folders to include entirely
    include_folders = [
        "src/segmentation",
        "scripts",
        "configs",
        "notebooks",
    ]

    print(f"Packaging code for Colab...")
    print(f"Output: {output_zip}")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in include_folders:
            folder_path = project_dir / folder
            if folder_path.exists():
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        # Skip __pycache__
                        if '__pycache__' in str(file_path):
                            continue
                        arcname = file_path.relative_to(project_dir)
                        zf.write(file_path, arcname)
                        print(f"  Added: {arcname}")

        # Create empty directories that are needed
        zf.writestr("model/.gitkeep", "")
        zf.writestr("data/.gitkeep", "")
        zf.writestr("logs/tensorboard/.gitkeep", "")

    # Get file size
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"\nCreated: {output_zip}")
    print(f"Size: {size_mb:.2f} MB")
    print("\nUpload this file to Colab or Google Drive.")

if __name__ == "__main__":
    package_for_colab()
