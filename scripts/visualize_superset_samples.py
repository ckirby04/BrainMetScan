"""
Visualize sample images from each dataset in the superset.
Shows one representative image from:
1. BrainMetShare (BMS) - Supervised training data
2. UCSF Brain Metastases - Supervised training data
3. Yale Brain Mets Longitudinal - Unlabeled pretraining data
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import nibabel as nib

# Paths
BASE_DIR = r"C:\Users\Clark\TalentAccelorator\brainMetShare\1.2"
PARENT_DIR = os.path.dirname(BASE_DIR)
SUPERSET_DIR = os.path.join(PARENT_DIR, "Superset")
TRAIN_DIR = os.path.join(SUPERSET_DIR, "full", "train")
PRETRAINING_DIR = os.path.join(SUPERSET_DIR, "pretraining")
OUTPUT_PATH = os.path.join(BASE_DIR, "superset_visualization.png")

# Dataset samples to visualize
SAMPLES = {
    "BrainMetShare (BMS)": {
        "path": os.path.join(TRAIN_DIR, "BMS_Mets_005"),
        "type": "Supervised Training\n(with segmentation masks)",
        "source": "Multi-center brain metastasis dataset",
        "has_seg": True
    },
    "UCSF Brain Metastases": {
        "path": os.path.join(TRAIN_DIR, "UCSF_100101A"),
        "type": "Supervised Training\n(with segmentation masks)",
        "source": "UCSF radiology archive",
        "has_seg": True
    },
    "Yale Brain Mets Longitudinal": {
        "path": os.path.join(PRETRAINING_DIR, "Yale_YG_01M98EKKAR50_2016-11-13"),
        "type": "Unlabeled Pretraining\n(no segmentation)",
        "source": "Yale longitudinal follow-up study",
        "has_seg": False
    }
}


def load_nifti(filepath):
    """Load a NIfTI file and return the data array."""
    if os.path.exists(filepath):
        img = nib.load(filepath)
        return img.get_fdata()
    return None


def get_middle_slice(volume, axis=2):
    """Get the middle slice along specified axis."""
    if volume is None:
        return None
    mid = volume.shape[axis] // 2
    if axis == 0:
        return volume[mid, :, :]
    elif axis == 1:
        return volume[:, mid, :]
    else:
        return volume[:, :, mid]


def normalize_slice(slice_data):
    """Normalize slice to 0-1 range for visualization."""
    if slice_data is None:
        return None
    vmin, vmax = np.percentile(slice_data, [1, 99])
    if vmax > vmin:
        normalized = (slice_data - vmin) / (vmax - vmin)
        return np.clip(normalized, 0, 1)
    return slice_data


def main():
    # Create figure with subplots
    # 3 datasets x 5 modalities (t1_pre, t1_gd, flair, t2, seg)
    fig, axes = plt.subplots(3, 5, figsize=(18, 12))

    modalities = ['t1_pre', 't1_gd', 'flair', 't2', 'seg']
    modality_labels = ['T1 Pre-contrast', 'T1 Post-Gd', 'FLAIR', 'T2-weighted', 'Segmentation']

    # Color scheme for dataset types
    colors = {
        'BrainMetShare (BMS)': '#2E86AB',  # Blue
        'UCSF Brain Metastases': '#A23B72',  # Purple
        'Yale Brain Mets Longitudinal': '#F18F01'  # Orange
    }

    for row_idx, (dataset_name, info) in enumerate(SAMPLES.items()):
        case_path = info['path']

        for col_idx, (modality, mod_label) in enumerate(zip(modalities, modality_labels)):
            ax = axes[row_idx, col_idx]

            # Load and display the volume
            nii_path = os.path.join(case_path, f"{modality}.nii.gz")
            volume = load_nifti(nii_path)

            if volume is not None:
                slice_data = get_middle_slice(volume)
                slice_norm = normalize_slice(slice_data)

                if modality == 'seg':
                    # Use categorical colormap for segmentation
                    ax.imshow(slice_data.T, origin='lower', cmap='hot', interpolation='nearest')
                else:
                    ax.imshow(slice_norm.T, origin='lower', cmap='gray', interpolation='bilinear')
            else:
                # No data available (e.g., Yale has no segmentation)
                ax.text(0.5, 0.5, 'N/A\n(Unlabeled)',
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes, color='gray')
                ax.set_facecolor('#f0f0f0')

            ax.axis('off')

            # Add modality labels to top row
            if row_idx == 0:
                ax.set_title(mod_label, fontsize=12, fontweight='bold', pad=10)

        # Add dataset label on the left
        axes[row_idx, 0].text(-0.15, 0.5,
                              f"{dataset_name}\n\n{info['type']}",
                              transform=axes[row_idx, 0].transAxes,
                              fontsize=11, fontweight='bold',
                              va='center', ha='right',
                              color=colors[dataset_name],
                              bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white',
                                       edgecolor=colors[dataset_name],
                                       alpha=0.9))

    # Add overall title
    fig.suptitle('Brain Metastasis Superset: Sample Images from Each Dataset',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add subtitle with dataset info
    subtitle = (f"Total: 566 supervised cases (BMS: 105 + UCSF: 461) + 1,430 pretraining cases (Yale)\n"
                f"Image format: 3D NIfTI (.nii.gz) | 4 MRI sequences per case")
    fig.text(0.5, 0.93, subtitle, ha='center', fontsize=10, style='italic', color='gray')

    # Add legend
    legend_text = (
        "Dataset Types:\n"
        "  BrainMetShare (BMS): Public multi-center dataset for brain metastasis segmentation\n"
        "  UCSF: Large clinical archive from UCSF radiology\n"
        "  Yale: Longitudinal follow-up study (unlabeled, used for self-supervised pretraining)"
    )
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=9,
             family='monospace', color='#333333',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', edgecolor='#cccccc'))

    plt.tight_layout(rect=[0.12, 0.08, 1, 0.92])

    # Save figure
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {OUTPUT_PATH}")
    plt.close()

    return OUTPUT_PATH


if __name__ == "__main__":
    main()
