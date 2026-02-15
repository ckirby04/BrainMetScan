"""Save visualization images for the 14 zero-dice cases for analysis."""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'preprocessed_256' / 'train'
OUT_DIR = ROOT / 'analysis' / 'zero_dice_cases'
TARGET_SIZE = (128, 128, 128)
SEQUENCES = ['t1_pre', 't1_gd', 'flair', 't2']

ZERO_DICE_CASES = [
    'BMS_Mets_238', 'UCSF_100108D', 'UCSF_100132A', 'UCSF_100199A',
    'UCSF_100209A', 'UCSF_100216B', 'UCSF_100224A', 'UCSF_100238A',
    'UCSF_100238D', 'UCSF_100317A', 'UCSF_100318A', 'UCSF_100333A',
    'UCSF_100364A', 'UCSF_100414B',
]

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def load_and_resize(path):
    data = nib.load(str(path)).get_fdata().astype(np.float32)
    factors = [t / s for t, s in zip(TARGET_SIZE, data.shape)]
    return zoom(data, factors, order=1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {OUT_DIR}")

    for case_id in ZERO_DICE_CASES:
        case_dir = DATA_DIR / case_id
        if not case_dir.exists():
            print(f"  SKIP {case_id} - directory not found")
            continue

        # Load mask
        mask_path = case_dir / 'seg.nii.gz'
        if not mask_path.exists():
            print(f"  SKIP {case_id} - no mask")
            continue

        mask = load_and_resize(mask_path)
        mask = (mask > 0.5).astype(np.float32)

        # Find slice with most foreground voxels (axial = last axis)
        fg_per_slice = mask.sum(axis=(0, 1))  # sum over H, W for each D slice
        if fg_per_slice.max() > 0:
            best_slice = int(np.argmax(fg_per_slice))
            total_fg = int(mask.sum())
        else:
            best_slice = mask.shape[2] // 2
            total_fg = 0

        # Load sequences
        volumes = {}
        for seq in SEQUENCES:
            path = case_dir / f'{seq}.nii.gz'
            if path.exists():
                volumes[seq] = load_and_resize(path)
            else:
                volumes[seq] = np.zeros(TARGET_SIZE, dtype=np.float32)

        # Create figure: 2 rows x 4 cols
        # Row 1: raw sequences at best slice
        # Row 2: sequences with mask contour overlay
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'{case_id} | slice={best_slice} | total_fg_voxels={total_fg}',
                     fontsize=14, fontweight='bold')

        mask_slice = mask[:, :, best_slice]

        for j, seq in enumerate(SEQUENCES):
            img_slice = volumes[seq][:, :, best_slice]

            # Row 1: raw image
            axes[0, j].imshow(img_slice.T, cmap='gray', origin='lower')
            axes[0, j].set_title(seq, fontsize=12)
            axes[0, j].axis('off')

            # Row 2: image + mask overlay
            axes[1, j].imshow(img_slice.T, cmap='gray', origin='lower')
            if mask_slice.max() > 0:
                axes[1, j].contour(mask_slice.T, levels=[0.5], colors='red',
                                   linewidths=2, origin='lower')
                # Semi-transparent mask fill
                mask_rgb = np.zeros((*mask_slice.T.shape, 4))
                mask_rgb[mask_slice.T > 0.5] = [1, 0, 0, 0.3]
                axes[1, j].imshow(mask_rgb, origin='lower')
            axes[1, j].set_title(f'{seq} + mask', fontsize=12)
            axes[1, j].axis('off')

        plt.tight_layout()
        out_path = OUT_DIR / f'{case_id}.png'
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {case_id} (fg_voxels={total_fg}, best_slice={best_slice})")

    print(f"\nDone! {len(ZERO_DICE_CASES)} images saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
