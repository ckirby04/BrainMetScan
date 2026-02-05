import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

sys.path.append('src/segmentation')
from unet import LightweightUNet3D

@torch.no_grad()
def sliding_window_inference(volume, model, window_size=(96, 96, 96), overlap=0.5, device='cuda'):
    """
    Perform inference on full 3D volume using sliding window

    Args:
        volume: Input tensor (1, C, H, W, D)
        model: Segmentation model
        window_size: Size of sliding window
        overlap: Overlap fraction between windows
        device: Device to run inference on

    Returns:
        Prediction tensor (1, 1, H, W, D)
    """
    model.eval()

    _, C, H, W, D = volume.shape
    wh, ww, wd = window_size

    # Calculate stride from overlap
    stride_h = int(wh * (1 - overlap))
    stride_w = int(ww * (1 - overlap))
    stride_d = int(wd * (1 - overlap))

    # Initialize output and count arrays
    prediction = torch.zeros((1, 1, H, W, D), device=device)
    count = torch.zeros((1, 1, H, W, D), device=device)

    # Slide window over volume
    for h in range(0, H - wh + 1, stride_h):
        for w in range(0, W - ww + 1, stride_w):
            for d in range(0, D - wd + 1, stride_d):
                # Extract window
                window = volume[:, :, h:h+wh, w:w+ww, d:d+wd]

                # Run inference
                output = model(window)
                output = torch.sigmoid(output)

                # Accumulate
                prediction[:, :, h:h+wh, w:w+ww, d:d+wd] += output
                count[:, :, h:h+wh, w:w+ww, d:d+wd] += 1

    # Average overlapping predictions
    prediction = prediction / torch.clamp(count, min=1)

    return prediction

def load_nifti(path):
    """Load NIfTI file and return data array"""
    nii = nib.load(str(path))
    return nii.get_fdata()

def normalize_volume(volume):
    """Z-score normalization"""
    mean = np.mean(volume)
    std = np.std(volume)
    if std > 0:
        return (volume - mean) / std
    return volume - mean

def visualize_segmentation(case_dir, model_path, device='cuda', output_file='segmentation_visualization.png'):
    """
    Visualize MRI with ground truth and predicted segmentation overlays

    Args:
        case_dir: Path to case directory containing MRI sequences and segmentation
        model_path: Path to trained model checkpoint
        device: 'cuda' or 'cpu'
        output_file: Output image file path
    """
    case_dir = Path(case_dir)

    print(f"Loading data from {case_dir}...")

    # Load MRI sequences
    t1_gd = load_nifti(case_dir / 't1_gd.nii.gz')
    t1_pre = load_nifti(case_dir / 't1_pre.nii.gz')
    flair = load_nifti(case_dir / 'flair.nii.gz')
    bravo = load_nifti(case_dir / 'bravo.nii.gz')

    # Load ground truth segmentation
    seg_gt = load_nifti(case_dir / 'seg.nii.gz')

    print(f"MRI shape: {t1_gd.shape}")
    print(f"Ground truth segmentation shape: {seg_gt.shape}")
    print(f"Number of lesion voxels (GT): {np.sum(seg_gt > 0)}")

    # Normalize sequences
    t1_gd_norm = normalize_volume(t1_gd)
    t1_pre_norm = normalize_volume(t1_pre)
    flair_norm = normalize_volume(flair)
    bravo_norm = normalize_volume(bravo)

    # Stack into 4-channel input
    volume = np.stack([t1_pre_norm, t1_gd_norm, flair_norm, bravo_norm], axis=0)
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)  # (1, 4, H, W, D)

    # Load model
    print(f"\nLoading model from {model_path}...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = LightweightUNet3D(in_channels=4, out_channels=1, base_channels=16, depth=3)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Run inference
    print("Running inference with sliding window...")
    with torch.no_grad():
        volume_tensor = volume_tensor.to(device)
        pred = sliding_window_inference(
            volume_tensor,
            model,
            window_size=(96, 96, 96),
            overlap=0.5,
            device=device
        )

    pred_np = pred.cpu().numpy()[0, 0]  # (H, W, D)
    pred_binary = (pred_np > 0.5).astype(np.uint8)

    print(f"Number of lesion voxels (Predicted): {np.sum(pred_binary > 0)}")

    # Find slices with lesions
    lesion_slices_gt = np.where(np.sum(seg_gt, axis=(0, 1)) > 0)[0]
    lesion_slices_pred = np.where(np.sum(pred_binary, axis=(0, 1)) > 0)[0]

    if len(lesion_slices_gt) > 0:
        # Select slices: beginning, middle, end of lesion region
        slice_indices = [
            lesion_slices_gt[0],
            lesion_slices_gt[len(lesion_slices_gt)//2],
            lesion_slices_gt[-1]
        ]
    else:
        # No lesions in GT, use middle slices
        mid_slice = t1_gd.shape[2] // 2
        slice_indices = [mid_slice - 10, mid_slice, mid_slice + 10]

    print(f"\nVisualizing slices: {slice_indices}")

    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for row, slice_idx in enumerate(slice_indices):
        # Column 1: MRI only (T1-gd)
        ax = axes[row, 0]
        ax.imshow(t1_gd[:, :, slice_idx].T, cmap='gray', origin='lower')
        ax.set_title(f'T1-Gd Slice {slice_idx}', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Column 2: MRI with Ground Truth overlay
        ax = axes[row, 1]
        ax.imshow(t1_gd[:, :, slice_idx].T, cmap='gray', origin='lower')

        # Overlay ground truth in red
        gt_slice = seg_gt[:, :, slice_idx].T
        masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
        ax.imshow(masked_gt, cmap='Reds', alpha=0.5, origin='lower', vmin=0, vmax=1)
        ax.set_title(f'Ground Truth Overlay', fontsize=12, fontweight='bold', color='red')
        ax.axis('off')

        # Column 3: MRI with Prediction overlay
        ax = axes[row, 2]
        ax.imshow(t1_gd[:, :, slice_idx].T, cmap='gray', origin='lower')

        # Overlay prediction in green
        pred_slice = pred_binary[:, :, slice_idx].T
        masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
        ax.imshow(masked_pred, cmap='Greens', alpha=0.5, origin='lower', vmin=0, vmax=1)
        ax.set_title(f'Model Prediction Overlay', fontsize=12, fontweight='bold', color='green')
        ax.axis('off')

    # Add overall title
    case_name = case_dir.name
    num_lesions_gt = np.sum(seg_gt > 0)
    num_lesions_pred = np.sum(pred_binary > 0)

    fig.suptitle(
        f'Brain Metastasis Segmentation: {case_name}\n' +
        f'Ground Truth: {num_lesions_gt} voxels | Prediction: {num_lesions_pred} voxels',
        fontsize=16, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    # Display the plot
    plt.show()

    return fig

if __name__ == '__main__':
    # Configuration
    case_dir = 'train/Mets_005'
    model_path = '../../models/best_model.pth'
    output_file = 'segmentation_visualization.png'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Case: {case_dir}")
    print(f"Model: {model_path}\n")

    visualize_segmentation(case_dir, model_path, device, output_file)
