"""
Comprehensive Model Evaluation Script
Tests the best model on the entire training dataset and generates a report of failed cases
"""

import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

import sys
sys.path.append('src/segmentation')
from unet import LightweightUNet3D
from enhanced_unet import DeepSupervisedUNet3D
from dataset import BrainMetDataset


def dice_coefficient(pred, target, threshold=0.5):
    """Calculate Dice coefficient between prediction and target"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # Apply threshold to prediction
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > 0).astype(np.float32)

    # Calculate dice
    intersection = np.sum(pred_binary * target_binary)
    dice = (2. * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)

    return dice


def load_model(checkpoint_path, device='cuda'):
    """Load the best model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint (check both top-level and args dict)
    if 'args' in checkpoint and checkpoint['args']:
        model_args = checkpoint['args']
    else:
        model_args = checkpoint

    base_channels = model_args.get('base_channels', 16)
    depth = model_args.get('depth', 3)
    dropout = model_args.get('dropout', 0.1)
    use_attention = model_args.get('use_attention', False)
    use_residual = model_args.get('use_residual', False)
    model_type = model_args.get('model_type', 'lightweight')
    use_deep_supervision = model_args.get('use_deep_supervision', True if model_type == 'deep_supervised' else False)

    print(f"Model config: base_channels={base_channels}, depth={depth}, dropout={dropout}")
    print(f"              model_type={model_type}, deep_supervision={use_deep_supervision}")

    # Create model based on type
    if model_type == 'deep_supervised':
        model = DeepSupervisedUNet3D(
            in_channels=4,
            out_channels=1,
            base_channels=base_channels,
            depth=depth,
            dropout_p=dropout,
            deep_supervision=use_deep_supervision
        ).to(device)
    else:
        model = LightweightUNet3D(
            in_channels=4,
            out_channels=1,
            base_channels=base_channels,
            depth=depth,
            dropout_p=dropout,
            use_attention=use_attention,
            use_residual=use_residual
        ).to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation Dice: {checkpoint.get('val_dice', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


@torch.no_grad()
def predict_full_volume(model, images, device='cuda', patch_size=(96, 96, 96), overlap=0.5):
    """
    Predict on full volume using sliding window

    Args:
        model: The segmentation model
        images: Input images (C, H, W, D) as numpy array
        device: Device to use
        patch_size: Size of sliding window
        overlap: Overlap ratio

    Returns:
        prediction: Full volume prediction (H, W, D)
    """
    model.eval()

    # Convert to torch tensor
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()

    C, H, W, D = images.shape
    wh, ww, wd = patch_size

    # Calculate stride
    sh = int(wh * (1 - overlap))
    sw = int(ww * (1 - overlap))
    sd = int(wd * (1 - overlap))

    # Output and count arrays
    output = torch.zeros((1, H, W, D))
    count = torch.zeros((1, H, W, D))

    # Sliding window
    h_starts = list(range(0, max(1, H - wh + 1), sh))
    if H > wh and h_starts[-1] + wh < H:
        h_starts.append(H - wh)

    w_starts = list(range(0, max(1, W - ww + 1), sw))
    if W > ww and w_starts[-1] + ww < W:
        w_starts.append(W - ww)

    d_starts = list(range(0, max(1, D - wd + 1), sd))
    if D > wd and d_starts[-1] + wd < D:
        d_starts.append(D - wd)

    # Process each window
    for h_start in h_starts:
        for w_start in w_starts:
            for d_start in d_starts:
                # Extract window
                h_end = min(h_start + wh, H)
                w_end = min(w_start + ww, W)
                d_end = min(d_start + wd, D)

                # Adjust starts if at boundary
                h_start_adj = max(0, h_end - wh)
                w_start_adj = max(0, w_end - ww)
                d_start_adj = max(0, d_end - wd)

                window = images[:, h_start_adj:h_start_adj+wh, w_start_adj:w_start_adj+ww, d_start_adj:d_start_adj+wd]
                window = window.unsqueeze(0).to(device)  # Add batch dim

                # Predict
                pred = model(window)
                # Handle deep supervision - take the final output
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]  # Main output is first
                pred = torch.sigmoid(pred)

                # Aggregate
                output[:, h_start_adj:h_start_adj+wh, w_start_adj:w_start_adj+ww, d_start_adj:d_start_adj+wd] += pred[0].cpu()
                count[:, h_start_adj:h_start_adj+wh, w_start_adj:w_start_adj+ww, d_start_adj:d_start_adj+wd] += 1

    # Average predictions
    output = output / (count + 1e-8)

    return output[0].numpy()


def load_case_data(case_dir, sequences=['t1_pre', 't1_gd', 'flair', 'bravo']):
    """Load a single case's images and mask"""
    images = []

    for seq in sequences:
        img_path = case_dir / f"{seq}.nii.gz"
        nii = nib.load(str(img_path))
        img = nii.get_fdata().astype(np.float32)

        # Normalize
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std

        images.append(img)

    # Stack sequences (C, H, W, D)
    images = np.stack(images, axis=0)

    # Load mask
    mask_path = case_dir / "seg.nii.gz"
    nii = nib.load(str(mask_path))
    mask = nii.get_fdata().astype(np.float32)
    mask = (mask > 0).astype(np.float32)

    return images, mask


def create_visualization(case_id, images, mask, prediction, dice_score, slice_idx=None):
    """Create a visualization comparing mask and prediction"""
    # Select middle slice if not specified
    if slice_idx is None:
        # Find slice with most mask content
        mask_sums = np.sum(mask, axis=(0, 1))
        slice_idx = np.argmax(mask_sums)

    # Get the T1-GD sequence (index 1) for visualization
    img_slice = images[1, :, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]
    pred_slice = prediction[:, :, slice_idx]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    axes[0].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('T1-GD Image')
    axes[0].axis('off')

    # Ground truth mask
    axes[1].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[1].imshow(mask_slice.T, cmap='Reds', alpha=0.5, origin='lower')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[2].imshow(pred_slice.T, cmap='Blues', alpha=0.5, origin='lower', vmin=0, vmax=1)
    axes[2].set_title('Model Prediction')
    axes[2].axis('off')

    # Overlay comparison
    axes[3].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[3].imshow(mask_slice.T, cmap='Reds', alpha=0.3, origin='lower')
    axes[3].imshow(pred_slice.T, cmap='Blues', alpha=0.3, origin='lower', vmin=0, vmax=1)
    axes[3].set_title('Overlay (Red=GT, Blue=Pred)')
    axes[3].axis('off')

    fig.suptitle(f'{case_id} - Dice Score: {dice_score:.4f} - Slice: {slice_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='../../models/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation')
    args = parser.parse_args()

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Paths
    checkpoint_path = args.checkpoint
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(checkpoint_path, device)

    # Get all cases
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Mets_')])
    print(f"\nFound {len(cases)} cases to evaluate\n")

    # Evaluation results
    results = []
    failed_cases = []

    # Evaluate each case
    print("Evaluating all cases...")
    for case_dir in tqdm(cases):
        case_id = case_dir.name

        try:
            # Load data
            images, mask = load_case_data(case_dir)

            # Predict
            prediction = predict_full_volume(model, images, device, patch_size=(96, 96, 96), overlap=0.5)

            # Calculate dice
            dice = dice_coefficient(prediction, mask, threshold=0.5)

            # Determine success
            success = "yes" if dice > 0.5 else "no"

            # Store results
            result = {
                'case_id': case_id,
                'dice_score': dice,
                'success': success
            }
            results.append(result)

            # Store failed cases for visualization
            if dice <= 0.5:
                failed_cases.append({
                    'case_id': case_id,
                    'images': images,
                    'mask': mask,
                    'prediction': prediction,
                    'dice_score': dice
                })

            print(f"{case_id}: Dice={dice:.4f} ({success})")

        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            results.append({
                'case_id': case_id,
                'dice_score': 0.0,
                'success': 'error'
            })

    # Calculate overall accuracy
    successful_cases = [r for r in results if r['success'] == 'yes']
    accuracy = len(successful_cases) / len(results) * 100
    avg_dice = np.mean([r['dice_score'] for r in results])

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total cases: {len(results)}")
    print(f"Successful (Dice > 0.5): {len(successful_cases)} ({accuracy:.2f}%)")
    print(f"Failed (Dice <= 0.5): {len(failed_cases)}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"{'='*60}\n")

    # Save results to CSV
    import csv
    csv_path = output_dir / 'evaluation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'dice_score', 'success'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {csv_path}")

    # Generate PDF report of failed cases
    if len(failed_cases) > 0:
        print(f"\nGenerating PDF report of {len(failed_cases)} failed cases...")
        pdf_path = output_dir / 'failed_cases_report.pdf'

        with PdfPages(pdf_path) as pdf:
            for failed_case in tqdm(failed_cases):
                fig = create_visualization(
                    failed_case['case_id'],
                    failed_case['images'],
                    failed_case['mask'],
                    failed_case['prediction'],
                    failed_case['dice_score']
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"Failed cases report saved to: {pdf_path}")
    else:
        print("\nNo failed cases to visualize!")

    # Summary text file
    summary_path = output_dir / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Total cases: {len(results)}\n")
        f.write(f"Successful (Dice > 0.5): {len(successful_cases)} ({accuracy:.2f}%)\n")
        f.write(f"Failed (Dice <= 0.5): {len(failed_cases)}\n")
        f.write(f"Average Dice Score: {avg_dice:.4f}\n\n")
        f.write("="*60 + "\n\n")
        f.write("DETAILED RESULTS:\n\n")
        for r in sorted(results, key=lambda x: x['dice_score']):
            f.write(f"{r['case_id']}: Dice={r['dice_score']:.4f} ({r['success']})\n")

    print(f"Summary saved to: {summary_path}")
    print("\nEvaluation complete!")


def evaluate_per_lesion_by_size(
    model,
    data_dir,
    device='cuda',
    patch_size=(96, 96, 96),
    overlap=0.5,
    threshold=0.5,
    size_buckets=None,
):
    """
    Evaluate lesion-level sensitivity broken down by size buckets.

    Args:
        model: Segmentation model
        data_dir: Path to data directory
        device: Device to use
        patch_size: Sliding window patch size
        overlap: Sliding window overlap
        threshold: Binary threshold
        size_buckets: Dict of bucket_name -> (min_voxels, max_voxels).
                      Default: tiny <500, small <2000, medium <5000, large >=5000

    Returns:
        Dict with per-bucket sensitivity and counts
    """
    from scipy import ndimage

    if size_buckets is None:
        size_buckets = {
            'tiny': (0, 500),
            'small': (500, 2000),
            'medium': (2000, 5000),
            'large': (5000, float('inf')),
        }

    data_dir = Path(data_dir)
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Mets_')])

    bucket_stats = {name: {'total': 0, 'detected': 0} for name in size_buckets}

    for case_dir in tqdm(cases, desc="Per-lesion evaluation"):
        try:
            images, mask = load_case_data(case_dir)
            prediction = predict_full_volume(model, images, device, patch_size, overlap)
            pred_binary = (prediction > threshold).astype(np.float32)

            # Label ground truth lesions
            gt_labels, n_gt = ndimage.label(mask)

            for i in range(1, n_gt + 1):
                lesion_mask = (gt_labels == i)
                lesion_size = lesion_mask.sum()

                # Determine bucket
                for bucket_name, (min_v, max_v) in size_buckets.items():
                    if min_v <= lesion_size < max_v:
                        bucket_stats[bucket_name]['total'] += 1
                        # Detected if >= 30% IoU overlap
                        overlap_voxels = (pred_binary * lesion_mask).sum()
                        if overlap_voxels > 0.3 * lesion_size:
                            bucket_stats[bucket_name]['detected'] += 1
                        break

        except Exception:
            continue

    # Compute sensitivity per bucket
    results = {}
    for name, stats in bucket_stats.items():
        total = stats['total']
        detected = stats['detected']
        sensitivity = detected / total if total > 0 else 0.0
        results[name] = {
            'total_lesions': total,
            'detected_lesions': detected,
            'sensitivity': round(sensitivity, 4),
        }

    return results


def generate_validation_report(
    model,
    data_dir,
    output_path,
    device='cuda',
    threshold=0.5,
):
    """
    Generate a formatted validation report with size-stratified metrics.

    Args:
        model: Segmentation model
        data_dir: Path to data directory
        output_path: Path for output text file
        device: Device to use
        threshold: Binary threshold
    """
    size_results = evaluate_per_lesion_by_size(model, data_dir, device, threshold=threshold)

    with open(output_path, 'w') as f:
        f.write("SIZE-STRATIFIED VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Bucket':<12} {'Total':>8} {'Detected':>10} {'Sensitivity':>12}\n")
        f.write("-" * 42 + "\n")

        total_all = 0
        detected_all = 0
        for name, stats in size_results.items():
            f.write(f"{name:<12} {stats['total_lesions']:>8} {stats['detected_lesions']:>10} "
                    f"{stats['sensitivity']*100:>11.1f}%\n")
            total_all += stats['total_lesions']
            detected_all += stats['detected_lesions']

        overall = detected_all / total_all if total_all > 0 else 0
        f.write("-" * 42 + "\n")
        f.write(f"{'OVERALL':<12} {total_all:>8} {detected_all:>10} {overall*100:>11.1f}%\n")

    print(f"Validation report saved to {output_path}")
    return size_results


if __name__ == "__main__":
    main()
