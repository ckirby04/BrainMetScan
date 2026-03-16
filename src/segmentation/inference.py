"""
Inference script for brain metastasis segmentation
Handles full 3D volumes with sliding window approach
"""

import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import os

from unet import LightweightUNet3D
from dataset import BrainMetDataset
from tta import MinimalTTA, TestTimeAugmentation


@torch.no_grad()
def sliding_window_inference(
    model,
    image,
    window_size=(96, 96, 96),
    overlap=0.5,
    device='cuda',
    use_tta=False,
    tta_mode='minimal'
):
    """
    Perform inference on full 3D volume using sliding window

    Args:
        model: Segmentation model
        image: Input tensor (C, H, W, D)
        window_size: Size of sliding window
        overlap: Overlap ratio between windows
        device: Device to use
        use_tta: Whether to use Test-Time Augmentation
        tta_mode: 'minimal' for fast TTA, 'full' for comprehensive TTA

    Returns:
        Segmentation prediction (1, H, W, D)
    """
    model.eval()

    # Initialize TTA if requested
    tta_predictor = None
    if use_tta:
        if tta_mode == 'minimal':
            tta_predictor = MinimalTTA(model, device)
        else:
            tta_predictor = TestTimeAugmentation(model, device)

    C, H, W, D = image.shape
    wh, ww, wd = window_size

    # Calculate stride
    sh = int(wh * (1 - overlap))
    sw = int(ww * (1 - overlap))
    sd = int(wd * (1 - overlap))

    # Output and count arrays
    output = torch.zeros((1, H, W, D), device=device)
    count = torch.zeros((1, H, W, D), device=device)

    # Sliding window
    h_starts = list(range(0, H - wh + 1, sh)) + [H - wh]
    w_starts = list(range(0, W - ww + 1, sw)) + [W - ww]
    d_starts = list(range(0, D - wd + 1, sd)) + [D - wd]

    total_windows = len(h_starts) * len(w_starts) * len(d_starts)

    pbar = tqdm(total=total_windows, desc="Sliding window inference")

    for h_start in h_starts:
        for w_start in w_starts:
            for d_start in d_starts:
                # Extract window
                h_end = h_start + wh
                w_end = w_start + ww
                d_end = d_start + wd

                window = image[:, h_start:h_end, w_start:w_end, d_start:d_end]
                window = window.unsqueeze(0)  # Add batch dim

                # Predict with or without TTA
                if use_tta:
                    pred, _ = tta_predictor.predict(window, threshold=0.5)
                else:
                    window = window.to(device)
                    pred = model(window)
                    pred = torch.sigmoid(pred)

                # Aggregate
                output[:, h_start:h_end, w_start:w_end, d_start:d_end] += pred[0].cpu()
                count[:, h_start:h_end, w_start:w_end, d_start:d_end] += 1

                pbar.update(1)

    pbar.close()

    # Average predictions
    output = output / count

    return output


def predict_case(
    model,
    case_dir,
    sequences=['t1_pre', 't1_gd', 'flair', 'bravo'],
    window_size=(96, 96, 96),
    overlap=0.5,
    device='cuda',
    use_tta=False,
    tta_mode='minimal'
):
    """
    Predict segmentation for a single case

    Args:
        model: Segmentation model
        case_dir: Path to case directory
        sequences: List of sequences to load
        window_size: Sliding window size
        overlap: Overlap ratio
        device: Device to use
        use_tta: Whether to use Test-Time Augmentation
        tta_mode: 'minimal' for fast TTA, 'full' for comprehensive TTA

    Returns:
        prediction: Binary segmentation (H, W, D)
        affine: NIfTI affine matrix for saving
    """
    case_dir = Path(case_dir)

    # Load sequences
    images = []
    affine = None

    for seq in sequences:
        img_path = case_dir / f"{seq}.nii.gz"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing {seq} for case {case_dir.name}")

        nii = nib.load(str(img_path))
        img = nii.get_fdata().astype(np.float32)

        # Store affine from first image
        if affine is None:
            affine = nii.affine

        # Normalize
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std

        images.append(img)

    # Stack sequences (C, H, W, D)
    image = np.stack(images, axis=0)
    image = torch.from_numpy(image).float()

    # Predict
    with torch.no_grad():
        pred = sliding_window_inference(
            model, image, window_size, overlap, device, use_tta, tta_mode
        )

    # Convert to binary
    pred = (pred[0] > 0.5).float().cpu().numpy()

    return pred, affine


def main(args):
    """Main inference function"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get model config from checkpoint if available
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        base_channels = model_args.get('base_channels', 16)
        depth = model_args.get('depth', 3)
        dropout = model_args.get('dropout', 0.1)
    else:
        base_channels = args.base_channels
        depth = args.depth
        dropout = 0.0  # No dropout for inference

    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=base_channels,
        depth=depth,
        dropout_p=dropout
    ).to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation Dice: {checkpoint.get('val_dice', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get cases to process
    input_dir = Path(args.input_dir)
    if input_dir.is_dir():
        # Process all cases in directory
        cases = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('Mets_')])
    else:
        raise ValueError(f"Input directory not found: {input_dir}")

    print(f"\nProcessing {len(cases)} cases...")

    # Process each case
    for case_dir in cases:
        case_id = case_dir.name
        print(f"\nProcessing {case_id}...")

        try:
            # Predict
            pred, affine = predict_case(
                model,
                case_dir,
                sequences=['t1_pre', 't1_gd', 'flair', 'bravo'],
                window_size=tuple(args.window_size),
                overlap=args.overlap,
                device=device,
                use_tta=args.use_tta,
                tta_mode=args.tta_mode
            )

            # Save prediction
            output_path = Path(args.output_dir) / f"{case_id}_pred.nii.gz"
            nii_out = nib.Nifti1Image(pred.astype(np.float32), affine)
            nib.save(nii_out, str(output_path))

            print(f"✓ Saved to {output_path}")

            # Print statistics
            num_voxels = np.sum(pred > 0)
            print(f"  Detected metastasis voxels: {num_voxels}")

        except Exception as e:
            print(f"✗ Error processing {case_id}: {e}")

    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Predictions saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on brain MRI cases")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory containing cases')
    parser.add_argument('--output_dir', type=str, default='../../outputs/segmentation',
                        help='Output directory for predictions')

    parser.add_argument('--window_size', type=int, nargs=3, default=[96, 96, 96],
                        help='Sliding window size')
    parser.add_argument('--overlap', type=float, default=0.75,
                        help='Overlap ratio for sliding window (0.75 recommended for small lesions)')

    parser.add_argument('--base_channels', type=int, default=16,
                        help='Base channels (if not in checkpoint)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Model depth (if not in checkpoint)')

    parser.add_argument('--use_tta', action='store_true',
                        help='Enable Test-Time Augmentation for better accuracy (~10%% improvement)')
    parser.add_argument('--tta_mode', type=str, default='minimal', choices=['minimal', 'full'],
                        help='TTA mode: minimal (4 augmentations, faster) or full (10+ augmentations)')

    args = parser.parse_args()

    main(args)
