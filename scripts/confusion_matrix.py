"""Generate confusion matrices for smart ensemble at different thresholds."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from segmentation.dataset import BrainMetDataset
from segmentation.unet import LightweightUNet3D


def main():
    print('Loading models...')
    device = torch.device('cuda')
    model_dir = Path(__file__).parent.parent / 'model'
    output_dir = Path(__file__).parent.parent / 'outputs'

    models = []
    patch_sizes = []
    names = ['exp3_12patch_maxfn', 'exp1_8patch', 'improved_24patch', 'improved_36patch']
    patches = [12, 8, 24, 36]

    for name, ps in zip(names, patches):
        path = model_dir / f'{name}_best.pth'
        if path.exists():
            model = LightweightUNet3D(in_channels=4, out_channels=1, use_attention=True, use_residual=True)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
            patch_sizes.append(ps)
            print(f'  Loaded {name}')

    print('Loading validation data...')
    data_dir = Path(__file__).parent.parent / 'data' / 'preprocessed_256' / 'train'
    ds = BrainMetDataset(
        data_dir=str(data_dir),
        sequences=['t1_pre', 't1_gd', 'flair', 't2'],
        patch_size=(24, 24, 24),
        target_size=None,
        transform=None
    )

    images, masks = [], []
    for i in tqdm(range(min(80, len(ds))), desc='Loading'):
        for _ in range(3):
            img, mask, _ = ds[i]
            images.append(img)
            masks.append(mask)
        if (i + 1) % 25 == 0:
            gc.collect()

    images = torch.stack(images)
    masks = torch.stack(masks)
    print(f'Loaded {len(images)} samples')

    print('Getting ensemble predictions (union fusion)...')
    all_preds = []
    loader = DataLoader(TensorDataset(images), batch_size=8, shuffle=False)

    for (batch,) in tqdm(loader, desc='Predicting'):
        batch = batch.to(device)
        preds = []
        for model, ps in zip(models, patch_sizes):
            if ps != 24:
                x = F.interpolate(batch, size=(ps, ps, ps), mode='trilinear', align_corners=False)
                with torch.no_grad():
                    p = torch.sigmoid(model(x))
                p = F.interpolate(p, size=(24, 24, 24), mode='trilinear', align_corners=False)
            else:
                with torch.no_grad():
                    p = torch.sigmoid(model(batch))
            preds.append(p)
        # Union (max across models)
        ensemble_pred = torch.stack(preds).max(dim=0)[0]
        all_preds.append(ensemble_pred.cpu())

    all_preds = torch.cat(all_preds)
    masks_cpu = masks.cpu()

    # Generate confusion matrices
    thresholds = [0.4, 0.5, 0.6, 0.7]

    print('\nGenerating confusion matrices...')
    print('=' * 60)

    for thresh in thresholds:
        pred_binary = (all_preds > thresh).float()

        TP = int(((pred_binary == 1) & (masks_cpu == 1)).sum().item())
        TN = int(((pred_binary == 0) & (masks_cpu == 0)).sum().item())
        FP = int(((pred_binary == 1) & (masks_cpu == 0)).sum().item())
        FN = int(((pred_binary == 0) & (masks_cpu == 1)).sum().item())

        cm = np.array([[TN, FP], [FN, TP]])

        sensitivity = TP / (TP + FN) * 100
        specificity = TN / (TN + FP) * 100
        precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        # Plot confusion matrix using matplotlib only
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap manually
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax, label='Count')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = 'white' if val > cm.max() / 2 else 'black'
                ax.text(j, i, f'{val:,}', ha='center', va='center',
                       fontsize=18, fontweight='bold', color=color)

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nNegative', 'Predicted\nPositive'], fontsize=11)
        ax.set_yticklabels(['Actual\nNegative', 'Actual\nPositive'], fontsize=11)

        title = f'Confusion Matrix - Threshold {thresh}\n'
        title += f'Sensitivity: {sensitivity:.1f}% | Specificity: {specificity:.1f}% | Precision: {precision:.1f}%'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')

        # Add summary box
        summary = f'TP: {TP:,}\nFP: {FP:,}\nTN: {TN:,}\nFN: {FN:,}\n\nF1: {f1:.1f}%'
        props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9)
        ax.text(1.25, 0.5, summary, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', bbox=props, family='monospace')

        plt.tight_layout()
        save_path = output_dir / f'confusion_matrix_thresh_{thresh}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f'\nThreshold {thresh}:')
        print(f'  Sensitivity: {sensitivity:.1f}%')
        print(f'  Specificity: {specificity:.1f}%')
        print(f'  Precision:   {precision:.1f}%')
        print(f'  F1 Score:    {f1:.1f}%')
        print(f'  TP={TP:,}  FP={FP:,}  TN={TN:,}  FN={FN:,}')
        print(f'  Saved: {save_path}')

    print('\n' + '=' * 60)
    print('Done! Confusion matrices saved to outputs/')


if __name__ == '__main__':
    main()
