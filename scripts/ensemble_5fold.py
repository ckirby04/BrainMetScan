"""
5-Fold nnU-Net Cross-Validation Evaluation.

Each case appears in exactly one fold's validation set. This aggregates all
folds to get a complete evaluation across all 566 cases.
"""

import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as ndimage_label

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'model'
NNUNET_BASE = ROOT / 'nnUNet' / 'nnUNet_results' / 'Dataset001_BrainMets'
DEFAULT_TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
LABELS_DIR = ROOT / 'nnUNet' / 'nnUNet_raw' / 'Dataset001_BrainMets' / 'labelsTr'


def voxel_dice(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin > 0)).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-8)


def voxel_sensitivity(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin > 0)).sum()
    if tp + fn == 0:
        return 1.0
    return float(tp) / float(tp + fn + 1e-8)


def voxel_precision(pred_bin, gt_bin):
    tp = ((pred_bin > 0) & (gt_bin > 0)).sum()
    fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
    if tp + fp == 0:
        return 1.0
    return float(tp) / float(tp + fp + 1e-8)


def lesion_detection(pred_bin, gt_bin, overlap_thresh=0.1):
    gt_labels, n_gt = ndimage_label(gt_bin)
    pred_labels, n_pred = ndimage_label(pred_bin)

    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0, {}

    # Count detected GT lesions + categorize by size
    tp_gt = 0
    size_stats = {'tiny': [0, 0], 'small': [0, 0],
                  'medium': [0, 0], 'large': [0, 0], 'huge': [0, 0]}

    for i in range(1, n_gt + 1):
        gt_mask = gt_labels == i
        vol = gt_mask.sum()
        # Classify: tiny<68, small<676, medium<6757, large<27027
        # (based on 0.739 mm^3 voxels: 50/500/5000/20000 mm^3)
        if vol < 68:
            cat = 'tiny'
        elif vol < 676:
            cat = 'small'
        elif vol < 6757:
            cat = 'medium'
        elif vol < 27027:
            cat = 'large'
        else:
            cat = 'huge'
        size_stats[cat][0] += 1  # total

        overlap = (gt_mask & (pred_bin > 0)).sum() / vol
        if overlap >= overlap_thresh:
            tp_gt += 1
        else:
            size_stats[cat][1] += 1  # missed
    fn = n_gt - tp_gt

    # Count valid pred lesions
    tp_pred = 0
    for i in range(1, n_pred + 1):
        pred_mask = pred_labels == i
        if pred_mask.sum() == 0:
            continue
        overlap = (pred_mask & (gt_bin > 0)).sum() / pred_mask.sum()
        if overlap >= overlap_thresh:
            tp_pred += 1
    fp = n_pred - tp_pred

    precision = tp_pred / (tp_pred + fp + 1e-8) if (tp_pred + fp) > 0 else 1.0
    recall = tp_gt / (tp_gt + fn + 1e-8) if (tp_gt + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, tp_gt, fp, fn, size_stats


def evaluate_all_folds(trainer_dir=DEFAULT_TRAINER):
    trainer_path = NNUNET_BASE / trainer_dir

    all_results = []
    all_lesion = {'precision': [], 'recall': [], 'f1': [],
                  'tp': 0, 'fp': 0, 'fn': 0}
    total_sizes = {'tiny': [0, 0], 'small': [0, 0],
                   'medium': [0, 0], 'large': [0, 0], 'huge': [0, 0]}
    fold_summaries = {}

    for fold in range(5):
        val_dir = trainer_path / f'fold_{fold}' / 'validation'
        if not val_dir.exists():
            print(f"  Fold {fold}: NO validation dir, skipping")
            continue

        pred_files = sorted(val_dir.glob('*.nii.gz'))
        fold_results = []
        t0 = time.time()

        for j, pred_file in enumerate(pred_files):
            case_id = pred_file.name.replace('.nii.gz', '')
            gt_file = LABELS_DIR / pred_file.name

            if not gt_file.exists():
                continue

            pred_data = nib.load(str(pred_file)).get_fdata().astype(np.uint8)
            gt_data = nib.load(str(gt_file)).get_fdata().astype(np.uint8)
            gt_bin = gt_data > 0
            pred_bin = pred_data > 0

            dice = voxel_dice(pred_bin, gt_bin)
            sens = voxel_sensitivity(pred_bin, gt_bin)
            prec = voxel_precision(pred_bin, gt_bin)
            l_prec, l_rec, l_f1, l_tp, l_fp, l_fn, sizes = lesion_detection(pred_bin, gt_bin)

            fold_results.append({
                'case_id': case_id, 'dice': dice,
                'sensitivity': sens, 'precision': prec, 'fold': fold,
            })

            all_lesion['precision'].append(l_prec)
            all_lesion['recall'].append(l_rec)
            all_lesion['f1'].append(l_f1)
            all_lesion['tp'] += l_tp
            all_lesion['fp'] += l_fp
            all_lesion['fn'] += l_fn

            for cat in total_sizes:
                total_sizes[cat][0] += sizes[cat][0]
                total_sizes[cat][1] += sizes[cat][1]

            if (j + 1) % 30 == 0:
                print(f"    [{j+1}/{len(pred_files)}]...")

        elapsed = time.time() - t0
        dices = [r['dice'] for r in fold_results]
        fold_summaries[fold] = {
            'cases': len(fold_results),
            'mean_dice': round(float(np.mean(dices)), 4),
            'median_dice': round(float(np.median(dices)), 4),
            'std_dice': round(float(np.std(dices)), 4),
        }
        all_results.extend(fold_results)
        print(f"  Fold {fold}: {len(fold_results)} cases, "
              f"mean Dice={np.mean(dices):.4f}, median={np.median(dices):.4f} "
              f"({elapsed:.0f}s)")

    # Aggregate
    all_dices = [r['dice'] for r in all_results]
    all_sens = [r['sensitivity'] for r in all_results]
    all_prec = [r['precision'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS-VALIDATION RESULTS ({len(all_results)} cases)")
    print(f"{'='*60}")
    print(f"  Mean Dice:        {np.mean(all_dices):.4f} (+/- {np.std(all_dices):.4f})")
    print(f"  Median Dice:      {np.median(all_dices):.4f}")
    print(f"  Mean Sensitivity: {np.mean(all_sens):.4f}")
    print(f"  Mean Precision:   {np.mean(all_prec):.4f}")

    # Lesion detection
    total_tp = all_lesion['tp']
    total_fp = all_lesion['fp']
    total_fn = all_lesion['fn']
    overall_rec = total_tp / (total_tp + total_fn + 1e-8)
    overall_prec = total_tp / (total_tp + total_fp + 1e-8)
    overall_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)

    print(f"\n  Lesion Detection:")
    print(f"    Precision: {overall_prec:.4f} ({total_tp} TP, {total_fp} FP)")
    print(f"    Recall:    {overall_rec:.4f} ({total_tp} detected, {total_fn} missed)")
    print(f"    F1:        {overall_f1:.4f}")

    print(f"\n  Missed Lesions by Size:")
    for cat in ['tiny', 'small', 'medium', 'large', 'huge']:
        total, missed = total_sizes[cat]
        rate = missed / total * 100 if total > 0 else 0
        det_rate = (total - missed) / total * 100 if total > 0 else 100
        print(f"    {cat:>8}: {total - missed}/{total} detected ({det_rate:.1f}%), "
              f"{missed} missed ({rate:.1f}%)")

    print(f"\n  Per-Fold Breakdown:")
    for fold in sorted(fold_summaries):
        s = fold_summaries[fold]
        print(f"    Fold {fold}: {s['cases']} cases, "
              f"Dice={s['mean_dice']:.4f} +/- {s['std_dice']:.4f}")

    print(f"\n  Comparison:")
    print(f"    Fold 0 alone:       0.7596 mean Dice (114 cases)")
    print(f"    5-fold aggregate:   {np.mean(all_dices):.4f} mean Dice ({len(all_results)} cases)")
    print(f"    Custom ensemble:    0.7475 mean Dice (84 cases)")
    print(f"    Human baseline:     ~0.85 median Dice")

    # Save
    output = {
        'method': '5-fold cross-validation (nnU-Net PlainConvUNet)',
        'total_cases': len(all_results),
        'aggregate': {
            'mean_dice': round(float(np.mean(all_dices)), 4),
            'median_dice': round(float(np.median(all_dices)), 4),
            'std_dice': round(float(np.std(all_dices)), 4),
            'mean_sensitivity': round(float(np.mean(all_sens)), 4),
            'mean_precision': round(float(np.mean(all_prec)), 4),
            'lesion_detection': {
                'precision': round(float(overall_prec), 4),
                'recall': round(float(overall_rec), 4),
                'f1': round(float(overall_f1), 4),
                'tp': int(total_tp), 'fp': int(total_fp), 'fn': int(total_fn),
            },
            'missed_by_size': {
                cat: {'total': total_sizes[cat][0], 'missed': total_sizes[cat][1]}
                for cat in ['tiny', 'small', 'medium', 'large', 'huge']
            },
        },
        'per_fold': fold_summaries,
        'per_case': [
            {'case_id': r['case_id'], 'fold': r['fold'],
             'dice': round(r['dice'], 4),
             'sensitivity': round(r['sensitivity'], 4),
             'precision': round(r['precision'], 4)}
            for r in all_results
        ],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = OUTPUT_DIR / 'nnunet_5fold_evaluation.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    return output


if __name__ == '__main__':
    print("5-Fold nnU-Net Cross-Validation Evaluation")
    print("=" * 60)
    evaluate_all_folds()
