# BrainMetScan

> Automated brain metastasis segmentation from multi-modal MRI using a
> multi-scale 3D U-Net ensemble trained on publicly available data.

## Clinical Motivation

- Brain metastases occur in 20-40% of cancer patients
- Accurate, fast segmentation is critical for stereotactic radiosurgery planning
- Small lesions (<10 mm) are routinely missed by radiologists under time pressure
- This project focuses specifically on improving detection of sub-centimeter lesions

## What This Is

A hybrid segmentation pipeline combining a custom lightweight 3D U-Net ensemble
with nnU-Net v2, trained on the [Stanford BrainMetShare](https://stanfordaimi.azurewebsites.net/datasets/) dataset (566 labeled cases).

| Metric | Custom Ensemble | nnU-Net v2 |
|---|---|---|
| Mean Voxel Dice | 0.748 | **0.760** |
| Median Voxel Dice | 0.781 | **0.810** |
| Voxel Sensitivity | **0.784** | 0.735 |
| Lesion Detection F1 | **0.753** | 0.725 |
| Lesion Recall | **0.725** | 0.654 |
| Tiny Lesion Dice (<100 vox) | **0.833** | 0.741 |
| Large Lesion Dice (>5k vox) | 0.844 | **0.853** |

**Stacking ensemble** (fusing both): **0.776 mean Dice, ~0.82 median**.

**Human inter-rater baseline**: Median Dice 0.85 (IQR 0.80-0.89) from the UCSF-BMSR paper.

Trained entirely on a single consumer GPU (RTX 5060 Ti, 16 GB VRAM).

## Architecture

### Custom Lightweight Ensemble

- **Network**: LightweightUNet3D with attention gates and residual connections
- **Input**: 4-channel MRI (T1-pre, T1-Gd, FLAIR, T2)
- **Ensemble**: 4 models trained at patch sizes 8, 12, 24, 36
- **Fusion**: Average probabilities, threshold 0.40
- **Loss**: 0.7 * Tversky (alpha=0.3, beta=0.7) + 0.3 * Focal
- **Strength**: High sensitivity, especially for tiny/small lesions

### nnU-Net v2

- **Network**: PlainConvUNet, 6 stages [32, 64, 128, 256, 320, 320]
- **Patch size**: 128^3, batch size 2
- **Training**: 1000 epochs, SGD + poly LR, Dice + CE loss, deep supervision
- **Strength**: High precision, better on medium-to-large lesions

### Stacking Meta-Learner

A lightweight 3D CNN (~25K params) trained on the probability outputs of both
the custom ensemble and nnU-Net, combining their complementary strengths.

## Installation

```bash
git clone https://github.com/ckirby04/BrainMetScan.git
cd BrainMetScan
pip install -r requirements.txt
```

Model weights are not included in the repository. Download them from
[GitHub Releases](https://github.com/ckirby04/BrainMetScan/releases) and place
them in the `model/` directory:

```bash
python scripts/download_weights.py   # or download manually from Releases
```

## Usage

### Inference (Ensemble)

```bash
python scripts/run_inference.py \
  --input path/to/patient_dir \
  --output path/to/output_dir \
  --threshold 0.4
```

Each patient directory should contain four NIfTI files:
`t1_pre.nii.gz`, `t1_gd.nii.gz`, `flair.nii.gz`, `t2.nii.gz`

### Evaluation

```bash
# Evaluate nnU-Net
python scripts/evaluate_nnunet.py --fold 0

# Evaluate custom ensemble
python scripts/lesionwise_eval.py

# Compare all models side-by-side
python scripts/compare_models.py
```

## Dataset

This project uses the [Stanford BrainMetShare](https://stanfordaimi.azurewebsites.net/datasets/)
dataset. Data is **not** included in this repository. See [`data/README.md`](data/README.md)
for download and setup instructions.

## Limitations

- Trained and validated on a single institution dataset (Stanford BrainMetShare)
- Generalizability to other scanners, field strengths, and protocols is unknown
- False positive rate needs reduction for clinical deployment
- **Seeking multi-site MRI data for cross-validation and external validation**

## Citation

```bibtex
@software{kirby2026brainmetscan,
  author = {Kirby, Clark},
  title  = {BrainMetScan: Multi-Scale Ensemble for Brain Metastasis Segmentation},
  year   = {2026},
  url    = {https://github.com/ckirby04/BrainMetScan}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Clark Kirby - University of Arkansas
- GitHub: [@ckirby04](https://github.com/ckirby04)
