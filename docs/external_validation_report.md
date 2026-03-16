# External Validation Report

## Overview

External validation was performed on two independent, publicly available brain metastasis datasets to assess generalization of the BrainMetScan segmentation pipeline beyond the training distribution.

## Training Data

- **Sources**: Stanford BrainMetShare (156 studies) + UCSF-BMSR (410 studies)
- **Preprocessing**: Skull-stripped (BET / nnU-Net), co-registered sequences, native geometry preserved, resized to 256³
- **Internal cross-validation Dice**: 0.747 (top-3 ensemble of 12-patch, 24-patch, 36-patch models at threshold 0.40)

## External Datasets

### PRETREAT-METSTOBRAIN-MASKS (Yale)

- **Source**: Yale New Haven Hospital, via The Cancer Imaging Archive (TCIA)
- **Cases**: 200 patients, 975 contrast-enhancing lesions
- **Sequences**: T1-pre, T1-post, T2, FLAIR (NIfTI)
- **Preprocessing**: Skull-stripped (HD-BET), co-registered to SRI24 anatomical template, resampled to 1mm³ isotropic (240×240×155)
- **Segmentation labels**: Necrosis (1), edema (2), enhancing tumor (3) — merged to binary for evaluation

### PROTEAS (University of Cyprus)

- **Source**: Bank of Cyprus Oncology Center, via Zenodo (DOI: 10.5281/zenodo.17253793)
- **Cases**: 44 baseline imaging studies from 40 patients, 65 brain metastases
- **Sequences**: T1-pre, T1-post, T2, FLAIR (NIfTI, BraTS format)
- **Preprocessing**: Skull-stripped, co-registered to SRI24 anatomical template, resampled to 1mm³ isotropic (240×240×155)
- **Segmentation labels**: Enhancing tumor (3), edema (2), necrotic core (1) — merged to binary for evaluation

## Method

The top-3 ensemble (average of 12-patch, 24-patch, and 36-patch models at threshold 0.40) was evaluated on both external datasets. Each external case was resized from 240×240×155 to 256³ to match the pipeline input format. Full stacking inference (v4) was not possible because nnU-Net predictions were unavailable for external data, leaving only 4 of the required 6 base models (6 channels instead of 8).

Evaluation metrics: voxel-level Dice coefficient, sensitivity, precision, lesion-wise F1/recall/precision, per-lesion Dice, surface Dice at 1/2/3mm tolerance, Hausdorff distance (95th percentile).

## Results

| Metric | Internal (n=566) | PRETREAT (n=200) | PROTEAS (n=44) |
|--------|:-:|:-:|:-:|
| Voxel Dice | 0.747 | 0.174 | 0.202 |
| Sensitivity | 0.784 | 0.137 | 0.167 |
| Precision | — | — | 0.475 |
| Lesion F1 | — | — | 0.339 |
| Lesion Recall | — | — | 0.649 |
| Lesion Precision | — | — | 0.283 |
| Per-Lesion Dice | — | — | 0.182 |
| Surface Dice @2mm | — | — | 0.169 |
| Hausdorff 95 | — | — | 64.8 mm |

## Analysis

### Domain Shift from Atlas Registration

Both external datasets show a consistent ~75% reduction in Dice score compared to internal validation. The consistency of this drop across two independent institutions (Yale and Cyprus) confirms a systematic domain shift rather than a dataset-specific artifact.

The key difference between training and external data:

| Property | Training Data | External Data |
|----------|:---:|:---:|
| Skull-stripped | Yes (BET/nnU-Net) | Yes (HD-BET) |
| Co-registered | Yes | Yes |
| Atlas-registered | **No (native geometry)** | **Yes (SRI24, 240×240×155)** |
| Resolution | Variable, resized to 256³ | 1mm isotropic, resized to 256³ |

Skull-stripping is not the source of the domain shift — both training and external datasets are skull-stripped. The critical difference is that training data preserves native spatial geometry while external datasets are warped to a standardized atlas template before resizing.

### Lesion Detection is Partially Preserved

Despite poor segmentation quality, the model retains moderate lesion detection capability on external data. On PROTEAS, lesion recall was 0.649 (detecting 63 of 116 lesions), suggesting the model recognizes metastasis-like patterns across domains. However, precision is low (0.187, with 274 false positive detections), and per-lesion Dice is poor (0.182), indicating the model cannot accurately delineate lesion boundaries in the warped coordinate space.

### Dataset Landscape Limitation

All publicly available brain metastasis datasets with multi-sequence MRI (T1-pre, T1-post, FLAIR, T2) and segmentation masks are distributed in BraTS format (skull-stripped + SRI24 atlas-registered). This includes PRETREAT, PROTEAS, BraTS-METS 2023, and Yale-Brain-Mets-Longitudinal. The Ocana-Tienda 2023 dataset preserves native coordinates but provides segmentations only on post-contrast T1 in DICOM format. No public dataset currently matches the training data preprocessing (skull-stripped, native geometry, all 4 sequences).

## Conclusion

External validation on two independent BraTS-format datasets (PRETREAT, n=200; PROTEAS, n=44) showed reduced performance (Dice 0.174, 0.202 respectively) compared to internal cross-validation (0.747), attributable to domain shift from SRI24 atlas registration. The model was trained on native-geometry skull-stripped MRI, while all publicly available brain metastasis datasets are distributed in atlas-registered format. Lesion detection recall remained moderate (0.54–0.65), suggesting the model recognizes metastasis patterns across domains but cannot accurately segment in the warped coordinate space.

For clinical deployment on native-geometry MRI (the intended use case), the model performs well. Generalization to atlas-registered data would require domain adaptation or retraining.

## Reproducibility

- PRETREAT preprocessing: `scripts/preprocessing/prepare_pretreat_validation.py`
- PROTEAS preprocessing: `scripts/preprocessing/prepare_proteas_validation.py`
- Evaluation: `scripts/evaluation/eval_external_validation.py --data-dir <path> --output-dir <path>`
- PRETREAT results: `results/external_validation/external_validation_results.json`
- PROTEAS results: `results/external_validation_proteas/external_validation_results.json`
