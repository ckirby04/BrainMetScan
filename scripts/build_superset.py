"""
Superset Builder - Consolidates multiple brain metastasis datasets into a unified training set.

Creates (in parent directory ../Superset/):
- ../Superset/full/train/       - All labeled cases combined
- ../Superset/full/test/        - All test cases combined
- ../Superset/high_quality/train/  - Quality-filtered subset
- ../Superset/high_quality/test/   - Quality-filtered test set
- ../Superset/pretraining/      - Unlabeled data for self-supervised pretraining
- ../Superset/metadata.csv      - Combined metadata with quality scores
"""

import os
import shutil
import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a single case"""
    case_id: str
    source: str
    snr: float                    # Signal-to-noise ratio
    contrast: float               # Lesion-to-background contrast
    sharpness: float              # Image sharpness (gradient magnitude)
    completeness: float           # Modality completeness (0-1)
    mask_volume: int              # Segmentation volume in voxels
    mask_quality: float           # Mask smoothness/consistency
    overall_score: float          # Weighted composite score
    is_high_quality: bool         # Above threshold
    notes: str = ""


class QualityFilter:
    """Computes quality metrics for brain MRI scans"""

    def __init__(self, snr_weight=0.25, contrast_weight=0.25,
                 sharpness_weight=0.2, completeness_weight=0.15,
                 mask_quality_weight=0.15, high_quality_percentile=70):
        self.weights = {
            'snr': snr_weight,
            'contrast': contrast_weight,
            'sharpness': sharpness_weight,
            'completeness': completeness_weight,
            'mask_quality': mask_quality_weight
        }
        self.high_quality_percentile = high_quality_percentile
        self.all_scores = []

    def compute_snr(self, img_data: np.ndarray) -> float:
        """Compute signal-to-noise ratio"""
        # Use brain region (non-zero voxels) vs background
        brain_mask = img_data > np.percentile(img_data, 10)
        if brain_mask.sum() == 0:
            return 0.0

        signal = img_data[brain_mask].mean()
        # Estimate noise from corners (background)
        corner_size = min(10, img_data.shape[0] // 10)
        corners = [
            img_data[:corner_size, :corner_size, :corner_size],
            img_data[-corner_size:, :corner_size, :corner_size],
            img_data[:corner_size, -corner_size:, :corner_size],
            img_data[:corner_size, :corner_size, -corner_size:],
        ]
        noise_std = np.std(np.concatenate([c.flatten() for c in corners]))

        if noise_std == 0:
            return 100.0  # Perfect SNR (unlikely)

        return float(signal / noise_std)

    def compute_contrast(self, img_data: np.ndarray, mask_data: Optional[np.ndarray]) -> float:
        """Compute lesion-to-background contrast (if mask available)"""
        if mask_data is None or mask_data.sum() == 0:
            # No mask - use intensity range as proxy
            p5, p95 = np.percentile(img_data[img_data > 0], [5, 95])
            return float((p95 - p5) / (p95 + 1e-6))

        lesion_intensity = img_data[mask_data > 0].mean()
        # Background is brain tissue around lesion
        dilated = mask_data.copy()
        background_mask = (img_data > np.percentile(img_data, 20)) & (mask_data == 0)
        if background_mask.sum() == 0:
            return 0.5
        background_intensity = img_data[background_mask].mean()

        contrast = abs(lesion_intensity - background_intensity) / (background_intensity + 1e-6)
        return float(min(contrast, 2.0))  # Cap at 2.0

    def compute_sharpness(self, img_data: np.ndarray) -> float:
        """Compute image sharpness using gradient magnitude"""
        # Compute gradients
        gx = np.diff(img_data, axis=0)
        gy = np.diff(img_data, axis=1)
        gz = np.diff(img_data, axis=2)

        # Gradient magnitude (trimmed to same size)
        min_shape = [min(gx.shape[i], gy.shape[i], gz.shape[i]) for i in range(3)]
        gx = gx[:min_shape[0], :min_shape[1], :min_shape[2]]
        gy = gy[:min_shape[0], :min_shape[1], :min_shape[2]]
        gz = gz[:min_shape[0], :min_shape[1], :min_shape[2]]

        gradient_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # Focus on brain region
        brain_mask = img_data[:min_shape[0], :min_shape[1], :min_shape[2]] > np.percentile(img_data, 20)
        if brain_mask.sum() == 0:
            return 0.0

        sharpness = gradient_mag[brain_mask].mean()
        return float(sharpness)

    def compute_mask_quality(self, mask_data: Optional[np.ndarray]) -> float:
        """Assess segmentation mask quality (smoothness, no isolated voxels)"""
        if mask_data is None or mask_data.sum() == 0:
            return 0.5  # Neutral if no mask

        # Check for isolated voxels (indicates noise/poor annotation)
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask_data > 0)

        if num_features == 0:
            return 0.0

        # Penalize many small disconnected regions
        sizes = ndimage.sum(mask_data > 0, labeled, range(1, num_features + 1))
        if len(sizes) == 0:
            return 0.5

        # Quality higher if fewer, larger connected components
        largest = max(sizes)
        total = mask_data.sum()

        # Ratio of largest component to total
        coherence = largest / (total + 1e-6)

        # Penalize too many components
        component_penalty = min(1.0, 5.0 / (num_features + 1))

        return float(coherence * 0.7 + component_penalty * 0.3)

    def evaluate_case(self, case_path: Path, source: str,
                      modality_map: Dict[str, str]) -> QualityMetrics:
        """Evaluate quality metrics for a single case"""
        case_id = case_path.name

        # Load available modalities
        img_data = None
        mask_data = None
        available_modalities = []

        for standard_name, filename in modality_map.items():
            filepath = case_path / filename
            if filepath.exists():
                available_modalities.append(standard_name)
                if standard_name == 't1_gd' and img_data is None:
                    # Use contrast-enhanced T1 as primary
                    try:
                        img_data = nib.load(str(filepath)).get_fdata()
                    except Exception as e:
                        logger.warning(f"Failed to load {filepath}: {e}")
                elif standard_name == 'seg':
                    try:
                        mask_data = nib.load(str(filepath)).get_fdata()
                    except Exception as e:
                        logger.warning(f"Failed to load mask {filepath}: {e}")

        # Fallback to any available modality
        if img_data is None:
            for standard_name, filename in modality_map.items():
                if standard_name != 'seg':
                    filepath = case_path / filename
                    if filepath.exists():
                        try:
                            img_data = nib.load(str(filepath)).get_fdata()
                            break
                        except:
                            continue

        if img_data is None:
            return QualityMetrics(
                case_id=case_id, source=source,
                snr=0.0, contrast=0.0, sharpness=0.0,
                completeness=0.0, mask_volume=0, mask_quality=0.0,
                overall_score=0.0, is_high_quality=False,
                notes="Failed to load any modality"
            )

        # Compute metrics
        snr = self.compute_snr(img_data)
        contrast = self.compute_contrast(img_data, mask_data)
        sharpness = self.compute_sharpness(img_data)
        completeness = len(available_modalities) / len([k for k in modality_map.keys() if k != 'seg'])
        mask_quality = self.compute_mask_quality(mask_data)
        mask_volume = int(mask_data.sum()) if mask_data is not None else 0

        # Normalize metrics to 0-1 range (will be calibrated across dataset)
        metrics = QualityMetrics(
            case_id=case_id,
            source=source,
            snr=snr,
            contrast=contrast,
            sharpness=sharpness,
            completeness=completeness,
            mask_volume=mask_volume,
            mask_quality=mask_quality,
            overall_score=0.0,  # Computed later after normalization
            is_high_quality=False,
            notes=""
        )

        return metrics

    def normalize_and_score(self, all_metrics: List[QualityMetrics]) -> List[QualityMetrics]:
        """Normalize metrics across dataset and compute overall scores"""
        if not all_metrics:
            return all_metrics

        # Collect raw values
        snr_vals = [m.snr for m in all_metrics]
        contrast_vals = [m.contrast for m in all_metrics]
        sharpness_vals = [m.sharpness for m in all_metrics]

        # Min-max normalization
        def normalize(vals):
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return [0.5] * len(vals)
            return [(v - min_v) / (max_v - min_v) for v in vals]

        snr_norm = normalize(snr_vals)
        contrast_norm = normalize(contrast_vals)
        sharpness_norm = normalize(sharpness_vals)

        # Compute overall scores
        for i, m in enumerate(all_metrics):
            score = (
                self.weights['snr'] * snr_norm[i] +
                self.weights['contrast'] * contrast_norm[i] +
                self.weights['sharpness'] * sharpness_norm[i] +
                self.weights['completeness'] * m.completeness +
                self.weights['mask_quality'] * m.mask_quality
            )
            m.overall_score = score

        # Determine high quality threshold
        scores = [m.overall_score for m in all_metrics]
        threshold = np.percentile(scores, self.high_quality_percentile)

        for m in all_metrics:
            m.is_high_quality = m.overall_score >= threshold

        return all_metrics


class SupersetBuilder:
    """Builds unified superset from multiple data sources"""

    def __init__(self, raw_data_dir: str, output_dir: str,
                 high_quality_percentile: int = 70,
                 num_workers: int = 4):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.quality_filter = QualityFilter(high_quality_percentile=high_quality_percentile)

        # Define modality mappings for each source
        self.source_configs = {
            'BrainMetShare': {
                'train_dir': 'train',
                'test_dir': 'test',
                'modality_map': {
                    't1_pre': 't1_pre.nii.gz',
                    't1_gd': 't1_gd.nii.gz',
                    'flair': 'flair.nii.gz',
                    't2': 'bravo.nii.gz',  # BRAVO is T2-weighted
                    'seg': 'seg.nii.gz'
                },
                'has_test_masks': False
            },
            'UCSF_BrainMetastases': {
                'train_dir': 'train',
                'test_dir': None,  # No separate test set
                'modality_map': {
                    't1_pre': '{case_id}_T1pre.nii.gz',
                    't1_gd': '{case_id}_T1post.nii.gz',
                    'flair': '{case_id}_FLAIR.nii.gz',
                    't2': '{case_id}_T2Synth.nii.gz',
                    'seg': '{case_id}_seg.nii.gz'
                },
                'has_test_masks': False
            },
            'Yale-Brain-Mets-Longitudinal': {
                'train_dir': None,  # Special structure
                'test_dir': None,
                'is_pretraining': True,  # No masks available
                'modality_map': {
                    't1_pre': '*_PRE.nii.gz',
                    't1_gd': '*_POST.nii.gz',
                    'flair': '*_FLAIR.nii.gz',
                    't2': '*_T2.nii.gz'
                }
            }
        }

        # Standard output naming
        self.standard_names = {
            't1_pre': 't1_pre.nii.gz',
            't1_gd': 't1_gd.nii.gz',
            'flair': 'flair.nii.gz',
            't2': 't2.nii.gz',
            'seg': 'seg.nii.gz'
        }

    def setup_directories(self):
        """Create output directory structure"""
        dirs = [
            self.output_dir / 'full' / 'train',
            self.output_dir / 'full' / 'test',
            self.output_dir / 'high_quality' / 'train',
            self.output_dir / 'high_quality' / 'test',
            self.output_dir / 'pretraining',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory structure at {self.output_dir}")

    def get_source_path(self, source: str, filename: str, case_id: str) -> str:
        """Resolve filename template with case_id"""
        if '{case_id}' in filename:
            return filename.format(case_id=case_id)
        return filename

    def copy_case(self, source: str, case_path: Path, output_path: Path,
                  modality_map: Dict[str, str], case_id: str) -> bool:
        """Copy a single case with standardized naming"""
        output_path.mkdir(parents=True, exist_ok=True)

        copied_any = False
        for standard_name, src_pattern in modality_map.items():
            src_filename = self.get_source_path(source, src_pattern, case_id)
            src_file = case_path / src_filename

            # Handle glob patterns
            if '*' in src_filename:
                import glob
                matches = list(case_path.glob(src_filename))
                if matches:
                    src_file = matches[0]
                else:
                    continue

            if src_file.exists():
                dst_file = output_path / self.standard_names[standard_name]
                shutil.copy2(src_file, dst_file)
                copied_any = True

        return copied_any

    def process_brainmetshare(self) -> Tuple[List[QualityMetrics], int]:
        """Process BrainMetShare dataset"""
        source = 'BrainMetShare'
        config = self.source_configs[source]
        source_dir = self.raw_data_dir / source

        metrics_list = []
        copied = 0

        # Process train
        train_dir = source_dir / config['train_dir']
        if train_dir.exists():
            cases = [d for d in train_dir.iterdir() if d.is_dir()]
            logger.info(f"Processing {len(cases)} BrainMetShare train cases")

            for case_path in tqdm(cases, desc="BrainMetShare train"):
                case_id = case_path.name
                new_case_id = f"BMS_{case_id}"

                # Copy to full/train
                output_path = self.output_dir / 'full' / 'train' / new_case_id
                if self.copy_case(source, case_path, output_path,
                                  config['modality_map'], case_id):
                    copied += 1

                    # Compute quality metrics
                    metrics = self.quality_filter.evaluate_case(
                        output_path, source,
                        {k: self.standard_names[k] for k in config['modality_map'].keys()}
                    )
                    metrics.case_id = new_case_id
                    metrics_list.append(metrics)

        # Process test
        test_dir = source_dir / config['test_dir']
        if test_dir.exists():
            cases = [d for d in test_dir.iterdir() if d.is_dir()]
            logger.info(f"Processing {len(cases)} BrainMetShare test cases")

            for case_path in tqdm(cases, desc="BrainMetShare test"):
                case_id = case_path.name
                new_case_id = f"BMS_{case_id}"

                output_path = self.output_dir / 'full' / 'test' / new_case_id
                if self.copy_case(source, case_path, output_path,
                                  config['modality_map'], case_id):
                    copied += 1

        return metrics_list, copied

    def process_ucsf(self) -> Tuple[List[QualityMetrics], int]:
        """Process UCSF Brain Metastases dataset"""
        source = 'UCSF_BrainMetastases'
        config = self.source_configs[source]
        source_dir = self.raw_data_dir / source

        metrics_list = []
        copied = 0

        train_dir = source_dir / config['train_dir']
        if train_dir.exists():
            cases = [d for d in train_dir.iterdir() if d.is_dir()]
            logger.info(f"Processing {len(cases)} UCSF train cases")

            for case_path in tqdm(cases, desc="UCSF train"):
                case_id = case_path.name
                new_case_id = f"UCSF_{case_id}"

                output_path = self.output_dir / 'full' / 'train' / new_case_id
                if self.copy_case(source, case_path, output_path,
                                  config['modality_map'], case_id):
                    copied += 1

                    metrics = self.quality_filter.evaluate_case(
                        output_path, source,
                        {k: self.standard_names[k] for k in config['modality_map'].keys() if k in self.standard_names}
                    )
                    metrics.case_id = new_case_id
                    metrics_list.append(metrics)

        return metrics_list, copied

    def process_yale_pretraining(self) -> int:
        """Process Yale dataset for pretraining (no masks)"""
        source = 'Yale-Brain-Mets-Longitudinal'
        source_dir = self.raw_data_dir / source

        if not source_dir.exists():
            logger.warning(f"Yale dataset not found at {source_dir}")
            return 0

        copied = 0
        patient_dirs = [d for d in source_dir.iterdir()
                        if d.is_dir() and d.name.startswith('YG_')]

        logger.info(f"Processing {len(patient_dirs)} Yale patients for pretraining")

        for patient_dir in tqdm(patient_dirs, desc="Yale pretraining"):
            patient_id = patient_dir.name

            # Each patient has multiple timepoints
            timepoint_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]

            for tp_dir in timepoint_dirs:
                timepoint = tp_dir.name
                new_case_id = f"Yale_{patient_id}_{timepoint}"

                output_path = self.output_dir / 'pretraining' / new_case_id
                output_path.mkdir(parents=True, exist_ok=True)

                # Find and copy files
                modalities_copied = 0
                for nii_file in tp_dir.glob('*.nii.gz'):
                    filename = nii_file.name.upper()

                    if '_PRE.' in filename or '_PRE_' in filename:
                        dst = output_path / 't1_pre.nii.gz'
                    elif '_POST.' in filename or '_POST_' in filename:
                        dst = output_path / 't1_gd.nii.gz'
                    elif '_FLAIR.' in filename or '_FLAIR_' in filename:
                        dst = output_path / 'flair.nii.gz'
                    elif '_T2.' in filename or '_T2_' in filename:
                        dst = output_path / 't2.nii.gz'
                    else:
                        continue

                    shutil.copy2(nii_file, dst)
                    modalities_copied += 1

                if modalities_copied > 0:
                    copied += 1

        return copied

    def create_high_quality_subset(self, metrics_list: List[QualityMetrics]):
        """Create high-quality subset based on quality scores"""
        # Normalize and score
        metrics_list = self.quality_filter.normalize_and_score(metrics_list)

        high_quality_count = 0
        for metrics in metrics_list:
            if metrics.is_high_quality:
                # Copy from full to high_quality
                src_path = self.output_dir / 'full' / 'train' / metrics.case_id
                dst_path = self.output_dir / 'high_quality' / 'train' / metrics.case_id

                if src_path.exists():
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    high_quality_count += 1

        logger.info(f"Created high-quality subset with {high_quality_count} cases")
        return metrics_list

    def save_metadata(self, metrics_list: List[QualityMetrics]):
        """Save combined metadata and quality scores"""
        # Convert to DataFrame
        data = [asdict(m) for m in metrics_list]
        df = pd.DataFrame(data)

        # Sort by quality score
        df = df.sort_values('overall_score', ascending=False)

        # Save CSV
        csv_path = self.output_dir / 'metadata.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved metadata to {csv_path}")

        # Save summary JSON
        summary = {
            'total_cases': int(len(df)),
            'high_quality_cases': int(df['is_high_quality'].sum()),
            'sources': {k: int(v) for k, v in df['source'].value_counts().to_dict().items()},
            'avg_quality_score': float(df['overall_score'].mean()),
            'quality_threshold': float(df[df['is_high_quality']]['overall_score'].min()) if df['is_high_quality'].any() else 0.0,
            'metrics_summary': {
                'snr': {'mean': float(df['snr'].mean()), 'std': float(df['snr'].std())},
                'contrast': {'mean': float(df['contrast'].mean()), 'std': float(df['contrast'].std())},
                'sharpness': {'mean': float(df['sharpness'].mean()), 'std': float(df['sharpness'].std())},
                'completeness': {'mean': float(df['completeness'].mean()), 'std': float(df['completeness'].std())},
            }
        }

        with open(self.output_dir / 'quality_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Quality Summary: {summary['high_quality_cases']}/{summary['total_cases']} high quality cases")

        return df

    def build(self):
        """Main build process"""
        logger.info("=" * 60)
        logger.info("SUPERSET BUILDER - Starting data consolidation")
        logger.info("=" * 60)

        self.setup_directories()

        all_metrics = []
        total_copied = 0

        # Process each source
        logger.info("\n[1/4] Processing BrainMetShare...")
        metrics, copied = self.process_brainmetshare()
        all_metrics.extend(metrics)
        total_copied += copied
        logger.info(f"  -> {copied} cases copied")

        logger.info("\n[2/4] Processing UCSF Brain Metastases...")
        metrics, copied = self.process_ucsf()
        all_metrics.extend(metrics)
        total_copied += copied
        logger.info(f"  -> {copied} cases copied")

        logger.info("\n[3/4] Processing Yale for pretraining...")
        pretraining_count = self.process_yale_pretraining()
        logger.info(f"  -> {pretraining_count} pretraining cases copied")

        logger.info("\n[4/4] Creating high-quality subset...")
        all_metrics = self.create_high_quality_subset(all_metrics)

        logger.info("\n[5/5] Saving metadata...")
        df = self.save_metadata(all_metrics)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("SUPERSET BUILD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total supervised training cases: {total_copied}")
        logger.info(f"High-quality subset: {df['is_high_quality'].sum()} cases")
        logger.info(f"Pretraining cases (Yale): {pretraining_count}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)

        return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build Superset from raw datasets')
    parser.add_argument('--raw-data', type=str,
                        default='Raw Data',
                        help='Path to raw data directory')
    parser.add_argument('--output', type=str,
                        default='../Superset',
                        help='Output directory for Superset (parent directory)')
    parser.add_argument('--quality-percentile', type=int, default=70,
                        help='Percentile threshold for high-quality (default: 70)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    raw_data_dir = script_dir / args.raw_data
    output_dir = script_dir / args.output

    builder = SupersetBuilder(
        raw_data_dir=str(raw_data_dir),
        output_dir=str(output_dir),
        high_quality_percentile=args.quality_percentile,
        num_workers=args.workers
    )

    builder.build()


if __name__ == '__main__':
    main()
