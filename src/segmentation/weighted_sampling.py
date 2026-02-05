"""
Weighted sampling strategies for handling difficult cases
Oversamples cases with small lesions and previously failed cases
"""

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, Sampler
from pathlib import Path
import nibabel as nib
from scipy import ndimage


# Cases that completely failed in previous training (Dice = 0.0)
DIFFICULT_CASES = [
    'Mets_011', 'Mets_013', 'Mets_040', 'Mets_090', 'Mets_092',
    'Mets_146', 'Mets_151', 'Mets_167', 'Mets_189', 'Mets_213',
    'Mets_234', 'Mets_248', 'Mets_273'
]


def calculate_lesion_volume(mask_path):
    """Calculate total lesion volume in a mask"""
    try:
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()
        return np.sum(mask > 0)
    except:
        return 0


def calculate_num_lesions(mask_path):
    """Count number of separate lesions"""
    try:
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()
        labeled_mask, num_lesions = ndimage.label(mask > 0)
        return num_lesions
    except:
        return 0


def get_case_weights(dataset, strategy='hybrid', difficulty_multiplier=10.0):
    """
    Calculate sampling weights for each case based on difficulty

    Args:
        dataset: BrainMetDataset instance
        strategy: Weighting strategy
            - 'volume': Weight by lesion volume (small = higher weight)
            - 'difficulty': Weight failed cases higher
            - 'hybrid': Combine both approaches (recommended)
            - 'uniform': Equal weights (baseline)
        difficulty_multiplier: How much more to sample difficult cases (default: 10.0 for strong focus)

    Returns:
        weights: List of sampling weights for each case
    """
    num_cases = len(dataset.cases)
    weights = np.ones(num_cases)

    if strategy == 'uniform':
        return weights.tolist()

    # Calculate volumes if needed
    if strategy in ['volume', 'hybrid']:
        volumes = []
        for case_dir in dataset.cases:
            mask_path = case_dir / "seg.nii.gz"
            volume = calculate_lesion_volume(mask_path)
            volumes.append(volume)

        volumes = np.array(volumes)

        # Inverse volume weighting (smaller lesions = higher weight)
        # Add constant to avoid division by zero
        volume_weights = 1.0 / (volumes + 100)

        # Normalize to mean=1
        volume_weights = volume_weights / volume_weights.mean()

        if strategy == 'volume':
            weights = volume_weights

    # Add difficulty weighting
    if strategy in ['difficulty', 'hybrid']:
        difficulty_weights = np.ones(num_cases)

        for idx, case_dir in enumerate(dataset.cases):
            case_id = case_dir.name
            if case_id in DIFFICULT_CASES:
                difficulty_weights[idx] = difficulty_multiplier

        if strategy == 'difficulty':
            weights = difficulty_weights
        elif strategy == 'hybrid':
            # Combine volume and difficulty weights
            weights = volume_weights * difficulty_weights

    # Normalize so weights sum to num_cases (for proper epoch length)
    weights = weights * num_cases / weights.sum()

    return weights.tolist()


def get_stratified_weights(dataset, num_bins=5, boost_small=2.0):
    """
    Stratified sampling based on lesion size bins

    Args:
        dataset: BrainMetDataset instance
        num_bins: Number of stratification bins
        boost_small: Multiplier for smallest bin

    Returns:
        weights: Sampling weights
    """
    # Calculate volumes
    volumes = []
    for case_dir in dataset.cases:
        mask_path = case_dir / "seg.nii.gz"
        volume = calculate_lesion_volume(mask_path)
        volumes.append(volume)

    volumes = np.array(volumes)

    # Create bins
    bins = np.percentile(volumes, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(volumes, bins[1:-1])

    # Calculate weights for each bin
    # Smaller bins get higher weights
    bin_weights = np.linspace(boost_small, 1.0, num_bins)

    # Assign weights based on bins
    weights = bin_weights[bin_indices]

    # Additional boost for known difficult cases
    for idx, case_dir in enumerate(dataset.cases):
        if case_dir.name in DIFFICULT_CASES:
            weights[idx] *= 2.0

    # Normalize
    weights = weights * len(dataset) / weights.sum()

    return weights.tolist()


def create_weighted_sampler(dataset, strategy='hybrid', **kwargs):
    """
    Create a WeightedRandomSampler for the dataset

    Args:
        dataset: BrainMetDataset instance
        strategy: Weighting strategy (see get_case_weights)
        **kwargs: Additional arguments for get_case_weights

    Returns:
        WeightedRandomSampler instance
    """
    if strategy == 'stratified':
        weights = get_stratified_weights(dataset, **kwargs)
    else:
        weights = get_case_weights(dataset, strategy=strategy, **kwargs)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler


class BalancedBatchSampler(Sampler):
    """
    Custom sampler that ensures each batch has a mix of difficult and easy cases

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        difficult_ratio: Fraction of batch that should be difficult cases (0-1)
    """
    def __init__(self, dataset, batch_size, difficult_ratio=0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.difficult_ratio = difficult_ratio

        # Identify difficult and easy indices
        self.difficult_indices = []
        self.easy_indices = []

        for idx, case_dir in enumerate(dataset.cases):
            if case_dir.name in DIFFICULT_CASES:
                self.difficult_indices.append(idx)
            else:
                self.easy_indices.append(idx)

        self.num_difficult = max(1, int(batch_size * difficult_ratio))
        self.num_easy = batch_size - self.num_difficult

        # Calculate number of batches
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            # Sample difficult cases
            difficult_batch = np.random.choice(
                self.difficult_indices,
                size=self.num_difficult,
                replace=True
            )

            # Sample easy cases
            easy_batch = np.random.choice(
                self.easy_indices,
                size=self.num_easy,
                replace=True
            )

            # Combine and shuffle
            batch = np.concatenate([difficult_batch, easy_batch])
            np.random.shuffle(batch)

            yield from batch.tolist()

    def __len__(self):
        return self.num_batches * self.batch_size


def print_sampling_statistics(dataset, sampler, num_samples=1000):
    """
    Print statistics about sampling distribution

    Args:
        dataset: Dataset instance
        sampler: Sampler instance
        num_samples: Number of samples to draw for analysis
    """
    # Sample indices
    if isinstance(sampler, WeightedRandomSampler):
        # Get weights
        weights = sampler.weights
        indices = torch.multinomial(
            torch.tensor(weights).float(),
            num_samples,
            replacement=True
        ).numpy()
    else:
        # Sample directly
        sampler_iter = iter(sampler)
        indices = [next(sampler_iter) for _ in range(min(num_samples, len(sampler)))]

    # Count occurrences
    from collections import Counter
    counter = Counter(indices)

    # Check difficult case sampling
    difficult_count = sum(
        counter[idx]
        for idx, case_dir in enumerate(dataset.cases)
        if case_dir.name in DIFFICULT_CASES
    )

    total_difficult = len([
        case for case in dataset.cases
        if case.name in DIFFICULT_CASES
    ])

    print("\n" + "=" * 80)
    print("SAMPLING STATISTICS")
    print("=" * 80)
    print(f"Total cases: {len(dataset.cases)}")
    print(f"Difficult cases: {total_difficult}")
    print(f"Samples drawn: {num_samples}")
    print(f"Difficult case samples: {difficult_count} ({100*difficult_count/num_samples:.1f}%)")
    print(f"Expected without weighting: {100*total_difficult/len(dataset.cases):.1f}%")
    print(f"Boost factor: {(difficult_count/num_samples) / (total_difficult/len(dataset.cases)):.2f}x")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent))
    from dataset import BrainMetDataset

    dataset = BrainMetDataset("data/train")

    # Test different strategies
    strategies = ['uniform', 'volume', 'difficulty', 'hybrid', 'stratified']

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        print("-" * 80)

        if strategy == 'stratified':
            sampler = create_weighted_sampler(dataset, strategy='stratified')
        else:
            sampler = create_weighted_sampler(dataset, strategy=strategy)

        print_sampling_statistics(dataset, sampler, num_samples=1000)
