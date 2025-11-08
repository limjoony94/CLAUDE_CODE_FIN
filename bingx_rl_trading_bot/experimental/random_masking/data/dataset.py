"""
PyTorch Dataset for Candle Data with Random Masking
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from loguru import logger

from .preprocessor import CandlePreprocessor
from .masking_strategy import RandomMaskingStrategy


class CandleDataset(Dataset):
    """
    PyTorch Dataset for time series candle data with random masking

    Supports three modes:
    - 'train': Apply random masking for multi-task learning
    - 'val': Forecasting only (no masking for validation)
    - 'test': Forecasting only (no masking for testing)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        seq_len: int = 100,
        pred_len: int = 10,
        masking_strategy: Optional[RandomMaskingStrategy] = None,
        mode: str = 'train',
        stride: int = 1
    ):
        """
        Initialize dataset

        Args:
            data_path: Path to parquet file (if loading from file)
            data: Preprocessed numpy array (if already loaded)
            seq_len: Input sequence length
            pred_len: Prediction horizon length (for evaluation)
            masking_strategy: RandomMaskingStrategy instance (for training)
            mode: 'train', 'val', or 'test'
            stride: Stride for sliding window (1 = no overlap, seq_len = no overlap)
        """
        # Load data
        if data is not None:
            self.data = data
        elif data_path is not None:
            logger.info(f"Loading data from {data_path}...")
            df = pd.read_parquet(data_path)
            self.data = df.values
        else:
            raise ValueError("Must provide either data_path or data")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.masking_strategy = masking_strategy
        self.mode = mode
        self.stride = stride

        # Calculate number of valid samples
        self.n_samples = (len(self.data) - seq_len - pred_len) // stride + 1

        logger.info(f"Initialized CandleDataset:")
        logger.info(f"  - Mode: {mode}")
        logger.info(f"  - Data shape: {self.data.shape}")
        logger.info(f"  - Seq length: {seq_len}")
        logger.info(f"  - Pred length: {pred_len}")
        logger.info(f"  - Stride: {stride}")
        logger.info(f"  - Total samples: {self.n_samples}")
        logger.info(f"  - Masking: {'Enabled' if masking_strategy else 'Disabled'}")

    def __len__(self) -> int:
        """Return number of samples in dataset"""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single sample

        Returns:
            For training mode (with masking):
                (masked_seq, target, mask_positions, task_type, future)

            For val/test mode (no masking):
                (sequence, future)
        """
        # Calculate actual index with stride
        actual_idx = idx * self.stride

        # Extract sequence and future
        sequence = self.data[actual_idx:actual_idx + self.seq_len]
        future = self.data[actual_idx + self.seq_len:actual_idx + self.seq_len + self.pred_len]

        # Convert to torch tensors
        sequence = torch.FloatTensor(sequence)
        future = torch.FloatTensor(future)

        if self.mode == 'train' and self.masking_strategy:
            # Apply random masking (single sample, so unsqueeze to add batch dim)
            batch_seq = sequence.unsqueeze(0)
            masked_seq, target, mask_pos, task_type = self.masking_strategy(batch_seq)

            # Remove batch dimension
            masked_seq = masked_seq.squeeze(0)
            target = target.squeeze(0)
            mask_pos = mask_pos.squeeze(0)
            task_type = task_type[0]

            return masked_seq, target, mask_pos, task_type, future

        else:
            # No masking for validation/test
            return sequence, future

    def get_sample_batch(self, batch_size: int = 4) -> Tuple:
        """
        Get a sample batch for visualization/debugging

        Args:
            batch_size: Number of samples to return

        Returns:
            Same format as __getitem__, but batched
        """
        indices = np.random.choice(len(self), batch_size, replace=False)
        samples = [self[idx] for idx in indices]

        # Collate based on mode
        if self.mode == 'train' and self.masking_strategy:
            masked_seqs, targets, mask_positions, task_types, futures = zip(*samples)

            return (
                torch.stack(masked_seqs),
                torch.stack(targets),
                torch.stack(mask_positions),
                list(task_types),
                torch.stack(futures)
            )
        else:
            sequences, futures = zip(*samples)
            return torch.stack(sequences), torch.stack(futures)


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    seq_len: int = 100,
    pred_len: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
    masking_strategy: Optional[RandomMaskingStrategy] = None,
    train_stride: int = 1,
    val_stride: int = 10
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders

    Args:
        train_data: Training data array
        val_data: Validation data array
        test_data: Test data array
        seq_len: Sequence length
        pred_len: Prediction length
        batch_size: Batch size
        num_workers: Number of data loading workers
        masking_strategy: Masking strategy for training
        train_stride: Stride for training samples (1 = all possible windows)
        val_stride: Stride for val/test samples (higher = less overlap)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = CandleDataset(
        data=train_data,
        seq_len=seq_len,
        pred_len=pred_len,
        masking_strategy=masking_strategy,
        mode='train',
        stride=train_stride
    )

    val_dataset = CandleDataset(
        data=val_data,
        seq_len=seq_len,
        pred_len=pred_len,
        mode='val',
        stride=val_stride
    )

    test_dataset = CandleDataset(
        data=test_data,
        seq_len=seq_len,
        pred_len=pred_len,
        mode='test',
        stride=val_stride
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  - Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  - Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  - Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def collate_fn_train(batch):
    """
    Custom collate function for training batches with variable-length targets

    Args:
        batch: List of (masked_seq, target, mask_pos, task_type, future)

    Returns:
        Batched tensors
    """
    masked_seqs, targets, mask_positions, task_types, futures = zip(*batch)

    # Stack sequences and futures
    masked_seqs = torch.stack(masked_seqs)
    futures = torch.stack(futures)
    mask_positions = torch.stack(mask_positions)

    # Pad targets to same length
    max_target_len = max(len(t) for t in targets)
    n_features = targets[0].shape[-1]

    padded_targets = torch.zeros(len(targets), max_target_len, n_features)
    for i, target in enumerate(targets):
        padded_targets[i, :len(target)] = target

    return masked_seqs, padded_targets, mask_positions, list(task_types), futures


if __name__ == '__main__':
    # Test dataset
    print("=" * 60)
    print("Testing CandleDataset")
    print("=" * 60)

    # Create synthetic data
    n_samples = 10000
    n_features = 15
    synthetic_data = np.random.randn(n_samples, n_features).astype(np.float32)

    # Initialize masking strategy
    masking_strategy = RandomMaskingStrategy(
        seq_len=100,
        mask_ratio_infill=0.4,
        mask_ratio_forecast=0.4,
        mask_ratio_sparse=0.2
    )

    # Create dataset
    dataset = CandleDataset(
        data=synthetic_data,
        seq_len=100,
        pred_len=10,
        masking_strategy=masking_strategy,
        mode='train',
        stride=1
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"\nSingle sample (training mode):")
    print(f"  - Masked sequence: {sample[0].shape}")
    print(f"  - Target: {sample[1].shape}")
    print(f"  - Mask positions: {sample[2].shape}")
    print(f"  - Task type: {sample[3]}")
    print(f"  - Future: {sample[4].shape}")

    # Test batch
    batch = dataset.get_sample_batch(batch_size=8)
    print(f"\nSample batch (8 samples):")
    print(f"  - Masked sequences: {batch[0].shape}")
    print(f"  - Targets: {batch[1].shape}")
    print(f"  - Mask positions: {batch[2].shape}")
    print(f"  - Task types: {batch[3]}")
    print(f"  - Futures: {batch[4].shape}")

    # Test dataloaders
    print("\n" + "=" * 60)
    print("Testing DataLoaders")
    print("=" * 60)

    # Split synthetic data
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)

    train_data = synthetic_data[:train_size]
    val_data = synthetic_data[train_size:train_size + val_size]
    test_data = synthetic_data[train_size + val_size:]

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        seq_len=100,
        pred_len=10,
        batch_size=64,
        num_workers=0,  # 0 for testing
        masking_strategy=masking_strategy
    )

    # Test iteration
    print(f"\nIterating through train_loader...")
    for i, batch in enumerate(train_loader):
        if i == 0:
            print(f"  Batch 0:")
            print(f"    - Masked sequences: {batch[0].shape}")
            print(f"    - Targets: {batch[1].shape}")
            print(f"    - Mask positions: {batch[2].shape}")
            print(f"    - Task types (first 5): {batch[3][:5]}")
            print(f"    - Futures: {batch[4].shape}")
        if i >= 2:
            break

    print("\n" + "=" * 60)
    print("All dataset tests passed! ✅")
    print("=" * 60)
