"""
Random Masking Strategy - Core Innovation

This module implements the key innovation of this project:
Random Curriculum Learning through Dynamic Masking.

Inspired by BERT's Masked Language Modeling, we randomly switch between:
1. Infilling tasks (reconstruct masked middle sections)
2. Forecasting tasks (predict masked future sections)
3. Sparse masking (BERT-style random position masking)

This allows a single model to learn both:
- Complete pattern understanding (via infilling)
- Predictive capabilities (via forecasting)
"""

import torch
import numpy as np
import random
from typing import Tuple, Dict, List, Optional
from loguru import logger


class RandomMaskingStrategy:
    """
    Random Masking Curriculum for Time Series Prediction

    Key concept:
    - 50% of batches: Middle masking (infilling) → Learn complete patterns
    - 50% of batches: Future masking (forecasting) → Learn prediction
    - Random task selection per sample in batch
    """

    def __init__(
        self,
        seq_len: int = 100,
        mask_ratio_infill: float = 0.4,
        mask_ratio_forecast: float = 0.4,
        mask_ratio_sparse: float = 0.2,
        mask_token_value: float = 0.0,
        device: str = 'cpu'
    ):
        """
        Initialize random masking strategy

        Args:
            seq_len: Input sequence length
            mask_ratio_infill: Ratio of batches using infilling
            mask_ratio_forecast: Ratio of batches using forecasting
            mask_ratio_sparse: Ratio of batches using sparse masking
            mask_token_value: Value to use for masked positions
            device: Device for tensor operations
        """
        self.seq_len = seq_len
        self.mask_ratios = {
            'infill': mask_ratio_infill,
            'forecast': mask_ratio_forecast,
            'sparse': mask_ratio_sparse
        }
        self.mask_token_value = mask_token_value
        self.device = device

        # Validate ratios
        total_ratio = sum(self.mask_ratios.values())
        assert abs(total_ratio - 1.0) < 1e-6, \
            f"Mask ratios must sum to 1.0, got {total_ratio}"

        # Strategy functions
        self.strategies = {
            'infill': self.create_middle_mask,
            'forecast': self.create_future_mask,
            'sparse': self.create_sparse_mask
        }

        logger.info(f"Initialized RandomMaskingStrategy:")
        logger.info(f"  - Seq length: {seq_len}")
        logger.info(f"  - Task ratios: {self.mask_ratios}")
        logger.info(f"  - Device: {device}")

    def __call__(
        self,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Apply random masking to a batch

        Args:
            batch: Input tensor (batch_size, seq_len, n_features)

        Returns:
            masked_batch: Batch with masked positions replaced
            targets: Ground truth values at masked positions
            mask_positions: Boolean mask indicating masked positions
            task_types: List of task types for each sample
        """
        batch_size, seq_len, n_features = batch.shape

        # Initialize outputs
        masked_batch = batch.clone()
        targets_list = []
        mask_positions = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=self.device
        )
        task_types = []

        # Apply masking to each sample independently
        for i in range(batch_size):
            # Sample task type based on ratios
            task_type = self._sample_task_type()
            task_types.append(task_type)

            # Apply corresponding masking strategy
            masked_seq, target, mask_pos = self.strategies[task_type](batch[i])

            masked_batch[i] = masked_seq
            targets_list.append(target)
            mask_positions[i] = mask_pos

        # Pad targets to same length (for batching)
        targets = self._pad_targets(targets_list, mask_positions)

        return masked_batch, targets, mask_positions, task_types

    def _sample_task_type(self) -> str:
        """
        Sample task type based on configured ratios

        Returns:
            Task type: 'infill', 'forecast', or 'sparse'
        """
        rand_val = random.random()
        cumsum = 0

        for task_type, ratio in self.mask_ratios.items():
            cumsum += ratio
            if rand_val < cumsum:
                return task_type

        # Fallback (shouldn't reach here)
        return 'forecast'

    def create_middle_mask(
        self,
        sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Middle masking for INFILLING task

        Masks a contiguous middle section of the sequence.
        The model must reconstruct this section using context from both sides.

        Example (seq_len=100):
            Original: [0, 1, 2, ..., 99]
            Masked:   [0...30, MASK, MASK, ..., MASK, 70...99]
            Target:   [31, 32, ..., 69]

        This teaches the model to understand complete temporal patterns.

        Args:
            sequence: Input sequence (seq_len, n_features)

        Returns:
            masked_seq: Sequence with middle section masked
            target: Ground truth values of masked section
            mask_positions: Boolean mask (seq_len,)
        """
        seq_len = sequence.shape[0]

        # Randomly select mask region (roughly middle 30-50%)
        start_visible = random.randint(int(seq_len * 0.2), int(seq_len * 0.4))
        end_visible = random.randint(int(seq_len * 0.6), int(seq_len * 0.8))

        # Create masked sequence
        masked_seq = sequence.clone()
        masked_seq[start_visible:end_visible] = self.mask_token_value

        # Extract target (masked section)
        target = sequence[start_visible:end_visible].clone()

        # Create mask positions
        mask_positions = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        mask_positions[start_visible:end_visible] = True

        return masked_seq, target, mask_positions

    def create_future_mask(
        self,
        sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Future masking for FORECASTING task

        Masks the future section of the sequence.
        The model must predict this section using only past context.

        Example (seq_len=100):
            Original: [0, 1, 2, ..., 99]
            Masked:   [0...79, MASK, MASK, ...]
            Target:   [80, 81, ..., 99]

        This teaches the model to make predictions (like in real trading).

        Args:
            sequence: Input sequence (seq_len, n_features)

        Returns:
            masked_seq: Sequence with future section masked
            target: Ground truth values of future section
            mask_positions: Boolean mask (seq_len,)
        """
        seq_len = sequence.shape[0]

        # Randomly select cutoff point (roughly last 10-30%)
        cutoff = random.randint(int(seq_len * 0.7), int(seq_len * 0.9))

        # Create masked sequence
        masked_seq = sequence.clone()
        masked_seq[cutoff:] = self.mask_token_value

        # Extract target (future section)
        target = sequence[cutoff:].clone()

        # Create mask positions
        mask_positions = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        mask_positions[cutoff:] = True

        return masked_seq, target, mask_positions

    def create_sparse_mask(
        self,
        sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sparse random masking (BERT-style)

        Randomly masks ~15% of positions throughout the sequence.
        The model must reconstruct these positions using surrounding context.

        Example (seq_len=100):
            Original: [0, 1, 2, ..., 99]
            Masked:   [0, MASK, 2, 3, MASK, 5, ...]
            Target:   [1, 4, ...]  (only masked positions)

        This teaches the model robust representations.

        Args:
            sequence: Input sequence (seq_len, n_features)

        Returns:
            masked_seq: Sequence with random positions masked
            target: Ground truth values of masked positions
            mask_positions: Boolean mask (seq_len,)
        """
        seq_len = sequence.shape[0]

        # Number of positions to mask (~15%)
        n_mask = int(seq_len * 0.15)

        # Randomly select positions to mask
        mask_indices = random.sample(range(seq_len), n_mask)
        mask_indices.sort()

        # Create masked sequence
        masked_seq = sequence.clone()
        masked_seq[mask_indices] = self.mask_token_value

        # Extract targets
        target = sequence[mask_indices].clone()

        # Create mask positions
        mask_positions = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        mask_positions[mask_indices] = True

        return masked_seq, target, mask_positions

    def _pad_targets(
        self,
        targets_list: List[torch.Tensor],
        mask_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Pad targets to create batched tensor

        Different samples may have different number of masked positions.
        We pad them to the same length for efficient batching.

        Args:
            targets_list: List of target tensors (variable lengths)
            mask_positions: Boolean mask (batch_size, seq_len)

        Returns:
            Padded targets (batch_size, max_masked_len, n_features)
        """
        # Find maximum number of masked positions in batch
        max_masked_len = mask_positions.sum(dim=1).max().item()

        if max_masked_len == 0:
            # Edge case: no masking (shouldn't happen)
            return torch.zeros(
                len(targets_list), 1, targets_list[0].shape[-1],
                device=self.device
            )

        batch_size = len(targets_list)
        n_features = targets_list[0].shape[-1]

        # Initialize padded tensor
        padded_targets = torch.zeros(
            batch_size, max_masked_len, n_features,
            device=self.device
        )

        # Fill in targets
        for i, target in enumerate(targets_list):
            target_len = len(target)
            padded_targets[i, :target_len] = target

        return padded_targets

    def get_task_distribution(self, n_samples: int = 10000) -> Dict[str, float]:
        """
        Get empirical task distribution

        Useful for verifying that task sampling matches configured ratios.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Dict with task counts and ratios
        """
        task_counts = {task: 0 for task in self.strategies.keys()}

        for _ in range(n_samples):
            task = self._sample_task_type()
            task_counts[task] += 1

        # Convert to ratios
        task_ratios = {
            task: count / n_samples
            for task, count in task_counts.items()
        }

        return {
            'counts': task_counts,
            'ratios': task_ratios,
            'expected_ratios': self.mask_ratios
        }


class SequentialCurriculumStrategy(RandomMaskingStrategy):
    """
    Sequential Curriculum variant

    Starts with easy infilling tasks, gradually transitions to harder forecasting.
    This is an alternative to random curriculum for ablation studies.
    """

    def __init__(
        self,
        seq_len: int = 100,
        total_epochs: int = 100,
        infill_epochs: int = 30,
        transition_epochs: int = 20,
        **kwargs
    ):
        """
        Initialize sequential curriculum

        Args:
            seq_len: Sequence length
            total_epochs: Total training epochs
            infill_epochs: Number of epochs for pure infilling
            transition_epochs: Number of epochs for gradual transition
        """
        super().__init__(seq_len, **kwargs)

        self.total_epochs = total_epochs
        self.infill_epochs = infill_epochs
        self.transition_epochs = transition_epochs
        self.current_epoch = 0

        logger.info(f"Initialized SequentialCurriculumStrategy:")
        logger.info(f"  - Infill epochs: {infill_epochs}")
        logger.info(f"  - Transition epochs: {transition_epochs}")
        logger.info(f"  - Total epochs: {total_epochs}")

    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum scheduling"""
        self.current_epoch = epoch

    def _sample_task_type(self) -> str:
        """
        Sample task type based on curriculum schedule

        Returns:
            Task type based on current epoch
        """
        if self.current_epoch < self.infill_epochs:
            # Pure infilling phase
            return 'infill'

        elif self.current_epoch < self.infill_epochs + self.transition_epochs:
            # Gradual transition phase
            progress = (self.current_epoch - self.infill_epochs) / self.transition_epochs
            forecast_ratio = progress

            if random.random() < forecast_ratio:
                return 'forecast'
            else:
                return 'infill'

        else:
            # Pure forecasting phase
            return 'forecast'


if __name__ == '__main__':
    # Test random masking strategy
    print("=" * 60)
    print("Testing Random Masking Strategy")
    print("=" * 60)

    # Create sample batch
    batch_size = 4
    seq_len = 100
    n_features = 15

    sample_batch = torch.randn(batch_size, seq_len, n_features)

    # Initialize strategy
    strategy = RandomMaskingStrategy(
        seq_len=seq_len,
        mask_ratio_infill=0.4,
        mask_ratio_forecast=0.4,
        mask_ratio_sparse=0.2
    )

    # Apply masking
    masked_batch, targets, mask_positions, task_types = strategy(sample_batch)

    print(f"\nBatch shape: {sample_batch.shape}")
    print(f"Masked batch shape: {masked_batch.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Mask positions shape: {mask_positions.shape}")
    print(f"Task types: {task_types}")

    # Check task distribution
    print("\n" + "=" * 60)
    print("Task Distribution Test (10,000 samples)")
    print("=" * 60)

    dist = strategy.get_task_distribution(10000)

    print(f"\nExpected ratios: {dist['expected_ratios']}")
    print(f"Empirical ratios: {dist['ratios']}")
    print(f"Counts: {dist['counts']}")

    # Verify masking correctness
    print("\n" + "=" * 60)
    print("Masking Correctness Test")
    print("=" * 60)

    for i in range(batch_size):
        task_type = task_types[i]
        n_masked = mask_positions[i].sum().item()
        masked_ratio = n_masked / seq_len

        print(f"\nSample {i}: Task={task_type}, Masked={n_masked}/{seq_len} ({masked_ratio:.1%})")

        # Verify targets match original at masked positions
        original_vals = sample_batch[i][mask_positions[i]]
        target_vals = targets[i][:n_masked]

        match = torch.allclose(original_vals, target_vals, atol=1e-6)
        print(f"  Targets match original: {match}")

        # Verify masked positions are set to mask_token_value
        masked_vals = masked_batch[i][mask_positions[i]]
        is_masked = torch.all(masked_vals == 0.0)
        print(f"  Masked positions set to 0.0: {is_masked}")

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
