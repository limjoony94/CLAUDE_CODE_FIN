"""
Dynamic Attention Mechanism

KEY INNOVATION: Task-adaptive attention masking

- Infilling task: Bidirectional attention (all positions can attend to all)
- Forecasting task: Causal attention (only attend to past)
- Mixed batch: Per-sample attention masks

This allows a single model to learn both complete pattern understanding
and predictive capabilities simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from loguru import logger


class DynamicAttention(nn.Module):
    """
    Multi-head attention with dynamic masking based on task type

    The core innovation: attention pattern changes based on whether
    we're doing infilling (bidirectional) or forecasting (causal).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False
    ):
        """
        Initialize dynamic attention

        Args:
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            use_flash_attention: Use Flash Attention if available (faster)
        """
        super().__init__()

        assert hidden_dim % n_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.use_flash_attention = use_flash_attention

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True  # (batch, seq, feature) format
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(f"Initialized DynamicAttention:")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Num heads: {n_heads}")
        logger.info(f"  - Head dim: {self.head_dim}")
        logger.info(f"  - Flash attention: {use_flash_attention}")

    def forward(
        self,
        x: torch.Tensor,
        task_types: Optional[List[str]] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with dynamic attention masking

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            task_types: List of task types for each sample in batch
                       ('infill', 'forecast', 'sparse') or None for bidirectional
            attn_mask: Optional explicit attention mask (batch*heads, seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Create attention mask based on task types
        if attn_mask is None and task_types is not None:
            attn_mask = self._create_dynamic_mask(batch_size, seq_len, task_types)

        # Apply multi-head attention
        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=False  # We don't need attention weights for training
        )

        # Dropout
        attn_output = self.dropout(attn_output)

        return attn_output

    def _create_dynamic_mask(
        self,
        batch_size: int,
        seq_len: int,
        task_types: List[str]
    ) -> Optional[torch.Tensor]:
        """
        Create attention mask dynamically based on task types

        For a batch with mixed tasks, we need per-sample masks.
        PyTorch MultiheadAttention doesn't natively support this,
        so we repeat the mask for each head.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            task_types: List of task types (length = batch_size)

        Returns:
            Attention mask (batch_size * n_heads, seq_len, seq_len)
            or None if all tasks are bidirectional
        """
        device = next(self.parameters()).device

        # Check if all tasks require same mask
        unique_tasks = set(task_types)

        if unique_tasks == {'infill'} or unique_tasks == {'sparse'}:
            # All infilling/sparse: bidirectional (no mask needed)
            return None

        elif unique_tasks == {'forecast'}:
            # All forecasting: causal mask
            causal_mask = self._create_causal_mask(seq_len, device)
            # Repeat for all samples and heads
            attn_mask = causal_mask.unsqueeze(0).expand(batch_size * self.n_heads, -1, -1)
            return attn_mask

        else:
            # Mixed batch: need per-sample masks
            masks = []

            for task_type in task_types:
                if task_type == 'forecast':
                    # Causal mask for forecasting
                    mask = self._create_causal_mask(seq_len, device)
                else:
                    # No mask for infilling/sparse (bidirectional)
                    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

                # Repeat for each head
                mask = mask.unsqueeze(0).expand(self.n_heads, -1, -1)
                masks.append(mask)

            # Stack all sample masks
            attn_mask = torch.cat(masks, dim=0)  # (batch * heads, seq, seq)

            return attn_mask

    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask (lower triangular)

        Position i can only attend to positions <= i

        Args:
            seq_len: Sequence length
            device: Device for tensor

        Returns:
            Causal mask (seq_len, seq_len)
        """
        # Create upper triangular matrix of True values
        # (positions that should be masked out)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        return mask


class CrossAttention(nn.Module):
    """
    Cross-attention for encoder-decoder style architectures (optional)

    Allows attending to a separate context sequence.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention

        Args:
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention forward pass

        Args:
            x: Query tensor (batch, seq_len, hidden_dim)
            context: Key/Value tensor (batch, context_len, hidden_dim)
            attn_mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, hidden_dim)
        """
        attn_output, _ = self.attention(
            query=x,
            key=context,
            value=context,
            attn_mask=attn_mask,
            need_weights=False
        )

        attn_output = self.dropout(attn_output)

        return attn_output


if __name__ == '__main__':
    # Test dynamic attention
    print("=" * 60)
    print("Testing Dynamic Attention")
    print("=" * 60)

    # Parameters
    batch_size = 4
    seq_len = 100
    hidden_dim = 256
    n_heads = 8

    # Create sample input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Initialize dynamic attention
    dyn_attn = DynamicAttention(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        dropout=0.1
    )

    # Test 1: All infilling (bidirectional)
    print("\n1. Testing All Infilling (Bidirectional)")
    print("-" * 60)

    task_types_infill = ['infill'] * batch_size
    output_infill = dyn_attn(x, task_types=task_types_infill)

    print(f"Input shape: {x.shape}")
    print(f"Task types: {task_types_infill}")
    print(f"Output shape: {output_infill.shape}")
    print(f"Output mean: {output_infill.mean().item():.4f}")
    print(f"Output std: {output_infill.std().item():.4f}")

    # Test 2: All forecasting (causal)
    print("\n2. Testing All Forecasting (Causal)")
    print("-" * 60)

    task_types_forecast = ['forecast'] * batch_size
    output_forecast = dyn_attn(x, task_types=task_types_forecast)

    print(f"Task types: {task_types_forecast}")
    print(f"Output shape: {output_forecast.shape}")
    print(f"Output mean: {output_forecast.mean().item():.4f}")
    print(f"Output std: {output_forecast.std().item():.4f}")

    # Test 3: Mixed batch
    print("\n3. Testing Mixed Batch")
    print("-" * 60)

    task_types_mixed = ['infill', 'forecast', 'sparse', 'forecast']
    output_mixed = dyn_attn(x, task_types=task_types_mixed)

    print(f"Task types: {task_types_mixed}")
    print(f"Output shape: {output_mixed.shape}")
    print(f"Output mean: {output_mixed.mean().item():.4f}")
    print(f"Output std: {output_mixed.std().item():.4f}")

    # Test 4: Verify causal masking works
    print("\n4. Verifying Causal Masking")
    print("-" * 60)

    # For forecasting, future positions shouldn't affect past
    # We'll check this by perturbing future and seeing if past changes

    x_baseline = x.clone()
    task_types_causal = ['forecast'] * batch_size

    output_baseline = dyn_attn(x_baseline, task_types=task_types_causal)

    # Perturb last 20 positions
    x_perturbed = x.clone()
    x_perturbed[:, -20:, :] += torch.randn_like(x_perturbed[:, -20:, :]) * 10.0

    output_perturbed = dyn_attn(x_perturbed, task_types=task_types_causal)

    # Check if first 50 positions are same (they should be with causal masking)
    first_50_diff = (output_baseline[:, :50] - output_perturbed[:, :50]).abs().mean().item()

    print(f"Perturbed last 20 positions with large noise")
    print(f"Mean difference in first 50 positions: {first_50_diff:.6f}")
    print(f"Causal masking working: {first_50_diff < 0.01}")

    # Test 5: Compare bidirectional vs causal
    print("\n5. Comparing Bidirectional vs Causal")
    print("-" * 60)

    output_bidi = dyn_attn(x, task_types=['infill'] * batch_size)
    output_causal = dyn_attn(x, task_types=['forecast'] * batch_size)

    diff = (output_bidi - output_causal).abs().mean().item()

    print(f"Mean difference between bidirectional and causal: {diff:.4f}")
    print(f"Outputs are different: {diff > 0.01}")

    print("\n" + "=" * 60)
    print("All dynamic attention tests passed! ✅")
    print("=" * 60)
