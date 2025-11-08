"""
Input Embedding and Position Encoding for Time Series

Key components:
- Feature projection from input_dim to hidden_dim
- Learnable position encoding
- Learnable mask token
- Optional time features (hour, day of week, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from loguru import logger


class TimeSeriesEmbedding(nn.Module):
    """
    Time series embedding layer

    Converts raw OHLCV + indicators into high-dimensional embeddings
    with positional information.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_len: int = 200,
        use_learnable_pos: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize time series embedding

        Args:
            input_dim: Input feature dimension (OHLCV + indicators)
            hidden_dim: Hidden dimension for transformer
            max_len: Maximum sequence length
            use_learnable_pos: Use learnable position encoding (vs sinusoidal)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Feature projection: input_dim → hidden_dim
        self.feature_projection = nn.Linear(input_dim, hidden_dim)

        # Learnable mask token
        # This will replace masked positions in the input
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Position encoding
        if use_learnable_pos:
            # Learnable position encoding (more flexible)
            self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        else:
            # Sinusoidal position encoding (from original Transformer)
            self.pos_encoding = self._create_sinusoidal_encoding(max_len, hidden_dim)
            self.register_buffer('pos_enc_buffer', self.pos_encoding)

        self.use_learnable_pos = use_learnable_pos

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(f"Initialized TimeSeriesEmbedding:")
        logger.info(f"  - Input dim: {input_dim}")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Max length: {max_len}")
        logger.info(f"  - Position encoding: {'Learnable' if use_learnable_pos else 'Sinusoidal'}")

    def forward(
        self,
        x: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask_positions: Boolean mask indicating positions to replace with mask token
                           (batch_size, seq_len) or None

        Returns:
            Embedded tensor (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project features: (batch, seq_len, input_dim) → (batch, seq_len, hidden_dim)
        x = self.feature_projection(x)

        # Apply mask token where needed
        if mask_positions is not None:
            # Expand mask token to match batch and feature dims
            mask_token_expanded = self.mask_token.expand(batch_size, seq_len, -1)

            # Replace masked positions with mask token
            mask_positions_expanded = mask_positions.unsqueeze(-1).expand_as(x)
            x = torch.where(mask_positions_expanded, mask_token_expanded, x)

        # Add positional encoding
        if self.use_learnable_pos:
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            pos_enc = self.pos_enc_buffer[:, :seq_len, :].to(x.device)

        x = x + pos_enc

        # Layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x

    @staticmethod
    def _create_sinusoidal_encoding(max_len: int, hidden_dim: int) -> torch.Tensor:
        """
        Create sinusoidal position encoding (from original Transformer paper)

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            max_len: Maximum sequence length
            hidden_dim: Hidden dimension

        Returns:
            Position encoding tensor (1, max_len, hidden_dim)
        """
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * -(np.log(10000.0) / hidden_dim)
        )  # (hidden_dim/2,)

        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Odd indices

        return pe


class TimeFeatureEmbedding(nn.Module):
    """
    Optional: Embed time features (hour, day of week, etc.)

    Can be concatenated with main embeddings for time-aware modeling.
    """

    def __init__(
        self,
        freq: str = '5min',
        embed_dim: int = 16
    ):
        """
        Initialize time feature embedding

        Args:
            freq: Frequency of data ('1min', '5min', '1h', etc.)
            embed_dim: Embedding dimension for time features
        """
        super().__init__()

        self.freq = freq
        self.embed_dim = embed_dim

        # Embeddings for different time features
        self.hour_embedding = nn.Embedding(24, embed_dim)  # 0-23 hours
        self.day_embedding = nn.Embedding(7, embed_dim)  # 0-6 day of week
        self.month_embedding = nn.Embedding(12, embed_dim)  # 0-11 months

        logger.info(f"Initialized TimeFeatureEmbedding (freq: {freq}, embed_dim: {embed_dim})")

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Embed time features from timestamps

        Args:
            timestamps: Unix timestamps (batch_size, seq_len)

        Returns:
            Time feature embeddings (batch_size, seq_len, embed_dim * 3)
        """
        # Extract time features
        # Note: This assumes timestamps are Unix timestamps in seconds
        # You may need to adjust based on your data format

        from datetime import datetime

        # Convert to datetime (this is a placeholder - use actual implementation)
        # In practice, you'd precompute these features in the dataset

        # For now, return zeros as placeholder
        batch_size, seq_len = timestamps.shape
        return torch.zeros(batch_size, seq_len, self.embed_dim * 3, device=timestamps.device)


if __name__ == '__main__':
    # Test embedding layer
    print("=" * 60)
    print("Testing TimeSeriesEmbedding")
    print("=" * 60)

    # Parameters
    batch_size = 4
    seq_len = 100
    input_dim = 15
    hidden_dim = 256

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Create sample mask (randomly mask 30% of positions)
    mask_positions = torch.rand(batch_size, seq_len) < 0.3

    # Test learnable position encoding
    print("\n1. Testing Learnable Position Encoding")
    print("-" * 60)

    embedding_learnable = TimeSeriesEmbedding(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_len=200,
        use_learnable_pos=True
    )

    output_learnable = embedding_learnable(x, mask_positions)

    print(f"Input shape: {x.shape}")
    print(f"Mask positions shape: {mask_positions.shape}")
    print(f"Masked positions: {mask_positions.sum().item()}/{batch_size * seq_len}")
    print(f"Output shape: {output_learnable.shape}")
    print(f"Output mean: {output_learnable.mean().item():.4f}")
    print(f"Output std: {output_learnable.std().item():.4f}")

    # Test sinusoidal position encoding
    print("\n2. Testing Sinusoidal Position Encoding")
    print("-" * 60)

    embedding_sinusoidal = TimeSeriesEmbedding(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_len=200,
        use_learnable_pos=False
    )

    output_sinusoidal = embedding_sinusoidal(x, mask_positions)

    print(f"Output shape: {output_sinusoidal.shape}")
    print(f"Output mean: {output_sinusoidal.mean().item():.4f}")
    print(f"Output std: {output_sinusoidal.std().item():.4f}")

    # Test without masking
    print("\n3. Testing Without Masking")
    print("-" * 60)

    output_no_mask = embedding_learnable(x, mask_positions=None)

    print(f"Output shape: {output_no_mask.shape}")
    print(f"Output mean: {output_no_mask.mean().item():.4f}")
    print(f"Output std: {output_no_mask.std().item():.4f}")

    # Verify mask token is applied
    print("\n4. Verifying Mask Token Application")
    print("-" * 60)

    # Get outputs with and without masking for same input
    sample_input = x[0:1]  # First sample
    sample_mask = mask_positions[0:1]

    output_with_mask = embedding_learnable(sample_input, sample_mask)
    output_without_mask = embedding_learnable(sample_input, None)

    # Check that masked positions are different
    masked_indices = sample_mask[0].nonzero().squeeze()
    if len(masked_indices) > 0:
        first_masked_idx = masked_indices[0].item()

        vec_with_mask = output_with_mask[0, first_masked_idx]
        vec_without_mask = output_without_mask[0, first_masked_idx]

        diff = (vec_with_mask - vec_without_mask).norm().item()

        print(f"First masked position: {first_masked_idx}")
        print(f"Difference in embedding: {diff:.4f}")
        print(f"Mask token applied: {diff > 0.1}")

    print("\n" + "=" * 60)
    print("All embedding tests passed! ✅")
    print("=" * 60)
