"""
Transformer Architecture for Candle Prediction

Components:
- TransformerBlock: Single layer with attention + feedforward
- CandleTransformer: Full transformer stack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from loguru import logger

from .embeddings import TimeSeriesEmbedding
from .attention import DynamicAttention


class TransformerBlock(nn.Module):
    """
    Single Transformer block

    Architecture:
    1. Dynamic Multi-Head Attention (with residual + layer norm)
    2. Feed-Forward Network (with residual + layer norm)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize transformer block

        Args:
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            ff_dim: Feed-forward dimension (usually 4 * hidden_dim)
            dropout: Dropout probability
            activation: Activation function ('gelu', 'relu')
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Dynamic attention
        self.attention = DynamicAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        task_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            task_types: Task types for dynamic attention masking

        Returns:
            Output tensor (batch, seq_len, hidden_dim)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, task_types=task_types)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x


class CandleTransformer(nn.Module):
    """
    Full Transformer model for candle prediction

    Architecture:
    1. Input Embedding (+ position encoding)
    2. Stack of Transformer Blocks
    3. Output Projection
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        use_learnable_pos: bool = True
    ):
        """
        Initialize candle transformer

        Args:
            input_dim: Input feature dimension (OHLCV + indicators)
            hidden_dim: Hidden dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_learnable_pos: Use learnable position encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input embedding
        self.embedding = TimeSeriesEmbedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_len=max_seq_len,
            use_learnable_pos=use_learnable_pos,
            dropout=dropout
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output projection (hidden_dim → input_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # Initialize weights
        self._init_weights()

        logger.info(f"Initialized CandleTransformer:")
        logger.info(f"  - Input dim: {input_dim}")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Num layers: {n_layers}")
        logger.info(f"  - Num heads: {n_heads}")
        logger.info(f"  - FF dim: {ff_dim}")
        logger.info(f"  - Total params: {self.count_parameters():,}")

    def forward(
        self,
        x: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        task_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask_positions: Boolean mask for masked positions (batch, seq_len)
            task_types: Task types for each sample in batch

        Returns:
            Predictions (batch, seq_len, input_dim)
        """
        # Embed input
        x = self.embedding(x, mask_positions=mask_positions)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, task_types=task_types)

        # Project to output
        predictions = self.output_projection(x)

        return predictions

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test transformer model
    print("=" * 60)
    print("Testing CandleTransformer")
    print("=" * 60)

    # Parameters
    batch_size = 4
    seq_len = 100
    input_dim = 15
    hidden_dim = 256
    n_layers = 6
    n_heads = 8
    ff_dim = 1024

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Create sample mask
    mask_positions = torch.rand(batch_size, seq_len) < 0.3

    # Task types (mixed batch)
    task_types = ['infill', 'forecast', 'sparse', 'forecast']

    # Initialize transformer
    transformer = CandleTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout=0.1
    )

    print(f"\nModel size: {transformer.count_parameters():,} parameters")

    # Test forward pass
    print("\n1. Testing Forward Pass")
    print("-" * 60)

    with torch.no_grad():
        predictions = transformer(x, mask_positions, task_types)

    print(f"Input shape: {x.shape}")
    print(f"Mask positions: {mask_positions.sum().item()}/{batch_size * seq_len}")
    print(f"Task types: {task_types}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions mean: {predictions.mean().item():.4f}")
    print(f"Predictions std: {predictions.std().item():.4f}")

    # Test without masking
    print("\n2. Testing Without Masking")
    print("-" * 60)

    with torch.no_grad():
        predictions_no_mask = transformer(x, mask_positions=None, task_types=None)

    print(f"Predictions shape: {predictions_no_mask.shape}")
    print(f"Predictions mean: {predictions_no_mask.mean().item():.4f}")
    print(f"Predictions std: {predictions_no_mask.std().item():.4f}")

    # Test gradient flow
    print("\n3. Testing Gradient Flow")
    print("-" * 60)

    # Forward pass
    predictions = transformer(x, mask_positions, task_types)

    # Create dummy loss
    loss = predictions.mean()

    # Backward pass
    loss.backward()

    # Check gradients
    total_grad_norm = 0
    for name, param in transformer.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2

    total_grad_norm = total_grad_norm ** 0.5

    print(f"Total gradient norm: {total_grad_norm:.4f}")
    print(f"Gradients computed: {total_grad_norm > 0}")

    # Test different batch sizes
    print("\n4. Testing Different Batch Sizes")
    print("-" * 60)

    for bs in [1, 8, 16]:
        x_test = torch.randn(bs, seq_len, input_dim)
        mask_test = torch.rand(bs, seq_len) < 0.3
        task_test = ['forecast'] * bs

        with torch.no_grad():
            pred_test = transformer(x_test, mask_test, task_test)

        print(f"Batch size {bs}: input {x_test.shape} → output {pred_test.shape}")

    # Test different sequence lengths
    print("\n5. Testing Different Sequence Lengths")
    print("-" * 60)

    for sl in [50, 100, 150]:
        x_test = torch.randn(batch_size, sl, input_dim)
        mask_test = torch.rand(batch_size, sl) < 0.3
        task_test = ['infill'] * batch_size

        with torch.no_grad():
            pred_test = transformer(x_test, mask_test, task_test)

        print(f"Seq length {sl}: input {x_test.shape} → output {pred_test.shape}")

    # Memory footprint
    print("\n6. Model Memory Footprint")
    print("-" * 60)

    param_memory = sum(p.numel() * p.element_size() for p in transformer.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in transformer.buffers())
    total_memory = param_memory + buffer_memory

    print(f"Parameter memory: {param_memory / 1024 / 1024:.2f} MB")
    print(f"Buffer memory: {buffer_memory / 1024 / 1024:.2f} MB")
    print(f"Total memory: {total_memory / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 60)
    print("All transformer tests passed! ✅")
    print("=" * 60)
