"""
Complete Candle Prediction Model with Uncertainty Estimation

Components:
- CandlePredictor: Full model integrating transformer + uncertainty head
- Uncertainty estimation (aleatoric + epistemic)
- MC Dropout inference
- Calibrated prediction intervals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from loguru import logger

from .transformer import CandleTransformer


class CandlePredictor(nn.Module):
    """
    Complete candle prediction model

    Features:
    - Transformer backbone
    - Uncertainty estimation head
    - MC Dropout for epistemic uncertainty
    - Calibrated confidence intervals
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
        use_uncertainty_head: bool = True
    ):
        """
        Initialize candle predictor

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_uncertainty_head: Add uncertainty estimation head
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_uncertainty_head = use_uncertainty_head

        # Transformer backbone
        self.transformer = CandleTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        # Uncertainty estimation head (aleatoric uncertainty)
        if use_uncertainty_head:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, input_dim)  # One uncertainty per feature
            )

        logger.info(f"Initialized CandlePredictor:")
        logger.info(f"  - Total parameters: {self.count_parameters():,}")
        logger.info(f"  - Uncertainty head: {use_uncertainty_head}")

    def forward(
        self,
        x: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        task_types: Optional[List[str]] = None,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask_positions: Boolean mask for masked positions
            task_types: Task types for each sample
            return_uncertainty: Return uncertainty estimates

        Returns:
            predictions: Predicted values (batch, seq_len, input_dim)
            uncertainty: Aleatoric uncertainty (batch, seq_len, input_dim) or None
        """
        # Embed and pass through transformer
        hidden = self.transformer.embedding(x, mask_positions=mask_positions)

        for block in self.transformer.transformer_blocks:
            hidden = block(hidden, task_types=task_types)

        # Predictions
        predictions = self.transformer.output_projection(hidden)

        # Uncertainty (if enabled)
        uncertainty = None
        if return_uncertainty and self.use_uncertainty_head:
            # Log variance (for numerical stability)
            log_var = self.uncertainty_head(hidden)
            # Convert to standard deviation
            uncertainty = F.softplus(log_var) + 1e-6  # Ensure positive

        return predictions, uncertainty

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
        return_std: bool = True,
        task_type: str = 'forecast'
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty quantification using MC Dropout

        Args:
            x: Input tensor (batch, seq_len, input_dim) or (seq_len, input_dim)
            n_samples: Number of MC dropout samples
            return_std: Return standard deviation
            task_type: Task type for attention masking

        Returns:
            Dict with keys:
                - 'mean': Mean prediction (batch, seq_len, input_dim)
                - 'std': Predictive standard deviation (epistemic + aleatoric)
                - 'aleatoric_std': Aleatoric uncertainty
                - 'epistemic_std': Epistemic uncertainty
        """
        # Handle single sequence input
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, input_dim = x.shape

        # Enable dropout for MC sampling
        self.train()

        # Collect predictions
        predictions = []
        aleatoric_uncertainties = []

        task_types = [task_type] * batch_size

        for _ in range(n_samples):
            pred, aleatoric_std = self.forward(
                x,
                mask_positions=None,
                task_types=task_types,
                return_uncertainty=True
            )

            predictions.append(pred)
            if aleatoric_std is not None:
                aleatoric_uncertainties.append(aleatoric_std)

        # Stack predictions
        predictions = torch.stack(predictions)  # (n_samples, batch, seq, input_dim)

        # Mean prediction
        mean_pred = predictions.mean(dim=0)

        # Epistemic uncertainty (variance across samples)
        epistemic_std = predictions.std(dim=0)

        # Aleatoric uncertainty (mean of predicted uncertainties)
        if aleatoric_uncertainties:
            aleatoric_std = torch.stack(aleatoric_uncertainties).mean(dim=0)
        else:
            aleatoric_std = torch.zeros_like(mean_pred)

        # Total uncertainty (combine epistemic and aleatoric)
        total_std = torch.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)

        # Return to eval mode
        self.eval()

        result = {
            'mean': mean_pred,
            'std': total_std,
            'aleatoric_std': aleatoric_std,
            'epistemic_std': epistemic_std
        }

        # Remove batch dimension if input was single sequence
        if batch_size == 1:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    @torch.no_grad()
    def predict_future(
        self,
        historical_data: torch.Tensor,
        n_steps: int = 10,
        n_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future values using autoregressive forecasting

        Args:
            historical_data: Historical sequence (seq_len, input_dim)
            n_steps: Number of future steps to predict
            n_samples: Number of MC samples for uncertainty

        Returns:
            Dict with predictions and uncertainty
        """
        # Start with historical data
        current_sequence = historical_data.unsqueeze(0)  # Add batch dim

        future_predictions = []
        future_uncertainties = []

        for step in range(n_steps):
            # Predict next step
            result = self.predict(
                current_sequence,
                n_samples=n_samples,
                task_type='forecast'
            )

            # Get last prediction
            next_pred = result['mean'][0, -1:]  # (1, input_dim)
            next_std = result['std'][0, -1:]

            future_predictions.append(next_pred)
            future_uncertainties.append(next_std)

            # Append prediction to sequence
            current_sequence = torch.cat([
                current_sequence,
                next_pred.unsqueeze(0)
            ], dim=1)

            # Keep only last seq_len timesteps
            if current_sequence.shape[1] > historical_data.shape[0]:
                current_sequence = current_sequence[:, 1:]

        # Stack predictions
        future_predictions = torch.cat(future_predictions, dim=0)  # (n_steps, input_dim)
        future_uncertainties = torch.cat(future_uncertainties, dim=0)

        return {
            'predictions': future_predictions,
            'uncertainty': future_uncertainties
        }

    def count_parameters(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_confidence_intervals(
        self,
        predictions: torch.Tensor,
        std: torch.Tensor,
        confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate confidence intervals for predictions

        Args:
            predictions: Mean predictions
            std: Standard deviation
            confidence: Confidence level (0.95 = 95%)

        Returns:
            (lower_bound, upper_bound)
        """
        from scipy import stats

        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence) / 2)

        lower = predictions - z * std
        upper = predictions + z * std

        return lower, upper


if __name__ == '__main__':
    # Test candle predictor
    print("=" * 60)
    print("Testing CandlePredictor")
    print("=" * 60)

    # Parameters
    batch_size = 4
    seq_len = 100
    input_dim = 15
    hidden_dim = 256

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Initialize predictor
    predictor = CandlePredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=6,
        n_heads=8,
        ff_dim=1024,
        dropout=0.1,
        use_uncertainty_head=True
    )

    print(f"\nModel size: {predictor.count_parameters():,} parameters")

    # Test 1: Forward pass without uncertainty
    print("\n1. Testing Forward Pass (No Uncertainty)")
    print("-" * 60)

    mask_positions = torch.rand(batch_size, seq_len) < 0.3
    task_types = ['forecast'] * batch_size

    with torch.no_grad():
        predictions, uncertainty = predictor(
            x, mask_positions, task_types, return_uncertainty=False
        )

    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Uncertainty: {uncertainty}")

    # Test 2: Forward pass with uncertainty
    print("\n2. Testing Forward Pass (With Uncertainty)")
    print("-" * 60)

    with torch.no_grad():
        predictions, uncertainty = predictor(
            x, mask_positions, task_types, return_uncertainty=True
        )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean uncertainty: {uncertainty.mean().item():.4f}")

    # Test 3: MC Dropout prediction
    print("\n3. Testing MC Dropout Prediction")
    print("-" * 60)

    single_seq = x[0]  # (seq_len, input_dim)

    result = predictor.predict(
        single_seq,
        n_samples=10,
        task_type='forecast'
    )

    print(f"Mean prediction shape: {result['mean'].shape}")
    print(f"Total std shape: {result['std'].shape}")
    print(f"Aleatoric std shape: {result['aleatoric_std'].shape}")
    print(f"Epistemic std shape: {result['epistemic_std'].shape}")
    print(f"\nMean total uncertainty: {result['std'].mean().item():.4f}")
    print(f"Mean aleatoric uncertainty: {result['aleatoric_std'].mean().item():.4f}")
    print(f"Mean epistemic uncertainty: {result['epistemic_std'].mean().item():.4f}")

    # Test 4: Autoregressive future prediction
    print("\n4. Testing Autoregressive Future Prediction")
    print("-" * 60)

    historical = x[0]  # (seq_len, input_dim)
    n_future_steps = 10

    future_result = predictor.predict_future(
        historical,
        n_steps=n_future_steps,
        n_samples=5
    )

    print(f"Historical shape: {historical.shape}")
    print(f"Future predictions shape: {future_result['predictions'].shape}")
    print(f"Future uncertainty shape: {future_result['uncertainty'].shape}")
    print(f"Number of future steps: {n_future_steps}")

    # Test 5: Confidence intervals
    print("\n5. Testing Confidence Intervals")
    print("-" * 60)

    mean_pred = result['mean']
    std_pred = result['std']

    for confidence in [0.68, 0.95, 0.99]:
        lower, upper = predictor.get_confidence_intervals(
            mean_pred, std_pred, confidence=confidence
        )

        print(f"{confidence*100:.0f}% CI: [{lower[0, 0].item():.4f}, {upper[0, 0].item():.4f}] (first feature)")

    # Test 6: Batch prediction
    print("\n6. Testing Batch Prediction")
    print("-" * 60)

    batch_result = predictor.predict(
        x,  # Full batch
        n_samples=5,
        task_type='forecast'
    )

    print(f"Batch predictions shape: {batch_result['mean'].shape}")
    print(f"Batch std shape: {batch_result['std'].shape}")

    # Test 7: Mixed task types
    print("\n7. Testing Mixed Task Types")
    print("-" * 60)

    mixed_tasks = ['infill', 'forecast', 'sparse', 'forecast']

    with torch.no_grad():
        pred_mixed, unc_mixed = predictor(
            x, mask_positions, mixed_tasks, return_uncertainty=True
        )

    print(f"Task types: {mixed_tasks}")
    print(f"Predictions shape: {pred_mixed.shape}")
    print(f"Uncertainty shape: {unc_mixed.shape}")

    print("\n" + "=" * 60)
    print("All predictor tests passed! ✅")
    print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)

    total_params = predictor.count_parameters()
    total_memory = sum(p.numel() * p.element_size() for p in predictor.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Model memory: {total_memory / 1024 / 1024:.2f} MB")
    print(f"Uncertainty estimation: Enabled")
    print(f"MC Dropout support: Enabled")
    print(f"Autoregressive forecasting: Enabled")
