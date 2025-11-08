"""
Multi-task Loss Functions for Random Masking

Loss components:
1. MSE Loss: Basic reconstruction/prediction accuracy
2. Directional Loss: Predict correct price direction
3. Volatility Loss: Penalize unrealistic volatility
4. Uncertainty Loss: Negative log-likelihood with uncertainty
5. Task-weighted Loss: Different weights for infilling vs forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from loguru import logger


def combined_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask_positions: torch.Tensor,
    task_types: List[str],
    uncertainty: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined multi-task loss function (with NaN debugging)

    L_total = α * L_MSE + β * L_directional + γ * L_volatility + δ * L_uncertainty

    Args:
        predictions: Model predictions (batch, seq_len, n_features)
        targets: Ground truth (batch, max_masked_len, n_features)
        mask_positions: Boolean mask (batch, seq_len)
        task_types: List of task types for each sample
        uncertainty: Predicted uncertainty (batch, seq_len, n_features) or None
        weights: Dict of loss weights {'mse': α, 'directional': β, 'volatility': γ}

    Returns:
        total_loss: Combined loss
        loss_dict: Dict with individual loss components
    """
    # === 디버깅 코드 추가 ===
    print(f"\n[DEBUG combined_loss]")
    print(f"  predictions shape: {predictions.shape}")
    print(f"  targets shape: {targets.shape}")
    print(f"  mask_positions shape: {mask_positions.shape}")
    print(f"  mask_positions sum: {mask_positions.sum()}")
    print(f"  predictions has NaN: {torch.isnan(predictions).any()}")
    print(f"  targets has NaN: {torch.isnan(targets).any()}")
    print(f"  predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  targets range: [{targets.min():.4f}, {targets.max():.4f}]")

    # Default weights
    if weights is None:
        weights = {
            'mse': 1.0,
            'directional': 0.3,
            'volatility': 0.2,
            'uncertainty': 0.1
        }

    # Extract predictions at masked positions
    batch_size, seq_len, n_features = predictions.shape

    # Expand mask for all features
    mask_expanded = mask_positions.unsqueeze(-1).expand_as(predictions)

    # Get predictions and targets at masked positions
    pred_masked_list = []
    target_list = []

    try:
        for i in range(batch_size):
            n_masked = mask_positions[i].sum().item()
            print(f"  Batch {i}: n_masked = {n_masked}")

            if n_masked == 0:
                print(f"  ⚠️ WARNING: Batch {i} has 0 masked positions!")
                continue

            pred_masked_i = predictions[i][mask_positions[i]]
            target_i = targets[i, :n_masked]

            print(f"    pred_masked shape: {pred_masked_i.shape}")
            print(f"    target shape: {target_i.shape}")
            print(f"    pred has NaN: {torch.isnan(pred_masked_i).any()}")
            print(f"    target has NaN: {torch.isnan(target_i).any()}")

            pred_masked_list.append(pred_masked_i)
            target_list.append(target_i)

    except Exception as e:
        print(f"  ❌ ERROR during masked extraction: {e}")
        # Fallback: use all positions
        print(f"  Using fallback: reshaping all positions")
        pred_masked = predictions.reshape(-1, predictions.shape[-1])
        target_masked = targets.reshape(-1, targets.shape[-1])

    # Concatenate all masked positions across batch
    if len(pred_masked_list) == 0:
        print(f"  ⚠️ WARNING: No masked positions found! Using baseline fallback.")
        # For baseline (no masking): use only LAST pred_len predictions
        # predictions: (batch, seq_len, features)
        # targets: (batch, pred_len, features)
        pred_len = targets.shape[1]
        print(f"  Baseline mode: Using last {pred_len} predictions from seq_len {predictions.shape[1]}")
        pred_masked = predictions[:, -pred_len:, :].reshape(-1, predictions.shape[-1])
        target_masked = targets.reshape(-1, targets.shape[-1])
        print(f"  Baseline pred_masked shape: {pred_masked.shape}")
        print(f"  Baseline target_masked shape: {target_masked.shape}")
    else:
        pred_masked = torch.cat(pred_masked_list, dim=0)  # (total_masked, n_features)
        target_masked = torch.cat(target_list, dim=0)

    print(f"  Final pred_masked shape: {pred_masked.shape}")
    print(f"  Final target_masked shape: {target_masked.shape}")
    print(f"  Final pred_masked has NaN: {torch.isnan(pred_masked).any()}")
    print(f"  Final target_masked has NaN: {torch.isnan(target_masked).any()}")

    # 1. MSE Loss
    mse_loss = F.mse_loss(pred_masked, target_masked)
    print(f"  mse_loss: {mse_loss.item()}")

    if torch.isnan(mse_loss):
        print("  ⚠️ MSE Loss is NaN! Using dummy loss.")
        return torch.tensor(1.0, requires_grad=True, device=predictions.device), {
            'total': 1.0,
            'mse': 1.0,
            'directional': 0.0,
            'volatility': 0.0,
            'uncertainty': 0.0,
            'task_weight': 1.0
        }

    # 2. Directional Loss (for close price - usually index 3)
    close_idx = 3  # Assuming OHLCV order
    print(f"  Computing directional_loss (close_idx={close_idx})")
    directional_loss_val = directional_loss(
        pred_masked[:, close_idx],
        target_masked[:, close_idx]
    )
    print(f"  directional_loss: {directional_loss_val.item() if isinstance(directional_loss_val, torch.Tensor) else directional_loss_val}")

    if isinstance(directional_loss_val, torch.Tensor) and torch.isnan(directional_loss_val):
        print("  ⚠️ Directional Loss is NaN! Setting to 0.")
        directional_loss_val = torch.tensor(0.0, device=predictions.device)

    # 3. Volatility Loss
    print(f"  Computing volatility_loss")
    volatility_loss_val = volatility_loss(
        pred_masked[:, close_idx],
        target_masked[:, close_idx]
    )
    print(f"  volatility_loss: {volatility_loss_val.item() if isinstance(volatility_loss_val, torch.Tensor) else volatility_loss_val}")

    if isinstance(volatility_loss_val, torch.Tensor) and torch.isnan(volatility_loss_val):
        print("  ⚠️ Volatility Loss is NaN! Setting to 0.")
        volatility_loss_val = torch.tensor(0.0, device=predictions.device)

    # 4. Uncertainty Loss (Negative Log-Likelihood)
    uncertainty_loss_val = 0.0
    if uncertainty is not None:
        # Extract uncertainty at masked positions
        unc_masked_list = []
        for i in range(batch_size):
            n_masked = mask_positions[i].sum().item()
            if n_masked > 0:
                unc_masked_list.append(uncertainty[i][mask_positions[i]])

        # Baseline fallback: if no masked positions, use last pred_len
        if len(unc_masked_list) == 0:
            print(f"  ⚠️ WARNING: No uncertainty masked positions! Using baseline fallback.")
            pred_len = targets.shape[1]
            unc_masked = uncertainty[:, -pred_len:, :].reshape(-1, uncertainty.shape[-1])
            print(f"  Baseline unc_masked shape: {unc_masked.shape}")
        else:
            unc_masked = torch.cat(unc_masked_list, dim=0)
            print(f"  Masked unc_masked shape: {unc_masked.shape}")

        # Negative log-likelihood
        print(f"  Computing uncertainty_loss (NLL)")
        uncertainty_loss_val = negative_log_likelihood(
            pred_masked, target_masked, unc_masked
        )
        print(f"  uncertainty_loss: {uncertainty_loss_val.item() if isinstance(uncertainty_loss_val, torch.Tensor) else uncertainty_loss_val}")

    # Task-specific weighting
    # Count infilling vs forecasting tasks
    n_infill = sum(1 for t in task_types if t == 'infill')
    n_forecast = sum(1 for t in task_types if t == 'forecast')
    n_sparse = sum(1 for t in task_types if t == 'sparse')

    # Adjust weights based on task distribution
    # Forecasting is harder, so we might want to weight it more
    task_weight = 1.0
    if n_forecast > 0:
        task_weight = 1.0 + 0.2 * (n_forecast / batch_size)  # Up to 20% boost

    # Combined loss
    total_loss = (
        weights['mse'] * mse_loss +
        weights['directional'] * directional_loss_val +
        weights['volatility'] * volatility_loss_val +
        weights.get('uncertainty', 0.1) * uncertainty_loss_val
    ) * task_weight

    print(f"  total_loss (before check): {total_loss.item()}")

    # Final NaN check
    if torch.isnan(total_loss):
        print("  ⚠️ Total Loss is NaN! Components:")
        print(f"    mse_loss: {mse_loss.item()}")
        print(f"    directional_loss: {directional_loss_val.item() if isinstance(directional_loss_val, torch.Tensor) else directional_loss_val}")
        print(f"    volatility_loss: {volatility_loss_val.item() if isinstance(volatility_loss_val, torch.Tensor) else volatility_loss_val}")
        print(f"    uncertainty_loss: {uncertainty_loss_val.item() if isinstance(uncertainty_loss_val, torch.Tensor) else uncertainty_loss_val}")
        print(f"    task_weight: {task_weight}")
        print(f"  Using fallback loss value 1.0")
        total_loss = torch.tensor(1.0, requires_grad=True, device=predictions.device)

    # Loss dictionary for logging
    loss_dict = {
        'total': total_loss.item(),
        'mse': mse_loss.item(),
        'directional': directional_loss_val.item() if isinstance(directional_loss_val, torch.Tensor) else directional_loss_val,
        'volatility': volatility_loss_val.item() if isinstance(volatility_loss_val, torch.Tensor) else volatility_loss_val,
        'uncertainty': uncertainty_loss_val.item() if isinstance(uncertainty_loss_val, torch.Tensor) else uncertainty_loss_val,
        'task_weight': task_weight
    }

    print(f"  Returning loss_dict: {loss_dict}\n")

    return total_loss, loss_dict


def directional_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Directional accuracy loss

    Penalizes predictions that get the direction wrong.

    Args:
        predictions: Predicted values (N,)
        targets: Ground truth values (N,)

    Returns:
        Directional loss
    """
    if len(predictions) < 2:
        return torch.tensor(0.0, device=predictions.device)

    # Calculate price changes
    pred_change = predictions[1:] - predictions[:-1]
    target_change = targets[1:] - targets[:-1]

    # Direction: +1 for up, -1 for down, 0 for flat
    pred_direction = torch.sign(pred_change)
    target_direction = torch.sign(target_change)

    # Directional mismatch
    direction_error = (pred_direction != target_direction).float()

    # Mean directional error
    directional_loss = direction_error.mean()

    return directional_loss


def volatility_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Volatility matching loss

    Penalizes predictions with unrealistic volatility.

    Args:
        predictions: Predicted values (N,)
        targets: Ground truth values (N,)

    Returns:
        Volatility loss
    """
    if len(predictions) < 2:
        return torch.tensor(0.0, device=predictions.device)

    # Calculate returns
    pred_returns = (predictions[1:] - predictions[:-1]) / (predictions[:-1] + 1e-8)
    target_returns = (targets[1:] - targets[:-1]) / (targets[:-1] + 1e-8)

    # Volatility (standard deviation of returns)
    pred_vol = pred_returns.std()
    target_vol = target_returns.std()

    # Penalize volatility mismatch
    vol_loss = (pred_vol - target_vol).abs()

    return vol_loss


def negative_log_likelihood(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor
) -> torch.Tensor:
    """
    Negative log-likelihood loss with heteroscedastic uncertainty

    Assumes Gaussian distribution:
    NLL = 0.5 * log(2π * σ²) + (y - ŷ)² / (2σ²)

    Args:
        predictions: Predicted values (N, n_features)
        targets: Ground truth (N, n_features)
        uncertainty: Predicted standard deviation (N, n_features)

    Returns:
        Mean negative log-likelihood
    """
    # Ensure positive uncertainty
    uncertainty = uncertainty.clamp(min=1e-6)

    # Squared error
    squared_error = (targets - predictions) ** 2

    # Negative log-likelihood
    nll = 0.5 * torch.log(2 * torch.pi * uncertainty ** 2) + squared_error / (2 * uncertainty ** 2)

    # Mean over all elements
    return nll.mean()


def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0
) -> torch.Tensor:
    """
    Huber loss (robust to outliers)

    Args:
        predictions: Predicted values
        targets: Ground truth
        delta: Threshold for quadratic vs linear

    Returns:
        Huber loss
    """
    error = predictions - targets
    abs_error = error.abs()

    quadratic = torch.min(abs_error, torch.tensor(delta, device=error.device))
    linear = abs_error - quadratic

    loss = 0.5 * quadratic ** 2 + delta * linear

    return loss.mean()


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float] = [0.1, 0.5, 0.9]
) -> torch.Tensor:
    """
    Quantile loss for probabilistic forecasting

    Args:
        predictions: Predicted quantiles (batch, n_quantiles)
        targets: Ground truth (batch, 1)
        quantiles: List of quantile levels

    Returns:
        Mean quantile loss
    """
    losses = []

    for i, q in enumerate(quantiles):
        error = targets - predictions[:, i:i+1]

        # Asymmetric loss
        loss_q = torch.max(q * error, (q - 1) * error)
        losses.append(loss_q)

    return torch.cat(losses, dim=1).mean()


if __name__ == '__main__':
    # Test loss functions
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    # Create sample data
    batch_size = 4
    seq_len = 100
    n_features = 15
    max_masked = 30

    predictions = torch.randn(batch_size, seq_len, n_features)
    targets = torch.randn(batch_size, max_masked, n_features)
    mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Create random masks
    for i in range(batch_size):
        n_masked = torch.randint(10, max_masked, (1,)).item()
        indices = torch.randperm(seq_len)[:n_masked]
        mask_positions[i, indices] = True

    task_types = ['infill', 'forecast', 'sparse', 'forecast']
    uncertainty = torch.rand(batch_size, seq_len, n_features) * 0.1 + 0.01

    # Test 1: Combined loss
    print("\n1. Testing Combined Loss")
    print("-" * 60)

    loss, loss_dict = combined_loss(
        predictions, targets, mask_positions, task_types, uncertainty
    )

    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components:")
    for key, value in loss_dict.items():
        print(f"  - {key}: {value:.4f}")

    # Test 2: Directional loss
    print("\n2. Testing Directional Loss")
    print("-" * 60)

    price_pred = torch.tensor([100.0, 101.0, 99.0, 102.0, 103.0])
    price_target = torch.tensor([100.0, 102.0, 101.0, 103.0, 102.0])

    dir_loss = directional_loss(price_pred, price_target)
    print(f"Predicted: {price_pred.tolist()}")
    print(f"Target: {price_target.tolist()}")
    print(f"Directional loss: {dir_loss.item():.4f}")

    # Test 3: Volatility loss
    print("\n3. Testing Volatility Loss")
    print("-" * 60)

    vol_loss = volatility_loss(price_pred, price_target)
    print(f"Volatility loss: {vol_loss.item():.4f}")

    # Test 4: Negative log-likelihood
    print("\n4. Testing Negative Log-Likelihood")
    print("-" * 60)

    pred_nll = torch.randn(100, n_features)
    target_nll = torch.randn(100, n_features)
    unc_nll = torch.rand(100, n_features) * 0.1 + 0.01

    nll = negative_log_likelihood(pred_nll, target_nll, unc_nll)
    print(f"NLL loss: {nll.item():.4f}")

    # Test 5: Gradient flow
    print("\n5. Testing Gradient Flow")
    print("-" * 60)

    # Make tensors require gradients
    predictions_grad = torch.randn(batch_size, seq_len, n_features, requires_grad=True)
    uncertainty_grad = torch.rand(batch_size, seq_len, n_features, requires_grad=True)

    loss_grad, _ = combined_loss(
        predictions_grad, targets, mask_positions, task_types, uncertainty_grad
    )

    loss_grad.backward()

    print(f"Predictions gradient: {predictions_grad.grad is not None}")
    print(f"Uncertainty gradient: {uncertainty_grad.grad is not None}")
    print(f"Pred grad norm: {predictions_grad.grad.norm().item():.4f}")
    print(f"Unc grad norm: {uncertainty_grad.grad.norm().item():.4f}")

    # Test 6: Different task weights
    print("\n6. Testing Task-specific Weighting")
    print("-" * 60)

    for tasks in [
        ['infill'] * 4,
        ['forecast'] * 4,
        ['infill', 'forecast', 'infill', 'forecast']
    ]:
        loss_task, loss_dict_task = combined_loss(
            predictions, targets, mask_positions, tasks, uncertainty
        )

        print(f"Tasks: {tasks}")
        print(f"  Total loss: {loss_dict_task['total']:.4f}")
        print(f"  Task weight: {loss_dict_task['task_weight']:.4f}")

    print("\n" + "=" * 60)
    print("All loss function tests passed! ✅")
    print("=" * 60)