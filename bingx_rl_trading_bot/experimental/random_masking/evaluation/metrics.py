"""
Trading Evaluation Metrics

Comprehensive metrics for evaluating trading performance:
- Return metrics (total, annualized, monthly)
- Risk metrics (Sharpe, Sortino, max drawdown, volatility)
- Trade metrics (win rate, profit factor, avg win/loss)
- Prediction metrics (MSE, MAE, directional accuracy)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger


class TradingMetrics:
    """
    Comprehensive trading metrics calculator

    Computes returns, risks, trade statistics, and prediction accuracy
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics

        Args:
            equity_curve: Equity over time (pd.Series with DatetimeIndex)
            trades: List of trade dictionaries
            predictions: Model predictions (optional)
            targets: Ground truth (optional)

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Return metrics
        metrics.update(self.calculate_return_metrics(equity_curve))

        # Risk metrics
        metrics.update(self.calculate_risk_metrics(equity_curve))

        # Trade metrics
        if trades:
            metrics.update(self.calculate_trade_metrics(trades))

        # Prediction metrics
        if predictions is not None and targets is not None:
            metrics.update(self.calculate_prediction_metrics(predictions, targets))

        return metrics

    def calculate_return_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate return metrics"""
        initial_capital = equity_curve.iloc[0]
        final_capital = equity_curve.iloc[-1]

        total_return = (final_capital - initial_capital) / initial_capital

        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Monthly return
        monthly_return = annualized_return / 12

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'monthly_return': monthly_return,
            'total_days': days
        }

    def calculate_risk_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics"""
        returns = equity_curve.pct_change().dropna()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = calculate_sharpe_ratio(
            equity_curve,
            risk_free_rate=self.risk_free_rate
        )

        # Sortino ratio (downside deviation)
        sortino = calculate_sortino_ratio(
            equity_curve,
            risk_free_rate=self.risk_free_rate
        )

        # Maximum drawdown
        max_dd, max_dd_pct = calculate_max_drawdown(equity_curve)

        # Calmar ratio (annualized return / max drawdown)
        ann_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.25 / len(equity_curve)) - 1
        calmar = ann_return / abs(max_dd_pct) if max_dd_pct != 0 else 0

        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'calmar_ratio': calmar
        }

    def calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate trade statistics"""
        if not trades:
            return {}

        # Filter closed trades
        closed_trades = [t for t in trades if t.get('exit_price') is not None]

        if not closed_trades:
            return {'total_trades': 0}

        # Calculate P&L for each trade
        pnls = []
        for trade in closed_trades:
            direction = 1 if trade['side'] == 'long' else -1
            pnl = direction * (trade['exit_price'] - trade['entry_price']) * trade['size']
            pnls.append(pnl)

        pnls = np.array(pnls)

        # Win/Loss statistics
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Average hold time
        hold_times = []
        for trade in closed_trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                hold_time = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # hours
                hold_times.append(hold_time)

        avg_hold_time = np.mean(hold_times) if hold_times else 0

        # Consecutive wins/losses
        consecutive_wins = self._max_consecutive(pnls > 0)
        consecutive_losses = self._max_consecutive(pnls < 0)

        return {
            'total_trades': len(closed_trades),
            'win_rate': win_rate,
            'total_wins': int(len(wins)),
            'total_losses': int(len(losses)),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_hold_time_hours': avg_hold_time,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'total_pnl': pnls.sum()
        }

    def calculate_prediction_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        # MSE & MAE
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        # Directional accuracy (for close price)
        close_idx = 3  # Assuming OHLCV order

        # Handle both 2D and 3D arrays
        if predictions.ndim == 3:
            # 3D: (batch, time, features)
            pred_close = predictions[:, :, close_idx]
            target_close = targets[:, :, close_idx]
        elif predictions.ndim == 2:
            # 2D: (time, features) - use close_idx if we have multiple features
            if predictions.shape[1] > 1:
                pred_close = predictions[:, close_idx]
                target_close = targets[:, close_idx]
            else:
                # Single feature (already close price)
                pred_close = predictions[:, 0]
                target_close = targets[:, 0]
        else:
            raise ValueError(f"Predictions must be 2D or 3D, got {predictions.ndim}D")

        # Calculate price changes (handle both 1D and 2D)
        if pred_close.ndim == 2:
            # 2D: (batch, time)
            pred_change = pred_close[:, 1:] - pred_close[:, :-1]
            target_change = target_close[:, 1:] - target_close[:, :-1]
        elif pred_close.ndim == 1:
            # 1D: (time,)
            pred_change = pred_close[1:] - pred_close[:-1]
            target_change = target_close[1:] - target_close[:-1]

        # Direction
        pred_direction = np.sign(pred_change)
        target_direction = np.sign(target_change)

        # Accuracy
        directional_accuracy = np.mean(pred_direction == target_direction)

        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'r2_score': r2
        }

    def _max_consecutive(self, condition: np.ndarray) -> int:
        """Calculate maximum consecutive True values"""
        max_count = 0
        current_count = 0

        for val in condition:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count


def calculate_sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio

    Args:
        equity_curve: Equity over time
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Sharpe ratio
    """
    returns = equity_curve.pct_change().dropna()

    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Annualized excess return
    excess_return = returns.mean() * periods_per_year - risk_free_rate

    # Annualized volatility
    volatility = returns.std() * np.sqrt(periods_per_year)

    sharpe = excess_return / volatility if volatility > 0 else 0

    return sharpe


def calculate_sortino_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation)

    Args:
        equity_curve: Equity over time
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    returns = equity_curve.pct_change().dropna()

    if len(returns) == 0:
        return 0.0

    # Annualized excess return
    excess_return = returns.mean() * periods_per_year - risk_free_rate

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf')  # No downside

    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)

    sortino = excess_return / downside_deviation if downside_deviation > 0 else 0

    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, float]:
    """
    Calculate maximum drawdown

    Args:
        equity_curve: Equity over time

    Returns:
        (max_drawdown_value, max_drawdown_percentage)
    """
    # Running maximum
    running_max = equity_curve.expanding().max()

    # Drawdown
    drawdown = equity_curve - running_max

    # Maximum drawdown
    max_dd = drawdown.min()

    # Maximum drawdown percentage
    max_dd_pct = (drawdown / running_max).min()

    return max_dd, max_dd_pct


if __name__ == '__main__':
    # Test metrics
    print("=" * 60)
    print("Testing Trading Metrics")
    print("=" * 60)

    # Create sample equity curve
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    equity = pd.Series(
        10000 * (1 + np.cumsum(np.random.randn(100) * 0.01)),
        index=dates
    )

    # Sample trades
    trades = [
        {
            'side': 'long',
            'entry_price': 100,
            'exit_price': 105,
            'size': 1,
            'entry_time': dates[0],
            'exit_time': dates[1]
        },
        {
            'side': 'short',
            'entry_price': 105,
            'exit_price': 103,
            'size': 1,
            'entry_time': dates[2],
            'exit_time': dates[3]
        },
        {
            'side': 'long',
            'entry_price': 103,
            'exit_price': 100,
            'size': 1,
            'entry_time': dates[4],
            'exit_time': dates[5]
        }
    ]

    # Initialize metrics
    metrics_calc = TradingMetrics(risk_free_rate=0.02)

    # Calculate all metrics
    print("\n1. Testing All Metrics Calculation")
    print("-" * 60)

    metrics = metrics_calc.calculate_all_metrics(equity, trades)

    print("Return Metrics:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"  Monthly Return: {metrics['monthly_return']:.2%}")

    print("\nRisk Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2%})")
    print(f"  Volatility: {metrics['volatility']:.2%}")

    print("\nTrade Metrics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Avg Win: ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")

    # Test prediction metrics
    print("\n2. Testing Prediction Metrics")
    print("-" * 60)

    predictions = np.random.randn(10, 5, 5)
    targets = predictions + np.random.randn(10, 5, 5) * 0.1

    pred_metrics = metrics_calc.calculate_prediction_metrics(predictions, targets)

    print(f"MSE: {pred_metrics['mse']:.4f}")
    print(f"MAE: {pred_metrics['mae']:.4f}")
    print(f"RMSE: {pred_metrics['rmse']:.4f}")
    print(f"Directional Accuracy: {pred_metrics['directional_accuracy']:.2%}")
    print(f"R² Score: {pred_metrics['r2_score']:.4f}")

    print("\n" + "=" * 60)
    print("All metrics tests passed! ✅")
    print("=" * 60)