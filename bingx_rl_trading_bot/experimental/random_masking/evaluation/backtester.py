"""
Backtesting Engine for Random Masking Candle Predictor

Features:
- Walk-forward validation
- Realistic position management
- Slippage and fee simulation
- Multiple exit strategies (TP/SL/Time)
- Detailed trade logging
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from ..trading.signal_generator import SignalGenerator
from ..trading.risk_manager import RiskManager
from .metrics import TradingMetrics


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # Max 10% of capital per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    max_hold_candles: int = 48  # Maximum hold time (48 * 5min = 4 hours)
    slippage_pct: float = 0.0005  # 0.05% slippage
    fee_pct: float = 0.0004  # 0.04% taker fee (BingX)
    leverage: float = 1.0  # Leverage multiplier
    kelly_fraction: float = 0.25  # Kelly criterion fraction


@dataclass
class BacktestResults:
    """Backtesting results"""
    equity_curve: pd.Series
    trades: List[Dict]
    metrics: Dict[str, float]
    daily_returns: pd.Series
    predictions_df: Optional[pd.DataFrame] = None

    # Summary statistics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


class Backtester:
    """
    Walk-forward backtesting engine

    Simulates realistic trading with:
    - Progressive data revelation (no look-ahead bias)
    - Position management (entry/exit logic)
    - Slippage and fees
    - Risk management
    """

    def __init__(
        self,
        model: nn.Module,
        config: BacktestConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize backtester

        Args:
            model: Trained CandlePredictor model
            config: Backtesting configuration
            device: Device for model inference
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Signal generator and risk manager
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(config)

        # Metrics calculator
        self.metrics_calc = TradingMetrics()

        logger.info(f"Initialized Backtester:")
        logger.info(f"  - Initial capital: ${config.initial_capital:,.2f}")
        logger.info(f"  - Max position size: {config.max_position_size:.1%}")
        logger.info(f"  - Stop loss: {config.stop_loss_pct:.2%}")
        logger.info(f"  - Take profit: {config.take_profit_pct:.2%}")

    def run(
        self,
        data: pd.DataFrame,
        seq_len: int = 100,
        verbose: bool = True
    ) -> BacktestResults:
        """
        Run walk-forward backtest

        Args:
            data: Historical OHLCV data with features
            seq_len: Sequence length for model input
            verbose: Print progress

        Returns:
            BacktestResults object
        """
        logger.info(f"Starting backtest on {len(data)} candles...")

        # Initialize state
        capital = self.config.initial_capital
        position = None  # Current position
        trades = []
        equity_history = []
        predictions_list = []

        # Walk forward through data
        for i in range(seq_len, len(data)):
            current_time = data.index[i]
            current_candle = data.iloc[i]

            # Get historical sequence
            sequence = data.iloc[i - seq_len:i].values
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            # Model prediction
            with torch.no_grad():
                result = self.model.predict(
                    sequence_tensor,
                    n_samples=10,
                    task_type='forecast'
                )

            # Get prediction for next candle
            # Handle both 2D (batch squeezed) and 3D tensors
            if result['mean'].dim() == 3:
                mean_pred = result['mean'][0, -1, :].cpu().numpy()  # (n_features,)
                uncertainty = result['std'][0, -1, :].cpu().numpy()
            elif result['mean'].dim() == 2:
                mean_pred = result['mean'][-1, :].cpu().numpy()  # (n_features,)
                uncertainty = result['std'][-1, :].cpu().numpy()
            else:
                raise ValueError(f"Unexpected prediction shape: {result['mean'].shape}")

            # Store predictions
            predictions_list.append({
                'time': current_time,
                'pred_close': mean_pred[3],  # Close price prediction
                'uncertainty': uncertainty[3],
                'actual_close': current_candle['close']
            })

            # Check if we have a position
            if position is not None:
                # Check exit conditions
                should_exit, exit_reason = self._check_exit_conditions(
                    position, current_candle, i
                )

                if should_exit:
                    # Exit position
                    trade = self._close_position(
                        position, current_candle, exit_reason
                    )
                    trades.append(trade)

                    # Update capital
                    capital += trade['pnl']
                    position = None

                    if verbose and len(trades) % 10 == 0:
                        logger.info(f"Trade #{len(trades)}: {exit_reason}, P&L: ${trade['pnl']:.2f}")

            # If no position, check entry signal
            if position is None:
                signal = self.signal_generator.generate_signal(
                    mean_pred,
                    uncertainty,
                    current_candle
                )

                if signal['action'] != 'hold':
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        capital,
                        signal['confidence'],
                        current_candle['close']
                    )

                    if position_size > 0:
                        # Enter position
                        position = self._open_position(
                            signal['action'],
                            current_candle,
                            position_size,
                            i
                        )

            # Record equity
            equity = capital
            if position is not None:
                # Add unrealized P&L
                current_price = current_candle['close']
                unrealized_pnl = self._calculate_pnl(
                    position['side'],
                    position['entry_price'],
                    current_price,
                    position['size']
                )
                equity += unrealized_pnl

            equity_history.append({
                'time': current_time,
                'equity': equity
            })

        # Close any remaining position
        if position is not None:
            final_candle = data.iloc[-1]
            trade = self._close_position(position, final_candle, 'end_of_data')
            trades.append(trade)
            capital += trade['pnl']

        # Create results
        equity_curve = pd.Series(
            [e['equity'] for e in equity_history],
            index=[e['time'] for e in equity_history]
        )

        predictions_df = pd.DataFrame(predictions_list).set_index('time')

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(
            equity_curve,
            trades,
            predictions=predictions_df[['pred_close']].values,
            targets=predictions_df[['actual_close']].values
        )

        # Calculate daily returns
        daily_returns = equity_curve.resample('D').last().pct_change().dropna()

        # Summary
        results = BacktestResults(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            daily_returns=daily_returns,
            predictions_df=predictions_df,
            total_return=metrics.get('total_return', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            max_drawdown_pct=metrics.get('max_drawdown_pct', 0),
            win_rate=metrics.get('win_rate', 0),
            profit_factor=metrics.get('profit_factor', 0)
        )

        logger.info(f"\nBacktest Complete!")
        logger.info(f"  Total Return: {results.total_return:.2%}")
        logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {results.max_drawdown_pct:.2%}")
        logger.info(f"  Win Rate: {results.win_rate:.2%}")
        logger.info(f"  Total Trades: {len(trades)}")

        return results

    def _open_position(
        self,
        side: str,
        candle: pd.Series,
        size: float,
        index: int
    ) -> Dict:
        """Open a new position"""
        entry_price = candle['close']

        # Apply slippage
        if side == 'long':
            entry_price *= (1 + self.config.slippage_pct)
        else:
            entry_price *= (1 - self.config.slippage_pct)

        # Apply leverage
        size *= self.config.leverage

        position = {
            'side': side,
            'entry_price': entry_price,
            'entry_time': candle.name,
            'entry_index': index,
            'size': size,
            'stop_loss': self._calculate_stop_loss(side, entry_price),
            'take_profit': self._calculate_take_profit(side, entry_price)
        }

        return position

    def _close_position(
        self,
        position: Dict,
        candle: pd.Series,
        reason: str
    ) -> Dict:
        """Close an existing position"""
        exit_price = candle['close']

        # Apply slippage
        if position['side'] == 'long':
            exit_price *= (1 - self.config.slippage_pct)
        else:
            exit_price *= (1 + self.config.slippage_pct)

        # Calculate P&L
        pnl = self._calculate_pnl(
            position['side'],
            position['entry_price'],
            exit_price,
            position['size']
        )

        # Apply fees
        entry_fee = position['entry_price'] * position['size'] * self.config.fee_pct
        exit_fee = exit_price * position['size'] * self.config.fee_pct
        pnl -= (entry_fee + exit_fee)

        trade = {
            **position,
            'exit_price': exit_price,
            'exit_time': candle.name,
            'exit_reason': reason,
            'pnl': pnl,
            'return_pct': pnl / (position['entry_price'] * position['size'])
        }

        return trade

    def _check_exit_conditions(
        self,
        position: Dict,
        candle: pd.Series,
        current_index: int
    ) -> Tuple[bool, str]:
        """Check if position should be exited"""
        current_price = candle['close']

        # Check stop loss
        if position['side'] == 'long':
            if current_price <= position['stop_loss']:
                return True, 'stop_loss'
        else:
            if current_price >= position['stop_loss']:
                return True, 'stop_loss'

        # Check take profit
        if position['side'] == 'long':
            if current_price >= position['take_profit']:
                return True, 'take_profit'
        else:
            if current_price <= position['take_profit']:
                return True, 'take_profit'

        # Check max hold time
        hold_time = current_index - position['entry_index']
        if hold_time >= self.config.max_hold_candles:
            return True, 'max_hold'

        return False, ''

    def _calculate_stop_loss(self, side: str, entry_price: float) -> float:
        """Calculate stop loss price"""
        if side == 'long':
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)

    def _calculate_take_profit(self, side: str, entry_price: float) -> float:
        """Calculate take profit price"""
        if side == 'long':
            return entry_price * (1 + self.config.take_profit_pct)
        else:
            return entry_price * (1 - self.config.take_profit_pct)

    def _calculate_pnl(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float
    ) -> float:
        """Calculate P&L for a trade"""
        if side == 'long':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        return pnl


if __name__ == '__main__':
    # Test backtester (minimal example)
    print("=" * 60)
    print("Testing Backtester")
    print("=" * 60)

    from ..models.predictor import CandlePredictor

    # Create dummy data
    dates = pd.date_range('2025-01-01', periods=500, freq='5T')
    price = 100 + np.cumsum(np.random.randn(500) * 0.1)

    data = pd.DataFrame({
        'open': price,
        'high': price * 1.01,
        'low': price * 0.99,
        'close': price,
        'volume': np.random.rand(500) * 1000
    }, index=dates)

    # Add dummy features
    for i in range(10):
        data[f'feature_{i}'] = np.random.randn(500)

    # Create dummy model
    model = CandlePredictor(
        input_dim=15,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        ff_dim=512,
        dropout=0.1
    )

    # Initialize backtester
    config = BacktestConfig(
        initial_capital=10000,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.06
    )

    backtester = Backtester(model, config, device='cpu')

    print("\nBacktester initialized successfully!")
    print(f"Config: {config}")

    print("\n" + "=" * 60)
    print("Backtester test setup complete! ✅")
    print("=" * 60)
    print("(Run full backtest with real data to test execution)")
