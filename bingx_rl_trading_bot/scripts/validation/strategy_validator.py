"""
Strategy Validator - Standard Research Protocol Implementation

This module implements the Two-Tier validation framework:
- Type 1: Signal Quality Verification
- Type 2: Actual Trading Verification
- Walk-Forward Analysis
- Monte Carlo Simulation

Usage:
    from scripts.validation.strategy_validator import StrategyValidator

    validator = StrategyValidator(
        signal_func=my_signal_function,
        tp_pct=2.5,
        sl_pct=2.0,
        leverage=3,
        position_pct=0.95,
        fee_pct=0.05
    )
    report = validator.validate(df)
    print(report)
"""

import random
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationThresholds:
    """Validation pass/fail thresholds"""
    # Type 1: Signal Quality
    min_signals: int = 100
    min_win_rate: float = 50.0
    min_expected_value: float = 0.0
    max_bars: int = 500  # Max bars to wait for exit in Type 1 validation

    # Type 2: Actual Trading
    min_total_pnl: float = 0.0
    max_drawdown: float = 50.0

    # Walk-Forward
    n_folds: int = 8
    min_pass_rate: float = 0.50

    # Monte Carlo
    n_simulations: int = 1000
    min_profitable_rate: float = 0.80


@dataclass
class Type1Result:
    """Type 1 validation result"""
    total_signals: int
    win_rate: float
    expected_value: float
    long_signals: int
    long_win_rate: float
    long_expected_value: float
    short_signals: int
    short_win_rate: float
    short_expected_value: float
    passed: bool


@dataclass
class Type2Result:
    """Type 2 validation result"""
    total_pnl_pct: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    final_balance: float
    passed: bool
    trades: List[Dict]


@dataclass
class WalkForwardResult:
    """Walk-Forward validation result"""
    pass_rate: float
    passes: int
    total_folds: int
    folds: List[Dict]
    passed: bool


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    profitable_rate: float
    mean_final_balance: float
    std_final_balance: float
    percentile_5: float
    percentile_95: float
    passed: bool


@dataclass
class ValidationReport:
    """Complete validation report"""
    strategy_name: str
    timestamp: str
    data_period: str
    total_candles: int

    type1: Type1Result
    type2: Type2Result
    walk_forward: WalkForwardResult
    monte_carlo: MonteCarloResult

    overall_passed: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'timestamp': self.timestamp,
            'data_period': self.data_period,
            'total_candles': self.total_candles,
            'type1': {
                'total_signals': self.type1.total_signals,
                'win_rate': self.type1.win_rate,
                'expected_value': self.type1.expected_value,
                'long_signals': self.type1.long_signals,
                'long_win_rate': self.type1.long_win_rate,
                'short_signals': self.type1.short_signals,
                'short_win_rate': self.type1.short_win_rate,
                'passed': self.type1.passed,
            },
            'type2': {
                'total_pnl_pct': self.type2.total_pnl_pct,
                'num_trades': self.type2.num_trades,
                'win_rate': self.type2.win_rate,
                'max_drawdown': self.type2.max_drawdown,
                'passed': self.type2.passed,
            },
            'walk_forward': {
                'pass_rate': self.walk_forward.pass_rate,
                'passes': self.walk_forward.passes,
                'total_folds': self.walk_forward.total_folds,
                'passed': self.walk_forward.passed,
            },
            'monte_carlo': {
                'profitable_rate': self.monte_carlo.profitable_rate,
                'mean_final_balance': self.monte_carlo.mean_final_balance,
                'passed': self.monte_carlo.passed,
            },
            'overall_passed': self.overall_passed,
        }

    def __str__(self) -> str:
        """Generate formatted report string"""
        lines = [
            "=" * 60,
            f"STRATEGY VALIDATION REPORT: {self.strategy_name}",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Data Period: {self.data_period}",
            f"Total Candles: {self.total_candles:,}",
            "",
            "-" * 60,
            "TYPE 1: SIGNAL QUALITY VERIFICATION",
            "-" * 60,
            f"  Total Signals: {self.type1.total_signals}",
            f"  Win Rate: {self.type1.win_rate:.1f}% (min: 50%)",
            f"  Expected Value: {self.type1.expected_value:.2f}% (min: 0%)",
            f"  LONG: {self.type1.long_signals} signals, {self.type1.long_win_rate:.1f}% WR",
            f"  SHORT: {self.type1.short_signals} signals, {self.type1.short_win_rate:.1f}% WR",
            f"  Status: {'✅ PASSED' if self.type1.passed else '❌ FAILED'}",
            "",
            "-" * 60,
            "TYPE 2: ACTUAL TRADING VERIFICATION",
            "-" * 60,
            f"  Total PnL: {self.type2.total_pnl_pct:+.2f}% (min: 0%)",
            f"  Trades: {self.type2.num_trades}",
            f"  Win Rate: {self.type2.win_rate:.1f}%",
            f"  Max Drawdown: {self.type2.max_drawdown:.1f}% (max: 50%)",
            f"  Final Balance: ${self.type2.final_balance:.2f}",
            f"  Status: {'✅ PASSED' if self.type2.passed else '❌ FAILED'}",
            "",
            "-" * 60,
            "WALK-FORWARD ANALYSIS",
            "-" * 60,
            f"  Pass Rate: {self.walk_forward.passes}/{self.walk_forward.total_folds} ({self.walk_forward.pass_rate*100:.0f}%)",
        ]

        for fold in self.walk_forward.folds:
            status = '✅' if fold['pass'] else '❌'
            lines.append(f"    Fold {fold['fold']}: {fold['pnl']:+.2f}% ({fold['trades']} trades) {status}")

        lines.extend([
            f"  Status: {'✅ PASSED' if self.walk_forward.passed else '❌ FAILED'}",
            "",
            "-" * 60,
            "MONTE CARLO SIMULATION",
            "-" * 60,
            f"  Simulations: {1000}",
            f"  Profitable Rate: {self.monte_carlo.profitable_rate:.1f}% (min: 80%)",
            f"  Mean Final Balance: ${self.monte_carlo.mean_final_balance:.2f}",
            f"  5th Percentile: ${self.monte_carlo.percentile_5:.2f}",
            f"  95th Percentile: ${self.monte_carlo.percentile_95:.2f}",
            f"  Status: {'✅ PASSED' if self.monte_carlo.passed else '❌ FAILED'}",
            "",
            "=" * 60,
            f"OVERALL: {'✅ APPROVED FOR PRODUCTION' if self.overall_passed else '❌ NOT READY'}",
            "=" * 60,
        ])

        return "\n".join(lines)


class StrategyValidator:
    """
    Strategy validation framework implementing the Standard Research Protocol.

    Args:
        signal_func: Function that generates signals. Signature: (df, i) -> 'LONG'/'SHORT'/None
        tp_pct: Take profit percentage
        sl_pct: Stop loss percentage
        leverage: Effective leverage
        position_pct: Position size as fraction of balance (0.0-1.0)
        fee_pct: Single-side fee percentage (doubled for round trip)
        strategy_name: Name for reporting
        thresholds: Custom validation thresholds
    """

    def __init__(
        self,
        signal_func: Callable[[pd.DataFrame, int], Optional[str]],
        tp_pct: float = 2.5,
        sl_pct: float = 2.0,
        leverage: float = 3.0,
        position_pct: float = 0.95,
        fee_pct: float = 0.05,
        strategy_name: str = "Strategy",
        thresholds: Optional[ValidationThresholds] = None,
    ):
        self.signal_func = signal_func
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.leverage = leverage
        self.position_pct = position_pct
        self.fee_pct = fee_pct
        self.strategy_name = strategy_name
        self.thresholds = thresholds or ValidationThresholds()

    def validate(self, df: pd.DataFrame, max_bars: Optional[int] = None) -> ValidationReport:
        """
        Run complete validation suite.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            max_bars: Maximum bars to wait for TP/SL (uses thresholds.max_bars if not provided)

        Returns:
            ValidationReport with all results
        """
        logger.info(f"Starting validation for {self.strategy_name}")

        # Use threshold max_bars if not explicitly provided
        effective_max_bars = max_bars if max_bars is not None else self.thresholds.max_bars

        # Run all validations
        type1 = self._validate_type1(df, effective_max_bars)
        type2 = self._validate_type2(df)
        walk_forward = self._validate_walk_forward(df)
        monte_carlo = self._validate_monte_carlo(type2.trades)

        # Overall pass requires all to pass
        overall_passed = (
            type1.passed and
            type2.passed and
            walk_forward.passed and
            monte_carlo.passed
        )

        # Build report
        data_period = f"{df.index[0]} to {df.index[-1]}" if hasattr(df.index, '__iter__') else "N/A"

        report = ValidationReport(
            strategy_name=self.strategy_name,
            timestamp=datetime.now().isoformat(),
            data_period=data_period,
            total_candles=len(df),
            type1=type1,
            type2=type2,
            walk_forward=walk_forward,
            monte_carlo=monte_carlo,
            overall_passed=overall_passed,
        )

        logger.info(f"Validation complete. Overall: {'PASSED' if overall_passed else 'FAILED'}")
        return report

    def _calculate_tp_sl(self, entry_price: float, direction: str) -> tuple:
        """Calculate TP and SL prices"""
        if direction == 'LONG':
            tp_price = entry_price * (1 + self.tp_pct / 100)
            sl_price = entry_price * (1 - self.sl_pct / 100)
        else:
            tp_price = entry_price * (1 - self.tp_pct / 100)
            sl_price = entry_price * (1 + self.sl_pct / 100)
        return tp_price, sl_price

    def _check_exit(self, bar: pd.Series, direction: str, tp_price: float, sl_price: float) -> tuple:
        """Check if TP or SL is hit"""
        if direction == 'LONG':
            if bar['high'] >= tp_price:
                return tp_price, 'TP'
            elif bar['low'] <= sl_price:
                return sl_price, 'SL'
        else:
            if bar['low'] <= tp_price:
                return tp_price, 'TP'
            elif bar['high'] >= sl_price:
                return sl_price, 'SL'
        return None, None

    def _calculate_pnl(self, entry_price: float, exit_price: float, direction: str) -> float:
        """Calculate PnL percentage including fees"""
        if direction == 'LONG':
            raw_pnl = (exit_price / entry_price - 1) * 100
        else:
            raw_pnl = (1 - exit_price / entry_price) * 100

        # Apply leverage
        leveraged_pnl = raw_pnl * self.leverage

        # Deduct fees (both sides)
        total_fee = self.fee_pct * 2 * self.leverage
        net_pnl = leveraged_pnl - total_fee

        return net_pnl

    def _validate_type1(self, df: pd.DataFrame, max_bars: int) -> Type1Result:
        """Type 1: Signal Quality Verification"""
        results = []

        for i in range(1, len(df) - max_bars):
            signal = self.signal_func(df, i)
            if signal is None:
                continue

            # Entry on next candle open
            entry_idx = i + 1
            entry_price = df.iloc[entry_idx]['open']
            tp_price, sl_price = self._calculate_tp_sl(entry_price, signal)

            # Search for exit (starting from entry_idx + 1)
            for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
                bar = df.iloc[j]
                exit_price, reason = self._check_exit(bar, signal, tp_price, sl_price)

                if exit_price:
                    pnl = self._calculate_pnl(entry_price, exit_price, signal)
                    results.append({
                        'signal': signal,
                        'pnl': pnl,
                        'reason': reason
                    })
                    break

        # Calculate statistics
        total = len(results)
        if total == 0:
            return Type1Result(
                total_signals=0, win_rate=0, expected_value=0,
                long_signals=0, long_win_rate=0, long_expected_value=0,
                short_signals=0, short_win_rate=0, short_expected_value=0,
                passed=False
            )

        wins = sum(1 for r in results if r['pnl'] > 0)
        win_rate = wins / total * 100
        expected_value = sum(r['pnl'] for r in results) / total

        # Long stats
        long_results = [r for r in results if r['signal'] == 'LONG']
        long_signals = len(long_results)
        long_wins = sum(1 for r in long_results if r['pnl'] > 0)
        long_win_rate = long_wins / long_signals * 100 if long_signals > 0 else 0
        long_ev = sum(r['pnl'] for r in long_results) / long_signals if long_signals > 0 else 0

        # Short stats
        short_results = [r for r in results if r['signal'] == 'SHORT']
        short_signals = len(short_results)
        short_wins = sum(1 for r in short_results if r['pnl'] > 0)
        short_win_rate = short_wins / short_signals * 100 if short_signals > 0 else 0
        short_ev = sum(r['pnl'] for r in short_results) / short_signals if short_signals > 0 else 0

        # Check pass conditions
        passed = (
            total >= self.thresholds.min_signals and
            win_rate >= self.thresholds.min_win_rate and
            expected_value > self.thresholds.min_expected_value
        )

        return Type1Result(
            total_signals=total,
            win_rate=win_rate,
            expected_value=expected_value,
            long_signals=long_signals,
            long_win_rate=long_win_rate,
            long_expected_value=long_ev,
            short_signals=short_signals,
            short_win_rate=short_win_rate,
            short_expected_value=short_ev,
            passed=passed
        )

    def _validate_type2(self, df: pd.DataFrame) -> Type2Result:
        """Type 2: Actual Trading Verification"""
        balance = 100.0
        position = None
        trades = []
        peak_balance = balance

        for i in range(1, len(df)):
            bar = df.iloc[i]

            # 1. Check for position exit first
            if position is not None:
                exit_price, reason = self._check_exit(
                    bar, position['direction'],
                    position['tp_price'], position['sl_price']
                )

                if exit_price:
                    pnl_pct = self._calculate_pnl(
                        position['entry_price'], exit_price,
                        position['direction']
                    )
                    pnl_dollar = balance * self.position_pct * (pnl_pct / 100)
                    balance += pnl_dollar

                    trades.append({
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollar': pnl_dollar,
                        'reason': reason,
                        'balance': balance
                    })

                    peak_balance = max(peak_balance, balance)
                    position = None

            # 2. Check for new entry (only if no position)
            if position is None:
                signal = self.signal_func(df, i - 1)

                if signal:
                    entry_price = bar['open']
                    tp_price, sl_price = self._calculate_tp_sl(entry_price, signal)

                    position = {
                        'direction': signal,
                        'entry_price': entry_price,
                        'tp_price': tp_price,
                        'sl_price': sl_price
                    }

        # Calculate max drawdown
        max_dd = 0
        peak = 100
        for trade in trades:
            peak = max(peak, trade['balance'])
            dd = (peak - trade['balance']) / peak * 100
            max_dd = max(max_dd, dd)

        # Calculate statistics
        total_pnl = balance - 100
        num_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl_pct'] > 0)
        win_rate = wins / num_trades * 100 if num_trades > 0 else 0

        # Check pass conditions
        passed = (
            total_pnl > self.thresholds.min_total_pnl and
            max_dd < self.thresholds.max_drawdown
        )

        return Type2Result(
            total_pnl_pct=total_pnl,
            num_trades=num_trades,
            win_rate=win_rate,
            max_drawdown=max_dd,
            final_balance=balance,
            passed=passed,
            trades=trades
        )

    def _validate_walk_forward(self, df: pd.DataFrame) -> WalkForwardResult:
        """Walk-Forward validation with N folds"""
        n_folds = self.thresholds.n_folds
        fold_size = len(df) // n_folds
        fold_results = []

        for fold_idx in range(n_folds):
            start_idx = fold_idx * fold_size
            end_idx = start_idx + fold_size if fold_idx < n_folds - 1 else len(df)
            fold_df = df.iloc[start_idx:end_idx].copy()

            result = self._validate_type2(fold_df)

            fold_results.append({
                'fold': fold_idx + 1,
                'pnl': result.total_pnl_pct,
                'trades': result.num_trades,
                'pass': result.total_pnl_pct > 0
            })

        passes = sum(1 for f in fold_results if f['pass'])
        pass_rate = passes / n_folds

        passed = pass_rate >= self.thresholds.min_pass_rate

        return WalkForwardResult(
            pass_rate=pass_rate,
            passes=passes,
            total_folds=n_folds,
            folds=fold_results,
            passed=passed
        )

    def _validate_monte_carlo(self, trades: List[Dict]) -> MonteCarloResult:
        """Monte Carlo simulation with trade shuffling"""
        if len(trades) < 5:
            return MonteCarloResult(
                profitable_rate=0,
                mean_final_balance=100,
                std_final_balance=0,
                percentile_5=100,
                percentile_95=100,
                passed=False
            )

        n_sims = self.thresholds.n_simulations
        final_balances = []

        for _ in range(n_sims):
            shuffled = random.sample(trades, len(trades))
            balance = 100.0

            for trade in shuffled:
                pnl_dollar = balance * self.position_pct * (trade['pnl_pct'] / 100)
                balance += pnl_dollar

            final_balances.append(balance)

        profitable = sum(1 for b in final_balances if b > 100)
        profitable_rate = profitable / n_sims * 100

        passed = profitable_rate >= self.thresholds.min_profitable_rate * 100

        return MonteCarloResult(
            profitable_rate=profitable_rate,
            mean_final_balance=np.mean(final_balances),
            std_final_balance=np.std(final_balances),
            percentile_5=np.percentile(final_balances, 5),
            percentile_95=np.percentile(final_balances, 95),
            passed=passed
        )


def validate_look_ahead_bias(signal_func_source: str) -> Dict[str, Any]:
    """
    Check for look-ahead bias patterns in signal function source code.

    Args:
        signal_func_source: Source code of the signal function

    Returns:
        Dictionary with check results
    """
    forbidden_patterns = [
        ('shift(-', 'Future reference with shift(-N)'),
        ('shift( -', 'Future reference with shift( -N)'),
        ('center=True', 'Centered rolling window'),
        ('center = True', 'Centered rolling window'),
    ]

    issues = []
    for pattern, description in forbidden_patterns:
        if pattern in signal_func_source:
            issues.append({
                'pattern': pattern,
                'description': description,
                'severity': 'CRITICAL'
            })

    return {
        'clean': len(issues) == 0,
        'issues': issues
    }
