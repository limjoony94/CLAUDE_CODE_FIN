"""
Trade-Outcome Exit Labeling
============================

Exit labeling based on actual trade outcomes (same philosophy as entry models).

Key Concept:
- At each timepoint during a position, ask: "Should I exit NOW or HOLD?"
- Label = 1 (EXIT) if exiting now is better than holding to final exit
- Label = 0 (HOLD) if holding leads to better outcome

Benefits vs Peak/Trough:
- Directly answers "when to exit" question
- Uses actual P&L outcomes (not price patterns)
- Consistent with Trade-Outcome Entry labeling philosophy
- Better handles sideways markets and false peaks

Author: Claude Code
Date: 2025-10-19
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ExitOutcome:
    """Container for exit decision outcomes"""
    current_pnl: float  # P&L if exit now
    final_pnl: float    # P&L at actual final exit
    should_exit: bool   # True if exiting now is better
    reason: str         # Why this is a good/bad exit point


class TradeOutcomeExitLabeling:
    """
    Trade-Outcome based Exit labeling

    For each position timepoint:
    1. Calculate P&L if we exit NOW
    2. Compare to P&L at final exit
    3. Label EXIT (1) if NOW >= FINAL (with tolerance)
    4. Label HOLD (0) if holding is better

    Special cases:
    - Near peak (within tolerance of best): EXIT
    - Before peak: HOLD (let it run)
    - After peak started falling: EXIT (protect profits)
    - Deep losses: EXIT if it gets worse (stop loss)
    - Small losses that recover: HOLD
    """

    def __init__(
        self,
        exit_tolerance: float = 0.005,  # 0.5% tolerance (4x leverage = 2% unleveraged)
        min_hold_candles: int = 3,      # Hold at least 3 candles (15 min)
        stop_loss_threshold: float = -0.03,  # -3% on 4x leverage
        take_profit_threshold: float = 0.03   # +3% on 4x leverage
    ):
        """
        Args:
            exit_tolerance: How close to peak to consider "good exit"
            min_hold_candles: Minimum hold time before considering exit
            stop_loss_threshold: Auto-exit threshold for losses
            take_profit_threshold: Auto-exit threshold for profits
        """
        self.exit_tolerance = exit_tolerance
        self.min_hold_candles = min_hold_candles
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold


    def create_long_exit_labels(
        self,
        df: pd.DataFrame,
        trades: List[Dict],
        leverage: float = 4.0
    ) -> Tuple[np.ndarray, List[ExitOutcome]]:
        """
        Create EXIT labels for LONG positions based on trade outcomes

        For each trade:
        1. Find entry point in df
        2. Simulate holding from entry to various exit points
        3. Label points where exiting is better than final outcome

        Args:
            df: DataFrame with OHLCV data
            trades: List of simulated trades with entry/exit info
            leverage: Position leverage (default 4x)

        Returns:
            labels: Binary array (1=EXIT, 0=HOLD)
            outcomes: List of ExitOutcome objects for analysis
        """
        labels = np.zeros(len(df), dtype=int)
        outcomes = []

        print(f"\n📊 Creating Trade-Outcome EXIT labels for LONG positions...")
        print(f"   Tolerance: {self.exit_tolerance*100:.2f}% | Min Hold: {self.min_hold_candles} candles")

        for trade in trades:
            if trade.get('side') != 'LONG':
                continue

            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            entry_price = trade['entry_price']
            final_exit_price = trade['exit_price']

            # Calculate final P&L (leveraged)
            final_pnl_pct = ((final_exit_price - entry_price) / entry_price) * leverage

            # For each timepoint in the position
            for idx in range(entry_idx, exit_idx + 1):
                # Skip minimum hold period
                candles_held = idx - entry_idx
                if candles_held < self.min_hold_candles:
                    continue

                current_price = df.iloc[idx]['close']
                current_pnl_pct = ((current_price - entry_price) / entry_price) * leverage

                # Decision logic
                should_exit = False
                reason = "HOLD (default)"

                # 1. Take Profit hit
                if current_pnl_pct >= self.take_profit_threshold:
                    should_exit = True
                    reason = f"TAKE_PROFIT ({current_pnl_pct*100:+.2f}%)"

                # 2. Stop Loss hit
                elif current_pnl_pct <= self.stop_loss_threshold:
                    should_exit = True
                    reason = f"STOP_LOSS ({current_pnl_pct*100:+.2f}%)"

                # 3. Near peak (within tolerance of best possible outcome)
                elif current_pnl_pct >= final_pnl_pct - self.exit_tolerance:
                    should_exit = True
                    reason = f"NEAR_PEAK (now: {current_pnl_pct*100:+.2f}% vs final: {final_pnl_pct*100:+.2f}%)"

                # 4. After peak started falling (protect profits)
                elif current_pnl_pct > 0 and current_pnl_pct < final_pnl_pct - self.exit_tolerance:
                    # Check if price is declining from here
                    future_window = min(5, exit_idx - idx)
                    if future_window > 0:
                        future_prices = df.iloc[idx:idx+future_window]['close'].values
                        if len(future_prices) > 1 and future_prices[-1] < current_price:
                            should_exit = True
                            reason = f"PROTECT_PROFIT (declining from {current_pnl_pct*100:+.2f}%)"

                # 5. Loss getting worse
                elif current_pnl_pct < 0 and current_pnl_pct < final_pnl_pct:
                    # Exit if loss will deepen
                    should_exit = True
                    reason = f"CUT_LOSS (now: {current_pnl_pct*100:+.2f}% vs final: {final_pnl_pct*100:+.2f}%)"

                # Record outcome
                outcome = ExitOutcome(
                    current_pnl=current_pnl_pct,
                    final_pnl=final_pnl_pct,
                    should_exit=should_exit,
                    reason=reason
                )
                outcomes.append(outcome)

                if should_exit:
                    labels[idx] = 1

        # Statistics
        exit_points = np.sum(labels)
        total_points = len([o for o in outcomes])
        exit_rate = exit_points / total_points if total_points > 0 else 0

        print(f"   ✅ Created {exit_points:,} EXIT labels from {total_points:,} position points ({exit_rate*100:.1f}%)")

        return labels, outcomes


    def create_short_exit_labels(
        self,
        df: pd.DataFrame,
        trades: List[Dict],
        leverage: float = 4.0
    ) -> Tuple[np.ndarray, List[ExitOutcome]]:
        """
        Create EXIT labels for SHORT positions based on trade outcomes

        Same logic as LONG but inverted (profit when price falls)

        Args:
            df: DataFrame with OHLCV data
            trades: List of simulated trades with entry/exit info
            leverage: Position leverage (default 4x)

        Returns:
            labels: Binary array (1=EXIT, 0=HOLD)
            outcomes: List of ExitOutcome objects for analysis
        """
        labels = np.zeros(len(df), dtype=int)
        outcomes = []

        print(f"\n📊 Creating Trade-Outcome EXIT labels for SHORT positions...")
        print(f"   Tolerance: {self.exit_tolerance*100:.2f}% | Min Hold: {self.min_hold_candles} candles")

        for trade in trades:
            if trade.get('side') != 'SHORT':
                continue

            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            entry_price = trade['entry_price']
            final_exit_price = trade['exit_price']

            # Calculate final P&L (leveraged, inverted for SHORT)
            final_pnl_pct = ((entry_price - final_exit_price) / entry_price) * leverage

            # For each timepoint in the position
            for idx in range(entry_idx, exit_idx + 1):
                # Skip minimum hold period
                candles_held = idx - entry_idx
                if candles_held < self.min_hold_candles:
                    continue

                current_price = df.iloc[idx]['close']
                current_pnl_pct = ((entry_price - current_price) / entry_price) * leverage

                # Decision logic (same as LONG)
                should_exit = False
                reason = "HOLD (default)"

                # 1. Take Profit hit
                if current_pnl_pct >= self.take_profit_threshold:
                    should_exit = True
                    reason = f"TAKE_PROFIT ({current_pnl_pct*100:+.2f}%)"

                # 2. Stop Loss hit
                elif current_pnl_pct <= self.stop_loss_threshold:
                    should_exit = True
                    reason = f"STOP_LOSS ({current_pnl_pct*100:+.2f}%)"

                # 3. Near peak (within tolerance of best possible outcome)
                elif current_pnl_pct >= final_pnl_pct - self.exit_tolerance:
                    should_exit = True
                    reason = f"NEAR_PEAK (now: {current_pnl_pct*100:+.2f}% vs final: {final_pnl_pct*100:+.2f}%)"

                # 4. After peak started falling (protect profits)
                elif current_pnl_pct > 0 and current_pnl_pct < final_pnl_pct - self.exit_tolerance:
                    # Check if price is rising from here (bad for SHORT)
                    future_window = min(5, exit_idx - idx)
                    if future_window > 0:
                        future_prices = df.iloc[idx:idx+future_window]['close'].values
                        if len(future_prices) > 1 and future_prices[-1] > current_price:
                            should_exit = True
                            reason = f"PROTECT_PROFIT (price rising from {current_pnl_pct*100:+.2f}%)"

                # 5. Loss getting worse
                elif current_pnl_pct < 0 and current_pnl_pct < final_pnl_pct:
                    # Exit if loss will deepen
                    should_exit = True
                    reason = f"CUT_LOSS (now: {current_pnl_pct*100:+.2f}% vs final: {final_pnl_pct*100:+.2f}%)"

                # Record outcome
                outcome = ExitOutcome(
                    current_pnl=current_pnl_pct,
                    final_pnl=final_pnl_pct,
                    should_exit=should_exit,
                    reason=reason
                )
                outcomes.append(outcome)

                if should_exit:
                    labels[idx] = 1

        # Statistics
        exit_points = np.sum(labels)
        total_points = len([o for o in outcomes])
        exit_rate = exit_points / total_points if total_points > 0 else 0

        print(f"   ✅ Created {exit_points:,} EXIT labels from {total_points:,} position points ({exit_rate*100:.1f}%)")

        return labels, outcomes


    def analyze_outcomes(self, outcomes: List[ExitOutcome]) -> Dict:
        """
        Analyze exit outcomes for quality assessment

        Args:
            outcomes: List of ExitOutcome objects

        Returns:
            Statistics dictionary
        """
        if not outcomes:
            return {}

        exit_outcomes = [o for o in outcomes if o.should_exit]
        hold_outcomes = [o for o in outcomes if not o.should_exit]

        # Exit quality (did we exit near peak?)
        exit_quality = []
        for o in exit_outcomes:
            quality = 1.0 - abs(o.current_pnl - o.final_pnl) / (abs(o.final_pnl) + 1e-6)
            exit_quality.append(quality)

        # Hold quality (did holding improve outcome?)
        hold_quality = []
        for o in hold_outcomes:
            improvement = o.final_pnl - o.current_pnl
            hold_quality.append(improvement)

        stats = {
            'total_points': len(outcomes),
            'exit_points': len(exit_outcomes),
            'hold_points': len(hold_outcomes),
            'exit_rate': len(exit_outcomes) / len(outcomes),
            'avg_exit_quality': np.mean(exit_quality) if exit_quality else 0,
            'avg_hold_improvement': np.mean(hold_quality) if hold_quality else 0,
            'exit_reasons': {}
        }

        # Count exit reasons
        for o in exit_outcomes:
            reason_type = o.reason.split('(')[0].strip()
            stats['exit_reasons'][reason_type] = stats['exit_reasons'].get(reason_type, 0) + 1

        return stats
