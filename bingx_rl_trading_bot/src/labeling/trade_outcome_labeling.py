"""
Trade-Outcome-Based Entry Labeling
===================================

Create entry labels based on actual trade outcomes (not just entry quality).

Philosophy:
- Label entries by simulating full trades (entry → hold → exit)
- Good entry = Profitable trade with good risk/reward
- Aligns training labels with actual trading performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.trade_simulator import TradeSimulator


class TradeOutcomeLabeling:
    """
    Create entry labels based on simulated trade outcomes

    Uses 2-of-3 scoring system with trade-outcome criteria:
    1. Profitability: Trade yields >= target profit
    2. Risk-Reward: Good MAE/MFE ratio
    3. Exit Efficiency: ML Exit works (no emergency exits)
    """

    def __init__(
        self,
        profit_threshold=0.02,  # 2% profit target (leveraged)
        mae_threshold=-0.02,  # Max 2% adverse move
        mfe_threshold=0.04,  # Min 4% favorable move (2:1 RR)
        scoring_threshold=2  # 2 of 3 criteria
    ):
        """
        Args:
            profit_threshold: Minimum leveraged profit for criterion 1
            mae_threshold: Maximum adverse excursion (negative)
            mfe_threshold: Minimum favorable excursion
            scoring_threshold: Number of criteria needed (default: 2 of 3)
        """
        self.profit_threshold = profit_threshold
        self.mae_threshold = mae_threshold
        self.mfe_threshold = mfe_threshold
        self.scoring_threshold = scoring_threshold

    def evaluate_trade_quality(self, trade_result):
        """
        Evaluate trade result against 2-of-3 criteria

        Args:
            trade_result: Dict from TradeSimulator.simulate_trade()

        Returns:
            tuple: (score, criteria_met dict)
        """
        if trade_result is None:
            return 0, {}

        score = 0
        criteria = {}

        # Criterion 1: Profitable Trade
        # Trade achieves target profit (leveraged)
        profitable = trade_result['leveraged_pnl_pct'] >= self.profit_threshold
        if profitable:
            score += 1
        criteria['profitable'] = profitable

        # Criterion 2: Good Risk-Reward Ratio
        # MAE < 2%, MFE > 4% (risk/reward > 2:1)
        good_rr = (
            trade_result['mae'] >= self.mae_threshold and
            trade_result['mfe'] >= self.mfe_threshold
        )
        if good_rr:
            score += 1
        criteria['good_risk_reward'] = good_rr

        # Criterion 3: Efficient Exit
        # ML Exit works (not stopped out or max held)
        efficient_exit = trade_result['exit_reason'] == 'ml_exit'
        if efficient_exit:
            score += 1
        criteria['efficient_exit'] = efficient_exit

        return score, criteria

    def create_entry_labels(self, df, simulator, side, show_progress=True):
        """
        Create entry labels for all candles using trade simulation

        Args:
            df: DataFrame with all features (including exit features)
            simulator: TradeSimulator instance
            side: 'LONG' or 'SHORT'
            show_progress: Show progress updates

        Returns:
            np.array: Binary labels (0 or 1) for each candle
        """
        labels = []
        criteria_stats = {
            'profitable': 0,
            'good_risk_reward': 0,
            'efficient_exit': 0
        }
        score_distribution = {0: 0, 1: 0, 2: 0, 3: 0}

        # Leave buffer at end (can't simulate if too close to end)
        total = len(df) - simulator.emergency_max_hold - 10

        if show_progress:
            print(f"\n{side} Entry Labeling (Trade-Outcome Based)")
            print(f"  Simulating {total:,} potential entries...")

        for i in range(total):
            if show_progress and i > 0 and i % 5000 == 0:
                print(f"  Progress: {i:,}/{total:,} ({i/total*100:.1f}%)")

            # Simulate trade from this entry
            trade_result = simulator.simulate_trade(df, i, side)

            # Evaluate trade quality
            score, criteria = self.evaluate_trade_quality(trade_result)

            # Update statistics
            for criterion, met in criteria.items():
                if met:
                    criteria_stats[criterion] += 1

            score_distribution[score] += 1

            # Label = 1 if score >= threshold
            label = 1 if score >= self.scoring_threshold else 0
            labels.append(label)

        # Pad remaining candles with 0 (insufficient data to simulate)
        remaining = len(df) - len(labels)
        labels.extend([0] * remaining)

        if show_progress:
            self._print_label_statistics(
                labels, criteria_stats, score_distribution, total, side
            )

        return np.array(labels)

    def _print_label_statistics(self, labels, criteria_stats, score_distribution, total, side):
        """Print labeling statistics"""
        print(f"\n  {side} Entry Label Statistics:")
        print(f"  Total Candles Evaluated: {total:,}")
        print(f"\n  Criterion Met Rates:")
        print(f"    1. Profitable (>={self.profit_threshold*100:.1f}%):  "
              f"{criteria_stats['profitable']:,} ({criteria_stats['profitable']/total*100:.1f}%)")
        print(f"    2. Good Risk-Reward (MAE<{abs(self.mae_threshold)*100:.1f}%, MFE>{self.mfe_threshold*100:.1f}%): "
              f"{criteria_stats['good_risk_reward']:,} ({criteria_stats['good_risk_reward']/total*100:.1f}%)")
        print(f"    3. Efficient Exit (ML Exit):     "
              f"{criteria_stats['efficient_exit']:,} ({criteria_stats['efficient_exit']/total*100:.1f}%)")

        print(f"\n  Score Distribution:")
        for score in range(4):
            count = score_distribution[score]
            pct = count / total * 100
            marker = " (labeled positive)" if score >= self.scoring_threshold else ""
            print(f"    Score {score} ({score}/3): {count:,} ({pct:.1f}%){marker}")

        positive_labels = sum(labels)
        positive_rate = positive_labels / len(labels) * 100
        print(f"\n  Positive Label Rate: {positive_labels:,}/{len(labels):,} ({positive_rate:.1f}%)")
        print(f"  ✅ {side} Entry labels created\n")


if __name__ == "__main__":
    """Test trade-outcome labeling"""
    from scripts.experiments.calculate_all_features import calculate_all_features
    from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
    from scripts.experiments.trade_simulator import load_exit_models

    print("="*80)
    print("Testing Trade-Outcome-Based Entry Labeling")
    print("="*80)

    # Load data
    print("\nLoading data...")
    DATA_DIR = PROJECT_ROOT / "data" / "historical"
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"  ✅ Loaded {len(df):,} candles")

    # Calculate features
    print("\nCalculating features...")
    df = calculate_all_features(df)
    df = prepare_exit_features(df)
    print(f"  ✅ Features ready")

    # Load exit models
    exit_models = load_exit_models()

    # Create simulators
    long_simulator = TradeSimulator(
        exit_model=exit_models['long'][0],
        exit_scaler=exit_models['long'][1],
        exit_features=exit_models['long'][2]
    )

    short_simulator = TradeSimulator(
        exit_model=exit_models['short'][0],
        exit_scaler=exit_models['short'][1],
        exit_features=exit_models['short'][2]
    )

    # Create labeler
    labeler = TradeOutcomeLabeling(
        profit_threshold=0.02,  # 2% leveraged profit
        mae_threshold=-0.02,  # Max 2% adverse
        mfe_threshold=0.04,  # Min 4% favorable
        scoring_threshold=2  # 2 of 3
    )

    # Test on subset (first 10,000 candles)
    print("\nTesting on first 10,000 candles...")
    df_test = df.head(10000).copy()

    # Create LONG labels
    long_labels = labeler.create_entry_labels(df_test, long_simulator, 'LONG')

    # Create SHORT labels
    short_labels = labeler.create_entry_labels(df_test, short_simulator, 'SHORT')

    print("\n" + "="*80)
    print("Trade-Outcome Labeling Test Complete")
    print("="*80)
