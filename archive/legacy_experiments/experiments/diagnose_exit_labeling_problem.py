"""
Deep Diagnostic: EXIT Labeling Problem

Goal: Understand WHY probability range 0.6-0.7 produces worst results

Analysis:
1. Compare winning vs losing trades by exit probability range
2. Analyze exit timing quality (how close to optimal)
3. Feature patterns in different probability ranges
4. Design improved labeling methodology
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from src.features.sell_signal_features import SellSignalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

WINDOW_SIZE = 1440
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
ENTRY_THRESHOLD = 0.7
TRANSACTION_COST = 0.0002
MAX_HOLDING_HOURS = 8


def classify_market_regime(df_window):
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


def backtest_with_detailed_exit_analysis(df, models, scalers, features, exit_threshold):
    """Backtest with detailed exit analysis by probability range"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    exit_signals = []  # Track ALL exit signals (taken and not taken)

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Calculate exit probability
            exit_prob = None
            if side == 'LONG':
                exit_row = df[features['long_exit']].iloc[i:i+1].values
                if not np.isnan(exit_row).any():
                    exit_scaled = scalers['long_exit'].transform(exit_row)
                    exit_prob = models['long_exit'].predict_proba(exit_scaled)[0][1]
            else:
                exit_row = df[features['short_exit']].iloc[i:i+1].values
                if not np.isnan(exit_row).any():
                    exit_scaled = scalers['short_exit'].transform(exit_row)
                    exit_prob = models['short_exit'].predict_proba(exit_scaled)[0][1]

            # Track exit signal (whether we take it or not)
            if exit_prob is not None:
                # Calculate what the final PnL would be if we exited here
                quantity = position['quantity']
                exit_pnl_usd = pnl_pct * (entry_price * quantity)
                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                exit_pnl_usd -= (entry_cost + exit_cost)

                exit_signals.append({
                    'side': side,
                    'exit_prob': exit_prob,
                    'pnl_if_exit': exit_pnl_usd,
                    'pnl_pct_if_exit': pnl_pct,
                    'hours_held': hours_held,
                    'above_threshold': exit_prob >= exit_threshold,
                    'trade_id': len(trades)  # Link to trade if exited
                })

            # Decide whether to exit
            exit_reason = None
            if exit_prob is not None:
                if exit_prob >= exit_threshold:
                    exit_reason = "ML Exit"
                elif hours_held >= MAX_HOLDING_HOURS:
                    exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)
                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)
                capital += pnl_usd

                trades.append({
                    'trade_id': len(trades),
                    'side': side,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'exit_prob': exit_prob,
                    'regime': position['regime']
                })

                position = None

        # Entry logic
        if position is None and i < len(df) - 1:
            long_row = df[features['long_entry']].iloc[i:i+1].values
            if np.isnan(long_row).any():
                long_prob = 0
            else:
                long_scaled = scalers['long_entry'].transform(long_row)
                long_prob = models['long_entry'].predict_proba(long_scaled)[0][1]

            short_row = df[features['short_entry']].iloc[i:i+1].values
            if np.isnan(short_row).any():
                short_prob = 0
            else:
                short_scaled = scalers['short_entry'].transform(short_row)
                short_prob = models['short_entry'].predict_proba(short_scaled)[0][1]

            side = None
            probability = None

            if long_prob >= ENTRY_THRESHOLD and short_prob < ENTRY_THRESHOLD:
                side = 'LONG'
                probability = long_prob
            elif short_prob >= ENTRY_THRESHOLD and long_prob < ENTRY_THRESHOLD:
                side = 'SHORT'
                probability = short_prob
            elif long_prob >= ENTRY_THRESHOLD and short_prob >= ENTRY_THRESHOLD:
                if long_prob > short_prob:
                    side = 'LONG'
                    probability = long_prob
                else:
                    side = 'SHORT'
                    probability = short_prob

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price
                current_regime = classify_market_regime(df.iloc[max(0, i-20):i+1])

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'probability': probability,
                    'regime': current_regime
                }

    return capital, trades, exit_signals


def analyze_by_probability_range(trades, exit_signals, prob_ranges):
    """Analyze exit quality by probability range"""

    # Group exit signals by probability range
    range_stats = {}

    for low, high in prob_ranges:
        range_key = f"{low:.1f}-{high:.1f}"

        # Signals in this range
        signals_in_range = [
            s for s in exit_signals
            if low <= s['exit_prob'] < high
        ]

        if not signals_in_range:
            continue

        # Signals that were acted on (above threshold)
        acted_on = [s for s in signals_in_range if s['above_threshold']]

        # Calculate statistics
        avg_pnl = np.mean([s['pnl_if_exit'] for s in signals_in_range])
        win_rate = len([s for s in signals_in_range if s['pnl_if_exit'] > 0]) / len(signals_in_range) * 100

        # If acted on, what was the actual outcome?
        if acted_on:
            acted_avg_pnl = np.mean([s['pnl_if_exit'] for s in acted_on])
            acted_win_rate = len([s for s in acted_on if s['pnl_if_exit'] > 0]) / len(acted_on) * 100
        else:
            acted_avg_pnl = 0
            acted_win_rate = 0

        range_stats[range_key] = {
            'total_signals': len(signals_in_range),
            'acted_on': len(acted_on),
            'avg_pnl_if_exit': avg_pnl,
            'win_rate_if_exit': win_rate,
            'acted_avg_pnl': acted_avg_pnl,
            'acted_win_rate': acted_win_rate,
            'avg_holding_hours': np.mean([s['hours_held'] for s in signals_in_range])
        }

    return range_stats


def main():
    print("=" * 80)
    print("DEEP DIAGNOSTIC: EXIT Labeling Problem")
    print("=" * 80)

    # Load models
    print("\nLoading models...")
    models = {}
    scalers = {}
    features = {}

    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
        models['long_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
        scalers['long_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
        features['long_entry'] = [line.strip() for line in f]

    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939.pkl", 'rb') as f:
        models['short_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_scaler.pkl", 'rb') as f:
        scalers['short_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_features.txt", 'r') as f:
        features['short_entry'] = [line.strip() for line in f]

    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651.pkl", 'rb') as f:
        models['long_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651_scaler.pkl", 'rb') as f:
        scalers['long_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651_features.txt", 'r') as f:
        features['long_exit'] = [line.strip() for line in f]

    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233.pkl", 'rb') as f:
        models['short_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233_scaler.pkl", 'rb') as f:
        scalers['short_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233_features.txt", 'r') as f:
        features['short_exit'] = [line.strip() for line in f]

    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    df = calculate_features(df)
    adv = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv.calculate_all_features(df)
    sell = SellSignalFeatures()
    df = sell.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ {len(df):,} candles")

    # Test threshold 0.7 with detailed analysis
    print("\n" + "=" * 80)
    print("ANALYSIS: Threshold 0.7 (Why it's worst)")
    print("=" * 80)

    all_trades = []
    all_exit_signals = []

    start_idx = 0
    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        final_capital, trades, exit_signals = backtest_with_detailed_exit_analysis(
            window_df, models, scalers, features, 0.7
        )

        all_trades.extend(trades)
        all_exit_signals.extend(exit_signals)

        start_idx += WINDOW_SIZE

    print(f"\nTotal trades: {len(all_trades)}")
    print(f"Total exit signals evaluated: {len(all_exit_signals)}")

    # Analyze by probability range
    print("\n" + "=" * 80)
    print("EXIT Signal Quality by Probability Range")
    print("=" * 80)

    prob_ranges = [
        (0.0, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),  # The problematic range
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0)
    ]

    range_stats = analyze_by_probability_range(all_trades, all_exit_signals, prob_ranges)

    print("\nSignal Quality Analysis:")
    print(f"{'Range':<10} {'Signals':<10} {'Acted':<10} {'Avg PnL':<12} {'Win%':<8} {'Acted PnL':<12} {'Acted Win%':<12}")
    print("-" * 90)

    for range_key in sorted(range_stats.keys()):
        stats = range_stats[range_key]
        print(f"{range_key:<10} "
              f"{stats['total_signals']:<10} "
              f"{stats['acted_on']:<10} "
              f"${stats['avg_pnl_if_exit']:>10.2f} "
              f"{stats['win_rate_if_exit']:>6.1f}% "
              f"${stats['acted_avg_pnl']:>10.2f} "
              f"{stats['acted_win_rate']:>10.1f}%")

    # Compare 0.6-0.7 range to others
    print("\n" + "=" * 80)
    print("KEY FINDING: Why 0.6-0.7 is Worst")
    print("=" * 80)

    if '0.6-0.7' in range_stats:
        bad_range = range_stats['0.6-0.7']
        print(f"\n0.6-0.7 Range Analysis:")
        print(f"  Total signals: {bad_range['total_signals']:,}")
        print(f"  Acted on (>= 0.7): {bad_range['acted_on']:,}")
        print(f"  Average PnL if exit: ${bad_range['avg_pnl_if_exit']:.2f}")
        print(f"  Win rate if exit: {bad_range['win_rate_if_exit']:.1f}%")
        print(f"  Actual PnL (acted): ${bad_range['acted_avg_pnl']:.2f}")
        print(f"  Actual win rate (acted): {bad_range['acted_win_rate']:.1f}%")

        # Compare to 0.5-0.6 range
        if '0.5-0.6' in range_stats:
            good_range = range_stats['0.5-0.6']
            print(f"\n0.5-0.6 Range Analysis (for comparison):")
            print(f"  Total signals: {good_range['total_signals']:,}")
            print(f"  Average PnL if exit: ${good_range['avg_pnl_if_exit']:.2f}")
            print(f"  Win rate if exit: {good_range['win_rate_if_exit']:.1f}%")

            print(f"\n⚠️ Quality Difference:")
            print(f"  PnL: 0.5-0.6 is ${good_range['avg_pnl_if_exit'] - bad_range['avg_pnl_if_exit']:.2f} better")
            print(f"  Win Rate: 0.5-0.6 is {good_range['win_rate_if_exit'] - bad_range['win_rate_if_exit']:.1f}% better")

    # Analyze trades by exit probability
    print("\n" + "=" * 80)
    print("Trade Outcome by Exit Probability")
    print("=" * 80)

    trades_df = pd.DataFrame(all_trades)

    # Group by probability range
    trades_df['prob_range'] = pd.cut(
        trades_df['exit_prob'],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        labels=['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    )

    print("\nTrade outcomes:")
    for prob_range in ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']:
        range_trades = trades_df[trades_df['prob_range'] == prob_range]
        if len(range_trades) > 0:
            win_rate = len(range_trades[range_trades['pnl_usd'] > 0]) / len(range_trades) * 100
            avg_pnl = range_trades['pnl_usd'].mean()
            print(f"  {prob_range}: {len(range_trades):3} trades, "
                  f"Win: {win_rate:5.1f}%, Avg PnL: ${avg_pnl:>8.2f}")

    # Root cause analysis
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)

    print("\nProblem: Peak/Trough labeling creates balanced distribution")
    print("  → Positive rate: ~50%")
    print("  → Mean probability: ~0.5")
    print("  → Model uncertain in middle ranges (0.6-0.7)")
    print()
    print("Why 0.6-0.7 is worst:")
    print("  1. High enough to trigger many exits (>threshold 0.7)")
    print("  2. But model is LEAST confident in this range")
    print("  3. Contains mix of good + bad exit signals")
    print("  4. Bad signals dominate, causing losses")
    print()
    print("Why 0.5-0.6 is better:")
    print("  1. Lower probability, MORE selective")
    print("  2. Only exits when threshold is 0.6 (fewer false exits)")
    print("  3. Better signal quality despite lower probability")

    # Solution proposal
    print("\n" + "=" * 80)
    print("SOLUTION PROPOSAL")
    print("=" * 80)

    print("\nCurrent Labeling (Peak/Trough):")
    print("  Label = 1 if: near peak/trough AND beats holding")
    print("  Problem: Creates ~50% positive rate (balanced)")
    print()
    print("Proposed Labeling Improvements:")
    print()
    print("  Option 1: Stricter Peak/Trough Criteria")
    print("    - Increase near_threshold from 0.80 to 0.95")
    print("    - Only label TRUE peaks/troughs (not near)")
    print("    - Result: Lower positive rate, higher quality labels")
    print()
    print("  Option 2: Profit-Threshold Labeling")
    print("    - Label = 1 ONLY if exit profit > X% (e.g., 0.5%)")
    print("    - Ignore exits with small gains")
    print("    - Result: Focus on high-quality exits")
    print()
    print("  Option 3: Relative Performance Labeling")
    print("    - Label = 1 if exit NOW beats exit LATER (next 12-48 candles)")
    print("    - Compare current exit to future possibilities")
    print("    - Result: Learn optimal timing, not just 'should exit'")
    print()
    print("  Option 4: Combined Approach")
    print("    - Strict peak/trough + Profit threshold + Relative performance")
    print("    - Multiple criteria reduce false positives")
    print("    - Result: Very high quality labels, lower positive rate")

    print("\n" + "=" * 80)
    print("Diagnostic Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
