"""
Baseline Test: LONG-Only Strategy
===================================

Test LONG-only strategy with the SAME framework as 4 ideas
This ensures fair comparison (apples-to-apples)
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

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BASELINE TEST: LONG-ONLY STRATEGY")
print("="*80)
print("\nTesting LONG-only with same framework as 4 ideas\n")

# Load LONG model
print("Loading LONG model...")
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ LONG model loaded\n")

# Load data - SAMPLE for speed
print("Loading data (sample for speed)...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

# Sample: Last 15,000 candles (~52 days) - enough for testing
df = df.tail(15000).reset_index(drop=True)
print(f"  ✅ Data loaded: {len(df)} candles (~{len(df)//288:.0f} days)\n")

print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().bfill().fillna(0)
print(f"  ✅ Features calculated\n")


def backtest_long_only(df, window_size=1440, step_size=288, threshold=0.65):
    """
    Backtest LONG-only strategy
    Same framework as 4 ideas for fair comparison
    """

    results = []
    num_windows = (len(df) - window_size) // step_size

    print(f"Backtesting LONG-Only:")
    print(f"  Threshold: {threshold}")
    print(f"  Testing {num_windows} windows...\n")

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Backtest this window
        position = None
        trades = []

        for i in range(len(window_df) - 1):
            current_candle = window_df.iloc[i]

            # Get LONG signal
            try:
                long_feat = window_df[long_feature_columns].iloc[i:i+1].values
                long_feat_scaled = long_scaler.transform(long_feat)
                long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
            except:
                long_prob = 0.0

            # Entry logic (LONG ONLY)
            if position is None and i < len(window_df) - 1:
                if long_prob >= threshold:
                    position = {
                        'side': 'LONG',
                        'entry_idx': i,
                        'entry_price': current_candle['close'],
                        'probability': long_prob
                    }

            # Exit logic
            if position is not None:
                time_in_position = i - position['entry_idx']
                pnl_pct = (current_candle['close'] - position['entry_price']) / position['entry_price']

                should_exit = False
                exit_reason = None

                # Exit conditions
                if time_in_position >= 240:  # 4 hours
                    should_exit = True
                    exit_reason = 'max_hold'
                elif pnl_pct >= 0.03:  # 3% TP
                    should_exit = True
                    exit_reason = 'take_profit'
                elif pnl_pct <= -0.015:  # 1.5% SL
                    should_exit = True
                    exit_reason = 'stop_loss'

                if should_exit:
                    trades.append({
                        'side': 'LONG',
                        'entry_price': position['entry_price'],
                        'exit_price': current_candle['close'],
                        'pnl_pct': pnl_pct,
                        'hold_time': time_in_position,
                        'exit_reason': exit_reason,
                    })
                    position = None

        # Calculate window metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)

            results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(trades),
                'short_trades': 0,
                'avg_return': trades_df['pnl_pct'].mean() * 100,
                'total_return': trades_df['pnl_pct'].sum() * 100,
                'win_rate': (trades_df['pnl_pct'] > 0).sum() / len(trades) * 100,
                'long_wr': (trades_df['pnl_pct'] > 0).sum() / len(trades) * 100,
                'short_wr': 0,
            })
        else:
            results.append({
                'window': window_idx,
                'total_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'avg_return': 0,
                'total_return': 0,
                'win_rate': 0,
                'long_wr': 0,
                'short_wr': 0,
            })

    return pd.DataFrame(results)


# Test different thresholds
thresholds = [0.60, 0.65, 0.70]

all_results = []

for threshold in thresholds:
    print(f"\n{'='*80}")
    print(f"Testing: Threshold {threshold}")
    print(f"{'='*80}\n")

    results_df = backtest_long_only(df, threshold=threshold)

    # Summary statistics
    avg_return = results_df['total_return'].mean()
    avg_trades = results_df['total_trades'].mean()
    avg_wr = results_df[results_df['win_rate'] > 0]['win_rate'].mean()

    print(f"\nRESULTS (Threshold {threshold}):")
    print(f"  Avg Return per Window: {avg_return:.2f}%")
    print(f"  Avg Trades per Window: {avg_trades:.1f}")
    print(f"  Overall Win Rate: {avg_wr:.1f}%")

    all_results.append({
        'config': f'LONG-only (threshold {threshold})',
        'threshold': threshold,
        'avg_return': avg_return,
        'avg_trades': avg_trades,
        'avg_long': avg_trades,
        'avg_short': 0,
        'win_rate': avg_wr
    })

# Final comparison
print(f"\n{'='*80}")
print("BASELINE (LONG-ONLY) RESULTS")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values('avg_return', ascending=False)

print(comparison_df.to_string(index=False))

best = comparison_df.iloc[0]
print(f"\n🏆 BEST LONG-ONLY CONFIGURATION:")
print(f"  Threshold: {best['threshold']}")
print(f"  Avg Return: {best['avg_return']:.2f}% per window")
print(f"  Avg Trades: {best['avg_trades']:.1f}")
print(f"  Win Rate: {best['win_rate']:.1f}%")

print(f"\n📊 THIS IS THE FAIR BASELINE FOR COMPARISON")
print(f"  All 4 ideas must beat: {best['avg_return']:.2f}% per window")

# Save results
output_file = RESULTS_DIR / "baseline_long_only_results.csv"
comparison_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print(f"\n{'='*80}")
print("BASELINE TEST COMPLETE")
print(f"{'='*80}\n")
