"""
Idea 2: Asymmetric Time Horizon
=================================

Concept: SHORT holds for 1 hour, LONG holds for 4 hours
- Minimize capital lock from SHORT positions
- Recover LONG opportunities faster
- SHORT becomes ultra-fast scalping strategy
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
print("IDEA 2: ASYMMETRIC TIME HORIZON")
print("="*80)
print("\nConcept: SHORT 1h, LONG 4h - Minimize capital lock\n")

# Load models
print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ Models loaded\n")

# Load and prepare data
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df)} candles\n")

print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().bfill().fillna(0)
print(f"  ✅ Features calculated\n")


def backtest_asymmetric_time(df, window_size=1440, step_size=288,
                             long_threshold=0.65, short_threshold=0.70,
                             long_max_hold=240, short_max_hold=60):
    """
    Backtest with Asymmetric Time Horizon

    Parameters:
    - long_max_hold: LONG max hold time (minutes/candles) - default 4h
    - short_max_hold: SHORT max hold time (minutes/candles) - default 1h
    """

    results = []
    num_windows = (len(df) - window_size) // step_size

    print(f"Backtesting Asymmetric Time Horizon:")
    print(f"  LONG max hold: {long_max_hold} candles ({long_max_hold/60:.1f}h)")
    print(f"  SHORT max hold: {short_max_hold} candles ({short_max_hold/60:.1f}h)")
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

            # Get signals
            try:
                long_feat = window_df[long_feature_columns].iloc[i:i+1].values
                long_feat_scaled = long_scaler.transform(long_feat)
                long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
            except:
                long_prob = 0.0

            try:
                short_feat = window_df[short_feature_columns].iloc[i:i+1].values
                short_feat_scaled = short_scaler.transform(short_feat)
                short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
            except:
                short_prob = 0.0

            # Entry logic (same as before)
            if position is None and i < len(window_df) - 1:
                if long_prob >= long_threshold:
                    position = {
                        'side': 'LONG',
                        'entry_idx': i,
                        'entry_price': current_candle['close'],
                        'probability': long_prob
                    }
                elif short_prob >= short_threshold:
                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': current_candle['close'],
                        'probability': short_prob
                    }

            # Exit logic (ASYMMETRIC TIME!)
            if position is not None:
                time_in_position = i - position['entry_idx']

                if position['side'] == 'LONG':
                    pnl_pct = (current_candle['close'] - position['entry_price']) / position['entry_price']
                else:  # SHORT
                    pnl_pct = (position['entry_price'] - current_candle['close']) / position['entry_price']

                should_exit = False
                exit_reason = None

                # ASYMMETRIC EXIT CONDITIONS
                if position['side'] == 'LONG':
                    # LONG: Normal 4-hour hold
                    if time_in_position >= long_max_hold:
                        should_exit = True
                        exit_reason = 'max_hold'
                    elif pnl_pct >= 0.03:  # 3% TP
                        should_exit = True
                        exit_reason = 'take_profit'
                    elif pnl_pct <= -0.015:  # 1.5% SL
                        should_exit = True
                        exit_reason = 'stop_loss'

                else:  # SHORT
                    # SHORT: Ultra-fast 1-hour hold
                    if time_in_position >= short_max_hold:
                        should_exit = True
                        exit_reason = 'max_hold'
                    elif pnl_pct >= 0.02:  # 2% TP (more aggressive)
                        should_exit = True
                        exit_reason = 'take_profit'
                    elif pnl_pct <= -0.01:  # 1% SL (tighter)
                        should_exit = True
                        exit_reason = 'stop_loss'
                    # Additional: Exit if momentum fades
                    elif time_in_position >= 30 and pnl_pct < 0.005:  # 30min, <0.5%
                        should_exit = True
                        exit_reason = 'momentum_loss'

                if should_exit:
                    trades.append({
                        'side': position['side'],
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

            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']

            results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'avg_return': trades_df['pnl_pct'].mean() * 100,
                'total_return': trades_df['pnl_pct'].sum() * 100,
                'win_rate': (trades_df['pnl_pct'] > 0).sum() / len(trades) * 100,
                'long_wr': (long_trades['pnl_pct'] > 0).sum() / len(long_trades) * 100 if len(long_trades) > 0 else 0,
                'short_wr': (short_trades['pnl_pct'] > 0).sum() / len(short_trades) * 100 if len(short_trades) > 0 else 0,
                'avg_long_hold': long_trades['hold_time'].mean() if len(long_trades) > 0 else 0,
                'avg_short_hold': short_trades['hold_time'].mean() if len(short_trades) > 0 else 0,
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
                'avg_long_hold': 0,
                'avg_short_hold': 0,
            })

    return pd.DataFrame(results)


# Test different SHORT hold times
hold_time_combinations = [
    {'short_max_hold': 60, 'desc': 'SHORT 1h (recommended)'},
    {'short_max_hold': 45, 'desc': 'SHORT 45min (ultra-fast)'},
    {'short_max_hold': 75, 'desc': 'SHORT 75min (moderate)'},
    {'short_max_hold': 240, 'desc': 'SHORT 4h (baseline for comparison)'},
]

all_results = []

for params in hold_time_combinations:
    print(f"\n{'='*80}")
    print(f"Testing: {params['desc']}")
    print(f"{'='*80}\n")

    results_df = backtest_asymmetric_time(
        df,
        long_max_hold=240,  # Always 4h
        short_max_hold=params['short_max_hold']
    )

    # Summary statistics
    avg_return = results_df['total_return'].mean()
    avg_trades = results_df['total_trades'].mean()
    avg_long = results_df['long_trades'].mean()
    avg_short = results_df['short_trades'].mean()
    avg_wr = results_df[results_df['win_rate'] > 0]['win_rate'].mean()
    avg_long_hold = results_df['avg_long_hold'].mean()
    avg_short_hold = results_df['avg_short_hold'].mean()

    print(f"\nRESULTS ({params['desc']}):")
    print(f"  Avg Return per Window: {avg_return:.2f}%")
    print(f"  Avg Trades per Window: {avg_trades:.1f}")
    print(f"    - LONG: {avg_long:.1f} (avg hold: {avg_long_hold:.0f} candles)")
    print(f"    - SHORT: {avg_short:.1f} (avg hold: {avg_short_hold:.0f} candles)")
    print(f"  Overall Win Rate: {avg_wr:.1f}%")

    all_results.append({
        'config': params['desc'],
        'short_max_hold': params['short_max_hold'],
        'avg_return': avg_return,
        'avg_trades': avg_trades,
        'avg_long': avg_long,
        'avg_short': avg_short,
        'win_rate': avg_wr,
        'avg_short_hold': avg_short_hold
    })

# Final comparison
print(f"\n{'='*80}")
print("IDEA 2 FINAL RESULTS COMPARISON")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values('avg_return', ascending=False)

print(comparison_df.to_string(index=False))

best = comparison_df.iloc[0]
print(f"\n🏆 BEST CONFIGURATION:")
print(f"  Config: {best['config']}")
print(f"  Avg Return: {best['avg_return']:.2f}% per window")
print(f"  Avg Trades: {best['avg_trades']:.1f} (LONG {best['avg_long']:.1f} + SHORT {best['avg_short']:.1f})")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Avg SHORT hold: {best['avg_short_hold']:.0f} candles")

# Capital lock analysis
baseline = comparison_df[comparison_df['short_max_hold'] == 240].iloc[0]
best_short_lock = best['avg_short'] * best['avg_short_hold']
baseline_short_lock = baseline['avg_short'] * baseline['short_max_hold']
reduction = (1 - best_short_lock / baseline_short_lock) * 100

print(f"\n📊 CAPITAL LOCK REDUCTION:")
print(f"  Baseline (4h SHORT): {baseline_short_lock:.1f} candle-trades")
print(f"  Optimized: {best_short_lock:.1f} candle-trades")
print(f"  Reduction: {reduction:.1f}%")

# Save results
output_file = RESULTS_DIR / "idea2_asymmetric_time_results.csv"
comparison_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print(f"\n{'='*80}")
print("IDEA 2 TEST COMPLETE")
print(f"{'='*80}\n")
