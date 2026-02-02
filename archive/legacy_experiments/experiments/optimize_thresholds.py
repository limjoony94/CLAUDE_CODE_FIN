"""
Baseline Threshold Optimization - Grid Search
==============================================

Test multiple threshold combinations to find optimal settings.

Current Baseline:
  LONG: 0.65, SHORT: 0.70, Gate: 0.001
  Performance: 1.36% avg return, 87.39% WR

Grid Search:
  LONG: [0.60, 0.65, 0.70, 0.75]
  SHORT: [0.68, 0.70, 0.72, 0.75]
  Total: 16 combinations
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from itertools import product

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Configuration
WINDOW_SIZE = 500
NUM_WINDOWS = 20
LEVERAGE = 4
GATE_THRESHOLD = 0.001

# Grid Search Parameters
LONG_THRESHOLDS = [0.60, 0.65, 0.70, 0.75]
SHORT_THRESHOLDS = [0.68, 0.70, 0.72, 0.75]

print("="*80)
print("BASELINE THRESHOLD OPTIMIZATION - GRID SEARCH")
print("="*80)
print(f"\nConfiguration:")
print(f"  Window Size: {WINDOW_SIZE} candles")
print(f"  Num Windows: {NUM_WINDOWS}")
print(f"  LONG Thresholds: {LONG_THRESHOLDS}")
print(f"  SHORT Thresholds: {SHORT_THRESHOLDS}")
print(f"  Total Combinations: {len(LONG_THRESHOLDS) * len(SHORT_THRESHOLDS)}")

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "-"*80)
print("Loading Data")
print("-"*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"✅ Loaded {len(df_full):,} candles")

print("\nCalculating features...")
df_full = calculate_all_features(df_full)
df_full = prepare_exit_features(df_full)
print(f"✅ Features calculated")

# Use data BEFORE last 5,000 (avoid training data)
df_test = df_full.iloc[:-5000].copy()
print(f"\n📊 Test Data: {len(df_test):,} candles")
print(f"   Date range: {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}")

# ============================================================================
# Load Models
# ============================================================================

print("\n" + "-"*80)
print("Loading Models")
print("-"*80)

def load_model_set(model_name):
    """Load model + scaler + features"""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_features.txt"

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]

    return model, scaler, features

# Baseline Entry models
baseline_long = load_model_set("xgboost_v4_phase4_advanced_lookahead3_thresh0")
baseline_short = load_model_set("xgboost_short_redesigned_20251016_233322")

# Exit models
exit_long = load_model_set("xgboost_long_exit_oppgating_improved_20251017_151624")
exit_short = load_model_set("xgboost_short_exit_oppgating_improved_20251017_152440")

print(f"✅ All models loaded")

# ============================================================================
# Backtest Function
# ============================================================================

def backtest_window(df_window, long_thresh, short_thresh, gate_thresh):
    """Backtest single window with given thresholds"""
    long_model, long_scaler, long_features = baseline_long
    short_model, short_scaler, short_features = baseline_short
    exit_long_model, exit_long_scaler, exit_long_features = exit_long
    exit_short_model, exit_short_scaler, exit_short_features = exit_short

    trades = []
    position = None

    for i in range(len(df_window) - 1):
        # Check exit if in position
        if position is not None:
            current_price = df_window['close'].iloc[i]
            entry_price = position['entry_price']
            side = position['side']
            hold_time = i - position['entry_idx']

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            leveraged_pnl = pnl_pct * LEVERAGE

            # Exit conditions
            exit_signal = False
            exit_reason = None

            # 1. ML Exit
            if side == 'LONG':
                exit_features_vals = df_window[exit_long_features].iloc[i].values.reshape(1, -1)
                exit_features_scaled = exit_long_scaler.transform(exit_features_vals)
                exit_prob = exit_long_model.predict_proba(exit_features_scaled)[0][1]
                if exit_prob >= 0.70:
                    exit_signal = True
                    exit_reason = 'ml_exit'
            else:
                exit_features_vals = df_window[exit_short_features].iloc[i].values.reshape(1, -1)
                exit_features_scaled = exit_short_scaler.transform(exit_features_vals)
                exit_prob = exit_short_model.predict_proba(exit_features_scaled)[0][1]
                if exit_prob >= 0.70:
                    exit_signal = True
                    exit_reason = 'ml_exit'

            # 2. Emergency Stop Loss
            if leveraged_pnl <= -0.04:
                exit_signal = True
                exit_reason = 'stop_loss'

            # 3. Emergency Max Hold
            if hold_time >= 96:
                exit_signal = True
                exit_reason = 'max_hold'

            if exit_signal:
                trades.append({
                    'side': side,
                    'pnl_pct': pnl_pct,
                    'leveraged_pnl': leveraged_pnl,
                    'hold_time': hold_time,
                    'exit_reason': exit_reason,
                    'profitable': leveraged_pnl > 0
                })
                position = None

        # Check entry if no position
        if position is None:
            # Get LONG entry signal
            long_features_vals = df_window[long_features].iloc[i].values.reshape(1, -1)
            long_features_scaled = long_scaler.transform(long_features_vals)
            long_prob = long_model.predict_proba(long_features_scaled)[0][1]

            # Get SHORT entry signal
            short_features_vals = df_window[short_features].iloc[i].values.reshape(1, -1)
            short_features_scaled = short_scaler.transform(short_features_vals)
            short_prob = short_model.predict_proba(short_features_scaled)[0][1]

            # Entry logic with opportunity gating
            if long_prob >= long_thresh:
                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_price': df_window['close'].iloc[i]
                }
            elif short_prob >= short_thresh:
                # Opportunity gating
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > gate_thresh:
                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': df_window['close'].iloc[i]
                    }

    # Calculate metrics
    if len(trades) == 0:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0
        }

    trades_df = pd.DataFrame(trades)

    return {
        'num_trades': len(trades),
        'win_rate': trades_df['profitable'].mean() * 100,
        'avg_return': trades_df['leveraged_pnl'].mean() * 100,
        'total_return': trades_df['leveraged_pnl'].sum() * 100,
        'long_trades': (trades_df['side'] == 'LONG').sum(),
        'short_trades': (trades_df['side'] == 'SHORT').sum()
    }

# ============================================================================
# Grid Search
# ============================================================================

print("\n" + "="*80)
print("RUNNING GRID SEARCH")
print("="*80)

# Sample windows
total_windows = len(df_test) // WINDOW_SIZE
step = total_windows // NUM_WINDOWS

windows = []
for w in range(NUM_WINDOWS):
    start_idx = w * step * WINDOW_SIZE
    end_idx = start_idx + WINDOW_SIZE
    if end_idx <= len(df_test):
        windows.append(df_test.iloc[start_idx:end_idx].reset_index(drop=True))

print(f"\n✅ Prepared {len(windows)} windows for testing")

# Grid search
all_results = []
total_combinations = len(LONG_THRESHOLDS) * len(SHORT_THRESHOLDS)
current = 0

for long_thresh, short_thresh in product(LONG_THRESHOLDS, SHORT_THRESHOLDS):
    current += 1
    print(f"\n[{current}/{total_combinations}] Testing LONG={long_thresh:.2f}, SHORT={short_thresh:.2f}")

    window_results = []

    for w, df_window in enumerate(windows):
        result = backtest_window(df_window, long_thresh, short_thresh, GATE_THRESHOLD)
        window_results.append(result)

    # Calculate aggregate metrics
    df_results = pd.DataFrame(window_results)

    agg_result = {
        'long_thresh': long_thresh,
        'short_thresh': short_thresh,
        'avg_trades': df_results['num_trades'].mean(),
        'avg_win_rate': df_results['win_rate'].mean(),
        'avg_return': df_results['avg_return'].mean(),
        'total_return': df_results['total_return'].sum(),
        'avg_long_trades': df_results['long_trades'].mean(),
        'avg_short_trades': df_results['short_trades'].mean(),
        'num_zero_trade_windows': (df_results['num_trades'] == 0).sum()
    }

    all_results.append(agg_result)

    print(f"  Avg Return: {agg_result['avg_return']:.2f}%, WR: {agg_result['avg_win_rate']:.1f}%, Trades: {agg_result['avg_trades']:.1f}")

# ============================================================================
# Analysis
# ============================================================================

print("\n" + "="*80)
print("GRID SEARCH RESULTS")
print("="*80)

df_grid = pd.DataFrame(all_results)
df_grid = df_grid.sort_values('avg_return', ascending=False)

print(f"\n{'Rank':<5} {'LONG':<6} {'SHORT':<7} {'Avg Ret':<10} {'WR':<8} {'Trades':<8} {'LONG/SHORT'}")
print("-"*80)

for idx, row in df_grid.head(10).iterrows():
    print(f"{idx+1:<5} {row['long_thresh']:.2f}   {row['short_thresh']:.2f}    "
          f"{row['avg_return']:>7.2f}%   {row['avg_win_rate']:>6.1f}%  "
          f"{row['avg_trades']:>6.1f}  "
          f"{row['avg_long_trades']:.1f}/{row['avg_short_trades']:.1f}")

# Best configuration
best = df_grid.iloc[0]
current_baseline = df_grid[(df_grid['long_thresh'] == 0.65) & (df_grid['short_thresh'] == 0.70)].iloc[0]

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print(f"\n🏆 BEST CONFIGURATION:")
print(f"   LONG Threshold: {best['long_thresh']:.2f}")
print(f"   SHORT Threshold: {best['short_thresh']:.2f}")
print(f"   Avg Return: {best['avg_return']:.2f}%")
print(f"   Win Rate: {best['avg_win_rate']:.1f}%")
print(f"   Avg Trades: {best['avg_trades']:.1f}")

print(f"\n📊 CURRENT BASELINE (LONG=0.65, SHORT=0.70):")
print(f"   Avg Return: {current_baseline['avg_return']:.2f}%")
print(f"   Win Rate: {current_baseline['avg_win_rate']:.1f}%")
print(f"   Avg Trades: {current_baseline['avg_trades']:.1f}")

improvement = (best['avg_return'] - current_baseline['avg_return']) / abs(current_baseline['avg_return']) * 100

print(f"\n{'='*80}")
if improvement > 5:
    print(f"✅ VERDICT: Best config shows {improvement:+.1f}% improvement")
    print(f"   Recommendation: Update thresholds to LONG={best['long_thresh']:.2f}, SHORT={best['short_thresh']:.2f}")
elif improvement > 0:
    print(f"⚠️ VERDICT: Modest improvement ({improvement:+.1f}%)")
    print(f"   Recommendation: Consider updating thresholds if consistent across more tests")
else:
    print(f"❌ VERDICT: Current baseline is optimal ({improvement:+.1f}%)")
    print(f"   Recommendation: Keep current thresholds (LONG=0.65, SHORT=0.70)")

print("\n📊 Full Results saved to CSV for detailed analysis")

# Save results
results_file = PROJECT_ROOT / "results" / "threshold_optimization_results.csv"
df_grid.to_csv(results_file, index=False)
print(f"   File: {results_file}")

print("="*80)
