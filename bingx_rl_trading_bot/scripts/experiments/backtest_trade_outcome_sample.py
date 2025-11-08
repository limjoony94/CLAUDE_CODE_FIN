"""
Backtest Trade-Outcome Sample Models
=====================================

Validate trade-outcome sample models on actual trading performance.

Models to Test:
1. Baseline (LONG: 13.7% precision, SHORT: redesigned)
2. Trade-Outcome Sample (LONG: 36.69%, SHORT: 20.78%)

Note: Using windows OUTSIDE of training data (5,000 candles)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Configuration
WINDOW_SIZE = 500  # 5-day windows (500 candles = 41.67 hours ≈ 2 days)
NUM_WINDOWS = 20   # Test 20 windows
LEVERAGE = 4

LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

print("="*80)
print("BACKTEST: TRADE-OUTCOME SAMPLE MODELS")
print("="*80)
print(f"\nConfiguration:")
print(f"  Window Size: {WINDOW_SIZE} candles (~2 days)")
print(f"  Num Windows: {NUM_WINDOWS}")
print(f"  LONG Threshold: {LONG_THRESHOLD}")
print(f"  SHORT Threshold: {SHORT_THRESHOLD}")
print(f"  Gate Threshold: {GATE_THRESHOLD}")

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "-"*80)
print("Loading Data")
print("-"*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"✅ Loaded {len(df_full):,} candles")

# Calculate all features
print("\nCalculating features...")
df_full = calculate_all_features(df_full)
df_full = prepare_exit_features(df_full)
print(f"✅ All features calculated")

# Use data BEFORE the last 5,000 candles (to avoid training data)
df_test = df_full.iloc[:-5000].copy()
print(f"\n📊 Test Data: {len(df_test):,} candles (excluding last 5,000 training sample)")
print(f"   Date range: {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}")

# ============================================================================
# Load Models
# ============================================================================

print("\n" + "-"*80)
print("Loading Models")
print("-"*80)

def load_model_set(model_name):
    """Load entry model + scaler + features"""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_features.txt"

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    scaler = joblib.load(scaler_path)

    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]

    return model, scaler, features

# Baseline models
print("\nLoading BASELINE models...")
baseline_long = load_model_set("xgboost_v4_phase4_advanced_lookahead3_thresh0")
baseline_short = load_model_set("xgboost_short_redesigned_20251016_233322")
print(f"  ✅ Baseline LONG: {len(baseline_long[2])} features")
print(f"  ✅ Baseline SHORT: {len(baseline_short[2])} features")

# Trade-Outcome Sample models
print("\nLoading TRADE-OUTCOME SAMPLE models...")
sample_long = load_model_set("xgboost_long_trade_outcome_sample_20251018_171324")
sample_short = load_model_set("xgboost_short_trade_outcome_sample_20251018_171324")
print(f"  ✅ Sample LONG: {len(sample_long[2])} features")
print(f"  ✅ Sample SHORT: {len(sample_short[2])} features")

# Load Exit models (same as used in trade simulator)
print("\nLoading EXIT models...")
exit_long = load_model_set("xgboost_long_exit_oppgating_improved_20251017_151624")
exit_short = load_model_set("xgboost_short_exit_oppgating_improved_20251017_152440")
print(f"  ✅ Exit LONG: {len(exit_long[2])} features")
print(f"  ✅ Exit SHORT: {len(exit_short[2])} features")

# ============================================================================
# Backtest Function
# ============================================================================

def backtest_window(df_window, entry_models, exit_models, strategy_name):
    """
    Backtest a single window with given models.

    Args:
        df_window: DataFrame with window data
        entry_models: (long_model, long_scaler, long_features,
                       short_model, short_scaler, short_features)
        exit_models: (long_model, long_scaler, long_features,
                      short_model, short_scaler, short_features)
        strategy_name: Name for reporting

    Returns:
        dict: Performance metrics
    """
    long_model, long_scaler, long_features, short_model, short_scaler, short_features = entry_models
    exit_long_model, exit_long_scaler, exit_long_features, exit_short_model, exit_short_scaler, exit_short_features = exit_models

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
                    'entry_idx': position['entry_idx'],
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
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
            if long_prob >= LONG_THRESHOLD:
                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_price': df_window['close'].iloc[i]
                }
            elif short_prob >= SHORT_THRESHOLD:
                # Opportunity gating
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': df_window['close'].iloc[i]
                    }

    # Calculate metrics
    if len(trades) == 0:
        return {
            'strategy': strategy_name,
            'num_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'long_trades': 0,
            'short_trades': 0
        }

    trades_df = pd.DataFrame(trades)

    return {
        'strategy': strategy_name,
        'num_trades': len(trades),
        'win_rate': trades_df['profitable'].mean() * 100,
        'avg_return': trades_df['leveraged_pnl'].mean() * 100,
        'total_return': trades_df['leveraged_pnl'].sum() * 100,
        'long_trades': (trades_df['side'] == 'LONG').sum(),
        'short_trades': (trades_df['side'] == 'SHORT').sum(),
        'ml_exit_pct': (trades_df['exit_reason'] == 'ml_exit').mean() * 100,
        'stop_loss_pct': (trades_df['exit_reason'] == 'stop_loss').mean() * 100
    }

# ============================================================================
# Run Backtests
# ============================================================================

print("\n" + "="*80)
print(f"RUNNING BACKTESTS ({NUM_WINDOWS} windows)")
print("="*80)

results_baseline = []
results_sample = []

# Sample windows from test data
total_windows = len(df_test) // WINDOW_SIZE
step = total_windows // NUM_WINDOWS

for w in range(NUM_WINDOWS):
    start_idx = w * step * WINDOW_SIZE
    end_idx = start_idx + WINDOW_SIZE

    if end_idx > len(df_test):
        break

    df_window = df_test.iloc[start_idx:end_idx].reset_index(drop=True)

    print(f"\nWindow {w+1}/{NUM_WINDOWS}: Candles {start_idx} to {end_idx}")

    # Baseline
    result_baseline = backtest_window(
        df_window,
        (*baseline_long, *baseline_short),
        (*exit_long, *exit_short),
        "Baseline"
    )
    results_baseline.append(result_baseline)

    # Trade-Outcome Sample
    result_sample = backtest_window(
        df_window,
        (*sample_long, *sample_short),
        (*exit_long, *exit_short),
        "Trade-Outcome Sample"
    )
    results_sample.append(result_sample)

    print(f"  Baseline:    {result_baseline['num_trades']} trades, "
          f"{result_baseline['win_rate']:.1f}% WR, "
          f"{result_baseline['total_return']:.2f}% return")
    print(f"  Sample:      {result_sample['num_trades']} trades, "
          f"{result_sample['win_rate']:.1f}% WR, "
          f"{result_sample['total_return']:.2f}% return")

# ============================================================================
# Compare Results
# ============================================================================

print("\n" + "="*80)
print("BACKTEST RESULTS COMPARISON")
print("="*80)

df_baseline = pd.DataFrame(results_baseline)
df_sample = pd.DataFrame(results_sample)

print(f"\n{'Metric':<25} {'Baseline':<15} {'Sample':<15} {'Change':<15}")
print("-"*70)

metrics = [
    ('Avg Trades per Window', 'num_trades', '.1f'),
    ('Avg Win Rate (%)', 'win_rate', '.2f'),
    ('Avg Return per Window (%)', 'avg_return', '.2f'),
    ('Total Return (%)', 'total_return', '.2f'),
    ('Avg LONG Trades', 'long_trades', '.1f'),
    ('Avg SHORT Trades', 'short_trades', '.1f'),
    ('ML Exit Rate (%)', 'ml_exit_pct', '.1f')
]

for label, col, fmt in metrics:
    baseline_val = df_baseline[col].mean()
    sample_val = df_sample[col].mean()

    if baseline_val != 0:
        change_pct = (sample_val - baseline_val) / abs(baseline_val) * 100
        change_str = f"{change_pct:+.1f}%"
    else:
        change_str = "N/A"

    # Format values based on fmt
    if fmt == '.1f':
        baseline_str = f"{baseline_val:.1f}"
        sample_str = f"{sample_val:.1f}"
    elif fmt == '.2f':
        baseline_str = f"{baseline_val:.2f}"
        sample_str = f"{sample_val:.2f}"
    else:
        baseline_str = f"{baseline_val}"
        sample_str = f"{sample_val}"

    print(f"{label:<25} {baseline_str:<15} {sample_str:<15} {change_str:<15}")

# ============================================================================
# Verdict
# ============================================================================

print("\n" + "="*80)
print("VERDICT")
print("="*80)

avg_return_baseline = df_baseline['avg_return'].mean()
avg_return_sample = df_sample['avg_return'].mean()
win_rate_baseline = df_baseline['win_rate'].mean()
win_rate_sample = df_sample['win_rate'].mean()

improvement = avg_return_sample > avg_return_baseline
win_rate_improved = win_rate_sample > win_rate_baseline

print(f"\nAvg Return per Window:")
print(f"  Baseline: {avg_return_baseline:.2f}%")
print(f"  Sample:   {avg_return_sample:.2f}%")
print(f"  Change:   {avg_return_sample - avg_return_baseline:+.2f}%")

print(f"\nWin Rate:")
print(f"  Baseline: {win_rate_baseline:.2f}%")
print(f"  Sample:   {win_rate_sample:.2f}%")
print(f"  Change:   {win_rate_sample - win_rate_baseline:+.2f}%")

if improvement and win_rate_improved:
    print("\n✅ VERDICT: Trade-Outcome Sample models OUTPERFORM Baseline")
    print("   Recommendation: Proceed with full dataset training")
elif improvement or win_rate_improved:
    print("\n⚠️ VERDICT: Mixed results - Partial improvement")
    print("   Recommendation: Analyze trade patterns before full training")
else:
    print("\n❌ VERDICT: Trade-Outcome Sample models UNDERPERFORM Baseline")
    print("   Recommendation: Revise labeling criteria (Risk-Reward thresholds)")

print("\n📊 Next Steps:")
print("   1. Analyze trade distribution (LONG vs SHORT)")
print("   2. Check for overtrading issues")
print("   3. Review Risk-Reward criterion (currently 0.1% pass rate)")
print("   4. If successful: Optimize for full dataset training")
print("="*80)
