"""
Backtest Relaxed Risk-Reward Models
====================================

Validate relaxed RR models on actual trading performance.

Models to Test:
1. Baseline (LONG: 13.7% precision, SHORT: redesigned)
2. Strict RR Sample (LONG: 36.69%, SHORT: 20.78%) [MAE -2%, MFE 4%]
3. Relaxed RR Sample (LONG: 36.69%, SHORT: 21.14%) [MAE -3%, MFE 2.5%]

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
print("BACKTEST: RELAXED RISK-REWARD MODELS")
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

# Strict RR Sample models
print("\nLoading STRICT RR SAMPLE models...")
strict_long = load_model_set("xgboost_long_trade_outcome_sample_20251018_171324")
strict_short = load_model_set("xgboost_short_trade_outcome_sample_20251018_171324")
print(f"  ✅ Strict LONG: {len(strict_long[2])} features")
print(f"  ✅ Strict SHORT: {len(strict_short[2])} features")

# Relaxed RR Sample models
print("\nLoading RELAXED RR SAMPLE models...")
relaxed_long = load_model_set("xgboost_long_relaxed_rr_20251018_175953")
relaxed_short = load_model_set("xgboost_short_relaxed_rr_20251018_175953")
print(f"  ✅ Relaxed LONG: {len(relaxed_long[2])} features")
print(f"  ✅ Relaxed SHORT: {len(relaxed_short[2])} features")

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
results_strict = []
results_relaxed = []

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

    # Strict RR
    result_strict = backtest_window(
        df_window,
        (*strict_long, *strict_short),
        (*exit_long, *exit_short),
        "Strict RR"
    )
    results_strict.append(result_strict)

    # Relaxed RR
    result_relaxed = backtest_window(
        df_window,
        (*relaxed_long, *relaxed_short),
        (*exit_long, *exit_short),
        "Relaxed RR"
    )
    results_relaxed.append(result_relaxed)

    print(f"  Baseline:    {result_baseline['num_trades']} trades, "
          f"{result_baseline['win_rate']:.1f}% WR, "
          f"{result_baseline['total_return']:.2f}% return")
    print(f"  Strict RR:   {result_strict['num_trades']} trades, "
          f"{result_strict['win_rate']:.1f}% WR, "
          f"{result_strict['total_return']:.2f}% return")
    print(f"  Relaxed RR:  {result_relaxed['num_trades']} trades, "
          f"{result_relaxed['win_rate']:.1f}% WR, "
          f"{result_relaxed['total_return']:.2f}% return")

# ============================================================================
# Compare Results
# ============================================================================

print("\n" + "="*80)
print("BACKTEST RESULTS COMPARISON")
print("="*80)

df_baseline = pd.DataFrame(results_baseline)
df_strict = pd.DataFrame(results_strict)
df_relaxed = pd.DataFrame(results_relaxed)

print(f"\n{'Metric':<30} {'Baseline':<12} {'Strict RR':<12} {'Relaxed RR':<12}")
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
    strict_val = df_strict[col].mean()
    relaxed_val = df_relaxed[col].mean()

    # Format values based on fmt
    if fmt == '.1f':
        baseline_str = f"{baseline_val:.1f}"
        strict_str = f"{strict_val:.1f}"
        relaxed_str = f"{relaxed_val:.1f}"
    elif fmt == '.2f':
        baseline_str = f"{baseline_val:.2f}"
        strict_str = f"{strict_val:.2f}"
        relaxed_str = f"{relaxed_val:.2f}"
    else:
        baseline_str = f"{baseline_val}"
        strict_str = f"{strict_val}"
        relaxed_str = f"{relaxed_val}"

    print(f"{label:<30} {baseline_str:<12} {strict_str:<12} {relaxed_str:<12}")

# ============================================================================
# Verdict
# ============================================================================

print("\n" + "="*80)
print("VERDICT")
print("="*80)

avg_return_baseline = df_baseline['avg_return'].mean()
avg_return_strict = df_strict['avg_return'].mean()
avg_return_relaxed = df_relaxed['avg_return'].mean()
win_rate_baseline = df_baseline['win_rate'].mean()
win_rate_strict = df_strict['win_rate'].mean()
win_rate_relaxed = df_relaxed['win_rate'].mean()

print(f"\nAvg Return per Window:")
print(f"  Baseline:    {avg_return_baseline:.2f}%")
print(f"  Strict RR:   {avg_return_strict:.2f}% ({avg_return_strict - avg_return_baseline:+.2f}%)")
print(f"  Relaxed RR:  {avg_return_relaxed:.2f}% ({avg_return_relaxed - avg_return_baseline:+.2f}%)")

print(f"\nWin Rate:")
print(f"  Baseline:    {win_rate_baseline:.2f}%")
print(f"  Strict RR:   {win_rate_strict:.2f}% ({win_rate_strict - win_rate_baseline:+.2f}%)")
print(f"  Relaxed RR:  {win_rate_relaxed:.2f}% ({win_rate_relaxed - win_rate_baseline:+.2f}%)")

# Determine best model
best_model = "Baseline"
best_return = avg_return_baseline

if avg_return_strict > best_return:
    best_model = "Strict RR"
    best_return = avg_return_strict

if avg_return_relaxed > best_return:
    best_model = "Relaxed RR"
    best_return = avg_return_relaxed

print(f"\n🏆 BEST MODEL: {best_model} ({best_return:.2f}% avg return)")

# Verdict
if best_model == "Baseline":
    print("\n❌ VERDICT: Both RR models UNDERPERFORM Baseline")
    print("   Recommendation: Keep Baseline models, proceed to Threshold optimization")
elif best_model == "Relaxed RR":
    print("\n✅ VERDICT: Relaxed RR OUTPERFORMS Baseline and Strict RR")
    print("   Recommendation: Consider using Relaxed RR models")
    if avg_return_relaxed > avg_return_baseline * 1.1:
        print("   Strong improvement (>10%) - Recommended for deployment")
    else:
        print("   Moderate improvement (<10%) - Further validation recommended")
else:
    print("\n⚠️ VERDICT: Strict RR shows improvement")
    print("   Recommendation: Analyze why Strict RR works but not Relaxed")

print("\n📊 Next Steps:")
if best_model == "Baseline":
    print("   1. Proceed to Threshold optimization on Baseline models")
    print("   2. Test threshold combinations (LONG: 0.65-0.75, SHORT: 0.70-0.75)")
    print("   3. Select best performing thresholds")
else:
    print("   1. Validate winning RR model on additional windows")
    print("   2. Consider full dataset training if sample results hold")
    print("   3. Compare with Threshold-optimized Baseline")
print("="*80)
