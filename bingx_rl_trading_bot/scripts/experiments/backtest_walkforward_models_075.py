"""
Backtest Walk-Forward Decoupled Entry Models (0.75/0.75)
==========================================================

Tests the new integrated Entry models with Exit models.
108-window backtest to validate Win Rate improvement.

Models:
- Entry: walkforward_decoupled_20251027_194313
- Exit: threshold_075_20251027_190512

Created: 2025-10-27
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

from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BACKTEST: WALK-FORWARD DECOUPLED ENTRY MODELS (0.75/0.75)")
print("="*80)
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load Data
print("-"*80)
print("STEP 1: Loading Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Prepare Exit features (adds 15 enhanced features for Exit models)
print("\nPreparing Exit features...")
df = prepare_exit_features(df)
print(f"✅ Exit features added ({len(df.columns)} total columns)")
print()

# Load Entry Models (NEW - Walk-Forward Decoupled)
print("-"*80)
print("STEP 2: Loading Entry Models (Walk-Forward Decoupled)")
print("-"*80)

# LONG Entry
with open(MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f]

# SHORT Entry
with open(MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f]

print(f"  ✅ LONG Entry: {len(long_entry_features)} features")
print(f"  ✅ SHORT Entry: {len(short_entry_features)} features")
print()

# Load Exit Models
print("-"*80)
print("STEP 3: Loading Exit Models")
print("-"*80)

# LONG Exit
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f]

# SHORT Exit
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"  ✅ LONG Exit: {len(long_exit_features)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_features)} features")
print()

# Backtest Function
def simulate_trade(df, entry_idx, side, exit_model, exit_scaler, exit_features):
    """Simulate a single trade"""
    entry_price = df['close'].iloc[entry_idx]

    max_hold_end = min(entry_idx + EMERGENCY_MAX_HOLD + 1, len(df))

    for i in range(1, max_hold_end - entry_idx):
        current_idx = entry_idx + i
        current_price = df['close'].iloc[current_idx]

        # Calculate P&L
        if side == 'LONG':
            pnl = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) / entry_price

        leveraged_pnl = pnl * LEVERAGE

        # Exit condition 1: Stop Loss
        if leveraged_pnl <= EMERGENCY_STOP_LOSS:
            return {
                'exit_idx': current_idx,
                'pnl': leveraged_pnl,
                'hold_time': i,
                'exit_reason': 'stop_loss'
            }

        # Exit condition 2: ML Exit
        try:
            exit_feat = df[exit_features].iloc[current_idx:current_idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    return {
                        'exit_idx': current_idx,
                        'pnl': leveraged_pnl,
                        'hold_time': i,
                        'exit_reason': 'ml_exit'
                    }
        except:
            pass

        # Exit condition 3: Max Hold
        if i >= EMERGENCY_MAX_HOLD:
            return {
                'exit_idx': current_idx,
                'pnl': leveraged_pnl,
                'hold_time': i,
                'exit_reason': 'max_hold'
            }

    # Fallback
    return {
        'exit_idx': max_hold_end - 1,
        'pnl': 0.0,
        'hold_time': max_hold_end - entry_idx - 1,
        'exit_reason': 'data_end'
    }

def backtest_window(df_window):
    """Backtest a 5-day window"""

    # Get Entry probabilities
    X_long = df_window[long_entry_features].values
    X_long_scaled = long_entry_scaler.transform(X_long)
    long_probs = long_entry_model.predict_proba(X_long_scaled)[:, 1]

    X_short = df_window[short_entry_features].values
    X_short_scaled = short_entry_scaler.transform(X_short)
    short_probs = short_entry_model.predict_proba(X_short_scaled)[:, 1]

    # Find entries
    long_entries = np.where(long_probs >= ENTRY_THRESHOLD)[0]
    short_entries = np.where(short_probs >= ENTRY_THRESHOLD)[0]

    trades = []

    # Simulate LONG trades
    for entry_idx in long_entries:
        if entry_idx >= len(df_window) - EMERGENCY_MAX_HOLD:
            continue

        result = simulate_trade(df_window, entry_idx, 'LONG', long_exit_model, long_exit_scaler, long_exit_features)
        result['entry_idx'] = entry_idx
        result['side'] = 'LONG'
        result['entry_prob'] = long_probs[entry_idx]
        trades.append(result)

    # Simulate SHORT trades
    for entry_idx in short_entries:
        if entry_idx >= len(df_window) - EMERGENCY_MAX_HOLD:
            continue

        result = simulate_trade(df_window, entry_idx, 'SHORT', short_exit_model, short_exit_scaler, short_exit_features)
        result['entry_idx'] = entry_idx
        result['side'] = 'SHORT'
        result['entry_prob'] = short_probs[entry_idx]
        trades.append(result)

    return trades

# Run 108-Window Backtest
print("-"*80)
print("STEP 4: Running 108-Window Backtest")
print("-"*80)
print()

WINDOW_SIZE = 1440  # 5 days = 1440 5-min candles
START_IDX = 0
results = []

for window_idx in range(108):
    start = START_IDX + (window_idx * WINDOW_SIZE)
    end = start + WINDOW_SIZE

    if end > len(df):
        break

    df_window = df.iloc[start:end].copy()
    df_window = df_window.reset_index(drop=True)

    # Backtest window
    trades = backtest_window(df_window)

    if len(trades) == 0:
        continue

    # Calculate metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    total_return = sum(t['pnl'] for t in trades)
    avg_return = total_return / total_trades if total_trades > 0 else 0

    long_trades = sum(1 for t in trades if t['side'] == 'LONG')
    short_trades = sum(1 for t in trades if t['side'] == 'SHORT')

    ml_exits = sum(1 for t in trades if t['exit_reason'] == 'ml_exit')
    sl_exits = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
    max_hold_exits = sum(1 for t in trades if t['exit_reason'] == 'max_hold')

    ml_exit_rate = ml_exits / total_trades if total_trades > 0 else 0

    results.append({
        'window': window_idx,
        'total_trades': total_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'win_rate': win_rate * 100,
        'avg_return': avg_return * 100,
        'total_return': total_return * 100,
        'ml_exit_rate': ml_exit_rate * 100,
        'sl_exits': sl_exits,
        'max_hold_exits': max_hold_exits
    })

    if (window_idx + 1) % 20 == 0:
        print(f"  Window {window_idx+1}/108 complete...")

print(f"\n✅ Backtest complete: {len(results)} windows")
print()

# Calculate Overall Statistics
print("="*80)
print("BACKTEST RESULTS")
print("="*80)
print()

df_results = pd.DataFrame(results)

total_trades = df_results['total_trades'].sum()
total_wins = (df_results['win_rate'] / 100 * df_results['total_trades']).sum()
overall_win_rate = total_wins / total_trades if total_trades > 0 else 0

weighted_avg_return = (df_results['avg_return'] * df_results['total_trades']).sum() / total_trades if total_trades > 0 else 0

long_total = df_results['long_trades'].sum()
short_total = df_results['short_trades'].sum()

ml_exits_total = (df_results['ml_exit_rate'] / 100 * df_results['total_trades']).sum()
ml_exit_rate_overall = ml_exits_total / total_trades if total_trades > 0 else 0

print(f"Total Windows: {len(results)}")
print(f"Total Trades: {total_trades:,}")
print(f"  LONG: {long_total:,} ({long_total/total_trades*100:.1f}%)")
print(f"  SHORT: {short_total:,} ({short_total/total_trades*100:.1f}%)")
print()
print(f"Overall Win Rate: {overall_win_rate*100:.2f}%")
print(f"Avg Return per Trade: {weighted_avg_return:.2f}%")
print(f"Avg Trades per Window: {df_results['total_trades'].mean():.1f}")
print()
print(f"Exit Distribution:")
print(f"  ML Exit: {ml_exit_rate_overall*100:.1f}%")
print(f"  Stop Loss: {df_results['sl_exits'].sum():,} ({df_results['sl_exits'].sum()/total_trades*100:.1f}%)")
print(f"  Max Hold: {df_results['max_hold_exits'].sum():,} ({df_results['max_hold_exits'].sum()/total_trades*100:.1f}%)")
print()

# Per-Window Statistics
print(f"Per-Window Statistics:")
print(f"  Avg Win Rate: {df_results['win_rate'].mean():.2f}%")
print(f"  Median Win Rate: {df_results['win_rate'].median():.2f}%")
print(f"  Avg Return: {df_results['total_return'].mean():.2f}%")
print(f"  Median Return: {df_results['total_return'].median():.2f}%")
print()

# Save Results
output_file = RESULTS_DIR / f"backtest_walkforward_models_075_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df_results.to_csv(output_file, index=False)
print(f"✅ Results saved: {output_file.name}")
print()

# Decision
print("="*80)
print("EVALUATION")
print("="*80)
print()

if overall_win_rate >= 0.50:
    print(f"✅ SUCCESS: Win Rate {overall_win_rate*100:.2f}% >= 50%")
    print()
    print("Next Steps:")
    print("  1. Deploy to production with 0.75/0.75 thresholds")
    print("  2. Update bot with new Entry models (walkforward_decoupled_20251027_194313)")
    print("  3. Monitor first week performance")
else:
    print(f"⚠️ BELOW TARGET: Win Rate {overall_win_rate*100:.2f}% < 50%")
    print()
    print("Recommendations:")
    print("  1. Analyze failure modes in low Win Rate windows")
    print("  2. Consider threshold adjustment (0.70, 0.80)")
    print("  3. Investigate feature importance")

print()
