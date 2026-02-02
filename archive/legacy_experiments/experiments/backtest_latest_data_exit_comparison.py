"""
EXIT Threshold Comparison - Latest Data (Oct 26 - Nov 1)
=========================================================
Using CURRENT production settings:
- Model: oppgating_improved_20251024_044510
- Data: Latest 7 days with PRODUCTION features
- Compare: EXIT 0.15, 0.20, 0.75
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

print("=" * 80)
print("EXIT THRESHOLD COMPARISON - LATEST DATA")
print("=" * 80)
print()

# Load data with PRODUCTION features
print("Loading market data...")
df = pd.read_csv(RESULTS_DIR / "BTCUSDT_5m_latest4weeks_PRODUCTION_FEATURES.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✓ Data loaded: {len(df):,} candles")
print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

# Load CURRENT production models (oppgating_improved)
print("Loading CURRENT production models...")

# LONG Exit
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl", 'rb') as f:
    long_exit_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

# SHORT Exit
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl", 'rb') as f:
    short_exit_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ LONG Exit: {len(long_exit_features)} features")
print(f"✓ SHORT Exit: {len(short_exit_features)} features")
print()

# Backtest parameters
LEVERAGE = 4
INITIAL_CAPITAL = 10000
EMERGENCY_MAX_HOLD = 120  # 10 hours (production setting)
THRESHOLDS_TO_TEST = [0.15, 0.20, 0.75]

def simulate_trade(df, entry_idx, side, exit_threshold):
    """Simulate a single trade with given exit threshold"""
    entry_price = df.iloc[entry_idx]['close']

    # Select model/scaler/features
    if side == 'LONG':
        exit_model = long_exit_model
        exit_scaler = long_exit_scaler
        exit_feats = long_exit_features
    else:  # SHORT
        exit_model = short_exit_model
        exit_scaler = short_exit_scaler
        exit_feats = short_exit_features

    # Simulate holding position
    for hold_time in range(1, min(len(df) - entry_idx, EMERGENCY_MAX_HOLD + 1)):
        current_idx = entry_idx + hold_time
        current_candle = df.iloc[current_idx]
        current_price = current_candle['close']

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price

        leveraged_pnl = pnl_pct * LEVERAGE

        # Check ML Exit
        try:
            exit_feat = current_candle[exit_feats].values.reshape(1, -1)
            exit_feat_scaled = exit_scaler.transform(exit_feat)
            exit_prob = exit_model.predict_proba(exit_feat_scaled)[0, 1]

            if exit_prob >= exit_threshold:
                return {
                    'exit_reason': 'ml_exit',
                    'hold_time': hold_time,
                    'pnl': leveraged_pnl,
                    'exit_prob': exit_prob
                }
        except:
            pass

        # Emergency Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            return {
                'exit_reason': 'max_hold',
                'hold_time': hold_time,
                'pnl': leveraged_pnl,
                'exit_prob': None
            }

    # End of data
    return {
        'exit_reason': 'end_of_data',
        'hold_time': len(df) - entry_idx - 1,
        'pnl': leveraged_pnl if 'leveraged_pnl' in locals() else 0,
        'exit_prob': None
    }

def backtest_exit_threshold(df, exit_threshold):
    """Backtest with given exit threshold"""
    results = []
    capital = INITIAL_CAPITAL

    # Simple strategy: Enter on every candle, alternate LONG/SHORT
    for i in range(0, len(df) - EMERGENCY_MAX_HOLD - 10, 20):  # Sample every 20 candles
        side = 'LONG' if i % 2 == 0 else 'SHORT'

        trade_result = simulate_trade(df, i, side, exit_threshold)

        pnl_amount = capital * trade_result['pnl']
        capital += pnl_amount

        results.append({
            'entry_idx': i,
            'side': side,
            'pnl': trade_result['pnl'],
            'pnl_amount': pnl_amount,
            'hold_time': trade_result['hold_time'],
            'exit_reason': trade_result['exit_reason'],
            'exit_prob': trade_result['exit_prob'],
            'capital': capital
        })

    df_results = pd.DataFrame(results)

    # Calculate metrics
    total_trades = len(df_results)
    wins = len(df_results[df_results['pnl'] > 0])
    losses = len(df_results[df_results['pnl'] < 0])
    win_rate = wins / total_trades if total_trades > 0 else 0

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    avg_pnl = df_results['pnl'].mean()

    ml_exits = len(df_results[df_results['exit_reason'] == 'ml_exit'])
    max_hold_exits = len(df_results[df_results['exit_reason'] == 'max_hold'])
    ml_exit_rate = ml_exits / total_trades if total_trades > 0 else 0

    return {
        'threshold': exit_threshold,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_pnl': avg_pnl,
        'final_capital': capital,
        'ml_exits': ml_exits,
        'max_hold_exits': max_hold_exits,
        'ml_exit_rate': ml_exit_rate,
        'avg_hold_time': df_results['hold_time'].mean()
    }

# Run backtests for all thresholds
print("Running backtests for EXIT thresholds...")
print()

all_results = []

for threshold in THRESHOLDS_TO_TEST:
    print(f"Testing EXIT threshold: {threshold:.2f}...")
    result = backtest_exit_threshold(df, threshold)
    all_results.append(result)

    print(f"  Trades: {result['total_trades']}")
    print(f"  Win Rate: {result['win_rate']:.1%}")
    print(f"  Total Return: {result['total_return']:+.2%}")
    print(f"  Final Capital: ${result['final_capital']:,.2f}")
    print(f"  ML Exit Rate: {result['ml_exit_rate']:.1%}")
    print(f"  Avg Hold Time: {result['avg_hold_time']:.1f} candles")
    print()

# Create comparison DataFrame
df_comparison = pd.DataFrame(all_results)

# Save results
output_file = RESULTS_DIR / f"latest_exit_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df_comparison.to_csv(output_file, index=False)

print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print()

print(df_comparison[['threshold', 'total_trades', 'win_rate', 'total_return', 'ml_exit_rate', 'avg_hold_time']].to_string(index=False))
print()

# Find winner
best_idx = df_comparison['total_return'].idxmax()
best_threshold = df_comparison.loc[best_idx, 'threshold']

print(f"✅ BEST THRESHOLD: {best_threshold:.2f}")
print(f"   Return: {df_comparison.loc[best_idx, 'total_return']:+.2%}")
print(f"   Win Rate: {df_comparison.loc[best_idx, 'win_rate']:.1%}")
print(f"   ML Exit Rate: {df_comparison.loc[best_idx, 'ml_exit_rate']:.1%}")
print()

print(f"Results saved: {output_file.name}")
print()
print("=" * 80)
