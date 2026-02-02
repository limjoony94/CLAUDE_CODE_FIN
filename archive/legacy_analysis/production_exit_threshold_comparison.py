"""
Production Exit Threshold Comparison
=====================================
Purpose: Compare Exit thresholds using ACTUAL production trades
Method: Replay production trades with different Exit thresholds
Data: Real trades from production state file + live market data
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Import prepare_exit_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

print("=" * 80)
print("PRODUCTION EXIT THRESHOLD COMPARISON")
print("=" * 80)
print("Method: Replay production trades with different Exit thresholds")
print("Data: Actual production trades + market data")
print()

# Load Exit models
print("Loading Exit models...")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ LONG Exit: {len(long_exit_features)} features")
print(f"✓ SHORT Exit: {len(short_exit_features)} features")
print()

# Load production state
print("Loading production state...")
with open(RESULTS_DIR / "opportunity_gating_bot_4x_state.json", 'r') as f:
    state = json.load(f)

trades = state.get('trades', [])
closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

print(f"✓ Found {len(closed_trades)} closed trades in production")
print()

# Load market data with Exit features
print("Loading market data...")
df_recent = pd.read_csv(RESULTS_DIR / "BTCUSDT_5m_latest4weeks_features_20251101.csv")
df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'])
df_recent = prepare_exit_features(df_recent)
print(f"✓ Market data: {len(df_recent):,} candles ({df_recent['timestamp'].min()} to {df_recent['timestamp'].max()})")
print()

# Exit thresholds to test
EXIT_THRESHOLDS = [0.15, 0.20, 0.75]
LEVERAGE = 4
EMERGENCY_STOP_LOSS = 0.03
EMERGENCY_MAX_HOLD_TIME = 120

print(f"Testing {len(EXIT_THRESHOLDS)} Exit thresholds: {EXIT_THRESHOLDS}")
print()

def replay_trade_with_threshold(trade, exit_threshold):
    """Replay a production trade with different Exit threshold"""

    # Extract trade info
    side = trade.get('side')

    # Normalize side (BUY → LONG, SELL → SHORT)
    if side == 'BUY':
        side = 'LONG'
    elif side == 'SELL':
        side = 'SHORT'

    entry_time = pd.to_datetime(trade.get('entry_time'))
    actual_exit_time = pd.to_datetime(trade.get('exit_time')) if trade.get('exit_time') else None
    entry_price = trade.get('entry_price')

    if not all([side, entry_time, entry_price]):
        return None

    if side not in ['LONG', 'SHORT']:
        return None

    # Find entry candle in market data
    entry_candles = df_recent[df_recent['timestamp'] >= entry_time].head(1)
    if entry_candles.empty:
        return None

    entry_idx = entry_candles.index[0]

    # Determine actual exit idx if available
    actual_exit_idx = None
    if actual_exit_time:
        exit_candles = df_recent[df_recent['timestamp'] >= actual_exit_time].head(1)
        if not exit_candles.empty:
            actual_exit_idx = exit_candles.index[0]

    # Select Exit model and features
    exit_model = long_exit_model if side == 'LONG' else short_exit_model
    exit_features_list = long_exit_features if side == 'LONG' else short_exit_features

    # Simulate holding position
    max_candles = min(EMERGENCY_MAX_HOLD_TIME, len(df_recent) - entry_idx)

    for hold_time in range(1, max_candles + 1):
        current_idx = entry_idx + hold_time
        if current_idx >= len(df_recent):
            break

        current_candle = df_recent.iloc[current_idx]
        current_price = current_candle['close']

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        leveraged_pnl_pct = pnl_pct * LEVERAGE

        # Check Emergency Stop Loss
        if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
            return {
                'exit_idx': current_idx,
                'exit_price': current_price,
                'hold_time': hold_time,
                'pnl_pct': leveraged_pnl_pct * 100,
                'exit_reason': 'stop_loss',
                'exit_prob': None
            }

        # Check Emergency Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD_TIME:
            return {
                'exit_idx': current_idx,
                'exit_price': current_price,
                'hold_time': hold_time,
                'pnl_pct': leveraged_pnl_pct * 100,
                'exit_reason': 'max_hold',
                'exit_prob': None
            }

        # Check ML Exit
        try:
            exit_feat = current_candle[exit_features_list].values.reshape(1, -1)
            exit_prob = exit_model.predict_proba(exit_feat)[0, 1]

            if exit_prob >= exit_threshold:
                return {
                    'exit_idx': current_idx,
                    'exit_price': current_price,
                    'hold_time': hold_time,
                    'pnl_pct': leveraged_pnl_pct * 100,
                    'exit_reason': 'ml_exit',
                    'exit_prob': exit_prob
                }
        except:
            # Missing features - skip ML Exit check
            pass

    # If loop completes without exit (shouldn't happen)
    current_idx = entry_idx + max_candles
    if current_idx >= len(df_recent):
        current_idx = len(df_recent) - 1
    current_candle = df_recent.iloc[current_idx]
    current_price = current_candle['close']

    if side == 'LONG':
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price

    leveraged_pnl_pct = pnl_pct * LEVERAGE

    return {
        'exit_idx': current_idx,
        'exit_price': current_price,
        'hold_time': max_candles,
        'pnl_pct': leveraged_pnl_pct * 100,
        'exit_reason': 'max_hold',
        'exit_prob': None
    }

# Compare thresholds
results = []

for threshold in EXIT_THRESHOLDS:
    print(f"Testing Exit threshold {threshold:.2f}...", flush=True)

    threshold_results = []

    for trade in closed_trades:
        result = replay_trade_with_threshold(trade, threshold)
        if result:
            threshold_results.append({
                'side': trade.get('side'),
                'entry_time': trade.get('entry_time'),
                'entry_price': trade.get('entry_price'),
                **result
            })

    if not threshold_results:
        print(f"  ❌ No valid replays")
        continue

    df_trades = pd.DataFrame(threshold_results)

    # Calculate metrics
    wins = df_trades[df_trades['pnl_pct'] > 0]
    losses = df_trades[df_trades['pnl_pct'] <= 0]

    win_rate = len(wins) / len(df_trades) * 100 if len(df_trades) > 0 else 0
    avg_return = df_trades['pnl_pct'].mean()

    ml_exits = len(df_trades[df_trades['exit_reason'] == 'ml_exit'])
    ml_exit_pct = ml_exits / len(df_trades) * 100 if len(df_trades) > 0 else 0

    avg_hold = df_trades['hold_time'].mean()

    results.append({
        'exit_threshold': threshold,
        'total_trades': len(df_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'ml_exit_pct': ml_exit_pct,
        'avg_hold': avg_hold
    })

    marker = "🔵 CURRENT" if threshold == 0.75 else ""
    print(f"  ✓ Trades {len(df_trades)} | WR {win_rate:.1f}% | Return {avg_return:+.2f}% | ML Exit {ml_exit_pct:.1f}% | Hold {avg_hold:.1f} {marker}")

print()
print("=" * 80)
print("PRODUCTION TRADES - EXIT THRESHOLD COMPARISON")
print("=" * 80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('exit_threshold').reset_index(drop=True)

print()
print("Exit   Trades  WR      Avg Ret   ML Exit   Avg Hold")
print("─" * 60)
for idx, row in df_results.iterrows():
    marker = "🔵" if row['exit_threshold'] == 0.75 else "  "
    print(f"{row['exit_threshold']:.2f}   {int(row['total_trades']):4d}   {row['win_rate']:5.1f}%  {row['avg_return']:+7.2f}%  {row['ml_exit_pct']:5.1f}%   {row['avg_hold']:5.1f} {marker}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = RESULTS_DIR / f"production_exit_threshold_comparison_{timestamp}.csv"
df_results.to_csv(output_file, index=False)

print()
print(f"✓ Results saved: {output_file.name}")
print()

# Analysis
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

best_return = df_results.loc[df_results['avg_return'].idxmax()]
best_wr = df_results.loc[df_results['win_rate'].idxmax()]
best_ml = df_results.loc[df_results['ml_exit_pct'].idxmax()]

print(f"Best Return: Exit {best_return['exit_threshold']:.2f} ({best_return['avg_return']:+.2f}%)")
print(f"Best Win Rate: Exit {best_wr['exit_threshold']:.2f} ({best_wr['win_rate']:.1f}%)")
print(f"Most ML Exits: Exit {best_ml['exit_threshold']:.2f} ({best_ml['ml_exit_pct']:.1f}%)")
print()

current_perf = df_results[df_results['exit_threshold'] == 0.75].iloc[0]
print(f"Current Production (Exit 0.75):")
print(f"  Return: {current_perf['avg_return']:+.2f}%")
print(f"  Win Rate: {current_perf['win_rate']:.1f}%")
print(f"  ML Exit: {current_perf['ml_exit_pct']:.1f}%")
print()

print("=" * 80)
