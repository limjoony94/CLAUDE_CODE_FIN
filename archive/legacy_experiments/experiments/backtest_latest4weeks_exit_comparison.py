"""
Backtest Latest 4-Week Data - Exit Threshold Comparison
========================================================
User Request: "최신 4주로 새로 받아서 백테스트를 진행해 주세요"
Purpose: Validate Exit threshold findings on fresh recent data
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Exit thresholds to test (including current production)
EXIT_THRESHOLDS = [0.15, 0.20, 0.75]

# Configuration (match actual production)
LONG_ENTRY_THRESHOLD = 0.80
SHORT_ENTRY_THRESHOLD = 0.80
EMERGENCY_STOP_LOSS = 0.03
EMERGENCY_MAX_HOLD_TIME = 120
LEVERAGE = 4
OPPORTUNITY_GATE_THRESHOLD = 0.001

print("=" * 80)
print("BACKTEST LATEST 4-WEEK DATA - EXIT THRESHOLD COMPARISON")
print("=" * 80)
print(f"Testing {len(EXIT_THRESHOLDS)} Exit thresholds: {EXIT_THRESHOLDS}")
print(f"Data: Latest 4 weeks (Oct 2025)")
print()

# Load models (ACTUAL production models)
print("Loading models...")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ LONG Entry: {len(long_entry_features)} features, threshold {LONG_ENTRY_THRESHOLD}")
print(f"✓ SHORT Entry: {len(short_entry_features)} features, threshold {SHORT_ENTRY_THRESHOLD}")
print(f"✓ LONG Exit: {len(long_exit_features)} features")
print(f"✓ SHORT Exit: {len(short_exit_features)} features")
print()

# Load latest 4-week data
print("Loading latest 4-week data...")
feature_files = list((DATA_DIR / "features").glob("BTCUSDT_5m_features_latest4weeks_*.csv"))
if not feature_files:
    print("❌ No feature file found!")
    exit(1)

latest_features = sorted(feature_files)[-1]
df_master = pd.read_csv(latest_features)
df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])

print(f"✓ Loaded {len(df_master):,} candles from {latest_features.name}")
print(f"✓ Data range: {df_master['timestamp'].min()} to {df_master['timestamp'].max()}")
print()

# Define 5-day windows (1440 candles per window)
window_size = 1440
num_windows = len(df_master) // window_size
print(f"Testing {num_windows} windows (5-day periods)")
print()

def backtest_threshold(exit_threshold_long, exit_threshold_short):
    """Run backtest with specific Exit thresholds"""

    all_trades = []

    for window_idx in range(num_windows):
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        df_window = df_master.iloc[start_idx:end_idx].copy()

        balance = 10000
        position = None

        for i in range(len(df_window)):
            current_candle = df_window.iloc[i]

            # Check exit if position open
            if position is not None:
                hold_time = i - position['entry_idx']
                current_price = current_candle['close']

                # Calculate P&L
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

                leveraged_pnl_pct = pnl_pct * LEVERAGE

                # Emergency Stop Loss (balance-based)
                if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount

                    all_trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'stop_loss'
                    })
                    position = None
                    continue

                # Emergency Max Hold
                if hold_time >= EMERGENCY_MAX_HOLD_TIME:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount

                    all_trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'max_hold'
                    })
                    position = None
                    continue

                # ML Exit
                exit_features = current_candle[long_exit_features if position['side'] == 'LONG' else short_exit_features].values.reshape(1, -1)
                exit_model = long_exit_model if position['side'] == 'LONG' else short_exit_model
                exit_threshold = exit_threshold_long if position['side'] == 'LONG' else exit_threshold_short

                exit_prob = exit_model.predict_proba(exit_features)[0, 1]

                if exit_prob >= exit_threshold:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount

                    all_trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'ml_exit'
                    })
                    position = None
                    continue

            # Entry signals (only if no position)
            if position is None:
                # LONG Entry
                long_feat = current_candle[long_entry_features].values.reshape(1, -1)
                long_feat_scaled = long_entry_scaler.transform(long_feat)
                long_prob = long_entry_model.predict_proba(long_feat_scaled)[0, 1]

                # SHORT Entry
                short_feat = current_candle[short_entry_features].values.reshape(1, -1)
                short_feat_scaled = short_entry_scaler.transform(short_feat)
                short_prob = short_entry_model.predict_proba(short_feat_scaled)[0, 1]

                # Opportunity Gating
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047

                if long_prob >= LONG_ENTRY_THRESHOLD:
                    position = {
                        'side': 'LONG',
                        'entry_price': current_candle['close'],
                        'entry_idx': i
                    }
                elif short_prob >= SHORT_ENTRY_THRESHOLD and (short_ev - long_ev) > OPPORTUNITY_GATE_THRESHOLD:
                    position = {
                        'side': 'SHORT',
                        'entry_price': current_candle['close'],
                        'entry_idx': i
                    }

    # Calculate metrics
    if not all_trades:
        return None

    df_trades = pd.DataFrame(all_trades)

    wins = df_trades[df_trades['pnl_amount'] > 0]
    losses = df_trades[df_trades['pnl_amount'] <= 0]

    win_rate = len(wins) / len(df_trades) * 100
    avg_return = df_trades['pnl_pct'].mean()

    ml_exits = len(df_trades[df_trades['exit_reason'] == 'ml_exit'])
    ml_exit_pct = ml_exits / len(df_trades) * 100

    avg_hold = df_trades['hold_time'].mean()

    long_trades = len(df_trades[df_trades['side'] == 'LONG'])
    short_trades = len(df_trades[df_trades['side'] == 'SHORT'])

    return {
        'total_trades': len(df_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'ml_exit_pct': ml_exit_pct,
        'avg_hold': avg_hold,
        'long_trades': long_trades,
        'short_trades': short_trades
    }

# Test all Exit thresholds
results = []

for threshold in EXIT_THRESHOLDS:
    print(f"Testing Exit threshold {threshold:.2f}...", end=" ", flush=True)

    metrics = backtest_threshold(threshold, threshold)

    if metrics is None:
        print("❌ No trades")
        continue

    result = {
        'exit_threshold': threshold,
        **metrics
    }

    results.append(result)

    marker = "🔵 CURRENT" if threshold == 0.75 else ""
    print(f"✓ WR {metrics['win_rate']:.1f}% | Return {metrics['avg_return']:+.2f}% | ML Exit {metrics['ml_exit_pct']:.1f}% | Hold {metrics['avg_hold']:.1f} {marker}")

print()
print("=" * 80)
print("LATEST 4-WEEK DATA - EXIT THRESHOLD COMPARISON")
print("=" * 80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('exit_threshold').reset_index(drop=True)

print()
print("Exit   Trades  WR      Avg Ret   ML Exit   Hold    LONG/SHORT")
print("─" * 70)
for idx, row in df_results.iterrows():
    marker = "🔵" if row['exit_threshold'] == 0.75 else "  "
    print(f"{row['exit_threshold']:.2f}   {row['total_trades']:4d}   {row['win_rate']:5.1f}%  {row['avg_return']:+7.2f}%  {row['ml_exit_pct']:5.1f}%   {row['avg_hold']:5.1f}   {row['long_trades']:3d}/{row['short_trades']:3d} {marker}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = RESULTS_DIR / f"latest4weeks_exit_comparison_{timestamp}.csv"
df_results.to_csv(output_file, index=False)

print()
print(f"✓ Results saved: {output_file.name}")
print()

# Summary
print("=" * 80)
print("COMPARISON WITH OLD DATA (108 WINDOWS)")
print("=" * 80)
print()
print("Exit   Old Data (108w)        Latest 4 Weeks")
print("─" * 70)
print("0.15   WR 49.9%, ML 87.1%     WR {:.1f}%, ML {:.1f}%".format(
    df_results[df_results['exit_threshold'] == 0.15]['win_rate'].values[0],
    df_results[df_results['exit_threshold'] == 0.15]['ml_exit_pct'].values[0]
) if 0.15 in df_results['exit_threshold'].values else "0.15   WR 49.9%, ML 87.1%     (not tested)")

print("0.20   WR 73.2%, ML 64.8%     WR {:.1f}%, ML {:.1f}%".format(
    df_results[df_results['exit_threshold'] == 0.20]['win_rate'].values[0],
    df_results[df_results['exit_threshold'] == 0.20]['ml_exit_pct'].values[0]
) if 0.20 in df_results['exit_threshold'].values else "0.20   WR 73.2%, ML 64.8%     (not tested)")

print("0.75   WR 81.5%, ML  0.0% 🔵   WR {:.1f}%, ML {:.1f}% 🔵".format(
    df_results[df_results['exit_threshold'] == 0.75]['win_rate'].values[0],
    df_results[df_results['exit_threshold'] == 0.75]['ml_exit_pct'].values[0]
) if 0.75 in df_results['exit_threshold'].values else "0.75   WR 81.5%, ML  0.0% 🔵   (not tested)")

print()
print("Key Insight:")
print("- If fresh data shows similar patterns → Old findings validated")
print("- If fresh data shows different patterns → Market regime changed")
print("=" * 80)
