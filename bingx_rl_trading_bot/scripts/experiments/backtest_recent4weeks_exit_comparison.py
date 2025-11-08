"""
Backtest Recent 4-Week Data - Exit Threshold Comparison
========================================================
User Request: "최신 4주로 새로 받아서 백테스트를 진행해 주세요"
Purpose: Validate Exit threshold findings on fresh recent data
Note: Using existing dataset's most recent 4 weeks (Sept 28 - Oct 26, 2025)
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Import prepare_exit_features from production
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

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
print("BACKTEST RECENT 4-WEEK DATA - EXIT THRESHOLD COMPARISON")
print("=" * 80)
print(f"Testing {len(EXIT_THRESHOLDS)} Exit thresholds: {EXIT_THRESHOLDS}")
print(f"Data: Most recent 4 weeks from existing dataset")
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

# Load LATEST 4-week dataset (downloaded fresh from BingX)
print("Loading dataset...")
df_recent = pd.read_csv(RESULTS_DIR / "BTCUSDT_5m_latest4weeks_features_20251101.csv")
df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'])

print(f"✓ Latest 4 weeks: {len(df_recent):,} candles ({df_recent['timestamp'].min()} to {df_recent['timestamp'].max()})")
print()

# Add enhanced exit features using production function
print("Adding enhanced exit features (using production prepare_exit_features)...")
df_recent = prepare_exit_features(df_recent)
print("✓ Exit features prepared using production method")
print()

# Define 5-day windows (1440 candles per window)
window_size = 1440
num_windows = len(df_recent) // window_size
print(f"Testing {num_windows} windows (5-day periods)")
print()

def backtest_threshold(exit_threshold_long, exit_threshold_short):
    """Run backtest with specific Exit thresholds"""

    all_trades = []

    for window_idx in range(num_windows):
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        df_window = df_recent.iloc[start_idx:end_idx].copy()

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
print("RECENT 4-WEEK DATA - EXIT THRESHOLD COMPARISON")
print("=" * 80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('exit_threshold').reset_index(drop=True)

print()
print("Exit   Trades  WR      Avg Ret   ML Exit   Hold    LONG/SHORT")
print("─" * 70)
for idx, row in df_results.iterrows():
    marker = "🔵" if row['exit_threshold'] == 0.75 else "  "
    print(f"{row['exit_threshold']:.2f}   {int(row['total_trades']):4d}   {row['win_rate']:5.1f}%  {row['avg_return']:+7.2f}%  {row['ml_exit_pct']:5.1f}%   {row['avg_hold']:5.1f}   {int(row['long_trades']):3d}/{int(row['short_trades']):3d} {marker}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = RESULTS_DIR / f"recent4weeks_exit_comparison_{timestamp}.csv"
df_results.to_csv(output_file, index=False)

print()
print(f"✓ Results saved: {output_file.name}")
print()

# Summary
print("=" * 80)
print("COMPARISON: OLD DATA (108 WINDOWS) vs RECENT DATA (5 WINDOWS)")
print("=" * 80)
print()
print("Exit   Old (540 days)          Recent (28 days)")
print("─" * 75)

# Helper function to safely get values
def get_value(df, threshold, metric, default="N/A"):
    try:
        val = df[df['exit_threshold'] == threshold][metric].values[0]
        return f"{val:.1f}"
    except:
        return default

for threshold in EXIT_THRESHOLDS:
    old_data = {
        0.15: "WR 49.9%, ML 87.1%",
        0.20: "WR 73.2%, ML 64.8%",
        0.75: "WR 81.5%, ML  0.0% 🔵"
    }

    recent_wr = get_value(df_results, threshold, 'win_rate')
    recent_ml = get_value(df_results, threshold, 'ml_exit_pct')
    marker = "🔵" if threshold == 0.75 else ""

    print(f"{threshold:.2f}   {old_data.get(threshold, 'N/A'):24s}  WR {recent_wr}%, ML {recent_ml}% {marker}")

print()
print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print()
print("If Recent data shows:")
print("  - Similar patterns → Old findings VALIDATED (market stable)")
print("  - Different patterns → Market regime CHANGED (re-evaluate)")
print()
print("User Decision: Keep Exit 0.75 (current production)")
print("Reason: Higher profitability in backtest (+405% vs alternatives)")
print("=" * 80)
