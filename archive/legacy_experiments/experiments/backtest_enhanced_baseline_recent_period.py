"""
Backtest Enhanced Baseline Models on Recent Period
===================================================

Test Enhanced Baseline models (20251024_012445) on the same period
as Strategy E, A, F tests (July 14 - October 26, 2025)

Fair comparison: Enhanced Baseline vs Alternative Strategies
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

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration
LEVERAGE = 4
INITIAL_BALANCE = 10000
TAKER_FEE = 0.0005

# Thresholds (from production config)
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75

# Emergency exits
EMERGENCY_STOP_LOSS = 0.03  # -3% of balance
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours

print("=" * 80)
print("BACKTEST - ENHANCED BASELINE MODELS (20251024_012445)")
print("=" * 80)
print()
print("Period: July 14 - October 26, 2025 (same as Strategy E, A, F tests)")
print("Models: Enhanced 5-fold CV Baseline")
print()

# Load data
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"
print("Loading data...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Load Entry models
print("Loading Entry models (20251024_012445)...")

long_entry_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
long_entry_scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
long_entry_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"

short_entry_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
short_entry_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
short_entry_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"

with open(long_entry_path, 'rb') as f:
    long_entry_model = pickle.load(f)
long_entry_scaler = joblib.load(long_entry_scaler_path)
with open(long_entry_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

with open(short_entry_path, 'rb') as f:
    short_entry_model = pickle.load(f)
short_entry_scaler = joblib.load(short_entry_scaler_path)
with open(short_entry_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

print(f"✅ LONG Entry: {len(long_entry_features)} features")
print(f"✅ SHORT Entry: {len(short_entry_features)} features")
print()

# Load Exit models (threshold 0.75)
print("Loading Exit models (threshold_075)...")

long_exit_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl"
long_exit_features_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt"

short_exit_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"
short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl"
short_exit_features_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt"

with open(long_exit_path, 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(long_exit_scaler_path)
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

with open(short_exit_path, 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(short_exit_scaler_path)
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print(f"✅ LONG Exit: {len(long_exit_features)} features")
print(f"✅ SHORT Exit: {len(short_exit_features)} features")
print()

# Dynamic position sizing function
def calculate_position_size(prob, prob_min, prob_max, min_size, max_size):
    """
    Calculate position size based on signal probability
    prob_min -> min_size, prob_max -> max_size
    """
    if prob <= prob_min:
        return min_size
    if prob >= prob_max:
        return max_size

    # Linear interpolation
    ratio = (prob - prob_min) / (prob_max - prob_min)
    position_size = min_size + ratio * (max_size - min_size)
    return position_size

# Backtesting loop
print("=" * 80)
print("RUNNING BACKTEST")
print("=" * 80)
print()

balance = INITIAL_BALANCE
position = None
trades = []
peak_balance = INITIAL_BALANCE
max_drawdown = 0

for idx in range(100, len(df)):
    if idx % 5000 == 0:
        progress = idx / len(df) * 100
        print(f"  Processing candle {idx:,}/{len(df):,} ({progress:.1f}%)")

    current_price = df['close'].iloc[idx]
    current_time = df['timestamp'].iloc[idx]

    # Check for exit if in position
    if position is not None:
        position['hold_time'] += 1

        # Calculate P&L
        if position['side'] == 'LONG':
            price_pnl = (current_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            price_pnl = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl = price_pnl * LEVERAGE

        # Calculate balance-based SL
        position_size_pct = position['position_value'] / balance
        price_sl_pct = EMERGENCY_STOP_LOSS / (position_size_pct * LEVERAGE)

        # Check emergency exits
        exit_reason = None

        # Stop Loss
        if leveraged_pnl <= -EMERGENCY_STOP_LOSS:
            exit_reason = 'stop_loss'

        # Max Hold
        elif position['hold_time'] >= EMERGENCY_MAX_HOLD_TIME:
            exit_reason = 'max_hold'

        # ML Exit
        else:
            # Prepare exit features
            if position['side'] == 'LONG':
                exit_feat_df = df[long_exit_features].iloc[idx:idx+1]
                if not exit_feat_df.isnull().values.any():
                    exit_scaled = long_exit_scaler.transform(exit_feat_df.values)
                    exit_prob = long_exit_model.predict_proba(exit_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                        exit_reason = 'ml_exit'
            else:  # SHORT
                exit_feat_df = df[short_exit_features].iloc[idx:idx+1]
                if not exit_feat_df.isnull().values.any():
                    exit_scaled = short_exit_scaler.transform(exit_feat_df.values)
                    exit_prob = short_exit_model.predict_proba(exit_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                        exit_reason = 'ml_exit'

        # Execute exit
        if exit_reason is not None:
            # Calculate fees
            fee_total = 2 * TAKER_FEE * LEVERAGE
            final_pnl_after_fees = leveraged_pnl - fee_total

            # Calculate dollar P&L
            pnl_dollars = position['position_value'] * final_pnl_after_fees

            # Update balance
            balance += pnl_dollars

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            # Record trade
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'entry_prob': position['entry_prob'],
                'hold_time': position['hold_time'],
                'pnl_pct': final_pnl_after_fees,
                'pnl_dollars': pnl_dollars,
                'balance': balance,
                'exit_reason': exit_reason,
                'position_size_pct': position['position_value'] / (balance - pnl_dollars)  # Original balance
            }
            trades.append(trade)

            # Clear position
            position = None

    # Check for entry if no position
    if position is None:
        # Calculate LONG probability
        long_feat_df = df[long_entry_features].iloc[idx:idx+1]
        if not long_feat_df.isnull().values.any():
            long_scaled = long_entry_scaler.transform(long_feat_df.values)
            long_prob = long_entry_model.predict_proba(long_scaled)[0][1]
        else:
            long_prob = 0.0

        # Calculate SHORT probability
        short_feat_df = df[short_entry_features].iloc[idx:idx+1]
        if not short_feat_df.isnull().values.any():
            short_scaled = short_entry_scaler.transform(short_feat_df.values)
            short_prob = short_entry_model.predict_proba(short_scaled)[0][1]
        else:
            short_prob = 0.0

        # Check LONG entry
        if long_prob >= LONG_THRESHOLD:
            # Dynamic position sizing (20-95%)
            position_size_pct = calculate_position_size(
                long_prob, LONG_THRESHOLD, 0.95, 0.20, 0.95
            )
            position_value = balance * position_size_pct

            # Enter LONG
            position = {
                'side': 'LONG',
                'entry_idx': idx,
                'entry_price': current_price,
                'entry_time': current_time,
                'entry_prob': long_prob,
                'position_value': position_value,
                'hold_time': 0
            }

        # Check SHORT entry (only if no LONG)
        elif short_prob >= SHORT_THRESHOLD:
            # Dynamic position sizing (20-95%)
            position_size_pct = calculate_position_size(
                short_prob, SHORT_THRESHOLD, 0.95, 0.20, 0.95
            )
            position_value = balance * position_size_pct

            # Enter SHORT
            position = {
                'side': 'SHORT',
                'entry_idx': idx,
                'entry_price': current_price,
                'entry_time': current_time,
                'entry_prob': short_prob,
                'position_value': position_value,
                'hold_time': 0
            }

print("  Processing complete!")
print()

# Results
print("=" * 80)
print("BACKTEST RESULTS - ENHANCED BASELINE (20251024_012445)")
print("=" * 80)
print()

if len(trades) == 0:
    print("⚠️ No trades executed")
    sys.exit(0)

# Convert to DataFrame
trades_df = pd.DataFrame(trades)

# Calculate metrics
total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE
win_trades = trades_df[trades_df['pnl_pct'] > 0]
loss_trades = trades_df[trades_df['pnl_pct'] <= 0]
win_rate = len(win_trades) / len(trades_df)

avg_win = win_trades['pnl_pct'].mean() if len(win_trades) > 0 else 0
avg_loss = loss_trades['pnl_pct'].mean() if len(loss_trades) > 0 else 0
avg_trade = trades_df['pnl_pct'].mean()

profit_factor = abs(win_trades['pnl_pct'].sum() / loss_trades['pnl_pct'].sum()) if len(loss_trades) > 0 and loss_trades['pnl_pct'].sum() != 0 else float('inf')

# Exit distribution
exit_counts = trades_df['exit_reason'].value_counts()

# Performance metrics
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Final Balance: ${balance:,.2f}")
print(f"Total Return: {total_return*100:+.2f}%")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")
print()

print(f"Total Trades: {len(trades_df)}")
print(f"Winning Trades: {len(win_trades)} ({len(win_trades)/len(trades_df)*100:.1f}%)")
print(f"Losing Trades: {len(loss_trades)} ({len(loss_trades)/len(trades_df)*100:.1f}%)")
print(f"Win Rate: {win_rate*100:.2f}%")
print()

print(f"Average Trade: {avg_trade*100:+.4f}%")
print(f"Average Win: {avg_win*100:+.4f}%")
print(f"Average Loss: {avg_loss*100:.4f}%")
print(f"Profit Factor: {profit_factor:.2f}x")
print()

print("Exit Distribution:")
for reason, count in exit_counts.items():
    pct = count / len(trades_df) * 100
    print(f"  {reason}: {count} ({pct:.1f}%)")
print()

# LONG vs SHORT
long_trades = trades_df[trades_df['side'] == 'LONG']
short_trades = trades_df[trades_df['side'] == 'SHORT']
print(f"LONG Trades: {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)")
print(f"SHORT Trades: {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)")
print()

# Average position size
avg_position_size = trades_df['position_size_pct'].mean()
print(f"Average Position Size: {avg_position_size*100:.1f}%")
print()

# Hold time analysis
print(f"Average Hold Time: {trades_df['hold_time'].mean():.1f} candles ({trades_df['hold_time'].mean()/12:.2f} hours)")
print(f"Max Hold Time: {trades_df['hold_time'].max()} candles")
print()

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = RESULTS_DIR / f"enhanced_baseline_recent_period_{timestamp}.csv"
trades_df.to_csv(results_file, index=False)
print(f"✅ Results saved: {results_file}")
print()

print("=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
