"""
Threshold Optimization on Validation Period (Oct 28 - Nov 4)
==============================================================

Grid Search:
  Entry Threshold: 0.80, 0.85, 0.90
  Exit Threshold: 0.70, 0.75, 0.80

Total combinations: 9

Models:
  NEW SHORT Entry: xgboost_short_entry_with_new_features_20251104_213043
  LONG Entry: xgboost_long_entry_enhanced_20251024_012445
  Exit: xgboost_*_exit_oppgating_improved_20251024_*

Goal: Find optimal thresholds to maximize:
  1. Total Return
  2. Profit Factor
  3. Win Rate
  4. Minimize: Fee Ratio, Trade Frequency
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from itertools import product

# Import production feature calculation
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Configuration
LEVERAGE = 4
STOP_LOSS = -0.03  # -3% balance-based
MAX_HOLD = 120  # 10 hours
FEE_RATE = 0.0005  # 0.05% per trade

# Threshold Grid
ENTRY_THRESHOLDS = [0.80, 0.85, 0.90]
EXIT_THRESHOLDS = [0.70, 0.75, 0.80]

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("THRESHOLD OPTIMIZATION - VALIDATION PERIOD")
print("="*80)
print()
print("📅 Test Period: Oct 28 - Nov 4, 2025")
print()
print("Grid Search:")
print(f"  Entry Thresholds: {ENTRY_THRESHOLDS}")
print(f"  Exit Thresholds: {EXIT_THRESHOLDS}")
print(f"  Total Combinations: {len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS)}")
print()

# ==============================================================================
# Load Data and Features (once)
# ==============================================================================

print("-"*80)
print("STEP 1: Loading Data and Calculating Features")
print("-"*80)

csv_file = sorted(DATA_DIR.glob("BTCUSDT_5m_raw_35days_*.csv"), reverse=True)[0]
df_raw = pd.read_csv(csv_file)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_raw[col] = df_raw[col].astype(float)

print(f"✅ {len(df_raw):,} raw candles")

df = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')
df = prepare_exit_features(df)
print(f"✅ {len(df):,} candles with features")

# Extract validation period
split_date = '2025-10-28'
df_val = df[df['timestamp'] >= split_date].copy().reset_index(drop=True)

print(f"\n📊 Validation Period: {len(df_val):,} candles")
print(f"   Date: {df_val['timestamp'].iloc[0]} to {df_val['timestamp'].iloc[-1]}")
print(f"   Price: ${df_val['close'].min():,.1f} - ${df_val['close'].max():,.1f}")
print()

# ==============================================================================
# Load Models (once)
# ==============================================================================

print("-"*80)
print("STEP 2: Loading Models")
print("-"*80)

# SHORT Entry (NEW)
short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

# LONG Entry (existing)
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

# Exit Models
long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print("✅ All models loaded")
print()

# ==============================================================================
# Calculate Probabilities (once)
# ==============================================================================

print("-"*80)
print("STEP 3: Calculating Probabilities")
print("-"*80)

X_long_entry = df_val[long_entry_features].values
X_long_entry_scaled = long_entry_scaler.transform(X_long_entry)
long_entry_probs = long_entry_model.predict_proba(X_long_entry_scaled)[:, 1]

X_short_entry = df_val[short_entry_features].values
X_short_entry_scaled = short_entry_scaler.transform(X_short_entry)
short_entry_probs = short_entry_model.predict_proba(X_short_entry_scaled)[:, 1]

X_long_exit = df_val[long_exit_features].values
X_long_exit_scaled = long_exit_scaler.transform(X_long_exit)
long_exit_probs = long_exit_model.predict_proba(X_long_exit_scaled)[:, 1]

X_short_exit = df_val[short_exit_features].values
X_short_exit_scaled = short_exit_scaler.transform(X_short_exit)
short_exit_probs = short_exit_model.predict_proba(X_short_exit_scaled)[:, 1]

print("✅ All probabilities calculated")
print()

# ==============================================================================
# Backtest Function
# ==============================================================================

def run_backtest(entry_thresh, exit_thresh):
    """Run backtest with given thresholds"""
    initial_balance = 100.0
    balance = initial_balance
    position = None
    trades = []

    for i in range(len(df_val)):
        current_price = df_val.iloc[i]['close']

        # Check if in position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            side = position['side']
            hold_time = i - entry_idx

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                exit_prob = long_exit_probs[i]
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                exit_prob = short_exit_probs[i]

            leveraged_pnl_pct = pnl_pct * LEVERAGE

            # Exit conditions
            exit_reason = None
            if leveraged_pnl_pct <= STOP_LOSS:
                exit_reason = "Stop Loss"
            elif exit_prob >= exit_thresh:
                exit_reason = "ML Exit"
            elif hold_time >= MAX_HOLD:
                exit_reason = "Max Hold"

            if exit_reason:
                pnl_usd = balance * leveraged_pnl_pct
                entry_fee = balance * FEE_RATE
                exit_fee = (balance + pnl_usd) * FEE_RATE
                total_fee = entry_fee + exit_fee
                pnl_net = pnl_usd - total_fee

                balance += pnl_net

                trades.append({
                    'side': side,
                    'pnl_net': pnl_net,
                    'total_fee': total_fee,
                    'pnl_gross': pnl_usd,
                    'hold_time': hold_time
                })

                position = None

        # Check for new entry
        if position is None:
            long_signal = long_entry_probs[i] >= entry_thresh
            short_signal = short_entry_probs[i] >= entry_thresh

            if long_signal or short_signal:
                long_ev = long_entry_probs[i] if long_signal else 0
                short_ev = short_entry_probs[i] if short_signal else 0

                if short_ev > long_ev + 0.001:
                    chosen_side = 'SHORT'
                elif long_signal:
                    chosen_side = 'LONG'
                else:
                    chosen_side = None

                if chosen_side:
                    position = {
                        'side': chosen_side,
                        'entry_idx': i,
                        'entry_price': current_price
                    }

    return balance, trades

# ==============================================================================
# Run Grid Search
# ==============================================================================

print("="*80)
print("RUNNING GRID SEARCH")
print("="*80)
print()

results = []

for entry_thresh, exit_thresh in product(ENTRY_THRESHOLDS, EXIT_THRESHOLDS):
    print(f"Testing Entry={entry_thresh:.2f}, Exit={exit_thresh:.2f}...", end=" ")

    balance, trades = run_backtest(entry_thresh, exit_thresh)

    if len(trades) == 0:
        print("❌ No trades")
        continue

    df_trades = pd.DataFrame(trades)

    # Calculate metrics
    total_return = (balance - 100) / 100 * 100
    num_trades = len(trades)
    wins = df_trades[df_trades['pnl_net'] > 0]
    losses = df_trades[df_trades['pnl_net'] <= 0]
    win_rate = len(wins) / num_trades * 100 if num_trades > 0 else 0
    profit_factor = wins['pnl_net'].sum() / abs(losses['pnl_net'].sum()) if len(losses) > 0 and losses['pnl_net'].sum() != 0 else float('inf')
    avg_hold = df_trades['hold_time'].mean()
    trades_per_day = num_trades / 7.5  # ~7.5 days in validation period
    total_fees = df_trades['total_fee'].sum()
    fee_ratio = total_fees / abs(df_trades['pnl_gross'].sum()) * 100 if df_trades['pnl_gross'].sum() != 0 else 0

    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']

    results.append({
        'entry_thresh': entry_thresh,
        'exit_thresh': exit_thresh,
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_hold_candles': avg_hold,
        'trades_per_day': trades_per_day,
        'fee_ratio': fee_ratio,
        'long_count': len(long_trades),
        'short_count': len(short_trades),
        'final_balance': balance
    })

    print(f"✅ Return={total_return:+.2f}%, Trades={num_trades}, WR={win_rate:.1f}%, PF={profit_factor:.2f}")

print()

# ==============================================================================
# Analyze Results
# ==============================================================================

print("="*80)
print("OPTIMIZATION RESULTS")
print("="*80)
print()

df_results = pd.DataFrame(results)

# Sort by total return
df_results = df_results.sort_values('total_return', ascending=False)

print("📊 Top 5 Configurations by Total Return:")
print()
for i, row in df_results.head(5).iterrows():
    print(f"{i+1}. Entry={row['entry_thresh']:.2f}, Exit={row['exit_thresh']:.2f}")
    print(f"   Return: {row['total_return']:+.2f}%")
    print(f"   Trades: {row['num_trades']:.0f} ({row['trades_per_day']:.1f}/day)")
    print(f"   Win Rate: {row['win_rate']:.1f}%")
    print(f"   Profit Factor: {row['profit_factor']:.2f}")
    print(f"   Avg Hold: {row['avg_hold_candles']:.1f} candles ({row['avg_hold_candles']/12:.1f}h)")
    print(f"   Fee Ratio: {row['fee_ratio']:.1f}%")
    print(f"   LONG/SHORT: {row['long_count']:.0f}/{row['short_count']:.0f}")
    print()

# Find best by different metrics
print("-"*80)
print("Best by Different Metrics:")
print("-"*80)
print()

best_return = df_results.loc[df_results['total_return'].idxmax()]
print(f"🏆 Best Return: Entry={best_return['entry_thresh']:.2f}, Exit={best_return['exit_thresh']:.2f}")
print(f"   Return: {best_return['total_return']:+.2f}%")
print()

if (df_results['profit_factor'] != float('inf')).any():
    valid_pf = df_results[df_results['profit_factor'] != float('inf')]
    best_pf = valid_pf.loc[valid_pf['profit_factor'].idxmax()]
    print(f"🏆 Best Profit Factor: Entry={best_pf['entry_thresh']:.2f}, Exit={best_pf['exit_thresh']:.2f}")
    print(f"   Profit Factor: {best_pf['profit_factor']:.2f}")
    print(f"   Return: {best_pf['total_return']:+.2f}%")
    print()

best_wr = df_results.loc[df_results['win_rate'].idxmax()]
print(f"🏆 Best Win Rate: Entry={best_wr['entry_thresh']:.2f}, Exit={best_wr['exit_thresh']:.2f}")
print(f"   Win Rate: {best_wr['win_rate']:.1f}%")
print(f"   Return: {best_wr['total_return']:+.2f}%")
print()

# Best balanced (return > 0, trades/day < 12, fee_ratio < 80%)
balanced = df_results[
    (df_results['total_return'] > 0) &
    (df_results['trades_per_day'] < 12) &
    (df_results['fee_ratio'] < 80)
]

if len(balanced) > 0:
    best_balanced = balanced.iloc[0]  # Already sorted by return
    print(f"⚖️  Best Balanced: Entry={best_balanced['entry_thresh']:.2f}, Exit={best_balanced['exit_thresh']:.2f}")
    print(f"   Return: {best_balanced['total_return']:+.2f}%")
    print(f"   Trades/Day: {best_balanced['trades_per_day']:.1f}")
    print(f"   Fee Ratio: {best_balanced['fee_ratio']:.1f}%")
    print(f"   Win Rate: {best_balanced['win_rate']:.1f}%")
    print()

# ==============================================================================
# Save Results
# ==============================================================================

print("-"*80)
print("Saving Results")
print("-"*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f"threshold_optimization_validation_{timestamp}.csv"
csv_path = RESULTS_DIR / csv_filename

df_results.to_csv(csv_path, index=False)
print(f"✅ Results saved: {csv_path}")

print()
print("="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print()

if len(balanced) > 0:
    print("🎯 RECOMMENDED THRESHOLDS:")
    print(f"   Entry: {best_balanced['entry_thresh']:.2f}")
    print(f"   Exit: {best_balanced['exit_thresh']:.2f}")
    print()
    print(f"   Expected Performance:")
    print(f"   - Return: {best_balanced['total_return']:+.2f}%")
    print(f"   - Trades/Day: {best_balanced['trades_per_day']:.1f}")
    print(f"   - Win Rate: {best_balanced['win_rate']:.1f}%")
    print(f"   - Fee Impact: {best_balanced['fee_ratio']:.1f}%")
else:
    print("⚠️  No balanced configuration found")
    print("   Using best return configuration:")
    print(f"   Entry: {best_return['entry_thresh']:.2f}")
    print(f"   Exit: {best_return['exit_thresh']:.2f}")

print()
print("="*80)
