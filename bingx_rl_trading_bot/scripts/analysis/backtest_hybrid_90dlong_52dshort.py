"""
Hybrid Model Backtest: 90-Day LONG + 52-Day SHORT
==================================================

Purpose: Validate hybrid approach combining best-of-both models

Hybrid Configuration:
  LONG Entry: 90-day model (95.20% max, 112 signals on Sep 29 - Oct 26)
  SHORT Entry: 52-day model (92.70% max, 412 signals on Sep 29 - Oct 26)
  Exit: 52-day models (proven effective)

Expected Performance:
  Period: Sep 29 - Oct 26, 2025 (28 days)
  Signals: ~18.7/day (112 LONG + 412 SHORT)
  Thresholds: LONG >= 0.85, SHORT >= 0.80, Exit >= 0.75

Created: 2025-11-06 21:35 KST
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

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Input
FEATURES_90D = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"
LABELS_90D = LABELS_DIR / "trade_outcome_labels_90days_20251106_193715.csv"

# Hybrid Models
MODEL_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900.pkl"  # 90-day
MODEL_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955.pkl"  # 52-day
MODEL_LONG_EXIT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955.pkl"
MODEL_SHORT_EXIT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955.pkl"

SCALER_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
SCALER_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955_scaler.pkl"
SCALER_LONG_EXIT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955_scaler.pkl"
SCALER_SHORT_EXIT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955_scaler.pkl"

FEATURES_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_features.txt"
FEATURES_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955_features.txt"
FEATURES_LONG_EXIT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955_features.txt"
FEATURES_SHORT_EXIT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955_features.txt"

# Configuration
ENTRY_THRESHOLD_LONG = 0.85
ENTRY_THRESHOLD_SHORT = 0.80
EXIT_THRESHOLD = 0.75
MAX_HOLD_CANDLES = 120  # 10 hours @ 5-min
STOP_LOSS_PCT = 0.03  # -3% balance-based
LEVERAGE = 4
STARTING_BALANCE = 300.0

# Validation Period
VAL_START = "2025-09-29"
VAL_END = "2025-10-26"

print("=" * 80)
print("HYBRID MODEL BACKTEST: 90-DAY LONG + 52-DAY SHORT")
print("=" * 80)
print()
print(f"🎯 Configuration:")
print(f"   LONG Entry: 90-day model (threshold: {ENTRY_THRESHOLD_LONG})")
print(f"   SHORT Entry: 52-day model (threshold: {ENTRY_THRESHOLD_SHORT})")
print(f"   Exit: 52-day models (threshold: {EXIT_THRESHOLD})")
print(f"   Leverage: {LEVERAGE}x")
print(f"   Stop Loss: -{STOP_LOSS_PCT*100}% balance")
print()
print(f"📅 Validation Period: {VAL_START} to {VAL_END}")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print()

print(f"📖 Loading features: {FEATURES_90D.name}")
df_features = pd.read_csv(FEATURES_90D)
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])

print(f"📖 Loading labels: {LABELS_90D.name}")
df_labels = pd.read_csv(LABELS_90D)
df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])

# Merge
df = df_features.merge(df_labels[['timestamp', 'close', 'signal_long', 'signal_short']],
                        on='timestamp', how='left', suffixes=('', '_label'))

# Filter to validation period
df_val = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)].copy()

print(f"✅ Validation data loaded:")
print(f"   Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
print(f"   Rows: {len(df_val):,}")
print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")
print()

# ============================================================================
# LOAD HYBRID MODELS
# ============================================================================

print("=" * 80)
print("LOADING HYBRID MODELS")
print("=" * 80)
print()

# Entry Models (HYBRID: 90-day LONG + 52-day SHORT)
with open(MODEL_LONG, 'rb') as f:
    model_long_entry = pickle.load(f)
with open(MODEL_SHORT, 'rb') as f:
    model_short_entry = pickle.load(f)

scaler_long_entry = joblib.load(SCALER_LONG)
scaler_short_entry = joblib.load(SCALER_SHORT)

with open(FEATURES_LONG, 'r') as f:
    features_long_entry = [line.strip() for line in f]
with open(FEATURES_SHORT, 'r') as f:
    features_short_entry = [line.strip() for line in f]

print(f"✅ Entry Models Loaded (HYBRID):")
print(f"   LONG: 90-day model ({len(features_long_entry)} features)")
print(f"   SHORT: 52-day model ({len(features_short_entry)} features)")
print()

# Exit Models (52-day)
with open(MODEL_LONG_EXIT, 'rb') as f:
    model_long_exit = pickle.load(f)
with open(MODEL_SHORT_EXIT, 'rb') as f:
    model_short_exit = pickle.load(f)

scaler_long_exit = joblib.load(SCALER_LONG_EXIT)
scaler_short_exit = joblib.load(SCALER_SHORT_EXIT)

with open(FEATURES_LONG_EXIT, 'r') as f:
    features_long_exit = [line.strip() for line in f]
with open(FEATURES_SHORT_EXIT, 'r') as f:
    features_short_exit = [line.strip() for line in f]

print(f"✅ Exit Models Loaded (52-day):")
print(f"   LONG Exit: {len(features_long_exit)} features")
print(f"   SHORT Exit: {len(features_short_exit)} features")
print()

# ============================================================================
# GENERATE ENTRY SIGNALS
# ============================================================================

print("=" * 80)
print("GENERATING ENTRY SIGNALS")
print("=" * 80)
print()

# LONG Entry (90-day model)
X_long = df_val[features_long_entry].copy()
X_long_scaled = scaler_long_entry.transform(X_long)
probs_long = model_long_entry.predict_proba(X_long_scaled)[:, 1]

# SHORT Entry (52-day model)
X_short = df_val[features_short_entry].copy()
X_short_scaled = scaler_short_entry.transform(X_short)
probs_short = model_short_entry.predict_proba(X_short_scaled)[:, 1]

df_val['prob_long_entry'] = probs_long
df_val['prob_short_entry'] = probs_short

print(f"🔵 LONG Entry Signals (90-day model, >= {ENTRY_THRESHOLD_LONG}):")
print(f"   Min: {probs_long.min():.4f} ({probs_long.min()*100:.2f}%)")
print(f"   Max: {probs_long.max():.4f} ({probs_long.max()*100:.2f}%)")
print(f"   Mean: {probs_long.mean():.4f} ({probs_long.mean()*100:.2f}%)")
print(f"   Signals: {(probs_long >= ENTRY_THRESHOLD_LONG).sum()} ({(probs_long >= ENTRY_THRESHOLD_LONG).sum() / len(probs_long) * 100:.2f}%)")
print()

print(f"🔴 SHORT Entry Signals (52-day model, >= {ENTRY_THRESHOLD_SHORT}):")
print(f"   Min: {probs_short.min():.4f} ({probs_short.min()*100:.2f}%)")
print(f"   Max: {probs_short.max():.4f} ({probs_short.max()*100:.2f}%)")
print(f"   Mean: {probs_short.mean():.4f} ({probs_short.mean()*100:.2f}%)")
print(f"   Signals: {(probs_short >= ENTRY_THRESHOLD_SHORT).sum()} ({(probs_short >= ENTRY_THRESHOLD_SHORT).sum() / len(probs_short) * 100:.2f}%)")
print()

# ============================================================================
# BACKTEST TRADING
# ============================================================================

print("=" * 80)
print("BACKTESTING HYBRID STRATEGY")
print("=" * 80)
print()

balance = STARTING_BALANCE
trades = []
position = None

for i in range(len(df_val)):
    current_row = df_val.iloc[i]
    current_time = current_row['timestamp']
    current_price = current_row['close']

    # Check for entry signals
    if position is None:
        long_signal = current_row['prob_long_entry'] >= ENTRY_THRESHOLD_LONG
        short_signal = current_row['prob_short_entry'] >= ENTRY_THRESHOLD_SHORT

        # Opportunity Gating: Only enter if EV(direction) > EV(other) + 0.001
        ev_long = current_row['prob_long_entry']
        ev_short = current_row['prob_short_entry']

        if long_signal and ev_long > (ev_short + 0.001):
            # Enter LONG
            position = {
                'side': 'LONG',
                'entry_price': current_price,
                'entry_time': current_time,
                'entry_prob': current_row['prob_long_entry'],
                'hold_candles': 0,
                'entry_balance': balance
            }
        elif short_signal and ev_short > (ev_long + 0.001):
            # Enter SHORT
            position = {
                'side': 'SHORT',
                'entry_price': current_price,
                'entry_time': current_time,
                'entry_prob': current_row['prob_short_entry'],
                'hold_candles': 0,
                'entry_balance': balance
            }

    # Check for exit if in position
    if position is not None:
        position['hold_candles'] += 1

        # Calculate returns
        if position['side'] == 'LONG':
            price_return = (current_price - position['entry_price']) / position['entry_price']
            leveraged_return = price_return * LEVERAGE

            # Get exit probability
            X_exit = df_val.iloc[i:i+1][features_long_exit].copy()
            X_exit_scaled = scaler_long_exit.transform(X_exit)
            exit_prob = model_long_exit.predict_proba(X_exit_scaled)[0, 1]

        else:  # SHORT
            price_return = (position['entry_price'] - current_price) / position['entry_price']
            leveraged_return = price_return * LEVERAGE

            # Get exit probability
            X_exit = df_val.iloc[i:i+1][features_short_exit].copy()
            X_exit_scaled = scaler_short_exit.transform(X_exit)
            exit_prob = model_short_exit.predict_proba(X_exit_scaled)[0, 1]

        # Exit conditions
        exit_reason = None

        # 1. Stop Loss (-3% balance)
        if leveraged_return <= -STOP_LOSS_PCT:
            exit_reason = "STOP_LOSS"

        # 2. ML Exit
        elif exit_prob >= EXIT_THRESHOLD:
            exit_reason = "ML_EXIT"

        # 3. Max Hold
        elif position['hold_candles'] >= MAX_HOLD_CANDLES:
            exit_reason = "MAX_HOLD"

        # Execute exit
        if exit_reason is not None:
            pnl = balance * leveraged_return
            balance += pnl

            trade = {
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'entry_prob': position['entry_prob'],
                'exit_prob': exit_prob,
                'hold_candles': position['hold_candles'],
                'price_return_pct': price_return * 100,
                'leveraged_return_pct': leveraged_return * 100,
                'pnl': pnl,
                'balance': balance,
                'exit_reason': exit_reason
            }

            trades.append(trade)
            position = None

# Close any remaining position at end
if position is not None:
    current_row = df_val.iloc[-1]
    current_price = current_row['close']

    if position['side'] == 'LONG':
        price_return = (current_price - position['entry_price']) / position['entry_price']
    else:
        price_return = (position['entry_price'] - current_price) / position['entry_price']

    leveraged_return = price_return * LEVERAGE
    pnl = balance * leveraged_return
    balance += pnl

    trade = {
        'entry_time': position['entry_time'],
        'exit_time': current_row['timestamp'],
        'side': position['side'],
        'entry_price': position['entry_price'],
        'exit_price': current_price,
        'entry_prob': position['entry_prob'],
        'exit_prob': 0.0,
        'hold_candles': position['hold_candles'],
        'price_return_pct': price_return * 100,
        'leveraged_return_pct': leveraged_return * 100,
        'pnl': pnl,
        'balance': balance,
        'exit_reason': 'END_OF_DATA'
    }

    trades.append(trade)

# ============================================================================
# RESULTS
# ============================================================================

print("=" * 80)
print("HYBRID BACKTEST RESULTS")
print("=" * 80)
print()

df_trades = pd.DataFrame(trades)

if len(df_trades) > 0:
    total_return = ((balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
    winners = df_trades[df_trades['pnl'] > 0]
    losers = df_trades[df_trades['pnl'] <= 0]
    win_rate = len(winners) / len(df_trades) * 100

    print(f"📊 Performance Summary:")
    print(f"   Starting Balance: ${STARTING_BALANCE:.2f}")
    print(f"   Ending Balance: ${balance:.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Net P&L: ${balance - STARTING_BALANCE:+.2f}")
    print()

    print(f"📈 Trading Statistics:")
    print(f"   Total Trades: {len(df_trades)}")
    print(f"   Winners: {len(winners)} ({len(winners)/len(df_trades)*100:.1f}%)")
    print(f"   Losers: {len(losers)} ({len(losers)/len(df_trades)*100:.1f}%)")
    print(f"   Win Rate: {win_rate:.2f}%")
    print()

    print(f"💰 P&L Breakdown:")
    print(f"   Total Profit: ${winners['pnl'].sum():+.2f}")
    print(f"   Total Loss: ${losers['pnl'].sum():+.2f}")
    print(f"   Avg Win: ${winners['pnl'].mean():+.2f}" if len(winners) > 0 else "   Avg Win: N/A")
    print(f"   Avg Loss: ${losers['pnl'].mean():+.2f}" if len(losers) > 0 else "   Avg Loss: N/A")
    print(f"   Profit Factor: {abs(winners['pnl'].sum() / losers['pnl'].sum()):.2f}×" if len(losers) > 0 and losers['pnl'].sum() != 0 else "   Profit Factor: N/A")
    print()

    # Direction breakdown
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']

    print(f"🔵 LONG Trades: {len(long_trades)} ({len(long_trades)/len(df_trades)*100:.1f}%)")
    if len(long_trades) > 0:
        long_winners = long_trades[long_trades['pnl'] > 0]
        print(f"   Win Rate: {len(long_winners)/len(long_trades)*100:.2f}%")
        print(f"   Total P&L: ${long_trades['pnl'].sum():+.2f}")
        print(f"   Avg Hold: {long_trades['hold_candles'].mean():.1f} candles ({long_trades['hold_candles'].mean()/12:.1f} hours)")
    print()

    print(f"🔴 SHORT Trades: {len(short_trades)} ({len(short_trades)/len(df_trades)*100:.1f}%)")
    if len(short_trades) > 0:
        short_winners = short_trades[short_trades['pnl'] > 0]
        print(f"   Win Rate: {len(short_winners)/len(short_trades)*100:.2f}%")
        print(f"   Total P&L: ${short_trades['pnl'].sum():+.2f}")
        print(f"   Avg Hold: {short_trades['hold_candles'].mean():.1f} candles ({short_trades['hold_candles'].mean()/12:.1f} hours)")
    print()

    # Exit mechanism breakdown
    print(f"🚪 Exit Mechanisms:")
    for reason in df_trades['exit_reason'].unique():
        count = (df_trades['exit_reason'] == reason).sum()
        pct = count / len(df_trades) * 100
        print(f"   {reason}: {count} ({pct:.1f}%)")
    print()

    # Save trades
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"backtest_hybrid_90dlong_52dshort_{timestamp_str}.csv"
    df_trades.to_csv(output_file, index=False)

    print(f"💾 Trades saved: {output_file.name}")
    print()

else:
    print("❌ No trades executed during validation period")
    print()

print("=" * 80)
print("COMPARISON VS INDIVIDUAL MODELS")
print("=" * 80)
print()

print("📊 Expected Performance (from fair comparison):")
print()
print("   90-Day LONG Only:")
print("      Max Probability: 95.20%")
print("      Signals: 112 (1.44%)")
print()
print("   52-Day SHORT Only:")
print("      Max Probability: 92.70%")
print("      Signals: 412 (5.30%)")
print()

if len(df_trades) > 0:
    print(f"   Hybrid (90-day LONG + 52-day SHORT):")
    print(f"      LONG Signals: {len(long_trades)}")
    print(f"      SHORT Signals: {len(short_trades)}")
    print(f"      Total: {len(df_trades)}")
    print(f"      Win Rate: {win_rate:.2f}%")
    print(f"      Return: {total_return:+.2f}%")
    print()

    print(f"✅ Hybrid backtest complete!")
    print(f"   Testing period: {VAL_START} to {VAL_END}")
    print(f"   Result: {len(df_trades)} trades, {win_rate:.2f}% WR, {total_return:+.2f}% return")
else:
    print("⚠️  No trades generated - check thresholds or models")

print()
