"""
Backtest 90-Day 5-Min Models on Validation Period
==================================================

Purpose: Validate 90-day 5-min models on out-of-sample data

Models: 20251106_170732 (Enhanced 5-Fold CV, relaxed labels 1.5% in 120min)
Period: Oct 9 - Nov 6, 2025 (28 days)
Configuration: Entry 0.85/0.80, Exit 0.75, 4x Leverage, -3% SL

Created: 2025-11-06 17:10 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Import feature calculators
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Input
FEATURES_FILE = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"

# Models (90-day 5-min)
TIMESTAMP = "20251106_170732"
LONG_ENTRY_MODEL = MODELS_DIR / f"xgboost_long_entry_90days_5min_{TIMESTAMP}.pkl"
SHORT_ENTRY_MODEL = MODELS_DIR / f"xgboost_short_entry_90days_5min_{TIMESTAMP}.pkl"
LONG_EXIT_MODEL = MODELS_DIR / f"xgboost_long_exit_90days_5min_{TIMESTAMP}.pkl"
SHORT_EXIT_MODEL = MODELS_DIR / f"xgboost_short_exit_90days_5min_{TIMESTAMP}.pkl"

LONG_ENTRY_SCALER = MODELS_DIR / f"xgboost_long_entry_90days_5min_{TIMESTAMP}_scaler.pkl"
SHORT_ENTRY_SCALER = MODELS_DIR / f"xgboost_short_entry_90days_5min_{TIMESTAMP}_scaler.pkl"
LONG_EXIT_SCALER = MODELS_DIR / f"xgboost_long_exit_90days_5min_{TIMESTAMP}_scaler.pkl"
SHORT_EXIT_SCALER = MODELS_DIR / f"xgboost_short_exit_90days_5min_{TIMESTAMP}_scaler.pkl"

LONG_ENTRY_FEATURES = MODELS_DIR / f"xgboost_long_entry_90days_5min_{TIMESTAMP}_features.txt"
SHORT_ENTRY_FEATURES = MODELS_DIR / f"xgboost_short_entry_90days_5min_{TIMESTAMP}_features.txt"
LONG_EXIT_FEATURES = MODELS_DIR / f"xgboost_long_exit_90days_5min_{TIMESTAMP}_features.txt"
SHORT_EXIT_FEATURES = MODELS_DIR / f"xgboost_short_exit_90days_5min_{TIMESTAMP}_features.txt"

# Configuration (same as production)
LONG_ENTRY_THRESHOLD = 0.85
SHORT_ENTRY_THRESHOLD = 0.80
EXIT_THRESHOLD = 0.75
STOP_LOSS_PCT = 0.03  # -3%
MAX_HOLD_CANDLES = 120  # 10 hours @ 5-min
LEVERAGE = 4
INITIAL_BALANCE = 300.0

# Validation period (out-of-sample)
VALIDATION_START = "2025-10-09"
VALIDATION_END = "2025-11-06"

print("=" * 80)
print("BACKTESTING 90-DAY 5-MIN MODELS ON VALIDATION PERIOD")
print("=" * 80)
print()
print(f"📂 Features: {FEATURES_FILE.name}")
print(f"🎯 Models: {TIMESTAMP} (Enhanced 5-Fold CV)")
print(f"📊 Validation: {VALIDATION_START} to {VALIDATION_END} (28 days)")
print(f"⚙️ Config: Entry {LONG_ENTRY_THRESHOLD}/{SHORT_ENTRY_THRESHOLD}, Exit {EXIT_THRESHOLD}, 4x Leverage")
print()

# ============================================================================
# LOAD MODELS
# ============================================================================

print("=" * 80)
print("LOADING MODELS")
print("=" * 80)
print()

with open(LONG_ENTRY_MODEL, 'rb') as f:
    model_long_entry = pickle.load(f)
print(f"✅ LONG Entry model loaded")

with open(SHORT_ENTRY_MODEL, 'rb') as f:
    model_short_entry = pickle.load(f)
print(f"✅ SHORT Entry model loaded")

with open(LONG_EXIT_MODEL, 'rb') as f:
    model_long_exit = pickle.load(f)
print(f"✅ LONG Exit model loaded")

with open(SHORT_EXIT_MODEL, 'rb') as f:
    model_short_exit = pickle.load(f)
print(f"✅ SHORT Exit model loaded")
print()

# Load scalers
with open(LONG_ENTRY_SCALER, 'rb') as f:
    scaler_long_entry = pickle.load(f)
with open(SHORT_ENTRY_SCALER, 'rb') as f:
    scaler_short_entry = pickle.load(f)
with open(LONG_EXIT_SCALER, 'rb') as f:
    scaler_long_exit = pickle.load(f)
with open(SHORT_EXIT_SCALER, 'rb') as f:
    scaler_short_exit = pickle.load(f)
print(f"✅ All scalers loaded")
print()

# Load feature lists
with open(LONG_ENTRY_FEATURES, 'r') as f:
    long_entry_features = [line.strip() for line in f]
with open(SHORT_ENTRY_FEATURES, 'r') as f:
    short_entry_features = [line.strip() for line in f]
with open(LONG_EXIT_FEATURES, 'r') as f:
    long_exit_features = [line.strip() for line in f]
with open(SHORT_EXIT_FEATURES, 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"✅ Feature lists loaded:")
print(f"   LONG Entry: {len(long_entry_features)} features")
print(f"   SHORT Entry: {len(short_entry_features)} features")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")
print()

# ============================================================================
# LOAD AND FILTER DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print()

print("📖 Loading features...")
df = pd.read_csv(FEATURES_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter to validation period
df = df[(df['timestamp'] >= VALIDATION_START) & (df['timestamp'] <= VALIDATION_END)].copy()

print(f"✅ Validation data loaded:")
print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Rows: {len(df):,}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# ============================================================================
# BACKTEST
# ============================================================================

print("=" * 80)
print("RUNNING BACKTEST")
print("=" * 80)
print()

balance = INITIAL_BALANCE
position = None
trades = []

for i in range(len(df)):
    row = df.iloc[i]
    timestamp = row['timestamp']
    price = row['close']

    # Skip if missing features
    if not all(f in df.columns for f in long_entry_features + short_entry_features):
        continue

    # ========================================================================
    # EXIT LOGIC (if in position)
    # ========================================================================

    if position is not None:
        side = position['side']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        hold_candles = i - position['entry_idx']

        # Calculate unrealized P&L
        if side == 'LONG':
            pnl_pct = (price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - price) / entry_price

        leveraged_pnl_pct = pnl_pct * LEVERAGE

        # Stop Loss check (balance-based -3%)
        if leveraged_pnl_pct <= -STOP_LOSS_PCT:
            exit_price = price
            exit_reason = 'Stop Loss'

            trade_pnl = balance * leveraged_pnl_pct
            balance += trade_pnl

            trades.append({
                'entry_time': entry_time,
                'exit_time': timestamp,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': trade_pnl,
                'pnl_pct': leveraged_pnl_pct,
                'hold_candles': hold_candles,
                'entry_prob': position['entry_prob'],
                'exit_prob': 0.0
            })

            position = None
            continue

        # Max Hold check
        if hold_candles >= MAX_HOLD_CANDLES:
            exit_price = price
            exit_reason = 'Max Hold'

            trade_pnl = balance * leveraged_pnl_pct
            balance += trade_pnl

            trades.append({
                'entry_time': entry_time,
                'exit_time': timestamp,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': trade_pnl,
                'pnl_pct': leveraged_pnl_pct,
                'hold_candles': hold_candles,
                'entry_prob': position['entry_prob'],
                'exit_prob': 0.0
            })

            position = None
            continue

        # ML Exit check
        try:
            if side == 'LONG':
                exit_features_df = pd.DataFrame([row[long_exit_features]])
                exit_features_scaled = scaler_long_exit.transform(exit_features_df)
                exit_prob = model_long_exit.predict_proba(exit_features_scaled)[0, 1]
            else:  # SHORT
                exit_features_df = pd.DataFrame([row[short_exit_features]])
                exit_features_scaled = scaler_short_exit.transform(exit_features_df)
                exit_prob = model_short_exit.predict_proba(exit_features_scaled)[0, 1]

            if exit_prob >= EXIT_THRESHOLD:
                exit_price = price
                exit_reason = 'ML Exit'

                trade_pnl = balance * leveraged_pnl_pct
                balance += trade_pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': trade_pnl,
                    'pnl_pct': leveraged_pnl_pct,
                    'hold_candles': hold_candles,
                    'entry_prob': position['entry_prob'],
                    'exit_prob': exit_prob
                })

                position = None
                continue
        except:
            pass

    # ========================================================================
    # ENTRY LOGIC (if no position)
    # ========================================================================

    if position is None:
        try:
            # Calculate entry probabilities
            long_entry_df = pd.DataFrame([row[long_entry_features]])
            long_entry_scaled = scaler_long_entry.transform(long_entry_df)
            long_prob = model_long_entry.predict_proba(long_entry_scaled)[0, 1]

            short_entry_df = pd.DataFrame([row[short_entry_features]])
            short_entry_scaled = scaler_short_entry.transform(short_entry_df)
            short_prob = model_short_entry.predict_proba(short_entry_scaled)[0, 1]

            # Entry decision (opportunity gating)
            if long_prob >= LONG_ENTRY_THRESHOLD and long_prob > short_prob:
                position = {
                    'side': 'LONG',
                    'entry_price': price,
                    'entry_time': timestamp,
                    'entry_idx': i,
                    'entry_prob': long_prob
                }
            elif short_prob >= SHORT_ENTRY_THRESHOLD and short_prob > long_prob:
                position = {
                    'side': 'SHORT',
                    'entry_price': price,
                    'entry_time': timestamp,
                    'entry_idx': i,
                    'entry_prob': short_prob
                }
        except:
            pass

# Close any remaining position at end
if position is not None:
    side = position['side']
    entry_price = position['entry_price']
    entry_time = position['entry_time']
    hold_candles = len(df) - position['entry_idx']

    exit_price = df.iloc[-1]['close']
    exit_time = df.iloc[-1]['timestamp']

    if side == 'LONG':
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    leveraged_pnl_pct = pnl_pct * LEVERAGE
    trade_pnl = balance * leveraged_pnl_pct
    balance += trade_pnl

    trades.append({
        'entry_time': entry_time,
        'exit_time': exit_time,
        'side': side,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'exit_reason': 'End of Period',
        'pnl': trade_pnl,
        'pnl_pct': leveraged_pnl_pct,
        'hold_candles': hold_candles,
        'entry_prob': position['entry_prob'],
        'exit_prob': 0.0
    })

# ============================================================================
# RESULTS
# ============================================================================

print("=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)
print()

if len(trades) == 0:
    print("❌ NO TRADES EXECUTED")
    print()
    print("Reason: Models did not generate entry signals above threshold")
    print(f"   LONG Entry Threshold: {LONG_ENTRY_THRESHOLD} (85%)")
    print(f"   SHORT Entry Threshold: {SHORT_ENTRY_THRESHOLD} (80%)")
    print()
    print("⚠️  This indicates severe probability calibration issues")
    print("   (same as 314-day 15-min models)")
else:
    df_trades = pd.DataFrame(trades)

    total_trades = len(df_trades)
    long_trades = len(df_trades[df_trades['side'] == 'LONG'])
    short_trades = len(df_trades[df_trades['side'] == 'SHORT'])

    winning_trades = len(df_trades[df_trades['pnl'] > 0])
    losing_trades = len(df_trades[df_trades['pnl'] <= 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    total_pnl = df_trades['pnl'].sum()
    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    avg_hold = df_trades['hold_candles'].mean()
    trades_per_day = total_trades / ((df['timestamp'].max() - df['timestamp'].min()).days)

    print(f"📊 Performance:")
    print(f"   Initial Balance: ${INITIAL_BALANCE:.2f}")
    print(f"   Final Balance: ${balance:.2f}")
    print(f"   Total P&L: ${total_pnl:+.2f}")
    print(f"   Return: {total_return:+.2f}%")
    print()

    print(f"📈 Trade Statistics:")
    print(f"   Total Trades: {total_trades}")
    print(f"   LONG: {long_trades} ({long_trades/total_trades*100:.1f}%)")
    print(f"   SHORT: {short_trades} ({short_trades/total_trades*100:.1f}%)")
    print(f"   Win Rate: {win_rate:.2f}% ({winning_trades}/{total_trades})")
    print(f"   Avg Hold: {avg_hold:.1f} candles ({avg_hold*5:.0f} min)")
    print(f"   Trades/Day: {trades_per_day:.2f}")
    print()

    # Exit reasons
    exit_reasons = df_trades['exit_reason'].value_counts()
    print(f"🚪 Exit Reasons:")
    for reason, count in exit_reasons.items():
        print(f"   {reason}: {count} ({count/total_trades*100:.1f}%)")
    print()

    # Save trades
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"backtest_90days_5min_{timestamp_str}.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"💾 Trades saved: {output_file.name}")
    print()

print("=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
print()
