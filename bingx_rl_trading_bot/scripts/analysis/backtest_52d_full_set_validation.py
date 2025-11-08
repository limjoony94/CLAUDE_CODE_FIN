"""
Backtest 52-Day Model Set (FULL SET - LONG + SHORT both 52-day)
================================================================

Purpose: Validate 52-day models as complete set for fair comparison

Models:
  LONG Entry: xgboost_long_entry_52day_20251106_140955.pkl
  SHORT Entry: xgboost_short_entry_52day_20251106_140955.pkl
  LONG Exit: xgboost_long_exit_52day_20251106_140955.pkl
  SHORT Exit: xgboost_short_exit_52day_20251106_140955.pkl

Validation Period: Sep 29 - Oct 26, 2025 (same as 90-day comparison)

NOTE: 52-day LONG Entry max probability is 81.57% (below 85% threshold)
      This means 52-day full set may generate 0 LONG signals

Configuration:
  Entry: LONG >= 0.85, SHORT >= 0.80
  Exit: 0.75
  Stop Loss: -3% balance
  Max Hold: 120 candles
  Leverage: 4x

Created: 2025-11-06 22:20 KST
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
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models (52-day Entry + Exit)
MODEL_LONG = MODELS_DIR / "xgboost_long_entry_52day_20251106_140955.pkl"
MODEL_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955.pkl"
MODEL_LONG_EXIT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955.pkl"
MODEL_SHORT_EXIT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955.pkl"

SCALER_LONG = MODELS_DIR / "xgboost_long_entry_52day_20251106_140955_scaler.pkl"
SCALER_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955_scaler.pkl"
SCALER_LONG_EXIT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955_scaler.pkl"
SCALER_SHORT_EXIT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955_scaler.pkl"

FEATURES_LONG = MODELS_DIR / "xgboost_long_entry_52day_20251106_140955_features.txt"
FEATURES_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955_features.txt"
FEATURES_LONG_EXIT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955_features.txt"
FEATURES_SHORT_EXIT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955_features.txt"

# Data (Use 90-day features as Sep 29 - Oct 26 is included)
FEATURES_DATA = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"

# Configuration
VAL_START = "2025-09-29"
VAL_END = "2025-10-26"
ENTRY_THRESHOLD_LONG = 0.85
ENTRY_THRESHOLD_SHORT = 0.80
EXIT_THRESHOLD = 0.75
MAX_HOLD_CANDLES = 120
STOP_LOSS_PCT = 0.03
LEVERAGE = 4
INITIAL_BALANCE = 300.0

print("=" * 80)
print("BACKTEST: 52-DAY MODEL SET (FULL SET)")
print("=" * 80)
print()
print(f"📊 Configuration:")
print(f"   Entry Models: 52-day (20251106_140955)")
print(f"   Exit Models: 52-day (20251106_140955)")
print(f"   Validation: {VAL_START} to {VAL_END}")
print(f"   Entry Threshold: LONG >= {ENTRY_THRESHOLD_LONG}, SHORT >= {ENTRY_THRESHOLD_SHORT}")
print(f"   Exit Threshold: {EXIT_THRESHOLD}")
print(f"   Stop Loss: -{STOP_LOSS_PCT*100}%")
print(f"   Max Hold: {MAX_HOLD_CANDLES} candles")
print(f"   Leverage: {LEVERAGE}x")
print()
print(f"⚠️  WARNING: 52-day LONG Entry max probability is 81.57%")
print(f"   This is BELOW 85% threshold → May generate 0 LONG signals")
print()

# ============================================================================
# LOAD MODELS
# ============================================================================

print("=" * 80)
print("LOADING MODELS")
print("=" * 80)
print()

# Load models
with open(MODEL_LONG, 'rb') as f:
    model_long = pickle.load(f)
with open(MODEL_SHORT, 'rb') as f:
    model_short = pickle.load(f)
with open(MODEL_LONG_EXIT, 'rb') as f:
    model_long_exit = pickle.load(f)
with open(MODEL_SHORT_EXIT, 'rb') as f:
    model_short_exit = pickle.load(f)

# Load scalers
scaler_long = joblib.load(SCALER_LONG)
scaler_short = joblib.load(SCALER_SHORT)
scaler_long_exit = joblib.load(SCALER_LONG_EXIT)
scaler_short_exit = joblib.load(SCALER_SHORT_EXIT)

# Load feature names
with open(FEATURES_LONG, 'r') as f:
    features_long = [line.strip() for line in f if line.strip()]
with open(FEATURES_SHORT, 'r') as f:
    features_short = [line.strip() for line in f if line.strip()]
with open(FEATURES_LONG_EXIT, 'r') as f:
    features_long_exit = [line.strip() for line in f if line.strip()]
with open(FEATURES_SHORT_EXIT, 'r') as f:
    features_short_exit = [line.strip() for line in f if line.strip()]

print(f"✅ Models loaded:")
print(f"   LONG Entry: {len(features_long)} features")
print(f"   SHORT Entry: {len(features_short)} features")
print(f"   LONG Exit: {len(features_long_exit)} features")
print(f"   SHORT Exit: {len(features_short_exit)} features")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print()

df = pd.read_csv(FEATURES_DATA)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter validation period
df_val = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)].copy()

print(f"📖 Validation Period:")
print(f"   Start: {df_val['timestamp'].min()}")
print(f"   End: {df_val['timestamp'].max()}")
print(f"   Rows: {len(df_val):,}")
print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")
print()

# ============================================================================
# GENERATE PROBABILITIES
# ============================================================================

print("=" * 80)
print("GENERATING PROBABILITIES")
print("=" * 80)
print()

# LONG Entry
X_long = df_val[features_long].copy()
X_long_scaled = scaler_long.transform(X_long)
probs_long_entry = model_long.predict_proba(X_long_scaled)[:, 1]
df_val['prob_long_entry'] = probs_long_entry

# SHORT Entry
X_short = df_val[features_short].copy()
X_short_scaled = scaler_short.transform(X_short)
probs_short_entry = model_short.predict_proba(X_short_scaled)[:, 1]
df_val['prob_short_entry'] = probs_short_entry

# LONG Exit
X_long_exit = df_val[features_long_exit].copy()
X_long_exit_scaled = scaler_long_exit.transform(X_long_exit)
probs_long_exit = model_long_exit.predict_proba(X_long_exit_scaled)[:, 1]
df_val['prob_long_exit'] = probs_long_exit

# SHORT Exit
X_short_exit = df_val[features_short_exit].copy()
X_short_exit_scaled = scaler_short_exit.transform(X_short_exit)
probs_short_exit = model_short_exit.predict_proba(X_short_exit_scaled)[:, 1]
df_val['prob_short_exit'] = probs_short_exit

print(f"✅ Probabilities generated:")
print(f"   LONG Entry: {probs_long_entry.min():.4f} - {probs_long_entry.max():.4f} (mean: {probs_long_entry.mean():.4f})")
print(f"   SHORT Entry: {probs_short_entry.min():.4f} - {probs_short_entry.max():.4f} (mean: {probs_short_entry.mean():.4f})")
print(f"   LONG Exit: {probs_long_exit.min():.4f} - {probs_long_exit.max():.4f} (mean: {probs_long_exit.mean():.4f})")
print(f"   SHORT Exit: {probs_short_exit.min():.4f} - {probs_short_exit.max():.4f} (mean: {probs_short_exit.mean():.4f})")
print()

# Signal counts
long_signals = (probs_long_entry >= ENTRY_THRESHOLD_LONG).sum()
short_signals = (probs_short_entry >= ENTRY_THRESHOLD_SHORT).sum()
print(f"📊 Signal Counts:")
print(f"   LONG >= {ENTRY_THRESHOLD_LONG}: {long_signals} ({long_signals/len(df_val)*100:.2f}%)")
print(f"   SHORT >= {ENTRY_THRESHOLD_SHORT}: {short_signals} ({short_signals/len(df_val)*100:.2f}%)")
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
trade_id = 0

for idx, row in df_val.iterrows():
    current_time = row['timestamp']
    current_price = row['close']

    # Position management
    if position is not None:
        position['hold_candles'] += 1

        # Calculate P&L
        if position['side'] == 'LONG':
            price_return = (current_price - position['entry_price']) / position['entry_price']
            exit_prob = row['prob_long_exit']
        else:  # SHORT
            price_return = (position['entry_price'] - current_price) / position['entry_price']
            exit_prob = row['prob_short_exit']

        leveraged_return = price_return * LEVERAGE
        pnl = position['size'] * leveraged_return

        # Exit conditions
        exit_reason = None

        # 1. Stop Loss
        if leveraged_return <= -STOP_LOSS_PCT:
            exit_reason = "STOP_LOSS"

        # 2. ML Exit
        elif exit_prob >= EXIT_THRESHOLD:
            exit_reason = "ML_EXIT"

        # 3. Max Hold
        elif position['hold_candles'] >= MAX_HOLD_CANDLES:
            exit_reason = "MAX_HOLD"

        # Execute exit
        if exit_reason:
            balance += pnl

            trades.append({
                'trade_id': position['trade_id'],
                'side': position['side'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': current_time,
                'exit_price': current_price,
                'exit_reason': exit_reason,
                'hold_candles': position['hold_candles'],
                'pnl': pnl,
                'return_pct': leveraged_return * 100,
                'size': position['size'],
                'balance_after': balance,
                'entry_prob': position['entry_prob'],
                'exit_prob': exit_prob if exit_reason in ['ML_EXIT', 'MAX_HOLD'] else None
            })

            position = None

    # Entry logic (if no position)
    if position is None:
        long_signal = row['prob_long_entry'] >= ENTRY_THRESHOLD_LONG
        short_signal = row['prob_short_entry'] >= ENTRY_THRESHOLD_SHORT

        # Opportunity Gating
        ev_long = row['prob_long_entry']
        ev_short = row['prob_short_entry']

        if long_signal and ev_long > (ev_short + 0.001):
            # Enter LONG
            position_size = balance * 0.95
            trade_id += 1
            position = {
                'trade_id': trade_id,
                'side': 'LONG',
                'entry_time': current_time,
                'entry_price': current_price,
                'size': position_size,
                'hold_candles': 0,
                'entry_prob': row['prob_long_entry']
            }

        elif short_signal and ev_short > (ev_long + 0.001):
            # Enter SHORT
            position_size = balance * 0.95
            trade_id += 1
            position = {
                'trade_id': trade_id,
                'side': 'SHORT',
                'entry_time': current_time,
                'entry_price': current_price,
                'size': position_size,
                'hold_candles': 0,
                'entry_prob': row['prob_short_entry']
            }

# Close final position if any
if position is not None:
    current_price = df_val.iloc[-1]['close']
    current_time = df_val.iloc[-1]['timestamp']

    if position['side'] == 'LONG':
        price_return = (current_price - position['entry_price']) / position['entry_price']
        exit_prob = df_val.iloc[-1]['prob_long_exit']
    else:
        price_return = (position['entry_price'] - current_price) / position['entry_price']
        exit_prob = df_val.iloc[-1]['prob_short_exit']

    leveraged_return = price_return * LEVERAGE
    pnl = position['size'] * leveraged_return
    balance += pnl

    trades.append({
        'trade_id': position['trade_id'],
        'side': position['side'],
        'entry_time': position['entry_time'],
        'entry_price': position['entry_price'],
        'exit_time': current_time,
        'exit_price': current_price,
        'exit_reason': 'END_OF_PERIOD',
        'hold_candles': position['hold_candles'],
        'pnl': pnl,
        'return_pct': leveraged_return * 100,
        'size': position['size'],
        'balance_after': balance,
        'entry_prob': position['entry_prob'],
        'exit_prob': exit_prob
    })

# ============================================================================
# RESULTS
# ============================================================================

print("=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)
print()

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(winning_trades) / len(trades_df) * 100

    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')

    print(f"💰 P&L Summary:")
    print(f"   Starting Balance: ${INITIAL_BALANCE:.2f}")
    print(f"   Ending Balance: ${balance:.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Total P&L: ${balance - INITIAL_BALANCE:+.2f}")
    print()

    print(f"📊 Trade Statistics:")
    print(f"   Total Trades: {len(trades_df)}")
    print(f"   Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
    print(f"   Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
    print(f"   Win Rate: {win_rate:.2f}%")
    print(f"   Profit Factor: {profit_factor:.2f}×")
    print()

    print(f"💵 Average Performance:")
    print(f"   Avg Win: ${avg_win:+.2f}")
    print(f"   Avg Loss: ${avg_loss:+.2f}")
    print(f"   Risk-Reward: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "   Risk-Reward: N/A")
    print()

    # Direction breakdown
    long_trades = trades_df[trades_df['side'] == 'LONG']
    short_trades = trades_df[trades_df['side'] == 'SHORT']

    print(f"📈 LONG Trades: {len(long_trades)}")
    if len(long_trades) > 0:
        long_wr = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100
        long_pnl = long_trades['pnl'].sum()
        print(f"   Win Rate: {long_wr:.2f}%")
        print(f"   Total P&L: ${long_pnl:+.2f}")
    print()

    print(f"📉 SHORT Trades: {len(short_trades)}")
    if len(short_trades) > 0:
        short_wr = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100
        short_pnl = short_trades['pnl'].sum()
        print(f"   Win Rate: {short_wr:.2f}%")
        print(f"   Total P&L: ${short_pnl:+.2f}")
    print()

    # Exit mechanism breakdown
    print(f"🚪 Exit Mechanisms:")
    for reason in trades_df['exit_reason'].unique():
        count = len(trades_df[trades_df['exit_reason'] == reason])
        pct = count / len(trades_df) * 100
        print(f"   {reason}: {count} ({pct:.1f}%)")
    print()

    # Save results
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"backtest_52d_full_set_{timestamp_str}.csv"
    trades_df.to_csv(output_file, index=False)

    print(f"💾 Results saved: {output_file.name}")
    print()

else:
    print("❌ No trades executed")
    print(f"   Reason: 52-day LONG Entry max ({probs_long_entry.max():.4f}) < threshold ({ENTRY_THRESHOLD_LONG})")
    print(f"   Result: 0 LONG signals, SHORT-only trading not allowed in this configuration")
    print()

print("=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
print()
