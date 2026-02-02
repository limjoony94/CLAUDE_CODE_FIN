"""
Backtest New Models on Recent 72 Hours
========================================

Test the newly deployed models (2025-10-21) on the most recent 72 hours of data
to validate performance before restarting the production bot.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("RECENT 72H BACKTEST - NEW MODELS (2025-10-21)")
print("="*80)

# ============================================================================
# STEP 1: Load Recent Data
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading Recent Data")
print("="*80)

# Use historical data and take the last 72 hours
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"\n✅ Loaded historical data: {len(df_full):,} candles")

# Take the last 72 hours (864 candles) + 200 buffer for features
candles_needed = 864 + 200
df = df_full.tail(candles_needed).reset_index(drop=True)

print(f"✅ Using last {candles_needed} candles for test")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# ============================================================================
# STEP 2: Calculate Features
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Calculating Features")
print("="*80)

print("\nCalculating entry features...")
df = calculate_all_features(df)

print("Calculating exit features...")
df = prepare_exit_features(df)

# Drop NaN rows from feature calculation
df = df.dropna().reset_index(drop=True)
print(f"✅ {len(df):,} candles after feature calculation")

# Take only the last 72 hours (864 candles)
df = df.tail(864).reset_index(drop=True)
print(f"✅ Using last 72 hours: {len(df):,} candles")
print(f"   Test period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# ============================================================================
# STEP 3: Load New Models
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Loading New Models (2025-10-21)")
print("="*80)

# LONG Entry Model
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616.pkl"
print(f"\n📦 Loading LONG Entry Model: {long_model_path.name}")
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

# SHORT Entry Model
short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616.pkl"
print(f"\n📦 Loading SHORT Entry Model: {short_model_path.name}")
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

# Exit Models (same as production)
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
print(f"\n📦 Loading LONG Exit Model: {long_exit_model_path.name}")
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
print(f"📦 Loading SHORT Exit Model: {short_exit_model_path.name}")
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print("\n✅ All models loaded successfully")

# ============================================================================
# STEP 4: Define Position Sizing Function
# ============================================================================

def calculate_position_size(signal_strength):
    """
    Simple position sizing based on signal strength.
    Maps signal strength to 20-95% range.
    """
    # Threshold-based mapping
    if signal_strength < 0.65:
        return 20.0
    elif signal_strength >= 0.95:
        return 95.0
    else:
        # Linear interpolation between 20-95%
        return 20.0 + ((signal_strength - 0.65) / (0.95 - 0.65)) * 75.0

# ============================================================================
# STEP 5: Run Backtest
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Running 72H Backtest")
print("="*80)

# Configuration
INITIAL_BALANCE = 10000.0
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
MAX_HOLD_PERIODS = 48  # 4 hours (48 * 5min)
TAKE_PROFIT_PCT = 3.0
STOP_LOSS_PCT = -6.0  # Balance-based (-6% of balance)
MAKER_FEE = 0.0002
TAKER_FEE = 0.0004

print(f"\nConfiguration:")
print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"  Leverage: {LEVERAGE}x")
print(f"  LONG Threshold: {LONG_THRESHOLD}")
print(f"  SHORT Threshold: {SHORT_THRESHOLD}")
print(f"  Gate Threshold: {GATE_THRESHOLD}")
print(f"  Max Hold: {MAX_HOLD_PERIODS} periods (4 hours)")
print(f"  Take Profit: {TAKE_PROFIT_PCT}%")
print(f"  Stop Loss: {STOP_LOSS_PCT}% (balance-based)")

# Backtest state
balance = INITIAL_BALANCE
position = None
trades = []

print(f"\nSimulating trades...")

for i in range(len(df)):
    current_price = df.loc[i, 'close']

    # If in position, check exit
    if position is not None:
        hold_periods = i - position['entry_idx']

        # Calculate current P&L
        if position['direction'] == 'LONG':
            price_change_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
        else:  # SHORT
            price_change_pct = ((position['entry_price'] - current_price) / position['entry_price']) * 100

        leveraged_pnl_pct = price_change_pct * LEVERAGE

        # Calculate balance-based stop loss
        position_size_pct = position['position_size'] / position['balance_at_entry']
        balance_sl_pct = STOP_LOSS_PCT
        price_sl_pct = balance_sl_pct / (position_size_pct * LEVERAGE)

        # Exit conditions
        exit_reason = None

        # 1. Emergency exits
        if leveraged_pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = 'take_profit'
        elif price_change_pct <= price_sl_pct:
            exit_reason = 'stop_loss'
        elif hold_periods >= MAX_HOLD_PERIODS:
            exit_reason = 'max_hold'
        else:
            # 2. ML Exit
            if position['direction'] == 'LONG':
                exit_features = df.loc[i, long_exit_feature_columns].values.reshape(1, -1)
                exit_features_scaled = long_exit_scaler.transform(exit_features)
                exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
                if exit_prob >= 0.72:
                    exit_reason = 'ml_exit'
            else:  # SHORT
                exit_features = df.loc[i, short_exit_feature_columns].values.reshape(1, -1)
                exit_features_scaled = short_exit_scaler.transform(exit_features)
                exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]
                if exit_prob >= 0.72:
                    exit_reason = 'ml_exit'

        if exit_reason:
            # Close position
            exit_fee = position['position_size'] * TAKER_FEE
            pnl = (position['position_size'] * leveraged_pnl_pct / 100) - exit_fee - position['entry_fee']
            balance += pnl

            trade_record = {
                'entry_time': df.loc[position['entry_idx'], 'timestamp'],
                'exit_time': df.loc[i, 'timestamp'],
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'position_size': position['position_size'],
                'leverage': LEVERAGE,
                'hold_periods': hold_periods,
                'leveraged_pnl_pct': leveraged_pnl_pct,
                'pnl': pnl,
                'exit_reason': exit_reason,
                'balance_after': balance
            }
            trades.append(trade_record)
            position = None

    # If no position, check entry
    if position is None and i < len(df) - MAX_HOLD_PERIODS:
        # Generate entry signals
        long_features = df.loc[i, long_feature_columns].values.reshape(1, -1)
        long_features_scaled = long_scaler.transform(long_features)
        long_prob = long_model.predict_proba(long_features_scaled)[0][1]

        short_features = df.loc[i, short_feature_columns].values.reshape(1, -1)
        short_features_scaled = short_scaler.transform(short_features)
        short_prob = short_model.predict_proba(short_features_scaled)[0][1]

        # Entry decision
        direction = None

        if long_prob >= LONG_THRESHOLD:
            direction = 'LONG'
        elif short_prob >= SHORT_THRESHOLD:
            # Opportunity gating
            long_ev = long_prob * 0.0041
            short_ev = short_prob * 0.0047
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > GATE_THRESHOLD:
                direction = 'SHORT'

        if direction:
            # Calculate position size
            signal_strength = long_prob if direction == 'LONG' else short_prob
            position_size_pct = calculate_position_size(signal_strength)
            position_size = balance * (position_size_pct / 100)

            # Enter position
            entry_fee = position_size * MAKER_FEE

            position = {
                'entry_idx': i,
                'entry_price': current_price,
                'direction': direction,
                'position_size': position_size,
                'entry_fee': entry_fee,
                'balance_at_entry': balance,
                'signal_strength': signal_strength
            }

# ============================================================================
# STEP 6: Analyze Results
# ============================================================================

print("\n" + "="*80)
print("RESULTS - 72H BACKTEST")
print("="*80)

if len(trades) == 0:
    print("\n⚠️  No trades executed during 72-hour period")
    print("   This could indicate:")
    print("   - Market conditions below entry thresholds")
    print("   - Opportunity gating preventing entries")
    print("   - Models being more selective")
else:
    trades_df = pd.DataFrame(trades)

    # Overall stats
    total_return_pct = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(winning_trades) / len(trades_df) * 100

    print(f"\n📊 Overall Performance:")
    print(f"   Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"   Final Balance: ${balance:,.2f}")
    print(f"   Total Return: {total_return_pct:+.2f}%")
    print(f"   Total Trades: {len(trades_df)}")
    print(f"   Win Rate: {win_rate:.1f}% ({len(winning_trades)}/{len(trades_df)})")

    # Trade breakdown
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']

    print(f"\n📈 Trade Breakdown:")
    print(f"   LONG: {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)")
    if len(long_trades) > 0:
        long_wr = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100
        print(f"         Win Rate: {long_wr:.1f}%")
    print(f"   SHORT: {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)")
    if len(short_trades) > 0:
        short_wr = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100
        print(f"          Win Rate: {short_wr:.1f}%")

    # Exit reasons
    print(f"\n🚪 Exit Reasons:")
    exit_counts = trades_df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        pct = count / len(trades_df) * 100
        print(f"   {reason}: {count} ({pct:.1f}%)")

    # Best and worst trades
    print(f"\n🏆 Best Trade:")
    best = trades_df.loc[trades_df['pnl'].idxmax()]
    print(f"   {best['direction']} @ ${best['entry_price']:,.2f}")
    print(f"   P&L: ${best['pnl']:+,.2f} ({best['leveraged_pnl_pct']:+.2f}%)")
    print(f"   Exit: {best['exit_reason']}")

    print(f"\n📉 Worst Trade:")
    worst = trades_df.loc[trades_df['pnl'].idxmin()]
    print(f"   {worst['direction']} @ ${worst['entry_price']:,.2f}")
    print(f"   P&L: ${worst['pnl']:+,.2f} ({worst['leveraged_pnl_pct']:+.2f}%)")
    print(f"   Exit: {worst['exit_reason']}")

    # Recent trades
    print(f"\n📜 Most Recent Trades:")
    for idx, trade in trades_df.tail(5).iterrows():
        print(f"   {trade['entry_time']} → {trade['exit_time']}")
        print(f"   {trade['direction']}: ${trade['pnl']:+,.2f} ({trade['leveraged_pnl_pct']:+.2f}%) - {trade['exit_reason']}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

if len(trades) > 0 and win_rate >= 60.0:
    print("\n✅ Models validated on recent data!")
    print("   Ready for production restart.")
elif len(trades) > 0:
    print(f"\n⚠️  Win rate ({win_rate:.1f}%) below target (60%)")
    print("   Review recent market conditions before restart.")
else:
    print("\n⚠️  No trades during 72h period")
    print("   Models may be very selective in current market.")
