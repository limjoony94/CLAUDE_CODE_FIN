#!/usr/bin/env python3
"""
72-Hour Recent Backtest: Entry 0.80 + Exit 0.80
===============================================

Quick backtest on most recent 72 hours (864 candles) of data
to see current strategy performance on latest market conditions.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("72-HOUR RECENT BACKTEST: Entry 0.80 + Exit 0.80")
print("="*80)

# Strategy Parameters (Current Production Settings)
LEVERAGE = 4
LONG_THRESHOLD = 0.80
SHORT_THRESHOLD = 0.80
GATE_THRESHOLD = 0.001
ML_EXIT_THRESHOLD_LONG = 0.80
ML_EXIT_THRESHOLD_SHORT = 0.80
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours
EMERGENCY_STOP_LOSS = -0.03  # -3% balance

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees
TAKER_FEE = 0.0005  # 0.05% per trade

# Load Entry Models
print("\nLoading models...")
long_model_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

# Load Exit Models
long_exit_model = pickle.load(open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb'))
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model = pickle.load(open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb'))
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ Models loaded")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Load data
print("\nLoading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

print(f"  Total data: {len(df_full):,} candles")
print(f"  Period: {df_full['timestamp'].min()} ~ {df_full['timestamp'].max()}")

# Get last 72 hours (864 candles)
candles_72h = 72 * 60 // 5  # 864 candles
df_72h = df_full.tail(candles_72h).copy()

print(f"\n72-Hour Period:")
print(f"  Start: {df_72h['timestamp'].min()}")
print(f"  End: {df_72h['timestamp'].max()}")
print(f"  Candles: {len(df_72h)}")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features_enhanced_v2(df_72h, phase='phase1')
df = prepare_exit_features(df)
print(f"  ✅ Features calculated: {len(df)} rows")

# Pre-calculate signals
print("\nPre-calculating signals...")
long_feat = df[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
df['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

short_feat = df[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
df['short_prob'] = short_model.predict_proba(short_feat_scaled)[:, 1]
print(f"  ✅ Signals calculated")

# Run backtest
print("\n" + "="*80)
print("RUNNING BACKTEST")
print("="*80)

trades = []
position = None
capital = INITIAL_CAPITAL
first_exit_signal_received = False

for i in range(len(df) - 1):
    current_price = df['close'].iloc[i]

    # Check for first exit signal
    if not first_exit_signal_received and position is None:
        try:
            exit_features = df[long_exit_feature_columns].iloc[i:i+1].values
            exit_scaled = long_exit_scaler.transform(exit_features)
            long_exit_prob = long_exit_model.predict_proba(exit_scaled)[0][1]

            exit_features = df[short_exit_feature_columns].iloc[i:i+1].values
            exit_scaled = short_exit_scaler.transform(exit_features)
            short_exit_prob = short_exit_model.predict_proba(exit_scaled)[0][1]

            if long_exit_prob >= ML_EXIT_THRESHOLD_LONG or short_exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                first_exit_signal_received = True
        except:
            pass

    # Entry logic
    if first_exit_signal_received and position is None:
        long_prob = df['long_prob'].iloc[i]
        short_prob = df['short_prob'].iloc[i]

        # LONG entry
        if long_prob >= LONG_THRESHOLD:
            sizing_result = sizer.get_position_size_simple(
                capital=capital,
                signal_strength=long_prob,
                leverage=LEVERAGE
            )

            position = {
                'side': 'LONG',
                'entry_idx': i,
                'entry_price': current_price,
                'position_size_pct': sizing_result['position_size_pct'],
                'leveraged_value': sizing_result['leveraged_value'],
                'quantity': sizing_result['leveraged_value'] / current_price
            }

        # SHORT entry (gated)
        elif short_prob >= SHORT_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN

            if (short_ev - long_ev) > GATE_THRESHOLD:
                sizing_result = sizer.get_position_size_simple(
                    capital=capital,
                    signal_strength=short_prob,
                    leverage=LEVERAGE
                )

                position = {
                    'side': 'SHORT',
                    'entry_idx': i,
                    'entry_price': current_price,
                    'position_size_pct': sizing_result['position_size_pct'],
                    'leveraged_value': sizing_result['leveraged_value'],
                    'quantity': sizing_result['leveraged_value'] / current_price
                }

    # Exit logic
    if position is not None:
        time_in_pos = i - position['entry_idx']
        current_price = df['close'].iloc[i]

        # Calculate P&L
        entry_notional = position['quantity'] * position['entry_price']
        current_notional = position['quantity'] * current_price

        if position['side'] == 'LONG':
            pnl_usd = current_notional - entry_notional
            price_change_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_usd = entry_notional - current_notional
            price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = price_change_pct * LEVERAGE
        balance_loss_pct = pnl_usd / capital

        # Exit conditions
        should_exit = False
        exit_reason = None

        # ML Exit
        try:
            if position['side'] == 'LONG':
                exit_model = long_exit_model
                exit_scaler = long_exit_scaler
                exit_features_list = long_exit_feature_columns
                ml_threshold = ML_EXIT_THRESHOLD_LONG
            else:
                exit_model = short_exit_model
                exit_scaler = short_exit_scaler
                exit_features_list = short_exit_feature_columns
                ml_threshold = ML_EXIT_THRESHOLD_SHORT

            exit_features_values = df[exit_features_list].iloc[i:i+1].values
            exit_features_scaled = exit_scaler.transform(exit_features_values)
            exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

            if exit_prob >= ml_threshold:
                should_exit = True
                exit_reason = f'ml_exit_{position["side"].lower()}'
        except:
            pass

        # Emergency Stop Loss
        if not should_exit and balance_loss_pct <= EMERGENCY_STOP_LOSS:
            should_exit = True
            exit_reason = 'emergency_stop_loss'

        # Emergency Max Hold
        if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
            should_exit = True
            exit_reason = 'emergency_max_hold'

        if should_exit:
            # Calculate commissions
            entry_commission = position['leveraged_value'] * TAKER_FEE
            exit_commission = position['quantity'] * current_price * TAKER_FEE
            total_commission = entry_commission + exit_commission

            # Net P&L
            net_pnl_usd = pnl_usd - total_commission
            capital += net_pnl_usd

            # Record trade
            trade = {
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl_pct': price_change_pct,
                'leveraged_pnl_pct': leveraged_pnl_pct,
                'pnl_usd': pnl_usd,
                'commission_usd': total_commission,
                'net_pnl_usd': net_pnl_usd,
                'hold_time': time_in_pos,
                'exit_reason': exit_reason,
                'position_size_pct': position['position_size_pct']
            }

            trades.append(trade)
            position = None

# Results
print("\n" + "="*80)
print("72-HOUR BACKTEST RESULTS")
print("="*80)

if len(trades) > 0:
    trades_df = pd.DataFrame(trades)

    total_return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades) * 100

    long_trades = trades_df[trades_df['side'] == 'LONG']
    short_trades = trades_df[trades_df['side'] == 'SHORT']

    print(f"\nPerformance:")
    print(f"  Return: {total_return_pct:+.2f}%")
    print(f"  Final Capital: ${capital:,.2f} (started with ${INITIAL_CAPITAL:,.2f})")
    print(f"  Win Rate: {win_rate:.1f}%")

    print(f"\nTrades:")
    print(f"  Total: {len(trades)}")
    print(f"  LONG: {len(long_trades)} ({len(long_trades)/len(trades)*100:.1f}%)")
    print(f"  SHORT: {len(short_trades)} ({len(short_trades)/len(trades)*100:.1f}%)")
    print(f"  Trades/day: {len(trades)/3:.1f}")

    print(f"\nTrade Metrics:")
    print(f"  Avg Return: {trades_df['leveraged_pnl_pct'].mean()*100:+.2f}%")
    print(f"  Best Trade: {trades_df['leveraged_pnl_pct'].max()*100:+.2f}%")
    print(f"  Worst Trade: {trades_df['leveraged_pnl_pct'].min()*100:+.2f}%")
    print(f"  Avg Position Size: {trades_df['position_size_pct'].mean()*100:.1f}%")
    print(f"  Avg Hold Time: {trades_df['hold_time'].mean():.1f} candles ({trades_df['hold_time'].mean()*5/60:.1f}h)")

    print(f"\nExit Reasons:")
    exit_counts = trades_df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        pct = count / len(trades) * 100
        reason_name = {
            'ml_exit_long': 'ML Exit (LONG)',
            'ml_exit_short': 'ML Exit (SHORT)',
            'emergency_stop_loss': 'Stop Loss',
            'emergency_max_hold': 'Max Hold'
        }.get(reason, reason)
        print(f"  {reason_name:<20s}: {count:>3d} ({pct:>5.1f}%)")

    print(f"\nCommissions:")
    print(f"  Total: ${trades_df['commission_usd'].sum():.2f}")
    print(f"  Avg per trade: ${trades_df['commission_usd'].mean():.2f}")

    # Daily extrapolation
    days_in_period = 3
    daily_return = total_return_pct / days_in_period
    print(f"\nDaily Extrapolation:")
    print(f"  Daily Return: {daily_return:+.2f}%")
    print(f"  Monthly (30d): {daily_return * 30:+.2f}%")

else:
    print("\n❌ No trades executed")

print("\n" + "="*80)
