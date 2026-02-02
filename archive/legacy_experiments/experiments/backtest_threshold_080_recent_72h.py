"""
Threshold 0.80 Production Backtest: Latest 72 Hours
====================================================

Tests CURRENT production configuration (Threshold 0.80) on the most recent 72 hours.
Configuration matches opportunity_gating_bot_4x.py (deployed 2025-10-30).
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
print("THRESHOLD 0.80 BACKTEST - LATEST 72 HOURS")
print("="*80)
print(f"Current Production Configuration (Deployed 2025-10-30)\n")

# =============================================================================
# PRODUCTION CONFIGURATION (Threshold 0.80 - deployed 2025-10-30)
# =============================================================================

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.80  # Threshold 0.80
SHORT_THRESHOLD = 0.80  # Threshold 0.80
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.80  # Threshold 0.80
ML_EXIT_THRESHOLD_SHORT = 0.80  # Threshold 0.80
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours (120 candles × 5min)
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Backtest Parameters
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005  # 0.05% per trade

# =============================================================================
# LOAD PRODUCTION MODELS (THRESHOLD 0.80)
# =============================================================================

print("Loading Threshold 0.80 Entry models (Walk-Forward - Oct 27)...")
long_model_path = MODELS_DIR / "xgboost_long_entry_walkforward_080_20251027_235741.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_entry_walkforward_080_20251027_235741_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_entry_walkforward_080_20251027_235741_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_entry_walkforward_080_20251027_235741.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_walkforward_080_20251027_235741_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_entry_walkforward_080_20251027_235741_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ LONG Entry: {len(long_feature_columns)} features")
print(f"  ✅ SHORT Entry: {len(short_feature_columns)} features\n")

# Load Exit Models
print("Loading Exit models (threshold_075 - Oct 27, used with 0.80 threshold)...")
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ LONG Exit: {len(long_exit_feature_columns)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_feature_columns)} features\n")

# Initialize Position Sizer
sizer = DynamicPositionSizer()

# =============================================================================
# LOAD MOST RECENT 72 HOURS DATA
# =============================================================================

print("Loading most recent 72 hours of data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Get last 72 hours (72 * 60 / 5 = 864 candles)
num_candles_72h = 864
df_72h = df.tail(num_candles_72h).copy()

print(f"  Period: {df_72h['timestamp'].iloc[0]} to {df_72h['timestamp'].iloc[-1]}")
print(f"  Total Candles: {len(df_72h)}")
print(f"  Duration: {(df_72h['timestamp'].iloc[-1] - df_72h['timestamp'].iloc[0]).total_seconds() / 3600:.1f} hours\n")

# =============================================================================
# CALCULATE FEATURES
# =============================================================================

print("Calculating features...")
df_features = calculate_all_features_enhanced_v2(df_72h.copy())

# Remove NaN rows
df_features = df_features.dropna().reset_index(drop=True)
print(f"  Valid Candles (after NaN removal): {len(df_features)}\n")

# =============================================================================
# BACKTEST
# =============================================================================

print("="*80)
print("RUNNING BACKTEST")
print("="*80)

capital = INITIAL_CAPITAL
position = None
trades = []
candle_states = []

for idx in range(len(df_features)):
    candle = df_features.iloc[idx]
    timestamp = candle['timestamp']
    close_price = candle['close']

    # Calculate Entry signals
    long_feat = candle[long_feature_columns].values.reshape(1, -1)
    long_feat_scaled = long_scaler.transform(long_feat)
    long_prob = long_model.predict_proba(long_feat_scaled)[0, 1]

    short_feat = candle[short_feature_columns].values.reshape(1, -1)
    short_feat_scaled = short_scaler.transform(short_feat)
    short_prob = short_model.predict_proba(short_feat_scaled)[0, 1]

    # Position Management
    if position is None:
        # Entry Logic
        if long_prob >= LONG_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            if opportunity_cost <= GATE_THRESHOLD:
                # Enter LONG
                sizing_result = sizer.get_position_size_simple(
                    capital=capital,
                    signal_strength=long_prob,
                    leverage=LEVERAGE
                )
                position_size_usd = sizing_result['position_value']
                quantity = position_size_usd / close_price
                leveraged_value = sizing_result['leveraged_value']

                position = {
                    'type': 'LONG',
                    'entry_price': close_price,
                    'entry_time': timestamp,
                    'entry_idx': idx,
                    'quantity': quantity,
                    'position_size_pct': sizing_result['position_size_pct'],
                    'leveraged_value': leveraged_value,
                    'entry_prob': long_prob
                }

        elif short_prob >= SHORT_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > GATE_THRESHOLD:
                # Enter SHORT
                sizing_result = sizer.get_position_size_simple(
                    capital=capital,
                    signal_strength=short_prob,
                    leverage=LEVERAGE
                )
                position_size_usd = sizing_result['position_value']
                quantity = position_size_usd / close_price
                leveraged_value = sizing_result['leveraged_value']

                position = {
                    'type': 'SHORT',
                    'entry_price': close_price,
                    'entry_time': timestamp,
                    'entry_idx': idx,
                    'quantity': quantity,
                    'position_size_pct': sizing_result['position_size_pct'],
                    'leveraged_value': leveraged_value,
                    'entry_prob': short_prob
                }

    else:
        # Exit Logic
        hold_time = idx - position['entry_idx']
        entry_price = position['entry_price']

        # Calculate P&L
        if position['type'] == 'LONG':
            price_change_pct = (close_price - entry_price) / entry_price
        else:  # SHORT
            price_change_pct = (entry_price - close_price) / entry_price

        leveraged_pnl_pct = price_change_pct * LEVERAGE

        # Calculate balance-based SL price
        price_sl_pct = abs(EMERGENCY_STOP_LOSS) / (position['position_size_pct'] * LEVERAGE)

        # Check Exit Conditions
        exit_reason = None

        # 1. ML Exit
        df_exit = pd.DataFrame([candle])
        df_exit = prepare_exit_features(df_exit)

        if position['type'] == 'LONG':
            exit_feat = df_exit[long_exit_feature_columns].values
            exit_feat_scaled = long_exit_scaler.transform(exit_feat)
            exit_prob = long_exit_model.predict_proba(exit_feat_scaled)[0, 1]

            if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                exit_reason = "ML_EXIT"
        else:
            exit_feat = df_exit[short_exit_feature_columns].values
            exit_feat_scaled = short_exit_scaler.transform(exit_feat)
            exit_prob = short_exit_model.predict_proba(exit_feat_scaled)[0, 1]

            if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                exit_reason = "ML_EXIT"

        # 2. Stop Loss
        if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
            exit_reason = "STOP_LOSS"

        # 3. Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD_TIME:
            exit_reason = "MAX_HOLD"

        # Execute Exit
        if exit_reason:
            entry_fee = position['leveraged_value'] * TAKER_FEE
            exit_value = position['leveraged_value'] * (1 + leveraged_pnl_pct)
            exit_fee = exit_value * TAKER_FEE

            net_pnl_usd = (exit_value - position['leveraged_value']) - entry_fee - exit_fee
            net_pnl_pct = net_pnl_usd / capital

            new_capital = capital + net_pnl_usd

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'type': position['type'],
                'entry_price': entry_price,
                'exit_price': close_price,
                'quantity': position['quantity'],
                'hold_time': hold_time,
                'leveraged_pnl_pct': leveraged_pnl_pct * 100,
                'net_pnl_usd': net_pnl_usd,
                'net_pnl_pct': net_pnl_pct * 100,
                'capital_before': capital,
                'capital_after': new_capital,
                'exit_reason': exit_reason,
                'entry_prob': position['entry_prob'],
                'position_size_pct': position['position_size_pct'] * 100
            })

            capital = new_capital
            position = None

# Close any remaining position at the end
if position is not None:
    close_price = df_features.iloc[-1]['close']
    timestamp = df_features.iloc[-1]['timestamp']
    hold_time = len(df_features) - 1 - position['entry_idx']
    entry_price = position['entry_price']

    if position['type'] == 'LONG':
        price_change_pct = (close_price - entry_price) / entry_price
    else:
        price_change_pct = (entry_price - close_price) / entry_price

    leveraged_pnl_pct = price_change_pct * LEVERAGE

    entry_fee = position['leveraged_value'] * TAKER_FEE
    exit_value = position['leveraged_value'] * (1 + leveraged_pnl_pct)
    exit_fee = exit_value * TAKER_FEE

    net_pnl_usd = (exit_value - position['leveraged_value']) - entry_fee - exit_fee
    net_pnl_pct = net_pnl_usd / capital

    new_capital = capital + net_pnl_usd

    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': timestamp,
        'type': position['type'],
        'entry_price': entry_price,
        'exit_price': close_price,
        'quantity': position['quantity'],
        'hold_time': hold_time,
        'leveraged_pnl_pct': leveraged_pnl_pct * 100,
        'net_pnl_usd': net_pnl_usd,
        'net_pnl_pct': net_pnl_pct * 100,
        'capital_before': capital,
        'capital_after': new_capital,
        'exit_reason': 'END_OF_BACKTEST',
        'entry_prob': position['entry_prob'],
        'position_size_pct': position['position_size_pct'] * 100
    })

    capital = new_capital

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "="*80)
print("BACKTEST RESULTS - THRESHOLD 0.80 (LAST 72 HOURS)")
print("="*80)

df_trades = pd.DataFrame(trades)

print(f"\n📊 Overall Performance:")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"  Final Capital: ${capital:,.2f}")
print(f"  Total Return: {((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100):+.2f}%")
print(f"  Net P&L: ${capital - INITIAL_CAPITAL:+,.2f}")

if len(trades) > 0:
    print(f"\n📈 Trading Statistics:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Trades per Day: {len(trades) / 3:.2f}")  # 72h = 3 days

    wins = df_trades[df_trades['net_pnl_usd'] > 0]
    losses = df_trades[df_trades['net_pnl_usd'] <= 0]

    print(f"  Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
    print(f"  Win Rate: {len(wins)/len(trades)*100:.2f}%")

    if len(wins) > 0:
        print(f"  Avg Win: ${wins['net_pnl_usd'].mean():+.2f} ({wins['net_pnl_pct'].mean():+.2f}%)")
    if len(losses) > 0:
        print(f"  Avg Loss: ${losses['net_pnl_usd'].mean():+.2f} ({losses['net_pnl_pct'].mean():+.2f}%)")

    print(f"\n📊 Position Distribution:")
    long_trades = df_trades[df_trades['type'] == 'LONG']
    short_trades = df_trades[df_trades['type'] == 'SHORT']
    print(f"  LONG: {len(long_trades)} ({len(long_trades)/len(trades)*100:.1f}%)")
    print(f"  SHORT: {len(short_trades)} ({len(short_trades)/len(trades)*100:.1f}%)")

    print(f"\n🚪 Exit Reasons:")
    for reason in df_trades['exit_reason'].value_counts().items():
        print(f"  {reason[0]}: {reason[1]} ({reason[1]/len(trades)*100:.1f}%)")

    print(f"\n⏱️ Hold Time Statistics:")
    print(f"  Avg Hold Time: {df_trades['hold_time'].mean():.1f} candles ({df_trades['hold_time'].mean() * 5 / 60:.2f} hours)")
    print(f"  Min: {df_trades['hold_time'].min()} candles")
    print(f"  Max: {df_trades['hold_time'].max()} candles")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"backtest_threshold_080_72h_{timestamp}.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file.name}")
else:
    print("\n⚠️ No trades executed during this period!")

print("\n" + "="*80)
