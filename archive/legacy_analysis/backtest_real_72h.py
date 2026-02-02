#!/usr/bin/env python3
"""
Real 72-Hour Backtest: Fetch from API + Test
==============================================

Fetch last 72 hours of data from BingX API and run backtest
with current production settings (Entry 0.80 + Exit 0.80).
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import datetime, timedelta
import yaml
import pytz

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

print("="*80)
print("REAL 72-HOUR BACKTEST: API Data + Entry 0.80 + Exit 0.80")
print("="*80)

# Use exact same period as CSV data
kst = pytz.timezone('Asia/Seoul')

# CSV 데이터와 동일한 기간 사용 (2025-10-23 11:00 ~ 2025-10-26 10:55 UTC)
start_time_utc = datetime(2025, 10, 23, 11, 0, 0, tzinfo=pytz.UTC)
end_time_utc = datetime(2025, 10, 26, 10, 55, 0, tzinfo=pytz.UTC)

start_time_kst = start_time_utc.astimezone(kst)
end_time_kst = end_time_utc.astimezone(kst)

print(f"\n📅 백테스트 기간 (CSV와 동일):")
print(f"시작 (UTC): {start_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"시작 (KST): {start_time_kst.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"종료 (UTC): {end_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"종료 (KST): {end_time_kst.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize API client
print("\nInitializing API client...")
config_file = CONFIG_DIR / "api_keys.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key'],
    testnet=False  # Use mainnet for real data
)

# Fetch data for exact same period as CSV
print("\nFetching data from BingX API...")
candles_needed = 864  # Exact number of candles in CSV

# Calculate time range in milliseconds
start_time_ms = int(start_time_utc.timestamp() * 1000)
end_time_ms = int(end_time_utc.timestamp() * 1000)

print(f"  Requesting {candles_needed} candles")
print(f"  Timestamps (ms): {start_time_ms} ~ {end_time_ms}")

try:
    # BingX API max is 1440 candles per request
    klines = client.get_klines(
        symbol="BTC-USDT",
        interval="5m",
        limit=min(candles_needed, 1440),  # Cap at API limit
        start_time=start_time_ms,  # 72 hours ago
        end_time=end_time_ms       # Now
    )

    # Convert to DataFrame
    df_api = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_api['timestamp'] = pd.to_datetime(df_api['timestamp'], unit='ms', utc=True)

    # Convert price columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_api[col] = pd.to_numeric(df_api[col], errors='coerce')

    # Display period
    print(f"  ✅ Fetched {len(df_api)} candles")

    if len(df_api) > 0:
        period_start_utc = df_api['timestamp'].iloc[0]
        period_end_utc = df_api['timestamp'].iloc[-1]

        if pd.notna(period_start_utc) and pd.notna(period_end_utc):
            period_start_kst = period_start_utc.tz_convert(kst)
            period_end_kst = period_end_utc.tz_convert(kst)

            print(f"  Period (UTC): {period_start_utc.strftime('%Y-%m-%d %H:%M')} ~ {period_end_utc.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Period (KST): {period_start_kst.strftime('%Y-%m-%d %H:%M')} ~ {period_end_kst.strftime('%Y-%m-%d %H:%M')}")
            print(f"  ✅ Matches CSV data period")
        else:
            print(f"  ⚠️ Warning: Invalid timestamps in data")
    else:
        print(f"  ⚠️ Warning: No data received")

except Exception as e:
    print(f"  ❌ API Error: {e}")
    sys.exit(1)

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.80
SHORT_THRESHOLD = 0.80
GATE_THRESHOLD = 0.001
ML_EXIT_THRESHOLD_LONG = 0.80
ML_EXIT_THRESHOLD_SHORT = 0.80
EMERGENCY_MAX_HOLD_TIME = 120
EMERGENCY_STOP_LOSS = -0.03

LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005

# Load models
print("\nLoading models...")
long_model = pickle.load(open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb'))
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb'))
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

long_exit_model = pickle.load(open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb'))
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model = pickle.load(open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb'))
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ All models loaded")

# Initialize position sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Calculate features
print("\nCalculating features...")
df = calculate_all_features_enhanced_v2(df_api, phase='phase1')
df = prepare_exit_features(df)
print(f"  ✅ Features calculated: {len(df)} rows (lost {len(df_api) - len(df)} to lookback)")

# Pre-calculate signals
print("\nPre-calculating signals...")
long_feat_scaled = long_scaler.transform(df[long_feature_columns].values)
df['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

short_feat_scaled = short_scaler.transform(df[short_feature_columns].values)
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
            exit_feat = df[long_exit_feature_columns].iloc[i:i+1].values
            long_exit_prob = long_exit_model.predict_proba(long_exit_scaler.transform(exit_feat))[0][1]

            exit_feat = df[short_exit_feature_columns].iloc[i:i+1].values
            short_exit_prob = short_exit_model.predict_proba(short_exit_scaler.transform(exit_feat))[0][1]

            if long_exit_prob >= ML_EXIT_THRESHOLD_LONG or short_exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                first_exit_signal_received = True
        except:
            pass

    # Entry logic
    if first_exit_signal_received and position is None:
        long_prob = df['long_prob'].iloc[i]
        short_prob = df['short_prob'].iloc[i]

        if long_prob >= LONG_THRESHOLD:
            sizing = sizer.get_position_size_simple(capital, long_prob, LEVERAGE)
            position = {
                'side': 'LONG',
                'entry_idx': i,
                'entry_price': current_price,
                'position_size_pct': sizing['position_size_pct'],
                'leveraged_value': sizing['leveraged_value'],
                'quantity': sizing['leveraged_value'] / current_price
            }

        elif short_prob >= SHORT_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN

            if (short_ev - long_ev) > GATE_THRESHOLD:
                sizing = sizer.get_position_size_simple(capital, short_prob, LEVERAGE)
                position = {
                    'side': 'SHORT',
                    'entry_idx': i,
                    'entry_price': current_price,
                    'position_size_pct': sizing['position_size_pct'],
                    'leveraged_value': sizing['leveraged_value'],
                    'quantity': sizing['leveraged_value'] / current_price
                }

    # Exit logic
    if position is not None:
        time_in_pos = i - position['entry_idx']

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

        should_exit = False
        exit_reason = None

        # ML Exit
        try:
            if position['side'] == 'LONG':
                exit_feat = df[long_exit_feature_columns].iloc[i:i+1].values
                exit_prob = long_exit_model.predict_proba(long_exit_scaler.transform(exit_feat))[0][1]
                ml_threshold = ML_EXIT_THRESHOLD_LONG
            else:
                exit_feat = df[short_exit_feature_columns].iloc[i:i+1].values
                exit_prob = short_exit_model.predict_proba(short_exit_scaler.transform(exit_feat))[0][1]
                ml_threshold = ML_EXIT_THRESHOLD_SHORT

            if exit_prob >= ml_threshold:
                should_exit = True
                exit_reason = f'ml_exit_{position["side"].lower()}'
        except:
            pass

        # Stop Loss
        if not should_exit and balance_loss_pct <= EMERGENCY_STOP_LOSS:
            should_exit = True
            exit_reason = 'emergency_stop_loss'

        # Max Hold
        if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
            should_exit = True
            exit_reason = 'emergency_max_hold'

        if should_exit:
            entry_commission = position['leveraged_value'] * TAKER_FEE
            exit_commission = position['quantity'] * current_price * TAKER_FEE
            total_commission = entry_commission + exit_commission

            net_pnl_usd = pnl_usd - total_commission
            capital += net_pnl_usd

            trades.append({
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
            })

            position = None

# Results
print("\n" + "="*80)
print("72-HOUR BACKTEST RESULTS (REAL API DATA)")
print("="*80)

if len(trades) > 0:
    trades_df = pd.DataFrame(trades)

    total_return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades) * 100

    long_trades = trades_df[trades_df['side'] == 'LONG']
    short_trades = trades_df[trades_df['side'] == 'SHORT']

    print(f"\n📊 Performance:")
    print(f"  Return: {total_return_pct:+.2f}%")
    print(f"  Final Capital: ${capital:,.2f} (started ${INITIAL_CAPITAL:,.2f})")
    print(f"  Win Rate: {win_rate:.1f}%")

    print(f"\n📈 Trades:")
    print(f"  Total: {len(trades)}")
    print(f"  LONG: {len(long_trades)} ({len(long_trades)/len(trades)*100:.1f}%)")
    print(f"  SHORT: {len(short_trades)} ({len(short_trades)/len(trades)*100:.1f}%)")
    print(f"  Trades/day: {len(trades)/3:.1f}")

    print(f"\n💰 Trade Metrics:")
    print(f"  Avg Return: {trades_df['leveraged_pnl_pct'].mean()*100:+.2f}%")
    print(f"  Best Trade: {trades_df['leveraged_pnl_pct'].max()*100:+.2f}%")
    print(f"  Worst Trade: {trades_df['leveraged_pnl_pct'].min()*100:+.2f}%")
    print(f"  Avg Position: {trades_df['position_size_pct'].mean()*100:.1f}%")
    print(f"  Avg Hold: {trades_df['hold_time'].mean():.1f} candles ({trades_df['hold_time'].mean()*5/60:.1f}h)")

    print(f"\n🚪 Exit Reasons:")
    for reason, count in trades_df['exit_reason'].value_counts().items():
        pct = count / len(trades) * 100
        reason_map = {
            'ml_exit_long': 'ML Exit (LONG)',
            'ml_exit_short': 'ML Exit (SHORT)',
            'emergency_stop_loss': 'Stop Loss',
            'emergency_max_hold': 'Max Hold'
        }
        print(f"  {reason_map.get(reason, reason):<20s}: {count:>3d} ({pct:>5.1f}%)")

    print(f"\n💸 Commissions:")
    print(f"  Total: ${trades_df['commission_usd'].sum():.2f}")
    print(f"  Per trade: ${trades_df['commission_usd'].mean():.2f}")

    daily_return = total_return_pct / 3
    print(f"\n📅 Extrapolation:")
    print(f"  Daily: {daily_return:+.2f}%")
    print(f"  Weekly (7d): {daily_return * 7:+.2f}%")
    print(f"  Monthly (30d): {daily_return * 30:+.2f}%")

else:
    print("\n❌ No trades executed during period")

print("\n" + "="*80)
