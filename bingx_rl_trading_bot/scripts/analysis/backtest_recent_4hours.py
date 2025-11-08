#!/usr/bin/env python3
"""
Backtest Recent 4 Hours with Production Models
Purpose: Analyze recent trading frequency with production configuration
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytz
import pandas as pd
import numpy as np
import joblib
import yaml
from src.api.bingx_client import BingXClient

# Load API keys
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

def load_api_keys():
    """Load API keys from config/api_keys.yaml"""
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

_api_config = load_api_keys()
API_KEY = _api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
API_SECRET = _api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))

# Production Models (2025-10-24)
LONG_ENTRY_MODEL = "models/xgboost_long_entry_enhanced_20251024_012445.pkl"
SHORT_ENTRY_MODEL = "models/xgboost_short_entry_enhanced_20251024_012445.pkl"
LONG_EXIT_MODEL = "models/xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
SHORT_EXIT_MODEL = "models/xgboost_short_exit_oppgating_improved_20251024_044510.pkl"

# Production Configuration (CURRENT - 2025-11-03)
LONG_THRESHOLD = 0.80  # Entry threshold
SHORT_THRESHOLD = 0.80  # Entry threshold
ML_EXIT_THRESHOLD_LONG = 0.70  # Exit threshold (Changed 2025-11-02)
ML_EXIT_THRESHOLD_SHORT = 0.70  # Exit threshold (Changed 2025-11-02)
EMERGENCY_STOP_LOSS = 0.03  # -3% balance
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours
LEVERAGE = 4
POSITION_SIZE_RANGE = (0.20, 0.95)
GATE_THRESHOLD = 0.001

print("=" * 80)
print("🔍 RECENT 2-HOUR BACKTEST - PRODUCTION MODELS")
print("=" * 80)
print(f"📅 Time Range: Last 2 hours (24 candles)")
print(f"🤖 Models: Enhanced 5-Fold CV (20251024_012445)")
print(f"⚙️  Entry Thresholds: LONG={LONG_THRESHOLD}, SHORT={SHORT_THRESHOLD}")
print(f"🚪 Exit Thresholds: LONG={ML_EXIT_THRESHOLD_LONG}, SHORT={ML_EXIT_THRESHOLD_SHORT}")
print(f"🛡️  Stop Loss: -{EMERGENCY_STOP_LOSS*100:.1f}% balance")
print(f"⏱️  Max Hold: {EMERGENCY_MAX_HOLD_TIME} candles (10 hours)")
print("=" * 80)
print()

# Initialize client
client = BingXClient(api_key=API_KEY, secret_key=API_SECRET, testnet=False)
SYMBOL = "BTC-USDT"

# Get current time and calculate 4-hour range
kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(kst)
four_hours_ago = now_kst - timedelta(hours=4)

print(f"📊 Fetching data...")
print(f"   Start (KST): {four_hours_ago.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   End (KST): {now_kst.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Fetch data (need more for indicators - get 1000 candles)
klines = client.get_klines(symbol=SYMBOL, interval="5m", limit=1000)
print(f"✅ Fetched {len(klines)} candles")

# Convert to DataFrame
df = pd.DataFrame(klines)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

# Store timestamps for filtering later (after features are calculated)
df_timestamps_kst = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert(kst)
print(f"📉 Total candles: {len(df)} candles")
print()

# Load models
print("🤖 Loading production models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
long_exit_model = joblib.load(LONG_EXIT_MODEL)
short_exit_model = joblib.load(SHORT_EXIT_MODEL)

long_entry_scaler = joblib.load(LONG_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))
short_entry_scaler = joblib.load(SHORT_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))
long_exit_scaler = joblib.load(LONG_EXIT_MODEL.replace('.pkl', '_scaler.pkl'))
short_exit_scaler = joblib.load(SHORT_EXIT_MODEL.replace('.pkl', '_scaler.pkl'))

# Load feature lists from .txt files
with open(LONG_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines() if line.strip()]
with open(SHORT_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]
with open(LONG_EXIT_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]
with open(SHORT_EXIT_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"   LONG Entry: {len(long_entry_features)} features")
print(f"   SHORT Entry: {len(short_entry_features)} features")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")
print()

# Import feature calculation functions
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2

# Function to prepare exit features
def prepare_exit_features(df):
    """Add exit-specific features to dataframe"""
    # Volume surge
    df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()

    # Price acceleration
    df['price_acceleration'] = df['close'].pct_change().rolling(5).mean()

    # Price vs MAs
    df['price_vs_ma20'] = (df['close'] - df['ma_20']) / df['ma_20']
    # Calculate MA50 if not exists
    if 'ma50' not in df.columns:
        df['ma50'] = df['close'].rolling(50).mean()
    df['price_vs_ma50'] = (df['close'] - df['ma50']) / df['ma50']

    # Volatility
    df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    # RSI features
    df['rsi_slope'] = df['rsi'].diff(5)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = df['rsi'].rolling(10).apply(
        lambda x: 1 if (x.iloc[-1] > x.iloc[0] and x.max() > 70) else 0, raw=False
    )

    # MACD features (macd_histogram is actually macd_diff)
    df['macd_histogram_slope'] = df['macd_diff'].diff(3)
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    # Price patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                         (df['high'].shift(1) > df['high'].shift(2))).astype(float)
    df['near_support'] = (df['low'] <= df['low'].rolling(20).min() * 1.02).astype(float)

    # Bollinger Band position (bb_lower/bb_upper are actually bb_low/bb_high)
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

    return df

# Calculate features
print("🔧 Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"✅ Features calculated: {len(df_features)} rows")

# Use the last 24 candles from feature dataframe (most recent 2 hours with features)
df_backtest_window = df_features.tail(24).copy()
first_time = pd.to_datetime(df_backtest_window.iloc[0]['timestamp']).strftime('%Y-%m-%d %H:%M')
last_time = pd.to_datetime(df_backtest_window.iloc[-1]['timestamp']).strftime('%Y-%m-%d %H:%M')

print(f"📉 Backtest window: {len(df_backtest_window)} candles (most recent with features)")
print(f"   From: {first_time}")
print(f"   To: {last_time}")
print()

# Run backtest
print("🎯 Running backtest...")
print()

initial_balance = 1000.0
balance = initial_balance
position = None
trades = []
candle_count = 0

for idx in df_backtest_window.index:
    candle_count += 1
    candle = df_features.loc[idx]
    current_time = candle['timestamp']
    current_price = candle['close']

    # Entry logic
    if position is None:
        # Get LONG signal
        long_features_df = df_features.loc[[idx], long_entry_features]
        long_features_scaled = long_entry_scaler.transform(long_features_df)
        long_prob = long_entry_model.predict_proba(long_features_scaled)[0, 1]

        # Get SHORT signal
        short_features_df = df_features.loc[[idx], short_entry_features]
        short_features_scaled = short_entry_scaler.transform(short_features_df)
        short_prob = short_entry_model.predict_proba(short_features_scaled)[0, 1]

        # Print signals every 6 candles (30 minutes)
        if candle_count % 6 == 1:
            print(f"[{current_time}] Signals - LONG: {long_prob:.4f} | SHORT: {short_prob:.4f} | Price: ${current_price:,.1f}")

        # Entry decision
        entered = False

        if long_prob >= LONG_THRESHOLD:
            # LONG entry
            position_size_pct = POSITION_SIZE_RANGE[0] + (POSITION_SIZE_RANGE[1] - POSITION_SIZE_RANGE[0]) * long_prob
            position = {
                'side': 'LONG',
                'entry_price': current_price,
                'entry_time': current_time,
                'entry_idx': idx,
                'position_size_pct': position_size_pct,
                'position_size_usd': balance * position_size_pct,
                'entry_prob': long_prob,
                'hold_candles': 0
            }
            entered = True

        elif short_prob >= SHORT_THRESHOLD:
            # Check opportunity gating
            long_ev = long_prob * 0.0041
            short_ev = short_prob * 0.0047
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > GATE_THRESHOLD:
                # SHORT entry
                position_size_pct = POSITION_SIZE_RANGE[0] + (POSITION_SIZE_RANGE[1] - POSITION_SIZE_RANGE[0]) * short_prob
                position = {
                    'side': 'SHORT',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'entry_idx': idx,
                    'position_size_pct': position_size_pct,
                    'position_size_usd': balance * position_size_pct,
                    'entry_prob': short_prob,
                    'hold_candles': 0
                }
                entered = True

        if entered:
            print(f"📥 [{current_time}] {position['side']} ENTRY")
            print(f"   Price: ${current_price:,.1f}")
            print(f"   Probability: {position['entry_prob']:.4f}")
            print(f"   Position Size: {position_size_pct*100:.1f}%")

    else:
        # Exit logic
        position['hold_candles'] += 1

        # Calculate current P&L
        if position['side'] == 'LONG':
            price_change = (current_price - position['entry_price']) / position['entry_price']
        else:
            price_change = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl = price_change * LEVERAGE

        # Check ML Exit
        exit_signal = False
        exit_reason = None

        if position['side'] == 'LONG':
            exit_features_df = df_features.loc[[idx], long_exit_features]
            exit_features_scaled = long_exit_scaler.transform(exit_features_df)
            exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0, 1]

            if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                exit_signal = True
                exit_reason = f"ML Exit ({exit_prob:.3f})"
        else:
            exit_features_df = df_features.loc[[idx], short_exit_features]
            exit_features_scaled = short_exit_scaler.transform(exit_features_df)
            exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0, 1]

            if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                exit_signal = True
                exit_reason = f"ML Exit ({exit_prob:.3f})"

        # Check Stop Loss
        balance_loss = leveraged_pnl * position['position_size_pct']
        if balance_loss <= -EMERGENCY_STOP_LOSS:
            exit_signal = True
            exit_reason = f"Stop Loss ({balance_loss*100:.2f}%)"

        # Check Max Hold
        if position['hold_candles'] >= EMERGENCY_MAX_HOLD_TIME:
            exit_signal = True
            exit_reason = f"Max Hold ({position['hold_candles']} candles)"

        if exit_signal:
            # Calculate final P&L
            pnl_usd = position['position_size_usd'] * leveraged_pnl
            balance += pnl_usd

            trade = {
                'side': position['side'],
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'hold_candles': position['hold_candles'],
                'entry_prob': position['entry_prob'],
                'exit_reason': exit_reason,
                'price_change_pct': price_change * 100,
                'leveraged_pnl_pct': leveraged_pnl * 100,
                'pnl_usd': pnl_usd,
                'position_size_pct': position['position_size_pct'] * 100
            }
            trades.append(trade)

            print(f"📤 [{current_time}] {position['side']} EXIT")
            print(f"   Price: ${current_price:,.1f}")
            print(f"   Hold: {position['hold_candles']} candles")
            print(f"   Exit: {exit_reason}")
            print(f"   P&L: ${pnl_usd:+.2f} ({leveraged_pnl*100:+.2f}%)")
            print()

            position = None

print("=" * 80)
print("📊 BACKTEST RESULTS")
print("=" * 80)
print()

# Results summary
if trades:
    df_trades = pd.DataFrame(trades)

    total_trades = len(trades)
    wins = sum(df_trades['pnl_usd'] > 0)
    losses = sum(df_trades['pnl_usd'] <= 0)
    win_rate = wins / total_trades * 100

    total_return = (balance - initial_balance) / initial_balance * 100
    avg_trade_pnl = df_trades['pnl_usd'].mean()

    # Exit reasons
    ml_exits = sum(df_trades['exit_reason'].str.contains('ML Exit'))
    sl_exits = sum(df_trades['exit_reason'].str.contains('Stop Loss'))
    maxhold_exits = sum(df_trades['exit_reason'].str.contains('Max Hold'))

    # Side distribution
    long_trades = sum(df_trades['side'] == 'LONG')
    short_trades = sum(df_trades['side'] == 'SHORT')

    print(f"💰 Performance:")
    print(f"   Initial Balance: ${initial_balance:,.2f}")
    print(f"   Final Balance: ${balance:,.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print()

    print(f"📈 Trading Statistics:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Trades per Hour: {total_trades/4:.1f}")
    print(f"   Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"   Avg Trade P&L: ${avg_trade_pnl:+.2f}")
    print()

    print(f"🎯 Exit Distribution:")
    print(f"   ML Exit: {ml_exits} ({ml_exits/total_trades*100:.1f}%)")
    print(f"   Stop Loss: {sl_exits} ({sl_exits/total_trades*100:.1f}%)")
    print(f"   Max Hold: {maxhold_exits} ({maxhold_exits/total_trades*100:.1f}%)")
    print()

    print(f"📊 Side Distribution:")
    print(f"   LONG: {long_trades} ({long_trades/total_trades*100:.1f}%)")
    print(f"   SHORT: {short_trades} ({short_trades/total_trades*100:.1f}%)")
    print()

    # Recent 1 hour analysis (last 12 candles)
    recent_trades = df_trades.tail(12) if len(df_trades) >= 12 else df_trades

    print(f"⏱️  Last 1 Hour Activity:")
    print(f"   Trades: {len(recent_trades)}")
    if len(recent_trades) > 0:
        print(f"   Avg Hold: {recent_trades['hold_candles'].mean():.1f} candles")
        print(f"   Return: {recent_trades['pnl_usd'].sum():+.2f} USD")
        print()

        print(f"   Recent Trades Detail:")
        for _, trade in recent_trades.iterrows():
            print(f"   - {trade['side']:5s} | Hold: {trade['hold_candles']:3.0f} | {trade['exit_reason']:20s} | ${trade['pnl_usd']:+6.2f}")

    print()

    # Trade details
    print("📋 All Trades:")
    for i, trade in enumerate(trades, 1):
        print(f"{i:2d}. {trade['side']:5s} | "
              f"Entry: {pd.to_datetime(trade['entry_time']).strftime('%H:%M')} | "
              f"Exit: {pd.to_datetime(trade['exit_time']).strftime('%H:%M')} | "
              f"Hold: {trade['hold_candles']:3.0f} | "
              f"{trade['exit_reason']:20s} | "
              f"${trade['pnl_usd']:+6.2f}")

else:
    print("⚠️  No trades executed in 4-hour window")

print()
print("=" * 80)
