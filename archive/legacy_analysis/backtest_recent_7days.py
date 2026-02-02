#!/usr/bin/env python3
"""
7-Day Backtest - Production Behavior Validation
Purpose: Compare production trading frequency with backtest over 7-day period
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime, timedelta
from pathlib import Path
import pytz
import pandas as pd
import numpy as np
import joblib
import yaml
from src.api.bingx_client import BingXClient
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2

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
API_KEY = _api_config.get('api_key', "")
API_SECRET = _api_config.get('secret_key', "")

# Production Models
LONG_ENTRY_MODEL = "models/xgboost_long_entry_enhanced_20251024_012445.pkl"
SHORT_ENTRY_MODEL = "models/xgboost_short_entry_enhanced_20251024_012445.pkl"
LONG_EXIT_MODEL = "models/xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
SHORT_EXIT_MODEL = "models/xgboost_short_exit_oppgating_improved_20251024_044510.pkl"

# Configuration (Production Settings)
ENTRY_THRESHOLD_LONG = 0.70
ENTRY_THRESHOLD_SHORT = 0.70
EXIT_THRESHOLD_LONG = 0.70
EXIT_THRESHOLD_SHORT = 0.70
EMERGENCY_STOP_LOSS = 0.03
EMERGENCY_MAX_HOLD_TIME = 120
LEVERAGE = 4
FEE_RATE = 0.001  # BingX: 0.1% per side (entry + exit = 0.2% total)

print("=" * 80)
print("📊 7-DAY BACKTEST - PRODUCTION BEHAVIOR VALIDATION")
print("=" * 80)
print(f"⏰ Period: Last 7 days (~1440 candles max from API)")
print(f"🤖 Models: Enhanced 5-Fold CV (20251024_012445)")
print(f"⚙️  Config: Entry {ENTRY_THRESHOLD_LONG}/{ENTRY_THRESHOLD_SHORT}, Exit {EXIT_THRESHOLD_LONG}/{EXIT_THRESHOLD_SHORT}")
print("=" * 80)
print()

# Initialize client
client = BingXClient(api_key=API_KEY, secret_key=API_SECRET, testnet=False)
SYMBOL = "BTC-USDT"

# Fetch maximum candles (API limit: 1440)
print("📡 Fetching candles (max 1440 from API)...")
klines = client.get_klines(symbol=SYMBOL, interval="5m", limit=1440)
print(f"✅ Fetched {len(klines)} candles")

# Convert to DataFrame (UTC timestamps)
df = pd.DataFrame(klines)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

time_range_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
print(f"📊 Time coverage: {time_range_days:.1f} days ({len(df)} candles)")
print(f"   From: {df.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
print(f"   To:   {df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
print()

# Calculate features
print("🔧 Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
print(f"✅ Features calculated: {len(df_features)} rows (lost {len(df) - len(df_features)} due to lookback)")
print()

# Load models
print("🤖 Loading models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
long_exit_model = joblib.load(LONG_EXIT_MODEL)
short_exit_model = joblib.load(SHORT_EXIT_MODEL)

long_entry_scaler = joblib.load(LONG_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))
short_entry_scaler = joblib.load(SHORT_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))
long_exit_scaler = joblib.load(LONG_EXIT_MODEL.replace('.pkl', '_scaler.pkl'))
short_exit_scaler = joblib.load(SHORT_EXIT_MODEL.replace('.pkl', '_scaler.pkl'))

# Load feature lists
with open(LONG_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines() if line.strip()]
with open(SHORT_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]
with open(LONG_EXIT_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]
with open(SHORT_EXIT_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Models loaded")
print()

# Prepare exit features
def prepare_exit_features(df):
    """Prepare EXIT features - Same as production"""
    
    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum features
    if 'sma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['sma_50']) / df['sma_50']
    elif 'ma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'sma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['sma_200']) / df['sma_200']
    elif 'ma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    # Volatility features
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)

    # RSI dynamics
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0

    # MACD dynamics
    df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3 if 'macd_hist' in df.columns else 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()

    # Support/Resistance
    if 'support_level' in df.columns and 'resistance_level' in df.columns:
        df['near_resistance'] = (df['close'] > df['resistance_level'] * 0.98).astype(float)
        df['near_support'] = (df['close'] < df['support_level'] * 1.02).astype(float)
    else:
        df['near_resistance'] = 0
        df['near_support'] = 0

    # Bollinger Band position
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5

    # Clean NaN
    df = df.ffill().bfill()
    
    return df

df_features = prepare_exit_features(df_features)

# Simple backtest simulation
print("🎯 Running backtest simulation...")
print("=" * 80)

initial_balance = 10000.0
balance = initial_balance
position = None
trades = []

for idx in range(len(df_features)):
    row = df_features.iloc[idx]
    timestamp = row['timestamp']
    price = row['close']

    # Entry signals
    try:
        long_feat_df = df_features.iloc[[idx]][long_entry_features]
        long_prob = long_entry_model.predict_proba(long_entry_scaler.transform(long_feat_df.values))[0][1]
    except:
        long_prob = 0.0

    try:
        short_feat_df = df_features.iloc[[idx]][short_entry_features]
        short_prob = short_entry_model.predict_proba(short_entry_scaler.transform(short_feat_df.values))[0][1]
    except:
        short_prob = 0.0

    # Exit signals (if in position)
    if position is not None:
        try:
            if position['side'] == 'LONG':
                exit_feat_df = df_features.iloc[[idx]][long_exit_features]
                exit_prob = long_exit_model.predict_proba(long_exit_scaler.transform(exit_feat_df.values))[0][1]
            else:
                exit_feat_df = df_features.iloc[[idx]][short_exit_features]
                exit_prob = short_exit_model.predict_proba(short_exit_scaler.transform(exit_feat_df.values))[0][1]
        except:
            exit_prob = 0.0

        # Check exit conditions
        hold_time = idx - position['entry_idx']

        if position['side'] == 'LONG':
            pnl_pct = (price - position['entry_price']) / position['entry_price'] * LEVERAGE
            sl_price = position['entry_price'] * (1 - EMERGENCY_STOP_LOSS / (position['size_pct'] * LEVERAGE))
            hit_sl = price <= sl_price
        else:
            pnl_pct = (position['entry_price'] - price) / position['entry_price'] * LEVERAGE
            sl_price = position['entry_price'] * (1 + EMERGENCY_STOP_LOSS / (position['size_pct'] * LEVERAGE))
            hit_sl = price >= sl_price

        exit_reason = None
        if exit_prob >= (EXIT_THRESHOLD_LONG if position['side'] == 'LONG' else EXIT_THRESHOLD_SHORT):
            exit_reason = 'ML_EXIT'
        elif hit_sl:
            exit_reason = 'STOP_LOSS'
        elif hold_time >= EMERGENCY_MAX_HOLD_TIME:
            exit_reason = 'MAX_HOLD'

        if exit_reason:
            # Calculate P&L
            gross_pnl = balance * pnl_pct

            # Calculate fees (0.1% entry + 0.1% exit on position value)
            position_value = balance * position['size_pct']
            entry_fee = position_value * FEE_RATE
            exit_fee = position_value * FEE_RATE
            total_fees = entry_fee + exit_fee

            # Net P&L after fees
            balance_change = gross_pnl - total_fees
            balance += balance_change

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': price,
                'hold_time': hold_time,
                'pnl_pct': pnl_pct * 100,
                'gross_pnl': gross_pnl,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'total_fees': total_fees,
                'balance_change': balance_change,
                'balance': balance,
                'exit_reason': exit_reason
            })
            position = None

    # Entry logic (if no position)
    if position is None:
        side = None
        if long_prob >= ENTRY_THRESHOLD_LONG:
            side = 'LONG'
        elif short_prob >= ENTRY_THRESHOLD_SHORT:
            side = 'SHORT'

        if side:
            position = {
                'side': side,
                'entry_time': timestamp,
                'entry_idx': idx,
                'entry_price': price,
                'size_pct': 0.95
            }

# Convert trades to DataFrame
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    # Calculate daily statistics
    trades_df['date'] = pd.to_datetime(trades_df['exit_time']).dt.date
    daily_stats = trades_df.groupby('date').agg({
        'side': 'count',  # trades per day
        'pnl_pct': 'mean',
        'balance_change': 'sum'
    }).rename(columns={'side': 'trades_count'})

    print(f"\n📊 DAILY TRADING STATISTICS")
    print("=" * 80)
    print(f"{'Date':<12} {'Trades':>8} {'Avg P&L%':>10} {'Daily P&L':>12}")
    print("=" * 80)
    for date, row in daily_stats.iterrows():
        print(f"{date} {row['trades_count']:>8.0f} {row['pnl_pct']:>9.2f}% ${row['balance_change']:>11.2f}")
    print("=" * 80)

    # Overall statistics
    total_return = (balance - initial_balance) / initial_balance * 100
    win_rate = len(trades_df[trades_df['pnl_pct'] > 0]) / len(trades_df) * 100
    avg_trades_per_day = daily_stats['trades_count'].mean()

    print(f"\n📈 BACKTEST SUMMARY ({time_range_days:.1f} days)")
    print("=" * 80)
    print(f"Total Trades: {len(trades_df)}")
    print(f"Trades/Day: {avg_trades_per_day:.1f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Final Balance: ${balance:,.2f}")
    print()

    # Exit distribution
    exit_dist = trades_df['exit_reason'].value_counts()
    print(f"Exit Distribution:")
    for reason, count in exit_dist.items():
        pct = count / len(trades_df) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    print("=" * 80)

    # Save results
    output_file = f"results/backtest_7day_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved results to: {output_file}")

    # Compare with production expectations
    print(f"\n⚠️  PRODUCTION COMPARISON")
    print("=" * 80)
    print(f"Expected (normal): 2-3 trades/day")
    print(f"Backtest result: {avg_trades_per_day:.1f} trades/day")

    if avg_trades_per_day > 50:
        print(f"❌ CRITICAL: Backtest shows RAPID TRADING ({avg_trades_per_day:.0f}/day)")
        print(f"   This matches production abnormal behavior!")
    elif avg_trades_per_day > 10:
        print(f"⚠️  WARNING: Higher than expected ({avg_trades_per_day:.0f}/day)")
    else:
        print(f"✅ Normal trading frequency")
    print("=" * 80)
else:
    print("⚠️  No trades executed in backtest period")