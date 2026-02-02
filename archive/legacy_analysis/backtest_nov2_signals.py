#!/usr/bin/env python3
"""
Generate backtest signals for November 2, 2025
Purpose: Compare production vs backtest probabilities for historical validation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime, timezone
import pytz
import pandas as pd
import joblib
import yaml
from src.api.bingx_client import BingXClient
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from pathlib import Path

# Load API keys
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

def load_api_keys():
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

_api_config = load_api_keys()
API_KEY = _api_config.get('api_key', "")
API_SECRET = _api_config.get('secret_key', "")

LONG_ENTRY_MODEL = "models/xgboost_long_entry_enhanced_20251024_012445.pkl"
SHORT_ENTRY_MODEL = "models/xgboost_short_entry_enhanced_20251024_012445.pkl"

print("=" * 80)
print("BACKTEST SIGNAL GENERATION - NOVEMBER 2, 2025")
print("=" * 80)
print("Purpose: Validate backtest vs production consistency (historical data)")
print("=" * 80)
print()

print("📡 Fetching candles from BingX API...")
client = BingXClient(api_key=API_KEY, secret_key=API_SECRET, testnet=False)
klines = client.get_klines(symbol="BTC-USDT", interval="5m", limit=800)

df = pd.DataFrame(klines)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Fetched {len(df)} candles")
print(f"   From: {df.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
print(f"   To:   {df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
print()

print("🔧 Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
print(f"✅ Features calculated: {len(df_features)} rows (lost {len(df) - len(df_features)} due to lookback)")
print()

# Filter to November 2 time periods
# Morning: 06:40-08:15 KST = 21:40-23:15 UTC (Nov 1)
# Evening: 13:15-14:50 KST = 04:15-05:50 UTC (Nov 2)
morning_start = datetime(2025, 11, 1, 21, 40, 0)
morning_end = datetime(2025, 11, 1, 23, 15, 0)
evening_start = datetime(2025, 11, 2, 4, 15, 0)
evening_end = datetime(2025, 11, 2, 5, 50, 0)

df_morning = df_features[
    (df_features['timestamp'] >= morning_start) &
    (df_features['timestamp'] <= morning_end)
].copy()

df_evening = df_features[
    (df_features['timestamp'] >= evening_start) &
    (df_features['timestamp'] <= evening_end)
].copy()

print(f"📊 Target periods:")
print(f"   Morning: {len(df_morning)} candles (06:40-08:15 KST)")
print(f"   Evening: {len(df_evening)} candles (13:15-14:50 KST)")

# Combine both periods
df_target = pd.concat([df_morning, df_evening], ignore_index=True).sort_values('timestamp')
print(f"   Total: {len(df_target)} candles")
print()

if len(df_target) == 0:
    print("❌ ERROR: No data in target range")
    sys.exit(1)

# Load models
print("🤖 Loading models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
long_entry_scaler = joblib.load(LONG_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))
short_entry_scaler = joblib.load(SHORT_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))

with open(LONG_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]
with open(SHORT_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    short_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Models loaded")
print(f"   LONG features: {len(long_features)}")
print(f"   SHORT features: {len(short_features)}")
print()

# Generate signals
print("🔍 Generating signals...")
kst = pytz.timezone('Asia/Seoul')
signals = []

for idx, row in df_target.iterrows():
    candle_time_utc = row['timestamp']
    candle_time_kst = pytz.utc.localize(candle_time_utc).astimezone(kst).replace(tzinfo=None)
    price = row['close']

    # LONG signal
    try:
        long_feat_df = df_target.loc[[idx], long_features]
        if long_feat_df.shape[1] == len(long_features):
            long_feat_scaled = long_entry_scaler.transform(long_feat_df.values)
            long_prob = long_entry_model.predict_proba(long_feat_scaled)[0][1]
        else:
            long_prob = 0.0
    except Exception as e:
        long_prob = 0.0

    # SHORT signal
    try:
        short_feat_df = df_target.loc[[idx], short_features]
        if short_feat_df.shape[1] == len(short_features):
            short_feat_scaled = short_entry_scaler.transform(short_feat_df.values)
            short_prob = short_entry_model.predict_proba(short_feat_scaled)[0][1]
        else:
            short_prob = 0.0
    except Exception as e:
        short_prob = 0.0

    signals.append({
        'time_utc': candle_time_utc,
        'time_kst': candle_time_kst,
        'price': price,
        'long_prob': long_prob,
        'short_prob': short_prob
    })

# Save results
signals_df = pd.DataFrame(signals)
output_file = "results/backtest_signals_nov2_20251103.csv"
signals_df.to_csv(output_file, index=False)

print(f"✅ Generated {len(signals)} signals")
print(f"💾 Saved to: {output_file}")
print()
print("=" * 80)
print("First 5 signals:")
print(signals_df.head().to_string())
print()
print("Last 5 signals:")
print(signals_df.tail().to_string())
print("=" * 80)
