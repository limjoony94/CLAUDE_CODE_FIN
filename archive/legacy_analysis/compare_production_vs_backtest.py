#!/usr/bin/env python3
"""
Compare Production vs Backtest Signals (Last 6 Hours)
Purpose: Verify that backtest replicates production behavior exactly
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

print("=" * 80)
print("📊 PRODUCTION vs BACKTEST SIGNAL COMPARISON")
print("=" * 80)
print(f"⏰ Time Range: Last 6 hours (72 candles)")
print(f"🤖 Models: Enhanced 5-Fold CV (20251024_012445)")
print("=" * 80)
print()

# Initialize client
client = BingXClient(api_key=API_KEY, secret_key=API_SECRET, testnet=False)
SYMBOL = "BTC-USDT"

# Fetch candles
print("📡 Fetching candles...")
klines = client.get_klines(symbol=SYMBOL, interval="5m", limit=1000)
print(f"✅ Fetched {len(klines)} candles")

# Convert to DataFrame (UTC timestamps)
df = pd.DataFrame(klines)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"📊 Raw data: {len(df)} candles")
print()

# Calculate features
print("🔧 Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
print(f"✅ Features calculated: {len(df_features)} rows (lost {len(df) - len(df_features)} due to lookback)")
print()

# Get last 72 candles (6 hours)
df_6h = df_features.tail(72).copy()
print(f"📉 Analysis window: Last 72 candles (6 hours)")
print(f"   From: {df_6h.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
print(f"   To:   {df_6h.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
print()

# Load models and scalers
print("🤖 Loading models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)

long_entry_scaler = joblib.load(LONG_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))
short_entry_scaler = joblib.load(SHORT_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))

# Load feature lists
with open(LONG_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]

with open(SHORT_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    short_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Models loaded")
print(f"   LONG features: {len(long_features)}")
print(f"   SHORT features: {len(short_features)}")
print()

# Calculate signals for every candle
print("🔍 Calculating signals for all 72 candles...")
print("=" * 80)

signals = []
kst = pytz.timezone('Asia/Seoul')

for idx, row in df_6h.iterrows():
    candle_time_utc = row['timestamp']
    candle_time_kst = pytz.utc.localize(candle_time_utc).astimezone(kst).replace(tzinfo=None)
    price = row['close']

    # LONG signal
    try:
        long_feat_df = df_6h.loc[[idx], long_features]
        if long_feat_df.shape[1] == len(long_features):
            long_feat_scaled = long_entry_scaler.transform(long_feat_df.values)
            long_prob = long_entry_model.predict_proba(long_feat_scaled)[0][1]
        else:
            long_prob = 0.0
    except Exception as e:
        long_prob = 0.0

    # SHORT signal
    try:
        short_feat_df = df_6h.loc[[idx], short_features]
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

# Convert to DataFrame
signals_df = pd.DataFrame(signals)

# Print comparison table (every 6 candles = 30 min)
print(f"{'Time (KST)':<20} {'Time (UTC)':<20} {'Price':>12} {'LONG':>8} {'SHORT':>8}")
print("=" * 80)

for i in range(0, len(signals_df), 6):
    row = signals_df.iloc[i]
    print(f"{row['time_kst'].strftime('%Y-%m-%d %H:%M'):<20} "
          f"{row['time_utc'].strftime('%Y-%m-%d %H:%M'):<20} "
          f"${row['price']:>11,.1f} "
          f"{row['long_prob']:>7.4f} "
          f"{row['short_prob']:>7.4f}")

print("=" * 80)
print()

# Save to CSV for detailed analysis
output_file = "results/production_vs_backtest_signals_6h.csv"
signals_df.to_csv(output_file, index=False)
print(f"💾 Saved detailed results to: {output_file}")
print()

print("=" * 80)
print("✅ COMPARISON COMPLETE")
print("=" * 80)
print()
print("📋 Next Steps:")
print("   1. Compare these backtest signals with production logs")
print("   2. Check for price/probability mismatches")
print("   3. Investigate any discrepancies")
