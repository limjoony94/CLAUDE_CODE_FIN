"""
Debug: Check Target Timestamps and Signals
===========================================

Check what happens to the target timestamps (2025-10-24 01:00, 01:10 KST)
during feature calculation.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import yaml
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from src.api.bingx_client import BingXClient

MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config" / "api_keys.yaml"

print("="*80)
print("TARGET TIMESTAMPS DEBUG")
print("="*80)
print()

# Load API keys
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']

client = BingXClient(api_key, secret_key)

# Target timestamps (KST → UTC)
target_times_kst = [
    '2025-10-24 01:00:00',
    '2025-10-24 01:10:00'
]

print("Target times:")
for time_kst in target_times_kst:
    ts_kst = pd.Timestamp(time_kst)
    ts_utc = ts_kst - timedelta(hours=9)
    print(f"  {time_kst} KST → {ts_utc} UTC")
print()

# Load data from API
print("Loading 1000 candles from API...")
candles = client.get_klines("BTC-USDT", "5m", limit=1000)

df_raw = pd.DataFrame(candles)
df_raw.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')

for col in ['open', 'high', 'low', 'close', 'volume']:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

print(f"✅ Loaded {len(df_raw)} candles")
print(f"   Range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
print()

# Check if target timestamps exist in RAW data
print("="*80)
print("CHECK 1: Target Timestamps in RAW DATA")
print("="*80)
print()

for time_kst in target_times_kst:
    ts_kst = pd.Timestamp(time_kst)
    ts_utc = ts_kst - timedelta(hours=9)

    exact = df_raw[df_raw['timestamp'] == ts_utc]

    if len(exact) > 0:
        row = exact.iloc[0]
        print(f"✅ FOUND in RAW: {time_kst} KST ({ts_utc} UTC)")
        print(f"   Price: ${row['close']:,.2f}")
    else:
        print(f"❌ NOT FOUND in RAW: {time_kst} KST ({ts_utc} UTC)")
        # Find nearest
        df_raw['time_diff'] = (df_raw['timestamp'] - ts_utc).abs()
        nearest = df_raw.nsmallest(1, 'time_diff').iloc[0]
        print(f"   Nearest: {nearest['timestamp']} (diff: {nearest['time_diff']})")
    print()

# Calculate features
print("="*80)
print("Calculating features...")
print("="*80)
print()

df_features = calculate_all_features_enhanced_v2(df_raw)

print(f"After features: {len(df_features)} rows")
print(f"   Range: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
print()

# Check if target timestamps exist AFTER feature calculation
print("="*80)
print("CHECK 2: Target Timestamps AFTER FEATURES")
print("="*80)
print()

for time_kst in target_times_kst:
    ts_kst = pd.Timestamp(time_kst)
    ts_utc = ts_kst - timedelta(hours=9)

    exact = df_features[df_features['timestamp'] == ts_utc]

    if len(exact) > 0:
        row = exact.iloc[0]
        print(f"✅ FOUND after features: {time_kst} KST ({ts_utc} UTC)")
        print(f"   Price: ${row['close']:,.2f}")
    else:
        print(f"❌ NOT FOUND after features: {time_kst} KST ({ts_utc} UTC)")
        # Find nearest
        df_features['time_diff'] = (df_features['timestamp'] - ts_utc).abs()
        nearest = df_features.nsmallest(1, 'time_diff').iloc[0]
        print(f"   Nearest: {nearest['timestamp']} (diff: {nearest['time_diff']})")
    print()

# Summary
print("="*80)
print("ANALYSIS")
print("="*80)
print()

print(f"Raw data range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
print(f"After features: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
print()

# Check how many rows were lost at the beginning
rows_lost = len(df_raw) - len(df_features)
time_lost = df_features['timestamp'].min() - df_raw['timestamp'].min()

print(f"Rows lost: {rows_lost}")
print(f"Time lost from start: {time_lost}")
print(f"Time lost in hours: {time_lost.total_seconds() / 3600:.1f}h")
print()

# Check if target time is within lost range
target_utc_1 = pd.Timestamp('2025-10-23 16:00:00')
if target_utc_1 < df_features['timestamp'].min():
    time_before_features = df_features['timestamp'].min() - target_utc_1
    print(f"❌ Target time (2025-10-24 01:00 KST) is {time_before_features} BEFORE feature data starts!")
    print(f"   Feature calculation lost this timestamp due to lookback requirements.")
else:
    print(f"✅ Target time is within feature data range")

print()
print("Diagnostic complete!")
