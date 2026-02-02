"""
NaN Issue Diagnosis

목표: 어떤 feature가 NaN을 생성하는지 파악
"""

import pandas as pd
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from src.api.bingx_client import BingXClient
import yaml

# Load API keys
CONFIG_DIR = PROJECT_ROOT / "config"
api_keys_file = CONFIG_DIR / "api_keys.yaml"
with open(api_keys_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    api_config = config.get('bingx', {}).get('testnet', {})

API_KEY = api_config.get('api_key')
API_SECRET = api_config.get('secret_key')

# Initialize client
client = BingXClient(
    api_key=API_KEY,
    secret_key=API_SECRET,
    testnet=True,
    timeout=30
)

print("=" * 80)
print("NaN Issue Diagnosis")
print("=" * 80)

# Get market data
print("\n1. Fetching market data...")
klines = client.get_klines(symbol="BTC-USDT", interval="5m", limit=400)
df = pd.DataFrame(klines)
df = df.rename(columns={'time': 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df[['open', 'high', 'low', 'close', 'volume']] = \
    df[['open', 'high', 'low', 'close', 'volume']].astype(float)
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"   Raw data: {len(df)} candles")
print(f"   Initial NaN count: {df.isnull().sum().sum()}")

# Calculate baseline features
print("\n2. Calculating baseline features (10)...")
df = calculate_features(df)
print(f"   After baseline: {len(df)} rows")

# Check NaN by column
nan_counts = df.isnull().sum()
nan_columns = nan_counts[nan_counts > 0].sort_values(ascending=False)
if len(nan_columns) > 0:
    print(f"   ❌ Columns with NaN after baseline:")
    for col, count in nan_columns.items():
        print(f"      {col}: {count} NaN ({count/len(df)*100:.1f}%)")
else:
    print(f"   ✅ No NaN after baseline features")

# Calculate advanced features
print("\n3. Calculating advanced features (27)...")
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
print(f"   After advanced: {len(df)} rows")

# Check NaN by column
nan_counts = df.isnull().sum()
nan_columns = nan_counts[nan_counts > 0].sort_values(ascending=False)
if len(nan_columns) > 0:
    print(f"   ❌ Columns with NaN after advanced features:")
    for col, count in nan_columns.items():
        print(f"      {col}: {count} NaN ({count/len(df)*100:.1f}%)")
else:
    print(f"   ✅ No NaN after advanced features")

# Apply ffill
print("\n4. Applying forward fill (ffill)...")
df_ffill = df.ffill()
nan_counts = df_ffill.isnull().sum()
nan_columns = nan_counts[nan_counts > 0].sort_values(ascending=False)
if len(nan_columns) > 0:
    print(f"   ❌ Columns with NaN after ffill:")
    for col, count in nan_columns.items():
        print(f"      {col}: {count} NaN ({count/len(df_ffill)*100:.1f}%)")
else:
    print(f"   ✅ No NaN after ffill")

# Apply dropna
print("\n5. Applying dropna()...")
df_clean = df_ffill.dropna()
rows_lost = len(df) - len(df_clean)
print(f"   Final data: {len(df_clean)} rows")
print(f"   Rows lost: {rows_lost} ({rows_lost/len(df)*100:.1f}%)")

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if rows_lost > 0:
    # Find which rows were dropped
    dropped_indices = set(df.index) - set(df_clean.index)
    print(f"\n❌ Problem: {rows_lost} rows dropped due to NaN")
    print(f"   Time lost: {rows_lost * 5} minutes = {rows_lost * 5 / 60:.1f} hours")
    print(f"\n   Dropped row indices: {sorted(list(dropped_indices))[:20]}...")

    # Check which columns had NaN in dropped rows
    dropped_df = df.loc[list(dropped_indices)]
    nan_in_dropped = dropped_df.isnull().sum()
    nan_in_dropped = nan_in_dropped[nan_in_dropped > 0].sort_values(ascending=False)

    print(f"\n   Columns with NaN in dropped rows:")
    for col, count in nan_in_dropped.head(10).items():
        print(f"      {col}: {count} NaN")

    # Check if NaN is at the beginning (warmup period issue)
    first_valid_idx = df_clean.index[0] if len(df_clean) > 0 else len(df)
    if first_valid_idx > 0:
        print(f"\n   ⚠️ First valid row: index {first_valid_idx}")
        print(f"   → NaN concentrated at beginning (warmup period issue)")
        print(f"\n   Root Cause:")
        print(f"   - Technical indicators need historical data to calculate")
        print(f"   - Some indicators (ATR, MACD, etc.) require 20-50 candles warmup")
        print(f"   - 400 candles requested, but first {first_valid_idx} are invalid")

        print(f"\n   Solutions:")
        print(f"   1. ✅ Increase LOOKBACK_CANDLES: 400 → 500")
        print(f"      → Ensures 400+ valid candles after warmup")
        print(f"   2. Optimize indicators to reduce warmup period")
        print(f"   3. Use bfill() for initial rows (risky - creates lookahead bias)")
    else:
        print(f"\n   ⚠️ NaN scattered throughout data (data quality issue)")
        print(f"   → Check API data quality or indicator calculation bugs")
else:
    print(f"\n✅ No rows lost to NaN!")
    print(f"   All 400 candles are valid after feature calculation")

print("\n" + "=" * 80)
print("Diagnosis Complete")
print("=" * 80)
