#!/usr/bin/env python3
"""
Compare API Data vs CSV Data
=============================

Compare OHLCV data from API with CSV to identify any differences
that could cause feature calculation discrepancies.
"""

import pandas as pd
import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

DATA_DIR = PROJECT_ROOT / "data" / "historical"
CONFIG_PATH = PROJECT_ROOT / "config" / "api_keys.yaml"

print("="*80)
print("API vs CSV DATA COMPARISON")
print("="*80)

# Load API keys
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key'],
    testnet=False
)

# Fetch recent 100 candles from API
print("\n1. Fetching from API...")
klines = client.get_klines("BTC-USDT", "5m", limit=100)
df_api = pd.DataFrame(klines)

# Rename 'time' to 'timestamp' for consistency
if 'time' in df_api.columns:
    df_api = df_api.rename(columns={'time': 'timestamp'})

df_api['timestamp'] = pd.to_datetime(df_api['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_api[col] = pd.to_numeric(df_api[col], errors='coerce')

print(f"  ✅ Fetched {len(df_api)} candles from API")
print(f"  Period: {df_api['timestamp'].min()} ~ {df_api['timestamp'].max()}")

# Load from CSV
print("\n2. Loading from CSV...")
df_csv = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])
print(f"  ✅ Loaded {len(df_csv):,} candles from CSV")

# Filter CSV to same period as API
min_time = df_api['timestamp'].min()
max_time = df_api['timestamp'].max()
df_csv_filtered = df_csv[
    (df_csv['timestamp'] >= min_time) &
    (df_csv['timestamp'] <= max_time)
].copy()
print(f"  ✅ Filtered to {len(df_csv_filtered)} candles (matching API period)")

# Merge on timestamp
df_merged = pd.merge(
    df_api,
    df_csv_filtered,
    on='timestamp',
    how='inner',
    suffixes=('_api', '_csv')
)

print(f"\n3. Comparison Results:")
print(f"  Matched timestamps: {len(df_merged)}/{len(df_api)}")

# Compare OHLCV
print(f"\n4. OHLCV Comparison:")
print(f"  {'Field':<10} {'Mean Diff':>15} {'Max Diff':>15} {'Identical':>10}")
print(f"  {'-'*50}")

for field in ['open', 'high', 'low', 'close', 'volume']:
    diff = df_merged[f'{field}_csv'] - df_merged[f'{field}_api']
    mean_diff = diff.mean()
    max_diff = diff.abs().max()
    identical = (diff.abs() < 0.01).sum()
    pct_identical = identical / len(df_merged) * 100

    print(f"  {field:<10} {mean_diff:>15.2f} {max_diff:>15.2f} {pct_identical:>9.1f}%")

# Show rows with differences
diff_threshold = 0.1
has_diff = (
    (df_merged['close_csv'] - df_merged['close_api']).abs() > diff_threshold
)

if has_diff.sum() > 0:
    print(f"\n⚠️  Found {has_diff.sum()} candles with close price difference > ${diff_threshold}")
    print(f"\nSample differences:")
    print(df_merged[has_diff][['timestamp', 'close_api', 'close_csv', 'volume_api', 'volume_csv']].head(10))
else:
    print(f"\n✅ All close prices match (within ${diff_threshold})")

print("\n" + "="*80)
