"""
Check feature values for suspicious patterns
Created: 2025-11-03
Purpose: Investigate over-trading issue - check if features have unusual values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features
import yaml

# Load API keys
config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Use mainnet keys
bingx_config = config['bingx']['mainnet']

# Initialize client
client = BingXClient(
    api_key=bingx_config['api_key'],
    secret_key=bingx_config['secret_key']
)

# Fetch recent data
print("Fetching 1000 recent candles...")
klines = client.get_klines("BTC-USDT", "5m", limit=1000)
df = pd.DataFrame(klines)

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Fetched {len(df)} candles")
print(f"   Time range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

# Calculate features
print("\nCalculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)

print(f"✅ Features calculated: {len(df_features)} rows, {df_features.shape[1]} columns")

# Check for suspicious patterns
print("\n" + "="*80)
print("FEATURE VALUE ANALYSIS")
print("="*80)

# Load LONG feature names
models_dir = PROJECT_ROOT / "models"
with open(models_dir / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines()]

print(f"\nAnalyzing {len(long_features)} LONG entry features...")

# Get latest candle features
latest = df_features.iloc[-1]

# Check for suspicious values
suspicious_count = 0
zero_count = 0
nan_count = 0
inf_count = 0

print(f"\nLATEST CANDLE ({latest['timestamp']}):")
print("-" * 80)

for feat in long_features[:20]:  # Check first 20 features
    if feat in latest.index:
        val = latest[feat]
        status = "✅"

        if pd.isna(val):
            status = "❌ NaN"
            nan_count += 1
            suspicious_count += 1
        elif np.isinf(val):
            status = "❌ Inf"
            inf_count += 1
            suspicious_count += 1
        elif val == 0.0:
            status = "⚠️  Zero"
            zero_count += 1

        print(f"  {feat:40s}: {val:12.6f}  {status}")
    else:
        print(f"  {feat:40s}: MISSING ❌")
        suspicious_count += 1

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS (First 20 features)")
print("="*80)
print(f"  NaN values: {nan_count}")
print(f"  Inf values: {inf_count}")
print(f"  Zero values: {zero_count}")
print(f"  Suspicious total: {suspicious_count}")

# Check all features for NaN/Inf
all_suspicious = []
for feat in long_features:
    if feat in latest.index:
        val = latest[feat]
        if pd.isna(val):
            all_suspicious.append((feat, "NaN"))
        elif np.isinf(val):
            all_suspicious.append((feat, "Inf"))

if all_suspicious:
    print(f"\n⚠️  FOUND {len(all_suspicious)} SUSPICIOUS VALUES IN ALL FEATURES:")
    for feat, issue in all_suspicious[:10]:  # Show first 10
        print(f"  - {feat}: {issue}")
    if len(all_suspicious) > 10:
        print(f"  ... and {len(all_suspicious) - 10} more")
else:
    print("\n✅ No NaN or Inf values found in any features")

# Check feature value ranges
print("\n" + "="*80)
print("FEATURE VALUE RANGES (Recent 10 candles)")
print("="*80)

recent = df_features.iloc[-10:]
for feat in long_features[:10]:  # Check first 10 features
    if feat in recent.columns:
        values = recent[feat]
        print(f"\n{feat}:")
        print(f"  Min: {values.min():12.6f}")
        print(f"  Max: {values.max():12.6f}")
        print(f"  Mean: {values.mean():12.6f}")
        print(f"  Std: {values.std():12.6f}")
        print(f"  Unique: {values.nunique()} values")

        # Check if constant
        if values.nunique() == 1:
            print(f"  ⚠️  CONSTANT VALUE!")

print("\n" + "="*80)
print("DONE")
print("="*80)
