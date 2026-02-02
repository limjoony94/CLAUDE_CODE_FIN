"""
Check top importance features for outliers
Created: 2025-11-03
Purpose: Find features driving persistent high LONG probabilities
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Load API keys
config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

bingx_config = config['bingx']['mainnet']
client = BingXClient(
    api_key=bingx_config['api_key'],
    secret_key=bingx_config['secret_key']
)

# Fetch data
print("Fetching 1000 candles...")
klines = client.get_klines("BTC-USDT", "5m", limit=1000)
df = pd.DataFrame(klines)

df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Fetched {len(df)} candles")

# Calculate features
print("Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)

print(f"✅ Features: {len(df_features)} rows\n")

# Load LONG Entry model components
MODELS_DIR = PROJECT_ROOT / "models"
long_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines()]

# Get feature importance
importance = long_model.feature_importances_
importance_sorted = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)

# Top 10 features
print("="*80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*80)

top_10_indices = []
top_10_names = []

for rank, (idx, imp) in enumerate(importance_sorted[:10], 1):
    feat_name = long_features[idx]
    top_10_indices.append(idx)
    top_10_names.append(feat_name)
    print(f"{rank:2d}. {feat_name:35s}: {imp:.6f}")

# Analyze recent 10 candles for top features
print("\n" + "="*80)
print("TOP FEATURES ANALYSIS - RECENT 10 CANDLES")
print("="*80)

recent = df_features.iloc[-10:]

print(f"\nTime range: {recent.iloc[0]['timestamp']} to {recent.iloc[-1]['timestamp']}")
print(f"Price range: ${recent.iloc[0]['close']:,.1f} to ${recent.iloc[-1]['close']:,.1f}")

# Collect outliers
all_outliers = []

for rank, (idx, feat_name) in enumerate(zip(top_10_indices, top_10_names), 1):
    print(f"\n{'-'*80}")
    print(f"Feature #{rank}: {feat_name} (importance: {importance[idx]:.6f})")
    print(f"{'-'*80}")

    scaler_mean = long_scaler.mean_[idx]
    scaler_std = long_scaler.scale_[idx]

    print(f"Scaler: mean={scaler_mean:.6f}, std={scaler_std:.6f}")
    print(f"\nRecent values:")

    outlier_count = 0
    extreme_count = 0

    for i in range(len(recent)):
        candle = recent.iloc[i:i+1]
        timestamp = candle['timestamp'].iloc[0]

        # Get raw value
        raw_val = candle[feat_name].iloc[0]

        # Get normalized value
        feat_df = candle[long_features]
        feat_normalized = long_scaler.transform(feat_df.values)
        z_score = feat_normalized[0][idx]

        # Check if outlier
        if abs(z_score) > 5:
            status = "🚨 EXTREME"
            extreme_count += 1
            all_outliers.append((feat_name, rank, z_score, raw_val))
        elif abs(z_score) > 3:
            status = "⚠️  OUTLIER"
            outlier_count += 1
            all_outliers.append((feat_name, rank, z_score, raw_val))
        else:
            status = "✅"

        print(f"  {i+1:2d}. {timestamp} | Raw: {raw_val:12.4f} | Z: {z_score:7.2f}  {status}")

    if extreme_count > 0 or outlier_count > 0:
        print(f"\n  ⚠️  Outliers detected: {outlier_count} (|Z|>3), {extreme_count} (|Z|>5)")

# Summary
print("\n" + "="*80)
print("OUTLIER SUMMARY")
print("="*80)

if all_outliers:
    print(f"\n⚠️  FOUND {len(all_outliers)} OUTLIERS IN TOP 10 FEATURES:")
    print(f"\nGrouped by feature:")
    print("-" * 80)

    from collections import defaultdict
    outliers_by_feature = defaultdict(list)

    for feat_name, rank, z_score, raw_val in all_outliers:
        outliers_by_feature[feat_name].append((z_score, raw_val))

    for feat_name, values in sorted(outliers_by_feature.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{feat_name}:")
        print(f"  Outliers: {len(values)} / 10 candles ({len(values)/10*100:.1f}%)")
        for z, raw in values:
            print(f"    Z={z:+7.2f}, Raw={raw:12.4f}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    # Find persistent outliers
    persistent = {k: v for k, v in outliers_by_feature.items() if len(v) >= 5}

    if persistent:
        print(f"\n🚨 PERSISTENT OUTLIERS DETECTED ({len(persistent)} features):")
        for feat_name, values in persistent.items():
            print(f"\n  {feat_name}:")
            print(f"    - {len(values)}/10 candles are outliers ({len(values)/10*100:.1f}%)")
            print(f"    - This feature is CONSISTENTLY producing extreme values")
            print(f"    - Likely MAIN DRIVER of high LONG probabilities!")
    else:
        print(f"\n⚠️  INTERMITTENT OUTLIERS:")
        print(f"    - Outliers present but not persistent (< 50% of candles)")
        print(f"    - May contribute to occasional probability spikes")
        print(f"    - But NOT causing sustained high probabilities")
else:
    print("\n✅ NO OUTLIERS DETECTED IN TOP 10 FEATURES")
    print("   All features within normal range (|Z| <= 3)")
    print("   High probabilities NOT caused by feature outliers")
    print("   Possible causes:")
    print("   - Market conditions genuinely match training patterns")
    print("   - Model properly detecting high-probability setups")

print("\n" + "="*80)
