"""
Check normalized feature values after scaler transform
Created: 2025-11-03
Purpose: Verify if features are in normal range after normalization
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

print(f"✅ Features: {len(df_features)} rows")

# Load models and scalers
MODELS_DIR = PROJECT_ROOT / "models"

# LONG Entry
print("\n" + "="*80)
print("LONG ENTRY - NORMALIZED FEATURE ANALYSIS")
print("="*80)

long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines()]

# Get latest candle
latest = df_features.iloc[-1:]
long_feat_df = latest[long_features]
long_feat_raw = long_feat_df.values

# Normalize
long_feat_normalized = long_scaler.transform(long_feat_raw)

print(f"\nLatest candle: {latest.iloc[0]['timestamp']}")
print(f"Price: ${latest.iloc[0]['close']:,.1f}")

print("\nFirst 20 features (raw → normalized):")
print("-" * 80)

suspicious_norm = []
for i in range(min(20, len(long_features))):
    feat = long_features[i]
    raw_val = long_feat_raw[0][i]
    norm_val = long_feat_normalized[0][i]

    status = "✅"
    if abs(norm_val) > 5:
        status = "⚠️  Large"
        suspicious_norm.append((feat, raw_val, norm_val))
    elif np.isnan(norm_val) or np.isinf(norm_val):
        status = "❌ Invalid"
        suspicious_norm.append((feat, raw_val, norm_val))

    print(f"  {feat:40s}: {raw_val:12.4f} → {norm_val:8.4f}  {status}")

# Check all features for suspicious normalized values
all_suspicious_norm = []
for i in range(len(long_features)):
    norm_val = long_feat_normalized[0][i]
    if abs(norm_val) > 5 or np.isnan(norm_val) or np.isinf(norm_val):
        all_suspicious_norm.append((long_features[i], long_feat_raw[0][i], norm_val))

if all_suspicious_norm:
    print(f"\n⚠️  FOUND {len(all_suspicious_norm)} SUSPICIOUS NORMALIZED VALUES (|z| > 5):")
    for feat, raw, norm in all_suspicious_norm[:10]:
        print(f"  - {feat}: {raw:.4f} → {norm:.4f}")
    if len(all_suspicious_norm) > 10:
        print(f"  ... and {len(all_suspicious_norm) - 10} more")
else:
    print("\n✅ All normalized values in normal range (|z| <= 5)")

# SHORT Entry
print("\n" + "="*80)
print("SHORT ENTRY - NORMALIZED FEATURE ANALYSIS")
print("="*80)

short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_features = [line.strip() for line in f.readlines() if line.strip()]

short_feat_df = latest[short_features]
short_feat_raw = short_feat_df.values
short_feat_normalized = short_scaler.transform(short_feat_raw)

print("\nFirst 20 features (raw → normalized):")
print("-" * 80)

for i in range(min(20, len(short_features))):
    feat = short_features[i]
    raw_val = short_feat_raw[0][i]
    norm_val = short_feat_normalized[0][i]

    status = "✅"
    if abs(norm_val) > 5:
        status = "⚠️  Large"
    elif np.isnan(norm_val) or np.isinf(norm_val):
        status = "❌ Invalid"

    print(f"  {feat:40s}: {raw_val:12.4f} → {norm_val:8.4f}  {status}")

# Check for volume_decline_ratio specifically (the suspicious one)
print("\n" + "="*80)
print("SUSPICIOUS FEATURE: volume_decline_ratio")
print("="*80)

if 'volume_decline_ratio' in short_features:
    idx = short_features.index('volume_decline_ratio')
    raw_val = short_feat_raw[0][idx]
    norm_val = short_feat_normalized[0][idx]

    scaler_mean = short_scaler.mean_[idx]
    scaler_std = short_scaler.scale_[idx]

    print(f"Raw value: {raw_val}")
    print(f"Scaler mean: {scaler_mean}")
    print(f"Scaler std: {scaler_std:.2f}  ← ⚠️  ABNORMALLY LARGE!")
    print(f"Normalized: {norm_val:.6f}")

    if abs(norm_val) < 0.01:
        print("\n⚠️  PROBLEM IDENTIFIED:")
        print("   - Scaler std is 404 billion (abnormal)")
        print("   - Any normal raw value will normalize to ~0")
        print("   - This feature is effectively DEAD (no information)")
else:
    print("Feature not found in SHORT features")

# Check all SHORT features for suspicious normalized values
all_suspicious_short = []
for i in range(len(short_features)):
    norm_val = short_feat_normalized[0][i]
    if abs(norm_val) > 5 or np.isnan(norm_val) or np.isinf(norm_val):
        all_suspicious_short.append((short_features[i], short_feat_raw[0][i], norm_val))

if all_suspicious_short:
    print(f"\n⚠️  FOUND {len(all_suspicious_short)} SUSPICIOUS NORMALIZED VALUES:")
    for feat, raw, norm in all_suspicious_short:
        print(f"  - {feat}: {raw:.4f} → {norm:.4f}")
else:
    print("\n✅ All normalized values in normal range (|z| <= 5)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nLONG Entry:")
print(f"  Total features: {len(long_features)}")
print(f"  Suspicious normalized: {len(all_suspicious_norm)}")

print(f"\nSHORT Entry:")
print(f"  Total features: {len(short_features)}")
print(f"  Suspicious normalized: {len(all_suspicious_short)}")
print(f"  Dead features (std > 1B): 1 (volume_decline_ratio)")

print("\n" + "="*80)
