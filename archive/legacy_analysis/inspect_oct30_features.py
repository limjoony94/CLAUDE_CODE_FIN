"""
Inspect Oct 30 09:45 Candle Features
====================================
Check for NaN, inf, or suspicious feature values
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

print("=" * 80)
print("INSPECT OCT 30 09:45 CANDLE FEATURES")
print("=" * 80)
print()

# Load data
df = pd.read_csv(RESULTS_DIR / "BTCUSDT_5m_latest4weeks_PRODUCTION_FEATURES.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Load model features
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"SHORT Exit Model: {len(short_exit_features)} features")
print()

# Find the candle
target_time = pd.to_datetime("2025-10-30 09:45:00")
candle = df[df['timestamp'] == target_time]

if candle.empty:
    print(f"❌ Candle not found at {target_time}")
else:
    print(f"✅ Candle found at {target_time}")
    print(f"   Index: {candle.index[0]}")
    print(f"   Close: ${candle.iloc[0]['close']:,.2f}")
    print()

    # Extract features
    features = candle[short_exit_features].iloc[0]

    # Check for NaN values
    nan_features = features[features.isna()]
    if len(nan_features) > 0:
        print(f"⚠️ Found {len(nan_features)} NaN features:")
        for feat_name in nan_features.index:
            print(f"   - {feat_name}")
        print()
    else:
        print("✅ No NaN features")
        print()

    # Check for inf values
    inf_features = features[np.isinf(features)]
    if len(inf_features) > 0:
        print(f"⚠️ Found {len(inf_features)} inf features:")
        for feat_name in inf_features.index:
            print(f"   - {feat_name}: {features[feat_name]}")
        print()
    else:
        print("✅ No inf features")
        print()

    # Show all feature values
    print("=" * 80)
    print("ALL FEATURE VALUES")
    print("=" * 80)
    print()
    for feat_name in short_exit_features:
        value = features[feat_name]
        if pd.isna(value):
            print(f"{feat_name:30s} = NaN")
        elif np.isinf(value):
            print(f"{feat_name:30s} = inf")
        else:
            print(f"{feat_name:30s} = {value:.6f}")

    print()
    print("=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)
    print()

    # Basic stats
    print(f"Min value: {features.min():.6f}")
    print(f"Max value: {features.max():.6f}")
    print(f"Mean value: {features.mean():.6f}")
    print(f"Std value: {features.std():.6f}")

    # Count extreme values
    extreme_low = (features < -100).sum()
    extreme_high = (features > 100).sum()

    if extreme_low > 0:
        print(f"\n⚠️ {extreme_low} features with values < -100:")
        extreme_feats = features[features < -100]
        for feat_name in extreme_feats.index:
            print(f"   - {feat_name}: {features[feat_name]:.2f}")

    if extreme_high > 0:
        print(f"\n⚠️ {extreme_high} features with values > 100:")
        extreme_feats = features[features > 100]
        for feat_name in extreme_feats.index:
            print(f"   - {feat_name}: {features[feat_name]:.2f}")

    print()
    print("=" * 80)
