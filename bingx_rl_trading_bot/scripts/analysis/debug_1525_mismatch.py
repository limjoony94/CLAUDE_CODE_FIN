#!/usr/bin/env python3
"""
Debug 15:25 Signal Mismatch
Detailed feature-level analysis for the outlier timepoint
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

print("=" * 80)
print("DEBUG: 15:25 Signal Mismatch Analysis")
print("=" * 80)

# Target timepoint
TARGET_UTC = '2025-10-26 15:25:00'
TARGET_KST = '2025-10-27 00:25:00'

PROD_LONG = 0.1911
PROD_SHORT = 0.2940

print(f"\n📍 Target Timepoint:")
print(f"   UTC: {TARGET_UTC}")
print(f"   KST: {TARGET_KST}")
print(f"   Production LONG: {PROD_LONG:.4f}")
print(f"   Production SHORT: {PROD_SHORT:.4f}")

# Load CSV
CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
print(f"\n1. Loading CSV...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter with lookback
from datetime import timedelta
target_time = pd.to_datetime(TARGET_UTC)
lookback_minutes = 1000 * 5

df_filtered = df[
    (df['timestamp'] >= target_time - timedelta(minutes=lookback_minutes)) &
    (df['timestamp'] <= target_time)
].copy()

print(f"   ✅ Loaded {len(df_filtered)} candles")
print(f"   Range: {df_filtered['timestamp'].min()} ~ {df_filtered['timestamp'].max()}")

# Calculate features
print(f"\n2. Calculating features...")
df_features = calculate_all_features_enhanced_v2(df_filtered, phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"   ✅ Features: {len(df_features)} rows, {len(df_features.columns)} columns")

# Find target row
target_row = df_features[df_features['timestamp'] == target_time]
if len(target_row) == 0:
    print(f"\n❌ Target timepoint not found in features!")
    print(f"   Available range: {df_features['timestamp'].min()} ~ {df_features['timestamp'].max()}")
    sys.exit(1)

target_row = target_row.iloc[0]
print(f"\n3. Target row found:")
print(f"   Close: ${target_row['close']:.1f}")
print(f"   Volume: {target_row['volume']:.0f}")

# Load models
print(f"\n4. Loading models...")
MODELS_DIR = PROJECT_ROOT / "models"

long_model = pickle.load(open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb'))
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines()]

short_model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb'))
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_features = [line.strip() for line in f.readlines()]

print(f"   ✅ LONG: {len(long_features)} features")
print(f"   ✅ SHORT: {len(short_features)} features")

# Check for NaN/Inf in features
print(f"\n5. Checking for NaN/Inf in features...")
long_feat_vals = target_row[long_features]
short_feat_vals = target_row[short_features]

long_nan_count = long_feat_vals.isna().sum()
long_inf_count = np.isinf(pd.to_numeric(long_feat_vals, errors='coerce')).sum()
short_nan_count = short_feat_vals.isna().sum()
short_inf_count = np.isinf(pd.to_numeric(short_feat_vals, errors='coerce')).sum()

print(f"   LONG features:")
print(f"     NaN: {long_nan_count}")
print(f"     Inf: {long_inf_count}")
print(f"   SHORT features:")
print(f"     NaN: {short_nan_count}")
print(f"     Inf: {short_inf_count}")

if long_nan_count > 0:
    print(f"\n   ⚠️ LONG NaN features:")
    for feat in long_features:
        if pd.isna(target_row[feat]):
            print(f"      {feat}: NaN")

if short_nan_count > 0:
    print(f"\n   ⚠️ SHORT NaN features:")
    for feat in short_features:
        if pd.isna(target_row[feat]):
            print(f"      {feat}: NaN")

# Generate predictions
print(f"\n6. Generating predictions...")
long_X = target_row[long_features].values.reshape(1, -1)
short_X = target_row[short_features].values.reshape(1, -1)

# Replace NaN with 0 (same as production might do)
long_X_clean = np.nan_to_num(long_X, nan=0.0, posinf=0.0, neginf=0.0)
short_X_clean = np.nan_to_num(short_X, nan=0.0, posinf=0.0, neginf=0.0)

long_X_scaled = long_scaler.transform(long_X_clean)
short_X_scaled = short_scaler.transform(short_X_clean)

long_pred = long_model.predict_proba(long_X_scaled)[0, 1]
short_pred = short_model.predict_proba(short_X_scaled)[0, 1]

print(f"   Backtest LONG:  {long_pred:.4f}")
print(f"   Backtest SHORT: {short_pred:.4f}")

# Compare with production
print(f"\n7. Comparison with production:")
print("=" * 80)
print(f"{'Signal':<10} {'Production':>12} {'Backtest':>12} {'Diff':>12} {'Diff %':>10}")
print("=" * 80)

long_diff = PROD_LONG - long_pred
short_diff = PROD_SHORT - short_pred

print(f"{'LONG':<10} {PROD_LONG:>12.4f} {long_pred:>12.4f} {long_diff:>+12.4f} {long_diff/long_pred*100:>9.2f}%")
print(f"{'SHORT':<10} {PROD_SHORT:>12.4f} {short_pred:>12.4f} {short_diff:>+12.4f} {short_diff/short_pred*100:>9.2f}%")
print("=" * 80)

# Show top features by importance
print(f"\n8. Sample feature values:")
print("=" * 80)
print(f"{'Feature':<40} {'Value':>15} {'Has NaN':>10}")
print("=" * 80)

for feat in long_features[:20]:  # First 20 features
    val = target_row[feat]
    is_nan = "⚠️ NaN" if pd.isna(val) else "✅"
    if pd.isna(val):
        print(f"{feat:<40} {'NaN':>15} {is_nan:>10}")
    else:
        print(f"{feat:<40} {val:>15.6f} {is_nan:>10}")

print("=" * 80)
print("\n📊 SUMMARY:")
if long_nan_count > 0 or short_nan_count > 0:
    print("⚠️ NaN values detected in features!")
    print("   → This could explain the mismatch")
    print("   → Production might handle NaN differently than backtest")
else:
    print("✅ No NaN values in features")
    print("   → Mismatch likely due to:")
    print("      1. Different candle data in lookback window (CSV vs API)")
    print("      2. Different feature calculation due to data differences")
    print("      3. Rounding/precision differences")

print("\n" + "=" * 80)
