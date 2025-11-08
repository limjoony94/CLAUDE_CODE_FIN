"""
CRITICAL DEBUG: Feature Generation & Data Integrity Analysis
=============================================================
Check if features are being generated correctly in production vs backtest
"""
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

print("="*80)
print("🔍 CRITICAL DEBUG: Feature Generation & Data Integrity")
print("="*80)

# Load model features
print("\nLoading model feature list...")
long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Model expects {len(long_feature_columns)} features")

# 1. CHECK PRODUCTION LOGS FOR ACTUAL FEATURE VALUES
print("\n" + "="*80)
print("1️⃣ ANALYZING PRODUCTION BOT FEATURE VALUES")
print("="*80)

# Parse production logs to extract feature values
log_file = LOGS_DIR / "bot_restart_20251021.log"
print(f"\nReading production log: {log_file}")

feature_samples = []
nan_warnings = []

with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

    for i, line in enumerate(lines):
        # Look for feature generation debug logs
        if "Feature sample (first 10):" in line:
            # Extract the actual values
            match = re.search(r'\[(.*?)\]', line)
            if match:
                values_str = match.group(1)
                try:
                    values = [float(x.strip()) for x in values_str.split()]
                    feature_samples.append(values[:10])
                except:
                    pass

        # Look for NaN warnings
        if "NaN" in line or "nan" in line.lower():
            nan_warnings.append(line.strip())

print(f"\n📊 Production Feature Samples Found: {len(feature_samples)}")

if feature_samples:
    print("\nSample feature values (first 10 features):")
    for i, sample in enumerate(feature_samples[-3:]):  # Last 3 samples
        print(f"  Sample {i+1}: {sample}")

        # Check for NaN or infinity
        has_nan = any(np.isnan(x) or np.isinf(x) for x in sample)
        if has_nan:
            print(f"    ⚠️  WARNING: Contains NaN or Inf!")

print(f"\n⚠️  NaN Warnings in Production Log: {len(nan_warnings)}")
if nan_warnings:
    print("First 5 warnings:")
    for warning in nan_warnings[:5]:
        print(f"  {warning}")

# 2. GENERATE FEATURES FROM SAME DATA AND COMPARE
print("\n" + "="*80)
print("2️⃣ GENERATING FEATURES FROM BACKTEST DATA")
print("="*80)

# Load recent data
data_path = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Take last 1000 candles (same as production bot)
df = df.tail(1000).copy()
print(f"\n✅ Loaded {len(df)} candles (last 1000)")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Calculate features
print("\nCalculating features...")
df_with_features = calculate_all_features(df)
print(f"✅ Features calculated")

# 3. CHECK FOR NaN VALUES
print("\n" + "="*80)
print("3️⃣ NaN ANALYSIS IN GENERATED FEATURES")
print("="*80)

print("\nChecking model features for NaN values...")
nan_counts = {}
for feature in long_feature_columns:
    if feature in df_with_features.columns:
        nan_count = df_with_features[feature].isna().sum()
        nan_pct = (nan_count / len(df_with_features)) * 100
        if nan_count > 0:
            nan_counts[feature] = {'count': nan_count, 'pct': nan_pct}

if nan_counts:
    print(f"\n⚠️  Found {len(nan_counts)} features with NaN values:")
    sorted_nans = sorted(nan_counts.items(), key=lambda x: x[1]['pct'], reverse=True)
    for feature, stats in sorted_nans[:20]:  # Top 20
        print(f"  {feature}: {stats['count']} NaN ({stats['pct']:.1f}%)")
else:
    print("\n✅ No NaN values found in model features")

# 4. CHECK FEATURE AVAILABILITY
print("\n" + "="*80)
print("4️⃣ FEATURE AVAILABILITY CHECK")
print("="*80)

print("\nChecking if all model features exist in generated data...")
missing_features = []
for feature in long_feature_columns:
    if feature not in df_with_features.columns:
        missing_features.append(feature)

if missing_features:
    print(f"\n⚠️  CRITICAL: {len(missing_features)} features MISSING from generated data!")
    print("Missing features:")
    for feature in missing_features:
        print(f"  - {feature}")
else:
    print("\n✅ All model features present in generated data")

# 5. COMPARE ACTUAL VALUES: LAST ROW
print("\n" + "="*80)
print("5️⃣ ACTUAL FEATURE VALUES COMPARISON")
print("="*80)

print("\nExtracting last row features (most recent)...")
last_row = df_with_features.iloc[-1]

print(f"\nLast candle timestamp: {last_row['timestamp']}")
print(f"Last candle close: ${last_row['close']:,.2f}")

# Extract feature values in order
feature_values = []
for feature in long_feature_columns:
    if feature in df_with_features.columns:
        value = last_row[feature]
        feature_values.append(value)
    else:
        feature_values.append(np.nan)

print("\nFirst 10 model features (backtest calculation):")
for i, (feature, value) in enumerate(zip(long_feature_columns[:10], feature_values[:10])):
    if np.isnan(value):
        print(f"  {i+1}. {feature}: NaN ⚠️")
    elif np.isinf(value):
        print(f"  {i+1}. {feature}: Inf ⚠️")
    else:
        print(f"  {i+1}. {feature}: {value:.6f}")

# 6. CHECK SCALER IMPACT
print("\n" + "="*80)
print("6️⃣ SCALER TRANSFORMATION CHECK")
print("="*80)

print("\nLoading scaler...")
long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

# Check scaler parameters
print(f"\nScaler type: {type(long_scaler)}")
if hasattr(long_scaler, 'mean_'):
    print(f"Scaler mean (first 10): {long_scaler.mean_[:10]}")
    print(f"Scaler scale (first 10): {long_scaler.scale_[:10]}")

# Try to scale the features
print("\nAttempting to scale features...")
try:
    feature_array = np.array(feature_values).reshape(1, -1)

    # Check for NaN before scaling
    nan_mask = np.isnan(feature_array)
    if nan_mask.any():
        nan_indices = np.where(nan_mask[0])[0]
        print(f"\n⚠️  WARNING: {len(nan_indices)} NaN values before scaling!")
        print(f"   NaN at indices: {nan_indices[:10].tolist()}")
        print(f"   Corresponding features:")
        for idx in nan_indices[:10]:
            print(f"     - {long_feature_columns[idx]}")

    scaled_features = long_scaler.transform(feature_array)

    # Check for NaN after scaling
    nan_mask_after = np.isnan(scaled_features)
    if nan_mask_after.any():
        print(f"\n⚠️  CRITICAL: NaN values AFTER scaling!")
    else:
        print(f"\n✅ Scaling successful, no NaN after scaling")
        print(f"   Scaled values (first 10): {scaled_features[0][:10]}")

except Exception as e:
    print(f"\n❌ ERROR during scaling: {e}")

# 7. COMPARE WITH PRODUCTION LOG VALUES
print("\n" + "="*80)
print("7️⃣ PRODUCTION vs BACKTEST COMPARISON")
print("="*80)

if feature_samples:
    print("\nComparing last production sample with backtest calculation:")
    print("\n{:<30} {:>15} {:>15} {:>15}".format("Feature", "Production", "Backtest", "Difference"))
    print("-" * 80)

    last_prod_sample = feature_samples[-1]
    backtest_sample = feature_values[:10]

    for i in range(min(len(last_prod_sample), len(backtest_sample))):
        prod_val = last_prod_sample[i]
        back_val = backtest_sample[i]

        if np.isnan(back_val) or np.isnan(prod_val):
            diff_str = "NaN"
            status = "⚠️"
        elif np.isinf(back_val) or np.isinf(prod_val):
            diff_str = "Inf"
            status = "⚠️"
        else:
            diff = abs(prod_val - back_val)
            if abs(back_val) > 1e-10:
                diff_pct = (diff / abs(back_val)) * 100
                if diff_pct > 10:
                    status = "⚠️"
                else:
                    status = "✅"
                diff_str = f"{diff_pct:.1f}%"
            else:
                status = "✅" if diff < 1e-6 else "⚠️"
                diff_str = f"{diff:.2e}"

        print(f"{status} {long_feature_columns[i]:<27} {prod_val:>14.6f} {back_val:>14.6f} {diff_str:>14}")

# 8. SUMMARY
print("\n" + "="*80)
print("💡 DIAGNOSTIC SUMMARY")
print("="*80)

issues_found = []

if missing_features:
    issues_found.append(f"⚠️  {len(missing_features)} features MISSING from calculation")

if nan_counts:
    high_nan_features = [f for f, s in nan_counts.items() if s['pct'] > 10]
    if high_nan_features:
        issues_found.append(f"⚠️  {len(high_nan_features)} features with >10% NaN values")

if nan_warnings:
    issues_found.append(f"⚠️  {len(nan_warnings)} NaN warnings in production log")

if not issues_found:
    print("\n✅ No critical issues detected in feature generation")
    print("   - All features present")
    print("   - No excessive NaN values")
    print("   - Feature calculations appear consistent")
else:
    print("\n⚠️  ISSUES DETECTED:")
    for issue in issues_found:
        print(f"   {issue}")

print("\n" + "="*80)
