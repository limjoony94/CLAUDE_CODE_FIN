"""
Root Cause Analysis: Feature Distribution Changes
==================================================
Analyze which features are causing the abnormal signal frequency
"""
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("🔬 ROOT CAUSE: Feature Distribution Analysis")
print("="*80)

# Load model and features
print("\nLoading LONG model...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Model loaded with {len(long_feature_columns)} features")

# Load data
print("\nLoading data...")
data_path = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_path)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
latest_date = df_full['timestamp'].max()

# Compare two periods
period_recent = (latest_date - timedelta(days=2), latest_date)
period_historical = (latest_date - timedelta(days=30), latest_date - timedelta(days=28))

print(f"\n📅 Periods:")
print(f"  Recent: {period_recent[0]} to {period_recent[1]}")
print(f"  Historical: {period_historical[0]} to {period_historical[1]}")

# Get data for both periods
df_recent = df_full[(df_full['timestamp'] >= period_recent[0]) &
                    (df_full['timestamp'] <= period_recent[1])].copy()
df_historical = df_full[(df_full['timestamp'] >= period_historical[0]) &
                        (df_full['timestamp'] <= period_historical[1])].copy()

print(f"\nRecent candles: {len(df_recent)}")
print(f"Historical candles: {len(df_historical)}")

# Calculate features
print("\nCalculating features for recent period...")
df_recent = calculate_all_features(df_recent)

print("Calculating features for historical period...")
df_historical = calculate_all_features(df_historical)

# Compare feature distributions
print("\n" + "="*80)
print("📊 FEATURE DISTRIBUTION COMPARISON")
print("="*80)

feature_changes = []

for feature in long_feature_columns:
    if feature not in df_recent.columns or feature not in df_historical.columns:
        continue

    recent_values = df_recent[feature].dropna()
    historical_values = df_historical[feature].dropna()

    if len(recent_values) == 0 or len(historical_values) == 0:
        continue

    recent_mean = recent_values.mean()
    historical_mean = historical_values.mean()
    recent_std = recent_values.std()
    historical_std = historical_values.std()

    # Calculate percentage change
    if abs(historical_mean) > 1e-10:
        pct_change = ((recent_mean - historical_mean) / abs(historical_mean)) * 100
    else:
        pct_change = 0

    # Calculate standardized difference
    if historical_std > 1e-10:
        std_diff = (recent_mean - historical_mean) / historical_std
    else:
        std_diff = 0

    feature_changes.append({
        'feature': feature,
        'recent_mean': recent_mean,
        'historical_mean': historical_mean,
        'pct_change': pct_change,
        'std_diff': std_diff,
        'abs_std_diff': abs(std_diff)
    })

df_changes = pd.DataFrame(feature_changes)
df_changes = df_changes.sort_values('abs_std_diff', ascending=False)

print("\n🔝 TOP 15 FEATURES WITH LARGEST CHANGES (by standardized difference):")
print("="*80)
top_features = df_changes.head(15)
for idx, row in top_features.iterrows():
    print(f"\n{row['feature']}:")
    print(f"  Recent Mean:     {row['recent_mean']:.6f}")
    print(f"  Historical Mean: {row['historical_mean']:.6f}")
    print(f"  Change:          {row['pct_change']:+.1f}%")
    print(f"  Std Difference:  {row['std_diff']:+.2f} σ")

# Get feature importances from model
print("\n" + "="*80)
print("🎯 FEATURE IMPORTANCE ANALYSIS")
print("="*80)

try:
    # Try to get feature importances
    importances = long_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': long_feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Cross-reference: Important features that changed significantly
    print("\n" + "="*80)
    print("⚠️  CRITICAL: Important Features That Changed Significantly")
    print("="*80)

    top_important = set(feature_importance.head(20)['feature'])
    top_changed = set(df_changes.head(20)['feature'])
    critical_features = top_important & top_changed

    if critical_features:
        print(f"\nFound {len(critical_features)} critical features:")
        for feature in critical_features:
            imp_row = feature_importance[feature_importance['feature'] == feature].iloc[0]
            change_row = df_changes[df_changes['feature'] == feature].iloc[0]

            print(f"\n📍 {feature}:")
            print(f"   Importance: {imp_row['importance']:.4f} (Rank #{feature_importance[feature_importance['feature'] == feature].index[0] + 1})")
            print(f"   Recent:     {change_row['recent_mean']:.6f}")
            print(f"   Historical: {change_row['historical_mean']:.6f}")
            print(f"   Change:     {change_row['pct_change']:+.1f}%")
            print(f"   Std Diff:   {change_row['std_diff']:+.2f} σ")
    else:
        print("\n✅ No critical overlap found")

except AttributeError:
    print("⚠️  Feature importances not available for this model")

# Summary
print("\n" + "="*80)
print("💡 ROOT CAUSE SUMMARY")
print("="*80)

max_change = df_changes.iloc[0]
print(f"\n1️⃣ Feature with largest change:")
print(f"   {max_change['feature']}")
print(f"   Change: {max_change['pct_change']:+.1f}%")
print(f"   Standardized difference: {max_change['std_diff']:+.2f} σ")

large_changes = len(df_changes[df_changes['abs_std_diff'] > 2])
print(f"\n2️⃣ Features with >2σ change: {large_changes} features")

extreme_changes = len(df_changes[df_changes['abs_std_diff'] > 3])
print(f"   Features with >3σ change: {extreme_changes} features")

if extreme_changes > 5:
    print(f"\n   ⚠️  WARNING: {extreme_changes} features changed >3σ - DATA REGIME SHIFT!")
elif extreme_changes > 2:
    print(f"\n   ⚠️  CAUTION: {extreme_changes} features changed >3σ - Significant change")
else:
    print(f"\n   ✅ NORMAL: Minimal extreme changes")

print("\n" + "="*80)
print("🎯 RECOMMENDATIONS")
print("="*80)

if extreme_changes > 5:
    print("\n⚠️  CRITICAL RECOMMENDATION:")
    print("   1. Recent market regime differs significantly from training data")
    print("   2. Model may not generalize well to current conditions")
    print("   3. OPTIONS:")
    print("      A) Retrain model on more recent data")
    print("      B) Implement adaptive thresholds based on feature distribution")
    print("      C) Add ensemble with regime detection")
    print("      D) Monitor closely and be ready to intervene")
elif large_changes > 10:
    print("\n⚠️  MODERATE RECOMMENDATION:")
    print("   1. Notable distribution shift detected")
    print("   2. Monitor model performance closely")
    print("   3. Consider threshold adjustments")
else:
    print("\n✅ NORMAL RECOMMENDATION:")
    print("   Continue monitoring, no immediate action required")

print("\n" + "="*80)
