"""
Analyze Dominant Features: Oct 18 vs Sep 23
Identify which features cause all-day high probabilities on Oct 18
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("FEATURE DOMINANCE ANALYSIS: OCT 18 vs SEP 23")
print("="*80)

# ============================================================================
# STEP 1: Load Trade-Outcome LONG model
# ============================================================================
print("\n📁 Loading Trade-Outcome LONG model...")

long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_names = [line.strip() for line in f.readlines()]

print(f"  ✅ Model loaded: {len(long_feature_names)} features")
print(f"  Model type: {type(long_model).__name__}")

# ============================================================================
# STEP 2: Extract Feature Importance
# ============================================================================
print("\n📊 Extracting Feature Importance...")

# XGBoost stores feature importance
feature_importance = long_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': long_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n🏆 Top 20 Most Important Features:")
print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12}")
print("-"*80)
for idx, row in feature_importance_df.head(20).iterrows():
    print(f"{idx+1:<6} {row['feature']:<35} {row['importance']:<12.6f}")

# ============================================================================
# STEP 3: Load and calculate features for both dates
# ============================================================================
print("\n📊 Loading data and calculating features...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

df_features = calculate_all_features(df.copy())
df_features = prepare_exit_features(df_features)
print(f"  ✅ Features calculated: {len(df_features):,} candles")

# Filter dates
df_sep23 = df_features[df_features['timestamp'].str.startswith('2025-09-23')].copy()
df_oct18 = df_features[df_features['timestamp'].str.startswith('2025-10-18')].copy()

print(f"\n  Sep 23 candles: {len(df_sep23)}")
print(f"  Oct 18 candles: {len(df_oct18)}")

# ============================================================================
# STEP 4: Compare feature distributions
# ============================================================================
print("\n🔍 Comparing Feature Distributions...")
print("="*80)

feature_comparison = []

for feature in long_feature_names:
    sep23_values = df_sep23[feature].values
    oct18_values = df_oct18[feature].values

    sep23_mean = sep23_values.mean()
    oct18_mean = oct18_values.mean()

    sep23_std = sep23_values.std()
    oct18_std = oct18_values.std()

    sep23_min = sep23_values.min()
    oct18_min = oct18_values.min()

    sep23_max = sep23_values.max()
    oct18_max = oct18_values.max()

    # Calculate difference
    mean_diff = oct18_mean - sep23_mean
    mean_diff_pct = (mean_diff / (abs(sep23_mean) + 1e-10)) * 100

    # Calculate variance ratio (stability measure)
    variance_ratio = oct18_std / (sep23_std + 1e-10)

    feature_comparison.append({
        'feature': feature,
        'sep23_mean': sep23_mean,
        'oct18_mean': oct18_mean,
        'mean_diff': mean_diff,
        'mean_diff_pct': mean_diff_pct,
        'sep23_std': sep23_std,
        'oct18_std': oct18_std,
        'variance_ratio': variance_ratio,
        'sep23_min': sep23_min,
        'oct18_min': oct18_min,
        'sep23_max': sep23_max,
        'oct18_max': oct18_max
    })

df_comparison = pd.DataFrame(feature_comparison)

# ============================================================================
# STEP 5: Merge with feature importance
# ============================================================================
df_analysis = df_comparison.merge(
    feature_importance_df,
    on='feature',
    how='left'
)

# Calculate "impact score" = importance × |mean_diff_pct|
df_analysis['impact_score'] = df_analysis['importance'] * np.abs(df_analysis['mean_diff_pct'])

# Sort by impact score
df_analysis = df_analysis.sort_values('impact_score', ascending=False)

# ============================================================================
# STEP 6: Display top features by impact
# ============================================================================
print("\n🎯 TOP FEATURES BY IMPACT (Importance × Difference):")
print("="*80)
print(f"{'Rank':<6} {'Feature':<30} {'Import':<10} {'Sep23':<12} {'Oct18':<12} {'Diff%':<10} {'Impact':<10}")
print("-"*100)

for idx, row in df_analysis.head(30).iterrows():
    print(f"{idx+1:<6} {row['feature']:<30} {row['importance']:<10.4f} "
          f"{row['sep23_mean']:<12.4f} {row['oct18_mean']:<12.4f} "
          f"{row['mean_diff_pct']:<10.1f} {row['impact_score']:<10.2f}")

# ============================================================================
# STEP 7: Analyze temporal stability on Oct 18
# ============================================================================
print("\n⏰ TEMPORAL STABILITY ANALYSIS - OCT 18:")
print("="*80)
print("\nAnalyzing if top features remain stable throughout Oct 18...")

# Select top 10 impact features
top_features = df_analysis.head(10)['feature'].tolist()

print(f"\n📊 Hourly Statistics for Top 10 Features on Oct 18:")
print("="*80)

for feature in top_features:
    print(f"\n📈 {feature}:")
    print(f"  {'Hour':<6} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("  " + "-"*60)

    hourly_stats = []
    for hour in range(24):
        hour_str = f"{hour:02d}"
        df_hour = df_oct18[df_oct18['timestamp'].str.contains(f' {hour_str}:')].copy()

        if len(df_hour) == 0:
            continue

        values = df_hour[feature].values
        hourly_stats.append({
            'hour': hour,
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max()
        })

    # Display hourly stats
    for stat in hourly_stats:
        print(f"  {stat['hour']:02d}:00  {stat['mean']:<12.4f} {stat['std']:<12.4f} "
              f"{stat['min']:<12.4f} {stat['max']:<12.4f}")

    # Calculate coefficient of variation (std/mean) across hours
    hourly_means = [s['mean'] for s in hourly_stats]
    if len(hourly_means) > 0:
        overall_mean = np.mean(hourly_means)
        overall_std = np.std(hourly_means)
        cv = (overall_std / abs(overall_mean)) * 100 if overall_mean != 0 else 0
        print(f"\n  ⚖️  Stability across hours (CV): {cv:.2f}%")
        if cv < 10:
            print(f"      → Very stable (explains all-day high probability)")
        elif cv < 30:
            print(f"      → Moderately stable")
        else:
            print(f"      → High variation")

# ============================================================================
# STEP 8: Calculate probabilities by feature bucket
# ============================================================================
print("\n📊 PROBABILITY BY FEATURE VALUE RANGES:")
print("="*80)

# Analyze top 3 impact features
for feature in top_features[:3]:
    print(f"\n📈 {feature}:")

    # Get combined data
    df_combined = pd.concat([
        df_sep23[['timestamp', feature] + long_feature_names].assign(date='Sep23'),
        df_oct18[['timestamp', feature] + long_feature_names].assign(date='Oct18')
    ])

    # Calculate probabilities
    probs = []
    for idx in range(len(df_combined)):
        row = df_combined.iloc[idx:idx+1]
        # Extract ONLY the 44 features needed
        features = row[long_feature_names].values.reshape(1, -1)
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        probs.append(prob)

    df_combined['prob'] = probs

    # Create buckets
    df_combined['bucket'] = pd.cut(df_combined[feature], bins=10)

    # Analyze by bucket
    bucket_stats = df_combined.groupby('bucket').agg({
        'prob': ['mean', 'count'],
        feature: 'mean'
    }).round(4)

    print(f"\n  Value Range Analysis:")
    print(f"  {'Range':<35} {'Count':<8} {'Avg Prob':<12} {'Range Mean':<12}")
    print("  " + "-"*80)

    for idx, row in bucket_stats.iterrows():
        prob_mean = row[('prob', 'mean')]
        count = int(row[('prob', 'count')])
        range_mean = row[(feature, 'mean')]
        print(f"  {str(idx):<35} {count:<8} {prob_mean*100:<11.2f}% {range_mean:<12.4f}")

    # Compare Sep23 vs Oct18 distributions
    sep23_feature_values = df_sep23[feature].values
    oct18_feature_values = df_oct18[feature].values

    print(f"\n  Distribution Comparison:")
    print(f"    Sep 23: mean={sep23_feature_values.mean():.4f}, "
          f"median={np.median(sep23_feature_values):.4f}, "
          f"std={sep23_feature_values.std():.4f}")
    print(f"    Oct 18: mean={oct18_feature_values.mean():.4f}, "
          f"median={np.median(oct18_feature_values):.4f}, "
          f"std={oct18_feature_values.std():.4f}")

# ============================================================================
# STEP 9: Final Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: DOMINANT FEATURES CAUSING OCT 18 HIGH PROBABILITIES")
print("="*80)

top_5 = df_analysis.head(5)

print(f"\n🎯 Top 5 Features by Impact Score:")
for idx, row in top_5.iterrows():
    print(f"\n{idx+1}. {row['feature']}")
    print(f"   Feature Importance: {row['importance']:.6f}")
    print(f"   Sep 23 mean: {row['sep23_mean']:.4f}")
    print(f"   Oct 18 mean: {row['oct18_mean']:.4f}")
    print(f"   Difference: {row['mean_diff']:.4f} ({row['mean_diff_pct']:+.1f}%)")
    print(f"   Impact Score: {row['impact_score']:.2f}")

    # Check if this feature is stable on Oct 18
    oct18_values = df_oct18[row['feature']].values
    cv = (oct18_values.std() / abs(oct18_values.mean())) * 100 if oct18_values.mean() != 0 else 0
    print(f"   Oct 18 Stability (CV): {cv:.2f}%", end="")
    if cv < 10:
        print(" → ✅ Very stable (explains all-day high prob)")
    elif cv < 30:
        print(" → ⚠️  Moderate variation")
    else:
        print(" → ❌ High variation")

print("\n💡 KEY INSIGHTS:")
print("  1. Features with high importance AND large difference → Dominant impact")
print("  2. Features stable throughout Oct 18 → Explains all-day high probabilities")
print("  3. 5-min candles stay high if underlying features remain in extreme range")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
