"""
Comprehensive Model Performance Analysis
=========================================

Analyze the new models (2025-10-21) to identify performance issues:
1. Training vs Test set performance
2. Overfitting analysis
3. Feature importance
4. Prediction distribution
5. Exit strategy effectiveness
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("MODEL PERFORMANCE ANALYSIS - DIAGNOSTIC REPORT")
print("="*80)

# ============================================================================
# STEP 1: Load Data and Features
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading Data")
print("="*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
df = df.dropna().reset_index(drop=True)
print(f"✅ {len(df):,} candles after features")

# ============================================================================
# STEP 2: Load Models and Labels
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Loading Models")
print("="*80)

# LONG Entry Model
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616.pkl"
print(f"\n📦 Loading LONG Entry Model: {long_model_path.name}")
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

# SHORT Entry Model
short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616.pkl"
print(f"📦 Loading SHORT Entry Model: {short_model_path.name}")
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

# ============================================================================
# STEP 3: Load Training Labels
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Loading Training Labels")
print("="*80)

# Load labels from training script output
labels_file = PROJECT_ROOT / "results" / "trade_outcome_labels_20251021.pkl"

if labels_file.exists():
    print(f"\n✅ Loading labels from: {labels_file.name}")
    with open(labels_file, 'rb') as f:
        labels_data = pickle.load(f)
    long_labels = labels_data['long_labels']
    short_labels = labels_data['short_labels']
else:
    print(f"\n⚠️  Labels file not found, skipping label analysis")
    long_labels = None
    short_labels = None

# ============================================================================
# STEP 4: Train/Test Split Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Train/Test Split Analysis")
print("="*80)

# Split data (80/20)
split_idx = int(len(df) * 0.8)

train_df = df[:split_idx].copy()
test_df = df[split_idx:].copy()

print(f"\n📊 Data Split:")
print(f"  Training Set: {len(train_df):,} candles ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Test Set: {len(test_df):,} candles ({len(test_df)/len(df)*100:.1f}%)")
print(f"\n  Train Period: {train_df['timestamp'].iloc[0]} → {train_df['timestamp'].iloc[-1]}")
print(f"  Test Period: {test_df['timestamp'].iloc[0]} → {test_df['timestamp'].iloc[-1]}")

# ============================================================================
# STEP 5: Prediction Distribution Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Prediction Distribution Analysis")
print("="*80)

# LONG predictions
print("\n🔵 LONG Entry Model:")
X_long_train = train_df[long_feature_columns].values
X_long_test = test_df[long_feature_columns].values

X_long_train_scaled = long_scaler.transform(X_long_train)
X_long_test_scaled = long_scaler.transform(X_long_test)

long_probs_train = long_model.predict_proba(X_long_train_scaled)[:, 1]
long_probs_test = long_model.predict_proba(X_long_test_scaled)[:, 1]

print(f"\n  Training Set Predictions:")
print(f"    Mean Probability: {long_probs_train.mean():.4f}")
print(f"    Std: {long_probs_train.std():.4f}")
print(f"    Min: {long_probs_train.min():.4f}")
print(f"    Max: {long_probs_train.max():.4f}")
print(f"    >= 0.65 threshold: {(long_probs_train >= 0.65).sum():,} ({(long_probs_train >= 0.65).mean()*100:.2f}%)")
print(f"    >= 0.80 (high conf): {(long_probs_train >= 0.80).sum():,} ({(long_probs_train >= 0.80).mean()*100:.2f}%)")

print(f"\n  Test Set Predictions:")
print(f"    Mean Probability: {long_probs_test.mean():.4f}")
print(f"    Std: {long_probs_test.std():.4f}")
print(f"    Min: {long_probs_test.min():.4f}")
print(f"    Max: {long_probs_test.max():.4f}")
print(f"    >= 0.65 threshold: {(long_probs_test >= 0.65).sum():,} ({(long_probs_test >= 0.65).mean()*100:.2f}%)")
print(f"    >= 0.80 (high conf): {(long_probs_test >= 0.80).sum():,} ({(long_probs_test >= 0.80).mean()*100:.2f}%)")

# SHORT predictions
print("\n🔴 SHORT Entry Model:")
X_short_train = train_df[short_feature_columns].values
X_short_test = test_df[short_feature_columns].values

X_short_train_scaled = short_scaler.transform(X_short_train)
X_short_test_scaled = short_scaler.transform(X_short_test)

short_probs_train = short_model.predict_proba(X_short_train_scaled)[:, 1]
short_probs_test = short_model.predict_proba(X_short_test_scaled)[:, 1]

print(f"\n  Training Set Predictions:")
print(f"    Mean Probability: {short_probs_train.mean():.4f}")
print(f"    Std: {short_probs_train.std():.4f}")
print(f"    Min: {short_probs_train.min():.4f}")
print(f"    Max: {short_probs_train.max():.4f}")
print(f"    >= 0.70 threshold: {(short_probs_train >= 0.70).sum():,} ({(short_probs_train >= 0.70).mean()*100:.2f}%)")
print(f"    >= 0.85 (high conf): {(short_probs_train >= 0.85).sum():,} ({(short_probs_train >= 0.85).mean()*100:.2f}%)")

print(f"\n  Test Set Predictions:")
print(f"    Mean Probability: {short_probs_test.mean():.4f}")
print(f"    Std: {short_probs_test.std():.4f}")
print(f"    Min: {short_probs_test.min():.4f}")
print(f"    Max: {short_probs_test.max():.4f}")
print(f"    >= 0.70 threshold: {(short_probs_test >= 0.70).sum():,} ({(short_probs_test >= 0.70).mean()*100:.2f}%)")
print(f"    >= 0.85 (high conf): {(short_probs_test >= 0.85).sum():,} ({(short_probs_test >= 0.85).mean()*100:.2f}%)")

# ============================================================================
# STEP 6: Feature Importance Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Feature Importance Analysis")
print("="*80)

# LONG model top features
print("\n🔵 LONG Entry Model - Top 15 Features:")
long_feature_importance = long_model.feature_importances_
long_feature_ranking = sorted(zip(long_feature_columns, long_feature_importance),
                               key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(long_feature_ranking[:15], 1):
    print(f"  {i:2d}. {feature:40s} {importance:.4f}")

# SHORT model top features
print("\n🔴 SHORT Entry Model - Top 15 Features:")
short_feature_importance = short_model.feature_importances_
short_feature_ranking = sorted(zip(short_feature_columns, short_feature_importance),
                                key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(short_feature_ranking[:15], 1):
    print(f"  {i:2d}. {feature:40s} {importance:.4f}")

# ============================================================================
# STEP 7: Calibration Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Prediction Calibration Analysis")
print("="*80)

# Compare train vs test prediction distributions
print("\n📊 Probability Distribution Comparison:")
print(f"\n  LONG Model:")
print(f"    Train Mean: {long_probs_train.mean():.4f}")
print(f"    Test Mean: {long_probs_test.mean():.4f}")
print(f"    Difference: {abs(long_probs_train.mean() - long_probs_test.mean()):.4f}")

prob_shift_long = long_probs_train.mean() - long_probs_test.mean()
if abs(prob_shift_long) > 0.05:
    print(f"    ⚠️  Significant probability shift detected!")
    if prob_shift_long > 0:
        print(f"    Model is MORE confident on training data (overfitting signal)")
    else:
        print(f"    Model is LESS confident on training data (unusual)")

print(f"\n  SHORT Model:")
print(f"    Train Mean: {short_probs_train.mean():.4f}")
print(f"    Test Mean: {short_probs_test.mean():.4f}")
print(f"    Difference: {abs(short_probs_train.mean() - short_probs_test.mean()):.4f}")

prob_shift_short = short_probs_train.mean() - short_probs_test.mean()
if abs(prob_shift_short) > 0.05:
    print(f"    ⚠️  Significant probability shift detected!")
    if prob_shift_short > 0:
        print(f"    Model is MORE confident on training data (overfitting signal)")
    else:
        print(f"    Model is LESS confident on training data (unusual)")

# ============================================================================
# STEP 8: Entry Signal Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Entry Signal Frequency Analysis")
print("="*80)

# LONG signals
long_entries_train = (long_probs_train >= 0.65).sum()
long_entries_test = (long_probs_test >= 0.65).sum()

print(f"\n🔵 LONG Entry Signals:")
print(f"  Training Set: {long_entries_train:,} / {len(train_df):,} ({long_entries_train/len(train_df)*100:.2f}%)")
print(f"  Test Set: {long_entries_test:,} / {len(test_df):,} ({long_entries_test/len(test_df)*100:.2f}%)")

expected_long_per_day_test = (long_entries_test / len(test_df)) * 288  # 288 = 5min candles per day
print(f"  Expected LONG entries/day (Test): {expected_long_per_day_test:.1f}")

# SHORT signals
short_entries_train = (short_probs_train >= 0.70).sum()
short_entries_test = (short_probs_test >= 0.70).sum()

print(f"\n🔴 SHORT Entry Signals:")
print(f"  Training Set: {short_entries_train:,} / {len(train_df):,} ({short_entries_train/len(train_df)*100:.2f}%)")
print(f"  Test Set: {short_entries_test:,} / {len(test_df):,} ({short_entries_test/len(test_df)*100:.2f}%)")

expected_short_per_day_test = (short_entries_test / len(test_df)) * 288
print(f"  Expected SHORT entries/day (Test): {expected_short_per_day_test:.1f}")

# ============================================================================
# STEP 9: Diagnostic Summary
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

issues = []

# 1. Check probability shift
if abs(prob_shift_long) > 0.05:
    issues.append(f"LONG model shows {abs(prob_shift_long):.3f} probability shift (overfitting)")
if abs(prob_shift_short) > 0.05:
    issues.append(f"SHORT model shows {abs(prob_shift_short):.3f} probability shift (overfitting)")

# 2. Check prediction variance
if long_probs_test.std() < 0.1:
    issues.append(f"LONG model has low prediction variance ({long_probs_test.std():.3f}) - may be underconfident")
if short_probs_test.std() < 0.1:
    issues.append(f"SHORT model has low prediction variance ({short_probs_test.std():.3f}) - may be underconfident")

# 3. Check entry frequency
if expected_long_per_day_test < 1:
    issues.append(f"LONG entry frequency too low ({expected_long_per_day_test:.1f}/day)")
if expected_long_per_day_test > 20:
    issues.append(f"LONG entry frequency too high ({expected_long_per_day_test:.1f}/day)")

if expected_short_per_day_test < 0.5:
    issues.append(f"SHORT entry frequency too low ({expected_short_per_day_test:.1f}/day)")
if expected_short_per_day_test > 10:
    issues.append(f"SHORT entry frequency too high ({expected_short_per_day_test:.1f}/day)")

if len(issues) == 0:
    print("\n✅ No major issues detected in model predictions")
else:
    print(f"\n⚠️  {len(issues)} Issues Detected:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nRecommendations will be provided based on these findings.")
