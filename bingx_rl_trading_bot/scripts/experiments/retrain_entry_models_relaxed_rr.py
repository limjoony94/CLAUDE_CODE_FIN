"""
Retrain Entry Models with Trade-Outcome Labels (SAMPLE DATA)
=============================================================

Quick validation using 5,000 candles sample.

This is a proof-of-concept to validate the Trade-Outcome labeling approach
before running on full dataset.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.experiments.trade_simulator import TradeSimulator, load_exit_models
from src.labeling.trade_outcome_labeling import TradeOutcomeLabeling

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# SAMPLE SIZE
SAMPLE_SIZE = 5000

print("="*80)
print(f"RETRAIN ENTRY MODELS: RELAXED RISK-REWARD (SAMPLE: {SAMPLE_SIZE:,} candles)")
print("="*80)
print("\n⚡ Relaxed Risk-Reward Criteria:")
print("   MAE: -2% → -3% (max adverse excursion)")
print("   MFE: 4% → 2.5% (min favorable excursion)")
print("   Expected: 10-20% positive label rate (vs 0.1% strict)\n")

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

print("\n" + "-"*80)
print("STEP 1: Loading and Preparing Data")
print("-"*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df_full):,} candles")

# Use last 5,000 candles (more recent data)
df = df_full.tail(SAMPLE_SIZE).reset_index(drop=True).copy()
print(f"📊 Using sample: Last {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate ALL features (LONG + SHORT + EXIT)
print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"✅ All features calculated ({len(df.columns)} columns)")

# Validate SHORT features exist
from scripts.experiments.calculate_all_features import SHORT_FEATURE_COLUMNS
missing_short = [f for f in SHORT_FEATURE_COLUMNS if f not in df.columns]
if missing_short:
    print(f"\n⚠️ WARNING: {len(missing_short)} SHORT features missing:")
    for f in missing_short[:10]:  # Show first 10
        print(f"   - {f}")
    if len(missing_short) > 10:
        print(f"   ... and {len(missing_short)-10} more")
else:
    print(f"✅ All {len(SHORT_FEATURE_COLUMNS)} SHORT features present")

# ============================================================================
# STEP 2: Load Exit Models and Create Simulators
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Loading Exit Models for Trade Simulation")
print("-"*80)

exit_models = load_exit_models()

long_simulator = TradeSimulator(
    exit_model=exit_models['long'][0],
    exit_scaler=exit_models['long'][1],
    exit_features=exit_models['long'][2],
    leverage=4,
    ml_exit_threshold=0.70,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

short_simulator = TradeSimulator(
    exit_model=exit_models['short'][0],
    exit_scaler=exit_models['short'][1],
    exit_features=exit_models['short'][2],
    leverage=4,
    ml_exit_threshold=0.70,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

print("✅ Trade simulators ready")

# ============================================================================
# STEP 3: Create Trade-Outcome Labels
# ============================================================================

print("\n" + "-"*80)
print("STEP 3: Creating Trade-Outcome Labels")
print("-"*80)

labeler = TradeOutcomeLabeling(
    profit_threshold=0.02,  # 2% leveraged profit
    mae_threshold=-0.03,  # Max 3% adverse (RELAXED from -2%)
    mfe_threshold=0.025,  # Min 2.5% favorable (RELAXED from 4%)
    scoring_threshold=2  # 2 of 3 criteria
)

# Create LONG labels
long_labels = labeler.create_entry_labels(df, long_simulator, 'LONG', show_progress=True)

# Create SHORT labels
short_labels = labeler.create_entry_labels(df, short_simulator, 'SHORT', show_progress=True)

# ============================================================================
# STEP 4: Retrain LONG Entry Model
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Retraining LONG Entry Model")
print("="*80)

# Get LONG feature columns from baseline model
baseline_long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(baseline_long_features_path, 'r') as f:
    long_feature_cols = [line.strip() for line in f.readlines() if line.strip()]

print(f"\nTotal LONG features: {len(long_feature_cols)}")

# Prepare data
X_long = df[long_feature_cols].values
y_long = long_labels

# Split train/test (80/20, no shuffle for time series)
split_idx = int(len(X_long) * 0.8)
X_train_long = X_long[:split_idx]
y_train_long = y_long[:split_idx]
X_test_long = X_long[split_idx:]
y_test_long = y_long[split_idx:]

print(f"Train samples: {len(X_train_long):,}")
print(f"Test samples: {len(X_test_long):,}")
print(f"Positive rate (train): {np.sum(y_train_long)/len(y_train_long)*100:.2f}%")
print(f"Positive rate (test): {np.sum(y_test_long)/len(y_test_long)*100:.2f}%")

# Scale features
scaler_long = StandardScaler()
X_train_long_scaled = scaler_long.fit_transform(X_train_long)
X_test_long_scaled = scaler_long.transform(X_test_long)

# Train XGBoost model
print("\nTraining LONG Entry Model...")
scale_pos_weight_long = len(y_train_long[y_train_long == 0]) / len(y_train_long[y_train_long == 1])
print(f"  Scale pos weight: {scale_pos_weight_long:.2f}")

model_long = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_long,
    random_state=42,
    eval_metric='logloss'
)

model_long.fit(X_train_long_scaled, y_train_long)

# Evaluate
y_pred_long = model_long.predict(X_test_long_scaled)

print("\n" + "-"*80)
print("LONG Entry Model Performance (Test Set)")
print("-"*80)
print(classification_report(y_test_long, y_pred_long, digits=4))

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path_long = MODELS_DIR / f"xgboost_long_relaxed_rr_{timestamp}.pkl"
scaler_path_long = MODELS_DIR / f"xgboost_long_relaxed_rr_{timestamp}_scaler.pkl"
features_path_long = MODELS_DIR / f"xgboost_long_relaxed_rr_{timestamp}_features.txt"
metadata_path_long = MODELS_DIR / f"xgboost_long_relaxed_rr_{timestamp}_metadata.json"

with open(model_path_long, 'wb') as f:
    pickle.dump(model_long, f)

joblib.dump(scaler_long, scaler_path_long)

with open(features_path_long, 'w') as f:
    f.write('\n'.join(long_feature_cols))

# Save metadata
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
metadata_long = {
    "labeling_method": "trade_outcome_relaxed_rr_SAMPLE",
    "sample_size": SAMPLE_SIZE,
    "labeling_criteria": {
        "profit_threshold": 0.02,
        "mae_threshold": -0.03,  # RELAXED
        "mfe_threshold": 0.025,  # RELAXED
        "scoring_threshold": 2
    },
    "num_features": len(long_feature_cols),
    "num_train_samples": len(X_train_long),
    "num_test_samples": len(X_test_long),
    "positive_rate_train": float(np.sum(y_train_long)/len(y_train_long)),
    "positive_rate_test": float(np.sum(y_test_long)/len(y_test_long)),
    "scores": {
        "accuracy": float(accuracy_score(y_test_long, y_pred_long)),
        "precision": float(precision_score(y_test_long, y_pred_long)),
        "recall": float(recall_score(y_test_long, y_pred_long)),
        "f1": float(f1_score(y_test_long, y_pred_long))
    }
}

import json
with open(metadata_path_long, 'w') as f:
    json.dump(metadata_long, f, indent=2)

print(f"\n✅ LONG Entry Model saved:")
print(f"   {model_path_long.name}")
print(f"   Precision: {metadata_long['scores']['precision']*100:.2f}%")

# ============================================================================
# STEP 5: Retrain SHORT Entry Model
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Retraining SHORT Entry Model")
print("="*80)

# Get SHORT feature columns from baseline model
baseline_short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(baseline_short_features_path, 'r') as f:
    short_feature_cols = [line.strip() for line in f.readlines() if line.strip()]

print(f"\nTotal SHORT features in baseline model: {len(short_feature_cols)}")

# Validate all features exist in DataFrame
missing_features = [f for f in short_feature_cols if f not in df.columns]
if missing_features:
    print(f"\n❌ ERROR: {len(missing_features)} features missing from DataFrame:")
    for f in missing_features[:10]:
        print(f"   - {f}")
    if len(missing_features) > 10:
        print(f"   ... and {len(missing_features)-10} more")
    print("\n⚠️ Cannot train SHORT model without all features. Skipping SHORT training.")
    print(f"   Available columns: {len(df.columns)}")
    print(f"   Required features: {len(short_feature_cols)}")
    print("\nDEBUG: Sample of available columns:")
    print(f"   {list(df.columns[:20])}")
else:
    print(f"✅ All {len(short_feature_cols)} SHORT features present in DataFrame")

# Prepare data only if all features exist
if not missing_features:
    X_short = df[short_feature_cols].values
    y_short = short_labels

    # Split train/test
    X_train_short = X_short[:split_idx]
    y_train_short = y_short[:split_idx]
    X_test_short = X_short[split_idx:]
    y_test_short = y_short[split_idx:]

    print(f"Train samples: {len(X_train_short):,}")
    print(f"Test samples: {len(X_test_short):,}")
    print(f"Positive rate (train): {np.sum(y_train_short)/len(y_train_short)*100:.2f}%")
    print(f"Positive rate (test): {np.sum(y_test_short)/len(y_test_short)*100:.2f}%")

    # Scale features
    scaler_short = StandardScaler()
    X_train_short_scaled = scaler_short.fit_transform(X_train_short)
    X_test_short_scaled = scaler_short.transform(X_test_short)

    # Train XGBoost model
    print("\nTraining SHORT Entry Model...")
    scale_pos_weight_short = len(y_train_short[y_train_short == 0]) / len(y_train_short[y_train_short == 1])
    print(f"  Scale pos weight: {scale_pos_weight_short:.2f}")

    model_short = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_short,
        random_state=42,
        eval_metric='logloss'
    )

    model_short.fit(X_train_short_scaled, y_train_short)

    # Evaluate
    y_pred_short = model_short.predict(X_test_short_scaled)

    print("\n" + "-"*80)
    print("SHORT Entry Model Performance (Test Set)")
    print("-"*80)
    print(classification_report(y_test_short, y_pred_short, digits=4))

    # Save model
    model_path_short = MODELS_DIR / f"xgboost_short_relaxed_rr_{timestamp}.pkl"
    scaler_path_short = MODELS_DIR / f"xgboost_short_relaxed_rr_{timestamp}_scaler.pkl"
    features_path_short = MODELS_DIR / f"xgboost_short_relaxed_rr_{timestamp}_features.txt"
    metadata_path_short = MODELS_DIR / f"xgboost_short_relaxed_rr_{timestamp}_metadata.json"

    with open(model_path_short, 'wb') as f:
        pickle.dump(model_short, f)

    joblib.dump(scaler_short, scaler_path_short)

    with open(features_path_short, 'w') as f:
        f.write('\n'.join(short_feature_cols))

    # Save metadata
    metadata_short = {
        "labeling_method": "trade_outcome_relaxed_rr_SAMPLE",
        "sample_size": SAMPLE_SIZE,
        "labeling_criteria": {
            "profit_threshold": 0.02,
            "mae_threshold": -0.03,  # RELAXED
            "mfe_threshold": 0.025,  # RELAXED
            "scoring_threshold": 2
        },
        "num_features": len(short_feature_cols),
        "num_train_samples": len(X_train_short),
        "num_test_samples": len(X_test_short),
        "positive_rate_train": float(np.sum(y_train_short)/len(y_train_short)),
        "positive_rate_test": float(np.sum(y_test_short)/len(y_test_short)),
        "scores": {
            "accuracy": float(accuracy_score(y_test_short, y_pred_short)),
            "precision": float(precision_score(y_test_short, y_pred_short)),
            "recall": float(recall_score(y_test_short, y_pred_short)),
            "f1": float(f1_score(y_test_short, y_pred_short))
        }
    }

    with open(metadata_path_short, 'w') as f:
        json.dump(metadata_short, f, indent=2)

    print(f"\n✅ SHORT Entry Model saved:")
    print(f"   {model_path_short.name}")
    print(f"   Precision: {metadata_short['scores']['precision']*100:.2f}%")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print(f"RETRAINING COMPLETE - RELAXED RISK-REWARD (SAMPLE: {SAMPLE_SIZE:,})")
print("="*80)
print(f"\n📊 Relaxed Criteria: MAE -3%, MFE 2.5% (vs Strict: MAE -2%, MFE 4%)")

print(f"\n✅ LONG Entry Model:")
print(f"   Precision: {metadata_long['scores']['precision']*100:.2f}%")
print(f"   Recall: {metadata_long['scores']['recall']*100:.2f}%")
print(f"   F1: {metadata_long['scores']['f1']*100:.2f}%")
print(f"   File: {model_path_long.name}")

if not missing_features:
    print(f"\n✅ SHORT Entry Model:")
    print(f"   Precision: {metadata_short['scores']['precision']*100:.2f}%")
    print(f"   Recall: {metadata_short['scores']['recall']*100:.2f}%")
    print(f"   F1: {metadata_short['scores']['f1']*100:.2f}%")
    print(f"   File: {model_path_short.name}")
else:
    print(f"\n⚠️ SHORT Entry Model:")
    print(f"   SKIPPED - Missing {len(missing_features)} required features")
    print(f"   See error messages above for details")

print(f"\n📝 NOTE: This is a SAMPLE validation using {SAMPLE_SIZE:,} candles")
print(f"   Full dataset training requires optimization (vectorization/parallelization)")
print(f"   Expected improvements for full data: Better generalization, more robust labels")

print("\n🎯 Next Steps:")
print("   1. Analyze these sample model results")
print("   2. Compare vs Baseline and 2-of-3 approaches")
print("   3. Optimize implementation for full dataset if results are promising")
print("="*80)
