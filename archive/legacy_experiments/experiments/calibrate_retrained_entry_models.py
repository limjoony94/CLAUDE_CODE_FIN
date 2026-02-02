"""
Calibrate Retrained Entry Models - Isotonic Regression
======================================================

Problem: XGBoost models are severely under-confident
  Training positive rate: 27.96% (LONG), 7.37% (SHORT)
  Actual predictions: mean 13.13% (LONG), 1.92% (SHORT)

Solution: Probability Calibration using Isotonic Regression
  Aligns model probabilities with actual positive rates
  Allows threshold 0.75 to achieve 4.0+ trades/day

Process:
  1. Load retrained models (20251029_081454)
  2. Use validation fold data for calibration
  3. Apply CalibratedClassifierCV (isotonic method)
  4. Save calibrated models with new timestamp

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV

# Configuration
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
CALIBRATION_SPLIT = 0.8  # Use last 20% of training data for calibration

print("="*80)
print("PROBABILITY CALIBRATION - RETRAINED ENTRY MODELS")
print("="*80)
print()
print("Purpose: Fix XGBoost under-confidence using Isotonic Regression")
print("Method: CalibratedClassifierCV on validation fold")
print()

# Load data
print("-"*80)
print("Loading Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")

# Use last 90 days for training (excluding 14-day holdout)
holdout_candles = 14 * 24 * 12
training_candles = 90 * 24 * 12
total_needed = holdout_candles + training_candles

train_start_idx = len(df) - total_needed
train_end_idx = len(df) - holdout_candles
df_train_full = df.iloc[train_start_idx:train_end_idx].copy()

# Split training data: 80% for model validation, 20% for calibration
calib_split_idx = int(len(df_train_full) * CALIBRATION_SPLIT)
df_calib = df_train_full.iloc[calib_split_idx:].copy().reset_index(drop=True)

print(f"✅ Training period: {len(df_train_full):,} candles (90 days)")
print(f"✅ Calibration set: {len(df_calib):,} candles (last 20%)")
print()

# Load LONG Entry Model
print("-"*80)
print("Calibrating LONG Entry Model")
print("-"*80)

long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded LONG Entry model (85 features)")

# Load LONG entry labels (from retraining)
df_long_labels = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "long_entry_labels.csv")
df_long_labels['timestamp'] = pd.to_datetime(df_long_labels['timestamp'])
df_calib['timestamp'] = pd.to_datetime(df_calib['timestamp'])

# Merge labels with calibration set
df_calib_long = df_calib.merge(df_long_labels, on='timestamp', how='inner')
print(f"✅ Calibration samples: {len(df_calib_long):,}")

# Prepare features and labels
X_calib_long = df_calib_long[long_entry_features].values
y_calib_long = df_calib_long['label'].values
X_calib_long_scaled = long_entry_scaler.transform(X_calib_long)

print(f"   Positive rate: {y_calib_long.mean()*100:.2f}%")

# Apply Isotonic Calibration
print("⏳ Applying Isotonic Regression...")
calibrated_long_model = CalibratedClassifierCV(
    estimator=long_entry_model,
    method='isotonic',
    cv='prefit'  # Use pre-trained model
)
calibrated_long_model.fit(X_calib_long_scaled, y_calib_long)
print("✅ LONG model calibrated")

# Verify calibration improvement
probs_before = long_entry_model.predict_proba(X_calib_long_scaled)[:, 1]
probs_after = calibrated_long_model.predict_proba(X_calib_long_scaled)[:, 1]

print()
print("Calibration Effect (on calibration set):")
print(f"  Before: mean {probs_before.mean()*100:.2f}%, median {np.median(probs_before)*100:.2f}%")
print(f"  After:  mean {probs_after.mean()*100:.2f}%, median {np.median(probs_after)*100:.2f}%")
print(f"  Actual positive rate: {y_calib_long.mean()*100:.2f}%")
print()

# Save calibrated LONG model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
calibrated_long_path = MODELS_DIR / f"xgboost_long_entry_calibrated_{timestamp}.pkl"
joblib.dump(calibrated_long_model, calibrated_long_path)
print(f"✅ Saved: {calibrated_long_path.name}")

# Save scaler and features (reuse from original)
calibrated_long_scaler_path = MODELS_DIR / f"xgboost_long_entry_calibrated_{timestamp}_scaler.pkl"
calibrated_long_features_path = MODELS_DIR / f"xgboost_long_entry_calibrated_{timestamp}_features.txt"
joblib.dump(long_entry_scaler, calibrated_long_scaler_path)
with open(calibrated_long_features_path, 'w') as f:
    f.write('\n'.join(long_entry_features))

print(f"✅ Saved: {calibrated_long_scaler_path.name}")
print(f"✅ Saved: {calibrated_long_features_path.name}")
print()

# Load SHORT Entry Model
print("-"*80)
print("Calibrating SHORT Entry Model")
print("-"*80)

short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded SHORT Entry model (79 features)")

# Load SHORT entry labels
df_short_labels = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "short_entry_labels.csv")
df_short_labels['timestamp'] = pd.to_datetime(df_short_labels['timestamp'])

# Merge labels with calibration set
df_calib_short = df_calib.merge(df_short_labels, on='timestamp', how='inner')
print(f"✅ Calibration samples: {len(df_calib_short):,}")

# Prepare features and labels
X_calib_short = df_calib_short[short_entry_features].values
y_calib_short = df_calib_short['label'].values
X_calib_short_scaled = short_entry_scaler.transform(X_calib_short)

print(f"   Positive rate: {y_calib_short.mean()*100:.2f}%")

# Apply Isotonic Calibration
print("⏳ Applying Isotonic Regression...")
calibrated_short_model = CalibratedClassifierCV(
    estimator=short_entry_model,
    method='isotonic',
    cv='prefit'
)
calibrated_short_model.fit(X_calib_short_scaled, y_calib_short)
print("✅ SHORT model calibrated")

# Verify calibration improvement
probs_before = short_entry_model.predict_proba(X_calib_short_scaled)[:, 1]
probs_after = calibrated_short_model.predict_proba(X_calib_short_scaled)[:, 1]

print()
print("Calibration Effect (on calibration set):")
print(f"  Before: mean {probs_before.mean()*100:.2f}%, median {np.median(probs_before)*100:.2f}%")
print(f"  After:  mean {probs_after.mean()*100:.2f}%, median {np.median(probs_after)*100:.2f}%")
print(f"  Actual positive rate: {y_calib_short.mean()*100:.2f}%")
print()

# Save calibrated SHORT model
calibrated_short_path = MODELS_DIR / f"xgboost_short_entry_calibrated_{timestamp}.pkl"
joblib.dump(calibrated_short_model, calibrated_short_path)
print(f"✅ Saved: {calibrated_short_path.name}")

# Save scaler and features
calibrated_short_scaler_path = MODELS_DIR / f"xgboost_short_entry_calibrated_{timestamp}_scaler.pkl"
calibrated_short_features_path = MODELS_DIR / f"xgboost_short_entry_calibrated_{timestamp}_features.txt"
joblib.dump(short_entry_scaler, calibrated_short_scaler_path)
with open(calibrated_short_features_path, 'w') as f:
    f.write('\n'.join(short_entry_features))

print(f"✅ Saved: {calibrated_short_scaler_path.name}")
print(f"✅ Saved: {calibrated_short_features_path.name}")
print()

print("="*80)
print("CALIBRATION COMPLETE")
print("="*80)
print()
print(f"Timestamp: {timestamp}")
print()
print("Next Steps:")
print("  1. Run backtest with calibrated models at threshold 0.75")
print("  2. Verify: 4.0+ trades/day with positive returns")
print("  3. Compare calibrated vs non-calibrated performance")
print()
