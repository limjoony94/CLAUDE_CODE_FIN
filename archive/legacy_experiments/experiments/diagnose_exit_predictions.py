"""
Diagnose Exit Model Prediction Distribution
===========================================

Check what probabilities the Exit models are actually predicting
to understand why ML Exit rate is 0%.

Created: 2025-10-27
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib

# Configuration
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("EXIT MODEL PREDICTION DIAGNOSTICS")
print("="*80)
print()

# Load Data
print("-"*80)
print("STEP 1: Loading Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")
print()

# Load Exit Models
print("-"*80)
print("STEP 2: Loading Exit Models")
print("-"*80)

# LONG Exit
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f]

# SHORT Exit
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"  LONG Exit: {len(long_exit_features)} features")
print(f"  SHORT Exit: {len(short_exit_features)} features")
print()

# Test Predictions
print("-"*80)
print("STEP 3: Testing Exit Model Predictions")
print("-"*80)
print()

def test_exit_predictions(model, scaler, features, side):
    """Test prediction distribution"""
    print(f"🔍 {side} Exit Model Predictions:")

    # Prepare features
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].values

    # Remove NaN
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]

    if len(X_clean) == 0:
        print(f"  ⚠️  No valid samples!")
        return

    # Scale and predict
    X_scaled = scaler.transform(X_clean)
    probs = model.predict_proba(X_scaled)[:, 1]

    # Statistics
    print(f"  Valid Samples: {len(probs):,}")
    print(f"  Min Probability: {probs.min():.4f}")
    print(f"  Max Probability: {probs.max():.4f}")
    print(f"  Mean Probability: {probs.mean():.4f}")
    print(f"  Median Probability: {np.median(probs):.4f}")
    print(f"  Std Deviation: {probs.std():.4f}")
    print()

    # Percentiles
    print(f"  Percentiles:")
    print(f"    10%: {np.percentile(probs, 10):.4f}")
    print(f"    25%: {np.percentile(probs, 25):.4f}")
    print(f"    50%: {np.percentile(probs, 50):.4f}")
    print(f"    75%: {np.percentile(probs, 75):.4f}")
    print(f"    90%: {np.percentile(probs, 90):.4f}")
    print(f"    95%: {np.percentile(probs, 95):.4f}")
    print(f"    99%: {np.percentile(probs, 99):.4f}")
    print()

    # Threshold Analysis
    print(f"  Threshold Analysis:")
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        count = (probs >= threshold).sum()
        pct = count / len(probs) * 100
        print(f"    >= {threshold:.2f}: {count:,} ({pct:.2f}%)")
    print()

# Test LONG Exit
test_exit_predictions(long_exit_model, long_exit_scaler, long_exit_features, "LONG")

# Test SHORT Exit
test_exit_predictions(short_exit_model, short_exit_scaler, short_exit_features, "SHORT")

print("="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print()
print("Next Steps:")
print("1. If max probability < 0.75: Threshold is too high")
print("2. If very few samples >= 0.75: Consider lower threshold (0.60-0.70)")
print("3. If predictions are flat: Model may need retraining")
print()
