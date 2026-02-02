"""
Trace exact model inputs: Training -> Backtest -> Production
Find any discrepancies in data flow
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
print("TRACE MODEL INPUTS: TRAINING -> BACKTEST -> PRODUCTION")
print("="*80)

# ============================================================================
# STEP 1: Load current production model
# ============================================================================
print("\n" + "="*80)
print("STEP 1: CURRENT PRODUCTION MODEL")
print("="*80)

model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"

print(f"\n📁 Model Files:")
print(f"  Model: {model_path.name}")
print(f"  Scaler: {scaler_path.name}")
print(f"  Features: {features_path.name}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)
scaler = joblib.load(scaler_path)
with open(features_path, 'r') as f:
    feature_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"\n✅ Model loaded:")
print(f"  Features: {len(feature_names)}")
print(f"  Model n_features: {model.n_features_in_}")
print(f"  Scaler n_features: {scaler.n_features_in_}")

print(f"\n📋 Feature List:")
for i, feat in enumerate(feature_names, 1):
    print(f"  {i:2d}. {feat}")

# ============================================================================
# STEP 2: Load data and calculate features (TRAINING/BACKTEST WAY)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CALCULATE FEATURES (TRAINING/BACKTEST WAY)")
print("="*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")

print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)

print(f"\n✅ Features calculated:")
print(f"  Total columns: {len(df.columns)}")
print(f"  Required features available: {all(f in df.columns for f in feature_names)}")

# Check if all required features exist
missing_features = [f for f in feature_names if f not in df.columns]
if missing_features:
    print(f"\n❌ MISSING FEATURES:")
    for feat in missing_features:
        print(f"  - {feat}")
else:
    print(f"  ✅ All {len(feature_names)} features present")

extra_columns = [c for c in df.columns if c not in feature_names and c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
print(f"\n📊 Extra columns (not used in model): {len(extra_columns)}")

# ============================================================================
# STEP 3: Test prediction on Oct 18 20:40:00 (production timestamp)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TEST PREDICTION (Oct 18, 20:40:00)")
print("="*80)

target_timestamp = "2025-10-18 20:40:00"
target_row = df[df['timestamp'] == target_timestamp]

if len(target_row) == 0:
    print(f"❌ Timestamp {target_timestamp} not found in data")
else:
    print(f"\n✅ Found timestamp: {target_timestamp}")
    print(f"  Price: ${target_row['close'].iloc[0]:,.2f}")

    # Extract features
    feature_values = target_row[feature_names].iloc[0]

    print(f"\n📊 Feature Values (first 10):")
    for i, (name, value) in enumerate(list(feature_values.items())[:10], 1):
        print(f"  {i:2d}. {name:<35} = {value:.6f}")

    # Check for NaN/Inf
    nan_count = feature_values.isna().sum()
    inf_count = np.isinf(feature_values).sum()

    if nan_count > 0:
        print(f"\n⚠️ NaN values: {nan_count}")
        nan_features = feature_values[feature_values.isna()].index.tolist()
        print(f"  NaN features: {nan_features}")

    if inf_count > 0:
        print(f"\n⚠️ Inf values: {inf_count}")
        inf_features = feature_values[np.isinf(feature_values)].index.tolist()
        print(f"  Inf features: {inf_features}")

    # Scale features
    try:
        features_array = feature_values.values.reshape(1, -1)
        scaled_features = scaler.transform(features_array)

        print(f"\n✅ Scaling successful")
        print(f"  Scaled values (first 10): {scaled_features[0][:10]}")
        print(f"  Scaled min: {scaled_features.min():.4f}")
        print(f"  Scaled max: {scaled_features.max():.4f}")
        print(f"  Scaled mean: {scaled_features.mean():.4f}")

        # Predict
        prediction = model.predict_proba(scaled_features)
        prob = prediction[0][1]

        print(f"\n✅ Prediction successful")
        print(f"  LONG probability: {prob:.4f} ({prob*100:.2f}%)")

        # Compare with production log
        print(f"\n📝 Production Log (2025-10-19 05:40):")
        print(f"  Timestamp: 2025-10-18 20:40:00")
        print(f"  LONG prob: 0.9543 (95.43%)")

        print(f"\n🔍 COMPARISON:")
        print(f"  Training/Backtest: {prob*100:.2f}%")
        print(f"  Production Log: 95.43%")
        print(f"  Difference: {(95.43 - prob*100):.2f}%")

    except Exception as e:
        print(f"\n❌ Error: {e}")

# ============================================================================
# STEP 4: Check scaler fit statistics
# ============================================================================
print("\n" + "="*80)
print("STEP 4: SCALER STATISTICS")
print("="*80)

print(f"\n📊 Scaler Information:")
print(f"  Type: {type(scaler).__name__}")
print(f"  N features: {scaler.n_features_in_}")

if hasattr(scaler, 'data_min_'):
    print(f"\n  Data range (training):")
    print(f"    Min values (first 5): {scaler.data_min_[:5]}")
    print(f"    Max values (first 5): {scaler.data_max_[:5]}")

if hasattr(scaler, 'scale_'):
    print(f"\n  Scale factors (first 5): {scaler.scale_[:5]}")

# ============================================================================
# STEP 5: Training data check (if available)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: TRAINING DATA VERIFICATION")
print("="*80)

print(f"\n📁 Checking for training metadata...")
metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
if metadata_path.exists():
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✅ Metadata found:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
else:
    print(f"⚠️  No metadata file found")

print("\n" + "="*80)
print("TRACE COMPLETE")
print("="*80)
