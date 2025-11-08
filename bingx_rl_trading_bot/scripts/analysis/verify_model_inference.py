#!/usr/bin/env python3
"""
Model Inference Verification
=============================
Verify that identical features produce identical signals.
If not, there's an issue in model loading or inference.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pickle
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

print("=" * 80)
print("MODEL INFERENCE VERIFICATION")
print("=" * 80)

# Test timestamps
test_signals = [
    {
        'timestamp_utc': '2025-10-26 15:05:00',
        'long_prod': 0.2126,
        'short_prod': 0.2276
    },
    {
        'timestamp_utc': '2025-10-26 15:10:00',
        'long_prod': 0.1993,
        'short_prod': 0.2260
    }
]

# 1. Load CSV data
CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
print(f"\n1. Loading CSV data...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter to test period + lookback
from datetime import timedelta
test_times = [pd.to_datetime(s['timestamp_utc']) for s in test_signals]
min_time = min(test_times)
max_time = max(test_times)
lookback_minutes = 1000 * 5

df_filtered = df[
    (df['timestamp'] >= min_time - timedelta(minutes=lookback_minutes)) &
    (df['timestamp'] <= max_time)
].copy()

print(f"   ✅ Loaded {len(df_filtered)} candles")
print(f"   Period: {df_filtered['timestamp'].min()} ~ {df_filtered['timestamp'].max()}")

# 2. Calculate features
print(f"\n2. Calculating features...")
df_features = calculate_all_features_enhanced_v2(df_filtered, phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"   ✅ Features calculated: {len(df_features)} rows")

# 3. Load models
print(f"\n3. Loading models...")
MODELS_DIR = PROJECT_ROOT / "models"

long_model = pickle.load(open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb'))
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines()]

short_model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb'))
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_features = [line.strip() for line in f.readlines()]

print(f"   ✅ Models loaded")
print(f"   LONG features: {len(long_features)}")
print(f"   SHORT features: {len(short_features)}")

# 4. Generate signals MULTIPLE TIMES to check consistency
print(f"\n4. Testing model inference consistency...")
print("=" * 80)

for signal in test_signals:
    ts = pd.to_datetime(signal['timestamp_utc'])
    print(f"\n📍 {ts}")
    print("-" * 80)

    row = df_features[df_features['timestamp'] == ts]
    if len(row) == 0:
        print("   ❌ Timestamp not found in features")
        continue

    row = row.iloc[0]

    # Extract features and ensure float type
    long_X = row[long_features].values.astype(float).reshape(1, -1)
    short_X = row[short_features].values.astype(float).reshape(1, -1)

    # Check for NaN/Inf
    if np.any(np.isnan(long_X)) or np.any(np.isinf(long_X)):
        print("   ⚠️ LONG features contain NaN/Inf!")
    if np.any(np.isnan(short_X)) or np.any(np.isinf(short_X)):
        print("   ⚠️ SHORT features contain NaN/Inf!")

    # Scale
    long_X_scaled = long_scaler.transform(long_X)
    short_X_scaled = short_scaler.transform(short_X)

    # Predict multiple times to check consistency
    print(f"\n   Running inference 5 times to verify consistency:")
    print(f"   {'Run':<5} {'LONG':>10} {'SHORT':>10}")
    print("   " + "-" * 30)

    long_preds = []
    short_preds = []

    for i in range(5):
        long_pred = long_model.predict_proba(long_X_scaled)[0, 1]
        short_pred = short_model.predict_proba(short_X_scaled)[0, 1]

        long_preds.append(long_pred)
        short_preds.append(short_pred)

        print(f"   {i+1:<5} {long_pred:>10.4f} {short_pred:>10.4f}")

    # Check consistency
    long_std = np.std(long_preds)
    short_std = np.std(short_preds)

    print(f"\n   Consistency check:")
    print(f"     LONG std:  {long_std:.10f} {'✅ Consistent' if long_std < 1e-10 else '❌ Inconsistent'}")
    print(f"     SHORT std: {short_std:.10f} {'✅ Consistent' if short_std < 1e-10 else '❌ Inconsistent'}")

    # Compare with production
    long_avg = np.mean(long_preds)
    short_avg = np.mean(short_preds)

    long_diff = long_avg - signal['long_prod']
    short_diff = short_avg - signal['short_prod']

    print(f"\n   Production comparison:")
    print(f"     LONG:  Backtest={long_avg:.4f}, Production={signal['long_prod']:.4f}, Diff={long_diff:+.4f}")
    print(f"     SHORT: Backtest={short_avg:.4f}, Production={signal['short_prod']:.4f}, Diff={short_diff:+.4f}")

    # Show feature values for debugging
    print(f"\n   Sample feature values (first 10):")
    print(f"   {'Feature':<40} {'Value':>15}")
    print("   " + "-" * 60)
    for feat in long_features[:10]:
        val = row[feat]
        print(f"   {feat:<40} {val:>15.6f}")

print("\n" + "=" * 80)
print("INFERENCE VERIFICATION COMPLETE")
print("=" * 80)

print("""
🔍 Analysis:

If model inference is CONSISTENT (std < 1e-10):
  ✅ Model and scaler are deterministic
  ✅ No randomness in predictions
  ✅ Same input always gives same output

If signals MATCH production (diff < 0.0001):
  ✅ Backtest pipeline matches production pipeline
  ✅ Feature calculation is identical
  ✅ Model loading is correct

If signals DON'T MATCH production:
  ⚠️ Possible causes:
    1. Different model file used in production
    2. Different scaler used in production
    3. Feature calculation differs
    4. Data preprocessing differs
""")
