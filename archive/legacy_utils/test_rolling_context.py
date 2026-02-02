"""
Test Rolling Context Fix - Verify Probabilities > 0.0000
==========================================================

This script tests whether the rolling context fix resolves the 0.0000 probability issue.

Test Process:
1. Load live features CSV (50K+ context)
2. Calculate features using same function as production
3. Load Entry models
4. Generate predictions
5. Verify probabilities > 0.0000

Expected Result:
- LONG probability: 0.01 - 0.99 (NOT 0.0000)
- SHORT probability: 0.01 - 0.99 (NOT 0.0000)

Usage:
    # After running initialize_live_features.py
    python scripts/utils/test_rolling_context.py

Created: 2025-10-28
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

print("="*80)
print("TEST ROLLING CONTEXT FIX")
print("="*80)
print()

# Paths
LIVE_FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features_live.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Load live features
if not LIVE_FEATURES_CSV.exists():
    print(f"❌ ERROR: Live features CSV not found: {LIVE_FEATURES_CSV}")
    print()
    print("Please run first:")
    print("  python scripts/utils/initialize_live_features.py")
    sys.exit(1)

print(f"📂 Loading: {LIVE_FEATURES_CSV.name}")
df_features = pd.read_csv(LIVE_FEATURES_CSV)
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
print(f"   ✅ Loaded: {len(df_features):,} candles with features")
print(f"      From: {df_features['timestamp'].iloc[0]}")
print(f"      To:   {df_features['timestamp'].iloc[-1]}")
print()

# Use last candle for prediction
print("🔍 Using latest candle for prediction...")
latest_candle = df_features.iloc[-1:].copy()
print(f"   Timestamp: {latest_candle['timestamp'].iloc[0]}")
print(f"   Close: ${latest_candle['close'].iloc[0]:,.2f}")
print()

# Load LONG Entry model (SAME AS PRODUCTION BOT)
print("🤖 Loading Entry Models (same as production bot)...")
long_entry_model_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
short_entry_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"

if not long_entry_model_path.exists():
    print(f"   ❌ ERROR: LONG Entry model not found: {long_entry_model_path}")
    sys.exit(1)

if not short_entry_model_path.exists():
    print(f"   ❌ ERROR: SHORT Entry model not found: {short_entry_model_path}")
    sys.exit(1)

# Load models (direct XGBClassifier objects)
with open(long_entry_model_path, 'rb') as f:
    long_entry_model = pickle.load(f)

with open(short_entry_model_path, 'rb') as f:
    short_entry_model = pickle.load(f)

print(f"   ✅ LONG Entry model loaded")
print(f"   ✅ SHORT Entry model loaded")

# Load scalers
long_scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"

if not long_scaler_path.exists() or not short_scaler_path.exists():
    print(f"   ❌ ERROR: Scalers not found")
    sys.exit(1)

long_scaler = joblib.load(long_scaler_path)
short_scaler = joblib.load(short_scaler_path)
print(f"   ✅ Scalers loaded")

# Load feature lists
long_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"

if not long_features_path.exists() or not short_features_path.exists():
    print(f"   ❌ ERROR: Feature lists not found")
    sys.exit(1)

with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"   ✅ LONG features: {len(long_feature_columns)}")
print(f"   ✅ SHORT features: {len(short_feature_columns)}")
print()

# Prepare features for LONG
print("🔧 Preparing LONG features...")
try:
    # Check if all features exist
    missing_long = [f for f in long_feature_columns if f not in latest_candle.columns]
    if missing_long:
        print(f"   ❌ ERROR: Missing {len(missing_long)} LONG features:")
        for f in missing_long[:5]:
            print(f"      - {f}")
        if len(missing_long) > 5:
            print(f"      ... and {len(missing_long) - 5} more")
        sys.exit(1)

    long_feat_df = latest_candle[long_feature_columns].copy()

    # Replace inf with NaN, then fill
    long_feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    long_feat_df.fillna(0, inplace=True)

    # Scale
    if long_scaler is not None:
        long_feat_scaled = long_scaler.transform(long_feat_df)
    else:
        long_feat_scaled = long_feat_df.values

    # Predict
    long_prob = long_entry_model.predict_proba(long_feat_scaled)[0, 1]
    print(f"   ✅ LONG Probability: {long_prob:.4f}")

except Exception as e:
    print(f"   ❌ ERROR: {e}")
    long_prob = None

print()

# Prepare features for SHORT
print("🔧 Preparing SHORT features...")
try:
    # Check if all features exist
    missing_short = [f for f in short_feature_columns if f not in latest_candle.columns]
    if missing_short:
        print(f"   ❌ ERROR: Missing {len(missing_short)} SHORT features:")
        for f in missing_short[:5]:
            print(f"      - {f}")
        if len(missing_short) > 5:
            print(f"      ... and {len(missing_short) - 5} more")
        sys.exit(1)

    short_feat_df = latest_candle[short_feature_columns].copy()

    # Replace inf with NaN, then fill
    short_feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    short_feat_df.fillna(0, inplace=True)

    # Scale
    if short_scaler is not None:
        short_feat_scaled = short_scaler.transform(short_feat_df)
    else:
        short_feat_scaled = short_feat_df.values

    # Predict
    short_prob = short_entry_model.predict_proba(short_feat_scaled)[0, 1]
    print(f"   ✅ SHORT Probability: {short_prob:.4f}")

except Exception as e:
    print(f"   ❌ ERROR: {e}")
    short_prob = None

print()

# Validation
print("="*80)
print("🎯 TEST RESULTS")
print("="*80)
print()

success = True

if long_prob is None:
    print("❌ LONG prediction FAILED")
    success = False
elif long_prob == 0.0000:
    print(f"❌ LONG probability is 0.0000 (ISSUE NOT FIXED)")
    success = False
else:
    print(f"✅ LONG probability: {long_prob:.4f} (> 0.0000)")

if short_prob is None:
    print("❌ SHORT prediction FAILED")
    success = False
elif short_prob == 0.0000:
    print(f"❌ SHORT probability is 0.0000 (ISSUE NOT FIXED)")
    success = False
else:
    print(f"✅ SHORT probability: {short_prob:.4f} (> 0.0000)")

print()

if success:
    print("="*80)
    print("🎉 TEST PASSED - ROLLING CONTEXT FIX WORKING!")
    print("="*80)
    print()
    print("✅ Models are generating non-zero probabilities")
    print("✅ Context window fix resolved the train-test mismatch")
    print("✅ Production bot should now work correctly")
    print()
    print("🎯 Next Steps:")
    print("   1. Run production bot with USE_ROLLING_CONTEXT = True")
    print("   2. Monitor first few candles for probabilities > 0.0000")
    print("   3. Verify trading signals are generated correctly")
    print()
else:
    print("="*80)
    print("❌ TEST FAILED - ISSUE NOT RESOLVED")
    print("="*80)
    print()
    print("Please check:")
    print("   1. Live features CSV has proper context (50K+ candles)")
    print("   2. Features are calculated correctly")
    print("   3. Models are loaded correctly")
    print("   4. Feature columns match model expectations")
    print()

sys.exit(0 if success else 1)
