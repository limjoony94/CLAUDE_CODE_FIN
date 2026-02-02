"""
Verify EXACT match between backtest and production feature calculation
Same input → Same features → Same prediction (no tolerance)
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
print("EXACT MATCH VERIFICATION: BACKTEST vs PRODUCTION")
print("="*80)

# ============================================================================
# STEP 1: Load model and feature names
# ============================================================================
model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"

print("\n📁 Loading model...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
scaler = joblib.load(scaler_path)
with open(features_path, 'r') as f:
    feature_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"  Model features: {model.n_features_in_}")
print(f"  Feature list: {len(feature_names)}")
print(f"  Scaler features: {scaler.n_features_in_}")

# ============================================================================
# STEP 2: Load data and calculate features (BACKTEST METHOD)
# ============================================================================
print("\n📊 Loading full dataset...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  Total candles: {len(df):,}")

print("\n🔧 Calculating features (BACKTEST METHOD)...")
df_backtest = calculate_all_features(df.copy())
df_backtest = prepare_exit_features(df_backtest)

# ============================================================================
# STEP 3: Calculate features (PRODUCTION METHOD - last 1440 candles)
# ============================================================================
print("\n🔧 Calculating features (PRODUCTION METHOD - last 1440)...")
df_production = df.tail(1440).copy()
df_production = calculate_all_features(df_production)
df_production = prepare_exit_features(df_production)

# ============================================================================
# STEP 4: Test on same timestamp from production log
# ============================================================================
target_timestamp = "2025-10-18 20:40:00"
print(f"\n🎯 Target: {target_timestamp}")

# Get from backtest
row_backtest = df_backtest[df_backtest['timestamp'] == target_timestamp]
# Get from production
row_production = df_production[df_production['timestamp'] == target_timestamp]

if len(row_backtest) == 0 or len(row_production) == 0:
    print(f"❌ Timestamp not found")
    sys.exit(1)

print(f"  Price: ${row_backtest['close'].iloc[0]:,.2f}")

# ============================================================================
# STEP 5: Extract features and compare EXACT VALUES
# ============================================================================
print(f"\n📋 Feature Comparison (first 20):")
print(f"{'Feature':<35} {'Backtest':<15} {'Production':<15} {'Match':<10}")
print("-"*80)

features_backtest_series = row_backtest[feature_names].iloc[0]
features_production_series = row_production[feature_names].iloc[0]

# Convert to numpy arrays for exact comparison
features_backtest = features_backtest_series.values
features_production = features_production_series.values

all_match = True
mismatch_count = 0
mismatch_details = []

for i, feat_name in enumerate(feature_names[:20]):
    val_bt = features_backtest[i]
    val_prod = features_production[i]
    match = "✅" if np.isclose(val_bt, val_prod, rtol=1e-12) else "❌"

    if not np.isclose(val_bt, val_prod, rtol=1e-12):
        all_match = False
        mismatch_count += 1
        mismatch_details.append((feat_name, val_bt, val_prod, abs(val_bt - val_prod)))

    print(f"{feat_name:<35} {val_bt:<15.6f} {val_prod:<15.6f} {match:<10}")

# Check all features
for i, feat_name in enumerate(feature_names[20:], start=20):
    val_bt = features_backtest[i]
    val_prod = features_production[i]
    if not np.isclose(val_bt, val_prod, rtol=1e-12):
        all_match = False
        mismatch_count += 1
        mismatch_details.append((feat_name, val_bt, val_prod, abs(val_bt - val_prod)))

print(f"\n📊 Total features: {len(feature_names)}")
print(f"  Matching: {len(feature_names) - mismatch_count}")
print(f"  Mismatching: {mismatch_count}")

if mismatch_count > 0:
    print(f"\n⚠️ MISMATCHES FOUND ({mismatch_count}):")
    print(f"{'Feature':<35} {'Backtest':<15} {'Production':<15} {'Diff':<15}")
    print("-"*80)
    for feat, val_bt, val_prod, diff in sorted(mismatch_details, key=lambda x: x[3], reverse=True):
        print(f"{feat:<35} {val_bt:<15.6f} {val_prod:<15.6f} {diff:<15.6f}")

# ============================================================================
# STEP 6: Scale and predict
# ============================================================================
print(f"\n🔬 Scaling and Prediction:")

# Backtest
features_bt_array = features_backtest.reshape(1, -1)
scaled_bt = scaler.transform(features_bt_array)
pred_bt = model.predict_proba(scaled_bt)
prob_bt = pred_bt[0][1]

# Production
features_prod_array = features_production.reshape(1, -1)
scaled_prod = scaler.transform(features_prod_array)
pred_prod = model.predict_proba(scaled_prod)
prob_prod = pred_prod[0][1]

print(f"\nBacktest Method:")
print(f"  LONG probability: {prob_bt:.6f} ({prob_bt*100:.4f}%)")

print(f"\nProduction Method:")
print(f"  LONG probability: {prob_prod:.6f} ({prob_prod*100:.4f}%)")

print(f"\nProduction Log (from 2025-10-19 05:40):")
print(f"  LONG probability: 0.9543 (95.43%)")

print(f"\n🔍 VERIFICATION:")
print(f"  Backtest vs Production: {abs(prob_bt - prob_prod):.6f} difference")
print(f"  Backtest vs Log: {abs(prob_bt - 0.9543):.6f} difference")
print(f"  Production vs Log: {abs(prob_prod - 0.9543):.6f} difference")

if abs(prob_bt - prob_prod) < 1e-10:
    print(f"\n✅ EXACT MATCH: Backtest and Production methods are identical")
else:
    print(f"\n❌ MISMATCH: Backtest and Production methods differ")
    print(f"  → Feature calculation or order issue detected")

if abs(prob_bt - 0.9543) < 0.01:
    print(f"\n✅ Backtest matches production log")
else:
    print(f"\n❌ Backtest does NOT match production log")
    print(f"  → Something different in actual production bot")

# ============================================================================
# STEP 7: Check feature order
# ============================================================================
print(f"\n📋 Feature Order Verification:")
print(f"  Checking if feature names match between backtest and production DataFrames...")

backtest_cols = list(df_backtest.columns)
production_cols = list(df_production.columns)

if backtest_cols == production_cols:
    print(f"  ✅ Column order matches")
else:
    print(f"  ❌ Column order differs")
    print(f"  Backtest columns: {len(backtest_cols)}")
    print(f"  Production columns: {len(production_cols)}")

    # Find differences
    only_in_backtest = set(backtest_cols) - set(production_cols)
    only_in_production = set(production_cols) - set(backtest_cols)

    if only_in_backtest:
        print(f"\n  Only in backtest: {only_in_backtest}")
    if only_in_production:
        print(f"  Only in production: {only_in_production}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
