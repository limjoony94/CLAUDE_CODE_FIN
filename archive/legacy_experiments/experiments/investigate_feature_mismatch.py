"""
Investigate: Why 47 features instead of 44?
Check what columns are created and compare with expected features
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("FEATURE MISMATCH INVESTIGATION")
print("="*80)

# Load expected feature names
long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    expected_features = [line.strip() for line in f.readlines()]

print(f"\n📋 Expected LONG features: {len(expected_features)}")
print(f"  {expected_features[:10]}...")

# Load sample data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file).tail(500)  # Just 500 rows for testing

print(f"\n📊 Calculating features on {len(df)} candles...")
df_features = calculate_all_features(df.copy())
print(f"  After calculate_all_features: {len(df_features.columns)} columns")
print(f"  Columns: {list(df_features.columns)[:20]}...")

df_features = prepare_exit_features(df_features)
print(f"  After prepare_exit_features: {len(df_features.columns)} columns")

# List ALL columns
print(f"\n📝 ALL COLUMNS ({len(df_features.columns)}):")
all_columns = list(df_features.columns)
for i, col in enumerate(all_columns, 1):
    print(f"  {i:3d}. {col}")

# Check which columns are in expected but not in DataFrame
print(f"\n❌ Expected features NOT in DataFrame:")
missing = set(expected_features) - set(all_columns)
if missing:
    for feat in missing:
        print(f"  - {feat}")
else:
    print("  None (all expected features exist)")

# Check which columns are in DataFrame but not in expected
print(f"\n❓ DataFrame columns NOT in expected features:")
extra = set(all_columns) - set(expected_features) - {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
if extra:
    for feat in sorted(extra):
        print(f"  - {feat}")
else:
    print("  None (no extra features)")

# Try to extract features in the exact order
print(f"\n🔍 Extracting features in expected order:")
try:
    test_row = df_features.iloc[0:1]
    features = test_row[expected_features].values
    print(f"  ✅ Successfully extracted {features.shape[1]} features")
    print(f"  Shape: {features.shape}")
except KeyError as e:
    print(f"  ❌ KeyError: {e}")
    print(f"\n  Missing columns:")
    for feat in expected_features:
        if feat not in df_features.columns:
            print(f"    - {feat}")

# Check for duplicate column names
print(f"\n🔄 Checking for duplicate column names:")
col_counts = pd.Series(all_columns).value_counts()
duplicates = col_counts[col_counts > 1]
if len(duplicates) > 0:
    print(f"  ⚠️  Found {len(duplicates)} duplicate column names:")
    for col, count in duplicates.items():
        print(f"    - {col}: appears {count} times")
else:
    print(f"  ✅ No duplicate column names")

# Check for duplicate feature names in expected list
print(f"\n🔄 Checking for duplicates in expected feature list:")
feature_counts = pd.Series(expected_features).value_counts()
dup_features = feature_counts[feature_counts > 1]
if len(dup_features) > 0:
    print(f"  ⚠️  Found {len(dup_features)} duplicate feature names:")
    for feat, count in dup_features.items():
        print(f"    - {feat}: appears {count} times")
        # Show positions
        positions = [i for i, f in enumerate(expected_features) if f == feat]
        print(f"      Positions: {positions}")
else:
    print(f"  ✅ No duplicate feature names")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
