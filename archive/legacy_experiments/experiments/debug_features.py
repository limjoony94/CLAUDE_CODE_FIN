"""Debug which features are created"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.experiments.feature_utils import calculate_short_features_optimized

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file).head(5000)  # Small sample

print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

print(f"\nBefore SHORT features: {len(df.columns)} columns")
df_before_short = set(df.columns)

df = calculate_short_features_optimized(df)

print(f"After SHORT features: {len(df.columns)} columns")
df_after_short = set(df.columns)

new_features = sorted(df_after_short - df_before_short)
print(f"\nNew features added by calculate_short_features_optimized: {len(new_features)}")
for i, feat in enumerate(new_features, 1):
    print(f"{i:2d}. {feat}")

# Load expected features
short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    expected_features = [line.strip() for line in f.readlines()]

print(f"\n\nExpected SHORT features: {len(expected_features)}")
for i, feat in enumerate(expected_features, 1):
    print(f"{i:2d}. {feat}")

# Find differences
missing = set(expected_features) - set(new_features)
extra = set(new_features) - set(expected_features)

if missing:
    print(f"\n❌ MISSING features ({len(missing)}):")
    for feat in sorted(missing):
        print(f"   - {feat}")

if extra:
    print(f"\n❌ EXTRA features ({len(extra)}):")
    for feat in sorted(extra):
        print(f"   - {feat}")

if not missing and not extra:
    print(f"\n✅ Perfect match! All {len(expected_features)} features present.")
