"""
Debug Exit Features: Check if all required features exist
"""
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Load Exit feature lists
print("Loading Exit model feature requirements...")
long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"\nLONG Exit expects: {len(long_exit_features)} features")
print(f"SHORT Exit expects: {len(short_exit_features)} features")

# Load sample data
print("\nLoading sample data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
df_sample = df_full.tail(500).copy()  # Use last 500 candles
print(f"Sample: {len(df_sample)} candles")

# Calculate features
print("\nCalculating basic features...")
df = calculate_all_features(df_sample)
print(f"After basic: {len(df.columns)} columns")

print("\nCalculating exit features...")
df = prepare_exit_features(df)
print(f"After exit features: {len(df.columns)} columns")

# Check LONG features
print("\n" + "="*80)
print("LONG EXIT FEATURE CHECK")
print("="*80)
missing_long = []
for feat in long_exit_features:
    if feat not in df.columns:
        missing_long.append(feat)
        print(f"❌ MISSING: {feat}")
    else:
        sample_val = df[feat].iloc[-1]
        print(f"✅ {feat}: {sample_val:.6f}" if isinstance(sample_val, (int, float)) else f"✅ {feat}: {sample_val}")

if missing_long:
    print(f"\n⚠️ {len(missing_long)} features MISSING for LONG Exit!")
else:
    print(f"\n✅ All {len(long_exit_features)} LONG Exit features present!")

# Check SHORT features
print("\n" + "="*80)
print("SHORT EXIT FEATURE CHECK")
print("="*80)
missing_short = []
for feat in short_exit_features:
    if feat not in df.columns:
        missing_short.append(feat)
        print(f"❌ MISSING: {feat}")
    else:
        sample_val = df[feat].iloc[-1]
        print(f"✅ {feat}: {sample_val:.6f}" if isinstance(sample_val, (int, float)) else f"✅ {feat}: {sample_val}")

if missing_short:
    print(f"\n⚠️ {len(missing_short)} features MISSING for SHORT Exit!")
else:
    print(f"\n✅ All {len(short_exit_features)} SHORT Exit features present!")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
if missing_long or missing_short:
    print("❌ Exit features are INCOMPLETE - ML Exit will FAIL!")
    print(f"\nMissing LONG: {missing_long}")
    print(f"Missing SHORT: {missing_short}")
else:
    print("✅ All Exit features present - ML Exit should work!")
