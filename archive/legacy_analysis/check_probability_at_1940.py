"""
Check LONG probability at Nov 1 19:40 for both datasets
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Load models
model_entry_long = joblib.load(PROJECT_ROOT / "models" / "xgboost_long_entry_enhanced_20251024_012445.pkl")
with open(PROJECT_ROOT / "models" / "xgboost_long_entry_enhanced_20251024_012445_features.txt") as f:
    entry_long_features = [line.strip() for line in f.readlines()]

print("=" * 80)
print("PROBABILITY CHECK AT NOV 1 19:40:00")
print("=" * 80)

# ====================
# 1. Load 28-day CSV data
# ====================
print("\n1️⃣ Loading 28-day CSV data...")
csv_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_raw_latest4weeks_20251104_014102.csv"
df_28day = pd.read_csv(csv_file)
df_28day['timestamp'] = pd.to_datetime(df_28day['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_28day[col] = df_28day[col].astype(float)

print(f"   Loaded: {len(df_28day)} candles")
print(f"   Range: {df_28day['timestamp'].min()} to {df_28day['timestamp'].max()}")

# Calculate features
print("   Calculating features...")
df_28day_features = calculate_all_features_enhanced_v2(df_28day.copy(), phase='phase1')
df_28day_features = prepare_exit_features(df_28day_features)
print(f"   Features: {len(df_28day_features)} rows")

# ====================
# 2. Load 2.5-day API data
# ====================
print("\n2️⃣ Loading 2.5-day API data...")
config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key']
)

klines = client.get_klines('BTC-USDT', '5m', limit=1000)
df_2_5day = pd.DataFrame(klines)
df_2_5day['timestamp'] = pd.to_datetime(df_2_5day['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_2_5day[col] = df_2_5day[col].astype(float)

print(f"   Loaded: {len(df_2_5day)} candles")
print(f"   Range: {df_2_5day['timestamp'].min()} to {df_2_5day['timestamp'].max()}")

# Calculate features
print("   Calculating features...")
df_2_5day_features = calculate_all_features_enhanced_v2(df_2_5day.copy(), phase='phase1')
df_2_5day_features = prepare_exit_features(df_2_5day_features)
print(f"   Features: {len(df_2_5day_features)} rows")

# ====================
# 3. Find Nov 1 19:40 in both
# ====================
target_time = pd.to_datetime('2025-11-01 19:40:00')

print("\n" + "=" * 80)
print("CHECKING NOV 1 19:40:00")
print("=" * 80)

# Check 28-day
if target_time in df_28day_features['timestamp'].values:
    row_28 = df_28day_features[df_28day_features['timestamp'] == target_time].iloc[0]
    idx_28 = df_28day_features[df_28day_features['timestamp'] == target_time].index[0]
    print(f"\n✅ 28-day: Found at row {idx_28}")

    # Get features
    X_28 = row_28[entry_long_features].values.reshape(1, -1)

    # Predict
    prob_28 = model_entry_long.predict_proba(X_28)[0, 1]

    print(f"   LONG probability: {prob_28:.6f} ({prob_28*100:.2f}%)")
    print(f"   Decision: {'ENTER' if prob_28 >= 0.80 else 'NOT ENTER'} (threshold: 0.80)")
else:
    print(f"\n❌ 28-day: NOT FOUND")

# Check 2.5-day
if target_time in df_2_5day_features['timestamp'].values:
    row_25 = df_2_5day_features[df_2_5day_features['timestamp'] == target_time].iloc[0]
    idx_25 = df_2_5day_features[df_2_5day_features['timestamp'] == target_time].index[0]
    print(f"\n✅ 2.5-day: Found at row {idx_25}")

    # Get features
    X_25 = row_25[entry_long_features].values.reshape(1, -1)

    # Predict
    prob_25 = model_entry_long.predict_proba(X_25)[0, 1]

    print(f"   LONG probability: {prob_25:.6f} ({prob_25*100:.2f}%)")
    print(f"   Decision: {'ENTER' if prob_25 >= 0.80 else 'NOT ENTER'} (threshold: 0.80)")
else:
    print(f"\n❌ 2.5-day: NOT FOUND")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

if target_time in df_28day_features['timestamp'].values and target_time in df_2_5day_features['timestamp'].values:
    diff = prob_28 - prob_25
    diff_pct = (diff / prob_28) * 100 if prob_28 != 0 else 0

    print(f"\nProbability difference:")
    print(f"   28-day:  {prob_28:.6f} ({prob_28*100:.2f}%)")
    print(f"   2.5-day: {prob_25:.6f} ({prob_25*100:.2f}%)")
    print(f"   Diff:    {diff:.6f} ({diff_pct:.2f}%)")

    if abs(diff) < 0.0001:
        print("\n✅ Probabilities are IDENTICAL!")
    else:
        print(f"\n{'✅' if abs(diff) < 0.01 else '❌'} Probabilities {'are SIMILAR' if abs(diff) < 0.01 else 'DIFFER significantly'}")

        if prob_28 >= 0.80 and prob_25 < 0.80:
            print("\n🔴 CRITICAL: 28-day would ENTER, but 2.5-day would NOT ENTER")
            print("   This explains why backtest trades are different!")
        elif prob_28 < 0.80 and prob_25 >= 0.80:
            print("\n🔴 CRITICAL: 2.5-day would ENTER, but 28-day would NOT ENTER")
            print("   This explains why backtest trades are different!")

print("\n" + "=" * 80)
