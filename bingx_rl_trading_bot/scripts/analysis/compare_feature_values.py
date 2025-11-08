"""
Feature Value Comparison - Find Root Cause
Compare feature values at same timestamp between 28-day and 2.5-day backtest
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features
from src.api.bingx_client import BingXClient
import yaml
import joblib

print("="*80)
print("FEATURE VALUE COMPARISON AT SAME TIMESTAMP")
print("="*80)

# Load models to get feature lists
MODELS_DIR = PROJECT_ROOT / "models"
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")

# Target timestamps to compare
target_timestamps = [
    "2025-11-01 19:40:00",  # Trade #1 entry in 28-day
    "2025-11-01 20:10:00",  # Trade #1 entry in 2.5-day
    "2025-11-02 17:30:00",  # Trade #2 entry (same in both)
    "2025-11-03 00:15:00",  # Stop Loss entry
    "2025-11-03 00:20:00",  # Stop Loss entry
]

print("\n1. Loading 28-day CSV data...")
csv_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_raw_latest4weeks_20251104_014102.csv"
df_28day = pd.read_csv(csv_file)
df_28day['timestamp'] = pd.to_datetime(df_28day['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_28day[col] = df_28day[col].astype(float)

print(f"   Loaded: {len(df_28day)} candles")
print(f"   Range: {df_28day['timestamp'].min()} to {df_28day['timestamp'].max()}")

print("\n2. Calculating features for 28-day data...")
df_28day_features = calculate_all_features_enhanced_v2(df_28day.copy(), phase='phase1')
df_28day_features = prepare_exit_features(df_28day_features)
print(f"   Features: {len(df_28day_features)} rows")

print("\n3. Loading 2.5-day API data...")
config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key']
)
klines = client.get_klines("BTC-USDT", "5m", limit=1000)
df_2_5day = pd.DataFrame(klines)
df_2_5day['timestamp'] = pd.to_datetime(df_2_5day['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_2_5day[col] = df_2_5day[col].astype(float)

print(f"   Loaded: {len(df_2_5day)} candles")
print(f"   Range: {df_2_5day['timestamp'].min()} to {df_2_5day['timestamp'].max()}")

print("\n4. Calculating features for 2.5-day data...")
df_2_5day_features = calculate_all_features_enhanced_v2(df_2_5day.copy(), phase='phase1')
df_2_5day_features = prepare_exit_features(df_2_5day_features)
print(f"   Features: {len(df_2_5day_features)} rows")

print("\n" + "="*80)
print("COMPARING FEATURES AT TARGET TIMESTAMPS")
print("="*80)

for target_ts in target_timestamps:
    print(f"\n{'='*80}")
    print(f"Timestamp: {target_ts}")
    print(f"{'='*80}")

    # Find matching rows
    ts_28day = df_28day_features[df_28day_features['timestamp'] == target_ts]
    ts_2_5day = df_2_5day_features[df_2_5day_features['timestamp'] == target_ts]

    if len(ts_28day) == 0:
        print(f"❌ Not found in 28-day data")
    else:
        print(f"✅ Found in 28-day data (row index: {ts_28day.index[0]})")

    if len(ts_2_5day) == 0:
        print(f"❌ Not found in 2.5-day data")
    else:
        print(f"✅ Found in 2.5-day data (row index: {ts_2_5day.index[0]})")

    if len(ts_28day) > 0 and len(ts_2_5day) > 0:
        # Compare basic OHLCV
        print(f"\n📊 OHLCV Comparison:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            val_28 = ts_28day.iloc[0][col]
            val_25 = ts_2_5day.iloc[0][col]
            match = "✅" if abs(val_28 - val_25) < 0.01 else "❌"
            print(f"   {col:8s}: 28day={val_28:12.2f}  |  2.5day={val_25:12.2f}  {match}")

        # Get LONG entry features
        features_28 = {}
        features_25 = {}
        for feat in long_entry_features:
            if feat in ts_28day.columns and feat in ts_2_5day.columns:
                features_28[feat] = ts_28day.iloc[0][feat]
                features_25[feat] = ts_2_5day.iloc[0][feat]

        # Compare features
        print(f"\n🔍 Feature Comparison (showing top 10 differences):")
        diffs = []
        for feat in features_28.keys():
            val_28 = features_28[feat]
            val_25 = features_25[feat]
            if pd.isna(val_28) or pd.isna(val_25):
                diff_pct = float('inf')
            elif abs(val_28) < 1e-10 and abs(val_25) < 1e-10:
                diff_pct = 0
            elif abs(val_28) < 1e-10:
                diff_pct = 100 if abs(val_25) > 1e-10 else 0
            else:
                diff_pct = abs(val_28 - val_25) / abs(val_28) * 100
            diffs.append({
                'feature': feat,
                'val_28': val_28,
                'val_25': val_25,
                'diff': val_28 - val_25,
                'diff_pct': diff_pct
            })

        diffs_sorted = sorted(diffs, key=lambda x: x['diff_pct'], reverse=True)
        for i, d in enumerate(diffs_sorted[:10]):
            if d['diff_pct'] < 0.01:
                break
            print(f"   {i+1}. {d['feature']}")
            print(f"      28day: {d['val_28']:.6f}  |  2.5day: {d['val_25']:.6f}")
            print(f"      Diff: {d['diff']:.6f} ({d['diff_pct']:.2f}%)")

        # Calculate probabilities
        features_28_list = [features_28[f] for f in long_entry_features]
        features_25_list = [features_25[f] for f in long_entry_features]

        features_28_scaled = long_entry_scaler.transform([features_28_list])
        features_25_scaled = long_entry_scaler.transform([features_25_list])

        prob_28 = long_entry_model.predict_proba(features_28_scaled)[0][1]
        prob_25 = long_entry_model.predict_proba(features_25_scaled)[0][1]

        print(f"\n🎯 LONG Entry Probability:")
        print(f"   28day: {prob_28:.4f} ({prob_28*100:.2f}%)")
        print(f"   2.5day: {prob_25:.4f} ({prob_25*100:.2f}%)")
        print(f"   Difference: {abs(prob_28 - prob_25):.4f} ({abs(prob_28 - prob_25)*100:.2f}%)")
        print(f"   Threshold: 0.80 (80%)")
        print(f"   28day would {'ENTER' if prob_28 >= 0.80 else 'NOT ENTER'}")
        print(f"   2.5day would {'ENTER' if prob_25 >= 0.80 else 'NOT ENTER'}")

print("\n" + "="*80)
print("✅ Feature Comparison Complete")
print("="*80)
