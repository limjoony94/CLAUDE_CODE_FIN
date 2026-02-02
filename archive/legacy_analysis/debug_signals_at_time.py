"""
Debug: Check signals at specific timestamps
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Load models
print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print("✅ Models loaded")
print()

# Load data
print("Loading data...")
df_path = DATA_DIR / "BTCUSDT_5m_updated.csv"
df_all = pd.read_csv(df_path)
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)

# Lookback
start_time = pd.Timestamp('2025-10-24 00:55:00')
LOOKBACK_CANDLES = 350
lookback_start = start_time - timedelta(minutes=5 * LOOKBACK_CANDLES)
end_time = pd.Timestamp('2025-10-27 01:00:00')

df = df_all[(df_all['timestamp'] >= lookback_start) & (df_all['timestamp'] <= end_time)].copy()
print(f"Loaded {len(df)} candles")
print()

# Calculate features
print("Calculating features...")
df_features = calculate_all_features_enhanced_v2(df)
print(f"✅ {len(df_features)} rows with features")
print()

# Check specific times (KST → UTC)
target_times_kst = [
    '2025-10-24 01:00:00',
    '2025-10-24 01:10:00'
]

print("="*80)
print("CHECKING TARGET TIMESTAMPS")
print("="*80)
print()

for time_kst in target_times_kst:
    ts_kst = pd.Timestamp(time_kst)
    ts_utc = ts_kst - timedelta(hours=9)  # KST to UTC

    print(f"Target: {time_kst} KST ({ts_utc} UTC)")

    # Find exact match
    exact = df_features[df_features['timestamp'] == ts_utc]

    if len(exact) > 0:
        row = exact.iloc[0]

        # Calculate LONG signal
        try:
            long_feat_df = pd.DataFrame([row])[long_feature_columns]
            if long_feat_df.shape[1] == len(long_feature_columns):
                long_feat_scaled = long_scaler.transform(long_feat_df.values)
                long_prob = long_model.predict_proba(long_feat_scaled)[0, 1]
            else:
                long_prob = 0.0
        except Exception as e:
            long_prob = 0.0
            print(f"  ❌ LONG error: {e}")

        # Calculate SHORT signal
        try:
            short_feat_df = pd.DataFrame([row])[short_feature_columns]
            if short_feat_df.shape[1] == len(short_feature_columns):
                short_feat_scaled = short_scaler.transform(short_feat_df.values)
                short_prob = short_model.predict_proba(short_feat_scaled)[0, 1]
            else:
                short_prob = 0.0
        except Exception as e:
            short_prob = 0.0
            print(f"  ❌ SHORT error: {e}")

        print(f"  ✅ FOUND")
        print(f"  Price: ${row['close']:,.2f}")
        print(f"  LONG:  {long_prob:.4f}")
        print(f"  SHORT: {short_prob:.4f}")
    else:
        # Find nearest
        df_near = df_features.copy()
        df_near['time_diff'] = (df_near['timestamp'] - ts_utc).abs()
        nearest = df_near.nsmallest(1, 'time_diff').iloc[0]

        print(f"  ❌ EXACT NOT FOUND")
        print(f"  Nearest: {nearest['timestamp']} (diff: {nearest['time_diff']})")

    print()

print("="*80)
print("CHECKING AROUND TARGET PERIOD")
print("="*80)
print()

# Check all signals in target period
target_start = pd.Timestamp('2025-10-23 15:55:00')  # 2025-10-24 00:55 KST
target_end = pd.Timestamp('2025-10-23 16:20:00')    # 2025-10-24 01:20 KST

df_target = df_features[(df_features['timestamp'] >= target_start) &
                        (df_features['timestamp'] <= target_end)].copy()

print(f"Period: {target_start} to {target_end} UTC")
print(f"Found {len(df_target)} candles")
print()

if len(df_target) > 0:
    for idx, row in df_target.iterrows():
        ts = row['timestamp']
        ts_kst = ts + timedelta(hours=9)

        # Calculate signals
        try:
            long_feat_df = pd.DataFrame([row])[long_feature_columns]
            long_feat_scaled = long_scaler.transform(long_feat_df.values)
            long_prob = long_model.predict_proba(long_feat_scaled)[0, 1]
        except:
            long_prob = 0.0

        try:
            short_feat_df = pd.DataFrame([row])[short_feature_columns]
            short_feat_scaled = short_scaler.transform(short_feat_df.values)
            short_prob = short_model.predict_proba(short_feat_scaled)[0, 1]
        except:
            short_prob = 0.0

        print(f"{ts_kst.strftime('%Y-%m-%d %H:%M:%S')} KST | LONG: {long_prob:.4f} | SHORT: {short_prob:.4f} | Price: ${row['close']:,.2f}")

print()
print("Debug complete!")
