#!/usr/bin/env python3
"""
Debug Signal Difference - Feature-Level Comparison
===================================================

Compare features between production and backtest for candles
with large signal differences to find root cause.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import re
from datetime import datetime
import pytz

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_FILE = PROJECT_ROOT / "logs" / "opportunity_gating_bot_4x_20251017.log"

print("="*80)
print("FEATURE-LEVEL DEBUG: Production vs Backtest")
print("="*80)

# Target candles with large differences
target_times_kst = [
    "2025-10-24 14:05",
    "2025-10-24 14:10",
    "2025-10-24 14:15"
]

# Parse production log signals
print("\n1. Parsing production signals...")
kst = pytz.timezone('Asia/Seoul')
prod_signals = []

model_deploy_time = datetime(2025, 10, 24, 11, 38, 0)
model_deploy_time_kst = kst.localize(model_deploy_time)

with open(LOG_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(r'\[Candle (\d{2}:\d{2}:\d{2}) KST\].*LONG: ([\d.]+) \| SHORT: ([\d.]+)', line)
        if match:
            time_str, long_prob, short_prob = match.groups()

            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
            if date_match:
                date_str = date_match.group(1)
                datetime_str = f"{date_str} {time_str}"

                dt_kst = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                dt_kst = kst.localize(dt_kst)

                if dt_kst >= model_deploy_time_kst:
                    dt_utc = dt_kst.astimezone(pytz.UTC)

                    prod_signals.append({
                        'timestamp_utc': dt_utc.replace(tzinfo=None),
                        'timestamp_kst': dt_kst.replace(tzinfo=None),
                        'long_prob': float(long_prob),
                        'short_prob': float(short_prob)
                    })

df_prod = pd.DataFrame(prod_signals)
print(f"  ✅ Parsed {len(df_prod)} production signals")

# Load CSV data
print("\n2. Loading CSV data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_csv = pd.read_csv(data_file)
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])

min_time = df_prod['timestamp_utc'].min()
max_time = df_prod['timestamp_utc'].max()
lookback_minutes = 1000 * 5

df_csv_filtered = df_csv[
    (df_csv['timestamp'] >= min_time - pd.Timedelta(minutes=lookback_minutes)) &
    (df_csv['timestamp'] <= max_time)
].copy()
print(f"  ✅ Loaded {len(df_csv_filtered)} candles")

# Calculate features
print("\n3. Calculating backtest features...")
df_features = calculate_all_features_enhanced_v2(df_csv_filtered, phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"  ✅ Features calculated: {len(df_features)} rows")

# Load models
print("\n4. Loading models...")
long_model = pickle.load(open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb'))
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]
print(f"  ✅ Models loaded ({len(long_feature_columns)} LONG features)")

# Calculate backtest signals
long_feat_scaled = long_scaler.transform(df_features[long_feature_columns].values)
df_features['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

# Merge
df_merged = pd.merge(
    df_prod,
    df_features[['timestamp', 'long_prob'] + long_feature_columns],
    left_on='timestamp_utc',
    right_on='timestamp',
    how='inner',
    suffixes=('_prod', '_backtest')
)

# Filter to target times
df_merged['time_str'] = df_merged['timestamp_kst'].dt.strftime('%Y-%m-%d %H:%M')
df_target = df_merged[df_merged['time_str'].isin(target_times_kst)]

print("\n" + "="*80)
print("DETAILED ANALYSIS OF LARGE DIFFERENCES")
print("="*80)

for _, row in df_target.iterrows():
    print(f"\n📍 Candle: {row['time_str']} KST")
    print(f"   LONG Production: {row['long_prob_prod']:.6f}")
    print(f"   LONG Backtest: {row['long_prob_backtest']:.6f}")
    print(f"   Difference: {row['long_prob_prod'] - row['long_prob_backtest']:+.6f}")

    # Compare features
    print(f"\n   🔍 Top 10 features with largest absolute differences:")
    feature_diffs = {}

    for feat in long_feature_columns:
        # We don't have production features, only backtest
        # Just show backtest feature values for now
        backtest_val = row[feat]
        feature_diffs[feat] = backtest_val

    # Sort by absolute value
    sorted_features = sorted(feature_diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    print(f"   {'Feature':<40} {'Backtest Value':>15}")
    print(f"   {'-'*55}")
    for feat, val in sorted_features:
        print(f"   {feat:<40} {val:>15.6f}")

print("\n" + "="*80)
print("NOTE: Production feature values not logged - cannot compare directly")
print("Recommendation: Add feature logging to production bot for debugging")
print("="*80)
