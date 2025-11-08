#!/usr/bin/env python3
"""
Compare Production Signals vs Backtest Signals
================================================

Compare actual production bot signals from logs with
backtest signals from CSV data at the same timestamps.
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
print("PRODUCTION vs BACKTEST SIGNAL COMPARISON")
print("="*80)

# Parse production log signals
print("\n1. Parsing production log signals...")
print("  📅 Filtering for signals after 2025-10-24 11:38 (current model deployment)")
kst = pytz.timezone('Asia/Seoul')
prod_signals = []

# Filter threshold: Only use signals after current model was deployed
model_deploy_time = datetime(2025, 10, 24, 11, 38, 0)
model_deploy_time_kst = kst.localize(model_deploy_time)

with open(LOG_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        # Match signal lines
        match = re.search(r'\[Candle (\d{2}:\d{2}:\d{2}) KST\].*LONG: ([\d.]+) \| SHORT: ([\d.]+)', line)
        if match:
            time_str, long_prob, short_prob = match.groups()

            # Extract date from log line
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
            if date_match:
                date_str = date_match.group(1)
                datetime_str = f"{date_str} {time_str}"

                # Parse as KST and convert to UTC
                dt_kst = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                dt_kst = kst.localize(dt_kst)

                # Only include signals after model deployment
                if dt_kst >= model_deploy_time_kst:
                    dt_utc = dt_kst.astimezone(pytz.UTC)

                    prod_signals.append({
                        'timestamp_utc': dt_utc.replace(tzinfo=None),
                        'timestamp_kst': dt_kst.replace(tzinfo=None),
                        'long_prob': float(long_prob),
                        'short_prob': float(short_prob)
                    })

df_prod = pd.DataFrame(prod_signals)
print(f"  ✅ Parsed {len(df_prod)} signals from production log")
print(f"  Period: {df_prod['timestamp_kst'].min()} ~ {df_prod['timestamp_kst'].max()} KST")

# Load CSV data
print("\n2. Loading CSV data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_csv = pd.read_csv(data_file)
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])
print(f"  ✅ Loaded {len(df_csv):,} candles")

# Filter CSV to production signal period
# IMPORTANT: Use same lookback as production bot (1000 candles = ~3.5 days)
min_time = df_prod['timestamp_utc'].min()
max_time = df_prod['timestamp_utc'].max()

# Calculate lookback: 1000 candles × 5 minutes = 5000 minutes = ~3.5 days
lookback_minutes = 1000 * 5
df_csv_filtered = df_csv[
    (df_csv['timestamp'] >= min_time - pd.Timedelta(minutes=lookback_minutes)) &
    (df_csv['timestamp'] <= max_time)
].copy()
print(f"  ✅ Filtered to {len(df_csv_filtered)} candles (with 1000-candle lookback, same as production)")

# Calculate features
print("\n3. Calculating features on CSV data...")
df_features = calculate_all_features_enhanced_v2(df_csv_filtered, phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"  ✅ Features calculated: {len(df_features)} rows")

# Load models
print("\n4. Loading ML models...")
long_model = pickle.load(open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb'))
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb'))
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]
print(f"  ✅ Models loaded")

# Calculate backtest signals
print("\n5. Calculating backtest signals...")
long_feat = df_features[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
df_features['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

short_feat = df_features[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
df_features['short_prob'] = short_model.predict_proba(short_feat_scaled)[:, 1]
print(f"  ✅ Signals calculated")

# Merge and compare
print("\n6. Comparing signals...")
df_prod['timestamp_utc'] = pd.to_datetime(df_prod['timestamp_utc'])
df_merged = pd.merge(
    df_prod,
    df_features[['timestamp', 'long_prob', 'short_prob', 'close']],
    left_on='timestamp_utc',
    right_on='timestamp',
    how='inner',
    suffixes=('_prod', '_backtest')
)

print(f"  ✅ Matched {len(df_merged)} signals")

# Calculate differences
df_merged['long_diff'] = df_merged['long_prob_backtest'] - df_merged['long_prob_prod']
df_merged['short_diff'] = df_merged['short_prob_backtest'] - df_merged['short_prob_prod']

# Results
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print(f"\n📊 Statistics:")
print(f"  Matched signals: {len(df_merged)}")
print(f"  Production signals parsed: {len(df_prod)}")
print(f"  Match rate: {len(df_merged)/len(df_prod)*100:.1f}%")

print(f"\n📈 LONG Signal Comparison:")
print(f"  Mean difference: {df_merged['long_diff'].mean():.6f}")
print(f"  Std difference: {df_merged['long_diff'].std():.6f}")
print(f"  Max difference: {df_merged['long_diff'].abs().max():.6f}")
print(f"  Median difference: {df_merged['long_diff'].median():.6f}")

print(f"\n📉 SHORT Signal Comparison:")
print(f"  Mean difference: {df_merged['short_diff'].mean():.6f}")
print(f"  Std difference: {df_merged['short_diff'].std():.6f}")
print(f"  Max difference: {df_merged['short_diff'].abs().max():.6f}")
print(f"  Median difference: {df_merged['short_diff'].median():.6f}")

# Show sample comparisons
print(f"\n🔍 Sample Comparisons (First 10):")
print("="*100)
print(f"{'Time (KST)':<20} {'Price':>12} {'LONG Prod':>10} {'LONG BT':>10} {'Diff':>10} {'SHORT Prod':>10} {'SHORT BT':>10} {'Diff':>10}")
print("="*100)

for _, row in df_merged.head(10).iterrows():
    print(f"{row['timestamp_kst'].strftime('%Y-%m-%d %H:%M'):<20} "
          f"${row['close']:>11,.1f} "
          f"{row['long_prob_prod']:>10.4f} "
          f"{row['long_prob_backtest']:>10.4f} "
          f"{row['long_diff']:>+10.4f} "
          f"{row['short_prob_prod']:>10.4f} "
          f"{row['short_prob_backtest']:>10.4f} "
          f"{row['short_diff']:>+10.4f}")

# Check for large discrepancies
large_diff_threshold = 0.05
large_diff = df_merged[
    (df_merged['long_diff'].abs() > large_diff_threshold) |
    (df_merged['short_diff'].abs() > large_diff_threshold)
]

if len(large_diff) > 0:
    print(f"\n⚠️  Large Discrepancies (>{large_diff_threshold}):")
    print(f"  Found {len(large_diff)} candles with large differences")
    print("\n  Top 5 by LONG difference:")
    for _, row in large_diff.nlargest(5, 'long_diff', keep='all').iterrows():
        print(f"    {row['timestamp_kst'].strftime('%Y-%m-%d %H:%M')}: "
              f"Prod={row['long_prob_prod']:.4f}, BT={row['long_prob_backtest']:.4f}, "
              f"Diff={row['long_diff']:+.4f}")
else:
    print(f"\n✅ No large discrepancies found (threshold: {large_diff_threshold})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
