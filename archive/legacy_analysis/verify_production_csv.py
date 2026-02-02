#!/usr/bin/env python3
"""
Verify Production Bot CSV-Based Signal Generation
==================================================

Simulate production bot's CSV-based signal generation and compare
with backtest signals to verify 100% alignment.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import datetime, timedelta
import pytz
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_FILE = PROJECT_ROOT / "logs" / "opportunity_gating_bot_4x_20251017.log"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_max.csv"

print("="*80)
print("PRODUCTION CSV-BASED SIGNAL VERIFICATION")
print("="*80)

# 1. Load production signals from log (current model only)
print("\n1. Parsing production signals (model deployed 2025-10-24 11:38)...")
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
print(f"  Period: {df_prod['timestamp_kst'].min()} ~ {df_prod['timestamp_kst'].max()} KST")

# 2. Load CSV data (SAME AS PRODUCTION)
print("\n2. Loading CSV data (production method)...")
df_csv = pd.read_csv(CSV_FILE)
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])

# Convert UTC to KST (SAME AS PRODUCTION)
df_csv['timestamp'] = df_csv['timestamp'].dt.tz_localize('UTC').dt.tz_convert(kst).dt.tz_localize(None)

# Filter to production period + 1000-candle lookback
min_time = df_prod['timestamp_kst'].min()
max_time = df_prod['timestamp_kst'].max()
lookback_minutes = 1000 * 5

df_csv_filtered = df_csv[
    (df_csv['timestamp'] >= min_time - timedelta(minutes=lookback_minutes)) &
    (df_csv['timestamp'] <= max_time)
].copy()

print(f"  ✅ Loaded {len(df_csv)} total candles from CSV")
print(f"  ✅ Filtered to {len(df_csv_filtered)} candles (1000-candle lookback)")
print(f"  Period: {df_csv_filtered['timestamp'].min()} ~ {df_csv_filtered['timestamp'].max()} KST")

# Convert back to UTC for feature calculation (models expect UTC)
df_csv_filtered['timestamp'] = df_csv_filtered['timestamp'].dt.tz_localize(kst).dt.tz_convert('UTC').dt.tz_localize(None)

# 3. Calculate features (SAME AS PRODUCTION)
print("\n3. Calculating features (production pipeline)...")
df_features = calculate_all_features_enhanced_v2(df_csv_filtered, phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"  ✅ Features calculated: {len(df_features)} rows")

# Convert timestamp back to KST for comparison
df_features['timestamp'] = pd.to_datetime(df_features['timestamp']).dt.tz_localize('UTC').dt.tz_convert(kst).dt.tz_localize(None)

# 4. Load models (SAME AS PRODUCTION)
print("\n4. Loading models...")
long_model = pickle.load(open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb'))
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb'))
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ Models loaded (LONG: {len(long_feature_columns)} features, SHORT: {len(short_feature_columns)} features)")

# 5. Generate signals (SAME AS PRODUCTION)
print("\n5. Generating signals (CSV-based production simulation)...")

# LONG signals
long_feat = df_features[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
df_features['long_prob_csv'] = long_model.predict_proba(long_feat_scaled)[:, 1]

# SHORT signals
short_feat = df_features[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
df_features['short_prob_csv'] = short_model.predict_proba(short_feat_scaled)[:, 1]

print(f"  ✅ Signals generated for {len(df_features)} candles")

# 6. Compare with production signals
print("\n6. Comparing CSV-based signals with production signals...")

# Merge
df_merged = pd.merge(
    df_prod,
    df_features[['timestamp', 'long_prob_csv', 'short_prob_csv', 'close']],
    left_on='timestamp_kst',
    right_on='timestamp',
    how='inner',
    suffixes=('_prod', '_csv')
)

print(f"  ✅ Matched {len(df_merged)}/{len(df_prod)} signals")

# Calculate differences
df_merged['long_diff'] = df_merged['long_prob_csv'] - df_merged['long_prob']
df_merged['short_diff'] = df_merged['short_prob_csv'] - df_merged['short_prob']

# 7. Results
print("\n" + "="*80)
print("VERIFICATION RESULTS")
print("="*80)

print(f"\n📊 Match Statistics:")
print(f"  Total production signals: {len(df_prod)}")
print(f"  Matched signals: {len(df_merged)}")
print(f"  Match rate: {len(df_merged)/len(df_prod)*100:.1f}%")

print(f"\n📈 LONG Signal Comparison (CSV vs Production):")
print(f"  Mean difference: {df_merged['long_diff'].mean():.6f} ({abs(df_merged['long_diff'].mean())/df_merged['long_prob'].mean()*100:.2f}%)")
print(f"  Std difference: {df_merged['long_diff'].std():.6f}")
print(f"  Max absolute diff: {df_merged['long_diff'].abs().max():.6f}")
print(f"  Median difference: {df_merged['long_diff'].median():.6f}")

print(f"\n📉 SHORT Signal Comparison (CSV vs Production):")
print(f"  Mean difference: {df_merged['short_diff'].mean():.6f} ({abs(df_merged['short_diff'].mean())/df_merged['short_prob'].mean()*100:.2f}%)")
print(f"  Std difference: {df_merged['short_diff'].std():.6f}")
print(f"  Max absolute diff: {df_merged['short_diff'].abs().max():.6f}")
print(f"  Median difference: {df_merged['short_diff'].median():.6f}")

# Precision check (within numerical precision)
threshold = 0.0001  # 0.01% threshold for floating point precision
long_match = (df_merged['long_diff'].abs() < threshold).sum()
short_match = (df_merged['short_diff'].abs() < threshold).sum()

print(f"\n✅ Precision Match Check (within {threshold}):")
print(f"  LONG: {long_match}/{len(df_merged)} ({long_match/len(df_merged)*100:.1f}%)")
print(f"  SHORT: {short_match}/{len(df_merged)} ({short_match/len(df_merged)*100:.1f}%)")

# Sample comparison
print(f"\n🔍 Sample Comparisons (Last 10):")
print("="*120)
print(f"{'Time (KST)':<20} {'LONG Prod':>10} {'LONG CSV':>10} {'L Diff':>10} {'SHORT Prod':>10} {'SHORT CSV':>10} {'S Diff':>10}")
print("="*120)

for _, row in df_merged.tail(10).iterrows():
    print(f"{row['timestamp_kst'].strftime('%Y-%m-%d %H:%M'):<20} "
          f"{row['long_prob']:>10.4f} "
          f"{row['long_prob_csv']:>10.4f} "
          f"{row['long_diff']:>+10.6f} "
          f"{row['short_prob']:>10.4f} "
          f"{row['short_prob_csv']:>10.4f} "
          f"{row['short_diff']:>+10.6f}")

# Check for large discrepancies
large_diff_threshold = 0.001  # 0.1%
large_diff = df_merged[
    (df_merged['long_diff'].abs() > large_diff_threshold) |
    (df_merged['short_diff'].abs() > large_diff_threshold)
]

if len(large_diff) > 0:
    print(f"\n⚠️  Large Discrepancies (>{large_diff_threshold}):")
    print(f"  Found {len(large_diff)} signals with differences > {large_diff_threshold}")
    print("\n  Top 5 by total difference:")
    large_diff['total_diff'] = large_diff['long_diff'].abs() + large_diff['short_diff'].abs()
    for _, row in large_diff.nlargest(5, 'total_diff').iterrows():
        print(f"    {row['timestamp_kst'].strftime('%Y-%m-%d %H:%M')}: "
              f"LONG diff={row['long_diff']:+.6f}, SHORT diff={row['short_diff']:+.6f}")
else:
    print(f"\n✅ No large discrepancies found (all within {large_diff_threshold})")

# Final verdict
print("\n" + "="*80)
print("VERDICT")
print("="*80)

if long_match/len(df_merged) > 0.99 and short_match/len(df_merged) > 0.99:
    print("\n✅ ✅ ✅ PRODUCTION CSV-BASED SIGNALS VERIFIED ✅ ✅ ✅")
    print("\nCSV-based production signals match expected backtest signals!")
    print(f"  - LONG: {long_match/len(df_merged)*100:.1f}% exact match")
    print(f"  - SHORT: {short_match/len(df_merged)*100:.1f}% exact match")
    print("\n🎯 Production is ready to generate identical signals to backtest.")
else:
    print("\n⚠️ VERIFICATION FAILED")
    print(f"\nSignal differences exceed acceptable threshold:")
    print(f"  - LONG match: {long_match/len(df_merged)*100:.1f}% (target: >99%)")
    print(f"  - SHORT match: {short_match/len(df_merged)*100:.1f}% (target: >99%)")
    print("\n🔍 Investigation needed - check data processing pipeline.")

print("\n" + "="*80)
