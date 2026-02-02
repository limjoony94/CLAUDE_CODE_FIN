#!/usr/bin/env python3
"""
Verify UTC-based signals from new bot (2025-10-27)
Filter incomplete candles based on CSV update time
"""

import pandas as pd
import pickle
import joblib
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# Test signals from new bot (UTC-based)
# IMPORTANT: Production bot logs show "2025-10-27 00:05 KST" = "2025-10-26 15:05 UTC"
# KST is 9 hours ahead of UTC, so dates can differ!
test_signals = [
    {
        'timestamp_utc': '2025-10-26 15:05:00',  # 2025-10-27 00:05:00 KST
        'timestamp_kst': '2025-10-27 00:05:00',
        'long_prod': 0.2126,
        'short_prod': 0.2276
    },
    {
        'timestamp_utc': '2025-10-26 15:10:00',  # 2025-10-27 00:10:00 KST
        'timestamp_kst': '2025-10-27 00:10:00',
        'long_prod': 0.1993,
        'short_prod': 0.2260
    },
    {
        'timestamp_utc': '2025-10-26 15:15:00',  # 2025-10-27 00:15:00 KST
        'timestamp_kst': '2025-10-27 00:15:00',
        'long_prod': 0.1721,
        'short_prod': 0.2442
    },
    {
        'timestamp_utc': '2025-10-26 15:20:00',  # 2025-10-27 00:20:00 KST
        'timestamp_kst': '2025-10-27 00:20:00',
        'long_prod': 0.1562,
        'short_prod': 0.2285
    },
    {
        'timestamp_utc': '2025-10-26 15:25:00',  # 2025-10-27 00:25:00 KST
        'timestamp_kst': '2025-10-27 00:25:00',
        'long_prod': 0.1911,
        'short_prod': 0.2940
    }
]

print("=" * 80)
print("UTC-BASED SIGNAL VERIFICATION (New Bot)")
print("=" * 80)

# Load CSV
CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
print(f"\n1. Loading CSV...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get CSV file update time to filter out incomplete candles
# os.path.getmtime() returns Unix timestamp, pd.to_datetime() interprets as UTC
csv_update_time_utc = pd.to_datetime(os.path.getmtime(CSV_FILE), unit='s')
# Calculate KST equivalent (UTC+9) for display
csv_update_time_kst = csv_update_time_utc + pd.Timedelta(hours=9)
print(f"   CSV updated at: {csv_update_time_kst.strftime('%Y-%m-%d %H:%M:%S')} KST")
print(f"                   {csv_update_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")

# Filter data
# IMPORTANT: Exclude incomplete candles BEFORE feature calculation
# because they cascade through MA/RSI/MACD calculations
test_times = [pd.to_datetime(s['timestamp_utc']) for s in test_signals]
min_time = min(test_times)

# Use CSV update time as max_time to exclude incomplete candles
max_time = csv_update_time_utc

print(f"\n2. Filtering data (exclude incomplete candles)...")
print(f"   Max time: {max_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"   Reason: Only use candles that were complete when CSV was updated")

# Add lookback
from datetime import timedelta
lookback_minutes = 1000 * 5

# CRITICAL: Filter by candle END time, not start time
# A 5-minute candle starting at 15:25 ends at 15:30
# If CSV was updated at 15:28, the 15:25 candle was still forming
df_filtered = df[
    (df['timestamp'] >= min_time - timedelta(minutes=lookback_minutes)) &
    (df['timestamp'] + timedelta(minutes=5) <= max_time)  # Candle END time must be <= CSV update time
].copy()

print(f"   ✅ Loaded {len(df_filtered)} candles")
print(f"   Period: {df_filtered['timestamp'].min()} ~ {df_filtered['timestamp'].max()} UTC")

# Calculate features (UTC-based, same as backtest)
print(f"\n3. Calculating features (UTC-based)...")
df_features = calculate_all_features_enhanced_v2(df_filtered, phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"   ✅ Features calculated: {len(df_features)} rows")

# Load models
print(f"\n4. Loading models...")
MODELS_DIR = PROJECT_ROOT / "models"
long_model = pickle.load(open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb'))
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines()]

short_model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb'))
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_features = [line.strip() for line in f.readlines()]

print(f"   ✅ Models loaded")

# Generate signals
print(f"\n5. Generating backtest signals (UTC-based)...")
long_X = df_features[long_features].values
long_X_scaled = long_scaler.transform(long_X)
df_features['long_pred'] = long_model.predict_proba(long_X_scaled)[:, 1]

short_X = df_features[short_features].values
short_X_scaled = short_scaler.transform(short_X)
df_features['short_pred'] = short_model.predict_proba(short_X_scaled)[:, 1]

print(f"   ✅ Signals generated")

# Filter test signals to only include complete candles
print(f"\n6. Filtering test signals (exclude incomplete candles)...")
complete_signals = []
skipped_signals = []

for signal in test_signals:
    ts_utc = pd.to_datetime(signal['timestamp_utc'])
    # Candle end time is start time + 5 minutes
    candle_end_time = ts_utc + pd.Timedelta(minutes=5)

    if candle_end_time > csv_update_time_utc:
        # Candle was still forming when CSV was updated
        skipped_signals.append(signal)
        print(f"   ⚠️ Skipping {signal['timestamp_kst']} KST ({signal['timestamp_utc']} UTC)")
        print(f"      Reason: Candle incomplete at CSV update time")
        print(f"      Candle ends: {candle_end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"      CSV updated: {csv_update_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    else:
        # Candle was complete
        complete_signals.append(signal)

print(f"\n   ✅ Testing {len(complete_signals)} complete candles")
print(f"   ⚠️ Skipped {len(skipped_signals)} incomplete candles")

# Compare
print(f"\n7. Comparing signals...")
print("=" * 80)
print(f"{'Time (UTC)':<20} {'Time (KST)':<20} {'Signal':<6} {'Prod':>8} {'Backtest':>8} {'Diff':>10} {'Match':<6}")
print("=" * 80)

results = []
for signal in complete_signals:
    ts_utc = pd.to_datetime(signal['timestamp_utc'])
    ts_kst = signal['timestamp_kst']

    # Find matching row
    row = df_features[df_features['timestamp'] == ts_utc]

    if len(row) == 0:
        print(f"{ts_utc} - NOT FOUND IN BACKTEST")
        continue

    row = row.iloc[0]
    long_bt = row['long_pred']
    short_bt = row['short_pred']

    long_diff = signal['long_prod'] - long_bt
    short_diff = signal['short_prod'] - short_bt

    long_match = "✅" if abs(long_diff) < 0.0001 else "❌"
    short_match = "✅" if abs(short_diff) < 0.0001 else "❌"

    print(f"{ts_utc}   {ts_kst}   LONG   {signal['long_prod']:>8.4f} {long_bt:>8.4f} {long_diff:>+10.6f} {long_match:<6}")
    print(f"{' ' * 20}   {' ' * 20}   SHORT  {signal['short_prod']:>8.4f} {short_bt:>8.4f} {short_diff:>+10.6f} {short_match:<6}")

    results.append({
        'time': ts_utc,
        'long_diff': abs(long_diff),
        'short_diff': abs(short_diff),
        'long_match': abs(long_diff) < 0.0001,
        'short_match': abs(short_diff) < 0.0001
    })

# Summary
print("=" * 80)
print("\n📊 VERIFICATION SUMMARY")
print("=" * 80)

print(f"\nTotal signals from production: {len(test_signals)}")
print(f"Complete candles (tested): {len(complete_signals)}")
print(f"Incomplete candles (skipped): {len(skipped_signals)}")

if len(skipped_signals) > 0:
    print(f"\n⚠️ Skipped signals:")
    for s in skipped_signals:
        print(f"   {s['timestamp_kst']} KST - Incomplete at CSV update time")

if len(results) == 0:
    print("\n⚠️ No complete candles to verify")
    print("All test signals were incomplete at CSV update time")
else:
    total = len(results)
    long_matches = sum(r['long_match'] for r in results)
    short_matches = sum(r['short_match'] for r in results)

    print(f"\n📊 Verification Results:")
    print(f"LONG matches: {long_matches}/{total} ({long_matches/total*100:.1f}%)")
    print(f"SHORT matches: {short_matches}/{total} ({short_matches/total*100:.1f}%)")

    if long_matches == total and short_matches == total:
        print("\n✅ ✅ ✅ ALL COMPLETE SIGNALS MATCH PERFECTLY ✅ ✅ ✅")
        print("\nUTC timezone fix successful!")
        print("Production signals 100% identical to backtest for complete candles.")
        print("\n🔍 Note: Incomplete candles excluded from verification")
        print("This is expected and correct - CSV data captured mid-formation")
    else:
        print("\n⚠️ Some signals don't match")
        avg_long_diff = sum(r['long_diff'] for r in results) / total
        avg_short_diff = sum(r['short_diff'] for r in results) / total
        print(f"Average LONG difference: {avg_long_diff:.6f}")
        print(f"Average SHORT difference: {avg_short_diff:.6f}")

print("\n" + "=" * 80)
