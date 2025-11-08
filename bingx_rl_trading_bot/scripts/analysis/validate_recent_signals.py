"""
Recent Signal Validation - Post Bug Fix
========================================

Validate signals from 2025-10-25 (after bug fix on 2025-10-24 11:38)
to verify backtest matches production methodology.

Test Signals:
1. 2025-10-25 00:10 KST - LONG: 0.8397 (above threshold 0.80)
2. 2025-10-25 00:15 KST - LONG: 0.8320 (above threshold 0.80)
3. 2025-10-25 00:25 KST - LONG: 0.8041 (above threshold 0.80)
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("RECENT SIGNAL VALIDATION (POST BUG FIX)")
print("="*80)
print()

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

print(f"✅ LONG model loaded ({len(long_feature_columns)} features)")
print(f"✅ SHORT model loaded ({len(short_feature_columns)} features)")
print()

# Load historical data
df_path = DATA_DIR / "BTCUSDT_5m_updated.csv"
df_all = pd.read_csv(df_path)
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)

print(f"Historical data: {len(df_all)} candles")
print(f"Range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
print()

# Test signals (KST times, but we use COMPLETED candle before that time)
test_signals = [
    {
        'display_time_kst': '2025-10-25 00:10:00',  # Display time in logs
        'candle_time_kst': '2025-10-25 00:05:00',   # Actual candle used (completed)
        'live_long': 0.7457,
        'live_short': 0.6686,
        'live_price': 110108.3
    },
    {
        'display_time_kst': '2025-10-25 00:15:00',
        'candle_time_kst': '2025-10-25 00:10:00',
        'live_long': 0.8397,
        'live_short': 0.5971,
        'live_price': 110389.0
    },
    {
        'display_time_kst': '2025-10-25 00:20:00',
        'candle_time_kst': '2025-10-25 00:15:00',
        'live_long': 0.8320,
        'live_short': 0.6197,
        'live_price': 110428.5
    },
    {
        'display_time_kst': '2025-10-25 00:30:00',
        'candle_time_kst': '2025-10-25 00:25:00',
        'live_long': 0.8041,
        'live_short': 0.5254,
        'live_price': 110519.7
    }
]

print("="*80)
print("SIGNAL VALIDATION")
print("="*80)
print()

results = []

for sig in test_signals:
    display_time_kst = pd.Timestamp(sig['display_time_kst'])
    candle_time_kst = pd.Timestamp(sig['candle_time_kst'])
    candle_time_utc = candle_time_kst - timedelta(hours=9)

    print(f"Test Signal: {display_time_kst} KST (uses {candle_time_kst} KST candle)")
    print(f"  UTC: {candle_time_utc}")
    print(f"  Live: LONG={sig['live_long']:.4f}, SHORT={sig['live_short']:.4f}, Price=${sig['live_price']:,.1f}")
    print()

    # Load data with sufficient lookback
    lookback_start = candle_time_utc - timedelta(days=3)  # 3 days lookback
    df_test = df_all[(df_all['timestamp'] >= lookback_start) &
                     (df_all['timestamp'] <= candle_time_utc)].copy()

    if len(df_test) == 0:
        print(f"  ❌ No data available for this period")
        print()
        continue

    print(f"  Loaded {len(df_test)} candles for feature calculation")

    # Calculate features
    df_features = calculate_all_features_enhanced_v2(df_test, phase='phase1')

    # Find target candle
    target_row = df_features[df_features['timestamp'] == candle_time_utc]

    if len(target_row) == 0:
        print(f"  ❌ Target candle NOT FOUND after feature calculation")
        print(f"     Feature range: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
        print()
        continue

    row = target_row.iloc[0]

    # Generate LONG signal
    try:
        long_feat = row[long_feature_columns].values.reshape(1, -1)
        long_feat_scaled = long_scaler.transform(long_feat)
        long_prob = long_model.predict_proba(long_feat_scaled)[0, 1]
    except Exception as e:
        print(f"  ❌ LONG prediction error: {e}")
        long_prob = None

    # Generate SHORT signal
    try:
        short_feat = row[short_feature_columns].values.reshape(1, -1)
        short_feat_scaled = short_scaler.transform(short_feat)
        short_prob = short_model.predict_proba(short_feat_scaled)[0, 1]
    except Exception as e:
        print(f"  ❌ SHORT prediction error: {e}")
        short_prob = None

    if long_prob is not None and short_prob is not None:
        print(f"  Backtest: LONG={long_prob:.4f}, SHORT={short_prob:.4f}, Price=${row['close']:,.1f}")
        print()

        # Calculate errors
        long_error = abs(long_prob - sig['live_long'])
        short_error = abs(short_prob - sig['live_short'])
        price_error = abs(row['close'] - sig['live_price'])

        print(f"  Difference:")
        print(f"    LONG:  {long_error:.4f} ({long_error/sig['live_long']*100:+.1f}%)")
        print(f"    SHORT: {short_error:.4f} ({short_error/sig['live_short']*100:+.1f}%)")
        print(f"    Price: ${price_error:,.2f}")
        print()

        # Tolerance check (1% error acceptable)
        long_match = long_error < 0.01
        short_match = short_error < 0.01

        if long_match and short_match:
            print(f"  ✅ SIGNALS MATCH (within 1% tolerance)")
            status = "MATCH"
        else:
            print(f"  ⚠️  SIGNALS DIFFER")
            status = "DIFFER"

        results.append({
            'time_kst': sig['display_time_kst'],
            'candle_kst': sig['candle_time_kst'],
            'status': status,
            'long_error': long_error,
            'short_error': short_error,
            'price_error': price_error
        })

    print()
    print("-" * 80)
    print()

# Summary
print("="*80)
print("VALIDATION SUMMARY")
print("="*80)
print()

if len(results) > 0:
    matches = sum(1 for r in results if r['status'] == 'MATCH')
    total = len(results)

    print(f"Total signals tested: {total}")
    print(f"Matches: {matches}/{total} ({matches/total*100:.1f}%)")
    print()

    if matches == total:
        print("✅ ALL SIGNALS MATCH!")
        print()
        print("Conclusion:")
        print("  - Backtest methodology matches production perfectly")
        print("  - No data loading issues")
        print("  - Signal consistency verified")
    elif matches > 0:
        print(f"⚠️  PARTIAL MATCH ({matches}/{total})")
        print()
        print("Signals that differ:")
        for r in results:
            if r['status'] == 'DIFFER':
                print(f"  {r['time_kst']} - LONG error: {r['long_error']:.4f}, SHORT error: {r['short_error']:.4f}")
    else:
        print("❌ NO MATCHES FOUND")
        print()
        print("Possible issues:")
        print("  - Different data sources")
        print("  - Feature calculation differences")
        print("  - Timing/lookback issues")

    # Average errors
    avg_long_error = np.mean([r['long_error'] for r in results])
    avg_short_error = np.mean([r['short_error'] for r in results])
    avg_price_error = np.mean([r['price_error'] for r in results])

    print()
    print(f"Average errors:")
    print(f"  LONG:  {avg_long_error:.4f}")
    print(f"  SHORT: {avg_short_error:.4f}")
    print(f"  Price: ${avg_price_error:,.2f}")
else:
    print("❌ No results to summarize")

print()
print("Validation complete!")
