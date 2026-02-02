#!/usr/bin/env python3
"""
Test CSV Loading for Production Bot
====================================

Test the new CSV loading functions to ensure they work correctly.
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import pytz

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CSV_DATA_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
CSV_UPDATE_SCRIPT = PROJECT_ROOT / "scripts" / "utils" / "update_historical_data.py"

print("="*80)
print("CSV LOADING TEST")
print("="*80)

# Current time (KST)
kst = pytz.timezone('Asia/Seoul')
current_time = datetime.now(kst).replace(tzinfo=None)

print(f"\n1. Current Time (KST): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Test CSV loading
print("\n2. Testing CSV load...")
try:
    df = pd.read_csv(CSV_DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert UTC to KST
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(kst).dt.tz_localize(None)

    # Get latest N candles
    limit = 1000
    df_latest = df.tail(limit + 10).copy()

    latest_candle = df_latest.iloc[-1]['timestamp']
    data_age_minutes = (current_time - latest_candle).total_seconds() / 60

    print(f"  ✅ CSV loaded successfully")
    print(f"  Total candles: {len(df):,}")
    print(f"  Latest {limit} candles: {len(df_latest)}")
    print(f"  Latest candle time: {latest_candle.strftime('%Y-%m-%d %H:%M:%S')} KST")
    print(f"  Data age: {data_age_minutes:.1f} minutes")

    if data_age_minutes > 10:
        print(f"  ⚠️ CSV is stale (>{10} minutes old)")
        print(f"  Recommendation: Run update_historical_data.py")
    else:
        print(f"  ✅ CSV is fresh (<{10} minutes old)")

except Exception as e:
    print(f"  ❌ CSV load error: {e}")

# Test CSV update check
print("\n3. Testing CSV freshness check...")
try:
    if not CSV_DATA_FILE.exists():
        print("  ❌ CSV file not found")
    else:
        df = pd.read_csv(CSV_DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest = df['timestamp'].max()

        latest_kst = latest.tz_localize('UTC').tz_convert(kst).tz_localize(None)
        age_minutes = (current_time - latest_kst).total_seconds() / 60

        if age_minutes < 6:
            print(f"  ✅ CSV is fresh ({age_minutes:.1f} min old) - no update needed")
        else:
            print(f"  📅 CSV is {age_minutes:.1f} min old - update recommended")
            print(f"  Run: python scripts/utils/update_historical_data.py")

except Exception as e:
    print(f"  ❌ Error: {e}")

# Verify data quality
print("\n4. Data Quality Check...")
try:
    df_sample = df_latest.tail(10)
    print(f"  Latest 10 candles:")
    print(f"  {'Time (KST)':<20} {'Close':>12} {'Volume':>12}")
    print(f"  {'-'*45}")
    for _, row in df_sample.iterrows():
        print(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} ${row['close']:>11,.1f} {row['volume']:>11,.1f}")

    # Check for NaN
    nan_count = df_sample.isna().sum().sum()
    if nan_count > 0:
        print(f"\n  ⚠️ Found {nan_count} NaN values in sample")
    else:
        print(f"\n  ✅ No NaN values in sample")

except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
