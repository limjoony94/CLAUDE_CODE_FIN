#!/usr/bin/env python3
"""
Timezone Hypothesis Verification
타임존 해석 차이가 데이터 불일치의 원인인지 검증
"""

import pandas as pd
import ccxt
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("TIMEZONE HYPOTHESIS VERIFICATION")
print("=" * 80)

# Target: 15:25 UTC candle
TARGET_UTC = '2025-10-26 15:25:00'
target_dt_utc = datetime(2025, 10, 26, 15, 25, 0, tzinfo=pytz.UTC)
target_ts_ms = int(target_dt_utc.timestamp() * 1000)

print(f"\n🎯 Target Candle:")
print(f"   UTC String: {TARGET_UTC}")
print(f"   UTC Datetime: {target_dt_utc}")
print(f"   Timestamp (ms): {target_ts_ms}")
print(f"   Timestamp (s): {target_ts_ms / 1000}")

# Test 1: How CSV was created (simulate collect_max_data.py)
print(f"\n\n1️⃣ CSV Creation Process (collect_max_data.py simulation)")
print("=" * 80)

exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})
ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', limit=10)

print(f"\nAPI returns raw timestamp (ms):")
for i, candle in enumerate(ohlcv[-3:]):
    ts_ms = candle[0]
    dt_utc = datetime.fromtimestamp(ts_ms/1000, pytz.UTC)
    print(f"  Candle {i}: {ts_ms} → {dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Simulate DataFrame conversion (Line 110-111 in collect_max_data.py)
df_test = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

print(f"\nBefore pd.to_datetime(unit='ms'):")
print(f"  Dtype: {df_test['timestamp'].dtype}")
print(f"  Sample values (raw ms):")
for i in range(min(3, len(df_test))):
    print(f"    {df_test.iloc[i]['timestamp']}")

df_test['timestamp'] = pd.to_datetime(df_test['timestamp'], unit='ms')

print(f"\nAfter pd.to_datetime(unit='ms'):")
print(f"  Dtype: {df_test['timestamp'].dtype}")
print(f"  Timezone aware? {df_test['timestamp'].dt.tz is not None}")
if df_test['timestamp'].dt.tz is not None:
    print(f"  Timezone: {df_test['timestamp'].dt.tz}")
print(f"  Sample values:")
for i in range(min(3, len(df_test))):
    print(f"    {df_test.iloc[i]['timestamp']}")

# What gets saved to CSV?
print(f"\nWhat string representation is saved to CSV?")
csv_string = df_test.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
print(f"  Format: {csv_string}")

# Test 2: How CSV is read (simulate verification script)
print(f"\n\n2️⃣ CSV Reading Process (verification script simulation)")
print("=" * 80)

CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(CSV_FILE)

print(f"\nAfter pd.read_csv():")
print(f"  Dtype: {df['timestamp'].dtype}")
print(f"  First 3 timestamps (raw strings):")
for i in range(3):
    print(f"    {df.iloc[i]['timestamp']}")

df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\nAfter pd.to_datetime() (no unit parameter):")
print(f"  Dtype: {df['timestamp'].dtype}")
print(f"  Timezone aware? {df['timestamp'].dt.tz is not None}")
if df['timestamp'].dt.tz is not None:
    print(f"  Timezone: {df['timestamp'].dt.tz}")
print(f"  First 3 timestamps:")
for i in range(3):
    print(f"    {df.iloc[i]['timestamp']}")

# Test 3: Timestamp matching logic
print(f"\n\n3️⃣ Timestamp Matching Test")
print("=" * 80)

# Method A: String comparison (what verification script might do)
print(f"\nMethod A: String comparison")
target_str = TARGET_UTC
csv_row_str = df[df['timestamp'] == target_str]
print(f"  Search: df[df['timestamp'] == '{target_str}']")
print(f"  Found: {len(csv_row_str)} rows")
if len(csv_row_str) > 0:
    print(f"  Close: {csv_row_str.iloc[0]['close']:.1f}")

# Method B: Datetime comparison
print(f"\nMethod B: Datetime comparison (naive)")
target_dt_naive = pd.to_datetime(TARGET_UTC)
csv_row_dt = df[df['timestamp'] == target_dt_naive]
print(f"  Search: df[df['timestamp'] == pd.to_datetime('{TARGET_UTC}')]")
print(f"  Found: {len(csv_row_dt)} rows")
if len(csv_row_dt) > 0:
    print(f"  Close: {csv_row_dt.iloc[0]['close']:.1f}")

# Method C: Timestamp (ms) comparison
print(f"\nMethod C: Convert CSV to timestamp and compare")
df_with_ts = df.copy()
df_with_ts['timestamp_ms'] = df_with_ts['timestamp'].astype('int64') // 10**6
csv_row_ts = df_with_ts[df_with_ts['timestamp_ms'] == target_ts_ms]
print(f"  Search: df[df['timestamp_ms'] == {target_ts_ms}]")
print(f"  Found: {len(csv_row_ts)} rows")
if len(csv_row_ts) > 0:
    print(f"  Close: {csv_row_ts.iloc[0]['close']:.1f}")

# Test 4: KST offset hypothesis
print(f"\n\n4️⃣ KST Offset Hypothesis Test")
print("=" * 80)

# If CSV interpreted timestamps as KST instead of UTC, they'd be 9 hours off
target_kst_as_utc = datetime(2025, 10, 27, 0, 25, 0)  # 00:25 KST interpreted as UTC
target_kst_offset_str = target_kst_as_utc.strftime('%Y-%m-%d %H:%M:%S')

print(f"\nIf '2025-10-27 00:25:00' (KST) was saved as '2025-10-27 00:25:00' (string):")
print(f"  And we search for '2025-10-26 15:25:00' (UTC):")
print(f"  We would be looking for WRONG candle (9 hour difference)")

csv_row_kst = df[df['timestamp'] == target_kst_offset_str]
print(f"\nSearch for KST-as-UTC: '{target_kst_offset_str}'")
print(f"  Found: {len(csv_row_kst)} rows")
if len(csv_row_kst) > 0:
    print(f"  Close: {csv_row_kst.iloc[0]['close']:.1f}")

# Test 5: UTC interpretation verification
print(f"\n\n5️⃣ UTC Interpretation Verification")
print("=" * 80)

# Create sample timestamp like collect_max_data.py does
sample_ts_ms = 1730000000000  # Arbitrary timestamp
sample_dt = pd.to_datetime(sample_ts_ms, unit='ms')

print(f"\nSample conversion (pd.to_datetime(unit='ms')):")
print(f"  Input (ms): {sample_ts_ms}")
print(f"  Output: {sample_dt}")
print(f"  Timezone: {sample_dt.tz}")

# What UTC time does this represent?
sample_utc = datetime.fromtimestamp(sample_ts_ms/1000, pytz.UTC)
print(f"  Verified UTC: {sample_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# If interpreted as KST
sample_kst = datetime.fromtimestamp(sample_ts_ms/1000, pytz.timezone('Asia/Seoul'))
print(f"  If KST: {sample_kst.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Test 6: Actual data comparison
print(f"\n\n6️⃣ Actual Data Comparison")
print("=" * 80)

# Get 15:25 UTC candle from CSV
csv_row = df[df['timestamp'] == TARGET_UTC]
if len(csv_row) > 0:
    csv_close = csv_row.iloc[0]['close']
    csv_volume = csv_row.iloc[0]['volume']
    print(f"\nCSV Data for {TARGET_UTC}:")
    print(f"  Close: ${csv_close:.1f}")
    print(f"  Volume: {csv_volume:.4f}")
else:
    print(f"\n❌ CSV: '{TARGET_UTC}' NOT FOUND")

# Get current API data for same timestamp
api_candle = None
for candle in ohlcv:
    if candle[0] == target_ts_ms:
        api_candle = candle
        break

if api_candle:
    print(f"\nCurrent API Data for timestamp {target_ts_ms}:")
    print(f"  Close: ${api_candle[4]:.1f}")
    print(f"  Volume: {api_candle[5]:.4f}")
else:
    # Fetch with 'since' to get older data
    print(f"\nFetching historical data with 'since={target_ts_ms}'...")
    since_ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', since=target_ts_ms, limit=10)

    for candle in since_ohlcv:
        if candle[0] == target_ts_ms:
            api_candle = candle
            break

    if api_candle:
        print(f"\nAPI Data for timestamp {target_ts_ms} (from 'since' query):")
        print(f"  Close: ${api_candle[4]:.1f}")
        print(f"  Volume: {api_candle[5]:.4f}")
    else:
        print(f"\n❌ API: Timestamp {target_ts_ms} NOT FOUND")

print("\n" + "=" * 80)
print("🔍 DIAGNOSIS")
print("=" * 80)

print("""
Key Questions to Answer:
1. Does pd.to_datetime(unit='ms') create UTC or naive datetime?
2. Does pd.read_csv() interpret '2025-10-26 15:25:00' as UTC or local?
3. Are CSV and API timestamps aligned (same reference)?
4. Is there a 9-hour offset (KST vs UTC issue)?

If timezone is the issue:
  - CSV might store "2025-10-27 00:25:00" (KST) as "2025-10-27 00:25:00" (naive)
  - Verification searches for "2025-10-26 15:25:00" (UTC)
  - Gets wrong candle or no match

Fix Strategy:
  - Ensure collect_max_data.py saves with explicit UTC: df['timestamp'].dt.tz_localize('UTC')
  - Ensure verification explicitly interprets as UTC: pd.to_datetime(df['timestamp'], utc=True)
  - OR: Always work with timestamps (ms) instead of datetime strings
""")

print("=" * 80)
