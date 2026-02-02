#!/usr/bin/env python3
"""
Find Matching Timestamps for Mismatched Data
데이터가 다를 때, 각 데이터가 상대방의 어느 타임스탬프에 존재하는지 확인
"""

import pandas as pd
import ccxt
from datetime import datetime
import pytz
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

print("=" * 80)
print("MATCHING TIMESTAMP FINDER")
print("데이터 불일치 시 실제 매칭되는 타임스탬프 찾기")
print("=" * 80)

# 1. Load CSV
CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df_csv = pd.read_csv(CSV_FILE)
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])

print(f"\n✅ CSV 로드: {len(df_csv)} 캔들")

# 2. Get API data
exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})
symbol = 'BTC/USDT:USDT'
timeframe = '5m'

# Get recent data (last 200 candles to have enough range)
ohlcv_recent = exchange.fetch_ohlcv(symbol, timeframe, limit=200)

# Also get data around target timestamp
target_timestamp_ms = int(datetime(2025, 10, 26, 15, 25, 0, tzinfo=pytz.UTC).timestamp() * 1000)
ohlcv_target = exchange.fetch_ohlcv(symbol, timeframe, since=target_timestamp_ms - (100*5*60*1000), limit=200)

# Combine and deduplicate
all_ohlcv = {}
for candle in ohlcv_recent + ohlcv_target:
    all_ohlcv[candle[0]] = candle

ohlcv_list = list(all_ohlcv.values())
ohlcv_list.sort(key=lambda x: x[0])

df_api = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_api['timestamp'] = pd.to_datetime(df_api['timestamp'], unit='ms')

print(f"✅ API 데이터: {len(df_api)} 캔들")

# Target data
print(f"\n" + "=" * 80)
print("🎯 TARGET: 2025-10-26 15:25:00 UTC")
print("=" * 80)

target_time = pd.to_datetime('2025-10-26 15:25:00')

csv_15_25 = df_csv[df_csv['timestamp'] == target_time].iloc[0]
api_15_25 = df_api[df_api['timestamp'] == target_time].iloc[0]

print(f"\nCSV 데이터 (15:25):")
print(f"  Close: ${csv_15_25['close']:.1f}")
print(f"  Volume: {csv_15_25['volume']:.4f}")
print(f"  High: ${csv_15_25['high']:.1f}")

print(f"\nAPI 데이터 (15:25):")
print(f"  Close: ${api_15_25['close']:.1f}")
print(f"  Volume: {api_15_25['volume']:.4f}")
print(f"  High: ${api_15_25['high']:.1f}")

# Question 1: CSV의 15:25 데이터가 현재 API의 어느 타임스탬프에 있는가?
print(f"\n" + "=" * 80)
print("❓ Question 1: CSV의 15:25 데이터가 현재 API의 어느 타임스탬프에 있는가?")
print("=" * 80)

csv_close = csv_15_25['close']
csv_volume = csv_15_25['volume']
csv_high = csv_15_25['high']

tolerance_price = 0.5
tolerance_volume = 0.1

# Close 매칭
print(f"\n🔍 CSV Close (${csv_close:.1f}) 매칭:")
api_close_matches = df_api[abs(df_api['close'] - csv_close) < tolerance_price]
if len(api_close_matches) > 0:
    print(f"   발견: {len(api_close_matches)} 개")
    for i, row in api_close_matches.head(5).iterrows():
        diff = abs(row['close'] - csv_close)
        print(f"   - {row['timestamp']}: ${row['close']:.1f} (차이: ${diff:.2f})")
else:
    print(f"   ❌ 매칭 없음")

# Volume 매칭
print(f"\n🔍 CSV Volume ({csv_volume:.4f}) 매칭:")
api_volume_matches = df_api[abs(df_api['volume'] - csv_volume) < tolerance_volume]
if len(api_volume_matches) > 0:
    print(f"   발견: {len(api_volume_matches)} 개")
    for i, row in api_volume_matches.head(5).iterrows():
        diff = abs(row['volume'] - csv_volume)
        print(f"   - {row['timestamp']}: {row['volume']:.4f} (차이: {diff:.4f})")
else:
    print(f"   ❌ 매칭 없음")

# Close + Volume 동시 매칭
print(f"\n🔍 CSV Close + Volume 동시 매칭:")
api_both_matches = df_api[
    (abs(df_api['close'] - csv_close) < tolerance_price) &
    (abs(df_api['volume'] - csv_volume) < tolerance_volume)
]
if len(api_both_matches) > 0:
    print(f"   ✅ 발견: {len(api_both_matches)} 개")
    for i, row in api_both_matches.iterrows():
        close_diff = abs(row['close'] - csv_close)
        vol_diff = abs(row['volume'] - csv_volume)
        print(f"\n   📍 {row['timestamp']}")
        print(f"      Close: ${row['close']:.1f} (차이: ${close_diff:.2f})")
        print(f"      Volume: {row['volume']:.4f} (차이: {vol_diff:.4f})")
        print(f"      High: ${row['high']:.1f}")

        # 15:25와의 시간 차이
        time_diff_minutes = (row['timestamp'] - target_time).total_seconds() / 60
        print(f"      15:25로부터: {time_diff_minutes:+.0f} 분")
else:
    print(f"   ❌ 정확한 매칭 없음")

# Question 2: API의 15:25 데이터가 CSV의 어느 타임스탬프에 있는가?
print(f"\n" + "=" * 80)
print("❓ Question 2: API의 15:25 데이터가 CSV의 어느 타임스탬프에 있는가?")
print("=" * 80)

api_close = api_15_25['close']
api_volume = api_15_25['volume']
api_high = api_15_25['high']

# Close 매칭
print(f"\n🔍 API Close (${api_close:.1f}) 매칭:")
csv_close_matches = df_csv[abs(df_csv['close'] - api_close) < tolerance_price]
if len(csv_close_matches) > 0:
    print(f"   발견: {len(csv_close_matches)} 개")
    for i, row in csv_close_matches.head(5).iterrows():
        diff = abs(row['close'] - api_close)
        print(f"   - {row['timestamp']}: ${row['close']:.1f} (차이: ${diff:.2f})")
else:
    print(f"   ❌ 매칭 없음")

# Volume 매칭
print(f"\n🔍 API Volume ({api_volume:.4f}) 매칭:")
csv_volume_matches = df_csv[abs(df_csv['volume'] - api_volume) < tolerance_volume]
if len(csv_volume_matches) > 0:
    print(f"   발견: {len(csv_volume_matches)} 개")
    for i, row in csv_volume_matches.head(5).iterrows():
        diff = abs(row['volume'] - api_volume)
        print(f"   - {row['timestamp']}: {row['volume']:.4f} (차이: {diff:.4f})")
else:
    print(f"   ❌ 매칭 없음")

# Close + Volume 동시 매칭
print(f"\n🔍 API Close + Volume 동시 매칭:")
csv_both_matches = df_csv[
    (abs(df_csv['close'] - api_close) < tolerance_price) &
    (abs(df_csv['volume'] - api_volume) < tolerance_volume)
]
if len(csv_both_matches) > 0:
    print(f"   ✅ 발견: {len(csv_both_matches)} 개")
    for i, row in csv_both_matches.iterrows():
        close_diff = abs(row['close'] - api_close)
        vol_diff = abs(row['volume'] - api_volume)
        print(f"\n   📍 {row['timestamp']}")
        print(f"      Close: ${row['close']:.1f} (차이: ${close_diff:.2f})")
        print(f"      Volume: {row['volume']:.4f} (차이: {vol_diff:.4f})")
        print(f"      High: ${row['high']:.1f}")

        # 15:25와의 시간 차이
        time_diff_minutes = (row['timestamp'] - target_time).total_seconds() / 60
        print(f"      15:25로부터: {time_diff_minutes:+.0f} 분")
else:
    print(f"   ❌ 정확한 매칭 없음")

# Question 3: 15:25 전후 타임스탬프 비교
print(f"\n" + "=" * 80)
print("📊 15:25 전후 타임스탬프 비교 (CSV vs API)")
print("=" * 80)

# Get surrounding timestamps
csv_surrounding = df_csv[
    (df_csv['timestamp'] >= pd.to_datetime('2025-10-26 15:15:00')) &
    (df_csv['timestamp'] <= pd.to_datetime('2025-10-26 15:35:00'))
].copy()

api_surrounding = df_api[
    (df_api['timestamp'] >= pd.to_datetime('2025-10-26 15:15:00')) &
    (df_api['timestamp'] <= pd.to_datetime('2025-10-26 15:35:00'))
].copy()

print(f"\n타임스탬프별 비교:")
print(f"{'Timestamp':<20} {'CSV Close':>12} {'API Close':>12} {'Diff':>10} {'Match':>8}")
print("-" * 70)

for ts in csv_surrounding['timestamp']:
    csv_row = csv_surrounding[csv_surrounding['timestamp'] == ts]
    api_row = api_surrounding[api_surrounding['timestamp'] == ts]

    if len(csv_row) > 0 and len(api_row) > 0:
        csv_close_val = csv_row.iloc[0]['close']
        api_close_val = api_row.iloc[0]['close']
        diff = api_close_val - csv_close_val
        match = "✅" if abs(diff) < tolerance_price else "❌"

        print(f"{str(ts):<20} ${csv_close_val:>11.1f} ${api_close_val:>11.1f} ${diff:>+9.1f} {match:>8}")

print(f"\n" + "=" * 80)
print("🎯 결론")
print("=" * 80)

print("""
만약:
- CSV의 15:25 데이터가 API의 다른 타임스탬프에서 발견되면
  → CSV 수집 시 타임스탬프 오프셋 문제

- API의 15:25 데이터가 CSV의 다른 타임스탬프에서 발견되면
  → 데이터 최종화로 인한 변경

- 둘 다 발견되지 않으면
  → BingX가 데이터를 업데이트했거나
  → 해당 캔들이 미완성이었다가 완성됨
""")

print("=" * 80)
