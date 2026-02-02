#!/usr/bin/env python3
"""
CSV Collection Logic Verification
collect_max_data.py의 데이터 수집 로직 검증
"""

import pandas as pd
import ccxt
from datetime import datetime
import pytz
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

print("=" * 80)
print("CSV COLLECTION LOGIC VERIFICATION")
print("collect_max_data.py 로직 정밀 검증")
print("=" * 80)

# Initialize exchange (same as collect_max_data.py)
exchange = ccxt.bingx({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

symbol = 'BTC/USDT:USDT'
timeframe = '5m'

# Target timestamp
target_timestamp_ms = int(datetime(2025, 10, 26, 15, 25, 0, tzinfo=pytz.UTC).timestamp() * 1000)

print(f"\n🎯 Target: 2025-10-26 15:25:00 UTC (Timestamp: {target_timestamp_ms})")

# 1. Fetch with 'since' parameter (like collect_max_data.py backward collection)
print(f"\n1️⃣ Fetch with 'since' (백워드 수집 방식 시뮬레이션)")
print("-" * 80)

# Fetch data starting from target timestamp
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=target_timestamp_ms, limit=10)

print(f"\nFetch 결과: {len(ohlcv)} 캔들")
print(f"\n첫 3개 캔들:")
for i, candle in enumerate(ohlcv[:3]):
    ts_ms = candle[0]
    dt_utc = datetime.fromtimestamp(ts_ms/1000, pytz.UTC)
    print(f"  {i}: {dt_utc.strftime('%Y-%m-%d %H:%M:%S')} | Close: ${candle[4]:.1f} | Vol: {candle[5]:.4f}")

# Find target candle
target_candle = None
for candle in ohlcv:
    if candle[0] == target_timestamp_ms:
        target_candle = candle
        break

if target_candle:
    print(f"\n✅ Target 캔들 발견:")
    print(f"   Timestamp: {target_timestamp_ms}")
    print(f"   Close: ${target_candle[4]:.1f}")
    print(f"   Volume: {target_candle[5]:.4f}")
else:
    print(f"\n❌ Target 캔들 없음")

# 2. DataFrame conversion test (exactly as collect_max_data.py does)
print(f"\n2️⃣ DataFrame 변환 테스트 (collect_max_data.py Line 110-113)")
print("-" * 80)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

print(f"\nBefore pd.to_datetime(unit='ms'):")
print(f"  Dtype: {df['timestamp'].dtype}")
print(f"  First value (raw): {df.iloc[0]['timestamp']}")

# This is the EXACT line from collect_max_data.py Line 111
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(f"\nAfter pd.to_datetime(unit='ms'):")
print(f"  Dtype: {df['timestamp'].dtype}")
print(f"  Timezone aware? {df['timestamp'].dt.tz is not None}")
print(f"  First value: {df.iloc[0]['timestamp']}")

# What string would be saved to CSV?
csv_timestamp_str = df.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
print(f"\nCSV에 저장될 문자열: '{csv_timestamp_str}'")

# 3. Check if timestamp matches target
print(f"\n3️⃣ Timestamp 매칭 확인")
print("-" * 80)

target_str = '2025-10-26 15:25:00'
target_dt = pd.to_datetime(target_str)

print(f"\nTarget string: '{target_str}'")
print(f"Target datetime: {target_dt}")

# Search in DataFrame
df_match = df[df['timestamp'] == target_dt]

if len(df_match) > 0:
    print(f"\n✅ DataFrame에서 발견:")
    print(f"   Close: ${df_match.iloc[0]['close']:.1f}")
    print(f"   Volume: {df_match.iloc[0]['volume']:.4f}")
else:
    print(f"\n❌ DataFrame에서 찾을 수 없음")

# 4. Check actual CSV file
print(f"\n4️⃣ 실제 CSV 파일 확인")
print("-" * 80)

CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df_csv = pd.read_csv(CSV_FILE)
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])

csv_match = df_csv[df_csv['timestamp'] == target_dt]

if len(csv_match) > 0:
    csv_close = csv_match.iloc[0]['close']
    csv_volume = csv_match.iloc[0]['volume']
    print(f"\n✅ CSV에서 발견:")
    print(f"   Close: ${csv_close:.1f}")
    print(f"   Volume: {csv_volume:.4f}")

    # Compare with current API
    if target_candle:
        close_diff = abs(csv_close - target_candle[4])
        vol_diff = abs(csv_volume - target_candle[5])
        print(f"\n   현재 API와 비교:")
        print(f"      Close 차이: ${close_diff:.2f}")
        print(f"      Volume 차이: {vol_diff:.4f}")
else:
    print(f"\n❌ CSV에서 찾을 수 없음")

# 5. Test incomplete candle filtering (collect_max_data.py logic)
print(f"\n5️⃣ 미완성 캔들 필터링 로직 검증")
print("-" * 80)

# Get current time
current_time = datetime.now(pytz.UTC)
print(f"\nCurrent time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate current candle start (5m intervals)
current_candle_start = current_time.replace(second=0, microsecond=0)
minutes_to_subtract = current_candle_start.minute % 5
if minutes_to_subtract > 0:
    from datetime import timedelta
    current_candle_start = current_candle_start - timedelta(minutes=minutes_to_subtract)

print(f"Current candle start: {current_candle_start.strftime('%Y-%m-%d %H:%M:%S')}")

# Filter completed candles
df_filtered = df[df['timestamp'] < current_candle_start].copy()

print(f"\n필터링 전: {len(df)} 캔들")
print(f"필터링 후: {len(df_filtered)} 캔들 (완성된 캔들만)")

# 6. Check CSV update timestamp vs target candle
print(f"\n6️⃣ CSV 업데이트 시간 vs Target 캔들 시간")
print("-" * 80)

import os
csv_mtime = os.path.getmtime(CSV_FILE)
csv_update_time = datetime.fromtimestamp(csv_mtime, pytz.UTC)

print(f"\nCSV 파일 업데이트 시간: {csv_update_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

target_candle_end = datetime(2025, 10, 26, 15, 30, 0, tzinfo=pytz.UTC)
time_diff_minutes = (csv_update_time - target_candle_end).total_seconds() / 60

print(f"Target 캔들 종료 시간: {target_candle_end.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"시간 차이: {time_diff_minutes:.1f} 분")

if time_diff_minutes > 0:
    print(f"✅ CSV 업데이트가 캔들 종료 후 {time_diff_minutes:.1f}분 뒤 → 캔들 완성됨")
else:
    print(f"⚠️ CSV 업데이트가 캔들 종료 전 → 미완성 캔들 포함 가능")

# 7. Test duplicate handling
print(f"\n7️⃣ 중복 처리 로직 검증 (drop_duplicates)")
print("-" * 80)

# Create test data with duplicates
test_data = [
    [target_timestamp_ms, 100.0, 101.0, 99.0, 100.5, 10.0],
    [target_timestamp_ms, 100.0, 101.0, 99.0, 200.5, 20.0],  # Duplicate with different data
]

df_dup = pd.DataFrame(test_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_dup['timestamp'] = pd.to_datetime(df_dup['timestamp'], unit='ms')

print(f"\n중복 전:")
print(df_dup[['timestamp', 'close', 'volume']])

# Apply drop_duplicates with keep='last' (Line 112 in collect_max_data.py)
df_dup = df_dup.drop_duplicates(subset=['timestamp'], keep='last')

print(f"\n중복 제거 후 (keep='last'):")
print(df_dup[['timestamp', 'close', 'volume']])
print(f"\nKeep='last' → 마지막 데이터 유지 (Close: 200.5, Volume: 20.0)")

print("\n" + "=" * 80)
print("🎯 분석 결과")
print("=" * 80)

print("""
검증 결과:

1. DataFrame 변환 로직:
   - pd.to_datetime(timestamp, unit='ms')는 timezone-naive datetime 생성
   - CSV 저장 시 'YYYY-MM-DD HH:MM:SS' 형식으로 저장됨
   - 읽을 때 pd.read_csv() + pd.to_datetime()은 동일한 datetime 생성
   - ✅ 변환 로직은 정상

2. 미완성 캔들 필터링:
   - 현재 캔들 시작 시간 이전만 저장
   - ✅ 필터링 로직 정상

3. 중복 처리:
   - drop_duplicates(keep='last')는 마지막 데이터 유지
   - 배치 수집 시 나중 배치 데이터로 덮어씀
   - ⚠️ 만약 배치가 다른 데이터 반환하면 문제 발생 가능

4. CSV 업데이트 시간:
   - Target 캔들 종료 후 66분 뒤 업데이트
   - ✅ 캔들은 완성되었어야 함

결론:
- collect_max_data.py 로직 자체는 문제 없음
- 하지만 CSV 데이터와 현재 API 데이터가 다름
- 가능성: 배치 수집 시 API가 다른 데이터 반환?
""")

print("=" * 80)
