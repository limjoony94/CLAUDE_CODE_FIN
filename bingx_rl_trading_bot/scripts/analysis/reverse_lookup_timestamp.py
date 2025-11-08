#!/usr/bin/env python3
"""
Reverse Timestamp Lookup
역방향 타임스탬프 추적: API 데이터 값이 CSV의 어느 타임스탬프에 있는지 확인
"""

import pandas as pd
import ccxt
from datetime import datetime
import pytz
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

print("=" * 80)
print("REVERSE TIMESTAMP LOOKUP")
print("역방향 타임스탬프 추적")
print("=" * 80)

# 1. Get API data for 15:25 UTC
print("\n1️⃣ API 데이터 가져오기 (15:25 UTC)")
print("-" * 80)

target_timestamp_ms = int(datetime(2025, 10, 26, 15, 25, 0, tzinfo=pytz.UTC).timestamp() * 1000)
print(f"Target: 2025-10-26 15:25:00 UTC (Timestamp: {target_timestamp_ms})")

exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})

# Fetch with 'since' to ensure we get the target candle
ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', since=target_timestamp_ms, limit=50)

api_candle = None
for candle in ohlcv:
    if candle[0] == target_timestamp_ms:
        api_candle = candle
        break

if not api_candle:
    print(f"❌ API에서 해당 타임스탬프를 찾을 수 없습니다")
    exit(1)

api_close = api_candle[4]
api_high = api_candle[2]
api_low = api_candle[3]
api_volume = api_candle[5]

print(f"\n✅ API 데이터 (Timestamp {target_timestamp_ms}):")
print(f"   Close:  ${api_close:.1f}")
print(f"   High:   ${api_high:.1f}")
print(f"   Low:    ${api_low:.1f}")
print(f"   Volume: {api_volume:.4f}")

# 2. Load CSV
print(f"\n2️⃣ CSV 로드")
print("-" * 80)

CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✅ CSV 로드 완료: {len(df)} 캔들")
print(f"   범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# 3. Find matching close price
print(f"\n3️⃣ API Close 값 매칭 (${api_close:.1f})")
print("-" * 80)

tolerance = 0.5  # $0.5 tolerance
matches_close = df[abs(df['close'] - api_close) < tolerance].copy()

print(f"\nClose 매칭 결과 (오차 < ${tolerance}):")
print(f"   발견: {len(matches_close)} 개")

if len(matches_close) > 0:
    print(f"\n   매칭된 타임스탬프:")
    for i, row in matches_close.head(10).iterrows():
        diff_close = abs(row['close'] - api_close)
        print(f"   - {row['timestamp']}: Close ${row['close']:.1f} (차이: ${diff_close:.2f})")

# 4. Find matching volume
print(f"\n4️⃣ API Volume 값 매칭 ({api_volume:.4f})")
print("-" * 80)

vol_tolerance = 0.1  # 0.1 BTC tolerance
matches_volume = df[abs(df['volume'] - api_volume) < vol_tolerance].copy()

print(f"\nVolume 매칭 결과 (오차 < {vol_tolerance}):")
print(f"   발견: {len(matches_volume)} 개")

if len(matches_volume) > 0:
    print(f"\n   매칭된 타임스탬프:")
    for i, row in matches_volume.head(10).iterrows():
        diff_vol = abs(row['volume'] - api_volume)
        print(f"   - {row['timestamp']}: Volume {row['volume']:.4f} (차이: {diff_vol:.4f})")

# 5. Find matching both close AND volume
print(f"\n5️⃣ Close + Volume 동시 매칭")
print("-" * 80)

matches_both = df[
    (abs(df['close'] - api_close) < tolerance) &
    (abs(df['volume'] - api_volume) < vol_tolerance)
].copy()

print(f"\nClose + Volume 동시 매칭:")
print(f"   발견: {len(matches_both)} 개")

if len(matches_both) > 0:
    print(f"\n   ✅ 정확히 매칭된 타임스탬프:")
    for i, row in matches_both.iterrows():
        diff_close = abs(row['close'] - api_close)
        diff_vol = abs(row['volume'] - api_volume)
        print(f"\n   Timestamp: {row['timestamp']}")
        print(f"      Close:  ${row['close']:.1f} (차이: ${diff_close:.2f})")
        print(f"      Volume: {row['volume']:.4f} (차이: {diff_vol:.4f})")

        # Check offset from target
        if isinstance(row['timestamp'], pd.Timestamp):
            target_dt = pd.to_datetime('2025-10-26 15:25:00')
            time_diff = (row['timestamp'] - target_dt).total_seconds() / 3600
            print(f"      오프셋: {time_diff:+.1f} 시간")
else:
    print(f"   ❌ 정확히 매칭되는 타임스탬프 없음")

# 6. Check if it's a KST offset
print(f"\n6️⃣ KST 오프셋 가설 검증 (9시간 차이)")
print("-" * 80)

# 15:25 UTC = 00:25 KST (next day)
kst_timestamp_str = '2025-10-27 00:25:00'
kst_row = df[df['timestamp'] == kst_timestamp_str]

print(f"\nKST 타임스탬프 확인: {kst_timestamp_str}")
if len(kst_row) > 0:
    kst_close = kst_row.iloc[0]['close']
    kst_volume = kst_row.iloc[0]['volume']
    print(f"   Close:  ${kst_close:.1f}")
    print(f"   Volume: {kst_volume:.4f}")

    # Compare with API
    close_diff = abs(kst_close - api_close)
    vol_diff = abs(kst_volume - api_volume)
    print(f"\n   API와 비교:")
    print(f"      Close 차이:  ${close_diff:.2f}")
    print(f"      Volume 차이: {vol_diff:.4f}")

    if close_diff < tolerance and vol_diff < vol_tolerance:
        print(f"\n   🎯 정확히 매칭! KST 오프셋 문제 확인됨!")
    else:
        print(f"\n   ❌ 매칭 안됨 - KST 오프셋 아님")
else:
    print(f"   ❌ {kst_timestamp_str} 타임스탬프 없음")

# 7. Check what CSV has for 15:25 UTC
print(f"\n7️⃣ CSV에서 15:25 UTC 타임스탬프 데이터")
print("-" * 80)

utc_timestamp_str = '2025-10-26 15:25:00'
utc_row = df[df['timestamp'] == utc_timestamp_str]

print(f"\nUTC 타임스탬프 확인: {utc_timestamp_str}")
if len(utc_row) > 0:
    csv_close = utc_row.iloc[0]['close']
    csv_volume = utc_row.iloc[0]['volume']
    print(f"   Close:  ${csv_close:.1f}")
    print(f"   Volume: {csv_volume:.4f}")

    # Compare with API
    close_diff = abs(csv_close - api_close)
    vol_diff = abs(csv_volume - api_volume)
    print(f"\n   API와 비교:")
    print(f"      Close 차이:  ${close_diff:.2f} ({'❌ 다름' if close_diff > tolerance else '✅ 같음'})")
    print(f"      Volume 차이: {vol_diff:.4f} ({'❌ 다름' if vol_diff > vol_tolerance else '✅ 같음'})")
else:
    print(f"   ❌ {utc_timestamp_str} 타임스탬프 없음")

print("\n" + "=" * 80)
print("🎯 진단 결과")
print("=" * 80)

print("""
예상 시나리오:

1. KST 오프셋 문제:
   - API 데이터 (15:25 UTC)가 CSV의 00:25 KST에 매칭됨
   - 타임존 해석 차이로 9시간 오프셋 발생
   - collect_max_data.py가 KST로 저장했거나, 읽을 때 KST로 해석

2. 데이터 업데이트:
   - CSV와 현재 API 데이터가 실제로 다름
   - 하지만 BingX는 데이터 수정 안함 (사용자 확인)
   - 따라서 타임스탬프 문제일 가능성 높음

3. 해결책:
   - collect_max_data.py: 명시적 UTC 저장 필요
   - 모든 스크립트: 명시적 UTC 해석 필요
   - 또는 타임스탬프(ms)만 사용하고 datetime 문자열 회피
""")

print("=" * 80)
