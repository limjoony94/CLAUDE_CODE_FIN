#!/usr/bin/env python3
"""
API Call Consistency Test
다른 API 호출 방식(since 유무)이 같은 캔들에 대해 다른 데이터를 반환하는지 테스트
"""

import ccxt
from datetime import datetime
import pytz

print("=" * 80)
print("API CALL CONSISTENCY TEST")
print("API 호출 방식별 데이터 일관성 검증")
print("=" * 80)

exchange = ccxt.bingx({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

symbol = 'BTC/USDT:USDT'
timeframe = '5m'

# Target timestamp
target_timestamp_ms = int(datetime(2025, 10, 26, 15, 25, 0, tzinfo=pytz.UTC).timestamp() * 1000)
target_dt = datetime.fromtimestamp(target_timestamp_ms/1000, pytz.UTC)

print(f"\n🎯 Target: {target_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"   Timestamp: {target_timestamp_ms}")

# Method 1: Fetch latest (no 'since' - like Batch 1)
print(f"\n1️⃣ Method 1: fetch_ohlcv(limit=1000) - NO 'since' parameter")
print("-" * 80)
print("   (collect_max_data.py의 첫 번째 배치 방식)")

ohlcv_method1 = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)

candle_method1 = None
for candle in ohlcv_method1:
    if candle[0] == target_timestamp_ms:
        candle_method1 = candle
        break

if candle_method1:
    print(f"\n✅ Method 1 결과:")
    print(f"   Timestamp: {candle_method1[0]}")
    print(f"   Open:   ${candle_method1[1]:.1f}")
    print(f"   High:   ${candle_method1[2]:.1f}")
    print(f"   Low:    ${candle_method1[3]:.1f}")
    print(f"   Close:  ${candle_method1[4]:.1f}")
    print(f"   Volume: {candle_method1[5]:.4f}")
else:
    print(f"\n❌ Method 1: Target 캔들 없음 (limit=1000 범위 밖)")

# Method 2: Fetch with 'since' (like Batch 2)
print(f"\n2️⃣ Method 2: fetch_ohlcv(since={target_timestamp_ms}, limit=1000)")
print("-" * 80)
print("   (collect_max_data.py의 두 번째+ 배치 방식)")

ohlcv_method2 = exchange.fetch_ohlcv(symbol, timeframe, since=target_timestamp_ms, limit=1000)

candle_method2 = None
for candle in ohlcv_method2:
    if candle[0] == target_timestamp_ms:
        candle_method2 = candle
        break

if candle_method2:
    print(f"\n✅ Method 2 결과:")
    print(f"   Timestamp: {candle_method2[0]}")
    print(f"   Open:   ${candle_method2[1]:.1f}")
    print(f"   High:   ${candle_method2[2]:.1f}")
    print(f"   Low:    ${candle_method2[3]:.1f}")
    print(f"   Close:  ${candle_method2[4]:.1f}")
    print(f"   Volume: {candle_method2[5]:.4f}")
else:
    print(f"\n❌ Method 2: Target 캔들 없음")

# Method 3: Fetch backward (simulate Batch 2 strategy)
print(f"\n3️⃣ Method 3: Backward collection simulation")
print("-" * 80)
print("   (collect_max_data.py Line 65: since = until - 1000 candles)")

# Calculate 'since' like collect_max_data.py does
# If we're in batch 2 and 'until' = target_timestamp,
# then since = target - (1000 * 5 * 60 * 1000) - 1
until = target_timestamp_ms + (5 * 60 * 1000)  # 1 candle after target
since_for_batch = until - (1000 * 5 * 60 * 1000) - 1

print(f"\n   Until (가장 최근 타임스탬프): {until}")
print(f"   Since (계산): {since_for_batch}")
print(f"   Since (datetime): {datetime.fromtimestamp(since_for_batch/1000, pytz.UTC)}")

ohlcv_method3 = exchange.fetch_ohlcv(symbol, timeframe, since=since_for_batch, limit=1000)

candle_method3 = None
for candle in ohlcv_method3:
    if candle[0] == target_timestamp_ms:
        candle_method3 = candle
        break

if candle_method3:
    print(f"\n✅ Method 3 결과:")
    print(f"   Timestamp: {candle_method3[0]}")
    print(f"   Open:   ${candle_method3[1]:.1f}")
    print(f"   High:   ${candle_method3[2]:.1f}")
    print(f"   Low:    ${candle_method3[3]:.1f}")
    print(f"   Close:  ${candle_method3[4]:.1f}")
    print(f"   Volume: {candle_method3[5]:.4f}")
else:
    print(f"\n❌ Method 3: Target 캔들 없음")

# Comparison
print(f"\n\n" + "=" * 80)
print("🔍 비교 분석")
print("=" * 80)

methods = {
    'Method 1 (no since)': candle_method1,
    'Method 2 (since=target)': candle_method2,
    'Method 3 (backward)': candle_method3,
}

if all(c is not None for c in [candle_method1, candle_method2, candle_method3]):
    print(f"\n모든 방법에서 캔들 발견 ✅")

    # Compare close prices
    closes = [c[4] for c in [candle_method1, candle_method2, candle_method3]]
    volumes = [c[5] for c in [candle_method1, candle_method2, candle_method3]]

    print(f"\nClose 가격 비교:")
    for name, candle in methods.items():
        print(f"   {name:<25}: ${candle[4]:.1f}")

    print(f"\nVolume 비교:")
    for name, candle in methods.items():
        print(f"   {name:<25}: {candle[5]:.4f}")

    # Check if all identical
    if len(set(closes)) == 1 and len(set(volumes)) == 1:
        print(f"\n✅ 결론: 모든 방법이 동일한 데이터 반환")
        print(f"   → API 호출 방식과 무관하게 일관된 데이터")
    else:
        print(f"\n❌ 결론: 다른 방법이 다른 데이터 반환!")
        print(f"   → 이것이 CSV 데이터 불일치의 원인!")
        print(f"\n   Close 가격 차이:")
        for i, (name, candle) in enumerate(methods.items()):
            if i > 0:
                diff = candle[4] - candle_method1[4]
                print(f"      {name} vs Method 1: ${diff:+.2f}")

        print(f"\n   Volume 차이:")
        for i, (name, candle) in enumerate(methods.items()):
            if i > 0:
                diff = candle[5] - candle_method1[5]
                print(f"      {name} vs Method 1: {diff:+.4f}")
else:
    print(f"\n⚠️ 일부 방법에서 캔들을 찾을 수 없음:")
    for name, candle in methods.items():
        status = "✅ 발견" if candle else "❌ 없음"
        print(f"   {name}: {status}")

# Test 4: CSV data comparison
print(f"\n\n" + "=" * 80)
print("📊 CSV 데이터와 비교")
print("=" * 80)

print(f"\nCSV 데이터 (2025-10-26 15:25:00):")
print(f"   Close:  $113,439.3")
print(f"   Volume: 27.3542")

if candle_method1:
    close_diff = abs(candle_method1[4] - 113439.3)
    vol_diff = abs(candle_method1[5] - 27.3542)
    print(f"\nMethod 1 (no since) vs CSV:")
    print(f"   Close 차이:  ${close_diff:.2f} ({'✅ 같음' if close_diff < 0.5 else '❌ 다름'})")
    print(f"   Volume 차이: {vol_diff:.4f} ({'✅ 같음' if vol_diff < 0.1 else '❌ 다름'})")

if candle_method3:
    close_diff = abs(candle_method3[4] - 113439.3)
    vol_diff = abs(candle_method3[5] - 27.3542)
    print(f"\nMethod 3 (backward) vs CSV:")
    print(f"   Close 차이:  ${close_diff:.2f} ({'✅ 같음' if close_diff < 0.5 else '❌ 다름'})")
    print(f"   Volume 차이: {vol_diff:.4f} ({'✅ 같음' if vol_diff < 0.1 else '❌ 다름'})")

print("\n" + "=" * 80)
print("🎯 최종 진단")
print("=" * 80)

print("""
가설 검증:

1. CSV는 collect_max_data.py의 배치 수집으로 생성됨
2. 배치 1 (limit=1000, no since): 최신 데이터
3. 배치 2+ (since 사용, backward): 과거 데이터

만약 Method 1과 Method 3가 다른 데이터를 반환한다면:
  → CSV에 저장된 데이터는 배치 수집 방식에 따라 달라질 수 있음
  → 이것이 프로덕션 봇(실시간 API)과 백테스트(CSV)의 차이 원인

해결책:
  1. CSV 재수집 (최신 API 데이터로 갱신)
  2. 또는 프로덕션 봇이 실시간 API 사용 (이미 사용 중)
  3. 백테스트는 CSV 데이터 기반 → 불일치 허용 가능
""")

print("=" * 80)
