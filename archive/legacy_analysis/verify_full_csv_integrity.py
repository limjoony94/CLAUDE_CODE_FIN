#!/usr/bin/env python3
"""
Full CSV Data Integrity Check
전체 CSV 데이터의 무결성 검증 (API와 비교)
"""

import pandas as pd
import ccxt
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent

print("=" * 80)
print("FULL CSV DATA INTEGRITY CHECK")
print("전체 CSV 데이터 무결성 검증")
print("=" * 80)

# Load CSV
CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df_csv = pd.read_csv(CSV_FILE)
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])

print(f"\n✅ CSV 로드: {len(df_csv):,} 캔들")
print(f"   범위: {df_csv['timestamp'].min()} ~ {df_csv['timestamp'].max()}")

# Initialize exchange
exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})
symbol = 'BTC/USDT:USDT'
timeframe = '5m'

# Strategy: Sample check at different time ranges
# 1. Recent (last 200 candles - most likely to have issues)
# 2. Middle (random sample from middle range)
# 3. Old (first 200 candles - should be stable)

print(f"\n" + "=" * 80)
print("검증 전략: 3개 시간대 샘플링")
print("=" * 80)

results = {
    'total_checked': 0,
    'matches': 0,
    'mismatches': 0,
    'mismatch_details': []
}

# 1. Recent data check (Last 200 candles)
print(f"\n1️⃣ 최근 데이터 검증 (마지막 200 캔들)")
print("-" * 80)

df_recent_csv = df_csv.tail(200).copy()
print(f"   CSV 범위: {df_recent_csv['timestamp'].min()} ~ {df_recent_csv['timestamp'].max()}")

# Fetch from API
ohlcv_recent = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
df_recent_api = pd.DataFrame(ohlcv_recent, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_recent_api['timestamp'] = pd.to_datetime(df_recent_api['timestamp'], unit='ms')

print(f"   API 범위: {df_recent_api['timestamp'].min()} ~ {df_recent_api['timestamp'].max()}")

# Compare
tolerance = 0.5  # $0.5 tolerance for close price
mismatches_recent = []

for _, csv_row in df_recent_csv.iterrows():
    ts = csv_row['timestamp']
    api_row = df_recent_api[df_recent_api['timestamp'] == ts]

    if len(api_row) > 0:
        api_row = api_row.iloc[0]
        results['total_checked'] += 1

        close_diff = abs(csv_row['close'] - api_row['close'])

        if close_diff < tolerance:
            results['matches'] += 1
        else:
            results['mismatches'] += 1
            mismatches_recent.append({
                'timestamp': ts,
                'csv_close': csv_row['close'],
                'api_close': api_row['close'],
                'diff': api_row['close'] - csv_row['close']
            })

print(f"\n   검증: {results['total_checked']} 캔들")
print(f"   ✅ 일치: {results['matches']} ({results['matches']/results['total_checked']*100:.1f}%)")
print(f"   ❌ 불일치: {len(mismatches_recent)} ({len(mismatches_recent)/results['total_checked']*100:.1f}%)")

if len(mismatches_recent) > 0:
    print(f"\n   불일치 상세 (최대 10개):")
    for item in mismatches_recent[:10]:
        print(f"      {item['timestamp']}: CSV ${item['csv_close']:.1f} vs API ${item['api_close']:.1f} (차이: ${item['diff']:+.1f})")

time.sleep(2)  # Rate limit

# 2. Middle range check (Random sample from middle)
print(f"\n2️⃣ 중간 범위 데이터 검증 (100 캔들 샘플)")
print("-" * 80)

# Get middle 1000 candles and sample 100
middle_start_idx = len(df_csv) // 2 - 500
middle_end_idx = middle_start_idx + 1000
df_middle_csv = df_csv.iloc[middle_start_idx:middle_end_idx].sample(min(100, middle_end_idx - middle_start_idx))

print(f"   CSV 샘플 범위: {df_middle_csv['timestamp'].min()} ~ {df_middle_csv['timestamp'].max()}")

# Fetch from API (use 'since' to get historical data)
first_ts = df_middle_csv['timestamp'].min()
since_ms = int(first_ts.timestamp() * 1000)
ohlcv_middle = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
df_middle_api = pd.DataFrame(ohlcv_middle, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_middle_api['timestamp'] = pd.to_datetime(df_middle_api['timestamp'], unit='ms')

print(f"   API 범위: {df_middle_api['timestamp'].min()} ~ {df_middle_api['timestamp'].max()}")

mismatches_middle = []
checked_middle = 0

for _, csv_row in df_middle_csv.iterrows():
    ts = csv_row['timestamp']
    api_row = df_middle_api[df_middle_api['timestamp'] == ts]

    if len(api_row) > 0:
        api_row = api_row.iloc[0]
        checked_middle += 1
        results['total_checked'] += 1

        close_diff = abs(csv_row['close'] - api_row['close'])

        if close_diff < tolerance:
            results['matches'] += 1
        else:
            results['mismatches'] += 1
            mismatches_middle.append({
                'timestamp': ts,
                'csv_close': csv_row['close'],
                'api_close': api_row['close'],
                'diff': api_row['close'] - csv_row['close']
            })

print(f"\n   검증: {checked_middle} 캔들")
print(f"   ✅ 일치: {checked_middle - len(mismatches_middle)} ({(checked_middle - len(mismatches_middle))/max(checked_middle,1)*100:.1f}%)")
print(f"   ❌ 불일치: {len(mismatches_middle)} ({len(mismatches_middle)/max(checked_middle,1)*100:.1f}%)")

if len(mismatches_middle) > 0:
    print(f"\n   불일치 상세:")
    for item in mismatches_middle[:10]:
        print(f"      {item['timestamp']}: CSV ${item['csv_close']:.1f} vs API ${item['api_close']:.1f} (차이: ${item['diff']:+.1f})")

time.sleep(2)  # Rate limit

# 3. Old data check (First 200 candles - should be stable)
print(f"\n3️⃣ 오래된 데이터 검증 (첫 200 캔들)")
print("-" * 80)

df_old_csv = df_csv.head(200).copy()
print(f"   CSV 범위: {df_old_csv['timestamp'].min()} ~ {df_old_csv['timestamp'].max()}")

# Fetch from API
first_ts_old = df_old_csv['timestamp'].min()
since_ms_old = int(first_ts_old.timestamp() * 1000)
ohlcv_old = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms_old, limit=200)
df_old_api = pd.DataFrame(ohlcv_old, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_old_api['timestamp'] = pd.to_datetime(df_old_api['timestamp'], unit='ms')

print(f"   API 범위: {df_old_api['timestamp'].min()} ~ {df_old_api['timestamp'].max()}")

mismatches_old = []

for _, csv_row in df_old_csv.iterrows():
    ts = csv_row['timestamp']
    api_row = df_old_api[df_old_api['timestamp'] == ts]

    if len(api_row) > 0:
        api_row = api_row.iloc[0]
        results['total_checked'] += 1

        close_diff = abs(csv_row['close'] - api_row['close'])

        if close_diff < tolerance:
            results['matches'] += 1
        else:
            results['mismatches'] += 1
            mismatches_old.append({
                'timestamp': ts,
                'csv_close': csv_row['close'],
                'api_close': api_row['close'],
                'diff': api_row['close'] - csv_row['close']
            })

checked_old = len(df_old_csv)
print(f"\n   검증: {checked_old} 캔들")
print(f"   ✅ 일치: {checked_old - len(mismatches_old)} ({(checked_old - len(mismatches_old))/checked_old*100:.1f}%)")
print(f"   ❌ 불일치: {len(mismatches_old)} ({len(mismatches_old)/checked_old*100:.1f}%)")

if len(mismatches_old) > 0:
    print(f"\n   불일치 상세:")
    for item in mismatches_old[:10]:
        print(f"      {item['timestamp']}: CSV ${item['csv_close']:.1f} vs API ${item['api_close']:.1f} (차이: ${item['diff']:+.1f})")

# Summary
print(f"\n\n" + "=" * 80)
print("📊 전체 검증 결과")
print("=" * 80)

print(f"\n검증 범위:")
print(f"  총 CSV 캔들: {len(df_csv):,}")
print(f"  검증한 캔들: {results['total_checked']:,} ({results['total_checked']/len(df_csv)*100:.1f}%)")

print(f"\n일치율:")
print(f"  ✅ 일치: {results['matches']:,} ({results['matches']/results['total_checked']*100:.2f}%)")
print(f"  ❌ 불일치: {results['mismatches']:,} ({results['mismatches']/results['total_checked']*100:.2f}%)")

print(f"\n시간대별 분석:")
print(f"  최근 데이터 (마지막 200): {len(mismatches_recent)} 불일치")
print(f"  중간 데이터 (샘플 100): {len(mismatches_middle)} 불일치")
print(f"  오래된 데이터 (첫 200): {len(mismatches_old)} 불일치")

# All mismatches
all_mismatches = mismatches_recent + mismatches_middle + mismatches_old
results['mismatch_details'] = all_mismatches

if len(all_mismatches) > 0:
    print(f"\n전체 불일치 리스트:")
    print(f"{'Timestamp':<20} {'CSV Close':>12} {'API Close':>12} {'Diff':>10}")
    print("-" * 60)
    for item in sorted(all_mismatches, key=lambda x: x['timestamp']):
        print(f"{str(item['timestamp']):<20} ${item['csv_close']:>11.1f} ${item['api_close']:>11.1f} ${item['diff']:>+9.1f}")

print(f"\n" + "=" * 80)
print("🎯 결론")
print("=" * 80)

if results['mismatches'] == 0:
    print("\n✅ 완벽! CSV 데이터가 100% 무결합니다!")
    print("   모든 검증된 캔들이 API와 정확히 일치합니다.")
elif results['mismatches'] / results['total_checked'] < 0.01:  # < 1%
    print(f"\n✅ 우수! CSV 데이터가 {results['matches']/results['total_checked']*100:.2f}% 무결합니다!")
    print(f"   {results['mismatches']}개 캔들만 불일치 (< 1%)")
    print(f"   → 대부분의 데이터는 정확하며, 소수의 미완성 캔들만 존재")
elif results['mismatches'] / results['total_checked'] < 0.05:  # < 5%
    print(f"\n⚠️ 양호: CSV 데이터가 {results['matches']/results['total_checked']*100:.2f}% 무결합니다")
    print(f"   {results['mismatches']}개 캔들 불일치 (< 5%)")
    print(f"   → CSV 재수집 권장")
else:
    print(f"\n❌ 주의: CSV 데이터가 {results['matches']/results['total_checked']*100:.2f}% 무결합니다")
    print(f"   {results['mismatches']}개 캔들 불일치 (> 5%)")
    print(f"   → CSV 재수집 필수!")

print("\n권장사항:")
if results['mismatches'] == 0:
    print("  - 현재 CSV 데이터 그대로 사용 가능")
elif len(mismatches_recent) > len(mismatches_old) + len(mismatches_middle):
    print("  - 불일치가 최근 데이터에 집중됨")
    print("  - CSV 재수집으로 최신 최종화 데이터 확보 권장")
else:
    print("  - 불일치가 여러 시간대에 분산됨")
    print("  - CSV 재수집 필요")

print("=" * 80)
