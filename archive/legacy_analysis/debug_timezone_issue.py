#!/usr/bin/env python3
"""
Timezone Issue Investigation
타임존 변환 문제 정밀 분석
"""

import pandas as pd
import ccxt
from datetime import datetime
import pytz

print("=" * 80)
print("TIMEZONE ISSUE INVESTIGATION")
print("=" * 80)

# 1. API에서 직접 가져온 raw timestamp 확인
print("\n1️⃣ API Raw Data Check")
print("-" * 80)

exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})
ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', limit=10)

print("Latest 3 candles from API:")
for i in range(3):
    candle = ohlcv[-(i+1)]
    ts_ms = candle[0]

    # Convert to different timezones
    dt_utc = datetime.fromtimestamp(ts_ms/1000, pytz.UTC)
    dt_kst = dt_utc.astimezone(pytz.timezone('Asia/Seoul'))
    dt_naive = datetime.fromtimestamp(ts_ms/1000)  # System timezone

    print(f"\nCandle {i+1}:")
    print(f"  Raw timestamp (ms): {ts_ms}")
    print(f"  UTC (aware):        {dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  KST (aware):        {dt_kst.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Naive (system):     {dt_naive.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Close:              {candle[4]:.1f}")

# 2. CSV 파일 읽기 - pandas가 어떻게 해석하는지 확인
print("\n\n2️⃣ CSV Data Interpretation")
print("-" * 80)

df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv')

# Raw CSV string 확인
print("First 3 rows (raw CSV strings):")
print(df.head(3)[['timestamp', 'close', 'volume']])

# pd.to_datetime 변환 후
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("\nAfter pd.to_datetime:")
print(f"  Dtype: {df['timestamp'].dtype}")
print(f"  Timezone aware? {df['timestamp'].dt.tz is not None}")
if df['timestamp'].dt.tz is not None:
    print(f"  Timezone: {df['timestamp'].dt.tz}")

print("\nLast 3 rows:")
for idx in range(-3, 0):
    row = df.iloc[idx]
    print(f"\n  Row {len(df) + idx}:")
    print(f"    timestamp: {row['timestamp']}")
    print(f"    close:     {row['close']:.1f}")
    print(f"    volume:    {row['volume']:.4f}")

# 3. 특정 timestamp 검증 - CSV vs API
print("\n\n3️⃣ Specific Candle Comparison (15:25 UTC)")
print("-" * 80)

# Target timestamp (UTC 기준)
target_utc_str = '2025-10-26 15:25:00'
target_dt_utc = datetime(2025, 10, 26, 15, 25, 0, tzinfo=pytz.UTC)
target_ts_ms = int(target_dt_utc.timestamp() * 1000)

print(f"Target: {target_utc_str} UTC")
print(f"Timestamp (ms): {target_ts_ms}")

# CSV에서 찾기
csv_row = df[df['timestamp'] == target_utc_str]
print(f"\nCSV search result:")
if len(csv_row) > 0:
    print(f"  ✅ Found")
    print(f"  Close:  {csv_row.iloc[0]['close']:.1f}")
    print(f"  Volume: {csv_row.iloc[0]['volume']:.4f}")
else:
    print(f"  ❌ Not found with string '{target_utc_str}'")

    # Timestamp로 검색
    csv_row_ts = df[df['timestamp'] == pd.to_datetime(target_utc_str)]
    if len(csv_row_ts) > 0:
        print(f"  ✅ Found with pd.to_datetime")
        print(f"  Close:  {csv_row_ts.iloc[0]['close']:.1f}")
        print(f"  Volume: {csv_row_ts.iloc[0]['volume']:.4f}")

# API에서 찾기
print(f"\nAPI search result:")
api_candle = None
for candle in ohlcv:
    if candle[0] == target_ts_ms:
        api_candle = candle
        break

if api_candle:
    print(f"  ✅ Found")
    print(f"  Close:  {api_candle[4]:.1f}")
    print(f"  Volume: {api_candle[5]:.4f}")
else:
    print(f"  ❌ Not found in recent candles")

# 4. collect_max_data.py 로직 시뮬레이션
print("\n\n4️⃣ collect_max_data.py Logic Simulation")
print("-" * 80)

# Fetch some candles
test_ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', limit=5)

print("API returns (raw):")
for i, candle in enumerate(test_ohlcv):
    print(f"  Candle {i}: timestamp={candle[0]}, close={candle[4]:.1f}")

# Convert like collect_max_data.py does (line 110-111)
df_test = pd.DataFrame(test_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
print(f"\nBefore pd.to_datetime:")
print(f"  dtype: {df_test['timestamp'].dtype}")
print(f"  values: {df_test['timestamp'].head(2).tolist()}")

df_test['timestamp'] = pd.to_datetime(df_test['timestamp'], unit='ms')
print(f"\nAfter pd.to_datetime(unit='ms'):")
print(f"  dtype: {df_test['timestamp'].dtype}")
print(f"  tz-aware? {df_test['timestamp'].dt.tz is not None}")
print(f"  values:\n{df_test[['timestamp', 'close']].head(2)}")

# 5. 프로덕션 봇이 사용하는 방식 확인
print("\n\n5️⃣ Production Bot Logic Check")
print("-" * 80)

# 프로덕션 봇은 BingxClient.get_klines()를 사용
# 이것도 결국 ccxt.fetch_ohlcv()를 호출하고 DataFrame으로 변환

print("Checking if production bot converts timestamps differently...")
# BingxClient는 timestamp를 그대로 유지하고 DataFrame으로 변환
# 확인 필요: 프로덕션 봇의 DataFrame 변환 로직

print("\n" + "=" * 80)
print("🎯 DIAGNOSIS")
print("=" * 80)

print("\nKey Questions:")
print("1. CSV timestamp column - timezone aware? UTC? KST?")
print("2. pandas read_csv - interprets as UTC or local?")
print("3. Production bot - converts timestamp to which timezone?")
print("4. Backtest scripts - expect which timezone?")

print("\nNext Steps:")
print("1. Check if CSV timestamps are naive (no TZ) or aware (with TZ)")
print("2. Check if production bot and backtest use same TZ interpretation")
print("3. Fix timezone consistency across all components")
print("=" * 80)
