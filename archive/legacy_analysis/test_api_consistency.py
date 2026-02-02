#!/usr/bin/env python3
"""
Test API Consistency - Multiple Calls for Same Candle
테스트: 동일한 캔들에 대해 API를 여러 번 호출했을 때 데이터가 일관된지 확인
"""

import ccxt
import pandas as pd
from datetime import datetime
import time

print("=" * 80)
print("API CONSISTENCY TEST")
print("=" * 80)

# Initialize exchange
exchange = ccxt.bingx({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

symbol = 'BTC/USDT:USDT'
timeframe = '5m'

# Target: 15:25 UTC candle (2025-10-26)
target_timestamp_ms = int(datetime(2025, 10, 26, 15, 25, 0).timestamp() * 1000)
target_timestamp_utc = datetime.fromtimestamp(target_timestamp_ms / 1000)

print(f"\n🎯 Target Candle: {target_timestamp_utc} UTC")
print(f"   Timestamp (ms): {target_timestamp_ms}")
print(f"   Candle ends at: {datetime.fromtimestamp((target_timestamp_ms + 5*60*1000) / 1000)} UTC")

# Test 1: Call API 3 times with limit=1000, no since
print(f"\n📊 Test 1: fetch_ohlcv with limit=1000 (no since)")
print(f"   Calling API 3 times and comparing results...")

results = []
for i in range(3):
    print(f"\n   Call {i+1}/3...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)

    # Find our target candle
    target_candle = None
    for candle in ohlcv:
        if candle[0] == target_timestamp_ms:
            target_candle = candle
            break

    if target_candle:
        results.append({
            'call': i+1,
            'timestamp': target_candle[0],
            'open': target_candle[1],
            'high': target_candle[2],
            'low': target_candle[3],
            'close': target_candle[4],
            'volume': target_candle[5]
        })
        print(f"      Close: {target_candle[4]:.1f}, Volume: {target_candle[5]:.4f}")
    else:
        print(f"      ❌ Target candle NOT FOUND in response")

    if i < 2:
        time.sleep(1)

# Compare results
print(f"\n📊 Comparison:")
print("="*80)
if len(results) == 3:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Check consistency
    close_values = df['close'].unique()
    volume_values = df['volume'].unique()

    if len(close_values) == 1 and len(volume_values) == 1:
        print(f"\n✅ API is CONSISTENT - All 3 calls returned identical data")
    else:
        print(f"\n❌ API is INCONSISTENT - Different values across calls!")
        print(f"   Close variations: {close_values}")
        print(f"   Volume variations: {volume_values}")
else:
    print(f"⚠️ Could not get target candle in all 3 calls")

# Test 2: Call with different 'since' values
print(f"\n\n📊 Test 2: Different 'since' parameter strategies")
print("="*80)

strategies = [
    {'name': 'Recent (limit=1000)', 'since': None, 'limit': 1000},
    {'name': 'Before target (since=15:00)', 'since': int(datetime(2025, 10, 26, 15, 0, 0).timestamp() * 1000), 'limit': 1000},
    {'name': 'Way before (since=14:00)', 'since': int(datetime(2025, 10, 26, 14, 0, 0).timestamp() * 1000), 'limit': 1000},
]

strategy_results = []
for strat in strategies:
    print(f"\n🔍 Strategy: {strat['name']}")

    params = {'limit': strat['limit']}
    if strat['since']:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=strat['since'], **params)
    else:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, **params)

    # Find target candle
    target_candle = None
    for candle in ohlcv:
        if candle[0] == target_timestamp_ms:
            target_candle = candle
            break

    if target_candle:
        strategy_results.append({
            'strategy': strat['name'],
            'close': target_candle[4],
            'volume': target_candle[5]
        })
        print(f"   Close: {target_candle[4]:.1f}, Volume: {target_candle[5]:.4f}")
    else:
        print(f"   ❌ Target candle NOT FOUND")

    time.sleep(1)

# Compare strategies
print(f"\n📊 Strategy Comparison:")
print("="*80)
if len(strategy_results) == len(strategies):
    df_strat = pd.DataFrame(strategy_results)
    print(df_strat.to_string(index=False))

    close_values = df_strat['close'].unique()
    volume_values = df_strat['volume'].unique()

    if len(close_values) == 1 and len(volume_values) == 1:
        print(f"\n✅ All strategies returned IDENTICAL data")
    else:
        print(f"\n❌ Different strategies returned DIFFERENT data!")
        print(f"   Close variations: {close_values}")
        print(f"   Volume variations: {volume_values}")

print(f"\n\n" + "="*80)
print("🎯 CONCLUSION")
print("="*80)
print("If all calls return identical data:")
print("  → BingX API is consistent for completed candles")
print("  → Problem is in how WE process/save the data")
print("")
print("If calls return different data:")
print("  → API might update historical data (rare but possible)")
print("  → Need to investigate BingX's data update policies")
print("="*80)
