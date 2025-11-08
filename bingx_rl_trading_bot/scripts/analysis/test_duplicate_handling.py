#!/usr/bin/env python3
"""
Test Duplicate Candle Handling in Batch Collection
테스트: 역순 배치 수집 시 중복 캔들 처리가 데이터를 변경하는지 확인
"""

import ccxt
import pandas as pd
from datetime import datetime
import time

print("=" * 80)
print("DUPLICATE CANDLE HANDLING TEST")
print("=" * 80)

# Initialize exchange
exchange = ccxt.bingx({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

symbol = 'BTC/USDT:USDT'
timeframe = '5m'

# Target candle
target_timestamp_ms = int(datetime(2025, 10, 26, 15, 25, 0).timestamp() * 1000)

print(f"\n🎯 Target: 2025-10-26 15:25:00 UTC (timestamp: {target_timestamp_ms})")

# Strategy 1: Fetch recent data (like Batch 1)
print(f"\n📊 Strategy 1: Fetch latest 1000 candles (recent batch)")
ohlcv_batch1 = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)

candle_batch1 = None
for c in ohlcv_batch1:
    if c[0] == target_timestamp_ms:
        candle_batch1 = c
        break

if candle_batch1:
    print(f"   Batch 1: Close={candle_batch1[4]:.1f}, Volume={candle_batch1[5]:.4f}")
else:
    print(f"   ❌ Not found in Batch 1")

time.sleep(2)

# Strategy 2: Fetch with 'since' before target (like Batch 2 in backwards collection)
print(f"\n📊 Strategy 2: Fetch from earlier time with 'since' parameter")
# Calculate since time: target - (1000 candles * 5 min * 60 sec * 1000 ms)
since_ms = target_timestamp_ms - (1000 * 5 * 60 * 1000)
ohlcv_batch2 = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)

candle_batch2 = None
for c in ohlcv_batch2:
    if c[0] == target_timestamp_ms:
        candle_batch2 = c
        break

if candle_batch2:
    print(f"   Batch 2: Close={candle_batch2[4]:.1f}, Volume={candle_batch2[5]:.4f}")
else:
    print(f"   ❌ Not found in Batch 2")

# Simulate collect_max_data.py behavior
print(f"\n\n🔄 Simulating collect_max_data.py Logic")
print("=" * 80)

all_candles = []

# Add Batch 1 (recent data)
print(f"Step 1: Extend with Batch 1 ({len(ohlcv_batch1)} candles)")
all_candles.extend(ohlcv_batch1)
print(f"   Total candles: {len(all_candles)}")

# Add Batch 2 (older data, but may overlap)
print(f"\nStep 2: Extend with Batch 2 ({len(ohlcv_batch2)} candles)")
all_candles.extend(ohlcv_batch2)
print(f"   Total candles: {len(all_candles)}")

# Convert to DataFrame
df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

print(f"\nStep 3: Before drop_duplicates")
target_rows = df[df['timestamp'] == target_timestamp_ms]
print(f"   Target candle appears {len(target_rows)} times")
if len(target_rows) > 0:
    for i, row in target_rows.iterrows():
        print(f"   Row {i}: Close={row['close']:.1f}, Volume={row['volume']:.4f}")

# Apply drop_duplicates with keep='last' (like collect_max_data.py line 112)
df = df.drop_duplicates(subset=['timestamp'], keep='last')

print(f"\nStep 4: After drop_duplicates(keep='last')")
target_row = df[df['timestamp'] == target_timestamp_ms]
if len(target_row) > 0:
    print(f"   Kept candle: Close={target_row.iloc[0]['close']:.1f}, Volume={target_row.iloc[0]['volume']:.4f}")

    # Check which batch it came from
    if candle_batch1 and target_row.iloc[0]['close'] == candle_batch1[4]:
        print(f"   ✅ Kept candle from Batch 1 (recent)")
    elif candle_batch2 and target_row.iloc[0]['close'] == candle_batch2[4]:
        print(f"   ⚠️ Kept candle from Batch 2 (older with 'since')")
    else:
        print(f"   ❓ Kept candle doesn't match either batch!")

print(f"\n\n" + "=" * 80)
print("🎯 ANALYSIS")
print("=" * 80)

if candle_batch1 and candle_batch2:
    if candle_batch1[4] == candle_batch2[4] and candle_batch1[5] == candle_batch2[5]:
        print("✅ Both batches returned IDENTICAL data")
        print("   → drop_duplicates behavior doesn't matter")
        print("   → Problem is NOT in the duplicate handling")
    else:
        print("❌ Batches returned DIFFERENT data!")
        print(f"   Batch 1 (recent):  Close={candle_batch1[4]:.1f}, Volume={candle_batch1[5]:.4f}")
        print(f"   Batch 2 (older):   Close={candle_batch2[4]:.1f}, Volume={candle_batch2[5]:.4f}")
        print(f"\n   With keep='last', DataFrame keeps: Batch 2 (added later)")
        print(f"   → This COULD explain the discrepancy if batches differ!")

print("=" * 80)
