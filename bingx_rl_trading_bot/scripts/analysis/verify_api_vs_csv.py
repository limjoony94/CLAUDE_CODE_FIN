#!/usr/bin/env python3
"""
Direct API vs CSV Data Comparison
==================================
Fetch the exact same 5 candles from BingX API and compare with CSV data
to determine if data source differences explain signal mismatches.
"""

import pandas as pd
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

print("=" * 80)
print("DIRECT API vs CSV DATA COMPARISON")
print("=" * 80)

# Test timestamps (UTC)
test_times = [
    "2025-10-26 15:05:00",
    "2025-10-26 15:10:00",
    "2025-10-26 15:15:00",
    "2025-10-26 15:20:00",
    "2025-10-26 15:25:00"
]

# 1. Fetch from BingX API
print("\n1. Fetching data from BingX API (mainnet)...")
CONFIG_FILE = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key'],
    testnet=False
)

# Need enough data for feature calculation (1000+ candles)
start_time = pd.to_datetime("2025-10-23 04:00:00")
end_time = pd.to_datetime("2025-10-26 15:30:00")

start_ms = int(start_time.timestamp() * 1000)
end_ms = int(end_time.timestamp() * 1000)

print(f"   Requesting: {start_time} ~ {end_time} UTC")
api_data = client.get_klines(
    symbol="BTC-USDT",
    interval="5m",
    start_time=start_ms,
    end_time=end_ms,
    limit=1440  # BingX API max limit
)

# Convert to DataFrame
df_api = pd.DataFrame(api_data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
df_api['timestamp'] = pd.to_datetime(df_api['time'], unit='ms', utc=True).dt.tz_localize(None)
df_api[['open', 'high', 'low', 'close', 'volume']] = df_api[['open', 'high', 'low', 'close', 'volume']].astype(float)
df_api = df_api.drop('time', axis=1)

print(f"   ✅ Fetched {len(df_api)} candles")
print(f"   Period: {df_api['timestamp'].min()} ~ {df_api['timestamp'].max()}")

# 2. Load CSV
print("\n2. Loading CSV data...")
CSV_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df_csv = pd.read_csv(CSV_FILE)
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])

# Filter to same period
df_csv = df_csv[
    (df_csv['timestamp'] >= df_api['timestamp'].min()) &
    (df_csv['timestamp'] <= df_api['timestamp'].max())
].copy()

print(f"   ✅ Loaded {len(df_csv)} candles (filtered)")
print(f"   Period: {df_csv['timestamp'].min()} ~ {df_csv['timestamp'].max()}")

# 3. Compare test candles
print("\n3. Comparing test candles (OHLCV)...")
print("=" * 140)
print(f"{'Timestamp':<20} {'Source':<6} {'Open':>12} {'High':>12} {'Low':>12} {'Close':>12} {'Volume':>15} {'Match':<10}")
print("=" * 140)

mismatches = []

for ts_str in test_times:
    ts = pd.to_datetime(ts_str)

    api_row = df_api[df_api['timestamp'] == ts]
    csv_row = df_csv[df_csv['timestamp'] == ts]

    if len(api_row) > 0 and len(csv_row) > 0:
        api_row = api_row.iloc[0]
        csv_row = csv_row.iloc[0]

        print(f"{ts_str:<20} {'API':<6} {api_row['open']:>12.2f} {api_row['high']:>12.2f} {api_row['low']:>12.2f} {api_row['close']:>12.2f} {api_row['volume']:>15.4f}")
        print(f"{'':<20} {'CSV':<6} {csv_row['open']:>12.2f} {csv_row['high']:>12.2f} {csv_row['low']:>12.2f} {csv_row['close']:>12.2f} {csv_row['volume']:>15.4f}")

        # Calculate differences
        close_diff = api_row['close'] - csv_row['close']
        high_diff = api_row['high'] - csv_row['high']
        low_diff = api_row['low'] - csv_row['low']
        volume_diff = api_row['volume'] - csv_row['volume']

        # Check if data matches (within 0.01 for prices)
        match = (abs(close_diff) < 0.01 and
                abs(high_diff) < 0.01 and
                abs(low_diff) < 0.01)

        match_status = "✅ MATCH" if match else f"❌ DIFF (close: {close_diff:+.2f})"

        print(f"{'':<20} {'DIFF':<6} {'':<12} {high_diff:>+12.2f} {low_diff:>+12.2f} {close_diff:>+12.2f} {volume_diff:>+15.4f} {match_status:<10}")
        print("-" * 140)

        if not match:
            mismatches.append({
                'timestamp': ts_str,
                'close_diff': close_diff,
                'api_close': api_row['close'],
                'csv_close': csv_row['close']
            })
    else:
        status = "NOT FOUND IN API" if len(api_row) == 0 else "NOT FOUND IN CSV"
        print(f"{ts_str:<20} {status}")
        print("-" * 140)

# 4. Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total = len(test_times)
matches = total - len(mismatches)

print(f"\nTotal test candles: {total}")
print(f"Matches: {matches}/{total} ({matches/total*100:.1f}%)")
print(f"Mismatches: {len(mismatches)}/{total} ({len(mismatches)/total*100:.1f}%)")

if len(mismatches) > 0:
    print(f"\n⚠️ DATA MISMATCHES FOUND:")
    print(f"\nTimestamp              API Close    CSV Close      Difference")
    print("-" * 70)
    for m in mismatches:
        print(f"{m['timestamp']:<20} {m['api_close']:>12.2f} {m['csv_close']:>12.2f} {m['close_diff']:>+12.2f}")

    print(f"\n🔍 ANALYSIS:")
    print(f"   - API data represents current/latest state from exchange")
    print(f"   - CSV data was captured at: {CSV_FILE.stat().st_mtime}")
    print(f"   - Differences suggest:")
    print(f"     1. CSV was updated while candles were still forming, OR")
    print(f"     2. BingX revised historical data after CSV was captured, OR")
    print(f"     3. Production bot fetched data at different completion state")

    print(f"\n✅ EXPLANATION FOR SIGNAL MISMATCHES:")
    print(f"   - Different underlying OHLCV data → different features → different signals")
    print(f"   - This is EXPECTED when comparing CSV snapshot vs real-time API data")
    print(f"   - Production signals are CORRECT for the data it receives")
    print(f"   - Backtest signals reflect CSV snapshot, not current API state")

else:
    print(f"\n✅ ALL DATA MATCHES PERFECTLY!")
    print(f"\n🔍 If signals still don't match, investigate:")
    print(f"   1. Feature calculation differences")
    print(f"   2. Model loading or preprocessing")
    print(f"   3. Numerical precision issues")

print("\n" + "=" * 80)
