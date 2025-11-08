"""
Download RAW OHLCV Data - 60 Days
==================================
Purpose: Get 60 days of raw price data for comprehensive backtest
"""

import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

print("=" * 80)
print("DOWNLOAD RAW OHLCV DATA - 60 DAYS")
print("=" * 80)
print()

# Initialize exchange
exchange = ccxt.bingx({
    'options': {'defaultType': 'swap'}
})

# Download last 60 days (enough for comprehensive backtest)
print("Downloading last 60 days of 5m candles...")
symbol = 'BTC/USDT:USDT'
timeframe = '5m'
since = int((datetime.now() - timedelta(days=60)).timestamp() * 1000)

all_candles = []
current_since = since

batch_count = 0
while True:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
        if not ohlcv:
            break

        all_candles.extend(ohlcv)
        batch_count += 1
        print(f"  Batch {batch_count}: Downloaded {len(all_candles)} candles total", end='\r')

        if len(ohlcv) < 1000:
            break

        current_since = ohlcv[-1][0] + 1

    except Exception as e:
        print(f"\n  Error: {e}")
        break

print(f"\n✓ Total downloaded: {len(all_candles)} candles")
print()

# Convert to DataFrame
df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Remove duplicates
df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

print(f"After deduplication: {len(df)} candles")
print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

# Save
output_file = RESULTS_DIR / "BTCUSDT_5m_RAW_OHLCV_60days.csv"
df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file.name}")
print()

# Statistics
days_covered = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
expected_candles = days_covered * 24 * 12  # 5min candles per day
completeness = len(df) / expected_candles * 100

print(f"Data Statistics:")
print(f"  Days covered: {days_covered:.1f}")
print(f"  Expected candles: {expected_candles:.0f}")
print(f"  Actual candles: {len(df)}")
print(f"  Completeness: {completeness:.1f}%")
print()

print("=" * 80)
