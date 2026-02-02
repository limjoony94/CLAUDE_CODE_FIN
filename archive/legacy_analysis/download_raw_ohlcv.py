"""
Download RAW OHLCV Data (No Features)
=====================================
Purpose: Get raw price data only, then apply production feature pipeline
"""

import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

print("=" * 80)
print("DOWNLOAD RAW OHLCV DATA")
print("=" * 80)
print()

# Initialize exchange
exchange = ccxt.bingx({
    'options': {'defaultType': 'swap'}
})

# Download last 7 days (enough for Oct 30)
print("Downloading last 7 days of 5m candles...")
symbol = 'BTC/USDT:USDT'
timeframe = '5m'
since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

all_candles = []
current_since = since

while True:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
        if not ohlcv:
            break

        all_candles.extend(ohlcv)
        print(f"  Downloaded: {len(all_candles)} candles", end='\r')

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
output_file = RESULTS_DIR / "BTCUSDT_5m_RAW_OHLCV_latest4weeks.csv"
df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file.name}")
print()

# Check Oct 30 candles
print("Verify Oct 30 candles:")
oct30_entry = df[df['timestamp'] == '2025-10-30 06:10:00']
oct30_exit = df[df['timestamp'] == '2025-10-30 09:50:00']

if not oct30_entry.empty:
    print(f"  Entry (UTC 06:10 = KST 15:10): ${oct30_entry.iloc[0]['close']:,.2f}")
else:
    print("  ❌ Entry candle not found")

if not oct30_exit.empty:
    print(f"  Exit (UTC 09:50 = KST 18:50): ${oct30_exit.iloc[0]['close']:,.2f}")
else:
    print("  ❌ Exit candle not found")

print()
print("=" * 80)
