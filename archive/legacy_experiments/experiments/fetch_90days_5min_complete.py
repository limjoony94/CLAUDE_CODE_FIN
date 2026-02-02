"""
Fetch 90-Day 5-Minute Historical Data for Complete Model Training
==================================================================

Purpose: Fetch 90 days of 5-min candle data from BingX Mainnet
         Compromise between 52 days (current) and 314 days (failed)

Advantages over 314-day 15-min:
- Proven 5-min timeframe (preserves critical timing)
- More training data than 52 days (1.7x more)
- Avoids 15-min aggregation loss

Output: BTCUSDT_5m_raw_90days_{timestamp}.csv

Created: 2025-11-06 16:40 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
import ccxt
from datetime import datetime, timedelta

# Configuration
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "5m"
LIMIT = 1440  # Maximum candles per request
TARGET_DAYS = 90  # Target days to fetch
MAX_REQUESTS = 100  # Safety limit

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FETCHING 90-DAY 5-MINUTE HISTORICAL DATA")
print("=" * 80)
print()
print(f"Symbol: {SYMBOL}")
print(f"Timeframe: {TIMEFRAME}")
print(f"Target: {TARGET_DAYS} days")
print(f"Max per request: {LIMIT} candles")
print()

# Load API keys
with open(PROJECT_ROOT / 'config' / 'api_keys.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize CCXT client (Mainnet for maximum historical data)
exchange = ccxt.bingx({
    'apiKey': config['bingx']['mainnet']['api_key'],
    'secret': config['bingx']['mainnet']['secret_key'],
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'  # Use perpetual futures
    }
})

print(f"🔗 Connected to BingX Mainnet via CCXT")
print()

# Calculate target candles
target_candles = TARGET_DAYS * 24 * 60 // 5  # 90 days * 288 candles/day = 25,920
print(f"Target candles: {target_candles:,} ({TARGET_DAYS} days @ 5-min)")
print()

# Fetch data with pagination
all_candles = []
request_count = 0
last_timestamp = None

print("📥 Fetching historical data...")
print()

while request_count < MAX_REQUESTS:
    request_count += 1

    # Prepare parameters
    params = {'limit': LIMIT}
    if last_timestamp:
        # Fetch older data (endTime = last_timestamp - 1ms)
        params['endTime'] = last_timestamp - 1

    # Fetch
    candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, params=params)

    if not candles:
        print(f"   Request {request_count}: No more data available")
        break

    # Convert to DataFrame for easier manipulation
    df_batch = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], unit='ms')

    # Check for complete duplicates (all timestamps already exist)
    if all_candles:
        existing_df = pd.concat(all_candles, ignore_index=True)
        existing_timestamps = set(existing_df['timestamp'])
        new_timestamps = set(df_batch['timestamp'])

        # If all new timestamps already exist, we've reached the limit
        if new_timestamps.issubset(existing_timestamps):
            print(f"   Request {request_count}: All data already exists, stopping")
            break

    # Add to collection
    all_candles.append(df_batch)

    # Update last timestamp
    last_timestamp = candles[0][0]  # Earliest timestamp in this batch

    # Progress
    total_candles = sum(len(df) for df in all_candles)
    oldest_time = df_batch['timestamp'].min()
    newest_time = all_candles[0]['timestamp'].max()
    days_collected = (newest_time - oldest_time).days

    print(f"   Request {request_count}: {len(df_batch)} candles")
    print(f"      Total: {total_candles:,} candles ({days_collected} days)")
    print(f"      Range: {oldest_time} to {newest_time}")
    print()

    # Check if we have enough
    if total_candles >= target_candles:
        print(f"✅ Target reached: {total_candles:,} candles ({days_collected} days)")
        break

# Combine all data
print("=" * 80)
print("COMBINING DATA")
print("=" * 80)
print()

df_all = pd.concat(all_candles, ignore_index=True)
df_all = df_all.sort_values('timestamp').reset_index(drop=True)

# Remove duplicates
before_dedup = len(df_all)
df_all = df_all.drop_duplicates(subset='timestamp', keep='first')
after_dedup = len(df_all)

if before_dedup != after_dedup:
    print(f"⚠️  Removed {before_dedup - after_dedup} duplicate candles")

print(f"Total candles: {len(df_all):,}")
print(f"Period: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
print(f"Days: {(df_all['timestamp'].max() - df_all['timestamp'].min()).days}")
print(f"Price range: ${df_all['low'].min():,.2f} - ${df_all['high'].max():,.2f}")
print(f"Candles per day: {len(df_all) / ((df_all['timestamp'].max() - df_all['timestamp'].min()).days):.1f}")
print()

# Save
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = DATA_DIR / f"BTCUSDT_5m_raw_90days_{timestamp_str}.csv"

df_all.to_csv(output_file, index=False)

print("=" * 80)
print("DATA COLLECTION COMPLETE")
print("=" * 80)
print()
print(f"✅ Saved: {output_file.name}")
print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
print(f"   Rows: {len(df_all):,}")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"📊 Collection Statistics:")
print(f"   API Requests: {request_count}")
print(f"   Candles: {len(df_all):,}")
print(f"   Days: {(df_all['timestamp'].max() - df_all['timestamp'].min()).days}")
print(f"   Timeframe: 5-minute")
print()
print("📈 Advantages vs 52-Day:")
print(f"   Data: {len(df_all):,} vs 15,003 candles (1.7× more)")
print(f"   Period: 90 days vs 52 days (73% increase)")
print()
print("✅ Advantages vs 314-Day 15-Min:")
print("   Timeframe: 5-min (proven) vs 15-min (failed)")
print("   Signal preservation: Critical timing info preserved")
print("   Calibration: 5-min models have proven calibration")
print()
print("📁 Next Steps:")
print("1. Calculate all features (entry + exit) on this data")
print("2. Generate entry and exit labels")
print("3. Train 4 models with Enhanced 5-Fold CV")
print("4. Validate on out-of-sample period")
print()
print("✅ Data collection complete!")
print(f"   File: {output_file}")
print()
