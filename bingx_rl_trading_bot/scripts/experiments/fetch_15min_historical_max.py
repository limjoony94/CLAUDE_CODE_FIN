#!/usr/bin/env python3
"""
Fetch Maximum Historical 15-Min Candle Data from BingX
======================================================

Purpose: Collect maximum available historical data using 15-min candles
Method: Multiple API requests with 1440 candles each, going back as far as possible

15-Min Candles vs 5-Min Candles:
  - 15-min: 1 candle per 15 minutes = 96 candles/day
  - 5-min: 1 candle per 5 minutes = 288 candles/day
  - Coverage: 15-min gives 3× longer historical period with same API limits

API Strategy:
  - Request 1440 candles per call (max limit)
  - Each batch = 15 days of data (1440 candles ÷ 96 candles/day)
  - Continue until API returns no more data
  - Expected: ~6-12 months of history (400-800 days)

Date: 2025-11-06
"""

import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime, timedelta
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

# Configuration
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "15m"
LIMIT = 1440  # Maximum candles per request
MAX_REQUESTS = 100  # Safety limit to prevent infinite loops
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_historical_candles_max(client, symbol, timeframe, limit, max_requests=100):
    """
    Fetch maximum historical candles by making multiple API requests

    Args:
        client: BingXClient instance
        symbol: Trading symbol (e.g., "BTC/USDT:USDT")
        timeframe: Candle timeframe (e.g., "15m")
        limit: Candles per request (max 1440)
        max_requests: Maximum API calls to prevent infinite loops

    Returns:
        DataFrame with all historical candles
    """
    all_candles = []
    request_count = 0
    end_time = None  # Start from current time (most recent)

    print(f"\n{'=' * 80}")
    print(f"FETCHING MAXIMUM HISTORICAL DATA")
    print('=' * 80)
    print(f"\nSymbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles per request: {limit}")
    print(f"Max requests: {max_requests}")
    print(f"\nStarting data collection...")

    while request_count < max_requests:
        request_count += 1

        print(f"\n{'─' * 80}")
        print(f"Request #{request_count}")
        print('─' * 80)

        # Fetch candles
        try:
            params = {"limit": limit}
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)

            candles = client.exchange.fetch_ohlcv(symbol, timeframe, params=params)

            if not candles or len(candles) == 0:
                print(f"  ⚠️  No more data available")
                break

            # Convert to DataFrame
            df_batch = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], unit='ms')

            print(f"  ✅ Received {len(df_batch)} candles")
            print(f"     Earliest: {df_batch['timestamp'].min()}")
            print(f"     Latest: {df_batch['timestamp'].max()}")
            print(f"     Days: {(df_batch['timestamp'].max() - df_batch['timestamp'].min()).days}")

            # Check for duplicates
            if all_candles:
                existing_timestamps = pd.concat(all_candles + [df_batch])['timestamp']
                duplicates = existing_timestamps.duplicated().sum()
                if duplicates > 0:
                    print(f"  ⚠️  {duplicates} duplicate candles detected (normal at batch boundaries)")

            all_candles.append(df_batch)

            # Update end_time for next request (go further back)
            oldest_timestamp = df_batch['timestamp'].min()
            end_time = oldest_timestamp - timedelta(milliseconds=1)

            # Check if we received fewer candles than requested (means we hit the limit)
            if len(df_batch) < limit:
                print(f"  ✅ Received {len(df_batch)} < {limit} candles - reached data limit")
                break

            # Rate limiting
            time.sleep(0.2)  # 200ms between requests

        except Exception as e:
            print(f"  ❌ Error: {e}")
            break

    if not all_candles:
        print("\n❌ No data collected!")
        return None

    # Combine all batches
    print(f"\n{'=' * 80}")
    print("COMBINING DATA")
    print('=' * 80)

    df_all = pd.concat(all_candles, ignore_index=True)

    # Remove duplicates (keep first occurrence)
    before_dedup = len(df_all)
    df_all = df_all.drop_duplicates(subset=['timestamp'], keep='first')
    after_dedup = len(df_all)

    if before_dedup > after_dedup:
        print(f"\n  🔧 Removed {before_dedup - after_dedup} duplicate candles")

    # Sort by timestamp
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)

    print(f"\n✅ COLLECTION COMPLETE")
    print(f"   Total Requests: {request_count}")
    print(f"   Total Candles: {len(df_all):,}")
    print(f"   Earliest: {df_all['timestamp'].min()}")
    print(f"   Latest: {df_all['timestamp'].max()}")
    print(f"   Time Span: {(df_all['timestamp'].max() - df_all['timestamp'].min()).days} days")

    return df_all

def main():
    """Main execution"""
    print(f"\n{'=' * 80}")
    print("15-MINUTE CANDLE HISTORICAL DATA COLLECTION")
    print('=' * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load API credentials
    config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize client (use Mainnet for maximum historical data)
    client = BingXClient(
        api_key=config['bingx']['mainnet']['api_key'],
        secret_key=config['bingx']['mainnet']['secret_key'],
        testnet=False  # Use Mainnet for longer history
    )

    # Fetch maximum historical data
    df = fetch_historical_candles_max(
        client=client,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        limit=LIMIT,
        max_requests=MAX_REQUESTS
    )

    if df is None or len(df) == 0:
        print("\n❌ No data to save!")
        return

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    days = (df['timestamp'].max() - df['timestamp'].min()).days
    output_file = OUTPUT_DIR / f"BTCUSDT_15m_raw_{days}days_{timestamp}.csv"

    print(f"\n{'=' * 80}")
    print("SAVING DATA")
    print('=' * 80)

    df.to_csv(output_file, index=False)

    print(f"\n✅ Data saved:")
    print(f"   File: {output_file.name}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Data summary
    print(f"\n📊 DATA SUMMARY")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Days: {days}")
    print(f"   Candles: {len(df):,}")
    print(f"   Candles per Day: {len(df) / max(days, 1):.1f} (expected: 96)")
    print(f"   Price Range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
    print(f"   Avg Price: ${df['close'].mean():,.2f}")

    # Timeframe validation
    time_diffs = df['timestamp'].diff()
    expected_diff = pd.Timedelta(minutes=15)
    gaps = time_diffs[time_diffs > expected_diff * 1.5]

    if len(gaps) > 0:
        print(f"\n⚠️  GAPS DETECTED:")
        print(f"   Count: {len(gaps)}")
        print(f"   Largest gap: {gaps.max()}")
    else:
        print(f"\n✅ NO GAPS - Continuous data")

    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print('=' * 80)
    print(f"\n1. Calculate features using 15-min candles")
    print(f"2. Retrain models with longer historical period")
    print(f"3. Compare 15-min vs 5-min model performance")

    print(f"\n{'=' * 80}")
    print("COLLECTION COMPLETE")
    print('=' * 80)

if __name__ == "__main__":
    main()
