"""
Fetch 35 days of 5m candle data from BingX for model retraining
Includes Nov 2025 falling market period
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datetime import datetime, timedelta
from src.api.bingx_client import BingXClient
import yaml
import time

# Load API keys
config_file = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    api_config = config.get('bingx', {}).get('testnet', {})

api_key = api_config.get('api_key')
secret_key = api_config.get('secret_key')

# Initialize client (mainnet to get real data)
client = BingXClient(api_key=api_key, secret_key=secret_key, testnet=False)

print("="*80)
print("FETCHING 35 DAYS OF DATA FROM BINGX")
print("="*80)

symbol = "BTC/USDT:USDT"
timeframe = "5m"
days_to_fetch = 35

# Calculate target number of candles
# 35 days * 24 hours * 60 minutes / 5 minutes = 10,080 candles
target_candles = days_to_fetch * 24 * 60 // 5
print(f"\n🎯 Target: {target_candles:,} candles ({days_to_fetch} days)")

# BingX API max: 1440 candles per request
max_limit = 1440
num_requests = (target_candles + max_limit - 1) // max_limit
print(f"   Requests needed: {num_requests} (limit {max_limit} per request)")

# Fetch data in chunks
all_data = []
end_time = None

for i in range(num_requests):
    print(f"\n📥 Request {i+1}/{num_requests}...")

    try:
        # Fetch with since parameter for historical data
        params = {'limit': max_limit}
        if end_time:
            params['until'] = int(end_time.timestamp() * 1000)

        ohlcv = client.exchange.fetch_ohlcv(symbol, timeframe=timeframe, params=params, limit=max_limit)

        if not ohlcv:
            print(f"   ⚠️  No data returned")
            break

        # Convert to DataFrame
        df_chunk = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')

        print(f"   ✅ Fetched {len(df_chunk):,} candles")
        print(f"      Range: {df_chunk['timestamp'].iloc[0]} to {df_chunk['timestamp'].iloc[-1]}")

        all_data.append(df_chunk)

        # Update end_time for next request (fetch older data)
        end_time = df_chunk['timestamp'].iloc[0]

        # Stop if we have enough data
        total_so_far = sum(len(df) for df in all_data)
        if total_so_far >= target_candles:
            print(f"   ✅ Reached target: {total_so_far:,} candles")
            break

        # Rate limit: wait 0.5s between requests
        if i < num_requests - 1:
            time.sleep(0.5)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        break

# Combine all chunks
if all_data:
    df_combined = pd.concat(all_data, ignore_index=True)

    # Remove duplicates and sort
    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    print(f"\n✅ Total fetched: {len(df_combined):,} candles")
    print(f"   Date range: {df_combined['timestamp'].iloc[0]} to {df_combined['timestamp'].iloc[-1]}")

    # Calculate actual days covered
    days_covered = (df_combined['timestamp'].iloc[-1] - df_combined['timestamp'].iloc[0]).total_seconds() / 86400
    print(f"   Days covered: {days_covered:.1f}")

    # Convert timestamp to string format
    df_combined['timestamp'] = df_combined['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = PROJECT_ROOT / "data" / "features" / f"BTCUSDT_5m_raw_35days_{timestamp}.csv"
    df_combined.to_csv(output_file, index=False)

    print(f"\n💾 Saved to: {output_file}")
    print(f"   Total candles: {len(df_combined):,}")

    # Show price statistics
    df_combined['close'] = df_combined['close'].astype(float)
    print(f"\n📈 Price Statistics:")
    print(f"   Min: ${df_combined['close'].min():,.2f}")
    print(f"   Max: ${df_combined['close'].max():,.2f}")
    print(f"   Mean: ${df_combined['close'].mean():,.2f}")
    print(f"   Current: ${df_combined['close'].iloc[-1]:,.2f}")

    # Check if we have Nov 2025 data
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    nov_data = df_combined[df_combined['timestamp'] >= '2025-11-01']
    print(f"\n📊 Nov 2025 Coverage:")
    print(f"   Nov candles: {len(nov_data):,}")
    if len(nov_data) > 0:
        print(f"   Nov range: {nov_data['timestamp'].iloc[0]} to {nov_data['timestamp'].iloc[-1]}")
        print(f"   ✅ Includes falling market period (Nov 3-4)")
    else:
        print(f"   ⚠️  No Nov 2025 data included!")

else:
    print("\n❌ No data fetched!")

print("\n" + "="*80)
print("DATA FETCH COMPLETE")
print("="*80)
