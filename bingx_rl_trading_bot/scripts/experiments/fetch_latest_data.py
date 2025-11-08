"""
Fetch latest 5m candle data from BingX to update training dataset
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datetime import datetime
from src.api.bingx_client import BingXClient
import yaml

# Load API keys
config_file = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    api_config = config.get('bingx', {}).get('testnet', {})  # Using same keys as bot

api_key = api_config.get('api_key')
secret_key = api_config.get('secret_key')

# Initialize client (mainnet to get real data)
client = BingXClient(api_key=api_key, secret_key=secret_key, testnet=False)

print("="*80)
print("FETCHING LATEST DATA FROM BINGX")
print("="*80)

# Load existing data
data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df_existing = pd.read_csv(data_file)
print(f"\n✅ Loaded existing data: {len(df_existing):,} candles")
print(f"   Last timestamp: {df_existing['timestamp'].iloc[-1]}")

# Parse last timestamp
last_time = pd.to_datetime(df_existing['timestamp'].iloc[-1])
print(f"   Last time: {last_time}")

# Fetch new data from last timestamp to now
# BingX API max: 1440 candles at once
print(f"\n📥 Fetching new data from {last_time} to now...")

symbol = "BTC/USDT:USDT"
timeframe = "5m"
limit = 1440  # Max allowed

# Fetch data
ohlcv = client.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# Convert to DataFrame
df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')

print(f"✅ Fetched {len(df_new):,} candles")
print(f"   Date range: {df_new['timestamp'].iloc[0]} to {df_new['timestamp'].iloc[-1]}")

# Filter only NEW candles (after last existing timestamp)
df_new = df_new[df_new['timestamp'] > last_time]
print(f"\n📊 New candles after {last_time}: {len(df_new):,}")

if len(df_new) == 0:
    print("⚠️  No new data to add!")
else:
    # Ensure timestamp format consistency
    df_new['timestamp'] = df_new['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Merge with existing data
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Remove duplicates based on timestamp
    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')

    # Convert to datetime for sorting, then back to string
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    df_combined['timestamp'] = df_combined['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n✅ Combined data: {len(df_combined):,} candles")
    print(f"   Date range: {df_combined['timestamp'].iloc[0]} to {df_combined['timestamp'].iloc[-1]}")

    # Save updated data
    output_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
    df_combined.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    print(f"   Total candles: {len(df_combined):,}")
    print(f"   New candles added: {len(df_new):,}")

    # Show price statistics
    print(f"\n📈 Price Statistics (Combined):")
    print(f"   Min: ${df_combined['close'].min():,.2f}")
    print(f"   Max: ${df_combined['close'].max():,.2f}")
    print(f"   Mean: ${df_combined['close'].mean():,.2f}")
    print(f"   Current: ${df_combined['close'].iloc[-1]:,.2f}")

    print(f"\n📈 New Data Price Range:")
    print(f"   Min: ${df_new['close'].min():,.2f}")
    print(f"   Max: ${df_new['close'].max():,.2f}")
    print(f"   Mean: ${df_new['close'].mean():,.2f}")

print("\n" + "="*80)
print("DATA UPDATE COMPLETE")
print("="*80)
