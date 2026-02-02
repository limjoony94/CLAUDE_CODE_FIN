"""
Fetch Latest 72 Hours Data from BingX API
==========================================

Fetches the most recent 72 hours (864 candles) of BTC/USDT 5-minute data
for production settings backtest.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

# Load API keys
config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Use MAINNET keys (production data)
api_key = config['bingx']['mainnet']['api_key']
api_secret = config['bingx']['mainnet']['secret_key']

print("="*80)
print("FETCHING LATEST 72 HOURS DATA FROM BINGX")
print("="*80)

# Initialize client
client = BingXClient(api_key=api_key, secret_key=api_secret, testnet=False)

# Fetch 864 candles (72 hours × 12 candles/hour)
print("\nFetching 864 5-minute candles (72 hours)...")
try:
    klines = client.get_klines(
        symbol='BTC-USDT',
        interval='5m',
        limit=864
    )

    # Convert to DataFrame (klines is a list of dicts with keys: time, open, high, low, close, volume)
    df = pd.DataFrame(klines)

    # Rename 'time' to 'timestamp' and convert to datetime
    df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Convert price/volume strings to floats
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"✅ Fetched {len(df):,} candles")
    print(f"   First candle: {df['timestamp'].iloc[0]}")
    print(f"   Last candle: {df['timestamp'].iloc[-1]}")

    # Save to file
    output_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_latest_72h.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")

    print(f"\nData Summary:")
    print(f"   Candles: {len(df)}")
    print(f"   Date Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"   Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

except Exception as e:
    print(f"❌ Error fetching data: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("DATA FETCH COMPLETE")
print("="*80)
