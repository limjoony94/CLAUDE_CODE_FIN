"""
Download Recent Data from BingX API
===================================
Download Oct 20-21 data to test signal frequency
"""
import pandas as pd
import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data" / "historical"

print("=" * 80)
print("📥 Downloading Recent Data (Oct 17-21)")
print("=" * 80)

# Load API config
config_path = CONFIG_DIR / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize client (MAINNET)
client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key'],
    testnet=False
)

# Fetch maximum available data (1440 candles = 5 days)
print("\n📊 Fetching data from BingX API (MAINNET)...")
print("   Requesting 1440 candles (5 days of 5m data)")

live_data = client.get_klines(
    symbol='BTC-USDT',
    interval='5m',
    limit=1440
)

# Convert to DataFrame
df = pd.DataFrame(live_data)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)
df['volume'] = df['volume'].astype(float)

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Select only necessary columns
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

print(f"\n✅ Downloaded {len(df)} candles")
print(f"   Start: {df['timestamp'].min()}")
print(f"   End:   {df['timestamp'].max()}")
print(f"   Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")

# Save to CSV
output_path = DATA_DIR / "BTCUSDT_5m_recent.csv"
df.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")

print("\n" + "=" * 80)
print("✅ DOWNLOAD COMPLETE")
print("=" * 80)
