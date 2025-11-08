"""
Download Latest 4-Week Data from BingX
======================================
User Request: "최신 4주로 새로 받아서 백테스트를 진행해 주세요"

Downloads fresh 4-week BTCUSDT 5-minute data for current backtest validation
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datetime import datetime, timedelta
import time
import yaml

from src.api.bingx_client import BingXClient

# Configuration
SYMBOL = "BTC-USDT"
INTERVAL = "5m"
WEEKS = 4
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
CONFIG_FILE = PROJECT_ROOT / "config" / "api_keys.yaml"

print("=" * 80)
print("DOWNLOADING LATEST 4-WEEK DATA FROM BINGX")
print("=" * 80)
print(f"Symbol: {SYMBOL}")
print(f"Interval: {INTERVAL}")
print(f"Period: Last {WEEKS} weeks")
print()

# Load API keys
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

# Initialize BingX client (use mainnet for latest real data)
mainnet_config = config['bingx']['mainnet']
client = BingXClient(
    api_key=mainnet_config['api_key'],
    secret_key=mainnet_config['secret_key']
)

# Calculate time range (4 weeks = 28 days)
end_time = datetime.now()
start_time = end_time - timedelta(days=WEEKS * 7)

print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Download data
print("Downloading data from BingX...")
all_candles = []
current_start = start_time

while current_start < end_time:
    # BingX limit: 1440 candles per request
    candles = client.get_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        start_time=int(current_start.timestamp() * 1000),
        limit=1440
    )

    if not candles:
        break

    all_candles.extend(candles)

    # Update start time for next batch
    last_candle_time = datetime.fromtimestamp(candles[-1]['time'] / 1000)
    current_start = last_candle_time + timedelta(minutes=5)

    print(f"  Downloaded {len(candles)} candles up to {last_candle_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Rate limiting
    time.sleep(0.2)

print()
print(f"✓ Total candles downloaded: {len(all_candles):,}")

# Convert to DataFrame
df = pd.DataFrame(all_candles)

# Rename 'time' to 'timestamp' for consistency
if 'time' in df.columns:
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    df = df.drop('time', axis=1)
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✓ Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"✓ Total duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print()

# Save raw data
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"BTCUSDT_5m_raw_latest4weeks_{timestamp_str}.csv"
df.to_csv(output_file, index=False)

print(f"✓ Saved raw data: {output_file.name}")
print()

# Display data info
print("Data Preview:")
print(df.head(10))
print()
print(df.info())
print()

print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print(f"Next Step: Calculate features using calculate_all_features.py")
print(f"File: {output_file}")
print("=" * 80)
