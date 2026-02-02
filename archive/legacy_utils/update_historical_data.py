"""
Update Historical Data
======================

Fetch latest candles from BingX API and append to historical data file.
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta
import yaml
import pytz

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

DATA_DIR = PROJECT_ROOT / "data" / "historical"
HIST_FILE = DATA_DIR / "BTCUSDT_5m_max.csv"  # Updated to use current file
CONFIG_PATH = PROJECT_ROOT / "config" / "api_keys.yaml"

# Load API keys
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']

print("="*80)
print("UPDATING HISTORICAL DATA")
print("="*80)
print()

# Initialize timezone
kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(kst)
now_utc = now_kst.astimezone(pytz.UTC)

print(f"Current time (KST): {now_kst.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load existing data
print(f"Loading existing data from: {HIST_FILE}")
df_hist = pd.read_csv(HIST_FILE)
df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])

print(f"Existing data: {len(df_hist):,} rows")
print(f"Date range: {df_hist['timestamp'].min()} to {df_hist['timestamp'].max()}")
print()

# Get last timestamp
last_timestamp = df_hist['timestamp'].max()
last_timestamp_kst = last_timestamp.tz_localize('UTC').tz_convert(kst)
print(f"Last timestamp (UTC): {last_timestamp}")
print(f"Last timestamp (KST): {last_timestamp_kst.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate how many candles to fetch (from last timestamp to now)
hours_diff = (now_utc.replace(tzinfo=None) - last_timestamp).total_seconds() / 3600
candles_needed = int(hours_diff * 12) + 20  # 12 candles per hour (5min), +20 buffer
candles_needed = min(candles_needed, 1440)  # BingX API limit

print(f"Time since last candle: {hours_diff:.1f} hours")
print(f"Fetching {candles_needed} candles from BingX...")
print()

# Initialize BingX client
client = BingXClient(
    api_key=api_key,
    secret_key=secret_key,
    testnet=False
)

# Fetch latest candles
try:
    candles = client.get_klines("BTC-USDT", "5m", limit=candles_needed)
    print(f"✅ Fetched {len(candles)} candles from API")
except Exception as e:
    print(f"❌ Error fetching candles: {e}")
    sys.exit(1)

# Convert to DataFrame
# Note: BingX API uses 'time' key, not 'timestamp'
if len(candles) > 0 and isinstance(candles[0], dict):
    df_new = pd.DataFrame(candles)
    # Rename 'time' to 'timestamp' for consistency
    if 'time' in df_new.columns:
        df_new = df_new.rename(columns={'time': 'timestamp'})
else:
    df_new = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Convert timestamp from milliseconds to datetime
df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')

# Convert price columns to float
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

# Filter to only new candles (after last_timestamp)
df_new = df_new[df_new['timestamp'] > last_timestamp].copy()

print(f"New candles: {len(df_new)}")

if len(df_new) > 0:
    print(f"Date range: {df_new['timestamp'].min()} to {df_new['timestamp'].max()}")
    print()

    # Append to historical data
    df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    # Remove duplicates
    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')

    print(f"Combined data: {len(df_combined):,} rows")
    print(f"Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
    print()

    # Create backup before saving
    backup_file = HIST_FILE.parent / f"BTCUSDT_5m_max_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Creating backup: {backup_file}")
    df_hist.to_csv(backup_file, index=False)
    print("✅ Backup created")
    print()

    # Save updated file
    print(f"Saving to: {HIST_FILE}")
    df_combined.to_csv(HIST_FILE, index=False)
    print("✅ Historical data updated successfully!")
    print()
    print(f"📊 Summary:")
    print(f"   Previous: {len(df_hist):,} rows → Latest: {len(df_combined):,} rows")
    print(f"   Added: {len(df_new):,} new candles")
    print(f"   Time coverage: {hours_diff:.1f} hours")
else:
    print("⚠️  No new candles to add (data already up to date)")

print()
print("Update complete!")
