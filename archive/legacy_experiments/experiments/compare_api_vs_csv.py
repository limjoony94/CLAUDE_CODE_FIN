"""
Compare BingX API data vs CSV file data for same timestamps
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import ccxt
import yaml
from datetime import datetime

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data" / "historical"

print("="*80)
print("BingX API vs CSV DATA COMPARISON")
print("="*80)

# Load API keys
with open(CONFIG_DIR / "api_keys.yaml", 'r') as f:
    config = yaml.safe_load(f)
    testnet_config = config['bingx']['testnet']

# Initialize BingX client
client = ccxt.bingx({
    'apiKey': testnet_config['api_key'],
    'secret': testnet_config['secret_key'],
    'options': {'defaultType': 'swap'}
})
client.set_sandbox_mode(True)

# Load CSV
csv_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_csv = pd.read_csv(csv_file)
print(f"\n📁 CSV file: {csv_file.name}")
print(f"  Total candles: {len(df_csv):,}")
print(f"  Date range: {df_csv['timestamp'].iloc[0]} to {df_csv['timestamp'].iloc[-1]}")

# Target timestamp (from production log)
target_timestamp = "2025-10-18 20:40:00"
target_dt = datetime.strptime(target_timestamp, "%Y-%m-%d %H:%M:%S")

print(f"\n🎯 Target: {target_timestamp}")

# Get from CSV
csv_row = df_csv[df_csv['timestamp'] == target_timestamp]
if len(csv_row) == 0:
    print(f"❌ Not found in CSV")
    sys.exit(1)

print(f"\n📊 CSV Data:")
print(f"  Open: ${csv_row['open'].iloc[0]:,.2f}")
print(f"  High: ${csv_row['high'].iloc[0]:,.2f}")
print(f"  Low: ${csv_row['low'].iloc[0]:,.2f}")
print(f"  Close: ${csv_row['close'].iloc[0]:,.2f}")
print(f"  Volume: {csv_row['volume'].iloc[0]:,.4f}")

# Fetch from BingX API (get broader window)
print(f"\n🌐 Fetching from BingX API...")
since = int(target_dt.timestamp() * 1000) - (5 * 60 * 1000 * 10)  # 10 candles before
ohlcv = client.fetch_ohlcv('BTC/USDT:USDT', '5m', since=since, limit=100)

# Convert to DataFrame
df_api = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_api['timestamp'] = pd.to_datetime(df_api['timestamp'], unit='ms')
df_api['timestamp'] = df_api['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

print(f"  Fetched {len(df_api)} candles")
print(f"  Range: {df_api['timestamp'].iloc[0]} to {df_api['timestamp'].iloc[-1]}")

# Find target in API data
api_row = df_api[df_api['timestamp'] == target_timestamp]

if len(api_row) == 0:
    print(f"\n⚠️  Target timestamp NOT found in API data")
    print(f"  Available timestamps around target:")
    for ts in df_api['timestamp'].values[-5:]:
        print(f"    {ts}")
else:
    print(f"\n📊 API Data:")
    print(f"  Open: ${api_row['open'].iloc[0]:,.2f}")
    print(f"  High: ${api_row['high'].iloc[0]:,.2f}")
    print(f"  Low: ${api_row['low'].iloc[0]:,.2f}")
    print(f"  Close: ${api_row['close'].iloc[0]:,.2f}")
    print(f"  Volume: {api_row['volume'].iloc[0]:,.4f}")

    print(f"\n🔍 COMPARISON:")
    print(f"{'Field':<12} {'CSV':<15} {'API':<15} {'Diff':<15} {'Match':<10}")
    print("-"*70)

    fields = ['open', 'high', 'low', 'close', 'volume']
    all_match = True

    for field in fields:
        csv_val = csv_row[field].iloc[0]
        api_val = api_row[field].iloc[0]
        diff = api_val - csv_val
        match = "✅" if abs(diff) < 0.01 else "❌"

        if abs(diff) >= 0.01:
            all_match = False

        print(f"{field:<12} ${csv_val:<14,.2f} ${api_val:<14,.2f} ${diff:<14,.2f} {match:<10}")

    if all_match:
        print(f"\n✅ CSV and API data MATCH for {target_timestamp}")
    else:
        print(f"\n❌ CSV and API data DIFFER for {target_timestamp}")
        print(f"  → This explains the 1.47% prediction probability difference!")
        print(f"  → BingX API returns different data than CSV file")

# Check a few more candles around target
print(f"\n📊 Checking surrounding candles...")
for offset in [-2, -1, 0, 1, 2]:
    idx = df_csv[df_csv['timestamp'] == target_timestamp].index[0] + offset
    if idx < 0 or idx >= len(df_csv):
        continue

    ts = df_csv.iloc[idx]['timestamp']
    csv_close = df_csv.iloc[idx]['close']

    api_match = df_api[df_api['timestamp'] == ts]
    if len(api_match) > 0:
        api_close = api_match['close'].iloc[0]
        diff = api_close - csv_close
        match = "✅" if abs(diff) < 0.01 else "❌"
        print(f"  {ts}: CSV ${csv_close:,.2f} | API ${api_close:,.2f} | Diff ${diff:,.2f} {match}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
