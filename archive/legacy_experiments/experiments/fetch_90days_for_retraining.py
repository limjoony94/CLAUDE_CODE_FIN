"""
Fetch 90 Days (3 Months) Historical Data for Model Retraining
==============================================================

Purpose: Download extended historical data (try 90 days first, then scale up)

Issue: 180-day download returned 0 candles
Solution: Try 90 days to find BingX API's historical depth limit

Data Plan:
  - Period: Aug 7 - Nov 5, 2025 (90 days)
  - Train: Aug 7 - Oct 8 (62 days)
  - Validation: Oct 9 - Nov 5 (28 days / 4 weeks)

Created: 2025-11-05 01:10 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# Configuration
EXCHANGE = 'bingx'
SYMBOL = 'BTC/USDT:USDT'
TIMEFRAME = '5m'
DAYS_TO_FETCH = 90  # 3 months

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("FETCHING 90 DAYS (3 MONTHS) HISTORICAL DATA")
print("="*80)
print()
print(f"📅 Target Period: Last 90 days (~3 months)")
print(f"⏰ Timeframe: {TIMEFRAME}")
print(f"💱 Symbol: {SYMBOL}")
print(f"🔢 Expected Candles: ~{DAYS_TO_FETCH * 24 * 12:,} (5-min candles)")
print()

# Initialize exchange
print("🔗 Connecting to BingX...")
exchange = ccxt.bingx({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    }
})

# Calculate date range
end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS_TO_FETCH)

print(f"📆 Date Range:")
print(f"   Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   End:   {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Days:  {DAYS_TO_FETCH}")
print()

# Fetch data in chunks (BingX limit: 1000 candles per request)
all_candles = []
current_since = int(start_date.timestamp() * 1000)
end_timestamp = int(end_date.timestamp() * 1000)

chunk_num = 0
total_expected_chunks = (DAYS_TO_FETCH * 24 * 12) // 1000 + 1

print(f"📊 Fetching data in chunks (limit: 1000 candles/request)...")
print(f"   Expected chunks: ~{total_expected_chunks}")
print()

while current_since < end_timestamp:
    chunk_num += 1
    try:
        print(f"   Chunk {chunk_num}/{total_expected_chunks}: ", end='', flush=True)

        candles = exchange.fetch_ohlcv(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            since=current_since,
            limit=1000
        )

        if not candles:
            print("No more data")
            break

        all_candles.extend(candles)
        current_since = candles[-1][0] + 1  # Start from next candle

        print(f"{len(candles)} candles (total: {len(all_candles):,})")

        # Rate limiting
        time.sleep(0.5)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Retrying in 5 seconds...")
        time.sleep(5)
        continue

print()
print(f"✅ Fetched {len(all_candles):,} candles")
print()

if len(all_candles) == 0:
    print("❌ ERROR: No candles fetched!")
    print()
    print("Possible reasons:")
    print("  1. BingX doesn't have data going back 90 days for this contract")
    print("  2. Contract launch date is more recent than 90 days ago")
    print("  3. API parameter issue")
    print()
    print("Next steps:")
    print("  - Check BingX contract launch date")
    print("  - Try shorter period (60 days, 30 days)")
    print("  - Use alternative data source (Binance, Bybit)")
    sys.exit(1)

# Convert to DataFrame
print("📊 Converting to DataFrame...")
df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
duplicates_removed = initial_rows - len(df)

if duplicates_removed > 0:
    print(f"   Removed {duplicates_removed} duplicate candles")

print(f"   Final rows: {len(df):,}")
print()

# Data quality check
print("🔍 Data Quality Check:")
print(f"   Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Actual Days: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"   Missing Candles: {DAYS_TO_FETCH * 24 * 12 - len(df):,}")
print(f"   Completeness: {len(df) / (DAYS_TO_FETCH * 24 * 12) * 100:.2f}%")
print()

# Save to CSV
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = DATA_DIR / f"BTCUSDT_5m_raw_90days_{timestamp_str}.csv"

print(f"💾 Saving to: {output_file.name}")
df.to_csv(output_file, index=False)
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# Training/Validation Split Info
train_end_date = df['timestamp'].max() - timedelta(days=28)
train_df = df[df['timestamp'] <= train_end_date]
val_df = df[df['timestamp'] > train_end_date]

print("="*80)
print("TRAIN/VALIDATION SPLIT RECOMMENDATION")
print("="*80)
print()
print(f"📚 Training Set (first {DAYS_TO_FETCH - 28} days):")
print(f"   Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Candles: {len(train_df):,}")
print(f"   Days: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days}")
print()
print(f"✅ Validation Set (last 28 days / 4 weeks):")
print(f"   Period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"   Candles: {len(val_df):,}")
print(f"   Days: {(val_df['timestamp'].max() - val_df['timestamp'].min()).days}")
print()

print("="*80)
print("NEXT STEPS")
print("="*80)
print()
print("1. Calculate features with production_features_v1.py")
print()
print("2. Train models:")
print(f"   → LONG Entry: {len(train_df):,} train candles")
print(f"   → SHORT Entry: {len(train_df):,} train candles")
print(f"   → Both Exit models: {len(train_df):,} train candles")
print()
print("3. Validate on 4-week holdout:")
print(f"   → {len(val_df):,} validation candles")
print()
print("4. Compare vs current models:")
print(f"   → Current LONG: 104 days training")
print(f"   → Current SHORT: 35 days training")
print(f"   → New models: {DAYS_TO_FETCH - 28} days training")
print()
print("✅ Data download complete!")
print(f"   File: {output_file}")
print()
