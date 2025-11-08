"""
Historical Data Download Script

목적: BingX API를 사용하여 최신 5분 캔들 데이터를 다운로드
- 기존 데이터가 있으면 마지막 시점부터 이어받기
- 없으면 처음부터 다운로드
- 데이터 검증 및 자동 저장

사용법:
    python scripts/production/download_historical_data.py
    python scripts/production/download_historical_data.py --days 60 --symbol BTC-USDT
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yaml
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"
CONFIG_DIR = PROJECT_ROOT / "config"

# Import BingX client
from src.api.bingx_client import BingXClient

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_api_config():
    """Load API credentials from config"""
    api_keys_file = CONFIG_DIR / "api_keys.yaml"

    if not api_keys_file.exists():
        logger.error(f"API keys file not found: {api_keys_file}")
        logger.error("Please create config/api_keys.yaml with your BingX API credentials")
        sys.exit(1)

    with open(api_keys_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config.get('bingx', {}).get('testnet', {})


def get_existing_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load existing data if available"""
    filename = f"{symbol.replace('-', '')}_{timeframe}_max.csv"
    filepath = DATA_DIR / filename

    if not filepath.exists():
        logger.info(f"No existing data found at {filepath}")
        return None

    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"✅ Loaded existing data: {len(df)} candles")
    logger.info(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    return df


def download_data_range(client: BingXClient, symbol: str, timeframe: str,
                        start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Download data for a specific time range

    BingX API limit: 500 candles per request
    """
    logger.info(f"Downloading data from {start_time} to {end_time}")

    all_data = []
    current_time = start_time

    # Calculate time step based on timeframe
    if timeframe == "5m":
        candle_duration = timedelta(minutes=5)
        max_candles_per_request = 500
    else:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return None

    request_count = 0
    max_requests = 1000  # Safety limit

    while current_time < end_time and request_count < max_requests:
        try:
            # Calculate end time for this batch
            batch_end = min(current_time + candle_duration * max_candles_per_request, end_time)

            # Convert to timestamps (milliseconds)
            start_ts = int(current_time.timestamp() * 1000)
            end_ts = int(batch_end.timestamp() * 1000)

            # Fetch klines
            klines = client.get_klines(
                symbol=symbol,
                interval=timeframe,
                start_time=start_ts,
                end_time=end_ts,
                limit=max_candles_per_request
            )

            if not klines:
                logger.warning(f"No data returned for {current_time}")
                break

            # Convert to DataFrame
            df_batch = pd.DataFrame(klines)
            df_batch = df_batch.rename(columns={'time': 'timestamp'})
            df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], unit='ms')
            df_batch[['open', 'high', 'low', 'close', 'volume']] = \
                df_batch[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df_batch = df_batch[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            all_data.append(df_batch)

            # Update current_time to last candle + 1 interval
            if len(df_batch) > 0:
                last_timestamp = df_batch['timestamp'].max()
                current_time = last_timestamp + candle_duration
                logger.info(f"  Downloaded {len(df_batch)} candles (up to {last_timestamp})")
            else:
                break

            request_count += 1

            # Rate limiting: wait between requests
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            break

    if not all_data:
        logger.error("No data downloaded!")
        return None

    # Combine all batches
    df_combined = pd.concat(all_data, ignore_index=True)

    # Remove duplicates (keep last occurrence)
    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')

    # Sort by timestamp
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    logger.success(f"✅ Downloaded {len(df_combined)} candles total")

    return df_combined


def merge_data(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge old and new data, removing duplicates"""
    if old_df is None:
        return new_df

    if new_df is None or len(new_df) == 0:
        return old_df

    # Combine
    df_combined = pd.concat([old_df, new_df], ignore_index=True)

    # Remove duplicates (keep last)
    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')

    # Sort by timestamp
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"✅ Merged data: {len(old_df)} old + {len(new_df)} new = {len(df_combined)} total")

    return df_combined


def validate_data(df: pd.DataFrame) -> bool:
    """Validate downloaded data"""
    if df is None or len(df) == 0:
        logger.error("Empty dataframe!")
        return False

    # Check required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        return False

    # Check for NaN
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values detected:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"  {col}: {count} NaN ({count/len(df)*100:.2f}%)")

    # Check for gaps
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff().dt.total_seconds() / 60  # minutes
    expected_diff = 5  # 5-minute candles

    gaps = time_diffs[time_diffs > expected_diff * 1.5]  # Allow 50% tolerance
    if len(gaps) > 0:
        logger.warning(f"⚠️ Found {len(gaps)} time gaps in data")
        for idx, gap in gaps.head(5).items():
            logger.warning(f"   Gap at {df_sorted.iloc[idx]['timestamp']}: {gap:.0f} minutes")

    logger.success(f"✅ Data validation passed: {len(df)} valid candles")
    return True


def save_data(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save data to CSV"""
    filename = f"{symbol.replace('-', '')}_{timeframe}_max.csv"
    filepath = DATA_DIR / filename

    # Backup existing file
    if filepath.exists():
        backup_path = filepath.with_suffix('.csv.backup')
        if backup_path.exists():
            backup_path.unlink()  # Remove old backup
        filepath.rename(backup_path)
        logger.info(f"Backed up existing file to {backup_path.name}")

    # Save new data
    df.to_csv(filepath, index=False)

    logger.success(f"✅ Data saved: {filepath}")
    logger.info(f"   Total candles: {len(df)}")
    logger.info(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    logger.info(f"   File size: {filepath.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Download historical data from BingX")
    parser.add_argument('--symbol', type=str, default='BTC-USDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='5m', help='Candle timeframe')
    parser.add_argument('--days', type=int, default=60, help='Number of days to download')
    parser.add_argument('--force', action='store_true', help='Force full download (ignore existing data)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Historical Data Download Script")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Days: {args.days}")
    logger.info("=" * 80)

    # Load API config
    api_config = load_api_config()

    # Initialize BingX client
    client = BingXClient(
        api_key=api_config.get('api_key'),
        secret_key=api_config.get('secret_key'),
        testnet=True,  # Use testnet for data download (same data)
        timeout=30
    )

    # Test connection
    if not client.ping():
        logger.error("Failed to connect to BingX API")
        sys.exit(1)

    logger.success("✅ Connected to BingX API")

    # Get existing data (unless --force)
    existing_df = None if args.force else get_existing_data(args.symbol, args.timeframe)

    # Determine download range
    end_time = datetime.now()

    if existing_df is not None and not args.force:
        # Download only new data since last timestamp
        start_time = existing_df['timestamp'].max() + timedelta(minutes=5)
        logger.info(f"Downloading NEW data from {start_time}")
    else:
        # Download full range
        start_time = end_time - timedelta(days=args.days)
        logger.info(f"Downloading FULL data from {start_time}")

    # Download data
    new_df = download_data_range(client, args.symbol, args.timeframe, start_time, end_time)

    if new_df is None or len(new_df) == 0:
        logger.warning("No new data downloaded")
        if existing_df is not None:
            logger.info("Keeping existing data")
            return
        else:
            logger.error("No data available!")
            sys.exit(1)

    # Merge with existing data
    final_df = merge_data(existing_df, new_df)

    # Validate
    if not validate_data(final_df):
        logger.error("Data validation failed!")
        sys.exit(1)

    # Save
    save_data(final_df, args.symbol, args.timeframe)

    logger.info("=" * 80)
    logger.success("✅ Data download complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
