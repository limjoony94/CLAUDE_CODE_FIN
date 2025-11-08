"""공개 데이터 수집 (메인넷, API 키 불필요)"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger


def main():
    setup_logger(log_level='INFO')

    logger.info("="*60)
    logger.info("PUBLIC DATA COLLECTION (No API Key Required)")
    logger.info("="*60)

    # 공개 메인넷 연결 (API 키 없이)
    exchange = ccxt.bingx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

    symbol = 'BTC/USDT:USDT'
    timeframe = '5m'

    # 목표: 60일치 (최대한 많이)
    days = 60
    logger.info(f"Collecting {days} days of {timeframe} data for {symbol}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    since = int(start_date.timestamp() * 1000)

    all_candles = []
    batch_count = 0
    max_batches = 100  # 최대 100회 (100,000 캔들 = 약 347일)

    logger.info(f"Start: {start_date}")
    logger.info(f"End: {end_date}")

    try:
        while batch_count < max_batches:
            logger.info(f"Batch {batch_count + 1}/{max_batches} - {datetime.fromtimestamp(since/1000)}")

            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )

            if not candles:
                logger.info("No more data")
                break

            all_candles.extend(candles)
            logger.info(f"  Fetched: {len(candles)}, Total: {len(all_candles)}")

            last_timestamp = candles[-1][0]

            if last_timestamp >= int(end_date.timestamp() * 1000):
                logger.info("Reached current time")
                break

            since = last_timestamp + 1
            batch_count += 1

            time.sleep(exchange.rateLimit / 1000)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    if not all_candles:
        logger.error("Failed to collect data!")
        return

    # DataFrame 변환
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 저장
    save_dir = project_root / 'data' / 'historical'
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / 'BTCUSDT_5m_max.csv'
    df.to_csv(filepath, index=False)

    actual_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400

    logger.info("\n" + "="*60)
    logger.info("SUCCESS!")
    logger.info("="*60)
    logger.info(f"Total Candles:  {len(df):>10,}")
    logger.info(f"Date Range:     {df['timestamp'].min()}")
    logger.info(f"           to   {df['timestamp'].max()}")
    logger.info(f"Actual Days:    {actual_days:>10.2f}")
    logger.info(f"File Size:      {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"Saved to:       {filepath}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
