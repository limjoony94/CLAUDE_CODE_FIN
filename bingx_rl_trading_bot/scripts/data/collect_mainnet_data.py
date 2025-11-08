"""
메인넷 실제 데이터 수집 (Testnet 아님!)

BingX 메인넷에서 실제 거래 데이터를 수집합니다.
OHLCV 데이터는 public이므로 API 키 불필요.
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from loguru import logger

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    logger.info("="*60)
    logger.info("MAINNET DATA COLLECTION")
    logger.info("="*60)
    logger.info("⚠️ Collecting REAL mainnet data (NOT testnet!)")
    logger.info("")

    # BingX 메인넷 연결 (API 키 불필요 - public data)
    exchange = ccxt.bingx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

    # 메인넷 모드 (sandbox 비활성화)
    logger.info("✅ Mainnet mode enabled")

    symbol = 'BTC/USDT:USDT'
    timeframe = '5m'

    # 테스트: 최근 30일부터 시작 (더 안정적)
    days = 100
    logger.info(f"Attempting to collect {days} days of {timeframe} mainnet data")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    since = int(start_date.timestamp() * 1000)

    logger.info(f"Date range: {start_date} → {end_date}")

    all_candles = []
    batch_count = 0
    max_batches = 50  # 최대 50회 요청 (50,000 캔들)

    try:
        while batch_count < max_batches:
            logger.info(f"Batch {batch_count + 1}/{max_batches} - From {datetime.fromtimestamp(since/1000)}")

            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )

            if not candles:
                logger.info("No more data available")
                break

            all_candles.extend(candles)
            batch_count += 1

            logger.info(f"  Fetched {len(candles)} candles (Total: {len(all_candles)})")

            last_timestamp = candles[-1][0]

            if last_timestamp >= int(end_date.timestamp() * 1000):
                logger.info("Reached current time")
                break

            since = last_timestamp + 1
            time.sleep(exchange.rateLimit / 1000)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    if not all_candles:
        logger.error("No data collected!")
        return

    # DataFrame 변환
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 저장 (기존 파일 덮어쓰기)
    save_dir = project_root / 'data' / 'historical'
    filepath = save_dir / 'BTCUSDT_5m_max.csv'

    # 백업 생성
    if filepath.exists():
        backup_path = save_dir / f'BTCUSDT_5m_max_testnet_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        logger.info(f"📦 Backing up old data to: {backup_path.name}")
        import shutil
        shutil.copy(filepath, backup_path)

    df.to_csv(filepath, index=False)

    actual_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400

    logger.info("\n" + "="*60)
    logger.info("MAINNET DATA COLLECTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Source:         BingX MAINNET (Real trading data)")
    logger.info(f"Total Candles:  {len(df):>10,}")
    logger.info(f"Date Range:     {df['timestamp'].min()} → {df['timestamp'].max()}")
    logger.info(f"Actual Days:    {actual_days:>10.2f}")
    logger.info(f"Saved to:       {filepath}")
    logger.info("="*60)
    logger.info("")
    logger.info("✅ Mainnet data is now ready for training!")
    logger.info("⚠️ This data reflects REAL market conditions")


if __name__ == "__main__":
    main()
