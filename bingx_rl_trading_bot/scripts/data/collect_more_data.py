"""CCXT로 더 많은 데이터 수집"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def main():
    """메인 함수"""
    setup_logger(log_level='INFO')

    logger.info("="*60)
    logger.info("Collecting More Data with CCXT")
    logger.info("="*60)

    # 설정 로드
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    api_keys = config_loader.load_api_keys()

    # CCXT BingX 클라이언트
    testnet_keys = api_keys['bingx']['testnet']

    exchange = ccxt.bingx({
        'apiKey': testnet_keys['api_key'],
        'secret': testnet_keys['secret_key'],
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',  # 선물
        }
    })

    if config['exchange']['testnet']:
        exchange.set_sandbox_mode(True)
        logger.info("Using testnet mode")

    symbol = 'BTC/USDT:USDT'
    timeframe = '5m'

    # 목표: 최대한 많은 데이터 수집
    days = 30  # 30일치 시도

    logger.info(f"Collecting {days} days of {timeframe} data for {symbol}")

    # 종료 시간 (현재)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    since = int(start_date.timestamp() * 1000)
    all_candles = []

    try:
        while True:
            logger.info(f"Fetching from {datetime.fromtimestamp(since/1000)}")

            # OHLCV 데이터 가져오기
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
            logger.info(f"Fetched {len(candles)} candles (Total: {len(all_candles)})")

            # 마지막 캔들의 타임스탬프
            last_timestamp = candles[-1][0]

            # 이미 최신 데이터까지 수집했으면 종료
            if last_timestamp >= int(end_date.timestamp() * 1000):
                break

            # 다음 요청 시작점
            since = last_timestamp + 1

            time.sleep(exchange.rateLimit / 1000)  # Rate limit 준수

            # 최대 3000개까지만 (메모리 고려)
            if len(all_candles) >= 3000:
                logger.info("Reached maximum candles (3000)")
                break

    except Exception as e:
        logger.error(f"Error collecting data: {str(e)}")

    if not all_candles:
        logger.error("No data collected!")
        return

    # DataFrame 변환
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 저장
    save_dir = project_root / 'data' / 'historical'
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / 'BTCUSDT_5m_extended.csv'

    df.to_csv(filepath, index=False)

    logger.info(f"\nData collected successfully!")
    logger.info(f"Total candles: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Actual days: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400:.2f}")
    logger.info(f"Saved to: {filepath}")

    logger.info("="*60)


if __name__ == "__main__":
    main()
