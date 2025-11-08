"""데이터 수집 스크립트"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
from src.data.data_collector import DataCollector
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from loguru import logger


def main():
    """메인 함수"""
    # 설정 로드
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    api_keys = config_loader.load_api_keys()

    # 로거 설정
    setup_logger(log_level='INFO')

    logger.info("="*60)
    logger.info("Data Collection Script")
    logger.info("="*60)

    # BingX 클라이언트 초기화
    testnet_keys = api_keys['bingx']['testnet']
    client = BingXClient(
        api_key=testnet_keys['api_key'],
        secret_key=testnet_keys['secret_key'],
        testnet=config['exchange']['testnet']
    )

    # 데이터 수집기 초기화
    data_collector = DataCollector(
        client=client,
        symbol=config['exchange']['symbol'],
        interval=config['trading']['timeframe']
    )

    # 과거 데이터 수집
    days = config['data']['historical_days']
    logger.info(f"Collecting {days} days of historical data...")

    df = data_collector.collect_historical_data(days=days, save=True)

    if not df.empty:
        logger.info(f"Successfully collected {len(df)} candles")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        logger.error("Failed to collect data")

    logger.info("="*60)
    logger.info("Data Collection Finished")
    logger.info("="*60)


if __name__ == "__main__":
    main()
