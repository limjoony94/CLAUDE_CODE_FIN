"""API 연결 테스트 스크립트"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from loguru import logger


def main():
    """메인 함수"""
    # 로거 설정
    setup_logger(log_level='INFO')

    logger.info("="*60)
    logger.info("BingX API Connection Test")
    logger.info("="*60)

    # 설정 로드
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    api_keys = config_loader.load_api_keys()

    # API 키 확인
    testnet_keys = api_keys['bingx']['testnet']
    logger.info(f"API Key: {testnet_keys['api_key'][:20]}...")

    # BingX 클라이언트 초기화
    client = BingXClient(
        api_key=testnet_keys['api_key'],
        secret_key=testnet_keys['secret_key'],
        testnet=config['exchange']['testnet']
    )

    # 1. Ping 테스트
    logger.info("\n[1/5] Testing connection...")
    if client.ping():
        logger.info("✓ Connection successful!")
    else:
        logger.error("✗ Connection failed!")
        return

    # 2. 거래소 정보 조회
    logger.info("\n[2/5] Getting exchange info...")
    try:
        exchange_info = client.get_exchange_info("BTC-USDT")
        logger.info(f"✓ Exchange info retrieved")
        logger.info(f"  Symbol: {exchange_info.get('symbol', 'N/A')}")
    except Exception as e:
        logger.error(f"✗ Failed to get exchange info: {str(e)}")

    # 3. 최신 가격 조회
    logger.info("\n[3/5] Getting latest price...")
    try:
        ticker = client.get_ticker("BTC-USDT")
        price = float(ticker.get('lastPrice', 0))
        logger.info(f"✓ Current BTC price: ${price:,.2f}")
    except Exception as e:
        logger.error(f"✗ Failed to get price: {str(e)}")

    # 4. 계정 잔고 조회 (인증 필요)
    logger.info("\n[4/5] Getting account balance...")
    try:
        balance_info = client.get_balance()
        balance = float(balance_info.get('balance', {}).get('balance', 0))
        logger.info(f"✓ Account balance: ${balance:,.2f} USDT")
    except Exception as e:
        logger.error(f"✗ Failed to get balance: {str(e)}")
        logger.warning("  인증이 필요한 기능입니다. API 키 권한을 확인하세요.")

    # 5. 최근 캔들 데이터 조회
    logger.info("\n[5/5] Getting recent candles...")
    try:
        klines = client.get_klines(
            symbol="BTC-USDT",
            interval="5m",
            limit=5
        )
        logger.info(f"✓ Retrieved {len(klines)} candles")
        if klines:
            latest = klines[-1]
            logger.info(f"  Latest candle:")
            logger.info(f"    Time: {latest.get('time', 'N/A')}")
            logger.info(f"    Open: ${float(latest.get('open', 0)):,.2f}")
            logger.info(f"    High: ${float(latest.get('high', 0)):,.2f}")
            logger.info(f"    Low: ${float(latest.get('low', 0)):,.2f}")
            logger.info(f"    Close: ${float(latest.get('close', 0)):,.2f}")
            logger.info(f"    Volume: {float(latest.get('volume', 0)):,.4f} BTC")
    except Exception as e:
        logger.error(f"✗ Failed to get candles: {str(e)}")

    logger.info("\n" + "="*60)
    logger.info("Connection Test Completed!")
    logger.info("="*60)
    logger.info("\n다음 단계:")
    logger.info("1. python scripts/collect_data.py    # 과거 데이터 수집")
    logger.info("2. python scripts/train.py           # 모델 훈련")
    logger.info("3. python scripts/backtest.py        # 백테스팅")
    logger.info("4. python scripts/live_trade.py --testnet --dry-run  # 모의 거래")


if __name__ == "__main__":
    main()
