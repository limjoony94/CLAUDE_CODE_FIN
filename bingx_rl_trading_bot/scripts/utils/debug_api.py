"""API 응답 구조 디버깅"""

import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
from src.utils.config_loader import ConfigLoader
from loguru import logger

def main():
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    api_keys = config_loader.load_api_keys()

    testnet_keys = api_keys['bingx']['testnet']
    client = BingXClient(
        api_key=testnet_keys['api_key'],
        secret_key=testnet_keys['secret_key'],
        testnet=config['exchange']['testnet']
    )

    logger.info("Testing klines API response structure...")

    try:
        # 최근 10개 캔들만 가져오기
        klines = client.get_klines(
            symbol="BTC-USDT",
            interval="5m",
            limit=10
        )

        logger.info(f"Response type: {type(klines)}")
        logger.info(f"Response length: {len(klines) if isinstance(klines, list) else 'N/A'}")

        if klines:
            logger.info(f"\nFirst candle structure:")
            logger.info(json.dumps(klines[0], indent=2))

            if len(klines) > 1:
                logger.info(f"\nSecond candle structure:")
                logger.info(json.dumps(klines[1], indent=2))
        else:
            logger.error("Empty response!")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
