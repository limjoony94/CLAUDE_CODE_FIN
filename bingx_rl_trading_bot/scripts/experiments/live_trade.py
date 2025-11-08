"""실시간 거래 스크립트"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
from src.agent.rl_agent import RLAgent
from src.environment.trading_env import TradingEnvironment
from src.trading.live_trader import LiveTrader
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from loguru import logger


def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='BingX Live Trading Bot')
    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Use testnet (default: False)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Dry run mode (default: True)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='best_model',
        help='Model name to load (default: best_model)'
    )

    args = parser.parse_args()

    # 설정 로드
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    api_keys = config_loader.load_api_keys()

    # testnet 설정 오버라이드
    if args.testnet:
        config['exchange']['testnet'] = True

    # 로거 설정
    setup_logger(
        log_level=config['logging']['level'],
        log_file=str(project_root / 'data' / 'logs' / 'live_trading.log')
    )

    logger.info("="*60)
    logger.info("Starting Live Trading Bot")
    logger.info("="*60)
    logger.info(f"Mode: {'TESTNET' if config['exchange']['testnet'] else 'MAINNET'}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("="*60)

    if not config['exchange']['testnet'] and not args.dry_run:
        logger.warning("⚠️  REAL MONEY TRADING MODE ⚠️")
        response = input("Are you sure you want to trade with real money? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Trading cancelled by user")
            return

    # API 키 선택
    if config['exchange']['testnet']:
        keys = api_keys['bingx']['testnet']
    else:
        keys = api_keys['bingx']['mainnet']

    # BingX 클라이언트 초기화
    client = BingXClient(
        api_key=keys['api_key'],
        secret_key=keys['secret_key'],
        testnet=config['exchange']['testnet']
    )

    # API 연결 테스트
    if not client.ping():
        logger.error("Failed to connect to BingX API")
        return

    # 더미 환경 생성 (에이전트 로드용)
    import pandas as pd
    dummy_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='5min'),
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50000] * 100,
        'volume': [100] * 100
    })

    dummy_env = TradingEnvironment(df=dummy_df)

    # 에이전트 로드
    logger.info(f"Loading model: {args.model}")
    agent = RLAgent(env=dummy_env)

    try:
        agent.load_model(args.model)
    except FileNotFoundError:
        logger.error(f"Model '{args.model}' not found! Please train the model first.")
        return

    # Live Trader 생성
    live_trader = LiveTrader(
        client=client,
        agent=agent,
        config=config,
        dry_run=args.dry_run
    )

    # 거래 시작
    try:
        live_trader.start()
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
        live_trader.stop()
    except Exception as e:
        logger.error(f"Trading error: {str(e)}")
        live_trader.stop()
        raise

    logger.info("="*60)
    logger.info("Live Trading Bot Stopped")
    logger.info("="*60)


if __name__ == "__main__":
    main()
