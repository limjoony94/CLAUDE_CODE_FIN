"""백테스팅 스크립트"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
from src.data.data_collector import DataCollector
from src.data.data_processor import DataProcessor
from src.indicators.technical_indicators import TechnicalIndicators
from src.environment.trading_env import TradingEnvironment
from src.agent.rl_agent import RLAgent
from src.backtesting.backtest_engine import BacktestEngine
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
    setup_logger(
        log_level=config['logging']['level'],
        log_file=str(project_root / 'data' / 'logs' / 'backtest.log')
    )

    logger.info("="*60)
    logger.info("Starting Backtest")
    logger.info("="*60)

    # BingX 클라이언트 초기화
    testnet_keys = api_keys['bingx']['testnet']
    client = BingXClient(
        api_key=testnet_keys['api_key'],
        secret_key=testnet_keys['secret_key'],
        testnet=config['exchange']['testnet']
    )

    # 데이터 수집
    logger.info("Loading historical data...")
    data_collector = DataCollector(
        client=client,
        symbol=config['exchange']['symbol'],
        interval=config['trading']['timeframe']
    )

    # 저장된 데이터 로드 또는 새로 수집
    df = data_collector.load_data()
    if df.empty:
        logger.info("No saved data found, collecting new data...")
        df = data_collector.collect_historical_data(
            days=config['data']['historical_days'],
            save=True
        )

    if df.empty:
        logger.error("No data available, exiting...")
        return

    # 기술적 지표 계산
    logger.info("Calculating technical indicators...")
    indicator_calculator = TechnicalIndicators(config['indicators'])
    df = indicator_calculator.calculate_all_indicators(df)

    # 데이터 전처리
    logger.info("Processing data...")
    data_processor = DataProcessor()
    df = data_processor.prepare_data(df, fit=True)

    # 백테스트 기간 설정 (테스트 데이터)
    _, _, test_df = data_processor.split_data(df, train_ratio=0.7, val_ratio=0.15)

    logger.info(f"Backtest period: {len(test_df)} candles")

    # 백테스트 환경 생성
    backtest_env = TradingEnvironment(
        df=test_df,
        initial_balance=config['backtesting']['initial_capital'],
        leverage=config['trading']['leverage'],
        transaction_fee=config['environment']['transaction_fee'],
        slippage=config['environment']['slippage'],
        max_position_size=config['trading']['max_position_size']
    )

    # 에이전트 로드
    logger.info("Loading trained model...")
    agent = RLAgent(env=backtest_env)

    try:
        agent.load_model('best_model')  # 또는 'final_model'
    except FileNotFoundError:
        logger.error("Trained model not found! Please train the model first.")
        return

    # 백테스트 엔진 생성
    backtest_engine = BacktestEngine(env=backtest_env, agent=agent)

    # 백테스트 실행
    logger.info("Running backtest...")
    results = backtest_engine.run_backtest(n_episodes=1, deterministic=True)

    # 결과 출력
    backtest_engine.print_summary()

    # 결과 저장
    backtest_engine.save_results('backtest_results.csv')

    # 결과 시각화 (선택)
    try:
        backtest_engine.plot_results()
    except Exception as e:
        logger.warning(f"Failed to plot results: {str(e)}")

    logger.info("="*60)
    logger.info("Backtest Finished")
    logger.info("="*60)


if __name__ == "__main__":
    main()
