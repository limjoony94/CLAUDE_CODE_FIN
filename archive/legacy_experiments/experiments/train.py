"""모델 훈련 스크립트"""

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
from src.agent.rl_agent import RLAgent, TradingCallback
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
        log_file=str(project_root / 'data' / 'logs' / 'training.log')
    )

    logger.info("="*60)
    logger.info("Starting Model Training")
    logger.info("="*60)

    # BingX 클라이언트 초기화
    testnet_keys = api_keys['bingx']['testnet']
    client = BingXClient(
        api_key=testnet_keys['api_key'],
        secret_key=testnet_keys['secret_key'],
        testnet=config['exchange']['testnet']
    )

    # 데이터 수집
    logger.info("Collecting historical data...")
    data_collector = DataCollector(
        client=client,
        symbol=config['exchange']['symbol'],
        interval=config['trading']['timeframe']
    )

    df = data_collector.collect_historical_data(
        days=config['data']['historical_days'],
        save=True
    )

    if df.empty:
        logger.error("No data collected, exiting...")
        return

    # 기술적 지표 계산
    logger.info("Calculating technical indicators...")
    indicator_calculator = TechnicalIndicators(config['indicators'])
    df = indicator_calculator.calculate_all_indicators(df)

    # 데이터 전처리
    logger.info("Processing data...")
    data_processor = DataProcessor()
    df = data_processor.prepare_data(df, fit=True)

    # 데이터 분할
    train_df, val_df, test_df = data_processor.split_data(
        df,
        train_ratio=0.7,
        val_ratio=0.15
    )

    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 훈련 환경 생성
    logger.info("Creating training environment...")
    train_env = TradingEnvironment(
        df=train_df,
        initial_balance=config['environment']['initial_balance'],
        leverage=config['trading']['leverage'],
        transaction_fee=config['environment']['transaction_fee'],
        slippage=config['environment']['slippage'],
        max_position_size=config['trading']['max_position_size'],
        reward_scaling=config['environment']['reward_scaling']
    )

    # 검증 환경 생성
    val_env = TradingEnvironment(
        df=val_df,
        initial_balance=config['environment']['initial_balance'],
        leverage=config['trading']['leverage'],
        transaction_fee=config['environment']['transaction_fee'],
        slippage=config['environment']['slippage'],
        max_position_size=config['trading']['max_position_size'],
        reward_scaling=config['environment']['reward_scaling']
    )

    # 에이전트 생성
    logger.info("Creating RL agent...")
    agent = RLAgent(
        env=train_env,
        config=config['rl']
    )

    agent.create_model()

    # 훈련 콜백
    callback = TradingCallback(verbose=1)

    # 훈련 시작
    logger.info("Starting training...")
    try:
        agent.train(
            total_timesteps=config['training']['total_timesteps'],
            eval_env=val_env,
            eval_freq=config['training']['eval_freq'],
            save_freq=config['training']['save_freq'],
            callback=callback
        )

        logger.info("Training completed successfully!")

        # 테스트 환경에서 평가
        logger.info("Evaluating on test set...")
        test_env = TradingEnvironment(
            df=test_df,
            initial_balance=config['environment']['initial_balance'],
            leverage=config['trading']['leverage'],
            transaction_fee=config['environment']['transaction_fee'],
            slippage=config['environment']['slippage'],
            max_position_size=config['trading']['max_position_size'],
            reward_scaling=config['environment']['reward_scaling']
        )

        # 에이전트 환경 변경
        agent.env = test_env

        # 평가
        eval_stats = agent.evaluate(n_episodes=config['training']['n_eval_episodes'])

        logger.info("Test Evaluation Results:")
        for key, value in eval_stats.items():
            logger.info(f"  {key}: {value:.4f}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    logger.info("="*60)
    logger.info("Training Script Finished")
    logger.info("="*60)


if __name__ == "__main__":
    main()
