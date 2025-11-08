"""개선된 환경으로 모델 훈련 v2"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
from src.data.data_collector import DataCollector
from src.data.data_processor import DataProcessor
from src.indicators.technical_indicators import TechnicalIndicators
from src.environment.trading_env_v2 import TradingEnvironmentV2
from src.agent.rl_agent import RLAgent, TradingCallback
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from loguru import logger


def main():
    """메인 함수"""
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    api_keys = config_loader.load_api_keys()

    setup_logger(
        log_level=config['logging']['level'],
        log_file=str(project_root / 'data' / 'logs' / 'training_v2.log')
    )

    logger.info("="*60)
    logger.info("Model Training V2 (Improved Reward Function)")
    logger.info("="*60)

    # BingX 클라이언트
    testnet_keys = api_keys['bingx']['testnet']
    client = BingXClient(
        api_key=testnet_keys['api_key'],
        secret_key=testnet_keys['secret_key'],
        testnet=config['exchange']['testnet']
    )

    # 데이터 로드
    logger.info("Loading data...")
    data_collector = DataCollector(
        client=client,
        symbol=config['exchange']['symbol'],
        interval=config['trading']['timeframe']
    )

    # 확장 데이터 우선 로드
    df = data_collector.load_data('BTCUSDT_5m_extended.csv')
    if df.empty:
        logger.info("Extended data not found, trying default...")
        df = data_collector.load_data()
    if df.empty:
        logger.info("Collecting new data...")
        df = data_collector.collect_historical_data(days=365, save=True)

    # 기술적 지표
    logger.info("Calculating indicators...")
    indicator_calculator = TechnicalIndicators(config['indicators'])
    df = indicator_calculator.calculate_all_indicators(df)

    # 전처리
    logger.info("Processing data...")
    data_processor = DataProcessor()
    df = data_processor.prepare_data(df, fit=True)

    # 분할
    train_df, val_df, test_df = data_processor.split_data(df, 0.7, 0.15)
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # V2 환경 생성
    logger.info("Creating V2 environments...")
    train_env = TradingEnvironmentV2(
        df=train_df,
        initial_balance=config['environment']['initial_balance'],
        leverage=config['trading']['leverage'],
        transaction_fee=config['environment']['transaction_fee'],
        slippage=config['environment']['slippage'],
        max_position_size=config['trading']['max_position_size'],
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        reward_scaling=100.0
    )

    val_env = TradingEnvironmentV2(
        df=val_df,
        initial_balance=config['environment']['initial_balance'],
        leverage=config['trading']['leverage'],
        transaction_fee=config['environment']['transaction_fee'],
        slippage=config['environment']['slippage'],
        max_position_size=config['trading']['max_position_size'],
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        reward_scaling=100.0
    )

    # 에이전트 생성
    logger.info("Creating RL agent...")
    agent = RLAgent(env=train_env, config=config['rl'])
    agent.create_model()

    # 훈련 (더 많은 타임스텝)
    logger.info("Starting training...")
    callback = TradingCallback(verbose=1)

    try:
        agent.train(
            total_timesteps=3000000,  # 3M 타임스텝
            eval_env=val_env,
            eval_freq=config['training']['eval_freq'],
            save_freq=config['training']['save_freq'],
            callback=callback
        )

        logger.info("Training completed!")

        # 모델 저장
        agent.save_model('best_model_v2')

        # 테스트
        logger.info("Testing on test set...")
        test_env = TradingEnvironmentV2(
            df=test_df,
            initial_balance=config['environment']['initial_balance'],
            leverage=config['trading']['leverage'],
            transaction_fee=config['environment']['transaction_fee'],
            slippage=config['environment']['slippage'],
            max_position_size=config['trading']['max_position_size'],
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            reward_scaling=100.0
        )

        agent.env = test_env
        eval_stats = agent.evaluate(n_episodes=10)

        logger.info("Test Results:")
        for key, value in eval_stats.items():
            logger.info(f"  {key}: {value:.4f}")

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        agent.save_model('model_v2_interrupted')
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    logger.info("="*60)
    logger.info("Training V2 Finished")
    logger.info("="*60)


if __name__ == "__main__":
    main()
