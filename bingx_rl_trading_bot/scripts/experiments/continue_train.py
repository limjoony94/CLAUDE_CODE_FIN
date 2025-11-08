"""기존 모델 추가 훈련 스크립트"""

import sys
import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description='Continue training existing model')
    parser.add_argument(
        '--model',
        type=str,
        default='best_model',
        help='Model name to load (default: best_model)'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=2000000,
        help='Additional timesteps to train (default: 2000000)'
    )

    args = parser.parse_args()

    # 설정 로드
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    api_keys = config_loader.load_api_keys()

    # 로거 설정
    setup_logger(
        log_level=config['logging']['level'],
        log_file=str(project_root / 'data' / 'logs' / 'continue_training.log')
    )

    logger.info("="*60)
    logger.info("Continue Model Training")
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Additional timesteps: {args.timesteps:,}")
    logger.info("="*60)

    # BingX 클라이언트 초기화
    testnet_keys = api_keys['bingx']['testnet']
    client = BingXClient(
        api_key=testnet_keys['api_key'],
        secret_key=testnet_keys['secret_key'],
        testnet=config['exchange']['testnet']
    )

    # 데이터 수집
    logger.info("Loading data...")
    data_collector = DataCollector(
        client=client,
        symbol=config['exchange']['symbol'],
        interval=config['trading']['timeframe']
    )

    # 저장된 데이터 로드
    df = data_collector.load_data()
    if df.empty:
        logger.info("No saved data, collecting new data...")
        df = data_collector.collect_historical_data(days=365, save=True)

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

    # 검증 환경
    val_env = TradingEnvironment(
        df=val_df,
        initial_balance=config['environment']['initial_balance'],
        leverage=config['trading']['leverage'],
        transaction_fee=config['environment']['transaction_fee'],
        slippage=config['environment']['slippage'],
        max_position_size=config['trading']['max_position_size'],
        reward_scaling=config['environment']['reward_scaling']
    )

    # 에이전트 생성 및 모델 로드
    logger.info(f"Loading model: {args.model}...")
    agent = RLAgent(env=train_env, config=config['rl'])

    try:
        agent.load_model(args.model)
        logger.info("Model loaded successfully!")
    except FileNotFoundError:
        logger.error(f"Model '{args.model}' not found!")
        return

    # 추가 훈련
    logger.info(f"Starting additional training for {args.timesteps:,} timesteps...")

    callback = TradingCallback(verbose=1)

    try:
        agent.train(
            total_timesteps=args.timesteps,
            eval_env=val_env,
            eval_freq=config['training']['eval_freq'],
            save_freq=config['training']['save_freq'],
            callback=callback
        )

        logger.info("Additional training completed!")

        # 모델 저장
        agent.save_model(f'{args.model}_extended')
        logger.info(f"Extended model saved as: {args.model}_extended")

        # 테스트 평가
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

        agent.env = test_env
        eval_stats = agent.evaluate(n_episodes=10)

        logger.info("Test Evaluation Results:")
        for key, value in eval_stats.items():
            logger.info(f"  {key}: {value:.4f}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        agent.save_model(f'{args.model}_interrupted')
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    logger.info("="*60)
    logger.info("Continue Training Finished")
    logger.info("="*60)


if __name__ == "__main__":
    main()
