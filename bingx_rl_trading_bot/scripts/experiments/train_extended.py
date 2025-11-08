"""확장 데이터(3997 캔들)로 모델 훈련"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v2 import TradingEnvironmentV2
from src.agent.rl_agent import RLAgent, TradingCallback
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
import pandas as pd
from loguru import logger


def main():
    """메인 함수"""
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    setup_logger(
        log_level='INFO',
        log_file=str(project_root / 'data' / 'logs' / 'training_extended.log')
    )

    logger.info("="*60)
    logger.info("Extended Data Training (3997 candles)")
    logger.info("="*60)

    # 확장 데이터 로드
    data_path = project_root / 'data' / 'historical' / 'BTCUSDT_5m_extended.csv'

    if not data_path.exists():
        logger.error("Extended data not found! Run collect_more_data.py first")
        return

    logger.info(f"Loading extended data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    logger.info(f"Loaded {len(df)} candles")

    # 기술적 지표 계산
    logger.info("Calculating indicators...")
    indicator_calculator = TechnicalIndicators(config['indicators'])
    df = indicator_calculator.calculate_all_indicators(df)
    logger.info(f"After indicators: {len(df)} rows")

    # 전처리
    logger.info("Processing data...")
    data_processor = DataProcessor()
    df = data_processor.prepare_data(df, fit=True)
    logger.info(f"After processing: {len(df)} rows")

    # 분할 (70/15/15)
    train_df, val_df, test_df = data_processor.split_data(df, 0.7, 0.15)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # V2 환경 생성
    logger.info("Creating V2 training environment...")
    train_env = TradingEnvironmentV2(
        df=train_df,
        initial_balance=10000.0,
        leverage=10,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        reward_scaling=100.0
    )

    logger.info("Creating V2 validation environment...")
    val_env = TradingEnvironmentV2(
        df=val_df,
        initial_balance=10000.0,
        leverage=10,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        reward_scaling=100.0
    )

    # 에이전트 생성
    logger.info("Creating RL agent...")
    agent = RLAgent(env=train_env, config=config['rl'])
    agent.create_model()

    # 훈련
    logger.info("Starting training for 3,000,000 timesteps...")
    logger.info("This will take several hours. Please wait...")

    callback = TradingCallback(verbose=1)

    try:
        agent.train(
            total_timesteps=3000000,
            eval_env=val_env,
            eval_freq=10000,
            save_freq=50000,
            callback=callback
        )

        logger.info("✓ Training completed!")

        # 최종 모델 저장
        agent.save_model('extended_model_v2')
        logger.info("✓ Model saved: extended_model_v2")

        # 테스트 평가
        logger.info("Evaluating on test set...")
        test_env = TradingEnvironmentV2(
            df=test_df,
            initial_balance=10000.0,
            leverage=10,
            transaction_fee=0.0004,
            slippage=0.0001,
            max_position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            reward_scaling=100.0
        )

        agent.env = test_env
        eval_stats = agent.evaluate(n_episodes=10)

        logger.info("\nTest Set Evaluation:")
        logger.info(f"  Mean Reward: {eval_stats['mean_reward']:.2f} (±{eval_stats['std_reward']:.2f})")
        logger.info(f"  Min Reward: {eval_stats['min_reward']:.2f}")
        logger.info(f"  Max Reward: {eval_stats['max_reward']:.2f}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        agent.save_model('extended_model_v2_interrupted')
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    logger.info("="*60)
    logger.info("Extended Training Finished")
    logger.info("="*60)


if __name__ == "__main__":
    main()
