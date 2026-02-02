"""V3 환경으로 훈련 - 순수 수익률 보상"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v3 import TradingEnvironmentV3
from src.agent.rl_agent import RLAgent, TradingCallback
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
import pandas as pd
from loguru import logger


def main():
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    setup_logger(
        log_level='INFO',
        log_file=str(project_root / 'data' / 'logs' / 'training_v3.log')
    )

    logger.info("="*60)
    logger.info("V3 Training - Pure PnL Reward")
    logger.info("Reward = Portfolio Return × 10000")
    logger.info("="*60)

    # 확장 데이터 로드
    data_path = project_root / 'data' / 'historical' / 'BTCUSDT_5m_extended.csv'

    if not data_path.exists():
        logger.error("Extended data not found!")
        return

    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    logger.info(f"Loaded {len(df)} candles")

    # 지표 계산
    indicator_calc = TechnicalIndicators(config['indicators'])
    df = indicator_calc.calculate_all_indicators(df)
    logger.info(f"After indicators: {len(df)} rows")

    # 전처리
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)
    logger.info(f"After processing: {len(df)} rows")

    # 분할
    train_df, val_df, test_df = processor.split_data(df, 0.7, 0.15)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # V3 환경 생성
    train_env = TradingEnvironmentV3(
        df=train_df,
        initial_balance=10000.0,
        leverage=10,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )

    val_env = TradingEnvironmentV3(
        df=val_df,
        initial_balance=10000.0,
        leverage=10,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )

    # 에이전트 생성
    logger.info("Creating RL agent...")
    agent = RLAgent(env=train_env, config=config['rl'])
    agent.create_model()

    # 훈련
    logger.info("Starting training for 3,000,000 timesteps...")
    logger.info("Mean Reward 100 = 1% profit, -100 = 1% loss")

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
        agent.save_model('final_model_v3')

        # 테스트 평가
        logger.info("\nEvaluating on test set...")
        test_env = TradingEnvironmentV3(
            df=test_df,
            initial_balance=10000.0,
            leverage=10,
            transaction_fee=0.0004,
            slippage=0.0001,
            max_position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )

        agent.env = test_env
        eval_stats = agent.evaluate(n_episodes=10)

        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Mean Reward: {eval_stats['mean_reward']:.2f}")
        logger.info(f"→ Estimated Return: {eval_stats['mean_reward']/10000*100:.2f}%")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        agent.save_model('model_v3_interrupted')
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
