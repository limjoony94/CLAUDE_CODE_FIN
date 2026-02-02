"""최종 훈련 - 최대 데이터(17,280 캔들)"""

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
        log_file=str(project_root / 'data' / 'logs' / 'training_final.log')
    )

    logger.info("="*70)
    logger.info("FINAL TRAINING - MAXIMUM DATA (17,280 CANDLES, 60 DAYS)")
    logger.info("="*70)

    # 최대 데이터 로드
    data_path = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'

    if not data_path.exists():
        logger.error("Max data not found! Run collect_public_data.py first")
        return

    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    logger.info(f"✓ Loaded {len(df):,} candles ({60} days)")

    # 지표 계산
    logger.info("Calculating indicators...")
    indicator_calc = TechnicalIndicators(config['indicators'])
    df = indicator_calc.calculate_all_indicators(df)
    logger.info(f"✓ After indicators: {len(df):,} rows")

    # 전처리
    logger.info("Processing data...")
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)
    logger.info(f"✓ After processing: {len(df):,} rows")

    # 분할 (70/15/15)
    train_df, val_df, test_df = processor.split_data(df, 0.7, 0.15)
    logger.info(f"✓ Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # V3 환경 생성
    logger.info("Creating environments...")
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
    logger.info("="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    logger.info("Total Timesteps: 5,000,000")
    logger.info("Eval Frequency:  10,000")
    logger.info("Save Frequency:  50,000")
    logger.info("Reward Meaning:  100 = 1% profit, -100 = 1% loss")
    logger.info("="*70)

    callback = TradingCallback(verbose=1)

    try:
        agent.train(
            total_timesteps=5000000,  # 5M 타임스텝
            eval_env=val_env,
            eval_freq=10000,
            save_freq=50000,
            callback=callback
        )

        logger.info("="*70)
        logger.info("✓ TRAINING COMPLETED!")
        logger.info("="*70)

        agent.save_model('final_model_max')
        logger.info("✓ Model saved: final_model_max")

        # 테스트 평가
        logger.info("\nEvaluating on test set (unseen data)...")
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
        eval_stats = agent.evaluate(n_episodes=20, deterministic=True)

        logger.info("\n" + "="*70)
        logger.info("TEST SET RESULTS (20 EPISODES)")
        logger.info("="*70)
        logger.info(f"Mean Reward:     {eval_stats['mean_reward']:>10.2f}")
        logger.info(f"Std Reward:      {eval_stats['std_reward']:>10.2f}")
        logger.info(f"Min Reward:      {eval_stats['min_reward']:>10.2f}")
        logger.info(f"Max Reward:      {eval_stats['max_reward']:>10.2f}")
        logger.info(f"{'─'*70}")
        logger.info(f"Est. Return:     {eval_stats['mean_reward']/10000*100:>10.2f}%")
        logger.info(f"Est. Min Return: {eval_stats['min_reward']/10000*100:>10.2f}%")
        logger.info(f"Est. Max Return: {eval_stats['max_reward']/10000*100:>10.2f}%")
        logger.info("="*70)

    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        agent.save_model('final_model_max_interrupted')
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    logger.info("\n" + "="*70)
    logger.info("FINAL TRAINING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
