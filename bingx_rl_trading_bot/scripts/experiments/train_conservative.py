"""보수적 설정으로 훈련 - 안정성 우선"""

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
        log_file=str(project_root / 'data' / 'logs' / 'training_conservative.log')
    )

    logger.info("="*70)
    logger.info("CONSERVATIVE TRAINING - STABILITY FOCUSED")
    logger.info("="*70)
    logger.info("Leverage: 3x (reduced from 10x)")
    logger.info("Position Size: 0.03 BTC (reduced from 0.1)")
    logger.info("Stop Loss: 1% (reduced from 2%)")
    logger.info("="*70)

    # 최대 데이터 로드
    data_path = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'

    if not data_path.exists():
        logger.error("Max data not found!")
        return

    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    logger.info(f"✓ Loaded {len(df):,} candles")

    # 지표
    indicator_calc = TechnicalIndicators(config['indicators'])
    df = indicator_calc.calculate_all_indicators(df)
    logger.info(f"✓ Indicators: {len(df):,} rows")

    # 전처리
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)
    logger.info(f"✓ Processed: {len(df):,} rows")

    # 분할
    train_df, val_df, test_df = processor.split_data(df, 0.7, 0.15)
    logger.info(f"✓ Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # 보수적 설정 환경
    train_env = TradingEnvironmentV3(
        df=train_df,
        initial_balance=10000.0,
        leverage=3,  # 10 → 3
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,  # 0.1 → 0.03
        stop_loss_pct=0.01,  # 0.02 → 0.01
        take_profit_pct=0.02  # 0.04 → 0.02
    )

    val_env = TradingEnvironmentV3(
        df=val_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.01,
        take_profit_pct=0.02
    )

    # 에이전트
    logger.info("Creating agent...")
    agent = RLAgent(env=train_env, config=config['rl'])
    agent.create_model()

    # 훈련
    logger.info("\n" + "="*70)
    logger.info("TRAINING START")
    logger.info("="*70)
    logger.info("Timesteps: 5,000,000")
    logger.info("Strategy: Conservative (low leverage, small positions)")
    logger.info("="*70)

    callback = TradingCallback(verbose=1)

    try:
        agent.train(
            total_timesteps=5000000,
            eval_env=val_env,
            eval_freq=10000,
            save_freq=50000,
            callback=callback
        )

        logger.info("✓ Training completed!")
        agent.save_model('conservative_model')

        # 테스트
        logger.info("\nTest set evaluation...")
        test_env = TradingEnvironmentV3(
            df=test_df,
            initial_balance=10000.0,
            leverage=3,
            transaction_fee=0.0004,
            slippage=0.0001,
            max_position_size=0.03,
            stop_loss_pct=0.01,
            take_profit_pct=0.02
        )

        agent.env = test_env
        eval_stats = agent.evaluate(n_episodes=20, deterministic=True)

        logger.info("\n" + "="*70)
        logger.info("TEST RESULTS (Conservative Strategy)")
        logger.info("="*70)
        logger.info(f"Mean Reward:     {eval_stats['mean_reward']:>10.2f}")
        logger.info(f"Std Reward:      {eval_stats['std_reward']:>10.2f}")
        logger.info(f"Min Reward:      {eval_stats['min_reward']:>10.2f}")
        logger.info(f"Max Reward:      {eval_stats['max_reward']:>10.2f}")
        logger.info(f"{'─'*70}")
        logger.info(f"Est. Return:     {eval_stats['mean_reward']/10000*100:>10.2f}%")
        logger.info(f"Est. Range:      {eval_stats['min_reward']/10000*100:>10.2f}% to {eval_stats['max_reward']/10000*100:>10.2f}%")
        logger.info("="*70)

        # 안정성 평가
        volatility = eval_stats['std_reward'] / abs(eval_stats['mean_reward']) if eval_stats['mean_reward'] != 0 else 0
        logger.info(f"\nStability Score: {1 / (1 + volatility):.2%}")

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        agent.save_model('conservative_model_interrupted')
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
