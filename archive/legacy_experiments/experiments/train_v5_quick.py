"""V5 빠른 테스트 훈련"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v5 import TradingEnvironmentV5
from src.agent.rl_agent import RLAgent
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from src.utils.config_loader import ConfigLoader
from loguru import logger


def main():
    logger.info("="*70)
    logger.info("V5 QUICK TEST - Pure PnL + Strong Constraints")
    logger.info("="*70)

    # 데이터
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv', parse_dates=['timestamp'])
    logger.info(f"Loaded {len(df):,} candles")

    # 지표
    indicator_calc = TechnicalIndicators()
    df = indicator_calc.calculate_all_indicators(df)

    # 전처리
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)

    # 분할
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]

    logger.info(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # V5 환경
    train_env = TradingEnvironmentV5(
        df=train_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.02,  # 1% → 2%
        take_profit_pct=0.03,  # 2% → 3%
        min_hold_steps=10  # 3 → 10
    )

    val_env = TradingEnvironmentV5(
        df=val_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        min_hold_steps=10
    )

    # 에이전트
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    agent = RLAgent(env=train_env, config=config['rl'])
    agent.create_model()

    logger.info("\n" + "="*70)
    logger.info("TRAINING (500K timesteps for quick test)")
    logger.info("="*70)
    logger.info("Changes from V4:")
    logger.info("  - Holding bonus: REMOVED")
    logger.info("  - Frequency penalty: -5 → -100 (20x stronger)")
    logger.info("  - Min hold: 3 → 10 steps")
    logger.info("  - Stop loss: 1% → 2%")
    logger.info("  - Take profit: 2% → 3%")
    logger.info("="*70)

    agent.train(
        total_timesteps=500000,
        eval_env=val_env,
        eval_freq=50000,
        save_freq=100000
    )

    logger.info("Training complete!")
    agent.save_model('v5_quick_test')

    # 테스트
    test_env = TradingEnvironmentV5(
        df=test_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        min_hold_steps=10
    )

    agent.env = test_env
    stats = agent.evaluate(n_episodes=10, deterministic=True)

    logger.info("\n" + "="*70)
    logger.info("QUICK TEST RESULTS")
    logger.info("="*70)
    logger.info(f"Mean Reward: {stats['mean_reward']:.2f}")
    logger.info(f"Est. Return: {stats['mean_reward']/10000*100:.2f}%")
    logger.info("="*70)

    # 실제 백테스팅
    logger.info("\nActual backtest...")
    obs, info = test_env.reset()
    done = False

    initial = info['portfolio_value']

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

    final = info['portfolio_value']
    actual_return = (final - initial) / initial * 100

    logger.info(f"\nActual Return: {actual_return:.2f}%")
    logger.info(f"Trades: {info['trade_count']}")
    logger.info(f"Win Rate: {info['win_rate']*100:.2f}%")

    logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()
