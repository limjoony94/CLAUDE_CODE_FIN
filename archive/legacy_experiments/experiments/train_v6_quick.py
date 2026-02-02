"""V6 빠른 검증 훈련 - 15분봉 + 적응형 보상"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v6 import TradingEnvironmentV6
from src.agent.rl_agent import RLAgent
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from src.utils.config_loader import ConfigLoader
from loguru import logger


def main():
    logger.info("="*70)
    logger.info("V6 QUICK TEST - Adaptive Reward Scale + 15min")
    logger.info("="*70)
    logger.info("")
    logger.info("🔑 핵심 개선사항:")
    logger.info("  1. 적응형 보상 스케일 (volatility 기반)")
    logger.info("  2. 행동 공간 거래 제약 (페널티 제거)")
    logger.info("  3. 15분봉 (노이즈 감소, 수수료 1/3)")
    logger.info("  4. 손절/익절 완화 (3%/5%)")
    logger.info("="*70)

    # 15분봉 데이터 로드
    df = pd.read_csv('data/historical/BTCUSDT_15m.csv', parse_dates=['timestamp'])
    logger.info(f"\n✓ Loaded 15-minute data: {len(df):,} candles")

    # 지표 계산
    indicator_calc = TechnicalIndicators()
    df = indicator_calc.calculate_all_indicators(df)
    logger.info(f"✓ Calculated indicators: {len(df):,} rows")

    # 전처리
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)
    logger.info(f"✓ Preprocessed: {len(df):,} rows")

    # 데이터 분할
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]

    logger.info(f"✓ Split - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # V6 환경 생성
    logger.info(f"\n{'='*70}")
    logger.info("Creating V6 environments...")
    logger.info(f"{'='*70}")

    train_env = TradingEnvironmentV6(
        df=train_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.03,  # 3%
        take_profit_pct=0.05,  # 5%
        min_hold_steps=10,
        volatility_window=50
    )

    val_env = TradingEnvironmentV6(
        df=val_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        min_hold_steps=10,
        volatility_window=50
    )

    # 에이전트 생성
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    agent = RLAgent(env=train_env, config=config['rl'])
    agent.create_model()

    # 훈련
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING (500K timesteps for quick validation)")
    logger.info(f"{'='*70}")
    logger.info("Expected outcomes:")
    logger.info("  ✓ Action diversity (std > 0.1)")
    logger.info("  ✓ Trade frequency 5~15%")
    logger.info("  ✓ NO Mode Collapse")
    logger.info("  ✓ Test return > 0%")
    logger.info(f"{'='*70}\n")

    agent.train(
        total_timesteps=500000,
        eval_env=val_env,
        eval_freq=50000,
        save_freq=100000
    )

    logger.info("\n✓ Training complete!")
    agent.save_model('v6_quick_test')

    # 테스트
    logger.info(f"\n{'='*70}")
    logger.info("TESTING on test set...")
    logger.info(f"{'='*70}")

    test_env = TradingEnvironmentV6(
        df=test_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        min_hold_steps=10,
        volatility_window=50
    )

    agent.env = test_env
    stats = agent.evaluate(n_episodes=10, deterministic=True)

    logger.info(f"\n{'='*70}")
    logger.info("QUICK TEST RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Mean Reward: {stats['mean_reward']:.2f}")
    logger.info(f"Est. Return: {stats['mean_reward']/10000*100:.4f}% (rough estimate)")
    logger.info(f"{'='*70}")

    # 실제 백테스팅
    logger.info("\nRunning actual backtest...")
    obs, info = test_env.reset()
    done = False

    initial_value = info['portfolio_value']

    # 행동 분석용
    actions = []
    rewards_list = []

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        actions.append(action[0])

        obs, reward, terminated, truncated, info = test_env.step(action)
        rewards_list.append(reward)

        done = terminated or truncated

    final_value = info['portfolio_value']
    actual_return = (final_value - initial_value) / initial_value * 100

    # 행동 분석
    actions = pd.Series(actions)

    logger.info(f"\n{'='*70}")
    logger.info("DETAILED RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Portfolio:")
    logger.info(f"  Initial: ${initial_value:,.2f}")
    logger.info(f"  Final: ${final_value:,.2f}")
    logger.info(f"  Return: {actual_return:.2f}%")

    logger.info(f"\nTrading:")
    logger.info(f"  Total Trades: {info['trade_count']}")
    logger.info(f"  Trade Frequency: {info['trade_count']/len(test_df)*100:.2f}%")
    logger.info(f"  Win Rate: {info['win_rate']*100:.2f}%")

    logger.info(f"\nAction Analysis:")
    logger.info(f"  Mean: {actions.mean():.4f}")
    logger.info(f"  Std: {actions.std():.4f}")
    logger.info(f"  Min/Max: {actions.min():.4f} / {actions.max():.4f}")

    logger.info(f"\nReward Analysis:")
    rewards_arr = pd.Series(rewards_list)
    logger.info(f"  Mean: {rewards_arr.mean():.4f}")
    logger.info(f"  Std: {rewards_arr.std():.2f}")
    logger.info(f"  Min/Max: {rewards_arr.min():.2f} / {rewards_arr.max():.2f}")

    # 성공 평가
    logger.info(f"\n{'='*70}")
    logger.info("SUCCESS CRITERIA EVALUATION")
    logger.info(f"{'='*70}")

    criteria = {
        "Action Diversity (std > 0.1)": actions.std() > 0.1,
        "Trade Frequency 5~15%": 5 < (info['trade_count']/len(test_df)*100) < 15,
        "Test Return > 0%": actual_return > 0,
        "Mode Collapse Avoided": actions.std() > 0.05
    }

    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} - {criterion}")

    success_rate = sum(criteria.values()) / len(criteria) * 100
    logger.info(f"\nOverall Success: {success_rate:.0f}% ({sum(criteria.values())}/{len(criteria)})")

    if success_rate >= 75:
        logger.info(f"\n🎉 V6 검증 성공! 전체 훈련 진행 권장")
    elif success_rate >= 50:
        logger.info(f"\n⚠️  부분 성공. 파라미터 조정 후 재시도 권장")
    else:
        logger.info(f"\n❌ 검증 실패. 추가 분석 필요")

    logger.info(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
