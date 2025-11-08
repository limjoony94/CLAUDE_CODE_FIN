"""환경 동작 디버깅"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v2 import TradingEnvironmentV2
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from loguru import logger


def test_environment():
    """환경 동작 상세 테스트"""

    # 간단한 테스트 데이터
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='5min'),
        'open': [50000 + i*10 for i in range(100)],
        'high': [50100 + i*10 for i in range(100)],
        'low': [49900 + i*10 for i in range(100)],
        'close': [50000 + i*10 for i in range(100)],
        'volume': [100] * 100
    })

    # 지표 계산
    indicator_calc = TechnicalIndicators()
    df = indicator_calc.calculate_all_indicators(df)

    # 전처리
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)

    logger.info(f"Test data: {len(df)} rows")

    # 환경 생성
    env = TradingEnvironmentV2(
        df=df,
        initial_balance=10000.0,
        leverage=10,
        max_position_size=0.1,
        reward_scaling=100.0
    )

    obs, info = env.reset()

    logger.info("\n" + "="*60)
    logger.info("STEP-BY-STEP ANALYSIS")
    logger.info("="*60)

    # 테스트 시나리오: 롱 포지션 진입 → 유지 → 청산
    actions = [
        [1.0],   # 풀 롱 진입
        [1.0],   # 유지
        [1.0],   # 유지
        [1.0],   # 유지
        [0.0],   # 청산
    ]

    for i, action in enumerate(actions):
        logger.info(f"\n--- Step {i+1} ---")
        logger.info(f"Action: {action[0]:.2f}")
        logger.info(f"Before - Balance: ${info['balance']:.2f}, Position: {info['position']:.4f}, PnL: ${info['total_pnl']:.2f}")

        obs, reward, terminated, truncated, info = env.step(np.array(action))

        logger.info(f"After  - Balance: ${info['balance']:.2f}, Position: {info['position']:.4f}, PnL: ${info['total_pnl']:.2f}")
        logger.info(f"Reward: {reward:.2f}")
        logger.info(f"Portfolio Value: ${info['portfolio_value']:.2f}")

        if terminated:
            logger.error(f"TERMINATED at step {i+1}!")
            break

    logger.info("\n" + "="*60)
    logger.info("MARGIN CALCULATION CHECK")
    logger.info("="*60)

    # 증거금 계산 검증
    position_size = 0.1  # BTC
    price = 50000
    leverage = 10

    logger.info(f"Position: {position_size} BTC")
    logger.info(f"Price: ${price:,.0f}")
    logger.info(f"Leverage: {leverage}x")

    # 현재 코드의 계산
    wrong_margin = position_size * price
    logger.info(f"\n❌ WRONG (current code): ${wrong_margin:,.2f}")

    # 올바른 계산
    correct_margin = position_size * price / leverage
    logger.info(f"✅ CORRECT (should be): ${correct_margin:,.2f}")

    logger.info(f"\n💡 Difference: ${wrong_margin - correct_margin:,.2f}")
    logger.info(f"   Current code requires {wrong_margin/correct_margin:.0f}x more margin!")

    logger.info("\n" + "="*60)
    logger.info("REWARD SCALING CHECK")
    logger.info("="*60)

    # 보상 스케일링 체크
    scenarios = [
        ("Small profit", 0.001, 0.0),
        ("Small loss", -0.001, 0.0),
        ("Bankruptcy", -1.0, -100.0),
        ("Hold position", 0.0, -0.01),
    ]

    for name, portfolio_return, base_penalty in scenarios:
        portfolio_reward = portfolio_return * 100
        total = (portfolio_reward + base_penalty) * 100.0
        logger.info(f"{name:20s}: {total:>10.1f}")

    logger.info("\n" + "="*60)


if __name__ == "__main__":
    test_environment()
