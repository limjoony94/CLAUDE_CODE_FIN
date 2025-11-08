"""V5 모델 실제 행동 심층 분석"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v5 import TradingEnvironmentV5
from src.agent.rl_agent import RLAgent
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from loguru import logger


def main():
    logger.info("="*70)
    logger.info("V5 모델 실제 행동 심층 분석")
    logger.info("="*70)

    # 데이터 로드
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv', parse_dates=['timestamp'])

    # 전처리
    indicator_calc = TechnicalIndicators()
    df = indicator_calc.calculate_all_indicators(df)

    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)

    # 테스트 세트만
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    test_df = df.iloc[train_size+val_size:].reset_index(drop=True)

    logger.info(f"테스트 세트: {len(test_df):,} 캔들")

    # 환경 생성
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

    # 모델 로드
    agent = RLAgent(env=test_env, config={'rl': {'learning_rate': 0.0003}})
    model_path = 'v5_quick_test'
    agent.load_model(model_path)

    logger.info(f"모델 로드: {model_path}")

    # 실행 및 데이터 수집
    obs, info = test_env.reset()

    # 분석 데이터
    actions = []
    positions = []
    rewards = []
    portfolio_values = []
    trades = []

    step = 0
    done = False

    logger.info("\n실행 중...")

    while not done:
        action, _ = agent.predict(obs, deterministic=True)

        # 행동 기록
        actions.append(action[0])

        # 스텝 실행
        prev_position = test_env.position
        obs, reward, terminated, truncated, info = test_env.step(action)

        # 데이터 기록
        positions.append(info['position'])
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])

        # 거래 발생 체크
        if abs(info['position'] - prev_position) > 0.001:
            trades.append({
                'step': step,
                'action': action[0],
                'prev_position': prev_position,
                'new_position': info['position'],
                'price': info['current_price'],
                'reward': reward
            })

        step += 1
        done = terminated or truncated

        if step % 500 == 0:
            logger.info(f"Step {step}/{len(test_df)}")

    # 분석
    logger.info("\n" + "="*70)
    logger.info("분석 결과")
    logger.info("="*70)

    # 1. 행동 분포
    actions = np.array(actions)
    logger.info(f"\n【행동 분포】")
    logger.info(f"평균 행동: {actions.mean():.4f}")
    logger.info(f"표준편차: {actions.std():.4f}")
    logger.info(f"최소/최대: {actions.min():.4f} / {actions.max():.4f}")

    # 행동 히스토그램
    hist, bins = np.histogram(actions, bins=10, range=(-1, 1))
    logger.info(f"\n행동 히스토그램:")
    for i, (count, bin_start) in enumerate(zip(hist, bins[:-1])):
        bin_end = bins[i+1]
        pct = count / len(actions) * 100
        logger.info(f"  [{bin_start:+.1f} ~ {bin_end:+.1f}]: {count:4d} ({pct:5.2f}%)")

    # 2. 포지션 분포
    positions = np.array(positions)
    logger.info(f"\n【포지션 분포】")
    logger.info(f"평균 포지션: {positions.mean():.4f}")
    logger.info(f"표준편차: {positions.std():.4f}")
    logger.info(f"0 포지션 비율: {(abs(positions) < 0.001).sum() / len(positions) * 100:.2f}%")

    # 3. 보상 분포
    rewards = np.array(rewards)
    logger.info(f"\n【보상 분포】")
    logger.info(f"평균 보상: {rewards.mean():.4f}")
    logger.info(f"표준편차: {rewards.std():.4f}")
    logger.info(f"최소/최대: {rewards.min():.2f} / {rewards.max():.2f}")
    logger.info(f"중앙값: {np.median(rewards):.4f}")

    # 보상 범위별 분포
    logger.info(f"\n보상 범위별 분포:")
    ranges = [
        (-float('inf'), -100),
        (-100, -10),
        (-10, -1),
        (-1, 0),
        (0, 1),
        (1, 10),
        (10, 100),
        (100, float('inf'))
    ]
    for low, high in ranges:
        count = ((rewards >= low) & (rewards < high)).sum()
        pct = count / len(rewards) * 100
        logger.info(f"  [{low:>6} ~ {high:<6}): {count:5d} ({pct:5.2f}%)")

    # 4. 거래 분석
    logger.info(f"\n【거래 분석】")
    logger.info(f"총 거래 수: {len(trades)}")
    logger.info(f"거래 빈도: {len(trades) / len(test_df) * 100:.2f}%")

    if len(trades) > 0:
        # 홀딩 기간 분석
        hold_durations = []
        for i in range(1, len(trades)):
            duration = trades[i]['step'] - trades[i-1]['step']
            hold_durations.append(duration)

        if hold_durations:
            hold_durations = np.array(hold_durations)
            logger.info(f"\n홀딩 기간:")
            logger.info(f"  평균: {hold_durations.mean():.2f} 스텝")
            logger.info(f"  중앙값: {np.median(hold_durations):.0f} 스텝")
            logger.info(f"  최소/최대: {hold_durations.min()} / {hold_durations.max()}")
            logger.info(f"  < 10 스텝: {(hold_durations < 10).sum()} ({(hold_durations < 10).sum() / len(hold_durations) * 100:.2f}%)")

        # 거래 시 보상 분석
        trade_rewards = [t['reward'] for t in trades]
        logger.info(f"\n거래 시 보상:")
        logger.info(f"  평균: {np.mean(trade_rewards):.2f}")
        logger.info(f"  중앙값: {np.median(trade_rewards):.2f}")
        logger.info(f"  최소/최대: {np.min(trade_rewards):.2f} / {np.max(trade_rewards):.2f}")

    # 5. 포트폴리오 성과
    portfolio_values = np.array(portfolio_values)
    final_value = portfolio_values[-1]
    initial_value = 10000.0
    total_return = (final_value - initial_value) / initial_value * 100

    logger.info(f"\n【포트폴리오 성과】")
    logger.info(f"초기 자산: ${initial_value:,.2f}")
    logger.info(f"최종 자산: ${final_value:,.2f}")
    logger.info(f"총 수익률: {total_return:.2f}%")
    logger.info(f"최대 자산: ${portfolio_values.max():,.2f}")
    logger.info(f"최소 자산: ${portfolio_values.min():,.2f}")

    # 6. 보상 vs 실제 수익률 비교
    logger.info(f"\n【보상 vs 실제 수익률】")
    logger.info(f"총 보상 합계: {rewards.sum():.2f}")
    logger.info(f"실제 수익률: {total_return:.2f}%")
    logger.info(f"보상/수익률 비율: {rewards.sum() / total_return if total_return != 0 else 'N/A'}")

    # 7. 문제 진단
    logger.info(f"\n" + "="*70)
    logger.info("문제 진단")
    logger.info("="*70)

    issues = []

    # 거래 빈도
    trade_freq = len(trades) / len(test_df)
    if trade_freq > 0.5:
        issues.append(f"⚠️  과도한 거래 빈도: {trade_freq*100:.1f}% (목표: <20%)")

    # 홀딩 기간
    if len(hold_durations) > 0:
        short_holds = (hold_durations < 10).sum() / len(hold_durations) * 100
        if short_holds > 50:
            issues.append(f"⚠️  짧은 홀딩: {short_holds:.1f}%가 10 스텝 미만")

    # 보상 음수
    negative_rewards = (rewards < 0).sum() / len(rewards) * 100
    if negative_rewards > 80:
        issues.append(f"⚠️  과도한 음수 보상: {negative_rewards:.1f}%")

    # 보상 스케일
    reward_std = rewards.std()
    if reward_std > 50:
        issues.append(f"⚠️  불안정한 보상: 표준편차 {reward_std:.2f}")

    if issues:
        for issue in issues:
            logger.info(issue)
    else:
        logger.info("✅ 주요 문제 없음")

    logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()
