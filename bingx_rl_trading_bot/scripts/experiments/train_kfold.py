"""K-Fold 교차 검증 훈련 - 과적합 방지"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v4 import TradingEnvironmentV4
from src.agent.rl_agent import RLAgent, TradingCallback
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from loguru import logger


def k_fold_train(k=5):
    """
    K-Fold 교차 검증 훈련

    전략:
    1. 데이터를 K개로 분할
    2. 각 Fold를 검증으로 사용하여 훈련
    3. 평균 성능이 가장 좋은 모델 선택
    """

    config_loader = ConfigLoader()
    config = config_loader.load_config()

    setup_logger(
        log_level='INFO',
        log_file=str(project_root / 'data' / 'logs' / 'training_kfold.log')
    )

    logger.info("="*70)
    logger.info(f"K-FOLD CROSS VALIDATION TRAINING (K={k})")
    logger.info("="*70)

    # 데이터 로드
    data_path = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    logger.info(f"✓ Loaded {len(df):,} candles")

    # 지표 계산
    indicator_calc = TechnicalIndicators(config['indicators'])
    df = indicator_calc.calculate_all_indicators(df)

    # 전처리 (정규화 유지)
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)  # 정규화 적용
    logger.info(f"✓ Processed {len(df):,} rows")

    # 테스트 세트 분리 (마지막 15%)
    test_size = int(len(df) * 0.15)
    test_df = df.iloc[-test_size:]
    train_val_df = df.iloc[:-test_size]

    logger.info(f"✓ Test set: {len(test_df):,} (held out)")
    logger.info(f"✓ Train+Val: {len(train_val_df):,}")

    # K-Fold 분할
    fold_size = len(train_val_df) // k
    fold_results = []

    for fold_idx in range(k):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold_idx + 1}/{k}")
        logger.info(f"{'='*70}")

        # 검증 세트 범위
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < k-1 else len(train_val_df)

        # 훈련/검증 분할
        val_fold = train_val_df.iloc[val_start:val_end]
        train_fold = pd.concat([
            train_val_df.iloc[:val_start],
            train_val_df.iloc[val_end:]
        ])

        logger.info(f"Train: {len(train_fold):,}, Val: {len(val_fold):,}")

        # V4 환경 생성
        train_env = TradingEnvironmentV4(
            df=train_fold,
            initial_balance=10000.0,
            leverage=3,
            transaction_fee=0.0004,
            slippage=0.0001,
            max_position_size=0.03,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            min_hold_steps=3
        )

        val_env = TradingEnvironmentV4(
            df=val_fold,
            initial_balance=10000.0,
            leverage=3,
            transaction_fee=0.0004,
            slippage=0.0001,
            max_position_size=0.03,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            min_hold_steps=3
        )

        # 에이전트 훈련
        agent = RLAgent(env=train_env, config=config['rl'])
        agent.create_model()

        callback = TradingCallback(verbose=0)

        logger.info(f"Training Fold {fold_idx + 1}...")

        agent.train(
            total_timesteps=1000000,  # Fold당 1M (총 5M)
            eval_env=val_env,
            eval_freq=50000,
            save_freq=200000,
            callback=callback
        )

        # Fold 평가
        agent.env = val_env
        fold_stats = agent.evaluate(n_episodes=5, deterministic=True)

        fold_results.append({
            'fold': fold_idx + 1,
            'val_reward': fold_stats['mean_reward'],
            'val_std': fold_stats['std_reward']
        })

        logger.info(f"Fold {fold_idx + 1} Val Reward: {fold_stats['mean_reward']:.2f}")

        # Fold 모델 저장
        agent.save_model(f'kfold_model_fold{fold_idx+1}')

    # K-Fold 결과 요약
    logger.info(f"\n{'='*70}")
    logger.info("K-FOLD RESULTS SUMMARY")
    logger.info(f"{'='*70}")

    for result in fold_results:
        logger.info(f"Fold {result['fold']}: {result['val_reward']:+.2f} (±{result['val_std']:.2f})")

    # 평균 성능
    avg_reward = np.mean([r['val_reward'] for r in fold_results])
    std_reward = np.std([r['val_reward'] for r in fold_results])

    logger.info(f"\nAverage Validation Reward: {avg_reward:+.2f} (±{std_reward:.2f})")
    logger.info(f"Estimated Return: {avg_reward/10000*100:+.2f}%")

    # 최고 Fold 선택
    best_fold = max(fold_results, key=lambda x: x['val_reward'])
    logger.info(f"\nBest Fold: {best_fold['fold']} (Reward: {best_fold['val_reward']:+.2f})")

    # 최고 모델로 테스트
    logger.info(f"\n{'='*70}")
    logger.info("FINAL TEST SET EVALUATION")
    logger.info(f"{'='*70}")

    test_env = TradingEnvironmentV4(
        df=test_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        min_hold_steps=3
    )

    best_agent = RLAgent(env=test_env)
    best_agent.load_model(f'kfold_model_fold{best_fold["fold"]}')

    test_stats = best_agent.evaluate(n_episodes=20, deterministic=True)

    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Mean Reward: {test_stats['mean_reward']:+.2f}")
    logger.info(f"  Std Reward: {test_stats['std_reward']:.2f}")
    logger.info(f"  Est. Return: {test_stats['mean_reward']/10000*100:+.2f}%")

    # 일반화 성능
    generalization = test_stats['mean_reward'] / best_fold['val_reward']
    logger.info(f"\nGeneralization Ratio: {generalization:.2f}x")
    logger.info(f"Overfitting: {'Low' if generalization > 0.7 else 'High'}")

    logger.info(f"\n{'='*70}")
    logger.info("K-FOLD TRAINING COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    k_fold_train(k=5)
