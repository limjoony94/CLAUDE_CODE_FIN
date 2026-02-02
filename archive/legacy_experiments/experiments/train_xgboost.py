"""XGBoost 모델 훈련 스크립트

강화학습 대비 개선 사항:
- 데이터 효율성: 60일 데이터로 충분
- 과적합 방지: 정규화 + early stopping
- 빠른 훈련: 5-30분 vs 5-20시간
- 해석 가능: Feature importance 제공
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.models.xgboost_trader import XGBoostTrader
from src.indicators.technical_indicators import TechnicalIndicators


def main():
    logger.info("=" * 80)
    logger.info("XGBoost Trading Model Training")
    logger.info("=" * 80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_dir = project_root / 'data' / 'historical'
    data_file = data_dir / 'BTCUSDT_5m_max.csv'  # 17,280 캔들

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please run collect_data.py first")
        return

    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} candles")

    # 2. 기술적 지표 계산
    logger.info("\n2. Calculating technical indicators...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)

    logger.info(f"Processed {len(df_processed)} candles with indicators")
    logger.info(f"Date range: {df_processed['timestamp'].iloc[0]} to {df_processed['timestamp'].iloc[-1]}")

    # 3. XGBoost 트레이더 초기화
    logger.info("\n3. Initializing XGBoost Trader...")

    trader = XGBoostTrader(
        lookahead=5,                    # 5봉 (25분) 앞 예측
        threshold_pct=0.002,            # 0.2% (수수료 0.08% + 여유)
        confidence_threshold=0.55,      # 55% 확신도 이상만 거래
        model_params=None               # 기본 파라미터 사용
    )

    # 4. 데이터 준비
    logger.info("\n4. Preparing train/val/test split...")

    train_df, val_df, test_df = trader.prepare_data(
        df_processed,
        train_ratio=0.7,   # 70% 훈련
        val_ratio=0.15,    # 15% 검증
        # 15% 테스트
    )

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # 5. 모델 훈련
    logger.info("\n5. Training XGBoost model...")
    logger.info("This may take 5-30 minutes depending on data size...")

    train_stats = trader.train(
        train_df=train_df,
        val_df=val_df,
        num_boost_round=500,           # 최대 반복
        early_stopping_rounds=50,      # 50회 개선 없으면 중단
        verbose=True
    )

    logger.info(f"\nTraining Statistics:")
    logger.info(f"Best iteration: {train_stats['best_iteration']}")
    logger.info(f"Train loss: {train_stats['train_loss']:.4f}")
    logger.info(f"Validation loss: {train_stats['val_loss']:.4f}")
    logger.info(f"Validation accuracy: {train_stats['val_accuracy']:.4f}")

    # 6. 평가
    logger.info("\n6. Evaluating on all sets...")

    # 훈련 세트
    logger.info("\n--- Training Set ---")
    train_results = trader.evaluate(train_df, name="Train")

    # 검증 세트
    logger.info("\n--- Validation Set ---")
    val_results = trader.evaluate(val_df, name="Validation")

    # 테스트 세트
    logger.info("\n--- Test Set ---")
    test_results = trader.evaluate(test_df, name="Test")

    # 7. 백테스팅
    logger.info("\n7. Running backtests...")

    # 검증 세트 백테스트
    logger.info("\n--- Validation Backtest ---")
    val_backtest = trader.backtest(
        val_df,
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    # 테스트 세트 백테스트
    logger.info("\n--- Test Backtest ---")
    test_backtest = trader.backtest(
        test_df,
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    # 8. 모델 저장
    logger.info("\n8. Saving model...")
    trader.save_model('xgboost_v1')
    logger.info("Model saved successfully")

    # 9. 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info("\n📊 Model Performance:")
    logger.info(f"Train Accuracy: {train_results['accuracy']:.4f}")
    logger.info(f"Val Accuracy:   {val_results['accuracy']:.4f}")
    logger.info(f"Test Accuracy:  {test_results['accuracy']:.4f}")

    logger.info("\n💰 Backtest Results:")
    logger.info(f"Validation Return: {val_backtest['total_return_pct']:.2f}%")
    logger.info(f"Test Return:       {test_backtest['total_return_pct']:.2f}%")

    logger.info(f"\nValidation Sharpe: {val_backtest['sharpe_ratio']:.2f}")
    logger.info(f"Test Sharpe:       {test_backtest['sharpe_ratio']:.2f}")

    logger.info(f"\nValidation Trades: {val_backtest['num_trades']}")
    logger.info(f"Test Trades:       {test_backtest['num_trades']}")

    logger.info(f"\nValidation Win Rate: {val_backtest['win_rate']*100:.1f}%")
    logger.info(f"Test Win Rate:       {test_backtest['win_rate']*100:.1f}%")

    # 과적합 비율
    if val_backtest['total_return_pct'] > 0:
        overfitting_ratio = abs(val_backtest['total_return_pct']) / abs(test_backtest['total_return_pct'] + 1e-8)
        logger.info(f"\n⚠️ Overfitting Ratio: {overfitting_ratio:.2f}x")
        if overfitting_ratio < 1.5:
            logger.info("✅ Low overfitting - Good generalization!")
        elif overfitting_ratio < 2.5:
            logger.info("⚠️ Moderate overfitting - Acceptable")
        else:
            logger.info("❌ High overfitting - Need more data or regularization")

    # RL과 비교 (FINAL_REPORT 기준)
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH RL (Conservative Model)")
    logger.info("=" * 80)

    rl_val_return = 0.59
    rl_test_return = -1.05
    rl_overfitting = 2.8

    logger.info("\n| Metric              | RL (PPO) | XGBoost | Improvement |")
    logger.info("|---------------------|----------|---------|-------------|")
    logger.info(f"| Val Return          | +{rl_val_return:.2f}%  | {val_backtest['total_return_pct']:+.2f}%  | {val_backtest['total_return_pct'] - rl_val_return:+.2f}% |")
    logger.info(f"| Test Return         | {rl_test_return:.2f}%  | {test_backtest['total_return_pct']:+.2f}%  | {test_backtest['total_return_pct'] - rl_test_return:+.2f}% |")

    if val_backtest['total_return_pct'] > 0:
        xgb_overfitting = abs(val_backtest['total_return_pct']) / abs(test_backtest['total_return_pct'] + 1e-8)
        logger.info(f"| Overfitting Ratio   | {rl_overfitting:.2f}x   | {xgb_overfitting:.2f}x   | {((rl_overfitting - xgb_overfitting) / rl_overfitting * 100):+.1f}% |")

    logger.info(f"| Sharpe (Test)       | -0.30    | {test_backtest['sharpe_ratio']:.2f}    | {test_backtest['sharpe_ratio'] - (-0.30):+.2f} |")
    logger.info(f"| Training Time       | 5-20h    | ~30min  | 10-40x faster |")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Training completed!")
    logger.info("=" * 80)

    # 결과를 파일로 저장
    results_dir = project_root / 'data' / 'trained_models' / 'xgboost'
    results_file = results_dir / 'training_results.txt'

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("XGBoost Training Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Model Performance:\n")
        f.write(f"Train Accuracy: {train_results['accuracy']:.4f}\n")
        f.write(f"Val Accuracy:   {val_results['accuracy']:.4f}\n")
        f.write(f"Test Accuracy:  {test_results['accuracy']:.4f}\n\n")

        f.write("Backtest Results:\n")
        f.write(f"Validation Return: {val_backtest['total_return_pct']:.2f}%\n")
        f.write(f"Test Return:       {test_backtest['total_return_pct']:.2f}%\n")
        f.write(f"Validation Sharpe: {val_backtest['sharpe_ratio']:.2f}\n")
        f.write(f"Test Sharpe:       {test_backtest['sharpe_ratio']:.2f}\n")
        f.write(f"Validation Trades: {val_backtest['num_trades']}\n")
        f.write(f"Test Trades:       {test_backtest['num_trades']}\n")
        f.write(f"Validation Win Rate: {val_backtest['win_rate']*100:.1f}%\n")
        f.write(f"Test Win Rate:       {test_backtest['win_rate']*100:.1f}%\n")

    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
