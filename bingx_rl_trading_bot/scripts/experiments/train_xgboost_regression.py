"""XGBoost Regression 모델 훈련 스크립트

핵심 개선:
1. 분류 → 회귀 문제로 전환
2. 클래스 불균형 근본 해결
3. SMOTE 불필요
4. 실제 수익률 직접 예측
5. lookahead=48 (4시간) - 균형잡힌 선택
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger

from src.models.xgboost_trader_regression import XGBoostTraderRegression
from src.indicators.technical_indicators import TechnicalIndicators


def main():
    logger.info("=" * 80)
    logger.info("XGBoost REGRESSION Trading Model Training")
    logger.info("Classification → Regression Problem")
    logger.info("=" * 80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} candles")

    # 2. 기술적 지표 계산
    logger.info("\n2. Calculating technical indicators...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)

    logger.info(f"Processed {len(df_processed)} candles")

    # 3. Regression XGBoost 초기화
    logger.info("\n3. Initializing REGRESSION XGBoost Trader...")

    trader = XGBoostTraderRegression(
        lookahead=48,                   # 4시간 (240분) ⭐
        long_threshold=0.015,           # 1.5% ⭐
        short_threshold=-0.015,         # -1.5% ⭐
        confidence_multiplier=0.5,      # 동적 임계값 조정
        model_params=None
    )

    # 4. 데이터 준비
    logger.info("\n4. Preparing train/val/test split...")

    train_df, val_df, test_df = trader.prepare_data(
        df_processed,
        train_ratio=0.7,
        val_ratio=0.15
    )

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # 5. SNR 계산
    logger.info("\n5. Validating SNR...")

    returns_5min = df_processed['close'].pct_change()
    std_5min = returns_5min.std()
    std_4h = std_5min * (48 ** 0.5)

    snr = 0.015 / std_4h
    logger.info(f"5분 표준편차: {std_5min*100:.4f}%")
    logger.info(f"4시간 예상 표준편차: {std_4h*100:.2f}%")
    logger.info(f"SNR (1.5% / {std_4h*100:.2f}%): {snr:.2f}")

    if snr > 2.0:
        logger.info(f"✅ SNR > 2.0 - 우수한 신호!")
    elif snr > 1.0:
        logger.info(f"⚠️ SNR > 1.0 - 신호가 노이즈보다 강함")
    else:
        logger.warning(f"❌ SNR < 1.0 - 신호보다 노이즈가 강함")

    # 6. 모델 훈련
    logger.info("\n6. Training REGRESSION XGBoost model...")
    logger.info("No SMOTE needed - regression naturally balanced!")

    train_stats = trader.train(
        train_df=train_df,
        val_df=val_df,
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose=True
    )

    logger.info(f"\nTraining Statistics:")
    logger.info(f"Best iteration: {train_stats['best_iteration']}")
    logger.info(f"Train RMSE: {train_stats['train_rmse']*100:.3f}%")
    logger.info(f"Val RMSE: {train_stats['val_rmse']*100:.3f}%")
    logger.info(f"Val MAE: {train_stats['val_mae']*100:.3f}%")
    logger.info(f"Val R²: {train_stats['val_r2']:.4f}")

    # 7. 평가
    logger.info("\n7. Evaluating on all sets...")

    logger.info("\n--- Training Set ---")
    train_results = trader.evaluate(train_df, name="Train")

    logger.info("\n--- Validation Set ---")
    val_results = trader.evaluate(val_df, name="Validation")

    logger.info("\n--- Test Set ---")
    test_results = trader.evaluate(test_df, name="Test")

    # 8. 백테스팅
    logger.info("\n8. Running REGRESSION backtests...")

    logger.info("\n--- Validation Backtest ---")
    val_signals, _ = trader.predict(val_df, use_confidence_multiplier=True)
    val_backtest = trader.backtest_fixed(
        val_df,
        val_signals,
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    logger.info("\n--- Test Backtest ---")
    test_signals, _ = trader.predict(test_df, use_confidence_multiplier=True)
    test_backtest = trader.backtest_fixed(
        test_df,
        test_signals,
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    # 9. 모델 저장
    logger.info("\n9. Saving REGRESSION model...")
    trader.save_model('xgboost_regression_v1')
    logger.info("Model saved successfully")

    # 10. 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info("\n📊 Regression Performance:")
    logger.info(f"Train RMSE: {train_results['rmse']*100:.3f}%")
    logger.info(f"Val RMSE:   {val_results['rmse']*100:.3f}%")
    logger.info(f"Test RMSE:  {test_results['rmse']*100:.3f}%")

    logger.info(f"\nTrain R²: {train_results['r2']:.4f}")
    logger.info(f"Val R²:   {val_results['r2']:.4f}")
    logger.info(f"Test R²:  {test_results['r2']:.4f}")

    logger.info("\n💰 REGRESSION Backtest Results:")
    logger.info(f"Validation Return: {val_backtest['total_return_pct']:.2f}%")
    logger.info(f"Test Return:       {test_backtest['total_return_pct']:.2f}%")

    logger.info(f"\nValidation Trades: {val_backtest['num_trades']}")
    logger.info(f"Test Trades:       {test_backtest['num_trades']}")

    logger.info(f"\nValidation Win Rate: {val_backtest['win_rate']*100:.1f}%")
    logger.info(f"Test Win Rate:       {test_backtest['win_rate']*100:.1f}%")

    # 11. 4-Way 비교
    logger.info("\n" + "=" * 80)
    logger.info("4-WAY COMPARISON: BUGGY vs FIXED vs IMPROVED vs REGRESSION")
    logger.info("=" * 80)

    logger.info("\n| Metric              | BUGGY      | FIXED     | IMPROVED  | REGRESSION | Winner |")
    logger.info("|---------------------|------------|-----------|-----------|------------|--------|")
    logger.info(f"| Model Type          | Classifier | Classifier| Classifier| Regressor  | ⭐ |")
    logger.info(f"| lookahead           | 5 (25min)  | 60 (5h)   | 288 (24h) | 48 (4h)    | ⭐ |")
    logger.info(f"| Class Imbalance     | Yes        | Yes       | SMOTE     | N/A        | ⭐ |")
    logger.info(f"| Test Return         | -1051.80%  | -2.05%    | -2.80%    | {test_backtest['total_return_pct']:+.2f}%  | {'🏆' if test_backtest['total_return_pct'] > -2.05 else '⚠️'} |")
    logger.info(f"| Test Win Rate       | 2.3%       | 0.0%      | 66.7%     | {test_backtest['win_rate']*100:.1f}%    | {'🏆' if test_backtest['win_rate'] > 0.667 else '⚠️'} |")
    logger.info(f"| Test Trades         | 867        | 1         | 9         | {test_backtest['num_trades']:4d}      | {'🏆' if 20 <= test_backtest['num_trades'] <= 100 else '⚠️'} |")
    logger.info(f"| Liquidated          | No         | False     | False     | {str(test_backtest['liquidated']):5s}     | {'✅' if not test_backtest['liquidated'] else '❌'} |")

    # 12. 회귀 모델 장점 분석
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION MODEL ADVANTAGES")
    logger.info("=" * 80)

    logger.info("\n✅ 근본적 개선:")
    logger.info("  1. 클래스 불균형 문제 소멸 (분류 → 회귀)")
    logger.info("  2. SMOTE 불필요 → 과적합 위험 감소")
    logger.info("  3. 실제 수익률 직접 예측")
    logger.info("  4. 동적 임계값 조정 가능")

    logger.info("\n📈 예측 품질:")
    logger.info(f"  - Direction Accuracy: {test_results['direction_accuracy']*100:.1f}%")
    logger.info(f"  - Signal Accuracy: {test_results['signal_accuracy']*100:.1f}%")
    logger.info(f"  - R² Score: {test_results['r2']:.4f}")

    logger.info("\n💼 거래 활성도:")
    if test_backtest['num_trades'] >= 20:
        logger.info(f"  ✅ {test_backtest['num_trades']} trades - 적절한 거래 빈도")
    elif test_backtest['num_trades'] >= 10:
        logger.info(f"  ⚠️ {test_backtest['num_trades']} trades - 거래 빈도 양호")
    else:
        logger.info(f"  ❌ {test_backtest['num_trades']} trades - 거래 빈도 낮음")

    logger.info("\n📊 수익성:")
    if test_backtest['total_return_pct'] > 5:
        logger.info(f"  🏆 {test_backtest['total_return_pct']:.2f}% - 우수한 성과!")
    elif test_backtest['total_return_pct'] > 0:
        logger.info(f"  ✅ {test_backtest['total_return_pct']:.2f}% - 수익 발생")
    else:
        logger.info(f"  ⚠️ {test_backtest['total_return_pct']:.2f}% - 손실 발생")

    logger.info("\n🎯 Win Rate:")
    if test_backtest['win_rate'] > 0.6:
        logger.info(f"  🏆 {test_backtest['win_rate']*100:.1f}% - 우수한 승률!")
    elif test_backtest['win_rate'] > 0.5:
        logger.info(f"  ✅ {test_backtest['win_rate']*100:.1f}% - 양호한 승률")
    else:
        logger.info(f"  ⚠️ {test_backtest['win_rate']*100:.1f}% - 개선 필요")

    # 13. 과적합 분석
    logger.info("\n" + "=" * 80)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("=" * 80)

    train_rmse = train_results['rmse']
    test_rmse = test_results['rmse']
    overfitting_ratio = train_rmse / test_rmse if test_rmse > 0 else 0

    logger.info(f"\nRMSE Ratio (Train/Test): {overfitting_ratio:.2f}x")
    if overfitting_ratio > 0.8 and overfitting_ratio < 1.2:
        logger.info("✅ 과적합 없음 - 우수한 일반화!")
    elif overfitting_ratio > 0.5:
        logger.info("⚠️ 약간의 과적합 - 허용 가능")
    else:
        logger.info("❌ 심각한 과적합 - 개선 필요")

    logger.info(f"\nR² Scores:")
    logger.info(f"  Train: {train_results['r2']:.4f}")
    logger.info(f"  Val:   {val_results['r2']:.4f}")
    logger.info(f"  Test:  {test_results['r2']:.4f}")

    # 14. 결과 저장
    results_dir = project_root / 'data' / 'trained_models' / 'xgboost_regression'
    results_file = results_dir / 'training_results.txt'

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("XGBoost REGRESSION Training Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"Model Type: Regression\n")
        f.write(f"lookahead: 48 (4 hours)\n")
        f.write(f"thresholds: -1.5% / +1.5%\n")
        f.write(f"SNR: {snr:.2f}\n\n")

        f.write("Regression Metrics:\n")
        f.write(f"Test RMSE: {test_results['rmse']*100:.3f}%\n")
        f.write(f"Test MAE: {test_results['mae']*100:.3f}%\n")
        f.write(f"Test R²: {test_results['r2']:.4f}\n")
        f.write(f"Direction Accuracy: {test_results['direction_accuracy']*100:.1f}%\n")
        f.write(f"Signal Accuracy: {test_results['signal_accuracy']*100:.1f}%\n\n")

        f.write("Backtest Results:\n")
        f.write(f"Validation Return: {val_backtest['total_return_pct']:.2f}%\n")
        f.write(f"Test Return:       {test_backtest['total_return_pct']:.2f}%\n")
        f.write(f"Validation Trades: {val_backtest['num_trades']}\n")
        f.write(f"Test Trades:       {test_backtest['num_trades']}\n")
        f.write(f"Validation Win Rate: {val_backtest['win_rate']*100:.1f}%\n")
        f.write(f"Test Win Rate:       {test_backtest['win_rate']*100:.1f}%\n")
        f.write(f"Validation Liquidated: {val_backtest['liquidated']}\n")
        f.write(f"Test Liquidated:       {test_backtest['liquidated']}\n")

    logger.info(f"\nResults saved to: {results_file}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ REGRESSION Training completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
