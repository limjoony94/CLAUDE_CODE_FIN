"""XGBoost Improved 모델 훈련 스크립트

Phase 1B + Phase 2A:
1. lookahead = 288 (24시간) - SNR 개선
2. threshold = 2.0% - 수수료 대비 충분한 수익
3. SMOTE - 클래스 불균형 해결
4. 모든 백테스팅 버그 수정 유지
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.models.xgboost_trader_improved import XGBoostTraderImproved
from src.indicators.technical_indicators import TechnicalIndicators


def main():
    logger.info("=" * 80)
    logger.info("XGBoost IMPROVED Trading Model Training")
    logger.info("Phase 1B (lookahead=288) + Phase 2A (SMOTE)")
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
    logger.info(f"Date range: {df_processed['timestamp'].iloc[0]} to {df_processed['timestamp'].iloc[-1]}")

    # 3. Improved XGBoost 초기화
    logger.info("\n3. Initializing IMPROVED XGBoost Trader...")

    trader = XGBoostTraderImproved(
        lookahead=288,              # 24시간 (1440분) ⭐
        threshold_pct=0.02,         # 2.0% ⭐
        confidence_threshold=0.60,  # 60% 확신도
        use_smote=True,             # SMOTE 활성화 ⭐
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

    # 5. SNR 계산 검증
    logger.info("\n5. Validating SNR improvement...")

    returns_5min = df_processed['close'].pct_change()
    std_5min = returns_5min.std()
    std_24h = std_5min * (288 ** 0.5)

    snr = 0.02 / std_24h
    logger.info(f"5분 표준편차: {std_5min*100:.4f}%")
    logger.info(f"24시간 예상 표준편차: {std_24h*100:.2f}%")
    logger.info(f"SNR (2.0% / {std_24h*100:.2f}%): {snr:.2f}")

    if snr > 2.0:
        logger.info(f"✅ SNR > 2.0 - 학습 가능한 신호!")
    elif snr > 1.0:
        logger.info(f"⚠️ SNR > 1.0 - 신호가 노이즈보다 강함")
    else:
        logger.warning(f"❌ SNR < 1.0 - 신호보다 노이즈가 강함")

    # 6. 모델 훈련 (SMOTE 포함)
    logger.info("\n6. Training IMPROVED XGBoost model with SMOTE...")
    logger.info("This may take 5-30 minutes...")

    train_stats = trader.train(
        train_df=train_df,
        val_df=val_df,
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose=True
    )

    logger.info(f"\nTraining Statistics:")
    logger.info(f"Best iteration: {train_stats['best_iteration']}")
    logger.info(f"Train loss: {train_stats['train_loss']:.4f}")
    logger.info(f"Validation loss: {train_stats['val_loss']:.4f}")
    logger.info(f"Validation accuracy: {train_stats['val_accuracy']:.4f}")

    # 7. 평가
    logger.info("\n7. Evaluating on all sets...")

    logger.info("\n--- Training Set ---")
    train_results = trader.evaluate(train_df, name="Train")

    logger.info("\n--- Validation Set ---")
    val_results = trader.evaluate(val_df, name="Validation")

    logger.info("\n--- Test Set ---")
    test_results = trader.evaluate(test_df, name="Test")

    # 8. 백테스팅
    logger.info("\n8. Running IMPROVED backtests...")

    logger.info("\n--- Validation Backtest ---")
    val_predictions, _ = trader.predict(val_df, use_confidence_threshold=True)
    val_backtest = trader.backtest_fixed(
        val_df,
        val_predictions,
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    logger.info("\n--- Test Backtest ---")
    test_predictions, _ = trader.predict(test_df, use_confidence_threshold=True)
    test_backtest = trader.backtest_fixed(
        test_df,
        test_predictions,
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    # 9. 모델 저장
    logger.info("\n9. Saving IMPROVED model...")
    trader.save_model('xgboost_improved_v1')
    logger.info("Model saved successfully")

    # 10. 결과 요약 및 비교
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVED TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info("\n📊 Model Performance:")
    logger.info(f"Train Accuracy: {train_results['accuracy']:.4f}")
    logger.info(f"Val Accuracy:   {val_results['accuracy']:.4f}")
    logger.info(f"Test Accuracy:  {test_results['accuracy']:.4f}")

    logger.info("\n💰 IMPROVED Backtest Results:")
    logger.info(f"Validation Return: {val_backtest['total_return_pct']:.2f}%")
    logger.info(f"Test Return:       {test_backtest['total_return_pct']:.2f}%")

    logger.info(f"\nValidation Trades: {val_backtest['num_trades']}")
    logger.info(f"Test Trades:       {test_backtest['num_trades']}")

    logger.info(f"\nValidation Win Rate: {val_backtest['win_rate']*100:.1f}%")
    logger.info(f"Test Win Rate:       {test_backtest['win_rate']*100:.1f}%")

    # 11. 3-Way 비교
    logger.info("\n" + "=" * 80)
    logger.info("3-WAY COMPARISON: BUGGY vs FIXED vs IMPROVED")
    logger.info("=" * 80)

    logger.info("\n| Metric              | BUGGY      | FIXED     | IMPROVED  | Status |")
    logger.info("|---------------------|------------|-----------|-----------|--------|")
    logger.info(f"| lookahead           | 5 (25min)  | 60 (5h)   | 288 (24h) | ✅ |")
    logger.info(f"| threshold           | 0.2%       | 1.0%      | 2.0%      | ✅ |")
    logger.info(f"| SMOTE               | No         | No        | Yes       | ✅ |")
    logger.info(f"| Test Return         | -1051.80%  | -2.05%    | {test_backtest['total_return_pct']:+.2f}%  | {'✅' if test_backtest['total_return_pct'] > -2.05 else '⚠️'} |")
    logger.info(f"| Test Win Rate       | 2.3%       | 0.0%      | {test_backtest['win_rate']*100:.1f}%    | {'✅' if test_backtest['win_rate'] > 0.2 else '⚠️'} |")
    logger.info(f"| Test Trades         | 867        | 1         | {test_backtest['num_trades']:4d}      | {'✅' if 10 <= test_backtest['num_trades'] <= 200 else '⚠️'} |")
    logger.info(f"| Liquidated          | No         | False     | {str(test_backtest['liquidated']):5s}     | {'✅' if not test_backtest['liquidated'] else '❌'} |")

    # 12. 개선 효과 분석
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("=" * 80)

    logger.info("\n📈 Phase 1B (lookahead=288) 효과:")
    logger.info(f"  - SNR: {snr:.2f} ({'✅ >2.0' if snr > 2.0 else '⚠️ >1.0' if snr > 1.0 else '❌ <1.0'})")
    logger.info(f"  - 신호 품질 개선으로 예측 가능성 향상")

    logger.info("\n🎯 Phase 2A (SMOTE) 효과:")
    logger.info(f"  - 클래스 불균형 완화")
    logger.info(f"  - LONG/SHORT 예측 증가 기대")

    logger.info("\n💼 거래 활성도:")
    if test_backtest['num_trades'] > 10:
        logger.info(f"  ✅ {test_backtest['num_trades']} trades - 적절한 거래 빈도")
    elif test_backtest['num_trades'] > 1:
        logger.info(f"  ⚠️ {test_backtest['num_trades']} trades - 거래 빈도 낮음")
    else:
        logger.info(f"  ❌ {test_backtest['num_trades']} trades - 너무 보수적")

    logger.info("\n📊 수익성:")
    if test_backtest['total_return_pct'] > 5:
        logger.info(f"  ✅ {test_backtest['total_return_pct']:.2f}% - 우수한 성과")
    elif test_backtest['total_return_pct'] > 0:
        logger.info(f"  ⚠️ {test_backtest['total_return_pct']:.2f}% - 수익 발생하나 개선 필요")
    else:
        logger.info(f"  ❌ {test_backtest['total_return_pct']:.2f}% - 손실 발생")

    # 13. 결과 저장
    results_dir = project_root / 'data' / 'trained_models' / 'xgboost_improved'
    results_file = results_dir / 'training_results.txt'

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("XGBoost IMPROVED Training Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"lookahead: 288 (24 hours)\n")
        f.write(f"threshold: 2.0%\n")
        f.write(f"SMOTE: Enabled\n")
        f.write(f"SNR: {snr:.2f}\n\n")

        f.write("Model Performance:\n")
        f.write(f"Train Accuracy: {train_results['accuracy']:.4f}\n")
        f.write(f"Val Accuracy:   {val_results['accuracy']:.4f}\n")
        f.write(f"Test Accuracy:  {test_results['accuracy']:.4f}\n\n")

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
    logger.info("✅ IMPROVED Training completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
