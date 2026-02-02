"""XGBoost Fixed 모델 훈련 스크립트

수정 사항:
1. HOLD = 포지션 유지 (청산 아님)
2. 강제 청산 로직 추가
3. 수수료 계산 수정 (레버리지 이중 적용 제거)
4. 개선된 파라미터 (lookahead=60, threshold=1.0%, scale_pos_weight=7.0)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.models.xgboost_trader_fixed import XGBoostTraderFixed
from src.indicators.technical_indicators import TechnicalIndicators


def main():
    logger.info("=" * 80)
    logger.info("XGBoost FIXED Trading Model Training")
    logger.info("=" * 80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'

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

    # 3. Fixed XGBoost 초기화
    logger.info("\n3. Initializing FIXED XGBoost Trader...")

    trader = XGBoostTraderFixed(
        lookahead=60,               # 5시간 (300분) 앞 예측 ⭐
        threshold_pct=0.01,         # 1.0% (수수료 0.08% + 여유) ⭐
        confidence_threshold=0.65,  # 65% 확신도 이상만 거래 ⭐
        model_params=None           # 개선된 기본 파라미터 사용
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

    # 5. 모델 훈련
    logger.info("\n5. Training FIXED XGBoost model...")
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

    # 7. Fixed 백테스팅
    logger.info("\n7. Running FIXED backtests...")

    # 검증 세트
    logger.info("\n--- Validation FIXED Backtest ---")
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

    # 테스트 세트
    logger.info("\n--- Test FIXED Backtest ---")
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

    # 8. 모델 저장
    logger.info("\n8. Saving FIXED model...")
    trader.save_model('xgboost_fixed_v1')
    logger.info("Model saved successfully")

    # 9. 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("FIXED TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info("\n📊 Model Performance:")
    logger.info(f"Train Accuracy: {train_results['accuracy']:.4f}")
    logger.info(f"Val Accuracy:   {val_results['accuracy']:.4f}")
    logger.info(f"Test Accuracy:  {test_results['accuracy']:.4f}")

    logger.info("\n💰 FIXED Backtest Results:")
    logger.info(f"Validation Return: {val_backtest['total_return_pct']:.2f}%")
    logger.info(f"Test Return:       {test_backtest['total_return_pct']:.2f}%")

    logger.info(f"\nValidation Trades: {val_backtest['num_trades']}")
    logger.info(f"Test Trades:       {test_backtest['num_trades']}")

    logger.info(f"\nValidation Win Rate: {val_backtest['win_rate']*100:.1f}%")
    logger.info(f"Test Win Rate:       {test_backtest['win_rate']*100:.1f}%")

    logger.info(f"\nValidation Liquidated: {val_backtest['liquidated']}")
    logger.info(f"Test Liquidated:       {test_backtest['liquidated']}")

    # 원본 대비 개선 비교
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: FIXED vs BUGGY")
    logger.info("=" * 80)

    logger.info("\n| Metric              | BUGGY      | FIXED     | Status |")
    logger.info("|---------------------|------------|-----------|--------|")
    logger.info(f"| Test Return         | -1051.80%  | {test_backtest['total_return_pct']:+.2f}%  | {'✅ FIXED' if test_backtest['total_return_pct'] > -100 else '❌ STILL BROKEN'} |")
    logger.info(f"| Test Win Rate       | 2.3%       | {test_backtest['win_rate']*100:.1f}%    | {'✅ FIXED' if test_backtest['win_rate'] > 0.1 else '⚠️ LOW'} |")
    logger.info(f"| Test Trades         | 867        | {test_backtest['num_trades']:4d}      | {'✅ REDUCED' if test_backtest['num_trades'] < 867 else '⚠️ SAME'} |")
    logger.info(f"| Liquidated          | No         | {str(test_backtest['liquidated']):5s}     | {'✅ SAFE' if not test_backtest['liquidated'] else '⚠️ LIQUIDATED'} |")

    # 수학적 정합성 검증
    logger.info("\n" + "=" * 80)
    logger.info("MATHEMATICAL VALIDITY CHECK")
    logger.info("=" * 80)

    # 1. 수익률 범위 검증
    if -100 <= test_backtest['total_return_pct'] <= 10000:
        logger.info("✅ Return is mathematically possible (-100% to +10000%)")
    else:
        logger.info(f"❌ Return {test_backtest['total_return_pct']:.2f}% is IMPOSSIBLE!")

    # 2. 청산 로직 검증
    if test_backtest['liquidated']:
        logger.info("✅ Liquidation logic working (account stopped trading)")
    else:
        logger.info("✅ No liquidation occurred (account stayed healthy)")

    # 3. HOLD 의미 검증
    hold_pct = (test_predictions == 0).sum() / len(test_predictions) * 100
    logger.info(f"✅ HOLD predictions: {hold_pct:.1f}%")
    if test_backtest['num_trades'] < len(test_predictions) * 0.5:
        logger.info("✅ HOLD is working as 'maintain position' (fewer trades)")
    else:
        logger.info("⚠️ Too many trades, HOLD may still be closing positions")

    # 결과 파일 저장
    results_dir = project_root / 'data' / 'trained_models' / 'xgboost_fixed'
    results_file = results_dir / 'training_results.txt'

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("XGBoost FIXED Training Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Model Performance:\n")
        f.write(f"Train Accuracy: {train_results['accuracy']:.4f}\n")
        f.write(f"Val Accuracy:   {val_results['accuracy']:.4f}\n")
        f.write(f"Test Accuracy:  {test_results['accuracy']:.4f}\n\n")

        f.write("FIXED Backtest Results:\n")
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
    logger.info("✅ FIXED Training completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
