"""REGRESSION 모델 예측값 분포 분석

목표: 왜 모든 임계값에서 거래가 0회인지 진단
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from src.models.xgboost_trader_regression import XGBoostTraderRegression
from src.indicators.technical_indicators import TechnicalIndicators


def main():
    logger.info("="*80)
    logger.info("REGRESSION Prediction Distribution Analysis")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} candles")

    # 2. 기술적 지표
    logger.info("\n2. Calculating indicators...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)

    # 3. 모델 로드
    logger.info("\n3. Loading model...")
    trader = XGBoostTraderRegression(
        lookahead=48,
        long_threshold=0.006,  # 0.6%
        short_threshold=-0.006,
        confidence_multiplier=0.5
    )

    trader.load_model('xgboost_regression_v1')

    # 4. 데이터 준비
    train_df, val_df, test_df = trader.prepare_data(
        df_processed,
        train_ratio=0.7,
        val_ratio=0.15
    )

    # 5. 예측 수행 (동적 임계값 OFF)
    logger.info("\n4. Analyzing predictions (NO dynamic threshold)...")

    # Test set 예측
    X_test = test_df[trader.feature_columns]
    y_test = test_df['target'].values

    import xgboost as xgb
    dtest = xgb.DMatrix(X_test, label=y_test)
    predictions = trader.model.predict(dtest)

    logger.info(f"\n예측값 통계:")
    logger.info(f"  Mean: {predictions.mean()*100:.4f}%")
    logger.info(f"  Std: {predictions.std()*100:.4f}%")
    logger.info(f"  Min: {predictions.min()*100:.4f}%")
    logger.info(f"  Max: {predictions.max()*100:.4f}%")

    logger.info(f"\n실제값 통계:")
    logger.info(f"  Mean: {y_test.mean()*100:.4f}%")
    logger.info(f"  Std: {y_test.std()*100:.4f}%")
    logger.info(f"  Min: {y_test.min()*100:.4f}%")
    logger.info(f"  Max: {y_test.max()*100:.4f}%")

    # 6. 임계값별 신호 개수
    logger.info("\n5. Signal counts at different thresholds (STATIC):")

    for threshold in [0.003, 0.005, 0.006, 0.008, 0.010]:
        long_signals = np.sum(predictions > threshold)
        short_signals = np.sum(predictions < -threshold)
        hold_signals = len(predictions) - long_signals - short_signals

        logger.info(f"\nThreshold ±{threshold*100:.1f}%:")
        logger.info(f"  LONG:  {long_signals:4d} ({long_signals/len(predictions)*100:.1f}%)")
        logger.info(f"  HOLD:  {hold_signals:4d} ({hold_signals/len(predictions)*100:.1f}%)")
        logger.info(f"  SHORT: {short_signals:4d} ({short_signals/len(predictions)*100:.1f}%)")

    # 7. 동적 임계값 분석
    logger.info("\n6. Analyzing DYNAMIC threshold logic...")

    # 동적 임계값 계산 재현
    threshold = 0.006
    confidence_multiplier = 0.5

    dynamic_long_thresholds = []
    dynamic_short_thresholds = []

    for pred in predictions:
        abs_pred = abs(pred)
        dynamic_long_threshold = threshold * (1 - confidence_multiplier * min(abs_pred / 0.05, 1.0))
        dynamic_short_threshold = -threshold * (1 - confidence_multiplier * min(abs_pred / 0.05, 1.0))

        dynamic_long_thresholds.append(dynamic_long_threshold)
        dynamic_short_thresholds.append(dynamic_short_threshold)

    dynamic_long_thresholds = np.array(dynamic_long_thresholds)
    dynamic_short_thresholds = np.array(dynamic_short_thresholds)

    logger.info(f"\nDynamic LONG Thresholds:")
    logger.info(f"  Mean: {dynamic_long_thresholds.mean()*100:.4f}%")
    logger.info(f"  Min: {dynamic_long_thresholds.min()*100:.4f}%")
    logger.info(f"  Max: {dynamic_long_thresholds.max()*100:.4f}%")

    logger.info(f"\nDynamic SHORT Thresholds:")
    logger.info(f"  Mean: {dynamic_short_thresholds.mean()*100:.4f}%")
    logger.info(f"  Min: {dynamic_short_thresholds.min()*100:.4f}%")
    logger.info(f"  Max: {dynamic_short_thresholds.max()*100:.4f}%")

    # 동적 임계값으로 신호 계산
    signals_dynamic = np.zeros(len(predictions))
    for i, pred in enumerate(predictions):
        if pred > dynamic_long_thresholds[i]:
            signals_dynamic[i] = 1  # LONG
        elif pred < dynamic_short_thresholds[i]:
            signals_dynamic[i] = -1  # SHORT
        else:
            signals_dynamic[i] = 0  # HOLD

    unique, counts = np.unique(signals_dynamic, return_counts=True)
    signal_dist = dict(zip(unique, counts))

    logger.info(f"\nDynamic Threshold Signals:")
    for sig in [-1, 0, 1]:
        count = signal_dist.get(sig, 0)
        sig_name = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}[sig]
        logger.info(f"  {sig_name}: {count:4d} ({count/len(predictions)*100:.1f}%)")

    # 8. 예측값 분포 분석
    logger.info("\n7. Prediction distribution analysis:")

    percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(predictions, p)
        logger.info(f"  {p:2d}th percentile: {val*100:+.4f}%")

    # 9. 근본 원인 진단
    logger.info("\n" + "="*80)
    logger.info("ROOT CAUSE DIAGNOSIS")
    logger.info("="*80)

    max_pred = predictions.max()
    min_pred = predictions.min()

    logger.info(f"\n문제 1: 예측 범위 제한")
    logger.info(f"  Max prediction: {max_pred*100:.4f}%")
    logger.info(f"  Threshold 0.6%: {max_pred < 0.006}")
    logger.info(f"  → {'❌ 최대 예측값도 0.6% 이하!' if max_pred < 0.006 else '✅ 일부 예측값이 0.6% 초과'}")

    logger.info(f"\n  Min prediction: {min_pred*100:.4f}%")
    logger.info(f"  Threshold -0.6%: {min_pred > -0.006}")
    logger.info(f"  → {'❌ 최소 예측값도 -0.6% 이상!' if min_pred > -0.006 else '✅ 일부 예측값이 -0.6% 미만'}")

    logger.info(f"\n문제 2: 예측값 집중도")
    std_pred = predictions.std()
    logger.info(f"  Prediction std: {std_pred*100:.4f}%")
    logger.info(f"  Target std: {y_test.std()*100:.3f}%")
    logger.info(f"  Ratio: {std_pred / y_test.std():.2f}x")
    logger.info(f"  → {'❌ 예측값이 과도하게 수축됨!' if std_pred < y_test.std() * 0.5 else '✅ 예측 범위 적절'}")

    logger.info(f"\n문제 3: R² Score")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)
    logger.info(f"  R² = {r2:.4f}")
    logger.info(f"  → {'❌ 모델이 평균값만 예측' if r2 < 0 else '✅ 모델이 패턴 학습' if r2 > 0.1 else '⚠️ 약한 예측력'}")

    # 10. 해결책 제안
    logger.info("\n" + "="*80)
    logger.info("SOLUTIONS")
    logger.info("="*80)

    if max_pred < 0.006:
        logger.info("\n✅ Solution 1: Percentile-based Thresholds")
        long_threshold_percentile = np.percentile(predictions, 90)
        short_threshold_percentile = np.percentile(predictions, 10)

        logger.info(f"  Use 90th percentile as LONG threshold: {long_threshold_percentile*100:.4f}%")
        logger.info(f"  Use 10th percentile as SHORT threshold: {short_threshold_percentile*100:.4f}%")

        # 테스트
        signals_percentile = np.where(
            predictions > long_threshold_percentile, 1,
            np.where(predictions < short_threshold_percentile, -1, 0)
        )

        unique, counts = np.unique(signals_percentile, return_counts=True)
        for sig, count in zip(unique, counts):
            sig_name = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}[sig]
            logger.info(f"    {sig_name}: {count} ({count/len(predictions)*100:.1f}%)")

    logger.info("\n✅ Solution 2: Re-train with different objective")
    logger.info("  - Current: reg:squarederror")
    logger.info("  - Try: reg:pseudohubererror (robust to outliers)")
    logger.info("  - Try: Different loss function")

    logger.info("\n✅ Solution 3: Feature engineering")
    logger.info("  - Add volatility regime features")
    logger.info("  - Add trend strength features")
    logger.info("  - Multi-timeframe aggregation")

    logger.info("\n" + "="*80)
    logger.info("✅ Debug completed!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
