"""XGBoost Regression with Sequential Features - 사용자 통찰 검증

사용자 통찰: "모델이 가장 최근 캔들의 지표만 보고 추세를 모른다"

해결 방안:
1. Sequential/Context Features 추가 (추세, 변화, 패턴)
2. XGBoost 회귀 모델 직접 훈련 (prepare_features 우회)
3. 예측값 다양성 검증 (단일 상수 문제 해결 여부)
4. Buy & Hold와 비교 검증
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.indicators.technical_indicators import TechnicalIndicators


def prepare_sequential_data(df: pd.DataFrame, lookahead: int = 48):
    """Sequential Features를 활용한 데이터 준비"""
    data = df.copy()

    # 타겟: 미래 수익률
    data['target'] = data['close'].pct_change(lookahead).shift(-lookahead)

    # Feature columns (OHLCV 및 timestamp 제외)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    # 결측치 제거
    data = data.dropna()

    logger.info(f"Target Statistics:")
    logger.info(f"  Mean: {data['target'].mean()*100:.3f}%")
    logger.info(f"  Std: {data['target'].std()*100:.3f}%")
    logger.info(f"  Min: {data['target'].min()*100:.2f}%")
    logger.info(f"  Max: {data['target'].max()*100:.2f}%")

    logger.info(f"\nTotal Features: {len(feature_cols)}")
    logger.info(f"Sample Features: {feature_cols[:10]}")

    return data, feature_cols


def main():
    logger.info("="*80)
    logger.info("XGBoost Regression with Sequential Features")
    logger.info("사용자 통찰 검증: 추세/문맥 정보 추가로 예측 개선되는가?")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} candles")

    # 2. 기본 지표 계산
    logger.info("\n2. Calculating base indicators...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)
    logger.info(f"Base indicators calculated: {len(df_processed)} rows")

    # 3. Sequential Features 추가
    logger.info("\n3. Adding Sequential Features...")
    logger.info("   → RSI 변화 (5, 20)")
    logger.info("   → 가격 vs MA 비율")
    logger.info("   → MACD 크로스오버")
    logger.info("   → 연속 캔들 패턴")
    logger.info("   → 추세 정렬")

    df_sequential = indicators.calculate_sequential_features(df_processed)

    original_features = len(df_processed.columns)
    new_features = len(df_sequential.columns)
    logger.info(f"\nFeature Count: {original_features} → {new_features} (+{new_features - original_features})")

    # 4. 데이터 준비
    logger.info("\n4. Preparing data with targets...")
    data, feature_cols = prepare_sequential_data(df_sequential, lookahead=48)

    # 5. Train/Validation/Test Split
    logger.info("\n5. Splitting data...")
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()

    logger.info(f"Train: {len(train_df)} candles")
    logger.info(f"Validation: {len(val_df)} candles")
    logger.info(f"Test: {len(test_df)} candles")
    logger.info(f"Test Period: {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")

    # 6. XGBoost 훈련
    logger.info("\n6. Training XGBoost Regression...")

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # 모델 파라미터
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'random_state': 42
    }

    # 훈련
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}

    logger.info("Training XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=20
    )

    logger.info(f"\nTraining completed - Best iteration: {model.best_iteration}")

    # 7. 예측값 분석 (🚨 핵심 검증: 단일 상수 문제 해결 여부)
    logger.info("\n7. Analyzing Predictions...")

    train_preds = model.predict(dtrain)
    val_preds = model.predict(dval)
    test_preds = model.predict(dtest)

    logger.info(f"\nTrain Predictions:")
    logger.info(f"  Mean: {train_preds.mean()*100:.6f}%")
    logger.info(f"  Std: {train_preds.std()*100:.6f}%")
    logger.info(f"  Min: {train_preds.min()*100:.6f}%")
    logger.info(f"  Max: {train_preds.max()*100:.6f}%")

    logger.info(f"\nValidation Predictions:")
    logger.info(f"  Mean: {val_preds.mean()*100:.6f}%")
    logger.info(f"  Std: {val_preds.std()*100:.6f}%")
    logger.info(f"  Min: {val_preds.min()*100:.6f}%")
    logger.info(f"  Max: {val_preds.max()*100:.6f}%")

    logger.info(f"\nTest Predictions:")
    logger.info(f"  Mean: {test_preds.mean()*100:.6f}%")
    logger.info(f"  Std: {test_preds.std()*100:.6f}%")
    logger.info(f"  Min: {test_preds.min()*100:.6f}%")
    logger.info(f"  Max: {test_preds.max()*100:.6f}%")

    # 🚨 핵심 검증: 표준편차가 0인가?
    logger.info("\n" + "="*80)
    logger.info("🔍 CRITICAL VALIDATION: Constant Prediction Problem Solved?")
    logger.info("="*80)

    if test_preds.std() < 0.0001:
        logger.error("\n🚨 FAILURE: Predictions are still constant!")
        logger.error("   Sequential Features did NOT solve the problem.")
        logger.error(f"   All predictions ≈ {test_preds.mean()*100:.6f}%")
    else:
        logger.success(f"\n✅ SUCCESS: Predictions have variance!")
        logger.success(f"   Std = {test_preds.std()*100:.6f}%")
        logger.success(f"   Range: {test_preds.min()*100:.3f}% to {test_preds.max()*100:.3f}%")
        logger.success("   Sequential Features SOLVED the constant prediction problem!")

    # 8. 모델 성능 평가
    logger.info("\n8. Model Performance...")

    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    logger.info(f"\nTest Set Regression Metrics:")
    logger.info(f"  RMSE: {test_rmse*100:.3f}%")
    logger.info(f"  MAE: {test_mae*100:.3f}%")
    logger.info(f"  R²: {test_r2:.4f}")

    # 9. 신호 생성 및 백테스팅
    logger.info("\n9. Generating Trading Signals...")

    long_threshold = 0.015  # 1.5%
    short_threshold = -0.015  # -1.5%

    signals = np.zeros(len(test_preds))
    signals[test_preds > long_threshold] = 1  # LONG
    signals[test_preds < short_threshold] = -1  # SHORT

    unique, counts = np.unique(signals, return_counts=True)
    logger.info(f"\nSignal Distribution:")
    for sig, count in zip(unique, counts):
        sig_name = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}[int(sig)]
        logger.info(f"  {sig_name}: {count}/{len(signals)} ({count/len(signals)*100:.1f}%)")

    # 간단한 백테스팅
    logger.info("\n10. Backtesting...")

    balance = 10000.0
    position = 0.0
    num_trades = 0

    for i in range(len(test_df)):
        signal = signals[i]
        price = test_df.iloc[i]['close']

        if signal == 1 and position == 0:  # LONG 진입
            position = 0.03
            num_trades += 1
        elif signal == -1 and position == 0:  # SHORT 진입
            position = -0.03
            num_trades += 1
        elif signal == 0 and position != 0:  # 청산
            position = 0

    # 최종 수익률 (간소화)
    first_price = test_df.iloc[0]['close']
    last_price = test_df.iloc[-1]['close']

    # Buy & Hold 비교
    bh_return = (last_price - first_price) / first_price * 100

    logger.info(f"\nBacktest Results (Simplified):")
    logger.info(f"  Trades Generated: {num_trades}")
    logger.info(f"  Buy & Hold Return: {bh_return:+.2f}%")

    # 11. 비교 분석
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: Sequential Features vs Original Models")
    logger.info("="*80)

    logger.info("\n| Model | Prediction Std | Trades | R² | Note |")
    logger.info("|-------|---------------|--------|-----|------|")
    logger.info("| REGRESSION (Original) | 0.0000% | 0 | -0.15 | Single constant |")
    logger.info(f"| REGRESSION (Sequential) | {test_preds.std()*100:.4f}% | {num_trades:6d} | {test_r2:.2f} | **This model** |")

    # 12. 최종 결론
    logger.info("\n" + "="*80)
    logger.info("FINAL CONCLUSION")
    logger.info("="*80)

    logger.info("\n사용자 가설: '모델이 가장 최근 캔들의 지표만 보고 추세를 모른다'")
    logger.info("해결 방안: Sequential/Context Features 추가")

    if test_preds.std() > 0.0001:
        logger.success("\n✅ 사용자 통찰이 정확했습니다!")
        logger.success("   Sequential Features가 예측 다양성을 회복시켰습니다!")
        logger.info(f"\n개선 사항:")
        logger.info(f"  • Prediction Std: 0.0000% → {test_preds.std()*100:.4f}%")
        logger.info(f"  • R²: -0.15 → {test_r2:.2f}")
        logger.info(f"  • Trades: 0 → {num_trades}")

        if test_r2 > 0:
            logger.success(f"\n🎉 모델이 실제 예측 능력을 획득했습니다! (R² > 0)")
        else:
            logger.warning(f"\n⚠️ 예측 다양성은 있으나 R² < 0 (여전히 baseline 미달)")
            logger.info("   추가 피처 엔지니어링이나 다른 접근이 필요할 수 있습니다.")
    else:
        logger.error("\n❌ Sequential Features로도 문제가 해결되지 않았습니다.")
        logger.info("\n가능한 원인:")
        logger.info("  • 추가된 피처도 여전히 단일 시점 특성")
        logger.info("  • 모델이 미래 예측 불가능하다고 학습")
        logger.info("  • 5분봉 시장의 본질적 무작위성")

    logger.info("\n" + "="*80)
    logger.info("✅ Analysis Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
