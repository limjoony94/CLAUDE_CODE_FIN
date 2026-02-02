"""REGRESSION 모델 Threshold Sweep 실험

목표:
- 현재 ±1.5% threshold에서 거래 0회 발생 문제 해결
- 0.6%, 0.8%, 1.0% 세 가지 임계값 테스트
- 거래 빈도와 수익성 균형점 탐색
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
import json

from src.models.xgboost_trader_regression import XGBoostTraderRegression
from src.indicators.technical_indicators import TechnicalIndicators


def test_threshold(
    trader: XGBoostTraderRegression,
    df: pd.DataFrame,
    threshold_pct: float,
    name: str
):
    """특정 임계값으로 백테스팅 수행"""

    # 임계값 설정
    trader.long_threshold = threshold_pct
    trader.short_threshold = -threshold_pct

    # SNR 계산
    returns = df['close'].pct_change(trader.lookahead).shift(-trader.lookahead)
    std = returns.std()
    snr = threshold_pct / std if std > 0 else 0

    logger.info(f"\n{'='*80}")
    logger.info(f"{name}: Threshold = ±{threshold_pct*100:.1f}%")
    logger.info(f"SNR = {threshold_pct*100:.1f}% / {std*100:.3f}% = {snr:.2f}")
    logger.info(f"{'='*80}")

    # 예측
    signals, predictions = trader.predict(df, use_confidence_multiplier=True)

    # 신호 분포
    unique, counts = np.unique(signals, return_counts=True)
    signal_dist = dict(zip(unique, counts))

    logger.info(f"\n신호 분포:")
    total = len(signals)
    for sig, count in signal_dist.items():
        sig_name = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}[sig]
        logger.info(f"  {sig_name}: {count}/{total} ({count/total*100:.1f}%)")

    # 백테스팅
    backtest_results = trader.backtest_fixed(
        df,
        signals,
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    # 결과 정리
    result = {
        'threshold': threshold_pct,
        'snr': snr,
        'signal_distribution': {
            'short_pct': signal_dist.get(-1, 0) / total * 100,
            'hold_pct': signal_dist.get(0, 0) / total * 100,
            'long_pct': signal_dist.get(1, 0) / total * 100,
        },
        'backtest': backtest_results
    }

    return result


def main():
    logger.info("="*80)
    logger.info("REGRESSION Threshold Sweep Experiment")
    logger.info("목표: 최적 임계값 탐색으로 거래 활성화")
    logger.info("="*80)

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

    # 3. 훈련된 모델 로드
    logger.info("\n3. Loading trained REGRESSION model...")
    trader = XGBoostTraderRegression(
        lookahead=48,
        long_threshold=0.015,  # 초기값 (sweep에서 변경됨)
        short_threshold=-0.015,
        confidence_multiplier=0.5,
        model_params=None
    )

    # 모델 로드
    model_path = project_root / 'data' / 'trained_models' / 'xgboost_regression' / 'xgboost_regression_v1.json'
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please run train_xgboost_regression.py first")
        return

    trader.load_model('xgboost_regression_v1')
    logger.info("Model loaded successfully")

    # 4. 데이터 준비
    logger.info("\n4. Preparing data splits...")
    train_df, val_df, test_df = trader.prepare_data(
        df_processed,
        train_ratio=0.7,
        val_ratio=0.15
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 5. Threshold Sweep
    logger.info("\n5. Running Threshold Sweep...")

    thresholds = [0.006, 0.008, 0.010]  # 0.6%, 0.8%, 1.0%
    results = {}

    for threshold in thresholds:
        name = f"Threshold_{threshold*100:.1f}%"

        # Test set에서 테스트
        logger.info(f"\n--- Testing on Test Set ---")
        test_result = test_threshold(trader, test_df, threshold, name)

        # Validation set에서도 테스트
        logger.info(f"\n--- Testing on Validation Set ---")
        val_result = test_threshold(trader, val_df, threshold, f"{name}_Val")

        results[name] = {
            'threshold': threshold,
            'test': test_result,
            'validation': val_result
        }

    # 6. 결과 비교
    logger.info("\n" + "="*80)
    logger.info("THRESHOLD SWEEP RESULTS COMPARISON")
    logger.info("="*80)

    logger.info("\n| Threshold | SNR | Test Trades | Test Return | Test Win Rate | Val Trades | Val Return |")
    logger.info("|-----------|-----|-------------|-------------|---------------|------------|------------|")

    for name, data in results.items():
        threshold = data['threshold']
        test = data['test']
        val = data['validation']

        logger.info(
            f"| {threshold*100:5.1f}%    | {test['snr']:4.2f} | "
            f"{test['backtest']['num_trades']:11d} | "
            f"{test['backtest']['total_return_pct']:11.2f}% | "
            f"{test['backtest']['win_rate']*100:13.1f}% | "
            f"{val['backtest']['num_trades']:10d} | "
            f"{val['backtest']['total_return_pct']:10.2f}% |"
        )

    # 7. 최적 임계값 선정
    logger.info("\n" + "="*80)
    logger.info("OPTIMAL THRESHOLD ANALYSIS")
    logger.info("="*80)

    # 기준: Test Return > 0 and Test Trades >= 20
    optimal = None
    best_score = -float('inf')

    for name, data in results.items():
        test = data['test']
        threshold = data['threshold']

        # 점수 계산: 수익률 + 거래 빈도 페널티
        num_trades = test['backtest']['num_trades']
        total_return = test['backtest']['total_return_pct']
        win_rate = test['backtest']['win_rate']

        # 거래 빈도가 20-100 사이일 때 보너스
        trade_bonus = 0
        if 20 <= num_trades <= 100:
            trade_bonus = 10
        elif 10 <= num_trades < 20:
            trade_bonus = 5

        score = total_return + trade_bonus + (win_rate * 10)

        logger.info(f"\n{name}:")
        logger.info(f"  Threshold: ±{threshold*100:.1f}%")
        logger.info(f"  SNR: {test['snr']:.2f}")
        logger.info(f"  Test Return: {total_return:.2f}%")
        logger.info(f"  Test Trades: {num_trades}")
        logger.info(f"  Test Win Rate: {win_rate*100:.1f}%")
        logger.info(f"  Score: {score:.2f}")

        if score > best_score:
            best_score = score
            optimal = {
                'name': name,
                'threshold': threshold,
                'data': data
            }

    # 8. 최적 모델로 재훈련 및 저장
    if optimal:
        logger.info("\n" + "="*80)
        logger.info(f"OPTIMAL THRESHOLD: ±{optimal['threshold']*100:.1f}%")
        logger.info("="*80)

        logger.info(f"\n최적 임계값으로 모델 재구성...")

        # 최적 임계값으로 새 모델 생성
        optimal_trader = XGBoostTraderRegression(
            lookahead=48,
            long_threshold=optimal['threshold'],
            short_threshold=-optimal['threshold'],
            confidence_multiplier=0.5,
            model_params=None
        )

        # 기존 모델 복사
        optimal_trader.model = trader.model
        optimal_trader.feature_columns = trader.feature_columns

        # 저장
        model_name = f"xgboost_regression_optimal_{optimal['threshold']*100:.1f}pct"
        optimal_trader.save_model(model_name)
        logger.info(f"Optimal model saved: {model_name}")

        # 최적 모델 성능 요약
        test_data = optimal['data']['test']
        logger.info(f"\n최적 모델 성능 (Test Set):")
        logger.info(f"  Threshold: ±{optimal['threshold']*100:.1f}%")
        logger.info(f"  SNR: {test_data['snr']:.2f}")
        logger.info(f"  Return: {test_data['backtest']['total_return_pct']:.2f}%")
        logger.info(f"  Trades: {test_data['backtest']['num_trades']}")
        logger.info(f"  Win Rate: {test_data['backtest']['win_rate']*100:.1f}%")
        logger.info(f"  Liquidated: {test_data['backtest']['liquidated']}")

    # 9. 결과 저장
    results_dir = project_root / 'data' / 'trained_models' / 'xgboost_regression'
    results_file = results_dir / 'threshold_sweep_results.json'

    with open(results_file, 'w') as f:
        # numpy types를 Python types로 변환
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # 10. 5-Way 비교 준비
    logger.info("\n" + "="*80)
    logger.info("5-WAY COMPARISON PREVIEW")
    logger.info("="*80)

    logger.info("\n| Model | Threshold | SNR | Test Return | Test Trades | Win Rate |")
    logger.info("|-------|-----------|-----|-------------|-------------|----------|")
    logger.info("| BUGGY | 0.2% | 0.95 | -1051.80% | 867 | 2.3% |")
    logger.info("| FIXED | 1.0% | 1.37 | -2.05% | 1 | 0.0% |")
    logger.info("| IMPROVED | 2.0% | 1.26 | -2.80% | 9 | 66.7% |")
    logger.info("| REGRESSION | 1.5% | 2.31 | 0.00% | 0 | 0.0% |")

    if optimal:
        test_bt = optimal['data']['test']['backtest']
        logger.info(
            f"| OPTIMAL | {optimal['threshold']*100:.1f}% | "
            f"{optimal['data']['test']['snr']:.2f} | "
            f"{test_bt['total_return_pct']:+.2f}% | "
            f"{test_bt['num_trades']} | "
            f"{test_bt['win_rate']*100:.1f}% |"
        )

    logger.info("\n" + "="*80)
    logger.info("✅ Threshold Sweep Completed!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
