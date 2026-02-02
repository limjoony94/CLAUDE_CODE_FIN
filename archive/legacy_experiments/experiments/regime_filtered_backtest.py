"""Regime-Filtered Backtest: Critical Validation

비판적 질문:
- Regime filter가 정말로 작동하는가?
- 4개 기간의 패턴이 우연인가, 실제 패턴인가?
- Threshold 0.08%가 overfitting인가?

검증 방법:
1. 전체 데이터에 regime filter 적용
2. Filtered vs Unfiltered 성과 비교
3. Threshold sensitivity analysis
4. 거래 빈도 vs 성과 분석

목표:
- Regime filter의 실제 효과 정량화
- False discovery 가능성 배제
- 실전 적용 가능성 판단
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb
from sklearn.metrics import r2_score

from src.indicators.technical_indicators import TechnicalIndicators


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """변동성 계산 (rolling std of returns)"""
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    return volatility


def detect_regime(volatility: float, threshold: float = 0.0008) -> str:
    """Regime detection based on volatility

    Args:
        volatility: Current volatility (as decimal, e.g., 0.001 = 0.1%)
        threshold: Threshold for high/low volatility

    Returns:
        'high_vol' or 'low_vol'
    """
    if volatility > threshold:
        return 'high_vol'
    else:
        return 'low_vol'


def backtest_with_regime_filter(
    df: pd.DataFrame,
    predictions: np.ndarray,
    volatility: pd.Series,
    entry_threshold: float = 0.003,
    vol_threshold: float = 0.0008,
    stop_loss_pct: float = 0.010,
    take_profit_pct: float = 0.030,
    initial_balance: float = 10000.0,
    position_size: float = 0.03,
    leverage: int = 3,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0001,
    use_filter: bool = True
) -> dict:
    """Regime-filtered 백테스팅

    Args:
        use_filter: True = high-vol에서만 거래, False = 항상 거래
    """

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []
    regime_log = []

    for i in range(len(df)):
        if i < 20:  # 최소 window size
            continue

        current_price = df.iloc[i]['close']
        signal = predictions[i]
        current_vol = volatility.iloc[i]
        regime = detect_regime(current_vol, vol_threshold)

        regime_log.append(regime)

        # 포지션 청산 체크 (항상 수행)
        if position != 0:
            price_change = (current_price - entry_price) / entry_price

            # LONG 포지션
            if position > 0:
                if price_change <= -stop_loss_pct:  # Stop-Loss
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance += pnl
                    balance -= position * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'SL'
                    trades[-1]['exit_regime'] = regime

                    position = 0.0
                    entry_price = 0.0

                elif price_change >= take_profit_pct:  # Take-Profit
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance += pnl
                    balance -= position * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'TP'
                    trades[-1]['exit_regime'] = regime

                    position = 0.0
                    entry_price = 0.0

            # SHORT 포지션
            elif position < 0:
                if price_change >= stop_loss_pct:  # Stop-Loss
                    exit_price = current_price * (1 + slippage)
                    pnl = abs(position) * (entry_price - exit_price) * leverage
                    balance += pnl
                    balance -= abs(position) * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'SL'
                    trades[-1]['exit_regime'] = regime

                    position = 0.0
                    entry_price = 0.0

                elif price_change <= -take_profit_pct:  # Take-Profit
                    exit_price = current_price * (1 + slippage)
                    pnl = abs(position) * (entry_price - exit_price) * leverage
                    balance += pnl
                    balance -= abs(position) * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'TP'
                    trades[-1]['exit_regime'] = regime

                    position = 0.0
                    entry_price = 0.0

        # 새 포지션 진입 (regime 체크)
        if position == 0:
            # Regime filter: high_vol에서만 거래
            if use_filter and regime == 'low_vol':
                continue  # Skip trade in low volatility

            if signal > entry_threshold:  # LONG
                position = position_size
                entry_price = current_price * (1 + slippage)
                balance -= position * entry_price * transaction_fee

                trades.append({
                    'type': 'LONG',
                    'entry': entry_price,
                    'index': i,
                    'entry_regime': regime,
                    'volatility': current_vol
                })

            elif signal < -entry_threshold:  # SHORT
                position = -position_size
                entry_price = current_price * (1 - slippage)
                balance -= abs(position) * entry_price * transaction_fee

                trades.append({
                    'type': 'SHORT',
                    'entry': entry_price,
                    'index': i,
                    'entry_regime': regime,
                    'volatility': current_vol
                })

    # 최종 청산
    if position != 0:
        final_price = df.iloc[-1]['close']
        if position > 0:
            pnl = position * (final_price - entry_price) * leverage
        else:
            pnl = abs(position) * (entry_price - final_price) * leverage

        balance += pnl
        balance -= abs(position) * final_price * transaction_fee

        if trades:
            trades[-1]['exit'] = final_price
            trades[-1]['pnl'] = pnl
            trades[-1]['exit_type'] = 'FINAL'

    # 통계
    total_return = (balance - initial_balance) / initial_balance * 100
    completed_trades = [t for t in trades if 'pnl' in t]

    if completed_trades:
        pnls = [t['pnl'] for t in completed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        # Regime별 거래
        high_vol_trades = [t for t in completed_trades if t.get('entry_regime') == 'high_vol']
        low_vol_trades = [t for t in completed_trades if t.get('entry_regime') == 'low_vol']
    else:
        win_rate = 0
        profit_factor = 0
        high_vol_trades = []
        low_vol_trades = []

    # Regime 분포
    high_vol_periods = regime_log.count('high_vol')
    low_vol_periods = regime_log.count('low_vol')

    return {
        'final_balance': balance,
        'total_return_pct': total_return,
        'num_trades': len(completed_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': completed_trades,
        'high_vol_trades': len(high_vol_trades),
        'low_vol_trades': len(low_vol_trades),
        'high_vol_periods': high_vol_periods,
        'low_vol_periods': low_vol_periods,
        'high_vol_pct': high_vol_periods / len(regime_log) if regime_log else 0
    }


def main():
    logger.info("="*80)
    logger.info("Regime-Filtered Backtest: Critical Validation")
    logger.info("비판적 검증: Regime filter가 정말 작동하는가?")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)

    # 2. Sequential Features 계산
    logger.info("\n2. Calculating Sequential Features...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)
    df_sequential = indicators.calculate_sequential_features(df_processed)

    # 타겟
    lookahead = 48
    df_sequential['target'] = df_sequential['close'].pct_change(lookahead).shift(-lookahead)

    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_sequential.columns if col not in exclude_cols]
    df_sequential = df_sequential.dropna()

    # Volatility 계산
    volatility = calculate_volatility(df_sequential, window=20)
    df_sequential['volatility'] = volatility

    logger.info(f"Processed data: {len(df_sequential)} candles")

    # 3. Train/Test Split (50/20/30)
    logger.info("\n3. Data Split...")
    n = len(df_sequential)
    train_end = int(n * 0.5)
    val_end = int(n * 0.7)

    train_df = df_sequential.iloc[:train_end].copy()
    val_df = df_sequential.iloc[train_end:val_end].copy()
    test_df = df_sequential.iloc[val_end:].copy()

    logger.info(f"Test Period: {test_df.iloc[0]['timestamp']} to {test_df.iloc[-1]['timestamp']}")
    logger.info(f"Test Size: {len(test_df)} candles ({len(test_df)/288:.1f} days)")

    # 4. 모델 훈련
    logger.info("\n4. Training XGBoost model...")

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

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

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )

    # 예측
    test_preds = model.predict(dtest)
    test_volatility = test_df['volatility']

    # 5. Baseline: Unfiltered (기존 전략)
    logger.info("\n" + "="*80)
    logger.info("5. BASELINE: Unfiltered Strategy (Trade Always)")
    logger.info("="*80)

    result_unfiltered = backtest_with_regime_filter(
        test_df,
        test_preds,
        test_volatility,
        entry_threshold=0.003,
        vol_threshold=0.0008,
        stop_loss_pct=0.010,
        take_profit_pct=0.030,
        use_filter=False  # No filter
    )

    logger.info(f"Return: {result_unfiltered['total_return_pct']:+.2f}%")
    logger.info(f"Trades: {result_unfiltered['num_trades']}")
    logger.info(f"Win Rate: {result_unfiltered['win_rate']*100:.1f}%")
    logger.info(f"Profit Factor: {result_unfiltered['profit_factor']:.2f}")
    logger.info(f"  High-Vol Trades: {result_unfiltered['high_vol_trades']}")
    logger.info(f"  Low-Vol Trades: {result_unfiltered['low_vol_trades']}")

    # 6. Regime-Filtered Strategy
    logger.info("\n" + "="*80)
    logger.info("6. REGIME-FILTERED: Trade Only in High Volatility")
    logger.info("="*80)

    result_filtered = backtest_with_regime_filter(
        test_df,
        test_preds,
        test_volatility,
        entry_threshold=0.003,
        vol_threshold=0.0008,  # 0.08% threshold
        stop_loss_pct=0.010,
        take_profit_pct=0.030,
        use_filter=True  # With filter
    )

    logger.info(f"Return: {result_filtered['total_return_pct']:+.2f}%")
    logger.info(f"Trades: {result_filtered['num_trades']}")
    logger.info(f"Win Rate: {result_filtered['win_rate']*100:.1f}%")
    logger.info(f"Profit Factor: {result_filtered['profit_factor']:.2f}")
    logger.info(f"  High-Vol Trades: {result_filtered['high_vol_trades']} (kept)")
    logger.info(f"  Low-Vol Trades: {result_filtered['low_vol_trades']} (should be 0)")

    logger.info(f"\nRegime Distribution:")
    logger.info(f"  High-Vol Periods: {result_filtered['high_vol_periods']} ({result_filtered['high_vol_pct']*100:.1f}%)")
    logger.info(f"  Low-Vol Periods: {result_filtered['low_vol_periods']} ({(1-result_filtered['high_vol_pct'])*100:.1f}%)")

    # 7. Threshold Sensitivity Analysis
    logger.info("\n" + "="*80)
    logger.info("7. THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("="*80)

    thresholds = [0.0006, 0.0007, 0.0008, 0.0009, 0.0010]  # 0.06% to 0.10%
    sensitivity_results = []

    for thresh in thresholds:
        result = backtest_with_regime_filter(
            test_df,
            test_preds,
            test_volatility,
            entry_threshold=0.003,
            vol_threshold=thresh,
            stop_loss_pct=0.010,
            take_profit_pct=0.030,
            use_filter=True
        )

        sensitivity_results.append({
            'threshold': thresh,
            'return': result['total_return_pct'],
            'trades': result['num_trades'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'high_vol_pct': result['high_vol_pct']
        })

    logger.info("\n| Threshold | Return | Trades | Win Rate | PF | High-Vol % |")
    logger.info("|-----------|--------|--------|----------|-----|------------|")
    for r in sensitivity_results:
        logger.info(
            f"| {r['threshold']*100:9.2f}% | {r['return']:+6.2f}% | {r['trades']:6d} | "
            f"{r['win_rate']*100:8.1f}% | {r['profit_factor']:3.2f} | {r['high_vol_pct']*100:10.1f}% |"
        )

    # 8. 비교 분석
    logger.info("\n" + "="*80)
    logger.info("8. COMPARISON: Filtered vs Unfiltered")
    logger.info("="*80)

    # Buy & Hold
    first_price = test_df.iloc[0]['close']
    last_price = test_df.iloc[-1]['close']
    bh_return = (last_price - first_price) / first_price * 100

    improvement = result_filtered['total_return_pct'] - result_unfiltered['total_return_pct']

    logger.info(f"\nBuy & Hold: {bh_return:+.2f}%")
    logger.info("")
    logger.info("| Strategy | Return | Trades | Win Rate | Profit Factor | vs B&H |")
    logger.info("|----------|--------|--------|----------|---------------|--------|")
    logger.info(
        f"| Unfiltered | {result_unfiltered['total_return_pct']:+6.2f}% | {result_unfiltered['num_trades']:6d} | "
        f"{result_unfiltered['win_rate']*100:8.1f}% | {result_unfiltered['profit_factor']:13.2f} | "
        f"{result_unfiltered['total_return_pct'] - bh_return:+6.2f}% |"
    )
    logger.info(
        f"| Filtered   | {result_filtered['total_return_pct']:+6.2f}% | {result_filtered['num_trades']:6d} | "
        f"{result_filtered['win_rate']*100:8.1f}% | {result_filtered['profit_factor']:13.2f} | "
        f"{result_filtered['total_return_pct'] - bh_return:+6.2f}% |"
    )

    logger.info(f"\nImprovement (Filtered - Unfiltered): {improvement:+.2f}%")

    # 9. 최종 판정
    logger.info("\n" + "="*80)
    logger.info("9. CRITICAL VERDICT")
    logger.info("="*80)

    logger.info(f"\n검증 결과:")
    logger.info(f"  Unfiltered: {result_unfiltered['total_return_pct']:+.2f}%")
    logger.info(f"  Filtered: {result_filtered['total_return_pct']:+.2f}%")
    logger.info(f"  Improvement: {improvement:+.2f}%")

    # Threshold stability
    returns = [r['return'] for r in sensitivity_results]
    best_return = max(returns)
    worst_return = min(returns)
    avg_return = np.mean(returns)

    logger.info(f"\nThreshold Sensitivity:")
    logger.info(f"  Best: {best_return:+.2f}%")
    logger.info(f"  Worst: {worst_return:+.2f}%")
    logger.info(f"  Average: {avg_return:+.2f}%")
    logger.info(f"  Range: {best_return - worst_return:.2f}%")

    if improvement > 2.0 and avg_return > result_unfiltered['total_return_pct']:
        logger.success("\n✅ REGIME FILTER VALIDATED!")
        logger.info("  - Filtered significantly outperforms unfiltered")
        logger.info("  - Improvement consistent across thresholds")
        logger.info("  권장: Regime-filtered strategy 사용")

    elif improvement > 0 and avg_return > 0:
        logger.warning("\n⚠️ REGIME FILTER SHOWS PROMISE")
        logger.info("  - Filtered modestly outperforms unfiltered")
        logger.info("  - Further optimization may help")
        logger.info("  권장: Paper trading으로 추가 검증")

    elif improvement <= 0:
        logger.error("\n❌ REGIME FILTER DOES NOT WORK")
        logger.info("  - Filtered performs worse than unfiltered")
        logger.info("  - Volatility threshold may be wrong")
        logger.info("  - 또는 regime dependency가 false discovery")
        logger.info("  권장: 다른 regime indicator 탐색 또는 전략 재고")

    else:
        logger.warning("\n⚠️ RESULTS INCONCLUSIVE")
        logger.info("  - Mixed signals from different tests")
        logger.info("  권장: 더 긴 데이터 수집 또는 다른 접근법")

    # 10. 최종 권장사항
    logger.info("\n" + "="*80)
    logger.info("10. FINAL RECOMMENDATION")
    logger.info("="*80)

    best_strategy = "Filtered" if result_filtered['total_return_pct'] > result_unfiltered['total_return_pct'] else "Unfiltered"
    best_return = max(result_filtered['total_return_pct'], result_unfiltered['total_return_pct'])

    logger.info(f"\n최적 전략: {best_strategy}")
    logger.info(f"  Return: {best_return:+.2f}%")
    logger.info(f"  vs Buy & Hold: {best_return - bh_return:+.2f}%")

    if best_return > bh_return and improvement > 1.0:
        logger.success("\n🎯 SUCCESS: Regime filter improves performance!")
        logger.info("다음 단계:")
        logger.info("  1. Paper trading with regime filter")
        logger.info("  2. Live monitoring of regime detection")
        logger.info("  3. Gradual scaling if successful")

    elif best_return > 0:
        logger.warning("\n⚠️ MARGINAL: Strategy shows potential but not dominant")
        logger.info("다음 단계:")
        logger.info("  1. Consider alternative regime indicators")
        logger.info("  2. Test on longer data period")
        logger.info("  3. Explore other timeframes (15m, 1h)")

    else:
        logger.error("\n❌ FAILURE: Strategy does not beat baseline")
        logger.info("권장사항:")
        logger.info("  1. Accept Buy & Hold as optimal")
        logger.info("  2. OR pivot to completely different approach")
        logger.info("  3. OR test different asset/market")

    logger.info("\n" + "="*80)
    logger.info("✅ Regime Filter Validation Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
