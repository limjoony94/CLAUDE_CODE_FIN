"""Threshold Sweep for Sequential Features Model

비판적 질문:
- Sequential Features 모델의 R² = -0.41 (예측 부정확)
- 하지만 threshold 조정으로 수익성 개선 가능한가?

실험:
1. ±0.3%, ±0.5%, ±0.8%, ±1.0%, ±1.5% threshold 테스트
2. 각 threshold에서 거래 빈도, 수익률, 수수료 계산
3. Buy & Hold 대비 성과 비교

예상 결과:
- 낮은 threshold → 거래 빈도 증가 → 수수료 증가
- R² < 0 → 예측 부정확 → 손실 가능성
- 하지만 실험적 검증 필요
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


def simple_backtest(
    df: pd.DataFrame,
    predictions: np.ndarray,
    long_threshold: float,
    short_threshold: float,
    initial_balance: float = 10000.0,
    position_size: float = 0.03,
    leverage: int = 3,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0001
) -> dict:
    """간단한 백테스팅"""

    # 신호 생성
    signals = np.zeros(len(predictions))
    signals[predictions > long_threshold] = 1  # LONG
    signals[predictions < short_threshold] = -1  # SHORT

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        signal = signals[i]
        price = df.iloc[i]['close']

        # 포지션 변경
        if signal == 1 and position == 0:  # LONG 진입
            position = position_size
            entry_price = price * (1 + slippage)
            balance -= position * entry_price * transaction_fee
            trades.append({'type': 'LONG', 'entry': entry_price, 'index': i})

        elif signal == -1 and position == 0:  # SHORT 진입
            position = -position_size
            entry_price = price * (1 - slippage)
            balance -= abs(position) * entry_price * transaction_fee
            trades.append({'type': 'SHORT', 'entry': entry_price, 'index': i})

        elif signal == 0 and position != 0:  # 청산
            exit_price = price * (1 - slippage * np.sign(position))

            # PnL 계산
            if position > 0:  # LONG 청산
                pnl = position * (exit_price - entry_price) * leverage
            else:  # SHORT 청산
                pnl = abs(position) * (entry_price - exit_price) * leverage

            balance += pnl
            balance -= abs(position) * exit_price * transaction_fee

            trades[-1]['exit'] = exit_price
            trades[-1]['pnl'] = pnl
            trades[-1]['exit_index'] = i

            position = 0.0
            entry_price = 0.0

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

    # 통계
    total_return = (balance - initial_balance) / initial_balance * 100
    completed_trades = [t for t in trades if 'pnl' in t]

    if completed_trades:
        pnls = [t['pnl'] for t in completed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    return {
        'final_balance': balance,
        'total_return_pct': total_return,
        'num_trades': len(completed_trades),
        'num_signals': int(np.sum(np.abs(signals))),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'trades': completed_trades
    }


def main():
    logger.info("="*80)
    logger.info("Threshold Sweep Experiment - Sequential Features Model")
    logger.info("비판적 검증: R² < 0 모델로 수익 가능한가?")
    logger.info("="*80)

    # 1. 데이터 로드 및 처리 (이전과 동일)
    logger.info("\n1. Loading and processing data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)

    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)
    df_sequential = indicators.calculate_sequential_features(df_processed)

    # 타겟 생성
    lookahead = 48
    df_sequential['target'] = df_sequential['close'].pct_change(lookahead).shift(-lookahead)

    # Feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_sequential.columns if col not in exclude_cols]

    df_sequential = df_sequential.dropna()

    # 2. Train/Test Split
    logger.info("\n2. Splitting data...")
    n = len(df_sequential)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df_sequential.iloc[:train_end].copy()
    val_df = df_sequential.iloc[train_end:val_end].copy()
    test_df = df_sequential.iloc[val_end:].copy()

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. 모델 훈련 (빠른 재현)
    logger.info("\n3. Training XGBoost model...")

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
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
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    # 예측
    test_preds = model.predict(dtest)
    test_r2 = r2_score(y_test, test_preds)

    logger.info(f"Model R²: {test_r2:.4f}")
    logger.info(f"Prediction Std: {test_preds.std()*100:.4f}%")
    logger.info(f"Prediction Range: {test_preds.min()*100:.3f}% to {test_preds.max()*100:.3f}%")

    # 4. Threshold Sweep
    logger.info("\n4. Threshold Sweep Experiment...")

    thresholds = [0.003, 0.005, 0.008, 0.010, 0.015]  # 0.3%, 0.5%, 0.8%, 1.0%, 1.5%

    results = []

    for thresh in thresholds:
        logger.info(f"\n--- Testing threshold: ±{thresh*100:.1f}% ---")

        result = simple_backtest(
            test_df,
            test_preds,
            long_threshold=thresh,
            short_threshold=-thresh,
            initial_balance=10000.0,
            position_size=0.03,
            leverage=3,
            transaction_fee=0.0004,
            slippage=0.0001
        )

        result['threshold'] = thresh
        results.append(result)

        logger.info(f"Signals: {result['num_signals']}")
        logger.info(f"Trades: {result['num_trades']}")
        logger.info(f"Return: {result['total_return_pct']:+.2f}%")
        if result['num_trades'] > 0:
            logger.info(f"Win Rate: {result['win_rate']*100:.1f}%")
            logger.info(f"Profit Factor: {result['profit_factor']:.2f}")

    # 5. 결과 비교
    logger.info("\n" + "="*80)
    logger.info("THRESHOLD SWEEP RESULTS")
    logger.info("="*80)

    # Buy & Hold
    first_price = test_df.iloc[0]['close']
    last_price = test_df.iloc[-1]['close']
    bh_return = (last_price - first_price) / first_price * 100

    logger.info(f"\nBuy & Hold: {bh_return:+.2f}%")
    logger.info(f"Model R²: {test_r2:.4f} (예측 정확도)")

    logger.info("\n| Threshold | Signals | Trades | Return | Win Rate | Profit Factor |")
    logger.info("|-----------|---------|--------|--------|----------|---------------|")

    for r in results:
        logger.info(
            f"| ±{r['threshold']*100:4.1f}% | {r['num_signals']:7d} | {r['num_trades']:6d} | "
            f"{r['total_return_pct']:+6.2f}% | {r['win_rate']*100:8.1f}% | {r['profit_factor']:13.2f} |"
        )

    # 6. 비판적 분석
    logger.info("\n" + "="*80)
    logger.info("CRITICAL ANALYSIS")
    logger.info("="*80)

    best_result = max(results, key=lambda x: x['total_return_pct'])

    logger.info(f"\n최고 성과 Threshold: ±{best_result['threshold']*100:.1f}%")
    logger.info(f"  Return: {best_result['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {best_result['num_trades']}")
    logger.info(f"  Win Rate: {best_result['win_rate']*100:.1f}%")

    logger.info(f"\nBuy & Hold: {bh_return:+.2f}%")
    logger.info(f"Difference: {best_result['total_return_pct'] - bh_return:+.2f}%")

    if best_result['total_return_pct'] > bh_return:
        logger.success(f"\n✅ ML이 Buy & Hold를 초과했습니다!")
        logger.info(f"   Threshold: ±{best_result['threshold']*100:.1f}%가 최적")
    else:
        logger.warning(f"\n⚠️ Buy & Hold가 여전히 우수합니다.")
        logger.info(f"   Gap: {bh_return - best_result['total_return_pct']:.2f}%")

    # 7. 거래 빈도 vs 수익률 분석
    logger.info("\n" + "="*80)
    logger.info("Trade Frequency vs Profitability")
    logger.info("="*80)

    logger.info("\n관찰:")
    for r in results:
        if r['num_trades'] > 0:
            avg_return_per_trade = r['total_return_pct'] / r['num_trades']
            logger.info(
                f"  Threshold ±{r['threshold']*100:.1f}%: "
                f"{r['num_trades']} trades, "
                f"{avg_return_per_trade:+.3f}% per trade"
            )

    # 8. 최종 결론
    logger.info("\n" + "="*80)
    logger.info("FINAL CONCLUSION")
    logger.info("="*80)

    logger.info("\n비판적 질문 답변:")

    logger.info("\nQ1: Threshold를 낮추면 거래가 발생하는가?")
    if any(r['num_trades'] > 0 for r in results):
        logger.success("A1: Yes, threshold 조정으로 거래 발생 확인")
        for r in results:
            if r['num_trades'] > 0:
                logger.info(f"    ±{r['threshold']*100:.1f}%: {r['num_trades']} trades")
    else:
        logger.error("A1: No, 모든 threshold에서 거래 없음")

    logger.info("\nQ2: R² < 0 모델로 수익을 낼 수 있는가?")
    if best_result['total_return_pct'] > 0:
        logger.warning(f"A2: 수익은 {best_result['total_return_pct']:+.2f}% 발생")
        logger.warning(f"    하지만 Buy & Hold ({bh_return:+.2f}%) 대비?")
        if best_result['total_return_pct'] > bh_return:
            logger.success("    → ML이 더 우수!")
        else:
            logger.warning(f"    → Buy & Hold가 {bh_return - best_result['total_return_pct']:.2f}% 더 우수")
    else:
        logger.error(f"A2: No, 최고 성과도 {best_result['total_return_pct']:+.2f}% (손실)")

    logger.info("\nQ3: 거래 빈도 증가가 수익성을 높이는가?")
    returns_by_trades = [(r['num_trades'], r['total_return_pct']) for r in results]
    returns_by_trades.sort(key=lambda x: x[0])

    if len(returns_by_trades) > 1:
        correlation = "증가" if returns_by_trades[-1][1] > returns_by_trades[0][1] else "감소"
        logger.info(f"A3: 거래 빈도 증가 시 수익률 {correlation}")
        logger.info(f"    최소 거래: {returns_by_trades[0][0]} trades → {returns_by_trades[0][1]:+.2f}%")
        logger.info(f"    최대 거래: {returns_by_trades[-1][0]} trades → {returns_by_trades[-1][1]:+.2f}%")

        if correlation == "감소":
            logger.warning("    ⚠️ 더 많은 거래 = 더 많은 수수료 손실!")

    logger.info("\n최종 권장사항:")
    if best_result['total_return_pct'] > bh_return:
        logger.success(f"✅ Threshold ±{best_result['threshold']*100:.1f}% 사용 권장")
        logger.success(f"   예상 수익: {best_result['total_return_pct']:+.2f}% (Buy & Hold 초과)")
    else:
        logger.warning("⚠️ Buy & Hold 유지 권장")
        logger.warning(f"   이유: ML 최고 성과 ({best_result['total_return_pct']:+.2f}%) < Buy & Hold ({bh_return:+.2f}%)")
        logger.warning(f"   Gap: {bh_return - best_result['total_return_pct']:.2f}%")

    logger.info("\n" + "="*80)
    logger.info("✅ Threshold Sweep Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
