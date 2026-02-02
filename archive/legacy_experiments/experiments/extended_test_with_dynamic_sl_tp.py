"""Extended Test Set (30%) + Dynamic ATR-based SL/TP

비판적 접근:
1. 데이터 제약 (2개월) 인정
2. Test set 확대 (15% → 30% = 18일)
3. 동적 SL/TP로 적응성 향상
4. Fixed vs Dynamic 비교

목표:
- 더 많은 거래로 통계적 신뢰도 향상
- ATR 기반 적응형 리스크 관리
- 변동성에 따른 최적 SL/TP
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range) 계산

    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = 14-period moving average of True Range
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def backtest_with_dynamic_sl_tp(
    df: pd.DataFrame,
    predictions: np.ndarray,
    entry_threshold: float = 0.003,
    atr_multiplier: float = 1.5,
    risk_reward_ratio: float = 3.0,
    initial_balance: float = 10000.0,
    position_size: float = 0.03,
    leverage: int = 3,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0001,
    mode: str = 'dynamic'  # 'dynamic' or 'fixed'
) -> dict:
    """동적 또는 고정 SL/TP 백테스팅

    Dynamic Mode:
      - ATR 기반 SL/TP 계산
      - 변동성 높을 때: 넓은 SL/TP
      - 변동성 낮을 때: 좁은 SL/TP

    Fixed Mode:
      - 고정 SL/TP (-1.0%, +3.0%)
      - 기존 방식과 동일
    """

    # ATR 계산
    atr_values = calculate_atr(df)

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    current_sl = 0.0
    current_tp = 0.0
    trades = []

    for i in range(len(df)):
        if i < 14:  # ATR 계산을 위한 최소 기간
            continue

        current_price = df.iloc[i]['close']
        signal = predictions[i]
        atr_pct = atr_values.iloc[i] / current_price  # ATR을 % 변환

        # 포지션 청산 체크 (먼저 수행)
        if position != 0:
            price_change = (current_price - entry_price) / entry_price

            # LONG 포지션
            if position > 0:
                if price_change <= -current_sl:  # Stop-Loss
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance += pnl
                    balance -= position * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['exit_type'] = 'SL'
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_index'] = i

                    position = 0.0
                    entry_price = 0.0

                elif price_change >= current_tp:  # Take-Profit
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance += pnl
                    balance -= position * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['exit_type'] = 'TP'
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_index'] = i

                    position = 0.0
                    entry_price = 0.0

            # SHORT 포지션
            elif position < 0:
                if price_change >= current_sl:  # Stop-Loss (SHORT는 가격 상승이 손실)
                    exit_price = current_price * (1 + slippage)
                    pnl = abs(position) * (entry_price - exit_price) * leverage
                    balance += pnl
                    balance -= abs(position) * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['exit_type'] = 'SL'
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_index'] = i

                    position = 0.0
                    entry_price = 0.0

                elif price_change <= -current_tp:  # Take-Profit (SHORT는 가격 하락이 수익)
                    exit_price = current_price * (1 + slippage)
                    pnl = abs(position) * (entry_price - exit_price) * leverage
                    balance += pnl
                    balance -= abs(position) * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['exit_type'] = 'TP'
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_index'] = i

                    position = 0.0
                    entry_price = 0.0

        # 새 포지션 진입
        if position == 0:
            # 동적 SL/TP 계산
            if mode == 'dynamic':
                stop_loss_pct = atr_pct * atr_multiplier
                take_profit_pct = stop_loss_pct * risk_reward_ratio
            else:  # fixed
                stop_loss_pct = 0.010  # -1.0%
                take_profit_pct = 0.030  # +3.0%

            # LONG 진입
            if signal > entry_threshold:
                position = position_size
                entry_price = current_price * (1 + slippage)
                current_sl = stop_loss_pct
                current_tp = take_profit_pct
                balance -= position * entry_price * transaction_fee

                trades.append({
                    'type': 'LONG',
                    'entry': entry_price,
                    'index': i,
                    'atr_pct': atr_pct,
                    'sl_pct': stop_loss_pct,
                    'tp_pct': take_profit_pct,
                    'mode': mode
                })

            # SHORT 진입
            elif signal < -entry_threshold:
                position = -position_size
                entry_price = current_price * (1 - slippage)
                current_sl = stop_loss_pct
                current_tp = take_profit_pct
                balance -= abs(position) * entry_price * transaction_fee

                trades.append({
                    'type': 'SHORT',
                    'entry': entry_price,
                    'index': i,
                    'atr_pct': atr_pct,
                    'sl_pct': stop_loss_pct,
                    'tp_pct': take_profit_pct,
                    'mode': mode
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
            trades[-1]['exit_type'] = 'FINAL'
            trades[-1]['pnl'] = pnl

    # 통계 계산
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

        # SL vs TP 통계
        sl_exits = [t for t in completed_trades if t.get('exit_type') == 'SL']
        tp_exits = [t for t in completed_trades if t.get('exit_type') == 'TP']
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        sl_exits = []
        tp_exits = []

    return {
        'final_balance': balance,
        'total_return_pct': total_return,
        'num_trades': len(completed_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'trades': completed_trades,
        'sl_exits': len(sl_exits),
        'tp_exits': len(tp_exits)
    }


def main():
    logger.info("="*80)
    logger.info("Extended Test Set (30%) + Dynamic ATR-based SL/TP")
    logger.info("비판적 접근: 데이터 제약 인정, 적응형 리스크 관리")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)

    logger.info(f"Total data: {len(df)} candles")
    logger.info(f"Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

    # 2. Sequential Features 계산
    logger.info("\n2. Calculating Sequential Features...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)
    df_sequential = indicators.calculate_sequential_features(df_processed)

    # 타겟 생성
    lookahead = 48
    df_sequential['target'] = df_sequential['close'].pct_change(lookahead).shift(-lookahead)

    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_sequential.columns if col not in exclude_cols]
    df_sequential = df_sequential.dropna()

    # 3. Train/Val/Test Split (50/20/30)
    logger.info("\n3. Data Split (50% Train / 20% Val / 30% Test)...")
    n = len(df_sequential)

    train_end = int(n * 0.5)
    val_end = int(n * 0.7)

    train_df = df_sequential.iloc[:train_end].copy()
    val_df = df_sequential.iloc[train_end:val_end].copy()
    test_df = df_sequential.iloc[val_end:].copy()

    logger.info(f"Train: {len(train_df)} candles ({len(train_df)/288:.1f} days)")
    logger.info(f"Val: {len(val_df)} candles ({len(val_df)/288:.1f} days)")
    logger.info(f"Test: {len(test_df)} candles ({len(test_df)/288:.1f} days)")
    logger.info(f"Test Period: {test_df.iloc[0]['timestamp']} to {test_df.iloc[-1]['timestamp']}")

    # 4. 모델 훈련
    logger.info("\n4. Training XGBoost model...")

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

    # 5. 백테스트: Fixed SL/TP (Baseline)
    logger.info("\n" + "="*80)
    logger.info("5. BASELINE: Fixed SL/TP (-1.0%, +3.0%)")
    logger.info("="*80)

    result_fixed = backtest_with_dynamic_sl_tp(
        test_df,
        test_preds,
        entry_threshold=0.003,
        mode='fixed',
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    logger.info(f"Trades: {result_fixed['num_trades']}")
    logger.info(f"Return: {result_fixed['total_return_pct']:+.2f}%")
    logger.info(f"Win Rate: {result_fixed['win_rate']*100:.1f}%")
    logger.info(f"Profit Factor: {result_fixed['profit_factor']:.2f}")
    logger.info(f"SL Exits: {result_fixed['sl_exits']}, TP Exits: {result_fixed['tp_exits']}")

    # 6. 백테스트: Dynamic ATR-based SL/TP
    logger.info("\n" + "="*80)
    logger.info("6. ENHANCED: Dynamic ATR-based SL/TP (1.5x ATR, 1:3 ratio)")
    logger.info("="*80)

    result_dynamic = backtest_with_dynamic_sl_tp(
        test_df,
        test_preds,
        entry_threshold=0.003,
        atr_multiplier=1.5,
        risk_reward_ratio=3.0,
        mode='dynamic',
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001
    )

    logger.info(f"Trades: {result_dynamic['num_trades']}")
    logger.info(f"Return: {result_dynamic['total_return_pct']:+.2f}%")
    logger.info(f"Win Rate: {result_dynamic['win_rate']*100:.1f}%")
    logger.info(f"Profit Factor: {result_dynamic['profit_factor']:.2f}")
    logger.info(f"SL Exits: {result_dynamic['sl_exits']}, TP Exits: {result_dynamic['tp_exits']}")

    # 7. ATR 분석
    if result_dynamic['num_trades'] > 0:
        logger.info("\n7. ATR Analysis (Dynamic Mode):")
        atr_pcts = [t['atr_pct'] for t in result_dynamic['trades'] if 'atr_pct' in t]
        sl_pcts = [t['sl_pct'] for t in result_dynamic['trades'] if 'sl_pct' in t]
        tp_pcts = [t['tp_pct'] for t in result_dynamic['trades'] if 'tp_pct' in t]

        logger.info(f"  ATR Range: {min(atr_pcts)*100:.3f}% to {max(atr_pcts)*100:.3f}%")
        logger.info(f"  ATR Mean: {np.mean(atr_pcts)*100:.3f}%")
        logger.info(f"  Dynamic SL Range: {min(sl_pcts)*100:.3f}% to {max(sl_pcts)*100:.3f}%")
        logger.info(f"  Dynamic TP Range: {min(tp_pcts)*100:.3f}% to {max(tp_pcts)*100:.3f}%")

    # 8. 비교 분석
    logger.info("\n" + "="*80)
    logger.info("8. COMPARISON: Fixed vs Dynamic")
    logger.info("="*80)

    # Buy & Hold
    first_price = test_df.iloc[0]['close']
    last_price = test_df.iloc[-1]['close']
    bh_return = (last_price - first_price) / first_price * 100

    logger.info(f"\nBuy & Hold: {bh_return:+.2f}%")
    logger.info("")
    logger.info("| Strategy | Return | Trades | Win Rate | Profit Factor | vs B&H |")
    logger.info("|----------|--------|--------|----------|---------------|--------|")
    logger.info(
        f"| Fixed SL/TP | {result_fixed['total_return_pct']:+6.2f}% | {result_fixed['num_trades']:6d} | "
        f"{result_fixed['win_rate']*100:8.1f}% | {result_fixed['profit_factor']:13.2f} | "
        f"{result_fixed['total_return_pct'] - bh_return:+6.2f}% |"
    )
    logger.info(
        f"| Dynamic SL/TP | {result_dynamic['total_return_pct']:+6.2f}% | {result_dynamic['num_trades']:6d} | "
        f"{result_dynamic['win_rate']*100:8.1f}% | {result_dynamic['profit_factor']:13.2f} | "
        f"{result_dynamic['total_return_pct'] - bh_return:+6.2f}% |"
    )

    # 개선도 계산
    improvement = result_dynamic['total_return_pct'] - result_fixed['total_return_pct']

    # 9. 통계적 유의성 평가
    logger.info("\n" + "="*80)
    logger.info("9. STATISTICAL SIGNIFICANCE")
    logger.info("="*80)

    max_trades = max(result_fixed['num_trades'], result_dynamic['num_trades'])

    logger.info(f"\nTotal Trades: {max_trades}")
    if max_trades >= 30:
        logger.success("✅ Sample size sufficient (n≥30) for statistical significance")
        statistical_confidence = "HIGH"
    elif max_trades >= 20:
        logger.warning("⚠️ Sample size adequate (20≤n<30), moderate confidence")
        statistical_confidence = "MEDIUM"
    elif max_trades >= 15:
        logger.warning("⚠️ Sample size marginal (15≤n<20), results may vary")
        statistical_confidence = "LOW"
    else:
        logger.error("❌ Sample size insufficient (n<15), cannot validate strategy")
        statistical_confidence = "VERY LOW"

    # 10. 최종 결론
    logger.info("\n" + "="*80)
    logger.info("10. FINAL ASSESSMENT")
    logger.info("="*80)

    logger.info(f"\n데이터 제약:")
    logger.info(f"  전체 기간: 59일 (2개월)")
    logger.info(f"  Test 기간: {len(test_df)/288:.1f}일 (30% split)")
    logger.info(f"  통계적 신뢰도: {statistical_confidence}")

    logger.info(f"\n성과 비교:")
    logger.info(f"  이전 결과 (15% Test): +6.00%, 6 trades, PF 2.83")
    logger.info(f"  Fixed (30% Test): {result_fixed['total_return_pct']:+.2f}%, {result_fixed['num_trades']} trades, PF {result_fixed['profit_factor']:.2f}")
    logger.info(f"  Dynamic (30% Test): {result_dynamic['total_return_pct']:+.2f}%, {result_dynamic['num_trades']} trades, PF {result_dynamic['profit_factor']:.2f}")

    if improvement > 0:
        logger.info(f"\n동적 SL/TP 개선도: +{improvement:.2f}%")
        if improvement > 2.0:
            logger.success("✅ 동적 SL/TP가 상당한 개선을 보임!")
        else:
            logger.info("✅ 동적 SL/TP가 소폭 개선")
    elif improvement < -2.0:
        logger.warning(f"\n⚠️ 동적 SL/TP가 악화: {improvement:.2f}%")
        logger.warning("   Fixed SL/TP 유지 권장")
    else:
        logger.info(f"\n동적 SL/TP 차이: {improvement:.2f}% (유사)")

    # 11. 다음 단계 권장
    logger.info("\n" + "="*80)
    logger.info("11. NEXT STEPS")
    logger.info("="*80)

    best_result = result_dynamic if result_dynamic['total_return_pct'] > result_fixed['total_return_pct'] else result_fixed
    best_mode = 'Dynamic' if result_dynamic['total_return_pct'] > result_fixed['total_return_pct'] else 'Fixed'

    logger.info(f"\n최적 전략: {best_mode} SL/TP")
    logger.info(f"  Return: {best_result['total_return_pct']:+.2f}%")
    logger.info(f"  Profit Factor: {best_result['profit_factor']:.2f}")
    logger.info(f"  Trades: {best_result['num_trades']}")

    if max_trades >= 20 and best_result['profit_factor'] > 2.5:
        logger.success("\n🎯 VALIDATION PASSED")
        logger.info("권장사항:")
        logger.info("  1. 현재 설정으로 페이퍼 트레이딩 준비")
        logger.info("  2. 실시간 데이터 파이프라인 구축")
        logger.info("  3. 모니터링 대시보드 설정")
    elif max_trades >= 15 and best_result['profit_factor'] > 2.0:
        logger.warning("\n⚠️ MARGINAL VALIDATION")
        logger.info("권장사항:")
        logger.info("  1. 더 긴 기간 데이터 수집 (3-6개월)")
        logger.info("  2. 또는 조심스럽게 페이퍼 트레이딩 시작")
        logger.info("  3. Walk-forward validation 고려")
    else:
        logger.warning("\n⚠️ VALIDATION UNCERTAIN")
        logger.info("권장사항:")
        logger.info("  1. Walk-forward validation 구현")
        logger.info("  2. 더 긴 데이터 수집")
        logger.info("  3. 대안 전략 고려 (regime detection, portfolio approach)")

    logger.info("\n" + "="*80)
    logger.info("✅ Analysis Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
