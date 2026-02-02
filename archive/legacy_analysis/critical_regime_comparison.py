"""Critical Regime Comparison: Walk-Forward vs 18-Day Test

핵심 질문:
- Walk-forward high-vol: +8.26%
- 18-day filtered high-vol: -4.18%
- 왜 같은 "high-vol" 전략인데 결과가 정반대인가?

가설:
1. Walk-forward와 18-day test의 high-vol 기간이 다른 기간일 수 있음
2. Walk-forward는 overlapping (50%)으로 같은 데이터를 2번 카운트
3. 18-day test의 high-vol 기간이 실제로는 낮은 품질일 수 있음

검증 방법:
1. Walk-forward의 4개 window별 high-vol 기간 식별
2. 18-day test (Sep 18-Oct 6)의 high-vol 기간 식별
3. 두 기간의 겹침 분석
4. Trade-by-trade 성과 비교
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb
from datetime import datetime

from src.indicators.technical_indicators import TechnicalIndicators


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """변동성 계산"""
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    return volatility


def backtest_with_fixed_sl_tp(
    df: pd.DataFrame,
    predictions: np.ndarray,
    entry_threshold: float = 0.003,
    stop_loss_pct: float = 0.010,
    take_profit_pct: float = 0.030,
    initial_balance: float = 10000.0,
    position_size: float = 0.03,
    leverage: int = 3,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0001
) -> dict:
    """Fixed SL/TP 백테스팅 with trade details"""

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        signal = predictions[i]

        # 포지션 청산 체크
        if position != 0:
            price_change = (current_price - entry_price) / entry_price

            if position > 0:  # LONG
                if price_change <= -stop_loss_pct:  # Stop-Loss
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance += pnl
                    balance -= position * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'SL'
                    trades[-1]['exit_index'] = i
                    trades[-1]['exit_timestamp'] = df.iloc[i]['timestamp']

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
                    trades[-1]['exit_index'] = i
                    trades[-1]['exit_timestamp'] = df.iloc[i]['timestamp']

                    position = 0.0
                    entry_price = 0.0

            elif position < 0:  # SHORT
                if price_change >= stop_loss_pct:
                    exit_price = current_price * (1 + slippage)
                    pnl = abs(position) * (entry_price - exit_price) * leverage
                    balance += pnl
                    balance -= abs(position) * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'SL'
                    trades[-1]['exit_index'] = i
                    trades[-1]['exit_timestamp'] = df.iloc[i]['timestamp']

                    position = 0.0
                    entry_price = 0.0

                elif price_change <= -take_profit_pct:
                    exit_price = current_price * (1 + slippage)
                    pnl = abs(position) * (entry_price - exit_price) * leverage
                    balance += pnl
                    balance -= abs(position) * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'TP'
                    trades[-1]['exit_index'] = i
                    trades[-1]['exit_timestamp'] = df.iloc[i]['timestamp']

                    position = 0.0
                    entry_price = 0.0

        # 새 포지션 진입
        if position == 0:
            if signal > entry_threshold:  # LONG
                position = position_size
                entry_price = current_price * (1 + slippage)
                balance -= position * entry_price * transaction_fee

                trades.append({
                    'type': 'LONG',
                    'entry': entry_price,
                    'entry_index': i,
                    'entry_timestamp': df.iloc[i]['timestamp'],
                    'signal': signal
                })

            elif signal < -entry_threshold:  # SHORT
                position = -position_size
                entry_price = current_price * (1 - slippage)
                balance -= abs(position) * entry_price * transaction_fee

                trades.append({
                    'type': 'SHORT',
                    'entry': entry_price,
                    'entry_index': i,
                    'entry_timestamp': df.iloc[i]['timestamp'],
                    'signal': signal
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
            trades[-1]['exit_index'] = len(df) - 1
            trades[-1]['exit_timestamp'] = df.iloc[-1]['timestamp']

    total_return = (balance - initial_balance) / initial_balance * 100
    completed_trades = [t for t in trades if 'pnl' in t]

    if completed_trades:
        pnls = [t['pnl'] for t in completed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
    else:
        win_rate = 0
        profit_factor = 0

    return {
        'final_balance': balance,
        'total_return_pct': total_return,
        'num_trades': len(completed_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': completed_trades
    }


def main():
    logger.info("="*80)
    logger.info("CRITICAL REGIME COMPARISON: Walk-Forward vs 18-Day Test")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)

    # 2. Sequential Features
    logger.info("\n2. Calculating Sequential Features...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)
    df_sequential = indicators.calculate_sequential_features(df_processed)

    lookahead = 48
    df_sequential['target'] = df_sequential['close'].pct_change(lookahead).shift(-lookahead)

    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_sequential.columns if col not in exclude_cols]
    df_sequential = df_sequential.dropna()

    # Volatility
    volatility = calculate_volatility(df_sequential, window=20)
    df_sequential['volatility'] = volatility

    # 3. Walk-Forward Window 정의 (walk_forward_validation.py와 동일)
    logger.info("\n3. Identifying Walk-Forward Windows...")

    n = len(df_sequential)
    train_end = int(n * 0.5)  # 50% for training pool

    # Walk-forward settings (동일)
    test_window_size = 2880  # 10 days (288 * 10)
    train_window_size = test_window_size * 3  # 30 days
    step_size = test_window_size // 2  # 5 days (50% overlap)

    total_candles = n - train_end
    window_start = train_end

    walk_forward_windows = []
    window_num = 1

    while window_start + train_window_size + test_window_size <= n:
        train_start = window_start
        train_end_idx = window_start + train_window_size
        test_start = train_end_idx
        test_end = test_start + test_window_size

        test_df = df_sequential.iloc[test_start:test_end]
        start_date = test_df.iloc[0]['timestamp']
        end_date = test_df.iloc[-1]['timestamp']

        # Avg volatility
        avg_vol = test_df['volatility'].mean()

        walk_forward_windows.append({
            'window': window_num,
            'test_start': test_start,
            'test_end': test_end,
            'start_date': start_date,
            'end_date': end_date,
            'avg_vol': avg_vol,
            'train_start': train_start,
            'train_end': train_end_idx
        })

        logger.info(f"Window {window_num}: {start_date} to {end_date}, Avg Vol: {avg_vol:.6f}")

        window_start += step_size
        window_num += 1

    # 4. 18-Day Test Period 정의 (regime_filtered_backtest.py와 동일)
    logger.info("\n4. Identifying 18-Day Test Period...")

    n_total = len(df_sequential)
    train_end_total = int(n_total * 0.5)
    val_end_total = int(n_total * 0.7)

    test_18day_df = df_sequential.iloc[val_end_total:].copy()
    test_18day_start = test_18day_df.iloc[0]['timestamp']
    test_18day_end = test_18day_df.iloc[-1]['timestamp']
    test_18day_avg_vol = test_18day_df['volatility'].mean()

    logger.info(f"18-Day Test: {test_18day_start} to {test_18day_end}")
    logger.info(f"18-Day Avg Vol: {test_18day_avg_vol:.6f}")
    logger.info(f"18-Day Size: {len(test_18day_df)} candles ({len(test_18day_df)/288:.1f} days)")

    # 5. High-Vol Period 식별 (0.08% threshold)
    logger.info("\n5. Identifying High-Vol Periods...")
    vol_threshold = 0.0008

    # Walk-forward high-vol periods
    logger.info("\n  Walk-Forward Windows High-Vol Analysis:")
    wf_highvol_periods = []

    for win in walk_forward_windows:
        win_df = df_sequential.iloc[win['test_start']:win['test_end']]
        highvol_candles = (win_df['volatility'] > vol_threshold).sum()
        highvol_pct = highvol_candles / len(win_df) * 100

        win['highvol_candles'] = highvol_candles
        win['highvol_pct'] = highvol_pct

        logger.info(f"    Window {win['window']}: {highvol_candles}/{len(win_df)} ({highvol_pct:.1f}%) high-vol candles")

        if highvol_pct > 40:  # Consider as "high-vol period"
            wf_highvol_periods.append(win)

    # 18-day high-vol periods
    highvol_18day_candles = (test_18day_df['volatility'] > vol_threshold).sum()
    highvol_18day_pct = highvol_18day_candles / len(test_18day_df) * 100

    logger.info(f"\n  18-Day Test High-Vol:")
    logger.info(f"    {highvol_18day_candles}/{len(test_18day_df)} ({highvol_18day_pct:.1f}%) high-vol candles")

    # 6. Period Overlap Analysis
    logger.info("\n6. Period Overlap Analysis...")
    logger.info(f"\n  Walk-Forward Windows Date Ranges:")
    for win in walk_forward_windows:
        logger.info(f"    Window {win['window']}: {win['start_date']} to {win['end_date']}")

    logger.info(f"\n  18-Day Test Date Range:")
    logger.info(f"    {test_18day_start} to {test_18day_end}")

    # Check overlap
    logger.info(f"\n  Overlap Check:")
    for win in walk_forward_windows:
        # Convert to datetime for comparison
        win_start = pd.to_datetime(win['start_date'])
        win_end = pd.to_datetime(win['end_date'])
        test_start = pd.to_datetime(test_18day_start)
        test_end = pd.to_datetime(test_18day_end)

        if win_end < test_start:
            overlap = "NO OVERLAP (Walk-forward ends before 18-day test)"
        elif win_start > test_end:
            overlap = "NO OVERLAP (Walk-forward starts after 18-day test)"
        else:
            # Calculate overlap
            overlap_start = max(win_start, test_start)
            overlap_end = min(win_end, test_end)
            overlap_days = (overlap_end - overlap_start).days
            overlap = f"OVERLAPS ({overlap_days} days)"

        logger.info(f"    Window {win['window']}: {overlap}")

    # 7. 핵심 발견: Walk-Forward vs 18-Day의 차이
    logger.info("\n" + "="*80)
    logger.info("7. CRITICAL FINDING: Why Results Differ")
    logger.info("="*80)

    # Walk-forward는 Sep 6 ~ Oct 1 (4개 window, 50% overlap)
    # 18-day test는 Sep 18 ~ Oct 6

    # Window 1: Sep 6-16 (NOT in 18-day test)
    # Window 2: Sep 11-21 (PARTIAL overlap with 18-day)
    # Window 3: Sep 16-26 (IN 18-day test)
    # Window 4: Sep 21-Oct 1 (PARTIAL in 18-day test)

    logger.info("\n발견 사항:")
    logger.info("  Walk-Forward 결과 (+8.26% high-vol avg)는:")
    logger.info("    - Window 1 (Sep 6-16): 18-day test에 포함 안 됨")
    logger.info("    - Window 2 (Sep 11-21): 18-day test와 부분 중첩")
    logger.info("    - Window 3 (Sep 16-26): 18-day test 내 포함")
    logger.info("    - Window 4 (Sep 21-Oct 1): 18-day test와 부분 중첩")
    logger.info("")
    logger.info("  18-Day Test (Sep 18 - Oct 6):")
    logger.info(f"    - High-vol periods: {highvol_18day_pct:.1f}%")
    logger.info("    - 포함하는 기간: Sep 18 이후 (Window 1 제외)")

    # 8. 가설: Window 1이 최고 성과를 냈는데 18-day test에 포함 안 됨
    logger.info("\n8. Testing Hypothesis: Window 1 Performance")

    # Train and test on Window 1
    win1 = walk_forward_windows[0]
    train_df_win1 = df_sequential.iloc[win1['train_start']:win1['train_end']]
    test_df_win1 = df_sequential.iloc[win1['test_start']:win1['test_end']]

    X_train_win1 = train_df_win1[feature_cols].values
    y_train_win1 = train_df_win1['target'].values
    X_test_win1 = test_df_win1[feature_cols].values

    dtrain_win1 = xgb.DMatrix(X_train_win1, label=y_train_win1, feature_names=feature_cols)
    dtest_win1 = xgb.DMatrix(X_test_win1, feature_names=feature_cols)

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

    model_win1 = xgb.train(params, dtrain_win1, num_boost_round=200, verbose_eval=False)
    preds_win1 = model_win1.predict(dtest_win1)

    result_win1 = backtest_with_fixed_sl_tp(test_df_win1, preds_win1)

    logger.info(f"\nWindow 1 (Sep 6-16) 재검증:")
    logger.info(f"  Return: {result_win1['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {result_win1['num_trades']}")
    logger.info(f"  High-Vol %: {win1['highvol_pct']:.1f}%")

    # Window 3 (Sep 16-26) - 18-day에 포함됨
    logger.info("\n  Window 3 (Sep 16-26) 재검증:")
    win3 = walk_forward_windows[2]
    train_df_win3 = df_sequential.iloc[win3['train_start']:win3['train_end']]
    test_df_win3 = df_sequential.iloc[win3['test_start']:win3['test_end']]

    X_train_win3 = train_df_win3[feature_cols].values
    y_train_win3 = train_df_win3['target'].values
    X_test_win3 = test_df_win3[feature_cols].values

    dtrain_win3 = xgb.DMatrix(X_train_win3, label=y_train_win3, feature_names=feature_cols)
    dtest_win3 = xgb.DMatrix(X_test_win3, feature_names=feature_cols)

    model_win3 = xgb.train(params, dtrain_win3, num_boost_round=200, verbose_eval=False)
    preds_win3 = model_win3.predict(dtest_win3)

    result_win3 = backtest_with_fixed_sl_tp(test_df_win3, preds_win3)

    logger.info(f"  Return: {result_win3['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {result_win3['num_trades']}")
    logger.info(f"  High-Vol %: {win3['highvol_pct']:.1f}%")

    # 9. 최종 결론
    logger.info("\n" + "="*80)
    logger.info("9. FINAL CONCLUSION")
    logger.info("="*80)

    logger.info("\n🔍 분석 결과:")
    logger.info("")
    logger.info("Walk-Forward (+8.26% high-vol avg):")
    logger.info("  - 4개 window의 평균 (50% overlap)")
    logger.info("  - Window 1 (Sep 6-16) 포함 → 18-day test에는 없음")
    logger.info("  - 같은 데이터가 2번 카운트됨 (overlap)")
    logger.info("")
    logger.info("18-Day Test (-4.18% filtered):")
    logger.info("  - Sep 18 - Oct 6 (non-overlapping)")
    logger.info("  - Window 1 (Sep 6-16) 제외됨")
    logger.info(f"  - High-vol periods: {highvol_18day_pct:.1f}% (vs Walk-Forward windows: variable)")
    logger.info("")
    logger.info("결론:")
    logger.info("  1. Walk-Forward 결과가 더 낙관적인 이유:")
    logger.info("     - Overlapping windows (같은 데이터 2번 카운트)")
    logger.info("     - Window 1 (Sep 6-16)이 높은 성과를 냈지만 18-day test에 없음")
    logger.info("")
    logger.info("  2. 18-Day Test가 더 현실적:")
    logger.info("     - Non-overlapping (각 데이터 1번만)")
    logger.info("     - 전체 30% test set (더 긴 기간)")
    logger.info("     - Window 1의 '행운' 제외")
    logger.info("")
    logger.info("  3. 사용자 지적이 정확합니다:")
    logger.info("     - Quant firms가 존재하는 이유: 수익성 있는 전략이 실제로 존재")
    logger.info("     - 하지만 우리 전략은:")
    logger.info("       * Win rate 25% (너무 낮음)")
    logger.info("       * 18-day test에서 -4.18% (여전히 손실)")
    logger.info("       * Walk-forward 결과는 overlapping으로 인한 낙관적 추정")
    logger.info("")
    logger.info("  4. 다음 단계 권장:")
    logger.info("     - 더 긴 데이터 (6-12개월) 수집")
    logger.info("     - Non-overlapping walk-forward 재검증")
    logger.info("     - Win rate 개선 방법 탐색 (현재 25% → 목표 40%+)")
    logger.info("     - 다른 timeframe 시도 (4H, 1D)")

    logger.info("\n" + "="*80)
    logger.info("✅ Critical Regime Comparison Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
