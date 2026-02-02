"""Walk-Forward Validation for Trading Strategy

비판적 질문:
- 전략이 완전히 실패인가?
- 아니면 특정 시장 국면에서만 작동하는가?

방법:
1. Rolling window approach (10일 test periods)
2. 각 기간마다 Train → Test
3. 성과의 일관성 분석
4. Regime pattern 식별

목표:
- 전략의 진짜 문제가 무엇인지 확인
- Regime-dependent라면 → filter 가능
- 근본적 실패라면 → 전략 포기
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
from datetime import datetime, timedelta

from src.indicators.technical_indicators import TechnicalIndicators


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
    """Fixed SL/TP 백테스팅 (간단 버전)"""

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
                    'index': i
                })

            elif signal < -entry_threshold:  # SHORT
                position = -position_size
                entry_price = current_price * (1 - slippage)
                balance -= abs(position) * entry_price * transaction_fee

                trades.append({
                    'type': 'SHORT',
                    'entry': entry_price,
                    'index': i
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
    logger.info("Walk-Forward Validation")
    logger.info("비판적 검증: 전략 실패 vs Regime Dependency")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"Total data: {len(df)} candles")
    logger.info(f"Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
    logger.info(f"Duration: {(df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).days} days")

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

    logger.info(f"Processed data: {len(df_sequential)} candles")

    # 3. Walk-Forward Validation Setup
    logger.info("\n3. Walk-Forward Validation Setup...")

    # 10일 = 2,880 캔들 (5분봉 12개/시간 × 24시간 × 10일)
    test_window_size = 2880  # 10 days
    train_window_size = test_window_size * 3  # 30 days for training

    # Rolling windows
    total_candles = len(df_sequential)
    results = []

    window_start = 0
    window_num = 0

    while window_start + train_window_size + test_window_size <= total_candles:
        window_num += 1

        train_start = window_start
        train_end = window_start + train_window_size
        test_start = train_end
        test_end = test_start + test_window_size

        # 데이터 분할
        train_df = df_sequential.iloc[train_start:train_end].copy()
        test_df = df_sequential.iloc[test_start:test_end].copy()

        train_period = f"{train_df.iloc[0]['timestamp'].strftime('%m/%d')} - {train_df.iloc[-1]['timestamp'].strftime('%m/%d')}"
        test_period = f"{test_df.iloc[0]['timestamp'].strftime('%m/%d')} - {test_df.iloc[-1]['timestamp'].strftime('%m/%d')}"

        logger.info(f"\n--- Window {window_num} ---")
        logger.info(f"Train: {train_period} ({len(train_df)} candles)")
        logger.info(f"Test: {test_period} ({len(test_df)} candles)")

        # 모델 훈련
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
        test_r2 = r2_score(y_test, test_preds)

        # 백테스트
        backtest_result = backtest_with_fixed_sl_tp(
            test_df,
            test_preds,
            entry_threshold=0.003,
            stop_loss_pct=0.010,
            take_profit_pct=0.030,
            initial_balance=10000.0,
            position_size=0.03,
            leverage=3,
            transaction_fee=0.0004,
            slippage=0.0001
        )

        # Buy & Hold
        first_price = test_df.iloc[0]['close']
        last_price = test_df.iloc[-1]['close']
        bh_return = (last_price - first_price) / first_price * 100

        # 시장 특성
        price_std = test_df['close'].pct_change().std() * 100
        avg_volume = test_df['volume'].mean()

        # 결과 저장
        result = {
            'window': window_num,
            'test_period': test_period,
            'test_start': test_df.iloc[0]['timestamp'],
            'test_end': test_df.iloc[-1]['timestamp'],
            'ml_return': backtest_result['total_return_pct'],
            'bh_return': bh_return,
            'trades': backtest_result['num_trades'],
            'win_rate': backtest_result['win_rate'],
            'profit_factor': backtest_result['profit_factor'],
            'r2': test_r2,
            'price_std': price_std,
            'avg_volume': avg_volume
        }
        results.append(result)

        logger.info(f"ML Return: {backtest_result['total_return_pct']:+.2f}%")
        logger.info(f"Buy & Hold: {bh_return:+.2f}%")
        logger.info(f"Trades: {backtest_result['num_trades']}, Win Rate: {backtest_result['win_rate']*100:.1f}%, PF: {backtest_result['profit_factor']:.2f}")
        logger.info(f"R²: {test_r2:.4f}, Volatility: {price_std:.3f}%")

        # 다음 window로 이동 (5일씩 shift)
        window_start += test_window_size // 2  # 50% overlap

    # 4. 결과 분석
    logger.info("\n" + "="*80)
    logger.info("4. WALK-FORWARD VALIDATION RESULTS")
    logger.info("="*80)

    results_df = pd.DataFrame(results)

    logger.info(f"\nTotal Windows: {len(results)}")
    logger.info("")
    logger.info("| Window | Test Period | ML Return | B&H Return | Trades | Win% | PF | R² |")
    logger.info("|--------|-------------|-----------|------------|--------|------|----|----|")

    for r in results:
        logger.info(
            f"| {r['window']:6d} | {r['test_period']:11s} | {r['ml_return']:+9.2f}% | "
            f"{r['bh_return']:+10.2f}% | {r['trades']:6d} | {r['win_rate']*100:4.1f}% | "
            f"{r['profit_factor']:4.2f} | {r['r2']:6.2f} |"
        )

    # 5. 통계 분석
    logger.info("\n" + "="*80)
    logger.info("5. STATISTICAL ANALYSIS")
    logger.info("="*80)

    ml_returns = results_df['ml_return'].values
    bh_returns = results_df['bh_return'].values

    logger.info(f"\nML Strategy:")
    logger.info(f"  Mean Return: {ml_returns.mean():+.2f}%")
    logger.info(f"  Std Dev: {ml_returns.std():.2f}%")
    logger.info(f"  Best: {ml_returns.max():+.2f}%")
    logger.info(f"  Worst: {ml_returns.min():+.2f}%")
    logger.info(f"  Win Periods: {(ml_returns > 0).sum()}/{len(ml_returns)} ({(ml_returns > 0).mean()*100:.1f}%)")

    logger.info(f"\nBuy & Hold:")
    logger.info(f"  Mean Return: {bh_returns.mean():+.2f}%")
    logger.info(f"  Std Dev: {bh_returns.std():.2f}%")
    logger.info(f"  Best: {bh_returns.max():+.2f}%")
    logger.info(f"  Worst: {bh_returns.min():+.2f}%")
    logger.info(f"  Win Periods: {(bh_returns > 0).sum()}/{len(bh_returns)} ({(bh_returns > 0).mean()*100:.1f}%)")

    # ML vs B&H
    ml_beats_bh = (ml_returns > bh_returns).sum()
    logger.info(f"\nML Beats B&H: {ml_beats_bh}/{len(results)} periods ({ml_beats_bh/len(results)*100:.1f}%)")

    # 6. Regime 분석
    logger.info("\n" + "="*80)
    logger.info("6. REGIME ANALYSIS")
    logger.info("="*80)

    # ML이 이긴 기간 vs 진 기간
    winning_periods = results_df[results_df['ml_return'] > results_df['bh_return']]
    losing_periods = results_df[results_df['ml_return'] <= results_df['bh_return']]

    if len(winning_periods) > 0:
        logger.info(f"\nML Winning Periods ({len(winning_periods)}):")
        logger.info(f"  Avg ML Return: {winning_periods['ml_return'].mean():+.2f}%")
        logger.info(f"  Avg B&H Return: {winning_periods['bh_return'].mean():+.2f}%")
        logger.info(f"  Avg Volatility: {winning_periods['price_std'].mean():.3f}%")
        logger.info(f"  Avg Profit Factor: {winning_periods['profit_factor'].mean():.2f}")
        logger.info(f"  Avg R²: {winning_periods['r2'].mean():.4f}")

    if len(losing_periods) > 0:
        logger.info(f"\nML Losing Periods ({len(losing_periods)}):")
        logger.info(f"  Avg ML Return: {losing_periods['ml_return'].mean():+.2f}%")
        logger.info(f"  Avg B&H Return: {losing_periods['bh_return'].mean():+.2f}%")
        logger.info(f"  Avg Volatility: {losing_periods['price_std'].mean():.3f}%")
        logger.info(f"  Avg Profit Factor: {losing_periods['profit_factor'].mean():.2f}")
        logger.info(f"  Avg R²: {losing_periods['r2'].mean():.4f}")

    # 7. 최종 결론
    logger.info("\n" + "="*80)
    logger.info("7. CRITICAL CONCLUSION")
    logger.info("="*80)

    avg_ml_return = ml_returns.mean()
    avg_bh_return = bh_returns.mean()
    consistency = (ml_returns > 0).mean()

    logger.info(f"\n평균 성과:")
    logger.info(f"  ML: {avg_ml_return:+.2f}%")
    logger.info(f"  B&H: {avg_bh_return:+.2f}%")
    logger.info(f"  Gap: {avg_ml_return - avg_bh_return:+.2f}%")

    logger.info(f"\n일관성:")
    logger.info(f"  ML Positive Periods: {consistency*100:.1f}%")
    logger.info(f"  ML Beats B&H: {ml_beats_bh/len(results)*100:.1f}%")

    # 최종 판정
    logger.info("\n" + "="*80)
    logger.info("FINAL VERDICT")
    logger.info("="*80)

    if avg_ml_return > avg_bh_return and consistency > 0.6:
        logger.success("\n✅ 전략이 일관되게 작동함")
        logger.info("  - 평균적으로 Buy & Hold 초과")
        logger.info("  - 60% 이상 기간에서 양의 수익")
        logger.info("  권장: Regime filter 없이 사용 가능")

    elif avg_ml_return > 0 and consistency > 0.5:
        logger.warning("\n⚠️ 전략이 부분적으로 작동함")
        logger.info("  - 평균적으로 양의 수익")
        logger.info(f"  - {consistency*100:.0f}%  기간에서 양의 수익")
        logger.info("  권장: Regime filter 구현으로 개선 시도")

    elif ml_beats_bh > 0:
        logger.warning("\n⚠️ 전략이 특정 국면에서만 작동함")
        logger.info(f"  - {ml_beats_bh}/{len(results)} 기간에서만 B&H 초과")
        logger.info("  - Regime-dependent 확인")
        logger.info("  권장: Regime detection 필수")

    else:
        logger.error("\n❌ 전략이 작동하지 않음")
        logger.info("  - 모든 기간에서 Buy & Hold 미달")
        logger.info("  - 근본적 실패")
        logger.info("  권장: 전략 포기 또는 완전 재설계")

    # 8. 다음 단계
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)

    if ml_beats_bh > 0:
        logger.info("\nRegime Filter 개발 방향:")
        logger.info("  1. Winning periods와 Losing periods 차이 분석")
        logger.info("  2. 변동성, 트렌드 강도, R² 등으로 regime 분류")
        logger.info("  3. Favorable regime에서만 거래")
        logger.info("  4. Unfavorable regime에서는 Hold 또는 B&H")
    else:
        logger.info("\n전략 포기 또는 근본적 재설계:")
        logger.info("  - 다른 시간봉 (15분, 1시간, 4시간)")
        logger.info("  - 다른 자산 (ETH, forex, stocks)")
        logger.info("  - 다른 접근법 (mean reversion, arbitrage)")
        logger.info("  - Buy & Hold 수용")

    logger.info("\n" + "="*80)
    logger.info("✅ Walk-Forward Validation Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
