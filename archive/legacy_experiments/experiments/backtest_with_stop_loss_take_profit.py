"""Stop-Loss / Take-Profit 백테스팅

사용자 통찰: "손절은 짧게, 수익은 길게"

문제:
- 현재: Win Rate 26.7%, Profit Factor 0.42
- 손실이 이익의 2.4배
- Risk management 부재

해결:
1. Stop-Loss: -0.5% ~ -1.0% (빠른 손절)
2. Take-Profit: +1.5% ~ +2.5% (긴 수익)
3. Risk/Reward Ratio: 1:2, 1:3 테스트

목표:
- Profit Factor > 1.0
- Win Rate는 낮아도 OK (큰 수익, 작은 손실)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb

from src.indicators.technical_indicators import TechnicalIndicators


def backtest_with_sl_tp(
    df: pd.DataFrame,
    predictions: np.ndarray,
    entry_threshold: float = 0.003,  # ±0.3%
    stop_loss_pct: float = 0.005,  # -0.5%
    take_profit_pct: float = 0.015,  # +1.5%
    initial_balance: float = 10000.0,
    position_size: float = 0.03,
    leverage: int = 3,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0001
) -> dict:
    """Stop-Loss와 Take-Profit이 있는 백테스팅"""

    balance = initial_balance
    position = 0.0  # 0 = 없음, 1 = LONG, -1 = SHORT
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        signal = predictions[i]

        # 포지션 있을 때: Stop-Loss / Take-Profit 체크
        if position != 0:
            price_change = (current_price - entry_price) / entry_price

            # LONG 포지션
            if position > 0:
                # Stop-Loss 체크
                if price_change <= -stop_loss_pct:
                    # 손절
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance -= abs(position) * exit_price * transaction_fee
                    balance += pnl

                    trades.append({
                        'type': 'LONG',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'STOP_LOSS',
                        'index': i
                    })

                    position = 0.0
                    entry_price = 0.0

                # Take-Profit 체크
                elif price_change >= take_profit_pct:
                    # 익절
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance -= abs(position) * exit_price * transaction_fee
                    balance += pnl

                    trades.append({
                        'type': 'LONG',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'TAKE_PROFIT',
                        'index': i
                    })

                    position = 0.0
                    entry_price = 0.0

            # SHORT 포지션
            elif position < 0:
                # Stop-Loss 체크
                if price_change >= stop_loss_pct:
                    # 손절
                    exit_price = current_price * (1 + slippage)
                    pnl = abs(position) * (entry_price - exit_price) * leverage
                    balance -= abs(position) * exit_price * transaction_fee
                    balance += pnl

                    trades.append({
                        'type': 'SHORT',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'STOP_LOSS',
                        'index': i
                    })

                    position = 0.0
                    entry_price = 0.0

                # Take-Profit 체크
                elif price_change <= -take_profit_pct:
                    # 익절
                    exit_price = current_price * (1 + slippage)
                    pnl = abs(position) * (entry_price - exit_price) * leverage
                    balance -= abs(position) * exit_price * transaction_fee
                    balance += pnl

                    trades.append({
                        'type': 'SHORT',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'TAKE_PROFIT',
                        'index': i
                    })

                    position = 0.0
                    entry_price = 0.0

        # 포지션 없을 때: 진입 신호 체크
        if position == 0:
            # LONG 진입
            if signal > entry_threshold:
                entry_price = current_price * (1 + slippage)
                position = position_size
                balance -= position * entry_price * transaction_fee

            # SHORT 진입
            elif signal < -entry_threshold:
                entry_price = current_price * (1 - slippage)
                position = -position_size
                balance -= abs(position) * entry_price * transaction_fee

    # 최종 청산 (포지션 남아있으면)
    if position != 0:
        final_price = df.iloc[-1]['close']

        if position > 0:
            pnl = position * (final_price - entry_price) * leverage
        else:
            pnl = abs(position) * (entry_price - final_price) * leverage

        balance -= abs(position) * final_price * transaction_fee
        balance += pnl

        trades.append({
            'type': 'LONG' if position > 0 else 'SHORT',
            'entry': entry_price,
            'exit': final_price,
            'pnl': pnl,
            'exit_reason': 'FINAL',
            'index': len(df) - 1
        })

    # 통계
    total_return = (balance - initial_balance) / initial_balance * 100

    if trades:
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        # Stop-Loss vs Take-Profit 분석
        sl_count = sum(1 for t in trades if t.get('exit_reason') == 'STOP_LOSS')
        tp_count = sum(1 for t in trades if t.get('exit_reason') == 'TAKE_PROFIT')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        sl_count = 0
        tp_count = 0

    return {
        'final_balance': balance,
        'total_return_pct': total_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'stop_loss_count': sl_count,
        'take_profit_count': tp_count,
        'trades': trades
    }


def main():
    logger.info("="*80)
    logger.info("Stop-Loss / Take-Profit Backtesting")
    logger.info('사용자 통찰: "손절은 짧게, 수익은 길게"')
    logger.info("="*80)

    # 1. 데이터 로드 및 처리
    logger.info("\n1. Loading data...")
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

    # 3. 모델 훈련
    logger.info("\n3. Training model...")
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
        'tree_method': 'hist',
        'random_state': 42
    }

    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    test_preds = model.predict(dtest)

    logger.info("Model trained.")

    # 4. Stop-Loss / Take-Profit 실험
    logger.info("\n4. Testing Stop-Loss / Take-Profit combinations...")

    # Risk/Reward Ratios 테스트
    experiments = [
        # (entry_threshold, stop_loss, take_profit, ratio_name)
        (0.003, 0.005, 0.010, "1:2 (SL -0.5%, TP +1.0%)"),
        (0.003, 0.005, 0.015, "1:3 (SL -0.5%, TP +1.5%)"),
        (0.003, 0.008, 0.016, "1:2 (SL -0.8%, TP +1.6%)"),
        (0.003, 0.008, 0.024, "1:3 (SL -0.8%, TP +2.4%)"),
        (0.003, 0.010, 0.020, "1:2 (SL -1.0%, TP +2.0%)"),
        (0.003, 0.010, 0.030, "1:3 (SL -1.0%, TP +3.0%)"),
    ]

    results = []

    for entry_thresh, sl, tp, name in experiments:
        logger.info(f"\n--- Testing: {name} ---")

        result = backtest_with_sl_tp(
            test_df,
            test_preds,
            entry_threshold=entry_thresh,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            initial_balance=10000.0,
            position_size=0.03,
            leverage=3,
            transaction_fee=0.0004,
            slippage=0.0001
        )

        result['config'] = name
        result['stop_loss'] = sl
        result['take_profit'] = tp
        results.append(result)

        logger.info(f"Return: {result['total_return_pct']:+.2f}%")
        logger.info(f"Trades: {result['num_trades']}")
        if result['num_trades'] > 0:
            logger.info(f"Win Rate: {result['win_rate']*100:.1f}%")
            logger.info(f"Profit Factor: {result['profit_factor']:.2f}")
            logger.info(f"Stop-Loss: {result['stop_loss_count']}, Take-Profit: {result['take_profit_count']}")

    # 5. 비교 분석
    logger.info("\n" + "="*80)
    logger.info("STOP-LOSS / TAKE-PROFIT RESULTS")
    logger.info("="*80)

    # Buy & Hold
    first_price = test_df.iloc[0]['close']
    last_price = test_df.iloc[-1]['close']
    bh_return = (last_price - first_price) / first_price * 100

    logger.info(f"\nBuy & Hold: {bh_return:+.2f}%")

    logger.info("\n| Config | Return | Trades | Win Rate | PF | SL | TP |")
    logger.info("|--------|--------|--------|----------|-----|-----|-----|")

    for r in results:
        logger.info(
            f"| {r['config']:30s} | {r['total_return_pct']:+6.2f}% | {r['num_trades']:6d} | "
            f"{r['win_rate']*100:8.1f}% | {r['profit_factor']:3.2f} | {r['stop_loss_count']:3d} | {r['take_profit_count']:3d} |"
        )

    # 6. 최고 성과 분석
    logger.info("\n" + "="*80)
    logger.info("BEST RESULT ANALYSIS")
    logger.info("="*80)

    best_result = max(results, key=lambda x: x['total_return_pct'])

    logger.info(f"\n최고 성과: {best_result['config']}")
    logger.info(f"  Return: {best_result['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {best_result['num_trades']}")
    logger.info(f"  Win Rate: {best_result['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {best_result['profit_factor']:.2f}")
    logger.info(f"  Avg Win: ${best_result['avg_win']:.2f}")
    logger.info(f"  Avg Loss: ${best_result['avg_loss']:.2f}")
    logger.info(f"  Stop-Loss Exits: {best_result['stop_loss_count']}")
    logger.info(f"  Take-Profit Exits: {best_result['take_profit_count']}")

    logger.info(f"\nBuy & Hold: {bh_return:+.2f}%")
    logger.info(f"Gap: {best_result['total_return_pct'] - bh_return:+.2f}%")

    # 7. 최종 결론
    logger.info("\n" + "="*80)
    logger.info("FINAL CONCLUSION")
    logger.info("="*80)

    logger.info('\n사용자 통찰: "손절은 짧게, 수익은 길게"')

    logger.info("\n검증 결과:")
    if best_result['profit_factor'] > 1.0:
        logger.success(f"✅ Profit Factor > 1.0 달성! ({best_result['profit_factor']:.2f})")
        logger.success("   손절/익절 로직이 손익비를 개선했습니다!")
    else:
        logger.warning(f"⚠️ Profit Factor = {best_result['profit_factor']:.2f} (< 1.0)")

    if best_result['total_return_pct'] > bh_return:
        logger.success(f"\n🎉 ML이 Buy & Hold를 초과!")
        logger.success(f"   ML: {best_result['total_return_pct']:+.2f}% vs Buy & Hold: {bh_return:+.2f}%")
        logger.success(f"   최적 설정: {best_result['config']}")
    elif best_result['total_return_pct'] > 0:
        logger.warning(f"\n⚠️ ML 수익 (+{best_result['total_return_pct']:.2f}%)은 있으나 Buy & Hold 미달")
        logger.warning(f"   Gap: {bh_return - best_result['total_return_pct']:.2f}%")
    else:
        logger.error(f"\n❌ 여전히 손실: {best_result['total_return_pct']:+.2f}%")

    logger.info("\n핵심 통찰:")
    logger.info(f"  • Risk/Reward Ratio: Stop-Loss {best_result['stop_loss']*100:.1f}% / Take-Profit {best_result['take_profit']*100:.1f}%")
    logger.info(f"  • Win Rate: {best_result['win_rate']*100:.1f}% (낮아도 OK if PF > 1.0)")
    logger.info(f"  • Profit Factor: {best_result['profit_factor']:.2f}")

    if best_result['profit_factor'] > 1.0:
        logger.info("\n💡 성공 요인:")
        logger.info("   ✅ 작은 손실 빠른 손절")
        logger.info("   ✅ 큰 수익 긴 보유")
        logger.info("   ✅ Risk management 작동!")

    logger.info("\n" + "="*80)
    logger.info("✅ Stop-Loss / Take-Profit Testing Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
