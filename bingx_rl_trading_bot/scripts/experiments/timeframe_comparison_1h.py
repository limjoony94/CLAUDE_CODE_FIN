"""1시간봉 vs 5분봉 비교 실험

비판적 질문:
- 5분봉에서 Buy & Hold 대비 -8.19% gap
- 장기 시간봉(1h)에서 Sequential Features가 더 효과적인가?

가설:
1. 1시간봉: 노이즈↓ → R² 개선
2. Sequential Features: 추세 명확 → 더 효과적
3. 거래 비용: 비중 감소 → 수익성 개선

실험:
1. 5분봉 → 1시간봉 리샘플링
2. Sequential Features 계산
3. XGBoost 훈련
4. SL/TP (1:3) 백테스팅
5. 5분봉 vs 1시간봉 비교
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


def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """5분봉 → 1시간봉 리샘플링"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # OHLCV 리샘플링
    resampled = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    resampled = resampled.reset_index()
    logger.info(f"Resampled: {len(df)} 5m candles → {len(resampled)} 1h candles")

    return resampled


def backtest_with_sl_tp_1h(
    df: pd.DataFrame,
    predictions: np.ndarray,
    entry_threshold: float = 0.003,
    stop_loss_pct: float = 0.010,  # 1시간봉용: -1%
    take_profit_pct: float = 0.030,  # 1시간봉용: +3%
    initial_balance: float = 10000.0,
    position_size: float = 0.03,
    leverage: int = 3,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0001
) -> dict:
    """1시간봉용 백테스팅"""

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        signal = predictions[i]

        # 포지션 있을 때: SL/TP 체크
        if position != 0:
            price_change = (current_price - entry_price) / entry_price

            # LONG 포지션
            if position > 0:
                if price_change <= -stop_loss_pct:
                    # Stop-Loss
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

                elif price_change >= take_profit_pct:
                    # Take-Profit
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
                if price_change >= stop_loss_pct:
                    # Stop-Loss
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

                elif price_change <= -take_profit_pct:
                    # Take-Profit
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

        # 포지션 없을 때: 진입
        if position == 0:
            if signal > entry_threshold:
                entry_price = current_price * (1 + slippage)
                position = position_size
                balance -= position * entry_price * transaction_fee

            elif signal < -entry_threshold:
                entry_price = current_price * (1 - slippage)
                position = -position_size
                balance -= abs(position) * entry_price * transaction_fee

    # 최종 청산
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
    logger.info("1시간봉 vs 5분봉 비교 실험")
    logger.info("가설: 장기 시간봉에서 Sequential Features가 더 효과적")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading 5m data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df_5m = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df_5m)} 5m candles")

    # 2. 1시간봉 리샘플링
    logger.info("\n2. Resampling to 1h...")
    df_1h = resample_to_1h(df_5m)

    # 3. 지표 계산 (1시간봉)
    logger.info("\n3. Calculating indicators (1h)...")
    indicators = TechnicalIndicators()
    df_1h_processed = indicators.calculate_all_indicators(df_1h)
    df_1h_sequential = indicators.calculate_sequential_features(df_1h_processed)

    logger.info(f"1h candles with indicators: {len(df_1h_sequential)}")

    # 4. 타겟 생성 (1시간봉: lookahead=48h)
    logger.info("\n4. Creating targets (lookahead=48h)...")
    lookahead = 48  # 48시간 후
    df_1h_sequential['target'] = df_1h_sequential['close'].pct_change(lookahead).shift(-lookahead)

    # Feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_1h_sequential.columns if col not in exclude_cols]

    df_1h_sequential = df_1h_sequential.dropna()

    logger.info(f"Final 1h dataset: {len(df_1h_sequential)} candles")
    logger.info(f"Features: {len(feature_cols)}")

    # 5. Train/Test Split
    logger.info("\n5. Splitting data...")
    n = len(df_1h_sequential)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df_1h_sequential.iloc[:train_end].copy()
    val_df = df_1h_sequential.iloc[train_end:val_end].copy()
    test_df = df_1h_sequential.iloc[val_end:].copy()

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Test period: {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")

    # 6. 모델 훈련
    logger.info("\n6. Training XGBoost (1h)...")

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

    # 7. 예측 및 평가
    logger.info("\n7. Evaluating predictions...")

    test_preds = model.predict(dtest)
    test_r2 = r2_score(y_test, test_preds)

    logger.info(f"\n1h Model Performance:")
    logger.info(f"  R²: {test_r2:.4f}")
    logger.info(f"  Prediction Std: {test_preds.std()*100:.4f}%")
    logger.info(f"  Prediction Range: {test_preds.min()*100:.3f}% to {test_preds.max()*100:.3f}%")

    # 8. SL/TP 백테스팅
    logger.info("\n8. Backtesting with SL/TP (1:3)...")

    result_1h = backtest_with_sl_tp_1h(
        test_df,
        test_preds,
        entry_threshold=0.003,
        stop_loss_pct=0.010,  # -1%
        take_profit_pct=0.030,  # +3%
        initial_balance=10000.0,
        position_size=0.03,
        leverage=3
    )

    logger.info(f"\n1h Results:")
    logger.info(f"  Return: {result_1h['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {result_1h['num_trades']}")
    logger.info(f"  Win Rate: {result_1h['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {result_1h['profit_factor']:.2f}")
    logger.info(f"  Avg Win: ${result_1h['avg_win']:.2f}")
    logger.info(f"  Avg Loss: ${result_1h['avg_loss']:.2f}")

    # 9. Buy & Hold (1h)
    first_price = test_df.iloc[0]['close']
    last_price = test_df.iloc[-1]['close']
    bh_return_1h = (last_price - first_price) / first_price * 100

    logger.info(f"\nBuy & Hold (1h): {bh_return_1h:+.2f}%")

    # 10. 비교 분석
    logger.info("\n" + "="*80)
    logger.info("5분봉 vs 1시간봉 비교")
    logger.info("="*80)

    # 5분봉 결과 (이전 실험)
    result_5m = {
        'return': 6.00,
        'trades': 6,
        'win_rate': 50.0,
        'profit_factor': 2.83,
        'r2': -0.41,
        'pred_std': 0.2895,
        'bh_return': 14.19
    }

    logger.info("\n| Metric | 5분봉 | 1시간봉 | 개선 |")
    logger.info("|--------|-------|---------|------|")
    logger.info(f"| R² | {result_5m['r2']:.2f} | {test_r2:.2f} | {test_r2 - result_5m['r2']:+.2f} |")
    logger.info(f"| Pred Std | {result_5m['pred_std']:.4f}% | {test_preds.std()*100:.4f}% | {(test_preds.std()*100 - result_5m['pred_std']):.4f}% |")
    logger.info(f"| Return | {result_5m['return']:+.2f}% | {result_1h['total_return_pct']:+.2f}% | {result_1h['total_return_pct'] - result_5m['return']:+.2f}% |")
    logger.info(f"| Trades | {result_5m['trades']:3d} | {result_1h['num_trades']:3d} | {result_1h['num_trades'] - result_5m['trades']:+3d} |")
    logger.info(f"| Win Rate | {result_5m['win_rate']:.1f}% | {result_1h['win_rate']*100:.1f}% | {result_1h['win_rate']*100 - result_5m['win_rate']:+.1f}% |")
    logger.info(f"| PF | {result_5m['profit_factor']:.2f} | {result_1h['profit_factor']:.2f} | {result_1h['profit_factor'] - result_5m['profit_factor']:+.2f} |")
    logger.info(f"| Buy&Hold | {result_5m['bh_return']:+.2f}% | {bh_return_1h:+.2f}% | {bh_return_1h - result_5m['bh_return']:+.2f}% |")

    # 11. 최종 결론
    logger.info("\n" + "="*80)
    logger.info("CRITICAL ANALYSIS")
    logger.info("="*80)

    logger.info("\n가설 검증:")

    logger.info("\nH1: 1시간봉에서 R² 개선?")
    if test_r2 > result_5m['r2']:
        logger.success(f"✅ Yes! R²: {result_5m['r2']:.2f} → {test_r2:.2f} ({test_r2 - result_5m['r2']:+.2f})")
    else:
        logger.warning(f"❌ No. R²: {result_5m['r2']:.2f} → {test_r2:.2f} ({test_r2 - result_5m['r2']:+.2f})")

    logger.info("\nH2: Sequential Features가 더 효과적?")
    if test_preds.std() > result_5m['pred_std'] / 100:
        logger.success(f"✅ Yes! Pred Std: {result_5m['pred_std']:.4f}% → {test_preds.std()*100:.4f}%")
    else:
        logger.warning(f"❌ No.")

    logger.info("\nH3: 수익성 개선?")
    if result_1h['total_return_pct'] > result_5m['return']:
        logger.success(f"✅ Yes! Return: {result_5m['return']:+.2f}% → {result_1h['total_return_pct']:+.2f}%")
    else:
        logger.warning(f"❌ No. Return: {result_5m['return']:+.2f}% → {result_1h['total_return_pct']:+.2f}%")

    logger.info("\nH4: Buy & Hold 초과?")
    if result_1h['total_return_pct'] > bh_return_1h:
        logger.success(f"🎉 Yes! ML {result_1h['total_return_pct']:+.2f}% > B&H {bh_return_1h:+.2f}%")
    else:
        logger.warning(f"⚠️ No. Gap: {bh_return_1h - result_1h['total_return_pct']:.2f}%")

    logger.info("\n" + "="*80)
    logger.info("최종 권장사항")
    logger.info("="*80)

    if result_1h['total_return_pct'] > result_5m['return'] and result_1h['profit_factor'] > result_5m['profit_factor']:
        logger.success("\n✅ 1시간봉이 5분봉보다 우수!")
        logger.success(f"   권장: 1시간봉 사용")
        logger.info(f"\n설정:")
        logger.info(f"  Timeframe: 1h")
        logger.info(f"  Entry Threshold: ±0.3%")
        logger.info(f"  Stop-Loss: -1.0%")
        logger.info(f"  Take-Profit: +3.0%")
        logger.info(f"\n예상 성과:")
        logger.info(f"  Return: {result_1h['total_return_pct']:+.2f}%")
        logger.info(f"  Profit Factor: {result_1h['profit_factor']:.2f}")
    else:
        logger.warning("\n⚠️ 5분봉이 여전히 우수하거나 비슷")
        logger.info(f"\n5분봉 설정 유지 권장:")
        logger.info(f"  Return: {result_5m['return']:+.2f}%")
        logger.info(f"  Profit Factor: {result_5m['profit_factor']:.2f}")

    logger.info("\n" + "="*80)
    logger.info("✅ Timeframe Comparison Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
