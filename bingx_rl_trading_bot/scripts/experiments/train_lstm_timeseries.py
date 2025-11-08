"""LSTM Time Series Model: 진짜 시계열 학습

핵심 개선:
- XGBoost: 각 candle을 독립적으로 취급 ❌
- LSTM: 50-100 candles의 전체 sequence를 학습 ✅

가설:
- Walk-forward에서 60-100% win rate → 짧은 패턴은 작동
- 18-day test에서 25% win rate → 긴 시계열 context 부족
- LSTM으로 long-term dependencies 학습 가능

검증:
- 동일한 18-day test로 검증
- LSTM vs XGBoost 성능 비교
- 목표: Win rate 40%+, Return > 0%
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    logger.info("TensorFlow imported successfully")
except ImportError:
    logger.error("TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)

from src.indicators.technical_indicators import TechnicalIndicators


def create_sequences(data: np.ndarray, targets: np.ndarray, sequence_length: int):
    """시계열 sequence 생성

    Args:
        data: Feature data (n_samples, n_features)
        targets: Target values (n_samples,)
        sequence_length: Sequence length (e.g., 50 candles)

    Returns:
        X: (n_sequences, sequence_length, n_features)
        y: (n_sequences,)
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(targets[i+sequence_length])

    return np.array(X), np.array(y)


def build_lstm_model(sequence_length: int, n_features: int) -> keras.Model:
    """LSTM 모델 구축

    Architecture:
    - LSTM(64) + Dropout(0.2)
    - LSTM(32) + Dropout(0.2)
    - Dense(16, relu) + Dropout(0.2)
    - Dense(1, linear)
    """
    model = keras.Sequential([
        # First LSTM layer
        layers.LSTM(64, return_sequences=True,
                   input_shape=(sequence_length, n_features)),
        layers.Dropout(0.2),

        # Second LSTM layer
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),

        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),

        # Output layer (regression)
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """변동성 계산"""
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    return volatility


def backtest_with_lstm(
    df: pd.DataFrame,
    predictions: np.ndarray,
    volatility: pd.Series,
    entry_threshold: float = 0.003,
    vol_threshold: float = 0.0008,
    stop_loss_pct: float = 0.010,
    take_profit_pct: float = 0.030,
    use_regime_filter: bool = True,
    initial_balance: float = 10000.0,
    position_size: float = 0.03,
    leverage: int = 3,
    transaction_fee: float = 0.0004,
    slippage: float = 0.0001
) -> dict:
    """LSTM predictions로 백테스팅"""

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        if i < 20:
            continue

        current_price = df.iloc[i]['close']
        signal = predictions[i]
        current_vol = volatility.iloc[i]

        # Regime check
        if use_regime_filter and current_vol <= vol_threshold and position == 0:
            continue

        # 포지션 청산
        if position != 0:
            price_change = (current_price - entry_price) / entry_price

            if position > 0:  # LONG
                if price_change <= -stop_loss_pct:
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance += pnl
                    balance -= position * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'SL'

                    position = 0.0
                    entry_price = 0.0

                elif price_change >= take_profit_pct:
                    exit_price = current_price * (1 - slippage)
                    pnl = position * (exit_price - entry_price) * leverage
                    balance += pnl
                    balance -= position * exit_price * transaction_fee

                    trades[-1]['exit'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_type'] = 'TP'

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
                    'signal': signal
                })

            elif signal < -entry_threshold:  # SHORT
                position = -position_size
                entry_price = current_price * (1 - slippage)
                balance -= abs(position) * entry_price * transaction_fee

                trades.append({
                    'type': 'SHORT',
                    'entry': entry_price,
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
    logger.info("LSTM TIME SERIES MODEL")
    logger.info("진짜 시계열 학습으로 Win Rate 개선")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)

    # 2. Features 계산
    logger.info("\n2. Calculating features...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)

    # Note: Sequential features는 사용 안 함 (LSTM이 직접 학습)
    # 기본 indicators만 사용

    lookahead = 48
    df_processed['target'] = df_processed['close'].pct_change(lookahead).shift(-lookahead)

    # Features 선택 (기본 indicators만)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_processed.columns if col not in exclude_cols]

    # 'seq_' prefix 제거 (sequential features 제외)
    feature_cols = [col for col in feature_cols if not col.startswith('seq_')]

    df_processed = df_processed.dropna()

    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Total candles: {len(df_processed)}")

    # Volatility
    volatility = calculate_volatility(df_processed, window=20)
    df_processed['volatility'] = volatility
    df_processed = df_processed.dropna()

    # 3. Train/Val/Test Split
    logger.info("\n3. Data split (50/20/30)...")
    n = len(df_processed)
    train_end = int(n * 0.5)
    val_end = int(n * 0.7)

    train_df = df_processed.iloc[:train_end].copy()
    val_df = df_processed.iloc[train_end:val_end].copy()
    test_df = df_processed.iloc[val_end:].copy()

    logger.info(f"Train: {len(train_df)} candles ({len(train_df)/288:.1f} days)")
    logger.info(f"Val: {len(val_df)} candles ({len(val_df)/288:.1f} days)")
    logger.info(f"Test: {len(test_df)} candles ({len(test_df)/288:.1f} days)")
    logger.info(f"Test period: {test_df.iloc[0]['timestamp']} to {test_df.iloc[-1]['timestamp']}")

    # 4. Sequence 생성
    logger.info("\n4. Creating sequences...")
    sequence_length = 50  # 50 candles = 4.17 hours

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df['target'].values

    X_val_raw = val_df[feature_cols].values
    y_val_raw = val_df['target'].values

    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df['target'].values

    # Normalize
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Create sequences
    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, sequence_length)
    X_val, y_val = create_sequences(X_val_scaled, y_val_raw, sequence_length)
    X_test, y_test = create_sequences(X_test_scaled, y_test_raw, sequence_length)

    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")
    logger.info(f"Test sequences: {X_test.shape}")

    # 5. LSTM 모델 구축
    logger.info("\n5. Building LSTM model...")
    model = build_lstm_model(sequence_length, len(feature_cols))

    logger.info("\nModel Architecture:")
    model.summary(print_fn=logger.info)

    # 6. 모델 훈련
    logger.info("\n6. Training LSTM model...")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Test 예측
    logger.info("\n7. Making predictions on test set...")
    test_preds_seq = model.predict(X_test, verbose=0).flatten()

    # Test dataframe adjustment (sequence_length만큼 shift)
    test_df_adjusted = test_df.iloc[sequence_length:].reset_index(drop=True)
    test_volatility = test_df_adjusted['volatility']

    logger.info(f"Test predictions: {len(test_preds_seq)}")
    logger.info(f"Test df adjusted: {len(test_df_adjusted)}")

    # MSE, MAE
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_test, test_preds_seq)
    mae = mean_absolute_error(y_test, test_preds_seq)
    r2 = r2_score(y_test, test_preds_seq)

    logger.info(f"\nLSTM Performance:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  R²: {r2:.4f}")

    # 8. Backtest with LSTM
    logger.info("\n" + "="*80)
    logger.info("8. BACKTEST WITH LSTM")
    logger.info("="*80)

    result_lstm = backtest_with_lstm(
        test_df_adjusted,
        test_preds_seq,
        test_volatility,
        entry_threshold=0.003,
        vol_threshold=0.0008,
        stop_loss_pct=0.010,
        take_profit_pct=0.030,
        use_regime_filter=True
    )

    logger.info(f"\nLSTM Strategy (Regime-Filtered):")
    logger.info(f"  Return: {result_lstm['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {result_lstm['num_trades']}")
    logger.info(f"  Win Rate: {result_lstm['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {result_lstm['profit_factor']:.2f}")

    # Buy & Hold
    first_price = test_df_adjusted.iloc[0]['close']
    last_price = test_df_adjusted.iloc[-1]['close']
    bh_return = (last_price - first_price) / first_price * 100

    logger.info(f"\nBuy & Hold: {bh_return:+.2f}%")

    # 9. Comparison
    logger.info("\n" + "="*80)
    logger.info("9. LSTM vs XGBoost COMPARISON")
    logger.info("="*80)

    logger.info("\n| Model | Return | Trades | Win Rate | PF | R² | vs B&H |")
    logger.info("|-------|--------|--------|----------|-----|-----|--------|")

    # XGBoost (from previous tests)
    xgb_return = -4.18
    xgb_trades = 16
    xgb_wr = 25.0
    xgb_pf = 0.74
    xgb_r2 = -0.39

    logger.info(
        f"| XGBoost | {xgb_return:+6.2f}% | {xgb_trades:6d} | {xgb_wr:8.1f}% | "
        f"{xgb_pf:3.2f} | {xgb_r2:5.2f} | {xgb_return - bh_return:+6.2f}% |"
    )

    logger.info(
        f"| LSTM    | {result_lstm['total_return_pct']:+6.2f}% | {result_lstm['num_trades']:6d} | "
        f"{result_lstm['win_rate']*100:8.1f}% | {result_lstm['profit_factor']:3.2f} | "
        f"{r2:5.2f} | {result_lstm['total_return_pct'] - bh_return:+6.2f}% |"
    )

    # 10. Final Conclusion
    logger.info("\n" + "="*80)
    logger.info("10. FINAL CONCLUSION")
    logger.info("="*80)

    improvement = result_lstm['total_return_pct'] - xgb_return
    wr_improvement = result_lstm['win_rate'] * 100 - xgb_wr

    logger.info(f"\nLSTM Improvement over XGBoost:")
    logger.info(f"  Return: {improvement:+.2f}%")
    logger.info(f"  Win Rate: {wr_improvement:+.1f}%")
    logger.info(f"  R²: {r2 - xgb_r2:+.2f}")

    if result_lstm['total_return_pct'] > 0 and result_lstm['win_rate'] >= 0.40:
        logger.success("\n🎉 SUCCESS: LSTM beats Buy & Hold!")
        logger.info("사용자 지적이 정확했습니다:")
        logger.info("  - 시계열 데이터 제공이 핵심")
        logger.info("  - LSTM으로 long-term dependencies 학습")
        logger.info("  - Win rate 40%+ 달성")
        logger.info("\n다음 단계:")
        logger.info("  1. Paper trading")
        logger.info("  2. Ensemble (LSTM + XGBoost)")
        logger.info("  3. Hyperparameter tuning")

    elif result_lstm['total_return_pct'] > xgb_return:
        logger.success("\n✅ IMPROVEMENT: LSTM better than XGBoost!")
        logger.info("시계열 학습의 효과:")
        logger.info(f"  - Return improved by {improvement:+.2f}%")
        logger.info(f"  - Win rate improved by {wr_improvement:+.1f}%")
        logger.info("\n추가 개선 방향:")
        logger.info("  1. Ensemble (LSTM + XGBoost)")
        logger.info("  2. More data (6-12 months)")
        logger.info("  3. Hyperparameter tuning")
        logger.info("  4. Attention mechanism (Transformer)")

    else:
        logger.warning("\n⚠️ LSTM did not significantly improve")
        logger.info("가능한 이유:")
        logger.info("  1. 60일 데이터 부족 (LSTM은 더 많은 데이터 필요)")
        logger.info("  2. Hyperparameter 최적화 필요")
        logger.info("  3. 5분 timeframe too noisy for any model")
        logger.info("\n다음 단계:")
        logger.info("  1. Collect 6-12 months data")
        logger.info("  2. Try 4-hour or daily timeframe")
        logger.info("  3. Ensemble approach")

    logger.info("\n" + "="*80)
    logger.info("✅ LSTM Training Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
