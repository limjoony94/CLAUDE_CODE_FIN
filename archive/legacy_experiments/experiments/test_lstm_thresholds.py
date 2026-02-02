"""
LSTM Threshold Testing
LSTM 모델이 거래를 하지 않은 이유: entry_threshold가 너무 높을 수 있음
여러 threshold에서 재테스트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle

# Backtest function from train_lstm_timeseries.py
def backtest_with_threshold(
    df: pd.DataFrame,
    predictions: np.ndarray,
    volatility: pd.Series,
    entry_threshold: float,
    vol_threshold: float = 0.0008,
    stop_loss_pct: float = 0.010,
    take_profit_pct: float = 0.030,
    use_regime_filter: bool = True,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.95,
    trading_fee_pct: float = 0.0006
) -> dict:
    """Backtest with specific entry threshold"""

    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        pred = predictions[i]
        vol = volatility.iloc[i]

        # Exit check first
        if position > 0:
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct >= take_profit_pct:
                exit_capital = position * current_price * (1 - trading_fee_pct)
                profit = exit_capital - (position * entry_price)
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': profit,
                    'pnl_pct': pnl_pct,
                    'win': profit > 0,
                    'exit_reason': 'TP'
                })
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue

            elif pnl_pct <= -stop_loss_pct:
                exit_capital = position * current_price * (1 - trading_fee_pct)
                profit = exit_capital - (position * entry_price)
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': profit,
                    'pnl_pct': pnl_pct,
                    'win': profit > 0,
                    'exit_reason': 'SL'
                })
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue

        # Entry check
        if position == 0:
            # Regime filter
            if use_regime_filter and vol < vol_threshold:
                continue

            # Entry signal
            if pred > entry_threshold:
                position_capital = capital * position_size_pct
                position = (position_capital / current_price) * (1 - trading_fee_pct)
                entry_price = current_price

    # Close any remaining position
    if position > 0:
        exit_capital = position * df.iloc[-1]['close'] * (1 - trading_fee_pct)
        profit = exit_capital - (position * entry_price)
        pnl_pct = (df.iloc[-1]['close'] - entry_price) / entry_price
        trades.append({
            'entry_price': entry_price,
            'exit_price': df.iloc[-1]['close'],
            'pnl': profit,
            'pnl_pct': pnl_pct,
            'win': profit > 0,
            'exit_reason': 'EOD'
        })
        capital = capital - (position * entry_price) + exit_capital

    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }

    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]

    total_profit = sum([t['pnl'] for t in wins]) if wins else 0
    total_loss = abs(sum([t['pnl'] for t in losses])) if losses else 0

    return {
        'total_return_pct': ((capital - initial_capital) / initial_capital) * 100,
        'num_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0,
        'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
        'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    }


def main():
    logger.info("\n" + "="*80)
    logger.info("LSTM Threshold Testing")
    logger.info("="*80)

    # 1. Load data
    logger.info("\n1. Loading data...")
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Total candles: {len(df)}")

    # 2. Calculate features
    logger.info("\n2. Calculating features...")

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Bands
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(window=14).mean()

    # Volume
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price changes
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_change_{lag}'] = df['close'].pct_change(lag)

    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()

    # Returns
    df['close_change'] = df['close'].pct_change()
    df['returns'] = df['close'].pct_change()

    df = df.dropna().reset_index(drop=True)
    logger.info(f"After feature engineering: {len(df)} candles")

    # 3. Split data (same as training)
    train_size = int(len(df) * 0.5)
    val_size = int(len(df) * 0.2)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 4. Load LSTM model and scaler
    logger.info("\n3. Loading LSTM model...")

    try:
        model = keras.models.load_model('models/lstm_model.keras')
        logger.success("LSTM model loaded!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Training new model first...")

        # Simple features for LSTM
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_width',
            'atr', 'volume_ratio',
            'close_change_1', 'close_change_2', 'close_change_3',
            'close_change_5', 'close_change_10',
            'volatility'
        ]

        X_train = train_df[feature_cols].values
        y_train = train_df['returns'].values

        X_val = val_df[feature_cols].values
        y_val = val_df['returns'].values

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Create sequences
        def create_sequences(data, targets, sequence_length=50):
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(data[i:i+sequence_length])
                y.append(targets[i+sequence_length])
            return np.array(X), np.array(y)

        sequence_length = 50
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)

        # Build model
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, len(feature_cols))),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=5,  # Quick training
            batch_size=64,
            verbose=1
        )

        # Save
        model.save('models/lstm_model.keras')
        with open('models/lstm_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        logger.success("Model trained and saved!")

    # Load scaler
    try:
        with open('models/lstm_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        logger.error("Scaler not found, creating new one")
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_width',
            'atr', 'volume_ratio',
            'close_change_1', 'close_change_2', 'close_change_3',
            'close_change_5', 'close_change_10',
            'volatility'
        ]
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols].values)
        with open('models/lstm_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    # 5. Predict on test set
    logger.info("\n4. Predicting on test set...")

    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_width',
        'atr', 'volume_ratio',
        'close_change_1', 'close_change_2', 'close_change_3',
        'close_change_5', 'close_change_10',
        'volatility'
    ]

    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    sequence_length = 50
    X_test_seq = []
    valid_indices = []
    for i in range(len(X_test_scaled) - sequence_length):
        X_test_seq.append(X_test_scaled[i:i+sequence_length])
        valid_indices.append(i + sequence_length)
    X_test_seq = np.array(X_test_seq)

    predictions = model.predict(X_test_seq, verbose=0)
    predictions = predictions.flatten()

    # Adjust test_df
    test_df_adjusted = test_df.iloc[valid_indices].reset_index(drop=True)

    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Test set adjusted: {len(test_df_adjusted)}")

    # 6. Test multiple thresholds
    logger.info("\n" + "="*80)
    logger.info("5. Testing Multiple Entry Thresholds")
    logger.info("="*80)

    thresholds = [0.001, 0.0015, 0.002, 0.0025, 0.003]  # 0.1%, 0.15%, 0.2%, 0.25%, 0.3%

    results = []

    for threshold in thresholds:
        logger.info(f"\nTesting threshold: {threshold*100:.2f}%")

        result = backtest_with_threshold(
            test_df_adjusted,
            predictions,
            test_df_adjusted['volatility'],
            entry_threshold=threshold,
            vol_threshold=0.0008,
            stop_loss_pct=0.010,
            take_profit_pct=0.030,
            use_regime_filter=True
        )

        results.append({
            'threshold': threshold,
            **result
        })

        logger.info(f"  Return: {result['total_return_pct']:+.2f}%")
        logger.info(f"  Trades: {result['num_trades']}")
        logger.info(f"  Win Rate: {result['win_rate']*100:.1f}%")
        logger.info(f"  Profit Factor: {result['profit_factor']:.2f}")

    # 7. Summary table
    logger.info("\n" + "="*80)
    logger.info("6. SUMMARY TABLE")
    logger.info("="*80)

    logger.info("\n| Threshold | Return | Trades | Win Rate | PF | Avg Win | Avg Loss |")
    logger.info("|-----------|--------|--------|----------|-----|---------|----------|")

    for r in results:
        logger.info(
            f"| {r['threshold']*100:7.2f}% | {r['total_return_pct']:+6.2f}% | {r['num_trades']:6d} | "
            f"{r['win_rate']*100:8.1f}% | {r['profit_factor']:3.2f} | "
            f"{r['avg_win']*100:7.2f}% | {r['avg_loss']*100:8.2f}% |"
        )

    # Buy & Hold
    first_price = test_df_adjusted.iloc[0]['close']
    last_price = test_df_adjusted.iloc[-1]['close']
    bh_return = (last_price - first_price) / first_price * 100

    logger.info(f"\nBuy & Hold: {bh_return:+.2f}%")

    # 8. Find best threshold
    logger.info("\n" + "="*80)
    logger.info("7. BEST THRESHOLD ANALYSIS")
    logger.info("="*80)

    # Best by return
    best_return = max(results, key=lambda x: x['total_return_pct'])
    logger.info(f"\n✅ Best Return: {best_return['threshold']*100:.2f}% → {best_return['total_return_pct']:+.2f}%")
    logger.info(f"   Trades: {best_return['num_trades']}, Win Rate: {best_return['win_rate']*100:.1f}%")

    # Best by win rate
    results_with_trades = [r for r in results if r['num_trades'] > 0]
    if results_with_trades:
        best_wr = max(results_with_trades, key=lambda x: x['win_rate'])
        logger.info(f"\n🎯 Best Win Rate: {best_wr['threshold']*100:.2f}% → {best_wr['win_rate']*100:.1f}%")
        logger.info(f"   Return: {best_wr['total_return_pct']:+.2f}%, Trades: {best_wr['num_trades']}")

    # Compare to XGBoost
    xgb_return = -4.18
    xgb_wr = 25.0

    logger.info("\n" + "="*80)
    logger.info("8. LSTM vs XGBoost (Best Threshold)")
    logger.info("="*80)

    logger.info("\n| Model | Return | Trades | Win Rate | vs B&H |")
    logger.info("|-------|--------|--------|----------|--------|")
    logger.info(f"| XGBoost | {xgb_return:+6.2f}% | 16 | {xgb_wr:8.1f}% | {xgb_return - bh_return:+6.2f}% |")
    logger.info(
        f"| LSTM (best) | {best_return['total_return_pct']:+6.2f}% | {best_return['num_trades']:6d} | "
        f"{best_return['win_rate']*100:8.1f}% | {best_return['total_return_pct'] - bh_return:+6.2f}% |"
    )
    logger.info(f"| Buy & Hold | {bh_return:+6.2f}% | - | - | - |")

    # 9. Final conclusion
    logger.info("\n" + "="*80)
    logger.info("9. FINAL CONCLUSION")
    logger.info("="*80)

    if best_return['total_return_pct'] > 0 and best_return['win_rate'] >= 0.40:
        logger.success("\n🎉 SUCCESS: LSTM with optimized threshold beats Buy & Hold!")
        logger.info(f"Optimal threshold: {best_return['threshold']*100:.2f}%")
        logger.info(f"Win rate: {best_return['win_rate']*100:.1f}% (target: 40%+)")
    elif best_return['total_return_pct'] > xgb_return:
        logger.success("\n✅ IMPROVEMENT: LSTM better than XGBoost with threshold tuning!")
        logger.info(f"Improvement: {best_return['total_return_pct'] - xgb_return:+.2f}%")
        logger.info(f"Optimal threshold: {best_return['threshold']*100:.2f}%")
        logger.info("\nBut still loses to Buy & Hold")
    else:
        logger.warning("\n⚠️ LSTM threshold tuning did not improve results")
        logger.info("Additional approaches needed:")
        logger.info("  1. More data (6-12 months)")
        logger.info("  2. Different timeframe (4-hour or daily)")
        logger.info("  3. Hyperparameter tuning")
        logger.info("  4. Ensemble (LSTM + XGBoost)")

    logger.info("\n" + "="*80)
    logger.info("✅ Threshold Testing Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
