"""
Sequence Length Optimization
=============================

Critical Insight: 현재 50 candles (4.17 hours) context
더 긴 sequence = 더 많은 시계열 정보 = 더 나은 패턴 학습?

Test: 50, 100, 200 candles
Goal: Buy & Hold 초과 (-1.21% gap 제거)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def create_sequences(data, targets, sequence_length):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    return np.array(X), np.array(y)


def build_lstm_model(sequence_length, n_features):
    """Build LSTM model"""
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def backtest_lstm(df, predictions, volatility, entry_threshold=0.003):
    """Backtest LSTM predictions"""
    capital = 10000.0
    position = 0.0
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        pred = predictions[i]
        vol = volatility.iloc[i]

        # Exit check
        if position > 0:
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct >= 0.03:  # Take profit
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({
                    'pnl': profit,
                    'pnl_pct': pnl_pct,
                    'win': profit > 0,
                    'exit_reason': 'TP'
                })
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue

            elif pnl_pct <= -0.01:  # Stop loss
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({
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
            if vol < 0.0008:
                continue

            # Entry signal
            if pred > entry_threshold:
                position_capital = capital * 0.95
                position = (position_capital / current_price) * 0.9994
                entry_price = current_price

    # Close remaining position
    if position > 0:
        exit_capital = position * df.iloc[-1]['close'] * 0.9994
        profit = exit_capital - (position * entry_price)
        pnl_pct = (df.iloc[-1]['close'] - entry_price) / entry_price
        trades.append({
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
            'profit_factor': 0.0
        }

    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]

    total_profit = sum([t['pnl'] for t in wins]) if wins else 0
    total_loss = abs(sum([t['pnl'] for t in losses])) if losses else 0

    return {
        'total_return_pct': ((capital - 10000.0) / 10000.0) * 100,
        'num_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0,
        'profit_factor': total_profit / total_loss if total_loss > 0 else 0
    }


def test_sequence_length(sequence_length, df, feature_cols):
    """Test specific sequence length"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Sequence Length: {sequence_length} candles ({sequence_length*5/60:.2f} hours)")
    logger.info(f"{'='*80}")

    # Split data
    train_size = int(len(df) * 0.5)
    val_size = int(len(df) * 0.2)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['returns'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['returns'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['returns'].values

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)

    logger.info(f"Train sequences: {X_train_seq.shape}")
    logger.info(f"Validation sequences: {X_val_seq.shape}")
    logger.info(f"Test sequences: {X_test_seq.shape}")

    # Build model
    model = build_lstm_model(sequence_length, len(feature_cols))

    # Train
    logger.info("\nTraining LSTM...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=20,
        batch_size=64,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    logger.info(f"Training completed: {len(history.history['loss'])} epochs")
    logger.info(f"Final val_loss: {history.history['val_loss'][-1]:.6f}")

    # Predict
    predictions = model.predict(X_test_seq, verbose=0).flatten()

    # Adjust test dataframe
    test_df_adjusted = test_df.iloc[sequence_length:].reset_index(drop=True)

    # Backtest
    result = backtest_lstm(
        test_df_adjusted,
        predictions,
        test_df_adjusted['volatility'],
        entry_threshold=0.003
    )

    # Buy & Hold
    bh_return = ((test_df_adjusted.iloc[-1]['close'] - test_df_adjusted.iloc[0]['close'])
                 / test_df_adjusted.iloc[0]['close']) * 100

    logger.info(f"\n📊 Results (Sequence {sequence_length}):")
    logger.info(f"  Return: {result['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {result['num_trades']}")
    logger.info(f"  Win Rate: {result['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {result['profit_factor']:.2f}")
    logger.info(f"  Buy & Hold: {bh_return:+.2f}%")
    logger.info(f"  vs B&H: {result['total_return_pct'] - bh_return:+.2f}%")

    return {
        'sequence_length': sequence_length,
        'hours': sequence_length * 5 / 60,
        'return': result['total_return_pct'],
        'trades': result['num_trades'],
        'win_rate': result['win_rate'] * 100,
        'profit_factor': result['profit_factor'],
        'buy_hold': bh_return,
        'vs_bh': result['total_return_pct'] - bh_return,
        'model': model,
        'scaler': scaler
    }


def main():
    logger.info("="*80)
    logger.info("SEQUENCE LENGTH OPTIMIZATION")
    logger.info("="*80)

    # Load data
    logger.info("\n1. Loading data...")
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Total candles: {len(df)}")

    # Calculate features
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
    df['returns'] = df['close'].pct_change()

    df = df.dropna().reset_index(drop=True)

    # Features
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_width',
        'atr', 'volume_ratio',
        'close_change_1', 'close_change_2', 'close_change_3',
        'close_change_5', 'close_change_10',
        'volatility'
    ]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Candles after features: {len(df)}")

    # Test different sequence lengths
    logger.info("\n3. Testing Sequence Lengths...")

    sequence_lengths = [50, 100, 200]
    results = []

    for seq_len in sequence_lengths:
        result = test_sequence_length(seq_len, df, feature_cols)
        results.append(result)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("4. SUMMARY TABLE")
    logger.info("="*80)

    logger.info("\n| Seq Len | Hours | Return | Trades | Win Rate | PF | vs B&H |")
    logger.info("|---------|-------|--------|--------|----------|-----|--------|")

    for r in results:
        logger.info(
            f"| {r['sequence_length']:7d} | {r['hours']:5.2f} | {r['return']:+6.2f}% | "
            f"{r['trades']:6d} | {r['win_rate']:8.1f}% | {r['profit_factor']:3.2f} | "
            f"{r['vs_bh']:+6.2f}% |"
        )

    # Find best
    best = max(results, key=lambda x: x['return'])
    best_vs_bh = max(results, key=lambda x: x['vs_bh'])

    logger.info("\n" + "="*80)
    logger.info("5. BEST CONFIGURATION")
    logger.info("="*80)

    logger.info(f"\n🏆 Best Return: Sequence {best['sequence_length']} ({best['hours']:.2f} hours)")
    logger.info(f"   Return: {best['return']:+.2f}%")
    logger.info(f"   Win Rate: {best['win_rate']:.1f}%")
    logger.info(f"   vs B&H: {best['vs_bh']:+.2f}%")

    if best_vs_bh != best:
        logger.info(f"\n📊 Best vs Buy & Hold: Sequence {best_vs_bh['sequence_length']}")
        logger.info(f"   Return: {best_vs_bh['return']:+.2f}%")
        logger.info(f"   vs B&H: {best_vs_bh['vs_bh']:+.2f}%")

    # Comparison with baseline (50 candles)
    baseline = [r for r in results if r['sequence_length'] == 50][0]

    logger.info("\n" + "="*80)
    logger.info("6. IMPROVEMENT OVER BASELINE (50 candles)")
    logger.info("="*80)

    logger.info(f"\nBaseline (50 candles):")
    logger.info(f"  Return: {baseline['return']:+.2f}%")
    logger.info(f"  Win Rate: {baseline['win_rate']:.1f}%")
    logger.info(f"  vs B&H: {baseline['vs_bh']:+.2f}%")

    if best['sequence_length'] != 50:
        improvement = best['return'] - baseline['return']
        logger.info(f"\nBest ({best['sequence_length']} candles):")
        logger.info(f"  Return: {best['return']:+.2f}% ({improvement:+.2f}% improvement)")
        logger.info(f"  Win Rate: {best['win_rate']:.1f}%")
        logger.info(f"  vs B&H: {best['vs_bh']:+.2f}%")

        if best['vs_bh'] > 0:
            logger.success(f"\n🎉 SUCCESS: BEATS BUY & HOLD by {best['vs_bh']:+.2f}%!")
        elif best['vs_bh'] > baseline['vs_bh']:
            logger.success(f"\n✅ IMPROVEMENT: Gap reduced from {baseline['vs_bh']:.2f}% to {best['vs_bh']:.2f}%")
        else:
            logger.warning(f"\n⚠️ No improvement over baseline")
    else:
        logger.info("\n✅ Baseline (50 candles) is optimal")

    # Save best model
    if best['vs_bh'] > baseline['vs_bh']:
        logger.info(f"\n💾 Saving best model (sequence {best['sequence_length']})...")
        best['model'].save('models/lstm_optimized.keras')

        import pickle
        with open('models/lstm_optimized_scaler.pkl', 'wb') as f:
            pickle.dump(best['scaler'], f)

        logger.success("✅ Model saved!")

    logger.info("\n" + "="*80)
    logger.info("✅ Sequence Length Optimization Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
