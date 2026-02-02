"""
SHORT Model - LSTM Approach

비판적 통찰:
- 8가지 ML 시도 모두 실패 → 문제는 모델이 아닌 접근법
- XGBoost/LightGBM: 각 시점을 독립적으로 봄
- LSTM: Sequence를 봄 → temporal patterns 포착

가설:
- SHORT는 "가격 움직임의 sequence"에서 예측 가능
- 5분봉 10개 연속 (50분)의 패턴이 하락 예측 가능

목표: SHORT win rate 60%+ 달성
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.short_specific_features import ShortSpecificFeatures
from scripts.production.market_regime_filter import MarketRegimeFilter

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Check TensorFlow availability
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"✅ TensorFlow {tf.__version__} available")
    GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"GPU: {'✅ Available' if GPU_AVAILABLE else '❌ Not available (using CPU)'}")
except ImportError:
    print("❌ TensorFlow not installed")
    print("Install: pip install tensorflow")
    exit(1)


def create_short_labels(df, lookahead=3, threshold=0.002):
    """Create SHORT labels (same as V2)"""
    labels = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        min_future = future_prices.min()
        decrease_pct = (current_price - min_future) / current_price

        if decrease_pct >= threshold:
            labels.append(1)  # SHORT
        else:
            labels.append(0)  # NO SHORT

    return np.array(labels)


def load_and_prepare_data():
    """Load data and calculate features"""
    print("="*80)
    print("Loading Data for LSTM Model")
    print("="*80)

    # Load data
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Calculate features
    df = calculate_features(df)

    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    # Remove NaN
    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Select features (use most important from previous experiments)
    feature_columns = [
        # Price & Volume
        'close', 'high', 'low', 'open', 'volume',

        # Trend
        'ema_21', 'ema_50', 'macd', 'macd_signal',

        # Volatility
        'atr', 'bb_upper', 'bb_lower', 'volatility_ratio',

        # Momentum
        'rsi', 'adx',

        # SHORT-specific
        'bearish_div_rsi', 'bearish_div_macd',
        'shooting_star', 'bearish_engulfing',
        'bb_upper_rejection', 'overbought_reversal_signal',
        'selling_volume_dominance', 'momentum_exhaustion',

        # Regime
        'di_plus', 'di_minus'
    ]

    # Filter available columns
    feature_columns = [f for f in feature_columns if f in df.columns]

    print(f"\nSelected {len(feature_columns)} features for LSTM")

    # Create labels
    labels = create_short_labels(df, lookahead=3, threshold=0.002)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel Distribution:")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")

    return df[feature_columns].values, labels, feature_columns, df


def create_sequences(X, y, sequence_length=10):
    """
    Create sequences for LSTM

    sequence_length: 10 candles = 50 minutes of history
    """
    X_seq, y_seq = [], []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])  # Last 10 candles
        y_seq.append(y[i])  # Label for current candle

    return np.array(X_seq), np.array(y_seq)


def create_lstm_model(input_shape, class_weights):
    """
    Create LSTM model for SHORT prediction

    Architecture:
    - LSTM layers for temporal learning
    - Dropout for regularization
    - Dense layers for classification
    """
    model = keras.Sequential([
        # LSTM layers
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),

        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.3),

        layers.LSTM(16),
        layers.Dropout(0.3),

        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile with class weights
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model


def train_and_backtest_lstm(X, y, df):
    """Train and backtest LSTM model"""
    print("\n" + "="*80)
    print("Training LSTM Model with Time Series Cross-Validation")
    print("="*80)

    SEQUENCE_LENGTH = 10  # 50 minutes

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    print(f"\nCreating sequences (length={SEQUENCE_LENGTH})...")
    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length=SEQUENCE_LENGTH)
    print(f"Sequences shape: {X_seq.shape}")
    print(f"Labels shape: {y_seq.shape}")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5 (LSTM is slower)

    all_trades = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
        print(f"\n--- Fold {fold} ---")

        X_train, X_val = X_seq[train_idx], X_seq[val_idx]
        y_train, y_val = y_seq[train_idx], y_seq[val_idx]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Calculate class weights
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

        print(f"Class weights: {class_weights}")

        # Create model
        model = create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            class_weights=class_weights
        )

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train
        print("Training LSTM...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            class_weight=class_weights,
            callbacks=[early_stopping],
            verbose=0
        )

        print(f"Training completed: {len(history.history['loss'])} epochs")

        # Predict on validation
        y_prob = model.predict(X_val, verbose=0).flatten()

        # Apply filters
        threshold = 0.65

        # Adjust indices for sequences
        val_idx_adjusted = val_idx + SEQUENCE_LENGTH

        for i, idx in enumerate(val_idx_adjusted):
            if idx >= len(df):
                continue

            model_signal = y_prob[i] >= threshold
            regime_allowed = df['short_allowed'].iloc[idx] == 1

            if model_signal and regime_allowed:
                actual_label = y_val[i]

                trade = {
                    'fold': fold,
                    'index': idx,
                    'probability': y_prob[i],
                    'regime': df['regime_trend'].iloc[idx],
                    'predicted': 1,
                    'actual': actual_label,
                    'correct': (1 == actual_label)
                }
                all_trades.append(trade)

        # Fold metrics
        fold_trades = [t for t in all_trades if t['fold'] == fold]
        if len(fold_trades) > 0:
            fold_correct = sum(t['correct'] for t in fold_trades)
            fold_total = len(fold_trades)
            fold_win_rate = fold_correct / fold_total

            fold_metrics.append({
                'fold': fold,
                'trades': fold_total,
                'correct': fold_correct,
                'win_rate': fold_win_rate
            })

            print(f"Fold {fold}: {fold_total} trades, {fold_correct} correct ({fold_win_rate*100:.1f}%)")

    # Overall results
    print(f"\n{'='*80}")
    print("LSTM Model Results")
    print(f"{'='*80}")

    if len(all_trades) == 0:
        print("\n❌ No trades generated")
        return 0.0, all_trades, None, scaler

    total_trades = len(all_trades)
    correct_trades = sum(t['correct'] for t in all_trades)
    overall_win_rate = correct_trades / total_trades

    print(f"\nOverall Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Correct: {correct_trades}")
    print(f"  Incorrect: {total_trades - correct_trades}")
    print(f"  Win Rate: {overall_win_rate*100:.1f}%")

    # Regime analysis
    if len(all_trades) > 0:
        trades_df = pd.DataFrame(all_trades)
        print(f"\nTrades by Regime:")
        for regime in trades_df['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            regime_correct = regime_trades['correct'].sum()
            regime_total = len(regime_trades)
            regime_wr = regime_correct / regime_total * 100 if regime_total > 0 else 0
            print(f"  {regime}: {regime_total} trades, {regime_correct} correct ({regime_wr:.1f}%)")

    # Train final model on all data
    print("\n" + "="*80)
    print("Training Final LSTM Model on All Data")
    print("="*80)

    X_seq_all, y_seq_all = create_sequences(X_scaled, y, sequence_length=SEQUENCE_LENGTH)

    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_seq_all),
        y=y_seq_all
    )
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

    final_model = create_lstm_model(
        input_shape=(X_seq_all.shape[1], X_seq_all.shape[2]),
        class_weights=class_weights
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )

    final_model.fit(
        X_seq_all, y_seq_all,
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0
    )

    print("Final model training completed")

    return overall_win_rate, all_trades, final_model, scaler


def main():
    """Main LSTM training pipeline"""
    print("="*80)
    print("SHORT Model - LSTM Approach")
    print("="*80)
    print("New Approach:")
    print("  - LSTM: Learn temporal sequences (not individual candles)")
    print("  - Sequence: 10 candles (50 minutes) → predict SHORT")
    print("  - Hypothesis: SHORT patterns exist in price sequences")
    print("="*80)

    # Load data
    X, y, feature_columns, df = load_and_prepare_data()

    # Train and backtest
    win_rate, trades, model, scaler = train_and_backtest_lstm(X, y, df)

    # Decision
    print("\n" + "="*80)
    print("Final Decision - LSTM Model")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ LSTM captured temporal patterns!")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"short_lstm_{timestamp}"

        model.save(MODELS_DIR / f"{model_name}.keras")

        with open(MODELS_DIR / f"{model_name}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        print(f"\nModel saved: {MODELS_DIR / model_name}.keras")

    elif win_rate >= 0.40:
        print(f"🔄 IMPROVING: SHORT win rate {win_rate*100:.1f}% (40-60%)")
        print(f"🔄 LSTM showing promise! Better than XGBoost")

    elif win_rate >= 0.30:
        print(f"⚠️ MARGINAL: SHORT win rate {win_rate*100:.1f}% (30-40%)")
        print(f"⚠️ Similar to XGBoost, temporal learning not enough")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")
        print(f"❌ LSTM also cannot solve the fundamental problem")

    print(f"\nComplete Progress Summary:")
    print(f"  XGBoost (best): 26.0%")
    print(f"  LSTM: {win_rate*100:.1f}%")

    return win_rate, trades, model


if __name__ == "__main__":
    win_rate, trades, model = main()
