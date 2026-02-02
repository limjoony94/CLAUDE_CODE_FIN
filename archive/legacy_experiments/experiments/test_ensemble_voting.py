"""
Ensemble Voting Strategy: LSTM + XGBoost
=========================================

Critical Insight:
- LSTM: Time series patterns (long-term trends)
- XGBoost: Current state patterns (short-term signals)
- Ensemble: Trade only when BOTH models agree

Expected Outcome: Higher win rate through selectivity
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from src.indicators.technical_indicators import TechnicalIndicators

# Feature engineering
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators (same as LSTM training)"""
    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price momentum
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['price_change_20'] = df['close'].pct_change(20)

    return df


def create_sequences(data, targets, sequence_length=50):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    return np.array(X), np.array(y)


def prepare_xgboost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for XGBoost (same as original training)"""
    df = df.copy()

    # Sequential features (momentum, trends)
    for period in [5, 10, 20]:
        df[f'return_{period}'] = df['close'].pct_change(period)
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()

    return df


def backtest_ensemble_voting(
    df: pd.DataFrame,
    lstm_predictions: np.ndarray,
    xgb_predictions: np.ndarray,
    volatility: pd.Series,
    lstm_threshold: float = 0.003,
    xgb_threshold: float = 0.003,
    vol_threshold: float = 0.0008,
    stop_loss_pct: float = 0.010,
    take_profit_pct: float = 0.030,
    use_regime_filter: bool = True,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.95,
    trading_fee_pct: float = 0.0006
) -> dict:
    """
    Backtest voting ensemble strategy

    Strategy: Trade ONLY when both LSTM and XGBoost predict > threshold
    """
    capital = initial_capital
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        current_time = df.iloc[i]['timestamp']
        current_vol = volatility.iloc[i]

        lstm_pred = lstm_predictions[i]
        xgb_pred = xgb_predictions[i]

        # Close existing position if needed
        if position is not None:
            entry_price = position['entry_price']
            price_change = (current_price - entry_price) / entry_price

            should_close = False
            exit_reason = None

            # Stop loss / Take profit
            if price_change <= -stop_loss_pct:
                should_close = True
                exit_reason = 'stop_loss'
            elif price_change >= take_profit_pct:
                should_close = True
                exit_reason = 'take_profit'

            # Regime filter exit (volatility drops)
            if use_regime_filter and current_vol < vol_threshold * 0.8:
                should_close = True
                exit_reason = 'regime_exit'

            if should_close:
                exit_fee = position['quantity'] * current_price * trading_fee_pct
                capital = position['quantity'] * current_price - exit_fee

                pnl = capital - position['capital_before']
                pnl_pct = (capital - position['capital_before']) / position['capital_before']

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'lstm_pred': position['lstm_pred'],
                    'xgb_pred': position['xgb_pred']
                })

                position = None

        # Enter new position if both models agree
        if position is None:
            # Regime filter
            regime_ok = True
            if use_regime_filter:
                regime_ok = current_vol > vol_threshold

            # VOTING: Both models must agree
            both_bullish = (lstm_pred > lstm_threshold) and (xgb_pred > xgb_threshold)

            if regime_ok and both_bullish:
                position_size = capital * position_size_pct
                entry_fee = position_size * trading_fee_pct
                quantity = (position_size - entry_fee) / current_price

                position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'capital_before': capital,
                    'lstm_pred': lstm_pred,
                    'xgb_pred': xgb_pred
                }

    # Close any remaining position
    if position is not None:
        final_price = df.iloc[-1]['close']
        final_time = df.iloc[-1]['timestamp']
        exit_fee = position['quantity'] * final_price * trading_fee_pct
        capital = position['quantity'] * final_price - exit_fee

        pnl = capital - position['capital_before']
        pnl_pct = (capital - position['capital_before']) / position['capital_before']

        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': final_time,
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': 'end_of_data',
            'lstm_pred': position['lstm_pred'],
            'xgb_pred': position['xgb_pred']
        })

    # Calculate metrics
    total_return_pct = ((capital - initial_capital) / initial_capital) * 100

    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'final_capital': initial_capital,
            'trades': []
        }

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0

    total_profit = sum(t['pnl'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    avg_win = np.mean([t['pnl_pct'] for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) * 100 if losses else 0

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_capital': capital,
        'trades': trades
    }


def main():
    logger.info("🎯 Ensemble Voting Strategy: LSTM + XGBoost")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Add technical indicators (same as test_lstm_thresholds.py)
    logger.info("Adding technical indicators...")

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

    # Feature columns (same as test_lstm_thresholds.py)
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_width',
        'atr', 'volume_ratio',
        'close_change_1', 'close_change_2', 'close_change_3',
        'close_change_5', 'close_change_10',
        'volatility'
    ]

    logger.info(f"Total features: {len(feature_cols)}")

    # Split data
    train_size = int(len(df) * 0.5)
    val_size = int(len(df) * 0.2)

    train_df = df[:train_size].copy()
    val_df = df[train_size:train_size + val_size].copy()
    test_df = df[train_size + val_size:].copy()

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ===== LSTM Predictions =====
    logger.info("\n📊 Generating LSTM predictions...")

    # Load LSTM model
    try:
        lstm_model = keras.models.load_model('models/lstm_model.keras')
        logger.info("✅ Loaded existing LSTM model")
    except:
        logger.error("❌ LSTM model not found. Please train it first.")
        return

    # Prepare LSTM data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    sequence_length = 50
    X_test_lstm, y_test = create_sequences(
        test_scaled,
        test_df['returns'].values,
        sequence_length
    )

    lstm_predictions = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    logger.info(f"LSTM predictions: {len(lstm_predictions)}")

    # ===== XGBoost Predictions =====
    logger.info("\n📊 Generating XGBoost predictions...")

    try:
        xgb_model = joblib.load('models/best_xgboost_model.pkl')
        logger.info("✅ Loaded existing XGBoost model")
    except:
        logger.error("❌ XGBoost model not found. Please train it first.")
        return

    # Align XGBoost test data with LSTM (skip first 50 rows)
    test_df_aligned = test_df.iloc[sequence_length:].reset_index(drop=True)
    X_test_xgb = test_df_aligned[feature_cols]

    xgb_predictions = xgb_model.predict(X_test_xgb)
    logger.info(f"XGBoost predictions: {len(xgb_predictions)}")

    # Align dataframes
    test_df_final = test_df_aligned.copy()
    volatility_test = test_df_final['volatility'].reset_index(drop=True)

    logger.info(f"Final test size: {len(test_df_final)}")

    # ===== Backtest Ensemble =====
    logger.info("\n🚀 Running Ensemble Voting Backtest...")
    logger.info("=" * 60)

    results = backtest_ensemble_voting(
        df=test_df_final,
        lstm_predictions=lstm_predictions,
        xgb_predictions=xgb_predictions,
        volatility=volatility_test,
        lstm_threshold=0.003,  # 0.3%
        xgb_threshold=0.003,   # 0.3%
        vol_threshold=0.0008,
        stop_loss_pct=0.010,
        take_profit_pct=0.030,
        use_regime_filter=True
    )

    # Display results
    logger.info("\n📊 ENSEMBLE VOTING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
    logger.info(f"Number of Trades: {results['num_trades']}")
    logger.info(f"Win Rate: {results['win_rate']:.1f}%")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"Average Win: {results['avg_win']:.2f}%")
    logger.info(f"Average Loss: {results['avg_loss']:.2f}%")
    logger.info(f"Final Capital: ${results['final_capital']:.2f}")

    # Calculate Buy & Hold
    buy_hold_return = ((test_df_final.iloc[-1]['close'] - test_df_final.iloc[0]['close'])
                       / test_df_final.iloc[0]['close']) * 100
    logger.info(f"\n📈 Buy & Hold Return: {buy_hold_return:.2f}%")
    logger.info(f"📊 Ensemble vs Buy & Hold: {results['total_return_pct'] - buy_hold_return:.2f}%")

    # Show sample trades
    if results['trades']:
        logger.info("\n📝 Sample Trades (first 5):")
        for i, trade in enumerate(results['trades'][:5]):
            logger.info(f"  Trade {i+1}: {trade['pnl_pct']*100:.2f}% | "
                       f"LSTM: {trade['lstm_pred']*100:.2f}% | "
                       f"XGB: {trade['xgb_pred']*100:.2f}% | "
                       f"Exit: {trade['exit_reason']}")

    # Compare with individual models
    logger.info("\n🎯 Strategy Comparison")
    logger.info("=" * 60)
    logger.info("Ensemble (Voting):")
    logger.info(f"  Return: {results['total_return_pct']:.2f}%")
    logger.info(f"  Win Rate: {results['win_rate']:.1f}%")
    logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"  Trades: {results['num_trades']}")

    logger.info("\nExpected Individual Performance (from previous tests):")
    logger.info("  LSTM: +6.04% return, 50% win rate, 8 trades")
    logger.info("  XGBoost: -4.18% return, 25% win rate, 16 trades")
    logger.info(f"  Buy & Hold: {buy_hold_return:.2f}%")

    # Conclusion
    logger.info("\n" + "=" * 60)
    if results['total_return_pct'] > buy_hold_return:
        logger.success(f"🎉 SUCCESS: Ensemble beats Buy & Hold by {results['total_return_pct'] - buy_hold_return:.2f}%!")
    elif results['total_return_pct'] > 6.04:
        logger.info(f"✅ IMPROVEMENT: Ensemble beats LSTM by {results['total_return_pct'] - 6.04:.2f}%")
    else:
        logger.warning(f"⚠️ Ensemble did not improve over LSTM alone")

    logger.info(f"Win Rate Target (40%+): {'✅ ACHIEVED' if results['win_rate'] >= 40 else '❌ NOT MET'}")
    logger.info(f"Profit Factor Target (1.5+): {'✅ ACHIEVED' if results['profit_factor'] >= 1.5 else '❌ NOT MET'}")


if __name__ == '__main__':
    main()
