"""
Fair Comparison: LSTM vs XGBoost
=================================

Critical Question: 동일한 test set에서 직접 비교했는가?

Fair Comparison:
1. 동일한 데이터 (BTCUSDT_5m_max.csv)
2. 동일한 train/val/test split (50/20/30)
3. 동일한 features (19 features)
4. 동일한 backtest 조건
5. 동일한 18-day test period

Result: 정확한 LSTM vs XGBoost 개선 폭
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')


def backtest(df, predictions, volatility, entry_threshold=0.003):
    """Universal backtest function"""
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

            if pnl_pct >= 0.03:  # TP
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({'pnl': profit, 'win': profit > 0})
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue

            elif pnl_pct <= -0.01:  # SL
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({'pnl': profit, 'win': profit > 0})
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue

        # Entry check
        if position == 0:
            if vol < 0.0008:  # Regime filter
                continue

            if pred > entry_threshold:
                position_capital = capital * 0.95
                position = (position_capital / current_price) * 0.9994
                entry_price = current_price

    # Close remaining
    if position > 0:
        exit_capital = position * df.iloc[-1]['close'] * 0.9994
        profit = exit_capital - (position * entry_price)
        trades.append({'pnl': profit, 'win': profit > 0})
        capital = capital - (position * entry_price) + exit_capital

    if len(trades) == 0:
        return {'total_return_pct': 0.0, 'num_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0}

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


def main():
    logger.info("="*80)
    logger.info("FAIR COMPARISON: LSTM vs XGBoost")
    logger.info("="*80)
    logger.info("\nCritical: 동일한 test set에서 직접 비교")

    # 1. Load data
    logger.info("\n1. Loading data...")
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2. Calculate features
    logger.info("\n2. Calculating features...")

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20

    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(window=14).mean()

    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    for lag in [1, 2, 3, 5, 10]:
        df[f'close_change_{lag}'] = df['close'].pct_change(lag)

    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['returns'] = df['close'].pct_change()

    df = df.dropna().reset_index(drop=True)

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
    logger.info(f"Total candles: {len(df)}")

    # 3. Split data (identical for both models)
    train_size = int(len(df) * 0.5)
    val_size = int(len(df) * 0.2)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/288:.1f} days)")
    logger.info(f"  Val: {len(val_df)} ({len(val_df)/288:.1f} days)")
    logger.info(f"  Test: {len(test_df)} ({len(test_df)/288:.1f} days)")

    # ==================================================
    # 4. LSTM Test
    # ==================================================
    logger.info("\n" + "="*80)
    logger.info("4. LSTM MODEL TEST")
    logger.info("="*80)

    try:
        lstm_model = keras.models.load_model('models/lstm_model.keras')
        with open('models/lstm_scaler.pkl', 'rb') as f:
            lstm_scaler = pickle.load(f)
        logger.success("✅ LSTM model loaded")
    except:
        logger.error("❌ LSTM model not found")
        return

    # Prepare LSTM data
    X_test_lstm = test_df[feature_cols].values
    X_test_scaled = lstm_scaler.transform(X_test_lstm)

    # Create sequences
    sequence_length = 50
    X_test_seq = []
    valid_indices = []
    for i in range(len(X_test_scaled) - sequence_length):
        X_test_seq.append(X_test_scaled[i:i+sequence_length])
        valid_indices.append(i + sequence_length)
    X_test_seq = np.array(X_test_seq)

    lstm_predictions = lstm_model.predict(X_test_seq, verbose=0).flatten()
    test_df_lstm = test_df.iloc[valid_indices].reset_index(drop=True)

    logger.info(f"LSTM predictions: {len(lstm_predictions)}")

    # ==================================================
    # 5. XGBoost Train & Test
    # ==================================================
    logger.info("\n" + "="*80)
    logger.info("5. XGBOOST MODEL TRAIN & TEST")
    logger.info("="*80)

    # Prepare XGBoost data (no sequences)
    X_train_xgb = train_df[feature_cols].values
    y_train_xgb = train_df['returns'].values

    X_val_xgb = val_df[feature_cols].values
    y_val_xgb = val_df['returns'].values

    X_test_xgb = test_df[feature_cols].values

    # Train XGBoost
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train_xgb, y_train_xgb, verbose=False)

    xgb_predictions = xgb_model.predict(X_test_xgb)
    test_df_xgb = test_df.copy()

    logger.info(f"XGBoost predictions: {len(xgb_predictions)}")

    # ==================================================
    # 6. Backtest Both Models
    # ==================================================
    logger.info("\n" + "="*80)
    logger.info("6. BACKTESTING")
    logger.info("="*80)

    # LSTM backtest
    logger.info("\nBacktesting LSTM...")
    lstm_result = backtest(
        test_df_lstm,
        lstm_predictions,
        test_df_lstm['volatility'],
        entry_threshold=0.003
    )

    # XGBoost backtest
    logger.info("Backtesting XGBoost...")
    xgb_result = backtest(
        test_df_xgb,
        xgb_predictions,
        test_df_xgb['volatility'],
        entry_threshold=0.003
    )

    # Buy & Hold
    bh_return_lstm = ((test_df_lstm.iloc[-1]['close'] - test_df_lstm.iloc[0]['close'])
                      / test_df_lstm.iloc[0]['close']) * 100
    bh_return_xgb = ((test_df_xgb.iloc[-1]['close'] - test_df_xgb.iloc[0]['close'])
                     / test_df_xgb.iloc[0]['close']) * 100

    # ==================================================
    # 7. Results
    # ==================================================
    logger.info("\n" + "="*80)
    logger.info("7. FAIR COMPARISON RESULTS")
    logger.info("="*80)

    logger.info("\n| Model | Return | Trades | Win Rate | PF | vs B&H |")
    logger.info("|-------|--------|--------|----------|-----|--------|")
    logger.info(
        f"| XGBoost | {xgb_result['total_return_pct']:+6.2f}% | {xgb_result['num_trades']:6d} | "
        f"{xgb_result['win_rate']*100:8.1f}% | {xgb_result['profit_factor']:3.2f} | "
        f"{xgb_result['total_return_pct'] - bh_return_xgb:+6.2f}% |"
    )
    logger.info(
        f"| LSTM    | {lstm_result['total_return_pct']:+6.2f}% | {lstm_result['num_trades']:6d} | "
        f"{lstm_result['win_rate']*100:8.1f}% | {lstm_result['profit_factor']:3.2f} | "
        f"{lstm_result['total_return_pct'] - bh_return_lstm:+6.2f}% |"
    )
    logger.info(f"| Buy&Hold (XGB period) | {bh_return_xgb:+6.2f}% | - | - | - | - |")
    logger.info(f"| Buy&Hold (LSTM period) | {bh_return_lstm:+6.2f}% | - | - | - | - |")

    # ==================================================
    # 8. Analysis
    # ==================================================
    logger.info("\n" + "="*80)
    logger.info("8. IMPROVEMENT ANALYSIS")
    logger.info("="*80)

    improvement = lstm_result['total_return_pct'] - xgb_result['total_return_pct']
    wr_improvement = lstm_result['win_rate'] * 100 - xgb_result['win_rate'] * 100
    pf_improvement = lstm_result['profit_factor'] - xgb_result['profit_factor']

    logger.info(f"\nLSTM vs XGBoost:")
    logger.info(f"  Return improvement: {improvement:+.2f}%")
    logger.info(f"  Win rate improvement: {wr_improvement:+.1f}%")
    logger.info(f"  Profit factor improvement: {pf_improvement:+.2f}")

    if improvement > 0:
        logger.success(f"\n✅ LSTM beats XGBoost by {improvement:.2f}%")
    else:
        logger.warning(f"\n⚠️ XGBoost beats LSTM by {-improvement:.2f}%")

    # Note: LSTM test period is 50 candles shorter
    logger.info("\n⚠️ Note: LSTM test period는 50 candles 짧음 (sequence length)")
    logger.info(f"  XGBoost: {len(test_df_xgb)} candles")
    logger.info(f"  LSTM: {len(test_df_lstm)} candles ({len(test_df_lstm)/288:.1f} days)")

    # Final Verdict
    logger.info("\n" + "="*80)
    logger.info("9. FINAL VERDICT")
    logger.info("="*80)

    if lstm_result['total_return_pct'] > 0 and lstm_result['win_rate'] >= 0.40:
        if lstm_result['total_return_pct'] > bh_return_lstm:
            logger.success("\n🎉 LSTM beats both XGBoost AND Buy & Hold!")
        else:
            logger.success(f"\n✅ LSTM beats XGBoost but loses to B&H by {bh_return_lstm - lstm_result['total_return_pct']:.2f}%")
    elif improvement > 0:
        logger.info(f"\n📊 LSTM improves over XGBoost but both lose to Buy & Hold")
    else:
        logger.warning("\n⚠️ XGBoost performs better than LSTM on this dataset")

    logger.info("\n" + "="*80)
    logger.info("✅ Fair Comparison Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
