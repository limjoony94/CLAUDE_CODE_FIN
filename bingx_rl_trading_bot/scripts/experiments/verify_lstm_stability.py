"""
LSTM Stability Verification
============================

Critical Question: test_lstm_thresholds.py의 +6.04% 결과가 진짜인가?

Test:
1. 저장된 LSTM 모델 재로드
2. 동일한 테스트 데이터로 재예측
3. 결과 비교: 6.04% 재현되는가?

If 재현됨 → 모델은 stable, 재학습 문제
If 재현 안됨 → 이전 결과가 우연, LSTM 불안정
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

            if pnl_pct >= 0.03:  # TP
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({
                    'pnl': profit,
                    'pnl_pct': pnl_pct,
                    'win': profit > 0
                })
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue

            elif pnl_pct <= -0.01:  # SL
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({
                    'pnl': profit,
                    'pnl_pct': pnl_pct,
                    'win': profit > 0
                })
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
        pnl_pct = (df.iloc[-1]['close'] - entry_price) / entry_price
        trades.append({
            'pnl': profit,
            'pnl_pct': pnl_pct,
            'win': profit > 0
        })
        capital = capital - (position * entry_price) + exit_capital

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


def main():
    logger.info("="*80)
    logger.info("LSTM STABILITY VERIFICATION")
    logger.info("="*80)
    logger.info("\nCritical Question: +6.04% 결과가 재현 가능한가?")

    # Load data
    logger.info("\n1. Loading data...")
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate features (same as test_lstm_thresholds.py)
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

    # Split (same as test_lstm_thresholds.py)
    train_size = int(len(df) * 0.5)
    val_size = int(len(df) * 0.2)

    test_df = df.iloc[train_size+val_size:].copy()

    logger.info(f"Test set: {len(test_df)} candles")

    # Load saved model
    logger.info("\n3. Loading saved LSTM model...")

    try:
        model = keras.models.load_model('models/lstm_model.keras')
        logger.success("✅ LSTM model loaded!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    try:
        with open('models/lstm_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.success("✅ Scaler loaded!")
    except Exception as e:
        logger.error(f"❌ Failed to load scaler: {e}")
        return

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

    # Predict
    logger.info("\n4. Generating predictions...")

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

    predictions = model.predict(X_test_seq, verbose=0).flatten()

    logger.info(f"Predictions generated: {len(predictions)}")

    # Adjust test dataframe
    test_df_adjusted = test_df.iloc[valid_indices].reset_index(drop=True)

    # Backtest
    logger.info("\n5. Running backtest...")

    result = backtest_lstm(
        test_df_adjusted,
        predictions,
        test_df_adjusted['volatility'],
        entry_threshold=0.003
    )

    # Buy & Hold
    bh_return = ((test_df_adjusted.iloc[-1]['close'] - test_df_adjusted.iloc[0]['close'])
                 / test_df_adjusted.iloc[0]['close']) * 100

    # Results
    logger.info("\n" + "="*80)
    logger.info("6. RESULTS")
    logger.info("="*80)

    logger.info("\n📊 Saved Model Performance:")
    logger.info(f"  Return: {result['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {result['num_trades']}")
    logger.info(f"  Win Rate: {result['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {result['profit_factor']:.2f}")

    logger.info(f"\n📈 Buy & Hold: {bh_return:+.2f}%")
    logger.info(f"📊 vs B&H: {result['total_return_pct'] - bh_return:+.2f}%")

    # Comparison with expected
    logger.info("\n" + "="*80)
    logger.info("7. COMPARISON WITH EXPECTED")
    logger.info("="*80)

    expected_return = 6.04
    expected_trades = 8
    expected_wr = 50.0

    logger.info("\n| Metric | Expected | Actual | Match? |")
    logger.info("|--------|----------|--------|--------|")
    logger.info(
        f"| Return | {expected_return:+6.2f}% | {result['total_return_pct']:+6.2f}% | "
        f"{'✅' if abs(result['total_return_pct'] - expected_return) < 0.5 else '❌'} |"
    )
    logger.info(
        f"| Trades | {expected_trades:6d} | {result['num_trades']:6d} | "
        f"{'✅' if result['num_trades'] == expected_trades else '❌'} |"
    )
    logger.info(
        f"| Win Rate | {expected_wr:6.1f}% | {result['win_rate']*100:6.1f}% | "
        f"{'✅' if abs(result['win_rate']*100 - expected_wr) < 5 else '❌'} |"
    )

    # Conclusion
    logger.info("\n" + "="*80)
    logger.info("8. CONCLUSION")
    logger.info("="*80)

    matches = 0
    if abs(result['total_return_pct'] - expected_return) < 0.5:
        matches += 1
    if result['num_trades'] == expected_trades:
        matches += 1
    if abs(result['win_rate']*100 - expected_wr) < 5:
        matches += 1

    if matches == 3:
        logger.success("\n🎉 STABLE: 저장된 모델이 동일한 결과를 재현합니다!")
        logger.info("✅ +6.04% 결과는 진짜입니다")
        logger.info("✅ LSTM 모델은 stable합니다")
        logger.info("\n문제: 재학습 시 random seed 미설정")
        logger.info("해결책: Random seed를 고정하여 재학습")
    elif matches >= 1:
        logger.warning("\n⚠️ PARTIALLY STABLE: 일부 metric만 재현")
        logger.info("저장된 모델은 작동하지만 완전히 동일하지는 않음")
        logger.info("가능한 원인: 데이터 preprocessing 차이, regime filter timing")
    else:
        logger.error("\n❌ UNSTABLE: 저장된 모델이 다른 결과를 생성")
        logger.info("이전 +6.04% 결과가 우연이었을 가능성")
        logger.info("LSTM 모델의 본질적 불안정성")

    logger.info("\n" + "="*80)
    logger.info("✅ Verification Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
