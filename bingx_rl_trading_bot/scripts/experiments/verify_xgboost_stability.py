"""
XGBoost Stability Verification
===============================

Critical Question: XGBoost +8.12%가 안정적인가, 아니면 운인가?

Test: Multiple random seeds
- seed 0, 1, 2, ..., 9 (10 runs)
- 평균 성능 측정
- 표준편차 확인

If stable → XGBoost가 진짜 우수
If unstable → Lucky seed
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def backtest(df, predictions, volatility):
    """Backtest function"""
    capital = 10000.0
    position = 0.0
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        pred = predictions[i]
        vol = volatility.iloc[i]

        if position > 0:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct >= 0.03:
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({'pnl': profit, 'win': profit > 0})
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue
            elif pnl_pct <= -0.01:
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({'pnl': profit, 'win': profit > 0})
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue

        if position == 0:
            if vol < 0.0008:
                continue
            if pred > 0.003:
                position_capital = capital * 0.95
                position = (position_capital / current_price) * 0.9994
                entry_price = current_price

    if position > 0:
        exit_capital = position * df.iloc[-1]['close'] * 0.9994
        profit = exit_capital - (position * entry_price)
        trades.append({'pnl': profit, 'win': profit > 0})
        capital = capital - (position * entry_price) + exit_capital

    if len(trades) == 0:
        return {'total_return_pct': 0.0, 'num_trades': 0, 'win_rate': 0.0}

    wins = [t for t in trades if t['win']]
    return {
        'total_return_pct': ((capital - 10000.0) / 10000.0) * 100,
        'num_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0
    }


def main():
    logger.info("="*80)
    logger.info("XGBOOST STABILITY VERIFICATION")
    logger.info("="*80)

    # Load data
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Features
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

    # Split
    train_size = int(len(df) * 0.5)
    val_size = int(len(df) * 0.2)

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df['returns'].values
    X_test = test_df[feature_cols].values

    # Test multiple seeds
    logger.info("\nTesting XGBoost with 10 different random seeds...")

    results = []
    for seed in range(10):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=seed
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)

        result = backtest(test_df, preds, test_df['volatility'])
        results.append({
            'seed': seed,
            'return': result['total_return_pct'],
            'trades': result['num_trades'],
            'win_rate': result['win_rate'] * 100
        })

        logger.info(f"  Seed {seed}: {result['total_return_pct']:+.2f}% | {result['num_trades']} trades | {result['win_rate']*100:.1f}% WR")

    # Statistics
    returns = [r['return'] for r in results]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)

    logger.info("\n" + "="*80)
    logger.info("STABILITY ANALYSIS")
    logger.info("="*80)

    logger.info(f"\nReturn Statistics:")
    logger.info(f"  Mean: {mean_return:+.2f}%")
    logger.info(f"  Std Dev: {std_return:.2f}%")
    logger.info(f"  Min: {min_return:+.2f}%")
    logger.info(f"  Max: {max_return:+.2f}%")
    logger.info(f"  Range: {max_return - min_return:.2f}%")

    # Buy & Hold
    bh = ((test_df.iloc[-1]['close'] - test_df.iloc[0]['close']) / test_df.iloc[0]['close']) * 100
    logger.info(f"\nBuy & Hold: {bh:+.2f}%")
    logger.info(f"XGBoost avg vs B&H: {mean_return - bh:+.2f}%")

    # Conclusion
    logger.info("\n" + "="*80)
    logger.info("CONCLUSION")
    logger.info("="*80)

    if std_return < 2.0:
        logger.success(f"\n✅ STABLE: XGBoost is consistent (std: {std_return:.2f}%)")
        if mean_return > bh:
            logger.success(f"🎉 XGBoost beats Buy & Hold by average {mean_return - bh:.2f}%!")
        else:
            logger.info(f"XGBoost loses to Buy & Hold by average {bh - mean_return:.2f}%")
    else:
        logger.warning(f"\n⚠️ UNSTABLE: High variance (std: {std_return:.2f}%)")
        logger.info("XGBoost performance is highly dependent on random seed")

    logger.info("\n" + "="*80)

if __name__ == "__main__":
    main()
