"""
XGBoost Rolling Window Validation
==================================

Critical Question: XGBoost +8.12%가 특정 기간에만 작동하는가?

Test Strategy:
- Multiple train/test splits
- Different time periods
- Check if +8.12% is consistent or overfitting

If consistent → Deploy
If inconsistent → Overfitting, don't deploy
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
    logger.info("XGBOOST ROLLING WINDOW VALIDATION")
    logger.info("="*80)
    logger.info("\nCritical Question: +8.12%가 특정 기간에만 작동하는가?")

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

    logger.info(f"\nTotal candles: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Rolling Window Strategy
    # Total: 17,206 candles ≈ 60 days
    # Strategy: Train on expanding window, test on next period

    train_size = int(len(df) * 0.5)  # 8,603 candles (30 days)
    test_size = int(len(df) * 0.15)   # 2,580 candles (9 days)

    logger.info(f"\nRolling Window Setup:")
    logger.info(f"  Initial train: {train_size} candles")
    logger.info(f"  Test window: {test_size} candles")

    results = []

    # Window 1: Train 0-50%, Test 50-65%
    logger.info("\n" + "="*80)
    logger.info("WINDOW 1: Train [0-50%], Test [50-65%]")
    logger.info("="*80)

    train_end = train_size
    test_start = train_end
    test_end = test_start + test_size

    if test_end > len(df):
        test_end = len(df)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    logger.info(f"  Train: {len(train_df)} candles ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    logger.info(f"  Test: {len(test_df)} candles ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    X_train = train_df[feature_cols].values
    y_train = train_df['returns'].values
    X_test = test_df[feature_cols].values

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)

    result = backtest(test_df, preds, test_df['volatility'])
    bh = ((test_df.iloc[-1]['close'] - test_df.iloc[0]['close']) / test_df.iloc[0]['close']) * 100

    results.append({
        'window': 1,
        'train_period': f"{train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}",
        'test_period': f"{test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}",
        'return': result['total_return_pct'],
        'trades': result['num_trades'],
        'win_rate': result['win_rate'] * 100,
        'buy_hold': bh,
        'vs_bh': result['total_return_pct'] - bh
    })

    logger.info(f"  XGBoost: {result['total_return_pct']:+.2f}% | {result['num_trades']} trades | {result['win_rate']*100:.1f}% WR")
    logger.info(f"  Buy & Hold: {bh:+.2f}%")
    logger.info(f"  vs B&H: {result['total_return_pct'] - bh:+.2f}%")

    # Window 2: Train 0-65%, Test 65-80%
    logger.info("\n" + "="*80)
    logger.info("WINDOW 2: Train [0-65%], Test [65-80%]")
    logger.info("="*80)

    train_end = test_end
    test_start = train_end
    test_end = test_start + test_size

    if test_end > len(df):
        test_end = len(df)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    logger.info(f"  Train: {len(train_df)} candles ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    logger.info(f"  Test: {len(test_df)} candles ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    X_train = train_df[feature_cols].values
    y_train = train_df['returns'].values
    X_test = test_df[feature_cols].values

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)

    result = backtest(test_df, preds, test_df['volatility'])
    bh = ((test_df.iloc[-1]['close'] - test_df.iloc[0]['close']) / test_df.iloc[0]['close']) * 100

    results.append({
        'window': 2,
        'train_period': f"{train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}",
        'test_period': f"{test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}",
        'return': result['total_return_pct'],
        'trades': result['num_trades'],
        'win_rate': result['win_rate'] * 100,
        'buy_hold': bh,
        'vs_bh': result['total_return_pct'] - bh
    })

    logger.info(f"  XGBoost: {result['total_return_pct']:+.2f}% | {result['num_trades']} trades | {result['win_rate']*100:.1f}% WR")
    logger.info(f"  Buy & Hold: {bh:+.2f}%")
    logger.info(f"  vs B&H: {result['total_return_pct'] - bh:+.2f}%")

    # Window 3: Train 0-80%, Test 80-100%
    logger.info("\n" + "="*80)
    logger.info("WINDOW 3: Train [0-80%], Test [80-100%]")
    logger.info("="*80)

    train_end = test_end
    test_start = train_end
    test_end = len(df)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    logger.info(f"  Train: {len(train_df)} candles ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    logger.info(f"  Test: {len(test_df)} candles ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    X_train = train_df[feature_cols].values
    y_train = train_df['returns'].values
    X_test = test_df[feature_cols].values

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)

    result = backtest(test_df, preds, test_df['volatility'])
    bh = ((test_df.iloc[-1]['close'] - test_df.iloc[0]['close']) / test_df.iloc[0]['close']) * 100

    results.append({
        'window': 3,
        'train_period': f"{train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}",
        'test_period': f"{test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}",
        'return': result['total_return_pct'],
        'trades': result['num_trades'],
        'win_rate': result['win_rate'] * 100,
        'buy_hold': bh,
        'vs_bh': result['total_return_pct'] - bh
    })

    logger.info(f"  XGBoost: {result['total_return_pct']:+.2f}% | {result['num_trades']} trades | {result['win_rate']*100:.1f}% WR")
    logger.info(f"  Buy & Hold: {bh:+.2f}%")
    logger.info(f"  vs B&H: {result['total_return_pct'] - bh:+.2f}%")

    # Summary Statistics
    logger.info("\n" + "="*80)
    logger.info("ROBUSTNESS ANALYSIS")
    logger.info("="*80)

    df_results = pd.DataFrame(results)

    logger.info("\n| Window | Return | Trades | Win Rate | vs B&H |")
    logger.info("|--------|--------|--------|----------|--------|")
    for _, row in df_results.iterrows():
        logger.info(
            f"| {row['window']} | {row['return']:+6.2f}% | {row['trades']:6d} | "
            f"{row['win_rate']:6.1f}% | {row['vs_bh']:+6.2f}% |"
        )

    # Statistics
    mean_return = df_results['return'].mean()
    std_return = df_results['return'].std()
    min_return = df_results['return'].min()
    max_return = df_results['return'].max()

    mean_vs_bh = df_results['vs_bh'].mean()
    positive_periods = (df_results['vs_bh'] > 0).sum()

    logger.info(f"\n" + "="*80)
    logger.info("STATISTICS")
    logger.info("="*80)

    logger.info(f"\nReturn Statistics:")
    logger.info(f"  Mean: {mean_return:+.2f}%")
    logger.info(f"  Std: {std_return:.2f}%")
    logger.info(f"  Min: {min_return:+.2f}%")
    logger.info(f"  Max: {max_return:+.2f}%")
    logger.info(f"  Range: {max_return - min_return:.2f}%")

    logger.info(f"\nvs Buy & Hold:")
    logger.info(f"  Mean: {mean_vs_bh:+.2f}%")
    logger.info(f"  Positive periods: {positive_periods}/{len(df_results)} ({positive_periods/len(df_results)*100:.0f}%)")

    # Conclusion
    logger.info("\n" + "="*80)
    logger.info("CRITICAL CONCLUSION")
    logger.info("="*80)

    if positive_periods == len(df_results):
        logger.success(f"\n🎉 ROBUST: XGBoost beats Buy & Hold in ALL {len(df_results)} periods!")
        logger.success(f"Average outperformance: {mean_vs_bh:+.2f}%")
        logger.success("✅ NOT overfitting to single period")
        logger.success("✅ Ready for paper trading deployment")
    elif positive_periods >= len(df_results) * 0.66:
        logger.info(f"\n⚠️ PARTIALLY ROBUST: XGBoost beats Buy & Hold in {positive_periods}/{len(df_results)} periods")
        logger.info(f"Average: {mean_vs_bh:+.2f}%")
        logger.warning("⚠️ Inconsistent performance across periods")
        logger.info("Recommendation: More testing or cautious deployment")
    else:
        logger.error(f"\n❌ NOT ROBUST: XGBoost only beats Buy & Hold in {positive_periods}/{len(df_results)} periods")
        logger.error(f"Average: {mean_vs_bh:+.2f}%")
        logger.error("❌ Likely overfitting to original test period")
        logger.error("Recommendation: DO NOT deploy")

    logger.info("\n" + "="*80)

if __name__ == "__main__":
    main()
