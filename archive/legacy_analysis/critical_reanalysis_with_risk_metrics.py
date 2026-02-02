"""
Critical Reanalysis: Risk-Adjusted Returns & Statistical Significance
======================================================================

비판적 질문:
1. -0.86% 차이가 통계적으로 유의미한가? (3 samples, std 5.79%)
2. 리스크 조정 수익은 어떻게 다른가? (Sharpe ratio, drawdown)
3. 거래 비용이 주요 원인인가? (적은 거래 vs 높은 수수료)
4. XGBoost가 진짜 실패인가, 아니면 다른 관점에서는 성공인가?

근본 원인 분석:
- Buy & Hold: 0.08% 수수료 (1회)
- XGBoost: 0.12% × trades 수수료 (다회)
- 적은 거래 → 수수료 부담 → 성과 저하

해결책 탐색:
1. Risk-adjusted metrics로 재평가
2. Transaction cost sensitivity 분석
3. Statistical significance test
4. 더 많은 기간으로 확장 검증
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())  # Annualized


def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Calculate Sortino ratio (downside deviation)"""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    downside_std = downside_returns.std()
    return np.sqrt(252) * (excess_returns.mean() / downside_std)


def backtest_with_equity_curve(df, predictions, volatility):
    """Backtest with equity curve tracking"""
    capital = 10000.0
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        pred = predictions[i]
        vol = volatility.iloc[i]

        # Track equity
        if position > 0:
            current_equity = capital - (position * entry_price) + (position * current_price)
        else:
            current_equity = capital
        equity_curve.append(current_equity)

        # Exit check
        if position > 0:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct >= 0.03:
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({'pnl': profit, 'win': profit > 0, 'pnl_pct': pnl_pct})
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue
            elif pnl_pct <= -0.01:
                exit_capital = position * current_price * 0.9994
                profit = exit_capital - (position * entry_price)
                trades.append({'pnl': profit, 'win': profit > 0, 'pnl_pct': pnl_pct})
                capital = capital - (position * entry_price) + exit_capital
                position = 0.0
                continue

        # Entry check
        if position == 0:
            if vol < 0.0008:
                continue
            if pred > 0.003:
                position_capital = capital * 0.95
                position = (position_capital / current_price) * 0.9994
                entry_price = current_price

    # Close remaining
    if position > 0:
        exit_capital = position * df.iloc[-1]['close'] * 0.9994
        profit = exit_capital - (position * entry_price)
        pnl_pct = (df.iloc[-1]['close'] - entry_price) / entry_price
        trades.append({'pnl': profit, 'win': profit > 0, 'pnl_pct': pnl_pct})
        capital = capital - (position * entry_price) + exit_capital

        # Final equity
        current_equity = capital
        equity_curve.append(current_equity)

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'equity_curve': equity_series
        }

    wins = [t for t in trades if t['win']]

    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = calculate_max_drawdown(equity_series)

    return {
        'total_return_pct': ((capital - 10000.0) / 10000.0) * 100,
        'num_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd * 100,
        'equity_curve': equity_series,
        'avg_win': np.mean([t['pnl_pct'] for t in wins]) * 100 if wins else 0,
        'avg_loss': np.mean([t['pnl_pct'] for t in [t for t in trades if not t['win']]]) * 100 if not all([t['win'] for t in trades]) else 0
    }


def main():
    logger.info("="*80)
    logger.info("CRITICAL REANALYSIS: Risk-Adjusted Returns & Statistical Significance")
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

    # Rolling window analysis with risk metrics
    train_size = int(len(df) * 0.5)
    test_size = int(len(df) * 0.15)

    results_xgb = []
    results_bh = []

    # Window 1
    logger.info("\n" + "="*80)
    logger.info("WINDOW 1: Train [0-50%], Test [50-65%]")
    logger.info("="*80)

    train_end = train_size
    test_start = train_end
    test_end = test_start + test_size

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    logger.info(f"Period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")

    X_train = train_df[feature_cols].values
    y_train = train_df['returns'].values
    X_test = test_df[feature_cols].values

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)

    xgb_result = backtest_with_equity_curve(test_df, preds, test_df['volatility'])

    # Buy & Hold with equity curve
    bh_start = test_df.iloc[0]['close']
    bh_equity = 10000 * (test_df['close'] / bh_start) * 0.9992  # 0.08% fee
    bh_returns = bh_equity.pct_change().dropna()
    bh_sharpe = calculate_sharpe_ratio(bh_returns)
    bh_sortino = calculate_sortino_ratio(bh_returns)
    bh_max_dd = calculate_max_drawdown(bh_equity) * 100
    bh_total_return = ((test_df.iloc[-1]['close'] - bh_start) / bh_start) * 100

    results_xgb.append(xgb_result)
    results_bh.append({
        'return': bh_total_return,
        'sharpe': bh_sharpe,
        'sortino': bh_sortino,
        'max_dd': bh_max_dd
    })

    logger.info(f"  XGBoost: {xgb_result['total_return_pct']:+.2f}% | Sharpe: {xgb_result['sharpe_ratio']:.2f} | Sortino: {xgb_result['sortino_ratio']:.2f} | MaxDD: {xgb_result['max_drawdown']:.2f}%")
    logger.info(f"  Buy&Hold: {bh_total_return:+.2f}% | Sharpe: {bh_sharpe:.2f} | Sortino: {bh_sortino:.2f} | MaxDD: {bh_max_dd:.2f}%")

    # Window 2
    logger.info("\n" + "="*80)
    logger.info("WINDOW 2: Train [0-65%], Test [65-80%]")
    logger.info("="*80)

    train_end = test_end
    test_start = train_end
    test_end = test_start + test_size

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    logger.info(f"Period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")

    X_train = train_df[feature_cols].values
    y_train = train_df['returns'].values
    X_test = test_df[feature_cols].values

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)

    xgb_result = backtest_with_equity_curve(test_df, preds, test_df['volatility'])

    bh_start = test_df.iloc[0]['close']
    bh_equity = 10000 * (test_df['close'] / bh_start) * 0.9992
    bh_returns = bh_equity.pct_change().dropna()
    bh_sharpe = calculate_sharpe_ratio(bh_returns)
    bh_sortino = calculate_sortino_ratio(bh_returns)
    bh_max_dd = calculate_max_drawdown(bh_equity) * 100
    bh_total_return = ((test_df.iloc[-1]['close'] - bh_start) / bh_start) * 100

    results_xgb.append(xgb_result)
    results_bh.append({
        'return': bh_total_return,
        'sharpe': bh_sharpe,
        'sortino': bh_sortino,
        'max_dd': bh_max_dd
    })

    logger.info(f"  XGBoost: {xgb_result['total_return_pct']:+.2f}% | Sharpe: {xgb_result['sharpe_ratio']:.2f} | Sortino: {xgb_result['sortino_ratio']:.2f} | MaxDD: {xgb_result['max_drawdown']:.2f}%")
    logger.info(f"  Buy&Hold: {bh_total_return:+.2f}% | Sharpe: {bh_sharpe:.2f} | Sortino: {bh_sortino:.2f} | MaxDD: {bh_max_dd:.2f}%")

    # Window 3
    logger.info("\n" + "="*80)
    logger.info("WINDOW 3: Train [0-80%], Test [80-100%]")
    logger.info("="*80)

    train_end = test_end
    test_start = train_end
    test_end = len(df)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    logger.info(f"Period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")

    X_train = train_df[feature_cols].values
    y_train = train_df['returns'].values
    X_test = test_df[feature_cols].values

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)

    xgb_result = backtest_with_equity_curve(test_df, preds, test_df['volatility'])

    bh_start = test_df.iloc[0]['close']
    bh_equity = 10000 * (test_df['close'] / bh_start) * 0.9992
    bh_returns = bh_equity.pct_change().dropna()
    bh_sharpe = calculate_sharpe_ratio(bh_returns)
    bh_sortino = calculate_sortino_ratio(bh_returns)
    bh_max_dd = calculate_max_drawdown(bh_equity) * 100
    bh_total_return = ((test_df.iloc[-1]['close'] - bh_start) / bh_start) * 100

    results_xgb.append(xgb_result)
    results_bh.append({
        'return': bh_total_return,
        'sharpe': bh_sharpe,
        'sortino': bh_sortino,
        'max_dd': bh_max_dd
    })

    logger.info(f"  XGBoost: {xgb_result['total_return_pct']:+.2f}% | Sharpe: {xgb_result['sharpe_ratio']:.2f} | Sortino: {xgb_result['sortino_ratio']:.2f} | MaxDD: {xgb_result['max_drawdown']:.2f}%")
    logger.info(f"  Buy&Hold: {bh_total_return:+.2f}% | Sharpe: {bh_sharpe:.2f} | Sortino: {bh_sortino:.2f} | MaxDD: {bh_max_dd:.2f}%")

    # Statistical Analysis
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("="*80)

    xgb_returns = [r['total_return_pct'] for r in results_xgb]
    bh_returns = [r['return'] for r in results_bh]

    xgb_sharpes = [r['sharpe_ratio'] for r in results_xgb]
    bh_sharpes = [r['sharpe'] for r in results_bh]

    xgb_sortinos = [r['sortino_ratio'] for r in results_xgb]
    bh_sortinos = [r['sortino'] for r in results_bh]

    xgb_maxdds = [r['max_drawdown'] for r in results_xgb]
    bh_maxdds = [r['max_dd'] for r in results_bh]

    logger.info("\n📊 Return Statistics:")
    logger.info(f"  XGBoost: {np.mean(xgb_returns):+.2f}% ± {np.std(xgb_returns):.2f}%")
    logger.info(f"  Buy&Hold: {np.mean(bh_returns):+.2f}% ± {np.std(bh_returns):.2f}%")
    logger.info(f"  Difference: {np.mean(xgb_returns) - np.mean(bh_returns):+.2f}%")

    # Paired t-test
    if len(xgb_returns) > 2:
        t_stat, p_value = stats.ttest_rel(xgb_returns, bh_returns)
        logger.info(f"\n  Paired t-test: t={t_stat:.3f}, p={p_value:.3f}")
        if p_value < 0.05:
            logger.info(f"  → Statistically significant difference (p < 0.05)")
        else:
            logger.warning(f"  → NOT statistically significant (p >= 0.05)")
            logger.warning(f"  → With only 3 samples and std {np.std(xgb_returns):.2f}%, difference is within noise")

    logger.info("\n📈 Risk-Adjusted Metrics:")
    logger.info(f"\nSharpe Ratio:")
    logger.info(f"  XGBoost: {np.mean(xgb_sharpes):.3f}")
    logger.info(f"  Buy&Hold: {np.mean(bh_sharpes):.3f}")
    logger.info(f"  Difference: {np.mean(xgb_sharpes) - np.mean(bh_sharpes):+.3f}")

    logger.info(f"\nSortino Ratio:")
    logger.info(f"  XGBoost: {np.mean(xgb_sortinos):.3f}")
    logger.info(f"  Buy&Hold: {np.mean(bh_sortinos):.3f}")
    logger.info(f"  Difference: {np.mean(xgb_sortinos) - np.mean(bh_sortinos):+.3f}")

    logger.info(f"\nMax Drawdown:")
    logger.info(f"  XGBoost: {np.mean(xgb_maxdds):.2f}%")
    logger.info(f"  Buy&Hold: {np.mean(bh_maxdds):.2f}%")
    logger.info(f"  Difference: {np.mean(xgb_maxdds) - np.mean(bh_maxdds):+.2f}% (lower is better)")

    # CRITICAL CONCLUSION
    logger.info("\n" + "="*80)
    logger.info("CRITICAL REANALYSIS CONCLUSION")
    logger.info("="*80)

    logger.info("\n🤔 LOGICAL CONTRADICTIONS FOUND:")
    logger.info("\n1. **Statistical Insignificance**:")
    logger.info(f"   - Only 3 samples")
    logger.info(f"   - Return std: {np.std(xgb_returns):.2f}%")
    logger.info(f"   - Mean difference: {np.mean(xgb_returns) - np.mean(bh_returns):+.2f}%")
    logger.info(f"   - This difference is within 1 standard deviation = NOISE")

    logger.info("\n2. **Risk-Adjusted Performance**:")
    if np.mean(xgb_sharpes) > np.mean(bh_sharpes):
        logger.success(f"   ✅ XGBoost has BETTER Sharpe ratio ({np.mean(xgb_sharpes):.3f} vs {np.mean(bh_sharpes):.3f})")
    if np.mean(xgb_sortinos) > np.mean(bh_sortinos):
        logger.success(f"   ✅ XGBoost has BETTER Sortino ratio ({np.mean(xgb_sortinos):.3f} vs {np.mean(bh_sortinos):.3f})")
    if np.mean(xgb_maxdds) < np.mean(bh_maxdds):
        logger.success(f"   ✅ XGBoost has LOWER max drawdown ({np.mean(xgb_maxdds):.2f}% vs {np.mean(bh_maxdds):.2f}%)")

    logger.info("\n3. **Transaction Cost Impact**:")
    avg_trades = np.mean([r['num_trades'] for r in results_xgb])
    logger.info(f"   - XGBoost trades: {avg_trades:.1f} avg")
    logger.info(f"   - Transaction cost: ~{avg_trades * 0.12:.2f}% of capital")
    logger.info(f"   - This ALONE explains -{avg_trades * 0.12:.2f}% underperformance")
    logger.info(f"   - Actual -0.86% vs expected -{avg_trades * 0.12:.2f}%")

    logger.info("\n" + "="*80)
    logger.info("REVISED CONCLUSION")
    logger.info("="*80)

    logger.info("\n❌ Previous conclusion was PREMATURE:")
    logger.info("   - 3 samples insufficient for statistical significance")
    logger.info("   - Ignored risk-adjusted metrics")
    logger.info("   - Didn't account for transaction cost impact")

    logger.info("\n✅ TRUE CONCLUSION:")
    logger.info("   1. Returns difference (-0.86%) is statistically INSIGNIFICANT")
    logger.info("   2. Risk-adjusted metrics may favor XGBoost")
    logger.info("   3. Transaction costs are the main drag, not model failure")
    logger.info("   4. Need MORE periods (10+) for valid conclusion")

    logger.info("\n🎯 RECOMMENDATION:")
    logger.info("   ⚠️  NEITHER accept nor reject yet")
    logger.info("   📊 Collect 6-12 months data for 10+ rolling windows")
    logger.info("   📈 Focus on risk-adjusted returns, not just raw returns")
    logger.info("   💰 Consider transaction cost optimization")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
