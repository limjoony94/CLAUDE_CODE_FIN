"""
Lookback Period Optimization via Grid Search

Purpose:
1. Test various lookback periods for key indicators (RSI, MACD, BB, ATR)
2. Train models with different period combinations
3. Backtest each combination
4. Find optimal periods based on performance metrics

Key Indicators to Optimize:
- RSI: [7, 10, 14, 20, 25]
- MACD Fast: [8, 12, 16]
- MACD Slow: [17, 26, 35]
- Bollinger Bands: [10, 15, 20, 25, 30]
- ATR: [7, 14, 21, 28]
- Support/Resistance: [30, 50, 100]
- Trendline: [20, 30, 50]

Performance Metrics:
- Sharpe Ratio (primary)
- Return
- Win Rate
- Max Drawdown
- Profit Factor

Author: Claude Code
Date: 2025-10-23
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "claudedocs"

# Import necessary modules (will be modified for each iteration)
import talib
import ta


class LookbackOptimizer:
    """
    Optimize lookback periods for technical indicators
    """

    def __init__(self, data_file=None):
        """Initialize with historical data"""
        if data_file is None:
            data_file = DATA_DIR / "BTCUSDT_5m_max.csv"

        print(f"Loading data: {data_file}")
        self.df_full = pd.read_csv(data_file)
        print(f"Data loaded: {len(self.df_full)} rows")

        # Use recent data for optimization (faster)
        self.df = self.df_full.tail(30000).copy()
        print(f"Using last 30,000 candles for optimization")

    def calculate_features_with_periods(self, df, periods):
        """
        Calculate technical indicators with specified lookback periods

        Args:
            df: OHLCV DataFrame
            periods: Dict with keys ['rsi', 'macd_fast', 'macd_slow', 'bb', 'atr']

        Returns:
            DataFrame with calculated features
        """
        df = df.copy()

        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=periods['rsi'])

        # MACD
        macd = ta.trend.MACD(
            df['close'],
            window_fast=periods['macd_fast'],
            window_slow=periods['macd_slow'],
            window_sign=9  # Signal period fixed at 9
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=periods['bb'], window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()

        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=periods['atr'])

        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price changes
        df['close_change_1'] = df['close'].pct_change(1)
        df['close_change_3'] = df['close'].pct_change(3)

        return df

    def create_simple_strategy_signals(self, df):
        """
        Create simple LONG/SHORT signals based on indicators

        Simple Rules:
        - LONG: RSI < 50 and MACD > 0 and Price < BB_LOW
        - SHORT: RSI > 50 and MACD < 0 and Price > BB_HIGH

        Returns:
            signals: 1 for LONG, -1 for SHORT, 0 for neutral
        """
        df['signal'] = 0

        # LONG signals
        long_condition = (
            (df['rsi'] < 50) &
            (df['macd'] > 0) &
            (df['close'] < df['bb_low'])
        )
        df.loc[long_condition, 'signal'] = 1

        # SHORT signals
        short_condition = (
            (df['rsi'] > 50) &
            (df['macd'] < 0) &
            (df['close'] > df['bb_high'])
        )
        df.loc[short_condition, 'signal'] = -1

        return df['signal']

    def backtest_simple_strategy(self, df, leverage=4, tp_pct=0.03, sl_pct=0.03, max_hold=120):
        """
        Backtest simple strategy with given parameters

        Args:
            df: DataFrame with signals
            leverage: Leverage multiplier
            tp_pct: Take profit percentage (on leveraged P&L)
            sl_pct: Stop loss percentage (on leveraged P&L)
            max_hold: Maximum hold time in candles

        Returns:
            Performance metrics dict
        """
        df = df.copy()
        df['returns'] = 0.0
        position = 0
        entry_price = 0
        entry_idx = 0

        trades = []

        for i in range(len(df)):
            # Check for exit conditions if in position
            if position != 0:
                current_price = df['close'].iloc[i]
                hold_time = i - entry_idx

                # Calculate leveraged P&L percentage
                if position == 1:  # LONG
                    pnl_pct = ((current_price - entry_price) / entry_price) * leverage
                else:  # SHORT
                    pnl_pct = ((entry_price - current_price) / entry_price) * leverage

                # Check exit conditions
                exit_reason = None

                if pnl_pct >= tp_pct:
                    exit_reason = 'TP'
                elif pnl_pct <= -sl_pct:
                    exit_reason = 'SL'
                elif hold_time >= max_hold:
                    exit_reason = 'MaxHold'

                if exit_reason:
                    # Close position
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'side': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'hold_time': hold_time,
                        'exit_reason': exit_reason
                    })

                    position = 0
                    entry_price = 0

            # Check for entry signals if no position
            if position == 0:
                signal = df['signal'].iloc[i]

                if signal == 1:  # LONG entry
                    position = 1
                    entry_price = df['close'].iloc[i]
                    entry_idx = i
                elif signal == -1:  # SHORT entry
                    position = -1
                    entry_price = df['close'].iloc[i]
                    entry_idx = i

        # Calculate performance metrics
        if len(trades) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0
            }

        trades_df = pd.DataFrame(trades)

        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]

        win_rate = len(wins) / len(trades_df)
        total_return = trades_df['pnl_pct'].sum()

        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0

        # Sharpe Ratio (annualized)
        if trades_df['pnl_pct'].std() > 0:
            sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252 * 288 / len(trades_df))
        else:
            sharpe = 0

        # Max Drawdown
        cumulative = (1 + trades_df['pnl_pct']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Profit Factor
        gross_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def grid_search(self, param_grid):
        """
        Perform grid search over parameter combinations

        Args:
            param_grid: Dict with lists of values for each parameter

        Returns:
            Results DataFrame sorted by performance
        """
        print("\n" + "="*80)
        print("LOOKBACK PERIOD GRID SEARCH")
        print("="*80)

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combinations = list(itertools.product(*param_values))

        print(f"\nTotal combinations to test: {len(combinations)}")
        print(f"Parameters: {param_names}")

        results = []

        for idx, combo in enumerate(combinations, 1):
            # Create period dict
            periods = dict(zip(param_names, combo))

            print(f"\n[{idx}/{len(combinations)}] Testing: {periods}")

            try:
                # Calculate features
                df_test = self.calculate_features_with_periods(self.df, periods)

                # Generate signals
                df_test['signal'] = self.create_simple_strategy_signals(df_test)

                # Clean NaN
                df_test = df_test.dropna()

                # Backtest
                metrics = self.backtest_simple_strategy(df_test)

                # Add periods to results
                result = {**periods, **metrics}
                results.append(result)

                # Print quick summary
                print(f"  Trades: {metrics['num_trades']}, WR: {metrics['win_rate']:.1%}, "
                      f"Return: {metrics['total_return']:.2%}, Sharpe: {metrics['sharpe_ratio']:.3f}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate composite score
        # Normalize each metric and weight them
        results_df['score'] = (
            results_df['sharpe_ratio'] / results_df['sharpe_ratio'].max() * 0.40 +
            results_df['total_return'] / results_df['total_return'].max() * 0.30 +
            results_df['win_rate'] / results_df['win_rate'].max() * 0.20 +
            (1 + results_df['max_drawdown']) / (1 + results_df['max_drawdown']).max() * 0.10
        )

        # Sort by score
        results_df = results_df.sort_values('score', ascending=False)

        return results_df


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("LOOKBACK PERIOD OPTIMIZATION - GRID SEARCH")
    print("="*80)

    # Initialize optimizer
    optimizer = LookbackOptimizer()

    # Define parameter grid (COARSE SEARCH FIRST)
    param_grid = {
        'rsi': [10, 14, 20],  # 3 values
        'macd_fast': [8, 12, 16],  # 3 values
        'macd_slow': [17, 26, 35],  # 3 values
        'bb': [15, 20, 25],  # 3 values
        'atr': [7, 14, 21],  # 3 values
    }

    print("\nParameter Grid (Coarse Search):")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal combinations: {total_combinations}")

    # Run grid search
    results = optimizer.grid_search(param_grid)

    # Display top 10 results
    print("\n" + "="*80)
    print("TOP 10 COMBINATIONS")
    print("="*80)

    top_10 = results.head(10)

    for idx, row in top_10.iterrows():
        print(f"\n{'='*80}")
        print(f"Rank {list(top_10.index).index(idx) + 1}: Score {row['score']:.4f}")
        print(f"{'='*80}")
        print(f"Parameters:")
        print(f"  RSI: {row['rsi']:.0f}, MACD: {row['macd_fast']:.0f}/{row['macd_slow']:.0f}, "
              f"BB: {row['bb']:.0f}, ATR: {row['atr']:.0f}")
        print(f"\nPerformance:")
        print(f"  Trades: {row['num_trades']:.0f}")
        print(f"  Win Rate: {row['win_rate']:.2%}")
        print(f"  Total Return: {row['total_return']:.2%}")
        print(f"  Sharpe Ratio: {row['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {row['max_drawdown']:.2%}")
        print(f"  Profit Factor: {row['profit_factor']:.2f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "LOOKBACK_PERIOD_OPTIMIZATION_20251023.csv"
    results.to_csv(results_file, index=False)
    print(f"\n✅ Results saved: {results_file}")

    # Generate report
    report_file = OUTPUT_DIR / "LOOKBACK_PERIOD_OPTIMIZATION_REPORT_20251023.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Lookback Period Optimization - Grid Search Results\n")
        f.write(f"**Date**: 2025-10-23\n")
        f.write(f"**Status**: ✅ Coarse Search Complete\n\n")
        f.write("---\n\n")

        f.write("## Parameter Grid\n\n")
        f.write("```yaml\n")
        for param, values in param_grid.items():
            f.write(f"{param}: {values}\n")
        f.write("```\n\n")

        f.write(f"Total Combinations: {total_combinations}\n\n")

        f.write("## Top 10 Results\n\n")
        f.write("| Rank | RSI | MACD | BB | ATR | WR | Return | Sharpe | Score |\n")
        f.write("|------|-----|------|----|----|-------|--------|--------|-------|\n")

        for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"| {rank} | {row['rsi']:.0f} | "
                   f"{row['macd_fast']:.0f}/{row['macd_slow']:.0f} | "
                   f"{row['bb']:.0f} | {row['atr']:.0f} | "
                   f"{row['win_rate']:.1%} | {row['total_return']:.1%} | "
                   f"{row['sharpe_ratio']:.3f} | {row['score']:.4f} |\n")

        f.write("\n## Recommendations\n\n")
        f.write("### Best Overall (Rank 1)\n")
        best = top_10.iloc[0]
        f.write(f"- RSI: {best['rsi']:.0f}\n")
        f.write(f"- MACD: {best['macd_fast']:.0f}/{best['macd_slow']:.0f}\n")
        f.write(f"- Bollinger Bands: {best['bb']:.0f}\n")
        f.write(f"- ATR: {best['atr']:.0f}\n\n")

        f.write(f"**Performance**:\n")
        f.write(f"- Win Rate: {best['win_rate']:.2%}\n")
        f.write(f"- Total Return: {best['total_return']:.2%}\n")
        f.write(f"- Sharpe Ratio: {best['sharpe_ratio']:.3f}\n")
        f.write(f"- Max Drawdown: {best['max_drawdown']:.2%}\n")
        f.write(f"- Profit Factor: {best['profit_factor']:.2f}\n\n")

        f.write("### Next Steps\n\n")
        f.write("1. **Fine-Grained Search**: Test variations around best combination\n")
        f.write("2. **Validation**: Test on different time periods\n")
        f.write("3. **ML Model Training**: Retrain models with optimal periods\n")
        f.write("4. **Backtest Validation**: Full backtest with production strategy\n")

    print(f"✅ Report saved: {report_file}")

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print("\nBest Combination:")
    best = top_10.iloc[0]
    print(f"  RSI: {best['rsi']:.0f}")
    print(f"  MACD: {best['macd_fast']:.0f}/{best['macd_slow']:.0f}")
    print(f"  BB: {best['bb']:.0f}")
    print(f"  ATR: {best['atr']:.0f}")
    print(f"\nPerformance:")
    print(f"  Sharpe: {best['sharpe_ratio']:.3f}")
    print(f"  Return: {best['total_return']:.2%}")
    print(f"  Win Rate: {best['win_rate']:.2%}")


if __name__ == "__main__":
    main()
