"""
Comprehensive Threshold Combination Backtest

목적: 여러 LONG/SHORT threshold 조합을 백테스트해서 수익성 기준으로 최적값 선택

테스트 조합:
- LONG thresholds: [0.70, 0.75, 0.80]
- SHORT thresholds: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
- Total: 21 combinations

성능 지표:
- Total Return (%)
- Sharpe Ratio
- Win Rate (%)
- Trades/Week
- Max Drawdown (%)
- LONG/SHORT Distribution (%)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from itertools import product

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Configuration
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("=" * 80)
print("COMPREHENSIVE THRESHOLD COMBINATION BACKTEST")
print("=" * 80)
print()

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"✅ Loaded {len(df)} candles")

# Calculate features
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures()
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features calculated: {len(df)} rows")

# Load models
with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl', 'rb') as f:
    long_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl', 'rb') as f:
    long_scaler = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_model_lookahead3_thresh0.3.pkl', 'rb') as f:
    short_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_model_lookahead3_thresh0.3_scaler.pkl', 'rb') as f:
    short_scaler = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt', 'r') as f:
    feature_columns = [line.strip() for line in f]

print(f"✅ Models loaded (using simple exit rules: SL/TP/MaxHold)")
print()

# Get predictions
X = df[feature_columns].values
X_long_scaled = long_scaler.transform(X)
X_short_scaled = short_scaler.transform(X)

prob_long_entry = long_model.predict_proba(X_long_scaled)[:, 1]
prob_short_entry = short_model.predict_proba(X_short_scaled)[:, 1]

df['prob_long_entry'] = prob_long_entry
df['prob_short_entry'] = prob_short_entry

print(f"✅ Entry probabilities calculated")

# Test range (last 20% of data for consistency)
test_size = int(len(df) * 0.2)
test_start = len(df) - test_size
df_test = df.iloc[test_start:].copy().reset_index(drop=True)

weeks = len(df_test) / (12 * 24 * 7)  # 5-min candles to weeks
print(f"✅ Test set: {len(df_test)} candles ({weeks:.1f} weeks)")
print()

# Threshold combinations to test
long_thresholds = [0.70, 0.75, 0.80]
short_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

print(f"Testing {len(long_thresholds)} × {len(short_thresholds)} = {len(long_thresholds) * len(short_thresholds)} combinations")
print()

results = []

for long_thresh, short_thresh in product(long_thresholds, short_thresholds):
    # Generate signals
    df_test['signal_long'] = (df_test['prob_long_entry'] >= long_thresh).astype(int)
    df_test['signal_short'] = (df_test['prob_short_entry'] >= short_thresh).astype(int)

    # Simulate trading
    position = None
    entry_price = 0
    entry_idx = 0
    direction = None
    trades = []
    equity = 1.0  # Start with 1.0 (100%)
    equity_curve = [1.0]

    for i in range(len(df_test)):
        current_price = df_test.loc[i, 'close']

        # Exit logic (if in position)
        if position is not None:
            hold_time = i - entry_idx

            # Calculate exit features
            if direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                pnl_from_peak = pnl_pct - df_test.loc[entry_idx:i, 'close'].pct_change().cumsum().max() if i > entry_idx else 0

                # Exit conditions
                should_exit = (
                    hold_time >= 48 or  # Max hold: 4 hours
                    pnl_pct <= -0.01 or  # Stop loss: -1%
                    pnl_pct >= 0.02  # Take profit: +2%
                )

                if should_exit:
                    trades.append({
                        'direction': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'hold_time': hold_time
                    })
                    equity *= (1 + pnl_pct * 0.99)  # 0.1% fee
                    equity_curve.append(equity)
                    position = None

            elif direction == 'SHORT':
                pnl_pct = (entry_price - current_price) / entry_price
                pnl_from_peak = pnl_pct - df_test.loc[entry_idx:i, 'close'].pct_change().mul(-1).cumsum().max() if i > entry_idx else 0

                # Exit conditions
                should_exit = (
                    hold_time >= 48 or  # Max hold: 4 hours
                    pnl_pct <= -0.01 or  # Stop loss: -1%
                    pnl_pct >= 0.02  # Take profit: +2%
                )

                if should_exit:
                    trades.append({
                        'direction': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'hold_time': hold_time
                    })
                    equity *= (1 + pnl_pct * 0.99)  # 0.1% fee
                    equity_curve.append(equity)
                    position = None

        # Entry logic (if not in position)
        if position is None:
            if df_test.loc[i, 'signal_long'] == 1:
                position = 'LONG'
                entry_price = current_price
                entry_idx = i
                direction = 'LONG'
            elif df_test.loc[i, 'signal_short'] == 1:
                position = 'SHORT'
                entry_price = current_price
                entry_idx = i
                direction = 'SHORT'

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_return = (equity - 1.0) * 100
        win_rate = (trades_df['pnl_pct'] > 0).sum() / len(trades_df) * 100
        trades_per_week = len(trades_df) / weeks

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(12 * 24 * 7) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_dd = drawdown.min()

        # Distribution
        long_trades = (trades_df['direction'] == 'LONG').sum()
        short_trades = (trades_df['direction'] == 'SHORT').sum()
        long_pct = long_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
        short_pct = short_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0

        results.append({
            'long_thresh': long_thresh,
            'short_thresh': short_thresh,
            'total_return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'trades_per_week': trades_per_week,
            'max_dd': max_dd,
            'long_pct': long_pct,
            'short_pct': short_pct,
            'total_trades': len(trades_df)
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by total return (descending)
results_df = results_df.sort_values('total_return', ascending=False)

print("=" * 80)
print("BACKTEST RESULTS (Sorted by Total Return)")
print("=" * 80)
print()
print(f"{'Rank':<5} {'LONG':<6} {'SHORT':<6} {'Return%':>8} {'Sharpe':>7} {'WinRate%':>9} {'Trades/W':>9} {'MaxDD%':>8} {'LONG%':>7} {'SHORT%':>7}")
print("-" * 80)

for idx, row in results_df.head(10).iterrows():
    marker = " ← BEST" if idx == results_df.index[0] else ""
    print(f"{results_df.index.get_loc(idx)+1:<5} {row['long_thresh']:<6.2f} {row['short_thresh']:<6.2f} {row['total_return']:>8.2f} {row['sharpe']:>7.2f} {row['win_rate']:>9.1f} {row['trades_per_week']:>9.1f} {row['max_dd']:>8.2f} {row['long_pct']:>7.1f} {row['short_pct']:>7.1f}{marker}")

# Save results
results_file = RESULTS_DIR / "threshold_combination_backtest_results.csv"
results_df.to_csv(results_file, index=False)
print()
print(f"✅ Results saved: {results_file}")

# Best configuration
best = results_df.iloc[0]

print()
print("=" * 80)
print("🏆 BEST CONFIGURATION")
print("=" * 80)
print(f"LONG Threshold: {best['long_thresh']:.2f}")
print(f"SHORT Threshold: {best['short_thresh']:.2f}")
print()
print(f"Performance:")
print(f"  Total Return: {best['total_return']:.2f}%")
print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Trades/Week: {best['trades_per_week']:.1f}")
print(f"  Max Drawdown: {best['max_dd']:.2f}%")
print()
print(f"Distribution:")
print(f"  LONG trades: {best['long_pct']:.1f}%")
print(f"  SHORT trades: {best['short_pct']:.1f}%")
print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print(f"Update phase4_dynamic_testnet_trading.py:")
print(f"  LONG_ENTRY_THRESHOLD = {best['long_thresh']:.2f}")
print(f"  SHORT_ENTRY_THRESHOLD = {best['short_thresh']:.2f}")
