"""
Backtest XGBoost Phase 4 - 3-Class Model

Tests 3-class classification model for LONG + SHORT positions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

def classify_market_regime(df_window):
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"

def backtest_strategy_3class(df, model, feature_columns, threshold=0.7):
    """
    Backtest with 3-class model

    Class 0: NEUTRAL
    Class 1: LONG
    Class 2: SHORT
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L based on position side
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Check exit conditions
            exit_reason = None
            if pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                total_cost = entry_cost + exit_cost
                pnl_usd -= total_cost

                capital += pnl_usd

                trades.append({
                    'side': side,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'long_prob': position['long_prob'],
                    'short_prob': position['short_prob']
                })

                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            # Get 3-class probabilities
            probs = model.predict_proba(features)[0]
            neutral_prob = probs[0]
            long_prob = probs[1]
            short_prob = probs[2]

            # Determine entry
            side = None
            if long_prob >= threshold:
                side = 'LONG'
            elif short_prob >= threshold:
                side = 'SHORT'

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'long_prob': long_prob,
                    'short_prob': short_prob
                }

    # Calculate metrics
    if len(trades) == 0:
        return trades, {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'num_long': 0,
            'num_short': 0,
            'win_rate': 0.0,
            'win_rate_long': 0.0,
            'win_rate_short': 0.0,
            'avg_pnl_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) * 100

    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    if len(long_trades) > 0:
        long_wins = [t for t in long_trades if t['pnl_usd'] > 0]
        win_rate_long = (len(long_wins) / len(long_trades)) * 100
    else:
        win_rate_long = 0.0

    if len(short_trades) > 0:
        short_wins = [t for t in short_trades if t['pnl_usd'] > 0]
        win_rate_short = (len(short_wins) / len(short_trades)) * 100
    else:
        win_rate_short = 0.0

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd']
        cumulative_returns.append(running_capital)

    if len(cumulative_returns) > 0:
        peak = cumulative_returns[0]
        max_dd = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = 0.0

    # Sharpe
    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    metrics = {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'win_rate': win_rate,
        'win_rate_long': win_rate_long,
        'win_rate_short': win_rate_short,
        'avg_pnl_pct': avg_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }

    return trades, metrics

def rolling_window_backtest(df, model, feature_columns, threshold=0.7):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_strategy_3class(window_df, model, feature_columns, threshold)

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'regime': regime,
            'xgb_return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'num_long': metrics['num_long'],
            'num_short': metrics['num_short'],
            'win_rate': metrics['win_rate'],
            'win_rate_long': metrics['win_rate_long'],
            'win_rate_short': metrics['win_rate_short'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


print("=" * 80)
print("XGBoost Phase 4 - 3-Class Model Backtest")
print("=" * 80)

# Load model
model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
print(f"✅ Model loaded: {model_file}")

# Load feature columns
feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]
print(f"✅ Features loaded: {len(feature_columns)} features")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate features
print("Calculating baseline features...")
df = calculate_features(df)

print("Calculating advanced technical features...")
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

df = df.ffill()
df = df.dropna()
print(f"✅ Features calculated: {len(df)} rows")

# Test
print(f"\n{'=' * 80}")
print(f"Testing 3-Class Model (threshold=0.7)")
print(f"{'=' * 80}")

results = rolling_window_backtest(df, model, feature_columns, threshold=0.7)

# Summary
print(f"\nResults ({len(results)} windows):")
print(f"  XGBoost Return: {results['xgb_return'].mean():.2f}% ± {results['xgb_return'].std():.2f}%")
print(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
print(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
print(f"\n  📊 Trade Breakdown:")
print(f"    Total Trades: {results['num_trades'].mean():.1f}")
print(f"    LONG Trades: {results['num_long'].mean():.1f}")
print(f"    SHORT Trades: {results['num_short'].mean():.1f}")
print(f"\n  🎯 Win Rates:")
print(f"    Overall: {results['win_rate'].mean():.1f}%")
print(f"    LONG: {results['win_rate_long'].mean():.1f}%")
print(f"    SHORT: {results['win_rate_short'].mean():.1f}%")
print(f"\n  📈 Risk Metrics:")
print(f"    Avg Sharpe: {results['sharpe'].mean():.3f}")
print(f"    Avg Max DD: {results['max_dd'].mean():.2f}%")

# By regime
print(f"\n{'=' * 80}")
print(f"By Market Regime:")
print(f"{'=' * 80}")
for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = results[results['regime'] == regime]
    if len(regime_df) > 0:
        print(f"\n  {regime} ({len(regime_df)} windows):")
        print(f"    XGBoost: {regime_df['xgb_return'].mean():.2f}%")
        print(f"    Buy & Hold: {regime_df['bh_return'].mean():.2f}%")
        print(f"    Difference: {regime_df['difference'].mean():.2f}%")
        print(f"    Trades: {regime_df['num_trades'].mean():.1f} (LONG: {regime_df['num_long'].mean():.1f}, SHORT: {regime_df['num_short'].mean():.1f})")
        print(f"    Win Rate: {regime_df['win_rate'].mean():.1f}% (LONG: {regime_df['win_rate_long'].mean():.1f}%, SHORT: {regime_df['win_rate_short'].mean():.1f}%)")

# Save
output_file = RESULTS_DIR / f"backtest_phase4_3class.csv"
results.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(results['xgb_return'], results['bh_return'])

print(f"\n{'=' * 80}")
print("Statistical Significance")
print(f"{'=' * 80}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

# Decision
print(f"\n{'=' * 80}")
print("🎯 3-Class Model Evaluation")
print(f"{'=' * 80}")

short_win_rate = results['win_rate_short'].mean()
overall_improvement = results['difference'].mean()

print(f"\n  Overall Performance: {overall_improvement:+.2f}% vs B&H")
print(f"  Win Rate: {results['win_rate'].mean():.1f}%")
print(f"  SHORT Win Rate: {short_win_rate:.1f}%")

if short_win_rate >= 60 and overall_improvement > 0:
    print(f"\n  ✅ 3-CLASS MODEL SUCCESS!")
    print(f"     - SHORT win rate: {short_win_rate:.1f}% (>= 60%)")
    print(f"     - Improvement: {overall_improvement:+.2f}%")
    print(f"\n  🎯 DEPLOY 3-class model")
else:
    print(f"\n  ❌ 3-CLASS MODEL NEEDS IMPROVEMENT")
    print(f"     - SHORT win rate: {short_win_rate:.1f}%")
    print(f"     - Improvement: {overall_improvement:+.2f}%")
    print(f"\n  🎯 NEXT: Retrain with class balancing")

print(f"\n{'=' * 80}\n")
