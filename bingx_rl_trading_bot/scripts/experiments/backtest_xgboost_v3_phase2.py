"""
Backtest XGBoost V3 (Phase 2) with Short-term Features

비판적 사고:
- "Phase 1: -1.86% vs B&H (거래 품질 낮음)"
- "Phase 2 (Short-term features +15): 거래 품질 향상?"
- "F1-Score +3.2% → 실제 성과 개선?"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import ta
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_xgboost_improved_v3_phase2 import calculate_features

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0006

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

def backtest_strategy(df, model, feature_columns, entry_threshold, min_volatility=0.0008):
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            pnl_pct = (current_price - entry_price) / entry_price

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
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'probability': position['probability']
                })

                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probability = model.predict_proba(features)[0][1]

            current_volatility = df['volatility'].iloc[i]
            if pd.isna(current_volatility) or current_volatility < min_volatility:
                continue

            should_enter = (probability > entry_threshold)

            if should_enter:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'probability': probability
                }

    # Calculate metrics
    if len(trades) == 0:
        return trades, {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) * 100

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
        'win_rate': win_rate,
        'avg_pnl_pct': avg_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }

    return trades, metrics

def rolling_window_backtest(df, model, feature_columns, entry_threshold):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_strategy(window_df, model, feature_columns, entry_threshold)

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
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)

print("=" * 80)
print("XGBoost V3 (Phase 2) Backtest - Short-term Features")
print("=" * 80)

# Load model
model_file = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
print(f"✅ Model loaded: {model_file}")

# Load feature columns
feature_file = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2_features.txt"
with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]
print(f"✅ Features loaded: {len(feature_columns)} features")

# Load data
data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate features (Phase 2 with short-term)
df = calculate_features(df)
df = df.dropna()
print(f"✅ Features calculated: {len(df)} rows after dropna")

# Test with threshold 0.3
threshold = 0.3

print(f"\n{'=' * 80}")
print(f"Testing Phase 2 Model with threshold {threshold:.2f}")
print(f"{'=' * 80}")

results = rolling_window_backtest(df, model, feature_columns, threshold)

# Summary
print(f"\nResults ({len(results)} windows):")
print(f"  XGBoost Return: {results['xgb_return'].mean():.2f}% ± {results['xgb_return'].std():.2f}%")
print(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
print(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
print(f"  🎯 Avg Trades per Window: {results['num_trades'].mean():.1f}")
print(f"  Avg Win Rate: {results['win_rate'].mean():.1f}%")
print(f"  Avg Sharpe: {results['sharpe'].mean():.3f}")
print(f"  Avg Max DD: {results['max_dd'].mean():.2f}%")

# By regime
print(f"\nBy Market Regime:")
for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = results[results['regime'] == regime]
    if len(regime_df) > 0:
        print(f"  {regime} ({len(regime_df)} windows):")
        print(f"    XGBoost: {regime_df['xgb_return'].mean():.2f}%")
        print(f"    Buy & Hold: {regime_df['bh_return'].mean():.2f}%")
        print(f"    Difference: {regime_df['difference'].mean():.2f}%")
        print(f"    Trades: {regime_df['num_trades'].mean():.1f}")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(results['xgb_return'], results['bh_return'])
print(f"\nStatistical Test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

# Save
output_file = RESULTS_DIR / f"backtest_v3_phase2_thresh{int(threshold*10)}.csv"
results.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

print(f"\n{'=' * 80}")
print("Phase 2 Backtest Complete!")
print(f"{'=' * 80}")

print(f"\n🎯 비판적 분석:")
print(f"\n  Phase 1 (18 features):")
print(f"    - Avg Return: -2.10%")
print(f"    - vs B&H: -2.01%")
print(f"    - Avg Trades: 18.5")
print(f"    - Win Rate: 47.6%")

print(f"\n  Phase 2 (33 features, +15 short-term):")
print(f"    - Avg Return: {results['xgb_return'].mean():.2f}%")
print(f"    - vs B&H: {results['difference'].mean():.2f}%")
print(f"    - Avg Trades: {results['num_trades'].mean():.1f}")
print(f"    - Win Rate: {results['win_rate'].mean():.1f}%")

# Comparison
improvement_return = results['difference'].mean() - (-2.01)
improvement_trades = results['num_trades'].mean() - 18.5
improvement_winrate = results['win_rate'].mean() - 47.6

print(f"\n  개선 효과:")
print(f"    - vs B&H: {improvement_return:+.2f}%p")
print(f"    - Trades: {improvement_trades:+.1f}")
print(f"    - Win Rate: {improvement_winrate:+.1f}%p")

if results['difference'].mean() > -2.01:
    print(f"\n  ✅ Phase 2 성공: Return vs B&H 개선!")
    if results['difference'].mean() > 0:
        print(f"  🎉 Buy & Hold 이김! (+{results['difference'].mean():.2f}%)")
else:
    print(f"\n  ⚠️ Phase 2 부분 성공: 추가 개선 필요")
