"""
Backtest Trained Exit Model vs Fixed Exit

목표: 훈련된 Exit Model의 실전 성능 검증
- Exit Model @ various thresholds (0.1, 0.2, 0.3, 0.5)
- vs Fixed Exit (TP 1.5%, SL 1%, MaxHold 2h)

Expected: Exit Model이 더 나은 타이밍으로 Fixed를 능가
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.0002
LEVERAGE = 4
ENTRY_THRESHOLD = 0.7

# Safety nets
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)
MAX_HOLDING_HOURS = 8

def classify_market_regime(df_window):
    """Classify market regime"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"

def backtest_exit_model(df, entry_model, exit_model, entry_feature_cols, exit_feature_cols,
                        entry_threshold, exit_threshold, leverage, position_size=0.95):
    """
    Backtest with trained exit model

    Args:
        exit_threshold: Probability threshold for exit signal (0-1)
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    liquidations = 0

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position with EXIT MODEL
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            entry_signal = position['entry_signal']
            hours_held = (i - entry_idx) / 12

            # P&L with leverage
            price_change_pct = (current_price - entry_price) / entry_price
            leveraged_pnl_pct = price_change_pct * leverage
            leveraged_pnl_usd = leveraged_pnl_pct * position['base_value']

            # Liquidation check
            liquidation_threshold = -0.95 / leverage
            if leveraged_pnl_pct <= liquidation_threshold:
                liquidations += 1
                exit_reason = "LIQUIDATION"
                leveraged_pnl_usd = -position['base_value']
                net_pnl_usd = leveraged_pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'exit_signal': None,
                })

                capital += net_pnl_usd
                position = None
                continue

            # Get exit signal from EXIT MODEL
            tech_features = df[entry_feature_cols].iloc[i:i+1].values[0]
            exit_reason = None
            exit_prob = None

            if not np.isnan(tech_features).any():
                # Position features
                position_features = np.array([
                    leveraged_pnl_pct,  # current P&L (leveraged)
                    hours_held,
                    entry_signal,
                    (current_price - entry_price) / entry_price,
                ])

                # Combined features for exit model
                exit_features = np.concatenate([tech_features, position_features]).reshape(1, -1)

                # Predict exit
                exit_prob = exit_model.predict_proba(exit_features)[0][1]

                # Exit decision based on threshold
                if exit_prob >= exit_threshold:
                    exit_reason = f"Exit Model (prob={exit_prob:.3f})"

            # Safety nets (emergency exits)
            if exit_reason is None:
                if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                    exit_reason = "Emergency SL"
                elif hours_held >= MAX_HOLDING_HOURS:
                    exit_reason = "Max Holding"

            if exit_reason:
                entry_cost = position['leveraged_value'] * TRANSACTION_COST
                exit_cost = (current_price / entry_price) * position['leveraged_value'] * TRANSACTION_COST
                total_cost = entry_cost + exit_cost

                net_pnl_usd = leveraged_pnl_usd - total_cost

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'exit_signal': exit_prob,
                })

                capital += net_pnl_usd
                position = None

        # Entry logic (same for all)
        if position is None and i < len(df) - 1:
            if capital <= 0:
                break

            features = df[entry_feature_cols].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            prob_long = entry_model.predict_proba(features)[0][1]

            if prob_long < entry_threshold:
                continue

            # Enter position
            base_value = capital * position_size
            leveraged_value = base_value * leverage

            position = {
                'entry_idx': i,
                'entry_price': current_price,
                'entry_signal': prob_long,
                'base_value': base_value,
                'leveraged_value': leveraged_value,
            }

    # Metrics
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'liquidations': 0,
            'final_capital': capital,
            'avg_holding_hours': 0.0,
            'avg_exit_signal': 0.0,
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd_net'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd_net']
        cumulative_returns.append(running_capital)

    peak = cumulative_returns[0]
    max_dd = 0
    for value in cumulative_returns:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe
    if len(trades) > 1:
        returns = [t['leveraged_pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    # Signal analysis
    exit_signals = [t['exit_signal'] for t in trades if t['exit_signal'] is not None]
    avg_exit_signal = np.mean(exit_signals) if len(exit_signals) > 0 else 0.0
    avg_holding_hours = np.mean([t['holding_hours'] for t in trades])

    metrics = {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'liquidations': liquidations,
        'final_capital': capital,
        'avg_holding_hours': avg_holding_hours,
        'avg_exit_signal': avg_exit_signal,
    }

    return metrics

def backtest_fixed_exit(df, entry_model, feature_cols, entry_threshold,
                       stop_loss, take_profit, max_holding, leverage, position_size=0.95):
    """Backtest with fixed exit strategy (for comparison)"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    liquidations = 0

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position with FIXED exits
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # P&L with leverage
            price_change_pct = (current_price - entry_price) / entry_price
            leveraged_pnl_pct = price_change_pct * leverage
            leveraged_pnl_usd = leveraged_pnl_pct * position['base_value']

            # Liquidation check
            liquidation_threshold = -0.95 / leverage
            if leveraged_pnl_pct <= liquidation_threshold:
                liquidations += 1
                exit_reason = "LIQUIDATION"
                leveraged_pnl_usd = -position['base_value']
                net_pnl_usd = leveraged_pnl_usd

                trades.append({
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                })

                capital += net_pnl_usd
                position = None
                continue

            # Fixed exits
            exit_reason = None
            if leveraged_pnl_pct <= -stop_loss:
                exit_reason = "Stop Loss"
            elif leveraged_pnl_pct >= take_profit:
                exit_reason = "Take Profit"
            elif hours_held >= max_holding:
                exit_reason = "Max Holding"

            if exit_reason:
                entry_cost = position['leveraged_value'] * TRANSACTION_COST
                exit_cost = (current_price / entry_price) * position['leveraged_value'] * TRANSACTION_COST
                total_cost = entry_cost + exit_cost

                net_pnl_usd = leveraged_pnl_usd - total_cost

                trades.append({
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                })

                capital += net_pnl_usd
                position = None

        # Entry logic
        if position is None and i < len(df) - 1:
            if capital <= 0:
                break

            features = df[feature_cols].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            prob_long = entry_model.predict_proba(features)[0][1]

            if prob_long < entry_threshold:
                continue

            base_value = capital * position_size
            leveraged_value = base_value * leverage

            position = {
                'entry_idx': i,
                'entry_price': current_price,
                'base_value': base_value,
                'leveraged_value': leveraged_value,
            }

    # Calculate metrics (same as exit model backtest)
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'liquidations': 0,
            'final_capital': capital,
            'avg_holding_hours': 0.0,
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd_net'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd_net']
        cumulative_returns.append(running_capital)

    peak = cumulative_returns[0]
    max_dd = 0
    for value in cumulative_returns:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        if dd > max_dd:
            max_dd = dd

    if len(trades) > 1:
        returns = [t['leveraged_pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    avg_holding_hours = np.mean([t['holding_hours'] for t in trades])

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'liquidations': liquidations,
        'final_capital': capital,
        'avg_holding_hours': avg_holding_hours,
    }

def rolling_window_backtest_exit_model(df, entry_model, exit_model, entry_feature_cols, exit_feature_cols,
                                       entry_threshold, exit_threshold, leverage, position_size=0.95):
    """Rolling window backtest with exit model"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        metrics = backtest_exit_model(
            window_df, entry_model, exit_model, entry_feature_cols, exit_feature_cols,
            entry_threshold, exit_threshold, leverage, position_size
        )

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        windows.append({
            'start_idx': start_idx,
            'regime': regime,
            'return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown'],
            'liquidations': metrics['liquidations'],
            'avg_holding_hours': metrics['avg_holding_hours'],
            'avg_exit_signal': metrics['avg_exit_signal'],
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)

def rolling_window_backtest_fixed(df, entry_model, feature_cols, entry_threshold,
                                  sl, tp, maxhold, leverage, position_size=0.95):
    """Rolling window backtest with fixed exit"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        metrics = backtest_fixed_exit(
            window_df, entry_model, feature_cols, entry_threshold,
            sl, tp, maxhold, leverage, position_size
        )

        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        windows.append({
            'start_idx': start_idx,
            'regime': regime,
            'return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown'],
            'liquidations': metrics['liquidations'],
            'avg_holding_hours': metrics['avg_holding_hours'],
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


print("=" * 80)
print("EXIT MODEL BACKTEST vs FIXED EXIT")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load models
print("\n1. Loading models...")
entry_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
entry_feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

with open(entry_model_file, 'rb') as f:
    entry_model = pickle.load(f)

with open(entry_feature_file, 'r') as f:
    entry_feature_cols = [line.strip() for line in f.readlines()]

print(f"✅ Entry model loaded: {len(entry_feature_cols)} features")

# Load exit model (latest)
import glob
exit_model_files = glob.glob(str(MODELS_DIR / "xgboost_exit_model_*.pkl"))
if not exit_model_files:
    raise FileNotFoundError("No exit model found!")

latest_exit_model = max(exit_model_files, key=lambda x: Path(x).stat().st_mtime)
with open(latest_exit_model, 'rb') as f:
    exit_model = pickle.load(f)

print(f"✅ Exit model loaded: {Path(latest_exit_model).name}")

# Exit feature columns (37 technical + 4 position)
exit_feature_file = latest_exit_model.replace('.pkl', '_features.txt')
with open(exit_feature_file, 'r') as f:
    exit_feature_cols = [line.strip() for line in f.readlines()]

print(f"✅ Exit features: {len(exit_feature_cols)} ({len(entry_feature_cols)} tech + 4 position)")

# Load data
print("\n2. Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate features
print("\n3. Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features ready: {len(df)} rows")

position_size = 0.95

# Test configurations
test_configs = [
    {
        'name': 'Fixed: TP1.5%/SL1%/2h',
        'type': 'fixed',
        'params': {'sl': 0.01, 'tp': 0.015, 'maxhold': 2}
    },
    {
        'name': 'Exit Model @ 0.5',
        'type': 'exit_model',
        'params': {'threshold': 0.5}
    },
    {
        'name': 'Exit Model @ 0.3',
        'type': 'exit_model',
        'params': {'threshold': 0.3}
    },
    {
        'name': 'Exit Model @ 0.2',
        'type': 'exit_model',
        'params': {'threshold': 0.2}
    },
    {
        'name': 'Exit Model @ 0.1',
        'type': 'exit_model',
        'params': {'threshold': 0.1}
    },
]

print(f"\n{'=' * 80}")
print("TESTING CONFIGURATIONS")
print(f"{'=' * 80}")
print(f"Total Strategies: {len(test_configs)}")

results_all = []

for idx, config in enumerate(test_configs):
    strategy_name = config['name']
    strategy_type = config['type']
    params = config['params']

    print(f"\n[{idx+1}/{len(test_configs)}] Testing: {strategy_name}")

    if strategy_type == 'fixed':
        results = rolling_window_backtest_fixed(
            df, entry_model, entry_feature_cols, ENTRY_THRESHOLD,
            params['sl'], params['tp'], params['maxhold'],
            LEVERAGE, position_size
        )
    else:
        results = rolling_window_backtest_exit_model(
            df, entry_model, exit_model, entry_feature_cols, exit_feature_cols,
            ENTRY_THRESHOLD, params['threshold'], LEVERAGE, position_size
        )

    # Aggregate
    avg_return = results['return'].mean()
    avg_difference = results['difference'].mean()
    avg_win_rate = results['win_rate'].mean()
    avg_sharpe = results['sharpe'].mean()
    avg_max_dd = results['max_dd'].mean()
    total_liquidations = results['liquidations'].sum()
    avg_num_trades = results['num_trades'].mean()
    avg_holding_hours = results['avg_holding_hours'].mean()
    avg_exit_signal = results['avg_exit_signal'].mean() if 'avg_exit_signal' in results.columns else 0.0

    results_all.append({
        'strategy_name': strategy_name,
        'strategy_type': strategy_type,
        'avg_return': avg_return,
        'vs_bh': avg_difference,
        'win_rate': avg_win_rate,
        'sharpe': avg_sharpe,
        'max_dd': avg_max_dd,
        'liquidations': total_liquidations,
        'avg_trades': avg_num_trades,
        'avg_holding_hours': avg_holding_hours,
        'avg_exit_signal': avg_exit_signal,
    })

    print(f"   Return: {avg_return:+.2f}% | Win Rate: {avg_win_rate:.1f}% | Sharpe: {avg_sharpe:.2f}")

# Convert to DataFrame
df_results = pd.DataFrame(results_all)
df_results = df_results.sort_values('avg_return', ascending=False)

# Save results
output_file = RESULTS_DIR / "exit_model_backtest_results.csv"
df_results.to_csv(output_file, index=False)

print(f"\n{'=' * 80}")
print("BACKTEST COMPLETE!")
print(f"{'=' * 80}")
print(f"Results saved: {output_file.name}")

# Display results
print(f"\n{'=' * 80}")
print("ALL STRATEGIES RANKED BY RETURN")
print(f"{'=' * 80}")
print(f"\n{'Rank':<5} {'Strategy':<25} {'Return':<10} {'vs B&H':<10} {'WinRate':<10} {'Sharpe':<8} {'MaxDD':<8} {'Hold(h)':<9}")
print("-" * 110)

for idx, row in df_results.iterrows():
    rank = df_results.index.get_loc(idx) + 1
    print(f"{rank:<5} "
          f"{row['strategy_name']:<25} "
          f"{row['avg_return']:>8.2f}%  "
          f"{row['vs_bh']:>8.2f}%  "
          f"{row['win_rate']:>8.1f}%  "
          f"{row['sharpe']:>6.2f}  "
          f"{row['max_dd']:>6.2f}%  "
          f"{row['avg_holding_hours']:>7.1f}h")

# Comparison
print(f"\n{'=' * 80}")
print("FIXED vs EXIT MODEL COMPARISON")
print(f"{'=' * 80}")

fixed_row = df_results[df_results['strategy_type'] == 'fixed'].iloc[0]
exit_model_rows = df_results[df_results['strategy_type'] == 'exit_model']
best_exit_model = exit_model_rows.iloc[0] if len(exit_model_rows) > 0 else None

print(f"\nFixed Exit (Baseline):")
print(f"  Strategy: {fixed_row['strategy_name']}")
print(f"  Return: {fixed_row['avg_return']:+.2f}%")
print(f"  Win Rate: {fixed_row['win_rate']:.1f}%")
print(f"  Sharpe: {fixed_row['sharpe']:.2f}")
print(f"  Rank: #{df_results.index.get_loc(fixed_row.name) + 1}")

if best_exit_model is not None:
    improvement = best_exit_model['avg_return'] - fixed_row['avg_return']
    improvement_pct = (improvement / abs(fixed_row['avg_return'])) * 100

    print(f"\nBest Exit Model:")
    print(f"  Strategy: {best_exit_model['strategy_name']}")
    print(f"  Return: {best_exit_model['avg_return']:+.2f}%")
    print(f"  Win Rate: {best_exit_model['win_rate']:.1f}%")
    print(f"  Sharpe: {best_exit_model['sharpe']:.2f}")
    print(f"  Rank: #{df_results.index.get_loc(best_exit_model.name) + 1}")
    print(f"  Avg Exit Signal: {best_exit_model['avg_exit_signal']:.3f}")

    print(f"\n🎯 COMPARISON:")
    if improvement > 0:
        print(f"  Return Gain: {improvement:+.2f}% ({improvement_pct:+.1f}%) ✅")
        print(f"  From: {fixed_row['avg_return']:.2f}% → To: {best_exit_model['avg_return']:.2f}%")
        print(f"  Status: EXIT MODEL WINS! 🎉")
    else:
        print(f"  Return Difference: {improvement:+.2f}% ({improvement_pct:+.1f}%) ⚠️")
        print(f"  From: {fixed_row['avg_return']:.2f}% → To: {best_exit_model['avg_return']:.2f}%")
        print(f"  Status: Fixed Exit still better")

print(f"\n{'=' * 80}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}")
