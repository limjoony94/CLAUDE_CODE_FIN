"""
Exit Strategy Optimization: TP/SL/MaxHold Grid Search

목표: 최적의 Exit Strategy 조합 발견
- Stop Loss: 0.5%, 1%, 1.5%, 2%
- Take Profit: 1.5%, 2%, 3%, 4%, 6%
- Max Holding: 2h, 4h, 6h, 8h

비교 대상:
- Current: 1% SL / 3% TP / 4h MaxHold
- 총 조합: 4 × 5 × 4 = 80개
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
LEVERAGE = 4  # Current production setting

# Grid Search Parameters
STOP_LOSS_VALUES = [0.005, 0.01, 0.015, 0.02]  # 0.5%, 1%, 1.5%, 2%
TAKE_PROFIT_VALUES = [0.015, 0.02, 0.03, 0.04, 0.06]  # 1.5%, 2%, 3%, 4%, 6%
MAX_HOLDING_VALUES = [2, 4, 6, 8]  # hours

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

def backtest_with_exit_params(df, model, feature_columns, threshold,
                               stop_loss, take_profit, max_holding_hours,
                               leverage, position_size=0.95):
    """
    Backtest with specific exit strategy parameters

    Args:
        stop_loss: Stop loss percentage (e.g., 0.01 = 1%)
        take_profit: Take profit percentage (e.g., 0.03 = 3%)
        max_holding_hours: Maximum holding time in hours
        leverage: Leverage multiplier
        position_size: Position size (0.95 = 95%)
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    liquidations = 0

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12  # 5-min candles

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
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'leveraged_pnl_usd': leveraged_pnl_usd,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                })

                capital += net_pnl_usd
                position = None
                continue

            # Normal exits (using provided parameters)
            exit_reason = None
            if leveraged_pnl_pct <= -stop_loss:
                exit_reason = "Stop Loss"
            elif leveraged_pnl_pct >= take_profit:
                exit_reason = "Take Profit"
            elif hours_held >= max_holding_hours:
                exit_reason = "Max Holding"

            if exit_reason:
                entry_cost = position['leveraged_value'] * TRANSACTION_COST
                exit_cost = (current_price / entry_price) * position['leveraged_value'] * TRANSACTION_COST
                total_cost = entry_cost + exit_cost

                net_pnl_usd = leveraged_pnl_usd - total_cost

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'leveraged_pnl_usd': leveraged_pnl_usd,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                })

                capital += net_pnl_usd
                position = None

        # Entry logic
        if position is None and i < len(df) - 1:
            if capital <= 0:
                break  # Account blown

            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probability = model.predict_proba(features)[0][1]

            if probability <= threshold:
                continue

            # Fixed position sizing
            base_value = capital * position_size
            leveraged_value = base_value * leverage

            position = {
                'entry_idx': i,
                'entry_price': current_price,
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
            'sl_exits': 0,
            'tp_exits': 0,
            'maxhold_exits': 0,
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd_net'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    # Exit reason breakdown
    sl_exits = len([t for t in trades if t['exit_reason'] == 'Stop Loss'])
    tp_exits = len([t for t in trades if t['exit_reason'] == 'Take Profit'])
    maxhold_exits = len([t for t in trades if t['exit_reason'] == 'Max Holding'])

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

    # Average holding time
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
        'sl_exits': sl_exits,
        'tp_exits': tp_exits,
        'maxhold_exits': maxhold_exits,
    }

    return metrics

def rolling_window_backtest(df, model, feature_columns, threshold,
                            stop_loss, take_profit, max_holding_hours,
                            leverage, position_size=0.95):
    """Rolling window backtest with exit parameters"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        metrics = backtest_with_exit_params(
            window_df, model, feature_columns, threshold,
            stop_loss, take_profit, max_holding_hours,
            leverage, position_size
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
            'sl_exits': metrics['sl_exits'],
            'tp_exits': metrics['tp_exits'],
            'maxhold_exits': metrics['maxhold_exits'],
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


print("=" * 80)
print("EXIT STRATEGY OPTIMIZATION: Grid Search")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load model
model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
print(f"✅ Model loaded")

# Load features
feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]
print(f"✅ Features loaded: {len(feature_columns)}")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Features
print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features ready: {len(df)} rows")

threshold = 0.7
position_size = 0.95  # Fixed 95%

# Grid Search
print(f"\n{'=' * 80}")
print("GRID SEARCH PARAMETERS")
print(f"{'=' * 80}")
print(f"Stop Loss: {[f'{x*100:.1f}%' for x in STOP_LOSS_VALUES]}")
print(f"Take Profit: {[f'{x*100:.1f}%' for x in TAKE_PROFIT_VALUES]}")
print(f"Max Holding: {[f'{x}h' for x in MAX_HOLDING_VALUES]}")
print(f"Total Combinations: {len(STOP_LOSS_VALUES) * len(TAKE_PROFIT_VALUES) * len(MAX_HOLDING_VALUES)}")
print(f"Leverage: {LEVERAGE}x")

results_all = []
total_combinations = len(STOP_LOSS_VALUES) * len(TAKE_PROFIT_VALUES) * len(MAX_HOLDING_VALUES)
current_combination = 0

print(f"\n{'=' * 80}")
print("RUNNING GRID SEARCH...")
print(f"{'=' * 80}")

for sl in STOP_LOSS_VALUES:
    for tp in TAKE_PROFIT_VALUES:
        for max_hold in MAX_HOLDING_VALUES:
            current_combination += 1

            # Progress
            progress = (current_combination / total_combinations) * 100
            print(f"\n[{current_combination}/{total_combinations}] ({progress:.1f}%) Testing: SL {sl*100:.1f}% / TP {tp*100:.1f}% / MaxHold {max_hold}h", end="")

            # Skip invalid combinations (SL >= TP)
            if sl >= tp:
                print(" → SKIP (SL >= TP)")
                continue

            # Run backtest
            results = rolling_window_backtest(
                df, model, feature_columns, threshold,
                sl, tp, max_hold, LEVERAGE, position_size
            )

            # Aggregate metrics
            avg_return = results['return'].mean()
            avg_difference = results['difference'].mean()
            avg_win_rate = results['win_rate'].mean()
            avg_sharpe = results['sharpe'].mean()
            avg_max_dd = results['max_dd'].mean()
            total_liquidations = results['liquidations'].sum()
            avg_num_trades = results['num_trades'].mean()
            avg_holding_hours = results['avg_holding_hours'].mean()

            # Exit breakdown
            total_sl = results['sl_exits'].sum()
            total_tp = results['tp_exits'].sum()
            total_maxhold = results['maxhold_exits'].sum()
            total_exits = total_sl + total_tp + total_maxhold

            sl_pct = (total_sl / total_exits * 100) if total_exits > 0 else 0
            tp_pct = (total_tp / total_exits * 100) if total_exits > 0 else 0
            maxhold_pct = (total_maxhold / total_exits * 100) if total_exits > 0 else 0

            results_all.append({
                'stop_loss_pct': sl * 100,
                'take_profit_pct': tp * 100,
                'max_holding_hours': max_hold,
                'avg_return': avg_return,
                'vs_bh': avg_difference,
                'win_rate': avg_win_rate,
                'sharpe': avg_sharpe,
                'max_dd': avg_max_dd,
                'liquidations': total_liquidations,
                'avg_trades': avg_num_trades,
                'avg_holding_hours': avg_holding_hours,
                'sl_exits_pct': sl_pct,
                'tp_exits_pct': tp_pct,
                'maxhold_exits_pct': maxhold_pct,
            })

            print(f" → Return: {avg_return:+.2f}% (vs B&H: {avg_difference:+.2f}%)")

# Convert to DataFrame
df_results = pd.DataFrame(results_all)

# Sort by average return
df_results = df_results.sort_values('avg_return', ascending=False)

# Save results
output_file = RESULTS_DIR / f"exit_strategy_optimization_results.csv"
df_results.to_csv(output_file, index=False)

print(f"\n{'=' * 80}")
print("GRID SEARCH COMPLETE!")
print(f"{'=' * 80}")
print(f"Results saved: {output_file.name}")

# Top 10 configurations
print(f"\n{'=' * 80}")
print("TOP 10 EXIT STRATEGIES")
print(f"{'=' * 80}")
print(f"\n{'Rank':<5} {'SL%':<6} {'TP%':<6} {'MaxH':<6} {'Return':<10} {'vs B&H':<10} {'WinRate':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8}")
print("-" * 95)

for idx, row in df_results.head(10).iterrows():
    print(f"{df_results.index.get_loc(idx)+1:<5} "
          f"{row['stop_loss_pct']:<6.1f} "
          f"{row['take_profit_pct']:<6.1f} "
          f"{row['max_holding_hours']:<6.0f} "
          f"{row['avg_return']:>8.2f}%  "
          f"{row['vs_bh']:>8.2f}%  "
          f"{row['win_rate']:>8.1f}%  "
          f"{row['sharpe']:>6.2f}  "
          f"{row['max_dd']:>6.2f}%  "
          f"{row['avg_trades']:>6.1f}")

# Current baseline comparison
current_sl = 1.0
current_tp = 3.0
current_maxhold = 4

current_config = df_results[
    (df_results['stop_loss_pct'] == current_sl) &
    (df_results['take_profit_pct'] == current_tp) &
    (df_results['max_holding_hours'] == current_maxhold)
]

if len(current_config) > 0:
    current_return = current_config['avg_return'].values[0]
    current_rank = df_results.index.get_loc(current_config.index[0]) + 1

    best_config = df_results.iloc[0]
    best_return = best_config['avg_return']
    improvement = ((best_return - current_return) / abs(current_return)) * 100

    print(f"\n{'=' * 80}")
    print("CURRENT vs BEST CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"\nCurrent (SL {current_sl}% / TP {current_tp}% / MaxHold {current_maxhold}h):")
    print(f"  Rank: #{current_rank} / {len(df_results)}")
    print(f"  Return: {current_return:+.2f}%")
    print(f"  vs B&H: {current_config['vs_bh'].values[0]:+.2f}%")
    print(f"  Win Rate: {current_config['win_rate'].values[0]:.1f}%")
    print(f"  Sharpe: {current_config['sharpe'].values[0]:.2f}")

    print(f"\nBest (SL {best_config['stop_loss_pct']:.1f}% / TP {best_config['take_profit_pct']:.1f}% / MaxHold {best_config['max_holding_hours']:.0f}h):")
    print(f"  Rank: #1")
    print(f"  Return: {best_return:+.2f}%")
    print(f"  vs B&H: {best_config['vs_bh']:+.2f}%")
    print(f"  Win Rate: {best_config['win_rate']:.1f}%")
    print(f"  Sharpe: {best_config['sharpe']:.2f}")

    print(f"\n🎯 IMPROVEMENT POTENTIAL:")
    print(f"  Return Gain: {best_return - current_return:+.2f}% ({improvement:+.1f}%)")
    print(f"  From: {current_return:.2f}% → To: {best_return:.2f}%")

# Analysis by exit reason
print(f"\n{'=' * 80}")
print("EXIT REASON ANALYSIS (Best Config)")
print(f"{'=' * 80}")
best = df_results.iloc[0]
print(f"\nSL {best['stop_loss_pct']:.1f}% / TP {best['take_profit_pct']:.1f}% / MaxHold {best['max_holding_hours']:.0f}h:")
print(f"  Stop Loss exits: {best['sl_exits_pct']:.1f}%")
print(f"  Take Profit exits: {best['tp_exits_pct']:.1f}%")
print(f"  Max Holding exits: {best['maxhold_exits_pct']:.1f}%")
print(f"  Avg Holding Time: {best['avg_holding_hours']:.1f} hours")

print(f"\n{'=' * 80}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}")
