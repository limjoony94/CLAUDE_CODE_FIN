"""
Exit Parameter Optimization

목적: Stop Loss, Take Profit, Max Holding, EXIT_THRESHOLD 최적 조합 찾기

테스트 조합:
- EXIT_THRESHOLD: [0.70, 0.75, 0.80]  # ML Exit model confidence
- STOP_LOSS: [0.01, 0.015, 0.02]  # -1%, -1.5%, -2%
- TAKE_PROFIT: [0.02, 0.03, 0.04]  # +2%, +3%, +4%
- MAX_HOLDING_HOURS: [3, 4, 6]  # hours

Total: 3 × 3 × 3 × 3 = 81 combinations

성능 지표:
- Total Return (%)
- Sharpe Ratio
- Win Rate (%)
- Avg Holding Time (hours)
- Trades/Week
- Max Drawdown (%)
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

# Fixed entry thresholds (already optimized)
LONG_ENTRY_THRESHOLD = 0.70
SHORT_ENTRY_THRESHOLD = 0.65

print("=" * 80)
print("EXIT PARAMETER OPTIMIZATION")
print("=" * 80)
print()

# Load data
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

# Load EXIT models
with open(MODELS_DIR / 'xgboost_v4_long_exit.pkl', 'rb') as f:
    long_exit_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_long_exit_scaler.pkl', 'rb') as f:
    long_exit_scaler = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_short_exit.pkl', 'rb') as f:
    short_exit_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_short_exit_scaler.pkl', 'rb') as f:
    short_exit_scaler = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt', 'r') as f:
    feature_columns = [line.strip() for line in f]

with open(MODELS_DIR / 'xgboost_v4_long_exit_features.txt', 'r') as f:
    exit_feature_columns = [line.strip() for line in f]

print(f"✅ Models loaded (Entry + Exit)")
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

# Parameter combinations to test
exit_thresholds = [0.70, 0.75, 0.80]
stop_losses = [0.01, 0.015, 0.02]  # -1%, -1.5%, -2%
take_profits = [0.02, 0.03, 0.04]  # +2%, +3%, +4%
max_holdings = [3, 4, 6]  # hours

total_combinations = len(exit_thresholds) * len(stop_losses) * len(take_profits) * len(max_holdings)
print(f"Testing {total_combinations} exit parameter combinations")
print(f"  EXIT_THRESHOLD: {exit_thresholds}")
print(f"  STOP_LOSS: {stop_losses}")
print(f"  TAKE_PROFIT: {take_profits}")
print(f"  MAX_HOLDING: {max_holdings} hours")
print()

results = []
combo_count = 0

for exit_thresh, stop_loss, take_profit, max_hold_hours in product(exit_thresholds, stop_losses, take_profits, max_holdings):
    combo_count += 1
    if combo_count % 10 == 0:
        print(f"Progress: {combo_count}/{total_combinations} combinations tested...", end='\r')

    # Generate entry signals (fixed thresholds)
    df_test['signal_long'] = (df_test['prob_long_entry'] >= LONG_ENTRY_THRESHOLD).astype(int)
    df_test['signal_short'] = (df_test['prob_short_entry'] >= SHORT_ENTRY_THRESHOLD).astype(int)

    # Simulate trading with EXIT model
    position = None
    entry_price = 0
    entry_idx = 0
    direction = None
    trades = []
    equity = 1.0  # Start with 1.0 (100%)
    equity_curve = [1.0]

    max_hold_candles = max_hold_hours * 12  # Convert hours to 5-min candles

    for i in range(len(df_test)):
        current_price = df_test.loc[i, 'close']

        # Exit logic (if in position)
        if position is not None:
            hold_time = i - entry_idx
            hours_held = hold_time / 12

            # Calculate P&L
            if direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Exit conditions
            should_exit = False
            exit_reason = None

            # 1. Stop Loss
            if pnl_pct <= -stop_loss:
                should_exit = True
                exit_reason = f"SL ({pnl_pct*100:.2f}%)"
            # 2. Take Profit
            elif pnl_pct >= take_profit:
                should_exit = True
                exit_reason = f"TP ({pnl_pct*100:.2f}%)"
            # 3. Max Holding
            elif hold_time >= max_hold_candles:
                should_exit = True
                exit_reason = f"MaxHold ({hours_held:.1f}h)"
            # 4. ML Exit Model
            else:
                # Get exit features (36 base + 8 position)
                base_features = df_test[feature_columns].iloc[i].values[:36]  # First 36 features

                # Position features (8 features)
                time_held_norm = hours_held / 1.0
                pnl_peak = max(pnl_pct, 0.0)
                pnl_trough = min(pnl_pct, 0.0)
                pnl_from_peak = pnl_pct - pnl_peak
                volatility = df_test['atr_pct'].iloc[i] if 'atr_pct' in df_test.columns else 0.01
                volume_change = 0.0  # Simplified
                momentum = 0.0  # Simplified

                position_features = np.array([
                    time_held_norm, pnl_pct, pnl_peak, pnl_trough,
                    pnl_from_peak, volatility, volume_change, momentum
                ])

                # Combined features (44 total)
                exit_features = np.concatenate([base_features, position_features]).reshape(1, -1)

                # Select model based on direction
                exit_model = long_exit_model if direction == 'LONG' else short_exit_model
                exit_scaler = long_exit_scaler if direction == 'LONG' else short_exit_scaler

                # Predict exit probability
                exit_features_scaled = exit_scaler.transform(exit_features)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                if exit_prob >= exit_thresh:
                    should_exit = True
                    exit_reason = f"ML ({exit_prob:.3f})"

            if should_exit:
                trades.append({
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'hold_time_hours': hours_held,
                    'exit_reason': exit_reason
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
        avg_holding = trades_df['hold_time_hours'].mean()

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

        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].str.split(' ', expand=True)[0].value_counts()
        ml_exit_pct = (exit_reasons.get('ML', 0) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        sl_pct = (exit_reasons.get('SL', 0) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        tp_pct = (exit_reasons.get('TP', 0) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        maxhold_pct = (exit_reasons.get('MaxHold', 0) / len(trades_df)) * 100 if len(trades_df) > 0 else 0

        results.append({
            'exit_threshold': exit_thresh,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_holding_hours': max_hold_hours,
            'total_return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'avg_holding_hours': avg_holding,
            'trades_per_week': trades_per_week,
            'max_dd': max_dd,
            'total_trades': len(trades_df),
            'ml_exit_pct': ml_exit_pct,
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'maxhold_pct': maxhold_pct
        })

print()  # Clear progress line

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by total return (descending)
results_df = results_df.sort_values('total_return', ascending=False)

print("=" * 120)
print("EXIT PARAMETER BACKTEST RESULTS (Sorted by Total Return)")
print("=" * 120)
print()
print(f"{'Rank':<5} {'Exit':<5} {'SL%':<5} {'TP%':<5} {'MaxH':<5} {'Return%':>8} {'Sharpe':>7} {'WinRate%':>9} {'AvgHold':>8} {'Trades/W':>9} {'MaxDD%':>8} {'ML%':>5} {'SL%':>5} {'TP%':>5} {'MH%':>5}")
print("-" * 120)

for idx, row in results_df.head(15).iterrows():
    rank = results_df.index.get_loc(idx) + 1
    marker = " ← BEST" if rank == 1 else ""
    print(f"{rank:<5} {row['exit_threshold']:<5.2f} {row['stop_loss']*100:<5.1f} {row['take_profit']*100:<5.1f} {row['max_holding_hours']:<5.0f} "
          f"{row['total_return']:>8.2f} {row['sharpe']:>7.2f} {row['win_rate']:>9.1f} {row['avg_holding_hours']:>8.2f} "
          f"{row['trades_per_week']:>9.1f} {row['max_dd']:>8.2f} {row['ml_exit_pct']:>5.1f} "
          f"{row['sl_pct']:>5.1f} {row['tp_pct']:>5.1f} {row['maxhold_pct']:>5.1f}{marker}")

# Save results
results_file = RESULTS_DIR / "exit_parameter_backtest_results.csv"
results_df.to_csv(results_file, index=False)
print()
print(f"✅ Results saved: {results_file}")

# Best configuration
best = results_df.iloc[0]

print()
print("=" * 80)
print("🏆 BEST EXIT PARAMETER CONFIGURATION")
print("=" * 80)
print(f"EXIT_THRESHOLD: {best['exit_threshold']:.2f}")
print(f"STOP_LOSS: {best['stop_loss']*100:.1f}% ({best['stop_loss']:.3f})")
print(f"TAKE_PROFIT: {best['take_profit']*100:.1f}% ({best['take_profit']:.3f})")
print(f"MAX_HOLDING_HOURS: {best['max_holding_hours']:.0f}")
print()
print(f"Performance:")
print(f"  Total Return: {best['total_return']:.2f}%")
print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Avg Holding: {best['avg_holding_hours']:.2f} hours")
print(f"  Trades/Week: {best['trades_per_week']:.1f}")
print(f"  Max Drawdown: {best['max_dd']:.2f}%")
print()
print(f"Exit Breakdown:")
print(f"  ML Exit: {best['ml_exit_pct']:.1f}%")
print(f"  Stop Loss: {best['sl_pct']:.1f}%")
print(f"  Take Profit: {best['tp_pct']:.1f}%")
print(f"  Max Hold: {best['maxhold_pct']:.1f}%")
print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print(f"Update phase4_dynamic_testnet_trading.py:")
print(f"  EXIT_THRESHOLD = {best['exit_threshold']:.2f}")
print(f"  STOP_LOSS = {best['stop_loss']:.3f}  # {best['stop_loss']*100:.1f}%")
print(f"  TAKE_PROFIT = {best['take_profit']:.3f}  # {best['take_profit']*100:.1f}%")
print(f"  MAX_HOLDING_HOURS = {best['max_holding_hours']:.0f}")
