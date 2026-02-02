"""
Position Sizing Parameter Optimization

목적: Dynamic Position Sizing 파라미터 최적 조합 찾기

고정 파라미터 (Exit 최적값):
- LONG_ENTRY_THRESHOLD: 0.70
- SHORT_ENTRY_THRESHOLD: 0.65
- EXIT_THRESHOLD: 0.70
- STOP_LOSS: 0.01 (1%)
- TAKE_PROFIT: 0.02 (2%)
- MAX_HOLDING_HOURS: 4

테스트 Position Sizing 파라미터:
- BASE_POSITION_PCT: [0.40, 0.50, 0.60]  # 40%, 50%, 60%
- MAX_POSITION_PCT: [0.90, 0.95, 1.00]  # 90%, 95%, 100%
- MIN_POSITION_PCT: [0.15, 0.20, 0.25]  # 15%, 20%, 25%

Total: 3 × 3 × 3 = 27 combinations
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

# Fixed parameters (optimized)
LONG_ENTRY_THRESHOLD = 0.70
SHORT_ENTRY_THRESHOLD = 0.65
EXIT_THRESHOLD = 0.70
STOP_LOSS = 0.01
TAKE_PROFIT = 0.02
MAX_HOLDING_HOURS = 4

print("=" * 80)
print("POSITION SIZING PARAMETER OPTIMIZATION")
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

print(f"✅ Models loaded")
print()

# Get predictions
X = df[feature_columns].values
X_long_scaled = long_scaler.transform(X)
X_short_scaled = short_scaler.transform(X)

prob_long_entry = long_model.predict_proba(X_long_scaled)[:, 1]
prob_short_entry = short_model.predict_proba(X_short_scaled)[:, 1]

df['prob_long_entry'] = prob_long_entry
df['prob_short_entry'] = prob_short_entry

# Test range
test_size = int(len(df) * 0.2)
test_start = len(df) - test_size
df_test = df.iloc[test_start:].copy().reset_index(drop=True)

weeks = len(df_test) / (12 * 24 * 7)
print(f"✅ Test set: {len(df_test)} candles ({weeks:.1f} weeks)")
print()

# Position sizing parameter combinations
base_positions = [0.40, 0.50, 0.60]
max_positions = [0.90, 0.95, 1.00]
min_positions = [0.15, 0.20, 0.25]

total_combinations = len(base_positions) * len(max_positions) * len(min_positions)
print(f"Testing {total_combinations} position sizing combinations")
print(f"  BASE_POSITION_PCT: {base_positions}")
print(f"  MAX_POSITION_PCT: {max_positions}")
print(f"  MIN_POSITION_PCT: {min_positions}")
print()

results = []
combo_count = 0

for base_pos, max_pos, min_pos in product(base_positions, max_positions, min_positions):
    combo_count += 1
    print(f"Progress: {combo_count}/{total_combinations} combinations tested...", end='\r')

    # Simple position sizing based on model confidence
    # Real implementation uses regime, volatility, streak - simplified here
    df_test['signal_long'] = (df_test['prob_long_entry'] >= LONG_ENTRY_THRESHOLD).astype(int)
    df_test['signal_short'] = (df_test['prob_short_entry'] >= SHORT_ENTRY_THRESHOLD).astype(int)

    # Simulate trading with dynamic position sizing
    position = None
    entry_price = 0
    entry_idx = 0
    direction = None
    position_size_pct = 0
    trades = []
    equity = 1.0
    equity_curve = [1.0]

    max_hold_candles = MAX_HOLDING_HOURS * 12

    for i in range(len(df_test)):
        current_price = df_test.loc[i, 'close']

        # Exit logic
        if position is not None:
            hold_time = i - entry_idx
            hours_held = hold_time / 12

            if direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Apply position sizing to P&L
            realized_pnl = pnl_pct * position_size_pct

            should_exit = False
            exit_reason = None

            if pnl_pct <= -STOP_LOSS:
                should_exit = True
                exit_reason = "SL"
            elif pnl_pct >= TAKE_PROFIT:
                should_exit = True
                exit_reason = "TP"
            elif hold_time >= max_hold_candles:
                should_exit = True
                exit_reason = "MaxHold"
            else:
                # Exit model
                base_features = df_test[feature_columns].iloc[i].values[:36]
                time_held_norm = hours_held / 1.0
                pnl_peak = max(pnl_pct, 0.0)
                pnl_trough = min(pnl_pct, 0.0)
                pnl_from_peak = pnl_pct - pnl_peak
                volatility = df_test['atr_pct'].iloc[i] if 'atr_pct' in df_test.columns else 0.01

                position_features = np.array([
                    time_held_norm, pnl_pct, pnl_peak, pnl_trough,
                    pnl_from_peak, volatility, 0.0, 0.0
                ])

                exit_features = np.concatenate([base_features, position_features]).reshape(1, -1)
                exit_model = long_exit_model if direction == 'LONG' else short_exit_model
                exit_scaler = long_exit_scaler if direction == 'LONG' else short_exit_scaler

                exit_features_scaled = exit_scaler.transform(exit_features)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                if exit_prob >= EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = "ML"

            if should_exit:
                trades.append({
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'realized_pnl': realized_pnl,
                    'position_size': position_size_pct,
                    'hold_time_hours': hours_held,
                    'exit_reason': exit_reason
                })
                equity *= (1 + realized_pnl * 0.99)  # 0.1% fee
                equity_curve.append(equity)
                position = None

        # Entry logic with position sizing
        if position is None:
            if df_test.loc[i, 'signal_long'] == 1:
                # Dynamic position sizing based on confidence
                confidence = df_test.loc[i, 'prob_long_entry']
                # Normalize confidence to [0, 1] range from threshold
                norm_confidence = (confidence - LONG_ENTRY_THRESHOLD) / (1 - LONG_ENTRY_THRESHOLD)
                # Position size = base + (max - base) * confidence
                position_size_pct = base_pos + (max_pos - base_pos) * norm_confidence
                position_size_pct = max(min_pos, min(max_pos, position_size_pct))

                position = 'LONG'
                entry_price = current_price
                entry_idx = i
                direction = 'LONG'

            elif df_test.loc[i, 'signal_short'] == 1:
                confidence = df_test.loc[i, 'prob_short_entry']
                norm_confidence = (confidence - SHORT_ENTRY_THRESHOLD) / (1 - SHORT_ENTRY_THRESHOLD)
                position_size_pct = base_pos + (max_pos - base_pos) * norm_confidence
                position_size_pct = max(min_pos, min(max_pos, position_size_pct))

                position = 'SHORT'
                entry_price = current_price
                entry_idx = i
                direction = 'SHORT'

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_return = (equity - 1.0) * 100
        win_rate = (trades_df['realized_pnl'] > 0).sum() / len(trades_df) * 100
        trades_per_week = len(trades_df) / weeks
        avg_holding = trades_df['hold_time_hours'].mean()
        avg_position_size = trades_df['position_size'].mean() * 100

        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(12 * 24 * 7) if returns.std() > 0 else 0
        else:
            sharpe = 0

        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_dd = drawdown.min()

        results.append({
            'base_position': base_pos,
            'max_position': max_pos,
            'min_position': min_pos,
            'total_return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'avg_holding_hours': avg_holding,
            'avg_position_size': avg_position_size,
            'trades_per_week': trades_per_week,
            'max_dd': max_dd,
            'total_trades': len(trades_df)
        })

print()

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_return', ascending=False)

print("=" * 110)
print("POSITION SIZING BACKTEST RESULTS (Sorted by Total Return)")
print("=" * 110)
print()
print(f"{'Rank':<5} {'Base%':<6} {'Max%':<5} {'Min%':<5} {'Return%':>8} {'Sharpe':>7} {'WinRate%':>9} {'AvgPos%':>8} {'AvgHold':>8} {'Trades/W':>9} {'MaxDD%':>8}")
print("-" * 110)

for idx, row in results_df.head(10).iterrows():
    rank = results_df.index.get_loc(idx) + 1
    marker = " ← BEST" if rank == 1 else ""
    print(f"{rank:<5} {row['base_position']*100:<6.0f} {row['max_position']*100:<5.0f} {row['min_position']*100:<5.0f} "
          f"{row['total_return']:>8.2f} {row['sharpe']:>7.2f} {row['win_rate']:>9.1f} {row['avg_position_size']:>8.1f} "
          f"{row['avg_holding_hours']:>8.2f} {row['trades_per_week']:>9.1f} {row['max_dd']:>8.2f}{marker}")

# Save results
results_file = RESULTS_DIR / "position_sizing_backtest_results.csv"
results_df.to_csv(results_file, index=False)
print()
print(f"✅ Results saved: {results_file}")

# Best configuration
best = results_df.iloc[0]

print()
print("=" * 80)
print("🏆 BEST POSITION SIZING CONFIGURATION")
print("=" * 80)
print(f"BASE_POSITION_PCT: {best['base_position']:.2f}  # {best['base_position']*100:.0f}%")
print(f"MAX_POSITION_PCT: {best['max_position']:.2f}  # {best['max_position']*100:.0f}%")
print(f"MIN_POSITION_PCT: {best['min_position']:.2f}  # {best['min_position']*100:.0f}%")
print()
print(f"Performance:")
print(f"  Total Return: {best['total_return']:.2f}%")
print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Avg Position Size: {best['avg_position_size']:.1f}%")
print(f"  Avg Holding: {best['avg_holding_hours']:.2f} hours")
print(f"  Trades/Week: {best['trades_per_week']:.1f}")
print(f"  Max Drawdown: {best['max_dd']:.2f}%")
print()
print("=" * 80)
