"""
Stop Loss Optimization: 30-Day Grid Search
===========================================

Tests multiple Stop Loss levels to find optimal setting:
- Tests SL range: -3% to -10% (in 1% increments)
- Evaluates: Return, Win Rate, Max Drawdown, Sharpe, SL Trigger Rate
- Uses 30-day historical data for robustness

Goal: Find SL level that maximizes risk-adjusted returns
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("STOP LOSS OPTIMIZATION - 30 DAY TEST")
print("="*80)
print(f"\nFinding optimal SL level through grid search\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategy Parameters (FIXED)
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters (FIXED except SL)
ML_EXIT_THRESHOLD_LONG = 0.70
ML_EXIT_THRESHOLD_SHORT = 0.72
EMERGENCY_MAX_HOLD_TIME = 96  # 8 hours

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Backtest Parameters
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005

# SL GRID SEARCH RANGE
SL_LEVELS = [-0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10]

print(f"Testing {len(SL_LEVELS)} Stop Loss levels:")
for sl in SL_LEVELS:
    print(f"  - {sl*100:.0f}% balance loss")
print()

# =============================================================================
# LOAD MODELS
# =============================================================================

print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ Models loaded\n")

# Initialize Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("Loading and preparing data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)

THIRTY_DAYS_CANDLES = 30 * 288
BUFFER_CANDLES = 200
TOTAL_CANDLES_NEEDED = THIRTY_DAYS_CANDLES + BUFFER_CANDLES

df_recent = df_full.tail(TOTAL_CANDLES_NEEDED).reset_index(drop=True)

start_time = time.time()
df = calculate_all_features(df_recent)
df = prepare_exit_features(df)
df = df.dropna().reset_index(drop=True)
df_test = df.tail(THIRTY_DAYS_CANDLES).reset_index(drop=True)

# Pre-calculate signals
long_feat = df_test[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
long_probs_array = long_model.predict_proba(long_feat_scaled)[:, 1]
df_test['long_prob'] = long_probs_array

short_feat = df_test[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
short_probs_array = short_model.predict_proba(short_feat_scaled)[:, 1]
df_test['short_prob'] = short_probs_array

prep_time = time.time() - start_time
print(f"  ✅ Data prepared ({prep_time:.1f}s)")
print(f"     Test period: {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}\n")


# =============================================================================
# BACKTEST FUNCTION WITH VARIABLE SL
# =============================================================================

def backtest_with_sl(test_df, stop_loss_pct):
    """
    Backtest with specified Stop Loss level
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL

    # Track capital curve for drawdown
    capital_curve = []

    for i in range(len(test_df) - 1):
        current_price = test_df['close'].iloc[i]

        # Entry logic
        if position is None:
            long_prob = test_df['long_prob'].iloc[i]
            short_prob = test_df['short_prob'].iloc[i]

            if long_prob >= LONG_THRESHOLD:
                sizing_result = sizer.get_position_size_simple(
                    capital=capital,
                    signal_strength=long_prob,
                    leverage=LEVERAGE
                )

                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_price': current_price,
                    'position_size_pct': sizing_result['position_size_pct'],
                    'position_value': sizing_result['position_value'],
                    'leveraged_value': sizing_result['leveraged_value'],
                    'quantity': sizing_result['leveraged_value'] / current_price,
                    'entry_prob': long_prob
                }

            elif short_prob >= SHORT_THRESHOLD:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN

                if (short_ev - long_ev) > GATE_THRESHOLD:
                    sizing_result = sizer.get_position_size_simple(
                        capital=capital,
                        signal_strength=short_prob,
                        leverage=LEVERAGE
                    )

                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': current_price,
                        'position_size_pct': sizing_result['position_size_pct'],
                        'position_value': sizing_result['position_value'],
                        'leveraged_value': sizing_result['leveraged_value'],
                        'quantity': sizing_result['leveraged_value'] / current_price,
                        'entry_prob': short_prob
                    }

        # Exit logic
        if position is not None:
            time_in_pos = i - position['entry_idx']
            current_price = test_df['close'].iloc[i]

            # Calculate P&L
            entry_notional = position['quantity'] * position['entry_price']
            current_notional = position['quantity'] * current_price

            if position['side'] == 'LONG':
                pnl_usd = current_notional - entry_notional
            else:
                pnl_usd = entry_notional - current_notional

            balance_loss_pct = pnl_usd / capital

            # Exit conditions
            should_exit = False
            exit_reason = None

            # 1. ML Exit (PRIMARY)
            try:
                if position['side'] == 'LONG':
                    exit_model = long_exit_model
                    exit_scaler = long_exit_scaler
                    exit_features_list = long_exit_feature_columns
                    ml_threshold = ML_EXIT_THRESHOLD_LONG
                else:
                    exit_model = short_exit_model
                    exit_scaler = short_exit_scaler
                    exit_features_list = short_exit_feature_columns
                    ml_threshold = ML_EXIT_THRESHOLD_SHORT

                exit_features_values = test_df[exit_features_list].iloc[i:i+1].values
                exit_features_scaled = exit_scaler.transform(exit_features_values)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                if exit_prob >= ml_threshold:
                    should_exit = True
                    exit_reason = f'ml_exit'
            except:
                pass

            # 2. Emergency Stop Loss (VARIABLE)
            if not should_exit and balance_loss_pct <= stop_loss_pct:
                should_exit = True
                exit_reason = 'emergency_stop_loss'

            # 3. Emergency Max Hold
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'emergency_max_hold'

            if should_exit:
                # Calculate fees
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission

                # Net P&L
                pnl_net = pnl_usd - total_commission
                capital += pnl_net

                # Record trade
                trades.append({
                    'pnl_net': pnl_net,
                    'exit_reason': exit_reason
                })

                position = None

        # Track capital
        capital_curve.append(capital)

    # Close remaining position
    if position is not None:
        current_price = test_df['close'].iloc[-1]
        entry_notional = position['quantity'] * position['entry_price']
        current_notional = position['quantity'] * current_price

        if position['side'] == 'LONG':
            pnl_usd = current_notional - entry_notional
        else:
            pnl_usd = entry_notional - current_notional

        entry_commission = position['leveraged_value'] * TAKER_FEE
        exit_commission = position['quantity'] * current_price * TAKER_FEE
        pnl_net = pnl_usd - entry_commission - exit_commission

        capital += pnl_net
        trades.append({'pnl_net': pnl_net, 'exit_reason': 'end_of_test'})

    # Calculate metrics
    return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    winners = [t for t in trades if t['pnl_net'] > 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0

    sl_trades = [t for t in trades if t['exit_reason'] == 'emergency_stop_loss']
    sl_rate = len(sl_trades) / len(trades) * 100 if trades else 0

    # Max drawdown
    capital_curve = np.array(capital_curve)
    running_max = np.maximum.accumulate(capital_curve)
    drawdown = (capital_curve - running_max) / running_max * 100
    max_drawdown = np.min(drawdown)

    # Sharpe (using daily returns)
    daily_returns = []
    candles_per_day = 288
    for day in range(1, len(capital_curve) // candles_per_day + 1):
        day_end = day * candles_per_day
        day_start = (day - 1) * candles_per_day
        if day_end < len(capital_curve):
            day_return = ((capital_curve[day_end] - capital_curve[day_start]) / capital_curve[day_start]) * 100
            daily_returns.append(day_return)

    if len(daily_returns) > 1:
        avg_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe = avg_daily / std_daily if std_daily > 0 else 0
    else:
        sharpe = 0

    return {
        'return_pct': return_pct,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'sl_rate': sl_rate,
        'total_trades': len(trades),
        'sl_count': len(sl_trades)
    }


# =============================================================================
# GRID SEARCH
# =============================================================================

print("="*80)
print("RUNNING STOP LOSS OPTIMIZATION")
print("="*80)
print()

results = []
optimization_start = time.time()

for sl in SL_LEVELS:
    sl_pct = sl * 100
    print(f"Testing SL = {sl_pct:.0f}%...", end=" ")

    test_start = time.time()
    result = backtest_with_sl(df_test, sl)
    test_time = time.time() - test_start

    result['sl_level'] = sl
    results.append(result)

    print(f"Return: {result['return_pct']:+.2f}%, WR: {result['win_rate']:.1f}%, "
          f"MDD: {result['max_drawdown']:.1f}%, Sharpe: {result['sharpe']:.3f}, "
          f"SL Rate: {result['sl_rate']:.1f}% ({test_time:.1f}s)")

optimization_time = time.time() - optimization_start
print(f"\n  ✅ Optimization complete ({optimization_time:.1f}s)\n")

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

print("="*80)
print("STOP LOSS OPTIMIZATION RESULTS")
print("="*80)

results_df = pd.DataFrame(results)

# Sort by different metrics
by_return = results_df.sort_values('return_pct', ascending=False)
by_sharpe = results_df.sort_values('sharpe', ascending=False)
by_winrate = results_df.sort_values('win_rate', ascending=False)
by_drawdown = results_df.sort_values('max_drawdown', ascending=False)

print(f"\n🏆 Top 3 by Total Return:")
for idx, row in by_return.head(3).iterrows():
    print(f"   {row['sl_level']*100:.0f}%: Return={row['return_pct']:+.2f}%, "
          f"WR={row['win_rate']:.1f}%, MDD={row['max_drawdown']:.1f}%, "
          f"Sharpe={row['sharpe']:.3f}, SL Rate={row['sl_rate']:.1f}%")

print(f"\n📊 Top 3 by Sharpe Ratio:")
for idx, row in by_sharpe.head(3).iterrows():
    print(f"   {row['sl_level']*100:.0f}%: Sharpe={row['sharpe']:.3f}, "
          f"Return={row['return_pct']:+.2f}%, WR={row['win_rate']:.1f}%, "
          f"MDD={row['max_drawdown']:.1f}%, SL Rate={row['sl_rate']:.1f}%")

print(f"\n🎯 Top 3 by Win Rate:")
for idx, row in by_winrate.head(3).iterrows():
    print(f"   {row['sl_level']*100:.0f}%: WR={row['win_rate']:.1f}%, "
          f"Return={row['return_pct']:+.2f}%, MDD={row['max_drawdown']:.1f}%, "
          f"Sharpe={row['sharpe']:.3f}, SL Rate={row['sl_rate']:.1f}%")

print(f"\n🛡️ Top 3 by Max Drawdown (best = least negative):")
for idx, row in by_drawdown.head(3).iterrows():
    print(f"   {row['sl_level']*100:.0f}%: MDD={row['max_drawdown']:.1f}%, "
          f"Return={row['return_pct']:+.2f}%, WR={row['win_rate']:.1f}%, "
          f"Sharpe={row['sharpe']:.3f}, SL Rate={row['sl_rate']:.1f}%")

# Composite score (weighted)
print(f"\n🎖️ Composite Score (weighted ranking):")
print(f"   Formula: 0.4×Return + 0.3×Sharpe + 0.2×WinRate - 0.1×|MaxDrawdown|")
print()

results_df['composite_score'] = (
    0.4 * (results_df['return_pct'] / results_df['return_pct'].max()) +
    0.3 * (results_df['sharpe'] / results_df['sharpe'].max()) +
    0.2 * (results_df['win_rate'] / results_df['win_rate'].max()) -
    0.1 * (abs(results_df['max_drawdown']) / abs(results_df['max_drawdown']).max())
)

by_composite = results_df.sort_values('composite_score', ascending=False)

for idx, row in by_composite.head(3).iterrows():
    print(f"   {row['sl_level']*100:.0f}%: Score={row['composite_score']:.3f}, "
          f"Return={row['return_pct']:+.2f}%, Sharpe={row['sharpe']:.3f}, "
          f"WR={row['win_rate']:.1f}%, MDD={row['max_drawdown']:.1f}%")

# Recommended SL
best_sl = by_composite.iloc[0]
print(f"\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\n✅ Optimal Stop Loss: {best_sl['sl_level']*100:.0f}% balance loss")
print(f"\n   Expected Performance:")
print(f"   - Return: {best_sl['return_pct']:+.2f}% (30 days)")
print(f"   - Win Rate: {best_sl['win_rate']:.1f}%")
print(f"   - Max Drawdown: {best_sl['max_drawdown']:.1f}%")
print(f"   - Sharpe Ratio: {best_sl['sharpe']:.3f}")
print(f"   - SL Trigger Rate: {best_sl['sl_rate']:.1f}%")
print(f"   - Total Trades: {best_sl['total_trades']}")
print(f"   - SL Triggers: {best_sl['sl_count']}")

# Current vs Optimal comparison
current_sl = -0.06
current_result = results_df[results_df['sl_level'] == current_sl].iloc[0]

print(f"\n📊 Current (-6%) vs Optimal ({best_sl['sl_level']*100:.0f}%):")
print(f"   Return: {current_result['return_pct']:+.2f}% → {best_sl['return_pct']:+.2f}% "
      f"({best_sl['return_pct'] - current_result['return_pct']:+.2f}%)")
print(f"   Win Rate: {current_result['win_rate']:.1f}% → {best_sl['win_rate']:.1f}% "
      f"({best_sl['win_rate'] - current_result['win_rate']:+.1f}%)")
print(f"   Max Drawdown: {current_result['max_drawdown']:.1f}% → {best_sl['max_drawdown']:.1f}% "
      f"({best_sl['max_drawdown'] - current_result['max_drawdown']:+.1f}%)")
print(f"   Sharpe: {current_result['sharpe']:.3f} → {best_sl['sharpe']:.3f} "
      f"({best_sl['sharpe'] - current_result['sharpe']:+.3f})")

print(f"\n" + "="*80)
print(f"Total Execution Time: {(time.time() - start_time):.1f}s")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# Save results
output_path = RESULTS_DIR / f"sl_optimization_30days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_path, index=False)
print(f"💾 Results saved to: {output_path}\n")
