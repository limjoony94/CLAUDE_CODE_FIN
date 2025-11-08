"""
Stop Loss Optimization for Opportunity Gating 4x Strategy
==========================================================

Test different stop loss values to find optimal risk management.

Strategy:
- Opportunity Gating with 4x Leverage
- Dynamic Position Sizing (20-95%)
- ML Exit + Emergency Stop Loss (variable)
- Emergency Max Hold (8 hours)

Test Range: -1% to -8% stop loss
"""

import pandas as pd
import numpy as np
import pickle
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

# Strategy Parameters (fixed)
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
ML_EXIT_THRESHOLD = 0.70
EMERGENCY_MAX_HOLD_TIME = 96  # 8 hours

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Capital & Fees
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005  # 0.05%

# Stop Loss values to test
STOP_LOSS_VALUES = [-0.01, -0.015, -0.02, -0.025, -0.03, -0.035, -0.04, -0.05, -0.06, -0.07, -0.08]

print("="*80)
print("STOP LOSS OPTIMIZATION - OPPORTUNITY GATING 4x")
print("="*80)
print(f"\nTesting {len(STOP_LOSS_VALUES)} different stop loss values\n")

# Load models
print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

# Exit Models
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
with open(long_exit_scaler_path, 'rb') as f:
    long_exit_scaler = pickle.load(f)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
with open(short_exit_scaler_path, 'rb') as f:
    short_exit_scaler = pickle.load(f)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ Models loaded\n")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Load data
print("Loading historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_full):,} candles\n")

# Calculate features
print("Calculating features...")
start_time = time.time()
df = calculate_all_features(df_full)
df = prepare_exit_features(df)
print(f"  ✅ Features calculated ({time.time() - start_time:.1f}s)\n")

# Pre-calculate signals
print("Pre-calculating signals...")
long_feat = df[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
df['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

short_feat = df[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
df['short_prob'] = short_model.predict_proba(short_feat_scaled)[:, 1]
print(f"  ✅ Signals ready\n")


def backtest_with_stop_loss(window_df, stop_loss):
    """Run backtest with specific stop loss value"""
    trades = []
    position = None
    capital = INITIAL_CAPITAL

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Entry logic
        if position is None:
            long_prob = window_df['long_prob'].iloc[i]
            short_prob = window_df['short_prob'].iloc[i]

            # LONG entry
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
                    'quantity': sizing_result['leveraged_value'] / current_price
                }

            # SHORT entry (gated)
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
                        'quantity': sizing_result['leveraged_value'] / current_price
                    }

        # Exit logic
        if position is not None:
            time_in_pos = i - position['entry_idx']
            current_price = window_df['close'].iloc[i]

            # Calculate P&L (direct notional value difference)
            entry_notional = position['quantity'] * position['entry_price']
            current_notional = position['quantity'] * current_price

            if position['side'] == 'LONG':
                pnl_usd = current_notional - entry_notional
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl_usd = entry_notional - current_notional
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Exit conditions
            should_exit = False
            exit_reason = None

            # 1. ML Exit
            try:
                if position['side'] == 'LONG':
                    exit_model = long_exit_model
                    exit_scaler = long_exit_scaler
                    exit_features_list = long_exit_feature_columns
                else:
                    exit_model = short_exit_model
                    exit_scaler = short_exit_scaler
                    exit_features_list = short_exit_feature_columns

                exit_features_values = window_df[exit_features_list].iloc[i:i+1].values
                exit_features_scaled = exit_scaler.transform(exit_features_values)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = 'ml_exit'
            except:
                pass

            # 2. Stop Loss (variable)
            if not should_exit and leveraged_pnl_pct <= stop_loss:
                should_exit = True
                exit_reason = 'stop_loss'

            # 3. Max Hold
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                # Fees
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission
                net_pnl_usd = pnl_usd - total_commission

                # Update capital
                capital += net_pnl_usd

                # Record trade
                trades.append({
                    'side': position['side'],
                    'pnl_pct': price_change_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'commission_usd': total_commission,
                    'net_pnl_usd': net_pnl_usd,
                    'hold_time': time_in_pos,
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct']
                })

                position = None

    return trades, capital


def run_optimization():
    """Test all stop loss values"""
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    all_results = []

    for stop_loss in STOP_LOSS_VALUES:
        print(f"Testing Stop Loss: {stop_loss*100:.1f}%")

        window_results = []

        for window_idx in range(num_windows):
            start_idx = window_idx * step_size
            end_idx = start_idx + window_size

            if end_idx > len(df):
                break

            window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            trades, final_capital = backtest_with_stop_loss(window_df, stop_loss)

            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                total_return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

                window_results.append({
                    'window': window_idx,
                    'total_trades': len(trades),
                    'long_trades': len(trades_df[trades_df['side'] == 'LONG']),
                    'short_trades': len(trades_df[trades_df['side'] == 'SHORT']),
                    'total_return_pct': total_return_pct,
                    'win_rate': (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades) * 100,
                    'final_capital': final_capital,
                    'stop_loss_exits': len(trades_df[trades_df['exit_reason'] == 'stop_loss']),
                    'ml_exits': len(trades_df[trades_df['exit_reason'] == 'ml_exit']),
                    'max_hold_exits': len(trades_df[trades_df['exit_reason'] == 'max_hold'])
                })

        if len(window_results) > 0:
            results_df = pd.DataFrame(window_results)

            # Calculate metrics
            avg_return = results_df['total_return_pct'].mean()
            std_return = results_df['total_return_pct'].std()
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            avg_win_rate = results_df['win_rate'].mean()
            avg_trades = results_df['total_trades'].mean()

            total_sl_exits = results_df['stop_loss_exits'].sum()
            total_ml_exits = results_df['ml_exits'].sum()
            total_mh_exits = results_df['max_hold_exits'].sum()
            total_exits = total_sl_exits + total_ml_exits + total_mh_exits

            all_results.append({
                'stop_loss_pct': stop_loss * 100,
                'avg_return_pct': avg_return,
                'std_return': std_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': avg_win_rate,
                'avg_trades': avg_trades,
                'stop_loss_exits': total_sl_exits,
                'ml_exits': total_ml_exits,
                'max_hold_exits': total_mh_exits,
                'sl_exit_pct': (total_sl_exits / total_exits * 100) if total_exits > 0 else 0,
                'windows': len(results_df)
            })

            print(f"  Return: {avg_return:.2f}% | Sharpe: {sharpe_ratio:.3f} | Win Rate: {avg_win_rate:.1f}%")
            print(f"  SL Exits: {total_sl_exits}/{total_exits} ({total_sl_exits/total_exits*100:.1f}%)\n")

    return pd.DataFrame(all_results)


# Run optimization
print("="*80)
print("RUNNING STOP LOSS OPTIMIZATION")
print("="*80)
print()

start_time = time.time()
results_df = run_optimization()
test_time = time.time() - start_time

print(f"\n{'='*80}")
print("OPTIMIZATION RESULTS")
print(f"{'='*80}\n")

# Sort by Sharpe ratio
results_sorted = results_df.sort_values('sharpe_ratio', ascending=False)

print("Top 5 Stop Loss Values (by Sharpe Ratio):")
print(results_sorted[['stop_loss_pct', 'avg_return_pct', 'sharpe_ratio', 'win_rate', 'sl_exit_pct']].head(5).to_string(index=False))

print(f"\n\nTop 5 Stop Loss Values (by Return):")
results_sorted_return = results_df.sort_values('avg_return_pct', ascending=False)
print(results_sorted_return[['stop_loss_pct', 'avg_return_pct', 'sharpe_ratio', 'win_rate', 'sl_exit_pct']].head(5).to_string(index=False))

# Best by Sharpe Ratio
best_sharpe = results_sorted.iloc[0]
print(f"\n{'='*80}")
print(f"RECOMMENDED STOP LOSS (Best Sharpe Ratio)")
print(f"{'='*80}")
print(f"Stop Loss: {best_sharpe['stop_loss_pct']:.1f}%")
print(f"Avg Return: {best_sharpe['avg_return_pct']:.2f}% per window")
print(f"Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
print(f"Win Rate: {best_sharpe['win_rate']:.1f}%")
print(f"Avg Trades: {best_sharpe['avg_trades']:.1f}")
print(f"Stop Loss Exits: {best_sharpe['sl_exit_pct']:.1f}% of all exits")
print(f"Test Time: {test_time:.1f}s")

# Save results
output_file = RESULTS_DIR / f"stop_loss_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*80}\n")
