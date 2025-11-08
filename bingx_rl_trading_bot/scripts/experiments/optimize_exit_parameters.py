"""
Exit Parameter Optimization: Stop Loss & Max Hold Time
=======================================================

Grid search optimization for Emergency Exit parameters:
- Stop Loss: -2.0% to -6.0% (0.5% steps)
- Max Hold: 4h to 12h (2h steps)

ML Exit thresholds remain fixed:
- LONG: 0.70
- SHORT: 0.72

Configuration:
  Strategy: Opportunity Gating with 4x Leverage
  ML Exit: Fixed (0.70/0.72)
  Optimize: Emergency Stop Loss & Max Hold
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import time
from datetime import datetime
import itertools

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("EXIT PARAMETER OPTIMIZATION: STOP LOSS & MAX HOLD")
print("="*80)
print(f"\nGrid Search: SL (-2% to -6%) × MaxHold (4h to 12h)\n")

# Strategy Parameters (FIXED)
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# ML Exit Parameters (FIXED - not optimized)
ML_EXIT_THRESHOLD_LONG = 0.70
ML_EXIT_THRESHOLD_SHORT = 0.72

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees (BingX)
TAKER_FEE = 0.0005  # 0.05% per trade

# ============================================================================
# OPTIMIZATION GRID
# ============================================================================

# Stop Loss range: -2.0% to -6.0% (0.5% steps)
STOP_LOSS_RANGE = np.arange(-0.02, -0.065, -0.005)  # [-0.02, -0.025, -0.03, ..., -0.06]

# Max Hold range: 4h to 12h (2h steps)
# Convert hours to candles (1 hour = 12 candles for 5min)
MAX_HOLD_HOURS = [4, 6, 8, 10, 12]
MAX_HOLD_RANGE = [h * 12 for h in MAX_HOLD_HOURS]  # [48, 72, 96, 120, 144] candles

print(f"Optimization Grid:")
print(f"  Stop Loss: {len(STOP_LOSS_RANGE)} values from {STOP_LOSS_RANGE[0]*100:.1f}% to {STOP_LOSS_RANGE[-1]*100:.1f}%")
print(f"  Max Hold: {len(MAX_HOLD_RANGE)} values from {MAX_HOLD_HOURS[0]}h to {MAX_HOLD_HOURS[-1]}h")
print(f"  Total combinations: {len(STOP_LOSS_RANGE) * len(MAX_HOLD_RANGE)}\n")

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
    short_feature_columns = [line.strip() for line in f.readlines()]

# Load Exit Models
print("Loading Exit models...")
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

# Load FULL data
print("Loading historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_full):,} candles (~{len(df_full)//288:.0f} days)\n")

# Calculate features
print("Calculating features...")
start_time = time.time()
df = calculate_all_features(df_full)
df = prepare_exit_features(df)
feature_time = time.time() - start_time
print(f"  ✅ Features calculated ({feature_time:.1f}s)\n")

# Pre-calculate signals
print("Pre-calculating signals...")
signal_start = time.time()
try:
    # LONG signals
    long_feat = df[long_feature_columns].values
    long_feat_scaled = long_scaler.transform(long_feat)
    long_probs_array = long_model.predict_proba(long_feat_scaled)[:, 1]
    df['long_prob'] = long_probs_array

    # SHORT signals
    short_feat = df[short_feature_columns].values
    short_feat_scaled = short_scaler.transform(short_feat)
    short_probs_array = short_model.predict_proba(short_feat_scaled)[:, 1]
    df['short_prob'] = short_probs_array

    signal_time = time.time() - signal_start
    print(f"  ✅ Signals pre-calculated ({signal_time:.1f}s)\n")
except Exception as e:
    print(f"  ❌ Error: {e}")
    df['long_prob'] = 0.0
    df['short_prob'] = 0.0
    print(f"  ⚠️ Using zero probabilities\n")


def backtest_with_params(window_df, stop_loss, max_hold_time):
    """
    Run backtest with specific exit parameters

    Args:
        window_df: DataFrame with features and signals
        stop_loss: Stop loss threshold (e.g., -0.04 for -4%)
        max_hold_time: Max hold time in candles (e.g., 96 for 8h)

    Returns:
        trades: List of trade dictionaries
        final_capital: Final capital after all trades
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Entry logic (unchanged)
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

        # Exit logic (with parameterized stop loss & max hold)
        if position is not None:
            time_in_pos = i - position['entry_idx']
            current_price = window_df['close'].iloc[i]

            # Calculate P&L
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

            # 1. ML Exit Model (PRIMARY)
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

                ml_threshold = ML_EXIT_THRESHOLD_LONG if position['side'] == 'LONG' else ML_EXIT_THRESHOLD_SHORT

                if exit_prob >= ml_threshold:
                    should_exit = True
                    exit_reason = f'ml_exit_{position["side"].lower()}'
            except Exception as e:
                pass

            # 2. Emergency Stop Loss (PARAMETERIZED)
            if not should_exit and leveraged_pnl_pct <= stop_loss:
                should_exit = True
                exit_reason = 'emergency_stop_loss'

            # 3. Emergency Max Hold (PARAMETERIZED)
            if not should_exit and time_in_pos >= max_hold_time:
                should_exit = True
                exit_reason = 'emergency_max_hold'

            if should_exit:
                # Calculate commissions
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission

                # Net P&L
                net_pnl_usd = pnl_usd - total_commission

                # Update capital
                capital += net_pnl_usd

                # Record trade
                trade = {
                    'side': position['side'],
                    'pnl_pct': price_change_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'commission_usd': total_commission,
                    'net_pnl_usd': net_pnl_usd,
                    'hold_time': time_in_pos,
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct']
                }

                trades.append(trade)
                position = None

    return trades, capital


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio from returns"""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0.0

    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized


def run_optimization():
    """
    Run grid search optimization

    Test all combinations of stop loss and max hold parameters
    """
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    all_results = []
    total_combinations = len(STOP_LOSS_RANGE) * len(MAX_HOLD_RANGE)
    combo_idx = 0

    print("="*80)
    print("RUNNING GRID SEARCH OPTIMIZATION")
    print("="*80)
    print()

    # Grid search
    for stop_loss in STOP_LOSS_RANGE:
        for max_hold_candles in MAX_HOLD_RANGE:
            combo_idx += 1
            max_hold_hours = max_hold_candles / 12

            print(f"\n[{combo_idx}/{total_combinations}] Testing: SL={stop_loss*100:.1f}%, MaxHold={max_hold_hours:.0f}h")

            window_results = []

            # Test across all windows
            for window_idx in range(num_windows):
                start_idx = window_idx * step_size
                end_idx = start_idx + window_size

                if end_idx > len(df):
                    break

                window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

                # Run backtest with current parameters
                trades, final_capital = backtest_with_params(
                    window_df,
                    stop_loss=stop_loss,
                    max_hold_time=max_hold_candles
                )

                if len(trades) > 0:
                    trades_df = pd.DataFrame(trades)

                    total_return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

                    window_results.append({
                        'window': window_idx,
                        'total_trades': len(trades),
                        'total_return_pct': total_return_pct,
                        'win_rate': (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades) * 100,
                        'avg_return': trades_df['leveraged_pnl_pct'].mean() * 100,
                        'final_capital': final_capital
                    })

            if len(window_results) > 0:
                results_df = pd.DataFrame(window_results)

                # Calculate aggregate metrics
                avg_window_return = results_df['total_return_pct'].mean()
                avg_trades = results_df['total_trades'].mean()
                avg_win_rate = results_df['win_rate'].mean()

                # Calculate Sharpe ratio from window returns
                sharpe = calculate_sharpe_ratio(results_df['total_return_pct'].values / 100)

                # Store results
                result = {
                    'stop_loss_pct': stop_loss * 100,
                    'max_hold_hours': max_hold_hours,
                    'avg_window_return': avg_window_return,
                    'avg_trades_per_window': avg_trades,
                    'avg_win_rate': avg_win_rate,
                    'sharpe_ratio': sharpe,
                    'num_windows': len(results_df)
                }

                all_results.append(result)

                print(f"  Return: {avg_window_return:+.2f}% | WR: {avg_win_rate:.1f}% | Sharpe: {sharpe:.3f} | Trades: {avg_trades:.1f}")

    return pd.DataFrame(all_results)


# Run optimization
print("Starting optimization...")
start_time = time.time()

results_df = run_optimization()

optimization_time = time.time() - start_time

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*80}\n")

print(f"Optimization Time: {optimization_time/60:.1f} minutes")
print(f"Total Combinations Tested: {len(results_df)}")

# Find best parameters by different metrics
print(f"\n{'='*80}")
print("TOP 5 CONFIGURATIONS BY METRIC")
print(f"{'='*80}\n")

# Best by Return
print("🏆 TOP 5 BY RETURN:")
top_return = results_df.nlargest(5, 'avg_window_return')
for idx, row in top_return.iterrows():
    print(f"  SL={row['stop_loss_pct']:+.1f}%, MaxHold={row['max_hold_hours']:.0f}h → "
          f"Return: {row['avg_window_return']:+.2f}% | WR: {row['avg_win_rate']:.1f}% | Sharpe: {row['sharpe_ratio']:.3f}")

# Best by Sharpe
print("\n📈 TOP 5 BY SHARPE RATIO:")
top_sharpe = results_df.nlargest(5, 'sharpe_ratio')
for idx, row in top_sharpe.iterrows():
    print(f"  SL={row['stop_loss_pct']:+.1f}%, MaxHold={row['max_hold_hours']:.0f}h → "
          f"Sharpe: {row['sharpe_ratio']:.3f} | Return: {row['avg_window_return']:+.2f}% | WR: {row['avg_win_rate']:.1f}%")

# Best by Win Rate
print("\n✅ TOP 5 BY WIN RATE:")
top_wr = results_df.nlargest(5, 'avg_win_rate')
for idx, row in top_wr.iterrows():
    print(f"  SL={row['stop_loss_pct']:+.1f}%, MaxHold={row['max_hold_hours']:.0f}h → "
          f"WR: {row['avg_win_rate']:.1f}% | Return: {row['avg_window_return']:+.2f}% | Sharpe: {row['sharpe_ratio']:.3f}")

# Current baseline (-4%, 8h)
current_baseline = results_df[
    (results_df['stop_loss_pct'] == -4.0) &
    (results_df['max_hold_hours'] == 8.0)
]

if len(current_baseline) > 0:
    baseline = current_baseline.iloc[0]
    print(f"\n📊 CURRENT BASELINE (SL=-4.0%, MaxHold=8h):")
    print(f"  Return: {baseline['avg_window_return']:+.2f}%")
    print(f"  Win Rate: {baseline['avg_win_rate']:.1f}%")
    print(f"  Sharpe: {baseline['sharpe_ratio']:.3f}")
    print(f"  Trades: {baseline['avg_trades_per_window']:.1f}")

# Save results
output_file = RESULTS_DIR / f"exit_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✅ Full results saved to: {output_file}")

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*80}\n")
