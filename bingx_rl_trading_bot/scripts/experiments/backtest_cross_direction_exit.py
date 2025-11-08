"""
Cross-Direction Exit Strategy Backtest: Entry Models as Exit Signals
=====================================================================

TEST HYPOTHESIS:
  LONG Exit → Use SHORT Entry model (high SHORT prob = good time to exit LONG)
  SHORT Exit → Use LONG Entry model (high LONG prob = good time to exit SHORT)

Configuration:
  Entry: Walk-Forward Decoupled models (20251027_194313)
    - LONG Threshold: 0.75
    - SHORT Threshold: 0.75

  Exit: CROSS-DIRECTION (NEW APPROACH)
    - LONG Exit: SHORT Entry model (threshold 0.75)
    - SHORT Exit: LONG Entry model (threshold 0.75)

  Leverage: 4x
  Position Sizing: Dynamic (20-95%)
  Period: Full historical data (108 windows, ~540 days)
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

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("CROSS-DIRECTION EXIT STRATEGY BACKTEST")
print("="*80)
print(f"\n💡 Hypothesis: Use opposite Entry model for Exit signals\n")
print(f"  LONG Exit → SHORT Entry model (high SHORT prob = exit LONG)")
print(f"  SHORT Exit → LONG Entry model (high LONG prob = exit SHORT)\n")

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.75
SHORT_THRESHOLD = 0.75
GATE_THRESHOLD = 0.001

# Exit Parameters (CROSS-DIRECTION)
ML_EXIT_THRESHOLD_LONG = 0.75  # Use SHORT Entry threshold for LONG exit
ML_EXIT_THRESHOLD_SHORT = 0.75  # Use LONG Entry threshold for SHORT exit
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours
EMERGENCY_STOP_LOSS = -0.03  # -3% balance

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees
TAKER_FEE = 0.0005  # 0.05%

# Load Entry Models (Walk-Forward Decoupled - 2025-10-27)
print("Loading Entry models (Walk-Forward Decoupled - 20251027_194313)...")
import joblib

long_model_path = MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ Entry models loaded: LONG({len(long_feature_columns)}), SHORT({len(short_feature_columns)})")
print(f"  ℹ️  These models will ALSO be used for EXIT signals (cross-direction)\n")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Load FULL data
print("Loading FULL historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Full data loaded: {len(df_full):,} candles (~{len(df_full)//288:.0f} days)\n")

# Calculate features
print("Calculating ALL features (Enhanced v2)...")
start_time = time.time()
df = calculate_all_features_enhanced_v2(df_full, phase='phase1')
feature_time = time.time() - start_time
print(f"  ✅ All features calculated ({feature_time:.1f}s)\n")

# Pre-calculate signals (VECTORIZED)
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
    print(f"  ❌ Error pre-calculating signals: {e}")
    df['long_prob'] = 0.0
    df['short_prob'] = 0.0
    print(f"  ⚠️ Using zero probabilities\n")


def backtest_cross_direction_exit(window_df):
    """
    CROSS-DIRECTION EXIT STRATEGY:
      - LONG position: Exit when SHORT Entry model triggers (high SHORT prob)
      - SHORT position: Exit when LONG Entry model triggers (high LONG prob)

    Each window tested independently with $10,000 starting capital.
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL
    recent_trades = []
    first_exit_signal_received = False

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Check for first exit signal (using BOTH entry models as exit signals)
        if not first_exit_signal_received and position is None:
            try:
                long_prob = window_df['long_prob'].iloc[i]
                short_prob = window_df['short_prob'].iloc[i]

                # If either model shows signal, mark as ready
                if long_prob >= ML_EXIT_THRESHOLD_SHORT or short_prob >= ML_EXIT_THRESHOLD_LONG:
                    first_exit_signal_received = True
            except:
                pass

        # Entry logic (ONLY after first exit signal)
        if first_exit_signal_received and position is None:
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

        # Exit logic (CROSS-DIRECTION + Emergency)
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
            balance_loss_pct = pnl_usd / capital

            # Exit conditions
            should_exit = False
            exit_reason = None

            # 1. CROSS-DIRECTION ML Exit (PRIMARY)
            try:
                # 🔄 KEY CHANGE: Use OPPOSITE entry model for exit
                if position['side'] == 'LONG':
                    # LONG exit: Use SHORT Entry model
                    exit_prob = window_df['short_prob'].iloc[i]
                    ml_threshold = ML_EXIT_THRESHOLD_LONG

                    if exit_prob >= ml_threshold:
                        should_exit = True
                        exit_reason = 'ml_exit_long_via_short_entry'
                else:  # SHORT
                    # SHORT exit: Use LONG Entry model
                    exit_prob = window_df['long_prob'].iloc[i]
                    ml_threshold = ML_EXIT_THRESHOLD_SHORT

                    if exit_prob >= ml_threshold:
                        should_exit = True
                        exit_reason = 'ml_exit_short_via_long_entry'
            except Exception as e:
                pass

            # 2. Emergency Stop Loss
            if not should_exit and balance_loss_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'emergency_stop_loss'

            # 3. Emergency Max Hold
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'emergency_max_hold'

            if should_exit:
                # Calculate commissions
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission

                net_pnl_usd = pnl_usd - total_commission
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
                    'position_size_pct': position['position_size_pct'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price
                }

                trades.append(trade)
                recent_trades.append(trade)

                if len(recent_trades) > 10:
                    recent_trades = recent_trades[-10:]

                position = None

    return trades, capital


def run_window_backtest(df):
    """
    Run backtest across all windows
    """
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    results = []
    all_trades = []

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        trades, final_capital = backtest_cross_direction_exit(window_df)

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            all_trades.extend(trades)

            total_return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            avg_leveraged_return = trades_df['leveraged_pnl_pct'].mean() * 100

            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']

            results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'avg_leveraged_return': avg_leveraged_return,
                'total_return_pct': total_return_pct,
                'win_rate': (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades) * 100,
                'final_capital': final_capital,
                'avg_position_size': trades_df['position_size_pct'].mean() * 100
            })

    return pd.DataFrame(results), all_trades


# Run backtest
print("="*80)
print("RUNNING CROSS-DIRECTION EXIT BACKTEST")
print("="*80)
print()

start_time = time.time()
results_df, all_trades = run_window_backtest(df)
test_time = time.time() - start_time

if len(results_df) > 0:
    # Calculate metrics
    avg_window_return = results_df['total_return_pct'].mean()
    avg_trades = results_df['total_trades'].mean()
    avg_long = results_df['long_trades'].mean()
    avg_short = results_df['short_trades'].mean()
    avg_wr = results_df['win_rate'].mean()
    avg_pos_size = results_df['avg_position_size'].mean()

    print(f"\n{'='*80}")
    print("CROSS-DIRECTION EXIT RESULTS (108 Windows)")
    print(f"{'='*80}\n")

    print(f"Performance (Each window starts with $10,000):")
    print(f"  Avg Return: {avg_window_return:.2f}% per 5-day window")
    print(f"  Win Rate: {avg_wr:.1f}%")
    print(f"  Trades: {avg_trades:.1f} per window")
    print(f"    LONG: {avg_long:.1f} ({avg_long/avg_trades*100:.1f}%)")
    print(f"    SHORT: {avg_short:.1f} ({avg_short/avg_trades*100:.1f}%)")
    print(f"  Avg Position Size: {avg_pos_size:.1f}%")
    print(f"  Test Time: {test_time:.1f}s")

    # Exit Reason Analysis
    if len(all_trades) > 0:
        all_trades_df = pd.DataFrame(all_trades)

        print(f"\n🚪 EXIT REASON ANALYSIS:")
        print(f"  Total Trades: {len(all_trades)}")

        exit_counts = all_trades_df['exit_reason'].value_counts()

        for reason, count in exit_counts.items():
            pct = count / len(all_trades) * 100

            if reason == 'ml_exit_long_via_short_entry':
                reason_name = "ML Exit (LONG via SHORT Entry)"
            elif reason == 'ml_exit_short_via_long_entry':
                reason_name = "ML Exit (SHORT via LONG Entry)"
            elif reason == 'emergency_stop_loss':
                reason_name = "Emergency Stop Loss"
            elif reason == 'emergency_max_hold':
                reason_name = "Emergency Max Hold"
            else:
                reason_name = reason

            print(f"    {reason_name:<40s}: {count:>4d} ({pct:>5.1f}%)")

        # ML Exit rate (cross-direction)
        ml_exits = len(all_trades_df[all_trades_df['exit_reason'].str.contains('ml_exit')])
        ml_exit_rate = ml_exits / len(all_trades) * 100
        print(f"\n  📊 ML Exit Rate (Cross-Direction): {ml_exit_rate:.1f}% ({ml_exits}/{len(all_trades)})")

    # Calculate Sharpe ratio
    if len(results_df) > 1:
        returns = results_df['total_return_pct'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(73)  # Annualized
        print(f"\n📈 Risk Metrics:")
        print(f"  Sharpe Ratio: {sharpe:.3f} (annualized)")
        print(f"  Max Drawdown: {results_df['total_return_pct'].min():.2f}%")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"backtest_cross_direction_exit_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE")
    print(f"{'='*80}\n")

else:
    print("❌ No trades executed during backtest")
