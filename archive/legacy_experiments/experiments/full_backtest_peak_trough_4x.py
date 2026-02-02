"""
Full Period Backtest: Opportunity Gating Strategy with 4x Leverage
===================================================================

Test Opportunity Gating with:
- 4x Leverage (matching Phase 4 system)
- Dynamic Position Sizing (20-95%)
- Full historical data (Aug-Oct 2025)

Configuration:
  LONG Threshold: 0.80
  SHORT Threshold: 0.80
  Gate Threshold: 0.001
  Leverage: 4x
  Position Sizing: Dynamic (20-95%)
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
print("FULL PERIOD BACKTEST: OPPORTUNITY GATING with 4x LEVERAGE")
print("="*80)
print(f"\nTesting with LEVERAGE and DYNAMIC SIZING\n")

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.75  # ROLLBACK 2025-10-28 - Back to proven 0.75 (matches 20251024 models)
SHORT_THRESHOLD = 0.75  # ROLLBACK 2025-10-28 - Back to proven 0.75 (matches 20251024 models)
GATE_THRESHOLD = 0.001

# Exit Parameters (ML Exit + Emergency Safety Nets)
# ROLLBACK 2025-10-28: Exit 0.75 threshold (0.80 models failed with 6% ML Exit rate)
# Using proven 0.75 models (20251024_043527/044510) with matching 0.75 thresholds
ML_EXIT_THRESHOLD_LONG = 0.75  # ROLLBACK 2025-10-28 - Back to proven 0.75 (matches 20251024 models)
ML_EXIT_THRESHOLD_SHORT = 0.75  # ROLLBACK 2025-10-28 - Back to proven 0.75 (matches 20251024 models)
EMERGENCY_MAX_HOLD_TIME = 120  # UPDATED 2025-10-27 - 10 hours max hold
EMERGENCY_STOP_LOSS = 0.03  # UPDATED 2025-10-27 - Balance-based SL (3% of balance)

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees (BingX)
TAKER_FEE = 0.0005  # 0.05% per trade (market orders)

# Load Entry Models (NEW - trained with 5-fold CV - 2025-10-24)
print("Loading Entry models (Enhanced with 5-fold CV - 20251024_012445)...")
import joblib

long_model_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

# Load Exit Models (PRODUCTION PIPELINE - 2025-10-30)
# ✅ UPDATED: Trained with production feature pipeline (20251030_230849)
# Expected: ML Exit usage ~92.7% (vs previous 1.6% with mismatched features)
print("Loading Exit models (Production Pipeline - 20251030_230849)...")
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_peak_trough_prodpipe_20251030_230849.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_peak_trough_prodpipe_20251030_230849_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_peak_trough_prodpipe_20251030_230849_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_peak_trough_prodpipe_20251030_230849.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_peak_trough_prodpipe_20251030_230849_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_peak_trough_prodpipe_20251030_230849_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ Entry models loaded: LONG({len(long_feature_columns)}), SHORT({len(short_feature_columns)})")
print(f"  ✅ Exit models loaded: LONG({len(long_exit_feature_columns)}), SHORT({len(short_exit_feature_columns)})\n")

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

# Calculate features (using enhanced_v2 to match training)
print("Calculating ALL features (Enhanced v2 + EXIT)...")
start_time = time.time()
df = calculate_all_features_enhanced_v2(df_full, phase='phase1')  # Match training phase!
df = prepare_exit_features(df)  # Add EXIT-specific features

# FIX: Rename features to match Exit model expectations
# Exit models were trained with different feature names
feature_rename_map = {
    'macd_diff': 'macd_histogram',  # MACD histogram naming (actual: macd_diff)
    'bb_high': 'bb_upper',          # Bollinger Band upper
    'bb_low': 'bb_lower',           # Bollinger Band lower
    'bb_mid': 'bb_middle'           # Bollinger Band middle
}
# Only rename if column exists
for old_name, new_name in feature_rename_map.items():
    if old_name in df.columns and new_name not in df.columns:
        df[new_name] = df[old_name]
        print(f"  📝 Renamed: {old_name} → {new_name}")

feature_time = time.time() - start_time
print(f"  ✅ All features calculated ({feature_time:.1f}s)\n")

# Pre-calculate signals (VECTORIZED - much faster!)
print("Pre-calculating signals...")
signal_start = time.time()
try:
    # LONG signals (vectorized)
    long_feat = df[long_feature_columns].values
    long_feat_scaled = long_scaler.transform(long_feat)
    long_probs_array = long_model.predict_proba(long_feat_scaled)[:, 1]
    df['long_prob'] = long_probs_array

    # SHORT signals (vectorized)
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


def backtest_opportunity_gating_4x(window_df):
    """
    Opportunity Gating Strategy with 4x Leverage + Dynamic Sizing (Entry After First Exit)

    NEW BEHAVIOR: Entry signals only accepted AFTER first exit signal occurs.
    This simulates real bot startup where exit model must be ready before trading.

    Each window tested independently with $10,000 starting capital
    for consistent performance measurement across all windows.
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL  # Each window starts with $10,000
    recent_trades = []
    first_exit_signal_received = False  # NEW: Track first exit signal

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # NEW: Check for exit signal even without position (to detect first exit signal)
        if not first_exit_signal_received and position is None:
            try:
                # Check LONG exit signal
                exit_features_values = window_df[long_exit_feature_columns].iloc[i:i+1].values
                exit_features_scaled = long_exit_scaler.transform(exit_features_values)
                long_exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]

                # Check SHORT exit signal
                exit_features_values = window_df[short_exit_feature_columns].iloc[i:i+1].values
                exit_features_scaled = short_exit_scaler.transform(exit_features_values)
                short_exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]

                # If either model shows exit signal, mark as ready
                if long_exit_prob >= ML_EXIT_THRESHOLD_LONG or short_exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                    first_exit_signal_received = True
                # Log at first candle of window (for every window)
                if i == 0:
                    print(f"  [Candle 0] Exit signals: LONG={long_exit_prob:.4f}, SHORT={short_exit_prob:.4f}, Detected={first_exit_signal_received}")
            except Exception as e:
                if i == 0:  # Log error only for first candle to avoid spam
                    print(f"  [Candle 0] ❌ Exit signal error: {e}")
                    missing = set(long_exit_feature_columns) - set(window_df.columns)
                    if missing:
                        print(f"  [Candle 0] Missing features: {missing}")
                pass

        # Entry logic (ONLY after first exit signal)
        if first_exit_signal_received and position is None:
            long_prob = window_df['long_prob'].iloc[i]
            short_prob = window_df['short_prob'].iloc[i]

            # LONG entry
            if long_prob >= LONG_THRESHOLD:
                # Dynamic position sizing
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
                # Calculate expected values
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN

                # Gate check
                if (short_ev - long_ev) > GATE_THRESHOLD:
                    # Dynamic position sizing
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

        # Exit logic (ML Exit + Emergency Safety Nets)
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

            # Calculate leveraged P&L % (for display/logging)
            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Calculate balance-based loss percentage for Stop Loss check
            balance_loss_pct = pnl_usd / capital

            # Exit conditions
            should_exit = False
            exit_reason = None

            # 1. ML Exit Model (PRIMARY)
            try:
                # Select appropriate exit model
                if position['side'] == 'LONG':
                    exit_model = long_exit_model
                    exit_scaler = long_exit_scaler
                    exit_features_list = long_exit_feature_columns
                else:  # SHORT
                    exit_model = short_exit_model
                    exit_scaler = short_exit_scaler
                    exit_features_list = short_exit_feature_columns

                # Get exit features from current candle
                exit_features_values = window_df[exit_features_list].iloc[i:i+1].values
                exit_features_scaled = exit_scaler.transform(exit_features_values)

                # Get exit probability
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                # Use appropriate threshold based on position side
                ml_threshold = ML_EXIT_THRESHOLD_LONG if position['side'] == 'LONG' else ML_EXIT_THRESHOLD_SHORT

                # Exit if probability exceeds threshold
                if exit_prob >= ml_threshold:
                    should_exit = True
                    exit_reason = f'ml_exit_{position["side"].lower()}'
            except Exception as e:
                # If ML Exit fails, continue to emergency exits
                pass

            # 2. Emergency Stop Loss (Balance-Based: -4% of total capital)
            if not should_exit and balance_loss_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'emergency_stop_loss'

            # 3. Emergency Max Hold (8 hours)
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'emergency_max_hold'

            if should_exit:
                # Calculate commissions (BingX Taker Fee: 0.05% per trade)
                entry_commission = position['leveraged_value'] * TAKER_FEE
                # Exit commission uses actual exit-time notional value (quantity × exit price)
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission

                # Net P&L after commissions
                net_pnl_usd = pnl_usd - total_commission

                # Update capital (P&L minus commissions)
                capital += net_pnl_usd

                # Record trade
                trade = {
                    'side': position['side'],
                    'pnl_pct': price_change_pct,  # Unleveraged
                    'leveraged_pnl_pct': leveraged_pnl_pct,  # With leverage
                    'pnl_usd': pnl_usd,
                    'commission_usd': total_commission,  # Track commission
                    'net_pnl_usd': net_pnl_usd,  # Net after commission
                    'hold_time': time_in_pos,
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price
                }

                trades.append(trade)
                recent_trades.append(trade)

                # Keep only last 10 trades
                if len(recent_trades) > 10:
                    recent_trades = recent_trades[-10:]

                position = None

    return trades, capital


def run_window_backtest(df):
    """
    Run backtest across all windows

    Each window tested independently with $10,000 starting capital.
    This provides consistent performance measurement across different market conditions.
    """
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size
    # DEBUG: Test only first 5 windows
    num_windows = min(num_windows, 5)

    results = []
    all_trades = []  # Collect ALL trades for exit reason analysis

    for window_idx in range(num_windows):
        print(f"\n[Window {window_idx+1}/{num_windows}]")
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Run strategy (each window starts with $10,000)
        trades, final_capital = backtest_opportunity_gating_4x(window_df)

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)

            # Add to all_trades collection
            all_trades.extend(trades)

            # Calculate metrics
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
print("RUNNING BACKTEST with 4x LEVERAGE + DYNAMIC SIZING")
print("="*80)
print()

start_time = time.time()
results_df, all_trades = run_window_backtest(df)
test_time = time.time() - start_time

if len(results_df) > 0:
    # Calculate performance metrics (independent window testing)
    avg_window_return = results_df['total_return_pct'].mean()
    avg_trades = results_df['total_trades'].mean()
    avg_long = results_df['long_trades'].mean()
    avg_short = results_df['short_trades'].mean()
    avg_wr = results_df['win_rate'].mean()
    avg_pos_size = results_df['avg_position_size'].mean()

    # Last window result (for reference only)
    last_window_capital = results_df['final_capital'].iloc[-1]

    print(f"\n{'='*80}")
    print("FULL PERIOD BACKTEST RESULTS (4x LEVERAGE) - INDEPENDENT WINDOW TESTING")
    print(f"{'='*80}\n")

    print(f"Performance (Each window starts with $10,000):")
    print(f"  Avg Window Return: {avg_window_return:.2f}% per window")
    print(f"  Win Rate: {avg_wr:.1f}%")
    print(f"  Trades: {avg_trades:.1f} per window (LONG {avg_long:.1f} + SHORT {avg_short:.1f})")
    print(f"  Avg Position Size: {avg_pos_size:.1f}%")
    print(f"  Windows: {len(results_df)}")
    print(f"  Test Time: {test_time:.1f}s")

    print(f"\n📊 Capital Summary:")
    print(f"  Starting Capital (per window): ${INITIAL_CAPITAL:,.2f}")
    print(f"  Avg Final Capital (per window): ${results_df['final_capital'].mean():,.2f}")
    print(f"  Last Window Result: ${last_window_capital:,.2f} ({(last_window_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:+.2f}%)")

    # Annualized calculations from average window return
    days_per_window = 5
    windows_per_year = 365 / days_per_window
    periods_elapsed = len(results_df)

    # Annualized return from average window performance
    # Formula: (1 + avg_return)^windows_per_year - 1
    annualized_return = (1 + avg_window_return / 100) ** windows_per_year - 1

    print(f"\n📈 Annualized Performance (from avg window return):")
    print(f"  Annualized Return: {annualized_return * 100:.1f}%")
    print(f"  (Based on {periods_elapsed} windows over {periods_elapsed * days_per_window} days)")

    # Transaction cost analysis
    transaction_cost_pct = 0.05
    slippage_pct = 0.02
    total_cost_per_trade = transaction_cost_pct + slippage_pct

    cost_per_window = avg_trades * (total_cost_per_trade / 100) * 100
    net_window_return = avg_window_return - cost_per_window

    print(f"\n💰 After Transaction Costs:")
    print(f"  Cost per trade: {total_cost_per_trade}%")
    print(f"  Cost per window: {cost_per_window:.2f}%")
    print(f"  Net window return: {net_window_return:.2f}% per window")

    # Comparison with no-leverage
    no_leverage_return = avg_window_return / LEVERAGE
    print(f"\n📈 Leverage Impact:")
    print(f"  With 4x Leverage: {avg_window_return:.2f}% per window")
    print(f"  Without Leverage (est): {no_leverage_return:.2f}% per window")
    print(f"  Leverage Multiplier: {avg_window_return / no_leverage_return:.2f}x")

    # Exit Reason Analysis
    if len(all_trades) > 0:
        all_trades_df = pd.DataFrame(all_trades)

        print(f"\n🚪 EXIT REASON ANALYSIS:")
        print(f"  Total Trades: {len(all_trades)}")

        # Count by exit reason
        exit_counts = all_trades_df['exit_reason'].value_counts()

        for reason, count in exit_counts.items():
            pct = count / len(all_trades) * 100

            # Format reason name
            if reason == 'ml_exit_long':
                reason_name = "ML Exit (LONG)"
            elif reason == 'ml_exit_short':
                reason_name = "ML Exit (SHORT)"
            elif reason == 'emergency_stop_loss':
                reason_name = "Emergency Stop Loss"
            elif reason == 'emergency_max_hold':
                reason_name = "Emergency Max Hold"
            else:
                reason_name = reason

            print(f"    {reason_name:<25s}: {count:>4d} ({pct:>5.1f}%)")

        # Calculate by side
        print(f"\n  Exit by Side:")
        long_exits = all_trades_df[all_trades_df['side'] == 'LONG']
        short_exits = all_trades_df[all_trades_df['side'] == 'SHORT']

        if len(long_exits) > 0:
            print(f"    LONG Exits ({len(long_exits)} trades):")
            long_exit_counts = long_exits['exit_reason'].value_counts()
            for reason, count in long_exit_counts.items():
                pct = count / len(long_exits) * 100
                reason_name = "ML Exit" if reason == 'ml_exit_long' else \
                             "Stop Loss" if reason == 'emergency_stop_loss' else \
                             "Max Hold" if reason == 'emergency_max_hold' else reason
                print(f"      {reason_name:<20s}: {count:>4d} ({pct:>5.1f}%)")

        if len(short_exits) > 0:
            print(f"    SHORT Exits ({len(short_exits)} trades):")
            short_exit_counts = short_exits['exit_reason'].value_counts()
            for reason, count in short_exit_counts.items():
                pct = count / len(short_exits) * 100
                reason_name = "ML Exit" if reason == 'ml_exit_short' else \
                             "Stop Loss" if reason == 'emergency_stop_loss' else \
                             "Max Hold" if reason == 'emergency_max_hold' else reason
                print(f"      {reason_name:<20s}: {count:>4d} ({pct:>5.1f}%)")

    # Save results
    output_file = RESULTS_DIR / f"full_backtest_opportunity_gating_4x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE")
    print(f"{'='*80}\n")

else:
    print("❌ No trades executed during backtest")
