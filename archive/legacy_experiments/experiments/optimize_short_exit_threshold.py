"""
Optimize SHORT Exit Threshold
==============================

Test different SHORT exit thresholds to find optimal value that:
1. Minimizes opportunity cost
2. Maintains win rate > 60%
3. Maximizes total return

Author: Claude Code
Date: 2025-10-18
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

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
ML_EXIT_THRESHOLD_LONG = 0.70  # Keep LONG as is
EMERGENCY_MAX_HOLD_TIME = 96  # 8 hours
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)
EMERGENCY_MAX_HOLD = 96
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005

print("="*80)
print("SHORT EXIT THRESHOLD OPTIMIZATION")
print("="*80)

# Load models
print("\nLoading models...")
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
    long_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
    long_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl", 'rb') as f:
    short_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl", 'rb') as f:
    short_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl", 'rb') as f:
    long_exit_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl", 'rb') as f:
    short_exit_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print("✅ Models loaded")

# Initialize sizer
sizer = DynamicPositionSizer(base_position_pct=0.50, max_position_pct=0.95, min_position_pct=0.20)

# Load and prepare data
print("\nLoading data...")
df_full = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"Calculating features...")
df = calculate_all_features(df_full)
df = prepare_exit_features(df)

# Pre-calculate signals
print("Pre-calculating signals...")
long_feat = df[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
df['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

short_feat = df[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
df['short_prob'] = short_model.predict_proba(short_feat_scaled)[:, 1]

print("✅ Data prepared\n")


def backtest_with_threshold(window_df, short_exit_threshold):
    """
    Backtest with specific SHORT exit threshold

    Args:
        window_df: DataFrame with features
        short_exit_threshold: Exit threshold for SHORT (0.58 ~ 0.70)

    Returns:
        dict with metrics
    """
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
                    capital=capital, signal_strength=long_prob, leverage=LEVERAGE
                )
                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_price': current_price,
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
                        capital=capital, signal_strength=short_prob, leverage=LEVERAGE
                    )
                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': current_price,
                        'position_value': sizing_result['position_value'],
                        'leveraged_value': sizing_result['leveraged_value'],
                        'quantity': sizing_result['leveraged_value'] / current_price
                    }

        # Exit logic
        if position is not None:
            time_in_pos = i - position['entry_idx']

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

            # ML Exit (use different threshold for LONG/SHORT)
            try:
                if position['side'] == 'LONG':
                    exit_features_values = window_df[long_exit_feature_columns].iloc[i:i+1].values
                    exit_features_scaled = long_exit_scaler.transform(exit_features_values)
                    exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                        should_exit = True
                        exit_reason = 'ml_exit_long'
                else:  # SHORT
                    exit_features_values = window_df[short_exit_feature_columns].iloc[i:i+1].values
                    exit_features_scaled = short_exit_scaler.transform(exit_features_values)
                    exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]

                    # Use optimized threshold for SHORT
                    if exit_prob >= short_exit_threshold:
                        should_exit = True
                        exit_reason = 'ml_exit_short'
            except Exception as e:
                pass

            # Emergency Stop Loss
            if not should_exit and leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'emergency_stop_loss'

            # Emergency Max Hold
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'emergency_max_hold'

            if should_exit:
                # Calculate commissions
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
                    'exit_reason': exit_reason
                })

                position = None

    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)

        long_trades = trades_df[trades_df['side'] == 'LONG']
        short_trades = trades_df[trades_df['side'] == 'SHORT']

        return {
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'win_rate': (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades),
            'avg_leveraged_return': trades_df['leveraged_pnl_pct'].mean(),
            'total_return_pct': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            'final_capital': capital,
            'short_win_rate': (short_trades['leveraged_pnl_pct'] > 0).sum() / len(short_trades) if len(short_trades) > 0 else 0,
            'short_avg_return': short_trades['leveraged_pnl_pct'].mean() if len(short_trades) > 0 else 0,
            'ml_exit_short': len(short_trades[short_trades['exit_reason'] == 'ml_exit_short']) if len(short_trades) > 0 else 0
        }
    else:
        return None


# Test thresholds
print("="*80)
print("TESTING SHORT EXIT THRESHOLDS")
print("="*80)

# Test range: 0.58 ~ 0.70 (step 0.02)
thresholds_to_test = np.arange(0.58, 0.72, 0.02)

# Use representative windows for faster testing
window_size = 1440
step_size = 288
num_test_windows = 20  # Use 20 windows for good representation
total_windows = (len(df) - window_size) // step_size
test_indices = np.linspace(0, total_windows - 1, num_test_windows, dtype=int)

print(f"\nTesting {len(thresholds_to_test)} thresholds on {num_test_windows} windows")
print(f"Total windows available: {total_windows}")
print()

results = []

for threshold in thresholds_to_test:
    print(f"Testing threshold {threshold:.2f}...", end=" ")

    window_results = []

    for idx in test_indices:
        start_idx = idx * step_size
        end_idx = start_idx + window_size

        if end_idx <= len(df):
            window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            result = backtest_with_threshold(window_df, threshold)

            if result is not None:
                window_results.append(result)

    if len(window_results) > 0:
        # Aggregate results
        avg_result = {
            'threshold': threshold,
            'total_trades': np.mean([r['total_trades'] for r in window_results]),
            'long_trades': np.mean([r['long_trades'] for r in window_results]),
            'short_trades': np.mean([r['short_trades'] for r in window_results]),
            'win_rate': np.mean([r['win_rate'] for r in window_results]),
            'avg_leveraged_return': np.mean([r['avg_leveraged_return'] for r in window_results]),
            'total_return_pct': np.mean([r['total_return_pct'] for r in window_results]),
            'final_capital': np.mean([r['final_capital'] for r in window_results]),
            'short_win_rate': np.mean([r['short_win_rate'] for r in window_results]),
            'short_avg_return': np.mean([r['short_avg_return'] for r in window_results]),
            'ml_exit_short_pct': np.mean([r['ml_exit_short'] / r['short_trades'] if r['short_trades'] > 0 else 0 for r in window_results])
        }

        results.append(avg_result)

        print(f"Return: {avg_result['total_return_pct']*100:>6.2f}% | "
              f"WR: {avg_result['win_rate']*100:>5.1f}% | "
              f"SHORT WR: {avg_result['short_win_rate']*100:>5.1f}% | "
              f"SHORT Ret: {avg_result['short_avg_return']*100:>+6.2f}%")


# Analyze results
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

# Sort by total return
results_df = results_df.sort_values('total_return_pct', ascending=False)

print("\n📊 Top 5 Thresholds by Total Return:")
print(results_df[['threshold', 'total_return_pct', 'win_rate', 'short_win_rate', 'short_avg_return']].head(5).to_string(index=False))

# Find optimal threshold (balance return and win rate)
print("\n" + "="*80)
print("OPTIMAL THRESHOLD RECOMMENDATION")
print("="*80)

# Filter by minimum win rate (60%)
viable_thresholds = results_df[results_df['win_rate'] >= 0.60]

if len(viable_thresholds) > 0:
    # Get best by total return
    optimal = viable_thresholds.iloc[0]

    print(f"\n✅ RECOMMENDED THRESHOLD: {optimal['threshold']:.2f}")
    print(f"\n📈 Performance:")
    print(f"  Total Return: {optimal['total_return_pct']*100:.2f}% per window")
    print(f"  Win Rate: {optimal['win_rate']*100:.1f}%")
    print(f"  Avg Trades: {optimal['total_trades']:.1f} (LONG {optimal['long_trades']:.1f} + SHORT {optimal['short_trades']:.1f})")

    print(f"\n🎯 SHORT Performance:")
    print(f"  Win Rate: {optimal['short_win_rate']*100:.1f}%")
    print(f"  Avg Return: {optimal['short_avg_return']*100:+.2f}%")
    print(f"  ML Exit Rate: {optimal['ml_exit_short_pct']*100:.1f}%")

    # Compare with baseline (0.70)
    baseline = results_df[results_df['threshold'] == 0.70]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        improvement = optimal['total_return_pct'] - baseline['total_return_pct']

        print(f"\n📊 Improvement vs Baseline (0.70):")
        print(f"  Return: {baseline['total_return_pct']*100:.2f}% → {optimal['total_return_pct']*100:.2f}% "
              f"({improvement*100:+.2f}pp)")
        print(f"  SHORT Return: {baseline['short_avg_return']*100:+.2f}% → {optimal['short_avg_return']*100:+.2f}% "
              f"({(optimal['short_avg_return'] - baseline['short_avg_return'])*100:+.2f}pp)")

        # Annualized
        baseline_annual = (1 + baseline['total_return_pct']) ** 73 - 1
        optimal_annual = (1 + optimal['total_return_pct']) ** 73 - 1

        print(f"\n🚀 Annualized Impact:")
        print(f"  Baseline: {baseline_annual*100:.0f}%")
        print(f"  Optimized: {optimal_annual*100:.0f}%")

    recommended_threshold = optimal['threshold']
else:
    print("\n⚠️ No thresholds meet minimum win rate requirement (60%)")
    print("   However, best overall performance:")

    # Get best by total return regardless of win rate
    optimal = results_df.iloc[0]

    print(f"\n✅ BEST THRESHOLD (relaxed constraint): {optimal['threshold']:.2f}")
    print(f"\n📈 Performance:")
    print(f"  Total Return: {optimal['total_return_pct']*100:.2f}% per window")
    print(f"  Win Rate: {optimal['win_rate']*100:.1f}%")
    print(f"  Avg Trades: {optimal['total_trades']:.1f} (LONG {optimal['long_trades']:.1f} + SHORT {optimal['short_trades']:.1f})")

    print(f"\n🎯 SHORT Performance:")
    print(f"  Win Rate: {optimal['short_win_rate']*100:.1f}%")
    print(f"  Avg Return: {optimal['short_avg_return']*100:+.2f}%")

    # Compare with baseline (0.70)
    baseline = results_df[results_df['threshold'] == 0.70]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        improvement = optimal['total_return_pct'] - baseline['total_return_pct']

        print(f"\n📊 Improvement vs Baseline (0.70):")
        print(f"  Return: {baseline['total_return_pct']*100:.2f}% → {optimal['total_return_pct']*100:.2f}% "
              f"({improvement*100:+.2f}pp)")
        print(f"  SHORT Return: {baseline['short_avg_return']*100:+.2f}% → {optimal['short_avg_return']*100:+.2f}% "
              f"({(optimal['short_avg_return'] - baseline['short_avg_return'])*100:+.2f}pp)")

    recommended_threshold = optimal['threshold']

# Save results
output_file = RESULTS_DIR / f"short_exit_threshold_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print(f"""
1. Review recommended threshold: {optimal['threshold']:.2f}

2. Update production code:
   File: scripts/production/opportunity_gating_bot_4x.py

   Add:
   ML_EXIT_THRESHOLD_LONG = 0.70
   ML_EXIT_THRESHOLD_SHORT = {optimal['threshold']:.2f}

   Modify check_exit_signal() to use side-specific thresholds

3. Monitor for 1 week:
   - SHORT win rate > 60%
   - SHORT avg return improvement
   - Total return vs backtest

4. Rollback if:
   - Win rate < 55%
   - Emergency exits increase > 10%
   - Total return < 80% of backtest
""")

print("="*80)