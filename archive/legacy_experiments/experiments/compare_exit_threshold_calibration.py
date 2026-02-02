"""
Exit Threshold Calibration Comparison
======================================

User Question: "exit signal 관련하여 현재 프로덕션 설정은 0.8이지만,
훈련했을 때는 0.75로 훈련했다고 알고 있습니다.
훈련했던 대로 설정하면 안될까요?"

Translation: "Exit signals are set to 0.8 in production but were trained
with 0.75. Shouldn't we use the training threshold?"

Purpose: Compare Exit threshold 0.75 vs 0.80 to determine optimal configuration

Test Scenarios:
  Scenario A: Entry 0.80 + Exit 0.75 (training threshold - model calibration)
  Scenario B: Entry 0.80 + Exit 0.80 (current production - higher quality)

Hypothesis: Exit 0.75 may perform better due to model calibration match,
            OR Exit 0.80 may filter better despite calibration mismatch

Models:
  - Exit Models: xgboost_long_exit_threshold_075_20251027_190512.pkl
  - Trained with threshold 0.75 (calibrated to this threshold)
  - Current production uses 0.80 (higher threshold than training)

Date: 2025-10-30
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# =============================================================================
# FIXED CONFIGURATION (Common to both scenarios)
# =============================================================================

# Entry Thresholds (FIXED at 0.80)
LONG_THRESHOLD = 0.80
SHORT_THRESHOLD = 0.80
GATE_THRESHOLD = 0.001

# Risk Parameters (FIXED)
EMERGENCY_STOP_LOSS = 0.03  # -3%
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Trading
LEVERAGE = 4
INITIAL_CAPITAL = 10000
TAKER_FEE = 0.0005  # 0.05% BingX taker fee

# =============================================================================
# MODEL PATHS (Current Production - trained with 0.75)
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"

# Entry Models (Walk-Forward 0.80 - PRODUCTION MODELS - 2025-10-27 23:57)
LONG_ENTRY_MODEL = MODELS_DIR / "xgboost_long_entry_walkforward_080_20251027_235741.pkl"
LONG_ENTRY_SCALER = MODELS_DIR / "xgboost_long_entry_walkforward_080_20251027_235741_scaler.pkl"
SHORT_ENTRY_MODEL = MODELS_DIR / "xgboost_short_entry_walkforward_080_20251027_235741.pkl"
SHORT_ENTRY_SCALER = MODELS_DIR / "xgboost_short_entry_walkforward_080_20251027_235741_scaler.pkl"

# Exit Models (Threshold 0.75 - TRAINING THRESHOLD)
LONG_EXIT_MODEL = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
LONG_EXIT_SCALER = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl"
SHORT_EXIT_MODEL = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"
SHORT_EXIT_SCALER = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl"

print("=" * 80)
print("EXIT THRESHOLD CALIBRATION COMPARISON")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nUser Question: Should we use Exit 0.75 (training threshold)")
print(f"               vs Exit 0.80 (current production)?")
print(f"\nFixed Configuration:")
print(f"  Entry: LONG={LONG_THRESHOLD}, SHORT={SHORT_THRESHOLD}, Gate={GATE_THRESHOLD}")
print(f"  Risk: SL=-{EMERGENCY_STOP_LOSS*100}%, MaxHold={EMERGENCY_MAX_HOLD_TIME/12}h")
print(f"  Leverage: {LEVERAGE}x")
print(f"\nTest Scenarios:")
print(f"  Scenario A: Entry 0.80 + Exit 0.75 (training threshold)")
print(f"  Scenario B: Entry 0.80 + Exit 0.80 (current production)")
print("=" * 80)

# =============================================================================
# LOAD MODELS
# =============================================================================

print("\nLoading models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
long_entry_scaler = joblib.load(LONG_ENTRY_SCALER)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
short_entry_scaler = joblib.load(SHORT_ENTRY_SCALER)
long_exit_model = joblib.load(LONG_EXIT_MODEL)
long_exit_scaler = joblib.load(LONG_EXIT_SCALER)
short_exit_model = joblib.load(SHORT_EXIT_MODEL)
short_exit_scaler = joblib.load(SHORT_EXIT_SCALER)

# Load feature lists
long_entry_features_path = MODELS_DIR / "xgboost_long_entry_walkforward_080_20251027_235741_features.txt"
with open(long_entry_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

short_entry_features_path = MODELS_DIR / "xgboost_short_entry_walkforward_080_20251027_235741_features.txt"
with open(short_entry_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print("✅ All models loaded")
print(f"  Exit models trained with threshold: 0.75")
print(f"  LONG Entry features: {len(long_entry_features)}")
print(f"  SHORT Entry features: {len(short_entry_features)}")
print(f"  LONG Exit features: {len(long_exit_features)}")
print(f"  SHORT Exit features: {len(short_exit_features)}")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading data (full period)...")
data_path = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_updated.csv"
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Total days: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"  Total candles: {len(df):,}")

# =============================================================================
# CALCULATE FEATURES
# =============================================================================

print("\nCalculating features...")
df_features = calculate_all_features_enhanced_v2(df)
print(f"✅ Features calculated ({len(df_features)} rows)")

print("\nPreparing exit features...")
df_features = prepare_exit_features(df_features)
print(f"✅ Exit features prepared ({len(df_features)} rows)")

# =============================================================================
# BACKTEST LOGIC
# =============================================================================

def dynamic_position_size(prob, min_size=0.20, max_size=0.95):
    """Dynamic position sizing based on signal strength"""
    if prob < 0.65:
        return min_size
    elif prob >= 0.85:
        return max_size
    else:
        return min_size + (max_size - min_size) * ((prob - 0.65) / (0.85 - 0.65))

def run_single_window_backtest(df_window, ml_exit_threshold_long, ml_exit_threshold_short):
    """Run backtest on a single window with specified exit thresholds"""

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df_window)):
        row = df_window.iloc[i]
        timestamp = row['timestamp']
        price = row['close']

        # === POSITION MANAGEMENT ===
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            position_size_pct = position['position_size_pct']

            # Calculate P&L
            if side == 'LONG':
                price_change_pct = (price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # P&L in USD
            leveraged_value = INITIAL_CAPITAL * position_size_pct * LEVERAGE
            pnl_usd = leveraged_value * price_change_pct

            # Check exit conditions
            should_exit = False
            exit_reason = None

            # 1. Emergency Stop Loss (-3% balance)
            if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = "STOP_LOSS"

            # 2. Emergency Max Hold (10 hours = 120 candles)
            elif i - position['entry_index'] >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = "MAX_HOLD"

            # 3. ML Exit (USE SPECIFIED THRESHOLD)
            else:
                if side == 'LONG':
                    X_exit = row[long_exit_features].values.reshape(1, -1)
                    X_exit_scaled = long_exit_scaler.transform(X_exit)
                    exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ml_exit_threshold_long:
                        should_exit = True
                        exit_reason = "ML_EXIT"

                else:  # SHORT
                    X_exit = row[short_exit_features].values.reshape(1, -1)
                    X_exit_scaled = short_exit_scaler.transform(X_exit)
                    exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ml_exit_threshold_short:
                        should_exit = True
                        exit_reason = "ML_EXIT"

            # Execute exit
            if should_exit:
                # Calculate commissions
                entry_commission = leveraged_value * TAKER_FEE
                exit_commission = position['quantity'] * price * TAKER_FEE if 'quantity' in position else leveraged_value * TAKER_FEE
                total_commission = entry_commission + exit_commission

                # Net P&L after fees
                net_pnl_usd = pnl_usd - total_commission
                capital += net_pnl_usd

                hold_time = i - position['entry_index']

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position_size_pct': position_size_pct,
                    'pnl_pct': leveraged_pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'commission_usd': total_commission,
                    'net_pnl_usd': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time
                })

                position = None

        # === ENTRY SIGNALS ===
        if position is None:
            # Get entry probabilities
            X_long = row[long_entry_features].values.reshape(1, -1)
            X_long_scaled = long_entry_scaler.transform(X_long)
            long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

            X_short = row[short_entry_features].values.reshape(1, -1)
            X_short_scaled = short_entry_scaler.transform(X_short)
            short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

            # Opportunity cost gating
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            # Entry logic (FIXED at 0.80)
            if long_prob >= LONG_THRESHOLD:
                side = 'LONG'
                entry_prob = long_prob
            elif short_prob >= SHORT_THRESHOLD and opportunity_cost > GATE_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob
            else:
                continue

            # Dynamic position sizing
            position_size_pct = dynamic_position_size(entry_prob)
            leveraged_value = INITIAL_CAPITAL * position_size_pct * LEVERAGE
            quantity = leveraged_value / price

            position = {
                'side': side,
                'entry_price': price,
                'entry_time': timestamp,
                'entry_index': i,
                'entry_prob': entry_prob,
                'position_size_pct': position_size_pct,
                'leveraged_value': leveraged_value,
                'quantity': quantity
            }

    return trades, capital

def run_108_window_backtest(ml_exit_threshold_long, ml_exit_threshold_short, scenario_name):
    """Run 108-window backtest with specified exit thresholds"""

    window_size = 1440  # 5 days
    step_size = 288     # 1 day step
    num_windows = (len(df_features) - window_size) // step_size

    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario_name}: Exit Threshold {ml_exit_threshold_long:.2f}")
    print(f"{'='*80}")
    print(f"  Window Size: {window_size} candles (5 days)")
    print(f"  Step Size: {step_size} candles (1 day)")
    print(f"  Total Windows: {num_windows}")

    window_results = []
    all_trades = []

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df_features):
            break

        window_df = df_features.iloc[start_idx:end_idx].reset_index(drop=True)

        # Run backtest for this window
        trades, final_capital = run_single_window_backtest(
            window_df,
            ml_exit_threshold_long,
            ml_exit_threshold_short
        )

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            all_trades.extend(trades)

            # Calculate window metrics
            total_return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']

            window_results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'return_pct': total_return_pct,
                'final_capital': final_capital,
                'win_rate': (trades_df['net_pnl_usd'] > 0).mean() * 100,
                'ml_exit_rate': (trades_df['exit_reason'] == 'ML_EXIT').mean() * 100,
                'avg_hold_time': trades_df['hold_time'].mean()
            })

    return pd.DataFrame(window_results), all_trades

# =============================================================================
# RUN BOTH SCENARIOS
# =============================================================================

print("\n" + "="*80)
print("RUNNING COMPARISON BACKTEST (2 SCENARIOS)")
print("="*80)

# Scenario A: Exit 0.75 (training threshold - model calibration)
results_A, trades_A = run_108_window_backtest(0.75, 0.75, "A (Exit 0.75 - Training Threshold)")

# Scenario B: Exit 0.80 (current production - higher quality filter)
results_B, trades_B = run_108_window_backtest(0.80, 0.80, "B (Exit 0.80 - Current Production)")

# =============================================================================
# COMPARISON ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("COMPARISON RESULTS - Exit 0.75 vs Exit 0.80")
print("="*80)

def print_scenario_stats(results_df, trades_list, scenario_name, exit_threshold):
    """Print statistics for a scenario"""

    if len(results_df) == 0:
        print(f"\n{scenario_name}: NO DATA")
        return None

    df_trades = pd.DataFrame(trades_list)
    wins = df_trades[df_trades['net_pnl_usd'] > 0]
    losses = df_trades[df_trades['net_pnl_usd'] <= 0]

    # Window-level statistics
    mean_return = results_df['return_pct'].mean()
    std_return = results_df['return_pct'].std()
    sharpe_ratio = (mean_return / std_return) * 8.544 if std_return > 0 else 0

    # Trade statistics
    total_trades = len(df_trades)
    win_rate = len(wins) / len(df_trades) * 100 if len(df_trades) > 0 else 0
    ml_exit_rate = (df_trades['exit_reason'] == 'ML_EXIT').mean() * 100

    # LONG/SHORT distribution
    long_pct = len(df_trades[df_trades['side'] == 'LONG']) / len(df_trades) * 100 if len(df_trades) > 0 else 0
    short_pct = 100 - long_pct

    print(f"\n{scenario_name} (Exit Threshold: {exit_threshold:.2f})")
    print(f"{'-'*80}")
    print(f"Return per 5-day window: {mean_return:+.2f}% ± {std_return:.2f}%")
    print(f"Sharpe Ratio (annualized): {sharpe_ratio:.3f}")
    print(f"Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Total Trades: {total_trades:,} ({total_trades/len(results_df):.1f} per window)")
    print(f"Trades/Day: {total_trades/(len(results_df)*5):.1f}")
    print(f"LONG/SHORT: {long_pct:.1f}% / {short_pct:.1f}%")
    print(f"ML Exit Usage: {ml_exit_rate:.1f}%")
    print(f"Avg Hold Time: {df_trades['hold_time'].mean():.1f} candles ({df_trades['hold_time'].mean()/12:.1f}h)")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['net_pnl_usd'].sum() / losses['net_pnl_usd'].sum())
        print(f"Profit Factor: {profit_factor:.2f}x")
        print(f"Avg Win: ${wins['net_pnl_usd'].mean():.2f}")
        print(f"Avg Loss: ${losses['net_pnl_usd'].mean():.2f}")

    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'trades_per_day': total_trades/(len(results_df)*5),
        'ml_exit_rate': ml_exit_rate,
        'avg_hold_time': df_trades['hold_time'].mean(),
        'long_pct': long_pct,
        'short_pct': short_pct
    }

# Print both scenarios
stats_A = print_scenario_stats(results_A, trades_A, "SCENARIO A", 0.75)
stats_B = print_scenario_stats(results_B, trades_B, "SCENARIO B", 0.80)

# =============================================================================
# SIDE-BY-SIDE COMPARISON TABLE
# =============================================================================

if stats_A is not None and stats_B is not None:
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<30} {'Exit 0.75':>15} {'Exit 0.80':>15} {'Winner':>15}")
    print("-"*80)

    metrics = [
        ('Return per 5-day window', 'mean_return', '%', True),
        ('Sharpe Ratio (annualized)', 'sharpe_ratio', '', True),
        ('Win Rate', 'win_rate', '%', True),
        ('Total Trades', 'total_trades', '', True),
        ('Trades per Day', 'trades_per_day', '', True),
        ('ML Exit Usage', 'ml_exit_rate', '%', True),
        ('Avg Hold Time (candles)', 'avg_hold_time', '', False),
    ]

    for metric_name, key, unit, higher_better in metrics:
        val_A = stats_A[key]
        val_B = stats_B[key]

        if higher_better:
            winner = "Exit 0.75" if val_A > val_B else "Exit 0.80"
        else:
            winner = "Exit 0.75" if val_A < val_B else "Exit 0.80"

        if unit == '%':
            print(f"{metric_name:<30} {val_A:>14.2f}% {val_B:>14.2f}% {winner:>15}")
        elif unit == '':
            print(f"{metric_name:<30} {val_A:>15.2f} {val_B:>15.2f} {winner:>15}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Calculate overall score (weighted)
    score_A = (
        stats_A['mean_return'] * 0.35 +  # Return weight: 35%
        stats_A['sharpe_ratio'] * 10 * 0.25 +  # Sharpe weight: 25% (scaled to ~0-100)
        stats_A['win_rate'] * 0.20 +  # Win rate weight: 20%
        stats_A['ml_exit_rate'] * 0.20  # ML exit weight: 20%
    )

    score_B = (
        stats_B['mean_return'] * 0.35 +
        stats_B['sharpe_ratio'] * 10 * 0.25 +
        stats_B['win_rate'] * 0.20 +
        stats_B['ml_exit_rate'] * 0.20
    )

    print(f"\nOverall Score (weighted):")
    print(f"  Scenario A (Exit 0.75): {score_A:.2f}")
    print(f"  Scenario B (Exit 0.80): {score_B:.2f}")

    if score_A > score_B:
        diff_pct = (score_A - score_B) / score_B * 100
        print(f"\n✅ WINNER: Scenario A (Exit 0.75 - Training Threshold)")
        print(f"   Advantage: {diff_pct:+.1f}% better overall score")
        print(f"\n💡 RECOMMENDATION: Use Exit threshold 0.75 (training threshold)")
        print(f"   Rationale: Model calibration - Exit models were trained with 0.75")
        print(f"   Action: Update production bot ML_EXIT_THRESHOLD_LONG/SHORT to 0.75")
    else:
        diff_pct = (score_B - score_A) / score_A * 100
        print(f"\n✅ WINNER: Scenario B (Exit 0.80 - Current Production)")
        print(f"   Advantage: {diff_pct:+.1f}% better overall score")
        print(f"\n💡 RECOMMENDATION: Keep Exit threshold 0.80 (current production)")
        print(f"   Rationale: Higher quality filtering outweighs calibration mismatch")
        print(f"   Action: No changes needed - current configuration is optimal")

    # Save comparison results
    comparison_df = pd.DataFrame({
        'Metric': [m[0] for m in metrics],
        'Exit_0.75': [stats_A[m[1]] for m in metrics],
        'Exit_0.80': [stats_B[m[1]] for m in metrics],
        'Winner': [
            "Exit 0.75" if (stats_A[m[1]] > stats_B[m[1]]) == m[3] else "Exit 0.80"
            for m in metrics
        ]
    })

    output_file = PROJECT_ROOT / "results" / f"exit_threshold_calibration_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✅ Comparison results saved: {output_file}")

    # Save detailed results for both scenarios
    results_A_file = PROJECT_ROOT / "results" / f"scenario_A_exit_075_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_A.to_csv(results_A_file, index=False)
    print(f"✅ Scenario A details saved: {results_A_file}")

    results_B_file = PROJECT_ROOT / "results" / f"scenario_B_exit_080_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_B.to_csv(results_B_file, index=False)
    print(f"✅ Scenario B details saved: {results_B_file}")

print("\n" + "="*80)
