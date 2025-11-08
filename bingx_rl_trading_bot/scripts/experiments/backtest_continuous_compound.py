"""
Continuous Compound Backtest: Production Models
================================================

Test with CONTINUOUS capital (compound returns).
Each window uses previous window's ending capital.

This simulates REAL trading where losses hurt future returns.

Models:
- LONG Entry: Trade-Outcome Full Dataset (2025-10-18)
- SHORT Entry: Trade-Outcome Full Dataset (2025-10-18)
- Exit: ML Exit Models (Opportunity Gating)

Configuration:
  LONG Threshold: 0.65
  SHORT Threshold: 0.70
  Gate Threshold: 0.001
  Leverage: 4x
  Position Sizing: Dynamic (20-95%)
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

print("="*80)
print("CONTINUOUS COMPOUND BACKTEST (REALISTIC)")
print("="*80)
print(f"\n⚠️  Each window uses PREVIOUS window's capital (REAL trading simulation)\n")

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.70
ML_EXIT_THRESHOLD_SHORT = 0.72
EMERGENCY_MAX_HOLD_TIME = 96  # 8 hours
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees
TAKER_FEE = 0.0005

# =============================================================================
# LOAD PRODUCTION MODELS
# =============================================================================
print("Loading PRODUCTION models...")

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
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

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

# Initialize Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  Total candles: {len(df_full):,}\n")

# Calculate features
print("Calculating features...")
start_time = time.time()
df = calculate_all_features(df_full.copy())
df = prepare_exit_features(df)
feature_time = time.time() - start_time
print(f"  ✅ Features calculated ({feature_time:.1f}s)\n")

# Pre-calculate signals
print("Pre-calculating signals...")
signal_start = time.time()

long_feat = df[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
long_probs_array = long_model.predict_proba(long_feat_scaled)[:, 1]
df['long_prob'] = long_probs_array

short_feat = df[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
short_probs_array = short_model.predict_proba(short_feat_scaled)[:, 1]
df['short_prob'] = short_probs_array

signal_time = time.time() - signal_start
print(f"  ✅ Signals pre-calculated ({signal_time:.1f}s)\n")


# =============================================================================
# CONTINUOUS BACKTEST FUNCTION
# =============================================================================
def backtest_window(window_df, starting_capital):
    """
    Backtest one window using STARTING_CAPITAL (from previous window)
    """
    trades = []
    position = None
    capital = starting_capital  # ← Use previous window's ending capital!

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Entry
        if position is None:
            long_prob = window_df['long_prob'].iloc[i]
            short_prob = window_df['short_prob'].iloc[i]

            # LONG
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

                entry_fee = sizing_result['leveraged_value'] * TAKER_FEE
                capital -= entry_fee
                position['entry_fee'] = entry_fee

            # SHORT (gated)
            elif short_prob >= SHORT_THRESHOLD:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
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

                    entry_fee = sizing_result['leveraged_value'] * TAKER_FEE
                    capital -= entry_fee
                    position['entry_fee'] = entry_fee

        # Exit
        else:
            candles_held = i - position['entry_idx']

            if position['side'] == 'LONG':
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            should_exit = False
            exit_reason = ""

            # ML Exit
            try:
                if position['side'] == 'LONG':
                    exit_feat = window_df[long_exit_feature_columns].iloc[i:i+1].values
                    exit_feat_scaled = long_exit_scaler.transform(exit_feat)
                    exit_prob = long_exit_model.predict_proba(exit_feat_scaled)[0][1]
                    ml_threshold = ML_EXIT_THRESHOLD_LONG
                else:
                    exit_feat = window_df[short_exit_feature_columns].iloc[i:i+1].values
                    exit_feat_scaled = short_exit_scaler.transform(exit_feat)
                    exit_prob = short_exit_model.predict_proba(exit_feat_scaled)[0][1]
                    ml_threshold = ML_EXIT_THRESHOLD_SHORT

                if exit_prob >= ml_threshold:
                    should_exit = True
                    exit_reason = f"ML Exit ({exit_prob:.3f})"
            except:
                pass

            # Emergency Stop Loss
            if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = f"Stop Loss ({leveraged_pnl_pct*100:.1f}%)"

            # Emergency Max Hold
            if candles_held >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = f"Max Hold ({candles_held})"

            if should_exit:
                entry_notional = position['quantity'] * position['entry_price']
                exit_notional = position['quantity'] * current_price

                if position['side'] == 'LONG':
                    pnl_usd = exit_notional - entry_notional
                else:
                    pnl_usd = entry_notional - exit_notional

                exit_fee = position['leveraged_value'] * TAKER_FEE
                pnl_usd_net = pnl_usd - position['entry_fee'] - exit_fee

                capital += pnl_usd_net

                trades.append({
                    'side': position['side'],
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd_net': pnl_usd_net,
                    'exit_reason': exit_reason
                })

                position = None

    return trades, capital


# =============================================================================
# RUN CONTINUOUS BACKTEST
# =============================================================================
print("="*80)
print("RUNNING CONTINUOUS COMPOUND BACKTEST (NON-OVERLAPPING)")
print("="*80)

window_size = 1440  # 5 days
step_size = 1440  # 5 days (NON-OVERLAPPING!)

capital = INITIAL_CAPITAL
window_results = []
all_trades = []

num_windows = (len(df) - window_size) // step_size

for window_idx in range(num_windows):
    start_idx = window_idx * step_size
    end_idx = start_idx + window_size

    if end_idx > len(df):
        break

    window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    starting_capital = capital  # Use previous window's ending capital

    trades, ending_capital = backtest_window(window_df, starting_capital)

    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades) * 100
        window_return_pct = (ending_capital - starting_capital) / starting_capital * 100

        window_results.append({
            'window': window_idx,
            'starting_capital': starting_capital,
            'ending_capital': ending_capital,
            'window_return_pct': window_return_pct,
            'trades': len(trades),
            'win_rate': win_rate
        })

        all_trades.extend(trades)

    capital = ending_capital  # ← CRITICAL: Next window uses this capital!

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "="*80)
print("CONTINUOUS COMPOUND BACKTEST RESULTS")
print("="*80)

if len(window_results) > 0:
    results_df = pd.DataFrame(window_results)

    final_capital = results_df['ending_capital'].iloc[-1]
    total_return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print(f"\n💰 COMPOUND Performance:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Capital: ${final_capital:,.2f}")
    print(f"  Total Return: ${final_capital - INITIAL_CAPITAL:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"  Windows: {len(results_df)}")
    print(f"  Total Days: ~{len(results_df) * 5:.0f} (5-day non-overlapping windows)")

    if len(all_trades) > 0:
        all_trades_df = pd.DataFrame(all_trades)
        wins = all_trades_df[all_trades_df['pnl_usd_net'] > 0]

        print(f"\n📊 Trade Statistics:")
        print(f"  Total Trades: {len(all_trades)}")
        print(f"  Win Rate: {len(wins)}/{len(all_trades)} ({len(wins)/len(all_trades)*100:.1f}%)")
        print(f"  Avg P&L: {all_trades_df['leveraged_pnl_pct'].mean()*100:+.2f}%")

    print(f"\n📈 Window Breakdown (first 10):")
    print(f"{'Window':<8} {'Start $':<12} {'End $':<12} {'Return':<10} {'Trades':<8}")
    print("-"*60)
    for _, row in results_df.head(10).iterrows():
        print(f"{row['window']:<8} ${row['starting_capital']:<11,.2f} ${row['ending_capital']:<11,.2f} "
              f"{row['window_return_pct']:<9.2f}% {row['trades']:<8.0f}")

    # Comparison with Independent Testing
    print(f"\n" + "="*80)
    print("COMPARISON: Compound vs Independent Window Testing")
    print("="*80)

    avg_window_return = results_df['window_return_pct'].mean()

    print(f"\n📊 Independent Window Testing (current backtest):")
    print(f"  Method: Each window starts with $10,000")
    print(f"  Avg Return per Window: {avg_window_return:.2f}%")
    print(f"  ⚠️  This does NOT represent real trading!")

    print(f"\n💰 Compound Return (REAL trading):")
    print(f"  Method: Each window uses previous ending capital")
    print(f"  Total Return: {total_return_pct:+.2f}%")
    print(f"  ✅ This represents REAL compound trading")

    print(f"\n⚠️  Key Difference:")
    print(f"  Independent avg {avg_window_return:.2f}% ≠ Compound {total_return_pct:.2f}%")
    print(f"  Reason: Losses hurt future returns (compound effect)")

else:
    print("\n⚠️  No trades executed")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)