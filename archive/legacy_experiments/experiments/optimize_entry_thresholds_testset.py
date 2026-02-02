"""
Optimize Entry Thresholds on Test Set Only
==========================================

Grid search for optimal LONG and SHORT entry thresholds.
CRITICAL: Uses ONLY Test Set to avoid data leakage.

Target Metrics:
- Win Rate >= 60%
- Entry Frequency: 3-5 trades/day
- Return: Maximize

Grid:
- LONG: [0.70, 0.75, 0.80, 0.85]
- SHORT: [0.75, 0.80, 0.85, 0.90]
- Total: 16 combinations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import itertools

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("THRESHOLD OPTIMIZATION - TEST SET ONLY")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Threshold grid
LONG_THRESHOLDS = [0.70, 0.75, 0.80, 0.85]
SHORT_THRESHOLDS = [0.75, 0.80, 0.85, 0.90]
GATE_THRESHOLD = 0.001  # Fixed

# Backtest config
INITIAL_BALANCE = 10000.0
LEVERAGE = 4
MAX_HOLD_PERIODS = 48  # 4 hours
TAKE_PROFIT_PCT = 3.0
STOP_LOSS_PCT = -6.0  # Balance-based
MAKER_FEE = 0.0002
TAKER_FEE = 0.0004

print(f"\n📊 Grid Search Configuration:")
print(f"  LONG Thresholds: {LONG_THRESHOLDS}")
print(f"  SHORT Thresholds: {SHORT_THRESHOLDS}")
print(f"  Total Combinations: {len(LONG_THRESHOLDS) * len(SHORT_THRESHOLDS)}")
print(f"\n  Backtest Config:")
print(f"    Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"    Leverage: {LEVERAGE}x")
print(f"    Max Hold: {MAX_HOLD_PERIODS} periods (4 hours)")
print(f"    Take Profit: {TAKE_PROFIT_PCT}%")
print(f"    Stop Loss: {STOP_LOSS_PCT}% (balance-based)")

# ============================================================================
# STEP 1: Load Test Set Only
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading Test Set ONLY")
print("="*80)

# Load full data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)

# Calculate features
print("\nCalculating features...")
df_full = calculate_all_features(df_full)
df_full = prepare_exit_features(df_full)
df_full = df_full.dropna().reset_index(drop=True)

# Split: Take ONLY Test Set (last 20%)
split_idx = int(len(df_full) * 0.8)
df = df_full[split_idx:].copy().reset_index(drop=True)

print(f"\n✅ Test Set Loaded:")
print(f"   Total candles: {len(df):,}")
print(f"   Period: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print(f"   Duration: {(len(df) * 5) / (60 * 24):.1f} days")
print(f"\n   ⚠️  Training data EXCLUDED - No data leakage")

# ============================================================================
# STEP 2: Load Models
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Loading Models")
print("="*80)

# LONG Entry Model
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_trade_outcome_full_20251021_214616_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

# SHORT Entry Model
short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_trade_outcome_full_20251021_214616_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

# Exit Models
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print("✅ All models loaded")

# ============================================================================
# STEP 3: Position Sizing Function
# ============================================================================

def calculate_position_size(signal_strength):
    """Simple position sizing: 20-95% based on signal strength."""
    if signal_strength < 0.65:
        return 20.0
    elif signal_strength >= 0.95:
        return 95.0
    else:
        return 20.0 + ((signal_strength - 0.65) / (0.95 - 0.65)) * 75.0

# ============================================================================
# STEP 4: Backtest Function
# ============================================================================

def run_backtest(df, long_threshold, short_threshold):
    """Run backtest with specific thresholds."""

    balance = INITIAL_BALANCE
    position = None
    trades = []

    # Pre-calculate predictions (optimization)
    X_long = df[long_feature_columns].values
    X_long_scaled = long_scaler.transform(X_long)
    long_probs = long_model.predict_proba(X_long_scaled)[:, 1]

    X_short = df[short_feature_columns].values
    X_short_scaled = short_scaler.transform(X_short)
    short_probs = short_model.predict_proba(X_short_scaled)[:, 1]

    for i in range(len(df)):
        current_price = df.loc[i, 'close']

        # Exit check
        if position is not None:
            hold_periods = i - position['entry_idx']

            # Calculate P&L
            if position['direction'] == 'LONG':
                price_change_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
            else:
                price_change_pct = ((position['entry_price'] - current_price) / position['entry_price']) * 100

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Balance-based stop loss
            position_size_pct = position['position_size'] / position['balance_at_entry']
            price_sl_pct = STOP_LOSS_PCT / (position_size_pct * LEVERAGE)

            exit_reason = None

            # Emergency exits
            if leveraged_pnl_pct >= TAKE_PROFIT_PCT:
                exit_reason = 'take_profit'
            elif price_change_pct <= price_sl_pct:
                exit_reason = 'stop_loss'
            elif hold_periods >= MAX_HOLD_PERIODS:
                exit_reason = 'max_hold'
            else:
                # ML Exit
                if position['direction'] == 'LONG':
                    exit_features = df.loc[i, long_exit_feature_columns].values.reshape(1, -1)
                    exit_features_scaled = long_exit_scaler.transform(exit_features)
                    exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
                    if exit_prob >= 0.72:
                        exit_reason = 'ml_exit'
                else:
                    exit_features = df.loc[i, short_exit_feature_columns].values.reshape(1, -1)
                    exit_features_scaled = short_exit_scaler.transform(exit_features)
                    exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]
                    if exit_prob >= 0.72:
                        exit_reason = 'ml_exit'

            if exit_reason:
                # Close position
                exit_fee = position['position_size'] * TAKER_FEE
                pnl = (position['position_size'] * leveraged_pnl_pct / 100) - exit_fee - position['entry_fee']
                balance += pnl

                trades.append({
                    'direction': position['direction'],
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'hold_periods': hold_periods
                })
                position = None

        # Entry check
        if position is None and i < len(df) - MAX_HOLD_PERIODS:
            long_prob = long_probs[i]
            short_prob = short_probs[i]

            direction = None

            if long_prob >= long_threshold:
                direction = 'LONG'
            elif short_prob >= short_threshold:
                # Opportunity gating
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    direction = 'SHORT'

            if direction:
                signal_strength = long_prob if direction == 'LONG' else short_prob
                position_size_pct = calculate_position_size(signal_strength)
                position_size = balance * (position_size_pct / 100)
                entry_fee = position_size * MAKER_FEE

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'direction': direction,
                    'position_size': position_size,
                    'entry_fee': entry_fee,
                    'balance_at_entry': balance,
                    'signal_strength': signal_strength
                }

    # Calculate metrics
    if len(trades) == 0:
        return {
            'return_pct': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'avg_trades_per_day': 0.0
        }

    total_return_pct = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = (winning_trades / len(trades)) * 100
    long_trades = sum(1 for t in trades if t['direction'] == 'LONG')
    short_trades = sum(1 for t in trades if t['direction'] == 'SHORT')

    test_days = (len(df) * 5) / (60 * 24)
    avg_trades_per_day = len(trades) / test_days

    return {
        'return_pct': total_return_pct,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'long_trades': long_trades,
        'short_trades': short_trades,
        'avg_trades_per_day': avg_trades_per_day,
        'final_balance': balance
    }

# ============================================================================
# STEP 5: Grid Search
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Running Grid Search")
print("="*80)

results = []
total_combinations = len(LONG_THRESHOLDS) * len(SHORT_THRESHOLDS)

print(f"\nTesting {total_combinations} combinations...")
print(f"Test Set: {(len(df) * 5) / (60 * 24):.1f} days\n")

for idx, (long_th, short_th) in enumerate(itertools.product(LONG_THRESHOLDS, SHORT_THRESHOLDS), 1):
    print(f"  [{idx:2d}/{total_combinations}] Testing LONG={long_th:.2f}, SHORT={short_th:.2f}...", end=" ")

    result = run_backtest(df, long_th, short_th)
    result['long_threshold'] = long_th
    result['short_threshold'] = short_th
    results.append(result)

    print(f"Return: {result['return_pct']:+6.2f}%, WR: {result['win_rate']:5.1f}%, Trades/day: {result['avg_trades_per_day']:.1f}")

# ============================================================================
# STEP 6: Analyze Results
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Analyzing Results")
print("="*80)

results_df = pd.DataFrame(results)

# Sort by return
results_df = results_df.sort_values('return_pct', ascending=False)

print("\n📊 Top 5 by Return:")
for idx, row in results_df.head(5).iterrows():
    print(f"  {row['long_threshold']:.2f} / {row['short_threshold']:.2f}: "
          f"{row['return_pct']:+6.2f}% | WR: {row['win_rate']:5.1f}% | "
          f"Trades: {row['total_trades']:.0f} ({row['avg_trades_per_day']:.1f}/day) | "
          f"L/S: {row['long_trades']}/{row['short_trades']}")

# Filter by criteria
print("\n📋 Filtering by Target Criteria:")
print("  Target: Win Rate >= 60%, Trades/day: 3-5")

filtered = results_df[
    (results_df['win_rate'] >= 60) &
    (results_df['avg_trades_per_day'] >= 3) &
    (results_df['avg_trades_per_day'] <= 5)
]

if len(filtered) > 0:
    print(f"\n✅ {len(filtered)} combinations meet criteria:\n")
    for idx, row in filtered.head(10).iterrows():
        print(f"  {row['long_threshold']:.2f} / {row['short_threshold']:.2f}: "
              f"{row['return_pct']:+6.2f}% | WR: {row['win_rate']:5.1f}% | "
              f"Trades: {row['total_trades']:.0f} ({row['avg_trades_per_day']:.1f}/day) | "
              f"L/S: {row['long_trades']}/{row['short_trades']}")

    # Best by return among filtered
    best = filtered.iloc[0]
    print(f"\n🏆 RECOMMENDED THRESHOLDS:")
    print(f"   LONG Threshold: {best['long_threshold']:.2f}")
    print(f"   SHORT Threshold: {best['short_threshold']:.2f}")
    print(f"\n   Expected Performance (Test Set):")
    print(f"     Return: {best['return_pct']:+.2f}%")
    print(f"     Win Rate: {best['win_rate']:.1f}%")
    print(f"     Trades: {best['total_trades']} ({best['avg_trades_per_day']:.1f}/day)")
    print(f"     LONG/SHORT: {best['long_trades']}/{best['short_trades']}")
else:
    print("\n⚠️  No combinations meet strict criteria")
    print("   Showing best by Win Rate:\n")

    best_wr = results_df.nlargest(5, 'win_rate')
    for idx, row in best_wr.iterrows():
        print(f"  {row['long_threshold']:.2f} / {row['short_threshold']:.2f}: "
              f"{row['return_pct']:+6.2f}% | WR: {row['win_rate']:5.1f}% | "
              f"Trades: {row['total_trades']:.0f} ({row['avg_trades_per_day']:.1f}/day)")

    best = best_wr.iloc[0]
    print(f"\n🎯 BEST WIN RATE THRESHOLDS:")
    print(f"   LONG Threshold: {best['long_threshold']:.2f}")
    print(f"   SHORT Threshold: {best['short_threshold']:.2f}")
    print(f"\n   Performance:")
    print(f"     Return: {best['return_pct']:+.2f}%")
    print(f"     Win Rate: {best['win_rate']:.1f}%")
    print(f"     Trades: {best['total_trades']} ({best['avg_trades_per_day']:.1f}/day)")

# ============================================================================
# STEP 7: Save Results
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Saving Results")
print("="*80)

results_file = PROJECT_ROOT / "results" / "threshold_optimization_testset.csv"
results_df.to_csv(results_file, index=False)
print(f"\n✅ Results saved to: {results_file.name}")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
