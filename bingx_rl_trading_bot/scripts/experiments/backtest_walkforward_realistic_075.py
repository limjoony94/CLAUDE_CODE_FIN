"""
Backtest Walk-Forward Decoupled Entry Models (0.75/0.75) - REALISTIC
====================================================================

Tests Walk-Forward Decoupled Entry models with REALISTIC position management.
- ONE position at a time (like production)
- Dynamic position sizing (20-95%)
- Opportunity gating for SHORT
- Balance-based Stop Loss (-3%)
- First exit signal gating

Models:
- Entry: walkforward_fixed_20251027_225529 (PERFORMANCE-BASED!)
- Exit: threshold_075_20251027_190512

Created: 2025-10-27
Updated: 2025-10-27 22:57 (NEW FIXED MODELS)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

# Configuration
LONG_THRESHOLD = 0.75
SHORT_THRESHOLD = 0.75
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03  # -3% balance
EMERGENCY_MAX_HOLD = 120  # 10 hours
INITIAL_CAPITAL = 10000  # $10,000 per window

# Opportunity Gating
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
GATE_THRESHOLD = 0.001

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BACKTEST: WALK-FORWARD DECOUPLED (0.75/0.75) - REALISTIC")
print("="*80)
print()
print(f"Configuration:")
print(f"  Entry Threshold: {LONG_THRESHOLD} (LONG), {SHORT_THRESHOLD} (SHORT)")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD_LONG} (LONG), {ML_EXIT_THRESHOLD_SHORT} (SHORT)")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS} (balance-based)")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles (10 hours)")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Position Management: ONE at a time (realistic)")
print()

# Load Data
print("-"*80)
print("STEP 1: Loading Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Prepare Exit features
print("\nPreparing Exit features...")
df = prepare_exit_features(df)
print(f"✅ Exit features added ({len(df.columns)} total columns)")
print()

# Load Entry Models (Walk-Forward FIXED - Performance-Based!)
print("-"*80)
print("STEP 2: Loading Entry Models (Walk-Forward FIXED)")
print("-"*80)

# LONG Entry
with open(MODELS_DIR / "xgboost_long_entry_walkforward_fixed_20251027_225529.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_walkforward_fixed_20251027_225529_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_walkforward_fixed_20251027_225529_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f]

# SHORT Entry
with open(MODELS_DIR / "xgboost_short_entry_walkforward_fixed_20251027_225529.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_walkforward_fixed_20251027_225529_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_walkforward_fixed_20251027_225529_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f]

print(f"  ✅ LONG Entry: {len(long_entry_features)} features")
print(f"  ✅ SHORT Entry: {len(short_entry_features)} features")
print()

# Load Exit Models
print("-"*80)
print("STEP 3: Loading Exit Models")
print("-"*80)

# LONG Exit
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f]

# SHORT Exit
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"  ✅ LONG Exit: {len(long_exit_features)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_features)} features")
print()

# Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Backtest Function
def backtest_window(window_df):
    """Backtest a 5-day window with REALISTIC position management"""

    # Generate Entry probabilities
    X_long = window_df[long_entry_features].values
    X_long_scaled = long_entry_scaler.transform(X_long)
    window_df['long_prob'] = long_entry_model.predict_proba(X_long_scaled)[:, 1]

    X_short = window_df[short_entry_features].values
    X_short_scaled = short_entry_scaler.transform(X_short)
    window_df['short_prob'] = short_entry_model.predict_proba(X_short_scaled)[:, 1]

    # Initialize
    position = None
    capital = INITIAL_CAPITAL
    trades = []
    first_exit_signal_received = False

    # Main loop
    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Check for first exit signal (without position)
        if not first_exit_signal_received and position is None:
            try:
                # Check LONG exit signal
                exit_feat_long = window_df[long_exit_features].iloc[i:i+1].values
                if not np.isnan(exit_feat_long).any():
                    exit_scaled_long = long_exit_scaler.transform(exit_feat_long)
                    long_exit_prob = long_exit_model.predict_proba(exit_scaled_long)[0][1]

                    if long_exit_prob >= ML_EXIT_THRESHOLD_LONG:
                        first_exit_signal_received = True
                        continue

                # Check SHORT exit signal
                exit_feat_short = window_df[short_exit_features].iloc[i:i+1].values
                if not np.isnan(exit_feat_short).any():
                    exit_scaled_short = short_exit_scaler.transform(exit_feat_short)
                    short_exit_prob = short_exit_model.predict_proba(exit_scaled_short)[0][1]

                    if short_exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                        first_exit_signal_received = True
            except:
                pass

        # Entry logic (ONLY after first exit signal AND no position)
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
                    'entry_prob': long_prob,
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
                    sizing_result = sizer.get_position_size_simple(
                        capital=capital,
                        signal_strength=short_prob,
                        leverage=LEVERAGE
                    )

                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': current_price,
                        'entry_prob': short_prob,
                        'position_size_pct': sizing_result['position_size_pct'],
                        'position_value': sizing_result['position_value'],
                        'leveraged_value': sizing_result['leveraged_value'],
                        'quantity': sizing_result['leveraged_value'] / current_price
                    }

        # Exit logic
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

            # Leveraged P&L %
            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Balance-based loss %
            balance_loss_pct = pnl_usd / capital

            # Exit conditions
            should_exit = False
            exit_reason = None

            # 1. Stop Loss (balance-based)
            if balance_loss_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'stop_loss'

            # 2. ML Exit
            if not should_exit:
                try:
                    if position['side'] == 'LONG':
                        exit_feat = window_df[long_exit_features].iloc[i:i+1].values
                        if not np.isnan(exit_feat).any():
                            exit_scaled = long_exit_scaler.transform(exit_feat)
                            exit_prob = long_exit_model.predict_proba(exit_scaled)[0][1]

                            if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                                should_exit = True
                                exit_reason = 'ml_exit'
                    else:  # SHORT
                        exit_feat = window_df[short_exit_features].iloc[i:i+1].values
                        if not np.isnan(exit_feat).any():
                            exit_scaled = short_exit_scaler.transform(exit_feat)
                            exit_prob = short_exit_model.predict_proba(exit_scaled)[0][1]

                            if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                                should_exit = True
                                exit_reason = 'ml_exit'
                except:
                    pass

            # 3. Max Hold
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD:
                should_exit = True
                exit_reason = 'max_hold'

            # Execute exit
            if should_exit:
                # Calculate fees (0.05% taker fee × 2 for entry + exit)
                fee_pct = 0.0005 * 2
                pnl_after_fees = pnl_usd - (position['leveraged_value'] * fee_pct)

                # Update capital
                capital += pnl_after_fees

                # Record trade
                trades.append({
                    'side': position['side'],
                    'entry_idx': position['entry_idx'],
                    'exit_idx': i,
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob'],
                    'hold_time': time_in_pos,
                    'pnl': pnl_after_fees,
                    'pnl_pct': (pnl_after_fees / INITIAL_CAPITAL) * 100,
                    'exit_reason': exit_reason
                })

                # Clear position
                position = None

    return trades, capital

# Run 108-Window Backtest
print("-"*80)
print("STEP 4: Running 108-Window Backtest (REALISTIC)")
print("-"*80)
print()

WINDOW_SIZE = 1440  # 5 days = 1440 5-min candles
START_IDX = 0
results = []

for window_idx in range(108):
    start = START_IDX + (window_idx * WINDOW_SIZE)
    end = start + WINDOW_SIZE

    if end > len(df):
        break

    df_window = df.iloc[start:end].copy()
    df_window = df_window.reset_index(drop=True)

    # Backtest window
    trades, final_capital = backtest_window(df_window)

    if len(trades) == 0:
        continue

    # Calculate metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    total_return_usd = final_capital - INITIAL_CAPITAL
    total_return_pct = (total_return_usd / INITIAL_CAPITAL) * 100
    avg_return = total_return_pct / total_trades if total_trades > 0 else 0

    long_trades = sum(1 for t in trades if t['side'] == 'LONG')
    short_trades = sum(1 for t in trades if t['side'] == 'SHORT')

    ml_exits = sum(1 for t in trades if t['exit_reason'] == 'ml_exit')
    sl_exits = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
    max_hold_exits = sum(1 for t in trades if t['exit_reason'] == 'max_hold')

    ml_exit_rate = ml_exits / total_trades if total_trades > 0 else 0

    results.append({
        'window': window_idx,
        'total_trades': total_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'win_rate': win_rate * 100,
        'avg_return': avg_return,
        'total_return': total_return_pct,
        'ml_exit_rate': ml_exit_rate * 100,
        'sl_exits': sl_exits,
        'max_hold_exits': max_hold_exits
    })

    if (window_idx + 1) % 20 == 0:
        print(f"  Window {window_idx+1}/108 complete...")

print(f"\n✅ Backtest complete: {len(results)} windows")
print()

# Calculate Overall Statistics
print("="*80)
print("BACKTEST RESULTS - REALISTIC")
print("="*80)
print()

df_results = pd.DataFrame(results)

total_trades = df_results['total_trades'].sum()
total_wins = (df_results['win_rate'] / 100 * df_results['total_trades']).sum()
overall_win_rate = total_wins / total_trades if total_trades > 0 else 0

weighted_avg_return = (df_results['avg_return'] * df_results['total_trades']).sum() / total_trades if total_trades > 0 else 0
weighted_total_return = (df_results['total_return'] * df_results['total_trades']).sum() / total_trades if total_trades > 0 else 0

long_total = df_results['long_trades'].sum()
short_total = df_results['short_trades'].sum()

ml_exits_total = (df_results['ml_exit_rate'] / 100 * df_results['total_trades']).sum()
ml_exit_rate_overall = ml_exits_total / total_trades if total_trades > 0 else 0

print(f"Total Windows: {len(results)}")
print(f"Total Trades: {total_trades:,}")
print(f"  LONG: {long_total:,} ({long_total/total_trades*100:.1f}%)")
print(f"  SHORT: {short_total:,} ({short_total/total_trades*100:.1f}%)")
print()
print(f"Overall Win Rate: {overall_win_rate*100:.2f}%")
print(f"Avg Return per Trade: {weighted_avg_return:.2f}%")
print(f"Avg Return per Window: {weighted_total_return:.2f}%")
print(f"Avg Trades per Window: {df_results['total_trades'].mean():.1f}")
print(f"Trades per Day: {df_results['total_trades'].mean() / 5:.2f}")
print()
print(f"Exit Distribution:")
print(f"  ML Exit: {ml_exit_rate_overall*100:.1f}%")
print(f"  Stop Loss: {df_results['sl_exits'].sum():,} ({df_results['sl_exits'].sum()/total_trades*100:.1f}%)")
print(f"  Max Hold: {df_results['max_hold_exits'].sum():,} ({df_results['max_hold_exits'].sum()/total_trades*100:.1f}%)")
print()

# Per-Window Statistics
print(f"Per-Window Statistics:")
print(f"  Avg Win Rate: {df_results['win_rate'].mean():.2f}%")
print(f"  Median Win Rate: {df_results['win_rate'].median():.2f}%")
print(f"  Avg Return: {df_results['total_return'].mean():.2f}%")
print(f"  Median Return: {df_results['total_return'].median():.2f}%")
print()

# Save Results
output_file = RESULTS_DIR / f"backtest_walkforward_realistic_075_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df_results.to_csv(output_file, index=False)
print(f"✅ Results saved: {output_file.name}")
print()

print("="*80)
print("COMPARISON WITH UNREALISTIC BACKTEST")
print("="*80)
print()
print(f"Unrealistic (no position management):")
print(f"  Trades per day: 59.65")
print(f"  Capital: Unlimited (multiple positions)")
print()
print(f"Realistic (ONE position at a time):")
print(f"  Trades per day: {df_results['total_trades'].mean() / 5:.2f}")
print(f"  Capital: Limited ($10,000 per window)")
print(f"  Position Management: Like production")
print()
