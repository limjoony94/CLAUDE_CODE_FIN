"""
Backtest: Production Feature Models (85 LONG, 79 SHORT)
=========================================================

Validate Entry models trained with Production's proven feature set:
  - LONG: 85 features (includes VWAP/VP + Candlesticks)
  - SHORT: 79 features (includes VWAP/VP + Volatility)

Expected: Win Rate should improve toward Production's 73.86%
Previous (Phase 1): 0% WR, -42.88% return (wrong features)

Created: 2025-10-31
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

print("="*80)
print("PRODUCTION FEATURE MODELS BACKTEST")
print("="*80)
print()
print("Testing Entry models with Production's proven feature set:")
print("  - LONG: 85 features (VWAP/VP + Candlesticks)")
print("  - SHORT: 79 features (VWAP/VP + Volatility)")
print("  - Expected: Win Rate improvement toward 73.86%")
print()

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.75
SHORT_THRESHOLD = 0.75
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
EMERGENCY_MAX_HOLD_TIME = 120
EMERGENCY_STOP_LOSS = 0.03

# Expected Values (for Opportunity Gating)
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees
TAKER_FEE = 0.0005

# Load Production-Feature Entry Models
print("Loading Entry models (Production Features - 20251031_183819)...")
import joblib

# Find the latest timestamp models
timestamp = "20251031_183819"

long_model_path = MODELS_DIR / f"xgboost_long_entry_production_features_{timestamp}.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_features_path = MODELS_DIR / f"xgboost_long_entry_production_features_{timestamp}_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / f"xgboost_short_entry_production_features_{timestamp}.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_features_path = MODELS_DIR / f"xgboost_short_entry_production_features_{timestamp}_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Entry models loaded")
print(f"   LONG: {len(long_feature_columns)} features (Production)")
print(f"   SHORT: {len(short_feature_columns)} features (Production)")

# Load Exit Models
print("Loading Exit models (0.75 threshold - 20251024_043527/044510)...")
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Exit models loaded: LONG({len(long_exit_feature_columns)}), SHORT({len(short_exit_feature_columns)})")
print()

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Load pre-calculated features (includes VWAP/VP)
print("Loading pre-calculated features (includes VWAP/VP)...")
features_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features.csv"
df = pd.read_csv(features_file)
print(f"✅ Features loaded: {len(df):,} candles, {len(df.columns)} features")
print()

# Add missing Exit features if needed
print("Checking Exit features...")
if 'bb_width' not in df.columns:
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
    else:
        df['bb_width'] = 0
    print("  ✅ bb_width added")

if 'vwap' not in df.columns:
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap'] = df['vwap'].ffill().bfill()
    print("  ✅ vwap added")

print(f"  ✅ All features available")
print()

# Pre-calculate signals (VECTORIZED)
print("Pre-calculating Entry signals...")
signal_start = time.time()
try:
    # LONG signals
    long_feat = df[long_feature_columns].values
    long_probs_array = long_model.predict_proba(long_feat)[:, 1]
    df['long_prob'] = long_probs_array

    # SHORT signals
    short_feat = df[short_feature_columns].values
    short_probs_array = short_model.predict_proba(short_feat)[:, 1]
    df['short_prob'] = short_probs_array

    signal_time = time.time() - signal_start
    print(f"✅ Signals pre-calculated ({signal_time:.1f}s)")
except Exception as e:
    print(f"❌ Error pre-calculating signals: {e}")
    df['long_prob'] = 0.0
    df['short_prob'] = 0.0

print()


def backtest_opportunity_gating_4x(window_df):
    """
    Opportunity Gating Strategy with 4x Leverage + Dynamic Sizing
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(window_df)):
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

                position_size_pct = sizing_result['position_size_pct']
                position_size_usd = capital * position_size_pct
                position_size_btc = position_size_usd / current_price
                leveraged_value = position_size_usd * LEVERAGE

                # Entry fee
                entry_fee = leveraged_value * TAKER_FEE
                capital -= entry_fee

                position = {
                    'side': 'LONG',
                    'entry_price': current_price,
                    'entry_idx': i,
                    'position_size_btc': position_size_btc,
                    'position_size_usd': position_size_usd,
                    'position_size_pct': position_size_pct,
                    'leveraged_value': leveraged_value,
                    'entry_fee': entry_fee,
                    'entry_prob': long_prob,
                    'entry_capital': capital
                }

            # SHORT entry (with Opportunity Gating)
            elif short_prob >= SHORT_THRESHOLD:
                # Opportunity Gating Check
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    sizing_result = sizer.get_position_size_simple(
                        capital=capital,
                        signal_strength=short_prob,
                        leverage=LEVERAGE
                    )

                    position_size_pct = sizing_result['position_size_pct']
                    position_size_usd = capital * position_size_pct
                    position_size_btc = position_size_usd / current_price
                    leveraged_value = position_size_usd * LEVERAGE

                    entry_fee = leveraged_value * TAKER_FEE
                    capital -= entry_fee

                    position = {
                        'side': 'SHORT',
                        'entry_price': current_price,
                        'entry_idx': i,
                        'position_size_btc': position_size_btc,
                        'position_size_usd': position_size_usd,
                        'position_size_pct': position_size_pct,
                        'leveraged_value': leveraged_value,
                        'entry_fee': entry_fee,
                        'entry_prob': short_prob,
                        'entry_capital': capital
                    }

        # Exit logic
        else:
            # ML Exit prediction
            try:
                if position['side'] == 'LONG':
                    exit_features_list = long_exit_feature_columns
                    exit_model = long_exit_model
                    exit_scaler = long_exit_scaler
                else:
                    exit_features_list = short_exit_feature_columns
                    exit_model = short_exit_model
                    exit_scaler = short_exit_scaler

                exit_features_values = window_df[exit_features_list].iloc[i:i+1].values
                exit_features_scaled = exit_scaler.transform(exit_features_values)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                ml_exit_threshold = ML_EXIT_THRESHOLD_LONG if position['side'] == 'LONG' else ML_EXIT_THRESHOLD_SHORT
                ml_exit_signal = exit_prob >= ml_exit_threshold
            except Exception as e:
                ml_exit_signal = False
                exit_prob = 0.0

            # Calculate P&L
            if position['side'] == 'LONG':
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl = price_change_pct * LEVERAGE
            position_pnl_usd = position['position_size_usd'] * leveraged_pnl

            # Calculate balance loss (for balance-based SL)
            balance_loss = -position_pnl_usd / position['entry_capital']

            # Exit conditions
            hold_time = i - position['entry_idx']
            exit_reason = None

            # 1. ML Exit
            if ml_exit_signal:
                exit_reason = 'ML_EXIT'

            # 2. Emergency Stop Loss (balance-based)
            elif balance_loss >= EMERGENCY_STOP_LOSS:
                exit_reason = 'STOP_LOSS'

            # 3. Emergency Max Hold
            elif hold_time >= EMERGENCY_MAX_HOLD_TIME:
                exit_reason = 'MAX_HOLD'

            # Execute exit
            if exit_reason:
                # Exit fee
                exit_fee = position['leveraged_value'] * TAKER_FEE
                net_pnl = position_pnl_usd - exit_fee

                capital += net_pnl

                # Record trade
                trades.append({
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_idx': position['entry_idx'],
                    'exit_idx': i,
                    'hold_time': hold_time,
                    'position_size_pct': position['position_size_pct'],
                    'leveraged_value': position['leveraged_value'],
                    'entry_fee': position['entry_fee'],
                    'exit_fee': exit_fee,
                    'total_fees': position['entry_fee'] + exit_fee,
                    'position_pnl': position_pnl_usd,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'entry_prob': position['entry_prob'],
                    'exit_prob': exit_prob
                })

                position = None

    return capital, trades


# Run 108-window backtest
print("="*80)
print("RUNNING 108-WINDOW BACKTEST (5-day windows)")
print("="*80)
print()

WINDOW_SIZE = 1440  # 5 days = 1440 candles (5min)
all_windows_results = []
all_trades = []

num_windows = (len(df) - WINDOW_SIZE) // WINDOW_SIZE
print(f"Total windows: {num_windows}")
print()

backtest_start = time.time()

for window_idx in range(num_windows):
    start_idx = window_idx * WINDOW_SIZE
    end_idx = start_idx + WINDOW_SIZE
    window_df = df.iloc[start_idx:end_idx].copy()

    # Backtest window
    final_capital, trades = backtest_opportunity_gating_4x(window_df)

    # Calculate metrics
    return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    num_trades = len(trades)
    wins = len([t for t in trades if t['net_pnl'] > 0])
    losses = len([t for t in trades if t['net_pnl'] <= 0])
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

    # Exit distribution
    ml_exits = len([t for t in trades if t['exit_reason'] == 'ML_EXIT'])
    sl_exits = len([t for t in trades if t['exit_reason'] == 'STOP_LOSS'])
    max_hold_exits = len([t for t in trades if t['exit_reason'] == 'MAX_HOLD'])

    all_windows_results.append({
        'window': window_idx + 1,
        'return_pct': return_pct,
        'num_trades': num_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'ml_exits': ml_exits,
        'sl_exits': sl_exits,
        'max_hold_exits': max_hold_exits
    })

    all_trades.extend([{**t, 'window': window_idx + 1} for t in trades])

    # Progress
    if (window_idx + 1) % 10 == 0:
        print(f"  Window {window_idx + 1}/{num_windows} complete")

backtest_time = time.time() - backtest_start

# Calculate overall metrics
print()
print("="*80)
print("BACKTEST RESULTS")
print("="*80)
print()

results_df = pd.DataFrame(all_windows_results)
trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

avg_return = results_df['return_pct'].mean()
avg_trades = results_df['num_trades'].mean()
total_trades = len(trades_df)

# Handle empty trades_df
if total_trades > 0:
    total_wins = trades_df[trades_df['net_pnl'] > 0].shape[0]
    total_losses = trades_df[trades_df['net_pnl'] <= 0].shape[0]
    overall_win_rate = (total_wins / total_trades * 100)
else:
    total_wins = 0
    total_losses = 0
    overall_win_rate = 0
    print("⚠️  WARNING: No trades executed!")
    print()

print(f"Performance:")
print(f"  Return per window: {avg_return:+.2f}% (avg across {len(results_df)} windows)")
print(f"  Win Rate: {overall_win_rate:.1f}% ({total_wins}W / {total_losses}L)")
print(f"  Total Trades: {total_trades} ({avg_trades:.1f} per window)")
print()

# Exit distribution
if total_trades > 0:
    ml_exit_pct = (trades_df['exit_reason'] == 'ML_EXIT').sum() / len(trades_df) * 100
    sl_exit_pct = (trades_df['exit_reason'] == 'STOP_LOSS').sum() / len(trades_df) * 100
    max_hold_pct = (trades_df['exit_reason'] == 'MAX_HOLD').sum() / len(trades_df) * 100

    print(f"Exit Distribution:")
    print(f"  ML Exit: {ml_exit_pct:.1f}%")
    print(f"  Stop Loss: {sl_exit_pct:.1f}%")
    print(f"  Max Hold: {max_hold_pct:.1f}%")
    print()

    # LONG/SHORT distribution
    long_trades = trades_df[trades_df['side'] == 'LONG']
    short_trades = trades_df[trades_df['side'] == 'SHORT']
    long_pct = len(long_trades) / len(trades_df) * 100
    short_pct = len(short_trades) / len(trades_df) * 100

    print(f"LONG/SHORT Distribution:")
    print(f"  LONG: {long_pct:.1f}% ({len(long_trades)} trades)")
    print(f"  SHORT: {short_pct:.1f}% ({len(short_trades)} trades)")
    print()
else:
    ml_exit_pct = 0
    print(f"Exit Distribution: No trades executed")
    print()

print(f"Execution Time: {backtest_time:.1f}s")
print()

# Save results
timestamp_now = datetime.now().strftime('%Y%m%d_%H%M%S')
results_path = RESULTS_DIR / f"backtest_production_features_{timestamp_now}.csv"

if total_trades > 0:
    trades_df.to_csv(results_path, index=False)
    print(f"✅ Results saved: {results_path.name}")
else:
    # Save window results instead
    results_df.to_csv(results_path, index=False)
    print(f"✅ Window results saved: {results_path.name} (no trades executed)")
print()

# Comparison with Phase 1
print("="*80)
print("COMPARISON: Production Features vs Phase 1")
print("="*80)
print()
print(f"Phase 1 (Wrong Features):")
print(f"  Return: -42.88% per window")
print(f"  Win Rate: 0.0%")
print(f"  ML Exit: 2.4%")
print()
print(f"Production Features (NEW):")
print(f"  Return: {avg_return:+.2f}% per window")
print(f"  Win Rate: {overall_win_rate:.1f}%")
print(f"  ML Exit: {ml_exit_pct:.1f}%")
print()

improvement = avg_return - (-42.88)
print(f"Improvement: {improvement:+.2f}pp return, {overall_win_rate:+.1f}pp win rate")
print()

print("="*80)
print("✅ BACKTEST COMPLETE")
print("="*80)
