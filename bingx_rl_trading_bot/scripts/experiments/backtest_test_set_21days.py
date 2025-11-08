"""
Test Set Backtest: 21-Day Out-of-Sample Validation
==================================================

Tests EXACT production configuration on COMPLETE TEST SET (out-of-sample):
- Test Period: 2025-09-27 ~ 2025-10-18 (21 days)
- Entry Models: Trade-Outcome Full Dataset (2025-10-18)
- Exit Models: Opportunity Gating (2025-10-17)
- 4x Leverage with Dynamic Sizing (20-95%)
- Opportunity Gating (gate = 0.001)

**IMPORTANT**: This uses 100% out-of-sample data (Test Set from 8:2 split)
Models were trained on 2025-07-01 ~ 2025-09-27 (Train Set 80%)
This backtest uses 2025-09-27 ~ 2025-10-18 (Test Set 20%)

Entry signals accepted AFTER first exit signal occurs.

Configuration matches opportunity_gating_bot_4x.py exactly.
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
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("TEST SET BACKTEST - 21 DAY OUT-OF-SAMPLE VALIDATION")
print("="*80)
print(f"\nTesting EXACT production configuration on COMPLETE TEST SET")
print(f"Period: 2025-09-27 ~ 2025-10-18 (100% out-of-sample data)\n")

# =============================================================================
# PRODUCTION CONFIGURATION (matches opportunity_gating_bot_4x.py)
# =============================================================================

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.70
ML_EXIT_THRESHOLD_SHORT = 0.72
EMERGENCY_MAX_HOLD_TIME = 96  # 8 hours (96 candles × 5min)
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Backtest Parameters
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005  # 0.05% per trade

# =============================================================================
# LOAD PRODUCTION MODELS (Trade-Outcome Full Dataset - Oct 18)
# =============================================================================

print("Loading PRODUCTION Entry models (Trade-Outcome Full Dataset - Oct 18)...")
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
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ LONG Entry: {len(long_feature_columns)} features")
print(f"  ✅ SHORT Entry: {len(short_feature_columns)} features\n")

# Load Exit Models (Opportunity Gating - Oct 17)
print("Loading Exit models (Opportunity Gating - Oct 17)...")
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

print(f"  ✅ LONG Exit: {len(long_exit_feature_columns)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_feature_columns)} features\n")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# =============================================================================
# LOAD AND PREPARE DATA - TEST SET (OUT-OF-SAMPLE)
# =============================================================================

print("Loading historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_full):,} candles\n")

# Calculate Train/Test split (8:2)
TRAIN_TEST_SPLIT = 0.8
split_idx = int(len(df_full) * TRAIN_TEST_SPLIT)

print(f"📊 Train/Test Split (8:2):")
print(f"   Train Set: 0 ~ {split_idx:,} candles (80%)")
print(f"   Test Set: {split_idx:,} ~ {len(df_full):,} candles (20%)")
print()

# Use ONLY Test Set (out-of-sample data)
# Add buffer before split point for feature calculation
BUFFER_CANDLES = 200
start_idx = max(0, split_idx - BUFFER_CANDLES)

df_test_with_buffer = df_full.iloc[start_idx:].reset_index(drop=True)
print(f"Using Test Set with buffer for feature calculation:")
print(f"  - Buffer start: {df_full['timestamp'].iloc[start_idx]}")
print(f"  - Test Set start (split point): {df_full['timestamp'].iloc[split_idx]}")
print(f"  - Test Set end: {df_full['timestamp'].iloc[-1]}")
print(f"  - Total candles (with buffer): {len(df_test_with_buffer):,}")
print()

df_recent = df_test_with_buffer
print(f"  ✅ Selected Test Set data")
print(f"     Date range: {df_recent['timestamp'].iloc[0]} to {df_recent['timestamp'].iloc[-1]}\n")

print("Calculating features...")
start_time = time.time()
df = calculate_all_features(df_recent)
df = prepare_exit_features(df)
feature_time = time.time() - start_time

# Drop NaN rows from feature calculation
df = df.dropna().reset_index(drop=True)
print(f"  ✅ Features calculated ({feature_time:.1f}s)")
print(f"     {len(df):,} candles after feature calculation\n")

# Extract only Test Set period (remove buffer)
# Find the index where Test Set starts (2025-09-27 01:10:00)
df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
test_set_start = pd.to_datetime(df_full['timestamp'].iloc[split_idx])
df_test = df[df['timestamp_dt'] >= test_set_start].reset_index(drop=True)

print(f"  ✅ Final test period: {len(df_test):,} candles (Test Set only)")
print(f"     Test period: {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}")
print(f"     Duration: {(df_test['timestamp_dt'].iloc[-1] - df_test['timestamp_dt'].iloc[0]).days} days\n")

# Pre-calculate signals (vectorized)
print("Pre-calculating signals...")
signal_start = time.time()
try:
    # LONG signals
    long_feat = df_test[long_feature_columns].values
    long_feat_scaled = long_scaler.transform(long_feat)
    long_probs_array = long_model.predict_proba(long_feat_scaled)[:, 1]
    df_test['long_prob'] = long_probs_array

    # SHORT signals
    short_feat = df_test[short_feature_columns].values
    short_feat_scaled = short_scaler.transform(short_feat)
    short_probs_array = short_model.predict_proba(short_feat_scaled)[:, 1]
    df_test['short_prob'] = short_probs_array

    signal_time = time.time() - signal_start
    print(f"  ✅ Signals pre-calculated ({signal_time:.1f}s)\n")
except Exception as e:
    print(f"  ❌ Error pre-calculating signals: {e}")
    df_test['long_prob'] = 0.0
    df_test['short_prob'] = 0.0
    print(f"  ⚠️ Using zero probabilities\n")


# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_7days(test_df):
    """
    Production Settings Backtest for 7 days (Entry After First Exit Signal)

    NEW BEHAVIOR: Entry signals only accepted AFTER first exit signal occurs.
    This simulates real bot startup where exit model must be ready before trading.

    Exact configuration from opportunity_gating_bot_4x.py
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL
    recent_trades = []
    first_exit_signal_received = False  # NEW: Track first exit signal

    # Track daily performance
    daily_capital = []
    daily_returns = []
    candles_per_day = 288

    for i in range(len(test_df) - 1):
        current_price = test_df['close'].iloc[i]

        # NEW: Check for exit signal even without position (to detect first exit signal)
        if not first_exit_signal_received and position is None:
            try:
                # Check LONG exit signal
                exit_features_values = test_df[long_exit_feature_columns].iloc[i:i+1].values
                exit_features_scaled = long_exit_scaler.transform(exit_features_values)
                long_exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]

                # Check SHORT exit signal
                exit_features_values = test_df[short_exit_feature_columns].iloc[i:i+1].values
                exit_features_scaled = short_exit_scaler.transform(exit_features_values)
                short_exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]

                # If either model shows exit signal, mark as ready
                if long_exit_prob >= ML_EXIT_THRESHOLD_LONG or short_exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                    first_exit_signal_received = True
                    print(f"  ✅ First exit signal detected at candle {i} (LONG: {long_exit_prob:.3f}, SHORT: {short_exit_prob:.3f})")
                    print(f"     Entry signals now ENABLED from this point forward\n")
            except:
                pass

        # Entry logic (ONLY after first exit signal)
        if first_exit_signal_received and position is None:
            long_prob = test_df['long_prob'].iloc[i]
            short_prob = test_df['short_prob'].iloc[i]

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
                    'quantity': sizing_result['leveraged_value'] / current_price,
                    'entry_prob': long_prob
                }

            # SHORT entry (gated)
            elif short_prob >= SHORT_THRESHOLD:
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
                        'position_size_pct': sizing_result['position_size_pct'],
                        'position_value': sizing_result['position_value'],
                        'leveraged_value': sizing_result['leveraged_value'],
                        'quantity': sizing_result['leveraged_value'] / current_price,
                        'entry_prob': short_prob
                    }

        # Exit logic
        if position is not None:
            time_in_pos = i - position['entry_idx']
            current_price = test_df['close'].iloc[i]

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

            # 1. ML Exit (PRIMARY)
            try:
                if position['side'] == 'LONG':
                    exit_model = long_exit_model
                    exit_scaler = long_exit_scaler
                    exit_features_list = long_exit_feature_columns
                    ml_threshold = ML_EXIT_THRESHOLD_LONG
                else:
                    exit_model = short_exit_model
                    exit_scaler = short_exit_scaler
                    exit_features_list = short_exit_feature_columns
                    ml_threshold = ML_EXIT_THRESHOLD_SHORT

                exit_features_values = test_df[exit_features_list].iloc[i:i+1].values
                exit_features_scaled = exit_scaler.transform(exit_features_values)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                if exit_prob >= ml_threshold:
                    should_exit = True
                    exit_reason = f'ml_exit_{position["side"].lower()}'
            except:
                pass

            # 2. Emergency Stop Loss (-6% of balance)
            if not should_exit and balance_loss_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'emergency_stop_loss'

            # 3. Emergency Max Hold (8 hours)
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'emergency_max_hold'

            if should_exit:
                # Calculate fees
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission

                # Net P&L
                pnl_net = pnl_usd - total_commission

                # Update capital
                capital += pnl_net

                # Record trade
                trade = {
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position.get('entry_prob', 0),
                    'position_size_pct': position['position_size_pct'],
                    'pnl_gross': pnl_usd,
                    'pnl_net': pnl_net,
                    'pnl_pct': (pnl_net / position['position_value']) * 100,
                    'exit_reason': exit_reason,
                    'hold_time': time_in_pos,
                    'fees': total_commission,
                    'candle_idx': i
                }
                trades.append(trade)

                # Update recent trades (for streak calculation)
                recent_trades.append({'pnl_net': pnl_net})
                if len(recent_trades) > 5:
                    recent_trades.pop(0)

                position = None

        # Track daily capital
        if (i + 1) % candles_per_day == 0:
            day_num = (i + 1) // candles_per_day
            daily_capital.append(capital)
            if day_num > 1:
                prev_capital = daily_capital[-2]
                day_return = ((capital - prev_capital) / prev_capital) * 100
                daily_returns.append(day_return)

    # Close any remaining position
    if position is not None:
        current_price = test_df['close'].iloc[-1]
        entry_notional = position['quantity'] * position['entry_price']
        current_notional = position['quantity'] * current_price

        if position['side'] == 'LONG':
            pnl_usd = current_notional - entry_notional
        else:
            pnl_usd = entry_notional - current_notional

        entry_commission = position['leveraged_value'] * TAKER_FEE
        exit_commission = position['quantity'] * current_price * TAKER_FEE
        pnl_net = pnl_usd - entry_commission - exit_commission

        capital += pnl_net

        trade = {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'entry_prob': position.get('entry_prob', 0),
            'position_size_pct': position['position_size_pct'],
            'pnl_gross': pnl_usd,
            'pnl_net': pnl_net,
            'pnl_pct': (pnl_net / position['position_value']) * 100,
            'exit_reason': 'end_of_test',
            'hold_time': len(test_df) - position['entry_idx'],
            'fees': entry_commission + exit_commission,
            'candle_idx': len(test_df) - 1
        }
        trades.append(trade)

    return {
        'final_capital': capital,
        'return_pct': ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100,
        'trades': trades,
        'total_trades': len(trades),
        'daily_capital': daily_capital,
        'daily_returns': daily_returns
    }


# =============================================================================
# RUN BACKTEST
# =============================================================================

print("="*80)
print("RUNNING TEST SET BACKTEST (OUT-OF-SAMPLE)")
print("="*80)
print()

backtest_start = time.time()
result = backtest_7days(df_test)
backtest_time = time.time() - backtest_start

print(f"  ✅ Backtest complete ({backtest_time:.1f}s)\n")

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

print("="*80)
print("TEST SET BACKTEST RESULTS - 21 DAYS (OUT-OF-SAMPLE)")
print("="*80)

trades = result['trades']
total_trades = result['total_trades']
final_capital = result['final_capital']
return_pct = result['return_pct']

# Win rate
winners = [t for t in trades if t['pnl_net'] > 0]
win_rate = len(winners) / len(trades) * 100 if trades else 0

# Side distribution
long_trades = [t for t in trades if t['side'] == 'LONG']
short_trades = [t for t in trades if t['side'] == 'SHORT']
long_pct = len(long_trades) / len(trades) * 100 if trades else 0
short_pct = len(short_trades) / len(trades) * 100 if trades else 0

# Exit reason distribution
exit_reasons = {}
for t in trades:
    reason = t['exit_reason']
    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

# Average trade metrics
avg_pnl = np.mean([t['pnl_net'] for t in trades]) if trades else 0
avg_hold_time = np.mean([t['hold_time'] for t in trades]) if trades else 0
avg_hold_hours = avg_hold_time * 5 / 60  # Convert 5-min candles to hours

# Daily performance
daily_returns = result['daily_returns']
avg_daily_return = np.mean(daily_returns) if daily_returns else 0
daily_volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0

test_days = (df_test['timestamp_dt'].iloc[-1] - df_test['timestamp_dt'].iloc[0]).days

print(f"\n📊 Overall Performance:")
print(f"   Test Period: {test_days} days ({len(df_test):,} candles) - OUT-OF-SAMPLE")
print(f"   Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"   Final Capital: ${final_capital:,.2f}")
print(f"   Total Return: {return_pct:,.2f}%")
print(f"   Avg Daily Return: {avg_daily_return:.2f}%")
print(f"   Daily Volatility: {daily_volatility:.2f}%")

print(f"\n📈 Trading Activity:")
print(f"   Total Trades: {total_trades}")
print(f"   Trades per Day: {total_trades/test_days:.1f}")
print(f"   Win Rate: {win_rate:.1f}% ({len(winners)}/{total_trades})")
print(f"   Avg P&L per Trade: ${avg_pnl:.2f}")
print(f"   Avg Hold Time: {avg_hold_hours:.1f} hours")

print(f"\n⚖️ Trade Distribution:")
print(f"   LONG: {len(long_trades)} ({long_pct:.1f}%)")
print(f"   SHORT: {len(short_trades)} ({short_pct:.1f}%)")

print(f"\n🚪 Exit Reasons:")
for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(trades) * 100
    print(f"   {reason}: {count} ({pct:.1f}%)")

# Average position sizing
avg_position_size = np.mean([t['position_size_pct'] for t in trades]) * 100 if trades else 0
print(f"\n💰 Position Sizing:")
print(f"   Average: {avg_position_size:.1f}%")
print(f"   Range: {min([t['position_size_pct'] for t in trades])*100:.1f}% - {max([t['position_size_pct'] for t in trades])*100:.1f}%")

# Best and worst trades
if trades:
    best_trade = max(trades, key=lambda x: x['pnl_net'])
    worst_trade = min(trades, key=lambda x: x['pnl_net'])

    print(f"\n🏆 Best Trade:")
    print(f"   {best_trade['side']}: ${best_trade['pnl_net']:.2f} ({best_trade['pnl_pct']:.2f}%)")
    print(f"   Exit: {best_trade['exit_reason']}, Hold: {best_trade['hold_time']*5/60:.1f} hours")

    print(f"\n📉 Worst Trade:")
    print(f"   {worst_trade['side']}: ${worst_trade['pnl_net']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
    print(f"   Exit: {worst_trade['exit_reason']}, Hold: {worst_trade['hold_time']*5/60:.1f} hours")

# Daily breakdown
print(f"\n📅 Daily Performance:")
daily_capital = result['daily_capital']
for day in range(len(daily_capital)):
    if day == 0:
        day_return = ((daily_capital[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        print(f"   Day {day+1}: ${daily_capital[day]:,.2f} ({day_return:+.2f}%)")
    else:
        day_return = ((daily_capital[day] - daily_capital[day-1]) / daily_capital[day-1]) * 100
        print(f"   Day {day+1}: ${daily_capital[day]:,.2f} ({day_return:+.2f}%)")

# Projected annualized return (using actual test period)
annualized_return = ((1 + return_pct/100) ** (365/test_days) - 1) * 100

print(f"\n📊 Projections:")
print(f"   If this {test_days}-day Test Set performance continues:")
print(f"   - Monthly Return: {((1 + return_pct/100) ** (30/test_days) - 1) * 100:.1f}%")
print(f"   - Annualized Return: {annualized_return:,.0f}% (theoretical)")
print(f"   ⚠️ Note: Past performance is not indicative of future results")

print(f"\n" + "="*80)
print("TEST SET OUT-OF-SAMPLE VALIDATION COMPLETE")
print("="*80)
print(f"\nTotal Execution Time: {(time.time() - start_time):.1f}s")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
