"""
Production Settings Backtest: Current Live Bot Configuration
=============================================================

Tests EXACT production configuration:
- Entry Models: Trade-Outcome Full Dataset (2025-10-18)
- Exit Models: Opportunity Gating (2025-10-17)
- 4x Leverage with Dynamic Sizing (20-95%)
- Opportunity Gating (gate = 0.001)

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
print("PRODUCTION SETTINGS BACKTEST")
print("="*80)
print(f"\nTesting EXACT production configuration (as of 2025-10-20)\n")

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
EMERGENCY_STOP_LOSS = -0.03  # -3% of total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%, Sharpe: 0.242)

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
# LOAD AND PREPARE DATA
# =============================================================================

print("Loading historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_full):,} candles (~{len(df_full)//288:.0f} days)\n")

print("Calculating features...")
start_time = time.time()
df = calculate_all_features(df_full)
df = prepare_exit_features(df)
feature_time = time.time() - start_time
print(f"  ✅ Features calculated ({feature_time:.1f}s)\n")

# Pre-calculate signals (vectorized)
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


# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_production_settings(window_df):
    """
    Production Settings Backtest

    Exact configuration from opportunity_gating_bot_4x.py
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL
    recent_trades = []

    for i in range(len(window_df) - 1):
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

                exit_features_values = window_df[exit_features_list].iloc[i:i+1].values
                exit_features_scaled = exit_scaler.transform(exit_features_values)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                if exit_prob >= ml_threshold:
                    should_exit = True
                    exit_reason = f'ml_exit_{position["side"].lower()}'
            except:
                pass

            # 2. Emergency Stop Loss (-5% of balance)
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
                    'fees': total_commission
                }
                trades.append(trade)

                # Update recent trades (for streak calculation)
                recent_trades.append({'pnl_net': pnl_net})
                if len(recent_trades) > 5:
                    recent_trades.pop(0)

                position = None

    # Close any remaining position
    if position is not None:
        current_price = window_df['close'].iloc[-1]
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
            'exit_reason': 'end_of_window',
            'hold_time': len(window_df) - position['entry_idx'],
            'fees': entry_commission + exit_commission
        }
        trades.append(trade)

    return {
        'final_capital': capital,
        'return_pct': ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100,
        'trades': trades,
        'total_trades': len(trades)
    }


# =============================================================================
# RUN BACKTEST
# =============================================================================

print("="*80)
print("RUNNING BACKTEST")
print("="*80)

# Use 5-day windows (288 candles per day × 5 = 1440 candles)
WINDOW_SIZE = 1440
n_windows = len(df) // WINDOW_SIZE

results = []
backtest_start = time.time()

for window_idx in range(n_windows):
    start_idx = window_idx * WINDOW_SIZE
    end_idx = start_idx + WINDOW_SIZE
    window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    result = backtest_production_settings(window_df)
    results.append(result)

    # Progress update
    if (window_idx + 1) % 5 == 0:
        elapsed = time.time() - backtest_start
        progress_pct = (window_idx + 1) / n_windows * 100
        print(f"  Progress: {window_idx+1}/{n_windows} windows ({progress_pct:.1f}%) - {elapsed:.1f}s")

backtest_time = time.time() - backtest_start
print(f"\n  ✅ Backtest complete ({backtest_time:.1f}s)\n")

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

print("="*80)
print("BACKTEST RESULTS - PRODUCTION SETTINGS")
print("="*80)

returns = [r['return_pct'] for r in results]
total_trades = sum([r['total_trades'] for r in results])
avg_trades_per_window = total_trades / len(results)

# Collect all trades
all_trades = []
for r in results:
    all_trades.extend(r['trades'])

# Win rate
winners = [t for t in all_trades if t['pnl_net'] > 0]
win_rate = len(winners) / len(all_trades) * 100 if all_trades else 0

# Side distribution
long_trades = [t for t in all_trades if t['side'] == 'LONG']
short_trades = [t for t in all_trades if t['side'] == 'SHORT']
long_pct = len(long_trades) / len(all_trades) * 100 if all_trades else 0
short_pct = len(short_trades) / len(all_trades) * 100 if all_trades else 0

# Exit reason distribution
exit_reasons = {}
for t in all_trades:
    reason = t['exit_reason']
    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

# Sharpe Ratio
returns_array = np.array(returns)
sharpe = np.mean(returns_array) / np.std(returns_array) if len(returns_array) > 1 else 0

print(f"\n📊 Overall Performance:")
print(f"   Windows Tested: {len(results)}")
print(f"   Avg Return: {np.mean(returns):.2f}% per window (5 days)")
print(f"   Median Return: {np.median(returns):.2f}%")
print(f"   Std Dev: {np.std(returns):.2f}%")
print(f"   Sharpe Ratio: {sharpe:.3f}")
print(f"   Best Window: {np.max(returns):.2f}%")
print(f"   Worst Window: {np.min(returns):.2f}%")

print(f"\n📈 Trading Activity:")
print(f"   Total Trades: {len(all_trades)}")
print(f"   Avg Trades/Window: {avg_trades_per_window:.1f}")
print(f"   Win Rate: {win_rate:.1f}%")

print(f"\n⚖️ Trade Distribution:")
print(f"   LONG: {len(long_trades)} ({long_pct:.1f}%)")
print(f"   SHORT: {len(short_trades)} ({short_pct:.1f}%)")

print(f"\n🚪 Exit Reasons:")
for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(all_trades) * 100
    print(f"   {reason}: {count} ({pct:.1f}%)")

# Average position sizing
avg_position_size = np.mean([t['position_size_pct'] for t in all_trades]) * 100
print(f"\n💰 Position Sizing:")
print(f"   Average: {avg_position_size:.1f}%")

# Capital growth projection
total_return_pct = np.mean(returns)
annualized_return = ((1 + total_return_pct/100) ** (365/5) - 1) * 100

print(f"\n📊 Projections:")
print(f"   If compounded continuously:")
print(f"   - Initial: $10,000")
print(f"   - After 1 window (5 days): ${INITIAL_CAPITAL * (1 + total_return_pct/100):,.0f}")
print(f"   - Annualized Return: {annualized_return:,.0f}% (theoretical)")

print(f"\n" + "="*80)
print("PRODUCTION SETTINGS VALIDATION COMPLETE")
print("="*80)
print(f"\nBacktest Time: {backtest_time:.1f}s")
print(f"Total Time: {(time.time() - start_time):.1f}s")
print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
