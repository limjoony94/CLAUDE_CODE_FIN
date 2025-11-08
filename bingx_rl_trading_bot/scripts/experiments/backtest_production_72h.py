"""
Production Settings Backtest: Latest 72 Hours
==============================================

Tests EXACT production configuration on the most recent 72 hours of data.
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
print("PRODUCTION SETTINGS BACKTEST - LATEST 72 HOURS")
print("="*80)
print(f"\nTesting EXACT production configuration on latest 72h data\n")

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
# LOAD PRODUCTION MODELS
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

# Load Exit Models
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

print("Loading latest 72-hour data...")
data_file = DATA_DIR / "BTCUSDT_5m_latest_72h.csv"
df_raw = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_raw):,} candles")
print(f"     First: {df_raw['timestamp'].iloc[0]}")
print(f"     Last: {df_raw['timestamp'].iloc[-1]}\n")

print("Calculating features...")
start_time = time.time()
df = calculate_all_features(df_raw)
df = prepare_exit_features(df)
feature_time = time.time() - start_time
print(f"  ✅ Features calculated ({feature_time:.1f}s)\n")

# Pre-calculate signals
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

def backtest_production_72h(df):
    """Production Settings Backtest on 72-hour data"""
    trades = []
    position = None
    capital = INITIAL_CAPITAL

    for i in range(len(df) - 1):
        current_price = df['close'].iloc[i]

        # Entry logic
        if position is None:
            long_prob = df['long_prob'].iloc[i]
            short_prob = df['short_prob'].iloc[i]

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
                    'entry_time': df['timestamp'].iloc[i],
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
                        'entry_time': df['timestamp'].iloc[i],
                        'position_size_pct': sizing_result['position_size_pct'],
                        'position_value': sizing_result['position_value'],
                        'leveraged_value': sizing_result['leveraged_value'],
                        'quantity': sizing_result['leveraged_value'] / current_price,
                        'entry_prob': short_prob
                    }

        # Exit logic
        if position is not None:
            time_in_pos = i - position['entry_idx']
            current_price = df['close'].iloc[i]

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

                exit_features_values = df[exit_features_list].iloc[i:i+1].values
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
                    'entry_time': position['entry_time'],
                    'exit_time': df['timestamp'].iloc[i],
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
                    'capital_after': capital
                }
                trades.append(trade)
                position = None

    # Close any remaining position
    if position is not None:
        current_price = df['close'].iloc[-1]
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
            'entry_time': position['entry_time'],
            'exit_time': df['timestamp'].iloc[-1],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'entry_prob': position.get('entry_prob', 0),
            'position_size_pct': position['position_size_pct'],
            'pnl_gross': pnl_usd,
            'pnl_net': pnl_net,
            'pnl_pct': (pnl_net / position['position_value']) * 100,
            'exit_reason': 'end_of_window',
            'hold_time': len(df) - position['entry_idx'],
            'fees': entry_commission + exit_commission,
            'capital_after': capital
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
print("RUNNING BACKTEST ON 72-HOUR DATA")
print("="*80)

backtest_start = time.time()
result = backtest_production_72h(df)
backtest_time = time.time() - backtest_start

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

print("\n" + "="*80)
print("BACKTEST RESULTS - PRODUCTION SETTINGS (72 HOURS)")
print("="*80)

trades = result['trades']

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

print(f"\n📊 Overall Performance:")
print(f"   Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"   Final Capital: ${result['final_capital']:,.2f}")
print(f"   Total Return: {result['return_pct']:.2f}%")
print(f"   Net P&L: ${result['final_capital'] - INITIAL_CAPITAL:,.2f}")

print(f"\n📈 Trading Activity:")
print(f"   Total Trades: {len(trades)}")
print(f"   Win Rate: {win_rate:.1f}%")
print(f"   Winners: {len(winners)}")
print(f"   Losers: {len(trades) - len(winners)}")

if trades:
    avg_pnl = np.mean([t['pnl_net'] for t in trades])
    best_trade = max(trades, key=lambda x: x['pnl_net'])
    worst_trade = min(trades, key=lambda x: x['pnl_net'])

    print(f"\n💰 P&L Statistics:")
    print(f"   Average P&L: ${avg_pnl:.2f}")
    print(f"   Best Trade: ${best_trade['pnl_net']:.2f} ({best_trade['side']}, {best_trade['pnl_pct']:.2f}%)")
    print(f"   Worst Trade: ${worst_trade['pnl_net']:.2f} ({worst_trade['side']}, {worst_trade['pnl_pct']:.2f}%)")

print(f"\n⚖️ Trade Distribution:")
print(f"   LONG: {len(long_trades)} ({long_pct:.1f}%)")
print(f"   SHORT: {len(short_trades)} ({short_pct:.1f}%)")

print(f"\n🚪 Exit Reasons:")
for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(trades) * 100 if trades else 0
    print(f"   {reason}: {count} ({pct:.1f}%)")

if trades:
    avg_position_size = np.mean([t['position_size_pct'] for t in trades]) * 100
    avg_hold_time = np.mean([t['hold_time'] for t in trades])
    print(f"\n💼 Position Statistics:")
    print(f"   Average Position Size: {avg_position_size:.1f}%")
    print(f"   Average Hold Time: {avg_hold_time:.1f} candles ({avg_hold_time * 5 / 60:.1f} hours)")

    total_fees = sum([t['fees'] for t in trades])
    print(f"\n💸 Fee Impact:")
    print(f"   Total Fees Paid: ${total_fees:.2f}")

# Trade-by-trade details
if trades:
    print(f"\n" + "="*80)
    print("TRADE-BY-TRADE DETAILS")
    print("="*80)
    for idx, t in enumerate(trades, 1):
        print(f"\n[Trade {idx}] {t['side']}")
        print(f"  Entry: {t['entry_time']} @ ${t['entry_price']:,.2f} (prob: {t['entry_prob']:.3f})")
        print(f"  Exit:  {t['exit_time']} @ ${t['exit_price']:,.2f} ({t['exit_reason']})")
        print(f"  Position: {t['position_size_pct']*100:.1f}%, Hold: {t['hold_time']} candles ({t['hold_time']*5/60:.1f}h)")
        print(f"  P&L: ${t['pnl_net']:+.2f} ({t['pnl_pct']:+.2f}%), Fees: ${t['fees']:.2f}")
        print(f"  Capital: ${t['capital_after']:,.2f}")

print(f"\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
print(f"\nBacktest Time: {backtest_time:.1f}s")
print(f"Total Time: {(time.time() - start_time):.1f}s")
print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
