"""
Production Settings Backtest: 30-Day Test with -3% Stop Loss
=============================================================

Tests production configuration with optimized Stop Loss:
- Entry Models: Trade-Outcome Full Dataset (2025-10-18)
- Exit Models: Opportunity Gating (2025-10-17)
- 4x Leverage with Dynamic Sizing (20-95%)
- **Stop Loss: -3% (OPTIMIZED from -6%)**

Validates the SL optimization findings.
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
print("PRODUCTION SETTINGS BACKTEST - 30 DAY TEST (SL = -3%)")
print("="*80)
print(f"\nTesting OPTIMIZED Stop Loss configuration\n")

# =============================================================================
# PRODUCTION CONFIGURATION
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
EMERGENCY_STOP_LOSS = -0.03  # ⭐ -3% of total balance (OPTIMIZED)

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
# LOAD AND PREPARE DATA - LAST 30 DAYS
# =============================================================================

print("Loading historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_full):,} candles\n")

THIRTY_DAYS_CANDLES = 30 * 288
BUFFER_CANDLES = 200
TOTAL_CANDLES_NEEDED = THIRTY_DAYS_CANDLES + BUFFER_CANDLES

print(f"Using last 30 days of data...")
df_recent = df_full.tail(TOTAL_CANDLES_NEEDED).reset_index(drop=True)
print(f"  ✅ Selected {len(df_recent):,} recent candles")
print(f"     Date range: {df_recent['timestamp'].iloc[0]} to {df_recent['timestamp'].iloc[-1]}\n")

print("Calculating features...")
start_time = time.time()
df = calculate_all_features(df_recent)
df = prepare_exit_features(df)
feature_time = time.time() - start_time

df = df.dropna().reset_index(drop=True)
print(f"  ✅ Features calculated ({feature_time:.1f}s)")
print(f"     {len(df):,} candles after feature calculation\n")

df_test = df.tail(THIRTY_DAYS_CANDLES).reset_index(drop=True)
print(f"  ✅ Final test period: {len(df_test):,} candles (30 days)")
print(f"     Test period: {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}\n")

# Pre-calculate signals (vectorized)
print("Pre-calculating signals...")
signal_start = time.time()
try:
    long_feat = df_test[long_feature_columns].values
    long_feat_scaled = long_scaler.transform(long_feat)
    long_probs_array = long_model.predict_proba(long_feat_scaled)[:, 1]
    df_test['long_prob'] = long_probs_array

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

def backtest_30days(test_df):
    """
    Production Settings Backtest for 30 days with -3% SL
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL
    recent_trades = []

    # Track daily performance
    daily_capital = []
    daily_returns = []
    candles_per_day = 288

    # Track capital curve
    capital_curve = []
    timestamps = []

    for i in range(len(test_df) - 1):
        current_price = test_df['close'].iloc[i]
        current_timestamp = test_df['timestamp'].iloc[i]

        # Entry logic
        if position is None:
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
                    'entry_timestamp': current_timestamp,
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
                        'entry_timestamp': current_timestamp,
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

            # 2. Emergency Stop Loss (-3%)
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
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': current_timestamp,
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

                position = None

        # Track capital curve
        capital_curve.append(capital)
        timestamps.append(current_timestamp)

        # Track daily capital
        if (i + 1) % candles_per_day == 0:
            day_num = (i + 1) // candles_per_day
            daily_capital.append(capital)
            if len(daily_capital) > 1:
                prev_capital = daily_capital[-2]
                day_return = ((capital - prev_capital) / prev_capital) * 100
                daily_returns.append(day_return)

    # Close any remaining position
    if position is not None:
        current_price = test_df['close'].iloc[-1]
        current_timestamp = test_df['timestamp'].iloc[-1]
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
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': current_timestamp,
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
        'daily_returns': daily_returns,
        'capital_curve': capital_curve,
        'timestamps': timestamps
    }


# =============================================================================
# RUN BACKTEST
# =============================================================================

print("="*80)
print("RUNNING 30-DAY BACKTEST (SL = -3%)")
print("="*80)
print()

backtest_start = time.time()
result = backtest_30days(df_test)
backtest_time = time.time() - backtest_start

print(f"  ✅ Backtest complete ({backtest_time:.1f}s)\n")

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

print("="*80)
print("30-DAY BACKTEST RESULTS (SL = -3%)")
print("="*80)

trades = result['trades']
total_trades = result['total_trades']
final_capital = result['final_capital']
return_pct = result['return_pct']

# Win rate
winners = [t for t in trades if t['pnl_net'] > 0]
losers = [t for t in trades if t['pnl_net'] <= 0]
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
avg_winner = np.mean([t['pnl_net'] for t in winners]) if winners else 0
avg_loser = np.mean([t['pnl_net'] for t in losers]) if losers else 0
avg_hold_time = np.mean([t['hold_time'] for t in trades]) if trades else 0
avg_hold_hours = avg_hold_time * 5 / 60

# Daily performance
daily_returns = result['daily_returns']
avg_daily_return = np.mean(daily_returns) if daily_returns else 0
daily_volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0

# Maximum drawdown
capital_curve = np.array(result['capital_curve'])
running_max = np.maximum.accumulate(capital_curve)
drawdown = (capital_curve - running_max) / running_max * 100
max_drawdown = np.min(drawdown)

# Sharpe Ratio (daily)
if len(daily_returns) > 1 and daily_volatility > 0:
    sharpe_daily = avg_daily_return / daily_volatility
else:
    sharpe_daily = 0

print(f"\n📊 Overall Performance:")
print(f"   Test Period: 30 days ({THIRTY_DAYS_CANDLES} candles)")
print(f"   Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"   Final Capital: ${final_capital:,.2f}")
print(f"   Total Return: {return_pct:,.2f}% ⭐")
print(f"   Max Drawdown: {max_drawdown:.2f}%")
print(f"   Avg Daily Return: {avg_daily_return:.2f}%")
print(f"   Daily Volatility: {daily_volatility:.2f}%")
print(f"   Sharpe Ratio (daily): {sharpe_daily:.3f}")

print(f"\n📈 Trading Activity:")
print(f"   Total Trades: {total_trades}")
print(f"   Trades per Day: {total_trades/30:.1f}")
print(f"   Win Rate: {win_rate:.1f}% ({len(winners)}/{total_trades})")
print(f"   Avg P&L per Trade: ${avg_pnl:.2f}")
print(f"   Avg Winner: ${avg_winner:.2f}")
print(f"   Avg Loser: ${avg_loser:.2f}")
print(f"   Profit Factor: {abs(avg_winner / avg_loser):.2f}x" if avg_loser != 0 else "   Profit Factor: N/A")
print(f"   Avg Hold Time: {avg_hold_hours:.1f} hours")

print(f"\n⚖️ Trade Distribution:")
print(f"   LONG: {len(long_trades)} ({long_pct:.1f}%)")
print(f"   SHORT: {len(short_trades)} ({short_pct:.1f}%)")

print(f"\n🚪 Exit Reasons:")
for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(trades) * 100
    print(f"   {reason}: {count} ({pct:.1f}%)")

# Stop Loss analysis
sl_trades = [t for t in trades if t['exit_reason'] == 'emergency_stop_loss']
sl_count = len(sl_trades)
sl_rate = sl_count / len(trades) * 100 if trades else 0
sl_total_loss = sum([t['pnl_net'] for t in sl_trades])

print(f"\n🛡️ Stop Loss Analysis (SL = -3%):")
print(f"   SL Triggers: {sl_count} ({sl_rate:.1f}%) ⭐")
print(f"   Total SL Loss: ${sl_total_loss:.2f}")
if sl_trades:
    print(f"   Avg SL Loss: ${sl_total_loss/sl_count:.2f}")
    print(f"   Largest SL Loss: ${min([t['pnl_net'] for t in sl_trades]):.2f}")

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
    print(f"   Date: {best_trade['entry_timestamp']}")
    print(f"   Exit: {best_trade['exit_reason']}, Hold: {best_trade['hold_time']*5/60:.1f} hours")

    print(f"\n📉 Worst Trade:")
    print(f"   {worst_trade['side']}: ${worst_trade['pnl_net']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
    print(f"   Date: {worst_trade['entry_timestamp']}")
    print(f"   Exit: {worst_trade['exit_reason']}, Hold: {worst_trade['hold_time']*5/60:.1f} hours")

# Day 15-16 specific analysis
print(f"\n📅 Day 15-16 Performance (Oct 15-16):")
day15_start = pd.to_datetime('2025-10-15 00:00:00')
day16_end = pd.to_datetime('2025-10-16 23:59:59')

day15_16_trades = [t for t in trades if
                   pd.to_datetime(t['entry_timestamp']) <= day16_end and
                   pd.to_datetime(t['exit_timestamp']) >= day15_start]

if day15_16_trades:
    day15_16_pnl = sum([t['pnl_net'] for t in day15_16_trades])
    day15_16_winners = len([t for t in day15_16_trades if t['pnl_net'] > 0])
    print(f"   Trades: {len(day15_16_trades)}")
    print(f"   Total P&L: ${day15_16_pnl:.2f} ⭐")
    print(f"   Win Rate: {day15_16_winners/len(day15_16_trades)*100:.1f}%")

# Projected returns
days_tested = 30
monthly_return = return_pct
annualized_return = ((1 + return_pct/100) ** (365/days_tested) - 1) * 100

print(f"\n📊 Projections:")
print(f"   Monthly Return (30 days): {monthly_return:.2f}% ⭐")
print(f"   Annualized Return: {annualized_return:,.0f}% (theoretical)")
print(f"   ⚠️ Note: Past performance is not indicative of future results")

print(f"\n" + "="*80)
print("30-DAY BACKTEST COMPLETE (SL = -3%)")
print("="*80)
print(f"\nTotal Execution Time: {(time.time() - start_time):.1f}s")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# Save trades to CSV
trades_df = pd.DataFrame(trades)
output_path = RESULTS_DIR / f"backtest_30days_sl3pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
trades_df.to_csv(output_path, index=False)
print(f"💾 Trades saved to: {output_path}")
