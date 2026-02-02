"""
Analyze LONG vs SHORT Performance
==================================

Analyze current production model performance by direction.

Configuration (Current Production):
  Entry: Walk-Forward Decoupled (20251027_194313)
    - LONG Threshold: 0.75
    - SHORT Threshold: 0.75
  Exit: threshold_075 (20251027_190512)
    - ML Exit Threshold: 0.75 (both)
  Stop Loss: -3% balance
  Max Hold: 10 hours
  Leverage: 4x
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

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("LONG vs SHORT PERFORMANCE ANALYSIS")
print("="*80)
print(f"\n📊 Current Production Configuration:\n")
print(f"  Entry: Walk-Forward Decoupled (0.75/0.75)")
print(f"  Exit: ML Exit (0.75/0.75) + Emergency SL/MaxHold")
print(f"  Leverage: 4x, Dynamic Position Sizing\n")

# Strategy Parameters (PRODUCTION)
LEVERAGE = 4
LONG_THRESHOLD = 0.75
SHORT_THRESHOLD = 0.75
GATE_THRESHOLD = 0.001

# Exit Parameters (PRODUCTION)
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours
EMERGENCY_STOP_LOSS = -0.03  # -3% balance

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees
TAKER_FEE = 0.0005  # 0.05%

# Load Entry Models (Walk-Forward Decoupled)
print("Loading Entry models (Walk-Forward Decoupled - 20251027_194313)...")
import joblib

long_model_path = MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

# Load Exit Models
print("Loading Exit models (threshold_075 - 20251027_190512)...")

long_exit_model_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ Models loaded successfully\n")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Load FULL data
print("Loading FULL historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Full data loaded: {len(df_full):,} candles\n")

# Calculate features
print("Calculating features...")
start_time = time.time()
df = calculate_all_features_enhanced_v2(df_full, phase='phase1')
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
    print(f"  ❌ Error: {e}")
    sys.exit(1)


def backtest_with_trade_details(window_df):
    """Backtest with detailed trade information"""
    trades = []
    position = None
    capital = INITIAL_CAPITAL
    first_exit_signal_received = False

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Check for first exit signal
        if not first_exit_signal_received and position is None:
            try:
                long_exit_feat = window_df[long_exit_feature_columns].iloc[i:i+1].values
                long_exit_scaled = long_exit_scaler.transform(long_exit_feat)
                long_exit_prob = long_exit_model.predict_proba(long_exit_scaled)[0][1]

                short_exit_feat = window_df[short_exit_feature_columns].iloc[i:i+1].values
                short_exit_scaled = short_exit_scaler.transform(short_exit_feat)
                short_exit_prob = short_exit_model.predict_proba(short_exit_scaled)[0][1]

                if long_exit_prob >= ML_EXIT_THRESHOLD_LONG or short_exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                    first_exit_signal_received = True
            except:
                pass

        # Entry logic
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
            else:
                pnl_usd = entry_notional - current_notional
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = price_change_pct * LEVERAGE
            balance_loss_pct = pnl_usd / capital

            should_exit = False
            exit_reason = None

            # ML Exit
            try:
                if position['side'] == 'LONG':
                    exit_feat = window_df[long_exit_feature_columns].iloc[i:i+1].values
                    exit_scaled = long_exit_scaler.transform(exit_feat)
                    exit_prob = long_exit_model.predict_proba(exit_scaled)[0][1]
                    ml_threshold = ML_EXIT_THRESHOLD_LONG
                else:
                    exit_feat = window_df[short_exit_feature_columns].iloc[i:i+1].values
                    exit_scaled = short_exit_scaler.transform(exit_feat)
                    exit_prob = short_exit_model.predict_proba(exit_scaled)[0][1]
                    ml_threshold = ML_EXIT_THRESHOLD_SHORT

                if exit_prob >= ml_threshold:
                    should_exit = True
                    exit_reason = f'ml_exit_{position["side"].lower()}'
            except:
                pass

            # Emergency Stop Loss
            if not should_exit and balance_loss_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'emergency_stop_loss'

            # Emergency Max Hold
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'emergency_max_hold'

            if should_exit:
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission

                net_pnl_usd = pnl_usd - total_commission
                capital += net_pnl_usd

                trade = {
                    'side': position['side'],
                    'pnl_pct': price_change_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'commission_usd': total_commission,
                    'net_pnl_usd': net_pnl_usd,
                    'hold_time': time_in_pos,
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob']
                }

                trades.append(trade)
                position = None

    return trades, capital


def run_analysis():
    """Run backtest and analyze LONG vs SHORT performance"""
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    all_trades = []

    print("="*80)
    print("RUNNING BACKTEST...")
    print("="*80)
    print()

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        trades, _ = backtest_with_trade_details(window_df)
        all_trades.extend(trades)

    if len(all_trades) == 0:
        print("❌ No trades executed")
        return

    # Analyze by direction
    all_trades_df = pd.DataFrame(all_trades)

    long_trades = all_trades_df[all_trades_df['side'] == 'LONG']
    short_trades = all_trades_df[all_trades_df['side'] == 'SHORT']

    print(f"\n{'='*80}")
    print("LONG vs SHORT PERFORMANCE COMPARISON")
    print(f"{'='*80}\n")

    # Overall stats
    print(f"📊 Overall Statistics:")
    print(f"  Total Trades: {len(all_trades_df)}")
    print(f"  LONG Trades: {len(long_trades)} ({len(long_trades)/len(all_trades_df)*100:.1f}%)")
    print(f"  SHORT Trades: {len(short_trades)} ({len(short_trades)/len(all_trades_df)*100:.1f}%)")

    # LONG Performance
    if len(long_trades) > 0:
        long_winners = long_trades[long_trades['leveraged_pnl_pct'] > 0]
        long_losers = long_trades[long_trades['leveraged_pnl_pct'] <= 0]

        print(f"\n📈 LONG Performance:")
        print(f"  Win Rate: {len(long_winners)/len(long_trades)*100:.2f}% ({len(long_winners)}W / {len(long_losers)}L)")
        print(f"  Avg Return: {long_trades['leveraged_pnl_pct'].mean()*100:.2f}%")
        print(f"  Avg Winner: {long_winners['leveraged_pnl_pct'].mean()*100:.2f}%" if len(long_winners) > 0 else "  Avg Winner: N/A")
        print(f"  Avg Loser: {long_losers['leveraged_pnl_pct'].mean()*100:.2f}%" if len(long_losers) > 0 else "  Avg Loser: N/A")
        print(f"  Total P&L: ${long_trades['net_pnl_usd'].sum():.2f}")
        print(f"  Avg Hold Time: {long_trades['hold_time'].mean():.1f} candles ({long_trades['hold_time'].mean()/12:.1f}h)")
        print(f"  Avg Position Size: {long_trades['position_size_pct'].mean()*100:.1f}%")

        # Exit reasons for LONG
        print(f"\n  Exit Reasons:")
        long_exit_counts = long_trades['exit_reason'].value_counts()
        for reason, count in long_exit_counts.items():
            pct = count / len(long_trades) * 100
            reason_name = "ML Exit" if reason == 'ml_exit_long' else \
                         "Stop Loss" if reason == 'emergency_stop_loss' else \
                         "Max Hold" if reason == 'emergency_max_hold' else reason
            print(f"    {reason_name:<20s}: {count:>4d} ({pct:>5.1f}%)")

    # SHORT Performance
    if len(short_trades) > 0:
        short_winners = short_trades[short_trades['leveraged_pnl_pct'] > 0]
        short_losers = short_trades[short_trades['leveraged_pnl_pct'] <= 0]

        print(f"\n📉 SHORT Performance:")
        print(f"  Win Rate: {len(short_winners)/len(short_trades)*100:.2f}% ({len(short_winners)}W / {len(short_losers)}L)")
        print(f"  Avg Return: {short_trades['leveraged_pnl_pct'].mean()*100:.2f}%")
        print(f"  Avg Winner: {short_winners['leveraged_pnl_pct'].mean()*100:.2f}%" if len(short_winners) > 0 else "  Avg Winner: N/A")
        print(f"  Avg Loser: {short_losers['leveraged_pnl_pct'].mean()*100:.2f}%" if len(short_losers) > 0 else "  Avg Loser: N/A")
        print(f"  Total P&L: ${short_trades['net_pnl_usd'].sum():.2f}")
        print(f"  Avg Hold Time: {short_trades['hold_time'].mean():.1f} candles ({short_trades['hold_time'].mean()/12:.1f}h)")
        print(f"  Avg Position Size: {short_trades['position_size_pct'].mean()*100:.1f}%")

        # Exit reasons for SHORT
        print(f"\n  Exit Reasons:")
        short_exit_counts = short_trades['exit_reason'].value_counts()
        for reason, count in short_exit_counts.items():
            pct = count / len(short_trades) * 100
            reason_name = "ML Exit" if reason == 'ml_exit_short' else \
                         "Stop Loss" if reason == 'emergency_stop_loss' else \
                         "Max Hold" if reason == 'emergency_max_hold' else reason
            print(f"    {reason_name:<20s}: {count:>4d} ({pct:>5.1f}%)")

    # Comparison
    if len(long_trades) > 0 and len(short_trades) > 0:
        print(f"\n⚖️  Comparison:")
        print(f"  LONG Win Rate: {len(long_trades[long_trades['leveraged_pnl_pct'] > 0])/len(long_trades)*100:.2f}%")
        print(f"  SHORT Win Rate: {len(short_trades[short_trades['leveraged_pnl_pct'] > 0])/len(short_trades)*100:.2f}%")
        print(f"  LONG Avg Return: {long_trades['leveraged_pnl_pct'].mean()*100:.2f}%")
        print(f"  SHORT Avg Return: {short_trades['leveraged_pnl_pct'].mean()*100:.2f}%")
        print(f"  LONG Total P&L: ${long_trades['net_pnl_usd'].sum():.2f}")
        print(f"  SHORT Total P&L: ${short_trades['net_pnl_usd'].sum():.2f}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


# Run analysis
start_time = time.time()
run_analysis()
analysis_time = time.time() - start_time
print(f"Analysis completed in {analysis_time:.1f}s")
