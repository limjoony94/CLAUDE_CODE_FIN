"""
Backtest Oct 14-19: Production Trade-Outcome Models
====================================================

Test production models (29.06% return, 85.3% win rate) on Oct 14-19 period.

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
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BACKTEST OCT 14-19: PRODUCTION TRADE-OUTCOME MODELS")
print("="*80)
print(f"\nPeriod: 2025-10-14 to 2025-10-19 (5 days)")
print(f"Models: Trade-Outcome Full Dataset (29.06% return, 85.3% win rate)\n")

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.70
ML_EXIT_THRESHOLD_SHORT = 0.72  # Optimized 2025-10-18
EMERGENCY_MAX_HOLD_TIME = 96  # 8 hours
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees (BingX)
TAKER_FEE = 0.0005  # 0.05% per trade

# =============================================================================
# LOAD PRODUCTION MODELS
# =============================================================================
print("Loading PRODUCTION models (Trade-Outcome Full Dataset)...")

# LONG Entry Model
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

# SHORT Entry Model
short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

# LONG Exit Model
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

# SHORT Exit Model
short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ Entry models: LONG({len(long_feature_columns)}), SHORT({len(short_feature_columns)})")
print(f"  ✅ Exit models: LONG({len(long_exit_feature_columns)}), SHORT({len(short_exit_feature_columns)})\n")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# =============================================================================
# LOAD AND FILTER DATA
# =============================================================================
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  Total candles: {len(df_full):,}")

# Filter Oct 14-19 period
print("\nFiltering Oct 14-19 period...")
df_period = df_full[
    (df_full['timestamp'] >= '2025-10-14') &
    (df_full['timestamp'] < '2025-10-20')
].copy()

print(f"  Oct 14-19 candles: {len(df_period):,}")
if len(df_period) > 0:
    print(f"  Time range: {df_period['timestamp'].iloc[0]} to {df_period['timestamp'].iloc[-1]}")
    print(f"  Price range: ${df_period['close'].min():,.2f} - ${df_period['close'].max():,.2f}")
else:
    print("\n❌ No data found for Oct 14-19 period!")
    sys.exit(1)

# Calculate features
print("\nCalculating features...")
start_time = time.time()
df = calculate_all_features(df_full.copy())  # Use full dataset for proper feature calculation
df = prepare_exit_features(df)
feature_time = time.time() - start_time
print(f"  ✅ Features calculated ({feature_time:.1f}s)")

# Filter features for Oct 14-19
df_test = df[
    (df['timestamp'] >= '2025-10-14') &
    (df['timestamp'] < '2025-10-20')
].copy()
print(f"  Test period features: {len(df_test):,} candles\n")

# Pre-calculate signals
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
    print(f"  ✅ Signals pre-calculated ({signal_time:.1f}s)")

    # Signal statistics
    print(f"\n📊 Signal Statistics:")
    print(f"  LONG probabilities:")
    print(f"    Mean: {df_test['long_prob'].mean():.4f} ({df_test['long_prob'].mean()*100:.2f}%)")
    print(f"    ≥65%: {(df_test['long_prob'] >= 0.65).sum():,} candles ({(df_test['long_prob'] >= 0.65).sum()/len(df_test)*100:.1f}%)")
    print(f"  SHORT probabilities:")
    print(f"    Mean: {df_test['short_prob'].mean():.4f} ({df_test['short_prob'].mean()*100:.2f}%)")
    print(f"    ≥70%: {(df_test['short_prob'] >= 0.70).sum():,} candles ({(df_test['short_prob'] >= 0.70).sum()/len(df_test)*100:.1f}%)\n")

except Exception as e:
    print(f"  ❌ Error: {e}")
    df_test['long_prob'] = 0.0
    df_test['short_prob'] = 0.0


# =============================================================================
# BACKTEST FUNCTION
# =============================================================================
def backtest_opportunity_gating(window_df):
    """
    Opportunity Gating Strategy with 4x Leverage + Dynamic Sizing
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL

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

                # Entry fee
                entry_fee = sizing_result['leveraged_value'] * TAKER_FEE
                capital -= entry_fee
                position['entry_fee'] = entry_fee

            # SHORT entry (gated)
            elif short_prob >= SHORT_THRESHOLD:
                # Opportunity cost check
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
                        'entry_prob': short_prob,
                        'opportunity_cost': opportunity_cost
                    }

                    # Entry fee
                    entry_fee = sizing_result['leveraged_value'] * TAKER_FEE
                    capital -= entry_fee
                    position['entry_fee'] = entry_fee

        # Exit logic
        else:
            current_price = window_df['close'].iloc[i]
            candles_held = i - position['entry_idx']

            # Calculate P&L
            if position['side'] == 'LONG':
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Exit checks
            should_exit = False
            exit_reason = ""

            # ML Exit
            try:
                if position['side'] == 'LONG':
                    exit_feat = window_df[long_exit_feature_columns].iloc[i:i+1].values
                    exit_feat_scaled = long_exit_scaler.transform(exit_feat)
                    exit_prob = long_exit_model.predict_proba(exit_feat_scaled)[0][1]
                    ml_threshold = ML_EXIT_THRESHOLD_LONG
                else:  # SHORT
                    exit_feat = window_df[short_exit_feature_columns].iloc[i:i+1].values
                    exit_feat_scaled = short_exit_scaler.transform(exit_feat)
                    exit_prob = short_exit_model.predict_proba(exit_feat_scaled)[0][1]
                    ml_threshold = ML_EXIT_THRESHOLD_SHORT

                if exit_prob >= ml_threshold:
                    should_exit = True
                    exit_reason = f"ML Exit ({exit_prob:.3f})"
            except:
                pass

            # Emergency Stop Loss (Balance-Based: -6% total balance)
            balance_loss_pct = leveraged_pnl_pct * position['position_size_pct']
            if balance_loss_pct <= -EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = f"Stop Loss ({leveraged_pnl_pct*100:.1f}%)"

            # Emergency Max Hold
            if candles_held >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = f"Max Hold ({candles_held} candles)"

            # Execute exit
            if should_exit:
                # Calculate P&L
                entry_notional = position['quantity'] * position['entry_price']
                exit_notional = position['quantity'] * current_price

                if position['side'] == 'LONG':
                    pnl_usd = exit_notional - entry_notional
                else:  # SHORT
                    pnl_usd = entry_notional - exit_notional

                # Exit fee
                exit_fee = position['leveraged_value'] * TAKER_FEE
                pnl_usd_net = pnl_usd - position['entry_fee'] - exit_fee

                # Update capital
                capital += pnl_usd_net

                # Record trade
                trades.append({
                    'side': position['side'],
                    'entry_idx': position['entry_idx'],
                    'exit_idx': i,
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob'],
                    'hold_candles': candles_held,
                    'price_change_pct': price_change_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'pnl_usd_net': pnl_usd_net,
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct']
                })

                position = None

    return trades, capital


# =============================================================================
# RUN BACKTEST
# =============================================================================
print("="*80)
print("RUNNING BACKTEST")
print("="*80)

trades, final_capital = backtest_opportunity_gating(df_test)

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "="*80)
print("BACKTEST RESULTS: OCT 14-19 (5 DAYS)")
print("="*80)

if len(trades) > 0:
    df_trades = pd.DataFrame(trades)

    total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    wins = df_trades[df_trades['pnl_usd_net'] > 0]
    losses = df_trades[df_trades['pnl_usd_net'] <= 0]
    win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0

    print(f"\n💰 Performance:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Capital: ${final_capital:,.2f}")
    print(f"  Total Return: ${final_capital - INITIAL_CAPITAL:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"  Per Day: {total_return_pct / 5:.2f}%")

    print(f"\n📊 Trade Statistics:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  LONG: {len(df_trades[df_trades['side'] == 'LONG'])} ({len(df_trades[df_trades['side'] == 'LONG'])/len(trades)*100:.1f}%)")
    print(f"  SHORT: {len(df_trades[df_trades['side'] == 'SHORT'])} ({len(df_trades[df_trades['side'] == 'SHORT'])/len(trades)*100:.1f}%)")
    print(f"  Win Rate: {len(wins)}/{len(trades)} ({win_rate:.1f}%)")
    print(f"  Avg Hold: {df_trades['hold_candles'].mean():.1f} candles ({df_trades['hold_candles'].mean()*5:.0f} min)")

    print(f"\n📈 P&L Analysis:")
    print(f"  Avg P&L: {df_trades['leveraged_pnl_pct'].mean()*100:+.2f}%")
    print(f"  Avg Win: {wins['leveraged_pnl_pct'].mean()*100:+.2f}% (${wins['pnl_usd_net'].mean():+,.2f})")
    print(f"  Avg Loss: {losses['leveraged_pnl_pct'].mean()*100:+.2f}% (${losses['pnl_usd_net'].mean():+,.2f})")
    print(f"  Best Trade: {df_trades['leveraged_pnl_pct'].max()*100:+.2f}% (${df_trades['pnl_usd_net'].max():+,.2f})")
    print(f"  Worst Trade: {df_trades['leveraged_pnl_pct'].min()*100:+.2f}% (${df_trades['pnl_usd_net'].min():+,.2f})")

    # Exit reason breakdown
    print(f"\n🚪 Exit Reasons:")
    exit_reasons = df_trades['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")

    # Daily breakdown
    print(f"\n📅 Daily Breakdown:")
    df_trades['entry_date'] = df_test.iloc[df_trades['entry_idx'].values]['timestamp'].str[:10].values
    daily_stats = df_trades.groupby('entry_date').agg({
        'pnl_usd_net': ['sum', 'count'],
        'leveraged_pnl_pct': 'mean'
    })

    for date, row in daily_stats.iterrows():
        day_pnl = row['pnl_usd_net']['sum']
        day_trades = row['pnl_usd_net']['count']
        day_avg = row['leveraged_pnl_pct']['mean'] * 100
        print(f"  {date}: ${day_pnl:+,.2f} ({day_trades} trades, avg {day_avg:+.2f}%)")

    # Sample trades
    print(f"\n📋 Sample Trades (first 10):")
    print(f"{'Side':<6} {'Entry':<10} {'Exit':<10} {'Hold':<8} {'P&L%':<8} {'Net P&L':<12} {'Reason':<20}")
    print("-"*90)
    for _, trade in df_trades.head(10).iterrows():
        print(f"{trade['side']:<6} ${trade['entry_price']:<9,.1f} ${trade['exit_price']:<9,.1f} "
              f"{trade['hold_candles']:<8.0f} {trade['leveraged_pnl_pct']*100:<7.2f}% "
              f"${trade['pnl_usd_net']:<11,.2f} {trade['exit_reason']:<20}")

else:
    print("\n⚠️  No trades executed during this period")
    print(f"   LONG signals: {(df_test['long_prob'] >= LONG_THRESHOLD).sum()}")
    print(f"   SHORT signals: {(df_test['short_prob'] >= SHORT_THRESHOLD).sum()}")
    print(f"   Check if thresholds are too high or signals too low")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
