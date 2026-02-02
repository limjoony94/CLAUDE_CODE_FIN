"""
Current Production Settings Backtest: 7-Day Test Period
========================================================

Tests CURRENT production configuration (as of 2025-10-25) on recent 7 days:
- Entry Models: Trade-Outcome Full Dataset (2025-10-18)
- Exit Models: Opportunity Gating Improved (2025-10-24)
- LONG Threshold: 0.70 (optimized 2025-10-24)
- ML Exit Threshold: 0.725 (both LONG/SHORT, adjusted 2025-10-25)
- Emergency Stop Loss: -3% balance
- Emergency Max Hold: 120 candles (10 hours)
- 4x Leverage with Dynamic Sizing (20-95%)
- Opportunity Gating (gate = 0.001)

Configuration matches opportunity_gating_bot_4x.py EXACTLY as of 2025-10-25.
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

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("CURRENT PRODUCTION SETTINGS BACKTEST - 7 DAY TEST (2025-10-25)")
print("="*80)
print(f"\nTesting CURRENT production configuration on recent 7 days\n")

# =============================================================================
# CURRENT PRODUCTION CONFIGURATION (as of 2025-10-25)
# =============================================================================

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.75  # EXPERIMENT 2025-10-25
SHORT_THRESHOLD = 0.75  # EXPERIMENT 2025-10-25
GATE_THRESHOLD = 0.001

# Exit Parameters (EXPERIMENT 2025-10-25)
ML_EXIT_THRESHOLD_LONG = 0.75    # EXPERIMENT 2025-10-25
ML_EXIT_THRESHOLD_SHORT = 0.75   # EXPERIMENT 2025-10-25
EMERGENCY_MAX_HOLD_TIME = 120    # 10 hours (120 candles × 5min)
EMERGENCY_STOP_LOSS = 0.03       # Balance-Based: -3% total balance

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Backtest Parameters
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005  # 0.05% per trade

# =============================================================================
# LOAD CURRENT PRODUCTION MODELS
# =============================================================================

print("Loading CURRENT PRODUCTION Entry models (Trade-Outcome Full Dataset - Oct 18)...")
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

# Load Exit Models (CURRENT - Opportunity Gating Improved - Oct 24)
print("Loading CURRENT Exit models (Opportunity Gating Improved - Oct 24)...")
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

print(f"  ✅ LONG Exit: {len(long_exit_feature_columns)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_feature_columns)} features\n")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# =============================================================================
# LOAD AND PREPARE DATA - LAST 7 DAYS
# =============================================================================

print("Loading historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_full):,} candles\n")

# Calculate 7 days worth of 5-minute candles + buffer for features
# 7 days × 288 candles/day = 2016 candles + 200 buffer = 2216 candles
SEVEN_DAYS_CANDLES = 7 * 288
BUFFER_CANDLES = 200
TOTAL_CANDLES_NEEDED = SEVEN_DAYS_CANDLES + BUFFER_CANDLES

print(f"Using last 7 days of data...")
print(f"  - 7 days = {SEVEN_DAYS_CANDLES} candles")
print(f"  - Buffer = {BUFFER_CANDLES} candles for feature calculation")
print(f"  - Total = {TOTAL_CANDLES_NEEDED} candles\n")

df_recent = df_full.tail(TOTAL_CANDLES_NEEDED).reset_index(drop=True)
print(f"  ✅ Selected {len(df_recent):,} recent candles")
print(f"     Date range: {df_recent['timestamp'].iloc[0]} to {df_recent['timestamp'].iloc[-1]}\n")

print("Calculating features...")
start_time = time.time()
df = calculate_all_features_enhanced_v2(df_recent)
df = prepare_exit_features(df)
feature_time = time.time() - start_time

# Drop NaN rows from feature calculation
df = df.dropna().reset_index(drop=True)
print(f"  ✅ Features calculated ({feature_time:.1f}s)")
print(f"     {len(df):,} candles after feature calculation\n")

# Take exactly 7 days (2016 candles)
df_test = df.tail(SEVEN_DAYS_CANDLES).reset_index(drop=True)
print(f"  ✅ Final test period: {len(df_test):,} candles (7 days)")
print(f"     Test period: {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}\n")

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

def backtest_current_production(test_df):
    """
    Current Production Settings Backtest for 7 days

    Exact configuration from opportunity_gating_bot_4x.py as of 2025-10-25
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL
    recent_trades = []

    # Track daily performance
    daily_capital = []
    daily_returns = []
    candles_per_day = 288

    for i in range(len(test_df) - 1):
        current_price = test_df['close'].iloc[i]

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
                position_size_pct = sizing_result['position_size_pct']
                position_value = capital * position_size_pct
                position_size_btc = (position_value * LEVERAGE) / current_price
                entry_fee = position_value * LEVERAGE * TAKER_FEE

                position = {
                    'type': 'LONG',
                    'entry_price': current_price,
                    'entry_candle': i,
                    'position_value': position_value,
                    'position_size_btc': position_size_btc,
                    'entry_fee': entry_fee,
                    'signal_strength': long_prob,
                    'position_size_pct': position_size_pct
                }

            # SHORT entry (Opportunity Gating)
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
                    position_size_pct = sizing_result['position_size_pct']
                    position_value = capital * position_size_pct
                    position_size_btc = (position_value * LEVERAGE) / current_price
                    entry_fee = position_value * LEVERAGE * TAKER_FEE

                    position = {
                        'type': 'SHORT',
                        'entry_price': current_price,
                        'entry_candle': i,
                        'position_value': position_value,
                        'position_size_btc': position_size_btc,
                        'entry_fee': entry_fee,
                        'signal_strength': short_prob,
                        'position_size_pct': position_size_pct,
                        'opportunity_cost': opportunity_cost
                    }

        # Exit logic
        else:
            leveraged_pnl_pct = 0.0
            candles_held = i - position['entry_candle']

            if position['type'] == 'LONG':
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                leveraged_pnl_pct = price_change_pct * LEVERAGE
            else:  # SHORT
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                leveraged_pnl_pct = price_change_pct * LEVERAGE

            balance_pnl_pct = leveraged_pnl_pct * position['position_size_pct']

            # Emergency Stop Loss (Balance-Based: -3% total balance)
            if balance_pnl_pct <= -EMERGENCY_STOP_LOSS:
                pnl = capital * balance_pnl_pct
                exit_fee = position['position_value'] * LEVERAGE * TAKER_FEE
                net_pnl = pnl - position['entry_fee'] - exit_fee
                capital += net_pnl

                trades.append({
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_candle': position['entry_candle'],
                    'exit_candle': i,
                    'candles_held': candles_held,
                    'pnl': net_pnl,
                    'pnl_pct': (net_pnl / INITIAL_CAPITAL) * 100,
                    'balance_pnl_pct': balance_pnl_pct,
                    'exit_reason': 'EMERGENCY_STOP_LOSS',
                    'signal_strength': position['signal_strength'],
                    'position_size_pct': position['position_size_pct']
                })

                recent_trades.append({'outcome': 'loss', 'pnl_pct': balance_pnl_pct})
                position = None
                continue

            # Emergency Max Hold Time (120 candles = 10 hours)
            if candles_held >= EMERGENCY_MAX_HOLD_TIME:
                pnl = capital * balance_pnl_pct
                exit_fee = position['position_value'] * LEVERAGE * TAKER_FEE
                net_pnl = pnl - position['entry_fee'] - exit_fee
                capital += net_pnl

                trades.append({
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_candle': position['entry_candle'],
                    'exit_candle': i,
                    'candles_held': candles_held,
                    'pnl': net_pnl,
                    'pnl_pct': (net_pnl / INITIAL_CAPITAL) * 100,
                    'balance_pnl_pct': balance_pnl_pct,
                    'exit_reason': 'EMERGENCY_MAX_HOLD',
                    'signal_strength': position['signal_strength'],
                    'position_size_pct': position['position_size_pct']
                })

                recent_trades.append({
                    'outcome': 'win' if balance_pnl_pct > 0 else 'loss',
                    'pnl_pct': balance_pnl_pct
                })
                position = None
                continue

            # ML Exit Signal
            try:
                if position['type'] == 'LONG':
                    exit_features_values = test_df[long_exit_feature_columns].iloc[i:i+1].values
                    exit_features_scaled = long_exit_scaler.transform(exit_features_values)
                    exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
                    exit_threshold = ML_EXIT_THRESHOLD_LONG
                else:  # SHORT
                    exit_features_values = test_df[short_exit_feature_columns].iloc[i:i+1].values
                    exit_features_scaled = short_exit_scaler.transform(exit_features_values)
                    exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]
                    exit_threshold = ML_EXIT_THRESHOLD_SHORT

                if exit_prob >= exit_threshold:
                    pnl = capital * balance_pnl_pct
                    exit_fee = position['position_value'] * LEVERAGE * TAKER_FEE
                    net_pnl = pnl - position['entry_fee'] - exit_fee
                    capital += net_pnl

                    trades.append({
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'entry_candle': position['entry_candle'],
                        'exit_candle': i,
                        'candles_held': candles_held,
                        'pnl': net_pnl,
                        'pnl_pct': (net_pnl / INITIAL_CAPITAL) * 100,
                        'balance_pnl_pct': balance_pnl_pct,
                        'exit_reason': 'ML_EXIT',
                        'exit_prob': exit_prob,
                        'signal_strength': position['signal_strength'],
                        'position_size_pct': position['position_size_pct']
                    })

                    recent_trades.append({
                        'outcome': 'win' if balance_pnl_pct > 0 else 'loss',
                        'pnl_pct': balance_pnl_pct
                    })
                    position = None
            except Exception as e:
                pass  # Continue if ML exit fails

        # Track daily performance
        if (i + 1) % candles_per_day == 0:
            daily_capital.append(capital)
            daily_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            daily_returns.append(daily_return)

    return trades, capital, daily_returns


# =============================================================================
# RUN BACKTEST
# =============================================================================

print("="*80)
print("RUNNING BACKTEST...")
print("="*80)
print()

start_time = time.time()
trades, final_capital, daily_returns = backtest_current_production(df_test)
backtest_time = time.time() - start_time

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

print("\n" + "="*80)
print("BACKTEST RESULTS - CURRENT PRODUCTION SETTINGS (2025-10-25)")
print("="*80)

# Configuration Summary
print("\n📋 CONFIGURATION:")
print(f"  Entry Thresholds: LONG={LONG_THRESHOLD}, SHORT={SHORT_THRESHOLD}")
print(f"  ML Exit Thresholds: LONG={ML_EXIT_THRESHOLD_LONG}, SHORT={ML_EXIT_THRESHOLD_SHORT}")
print(f"  Emergency Stop Loss: -{EMERGENCY_STOP_LOSS*100:.1f}% balance")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD_TIME} candles ({EMERGENCY_MAX_HOLD_TIME*5/60:.1f} hours)")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Opportunity Gate: {GATE_THRESHOLD}")

# Overall Performance
total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
print(f"\n💰 OVERALL PERFORMANCE:")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"  Final Capital: ${final_capital:,.2f}")
print(f"  Total Return: {total_return:+.2f}%")
print(f"  Total Trades: {len(trades)}")
print(f"  Backtest Time: {backtest_time:.1f}s")

if len(trades) > 0:
    # Trade Statistics
    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] <= 0]

    win_rate = (len(wins) / len(df_trades)) * 100
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    avg_trade = df_trades['pnl'].mean()

    print(f"\n📊 TRADE STATISTICS:")
    print(f"  Win Rate: {win_rate:.1f}% ({len(wins)}/{len(df_trades)})")
    print(f"  Average Win: ${avg_win:,.2f}")
    print(f"  Average Loss: ${avg_loss:,.2f}")
    print(f"  Average Trade: ${avg_trade:,.2f}")

    # Best and Worst Trades
    best_trade = df_trades.loc[df_trades['pnl'].idxmax()]
    worst_trade = df_trades.loc[df_trades['pnl'].idxmin()]

    print(f"\n🏆 BEST TRADE:")
    print(f"  Type: {best_trade['type']}")
    print(f"  P&L: ${best_trade['pnl']:,.2f} ({best_trade['pnl_pct']:.2f}%)")
    print(f"  Exit: {best_trade['exit_reason']}")

    print(f"\n📉 WORST TRADE:")
    print(f"  Type: {worst_trade['type']}")
    print(f"  P&L: ${worst_trade['pnl']:,.2f} ({worst_trade['pnl_pct']:.2f}%)")
    print(f"  Exit: {worst_trade['exit_reason']}")

    # Exit Reason Distribution
    exit_reasons = df_trades['exit_reason'].value_counts()
    print(f"\n🚪 EXIT REASONS:")
    for reason, count in exit_reasons.items():
        pct = (count / len(df_trades)) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    # Position Type Distribution
    long_trades = df_trades[df_trades['type'] == 'LONG']
    short_trades = df_trades[df_trades['type'] == 'SHORT']

    print(f"\n📈 POSITION TYPES:")
    print(f"  LONG: {len(long_trades)} ({len(long_trades)/len(df_trades)*100:.1f}%)")
    print(f"  SHORT: {len(short_trades)} ({len(short_trades)/len(df_trades)*100:.1f}%)")

    # Daily Performance
    if len(daily_returns) > 0:
        print(f"\n📅 DAILY RETURNS:")
        for day, ret in enumerate(daily_returns, 1):
            print(f"  Day {day}: {ret:+.2f}%")

    # Risk Metrics
    returns_array = df_trades['pnl_pct'].values
    max_drawdown = 0
    peak = 0
    cumulative = 0

    for ret in returns_array:
        cumulative += ret
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    sharpe_ratio = 0
    if len(returns_array) > 1 and returns_array.std() != 0:
        sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(len(returns_array))

    print(f"\n⚠️ RISK METRICS:")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")

    # Detailed Trade Log
    print(f"\n📝 DETAILED TRADE LOG:")
    print(f"{'Type':<6} {'Entry':<10} {'Exit':<10} {'Held':<6} {'P&L %':<8} {'Exit Reason':<20} {'Signal':<7}")
    print("-" * 80)

    for _, trade in df_trades.iterrows():
        print(f"{trade['type']:<6} "
              f"${trade['entry_price']:<9.2f} "
              f"${trade['exit_price']:<9.2f} "
              f"{trade['candles_held']:<6} "
              f"{trade['pnl_pct']:+7.2f}% "
              f"{trade['exit_reason']:<20} "
              f"{trade['signal_strength']:.3f}")

else:
    print("\n⚠️ NO TRADES EXECUTED")
    print("  Check if entry thresholds are too high or signals are too weak")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
