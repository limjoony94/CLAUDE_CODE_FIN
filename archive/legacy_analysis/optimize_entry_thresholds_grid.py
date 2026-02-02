"""
Entry Threshold Grid Search Optimization

Grid search over LONG_THRESHOLD and SHORT_THRESHOLD to find optimal entry parameters.
Evaluates performance using 30-day backtest with composite scoring.

Author: Claude Code
Date: 2025-10-23
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import sys
import os
import time
from itertools import product

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.bingx_client import BingXClient
from scripts.production.dynamic_position_sizing import DynamicPositionSizer
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
import yaml

def load_api_keys():
    """Load API keys from config file"""
    with open('config/api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

# =============================================================================
# GRID SEARCH PARAMETERS
# =============================================================================

# Entry threshold ranges
LONG_THRESHOLD_RANGE = [0.55, 0.60, 0.65, 0.70, 0.75]
SHORT_THRESHOLD_RANGE = [0.60, 0.65, 0.70, 0.75, 0.80]

# Fixed parameters (from current optimal settings)
GATE_THRESHOLD = 0.001
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
EMERGENCY_STOP_LOSS = 0.03  # -3%
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours

# Backtest period (5 days - API limit 1440 candles)
START_DATE = "2025-10-18"
END_DATE = "2025-10-22"

# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

SYMBOL = "BTC-USDT"
INITIAL_BALANCE = 10000
LEVERAGE = 4
TRADING_FEE_RATE = 0.0005  # 0.05%

# =============================================================================
# LOAD MODELS
# =============================================================================

print("=" * 80)
print("ENTRY THRESHOLD OPTIMIZATION - GRID SEARCH")
print("=" * 80)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Grid Size: {len(LONG_THRESHOLD_RANGE)} × {len(SHORT_THRESHOLD_RANGE)} = {len(LONG_THRESHOLD_RANGE) * len(SHORT_THRESHOLD_RANGE)} combinations")
print()

# Load models (UPDATED: NEW Entry models with 5-fold CV - 2025-10-24)
model_dir = "models"
long_entry_model = joblib.load(f"{model_dir}/xgboost_long_entry_enhanced_20251024_012445.pkl")
short_entry_model = joblib.load(f"{model_dir}/xgboost_short_entry_enhanced_20251024_012445.pkl")
long_exit_model = joblib.load(f"{model_dir}/xgboost_long_exit_enhanced_20251023_212712.pkl")
short_exit_model = joblib.load(f"{model_dir}/xgboost_short_exit_enhanced_20251023_212712.pkl")

long_entry_scaler = joblib.load(f"{model_dir}/xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
short_entry_scaler = joblib.load(f"{model_dir}/xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
long_exit_scaler = joblib.load(f"{model_dir}/xgboost_long_exit_enhanced_20251023_212712_scaler.pkl")
short_exit_scaler = joblib.load(f"{model_dir}/xgboost_short_exit_enhanced_20251023_212712_scaler.pkl")

# Load feature columns (like production bot)
with open(f"{model_dir}/xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_feature_columns = [line.strip() for line in f.readlines()]

with open(f"{model_dir}/xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_feature_columns = [line.strip() for line in f.readlines()]

with open(f"{model_dir}/xgboost_long_exit_enhanced_20251023_212712_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines()]

with open(f"{model_dir}/xgboost_short_exit_enhanced_20251023_212712_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines()]

print("✅ Models loaded")
print()

# Initialize position sizer
position_sizer = DynamicPositionSizer()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_data():
    """
    Load and prepare historical data (called once)

    Returns prepared DataFrame with all features
    """
    print("📊 Preparing data...")
    print("  1/3 Loading from API...")

    # Load API keys
    config = load_api_keys()

    # Initialize client
    client = BingXClient(
        api_key=config['bingx']['mainnet']['api_key'],
        secret_key=config['bingx']['mainnet']['secret_key'],
        testnet=False
    )

    # Fetch historical data (last 1440 candles = 5 days at 5m interval)
    candles = client.get_klines(
        symbol=SYMBOL,
        interval="5m",
        limit=1440  # Get latest 1440 candles (no start_time - API doesn't support both)
    )

    if candles is None or len(candles) == 0:
        print(f"  ERROR: No data returned from API!")
        raise Exception("Failed to fetch historical data from BingX")

    # API returns dict with 'time' key, not 'timestamp'
    df = pd.DataFrame(candles)
    df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Ensure correct column order and types
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Date range info
    actual_start = df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
    actual_end = df['timestamp'].max().strftime('%Y-%m-%d %H:%M')

    print(f"  2/3 Loaded {len(df)} candles ({actual_start} to {actual_end})")

    # Calculate features (UPDATED: phase='phase1' for NEW Entry models)
    df = calculate_all_features_enhanced_v2(df, phase='phase1')

    # Add exit-specific features
    df = prepare_exit_features(df)

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    print(f"  3/3 Features calculated, {len(df)} candles ready")
    print()

    return df

def run_backtest(df, long_threshold, short_threshold):
    """
    Run backtest with given entry thresholds on prepared data

    Args:
        df: Prepared DataFrame with all features
        long_threshold: Probability threshold for LONG entry
        short_threshold: Probability threshold for SHORT entry

    Returns dict with performance metrics
    """
    # Use the prepared data (no API call needed)

    # Initialize tracking
    balance = INITIAL_BALANCE
    position = None
    trades = []

    # Simulate trading
    for i in range(len(df)):
        current_candle = df.iloc[i]
        current_price = current_candle['close']
        current_time = current_candle['timestamp']

        # Check exit conditions
        if position is not None:
            # Calculate P&L
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = pnl_pct * LEVERAGE

            # Time in position
            time_in_position = i - position['entry_idx']

            # Exit conditions
            exit_triggered = False
            exit_reason = None

            # 1. Emergency Stop Loss (-3%)
            if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                exit_triggered = True
                exit_reason = "STOP_LOSS"

            # 2. Emergency Max Hold (120 candles = 10 hours)
            elif time_in_position >= EMERGENCY_MAX_HOLD_TIME:
                exit_triggered = True
                exit_reason = "MAX_HOLD"

            # 3. ML Exit Signal
            else:
                # Get exit features
                if position['side'] == 'LONG':
                    exit_features = df.iloc[i][long_exit_feature_columns].values.reshape(1, -1)
                    exit_features_scaled = long_exit_scaler.transform(exit_features)
                    exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
                    ml_threshold = ML_EXIT_THRESHOLD_LONG
                else:
                    exit_features = df.iloc[i][short_exit_feature_columns].values.reshape(1, -1)
                    exit_features_scaled = short_exit_scaler.transform(exit_features)
                    exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]
                    ml_threshold = ML_EXIT_THRESHOLD_SHORT

                if exit_prob >= ml_threshold:
                    exit_triggered = True
                    exit_reason = "ML_EXIT"

            # Execute exit
            if exit_triggered:
                # Calculate final P&L
                gross_pnl = position['position_value'] * leveraged_pnl_pct

                # Calculate fees
                entry_fee = position['position_value'] * LEVERAGE * TRADING_FEE_RATE
                exit_fee = position['position_value'] * LEVERAGE * TRADING_FEE_RATE
                total_fees = entry_fee + exit_fee

                net_pnl = gross_pnl - total_fees

                # Update balance
                balance += net_pnl

                # Record trade
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'position_size_pct': position['position_size_pct'],
                    'position_value': position['position_value'],
                    'pnl': net_pnl,
                    'pnl_pct': leveraged_pnl_pct,
                    'exit_reason': exit_reason,
                    'hold_time': time_in_position
                })

                # Clear position
                position = None

        # Check entry conditions (if no position)
        if position is None:
            # Get entry features
            long_features = df.iloc[i][long_entry_feature_columns].values.reshape(1, -1)
            short_features = df.iloc[i][short_entry_feature_columns].values.reshape(1, -1)

            long_features_scaled = long_entry_scaler.transform(long_features)
            short_features_scaled = short_entry_scaler.transform(short_features)

            long_prob = long_entry_model.predict_proba(long_features_scaled)[0][1]
            short_prob = short_entry_model.predict_proba(short_features_scaled)[0][1]

            # Entry logic
            entered = False

            # LONG entry
            if long_prob >= long_threshold:
                sizing_result = position_sizer.get_position_size_simple(balance, long_prob, LEVERAGE)
                position_size_pct = sizing_result['position_size_pct']
                position_value = sizing_result['position_value']

                position = {
                    'side': 'LONG',
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'entry_idx': i,
                    'position_size_pct': position_size_pct,
                    'position_value': position_value
                }
                entered = True

            # SHORT entry (with opportunity gate)
            if not entered and short_prob >= short_threshold:
                # Calculate opportunity cost
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    sizing_result = position_sizer.get_position_size_simple(balance, short_prob, LEVERAGE)
                    position_size_pct = sizing_result['position_size_pct']
                    position_value = sizing_result['position_value']

                    position = {
                        'side': 'SHORT',
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'entry_idx': i,
                        'position_size_pct': position_size_pct,
                        'position_value': position_value
                    }

    # Close any open position at end
    if position is not None:
        current_price = df.iloc[-1]['close']

        if position['side'] == 'LONG':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = pnl_pct * LEVERAGE
        gross_pnl = position['position_value'] * leveraged_pnl_pct

        entry_fee = position['position_value'] * LEVERAGE * TRADING_FEE_RATE
        exit_fee = position['position_value'] * LEVERAGE * TRADING_FEE_RATE
        total_fees = entry_fee + exit_fee

        net_pnl = gross_pnl - total_fees
        balance += net_pnl

        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'position_size_pct': position['position_size_pct'],
            'position_value': position['position_value'],
            'pnl': net_pnl,
            'pnl_pct': leveraged_pnl_pct,
            'exit_reason': 'END_OF_PERIOD',
            'hold_time': len(df) - position['entry_idx']
        })

    # Calculate metrics
    if len(trades) == 0:
        return {
            'return': 0,
            'win_rate': 0,
            'num_trades': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'ml_exit_rate': 0,
            'sl_rate': 0,
            'max_hold_rate': 0
        }

    trades_df = pd.DataFrame(trades)

    total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    wins = trades_df[trades_df['pnl'] > 0]
    win_rate = (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0

    # Calculate max drawdown
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = (drawdown.min() / INITIAL_BALANCE) * 100

    # Calculate Sharpe ratio
    returns = trades_df['pnl'] / INITIAL_BALANCE
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(len(trades_df))) if returns.std() > 0 else 0

    # Calculate profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Exit reason breakdown
    exit_reasons = trades_df['exit_reason'].value_counts()
    ml_exit_rate = (exit_reasons.get('ML_EXIT', 0) / len(trades_df)) * 100
    sl_rate = (exit_reasons.get('STOP_LOSS', 0) / len(trades_df)) * 100
    max_hold_rate = (exit_reasons.get('MAX_HOLD', 0) / len(trades_df)) * 100

    return {
        'return': total_return,
        'win_rate': win_rate,
        'num_trades': len(trades_df),
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'ml_exit_rate': ml_exit_rate,
        'sl_rate': sl_rate,
        'max_hold_rate': max_hold_rate
    }

def calculate_composite_score(metrics):
    """
    Calculate composite score for ranking

    Weighted combination of normalized metrics
    """
    # Normalize metrics (0-1 scale)
    return_norm = min(max(metrics['return'] / 100, 0), 1)  # Cap at 100%
    win_rate_norm = metrics['win_rate'] / 100
    mdd_norm = max(1 - abs(metrics['max_drawdown']) / 50, 0)  # Cap at -50%
    sharpe_norm = min(max(metrics['sharpe_ratio'] / 2, 0), 1)  # Cap at 2.0
    pf_norm = min(max((metrics['profit_factor'] - 1) / 2, 0), 1)  # Cap at 3.0

    # Weighted average
    score = (
        return_norm * 0.35 +
        win_rate_norm * 0.20 +
        mdd_norm * 0.20 +
        sharpe_norm * 0.15 +
        pf_norm * 0.10
    )

    return score

# =============================================================================
# GRID SEARCH
# =============================================================================

# Prepare data once (single API call)
df_prepared = prepare_data()

print("Starting grid search...")
print()

results = []
total_combinations = len(LONG_THRESHOLD_RANGE) * len(SHORT_THRESHOLD_RANGE)
start_time = time.time()

for idx, (long_th, short_th) in enumerate(product(LONG_THRESHOLD_RANGE, SHORT_THRESHOLD_RANGE), 1):
    print(f"[{idx}/{total_combinations}] Testing LONG={long_th:.2f}, SHORT={short_th:.2f}...", end=" ")

    try:
        metrics = run_backtest(df_prepared, long_th, short_th)
        score = calculate_composite_score(metrics)

        result = {
            'long_threshold': long_th,
            'short_threshold': short_th,
            'return': metrics['return'],
            'win_rate': metrics['win_rate'],
            'num_trades': metrics['num_trades'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'profit_factor': metrics['profit_factor'],
            'ml_exit_rate': metrics['ml_exit_rate'],
            'sl_rate': metrics['sl_rate'],
            'max_hold_rate': metrics['max_hold_rate'],
            'composite_score': score
        }

        results.append(result)

        print(f"Return: {metrics['return']:+.2f}%, WR: {metrics['win_rate']:.1f}%, Score: {score:.4f}")

    except Exception as e:
        print(f"ERROR: {e}")

elapsed_time = time.time() - start_time

print()
print("=" * 80)
print("GRID SEARCH COMPLETE")
print("=" * 80)
print(f"Total time: {elapsed_time/60:.1f} minutes")
print()

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('composite_score', ascending=False)

# Save results
output_file = "results/entry_threshold_optimization_grid_20251023.csv"
results_df.to_csv(output_file, index=False)
print(f"✅ Results saved to: {output_file}")
print()

# Display top 10
print("=" * 80)
print("TOP 10 CONFIGURATIONS")
print("=" * 80)
print()

for i, row in results_df.head(10).iterrows():
    rank = results_df.index.get_loc(i) + 1
    print(f"Rank {rank}: LONG={row['long_threshold']:.2f}, SHORT={row['short_threshold']:.2f} (Score: {row['composite_score']:.4f})")
    print(f"  Return: {row['return']:+.2f}% | WR: {row['win_rate']:.1f}% | Trades: {int(row['num_trades'])}")
    print(f"  MDD: {row['max_drawdown']:.2f}% | Sharpe: {row['sharpe_ratio']:.3f} | PF: {row['profit_factor']:.2f}x")
    print(f"  Exits: ML {row['ml_exit_rate']:.1f}%, SL {row['sl_rate']:.1f}%, MaxHold {row['max_hold_rate']:.1f}%")
    print()

# Baseline comparison (current settings)
baseline = results_df[(results_df['long_threshold'] == 0.65) & (results_df['short_threshold'] == 0.70)]
if not baseline.empty:
    baseline_row = baseline.iloc[0]
    baseline_rank = results_df.index.get_loc(baseline.index[0]) + 1

    print("=" * 80)
    print("BASELINE (CURRENT SETTINGS)")
    print("=" * 80)
    print(f"Rank {baseline_rank}: LONG=0.65, SHORT=0.70 (Score: {baseline_row['composite_score']:.4f})")
    print(f"Return: {baseline_row['return']:+.2f}% | WR: {baseline_row['win_rate']:.1f}% | Trades: {int(baseline_row['num_trades'])}")
    print(f"MDD: {baseline_row['max_drawdown']:.2f}% | Sharpe: {baseline_row['sharpe_ratio']:.3f} | PF: {baseline_row['profit_factor']:.2f}x")
    print()

    # Calculate improvement
    best_row = results_df.iloc[0]
    improvement_return = ((best_row['return'] - baseline_row['return']) / abs(baseline_row['return'])) * 100 if baseline_row['return'] != 0 else 0
    improvement_wr = best_row['win_rate'] - baseline_row['win_rate']
    improvement_sharpe = ((best_row['sharpe_ratio'] - baseline_row['sharpe_ratio']) / baseline_row['sharpe_ratio']) * 100 if baseline_row['sharpe_ratio'] != 0 else 0

    print("=" * 80)
    print("IMPROVEMENT vs BASELINE")
    print("=" * 80)
    print(f"Return: {improvement_return:+.1f}% (from {baseline_row['return']:.2f}% to {best_row['return']:.2f}%)")
    print(f"Win Rate: {improvement_wr:+.1f}pp (from {baseline_row['win_rate']:.1f}% to {best_row['win_rate']:.1f}%)")
    print(f"Sharpe Ratio: {improvement_sharpe:+.1f}% (from {baseline_row['sharpe_ratio']:.3f} to {best_row['sharpe_ratio']:.3f})")
    print()

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
best = results_df.iloc[0]
print(f"Optimal: LONG_THRESHOLD = {best['long_threshold']:.2f}, SHORT_THRESHOLD = {best['short_threshold']:.2f}")
print()
print("Update production bot:")
print(f"  LONG_THRESHOLD = {best['long_threshold']:.2f}")
print(f"  SHORT_THRESHOLD = {best['short_threshold']:.2f}")
print()
