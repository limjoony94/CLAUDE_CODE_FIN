"""
Exit Strategy Comparison: Baseline vs Improvements
==================================================

Compare 4 different exit configurations:
1. BASELINE: Current system (ML Exit 0.70 + Emergency)
2. TAKE_PROFIT: Baseline + Fixed/Trailing Take Profit
3. DYNAMIC_THRESHOLD: Baseline + Volatility-based ML Threshold
4. COMBINED: Take Profit + Dynamic Threshold

All tested on same data with same entry logic for fair comparison.
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
print("EXIT STRATEGY COMPARISON: BASELINE vs IMPROVEMENTS")
print("="*80)
print("\nTesting 4 configurations on same data\n")

# Strategy Parameters (FIXED across all tests)
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters - BASELINE
ML_EXIT_THRESHOLD_BASE = 0.70
EMERGENCY_MAX_HOLD_TIME = 96  # 8 hours
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)

# NEW: Take Profit Parameters
FIXED_TAKE_PROFIT = 0.03  # 3% (on 4x leveraged P&L)
TRAILING_TP_ACTIVATION = 0.02  # 2% profit to activate trailing
TRAILING_TP_DRAWDOWN = 0.10  # 10% drawdown from peak

# NEW: Dynamic Threshold Parameters
VOLATILITY_HIGH = 0.02  # 2% volatility threshold
VOLATILITY_LOW = 0.01  # 1% volatility threshold
ML_THRESHOLD_HIGH_VOL = 0.65  # Exit faster in high volatility
ML_THRESHOLD_LOW_VOL = 0.75  # Exit slower in low volatility

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Capital & Fees
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005  # 0.05%

# Load models
print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

# Exit Models
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
with open(long_exit_scaler_path, 'rb') as f:
    long_exit_scaler = pickle.load(f)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
with open(short_exit_scaler_path, 'rb') as f:
    short_exit_scaler = pickle.load(f)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ All models loaded\n")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Load data
print("Loading historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_full):,} candles\n")

# Calculate features
print("Calculating features...")
start_time = time.time()
df = calculate_all_features(df_full)
df = prepare_exit_features(df)
print(f"  ✅ Features calculated ({time.time() - start_time:.1f}s)\n")

# Pre-calculate signals
print("Pre-calculating signals...")
long_feat = df[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
df['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

short_feat = df[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
df['short_prob'] = short_model.predict_proba(short_feat_scaled)[:, 1]
print(f"  ✅ Signals ready\n")


def calculate_market_volatility(window_df, i, lookback=20):
    """Calculate recent market volatility (rolling std of returns)"""
    if i < lookback:
        lookback = i

    if lookback < 2:
        return 0.015  # Default mid-range volatility

    recent_prices = window_df['close'].iloc[max(0, i-lookback):i+1]
    returns = recent_prices.pct_change().dropna()

    if len(returns) < 2:
        return 0.015

    return returns.std()


def check_exit_baseline(position, current_price, i, window_df, exit_prob):
    """BASELINE: Current system (ML Exit + Emergency)"""
    should_exit = False
    exit_reason = None

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
    time_in_pos = i - position['entry_idx']

    # 1. ML Exit
    if exit_prob >= ML_EXIT_THRESHOLD_BASE:
        should_exit = True
        exit_reason = 'ml_exit'

    # 2. Emergency Stop Loss
    if not should_exit and leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
        should_exit = True
        exit_reason = 'emergency_stop_loss'

    # 3. Emergency Max Hold
    if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
        should_exit = True
        exit_reason = 'emergency_max_hold'

    return should_exit, exit_reason, leveraged_pnl_pct, pnl_usd


def check_exit_take_profit(position, current_price, i, window_df, exit_prob):
    """TAKE_PROFIT: Baseline + Take Profit logic"""
    should_exit = False
    exit_reason = None

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
    time_in_pos = i - position['entry_idx']

    # Track peak profit for trailing TP
    if 'peak_pnl_pct' not in position:
        position['peak_pnl_pct'] = leveraged_pnl_pct
    else:
        position['peak_pnl_pct'] = max(position['peak_pnl_pct'], leveraged_pnl_pct)

    # NEW: Fixed Take Profit (checked FIRST)
    if leveraged_pnl_pct >= FIXED_TAKE_PROFIT:
        should_exit = True
        exit_reason = 'fixed_take_profit'

    # NEW: Trailing Take Profit
    if not should_exit and position['peak_pnl_pct'] >= TRAILING_TP_ACTIVATION:
        drawdown_from_peak = (position['peak_pnl_pct'] - leveraged_pnl_pct) / position['peak_pnl_pct']
        if drawdown_from_peak >= TRAILING_TP_DRAWDOWN:
            should_exit = True
            exit_reason = 'trailing_take_profit'

    # Original: ML Exit
    if not should_exit and exit_prob >= ML_EXIT_THRESHOLD_BASE:
        should_exit = True
        exit_reason = 'ml_exit'

    # Emergency Stop Loss
    if not should_exit and leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
        should_exit = True
        exit_reason = 'emergency_stop_loss'

    # Emergency Max Hold
    if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
        should_exit = True
        exit_reason = 'emergency_max_hold'

    return should_exit, exit_reason, leveraged_pnl_pct, pnl_usd


def check_exit_dynamic_threshold(position, current_price, i, window_df, exit_prob):
    """DYNAMIC_THRESHOLD: Baseline + Volatility-based ML threshold"""
    should_exit = False
    exit_reason = None

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
    time_in_pos = i - position['entry_idx']

    # NEW: Calculate market volatility
    volatility = calculate_market_volatility(window_df, i)

    # NEW: Adjust ML Exit threshold based on volatility
    if volatility > VOLATILITY_HIGH:
        ml_threshold = ML_THRESHOLD_HIGH_VOL  # 0.65 - exit faster
    elif volatility < VOLATILITY_LOW:
        ml_threshold = ML_THRESHOLD_LOW_VOL  # 0.75 - exit slower
    else:
        ml_threshold = ML_EXIT_THRESHOLD_BASE  # 0.70 - normal

    # ML Exit with dynamic threshold
    if exit_prob >= ml_threshold:
        should_exit = True
        exit_reason = f'ml_exit_vol_{volatility:.4f}'

    # Emergency Stop Loss
    if not should_exit and leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
        should_exit = True
        exit_reason = 'emergency_stop_loss'

    # Emergency Max Hold
    if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
        should_exit = True
        exit_reason = 'emergency_max_hold'

    return should_exit, exit_reason, leveraged_pnl_pct, pnl_usd


def check_exit_combined(position, current_price, i, window_df, exit_prob):
    """COMBINED: Take Profit + Dynamic Threshold"""
    should_exit = False
    exit_reason = None

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
    time_in_pos = i - position['entry_idx']

    # Track peak profit
    if 'peak_pnl_pct' not in position:
        position['peak_pnl_pct'] = leveraged_pnl_pct
    else:
        position['peak_pnl_pct'] = max(position['peak_pnl_pct'], leveraged_pnl_pct)

    # Fixed Take Profit
    if leveraged_pnl_pct >= FIXED_TAKE_PROFIT:
        should_exit = True
        exit_reason = 'fixed_take_profit'

    # Trailing Take Profit
    if not should_exit and position['peak_pnl_pct'] >= TRAILING_TP_ACTIVATION:
        drawdown_from_peak = (position['peak_pnl_pct'] - leveraged_pnl_pct) / position['peak_pnl_pct']
        if drawdown_from_peak >= TRAILING_TP_DRAWDOWN:
            should_exit = True
            exit_reason = 'trailing_take_profit'

    # Dynamic ML Exit
    if not should_exit:
        volatility = calculate_market_volatility(window_df, i)

        if volatility > VOLATILITY_HIGH:
            ml_threshold = ML_THRESHOLD_HIGH_VOL
        elif volatility < VOLATILITY_LOW:
            ml_threshold = ML_THRESHOLD_LOW_VOL
        else:
            ml_threshold = ML_EXIT_THRESHOLD_BASE

        if exit_prob >= ml_threshold:
            should_exit = True
            exit_reason = f'ml_exit_vol_{volatility:.4f}'

    # Emergency Stop Loss
    if not should_exit and leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
        should_exit = True
        exit_reason = 'emergency_stop_loss'

    # Emergency Max Hold
    if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
        should_exit = True
        exit_reason = 'emergency_max_hold'

    return should_exit, exit_reason, leveraged_pnl_pct, pnl_usd


def backtest_with_exit_strategy(window_df, exit_strategy):
    """
    Run backtest with specified exit strategy

    Args:
        window_df: Data window
        exit_strategy: 'baseline', 'take_profit', 'dynamic_threshold', 'combined'
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL

    # Select exit function
    if exit_strategy == 'baseline':
        check_exit = check_exit_baseline
    elif exit_strategy == 'take_profit':
        check_exit = check_exit_take_profit
    elif exit_strategy == 'dynamic_threshold':
        check_exit = check_exit_dynamic_threshold
    elif exit_strategy == 'combined':
        check_exit = check_exit_combined
    else:
        raise ValueError(f"Unknown exit strategy: {exit_strategy}")

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Entry logic (SAME for all strategies)
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
                    'quantity': sizing_result['leveraged_value'] / current_price
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
                        'quantity': sizing_result['leveraged_value'] / current_price
                    }

        # Exit logic (DIFFERENT per strategy)
        if position is not None:
            # Get ML exit probability
            try:
                if position['side'] == 'LONG':
                    exit_model = long_exit_model
                    exit_scaler = long_exit_scaler
                    exit_features_list = long_exit_feature_columns
                else:
                    exit_model = short_exit_model
                    exit_scaler = short_exit_scaler
                    exit_features_list = short_exit_feature_columns

                exit_features_values = window_df[exit_features_list].iloc[i:i+1].values
                exit_features_scaled = exit_scaler.transform(exit_features_values)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]
            except:
                exit_prob = 0.0

            # Check exit with strategy-specific logic
            should_exit, exit_reason, leveraged_pnl_pct, pnl_usd = check_exit(
                position, current_price, i, window_df, exit_prob
            )

            if should_exit:
                # Calculate fees
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission
                net_pnl_usd = pnl_usd - total_commission

                # Update capital
                capital += net_pnl_usd

                # Record trade
                trades.append({
                    'side': position['side'],
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'commission_usd': total_commission,
                    'net_pnl_usd': net_pnl_usd,
                    'hold_time': i - position['entry_idx'],
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct']
                })

                position = None

    return trades, capital


def run_comparison():
    """Run all 4 strategies and compare"""
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    strategies = ['baseline', 'take_profit', 'dynamic_threshold', 'combined']
    all_results = {s: [] for s in strategies}

    print("="*80)
    print("RUNNING COMPARISON")
    print("="*80)
    print()

    for strategy in strategies:
        print(f"Testing: {strategy.upper()}")
        start_time = time.time()

        for window_idx in range(num_windows):
            start_idx = window_idx * step_size
            end_idx = start_idx + window_size

            if end_idx > len(df):
                break

            window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            trades, final_capital = backtest_with_exit_strategy(window_df, strategy)

            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                total_return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

                all_results[strategy].append({
                    'window': window_idx,
                    'total_trades': len(trades),
                    'long_trades': len(trades_df[trades_df['side'] == 'LONG']),
                    'short_trades': len(trades_df[trades_df['side'] == 'SHORT']),
                    'total_return_pct': total_return_pct,
                    'win_rate': (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades) * 100,
                    'final_capital': final_capital
                })

        elapsed = time.time() - start_time
        print(f"  ✅ Complete ({elapsed:.1f}s)\n")

    return all_results


# Run comparison
start_time = time.time()
all_results = run_comparison()
total_time = time.time() - start_time

# Analysis
print("="*80)
print("COMPARISON RESULTS")
print("="*80)
print()

summary = []

for strategy in ['baseline', 'take_profit', 'dynamic_threshold', 'combined']:
    results_df = pd.DataFrame(all_results[strategy])

    if len(results_df) > 0:
        avg_return = results_df['total_return_pct'].mean()
        std_return = results_df['total_return_pct'].std()
        sharpe = avg_return / std_return if std_return > 0 else 0
        win_rate = results_df['win_rate'].mean()
        avg_trades = results_df['total_trades'].mean()

        summary.append({
            'strategy': strategy,
            'avg_return_pct': avg_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'avg_trades': avg_trades,
            'windows': len(results_df)
        })

        print(f"📊 {strategy.upper()}")
        print(f"  Avg Return: {avg_return:.2f}% per window")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Avg Trades: {avg_trades:.1f}")
        print()

# Create summary table
summary_df = pd.DataFrame(summary)

if len(summary_df) > 0:
    # Sort by Sharpe Ratio
    summary_df = summary_df.sort_values('sharpe_ratio', ascending=False)

    print("="*80)
    print("SUMMARY TABLE (Sorted by Sharpe Ratio)")
    print("="*80)
    print()
    print(summary_df[['strategy', 'avg_return_pct', 'sharpe_ratio', 'win_rate', 'avg_trades']].to_string(index=False))
    print()

    # Calculate improvements vs baseline
    baseline = summary_df[summary_df['strategy'] == 'baseline'].iloc[0]

    print("="*80)
    print("IMPROVEMENT vs BASELINE")
    print("="*80)
    print()

    for _, row in summary_df.iterrows():
        if row['strategy'] != 'baseline':
            return_improvement = ((row['avg_return_pct'] - baseline['avg_return_pct']) / baseline['avg_return_pct'] * 100)
            sharpe_improvement = ((row['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100)

            print(f"📈 {row['strategy'].upper()}")
            print(f"  Return: {row['avg_return_pct']:.2f}% ({return_improvement:+.1f}%)")
            print(f"  Sharpe: {row['sharpe_ratio']:.3f} ({sharpe_improvement:+.1f}%)")
            print(f"  Win Rate: {row['win_rate']:.1f}% ({row['win_rate'] - baseline['win_rate']:+.1f}pp)")
            print()

    # Save results
    output_file = RESULTS_DIR / f"exit_strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")

    print(f"\nTotal test time: {total_time:.1f}s")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
