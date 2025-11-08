"""
Strategy Parameter Optimization

Purpose: Test if strategy parameters (TP/SL/Max Hold) can improve performance
         without changing the model

Methodology:
1. Grid search across:
   - Take Profit (TP): 1.5%, 2%, 2.5%, 3%, 3.5%, 4%
   - Stop Loss (SL): 0.5%, 0.75%, 1%, 1.25%, 1.5%
   - Max Holding: 2h, 3h, 4h, 5h, 6h
2. For each combination, measure returns and win rate
3. Identify optimal parameters

This addresses: "Can we improve 5-10% just by tuning parameters?"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from itertools import product
import time

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Fixed parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
TRANSACTION_COST = 0.0002  # 0.02% maker fee
THRESHOLD = 0.7  # Use optimal threshold from sensitivity analysis

# Parameter ranges to test
TP_RANGE = [0.015, 0.020, 0.025, 0.030, 0.035, 0.040]  # 1.5% to 4%
SL_RANGE = [0.005, 0.0075, 0.010, 0.0125, 0.015]  # 0.5% to 1.5%
MAX_HOLD_RANGE = [2, 3, 4, 5, 6]  # 2h to 6h

# Quick mode: Test subset for faster iteration
QUICK_MODE = True  # Set to False for full grid search

if QUICK_MODE:
    # Test fewer combinations for faster results
    TP_RANGE = [0.020, 0.025, 0.030, 0.035, 0.040]
    SL_RANGE = [0.0075, 0.010, 0.0125]
    MAX_HOLD_RANGE = [3, 4, 5]


def classify_market_regime(df_window):
    """Classify market regime based on price movement"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


def backtest_with_params(df, long_model, short_model, long_scaler, short_scaler,
                         feature_columns, tp, sl, max_hold_hours):
    """Backtest with specific strategy parameters"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L based on position side
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Check exit conditions (with current parameters)
            exit_reason = None
            if pnl_pct <= -sl:
                exit_reason = "Stop Loss"
            elif pnl_pct >= tp:
                exit_reason = "Take Profit"
            elif hours_held >= max_hold_hours:
                exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                # Transaction costs
                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'side': side,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held
                })

                position = None

        # Look for entry (LONG or SHORT)
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            # Get predictions from BOTH models
            features_scaled_long = long_scaler.transform(features)
            long_prob = long_model.predict_proba(features_scaled_long)[0][1]

            features_scaled_short = short_scaler.transform(features)
            short_prob = short_model.predict_proba(features_scaled_short)[0][1]

            # Determine entry direction
            side = None

            if long_prob >= THRESHOLD and short_prob < THRESHOLD:
                side = 'LONG'
            elif short_prob >= THRESHOLD and long_prob < THRESHOLD:
                side = 'SHORT'
            elif long_prob >= THRESHOLD and short_prob >= THRESHOLD:
                side = 'LONG' if long_prob > short_prob else 'SHORT'

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity
                }

    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_pct': 0.0,
            'sharpe_ratio': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) * 100

    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_pnl_pct': avg_pnl_pct,
        'sharpe_ratio': sharpe
    }


def rolling_window_backtest_params(df, long_model, short_model, long_scaler, short_scaler,
                                   feature_columns, tp, sl, max_hold_hours):
    """Rolling window backtest with specific parameters"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        metrics = backtest_with_params(
            window_df, long_model, short_model, long_scaler, short_scaler,
            feature_columns, tp, sl, max_hold_hours
        )

        windows.append(metrics)
        start_idx += WINDOW_SIZE

    # Aggregate across windows
    return {
        'avg_return': np.mean([w['total_return_pct'] for w in windows]),
        'std_return': np.std([w['total_return_pct'] for w in windows]),
        'avg_trades': np.mean([w['num_trades'] for w in windows]),
        'avg_win_rate': np.mean([w['win_rate'] for w in windows]),
        'avg_sharpe': np.mean([w['sharpe_ratio'] for w in windows])
    }


def main():
    print("=" * 80)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 80)
    print("\n🎯 Purpose: Test if strategy parameters can improve performance")
    print("           without changing the model")

    if QUICK_MODE:
        print("\n⚡ QUICK MODE: Testing subset of parameter combinations")
        print(f"   TP Range: {[f'{x*100:.1f}%' for x in TP_RANGE]}")
        print(f"   SL Range: {[f'{x*100:.2f}%' for x in SL_RANGE]}")
        print(f"   Max Hold: {MAX_HOLD_RANGE}h")
        total_combinations = len(TP_RANGE) * len(SL_RANGE) * len(MAX_HOLD_RANGE)
    else:
        print("\n🔬 FULL MODE: Testing all parameter combinations")
        total_combinations = len(TP_RANGE) * len(SL_RANGE) * len(MAX_HOLD_RANGE)

    print(f"\n   Total Combinations: {total_combinations}")

    # Load models
    long_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    long_scaler_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
    short_model_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3.pkl"
    short_scaler_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_scaler.pkl"

    with open(long_model_file, 'rb') as f:
        long_model = pickle.load(f)
    with open(long_scaler_file, 'rb') as f:
        long_scaler = pickle.load(f)
    with open(short_model_file, 'rb') as f:
        short_model = pickle.load(f)
    with open(short_scaler_file, 'rb') as f:
        short_scaler = pickle.load(f)
    print(f"✅ Models loaded")

    # Load feature columns
    feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    print(f"✅ Features loaded: {len(feature_columns)} features")

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Data loaded: {len(df)} candles")

    # Calculate features
    print("\nCalculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ Features calculated: {len(df)} rows")

    # Grid search
    print("\n" + "=" * 80)
    print("🔍 Running Grid Search...")
    print("=" * 80)

    results = []
    start_time = time.time()

    for idx, (tp, sl, max_hold) in enumerate(product(TP_RANGE, SL_RANGE, MAX_HOLD_RANGE), 1):
        # Skip invalid combinations (SL > TP)
        if sl >= tp:
            continue

        print(f"\n[{idx}/{total_combinations}] Testing TP:{tp*100:.1f}% SL:{sl*100:.2f}% Hold:{max_hold}h...", end=" ")

        metrics = rolling_window_backtest_params(
            df, long_model, short_model, long_scaler, short_scaler,
            feature_columns, tp, sl, max_hold
        )

        results.append({
            'TP': tp * 100,
            'SL': sl * 100,
            'Max_Hold': max_hold,
            'Avg_Return': metrics['avg_return'],
            'Std_Return': metrics['std_return'],
            'Avg_Trades': metrics['avg_trades'],
            'Win_Rate': metrics['avg_win_rate'],
            'Sharpe': metrics['avg_sharpe']
        })

        print(f"Return: {metrics['avg_return']:+.2f}% | WR: {metrics['avg_win_rate']:.1f}%")

    elapsed = time.time() - start_time
    print(f"\n✅ Grid search complete in {elapsed:.1f}s ({elapsed/len(results):.1f}s per combination)")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Avg_Return', ascending=False)

    # Display top results
    print("\n" + "=" * 80)
    print("📊 TOP 10 PARAMETER COMBINATIONS (by Return)")
    print("=" * 80)
    print("\n" + results_df.head(10).to_string(index=False))

    # Baseline comparison (current parameters)
    baseline_tp = 3.0
    baseline_sl = 1.0
    baseline_hold = 4
    baseline_row = results_df[
        (results_df['TP'] == baseline_tp) &
        (results_df['SL'] == baseline_sl) &
        (results_df['Max_Hold'] == baseline_hold)
    ]

    print("\n" + "=" * 80)
    print("🎯 OPTIMIZATION RESULTS")
    print("=" * 80)

    if len(baseline_row) > 0:
        baseline_return = baseline_row['Avg_Return'].values[0]
        baseline_wr = baseline_row['Win_Rate'].values[0]

        best_row = results_df.iloc[0]
        best_return = best_row['Avg_Return']
        best_wr = best_row['Win_Rate']

        print(f"\n  Current Parameters (Baseline):")
        print(f"    TP: {baseline_tp}% | SL: {baseline_sl}% | Max Hold: {baseline_hold}h")
        print(f"    Return: {baseline_return:+.2f}%")
        print(f"    Win Rate: {baseline_wr:.1f}%")

        print(f"\n  Best Parameters (Optimized):")
        print(f"    TP: {best_row['TP']:.1f}% | SL: {best_row['SL']:.2f}% | Max Hold: {best_row['Max_Hold']:.0f}h")
        print(f"    Return: {best_return:+.2f}%")
        print(f"    Win Rate: {best_wr:.1f}%")

        improvement_return = best_return - baseline_return
        improvement_wr = best_wr - baseline_wr

        print(f"\n  Improvement:")
        print(f"    Return: {improvement_return:+.2f}% ({improvement_return/abs(baseline_return)*100:+.1f}%)")
        print(f"    Win Rate: {improvement_wr:+.1f}%p")

        if improvement_return > 0.5:
            print(f"\n  ✅ OPTIMIZATION SUCCESSFUL!")
            print(f"     Strategy parameters can improve performance by {improvement_return:.2f}%")
            print(f"     Recommend updating to optimized parameters")
        else:
            print(f"\n  ⚠️ MARGINAL IMPROVEMENT")
            print(f"     Current parameters already near-optimal")
    else:
        print("\n  ⚠️ Baseline parameters not in tested range")
        best_row = results_df.iloc[0]
        print(f"\n  Best Parameters Found:")
        print(f"    TP: {best_row['TP']:.1f}% | SL: {best_row['SL']:.2f}% | Max Hold: {best_row['Max_Hold']:.0f}h")
        print(f"    Return: {best_row['Avg_Return']:+.2f}%")
        print(f"    Win Rate: {best_row['Win_Rate']:.1f}%")

    # Save results
    output_file = RESULTS_DIR / "strategy_parameter_optimization.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Full results saved: {output_file}")

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    if QUICK_MODE:
        print("\n💡 Tip: Set QUICK_MODE=False for full grid search")


if __name__ == "__main__":
    main()
