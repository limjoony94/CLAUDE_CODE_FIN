"""
Idea 4: Hybrid Position Sizing
================================

Concept: Always keep 10% reserve capital for opportunity switching
- Active position: 90% of capital
- Reserve: 10% for better opportunities
- Dynamic switching when significantly better signal appears
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("IDEA 4: HYBRID POSITION SIZING")
print("="*80)
print("\nConcept: 90% active + 10% reserve for dynamic switching\n")

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

print(f"  ✅ Models loaded\n")

# Load and prepare data
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df)} candles\n")

print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().bfill().fillna(0)
print(f"  ✅ Features calculated\n")


def backtest_hybrid_sizing(df, window_size=1440, step_size=288,
                           long_threshold=0.65, short_threshold=0.70,
                           reserve_ratio=0.10, switch_improvement_threshold=0.15,
                           long_avg_return=0.0041, short_avg_return=0.0047):
    """
    Backtest with Hybrid Position Sizing

    Parameters:
    - reserve_ratio: Reserve capital ratio (default 10%)
    - switch_improvement_threshold: Min improvement to switch (default 15%)
    """

    results = []
    num_windows = (len(df) - window_size) // step_size

    print(f"Backtesting Hybrid Position Sizing:")
    print(f"  Reserve Ratio: {reserve_ratio*100:.0f}%")
    print(f"  Active Position: {(1-reserve_ratio)*100:.0f}%")
    print(f"  Switch Threshold: {switch_improvement_threshold*100:.0f}% improvement")
    print(f"  Testing {num_windows} windows...\n")

    switch_stats = {'attempted': 0, 'executed': 0, 'rejected': 0}

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Position tracking
        active_position = None  # 90% capital
        reserve_position = None  # 10% capital (for switching)
        trades = []

        for i in range(len(window_df) - 1):
            current_candle = window_df.iloc[i]

            # Get signals
            try:
                long_feat = window_df[long_feature_columns].iloc[i:i+1].values
                long_feat_scaled = long_scaler.transform(long_feat)
                long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
            except:
                long_prob = 0.0

            try:
                short_feat = window_df[short_feature_columns].iloc[i:i+1].values
                short_feat_scaled = short_scaler.transform(short_feat)
                short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
            except:
                short_prob = 0.0

            # Calculate expected values
            long_ev = long_prob * long_avg_return
            short_ev = short_prob * short_avg_return

            # Entry/Switch logic
            if active_position is None and reserve_position is None:
                # No position - enter normally
                if long_prob >= long_threshold:
                    active_position = {
                        'side': 'LONG',
                        'entry_idx': i,
                        'entry_price': current_candle['close'],
                        'size': 1 - reserve_ratio,  # 90%
                        'probability': long_prob,
                        'expected_value': long_ev
                    }
                elif short_prob >= short_threshold:
                    active_position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': current_candle['close'],
                        'size': 1 - reserve_ratio,  # 90%
                        'probability': short_prob,
                        'expected_value': short_ev
                    }

            elif active_position is not None and reserve_position is None:
                # Have active position - check for switching opportunity
                current_ev = active_position['expected_value']
                time_in_position = i - active_position['entry_idx']

                # Calculate current position P&L
                if active_position['side'] == 'LONG':
                    current_pnl = (current_candle['close'] - active_position['entry_price']) / active_position['entry_price']
                else:
                    current_pnl = (active_position['entry_price'] - current_candle['close']) / active_position['entry_price']

                # Check for better opposite direction signal
                should_switch = False
                new_side = None
                new_ev = 0

                if active_position['side'] == 'LONG' and short_prob >= short_threshold:
                    # Currently LONG, SHORT signal appears
                    improvement = (short_ev - current_ev) / current_ev
                    if improvement > switch_improvement_threshold:
                        should_switch = True
                        new_side = 'SHORT'
                        new_ev = short_ev

                elif active_position['side'] == 'SHORT' and long_prob >= long_threshold:
                    # Currently SHORT, LONG signal appears
                    improvement = (long_ev - current_ev) / current_ev
                    if improvement > switch_improvement_threshold:
                        should_switch = True
                        new_side = 'LONG'
                        new_ev = long_ev

                if should_switch:
                    switch_stats['attempted'] += 1

                    # Additional safety checks
                    not_deeply_profitable = current_pnl < 0.015  # < 1.5%
                    not_too_early = time_in_position >= 10  # At least 10 candles

                    if not_deeply_profitable and not_too_early:
                        # Execute switch: Open reserve position
                        switch_stats['executed'] += 1

                        reserve_position = {
                            'side': new_side,
                            'entry_idx': i,
                            'entry_price': current_candle['close'],
                            'size': reserve_ratio,  # 10%
                            'probability': short_prob if new_side == 'SHORT' else long_prob,
                            'expected_value': new_ev,
                            'is_switch': True
                        }
                    else:
                        switch_stats['rejected'] += 1

            # Exit logic for active position
            if active_position is not None:
                time_in_position = i - active_position['entry_idx']

                if active_position['side'] == 'LONG':
                    pnl_pct = (current_candle['close'] - active_position['entry_price']) / active_position['entry_price']
                else:
                    pnl_pct = (active_position['entry_price'] - current_candle['close']) / active_position['entry_price']

                should_exit = False
                exit_reason = None

                # Standard exit conditions
                if time_in_position >= 240:
                    should_exit = True
                    exit_reason = 'max_hold'
                elif pnl_pct >= 0.03:
                    should_exit = True
                    exit_reason = 'take_profit'
                elif pnl_pct <= -0.015:
                    should_exit = True
                    exit_reason = 'stop_loss'

                if should_exit:
                    # Close active position
                    trades.append({
                        'side': active_position['side'],
                        'entry_price': active_position['entry_price'],
                        'exit_price': current_candle['close'],
                        'pnl_pct': pnl_pct,
                        'size': active_position['size'],
                        'hold_time': time_in_position,
                        'exit_reason': exit_reason,
                        'position_type': 'active'
                    })

                    # If we have reserve position, promote it to active
                    if reserve_position is not None:
                        # Scale up reserve to active size
                        active_position = {
                            'side': reserve_position['side'],
                            'entry_idx': reserve_position['entry_idx'],
                            'entry_price': reserve_position['entry_price'],
                            'size': 1 - reserve_ratio,  # Scale to 90%
                            'probability': reserve_position['probability'],
                            'expected_value': reserve_position['expected_value']
                        }
                        reserve_position = None
                    else:
                        active_position = None

            # Exit logic for reserve position (if exists independently)
            if reserve_position is not None and active_position is None:
                time_in_position = i - reserve_position['entry_idx']

                if reserve_position['side'] == 'LONG':
                    pnl_pct = (current_candle['close'] - reserve_position['entry_price']) / reserve_position['entry_price']
                else:
                    pnl_pct = (reserve_position['entry_price'] - current_candle['close']) / reserve_position['entry_price']

                should_exit = False
                exit_reason = None

                if time_in_position >= 240:
                    should_exit = True
                    exit_reason = 'max_hold'
                elif pnl_pct >= 0.03:
                    should_exit = True
                    exit_reason = 'take_profit'
                elif pnl_pct <= -0.015:
                    should_exit = True
                    exit_reason = 'stop_loss'

                if should_exit:
                    trades.append({
                        'side': reserve_position['side'],
                        'entry_price': reserve_position['entry_price'],
                        'exit_price': current_candle['close'],
                        'pnl_pct': pnl_pct,
                        'size': reserve_position['size'],
                        'hold_time': time_in_position,
                        'exit_reason': exit_reason,
                        'position_type': 'reserve'
                    })
                    reserve_position = None

        # Calculate window metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)

            # Weight P&L by position size
            trades_df['weighted_pnl'] = trades_df['pnl_pct'] * trades_df['size']

            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']

            results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'avg_return': trades_df['weighted_pnl'].mean() * 100,
                'total_return': trades_df['weighted_pnl'].sum() * 100,
                'win_rate': (trades_df['pnl_pct'] > 0).sum() / len(trades) * 100,
                'long_wr': (long_trades['pnl_pct'] > 0).sum() / len(long_trades) * 100 if len(long_trades) > 0 else 0,
                'short_wr': (short_trades['pnl_pct'] > 0).sum() / len(short_trades) * 100 if len(short_trades) > 0 else 0,
            })
        else:
            results.append({
                'window': window_idx,
                'total_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'avg_return': 0,
                'total_return': 0,
                'win_rate': 0,
                'long_wr': 0,
                'short_wr': 0,
            })

    results_df = pd.DataFrame(results)

    # Switch statistics
    print(f"\n  Switch Statistics:")
    print(f"    Attempted: {switch_stats['attempted']}")
    print(f"    Executed: {switch_stats['executed']}")
    print(f"    Rejected: {switch_stats['rejected']}")

    return results_df, switch_stats


# Test different configurations
config_combinations = [
    {'reserve': 0.10, 'switch_thresh': 0.15, 'desc': 'Standard (10% reserve, 15% switch)'},
    {'reserve': 0.10, 'switch_thresh': 0.20, 'desc': 'Conservative switching'},
    {'reserve': 0.15, 'switch_thresh': 0.15, 'desc': 'Higher reserve (15%)'},
    {'reserve': 0.05, 'switch_thresh': 0.15, 'desc': 'Lower reserve (5%)'},
]

all_results = []

for params in config_combinations:
    print(f"\n{'='*80}")
    print(f"Testing: {params['desc']}")
    print(f"{'='*80}\n")

    results_df, switch_stats = backtest_hybrid_sizing(
        df,
        reserve_ratio=params['reserve'],
        switch_improvement_threshold=params['switch_thresh']
    )

    # Summary statistics
    avg_return = results_df['total_return'].mean()
    avg_trades = results_df['total_trades'].mean()
    avg_long = results_df['long_trades'].mean()
    avg_short = results_df['short_trades'].mean()
    avg_wr = results_df[results_df['win_rate'] > 0]['win_rate'].mean()

    print(f"\nRESULTS ({params['desc']}):")
    print(f"  Avg Return per Window: {avg_return:.2f}%")
    print(f"  Avg Trades per Window: {avg_trades:.1f}")
    print(f"    - LONG: {avg_long:.1f}")
    print(f"    - SHORT: {avg_short:.1f}")
    print(f"  Overall Win Rate: {avg_wr:.1f}%")

    all_results.append({
        'config': params['desc'],
        'reserve_ratio': params['reserve'],
        'switch_threshold': params['switch_thresh'],
        'avg_return': avg_return,
        'avg_trades': avg_trades,
        'avg_long': avg_long,
        'avg_short': avg_short,
        'win_rate': avg_wr,
        'switches_executed': switch_stats['executed']
    })

# Final comparison
print(f"\n{'='*80}")
print("IDEA 4 FINAL RESULTS COMPARISON")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values('avg_return', ascending=False)

print(comparison_df.to_string(index=False))

best = comparison_df.iloc[0]
print(f"\n🏆 BEST CONFIGURATION:")
print(f"  Config: {best['config']}")
print(f"  Avg Return: {best['avg_return']:.2f}% per window")
print(f"  Avg Trades: {best['avg_trades']:.1f} (LONG {best['avg_long']:.1f} + SHORT {best['avg_short']:.1f})")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Switches: {best['switches_executed']}")

# Save results
output_file = RESULTS_DIR / "idea4_hybrid_sizing_results.csv"
comparison_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print(f"\n{'='*80}")
print("IDEA 4 TEST COMPLETE")
print(f"{'='*80}\n")
