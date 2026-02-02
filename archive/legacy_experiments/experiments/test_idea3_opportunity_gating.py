"""
Idea 3: Opportunity Cost Gating
=================================

Concept: Only enter SHORT when opportunity cost is acceptable
- Calculate expected value of both LONG and SHORT
- SHORT entry requires clear advantage over LONG
- Preserves LONG opportunities when they're competitive
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
print("IDEA 3: OPPORTUNITY COST GATING")
print("="*80)
print("\nConcept: SHORT only when clearly better than LONG\n")

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


def backtest_opportunity_gating(df, window_size=1440, step_size=288,
                                long_threshold=0.65, short_threshold=0.70,
                                gate_threshold=0.0015,
                                long_avg_return=0.0041, short_avg_return=0.0047):
    """
    Backtest with Opportunity Cost Gating

    Parameters:
    - gate_threshold: Minimum opportunity cost to enter SHORT (default 0.15%)
    - long_avg_return: Average LONG return (0.41%)
    - short_avg_return: Average SHORT return (0.47%)
    """

    results = []
    num_windows = (len(df) - window_size) // step_size

    print(f"Backtesting Opportunity Cost Gating:")
    print(f"  Gate Threshold: {gate_threshold*100:.2f}%")
    print(f"  LONG avg return: {long_avg_return*100:.2f}%")
    print(f"  SHORT avg return: {short_avg_return*100:.2f}%")
    print(f"  Testing {num_windows} windows...\n")

    gating_stats = {'blocked': 0, 'allowed': 0, 'total_short_signals': 0}

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Backtest this window
        position = None
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

            # Entry logic with OPPORTUNITY COST GATING
            if position is None and i < len(window_df) - 1:
                # Priority 1: LONG signal
                if long_prob >= long_threshold:
                    position = {
                        'side': 'LONG',
                        'entry_idx': i,
                        'entry_price': current_candle['close'],
                        'probability': long_prob
                    }

                # Priority 2: SHORT with gating
                elif short_prob >= short_threshold:
                    gating_stats['total_short_signals'] += 1

                    # Calculate expected values
                    long_ev = long_prob * long_avg_return
                    short_ev = short_prob * short_avg_return
                    opportunity_cost = short_ev - long_ev

                    if opportunity_cost > gate_threshold:
                        # SHORT advantage is sufficient
                        gating_stats['allowed'] += 1
                        position = {
                            'side': 'SHORT',
                            'entry_idx': i,
                            'entry_price': current_candle['close'],
                            'probability': short_prob,
                            'opportunity_cost': opportunity_cost
                        }
                    else:
                        # Block SHORT - not worth sacrificing LONG opportunity
                        gating_stats['blocked'] += 1

            # Exit logic (standard)
            if position is not None:
                time_in_position = i - position['entry_idx']

                if position['side'] == 'LONG':
                    pnl_pct = (current_candle['close'] - position['entry_price']) / position['entry_price']
                else:  # SHORT
                    pnl_pct = (position['entry_price'] - current_candle['close']) / position['entry_price']

                should_exit = False
                exit_reason = None

                # Exit conditions
                if time_in_position >= 240:  # 4 hours
                    should_exit = True
                    exit_reason = 'max_hold'
                elif pnl_pct >= 0.03:  # 3% TP
                    should_exit = True
                    exit_reason = 'take_profit'
                elif pnl_pct <= -0.015:  # 1.5% SL
                    should_exit = True
                    exit_reason = 'stop_loss'

                if should_exit:
                    trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_candle['close'],
                        'pnl_pct': pnl_pct,
                        'hold_time': time_in_position,
                        'exit_reason': exit_reason,
                    })
                    position = None

        # Calculate window metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)

            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']

            results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'avg_return': trades_df['pnl_pct'].mean() * 100,
                'total_return': trades_df['pnl_pct'].sum() * 100,
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

    # Add gating statistics
    if gating_stats['total_short_signals'] > 0:
        block_rate = gating_stats['blocked'] / gating_stats['total_short_signals'] * 100
    else:
        block_rate = 0

    print(f"\n  Gating Statistics:")
    print(f"    Total SHORT signals: {gating_stats['total_short_signals']}")
    print(f"    Allowed: {gating_stats['allowed']}")
    print(f"    Blocked: {gating_stats['blocked']} ({block_rate:.1f}%)")

    return results_df, gating_stats


# Test different gate thresholds
gate_threshold_combinations = [
    {'gate': 0.0010, 'desc': 'Gate 0.10% (lenient)'},
    {'gate': 0.0015, 'desc': 'Gate 0.15% (recommended)'},
    {'gate': 0.0020, 'desc': 'Gate 0.20% (moderate)'},
    {'gate': 0.0025, 'desc': 'Gate 0.25% (strict)'},
]

all_results = []

for params in gate_threshold_combinations:
    print(f"\n{'='*80}")
    print(f"Testing: {params['desc']}")
    print(f"{'='*80}\n")

    results_df, gating_stats = backtest_opportunity_gating(
        df,
        gate_threshold=params['gate']
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
        'gate_threshold': params['gate'],
        'avg_return': avg_return,
        'avg_trades': avg_trades,
        'avg_long': avg_long,
        'avg_short': avg_short,
        'win_rate': avg_wr,
        'blocked_pct': gating_stats['blocked'] / gating_stats['total_short_signals'] * 100 if gating_stats['total_short_signals'] > 0 else 0
    })

# Final comparison
print(f"\n{'='*80}")
print("IDEA 3 FINAL RESULTS COMPARISON")
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
print(f"  SHORT blocked: {best['blocked_pct']:.1f}%")

# Save results
output_file = RESULTS_DIR / "idea3_opportunity_gating_results.csv"
comparison_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print(f"\n{'='*80}")
print("IDEA 3 TEST COMPLETE")
print(f"{'='*80}\n")
