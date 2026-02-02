"""
Unified Strategy Comparison
============================

Test ALL 5 strategies with same data and framework:
1. LONG-only (baseline)
2. Signal Fusion
3. Asymmetric Time
4. Opportunity Cost Gating
5. Hybrid Position Sizing

Fair comparison - same data, same conditions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("UNIFIED STRATEGY COMPARISON - ALL 5 STRATEGIES")
print("="*80)
print("\n공정한 비교를 위한 통합 테스트\n")

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

# Load data - SAMPLE for speed (15,000 candles = ~52 days)
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
df = df_full.tail(15000).reset_index(drop=True)
print(f"  ✅ Data loaded: {len(df)} candles (~{len(df)//288:.0f} days)\n")

print("Calculating features (한 번만 - LONG + SHORT all features)...")
start_time = time.time()
df = calculate_all_features(df)
feature_time = time.time() - start_time
print(f"  ✅ All features calculated ({feature_time:.1f}s)\n")

# Pre-calculate all signals (한 번만!)
print("Pre-calculating all signals...")
long_probs = []
short_probs = []

for i in range(len(df)):
    try:
        long_feat = df[long_feature_columns].iloc[i:i+1].values
        long_feat_scaled = long_scaler.transform(long_feat)
        long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
    except:
        long_prob = 0.0
    long_probs.append(long_prob)

    try:
        short_feat = df[short_feature_columns].iloc[i:i+1].values
        short_feat_scaled = short_scaler.transform(short_feat)
        short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
    except:
        short_prob = 0.0
    short_probs.append(short_prob)

df['long_prob'] = long_probs
df['short_prob'] = short_probs
print(f"  ✅ Signals pre-calculated\n")


def run_window_backtest(df, strategy_func, strategy_name):
    """Run backtest across multiple windows"""
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    results = []

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Run strategy on this window
        trades = strategy_func(window_df)

        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)

            # Handle weighted P&L if size column exists
            if 'size' in trades_df.columns:
                trades_df['weighted_pnl'] = trades_df['pnl_pct'] * trades_df['size']
                avg_return = trades_df['weighted_pnl'].mean() * 100
                total_return = trades_df['weighted_pnl'].sum() * 100
            else:
                avg_return = trades_df['pnl_pct'].mean() * 100
                total_return = trades_df['pnl_pct'].sum() * 100

            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']

            results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'avg_return': avg_return,
                'total_return': total_return,
                'win_rate': (trades_df['pnl_pct'] > 0).sum() / len(trades) * 100,
            })

    return pd.DataFrame(results)


# Strategy 1: LONG-only Baseline
def strategy_long_only(window_df, threshold=0.65):
    """LONG-only baseline"""
    trades = []
    position = None

    for i in range(len(window_df) - 1):
        if position is None:
            if window_df['long_prob'].iloc[i] >= threshold:
                position = {
                    'entry_idx': i,
                    'entry_price': window_df['close'].iloc[i]
                }

        if position is not None:
            time_in_pos = i - position['entry_idx']
            pnl_pct = (window_df['close'].iloc[i] - position['entry_price']) / position['entry_price']

            if time_in_pos >= 240 or pnl_pct >= 0.03 or pnl_pct <= -0.015:
                trades.append({
                    'side': 'LONG',
                    'pnl_pct': pnl_pct,
                    'hold_time': time_in_pos
                })
                position = None

    return trades


# Strategy 2: Signal Fusion
def strategy_signal_fusion(window_df, market_bias=0.10, fusion_threshold=0.20):
    """Signal fusion with market bias"""
    trades = []
    position = None

    for i in range(len(window_df) - 1):
        long_adj = window_df['long_prob'].iloc[i] * (1 + market_bias)
        short_adj = window_df['short_prob'].iloc[i] * (1 - market_bias)
        signal = long_adj - short_adj

        if position is None:
            if signal > fusion_threshold:
                position = {'side': 'LONG', 'entry_idx': i, 'entry_price': window_df['close'].iloc[i]}
            elif signal < -fusion_threshold:
                position = {'side': 'SHORT', 'entry_idx': i, 'entry_price': window_df['close'].iloc[i]}

        if position is not None:
            time_in_pos = i - position['entry_idx']
            if position['side'] == 'LONG':
                pnl_pct = (window_df['close'].iloc[i] - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - window_df['close'].iloc[i]) / position['entry_price']

            if time_in_pos >= 240 or pnl_pct >= 0.03 or pnl_pct <= -0.015:
                trades.append({'side': position['side'], 'pnl_pct': pnl_pct, 'hold_time': time_in_pos})
                position = None

    return trades


# Strategy 3: Asymmetric Time
def strategy_asymmetric_time(window_df, short_max_hold=60):
    """Asymmetric time horizon"""
    trades = []
    position = None

    for i in range(len(window_df) - 1):
        if position is None:
            if window_df['long_prob'].iloc[i] >= 0.65:
                position = {'side': 'LONG', 'entry_idx': i, 'entry_price': window_df['close'].iloc[i]}
            elif window_df['short_prob'].iloc[i] >= 0.70:
                position = {'side': 'SHORT', 'entry_idx': i, 'entry_price': window_df['close'].iloc[i]}

        if position is not None:
            time_in_pos = i - position['entry_idx']
            if position['side'] == 'LONG':
                pnl_pct = (window_df['close'].iloc[i] - position['entry_price']) / position['entry_price']
                max_hold = 240
                tp, sl = 0.03, -0.015
            else:
                pnl_pct = (position['entry_price'] - window_df['close'].iloc[i]) / position['entry_price']
                max_hold = short_max_hold  # Asymmetric!
                tp, sl = 0.02, -0.01

            if time_in_pos >= max_hold or pnl_pct >= tp or pnl_pct <= sl:
                trades.append({'side': position['side'], 'pnl_pct': pnl_pct, 'hold_time': time_in_pos})
                position = None

    return trades


# Strategy 4: Opportunity Gating
def strategy_opportunity_gating(window_df, gate_threshold=0.0015):
    """Opportunity cost gating"""
    trades = []
    position = None

    for i in range(len(window_df) - 1):
        if position is None:
            if window_df['long_prob'].iloc[i] >= 0.65:
                position = {'side': 'LONG', 'entry_idx': i, 'entry_price': window_df['close'].iloc[i]}
            elif window_df['short_prob'].iloc[i] >= 0.70:
                # Gate check
                long_ev = window_df['long_prob'].iloc[i] * 0.0041
                short_ev = window_df['short_prob'].iloc[i] * 0.0047
                if (short_ev - long_ev) > gate_threshold:
                    position = {'side': 'SHORT', 'entry_idx': i, 'entry_price': window_df['close'].iloc[i]}

        if position is not None:
            time_in_pos = i - position['entry_idx']
            if position['side'] == 'LONG':
                pnl_pct = (window_df['close'].iloc[i] - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - window_df['close'].iloc[i]) / position['entry_price']

            if time_in_pos >= 240 or pnl_pct >= 0.03 or pnl_pct <= -0.015:
                trades.append({'side': position['side'], 'pnl_pct': pnl_pct, 'hold_time': time_in_pos})
                position = None

    return trades


# Strategy 5: Hybrid Sizing (simplified for speed)
def strategy_hybrid_sizing(window_df, reserve_ratio=0.10):
    """Hybrid position sizing"""
    trades = []
    active_position = None

    for i in range(len(window_df) - 1):
        if active_position is None:
            if window_df['long_prob'].iloc[i] >= 0.65:
                active_position = {'side': 'LONG', 'entry_idx': i, 'entry_price': window_df['close'].iloc[i], 'size': 1-reserve_ratio}
            elif window_df['short_prob'].iloc[i] >= 0.70:
                active_position = {'side': 'SHORT', 'entry_idx': i, 'entry_price': window_df['close'].iloc[i], 'size': 1-reserve_ratio}

        if active_position is not None:
            time_in_pos = i - active_position['entry_idx']
            if active_position['side'] == 'LONG':
                pnl_pct = (window_df['close'].iloc[i] - active_position['entry_price']) / active_position['entry_price']
            else:
                pnl_pct = (active_position['entry_price'] - window_df['close'].iloc[i]) / active_position['entry_price']

            if time_in_pos >= 240 or pnl_pct >= 0.03 or pnl_pct <= -0.015:
                trades.append({
                    'side': active_position['side'],
                    'pnl_pct': pnl_pct,
                    'size': active_position['size'],
                    'hold_time': time_in_pos
                })
                active_position = None

    return trades


# Run all strategies
print("="*80)
print("TESTING ALL 5 STRATEGIES")
print("="*80 + "\n")

all_results = []

strategies = [
    ('1. LONG-only (Baseline)', lambda df: strategy_long_only(df)),
    ('2. Signal Fusion', lambda df: strategy_signal_fusion(df)),
    ('3. Asymmetric Time', lambda df: strategy_asymmetric_time(df)),
    ('4. Opportunity Gating', lambda df: strategy_opportunity_gating(df)),
    ('5. Hybrid Sizing', lambda df: strategy_hybrid_sizing(df)),
]

for strategy_name, strategy_func in strategies:
    print(f"\nTesting: {strategy_name}")
    print("-" * 80)

    start_time = time.time()
    results_df = run_window_backtest(df, strategy_func, strategy_name)
    test_time = time.time() - start_time

    if len(results_df) > 0:
        avg_return = results_df['total_return'].mean()
        avg_trades = results_df['total_trades'].mean()
        avg_long = results_df['long_trades'].mean()
        avg_short = results_df['short_trades'].mean()
        avg_wr = results_df['win_rate'].mean()

        print(f"  Return: {avg_return:.2f}% per window")
        print(f"  Trades: {avg_trades:.1f} (LONG {avg_long:.1f} + SHORT {avg_short:.1f})")
        print(f"  Win Rate: {avg_wr:.1f}%")
        print(f"  Time: {test_time:.1f}s")

        all_results.append({
            'Strategy': strategy_name,
            'Avg Return (%)': avg_return,
            'Avg Trades': avg_trades,
            'Avg LONG': avg_long,
            'Avg SHORT': avg_short,
            'Win Rate (%)': avg_wr
        })

# Final comparison
print("\n" + "="*80)
print("FINAL COMPARISON - ALL 5 STRATEGIES")
print("="*80 + "\n")

comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values('Avg Return (%)', ascending=False)

print(comparison_df.to_string(index=False))

# Winner
winner = comparison_df.iloc[0]
baseline = comparison_df[comparison_df['Strategy'].str.contains('LONG-only')].iloc[0]

print(f"\n{'='*80}")
print("🏆 WINNER")
print(f"{'='*80}\n")
print(f"{winner['Strategy']}")
print(f"  Return: {winner['Avg Return (%)']:.2f}% per window")
print(f"  Trades: {winner['Avg Trades']:.1f}")
print(f"  Win Rate: {winner['Win Rate (%)']:.1f}%")

gap = winner['Avg Return (%)'] - baseline['Avg Return (%)']
gap_pct = (gap / baseline['Avg Return (%)']) * 100 if baseline['Avg Return (%)'] != 0 else 0

print(f"\n📊 vs Baseline:")
print(f"  Baseline: {baseline['Avg Return (%)']:.2f}%")
print(f"  Winner: {winner['Avg Return (%)']:.2f}%")
print(f"  Gap: {gap:+.2f}% ({gap_pct:+.1f}%)")

if gap > 0:
    print(f"  ✅ WINNER BEATS BASELINE!")
else:
    print(f"  ❌ Baseline still best")

# Save
output_file = RESULTS_DIR / "unified_comparison_all_strategies.csv"
comparison_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print(f"\n{'='*80}")
print("ALL TESTS COMPLETE")
print(f"{'='*80}\n")
