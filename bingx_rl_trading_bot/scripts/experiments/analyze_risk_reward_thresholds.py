"""
Analyze Risk-Reward Threshold Options
======================================

Test different MAE/MFE threshold combinations to find optimal balance.

Current (too strict):
  MAE < -2%, MFE > 4% → 0.1% pass rate

Goal: Find thresholds with 10-30% pass rate
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.experiments.trade_simulator import TradeSimulator, load_exit_models

DATA_DIR = PROJECT_ROOT / "data" / "historical"

print("="*80)
print("RISK-REWARD THRESHOLD ANALYSIS")
print("="*80)

# ============================================================================
# Load Sample Data
# ============================================================================

print("\n" + "-"*80)
print("Loading Sample Data (5,000 candles)")
print("-"*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)

# Use last 5,000 candles
SAMPLE_SIZE = 5000
df = df_full.tail(SAMPLE_SIZE).reset_index(drop=True).copy()

print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"✅ Features calculated")

# ============================================================================
# Load Exit Models
# ============================================================================

print("\n" + "-"*80)
print("Loading Exit Models")
print("-"*80)

exit_models = load_exit_models()

long_simulator = TradeSimulator(
    exit_model=exit_models['long'][0],
    exit_scaler=exit_models['long'][1],
    exit_features=exit_models['long'][2],
    leverage=4,
    ml_exit_threshold=0.70,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

short_simulator = TradeSimulator(
    exit_model=exit_models['short'][0],
    exit_scaler=exit_models['short'][1],
    exit_features=exit_models['short'][2],
    leverage=4,
    ml_exit_threshold=0.70,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

print("✅ Simulators ready")

# ============================================================================
# Simulate Trades
# ============================================================================

print("\n" + "-"*80)
print("Simulating Trades")
print("-"*80)

def simulate_all_trades(df, simulator, side):
    """Simulate trades from all candles"""
    results = []

    for i in range(len(df) - 96):  # Leave room for max hold
        if i % 1000 == 0:
            print(f"  {side}: {i}/{len(df)-96} candles simulated...")

        result = simulator.simulate_trade(df, i, side)
        results.append(result)

    return results

print("Simulating LONG trades...")
long_results = simulate_all_trades(df, long_simulator, 'LONG')

print("\nSimulating SHORT trades...")
short_results = simulate_all_trades(df, short_simulator, 'SHORT')

print(f"\n✅ Simulations complete:")
print(f"   LONG:  {len(long_results)} trades")
print(f"   SHORT: {len(short_results)} trades")

# Convert to DataFrames
df_long = pd.DataFrame(long_results)
df_short = pd.DataFrame(short_results)

# ============================================================================
# Analyze Threshold Options
# ============================================================================

print("\n" + "="*80)
print("THRESHOLD ANALYSIS")
print("="*80)

# Define threshold options
threshold_options = [
    # (MAE, MFE, Description)
    (-0.02, 0.04, "Current (Strict)"),
    (-0.03, 0.03, "Conservative"),
    (-0.03, 0.025, "Moderate 1"),
    (-0.04, 0.025, "Moderate 2"),
    (-0.04, 0.02, "Relaxed"),
    (-0.05, 0.02, "Very Relaxed"),
]

def analyze_threshold(df_results, mae_thresh, mfe_thresh, side):
    """Analyze single threshold combination"""
    # Apply Risk-Reward criterion
    rr_pass = (df_results['mae'] >= mae_thresh) & (df_results['mfe'] >= mfe_thresh)

    # Other criteria
    profitable = df_results['leveraged_pnl_pct'] >= 0.02
    ml_exit = df_results['exit_reason'] == 'ml_exit'

    # Score distribution
    score = profitable.astype(int) + rr_pass.astype(int) + ml_exit.astype(int)

    # Positive labels (score >= 2)
    positive = score >= 2

    return {
        'side': side,
        'mae_thresh': mae_thresh,
        'mfe_thresh': mfe_thresh,
        'rr_pass_rate': rr_pass.mean() * 100,
        'profitable_rate': profitable.mean() * 100,
        'ml_exit_rate': ml_exit.mean() * 100,
        'positive_label_rate': positive.mean() * 100,
        'score_0': (score == 0).sum(),
        'score_1': (score == 1).sum(),
        'score_2': (score == 2).sum(),
        'score_3': (score == 3).sum(),
    }

print(f"\n{'Description':<20} {'MAE':<8} {'MFE':<8} {'RR Pass':<10} {'Positive':<10} {'LONG/SHORT'}")
print("-"*80)

results = []

for mae, mfe, desc in threshold_options:
    long_analysis = analyze_threshold(df_long, mae, mfe, 'LONG')
    short_analysis = analyze_threshold(df_short, mae, mfe, 'SHORT')

    print(f"{desc:<20} {mae*100:>6.1f}% {mfe*100:>6.1f}% "
          f"{long_analysis['rr_pass_rate']:>8.1f}% "
          f"{long_analysis['positive_label_rate']:>8.1f}% "
          f"LONG")
    print(f"{'':20} {'':8} {'':8} "
          f"{short_analysis['rr_pass_rate']:>8.1f}% "
          f"{short_analysis['positive_label_rate']:>8.1f}% "
          f"SHORT")
    print()

    results.append({
        'desc': desc,
        'mae': mae,
        'mfe': mfe,
        'long': long_analysis,
        'short': short_analysis
    })

# ============================================================================
# Detailed Analysis for Top Candidates
# ============================================================================

print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

# Recommend options with 10-30% positive label rate
recommendations = []

for r in results:
    long_pos = r['long']['positive_label_rate']
    short_pos = r['short']['positive_label_rate']

    if 10 <= long_pos <= 35 and 10 <= short_pos <= 35:
        recommendations.append(r)

if recommendations:
    print(f"\n✅ Found {len(recommendations)} viable options (10-35% positive label rate):\n")

    for r in recommendations:
        print(f"{'='*80}")
        print(f"{r['desc']} - MAE: {r['mae']*100:.1f}%, MFE: {r['mfe']*100:.1f}%")
        print(f"{'='*80}")

        print(f"\nLONG:")
        print(f"  Criterion Pass Rates:")
        print(f"    Profitable (>=2%): {r['long']['profitable_rate']:.1f}%")
        print(f"    Risk-Reward:       {r['long']['rr_pass_rate']:.1f}%")
        print(f"    ML Exit:           {r['long']['ml_exit_rate']:.1f}%")
        print(f"  Score Distribution:")
        print(f"    0/3: {r['long']['score_0']:>5} ({r['long']['score_0']/len(df_long)*100:.1f}%)")
        print(f"    1/3: {r['long']['score_1']:>5} ({r['long']['score_1']/len(df_long)*100:.1f}%)")
        print(f"    2/3: {r['long']['score_2']:>5} ({r['long']['score_2']/len(df_long)*100:.1f}%)")
        print(f"    3/3: {r['long']['score_3']:>5} ({r['long']['score_3']/len(df_long)*100:.1f}%)")
        print(f"  Positive Label Rate: {r['long']['positive_label_rate']:.1f}%")

        print(f"\nSHORT:")
        print(f"  Criterion Pass Rates:")
        print(f"    Profitable (>=2%): {r['short']['profitable_rate']:.1f}%")
        print(f"    Risk-Reward:       {r['short']['rr_pass_rate']:.1f}%")
        print(f"    ML Exit:           {r['short']['ml_exit_rate']:.1f}%")
        print(f"  Score Distribution:")
        print(f"    0/3: {r['short']['score_0']:>5} ({r['short']['score_0']/len(df_short)*100:.1f}%)")
        print(f"    1/3: {r['short']['score_1']:>5} ({r['short']['score_1']/len(df_short)*100:.1f}%)")
        print(f"    2/3: {r['short']['score_2']:>5} ({r['short']['score_2']/len(df_short)*100:.1f}%)")
        print(f"    3/3: {r['short']['score_3']:>5} ({r['short']['score_3']/len(df_short)*100:.1f}%)")
        print(f"  Positive Label Rate: {r['short']['positive_label_rate']:.1f}%")
        print()
else:
    print("\n⚠️ No options found in 10-35% range. Consider wider criteria.")

# ============================================================================
# Recommendation
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if recommendations:
    # Choose option closest to 20% positive rate
    best = min(recommendations,
               key=lambda r: abs((r['long']['positive_label_rate'] +
                                 r['short']['positive_label_rate'])/2 - 20))

    print(f"\n✅ RECOMMENDED: {best['desc']}")
    print(f"   MAE threshold: {best['mae']*100:.1f}%")
    print(f"   MFE threshold: {best['mfe']*100:.1f}%")
    print(f"   LONG positive rate: {best['long']['positive_label_rate']:.1f}%")
    print(f"   SHORT positive rate: {best['short']['positive_label_rate']:.1f}%")
    print(f"   Average positive rate: {(best['long']['positive_label_rate'] + best['short']['positive_label_rate'])/2:.1f}%")

    print(f"\n📊 vs Current (Strict):")
    current = results[0]
    print(f"   Risk-Reward Pass Rate:")
    print(f"     LONG:  {current['long']['rr_pass_rate']:.1f}% → {best['long']['rr_pass_rate']:.1f}% "
          f"({best['long']['rr_pass_rate'] - current['long']['rr_pass_rate']:+.1f}%)")
    print(f"     SHORT: {current['short']['rr_pass_rate']:.1f}% → {best['short']['rr_pass_rate']:.1f}% "
          f"({best['short']['rr_pass_rate'] - current['short']['rr_pass_rate']:+.1f}%)")
    print(f"   Positive Label Rate:")
    print(f"     LONG:  {current['long']['positive_label_rate']:.1f}% → {best['long']['positive_label_rate']:.1f}% "
          f"({best['long']['positive_label_rate'] - current['long']['positive_label_rate']:+.1f}%)")
    print(f"     SHORT: {current['short']['positive_label_rate']:.1f}% → {best['short']['positive_label_rate']:.1f}% "
          f"({best['short']['positive_label_rate'] - current['short']['positive_label_rate']:+.1f}%)")
else:
    print("\n⚠️ No optimal threshold found. Consider:")
    print("   1. Adjust target positive label rate range")
    print("   2. Review simulation logic")
    print("   3. Try different profit threshold (currently 2%)")

print("\n" + "="*80)
print("Next Steps:")
print("  1. Review recommended thresholds")
print("  2. Retrain Entry models with relaxed criteria")
print("  3. Backtest and compare vs Baseline")
print("="*80)
