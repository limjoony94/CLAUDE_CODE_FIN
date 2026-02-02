"""
Critical Analysis of Ultra-5 Results

비판적 질문들:
1. +1.26% vs B&H가 정말 의미있는가?
2. Bull 시장에서 -5.09% 실패는 치명적이지 않은가?
3. 0 거래 windows 36%는 너무 보수적이지 않은가?
4. 통계적 유의성이 정말 있는가?
5. Overfitting 위험은 없는가?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

print("=" * 80)
print("CRITICAL ANALYSIS: Ultra-5 Results")
print("=" * 80)

# Load Ultra-5 results
ultra5_file = RESULTS_DIR / "backtest_hybrid_v4_ultraconservative.csv"
df = pd.read_csv(ultra5_file)
df_ultra5 = df[df['config_name'] == 'Ultra-5 (Extreme)'].copy()

print(f"\n📊 Raw Data ({len(df_ultra5)} windows):\n")
print(df_ultra5[['regime', 'hybrid_return', 'bh_return', 'difference', 'num_trades', 'win_rate']].to_string(index=False))

# Critical Analysis 1: Distribution of results
print(f"\n{'=' * 80}")
print("CRITICAL ANALYSIS 1: Result Distribution")
print(f"{'=' * 80}\n")

positive_windows = df_ultra5[df_ultra5['difference'] > 0]
negative_windows = df_ultra5[df_ultra5['difference'] <= 0]

print(f"Positive windows (beat B&H): {len(positive_windows)} ({len(positive_windows)/len(df_ultra5)*100:.1f}%)")
print(f"  Average advantage: {positive_windows['difference'].mean():.2f}%")
print(f"  Max advantage: {positive_windows['difference'].max():.2f}%")

print(f"\nNegative windows (lost to B&H): {len(negative_windows)} ({len(negative_windows)/len(df_ultra5)*100:.1f}%)")
print(f"  Average disadvantage: {negative_windows['difference'].mean():.2f}%")
print(f"  Max disadvantage: {negative_windows['difference'].min():.2f}%")

print(f"\n⚠️ Critical Insight:")
print(f"  Win rate (windows): {len(positive_windows)}/{len(df_ultra5)} = {len(positive_windows)/len(df_ultra5)*100:.1f}%")
if len(positive_windows)/len(df_ultra5) < 0.7:
    print(f"  ❌ Less than 70% success rate - NOT RELIABLE!")

# Critical Analysis 2: Zero-trade windows
print(f"\n{'=' * 80}")
print("CRITICAL ANALYSIS 2: Zero-Trade Windows")
print(f"{'=' * 80}\n")

zero_trade_windows = df_ultra5[df_ultra5['num_trades'] == 0]
few_trade_windows = df_ultra5[df_ultra5['num_trades'] <= 1]

print(f"Zero-trade windows: {len(zero_trade_windows)} ({len(zero_trade_windows)/len(df_ultra5)*100:.1f}%)")
print(f"≤1 trade windows: {len(few_trade_windows)} ({len(few_trade_windows)/len(df_ultra5)*100:.1f}%)\n")

if len(zero_trade_windows) > 0:
    print("Zero-trade windows detail:")
    print(zero_trade_windows[['regime', 'bh_return', 'difference']].to_string(index=False))

    print(f"\n⚠️ Critical Insight:")
    print(f"  In zero-trade windows, we miss:")
    zero_positive_bh = zero_trade_windows[zero_trade_windows['bh_return'] > 0]
    if len(zero_positive_bh) > 0:
        print(f"    - {len(zero_positive_bh)} positive B&H periods (avg: {zero_positive_bh['bh_return'].mean():.2f}%)")
        print(f"    - Total opportunity cost: {zero_positive_bh['bh_return'].sum():.2f}%")
        print(f"  ❌ TOO CONSERVATIVE - Missing bull market opportunities!")

# Critical Analysis 3: Bull market performance
print(f"\n{'=' * 80}")
print("CRITICAL ANALYSIS 3: Bull Market Performance")
print(f"{'=' * 80}\n")

bull_windows = df_ultra5[df_ultra5['regime'] == 'Bull']
bear_windows = df_ultra5[df_ultra5['regime'] == 'Bear']
sideways_windows = df_ultra5[df_ultra5['regime'] == 'Sideways']

print("Bull Market:")
if len(bull_windows) > 0:
    print(f"  Windows: {len(bull_windows)}")
    print(f"  Hybrid avg: {bull_windows['hybrid_return'].mean():.2f}%")
    print(f"  B&H avg: {bull_windows['bh_return'].mean():.2f}%")
    print(f"  Difference: {bull_windows['difference'].mean():.2f}%")
    print(f"  Avg trades: {bull_windows['num_trades'].mean():.1f}")

    if bull_windows['difference'].mean() < -2.0:
        print(f"  🚨 CRITICAL FAILURE: Losing {abs(bull_windows['difference'].mean()):.2f}% to B&H in bull markets!")
        print(f"  This is NOT acceptable for a trading system.")

print("\nBear Market:")
if len(bear_windows) > 0:
    print(f"  Windows: {len(bear_windows)}")
    print(f"  Hybrid avg: {bear_windows['hybrid_return'].mean():.2f}%")
    print(f"  B&H avg: {bear_windows['bh_return'].mean():.2f}%")
    print(f"  Difference: {bear_windows['difference'].mean():.2f}%")
    print(f"  Avg trades: {bear_windows['num_trades'].mean():.1f}")

print("\nSideways Market:")
if len(sideways_windows) > 0:
    print(f"  Windows: {len(sideways_windows)}")
    print(f"  Hybrid avg: {sideways_windows['hybrid_return'].mean():.2f}%")
    print(f"  B&H avg: {sideways_windows['bh_return'].mean():.2f}%")
    print(f"  Difference: {sideways_windows['difference'].mean():.2f}%")
    print(f"  Avg trades: {sideways_windows['num_trades'].mean():.1f}")

# Critical Analysis 4: Statistical significance
print(f"\n{'=' * 80}")
print("CRITICAL ANALYSIS 4: Statistical Significance")
print(f"{'=' * 80}\n")

# Paired t-test
t_stat, p_value = stats.ttest_rel(df_ultra5['hybrid_return'], df_ultra5['bh_return'])

print(f"Paired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant (p < 0.05): {'✅ Yes' if p_value < 0.05 else '❌ No'}")

if p_value >= 0.05:
    print(f"\n⚠️ Critical Insight:")
    print(f"  Results are NOT statistically significant!")
    print(f"  Cannot confidently claim the strategy beats B&H")

# Effect size (Cohen's d)
mean_diff = df_ultra5['difference'].mean()
std_diff = df_ultra5['difference'].std()
cohens_d = mean_diff / std_diff if std_diff > 0 else 0

print(f"\nEffect Size (Cohen's d): {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    print(f"  ❌ Very small effect size - differences are negligible")
elif abs(cohens_d) < 0.5:
    print(f"  ⚠️ Small effect size")
elif abs(cohens_d) < 0.8:
    print(f"  ✅ Medium effect size")
else:
    print(f"  ✅ Large effect size")

# Critical Analysis 5: Variance and Consistency
print(f"\n{'=' * 80}")
print("CRITICAL ANALYSIS 5: Variance and Consistency")
print(f"{'=' * 80}\n")

print(f"Difference (vs B&H) statistics:")
print(f"  Mean: {df_ultra5['difference'].mean():.2f}%")
print(f"  Std Dev: {df_ultra5['difference'].std():.2f}%")
print(f"  Min: {df_ultra5['difference'].min():.2f}%")
print(f"  Max: {df_ultra5['difference'].max():.2f}%")
print(f"  Range: {df_ultra5['difference'].max() - df_ultra5['difference'].min():.2f}%")

cv = (df_ultra5['difference'].std() / abs(df_ultra5['difference'].mean())) if df_ultra5['difference'].mean() != 0 else float('inf')
print(f"\nCoefficient of Variation: {cv:.2f}")
if cv > 2.0:
    print(f"  ❌ Very high variance - results are INCONSISTENT")
    print(f"  Cannot rely on this strategy for consistent performance")

# Critical Analysis 6: Trade frequency analysis
print(f"\n{'=' * 80}")
print("CRITICAL ANALYSIS 6: Trade Frequency")
print(f"{'=' * 80}\n")

print(f"Trade frequency distribution:")
print(f"  0 trades: {len(df_ultra5[df_ultra5['num_trades'] == 0])} windows")
print(f"  1-2 trades: {len(df_ultra5[(df_ultra5['num_trades'] >= 1) & (df_ultra5['num_trades'] <= 2)])} windows")
print(f"  3-5 trades: {len(df_ultra5[(df_ultra5['num_trades'] >= 3) & (df_ultra5['num_trades'] <= 5)])} windows")
print(f"  >5 trades: {len(df_ultra5[df_ultra5['num_trades'] > 5])} windows")

avg_trades_per_day = df_ultra5['num_trades'].mean() / 5  # 5 days per window
print(f"\nAverage trades per day: {avg_trades_per_day:.2f}")

if avg_trades_per_day < 0.5:
    print(f"  ⚠️ Less than 0.5 trades/day - very conservative")
    print(f"  May miss many profitable opportunities")

# Final Verdict
print(f"\n{'=' * 80}")
print("FINAL CRITICAL VERDICT")
print(f"{'=' * 80}\n")

issues = []
warnings = []
successes = []

# Check statistical significance
if p_value >= 0.05:
    issues.append("NOT statistically significant (p >= 0.05)")
else:
    successes.append("Statistically significant (p < 0.05)")

# Check bull market performance
if len(bull_windows) > 0 and bull_windows['difference'].mean() < -2.0:
    issues.append(f"CRITICAL FAILURE in bull markets ({bull_windows['difference'].mean():.2f}% vs B&H)")

# Check consistency
if cv > 2.0:
    issues.append(f"Very high variance (CV = {cv:.2f}) - inconsistent results")

# Check success rate
win_rate = len(positive_windows) / len(df_ultra5)
if win_rate < 0.7:
    warnings.append(f"Success rate only {win_rate*100:.1f}% (< 70%)")
else:
    successes.append(f"Success rate {win_rate*100:.1f}%")

# Check zero-trade rate
zero_rate = len(zero_trade_windows) / len(df_ultra5)
if zero_rate > 0.3:
    warnings.append(f"Too conservative: {zero_rate*100:.1f}% windows with zero trades")

print("✅ SUCCESSES:")
for s in successes:
    print(f"  - {s}")

if warnings:
    print("\n⚠️ WARNINGS:")
    for w in warnings:
        print(f"  - {w}")

if issues:
    print("\n🚨 CRITICAL ISSUES:")
    for i in issues:
        print(f"  - {i}")

print(f"\n{'=' * 80}")
if len(issues) > 0:
    print("❌ VERDICT: Ultra-5 is NOT RELIABLE")
    print("=" * 80)
    print("\nRecommendation:")
    print("  1. Ultra-5 is TOO CONSERVATIVE")
    print("  2. CRITICAL failure in bull markets")
    print("  3. Need to find BALANCED configuration")
    print("  4. Consider regime-specific strategies")
elif len(warnings) > 1:
    print("⚠️ VERDICT: Ultra-5 has SIGNIFICANT LIMITATIONS")
    print("=" * 80)
    print("\nRecommendation:")
    print("  1. Use with caution")
    print("  2. Consider more balanced approach")
    print("  3. May need regime-specific optimization")
else:
    print("✅ VERDICT: Ultra-5 is ACCEPTABLE")
    print("=" * 80)

print(f"\n{'=' * 80}")
print("Critical Analysis Complete")
print(f"{'=' * 80}")
