"""
Analyze 108-window backtest results for overfitting detection
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
results_file = PROJECT_ROOT / "results" / "full_backtest_OPTION_B_threshold_080_20251026_145426.csv"

print("=" * 80)
print("108-WINDOW BACKTEST CONSISTENCY ANALYSIS")
print("=" * 80)
print(f"File: {results_file.name}")
print("=" * 80)

# Load data
df = pd.read_csv(results_file)
print(f"\n✅ Loaded {len(df)} windows\n")

# Basic statistics
print("=" * 80)
print("1. RETURN DISTRIBUTION (5-day per window)")
print("=" * 80)
returns = df['total_return_pct']
print(f"Mean Return: {returns.mean():.2f}%")
print(f"Median Return: {returns.median():.2f}%")
print(f"Std Dev: {returns.std():.2f}%")
print(f"Min Return: {returns.min():.2f}% (Window {returns.idxmin()})")
print(f"Max Return: {returns.max():.2f}% (Window {returns.idxmax()})")
print(f"\nPercentiles:")
print(f"  10th: {returns.quantile(0.10):.2f}%")
print(f"  25th: {returns.quantile(0.25):.2f}%")
print(f"  50th: {returns.quantile(0.50):.2f}%")
print(f"  75th: {returns.quantile(0.75):.2f}%")
print(f"  90th: {returns.quantile(0.90):.2f}%")

# Win rate distribution
print("\n" + "=" * 80)
print("2. WIN RATE DISTRIBUTION")
print("=" * 80)
win_rates = df['win_rate']
print(f"Mean Win Rate: {win_rates.mean():.2f}%")
print(f"Median Win Rate: {win_rates.median():.2f}%")
print(f"Std Dev: {win_rates.std():.2f}%")
print(f"Min Win Rate: {win_rates.min():.2f}% (Window {win_rates.idxmin()})")
print(f"Max Win Rate: {win_rates.max():.2f}% (Window {win_rates.idxmax()})")

# Profitable windows
print("\n" + "=" * 80)
print("3. PROFITABILITY CONSISTENCY")
print("=" * 80)
profitable = (returns > 0).sum()
unprofitable = (returns <= 0).sum()
print(f"Profitable Windows: {profitable} ({profitable/len(df)*100:.1f}%)")
print(f"Unprofitable Windows: {unprofitable} ({unprofitable/len(df)*100:.1f}%)")
print(f"\nReturn Ranges:")
print(f"  < 0%: {(returns < 0).sum()} windows ({(returns < 0).sum()/len(df)*100:.1f}%)")
print(f"  0-5%: {((returns >= 0) & (returns < 5)).sum()} windows")
print(f"  5-10%: {((returns >= 5) & (returns < 10)).sum()} windows")
print(f"  10-20%: {((returns >= 10) & (returns < 20)).sum()} windows")
print(f"  20-50%: {((returns >= 20) & (returns < 50)).sum()} windows")
print(f"  50-100%: {((returns >= 50) & (returns < 100)).sum()} windows")
print(f"  > 100%: {(returns >= 100).sum()} windows ⚠️")

# Outlier analysis
print("\n" + "=" * 80)
print("4. OUTLIER ANALYSIS")
print("=" * 80)
Q1 = returns.quantile(0.25)
Q3 = returns.quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_high = Q3 + 1.5 * IQR
outlier_threshold_low = Q1 - 1.5 * IQR

outliers_high = df[returns > outlier_threshold_high]
outliers_low = df[returns < outlier_threshold_low]

print(f"IQR: {IQR:.2f}%")
print(f"Outlier thresholds: < {outlier_threshold_low:.2f}% or > {outlier_threshold_high:.2f}%")
print(f"\nHigh outliers: {len(outliers_high)} windows")
if len(outliers_high) > 0:
    print("  Windows:", outliers_high.index.tolist()[:10])
    print(f"  Returns: {outliers_high['total_return_pct'].tolist()[:10]}")
    print(f"  Total contribution: {outliers_high['total_return_pct'].sum():.2f}%")
    print(f"  Contribution to total return: {outliers_high['total_return_pct'].sum() / returns.sum() * 100:.1f}%")

print(f"\nLow outliers: {len(outliers_low)} windows")
if len(outliers_low) > 0:
    print("  Windows:", outliers_low.index.tolist())
    print(f"  Returns: {outliers_low['total_return_pct'].tolist()}")

# Temporal consistency
print("\n" + "=" * 80)
print("5. TEMPORAL CONSISTENCY (Overfitting Check)")
print("=" * 80)

# Split into 4 quarters
quarter_size = len(df) // 4
quarters = []
for i in range(4):
    start_idx = i * quarter_size
    end_idx = (i + 1) * quarter_size if i < 3 else len(df)
    quarter_df = df.iloc[start_idx:end_idx]
    quarters.append({
        'quarter': i + 1,
        'windows': f"{start_idx}-{end_idx-1}",
        'mean_return': quarter_df['total_return_pct'].mean(),
        'median_return': quarter_df['total_return_pct'].median(),
        'win_rate': quarter_df['win_rate'].mean(),
        'profitable_pct': (quarter_df['total_return_pct'] > 0).sum() / len(quarter_df) * 100
    })

print("\nQuarterly Performance:")
print("-" * 80)
print(f"{'Quarter':<10} {'Windows':<15} {'Mean Return':<15} {'Win Rate':<15} {'Profitable %':<15}")
print("-" * 80)
for q in quarters:
    print(f"Q{q['quarter']:<9} {q['windows']:<15} {q['mean_return']:>12.2f}% {q['win_rate']:>12.2f}% {q['profitable_pct']:>13.1f}%")

# Check for degradation over time
early_half = df.iloc[:54]['total_return_pct'].mean()
late_half = df.iloc[54:]['total_return_pct'].mean()
degradation = (late_half - early_half) / early_half * 100

print(f"\nFirst half (0-53): Mean {early_half:.2f}%")
print(f"Second half (54-107): Mean {late_half:.2f}%")
print(f"Degradation: {degradation:+.1f}% {'⚠️ CONCERN' if degradation < -20 else '✅ OK'}")

# Trade frequency analysis
print("\n" + "=" * 80)
print("6. TRADE FREQUENCY ANALYSIS")
print("=" * 80)
trades = df['total_trades']
print(f"Mean Trades per Window: {trades.mean():.1f}")
print(f"Median Trades per Window: {trades.median():.1f}")
print(f"Std Dev: {trades.std():.1f}")
print(f"Min Trades: {trades.min()} (Window {trades.idxmin()})")
print(f"Max Trades: {trades.max()} (Window {trades.idxmax()})")

# Long/Short distribution
print("\n" + "=" * 80)
print("7. LONG/SHORT DISTRIBUTION")
print("=" * 80)
total_long = df['long_trades'].sum()
total_short = df['short_trades'].sum()
total_trades_all = total_long + total_short
print(f"Total LONG trades: {total_long} ({total_long/total_trades_all*100:.1f}%)")
print(f"Total SHORT trades: {total_short} ({total_short/total_trades_all*100:.1f}%)")
print(f"Total trades: {total_trades_all}")

# Overfitting verdict
print("\n" + "=" * 80)
print("8. OVERFITTING ASSESSMENT")
print("=" * 80)

overfitting_signals = []

# Check 1: Extreme outliers dominating returns
outlier_contribution = outliers_high['total_return_pct'].sum() / returns.sum() * 100
if outlier_contribution > 50:
    overfitting_signals.append(f"⚠️ Outliers contribute {outlier_contribution:.1f}% of total returns (>50%)")

# Check 2: Temporal degradation
if degradation < -20:
    overfitting_signals.append(f"⚠️ Late-half performance degraded by {degradation:.1f}% (>20%)")

# Check 3: High variance
cv = returns.std() / abs(returns.mean())  # Coefficient of variation
if cv > 2.0:
    overfitting_signals.append(f"⚠️ High variance (CV={cv:.2f} > 2.0)")

# Check 4: Low consistency
if profitable/len(df) < 0.70:
    overfitting_signals.append(f"⚠️ Low profitability consistency ({profitable/len(df)*100:.1f}% < 70%)")

if len(overfitting_signals) == 0:
    print("✅ NO OVERFITTING DETECTED")
    print("\nReasons:")
    print(f"  • Outlier contribution: {outlier_contribution:.1f}% (< 50%) ✅")
    print(f"  • Temporal stability: {degradation:+.1f}% degradation (> -20%) ✅")
    print(f"  • Reasonable variance: CV={cv:.2f} (< 2.0) ✅")
    print(f"  • High consistency: {profitable/len(df)*100:.1f}% profitable (> 70%) ✅")
else:
    print("⚠️ POTENTIAL OVERFITTING DETECTED")
    print("\nSignals:")
    for signal in overfitting_signals:
        print(f"  {signal}")

# Summary statistics
print("\n" + "=" * 80)
print("9. SUMMARY STATISTICS")
print("=" * 80)
print(f"Total Windows: {len(df)}")
print(f"Average Return per Window: {returns.mean():.2f}%")
print(f"Cumulative Return: {returns.sum():.2f}%")
print(f"Average Win Rate: {win_rates.mean():.2f}%")
print(f"Average Trades per Window: {trades.mean():.1f}")
print(f"Consistency (Profitable Windows): {profitable/len(df)*100:.1f}%")

print("\n" + "=" * 80)
