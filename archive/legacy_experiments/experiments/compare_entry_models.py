"""
Compare Baseline vs Improved Entry Models
==========================================

Comprehensive comparison of backtest results between:
- BASELINE: Original Entry models (peak/trough labeling)
- IMPROVED: New Entry models (2-of-3 labeling system)
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Load backtest results
print("=" * 80)
print("BASELINE vs IMPROVED ENTRY MODELS COMPARISON")
print("=" * 80)

baseline_file = RESULTS_DIR / "full_backtest_opportunity_gating_4x_20251018_042912.csv"
improved_file = RESULTS_DIR / "full_backtest_opportunity_gating_4x_20251018_053404.csv"

print(f"\nLoading results...")
print(f"  Baseline: {baseline_file.name}")
print(f"  Improved: {improved_file.name}")

baseline_df = pd.read_csv(baseline_file)
improved_df = pd.read_csv(improved_file)

print(f"\n  ✅ Baseline: {len(baseline_df)} windows")
print(f"  ✅ Improved: {len(improved_df)} windows")

# Calculate averages
print(f"\n{'-' * 80}")
print("PERFORMANCE COMPARISON")
print(f"{'-' * 80}\n")

metrics = {
    'Total Trades': ('total_trades', ':.1f'),
    'LONG Trades': ('long_trades', ':.1f'),
    'SHORT Trades': ('short_trades', ':.1f'),
    'Win Rate (%)': ('win_rate', ':.1f'),
    'Avg Window Return (%)': ('total_return_pct', ':.2f'),
    'Avg Position Size (%)': ('avg_position_size', ':.1f')
}

results = []

for metric_name, (col, fmt) in metrics.items():
    baseline_val = baseline_df[col].mean()
    improved_val = improved_df[col].mean()
    change = improved_val - baseline_val
    change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0

    # Format values
    if fmt == ':.1f':
        baseline_str = f"{baseline_val:.1f}"
        improved_str = f"{improved_val:.1f}"
        change_str = f"{change:+.1f}"
    else:  # :.2f
        baseline_str = f"{baseline_val:.2f}"
        improved_str = f"{improved_val:.2f}"
        change_str = f"{change:+.2f}"

    # Change indicator
    if abs(change_pct) < 1:
        indicator = "→"  # Negligible
    elif change_pct > 0:
        indicator = "↑"  # Better (for most metrics)
    else:
        indicator = "↓"  # Worse

    # For win rate and return, higher is better
    # For trades, lower might be better (less overtrading)
    if metric_name in ['Total Trades', 'LONG Trades', 'SHORT Trades']:
        if change_pct > 0:
            indicator = "⚠️"  # More trades (potential overtrading)
        else:
            indicator = "✅"  # Fewer trades (good)

    print(f"{metric_name:30s}: {baseline_str:>10s} → {improved_str:>10s}  {indicator} ({change_str:>8s} / {change_pct:+6.1f}%)")

    results.append({
        'Metric': metric_name,
        'Baseline': baseline_val,
        'Improved': improved_val,
        'Change': change,
        'Change_Pct': change_pct
    })

# Problematic windows analysis
print(f"\n{'-' * 80}")
print("PROBLEMATIC WINDOWS ANALYSIS")
print(f"{'-' * 80}\n")

# Win rate < 40%
baseline_low_wr = (baseline_df['win_rate'] < 40).sum()
improved_low_wr = (improved_df['win_rate'] < 40).sum()

print(f"Windows with Win Rate < 40%:")
print(f"  Baseline: {baseline_low_wr:2d} windows")
print(f"  Improved: {improved_low_wr:2d} windows")
print(f"  Change:   {improved_low_wr - baseline_low_wr:+3d} ({'better' if improved_low_wr < baseline_low_wr else 'WORSE'})")

# Overtrading (>50 trades)
baseline_overtrade = (baseline_df['total_trades'] > 50).sum()
improved_overtrade = (improved_df['total_trades'] > 50).sum()

print(f"\nWindows with >50 Trades (Overtrading):")
print(f"  Baseline: {baseline_overtrade:2d} windows")
print(f"  Improved: {improved_overtrade:2d} windows")
print(f"  Change:   {improved_overtrade - baseline_overtrade:+3d} ({'better' if improved_overtrade < baseline_overtrade else 'WORSE'})")

# High trade windows
baseline_high_trades = baseline_df['total_trades'].max()
improved_high_trades = improved_df['total_trades'].max()

print(f"\nMaximum Trades per Window:")
print(f"  Baseline: {baseline_high_trades:.0f} trades")
print(f"  Improved: {improved_high_trades:.0f} trades")
print(f"  Change:   {improved_high_trades - baseline_high_trades:+.0f} ({'better' if improved_high_trades < baseline_high_trades else 'WORSE'})")

# Capital distribution
print(f"\n{'-' * 80}")
print("CAPITAL DISTRIBUTION (Final Capital per Window)")
print(f"{'-' * 80}\n")

baseline_final = baseline_df['final_capital']
improved_final = improved_df['final_capital']

print(f"Baseline:")
print(f"  Mean:   ${baseline_final.mean():,.2f}")
print(f"  Median: ${baseline_final.median():,.2f}")
print(f"  Std:    ${baseline_final.std():,.2f}")
print(f"  Min:    ${baseline_final.min():,.2f}")
print(f"  Max:    ${baseline_final.max():,.2f}")

print(f"\nImproved:")
print(f"  Mean:   ${improved_final.mean():,.2f}")
print(f"  Median: ${improved_final.median():,.2f}")
print(f"  Std:    ${improved_final.std():,.2f}")
print(f"  Min:    ${improved_final.min():,.2f}")
print(f"  Max:    ${improved_final.max():,.2f}")

# Window-by-window comparison
print(f"\n{'-' * 80}")
print("WINDOW-BY-WINDOW ANALYSIS (First 20 windows)")
print(f"{'-' * 80}\n")

print(f"{'Window':<10s} {'Base WR':<10s} {'Imp WR':<10s} {'Base Ret':<12s} {'Imp Ret':<12s} {'Base #':<8s} {'Imp #':<8s}")
print("-" * 80)

for i in range(min(20, len(baseline_df))):
    base_wr = baseline_df['win_rate'].iloc[i]
    imp_wr = improved_df['win_rate'].iloc[i]
    base_ret = baseline_df['total_return_pct'].iloc[i]
    imp_ret = improved_df['total_return_pct'].iloc[i]
    base_trades = baseline_df['total_trades'].iloc[i]
    imp_trades = improved_df['total_trades'].iloc[i]

    wr_indicator = "✅" if imp_wr >= base_wr else "❌"
    ret_indicator = "✅" if imp_ret >= base_ret else "❌"

    print(f"Window {i:<3d} {base_wr:>6.1f}%    {imp_wr:>6.1f}% {wr_indicator}  {base_ret:>+7.2f}%     {imp_ret:>+7.2f}% {ret_indicator}  {base_trades:>4.0f}    {imp_trades:>4.0f}")

# Overall assessment
print(f"\n{'-' * 80}")
print("OVERALL ASSESSMENT")
print(f"{'-' * 80}\n")

# Count improvements
wr_improvements = (improved_df['win_rate'] > baseline_df['win_rate']).sum()
ret_improvements = (improved_df['total_return_pct'] > baseline_df['total_return_pct']).sum()
trade_reductions = (improved_df['total_trades'] < baseline_df['total_trades']).sum()

total_windows = len(baseline_df)

print(f"Windows where Improved Model is Better:")
print(f"  Win Rate:     {wr_improvements}/{total_windows} ({wr_improvements/total_windows*100:.1f}%)")
print(f"  Returns:      {ret_improvements}/{total_windows} ({ret_improvements/total_windows*100:.1f}%)")
print(f"  Fewer Trades: {trade_reductions}/{total_windows} ({trade_reductions/total_windows*100:.1f}%)")

# Final verdict
avg_wr_change = improved_df['win_rate'].mean() - baseline_df['win_rate'].mean()
avg_ret_change = improved_df['total_return_pct'].mean() - baseline_df['total_return_pct'].mean()
avg_trade_change = improved_df['total_trades'].mean() - baseline_df['total_trades'].mean()

print(f"\n{'=' * 80}")
print("VERDICT")
print(f"{'=' * 80}\n")

if avg_wr_change > 2 and avg_ret_change > 2:
    verdict = "✅ IMPROVED MODELS ARE CLEARLY BETTER"
elif avg_wr_change > 0 and avg_ret_change > 0:
    verdict = "✅ IMPROVED MODELS SHOW MODEST IMPROVEMENT"
elif avg_wr_change < -2 or avg_ret_change < -2:
    verdict = "❌ IMPROVED MODELS UNDERPERFORM BASELINE"
else:
    verdict = "⚠️ MIXED RESULTS - FURTHER ANALYSIS NEEDED"

print(verdict)

print(f"\nKey Observations:")
print(f"  Win Rate Change:  {avg_wr_change:+.2f}% ({'better' if avg_wr_change > 0 else 'WORSE'})")
print(f"  Return Change:    {avg_ret_change:+.2f}% ({'better' if avg_ret_change > 0 else 'WORSE'})")
print(f"  Trade Change:     {avg_trade_change:+.1f} ({'more overtrading' if avg_trade_change > 0 else 'less trades'})")

print(f"\n{'=' * 80}")
print("COMPARISON COMPLETE")
print(f"{'=' * 80}\n")
