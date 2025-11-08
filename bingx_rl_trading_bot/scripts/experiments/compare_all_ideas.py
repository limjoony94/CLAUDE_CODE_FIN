"""
Final Comparison: All 4 Ideas
===============================

Compare results from all 4 innovative solutions
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("FINAL COMPARISON: 4 INNOVATIVE SOLUTIONS")
print("="*80)
print()

# Load results from all tests
results_files = {
    'Idea 1: Signal Fusion': RESULTS_DIR / "idea1_signal_fusion_results.csv",
    'Idea 2: Asymmetric Time': RESULTS_DIR / "idea2_asymmetric_time_results.csv",
    'Idea 3: Opportunity Gating': RESULTS_DIR / "idea3_opportunity_gating_results.csv",
    'Idea 4: Hybrid Sizing': RESULTS_DIR / "idea4_hybrid_sizing_results.csv",
}

all_best_results = []

for idea_name, file_path in results_files.items():
    if file_path.exists():
        df = pd.read_csv(file_path)
        # Get best configuration from each idea
        best = df.loc[df['avg_return'].idxmax()]

        all_best_results.append({
            'Idea': idea_name,
            'Config': best['config'],
            'Avg Return (%)': best['avg_return'],
            'Avg Trades': best['avg_trades'],
            'Avg LONG': best['avg_long'],
            'Avg SHORT': best['avg_short'],
            'Win Rate (%)': best['win_rate']
        })
        print(f"✅ Loaded: {idea_name}")
        print(f"   Best config: {best['config']}")
        print(f"   Return: {best['avg_return']:.2f}%\n")
    else:
        print(f"⚠️  Missing: {idea_name}")
        print(f"   File not found: {file_path}\n")

if len(all_best_results) == 0:
    print("\n❌ No results found. Tests may still be running.\n")
    exit()

# Create comparison dataframe
comparison_df = pd.DataFrame(all_best_results)
comparison_df = comparison_df.sort_values('Avg Return (%)', ascending=False)

print("\n" + "="*80)
print("FINAL RESULTS - BEST CONFIGURATION FROM EACH IDEA")
print("="*80 + "\n")

print(comparison_df.to_string(index=False))

# Winner analysis
winner = comparison_df.iloc[0]
baseline_long_only = 10.14  # LONG-only baseline

print(f"\n" + "="*80)
print("🏆 WINNER ANALYSIS")
print("="*80 + "\n")

print(f"Winner: {winner['Idea']}")
print(f"Config: {winner['Config']}")
print(f"\nPerformance:")
print(f"  Return per Window: {winner['Avg Return (%)']:.2f}%")
print(f"  Trades per Window: {winner['Avg Trades']:.1f}")
print(f"    - LONG: {winner['Avg LONG']:.1f}")
print(f"    - SHORT: {winner['Avg SHORT']:.1f}")
print(f"  Win Rate: {winner['Win Rate (%)']:.1f}%")

print(f"\nComparison to Baselines:")
print(f"  LONG-only baseline: +{baseline_long_only:.2f}%")
print(f"  Winner: +{winner['Avg Return (%)']:.2f}%")

gap = winner['Avg Return (%)'] - baseline_long_only
gap_pct = (gap / baseline_long_only) * 100

if gap >= 0:
    print(f"  ✅ WINNER BEATS LONG-ONLY by {gap:.2f}% ({gap_pct:+.1f}%)")
else:
    print(f"  ❌ Still below LONG-only by {abs(gap):.2f}% ({gap_pct:.1f}%)")

# Rank all ideas
print(f"\n" + "="*80)
print("RANKING")
print("="*80 + "\n")

for rank, row in enumerate(comparison_df.itertuples(), 1):
    gap_to_baseline = row._3 - baseline_long_only  # Avg Return (%)
    print(f"{rank}. {row.Idea}")
    print(f"   Return: {row._3:.2f}% (gap: {gap_to_baseline:+.2f}%)")
    print(f"   Config: {row.Config}")
    print()

# Save final comparison
output_file = RESULTS_DIR / "final_comparison_all_ideas.csv"
comparison_df.to_csv(output_file, index=False)
print(f"✅ Final comparison saved to: {output_file}\n")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
