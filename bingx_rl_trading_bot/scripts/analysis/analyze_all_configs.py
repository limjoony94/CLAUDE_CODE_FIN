"""
Comprehensive Analysis of All Threshold Configurations

목표: 통계적으로 유의하고 모든 시장 상태에서 작동하는 설정 찾기
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

print("=" * 80)
print("COMPREHENSIVE CONFIG ANALYSIS")
print("=" * 80)

# Load all configs
all_configs_file = RESULTS_DIR / "backtest_hybrid_v4_all_configs.csv"
df = pd.read_csv(all_configs_file)

configs = df['config_name'].unique()

results_summary = []

for config in configs:
    config_df = df[df['config_name'] == config].copy()

    # Basic stats
    avg_diff = config_df['difference'].mean()
    avg_trades = config_df['num_trades'].mean()
    avg_winrate = config_df['win_rate'].mean()

    # Statistical significance
    t_stat, p_value = stats.ttest_rel(config_df['hybrid_return'], config_df['bh_return'])

    # Regime performance
    bull = config_df[config_df['regime'] == 'Bull']
    bear = config_df[config_df['regime'] == 'Bear']
    sideways = config_df[config_df['regime'] == 'Sideways']

    bull_diff = bull['difference'].mean() if len(bull) > 0 else 0
    bear_diff = bear['difference'].mean() if len(bear) > 0 else 0
    sideways_diff = sideways['difference'].mean() if len(sideways) > 0 else 0

    # Consistency
    std_diff = config_df['difference'].std()
    cv = (std_diff / abs(avg_diff)) if avg_diff != 0 else float('inf')

    # Success rate
    positive_windows = len(config_df[config_df['difference'] > 0])
    success_rate = positive_windows / len(config_df)

    # Zero-trade rate
    zero_trades = len(config_df[config_df['num_trades'] == 0])
    zero_rate = zero_trades / len(config_df)

    results_summary.append({
        'Config': config,
        'vs_BH': avg_diff,
        'Trades': avg_trades,
        'WinRate': avg_winrate,
        'p_value': p_value,
        'Significant': p_value < 0.05,
        'Bull': bull_diff,
        'Bear': bear_diff,
        'Sideways': sideways_diff,
        'CV': cv,
        'SuccessRate': success_rate,
        'ZeroRate': zero_rate
    })

# Convert to DataFrame
summary_df = pd.DataFrame(results_summary)

# Sort by vs_BH
summary_df = summary_df.sort_values('vs_BH', ascending=False)

print(f"\n📊 All Configurations Summary:\n")
print(summary_df[['Config', 'vs_BH', 'Trades', 'p_value', 'Significant', 'Bull', 'SuccessRate']].to_string(index=False))

# Find best viable config
print(f"\n{'=' * 80}")
print("FINDING BEST VIABLE CONFIGURATION")
print(f"{'=' * 80}\n")

print("Criteria:")
print("  1. Statistically significant (p < 0.05)")
print("  2. Bull market: >= -2.0% (not catastrophic)")
print("  3. Success rate >= 65%")
print("  4. Coefficient of Variation < 3.0 (reasonable consistency)")

viable_configs = summary_df[
    (summary_df['Significant'] == True) &
    (summary_df['Bull'] >= -2.0) &
    (summary_df['SuccessRate'] >= 0.65) &
    (summary_df['CV'] < 3.0)
]

if len(viable_configs) > 0:
    print(f"\n✅ Found {len(viable_configs)} viable configuration(s):\n")
    print(viable_configs[['Config', 'vs_BH', 'Trades', 'Bull', 'Bear', 'Sideways']].to_string(index=False))

    best_viable = viable_configs.iloc[0]
    print(f"\n🏆 BEST VIABLE CONFIG: {best_viable['Config']}")
    print(f"  vs B&H: {best_viable['vs_BH']:.2f}%")
    print(f"  p-value: {best_viable['p_value']:.4f} ✅")
    print(f"  Bull: {best_viable['Bull']:.2f}%")
    print(f"  Bear: {best_viable['Bear']:.2f}%")
    print(f"  Sideways: {best_viable['Sideways']:.2f}%")
    print(f"  Trades: {best_viable['Trades']:.1f}")
    print(f"  Success Rate: {best_viable['SuccessRate']*100:.1f}%")
else:
    print("\n❌ NO viable configuration found meeting all criteria!")
    print("\nRelaxing criteria...\n")

    # Relax: Allow higher CV
    viable_configs = summary_df[
        (summary_df['Significant'] == True) &
        (summary_df['Bull'] >= -2.0) &
        (summary_df['SuccessRate'] >= 0.60)
    ]

    if len(viable_configs) > 0:
        print(f"✅ Found {len(viable_configs)} config(s) with relaxed criteria:\n")
        print(viable_configs[['Config', 'vs_BH', 'Trades', 'Bull', 'CV', 'SuccessRate']].to_string(index=False))

        best_viable = viable_configs.iloc[0]
        print(f"\n🏆 BEST VIABLE CONFIG (relaxed): {best_viable['Config']}")
        print(f"  vs B&H: {best_viable['vs_BH']:.2f}%")
        print(f"  p-value: {best_viable['p_value']:.4f} ✅")
        print(f"  Bull: {best_viable['Bull']:.2f}%")
    else:
        print("\n❌ Still no viable config!")
        print("\nBest statistical significance configs:\n")

        sig_configs = summary_df[summary_df['Significant'] == True].head(3)
        if len(sig_configs) > 0:
            print(sig_configs[['Config', 'vs_BH', 'Trades', 'Bull', 'p_value']].to_string(index=False))
        else:
            print("No statistically significant configs found!")

# Critical recommendations
print(f"\n{'=' * 80}")
print("CRITICAL RECOMMENDATIONS")
print(f"{'=' * 80}\n")

if len(viable_configs) == 0:
    print("🚨 NONE of the tested configurations are viable!")
    print("\nReasons:")

    # Check why each failed
    for _, row in summary_df.head(5).iterrows():
        print(f"\n{row['Config']}:")
        if not row['Significant']:
            print(f"  ❌ Not statistically significant (p={row['p_value']:.4f})")
        if row['Bull'] < -2.0:
            print(f"  ❌ Bull market failure ({row['Bull']:.2f}%)")
        if row['SuccessRate'] < 0.65:
            print(f"  ⚠️ Low success rate ({row['SuccessRate']*100:.1f}%)")
        if row['CV'] >= 3.0:
            print(f"  ⚠️ High variance (CV={row['CV']:.2f})")

    print(f"\n{'=' * 80}")
    print("NEXT STEPS:")
    print(f"{'=' * 80}\n")

    print("1. REGIME-SPECIFIC STRATEGIES:")
    print("   - Different thresholds for Bull/Bear/Sideways")
    print("   - Bull: Use LOWER thresholds (more aggressive)")
    print("   - Bear: Use HIGHER thresholds (more conservative)")
    print("   - Sideways: Use MODERATE thresholds")

    print("\n2. ALTERNATIVE APPROACH:")
    print("   - Consider Conservative (-0.66% vs B&H)")
    print("   - While not positive, it's close AND more consistent")
    print("   - Further optimization within Conservative range")

    print("\n3. REALITY CHECK:")
    print("   - Maybe beating B&H consistently is VERY HARD")
    print("   - -0.66% with MUCH lower volatility might be acceptable")
    print("   - Focus on risk-adjusted returns (Sharpe ratio)")

print(f"\n{'=' * 80}")
print("Analysis Complete")
print(f"{'=' * 80}")
