"""
Pattern Selection Analysis - Finding Optimal Balance
Ground Truth Classification Based

Created: 2026-02-04
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Load discovery results
results_path = Path(__file__).parent.parent.parent / "results" / "unified_pattern_discovery.json"
with open(results_path, 'r') as f:
    data = json.load(f)

patterns = data['per_pattern_optimal']
print(f"Total patterns discovered: {len(patterns)}")
print()

# Convert to DataFrame for analysis
df = pd.DataFrame(patterns)

# Add short code pattern column
def to_short_code(pattern_name):
    """Convert full pattern name to short code."""
    type_map = {
        'DOJI': 'D', 'DRAGONFLY': 'DF', 'GRAVESTONE': 'GS',
        'HAMMER': 'H', 'INV_HAMMER': 'IH', 'SPINNING_TOP': 'ST',
        'MARUBOZU_UP': 'MU', 'MARUBOZU_DOWN': 'MD',
        'BIG_UP': 'BU', 'BIG_DOWN': 'BD',
        'MED_UP': 'U', 'MED_DOWN': 'DN'
    }
    parts = pattern_name.split('-')
    return '-'.join(type_map.get(p, p) for p in parts)

df['short_code'] = df['pattern'].apply(to_short_code)

# Add pnl column if missing (use total_pnl)
if 'pnl' not in df.columns and 'total_pnl' in df.columns:
    df['pnl'] = df['total_pnl']

# Basic statistics
print("=" * 80)
print("PATTERN DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

# Direction breakdown
long_patterns = df[df['direction'] == 'LONG']
short_patterns = df[df['direction'] == 'SHORT']
print(f"LONG patterns:  {len(long_patterns)}")
print(f"SHORT patterns: {len(short_patterns)}")
print()

# Key metric distributions
print("=== KEY METRIC DISTRIBUTIONS ===")
print()
print(f"Monte Carlo (MC) p-value:")
print(f"  MC < 0.001: {len(df[df['mc'] < 0.001])} patterns")
print(f"  MC < 0.005: {len(df[df['mc'] < 0.005])} patterns")
print(f"  MC < 0.01:  {len(df[df['mc'] < 0.01])} patterns")
print(f"  MC < 0.02:  {len(df[df['mc'] < 0.02])} patterns")
print(f"  MC < 0.05:  {len(df[df['mc'] < 0.05])} patterns")
print()

print(f"Walk-Forward (WF) score:")
print(f"  WF = 5/5: {len(df[df['wf'] == 5])} patterns")
print(f"  WF >= 4/5: {len(df[df['wf'] >= 4])} patterns")
print(f"  WF >= 3/5: {len(df[df['wf'] >= 3])} patterns")
print()

print(f"Win Rate (WR):")
print(f"  WR >= 70%: {len(df[df['wr'] >= 70])} patterns")
print(f"  WR >= 65%: {len(df[df['wr'] >= 65])} patterns")
print(f"  WR >= 60%: {len(df[df['wr'] >= 60])} patterns")
print(f"  WR >= 55%: {len(df[df['wr'] >= 55])} patterns")
print()

print(f"Trade Count:")
print(f"  Trades >= 100: {len(df[df['trades'] >= 100])} patterns")
print(f"  Trades >= 75:  {len(df[df['trades'] >= 75])} patterns")
print(f"  Trades >= 50:  {len(df[df['trades'] >= 50])} patterns")
print(f"  Trades >= 30:  {len(df[df['trades'] >= 30])} patterns")
print()

# Multi-criteria filtering analysis
print("=" * 80)
print("MULTI-CRITERIA FILTERING SCENARIOS")
print("=" * 80)
print()

scenarios = [
    # (MC, WF, WR_L, WR_S, Trades, name)
    (0.01, 5, 60, 55, 50, "Ultra Strict (current)"),
    (0.01, 4, 60, 55, 50, "Strict (WF≥4)"),
    (0.01, 4, 55, 50, 50, "Moderate A"),
    (0.02, 4, 55, 50, 40, "Moderate B"),
    (0.02, 4, 55, 50, 30, "Balanced A"),
    (0.03, 4, 55, 50, 30, "Balanced B"),
    (0.03, 3, 55, 50, 30, "Relaxed A"),
    (0.05, 3, 50, 50, 30, "Relaxed B"),
]

results_table = []

for mc_th, wf_th, wr_l_th, wr_s_th, trades_th, name in scenarios:
    # LONG filtering
    long_filtered = long_patterns[
        (long_patterns['mc'] < mc_th) &
        (long_patterns['wf'] >= wf_th) &
        (long_patterns['wr'] >= wr_l_th) &
        (long_patterns['trades'] >= trades_th)
    ]

    # SHORT filtering
    short_filtered = short_patterns[
        (short_patterns['mc'] < mc_th) &
        (short_patterns['wf'] >= wf_th) &
        (short_patterns['wr'] >= wr_s_th) &
        (short_patterns['trades'] >= trades_th)
    ]

    total = len(long_filtered) + len(short_filtered)
    avg_wr = 0
    avg_mc = 0
    total_trades = 0

    if total > 0:
        combined = pd.concat([long_filtered, short_filtered])
        avg_wr = combined['wr'].mean()
        avg_mc = combined['mc'].mean()
        total_trades = combined['trades'].sum()

    results_table.append({
        'name': name,
        'mc': mc_th,
        'wf': wf_th,
        'wr_l': wr_l_th,
        'wr_s': wr_s_th,
        'trades': trades_th,
        'long': len(long_filtered),
        'short': len(short_filtered),
        'total': total,
        'avg_wr': avg_wr,
        'avg_mc': avg_mc,
        'total_trades': total_trades
    })

print(f"{'Scenario':<20} {'MC':<6} {'WF':<4} {'WR_L':<5} {'WR_S':<5} {'Tr':<4} | {'L':>3} {'S':>3} {'Tot':>4} | {'AvgWR':>6} {'AvgMC':>7} {'TotTr':>6}")
print("-" * 100)
for r in results_table:
    print(f"{r['name']:<20} {r['mc']:<6} {r['wf']:<4} {r['wr_l']:<5} {r['wr_s']:<5} {r['trades']:<4} | {r['long']:>3} {r['short']:>3} {r['total']:>4} | {r['avg_wr']:>6.1f} {r['avg_mc']:>7.4f} {r['total_trades']:>6}")

print()

# Detailed analysis of promising scenarios
print("=" * 80)
print("DETAILED PATTERN LIST FOR PROMISING SCENARIOS")
print("=" * 80)
print()

# Scenario: Balanced B (MC<0.03, WF>=4, WR>=55/50, Trades>=30)
mc_th, wf_th, wr_l_th, wr_s_th, trades_th = 0.03, 4, 55, 50, 30

long_balanced = long_patterns[
    (long_patterns['mc'] < mc_th) &
    (long_patterns['wf'] >= wf_th) &
    (long_patterns['wr'] >= wr_l_th) &
    (long_patterns['trades'] >= trades_th)
].copy()
long_balanced['score'] = long_balanced['wr'] * np.sqrt(long_balanced['trades']) * (1 - long_balanced['mc'])

short_balanced = short_patterns[
    (short_patterns['mc'] < mc_th) &
    (short_patterns['wf'] >= wf_th) &
    (short_patterns['wr'] >= wr_s_th) &
    (short_patterns['trades'] >= trades_th)
].copy()
short_balanced['score'] = short_balanced['wr'] * np.sqrt(short_balanced['trades']) * (1 - short_balanced['mc'])

print(f"=== BALANCED B SCENARIO (MC<0.03, WF>=4, WR>=55/50, Trades>=30) ===")
print()

print(f"LONG Patterns ({len(long_balanced)}):")
print(f"{'Pattern':<20} {'WR':>6} {'MC':>8} {'WF':>3} {'Trades':>6} {'PnL':>8} {'Score':>8}")
print("-" * 65)
for _, row in long_balanced.sort_values('score', ascending=False).iterrows():
    print(f"{row['short_code']:<20} {row['wr']:>6.1f} {row['mc']:>8.4f} {row['wf']:>3} {row['trades']:>6} {row['pnl']:>8.1f} {row['score']:>8.1f}")

print()
print(f"SHORT Patterns ({len(short_balanced)}):")
print(f"{'Pattern':<20} {'WR':>6} {'MC':>8} {'WF':>3} {'Trades':>6} {'PnL':>8} {'Score':>8}")
print("-" * 65)
for _, row in short_balanced.sort_values('score', ascending=False).iterrows():
    print(f"{row['short_code']:<20} {row['wr']:>6.1f} {row['mc']:>8.4f} {row['wf']:>3} {row['trades']:>6} {row['pnl']:>8.1f} {row['score']:>8.1f}")

print()

# Optimal selection recommendation
print("=" * 80)
print("OPTIMAL PORTFOLIO RECOMMENDATION")
print("=" * 80)
print()

# Score-based top selection from Balanced B
top_long = long_balanced.nlargest(10, 'score')
top_short = short_balanced.nlargest(10, 'score')

print(f"Top 10 LONG by Score:")
for i, (_, row) in enumerate(top_long.iterrows(), 1):
    print(f"  {i:2}. {row['pattern']:<20} WR={row['wr']:.1f}%, MC={row['mc']:.4f}, WF={row['wf']}, Tr={row['trades']}")

print()
print(f"Top 10 SHORT by Score:")
for i, (_, row) in enumerate(top_short.iterrows(), 1):
    print(f"  {i:2}. {row['pattern']:<20} WR={row['wr']:.1f}%, MC={row['mc']:.4f}, WF={row['wf']}, Tr={row['trades']}")

# Final statistics
print()
print("=" * 80)
print("PORTFOLIO SIZE COMPARISON")
print("=" * 80)
print()

for n_long, n_short in [(5, 5), (7, 7), (10, 10), (15, 15)]:
    if len(long_balanced) >= n_long and len(short_balanced) >= n_short:
        sel_long = long_balanced.nlargest(n_long, 'score')
        sel_short = short_balanced.nlargest(n_short, 'score')
        combined = pd.concat([sel_long, sel_short])
        print(f"{n_long}L + {n_short}S ({n_long + n_short} total):")
        print(f"  Avg WR: {combined['wr'].mean():.1f}%")
        print(f"  Avg MC: {combined['mc'].mean():.4f}")
        print(f"  Total Trades: {combined['trades'].sum()}")
        print(f"  Min WF: {combined['wf'].min()}/5")
        print()
