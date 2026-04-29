"""Deep Review — Combined R26 (grid) + R5 (carry) capital allocation analysis.

Analysis D: Does combining 2 highest-bootstrap-stable orthogonal mechanisms
overcome each individual's daily ceiling?

R26: volatility harvest (grid), runs during ranging, +0.05%/day.
R5: funding carry, runs during high-funding regime, +0.009%/day.

Strategy combination methods:
1. **Equal capital split**: $750 each. Each runs at 50% capital.
2. **Time-multiplex**: full $1500 to whichever is "active" today.
3. **Independent**: each gets full $1500 nominally; assume independence.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / 'results'

# R26 result
r26_results_files = sorted(RESULTS.glob('round26_grid_ranging_*.json'))
r5_results_files = sorted(RESULTS.glob('path_b_r5_carry_oos_*.json'))

print('=' * 100)
print('Deep Review — Combined R26 + R5 capital allocation')
print('=' * 100)

if not r26_results_files:
    print('ERROR: R26 result not found')
    exit(1)
if not r5_results_files:
    print('ERROR: R5 result not found')
    exit(1)

with open(r26_results_files[-1]) as f:
    r26 = json.load(f)
with open(r5_results_files[-1]) as f:
    r5 = json.load(f)

print(f'R26: {r26_results_files[-1].name}')
print(f'R5:  {r5_results_files[-1].name}\n')

# Extract metrics
r26_summary = r26.get('summary', {})
r5_summary = r5.get('full_sample', {})

r26_daily = r26_summary.get('daily_pct', 0)
r26_cum = r26_summary.get('cum_net_pct', 0)
r26_n_days = r26_summary.get('cum_net_pct', 0) / r26_daily if r26_daily > 0 else 0
r26_bs_pos_rate = r26.get('c4_bootstrap', {}).get('pos_rate', 0)
r26_n_trades = r26_summary.get('n_trades', 0)
r26_worst_5d = r26_summary.get('worst_5d_pct', 0)

r5_daily = r5_summary.get('avg_daily_net_pct', 0)
r5_apy = r5_summary.get('annualized_net_apy_pct', 0)
r5_bs_data = r5.get('test_2_bootstrap', {})
r5_bs_pos_rate = r5_bs_data.get('pos_rate', 0)
r5_n_days = r5_summary.get('n_days', 0)

print('=== Individual baselines ===')
print(f'R26: daily {r26_daily:+.4f}%, cum_net {r26_cum:+.2f}%, n_trades {r26_n_trades}, '
      f'BS_pos {r26_bs_pos_rate:.4f}, worst_5d {r26_worst_5d:+.2f}%')
print(f'R5:  daily {r5_daily:+.4f}%, apy {r5_apy:+.2f}%, n_days {r5_n_days}, '
      f'BS_pos {r5_bs_pos_rate:.4f}\n')

# === Method 1: Equal capital split ===
# Each strategy operates on 50% of capital
# Their per-strategy returns are halved when measured against full capital
m1_r26_daily_on_full = r26_daily * 0.5
m1_r5_daily_on_full = r5_daily * 0.5
m1_combined_daily = m1_r26_daily_on_full + m1_r5_daily_on_full
print('=== Method 1: Equal capital split ($750/$750) ===')
print(f'R26 contribution to full $1500: {m1_r26_daily_on_full:+.4f}%/day')
print(f'R5  contribution to full $1500: {m1_r5_daily_on_full:+.4f}%/day')
print(f'Combined: {m1_combined_daily:+.4f}%/day  → vs target 0.20%: '
      f'{"PASS" if m1_combined_daily >= 0.20 else f"FAIL ({0.20/m1_combined_daily:.1f}× under)"}\n')

# === Method 2: Time-multiplex (full capital to whichever is active) ===
# R26 active during ranging (52% of bars in R26 result)
# R5 active during funding regime (~60% of days based on R5 prior result)
# Overlap: assume independent → both active 0.52 × 0.60 = 31.2%
# Time-multiplex assumption: full capital to one at a time when both active, equal split
ranging_frac = r26_summary.get('ranging_fraction', 0.52)
r5_active_frac = 0.60  # rough estimate from R5 78% positive_regime / 800d

p_only_r26 = ranging_frac * (1 - r5_active_frac)
p_only_r5 = (1 - ranging_frac) * r5_active_frac
p_both = ranging_frac * r5_active_frac
p_neither = (1 - ranging_frac) * (1 - r5_active_frac)

# When only R26 active: full capital → r26 daily
# When only R5 active: full capital → r5 daily
# When both: equal split → 0.5 of each
# When neither: 0
m2_combined_daily = (
    p_only_r26 * r26_daily +
    p_only_r5 * r5_daily +
    p_both * 0.5 * (r26_daily + r5_daily) +
    p_neither * 0
)
print('=== Method 2: Time-multiplex (full capital to active) ===')
print(f'P(only R26 active): {p_only_r26:.4f}, P(only R5): {p_only_r5:.4f}, '
      f'P(both): {p_both:.4f}, P(neither): {p_neither:.4f}')
print(f'Combined daily: {m2_combined_daily:+.4f}%  → vs target 0.20%: '
      f'{"PASS" if m2_combined_daily >= 0.20 else f"FAIL ({0.20/m2_combined_daily:.1f}× under)"}\n')

# === Method 3: Independence assumption (same capital, parallel positions) ===
# Both strategies use full $1500 simultaneously when active.
# This requires capital to actually NOT be constrained — i.e., R5 spot+perp
# uses different account/account portion than R26 grid limits.
# Fallback assumption: feasible at retail BingX (margin accounts cross-cover).
m3_combined_daily = r26_daily + r5_daily
print('=== Method 3: Capital independence (both use full $1500) ===')
print(f'Combined daily: {m3_combined_daily:+.4f}%  → vs target 0.20%: '
      f'{"PASS" if m3_combined_daily >= 0.20 else f"FAIL ({0.20/m3_combined_daily:.1f}× under)"}')
print('NOTE: Method 3 not realistic — R5 ties up cash for spot leg, R26 ties up')
print('cash for limit orders. Cannot share same capital simultaneously.\n')

# === Method 4: R26 with R5-style funding accrual on grid open positions ===
# When R26 is in a long position (buy filled), it could capture funding
# while the position is open. This is "free" funding for the LONG fraction
# of grid open positions.
# Assume: 50% of open time is LONG (symmetric grid), 50% is SHORT
# Net funding exposure: 0 (LONG receives + funding, SHORT pays + funding cancel out
# in expectation). So no benefit from this addition.
print('=== Method 4: R26 + funding accrual on open positions ===')
print('Symmetric grid → LONG funding = -SHORT funding in expectation')
print('Net funding contribution: 0. Insufficient.\n')

# === Method 5: R26 + leverage on R26 only (NOT user-target — for comparison) ===
# At 4× leverage on R26: theoretical 0.20%/day, ruin tail TBD
# Already covered in R5+leverage analysis (#152)
print('=== Method 5: R26 leverage 4× (separate from this analysis) ===')
print('Per advisor R5+leverage frontier method, R26 4× theoretical 0.20%/day')
print('Drift drawdown × 4 = -24% over 720d, ruin probability TBD')
print('User explicit constraint: NO leverage for 0.20% target (1% if leveraged)')
print('-> NOT a path to user (β) requirement\n')

# === Verdict ===
print('=' * 100)
print('VERDICT - Combined mechanism analysis')
print('=' * 100)
best = max(m1_combined_daily, m2_combined_daily)
print(f'Best combined daily achievable at 1× capital: {best:+.4f}%/day')
print(f'User target: +0.20%/day')
print(f'Gap factor: {0.20 / best:.2f}× (at best combined method)')
print(f'\nIndividual ceilings:')
print(f'  R26 alone: {r26_daily:+.4f}%/day')
print(f'  R5  alone: {r5_daily:+.4f}%/day')
print(f'  Best combined: {best:+.4f}%/day')
print(f'\nConclusion: Combining R26 + R5 does not exceed R26 alone meaningfully')
print(f'(time-multiplex {m2_combined_daily:+.4f} vs R26 alone {r26_daily:+.4f})')

out = {
    'date': datetime.now(timezone.utc).isoformat(),
    'analysis': 'D - combined mechanism daily ceiling',
    'r26_alone_daily_pct': r26_daily,
    'r5_alone_daily_pct': r5_daily,
    'method_1_equal_split_daily_pct': m1_combined_daily,
    'method_2_time_multiplex_daily_pct': m2_combined_daily,
    'method_3_independence_daily_pct': m3_combined_daily,
    'best_combined_daily_pct': best,
    'user_target_daily_pct': 0.20,
    'gap_factor': 0.20 / best,
}
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
p = RESULTS / f'deep_review_combined_{ts}.json'
with open(p, 'w') as fp:
    json.dump(out, fp, indent=2, default=str)
print(f'\nSaved: {p}')
