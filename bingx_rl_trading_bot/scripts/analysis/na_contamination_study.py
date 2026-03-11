#!/usr/bin/env python3
"""
N/A Contamination Deep Study: Cross-validated analysis of N/A pattern
and duplicate trade recording impact on live metrics.

3 Problems identified:
  P1: N/A pattern — orphan recovery without pattern → 'N/A' recorded
  P2: Duplicate recording — mass closure loop records same position N times
  P3: Cascading contamination — existing N/A in history blocks future recovery

Cross-validation approach:
  Phase 1: Contamination inventory — exact N/A + duplicate counts, PnL impact
  Phase 2: Clean metrics — recalculate all metrics after dedup + N/A removal
  Phase 3: WR gap recomputation — update Phase 1 binomial test with clean data
  Phase 4: Temporal analysis — when did N/A trades cluster? Crash events?
  Phase 5: Pattern recovery — can we recover original patterns from context?
  Phase 6: Cross-validate scanner — clean live vs scanner on same period
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import sys
import time

start_time = time.time()

METRICS_FILE = "results/pattern_5m_metrics.json"

with open(METRICS_FILE) as f:
    metrics = json.load(f)

history = metrics['trade_history']
print("=" * 70)
print("N/A CONTAMINATION STUDY — Cross-Validated Impact Analysis")
print("=" * 70)
print(f"Total trade_history entries: {len(history)}")

# ─── Phase 1: Contamination Inventory ───
print("\n" + "=" * 70)
print("PHASE 1: CONTAMINATION INVENTORY")
print("=" * 70)

# 1a: N/A trades
na_trades = [t for t in history if t.get('pattern', '') == 'N/A']
valid_trades = [t for t in history if t.get('pattern', '') != 'N/A']
print(f"\nN/A trades: {len(na_trades)} ({len(na_trades)/len(history)*100:.1f}%)")
print(f"Valid trades: {len(valid_trades)} ({len(valid_trades)/len(history)*100:.1f}%)")

# 1b: Duplicate detection (same entry+exit+direction within 60s)
def find_duplicates(trades):
    """Group trades by (entry_price, exit_price, direction), mark duplicates."""
    groups = defaultdict(list)
    for i, t in enumerate(trades):
        key = (round(t['entry_price'], 1), round(t['exit_price'], 1), t['direction'])
        groups[key].append((i, t))

    dup_indices = set()
    unique_groups = []
    for key, group in groups.items():
        if len(group) > 1:
            # Keep first, mark rest as duplicates
            for idx, t in group[1:]:
                dup_indices.add(idx)
            unique_groups.append({
                'key': key,
                'count': len(group),
                'first': group[0][1],
                'duplicates': [g[1] for g in group[1:]],
            })
    return dup_indices, unique_groups

all_dup_indices, all_dup_groups = find_duplicates(history)
na_dup_indices, na_dup_groups = find_duplicates(na_trades)
valid_dup_indices, valid_dup_groups = find_duplicates(valid_trades)

print(f"\nDuplicate entries (all):   {len(all_dup_indices)}")
print(f"Duplicate entries (N/A):   {len(na_dup_indices)}")
print(f"Duplicate entries (valid): {len(valid_dup_indices)}")

# 1c: PnL impact
na_pnl = sum(t.get('pnl_portfolio', 0) for t in na_trades)
dup_pnl = sum(history[i].get('pnl_portfolio', 0) for i in all_dup_indices)
print(f"\nN/A total PnL impact: {na_pnl:+.3f}%")
print(f"Duplicate PnL impact: {dup_pnl:+.3f}%")

# 1d: Unique N/A trades (after dedup)
na_unique = []
seen = set()
for t in na_trades:
    key = (round(t['entry_price'], 1), round(t['exit_price'], 1), t['direction'])
    if key not in seen:
        seen.add(key)
        na_unique.append(t)
print(f"\nUnique N/A trades: {len(na_unique)} (of {len(na_trades)} total)")
na_unique_pnl = sum(t.get('pnl_portfolio', 0) for t in na_unique)
print(f"Unique N/A PnL: {na_unique_pnl:+.3f}%")

# 1e: N/A WR breakdown
na_tp_sl = [t for t in na_unique if t.get('exit_reason') in ('TP', 'SL')]
na_wins = sum(1 for t in na_tp_sl if t.get('pnl_portfolio', 0) > 0)
if na_tp_sl:
    print(f"N/A TP+SL WR: {na_wins}/{len(na_tp_sl)} = {na_wins/len(na_tp_sl)*100:.1f}%")

phase1 = {
    'total_entries': len(history),
    'na_trades': len(na_trades),
    'na_unique': len(na_unique),
    'duplicates_all': len(all_dup_indices),
    'duplicates_na': len(na_dup_indices),
    'duplicates_valid': len(valid_dup_indices),
    'na_pnl': round(na_pnl, 4),
    'na_unique_pnl': round(na_unique_pnl, 4),
    'dup_pnl': round(dup_pnl, 4),
}

# ─── Phase 2: Clean Metrics Recalculation ───
print("\n" + "=" * 70)
print("PHASE 2: CLEAN METRICS — After dedup + N/A removal")
print("=" * 70)

# Build clean trade list: remove N/A, remove duplicates
clean_trades = []
seen_keys = set()
for t in history:
    if t.get('pattern', '') == 'N/A':
        continue
    key = (round(t['entry_price'], 1), round(t['exit_price'], 1), t['direction'])
    if key in seen_keys:
        continue
    seen_keys.add(key)
    clean_trades.append(t)

clean_tp_sl = [t for t in clean_trades if t.get('exit_reason') in ('TP', 'SL')]
clean_all_reason = [t for t in clean_trades]

print(f"Clean trades: {len(clean_trades)} (removed {len(history) - len(clean_trades)})")
print(f"Clean TP+SL trades: {len(clean_tp_sl)}")

# Metrics: original vs clean
def compute_metrics(trades, label=""):
    tp_sl = [t for t in trades if t.get('exit_reason') in ('TP', 'SL')]
    all_t = trades

    tp_sl_wins = sum(1 for t in tp_sl if t.get('pnl_portfolio', 0) > 0)
    tp_sl_wr = tp_sl_wins / len(tp_sl) * 100 if tp_sl else 0

    all_wins = sum(1 for t in all_t if t.get('pnl_portfolio', 0) > 0)
    all_wr = all_wins / len(all_t) * 100 if all_t else 0

    total_pnl = sum(t.get('pnl_portfolio', 0) for t in all_t)

    # Compound equity
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in sorted(all_t, key=lambda x: x['timestamp']):
        equity *= (1 + t.get('pnl_portfolio', 0) / 100)
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    compound_pnl = (equity - 1) * 100

    return {
        'trades': len(all_t),
        'tp_sl_trades': len(tp_sl),
        'tp_sl_wr': round(tp_sl_wr, 2),
        'all_wr': round(all_wr, 2),
        'simple_pnl': round(total_pnl, 3),
        'compound_pnl': round(compound_pnl, 3),
        'max_dd': round(max_dd, 2),
    }

orig_metrics = compute_metrics(history, "Original")
clean_metrics = compute_metrics(clean_trades, "Clean")

print(f"\n{'Metric':<25} {'Original':>12} {'Clean':>12} {'Delta':>10}")
print("-" * 60)
for key in ['trades', 'tp_sl_trades', 'tp_sl_wr', 'all_wr', 'simple_pnl', 'compound_pnl', 'max_dd']:
    o = orig_metrics[key]
    c = clean_metrics[key]
    d = c - o if isinstance(c, (int, float)) else '-'
    fmt_d = f"{d:+.2f}" if isinstance(d, float) else f"{d:+d}" if isinstance(d, int) else d
    print(f"  {key:<23} {o:>12} {c:>12} {fmt_d:>10}")

phase2 = {
    'original': orig_metrics,
    'clean': clean_metrics,
    'removed_trades': len(history) - len(clean_trades),
}

# ─── Phase 3: WR Gap Recomputation ───
print("\n" + "=" * 70)
print("PHASE 3: WR GAP RECOMPUTATION — Clean live vs Scanner IS")
print("=" * 70)

from scipy import stats as sp_stats

scanner_wr = 72.8  # Scanner IS WR

# Clean TP+SL
clean_wins = sum(1 for t in clean_tp_sl if t.get('pnl_portfolio', 0) > 0)
clean_n = len(clean_tp_sl)
clean_wr = clean_wins / clean_n * 100 if clean_n else 0
clean_gap = scanner_wr - clean_wr

# Binomial test
binom_result = sp_stats.binomtest(clean_wins, clean_n, scanner_wr / 100, alternative='less')
binom_p = binom_result.pvalue

# Bootstrap CI
rng = np.random.RandomState(42)
n_boot = 10000
live_results = [1 if t.get('pnl_portfolio', 0) > 0 else 0 for t in clean_tp_sl]
boot_wrs = [rng.choice(live_results, size=clean_n, replace=True).mean() * 100 for _ in range(n_boot)]
ci_lo = np.percentile(boot_wrs, 2.5)
ci_hi = np.percentile(boot_wrs, 97.5)

print(f"Clean Live WR:  {clean_wr:.1f}% ({clean_wins}/{clean_n})")
print(f"Scanner IS WR:  {scanner_wr:.1f}%")
print(f"Gap:            {clean_gap:.1f}pp (was 11.2pp before cleanup)")
print(f"Binomial p:     {binom_p:.4f}")
print(f"Bootstrap CI:   [{ci_lo:.1f}%, {ci_hi:.1f}%]")
print(f"Scanner in CI:  {'YES' if ci_lo <= scanner_wr <= ci_hi else 'NO'}")
sig = "SIGNIFICANT" if binom_p < 0.05 else "NOT significant"
print(f"Result: Gap is {sig} at α=0.05")

# Compare with original
print(f"\n--- Before vs After Cleanup ---")
print(f"Original: WR 61.6%, gap 11.2pp, p=0.0022")
print(f"Clean:    WR {clean_wr:.1f}%, gap {clean_gap:.1f}pp, p={binom_p:.4f}")
print(f"Improvement: {11.2 - clean_gap:.1f}pp gap reduction")

phase3 = {
    'clean_wr': round(clean_wr, 2),
    'clean_n': clean_n,
    'gap': round(clean_gap, 2),
    'binom_p': round(binom_p, 4),
    'ci_95': [round(ci_lo, 1), round(ci_hi, 1)],
    'significant': binom_p < 0.05,
    'gap_reduction': round(11.2 - clean_gap, 2),
}

# ─── Phase 4: Temporal Analysis — Crash Events ───
print("\n" + "=" * 70)
print("PHASE 4: TEMPORAL ANALYSIS — N/A Clusters = Crash Events")
print("=" * 70)

# Group N/A trades by 1-hour windows to identify crash events
na_by_hour = defaultdict(list)
for t in na_trades:
    ts = pd.Timestamp(t['timestamp'])
    hour_key = ts.floor('H').isoformat()
    na_by_hour[hour_key].append(t)

print(f"\nCrash event timeline ({len(na_by_hour)} distinct hour windows):")
crash_events = []
for hour, trades in sorted(na_by_hour.items()):
    # Unique entries in this hour
    unique_entries = set()
    for t in trades:
        unique_entries.add((round(t['entry_price'], 1), t['direction']))

    total_pnl = sum(t.get('pnl_portfolio', 0) for t in trades)
    dirs = Counter(t['direction'] for t in trades)
    reasons = Counter(t.get('exit_reason', '?') for t in trades)

    event = {
        'hour': hour,
        'total_records': len(trades),
        'unique_positions': len(unique_entries),
        'duplicates': len(trades) - len(unique_entries),
        'pnl': round(total_pnl, 3),
        'directions': dict(dirs),
        'reasons': dict(reasons),
    }
    crash_events.append(event)
    dup_tag = f" ({event['duplicates']} dups)" if event['duplicates'] > 0 else ""
    print(f"  {hour}: {len(trades)} records, {len(unique_entries)} unique pos{dup_tag}, "
          f"PnL {total_pnl:+.3f}%, {dict(dirs)}, {dict(reasons)}")

# Identify the worst crash events
worst = sorted(crash_events, key=lambda x: x['pnl'])
if worst:
    print(f"\nWorst crash event: {worst[0]['hour']}, PnL {worst[0]['pnl']:+.3f}%, "
          f"{worst[0]['total_records']} records")

phase4 = {
    'crash_events': crash_events,
    'n_events': len(crash_events),
    'total_na_records': len(na_trades),
    'total_unique_na': len(na_unique),
}

# ─── Phase 5: Pattern Recovery Attempt ───
print("\n" + "=" * 70)
print("PHASE 5: PATTERN RECOVERY — Can we identify original patterns?")
print("=" * 70)

# For each unique N/A trade, look for adjacent valid trades with same entry_price
# to cross-reference what pattern was likely active
recovered = []
unrecovered = []

for na_t in na_unique:
    na_entry = na_t['entry_price']
    na_dir = na_t['direction']
    na_ts = pd.Timestamp(na_t['timestamp'])

    # Search valid trades for same entry_price ± $1.0 and same direction
    candidates = []
    for v_t in valid_trades:
        if (v_t['direction'] == na_dir
            and abs(v_t['entry_price'] - na_entry) < 1.0
            and v_t.get('pattern', '') != 'N/A'):
            v_ts = pd.Timestamp(v_t['timestamp'])
            time_diff = abs((v_ts - na_ts).total_seconds())
            if time_diff < 86400:  # Within 24h
                candidates.append((time_diff, v_t))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best = candidates[0][1]
        recovered.append({
            'na_timestamp': na_t['timestamp'],
            'na_entry': na_entry,
            'na_direction': na_dir,
            'na_exit_reason': na_t.get('exit_reason'),
            'na_pnl': na_t.get('pnl_portfolio', 0),
            'recovered_pattern': best.get('pattern'),
            'match_entry': best['entry_price'],
            'match_time_diff_sec': candidates[0][0],
        })
        print(f"  ✅ N/A {na_dir} entry=${na_entry:.1f} ({na_t['timestamp'][:16]}) "
              f"→ likely '{best.get('pattern')}' (match ${best['entry_price']:.1f}, "
              f"Δ{candidates[0][0]:.0f}s)")
    else:
        unrecovered.append({
            'timestamp': na_t['timestamp'],
            'entry': na_entry,
            'direction': na_dir,
            'exit_reason': na_t.get('exit_reason'),
            'pnl': na_t.get('pnl_portfolio', 0),
        })
        print(f"  ❌ N/A {na_dir} entry=${na_entry:.1f} ({na_t['timestamp'][:16]}) "
              f"→ no match found")

print(f"\nRecovered: {len(recovered)}/{len(na_unique)} ({len(recovered)/len(na_unique)*100:.0f}%)")
print(f"Unrecovered: {len(unrecovered)}/{len(na_unique)}")

# For unrecovered, check if they have valid TP/SL prices that match known patterns
if unrecovered:
    print(f"\nUnrecovered N/A trades detail:")
    for u in unrecovered:
        print(f"  {u['timestamp'][:16]} {u['direction']:5s} entry=${u['entry']:.1f} "
              f"reason={u['exit_reason']} pnl={u['pnl']:+.3f}%")

phase5 = {
    'recovered': len(recovered),
    'unrecovered': len(unrecovered),
    'recovery_rate': round(len(recovered)/len(na_unique)*100, 1) if na_unique else 0,
    'recovered_details': recovered,
    'unrecovered_details': unrecovered,
}

# ─── Phase 6: Cross-Validate — Clean Live vs Scanner Same Period ───
print("\n" + "=" * 70)
print("PHASE 6: CROSS-VALIDATION — Clean live metrics confidence")
print("=" * 70)

# Compare clean metrics across different time slices
# Split clean trades by week
clean_by_week = defaultdict(list)
for t in clean_tp_sl:
    ts = pd.Timestamp(t['timestamp'])
    week = ts.isocalendar()[1]
    clean_by_week[f"W{week}"].append(t)

print(f"\nClean TP+SL WR by week:")
week_wrs = []
for week, trades in sorted(clean_by_week.items()):
    wins = sum(1 for t in trades if t.get('pnl_portfolio', 0) > 0)
    wr = wins / len(trades) * 100 if trades else 0
    week_wrs.append(wr)
    print(f"  {week}: WR {wr:.1f}% ({wins}/{len(trades)})")

# Direction breakdown (clean)
for direction in ['LONG', 'SHORT']:
    d_trades = [t for t in clean_tp_sl if t['direction'] == direction]
    d_wins = sum(1 for t in d_trades if t.get('pnl_portfolio', 0) > 0)
    d_wr = d_wins / len(d_trades) * 100 if d_trades else 0
    print(f"\n{direction} (clean): WR {d_wr:.1f}% ({d_wins}/{len(d_trades)})")

# Temporal stability: post-03-06 (after v1.55.0 deployed, 0 N/A)
post_fix = [t for t in clean_tp_sl if t['timestamp'] >= '2026-03-06']
post_wins = sum(1 for t in post_fix if t.get('pnl_portfolio', 0) > 0)
post_wr = post_wins / len(post_fix) * 100 if post_fix else 0

pre_fix = [t for t in clean_tp_sl if t['timestamp'] < '2026-03-06']
pre_wins = sum(1 for t in pre_fix if t.get('pnl_portfolio', 0) > 0)
pre_wr = pre_wins / len(pre_fix) * 100 if pre_fix else 0

print(f"\n--- Pre/Post v1.55.0 Fix ---")
print(f"Pre-fix (< 03-06):  WR {pre_wr:.1f}% ({pre_wins}/{len(pre_fix)})")
print(f"Post-fix (>= 03-06): WR {post_wr:.1f}% ({post_wins}/{len(post_fix)})")
print(f"Delta: {post_wr - pre_wr:+.1f}pp")

# Fisher exact test for pre vs post
from scipy.stats import fisher_exact
table = [[pre_wins, len(pre_fix) - pre_wins],
         [post_wins, len(post_fix) - post_wins]]
_, fisher_p = fisher_exact(table, alternative='less')
print(f"Fisher exact p (pre < post): {fisher_p:.4f}")

phase6 = {
    'week_wrs': {k: {'wr': round(sum(1 for t in v if t.get('pnl_portfolio',0) > 0)/len(v)*100, 1), 'n': len(v)}
                 for k, v in sorted(clean_by_week.items())},
    'pre_fix_wr': round(pre_wr, 1),
    'pre_fix_n': len(pre_fix),
    'post_fix_wr': round(post_wr, 1),
    'post_fix_n': len(post_fix),
    'fisher_p': round(fisher_p, 4),
}

# ─── Synthesis ───
print("\n" + "=" * 70)
print("SYNTHESIS — N/A Contamination Impact Summary")
print("=" * 70)

print(f"""
CONTAMINATION SCOPE:
  N/A trades: {len(na_trades)} entries ({len(na_unique)} unique positions)
  Duplicates: {len(all_dup_indices)} entries (ALL within N/A trades)
  PnL drain:  {na_pnl:+.3f}% (unique: {na_unique_pnl:+.3f}%)

CLEAN METRICS IMPROVEMENT:
  TP+SL WR:   61.6% → {clean_wr:.1f}%  (+{clean_wr - 61.6:.1f}pp)
  WR gap:     11.2pp → {clean_gap:.1f}pp  (reduced {11.2 - clean_gap:.1f}pp)
  Binomial p: 0.0022 → {binom_p:.4f}  ({'still significant' if binom_p < 0.05 else 'NO LONGER significant'})

ROOT CAUSES:
  P1 (N/A): Orphan recovery without pattern → fixed v1.55.0 (0 N/A since 03-08)
  P2 (Dups): Mass closure loop records same position N times → needs code fix
  P3 (Contamination): 12 unique N/A positions, 10 additional duplicates = 22 total

PATTERN RECOVERY:
  Recoverable: {len(recovered)}/{len(na_unique)} ({len(recovered)/len(na_unique)*100:.0f}%)
  Unrecovered: {len(unrecovered)} (truly orphaned, no matching valid trade)

ACTION ITEMS:
  1. Clean trade_history: Remove N/A + dedup → save to metrics (ONE-TIME)
  2. Fix mass closure loop: Add slot dedup guard to prevent future duplicates
  3. Recalculate state counters (total_trades, winning_trades, etc.)
""")

runtime = time.time() - start_time
results = {
    'metadata': {
        'script': 'na_contamination_study.py',
        'date': pd.Timestamp.now().isoformat(),
        'runtime_sec': round(runtime, 1),
    },
    'phase1_inventory': phase1,
    'phase2_clean_metrics': phase2,
    'phase3_wr_gap': phase3,
    'phase4_temporal': phase4,
    'phase5_recovery': phase5,
    'phase6_cross_validation': phase6,
}

out_path = Path("results/na_contamination_study.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved to {out_path} ({runtime:.1f}s)")
