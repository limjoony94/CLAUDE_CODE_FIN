"""
TP/SL Lineage Analysis & Selective Re-optimization
====================================================
각 패턴의 TP/SL 출처 추적 + 선별적 업데이트 전략 비교.

Lineage Categories:
  A. v1.26.4 optimized (31): PnL maximization + 5-phase validation
  B. Legacy v1.26.1 T5 (20): R:R >= 0.75 objective (다른 로직)
  C. v1.26.4 rejected (1): DN-MD-DN (deep validation fail)

All values then TP * 0.7 in v1.27.0 (Uniform TP 70%).

Question: 20 legacy patterns are from DIFFERENT optimization logic.
Would selectively updating them improve the portfolio?

Hybrid strategies tested:
  H1: Update only 20 legacy patterns
  H2: Update only patterns with current MC > 0.01 (weak statistical evidence)
  H3: Update only high-confidence (score >= 3/4) patterns
  H4: Update only patterns where new TP/SL has SAME SL (TP-only change)
  H5: Apply per-pattern optimal TP scaling (not uniform 70%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL, VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
)

MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
MC_THRESHOLD = 0.01

print("=" * 80)
print("TP/SL Lineage Analysis & Selective Re-optimization")
print("=" * 80)

# ============================================================
# Load data + classify (standard)
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values
types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

print("Done.\n")


# ============================================================
# Engine functions
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnls.append((tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT)
                break
            elif ht:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hs:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 5:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def collect_trades_1pos(pattern_map):
    raw_trades = []
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
        entry_bar = i + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp / 100)
            slp = entry * (1 - sl / 100)
        else:
            tpp = entry * (1 - tp / 100)
            slp = entry * (1 + sl / 100)
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                raw_trades.append((entry_bar, j, pnl))
                break
            elif ht:
                raw_trades.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                raw_trades.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in raw_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def eval_portfolio(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0, 'max_consec': 0}
    pnl_list = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wps = []; lps = []
    streak = 0; max_streak = 0
    for p in pnl_list:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0:
            wins += 1; wps.append(p)
            streak = 0
        else:
            lps.append(abs(p))
            streak += 1
            if streak > max_streak: max_streak = streak
    tw = sum(wps); tl = sum(lps)
    wr = wins / len(pnl_list) * 100
    return {
        'pnl': round(cum, 1), 'trades': len(pnl_list), 'wr': round(wr, 1),
        'mdd': round(mdd, 1), 'pf': round(tw / tl if tl > 0 else 999, 2),
        'pnl_mdd': round(cum / mdd if mdd > 0 else 999, 1),
        'max_consec': max_streak,
    }


def portfolio_mc_mdd(trades, n_sims=10000):
    pnl_list = np.array([t[2] for t in trades])
    mdds = []
    for _ in range(n_sims):
        shuffled = np.random.permutation(pnl_list)
        cum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        mdds.append(np.max(dd))
    return np.array(mdds)


# ============================================================
# Build lineage maps
# ============================================================
# Load v1.26.4 optimization changes
opt_path = Path(__file__).resolve().parent / '../../results/tp_sl_optimization_v1264.json'
with open(opt_path) as f:
    v1264_data = json.load(f)
v1264_changes = {c['pattern'] for c in v1264_data['changes']}

# Load v1.26.4 deep validation rejects
dv_path = Path(__file__).resolve().parent / '../../results/tp_sl_deep_validation.json'
with open(dv_path) as f:
    dv_data = json.load(f)
v1264_rejected = set(dv_data['phase5_final']['rejected'])

# Load re-optimization results
reopt_path = Path(__file__).resolve().parent / '../../results/tp_sl_reoptimization_v1270.json'
with open(reopt_path) as f:
    reopt_data = json.load(f)

reopt_grid = reopt_data['phase1_grid_search']['per_pattern']
reopt_valid = reopt_data['phase2_validation']['per_pattern']

# Current v1.27.0 map
current_map = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('LONG', tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('SHORT', tp, sl)

# Classify lineage
legacy_pats = set()  # from v1.26.1 T5
v1264_pats = set()   # from v1.26.4 grid search
for pat in current_map:
    if pat in v1264_changes and pat not in v1264_rejected:
        v1264_pats.add(pat)
    else:
        legacy_pats.add(pat)  # includes DN-MD-DN (rejected)

print(f"Lineage Classification:")
print(f"  v1.26.4 optimized: {len(v1264_pats)} patterns")
print(f"  Legacy (v1.26.1):  {len(legacy_pats)} patterns")


# ============================================================
# Phase 1: Per-pattern analysis by lineage
# ============================================================
print(f"\n{'='*80}")
print("PHASE 1: Per-Pattern Health Check by Lineage")
print(f"{'='*80}\n")

# For each pattern, compute current MC and check if it passes constraints
print(f"  {'Pattern':<14} {'Dir':>5} {'Lineage':>8} {'TP/SL':>7} {'Trades':>6} {'WR':>6} "
      f"{'PnL':>8} {'MC':>7} {'Edge':>6} {'Status':<12}")
print("  " + "-" * 95)

pattern_health = {}
for pat in sorted(current_map.keys()):
    direction, cur_tp, cur_sl = current_map[pat]
    lineage = 'v1.26.4' if pat in v1264_pats else 'LEGACY'
    indices = pattern_indices.get(pat, np.array([]))

    pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    n = len(pnls)
    pnl = sum(pnls) if pnls else 0
    wr = sum(1 for x in pnls if x > 0) / n * 100 if n > 0 else 0
    mc = mc_test(pnls) if n >= 5 else 1.0

    # Edge: actual WR vs random walk baseline
    rw_wr = cur_sl / (cur_tp + cur_sl) * 100
    edge = wr - rw_wr

    # Status flags
    flags = []
    if mc >= 0.01:
        flags.append('MC_FAIL')
    if edge < 5:
        flags.append('LOW_EDGE')
    if n < 15:
        flags.append('FEW_TRADES')
    status = ', '.join(flags) if flags else 'OK'

    flag_marker = '*' if lineage == 'LEGACY' else ' '
    print(f"{flag_marker} {pat:<13} {direction:>5} {lineage:>8} {cur_tp}/{cur_sl:>3} {n:>6} "
          f"{wr:>5.1f}% {pnl:>+7.1f}% {mc:>7.4f} {edge:>+5.1f}pp {status:<12}")

    pattern_health[pat] = {
        'lineage': lineage, 'direction': direction,
        'tp': cur_tp, 'sl': cur_sl,
        'trades': n, 'wr': round(wr, 1), 'pnl': round(pnl, 1),
        'mc': round(mc, 4), 'edge': round(edge, 1),
        'status': status,
    }

# Summary by lineage
print(f"\n  Summary by Lineage:")
for cat in ['v1.26.4', 'LEGACY']:
    pats = [p for p, h in pattern_health.items() if h['lineage'] == cat]
    mc_fail = sum(1 for p in pats if pattern_health[p]['mc'] >= 0.01)
    low_edge = sum(1 for p in pats if pattern_health[p]['edge'] < 5)
    avg_mc = np.mean([pattern_health[p]['mc'] for p in pats])
    avg_edge = np.mean([pattern_health[p]['edge'] for p in pats])
    avg_pnl = np.mean([pattern_health[p]['pnl'] for p in pats])
    print(f"    {cat:>8}: {len(pats)} patterns, MC fail: {mc_fail}, low edge: {low_edge}, "
          f"avg MC: {avg_mc:.4f}, avg edge: {avg_edge:+.1f}pp, avg PnL: {avg_pnl:+.1f}%")


# ============================================================
# Phase 2: Hybrid Strategy Testing
# ============================================================
print(f"\n{'='*80}")
print("PHASE 2: Hybrid Strategy Testing")
print(f"{'='*80}")

# Get new optimal TP/SL from re-optimization
new_tpsl = {}
for pat, r in reopt_grid.items():
    if r['best'] is not None:
        new_tpsl[pat] = (r['best']['tp'], r['best']['sl'])

# Strategy H0: Current v1.27.0 (baseline)
h0_map = dict(current_map)

# Strategy H1: Update ONLY 20 legacy patterns
h1_map = dict(current_map)
h1_changes = []
for pat in legacy_pats:
    if pat in new_tpsl:
        direction = current_map[pat][0]
        new_tp, new_sl = new_tpsl[pat]
        old_tp, old_sl = current_map[pat][1], current_map[pat][2]
        if new_tp != old_tp or new_sl != old_sl:
            h1_map[pat] = (direction, new_tp, new_sl)
            h1_changes.append(pat)

# Strategy H2: Update only patterns with MC > 0.01
h2_map = dict(current_map)
h2_changes = []
for pat in current_map:
    if pattern_health[pat]['mc'] >= 0.01 and pat in new_tpsl:
        direction = current_map[pat][0]
        new_tp, new_sl = new_tpsl[pat]
        old_tp, old_sl = current_map[pat][1], current_map[pat][2]
        if new_tp != old_tp or new_sl != old_sl:
            h2_map[pat] = (direction, new_tp, new_sl)
            h2_changes.append(pat)

# Strategy H3: Update only high-confidence (score >= 3) patterns
h3_map = dict(current_map)
h3_changes = []
for pat in current_map:
    if pat in reopt_valid and reopt_valid[pat]['composite_score'] >= 3:
        if pat in new_tpsl:
            direction = current_map[pat][0]
            new_tp, new_sl = new_tpsl[pat]
            old_tp, old_sl = current_map[pat][1], current_map[pat][2]
            if new_tp != old_tp or new_sl != old_sl:
                h3_map[pat] = (direction, new_tp, new_sl)
                h3_changes.append(pat)

# Strategy H4: Update only legacy + MC fail patterns (union of H1 & H2)
h4_map = dict(current_map)
h4_targets = legacy_pats | {p for p in current_map if pattern_health[p]['mc'] >= 0.01}
h4_changes = []
for pat in h4_targets:
    if pat in new_tpsl:
        direction = current_map[pat][0]
        new_tp, new_sl = new_tpsl[pat]
        old_tp, old_sl = current_map[pat][1], current_map[pat][2]
        if new_tp != old_tp or new_sl != old_sl:
            h4_map[pat] = (direction, new_tp, new_sl)
            h4_changes.append(pat)

# Strategy H5: Per-pattern optimal TP scale
# For each pattern, find the best TP scale (60%-100%) that maximizes PnL while keeping constraints
print(f"\n  Computing per-pattern optimal TP scale...")
h5_map = dict(current_map)
h5_changes = []
scales = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

for pat in current_map:
    direction, cur_tp, cur_sl = current_map[pat]
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 10:
        continue

    # The "base TP" before v1.27.0's 70% scaling
    # Current TP = base_tp * 0.7, so base_tp = current_tp / 0.7
    base_tp = cur_tp / 0.7

    best_scale = 0.70  # default to current
    best_metric = -999
    for scale in scales:
        test_tp = round(base_tp * scale, 2)
        pnls = bt_fixed(indices, direction, test_tp, cur_sl)
        if len(pnls) < 10:
            continue
        total_pnl = sum(pnls)
        if total_pnl <= 0:
            continue
        mc_quick = mc_test(pnls, 3000)
        if mc_quick >= 0.03:
            continue
        # Metric: PnL (simple, same as grid search objective)
        if total_pnl > best_metric:
            best_metric = total_pnl
            best_scale = scale

    if best_scale != 0.70:
        new_tp = round(base_tp * best_scale, 2)
        if new_tp != cur_tp:
            h5_map[pat] = (direction, new_tp, cur_sl)
            h5_changes.append((pat, best_scale))

# Strategy H6: Legacy patterns only + per-pattern TP scale
h6_map = dict(current_map)
h6_changes = []
for pat in legacy_pats:
    direction, cur_tp, cur_sl = current_map[pat]
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 10:
        continue
    base_tp = cur_tp / 0.7
    best_scale = 0.70
    best_metric = -999
    for scale in scales:
        test_tp = round(base_tp * scale, 2)
        pnls = bt_fixed(indices, direction, test_tp, cur_sl)
        if len(pnls) < 10:
            continue
        total_pnl = sum(pnls)
        if total_pnl <= 0:
            continue
        mc_quick = mc_test(pnls, 3000)
        if mc_quick >= 0.03:
            continue
        if total_pnl > best_metric:
            best_metric = total_pnl
            best_scale = scale
    if best_scale != 0.70:
        new_tp = round(base_tp * best_scale, 2)
        if new_tp != cur_tp:
            h6_map[pat] = (direction, new_tp, cur_sl)
            h6_changes.append((pat, best_scale))

# Evaluate all strategies
strategies = [
    ('H0: Current v1.27.0', h0_map, 0),
    ('H1: Legacy only (grid)', h1_map, len(h1_changes)),
    ('H2: MC>0.01 only (grid)', h2_map, len(h2_changes)),
    ('H3: Score>=3 only (grid)', h3_map, len(h3_changes)),
    ('H4: Legacy+MC fail (grid)', h4_map, len(h4_changes)),
    ('H5: Per-pat TP scale (all)', h5_map, len(h5_changes)),
    ('H6: Per-pat TP scale (legacy)', h6_map, len(h6_changes)),
]

print(f"\n  --- Portfolio Comparison ---\n")
print(f"  {'Strategy':<30} {'Chg':>4} {'Trades':>7} {'WR':>7} {'PnL':>10} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8} {'MCL':>4}")
print("  " + "-" * 90)

results = {}
for name, pmap, n_chg in strategies:
    trades = collect_trades_1pos(pmap)
    stats = eval_portfolio(trades)
    marker = " <<<" if name.startswith('H0') else ""
    print(f"  {name:<30} {n_chg:>4} {stats['trades']:>7} {stats['wr']:>6.1f}% "
          f"{stats['pnl']:>+9.1f}% {stats['mdd']:>6.1f}% {stats['pf']:>5.2f} "
          f"{stats['pnl_mdd']:>7.1f}x {stats['max_consec']:>3}{marker}")
    results[name] = {'stats': stats, 'trades': trades, 'n_changes': n_chg}


# ============================================================
# Phase 3: Deep analysis of best hybrid
# ============================================================
print(f"\n{'='*80}")
print("PHASE 3: Best Hybrid Deep Analysis")
print(f"{'='*80}")

# Find best by PnL/MDD among non-H0 strategies
best_strat = max(
    [(name, r) for name, r in results.items() if not name.startswith('H0')],
    key=lambda x: x[1]['stats']['pnl_mdd']
)
best_name = best_strat[0]
best_stats = best_strat[1]['stats']
best_trades = best_strat[1]['trades']

print(f"\n  Best hybrid by PnL/MDD: {best_name}")
print(f"    PnL/MDD: {best_stats['pnl_mdd']}x (vs H0: {results['H0: Current v1.27.0']['stats']['pnl_mdd']}x)")

# Also check if any strategy beats H0 on PnL/MDD
h0_pnl_mdd = results['H0: Current v1.27.0']['stats']['pnl_mdd']
better_strategies = [(n, r) for n, r in results.items()
                     if not n.startswith('H0') and r['stats']['pnl_mdd'] > h0_pnl_mdd]

if better_strategies:
    print(f"\n  Strategies that BEAT v1.27.0 on PnL/MDD:")
    for name, r in sorted(better_strategies, key=lambda x: -x[1]['stats']['pnl_mdd']):
        s = r['stats']
        delta = s['pnl_mdd'] - h0_pnl_mdd
        print(f"    {name}: {s['pnl_mdd']}x ({delta:+.1f}x vs H0)")
else:
    print(f"\n  No hybrid strategy beats v1.27.0 on PnL/MDD ({h0_pnl_mdd}x)")

# Portfolio MC for best hybrid
print(f"\n  Portfolio MC (Sign Randomization, {MC_SIMS} sims):")
h0_trades = results['H0: Current v1.27.0']['trades']
pnl_h0 = np.array([t[2] for t in h0_trades])
pnl_best = np.array([t[2] for t in best_trades])

signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_h0)))
mc_h0 = float(np.mean(np.sum(pnl_h0 * signs, axis=1) >= np.sum(pnl_h0)))
signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_best)))
mc_best = float(np.mean(np.sum(pnl_best * signs, axis=1) >= np.sum(pnl_best)))

print(f"    H0 (Current):    p={mc_h0:.4f} ({'PASS' if mc_h0 < 0.01 else 'FAIL'})")
print(f"    {best_name}: p={mc_best:.4f} ({'PASS' if mc_best < 0.01 else 'FAIL'})")

# MC MDD distribution
print(f"\n  MC MDD Distribution:")
mdds_h0 = portfolio_mc_mdd(h0_trades)
mdds_best = portfolio_mc_mdd(best_trades)

print(f"    {'':>12} {'H0 Current':>12} {best_name.split(':')[0]:>12}")
for pctl in [50, 90, 95, 99]:
    v0 = np.percentile(mdds_h0, pctl)
    vb = np.percentile(mdds_best, pctl)
    print(f"    P{pctl:<10} {v0:>11.1f}% {vb:>11.1f}%")


# ============================================================
# Phase 4: Per-pattern TP scale distribution
# ============================================================
print(f"\n{'='*80}")
print("PHASE 4: Per-Pattern Optimal TP Scale Distribution")
print(f"{'='*80}")

# For ALL 52 patterns, find optimal TP scale
print(f"\n  {'Pattern':<14} {'Dir':>5} {'Lineage':>8} {'Base TP':>7} {'Opt Scale':>9} {'Opt TP':>7} "
      f"{'Cur PnL':>8} {'Opt PnL':>8} {'Delta':>7}")
print("  " + "-" * 90)

scale_dist = {'v1.26.4': [], 'LEGACY': []}

for pat in sorted(current_map.keys()):
    direction, cur_tp, cur_sl = current_map[pat]
    lineage = 'v1.26.4' if pat in v1264_pats else 'LEGACY'
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 5:
        continue

    base_tp = cur_tp / 0.7

    # Current PnL
    cur_pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    cur_pnl = sum(cur_pnls)

    # Find best scale
    best_scale = 0.70
    best_pnl = cur_pnl
    for scale in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        test_tp = round(base_tp * scale, 2)
        pnls = bt_fixed(indices, direction, test_tp, cur_sl)
        if len(pnls) < 5:
            continue
        total = sum(pnls)
        if total > best_pnl:
            best_pnl = total
            best_scale = scale

    opt_tp = round(base_tp * best_scale, 2)
    delta = best_pnl - cur_pnl
    marker = '*' if lineage == 'LEGACY' else ' '
    scale_str = f"{best_scale*100:.0f}%"
    changed_marker = " <-" if best_scale != 0.70 else ""

    print(f"{marker} {pat:<13} {direction:>5} {lineage:>8} {base_tp:>7.2f} {scale_str:>9} "
          f"{opt_tp:>7.2f} {cur_pnl:>+7.1f}% {best_pnl:>+7.1f}% {delta:>+6.1f}%{changed_marker}")

    scale_dist[lineage].append({
        'pattern': pat, 'best_scale': best_scale, 'delta': delta,
    })

# Distribution summary
print(f"\n  Optimal TP Scale Distribution:")
for cat in ['v1.26.4', 'LEGACY']:
    scales_list = [s['best_scale'] for s in scale_dist[cat]]
    at_70 = sum(1 for s in scales_list if s == 0.70)
    below_70 = sum(1 for s in scales_list if s < 0.70)
    above_70 = sum(1 for s in scales_list if s > 0.70)
    avg_scale = np.mean(scales_list) * 100
    median_scale = np.median(scales_list) * 100
    print(f"    {cat:>8}: avg={avg_scale:.0f}%, median={median_scale:.0f}%, "
          f"at 70%: {at_70}, below: {below_70}, above: {above_70}")


# ============================================================
# Phase 5: Focused analysis — legacy patterns only
# ============================================================
print(f"\n{'='*80}")
print("PHASE 5: Legacy Pattern Deep Dive")
print(f"{'='*80}")

print(f"\n  20 legacy patterns from v1.26.1 T5 (R:R >= 0.75 objective):")
print(f"  These were optimized with DIFFERENT logic than v1.26.4 patterns.\n")

print(f"  {'Pattern':<14} {'Dir':>5} {'v1.27.0 TP/SL':>12} {'Trades':>6} {'WR':>6} {'PnL':>8} "
      f"{'MC':>7} {'Edge':>6} {'Reopt TP/SL':>12} {'Reopt PnL':>9} {'Verdict':<10}")
print("  " + "-" * 110)

legacy_verdicts = []
for pat in sorted(legacy_pats):
    h = pattern_health[pat]
    direction = h['direction']

    # Re-optimization result
    r = reopt_grid.get(pat, {})
    best = r.get('best')
    v = reopt_valid.get(pat, {})

    if best:
        new_tpsl_str = f"{best['tp']}/{best['sl']}"
        new_pnl = best['total_pnl']
        score = v.get('composite_score', 0)
    else:
        new_tpsl_str = "---"
        new_pnl = 0
        score = 0

    # Verdict for this pattern
    if h['mc'] >= 0.01:
        verdict = "NEEDS_FIX"
    elif best and new_pnl > h['pnl'] * 1.1 and score >= 3:
        verdict = "UPGRADE"
    elif best and new_pnl > h['pnl'] and score >= 2:
        verdict = "CONSIDER"
    else:
        verdict = "KEEP"

    legacy_verdicts.append({
        'pattern': pat, 'verdict': verdict,
        'current_pnl': h['pnl'], 'new_pnl': new_pnl if best else h['pnl'],
        'score': score,
    })

    cur_tpsl = f"{h['tp']}/{h['sl']}"
    print(f"  {pat:<14} {direction:>5} {cur_tpsl:>12} {h['trades']:>6} {h['wr']:>5.1f}% "
          f"{h['pnl']:>+7.1f}% {h['mc']:>7.4f} {h['edge']:>+5.1f}pp "
          f"{new_tpsl_str:>12} {new_pnl:>+8.1f}% {verdict:<10}")

n_needs_fix = sum(1 for v in legacy_verdicts if v['verdict'] == 'NEEDS_FIX')
n_upgrade = sum(1 for v in legacy_verdicts if v['verdict'] == 'UPGRADE')
n_consider = sum(1 for v in legacy_verdicts if v['verdict'] == 'CONSIDER')
n_keep = sum(1 for v in legacy_verdicts if v['verdict'] == 'KEEP')

print(f"\n  Legacy Verdict Summary:")
print(f"    NEEDS_FIX (MC>=0.01): {n_needs_fix}")
print(f"    UPGRADE (>10% PnL improvement, score>=3): {n_upgrade}")
print(f"    CONSIDER (improvement, score>=2): {n_consider}")
print(f"    KEEP (already good): {n_keep}")


# ============================================================
# Phase 6: Selective upgrade portfolio test
# ============================================================
print(f"\n{'='*80}")
print("PHASE 6: Selective Legacy Upgrade — Portfolio Impact")
print(f"{'='*80}")

# Build selective maps based on legacy verdicts
selective_fix = dict(current_map)  # Only NEEDS_FIX
selective_upgrade = dict(current_map)  # NEEDS_FIX + UPGRADE
selective_all = dict(current_map)  # NEEDS_FIX + UPGRADE + CONSIDER

fix_list = []
upgrade_list = []
consider_list = []

for v in legacy_verdicts:
    pat = v['pattern']
    if pat not in new_tpsl:
        continue
    direction = current_map[pat][0]
    new_tp, new_sl = new_tpsl[pat]

    if v['verdict'] == 'NEEDS_FIX':
        selective_fix[pat] = (direction, new_tp, new_sl)
        selective_upgrade[pat] = (direction, new_tp, new_sl)
        selective_all[pat] = (direction, new_tp, new_sl)
        fix_list.append(pat)
    elif v['verdict'] == 'UPGRADE':
        selective_upgrade[pat] = (direction, new_tp, new_sl)
        selective_all[pat] = (direction, new_tp, new_sl)
        upgrade_list.append(pat)
    elif v['verdict'] == 'CONSIDER':
        selective_all[pat] = (direction, new_tp, new_sl)
        consider_list.append(pat)

sel_strategies = [
    ('Current v1.27.0', current_map, 0),
    ('Fix MC>0.01 only', selective_fix, len(fix_list)),
    ('Fix + Upgrade', selective_upgrade, len(fix_list) + len(upgrade_list)),
    ('Fix + Upgrade + Consider', selective_all, len(fix_list) + len(upgrade_list) + len(consider_list)),
]

print(f"\n  {'Strategy':<28} {'Chg':>4} {'Trades':>7} {'WR':>7} {'PnL':>10} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8} {'MCL':>4}")
print("  " + "-" * 85)

for name, pmap, n_chg in sel_strategies:
    trades = collect_trades_1pos(pmap)
    stats = eval_portfolio(trades)
    marker = " <<<" if name.startswith('Current') else ""
    print(f"  {name:<28} {n_chg:>4} {stats['trades']:>7} {stats['wr']:>6.1f}% "
          f"{stats['pnl']:>+9.1f}% {stats['mdd']:>6.1f}% {stats['pf']:>5.2f} "
          f"{stats['pnl_mdd']:>7.1f}x {stats['max_consec']:>3}{marker}")

# Detail changes
if fix_list:
    print(f"\n  NEEDS_FIX patterns ({len(fix_list)}):")
    for pat in fix_list:
        d, old_tp, old_sl = current_map[pat]
        new_tp, new_sl = new_tpsl[pat]
        print(f"    {pat} ({d}): {old_tp}/{old_sl} -> {new_tp}/{new_sl}")

if upgrade_list:
    print(f"\n  UPGRADE patterns ({len(upgrade_list)}):")
    for pat in upgrade_list:
        d, old_tp, old_sl = current_map[pat]
        new_tp, new_sl = new_tpsl[pat]
        print(f"    {pat} ({d}): {old_tp}/{old_sl} -> {new_tp}/{new_sl}")

if consider_list:
    print(f"\n  CONSIDER patterns ({len(consider_list)}):")
    for pat in consider_list:
        d, old_tp, old_sl = current_map[pat]
        new_tp, new_sl = new_tpsl[pat]
        print(f"    {pat} ({d}): {old_tp}/{old_sl} -> {new_tp}/{new_sl}")


# ============================================================
# Save results
# ============================================================
output = {
    'meta': {
        'script': 'tp_sl_lineage_analysis.py',
        'timestamp': datetime.now().isoformat(),
    },
    'lineage': {
        'v1264_optimized': sorted(v1264_pats),
        'legacy_v1261': sorted(legacy_pats),
    },
    'pattern_health': pattern_health,
    'hybrid_comparison': {
        name.split(':')[0].strip(): {
            'changes': n_chg,
            'stats': results[name]['stats'] if name in results else eval_portfolio(collect_trades_1pos(pmap)),
        }
        for name, pmap, n_chg in strategies
    },
    'legacy_verdicts': legacy_verdicts,
    'scale_distribution': scale_dist,
    'selective_upgrade': {
        'fix_list': fix_list,
        'upgrade_list': upgrade_list,
        'consider_list': consider_list,
    },
}

out_path = Path(__file__).resolve().parent / '../../results/tp_sl_lineage_analysis.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done.")
