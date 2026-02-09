"""
v1.26.4 TP/SL Optimization Research
=====================================
52개 전체 패턴에 대해 MC<0.01 + WF>=4 제약 하에서 최적 TP/SL 탐색.
현재 v1.26.3 설정 대비 개선 가능한 패턴을 찾고 포트폴리오 영향 분석.

Objective: maximize total PnL subject to:
  - MC < 0.01 (statistical significance)
  - WF >= 4/5 (temporal stability)
  - PnL > 0 (profitable)
  - Period >= 2/3 (multi-period stability)
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
    PATTERN_OPTIMAL_TPSL, PATTERN_STATS,
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

# Constraints
MC_THRESHOLD = 0.01
WF_THRESHOLD = 4
PERIOD_THRESHOLD = 2
MIN_TRADES = 10  # relaxed for small patterns

print("=" * 80)
print("v1.26.4 TP/SL Optimization Research")
print("  Constraint: MC < 0.01, WF >= 4/5, Period >= 2/3, PnL > 0")
print("  Grid: TP/SL in", TP_SL_GRID)
print("=" * 80)

# ============================================================
# Load and classify data
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
dates = df['timestamp'].values

period_masks = {
    'P1': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'P2': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'P3': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

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

print("Classification complete.")


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


def walk_forward_test(indices, direction, tp_pct, sl_pct, n_folds=5):
    si = np.sort(indices)
    fs = len(si) // n_folds
    if fs < 3:
        return 0
    profitable = 0
    for f in range(n_folds):
        s = f * fs
        e = s + fs if f < n_folds - 1 else len(si)
        pnls = bt_fixed(si[s:e], direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


def period_test(indices, direction, tp_pct, sl_pct):
    profitable = 0
    for mask in period_masks.values():
        pidx = indices[mask[indices]]
        if len(pidx) < 3:
            continue
        pnls = bt_fixed(pidx, direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


def eval_combo(indices, direction, tp, sl):
    """Full evaluation of a TP/SL combo. Returns dict or None if filtered."""
    pnls = bt_fixed(indices, direction, tp, sl)
    n = len(pnls)
    if n < MIN_TRADES:
        return None

    total_pnl = sum(pnls)
    if total_pnl <= 0:
        return None

    wr = sum(1 for x in pnls if x > 0) / n * 100
    avg_w = np.mean([x for x in pnls if x > 0]) if any(x > 0 for x in pnls) else 0
    avg_l = np.mean([abs(x) for x in pnls if x <= 0]) if any(x <= 0 for x in pnls) else 0
    exp = (wr / 100) * avg_w - (1 - wr / 100) * avg_l

    # Quick MC screen (3000 sims)
    mc_quick = mc_test(pnls, 3000)
    if mc_quick >= 0.03:  # liberal quick screen
        return None

    wf = walk_forward_test(indices, direction, tp, sl)
    if wf < WF_THRESHOLD:
        return None

    pp = period_test(indices, direction, tp, sl)
    if pp < PERIOD_THRESHOLD:
        return None

    # Full MC (10000 sims)
    mc = mc_test(pnls, MC_SIMS)
    if mc >= MC_THRESHOLD:
        return None

    rr = tp / sl if sl > 0 else 999
    return {
        'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'trades': n, 'wr': round(wr, 1), 'total_pnl': round(total_pnl, 1),
        'avg_win': round(avg_w, 2), 'avg_loss': round(avg_l, 2),
        'exp': round(exp, 3), 'mc': round(mc, 4), 'wf': wf, 'pp': pp,
    }


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
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'exp': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wps = []; lps = []
    for p in pnl_list:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0: wins += 1; wps.append(p)
        else: lps.append(abs(p))
    aw = np.mean(wps) if wps else 0
    al = np.mean(lps) if lps else 0
    tw = sum(wps); tl = sum(lps)
    wr = wins / len(pnl_list) * 100
    exp = (wr / 100) * aw - (1 - wr / 100) * al
    return {
        'pnl': cum, 'trades': len(pnl_list), 'wr': wr, 'mdd': mdd,
        'pf': tw / tl if tl > 0 else 999, 'exp': exp,
        'pnl_mdd': cum / mdd if mdd > 0 else 999,
    }


# ============================================================
# Build current v1.26.3 map
# ============================================================
current_map = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('LONG', tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('SHORT', tp, sl)

print(f"\nCurrent v1.26.3: {len(current_map)} patterns")


# ============================================================
# Phase 1: Per-pattern grid search
# ============================================================
print(f"\n{'='*80}")
print("Phase 1: Per-Pattern TP/SL Grid Search (MC<0.01, WF>=4, Period>=2)")
print(f"{'='*80}")

t0 = time.time()
optimization_results = {}

for pat, (direction, cur_tp, cur_sl) in sorted(current_map.items()):
    indices = pattern_indices.get(pat, np.array([]))

    # Current performance (no constraints)
    cur_pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    cur_n = len(cur_pnls)
    cur_pnl = sum(cur_pnls) if cur_pnls else 0
    cur_wr = sum(1 for x in cur_pnls if x > 0) / cur_n * 100 if cur_n > 0 else 0
    cur_mc = mc_test(cur_pnls) if cur_n >= 5 else 1.0
    cur_wf = walk_forward_test(indices, direction, cur_tp, cur_sl)
    cur_pp = period_test(indices, direction, cur_tp, cur_sl)
    cur_rr = cur_tp / cur_sl if cur_sl > 0 else 999
    cur_aw = np.mean([x for x in cur_pnls if x > 0]) if any(x > 0 for x in cur_pnls) else 0
    cur_al = np.mean([abs(x) for x in cur_pnls if x <= 0]) if any(x <= 0 for x in cur_pnls) else 0
    cur_exp = (cur_wr / 100) * cur_aw - (1 - cur_wr / 100) * cur_al

    cur_passes = cur_mc < MC_THRESHOLD and cur_wf >= WF_THRESHOLD and cur_pnl > 0

    # Grid search for alternatives
    valid_combos = []
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            result = eval_combo(indices, direction, tp, sl)
            if result:
                valid_combos.append(result)

    # Best by total PnL
    best = max(valid_combos, key=lambda x: x['total_pnl']) if valid_combos else None

    # Is best different from current?
    changed = False
    if best and (best['tp'] != cur_tp or best['sl'] != cur_sl):
        changed = True

    # Is best better than current?
    improved = False
    if best and changed and best['total_pnl'] > cur_pnl:
        improved = True

    optimization_results[pat] = {
        'direction': direction,
        'current': {
            'tp': cur_tp, 'sl': cur_sl, 'rr': round(cur_rr, 2),
            'trades': cur_n, 'wr': round(cur_wr, 1), 'total_pnl': round(cur_pnl, 1),
            'exp': round(cur_exp, 3), 'mc': round(cur_mc, 4), 'wf': cur_wf, 'pp': cur_pp,
            'passes': cur_passes,
        },
        'best': best,
        'n_valid': len(valid_combos),
        'changed': changed,
        'improved': improved,
    }

elapsed = time.time() - t0
print(f"  Grid search complete in {elapsed:.1f}s")


# ============================================================
# Phase 2: Results table
# ============================================================
print(f"\n{'='*80}")
print("Phase 2: Optimization Results")
print(f"{'='*80}\n")

# Count categories
n_already_optimal = 0
n_improved = 0
n_cur_fails = 0
n_no_valid = 0

for pat, r in optimization_results.items():
    cur = r['current']
    best = r['best']
    if not cur['passes']:
        n_cur_fails += 1
    if best is None:
        n_no_valid += 1
    elif not r['changed']:
        n_already_optimal += 1
    elif r['improved']:
        n_improved += 1

print(f"  Current passes constraints: {sum(1 for r in optimization_results.values() if r['current']['passes'])}/52")
print(f"  Current fails constraints: {n_cur_fails}/52")
print(f"  Already optimal (current = best): {n_already_optimal}")
print(f"  Better alternative found: {n_improved}")
print(f"  No valid combo exists: {n_no_valid}")

# Show patterns where improvement found
print(f"\n--- Patterns with Better TP/SL Available ---")
print(f"{'Pattern':<12} {'Dir':>5} | {'CUR TP/SL':>9} {'PnL':>8} {'MC':>7} {'WF':>4} | {'OPT TP/SL':>9} {'PnL':>8} {'MC':>7} {'WF':>4} | {'PnL Diff':>9}")
print("-" * 100)

improved_list = []
for pat in sorted(optimization_results.keys()):
    r = optimization_results[pat]
    if not r['improved']:
        continue
    cur = r['current']
    best = r['best']
    pnl_diff = best['total_pnl'] - cur['total_pnl']
    cur_tpsl = f"{cur['tp']}/{cur['sl']}"
    best_tpsl = f"{best['tp']}/{best['sl']}"
    mc_cur_str = f"{'PASS' if cur['mc'] < MC_THRESHOLD else 'FAIL'}"
    mc_best_str = "PASS"  # guaranteed by eval_combo
    print(f"{pat:<12} {r['direction']:>5} | {cur_tpsl:>9} {cur['total_pnl']:>+7.1f}% {cur['mc']:>7.4f} {cur['wf']:>2}/5 | {best_tpsl:>9} {best['total_pnl']:>+7.1f}% {best['mc']:>7.4f} {best['wf']:>2}/5 | {pnl_diff:>+8.1f}%")
    improved_list.append((pat, r))

# Show patterns where current fails and best exists
print(f"\n--- Currently Failing Patterns (MC>=0.01 or WF<4) with Valid Alternative ---")
print(f"{'Pattern':<12} {'Dir':>5} | {'CUR TP/SL':>9} {'PnL':>8} {'MC':>7} {'WF':>4} | {'FIX TP/SL':>9} {'PnL':>8} {'MC':>7} {'WF':>4} | {'Note':>12}")
print("-" * 105)

fixed_list = []
for pat in sorted(optimization_results.keys()):
    r = optimization_results[pat]
    cur = r['current']
    if cur['passes']:
        continue
    best = r['best']
    cur_tpsl = f"{cur['tp']}/{cur['sl']}"
    if best:
        best_tpsl = f"{best['tp']}/{best['sl']}"
        note = "FIXABLE"
        print(f"{pat:<12} {r['direction']:>5} | {cur_tpsl:>9} {cur['total_pnl']:>+7.1f}% {cur['mc']:>7.4f} {cur['wf']:>2}/5 | {best_tpsl:>9} {best['total_pnl']:>+7.1f}% {best['mc']:>7.4f} {best['wf']:>2}/5 | {note:>12}")
        fixed_list.append((pat, r))
    else:
        print(f"{pat:<12} {r['direction']:>5} | {cur_tpsl:>9} {cur['total_pnl']:>+7.1f}% {cur['mc']:>7.4f} {cur['wf']:>2}/5 | {'---':>9} {'---':>8} {'---':>7} {'---':>4} | {'NO FIX':>12}")

# Show patterns where current is already optimal
print(f"\n--- Already Optimal ({n_already_optimal} patterns) ---")
print(f"  (current TP/SL is the best among all valid combos)")
already_opt = []
for pat in sorted(optimization_results.keys()):
    r = optimization_results[pat]
    if r['best'] and not r['changed']:
        already_opt.append(pat)
for i in range(0, len(already_opt), 8):
    print(f"  {', '.join(already_opt[i:i+8])}")


# ============================================================
# Phase 3: Portfolio Impact
# ============================================================
print(f"\n{'='*80}")
print("Phase 3: Portfolio Impact Simulation")
print(f"{'='*80}\n")

# Build optimized map
opt_map = {}
changes_applied = []
for pat, (direction, cur_tp, cur_sl) in current_map.items():
    r = optimization_results[pat]
    best = r['best']
    if best and (r['improved'] or not r['current']['passes']):
        # Use best if it improves PnL or fixes a failing pattern
        if best['total_pnl'] > 0:
            opt_map[pat] = (direction, best['tp'], best['sl'])
            if best['tp'] != cur_tp or best['sl'] != cur_sl:
                changes_applied.append((pat, direction, cur_tp, cur_sl, best['tp'], best['sl']))
        else:
            opt_map[pat] = (direction, cur_tp, cur_sl)
    else:
        opt_map[pat] = (direction, cur_tp, cur_sl)

print(f"Changes applied: {len(changes_applied)}")
for pat, d, old_tp, old_sl, new_tp, new_sl in changes_applied:
    print(f"  {pat} ({d}): {old_tp}/{old_sl} -> {new_tp}/{new_sl}")

# Evaluate both portfolios
t_cur = collect_trades_1pos(current_map)
t_opt = collect_trades_1pos(opt_map)
r_cur = eval_portfolio(t_cur)
r_opt = eval_portfolio(t_opt)

print(f"\n  {'':>16} {'v1.26.3 (cur)':>14} {'Optimized':>14} {'Delta':>10}")
print(f"  {'-'*58}")
for k, label in [('trades', 'Trades'), ('wr', 'WR (%)'), ('pnl', 'PnL (%)'),
                  ('mdd', 'MDD (%)'), ('pf', 'PF'), ('exp', 'Exp/trade (%)'),
                  ('pnl_mdd', 'PnL/MDD')]:
    o = r_cur[k]; n = r_opt[k]; d = n - o
    if k in ('wr', 'pnl', 'mdd', 'exp'):
        print(f"  {label:>16} {o:>13.1f}% {n:>13.1f}% {d:>+9.1f}%")
    elif k in ('pf', 'pnl_mdd'):
        u = 'x' if k == 'pnl_mdd' else ''
        print(f"  {label:>16} {o:>13.2f}{u} {n:>13.2f}{u} {d:>+9.2f}{u}")
    else:
        print(f"  {label:>16} {o:>14} {n:>14} {d:>+10}")

# Portfolio WF for both
print(f"\n  Portfolio WF:")
fold_size = n_bars // 5
for label, pmap in [('v1.26.3', current_map), ('Optimized', opt_map)]:
    wf_cnt = 0
    print(f"  {label:>12}: ", end='')
    for f in range(5):
        s = f * fold_size; e = s + fold_size if f < 4 else n_bars
        ft = []; last_exit = -1
        for i in range(max(s, 2), e):
            pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
            if pat not in pmap: continue
            dr, tp, sl = pmap[pat]
            if i + 1 >= n_bars: continue
            eb = i + 1
            if eb <= last_exit: continue
            entry = opens[eb]
            if entry <= 0: continue
            if dr == 'LONG':
                tpp = entry * (1 + tp / 100); slp = entry * (1 - sl / 100)
            else:
                tpp = entry * (1 - tp / 100); slp = entry * (1 + sl / 100)
            for j in range(eb + 1, min(eb + MAX_BARS, n_bars)):
                ht = (highs[j] >= tpp) if dr == 'LONG' else (lows[j] <= tpp)
                hs = (lows[j] <= slp) if dr == 'LONG' else (highs[j] >= slp)
                if ht and hs:
                    pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                    ft.append((eb, j, pnl)); last_exit = j; break
                elif ht: ft.append((eb, j, tp * LEVERAGE - FEE_PCT)); last_exit = j; break
                elif hs: ft.append((eb, j, -sl * LEVERAGE - FEE_PCT)); last_exit = j; break
        fpnl = sum(t[2] for t in ft)
        ok = fpnl > 0
        if ok: wf_cnt += 1
        print(f"F{f+1}:{fpnl:+.1f}%({'OK' if ok else 'X'}) ", end='')
    print(f"  => {wf_cnt}/5")

# WR resilience
for label, stats in [('v1.26.3', r_cur), ('Optimized', r_opt)]:
    aw = stats.get('avg_win', 0)
    al = stats.get('avg_loss', 0)
    if aw == 0 and al == 0:
        continue
    # Recalculate from trades
    if label == 'v1.26.3':
        plist = [t[2] for t in t_cur]
    else:
        plist = [t[2] for t in t_opt]
    aw2 = np.mean([p for p in plist if p > 0]) if any(p > 0 for p in plist) else 0
    al2 = np.mean([abs(p) for p in plist if p <= 0]) if any(p <= 0 for p in plist) else 0
    be_wr = al2 / (aw2 + al2) * 100 if (aw2 + al2) > 0 else 0
    max_drop = stats['wr'] - be_wr
    print(f"  {label:>12} WR Resilience: WR {stats['wr']:.1f}% -> BE {be_wr:.1f}% -> drop tolerance {max_drop:+.1f}pp")


# ============================================================
# Phase 4: Recommended Changes
# ============================================================
print(f"\n{'='*80}")
print("Phase 4: Recommended Changes for v1.26.4")
print(f"{'='*80}\n")

if changes_applied:
    print(f"PATTERN_OPTIMAL_TPSL changes ({len(changes_applied)}):")
    print(f"{'Pattern':<12} {'Dir':>5} {'Old TP/SL':>9} {'New TP/SL':>9} {'Old PnL':>8} {'New PnL':>8} {'Old MC':>7} {'New MC':>7}")
    print("-" * 75)
    for pat, d, old_tp, old_sl, new_tp, new_sl in changes_applied:
        r = optimization_results[pat]
        old_pnl = r['current']['total_pnl']
        old_mc = r['current']['mc']
        new_pnl = r['best']['total_pnl']
        new_mc = r['best']['mc']
        print(f"{pat:<12} {d:>5} {old_tp}/{old_sl:>5} {new_tp}/{new_sl:>5} {old_pnl:>+7.1f}% {new_pnl:>+7.1f}% {old_mc:>7.4f} {new_mc:>7.4f}")
else:
    print("No changes recommended — current settings are already optimal within constraints.")

# ============================================================
# Save results
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v1.26.4_research',
    'constraints': {
        'mc_threshold': MC_THRESHOLD,
        'wf_threshold': WF_THRESHOLD,
        'period_threshold': PERIOD_THRESHOLD,
        'min_trades': MIN_TRADES,
        'tp_sl_grid': TP_SL_GRID,
    },
    'summary': {
        'total_patterns': len(optimization_results),
        'current_passes': sum(1 for r in optimization_results.values() if r['current']['passes']),
        'already_optimal': n_already_optimal,
        'improved': n_improved,
        'no_valid': n_no_valid,
        'changes_applied': len(changes_applied),
    },
    'changes': [
        {'pattern': pat, 'direction': d,
         'old_tp': otp, 'old_sl': osl, 'new_tp': ntp, 'new_sl': nsl}
        for pat, d, otp, osl, ntp, nsl in changes_applied
    ],
    'portfolio_comparison': {
        'current': {k: round(v, 3) if isinstance(v, float) else v for k, v in r_cur.items()},
        'optimized': {k: round(v, 3) if isinstance(v, float) else v for k, v in r_opt.items()},
    },
    'per_pattern': {pat: r for pat, r in optimization_results.items()},
}

out_path = Path(__file__).resolve().parent / '../../results/tp_sl_optimization_v1264.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done.")
