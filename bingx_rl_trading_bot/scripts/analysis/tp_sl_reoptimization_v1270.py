"""
v1.27.0 TP/SL Re-optimization
===============================
52개 전체 패턴에 대해 처음부터 TP/SL 재최적화.
v1.27.0 Uniform TP 70% 이후 패턴별 개별 최적값 탐색.

Phase 1: Grid Search (MC<0.01, WF>=4/5, Period>=2/3)
Phase 2: Deep Validation (CV, Plateau, Edge, OOS, Composite>=2)
Phase 3: Portfolio Validation (MC MDD, WF, Robustness, Consecutive Loss)
Phase 4: Comparison vs v1.27.0 Current Values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL, VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
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
MIN_TRADES = 10
EDGE_THRESHOLD = 5  # pp excess WR over random walk baseline
COMPOSITE_THRESHOLD = 2  # minimum score out of 4

print("=" * 80)
print("v1.27.0 TP/SL Re-optimization — Fresh Grid Search + Deep Validation")
print("  Constraints: MC<0.01, WF>=4/5, Period>=2/3, Edge>5pp, Composite>=2")
print(f"  Grid: TP/SL in {TP_SL_GRID}")
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
    """Full evaluation of a TP/SL combo with all constraints."""
    pnls = bt_fixed(indices, direction, tp, sl)
    n = len(pnls)
    if n < MIN_TRADES:
        return None
    total_pnl = sum(pnls)
    if total_pnl <= 0:
        return None
    wr = sum(1 for x in pnls if x > 0) / n * 100
    # Quick MC screen
    mc_quick = mc_test(pnls, 3000)
    if mc_quick >= 0.03:
        return None
    wf = walk_forward_test(indices, direction, tp, sl)
    if wf < WF_THRESHOLD:
        return None
    pp = period_test(indices, direction, tp, sl)
    if pp < PERIOD_THRESHOLD:
        return None
    # Full MC
    mc = mc_test(pnls, MC_SIMS)
    if mc >= MC_THRESHOLD:
        return None
    rr = tp / sl if sl > 0 else 999
    avg_w = np.mean([x for x in pnls if x > 0]) if any(x > 0 for x in pnls) else 0
    avg_l = np.mean([abs(x) for x in pnls if x <= 0]) if any(x <= 0 for x in pnls) else 0
    return {
        'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'trades': n, 'wr': round(wr, 1), 'total_pnl': round(total_pnl, 1),
        'avg_win': round(avg_w, 2), 'avg_loss': round(avg_l, 2),
        'mc': round(mc, 4), 'wf': wf, 'pp': pp,
    }


def find_best_tpsl(indices, direction, min_trades=5):
    """Find best TP/SL by total PnL with MC<0.01 constraint."""
    best = None
    best_pnl = -999
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            pnls = bt_fixed(indices, direction, tp, sl)
            if len(pnls) < min_trades:
                continue
            total = sum(pnls)
            if total <= 0:
                continue
            mc_quick = mc_test(pnls, 3000)
            if mc_quick >= 0.03:
                continue
            mc_full = mc_test(pnls, MC_SIMS)
            if mc_full >= MC_THRESHOLD:
                continue
            if total > best_pnl:
                best_pnl = total
                best = (tp, sl, total, len(pnls), mc_full)
    return best


def collect_trades_1pos(pattern_map, start_bar=0, end_bar=None):
    if end_bar is None:
        end_bar = n_bars
    raw_trades = []
    for i in range(max(2, start_bar), end_bar):
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
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'pnl_mdd': 0, 'max_consec_loss': 0}
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
    aw = np.mean(wps) if wps else 0
    al = np.mean(lps) if lps else 0
    tw = sum(wps); tl = sum(lps)
    wr = wins / len(pnl_list) * 100
    return {
        'pnl': cum, 'trades': len(pnl_list), 'wr': wr, 'mdd': mdd,
        'pf': tw / tl if tl > 0 else 999,
        'avg_win': aw, 'avg_loss': al,
        'pnl_mdd': cum / mdd if mdd > 0 else 999,
        'max_consec_loss': max_streak,
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


def analyze_consecutive_losses(trades):
    pnl_list = [t[2] for t in trades]
    streaks = []
    current_streak = 0
    current_loss_sum = 0.0
    for p in pnl_list:
        if p < 0:
            current_streak += 1
            current_loss_sum += p
        else:
            if current_streak > 0:
                streaks.append((current_streak, current_loss_sum))
            current_streak = 0
            current_loss_sum = 0.0
    if current_streak > 0:
        streaks.append((current_streak, current_loss_sum))
    max_streak = max((s[0] for s in streaks), default=0)
    max_loss_sum = min((s[1] for s in streaks), default=0)
    streak_counts = defaultdict(int)
    for s in streaks:
        streak_counts[s[0]] += 1
    return {
        'max_streak': max_streak,
        'max_loss_sum': max_loss_sum,
        'streak_dist': dict(streak_counts),
    }


# ============================================================
# Build current v1.27.0 map
# ============================================================
current_map = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('LONG', tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('SHORT', tp, sl)

print(f"\nCurrent v1.27.0: {len(current_map)} patterns "
      f"({len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S)")


# ============================================================
# Phase 1: Per-pattern Grid Search
# ============================================================
print(f"\n{'='*80}")
print("PHASE 1: Per-Pattern TP/SL Grid Search")
print(f"  Grid: {len(TP_SL_GRID)}x{len(TP_SL_GRID)} = {len(TP_SL_GRID)**2} combos per pattern")
print(f"  Constraints: MC<{MC_THRESHOLD}, WF>={WF_THRESHOLD}/5, Period>={PERIOD_THRESHOLD}/3")
print(f"{'='*80}")

t0 = time.time()
grid_results = {}

for pat, (direction, cur_tp, cur_sl) in sorted(current_map.items()):
    indices = pattern_indices.get(pat, np.array([]))

    # Current performance
    cur_pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    cur_n = len(cur_pnls)
    cur_pnl = sum(cur_pnls) if cur_pnls else 0
    cur_wr = sum(1 for x in cur_pnls if x > 0) / cur_n * 100 if cur_n > 0 else 0
    cur_mc = mc_test(cur_pnls) if cur_n >= 5 else 1.0
    cur_wf = walk_forward_test(indices, direction, cur_tp, cur_sl)

    # Grid search
    valid_combos = []
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            result = eval_combo(indices, direction, tp, sl)
            if result:
                valid_combos.append(result)

    best = max(valid_combos, key=lambda x: x['total_pnl']) if valid_combos else None
    changed = best is not None and (best['tp'] != cur_tp or best['sl'] != cur_sl)
    improved = changed and best['total_pnl'] > cur_pnl

    grid_results[pat] = {
        'direction': direction,
        'current': {
            'tp': cur_tp, 'sl': cur_sl,
            'trades': cur_n, 'wr': round(cur_wr, 1),
            'total_pnl': round(cur_pnl, 1), 'mc': round(cur_mc, 4), 'wf': cur_wf,
        },
        'best': best,
        'n_valid': len(valid_combos),
        'changed': changed,
        'improved': improved,
    }

elapsed = time.time() - t0
print(f"\n  Grid search complete in {elapsed:.1f}s")

# Summary
n_improved = sum(1 for r in grid_results.values() if r['improved'])
n_same = sum(1 for r in grid_results.values() if r['best'] and not r['changed'])
n_no_valid = sum(1 for r in grid_results.values() if r['best'] is None)
n_changed_worse = sum(1 for r in grid_results.values() if r['changed'] and not r['improved'])

print(f"\n  Results:")
print(f"    Better alternative found: {n_improved}")
print(f"    Current is already best:  {n_same}")
print(f"    Changed but worse:        {n_changed_worse}")
print(f"    No valid combo exists:    {n_no_valid}")

# Show improvements
print(f"\n  --- Patterns with Better TP/SL ---")
print(f"  {'Pattern':<14} {'Dir':>5} | {'CUR TP/SL':>9} {'PnL':>8} {'MC':>7} | {'NEW TP/SL':>9} {'PnL':>8} {'MC':>7} | {'Diff':>8}")
print("  " + "-" * 90)

for pat in sorted(grid_results.keys()):
    r = grid_results[pat]
    if not r['improved']:
        continue
    cur = r['current']
    best = r['best']
    diff = best['total_pnl'] - cur['total_pnl']
    print(f"  {pat:<14} {r['direction']:>5} | {cur['tp']}/{cur['sl']:>5} {cur['total_pnl']:>+7.1f}% {cur['mc']:>7.4f} "
          f"| {best['tp']}/{best['sl']:>5} {best['total_pnl']:>+7.1f}% {best['mc']:>7.4f} | {diff:>+7.1f}%")


# ============================================================
# Phase 2: Deep Validation
# ============================================================
print(f"\n{'='*80}")
print("PHASE 2: Deep Validation (CV + Plateau + Edge + OOS)")
print(f"{'='*80}")

t0 = time.time()
n_folds = 5
validation_results = {}

# Only validate patterns where grid found a candidate (best exists)
candidates = {pat: r for pat, r in grid_results.items() if r['best'] is not None}
print(f"\n  Validating {len(candidates)} candidates...")

for pat, r in sorted(candidates.items()):
    direction = r['direction']
    best = r['best']
    new_tp, new_sl = best['tp'], best['sl']
    indices = pattern_indices.get(pat, np.array([]))
    si = np.sort(indices)

    # --- 2A: Leave-One-Fold-Out CV ---
    fs = len(si) // n_folds
    cv_stable = False
    cv_oos_positive = 0
    cv_consensus = None

    if fs >= 3:
        selections = []
        oos_pnls = []
        for held_out in range(n_folds):
            train_idx = []
            test_idx = []
            for f in range(n_folds):
                s = f * fs
                e = s + fs if f < n_folds - 1 else len(si)
                if f == held_out:
                    test_idx = si[s:e]
                else:
                    train_idx.extend(si[s:e])
            train_idx = np.array(train_idx)

            fold_best = find_best_tpsl(train_idx, direction, min_trades=3)
            if fold_best:
                tp_f, sl_f = fold_best[0], fold_best[1]
                selections.append((tp_f, sl_f))
                test_pnls = bt_fixed(test_idx, direction, tp_f, sl_f)
                oos_pnls.append(sum(test_pnls) if test_pnls else 0)
            else:
                selections.append(None)
                oos_pnls.append(0)

        valid_sels = [s for s in selections if s is not None]
        if valid_sels:
            most_common = Counter(valid_sels).most_common(1)[0]
            cv_consensus = most_common[0]
            cv_stable = most_common[1] >= 3
        cv_oos_positive = sum(1 for p in oos_pnls if p > 0)

    # --- 2B: Neighborhood Stability ---
    tp_idx = TP_SL_GRID.index(new_tp) if new_tp in TP_SL_GRID else -1
    sl_idx = TP_SL_GRID.index(new_sl) if new_sl in TP_SL_GRID else -1
    plateau = False
    n_profitable_nb = 0
    n_total_nb = 0

    if tp_idx >= 0 and sl_idx >= 0:
        for dtp in [-1, 0, 1]:
            for dsl in [-1, 0, 1]:
                ti = tp_idx + dtp
                si_n = sl_idx + dsl
                if 0 <= ti < len(TP_SL_GRID) and 0 <= si_n < len(TP_SL_GRID):
                    ntp = TP_SL_GRID[ti]
                    nsl = TP_SL_GRID[si_n]
                    nb_pnls = bt_fixed(indices, direction, ntp, nsl)
                    n_total_nb += 1
                    if len(nb_pnls) >= 5 and sum(nb_pnls) > 0:
                        n_profitable_nb += 1
        plateau = n_total_nb > 0 and n_profitable_nb >= n_total_nb * 0.6

    # --- 2C: Edge Test ---
    rw_wr = new_sl / (new_tp + new_sl) * 100  # random walk baseline
    pnls_new = bt_fixed(indices, direction, new_tp, new_sl)
    act_wr = sum(1 for p in pnls_new if p > 0) / len(pnls_new) * 100 if pnls_new else 0
    excess_wr = act_wr - rw_wr
    has_edge = excess_wr >= EDGE_THRESHOLD

    # --- 2D: Composite Score ---
    score = 0
    if cv_stable: score += 1
    if plateau: score += 1
    if has_edge: score += 1
    if cv_oos_positive >= 3: score += 1

    verdict = "ACCEPT" if score >= COMPOSITE_THRESHOLD else "REJECT"

    validation_results[pat] = {
        'tp': new_tp, 'sl': new_sl,
        'cv_stable': cv_stable,
        'cv_consensus': cv_consensus,
        'cv_oos_positive': cv_oos_positive,
        'plateau': plateau,
        'n_profitable_nb': n_profitable_nb,
        'n_total_nb': n_total_nb,
        'has_edge': has_edge,
        'rw_wr': round(rw_wr, 1),
        'act_wr': round(act_wr, 1),
        'excess_wr': round(excess_wr, 1),
        'composite_score': score,
        'verdict': verdict,
    }

elapsed = time.time() - t0
print(f"  Validation complete in {elapsed:.1f}s\n")

# Display results
print(f"  {'Pattern':<14} {'TP/SL':>7} {'CV':>4} {'Plateau':>8} {'Edge':>6} {'OOS':>5} {'Score':>6} {'Verdict':>8}")
print("  " + "-" * 65)

accepted = []
rejected = []
for pat in sorted(validation_results.keys()):
    v = validation_results[pat]
    cv_str = "Y" if v['cv_stable'] else "N"
    pl_str = "Y" if v['plateau'] else "N"
    eg_str = f"+{v['excess_wr']:.0f}" if v['has_edge'] else f"{v['excess_wr']:.0f}"
    oos_str = f"{v['cv_oos_positive']}/5"
    tpsl = f"{v['tp']}/{v['sl']}"
    print(f"  {pat:<14} {tpsl:>7}  {cv_str:>3}  {pl_str:>7}  {eg_str:>5}  {oos_str:>4}  {v['composite_score']:>4}/4  {v['verdict']:>7}")
    if v['verdict'] == 'ACCEPT':
        accepted.append(pat)
    else:
        rejected.append(pat)

print(f"\n  ACCEPT: {len(accepted)}, REJECT: {len(rejected)}")


# ============================================================
# Phase 3: Build Final Maps & Portfolio Validation
# ============================================================
print(f"\n{'='*80}")
print("PHASE 3: Portfolio Validation")
print(f"{'='*80}")

# Build optimized map: use new TP/SL only for patterns that:
# 1. Grid search found better alternative (improved=True OR best exists and passes)
# 2. Deep validation passed (composite >= 2)
optimized_map = {}
changes = []

for pat, (direction, cur_tp, cur_sl) in current_map.items():
    r = grid_results[pat]
    best = r['best']

    # Use optimized value only if validated
    use_new = False
    if best is not None and pat in validation_results:
        v = validation_results[pat]
        if v['verdict'] == 'ACCEPT':
            new_tp, new_sl = best['tp'], best['sl']
            if new_tp != cur_tp or new_sl != cur_sl:
                use_new = True

    if use_new:
        optimized_map[pat] = (direction, new_tp, new_sl)
        changes.append({
            'pattern': pat, 'direction': direction,
            'old_tp': cur_tp, 'old_sl': cur_sl,
            'new_tp': new_tp, 'new_sl': new_sl,
            'old_pnl': r['current']['total_pnl'],
            'new_pnl': best['total_pnl'],
            'score': validation_results[pat]['composite_score'],
        })
    else:
        optimized_map[pat] = (direction, cur_tp, cur_sl)

print(f"\n  Patterns changed: {len(changes)}")
if changes:
    print(f"\n  {'Pattern':<14} {'Dir':>5} {'Old':>7} {'New':>7} {'Old PnL':>8} {'New PnL':>8} {'Score':>6}")
    print("  " + "-" * 65)
    for c in sorted(changes, key=lambda x: x['pattern']):
        print(f"  {c['pattern']:<14} {c['direction']:>5} {c['old_tp']}/{c['old_sl']:>3} "
              f"{c['new_tp']}/{c['new_sl']:>3} {c['old_pnl']:>+7.1f}% {c['new_pnl']:>+7.1f}% {c['score']:>4}/4")

# Portfolio evaluation
print(f"\n  --- Portfolio Comparison ---")
t_cur = collect_trades_1pos(current_map)
t_opt = collect_trades_1pos(optimized_map)
r_cur = eval_portfolio(t_cur)
r_opt = eval_portfolio(t_opt)

print(f"\n  {'Metric':<20} {'v1.27.0 (cur)':>14} {'Optimized':>14} {'Delta':>10}")
print("  " + "-" * 62)
for k, label, fmt in [
    ('trades', 'Trades', 'd'), ('wr', 'WR (%)', '.1f'),
    ('pnl', 'PnL (%)', '.1f'), ('mdd', 'MDD (%)', '.1f'),
    ('pf', 'PF', '.2f'), ('pnl_mdd', 'PnL/MDD', '.1f'),
    ('max_consec_loss', 'Max Consec Loss', 'd'),
]:
    o = r_cur[k]; n = r_opt[k]; d = n - o
    if fmt == 'd':
        print(f"  {label:<20} {o:>14} {n:>14} {d:>+10}")
    elif k in ('wr', 'pnl', 'mdd'):
        print(f"  {label:<20} {o:>13.1f}% {n:>13.1f}% {d:>+9.1f}%")
    elif k == 'pnl_mdd':
        print(f"  {label:<20} {o:>13.1f}x {n:>13.1f}x {d:>+9.1f}x")
    else:
        print(f"  {label:<20} {o:>14.2f} {n:>14.2f} {d:>+10.2f}")

# 3A: Portfolio MC (Sign Randomization)
print(f"\n  --- Portfolio MC (Sign Randomization, {MC_SIMS} sims) ---")
pnl_arr_cur = np.array([t[2] for t in t_cur])
pnl_arr_opt = np.array([t[2] for t in t_opt])

signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_arr_cur)))
mc_p_cur = float(np.mean(np.sum(pnl_arr_cur * signs, axis=1) >= np.sum(pnl_arr_cur)))
signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_arr_opt)))
mc_p_opt = float(np.mean(np.sum(pnl_arr_opt * signs, axis=1) >= np.sum(pnl_arr_opt)))

print(f"    Current MC p-value:   {mc_p_cur:.4f} ({'PASS' if mc_p_cur < 0.01 else 'FAIL'})")
print(f"    Optimized MC p-value: {mc_p_opt:.4f} ({'PASS' if mc_p_opt < 0.01 else 'FAIL'})")

# 3B: Portfolio MC MDD Distribution
print(f"\n  --- Portfolio MC MDD Distribution ({MC_SIMS} sims) ---")
t0 = time.time()
mdds_cur = portfolio_mc_mdd(t_cur)
mdds_opt = portfolio_mc_mdd(t_opt)
elapsed = time.time() - t0
print(f"    (completed in {elapsed:.1f}s)")

print(f"\n    {'Percentile':<12} {'Current':>10} {'Optimized':>10}")
print("    " + "-" * 35)
for pctl, label in [(50, 'P50'), (90, 'P90'), (95, 'P95'), (99, 'P99')]:
    v_cur = np.percentile(mdds_cur, pctl)
    v_opt = np.percentile(mdds_opt, pctl)
    print(f"    {label:<12} {v_cur:>9.1f}% {v_opt:>9.1f}%")

# 3C: Walk-Forward Portfolio (5-fold temporal)
print(f"\n  --- Portfolio Walk-Forward (5-fold) ---")
for label, trades in [('Current', t_cur), ('Optimized', t_opt)]:
    sorted_trades = sorted(trades, key=lambda x: x[0])
    fold_size = len(sorted_trades) // 5
    wf_count = 0
    folds_str = []
    for f in range(5):
        s = f * fold_size
        e = s + fold_size if f < 4 else len(sorted_trades)
        fpnl = sum(t[2] for t in sorted_trades[s:e])
        ok = fpnl > 0
        if ok: wf_count += 1
        folds_str.append(f"F{f+1}:{fpnl:+.1f}%({'OK' if ok else 'X'})")
    print(f"    {label:>10}: {' '.join(folds_str)}  => {wf_count}/5")

# 3D: Robustness Sweep (TP scale 60-100%)
print(f"\n  --- Robustness Sweep (TP Scale 60-100% on Optimized) ---")
scales = [0.60, 0.65, 0.67, 0.70, 0.73, 0.75, 0.80, 0.85, 0.90, 1.00]
sweep_results = []
print(f"    {'Scale':>6}  {'PnL':>10}  {'Trades':>7}  {'WR':>7}  {'MDD':>7}  {'PnL/MDD':>8}")
print("    " + "-" * 55)

for scale in scales:
    port = {}
    for pat in optimized_map:
        direction, tp, sl = optimized_map[pat]
        port[pat] = (direction, round(tp * scale, 2), sl)
    trades = collect_trades_1pos(port)
    stats = eval_portfolio(trades)
    marker = " <<<" if scale == 1.0 else ""
    print(f"    {scale*100:>5.0f}%  {stats['pnl']:>+9.1f}%  {stats['trades']:>7d}  "
          f"{stats['wr']:>6.1f}%  {stats['mdd']:>6.1f}%  {stats['pnl_mdd']:>7.1f}x{marker}")
    sweep_results.append({
        'scale': scale, 'pnl': round(stats['pnl'], 1),
        'trades': stats['trades'], 'wr': round(stats['wr'], 1),
        'mdd': round(stats['mdd'], 1), 'pnl_mdd': round(stats['pnl_mdd'], 1),
    })

best_sweep = max(sweep_results, key=lambda x: x['pnl_mdd'])
plateau_90 = [s for s in sweep_results if s['pnl_mdd'] >= best_sweep['pnl_mdd'] * 0.90]
print(f"\n    Best PnL/MDD: {best_sweep['scale']*100:.0f}% ({best_sweep['pnl_mdd']:.1f}x)")
print(f"    90% plateau: {len(plateau_90)} scale points "
      f"({min(s['scale'] for s in plateau_90)*100:.0f}%~{max(s['scale'] for s in plateau_90)*100:.0f}%)")

# 3E: Consecutive Loss Analysis
print(f"\n  --- Consecutive Loss Analysis ---")
consec_cur = analyze_consecutive_losses(t_cur)
consec_opt = analyze_consecutive_losses(t_opt)

print(f"    {'Metric':<30} {'Current':>10} {'Optimized':>10}")
print("    " + "-" * 55)
print(f"    {'Max consecutive losses':<30} {consec_cur['max_streak']:>10d} {consec_opt['max_streak']:>10d}")
print(f"    {'Worst streak PnL':<30} {consec_cur['max_loss_sum']:>+9.1f}% {consec_opt['max_loss_sum']:>+9.1f}%")
for length in [1, 2, 3, 4, 5]:
    c_cur = consec_cur['streak_dist'].get(length, 0)
    c_opt = consec_opt['streak_dist'].get(length, 0)
    if c_cur > 0 or c_opt > 0:
        print(f"    {f'{length}x consecutive':<30} {c_cur:>10d} {c_opt:>10d}")


# ============================================================
# Phase 4: Final Recommendation
# ============================================================
print(f"\n{'='*80}")
print("PHASE 4: Final Recommendation")
print(f"{'='*80}")

# Per-pattern recommendations
print(f"\n  Per-pattern recommendations:")
recommendations = []
for pat, (direction, cur_tp, cur_sl) in sorted(current_map.items()):
    r = grid_results[pat]
    best = r['best']
    opt_dir, opt_tp, opt_sl = optimized_map[pat]

    if opt_tp != cur_tp or opt_sl != cur_sl:
        action = "ACCEPT"
        reason = f"PnL {r['current']['total_pnl']:+.1f}% -> {best['total_pnl']:+.1f}%, score {validation_results[pat]['composite_score']}/4"
    elif best is not None and pat in validation_results and validation_results[pat]['verdict'] == 'REJECT':
        action = "KEEP (validation fail)"
        reason = f"Grid found {best['tp']}/{best['sl']} but score {validation_results[pat]['composite_score']}/4 < {COMPOSITE_THRESHOLD}"
    elif best is not None and not r['changed']:
        action = "KEEP (already optimal)"
        reason = f"Current {cur_tp}/{cur_sl} is best in grid"
    elif best is None:
        action = "KEEP (no alternative)"
        reason = "No valid combo in grid"
    else:
        action = "KEEP"
        reason = "Current is better or equal"

    recommendations.append({
        'pattern': pat, 'direction': direction,
        'action': action, 'reason': reason,
        'tp': opt_tp, 'sl': opt_sl,
    })

# Summary counts
n_accept = sum(1 for r in recommendations if r['action'] == 'ACCEPT')
n_keep_optimal = sum(1 for r in recommendations if 'already optimal' in r['action'])
n_keep_fail = sum(1 for r in recommendations if 'validation fail' in r['action'])
n_keep_noalt = sum(1 for r in recommendations if 'no alternative' in r['action'])
n_keep_other = len(recommendations) - n_accept - n_keep_optimal - n_keep_fail - n_keep_noalt

print(f"    ACCEPT (new TP/SL):          {n_accept}")
print(f"    KEEP (already optimal):      {n_keep_optimal}")
print(f"    KEEP (validation failed):    {n_keep_fail}")
print(f"    KEEP (no valid alternative): {n_keep_noalt}")
print(f"    KEEP (other):                {n_keep_other}")

# Portfolio verdict
print(f"\n  Portfolio-level verdict:")
issues = []
advantages = []

if mc_p_opt >= 0.01:
    issues.append(f"Optimized portfolio MC FAIL (p={mc_p_opt:.4f})")
if r_opt['pnl'] < r_cur['pnl']:
    issues.append(f"PnL decreased: {r_cur['pnl']:+.1f}% -> {r_opt['pnl']:+.1f}%")
if r_opt['mdd'] > r_cur['mdd'] * 1.1:
    issues.append(f"MDD increased >10%: {r_cur['mdd']:.1f}% -> {r_opt['mdd']:.1f}%")

if r_opt['pnl'] > r_cur['pnl']:
    advantages.append(f"PnL improved: {r_cur['pnl']:+.1f}% -> {r_opt['pnl']:+.1f}%")
if r_opt['mdd'] < r_cur['mdd']:
    advantages.append(f"MDD reduced: {r_cur['mdd']:.1f}% -> {r_opt['mdd']:.1f}%")
if r_opt['pnl_mdd'] > r_cur['pnl_mdd']:
    advantages.append(f"PnL/MDD improved: {r_cur['pnl_mdd']:.1f}x -> {r_opt['pnl_mdd']:.1f}x")
if r_opt['wr'] > r_cur['wr']:
    advantages.append(f"WR improved: {r_cur['wr']:.1f}% -> {r_opt['wr']:.1f}%")
if r_opt['max_consec_loss'] < r_cur['max_consec_loss']:
    advantages.append(f"Max consec loss reduced: {r_cur['max_consec_loss']} -> {r_opt['max_consec_loss']}")

if issues:
    print(f"    Issues ({len(issues)}):")
    for iss in issues:
        print(f"      - {iss}")
if advantages:
    print(f"    Advantages ({len(advantages)}):")
    for adv in advantages:
        print(f"      + {adv}")

if not issues and len(advantages) >= 2:
    verdict = "ADOPT"
    print(f"\n    VERDICT: ADOPT optimized portfolio")
elif not issues:
    verdict = "MARGINAL"
    print(f"\n    VERDICT: MARGINAL improvement — review manually")
else:
    verdict = "KEEP_CURRENT"
    print(f"\n    VERDICT: KEEP current v1.27.0 portfolio")

# Final recommended PATTERN_OPTIMAL_TPSL
print(f"\n  --- Recommended PATTERN_OPTIMAL_TPSL ---")
if changes:
    for c in sorted(changes, key=lambda x: x['pattern']):
        print(f"    '{c['pattern']}': ({c['new_tp']}, {c['new_sl']}),  # was ({c['old_tp']}, {c['old_sl']})")
else:
    print(f"    No changes — current v1.27.0 TP/SL values are optimal")


# ============================================================
# Save results
# ============================================================
output = {
    'meta': {
        'script': 'tp_sl_reoptimization_v1270.py',
        'version': 'v1.27.0',
        'timestamp': datetime.now().isoformat(),
        'total_patterns': len(current_map),
        'grid': TP_SL_GRID,
        'constraints': {
            'mc_threshold': MC_THRESHOLD,
            'wf_threshold': WF_THRESHOLD,
            'period_threshold': PERIOD_THRESHOLD,
            'min_trades': MIN_TRADES,
            'edge_threshold': EDGE_THRESHOLD,
            'composite_threshold': COMPOSITE_THRESHOLD,
        },
    },
    'phase1_grid_search': {
        'summary': {
            'improved': n_improved,
            'already_optimal': n_same,
            'no_valid': n_no_valid,
        },
        'per_pattern': {pat: r for pat, r in grid_results.items()},
    },
    'phase2_validation': {
        'summary': {
            'accepted': len(accepted),
            'rejected': len(rejected),
        },
        'per_pattern': validation_results,
    },
    'phase3_portfolio': {
        'current': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_cur.items()},
        'optimized': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_opt.items()},
        'mc_sign': {'current_p': mc_p_cur, 'optimized_p': mc_p_opt},
        'mc_mdd': {
            'current': {f'p{p}': round(float(np.percentile(mdds_cur, p)), 1) for p in [50, 90, 95, 99]},
            'optimized': {f'p{p}': round(float(np.percentile(mdds_opt, p)), 1) for p in [50, 90, 95, 99]},
        },
        'robustness_sweep': sweep_results,
        'consecutive_loss': {
            'current': {k: v for k, v in consec_cur.items()},
            'optimized': {k: v for k, v in consec_opt.items()},
        },
    },
    'phase4_recommendation': {
        'verdict': verdict,
        'changes': changes,
        'issues': issues,
        'advantages': advantages,
        'per_pattern': recommendations,
    },
}

out_path = Path(__file__).resolve().parent / '../../results/tp_sl_reoptimization_v1270.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done.")
