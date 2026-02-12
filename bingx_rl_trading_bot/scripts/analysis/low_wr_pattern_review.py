"""
Low-WR Pattern Deep Review & Optimization (v1.27.2)
=====================================================
v1.27.1 라이브에서 MD-ST-BD(WR 57.6%, SL 0.5%) 1분 만에 SL 히트,
DN-D-BD(WR 54.5%) 230분 후 SL 히트 등 연속 손실 발생.

포트폴리오 하위 50% WR 패턴(26개)에 대한 전면 검토 및 최적화.

Phase 1: Edge & Viability Audit (26개 전수)
Phase 2: TP/SL Re-optimization (KEEP + OPTIMIZE 패턴)
Phase 3: Portfolio Impact Simulation (A/B/C 시나리오)
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
    PATTERN_STATS,
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
MIN_SL_PCT = 0.5  # v1.27.2: execution feasibility — SL must be >= 0.5%

# Spread/slippage for feasibility analysis
SPREAD_PCT = 0.05   # BingX BTC typical spread
SLIPPAGE_PCT = 0.02  # market order slippage

print("=" * 80)
print("Low-WR Pattern Deep Review & Optimization (v1.27.2)")
print("  Target: Bottom 50% WR patterns (26 patterns)")
print(f"  Constraints: MC<{MC_THRESHOLD}, WF>={WF_THRESHOLD}/5, SL>={MIN_SL_PCT}%")
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
# Engine functions (reused from tp_sl_reoptimization_v1270.py)
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


def find_best_tpsl(indices, direction, min_trades=5, require_min_sl=False):
    """Find best TP/SL by total PnL with MC<0.01 constraint."""
    best = None
    best_pnl = -999
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            if require_min_sl and sl < MIN_SL_PCT:
                continue
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


# ============================================================
# Build current v1.27.1 map
# ============================================================
current_map = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('LONG', tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('SHORT', tp, sl)

print(f"\nCurrent v1.27.1: {len(current_map)} patterns "
      f"({len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S)")


# ============================================================
# Identify bottom 26 by WR
# ============================================================
all_patterns_wr = []
for pat, stats in PATTERN_STATS.items():
    all_patterns_wr.append((pat, stats['wr'], stats['direction']))
all_patterns_wr.sort(key=lambda x: x[1])

n_bottom = len(all_patterns_wr) // 2  # 26 of 52
bottom_patterns = {p[0] for p in all_patterns_wr[:n_bottom]}
top_patterns = {p[0] for p in all_patterns_wr[n_bottom:]}

print(f"\nBottom {n_bottom} patterns by WR (target for review):")
print(f"  WR range: {all_patterns_wr[0][1]:.1f}% ~ {all_patterns_wr[n_bottom-1][1]:.1f}%")
print(f"  Top 26 WR range: {all_patterns_wr[n_bottom][1]:.1f}% ~ {all_patterns_wr[-1][1]:.1f}%")

# Show target patterns sorted by WR
print(f"\n  {'Pattern':<14} {'Dir':>5} {'WR':>6} {'TP/SL':>7} {'R:R':>5} {'Trades':>7} {'MC':>7}")
print("  " + "-" * 60)
for pat, wr, direction in all_patterns_wr[:n_bottom]:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    rr = tp / sl if sl > 0 else 999
    mc = PATTERN_STATS[pat]['mc']
    trades = PATTERN_STATS[pat]['trades']
    print(f"  {pat:<14} {direction:>5} {wr:>5.1f}% {tp}/{sl:>3} {rr:>5.2f} {trades:>7} {mc:>7.4f}")


# ============================================================
# PHASE 1: Edge & Viability Audit
# ============================================================
print(f"\n{'='*80}")
print("PHASE 1: Edge & Viability Audit (26 patterns)")
print(f"  - Edge Decomposition: excess WR vs Random Walk baseline")
print(f"  - Execution Feasibility: SL < {MIN_SL_PCT}% analysis")
print(f"  - Sample Confidence: Bayesian for small samples (n < 25)")
print(f"  - MC Retest: sign randomization 10k sims")
print(f"{'='*80}")

t0 = time.time()

# Random Walk baseline WR
np.random.seed(42)
sample_idx = np.random.choice(np.arange(22, n_bars - MAX_BARS), size=500, replace=False)

audit_results = {}

for pat, wr_stat, _ in all_patterns_wr[:n_bottom]:
    direction, cur_tp, cur_sl = current_map[pat]
    indices = pattern_indices.get(pat, np.array([]))

    # --- 1A: Edge Decomposition ---
    # Random walk WR = SL / (TP + SL) * 100 (distance-based)
    rw_wr = cur_sl / (cur_tp + cur_sl) * 100

    # Actual backtest
    pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    n_trades = len(pnls)
    if n_trades > 0:
        actual_wr = sum(1 for x in pnls if x > 0) / n_trades * 100
        total_pnl = sum(pnls)
    else:
        actual_wr = 0
        total_pnl = 0
    excess_wr = actual_wr - rw_wr

    # Also check with random entries (empirical baseline)
    base_pnls = bt_fixed(sample_idx, direction, cur_tp, cur_sl)
    empirical_rw_wr = sum(1 for x in base_pnls if x > 0) / len(base_pnls) * 100 if base_pnls else 50
    empirical_excess = actual_wr - empirical_rw_wr

    # --- 1B: Execution Feasibility ---
    execution_cost = SPREAD_PCT + SLIPPAGE_PCT  # 0.07% per side
    effective_sl = cur_sl - execution_cost  # SL net of execution costs
    sl_feasible = cur_sl >= MIN_SL_PCT
    sl_risk = "OK" if cur_sl >= 1.0 else ("MARGINAL" if cur_sl >= MIN_SL_PCT else "DANGER")

    # For very tight SL, simulate impact
    if cur_sl < 1.0:
        # Rerun with tighter effective SL
        pnls_tight = bt_fixed(indices, direction, cur_tp, cur_sl * 0.9)
        wr_tight = sum(1 for x in pnls_tight if x > 0) / len(pnls_tight) * 100 if pnls_tight else 0
        wr_degradation = actual_wr - wr_tight
    else:
        wr_degradation = 0

    # --- 1C: Sample Confidence ---
    # Wilson score interval for WR
    if n_trades > 0:
        p_hat = actual_wr / 100
        z = 1.96  # 95% CI
        denom = 1 + z**2 / n_trades
        center = (p_hat + z**2 / (2 * n_trades)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_trades)) / n_trades) / denom
        ci_low = max(0, (center - margin)) * 100
        ci_high = min(1, (center + margin)) * 100
    else:
        ci_low = 0
        ci_high = 100
    small_sample = n_trades < 25

    # --- 1D: MC Retest ---
    mc_pvalue = mc_test(pnls) if n_trades >= 5 else 1.0

    # --- 1E: Verdict ---
    remove_reasons = []
    optimize_reasons = []

    if excess_wr < EDGE_THRESHOLD:
        remove_reasons.append(f"Edge {excess_wr:+.1f}pp < {EDGE_THRESHOLD}pp threshold")
    if mc_pvalue >= 0.05:
        remove_reasons.append(f"MC p={mc_pvalue:.4f} >= 0.05")
    if not sl_feasible:
        remove_reasons.append(f"SL {cur_sl}% < {MIN_SL_PCT}% (execution infeasible)")

    if sl_risk == "MARGINAL":
        optimize_reasons.append(f"SL {cur_sl}% marginal — try wider SL")
    if mc_pvalue >= MC_THRESHOLD and mc_pvalue < 0.05:
        optimize_reasons.append(f"MC p={mc_pvalue:.4f} borderline — try re-optimize")
    if excess_wr >= EDGE_THRESHOLD and excess_wr < 10:
        optimize_reasons.append(f"Edge {excess_wr:+.1f}pp moderate — room for improvement")

    if remove_reasons:
        verdict = "REMOVE"
        reasons = remove_reasons
    elif optimize_reasons:
        verdict = "OPTIMIZE"
        reasons = optimize_reasons
    else:
        verdict = "KEEP"
        reasons = ["Passes all viability checks"]

    audit_results[pat] = {
        'direction': direction,
        'tp': cur_tp, 'sl': cur_sl,
        'trades': n_trades, 'total_pnl': round(total_pnl, 1),
        'actual_wr': round(actual_wr, 1),
        'rw_wr': round(rw_wr, 1),
        'excess_wr': round(excess_wr, 1),
        'empirical_rw_wr': round(empirical_rw_wr, 1),
        'empirical_excess': round(empirical_excess, 1),
        'sl_risk': sl_risk,
        'effective_sl': round(effective_sl, 2),
        'wr_degradation': round(wr_degradation, 1),
        'ci_95': [round(ci_low, 1), round(ci_high, 1)],
        'small_sample': small_sample,
        'mc_pvalue': round(mc_pvalue, 4),
        'verdict': verdict,
        'reasons': reasons,
    }

elapsed = time.time() - t0
print(f"\n  Audit complete in {elapsed:.1f}s\n")

# Display audit results
print(f"  {'Pattern':<14} {'Dir':>5} {'WR':>5} {'RW':>5} {'Excess':>7} {'MC':>7} {'SL Risk':>8} {'Verdict':>8}")
print("  " + "-" * 70)

n_remove = 0
n_optimize = 0
n_keep = 0

for pat, wr_stat, _ in all_patterns_wr[:n_bottom]:
    a = audit_results[pat]
    v_marker = {'REMOVE': '***', 'OPTIMIZE': ' * ', 'KEEP': '   '}
    print(f"  {pat:<14} {a['direction']:>5} {a['actual_wr']:>4.1f}% {a['rw_wr']:>4.1f}% {a['excess_wr']:>+6.1f}pp "
          f"{a['mc_pvalue']:>7.4f} {a['sl_risk']:>8} {a['verdict']:>7}{v_marker[a['verdict']]}")

    if a['verdict'] == 'REMOVE':
        n_remove += 1
    elif a['verdict'] == 'OPTIMIZE':
        n_optimize += 1
    else:
        n_keep += 1

print(f"\n  Summary: REMOVE={n_remove}, OPTIMIZE={n_optimize}, KEEP={n_keep}")

# Show reasons for REMOVE
if n_remove > 0:
    print(f"\n  --- REMOVE Reasons ---")
    for pat, wr_stat, _ in all_patterns_wr[:n_bottom]:
        a = audit_results[pat]
        if a['verdict'] == 'REMOVE':
            print(f"    {pat}: {'; '.join(a['reasons'])}")

# Show reasons for OPTIMIZE
if n_optimize > 0:
    print(f"\n  --- OPTIMIZE Reasons ---")
    for pat, wr_stat, _ in all_patterns_wr[:n_bottom]:
        a = audit_results[pat]
        if a['verdict'] == 'OPTIMIZE':
            print(f"    {pat}: {'; '.join(a['reasons'])}")


# ============================================================
# PHASE 2: TP/SL Re-optimization
# ============================================================
print(f"\n{'='*80}")
print("PHASE 2: TP/SL Re-optimization (KEEP + OPTIMIZE patterns)")
print(f"  Grid: {len(TP_SL_GRID)}x{len(TP_SL_GRID)} = {len(TP_SL_GRID)**2} combos")
print(f"  Extra constraint: SL >= {MIN_SL_PCT}%")
print(f"{'='*80}")

t0 = time.time()
n_folds = 5
optimization_results = {}

# Only optimize patterns with KEEP or OPTIMIZE verdict
target_patterns = [pat for pat in audit_results if audit_results[pat]['verdict'] in ('KEEP', 'OPTIMIZE')]
print(f"\n  Optimizing {len(target_patterns)} patterns...")

for pat in sorted(target_patterns):
    a = audit_results[pat]
    direction = a['direction']
    cur_tp, cur_sl = a['tp'], a['sl']
    indices = pattern_indices.get(pat, np.array([]))

    # --- 2A: Grid Search with SL >= MIN_SL_PCT ---
    valid_combos = []
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            if sl < MIN_SL_PCT:
                continue
            result = eval_combo(indices, direction, tp, sl)
            if result:
                valid_combos.append(result)

    best = max(valid_combos, key=lambda x: x['total_pnl']) if valid_combos else None

    # Also evaluate current TP/SL
    cur_pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    cur_pnl = sum(cur_pnls) if cur_pnls else 0
    cur_n = len(cur_pnls)
    cur_wr_val = sum(1 for x in cur_pnls if x > 0) / cur_n * 100 if cur_n > 0 else 0

    changed = best is not None and (best['tp'] != cur_tp or best['sl'] != cur_sl)
    improved = changed and best['total_pnl'] > cur_pnl

    # --- 2B: Deep Validation (if changed) ---
    validation = None
    if best is not None:
        new_tp, new_sl = best['tp'], best['sl']
        si = np.sort(indices)

        # CV Stability
        fs = len(si) // n_folds
        cv_stable = False
        cv_oos_positive = 0
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
                fold_best = find_best_tpsl(train_idx, direction, min_trades=3, require_min_sl=True)
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
                cv_stable = most_common[1] >= 3
            cv_oos_positive = sum(1 for p in oos_pnls if p > 0)

        # Plateau Test
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
                        if nsl < MIN_SL_PCT:
                            continue
                        nb_pnls = bt_fixed(indices, direction, ntp, nsl)
                        n_total_nb += 1
                        if len(nb_pnls) >= 5 and sum(nb_pnls) > 0:
                            n_profitable_nb += 1
            plateau = n_total_nb > 0 and n_profitable_nb >= n_total_nb * 0.6

        # Edge Test
        rw_wr = new_sl / (new_tp + new_sl) * 100
        pnls_new = bt_fixed(indices, direction, new_tp, new_sl)
        act_wr = sum(1 for p in pnls_new if p > 0) / len(pnls_new) * 100 if pnls_new else 0
        new_excess_wr = act_wr - rw_wr
        has_edge = new_excess_wr >= EDGE_THRESHOLD

        # Composite Score
        score = 0
        if cv_stable: score += 1
        if plateau: score += 1
        if has_edge: score += 1
        if cv_oos_positive >= 3: score += 1

        val_verdict = "ACCEPT" if score >= COMPOSITE_THRESHOLD else "REJECT"

        validation = {
            'tp': new_tp, 'sl': new_sl,
            'cv_stable': cv_stable,
            'cv_oos_positive': cv_oos_positive,
            'plateau': plateau,
            'n_profitable_nb': n_profitable_nb,
            'n_total_nb': n_total_nb,
            'has_edge': has_edge,
            'rw_wr': round(rw_wr, 1),
            'act_wr': round(act_wr, 1),
            'excess_wr': round(new_excess_wr, 1),
            'composite_score': score,
            'verdict': val_verdict,
        }

    optimization_results[pat] = {
        'direction': direction,
        'current': {'tp': cur_tp, 'sl': cur_sl, 'pnl': round(cur_pnl, 1),
                    'trades': cur_n, 'wr': round(cur_wr_val, 1)},
        'best': best,
        'n_valid': len(valid_combos),
        'changed': changed,
        'improved': improved,
        'validation': validation,
    }

elapsed = time.time() - t0
print(f"  Optimization complete in {elapsed:.1f}s\n")

# Display results
print(f"  {'Pattern':<14} {'Dir':>5} | {'CUR TP/SL':>9} {'PnL':>8} | {'NEW TP/SL':>9} {'PnL':>8} | {'CV':>3} {'Plt':>3} {'Edg':>4} {'OOS':>4} {'Sc':>3} {'V':>7}")
print("  " + "-" * 95)

n_phase2_accept = 0
n_phase2_reject = 0
n_phase2_same = 0

for pat in sorted(optimization_results.keys()):
    o = optimization_results[pat]
    cur = o['current']
    best = o['best']
    val = o['validation']

    if best is None:
        print(f"  {pat:<14} {o['direction']:>5} | {cur['tp']}/{cur['sl']:>5} {cur['pnl']:>+7.1f}% | {'---':>9} {'---':>8} | {'---':>3} {'---':>3} {'---':>4} {'---':>4} {'---':>3} {'NO ALT':>7}")
        n_phase2_reject += 1
        continue

    if not o['changed']:
        print(f"  {pat:<14} {o['direction']:>5} | {cur['tp']}/{cur['sl']:>5} {cur['pnl']:>+7.1f}% | {'same':>9} {'':>8} | {'---':>3} {'---':>3} {'---':>4} {'---':>4} {'---':>3} {'SAME':>7}")
        n_phase2_same += 1
        continue

    cv_str = "Y" if val['cv_stable'] else "N"
    pl_str = "Y" if val['plateau'] else "N"
    eg_str = f"+{val['excess_wr']:.0f}" if val['has_edge'] else f"{val['excess_wr']:.0f}"
    oos_str = f"{val['cv_oos_positive']}/5"
    v_str = val['verdict']

    if v_str == 'ACCEPT':
        n_phase2_accept += 1
    else:
        n_phase2_reject += 1

    print(f"  {pat:<14} {o['direction']:>5} | {cur['tp']}/{cur['sl']:>5} {cur['pnl']:>+7.1f}% "
          f"| {best['tp']}/{best['sl']:>5} {best['total_pnl']:>+7.1f}% "
          f"| {cv_str:>3} {pl_str:>3} {eg_str:>4} {oos_str:>4} {val['composite_score']:>2}/4 {v_str:>7}")

print(f"\n  Phase 2 Summary: ACCEPT={n_phase2_accept}, REJECT={n_phase2_reject}, SAME={n_phase2_same}")


# ============================================================
# PHASE 3: Portfolio Impact Simulation
# ============================================================
print(f"\n{'='*80}")
print("PHASE 3: Portfolio Impact Simulation")
print(f"  Scenario A: v1.27.1 baseline (52 patterns)")
print(f"  Scenario B: REMOVE only (remove flagged patterns)")
print(f"  Scenario C: REMOVE + OPTIMIZE (remove + re-optimize)")
print(f"{'='*80}")

t0 = time.time()

# Build scenario maps
# Scenario A: baseline
map_a = dict(current_map)

# Scenario B: remove REMOVE-flagged patterns
map_b = {}
removed_patterns = []
for pat, (direction, tp, sl) in current_map.items():
    if pat in audit_results and audit_results[pat]['verdict'] == 'REMOVE':
        removed_patterns.append(pat)
        continue
    map_b[pat] = (direction, tp, sl)

# Scenario C: remove + optimize
map_c = {}
optimized_patterns = []
for pat, (direction, tp, sl) in current_map.items():
    if pat in audit_results and audit_results[pat]['verdict'] == 'REMOVE':
        continue  # removed
    # Check if optimization produced accepted result
    if pat in optimization_results:
        o = optimization_results[pat]
        if o['changed'] and o['validation'] and o['validation']['verdict'] == 'ACCEPT':
            new_tp = o['best']['tp']
            new_sl = o['best']['sl']
            map_c[pat] = (direction, new_tp, new_sl)
            optimized_patterns.append({
                'pattern': pat, 'direction': direction,
                'old_tp': tp, 'old_sl': sl,
                'new_tp': new_tp, 'new_sl': new_sl,
                'old_pnl': o['current']['pnl'],
                'new_pnl': o['best']['total_pnl'],
                'score': o['validation']['composite_score'],
            })
            continue
    map_c[pat] = (direction, tp, sl)

print(f"\n  Scenario A: {len(map_a)} patterns (baseline)")
print(f"  Scenario B: {len(map_b)} patterns ({len(removed_patterns)} removed: {', '.join(removed_patterns)})")
print(f"  Scenario C: {len(map_c)} patterns ({len(removed_patterns)} removed + {len(optimized_patterns)} optimized)")

if optimized_patterns:
    print(f"\n  Optimized patterns in Scenario C:")
    for op in sorted(optimized_patterns, key=lambda x: x['pattern']):
        print(f"    {op['pattern']}: {op['old_tp']}/{op['old_sl']} -> {op['new_tp']}/{op['new_sl']} "
              f"(PnL {op['old_pnl']:+.1f}% -> {op['new_pnl']:+.1f}%, score {op['score']}/4)")

# Evaluate all scenarios
print(f"\n  --- Portfolio Comparison ---")
t_a = collect_trades_1pos(map_a)
t_b = collect_trades_1pos(map_b)
t_c = collect_trades_1pos(map_c)
r_a = eval_portfolio(t_a)
r_b = eval_portfolio(t_b)
r_c = eval_portfolio(t_c)

print(f"\n  {'Metric':<20} {'A (v1.27.1)':>14} {'B (Remove)':>14} {'C (Rem+Opt)':>14}")
print("  " + "-" * 66)
for k, label, fmt in [
    ('trades', 'Trades', 'd'), ('wr', 'WR (%)', '.1f'),
    ('pnl', 'PnL (%)', '.1f'), ('mdd', 'MDD (%)', '.1f'),
    ('pf', 'PF', '.2f'), ('pnl_mdd', 'PnL/MDD', '.1f'),
    ('max_consec_loss', 'Max Consec Loss', 'd'),
    ('avg_win', 'Avg Win (%)', '.2f'), ('avg_loss', 'Avg Loss (%)', '.2f'),
]:
    va, vb, vc = r_a[k], r_b[k], r_c[k]
    if fmt == 'd':
        print(f"  {label:<20} {va:>14} {vb:>14} {vc:>14}")
    elif k in ('wr', 'pnl', 'mdd'):
        print(f"  {label:<20} {va:>13.1f}% {vb:>13.1f}% {vc:>13.1f}%")
    elif k == 'pnl_mdd':
        print(f"  {label:<20} {va:>13.1f}x {vb:>13.1f}x {vc:>13.1f}x")
    elif k in ('avg_win', 'avg_loss'):
        print(f"  {label:<20} {va:>13.2f}% {vb:>13.2f}% {vc:>13.2f}%")
    else:
        print(f"  {label:<20} {va:>14.2f} {vb:>14.2f} {vc:>14.2f}")

# Delta vs baseline
print(f"\n  {'Delta vs A':<20} {'':>14} {'B-A':>14} {'C-A':>14}")
print("  " + "-" * 66)
for k, label in [('pnl', 'PnL'), ('mdd', 'MDD'), ('pnl_mdd', 'PnL/MDD'),
                  ('wr', 'WR'), ('pf', 'PF'), ('trades', 'Trades')]:
    db = r_b[k] - r_a[k]
    dc = r_c[k] - r_a[k]
    if k in ('pnl', 'mdd', 'wr'):
        print(f"  {label:<20} {'':>14} {db:>+13.1f}% {dc:>+13.1f}%")
    elif k == 'pnl_mdd':
        print(f"  {label:<20} {'':>14} {db:>+13.1f}x {dc:>+13.1f}x")
    elif k == 'trades':
        print(f"  {label:<20} {'':>14} {db:>+14} {dc:>+14}")
    else:
        print(f"  {label:<20} {'':>14} {db:>+14.2f} {dc:>+14.2f}")

# Portfolio MC (Sign Randomization)
print(f"\n  --- Portfolio MC (Sign Randomization, {MC_SIMS} sims) ---")
for label, trades in [('A (baseline)', t_a), ('B (remove)', t_b), ('C (rem+opt)', t_c)]:
    pnl_arr = np.array([t[2] for t in trades])
    signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_arr)))
    mc_p = float(np.mean(np.sum(pnl_arr * signs, axis=1) >= np.sum(pnl_arr)))
    print(f"    {label:>15}: p={mc_p:.4f} ({'PASS' if mc_p < 0.01 else 'FAIL'})")

# Portfolio MC MDD Distribution
print(f"\n  --- Portfolio MC MDD Distribution ({MC_SIMS} sims) ---")
mdds_a = portfolio_mc_mdd(t_a)
mdds_b = portfolio_mc_mdd(t_b)
mdds_c = portfolio_mc_mdd(t_c)

print(f"\n    {'Percentile':<12} {'A (v1.27.1)':>12} {'B (Remove)':>12} {'C (Rem+Opt)':>12}")
print("    " + "-" * 52)
for pctl, label in [(50, 'P50'), (90, 'P90'), (95, 'P95'), (99, 'P99')]:
    va = np.percentile(mdds_a, pctl)
    vb = np.percentile(mdds_b, pctl)
    vc = np.percentile(mdds_c, pctl)
    print(f"    {label:<12} {va:>11.1f}% {vb:>11.1f}% {vc:>11.1f}%")

# Walk-Forward Portfolio (5-fold temporal)
print(f"\n  --- Portfolio Walk-Forward (5-fold) ---")
for label, trades in [('A (v1.27.1)', t_a), ('B (remove)', t_b), ('C (rem+opt)', t_c)]:
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
    print(f"    {label:>15}: {' '.join(folds_str)}  => {wf_count}/5")

elapsed = time.time() - t0
print(f"\n  Phase 3 complete in {elapsed:.1f}s")


# ============================================================
# FINAL RECOMMENDATION
# ============================================================
print(f"\n{'='*80}")
print("FINAL RECOMMENDATION")
print(f"{'='*80}")

# Determine best scenario
scenarios = {'A': r_a, 'B': r_b, 'C': r_c}
best_scenario = max(scenarios.keys(), key=lambda s: scenarios[s]['pnl_mdd'])

print(f"\n  Best PnL/MDD ratio: Scenario {best_scenario} ({scenarios[best_scenario]['pnl_mdd']:.1f}x)")

# Per-pattern final verdicts
print(f"\n  --- Per-Pattern Final Verdicts ---")
print(f"  {'Pattern':<14} {'Dir':>5} {'WR':>5} {'Audit':>8} {'Opt':>7} {'Final':>8} {'Action':>30}")
print("  " + "-" * 85)

final_verdicts = {}
for pat, wr_stat, _ in all_patterns_wr[:n_bottom]:
    a = audit_results[pat]
    direction = a['direction']
    tp, sl = a['tp'], a['sl']

    if a['verdict'] == 'REMOVE':
        final_action = "REMOVE"
        action_detail = f"Remove ({'; '.join(a['reasons'][:1])})"
    elif pat in optimization_results:
        o = optimization_results[pat]
        if o['changed'] and o['validation'] and o['validation']['verdict'] == 'ACCEPT':
            final_action = "OPTIMIZE"
            action_detail = f"TP/SL {tp}/{sl} -> {o['best']['tp']}/{o['best']['sl']}"
        elif o['best'] and not o['changed']:
            final_action = "KEEP"
            action_detail = f"Current {tp}/{sl} is optimal"
        else:
            final_action = "KEEP"
            action_detail = f"Keep {tp}/{sl} (validation fail or no alt)"
    else:
        final_action = "KEEP"
        action_detail = f"Keep {tp}/{sl}"

    final_verdicts[pat] = {
        'action': final_action,
        'detail': action_detail,
        'direction': direction,
        'tp': tp, 'sl': sl,
    }
    print(f"  {pat:<14} {direction:>5} {a['actual_wr']:>4.1f}% {a['verdict']:>8} "
          f"{'---' if pat not in optimization_results else (optimization_results[pat]['validation']['verdict'] if optimization_results[pat].get('validation') else 'N/A'):>7} "
          f"{final_action:>8} {action_detail:>30}")

# Summary counts
n_final_remove = sum(1 for v in final_verdicts.values() if v['action'] == 'REMOVE')
n_final_optimize = sum(1 for v in final_verdicts.values() if v['action'] == 'OPTIMIZE')
n_final_keep = sum(1 for v in final_verdicts.values() if v['action'] == 'KEEP')

print(f"\n  Final: REMOVE={n_final_remove}, OPTIMIZE={n_final_optimize}, KEEP={n_final_keep}")
print(f"  New portfolio size: {52 - n_final_remove} patterns (was 52)")

# Portfolio-level verdict
print(f"\n  --- Portfolio Verdict ---")
issues = []
advantages = []

# Compare best non-A scenario vs A
for sc_label, sc_key in [('B', 'B'), ('C', 'C')]:
    r_sc = scenarios[sc_key]
    if r_sc['pnl_mdd'] > r_a['pnl_mdd'] * 1.03:
        advantages.append(f"Scenario {sc_label} PnL/MDD {r_sc['pnl_mdd']:.1f}x > A {r_a['pnl_mdd']:.1f}x (+{(r_sc['pnl_mdd']/r_a['pnl_mdd']-1)*100:.1f}%)")
    if r_sc['pnl'] > r_a['pnl']:
        advantages.append(f"Scenario {sc_label} PnL {r_sc['pnl']:+.1f}% > A {r_a['pnl']:+.1f}%")
    if r_sc['mdd'] < r_a['mdd'] * 0.95:
        advantages.append(f"Scenario {sc_label} MDD {r_sc['mdd']:.1f}% < A {r_a['mdd']:.1f}% (-{(1-r_sc['mdd']/r_a['mdd'])*100:.1f}%)")

    if r_sc['pnl'] < r_a['pnl'] * 0.95:
        issues.append(f"Scenario {sc_label} PnL {r_sc['pnl']:+.1f}% < A {r_a['pnl']:+.1f}% ({(r_sc['pnl']/r_a['pnl']-1)*100:+.1f}%)")
    if r_sc['mdd'] > r_a['mdd'] * 1.1:
        issues.append(f"Scenario {sc_label} MDD {r_sc['mdd']:.1f}% > A {r_a['mdd']:.1f}% (+{(r_sc['mdd']/r_a['mdd']-1)*100:.1f}%)")

if advantages:
    print(f"  Advantages:")
    for adv in advantages:
        print(f"    + {adv}")
if issues:
    print(f"  Issues:")
    for iss in issues:
        print(f"    - {iss}")

# Final verdict
if best_scenario == 'A':
    overall_verdict = "KEEP_CURRENT"
    print(f"\n  VERDICT: KEEP v1.27.1 as-is (baseline is best)")
elif best_scenario == 'B' and r_b['pnl_mdd'] > r_a['pnl_mdd'] * 1.03:
    overall_verdict = "ADOPT_B"
    print(f"\n  VERDICT: ADOPT Scenario B — remove {n_final_remove} patterns")
elif best_scenario == 'C' and r_c['pnl_mdd'] > r_a['pnl_mdd'] * 1.03:
    overall_verdict = "ADOPT_C"
    print(f"\n  VERDICT: ADOPT Scenario C — remove {n_final_remove} + optimize {n_final_optimize} patterns")
else:
    overall_verdict = "MARGINAL"
    print(f"\n  VERDICT: MARGINAL improvement — review manually")

# Show recommended changes for implementation
if overall_verdict in ('ADOPT_B', 'ADOPT_C'):
    print(f"\n  --- Recommended Changes ---")
    if removed_patterns:
        print(f"\n  Patterns to REMOVE from VALIDATED_*_PATTERNS:")
        for pat in sorted(removed_patterns):
            d = PATTERN_STATS[pat]['direction']
            print(f"    {pat} ({d})")
    if overall_verdict == 'ADOPT_C' and optimized_patterns:
        print(f"\n  Patterns to UPDATE in PATTERN_OPTIMAL_TPSL:")
        for op in sorted(optimized_patterns, key=lambda x: x['pattern']):
            print(f"    '{op['pattern']}': ({op['new_tp']}, {op['new_sl']}),  # was ({op['old_tp']}, {op['old_sl']})")


# ============================================================
# Save results
# ============================================================
output = {
    'meta': {
        'script': 'low_wr_pattern_review.py',
        'version': 'v1.27.2',
        'timestamp': datetime.now().isoformat(),
        'total_patterns': len(current_map),
        'target_patterns': len(bottom_patterns),
        'grid': TP_SL_GRID,
        'constraints': {
            'mc_threshold': MC_THRESHOLD,
            'wf_threshold': WF_THRESHOLD,
            'period_threshold': PERIOD_THRESHOLD,
            'min_trades': MIN_TRADES,
            'edge_threshold': EDGE_THRESHOLD,
            'composite_threshold': COMPOSITE_THRESHOLD,
            'min_sl_pct': MIN_SL_PCT,
        },
    },
    'audit': {pat: r for pat, r in audit_results.items()},
    'verdicts': {pat: v for pat, v in final_verdicts.items()},
    'optimization': {pat: {
        'direction': r['direction'],
        'current': r['current'],
        'best': r['best'],
        'n_valid': r['n_valid'],
        'changed': r['changed'],
        'improved': r['improved'],
        'validation': r['validation'],
    } for pat, r in optimization_results.items()},
    'portfolio': {
        'scenario_a': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_a.items()},
        'scenario_b': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_b.items()},
        'scenario_c': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_c.items()},
        'removed_patterns': removed_patterns,
        'optimized_patterns': optimized_patterns,
        'mc_mdd': {
            'a': {f'p{p}': round(float(np.percentile(mdds_a, p)), 1) for p in [50, 90, 95, 99]},
            'b': {f'p{p}': round(float(np.percentile(mdds_b, p)), 1) for p in [50, 90, 95, 99]},
            'c': {f'p{p}': round(float(np.percentile(mdds_c, p)), 1) for p in [50, 90, 95, 99]},
        },
    },
    'recommendations': {
        'overall_verdict': overall_verdict,
        'best_scenario': best_scenario,
        'final_verdicts': final_verdicts,
        'issues': issues,
        'advantages': advantages,
    },
}

out_path = Path(__file__).resolve().parent / '../../results/low_wr_pattern_review.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done.")
