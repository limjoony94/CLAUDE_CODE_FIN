#!/usr/bin/env python3
"""
Extended Rediscovery v6: Hybrid Enhancement & Alternative Strategies
====================================================================
v5 showed B_HYBRID (5 stable TP/SL) barely improves safety margin (2.6→2.8pp).
Need stronger hybrid approaches for the 51-pattern portfolio.

Key insight from v5: Stable patterns share TP↓ + SL↑ pattern → higher WR, wider safety.
Can we systematically apply this principle to ALL patterns?

Phase 1: Systematic TP/SL Transform Sweep
  - TP scale × SL scale (2D sweep, 48 combos)
  - Find (tp_scale, sl_scale) maximizing pre-overlap PnL/MDD

Phase 2: Per-Pattern Pre-overlap TP/SL (relaxed criteria)
  - Grid search all 51 patterns on pre-overlap
  - Relaxed: positive PnL + edge > 3pp (no strict MC)
  - How many patterns find a qualifying pre-overlap TP/SL?

Phase 3: Cross-Period Intersection TP/SL
  - Find TP/SL combos good on BOTH pre-overlap AND in-sample
  - "Good" = positive PnL + edge > 3pp
  - Apply intersection → genuine robust TP/SL

Phase 4: Universal TP/SL Grid Search
  - Single TP/SL for ALL patterns — which maximizes pre-overlap PnL/MDD?

Phase 5: Best Hybrid Assembly & Comparison
  - Tier 1: Intersection TP/SL (where available)
  - Tier 2: Systematic transform (for remaining)
  - Tier 3: Current TP/SL (fallback)
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

PATTERNS_CURRENT = {
    "BD-BD-U": ("LONG", 1.4, 3.0), "BD-MU-BD": ("LONG", 0.7, 0.7),
    "BD-ST-U": ("LONG", 1.4, 2.0), "BU-BU-BD": ("LONG", 2.1, 3.0),
    "D-MU-U": ("LONG", 1.05, 2.0), "DN-BD-BD": ("LONG", 1.4, 1.5),
    "DN-DF-MU": ("LONG", 1.05, 3.0), "DN-DF-ST": ("LONG", 1.4, 3.0),
    "DN-DN-H": ("LONG", 0.7, 2.5), "DN-MD-DN": ("LONG", 1.5, 3.0),
    "GS-ST-ST": ("LONG", 1.05, 2.0), "H-BU-BU": ("LONG", 1.4, 3.0),
    "H-MU-MD": ("LONG", 0.7, 2.0), "IH-MD-MD": ("LONG", 0.7, 2.0),
    "IH-ST-MU": ("LONG", 0.35, 2.5), "MD-BU-MD": ("LONG", 1.4, 2.5),
    "MD-DN-MU": ("LONG", 1.0, 1.0), "MD-H-MD": ("LONG", 0.7, 0.7),
    "MD-MD-ST": ("LONG", 1.05, 2.0), "MD-ST-BD": ("LONG", 1.0, 0.5),
    "MD-ST-MD": ("LONG", 2.0, 2.0), "MU-BD-ST": ("LONG", 1.4, 3.0),
    "MU-DF-U": ("LONG", 1.05, 2.0), "MU-H-MU": ("LONG", 1.5, 0.7),
    "MU-IH-DN": ("LONG", 1.4, 2.0), "MU-U-H": ("LONG", 1.75, 3.0),
    "U-H-MU": ("LONG", 1.5, 2.0), "U-MD-GS": ("LONG", 0.35, 2.5),
    "U-MD-MD": ("LONG", 1.5, 0.7), "U-MU-H": ("LONG", 1.05, 3.0),
    "U-MU-IH": ("LONG", 2.1, 2.5), "U-ST-DF": ("LONG", 1.4, 3.0),
    "BD-BU-DN": ("SHORT", 3.0, 3.0), "BD-D-D": ("SHORT", 1.05, 3.0),
    "BD-U-H": ("SHORT", 2.5, 3.0), "BU-MD-MD": ("SHORT", 3.0, 2.0),
    "BU-ST-GS": ("SHORT", 0.7, 2.5), "D-BD-ST": ("SHORT", 1.75, 3.0),
    "D-DN-DN": ("SHORT", 2.5, 3.0), "DN-BD-BU": ("SHORT", 1.75, 3.0),
    "DN-D-BD": ("SHORT", 1.75, 1.0), "DN-DF-DN": ("SHORT", 2.0, 2.0),
    "H-U-BD": ("SHORT", 2.1, 2.0), "IH-ST-ST": ("SHORT", 1.4, 3.0),
    "MD-MD-MD": ("SHORT", 3.0, 3.0), "ST-BD-BU": ("SHORT", 2.1, 3.0),
    "ST-DN-BU": ("SHORT", 1.4, 3.0), "ST-DN-U": ("SHORT", 2.1, 3.0),
    "ST-MU-ST": ("SHORT", 1.4, 3.0), "U-GS-DN": ("SHORT", 3.0, 3.0),
    "U-ST-DN": ("SHORT", 1.4, 3.0),
}


# =======================================================================
# Core functions (from v4/v5)
# =======================================================================
def load_and_classify(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    n = len(df)
    body_abs = np.abs(c - o)
    avg_b20 = pd.Series(body_abs).rolling(20).mean().values
    types = []
    for i in range(n):
        row = pd.Series({'open': o[i], 'high': h[i], 'low': l[i], 'close': c[i]})
        ab = avg_b20[i] if not pd.isna(avg_b20[i]) else 1.0
        types.append(classify_candle(row, ab).value)
    return o, h, l, c, np.array(types), df['timestamp'].values, n


def bt_pattern(indices, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100); slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100); slp = entry * (1 + sl_pct / 100)
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                bo = opens[j]; pnls.append((tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE_PCT)
                break
            elif ht:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hs:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls


def collect_trades_1pos(types, opens, highs, lows, n_bars, pattern_map,
                        start_bar=0, end_bar=None):
    if end_bar is None:
        end_bar = n_bars
    raw = []
    for i in range(max(2, start_bar), end_bar):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
        d, tp, sl = pattern_map[pat]
        eb = i + 1
        if d == 'LONG':
            tpp = entry * (1 + tp / 100); slp = entry * (1 - sl / 100)
        else:
            tpp = entry * (1 - tp / 100); slp = entry * (1 + sl / 100)
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if d == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if d == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                bo = opens[j]; pnl = (tp if abs(tpp - bo) <= abs(slp - bo) else -sl) * LEVERAGE - FEE_PCT
                raw.append((eb, j, pnl, pat)); break
            elif ht:
                raw.append((eb, j, tp * LEVERAGE - FEE_PCT, pat)); break
            elif hs:
                raw.append((eb, j, -sl * LEVERAGE - FEE_PCT, pat)); break
    raw.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl, pat in raw:
        if eb > last_exit:
            filtered.append((eb, xb, pnl, pat))
            last_exit = xb
    return filtered


def portfolio_stats(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0,
                'avg_win': 0, 'avg_loss': 0}
    pnls = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wsum = 0; lsum = 0
    wins_list = []; losses_list = []
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0:
            wins += 1; wsum += p; wins_list.append(p)
        else:
            lsum += abs(p); losses_list.append(p)
    avg_w = float(np.mean(wins_list)) if wins_list else 0
    avg_l = float(np.mean(losses_list)) if losses_list else 0
    return {
        'pnl': round(cum, 1), 'trades': len(pnls),
        'wr': round(wins / len(pnls) * 100, 1),
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_mdd': round(cum / mdd, 1) if mdd > 0 else 999,
        'avg_win': round(avg_w, 3), 'avg_loss': round(avg_l, 3),
    }


def safety_margin(stats):
    """Calculate safety margin: actual WR - break-even WR."""
    aw = stats['avg_win']
    al = abs(stats['avg_loss']) if stats['avg_loss'] != 0 else 1
    if aw + al == 0:
        return 0
    be_wr = al / (aw + al) * 100
    return round(stats['wr'] - be_wr, 1)


def portfolio_mc(trades, n_sims=MC_SIMS):
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t[2] for t in trades])
    real_pnl = np.sum(pnls)
    count = 0
    for _ in range(n_sims):
        shuffled = pnls * np.random.choice([-1, 1], size=len(pnls))
        if np.sum(shuffled) >= real_pnl:
            count += 1
    return count / n_sims


# =======================================================================
# MAIN
# =======================================================================
print("=" * 70)
print("Extended Rediscovery v6: Hybrid Enhancement & Alternative Strategies")
print("=" * 70)
t0 = time.time()

# Load data
print("\nLoading 720d Binance data...", flush=True)
bn_path = DATA_DIR / "btc_5m_720days_binance.csv"
opens, highs, lows, closes, types, timestamps, n_bars = load_and_classify(bn_path)

overlap_start_ts = np.datetime64('2025-05-05T15:00:00')
overlap_end_ts = np.datetime64('2026-01-30T14:55:00')
overlap_start_bar = int(np.searchsorted(timestamps, overlap_start_ts))
overlap_end_bar = int(np.searchsorted(timestamps, overlap_end_ts, side='right'))

print(f"Total: {n_bars} bars, Pre-overlap: {overlap_start_bar}, In-sample: {overlap_end_bar - overlap_start_bar}")

# Build pattern indices for pre-overlap and in-sample
pre_indices = defaultdict(list)
is_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    if i < overlap_start_bar:
        pre_indices[pat].append(i)
    elif i < overlap_end_bar:
        is_indices[pat].append(i)
for k in pre_indices:
    pre_indices[k] = np.array(pre_indices[k])
for k in is_indices:
    is_indices[k] = np.array(is_indices[k])

results = {}

# =======================================================================
# Phase 1: Systematic TP/SL Transform Sweep
# =======================================================================
print("\n" + "=" * 70)
print("Phase 1: Systematic TP/SL Transform Sweep")
print("=" * 70, flush=True)

tp_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sl_scales = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

results['phase1'] = {'grid': [], 'best_pre_pnl_mdd': None, 'best_pre_safety': None}

print(f"\n{'TP_sc':>6} {'SL_sc':>6} │ {'Pre_PnL':>8} {'Pre_MDD':>8} {'P/M':>6} "
      f"{'Pre_WR':>7} {'Safety':>7} │ {'IS_PnL':>8} {'IS_MDD':>8} {'IS_P/M':>7}", flush=True)
print("-" * 90)

best_pre_pm = {'pnl_mdd': 0}
best_pre_safety = {'margin': -100}

for tp_sc in tp_scales:
    for sl_sc in sl_scales:
        # Build transformed pattern map
        pmap = {}
        for pat, (d, tp, sl) in PATTERNS_CURRENT.items():
            new_tp = round(tp * tp_sc, 2)
            new_sl = min(round(sl * sl_sc, 2), 3.0)
            if new_tp < 0.1:
                new_tp = 0.1
            pmap[pat] = (d, new_tp, new_sl)

        # Test on pre-overlap and in-sample
        pre_trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap,
                                         0, overlap_start_bar)
        is_trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap,
                                        overlap_start_bar, overlap_end_bar)
        pre_s = portfolio_stats(pre_trades)
        is_s = portfolio_stats(is_trades)
        pre_margin = safety_margin(pre_s)
        is_margin = safety_margin(is_s)

        row = {
            'tp_scale': tp_sc, 'sl_scale': sl_sc,
            'pre': pre_s, 'pre_margin': pre_margin,
            'is': is_s, 'is_margin': is_margin,
        }
        results['phase1']['grid'].append(row)

        flag = ""
        if pre_s['pnl'] > 0 and pre_s['pnl_mdd'] > best_pre_pm.get('pnl_mdd', 0):
            best_pre_pm = {'tp_scale': tp_sc, 'sl_scale': sl_sc,
                           'pnl_mdd': pre_s['pnl_mdd'], **pre_s, 'margin': pre_margin}
            flag = " ★PM"
        if pre_margin > best_pre_safety.get('margin', -100) and pre_s['pnl'] > 0:
            best_pre_safety = {'tp_scale': tp_sc, 'sl_scale': sl_sc,
                               'margin': pre_margin, **pre_s}
            flag += " ★SF"

        print(f"{tp_sc:>6.1f} {sl_sc:>6.1f} │ {pre_s['pnl']:>+7.1f}% {pre_s['mdd']:>7.1f}% "
              f"{pre_s['pnl_mdd']:>5.1f}x {pre_s['wr']:>6.1f}% {pre_margin:>+6.1f}pp │ "
              f"{is_s['pnl']:>+7.1f}% {is_s['mdd']:>7.1f}% {is_s['pnl_mdd']:>6.1f}x{flag}",
              flush=True)

results['phase1']['best_pre_pnl_mdd'] = best_pre_pm
results['phase1']['best_pre_safety'] = best_pre_safety

print(f"\n★ Best Pre PnL/MDD: TP×{best_pre_pm.get('tp_scale','?')} SL×{best_pre_pm.get('sl_scale','?')} "
      f"→ PnL/MDD {best_pre_pm.get('pnl_mdd', 0):.1f}x, Safety {best_pre_pm.get('margin', 0):+.1f}pp")
print(f"★ Best Pre Safety:  TP×{best_pre_safety.get('tp_scale','?')} SL×{best_pre_safety.get('sl_scale','?')} "
      f"→ Safety {best_pre_safety.get('margin', 0):+.1f}pp, PnL {best_pre_safety.get('pnl', 0):+.1f}%")

# =======================================================================
# Phase 2: Per-Pattern Pre-overlap TP/SL Grid Search (relaxed)
# =======================================================================
print("\n" + "=" * 70)
print("Phase 2: Per-Pattern Pre-overlap TP/SL Search (relaxed criteria)")
print("=" * 70, flush=True)

results['phase2'] = {}
qualifying = {}
no_qualifying = []

for pat, (d, cur_tp, cur_sl) in PATTERNS_CURRENT.items():
    idx = pre_indices.get(pat, np.array([]))
    if len(idx) < 3:
        no_qualifying.append(pat)
        results['phase2'][pat] = {'status': 'NO_DATA', 'pre_trades': len(idx)}
        continue

    best = None
    best_score = -999
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            pnls = bt_pattern(idx, d, tp, sl, opens, highs, lows, n_bars)
            n = len(pnls)
            if n < 5:
                continue
            total = sum(pnls)
            if total <= 0:
                continue
            wr = sum(1 for x in pnls if x > 0) / n * 100
            rw_wr = sl / (tp + sl) * 100
            edge = wr - rw_wr
            if edge < 3.0:  # Relaxed: 3pp instead of 5pp
                continue
            # Score: PnL/trade (normalized)
            score = total / n
            if score > best_score:
                best_score = score
                best = {'tp': tp, 'sl': sl, 'pnl': round(total, 1), 'wr': round(wr, 1),
                        'trades': n, 'edge': round(edge, 1), 'pnl_per_trade': round(score, 3)}

    # Also check current TP/SL on pre-overlap
    cur_pnls = bt_pattern(idx, d, cur_tp, cur_sl, opens, highs, lows, n_bars)
    cur_total = sum(cur_pnls) if cur_pnls else 0
    cur_wr = sum(1 for x in cur_pnls if x > 0) / len(cur_pnls) * 100 if cur_pnls else 0

    if best:
        qualifying[pat] = (d, best['tp'], best['sl'])
        results['phase2'][pat] = {
            'status': 'FOUND',
            'current_pre_pnl': round(cur_total, 1),
            'current_pre_wr': round(cur_wr, 1),
            'best': best,
        }
    else:
        no_qualifying.append(pat)
        results['phase2'][pat] = {
            'status': 'NO_QUALIFYING',
            'current_pre_pnl': round(cur_total, 1),
            'current_pre_wr': round(cur_wr, 1),
        }

print(f"\nQualifying patterns: {len(qualifying)}/51")
print(f"No qualifying TP/SL: {len(no_qualifying)}/51")

# Print qualifying patterns
print(f"\n{'Pattern':<14} {'Dir':<6} {'Cur TP/SL':>10} {'Pre PnL Cur':>12} │ "
      f"{'Best TP/SL':>10} {'Pre PnL Best':>13} {'Edge':>6}", flush=True)
print("-" * 85)
for pat in sorted(qualifying.keys()):
    d, tp, sl = qualifying[pat]
    r = results['phase2'][pat]
    cur_d, cur_tp, cur_sl = PATTERNS_CURRENT[pat]
    print(f"{pat:<14} {d:<6} {cur_tp:>4.2f}/{cur_sl:<4.2f} {r['current_pre_pnl']:>+11.1f}% │ "
          f"{tp:>4.2f}/{sl:<4.2f} {r['best']['pnl']:>+12.1f}% {r['best']['edge']:>5.1f}pp",
          flush=True)

# Build portfolio with qualifying pre-overlap TP/SL
pre_opt_map = dict(PATTERNS_CURRENT)  # start with current
for pat, val in qualifying.items():
    pre_opt_map[pat] = val

# =======================================================================
# Phase 3: Cross-Period Intersection TP/SL
# =======================================================================
print("\n" + "=" * 70)
print("Phase 3: Cross-Period Intersection TP/SL")
print("=" * 70, flush=True)

results['phase3'] = {}
intersection = {}
EDGE_RELAX = 3.0  # pp

for pat, (d, cur_tp, cur_sl) in PATTERNS_CURRENT.items():
    pre_idx = pre_indices.get(pat, np.array([]))
    is_idx = is_indices.get(pat, np.array([]))

    if len(pre_idx) < 3 or len(is_idx) < 3:
        results['phase3'][pat] = {'status': 'NO_DATA'}
        continue

    # Find TP/SL good on BOTH periods
    good_combos = []
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            # Pre-overlap check
            pre_pnls = bt_pattern(pre_idx, d, tp, sl, opens, highs, lows, n_bars)
            if len(pre_pnls) < 3 or sum(pre_pnls) <= 0:
                continue
            pre_wr = sum(1 for x in pre_pnls if x > 0) / len(pre_pnls) * 100
            pre_edge = pre_wr - sl / (tp + sl) * 100
            if pre_edge < EDGE_RELAX:
                continue

            # In-sample check
            is_pnls = bt_pattern(is_idx, d, tp, sl, opens, highs, lows, n_bars)
            if len(is_pnls) < 3 or sum(is_pnls) <= 0:
                continue
            is_wr = sum(1 for x in is_pnls if x > 0) / len(is_pnls) * 100
            is_edge = is_wr - sl / (tp + sl) * 100
            if is_edge < EDGE_RELAX:
                continue

            # Score: average of pre and is PnL/trade
            avg_score = (sum(pre_pnls) / len(pre_pnls) + sum(is_pnls) / len(is_pnls)) / 2
            good_combos.append({
                'tp': tp, 'sl': sl,
                'pre_pnl': round(sum(pre_pnls), 1), 'pre_wr': round(pre_wr, 1),
                'is_pnl': round(sum(is_pnls), 1), 'is_wr': round(is_wr, 1),
                'avg_score': round(avg_score, 3),
                'pre_edge': round(pre_edge, 1), 'is_edge': round(is_edge, 1),
            })

    if good_combos:
        # Pick best by avg_score
        best = max(good_combos, key=lambda x: x['avg_score'])
        intersection[pat] = (d, best['tp'], best['sl'])
        results['phase3'][pat] = {
            'status': 'FOUND',
            'n_combos': len(good_combos),
            'best': best,
        }
    else:
        results['phase3'][pat] = {'status': 'NO_INTERSECTION', 'n_combos': 0}

print(f"\nIntersection patterns: {len(intersection)}/51")

print(f"\n{'Pattern':<14} {'Dir':<6} {'Cur TP/SL':>10} │ {'Int TP/SL':>10} "
      f"{'Pre PnL':>8} {'IS PnL':>8} {'Combos':>7}", flush=True)
print("-" * 75)
for pat in sorted(intersection.keys()):
    d, tp, sl = intersection[pat]
    r = results['phase3'][pat]
    cur_d, cur_tp, cur_sl = PATTERNS_CURRENT[pat]
    print(f"{pat:<14} {d:<6} {cur_tp:>4.2f}/{cur_sl:<4.2f} │ {tp:>4.2f}/{sl:<4.2f} "
          f"{r['best']['pre_pnl']:>+7.1f}% {r['best']['is_pnl']:>+7.1f}% {r['n_combos']:>7}",
          flush=True)

# =======================================================================
# Phase 4: Universal TP/SL Grid Search
# =======================================================================
print("\n" + "=" * 70)
print("Phase 4: Universal TP/SL Grid Search")
print("=" * 70, flush=True)

results['phase4'] = {'grid': [], 'best_pre_pnl_mdd': None}
best_uni = {'pnl_mdd': 0}

print(f"\n{'TP':>5} {'SL':>5} │ {'Pre_PnL':>8} {'Pre_MDD':>8} {'P/M':>6} "
      f"{'Pre_WR':>7} {'Safety':>7} │ {'IS_PnL':>8} {'IS_MDD':>8}", flush=True)
print("-" * 80)

for tp in TP_SL_GRID:
    for sl in TP_SL_GRID:
        pmap = {pat: (d, tp, sl) for pat, (d, _, _) in PATTERNS_CURRENT.items()}

        pre_trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap,
                                         0, overlap_start_bar)
        is_trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap,
                                        overlap_start_bar, overlap_end_bar)
        pre_s = portfolio_stats(pre_trades)
        is_s = portfolio_stats(is_trades)
        pre_margin = safety_margin(pre_s)

        row = {'tp': tp, 'sl': sl, 'pre': pre_s, 'pre_margin': pre_margin, 'is': is_s}
        results['phase4']['grid'].append(row)

        flag = ""
        if pre_s['pnl'] > 0 and pre_s['pnl_mdd'] > best_uni.get('pnl_mdd', 0):
            best_uni = {'tp': tp, 'sl': sl, 'pnl_mdd': pre_s['pnl_mdd'],
                        **pre_s, 'margin': pre_margin}
            flag = " ★"

        if pre_s['pnl'] > 0:
            print(f"{tp:>5.2f} {sl:>5.2f} │ {pre_s['pnl']:>+7.1f}% {pre_s['mdd']:>7.1f}% "
                  f"{pre_s['pnl_mdd']:>5.1f}x {pre_s['wr']:>6.1f}% {pre_margin:>+6.1f}pp │ "
                  f"{is_s['pnl']:>+7.1f}% {is_s['mdd']:>7.1f}%{flag}", flush=True)

results['phase4']['best_pre_pnl_mdd'] = best_uni
print(f"\n★ Best Universal: TP {best_uni.get('tp','?')}/SL {best_uni.get('sl','?')} "
      f"→ Pre PnL/MDD {best_uni.get('pnl_mdd', 0):.1f}x, Safety {best_uni.get('margin', 0):+.1f}pp")

# =======================================================================
# Phase 5: Best Hybrid Assembly & Full Comparison
# =======================================================================
print("\n" + "=" * 70)
print("Phase 5: Best Hybrid Assembly & Full Comparison")
print("=" * 70, flush=True)

# Build scenario maps
scenarios = {}

# A. Baseline
scenarios['A_BASELINE'] = dict(PATTERNS_CURRENT)

# B. Systematic Transform (best from Phase 1 by PnL/MDD)
tp_sc_pm = best_pre_pm.get('tp_scale', 1.0)
sl_sc_pm = best_pre_pm.get('sl_scale', 1.0)
b_map = {}
for pat, (d, tp, sl) in PATTERNS_CURRENT.items():
    b_map[pat] = (d, round(max(tp * tp_sc_pm, 0.1), 2), min(round(sl * sl_sc_pm, 2), 3.0))
scenarios[f'B_TRANSFORM_TP{tp_sc_pm}_SL{sl_sc_pm}'] = b_map

# C. Systematic Transform (best from Phase 1 by Safety)
tp_sc_sf = best_pre_safety.get('tp_scale', 1.0)
sl_sc_sf = best_pre_safety.get('sl_scale', 1.0)
c_map = {}
for pat, (d, tp, sl) in PATTERNS_CURRENT.items():
    c_map[pat] = (d, round(max(tp * tp_sc_sf, 0.1), 2), min(round(sl * sl_sc_sf, 2), 3.0))
scenarios[f'C_TRANSFORM_TP{tp_sc_sf}_SL{sl_sc_sf}'] = c_map

# D. Pre-overlap Optimized (Phase 2 qualifying, rest current)
scenarios['D_PRE_OPT_HYBRID'] = dict(pre_opt_map)

# E. Intersection Hybrid (Phase 3 intersection, rest current)
e_map = dict(PATTERNS_CURRENT)
for pat, val in intersection.items():
    e_map[pat] = val
scenarios['E_INTERSECTION'] = e_map

# F. Universal TP/SL (Phase 4 best)
uni_tp = best_uni.get('tp', 1.0)
uni_sl = best_uni.get('sl', 3.0)
f_map = {pat: (d, uni_tp, uni_sl) for pat, (d, _, _) in PATTERNS_CURRENT.items()}
scenarios[f'F_UNIVERSAL_{uni_tp}_{uni_sl}'] = f_map

# G. Intersection + Transform fallback
# For patterns with intersection → use intersection TP/SL
# For patterns without → use best systematic transform
g_map = {}
for pat, (d, tp, sl) in PATTERNS_CURRENT.items():
    if pat in intersection:
        g_map[pat] = intersection[pat]
    else:
        g_map[pat] = (d, round(max(tp * tp_sc_pm, 0.1), 2), min(round(sl * sl_sc_pm, 2), 3.0))
scenarios['G_INTERSECT_PLUS_TRANSFORM'] = g_map

# Test all on pre-overlap, in-sample, full
print(f"\n{'Scenario':<30} {'Period':<10} {'PnL':>8} {'Trades':>7} {'WR':>6} "
      f"{'MDD':>7} {'P/M':>6} {'Safety':>7}", flush=True)
print("-" * 90)

results['phase5'] = {}
periods = {
    'pre': (0, overlap_start_bar),
    'is': (overlap_start_bar, overlap_end_bar),
    'full': (0, n_bars),
}

for sname, pmap in scenarios.items():
    results['phase5'][sname] = {}
    for pname, (sb, eb) in periods.items():
        trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap, sb, eb)
        stats = portfolio_stats(trades)
        margin = safety_margin(stats)
        results['phase5'][sname][pname] = {**stats, 'safety_margin': margin}
        print(f"{sname:<30} {pname:<10} {stats['pnl']:>+7.1f}% {stats['trades']:>7} "
              f"{stats['wr']:>5.1f}% {stats['mdd']:>6.1f}% {stats['pnl_mdd']:>5.1f}x "
              f"{margin:>+6.1f}pp", flush=True)
    print()

# Portfolio MC for top scenarios
print("\nPortfolio MC (full 720d):", flush=True)
results['phase5_mc'] = {}
for sname, pmap in scenarios.items():
    trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap, 0, n_bars)
    p = portfolio_mc(trades)
    results['phase5_mc'][sname] = round(p, 4)
    print(f"  {sname:<30} MC p={p:.4f} {'PASS' if p < 0.01 else 'FAIL'}", flush=True)

# =======================================================================
# Phase 6: Walk-Forward Validation of Top Candidates
# =======================================================================
print("\n" + "=" * 70)
print("Phase 6: Walk-Forward Validation (3 folds)")
print("=" * 70, flush=True)

bars_per_180d = 180 * 288
fold_bounds = [
    (0, bars_per_180d, bars_per_180d, 2 * bars_per_180d),
    (0, 2 * bars_per_180d, 2 * bars_per_180d, 3 * bars_per_180d),
    (0, 3 * bars_per_180d, 3 * bars_per_180d, min(4 * bars_per_180d, n_bars)),
]

results['phase6'] = {}
for sname, pmap in scenarios.items():
    folds = []
    total_pnl = 0
    for fi, (ts, te, vs, ve) in enumerate(fold_bounds, 1):
        if ve > n_bars:
            ve = n_bars
        trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap, vs, ve)
        s = portfolio_stats(trades)
        m = safety_margin(s)
        folds.append({'fold': fi, **s, 'safety': m})
        total_pnl += s['pnl']

    n_pos = sum(1 for f in folds if f['pnl'] > 0)
    results['phase6'][sname] = {
        'folds': folds, 'total_oos_pnl': round(total_pnl, 1),
        'folds_positive': n_pos,
    }
    print(f"\n{sname}:", flush=True)
    for f in folds:
        print(f"  F{f['fold']}: PnL {f['pnl']:>+8.1f}%, WR {f['wr']:>5.1f}%, "
              f"MDD {f['mdd']:>5.1f}%, Safety {f['safety']:>+5.1f}pp", flush=True)
    print(f"  TOTAL: {total_pnl:>+8.1f}%, {n_pos}/3 positive", flush=True)

# =======================================================================
# FINAL RANKING
# =======================================================================
print("\n" + "=" * 70)
print("FINAL RANKING")
print("=" * 70, flush=True)

# Rank by pre-overlap safety margin (primary) then PnL/MDD (secondary)
ranking_data = []
for sname in scenarios:
    pre = results['phase5'][sname]['pre']
    full = results['phase5'][sname]['full']
    wf = results['phase6'][sname]
    mc = results['phase5_mc'].get(sname, 1.0)
    ranking_data.append({
        'scenario': sname,
        'pre_safety': pre['safety_margin'],
        'pre_pnl_mdd': pre['pnl_mdd'],
        'pre_pnl': pre['pnl'],
        'full_pnl': full['pnl'],
        'full_mdd': full['mdd'],
        'full_pnl_mdd': full['pnl_mdd'],
        'wf_total': wf['total_oos_pnl'],
        'wf_positive': wf['folds_positive'],
        'mc': mc,
    })

# Sort: safety margin desc, then pre_pnl_mdd desc
ranking_data.sort(key=lambda x: (x['pre_safety'], x['pre_pnl_mdd']), reverse=True)
results['ranking'] = ranking_data

print(f"\n{'Rank':>4} {'Scenario':<30} {'Pre Safety':>10} {'Pre P/M':>8} "
      f"{'720d PnL':>9} {'720d MDD':>9} {'WF Total':>9} {'MC':>6}", flush=True)
print("-" * 100)

for i, r in enumerate(ranking_data, 1):
    mc_str = "PASS" if r['mc'] < 0.01 else "FAIL"
    print(f"{i:>4} {r['scenario']:<30} {r['pre_safety']:>+9.1f}pp {r['pre_pnl_mdd']:>7.1f}x "
          f"{r['full_pnl']:>+8.1f}% {r['full_mdd']:>8.1f}% {r['wf_total']:>+8.1f}% "
          f"{mc_str:>6}", flush=True)

# Show baseline comparison for top candidate
if ranking_data:
    top = ranking_data[0]
    base = next(r for r in ranking_data if r['scenario'] == 'A_BASELINE')
    print(f"\n--- Top vs Baseline ---")
    print(f"Safety margin:  {base['pre_safety']:+.1f}pp → {top['pre_safety']:+.1f}pp "
          f"(Δ {top['pre_safety'] - base['pre_safety']:+.1f}pp)")
    print(f"Pre PnL/MDD:    {base['pre_pnl_mdd']:.1f}x → {top['pre_pnl_mdd']:.1f}x "
          f"(Δ {top['pre_pnl_mdd'] - base['pre_pnl_mdd']:+.1f}x)")
    print(f"720d PnL:       {base['full_pnl']:+.1f}% → {top['full_pnl']:+.1f}% "
          f"(Δ {top['full_pnl'] - base['full_pnl']:+.1f}%)")
    print(f"720d MDD:       {base['full_mdd']:.1f}% → {top['full_mdd']:.1f}% "
          f"(Δ {top['full_mdd'] - base['full_mdd']:+.1f}%)")

elapsed = time.time() - t0
results['metadata'] = {
    'script': 'extended_rediscovery_v6_hybrid_enhancement.py',
    'data': str(bn_path),
    'n_bars': n_bars,
    'elapsed_seconds': round(elapsed, 1),
}

out_path = RESULTS_DIR / "extended_rediscovery_v6_hybrid_enhancement.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print(f"Elapsed: {elapsed:.1f}s")
