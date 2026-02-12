#!/usr/bin/env python3
"""
Extended Rediscovery v4: 720-day Walk-Forward & Robust TP/SL
============================================================
v3 confirmed: Pattern edge is REAL (pre-overlap +235%, 8/9 quarters positive).
But in-sample WR 84.9% >> pre-overlap WR 64.6% — TP/SL overfit?

Phase 1: Per-Pattern Pre-overlap Breakdown (which patterns drive the WR gap?)
Phase 2: Pre-overlap TP/SL Grid Search (optimize on 437d OOS data)
Phase 3: Cross-Validation Matrix (4 scenarios: current vs pre-overlap optimal)
Phase 4: 720d Expanding Window WF (3 folds, realistic forward performance)
Phase 5: Look-ahead Decomposition Revision (pattern selection vs TP/SL vs genuine)
Phase 6: Robust TP/SL Identification (profitable across ALL periods)
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

# Import production classify_candle (CRITICAL — v1/v2 bug was using custom function)
sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
MC_THRESHOLD = 0.01
EDGE_THRESHOLD = 5  # pp excess WR over random walk baseline
MIN_TRADES = 10

# v1.27.3 — 51 patterns with CURRENT TP/SL (270d-optimized)
PATTERNS = {
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


# ============================================================
# Data loading & classification
# ============================================================
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


# ============================================================
# Core backtest functions (reference protocol)
# ============================================================
def bt_pattern(indices, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest a single pattern/direction/TP/SL. Returns list of PnLs."""
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


def collect_trades_1pos(types, opens, highs, lows, n_bars, pattern_map,
                        start_bar=0, end_bar=None):
    """1-position-at-a-time portfolio backtest. Returns list of (entry, exit, pnl, pat)."""
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
                pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
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
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0}
    pnls = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wsum = 0; lsum = 0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0:
            wins += 1; wsum += p
        else:
            lsum += abs(p)
    return {
        'pnl': round(cum, 1), 'trades': len(pnls),
        'wr': round(wins / len(pnls) * 100, 1),
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_mdd': round(cum / mdd, 1) if mdd > 0 else 999,
    }


def grid_search_pattern(indices, direction, opens, highs, lows, n_bars):
    """Grid search TP/SL for a pattern. Returns best (tp, sl, pnl, wr, mc, trades, edge)."""
    best = None
    rw_baseline = {}  # cache random walk WR per TP/SL combo
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            pnls = bt_pattern(indices, direction, tp, sl, opens, highs, lows, n_bars)
            n = len(pnls)
            if n < MIN_TRADES:
                continue
            total = sum(pnls)
            if total <= 0:
                continue
            wr = sum(1 for x in pnls if x > 0) / n * 100
            rw_wr = sl / (tp + sl) * 100
            edge = wr - rw_wr
            if edge < EDGE_THRESHOLD:
                continue
            mc = mc_test(pnls, 3000)  # quick screen
            if mc >= 0.03:
                continue
            mc = mc_test(pnls, MC_SIMS)  # full test
            if mc >= MC_THRESHOLD:
                continue
            score = total
            if best is None or score > best[2]:
                best = (tp, sl, total, wr, mc, n, edge)
    return best


# ============================================================
print("=" * 70)
print("Extended Rediscovery v4: 720-day Walk-Forward & Robust TP/SL")
print("=" * 70)
t0 = time.time()

# Load data
print("\nPhase 0: Loading 720d Binance data...", flush=True)
bn_path = DATA_DIR / "btc_5m_720days_binance.csv"
opens, highs, lows, closes, types, timestamps, n_bars = load_and_classify(bn_path)

# Define period boundaries (bar indices)
# Pre-overlap: 2024-02-23 ~ 2025-05-05 (437d)
# In-sample:   2025-05-05 ~ 2026-01-30 (269d)
# Post-overlap: 2026-01-30 ~ 2026-02-12 (12d)
overlap_start_ts = np.datetime64('2025-05-05T15:00:00')
overlap_end_ts = np.datetime64('2026-01-30T14:55:00')
overlap_start_bar = int(np.searchsorted(timestamps, overlap_start_ts))
overlap_end_bar = int(np.searchsorted(timestamps, overlap_end_ts, side='right'))

print(f"Total bars: {n_bars}")
print(f"Pre-overlap: bars 0-{overlap_start_bar} ({overlap_start_bar} bars)")
print(f"In-sample:   bars {overlap_start_bar}-{overlap_end_bar} ({overlap_end_bar - overlap_start_bar} bars)")
print(f"Post-overlap: bars {overlap_end_bar}-{n_bars} ({n_bars - overlap_end_bar} bars)")

# Build pattern indices for pre-overlap and in-sample
pre_indices = defaultdict(list)
insample_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    if i < overlap_start_bar:
        pre_indices[pat].append(i)
    elif i < overlap_end_bar:
        insample_indices[pat].append(i)
for k in pre_indices:
    pre_indices[k] = np.array(pre_indices[k])
for k in insample_indices:
    insample_indices[k] = np.array(insample_indices[k])

results = {}

# ============================================================
# Phase 1: Per-Pattern Pre-overlap vs In-sample Breakdown
# ============================================================
print("\n" + "=" * 70)
print("Phase 1: Per-Pattern Pre-overlap vs In-sample Breakdown")
print("=" * 70, flush=True)

print(f"\n{'Pattern':<14} {'Dir':<6} {'Pre_Tr':>6} {'Pre_WR':>7} {'Pre_PnL':>8}  "
      f"{'IS_Tr':>6} {'IS_WR':>7} {'IS_PnL':>8}  {'WR_Gap':>7}", flush=True)
print("-" * 85)

phase1 = {}
pre_total_trades = 0; pre_total_wins = 0; pre_total_pnl = 0
is_total_trades = 0; is_total_wins = 0; is_total_pnl = 0

for pat, (d, tp, sl) in sorted(PATTERNS.items()):
    # Pre-overlap
    pre_idx = pre_indices.get(pat, np.array([], dtype=int))
    pre_pnls = bt_pattern(pre_idx, d, tp, sl, opens, highs, lows, n_bars) if len(pre_idx) > 0 else []
    pre_n = len(pre_pnls)
    pre_wr = sum(1 for x in pre_pnls if x > 0) / pre_n * 100 if pre_n > 0 else 0
    pre_pnl = sum(pre_pnls)

    # In-sample
    is_idx = insample_indices.get(pat, np.array([], dtype=int))
    is_pnls = bt_pattern(is_idx, d, tp, sl, opens, highs, lows, n_bars) if len(is_idx) > 0 else []
    is_n = len(is_pnls)
    is_wr = sum(1 for x in is_pnls if x > 0) / is_n * 100 if is_n > 0 else 0
    is_pnl = sum(is_pnls)

    wr_gap = is_wr - pre_wr if pre_n > 0 and is_n > 0 else float('nan')

    phase1[pat] = {
        'direction': d, 'tp': tp, 'sl': sl,
        'pre_trades': pre_n, 'pre_wr': round(pre_wr, 1), 'pre_pnl': round(pre_pnl, 1),
        'is_trades': is_n, 'is_wr': round(is_wr, 1), 'is_pnl': round(is_pnl, 1),
        'wr_gap': round(wr_gap, 1) if not np.isnan(wr_gap) else None,
    }

    pre_total_trades += pre_n; pre_total_wins += sum(1 for x in pre_pnls if x > 0)
    pre_total_pnl += pre_pnl
    is_total_trades += is_n; is_total_wins += sum(1 for x in is_pnls if x > 0)
    is_total_pnl += is_pnl

    wr_gap_str = f"{wr_gap:+.1f}pp" if not np.isnan(wr_gap) else "  N/A"
    print(f"{pat:<14} {d:<6} {pre_n:>6} {pre_wr:>6.1f}% {pre_pnl:>+8.1f}  "
          f"{is_n:>6} {is_wr:>6.1f}% {is_pnl:>+8.1f}  {wr_gap_str}", flush=True)

pre_wr_total = pre_total_wins / pre_total_trades * 100 if pre_total_trades > 0 else 0
is_wr_total = is_total_wins / is_total_trades * 100 if is_total_trades > 0 else 0
print(f"\n{'TOTAL':<14} {'':6} {pre_total_trades:>6} {pre_wr_total:>6.1f}% {pre_total_pnl:>+8.1f}  "
      f"{is_total_trades:>6} {is_wr_total:>6.1f}% {is_total_pnl:>+8.1f}  "
      f"{is_wr_total - pre_wr_total:+.1f}pp")

# Count patterns with negative pre-overlap PnL
pre_neg = sum(1 for v in phase1.values() if v['pre_trades'] > 0 and v['pre_pnl'] < 0)
pre_pos = sum(1 for v in phase1.values() if v['pre_trades'] > 0 and v['pre_pnl'] > 0)
pre_zero = sum(1 for v in phase1.values() if v['pre_trades'] == 0)
print(f"\nPre-overlap: {pre_pos} positive, {pre_neg} negative, {pre_zero} no trades")
results['phase1'] = phase1

# ============================================================
# Phase 2: Pre-overlap TP/SL Grid Search
# ============================================================
print("\n" + "=" * 70)
print("Phase 2: Pre-overlap TP/SL Grid Search (437d)")
print("=" * 70, flush=True)

phase2 = {}
improved = 0; unchanged = 0; failed = 0

for pat, (d, cur_tp, cur_sl) in sorted(PATTERNS.items()):
    pre_idx = pre_indices.get(pat, np.array([], dtype=int))
    if len(pre_idx) < MIN_TRADES:
        phase2[pat] = {'status': 'INSUFFICIENT_DATA', 'pre_trades': len(pre_idx)}
        failed += 1
        continue

    best = grid_search_pattern(pre_idx, d, opens, highs, lows, n_bars)
    if best is None:
        # No qualifying TP/SL found on pre-overlap
        phase2[pat] = {'status': 'NO_QUALIFYING_TPSL', 'pre_trades': len(pre_idx)}
        failed += 1
        continue

    new_tp, new_sl, new_pnl, new_wr, new_mc, new_n, new_edge = best
    # Compare with current
    cur_pnls = bt_pattern(pre_idx, d, cur_tp, cur_sl, opens, highs, lows, n_bars)
    cur_pnl = sum(cur_pnls)
    cur_wr_val = sum(1 for x in cur_pnls if x > 0) / len(cur_pnls) * 100 if cur_pnls else 0

    if new_pnl > cur_pnl * 1.1:  # >10% improvement
        status = 'IMPROVED'
        improved += 1
    else:
        status = 'CURRENT_OK'
        unchanged += 1

    phase2[pat] = {
        'status': status,
        'current': {'tp': cur_tp, 'sl': cur_sl, 'pnl': round(cur_pnl, 1),
                    'wr': round(cur_wr_val, 1), 'trades': len(cur_pnls)},
        'pre_optimal': {'tp': new_tp, 'sl': new_sl, 'pnl': round(new_pnl, 1),
                        'wr': round(new_wr, 1), 'mc': round(new_mc, 4),
                        'trades': new_n, 'edge': round(new_edge, 1)},
    }

print(f"\nGrid search results: {improved} IMPROVED, {unchanged} CURRENT_OK, {failed} FAILED")
print(f"\nTop improvements (pre-overlap PnL gain):", flush=True)
print(f"{'Pattern':<14} {'Cur TP/SL':>10} {'Cur PnL':>8} {'New TP/SL':>10} {'New PnL':>8} {'Gain':>8}")
print("-" * 68)

for pat in sorted(phase2.keys(),
                  key=lambda p: (phase2[p].get('pre_optimal', {}).get('pnl', 0) -
                                 phase2[p].get('current', {}).get('pnl', 0)),
                  reverse=True)[:15]:
    info = phase2[pat]
    if info['status'] not in ('IMPROVED', 'CURRENT_OK'):
        continue
    cur = info['current']; opt = info['pre_optimal']
    gain = opt['pnl'] - cur['pnl']
    print(f"{pat:<14} {cur['tp']:>4}/{cur['sl']:<4}  {cur['pnl']:>+7.1f} "
          f"{opt['tp']:>4}/{opt['sl']:<4}  {opt['pnl']:>+7.1f} {gain:>+7.1f}")

results['phase2'] = phase2

# ============================================================
# Phase 3: Cross-Validation Matrix
# ============================================================
print("\n" + "=" * 70)
print("Phase 3: Cross-Validation Matrix")
print("=" * 70, flush=True)

# Build pre-overlap optimal pattern map
pre_optimal_map = {}
for pat, (d, cur_tp, cur_sl) in PATTERNS.items():
    info = phase2.get(pat, {})
    if info.get('status') in ('IMPROVED', 'CURRENT_OK') and 'pre_optimal' in info:
        opt = info['pre_optimal']
        pre_optimal_map[pat] = (d, opt['tp'], opt['sl'])
    # Patterns that failed grid search are excluded from pre-optimal portfolio

# Scenario A: Current TP/SL → Pre-overlap
tr_a = collect_trades_1pos(types, opens, highs, lows, n_bars, PATTERNS,
                           start_bar=0, end_bar=overlap_start_bar)
st_a = portfolio_stats(tr_a)
print(f"\nA: Current TP/SL → Pre-overlap:    {st_a['trades']:>4} trades, "
      f"WR {st_a['wr']:>5.1f}%, PnL {st_a['pnl']:>+8.1f}%, MDD {st_a['mdd']:>5.1f}%")

# Scenario B: Current TP/SL → In-sample
tr_b = collect_trades_1pos(types, opens, highs, lows, n_bars, PATTERNS,
                           start_bar=overlap_start_bar, end_bar=overlap_end_bar)
st_b = portfolio_stats(tr_b)
print(f"B: Current TP/SL → In-sample:      {st_b['trades']:>4} trades, "
      f"WR {st_b['wr']:>5.1f}%, PnL {st_b['pnl']:>+8.1f}%, MDD {st_b['mdd']:>5.1f}%")

# Scenario C: Pre-overlap optimal → In-sample (REVERSE VALIDATION)
tr_c = collect_trades_1pos(types, opens, highs, lows, n_bars, pre_optimal_map,
                           start_bar=overlap_start_bar, end_bar=overlap_end_bar)
st_c = portfolio_stats(tr_c)
print(f"C: Pre-opt TP/SL → In-sample:      {st_c['trades']:>4} trades, "
      f"WR {st_c['wr']:>5.1f}%, PnL {st_c['pnl']:>+8.1f}%, MDD {st_c['mdd']:>5.1f}%")

# Scenario D: Pre-overlap optimal → Pre-overlap (in-sample check)
tr_d = collect_trades_1pos(types, opens, highs, lows, n_bars, pre_optimal_map,
                           start_bar=0, end_bar=overlap_start_bar)
st_d = portfolio_stats(tr_d)
print(f"D: Pre-opt TP/SL → Pre-overlap:    {st_d['trades']:>4} trades, "
      f"WR {st_d['wr']:>5.1f}%, PnL {st_d['pnl']:>+8.1f}%, MDD {st_d['mdd']:>5.1f}%")

# Scenario E: Current TP/SL → Full 720d
tr_e = collect_trades_1pos(types, opens, highs, lows, n_bars, PATTERNS)
st_e = portfolio_stats(tr_e)
print(f"E: Current TP/SL → Full 720d:      {st_e['trades']:>4} trades, "
      f"WR {st_e['wr']:>5.1f}%, PnL {st_e['pnl']:>+8.1f}%, MDD {st_e['mdd']:>5.1f}%")

# Scenario F: Pre-overlap optimal → Full 720d
tr_f = collect_trades_1pos(types, opens, highs, lows, n_bars, pre_optimal_map)
st_f = portfolio_stats(tr_f)
print(f"F: Pre-opt TP/SL → Full 720d:      {st_f['trades']:>4} trades, "
      f"WR {st_f['wr']:>5.1f}%, PnL {st_f['pnl']:>+8.1f}%, MDD {st_f['mdd']:>5.1f}%")

print(f"\nCross-validation summary:")
print(f"  Current TP/SL:  Pre-overlap {st_a['pnl']:>+8.1f}% → In-sample {st_b['pnl']:>+8.1f}%  (gap {st_b['pnl']-st_a['pnl']:>+.1f}%)")
print(f"  Pre-opt TP/SL:  Pre-overlap {st_d['pnl']:>+8.1f}% → In-sample {st_c['pnl']:>+8.1f}%  (gap {st_c['pnl']-st_d['pnl']:>+.1f}%)")
print(f"  WR comparison:  Current Pre {st_a['wr']:.1f}% / IS {st_b['wr']:.1f}%  |  Pre-opt Pre {st_d['wr']:.1f}% / IS {st_c['wr']:.1f}%")

results['phase3'] = {
    'A_current_pre': st_a, 'B_current_insample': st_b,
    'C_preopt_insample': st_c, 'D_preopt_pre': st_d,
    'E_current_full': st_e, 'F_preopt_full': st_f,
    'pre_optimal_patterns': len(pre_optimal_map),
}

# ============================================================
# Phase 4: 720d Expanding Window Walk-Forward (3 folds)
# ============================================================
print("\n" + "=" * 70)
print("Phase 4: 720d Expanding Window Walk-Forward")
print("=" * 70, flush=True)

# Define folds by dates
# ~720 days: 2024-02-23 ~ 2026-02-12
# Each test period: ~180 days
fold_dates = [
    ('F1', '2024-02-23', '2024-08-20', '2024-08-20', '2025-02-16'),
    ('F2', '2024-02-23', '2025-02-16', '2025-02-16', '2025-08-15'),
    ('F3', '2024-02-23', '2025-08-15', '2025-08-15', '2026-02-12'),
]

phase4 = {}
all_fold_oos_pnls = []

for fname, train_start, train_end, test_start, test_end in fold_dates:
    print(f"\n--- {fname}: Train {train_start}~{train_end}, Test {test_start}~{test_end} ---", flush=True)

    # Convert to bar indices
    ts_train_s = np.datetime64(train_start)
    ts_train_e = np.datetime64(train_end)
    ts_test_s = np.datetime64(test_start)
    ts_test_e = np.datetime64(test_end)

    train_s = max(0, int(np.searchsorted(timestamps, ts_train_s)))
    train_e = int(np.searchsorted(timestamps, ts_train_e))
    test_s = int(np.searchsorted(timestamps, ts_test_s))
    test_e = min(n_bars, int(np.searchsorted(timestamps, ts_test_e, side='right')))

    train_days = (ts_train_e - ts_train_s) / np.timedelta64(1, 'D')
    test_days = (ts_test_e - ts_test_s) / np.timedelta64(1, 'D')
    print(f"  Train: bars {train_s}-{train_e} ({int(train_days)}d), "
          f"Test: bars {test_s}-{test_e} ({int(test_days)}d)")

    # Build pattern indices for training period
    train_pat_idx = defaultdict(list)
    for i in range(max(2, train_s), train_e):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        train_pat_idx[pat].append(i)
    for k in train_pat_idx:
        train_pat_idx[k] = np.array(train_pat_idx[k])

    # Grid search on training data (only for our 51 patterns)
    fold_map = {}
    fold_stats = {'qualifying': 0, 'insufficient': 0, 'no_qualifying': 0}
    for pat, (d, _, _) in PATTERNS.items():
        idx = train_pat_idx.get(pat, np.array([], dtype=int))
        if len(idx) < MIN_TRADES:
            fold_stats['insufficient'] += 1
            continue
        best = grid_search_pattern(idx, d, opens, highs, lows, n_bars)
        if best is None:
            fold_stats['no_qualifying'] += 1
            continue
        new_tp, new_sl = best[0], best[1]
        fold_map[pat] = (d, new_tp, new_sl)
        fold_stats['qualifying'] += 1

    print(f"  Patterns qualifying: {fold_stats['qualifying']}, "
          f"insufficient: {fold_stats['insufficient']}, "
          f"no qualifying TP/SL: {fold_stats['no_qualifying']}")

    # Test on OOS with fold-optimized TP/SL
    oos_trades = collect_trades_1pos(types, opens, highs, lows, n_bars, fold_map,
                                     start_bar=test_s, end_bar=test_e)
    oos_st = portfolio_stats(oos_trades)
    print(f"  OOS result: {oos_st['trades']} trades, WR {oos_st['wr']:.1f}%, "
          f"PnL {oos_st['pnl']:+.1f}%, MDD {oos_st['mdd']:.1f}%")

    # Also test with CURRENT TP/SL on same OOS period (for comparison)
    cur_oos = collect_trades_1pos(types, opens, highs, lows, n_bars, PATTERNS,
                                  start_bar=test_s, end_bar=test_e)
    cur_oos_st = portfolio_stats(cur_oos)
    print(f"  Current TP/SL on same OOS: {cur_oos_st['trades']} trades, WR {cur_oos_st['wr']:.1f}%, "
          f"PnL {cur_oos_st['pnl']:+.1f}%")

    diff = oos_st['pnl'] - cur_oos_st['pnl']
    print(f"  Fold-optimized vs Current: {diff:+.1f}%")

    phase4[fname] = {
        'train_period': f"{train_start}~{train_end}",
        'test_period': f"{test_start}~{test_end}",
        'patterns_qualifying': fold_stats['qualifying'],
        'fold_optimized_oos': oos_st,
        'current_tpsl_oos': cur_oos_st,
        'diff_pnl': round(diff, 1),
        'fold_map_tpsl': {p: (d, tp, sl) for p, (d, tp, sl) in fold_map.items()},
    }
    all_fold_oos_pnls.append(oos_st['pnl'])

# WF summary
print(f"\n--- Walk-Forward Summary ---")
total_wf_pnl = sum(all_fold_oos_pnls)
folds_positive = sum(1 for x in all_fold_oos_pnls if x > 0)
print(f"Total OOS PnL (fold-optimized): {total_wf_pnl:+.1f}%")
print(f"Folds positive: {folds_positive}/3")
print(f"Individual fold OOS PnLs: {[f'{x:+.1f}%' for x in all_fold_oos_pnls]}")

results['phase4'] = phase4

# ============================================================
# Phase 5: Look-ahead Decomposition Revision
# ============================================================
print("\n" + "=" * 70)
print("Phase 5: Look-ahead Decomposition Revision")
print("=" * 70, flush=True)

# Decomposition:
# Total PnL (270d current) = Pattern Selection + TP/SL Optimization + Genuine Edge
# We can estimate using:
# - Genuine edge (direction only): pre-overlap performance with any profitable TP/SL
# - TP/SL contribution: current vs fold-optimized performance gap
# - Pattern selection: in-sample vs expanding-window selection gap

# A: Full in-sample result (current TP/SL on in-sample) = pattern+tpsl+genuine
full_is = st_b['pnl']  # Phase 3 scenario B

# B: WF fold-optimized on in-sample test period (F3 covers most of in-sample)
# Use F3 fold-optimized OOS as "pattern selection genuine + TP/SL fold-optimized"
f3_oos = phase4.get('F3', {}).get('fold_optimized_oos', {}).get('pnl', 0)

# C: Current TP/SL on pre-overlap = pattern genuine (no pattern selection bias, no TP/SL bias)
pre_genuine = st_a['pnl']

# D: Pre-overlap optimal on in-sample = reverse validation
reverse_val = st_c['pnl']

print(f"\nDecomposition components:")
print(f"  A. Full in-sample (current TP/SL):     {full_is:>+8.1f}%  (pattern selection + TP/SL + genuine)")
print(f"  B. WF F3 fold-optimized OOS:           {f3_oos:>+8.1f}%  (genuine + TP/SL fold-opt)")
print(f"  C. Current TP/SL on pre-overlap:       {pre_genuine:>+8.1f}%  (direction edge only)")
print(f"  D. Pre-opt TP/SL on in-sample:         {reverse_val:>+8.1f}%  (reverse validation)")

# Estimate contributions
# Annualize pre-overlap to compare (437d → 270d equivalent)
pre_annualized = pre_genuine * (269 / 437) if pre_genuine != 0 else 0
print(f"\n  C annualized to 270d equivalent:       {pre_annualized:>+8.1f}%")
print(f"\n  Look-ahead decomposition (revised):")
print(f"    Genuine direction edge (C annualized): {pre_annualized:>+8.1f}% ({pre_annualized/full_is*100:.1f}% of total)" if full_is > 0 else "")
tpsl_contribution = full_is - pre_annualized if full_is > 0 else 0
print(f"    TP/SL + pattern selection bias:         {tpsl_contribution:>+8.1f}% ({tpsl_contribution/full_is*100:.1f}% of total)" if full_is > 0 else "")

results['phase5'] = {
    'full_insample': full_is,
    'wf_f3_oos': f3_oos,
    'pre_overlap_genuine': pre_genuine,
    'pre_overlap_annualized': round(pre_annualized, 1),
    'reverse_validation': reverse_val,
    'genuine_pct': round(pre_annualized / full_is * 100, 1) if full_is > 0 else 0,
    'bias_pct': round(tpsl_contribution / full_is * 100, 1) if full_is > 0 else 0,
}

# ============================================================
# Phase 6: Robust TP/SL Identification
# ============================================================
print("\n" + "=" * 70)
print("Phase 6: Robust TP/SL Identification")
print("=" * 70, flush=True)

# Find patterns where fold-optimized TP/SL are consistent across folds
fold_tpsls = {}
for fname in ['F1', 'F2', 'F3']:
    fold_data = phase4.get(fname, {})
    for pat, (d, tp, sl) in fold_data.get('fold_map_tpsl', {}).items():
        if pat not in fold_tpsls:
            fold_tpsls[pat] = []
        fold_tpsls[pat].append((fname, tp, sl))

print(f"\n{'Pattern':<14} {'Cur TP/SL':>10} {'F1 TP/SL':>10} {'F2 TP/SL':>10} {'F3 TP/SL':>10} {'Folds':>5} {'Stable':>7}")
print("-" * 78)

phase6 = {}
stable_count = 0
for pat, (d, cur_tp, cur_sl) in sorted(PATTERNS.items()):
    folds = fold_tpsls.get(pat, [])
    f1 = f2 = f3 = "  -/-  "
    tps = []; sls = []
    for fn, tp, sl in folds:
        tps.append(tp); sls.append(sl)
        if fn == 'F1': f1 = f"{tp:>4}/{sl:<4}"
        elif fn == 'F2': f2 = f"{tp:>4}/{sl:<4}"
        elif fn == 'F3': f3 = f"{tp:>4}/{sl:<4}"

    n_folds = len(folds)
    # Stable = same TP/SL across all qualifying folds, or majority agree
    if n_folds >= 2:
        tp_mode = max(set(tps), key=tps.count)
        sl_mode = max(set(sls), key=sls.count)
        agree = sum(1 for t, s in zip(tps, sls) if t == tp_mode and s == sl_mode)
        stable = agree >= 2
    else:
        stable = False

    if stable:
        stable_count += 1

    phase6[pat] = {
        'current': f"{cur_tp}/{cur_sl}",
        'folds': [(fn, tp, sl) for fn, tp, sl in folds],
        'n_folds': n_folds,
        'stable': stable,
    }

    stable_str = "  YES" if stable else "   no"
    print(f"{pat:<14} {cur_tp:>4}/{cur_sl:<4}  {f1:>10} {f2:>10} {f3:>10} {n_folds:>5} {stable_str}")

print(f"\nStable patterns (consistent TP/SL across folds): {stable_count}/51")

results['phase6'] = phase6

# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70, flush=True)

print(f"""
Cross-Validation Summary:
  Current TP/SL:    Pre-overlap {st_a['pnl']:>+8.1f}% | In-sample {st_b['pnl']:>+8.1f}% | Full 720d {st_e['pnl']:>+8.1f}%
  Pre-opt TP/SL:    Pre-overlap {st_d['pnl']:>+8.1f}% | In-sample {st_c['pnl']:>+8.1f}% | Full 720d {st_f['pnl']:>+8.1f}%

Walk-Forward (fold-optimized):
  F1 OOS: {all_fold_oos_pnls[0]:>+8.1f}%  F2 OOS: {all_fold_oos_pnls[1]:>+8.1f}%  F3 OOS: {all_fold_oos_pnls[2]:>+8.1f}%
  Total WF OOS: {total_wf_pnl:>+8.1f}%  ({folds_positive}/3 folds positive)

Look-ahead Decomposition (revised):
  Genuine direction edge: {pre_annualized:>+8.1f}% ({results['phase5']['genuine_pct']:.0f}% of in-sample PnL)
  TP/SL + selection bias: {tpsl_contribution:>+8.1f}% ({results['phase5']['bias_pct']:.0f}% of in-sample PnL)

Stable TP/SL patterns: {stable_count}/51 ({stable_count/51*100:.0f}%)
""")

# Save
elapsed = time.time() - t0
results['metadata'] = {
    'script': 'extended_rediscovery_v4_robust_tpsl.py',
    'data': str(bn_path),
    'n_bars': n_bars,
    'pre_overlap_bars': overlap_start_bar,
    'insample_bars': overlap_end_bar - overlap_start_bar,
    'elapsed_seconds': round(elapsed, 1),
}

out_path = RESULTS_DIR / "extended_rediscovery_v4_robust_tpsl.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Total time: {elapsed:.1f}s")
print(f"Results saved: {out_path}")
