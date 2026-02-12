#!/usr/bin/env python3
"""
Extended Rediscovery v5: Portfolio Scenario Comparison
======================================================
v4 결과 활용 — stable TP/SL, pre-overlap 성과 기반 포트폴리오 시나리오 비교.

6 Scenarios:
  A. BASELINE     — 51 patterns, current TP/SL (v1.27.1)
  B. HYBRID       — 5 stable patterns → stable TP/SL, 46 keep current
  C. PRE_POSITIVE — 27 pre-overlap positive patterns, current TP/SL
  D. PRE_MC_PASS  — 9 pre-overlap MC+edge pass patterns, pre-optimized TP/SL
  E. STABLE_ONLY  — 5 stable patterns, stable TP/SL
  F. HYBRID_PREPOS — 27 pre-positive patterns + 5 stable TP/SL overrides

Each scenario tested on:
  - Pre-overlap (437d, genuine OOS)
  - In-sample (270d)
  - Full 720d
  - Portfolio MC (10k sims)

Phase 3: 3-fold expanding window Walk-Forward for top scenarios
Phase 4: Risk-adjusted comparison and recommendation
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

# Import production classify_candle (CRITICAL)
sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 10000

# =======================================================================
# SCENARIO DEFINITIONS (from v4 results)
# =======================================================================

# A. BASELINE — 51 patterns, current TP/SL
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

# v4 Phase 6: 5 Stable TP/SL (3/3 or 2/2 fold agreement)
STABLE_TPSL = {
    "DN-D-BD":  ("SHORT", 1.5, 2.5),   # F2+F3 unanimous
    "MD-BU-MD": ("LONG",  1.0, 1.5),   # F1+F2+F3 unanimous
    "MU-IH-DN": ("LONG",  0.5, 2.0),   # F2+F3
    "U-H-MU":   ("LONG",  0.5, 3.0),   # F2+F3
    "U-MU-IH":  ("LONG",  0.5, 3.0),   # F1+F2
}

# v4 Phase 1: 27 pre-overlap positive PnL patterns
PRE_POSITIVE_PATTERNS = [
    "BD-BD-U", "BD-D-D", "BU-BU-BD", "D-BD-ST", "D-DN-DN", "D-MU-U",
    "DN-BD-BD", "DN-BD-BU", "DN-D-BD", "DN-DF-MU", "DN-DF-ST",
    "IH-MD-MD", "IH-ST-ST", "MD-BU-MD", "MD-H-MD", "MD-MD-MD",
    "MD-ST-BD", "MU-IH-DN", "ST-DN-BU", "ST-DN-U", "ST-MU-ST",
    "U-GS-DN", "U-MD-GS", "U-MU-H", "U-MU-IH", "U-ST-DF", "U-ST-DN",
]

# v4 Phase 2: 9 pre-overlap MC+edge pass patterns with their pre-optimized TP/SL
PRE_MC_PASS_PATTERNS = {
    "BU-BU-BD": ("LONG",  2.1, 3.0),    # CURRENT_OK
    "DN-D-BD":  ("SHORT", 1.5, 2.5),    # IMPROVED
    "DN-DF-MU": ("LONG",  1.05, 3.0),   # CURRENT_OK
    "MD-H-MD":  ("LONG",  0.7, 2.5),    # IMPROVED
    "MD-MD-MD": ("SHORT", 3.0, 3.0),    # CURRENT_OK
    "MU-IH-DN": ("LONG",  1.4, 2.0),    # CURRENT_OK
    "U-H-MU":   ("LONG",  0.5, 3.0),    # IMPROVED
    "U-MD-GS":  ("LONG",  0.3, 2.0),    # IMPROVED
    "U-MU-IH":  ("LONG",  2.0, 3.0),    # IMPROVED
}


def build_scenarios():
    """Build pattern_map for each scenario."""
    scenarios = {}

    # A. BASELINE — all 51, current TP/SL
    scenarios['A_BASELINE'] = dict(PATTERNS_CURRENT)

    # B. HYBRID — 5 stable get new TP/SL, rest keep current
    b = dict(PATTERNS_CURRENT)
    for pat, (d, tp, sl) in STABLE_TPSL.items():
        b[pat] = (d, tp, sl)
    scenarios['B_HYBRID'] = b

    # C. PRE_POSITIVE — 27 patterns, current TP/SL
    c = {p: PATTERNS_CURRENT[p] for p in PRE_POSITIVE_PATTERNS}
    scenarios['C_PRE_POSITIVE'] = c

    # D. PRE_MC_PASS — 9 patterns, pre-optimized TP/SL
    scenarios['D_PRE_MC_PASS'] = dict(PRE_MC_PASS_PATTERNS)

    # E. STABLE_ONLY — 5 patterns, stable TP/SL
    scenarios['E_STABLE_ONLY'] = dict(STABLE_TPSL)

    # F. HYBRID_PREPOS — 27 pre-positive + 5 stable overrides
    f = {p: PATTERNS_CURRENT[p] for p in PRE_POSITIVE_PATTERNS}
    for pat, (d, tp, sl) in STABLE_TPSL.items():
        if pat in f:
            f[pat] = (d, tp, sl)
    scenarios['F_HYBRID_PREPOS'] = f

    return scenarios


# =======================================================================
# Core functions (from v4, identical)
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
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0,
                'max_consec_loss': 0, 'avg_win': 0, 'avg_loss': 0}
    pnls = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wsum = 0; lsum = 0; wc = 0; lc = 0
    consec = 0; max_consec = 0
    wins_list = []; losses_list = []
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0:
            wins += 1; wsum += p; wins_list.append(p)
            consec = 0
        else:
            lsum += abs(p); losses_list.append(p)
            consec += 1
            if consec > max_consec: max_consec = consec
    return {
        'pnl': round(cum, 1), 'trades': len(pnls),
        'wr': round(wins / len(pnls) * 100, 1),
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_mdd': round(cum / mdd, 1) if mdd > 0 else 999,
        'max_consec_loss': max_consec,
        'avg_win': round(np.mean(wins_list), 3) if wins_list else 0,
        'avg_loss': round(np.mean(losses_list), 3) if losses_list else 0,
    }


def portfolio_mc(trades, n_sims=MC_SIMS):
    """Portfolio-level MC: shuffle trade order, compare cumulative PnL."""
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


def mc_mdd_p99(trades, n_sims=5000):
    """MC MDD P99: shuffle trade order, compute MDD distribution."""
    if len(trades) < 5:
        return 0
    pnls = np.array([t[2] for t in trades])
    mdds = []
    for _ in range(n_sims):
        np.random.shuffle(pnls)
        cum = 0; peak = 0; mdd = 0
        for p in pnls:
            cum += p
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > mdd: mdd = dd
        mdds.append(mdd)
    return round(np.percentile(mdds, 99), 1)


# =======================================================================
# MAIN
# =======================================================================
print("=" * 70)
print("Extended Rediscovery v5: Portfolio Scenario Comparison")
print("=" * 70)
t0 = time.time()

# Load 720d Binance data
print("\nLoading 720d Binance data...", flush=True)
bn_path = DATA_DIR / "btc_5m_720days_binance.csv"
opens, highs, lows, closes, types, timestamps, n_bars = load_and_classify(bn_path)

# Period boundaries
overlap_start_ts = np.datetime64('2025-05-05T15:00:00')
overlap_end_ts = np.datetime64('2026-01-30T14:55:00')
overlap_start_bar = int(np.searchsorted(timestamps, overlap_start_ts))
overlap_end_bar = int(np.searchsorted(timestamps, overlap_end_ts, side='right'))

print(f"Total bars: {n_bars}")
print(f"Pre-overlap: 0-{overlap_start_bar} ({overlap_start_bar} bars, ~{overlap_start_bar // 288}d)")
print(f"In-sample:   {overlap_start_bar}-{overlap_end_bar} ({overlap_end_bar - overlap_start_bar} bars, ~{(overlap_end_bar - overlap_start_bar) // 288}d)")

scenarios = build_scenarios()

# ====================================================================
# Phase 1: All scenarios × 3 periods
# ====================================================================
print("\n" + "=" * 70)
print("Phase 1: Scenario × Period Matrix")
print("=" * 70, flush=True)

periods = {
    'pre_overlap': (0, overlap_start_bar),
    'in_sample': (overlap_start_bar, overlap_end_bar),
    'full_720d': (0, n_bars),
}

results = {}
results['phase1'] = {}

print(f"\n{'Scenario':<20} {'Period':<14} {'PnL':>8} {'Trades':>7} {'WR':>6} "
      f"{'MDD':>7} {'PnL/MDD':>8} {'PF':>6} {'MCL':>4}", flush=True)
print("-" * 85)

for sname, pmap in scenarios.items():
    results['phase1'][sname] = {}
    for pname, (sb, eb) in periods.items():
        trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap, sb, eb)
        stats = portfolio_stats(trades)
        results['phase1'][sname][pname] = stats
        print(f"{sname:<20} {pname:<14} {stats['pnl']:>+8.1f} {stats['trades']:>7} "
              f"{stats['wr']:>5.1f}% {stats['mdd']:>6.1f}% {stats['pnl_mdd']:>7.1f}x "
              f"{stats['pf']:>5.2f} {stats['max_consec_loss']:>4}", flush=True)
    print()

# ====================================================================
# Phase 2: Portfolio MC for each scenario (full 720d)
# ====================================================================
print("\n" + "=" * 70)
print("Phase 2: Portfolio MC Test (full 720d)")
print("=" * 70, flush=True)

results['phase2'] = {}
print(f"\n{'Scenario':<20} {'MC p-value':>12} {'MC MDD P99':>12} {'Status':>10}", flush=True)
print("-" * 60)

for sname, pmap in scenarios.items():
    trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap, 0, n_bars)
    p = portfolio_mc(trades)
    mdd_p99 = mc_mdd_p99(trades)
    status = "PASS" if p < 0.01 else "FAIL"
    results['phase2'][sname] = {'mc_p': round(p, 4), 'mdd_p99': mdd_p99, 'status': status}
    print(f"{sname:<20} {p:>12.4f} {mdd_p99:>11.1f}% {status:>10}", flush=True)

# ====================================================================
# Phase 3: Expanding Window Walk-Forward (top scenarios)
# ====================================================================
print("\n" + "=" * 70)
print("Phase 3: Expanding Window Walk-Forward (3 folds)")
print("=" * 70, flush=True)

# Fold boundaries (180d test each)
bars_per_180d = 180 * 288
fold_boundaries = [
    (0, bars_per_180d, bars_per_180d, 2 * bars_per_180d),         # F1: 180d train, 180d test
    (0, 2 * bars_per_180d, 2 * bars_per_180d, 3 * bars_per_180d), # F2: 360d train, 180d test
    (0, 3 * bars_per_180d, 3 * bars_per_180d, min(4 * bars_per_180d, n_bars)), # F3: 540d train, rest test
]

# Test scenarios A (baseline), B (hybrid), C (pre_positive), F (hybrid_prepos)
wf_scenarios = ['A_BASELINE', 'B_HYBRID', 'C_PRE_POSITIVE', 'F_HYBRID_PREPOS']
results['phase3'] = {}

for sname in wf_scenarios:
    pmap = scenarios[sname]
    fold_results = []
    total_oos_pnl = 0
    total_oos_trades = 0

    for fi, (train_s, train_e, test_s, test_e) in enumerate(fold_boundaries, 1):
        if test_e > n_bars:
            test_e = n_bars
        # OOS performance
        trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap, test_s, test_e)
        stats = portfolio_stats(trades)
        fold_results.append({
            'fold': fi,
            'train_bars': train_e - train_s,
            'test_bars': test_e - test_s,
            **stats
        })
        total_oos_pnl += stats['pnl']
        total_oos_trades += stats['trades']

    results['phase3'][sname] = {
        'folds': fold_results,
        'total_oos_pnl': round(total_oos_pnl, 1),
        'total_oos_trades': total_oos_trades,
        'folds_positive': sum(1 for f in fold_results if f['pnl'] > 0),
    }

    print(f"\n{sname}:", flush=True)
    for fr in fold_results:
        print(f"  F{fr['fold']}: PnL {fr['pnl']:>+8.1f}%, {fr['trades']:>4} trades, "
              f"WR {fr['wr']:>5.1f}%, MDD {fr['mdd']:>5.1f}%", flush=True)
    print(f"  TOTAL OOS: PnL {total_oos_pnl:>+8.1f}%, {total_oos_trades} trades, "
          f"{results['phase3'][sname]['folds_positive']}/3 folds positive", flush=True)

# ====================================================================
# Phase 4: Risk-Adjusted Comparison & Annualized Performance
# ====================================================================
print("\n" + "=" * 70)
print("Phase 4: Risk-Adjusted Comparison")
print("=" * 70, flush=True)

results['phase4'] = {}

print(f"\n{'Scenario':<20} {'Patterns':>8} {'720d PnL':>9} {'720d MDD':>9} "
      f"{'PnL/MDD':>8} {'Pre PnL':>8} {'Pre MDD':>8} {'P/M Pre':>8}", flush=True)
print("-" * 90)

for sname, pmap in scenarios.items():
    n_pat = len(pmap)
    full = results['phase1'][sname]['full_720d']
    pre = results['phase1'][sname]['pre_overlap']
    mc = results['phase2'][sname]

    # Annualized (720d ≈ 2 years)
    ann_pnl = full['pnl'] / 2.0

    results['phase4'][sname] = {
        'n_patterns': n_pat,
        'full_pnl': full['pnl'], 'full_mdd': full['mdd'], 'full_pnl_mdd': full['pnl_mdd'],
        'pre_pnl': pre['pnl'], 'pre_mdd': pre['mdd'], 'pre_pnl_mdd': pre['pnl_mdd'],
        'ann_pnl': round(ann_pnl, 1),
        'mc_p': mc['mc_p'], 'mdd_p99': mc['mdd_p99'],
        'full_wr': full['wr'], 'pre_wr': pre['wr'],
        'full_trades': full['trades'], 'pre_trades': pre['trades'],
    }

    print(f"{sname:<20} {n_pat:>8} {full['pnl']:>+8.1f}% {full['mdd']:>8.1f}% "
          f"{full['pnl_mdd']:>7.1f}x {pre['pnl']:>+7.1f}% {pre['mdd']:>7.1f}% "
          f"{pre['pnl_mdd']:>7.1f}x", flush=True)

# ====================================================================
# Phase 5: Trade-level Analysis per Scenario
# ====================================================================
print("\n" + "=" * 70)
print("Phase 5: Win/Loss Characteristics (Pre-overlap)")
print("=" * 70, flush=True)

results['phase5'] = {}

print(f"\n{'Scenario':<20} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} "
      f"{'BE_WR':>6} {'Margin':>7} {'Safety':>7}", flush=True)
print("-" * 70)

for sname, pmap in scenarios.items():
    trades = collect_trades_1pos(types, opens, highs, lows, n_bars, pmap, 0, overlap_start_bar)
    stats = portfolio_stats(trades)

    avg_w = stats['avg_win']
    avg_l = abs(stats['avg_loss']) if stats['avg_loss'] != 0 else 1
    rr = round(avg_w / avg_l, 3) if avg_l > 0 else 999
    be_wr = round(avg_l / (avg_w + avg_l) * 100, 1) if (avg_w + avg_l) > 0 else 50
    margin = round(stats['wr'] - be_wr, 1)
    safety = "SAFE" if margin > 5 else ("THIN" if margin > 0 else "DANGER")

    results['phase5'][sname] = {
        'avg_win': avg_w, 'avg_loss': stats['avg_loss'],
        'rr': rr, 'be_wr': be_wr, 'margin_pp': margin, 'safety': safety,
        'wr': stats['wr'], 'trades': stats['trades'],
    }

    print(f"{sname:<20} {avg_w:>+7.3f} {stats['avg_loss']:>+7.3f} {rr:>5.3f} "
          f"{be_wr:>5.1f}% {margin:>+6.1f}pp {safety:>7}", flush=True)

# ====================================================================
# Phase 6: Per-Pattern Contribution in Hybrid vs Baseline (pre-overlap)
# ====================================================================
print("\n" + "=" * 70)
print("Phase 6: Stable TP/SL Impact (Pre-overlap, per-pattern)")
print("=" * 70, flush=True)

results['phase6'] = {}

for pat, (d, new_tp, new_sl) in STABLE_TPSL.items():
    old_d, old_tp, old_sl = PATTERNS_CURRENT[pat]

    # Collect trades for this pattern only — current vs stable
    old_map = {pat: (old_d, old_tp, old_sl)}
    new_map = {pat: (d, new_tp, new_sl)}

    old_trades = collect_trades_1pos(types, opens, highs, lows, n_bars, old_map, 0, overlap_start_bar)
    new_trades = collect_trades_1pos(types, opens, highs, lows, n_bars, new_map, 0, overlap_start_bar)

    old_s = portfolio_stats(old_trades)
    new_s = portfolio_stats(new_trades)

    results['phase6'][pat] = {
        'current': {'tp': old_tp, 'sl': old_sl, **old_s},
        'stable': {'tp': new_tp, 'sl': new_sl, **new_s},
        'pnl_diff': round(new_s['pnl'] - old_s['pnl'], 1),
        'wr_diff': round(new_s['wr'] - old_s['wr'], 1),
    }

    print(f"\n{pat} ({d}):", flush=True)
    print(f"  Current TP/SL {old_tp}/{old_sl}: PnL {old_s['pnl']:>+7.1f}%, "
          f"WR {old_s['wr']:>5.1f}%, {old_s['trades']} trades", flush=True)
    print(f"  Stable  TP/SL {new_tp}/{new_sl}: PnL {new_s['pnl']:>+7.1f}%, "
          f"WR {new_s['wr']:>5.1f}%, {new_s['trades']} trades", flush=True)
    print(f"  Diff: PnL {new_s['pnl'] - old_s['pnl']:>+7.1f}%, WR {new_s['wr'] - old_s['wr']:>+5.1f}pp",
          flush=True)

# ====================================================================
# Summary & Recommendation
# ====================================================================
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATION")
print("=" * 70, flush=True)

# Rank by pre-overlap PnL/MDD (most conservative metric)
ranking = sorted(results['phase4'].items(),
                 key=lambda x: x[1]['pre_pnl_mdd'], reverse=True)

print(f"\nRanking by Pre-overlap PnL/MDD (most conservative OOS metric):", flush=True)
print(f"{'Rank':>4} {'Scenario':<20} {'Pre PnL/MDD':>12} {'720d PnL/MDD':>13} "
      f"{'MC':>8} {'Patterns':>8}", flush=True)
print("-" * 70)

for i, (sname, data) in enumerate(ranking, 1):
    mc_str = "PASS" if data['mc_p'] < 0.01 else "FAIL"
    print(f"{i:>4} {sname:<20} {data['pre_pnl_mdd']:>11.1f}x {data['full_pnl_mdd']:>12.1f}x "
          f"{mc_str:>8} {data['n_patterns']:>8}", flush=True)

results['ranking'] = [
    {'rank': i+1, 'scenario': sname,
     'pre_pnl_mdd': data['pre_pnl_mdd'], 'full_pnl_mdd': data['full_pnl_mdd'],
     'mc_status': 'PASS' if data['mc_p'] < 0.01 else 'FAIL'}
    for i, (sname, data) in enumerate(ranking)
]

elapsed = time.time() - t0
results['metadata'] = {
    'script': 'extended_rediscovery_v5_portfolio_scenarios.py',
    'data': str(bn_path),
    'n_bars': n_bars,
    'n_scenarios': len(scenarios),
    'elapsed_seconds': round(elapsed, 1),
}

# Save results
out_path = RESULTS_DIR / "extended_rediscovery_v5_portfolio_scenarios.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Elapsed: {elapsed:.1f}s")
