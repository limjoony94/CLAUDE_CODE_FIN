#!/usr/bin/env python3
"""
Universal TP/SL Full Sweep — 0.05% Step, Compound Returns
==========================================================
TP 0.50~3.00%, SL 0.50~3.00% (0.05% step = 51x51 = 2,601 combos)

For each combo:
  - Dynamic pattern discovery (edge >= 5pp, min_trades >= 10)
  - 1-pos-at-a-time compound portfolio
  - No MC test (sweep speed; top candidates need post-hoc MC validation)

Speed: precompute per-signal first-hit bars for all thresholds.
Data: 270d BingX (consistent with production scanner).
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
POSITION_SIZE = 0.95
EDGE_THRESHOLD = 5.0
MIN_TRADES = 10

TP_MIN, TP_MAX = 0.50, 3.00
SL_MIN, SL_MAX = 0.50, 3.00
STEP = 0.05

DATA_FILE = PROJECT_ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
OUTPUT_FILE = PROJECT_ROOT / 'results' / 'tpsl_sweep_005_compound.json'


# ======================================================================
# Phase 1: Load and classify
# ======================================================================
def load_and_classify(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    n = len(df)
    body_abs = np.abs(c - o)
    avg_b20 = pd.Series(body_abs).rolling(20).mean().values
    types = []
    for i in range(n):
        row = pd.Series({'open': o[i], 'high': h[i], 'low': l[i], 'close': c[i]})
        ab = avg_b20[i] if not np.isnan(avg_b20[i]) else 1.0
        types.append(classify_candle(row, ab).value)
    return o, h, l, c, types, n


# ======================================================================
# Phase 2: Precompute first-hit bars
# ======================================================================
def precompute_first_hits(opens, highs, lows, n, thresholds):
    """For each bar i (signal bar), precompute first bar where
    upside/downside from entry (opens[i+1]) >= each threshold.

    Returns:
        entry_prices: [n] float64
        up_first: [n, n_thresh] int32 — absolute bar index or -1
        dn_first: [n, n_thresh] int32 — absolute bar index or -1
    """
    n_thresh = len(thresholds)
    entry_prices = np.zeros(n, dtype=np.float64)
    up_first = np.full((n, n_thresh), -1, dtype=np.int32)
    dn_first = np.full((n, n_thresh), -1, dtype=np.int32)

    for i in range(2, n):
        eb = i + 1
        if eb >= n:
            break
        entry = opens[eb]
        if entry <= 0:
            continue
        entry_prices[i] = entry

        scan_start = i + 2
        scan_end = min(i + 2 + MAX_BARS, n)
        if scan_start >= scan_end:
            continue

        bar_h = highs[scan_start:scan_end]
        bar_l = lows[scan_start:scan_end]

        up_pcts = (bar_h / entry - 1.0) * 100.0
        dn_pcts = (1.0 - bar_l / entry) * 100.0

        up_cm = np.maximum.accumulate(up_pcts)
        dn_cm = np.maximum.accumulate(dn_pcts)

        for k in range(n_thresh):
            t = thresholds[k]
            idx = np.searchsorted(up_cm, t)
            if idx < len(up_cm):
                up_first[i, k] = scan_start + idx
            idx = np.searchsorted(dn_cm, t)
            if idx < len(dn_cm):
                dn_first[i, k] = scan_start + idx

    return entry_prices, up_first, dn_first


# ======================================================================
# Phase 3: Compound stats
# ======================================================================
def calc_compound_stats(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'pnl_mdd': 0, 'safety': 0, 'min_eq': 100}
    equity = 100.0
    peak = 100.0
    mdd = 0.0
    min_eq = 100.0
    w_sum = 0.0
    l_sum = 0.0
    wins = 0

    for _, _, pnl_pct in trades:
        equity *= (1 + pnl_pct / 100 * POSITION_SIZE)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
        if equity < min_eq:
            min_eq = equity
        if pnl_pct > 0:
            wins += 1
            w_sum += pnl_pct
        else:
            l_sum += abs(pnl_pct)

    n_t = len(trades)
    wr = wins / n_t * 100
    pnl = (equity / 100 - 1) * 100
    pf = w_sum / l_sum if l_sum > 0 else 999
    pnl_mdd = pnl / mdd if mdd > 0 else 999
    avg_w = w_sum / wins if wins > 0 else 0
    avg_l = l_sum / (n_t - wins) if (n_t - wins) > 0 else 0
    be_wr = avg_l / (avg_w + avg_l) * 100 if (avg_w + avg_l) > 0 else 50
    safety = wr - be_wr

    return {
        'pnl': round(pnl, 1),
        'trades': n_t,
        'wr': round(wr, 1),
        'mdd': round(mdd, 1),
        'pf': round(pf, 2),
        'pnl_mdd': round(pnl_mdd, 1) if pnl_mdd < 999 else 999,
        'safety': round(safety, 1),
        'min_eq': round(min_eq, 1),
    }


# ======================================================================
# Phase 4: Sweep
# ======================================================================
def sweep(pattern_signals, entry_prices, up_first, dn_first, opens,
          thresholds, tp_values, sl_values):
    n_combos = len(tp_values) * len(sl_values)
    results = []
    done = 0
    t0 = time.time()

    thresh_idx = {round(t, 4): i for i, t in enumerate(thresholds)}

    for tp in tp_values:
        tp_i = thresh_idx[round(tp, 4)]
        tp_win = tp * LEVERAGE - FEE_PCT
        tp_loss_neg = -(tp * LEVERAGE + FEE_PCT)  # if used as SL in reverse

        for sl in sl_values:
            sl_i = thresh_idx[round(sl, 4)]
            sl_loss = -(sl * LEVERAGE + FEE_PCT)
            be_wr = sl / (tp + sl) * 100

            all_trades = []
            n_pat = 0
            n_long = 0
            n_short = 0

            for pat, sig_arr in pattern_signals.items():
                best_edge = -999
                best_dir = None
                best_trades = None

                for direction in ('LONG', 'SHORT'):
                    if direction == 'LONG':
                        tp_bars = up_first[sig_arr, tp_i]
                        sl_bars = dn_first[sig_arr, sl_i]
                    else:
                        tp_bars = dn_first[sig_arr, tp_i]
                        sl_bars = up_first[sig_arr, sl_i]

                    tp_v = tp_bars >= 0
                    sl_v = sl_bars >= 0

                    # Categories
                    tp_only = tp_v & ~sl_v
                    sl_only = sl_v & ~tp_v
                    both = tp_v & sl_v
                    tp_first = both & (tp_bars < sl_bars)
                    sl_first = both & (sl_bars < tp_bars)
                    same = both & (tp_bars == sl_bars)

                    # Same-bar resolution
                    same_tp_wins = np.zeros(len(sig_arr), dtype=bool)
                    same_idx = np.where(same)[0]
                    for si in same_idx:
                        bar = tp_bars[si]
                        bo = opens[bar]
                        entry = entry_prices[sig_arr[si]]
                        if direction == 'LONG':
                            tpp = entry * (1 + tp / 100)
                            slp = entry * (1 - sl / 100)
                        else:
                            tpp = entry * (1 - tp / 100)
                            slp = entry * (1 + sl / 100)
                        same_tp_wins[si] = abs(tpp - bo) <= abs(slp - bo)

                    is_win = tp_only | tp_first | (same & same_tp_wins)
                    is_loss = sl_only | sl_first | (same & ~same_tp_wins)
                    w = int(np.sum(is_win))
                    l = int(np.sum(is_loss))
                    total = w + l

                    if total < MIN_TRADES:
                        continue
                    wr = w / total * 100
                    edge = wr - be_wr
                    if edge < EDGE_THRESHOLD:
                        continue

                    if edge > best_edge:
                        best_edge = edge
                        best_dir = direction
                        valid = is_win | is_loss
                        entry_bar_vals = sig_arr + 1
                        exit_bars = np.where(is_win, tp_bars,
                                             np.where(is_loss, sl_bars, -1))
                        pnl_vals = np.where(is_win, tp_win, sl_loss)
                        vi = np.where(valid)[0]
                        best_trades = list(zip(
                            entry_bar_vals[vi].tolist(),
                            exit_bars[vi].tolist(),
                            pnl_vals[vi].tolist()
                        ))

                if best_trades is not None:
                    n_pat += 1
                    if best_dir == 'LONG':
                        n_long += 1
                    else:
                        n_short += 1
                    all_trades.extend(best_trades)

            # 1-pos-at-a-time
            all_trades.sort()
            filtered = []
            last_exit = -1
            for eb, xb, pnl in all_trades:
                if eb > last_exit:
                    filtered.append((eb, xb, pnl))
                    last_exit = xb

            stats = calc_compound_stats(filtered)
            stats['n_patterns'] = n_pat
            stats['n_long'] = n_long
            stats['n_short'] = n_short
            results.append({'tp': round(tp, 2), 'sl': round(sl, 2), **stats})
            done += 1

        elapsed = time.time() - t0
        eta = elapsed / done * (n_combos - done) if done > 0 else 0
        pct = done / n_combos * 100
        # Print summary for this TP row
        row_results = results[-len(sl_values):]
        best_row = max(row_results, key=lambda x: x['pnl_mdd'] if x['pnl_mdd'] < 999 else -1)
        print(f"  TP={tp:.2f} | best SL={best_row['sl']:.2f} "
              f"P/M={best_row['pnl_mdd']:>5.1f}x PnL={best_row['pnl']:>+7.1f}% "
              f"MDD={best_row['mdd']:>5.1f}% pat={best_row['n_patterns']:>3} "
              f"| {pct:.0f}% ETA {eta:.0f}s", flush=True)

    return results


# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Universal TP/SL Full Sweep — 0.05% Step, Compound Returns")
    print(f"Range: TP {TP_MIN}~{TP_MAX}%, SL {SL_MIN}~{SL_MAX}%, step {STEP}%")
    print(f"Filters: edge >= {EDGE_THRESHOLD}pp, min_trades >= {MIN_TRADES}")
    print(f"Leverage: {LEVERAGE}x | Fee: {FEE_PCT}% | Position: {POSITION_SIZE*100:.0f}%")
    print("NOTE: MC test skipped for speed — top candidates need validation")
    print("=" * 70)
    t_start = time.time()

    # Phase 1
    print("\n[Phase 1] Loading and classifying candles...", flush=True)
    opens, highs, lows, closes, types, n = load_and_classify(DATA_FILE)
    print(f"  {n:,} bars loaded from {DATA_FILE.name}")

    # Build signal index
    signal_index = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        signal_index.setdefault(pat, [])
        signal_index[pat].append(i)
    # Convert to numpy arrays
    pattern_signals = {p: np.array(b, dtype=np.int32) for p, b in signal_index.items()}
    print(f"  {len(pattern_signals)} unique patterns, "
          f"{sum(len(v) for v in pattern_signals.values()):,} total signals")

    # Phase 2
    print("\n[Phase 2] Precomputing first-hit bars...", flush=True)
    thresholds = np.round(np.arange(TP_MIN, TP_MAX + STEP/2, STEP), 4)
    n_thresh = len(thresholds)
    print(f"  {n_thresh} threshold levels: {thresholds[0]:.2f} ~ {thresholds[-1]:.2f}%")

    t2 = time.time()
    entry_prices, up_first, dn_first = precompute_first_hits(
        opens, highs, lows, n, thresholds)
    print(f"  Done in {time.time()-t2:.1f}s | "
          f"Arrays: up_first {up_first.shape}, dn_first {dn_first.shape}")

    # Phase 3
    tp_values = thresholds.copy()
    sl_values = thresholds.copy()
    n_combos = len(tp_values) * len(sl_values)
    print(f"\n[Phase 3] Sweeping {len(tp_values)}x{len(sl_values)} = "
          f"{n_combos:,} TP/SL combinations...", flush=True)

    results = sweep(pattern_signals, entry_prices, up_first, dn_first,
                    opens, thresholds, tp_values, sl_values)

    # Phase 4: Rankings
    elapsed = time.time() - t_start
    positive = [r for r in results if r['pnl'] > 0 and r['trades'] > 0]
    safe = [r for r in positive if r['mdd'] < 50]

    print(f"\n{'='*70}")
    print(f"RESULTS — {len(positive)}/{n_combos} positive, "
          f"{len(safe)} with MDD<50%")
    print(f"{'='*70}")

    # Top by PnL/MDD (MDD < 50%)
    by_pm = sorted(safe, key=lambda x: x['pnl_mdd'], reverse=True)[:30]
    print(f"\nTop 30 by PnL/MDD (MDD < 50%)")
    print(f"{'#':>3} {'TP':>5} {'SL':>5} | {'P/M':>6} {'PnL':>8} {'MDD':>6} "
          f"{'WR':>6} {'Sfty':>6} {'Trd':>4} {'Pat':>4} {'L/S':>6} {'MinEq':>6}")
    print("-" * 85)
    for i, r in enumerate(by_pm, 1):
        print(f"{i:>3} {r['tp']:>5.2f} {r['sl']:>5.2f} | "
              f"{r['pnl_mdd']:>5.1f}x {r['pnl']:>+7.1f}% {r['mdd']:>5.1f}% "
              f"{r['wr']:>5.1f}% {r['safety']:>+5.1f} {r['trades']:>4} "
              f"{r['n_patterns']:>4} {r['n_long']:>2}/{r['n_short']:<2} "
              f"{r['min_eq']:>5.1f}")

    # Top by Safety
    by_sf = sorted(safe, key=lambda x: x['safety'], reverse=True)[:20]
    print(f"\nTop 20 by Safety (MDD < 50%)")
    print(f"{'#':>3} {'TP':>5} {'SL':>5} | {'Sfty':>6} {'P/M':>6} {'PnL':>8} "
          f"{'MDD':>6} {'WR':>6} {'Trd':>4} {'Pat':>4}")
    print("-" * 70)
    for i, r in enumerate(by_sf, 1):
        print(f"{i:>3} {r['tp']:>5.2f} {r['sl']:>5.2f} | "
              f"{r['safety']:>+5.1f} {r['pnl_mdd']:>5.1f}x {r['pnl']:>+7.1f}% "
              f"{r['mdd']:>5.1f}% {r['wr']:>5.1f}% {r['trades']:>4} "
              f"{r['n_patterns']:>4}")

    # Top by absolute PnL
    by_pnl = sorted(safe, key=lambda x: x['pnl'], reverse=True)[:20]
    print(f"\nTop 20 by Absolute PnL (MDD < 50%)")
    print(f"{'#':>3} {'TP':>5} {'SL':>5} | {'PnL':>8} {'P/M':>6} {'MDD':>6} "
          f"{'WR':>6} {'Trd':>4} {'Pat':>4}")
    print("-" * 65)
    for i, r in enumerate(by_pnl, 1):
        print(f"{i:>3} {r['tp']:>5.2f} {r['sl']:>5.2f} | "
              f"{r['pnl']:>+7.1f}% {r['pnl_mdd']:>5.1f}x {r['mdd']:>5.1f}% "
              f"{r['wr']:>5.1f}% {r['trades']:>4} {r['n_patterns']:>4}")

    # Top by lowest MDD
    by_mdd = sorted([r for r in positive if r['trades'] >= 50],
                    key=lambda x: x['mdd'])[:20]
    print(f"\nTop 20 by Lowest MDD (trades >= 50)")
    print(f"{'#':>3} {'TP':>5} {'SL':>5} | {'MDD':>6} {'PnL':>8} {'P/M':>6} "
          f"{'WR':>6} {'Trd':>4} {'Pat':>4}")
    print("-" * 65)
    for i, r in enumerate(by_mdd, 1):
        print(f"{i:>3} {r['tp']:>5.2f} {r['sl']:>5.2f} | "
              f"{r['mdd']:>5.1f}% {r['pnl']:>+7.1f}% {r['pnl_mdd']:>5.1f}x "
              f"{r['wr']:>5.1f}% {r['trades']:>4} {r['n_patterns']:>4}")

    # Comparison: TP 0.8/SL 1.2 vs TP 2.1/SL 3.0
    print(f"\n{'='*70}")
    print("Key TP/SL Comparisons")
    print(f"{'='*70}")
    targets = [(0.80, 1.20), (1.00, 1.50), (1.50, 2.00), (2.00, 3.00), (2.10, 3.00)]
    for tp_t, sl_t in targets:
        r = next((x for x in results
                  if abs(x['tp'] - tp_t) < 0.001 and abs(x['sl'] - sl_t) < 0.001), None)
        if r:
            print(f"  TP {r['tp']:.2f}/SL {r['sl']:.2f}: PnL {r['pnl']:>+7.1f}% "
                  f"MDD {r['mdd']:>5.1f}% P/M {r['pnl_mdd']:>5.1f}x "
                  f"WR {r['wr']:>5.1f}% Trades {r['trades']:>4} "
                  f"Pat {r['n_patterns']:>3} ({r['n_long']}L/{r['n_short']}S) "
                  f"Safety {r['safety']:>+5.1f}")

    # Heatmap-ready 2D arrays
    tp_arr = sorted(set(r['tp'] for r in results))
    sl_arr = sorted(set(r['sl'] for r in results))
    lookup = {(r['tp'], r['sl']): r for r in results}

    heatmap = {}
    for metric in ['pnl', 'mdd', 'pnl_mdd', 'wr', 'trades', 'n_patterns', 'safety']:
        grid_2d = []
        for tp in tp_arr:
            row = []
            for sl in sl_arr:
                r = lookup.get((tp, sl))
                row.append(r[metric] if r else 0)
            grid_2d.append(row)
        heatmap[metric] = grid_2d

    # Save
    output = {
        'params': {
            'tp_range': [TP_MIN, TP_MAX],
            'sl_range': [SL_MIN, SL_MAX],
            'step': STEP,
            'n_combos': n_combos,
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'position_size': POSITION_SIZE,
            'edge_threshold': EDGE_THRESHOLD,
            'min_trades': MIN_TRADES,
            'data_file': DATA_FILE.name,
            'mc_test': False,
        },
        'summary': {
            'positive_combos': len(positive),
            'safe_combos_mdd50': len(safe),
            'elapsed_seconds': round(elapsed, 1),
        },
        'rankings': {
            'top_pnl_mdd': by_pm[:20],
            'top_safety': by_sf[:20],
            'top_pnl': by_pnl[:20],
            'top_low_mdd': by_mdd[:20],
        },
        'heatmap': {
            'tp_axis': tp_arr,
            'sl_axis': sl_arr,
            **heatmap,
        },
        'grid': results,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
