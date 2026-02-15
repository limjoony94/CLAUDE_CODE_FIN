#!/usr/bin/env python3
"""
Universal TP/SL Study v3 — Improved Methodology
================================================
v2 대비 6가지 방법론 개선:

1. True Expanding Window WF (과거 데이터만으로 훈련)
   - v2: Cross-validation (IS가 미래 포함) → 비현실적
   - v3: [0..T1] train → [T1..T2] test, [0..T2] train → [T2..T3] test

2. Fresh Universe Scan per Fold (pre-selection bias 제거)
   - v2: 전체기간 MC-pass → WF에서 확인만
   - v3: 각 fold IS에서 3456패턴 독립 전수스캔

3. MC Robustness (다중 seed)
   - v2: seed=42 고정 단일 결과
   - v3: 3개 seed 전부 통과 요구

4. Edge Threshold Sensitivity
   - v2: MC p<0.01만
   - v3: edge threshold 0/3/5/8pp 비교

5. Timeout Analysis
   - 신호 대비 실제 체결률, timeout 비율 보고

6. Pattern Stability (cross-fold 재현율)
   - 각 패턴이 몇 개 fold에서 재발견되는지 추적

Protocol: production classify_candle, LEVERAGE=3, timeout DROP, 0.10% fee
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

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
MC_THRESHOLD = 0.01
MIN_TRADES = 10
MC_SEEDS = [42, 123, 7777]  # 3 seeds for robustness

CONFIGS = [
    {'name': 'TP1.5/SL1.0', 'tp': 1.5, 'sl': 1.0},
    {'name': 'TP2.1/SL3.0', 'tp': 2.1, 'sl': 3.0},
]

EDGE_THRESHOLDS = [0, 3, 5, 8]  # pp above break-even
N_WF_FOLDS = 4  # expanding window folds


# ===================================================================
# Core functions
# ===================================================================

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


def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals_detailed(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest with timeout tracking. Returns (trades, n_timeout)."""
    trades = []
    n_timeout = 0
    for sig in signal_bars:
        if sig + 1 >= n_bars:
            continue
        entry = opens[sig + 1]
        if entry <= 0:
            continue
        eb = sig + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        exited = False
        for j in range(sig + 2, min(sig + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                bo = opens[j]
                pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append((eb, j, pnl)); exited = True; break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE_PCT)); exited = True; break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT)); exited = True; break
        if not exited:
            n_timeout += 1
    return trades, n_timeout


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest (timeout dropped, no tracking)."""
    trades, _ = bt_signals_detailed(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars)
    return trades


def portfolio_1pos(all_trades):
    if not all_trades:
        return []
    all_trades.sort(key=lambda x: x[0])
    filtered = []; last_exit = -1
    for eb, xb, pnl in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl)); last_exit = xb
    return filtered


def calc_stats(trades):
    if not trades:
        return {'pnl': 0, 'compound_pnl': 0, 'trades': 0, 'wr': 0,
                'compound_mdd': 0, 'pf': 0, 'pnl_mdd': 0,
                'avg_win': 0, 'avg_loss': 0, 'safety': 0, 'max_cl': 0}
    pnls = [t[2] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    cum = 0; peak = 0; mdd = 0; cl = 0; max_cl = 0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        if peak - cum > mdd: mdd = peak - cum
        if p > 0: cl = 0
        else: cl += 1; max_cl = max(max_cl, cl)
    eq = 1.0; cp = 1.0; cmdd = 0
    for p in pnls:
        eq *= (1 + p / 100)
        if eq > cp: cp = eq
        cdd = (cp - eq) / cp * 100
        if cdd > cmdd: cmdd = cdd
    aw = float(np.mean(wins)) if wins else 0
    al = float(np.mean(losses)) if losses else 0
    be = abs(al) / (aw + abs(al)) * 100 if (aw + abs(al)) > 0 else 50
    wr = len(wins) / len(pnls) * 100
    ws = sum(wins); ls = sum(abs(x) for x in losses)
    return {
        'pnl': round(cum, 2), 'compound_pnl': round((eq - 1) * 100, 2),
        'trades': len(pnls), 'wr': round(wr, 1),
        'compound_mdd': round(cmdd, 2),
        'pf': round(ws / ls, 2) if ls > 0 else 999,
        'pnl_mdd': round((eq - 1) * 100 / cmdd, 2) if cmdd > 0 else 999,
        'avg_win': round(aw, 3), 'avg_loss': round(al, 3),
        'safety': round(wr - be, 1), 'max_cl': max_cl,
    }


def mc_test_single(pnls_arr, seed, n_sims=MC_SIMS):
    """Single-seed MC test."""
    rng = np.random.default_rng(seed)
    actual = np.sum(pnls_arr)
    signs = rng.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    return float(np.mean((signs @ pnls_arr) >= actual))


def mc_test_robust(pnls, seeds=MC_SEEDS, n_sims=MC_SIMS):
    """Multi-seed MC test. Returns max p-value across seeds (conservative)."""
    if len(pnls) < 5:
        return 1.0
    arr = np.array(pnls)
    p_values = [mc_test_single(arr, s, n_sims) for s in seeds]
    return max(p_values)  # most conservative


# ===================================================================
# Universe scan on a bar range
# ===================================================================

def scan_universe(signal_index, direction_list, tp, sl, opens, highs, lows, n,
                  bar_start, bar_end, min_trades, edge_threshold, be_wr):
    """Scan all patterns within [bar_start, bar_end) range.
    Returns list of MC-pass patterns with edge > threshold."""
    results = []
    for pat_name, all_sigs in signal_index.items():
        sigs = [s for s in all_sigs if bar_start <= s < bar_end]
        for direction in direction_list:
            trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n)
            if len(trades) < min_trades:
                continue
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            edge = wr - be_wr
            if edge < edge_threshold:
                continue
            mc_p = mc_test_robust(pnls)
            if mc_p < MC_THRESHOLD:
                results.append({
                    'pattern': pat_name, 'direction': direction,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'edge': round(edge, 1), 'pnl': round(sum(pnls), 2),
                    'mc_p': round(mc_p, 4),
                })
    return results


# ===================================================================
# Timeout analysis
# ===================================================================

def timeout_analysis(signal_index, mc_patterns, tp, sl, opens, highs, lows, n):
    """Analyze timeout rates for MC-pass patterns."""
    total_signals = 0
    total_trades = 0
    total_timeout = 0
    hold_bars = []

    for r in mc_patterns:
        sigs = signal_index[r['pattern']]
        trades, n_to = bt_signals_detailed(sigs, r['direction'], tp, sl, opens, highs, lows, n)
        total_signals += len(sigs)
        total_trades += len(trades)
        total_timeout += n_to
        for eb, xb, _ in trades:
            hold_bars.append(xb - eb)

    timeout_rate = total_timeout / (total_trades + total_timeout) * 100 if (total_trades + total_timeout) > 0 else 0
    avg_hold = float(np.mean(hold_bars)) if hold_bars else 0
    med_hold = float(np.median(hold_bars)) if hold_bars else 0

    return {
        'total_signals': total_signals,
        'total_trades': total_trades,
        'total_timeout': total_timeout,
        'timeout_rate': round(timeout_rate, 1),
        'avg_hold_bars': round(avg_hold, 1),
        'median_hold_bars': round(med_hold, 1),
        'avg_hold_hours': round(avg_hold * 5 / 60, 1),  # 5m bars
    }


# ===================================================================
# True Expanding Window Walk-Forward
# ===================================================================

def expanding_window_wf(signal_index, tp, sl, opens, highs, lows, n, timestamps,
                        n_folds, edge_threshold, be_wr, config_name):
    """True expanding window WF with fresh scan per fold.
    Split into n_folds+1 equal segments. First segment = initial IS.
    Each subsequent segment = OOS, with all prior segments as IS."""

    seg_size = n // (n_folds + 1)
    all_wf_results = []
    pattern_appearances = Counter()  # pattern stability tracking

    for fold in range(n_folds):
        # Expanding: IS grows each fold, OOS is always the next segment
        # fold 0: IS=[0, seg), OOS=[seg, 2*seg)
        # fold 1: IS=[0, 2*seg), OOS=[2*seg, 3*seg)
        # fold N-1: IS=[0, N*seg), OOS=[N*seg, n)
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n

        is_start_date = pd.Timestamp(timestamps[0]).strftime('%Y-%m-%d')
        is_end_date = pd.Timestamp(timestamps[min(is_end - 1, n - 1)]).strftime('%Y-%m-%d')
        oos_start_date = pd.Timestamp(timestamps[oos_start]).strftime('%Y-%m-%d')
        oos_end_date = pd.Timestamp(timestamps[min(oos_end - 1, n - 1)]).strftime('%Y-%m-%d')

        # Fresh universe scan on IS
        is_patterns = scan_universe(
            signal_index, ['LONG', 'SHORT'], tp, sl, opens, highs, lows, n,
            bar_start=0, bar_end=is_end,
            min_trades=MIN_TRADES, edge_threshold=edge_threshold, be_wr=be_wr
        )

        # Track pattern stability
        for r in is_patterns:
            pattern_appearances[(r['pattern'], r['direction'])] += 1

        # OOS test with IS-discovered patterns
        oos_trades = []
        for r in is_patterns:
            oos_sigs = [s for s in signal_index[r['pattern']] if oos_start <= s < oos_end]
            oos_trades.extend(bt_signals(oos_sigs, r['direction'], tp, sl, opens, highs, lows, n))

        oos_port = portfolio_1pos(oos_trades)
        oos_stats = calc_stats(oos_port)
        n_long = sum(1 for r in is_patterns if r['direction'] == 'LONG')
        n_short = sum(1 for r in is_patterns if r['direction'] == 'SHORT')

        wf_r = {
            'fold': fold + 1,
            'is_period': f"{is_start_date} ~ {is_end_date}",
            'oos_period': f"{oos_start_date} ~ {oos_end_date}",
            'is_bars': is_end,
            'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_patterns),
            'is_long': n_long, 'is_short': n_short,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_compound': oos_stats['compound_pnl'],
            'oos_mdd': oos_stats['compound_mdd'],
            'oos_pf': oos_stats['pf'],
            'oos_positive': oos_stats['pnl'] > 0,
        }
        all_wf_results.append(wf_r)

    return all_wf_results, pattern_appearances


# ===================================================================
# Full improved pipeline
# ===================================================================

def full_pipeline_v3(cfg, signal_index, opens, highs, lows, n, timestamps):
    """Complete v3 pipeline for one TP/SL config."""
    tp = cfg['tp']; sl = cfg['sl']; name = cfg['name']
    be_wr = sl / (tp + sl) * 100
    rr = tp / sl

    print(f"\n{'='*70}")
    print(f"  {name} | R:R={rr:.2f} | BE WR={be_wr:.1f}%")
    print(f"  Win={tp*LEVERAGE - FEE_PCT:+.2f}% | Loss={-sl*LEVERAGE - FEE_PCT:-.2f}%")
    print(f"{'='*70}")

    # ── Step 1: Full-period scan (reference baseline) ──
    print(f"\n  [1] Full-period scan (reference only — NOT used for WF)")
    full_patterns = scan_universe(
        signal_index, ['LONG', 'SHORT'], tp, sl, opens, highs, lows, n,
        0, n, MIN_TRADES, 0, be_wr  # edge_threshold=0 for full count
    )
    full_patterns.sort(key=lambda x: x['pnl'], reverse=True)
    n_long = sum(1 for r in full_patterns if r['direction'] == 'LONG')
    n_short = sum(1 for r in full_patterns if r['direction'] == 'SHORT')
    print(f"      MC-pass (robust 3-seed): {len(full_patterns)} ({n_long}L+{n_short}S)")

    # Full-period portfolio stats
    all_trades = []
    for r in full_patterns:
        all_trades.extend(bt_signals(signal_index[r['pattern']], r['direction'], tp, sl, opens, highs, lows, n))
    full_port = portfolio_1pos(all_trades)
    full_stats = calc_stats(full_port)
    full_mc = mc_test_robust([t[2] for t in full_port])
    print(f"      Portfolio: {full_stats['trades']} tr, WR {full_stats['wr']:.1f}%, "
          f"Cpd {full_stats['compound_pnl']:.1f}%, MDD {full_stats['compound_mdd']:.1f}%, "
          f"PnL/MDD {full_stats['pnl_mdd']:.1f}x")

    # Top patterns
    print(f"\n      {'Pattern':<15} {'Dir':<6} {'Tr':>4} {'WR':>6} {'Edge':>6} {'PnL':>8} {'MC':>7}")
    print(f"      {'-'*56}")
    for r in full_patterns[:20]:
        print(f"      {r['pattern']:<15} {r['direction']:<6} {r['trades']:>4} "
              f"{r['wr']:>5.1f}% {r['edge']:>5.1f}pp {r['pnl']:>7.1f}% {r['mc_p']:>6.4f}")
    if len(full_patterns) > 20:
        print(f"      ... +{len(full_patterns)-20} more")

    # ── Step 2: Timeout analysis ──
    print(f"\n  [2] Timeout analysis")
    to_info = timeout_analysis(signal_index, full_patterns, tp, sl, opens, highs, lows, n)
    print(f"      Signals: {to_info['total_signals']} → Trades: {to_info['total_trades']} + "
          f"Timeout: {to_info['total_timeout']} ({to_info['timeout_rate']:.1f}%)")
    print(f"      Hold time: avg {to_info['avg_hold_bars']} bars ({to_info['avg_hold_hours']}h), "
          f"median {to_info['median_hold_bars']} bars")

    # ── Step 3: True Expanding Window WF ──
    print(f"\n  [3] True Expanding Window WF ({N_WF_FOLDS} folds, fresh scan per fold)")

    # Run with default edge threshold = 0
    wf_results, pattern_stability = expanding_window_wf(
        signal_index, tp, sl, opens, highs, lows, n, timestamps,
        N_WF_FOLDS, edge_threshold=0, be_wr=be_wr, config_name=name
    )

    for r in wf_results:
        sign = "+" if r['oos_positive'] else ""
        print(f"      Fold {r['fold']}: IS {r['is_period']} ({r['is_patterns']}pat={r['is_long']}L+{r['is_short']}S) "
              f"→ OOS {r['oos_period']} | {r['oos_trades']}tr WR{r['oos_wr']:>5.1f}% "
              f"PnL{sign}{r['oos_pnl']:.1f}% Cpd{sign}{r['oos_compound']:.1f}%")

    pos_folds = sum(1 for r in wf_results if r['oos_positive'])
    wf_oos_pnl = sum(r['oos_pnl'] for r in wf_results)
    wf_oos_trades = sum(r['oos_trades'] for r in wf_results)
    print(f"\n      WF Result: {pos_folds}/{N_WF_FOLDS} positive | "
          f"OOS total PnL: {wf_oos_pnl:.1f}% | OOS trades: {wf_oos_trades}")

    # ── Step 4: Pattern stability ──
    print(f"\n  [4] Pattern stability (appeared in N/{N_WF_FOLDS} folds)")
    stability_dist = Counter(pattern_stability.values())
    for k in sorted(stability_dist.keys(), reverse=True):
        pats = [(p, d) for (p, d), v in pattern_stability.items() if v == k]
        print(f"      {k}/{N_WF_FOLDS} folds: {len(pats)} patterns", end="")
        if len(pats) <= 10:
            print(f" — {', '.join(f'{p}({d[0]})' for p,d in sorted(pats))}")
        else:
            print(f" — {', '.join(f'{p}({d[0]})' for p,d in sorted(pats)[:8])}...")

    stable_patterns = [(p, d) for (p, d), v in pattern_stability.items() if v == N_WF_FOLDS]
    print(f"\n      Perfectly stable ({N_WF_FOLDS}/{N_WF_FOLDS}): {len(stable_patterns)} patterns")

    # ── Step 5: Edge threshold sensitivity ──
    print(f"\n  [5] Edge threshold sensitivity (expanding WF)")
    sensitivity_results = {}
    for et in EDGE_THRESHOLDS:
        wf_et, _ = expanding_window_wf(
            signal_index, tp, sl, opens, highs, lows, n, timestamps,
            N_WF_FOLDS, edge_threshold=et, be_wr=be_wr, config_name=name
        )
        pf = sum(1 for r in wf_et if r['oos_positive'])
        total_oos = sum(r['oos_pnl'] for r in wf_et)
        total_tr = sum(r['oos_trades'] for r in wf_et)
        avg_pat = np.mean([r['is_patterns'] for r in wf_et])
        sensitivity_results[et] = {
            'pos_folds': pf, 'oos_pnl': round(total_oos, 1),
            'oos_trades': total_tr, 'avg_patterns': round(avg_pat, 1)
        }
        print(f"      Edge>={et}pp: {pf}/{N_WF_FOLDS} positive | "
              f"OOS PnL: {total_oos:.1f}% | OOS trades: {total_tr} | "
              f"Avg patterns/fold: {avg_pat:.0f}")

    # ── Step 6: Stable-pattern-only WF ──
    print(f"\n  [6] Stable-pattern-only portfolio (patterns in ALL {N_WF_FOLDS} folds)")
    if stable_patterns:
        stable_trades = []
        for p, d in stable_patterns:
            stable_trades.extend(bt_signals(signal_index[p], d, tp, sl, opens, highs, lows, n))
        stable_port = portfolio_1pos(stable_trades)
        stable_stats = calc_stats(stable_port)
        print(f"      {len(stable_patterns)} patterns, {stable_stats['trades']} trades")
        print(f"      WR: {stable_stats['wr']:.1f}%, Safety: {stable_stats['safety']:.1f}pp")
        print(f"      Compound PnL: {stable_stats['compound_pnl']:.1f}%, "
              f"MDD: {stable_stats['compound_mdd']:.1f}%, "
              f"PnL/MDD: {stable_stats['pnl_mdd']:.1f}x")
    else:
        stable_stats = calc_stats([])
        print(f"      No perfectly stable patterns found.")

    return {
        'config': cfg,
        'full_scan': {'patterns': full_patterns, 'portfolio': full_stats, 'mc': full_mc},
        'timeout': to_info,
        'wf': wf_results,
        'wf_positive_folds': pos_folds,
        'wf_oos_pnl': wf_oos_pnl,
        'wf_oos_trades': wf_oos_trades,
        'pattern_stability': {f"{p}_{d}": v for (p, d), v in pattern_stability.items()},
        'stable_patterns': [list(x) for x in stable_patterns],
        'stable_portfolio': stable_stats,
        'edge_sensitivity': sensitivity_results,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("="*70)
    print("Universal TP/SL Study v3 — Improved Methodology")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    print(f"\nMethodology improvements over v2:")
    print(f"  1. True expanding window WF (no future data in IS)")
    print(f"  2. Fresh universe scan per fold (no pre-selection bias)")
    print(f"  3. Robust MC: 3 seeds ({MC_SEEDS}), max p-value < {MC_THRESHOLD}")
    print(f"  4. Edge threshold sensitivity: {EDGE_THRESHOLDS} pp")
    print(f"  5. Timeout analysis")
    print(f"  6. Cross-fold pattern stability tracking")

    csv_path = DATA_DIR / "btc_5m_270days_reclassified.csv"
    print(f"\nLoading: {csv_path}")
    t0 = time.time()
    opens, highs, lows, closes, types, timestamps, n = load_and_classify(str(csv_path))
    signal_index = build_signal_index(types, n)
    print(f"  {n} bars, {len(signal_index)} unique patterns ({time.time()-t0:.1f}s)")
    print(f"  Period: {pd.Timestamp(timestamps[0]).strftime('%Y-%m-%d')} ~ "
          f"{pd.Timestamp(timestamps[-1]).strftime('%Y-%m-%d')}")
    seg_size = n // (N_WF_FOLDS + 1)
    print(f"  WF: {N_WF_FOLDS} folds, segment ~{seg_size} bars (~{seg_size*5//60//24} days)")

    # Run both configs
    results = {}
    for cfg in CONFIGS:
        results[cfg['name']] = full_pipeline_v3(cfg, signal_index, opens, highs, lows, n, timestamps)

    # ── FAIR COMPARISON ──
    names = [c['name'] for c in CONFIGS]
    r0 = results[names[0]]; r1 = results[names[1]]

    print(f"\n{'='*70}")
    print(f"FAIR COMPARISON (v3 Improved Methodology)")
    print(f"{'='*70}")

    f0 = r0['full_scan']['portfolio']; f1 = r1['full_scan']['portfolio']
    s0 = r0['stable_portfolio']; s1 = r1['stable_portfolio']

    rows = [
        ("── Full Period (ref) ──",    "",              ""),
        ("MC-pass patterns",           f"{len(r0['full_scan']['patterns'])}",  f"{len(r1['full_scan']['patterns'])}"),
        ("Portfolio trades",           f"{f0['trades']}",                      f"{f1['trades']}"),
        ("WR / Safety",                f"{f0['wr']:.1f}% / {f0['safety']:.1f}pp",  f"{f1['wr']:.1f}% / {f1['safety']:.1f}pp"),
        ("Compound PnL",              f"{f0['compound_pnl']:.1f}%",           f"{f1['compound_pnl']:.1f}%"),
        ("MDD / PnL÷MDD",             f"{f0['compound_mdd']:.1f}% / {f0['pnl_mdd']:.1f}x",
                                       f"{f1['compound_mdd']:.1f}% / {f1['pnl_mdd']:.1f}x"),
        ("── Timeout ──",             "",              ""),
        ("Timeout rate",               f"{r0['timeout']['timeout_rate']:.1f}%",  f"{r1['timeout']['timeout_rate']:.1f}%"),
        ("Avg hold",                   f"{r0['timeout']['avg_hold_hours']}h",   f"{r1['timeout']['avg_hold_hours']}h"),
        ("── True Expanding WF ──",   "",              ""),
        ("Positive folds",             f"{r0['wf_positive_folds']}/{N_WF_FOLDS}",  f"{r1['wf_positive_folds']}/{N_WF_FOLDS}"),
        ("OOS total PnL",             f"{r0['wf_oos_pnl']:.1f}%",            f"{r1['wf_oos_pnl']:.1f}%"),
        ("OOS total trades",          f"{r0['wf_oos_trades']}",               f"{r1['wf_oos_trades']}"),
        ("── Stability ──",           "",              ""),
        ("Stable patterns (all folds)", f"{len(r0['stable_patterns'])}",       f"{len(r1['stable_patterns'])}"),
        ("Stable portfolio PnL",      f"{s0['compound_pnl']:.1f}%",           f"{s1['compound_pnl']:.1f}%"),
        ("Stable portfolio MDD",      f"{s0['compound_mdd']:.1f}%",           f"{s1['compound_mdd']:.1f}%"),
    ]

    print(f"\n  {'Metric':<28} {names[0]:>18} {names[1]:>18}")
    print(f"  {'-'*66}")
    for label, v0, v1 in rows:
        if label.startswith("──"):
            print(f"  {label:<28} {v0:>18} {v1:>18}")
        else:
            print(f"  {label:<28} {v0:>18} {v1:>18}")

    # Edge sensitivity comparison
    print(f"\n  Edge Threshold Sensitivity (OOS PnL / Positive Folds):")
    print(f"  {'Threshold':<12} {names[0]:>22} {names[1]:>22}")
    print(f"  {'-'*58}")
    for et in EDGE_THRESHOLDS:
        e0 = r0['edge_sensitivity'][et]; e1 = r1['edge_sensitivity'][et]
        print(f"  Edge>={et}pp     "
              f"{e0['oos_pnl']:>7.1f}% ({e0['pos_folds']}/{N_WF_FOLDS}, {e0['avg_patterns']:.0f}pat)  "
              f"{e1['oos_pnl']:>7.1f}% ({e1['pos_folds']}/{N_WF_FOLDS}, {e1['avg_patterns']:.0f}pat)")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'v3_improved',
        'improvements': [
            'True expanding window WF',
            'Fresh scan per fold',
            'Robust MC (3 seeds)',
            'Edge threshold sensitivity',
            'Timeout analysis',
            'Pattern stability tracking',
        ],
        'configs': CONFIGS,
        'constants': {'leverage': LEVERAGE, 'fee': FEE_PCT, 'mc_sims': MC_SIMS,
                       'mc_threshold': MC_THRESHOLD, 'mc_seeds': MC_SEEDS,
                       'min_trades': MIN_TRADES, 'max_bars': MAX_BARS,
                       'n_wf_folds': N_WF_FOLDS, 'edge_thresholds': EDGE_THRESHOLDS},
    }
    for name in names:
        r = results[name]
        output[name] = {
            'full_scan_patterns': r['full_scan']['patterns'],
            'full_scan_portfolio': r['full_scan']['portfolio'],
            'full_scan_mc': r['full_scan']['mc'],
            'timeout': r['timeout'],
            'wf': r['wf'],
            'wf_positive_folds': r['wf_positive_folds'],
            'wf_oos_pnl': r['wf_oos_pnl'],
            'wf_oos_trades': r['wf_oos_trades'],
            'stable_patterns': r['stable_patterns'],
            'stable_portfolio': r['stable_portfolio'],
            'edge_sensitivity': r['edge_sensitivity'],
        }

    out_path = RESULTS_DIR / "universal_tpsl_study_v3.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"\n  Total runtime: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
