#!/usr/bin/env python3
"""
Universal TP 1.5% / SL 1.0% Pattern Study
==========================================
Research: TP 1.5 / SL 1.0 (R:R = 1.50) 세팅으로 전체 패턴 스캔
- 현재: Universal TP 2.1 / SL 3.0 (R:R = 0.70, WR 의존형)
- 연구: Universal TP 1.5 / SL 1.0 (R:R = 1.50, edge 의존형)

Phase 1: 270d 전수 스캔 (3456 패턴 = 12^3 × 2방향)
  - TP 1.5 / SL 1.0 고정
  - MC filter (p < 0.01), min 10 trades
  - 1-pos-at-a-time 포트폴리오

Phase 2: 5-fold Walk-Forward 검증
  - MC pass 패턴으로 WF OOS 성과

Phase 3: 현재 TP 2.1/SL 3.0 대비 비교
  - 동일 패턴셋, 동일 기간으로 head-to-head

Phase 4: Compound PnL + MDD 산출

Protocol: production classify_candle, LEVERAGE=3, timeout DROP, 0.10% fee
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

# === Constants ===
FEE_PCT = 0.10       # 0.05% × 2
LEVERAGE = 3
MAX_BARS = 500        # timeout: skip trades not closed within 500 bars
MC_SIMS = 10000
MC_THRESHOLD = 0.01
MIN_TRADES = 10

# Study parameters
STUDY_TP = 1.5
STUDY_SL = 1.0

# Comparison baseline
BASELINE_TP = 2.1
BASELINE_SL = 3.0

CANDLE_TYPES = ['D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN']


# ===================================================================
# Core functions
# ===================================================================

def load_and_classify(csv_path):
    """Load CSV and classify candles using production classifier."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
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
    """Build index: pattern_name -> list of signal bars."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest given signal bars. Returns list of (entry_bar, exit_bar, pnl).
    Timeout trades (>MAX_BARS) are DROPPED, not included."""
    trades = []
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
                # Both hit same bar: use distance from open
                bo = opens[j]
                pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append((eb, j, pnl))
                exited = True
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE_PCT))
                exited = True
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT))
                exited = True
                break
        # timeout: DROP (not appended)
    return trades


def portfolio_1pos(all_trades):
    """1-position-at-a-time: sort by entry, skip overlapping."""
    if not all_trades:
        return []
    all_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def calc_stats(trades):
    """Calculate comprehensive statistics."""
    if not trades:
        return {'pnl': 0, 'compound_pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0,
                'compound_mdd': 0, 'pf': 0, 'pnl_mdd': 0, 'avg_win': 0, 'avg_loss': 0,
                'safety': 0, 'max_consec_loss': 0}

    pnls = [t[2] for t in trades]

    # Simple returns stats
    cum = 0; peak = 0; mdd = 0
    wins_list = []; losses_list = []
    consec_loss = 0; max_cl = 0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins_list.append(p)
            consec_loss = 0
        else:
            losses_list.append(p)
            consec_loss += 1
            if consec_loss > max_cl:
                max_cl = consec_loss

    # Compound returns
    equity = 1.0; c_peak = 1.0; c_mdd = 0
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > c_peak:
            c_peak = equity
        c_dd = (c_peak - equity) / c_peak * 100
        if c_dd > c_mdd:
            c_mdd = c_dd
    compound_pnl = (equity - 1) * 100

    avg_w = float(np.mean(wins_list)) if wins_list else 0
    avg_l = float(np.mean(losses_list)) if losses_list else 0
    be_wr = abs(avg_l) / (avg_w + abs(avg_l)) * 100 if (avg_w + abs(avg_l)) > 0 else 50
    wr = len(wins_list) / len(pnls) * 100
    wsum = sum(wins_list)
    lsum = sum(abs(x) for x in losses_list)

    return {
        'pnl': round(cum, 2),
        'compound_pnl': round(compound_pnl, 2),
        'trades': len(pnls),
        'wr': round(wr, 1),
        'mdd': round(mdd, 2),
        'compound_mdd': round(c_mdd, 2),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_mdd': round(compound_pnl / c_mdd, 2) if c_mdd > 0 else 999,
        'avg_win': round(avg_w, 3),
        'avg_loss': round(avg_l, 3),
        'safety': round(wr - be_wr, 1),
        'max_consec_loss': max_cl,
    }


def mc_test(pnls, n_sims=MC_SIMS):
    """Sign randomization MC test."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    rand_sums = signs @ pnls_arr
    return float(np.mean(rand_sums >= actual))


# ===================================================================
# Phase 1: Full Universe Scan — TP 1.5 / SL 1.0
# ===================================================================

def phase1_scan(types, opens, highs, lows, n, signal_index):
    """Scan all 3456 patterns with TP 1.5 / SL 1.0."""
    print("\n" + "="*70)
    print("Phase 1: Full Universe Scan — TP 1.5% / SL 1.0%")
    print("="*70)

    be_wr = STUDY_SL / (STUDY_TP + STUDY_SL) * 100
    print(f"  R:R = {STUDY_TP/STUDY_SL:.2f}, Break-even WR = {be_wr:.1f}%")
    print(f"  Win PnL = +{STUDY_TP * LEVERAGE - FEE_PCT:.2f}%, Loss PnL = -{STUDY_SL * LEVERAGE + FEE_PCT:.2f}%")

    results = []
    directions = ['LONG', 'SHORT']

    for pat_name, sig_bars in signal_index.items():
        for direction in directions:
            trades = bt_signals(sig_bars, direction, STUDY_TP, STUDY_SL, opens, highs, lows, n)

            if len(trades) < MIN_TRADES:
                continue

            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            total_pnl = sum(pnls)
            mc_p = mc_test(pnls)

            results.append({
                'pattern': pat_name,
                'direction': direction,
                'trades': len(trades),
                'wr': round(wr, 1),
                'edge': round(wr - be_wr, 1),
                'pnl': round(total_pnl, 2),
                'avg_pnl': round(total_pnl / len(trades), 3),
                'mc_p': round(mc_p, 4),
                'mc_pass': mc_p < MC_THRESHOLD,
            })

    # Summary
    total = len(results)
    mc_pass = [r for r in results if r['mc_pass']]
    positive = [r for r in results if r['pnl'] > 0]

    print(f"\n  Total patterns (>={MIN_TRADES} trades): {total}")
    print(f"  Positive PnL: {len(positive)} ({len(positive)/total*100:.1f}%)")
    print(f"  MC pass (p<{MC_THRESHOLD}): {len(mc_pass)}")

    if mc_pass:
        mc_pass.sort(key=lambda x: x['pnl'], reverse=True)
        print(f"\n  {'Pattern':<15} {'Dir':<6} {'Trades':>6} {'WR':>6} {'Edge':>6} {'PnL':>8} {'MC':>8}")
        print(f"  {'-'*60}")
        for r in mc_pass:
            print(f"  {r['pattern']:<15} {r['direction']:<6} {r['trades']:>6} {r['wr']:>5.1f}% {r['edge']:>5.1f}pp {r['pnl']:>7.1f}% {r['mc_p']:>7.4f}")

    return results, mc_pass


# ===================================================================
# Phase 2: Portfolio + Walk-Forward
# ===================================================================

def phase2_portfolio_wf(mc_patterns, types, opens, highs, lows, n, signal_index, timestamps):
    """Build 1-pos portfolio from MC-pass patterns + 5-fold WF."""
    print("\n" + "="*70)
    print("Phase 2: Portfolio Construction + 5-Fold Walk-Forward")
    print("="*70)

    # Full-period portfolio
    all_trades = []
    for r in mc_patterns:
        sig_bars = signal_index[r['pattern']]
        trades = bt_signals(sig_bars, r['direction'], STUDY_TP, STUDY_SL, opens, highs, lows, n)
        for t in trades:
            all_trades.append((*t, r['pattern'], r['direction']))

    portfolio_trades = portfolio_1pos([(t[0], t[1], t[2]) for t in all_trades])
    full_stats = calc_stats(portfolio_trades)
    portfolio_mc = mc_test([t[2] for t in portfolio_trades])

    print(f"\n  Full Period Portfolio ({len(mc_patterns)} patterns):")
    print(f"    Trades: {full_stats['trades']}")
    print(f"    WR: {full_stats['wr']:.1f}%")
    print(f"    Simple PnL: {full_stats['pnl']:.1f}%")
    print(f"    Compound PnL: {full_stats['compound_pnl']:.1f}%")
    print(f"    Compound MDD: {full_stats['compound_mdd']:.1f}%")
    print(f"    PnL/MDD: {full_stats['pnl_mdd']:.1f}x")
    print(f"    PF: {full_stats['pf']:.2f}")
    print(f"    Max Consec Loss: {full_stats['max_consec_loss']}")
    print(f"    Safety (WR - BE_WR): {full_stats['safety']:.1f}pp")
    print(f"    Portfolio MC: {portfolio_mc:.4f}")

    # 5-Fold Walk-Forward
    print(f"\n  5-Fold Walk-Forward:")
    fold_size = n // 5
    wf_results = []

    for fold in range(5):
        oos_start = fold * fold_size
        oos_end = (fold + 1) * fold_size if fold < 4 else n

        # IS = everything except OOS fold
        is_bars_set = set(range(0, oos_start)) | set(range(oos_end, n))
        oos_bars_set = set(range(oos_start, oos_end))

        # Re-scan on IS, test on OOS
        fold_pass_patterns = []
        for r in mc_patterns:
            sig_bars = signal_index[r['pattern']]
            is_sigs = [s for s in sig_bars if s in is_bars_set]
            is_trades = bt_signals(is_sigs, r['direction'], STUDY_TP, STUDY_SL, opens, highs, lows, n)
            if len(is_trades) < 5:
                continue
            is_pnls = [t[2] for t in is_trades]
            is_mc = mc_test(is_pnls)
            if is_mc < MC_THRESHOLD and sum(is_pnls) > 0:
                fold_pass_patterns.append(r)

        # OOS portfolio
        oos_all_trades = []
        for r in fold_pass_patterns:
            sig_bars = signal_index[r['pattern']]
            oos_sigs = [s for s in sig_bars if s in oos_bars_set]
            oos_trades = bt_signals(oos_sigs, r['direction'], STUDY_TP, STUDY_SL, opens, highs, lows, n)
            oos_all_trades.extend(oos_trades)

        oos_port = portfolio_1pos(oos_all_trades)
        oos_stats = calc_stats(oos_port)

        ts_start = pd.Timestamp(timestamps[oos_start]).strftime('%Y-%m-%d')
        ts_end = pd.Timestamp(timestamps[min(oos_end - 1, n - 1)]).strftime('%Y-%m-%d')

        result = {
            'fold': fold + 1,
            'period': f"{ts_start} ~ {ts_end}",
            'is_patterns': len(fold_pass_patterns),
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_compound': oos_stats['compound_pnl'],
            'oos_mdd': oos_stats['compound_mdd'],
            'oos_positive': oos_stats['pnl'] > 0,
        }
        wf_results.append(result)
        sign = "+" if result['oos_positive'] else ""
        print(f"    Fold {fold+1}: {result['period']} | {result['is_patterns']} pat | "
              f"{result['oos_trades']} trades | WR {result['oos_wr']:.1f}% | "
              f"PnL {sign}{result['oos_pnl']:.1f}%")

    positive_folds = sum(1 for r in wf_results if r['oos_positive'])
    print(f"\n    WF Result: {positive_folds}/5 folds positive")

    return full_stats, portfolio_mc, wf_results


# ===================================================================
# Phase 3: Head-to-Head vs Current TP 2.1/SL 3.0
# ===================================================================

def phase3_comparison(mc_patterns, types, opens, highs, lows, n, signal_index, timestamps):
    """Compare same pattern set with study TP/SL vs baseline TP/SL."""
    print("\n" + "="*70)
    print(f"Phase 3: Head-to-Head — TP {STUDY_TP}/SL {STUDY_SL} vs TP {BASELINE_TP}/SL {BASELINE_SL}")
    print("="*70)

    configs = [
        (f"Study (TP {STUDY_TP}/SL {STUDY_SL})", STUDY_TP, STUDY_SL),
        (f"Baseline (TP {BASELINE_TP}/SL {BASELINE_SL})", BASELINE_TP, BASELINE_SL),
    ]

    comparison = {}
    for label, tp, sl in configs:
        all_trades = []
        for r in mc_patterns:
            sig_bars = signal_index[r['pattern']]
            trades = bt_signals(sig_bars, r['direction'], tp, sl, opens, highs, lows, n)
            all_trades.extend(trades)

        portfolio = portfolio_1pos(all_trades)
        st = calc_stats(portfolio)
        mc_p = mc_test([t[2] for t in portfolio])
        comparison[label] = {**st, 'mc_p': mc_p}

    # Print comparison table
    print(f"\n  {'Metric':<22} {'Study':>15} {'Baseline':>15} {'Delta':>12}")
    print(f"  {'-'*65}")

    labels = list(comparison.keys())
    s = comparison[labels[0]]
    b = comparison[labels[1]]

    metrics = [
        ('Trades', 'trades', '', 0),
        ('Win Rate', 'wr', '%', 1),
        ('Simple PnL', 'pnl', '%', 1),
        ('Compound PnL', 'compound_pnl', '%', 1),
        ('Compound MDD', 'compound_mdd', '%', 1),
        ('PnL/MDD', 'pnl_mdd', 'x', 1),
        ('Profit Factor', 'pf', '', 2),
        ('Avg Win', 'avg_win', '%', 3),
        ('Avg Loss', 'avg_loss', '%', 3),
        ('Safety', 'safety', 'pp', 1),
        ('Max Consec Loss', 'max_consec_loss', '', 0),
        ('MC p-value', 'mc_p', '', 4),
    ]

    for label, key, unit, dec in metrics:
        sv = s[key]; bv = b[key]
        delta = sv - bv
        fmt = f".{dec}f"
        sign = "+" if delta > 0 else ""
        print(f"  {label:<22} {sv:>14{fmt}}{unit} {bv:>14{fmt}}{unit} {sign}{delta:>10{fmt}}{unit}")

    return comparison


# ===================================================================
# Phase 4: Also scan with baseline TP/SL for fair comparison
# ===================================================================

def phase4_baseline_scan(types, opens, highs, lows, n, signal_index):
    """Scan all patterns with baseline TP 2.1/SL 3.0 for comparison."""
    print("\n" + "="*70)
    print(f"Phase 4: Baseline Full Scan — TP {BASELINE_TP}% / SL {BASELINE_SL}%")
    print("="*70)

    be_wr = BASELINE_SL / (BASELINE_TP + BASELINE_SL) * 100

    mc_pass_baseline = []
    directions = ['LONG', 'SHORT']

    for pat_name, sig_bars in signal_index.items():
        for direction in directions:
            trades = bt_signals(sig_bars, direction, BASELINE_TP, BASELINE_SL, opens, highs, lows, n)
            if len(trades) < MIN_TRADES:
                continue
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            mc_p = mc_test(pnls)
            if mc_p < MC_THRESHOLD:
                mc_pass_baseline.append({
                    'pattern': pat_name, 'direction': direction,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'edge': round(wr - be_wr, 1), 'pnl': round(sum(pnls), 2),
                    'mc_p': round(mc_p, 4),
                })

    # Baseline portfolio
    all_trades = []
    for r in mc_pass_baseline:
        sig_bars = signal_index[r['pattern']]
        trades = bt_signals(sig_bars, r['direction'], BASELINE_TP, BASELINE_SL, opens, highs, lows, n)
        all_trades.extend(trades)

    portfolio = portfolio_1pos(all_trades)
    bl_stats = calc_stats(portfolio)
    bl_mc = mc_test([t[2] for t in portfolio])

    print(f"  Baseline MC pass patterns: {len(mc_pass_baseline)}")
    print(f"  Portfolio: {bl_stats['trades']} trades, WR {bl_stats['wr']:.1f}%, "
          f"Compound PnL {bl_stats['compound_pnl']:.1f}%, MDD {bl_stats['compound_mdd']:.1f}%, "
          f"PnL/MDD {bl_stats['pnl_mdd']:.1f}x, MC {bl_mc:.4f}")

    return mc_pass_baseline, bl_stats, bl_mc


# ===================================================================
# Main
# ===================================================================

def main():
    print("="*70)
    print("Universal TP 1.5% / SL 1.0% Pattern Study")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    # Load data
    csv_path = DATA_DIR / "btc_5m_270days_reclassified.csv"
    print(f"\nLoading data: {csv_path}")
    t0 = time.time()
    opens, highs, lows, closes, types, timestamps, n = load_and_classify(str(csv_path))
    print(f"  {n} bars loaded ({time.time()-t0:.1f}s)")
    print(f"  Period: {pd.Timestamp(timestamps[0]).strftime('%Y-%m-%d')} ~ "
          f"{pd.Timestamp(timestamps[-1]).strftime('%Y-%m-%d')}")

    # Build signal index
    signal_index = build_signal_index(types, n)
    unique_patterns = len(signal_index)
    print(f"  Unique 3-candle patterns: {unique_patterns}")

    # Study parameters
    print(f"\n  Study: TP {STUDY_TP}% / SL {STUDY_SL}% (R:R = {STUDY_TP/STUDY_SL:.2f})")
    print(f"  Baseline: TP {BASELINE_TP}% / SL {BASELINE_SL}% (R:R = {BASELINE_TP/BASELINE_SL:.2f})")
    print(f"  Leverage: {LEVERAGE}x, Fee: {FEE_PCT}%")
    print(f"  MC: {MC_SIMS} sims, threshold p<{MC_THRESHOLD}")

    # Phase 1: Study TP/SL scan
    all_results, mc_pass_study = phase1_scan(types, opens, highs, lows, n, signal_index)

    if not mc_pass_study:
        print("\n  NO MC-pass patterns found. Study ends here.")
        return

    # Phase 2: Portfolio + WF
    full_stats, portfolio_mc, wf_results = phase2_portfolio_wf(
        mc_pass_study, types, opens, highs, lows, n, signal_index, timestamps
    )

    # Phase 3: Head-to-head comparison (same study patterns, different TP/SL)
    comparison = phase3_comparison(
        mc_pass_study, types, opens, highs, lows, n, signal_index, timestamps
    )

    # Phase 4: Baseline independent scan for fair comparison
    mc_pass_baseline, bl_stats, bl_mc = phase4_baseline_scan(
        types, opens, highs, lows, n, signal_index
    )

    # === Final Summary ===
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n  {'Metric':<25} {'TP1.5/SL1.0':>15} {'TP2.1/SL3.0':>15}")
    print(f"  {'-'*55}")
    print(f"  {'R:R':<25} {STUDY_TP/STUDY_SL:>15.2f} {BASELINE_TP/BASELINE_SL:>15.2f}")
    print(f"  {'BE WR':<25} {STUDY_SL/(STUDY_TP+STUDY_SL)*100:>14.1f}% {BASELINE_SL/(BASELINE_TP+BASELINE_SL)*100:>14.1f}%")
    print(f"  {'MC-pass patterns':<25} {len(mc_pass_study):>15} {len(mc_pass_baseline):>15}")
    print(f"  {'Portfolio trades':<25} {full_stats['trades']:>15} {bl_stats['trades']:>15}")
    print(f"  {'Win Rate':<25} {full_stats['wr']:>14.1f}% {bl_stats['wr']:>14.1f}%")
    print(f"  {'Compound PnL':<25} {full_stats['compound_pnl']:>14.1f}% {bl_stats['compound_pnl']:>14.1f}%")
    print(f"  {'Compound MDD':<25} {full_stats['compound_mdd']:>14.1f}% {bl_stats['compound_mdd']:>14.1f}%")
    print(f"  {'PnL/MDD':<25} {full_stats['pnl_mdd']:>14.1f}x {bl_stats['pnl_mdd']:>14.1f}x")
    print(f"  {'Profit Factor':<25} {full_stats['pf']:>15.2f} {bl_stats['pf']:>15.2f}")
    print(f"  {'Safety (WR-BE)':<25} {full_stats['safety']:>14.1f}pp {bl_stats['safety']:>14.1f}pp")
    print(f"  {'Max Consec Loss':<25} {full_stats['max_consec_loss']:>15} {bl_stats['max_consec_loss']:>15}")
    print(f"  {'Portfolio MC':<25} {portfolio_mc:>15.4f} {bl_mc:>15.4f}")

    wf_pos = sum(1 for r in wf_results if r['oos_positive'])
    print(f"  {'WF positive folds':<25} {f'{wf_pos}/5':>15} {'N/A':>15}")

    # Overlap analysis
    study_set = {(r['pattern'], r['direction']) for r in mc_pass_study}
    baseline_set = {(r['pattern'], r['direction']) for r in mc_pass_baseline}
    overlap = study_set & baseline_set
    study_only = study_set - baseline_set
    baseline_only = baseline_set - study_set

    print(f"\n  Pattern Overlap Analysis:")
    print(f"    Both TP/SL: {len(overlap)} patterns")
    print(f"    TP1.5/SL1.0 only: {len(study_only)} patterns")
    print(f"    TP2.1/SL3.0 only: {len(baseline_only)} patterns")

    if overlap:
        print(f"\n    Shared patterns:")
        for pat, d in sorted(overlap):
            print(f"      {pat} ({d})")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'study_tp': STUDY_TP, 'study_sl': STUDY_SL,
            'baseline_tp': BASELINE_TP, 'baseline_sl': BASELINE_SL,
            'leverage': LEVERAGE, 'fee': FEE_PCT,
            'mc_sims': MC_SIMS, 'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES, 'max_bars': MAX_BARS,
        },
        'phase1_all': all_results,
        'phase1_mc_pass': mc_pass_study,
        'phase2_portfolio': full_stats,
        'phase2_portfolio_mc': portfolio_mc,
        'phase2_wf': wf_results,
        'phase3_comparison': comparison,
        'phase4_baseline_patterns': mc_pass_baseline,
        'phase4_baseline_stats': bl_stats,
        'phase4_baseline_mc': bl_mc,
        'overlap_analysis': {
            'both': [list(x) for x in sorted(overlap)],
            'study_only': [list(x) for x in sorted(study_only)],
            'baseline_only': [list(x) for x in sorted(baseline_only)],
        },
    }

    out_path = RESULTS_DIR / "universal_tp15_sl10_study.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
