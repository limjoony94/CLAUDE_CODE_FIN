#!/usr/bin/env python3
"""
SL Threshold Study — Live Performance Diagnostic
=================================================
Purpose: 최근 라이브 거래에서 SL<1.0% 패턴 즉사 문제 확인 후
         SL threshold별 패턴 필터링의 영향을 체계적으로 검증.

Analysis:
  1. Current 50 patterns SL threshold 분포 분석
  2. SL threshold별 시나리오 백테스트 (compound, 1-pos-at-a-time)
  3. Expanding Window WF (3-fold) per scenario
  4. Direction balance (L/S ratio) 영향 분석
  5. Holding time vs SL size 상관관계
  6. MC test per scenario (3-seed robust)
  7. Break-even margin (safety) 분석

Protocol: production classify_candle, LEVERAGE=3, timeout DROP, 0.10% fee
Reference: universal_tpsl_study_v3.py
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

# Production classify_candle import (MANDATORY — no self-made classification)
sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

# ===================================================================
# Constants — match Standard Research Protocol
# ===================================================================
FEE_PCT = 0.10       # 0.05% x 2
LEVERAGE = 3
MAX_BARS = 500        # timeout threshold
MC_SIMS = 10000
MC_THRESHOLD = 0.01
MC_SEEDS = [42, 123, 7777]  # multi-seed robustness
N_WF_FOLDS = 3       # expanding window folds (match scanner)

# SL threshold scenarios to test
SL_THRESHOLDS = [0.5, 0.7, 1.0, 1.5, 2.0]


# ===================================================================
# Core functions (from universal_tpsl_study_v3.py)
# ===================================================================

def load_and_classify(csv_path):
    """Load data and classify candles using production function."""
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
    """Build pattern -> bar indices mapping."""
    idx = {}
    for i in range(2, n):
        pat = "{}-{}-{}".format(types[i-2], types[i-1], types[i])
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals_detailed(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest with per-pattern TP/SL. Returns (trades_list, n_timeout).
    Each trade: (entry_bar, exit_bar, pnl_pct, hold_bars)."""
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
                # Same-bar: use distance from bar open (NOT entry) — critical fix
                bo = opens[j]
                pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append((eb, j, pnl, j - eb))
                exited = True
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE_PCT, j - eb))
                exited = True
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT, j - eb))
                exited = True
                break
        if not exited:
            n_timeout += 1
    return trades, n_timeout


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest, timeout dropped."""
    trades, _ = bt_signals_detailed(signal_bars, direction, tp_pct, sl_pct,
                                     opens, highs, lows, n_bars)
    return trades


def portfolio_1pos(all_trades):
    """1-position-at-a-time serialization."""
    if not all_trades:
        return []
    all_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for t in all_trades:
        eb = t[0]
        xb = t[1]
        if eb > last_exit:
            filtered.append(t)
            last_exit = xb
    return filtered


def calc_stats(trades):
    """Calculate comprehensive stats with compound sizing."""
    if not trades:
        return {'pnl': 0, 'compound_pnl': 0, 'trades': 0, 'wr': 0,
                'compound_mdd': 0, 'pf': 0, 'pnl_mdd': 0,
                'avg_win': 0, 'avg_loss': 0, 'safety': 0, 'max_cl': 0,
                'avg_hold': 0, 'median_hold': 0}
    pnls = [t[2] for t in trades]
    holds = [t[3] for t in trades] if len(trades[0]) > 3 else []
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Simple PnL
    cum = 0; peak = 0; mdd = 0; cl = 0; max_cl = 0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        if peak - cum > mdd: mdd = peak - cum
        if p > 0: cl = 0
        else: cl += 1; max_cl = max(max_cl, cl)

    # Compound equity
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
    avg_h = float(np.mean(holds)) if holds else 0
    med_h = float(np.median(holds)) if holds else 0

    return {
        'pnl': round(cum, 2),
        'compound_pnl': round((eq - 1) * 100, 2),
        'trades': len(pnls),
        'wr': round(wr, 1),
        'compound_mdd': round(cmdd, 2),
        'pf': round(ws / ls, 2) if ls > 0 else 999,
        'pnl_mdd': round((eq - 1) * 100 / cmdd, 2) if cmdd > 0 else 999,
        'avg_win': round(aw, 3),
        'avg_loss': round(al, 3),
        'safety': round(wr - be, 1),
        'max_cl': max_cl,
        'avg_hold': round(avg_h, 1),
        'median_hold': round(med_h, 1),
    }


def mc_test_single(pnls_arr, seed, n_sims=MC_SIMS):
    """Single-seed MC sign randomization test."""
    rng = np.random.default_rng(seed)
    actual = np.sum(pnls_arr)
    signs = rng.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    return float(np.mean((signs @ pnls_arr) >= actual))


def mc_test_robust(pnls, seeds=MC_SEEDS, n_sims=MC_SIMS):
    """Multi-seed MC test. Returns max p-value (conservative)."""
    if len(pnls) < 5:
        return 1.0
    arr = np.array(pnls)
    p_values = [mc_test_single(arr, s, n_sims) for s in seeds]
    return max(p_values)


# ===================================================================
# Load dynamic patterns
# ===================================================================

def load_dynamic_patterns(json_path):
    """Load current dynamic patterns with per-pattern TP/SL."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    patterns = []
    for key, detail in data['pattern_details'].items():
        patterns.append({
            'key': key,
            'pattern': detail['pattern'],
            'direction': detail['direction'],
            'tp': detail['tp'],
            'sl': detail['sl'],
            'trades': detail['trades'],
            'wr': detail['wr'],
            'edge': detail['edge'],
            'mc_p': detail['mc_p'],
        })
    return patterns, data


# ===================================================================
# Expanding Window WF with per-pattern TP/SL
# ===================================================================

def expanding_wf_scenario(pattern_set, signal_index, opens, highs, lows, n, timestamps, n_folds):
    """Run expanding window WF for a specific pattern set (with per-pattern TP/SL).
    Unlike the reference which does fresh scan per fold, here we test a FIXED pattern set
    across WF folds to see if the fixed set maintains edge."""

    seg_size = n // (n_folds + 1)
    wf_results = []

    for fold in range(n_folds):
        oos_start = (fold + 1) * seg_size
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n

        oos_start_date = pd.Timestamp(timestamps[oos_start]).strftime('%Y-%m-%d')
        oos_end_date = pd.Timestamp(timestamps[min(oos_end - 1, n - 1)]).strftime('%Y-%m-%d')

        # Backtest pattern set on OOS period only
        oos_trades = []
        n_long = 0; n_short = 0
        for p in pattern_set:
            pat_name = p['pattern']
            direction = p['direction']
            tp = p['tp']; sl = p['sl']
            if pat_name not in signal_index:
                continue
            oos_sigs = [s for s in signal_index[pat_name] if oos_start <= s < oos_end]
            trades = bt_signals(oos_sigs, direction, tp, sl, opens, highs, lows, n)
            oos_trades.extend(trades)
            if direction == 'LONG': n_long += 1
            else: n_short += 1

        oos_port = portfolio_1pos(oos_trades)
        oos_stats = calc_stats(oos_port)

        wf_results.append({
            'fold': fold + 1,
            'oos_period': "{} ~ {}".format(oos_start_date, oos_end_date),
            'oos_bars': oos_end - oos_start,
            'n_patterns': len(pattern_set),
            'n_long': n_long, 'n_short': n_short,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_compound': oos_stats['compound_pnl'],
            'oos_mdd': oos_stats['compound_mdd'],
            'oos_pf': oos_stats['pf'],
            'oos_safety': oos_stats['safety'],
            'oos_positive': oos_stats['pnl'] > 0,
        })

    return wf_results


# ===================================================================
# Main analysis
# ===================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("SL Threshold Study — Live Performance Diagnostic")
    print("Date: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M')))
    print("=" * 70)

    # ── Load data ──
    csv_path = DATA_DIR / "btc_5m_270days_reclassified.csv"
    json_path = RESULTS_DIR / "dynamic_patterns.json"

    print("\nLoading data: {}".format(csv_path))
    opens, highs, lows, closes, types, timestamps, n = load_and_classify(str(csv_path))
    signal_index = build_signal_index(types, n)
    print("  {} bars, {} unique patterns".format(n, len(signal_index)))
    print("  Period: {} ~ {}".format(
        pd.Timestamp(timestamps[0]).strftime('%Y-%m-%d'),
        pd.Timestamp(timestamps[-1]).strftime('%Y-%m-%d')
    ))

    print("\nLoading patterns: {}".format(json_path))
    all_patterns, raw_data = load_dynamic_patterns(json_path)
    print("  {} patterns ({}L + {}S)".format(
        len(all_patterns),
        sum(1 for p in all_patterns if p['direction'] == 'LONG'),
        sum(1 for p in all_patterns if p['direction'] == 'SHORT'),
    ))

    # ── Section 1: SL distribution analysis ──
    print("\n" + "=" * 70)
    print("  [1] SL Distribution Analysis")
    print("=" * 70)

    sl_values = sorted(set(p['sl'] for p in all_patterns))
    print("\n  SL value distribution:")
    print("  {:>6} {:>8} {:>8} {:>8}  Patterns".format("SL%", "Count", "LONG", "SHORT"))
    print("  " + "-" * 56)
    for sl in sl_values:
        pats = [p for p in all_patterns if p['sl'] == sl]
        n_l = sum(1 for p in pats if p['direction'] == 'LONG')
        n_s = sum(1 for p in pats if p['direction'] == 'SHORT')
        names = ', '.join("{}({})".format(p['pattern'], p['direction'][0]) for p in pats)
        print("  {:>5.1f}% {:>8} {:>8} {:>8}  {}".format(sl, len(pats), n_l, n_s, names))

    # SL threshold groups
    print("\n  SL threshold groups (cumulative removal):")
    for thresh in SL_THRESHOLDS:
        removed = [p for p in all_patterns if p['sl'] < thresh]
        remaining = [p for p in all_patterns if p['sl'] >= thresh]
        r_l = sum(1 for p in remaining if p['direction'] == 'LONG')
        r_s = sum(1 for p in remaining if p['direction'] == 'SHORT')
        print("  SL<{:.1f}%: Remove {} patterns → {} remaining ({}L+{}S)".format(
            thresh, len(removed), len(remaining), r_l, r_s))

    # ── Section 2: Per-pattern analysis (SL vs practical metrics) ──
    print("\n" + "=" * 70)
    print("  [2] Per-Pattern SL Feasibility Analysis")
    print("=" * 70)

    print("\n  Spread+slippage assumption: 0.07% (BingX BTC 5m)")
    print("  SL adjusted = SL - 0.07% (effective SL after friction)")
    print("\n  {:>15} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7} {:>5} {:>8}".format(
        "Pattern", "Dir", "TP%", "SL%", "SL_eff", "BE_WR", "Act_WR", "Edge", "Status"))
    print("  " + "-" * 80)

    spread_slip = 0.07  # % total friction
    risky_patterns = []
    for p in sorted(all_patterns, key=lambda x: x['sl']):
        sl_eff = p['sl'] - spread_slip
        if sl_eff <= 0:
            status = "DEAD"
        elif sl_eff < 0.3:
            status = "DANGER"
        elif p['sl'] < 1.0:
            status = "RISKY"
        else:
            status = "OK"

        # BE WR with effective SL
        be_wr_eff = sl_eff / (p['tp'] + sl_eff) * 100 if (p['tp'] + sl_eff) > 0 else 99
        safety = p['wr'] - be_wr_eff

        d = p['direction'][0]
        print("  {:>15} {:>5} {:>5.1f} {:>5.1f} {:>5.2f}% {:>5.1f}% {:>5.1f}% {:>5.1f} {:>8}".format(
            p['pattern'], d, p['tp'], p['sl'], sl_eff, be_wr_eff, p['wr'], safety, status))

        if status in ('DEAD', 'DANGER', 'RISKY'):
            risky_patterns.append({
                'pattern': p['pattern'], 'direction': p['direction'],
                'sl': p['sl'], 'sl_eff': round(sl_eff, 2), 'status': status,
                'be_wr_eff': round(be_wr_eff, 1), 'safety': round(safety, 1),
            })

    print("\n  Risky patterns (SL < 1.0%): {}".format(len(risky_patterns)))
    for rp in risky_patterns:
        print("    {} ({}): SL {:.1f}% → eff {:.2f}% | BE WR {:.1f}% | Safety {:.1f}pp [{}]".format(
            rp['pattern'], rp['direction'][0], rp['sl'], rp['sl_eff'],
            rp['be_wr_eff'], rp['safety'], rp['status']))

    # ── Section 3: Scenario Backtests ──
    print("\n" + "=" * 70)
    print("  [3] Scenario Backtests (270d, compound, 1-pos-at-a-time)")
    print("=" * 70)

    # Define scenarios
    scenarios = [
        {'name': 'A_BASELINE (50pat)', 'min_sl': 0, 'desc': 'All 50 patterns'},
        {'name': 'B_SL>=0.7 (47pat)', 'min_sl': 0.7, 'desc': 'Remove SL<0.7%'},
        {'name': 'C_SL>=1.0 (42pat)', 'min_sl': 1.0, 'desc': 'Remove SL<1.0%'},
        {'name': 'D_SL>=1.5 (39pat)', 'min_sl': 1.5, 'desc': 'Remove SL<1.5%'},
        {'name': 'E_SL>=2.0 (33pat)', 'min_sl': 2.0, 'desc': 'Remove SL<2.0%'},
    ]

    scenario_results = {}
    for sc in scenarios:
        min_sl = sc['min_sl']
        pat_set = [p for p in all_patterns if p['sl'] >= min_sl] if min_sl > 0 else all_patterns
        n_l = sum(1 for p in pat_set if p['direction'] == 'LONG')
        n_s = sum(1 for p in pat_set if p['direction'] == 'SHORT')

        # Full-period backtest with per-pattern TP/SL
        all_trades = []
        pattern_stats = []
        total_timeout = 0
        for p in pat_set:
            if p['pattern'] not in signal_index:
                continue
            sigs = signal_index[p['pattern']]
            trades, n_to = bt_signals_detailed(sigs, p['direction'], p['tp'], p['sl'],
                                                opens, highs, lows, n)
            all_trades.extend(trades)
            total_timeout += n_to
            if trades:
                pnls = [t[2] for t in trades]
                pattern_stats.append({
                    'pattern': p['pattern'], 'direction': p['direction'],
                    'tp': p['tp'], 'sl': p['sl'],
                    'trades': len(trades), 'timeout': n_to,
                    'wr': round(len([x for x in pnls if x > 0]) / len(pnls) * 100, 1),
                    'pnl': round(sum(pnls), 2),
                    'avg_hold': round(np.mean([t[3] for t in trades]), 1),
                })

        port = portfolio_1pos(all_trades)
        stats = calc_stats(port)
        mc_p = mc_test_robust([t[2] for t in port]) if port else 1.0

        # Direction breakdown
        long_trades = [t for t in port if any(
            p['direction'] == 'LONG' and p['pattern'] in signal_index
            for p in pat_set
        )]

        scenario_results[sc['name']] = {
            'min_sl': min_sl,
            'patterns': len(pat_set),
            'n_long': n_l, 'n_short': n_s,
            'pattern_set': pat_set,
            'stats': stats,
            'mc_p': mc_p,
            'timeout': total_timeout,
            'pattern_stats': pattern_stats,
        }

        removed = [p for p in all_patterns if p['sl'] < min_sl] if min_sl > 0 else []
        removed_str = ', '.join("{}({})".format(p['pattern'], p['direction'][0]) for p in removed)

        print("\n  --- {} ---".format(sc['name']))
        print("  {}: {} patterns ({}L+{}S)".format(sc['desc'], len(pat_set), n_l, n_s))
        if removed:
            print("  Removed: {}".format(removed_str))
        print("  Portfolio: {} trades | WR {:.1f}% | Safety {:.1f}pp".format(
            stats['trades'], stats['wr'], stats['safety']))
        print("  Compound PnL: {:.1f}% | MDD: {:.1f}% | PnL/MDD: {:.1f}x | PF: {:.2f}".format(
            stats['compound_pnl'], stats['compound_mdd'], stats['pnl_mdd'], stats['pf']))
        print("  Avg win: {:.2f}% | Avg loss: {:.2f}% | Max CL: {} | MC p: {:.4f}".format(
            stats['avg_win'], stats['avg_loss'], stats['max_cl'], mc_p))
        print("  Avg hold: {:.1f} bars ({:.1f}h) | Median hold: {:.1f} bars".format(
            stats['avg_hold'], stats['avg_hold'] * 5 / 60, stats['median_hold']))
        print("  Timeout: {} (dropped from analysis)".format(total_timeout))

    # ── Section 4: Expanding Window WF per scenario ──
    print("\n" + "=" * 70)
    print("  [4] Expanding Window WF ({} folds, fixed pattern set)".format(N_WF_FOLDS))
    print("=" * 70)

    seg_size = n // (N_WF_FOLDS + 1)
    print("  Segment: ~{} bars (~{} days)".format(seg_size, seg_size * 5 // 60 // 24))

    wf_summary = {}
    for sc_name, sc_data in scenario_results.items():
        pat_set = sc_data['pattern_set']
        wf_results = expanding_wf_scenario(pat_set, signal_index, opens, highs, lows,
                                            n, timestamps, N_WF_FOLDS)

        pos_folds = sum(1 for r in wf_results if r['oos_positive'])
        oos_total_pnl = sum(r['oos_pnl'] for r in wf_results)
        oos_total_trades = sum(r['oos_trades'] for r in wf_results)
        avg_oos_wr = np.mean([r['oos_wr'] for r in wf_results if r['oos_trades'] > 0])
        avg_oos_safety = np.mean([r['oos_safety'] for r in wf_results if r['oos_trades'] > 0])

        print("\n  --- {} ({} patterns) ---".format(sc_name, sc_data['patterns']))
        for r in wf_results:
            sign = "+" if r['oos_pnl'] > 0 else ""
            print("    Fold {}: {} | {}tr WR{:>5.1f}% Safety{:>5.1f}pp | PnL {}{:.1f}% Cpd {}{:.1f}% MDD {:.1f}%".format(
                r['fold'], r['oos_period'], r['oos_trades'], r['oos_wr'],
                r['oos_safety'], sign, r['oos_pnl'], sign, r['oos_compound'],
                r['oos_mdd']))
        print("    WF Result: {}/{} positive | OOS total PnL: {:.1f}% | OOS trades: {} | Avg WR: {:.1f}%".format(
            pos_folds, N_WF_FOLDS, oos_total_pnl, oos_total_trades, avg_oos_wr))

        wf_summary[sc_name] = {
            'folds': wf_results,
            'pos_folds': pos_folds,
            'oos_total_pnl': round(oos_total_pnl, 1),
            'oos_total_trades': oos_total_trades,
            'avg_oos_wr': round(avg_oos_wr, 1),
            'avg_oos_safety': round(avg_oos_safety, 1),
        }

    # ── Section 5: Holding time vs SL correlation ──
    print("\n" + "=" * 70)
    print("  [5] Holding Time vs SL Size Correlation")
    print("=" * 70)

    sl_hold_data = []
    for p in all_patterns:
        if p['pattern'] not in signal_index:
            continue
        sigs = signal_index[p['pattern']]
        trades, _ = bt_signals_detailed(sigs, p['direction'], p['tp'], p['sl'],
                                         opens, highs, lows, n)
        if trades:
            holds = [t[3] for t in trades]
            win_holds = [t[3] for t in trades if t[2] > 0]
            loss_holds = [t[3] for t in trades if t[2] <= 0]
            sl_hold_data.append({
                'pattern': p['pattern'], 'direction': p['direction'],
                'sl': p['sl'], 'tp': p['tp'],
                'avg_hold': round(np.mean(holds), 1),
                'median_hold': round(np.median(holds), 1),
                'avg_win_hold': round(np.mean(win_holds), 1) if win_holds else 0,
                'avg_loss_hold': round(np.mean(loss_holds), 1) if loss_holds else 0,
            })

    sl_hold_data.sort(key=lambda x: x['sl'])
    print("\n  {:>15} {:>5} {:>5} {:>5} {:>8} {:>8} {:>10} {:>10}".format(
        "Pattern", "Dir", "TP%", "SL%", "AvgHold", "MedHold", "WinHold", "LossHold"))
    print("  " + "-" * 78)
    for d in sl_hold_data:
        print("  {:>15} {:>5} {:>5.1f} {:>5.1f} {:>7.1f}b {:>7.1f}b {:>9.1f}b {:>9.1f}b".format(
            d['pattern'], d['direction'][0], d['tp'], d['sl'],
            d['avg_hold'], d['median_hold'], d['avg_win_hold'], d['avg_loss_hold']))

    # Correlation
    sls = [d['sl'] for d in sl_hold_data]
    avg_holds = [d['avg_hold'] for d in sl_hold_data]
    if len(sls) > 2:
        corr = np.corrcoef(sls, avg_holds)[0, 1]
        print("\n  SL% vs Avg Hold correlation: {:.3f}".format(corr))
        # Group stats
        for thresh_lo, thresh_hi, label in [(0, 1.0, 'SL<1.0%'), (1.0, 2.0, '1.0<=SL<2.0%'),
                                              (2.0, 5.0, 'SL>=2.0%')]:
            group = [d for d in sl_hold_data if thresh_lo <= d['sl'] < thresh_hi]
            if group:
                gm = np.mean([d['avg_hold'] for d in group])
                gl = np.mean([d['avg_loss_hold'] for d in group])
                print("  {}: avg hold {:.1f} bars ({:.1f}h), avg loss hold {:.1f} bars ({:.1f}h)".format(
                    label, gm, gm * 5 / 60, gl, gl * 5 / 60))

    # ── Section 6: Direction balance analysis ──
    print("\n" + "=" * 70)
    print("  [6] Direction Balance Analysis")
    print("=" * 70)

    for sc_name, sc_data in scenario_results.items():
        n_l = sc_data['n_long']; n_s = sc_data['n_short']
        ratio = n_l / n_s if n_s > 0 else 999
        print("  {}: {}L + {}S (ratio {:.2f})".format(sc_name, n_l, n_s, ratio))

    # ── Section 7: Comparative Summary ──
    print("\n" + "=" * 70)
    print("  [7] COMPARATIVE SUMMARY")
    print("=" * 70)

    header = "{:<24} {:>5} {:>5} {:>7} {:>6} {:>7} {:>7} {:>7} {:>5} {:>5}".format(
        "Scenario", "Pats", "L/S", "Trades", "WR%", "CpdPnL", "MDD%", "PnL/M", "WF+", "MC")
    print("\n  " + header)
    print("  " + "-" * len(header))

    for sc_name in scenario_results:
        sd = scenario_results[sc_name]
        st = sd['stats']
        wf = wf_summary[sc_name]
        ls_str = "{}/{}".format(sd['n_long'], sd['n_short'])
        print("  {:<24} {:>5} {:>5} {:>7} {:>5.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}x {:>3}/{} {:>5.4f}".format(
            sc_name, sd['patterns'], ls_str, st['trades'], st['wr'],
            st['compound_pnl'], st['compound_mdd'], st['pnl_mdd'],
            wf['pos_folds'], N_WF_FOLDS, sd['mc_p']))

    # WF OOS comparison
    print("\n  WF OOS Summary:")
    header2 = "{:<24} {:>5} {:>8} {:>7} {:>6} {:>7}".format(
        "Scenario", "WF+", "OOS_PnL", "OOS_Tr", "OOS_WR", "Safety")
    print("  " + header2)
    print("  " + "-" * len(header2))
    for sc_name in scenario_results:
        wf = wf_summary[sc_name]
        print("  {:<24} {:>3}/{} {:>7.1f}% {:>7} {:>5.1f}% {:>6.1f}pp".format(
            sc_name, wf['pos_folds'], N_WF_FOLDS, wf['oos_total_pnl'],
            wf['oos_total_trades'], wf['avg_oos_wr'], wf['avg_oos_safety']))

    # ── Section 8: Recommendation ──
    print("\n" + "=" * 70)
    print("  [8] RECOMMENDATION")
    print("=" * 70)

    # Find best scenario by WF OOS PnL (most reliable metric)
    best_wf = max(wf_summary.items(), key=lambda x: x[1]['oos_total_pnl'])
    best_safety = max(wf_summary.items(), key=lambda x: x[1]['avg_oos_safety'])
    best_pnl = max(scenario_results.items(), key=lambda x: x[1]['stats']['compound_pnl'])

    print("\n  Best WF OOS PnL: {} ({:.1f}%)".format(best_wf[0], best_wf[1]['oos_total_pnl']))
    print("  Best WF OOS Safety: {} ({:.1f}pp)".format(best_safety[0], best_safety[1]['avg_oos_safety']))
    print("  Best IS Compound PnL: {} ({:.1f}%)".format(best_pnl[0], best_pnl[1]['stats']['compound_pnl']))

    # Specific risk check: SL<1.0% patterns
    sl_below_1 = [p for p in all_patterns if p['sl'] < 1.0]
    if sl_below_1:
        print("\n  SL<1.0% patterns ({}):".format(len(sl_below_1)))
        for p in sl_below_1:
            sl_eff = p['sl'] - spread_slip
            sl_lev = p['sl'] * LEVERAGE
            print("    {} ({}): SL {:.1f}% → lev {:.1f}% | eff SL {:.2f}% → eff lev {:.2f}%".format(
                p['pattern'], p['direction'][0], p['sl'], sl_lev, sl_eff, sl_eff * LEVERAGE))
            print("      In 5m: price move of just {:.2f}% triggers SL".format(p['sl']))

    # Auto-recommendation logic
    baseline_wf = wf_summary['A_BASELINE (50pat)']
    print("\n  Decision Matrix:")
    for sc_name in scenario_results:
        if sc_name == 'A_BASELINE (50pat)':
            continue
        sd = scenario_results[sc_name]
        wf = wf_summary[sc_name]
        bl_stats = scenario_results['A_BASELINE (50pat)']['stats']

        pnl_change = sd['stats']['compound_pnl'] - bl_stats['compound_pnl']
        mdd_change = sd['stats']['compound_mdd'] - bl_stats['compound_mdd']
        wf_pnl_change = wf['oos_total_pnl'] - baseline_wf['oos_total_pnl']
        safety_change = wf['avg_oos_safety'] - baseline_wf['avg_oos_safety']

        verdict = "NEUTRAL"
        if wf_pnl_change > 0 and safety_change >= 0:
            verdict = "ADOPT"
        elif wf_pnl_change > 0 and safety_change < -2:
            verdict = "CAUTION"
        elif wf_pnl_change < 0 and safety_change > 2:
            verdict = "SAFETY_PICK"
        elif wf_pnl_change < 0:
            verdict = "REJECT"

        print("  {} vs Baseline:".format(sc_name))
        print("    IS PnL: {:+.1f}% | IS MDD: {:+.1f}% | WF OOS PnL: {:+.1f}% | WF Safety: {:+.1f}pp → {}".format(
            pnl_change, mdd_change, wf_pnl_change, safety_change, verdict))

    # ── Save results ──
    output = {
        'timestamp': datetime.now().isoformat(),
        'study': 'SL Threshold Study',
        'protocol': {
            'leverage': LEVERAGE, 'fee': FEE_PCT, 'max_bars': MAX_BARS,
            'mc_sims': MC_SIMS, 'mc_seeds': MC_SEEDS, 'mc_threshold': MC_THRESHOLD,
            'n_wf_folds': N_WF_FOLDS, 'spread_slippage': spread_slip,
        },
        'risky_patterns': risky_patterns,
        'sl_hold_correlation': round(corr, 3) if len(sls) > 2 else None,
        'scenarios': {},
        'wf_summary': {},
        'recommendation': {
            'best_wf_oos': best_wf[0],
            'best_wf_safety': best_safety[0],
            'best_is_pnl': best_pnl[0],
        },
    }

    for sc_name, sd in scenario_results.items():
        output['scenarios'][sc_name] = {
            'min_sl': sd['min_sl'],
            'patterns': sd['patterns'],
            'n_long': sd['n_long'], 'n_short': sd['n_short'],
            'stats': sd['stats'],
            'mc_p': sd['mc_p'],
        }

    for sc_name, wf in wf_summary.items():
        output['wf_summary'][sc_name] = {
            'folds': wf['folds'],
            'pos_folds': wf['pos_folds'],
            'oos_total_pnl': wf['oos_total_pnl'],
            'oos_total_trades': wf['oos_total_trades'],
            'avg_oos_wr': wf['avg_oos_wr'],
            'avg_oos_safety': wf['avg_oos_safety'],
        }

    out_path = RESULTS_DIR / "sl_threshold_study.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("\n  Saved: {}".format(out_path))

    elapsed = time.time() - t0
    print("\n  Total runtime: {:.0f}s ({:.1f}min)".format(elapsed, elapsed / 60))
    print("=" * 70)


if __name__ == '__main__':
    main()
