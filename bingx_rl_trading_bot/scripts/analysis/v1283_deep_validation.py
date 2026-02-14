#!/usr/bin/env python3
"""
v1.28.3 Deep Validation Research — 51패턴 심층 검증

5-Phase:
  1. Portfolio Walk-Forward (3-fold) — 포트폴리오 수준 OOS 검증
  2. MC Drawdown Simulation — P95/P99 MDD tail risk 추정
  3. Slippage Sensitivity — 0/2/5/7/10 bps 슬리피지 영향
  4. Quarterly Edge Stability — 분기별 edge 일관성
  5. WF FAIL Pattern Removal — FAIL 8개 제거 시 포트폴리오 변화

Uses production classify_candle(), LEVERAGE=3, timeout DROP.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants
# ============================================================
FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 5000
UNI_TP = 2.0
UNI_SL = 3.0
BASELINE_WR = UNI_SL / (UNI_TP + UNI_SL) * 100  # 60.0%
MIN_TRADES = 20
EDGE_THRESHOLD = 5.0
MC_THRESHOLD = 0.01

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')


# ============================================================
# Core Functions (from scanner, verified)
# ============================================================

def load_and_classify(data_file):
    df = pd.read_csv(data_file)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()
    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, df.at[i, 'avg_body'])
        types.append(ct.value)
    df['candle_type'] = types
    return df


def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars, extra_fee=0.0):
    """Backtest with optional extra slippage fee."""
    total_fee = FEE_PCT + extra_fee
    trades = []
    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if direction == 'LONG':
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (tp_pct if dist_tp <= dist_sl else -sl_pct) * LEVERAGE - total_fee
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - total_fee))
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - total_fee))
                break
    return trades


def mc_test(pnls, n_sims=MC_SIMS):
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    rand_sums = signs @ pnls_arr
    return float(np.mean(rand_sums >= actual))


def portfolio_1pos(all_trades):
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


def calc_compound_stats(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'avg_win': 0, 'avg_loss': 0, 'max_consec_loss': 0}
    pnls = [t[2] for t in trades]
    equity = 100.0
    peak = 100.0
    mdd = 0
    wins = []
    losses = []
    consec_loss = 0
    max_consec = 0
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
            consec_loss = 0
        else:
            losses.append(p)
            consec_loss += 1
            max_consec = max(max_consec, consec_loss)
    total_pnl = equity - 100.0
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(total_pnl, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(np.mean(losses), 2) if losses else 0,
        'max_consec_loss': max_consec,
    }


def calc_simple_stats(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    pnls = [t[2] for t in trades]
    cum = 0
    peak = 0
    mdd = 0
    wins = []
    losses = []
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
        else:
            losses.append(p)
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(cum, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
    }


def discover_patterns(types, opens, highs, lows, n, start_idx, end_idx):
    """Discover patterns in [start_idx, end_idx) region."""
    signal_index = {}
    for i in range(max(2, start_idx), end_idx):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in signal_index:
            signal_index[pat] = []
        signal_index[pat].append(i)

    selected = {}
    for pat_name, sigs in signal_index.items():
        for direction in ['LONG', 'SHORT']:
            if len(sigs) < MIN_TRADES:
                continue
            trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            if len(trades) < MIN_TRADES:
                continue
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            edge = wr - BASELINE_WR
            if edge < EDGE_THRESHOLD:
                continue
            p = mc_test(pnls)
            if p >= MC_THRESHOLD:
                continue
            key = f"{pat_name}_{direction}"
            selected[key] = {'pattern': pat_name, 'direction': direction}
    return selected


# ============================================================
# Phase 1: Portfolio Walk-Forward (3-fold)
# ============================================================

def phase1_portfolio_wf(types, opens, highs, lows, n, signal_index):
    """Portfolio-level Walk-Forward: discover patterns in IS, test as portfolio in OOS."""
    print("\n" + "=" * 70)
    print("Phase 1: Portfolio Walk-Forward (3-fold)")
    print("=" * 70)

    n_folds = 3
    fold_size = n // n_folds
    results = []

    for fold in range(n_folds):
        oos_start = fold * fold_size
        oos_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

        # IS = everything except OOS
        is_ranges = []
        if oos_start > 0:
            is_ranges.append((0, oos_start))
        if oos_end < n:
            is_ranges.append((oos_end, n))

        # Discover patterns in IS
        is_selected = {}
        for is_start, is_end in is_ranges:
            found = discover_patterns(types, opens, highs, lows, n, is_start, is_end)
            is_selected.update(found)

        # Build OOS portfolio
        oos_trades = []
        for key, info in is_selected.items():
            pat = info['pattern']
            direction = info['direction']
            if pat in signal_index:
                sigs = [s for s in signal_index[pat] if oos_start <= s < oos_end]
                if sigs:
                    trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
                    oos_trades.extend(trades)

        portfolio = portfolio_1pos(oos_trades)
        stats_c = calc_compound_stats(portfolio)
        stats_s = calc_simple_stats(portfolio)

        wr = stats_c['wr']
        edge = wr - BASELINE_WR
        pnls = [t[2] for t in portfolio]
        mc_p = mc_test(pnls) if pnls else 1.0

        fold_result = {
            'fold': fold + 1,
            'oos_range': f"{oos_start}-{oos_end}",
            'is_patterns': len(is_selected),
            'oos_trades': stats_c['trades'],
            'oos_wr': wr,
            'oos_edge': round(edge, 1),
            'oos_pnl_compound': stats_c['pnl'],
            'oos_pnl_simple': stats_s['pnl'],
            'oos_mdd_compound': stats_c['mdd'],
            'oos_mdd_simple': stats_s['mdd'],
            'oos_pf': stats_c['pf'],
            'oos_mc_p': round(mc_p, 4),
            'oos_max_consec_loss': stats_c['max_consec_loss'],
        }
        results.append(fold_result)

        status = "PASS" if stats_c['pnl'] > 0 else "FAIL"
        print(f"\n  Fold {fold+1}: OOS [{oos_start}:{oos_end}]")
        print(f"    IS Patterns discovered: {len(is_selected)}")
        print(f"    OOS: {stats_c['trades']} trades, WR {wr:.1f}%, Edge {edge:.1f}pp")
        print(f"    OOS PnL: {stats_c['pnl']:.1f}% (compound), {stats_s['pnl']:.1f}% (simple)")
        print(f"    OOS MDD: {stats_c['mdd']:.1f}% (compound), {stats_s['mdd']:.1f}% (simple)")
        print(f"    PF: {stats_c['pf']:.2f}, MC p: {mc_p:.4f}, Max Consec Loss: {stats_c['max_consec_loss']}")
        print(f"    → {status}")

    positive = len([r for r in results if r['oos_pnl_compound'] > 0])
    avg_edge = np.mean([r['oos_edge'] for r in results])
    avg_wr = np.mean([r['oos_wr'] for r in results])

    print(f"\n  Summary: {positive}/{n_folds} positive, Avg OOS Edge: {avg_edge:.1f}pp, Avg OOS WR: {avg_wr:.1f}%")
    verdict = "STRONG" if positive == n_folds else ("PASS" if positive >= 2 else "FAIL")
    print(f"  Verdict: {verdict}")

    return {'folds': results, 'positive': positive, 'avg_edge': round(avg_edge, 1), 'avg_wr': round(avg_wr, 1), 'verdict': verdict}


# ============================================================
# Phase 2: MC Drawdown Simulation
# ============================================================

def phase2_mc_drawdown(types, opens, highs, lows, n, signal_index, current_patterns):
    """Monte Carlo drawdown: shuffle trade order 10k times."""
    print("\n" + "=" * 70)
    print("Phase 2: MC Drawdown Simulation (10,000 sims)")
    print("=" * 70)

    # Get all portfolio trades
    all_trades = []
    for key, info in current_patterns.items():
        pat = info['pattern']
        direction = info['direction']
        if pat in signal_index:
            sigs = signal_index[pat]
            trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            all_trades.extend(trades)

    portfolio = portfolio_1pos(all_trades)
    pnls = np.array([t[2] for t in portfolio])
    n_trades = len(pnls)
    print(f"  Portfolio trades: {n_trades}")

    actual_stats = calc_compound_stats(portfolio)
    print(f"  Actual: PnL {actual_stats['pnl']:.1f}%, MDD {actual_stats['mdd']:.1f}%, WR {actual_stats['wr']:.1f}%")

    # MC simulation — shuffle trade PnLs
    n_sims = 10000
    mc_mdds = []
    mc_pnls = []
    mc_max_consec = []

    for _ in range(n_sims):
        shuffled = np.random.permutation(pnls)
        equity = 100.0
        peak = 100.0
        mdd = 0
        consec = 0
        max_c = 0
        for p in shuffled:
            equity *= (1 + p / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > mdd:
                mdd = dd
            if p <= 0:
                consec += 1
                max_c = max(max_c, consec)
            else:
                consec = 0
        mc_mdds.append(mdd)
        mc_pnls.append(equity - 100.0)
        mc_max_consec.append(max_c)

    mc_mdds = np.array(mc_mdds)
    mc_pnls = np.array(mc_pnls)
    mc_max_consec = np.array(mc_max_consec)

    p50_mdd = np.percentile(mc_mdds, 50)
    p75_mdd = np.percentile(mc_mdds, 75)
    p90_mdd = np.percentile(mc_mdds, 90)
    p95_mdd = np.percentile(mc_mdds, 95)
    p99_mdd = np.percentile(mc_mdds, 99)
    loss_pct = np.mean(mc_pnls < 0) * 100

    print(f"\n  MC MDD Distribution ({n_sims} sims):")
    print(f"    P50: {p50_mdd:.1f}%")
    print(f"    P75: {p75_mdd:.1f}%")
    print(f"    P90: {p90_mdd:.1f}%")
    print(f"    P95: {p95_mdd:.1f}%")
    print(f"    P99: {p99_mdd:.1f}%")
    print(f"    Actual MDD: {actual_stats['mdd']:.1f}%")
    print(f"    Loss probability: {loss_pct:.2f}%")
    print(f"    Max Consecutive Loss — P50: {np.percentile(mc_max_consec, 50):.0f}, P95: {np.percentile(mc_max_consec, 95):.0f}, P99: {np.percentile(mc_max_consec, 99):.0f}")

    return {
        'n_trades': n_trades,
        'n_sims': n_sims,
        'actual_mdd': actual_stats['mdd'],
        'actual_pnl': actual_stats['pnl'],
        'mdd_p50': round(p50_mdd, 1),
        'mdd_p75': round(p75_mdd, 1),
        'mdd_p90': round(p90_mdd, 1),
        'mdd_p95': round(p95_mdd, 1),
        'mdd_p99': round(p99_mdd, 1),
        'loss_probability_pct': round(loss_pct, 2),
        'max_consec_loss_p95': int(np.percentile(mc_max_consec, 95)),
        'max_consec_loss_p99': int(np.percentile(mc_max_consec, 99)),
    }


# ============================================================
# Phase 3: Slippage Sensitivity
# ============================================================

def phase3_slippage(types, opens, highs, lows, n, signal_index, current_patterns):
    """Test portfolio performance under various slippage levels."""
    print("\n" + "=" * 70)
    print("Phase 3: Slippage Sensitivity (0/2/5/7/10 bps)")
    print("=" * 70)

    slip_levels = [0, 0.02, 0.05, 0.07, 0.10]  # extra fee as %
    results = []

    print(f"\n  {'Slippage':>10} {'Trades':>7} {'WR':>7} {'PnL(C)':>10} {'MDD(C)':>8} {'PnL/MDD':>8} {'PF':>6} {'E[trade]':>9}")
    print("  " + "-" * 75)

    for slip in slip_levels:
        all_trades = []
        for key, info in current_patterns.items():
            pat = info['pattern']
            direction = info['direction']
            if pat in signal_index:
                sigs = signal_index[pat]
                trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n, extra_fee=slip)
                all_trades.extend(trades)

        portfolio = portfolio_1pos(all_trades)
        stats = calc_compound_stats(portfolio)
        pnl_mdd = round(stats['pnl'] / stats['mdd'], 1) if stats['mdd'] > 0 else 999
        e_trade = round(stats['pnl'] / stats['trades'], 2) if stats['trades'] > 0 else 0

        results.append({
            'slippage_bps': round(slip * 100, 1),
            'trades': stats['trades'],
            'wr': stats['wr'],
            'pnl': stats['pnl'],
            'mdd': stats['mdd'],
            'pnl_mdd': pnl_mdd,
            'pf': stats['pf'],
            'e_trade': e_trade,
        })

        print(f"  {slip*100:>8.1f}bps {stats['trades']:>7} {stats['wr']:>6.1f}% {stats['pnl']:>9.1f}% {stats['mdd']:>7.1f}% {pnl_mdd:>7.1f}x {stats['pf']:>5.2f} {e_trade:>8.2f}%")

    # Break-even slippage
    baseline_pnl = results[0]['pnl']
    be_slip = None
    for i in range(1, len(results)):
        if results[i]['pnl'] <= 0:
            # Interpolate
            prev = results[i-1]
            curr = results[i]
            if prev['pnl'] > 0 and curr['pnl'] <= 0:
                ratio = prev['pnl'] / (prev['pnl'] - curr['pnl'])
                be_slip = prev['slippage_bps'] + ratio * (curr['slippage_bps'] - prev['slippage_bps'])
            break

    if be_slip:
        print(f"\n  Break-even slippage: ~{be_slip:.1f} bps")
    else:
        print(f"\n  All slippage levels remain profitable")

    # Impact per 1bps
    if len(results) >= 2:
        pnl_0 = results[0]['pnl']
        pnl_10 = results[-1]['pnl']
        impact = (pnl_0 - pnl_10) / 10
        print(f"  PnL impact per 1bps slippage: ~{impact:.1f}% compound")

    return {'levels': results, 'be_slippage_bps': be_slip}


# ============================================================
# Phase 4: Quarterly Edge Stability
# ============================================================

def phase4_quarterly(types, opens, highs, lows, n, signal_index, current_patterns):
    """Analyze edge stability across quarters using bar-index based splits."""
    print("\n" + "=" * 70)
    print("Phase 4: Quarterly Edge Stability")
    print("=" * 70)

    # Get all portfolio trades with bar indices
    all_trades = []
    for key, info in current_patterns.items():
        pat = info['pattern']
        direction = info['direction']
        if pat in signal_index:
            sigs = signal_index[pat]
            trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            all_trades.extend(trades)

    portfolio = portfolio_1pos(all_trades)

    # Split into ~quarterly chunks (270 days / 4 ≈ 67.5 days ≈ 19440 bars per quarter)
    bars_per_q = n // 4
    quarters = []
    for q in range(4):
        q_start = q * bars_per_q
        q_end = (q + 1) * bars_per_q if q < 3 else n
        q_trades = [(eb, xb, pnl) for eb, xb, pnl in portfolio if q_start <= eb < q_end]
        quarters.append((q + 1, q_start, q_end, q_trades))

    print(f"\n  {'Quarter':>8} {'Bars':>12} {'Trades':>7} {'WR':>7} {'Edge':>7} {'PnL(S)':>9} {'MDD(S)':>8} {'PF':>6}")
    print("  " + "-" * 70)

    q_results = []
    for q_num, q_start, q_end, q_trades in quarters:
        if not q_trades:
            q_results.append({'quarter': q_num, 'trades': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'mdd': 0, 'pf': 0})
            print(f"  Q{q_num:>7} [{q_start:>5}:{q_end:<5}] {0:>7} {'N/A':>7} {'N/A':>7} {'N/A':>9} {'N/A':>8} {'N/A':>6}")
            continue

        stats = calc_simple_stats(q_trades)
        edge = stats['wr'] - BASELINE_WR

        q_results.append({
            'quarter': q_num,
            'trades': stats['trades'],
            'wr': stats['wr'],
            'edge': round(edge, 1),
            'pnl': stats['pnl'],
            'mdd': stats['mdd'],
            'pf': stats['pf'],
        })

        flag = " ***" if stats['pnl'] < 0 else ""
        print(f"  Q{q_num:>7} [{q_start:>5}:{q_end:<5}] {stats['trades']:>7} {stats['wr']:>6.1f}% {edge:>6.1f}pp {stats['pnl']:>8.1f}% {stats['mdd']:>7.1f}% {stats['pf']:>5.2f}{flag}")

    positive_q = len([q for q in q_results if q['pnl'] > 0])
    avg_edge = np.mean([q['edge'] for q in q_results if q['trades'] > 0])
    edge_std = np.std([q['edge'] for q in q_results if q['trades'] > 0])

    print(f"\n  Positive quarters: {positive_q}/4")
    print(f"  Edge: mean {avg_edge:.1f}pp, std {edge_std:.1f}pp")
    print(f"  Edge stability (CV): {edge_std/avg_edge:.2f}" if avg_edge > 0 else "  Edge stability: N/A")

    consistency = "STRONG" if positive_q == 4 else ("GOOD" if positive_q == 3 else ("WEAK" if positive_q == 2 else "FAIL"))
    print(f"  Verdict: {consistency}")

    return {'quarters': q_results, 'positive_quarters': positive_q, 'avg_edge': round(avg_edge, 1), 'edge_std': round(edge_std, 1), 'verdict': consistency}


# ============================================================
# Phase 5: WF FAIL Pattern Removal Impact
# ============================================================

def phase5_wf_fail_removal(types, opens, highs, lows, n, signal_index, current_patterns):
    """Compare: current 51 vs removing the 8 WF FAIL patterns."""
    print("\n" + "=" * 70)
    print("Phase 5: WF FAIL Pattern Removal Impact")
    print("=" * 70)

    # Load WF results from phase 1's per-pattern analysis
    filter_file = os.path.join(RESULTS_DIR, 'pattern_significance_filter.json')
    if not os.path.exists(filter_file):
        print("  ERROR: pattern_significance_filter.json not found")
        return {}

    with open(filter_file, 'r') as f:
        filter_data = json.load(f)

    verdicts = filter_data.get('verdicts', {})

    # Identify WF FAIL patterns in current set
    fail_in_current = []
    for key in current_patterns:
        if verdicts.get(key) == 'FAIL':
            fail_in_current.append(key)

    print(f"\n  WF FAIL patterns in current 51: {len(fail_in_current)}")
    for k in sorted(fail_in_current):
        info = current_patterns[k]
        print(f"    {k}")

    # Scenario A: Current 51 patterns
    all_trades_a = []
    for key, info in current_patterns.items():
        pat = info['pattern']
        direction = info['direction']
        if pat in signal_index:
            trades = bt_signals(signal_index[pat], direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            all_trades_a.extend(trades)
    port_a = portfolio_1pos(all_trades_a)
    stats_a = calc_compound_stats(port_a)
    stats_a_s = calc_simple_stats(port_a)

    # Scenario B: Remove WF FAIL patterns
    remaining = {k: v for k, v in current_patterns.items() if k not in fail_in_current}
    all_trades_b = []
    for key, info in remaining.items():
        pat = info['pattern']
        direction = info['direction']
        if pat in signal_index:
            trades = bt_signals(signal_index[pat], direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            all_trades_b.extend(trades)
    port_b = portfolio_1pos(all_trades_b)
    stats_b = calc_compound_stats(port_b)
    stats_b_s = calc_simple_stats(port_b)

    # Scenario C: WF PASS only
    pass_only = {k: v for k, v in current_patterns.items() if verdicts.get(k) == 'PASS'}
    all_trades_c = []
    for key, info in pass_only.items():
        pat = info['pattern']
        direction = info['direction']
        if pat in signal_index:
            trades = bt_signals(signal_index[pat], direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            all_trades_c.extend(trades)
    port_c = portfolio_1pos(all_trades_c)
    stats_c = calc_compound_stats(port_c)
    stats_c_s = calc_simple_stats(port_c)

    print(f"\n  {'Scenario':<30} {'Pats':>5} {'Trades':>7} {'WR':>7} {'PnL(S)':>9} {'MDD(S)':>8} {'PnL/MDD':>8} {'PF':>6}")
    print("  " + "-" * 80)

    for label, pats, stats_s, stats_comp in [
        ('A: Current 51', len(current_patterns), stats_a_s, stats_a),
        (f'B: Remove FAIL ({len(fail_in_current)})', len(remaining), stats_b_s, stats_b),
        (f'C: PASS only ({len(pass_only)})', len(pass_only), stats_c_s, stats_c),
    ]:
        pnl_mdd = round(stats_s['pnl'] / stats_s['mdd'], 1) if stats_s['mdd'] > 0 else 999
        print(f"  {label:<30} {pats:>5} {stats_s['trades']:>7} {stats_s['wr']:>6.1f}% {stats_s['pnl']:>8.1f}% {stats_s['mdd']:>7.1f}% {pnl_mdd:>7.1f}x {stats_s['pf']:>5.2f}")

    # Portfolio WF for each scenario
    print(f"\n  Portfolio-Level WF (3-fold) for each scenario:")
    scenarios = {
        'A_current': current_patterns,
        'B_no_fail': remaining,
        'C_pass_only': pass_only,
    }

    wf_comparison = {}
    for name, pats in scenarios.items():
        n_folds = 3
        fold_size = n // n_folds
        pos_folds = 0
        oos_edges = []

        for fold in range(n_folds):
            oos_start = fold * fold_size
            oos_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

            # IS discovery
            is_selected = {}
            for is_start, is_end in ([(0, oos_start)] if oos_start > 0 else []) + ([(oos_end, n)] if oos_end < n else []):
                found = discover_patterns(types, opens, highs, lows, n, is_start, is_end)
                is_selected.update(found)

            # Intersect with scenario patterns
            scenario_keys = set(pats.keys())
            is_in_scenario = {k: v for k, v in is_selected.items() if k in scenario_keys}

            # OOS
            oos_trades = []
            for key, info in is_in_scenario.items():
                pat = info['pattern']
                direction = info['direction']
                if pat in signal_index:
                    sigs = [s for s in signal_index[pat] if oos_start <= s < oos_end]
                    if sigs:
                        trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
                        oos_trades.extend(trades)

            port = portfolio_1pos(oos_trades)
            pnls = [t[2] for t in port]
            if pnls:
                wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                oos_edges.append(wr - BASELINE_WR)
                if sum(pnls) > 0:
                    pos_folds += 1
            else:
                oos_edges.append(0)

        avg_e = np.mean(oos_edges)
        wf_comparison[name] = {
            'positive_folds': pos_folds,
            'avg_oos_edge': round(avg_e, 1),
        }
        print(f"    {name}: {pos_folds}/3 positive, Avg OOS Edge: {avg_e:.1f}pp")

    return {
        'fail_patterns': fail_in_current,
        'scenarios': {
            'A_current': {'patterns': len(current_patterns), 'trades': stats_a_s['trades'], 'wr': stats_a_s['wr'], 'pnl': stats_a_s['pnl'], 'mdd': stats_a_s['mdd']},
            'B_no_fail': {'patterns': len(remaining), 'trades': stats_b_s['trades'], 'wr': stats_b_s['wr'], 'pnl': stats_b_s['pnl'], 'mdd': stats_b_s['mdd']},
            'C_pass_only': {'patterns': len(pass_only), 'trades': stats_c_s['trades'], 'wr': stats_c_s['wr'], 'pnl': stats_c_s['pnl'], 'mdd': stats_c_s['mdd']},
        },
        'wf_comparison': wf_comparison,
    }


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(42)
    t0 = time.time()

    print("=" * 70)
    print("v1.28.3 Deep Validation Research — 51 Patterns")
    print("Universal TP 2.0% / SL 3.0% | min_trades 20 | LEVERAGE 3x")
    print("=" * 70)

    # Load data
    print("\nLoading and classifying data...")
    df = load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    signal_index = build_signal_index(types, n)

    # Load current patterns
    with open(os.path.join(RESULTS_DIR, 'dynamic_patterns.json'), 'r') as f:
        dp = json.load(f)
    current_patterns = {k: {'pattern': v['pattern'], 'direction': v['direction']}
                       for k, v in dp['pattern_details'].items()}
    print(f"Current patterns: {len(current_patterns)} ({dp['pattern_count']['long']}L+{dp['pattern_count']['short']}S)")

    # Phase 1
    t1 = time.time()
    p1 = phase1_portfolio_wf(types, opens, highs, lows, n, signal_index)
    print(f"  [Phase 1 time: {time.time()-t1:.1f}s]")

    # Phase 2
    t2 = time.time()
    p2 = phase2_mc_drawdown(types, opens, highs, lows, n, signal_index, current_patterns)
    print(f"  [Phase 2 time: {time.time()-t2:.1f}s]")

    # Phase 3
    t3 = time.time()
    p3 = phase3_slippage(types, opens, highs, lows, n, signal_index, current_patterns)
    print(f"  [Phase 3 time: {time.time()-t3:.1f}s]")

    # Phase 4
    t4 = time.time()
    p4 = phase4_quarterly(types, opens, highs, lows, n, signal_index, current_patterns)
    print(f"  [Phase 4 time: {time.time()-t4:.1f}s]")

    # Phase 5
    t5 = time.time()
    p5 = phase5_wf_fail_removal(types, opens, highs, lows, n, signal_index, current_patterns)
    print(f"  [Phase 5 time: {time.time()-t5:.1f}s]")

    # ============================================================
    # Final Summary
    # ============================================================
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  1. Portfolio WF: {p1['verdict']} ({p1['positive']}/3 positive, Avg Edge {p1['avg_edge']}pp, Avg WR {p1['avg_wr']}%)")
    print(f"  2. MC Drawdown: P95 MDD={p2['mdd_p95']}%, P99 MDD={p2['mdd_p99']}%, Loss prob={p2['loss_probability_pct']}%")
    be_slip = p3.get('be_slippage_bps')
    slip_msg = 'All levels profitable' if be_slip is None else f'BE at {be_slip:.1f}bps'
    print(f"  3. Slippage: {slip_msg}")
    print(f"  4. Quarterly: {p4['verdict']} ({p4['positive_quarters']}/4 positive, Edge CV={p4['edge_std']/p4['avg_edge']:.2f})" if p4['avg_edge'] > 0 else f"  4. Quarterly: {p4['verdict']}")

    if p5:
        wf_b = p5.get('wf_comparison', {}).get('B_no_fail', {})
        print(f"  5. FAIL Removal: {len(p5.get('fail_patterns',[]))} FAIL patterns → WF {wf_b.get('positive_folds',0)}/3, Edge {wf_b.get('avg_oos_edge',0)}pp")

    print(f"\n  Total time: {total_time:.1f}s")

    # Save
    output = {
        'version': 'v1.28.3',
        'date': '2026-02-14',
        'patterns': len(current_patterns),
        'uni_tp': UNI_TP,
        'uni_sl': UNI_SL,
        'min_trades': MIN_TRADES,
        'phase1_portfolio_wf': p1,
        'phase2_mc_drawdown': p2,
        'phase3_slippage': p3,
        'phase4_quarterly': p4,
        'phase5_wf_fail_removal': p5,
    }

    out_path = os.path.join(RESULTS_DIR, 'v1283_deep_validation.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
