#!/usr/bin/env python3
"""
R:R Regime Bias Study — TP/SL 설정이 시장 레짐에 의존하는지 분석
================================================================

Core Question:
  현재 59패턴의 높은 WR이 진짜 패턴 edge인가, 아니면 하락장에서
  TP < SL (unfavorable R:R)이 자연스럽게 높은 WR을 만드는 것인가?

Insight:
  - SHORT, TP=1%, SL=4% → 하락장에서 1% 하락은 쉬움 → WR 높음 (regime gift)
  - SHORT, TP=3%, SL=1% → 하락장에서도 3% 하락 before 1% 반등은 어려움 → 진짜 edge
  - LONG, TP=3%, SL=1% → 하락장에서 3% 상승은 어려움 → 낮은 WR 예상
  - LONG, TP=1%, SL=3% → 하락장에서도 1% 반등은 자주 → regime gift

따라서:
  R:R >= 1 (TP >= SL) 패턴이 높은 WR → 시장 레짐 무관 genuine edge
  R:R < 1 (TP < SL) 패턴이 높은 WR → 레짐 의존 가능성 높음

Analysis:
  1. 전 패턴 R:R 분류 + 방향별 WR/PnL
  2. R:R >= 1 패턴만으로 포트폴리오 구성 → WF
  3. R:R < 1 패턴만으로 포트폴리오 구성 → WF 비교
  4. R:R threshold sweep (0.5, 0.75, 1.0, 1.25) → 최적점
  5. 반전 테스트: LONG 패턴을 SHORT으로, SHORT을 LONG으로 → 레짐 검증

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based)
  - Sizing: compound (1/N)
  - Fee: 0.30% capital-space
  - WF: 3-fold expanding window (270d overlap)
  - ATR scaling: BOTH_a14_w576_0.6-1.7
"""

import os
import sys
import json
import time
import logging

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.multi_position_diversification_study import (
    load_and_classify, load_patterns, build_signal_index,
    compute_atr_ratio, calc_stats, find_overlap_bar,
    FEE_PCT, LEVERAGE, FEE, MAX_BARS, BARS_PER_DAY,
    ATR_PERIOD, ATR_WINDOW, CLAMP_LO, CLAMP_HI,
    SLIPPAGE_BUFFER,
)
from scripts.analysis.fifo_vs_hedge_backtest_study import (
    build_signal_schedule, ActivePosition, pnl_mdd_ratio,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('rr_regime_bias')

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'rr_regime_bias_study.json')


# ===================================================================
# Pattern-filtered signal schedule
# ===================================================================

def build_filtered_schedule(patterns, sig_index, atr_ratio, opens, n_bars,
                            pattern_filter=None, flip_direction=False):
    """
    Build signal schedule with optional pattern filter and direction flip.

    pattern_filter: set of pattern names to include (None = all)
    flip_direction: if True, reverse LONG<->SHORT (regime test)
    """
    schedule = {}
    for p in patterns:
        if pattern_filter and p['pattern'] not in pattern_filter:
            continue

        direction = 1 if p['direction'] == 'LONG' else -1
        if flip_direction:
            direction = -direction

        base_tp, base_sl = p['tp'], p['sl']

        bars = sig_index.get(p['pattern'])
        if not bars:
            continue

        for sig_bar in bars:
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
            else:
                r = 1.0

            tp = base_tp * r + SLIPPAGE_BUFFER
            sl = max(0.1, base_sl * r - SLIPPAGE_BUFFER)

            if entry_bar not in schedule:
                schedule[entry_bar] = []
            schedule[entry_bar].append((p['pattern'], direction, tp, sl, entry_price, sig_bar))

    return schedule


# ===================================================================
# Simulation (reuse from timeout study)
# ===================================================================

def simulate_hedge(max_positions, schedule, opens, highs, lows,
                   lo_bar, hi_bar, timeout_bars=None):
    slots = {}
    slot_counter = 0
    results = []
    weight = 1.0 / max_positions
    signal_count = 0
    entered_count = 0
    skipped_count = 0
    n_bars = len(opens)

    def _close_slot(sid, exit_bar, exit_price, reason):
        pos = slots.pop(sid)
        pm = pos.direction * (exit_price / pos.entry_price - 1) * 100
        pnl = pm * LEVERAGE - FEE
        scaled = pnl * weight
        results.append({
            'entry_bar': pos.entry_bar, 'exit_bar': exit_bar,
            'scaled_pnl': scaled, 'raw_pnl': pnl,
            'reason': reason, 'direction': pos.direction,
            'pattern': pos.pattern,
        })

    for bar in range(lo_bar, hi_bar):
        if bar >= n_bars:
            break

        sids_to_close = []
        for sid, pos in slots.items():
            h, l, o = highs[bar], lows[bar], opens[bar]
            if pos.direction == 1:
                tp_hit = h >= pos.tp_price
                sl_hit = l <= pos.sl_price
            else:
                tp_hit = l <= pos.tp_price
                sl_hit = h >= pos.sl_price

            if tp_hit and sl_hit:
                tp_dist = abs(pos.tp_price - o)
                sl_dist = abs(pos.sl_price - o)
                if tp_dist <= sl_dist:
                    tp_hit, sl_hit = True, False
                else:
                    tp_hit, sl_hit = False, True

            if tp_hit:
                sids_to_close.append((sid, bar, pos.tp_price, 'TP'))
            elif sl_hit:
                sids_to_close.append((sid, bar, pos.sl_price, 'SL'))

        for sid, ebar, eprice, reason in sids_to_close:
            if sid in slots:
                _close_slot(sid, ebar, eprice, reason)

        if timeout_bars and timeout_bars > 0:
            for sid in list(slots):
                if sid not in slots:
                    continue
                pos = slots[sid]
                if bar - pos.entry_bar >= timeout_bars:
                    exit_price = opens[bar] if bar < n_bars else opens[-1]
                    _close_slot(sid, bar, exit_price, 'TIMEOUT')

        signals = schedule.get(bar)
        if not signals:
            continue

        active_patterns = {s.pattern for s in slots.values()}
        signals_sorted = sorted(signals, key=lambda s: -s[1])

        chosen = None
        for pat_name, direction, tp, sl, entry_price, sig_bar in signals_sorted:
            if pat_name in active_patterns:
                continue
            chosen = (pat_name, direction, tp, sl, entry_price, sig_bar)
            break

        if chosen is None:
            continue

        signal_count += 1
        pat_name, sig_dir, tp, sl, entry_price, sig_bar = chosen

        if len(slots) < max_positions:
            slot_counter += 1
            sid = f's{slot_counter}'
            pos = ActivePosition(pat_name, sig_dir, entry_price, tp, sl, bar, sig_bar)
            slots[sid] = pos
            entered_count += 1
        else:
            skipped_count += 1

    for sid in list(slots):
        exit_price = opens[min(hi_bar - 1, n_bars - 1)]
        _close_slot(sid, hi_bar - 1, exit_price, 'END')

    return results, signal_count, entered_count, skipped_count


def run_wf(max_positions, schedule, opens, highs, lows, overlap_bar, n_bars,
           timeout_bars=None):
    avail_bars = n_bars - overlap_bar
    quarter = avail_bars // 4
    folds = [
        (overlap_bar, overlap_bar + quarter,
         overlap_bar + quarter, overlap_bar + 2 * quarter),
        (overlap_bar, overlap_bar + 2 * quarter,
         overlap_bar + 2 * quarter, overlap_bar + 3 * quarter),
        (overlap_bar, overlap_bar + 3 * quarter,
         overlap_bar + 3 * quarter, min(overlap_bar + avail_bars, n_bars)),
    ]
    wf_results = []
    for fold_idx, (is_lo, is_hi, oos_lo, oos_hi) in enumerate(folds):
        oos_res, _, _, _ = simulate_hedge(
            max_positions, schedule, opens, highs, lows,
            oos_lo, oos_hi, timeout_bars)
        oos_port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in oos_res]
        oos_stats = calc_stats(oos_port)
        wf_results.append({
            'fold': fold_idx + 1,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['cmp_pnl'],
            'oos_mdd': oos_stats['cmp_mdd'],
            'pass': oos_stats['cmp_pnl'] > 0,
        })
    pass_count = sum(1 for f in wf_results if f['pass'])
    return wf_results, f"{'PASS' if pass_count == 3 else 'FAIL'} ({pass_count}/3)"


# ===================================================================
# Per-pattern analysis
# ===================================================================

def analyze_per_pattern(patterns, sig_index, atr_ratio, opens, highs, lows, n_bars,
                        overlap_bar):
    """Compute per-pattern stats within overlap region (270d)."""
    pattern_stats = []

    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue

        is_long = p['direction'] == 'LONG'
        direction = 1 if is_long else -1
        base_tp, base_sl = p['tp'], p['sl']
        rr = base_tp / base_sl if base_sl > 0 else 999

        wins = 0
        losses = 0
        total_pnl = 0.0
        trades_in_range = 0

        for sig_bar in bars:
            if sig_bar < overlap_bar:
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
            else:
                r = 1.0

            tp_pct = base_tp * r + SLIPPAGE_BUFFER
            sl_pct = max(0.1, base_sl * r - SLIPPAGE_BUFFER)

            if is_long:
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)

            # Resolve trade (no timeout)
            resolved = False
            for j in range(entry_bar + 1, n_bars):
                h, l, o = highs[j], lows[j], opens[j]
                if is_long:
                    tp_hit = h >= tp_price
                    sl_hit = l <= sl_price
                else:
                    tp_hit = l <= tp_price
                    sl_hit = h >= sl_price

                if tp_hit and sl_hit:
                    tp_dist = abs(tp_price - o)
                    sl_dist = abs(sl_price - o)
                    if tp_dist <= sl_dist:
                        tp_hit, sl_hit = True, False
                    else:
                        tp_hit, sl_hit = False, True

                if tp_hit:
                    pnl = tp_pct * LEVERAGE - FEE
                    wins += 1
                    total_pnl += pnl
                    resolved = True
                    break
                elif sl_hit:
                    pnl = -sl_pct * LEVERAGE - FEE
                    losses += 1
                    total_pnl += pnl
                    resolved = True
                    break

            if not resolved:
                # End-of-data: market close at last bar
                exit_price = opens[-1]
                if is_long:
                    pm = (exit_price / entry_price - 1) * 100
                else:
                    pm = (1 - exit_price / entry_price) * 100
                pnl = pm * LEVERAGE - FEE
                if pnl >= 0:
                    wins += 1
                else:
                    losses += 1
                total_pnl += pnl

            trades_in_range += 1

        total = wins + losses
        if total == 0:
            continue

        wr = wins / total * 100
        edge = total_pnl / total

        # Expected PnL under random walk (no regime edge)
        # For a random walk, P(TP hit before SL) ≈ SL / (TP + SL) [symmetric]
        random_wr = base_sl / (base_tp + base_sl) * 100
        random_edge = (random_wr / 100) * (base_tp * LEVERAGE - FEE) + \
                      (1 - random_wr / 100) * (-base_sl * LEVERAGE - FEE)
        wr_excess = wr - random_wr  # WR above random walk expectation

        pattern_stats.append({
            'pattern': p['pattern'],
            'direction': p['direction'],
            'tp': base_tp,
            'sl': base_sl,
            'rr': round(rr, 3),
            'trades': total,
            'wins': wins,
            'losses': losses,
            'wr': round(wr, 1),
            'total_pnl': round(total_pnl, 2),
            'edge_per_trade': round(edge, 3),
            'random_wr': round(random_wr, 1),
            'wr_excess': round(wr_excess, 1),
            'random_edge': round(random_edge, 3),
            'genuine_edge': round(edge - random_edge, 3),
        })

    return sorted(pattern_stats, key=lambda x: -x['rr'])


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    types = df['rctype'].values

    patterns = load_patterns()
    sig_index = build_signal_index(types, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    overlap_bar = find_overlap_bar(df)
    logger.info(f"Overlap bar: {overlap_bar}")

    output = {}

    # ==================================================================
    # SECTION 1: PER-PATTERN R:R ANALYSIS
    # ==================================================================
    print("\n" + "=" * 90)
    print("  SECTION 1: PER-PATTERN R:R ANALYSIS (270d overlap, no timeout)")
    print("  Random WR = SL/(TP+SL) — WR expected under symmetric random walk")
    print("  WR Excess = actual WR - random WR — genuine pattern edge over regime")
    print("=" * 90)

    pat_stats = analyze_per_pattern(patterns, sig_index, atr_ratio, opens, highs, lows,
                                    n_bars, overlap_bar)
    output['section1_per_pattern'] = pat_stats

    # Group by R:R bucket
    rr_high = [p for p in pat_stats if p['rr'] >= 1.0]    # Favorable R:R
    rr_mid = [p for p in pat_stats if 0.5 <= p['rr'] < 1.0]
    rr_low = [p for p in pat_stats if p['rr'] < 0.5]

    print(f"\n  R:R >= 1.0 (favorable, {len(rr_high)} patterns) — regime-resistant edge candidates:")
    print(f"  {'Pattern':>15s} {'Dir':>5s} | {'TP':>5s} {'SL':>5s} {'R:R':>5s} | {'Tr':>4s} {'WR':>6s} {'RandWR':>7s} {'Excess':>7s} | {'Edge':>7s} {'GenEdge':>8s}")
    print("  " + "-" * 95)
    for p in rr_high:
        marker = " ★" if p['wr_excess'] > 10 else ""
        print(f"  {p['pattern']:>15s} {p['direction']:>5s} | {p['tp']:5.2f} {p['sl']:5.2f} {p['rr']:5.2f} | "
              f"{p['trades']:4d} {p['wr']:5.1f}% {p['random_wr']:6.1f}% {p['wr_excess']:>+6.1f}% | "
              f"{p['edge_per_trade']:>+7.3f} {p['genuine_edge']:>+8.3f}{marker}")

    print(f"\n  0.5 <= R:R < 1.0 (moderate, {len(rr_mid)} patterns):")
    print(f"  {'Pattern':>15s} {'Dir':>5s} | {'TP':>5s} {'SL':>5s} {'R:R':>5s} | {'Tr':>4s} {'WR':>6s} {'RandWR':>7s} {'Excess':>7s} | {'Edge':>7s} {'GenEdge':>8s}")
    print("  " + "-" * 95)
    for p in rr_mid:
        marker = " ★" if p['wr_excess'] > 10 else ""
        print(f"  {p['pattern']:>15s} {p['direction']:>5s} | {p['tp']:5.2f} {p['sl']:5.2f} {p['rr']:5.2f} | "
              f"{p['trades']:4d} {p['wr']:5.1f}% {p['random_wr']:6.1f}% {p['wr_excess']:>+6.1f}% | "
              f"{p['edge_per_trade']:>+7.3f} {p['genuine_edge']:>+8.3f}{marker}")

    print(f"\n  R:R < 0.5 (unfavorable, {len(rr_low)} patterns) — most regime-dependent:")
    print(f"  {'Pattern':>15s} {'Dir':>5s} | {'TP':>5s} {'SL':>5s} {'R:R':>5s} | {'Tr':>4s} {'WR':>6s} {'RandWR':>7s} {'Excess':>7s} | {'Edge':>7s} {'GenEdge':>8s}")
    print("  " + "-" * 95)
    for p in rr_low:
        marker = " ★" if p['wr_excess'] > 10 else ""
        print(f"  {p['pattern']:>15s} {p['direction']:>5s} | {p['tp']:5.2f} {p['sl']:5.2f} {p['rr']:5.2f} | "
              f"{p['trades']:4d} {p['wr']:5.1f}% {p['random_wr']:6.1f}% {p['wr_excess']:>+6.1f}% | "
              f"{p['edge_per_trade']:>+7.3f} {p['genuine_edge']:>+8.3f}{marker}")

    # Summary stats
    for label, group in [('R:R>=1.0', rr_high), ('0.5<=R:R<1.0', rr_mid), ('R:R<0.5', rr_low)]:
        if group:
            avg_wr = np.mean([p['wr'] for p in group])
            avg_excess = np.mean([p['wr_excess'] for p in group])
            avg_genuine = np.mean([p['genuine_edge'] for p in group])
            total_trades = sum(p['trades'] for p in group)
            print(f"\n  {label}: {len(group)} patterns, {total_trades} trades, "
                  f"avg WR {avg_wr:.1f}%, avg WR excess {avg_excess:+.1f}pp, "
                  f"avg genuine edge {avg_genuine:+.3f}%/trade")

    # ==================================================================
    # SECTION 2: R:R THRESHOLD PORTFOLIO COMPARISON
    # ==================================================================
    print("\n" + "=" * 90)
    print("  SECTION 2: R:R THRESHOLD PORTFOLIOS (N=9, various timeout)")
    print("=" * 90)

    rr_thresholds = [0.0, 0.5, 0.75, 1.0, 1.25]
    timeout_modes = [('NT', None), ('T288', 288), ('T864', 864)]

    section2 = {}
    print(f"\n  {'RR_min':>6s} {'Timeout':>7s} | {'#Pat':>5s} {'Trades':>7s} | {'WR':>6s} {'PF':>6s} | "
          f"{'CmpPnL':>9s} {'CmpMDD':>8s} {'PnL/MDD':>8s} | {'WF':>10s}")
    print("  " + "-" * 90)

    for rr_min in rr_thresholds:
        # Select patterns with R:R >= threshold
        selected = {p['pattern'] for p in pat_stats if p['rr'] >= rr_min}
        n_pat = len(selected)

        for to_label, tb in timeout_modes:
            label = f"rr{rr_min}_{to_label}"

            sched = build_filtered_schedule(
                patterns, sig_index, atr_ratio, opens, n_bars,
                pattern_filter=selected if rr_min > 0 else None)

            results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge(
                9, sched, opens, highs, lows, overlap_bar, n_bars, tb)
            portfolio = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in results]
            stats = calc_stats(portfolio)
            ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])

            wf_folds, wf_verdict = run_wf(9, sched, opens, highs, lows,
                                           overlap_bar, n_bars, tb)

            section2[label] = {
                **stats,
                'n_patterns': n_pat,
                'pnl_mdd_ratio': ratio,
                'rr_threshold': rr_min,
                'timeout': tb,
                'wf_verdict': wf_verdict,
                'wf_folds': wf_folds,
            }

            print(f"  {rr_min:6.2f} {to_label:>7s} | {n_pat:5d} {stats['trades']:7d} | "
                  f"{stats['wr']:5.1f}% {stats['pf']:6.2f} | "
                  f"{stats['cmp_pnl']:>+9.1f} {stats['cmp_mdd']:8.1f} {ratio:8.2f} | "
                  f"{wf_verdict:>10s}")

    output['section2_rr_threshold'] = section2

    # ==================================================================
    # SECTION 3: WR EXCESS FILTER (genuine edge patterns only)
    # ==================================================================
    print("\n" + "=" * 90)
    print("  SECTION 3: WR EXCESS FILTER (WR_excess > threshold)")
    print("  Only patterns where WR significantly exceeds random walk expectation")
    print("=" * 90)

    excess_thresholds = [0, 5, 10, 15, 20]
    section3 = {}

    print(f"\n  {'Excess':>7s} {'Timeout':>7s} | {'#Pat':>5s} {'Trades':>7s} | {'WR':>6s} | "
          f"{'CmpPnL':>9s} {'CmpMDD':>8s} {'PnL/MDD':>8s} | {'WF':>10s}")
    print("  " + "-" * 85)

    for excess_min in excess_thresholds:
        selected = {p['pattern'] for p in pat_stats if p['wr_excess'] >= excess_min}
        n_pat = len(selected)

        for to_label, tb in [('NT', None), ('T864', 864)]:
            label = f"excess{excess_min}_{to_label}"

            sched = build_filtered_schedule(
                patterns, sig_index, atr_ratio, opens, n_bars,
                pattern_filter=selected if excess_min > 0 else None)

            results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge(
                9, sched, opens, highs, lows, overlap_bar, n_bars, tb)
            portfolio = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in results]
            stats = calc_stats(portfolio)
            ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])

            wf_folds, wf_verdict = run_wf(9, sched, opens, highs, lows,
                                           overlap_bar, n_bars, tb)

            section3[label] = {
                **stats,
                'n_patterns': n_pat,
                'pnl_mdd_ratio': ratio,
                'wf_verdict': wf_verdict,
                'wf_folds': wf_folds,
            }

            print(f"  {excess_min:>6d}pp {to_label:>7s} | {n_pat:5d} {stats['trades']:7d} | "
                  f"{stats['wr']:5.1f}% | "
                  f"{stats['cmp_pnl']:>+9.1f} {stats['cmp_mdd']:8.1f} {ratio:8.2f} | "
                  f"{wf_verdict:>10s}")

    output['section3_wr_excess'] = section3

    # ==================================================================
    # SECTION 4: DIRECTION FLIP TEST (regime validation)
    # ==================================================================
    print("\n" + "=" * 90)
    print("  SECTION 4: DIRECTION FLIP TEST")
    print("  If edge is genuine: flipped direction should LOSE money")
    print("  If edge is regime: flipped direction might also win (regime carries both)")
    print("=" * 90)

    section4 = {}
    flip_tests = [
        ('normal_all', None, False),
        ('flipped_all', None, True),
        ('normal_rr1', {p['pattern'] for p in pat_stats if p['rr'] >= 1.0}, False),
        ('flipped_rr1', {p['pattern'] for p in pat_stats if p['rr'] >= 1.0}, True),
    ]

    for to_label, tb in [('NT', None), ('T864', 864)]:
        print(f"\n  Timeout: {to_label}")
        print(f"  {'Mode':>15s} | {'#Pat':>5s} {'Trades':>7s} | {'WR':>6s} | "
              f"{'CmpPnL':>9s} {'CmpMDD':>8s} {'PnL/MDD':>8s} | {'WF':>10s}")
        print("  " + "-" * 85)

        for label, pfilter, flip in flip_tests:
            full_label = f"{label}_{to_label}"
            n_pat = len(pfilter) if pfilter else len(patterns)

            sched = build_filtered_schedule(
                patterns, sig_index, atr_ratio, opens, n_bars,
                pattern_filter=pfilter, flip_direction=flip)

            results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge(
                9, sched, opens, highs, lows, overlap_bar, n_bars, tb)
            portfolio = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in results]
            stats = calc_stats(portfolio)
            ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])

            wf_folds, wf_verdict = run_wf(9, sched, opens, highs, lows,
                                           overlap_bar, n_bars, tb)

            section4[full_label] = {
                **stats,
                'n_patterns': n_pat,
                'pnl_mdd_ratio': ratio,
                'flipped': flip,
                'wf_verdict': wf_verdict,
                'wf_folds': wf_folds,
            }

            print(f"  {label:>15s} | {n_pat:5d} {stats['trades']:7d} | "
                  f"{stats['wr']:5.1f}% | "
                  f"{stats['cmp_pnl']:>+9.1f} {stats['cmp_mdd']:8.1f} {ratio:8.2f} | "
                  f"{wf_verdict:>10s}")

    output['section4_flip_test'] = section4

    # ==================================================================
    # SECTION 5: GENUINE EDGE PATTERNS (R:R >= 1 AND WR excess > 10pp)
    # ==================================================================
    print("\n" + "=" * 90)
    print("  SECTION 5: GENUINE EDGE CANDIDATES")
    print("  Criteria: R:R >= 1.0 AND WR excess > 10pp over random walk")
    print("=" * 90)

    genuine = [p for p in pat_stats if p['rr'] >= 1.0 and p['wr_excess'] > 10]
    output['section5_genuine'] = genuine

    print(f"\n  Found {len(genuine)} genuine edge patterns:")
    print(f"  {'Pattern':>15s} {'Dir':>5s} | {'TP':>5s} {'SL':>5s} {'R:R':>5s} | "
          f"{'Tr':>4s} {'WR':>6s} {'RandWR':>7s} {'Excess':>7s} | {'GenEdge':>8s}")
    print("  " + "-" * 85)
    for p in sorted(genuine, key=lambda x: -x['genuine_edge']):
        print(f"  {p['pattern']:>15s} {p['direction']:>5s} | {p['tp']:5.2f} {p['sl']:5.2f} {p['rr']:5.2f} | "
              f"{p['trades']:4d} {p['wr']:5.1f}% {p['random_wr']:6.1f}% {p['wr_excess']:>+6.1f}% | "
              f"{p['genuine_edge']:>+8.3f}")

    if genuine:
        genuine_set = {p['pattern'] for p in genuine}
        for to_label, tb in [('NT', None), ('T864', 864)]:
            sched = build_filtered_schedule(
                patterns, sig_index, atr_ratio, opens, n_bars,
                pattern_filter=genuine_set)
            results, _, _, _ = simulate_hedge(
                9, sched, opens, highs, lows, overlap_bar, n_bars, tb)
            portfolio = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in results]
            stats = calc_stats(portfolio)
            ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
            wf_folds, wf_verdict = run_wf(9, sched, opens, highs, lows,
                                           overlap_bar, n_bars, tb)

            print(f"\n  Portfolio ({to_label}): {stats['trades']} trades, WR {stats['wr']}%, "
                  f"PnL {stats['cmp_pnl']:+.1f}%, MDD {stats['cmp_mdd']:.1f}%, "
                  f"PnL/MDD {ratio:.2f}, WF {wf_verdict}")

    # ==================================================================
    # SECTION 6: VERDICT
    # ==================================================================
    print("\n" + "=" * 90)
    print("  SECTION 6: VERDICT")
    print("=" * 90)

    # Summarize key findings
    normal_nt = section4.get('normal_all_NT', {})
    flipped_nt = section4.get('flipped_all_NT', {})
    normal_864 = section4.get('normal_all_T864', {})
    flipped_864 = section4.get('flipped_all_T864', {})

    print(f"\n  Direction Flip Validation:")
    if flipped_nt:
        print(f"    Normal  (NT):  PnL {normal_nt.get('cmp_pnl', 0):+.1f}%, WR {normal_nt.get('wr', 0):.1f}%")
        print(f"    Flipped (NT):  PnL {flipped_nt.get('cmp_pnl', 0):+.1f}%, WR {flipped_nt.get('wr', 0):.1f}%")
        if flipped_nt.get('cmp_pnl', 0) < 0:
            print(f"    → Flip loses money → edge has genuine direction component")
        else:
            print(f"    → WARNING: Flip also profits → regime may dominate over pattern edge")

    if flipped_864:
        print(f"    Normal  (T864): PnL {normal_864.get('cmp_pnl', 0):+.1f}%")
        print(f"    Flipped (T864): PnL {flipped_864.get('cmp_pnl', 0):+.1f}%")

    # R:R analysis summary
    print(f"\n  R:R Distribution:")
    print(f"    R:R >= 1.0: {len(rr_high)} patterns ({len(rr_high)/len(pat_stats)*100:.0f}%)")
    print(f"    0.5 <= R:R < 1.0: {len(rr_mid)} patterns ({len(rr_mid)/len(pat_stats)*100:.0f}%)")
    print(f"    R:R < 0.5: {len(rr_low)} patterns ({len(rr_low)/len(pat_stats)*100:.0f}%)")
    if genuine:
        print(f"    Genuine edge (R:R>=1 & excess>10pp): {len(genuine)} patterns")

    output['section6_verdict'] = {
        'rr_distribution': {
            'rr_ge_1': len(rr_high),
            'rr_05_1': len(rr_mid),
            'rr_lt_05': len(rr_low),
            'genuine_edge': len(genuine),
        },
        'flip_test_nt': {
            'normal_pnl': normal_nt.get('cmp_pnl', 0),
            'flipped_pnl': flipped_nt.get('cmp_pnl', 0),
        },
    }

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Results saved to {OUTPUT_FILE}")
    print("=" * 90)


if __name__ == '__main__':
    main()
