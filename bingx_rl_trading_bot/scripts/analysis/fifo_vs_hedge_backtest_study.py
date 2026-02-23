#!/usr/bin/env python3
"""
FIFO vs Hedge vs Close-ALL Multi-Position Backtest Comparison
=============================================================

Context:
  기존 multi_position_diversification_study.py의 simulate_portfolio()는
  **방향 무관** 슬롯 선택(LONG+SHORT 혼합, CLOSE_OLDEST 없음)으로 시뮬레이션.
  프로덕션은 **FIFO 동일방향** 제약(_route_signal: same-dir + CLOSE_OLDEST) 사용.
  → N=5 배포 근거였던 백테스트가 프로덕션과 불일치.

Three strategies:
  A) Hedge:     방향 무관 혼합 허용, 꽉 차면 SKIP
  B) FIFO:      동일방향만, 반대 → CLOSE_OLDEST 1개, 빈 슬롯이면 즉시 OPEN
  C) Close-ALL: 동일방향만, 반대 → 전체 청산 → 즉시 OPEN 1개

Data: btc_5m_720days_binance.csv (207,360 bars)
Patterns: dynamic_patterns.json (59 patterns: 12L + 47S)
ATR scaling: BOTH_a14_w576_0.6-1.7 (production params)

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based same-bar resolution)
  - Sizing: compound (1/N per slot)
  - Fee: 0.10% x 2 = 0.30% capital-space
  - Timeout: 288 bars (24h) -> DROP
  - WF: 3-fold expanding window within 270d overlap region
"""

import os
import sys
import json
import time
import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup + imports from existing study
# ---------------------------------------------------------------------------
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fifo_vs_hedge')

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'fifo_vs_hedge_backtest_study.json')


# ===================================================================
# SIGNAL SCHEDULE (pre-compute all signals with ATR-scaled TP/SL)
# ===================================================================

def build_signal_schedule(patterns, sig_index, atr_ratio, opens, n_bars):
    """
    Pre-compute entry schedule: bar_idx -> list of signal tuples.

    Entry bar = sig_bar + 1 (next bar open).
    TP/SL are ATR-scaled, slippage-buffered price-space percentages.
    """
    schedule = {}
    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue
        direction = 1 if p['direction'] == 'LONG' else -1
        base_tp, base_sl = p['tp'], p['sl']

        for sig_bar in bars:
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            # ATR scaling
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
# MULTI-POSITION SIMULATOR
# ===================================================================

class ActivePosition:
    __slots__ = ('pattern', 'direction', 'entry_price', 'tp_pct', 'sl_pct',
                 'tp_price', 'sl_price', 'entry_bar', 'sig_bar')

    def __init__(self, pattern, direction, entry_price, tp_pct, sl_pct, entry_bar, sig_bar):
        self.pattern = pattern
        self.direction = direction
        self.entry_price = entry_price
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.entry_bar = entry_bar
        self.sig_bar = sig_bar

        if direction == 1:  # LONG
            self.tp_price = entry_price * (1 + tp_pct / 100)
            self.sl_price = entry_price * (1 - sl_pct / 100)
        else:  # SHORT
            self.tp_price = entry_price * (1 - tp_pct / 100)
            self.sl_price = entry_price * (1 + sl_pct / 100)


def simulate_strategy(strategy, max_positions, schedule, opens, highs, lows,
                      lo_bar, hi_bar):
    """
    Bar-by-bar multi-position simulation with strategy-specific routing.

    strategy: 'hedge' | 'fifo' | 'close_all'
    Returns: (results_list, close_oldest_count, close_oldest_pnls)
      results_list items: (entry_bar, exit_bar, scaled_pnl, exit_reason, direction, pattern)
    """
    slots = {}  # slot_id -> ActivePosition
    slot_counter = 0
    active_direction = None  # 1 or -1 or None (only for fifo/close_all)
    results = []
    weight = 1.0 / max_positions
    close_oldest_count = 0
    close_oldest_pnls = []

    def _close_slot(sid, exit_bar, exit_price, reason):
        nonlocal active_direction, close_oldest_count
        pos = slots.pop(sid)
        pm = pos.direction * (exit_price / pos.entry_price - 1) * 100
        pnl = pm * LEVERAGE - FEE
        scaled = pnl * weight
        results.append((pos.entry_bar, exit_bar, scaled, reason, pos.direction, pos.pattern))
        if reason == 'CLOSE_OLDEST':
            close_oldest_count += 1
            close_oldest_pnls.append(pnl)
        if not slots:
            active_direction = None

    n_bars = len(opens)

    for bar in range(lo_bar, hi_bar):
        # --- 1. Check TP/SL on active positions ---
        sids_to_close = []
        for sid, pos in slots.items():
            if bar >= n_bars:
                break
            h, l, o = highs[bar], lows[bar], opens[bar]

            if pos.direction == 1:  # LONG
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

        # --- 2. Check timeouts (288 bars) ---
        for sid in list(slots):
            if sid not in slots:
                continue
            pos = slots[sid]
            if bar - pos.entry_bar >= MAX_BARS:
                exit_price = opens[bar] if bar < n_bars else opens[-1]
                _close_slot(sid, bar, exit_price, 'TIMEOUT')

        # --- 3. Process signals for this bar ---
        signals = schedule.get(bar)
        if not signals:
            continue

        # Signal selection: LONG first (direction==1 before -1), no duplicate patterns
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

        pat_name, sig_dir, tp, sl, entry_price, sig_bar = chosen
        n_slots = len(slots)

        # --- Strategy-specific routing ---
        if strategy == 'hedge':
            if n_slots < max_positions:
                action = 'OPEN'
            else:
                continue  # SKIP

        elif strategy == 'fifo':
            if n_slots == 0:
                action = 'OPEN'
            elif active_direction == sig_dir:
                if n_slots < max_positions:
                    action = 'OPEN'
                else:
                    continue  # SKIP (full, same dir)
            else:
                # Opposite direction: CLOSE_OLDEST
                oldest_sid = min(slots, key=lambda s: slots[s].entry_bar)
                _close_slot(oldest_sid, bar, opens[bar], 'CLOSE_OLDEST')
                if not slots:
                    action = 'OPEN'
                else:
                    continue  # still same-direction slots remain

        elif strategy == 'close_all':
            if n_slots == 0:
                action = 'OPEN'
            elif active_direction == sig_dir:
                if n_slots < max_positions:
                    action = 'OPEN'
                else:
                    continue  # SKIP (full, same dir)
            else:
                # Opposite direction: CLOSE ALL
                for sid in list(slots):
                    _close_slot(sid, bar, opens[bar], 'CLOSE_ALL')
                action = 'OPEN'
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if action == 'OPEN':
            slot_counter += 1
            sid = f's{slot_counter}'
            pos = ActivePosition(pat_name, sig_dir, entry_price, tp, sl, bar, sig_bar)
            slots[sid] = pos
            active_direction = sig_dir

    # Close remaining open positions at last bar
    for sid in list(slots):
        exit_price = opens[hi_bar - 1] if hi_bar - 1 < n_bars else opens[-1]
        _close_slot(sid, hi_bar - 1, exit_price, 'END')

    return results, close_oldest_count, close_oldest_pnls


def pnl_mdd_ratio(cmp_pnl, cmp_mdd):
    """PnL/MDD ratio: positive PnL -> PnL/MDD, negative PnL -> -(abs_PnL/MDD)."""
    if cmp_mdd <= 0:
        return 0.0
    if cmp_pnl >= 0:
        return round(cmp_pnl / cmp_mdd, 2)
    return round(-abs(cmp_pnl) / cmp_mdd, 2)


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    # Load data
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
    avail_bars = n_bars - overlap_bar
    avail_days = avail_bars / BARS_PER_DAY
    logger.info(f"Overlap bar: {overlap_bar} | Available: {avail_bars} bars ({avail_days:.1f} days)")

    # Pre-compute signal schedule
    schedule = build_signal_schedule(patterns, sig_index, atr_ratio, opens, n_bars)
    total_signals = sum(len(v) for v in schedule.values())
    logger.info(f"Signal schedule: {len(schedule)} bars with signals, {total_signals} total signal-entries")

    output = {}

    # ==================================================================
    # SECTION 1: Strategy comparison — 270d overlap region (N=5)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: STRATEGY COMPARISON (270d overlap, N=5)")
    print("=" * 70)

    strategies = ['hedge', 'fifo', 'close_all']
    N = 5
    section1 = {}

    for strat in strategies:
        results, co_count, co_pnls = simulate_strategy(
            strat, N, schedule, opens, highs, lows, overlap_bar, n_bars
        )
        portfolio = [(r[0], r[1], r[2]) for r in results]
        stats = calc_stats(portfolio)
        stats['close_oldest_count'] = co_count
        stats['close_oldest_avg_pnl'] = round(float(np.mean(co_pnls)), 2) if co_pnls else 0

        # Direction breakdown
        long_trades = [r for r in results if r[4] == 1]
        short_trades = [r for r in results if r[4] == -1]
        stats['long_trades'] = len(long_trades)
        stats['short_trades'] = len(short_trades)

        # Exit reason breakdown
        exit_counts = {}
        for r in results:
            reason = r[3]
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
        stats['exit_reasons'] = exit_counts
        stats['pnl_mdd_ratio'] = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])

        section1[strat] = stats

    output['section1_strategy_comparison_270d'] = section1

    # Print comparison table
    print(f"\n  {'Metric':20s} | {'Hedge':>10s} | {'FIFO':>10s} | {'Close-ALL':>10s}")
    print("  " + "-" * 60)
    for key in ['trades', 'wr', 'pf', 'add_pnl', 'cmp_pnl', 'cmp_mdd', 'add_mdd',
                'pnl_mdd_ratio', 'long_trades', 'short_trades', 'close_oldest_count']:
        vals = [str(section1[s].get(key, '-')) for s in strategies]
        print(f"  {key:20s} | {vals[0]:>10s} | {vals[1]:>10s} | {vals[2]:>10s}")

    # Exit reasons detail
    print("\n  Exit reasons:")
    all_reasons = set()
    for strat in strategies:
        all_reasons.update(section1[strat].get('exit_reasons', {}).keys())
    for reason in sorted(all_reasons):
        vals = [str(section1[s].get('exit_reasons', {}).get(reason, 0)) for s in strategies]
        print(f"    {reason:20s} | {vals[0]:>10s} | {vals[1]:>10s} | {vals[2]:>10s}")

    # --- Sub-section: 720d full-period comparison (reference) ---
    print(f"\n  --- 720d full-period reference (patterns may not work before overlap) ---")
    section1_720d = {}
    for strat in strategies:
        results, co_count, co_pnls = simulate_strategy(
            strat, N, schedule, opens, highs, lows, 0, n_bars
        )
        portfolio = [(r[0], r[1], r[2]) for r in results]
        stats = calc_stats(portfolio)
        stats['close_oldest_count'] = co_count
        stats['pnl_mdd_ratio'] = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        section1_720d[strat] = stats

    output['section1_720d_reference'] = section1_720d

    print(f"  {'Metric':20s} | {'Hedge':>10s} | {'FIFO':>10s} | {'Close-ALL':>10s}")
    print("  " + "-" * 60)
    for key in ['trades', 'wr', 'cmp_pnl', 'cmp_mdd', 'pnl_mdd_ratio']:
        vals = [str(section1_720d[s].get(key, '-')) for s in strategies]
        print(f"  {key:20s} | {vals[0]:>10s} | {vals[1]:>10s} | {vals[2]:>10s}")

    # ==================================================================
    # SECTION 2: WF 3-fold validation within 270d overlap region
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: WALK-FORWARD 3-FOLD VALIDATION (within 270d)")
    print("=" * 70)

    # Compute WF fold boundaries within 270d overlap region
    quarter_bars = avail_bars // 4
    wf_folds_bars = [
        # (is_lo, is_hi, oos_lo, oos_hi) — all absolute bar indices
        (overlap_bar, overlap_bar + quarter_bars,
         overlap_bar + quarter_bars, overlap_bar + 2 * quarter_bars),
        (overlap_bar, overlap_bar + 2 * quarter_bars,
         overlap_bar + 2 * quarter_bars, overlap_bar + 3 * quarter_bars),
        (overlap_bar, overlap_bar + 3 * quarter_bars,
         overlap_bar + 3 * quarter_bars, min(overlap_bar + avail_bars, n_bars)),
    ]

    quarter_days = quarter_bars / BARS_PER_DAY

    section2 = {}
    for strat in strategies:
        strat_wf = []
        all_pass = True
        for fold_idx, (is_lo, is_hi, oos_lo, oos_hi) in enumerate(wf_folds_bars):
            # IS
            is_results, _, _ = simulate_strategy(strat, N, schedule, opens, highs, lows, is_lo, is_hi)
            is_portfolio = [(r[0], r[1], r[2]) for r in is_results]
            is_stats = calc_stats(is_portfolio)

            # OOS
            oos_results, _, _ = simulate_strategy(strat, N, schedule, opens, highs, lows, oos_lo, oos_hi)
            oos_portfolio = [(r[0], r[1], r[2]) for r in oos_results]
            oos_stats = calc_stats(oos_portfolio)

            passed = oos_stats['cmp_pnl'] > 0
            if not passed:
                all_pass = False

            is_start_d = round((is_lo - overlap_bar) / BARS_PER_DAY)
            is_end_d = round((is_hi - overlap_bar) / BARS_PER_DAY)
            oos_start_d = round((oos_lo - overlap_bar) / BARS_PER_DAY)
            oos_end_d = round((oos_hi - overlap_bar) / BARS_PER_DAY)

            fold_data = {
                'fold': fold_idx + 1,
                'is_days': f"{is_start_d}-{is_end_d}",
                'oos_days': f"{oos_start_d}-{oos_end_d}",
                'is_trades': is_stats['trades'],
                'is_wr': is_stats['wr'],
                'is_pnl': is_stats['cmp_pnl'],
                'oos_trades': oos_stats['trades'],
                'oos_wr': oos_stats['wr'],
                'oos_pnl': oos_stats['cmp_pnl'],
                'oos_mdd': oos_stats['cmp_mdd'],
                'pass': passed,
            }
            strat_wf.append(fold_data)

        pass_count = sum(1 for f in strat_wf if f['pass'])
        section2[strat] = {
            'folds': strat_wf,
            'verdict': f"{'PASS' if all_pass else 'FAIL'} ({pass_count}/3)",
        }

    output['section2_wf_validation'] = section2

    # Print WF results
    for strat in strategies:
        wf = section2[strat]
        print(f"\n  Strategy: {strat.upper()} -- Verdict: {wf['verdict']}")
        print(f"    {'Fold':>5s} | {'IS Days':>10s} | {'IS PnL':>8s} | {'OOS Days':>10s} | {'OOS Tr':>7s} | {'OOS WR':>7s} | {'OOS PnL':>8s} | {'OOS MDD':>8s} | {'Pass':>5s}")
        for f in wf['folds']:
            print(f"    {f['fold']:5d} | {f['is_days']:>10s} | {f['is_pnl']:>+8.1f} | {f['oos_days']:>10s} | {f['oos_trades']:7d} | {f['oos_wr']:6.1f}% | {f['oos_pnl']:>+8.1f} | {f['oos_mdd']:8.1f} | {'PASS' if f['pass'] else 'FAIL':>5s}")

    # ==================================================================
    # SECTION 3: CLOSE_OLDEST analysis (FIFO only, 270d)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: CLOSE_OLDEST ANALYSIS (FIFO, 270d)")
    print("=" * 70)

    fifo_results, fifo_co_count, fifo_co_pnls = simulate_strategy(
        'fifo', N, schedule, opens, highs, lows, overlap_bar, n_bars
    )

    # Direction changes via trades
    dir_changes = 0
    prev_dir = None
    for r in fifo_results:
        if prev_dir is not None and r[4] != prev_dir:
            dir_changes += 1
        prev_dir = r[4]

    co_wins = [p for p in fifo_co_pnls if p >= 0] if fifo_co_pnls else []
    co_losses = [p for p in fifo_co_pnls if p < 0] if fifo_co_pnls else []

    section3 = {
        'close_oldest_count': fifo_co_count,
        'co_avg_pnl': round(float(np.mean(fifo_co_pnls)), 2) if fifo_co_pnls else 0,
        'co_median_pnl': round(float(np.median(fifo_co_pnls)), 2) if fifo_co_pnls else 0,
        'co_win_count': len(co_wins),
        'co_loss_count': len(co_losses),
        'co_total_pnl': round(float(sum(fifo_co_pnls)), 2) if fifo_co_pnls else 0,
        'direction_changes': dir_changes,
    }
    output['section3_close_oldest'] = section3

    print(f"  Forced closures:       {fifo_co_count}")
    if fifo_co_pnls:
        print(f"  Avg PnL (slot):        {section3['co_avg_pnl']:+.2f}%")
        print(f"  Median PnL (slot):     {section3['co_median_pnl']:+.2f}%")
        print(f"  Win/Loss:              {len(co_wins)}W / {len(co_losses)}L")
        print(f"  Total PnL (slot):      {section3['co_total_pnl']:+.2f}%")
        print(f"  PnL range:             [{min(fifo_co_pnls):+.2f}, {max(fifo_co_pnls):+.2f}]")
    print(f"  Direction changes:     {dir_changes}")

    # ==================================================================
    # SECTION 4: FIFO N-Sweep (N=1..10, 270d)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: FIFO N-SWEEP (N=1..10, 270d)")
    print("=" * 70)

    section4 = {}
    print(f"\n  {'N':>3s} | {'Trades':>7s} | {'WR':>6s} | {'PF':>6s} | {'CmpPnL':>9s} | {'CmpMDD':>8s} | {'PnL/MDD':>8s} | {'CO#':>5s}")
    print("  " + "-" * 72)

    for n_pos in range(1, 11):
        results, co_count, _ = simulate_strategy(
            'fifo', n_pos, schedule, opens, highs, lows, overlap_bar, n_bars
        )
        portfolio = [(r[0], r[1], r[2]) for r in results]
        stats = calc_stats(portfolio)
        stats['close_oldest_count'] = co_count
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        stats['pnl_mdd_ratio'] = ratio
        section4[str(n_pos)] = stats

        print(f"  {n_pos:3d} | {stats['trades']:7d} | {stats['wr']:5.1f}% | {stats['pf']:6.2f} | "
              f"{stats['cmp_pnl']:>+9.1f} | {stats['cmp_mdd']:8.1f} | {ratio:8.2f} | {co_count:5d}")

    output['section4_fifo_n_sweep'] = section4

    # --- Also show Hedge N-sweep for comparison ---
    print(f"\n  HEDGE N-SWEEP (reference):")
    print(f"  {'N':>3s} | {'Trades':>7s} | {'WR':>6s} | {'PF':>6s} | {'CmpPnL':>9s} | {'CmpMDD':>8s} | {'PnL/MDD':>8s}")
    print("  " + "-" * 66)

    section4_hedge = {}
    for n_pos in range(1, 11):
        results, _, _ = simulate_strategy(
            'hedge', n_pos, schedule, opens, highs, lows, overlap_bar, n_bars
        )
        portfolio = [(r[0], r[1], r[2]) for r in results]
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        stats['pnl_mdd_ratio'] = ratio
        section4_hedge[str(n_pos)] = stats

        print(f"  {n_pos:3d} | {stats['trades']:7d} | {stats['wr']:5.1f}% | {stats['pf']:6.2f} | "
              f"{stats['cmp_pnl']:>+9.1f} | {stats['cmp_mdd']:8.1f} | {ratio:8.2f}")

    output['section4_hedge_n_sweep'] = section4_hedge

    # ==================================================================
    # SECTION 5: Final verdict
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: FINAL VERDICT")
    print("=" * 70)

    print(f"\n  270d overlap period:")
    print(f"  {'Strategy':12s} | {'PnL':>10s} | {'MDD':>8s} | {'PnL/MDD':>8s} | {'WR':>6s} | {'WF':>12s} | {'CO#':>5s}")
    print("  " + "-" * 72)
    for strat in strategies:
        s1 = section1[strat]
        wf_verdict = section2[strat]['verdict']
        print(f"  {strat:12s} | {s1['cmp_pnl']:>+10.1f} | {s1['cmp_mdd']:8.1f} | {s1['pnl_mdd_ratio']:8.2f} | {s1['wr']:5.1f}% | {wf_verdict:>12s} | {s1.get('close_oldest_count', 0):5d}")

    # Determine winner
    wf_pass = {s: '3/3' in section2[s]['verdict'] for s in strategies}
    passing = [s for s in strategies if wf_pass[s]]

    if passing:
        best = max(passing, key=lambda s: section1[s]['pnl_mdd_ratio'])
        verdict_text = f"BEST: {best.upper()} (WF PASS, highest PnL/MDD among passing)"
    else:
        # When no strategy passes WF, pick by highest cmp_pnl (least loss / most profit)
        best = max(strategies, key=lambda s: section1[s]['cmp_pnl'])
        verdict_text = f"BEST: {best.upper()} (no strategy passed WF 3/3, highest PnL)"

    print(f"\n  >>> {verdict_text}")

    # Key insights
    print("\n  Key insights:")
    hedge_pnl = section1['hedge']['cmp_pnl']
    fifo_pnl = section1['fifo']['cmp_pnl']
    ca_pnl = section1['close_all']['cmp_pnl']
    print(f"    Hedge vs FIFO:      Hedge {'+' if hedge_pnl > fifo_pnl else ''}{hedge_pnl - fifo_pnl:.1f}pp better")
    print(f"    Hedge vs Close-ALL: Hedge {'+' if hedge_pnl > ca_pnl else ''}{hedge_pnl - ca_pnl:.1f}pp better")
    print(f"    FIFO CO cost:       {section3['close_oldest_count']} forced closures, avg PnL {section3['co_avg_pnl']:+.2f}% (total {section3['co_total_pnl']:+.1f}%)")
    print(f"    Direction changes:  {section3['direction_changes']}")

    # Optimal N comparison
    best_fifo_n = max(section4, key=lambda k: section4[k]['cmp_pnl'])
    best_hedge_n = max(section4_hedge, key=lambda k: section4_hedge[k]['cmp_pnl'])
    print(f"\n    Optimal FIFO N:     N={best_fifo_n} (PnL {section4[best_fifo_n]['cmp_pnl']:+.1f}%)")
    print(f"    Optimal Hedge N:    N={best_hedge_n} (PnL {section4_hedge[best_hedge_n]['cmp_pnl']:+.1f}%)")

    output['section5_verdict'] = {
        'winner': best,
        'verdict': verdict_text,
        'wf_pass': {s: wf_pass.get(s, False) for s in strategies},
        'hedge_vs_fifo_pnl_diff': round(hedge_pnl - fifo_pnl, 1),
        'hedge_vs_close_all_pnl_diff': round(hedge_pnl - ca_pnl, 1),
        'optimal_fifo_n': int(best_fifo_n),
        'optimal_hedge_n': int(best_hedge_n),
    }

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Results saved to {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == '__main__':
    main()
