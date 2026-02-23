#!/usr/bin/env python3
"""
TIMEOUT Impact Study: 288-bar timeout vs NO timeout (hold until TP/SL)
======================================================================

Question:
  fifo_vs_hedge study validated Hedge N=5 with 288-bar (24h) TIMEOUT -> +174%, WF 3/3.
  Production bot has NO timeout. Which is better?
  If NO timeout wins, what N gives ~50% signal hit rate?

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based same-bar resolution)
  - Sizing: compound (1/N per slot)
  - Fee: 0.10% x 2 = 0.30% capital-space (FEE_PCT * LEVERAGE)
  - WF: 3-fold expanding window within 270d overlap region
  - ATR scaling: BOTH_a14_w576_0.6-1.7

Modes:
  A) TIMEOUT=288:  positions close at market after 288 bars (24h)
  B) NO_TIMEOUT:   positions hold until TP or SL (or data END)
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
logger = logging.getLogger('timeout_impact')

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'timeout_impact_study.json')


def simulate_hedge(max_positions, schedule, opens, highs, lows,
                   lo_bar, hi_bar, timeout_bars=288):
    """
    Hedge-mode simulation with configurable timeout.

    timeout_bars: int or None
        288 = close after 288 bars (24h)
        None/0 = hold until TP/SL (no timeout)

    Returns: (results_list, signal_count, entered_count, skipped_count)
      results_list items: (entry_bar, exit_bar, scaled_pnl, exit_reason, direction, pattern)
    """
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
        results.append((pos.entry_bar, exit_bar, scaled, reason, pos.direction, pos.pattern))

    for bar in range(lo_bar, hi_bar):
        if bar >= n_bars:
            break

        # 1. Check TP/SL
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

        # 2. Check timeout (only if enabled)
        if timeout_bars and timeout_bars > 0:
            for sid in list(slots):
                if sid not in slots:
                    continue
                pos = slots[sid]
                if bar - pos.entry_bar >= timeout_bars:
                    exit_price = opens[bar] if bar < n_bars else opens[-1]
                    _close_slot(sid, bar, exit_price, 'TIMEOUT')

        # 3. Process signals
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

    # Close remaining at end
    for sid in list(slots):
        exit_price = opens[min(hi_bar - 1, n_bars - 1)]
        _close_slot(sid, hi_bar - 1, exit_price, 'END')

    return results, signal_count, entered_count, skipped_count


def run_wf(max_positions, schedule, opens, highs, lows, overlap_bar, n_bars,
           timeout_bars=288):
    """3-fold expanding window WF within 270d overlap region."""
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
        is_res, _, _, _ = simulate_hedge(max_positions, schedule, opens, highs, lows,
                                         is_lo, is_hi, timeout_bars)
        is_port = [(r[0], r[1], r[2]) for r in is_res]
        is_stats = calc_stats(is_port)

        oos_res, oos_sig, oos_ent, oos_skip = simulate_hedge(
            max_positions, schedule, opens, highs, lows,
            oos_lo, oos_hi, timeout_bars)
        oos_port = [(r[0], r[1], r[2]) for r in oos_res]
        oos_stats = calc_stats(oos_port)

        passed = oos_stats['cmp_pnl'] > 0
        is_start_d = round((is_lo - overlap_bar) / BARS_PER_DAY)
        is_end_d = round((is_hi - overlap_bar) / BARS_PER_DAY)
        oos_start_d = round((oos_lo - overlap_bar) / BARS_PER_DAY)
        oos_end_d = round((oos_hi - overlap_bar) / BARS_PER_DAY)

        wf_results.append({
            'fold': fold_idx + 1,
            'is_days': f"{is_start_d}-{is_end_d}",
            'oos_days': f"{oos_start_d}-{oos_end_d}",
            'is_trades': is_stats['trades'],
            'is_pnl': is_stats['cmp_pnl'],
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['cmp_pnl'],
            'oos_mdd': oos_stats['cmp_mdd'],
            'pass': passed,
        })

    pass_count = sum(1 for f in wf_results if f['pass'])
    return wf_results, f"{'PASS' if pass_count == 3 else 'FAIL'} ({pass_count}/3)"


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
    avail_bars = n_bars - overlap_bar
    avail_days = avail_bars / BARS_PER_DAY
    logger.info(f"Overlap bar: {overlap_bar} | Available: {avail_bars} bars ({avail_days:.1f} days)")

    schedule = build_signal_schedule(patterns, sig_index, atr_ratio, opens, n_bars)
    total_signals = sum(len(v) for v in schedule.values())
    logger.info(f"Signal schedule: {len(schedule)} bars, {total_signals} signals")

    output = {}

    # ==================================================================
    # SECTION 1: TIMEOUT=288 vs NO_TIMEOUT (Hedge, N=5, 270d overlap)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: TIMEOUT=288 vs NO_TIMEOUT (Hedge N=5, 270d)")
    print("=" * 70)

    modes = [
        ('timeout_288', 288),
        ('no_timeout', None),
    ]

    section1 = {}
    for mode_name, tb in modes:
        results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge(
            5, schedule, opens, highs, lows, overlap_bar, n_bars, tb
        )
        portfolio = [(r[0], r[1], r[2]) for r in results]
        stats = calc_stats(portfolio)

        exit_counts = {}
        for r in results:
            exit_counts[r[3]] = exit_counts.get(r[3], 0) + 1

        # Holding time stats (in bars)
        hold_bars = [r[1] - r[0] for r in results]
        avg_hold = float(np.mean(hold_bars)) if hold_bars else 0
        median_hold = float(np.median(hold_bars)) if hold_bars else 0
        max_hold = int(max(hold_bars)) if hold_bars else 0

        hit_rate = ent_cnt / sig_cnt * 100 if sig_cnt > 0 else 0

        section1[mode_name] = {
            **stats,
            'exit_reasons': exit_counts,
            'pnl_mdd_ratio': pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd']),
            'signals_total': sig_cnt,
            'signals_entered': ent_cnt,
            'signals_skipped': skip_cnt,
            'hit_rate': round(hit_rate, 1),
            'avg_hold_bars': round(avg_hold, 1),
            'median_hold_bars': round(median_hold, 1),
            'max_hold_bars': max_hold,
            'avg_hold_hours': round(avg_hold * 5 / 60, 1),
            'median_hold_hours': round(median_hold * 5 / 60, 1),
        }

    output['section1_comparison'] = section1

    print(f"\n  {'Metric':22s} | {'TIMEOUT=288':>14s} | {'NO_TIMEOUT':>14s}")
    print("  " + "-" * 56)
    for key in ['trades', 'wr', 'pf', 'cmp_pnl', 'cmp_mdd', 'pnl_mdd_ratio',
                'add_pnl', 'add_edge',
                'signals_total', 'signals_entered', 'signals_skipped', 'hit_rate',
                'avg_hold_hours', 'median_hold_hours', 'max_hold_bars']:
        v1 = section1['timeout_288'].get(key, '-')
        v2 = section1['no_timeout'].get(key, '-')
        fmt = f"{v1:>14}" if isinstance(v1, (int, str)) else f"{v1:>14.1f}"
        fmt2 = f"{v2:>14}" if isinstance(v2, (int, str)) else f"{v2:>14.1f}"
        print(f"  {key:22s} | {fmt} | {fmt2}")

    print("\n  Exit reasons:")
    all_reasons = set()
    for m in section1.values():
        all_reasons.update(m.get('exit_reasons', {}).keys())
    for reason in sorted(all_reasons):
        v1 = section1['timeout_288'].get('exit_reasons', {}).get(reason, 0)
        v2 = section1['no_timeout'].get('exit_reasons', {}).get(reason, 0)
        print(f"    {reason:16s} | {v1:>14d} | {v2:>14d}")

    # ==================================================================
    # SECTION 2: WF Validation for both modes
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: WF 3-FOLD VALIDATION")
    print("=" * 70)

    section2 = {}
    for mode_name, tb in modes:
        folds, verdict = run_wf(5, schedule, opens, highs, lows,
                                overlap_bar, n_bars, tb)
        section2[mode_name] = {'folds': folds, 'verdict': verdict}

        print(f"\n  {mode_name.upper()} -- {verdict}")
        print(f"    {'Fold':>5s} | {'OOS Days':>10s} | {'OOS Tr':>7s} | {'OOS WR':>7s} | {'OOS PnL':>8s} | {'OOS MDD':>8s} | {'Pass':>5s}")
        for f in folds:
            print(f"    {f['fold']:5d} | {f['oos_days']:>10s} | {f['oos_trades']:7d} | {f['oos_wr']:6.1f}% | {f['oos_pnl']:>+8.1f} | {f['oos_mdd']:8.1f} | {'PASS' if f['pass'] else 'FAIL':>5s}")

    output['section2_wf'] = section2

    # ==================================================================
    # SECTION 3: NO_TIMEOUT N-sweep (N=1..15) for hit rate optimization
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: NO_TIMEOUT N-SWEEP (N=1..15, 270d, Hedge)")
    print("=" * 70)

    section3 = {}
    print(f"\n  {'N':>3s} | {'Trades':>7s} | {'WR':>6s} | {'PF':>6s} | {'CmpPnL':>9s} | {'CmpMDD':>8s} | {'PnL/MDD':>8s} | {'HitRate':>8s} | {'AvgHold':>8s}")
    print("  " + "-" * 82)

    for n_pos in range(1, 16):
        results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge(
            n_pos, schedule, opens, highs, lows, overlap_bar, n_bars, None
        )
        portfolio = [(r[0], r[1], r[2]) for r in results]
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        hit_rate = ent_cnt / sig_cnt * 100 if sig_cnt > 0 else 0
        hold_bars = [r[1] - r[0] for r in results]
        avg_hold_h = float(np.mean(hold_bars)) * 5 / 60 if hold_bars else 0

        section3[str(n_pos)] = {
            **stats,
            'pnl_mdd_ratio': ratio,
            'signals_total': sig_cnt,
            'signals_entered': ent_cnt,
            'signals_skipped': skip_cnt,
            'hit_rate': round(hit_rate, 1),
            'avg_hold_hours': round(avg_hold_h, 1),
        }

        print(f"  {n_pos:3d} | {stats['trades']:7d} | {stats['wr']:5.1f}% | {stats['pf']:6.2f} | "
              f"{stats['cmp_pnl']:>+9.1f} | {stats['cmp_mdd']:8.1f} | {ratio:8.2f} | "
              f"{hit_rate:7.1f}% | {avg_hold_h:7.1f}h")

    output['section3_no_timeout_n_sweep'] = section3

    # Also: TIMEOUT=288 N-sweep for fair comparison
    print(f"\n  TIMEOUT=288 N-SWEEP (reference):")
    print(f"  {'N':>3s} | {'Trades':>7s} | {'WR':>6s} | {'PF':>6s} | {'CmpPnL':>9s} | {'CmpMDD':>8s} | {'PnL/MDD':>8s} | {'HitRate':>8s} | {'AvgHold':>8s}")
    print("  " + "-" * 82)

    section3_to = {}
    for n_pos in range(1, 16):
        results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge(
            n_pos, schedule, opens, highs, lows, overlap_bar, n_bars, 288
        )
        portfolio = [(r[0], r[1], r[2]) for r in results]
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        hit_rate = ent_cnt / sig_cnt * 100 if sig_cnt > 0 else 0
        hold_bars = [r[1] - r[0] for r in results]
        avg_hold_h = float(np.mean(hold_bars)) * 5 / 60 if hold_bars else 0

        section3_to[str(n_pos)] = {
            **stats,
            'pnl_mdd_ratio': ratio,
            'signals_total': sig_cnt,
            'signals_entered': ent_cnt,
            'signals_skipped': skip_cnt,
            'hit_rate': round(hit_rate, 1),
            'avg_hold_hours': round(avg_hold_h, 1),
        }

        print(f"  {n_pos:3d} | {stats['trades']:7d} | {stats['wr']:5.1f}% | {stats['pf']:6.2f} | "
              f"{stats['cmp_pnl']:>+9.1f} | {stats['cmp_mdd']:8.1f} | {ratio:8.2f} | "
              f"{hit_rate:7.1f}% | {avg_hold_h:7.1f}h")

    output['section3_timeout_n_sweep'] = section3_to

    # ==================================================================
    # SECTION 4: WF for promising NO_TIMEOUT N values
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: WF VALIDATION FOR PROMISING N VALUES")
    print("=" * 70)

    # Pick N values where no_timeout cmp_pnl > 0 and hit_rate approaching 50%
    promising_n = []
    for n_str, data in sorted(section3.items(), key=lambda x: int(x[0])):
        n = int(n_str)
        if data['cmp_pnl'] > 0 or n <= 5:
            promising_n.append(n)
    # Also always include N where hit_rate crosses 50%
    for n_str, data in section3.items():
        n = int(n_str)
        if 45 <= data['hit_rate'] <= 55 and n not in promising_n:
            promising_n.append(n)
    promising_n = sorted(set(promising_n))

    section4 = {}
    for n_pos in promising_n:
        # NO_TIMEOUT WF
        folds_nt, verdict_nt = run_wf(n_pos, schedule, opens, highs, lows,
                                      overlap_bar, n_bars, None)
        # TIMEOUT=288 WF
        folds_to, verdict_to = run_wf(n_pos, schedule, opens, highs, lows,
                                      overlap_bar, n_bars, 288)

        nt_data = section3[str(n_pos)]
        to_data = section3_to[str(n_pos)]

        section4[str(n_pos)] = {
            'no_timeout': {
                'cmp_pnl': nt_data['cmp_pnl'],
                'cmp_mdd': nt_data['cmp_mdd'],
                'pnl_mdd_ratio': nt_data['pnl_mdd_ratio'],
                'hit_rate': nt_data['hit_rate'],
                'wf_verdict': verdict_nt,
                'wf_folds': folds_nt,
            },
            'timeout_288': {
                'cmp_pnl': to_data['cmp_pnl'],
                'cmp_mdd': to_data['cmp_mdd'],
                'pnl_mdd_ratio': to_data['pnl_mdd_ratio'],
                'hit_rate': to_data['hit_rate'],
                'wf_verdict': verdict_to,
                'wf_folds': folds_to,
            },
        }

        print(f"\n  N={n_pos}:")
        print(f"    NO_TIMEOUT:  PnL {nt_data['cmp_pnl']:>+8.1f}% | MDD {nt_data['cmp_mdd']:6.1f}% | "
              f"PnL/MDD {nt_data['pnl_mdd_ratio']:6.2f} | Hit {nt_data['hit_rate']:5.1f}% | WF {verdict_nt}")
        print(f"    TIMEOUT=288: PnL {to_data['cmp_pnl']:>+8.1f}% | MDD {to_data['cmp_mdd']:6.1f}% | "
              f"PnL/MDD {to_data['pnl_mdd_ratio']:6.2f} | Hit {to_data['hit_rate']:5.1f}% | WF {verdict_to}")

    output['section4_wf_comparison'] = section4

    # ==================================================================
    # SECTION 5: VERDICT
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: VERDICT")
    print("=" * 70)

    # Find best config: WF PASS + highest PnL/MDD
    best_config = None
    best_ratio = -999
    for n_str, data in section4.items():
        for mode in ['no_timeout', 'timeout_288']:
            if '3/3' in data[mode]['wf_verdict'] and data[mode]['pnl_mdd_ratio'] > best_ratio:
                best_ratio = data[mode]['pnl_mdd_ratio']
                best_config = (int(n_str), mode)

    if best_config:
        n_best, mode_best = best_config
        d = section4[str(n_best)][mode_best]
        print(f"\n  BEST: Hedge N={n_best}, {mode_best}")
        print(f"    PnL: {d['cmp_pnl']:+.1f}% | MDD: {d['cmp_mdd']:.1f}% | PnL/MDD: {d['pnl_mdd_ratio']:.2f}")
        print(f"    Hit Rate: {d['hit_rate']:.1f}% | WF: {d['wf_verdict']}")
    else:
        print("\n  WARNING: No configuration passed WF 3/3")
        # Fall back to highest PnL/MDD regardless of WF
        for n_str, data in section4.items():
            for mode in ['no_timeout', 'timeout_288']:
                if data[mode]['pnl_mdd_ratio'] > best_ratio:
                    best_ratio = data[mode]['pnl_mdd_ratio']
                    best_config = (int(n_str), mode)
        if best_config:
            n_best, mode_best = best_config
            d = section4[str(n_best)][mode_best]
            print(f"  Best (no WF filter): Hedge N={n_best}, {mode_best}")
            print(f"    PnL: {d['cmp_pnl']:+.1f}% | MDD: {d['cmp_mdd']:.1f}% | PnL/MDD: {d['pnl_mdd_ratio']:.2f}")

    output['section5_verdict'] = {
        'best_n': best_config[0] if best_config else None,
        'best_mode': best_config[1] if best_config else None,
        'best_pnl_mdd_ratio': best_ratio,
    }

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Results saved to {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == '__main__':
    main()
