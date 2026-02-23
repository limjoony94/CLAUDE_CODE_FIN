#!/usr/bin/env python3
"""
N=9 Alternatives Deep Study — Direction-Neutral Improvements
=============================================================

Question:
  N=5→9 확장 외에 더 나은 방향-중립적 개선이 존재하는가?

Scenarios (ALL direction-neutral):
  1. N fine-sweep (5,7,8,9,10,11,13) + WF + direction breakdown
  2. Moderate timeout (None, 576=48h, 864=72h, 1440=5d) × N=9
  3. Hybrid timeout: short timeout (576) + larger N vs no timeout + N=9
  4. ATR-adaptive N: low-vol → N↑, high-vol → N↓
  5. Holding time analysis: per-bucket PnL (0-6h, 6-24h, 24h+)

Key constraint:
  NO direction-specific rules. LONG and SHORT are treated identically.
  Data has bearish bias — any direction-specific finding is regime artifact.

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based same-bar resolution)
  - Sizing: compound (1/N per slot)
  - Fee: 0.10% x 2 = 0.30% capital-space (FEE_PCT * LEVERAGE)
  - WF: 3-fold expanding window within 270d overlap region
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
logger = logging.getLogger('n9_alternatives')

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'n9_alternatives_study.json')


# ===================================================================
# SIMULATION ENGINE (extended from timeout_impact_study)
# ===================================================================

def simulate_hedge_extended(max_positions, schedule, opens, highs, lows,
                            lo_bar, hi_bar, timeout_bars=None,
                            atr_ratio=None, atr_adaptive_n=False,
                            base_n=9, low_vol_n=13, high_vol_n=7,
                            vol_threshold_lo=0.8, vol_threshold_hi=1.2):
    """
    Hedge-mode simulation with extended features.

    New: atr_adaptive_n — dynamically adjust max_positions based on ATR ratio.
      low-vol (ratio < vol_threshold_lo) → low_vol_n slots
      high-vol (ratio > vol_threshold_hi) → high_vol_n slots
      normal → base_n slots
    Note: adaptive N only affects NEW entries, existing positions stay open.
    """
    slots = {}
    slot_counter = 0
    results = []
    n_bars = len(opens)

    signal_count = 0
    entered_count = 0
    skipped_count = 0

    def _effective_n(bar):
        if not atr_adaptive_n or atr_ratio is None:
            return max_positions
        if bar >= len(atr_ratio) or np.isnan(atr_ratio[bar]):
            return base_n
        r = atr_ratio[bar]
        if r < vol_threshold_lo:
            return low_vol_n
        elif r > vol_threshold_hi:
            return high_vol_n
        return base_n

    def _close_slot(sid, exit_bar, exit_price, reason):
        pos = slots.pop(sid)
        pm = pos.direction * (exit_price / pos.entry_price - 1) * 100
        pnl = pm * LEVERAGE - FEE
        weight = 1.0 / _effective_n(pos.entry_bar)
        scaled = pnl * weight
        results.append({
            'entry_bar': pos.entry_bar,
            'exit_bar': exit_bar,
            'scaled_pnl': scaled,
            'raw_pnl': pnl,
            'reason': reason,
            'direction': pos.direction,
            'pattern': pos.pattern,
            'hold_bars': exit_bar - pos.entry_bar,
        })

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

        # 2. Check timeout
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

        eff_n = _effective_n(bar)
        if len(slots) < eff_n:
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


def results_to_portfolio(results):
    """Convert extended results to portfolio format for calc_stats."""
    return [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in results]


def run_wf_extended(sim_kwargs, schedule, opens, highs, lows, overlap_bar, n_bars):
    """3-fold expanding window WF."""
    avail_bars = n_bars - overlap_bar
    quarter = avail_bars // 4
    folds_def = [
        (overlap_bar, overlap_bar + quarter,
         overlap_bar + quarter, overlap_bar + 2 * quarter),
        (overlap_bar, overlap_bar + 2 * quarter,
         overlap_bar + 2 * quarter, overlap_bar + 3 * quarter),
        (overlap_bar, overlap_bar + 3 * quarter,
         overlap_bar + 3 * quarter, min(overlap_bar + avail_bars, n_bars)),
    ]
    wf_results = []
    for fold_idx, (is_lo, is_hi, oos_lo, oos_hi) in enumerate(folds_def):
        oos_res, oos_sig, oos_ent, oos_skip = simulate_hedge_extended(
            schedule=schedule, opens=opens, highs=highs, lows=lows,
            lo_bar=oos_lo, hi_bar=oos_hi, **sim_kwargs)
        oos_port = results_to_portfolio(oos_res)
        oos_stats = calc_stats(oos_port)
        passed = oos_stats['cmp_pnl'] > 0

        wf_results.append({
            'fold': fold_idx + 1,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['cmp_pnl'],
            'oos_mdd': oos_stats['cmp_mdd'],
            'pass': passed,
        })

    pass_count = sum(1 for f in wf_results if f['pass'])
    return wf_results, f"{'PASS' if pass_count == 3 else 'FAIL'} ({pass_count}/3)"


def direction_breakdown(results):
    """Per-direction stats from extended results."""
    long_pnls = [r['scaled_pnl'] for r in results if r['direction'] == 1]
    short_pnls = [r['scaled_pnl'] for r in results if r['direction'] == -1]

    def _dir_stats(pnls, label):
        if not pnls:
            return {'label': label, 'trades': 0, 'sum_pnl': 0, 'wr': 0, 'avg_pnl': 0}
        wins = sum(1 for p in pnls if p >= 0)
        return {
            'label': label,
            'trades': len(pnls),
            'sum_pnl': round(sum(pnls), 2),
            'wr': round(wins / len(pnls) * 100, 1),
            'avg_pnl': round(sum(pnls) / len(pnls), 3),
        }

    return {
        'long': _dir_stats(long_pnls, 'LONG'),
        'short': _dir_stats(short_pnls, 'SHORT'),
    }


def holding_time_breakdown(results):
    """PnL by holding time bucket."""
    buckets = {
        '0-6h': (0, 72),       # 0-72 bars
        '6-24h': (72, 288),    # 72-288 bars
        '24-48h': (288, 576),  # 288-576 bars
        '48h+': (576, 999999),
    }
    out = {}
    for label, (lo, hi) in buckets.items():
        trades = [r for r in results if lo <= r['hold_bars'] < hi]
        pnls = [r['scaled_pnl'] for r in trades]
        if pnls:
            wins = sum(1 for p in pnls if p >= 0)
            out[label] = {
                'trades': len(pnls),
                'wr': round(wins / len(pnls) * 100, 1),
                'sum_pnl': round(sum(pnls), 2),
                'avg_pnl': round(sum(pnls) / len(pnls), 3),
            }
        else:
            out[label] = {'trades': 0, 'wr': 0, 'sum_pnl': 0, 'avg_pnl': 0}
    return out


def exit_reason_breakdown(results):
    """Exit reason counts and PnL."""
    reasons = {}
    for r in results:
        reason = r['reason']
        if reason not in reasons:
            reasons[reason] = {'count': 0, 'sum_pnl': 0}
        reasons[reason]['count'] += 1
        reasons[reason]['sum_pnl'] += r['scaled_pnl']
    for v in reasons.values():
        v['sum_pnl'] = round(v['sum_pnl'], 2)
    return reasons


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
    avail_bars = n_bars - overlap_bar
    avail_days = avail_bars / BARS_PER_DAY
    logger.info(f"Overlap bar: {overlap_bar} | Available: {avail_bars} bars ({avail_days:.1f} days)")

    schedule = build_signal_schedule(patterns, sig_index, atr_ratio, opens, n_bars)
    total_signals = sum(len(v) for v in schedule.values())
    logger.info(f"Signal schedule: {len(schedule)} bars, {total_signals} signals")

    output = {}

    # ==================================================================
    # SECTION 1: N FINE-SWEEP (5..13) + WF + direction breakdown
    # ==================================================================
    print("\n" + "=" * 80)
    print("  SECTION 1: N FINE-SWEEP (NO_TIMEOUT, direction-neutral)")
    print("=" * 80)

    n_values = [5, 7, 8, 9, 10, 11, 13]
    section1 = {}

    print(f"\n  {'N':>3s} | {'Trades':>7s} | {'WR':>6s} | {'PF':>6s} | {'CmpPnL':>9s} | "
          f"{'CmpMDD':>8s} | {'PnL/MDD':>8s} | {'Hit%':>7s} | "
          f"{'L_Tr':>5s} | {'L_PnL':>8s} | {'S_Tr':>5s} | {'S_PnL':>8s} | {'WF':>10s}")
    print("  " + "-" * 120)

    for n_pos in n_values:
        sim_kwargs = dict(max_positions=n_pos, timeout_bars=None)
        results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge_extended(
            schedule=schedule, opens=opens, highs=highs, lows=lows,
            lo_bar=overlap_bar, hi_bar=n_bars, **sim_kwargs)

        portfolio = results_to_portfolio(results)
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        hit_rate = ent_cnt / sig_cnt * 100 if sig_cnt > 0 else 0

        dir_bk = direction_breakdown(results)
        hold_bk = holding_time_breakdown(results)
        exit_bk = exit_reason_breakdown(results)

        # WF
        wf_folds, wf_verdict = run_wf_extended(
            sim_kwargs, schedule, opens, highs, lows, overlap_bar, n_bars)

        section1[str(n_pos)] = {
            **stats,
            'pnl_mdd_ratio': ratio,
            'hit_rate': round(hit_rate, 1),
            'signals_total': sig_cnt,
            'entered': ent_cnt,
            'skipped': skip_cnt,
            'direction': dir_bk,
            'holding_time': hold_bk,
            'exit_reasons': exit_bk,
            'wf_verdict': wf_verdict,
            'wf_folds': wf_folds,
        }

        l = dir_bk['long']
        s = dir_bk['short']
        print(f"  {n_pos:3d} | {stats['trades']:7d} | {stats['wr']:5.1f}% | {stats['pf']:6.2f} | "
              f"{stats['cmp_pnl']:>+9.1f} | {stats['cmp_mdd']:8.1f} | {ratio:8.2f} | "
              f"{hit_rate:6.1f}% | "
              f"{l['trades']:5d} | {l['sum_pnl']:>+8.2f} | "
              f"{s['trades']:5d} | {s['sum_pnl']:>+8.2f} | {wf_verdict:>10s}")

    output['section1_n_sweep'] = section1

    # ==================================================================
    # SECTION 2: TIMEOUT SWEEP (direction-neutral) at N=9
    # ==================================================================
    print("\n" + "=" * 80)
    print("  SECTION 2: TIMEOUT SWEEP (N=9, direction-neutral)")
    print("=" * 80)

    timeout_values = [
        ('no_timeout', None),
        ('48h_576', 576),
        ('72h_864', 864),
        ('5d_1440', 1440),
        ('10d_2880', 2880),
        ('24h_288', 288),
    ]

    section2 = {}
    print(f"\n  {'Timeout':>12s} | {'Trades':>7s} | {'WR':>6s} | {'PF':>6s} | {'CmpPnL':>9s} | "
          f"{'CmpMDD':>8s} | {'PnL/MDD':>8s} | {'Hit%':>7s} | "
          f"{'TP':>5s} | {'SL':>5s} | {'TO':>5s} | {'END':>5s} | {'WF':>10s}")
    print("  " + "-" * 120)

    for label, tb in timeout_values:
        sim_kwargs = dict(max_positions=9, timeout_bars=tb)
        results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge_extended(
            schedule=schedule, opens=opens, highs=highs, lows=lows,
            lo_bar=overlap_bar, hi_bar=n_bars, **sim_kwargs)

        portfolio = results_to_portfolio(results)
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        hit_rate = ent_cnt / sig_cnt * 100 if sig_cnt > 0 else 0
        exit_bk = exit_reason_breakdown(results)

        wf_folds, wf_verdict = run_wf_extended(
            sim_kwargs, schedule, opens, highs, lows, overlap_bar, n_bars)

        section2[label] = {
            **stats,
            'pnl_mdd_ratio': ratio,
            'hit_rate': round(hit_rate, 1),
            'exit_reasons': exit_bk,
            'direction': direction_breakdown(results),
            'holding_time': holding_time_breakdown(results),
            'wf_verdict': wf_verdict,
            'wf_folds': wf_folds,
        }

        tp_c = exit_bk.get('TP', {}).get('count', 0)
        sl_c = exit_bk.get('SL', {}).get('count', 0)
        to_c = exit_bk.get('TIMEOUT', {}).get('count', 0)
        end_c = exit_bk.get('END', {}).get('count', 0)

        print(f"  {label:>12s} | {stats['trades']:7d} | {stats['wr']:5.1f}% | {stats['pf']:6.2f} | "
              f"{stats['cmp_pnl']:>+9.1f} | {stats['cmp_mdd']:8.1f} | {ratio:8.2f} | "
              f"{hit_rate:6.1f}% | "
              f"{tp_c:5d} | {sl_c:5d} | {to_c:5d} | {end_c:5d} | {wf_verdict:>10s}")

    output['section2_timeout_sweep'] = section2

    # ==================================================================
    # SECTION 3: TIMEOUT × N MATRIX (direction-neutral)
    # ==================================================================
    print("\n" + "=" * 80)
    print("  SECTION 3: TIMEOUT × N MATRIX (best combos)")
    print("=" * 80)

    combos = [
        (7, None), (7, 576), (7, 864),
        (9, None), (9, 576), (9, 864),
        (11, None), (11, 576), (11, 864),
        (13, None), (13, 576),
    ]

    section3 = {}
    print(f"\n  {'N':>3s} {'Timeout':>8s} | {'Trades':>7s} | {'WR':>6s} | {'CmpPnL':>9s} | "
          f"{'CmpMDD':>8s} | {'PnL/MDD':>8s} | {'Hit%':>7s} | {'WF':>10s}")
    print("  " + "-" * 90)

    for n_pos, tb in combos:
        label = f"N{n_pos}_{'NT' if tb is None else f'T{tb}'}"
        sim_kwargs = dict(max_positions=n_pos, timeout_bars=tb)
        results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge_extended(
            schedule=schedule, opens=opens, highs=highs, lows=lows,
            lo_bar=overlap_bar, hi_bar=n_bars, **sim_kwargs)

        portfolio = results_to_portfolio(results)
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        hit_rate = ent_cnt / sig_cnt * 100 if sig_cnt > 0 else 0

        wf_folds, wf_verdict = run_wf_extended(
            sim_kwargs, schedule, opens, highs, lows, overlap_bar, n_bars)

        section3[label] = {
            **stats,
            'pnl_mdd_ratio': ratio,
            'hit_rate': round(hit_rate, 1),
            'direction': direction_breakdown(results),
            'wf_verdict': wf_verdict,
            'wf_folds': wf_folds,
        }

        tb_str = 'None' if tb is None else str(tb)
        print(f"  {n_pos:3d} {tb_str:>8s} | {stats['trades']:7d} | {stats['wr']:5.1f}% | "
              f"{stats['cmp_pnl']:>+9.1f} | {stats['cmp_mdd']:8.1f} | {ratio:8.2f} | "
              f"{hit_rate:6.1f}% | {wf_verdict:>10s}")

    output['section3_timeout_n_matrix'] = section3

    # ==================================================================
    # SECTION 4: ATR-ADAPTIVE N
    # ==================================================================
    print("\n" + "=" * 80)
    print("  SECTION 4: ATR-ADAPTIVE N (direction-neutral)")
    print("=" * 80)

    adaptive_configs = [
        # (label, base_n, low_vol_n, high_vol_n, vol_lo, vol_hi)
        ('adapt_9_13_7', 9, 13, 7, 0.8, 1.2),
        ('adapt_9_11_7', 9, 11, 7, 0.8, 1.2),
        ('adapt_9_13_5', 9, 13, 5, 0.8, 1.2),
        ('adapt_9_11_5', 9, 11, 5, 0.7, 1.3),
        ('adapt_9_15_5', 9, 15, 5, 0.8, 1.2),
        ('fixed_9',      9, 9,  9, 0.8, 1.2),   # control
    ]

    section4 = {}
    print(f"\n  {'Config':>18s} | {'Trades':>7s} | {'WR':>6s} | {'CmpPnL':>9s} | "
          f"{'CmpMDD':>8s} | {'PnL/MDD':>8s} | {'Hit%':>7s} | {'WF':>10s}")
    print("  " + "-" * 95)

    for label, base_n, low_n, high_n, vlo, vhi in adaptive_configs:
        sim_kwargs = dict(
            max_positions=base_n, timeout_bars=None,
            atr_ratio=atr_ratio, atr_adaptive_n=True,
            base_n=base_n, low_vol_n=low_n, high_vol_n=high_n,
            vol_threshold_lo=vlo, vol_threshold_hi=vhi,
        )
        results, sig_cnt, ent_cnt, skip_cnt = simulate_hedge_extended(
            schedule=schedule, opens=opens, highs=highs, lows=lows,
            lo_bar=overlap_bar, hi_bar=n_bars, **sim_kwargs)

        portfolio = results_to_portfolio(results)
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        hit_rate = ent_cnt / sig_cnt * 100 if sig_cnt > 0 else 0

        wf_folds, wf_verdict = run_wf_extended(
            sim_kwargs, schedule, opens, highs, lows, overlap_bar, n_bars)

        section4[label] = {
            **stats,
            'pnl_mdd_ratio': ratio,
            'hit_rate': round(hit_rate, 1),
            'direction': direction_breakdown(results),
            'holding_time': holding_time_breakdown(results),
            'wf_verdict': wf_verdict,
            'wf_folds': wf_folds,
            'config': {'base_n': base_n, 'low_vol_n': low_n, 'high_vol_n': high_n,
                       'vol_lo': vlo, 'vol_hi': vhi},
        }

        print(f"  {label:>18s} | {stats['trades']:7d} | {stats['wr']:5.1f}% | "
              f"{stats['cmp_pnl']:>+9.1f} | {stats['cmp_mdd']:8.1f} | {ratio:8.2f} | "
              f"{hit_rate:6.1f}% | {wf_verdict:>10s}")

    output['section4_atr_adaptive_n'] = section4

    # ==================================================================
    # SECTION 5: HOLDING TIME DEEP ANALYSIS (N=9, no timeout)
    # ==================================================================
    print("\n" + "=" * 80)
    print("  SECTION 5: HOLDING TIME ANALYSIS (N=9, no timeout)")
    print("=" * 80)

    sim_kwargs = dict(max_positions=9, timeout_bars=None)
    results, _, _, _ = simulate_hedge_extended(
        schedule=schedule, opens=opens, highs=highs, lows=lows,
        lo_bar=overlap_bar, hi_bar=n_bars, **sim_kwargs)

    hold_bk = holding_time_breakdown(results)
    section5 = {'holding_time': hold_bk}

    # More granular buckets
    granular_buckets = {
        '0-2h': (0, 24),
        '2-6h': (24, 72),
        '6-12h': (72, 144),
        '12-24h': (144, 288),
        '24-48h': (288, 576),
        '48-96h': (576, 1152),
        '96h+': (1152, 999999),
    }
    granular = {}
    for label, (lo, hi) in granular_buckets.items():
        trades = [r for r in results if lo <= r['hold_bars'] < hi]
        pnls = [r['scaled_pnl'] for r in trades]
        if pnls:
            wins = sum(1 for p in pnls if p >= 0)
            avg_hold_h = np.mean([r['hold_bars'] for r in trades]) * 5 / 60
            granular[label] = {
                'trades': len(pnls),
                'wr': round(wins / len(pnls) * 100, 1),
                'sum_pnl': round(sum(pnls), 2),
                'avg_pnl': round(sum(pnls) / len(pnls), 3),
                'avg_hold_hours': round(avg_hold_h, 1),
            }
        else:
            granular[label] = {'trades': 0, 'wr': 0, 'sum_pnl': 0, 'avg_pnl': 0, 'avg_hold_hours': 0}
    section5['granular'] = granular

    # Holding percentiles
    if results:
        hold_bars_all = [r['hold_bars'] for r in results]
        section5['percentiles'] = {
            'p25': round(np.percentile(hold_bars_all, 25) * 5 / 60, 1),
            'p50': round(np.percentile(hold_bars_all, 50) * 5 / 60, 1),
            'p75': round(np.percentile(hold_bars_all, 75) * 5 / 60, 1),
            'p90': round(np.percentile(hold_bars_all, 90) * 5 / 60, 1),
            'p95': round(np.percentile(hold_bars_all, 95) * 5 / 60, 1),
            'p99': round(np.percentile(hold_bars_all, 99) * 5 / 60, 1),
            'mean': round(np.mean(hold_bars_all) * 5 / 60, 1),
        }

    # Long-tail analysis: trades still open after 48h
    long_tail = [r for r in results if r['hold_bars'] >= 576]
    if long_tail:
        lt_pnls = [r['scaled_pnl'] for r in long_tail]
        lt_wins = sum(1 for p in lt_pnls if p >= 0)
        section5['long_tail_48h_plus'] = {
            'trades': len(long_tail),
            'pct_of_total': round(len(long_tail) / len(results) * 100, 1),
            'wr': round(lt_wins / len(long_tail) * 100, 1),
            'sum_pnl': round(sum(lt_pnls), 2),
            'avg_pnl': round(sum(lt_pnls) / len(long_tail), 3),
            'slot_days_consumed': round(sum(r['hold_bars'] for r in long_tail) / BARS_PER_DAY, 1),
        }
    else:
        section5['long_tail_48h_plus'] = {'trades': 0}

    output['section5_holding_time'] = section5

    print(f"\n  {'Bucket':>10s} | {'Trades':>7s} | {'WR':>6s} | {'SumPnL':>9s} | {'AvgPnL':>8s} | {'AvgHold':>8s}")
    print("  " + "-" * 60)
    for label, data in granular.items():
        print(f"  {label:>10s} | {data['trades']:7d} | {data['wr']:5.1f}% | "
              f"{data['sum_pnl']:>+9.2f} | {data['avg_pnl']:>+8.3f} | "
              f"{data.get('avg_hold_hours', 0):7.1f}h")

    if 'percentiles' in section5:
        p = section5['percentiles']
        print(f"\n  Hold time percentiles (hours):")
        print(f"    P25={p['p25']}h  P50={p['p50']}h  P75={p['p75']}h  P90={p['p90']}h  P95={p['p95']}h  P99={p['p99']}h  Mean={p['mean']}h")

    if section5.get('long_tail_48h_plus', {}).get('trades', 0) > 0:
        lt = section5['long_tail_48h_plus']
        print(f"\n  Long-tail (48h+): {lt['trades']} trades ({lt['pct_of_total']:.1f}%), "
              f"WR {lt['wr']:.1f}%, SumPnL {lt['sum_pnl']:+.2f}, "
              f"consuming {lt['slot_days_consumed']:.1f} slot-days")

    # ==================================================================
    # SECTION 6: VERDICT
    # ==================================================================
    print("\n" + "=" * 80)
    print("  SECTION 6: VERDICT — BEST DIRECTION-NEUTRAL CONFIGURATION")
    print("=" * 80)

    # Collect all WF-passing configurations
    candidates = []

    # From section 1 (N sweep)
    for n_str, data in section1.items():
        if '3/3' in data['wf_verdict']:
            candidates.append({
                'label': f'N={n_str}_NT',
                'pnl': data['cmp_pnl'],
                'mdd': data['cmp_mdd'],
                'ratio': data['pnl_mdd_ratio'],
                'wr': data['wr'],
                'trades': data['trades'],
                'hit_rate': data['hit_rate'],
                'source': 'section1',
            })

    # From section 2 (timeout sweep)
    for label, data in section2.items():
        if '3/3' in data['wf_verdict']:
            candidates.append({
                'label': f'N9_{label}',
                'pnl': data['cmp_pnl'],
                'mdd': data['cmp_mdd'],
                'ratio': data['pnl_mdd_ratio'],
                'wr': data['wr'],
                'trades': data['trades'],
                'hit_rate': data['hit_rate'],
                'source': 'section2',
            })

    # From section 3 (matrix)
    for label, data in section3.items():
        if '3/3' in data['wf_verdict']:
            candidates.append({
                'label': label,
                'pnl': data['cmp_pnl'],
                'mdd': data['cmp_mdd'],
                'ratio': data['pnl_mdd_ratio'],
                'wr': data['wr'],
                'trades': data['trades'],
                'hit_rate': data['hit_rate'],
                'source': 'section3',
            })

    # From section 4 (adaptive)
    for label, data in section4.items():
        if '3/3' in data['wf_verdict']:
            candidates.append({
                'label': f'adaptive_{label}',
                'pnl': data['cmp_pnl'],
                'mdd': data['cmp_mdd'],
                'ratio': data['pnl_mdd_ratio'],
                'wr': data['wr'],
                'trades': data['trades'],
                'hit_rate': data['hit_rate'],
                'source': 'section4',
            })

    if candidates:
        # Sort by PnL/MDD ratio
        candidates.sort(key=lambda c: -c['ratio'])

        print(f"\n  ALL WF 3/3 PASS candidates (sorted by PnL/MDD):\n")
        print(f"  {'Rank':>4s} | {'Config':>25s} | {'Trades':>7s} | {'WR':>6s} | {'CmpPnL':>9s} | "
              f"{'CmpMDD':>8s} | {'PnL/MDD':>8s} | {'Hit%':>7s}")
        print("  " + "-" * 95)
        for i, c in enumerate(candidates):
            marker = " ★" if i == 0 else ""
            print(f"  {i+1:4d} | {c['label']:>25s} | {c['trades']:7d} | {c['wr']:5.1f}% | "
                  f"{c['pnl']:>+9.1f} | {c['mdd']:8.1f} | {c['ratio']:8.2f} | "
                  f"{c['hit_rate']:6.1f}%{marker}")

        best = candidates[0]
        print(f"\n  ★ BEST: {best['label']}")
        print(f"    PnL: {best['pnl']:+.1f}% | MDD: {best['mdd']:.1f}% | "
              f"PnL/MDD: {best['ratio']:.2f} | WR: {best['wr']:.1f}% | "
              f"Trades: {best['trades']} | Hit: {best['hit_rate']:.1f}%")

        # Compare to baseline N=9 no timeout
        baseline = section1.get('9', {})
        if baseline:
            print(f"\n  vs Baseline (N=9 NO_TIMEOUT):")
            print(f"    PnL: {baseline['cmp_pnl']:+.1f}% | MDD: {baseline['cmp_mdd']:.1f}% | "
                  f"PnL/MDD: {baseline['pnl_mdd_ratio']:.2f}")
            if best['ratio'] > baseline['pnl_mdd_ratio']:
                pct_improve = (best['ratio'] / baseline['pnl_mdd_ratio'] - 1) * 100
                print(f"    Improvement: +{pct_improve:.1f}% PnL/MDD")
            else:
                print(f"    N=9 NO_TIMEOUT remains best")
    else:
        print("\n  WARNING: No configuration passed WF 3/3!")

    output['section6_verdict'] = {
        'candidates_wf_pass': candidates,
        'best': candidates[0] if candidates else None,
    }

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Results saved to {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == '__main__':
    main()
