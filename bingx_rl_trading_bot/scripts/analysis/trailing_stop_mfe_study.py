#!/usr/bin/env python3
"""
Trailing Stop MFE Study
=======================

Problem 3 - Passive Management:
  After entry, only TP/SL/timeout waiting.
  MFE->SL reversal frequency unmeasured.
  No bar-by-bar MFE path tracking.

Five Phases:
  1. Bar-by-bar MFE path tracking (all patterns)
  2. MFE->SL reversal frequency analysis
  3. Trailing stop simulation (9 combos)
  4. Portfolio impact + WF 3-fold validation
  5. Final judgment

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based, same-bar from bar OPEN)
  - Sizing: compound (1/N per slot)
  - Fee: FEE_PCT * LEVERAGE = 0.30% capital-space
  - Timeout: 864 bars (72h) -> DROP
  - WF: 3-fold expanding window within 270d overlap of 720d data
  - classify_candle: production import (NEVER self-implement)
"""

import os
import sys
import json
import time
import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
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
from scripts.analysis.tpsl_regime_bias_study import (
    backtest_pattern_trades, compute_wr_and_edge,
    build_custom_schedule, simulate_hedge_simple,
    run_wf_3fold, load_patterns_from_file,
)
from scripts.analysis.fifo_vs_hedge_backtest_study import (
    ActivePosition, pnl_mdd_ratio,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('trailing_stop_mfe')

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_720D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
PATTERNS_59_BACKUP = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns_59pat_backup.json')
STUDY1_OUTPUT = os.path.join(_PROJECT_ROOT, 'results', 'long_restore_and_direction_cap_study.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'trailing_stop_mfe_study.json')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMEOUT_T864 = 864
MAX_POSITIONS = 9

# Phase 2: MFE->SL reversal thresholds (% of TP reached before SL)
MFE_THRESHOLDS = [30, 50, 70, 90]

# Phase 3: Trailing stop grid
ACTIVATION_PCTS = [30, 50, 70]    # X: MFE >= TP * X% triggers trailing
TRAIL_PCTS = [0, 30, 50]          # Y: 0=breakeven(SL->entry), >0=dynamic trail


# ===================================================================
# PHASE 1: Bar-by-Bar MFE Path Tracking
# ===================================================================

def compute_mfe_timeline(sig_bars, direction_int, tp_pct, sl_pct,
                         opens, highs, lows, n_bars,
                         atr_ratio=None, use_atr=True,
                         lo_bar=0, timeout_bars=864):
    """
    Track bar-by-bar MFE/MAE path for each trade.

    Returns list of dicts per trade with:
      entry_bar, exit_bar, exit_reason, peak_mfe_pct, final_mae_pct,
      peak_offset, tp_pct_eff, sl_pct_eff, timeline (list of bar snapshots)
    """
    trades = []

    for sig_bar in sig_bars:
        if sig_bar < lo_bar:
            continue
        entry_bar = sig_bar + 1
        if entry_bar >= n_bars:
            continue

        entry_price = opens[entry_bar]
        if entry_price <= 0:
            continue

        # ATR scaling
        scale = 1.0
        if use_atr and atr_ratio is not None and entry_bar < len(atr_ratio):
            scale = atr_ratio[entry_bar]
            scale = max(CLAMP_LO, min(CLAMP_HI, scale))

        eff_tp = tp_pct * scale
        eff_sl = sl_pct * scale

        if direction_int == 1:
            tp_price = entry_price * (1 + eff_tp / 100)
            sl_price = entry_price * (1 - eff_sl / 100)
        else:
            tp_price = entry_price * (1 - eff_tp / 100)
            sl_price = entry_price * (1 + eff_sl / 100)

        timeline = []
        running_mfe = 0.0
        running_mae = 0.0
        exit_bar = None
        exit_reason = None
        peak_offset = 0

        for offset in range(timeout_bars + 1):
            bar = entry_bar + offset
            if bar >= n_bars:
                exit_reason = 'TIMEOUT'
                exit_bar = bar - 1
                break

            h, l, o = highs[bar], lows[bar], opens[bar]

            # Compute excursions at this bar (price-space %)
            if direction_int == 1:
                bar_mfe = (h - entry_price) / entry_price * 100
                bar_mae = (entry_price - l) / entry_price * 100
            else:
                bar_mfe = (entry_price - l) / entry_price * 100
                bar_mae = (h - entry_price) / entry_price * 100

            running_mfe = max(running_mfe, bar_mfe)
            running_mae = max(running_mae, bar_mae)
            dd_from_peak = running_mfe - bar_mfe

            if running_mfe == bar_mfe:
                peak_offset = offset

            timeline.append({
                'offset': offset,
                'mfe_pct': round(running_mfe, 4),
                'mae_pct': round(running_mae, 4),
                'dd_from_peak': round(dd_from_peak, 4),
            })

            # Check TP/SL
            if direction_int == 1:
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
                exit_bar = bar
                exit_reason = 'TP'
                break
            elif sl_hit:
                exit_bar = bar
                exit_reason = 'SL'
                break

        if exit_reason is None:
            exit_reason = 'TIMEOUT'
            exit_bar = min(entry_bar + timeout_bars, n_bars - 1)

        trades.append({
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'exit_reason': exit_reason,
            'timeline': timeline,
            'peak_mfe_pct': round(running_mfe, 4),
            'final_mae_pct': round(running_mae, 4),
            'peak_offset': peak_offset,
            'tp_pct_eff': round(eff_tp, 4),
            'sl_pct_eff': round(eff_sl, 4),
        })

    return trades


def phase1_mfe_paths(all_patterns, sig_index, opens, highs, lows,
                     n_bars, atr_ratio, overlap_bar):
    """Compute MFE timelines for all patterns. Return per-pattern stats."""
    print("\n" + "=" * 110)
    print("  PHASE 1: BAR-BY-BAR MFE PATH TRACKING")
    print(f"  Patterns: {len(all_patterns)}, Timeout: {TIMEOUT_T864}")
    print("=" * 110)

    t0 = time.time()
    pattern_results = {}
    all_trades_flat = []

    for p in all_patterns:
        pattern = p['pattern']
        direction_int = 1 if p['direction'] == 'LONG' else -1
        tp, sl = p['tp'], p['sl']

        bars = sig_index.get(pattern)
        if not bars:
            continue

        trades = compute_mfe_timeline(
            bars, direction_int, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)

        if not trades:
            continue

        # Aggregate stats per pattern
        tp_count = sum(1 for t in trades if t['exit_reason'] == 'TP')
        sl_count = sum(1 for t in trades if t['exit_reason'] == 'SL')
        to_count = sum(1 for t in trades if t['exit_reason'] == 'TIMEOUT')
        peak_mfes = [t['peak_mfe_pct'] for t in trades]
        peak_offsets = [t['peak_offset'] for t in trades]

        pattern_results[pattern] = {
            'direction': p['direction'],
            'tp': tp, 'sl': sl,
            'total_trades': len(trades),
            'tp_count': tp_count, 'sl_count': sl_count, 'to_count': to_count,
            'peak_mfe_median': round(float(np.median(peak_mfes)), 3),
            'peak_mfe_mean': round(float(np.mean(peak_mfes)), 3),
            'peak_mfe_p25': round(float(np.percentile(peak_mfes, 25)), 3),
            'peak_mfe_p75': round(float(np.percentile(peak_mfes, 75)), 3),
            'peak_offset_median': int(np.median(peak_offsets)),
            'peak_offset_mean': round(float(np.mean(peak_offsets)), 1),
        }

        for t in trades:
            t['pattern'] = pattern
            t['direction'] = p['direction']
        all_trades_flat.extend(trades)

    elapsed = time.time() - t0

    # Print summary
    print(f"\n  {'Pattern':<14} {'Dir':>5} {'TP':>5} {'SL':>5} | "
          f"{'N':>4} {'TP#':>4} {'SL#':>4} {'TO#':>4} | "
          f"{'MFE_med':>8} {'MFE_p25':>8} {'MFE_p75':>8} {'PeakBar':>8}")
    print("  " + "-" * 100)

    for pattern in sorted(pattern_results):
        s = pattern_results[pattern]
        print(f"  {pattern:<14} {s['direction']:>5} "
              f"{s['tp']:5.2f} {s['sl']:5.2f} | "
              f"{s['total_trades']:4d} {s['tp_count']:4d} "
              f"{s['sl_count']:4d} {s['to_count']:4d} | "
              f"{s['peak_mfe_median']:8.3f} {s['peak_mfe_p25']:8.3f} "
              f"{s['peak_mfe_p75']:8.3f} {s['peak_offset_median']:8d}")

    print(f"\n  Total trades tracked: {len(all_trades_flat)}, "
          f"Patterns: {len(pattern_results)}, Elapsed: {elapsed:.1f}s")

    return pattern_results, all_trades_flat


# ===================================================================
# PHASE 2: MFE->SL Reversal Frequency
# ===================================================================

def phase2_mfe_reversal(all_trades):
    """
    For trades ending in SL, check if MFE reached TP*X% before SL hit.
    These are trades a trailing stop could have saved.
    """
    print("\n" + "=" * 110)
    print("  PHASE 2: MFE->SL REVERSAL FREQUENCY")
    print(f"  Thresholds: {MFE_THRESHOLDS}% of TP")
    print("=" * 110)

    sl_trades = [t for t in all_trades if t['exit_reason'] == 'SL']
    tp_trades = [t for t in all_trades if t['exit_reason'] == 'TP']
    to_trades = [t for t in all_trades if t['exit_reason'] == 'TIMEOUT']
    total = len(all_trades)

    print(f"\n  Total trades: {total}")
    print(f"  TP exits:      {len(tp_trades)} ({len(tp_trades)/max(total,1)*100:.1f}%)")
    print(f"  SL exits:      {len(sl_trades)} ({len(sl_trades)/max(total,1)*100:.1f}%)")
    print(f"  TIMEOUT exits: {len(to_trades)} ({len(to_trades)/max(total,1)*100:.1f}%)")

    # For each threshold X, count SL trades where peak_mfe >= tp_eff * X/100
    reversal_stats = {}
    for x in MFE_THRESHOLDS:
        reversal_count = 0
        for t in sl_trades:
            tp_eff = t.get('tp_pct_eff', 0)
            threshold_pct = tp_eff * x / 100
            if t['peak_mfe_pct'] >= threshold_pct:
                reversal_count += 1

        pct_of_sl = reversal_count / max(len(sl_trades), 1) * 100
        pct_of_all = reversal_count / max(total, 1) * 100

        reversal_stats[x] = {
            'threshold_pct': x,
            'reversal_count': reversal_count,
            'sl_trades': len(sl_trades),
            'pct_of_sl': round(pct_of_sl, 1),
            'pct_of_all': round(pct_of_all, 1),
        }

    # Print
    print(f"\n  {'Threshold':>12} | {'Reversals':>10} {'of SL':>8} {'of All':>8}")
    print("  " + "-" * 48)
    for x in sorted(reversal_stats):
        s = reversal_stats[x]
        print(f"  MFE>={x:2d}%TP  | {s['reversal_count']:10d} "
              f"{s['pct_of_sl']:7.1f}% {s['pct_of_all']:7.1f}%")

    # Direction breakdown
    print(f"\n  --- Direction Breakdown ---")
    for direction in ['LONG', 'SHORT']:
        dir_sl = [t for t in sl_trades if t['direction'] == direction]
        dir_all = [t for t in all_trades if t['direction'] == direction]
        print(f"\n  {direction}: {len(dir_sl)} SL trades / {len(dir_all)} total")
        for x in MFE_THRESHOLDS:
            count = 0
            for t in dir_sl:
                tp_eff = t.get('tp_pct_eff', 0)
                threshold_pct = tp_eff * x / 100
                if t['peak_mfe_pct'] >= threshold_pct:
                    count += 1
            pct = count / max(len(dir_sl), 1) * 100
            print(f"    MFE>={x:2d}%TP: {count}/{len(dir_sl)} = {pct:.1f}%")

    return reversal_stats


# ===================================================================
# PHASE 3: Trailing Stop Simulation
# ===================================================================

def simulate_hedge_with_trailing(max_positions, schedule,
                                 opens, highs, lows,
                                 lo_bar, hi_bar,
                                 activation_pct, trail_pct,
                                 timeout_bars=None):
    """
    Hedge simulation with trailing stop.

    activation_pct: MFE must reach TP * activation_pct / 100 to activate.
    trail_pct: 0 = breakeven (SL -> entry),
               >0 = dynamic (SL -> entry + (running_mfe - entry) * trail_pct%).
    Trailing SL only moves in protective direction (never retreats).
    """
    slots = {}
    slot_counter = 0
    results = []
    weight = 1.0 / max_positions
    n_bars = len(opens)

    # Per-slot trailing state
    slot_mfe_price = {}       # sid -> running best favorable price
    slot_trail_active = {}    # sid -> bool
    slot_trail_sl = {}        # sid -> current trailing SL price

    def _close_slot(sid, exit_bar, exit_price, reason):
        pos = slots.pop(sid)
        slot_mfe_price.pop(sid, None)
        slot_trail_active.pop(sid, None)
        slot_trail_sl.pop(sid, None)

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

        h, l, o = highs[bar], lows[bar], opens[bar]

        sids_to_close = []
        for sid, pos in slots.items():
            # --- Update running MFE price ---
            if pos.direction == 1:
                cur_mfe = slot_mfe_price.get(sid, pos.entry_price)
                slot_mfe_price[sid] = max(cur_mfe, h)
            else:
                cur_mfe = slot_mfe_price.get(sid, pos.entry_price)
                slot_mfe_price[sid] = min(cur_mfe, l)

            # --- Trailing activation check ---
            mfe_pct = abs(slot_mfe_price[sid] - pos.entry_price) / pos.entry_price * 100
            tp_threshold = pos.tp_pct * activation_pct / 100

            if not slot_trail_active.get(sid, False):
                if mfe_pct >= tp_threshold:
                    slot_trail_active[sid] = True
                    # Initial trailing SL
                    if trail_pct == 0:
                        # Strategy A: breakeven
                        slot_trail_sl[sid] = pos.entry_price
                    else:
                        # Strategy B: dynamic
                        mfe_dist = abs(slot_mfe_price[sid] - pos.entry_price)
                        trail_dist = mfe_dist * trail_pct / 100
                        if pos.direction == 1:
                            slot_trail_sl[sid] = pos.entry_price + trail_dist
                        else:
                            slot_trail_sl[sid] = pos.entry_price - trail_dist
            elif trail_pct > 0:
                # Update dynamic trailing SL (only moves protectively)
                mfe_dist = abs(slot_mfe_price[sid] - pos.entry_price)
                trail_dist = mfe_dist * trail_pct / 100
                if pos.direction == 1:
                    new_trail = pos.entry_price + trail_dist
                    slot_trail_sl[sid] = max(slot_trail_sl.get(sid, 0), new_trail)
                else:
                    new_trail = pos.entry_price - trail_dist
                    slot_trail_sl[sid] = min(
                        slot_trail_sl.get(sid, float('inf')), new_trail)

            # --- Effective SL: trailing if active, else original ---
            if slot_trail_active.get(sid, False):
                eff_sl = slot_trail_sl[sid]
            else:
                eff_sl = pos.sl_price

            # --- Check TP / SL ---
            if pos.direction == 1:
                tp_hit = h >= pos.tp_price
                sl_hit = l <= eff_sl
            else:
                tp_hit = l <= pos.tp_price
                sl_hit = h >= eff_sl

            if tp_hit and sl_hit:
                tp_dist = abs(pos.tp_price - o)
                sl_dist = abs(eff_sl - o)
                if tp_dist <= sl_dist:
                    tp_hit, sl_hit = True, False
                else:
                    tp_hit, sl_hit = False, True

            if tp_hit:
                sids_to_close.append((sid, bar, pos.tp_price, 'TP'))
            elif sl_hit:
                reason = 'TRAIL' if slot_trail_active.get(sid, False) else 'SL'
                sids_to_close.append((sid, bar, eff_sl, reason))

        for sid, ebar, eprice, reason in sids_to_close:
            if sid in slots:
                _close_slot(sid, ebar, eprice, reason)

        # Check timeouts
        if timeout_bars and timeout_bars > 0:
            for sid in list(slots):
                if sid not in slots:
                    continue
                pos = slots[sid]
                if bar - pos.entry_bar >= timeout_bars:
                    slots.pop(sid)  # DROP
                    slot_mfe_price.pop(sid, None)
                    slot_trail_active.pop(sid, None)
                    slot_trail_sl.pop(sid, None)

        # Process signals
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

        pat_name, sig_dir, tp, sl, entry_price, sig_bar = chosen

        if len(slots) < max_positions:
            slot_counter += 1
            sid = f's{slot_counter}'
            pos = ActivePosition(pat_name, sig_dir, entry_price, tp, sl, bar, sig_bar)
            slots[sid] = pos
            slot_mfe_price[sid] = entry_price
            slot_trail_active[sid] = False

    # Close remaining at end
    for sid in list(slots):
        exit_price = opens[min(hi_bar - 1, n_bars - 1)]
        _close_slot(sid, hi_bar - 1, exit_price, 'END')

    return results


def phase3_trailing_stop(all_patterns, sig_index, opens, highs, lows,
                         n_bars, atr_ratio, overlap_bar):
    """Simulate 9 trailing combos and compare with no-trail baseline."""
    print("\n" + "=" * 110)
    print("  PHASE 3: TRAILING STOP SIMULATION")
    print(f"  Activation X: {ACTIVATION_PCTS}% of TP")
    print(f"  Trail Y:      {TRAIL_PCTS}% (0=breakeven)")
    print(f"  Combos: {len(ACTIVATION_PCTS) * len(TRAIL_PCTS)}")
    print(f"  Hedge N={MAX_POSITIONS}, T864, ATR-scaled")
    print("=" * 110)

    t0 = time.time()
    sched = build_custom_schedule(all_patterns, sig_index, atr_ratio, opens, n_bars)

    # Baseline: no trailing stop
    baseline_res = simulate_hedge_simple(
        MAX_POSITIONS, sched, opens, highs, lows,
        overlap_bar, n_bars, TIMEOUT_T864)
    baseline_port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in baseline_res]
    baseline_stats = calc_stats(baseline_port)
    baseline_stats['pnl_mdd_ratio'] = pnl_mdd_ratio(
        baseline_stats['cmp_pnl'], baseline_stats['cmp_mdd'])
    baseline_stats['trail_exits'] = 0

    # Trail combos
    combo_results = {}
    for x in ACTIVATION_PCTS:
        for y in TRAIL_PCTS:
            label = f"X{x}_Y{y}"
            sim = simulate_hedge_with_trailing(
                MAX_POSITIONS, sched, opens, highs, lows,
                overlap_bar, n_bars, x, y,
                timeout_bars=TIMEOUT_T864)
            port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in sim]
            stats = calc_stats(port)
            stats['pnl_mdd_ratio'] = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])

            trail_count = sum(1 for r in sim if r['reason'] == 'TRAIL')
            stats['trail_exits'] = trail_count
            stats['activation_pct'] = x
            stats['trail_pct'] = y

            # Improvement vs baseline
            bl_ratio = baseline_stats['pnl_mdd_ratio']
            if bl_ratio > 0:
                improvement = (stats['pnl_mdd_ratio'] / bl_ratio - 1) * 100
            else:
                improvement = 0
            stats['improvement_pct'] = round(improvement, 1)

            combo_results[label] = stats

    elapsed = time.time() - t0

    # Print
    print(f"\n  Baseline (no trail): trades={baseline_stats['trades']}, "
          f"WR={baseline_stats['wr']:.1f}%, PnL={baseline_stats['cmp_pnl']:+.1f}%, "
          f"MDD={baseline_stats['cmp_mdd']:.1f}%, "
          f"PnL/MDD={baseline_stats['pnl_mdd_ratio']:.2f}")

    print(f"\n  {'Combo':>10} | {'Trades':>7} {'WR':>6} {'PF':>6} | "
          f"{'PnL':>9} {'MDD':>8} {'PnL/MDD':>8} | "
          f"{'Trail#':>7} {'Improv':>8}")
    print("  " + "-" * 85)

    for label in sorted(combo_results):
        s = combo_results[label]
        print(f"  {label:>10} | {s['trades']:7d} {s['wr']:5.1f}% {s['pf']:6.2f} | "
              f"{s['cmp_pnl']:>+9.1f} {s['cmp_mdd']:8.1f} "
              f"{s['pnl_mdd_ratio']:8.2f} | "
              f"{s['trail_exits']:7d} {s['improvement_pct']:>+7.1f}%")

    # Find best
    best_label = max(combo_results, key=lambda k: combo_results[k]['pnl_mdd_ratio'])
    best = combo_results[best_label]
    print(f"\n  Best combo: {best_label} "
          f"(PnL/MDD={best['pnl_mdd_ratio']:.2f}, "
          f"improvement={best['improvement_pct']:+.1f}%)  [{elapsed:.1f}s]")

    return baseline_stats, combo_results, best_label


# ===================================================================
# PHASE 4: Portfolio Impact + WF 3-fold
# ===================================================================

def run_wf_3fold_trailing(max_positions, schedule, opens, highs, lows,
                          overlap_bar, n_bars, activation_pct, trail_pct,
                          timeout_bars=None):
    """3-fold expanding window WF with trailing stop."""
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
        oos_res = simulate_hedge_with_trailing(
            max_positions, schedule, opens, highs, lows,
            oos_lo, oos_hi, activation_pct, trail_pct,
            timeout_bars)
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


def phase4_portfolio_wf(all_patterns, best_combo_label, combo_results,
                        sig_index, opens, highs, lows, n_bars,
                        atr_ratio, overlap_bar):
    """WF 3-fold for best trailing strategy vs no-trail baseline."""
    print("\n" + "=" * 110)
    print("  PHASE 4: PORTFOLIO IMPACT + WF 3-FOLD VALIDATION")
    print(f"  Best combo: {best_combo_label}")
    print("=" * 110)

    best = combo_results[best_combo_label]
    act_pct = best['activation_pct']
    trl_pct = best['trail_pct']

    sched = build_custom_schedule(all_patterns, sig_index, atr_ratio, opens, n_bars)

    # WF for trailing
    wf_trail, verdict_trail = run_wf_3fold_trailing(
        MAX_POSITIONS, sched, opens, highs, lows,
        overlap_bar, n_bars, act_pct, trl_pct,
        TIMEOUT_T864)

    # WF for baseline (no trail)
    wf_baseline, verdict_baseline = run_wf_3fold(
        MAX_POSITIONS, sched, opens, highs, lows,
        overlap_bar, n_bars, TIMEOUT_T864)

    print(f"\n  {'Fold':>4} | {'Trail':>10} {'PnL':>8} {'MDD':>8} | "
          f"{'No Trail':>10} {'PnL':>8} {'MDD':>8}")
    print("  " + "-" * 65)
    for ft, fb in zip(wf_trail, wf_baseline):
        pt = 'PASS' if ft['pass'] else 'FAIL'
        pb = 'PASS' if fb['pass'] else 'FAIL'
        print(f"  {ft['fold']:4d} | {pt:>10} {ft['oos_pnl']:>+8.1f} "
              f"{ft['oos_mdd']:8.1f} | "
              f"{pb:>10} {fb['oos_pnl']:>+8.1f} {fb['oos_mdd']:8.1f}")

    print(f"\n  Trail ({best_combo_label}): {verdict_trail}")
    print(f"  No Trail (baseline):    {verdict_baseline}")

    # If best fails WF, try other combos in descending PnL/MDD order
    fallback = None
    if '3/3' not in verdict_trail:
        print(f"\n  Best combo failed WF, trying alternatives...")
        for label in sorted(combo_results,
                            key=lambda k: -combo_results[k]['pnl_mdd_ratio']):
            if label == best_combo_label:
                continue
            alt = combo_results[label]
            wf_alt, verdict_alt = run_wf_3fold_trailing(
                MAX_POSITIONS, sched, opens, highs, lows,
                overlap_bar, n_bars,
                alt['activation_pct'], alt['trail_pct'],
                TIMEOUT_T864)
            if '3/3' in verdict_alt:
                fallback = {
                    'label': label, 'wf': wf_alt, 'verdict': verdict_alt,
                    'activation_pct': alt['activation_pct'],
                    'trail_pct': alt['trail_pct'],
                }
                print(f"    Fallback: {label} -> {verdict_alt}")
                break

    return {
        'best': {
            'label': best_combo_label,
            'wf': wf_trail, 'verdict': verdict_trail,
            'activation_pct': act_pct, 'trail_pct': trl_pct,
        },
        'baseline': {'wf': wf_baseline, 'verdict': verdict_baseline},
        'fallback': fallback,
    }


# ===================================================================
# PHASE 5: Final Judgment
# ===================================================================

def phase5_judgment(p1_pattern_results, p2_reversal, p3_baseline, p3_combos,
                    best_combo_label, p4_wf):
    """Final judgment: adopt trailing stop or not."""
    print("\n" + "=" * 110)
    print("  PHASE 5: FINAL JUDGMENT")
    print("=" * 110)

    # Determine best WF-passing combo
    wf_result = p4_wf['best']
    effective_label = best_combo_label
    if '3/3' not in wf_result['verdict'] and p4_wf.get('fallback'):
        wf_result = p4_wf['fallback']
        effective_label = wf_result['label']

    wf_pass = '3/3' in wf_result.get('verdict', '')
    baseline_wf_pass = '3/3' in p4_wf['baseline']['verdict']

    # PnL/MDD improvement
    best_combo = p3_combos.get(effective_label, {})
    improvement = best_combo.get('improvement_pct', 0)

    # Adoption criteria: WF PASS + PnL/MDD > 5% improvement
    adopt = wf_pass and improvement > 5.0

    print(f"\n  Adoption Criteria:")
    print(f"    1. WF 3/3 PASS:              "
          f"{'YES' if wf_pass else 'NO'} ({wf_result.get('verdict', 'N/A')})")
    print(f"    2. PnL/MDD > 5% improvement: "
          f"{'YES' if improvement > 5 else 'NO'} ({improvement:+.1f}%)")
    print(f"    -> ADOPT: {'YES' if adopt else 'NO'}")

    # MFE reversal summary
    if p2_reversal:
        r50 = p2_reversal.get(50, {})
        r70 = p2_reversal.get(70, {})
        print(f"\n  MFE->SL Reversal (trades reaching X% of TP then hitting SL):")
        print(f"    50% of TP reached then SL: "
              f"{r50.get('reversal_count', 0)} = {r50.get('pct_of_sl', 0):.1f}% of SL trades")
        print(f"    70% of TP reached then SL: "
              f"{r70.get('reversal_count', 0)} = {r70.get('pct_of_sl', 0):.1f}% of SL trades")

    # Overfit risk assessment
    n_combos = len(ACTIVATION_PCTS) * len(TRAIL_PCTS)
    print(f"\n  Overfit Risk Assessment:")
    print(f"    Combos tested: {n_combos}")
    print(f"    Baseline WF:   {'PASS' if baseline_wf_pass else 'FAIL'}")
    print(f"    Trail WF:      {'PASS' if wf_pass else 'FAIL'}")
    if adopt:
        print(f"    Risk: MODERATE (1 extra parameter [activation%], small grid)")
    else:
        print(f"    Risk: N/A (not adopted)")

    if adopt:
        act = wf_result.get('activation_pct',
                            best_combo.get('activation_pct', 0))
        trail = wf_result.get('trail_pct',
                              best_combo.get('trail_pct', 0))
        print(f"\n  --- PRODUCTION CHANGES ---")
        print(f"  1. Add trailing stop to position_monitor.py:")
        print(f"     activation_threshold: MFE >= TP * {act}%")
        if trail == 0:
            print(f"     strategy: BREAKEVEN (SL -> entry)")
        else:
            print(f"     strategy: DYNAMIC (SL -> entry + (MFE-entry) * {trail}%)")
        print(f"  2. Track per-position running MFE in bot state")
        print(f"  3. Update TP/SL verify logic to handle trailing SL")
    else:
        print(f"\n  --- NO PRODUCTION CHANGES ---")
        print(f"  Trailing stop does NOT meet adoption criteria.")
        if not wf_pass:
            print(f"  Reason: WF FAIL -- overfit to in-sample")
        elif improvement <= 5:
            print(f"  Reason: Improvement {improvement:+.1f}% <= 5% threshold")

    return {
        'adopt': adopt,
        'best_combo': effective_label,
        'wf_pass': wf_pass,
        'improvement_pct': round(improvement, 1),
        'activation_pct': wf_result.get('activation_pct', 0),
        'trail_pct': wf_result.get('trail_pct', 0),
    }


# ===================================================================
# MAIN
# ===================================================================

def _load_study1_expanded():
    """Load expanded patterns from Study 1 output, fallback to current."""
    if os.path.exists(STUDY1_OUTPUT):
        try:
            with open(STUDY1_OUTPUT) as f:
                s1 = json.load(f)
            restored = s1.get('restored_longs', [])
            if restored:
                current = load_patterns()
                expanded = list(current) + restored
                n_l = sum(1 for p in expanded if p['direction'] == 'LONG')
                n_s = sum(1 for p in expanded if p['direction'] == 'SHORT')
                logger.info(f"Study 1 expanded: {len(current)}+{len(restored)}"
                            f"={len(expanded)} ({n_l}L+{n_s}S)")
                return expanded
        except Exception as e:
            logger.warning(f"Failed to load Study 1: {e}")

    # Fallback: current patterns only
    current = load_patterns()
    logger.info(f"Study 1 not available, using current {len(current)} patterns")
    return current


def main():
    t0 = time.time()

    print("=" * 110)
    print("  TRAILING STOP MFE STUDY")
    print("=" * 110)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n--- Loading 720d data ---")
    df = load_and_classify(DATA_720D)
    n_bars = len(df)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    types = df['rctype'].values

    overlap_bar = find_overlap_bar(df)
    avail_days = (n_bars - overlap_bar) / BARS_PER_DAY
    logger.info(f"720d: {n_bars} bars, overlap: {overlap_bar}, avail: {avail_days:.1f}d")

    sig_index = build_signal_index(types, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    # ------------------------------------------------------------------
    # Load patterns (prefer Study 1 expanded set)
    # ------------------------------------------------------------------
    all_patterns = _load_study1_expanded()
    n_long = sum(1 for p in all_patterns if p['direction'] == 'LONG')
    n_short = sum(1 for p in all_patterns if p['direction'] == 'SHORT')
    logger.info(f"Patterns: {len(all_patterns)} ({n_long}L + {n_short}S)")

    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'study': 'Trailing Stop MFE Study',
        'data_bars': n_bars,
        'overlap_bar': overlap_bar,
        'n_patterns': len(all_patterns),
        'n_long': n_long, 'n_short': n_short,
        'parameters': {
            'max_positions': MAX_POSITIONS,
            'timeout': TIMEOUT_T864,
            'mfe_thresholds': MFE_THRESHOLDS,
            'activation_pcts': ACTIVATION_PCTS,
            'trail_pcts': TRAIL_PCTS,
            'leverage': LEVERAGE,
            'fee': FEE,
        },
    }

    # ------------------------------------------------------------------
    # PHASE 1
    # ------------------------------------------------------------------
    p1_patterns, p1_trades = phase1_mfe_paths(
        all_patterns, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)
    output['phase1_mfe_paths'] = p1_patterns

    # ------------------------------------------------------------------
    # PHASE 2
    # ------------------------------------------------------------------
    p2_reversal = phase2_mfe_reversal(p1_trades)
    output['phase2_mfe_reversal'] = {
        str(k): v for k, v in p2_reversal.items()
    }

    # ------------------------------------------------------------------
    # PHASE 3
    # ------------------------------------------------------------------
    p3_baseline, p3_combos, best_label = phase3_trailing_stop(
        all_patterns, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)
    output['phase3_trailing_stop'] = {
        'baseline': p3_baseline,
        'combos': p3_combos,
        'best_combo': best_label,
    }

    # ------------------------------------------------------------------
    # PHASE 4
    # ------------------------------------------------------------------
    p4_wf = phase4_portfolio_wf(
        all_patterns, best_label, p3_combos,
        sig_index, opens, highs, lows, n_bars,
        atr_ratio, overlap_bar)
    output['phase4_wf'] = p4_wf

    # ------------------------------------------------------------------
    # PHASE 5
    # ------------------------------------------------------------------
    p5_judgment = phase5_judgment(
        p1_patterns, p2_reversal, p3_baseline, p3_combos,
        best_label, p4_wf)
    output['phase5_judgment'] = p5_judgment

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(f"  Results saved to {OUTPUT_FILE}")
    print("=" * 110)


if __name__ == '__main__':
    main()
