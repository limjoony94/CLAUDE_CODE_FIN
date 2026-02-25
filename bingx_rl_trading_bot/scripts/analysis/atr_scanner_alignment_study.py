#!/usr/bin/env python3
"""
ATR Scanner-Production Alignment Study
=======================================

Core Question:
  Scanner selects patterns using fixed TP/SL, but Production trades with
  ATR-scaled TP/SL. Does integrating ATR into the scanner's MAE/MFE
  discovery improve pattern selection and WF OOS performance?

Four Phases (Go/No-Go gates):
  Phase 1: ATR-Integrated Scanner — compare Fixed vs ATR pattern selection
  Phase 2: Portfolio WF Validation — Hedge N=9, Direction Cap 6, T864
  Phase 3: Parameter Sweep — ATR period/window/clamp optimization
  Phase 4: Production Candidate — final deploy decision

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based, same-bar resolution from bar OPEN)
  - Sizing: compound (1/N per slot)
  - Fee: FEE_PCT * LEVERAGE = 0.30% capital-space
  - Timeout: 288 bars (24h) -> DROP from PnL
  - WF: 3-fold expanding window within 270d overlap of 720d data
  - classify_candle: production import (NEVER self-implement)

Data:
  - 270d: btc_5m_270days_reclassified.csv (IS ground truth)
  - 720d: btc_5m_720days_binance.csv (WF with overlap region)

Baseline: v1.33.0 (35 patterns, 9L+26S, Compact TP/SL, WR 68.1%, edge 0.221%)
"""

import os
import sys
import json
import time
import logging

from multiprocessing import Pool, cpu_count

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
from scripts.analysis.fifo_vs_hedge_backtest_study import (
    build_signal_schedule, ActivePosition, pnl_mdd_ratio,
)
from scripts.analysis.tpsl_regime_bias_study import (
    backtest_pattern_trades, compute_wr_and_edge,
    simulate_hedge_simple, run_wf_3fold,
    build_custom_schedule, load_patterns_from_file,
)
from scripts.scanner.pattern_scanner import (
    mc_test, compute_excursions, derive_tp_sl,
    MC_SEEDS, MC_SIMS, TP_PERCENTILES, SL_PERCENTILES,
    MAX_BASELINE_WR,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('atr_scanner_alignment')

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_270D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
DATA_720D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'atr_scanner_alignment_study.json')

# ---------------------------------------------------------------------------
# Quality filters (v1.33.0 standard)
# ---------------------------------------------------------------------------
EDGE_THRESHOLD = 21.8      # pp
WR_THRESHOLD = 60.0        # %
SL_MIN = 1.0               # %
MC_THRESHOLD = 0.01
MIN_TRADES = 25
WR_EXCESS_MIN = 5.0        # pp

# Portfolio simulation
MAX_POSITIONS = 9
DIRECTION_CAP = 6
TIMEOUT_T864 = 864          # 72h

# Compact TP/SL caps (v1.33.0)
COMPACT_TP_MAX = 2.0
COMPACT_SL_MAX = 2.5

# Phase 3: Parameter sweep
ATR_PERIODS = [7, 14, 21, 28]
ATR_WINDOWS = [288, 576, 864, 1152]
CLAMP_LOS = [0.5, 0.6, 0.7, 0.8]
CLAMP_HIS = [1.3, 1.5, 1.7, 2.0]


# ===================================================================
# ATR-AWARE BACKTEST: bt_signals with ATR scaling
# ===================================================================

def bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                   opens, highs, lows, n_bars,
                   atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI):
    """
    Backtest with ATR-scaled TP/SL per signal.

    Same protocol as scanner bt_signals() but applies ATR ratio scaling
    to TP/SL at each signal bar. Timeout trades are DROPPED.

    Returns list of (entry_bar, exit_bar, pnl).
    """
    fee = FEE_PCT * LEVERAGE
    is_long = (direction == 'LONG')
    trades = []

    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1

        # ATR scaling at signal bar
        if (atr_ratio is not None and idx < len(atr_ratio)
                and not np.isnan(atr_ratio[idx])):
            r = max(clamp_lo, min(clamp_hi, atr_ratio[idx]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE_BUFFER
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

        if is_long:
            tpp = entry * (1 + eff_tp / 100)
            slp = entry * (1 - eff_sl / 100)
        else:
            tpp = entry * (1 - eff_tp / 100)
            slp = entry * (1 + eff_sl / 100)

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if is_long:
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (eff_tp if dist_tp <= dist_sl else -eff_sl) * LEVERAGE - fee
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, eff_tp * LEVERAGE - fee))
                break
            elif hs:
                trades.append((eb, j, -eff_sl * LEVERAGE - fee))
                break
        # Timeout → DROP

    return trades


# ===================================================================
# ATR-AWARE MAE/MFE GRID SEARCH
# ===================================================================

def grid_search_mae_mfe_atr(signal_bars, direction, opens, highs, lows, n_bars,
                             atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI,
                             max_bars=MAX_BARS, min_tr=20,
                             max_baseline_wr=MAX_BASELINE_WR):
    """
    Grid search over MFE/MAE percentiles → derive base TP/SL →
    backtest with ATR-scaled execution → select best by PnL/MDD.

    Key difference from scanner's grid_search_mae_mfe():
      - Excursions are computed WITHOUT TP/SL (same as scanner)
      - Derived TP/SL are used as BASE values
      - Backtest applies ATR scaling at each signal (production-aligned)
    """
    excursions = compute_excursions(signal_bars, direction, opens, highs, lows,
                                    n_bars, max_bars)
    if len(excursions) < min_tr:
        return None, None

    best = None
    best_score = -9999

    for tp_p in TP_PERCENTILES:
        for sl_p in SL_PERCENTILES:
            tp, sl = derive_tp_sl(excursions, tp_p, sl_p)
            if tp is None or sl is None:
                continue
            if tp < 0.1 or sl < 0.3:
                continue

            bwr = sl / (tp + sl) * 100
            if bwr > max_baseline_wr:
                continue

            # ATR-scaled backtest
            trades = bt_signals_atr(signal_bars, direction, tp, sl,
                                    opens, highs, lows, n_bars,
                                    atr_ratio, clamp_lo, clamp_hi)
            if len(trades) < min_tr:
                continue

            pnls = [t[2] for t in trades]
            cum = 0; peak = 0; mdd_val = 0; w = 0
            for p in pnls:
                cum += p
                if cum > peak: peak = cum
                dd = peak - cum
                if dd > mdd_val: mdd_val = dd
                if p > 0: w += 1

            wr = w / len(pnls) * 100
            score = (cum / mdd_val) if mdd_val > 0 else cum

            if score > best_score:
                best_score = score
                best = {
                    'tp': tp, 'sl': sl,
                    'tp_percentile': tp_p, 'sl_percentile': sl_p,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'pnl': round(cum, 1), 'mdd': round(mdd_val, 1),
                }

    # Excursion stats
    mfes = np.array([e['mfe_pct'] for e in excursions])
    maes = np.array([e['mae_pct'] for e in excursions])
    bars_mfe = np.array([e['bars_to_mfe'] for e in excursions])

    exc_stats = {
        'n_signals': len(excursions),
        'mfe_median': round(float(np.median(mfes)), 3),
        'mfe_mean': round(float(np.mean(mfes)), 3),
        'mae_median': round(float(np.median(maes)), 3),
        'mae_mean': round(float(np.mean(maes)), 3),
        'mae_p90': round(float(np.percentile(maes, 90)), 3),
        'bars_to_mfe_median': round(float(np.median(bars_mfe)), 1),
        'mfe_mae_ratio': round(float(np.median(mfes) / np.median(maes))
                               if np.median(maes) > 0 else 0, 3),
    }

    return best, exc_stats


# ===================================================================
# COMPACT TP/SL CAPPING
# ===================================================================

def apply_compact_cap(tp, sl):
    """Cap TP/SL to Compact range (v1.33.0)."""
    return min(tp, COMPACT_TP_MAX), min(sl, COMPACT_SL_MAX)


# ===================================================================
# QUALITY FILTER (same as production scanner)
# ===================================================================

def quality_filter(pattern_key, direction, tp, sl, trades_list,
                   opens, highs, lows, n_bars, signal_bars):
    """
    Apply production quality filters.
    Returns (passed: bool, info: dict).
    """
    pnls = [t[2] for t in trades_list]
    n_trades = len(trades_list)

    if n_trades < MIN_TRADES:
        return False, {'reason': f'trades {n_trades} < {MIN_TRADES}'}

    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n_trades * 100

    if wr < WR_THRESHOLD:
        return False, {'reason': f'WR {wr:.1f}% < {WR_THRESHOLD}%'}

    if sl < SL_MIN:
        return False, {'reason': f'SL {sl:.2f}% < {SL_MIN}%'}

    # Edge (WR - baseline WR)
    baseline_wr = sl / (tp + sl) * 100
    edge = wr - baseline_wr

    if edge < EDGE_THRESHOLD:
        return False, {'reason': f'edge {edge:.1f}pp < {EDGE_THRESHOLD}pp'}

    # WR Excess
    wr_excess = wr - baseline_wr
    if wr_excess < WR_EXCESS_MIN:
        return False, {'reason': f'WR Excess {wr_excess:.1f}pp < {WR_EXCESS_MIN}pp'}

    # MC test
    mc_p = mc_test(np.array(pnls))
    if mc_p > MC_THRESHOLD:
        return False, {'reason': f'MC p={mc_p:.4f} > {MC_THRESHOLD}'}

    per_trade_edge = sum(pnls) / n_trades

    return True, {
        'pattern': pattern_key, 'direction': direction,
        'tp': tp, 'sl': sl, 'trades': n_trades,
        'wr': round(wr, 1), 'edge': round(edge, 1),
        'wr_excess': round(wr_excess, 1),
        'mc_p': round(mc_p, 4),
        'per_trade_edge': round(per_trade_edge, 3),
        'baseline_wr': round(baseline_wr, 1),
    }


# ===================================================================
# WORKER FOR PARALLEL SCAN
# ===================================================================

def _scan_worker_fixed(args):
    """Worker: Fixed TP/SL scan (scanner's original approach)."""
    (pat_key, direction, sig_bars, opens, highs, lows, n_bars) = args
    from scripts.scanner.pattern_scanner import bt_signals, grid_search_mae_mfe

    best, exc_stats = grid_search_mae_mfe(
        sig_bars, direction, opens, highs, lows, n_bars,
        MAX_BARS, MIN_TRADES, MAX_BASELINE_WR)

    if best is None:
        return None

    tp, sl = apply_compact_cap(best['tp'], best['sl'])
    trades = bt_signals(sig_bars, direction, tp, sl, opens, highs, lows, n_bars)

    passed, info = quality_filter(pat_key, direction, tp, sl, trades,
                                  opens, highs, lows, n_bars, sig_bars)
    if not passed:
        return None

    info['exc_stats'] = exc_stats
    info['raw_tp'] = best['tp']
    info['raw_sl'] = best['sl']
    return info


def _scan_worker_atr(args):
    """Worker: ATR-scaled TP/SL scan."""
    (pat_key, direction, sig_bars, opens, highs, lows, n_bars,
     atr_ratio, clamp_lo, clamp_hi) = args

    best, exc_stats = grid_search_mae_mfe_atr(
        sig_bars, direction, opens, highs, lows, n_bars,
        atr_ratio, clamp_lo, clamp_hi,
        MAX_BARS, MIN_TRADES, MAX_BASELINE_WR)

    if best is None:
        return None

    tp, sl = apply_compact_cap(best['tp'], best['sl'])
    trades = bt_signals_atr(sig_bars, direction, tp, sl,
                            opens, highs, lows, n_bars,
                            atr_ratio, clamp_lo, clamp_hi)

    passed, info = quality_filter(pat_key, direction, tp, sl, trades,
                                  opens, highs, lows, n_bars, sig_bars)
    if not passed:
        return None

    info['exc_stats'] = exc_stats
    info['raw_tp'] = best['tp']
    info['raw_sl'] = best['sl']
    return info


# ===================================================================
# FULL SCAN PIPELINE
# ===================================================================

def run_full_scan(types, opens, highs, lows, n_bars,
                  atr_ratio=None, mode='fixed',
                  clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI):
    """
    Run full pattern scan in fixed or ATR mode.

    Returns dict: {pattern_key: info_dict}
    """
    sig_index = build_signal_index(types, n_bars)
    logger.info(f"  {mode.upper()} scan: {len(sig_index)} unique patterns")

    tasks = []
    for pat_key, sig_bars in sig_index.items():
        if len(sig_bars) < MIN_TRADES:
            continue
        for direction in ['LONG', 'SHORT']:
            if mode == 'fixed':
                tasks.append((pat_key, direction, sig_bars,
                              opens, highs, lows, n_bars))
            else:
                tasks.append((pat_key, direction, sig_bars,
                              opens, highs, lows, n_bars,
                              atr_ratio, clamp_lo, clamp_hi))

    worker = _scan_worker_fixed if mode == 'fixed' else _scan_worker_atr
    logger.info(f"  Processing {len(tasks)} pattern-direction combos...")

    results = {}
    # Sequential (multiprocessing has issues with numpy array sharing)
    for i, task in enumerate(tasks):
        if (i + 1) % 100 == 0:
            logger.info(f"  ... {i + 1}/{len(tasks)}")
        r = worker(task)
        if r is not None:
            key = f"{r['pattern']}_{r['direction']}"
            results[key] = r

    logger.info(f"  {mode.upper()} scan: {len(results)} patterns passed quality filter")
    return results


# ===================================================================
# DIRECTION CAP SCHEDULE BUILDER
# ===================================================================

def build_schedule_with_direction_cap(patterns_list, sig_index, atr_ratio,
                                       opens, n_bars, use_atr=True,
                                       clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI):
    """
    Build signal schedule with ATR scaling (or fixed if use_atr=False).
    Compatible with simulate_hedge_simple().
    """
    schedule = {}
    for p in patterns_list:
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

            if use_atr and atr_ratio is not None:
                if sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                    r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
                else:
                    r = 1.0
            else:
                r = 1.0

            tp = base_tp * r + SLIPPAGE_BUFFER
            sl = max(0.1, base_sl * r - SLIPPAGE_BUFFER)

            if entry_bar not in schedule:
                schedule[entry_bar] = []
            schedule[entry_bar].append(
                (p['pattern'], direction, tp, sl, entry_price, sig_bar))

    return schedule


def simulate_hedge_with_cap(max_positions, direction_cap, schedule,
                            opens, highs, lows, lo_bar, hi_bar,
                            timeout_bars=None):
    """
    Hedge simulation with direction cap.
    Extension of simulate_hedge_simple() that limits same-direction positions.
    """
    slots = {}
    slot_counter = 0
    results = []
    weight = 1.0 / max_positions
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

        # Check TP/SL
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

        # Check timeouts
        if timeout_bars and timeout_bars > 0:
            for sid in list(slots):
                if sid not in slots:
                    continue
                pos = slots[sid]
                if bar - pos.entry_bar >= timeout_bars:
                    slots.pop(sid)  # TIMEOUT -> DROP

        # Process signals
        signals = schedule.get(bar)
        if not signals:
            continue

        active_patterns = {s.pattern for s in slots.values()}
        # Count directions
        long_count = sum(1 for s in slots.values() if s.direction == 1)
        short_count = sum(1 for s in slots.values() if s.direction == -1)

        signals_sorted = sorted(signals, key=lambda s: -s[1])

        for pat_name, sig_dir, tp, sl, entry_price, sig_bar in signals_sorted:
            if len(slots) >= max_positions:
                break
            if pat_name in active_patterns:
                continue
            # Direction cap check
            if sig_dir == 1 and long_count >= direction_cap:
                continue
            if sig_dir == -1 and short_count >= direction_cap:
                continue

            slot_counter += 1
            sid = f's{slot_counter}'
            pos = ActivePosition(pat_name, sig_dir, entry_price, tp, sl, bar, sig_bar)
            slots[sid] = pos
            active_patterns.add(pat_name)
            if sig_dir == 1:
                long_count += 1
            else:
                short_count += 1

    # Close remaining at end
    for sid in list(slots):
        exit_price = opens[min(hi_bar - 1, n_bars - 1)]
        _close_slot(sid, hi_bar - 1, exit_price, 'END')

    return results


def run_wf_3fold_with_cap(max_positions, direction_cap, schedule,
                          opens, highs, lows, overlap_bar, n_bars,
                          timeout_bars=None):
    """3-fold expanding window WF with direction cap."""
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
        oos_res = simulate_hedge_with_cap(
            max_positions, direction_cap, schedule,
            opens, highs, lows, oos_lo, oos_hi, timeout_bars)
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
# PHASE 1: ATR-Integrated Scanner Backtest
# ===================================================================

def phase1_scanner_comparison(types_270, opens_270, highs_270, lows_270,
                               closes_270, n_270):
    """
    Compare Fixed vs ATR pattern selection on 270d data.
    """
    print("\n" + "=" * 110)
    print("  PHASE 1: ATR-Integrated Scanner Backtest")
    print("  Compare Fixed TP/SL scan vs ATR-scaled TP/SL scan")
    print("=" * 110)

    # Compute ATR ratio for 270d data
    atr_ratio = compute_atr_ratio(highs_270, lows_270, closes_270,
                                   ATR_PERIOD, ATR_WINDOW)
    valid_ratio = atr_ratio[~np.isnan(atr_ratio)]
    clamped = np.clip(valid_ratio, CLAMP_LO, CLAMP_HI)
    print(f"\n  ATR ratio stats: mean={np.mean(clamped):.4f}, "
          f"median={np.median(clamped):.4f}, "
          f"std={np.std(clamped):.4f}, "
          f"within_clamps={np.mean((valid_ratio >= CLAMP_LO) & (valid_ratio <= CLAMP_HI))*100:.1f}%")

    # Fixed scan
    t0 = time.time()
    fixed_results = run_full_scan(types_270, opens_270, highs_270, lows_270,
                                   n_270, mode='fixed')
    t_fixed = time.time() - t0
    print(f"\n  Fixed scan: {len(fixed_results)} patterns ({t_fixed:.1f}s)")

    # ATR scan
    t0 = time.time()
    atr_results = run_full_scan(types_270, opens_270, highs_270, lows_270,
                                 n_270, atr_ratio=atr_ratio, mode='atr')
    t_atr = time.time() - t0
    print(f"  ATR scan:   {len(atr_results)} patterns ({t_atr:.1f}s)")

    # Compare pattern sets
    fixed_keys = set(fixed_results.keys())
    atr_keys = set(atr_results.keys())
    overlap = fixed_keys & atr_keys
    only_fixed = fixed_keys - atr_keys
    only_atr = atr_keys - fixed_keys

    print(f"\n  Overlap: {len(overlap)}, Only Fixed: {len(only_fixed)}, Only ATR: {len(only_atr)}")

    # Compare average quality metrics
    def _avg_metrics(results):
        if not results:
            return {}
        wrs = [v['wr'] for v in results.values()]
        edges = [v['edge'] for v in results.values()]
        wr_excess = [v['wr_excess'] for v in results.values()]
        per_trade = [v['per_trade_edge'] for v in results.values()]
        return {
            'count': len(results),
            'avg_wr': round(np.mean(wrs), 1),
            'avg_edge': round(np.mean(edges), 1),
            'avg_wr_excess': round(np.mean(wr_excess), 1),
            'avg_per_trade_edge': round(np.mean(per_trade), 3),
        }

    fixed_metrics = _avg_metrics(fixed_results)
    atr_metrics = _avg_metrics(atr_results)

    print(f"\n  {'Metric':>20} | {'Fixed':>10} | {'ATR':>10} | {'Delta':>10}")
    print("  " + "-" * 60)
    for key in ['count', 'avg_wr', 'avg_edge', 'avg_wr_excess', 'avg_per_trade_edge']:
        fv = fixed_metrics.get(key, 0)
        av = atr_metrics.get(key, 0)
        delta = av - fv if isinstance(fv, (int, float)) else 'N/A'
        print(f"  {key:>20} | {fv:>10} | {av:>10} | {delta:>+10}" if isinstance(delta, (int, float))
              else f"  {key:>20} | {fv:>10} | {av:>10} | {'N/A':>10}")

    # Direction breakdown
    for label, results in [('Fixed', fixed_results), ('ATR', atr_results)]:
        longs = sum(1 for v in results.values() if v['direction'] == 'LONG')
        shorts = sum(1 for v in results.values() if v['direction'] == 'SHORT')
        print(f"  {label}: {longs}L + {shorts}S = {longs + shorts}")

    # Go/No-Go
    delta_excess = atr_metrics.get('avg_wr_excess', 0) - fixed_metrics.get('avg_wr_excess', 0)
    delta_count = atr_metrics.get('count', 0) - fixed_metrics.get('count', 0)
    go = delta_excess >= 2.0 or abs(delta_count) >= 5
    verdict = "GO" if go else "STOP"
    reason = (f"WR Excess delta: {delta_excess:+.1f}pp, "
              f"Count delta: {delta_count:+d}")
    print(f"\n  Phase 1 verdict: {verdict} ({reason})")

    return {
        'fixed': {
            'patterns': {k: {kk: vv for kk, vv in v.items() if kk != 'exc_stats'}
                         for k, v in fixed_results.items()},
            'metrics': fixed_metrics,
        },
        'atr': {
            'patterns': {k: {kk: vv for kk, vv in v.items() if kk != 'exc_stats'}
                         for k, v in atr_results.items()},
            'metrics': atr_metrics,
        },
        'comparison': {
            'overlap': len(overlap),
            'only_fixed': len(only_fixed),
            'only_atr': len(only_atr),
            'only_fixed_keys': sorted(only_fixed),
            'only_atr_keys': sorted(only_atr),
            'delta_wr_excess': round(delta_excess, 1),
            'delta_count': delta_count,
        },
        'atr_ratio_stats': {
            'mean': round(float(np.mean(clamped)), 4),
            'median': round(float(np.median(clamped)), 4),
            'std': round(float(np.std(clamped)), 4),
            'pct_within_clamps': round(float(np.mean(
                (valid_ratio >= CLAMP_LO) & (valid_ratio <= CLAMP_HI)) * 100), 1),
        },
        'verdict': verdict,
        'reason': reason,
        'fixed_results_full': fixed_results,
        'atr_results_full': atr_results,
    }


# ===================================================================
# PHASE 2: Portfolio WF Validation
# ===================================================================

def phase2_portfolio_wf(phase1_data, types_720, opens_720, highs_720, lows_720,
                         closes_720, n_720, overlap_bar):
    """
    Compare Fixed vs ATR portfolio WF OOS performance.
    Hedge N=9, Direction Cap 6, T864.
    """
    print("\n" + "=" * 110)
    print("  PHASE 2: Portfolio WF Validation")
    print(f"  Hedge N={MAX_POSITIONS}, Direction Cap {DIRECTION_CAP}, T={TIMEOUT_T864}")
    print("=" * 110)

    atr_ratio_720 = compute_atr_ratio(highs_720, lows_720, closes_720,
                                       ATR_PERIOD, ATR_WINDOW)
    sig_index_720 = build_signal_index(types_720, n_720)

    scenarios = {}

    for label, scan_results, use_atr in [
        ('fixed', phase1_data['fixed_results_full'], False),
        ('atr', phase1_data['atr_results_full'], True),
    ]:
        # Convert scan results to pattern list
        patterns_list = [
            {'pattern': v['pattern'], 'direction': v['direction'],
             'tp': v['tp'], 'sl': v['sl']}
            for v in scan_results.values()
        ]

        if not patterns_list:
            print(f"\n  {label.upper()}: No patterns → SKIP")
            scenarios[label] = {'verdict': 'SKIP', 'reason': 'no_patterns'}
            continue

        # Debug: check pattern matching
        matched = sum(1 for p in patterns_list if p['pattern'] in sig_index_720)
        print(f"    Pattern match: {matched}/{len(patterns_list)} found in 720d index")

        # Build schedule
        sched = build_schedule_with_direction_cap(
            patterns_list, sig_index_720, atr_ratio_720,
            opens_720, n_720, use_atr=use_atr)
        in_range = sum(1 for bar in sched if bar >= overlap_bar)
        print(f"    Schedule: {len(sched)} total bars, {in_range} in overlap region")

        # Full IS simulation
        is_res = simulate_hedge_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched,
            opens_720, highs_720, lows_720,
            overlap_bar, n_720, TIMEOUT_T864)
        is_port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in is_res]
        is_stats = calc_stats(is_port)
        is_ratio = pnl_mdd_ratio(is_stats['cmp_pnl'], is_stats['cmp_mdd'])

        # WF validation
        wf_folds, wf_verdict = run_wf_3fold_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched,
            opens_720, highs_720, lows_720,
            overlap_bar, n_720, TIMEOUT_T864)

        scenarios[label] = {
            'n_patterns': len(patterns_list),
            'is_stats': is_stats,
            'is_pnl_mdd': is_ratio,
            'wf_verdict': wf_verdict,
            'wf_folds': wf_folds,
        }

        n_long = sum(1 for p in patterns_list if p['direction'] == 'LONG')
        n_short = len(patterns_list) - n_long

        print(f"\n  {label.upper()} ({n_long}L+{n_short}S = {len(patterns_list)} patterns):")
        print(f"    IS: Trades={is_stats['trades']}, WR={is_stats['wr']}%, "
              f"PnL={is_stats['cmp_pnl']:+.1f}%, MDD={is_stats['cmp_mdd']:.1f}%, "
              f"PnL/MDD={is_ratio}")
        print(f"    WF: {wf_verdict}")
        for f in wf_folds:
            status = "PASS" if f['pass'] else "FAIL"
            print(f"      Fold {f['fold']}: OOS Trades={f['oos_trades']}, "
                  f"WR={f['oos_wr']}%, PnL={f['oos_pnl']:+.1f}%, "
                  f"MDD={f['oos_mdd']:.1f}% [{status}]")

    # Go/No-Go
    fixed_wf = scenarios.get('fixed', {}).get('wf_verdict', 'FAIL')
    atr_wf = scenarios.get('atr', {}).get('wf_verdict', 'FAIL')
    atr_ratio_is = scenarios.get('atr', {}).get('is_pnl_mdd', 0)
    fixed_ratio_is = scenarios.get('fixed', {}).get('is_pnl_mdd', 0)

    go = ('PASS (3/3)' in atr_wf and atr_ratio_is > fixed_ratio_is)
    verdict = "GO" if go else "STOP"
    reason = (f"ATR WF={atr_wf}, PnL/MDD={atr_ratio_is} vs Fixed={fixed_ratio_is}")
    print(f"\n  Phase 2 verdict: {verdict} ({reason})")

    return {
        'scenarios': {k: {kk: vv for kk, vv in v.items()}
                      for k, v in scenarios.items()},
        'verdict': verdict,
        'reason': reason,
    }


# ===================================================================
# PHASE 3: Parameter Sweep
# ===================================================================

def phase3_param_sweep(types_270, opens_270, highs_270, lows_270, closes_270, n_270,
                        types_720, opens_720, highs_720, lows_720, closes_720, n_720,
                        overlap_bar):
    """
    2-stage parameter sweep:
      Stage A: period × window (16 combos) with default clamps
      Stage B: clamp_lo × clamp_hi (16 combos) with best period/window
    """
    print("\n" + "=" * 110)
    print("  PHASE 3: Parameter Sweep (2-stage)")
    print("  Stage A: ATR period × window | Stage B: clamp_lo × clamp_hi")
    print("=" * 110)

    sig_index_720 = build_signal_index(types_720, n_720)

    # Stage A: period × window
    print("\n  --- Stage A: Period × Window ---")
    stage_a_results = []

    for period in ATR_PERIODS:
        for window in ATR_WINDOWS:
            t0 = time.time()

            # Compute ATR ratio with this param combo
            atr_270 = compute_atr_ratio(highs_270, lows_270, closes_270, period, window)
            atr_720 = compute_atr_ratio(highs_720, lows_720, closes_720, period, window)

            # Quick scan on 270d
            scan_results = run_full_scan(
                types_270, opens_270, highs_270, lows_270, n_270,
                atr_ratio=atr_270, mode='atr',
                clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI)

            if len(scan_results) < 5:
                print(f"    p={period} w={window}: {len(scan_results)} patterns (< 5, skip)")
                continue

            # Portfolio WF
            patterns_list = [
                {'pattern': v['pattern'], 'direction': v['direction'],
                 'tp': v['tp'], 'sl': v['sl']}
                for v in scan_results.values()
            ]

            sched = build_schedule_with_direction_cap(
                patterns_list, sig_index_720, atr_720,
                opens_720, n_720, use_atr=True,
                clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI)

            wf_folds, wf_verdict = run_wf_3fold_with_cap(
                MAX_POSITIONS, DIRECTION_CAP, sched,
                opens_720, highs_720, lows_720,
                overlap_bar, n_720, TIMEOUT_T864)

            # IS stats
            is_res = simulate_hedge_with_cap(
                MAX_POSITIONS, DIRECTION_CAP, sched,
                opens_720, highs_720, lows_720,
                overlap_bar, n_720, TIMEOUT_T864)
            is_port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in is_res]
            is_stats = calc_stats(is_port)
            is_ratio = pnl_mdd_ratio(is_stats['cmp_pnl'], is_stats['cmp_mdd'])

            elapsed = time.time() - t0
            print(f"    p={period:>2} w={window:>4}: {len(scan_results):>3} pats, "
                  f"PnL={is_stats['cmp_pnl']:>+8.1f}%, MDD={is_stats['cmp_mdd']:>6.1f}%, "
                  f"PnL/MDD={is_ratio:>7.2f}, WF={wf_verdict:>12} ({elapsed:.0f}s)")

            stage_a_results.append({
                'period': period, 'window': window,
                'n_patterns': len(scan_results),
                'is_pnl': is_stats['cmp_pnl'], 'is_mdd': is_stats['cmp_mdd'],
                'pnl_mdd': is_ratio,
                'wf_verdict': wf_verdict,
                'wf_folds': wf_folds,
            })

    # Find best period/window (PASS only, then best PnL/MDD)
    passing_a = [r for r in stage_a_results if 'PASS (3/3)' in r['wf_verdict']]
    if not passing_a:
        passing_a = stage_a_results  # fallback to all if none pass

    best_a = max(passing_a, key=lambda r: r['pnl_mdd']) if passing_a else None

    if best_a:
        print(f"\n  Best Stage A: p={best_a['period']} w={best_a['window']} "
              f"(PnL/MDD={best_a['pnl_mdd']}, WF={best_a['wf_verdict']})")
    else:
        print("\n  Stage A: No results")
        return {'stage_a': stage_a_results, 'stage_b': [], 'best': None,
                'verdict': 'STOP', 'reason': 'no_stage_a_results'}

    # Stage B: clamp sweep with best period/window
    print(f"\n  --- Stage B: Clamp Sweep (p={best_a['period']}, w={best_a['window']}) ---")
    stage_b_results = []
    best_period = best_a['period']
    best_window = best_a['window']

    for c_lo in CLAMP_LOS:
        for c_hi in CLAMP_HIS:
            if c_lo >= c_hi:
                continue

            t0 = time.time()
            atr_270 = compute_atr_ratio(highs_270, lows_270, closes_270,
                                         best_period, best_window)
            atr_720 = compute_atr_ratio(highs_720, lows_720, closes_720,
                                         best_period, best_window)

            scan_results = run_full_scan(
                types_270, opens_270, highs_270, lows_270, n_270,
                atr_ratio=atr_270, mode='atr',
                clamp_lo=c_lo, clamp_hi=c_hi)

            if len(scan_results) < 5:
                print(f"    lo={c_lo} hi={c_hi}: {len(scan_results)} patterns (< 5, skip)")
                continue

            patterns_list = [
                {'pattern': v['pattern'], 'direction': v['direction'],
                 'tp': v['tp'], 'sl': v['sl']}
                for v in scan_results.values()
            ]

            sched = build_schedule_with_direction_cap(
                patterns_list, sig_index_720, atr_720,
                opens_720, n_720, use_atr=True,
                clamp_lo=c_lo, clamp_hi=c_hi)

            wf_folds, wf_verdict = run_wf_3fold_with_cap(
                MAX_POSITIONS, DIRECTION_CAP, sched,
                opens_720, highs_720, lows_720,
                overlap_bar, n_720, TIMEOUT_T864)

            is_res = simulate_hedge_with_cap(
                MAX_POSITIONS, DIRECTION_CAP, sched,
                opens_720, highs_720, lows_720,
                overlap_bar, n_720, TIMEOUT_T864)
            is_port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in is_res]
            is_stats = calc_stats(is_port)
            is_ratio = pnl_mdd_ratio(is_stats['cmp_pnl'], is_stats['cmp_mdd'])

            elapsed = time.time() - t0
            print(f"    lo={c_lo:.1f} hi={c_hi:.1f}: {len(scan_results):>3} pats, "
                  f"PnL={is_stats['cmp_pnl']:>+8.1f}%, MDD={is_stats['cmp_mdd']:>6.1f}%, "
                  f"PnL/MDD={is_ratio:>7.2f}, WF={wf_verdict:>12} ({elapsed:.0f}s)")

            stage_b_results.append({
                'clamp_lo': c_lo, 'clamp_hi': c_hi,
                'n_patterns': len(scan_results),
                'is_pnl': is_stats['cmp_pnl'], 'is_mdd': is_stats['cmp_mdd'],
                'pnl_mdd': is_ratio,
                'wf_verdict': wf_verdict,
                'wf_folds': wf_folds,
            })

    # Find best overall
    passing_b = [r for r in stage_b_results if 'PASS (3/3)' in r['wf_verdict']]
    if not passing_b:
        passing_b = stage_b_results

    best_b = max(passing_b, key=lambda r: r['pnl_mdd']) if passing_b else None

    # Compare best_b vs default (best_a with default clamps)
    default_ratio = best_a['pnl_mdd']
    best_ratio = best_b['pnl_mdd'] if best_b else 0

    improvement = ((best_ratio - default_ratio) / default_ratio * 100
                   if default_ratio > 0 else 0)
    go = improvement >= 10.0  # 10% improvement threshold
    verdict = "GO" if go else "STOP"
    reason = (f"Best params PnL/MDD={best_ratio:.2f} vs default={default_ratio:.2f} "
              f"({improvement:+.1f}%)")

    if best_b:
        print(f"\n  Best Stage B: lo={best_b['clamp_lo']} hi={best_b['clamp_hi']} "
              f"(PnL/MDD={best_b['pnl_mdd']}, WF={best_b['wf_verdict']})")
    print(f"\n  Phase 3 verdict: {verdict} ({reason})")

    best_params = {
        'period': best_period, 'window': best_window,
        'clamp_lo': best_b['clamp_lo'] if best_b else CLAMP_LO,
        'clamp_hi': best_b['clamp_hi'] if best_b else CLAMP_HI,
    }

    return {
        'stage_a': stage_a_results,
        'stage_b': stage_b_results,
        'best_a': best_a,
        'best_b': best_b,
        'best_params': best_params,
        'default_ratio': default_ratio,
        'best_ratio': best_ratio,
        'improvement_pct': round(improvement, 1),
        'verdict': verdict,
        'reason': reason,
    }


# ===================================================================
# PHASE 4: Production Candidate Evaluation
# ===================================================================

def phase4_production_candidate(phase1_data, phase2_data, phase3_data):
    """Evaluate results and determine production action."""
    print("\n" + "=" * 110)
    print("  PHASE 4: Production Candidate Evaluation")
    print("=" * 110)

    verdicts = {
        'phase1': phase1_data['verdict'],
        'phase2': phase2_data['verdict'],
        'phase3': phase3_data['verdict'] if phase3_data else 'SKIPPED',
    }

    print(f"\n  Phase 1 (Scanner Comparison): {verdicts['phase1']}")
    print(f"  Phase 2 (Portfolio WF):       {verdicts['phase2']}")
    print(f"  Phase 3 (Parameter Sweep):    {verdicts['phase3']}")

    # Determine action
    if verdicts['phase2'] == 'GO':
        if verdicts['phase3'] == 'GO':
            action = 'A_SCANNER_REPLACE_OPTIMIZED'
            desc = 'Replace scanner with ATR-integrated mode + optimized params'
        else:
            action = 'B_SCANNER_REPLACE_DEFAULT'
            desc = 'Replace scanner with ATR-integrated mode (default params)'
    elif verdicts['phase1'] == 'GO':
        action = 'C_PATTERNS_ONLY'
        desc = 'ATR improves individual patterns but not portfolio → consider pattern-level update'
    else:
        action = 'D_STOP'
        desc = 'No significant improvement → maintain current v1.33.0'

    # ATR pattern count check
    atr_count = phase1_data.get('atr', {}).get('metrics', {}).get('count', 0)
    if atr_count < 15 and action in ('A_SCANNER_REPLACE_OPTIMIZED', 'B_SCANNER_REPLACE_DEFAULT'):
        action = 'C_PATTERNS_ONLY'
        desc += f' (ATR pattern count {atr_count} < 15 minimum)'

    print(f"\n  Recommended action: {action}")
    print(f"  Description: {desc}")

    best_params = phase3_data.get('best_params') if phase3_data else None
    if best_params and action == 'A_SCANNER_REPLACE_OPTIMIZED':
        print(f"\n  Optimal params:")
        print(f"    ATR period: {best_params['period']}")
        print(f"    ATR window: {best_params['window']}")
        print(f"    Clamp: [{best_params['clamp_lo']}, {best_params['clamp_hi']}]")

    return {
        'verdicts': verdicts,
        'action': action,
        'description': desc,
        'best_params': best_params,
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    print("=" * 110)
    print("  ATR SCANNER-PRODUCTION ALIGNMENT STUDY")
    print("  Fixed vs ATR-scaled Scanner → Pattern Selection → Portfolio WF")
    print(f"  Baseline: v1.33.0 (35 patterns, Compact TP/SL, WR 68.1%)")
    print("=" * 110)

    # ------------------------------------------------------------------
    # Load 270d data (IS)
    # ------------------------------------------------------------------
    print("\n--- Loading 270d IS data ---")
    df_270 = load_and_classify(DATA_270D)
    n_270 = len(df_270)
    opens_270 = df_270['open'].values.astype(np.float64)
    highs_270 = df_270['high'].values.astype(np.float64)
    lows_270 = df_270['low'].values.astype(np.float64)
    closes_270 = df_270['close'].values.astype(np.float64)
    # Always use 'rctype' from fresh classification (short codes like "BD", "BU")
    # Never use pre-existing 'candle_type' column (may have full enum repr)
    types_270 = df_270['rctype'].values
    print(f"  270d: {n_270} bars, types sample: {types_270[100:103]}")

    # ------------------------------------------------------------------
    # Load 720d data (WF)
    # ------------------------------------------------------------------
    print("\n--- Loading 720d WF data ---")
    df_720 = load_and_classify(DATA_720D)
    n_720 = len(df_720)
    opens_720 = df_720['open'].values.astype(np.float64)
    highs_720 = df_720['high'].values.astype(np.float64)
    lows_720 = df_720['low'].values.astype(np.float64)
    closes_720 = df_720['close'].values.astype(np.float64)
    types_720 = df_720['rctype'].values
    overlap_bar = find_overlap_bar(df_720)
    avail_bars = n_720 - overlap_bar
    avail_days = avail_bars / BARS_PER_DAY
    print(f"  720d: {n_720} bars, overlap bar: {overlap_bar}, "
          f"avail: {avail_bars} bars ({avail_days:.1f} days)")

    all_results = {}

    # ------------------------------------------------------------------
    # Phase 1: Scanner Comparison
    # ------------------------------------------------------------------
    phase1 = phase1_scanner_comparison(types_270, opens_270, highs_270, lows_270,
                                        closes_270, n_270)
    # Remove full results before saving (too large)
    all_results['phase1'] = {k: v for k, v in phase1.items()
                             if k not in ('fixed_results_full', 'atr_results_full')}

    # ------------------------------------------------------------------
    # Phase 2: Portfolio WF (always run, even if Phase 1 is STOP)
    # ------------------------------------------------------------------
    phase2 = phase2_portfolio_wf(phase1, types_720, opens_720, highs_720, lows_720,
                                  closes_720, n_720, overlap_bar)
    all_results['phase2'] = phase2

    # ------------------------------------------------------------------
    # Phase 3: Parameter Sweep (only if Phase 2 GO)
    # ------------------------------------------------------------------
    phase3 = None
    if phase2['verdict'] == 'GO':
        phase3 = phase3_param_sweep(
            types_270, opens_270, highs_270, lows_270, closes_270, n_270,
            types_720, opens_720, highs_720, lows_720, closes_720, n_720,
            overlap_bar)
        all_results['phase3'] = phase3
    else:
        print("\n  Phase 3: SKIPPED (Phase 2 not GO)")
        all_results['phase3'] = {'verdict': 'SKIPPED', 'reason': 'phase2_not_go'}

    # ------------------------------------------------------------------
    # Phase 4: Production Candidate
    # ------------------------------------------------------------------
    phase4 = phase4_production_candidate(phase1, phase2, phase3)
    all_results['phase4'] = phase4

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    all_results['metadata'] = {
        'study': 'atr_scanner_alignment',
        'baseline': 'v1.33.0 (35 patterns, Compact TP/SL)',
        'data_270d': os.path.basename(DATA_270D),
        'data_720d': os.path.basename(DATA_720D),
        'elapsed_seconds': round(elapsed, 1),
        'max_positions': MAX_POSITIONS,
        'direction_cap': DIRECTION_CAP,
        'timeout_bars': TIMEOUT_T864,
        'quality_filters': {
            'edge_threshold': EDGE_THRESHOLD,
            'wr_threshold': WR_THRESHOLD,
            'sl_min': SL_MIN,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
            'wr_excess_min': WR_EXCESS_MIN,
        },
        'compact_caps': {'tp_max': COMPACT_TP_MAX, 'sl_max': COMPACT_SL_MAX},
    }

    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n  Final action: {phase4['action']}")
    print(f"  Description:  {phase4['description']}")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {OUTPUT_FILE}")

    return all_results


if __name__ == '__main__':
    main()
