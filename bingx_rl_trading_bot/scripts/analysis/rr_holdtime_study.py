#!/usr/bin/env python3
"""
R:R & Hold Time Study — Investigating Live Trading Gap

Live data (209 trades): WR 54.5%, avg win +4.17%, avg loss -6.70%, R:R=0.622
  breakeven WR = 61.6%, actual WR = 54.5% => -7.1pp below breakeven
  6-20h holds are net negative. Current timeout = 288 bars (24h).

Scenarios tested:
  A) Baseline (current production)
  B) SL tightening: mult [0.7, 0.8, 0.9]
  C) TP expansion: mult [1.1, 1.2, 1.3]
  D) Timeout reduction: [144, 192, 240, 288] bars
  E) Mid-hold exit: at 50% of timeout, if unrealized PnL < -X%, close early
  F) Combined best: top SL x TP x Timeout combo

Protocol:
  - N-pos (9 slots, dir_cap 7, agg_risk 8/15, cascade 85%)
  - Compound sizing, fee 0.10% (0.05x2), slippage 0.02%, leverage 3x
  - Expanding window WF (3-fold), MC (3 seeds, p<0.01)
  - Pattern selection: edge>=18pp, WR>=60%, SL>=1.0%, MC<0.01, min_trades>=25
  - Neutral window discovery, ATR-scaled TP/SL

Output: results/rr_holdtime_study.json
"""

import os
import sys
import json
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# Import scanner functions (reuse production-grade backtest engine)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))
from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    mc_test, compute_atr_ratio, compute_ema_slope,
    portfolio_npos, calc_stats_compound,
    scan_universe_range, _check_exit_npos,
    FEE_PCT, LEVERAGE, MAX_BARS, SLIPPAGE_BUFFER,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_EDGE_THRESHOLD, DEFAULT_MC_THRESHOLD, DEFAULT_MIN_TRADES,
    MC_SEEDS, TIMEOUT_BARS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('rr_holdtime_study')

# ============================================================
# Constants
# ============================================================
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'rr_holdtime_study.json')

EDGE_THRESHOLD = 18.0
MC_THRESHOLD = 0.01
MIN_TRADES = 25
WF_FOLDS = 3

# Scenario parameters
SL_MULTS = [0.7, 0.8, 0.9]
TP_MULTS = [1.1, 1.2, 1.3]
TIMEOUT_BARS_LIST = [144, 192, 240, 288]
MID_HOLD_THRESHOLDS = [0.5, 1.0, 1.5]  # close if unrealized < -X%

NPOS_KWARGS_BASE = dict(
    n_slots=DEFAULT_N_SLOTS,
    direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=0.3,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
    agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
    momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO,
    clamp_hi=DEFAULT_ATR_CLAMP_HI,
    cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
)


# ============================================================
# Modified portfolio_npos with timeout/mid-hold overrides
# ============================================================
def portfolio_npos_modified(signal_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, start_bar, end_bar,
                            timeout_bars_override=TIMEOUT_BARS,
                            mid_hold_pct=None, mid_hold_threshold=None,
                            **kwargs):
    """N-pos portfolio sim with configurable timeout and mid-hold exit.

    mid_hold_pct: fraction of timeout at which to check unrealized PnL (e.g. 0.5)
    mid_hold_threshold: if unrealized PnL < -threshold%, close position (counted as loss)
    """
    n_slots = kwargs.get('n_slots', DEFAULT_N_SLOTS)
    direction_cap = kwargs.get('direction_cap', DEFAULT_DIRECTION_CAP)
    regime_mult = kwargs.get('regime_mult', 0.3)
    agg_risk_counter = kwargs.get('agg_risk_counter', DEFAULT_AGG_RISK_COUNTER)
    agg_risk_with = kwargs.get('agg_risk_with', DEFAULT_AGG_RISK_WITH)
    momentum_lookback = kwargs.get('momentum_lookback', DEFAULT_MOMENTUM_LOOKBACK)
    momentum_threshold = kwargs.get('momentum_threshold', DEFAULT_MOMENTUM_THRESHOLD)
    momentum_cooldown = kwargs.get('momentum_cooldown', DEFAULT_MOMENTUM_COOLDOWN)
    clamp_lo = kwargs.get('clamp_lo', DEFAULT_ATR_CLAMP_LO)
    clamp_hi = kwargs.get('clamp_hi', DEFAULT_ATR_CLAMP_HI)
    cascade_tighten_pct = kwargs.get('cascade_tighten_pct', DEFAULT_CASCADE_TIGHTEN_PCT)

    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0

    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0, 'max_pos': 0}
    mid_hold_exits = 0

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    # Mid-hold check bar (relative to entry)
    mid_hold_bar = int(timeout_bars_override * mid_hold_pct) if mid_hold_pct is not None else None

    for bar in range(start_bar, end_bar):
        # 1. Check exits (TP/SL/timeout)
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                      atr_ratio, fee, clamp_lo, clamp_hi,
                                      timeout_bars_override)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1

        # Mid-hold exit check for positions not already closed
        if mid_hold_bar is not None and mid_hold_threshold is not None:
            for pos in positions:
                if pos['slot'] in closed_slots:
                    continue
                entry_bar = pos['entry_bar']
                hold_bars = bar - entry_bar
                if hold_bars != mid_hold_bar:
                    continue
                # Check unrealized PnL at close price
                entry_price = opens[entry_bar]
                if entry_price <= 0:
                    continue
                current_price = closes[bar] if bar < len(closes) else opens[bar]
                if pos['direction'] == 'LONG':
                    unrealized = (current_price / entry_price - 1) * 100 * LEVERAGE
                else:
                    unrealized = (1 - current_price / entry_price) * 100 * LEVERAGE
                unrealized -= fee

                if unrealized < -mid_hold_threshold:
                    sm = pos.get('size_mult', 1.0)
                    pnl_portfolio = unrealized * (size_pct / 100) * sm
                    trades.append({
                        'entry_bar': entry_bar, 'exit_bar': bar,
                        'pnl_slot': unrealized, 'reason': 'MID_HOLD_EXIT',
                        'pattern': pos['pattern'], 'direction': pos['direction'],
                        'size_mult': sm, 'pnl_portfolio': pnl_portfolio,
                        'drop': False,
                    })
                    closed_slots.append(pos['slot'])
                    bar_pnl_sum += pnl_portfolio
                    mid_hold_exits += 1

        # Cascade SL tightening after SL exits
        if cascade_tighten_pct > 0 and bar_sl_count > 0:
            keep_ratio = 1.0 - cascade_tighten_pct / 100.0
            sl_directions = set()
            for t in trades[len(trades) - len(closed_slots):]:
                if t.get('reason') == 'SL':
                    sl_directions.add(t['direction'])
            for sl_dir in sl_directions:
                for pos in positions:
                    if pos['slot'] in closed_slots:
                        continue
                    if pos['direction'] != sl_dir:
                        continue
                    sig = pos['signal_bar']
                    if (atr_ratio is not None and sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[sig])):
                        r = max(clamp_lo, min(clamp_hi, atr_ratio[sig]))
                    else:
                        r = 1.0
                    cur_eff_sl = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard
        if momentum_lookback > 0 and momentum_threshold > 0 and bar >= momentum_lookback:
            price_now = closes[bar]
            price_ago = closes[bar - momentum_lookback]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > momentum_threshold:
                    momentum_pause_until['SHORT'] = bar + momentum_cooldown
                elif pct_change < -momentum_threshold:
                    momentum_pause_until['LONG'] = bar + momentum_cooldown

        # 2. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= n_slots:
                total_blocked['max_pos'] += 1
                continue

            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                total_blocked['dir_cap'] += 1
                continue

            if any(p['pattern'] == pat for p in positions):
                total_blocked['dup_pat'] += 1
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            if momentum_lookback > 0 and bar < momentum_pause_until.get(direction, -1):
                total_blocked['momentum'] += 1
                continue

            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            if agg_risk_counter > 0 or agg_risk_with > 0:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                cap_pct = agg_risk_counter if is_counter else agg_risk_with

                existing_exposure = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_sl = p['sl_pct']
                        p_sig = p['signal_bar']
                        if (atr_ratio is not None and p_sig < len(atr_ratio)
                                and not np.isnan(atr_ratio[p_sig])):
                            p_r = max(clamp_lo, min(clamp_hi, atr_ratio[p_sig]))
                        else:
                            p_r = 1.0
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / n_slots) * LEVERAGE * p_sm

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
                new_eff_sl = sl_pct * new_r
                new_exposure = new_eff_sl * (1.0 / n_slots) * LEVERAGE * sm

                if existing_exposure + new_exposure > cap_pct:
                    total_blocked['agg_risk'] += 1
                    continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
            })

        if len(positions) > max_sim_positions:
            max_sim_positions = len(positions)

    # Force-close remaining
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
        })

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'blocked': total_blocked,
        'mid_hold_exits': mid_hold_exits,
    }
    return trades, stats


# ============================================================
# Metrics helpers
# ============================================================
def compute_rr_metrics(trades):
    """Compute R:R metrics from trade list."""
    if not trades:
        return {'avg_win': 0, 'avg_loss': 0, 'rr': 0, 'breakeven_wr': 100}
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [t['pnl_slot'] for t in trades if t['pnl_slot'] <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
    breakeven_wr = (1 / (1 + rr)) * 100 if rr > 0 else 100
    return {
        'avg_win': round(float(avg_win), 3),
        'avg_loss': round(float(avg_loss), 3),
        'rr': round(float(rr), 3),
        'breakeven_wr': round(float(breakeven_wr), 1),
    }


def compute_hold_time_stats(trades):
    """Compute hold time distribution by bucket."""
    if not trades:
        return {}
    buckets = {}
    for t in trades:
        h = t['exit_bar'] - t['entry_bar']
        hours = (h * 5) / 60
        if hours <= 2:
            bucket = '0-2h'
        elif hours <= 6:
            bucket = '2-6h'
        elif hours <= 12:
            bucket = '6-12h'
        elif hours <= 20:
            bucket = '12-20h'
        else:
            bucket = '20-24h'
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(t['pnl_slot'])

    result = {}
    for bucket in sorted(buckets.keys()):
        arr = buckets[bucket]
        result[bucket] = {
            'count': len(arr),
            'avg_pnl': round(float(np.mean(arr)), 3),
            'sum_pnl': round(float(np.sum(arr)), 2),
            'wr': round(len([x for x in arr if x > 0]) / len(arr) * 100, 1),
        }
    return result


# ============================================================
# WF with modified portfolio sim
# ============================================================
def expanding_wf_modified(signal_index, opens, highs, lows, closes, n_bars,
                          n_folds, timeout_bars_override=TIMEOUT_BARS,
                          mid_hold_pct=None, mid_hold_threshold=None,
                          tp_mult=1.0, sl_mult=1.0):
    """Expanding window WF with modified TP/SL/timeout/mid-hold.

    IS pattern selection uses standard scan_universe_range (mae_mfe mode).
    OOS evaluation uses portfolio_npos_modified with overrides applied
    to each IS-selected pattern's TP/SL.
    """
    seg_size = n_bars // (n_folds + 1)
    folds = []

    for fold in range(n_folds):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        logger.info(f"  WF Fold {fold+1}/{n_folds}: IS=[0,{is_end}), OOS=[{oos_start},{oos_end})")

        # ATR on IS only (prevent look-ahead)
        fold_atr = compute_atr_ratio(
            highs[:is_end], lows[:is_end], closes[:is_end],
            DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)

        # IS pattern selection — standard, no TP/SL modification
        is_patterns = scan_universe_range(
            signal_index, opens, highs, lows, n_bars,
            bar_start=0, bar_end=is_end, mode='mae_mfe',
            min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
            mc_threshold=MC_THRESHOLD, max_baseline_wr=70.0,
            atr_ratio=fold_atr, clamp_lo=DEFAULT_ATR_CLAMP_LO,
            clamp_hi=DEFAULT_ATR_CLAMP_HI,
        )

        n_long = sum(1 for r in is_patterns if r['direction'] == 'LONG')
        n_short = sum(1 for r in is_patterns if r['direction'] == 'SHORT')

        # OOS: ATR on data up to oos_end
        oos_atr = compute_atr_ratio(
            highs[:oos_end], lows[:oos_end], closes[:oos_end],
            DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
        oos_ema = compute_ema_slope(closes[:oos_end])

        # Build signal tuples with modified TP/SL
        oos_signal_tuples = []
        for r in is_patterns:
            mod_tp = r['tp'] * tp_mult
            mod_sl = max(1.0, r['sl'] * sl_mult)  # SL >= 1.0% floor
            for s in signal_index[r['pattern']]:
                if oos_start <= s < oos_end:
                    oos_signal_tuples.append(
                        (s, r['pattern'], r['direction'], mod_tp, mod_sl))

        oos_trades, oos_npos_stats = portfolio_npos_modified(
            oos_signal_tuples, opens, highs, lows, closes, oos_end,
            oos_atr, oos_ema, oos_start, oos_end,
            timeout_bars_override=timeout_bars_override,
            mid_hold_pct=mid_hold_pct,
            mid_hold_threshold=mid_hold_threshold,
            **NPOS_KWARGS_BASE,
        )

        oos_stats = calc_stats_compound(oos_trades)
        rr_stats = compute_rr_metrics(oos_trades)
        hold_stats = compute_hold_time_stats(oos_trades)

        fold_result = {
            'fold': fold + 1,
            'is_bars': is_end,
            'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_patterns),
            'is_long': n_long, 'is_short': n_short,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_mdd': oos_stats['mdd'],
            'oos_pnl_mdd': oos_stats.get('pnl_mdd', 0),
            'oos_positive': oos_stats['pnl'] > 0,
            'oos_rr': rr_stats,
            'oos_hold_time': hold_stats,
            'oos_blocked': oos_npos_stats.get('blocked', {}),
            'mid_hold_exits': oos_npos_stats.get('mid_hold_exits', 0),
        }
        folds.append(fold_result)
        logger.info(f"    IS: {len(is_patterns)} pat ({n_long}L+{n_short}S) | "
                     f"OOS: {oos_stats['trades']} tr, WR {oos_stats['wr']}%, "
                     f"PnL {oos_stats['pnl']}%, R:R {rr_stats['rr']}")

    positive_folds = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_pnl'] for f in folds), 2)
    total_oos_trades = sum(f['oos_trades'] for f in folds)
    folds_with_trades = [f for f in folds if f['oos_trades'] > 0]
    avg_wr = round(np.mean([f['oos_wr'] for f in folds_with_trades]), 1) if folds_with_trades else 0
    avg_rr = round(np.mean([f['oos_rr']['rr'] for f in folds_with_trades]), 3) if folds_with_trades else 0

    return {
        'n_folds': n_folds,
        'positive_folds': positive_folds,
        'total_oos_pnl': total_oos_pnl,
        'total_oos_trades': total_oos_trades,
        'avg_oos_wr': avg_wr,
        'avg_oos_rr': float(avg_rr),
        'folds': folds,
        'wf_pass': positive_folds == n_folds,
    }


def run_is_npos(signal_index, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, start_bar, end_bar,
                timeout_bars_override=TIMEOUT_BARS,
                mid_hold_pct=None, mid_hold_threshold=None,
                tp_mult=1.0, sl_mult=1.0, patterns=None):
    """Run full-IS N-pos evaluation with given parameters."""
    signal_tuples = []
    for r in patterns:
        mod_tp = r['tp'] * tp_mult
        mod_sl = max(1.0, r['sl'] * sl_mult)
        for s in signal_index[r['pattern']]:
            if start_bar <= s < end_bar:
                signal_tuples.append(
                    (s, r['pattern'], r['direction'], mod_tp, mod_sl))

    trades, npos_stats = portfolio_npos_modified(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        timeout_bars_override=timeout_bars_override,
        mid_hold_pct=mid_hold_pct,
        mid_hold_threshold=mid_hold_threshold,
        **NPOS_KWARGS_BASE,
    )
    stats = calc_stats_compound(trades)
    rr_stats = compute_rr_metrics(trades)
    hold_stats = compute_hold_time_stats(trades)
    return {
        **stats,
        'rr': rr_stats,
        'hold_time': hold_stats,
        'blocked': npos_stats.get('blocked', {}),
        'mid_hold_exits': npos_stats.get('mid_hold_exits', 0),
    }


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("R:R & Hold Time Study — Live Trading Gap Investigation")
    logger.info("=" * 70)

    # 1. Load data
    logger.info("Loading data...")
    df = load_and_classify(DATA_FILE)
    n_total = len(df)
    logger.info(f"Total bars: {n_total}")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values

    # 2. Neutral window
    logger.info("Finding neutral window...")
    nw = find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=288)
    if nw is None:
        logger.warning("No neutral window found, using full data")
        start_bar, end_bar = 0, n_total
    else:
        start_bar, end_bar = nw
        nw_days = (end_bar - start_bar) / 288
        logger.info(f"Neutral window: bars [{start_bar}, {end_bar}), {nw_days:.0f} days")

    n_bars = end_bar

    # 3. Build signal index on neutral-sliced data
    types_list = list(types[:n_bars])
    signal_index = build_signal_index(types_list, n_bars)

    # 4. ATR and EMA on full neutral range
    atr_ratio = compute_atr_ratio(highs[:n_bars], lows[:n_bars], closes[:n_bars],
                                   DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
    ema_slope = compute_ema_slope(closes[:n_bars])

    # 5. IS pattern selection (full neutral window)
    logger.info("Running IS pattern selection (mae_mfe mode)...")
    is_patterns = scan_universe_range(
        signal_index, opens, highs, lows, n_bars,
        bar_start=start_bar, bar_end=end_bar, mode='mae_mfe',
        min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
        mc_threshold=MC_THRESHOLD, max_baseline_wr=70.0,
        atr_ratio=atr_ratio, clamp_lo=DEFAULT_ATR_CLAMP_LO,
        clamp_hi=DEFAULT_ATR_CLAMP_HI,
    )
    n_long = sum(1 for r in is_patterns if r['direction'] == 'LONG')
    n_short = sum(1 for r in is_patterns if r['direction'] == 'SHORT')
    logger.info(f"IS patterns: {len(is_patterns)} ({n_long}L + {n_short}S)")

    results = {
        'metadata': {
            'script': 'rr_holdtime_study.py',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_file': DATA_FILE,
            'total_bars': n_total,
            'neutral_window': [start_bar, end_bar] if nw else None,
            'neutral_days': round((end_bar - start_bar) / 288, 1) if nw else None,
            'is_patterns': len(is_patterns),
            'is_long': n_long, 'is_short': n_short,
            'wf_folds': WF_FOLDS,
            'live_context': {
                'trades': 209, 'wr': 54.5, 'avg_win': 4.17, 'avg_loss': -6.70,
                'rr': 0.622, 'breakeven_wr': 61.6,
            },
            'protocol': {
                'entry': 'next-bar open',
                'exit': 'intrabar high/low (distance-based)',
                'sizing': 'compound',
                'fee_pct': FEE_PCT,
                'slippage_pct': SLIPPAGE_BUFFER,
                'leverage': LEVERAGE,
                'n_slots': DEFAULT_N_SLOTS,
                'direction_cap': DEFAULT_DIRECTION_CAP,
                'agg_risk': f'{DEFAULT_AGG_RISK_COUNTER}/{DEFAULT_AGG_RISK_WITH}',
                'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
                'timeout_bars': TIMEOUT_BARS,
            },
        },
        'scenarios': {},
    }

    # ============================================================
    # A) BASELINE — current production settings
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("A) BASELINE (current production)")
    logger.info("=" * 50)

    baseline_is = run_is_npos(
        signal_index, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        patterns=is_patterns,
    )
    logger.info(f"  IS: {baseline_is['trades']} trades, WR {baseline_is['wr']}%, "
                f"PnL {baseline_is['pnl']}%, MDD {baseline_is['mdd']}%, "
                f"R:R {baseline_is['rr']['rr']}")

    baseline_wf = expanding_wf_modified(
        signal_index, opens, highs, lows, closes, n_bars, WF_FOLDS)

    results['scenarios']['A_baseline'] = {
        'description': 'Current production (TP/SL unchanged, timeout=288)',
        'params': {'tp_mult': 1.0, 'sl_mult': 1.0, 'timeout_bars': 288},
        'is': baseline_is,
        'wf': baseline_wf,
    }

    # ============================================================
    # B) SL Tightening
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("B) SL Tightening")
    logger.info("=" * 50)

    for sl_m in SL_MULTS:
        key = f'B_sl_{int(sl_m*100)}'
        logger.info(f"\n--- {key}: SL * {sl_m} ---")

        is_result = run_is_npos(
            signal_index, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            sl_mult=sl_m, patterns=is_patterns,
        )
        logger.info(f"  IS: {is_result['trades']} tr, WR {is_result['wr']}%, "
                    f"PnL {is_result['pnl']}%, R:R {is_result['rr']['rr']}")

        wf_result = expanding_wf_modified(
            signal_index, opens, highs, lows, closes, n_bars, WF_FOLDS,
            sl_mult=sl_m)

        results['scenarios'][key] = {
            'description': f'SL tightened to {int(sl_m*100)}% of original',
            'params': {'tp_mult': 1.0, 'sl_mult': sl_m, 'timeout_bars': 288},
            'is': is_result,
            'wf': wf_result,
        }

    # ============================================================
    # C) TP Expansion
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("C) TP Expansion")
    logger.info("=" * 50)

    for tp_m in TP_MULTS:
        key = f'C_tp_{int(tp_m*100)}'
        logger.info(f"\n--- {key}: TP * {tp_m} ---")

        is_result = run_is_npos(
            signal_index, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            tp_mult=tp_m, patterns=is_patterns,
        )
        logger.info(f"  IS: {is_result['trades']} tr, WR {is_result['wr']}%, "
                    f"PnL {is_result['pnl']}%, R:R {is_result['rr']['rr']}")

        wf_result = expanding_wf_modified(
            signal_index, opens, highs, lows, closes, n_bars, WF_FOLDS,
            tp_mult=tp_m)

        results['scenarios'][key] = {
            'description': f'TP expanded to {int(tp_m*100)}% of original',
            'params': {'tp_mult': tp_m, 'sl_mult': 1.0, 'timeout_bars': 288},
            'is': is_result,
            'wf': wf_result,
        }

    # ============================================================
    # D) Timeout Reduction
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("D) Timeout Reduction")
    logger.info("=" * 50)

    for to_bars in TIMEOUT_BARS_LIST:
        key = f'D_timeout_{to_bars}'
        hours = to_bars * 5 / 60
        logger.info(f"\n--- {key}: timeout={to_bars} bars ({hours:.0f}h) ---")

        is_result = run_is_npos(
            signal_index, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            timeout_bars_override=to_bars, patterns=is_patterns,
        )
        logger.info(f"  IS: {is_result['trades']} tr, WR {is_result['wr']}%, "
                    f"PnL {is_result['pnl']}%, R:R {is_result['rr']['rr']}")

        wf_result = expanding_wf_modified(
            signal_index, opens, highs, lows, closes, n_bars, WF_FOLDS,
            timeout_bars_override=to_bars)

        results['scenarios'][key] = {
            'description': f'Timeout = {to_bars} bars ({hours:.0f}h)',
            'params': {'tp_mult': 1.0, 'sl_mult': 1.0, 'timeout_bars': to_bars},
            'is': is_result,
            'wf': wf_result,
        }

    # ============================================================
    # E) Mid-Hold Exit
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("E) Mid-Hold Exit (at 50% of timeout)")
    logger.info("=" * 50)

    for mh_thresh in MID_HOLD_THRESHOLDS:
        key = f'E_midhold_{int(mh_thresh*10)}'
        logger.info(f"\n--- {key}: close if unrealized < -{mh_thresh}% at 50% timeout ---")

        is_result = run_is_npos(
            signal_index, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            mid_hold_pct=0.5, mid_hold_threshold=mh_thresh,
            patterns=is_patterns,
        )
        logger.info(f"  IS: {is_result['trades']} tr, WR {is_result['wr']}%, "
                    f"PnL {is_result['pnl']}%, mid-hold exits: {is_result['mid_hold_exits']}, "
                    f"R:R {is_result['rr']['rr']}")

        wf_result = expanding_wf_modified(
            signal_index, opens, highs, lows, closes, n_bars, WF_FOLDS,
            mid_hold_pct=0.5, mid_hold_threshold=mh_thresh)

        results['scenarios'][key] = {
            'description': f'Mid-hold exit: close if PnL < -{mh_thresh}% at bar {int(288*0.5)} (50% of 288)',
            'params': {'tp_mult': 1.0, 'sl_mult': 1.0, 'timeout_bars': 288,
                       'mid_hold_pct': 0.5, 'mid_hold_threshold': mh_thresh},
            'is': is_result,
            'wf': wf_result,
        }

    # ============================================================
    # F) Combined Best — best from each category, combine
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("F) Combined Best (top SL x TP x Timeout)")
    logger.info("=" * 50)

    def pick_best_is(prefix):
        """Pick scenario with best IS PnL/MDD from given prefix group."""
        best_key, best_val = None, -999
        for k, v in results['scenarios'].items():
            if k.startswith(prefix) and v['is']['pnl_mdd'] > best_val:
                best_key = k
                best_val = v['is']['pnl_mdd']
        return best_key

    best_sl_key = pick_best_is('B_sl_')
    best_tp_key = pick_best_is('C_tp_')
    best_to_key = pick_best_is('D_timeout_')
    best_mh_key = pick_best_is('E_midhold_')

    best_sl_mult = results['scenarios'][best_sl_key]['params']['sl_mult'] if best_sl_key else 1.0
    best_tp_mult = results['scenarios'][best_tp_key]['params']['tp_mult'] if best_tp_key else 1.0
    best_to_bars = results['scenarios'][best_to_key]['params']['timeout_bars'] if best_to_key else 288
    best_mh_thresh = results['scenarios'][best_mh_key]['params'].get('mid_hold_threshold') if best_mh_key else None

    combos = []
    # SL + TP
    if best_sl_key and best_tp_key:
        combos.append(('F1_sl_tp', best_sl_mult, best_tp_mult, 288, None, None))
    # SL + Timeout
    if best_sl_key and best_to_key and best_to_bars != 288:
        combos.append(('F2_sl_timeout', best_sl_mult, 1.0, best_to_bars, None, None))
    # TP + Timeout
    if best_tp_key and best_to_key and best_to_bars != 288:
        combos.append(('F3_tp_timeout', 1.0, best_tp_mult, best_to_bars, None, None))
    # All three
    if best_sl_key and best_tp_key and best_to_key:
        combos.append(('F4_all', best_sl_mult, best_tp_mult, best_to_bars, None, None))
    # SL + mid-hold
    if best_sl_key and best_mh_key and best_mh_thresh is not None:
        combos.append(('F5_sl_midhold', best_sl_mult, 1.0, 288, 0.5, best_mh_thresh))

    for combo_key, sl_m, tp_m, to_b, mh_pct, mh_th in combos:
        hours = to_b * 5 / 60
        desc = f'SL*{sl_m}, TP*{tp_m}, TO={to_b}({hours:.0f}h)'
        if mh_pct is not None:
            desc += f', mid-hold -{mh_th}%'
        logger.info(f"\n--- {combo_key}: {desc} ---")

        is_result = run_is_npos(
            signal_index, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            tp_mult=tp_m, sl_mult=sl_m, timeout_bars_override=to_b,
            mid_hold_pct=mh_pct, mid_hold_threshold=mh_th,
            patterns=is_patterns,
        )
        logger.info(f"  IS: {is_result['trades']} tr, WR {is_result['wr']}%, "
                    f"PnL {is_result['pnl']}%, PnL/MDD {is_result['pnl_mdd']}, "
                    f"R:R {is_result['rr']['rr']}")

        wf_result = expanding_wf_modified(
            signal_index, opens, highs, lows, closes, n_bars, WF_FOLDS,
            tp_mult=tp_m, sl_mult=sl_m, timeout_bars_override=to_b,
            mid_hold_pct=mh_pct, mid_hold_threshold=mh_th)

        results['scenarios'][combo_key] = {
            'description': desc,
            'params': {'tp_mult': tp_m, 'sl_mult': sl_m, 'timeout_bars': to_b,
                       'mid_hold_pct': mh_pct, 'mid_hold_threshold': mh_th},
            'is': is_result,
            'wf': wf_result,
        }

    # ============================================================
    # Summary
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    summary_rows = []
    for key, sc in results['scenarios'].items():
        wf = sc['wf']
        is_data = sc['is']
        summary_rows.append({
            'scenario': key,
            'is_trades': is_data['trades'],
            'is_wr': is_data['wr'],
            'is_pnl': is_data['pnl'],
            'is_mdd': is_data['mdd'],
            'is_pnl_mdd': is_data['pnl_mdd'],
            'is_rr': is_data['rr']['rr'],
            'is_breakeven_wr': is_data['rr']['breakeven_wr'],
            'wf_pass': wf['wf_pass'],
            'wf_positive_folds': wf['positive_folds'],
            'wf_total_oos_pnl': wf['total_oos_pnl'],
            'wf_avg_oos_wr': wf['avg_oos_wr'],
            'wf_avg_oos_rr': wf['avg_oos_rr'],
        })

    results['summary'] = summary_rows

    # Print summary table
    header = (f"{'Scenario':<20} {'IS WR':>6} {'IS PnL':>8} {'IS MDD':>7} {'PnL/MDD':>8} "
              f"{'R:R':>5} {'BkEv':>5} {'WF':>4} {'OOS PnL':>8} {'OOS WR':>7} {'OOS R:R':>7}")
    logger.info(header)
    logger.info("-" * len(header))
    for row in summary_rows:
        wf_status = "PASS" if row['wf_pass'] else "FAIL"
        logger.info(f"{row['scenario']:<20} {row['is_wr']:>5.1f}% {row['is_pnl']:>7.1f}% "
                    f"{row['is_mdd']:>6.1f}% {row['is_pnl_mdd']:>7.1f} "
                    f"{row['is_rr']:>5.3f} {row['is_breakeven_wr']:>4.1f}% "
                    f"{wf_status:>4} {row['wf_total_oos_pnl']:>7.1f}% "
                    f"{row['wf_avg_oos_wr']:>5.1f}% {row['wf_avg_oos_rr']:>6.3f}")

    # Verdict
    baseline_oos = results['scenarios']['A_baseline']['wf']['total_oos_pnl']
    improvements = []
    for key, sc in results['scenarios'].items():
        if key == 'A_baseline':
            continue
        if sc['wf']['wf_pass'] and sc['wf']['total_oos_pnl'] > baseline_oos:
            delta = sc['wf']['total_oos_pnl'] - baseline_oos
            improvements.append((key, delta, sc['wf']['total_oos_pnl']))

    improvements.sort(key=lambda x: -x[1])

    if improvements:
        logger.info(f"\nVERDICT: {len(improvements)} scenario(s) beat baseline WF OOS:")
        for name, delta, oos_pnl in improvements[:5]:
            logger.info(f"  {name}: OOS +{oos_pnl:.1f}% (delta +{delta:.1f}%)")
        results['verdict'] = 'GO'
        results['best_scenario'] = improvements[0][0]
        results['best_delta'] = round(improvements[0][1], 2)
    else:
        logger.info("\nVERDICT: No scenario beats baseline WF OOS. KEEP current.")
        results['verdict'] = 'KEEP_BASELINE'
        results['best_scenario'] = 'A_baseline'
        results['best_delta'] = 0

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {OUTPUT_FILE}")

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
