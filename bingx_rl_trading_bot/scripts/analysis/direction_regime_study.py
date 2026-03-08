#!/usr/bin/env python3
"""
Direction Regime Study — Can a trend filter reduce counter-regime losses?
=========================================================================

Problem: Live trading shows extreme direction asymmetry:
  - Downtrend: LONG 0% WR (-76.77%), SHORT 100% WR (+117.66%)
  - Uptrend:   LONG 64.3% WR (+35.59%), SHORT 41.2% WR (-195.32%)
  Counter-regime direction ALWAYS dominates losses.

Hypotheses (all N-pos simulation):
  H0: Baseline (current production — no trend filter)
  H1: SMA trend gate — LONG only when price > SMA(N), SHORT when price < SMA(N)
      SMA periods: [50, 100, 200, 500] (in 5m bars)
  H2: Momentum trend gate — LONG only when 1h return > 0, SHORT when < 0
      Lookback: [12, 24, 48, 72] bars
  H3: Counter-direction sizing reduction — reduce counter-regime size
      Size mults: [0.25, 0.5, 0.75] with SMA(200) trend detection
  H4: Asymmetric direction cap — cap counter-regime direction at [2, 3, 4]
      Trend detection: SMA(200) crossover
  H5: Best combo from H1-H4

Protocol:
  - Data: data/btc_5m_270days_reclassified.csv
  - Patterns: results/dynamic_patterns.json (131 patterns, 59L+72S)
  - Production classify_candle import
  - LEVERAGE=3, FEE_PCT=0.10, slippage 0.02%
  - N-pos: 9 slots, dir_cap 7, agg_risk 8/15, cascade SL 85%, momentum guard
  - ATR-scaled TP/SL clamp [0.5, 1.5], timeout 288 bars
  - Neutral window discovery
  - WF: 3-fold expanding window
  - MC: 3 seeds (42, 123, 7), p<0.01
  - Random discrimination test for WF-passing hypotheses

Output: results/direction_regime_study.json
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, compute_atr_ratio,
    compute_ema_slope, portfolio_npos, calc_stats_compound,
    find_neutral_window, expanding_window_wf,
    LEVERAGE, FEE_PCT, DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_REGIME_MULT, DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, TIMEOUT_BARS,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_CASCADE_TIGHTEN_PCT,
)

# ============================================================
# CONSTANTS
# ============================================================
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'direction_regime_study.json')
BARS_PER_DAY = 288
WF_FOLDS = 3
EDGE_THRESHOLD = 18.0
MC_THRESHOLD = 0.01
MIN_TRADES = 25
NEUTRAL_TOL = 1.0
MC_SEEDS = [42, 123, 7]
MC_SIMS = 10000
RANDOM_DISCRIM_SETS = 10  # Number of random signal sets for discrimination test


# ============================================================
# DATA LOADING
# ============================================================
def load_patterns():
    """Load current dynamic patterns."""
    with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    patterns_tpsl = data.get('patterns_tpsl') or {}
    long_pats = (data.get('patterns') or {}).get('long') or []
    short_pats = (data.get('patterns') or {}).get('short') or []
    return long_pats, short_pats, patterns_tpsl


def collect_signals(signal_index, long_pats, short_pats, patterns_tpsl,
                    start_bar=0, end_bar=None):
    """Collect signal tuples for portfolio simulation."""
    tuples = []
    default_tp, default_sl = 2.0, 3.0
    for pat in long_pats:
        tpsl = patterns_tpsl.get(pat) or [default_tp, default_sl]
        for s in signal_index.get(pat, []):
            if start_bar <= s < (end_bar if end_bar else float('inf')):
                tuples.append((s, pat, 'LONG', tpsl[0], tpsl[1]))
    for pat in short_pats:
        tpsl = patterns_tpsl.get(pat) or [default_tp, default_sl]
        for s in signal_index.get(pat, []):
            if start_bar <= s < (end_bar if end_bar else float('inf')):
                tuples.append((s, pat, 'SHORT', tpsl[0], tpsl[1]))
    return tuples


def build_npos_kwargs(**overrides):
    """Build N-pos kwargs with production defaults."""
    kw = {
        'n_slots': DEFAULT_N_SLOTS,
        'direction_cap': DEFAULT_DIRECTION_CAP,
        'regime_mult': DEFAULT_REGIME_MULT,
        'agg_risk_counter': DEFAULT_AGG_RISK_COUNTER,
        'agg_risk_with': DEFAULT_AGG_RISK_WITH,
        'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK,
        'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD,
        'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN,
        'clamp_lo': DEFAULT_ATR_CLAMP_LO,
        'clamp_hi': DEFAULT_ATR_CLAMP_HI,
        'timeout_bars': TIMEOUT_BARS,
        'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
    }
    kw.update(overrides)
    return kw


# ============================================================
# TREND FILTER: Modified portfolio_npos with trend gates
# ============================================================
def portfolio_npos_trend_filtered(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        trend_filter=None, trend_params=None,
        **npos_kwargs):
    """
    N-pos portfolio simulator with optional trend filters.

    trend_filter: 'sma_gate', 'momentum_gate', 'sizing_reduction',
                  'asymmetric_cap', or None (baseline)
    trend_params: dict with filter-specific parameters
    """
    if trend_filter is None:
        return portfolio_npos(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar, **npos_kwargs)

    params = trend_params or {}
    n_slots = npos_kwargs.get('n_slots', DEFAULT_N_SLOTS)
    direction_cap = npos_kwargs.get('direction_cap', DEFAULT_DIRECTION_CAP)
    regime_mult = npos_kwargs.get('regime_mult', DEFAULT_REGIME_MULT)
    agg_risk_counter = npos_kwargs.get('agg_risk_counter', DEFAULT_AGG_RISK_COUNTER)
    agg_risk_with = npos_kwargs.get('agg_risk_with', DEFAULT_AGG_RISK_WITH)
    momentum_lookback = npos_kwargs.get('momentum_lookback', DEFAULT_MOMENTUM_LOOKBACK)
    momentum_threshold = npos_kwargs.get('momentum_threshold', DEFAULT_MOMENTUM_THRESHOLD)
    momentum_cooldown = npos_kwargs.get('momentum_cooldown', DEFAULT_MOMENTUM_COOLDOWN)
    clamp_lo = npos_kwargs.get('clamp_lo', DEFAULT_ATR_CLAMP_LO)
    clamp_hi = npos_kwargs.get('clamp_hi', DEFAULT_ATR_CLAMP_HI)
    timeout_bars = npos_kwargs.get('timeout_bars', TIMEOUT_BARS)
    cascade_tighten_pct = npos_kwargs.get('cascade_tighten_pct', DEFAULT_CASCADE_TIGHTEN_PCT)

    # Pre-compute trend indicators
    sma_values = None
    momentum_returns = None

    if trend_filter in ('sma_gate', 'sizing_reduction', 'asymmetric_cap'):
        sma_period = params.get('sma_period', 200)
        sma_values = pd.Series(closes[:end_bar]).rolling(sma_period).mean().values

    if trend_filter == 'momentum_gate':
        mom_lookback = params.get('lookback', 12)
        momentum_returns = np.full(end_bar, 0.0)
        for i in range(mom_lookback, end_bar):
            if closes[i - mom_lookback] > 0:
                momentum_returns[i] = (closes[i] / closes[i - mom_lookback] - 1) * 100

    # Filter signals based on trend
    filtered_tuples = []
    blocked_by_trend = {'LONG': 0, 'SHORT': 0}
    reduced_sizing = {'LONG': 0, 'SHORT': 0}

    for sig_bar, pat, direction, tp_pct, sl_pct in signal_tuples:
        if sig_bar < start_bar or sig_bar >= end_bar:
            continue

        allow = True
        size_mult_override = None

        if trend_filter == 'sma_gate':
            if sma_values is not None and sig_bar < len(sma_values) and not np.isnan(sma_values[sig_bar]):
                sma_val = sma_values[sig_bar]
                price = closes[sig_bar]
                if direction == 'LONG' and price < sma_val:
                    allow = False
                    blocked_by_trend['LONG'] += 1
                elif direction == 'SHORT' and price > sma_val:
                    allow = False
                    blocked_by_trend['SHORT'] += 1

        elif trend_filter == 'momentum_gate':
            if momentum_returns is not None and sig_bar < len(momentum_returns):
                ret = momentum_returns[sig_bar]
                if direction == 'LONG' and ret < 0:
                    allow = False
                    blocked_by_trend['LONG'] += 1
                elif direction == 'SHORT' and ret > 0:
                    allow = False
                    blocked_by_trend['SHORT'] += 1

        elif trend_filter == 'sizing_reduction':
            size_mult_factor = params.get('size_mult', 0.5)
            if sma_values is not None and sig_bar < len(sma_values) and not np.isnan(sma_values[sig_bar]):
                sma_val = sma_values[sig_bar]
                price = closes[sig_bar]
                # Counter-regime: reduce size
                if direction == 'LONG' and price < sma_val:
                    size_mult_override = size_mult_factor
                    reduced_sizing['LONG'] += 1
                elif direction == 'SHORT' and price > sma_val:
                    size_mult_override = size_mult_factor
                    reduced_sizing['SHORT'] += 1

        if allow:
            filtered_tuples.append((sig_bar, pat, direction, tp_pct, sl_pct))

    # For asymmetric_cap: modify direction_cap dynamically
    # We implement this by splitting signals and running with bar-by-bar cap logic
    if trend_filter == 'asymmetric_cap':
        counter_cap = params.get('counter_cap', 3)
        return _portfolio_npos_asymmetric_cap(
            filtered_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            sma_values, counter_cap, direction_cap,
            n_slots, regime_mult,
            agg_risk_counter, agg_risk_with,
            momentum_lookback, momentum_threshold, momentum_cooldown,
            clamp_lo, clamp_hi, timeout_bars, cascade_tighten_pct,
            blocked_by_trend)

    # For sizing_reduction, we need to modify the signal tuples to carry size info
    # We do this by modifying the portfolio_npos call
    if trend_filter == 'sizing_reduction':
        return _portfolio_npos_with_sizing(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            sma_values, params.get('size_mult', 0.5),
            n_slots, direction_cap, regime_mult,
            agg_risk_counter, agg_risk_with,
            momentum_lookback, momentum_threshold, momentum_cooldown,
            clamp_lo, clamp_hi, timeout_bars, cascade_tighten_pct,
            reduced_sizing)

    # For sma_gate and momentum_gate: run standard npos with filtered signals
    trades, raw_stats = portfolio_npos(
        filtered_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        n_slots=n_slots, direction_cap=direction_cap,
        regime_mult=regime_mult,
        agg_risk_counter=agg_risk_counter, agg_risk_with=agg_risk_with,
        momentum_lookback=momentum_lookback,
        momentum_threshold=momentum_threshold,
        momentum_cooldown=momentum_cooldown,
        clamp_lo=clamp_lo, clamp_hi=clamp_hi,
        timeout_bars=timeout_bars,
        cascade_tighten_pct=cascade_tighten_pct)

    raw_stats['blocked_by_trend'] = blocked_by_trend
    return trades, raw_stats


def _portfolio_npos_asymmetric_cap(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        sma_values, counter_cap, default_cap,
        n_slots, regime_mult,
        agg_risk_counter, agg_risk_with,
        momentum_lookback, momentum_threshold, momentum_cooldown,
        clamp_lo, clamp_hi, timeout_bars, cascade_tighten_pct,
        blocked_by_trend):
    """N-pos with asymmetric direction caps based on SMA trend."""
    from scripts.scanner.pattern_scanner import _check_exit_npos, SLIPPAGE_BUFFER

    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0,
                     'dup_pat': 0, 'max_pos': 0, 'trend_cap': 0}
    corr_events = []
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check exits (identical to scanner portfolio_npos)
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                      atr_ratio, fee, clamp_lo, clamp_hi,
                                      timeout_bars)
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

        # Cascade SL tightening
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

        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum)
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct
            corr_events.append((bar, bar_sl_count, loss_pct))

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

        # 2. Determine dynamic direction cap based on SMA trend
        is_uptrend = True  # default
        if sma_values is not None and bar < len(sma_values) and not np.isnan(sma_values[bar]):
            is_uptrend = closes[bar] > sma_values[bar]

        long_cap = default_cap if is_uptrend else counter_cap
        short_cap = counter_cap if is_uptrend else default_cap

        # 3. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= n_slots:
                total_blocked['max_pos'] += 1
                continue

            # Asymmetric direction cap
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            effective_cap = long_cap if direction == 'LONG' else short_cap
            if dir_count >= effective_cap:
                total_blocked['trend_cap'] += 1
                continue

            # Standard direction cap (should not trigger if asymmetric is tighter)
            if dir_count >= default_cap:
                total_blocked['dir_cap'] += 1
                continue

            # Duplicate pattern check
            if any(p['pattern'] == pat for p in positions):
                total_blocked['dup_pat'] += 1
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            # Momentum guard
            if momentum_lookback > 0 and bar < momentum_pause_until.get(direction, -1):
                total_blocked['momentum'] += 1
                continue

            # Regime sizing
            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            # Aggregate risk cap
            if agg_risk_counter > 0 or agg_risk_with > 0:
                is_up = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_up and direction == 'SHORT') or
                              (not is_up and direction == 'LONG'))
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
        'corr_events': len(corr_events),
        'blocked': total_blocked,
        'blocked_by_trend': blocked_by_trend,
    }
    return trades, stats


def _portfolio_npos_with_sizing(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        sma_values, counter_size_mult,
        n_slots, direction_cap, regime_mult,
        agg_risk_counter, agg_risk_with,
        momentum_lookback, momentum_threshold, momentum_cooldown,
        clamp_lo, clamp_hi, timeout_bars, cascade_tighten_pct,
        reduced_sizing):
    """N-pos with counter-regime sizing reduction based on SMA trend."""
    from scripts.scanner.pattern_scanner import _check_exit_npos

    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0,
                     'dup_pat': 0, 'max_pos': 0}
    corr_events = []
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check exits
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                      atr_ratio, fee, clamp_lo, clamp_hi,
                                      timeout_bars)
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

        # Cascade SL tightening
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

        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum)
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct
            corr_events.append((bar, bar_sl_count, loss_pct))

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

            # Regime sizing
            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            # Apply counter-regime sizing reduction
            if sma_values is not None and sig_bar < len(sma_values) and not np.isnan(sma_values[sig_bar]):
                sma_val = sma_values[sig_bar]
                price = closes[sig_bar]
                if direction == 'LONG' and price < sma_val:
                    sm *= counter_size_mult
                elif direction == 'SHORT' and price > sma_val:
                    sm *= counter_size_mult

            # Aggregate risk cap
            if agg_risk_counter > 0 or agg_risk_with > 0:
                is_up = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_up and direction == 'SHORT') or
                              (not is_up and direction == 'LONG'))
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
        'corr_events': len(corr_events),
        'blocked': total_blocked,
        'reduced_sizing': reduced_sizing,
    }
    return trades, stats


# ============================================================
# DIRECTION-SPECIFIC STATS
# ============================================================
def direction_stats(trades):
    """Compute direction-specific WR and PnL from trade list."""
    long_trades = [t for t in trades if t.get('direction') == 'LONG']
    short_trades = [t for t in trades if t.get('direction') == 'SHORT']

    def _stats(tlist):
        if not tlist:
            return {'trades': 0, 'wr': 0.0, 'pnl': 0.0}
        wins = sum(1 for t in tlist if t['pnl_slot'] > 0)
        total_pnl = sum(t['pnl_portfolio'] for t in tlist)
        return {
            'trades': len(tlist),
            'wr': round(wins / len(tlist) * 100, 1),
            'pnl': round(total_pnl, 2),
        }

    return {'LONG': _stats(long_trades), 'SHORT': _stats(short_trades)}


# ============================================================
# RUN SCENARIO
# ============================================================
def run_scenario(name, signal_tuples, opens, highs, lows, closes, n_bars,
                 atr_ratio, ema_slope, start_bar, end_bar,
                 trend_filter=None, trend_params=None, **npos_kwargs):
    """Run a single scenario and return stats with direction breakdown."""
    trades, raw_stats = portfolio_npos_trend_filtered(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        trend_filter=trend_filter, trend_params=trend_params,
        **npos_kwargs)

    stats = calc_stats_compound(trades)
    stats.update(raw_stats)
    stats['direction'] = direction_stats(trades)
    return stats


# ============================================================
# MONTE CARLO SIGN RANDOMIZATION
# ============================================================
def mc_sign_test(trades, seeds=None, n_sims=MC_SIMS):
    """Monte Carlo sign randomization. Returns max p-value across seeds."""
    if seeds is None:
        seeds = MC_SEEDS
    if not trades:
        return 1.0

    pnls = np.array([t['pnl_portfolio'] for t in sorted(trades, key=lambda x: x['entry_bar'])])
    observed = pnls.sum()
    max_p = 0.0

    for seed in seeds:
        rng = np.random.RandomState(seed)
        count_ge = 0
        for _ in range(n_sims):
            signs = rng.choice([-1, 1], size=len(pnls))
            sim_pnl = (pnls * signs).sum()
            if sim_pnl >= observed:
                count_ge += 1
        p = count_ge / n_sims
        if p > max_p:
            max_p = p

    return max_p


# ============================================================
# WALK-FORWARD WITH TREND FILTER
# ============================================================
def wf_with_trend_filter(signal_index, long_pats, short_pats, patterns_tpsl,
                         opens, highs, lows, closes, n_bars,
                         trend_filter, trend_params, n_folds=WF_FOLDS):
    """3-fold expanding window WF with trend filter applied to OOS."""
    seg_size = n_bars // (n_folds + 1)
    folds = []

    for fold in range(n_folds):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        # Compute ATR on IS data only
        fold_atr = compute_atr_ratio(
            highs[:is_end], lows[:is_end], closes[:is_end],
            DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)

        # IS: scan patterns (no trend filter in IS — standard pattern discovery)
        from scripts.scanner.pattern_scanner import scan_universe_range
        is_patterns = scan_universe_range(
            signal_index, opens, highs, lows, n_bars,
            bar_start=0, bar_end=is_end, mode='mae_mfe',
            min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
            mc_threshold=MC_THRESHOLD, max_baseline_wr=70.0,
            atr_ratio=fold_atr, clamp_lo=DEFAULT_ATR_CLAMP_LO,
            clamp_hi=DEFAULT_ATR_CLAMP_HI)

        # OOS: apply trend filter
        oos_atr = compute_atr_ratio(
            highs[:oos_end], lows[:oos_end], closes[:oos_end],
            DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
        oos_ema_slope = compute_ema_slope(closes[:oos_end])

        # Build OOS signal tuples from IS-discovered patterns
        oos_signal_tuples = []
        is_long = []
        is_short = []
        for r in is_patterns:
            direction = r['direction']
            if direction == 'LONG':
                is_long.append(r['pattern'])
            else:
                is_short.append(r['pattern'])
            for s in signal_index[r['pattern']]:
                if oos_start <= s < oos_end:
                    oos_signal_tuples.append(
                        (s, r['pattern'], direction, r['tp'], r['sl']))

        npos_kwargs = build_npos_kwargs()
        trades, raw_stats = portfolio_npos_trend_filtered(
            oos_signal_tuples, opens, highs, lows, closes, oos_end,
            oos_atr, oos_ema_slope, oos_start, oos_end,
            trend_filter=trend_filter, trend_params=trend_params,
            **npos_kwargs)

        oos_stats = calc_stats_compound(trades)
        oos_dir = direction_stats(trades)

        fold_result = {
            'fold': fold + 1,
            'is_bars': is_end,
            'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_patterns),
            'is_long': len(is_long),
            'is_short': len(is_short),
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_mdd': oos_stats['mdd'],
            'oos_pnl_mdd': oos_stats.get('pnl_mdd', 0),
            'oos_positive': oos_stats['pnl'] > 0,
            'oos_direction': oos_dir,
            'blocked_by_trend': raw_stats.get('blocked_by_trend', {}),
        }
        folds.append(fold_result)

    positive_folds = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_pnl'] for f in folds), 2)
    wf_pass = positive_folds == n_folds

    return {
        'folds': folds,
        'positive_folds': positive_folds,
        'total_oos_pnl': total_oos_pnl,
        'wf_pass': wf_pass,
    }


# ============================================================
# RANDOM DISCRIMINATION TEST
# ============================================================
def random_discrimination_test(signal_index, long_pats, short_pats, patterns_tpsl,
                               opens, highs, lows, closes, n_bars,
                               trend_filter, trend_params,
                               n_random=RANDOM_DISCRIM_SETS):
    """Generate random signals at same frequency, test WF pass rate.
    If >50% pass => NON-DISCRIMINATING."""
    # Count real signals
    all_signal_bars = []
    for pat in long_pats + short_pats:
        all_signal_bars.extend(signal_index.get(pat, []))
    real_freq = len(all_signal_bars)

    # Get all pattern names and their TP/SL
    all_pats = long_pats + short_pats
    all_tpsl = []
    for pat in all_pats:
        tpsl = patterns_tpsl.get(pat) or [2.0, 3.0]
        direction = 'LONG' if pat in long_pats else 'SHORT'
        all_tpsl.append((pat, direction, tpsl[0], tpsl[1]))

    passes = 0
    results = []

    for i in range(n_random):
        seed = 1000 + i
        rng = np.random.RandomState(seed)

        # Generate random signal tuples with same frequency
        random_tuples = []
        n_signals = real_freq // len(all_pats) if all_pats else 0
        for pat, direction, tp, sl in all_tpsl:
            bars = sorted(rng.choice(range(200, n_bars - 300), size=min(n_signals, n_bars - 500), replace=False))
            for b in bars:
                random_tuples.append((b, pat, direction, tp, sl))

        # Run WF with random signals
        seg_size = n_bars // (WF_FOLDS + 1)
        fold_results = []
        for fold in range(WF_FOLDS):
            is_end = (fold + 1) * seg_size
            oos_start = is_end
            oos_end = (fold + 2) * seg_size if fold < WF_FOLDS - 1 else n_bars

            oos_atr = compute_atr_ratio(
                highs[:oos_end], lows[:oos_end], closes[:oos_end],
                DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
            oos_ema = compute_ema_slope(closes[:oos_end])

            oos_tuples = [(s, p, d, tp, sl) for s, p, d, tp, sl in random_tuples
                          if oos_start <= s < oos_end]

            npos_kwargs = build_npos_kwargs()
            trades, _ = portfolio_npos_trend_filtered(
                oos_tuples, opens, highs, lows, closes, oos_end,
                oos_atr, oos_ema, oos_start, oos_end,
                trend_filter=trend_filter, trend_params=trend_params,
                **npos_kwargs)

            stats = calc_stats_compound(trades)
            fold_results.append(stats['pnl'] > 0)

        all_pass = all(fold_results)
        if all_pass:
            passes += 1
        results.append({'seed': seed, 'wf_pass': all_pass})

    pass_rate = passes / n_random * 100
    discriminating = pass_rate <= 50

    return {
        'n_random': n_random,
        'passes': passes,
        'pass_rate': round(pass_rate, 1),
        'discriminating': discriminating,
        'results': results,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("DIRECTION REGIME STUDY")
    print("Can a trend filter reduce counter-regime losses?")
    print("=" * 70)

    t_start = time.time()
    all_results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'hypotheses': {}}

    # Load data
    print("\nLoading data...")
    df = load_and_classify(DATA_FILE)

    # Neutral window
    closes_all = df['close'].values
    nw_result = find_neutral_window(closes_all, tol_pct=NEUTRAL_TOL)
    if nw_result is not None:
        ns, ne = nw_result
        drift = (closes_all[ne] / closes_all[ns] - 1) * 100
        neutral_days = (ne - ns) / BARS_PER_DAY
        df = df.iloc[ns:ne + 1].reset_index(drop=True)
        print(f"Neutral window: {neutral_days:.0f}d (bars {ns}-{ne}), drift {drift:+.2f}%")
        all_results['neutral_window'] = {
            'start_bar': int(ns), 'end_bar': int(ne),
            'days': round(neutral_days, 1), 'drift_pct': round(drift, 2),
        }
    else:
        print("No neutral window found, using full data")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values
    n_bars = len(df)
    print(f"Data: {n_bars} bars ({n_bars/BARS_PER_DAY:.0f} days)")

    # Build signal index
    signal_index = build_signal_index(types, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes,
                                  DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
    ema_slope = compute_ema_slope(closes)

    # Load patterns
    long_pats, short_pats, patterns_tpsl = load_patterns()
    print(f"Patterns: {len(long_pats)}L + {len(short_pats)}S = {len(long_pats)+len(short_pats)} total")

    signal_tuples = collect_signals(signal_index, long_pats, short_pats,
                                    patterns_tpsl, 0, n_bars)
    print(f"Total signals: {len(signal_tuples)}")

    npos_kwargs = build_npos_kwargs()

    # ================================================================
    # H0: BASELINE
    # ================================================================
    print("\n" + "=" * 70)
    print("H0: BASELINE (current production — no trend filter)")
    print("=" * 70)

    h0_stats = run_scenario('H0_baseline', signal_tuples, opens, highs, lows,
                            closes, n_bars, atr_ratio, ema_slope, 0, n_bars,
                            **npos_kwargs)
    pnl_mdd = round(h0_stats['pnl'] / h0_stats['mdd'], 2) if h0_stats['mdd'] > 0 else 0

    print(f"  Trades: {h0_stats['trades']}, WR: {h0_stats['wr']}%, "
          f"PnL: {h0_stats['pnl']}%, MDD: {h0_stats['mdd']}%, PnL/MDD: {pnl_mdd}x")
    print(f"  LONG:  {h0_stats['direction']['LONG']}")
    print(f"  SHORT: {h0_stats['direction']['SHORT']}")

    all_results['hypotheses']['H0_baseline'] = {
        'description': 'Current production — no trend filter',
        'stats': h0_stats, 'pnl_mdd': pnl_mdd,
    }

    # ================================================================
    # H1: SMA TREND GATE
    # ================================================================
    print("\n" + "=" * 70)
    print("H1: SMA TREND GATE")
    print("  LONG only when price > SMA(N), SHORT when price < SMA(N)")
    print("=" * 70)

    h1_results = {}
    sma_periods = [50, 100, 200, 500]

    print(f"\n{'SMA':<8} {'Trades':>7} {'WR%':>6} {'PnL%':>9} {'MDD%':>7} "
          f"{'PnL/MDD':>8} {'L_WR':>6} {'L_PnL':>8} {'S_WR':>6} {'S_PnL':>8} {'Blocked':>10}")
    print("-" * 95)

    for period in sma_periods:
        name = f"H1_SMA_{period}"
        stats = run_scenario(name, signal_tuples, opens, highs, lows,
                             closes, n_bars, atr_ratio, ema_slope, 0, n_bars,
                             trend_filter='sma_gate', trend_params={'sma_period': period},
                             **npos_kwargs)
        pm = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
        ld = stats['direction']['LONG']
        sd = stats['direction']['SHORT']
        blk = stats.get('blocked_by_trend', {})
        blk_total = blk.get('LONG', 0) + blk.get('SHORT', 0)
        print(f"SMA{period:<5} {stats['trades']:>7} {stats['wr']:>5.1f}% "
              f"{stats['pnl']:>8.1f}% {stats['mdd']:>6.1f}% {pm:>7.1f}x "
              f"{ld['wr']:>5.1f}% {ld['pnl']:>7.1f}% "
              f"{sd['wr']:>5.1f}% {sd['pnl']:>7.1f}% {blk_total:>10}")
        h1_results[name] = {'stats': stats, 'pnl_mdd': pm, 'period': period}

    # Find best H1
    best_h1_name = max(h1_results, key=lambda k: h1_results[k]['pnl_mdd'])
    best_h1 = h1_results[best_h1_name]
    print(f"\n  Best H1: {best_h1_name} (PnL/MDD {best_h1['pnl_mdd']}x)")
    all_results['hypotheses']['H1_sma_gate'] = {
        'description': 'SMA trend gate — block counter-regime entries',
        'variants': h1_results,
        'best': best_h1_name,
    }

    # ================================================================
    # H2: MOMENTUM TREND GATE
    # ================================================================
    print("\n" + "=" * 70)
    print("H2: MOMENTUM TREND GATE")
    print("  LONG only when N-bar return > 0, SHORT when < 0")
    print("=" * 70)

    h2_results = {}
    mom_lookbacks = [12, 24, 48, 72]

    print(f"\n{'LB':<8} {'Trades':>7} {'WR%':>6} {'PnL%':>9} {'MDD%':>7} "
          f"{'PnL/MDD':>8} {'L_WR':>6} {'L_PnL':>8} {'S_WR':>6} {'S_PnL':>8} {'Blocked':>10}")
    print("-" * 95)

    for lb in mom_lookbacks:
        name = f"H2_MOM_{lb}"
        stats = run_scenario(name, signal_tuples, opens, highs, lows,
                             closes, n_bars, atr_ratio, ema_slope, 0, n_bars,
                             trend_filter='momentum_gate',
                             trend_params={'lookback': lb},
                             **npos_kwargs)
        pm = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
        ld = stats['direction']['LONG']
        sd = stats['direction']['SHORT']
        blk = stats.get('blocked_by_trend', {})
        blk_total = blk.get('LONG', 0) + blk.get('SHORT', 0)
        print(f"LB{lb:<6} {stats['trades']:>7} {stats['wr']:>5.1f}% "
              f"{stats['pnl']:>8.1f}% {stats['mdd']:>6.1f}% {pm:>7.1f}x "
              f"{ld['wr']:>5.1f}% {ld['pnl']:>7.1f}% "
              f"{sd['wr']:>5.1f}% {sd['pnl']:>7.1f}% {blk_total:>10}")
        h2_results[name] = {'stats': stats, 'pnl_mdd': pm, 'lookback': lb}

    best_h2_name = max(h2_results, key=lambda k: h2_results[k]['pnl_mdd'])
    best_h2 = h2_results[best_h2_name]
    print(f"\n  Best H2: {best_h2_name} (PnL/MDD {best_h2['pnl_mdd']}x)")
    all_results['hypotheses']['H2_momentum_gate'] = {
        'description': 'Momentum trend gate — block counter-momentum entries',
        'variants': h2_results,
        'best': best_h2_name,
    }

    # ================================================================
    # H3: COUNTER-DIRECTION SIZING REDUCTION
    # ================================================================
    print("\n" + "=" * 70)
    print("H3: COUNTER-DIRECTION SIZING REDUCTION (SMA 200)")
    print("  Counter-regime entries at reduced size, not blocked")
    print("=" * 70)

    h3_results = {}
    size_mults = [0.25, 0.5, 0.75]

    print(f"\n{'Mult':<8} {'Trades':>7} {'WR%':>6} {'PnL%':>9} {'MDD%':>7} "
          f"{'PnL/MDD':>8} {'L_WR':>6} {'L_PnL':>8} {'S_WR':>6} {'S_PnL':>8}")
    print("-" * 85)

    for mult in size_mults:
        name = f"H3_SIZE_{int(mult*100)}"
        stats = run_scenario(name, signal_tuples, opens, highs, lows,
                             closes, n_bars, atr_ratio, ema_slope, 0, n_bars,
                             trend_filter='sizing_reduction',
                             trend_params={'sma_period': 200, 'size_mult': mult},
                             **npos_kwargs)
        pm = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
        ld = stats['direction']['LONG']
        sd = stats['direction']['SHORT']
        print(f"x{mult:<7} {stats['trades']:>7} {stats['wr']:>5.1f}% "
              f"{stats['pnl']:>8.1f}% {stats['mdd']:>6.1f}% {pm:>7.1f}x "
              f"{ld['wr']:>5.1f}% {ld['pnl']:>7.1f}% "
              f"{sd['wr']:>5.1f}% {sd['pnl']:>7.1f}%")
        h3_results[name] = {'stats': stats, 'pnl_mdd': pm, 'size_mult': mult}

    best_h3_name = max(h3_results, key=lambda k: h3_results[k]['pnl_mdd'])
    best_h3 = h3_results[best_h3_name]
    print(f"\n  Best H3: {best_h3_name} (PnL/MDD {best_h3['pnl_mdd']}x)")
    all_results['hypotheses']['H3_sizing_reduction'] = {
        'description': 'Counter-regime sizing reduction with SMA(200)',
        'variants': h3_results,
        'best': best_h3_name,
    }

    # ================================================================
    # H4: ASYMMETRIC DIRECTION CAP
    # ================================================================
    print("\n" + "=" * 70)
    print("H4: ASYMMETRIC DIRECTION CAP (SMA 200)")
    print("  Counter-regime direction capped at [2, 3, 4] instead of 7")
    print("=" * 70)

    h4_results = {}
    counter_caps = [2, 3, 4]

    print(f"\n{'Cap':<8} {'Trades':>7} {'WR%':>6} {'PnL%':>9} {'MDD%':>7} "
          f"{'PnL/MDD':>8} {'L_WR':>6} {'L_PnL':>8} {'S_WR':>6} {'S_PnL':>8} {'TrendBlk':>10}")
    print("-" * 95)

    for cap in counter_caps:
        name = f"H4_CAP_{cap}"
        stats = run_scenario(name, signal_tuples, opens, highs, lows,
                             closes, n_bars, atr_ratio, ema_slope, 0, n_bars,
                             trend_filter='asymmetric_cap',
                             trend_params={'sma_period': 200, 'counter_cap': cap},
                             **npos_kwargs)
        pm = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
        ld = stats['direction']['LONG']
        sd = stats['direction']['SHORT']
        blk = stats.get('blocked', {})
        trend_blk = blk.get('trend_cap', 0)
        print(f"Cap{cap:<5} {stats['trades']:>7} {stats['wr']:>5.1f}% "
              f"{stats['pnl']:>8.1f}% {stats['mdd']:>6.1f}% {pm:>7.1f}x "
              f"{ld['wr']:>5.1f}% {ld['pnl']:>7.1f}% "
              f"{sd['wr']:>5.1f}% {sd['pnl']:>7.1f}% {trend_blk:>10}")
        h4_results[name] = {'stats': stats, 'pnl_mdd': pm, 'counter_cap': cap}

    best_h4_name = max(h4_results, key=lambda k: h4_results[k]['pnl_mdd'])
    best_h4 = h4_results[best_h4_name]
    print(f"\n  Best H4: {best_h4_name} (PnL/MDD {best_h4['pnl_mdd']}x)")
    all_results['hypotheses']['H4_asymmetric_cap'] = {
        'description': 'Asymmetric direction cap based on SMA(200)',
        'variants': h4_results,
        'best': best_h4_name,
    }

    # ================================================================
    # IS SUMMARY — Find best overall hypothesis
    # ================================================================
    print("\n" + "=" * 70)
    print("IS SUMMARY — All Hypotheses vs Baseline")
    print("=" * 70)

    summary_rows = [('H0_baseline', pnl_mdd, h0_stats)]
    for name, data in h1_results.items():
        summary_rows.append((name, data['pnl_mdd'], data['stats']))
    for name, data in h2_results.items():
        summary_rows.append((name, data['pnl_mdd'], data['stats']))
    for name, data in h3_results.items():
        summary_rows.append((name, data['pnl_mdd'], data['stats']))
    for name, data in h4_results.items():
        summary_rows.append((name, data['pnl_mdd'], data['stats']))

    summary_rows.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Hypothesis':<20} {'PnL/MDD':>8} {'PnL%':>9} {'MDD%':>7} {'WR%':>6} {'Trades':>7}")
    print("-" * 60)
    for name, pm, stats in summary_rows:
        marker = " <-- baseline" if name == 'H0_baseline' else ""
        print(f"{name:<20} {pm:>7.1f}x {stats['pnl']:>8.1f}% {stats['mdd']:>6.1f}% "
              f"{stats['wr']:>5.1f}% {stats['trades']:>7}{marker}")

    # Identify candidates that beat baseline PnL/MDD
    baseline_pnl_mdd = pnl_mdd
    candidates = [(name, pm, stats) for name, pm, stats in summary_rows
                  if pm > baseline_pnl_mdd and name != 'H0_baseline']

    if not candidates:
        print("\n  ** No hypothesis beats baseline PnL/MDD **")
        print("  VERDICT: KEEP BASELINE (all trend filters hurt PnL/MDD)")
        all_results['verdict'] = 'KEEP_BASELINE'
        all_results['reason'] = 'No hypothesis beats baseline PnL/MDD in IS'

    else:
        print(f"\n  {len(candidates)} candidates beat baseline PnL/MDD ({baseline_pnl_mdd}x)")

        # ================================================================
        # WF VALIDATION for candidates + best of each H
        # ================================================================
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION (3-fold expanding window)")
        print("=" * 70)

        wf_candidates = []
        # Include best of each H + top 3 overall
        to_validate = set()
        to_validate.add(best_h1_name)
        to_validate.add(best_h2_name)
        to_validate.add(best_h3_name)
        to_validate.add(best_h4_name)
        for name, pm, stats in candidates[:3]:
            to_validate.add(name)

        # Map names to trend filter configs
        filter_configs = {}
        for name, data in h1_results.items():
            filter_configs[name] = ('sma_gate', {'sma_period': data['period']})
        for name, data in h2_results.items():
            filter_configs[name] = ('momentum_gate', {'lookback': data['lookback']})
        for name, data in h3_results.items():
            filter_configs[name] = ('sizing_reduction', {'sma_period': 200, 'size_mult': data['size_mult']})
        for name, data in h4_results.items():
            filter_configs[name] = ('asymmetric_cap', {'sma_period': 200, 'counter_cap': data['counter_cap']})

        # Baseline WF
        print("\n--- H0 Baseline WF ---")
        wf_h0 = wf_with_trend_filter(signal_index, long_pats, short_pats,
                                      patterns_tpsl, opens, highs, lows,
                                      closes, n_bars, None, None)
        print(f"  {wf_h0['positive_folds']}/{WF_FOLDS} positive, "
              f"OOS PnL {wf_h0['total_oos_pnl']}%")
        for f in wf_h0['folds']:
            print(f"  Fold {f['fold']}: WR {f['oos_wr']}%, PnL {f['oos_pnl']}%, "
                  f"MDD {f['oos_mdd']}%")
        all_results['wf_baseline'] = wf_h0

        wf_results = {}
        for name in sorted(to_validate):
            if name not in filter_configs:
                continue
            tf, tp = filter_configs[name]
            print(f"\n--- {name} WF ---")
            t0 = time.time()
            wf = wf_with_trend_filter(signal_index, long_pats, short_pats,
                                       patterns_tpsl, opens, highs, lows,
                                       closes, n_bars, tf, tp)
            elapsed = time.time() - t0
            print(f"  {wf['positive_folds']}/{WF_FOLDS} positive, "
                  f"OOS PnL {wf['total_oos_pnl']}% ({elapsed:.1f}s)")
            for f in wf['folds']:
                d = f.get('oos_direction', {})
                l_info = f"L:{d.get('LONG', {}).get('wr', 0):.0f}%/{d.get('LONG', {}).get('pnl', 0):.1f}%"
                s_info = f"S:{d.get('SHORT', {}).get('wr', 0):.0f}%/{d.get('SHORT', {}).get('pnl', 0):.1f}%"
                print(f"  Fold {f['fold']}: WR {f['oos_wr']}%, PnL {f['oos_pnl']}%, "
                      f"MDD {f['oos_mdd']}%, {l_info}, {s_info}")
            wf_results[name] = wf

            # Mark pass/fail
            is_pm = None
            for row_name, row_pm, _ in summary_rows:
                if row_name == name:
                    is_pm = row_pm
                    break
            delta_vs_baseline = wf['total_oos_pnl'] - wf_h0['total_oos_pnl']
            verdict = 'PASS' if wf['wf_pass'] else 'FAIL'
            print(f"  WF: {verdict}, OOS delta vs baseline: {delta_vs_baseline:+.1f}%")

        all_results['wf_results'] = wf_results

        # ================================================================
        # RANDOM DISCRIMINATION TEST for WF-passing hypotheses
        # ================================================================
        wf_passing = [name for name, wf in wf_results.items() if wf['wf_pass']]

        if wf_passing:
            print("\n" + "=" * 70)
            print("RANDOM DISCRIMINATION TEST (10 random signal sets)")
            print("If >50% of random pass WF => NON-DISCRIMINATING")
            print("=" * 70)

            discrim_results = {}
            for name in wf_passing:
                tf, tp = filter_configs[name]
                print(f"\n  Testing {name}...")
                dr = random_discrimination_test(
                    signal_index, long_pats, short_pats, patterns_tpsl,
                    opens, highs, lows, closes, n_bars, tf, tp)
                status = "DISCRIMINATING" if dr['discriminating'] else "NON-DISCRIMINATING"
                print(f"  {name}: {dr['passes']}/{dr['n_random']} random pass "
                      f"({dr['pass_rate']}%) => {status}")
                discrim_results[name] = dr

            all_results['discrimination_test'] = discrim_results

            # Final candidates: WF pass + discriminating
            final_candidates = [name for name in wf_passing
                               if discrim_results.get(name, {}).get('discriminating', False)]

            if final_candidates:
                # ================================================================
                # H5: BEST COMBO (if multiple candidates)
                # ================================================================
                if len(final_candidates) >= 2:
                    print("\n" + "=" * 70)
                    print("H5: BEST COMBO — Combining top candidates")
                    print("=" * 70)
                    # Try combining best H1/H2 gate with best H3/H4 modification
                    # (This is hypothesis-dependent; we try the top 2 IS winners)
                    print("  Skipping H5 combo — test individual candidates first")

                # MC test on final candidates
                print("\n" + "=" * 70)
                print("MONTE CARLO TEST (3 seeds, 10k sims)")
                print("=" * 70)

                mc_results = {}
                for name in final_candidates:
                    tf, tp = filter_configs[name]
                    # Run full IS with filter
                    trades, _ = portfolio_npos_trend_filtered(
                        signal_tuples, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, 0, n_bars,
                        trend_filter=tf, trend_params=tp, **npos_kwargs)
                    mc_p = mc_sign_test(trades)
                    status = "PASS" if mc_p < MC_THRESHOLD else "FAIL"
                    print(f"  {name}: MC p={mc_p:.4f} => {status}")
                    mc_results[name] = {'p_value': mc_p, 'pass': mc_p < MC_THRESHOLD}

                all_results['mc_results'] = mc_results

                # Final verdict
                fully_validated = [name for name in final_candidates
                                   if mc_results.get(name, {}).get('pass', False)]

                if fully_validated:
                    best_final = max(fully_validated,
                                     key=lambda n: next(pm for nm, pm, _ in summary_rows if nm == n))
                    best_pm = next(pm for nm, pm, _ in summary_rows if nm == best_final)
                    delta_pm = best_pm - baseline_pnl_mdd

                    print(f"\n  FINAL VERDICT: {best_final} PASS")
                    print(f"  PnL/MDD: {best_pm}x (baseline {baseline_pnl_mdd}x, delta {delta_pm:+.1f}x)")
                    all_results['verdict'] = f'GO_{best_final}'
                    all_results['best_candidate'] = best_final
                    all_results['best_pnl_mdd'] = best_pm
                    all_results['baseline_pnl_mdd'] = baseline_pnl_mdd
                else:
                    print("\n  VERDICT: KEEP BASELINE (MC failed for all candidates)")
                    all_results['verdict'] = 'KEEP_BASELINE'
                    all_results['reason'] = 'All candidates failed MC test'
            else:
                print("\n  VERDICT: KEEP BASELINE (no discriminating WF-passing candidates)")
                all_results['verdict'] = 'KEEP_BASELINE'
                all_results['reason'] = 'WF-passing candidates are non-discriminating'
        else:
            print("\n  VERDICT: KEEP BASELINE (no candidates pass WF)")
            all_results['verdict'] = 'KEEP_BASELINE'
            all_results['reason'] = 'No candidates pass WF 3/3'

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    elapsed = time.time() - t_start
    all_results['elapsed_seconds'] = round(elapsed, 1)
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Clean stats for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_results = clean_for_json(all_results)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Final summary
    print("\n" + "=" * 70)
    print(f"VERDICT: {all_results.get('verdict', 'UNKNOWN')}")
    if all_results.get('reason'):
        print(f"REASON: {all_results['reason']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
