#!/usr/bin/env python3
"""
Cascade SL Deep Improvement Study (v2 — corrected)

Current: tighten_pct=85% (SL distance * 0.15 after same-dir SL exit)

Phases:
  1. Baseline Characterization: Cascade ON vs OFF (IS + WF), cascade stats
  2. Keep Ratio Sweep: 0.05-1.0 (IS + WF top 3)
  3. Alternative Strategies: proportional, graduated, pos-count, trailing, time-decay
  4. Cascade + Time-Decay TP interaction (v1.62.0 synergy)
  5. MC Discrimination (3-seed)
  6. WF Final Validation (best overall)
  7. Summary & Recommendation

Standard Research Protocol: next-bar entry, intrabar exit, 0.10% fee,
  0.02% slippage, 3x leverage, compound sizing, tp_scale=0.72.

FIXES vs v1 study:
  - WF fold formula corrected: is_end = int(n*(fi+1)/(n_folds+1))
  - tp_scale_factor=0.72 applied to all signals
  - vol_mult cap (MAX_DAILY_LOSS_PCT/LEVERAGE/sl_pct) in all custom simulators
  - MTM MDD tracked in all custom simulators
"""

import json
import numpy as np
import time
import sys
from pathlib import Path
from collections import defaultdict

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope, _check_exit_npos,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT, MAX_DAILY_LOSS_PCT,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/cascade_sl_deep_study.json"

TP_SCALE_FACTOR = 0.72  # v1.61.0

NPOS_DEFAULTS = dict(
    n_slots=DEFAULT_N_SLOTS,
    direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
    agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
    momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO,
    clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS,
)


# ─── Helper functions ───

def run_npos(signals, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope,
             start_bar, end_bar, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
             **extra_kwargs):
    """Run portfolio_npos with optional cascade override."""
    kwargs = {**NPOS_DEFAULTS}
    kwargs.update(extra_kwargs)
    kwargs['cascade_tighten_pct'] = cascade_tighten_pct

    trades, raw_stats = portfolio_npos(
        signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        **kwargs
    )
    stats = calc_stats_compound(trades)
    if raw_stats.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw_stats['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw_stats.items() if k not in stats})
    return trades, stats


def run_wf(signals, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope,
           ns, ne, n_folds=3, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
           **extra_kwargs):
    """Expanding-window WF. CORRECTED fold formula."""
    total = ne - ns
    results = []

    for fi in range(n_folds):
        is_end = ns + int(total * (fi + 1) / (n_folds + 1))
        oos_start = is_end
        oos_end = ns + int(total * (fi + 2) / (n_folds + 1))
        if oos_start >= oos_end or oos_start >= ne:
            continue

        trades, stats = run_npos(
            signals, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            cascade_tighten_pct=cascade_tighten_pct,
            **extra_kwargs
        )
        results.append({
            'fold': fi + 1,
            'oos_start': int(oos_start),
            'oos_end': int(oos_end),
            'pnl': stats.get('pnl', 0),
            'wr': stats.get('wr', 0),
            'trades': stats.get('trades', 0),
            'mdd': stats.get('mdd', 0),
        })

    oos_total_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) and len(results) == n_folds
    return results, oos_total_pnl, all_pass


def run_wf_custom(sim_fn, signals, opens, highs, lows, closes, n_bars,
                  atr_ratio, ema_slope, ns, ne, n_folds=3, **sim_kwargs):
    """Expanding-window WF for custom simulator functions."""
    total = ne - ns
    results = []

    for fi in range(n_folds):
        is_end = ns + int(total * (fi + 1) / (n_folds + 1))
        oos_start = is_end
        oos_end = ns + int(total * (fi + 2) / (n_folds + 1))
        if oos_start >= oos_end or oos_start >= ne:
            continue

        out = sim_fn(signals, opens, highs, lows, closes, n_bars,
                     atr_ratio, ema_slope, oos_start, oos_end, **sim_kwargs)
        # Handle both (trades, stats) and (trades, stats, extra) returns
        if isinstance(out, tuple) and len(out) >= 2:
            trades, stats_raw = out[0], out[1]
        else:
            trades, stats_raw = out, {}

        stats = calc_stats_compound(trades)
        if isinstance(stats_raw, dict) and stats_raw.get('mdd_mtm', 0) > 0:
            stats['mdd'] = stats_raw['mdd_mtm']

        results.append({
            'fold': fi + 1,
            'oos_start': int(oos_start),
            'oos_end': int(oos_end),
            'pnl': stats.get('pnl', 0),
            'wr': stats.get('wr', 0),
            'trades': stats.get('trades', 0),
            'mdd': stats.get('mdd', 0),
        })

    oos_total_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) and len(results) == n_folds
    return results, oos_total_pnl, all_pass


def fmt_stats(stats):
    return (f"PnL {stats.get('pnl', 0):+.1f}%, "
            f"WR {stats.get('wr', 0):.1f}%, "
            f"Trades {stats.get('trades', 0)}, "
            f"MDD {stats.get('mdd', 0):.2f}%")


def pnl_mdd(stats):
    mdd = stats.get('mdd', 0)
    return stats.get('pnl', 0) / max(mdd, 0.01) if mdd > 0 else 0


def _get_vol_mult(atr_ratio, sig_bar, sl_pct, clamp_lo, clamp_hi):
    """Compute ATR vol_mult with production parity cap."""
    if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
    else:
        r = 1.0
    if sl_pct > 0:
        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)
    return r


# ─── Custom N-pos Simulator (full production parity) ───

def portfolio_npos_custom(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    cascade_fn=None,  # fn(positions, closed_slots, sl_exit_info, bar) -> modifies positions in-place
    track_cascade=False,
):
    """Full N-pos simulator with pluggable cascade function.

    cascade_fn receives:
      positions: list of open position dicts
      closed_slots: set of slot ids being closed this bar
      sl_exit_info: list of dicts {'direction': str, 'pnl_slot': float, 'sl_pct': float}
      bar: current bar index

    It should modify positions in-place (set eff_sl_override, cascade_count, etc).
    """
    size_pct = 100.0 / DEFAULT_N_SLOTS
    fee = FEE_PCT * LEVERAGE
    clamp_lo, clamp_hi = DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI

    positions = []
    trades = []
    equity = 100.0
    peak_eq = 100.0
    max_dd_mtm = 0.0
    momentum_pause = {'LONG': -1, 'SHORT': -1}
    cascade_events = [] if track_cascade else None
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0, 'max_pos': 0}

    signals_in_range = sorted(
        [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples if start_bar <= s < end_bar],
        key=lambda x: x[0]
    )
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        closed_slots = set()
        bar_pnl_sum = 0.0
        sl_exit_info = []

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                       atr_ratio, fee, clamp_lo, clamp_hi, TIMEOUT_BARS)
            if result is None:
                continue
            if result.get('drop', False):
                closed_slots.add(pos['slot'])
                continue
            result['pattern'] = pos['pattern']
            result['direction'] = pos['direction']
            sm = pos.get('size_mult', 1.0)
            result['size_mult'] = sm
            pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
            result['pnl_portfolio'] = pnl_portfolio
            trades.append(result)
            closed_slots.add(pos['slot'])
            bar_pnl_sum += pnl_portfolio

            if result['reason'] == 'SL':
                sl_exit_info.append({
                    'direction': pos['direction'],
                    'pnl_slot': result['pnl_slot'],
                    'sl_pct': pos['sl_pct'],
                })

        # Apply cascade
        if cascade_fn is not None and sl_exit_info:
            affected = cascade_fn(positions, closed_slots, sl_exit_info, bar,
                                   atr_ratio, clamp_lo, clamp_hi)
            if track_cascade and affected and affected > 0:
                dirs = list(set(e['direction'] for e in sl_exit_info))
                cascade_events.append({
                    'bar': int(bar),
                    'sl_exits': len(sl_exit_info),
                    'directions': dirs,
                    'affected_positions': affected,
                })

        positions = [p for p in positions if p['slot'] not in closed_slots]
        equity += bar_pnl_sum
        if equity > peak_eq:
            peak_eq = equity

        # MTM MDD
        if positions and bar < n_bars:
            mtm_eq = equity
            for pos in positions:
                eb = pos['entry_bar']
                if eb >= n_bars or bar < eb:
                    continue
                ep = opens[eb]
                if ep <= 0:
                    continue
                if pos['direction'] == 'LONG':
                    unr = (closes[bar] / ep - 1) * 100 * LEVERAGE
                else:
                    unr = (1 - closes[bar] / ep) * 100 * LEVERAGE
                sm = pos.get('size_mult', 1.0)
                mtm_eq += unr * (size_pct / 100) * sm
            if mtm_eq > peak_eq:
                peak_eq = mtm_eq
            dd = (peak_eq - mtm_eq) / peak_eq * 100 if peak_eq > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd
        elif not positions:
            if equity > peak_eq:
                peak_eq = equity
            dd = (peak_eq - equity) / peak_eq * 100 if peak_eq > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd

        # Momentum guard
        if DEFAULT_MOMENTUM_LOOKBACK > 0 and DEFAULT_MOMENTUM_THRESHOLD > 0 and bar >= DEFAULT_MOMENTUM_LOOKBACK:
            pn = closes[bar]
            pa = closes[bar - DEFAULT_MOMENTUM_LOOKBACK]
            if pa > 0:
                pc = (pn / pa - 1) * 100
                if pc > DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['SHORT'] = bar + DEFAULT_MOMENTUM_COOLDOWN
                elif pc < -DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['LONG'] = bar + DEFAULT_MOMENTUM_COOLDOWN

        # Entries
        while sig_idx < len(signals_in_range) and signals_in_range[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_in_range[sig_idx]
            sig_idx += 1

            if len(positions) >= DEFAULT_N_SLOTS:
                total_blocked['max_pos'] += 1
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DEFAULT_DIRECTION_CAP:
                total_blocked['dir_cap'] += 1
                continue
            if any(p['pattern'] == pat for p in positions):
                total_blocked['dup_pat'] += 1
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            if DEFAULT_MOMENTUM_LOOKBACK > 0 and bar < momentum_pause.get(direction, -1):
                total_blocked['momentum'] += 1
                continue

            # Regime sizing
            sm = 1.0
            if DEFAULT_REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = DEFAULT_REGIME_MULT
                elif s <= 0 and direction == 'LONG':
                    sm = DEFAULT_REGIME_MULT

            # Aggregate risk cap
            if DEFAULT_AGG_RISK_COUNTER > 0 or DEFAULT_AGG_RISK_WITH > 0:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                cap_pct = DEFAULT_AGG_RISK_COUNTER if is_counter else DEFAULT_AGG_RISK_WITH

                existing_exp = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_r = _get_vol_mult(atr_ratio, p['signal_bar'], p['sl_pct'],
                                           clamp_lo, clamp_hi)
                        p_sm = p.get('size_mult', 1.0)
                        existing_exp += p['sl_pct'] * p_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * p_sm

                new_r = _get_vol_mult(atr_ratio, sig_bar, sl_pct, clamp_lo, clamp_hi)
                new_exp = sl_pct * new_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * sm

                if existing_exp + new_exp > cap_pct:
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

    stats = calc_stats_compound(trades)
    raw_stats = {
        'mdd_mtm': round(max_dd_mtm, 2),
        'blocked': total_blocked,
    }
    if raw_stats['mdd_mtm'] > 0:
        stats['mdd'] = raw_stats['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw_stats.items() if k not in stats})

    if track_cascade:
        return trades, stats, cascade_events
    return trades, stats


# ─── Cascade Strategy Functions ───

def make_fixed_cascade(keep_ratio):
    """Standard fixed cascade: multiply SL by keep_ratio."""
    def cascade_fn(positions, closed_slots, sl_exit_info, bar,
                   atr_ratio, clamp_lo, clamp_hi):
        if keep_ratio >= 1.0:
            return 0
        affected = 0
        sl_dirs = set(e['direction'] for e in sl_exit_info)
        for sl_dir in sl_dirs:
            for pos in positions:
                if pos['slot'] in closed_slots or pos['direction'] != sl_dir:
                    continue
                r = _get_vol_mult(atr_ratio, pos['signal_bar'], pos['sl_pct'],
                                 clamp_lo, clamp_hi)
                cur = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                pos['eff_sl_override'] = cur * keep_ratio
                pos['cascade_count'] = pos.get('cascade_count', 0) + 1
                affected += 1
        return affected
    return cascade_fn


def make_proportional_cascade(base_keep=0.15):
    """Proportional: keep_ratio scales with loss severity.
    Bigger SL loss -> more tightening. Small loss -> less tightening.
    keep = base_keep + (1-base_keep) * (1 - loss_severity)
    where loss_severity = abs(pnl) / (sl_pct * leverage)
    """
    def cascade_fn(positions, closed_slots, sl_exit_info, bar,
                   atr_ratio, clamp_lo, clamp_hi):
        affected = 0
        for sl_info in sl_exit_info:
            sl_dir = sl_info['direction']
            # loss severity: how much of the max SL was hit (0=tiny, 1=full SL)
            max_loss = sl_info['sl_pct'] * LEVERAGE
            severity = min(1.0, abs(sl_info['pnl_slot']) / max(max_loss, 0.01))
            # More severe loss -> lower keep -> more tightening
            keep = base_keep + (1.0 - base_keep) * (1.0 - severity)

            for pos in positions:
                if pos['slot'] in closed_slots or pos['direction'] != sl_dir:
                    continue
                r = _get_vol_mult(atr_ratio, pos['signal_bar'], pos['sl_pct'],
                                 clamp_lo, clamp_hi)
                cur = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                pos['eff_sl_override'] = cur * keep
                pos['cascade_count'] = pos.get('cascade_count', 0) + 1
                affected += 1
        return affected
    return cascade_fn


def make_graduated_cascade(steps=None):
    """Graduated: 1st cascade -> keep=0.50, 2nd -> 0.25, 3rd+ -> 0.10."""
    if steps is None:
        steps = [0.50, 0.25, 0.10]

    def cascade_fn(positions, closed_slots, sl_exit_info, bar,
                   atr_ratio, clamp_lo, clamp_hi):
        affected = 0
        sl_dirs = set(e['direction'] for e in sl_exit_info)
        for sl_dir in sl_dirs:
            for pos in positions:
                if pos['slot'] in closed_slots or pos['direction'] != sl_dir:
                    continue
                cc = pos.get('cascade_count', 0)
                keep = steps[min(cc, len(steps) - 1)]

                r = _get_vol_mult(atr_ratio, pos['signal_bar'], pos['sl_pct'],
                                 clamp_lo, clamp_hi)
                cur = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                pos['eff_sl_override'] = cur * keep
                pos['cascade_count'] = cc + 1
                affected += 1
        return affected
    return cascade_fn


def make_poscount_cascade(min_pos=2, base_keep=0.15):
    """Position-count cascade: only activates when >=min_pos same-dir positions.
    Intensity scales with count: more positions -> more tightening.
    1 pos -> no cascade (skip). 2 -> mild. 3+ -> full.
    """
    def cascade_fn(positions, closed_slots, sl_exit_info, bar,
                   atr_ratio, clamp_lo, clamp_hi):
        affected = 0
        sl_dirs = set(e['direction'] for e in sl_exit_info)
        for sl_dir in sl_dirs:
            same_dir = [p for p in positions
                        if p['slot'] not in closed_slots and p['direction'] == sl_dir]
            n = len(same_dir)
            if n < min_pos:
                continue
            # Scale: at min_pos -> keep=0.6, at 5+ -> keep=base_keep
            scale = min(1.0, (n - min_pos + 1) / 4.0)
            keep = 1.0 - scale * (1.0 - base_keep)

            for pos in same_dir:
                r = _get_vol_mult(atr_ratio, pos['signal_bar'], pos['sl_pct'],
                                 clamp_lo, clamp_hi)
                cur = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                pos['eff_sl_override'] = cur * keep
                pos['cascade_count'] = pos.get('cascade_count', 0) + 1
                affected += 1
        return affected
    return cascade_fn


def make_timedecay_cascade(base_keep=0.15, recovery_bars=72):
    """Time-decay cascade: tighten SL then gradually restore toward original.
    After cascade, SL recovers linearly over recovery_bars back to original.

    Implementation: set eff_sl_override on cascade, then each bar
    the position restores a fraction. We track recovery state via
    cascade_restore_start and cascade_original_sl.
    """
    # Note: because _check_exit_npos reads eff_sl_override, we need to
    # update it every bar for positions in recovery. This requires hooking
    # into the bar loop. We'll use a two-part approach:
    # 1. cascade_fn sets the tightened SL and marks recovery state
    # 2. We also need a pre-exit hook... but we can't easily do that.
    #
    # Alternative: just track the initial tighten and restore target.
    # The position's eff_sl_override gets updated each time cascade_fn is called.
    # For recovery, we'd need to modify the main loop. Instead, let's implement
    # this as a full custom simulator since it needs per-bar updates.
    pass  # Handled separately below


def portfolio_npos_timedecay_cascade(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    base_keep=0.15, recovery_bars=72,
):
    """N-pos with time-decay cascade: tighten on SL, then gradually restore."""
    size_pct = 100.0 / DEFAULT_N_SLOTS
    fee = FEE_PCT * LEVERAGE
    clamp_lo, clamp_hi = DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI

    positions = []
    trades = []
    equity = 100.0
    peak_eq = 100.0
    max_dd_mtm = 0.0
    momentum_pause = {'LONG': -1, 'SHORT': -1}

    signals_in_range = sorted(
        [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples if start_bar <= s < end_bar],
        key=lambda x: x[0]
    )
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # Pre-exit: restore cascade-tightened SLs toward original
        for pos in positions:
            restore_info = pos.get('_cascade_restore')
            if restore_info is not None:
                elapsed = bar - restore_info['start_bar']
                if elapsed >= recovery_bars:
                    # Fully restored
                    pos['eff_sl_override'] = restore_info['original_sl']
                    pos.pop('_cascade_restore', None)
                else:
                    # Linear interpolation from tightened -> original
                    frac = elapsed / recovery_bars
                    tightened = restore_info['tightened_sl']
                    original = restore_info['original_sl']
                    pos['eff_sl_override'] = tightened + (original - tightened) * frac

        closed_slots = set()
        bar_pnl_sum = 0.0
        sl_exit_info = []

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                       atr_ratio, fee, clamp_lo, clamp_hi, TIMEOUT_BARS)
            if result is None:
                continue
            if result.get('drop', False):
                closed_slots.add(pos['slot'])
                continue
            result['pattern'] = pos['pattern']
            result['direction'] = pos['direction']
            sm = pos.get('size_mult', 1.0)
            result['size_mult'] = sm
            pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
            result['pnl_portfolio'] = pnl_portfolio
            trades.append(result)
            closed_slots.add(pos['slot'])
            bar_pnl_sum += pnl_portfolio
            if result['reason'] == 'SL':
                sl_exit_info.append({'direction': pos['direction']})

        # Apply cascade + schedule recovery
        if sl_exit_info:
            sl_dirs = set(e['direction'] for e in sl_exit_info)
            for sl_dir in sl_dirs:
                for pos in positions:
                    if pos['slot'] in closed_slots or pos['direction'] != sl_dir:
                        continue
                    r = _get_vol_mult(atr_ratio, pos['signal_bar'], pos['sl_pct'],
                                     clamp_lo, clamp_hi)
                    original_sl = pos['sl_pct'] * r
                    cur = pos.get('eff_sl_override') or original_sl
                    tightened = cur * base_keep
                    pos['eff_sl_override'] = tightened
                    pos['cascade_count'] = pos.get('cascade_count', 0) + 1
                    pos['_cascade_restore'] = {
                        'start_bar': bar,
                        'tightened_sl': tightened,
                        'original_sl': original_sl,
                    }

        positions = [p for p in positions if p['slot'] not in closed_slots]
        equity += bar_pnl_sum
        if equity > peak_eq:
            peak_eq = equity

        # MTM MDD
        if positions and bar < n_bars:
            mtm_eq = equity
            for pos in positions:
                eb = pos['entry_bar']
                if eb >= n_bars or bar < eb:
                    continue
                ep = opens[eb]
                if ep <= 0:
                    continue
                if pos['direction'] == 'LONG':
                    unr = (closes[bar] / ep - 1) * 100 * LEVERAGE
                else:
                    unr = (1 - closes[bar] / ep) * 100 * LEVERAGE
                sm = pos.get('size_mult', 1.0)
                mtm_eq += unr * (size_pct / 100) * sm
            if mtm_eq > peak_eq:
                peak_eq = mtm_eq
            dd = (peak_eq - mtm_eq) / peak_eq * 100 if peak_eq > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd
        elif not positions:
            if equity > peak_eq:
                peak_eq = equity
            dd = (peak_eq - equity) / peak_eq * 100 if peak_eq > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd

        # Momentum guard
        if DEFAULT_MOMENTUM_LOOKBACK > 0 and DEFAULT_MOMENTUM_THRESHOLD > 0 and bar >= DEFAULT_MOMENTUM_LOOKBACK:
            pn = closes[bar]
            pa = closes[bar - DEFAULT_MOMENTUM_LOOKBACK]
            if pa > 0:
                pc = (pn / pa - 1) * 100
                if pc > DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['SHORT'] = bar + DEFAULT_MOMENTUM_COOLDOWN
                elif pc < -DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['LONG'] = bar + DEFAULT_MOMENTUM_COOLDOWN

        # Entries
        while sig_idx < len(signals_in_range) and signals_in_range[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_in_range[sig_idx]
            sig_idx += 1

            if len(positions) >= DEFAULT_N_SLOTS:
                continue
            if sum(1 for p in positions if p['direction'] == direction) >= DEFAULT_DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            if DEFAULT_MOMENTUM_LOOKBACK > 0 and bar < momentum_pause.get(direction, -1):
                continue

            sm = 1.0
            if DEFAULT_REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = DEFAULT_REGIME_MULT
                elif s <= 0 and direction == 'LONG':
                    sm = DEFAULT_REGIME_MULT

            if DEFAULT_AGG_RISK_COUNTER > 0 or DEFAULT_AGG_RISK_WITH > 0:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                cap_pct = DEFAULT_AGG_RISK_COUNTER if is_counter else DEFAULT_AGG_RISK_WITH
                existing_exp = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_r = _get_vol_mult(atr_ratio, p['signal_bar'], p['sl_pct'],
                                           clamp_lo, clamp_hi)
                        existing_exp += p['sl_pct'] * p_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * p.get('size_mult', 1.0)
                new_r = _get_vol_mult(atr_ratio, sig_bar, sl_pct, clamp_lo, clamp_hi)
                new_exp = sl_pct * new_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * sm
                if existing_exp + new_exp > cap_pct:
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

    stats = calc_stats_compound(trades)
    raw_stats = {'mdd_mtm': round(max_dd_mtm, 2)}
    if raw_stats['mdd_mtm'] > 0:
        stats['mdd'] = raw_stats['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    return trades, stats


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("CASCADE SL DEEP IMPROVEMENT STUDY (v2 — corrected)")
print("=" * 70)

df = load_and_classify(DATA_FILE)
opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
n_bars = len(df)
type_codes = df['candle_type'].values

atr_ratio = compute_atr_ratio(highs, lows, closes)
ema_slope = compute_ema_slope(closes)

with open(PATTERNS_FILE) as f:
    pat_data = json.load(f)
pattern_details = pat_data.get('pattern_details', {})

signal_index = build_signal_index(type_codes, n_bars)

# Build signal tuples WITH tp_scale_factor=0.72
signals = []
for key, info in pattern_details.items():
    pat = info.get('pattern') or key.rsplit('_', 1)[0]
    direction = info['direction']
    tp = max(0.3, info['tp'] * TP_SCALE_FACTOR)
    sl = info['sl']
    if pat in signal_index:
        for bar in signal_index[pat]:
            signals.append((bar, key, direction, tp, sl))
signals.sort(key=lambda x: x[0])

ns, ne = find_neutral_window(closes)
n_sig_nw = len([s for s in signals if ns <= s[0] < ne])
print(f"Loaded {len(pattern_details)} patterns, {n_bars} bars, "
      f"{len(signals)} signals ({n_sig_nw} in NW {ns}-{ne}, {(ne-ns)//288:.0f}d)")
print(f"Config: tp_scale={TP_SCALE_FACTOR}, LEVERAGE={LEVERAGE}, "
      f"FEE={FEE_PCT}%, TIMEOUT={TIMEOUT_BARS}")

all_results = {
    "study": "cascade_sl_deep_improvement_v2",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "data_bars": n_bars,
    "patterns": len(pattern_details),
    "signals": len(signals),
    "signals_in_nw": n_sig_nw,
    "neutral_window": [int(ns), int(ne)],
    "tp_scale_factor": TP_SCALE_FACTOR,
    "baseline_cascade_pct": DEFAULT_CASCADE_TIGHTEN_PCT,
}


# ═══════════════════════════════════════════════════════════
# PHASE 1: Baseline Characterization
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: Baseline — Cascade ON (85%) vs OFF")
print("=" * 70)

# IS full neutral window
trades_on, stats_on = run_npos(signals, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)
trades_off, stats_off = run_npos(signals, opens, highs, lows, closes, n_bars,
                                  atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=0)

pm_on, pm_off = pnl_mdd(stats_on), pnl_mdd(stats_off)
print(f"  ON  (85%): {fmt_stats(stats_on)}, PnL/MDD {pm_on:.1f}x")
print(f"  OFF  (0%): {fmt_stats(stats_off)}, PnL/MDD {pm_off:.1f}x")
print(f"  Delta: PnL {stats_on['pnl']-stats_off['pnl']:+.1f}%, "
      f"MDD {stats_on.get('mdd',0)-stats_off.get('mdd',0):+.2f}%")

# Cascade event tracking
_, stats_tracked, cascade_events = portfolio_npos_custom(
    signals, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne,
    cascade_fn=make_fixed_cascade(0.15), track_cascade=True)

total_events = len(cascade_events)
total_affected = sum(e['affected_positions'] for e in cascade_events)
avg_affected = total_affected / total_events if total_events > 0 else 0
multi_sl = sum(1 for e in cascade_events if e['sl_exits'] >= 2)
dir_dist = defaultdict(int)
for e in cascade_events:
    for d in e['directions']:
        dir_dist[d] += 1

# Cascade depth: count max cascade_count across a single position
cascade_depth_stats = []
for t in trades_on:
    # We can estimate depth from pnl patterns, but better to get from tracked sim
    pass

print(f"\n  Cascade events: {total_events}")
print(f"  Total positions affected: {total_affected}")
print(f"  Avg affected/event: {avg_affected:.1f}")
print(f"  Multi-SL bars: {multi_sl}")
print(f"  Direction: LONG {dir_dist.get('LONG',0)}, SHORT {dir_dist.get('SHORT',0)}")

# WF
wf_on, oos_on, pass_on = run_wf(signals, opens, highs, lows, closes, n_bars,
                                  atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)
wf_off, oos_off, pass_off = run_wf(signals, opens, highs, lows, closes, n_bars,
                                     atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=0)

wf_on_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%({r['trades']}t)" for r in wf_on)
wf_off_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%({r['trades']}t)" for r in wf_off)
print(f"\n  WF ON:  OOS {oos_on:+.1f}% {'PASS' if pass_on else 'FAIL'} [{wf_on_str}]")
print(f"  WF OFF: OOS {oos_off:+.1f}% {'PASS' if pass_off else 'FAIL'} [{wf_off_str}]")

all_results["phase1_baseline"] = {
    "cascade_on": {"pnl": stats_on['pnl'], "wr": stats_on.get('wr', 0),
                   "mdd": stats_on.get('mdd', 0), "trades": stats_on.get('trades', 0),
                   "pnl_mdd": round(pm_on, 2),
                   "wf_folds": wf_on, "oos_pnl": oos_on, "wf_pass": pass_on},
    "cascade_off": {"pnl": stats_off['pnl'], "wr": stats_off.get('wr', 0),
                    "mdd": stats_off.get('mdd', 0), "trades": stats_off.get('trades', 0),
                    "pnl_mdd": round(pm_off, 2),
                    "wf_folds": wf_off, "oos_pnl": oos_off, "wf_pass": pass_off},
    "cascade_events": total_events,
    "positions_affected": total_affected,
    "avg_affected": round(avg_affected, 1),
    "multi_sl_bars": multi_sl,
    "dir_distribution": dict(dir_dist),
}


# ═══════════════════════════════════════════════════════════
# PHASE 2: Keep Ratio Sweep
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Keep Ratio Sweep")
print("=" * 70)

# keep_ratio -> cascade_tighten_pct mapping:
# keep=0.05 -> pct=95, keep=0.15 -> pct=85, keep=1.0 -> pct=0

keep_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0]
sweep_results = []

for keep in keep_ratios:
    pct = int(round((1.0 - keep) * 100))
    _, stats = run_npos(signals, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=pct)
    pm = pnl_mdd(stats)
    row = {
        "keep_ratio": keep,
        "tighten_pct": pct,
        "is_pnl": stats['pnl'],
        "is_wr": stats.get('wr', 0),
        "is_mdd": stats.get('mdd', 0),
        "is_trades": stats.get('trades', 0),
        "is_pnl_mdd": round(pm, 2),
    }
    sweep_results.append(row)
    marker = " <-- CURRENT" if keep == 0.15 else (" <-- OFF" if keep == 1.0 else "")
    print(f"  keep={keep:.2f} (pct={pct:2d}%): PnL {stats['pnl']:+.1f}%, "
          f"WR {stats.get('wr',0):.1f}%, MDD {stats.get('mdd',0):.2f}%, "
          f"PnL/MDD {pm:.1f}x, Trades {stats.get('trades',0)}{marker}")

# WF top 3 by PnL/MDD
sorted_sweep = sorted(sweep_results, key=lambda r: r['is_pnl_mdd'], reverse=True)
top3_keeps = [r['keep_ratio'] for r in sorted_sweep[:3]]
print(f"\n  Top 3 by IS PnL/MDD: keep={top3_keeps}")

p2_wf = {}
for r in sorted_sweep[:3]:
    keep = r['keep_ratio']
    pct = r['tighten_pct']
    wf_folds, oos, wf_pass = run_wf(signals, opens, highs, lows, closes, n_bars,
                                      atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=pct)
    p2_wf[keep] = {"oos_pnl": round(oos, 1), "wf_pass": wf_pass, "folds": wf_folds}
    marker = " <-- CURRENT" if keep == 0.15 else ""
    folds_str = ', '.join(f"F{f['fold']}:{f['pnl']:+.1f}%" for f in wf_folds)
    print(f"  WF keep={keep:.2f}: OOS {oos:+.1f}% {'PASS' if wf_pass else 'FAIL'} "
          f"[{folds_str}]{marker}")

# Also WF current if not in top 3
if 0.15 not in top3_keeps:
    wf_folds, oos, wf_pass = run_wf(signals, opens, highs, lows, closes, n_bars,
                                      atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)
    p2_wf[0.15] = {"oos_pnl": round(oos, 1), "wf_pass": wf_pass, "folds": wf_folds}
    folds_str = ', '.join(f"F{f['fold']}:{f['pnl']:+.1f}%" for f in wf_folds)
    print(f"  WF keep=0.15: OOS {oos:+.1f}% {'PASS' if wf_pass else 'FAIL'} "
          f"[{folds_str}] <-- CURRENT")

all_results["phase2_sweep"] = {
    "configs": sweep_results,
    "top3": [{"keep": r['keep_ratio'], "pnl_mdd": r['is_pnl_mdd']} for r in sorted_sweep[:3]],
    "wf_validation": {str(k): v for k, v in p2_wf.items()},
}


# ═══════════════════════════════════════════════════════════
# PHASE 3: Alternative Cascade Strategies
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Alternative Cascade Strategies")
print("=" * 70)

strategies = {
    'fixed_0.15': {
        'desc': 'Current: fixed keep=0.15 (compounds)',
        'fn': make_fixed_cascade(0.15),
    },
    'proportional': {
        'desc': 'Proportional: keep scales with loss severity',
        'fn': make_proportional_cascade(base_keep=0.15),
    },
    'graduated': {
        'desc': 'Graduated: 1st=0.50, 2nd=0.25, 3rd+=0.10',
        'fn': make_graduated_cascade([0.50, 0.25, 0.10]),
    },
    'poscount_2': {
        'desc': 'Pos-count: only >=2 same-dir, intensity scales',
        'fn': make_poscount_cascade(min_pos=2, base_keep=0.15),
    },
    'poscount_3': {
        'desc': 'Pos-count: only >=3 same-dir, intensity scales',
        'fn': make_poscount_cascade(min_pos=3, base_keep=0.15),
    },
    'no_cascade': {
        'desc': 'No cascade (OFF)',
        'fn': None,
    },
}

p3_results = {}

for name, cfg in strategies.items():
    trades, stats = portfolio_npos_custom(
        signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        cascade_fn=cfg['fn']
    )
    pm = pnl_mdd(stats)
    marker = " <-- CURRENT" if name == 'fixed_0.15' else ""
    print(f"  {name:16s}: PnL {stats['pnl']:+.1f}%, WR {stats.get('wr',0):.1f}%, "
          f"MDD {stats.get('mdd',0):.2f}%, PnL/MDD {pm:.1f}x, "
          f"Trades {stats.get('trades',0)}{marker}")
    p3_results[name] = {
        "desc": cfg['desc'],
        "pnl": stats['pnl'], "wr": stats.get('wr', 0),
        "mdd": stats.get('mdd', 0), "trades": stats.get('trades', 0),
        "pnl_mdd": round(pm, 2),
    }

# Time-decay cascade variants
print("\n  Time-decay cascade variants (tighten then restore):")
td_variants = [
    ('timedecay_36', 0.15, 36),   # 3h recovery
    ('timedecay_72', 0.15, 72),   # 6h recovery
    ('timedecay_144', 0.15, 144), # 12h recovery
]

for name, bk, rb in td_variants:
    trades, stats = portfolio_npos_timedecay_cascade(
        signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        base_keep=bk, recovery_bars=rb
    )
    pm = pnl_mdd(stats)
    print(f"  {name:16s}: PnL {stats['pnl']:+.1f}%, WR {stats.get('wr',0):.1f}%, "
          f"MDD {stats.get('mdd',0):.2f}%, PnL/MDD {pm:.1f}x, "
          f"Trades {stats.get('trades',0)}  (recovery={rb} bars={rb*5//60}h)")
    p3_results[name] = {
        "desc": f"Time-decay: keep={bk}, recovery={rb}bars ({rb*5//60}h)",
        "pnl": stats['pnl'], "wr": stats.get('wr', 0),
        "mdd": stats.get('mdd', 0), "trades": stats.get('trades', 0),
        "pnl_mdd": round(pm, 2),
    }

# Find best strategy
best_strat = max(p3_results.items(), key=lambda x: x[1]['pnl_mdd'])
print(f"\n  Best strategy: {best_strat[0]} (PnL/MDD {best_strat[1]['pnl_mdd']:.1f}x)")

all_results["phase3_alternatives"] = p3_results


# ═══════════════════════════════════════════════════════════
# PHASE 4: WF for top Phase 3 strategies
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: WF Validation — Top Phase 3 Strategies")
print("=" * 70)

# Get top 3 from Phase 3 (excluding no_cascade and current)
p3_sorted = sorted([(k, v) for k, v in p3_results.items()],
                   key=lambda x: x[1]['pnl_mdd'], reverse=True)
top_to_wf = [k for k, v in p3_sorted[:5] if k != 'no_cascade']

p4_results = {}

for name in top_to_wf:
    # Map strategy name back to sim function
    if name == 'fixed_0.15':
        fn_for_wf = make_fixed_cascade(0.15)
    elif name == 'proportional':
        fn_for_wf = make_proportional_cascade(base_keep=0.15)
    elif name == 'graduated':
        fn_for_wf = make_graduated_cascade([0.50, 0.25, 0.10])
    elif name == 'poscount_2':
        fn_for_wf = make_poscount_cascade(min_pos=2, base_keep=0.15)
    elif name == 'poscount_3':
        fn_for_wf = make_poscount_cascade(min_pos=3, base_keep=0.15)
    elif name.startswith('timedecay_'):
        # For time-decay, use dedicated WF
        rb = int(name.split('_')[1])
        total = ne - ns
        folds_res = []
        for fi in range(3):
            is_end = ns + int(total * (fi + 1) / 4)
            oos_start = is_end
            oos_end = ns + int(total * (fi + 2) / 4)
            if oos_start >= oos_end:
                continue
            trades, stats = portfolio_npos_timedecay_cascade(
                signals, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, oos_start, oos_end,
                base_keep=0.15, recovery_bars=rb
            )
            folds_res.append({
                'fold': fi + 1, 'oos_start': int(oos_start),
                'oos_end': int(oos_end),
                'pnl': stats.get('pnl', 0), 'wr': stats.get('wr', 0),
                'trades': stats.get('trades', 0), 'mdd': stats.get('mdd', 0),
            })
        oos_total = sum(r['pnl'] for r in folds_res)
        wf_pass = all(r['pnl'] > 0 for r in folds_res) and len(folds_res) == 3
        p4_results[name] = {"oos_pnl": round(oos_total, 1), "wf_pass": wf_pass,
                            "folds": folds_res}
        folds_str = ', '.join(f"F{f['fold']}:{f['pnl']:+.1f}%" for f in folds_res)
        print(f"  {name:16s}: OOS {oos_total:+.1f}% {'PASS' if wf_pass else 'FAIL'} [{folds_str}]")
        continue
    else:
        continue

    # Standard custom sim WF
    wf_folds, oos, wf_pass = run_wf_custom(
        portfolio_npos_custom, signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne, n_folds=3,
        cascade_fn=fn_for_wf
    )
    p4_results[name] = {"oos_pnl": round(oos, 1), "wf_pass": wf_pass, "folds": wf_folds}
    marker = " <-- CURRENT" if name == 'fixed_0.15' else ""
    folds_str = ', '.join(f"F{f['fold']}:{f['pnl']:+.1f}%" for f in wf_folds)
    print(f"  {name:16s}: OOS {oos:+.1f}% {'PASS' if wf_pass else 'FAIL'} [{folds_str}]{marker}")

all_results["phase4_wf_validation"] = {k: v for k, v in p4_results.items()}


# ═══════════════════════════════════════════════════════════
# PHASE 5: MC Discrimination (3 seeds)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: MC Discrimination — 3 seeds")
print("=" * 70)

mc_seeds = [42, 123, 7]
mc_results = []

for seed in mc_seeds:
    rng = np.random.RandomState(seed)
    random_signals = []
    for sig_bar, pat, direction, tp_pct, sl_pct in signals:
        rand_dir = rng.choice(['LONG', 'SHORT'])
        random_signals.append((sig_bar, pat, rand_dir, tp_pct, sl_pct))

    _, stats_r_on = run_npos(random_signals, opens, highs, lows, closes, n_bars,
                              atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)
    _, stats_r_off = run_npos(random_signals, opens, highs, lows, closes, n_bars,
                               atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=0)

    delta = stats_r_on['pnl'] - stats_r_off['pnl']
    wf_folds, oos, wf_pass = run_wf(random_signals, opens, highs, lows, closes, n_bars,
                                      atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)

    mc_results.append({
        "seed": seed,
        "random_on_pnl": round(stats_r_on['pnl'], 1),
        "random_off_pnl": round(stats_r_off['pnl'], 1),
        "delta_pnl": round(delta, 1),
        "wf_pass": wf_pass,
        "oos_pnl": round(oos, 1),
    })
    print(f"  Seed {seed}: ON {stats_r_on['pnl']:+.1f}%, OFF {stats_r_off['pnl']:+.1f}%, "
          f"Delta {delta:+.1f}%, WF {'PASS' if wf_pass else 'FAIL'}")

random_helps = sum(1 for r in mc_results if r['delta_pnl'] > 0)
random_wf_pass = sum(1 for r in mc_results if r['wf_pass'])

# Real cascade benefit
real_delta = stats_on['pnl'] - stats_off['pnl']
# Check: do random directions also show cascade benefit? If yes -> mechanical (non-discriminating for patterns)
# But cascade is a MECHANICAL protection, so random helping is expected
disc_note = ("MECHANICAL_IMPROVEMENT" if random_helps == len(mc_seeds) else
             "PATTERN_DEPENDENT" if random_helps == 0 else "MIXED")
print(f"\n  Random cascade helps: {random_helps}/{len(mc_seeds)}")
print(f"  Random WF PASS: {random_wf_pass}/{len(mc_seeds)}")
print(f"  Real delta: {real_delta:+.1f}%, Random avg delta: "
      f"{np.mean([r['delta_pnl'] for r in mc_results]):+.1f}%")
print(f"  Cascade nature: {disc_note}")

all_results["phase5_mc"] = {
    "seeds": mc_results,
    "random_helps": random_helps,
    "random_wf_pass": random_wf_pass,
    "real_delta_pnl": round(real_delta, 1),
    "cascade_nature": disc_note,
}


# ═══════════════════════════════════════════════════════════
# PHASE 6: Summary & Recommendation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: SUMMARY & RECOMMENDATION")
print("=" * 70)

# Phase 1 summary
print(f"\n  Phase 1 (Baseline):")
print(f"    Cascade ON:  IS PnL/MDD {pm_on:.1f}x, OOS {oos_on:+.1f}% {'PASS' if pass_on else 'FAIL'}")
print(f"    Cascade OFF: IS PnL/MDD {pm_off:.1f}x, OOS {oos_off:+.1f}% {'PASS' if pass_off else 'FAIL'}")
print(f"    Events: {total_events}, avg affected: {avg_affected:.1f}/event")

# Phase 2 summary
best_keep_is = sorted_sweep[0]
print(f"\n  Phase 2 (Keep Ratio Sweep):")
print(f"    Best IS: keep={best_keep_is['keep_ratio']} (PnL/MDD {best_keep_is['is_pnl_mdd']:.1f}x)")
current_is = next((r for r in sweep_results if r['keep_ratio'] == 0.15), None)
if current_is:
    print(f"    Current: keep=0.15 (PnL/MDD {current_is['is_pnl_mdd']:.1f}x)")
# Check WF for best sweep
best_keep_wf = p2_wf.get(best_keep_is['keep_ratio'], {})
print(f"    Best WF: {'PASS' if best_keep_wf.get('wf_pass') else 'FAIL'} "
      f"(OOS {best_keep_wf.get('oos_pnl', 0):+.1f}%)")

# Phase 3 summary
print(f"\n  Phase 3 (Alternatives):")
for name, v in sorted(p3_results.items(), key=lambda x: -x[1]['pnl_mdd']):
    marker = " <-- CURRENT" if name == 'fixed_0.15' else ""
    wf_info = p4_results.get(name, {})
    wf_str = f", WF {'PASS' if wf_info.get('wf_pass') else 'FAIL'} OOS {wf_info.get('oos_pnl', 0):+.1f}%" if wf_info else ""
    print(f"    {name:16s}: PnL/MDD {v['pnl_mdd']:6.1f}x{wf_str}{marker}")

# Final recommendation
# Collect all candidates with WF results
candidates = []

# From Phase 2
for keep, wf_data in p2_wf.items():
    keep_f = float(keep)
    is_data = next((r for r in sweep_results if abs(r['keep_ratio'] - keep_f) < 0.001), None)
    if is_data and wf_data.get('wf_pass'):
        candidates.append({
            'name': f"fixed_keep_{keep}",
            'is_pnl_mdd': is_data['is_pnl_mdd'],
            'oos_pnl': wf_data['oos_pnl'],
            'wf_pass': True,
        })

# From Phase 4
for name, wf_data in p4_results.items():
    if wf_data.get('wf_pass'):
        is_data = p3_results.get(name, {})
        candidates.append({
            'name': name,
            'is_pnl_mdd': is_data.get('pnl_mdd', 0),
            'oos_pnl': wf_data['oos_pnl'],
            'wf_pass': True,
        })

current_pm = pm_on  # Current cascade ON PnL/MDD

if candidates:
    best_candidate = max(candidates, key=lambda c: c['is_pnl_mdd'])
    improvement = (best_candidate['is_pnl_mdd'] / current_pm - 1) * 100 if current_pm > 0 else 0

    if best_candidate['name'] != 'fixed_keep_0.15' and improvement > 5:
        recommendation = f"CHANGE to {best_candidate['name']}"
        explanation = (f"IS PnL/MDD {best_candidate['is_pnl_mdd']:.1f}x > current {current_pm:.1f}x "
                       f"(+{improvement:.0f}%), WF PASS, OOS {best_candidate['oos_pnl']:+.1f}%")
    else:
        recommendation = "KEEP_BASELINE (keep=0.15, tighten=85%)"
        if improvement <= 5:
            explanation = f"Best improvement only +{improvement:.1f}% (< 5% threshold)"
        else:
            explanation = "Current is optimal among WF-passing candidates"
else:
    recommendation = "KEEP_BASELINE (keep=0.15, tighten=85%)"
    explanation = "No alternative passes WF validation"

print(f"\n  {'='*50}")
print(f"  RECOMMENDATION: {recommendation}")
print(f"  Reason: {explanation}")
print(f"  {'='*50}")

all_results["recommendation"] = {
    "action": recommendation,
    "reason": explanation,
    "current_keep": 0.15,
    "current_pnl_mdd": round(current_pm, 2),
    "candidates_wf_pass": len(candidates),
    "best_candidate": candidates[0] if candidates else None,
}

elapsed = time.time() - start_time
all_results["elapsed_seconds"] = round(elapsed, 1)
print(f"\n  Elapsed: {elapsed:.1f}s")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Results saved to {OUTPUT_FILE}")
