#!/usr/bin/env python3
"""
Time-Decay TP Exit Strategy Research
=====================================
Hypothesis: As holding time increases, pattern signal validity decays.
Gradually reducing TP should: (1) reflect time-value decay, (2) capture
partial profits, (3) accelerate slot turnover.

Variants:
  - baseline:     Fixed TP, 288-bar timeout (current production)
  - linear_144:   Linear decay from bar 144 (12h) to 288, TP 100%->50%
  - linear_96:    Linear decay from bar 96 (8h) to 288, TP 100%->50%
  - step_2:       Step: TP*0.75 at 96 bars, TP*0.50 at 192 bars
  - step_3:       Step: TP*0.85 at 72, TP*0.65 at 144, TP*0.45 at 216
  - aggressive:   Linear decay from bar 72 (6h) to 216 (18h), TP 100%->40%
  - conservative: Linear decay from bar 192 (16h) to 288, TP 100%->60%
  - exp_decay:    Exponential decay: TP * 0.997^bars_held (half-life ~231 bars)

Protocol: Standard Research Protocol
  - Entry: next-bar open after signal
  - Exit: intrabar high/low (distance-based, same-bar resolution)
  - Fee: FEE_PCT * LEVERAGE (0.10% * 3 = 0.30%)
  - Timeout: 288 bars (DROP from PnL)
  - Leverage: 3x
  - Compound sizing
  - Production classify_candle import

Output: results/time_decay_tp.json
"""

import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Production imports (CRITICAL: never self-implement classify_candle)
from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

from scripts.scanner.pattern_scanner import (
    load_and_classify,
    find_neutral_window,
    build_signal_index,
    compute_atr_ratio,
    compute_ema_slope,
    calc_stats_compound,
    FEE_PCT,
    LEVERAGE,
    MAX_BARS,
    SLIPPAGE_BUFFER,
    MAX_DAILY_LOSS_PCT,
    DEFAULT_N_SLOTS,
    DEFAULT_DIRECTION_CAP,
    DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER,
    DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK,
    DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO,
    DEFAULT_ATR_CLAMP_HI,
    TIMEOUT_BARS,
)

# ============================================================
# Constants
# ============================================================
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'time_decay_tp.json')
TP_SCALE_FACTOR = 0.72  # v1.61.0: post-discovery TP scaling


# ============================================================
# Decay Functions
# ============================================================
def decay_none(tp_pct, bars_held, max_bars):
    """No decay (baseline)."""
    return tp_pct


def decay_linear_144(tp_pct, bars_held, max_bars):
    """Linear decay starting at bar 144 (12h), reaching 50% at bar 288."""
    start = 144
    if bars_held <= start:
        return tp_pct
    progress = min(1.0, (bars_held - start) / (max_bars - start))
    return tp_pct * (1.0 - 0.50 * progress)


def decay_linear_96(tp_pct, bars_held, max_bars):
    """Linear decay starting at bar 96 (8h), reaching 50% at bar 288."""
    start = 96
    if bars_held <= start:
        return tp_pct
    progress = min(1.0, (bars_held - start) / (max_bars - start))
    return tp_pct * (1.0 - 0.50 * progress)


def decay_step_2(tp_pct, bars_held, max_bars):
    """Two-step: TP*0.75 at 96 bars, TP*0.50 at 192 bars."""
    if bars_held >= 192:
        return tp_pct * 0.50
    if bars_held >= 96:
        return tp_pct * 0.75
    return tp_pct


def decay_step_3(tp_pct, bars_held, max_bars):
    """Three-step: TP*0.85 at 72, TP*0.65 at 144, TP*0.45 at 216."""
    if bars_held >= 216:
        return tp_pct * 0.45
    if bars_held >= 144:
        return tp_pct * 0.65
    if bars_held >= 72:
        return tp_pct * 0.85
    return tp_pct


def decay_aggressive(tp_pct, bars_held, max_bars):
    """Aggressive: linear from bar 72 to 216, 100%->40%."""
    start, end = 72, 216
    if bars_held <= start:
        return tp_pct
    if bars_held >= end:
        return tp_pct * 0.40
    progress = (bars_held - start) / (end - start)
    return tp_pct * (1.0 - 0.60 * progress)


def decay_conservative(tp_pct, bars_held, max_bars):
    """Conservative: linear from bar 192 to 288, 100%->60%."""
    start = 192
    if bars_held <= start:
        return tp_pct
    progress = min(1.0, (bars_held - start) / (max_bars - start))
    return tp_pct * (1.0 - 0.40 * progress)


def decay_exponential(tp_pct, bars_held, max_bars):
    """Exponential decay: TP * 0.997^bars_held. Half-life ~231 bars (19.25h)."""
    return tp_pct * (0.997 ** bars_held)


DECAY_VARIANTS = {
    'baseline':     decay_none,
    'linear_144':   decay_linear_144,
    'linear_96':    decay_linear_96,
    'step_2':       decay_step_2,
    'step_3':       decay_step_3,
    'aggressive':   decay_aggressive,
    'conservative': decay_conservative,
    'exp_decay':    decay_exponential,
}


# ============================================================
# Modified N-pos exit check with time-decay TP
# ============================================================
def check_exit_decay(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee,
                     clamp_lo, clamp_hi, timeout_bars, decay_fn):
    """Check exit with time-decaying TP. Mirrors _check_exit_npos exactly
    except TP is adjusted by decay_fn(bars_held)."""
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None
    entry = opens[entry_bar]
    if entry <= 0:
        return None

    tp_pct = pos['tp_pct']
    sl_pct = pos['sl_pct']
    direction = pos['direction']
    sig_bar = pos['signal_bar']

    # ATR scaling
    if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
    else:
        r = 1.0
    # Daily loss cap
    if sl_pct > 0:
        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)

    # --- TIME DECAY: adjust TP based on bars held ---
    bars_held = bar - entry_bar
    decayed_tp_pct = decay_fn(tp_pct, bars_held, timeout_bars)

    eff_tp = decayed_tp_pct * r + SLIPPAGE_BUFFER

    # Cascade SL override
    eff_sl_override = pos.get('eff_sl_override')
    if eff_sl_override is not None:
        eff_sl = max(0.1, eff_sl_override - SLIPPAGE_BUFFER)
    else:
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    # Timeout = DROP
    if bars_held >= timeout_bars:
        return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': 0,
                'reason': 'TIMEOUT', 'drop': True}

    h, l = highs[bar], lows[bar]
    if direction == 'LONG':
        hit_tp = h >= tp_price
        hit_sl = l <= sl_price
    else:
        hit_tp = l <= tp_price
        hit_sl = h >= sl_price

    if not hit_tp and not hit_sl:
        return None

    # Same-bar resolution (distance from bar open)
    if hit_tp and hit_sl:
        tp_dist = abs(tp_price - opens[bar])
        sl_dist = abs(sl_price - opens[bar])
        if tp_dist <= sl_dist:
            exit_price, reason = tp_price, 'TP'
        else:
            exit_price, reason = sl_price, 'SL'
    elif hit_tp:
        exit_price, reason = tp_price, 'TP'
    else:
        exit_price, reason = sl_price, 'SL'

    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * LEVERAGE
    else:
        pnl = (1 - exit_price / entry) * 100 * LEVERAGE
    pnl -= fee

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
            'reason': reason, 'drop': False}


# ============================================================
# N-pos portfolio simulator with time-decay TP
# ============================================================
def portfolio_npos_decay(signal_tuples, opens, highs, lows, closes, n_bars,
                         atr_ratio, ema_slope, start_bar, end_bar,
                         decay_fn,
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
                         cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT):
    """N-position portfolio simulator with time-decaying TP.

    Mirrors portfolio_npos() exactly, except uses check_exit_decay()
    with the specified decay_fn for TP adjustment.

    Returns: (trades_list, stats_dict)
    """
    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd_mtm = 0.0

    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0, 'max_pos': 0}
    total_timeouts = 0
    total_holding_bars = 0

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Filter and sort signals
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
            result = check_exit_decay(pos, bar, opens, highs, lows, n_bars,
                                      atr_ratio, fee, clamp_lo, clamp_hi,
                                      timeout_bars, decay_fn)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    total_timeouts += 1
                    total_holding_bars += (bar - pos['entry_bar'])
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
                total_holding_bars += (result['exit_bar'] - result['entry_bar'])
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
                    p_sl = pos['sl_pct']
                    if p_sl > 0:
                        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                    cur_eff_sl = pos.get('eff_sl_override') or (p_sl * r)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]

        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum)
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct

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

            # Aggregate risk cap
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
                        if p_sl > 0:
                            p_r = min(p_r, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / n_slots) * LEVERAGE * p_sm

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
                if sl_pct > 0:
                    new_r = min(new_r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)
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

        # MTM MDD
        if positions and bar < n_bars:
            mtm_equity = equity
            for pos in positions:
                eb = pos['entry_bar']
                if eb >= n_bars or bar < eb:
                    continue
                entry_price = opens[eb]
                if entry_price <= 0:
                    continue
                if pos['direction'] == 'LONG':
                    unr = (closes[bar] / entry_price - 1) * 100 * LEVERAGE
                else:
                    unr = (1 - closes[bar] / entry_price) * 100 * LEVERAGE
                sm_pos = pos.get('size_mult', 1.0)
                mtm_equity += unr * (size_pct / 100) * sm_pos
            if mtm_equity > peak_equity:
                peak_equity = mtm_equity
            dd = (peak_equity - mtm_equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd
        elif not positions:
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd

    # Force-close remaining (OOS end)
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
        sm_pos = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm_pos,
            'pnl_portfolio': pnl * (size_pct / 100) * sm_pos,
        })
        total_holding_bars += (exit_bar - entry_bar)

    n_total = len(trades) + total_timeouts
    avg_hold = total_holding_bars / n_total if n_total > 0 else 0

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'mdd_mtm': round(max_dd_mtm, 2),
        'blocked': total_blocked,
        'timeouts': total_timeouts,
        'avg_holding_bars': round(avg_hold, 1),
    }
    return trades, stats


# ============================================================
# Load data and patterns
# ============================================================
def load_data_and_patterns():
    """Load market data, classify, compute indicators, load pattern config."""
    print("Loading and classifying data...")
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    type_codes = df['candle_type'].tolist()

    print(f"  Data: {n_bars} bars ({n_bars/288:.0f} days)")

    # Neutral window
    nw = find_neutral_window(closes, tol_pct=1.0)
    if nw is None:
        raise RuntimeError("No neutral window found")
    nw_start, nw_end = nw
    print(f"  Neutral window: [{nw_start}, {nw_end}] ({(nw_end-nw_start)/288:.0f} days)")

    # Signal index
    signal_index = build_signal_index(type_codes, n_bars)

    # ATR ratio + EMA slope
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    # Load dynamic patterns
    with open(PATTERNS_FILE, 'r') as f:
        dp = json.load(f)

    pattern_details = dp['pattern_details']

    # Build signal tuples with TP_SCALE_FACTOR applied
    signal_tuples = []
    for key, info in pattern_details.items():
        pat = info['pattern']
        direction = info['direction']
        tp = info['tp'] * TP_SCALE_FACTOR  # v1.57.0: TP * 0.5
        sl = info['sl']

        if pat not in signal_index:
            continue
        for sig_bar in signal_index[pat]:
            if nw_start <= sig_bar < nw_end:
                signal_tuples.append((sig_bar, pat, direction, tp, sl))

    signal_tuples.sort(key=lambda x: x[0])
    print(f"  Patterns: {len(pattern_details)} | Signals in NW: {len(signal_tuples)}")

    return {
        'n_bars': n_bars,
        'opens': opens,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'atr_ratio': atr_ratio,
        'ema_slope': ema_slope,
        'signal_tuples': signal_tuples,
        'nw_start': nw_start,
        'nw_end': nw_end,
        'signal_index': signal_index,
        'pattern_details': pattern_details,
        'type_codes': type_codes,
    }


# ============================================================
# Phase 1: IS Screening (all variants, N-pos)
# ============================================================
def run_is_screening(data):
    """Run all decay variants on IS (full neutral window) with N-pos."""
    print("\n" + "="*60)
    print("PHASE 1: IS Screening (N-pos, all variants)")
    print("="*60)

    results = {}
    for name, decay_fn in DECAY_VARIANTS.items():
        t0 = time.time()
        trades, npos_stats = portfolio_npos_decay(
            data['signal_tuples'],
            data['opens'], data['highs'], data['lows'], data['closes'],
            data['n_bars'], data['atr_ratio'], data['ema_slope'],
            data['nw_start'], data['nw_end'],
            decay_fn=decay_fn,
        )
        stats = calc_stats_compound(trades)
        elapsed = time.time() - t0

        # Exit reason breakdown
        tp_count = sum(1 for t in trades if t.get('reason') == 'TP')
        sl_count = sum(1 for t in trades if t.get('reason') == 'SL')

        results[name] = {
            'trades': stats['trades'],
            'wr': stats['wr'],
            'pnl': stats['pnl'],
            'mdd': stats['mdd'],
            'pnl_mdd': stats['pnl_mdd'],
            'mdd_mtm': npos_stats['mdd_mtm'],
            'timeouts': npos_stats['timeouts'],
            'avg_holding_bars': npos_stats['avg_holding_bars'],
            'tp_exits': tp_count,
            'sl_exits': sl_count,
            'blocked': npos_stats['blocked'],
            'elapsed_s': round(elapsed, 1),
        }

        pnl_mdd_mtm = stats['pnl'] / npos_stats['mdd_mtm'] if npos_stats['mdd_mtm'] > 0 else 0
        print(f"  {name:18s} | trades={stats['trades']:4d} WR={stats['wr']:5.1f}% "
              f"PnL={stats['pnl']:+8.1f}% MDD(mtm)={npos_stats['mdd_mtm']:5.2f}% "
              f"PnL/MDD={pnl_mdd_mtm:6.1f}x "
              f"TO={npos_stats['timeouts']:4d} hold={npos_stats['avg_holding_bars']:5.1f}b "
              f"TP={tp_count} SL={sl_count} [{elapsed:.1f}s]")

    return results


# ============================================================
# Phase 2: WF Validation (top variants)
# ============================================================
def run_wf_validation(data, is_results, n_folds=3, top_n=3, force_include=None):
    """Run expanding-window WF on top variants by PnL/MDD(MTM)."""
    print("\n" + "="*60)
    print(f"PHASE 2: Walk-Forward Validation ({n_folds}-fold, top {top_n} variants)")
    print("="*60)

    # Rank by PnL/MDD(MTM)
    ranked = []
    for name, r in is_results.items():
        pnl_mdd = r['pnl'] / r['mdd_mtm'] if r['mdd_mtm'] > 0 else 0
        ranked.append((name, pnl_mdd, r['pnl']))
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Always include baseline + top variants + forced
    top_names = set()
    top_names.add('baseline')
    for name, _, _ in ranked[:top_n]:
        top_names.add(name)
    if force_include:
        for name in force_include:
            if name in is_results:
                top_names.add(name)
    top_names = list(top_names)

    print(f"  Selected for WF: {top_names}")

    nw_start = data['nw_start']
    nw_end = data['nw_end']
    nw_bars = nw_end - nw_start
    seg_size = nw_bars // (n_folds + 1)

    wf_results = {}

    for variant_name in top_names:
        decay_fn = DECAY_VARIANTS[variant_name]
        print(f"\n  --- {variant_name} ---")

        fold_results = []
        for fold in range(n_folds):
            is_end = nw_start + (fold + 1) * seg_size
            oos_start = is_end
            oos_end = nw_start + (fold + 2) * seg_size if fold < n_folds - 1 else nw_end

            if oos_start >= oos_end:
                print(f"    Fold {fold+1}: SKIP (no OOS bars)")
                continue

            # Rebuild signal tuples for OOS range
            oos_signal_tuples = []
            for key, info in data['pattern_details'].items():
                pat = info['pattern']
                direction = info['direction']
                tp = info['tp'] * TP_SCALE_FACTOR
                sl = info['sl']
                if pat not in data['signal_index']:
                    continue
                for sig_bar in data['signal_index'][pat]:
                    if oos_start <= sig_bar < oos_end:
                        oos_signal_tuples.append((sig_bar, pat, direction, tp, sl))

            if not oos_signal_tuples:
                print(f"    Fold {fold+1}: SKIP (no OOS signals)")
                fold_results.append({'fold': fold+1, 'oos_pnl': 0, 'oos_trades': 0})
                continue

            # Compute ATR/EMA on data up to oos_end (no look-ahead: ATR uses IS data)
            oos_atr = compute_atr_ratio(
                data['highs'][:oos_end], data['lows'][:oos_end], data['closes'][:oos_end])
            oos_ema = compute_ema_slope(data['closes'][:oos_end])

            trades, npos_stats = portfolio_npos_decay(
                oos_signal_tuples,
                data['opens'], data['highs'], data['lows'], data['closes'],
                oos_end, oos_atr, oos_ema,
                oos_start, oos_end,
                decay_fn=decay_fn,
            )
            stats = calc_stats_compound(trades)

            fold_results.append({
                'fold': fold + 1,
                'is_bars': is_end - nw_start,
                'oos_bars': oos_end - oos_start,
                'oos_trades': stats['trades'],
                'oos_wr': stats['wr'],
                'oos_pnl': stats['pnl'],
                'oos_mdd': stats['mdd'],
                'oos_mdd_mtm': npos_stats['mdd_mtm'],
                'timeouts': npos_stats['timeouts'],
                'avg_holding_bars': npos_stats['avg_holding_bars'],
            })

            print(f"    Fold {fold+1}: OOS trades={stats['trades']:3d} "
                  f"WR={stats['wr']:5.1f}% PnL={stats['pnl']:+7.1f}% "
                  f"MDD(mtm)={npos_stats['mdd_mtm']:5.2f}% "
                  f"TO={npos_stats['timeouts']} hold={npos_stats['avg_holding_bars']:.0f}b")

        total_oos = sum(f['oos_pnl'] for f in fold_results)
        n_pass = sum(1 for f in fold_results if f['oos_pnl'] > 0)
        verdict = "PASS" if n_pass == n_folds else "FAIL"

        wf_results[variant_name] = {
            'folds': fold_results,
            'total_oos_pnl': round(total_oos, 1),
            'n_pass': n_pass,
            'n_folds': n_folds,
            'verdict': verdict,
        }
        print(f"    WF: {n_pass}/{n_folds} PASS | Total OOS: {total_oos:+.1f}% | {verdict}")

    return wf_results


# ============================================================
# Phase 3: Discrimination test (baseline vs best variant)
# ============================================================
def run_discrimination_test(data, is_results, best_variant):
    """Quick discrimination: compare best variant vs baseline on IS.
    Report the gap and direction."""
    baseline = is_results.get('baseline', {})
    best = is_results.get(best_variant, {})

    b_pnl_mdd = baseline['pnl'] / baseline['mdd_mtm'] if baseline.get('mdd_mtm', 0) > 0 else 0
    v_pnl_mdd = best['pnl'] / best['mdd_mtm'] if best.get('mdd_mtm', 0) > 0 else 0

    gap_pnl = best['pnl'] - baseline['pnl']
    gap_mdd = best['mdd_mtm'] - baseline['mdd_mtm']
    gap_pnl_mdd = v_pnl_mdd - b_pnl_mdd
    gap_hold = best['avg_holding_bars'] - baseline['avg_holding_bars']

    return {
        'baseline_pnl': baseline['pnl'],
        'baseline_mdd_mtm': baseline['mdd_mtm'],
        'baseline_pnl_mdd': round(b_pnl_mdd, 1),
        'variant_pnl': best['pnl'],
        'variant_mdd_mtm': best['mdd_mtm'],
        'variant_pnl_mdd': round(v_pnl_mdd, 1),
        'gap_pnl': round(gap_pnl, 1),
        'gap_mdd': round(gap_mdd, 2),
        'gap_pnl_mdd': round(gap_pnl_mdd, 1),
        'gap_avg_hold_bars': round(gap_hold, 1),
    }


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("Time-Decay TP Exit Strategy Research")
    print("="*60)

    # Load data
    data = load_data_and_patterns()

    # Phase 1: IS screening
    is_results = run_is_screening(data)

    # Rank by PnL/MDD(MTM)
    ranked = []
    for name, r in is_results.items():
        pnl_mdd = r['pnl'] / r['mdd_mtm'] if r['mdd_mtm'] > 0 else 0
        ranked.append((name, round(pnl_mdd, 1)))
    ranked.sort(key=lambda x: x[1], reverse=True)

    print("\n--- IS Ranking by PnL/MDD(MTM) ---")
    for i, (name, pnl_mdd) in enumerate(ranked):
        marker = " <-- BASELINE" if name == 'baseline' else ""
        print(f"  {i+1}. {name:18s} PnL/MDD(mtm) = {pnl_mdd:6.1f}x{marker}")

    # Best non-baseline variant
    best_variant = ranked[0][0]
    if best_variant == 'baseline':
        best_variant = ranked[1][0] if len(ranked) > 1 else 'baseline'
    best_pnl_mdd = ranked[0][1]

    # Phase 2: WF validation (top 3 + baseline + exp_decay forced)
    wf_results = run_wf_validation(data, is_results, n_folds=3, top_n=3,
                                   force_include=['exp_decay', 'step_2', 'aggressive'])

    # Phase 3: Discrimination
    disc = run_discrimination_test(data, is_results, best_variant)

    print("\n" + "="*60)
    print("DISCRIMINATION: baseline vs", best_variant)
    print(f"  PnL gap:     {disc['gap_pnl']:+.1f}%")
    print(f"  MDD gap:     {disc['gap_mdd']:+.2f}% (MTM)")
    print(f"  PnL/MDD gap: {disc['gap_pnl_mdd']:+.1f}x")
    print(f"  Avg hold gap: {disc['gap_avg_hold_bars']:+.1f} bars")

    # Determine recommendation
    baseline_wf = wf_results.get('baseline', {})
    best_wf = wf_results.get(best_variant, {})

    recommendation = "KEEP_BASELINE"
    reason = []

    # Check ALL WF-tested variants vs baseline
    base_oos = baseline_wf.get('total_oos_pnl', 0)
    best_oos_variant = None
    best_oos_pnl = base_oos

    for vname, vwf in wf_results.items():
        if vname == 'baseline':
            continue
        if vwf.get('verdict') == 'PASS':
            v_oos = vwf.get('total_oos_pnl', 0)
            if v_oos > best_oos_pnl:
                best_oos_pnl = v_oos
                best_oos_variant = vname

    if best_oos_variant:
        gap_pct = (best_oos_pnl - base_oos) / base_oos * 100 if base_oos > 0 else 0
        recommendation = f"CONSIDER {best_oos_variant}"
        reason.append(f"OOS {best_oos_variant}: +{best_oos_pnl:.1f}% vs baseline +{base_oos:.1f}% (gap +{gap_pct:.1f}%)")
        # Also check IS PnL/MDD improvement
        v_is = is_results.get(best_oos_variant, {})
        b_is = is_results.get('baseline', {})
        v_pnl_mdd = v_is['pnl'] / v_is['mdd_mtm'] if v_is.get('mdd_mtm', 0) > 0 else 0
        b_pnl_mdd = b_is['pnl'] / b_is['mdd_mtm'] if b_is.get('mdd_mtm', 0) > 0 else 0
        reason.append(f"IS PnL/MDD: {v_pnl_mdd:.1f}x vs baseline {b_pnl_mdd:.1f}x")
    else:
        reason.append(f"No variant beats baseline OOS ({base_oos:.1f}%)")

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {recommendation}")
    for r in reason:
        print(f"  - {r}")
    print(f"Total time: {elapsed:.0f}s")

    # Save results
    output = {
        'study': 'time_decay_tp',
        'date': datetime.now().isoformat(),
        'script': 'scripts/analysis/time_decay_tp.py',
        'data_file': DATA_FILE,
        'data_bars': data['n_bars'],
        'neutral_window': [data['nw_start'], data['nw_end']],
        'tp_scale_factor': TP_SCALE_FACTOR,
        'n_patterns': len(data['pattern_details']),
        'n_signals': len(data['signal_tuples']),
        'variants': list(DECAY_VARIANTS.keys()),
        'is_results': is_results,
        'is_ranking': ranked,
        'wf_results': wf_results,
        'discrimination': disc,
        'recommendation': recommendation,
        'reason': reason,
        'elapsed_s': round(elapsed, 1),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
