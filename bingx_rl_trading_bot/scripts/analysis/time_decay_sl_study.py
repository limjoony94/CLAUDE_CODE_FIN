#!/usr/bin/env python3
"""
Time-Decay SL Study — progressively tighten SL as position age increases.

Concept: As a position ages, gradually reduce SL distance toward entry price.
This forces older positions to exit with smaller losses rather than maximum SL.

Sweep: grace_period (bars) x min_sl_fraction (fraction of original SL)
Fixed: full_decay_bars = timeout_bars = 288

Standard Research Protocol:
- Entry: next bar Open after signal
- Exit: Intrabar High/Low (distance-based)
- Sizing: Compound (multiplicative)
- Fee: 0.05% x 2 = 0.10%, Slippage: 0.02%
- Leverage: 3x
- N-pos: 9 slots, direction cap 7, aggregate risk 8/15
- Cascade SL 85%, momentum guard 1.5%/15min/1h, timeout 288
- tp_scale_factor: 0.72
- ATR: period=14, window=576, clamp [0.5, 1.5]
- WF: 3-fold expanding window
- MC: 3 seeds (42, 123, 7), p < 0.01

Output:
- Script: scripts/analysis/time_decay_sl_study.py
- Results: results/time_decay_sl_study.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, compute_atr_ratio, compute_ema_slope,
    calc_stats_compound, find_neutral_window,
    FEE_PCT, LEVERAGE, MAX_BARS, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS,
    DEFAULT_CASCADE_TIGHTEN_PCT,
)
from scripts.production.pattern_5m.indicators import classify_candle

# ============================================================
# Constants
# ============================================================
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'time_decay_sl_study.json')

TP_SCALE_FACTOR = 0.72
MC_SEEDS = [42, 123, 7]
MC_SIMS = 5000
WF_FOLDS = 3

# Sweep parameters
GRACE_PERIODS = [36, 72, 144]  # bars before decay starts (3h, 6h, 12h)
MIN_SL_FRACTIONS = [0.25, 0.50, 0.75]  # minimum SL as fraction of original
FULL_DECAY_BARS = 288  # fixed = timeout_bars


# ============================================================
# Modified N-pos exit check with time-decay SL
# ============================================================
def _check_exit_npos_decay(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee,
                           clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
                           timeout_bars=TIMEOUT_BARS,
                           grace_period=72, min_sl_fraction=0.50,
                           full_decay_bars=FULL_DECAY_BARS):
    """Check exit with time-decay SL.

    Same as scanner _check_exit_npos but with progressive SL tightening:
    - age <= grace_period: no decay (factor = 1.0)
    - grace_period < age < full_decay_bars: linear interpolation to min_sl_fraction
    - age >= full_decay_bars: timeout DROP (same as production)

    Time-decay SL interacts with cascade SL: uses min(cascade_tightened, time_decayed).
    """
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

    if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
    else:
        r = 1.0
    # v1.59.5: cap vol_mult (production parity)
    if sl_pct > 0:
        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)

    eff_tp = tp_pct * r + SLIPPAGE_BUFFER

    # Base SL (before any tightening)
    eff_sl_override = pos.get('eff_sl_override')
    if eff_sl_override is not None:
        base_eff_sl = max(0.1, eff_sl_override - SLIPPAGE_BUFFER)
    else:
        base_eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

    # Time-decay factor
    age_bars = bar - entry_bar
    if age_bars <= grace_period:
        decay_factor = 1.0
    elif age_bars >= full_decay_bars:
        # Timeout
        return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': 0,
                'reason': 'TIMEOUT', 'drop': True}
    else:
        # Linear interpolation from 1.0 to min_sl_fraction
        progress = (age_bars - grace_period) / (full_decay_bars - grace_period)
        decay_factor = 1.0 - progress * (1.0 - min_sl_fraction)

    eff_sl = base_eff_sl * decay_factor
    eff_sl = max(0.1, eff_sl)  # floor

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    # Check timeout (should have been caught above, but safety)
    hold = bar - entry_bar
    if hold >= timeout_bars:
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

    if hit_tp and hit_sl:
        tp_dist = abs(tp_price - opens[bar])
        sl_dist = abs(sl_price - opens[bar])
        if tp_dist <= sl_dist:
            exit_price, reason = tp_price, 'TP'
        else:
            exit_price, reason = sl_price, 'TIME_DECAY_SL' if decay_factor < 1.0 else 'SL'
    elif hit_tp:
        exit_price, reason = tp_price, 'TP'
    else:
        # SL hit — was it because of time decay?
        reason = 'TIME_DECAY_SL' if decay_factor < 1.0 else 'SL'
        exit_price = sl_price

    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * LEVERAGE
    else:
        pnl = (1 - exit_price / entry) * 100 * LEVERAGE
    pnl -= fee

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
            'reason': reason, 'drop': False, 'age_bars': age_bars,
            'decay_factor': decay_factor}


# ============================================================
# Modified portfolio_npos with time-decay SL
# ============================================================
def portfolio_npos_decay(signal_tuples, opens, highs, lows, closes, n_bars,
                         atr_ratio, ema_slope, start_bar, end_bar,
                         n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
                         regime_mult=DEFAULT_REGIME_MULT,
                         agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
                         agg_risk_with=DEFAULT_AGG_RISK_WITH,
                         momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
                         momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
                         momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
                         clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
                         timeout_bars=TIMEOUT_BARS,
                         cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
                         grace_period=72, min_sl_fraction=0.50,
                         full_decay_bars=FULL_DECAY_BARS):
    """N-position portfolio simulator with time-decay SL.

    Identical to scanner portfolio_npos except:
    - _check_exit_npos_decay used instead of _check_exit_npos
    - Tracks TIME_DECAY_SL exits separately
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
    corr_events = []

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Decay-specific counters
    decay_exits = 0
    decay_exit_ages = []
    decay_exit_pnls = []
    normal_sl_exits = 0

    # Filter and sort signals in range
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
            result = _check_exit_npos_decay(pos, bar, opens, highs, lows, n_bars,
                                            atr_ratio, fee, clamp_lo, clamp_hi,
                                            timeout_bars,
                                            grace_period, min_sl_fraction,
                                            full_decay_bars)
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

                if result['reason'] == 'TIME_DECAY_SL':
                    decay_exits += 1
                    decay_exit_ages.append(result.get('age_bars', 0))
                    decay_exit_pnls.append(result['pnl_slot'])
                    bar_sl_count += 1  # still counts as SL for cascade
                elif result['reason'] == 'SL':
                    normal_sl_exits += 1
                    bar_sl_count += 1

        # Cascade SL tightening: after SL exits, tighten same-dir remaining SLs
        # Both SL and TIME_DECAY_SL trigger cascade
        if cascade_tighten_pct > 0 and bar_sl_count > 0:
            keep_ratio = 1.0 - cascade_tighten_pct / 100.0
            sl_directions = set()
            for t in trades[len(trades) - len(closed_slots):]:
                if t.get('reason') in ('SL', 'TIME_DECAY_SL'):
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

        # 2. Process entries (identical to production scanner)
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

        # Mark-to-market MDD
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
                sm = pos.get('size_mult', 1.0)
                mtm_equity += unr * (size_pct / 100) * sm
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
        'mdd_mtm': round(max_dd_mtm, 2),
        'decay_exits': decay_exits,
        'normal_sl_exits': normal_sl_exits,
        'decay_exit_avg_age': round(np.mean(decay_exit_ages), 1) if decay_exit_ages else 0,
        'decay_exit_avg_pnl': round(np.mean(decay_exit_pnls), 3) if decay_exit_pnls else 0,
    }
    return trades, stats


# ============================================================
# Helper: build signal tuples from dynamic patterns
# ============================================================
def build_signal_tuples(patterns_data, signal_index, tp_scale=TP_SCALE_FACTOR):
    """Build signal tuples from dynamic_patterns.json pattern_details.

    pattern_details keys are like 'BD-BD-BU_LONG' with direction suffix.
    Each entry has 'pattern' (bare name) and 'direction' fields.
    """
    pat_details = patterns_data.get('pattern_details') or {}

    tuples = []
    for key, details in pat_details.items():
        pat = details.get('pattern', key.rsplit('_', 1)[0])
        direction = details.get('direction', key.rsplit('_', 1)[-1])
        tp = details['tp'] * tp_scale
        tp = max(0.3, tp)  # floor
        sl = details['sl']

        bars = signal_index.get(pat, [])
        for sig_bar in bars:
            tuples.append((sig_bar, pat, direction, tp, sl))

    return sorted(tuples, key=lambda x: x[0])


# ============================================================
# Monte Carlo sign-randomization test
# ============================================================
def mc_test(trades, seed, n_sims=MC_SIMS):
    """Monte Carlo sign-randomization. Returns p-value."""
    if not trades:
        return 1.0
    pnls = [t['pnl_portfolio'] for t in trades]
    real_sum = sum(pnls)
    rng = np.random.RandomState(seed)
    pnl_arr = np.array(pnls)
    count = 0
    for _ in range(n_sims):
        signs = rng.choice([-1, 1], size=len(pnl_arr))
        if np.sum(pnl_arr * signs) >= real_sum:
            count += 1
    return count / n_sims


# ============================================================
# Walk-forward expanding window
# ============================================================
def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, start_bar, end_bar,
           n_folds=WF_FOLDS, **decay_kwargs):
    """3-fold expanding window WF. Returns list of fold results."""
    n = end_bar - start_bar
    folds = []

    for fi in range(n_folds):
        is_end = start_bar + int(n * (fi + 1) / (n_folds + 1))
        oos_start = is_end
        oos_end = start_bar + int(n * (fi + 2) / (n_folds + 1))

        # IS run (for reference)
        is_trades, is_stats = portfolio_npos_decay(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, is_end,
            **decay_kwargs)
        is_metrics = calc_stats_compound(is_trades)

        # OOS run
        oos_trades, oos_stats = portfolio_npos_decay(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            **decay_kwargs)
        oos_metrics = calc_stats_compound(oos_trades)

        folds.append({
            'fold': fi + 1,
            'is_bars': f"{start_bar}-{is_end}",
            'oos_bars': f"{oos_start}-{oos_end}",
            'is': {**is_metrics, 'mdd_mtm': is_stats['mdd_mtm'],
                   'decay_exits': is_stats['decay_exits']},
            'oos': {**oos_metrics, 'mdd_mtm': oos_stats['mdd_mtm'],
                    'decay_exits': oos_stats['decay_exits']},
            'oos_pass': oos_metrics['pnl'] > 0,
        })

    return folds


# ============================================================
# R:R calculation
# ============================================================
def compute_rr(trades):
    """Compute average reward-to-risk ratio from trades."""
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [abs(t['pnl_slot']) for t in trades if t['pnl_slot'] < 0]
    if not wins or not losses:
        return 0.0
    return np.mean(wins) / np.mean(losses)


# ============================================================
# Main study
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("Time-Decay SL Study")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values.tolist()
    n_bars = len(df)
    print(f"  Bars: {n_bars}")

    # 2. Compute indicators
    print("[2/5] Computing indicators...")
    atr_ratio = compute_atr_ratio(highs, lows, closes, atr_period=14, window=576)
    ema_slope = compute_ema_slope(closes)

    # Neutral window
    nw = find_neutral_window(closes, tol_pct=1.0, min_days=90)
    if nw:
        start_bar, end_bar = nw
        print(f"  Neutral window: bars {start_bar}-{end_bar} ({(end_bar - start_bar) / 288:.0f} days)")
    else:
        start_bar, end_bar = 0, n_bars
        print("  No neutral window found, using full data")

    # 3. Build signals
    print("[3/5] Building signal index...")
    signal_index = build_signal_index(types, n_bars)

    with open(PATTERNS_FILE) as f:
        patterns_data = json.load(f)

    signal_tuples = build_signal_tuples(patterns_data, signal_index)
    print(f"  Total signals: {len(signal_tuples)}")
    print(f"  Patterns: {len(patterns_data.get('pattern_details', {}))}")

    # 4. Run baseline (no decay)
    print("\n[4/5] Running baseline (no decay)...")
    baseline_trades, baseline_stats = portfolio_npos_decay(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        grace_period=999999, min_sl_fraction=1.0,  # effectively no decay
        full_decay_bars=999999)
    baseline_metrics = calc_stats_compound(baseline_trades)
    baseline_rr = compute_rr(baseline_trades)

    # Baseline exit breakdown
    bl_tp = sum(1 for t in baseline_trades if t['reason'] == 'TP')
    bl_sl = sum(1 for t in baseline_trades if t['reason'] == 'SL')
    bl_oos = sum(1 for t in baseline_trades if t['reason'] == 'OOS_END')

    print(f"  Trades: {baseline_metrics['trades']}  WR: {baseline_metrics['wr']}%")
    print(f"  PnL: {baseline_metrics['pnl']:.1f}%  MDD(MTM): {baseline_stats['mdd_mtm']:.2f}%")
    print(f"  PnL/MDD: {baseline_metrics['pnl'] / baseline_stats['mdd_mtm']:.1f}x" if baseline_stats['mdd_mtm'] > 0 else "  PnL/MDD: inf")
    print(f"  R:R: {baseline_rr:.3f}")
    print(f"  Exits — TP: {bl_tp}, SL: {bl_sl}, OOS_END: {bl_oos}")

    baseline_result = {
        'label': 'BASELINE (no decay)',
        'grace_period': None, 'min_sl_fraction': None,
        'metrics': baseline_metrics,
        'mdd_mtm': baseline_stats['mdd_mtm'],
        'pnl_mdd_mtm': round(baseline_metrics['pnl'] / baseline_stats['mdd_mtm'], 2) if baseline_stats['mdd_mtm'] > 0 else 0,
        'rr': round(baseline_rr, 4),
        'exits': {'TP': bl_tp, 'SL': bl_sl, 'TIME_DECAY_SL': 0, 'OOS_END': bl_oos},
        'decay_exits': 0,
        'normal_sl_exits': bl_sl,
    }

    # 5. Sweep grid
    print(f"\n[5/5] Running sweep: {len(GRACE_PERIODS)} grace x {len(MIN_SL_FRACTIONS)} fractions = {len(GRACE_PERIODS) * len(MIN_SL_FRACTIONS)} combos...")

    results = []
    for gp in GRACE_PERIODS:
        for msf in MIN_SL_FRACTIONS:
            label = f"gp={gp}({gp*5/60:.0f}h)_msf={msf}"
            print(f"\n  --- {label} ---")

            trades, stats = portfolio_npos_decay(
                signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, start_bar, end_bar,
                grace_period=gp, min_sl_fraction=msf,
                full_decay_bars=FULL_DECAY_BARS)

            metrics = calc_stats_compound(trades)
            rr = compute_rr(trades)

            # Exit breakdown
            n_tp = sum(1 for t in trades if t['reason'] == 'TP')
            n_sl = sum(1 for t in trades if t['reason'] == 'SL')
            n_decay = sum(1 for t in trades if t['reason'] == 'TIME_DECAY_SL')
            n_oos = sum(1 for t in trades if t['reason'] == 'OOS_END')

            # TP preservation: how many TPs in baseline are still TPs here?
            # (approximate: compare TP counts)
            tp_preserved_pct = (n_tp / bl_tp * 100) if bl_tp > 0 else 0

            pnl_mdd = metrics['pnl'] / stats['mdd_mtm'] if stats['mdd_mtm'] > 0 else 0

            print(f"    Trades: {metrics['trades']}  WR: {metrics['wr']}%")
            print(f"    PnL: {metrics['pnl']:.1f}%  MDD(MTM): {stats['mdd_mtm']:.2f}%  PnL/MDD: {pnl_mdd:.1f}x")
            print(f"    R:R: {rr:.3f}")
            print(f"    Exits — TP: {n_tp}, SL: {n_sl}, TIME_DECAY_SL: {n_decay}, OOS_END: {n_oos}")
            print(f"    TP preserved: {tp_preserved_pct:.1f}%")
            print(f"    Avg decay exit age: {stats['decay_exit_avg_age']:.0f} bars ({stats['decay_exit_avg_age']*5/60:.1f}h)")
            print(f"    Avg decay exit PnL: {stats['decay_exit_avg_pnl']:.3f}%")

            # Delta vs baseline
            dpnl = metrics['pnl'] - baseline_metrics['pnl']
            dmdd = stats['mdd_mtm'] - baseline_stats['mdd_mtm']
            dwr = metrics['wr'] - baseline_metrics['wr']

            results.append({
                'label': label,
                'grace_period': gp,
                'grace_period_hours': gp * 5 / 60,
                'min_sl_fraction': msf,
                'full_decay_bars': FULL_DECAY_BARS,
                'metrics': metrics,
                'mdd_mtm': stats['mdd_mtm'],
                'pnl_mdd_mtm': round(pnl_mdd, 2),
                'rr': round(rr, 4),
                'exits': {'TP': n_tp, 'SL': n_sl, 'TIME_DECAY_SL': n_decay, 'OOS_END': n_oos},
                'decay_exits': n_decay,
                'normal_sl_exits': n_sl,
                'decay_exit_avg_age': stats['decay_exit_avg_age'],
                'decay_exit_avg_pnl': stats['decay_exit_avg_pnl'],
                'tp_preserved_pct': round(tp_preserved_pct, 1),
                'delta_vs_baseline': {
                    'pnl': round(dpnl, 2),
                    'mdd_mtm': round(dmdd, 2),
                    'wr': round(dwr, 1),
                },
            })

    # Sort by PnL/MDD
    results.sort(key=lambda x: x['pnl_mdd_mtm'], reverse=True)

    # Top 3 for WF + MC
    print("\n" + "=" * 70)
    print("Top 3 by PnL/MDD(MTM) — running WF + MC")
    print("=" * 70)

    top3 = results[:3]
    for rank, r in enumerate(top3, 1):
        gp = r['grace_period']
        msf = r['min_sl_fraction']
        print(f"\n--- #{rank}: {r['label']} (PnL/MDD {r['pnl_mdd_mtm']}x) ---")

        decay_kwargs = dict(
            grace_period=gp, min_sl_fraction=msf,
            full_decay_bars=FULL_DECAY_BARS)

        # WF
        folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, start_bar, end_bar,
                       n_folds=WF_FOLDS, **decay_kwargs)

        total_oos_pnl = sum(f['oos']['pnl'] for f in folds)
        all_pass = all(f['oos_pass'] for f in folds)

        print(f"  WF: {'PASS' if all_pass else 'FAIL'} ({sum(f['oos_pass'] for f in folds)}/{WF_FOLDS})")
        for f in folds:
            print(f"    Fold {f['fold']}: OOS PnL {f['oos']['pnl']:.1f}%  WR {f['oos']['wr']}%  Decay exits: {f['oos']['decay_exits']}")
        print(f"  Total OOS PnL: {total_oos_pnl:.1f}%")

        r['wf'] = {
            'folds': folds,
            'total_oos_pnl': round(total_oos_pnl, 2),
            'all_pass': all_pass,
            'pass_count': sum(f['oos_pass'] for f in folds),
        }

        # MC (on IS trades)
        is_trades, _ = portfolio_npos_decay(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            **decay_kwargs)
        mc_pvals = []
        for seed in MC_SEEDS:
            p = mc_test(is_trades, seed)
            mc_pvals.append(p)
        max_p = max(mc_pvals)
        print(f"  MC p-values: {mc_pvals} max={max_p:.4f} {'PASS' if max_p < 0.01 else 'FAIL'}")

        r['mc'] = {
            'seeds': MC_SEEDS,
            'p_values': mc_pvals,
            'max_p': max_p,
            'pass': max_p < 0.01,
        }

    # Run WF + MC on baseline too
    print("\n--- BASELINE WF + MC ---")
    baseline_decay_kwargs = dict(
        grace_period=999999, min_sl_fraction=1.0, full_decay_bars=999999)
    baseline_folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, start_bar, end_bar,
                            n_folds=WF_FOLDS, **baseline_decay_kwargs)
    baseline_oos_pnl = sum(f['oos']['pnl'] for f in baseline_folds)
    baseline_all_pass = all(f['oos_pass'] for f in baseline_folds)
    print(f"  WF: {'PASS' if baseline_all_pass else 'FAIL'} ({sum(f['oos_pass'] for f in baseline_folds)}/{WF_FOLDS})")
    for f in baseline_folds:
        print(f"    Fold {f['fold']}: OOS PnL {f['oos']['pnl']:.1f}%  WR {f['oos']['wr']}%")
    print(f"  Total OOS PnL: {baseline_oos_pnl:.1f}%")

    baseline_mc = []
    for seed in MC_SEEDS:
        p = mc_test(baseline_trades, seed)
        baseline_mc.append(p)
    print(f"  MC p-values: {baseline_mc} max={max(baseline_mc):.4f}")

    baseline_result['wf'] = {
        'folds': baseline_folds,
        'total_oos_pnl': round(baseline_oos_pnl, 2),
        'all_pass': baseline_all_pass,
        'pass_count': sum(f['oos_pass'] for f in baseline_folds),
    }
    baseline_result['mc'] = {
        'seeds': MC_SEEDS,
        'p_values': baseline_mc,
        'max_p': max(baseline_mc),
        'pass': max(baseline_mc) < 0.01,
    }

    # Discrimination test: does best beat baseline OOS?
    print("\n" + "=" * 70)
    print("DISCRIMINATION TEST")
    print("=" * 70)
    best = top3[0] if top3 else None
    if best and best.get('wf'):
        gap = best['wf']['total_oos_pnl'] - baseline_oos_pnl
        print(f"  Best: {best['label']}")
        print(f"  Best OOS: {best['wf']['total_oos_pnl']:.1f}%  vs  Baseline OOS: {baseline_oos_pnl:.1f}%")
        print(f"  Gap: {gap:+.1f}%")
        disc = gap > 0 and best['wf']['all_pass'] and best.get('mc', {}).get('pass', False)
        print(f"  Verdict: {'DISCRIMINATING' if disc else 'NON-DISCRIMINATING'}")
    else:
        gap = 0
        disc = False
        print("  No valid candidates")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Config':<30} {'Trades':>6} {'WR':>6} {'PnL':>8} {'MDD':>7} {'PnL/MDD':>8} {'R:R':>6} {'DecaySL':>7} {'TP%':>5}")
    print("-" * 95)
    bl = baseline_result
    print(f"{'BASELINE':<30} {bl['metrics']['trades']:>6} {bl['metrics']['wr']:>5.1f}% {bl['metrics']['pnl']:>7.1f}% {bl['mdd_mtm']:>6.2f}% {bl['pnl_mdd_mtm']:>7.1f}x {bl['rr']:>5.3f} {0:>7} {'100.0':>5}")
    for r in results:
        print(f"{r['label']:<30} {r['metrics']['trades']:>6} {r['metrics']['wr']:>5.1f}% {r['metrics']['pnl']:>7.1f}% {r['mdd_mtm']:>6.2f}% {r['pnl_mdd_mtm']:>7.1f}x {r['rr']:>5.3f} {r['decay_exits']:>7} {r['tp_preserved_pct']:>4.1f}%")

    elapsed = time.time() - t0

    # Save
    output = {
        'study': 'time_decay_sl',
        'generated_at': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data_file': DATA_FILE,
        'data_bars': n_bars,
        'neutral_window': {'start': start_bar, 'end': end_bar,
                           'days': round((end_bar - start_bar) / 288, 1)},
        'parameters': {
            'tp_scale_factor': TP_SCALE_FACTOR,
            'grace_periods': GRACE_PERIODS,
            'min_sl_fractions': MIN_SL_FRACTIONS,
            'full_decay_bars': FULL_DECAY_BARS,
            'timeout_bars': TIMEOUT_BARS,
            'n_slots': DEFAULT_N_SLOTS,
            'direction_cap': DEFAULT_DIRECTION_CAP,
            'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
            'wf_folds': WF_FOLDS,
            'mc_seeds': MC_SEEDS,
        },
        'baseline': baseline_result,
        'sweep_results': results,
        'top3_labels': [r['label'] for r in top3],
        'discrimination': {
            'best_label': best['label'] if best else None,
            'best_oos_pnl': best['wf']['total_oos_pnl'] if best and best.get('wf') else None,
            'baseline_oos_pnl': baseline_oos_pnl,
            'gap': round(gap, 2),
            'discriminating': disc,
        },
        'verdict': 'GO' if disc else 'KEEP_BASELINE',
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Final verdict
    print("\n" + "=" * 70)
    v = 'GO' if disc else 'KEEP_BASELINE'
    print(f"VERDICT: {v}")
    print("=" * 70)


if __name__ == '__main__':
    main()
