#!/usr/bin/env python3
"""
Cascade SL Depth Study
========================

Question: Is the 75% Cascade SL Tightening (v1.41.0) functionally
equivalent to an immediate market close of all same-direction positions?

The current Cascade SL reduces remaining same-direction SL distances by 75%
(keep 25%) after any SL exit. If the tightened SLs consistently trigger
within 1 bar, the mechanism is effectively a delayed "close all same-dir."

Scenarios:
  1. baseline         — No cascade, no immediate close (v1.42.0 minus cascade)
  2. cascade_t25      — Cascade SL tighten 25% (keep 75%)
  3. cascade_t50      — Cascade SL tighten 50% (keep 50%)
  4. cascade_t75      — Cascade SL tighten 75% (keep 25%) [current v1.41.0]
  5. immediate_close  — After SL, market-close ALL remaining same-dir at bar close
  6. partial_close_50 — After SL, close worst 50% same-dir at bar close

Phases:
  1. IS full neutral window — all 6 scenarios + cascade tracking
  2. WF 3-fold expanding window — top 3 + baseline
  3. Cluster analysis — cluster event comparison

Baseline: Production v1.42.0 (M2/M4 disabled, P3 CascadeSL is the subject)

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE = 3 FIXED (v1.42.0: adaptive disabled)
  - FEE = FEE_PCT * LEVERAGE (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - WF: 3-fold expanding window

Author: Research Agent
Date: 2026-03-04
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

warnings.filterwarnings('ignore')

# ============================================================
# Constants — matching production v1.42.0 exactly
# ============================================================
LEVERAGE = 3          # v1.42.0: FIXED (adaptive leverage DISABLED)
FEE_PCT = 0.10
SLIPPAGE_BUFFER = 0.02
TIMEOUT_BARS = 864
N_SLOTS = 9
DIRECTION_CAP = 7
AGG_RISK_COUNTER = 3.0
AGG_RISK_WITH = 7.0
MOMENTUM_LOOKBACK = 6
MOMENTUM_THRESHOLD = 1.0
MOMENTUM_COOLDOWN = 6
ATR_PERIOD = 14
ATR_WINDOW = 576
ATR_CLAMP_LO = 0.6
ATR_CLAMP_HI = 1.7
EMA_PERIOD = 20
EMA_LOOKBACK = 5
BARS_PER_DAY = 288

# MDD sizing params (production v1.35.2)
MDD_FULL_BELOW = 3.0
MDD_MIN_ABOVE = 15.0
MDD_MIN_SCALE = 0.25

# v1.42.0: Regime sizing DISABLED — regime_mult effectively unused for sizing
# but EMA slope still used for agg_risk counter/with classification
REGIME_MULT = None  # disabled

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'cascade_sl_depth_study.json')


# ============================================================
# Data Loading & Preprocessing
# ============================================================

def load_and_classify(data_file):
    df = pd.read_csv(data_file)
    if 'open' not in df.columns:
        df.columns = [c.lower() for c in df.columns]
    if 'type_code' in df.columns:
        df['rctype'] = df['type_code']
    elif 'rctype' not in df.columns:
        if 'avg_body' not in df.columns:
            df['avg_body'] = (abs(df['close'] - df['open'])).rolling(AVG_BODY_WINDOW).mean()
        types = []
        for i in range(len(df)):
            row = df.iloc[i]
            ab = row.get('avg_body', 1.0)
            if pd.isna(ab):
                ab = 1.0
            types.append(classify_candle(row, ab))
        df['rctype'] = types
    print(f"  Loaded {len(df)} bars, {df['rctype'].nunique()} candle types")
    return df


def compute_atr_ratio(df):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False).mean().values
    med = pd.Series(atr).rolling(ATR_WINDOW, min_periods=1).median().values
    return np.where(med > 0, atr / med, 1.0)


def compute_ema_slope(closes):
    ema = pd.Series(closes).ewm(span=EMA_PERIOD, adjust=False).mean().values
    slope = np.full(len(closes), 0.0)
    for i in range(EMA_LOOKBACK, len(closes)):
        slope[i] = ema[i] - ema[i - EMA_LOOKBACK]
    return slope


def find_neutral_window(closes, tol_pct=1.0):
    n = len(closes)
    best_start, best_end, best_len = 0, n - 1, 0
    for i in range(n):
        for j in range(n - 1, i + best_len - 1, -1):
            if closes[i] > 0 and abs(closes[j] / closes[i] - 1) * 100 <= tol_pct:
                length = j - i
                if length > best_len:
                    best_start, best_end, best_len = i, j, length
                break
    return best_start, best_end


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ============================================================
# Exit Check (v1.42.0: fixed leverage)
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio):
    """Check if position exits at this bar. Uses pos['sl_pct'] which may have
    been modified by cascade tightening."""
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None
    entry = opens[entry_bar]
    if entry <= 0:
        return None

    sig_bar = pos['signal_bar']
    if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = clamp(atr_ratio[sig_bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
    else:
        r = 1.0

    eff_tp = pos['tp_pct'] * r + SLIPPAGE_BUFFER
    # Use current sl_pct (may be tightened by cascade)
    eff_sl = max(0.1, pos['sl_pct'] * r - SLIPPAGE_BUFFER)
    direction = pos['direction']

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    if (bar - entry_bar) >= TIMEOUT_BARS:
        return {'pnl_slot': 0, 'reason': 'TIMEOUT', 'drop': True,
                'entry_bar': entry_bar, 'exit_bar': bar}

    h, l = highs[bar], lows[bar]
    if direction == 'LONG':
        hit_tp, hit_sl = h >= tp_price, l <= sl_price
    else:
        hit_tp, hit_sl = l <= tp_price, h >= sl_price

    if not hit_tp and not hit_sl:
        return None

    if hit_tp and hit_sl:
        if abs(tp_price - opens[bar]) <= abs(sl_price - opens[bar]):
            exit_price, reason = tp_price, 'TP'
        else:
            exit_price, reason = sl_price, 'SL'
    elif hit_tp:
        exit_price, reason = tp_price, 'TP'
    else:
        exit_price, reason = sl_price, 'SL'

    fee = FEE_PCT * LEVERAGE
    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * LEVERAGE
    else:
        pnl = (1 - exit_price / entry) * 100 * LEVERAGE
    pnl -= fee

    return {'pnl_slot': pnl, 'reason': reason, 'drop': False,
            'entry_bar': entry_bar, 'exit_bar': bar, 'leverage': LEVERAGE,
            'sl_price': sl_price, 'tp_price': tp_price, 'exit_price': exit_price}


def _get_mdd_size_scale(equity, peak_equity):
    if peak_equity <= 0:
        return 1.0
    dd_pct = (peak_equity - equity) / peak_equity * 100
    if dd_pct <= MDD_FULL_BELOW:
        return 1.0
    if dd_pct >= MDD_MIN_ABOVE:
        return MDD_MIN_SCALE
    return 1.0 - (1.0 - MDD_MIN_SCALE) * (dd_pct - MDD_FULL_BELOW) / (MDD_MIN_ABOVE - MDD_FULL_BELOW)


def _compute_unrealized_pnl(pos, bar_close, opens):
    """Compute unrealized PnL for a position at a given price."""
    entry = opens[pos['entry_bar']]
    if entry <= 0:
        return 0.0
    fee = FEE_PCT * LEVERAGE
    if pos['direction'] == 'LONG':
        pnl = (bar_close / entry - 1) * 100 * LEVERAGE
    else:
        pnl = (1 - bar_close / entry) * 100 * LEVERAGE
    pnl -= fee
    return pnl


# ============================================================
# Portfolio Simulator with per-group cascade tracking
# ============================================================

def portfolio_sim_tracked(signal_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, start_bar, end_bar,
                          cascade_tighten_pct=0.0,
                          immediate_close=False,
                          partial_close_pct=0.0):
    """N-pos portfolio sim with production v1.42.0 features + per-event cascade tracking.

    Production features always ON:
      - N=9 compound, direction_cap=7, agg_risk_cap(3/7%),
        momentum_guard, timeout(864 DROP), ATR-scaled TP/SL,
        FIXED leverage=3, MDD sizing
      - v1.42.0: M2 Regime sizing DISABLED, M4 Adaptive leverage DISABLED
      - P3 Cascade SL: controlled by cascade_tighten_pct parameter

    For each cascade fire, tracks which tightened positions exit within 1/3/12 bars
    as a group, so we can answer: "does 75% cascade == immediate close?"
    """
    size_pct = 100.0 / N_SLOTS
    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}
    cluster_events = []

    # Per-event cascade tracking
    cascade_fires = 0
    # Each cascade event: {bar, direction, n_tightened, exits_within_1, exits_within_3,
    #                       exits_within_12, tp_survivals, all_exit_within_1}
    cascade_event_log = []
    # Map: position slot -> list of (cascade_bar, cascade_event_idx)
    pos_cascade_map = {}

    equity_history = []

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # --- Exits ---
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_exits = []

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                result['size_mult'] = sm
                result['leverage'] = LEVERAGE
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio

                # Update cascade event tracking
                slot_id = pos['slot']
                if slot_id in pos_cascade_map:
                    for (c_bar, c_idx) in pos_cascade_map[slot_id]:
                        bars_since = bar - c_bar
                        evt = cascade_event_log[c_idx]
                        if result['reason'] == 'SL':
                            if bars_since <= 1:
                                evt['exits_within_1'] += 1
                            if bars_since <= 3:
                                evt['exits_within_3'] += 1
                            if bars_since <= 12:
                                evt['exits_within_12'] += 1
                        elif result['reason'] == 'TP':
                            evt['tp_survivals'] += 1
                        evt['total_exited'] += 1

                if result['reason'] == 'SL':
                    bar_sl_exits.append({
                        'direction': pos['direction'],
                        'pnl_portfolio': pnl_portfolio,
                        'pattern': pos['pattern'],
                    })

        positions = [p for p in positions if p['slot'] not in closed_slots]
        # Clean up cascade map for closed slots
        for slot_id in closed_slots:
            pos_cascade_map.pop(slot_id, None)

        # --- Post-SL Actions ---
        for direction in ['LONG', 'SHORT']:
            dir_sl_exits = [e for e in bar_sl_exits if e['direction'] == direction]
            if not dir_sl_exits:
                continue

            same_dir_remaining = [p for p in positions if p['direction'] == direction]
            if not same_dir_remaining:
                continue

            # Cascade SL Tightening
            if cascade_tighten_pct > 0:
                cascade_fires += 1
                evt_idx = len(cascade_event_log)
                n_tightened = 0
                for other_pos in same_dir_remaining:
                    old_sl = other_pos['sl_pct']
                    other_pos['sl_pct'] = old_sl * (1 - cascade_tighten_pct / 100)
                    n_tightened += 1
                    slot_id = other_pos['slot']
                    if slot_id not in pos_cascade_map:
                        pos_cascade_map[slot_id] = []
                    pos_cascade_map[slot_id].append((bar, evt_idx))

                cascade_event_log.append({
                    'bar': bar, 'direction': direction,
                    'n_tightened': n_tightened,
                    'exits_within_1': 0, 'exits_within_3': 0,
                    'exits_within_12': 0, 'tp_survivals': 0,
                    'total_exited': 0,
                })

            # Immediate Close
            if immediate_close:
                cascade_fires += 1
                close_slots = set()
                for pos in same_dir_remaining:
                    close_price = closes[bar]
                    pnl_slot = _compute_unrealized_pnl(pos, close_price, opens)
                    sm = pos.get('size_mult', 1.0)
                    pnl_portfolio = pnl_slot * (size_pct / 100) * sm
                    trades.append({
                        'pnl_slot': pnl_slot, 'reason': 'MARKET_CASCADE',
                        'drop': False, 'entry_bar': pos['entry_bar'],
                        'exit_bar': bar, 'leverage': LEVERAGE,
                        'pattern': pos['pattern'], 'direction': pos['direction'],
                        'pnl_portfolio': pnl_portfolio, 'size_mult': sm,
                    })
                    bar_pnl_sum += pnl_portfolio
                    close_slots.add(pos['slot'])
                    pos_cascade_map.pop(pos['slot'], None)
                positions = [p for p in positions if p['slot'] not in close_slots]

            # Partial Close 50%
            if partial_close_pct > 0:
                cascade_fires += 1
                same_dir_remaining = [p for p in positions if p['direction'] == direction]
                if same_dir_remaining:
                    ranked = sorted(same_dir_remaining,
                                    key=lambda p: _compute_unrealized_pnl(p, closes[bar], opens))
                    n_to_close = max(1, int(np.ceil(len(ranked) * partial_close_pct / 100)))
                    to_close = ranked[:n_to_close]
                    close_slots = set()
                    for pos in to_close:
                        close_price = closes[bar]
                        pnl_slot = _compute_unrealized_pnl(pos, close_price, opens)
                        sm = pos.get('size_mult', 1.0)
                        pnl_portfolio = pnl_slot * (size_pct / 100) * sm
                        trades.append({
                            'pnl_slot': pnl_slot, 'reason': 'PARTIAL_CASCADE',
                            'drop': False, 'entry_bar': pos['entry_bar'],
                            'exit_bar': bar, 'leverage': LEVERAGE,
                            'pattern': pos['pattern'], 'direction': pos['direction'],
                            'pnl_portfolio': pnl_portfolio, 'size_mult': sm,
                        })
                        bar_pnl_sum += pnl_portfolio
                        close_slots.add(pos['slot'])
                        pos_cascade_map.pop(pos['slot'], None)
                    positions = [p for p in positions if p['slot'] not in close_slots]

        # Detect cluster events (2+ same-dir SL exits in same bar)
        for direction in ['LONG', 'SHORT']:
            dir_exits = [e for e in bar_sl_exits if e['direction'] == direction]
            if len(dir_exits) >= 2:
                total_loss = sum(e['pnl_portfolio'] for e in dir_exits)
                cluster_events.append({
                    'bar': bar, 'direction': direction,
                    'n_hits': len(dir_exits), 'total_loss': total_loss
                })

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        equity_history.append(equity)

        # Momentum guard
        if MOMENTUM_LOOKBACK > 0 and bar >= MOMENTUM_LOOKBACK:
            price_now, price_ago = closes[bar], closes[bar - MOMENTUM_LOOKBACK]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > MOMENTUM_THRESHOLD:
                    momentum_pause_until['SHORT'] = bar + MOMENTUM_COOLDOWN
                elif pct_change < -MOMENTUM_THRESHOLD:
                    momentum_pause_until['LONG'] = bar + MOMENTUM_COOLDOWN

        # --- Entries ---
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            if bar < momentum_pause_until.get(direction, -1):
                continue

            sm = 1.0
            mdd_scale = _get_mdd_size_scale(equity, peak_equity)
            sm *= mdd_scale

            is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
            is_counter = ((is_uptrend and direction == 'SHORT') or
                          (not is_uptrend and direction == 'LONG'))
            cap_pct = AGG_RISK_COUNTER if is_counter else AGG_RISK_WITH

            existing_exposure = 0.0
            for p in positions:
                if p['direction'] == direction:
                    p_sig = p['signal_bar']
                    p_r = 1.0
                    if (atr_ratio is not None and p_sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[p_sig])):
                        p_r = clamp(atr_ratio[p_sig], ATR_CLAMP_LO, ATR_CLAMP_HI)
                    orig_sl = p.get('original_sl_pct', p['sl_pct'])
                    existing_exposure += (orig_sl * p_r * (1.0 / N_SLOTS)
                                          * LEVERAGE * p.get('size_mult', 1.0))

            new_r = 1.0
            if (atr_ratio is not None and sig_bar < len(atr_ratio)
                    and not np.isnan(atr_ratio[sig_bar])):
                new_r = clamp(atr_ratio[sig_bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
            new_exposure = sl_pct * new_r * (1.0 / N_SLOTS) * LEVERAGE * sm
            if existing_exposure + new_exposure > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}", 'signal_bar': sig_bar,
                'entry_bar': entry_bar, 'direction': direction,
                'pattern': pat, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'original_sl_pct': sl_pct,
                'size_mult': sm, 'leverage': LEVERAGE,
            })

    # Force-close remaining
    for pos in positions:
        if pos['entry_bar'] >= n_bars:
            continue
        entry = opens[pos['entry_bar']]
        if entry <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        fee = FEE_PCT * LEVERAGE
        if pos['direction'] == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': pos['entry_bar'], 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
            'leverage': LEVERAGE,
        })

    # Build cascade tracking summary
    cascade_tracking = {
        'cascade_fires': cascade_fires,
    }

    if cascade_event_log:
        total_tightened = sum(e['n_tightened'] for e in cascade_event_log)
        total_exited = sum(e['total_exited'] for e in cascade_event_log)
        total_1bar = sum(e['exits_within_1'] for e in cascade_event_log)
        total_3bar = sum(e['exits_within_3'] for e in cascade_event_log)
        total_12bar = sum(e['exits_within_12'] for e in cascade_event_log)
        total_tp = sum(e['tp_survivals'] for e in cascade_event_log)

        # How many events had ALL tightened positions exit within 1 bar?
        all_exit_1bar = 0
        for evt in cascade_event_log:
            if evt['n_tightened'] > 0 and evt['exits_within_1'] >= evt['n_tightened']:
                all_exit_1bar += 1

        cascade_tracking.update({
            'total_cascade_events': len(cascade_event_log),
            'total_tightened_positions': total_tightened,
            'total_exited_tracked': total_exited,
            'post_cascade_sl_hits_1bar': total_1bar,
            'post_cascade_sl_hits_3bar': total_3bar,
            'post_cascade_sl_hits_12bar': total_12bar,
            'post_cascade_tp_hits': total_tp,
            'cascade_all_exit_1bar_events': all_exit_1bar,
            'cascade_all_exit_1bar_pct': round(
                all_exit_1bar / len(cascade_event_log) * 100, 1)
                if cascade_event_log else 0.0,
            'cascade_immediate_pct': round(
                total_1bar / max(1, total_exited) * 100, 1),
        })
    else:
        cascade_tracking.update({
            'total_cascade_events': 0,
            'total_tightened_positions': 0,
            'total_exited_tracked': 0,
            'post_cascade_sl_hits_1bar': 0,
            'post_cascade_sl_hits_3bar': 0,
            'post_cascade_sl_hits_12bar': 0,
            'post_cascade_tp_hits': 0,
            'cascade_all_exit_1bar_events': 0,
            'cascade_all_exit_1bar_pct': 0.0,
            'cascade_immediate_pct': 0.0,
        })

    return trades, equity_history, cluster_events, cascade_tracking


# ============================================================
# Stats Calculation
# ============================================================

def calc_stats(trades, equity_history, cluster_events=None):
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0,
                'corr_sl_events': 0, 'corr_sl_trades': 0, 'corr_sl_loss': 0,
                'cluster_events': 0, 'cluster_loss': 0}

    wins = sum(1 for t in trades if t['pnl_slot'] > 0)
    sorted_t = sorted(trades, key=lambda x: x['entry_bar'])
    eq = 100.0
    pk = eq
    mdd = 0.0
    for t in sorted_t:
        eq += t['pnl_portfolio']
        if eq > pk:
            pk = eq
        d = (pk - eq) / pk * 100 if pk > 0 else 0
        if d > mdd:
            mdd = d

    total_pnl = eq - 100.0
    wr = wins / len(trades) * 100 if trades else 0

    # Correlated SL detection (same-bar SL exits)
    sl_trades = [t for t in sorted_t if t.get('reason') == 'SL']
    same_bar_sl_groups = {}
    for t in sl_trades:
        key = (t['exit_bar'], t['direction'])
        same_bar_sl_groups.setdefault(key, []).append(t)
    corr_sl_events_dict = {k: v for k, v in same_bar_sl_groups.items() if len(v) >= 2}
    n_corr_events = len(corr_sl_events_dict)
    n_corr_sl = sum(len(v) for v in corr_sl_events_dict.values())
    corr_loss = sum(sum(t['pnl_portfolio'] for t in v) for v in corr_sl_events_dict.values())

    # Cluster events from sim
    n_cluster_events = 0
    total_cluster_loss = 0.0
    if cluster_events:
        seen = set()
        for ce in cluster_events:
            key = (ce['bar'], ce['direction'])
            if key not in seen:
                seen.add(key)
                n_cluster_events += 1
                total_cluster_loss += ce['total_loss']

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(total_pnl / mdd, 2) if mdd > 0 else 0,
        'corr_sl_events': n_corr_events,
        'corr_sl_trades': n_corr_sl,
        'corr_sl_loss': round(corr_loss, 2),
        'cluster_events': n_cluster_events,
        'cluster_loss': round(total_cluster_loss, 2),
    }


# ============================================================
# WF 3-fold expanding window
# ============================================================

def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, neutral_start, neutral_end,
           scenario_kwargs, n_folds=3):
    total = neutral_end - neutral_start
    seg_size = total // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        oos_start = neutral_start + seg_size * (fold + 1)
        oos_end = (neutral_start + seg_size * (fold + 2)
                   if fold < n_folds - 1 else neutral_end)

        trades, eq_hist, cluster_events, cascade_tracking = portfolio_sim_tracked(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            **scenario_kwargs)
        stats = calc_stats(trades, eq_hist, cluster_events)
        stats['fold'] = fold + 1
        stats['oos_bars'] = int(oos_end - oos_start)
        stats['cascade_tracking'] = cascade_tracking
        results.append(stats)

    return results


# ============================================================
# Scenario Definitions
# ============================================================

SCENARIOS = [
    ('baseline',        {'cascade_tighten_pct': 0.0}),
    ('cascade_t25',     {'cascade_tighten_pct': 25.0}),
    ('cascade_t50',     {'cascade_tighten_pct': 50.0}),
    ('cascade_t75',     {'cascade_tighten_pct': 75.0}),
    ('immediate_close', {'immediate_close': True}),
    ('partial_close_50', {'partial_close_pct': 50.0}),
]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("CASCADE SL DEPTH STUDY")
    print("Is 75% Cascade SL Tightening == Immediate Close?")
    print("=" * 78)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Baseline: Production v1.42.0 (M2/M4 disabled, fixed leverage=3)")
    print(f"Subject: P3 Cascade SL Tightening depth analysis")

    # ---- Load Data ----
    print("\n[1] Loading data...")
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    print("\n[2] Computing indicators...")
    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)

    neutral_start, neutral_end = find_neutral_window(closes)
    n_neutral = neutral_end - neutral_start
    print(f"  Neutral window: bars {neutral_start}-{neutral_end} ({n_neutral} bars, "
          f"{n_neutral / BARS_PER_DAY:.0f}d)")

    # ---- Load Patterns ----
    print("\n[3] Loading patterns...")
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pats_raw = pat_data['patterns']
    tpsl = pat_data.get('patterns_tpsl', {})

    pat_lookup = {}
    for pat_name in pats_raw.get('long', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        pat_lookup[pat_name] = {'direction': 'LONG', 'tp': tp_sl[0], 'sl': tp_sl[1]}
    for pat_name in pats_raw.get('short', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        pat_lookup[pat_name] = {'direction': 'SHORT', 'tp': tp_sl[0], 'sl': tp_sl[1]}
    n_long = len(pats_raw.get('long', []))
    n_short = len(pats_raw.get('short', []))
    print(f"  {len(pat_lookup)} patterns ({n_long}L + {n_short}S)")

    # ---- Build Signals ----
    print("\n[4] Building signal index...")
    rctypes = df['rctype'].values
    signal_tuples = []
    for i in range(2, n_bars):
        tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if tri in pat_lookup:
            p = pat_lookup[tri]
            signal_tuples.append((i, tri, p['direction'], p['tp'], p['sl']))

    n_in_neutral = sum(1 for s in signal_tuples if neutral_start <= s[0] < neutral_end)
    print(f"  {len(signal_tuples)} total signals, {n_in_neutral} in neutral window")

    # ================================================================
    # PHASE 1: IS Full Neutral Window — All 6 Scenarios
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 1: IS Full Neutral Window — All Scenarios")
    print("=" * 78)

    phase1_results = {}

    for name, kwargs in SCENARIOS:
        print(f"\n  Running {name}...")
        trades, eq_hist, ce, cascade_tracking = portfolio_sim_tracked(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            **kwargs)
        stats = calc_stats(trades, eq_hist, ce)
        stats['cascade_tracking'] = cascade_tracking
        phase1_results[name] = stats

    # Print summary table
    print("\n  " + "-" * 100)
    print(f"  {'Scenario':<20} {'PnL%':>9} {'MDD%':>7} {'P/M':>7} {'WR%':>6} "
          f"{'Trades':>6} {'CascFires':>9} {'Immed%':>7} {'AvgBarsPost':>11} "
          f"{'ClustEv':>7} {'ClustL%':>8}")
    print("  " + "-" * 100)

    for name, _ in SCENARIOS:
        s = phase1_results[name]
        ct = s['cascade_tracking']
        cf = ct['cascade_fires']
        imm_pct = ct.get('cascade_immediate_pct', 0.0)
        # For cascade scenarios, compute avg bars to exit
        total_exited = ct.get('total_exited_tracked', 0)
        sl_1bar = ct.get('post_cascade_sl_hits_1bar', 0)
        tp_hits = ct.get('post_cascade_tp_hits', 0)
        all_exit_events = ct.get('cascade_all_exit_1bar_events', 0)
        total_events = ct.get('total_cascade_events', 0)

        # Determine avg bars (for non-cascade scenarios, N/A)
        if name.startswith('cascade') and total_exited > 0:
            # We don't have direct avg bars in the tracked sim, but we have
            # the 1bar/3bar/12bar counts to infer distribution
            avg_bars_str = f"{sl_1bar}/{ct.get('post_cascade_sl_hits_3bar', 0)}/{ct.get('post_cascade_sl_hits_12bar', 0)}"
        else:
            avg_bars_str = "N/A"

        print(f"  {name:<20} {s['pnl']:>+9.2f} {s['mdd']:>7.2f} {s['pnl_mdd']:>7.2f} "
              f"{s['wr']:>6.1f} {s['trades']:>6} {cf:>9} {imm_pct:>7.1f} "
              f"{avg_bars_str:>11} {s['cluster_events']:>7} {s['cluster_loss']:>+8.2f}")

    # Detailed cascade tracking for cascade scenarios
    print("\n  --- Detailed Cascade Tracking ---")
    for name, _ in SCENARIOS:
        if not name.startswith('cascade'):
            continue
        s = phase1_results[name]
        ct = s['cascade_tracking']
        if ct['cascade_fires'] == 0:
            continue

        total_events = ct.get('total_cascade_events', 0)
        total_tight = ct.get('total_tightened_positions', 0)
        total_exited = ct.get('total_exited_tracked', 0)
        sl_1 = ct.get('post_cascade_sl_hits_1bar', 0)
        sl_3 = ct.get('post_cascade_sl_hits_3bar', 0)
        sl_12 = ct.get('post_cascade_sl_hits_12bar', 0)
        tp = ct.get('post_cascade_tp_hits', 0)
        all_1bar = ct.get('cascade_all_exit_1bar_events', 0)
        all_1bar_pct = ct.get('cascade_all_exit_1bar_pct', 0.0)
        imm_pct = ct.get('cascade_immediate_pct', 0.0)

        print(f"\n  {name}:")
        print(f"    Cascade events: {total_events}")
        print(f"    Total tightened positions: {total_tight}")
        print(f"    Exited (tracked): {total_exited}")
        print(f"    SL hit within 1 bar: {sl_1} ({sl_1/max(1,total_exited)*100:.1f}%)")
        print(f"    SL hit within 3 bars: {sl_3} ({sl_3/max(1,total_exited)*100:.1f}%)")
        print(f"    SL hit within 12 bars: {sl_12} ({sl_12/max(1,total_exited)*100:.1f}%)")
        print(f"    TP survivals: {tp} ({tp/max(1,total_exited)*100:.1f}%)")
        print(f"    Events where ALL tightened exit in 1 bar: {all_1bar}/{total_events} ({all_1bar_pct:.1f}%)")
        print(f"    Cascade immediate exit %: {imm_pct:.1f}%")

    # Print the key comparison
    print("\n  --- Key Comparison: cascade_t75 vs immediate_close ---")
    c75 = phase1_results.get('cascade_t75', {})
    imm = phase1_results.get('immediate_close', {})
    if c75 and imm:
        print(f"    {'Metric':<25} {'cascade_t75':>14} {'immediate_close':>14} {'Delta':>10}")
        print(f"    {'-'*65}")
        for metric in ['pnl', 'mdd', 'pnl_mdd', 'wr', 'trades',
                        'cluster_events', 'cluster_loss']:
            v1 = c75.get(metric, 0)
            v2 = imm.get(metric, 0)
            delta = v1 - v2 if isinstance(v1, (int, float)) else 'N/A'
            if isinstance(v1, float):
                print(f"    {metric:<25} {v1:>14.2f} {v2:>14.2f} {delta:>+10.2f}")
            else:
                print(f"    {metric:<25} {v1:>14} {v2:>14} {delta:>+10}")

        ct75 = c75.get('cascade_tracking', {})
        imm_pct = ct75.get('cascade_immediate_pct', 0.0)
        all_1bar_pct = ct75.get('cascade_all_exit_1bar_pct', 0.0)
        print(f"\n    cascade_t75 immediate exit rate: {imm_pct:.1f}%")
        print(f"    cascade_t75 all-in-1-bar event rate: {all_1bar_pct:.1f}%")
        if imm_pct >= 90:
            print(f"    CONCLUSION: cascade_t75 is effectively equivalent to immediate close "
                  f"({imm_pct:.0f}% of tightened positions exit within 1 bar)")
        elif imm_pct >= 70:
            print(f"    CONCLUSION: cascade_t75 is mostly equivalent to immediate close "
                  f"({imm_pct:.0f}% within 1 bar), but {100-imm_pct:.0f}% survive briefly")
        else:
            print(f"    CONCLUSION: cascade_t75 is NOT equivalent to immediate close "
                  f"(only {imm_pct:.0f}% exit within 1 bar)")

    # ================================================================
    # PHASE 2: WF 3-fold Expanding Window
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 2: WF 3-fold Expanding Window Validation")
    print("=" * 78)

    # Select top 3 by PnL/MDD + baseline
    phase1_ranked = sorted(
        [(name, phase1_results[name]) for name, _ in SCENARIOS],
        key=lambda x: x[1]['pnl_mdd'], reverse=True
    )
    top3_names = [name for name, _ in phase1_ranked[:3]]
    if 'baseline' not in top3_names:
        top3_names.append('baseline')

    # Always include cascade_t75 and immediate_close for comparison
    for must_include in ['cascade_t75', 'immediate_close']:
        if must_include not in top3_names:
            top3_names.append(must_include)

    # Deduplicate
    seen = set()
    wf_scenarios = []
    for name in top3_names:
        if name not in seen:
            seen.add(name)
            wf_scenarios.append(name)

    wf_kwargs_map = {name: kwargs for name, kwargs in SCENARIOS}

    print(f"  Scenarios for WF: {wf_scenarios}")

    wf_results = {}
    for name in wf_scenarios:
        kwargs = wf_kwargs_map[name]
        folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, neutral_start, neutral_end,
                       scenario_kwargs=kwargs)
        wf_results[name] = folds

        all_pass = all(f['pnl'] > 0 for f in folds)
        min_pnl = min(f['pnl'] for f in folds)
        total_pnl = sum(f['pnl'] for f in folds)

        print(f"\n  {name}:")
        for f in folds:
            status = "PASS" if f['pnl'] > 0 else "FAIL"
            ct_f = f.get('cascade_tracking', {})
            cf = ct_f.get('cascade_fires', 0)
            print(f"    Fold {f['fold']}: PnL {f['pnl']:>+8.2f}%, WR {f['wr']:>5.1f}%, "
                  f"MDD {f['mdd']:>5.2f}%, CascFires {cf:>3} | {status}")
        verdict = '3/3 PASS' if all_pass else 'FAIL'
        print(f"    Verdict: {verdict} (min fold {min_pnl:+.2f}%, total OOS {total_pnl:+.2f}%)")

    # ================================================================
    # PHASE 3: Cluster Analysis
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 3: Cluster Analysis Comparison")
    print("=" * 78)

    print(f"\n  {'Scenario':<20} {'ClustEvents':>11} {'ClustLoss%':>10} "
          f"{'CorrSLEvents':>12} {'CorrSLLoss%':>11}")
    print("  " + "-" * 68)

    for name, _ in SCENARIOS:
        s = phase1_results[name]
        print(f"  {name:<20} {s['cluster_events']:>11} {s['cluster_loss']:>+10.2f} "
              f"{s['corr_sl_events']:>12} {s['corr_sl_loss']:>+11.2f}")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print("\n" + "=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)

    # Print all IS results in clean format
    print(f"\n  All IS Results:")
    for name, _ in SCENARIOS:
        s = phase1_results[name]
        ct = s['cascade_tracking']
        cf = ct['cascade_fires']
        imm_pct = ct.get('cascade_immediate_pct', 0.0)
        avg_bars = 'N/A'
        # Compute average bars from 1/3/12 bar distribution for cascade scenarios
        if name.startswith('cascade') and cf > 0:
            total_exited = ct.get('total_exited_tracked', 0)
            if total_exited > 0:
                sl_1 = ct.get('post_cascade_sl_hits_1bar', 0)
                sl_3 = ct.get('post_cascade_sl_hits_3bar', 0)
                sl_12 = ct.get('post_cascade_sl_hits_12bar', 0)
                avg_bars = f"1b:{sl_1} 3b:{sl_3} 12b:{sl_12}"

        print(f"\n  [{name}]")
        print(f"    PnL: {s['pnl']:+.1f}%  MDD: {s['mdd']:.1f}%  "
              f"PnL/MDD: {s['pnl_mdd']:.1f}  WR: {s['wr']:.1f}%  Trades: {s['trades']}")
        print(f"    Cascade fires: {cf}  Post-cascade immediate exit: {imm_pct:.1f}%  "
              f"Post-cascade exits: {avg_bars}")
        print(f"    Cluster events: {s['cluster_events']}  "
              f"Cluster loss: {s['cluster_loss']:+.1f}%")

    # Answer the core question
    print("\n  --- Core Question: Is cascade_t75 == immediate_close? ---")
    c75 = phase1_results.get('cascade_t75', {})
    imm = phase1_results.get('immediate_close', {})
    if c75 and imm:
        ct75 = c75.get('cascade_tracking', {})
        imm_pct = ct75.get('cascade_immediate_pct', 0.0)
        all_1bar_pct = ct75.get('cascade_all_exit_1bar_pct', 0.0)

        pnl_diff = abs(c75['pnl'] - imm['pnl'])
        mdd_diff = abs(c75['mdd'] - imm['mdd'])

        print(f"    PnL difference: {pnl_diff:.2f}%")
        print(f"    MDD difference: {mdd_diff:.2f}%")
        print(f"    Tightened positions exiting within 1 bar: {imm_pct:.1f}%")
        print(f"    All-in-1-bar events: {all_1bar_pct:.1f}%")

        if pnl_diff < 5.0 and mdd_diff < 2.0 and imm_pct >= 80:
            print(f"\n    VERDICT: YES — cascade_t75 is functionally equivalent to immediate close.")
            print(f"    The results are within noise ({pnl_diff:.1f}% PnL, {mdd_diff:.1f}% MDD)")
            print(f"    and {imm_pct:.0f}% of tightened positions exit within 1 bar.")
        elif imm_pct >= 60:
            print(f"\n    VERDICT: MOSTLY — cascade_t75 approximates immediate close.")
            print(f"    {imm_pct:.0f}% of positions exit within 1 bar, but there are differences:")
            print(f"    PnL gap: {pnl_diff:.1f}%, MDD gap: {mdd_diff:.1f}%")
        else:
            print(f"\n    VERDICT: NO — cascade_t75 behaves differently from immediate close.")
            print(f"    Only {imm_pct:.0f}% exit within 1 bar.")
            print(f"    PnL gap: {pnl_diff:.1f}%, MDD gap: {mdd_diff:.1f}%")

        # Recommendation
        best_scenario = max(phase1_results.items(),
                            key=lambda x: x[1]['pnl_mdd'])
        print(f"\n    Best PnL/MDD scenario: {best_scenario[0]} "
              f"(PnL/MDD {best_scenario[1]['pnl_mdd']:.1f})")

        # Check WF results for best
        if best_scenario[0] in wf_results:
            wf_folds = wf_results[best_scenario[0]]
            wf_pass = all(f['pnl'] > 0 for f in wf_folds)
            print(f"    WF status: {'3/3 PASS' if wf_pass else 'FAIL'}")

    # ================================================================
    # Save Results
    # ================================================================
    output = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'study': 'cascade_sl_depth_study',
            'question': 'Is 75% Cascade SL Tightening equivalent to immediate close?',
            'baseline': 'v1.42.0 (M2/M4 disabled, fixed leverage=3)',
            'data_bars': int(n_bars),
            'neutral_window': [int(neutral_start), int(neutral_end)],
            'patterns': f"{len(pat_lookup)} ({n_long}L + {n_short}S)",
            'leverage': LEVERAGE,
            'scenarios': [name for name, _ in SCENARIOS],
        },
        'phase1_is': {},
        'phase2_wf': {},
        'phase3_clusters': {},
    }

    for name, _ in SCENARIOS:
        s = phase1_results[name]
        output['phase1_is'][name] = {
            'pnl': s['pnl'], 'mdd': s['mdd'], 'pnl_mdd': s['pnl_mdd'],
            'wr': s['wr'], 'trades': s['trades'],
            'cluster_events': s['cluster_events'],
            'cluster_loss': s['cluster_loss'],
            'corr_sl_events': s['corr_sl_events'],
            'corr_sl_loss': s['corr_sl_loss'],
            'cascade_tracking': s['cascade_tracking'],
        }

    for name in wf_scenarios:
        if name in wf_results:
            folds = wf_results[name]
            all_pass = all(f['pnl'] > 0 for f in folds)
            output['phase2_wf'][name] = {
                'folds': [{
                    'fold': f['fold'], 'pnl': f['pnl'], 'wr': f['wr'],
                    'mdd': f['mdd'], 'trades': f['trades'],
                    'cascade_tracking': f.get('cascade_tracking', {}),
                } for f in folds],
                'all_pass': all_pass,
                'min_fold_pnl': min(f['pnl'] for f in folds),
                'total_oos_pnl': sum(f['pnl'] for f in folds),
            }

    # Core answer
    c75 = phase1_results.get('cascade_t75', {})
    imm = phase1_results.get('immediate_close', {})
    if c75 and imm:
        ct75 = c75.get('cascade_tracking', {})
        output['verdict'] = {
            'pnl_difference': round(abs(c75['pnl'] - imm['pnl']), 2),
            'mdd_difference': round(abs(c75['mdd'] - imm['mdd']), 2),
            'cascade_immediate_exit_pct': ct75.get('cascade_immediate_pct', 0.0),
            'cascade_all_in_1bar_pct': ct75.get('cascade_all_exit_1bar_pct', 0.0),
            'equivalent': (abs(c75['pnl'] - imm['pnl']) < 5.0
                           and abs(c75['mdd'] - imm['mdd']) < 2.0
                           and ct75.get('cascade_immediate_pct', 0.0) >= 80),
        }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
