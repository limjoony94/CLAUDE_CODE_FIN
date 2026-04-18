#!/usr/bin/env python3
"""
Trail Exit Mechanism Alternatives — C1 Breakout v2.6
=====================================================
9 variants (A-I) of the trailing TP mechanism. Strategy base unchanged:
  channel_period=15, body_min_ratio=0.4, atr_period=14, max_sl_atr=3.3,
  emergency_sl_pct=3.0, max_hold_bars=192, sl_min_pct=0.15, sl_max_pct=3.0,
  min_bars_between=2, fractal_lookback=10.

Uses production indicators + entry/SL logic (imported). Exit logic dispatched
per variant. N=1, SIZE_PCT=100, LEVERAGE=1, FEE=0.10% RT — matches production
baseline (+169.5% additive 1x on 333d).

Reports per variant:
  total PnL, WR, R:R, trades/day, exit breakdown (SL / zero trail / pos trail /
  neg trail / emergency / timeout), max single winner, MDD (compound 1x),
  WF 5-fold PASS count.

Saves incrementally to results/trail_alternatives_comparison.json.
"""

import os, sys, json, math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings)

# ─── Constants (production baseline) ───
FEE_PCT = 0.10           # round trip (taker 0.05% × 2)
LEVERAGE = 1             # additive 1x comparison
EMERGENCY_SL = 3.0       # percent
TIMEOUT_BARS = 192       # 48h / 15m
MAX_POSITIONS = 1        # production N=1
SIZE_PCT = 100.0         # full equity per position
ZERO_EPS = 0.05          # |pnl%| < 0.05 counted as "zero" bin

DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data',
    'btc_5m_270days_reclassified.csv')

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results',
    'trail_alternatives_comparison.json')


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

def load_15m_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['group'] = df.index // 3
    agg = df.groupby('group').agg(
        timestamp=('timestamp', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index(drop=True)
    return agg


def precompute(opens, highs, lows, closes, channel_period=15, atr_period=14,
               fractal_lookback=10):
    h = highs.tolist()
    l = lows.tolist()
    c = closes.tolist()
    atr = compute_atr(h, l, c, atr_period)
    ch_high, ch_low = compute_channel(h, l, channel_period)
    sw_low, sw_high = compute_fractal_swings(h, l, fractal_lookback)
    return (np.array(atr), np.array(ch_high), np.array(ch_low),
            np.array(sw_low), np.array(sw_high))


# ═══════════════════════════════════════════════════════════════════════
# Entry (replicated from C1BreakoutSignal — strict, no modifications)
# ═══════════════════════════════════════════════════════════════════════

def check_entry(bar_o, bar_h, bar_l, bar_c, ch_high, ch_low, atr_val,
                sw_low, sw_high,
                body_min_ratio=0.4, max_sl_atr=3.3,
                sl_min_pct=0.15, sl_max_pct=3.0):
    if (math.isnan(ch_high) or math.isnan(ch_low) or math.isnan(atr_val)
            or atr_val <= 0):
        return None

    if bar_c > ch_high:
        direction = 'LONG'
    elif bar_c < ch_low:
        direction = 'SHORT'
    else:
        return None

    rng = bar_h - bar_l
    if rng <= 0:
        return None
    body = bar_c - bar_o
    if abs(body) / rng < body_min_ratio:
        return None
    if direction == 'LONG' and body <= 0:
        return None
    if direction == 'SHORT' and body >= 0:
        return None

    # SL will be computed at actual entry_price (next bar open) in loop
    return {'direction': direction}


def compute_sl(direction, entry_price, atr_val, sw_low, sw_high,
               max_sl_atr=3.3, sl_min_pct=0.15, sl_max_pct=3.0):
    if direction == 'LONG':
        atr_sl = entry_price - max_sl_atr * atr_val
        fractal_sl = sw_low if not math.isnan(sw_low) else atr_sl
        sl_price = max(fractal_sl, atr_sl)
    else:
        atr_sl = entry_price + max_sl_atr * atr_val
        fractal_sl = sw_high if not math.isnan(sw_high) else atr_sl
        sl_price = min(fractal_sl, atr_sl)
    sl_dist = abs(entry_price - sl_price) / entry_price * 100
    if sl_dist < sl_min_pct or sl_dist > sl_max_pct:
        return None
    return sl_price


# ═══════════════════════════════════════════════════════════════════════
# Exit dispatchers — one per variant
# ═══════════════════════════════════════════════════════════════════════
# All receive a mutable `pos` dict and return either None (hold) or a
# dict {'reason', 'exit_price'[, 'partial_frac'(I only)]}
#
# Exit priority is enforced in the main loop:
#   1) SL (intrabar low/high vs sl_price)
#   2) Emergency (hard pct from entry)
#   3) Timeout
#   4) Variant-specific trail / TP
# ═══════════════════════════════════════════════════════════════════════

def _pnl_long(entry, px):
    return (px / entry - 1) * 100


def _pnl_short(entry, px):
    return (1 - px / entry) * 100


def _best_pnl(direction, entry, best):
    return _pnl_long(entry, best) if direction == 'LONG' else _pnl_short(entry, best)


def _cur_pnl(direction, entry, close):
    return _pnl_long(entry, close) if direction == 'LONG' else _pnl_short(entry, close)


# ── A: Baseline (current production) ─────────────────────────────────
def exit_A(pos, bar_h, bar_l, bar_c, atr_val, params):
    """trail_activation=0.05%, trail_dist = trail_K*ATR/close*100."""
    trail_K = params['trail_K']
    activation = params['trail_activation_pct']
    best_pnl = _best_pnl(pos['direction'], pos['entry_price'], pos['best_price'])
    cur_pnl = _cur_pnl(pos['direction'], pos['entry_price'], bar_c)
    if best_pnl <= activation:
        return None
    if math.isnan(atr_val) or atr_val <= 0:
        return None
    trail_dist_pct = trail_K * atr_val / bar_c * 100
    drawdown = best_pnl - cur_pnl
    if drawdown >= trail_dist_pct:
        realized = max(0, best_pnl - trail_dist_pct)
        if pos['direction'] == 'LONG':
            exit_price = pos['entry_price'] * (1 + realized / 100)
        else:
            exit_price = pos['entry_price'] * (1 - realized / 100)
        return {'reason': 'TRAIL_TP', 'exit_price': exit_price}
    return None


# ── B: Higher Activation 0.5% ────────────────────────────────────────
def exit_B(pos, bar_h, bar_l, bar_c, atr_val, params):
    """Same as A but activation = 0.5%."""
    p2 = dict(params)
    p2['trail_activation_pct'] = 0.5
    return exit_A(pos, bar_h, bar_l, bar_c, atr_val, p2)


# ── C: Dynamic activation = 1× ATR at entry (%) ──────────────────────
def exit_C(pos, bar_h, bar_l, bar_c, atr_val, params):
    """Activation threshold = 1×ATR_entry / entry_price × 100 (per-trade)."""
    p2 = dict(params)
    p2['trail_activation_pct'] = pos['c_activation_pct']  # set at entry
    return exit_A(pos, bar_h, bar_l, bar_c, atr_val, p2)


# ── D: Chandelier Exit (trail from best, absolute $) ─────────────────
def exit_D(pos, bar_h, bar_l, bar_c, atr_val, params):
    """trail_dist = trail_K * ATR_now ($). Trigger: intrabar touches stop_price.

    LONG: stop = best_high - trail_dist; exit if bar_low <= stop
    SHORT: stop = best_low + trail_dist; exit if bar_high >= stop
    No activation threshold (Chandelier runs from entry).
    Fill at stop price (stop-style, not close).
    """
    trail_K = params['trail_K']  # 3.0 in variant D
    if math.isnan(atr_val) or atr_val <= 0:
        return None
    trail_dist = trail_K * atr_val
    if pos['direction'] == 'LONG':
        stop = pos['best_price'] - trail_dist
        if stop <= pos['entry_price']:
            # Chandelier stop hasn't cleared entry yet — still below SL priority
            return None
        if bar_l <= stop:
            return {'reason': 'TRAIL_TP', 'exit_price': stop}
    else:
        stop = pos['best_price'] + trail_dist
        if stop >= pos['entry_price']:
            return None
        if bar_h >= stop:
            return {'reason': 'TRAIL_TP', 'exit_price': stop}
    return None


# ── E: Step-up Trail (3 tiers by ATR profit) ─────────────────────────
def exit_E(pos, bar_h, bar_l, bar_c, atr_val, params):
    """Tier1 (0 to 1ATR profit): no trail. Tier2 (1-2ATR): trail_K=2.5.
    Tier3 (>=2ATR): trail_K=3.0. Uses close-based formula like A."""
    if math.isnan(atr_val) or atr_val <= 0:
        return None
    entry = pos['entry_price']
    atr_entry = pos['atr_entry']
    # Profit in ATR units (using best price)
    if pos['direction'] == 'LONG':
        profit_atr = (pos['best_price'] - entry) / atr_entry
    else:
        profit_atr = (entry - pos['best_price']) / atr_entry
    if profit_atr < 1.0:
        return None  # Tier 1 — no trail
    tier_K = 2.5 if profit_atr < 2.0 else 3.0
    best_pnl = _best_pnl(pos['direction'], entry, pos['best_price'])
    cur_pnl = _cur_pnl(pos['direction'], entry, bar_c)
    trail_dist_pct = tier_K * atr_val / bar_c * 100
    drawdown = best_pnl - cur_pnl
    if drawdown >= trail_dist_pct:
        realized = max(0, best_pnl - trail_dist_pct)
        if pos['direction'] == 'LONG':
            exit_price = entry * (1 + realized / 100)
        else:
            exit_price = entry * (1 - realized / 100)
        return {'reason': 'TRAIL_TP', 'exit_price': exit_price}
    return None


# ── F: Break-even stop + delayed trail ───────────────────────────────
def exit_F(pos, bar_h, bar_l, bar_c, atr_val, params):
    """Mutates pos['sl_price'] when reaches break-even / tier thresholds.
    - profit >= 1ATR: sl_price = entry (break-even)
    - profit >= 2ATR: start trailing at trail_K=2.5 from best (A formula)
    Otherwise: original SL unchanged.

    Note: SL is checked by caller before this function; we just mutate the SL
    and optionally return trail-exit.
    """
    entry = pos['entry_price']
    atr_entry = pos['atr_entry']
    if pos['direction'] == 'LONG':
        profit_atr = (pos['best_price'] - entry) / atr_entry
    else:
        profit_atr = (entry - pos['best_price']) / atr_entry

    # Upgrade SL to break-even once +1 ATR achieved
    if profit_atr >= 1.0 and not pos.get('be_set', False):
        if pos['direction'] == 'LONG':
            pos['sl_price'] = max(pos['sl_price'], entry)
        else:
            pos['sl_price'] = min(pos['sl_price'], entry)
        pos['be_set'] = True

    # Activate trail only after +2 ATR
    if profit_atr < 2.0:
        return None
    return exit_A(pos, bar_h, bar_l, bar_c, atr_val,
                  {'trail_K': params['trail_K'], 'trail_activation_pct': 0.0})


# ── G: Tighter Trail (K=1.5) ─────────────────────────────────────────
def exit_G(pos, bar_h, bar_l, bar_c, atr_val, params):
    p2 = dict(params)
    p2['trail_K'] = 1.5
    return exit_A(pos, bar_h, bar_l, bar_c, atr_val, p2)


# ── H: Looser Trail (K=3.5) ──────────────────────────────────────────
def exit_H(pos, bar_h, bar_l, bar_c, atr_val, params):
    p2 = dict(params)
    p2['trail_K'] = 3.5
    return exit_A(pos, bar_h, bar_l, bar_c, atr_val, p2)


# ── I: Partial Scale-Out (50% at +1.5 ATR, trail rest) ───────────────
def exit_I(pos, bar_h, bar_l, bar_c, atr_val, params):
    """Two-phase exit:
      Phase 1 (before TP1 hit): if intrabar hits TP1 price → exit 50% at TP1,
        mark pos['tp1_hit']=True and pos['tp1_pnl_contrib'] stored. Still hold.
      Phase 2 (after TP1 hit): trail remaining 50% with trail_K=2.5 (A formula).
    Fees: each leg pays full 0.10% RT on its half-notional → weighted total fee
    applied outside. For simplicity we charge FEE_PCT once weighted per leg.

    Accounting scheme: single trade, weighted PnL = 0.5*tp1_raw + 0.5*trail_raw,
    weighted fee = FEE_PCT (since each half is 50% of notional; entry+exit
    per half is 0.10%, and summed across halves = 0.10% on full notional).
    """
    entry = pos['entry_price']
    atr_entry = pos['atr_entry']
    tp1_dist = 1.5 * atr_entry
    if pos['direction'] == 'LONG':
        tp1_price = entry + tp1_dist
    else:
        tp1_price = entry - tp1_dist

    # Phase 1: TP1 not yet hit
    if not pos.get('tp1_hit', False):
        hit = (bar_h >= tp1_price if pos['direction'] == 'LONG'
               else bar_l <= tp1_price)
        if hit:
            pos['tp1_hit'] = True
            pos['tp1_exit_price'] = tp1_price
        # After marking TP1 hit, check trail in same bar for the remainder
        # (use updated best_price which may already include this bar's extreme)

    # Phase 2: TP1 hit, trail remaining half
    if pos.get('tp1_hit', False):
        res = exit_A(pos, bar_h, bar_l, bar_c, atr_val,
                     {'trail_K': 2.5, 'trail_activation_pct': 0.05})
        if res is not None:
            # Weighted exit price (for record keeping — actual PnL computed
            # from tp1_exit_price and trail exit_price in trade record)
            res['partial_frac'] = 0.5
            res['tp1_exit_price'] = pos['tp1_exit_price']
            return res
    return None


EXIT_DISPATCHERS = {
    'A': exit_A, 'B': exit_B, 'C': exit_C, 'D': exit_D, 'E': exit_E,
    'F': exit_F, 'G': exit_G, 'H': exit_H, 'I': exit_I,
}

VARIANT_PARAMS = {
    'A': {'trail_K': 2.5, 'trail_activation_pct': 0.05},
    'B': {'trail_K': 2.5, 'trail_activation_pct': 0.5},
    'C': {'trail_K': 2.5, 'trail_activation_pct': None},  # set per-trade
    'D': {'trail_K': 3.0, 'trail_activation_pct': 0.0},   # no activation
    'E': {'trail_K': 2.5, 'trail_activation_pct': None},  # tiered
    'F': {'trail_K': 2.5, 'trail_activation_pct': None},  # gated by +2 ATR
    'G': {'trail_K': 1.5, 'trail_activation_pct': 0.05},
    'H': {'trail_K': 3.5, 'trail_activation_pct': 0.05},
    'I': {'trail_K': 2.5, 'trail_activation_pct': 0.05},
}

VARIANT_DESC = {
    'A': 'Baseline (K=2.5, act=0.05%)',
    'B': 'Activation 0.5% (K=2.5)',
    'C': 'Dynamic activation 1×ATR',
    'D': 'Chandelier K=3.0 ($ stop from best)',
    'E': 'Step-up tiers (1/2/3 ATR)',
    'F': 'Break-even @ 1ATR + trail @ 2ATR',
    'G': 'Tighter trail (K=1.5)',
    'H': 'Looser trail (K=3.5)',
    'I': 'Partial scale-out 50% @ 1.5ATR + trail',
}


# ═══════════════════════════════════════════════════════════════════════
# Backtest core
# ═══════════════════════════════════════════════════════════════════════

def backtest_variant(variant, opens, highs, lows, closes, n,
                     channel_period=15, body_min_ratio=0.4,
                     atr_period=14, max_sl_atr=3.3,
                     fractal_lookback=10, sl_min_pct=0.15, sl_max_pct=3.0,
                     min_bars_between=2, emergency_sl_pct=3.0,
                     timeout_bars=192):
    o = opens.astype(float); h = highs.astype(float)
    l = lows.astype(float); c = closes.astype(float)
    atr_vals, ch_high, ch_low, sw_low, sw_high = precompute(
        o, h, l, c, channel_period, atr_period, fractal_lookback)

    params = VARIANT_PARAMS[variant]
    dispatch = EXIT_DISPATCHERS[variant]

    positions = []
    trades = []
    last_entry_bar = -10
    warmup = max(channel_period + 10, 25, atr_period + fractal_lookback)

    for bar in range(warmup, n):
        # ── Exit loop ──
        closed = []
        for idx, pos in enumerate(positions):
            ep = pos['entry_price']
            d = pos['direction']
            bh = bar - pos['entry_bar']

            # Update best price BEFORE exit checks (intrabar extreme)
            if d == 'LONG':
                pos['best_price'] = max(pos['best_price'], h[bar])
            else:
                pos['best_price'] = min(pos['best_price'], l[bar])

            exit_info = None

            # 1. SL (intrabar)
            if d == 'LONG' and l[bar] <= pos['sl_price']:
                exit_info = {'reason': 'SL', 'exit_price': pos['sl_price']}
            elif d == 'SHORT' and h[bar] >= pos['sl_price']:
                exit_info = {'reason': 'SL', 'exit_price': pos['sl_price']}

            # 2. Emergency SL
            if exit_info is None:
                if d == 'LONG':
                    worst_pnl = (l[bar] / ep - 1) * 100
                else:
                    worst_pnl = (1 - h[bar] / ep) * 100
                if worst_pnl <= -emergency_sl_pct:
                    if d == 'LONG':
                        ex_px = ep * (1 - emergency_sl_pct / 100)
                    else:
                        ex_px = ep * (1 + emergency_sl_pct / 100)
                    exit_info = {'reason': 'EMERGENCY', 'exit_price': ex_px}

            # 3. Timeout
            if exit_info is None and bh >= timeout_bars:
                exit_info = {'reason': 'TIMEOUT', 'exit_price': c[bar]}

            # 4. Variant-specific trail
            if exit_info is None:
                atr_now = atr_vals[bar] if not math.isnan(atr_vals[bar]) else 0
                exit_info = dispatch(pos, h[bar], l[bar], c[bar], atr_now, params)

            if exit_info is not None:
                # Build trade record
                exit_price = exit_info['exit_price']
                # Variant I: if TP1 already fired (at any earlier bar), the
                # remaining 50% is what exits here regardless of reason
                # (SL/Emergency/Timeout/Trail). Weighted PnL = 0.5*TP1 + 0.5*exit.
                if variant == 'I' and pos.get('tp1_hit', False):
                    tp1_px = pos['tp1_exit_price']
                    if d == 'LONG':
                        raw_tp1 = _pnl_long(ep, tp1_px)
                        raw_trail = _pnl_long(ep, exit_price)
                    else:
                        raw_tp1 = _pnl_short(ep, tp1_px)
                        raw_trail = _pnl_short(ep, exit_price)
                    raw_pnl = 0.5 * raw_tp1 + 0.5 * raw_trail
                else:
                    if d == 'LONG':
                        raw_pnl = _pnl_long(ep, exit_price)
                    else:
                        raw_pnl = _pnl_short(ep, exit_price)

                trade_pnl = raw_pnl * LEVERAGE - FEE_PCT
                trades.append({
                    'entry_bar': pos['entry_bar'],
                    'exit_bar': bar,
                    'direction': d,
                    'pnl': trade_pnl,
                    'reason': exit_info['reason'],
                    'bars_held': bh,
                    'entry_price': ep,
                    'exit_price': exit_price,
                })
                closed.append(idx)

        for idx in sorted(closed, reverse=True):
            positions.pop(idx)

        # ── Entry ──
        if (len(positions) < MAX_POSITIONS and bar + 1 < n and
                bar - last_entry_bar >= min_bars_between and
                not math.isnan(atr_vals[bar]) and atr_vals[bar] > 0):

            sig = check_entry(
                o[bar], h[bar], l[bar], c[bar],
                ch_high[bar], ch_low[bar], atr_vals[bar],
                sw_low[bar], sw_high[bar],
                body_min_ratio=body_min_ratio, max_sl_atr=max_sl_atr,
                sl_min_pct=sl_min_pct, sl_max_pct=sl_max_pct,
            )
            if sig is not None:
                entry_price = o[bar + 1]
                direction = sig['direction']
                sl_price = compute_sl(
                    direction, entry_price, atr_vals[bar],
                    sw_low[bar], sw_high[bar],
                    max_sl_atr=max_sl_atr,
                    sl_min_pct=sl_min_pct, sl_max_pct=sl_max_pct,
                )
                if sl_price is None:
                    continue

                pos = {
                    'entry_bar': bar + 1,
                    'entry_price': entry_price,
                    'direction': direction,
                    'sl_price': sl_price,
                    'best_price': entry_price,
                    'atr_entry': atr_vals[bar],
                }
                # Variant-specific init
                if variant == 'C':
                    # Dynamic activation threshold = 1 × ATR / entry * 100
                    pos['c_activation_pct'] = atr_vals[bar] / entry_price * 100
                positions.append(pos)
                last_entry_bar = bar

    return trades


# ═══════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════

def compound_equity(trades):
    eq = 100.0
    peak = 100.0
    mdd = 0.0
    for t in sorted(trades, key=lambda x: x['exit_bar']):
        eq += t['pnl'] * (SIZE_PCT / 100) * (eq / 100)
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        mdd = max(mdd, dd)
    return eq - 100, mdd


def additive_pnl(trades):
    return sum(t['pnl'] for t in trades)


def stats_for_trades(trades, n_bars):
    if not trades:
        return {
            'trades': 0, 'wr': 0, 'rr': 0,
            'pnl_additive_1x': 0, 'pnl_compound': 0, 'mdd': 0,
            'trades_per_day': 0, 'daily_pnl': 0,
            'max_winner': 0, 'max_loser': 0,
            'avg_bars_held': 0, 'exits': {},
            'trail_bins': {'sl': 0, 'zero_trail': 0, 'pos_trail': 0,
                           'neg_trail': 0, 'emergency': 0, 'timeout': 0},
        }
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pnl_add = sum(pnls)
    pnl_comp, mdd = compound_equity(trades)
    wr = len(wins) / len(trades) * 100
    wa = np.mean(wins) if wins else 0
    la = np.mean(losses) if losses else 0
    rr = abs(wa / la) if la != 0 else float('inf')
    n_days = n_bars * 15 / 1440

    # Exit breakdown — with zero-epsilon binning
    reasons = defaultdict(int)
    for t in trades:
        reasons[t['reason']] += 1

    # Trail bins: SL / EMERGENCY / TIMEOUT by reason; TRAIL_TP split by PnL sign
    bins = {'sl': 0, 'zero_trail': 0, 'pos_trail': 0, 'neg_trail': 0,
            'emergency': 0, 'timeout': 0}
    for t in trades:
        r = t['reason']
        p = t['pnl']
        if r == 'SL':
            bins['sl'] += 1
        elif r == 'EMERGENCY':
            bins['emergency'] += 1
        elif r == 'TIMEOUT':
            bins['timeout'] += 1
        elif r == 'TRAIL_TP':
            if abs(p) < ZERO_EPS:
                bins['zero_trail'] += 1
            elif p > 0:
                bins['pos_trail'] += 1
            else:
                bins['neg_trail'] += 1

    return {
        'trades': len(trades),
        'wr': round(wr, 2),
        'rr': round(rr, 2) if not math.isinf(rr) else float('inf'),
        'pnl_additive_1x': round(pnl_add, 2),
        'pnl_compound': round(pnl_comp, 2),
        'mdd': round(mdd, 2),
        'trades_per_day': round(len(trades) / n_days, 2) if n_days > 0 else 0,
        'daily_pnl': round(pnl_add / n_days, 4) if n_days > 0 else 0,
        'max_winner': round(max(pnls), 3),
        'max_loser': round(min(pnls), 3),
        'avg_bars_held': round(np.mean([t['bars_held'] for t in trades]), 1),
        'exits': dict(reasons),
        'trail_bins': bins,
        'pnl_mdd': round(pnl_add / mdd, 2) if mdd > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Walk-Forward (5-fold expanding window)
# ═══════════════════════════════════════════════════════════════════════

def walk_forward(variant, df):
    n = len(df)
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values

    folds = []
    for fi in range(5):
        is_end = int(n * (fi + 1) / (5 + 1))
        oos_end = int(n * (fi + 2) / (5 + 1))
        oos_n = oos_end - is_end
        oos_trades = backtest_variant(
            variant, o[is_end:oos_end], h[is_end:oos_end],
            l[is_end:oos_end], c[is_end:oos_end], oos_n)
        oos_stats = stats_for_trades(oos_trades, oos_n)
        folds.append(oos_stats['pnl_additive_1x'])
    passes = sum(1 for p in folds if p > 0)
    return folds, passes


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("  Trail Alternatives Comparison — C1 Breakout v2.6 (N=1, LEV=1)")
    print(f"  Fee={FEE_PCT}% RT | Size={SIZE_PCT}% | Emergency={EMERGENCY_SL}%")
    print(f"  Zero-bin epsilon: |pnl| < {ZERO_EPS}%")
    print("=" * 78)

    df = load_15m_data()
    n = len(df)
    n_days = n * 15 / 1440
    print(f"\n  Data: {n} bars ({n_days:.0f} days), "
          f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    all_results = {'meta': {
        'n_bars': n, 'n_days': round(n_days, 1),
        'fee_pct': FEE_PCT, 'leverage': LEVERAGE,
        'max_positions': MAX_POSITIONS, 'size_pct': SIZE_PCT,
        'zero_eps_pct': ZERO_EPS,
        'period_start': str(df['timestamp'].iloc[0]),
        'period_end': str(df['timestamp'].iloc[-1]),
    }, 'variants': {}}

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values

    for vkey in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        print("\n" + "=" * 78)
        print(f"  VARIANT {vkey}: {VARIANT_DESC[vkey]}")
        print("=" * 78)

        trades = backtest_variant(vkey, o, h, l, c, n)
        s = stats_for_trades(trades, n)

        print(f"  Trades:          {s['trades']}")
        print(f"  WR:              {s['wr']}%")
        print(f"  R:R:             {s['rr']}")
        print(f"  PnL (add 1x):    {s['pnl_additive_1x']:+.2f}%")
        print(f"  PnL (compound):  {s['pnl_compound']:+.2f}%")
        print(f"  MDD:             {s['mdd']:.2f}%")
        print(f"  PnL/MDD:         {s['pnl_mdd']}")
        print(f"  Trades/day:      {s['trades_per_day']}")
        print(f"  Max winner:      {s['max_winner']:+.3f}%")
        print(f"  Max loser:       {s['max_loser']:+.3f}%")
        print(f"  Avg bars held:   {s['avg_bars_held']}")
        print(f"  Exits (raw):     {s['exits']}")
        b = s['trail_bins']
        total = s['trades'] if s['trades'] else 1
        print(f"  Breakdown (% of trades):")
        print(f"    SL:          {b['sl']:>4} ({b['sl']/total*100:.1f}%)")
        print(f"    Trail zero:  {b['zero_trail']:>4} ({b['zero_trail']/total*100:.1f}%)  [|pnl|<{ZERO_EPS}%]")
        print(f"    Trail pos:   {b['pos_trail']:>4} ({b['pos_trail']/total*100:.1f}%)")
        print(f"    Trail neg:   {b['neg_trail']:>4} ({b['neg_trail']/total*100:.1f}%)")
        print(f"    Emergency:   {b['emergency']:>4} ({b['emergency']/total*100:.1f}%)")
        print(f"    Timeout:     {b['timeout']:>4} ({b['timeout']/total*100:.1f}%)")

        # Walk-Forward
        wf_folds, wf_passes = walk_forward(vkey, df)
        print(f"\n  Walk-Forward 5-fold OOS (additive 1x per fold):")
        for i, p in enumerate(wf_folds):
            print(f"    F{i+1}: {p:+.2f}% {'[P]' if p > 0 else '[F]'}")
        print(f"  WF PASS: {wf_passes}/5 "
              f"{'[ROBUST]' if wf_passes >= 3 else '[FRAGILE]'}")

        all_results['variants'][vkey] = {
            'desc': VARIANT_DESC[vkey],
            'params': VARIANT_PARAMS[vkey],
            'stats': s,
            'wf_folds_oos': [round(p, 2) for p in wf_folds],
            'wf_passes_of_5': wf_passes,
        }

        # Incremental save
        with open(RESULTS_PATH, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # ═══ Summary ═══
    print("\n" + "=" * 78)
    print("  SUMMARY — Ranked Comparison")
    print("=" * 78)

    rows = []
    for vkey, v in all_results['variants'].items():
        s = v['stats']
        per_trade = s['pnl_additive_1x'] / s['trades'] if s['trades'] else 0
        rows.append({
            'v': vkey, 'desc': VARIANT_DESC[vkey],
            'pnl': s['pnl_additive_1x'], 'trades': s['trades'],
            'wr': s['wr'], 'rr': s['rr'], 'mdd': s['mdd'],
            'pnl_mdd': s['pnl_mdd'],
            'per_trade': round(per_trade, 4),
            'max_win': s['max_winner'], 'wf_pass': v['wf_passes_of_5'],
            'zero_pct': round(s['trail_bins']['zero_trail'] /
                              max(s['trades'], 1) * 100, 1),
            'neg_trail_pct': round(s['trail_bins']['neg_trail'] /
                                   max(s['trades'], 1) * 100, 1),
        })

    # Table
    hdr = (f"  {'V':<2} {'Desc':<38} {'PnL%':>7} {'Trd':>4} "
           f"{'WR%':>5} {'R:R':>5} {'MDD%':>6} {'P/MDD':>6} "
           f"{'$/trd':>6} {'MaxW':>6} {'Zero%':>6} {'NegT%':>6} {'WF':>3}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    # Ranked by PnL
    print("\n  [Ranked by Total PnL (additive 1x)]")
    for r in sorted(rows, key=lambda x: -x['pnl']):
        print(f"  {r['v']:<2} {r['desc']:<38} {r['pnl']:>+7.1f} "
              f"{r['trades']:>4} {r['wr']:>5.1f} {r['rr']:>5.2f} "
              f"{r['mdd']:>6.2f} {r['pnl_mdd']:>6.2f} "
              f"{r['per_trade']:>+6.3f} {r['max_win']:>+6.2f} "
              f"{r['zero_pct']:>6.1f} {r['neg_trail_pct']:>6.1f} "
              f"{r['wf_pass']:>1}/5")

    print("\n  [Ranked by PnL/MDD (risk-adjusted)]")
    for r in sorted(rows, key=lambda x: -x['pnl_mdd']):
        print(f"  {r['v']:<2} {r['desc']:<38} PnL={r['pnl']:>+7.1f}% "
              f"MDD={r['mdd']:>5.2f}% PnL/MDD={r['pnl_mdd']:>5.2f} "
              f"WF={r['wf_pass']}/5")

    print("\n  [Ranked by PnL per trade (efficiency)]")
    for r in sorted(rows, key=lambda x: -x['per_trade']):
        print(f"  {r['v']:<2} {r['desc']:<38} per-trade={r['per_trade']:>+7.4f}% "
              f"trades={r['trades']:>4} total={r['pnl']:>+7.1f}%")

    # Verdict
    print("\n" + "=" * 78)
    print("  VERDICTS")
    print("=" * 78)
    baseline = next(r for r in rows if r['v'] == 'A')
    for r in rows:
        vkey = r['v']
        verdict = []
        if r['pnl'] > baseline['pnl']:
            verdict.append(f"PnL +{r['pnl']-baseline['pnl']:.1f}pp vs A")
        else:
            verdict.append(f"PnL {r['pnl']-baseline['pnl']:+.1f}pp vs A")
        if r['zero_pct'] < baseline['zero_pct']:
            verdict.append(f"fewer zero exits ({r['zero_pct']}% < {baseline['zero_pct']}%)")
        if r['mdd'] < baseline['mdd']:
            verdict.append(f"lower MDD")
        elif r['mdd'] > baseline['mdd'] * 1.2:
            verdict.append(f"higher MDD")
        if r['wf_pass'] >= 3:
            verdict.append(f"WF {r['wf_pass']}/5 OK")
        else:
            verdict.append(f"WF {r['wf_pass']}/5 WEAK")
        print(f"  {vkey}: {'; '.join(verdict)}")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
