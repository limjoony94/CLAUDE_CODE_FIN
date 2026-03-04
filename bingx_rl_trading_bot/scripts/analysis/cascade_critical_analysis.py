#!/usr/bin/env python3
"""
Cascade SL Critical Analysis
==============================

Critical stress-testing of cascade_t75 results for consistency,
statistical significance, and overfitting concerns.

Tests:
  1. PnL Decomposition   — Where does cascade advantage come from?
  2. Bootstrap CI         — Is the difference statistically significant?
  3. Temporal Consistency — Does it work in all sub-periods?
  4. Cluster Paradox      — Are cascade-induced clusters real or artificial?
  5. Monotonicity Check   — Is the tighten_pct relationship smooth?
  6. Compound vs Simple   — Does advantage persist without compound amplification?
  7. Direction Cap Sens.  — Is cascade a workaround for excessive same-dir exposure?

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE = 3 FIXED (v1.42.0: adaptive disabled)
  - FEE = FEE_PCT * LEVERAGE (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing (except Test 6)

Author: Research Agent
Date: 2026-03-04
"""

import os
import sys
import json
import warnings
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

warnings.filterwarnings('ignore')

# ============================================================
# Constants -- matching production v1.42.0 exactly
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

# v1.42.0: Regime sizing DISABLED
REGIME_MULT = None

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'cascade_critical_analysis.json')


# ============================================================
# Data Loading & Preprocessing (from cascade_sl_depth_study.py)
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
# Portfolio Simulator with trade tagging for decomposition
# ============================================================

def portfolio_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                  atr_ratio, ema_slope, start_bar, end_bar,
                  cascade_tighten_pct=0.0,
                  direction_cap_override=None,
                  use_simple_sizing=False,
                  tag_trades=False):
    """N-pos portfolio sim with production v1.42.0 features.

    Production features always ON:
      - N=9 compound, direction_cap=7 (or override), agg_risk_cap(3/7%),
        momentum_guard, timeout(864 DROP), ATR-scaled TP/SL,
        FIXED leverage=3, MDD sizing
      - v1.42.0: M2 Regime sizing DISABLED, M4 Adaptive leverage DISABLED

    Additional modes:
      - tag_trades: tag each trade as 'organic', 'cascade_forced', or 'post_cascade'
      - use_simple_sizing: additive PnL instead of compound
      - direction_cap_override: override DIRECTION_CAP
    """
    dir_cap = direction_cap_override if direction_cap_override is not None else DIRECTION_CAP
    size_pct = 100.0 / N_SLOTS
    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}
    cluster_events = []

    # For trade tagging (Test 1 decomposition)
    # Track extra capacity created by cascade forced exits.
    # cascade_extra_capacity counts how many slots were freed EARLY by cascade
    # that would not have been freed at that bar without cascade.
    cascade_extra_capacity = 0
    cascade_modified_slots = set()  # Slot IDs with modified sl_pct

    # For cluster paradox (Test 4)
    # Track which positions have been cascade-modified
    # so we can classify cluster events as natural vs cascade-induced

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

                if use_simple_sizing:
                    pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                else:
                    pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                result['size_mult'] = sm
                result['leverage'] = LEVERAGE

                # Tagging for decomposition
                if tag_trades:
                    slot_id = pos['slot']
                    if pos.get('cascade_modified', False):
                        result['trade_tag'] = 'cascade_forced'
                    elif pos.get('post_cascade_entry', False):
                        result['trade_tag'] = 'post_cascade'
                    else:
                        result['trade_tag'] = 'organic'
                    result['sl_was_modified'] = pos.get('cascade_modified', False)
                    result['original_sl_pct'] = pos.get('original_sl_pct', pos['sl_pct'])
                    result['final_sl_pct'] = pos['sl_pct']

                trades.append(result)
                closed_slots.append(pos['slot'])

                if not use_simple_sizing:
                    bar_pnl_sum += pnl_portfolio
                # simple sizing: add to running sum at end

                if result['reason'] == 'SL':
                    bar_sl_exits.append({
                        'direction': pos['direction'],
                        'pnl_portfolio': pnl_portfolio,
                        'pattern': pos['pattern'],
                        'cascade_modified': pos.get('cascade_modified', False),
                    })

                    # If this was a cascade-modified position that hit SL,
                    # it exited earlier than it would have without cascade -> extra capacity
                    if tag_trades and pos.get('cascade_modified', False):
                        cascade_extra_capacity += 1

        positions = [p for p in positions if p['slot'] not in closed_slots]
        # Clean up cascade tracking for closed slots
        for slot_id in closed_slots:
            cascade_modified_slots.discard(slot_id)

        # --- Post-SL Actions: Cascade SL Tightening ---
        for direction in ['LONG', 'SHORT']:
            dir_sl_exits = [e for e in bar_sl_exits if e['direction'] == direction]
            if not dir_sl_exits:
                continue

            same_dir_remaining = [p for p in positions if p['direction'] == direction]
            if not same_dir_remaining:
                continue

            if cascade_tighten_pct > 0:
                for other_pos in same_dir_remaining:
                    old_sl = other_pos['sl_pct']
                    other_pos['sl_pct'] = old_sl * (1 - cascade_tighten_pct / 100)
                    other_pos['cascade_modified'] = True
                    cascade_modified_slots.add(other_pos['slot'])

        # Detect cluster events (2+ same-dir SL exits in same bar)
        for direction in ['LONG', 'SHORT']:
            dir_exits = [e for e in bar_sl_exits if e['direction'] == direction]
            if len(dir_exits) >= 2:
                total_loss = sum(e['pnl_portfolio'] for e in dir_exits)
                any_cascade_modified = any(e.get('cascade_modified', False) for e in dir_exits)
                cluster_events.append({
                    'bar': bar, 'direction': direction,
                    'n_hits': len(dir_exits), 'total_loss': total_loss,
                    'has_cascade_modified': any_cascade_modified,
                })

        if use_simple_sizing:
            # Simple: just add PnL contributions
            bar_pnl_sum = sum(t['pnl_portfolio'] for t in trades
                              if t.get('exit_bar') == bar and not t.get('drop', False))

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
            if dir_count >= dir_cap:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            if bar < momentum_pause_until.get(direction, -1):
                continue

            sm = 1.0
            if not use_simple_sizing:
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

            slot_id = f"{pat}_{sig_bar}"

            # Determine if this is a post-cascade entry:
            # If cascade has created extra capacity (freed slots earlier than
            # baseline would), then new entries that benefit from that capacity
            # are tagged as post_cascade. Decrement capacity when used.
            is_post_cascade = False
            if tag_trades and cascade_extra_capacity > 0:
                is_post_cascade = True
                cascade_extra_capacity -= 1

            positions.append({
                'slot': slot_id, 'signal_bar': sig_bar,
                'entry_bar': entry_bar, 'direction': direction,
                'pattern': pat, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'original_sl_pct': sl_pct,
                'size_mult': sm, 'leverage': LEVERAGE,
                'cascade_modified': False,
                'post_cascade_entry': is_post_cascade,
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
        trade_entry = {
            'entry_bar': pos['entry_bar'], 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
            'leverage': LEVERAGE,
        }
        if tag_trades:
            if pos.get('cascade_modified', False):
                trade_entry['trade_tag'] = 'cascade_forced'
            elif pos.get('post_cascade_entry', False):
                trade_entry['trade_tag'] = 'post_cascade'
            else:
                trade_entry['trade_tag'] = 'organic'
        trades.append(trade_entry)

    return trades, equity_history, cluster_events, max_dd


def calc_stats(trades, equity_history):
    """Compute summary stats from trades."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0}

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

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(total_pnl / mdd, 2) if mdd > 0 else 0,
    }


# ============================================================
# TEST 1: PnL Decomposition
# ============================================================

def test1_pnl_decomposition(signal_tuples, opens, highs, lows, closes, n_bars,
                             atr_ratio, ema_slope, start_bar, end_bar):
    """Decompose cascade_t75 PnL into organic, cascade_forced, post_cascade."""
    print("\n[TEST 1: PNL DECOMPOSITION]")
    print("-" * 70)
    print("Question: WHERE does the cascade PnL advantage come from?")
    print("  (a) Loss reduction on forced exits?")
    print("  (b) Slot recycling (new entries in freed slots)?")

    # Run baseline (no cascade, with tagging for comparison)
    baseline_trades, baseline_eq, _, _ = portfolio_sim(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        cascade_tighten_pct=0.0, tag_trades=True)
    baseline_stats = calc_stats(baseline_trades, baseline_eq)

    # Run cascade_t75 with tagging
    cascade_trades, cascade_eq, _, _ = portfolio_sim(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        cascade_tighten_pct=75.0, tag_trades=True)
    cascade_stats = calc_stats(cascade_trades, cascade_eq)

    # Decompose cascade trades by tag
    organic = [t for t in cascade_trades if t.get('trade_tag') == 'organic']
    forced = [t for t in cascade_trades if t.get('trade_tag') == 'cascade_forced']
    post_cascade = [t for t in cascade_trades if t.get('trade_tag') == 'post_cascade']

    organic_pnl = sum(t['pnl_portfolio'] for t in organic)
    forced_pnl = sum(t['pnl_portfolio'] for t in forced)
    post_cascade_pnl = sum(t['pnl_portfolio'] for t in post_cascade)

    forced_avg = np.mean([t['pnl_portfolio'] for t in forced]) if forced else 0
    post_cascade_avg = np.mean([t['pnl_portfolio'] for t in post_cascade]) if post_cascade else 0

    # Baseline organic (all trades are effectively organic)
    baseline_pnl = sum(t['pnl_portfolio'] for t in baseline_trades)

    print(f"\n  Baseline: {len(baseline_trades)} trades, PnL {baseline_pnl:+.1f}%")
    print(f"  Cascade:  {len(cascade_trades)} trades, PnL {cascade_stats['pnl']:+.1f}%")
    print()
    print(f"  PnL Decomposition (cascade_t75):")
    print(f"    Organic trades:          {len(organic):>4} trades, PnL {organic_pnl:>+8.1f}%")
    print(f"    Cascade forced exits:    {len(forced):>4} trades, avg PnL {forced_avg:>+6.2f}%, "
          f"total PnL {forced_pnl:>+8.1f}%")
    print(f"    Post-cascade new entries: {len(post_cascade):>4} trades, avg PnL {post_cascade_avg:>+6.2f}%, "
          f"total PnL {post_cascade_pnl:>+8.1f}%")
    print(f"    Baseline organic equiv:  {len(baseline_trades):>4} trades, PnL {baseline_pnl:>+8.1f}%")

    # Analyze forced exit PnL distribution
    if forced:
        forced_pnls = [t['pnl_portfolio'] for t in forced]
        wins_forced = sum(1 for p in forced_pnls if p > 0)
        print(f"\n  Forced exit analysis:")
        print(f"    Wins: {wins_forced}/{len(forced)} ({wins_forced/len(forced)*100:.1f}%)")
        print(f"    Mean PnL: {np.mean(forced_pnls):+.3f}%")
        print(f"    Median PnL: {np.median(forced_pnls):+.3f}%")

        # Compare forced exits to what would have happened (baseline SL)
        forced_sl_pcts = [t.get('original_sl_pct', 0) for t in forced]
        forced_final_sl = [t.get('final_sl_pct', 0) for t in forced]
        if forced_sl_pcts and forced_final_sl:
            avg_original = np.mean(forced_sl_pcts)
            avg_final = np.mean(forced_final_sl)
            print(f"    Avg original SL: {avg_original:.3f}%")
            print(f"    Avg tightened SL: {avg_final:.3f}%")
            print(f"    SL reduction: {(1 - avg_final/avg_original)*100:.1f}%")

    # Interpretation
    print(f"\n  Interpretation:")
    if forced_pnl > 0:
        print(f"    Cascade forced exits contribute POSITIVE PnL ({forced_pnl:+.1f}%)")
        print(f"    -> Cascade cuts losses early, turning some into small wins")
    else:
        saved_loss = baseline_pnl - organic_pnl
        print(f"    Cascade forced exits contribute NEGATIVE PnL ({forced_pnl:+.1f}%)")
        if abs(forced_pnl) < abs(saved_loss) and post_cascade_pnl > 0:
            print(f"    -> Advantage from slot recycling: post-cascade entries add {post_cascade_pnl:+.1f}%")
        elif organic_pnl > baseline_pnl:
            print(f"    -> Organic trades also differ ({organic_pnl:+.1f}% vs baseline {baseline_pnl:+.1f}%)")
            print(f"    -> Path-dependent: cascade changes which trades enter subsequently")

    return {
        'baseline_trades': len(baseline_trades),
        'baseline_pnl': round(baseline_pnl, 2),
        'cascade_trades': len(cascade_trades),
        'cascade_pnl': round(cascade_stats['pnl'], 2),
        'organic_trades': len(organic),
        'organic_pnl': round(organic_pnl, 2),
        'forced_trades': len(forced),
        'forced_pnl': round(forced_pnl, 2),
        'forced_avg_pnl': round(forced_avg, 4),
        'post_cascade_trades': len(post_cascade),
        'post_cascade_pnl': round(post_cascade_pnl, 2),
        'post_cascade_avg_pnl': round(post_cascade_avg, 4),
    }


# ============================================================
# TEST 2: Bootstrap CI
# ============================================================

def test2_bootstrap_ci(signal_tuples, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, start_bar, end_bar):
    """Bootstrap confidence interval for baseline vs cascade_t75 difference."""
    print("\n[TEST 2: BOOTSTRAP CONFIDENCE INTERVAL]")
    print("-" * 70)
    print("Question: Is cascade_t75 statistically better than baseline?")

    # Get per-trade PnLs
    baseline_trades, _, _, _ = portfolio_sim(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        cascade_tighten_pct=0.0)

    cascade_trades, _, _, _ = portfolio_sim(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        cascade_tighten_pct=75.0)

    baseline_pnls = np.array([t['pnl_portfolio'] for t in baseline_trades])
    cascade_pnls = np.array([t['pnl_portfolio'] for t in cascade_trades])

    rng = np.random.RandomState(42)
    n_boot = 1000

    baseline_totals = np.zeros(n_boot)
    cascade_totals = np.zeros(n_boot)

    for i in range(n_boot):
        b_idx = rng.choice(len(baseline_pnls), size=len(baseline_pnls), replace=True)
        c_idx = rng.choice(len(cascade_pnls), size=len(cascade_pnls), replace=True)
        baseline_totals[i] = np.sum(baseline_pnls[b_idx])
        cascade_totals[i] = np.sum(cascade_pnls[c_idx])

    diff_totals = cascade_totals - baseline_totals

    b_lo, b_hi = np.percentile(baseline_totals, [2.5, 97.5])
    c_lo, c_hi = np.percentile(cascade_totals, [2.5, 97.5])
    d_lo, d_hi = np.percentile(diff_totals, [2.5, 97.5])

    actual_baseline = np.sum(baseline_pnls)
    actual_cascade = np.sum(cascade_pnls)
    actual_diff = actual_cascade - actual_baseline

    print(f"\n  Baseline: {len(baseline_pnls)} trades")
    print(f"    Actual total PnL: {actual_baseline:+.2f}%")
    print(f"    95% CI: [{b_lo:+.2f}%, {b_hi:+.2f}%]")

    print(f"\n  Cascade_t75: {len(cascade_pnls)} trades")
    print(f"    Actual total PnL: {actual_cascade:+.2f}%")
    print(f"    95% CI: [{c_lo:+.2f}%, {c_hi:+.2f}%]")

    print(f"\n  Difference (cascade - baseline):")
    print(f"    Actual: {actual_diff:+.2f}%")
    print(f"    95% CI: [{d_lo:+.2f}%, {d_hi:+.2f}%]")

    includes_zero = d_lo <= 0 <= d_hi
    print(f"\n  CI includes zero: {'YES' if includes_zero else 'NO'}")

    if includes_zero:
        print(f"  CONCLUSION: Cascade_t75 is NOT statistically significantly better.")
        print(f"  The observed advantage may be due to chance.")
        significant = False
    else:
        if d_lo > 0:
            print(f"  CONCLUSION: Cascade_t75 is statistically significantly BETTER.")
            significant = True
        else:
            print(f"  CONCLUSION: Cascade_t75 is statistically significantly WORSE.")
            significant = True

    # Additional: what fraction of bootstrap samples show cascade > baseline?
    pct_better = np.mean(diff_totals > 0) * 100
    print(f"\n  Bootstrap samples where cascade > baseline: {pct_better:.1f}%")

    return {
        'baseline_pnl': round(actual_baseline, 2),
        'cascade_pnl': round(actual_cascade, 2),
        'difference': round(actual_diff, 2),
        'baseline_ci_95': [round(b_lo, 2), round(b_hi, 2)],
        'cascade_ci_95': [round(c_lo, 2), round(c_hi, 2)],
        'difference_ci_95': [round(d_lo, 2), round(d_hi, 2)],
        'ci_includes_zero': includes_zero,
        'pct_cascade_better': round(pct_better, 1),
        'significant': significant,
    }


# ============================================================
# TEST 3: Temporal Consistency
# ============================================================

def test3_temporal_consistency(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, start_bar, end_bar):
    """Split neutral window into 4 quarters and run each independently."""
    print("\n[TEST 3: TEMPORAL CONSISTENCY -- SUB-PERIOD ANALYSIS]")
    print("-" * 70)
    print("Question: Does cascade_t75 outperform baseline in ALL quarters?")

    total_bars = end_bar - start_bar
    quarter_size = total_bars // 4
    quarters = []
    for q in range(4):
        q_start = start_bar + q * quarter_size
        q_end = start_bar + (q + 1) * quarter_size if q < 3 else end_bar
        quarters.append((q_start, q_end))

    results = []
    cascade_wins = 0
    print(f"\n  {'Quarter':<10} {'Bars':>6} {'BL PnL%':>9} {'BL MDD%':>8} {'BL Trades':>9} "
          f"{'C75 PnL%':>9} {'C75 MDD%':>8} {'C75 Trades':>10} {'Winner':>8}")
    print("  " + "-" * 90)

    for q_idx, (q_start, q_end) in enumerate(quarters):
        q_name = f"Q{q_idx + 1}"
        q_bars = q_end - q_start

        # Baseline
        bl_trades, bl_eq, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, q_start, q_end,
            cascade_tighten_pct=0.0)
        bl_stats = calc_stats(bl_trades, bl_eq)

        # Cascade
        c75_trades, c75_eq, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, q_start, q_end,
            cascade_tighten_pct=75.0)
        c75_stats = calc_stats(c75_trades, c75_eq)

        winner = "cascade" if c75_stats['pnl'] > bl_stats['pnl'] else "baseline"
        if c75_stats['pnl'] > bl_stats['pnl']:
            cascade_wins += 1

        print(f"  {q_name:<10} {q_bars:>6} {bl_stats['pnl']:>+9.2f} {bl_stats['mdd']:>8.2f} "
              f"{bl_stats['trades']:>9} {c75_stats['pnl']:>+9.2f} {c75_stats['mdd']:>8.2f} "
              f"{c75_stats['trades']:>10} {winner:>8}")

        results.append({
            'quarter': q_name,
            'bars': q_bars,
            'baseline': {'pnl': bl_stats['pnl'], 'mdd': bl_stats['mdd'], 'trades': bl_stats['trades']},
            'cascade': {'pnl': c75_stats['pnl'], 'mdd': c75_stats['mdd'], 'trades': c75_stats['trades']},
            'winner': winner,
        })

    consistency = cascade_wins == 4
    print(f"\n  Cascade wins: {cascade_wins}/4 quarters")
    if consistency:
        print(f"  CONCLUSION: Cascade_t75 is temporally CONSISTENT (wins all 4 quarters).")
    elif cascade_wins >= 3:
        print(f"  CONCLUSION: Cascade_t75 is MOSTLY consistent ({cascade_wins}/4 quarters).")
    else:
        print(f"  CONCLUSION: Cascade_t75 is INCONSISTENT -- only wins {cascade_wins}/4 quarters.")
        print(f"  This suggests overfitting to specific market conditions.")

    return {
        'quarters': results,
        'cascade_wins': cascade_wins,
        'consistent': consistency,
    }


# ============================================================
# TEST 4: Cluster Loss Paradox
# ============================================================

def test4_cluster_paradox(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, start_bar, end_bar):
    """Decompose cluster events into natural vs cascade-induced."""
    print("\n[TEST 4: CLUSTER LOSS PARADOX]")
    print("-" * 70)
    print("Question: Does cascade CREATE more cluster events by tightening SLs?")
    print("  cascade_t75 has more cluster events than baseline, but are they 'real'?")

    # Baseline clusters
    bl_trades, bl_eq, bl_clusters, _ = portfolio_sim(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        cascade_tighten_pct=0.0)

    # Cascade clusters (with cascade_modified flag)
    c75_trades, c75_eq, c75_clusters, _ = portfolio_sim(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        cascade_tighten_pct=75.0)

    # Classify cascade clusters
    natural_clusters = [c for c in c75_clusters if not c.get('has_cascade_modified', False)]
    cascade_induced = [c for c in c75_clusters if c.get('has_cascade_modified', False)]

    bl_cluster_loss = sum(c['total_loss'] for c in bl_clusters)
    natural_loss = sum(c['total_loss'] for c in natural_clusters)
    induced_loss = sum(c['total_loss'] for c in cascade_induced)
    c75_cluster_loss = sum(c['total_loss'] for c in c75_clusters)

    print(f"\n  Baseline clusters:")
    print(f"    Events: {len(bl_clusters)}")
    print(f"    Total cluster loss: {bl_cluster_loss:+.2f}%")

    print(f"\n  Cascade_t75 clusters:")
    print(f"    Total events: {len(c75_clusters)}")
    print(f"    Natural clusters (no cascade-modified positions): {len(natural_clusters)}")
    print(f"    Cascade-induced clusters (includes modified SLs): {len(cascade_induced)}")
    print(f"    Total cluster loss: {c75_cluster_loss:+.2f}%")
    print(f"      Natural cluster loss: {natural_loss:+.2f}%")
    print(f"      Cascade-induced loss: {induced_loss:+.2f}%")

    # Analysis
    pct_induced = len(cascade_induced) / max(1, len(c75_clusters)) * 100
    print(f"\n  Cascade-induced fraction: {pct_induced:.1f}% of cluster events")

    if pct_induced > 80:
        print(f"  CONCLUSION: PARADOX CONFIRMED -- {pct_induced:.0f}% of cluster events are ")
        print(f"    cascade-INDUCED. The mechanism creates the clusters it claims to protect against.")
    elif pct_induced > 50:
        print(f"  CONCLUSION: PARTIAL PARADOX -- {pct_induced:.0f}% of cluster events are induced.")
        print(f"    The cascade creates more cluster events than it prevents.")
    else:
        print(f"  CONCLUSION: PARADOX NOT CONFIRMED -- most clusters are natural ({100-pct_induced:.0f}%).")

    # Net effect: does cascade reduce total cluster losses despite creating more events?
    print(f"\n  Net cluster loss comparison:")
    print(f"    Baseline total loss from clusters: {bl_cluster_loss:+.2f}%")
    print(f"    Cascade total loss from clusters:  {c75_cluster_loss:+.2f}%")
    if abs(c75_cluster_loss) < abs(bl_cluster_loss):
        print(f"    Cascade REDUCES cluster loss magnitude (despite more events)")
    else:
        print(f"    Cascade INCREASES cluster loss magnitude ({abs(c75_cluster_loss)-abs(bl_cluster_loss):+.2f}%)")

    return {
        'baseline_clusters': len(bl_clusters),
        'baseline_cluster_loss': round(bl_cluster_loss, 2),
        'cascade_total_clusters': len(c75_clusters),
        'cascade_natural_clusters': len(natural_clusters),
        'cascade_induced_clusters': len(cascade_induced),
        'cascade_cluster_loss': round(c75_cluster_loss, 2),
        'natural_cluster_loss': round(natural_loss, 2),
        'induced_cluster_loss': round(induced_loss, 2),
        'pct_induced': round(pct_induced, 1),
    }


# ============================================================
# TEST 5: Monotonicity Check
# ============================================================

def test5_monotonicity(signal_tuples, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, start_bar, end_bar):
    """Check if tighten_pct vs metrics shows a monotonic relationship."""
    print("\n[TEST 5: MONOTONICITY CHECK]")
    print("-" * 70)
    print("Question: Is the relationship between tighten_pct and PnL/MDD smooth?")
    print("  If non-monotonic (e.g., t50 < baseline < t75), this is suspicious.")

    tighten_pcts = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    results = []

    print(f"\n  {'Tighten%':>9} {'PnL%':>9} {'MDD%':>7} {'P/M':>7} {'WR%':>6} {'Trades':>6}")
    print("  " + "-" * 50)

    for t_pct in tighten_pcts:
        trades, eq_hist, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            cascade_tighten_pct=float(t_pct))
        stats = calc_stats(trades, eq_hist)
        results.append({'tighten_pct': t_pct, **stats})

        print(f"  {t_pct:>8}% {stats['pnl']:>+9.2f} {stats['mdd']:>7.2f} "
              f"{stats['pnl_mdd']:>7.2f} {stats['wr']:>6.1f} {stats['trades']:>6}")

    # Check monotonicity of PnL/MDD
    pnl_mdd_values = [r['pnl_mdd'] for r in results]
    pnl_values = [r['pnl'] for r in results]

    # Count direction changes in PnL/MDD
    direction_changes_pm = 0
    for i in range(1, len(pnl_mdd_values) - 1):
        if ((pnl_mdd_values[i] - pnl_mdd_values[i-1]) *
                (pnl_mdd_values[i+1] - pnl_mdd_values[i]) < 0):
            direction_changes_pm += 1

    direction_changes_pnl = 0
    for i in range(1, len(pnl_values) - 1):
        if ((pnl_values[i] - pnl_values[i-1]) *
                (pnl_values[i+1] - pnl_values[i]) < 0):
            direction_changes_pnl += 1

    # Check if baseline (0%) is the worst
    baseline_pm = pnl_mdd_values[0]
    max_pm = max(pnl_mdd_values)
    max_idx = pnl_mdd_values.index(max_pm)
    max_tighten = tighten_pcts[max_idx]

    print(f"\n  PnL/MDD direction changes: {direction_changes_pm} (in {len(tighten_pcts)} points)")
    print(f"  PnL direction changes: {direction_changes_pnl}")
    print(f"  Peak PnL/MDD: {max_pm:.2f} at tighten={max_tighten}%")
    print(f"  Baseline (0%): {baseline_pm:.2f}")

    is_monotonic_pm = direction_changes_pm <= 1
    is_monotonic_pnl = direction_changes_pnl <= 1

    if is_monotonic_pm:
        print(f"  CONCLUSION: PnL/MDD is MONOTONIC (or single-peaked) -- relationship is clean.")
    else:
        print(f"  CONCLUSION: PnL/MDD is NON-MONOTONIC ({direction_changes_pm} reversals) -- SUSPICIOUS.")
        print(f"  Erratic behavior suggests the mechanism's benefit is not robust.")

    return {
        'results': results,
        'direction_changes_pnl_mdd': direction_changes_pm,
        'direction_changes_pnl': direction_changes_pnl,
        'peak_pnl_mdd': round(max_pm, 2),
        'peak_tighten_pct': max_tighten,
        'baseline_pnl_mdd': round(baseline_pm, 2),
        'monotonic_pnl_mdd': is_monotonic_pm,
        'monotonic_pnl': is_monotonic_pnl,
    }


# ============================================================
# TEST 6: Compound vs Simple PnL
# ============================================================

def test6_compound_vs_simple(signal_tuples, opens, highs, lows, closes, n_bars,
                              atr_ratio, ema_slope, start_bar, end_bar):
    """Compare cascade advantage under compound vs simple (additive) sizing."""
    print("\n[TEST 6: COMPOUND VS SIMPLE PNL]")
    print("-" * 70)
    print("Question: Does the cascade advantage persist without compound amplification?")
    print("  Simple sizing: each trade PnL adds to running sum, no equity multiplication.")

    results = {}
    for mode_name, use_simple in [('compound', False), ('simple', True)]:
        bl_trades, bl_eq, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            cascade_tighten_pct=0.0, use_simple_sizing=use_simple)
        bl_stats = calc_stats(bl_trades, bl_eq)

        c75_trades, c75_eq, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            cascade_tighten_pct=75.0, use_simple_sizing=use_simple)
        c75_stats = calc_stats(c75_trades, c75_eq)

        results[mode_name] = {
            'baseline': bl_stats,
            'cascade': c75_stats,
            'delta_pnl': round(c75_stats['pnl'] - bl_stats['pnl'], 2),
            'delta_mdd': round(c75_stats['mdd'] - bl_stats['mdd'], 2),
        }

    print(f"\n  {'Mode':<10} {'BL PnL%':>9} {'C75 PnL%':>9} {'Delta PnL':>10} "
          f"{'BL MDD%':>8} {'C75 MDD%':>8} {'Delta MDD':>10}")
    print("  " + "-" * 70)

    for mode_name in ['compound', 'simple']:
        r = results[mode_name]
        print(f"  {mode_name:<10} {r['baseline']['pnl']:>+9.2f} {r['cascade']['pnl']:>+9.2f} "
              f"{r['delta_pnl']:>+10.2f} {r['baseline']['mdd']:>8.2f} "
              f"{r['cascade']['mdd']:>8.2f} {r['delta_mdd']:>+10.2f}")

    compound_delta = results['compound']['delta_pnl']
    simple_delta = results['simple']['delta_pnl']

    print(f"\n  Compound advantage delta: {compound_delta:+.2f}%")
    print(f"  Simple advantage delta:   {simple_delta:+.2f}%")

    if simple_delta > 0 and compound_delta > 0:
        amplification = compound_delta / simple_delta if simple_delta != 0 else float('inf')
        print(f"  Amplification factor: {amplification:.2f}x")
        print(f"  CONCLUSION: Cascade advantage persists in SIMPLE mode ({simple_delta:+.1f}%).")
        if amplification > 5:
            print(f"  WARNING: Compound amplifies the advantage {amplification:.0f}x.")
            print(f"  The headline number is mostly compound effect, not raw edge.")
        genuine = True
    elif simple_delta <= 0 and compound_delta > 0:
        print(f"  CONCLUSION: Cascade advantage ONLY exists in compound mode.")
        print(f"  In simple mode, cascade is WORSE ({simple_delta:+.1f}%).")
        print(f"  This means compound amplification of sequential wins creates the illusion.")
        genuine = False
    elif simple_delta > 0 and compound_delta <= 0:
        print(f"  CONCLUSION: Unusual -- simple mode shows advantage but compound does not.")
        genuine = False
    else:
        print(f"  CONCLUSION: Cascade is worse in BOTH modes.")
        genuine = False

    return {
        'compound': results['compound'],
        'simple': results['simple'],
        'compound_delta': compound_delta,
        'simple_delta': simple_delta,
        'genuine_in_simple': genuine,
    }


# ============================================================
# TEST 7: Direction Cap Sensitivity
# ============================================================

def test7_direction_cap_sensitivity(signal_tuples, opens, highs, lows, closes, n_bars,
                                     atr_ratio, ema_slope, start_bar, end_bar):
    """Test cascade_t75 at different direction caps."""
    print("\n[TEST 7: DIRECTION CAP SENSITIVITY]")
    print("-" * 70)
    print("Question: Does cascade only help with high direction caps (more same-dir positions)?")
    print("  If so, cascade is a workaround for excessive exposure, not a genuine improvement.")

    caps = [3, 5, 7, 9]
    results = []

    print(f"\n  {'DirCap':>6} {'BL PnL%':>9} {'BL MDD%':>8} {'BL P/M':>7} "
          f"{'C75 PnL%':>9} {'C75 MDD%':>8} {'C75 P/M':>7} {'Delta P/M':>9}")
    print("  " + "-" * 75)

    for cap in caps:
        bl_trades, bl_eq, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            cascade_tighten_pct=0.0, direction_cap_override=cap)
        bl_stats = calc_stats(bl_trades, bl_eq)

        c75_trades, c75_eq, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            cascade_tighten_pct=75.0, direction_cap_override=cap)
        c75_stats = calc_stats(c75_trades, c75_eq)

        delta_pm = c75_stats['pnl_mdd'] - bl_stats['pnl_mdd']
        results.append({
            'direction_cap': cap,
            'baseline': bl_stats,
            'cascade': c75_stats,
            'delta_pnl_mdd': round(delta_pm, 2),
        })

        print(f"  {cap:>6} {bl_stats['pnl']:>+9.2f} {bl_stats['mdd']:>8.2f} "
              f"{bl_stats['pnl_mdd']:>7.2f} {c75_stats['pnl']:>+9.2f} "
              f"{c75_stats['mdd']:>8.2f} {c75_stats['pnl_mdd']:>7.2f} {delta_pm:>+9.2f}")

    # Analysis: does cascade benefit scale with cap?
    deltas = [r['delta_pnl_mdd'] for r in results]
    positive_deltas = sum(1 for d in deltas if d > 0)

    # Check if cascade benefit increases with cap
    increasing = all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))

    print(f"\n  Cascade PnL/MDD improvement by cap: [{', '.join(f'{d:+.2f}' for d in deltas)}]")
    print(f"  Positive in {positive_deltas}/{len(caps)} caps")

    if positive_deltas == len(caps):
        print(f"  CONCLUSION: Cascade helps at ALL direction caps.")
        print(f"  -> NOT a workaround for excessive exposure. Genuine mechanism.")
        is_workaround = False
    elif deltas[-1] > 0 and deltas[0] <= 0:
        print(f"  CONCLUSION: Cascade ONLY helps at high caps (cap={caps[-1]}).")
        print(f"  -> IS a workaround for excessive same-direction exposure.")
        is_workaround = True
    else:
        low_cap_delta = deltas[0]
        high_cap_delta = deltas[-1]
        print(f"  Low cap (3) delta: {low_cap_delta:+.2f}")
        print(f"  High cap (9) delta: {high_cap_delta:+.2f}")
        if abs(high_cap_delta) > 3 * abs(low_cap_delta) and high_cap_delta > 0:
            print(f"  CONCLUSION: Cascade benefit disproportionately at high caps -- partial workaround.")
            is_workaround = True
        else:
            print(f"  CONCLUSION: Mixed results, no clear dependency on direction cap.")
            is_workaround = False

    return {
        'results': results,
        'positive_deltas': positive_deltas,
        'is_workaround': is_workaround,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("CRITICAL ANALYSIS: CASCADE SL TIGHTENING")
    print("=" * 78)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Baseline: Production v1.42.0 (M2/M4 disabled, fixed leverage=3)")
    print(f"Subject: P3 Cascade SL Tightening (75%) critical stress-test")

    # ---- Load Data ----
    print("\n[SETUP] Loading data...")
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    print("[SETUP] Computing indicators...")
    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)

    neutral_start, neutral_end = find_neutral_window(closes)
    n_neutral = neutral_end - neutral_start
    print(f"  Neutral window: bars {neutral_start}-{neutral_end} ({n_neutral} bars, "
          f"{n_neutral / BARS_PER_DAY:.0f}d)")

    # ---- Load Patterns ----
    print("[SETUP] Loading patterns...")
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
    print("[SETUP] Building signal index...")
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
    # Run All Tests
    # ================================================================
    all_results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'study': 'cascade_critical_analysis',
            'baseline': 'v1.42.0 (M2/M4 disabled, fixed leverage=3)',
            'data_bars': int(n_bars),
            'neutral_window': [int(neutral_start), int(neutral_end)],
            'neutral_days': round(n_neutral / BARS_PER_DAY, 1),
            'patterns': f"{len(pat_lookup)} ({n_long}L + {n_short}S)",
            'signals_in_neutral': n_in_neutral,
            'leverage': LEVERAGE,
        }
    }

    # TEST 1
    print("\n" + "=" * 78)
    all_results['test1_decomposition'] = test1_pnl_decomposition(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end)

    # TEST 2
    print("\n" + "=" * 78)
    all_results['test2_bootstrap_ci'] = test2_bootstrap_ci(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end)

    # TEST 3
    print("\n" + "=" * 78)
    all_results['test3_temporal'] = test3_temporal_consistency(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end)

    # TEST 4
    print("\n" + "=" * 78)
    all_results['test4_cluster_paradox'] = test4_cluster_paradox(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end)

    # TEST 5
    print("\n" + "=" * 78)
    all_results['test5_monotonicity'] = test5_monotonicity(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end)

    # TEST 6
    print("\n" + "=" * 78)
    all_results['test6_compound_vs_simple'] = test6_compound_vs_simple(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end)

    # TEST 7
    print("\n" + "=" * 78)
    all_results['test7_direction_cap'] = test7_direction_cap_sensitivity(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end)

    # ================================================================
    # OVERALL VERDICT
    # ================================================================
    print("\n" + "=" * 78)
    print("OVERALL CRITICAL VERDICT")
    print("=" * 78)

    concerns = []
    positive = []

    # Test 1: Decomposition
    t1 = all_results['test1_decomposition']
    if t1['forced_pnl'] < -10:
        concerns.append(f"Test 1: Cascade forced exits contribute {t1['forced_pnl']:+.1f}% "
                         f"(significant drag)")
    if t1['post_cascade_trades'] < 5:
        concerns.append(f"Test 1: Only {t1['post_cascade_trades']} post-cascade entries "
                         f"(slot recycling minimal)")
    if t1['cascade_pnl'] > t1['baseline_pnl']:
        positive.append(f"Test 1: Cascade PnL {t1['cascade_pnl']:+.1f}% > baseline {t1['baseline_pnl']:+.1f}%")

    # Test 2: Bootstrap
    t2 = all_results['test2_bootstrap_ci']
    if t2['ci_includes_zero']:
        concerns.append(f"Test 2: Bootstrap CI includes zero [{t2['difference_ci_95'][0]:+.1f}%, "
                         f"{t2['difference_ci_95'][1]:+.1f}%] -- NOT significant")
    else:
        if t2['difference_ci_95'][0] > 0:
            positive.append(f"Test 2: Statistically significant improvement "
                             f"(CI [{t2['difference_ci_95'][0]:+.1f}%, {t2['difference_ci_95'][1]:+.1f}%])")

    # Test 3: Temporal
    t3 = all_results['test3_temporal']
    if t3['cascade_wins'] < 3:
        concerns.append(f"Test 3: Cascade only wins {t3['cascade_wins']}/4 quarters "
                         f"-- temporally inconsistent")
    elif t3['cascade_wins'] == 4:
        positive.append(f"Test 3: Cascade wins all 4 quarters -- temporally consistent")
    else:
        positive.append(f"Test 3: Cascade wins {t3['cascade_wins']}/4 quarters -- mostly consistent")

    # Test 4: Cluster paradox
    t4 = all_results['test4_cluster_paradox']
    if t4['pct_induced'] > 50:
        concerns.append(f"Test 4: {t4['pct_induced']:.0f}% of cluster events are cascade-INDUCED "
                         f"(self-fulfilling mechanism)")

    # Test 5: Monotonicity
    t5 = all_results['test5_monotonicity']
    if not t5['monotonic_pnl_mdd']:
        concerns.append(f"Test 5: PnL/MDD is non-monotonic ({t5['direction_changes_pnl_mdd']} reversals) "
                         f"-- erratic relationship")
    else:
        positive.append(f"Test 5: PnL/MDD is monotonic -- clean relationship")

    # Test 6: Compound vs Simple
    t6 = all_results['test6_compound_vs_simple']
    if not t6['genuine_in_simple']:
        concerns.append(f"Test 6: Cascade advantage disappears in simple mode "
                         f"(simple delta: {t6['simple_delta']:+.1f}%)")
    else:
        positive.append(f"Test 6: Cascade advantage persists in simple mode "
                         f"({t6['simple_delta']:+.1f}%)")

    # Test 7: Direction cap
    t7 = all_results['test7_direction_cap']
    if t7['is_workaround']:
        concerns.append(f"Test 7: Cascade benefit depends on direction cap "
                         f"-- workaround for excessive exposure")
    else:
        if t7['positive_deltas'] == len(t7['results']):
            positive.append(f"Test 7: Cascade helps at ALL direction caps -- genuine mechanism")
        else:
            positive.append(f"Test 7: Cascade helps at {t7['positive_deltas']}/{len(t7['results'])} caps")

    # Print verdict
    print(f"\n  POSITIVE FINDINGS:")
    for i, p in enumerate(positive):
        print(f"    {i+1}. {p}")
    if not positive:
        print(f"    (none)")

    print(f"\n  CONCERNS:")
    for i, c in enumerate(concerns):
        print(f"    {i+1}. {c}")
    if not concerns:
        print(f"    (none)")

    # Final assessment
    n_concerns = len(concerns)
    n_positive = len(positive)

    print(f"\n  Score: {n_positive} positive / {n_concerns} concerns")

    if n_concerns == 0:
        verdict = "GENUINE"
        explanation = "All tests pass. Cascade SL tightening appears to be a genuine improvement."
    elif n_concerns <= 2 and n_positive >= 4:
        verdict = "GENUINE (with caveats)"
        explanation = ("Most tests pass, minor concerns exist. "
                       "Cascade appears genuine but some aspects warrant monitoring.")
    elif n_concerns <= 3 and n_positive >= 3:
        verdict = "SUSPICIOUS"
        explanation = ("Mixed results. The cascade advantage may be partially real but "
                       "amplified by compound effects or specific market conditions.")
    elif n_concerns >= 4:
        verdict = "OVERFITTED"
        explanation = ("Multiple critical tests fail. The cascade advantage is likely "
                       "an artifact of overfitting, compound amplification, or self-fulfilling mechanics.")
    else:
        verdict = "SUSPICIOUS"
        explanation = "Insufficient evidence to confirm genuine improvement."

    print(f"\n  FINAL ASSESSMENT: {verdict}")
    print(f"  {explanation}")

    all_results['verdict'] = {
        'assessment': verdict,
        'explanation': explanation,
        'n_positive': n_positive,
        'n_concerns': n_concerns,
        'positive': positive,
        'concerns': concerns,
    }

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
