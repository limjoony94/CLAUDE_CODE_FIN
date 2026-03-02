#!/usr/bin/env python3
"""
MDD Reduction Techniques Study
===============================

Goal: Find NEW MDD reduction techniques that reduce MDD more than they reduce PnL
(i.e., improve PnL/MDD ratio). Key criterion from user:
  "pnl/mdd에서 mdd 감소 폭이 더 많다면 반영해볼 만하다"

Hypotheses:
  H1: Equity Curve Trading — pause/reduce when equity < EMA(equity)
  H2: Dynamic N — reduce max_positions during drawdown
  H3: Volatility Pause — skip entries during extreme ATR spikes
  H4: Intraday Loss Limit — tighter daily loss cap
  H5: Trailing Equity Stop — pause when equity drops X% from N-trade peak
  H6: Correlation-Aware Entry — skip when direction concentration + counter-regime
  H7: Tighter MDD Sizing Parameters — faster/harder DD reaction
  H8: Combined Best — top 2-3 GO techniques combined

Phases:
  1. IS comparison (all hypotheses, neutral window)
  2. Top candidates selection (MDD reduction% > PnL reduction%)
  3. WF 3-fold expanding window
  4. Extended risk metrics for WF-PASS candidates
  5. Combination test (top 2-3 individual techniques)
  6. Verdict

Baseline includes ALL production features:
  N=9, compound, direction_cap=7, regime*0.3, agg_risk_cap(3/7%),
  momentum_guard, timeout(864 DROP), ATR-scaled, adaptive leverage
  (wr_confidence w=12), loss_burst_brake(2 losses/24h -> 12h block)

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE adaptive (1-3x, wr_confidence w=12)
  - FEE = FEE_PCT * leverage (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - WF: 3-fold expanding window

Author: Research Agent
Date: 2026-03-02
"""

import math
import os
import sys
import json
import warnings
from collections import deque
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
# Constants — matching production exactly
# ============================================================
MAX_LEVERAGE = 3
MIN_LEVERAGE = 1
FEE_PCT = 0.10
SLIPPAGE_BUFFER = 0.02
TIMEOUT_BARS = 864
N_SLOTS = 9
DIRECTION_CAP = 7
REGIME_MULT = 0.3
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

# Loss burst brake params (v1.37.0 production)
LBB_THRESHOLD = 2
LBB_WINDOW_BARS = 288
LBB_BLOCK_BARS = 144

# MDD sizing params (production v1.35.2)
MDD_FULL_BELOW = 3.0
MDD_MIN_ABOVE = 15.0
MDD_MIN_SCALE = 0.25

# Adaptive leverage (production v1.39.0)
ADAPTIVE_LEV_WINDOW = 12
EXPECTED_WR = 0.732
REF_EDGE = 0.00126

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'mdd_reduction_study.json')


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
# Adaptive Leverage (production: wr_confidence, w=12)
# ============================================================

def get_leverage_wr_confidence(rolling_wr_frac, rolling_edge_frac):
    """WR Confidence method — production v1.39.0."""
    if EXPECTED_WR <= 0:
        return MIN_LEVERAGE
    wr_conf = clamp(rolling_wr_frac / EXPECTED_WR, 0, 1)
    edge_q = clamp(rolling_edge_frac / REF_EDGE, 0, 1.5) if REF_EDGE > 0 else 0.5
    combined = clamp(wr_conf * edge_q, 0, 1)
    return MIN_LEVERAGE + (MAX_LEVERAGE - MIN_LEVERAGE) * combined


# ============================================================
# Portfolio Simulator with all production features + MDD hooks
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio):
    """Check exit for a single position."""
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

    entry_lev = pos['leverage']
    fee = FEE_PCT * entry_lev
    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * entry_lev
    else:
        pnl = (1 - exit_price / entry) * 100 * entry_lev
    pnl -= fee

    return {'pnl_slot': pnl, 'reason': reason, 'drop': False,
            'entry_bar': entry_bar, 'exit_bar': bar, 'leverage': entry_lev}


def _get_mdd_size_scale(equity, peak_equity, full_below, min_above, min_scale):
    """MDD-based position sizing scale factor."""
    if peak_equity <= 0:
        return 1.0
    dd_pct = (peak_equity - equity) / peak_equity * 100
    if dd_pct <= full_below:
        return 1.0
    if dd_pct >= min_above:
        return min_scale
    return 1.0 - (1.0 - min_scale) * (dd_pct - full_below) / (min_above - full_below)


def portfolio_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                  atr_ratio, ema_slope, start_bar, end_bar,
                  # MDD reduction technique parameters (all default=off for baseline)
                  # H1: Equity Curve Trading
                  h1_enabled=False, h1_ema_period=20, h1_action='skip',
                  h1_size_mult=0.5,
                  # H2: Dynamic N
                  h2_enabled=False, h2_dd_threshold=5.0, h2_reduced_n=7,
                  # H3: Volatility Pause
                  h3_enabled=False, h3_atr_threshold=2.0, h3_pause_bars=24,
                  # H4: Intraday Loss Limit
                  h4_enabled=False, h4_daily_limit=13.0,
                  # H5: Trailing Equity Stop
                  h5_enabled=False, h5_drop_pct=5.0, h5_lookback_trades=30,
                  h5_pause_bars=288,
                  # H6: Correlation-Aware Entry
                  h6_enabled=False, h6_dir_pct_threshold=0.8,
                  # H7: Tighter MDD Sizing
                  h7_full_below=MDD_FULL_BELOW, h7_min_above=MDD_MIN_ABOVE,
                  h7_min_scale=MDD_MIN_SCALE):
    """N-pos portfolio with all production features + optional MDD reduction hooks.

    Production features always ON:
      - N=9 compound, direction_cap=7, regime*0.3, agg_risk_cap(3/7%),
        momentum_guard, timeout(864 DROP), ATR-scaled TP/SL,
        adaptive leverage (wr_confidence w=12), loss_burst_brake,
        MDD sizing (with h7_ params)
    """
    size_pct = 100.0 / N_SLOTS
    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0

    # Rolling WR trackers for adaptive leverage
    recent_results = deque(maxlen=ADAPTIVE_LEV_WINDOW)
    recent_pnls = deque(maxlen=ADAPTIVE_LEV_WINDOW)

    # Loss burst brake state: per-direction loss timestamps
    lbb_losses = {'LONG': [], 'SHORT': []}
    lbb_blocked_until = {'LONG': -1, 'SHORT': -1}

    # Momentum guard state
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # H1: Equity curve EMA
    equity_history = []

    # H4: Intraday tracking
    day_start_equity = equity
    day_start_bar = start_bar

    # H5: Trailing equity stop from rolling peak
    trade_equities = deque(maxlen=max(h5_lookback_trades, 1))
    h5_pause_until = -1

    # H3: Volatility pause state
    h3_pause_until = -1

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # --- Exits ---
        closed_slots = []
        bar_pnl_sum = 0.0

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
                result['leverage'] = pos['leverage']
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio

                is_win = result['pnl_slot'] > 0
                recent_results.append(1 if is_win else 0)
                recent_pnls.append(result['pnl_slot'])

                # Loss burst brake: record losses
                if not is_win:
                    d = pos['direction']
                    lbb_losses[d].append(bar)
                    # Prune old losses outside window
                    lbb_losses[d] = [b for b in lbb_losses[d]
                                     if bar - b <= LBB_WINDOW_BARS]
                    if len(lbb_losses[d]) >= LBB_THRESHOLD:
                        lbb_blocked_until[d] = bar + LBB_BLOCK_BARS
                        lbb_losses[d] = []

        positions = [p for p in positions if p['slot'] not in closed_slots]

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        equity_history.append(equity)

        # H5: Record equity after trades close
        if bar_pnl_sum != 0:
            trade_equities.append(equity)

        # H4: Intraday loss limit check (reset each day)
        if (bar - day_start_bar) >= BARS_PER_DAY:
            day_start_equity = equity
            day_start_bar = bar

        # Momentum guard
        if MOMENTUM_LOOKBACK > 0 and bar >= MOMENTUM_LOOKBACK:
            price_now, price_ago = closes[bar], closes[bar - MOMENTUM_LOOKBACK]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > MOMENTUM_THRESHOLD:
                    momentum_pause_until['SHORT'] = bar + MOMENTUM_COOLDOWN
                elif pct_change < -MOMENTUM_THRESHOLD:
                    momentum_pause_until['LONG'] = bar + MOMENTUM_COOLDOWN

        # H3: Volatility pause check
        if h3_enabled and bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
            if atr_ratio[bar] > h3_atr_threshold:
                h3_pause_until = bar + h3_pause_bars

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

            # Momentum guard
            if bar < momentum_pause_until.get(direction, -1):
                continue

            # Loss burst brake
            if bar < lbb_blocked_until.get(direction, -1):
                continue

            # H1: Equity curve trading
            h1_mult = 1.0
            if h1_enabled and len(equity_history) >= h1_ema_period:
                eq_ema = np.mean(equity_history[-h1_ema_period:])
                if equity < eq_ema:
                    if h1_action == 'skip':
                        continue
                    else:  # 'reduce'
                        h1_mult = h1_size_mult

            # H2: Dynamic N during drawdown
            if h2_enabled:
                if dd >= h2_dd_threshold and len(positions) >= h2_reduced_n:
                    continue

            # H3: Volatility pause
            if h3_enabled and bar < h3_pause_until:
                continue

            # H4: Intraday loss limit
            if h4_enabled and day_start_equity > 0:
                intraday_loss = (day_start_equity - equity) / day_start_equity * 100
                if intraday_loss >= h4_daily_limit:
                    continue

            # H5: Trailing equity stop
            if h5_enabled and bar < h5_pause_until:
                continue
            if h5_enabled and len(trade_equities) >= 2:
                rolling_peak = max(trade_equities)
                if rolling_peak > 0:
                    rolling_drop = (rolling_peak - equity) / rolling_peak * 100
                    if rolling_drop >= h5_drop_pct:
                        h5_pause_until = bar + h5_pause_bars
                        continue

            # H6: Correlation-aware entry
            if h6_enabled and len(positions) >= 2:
                same_dir = sum(1 for p in positions if p['direction'] == direction)
                dir_ratio = same_dir / len(positions)
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                if dir_ratio >= h6_dir_pct_threshold and is_counter:
                    continue

            # Regime sizing
            sm = 1.0
            if REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if (s > 0 and direction == 'SHORT') or (s <= 0 and direction == 'LONG'):
                    sm = REGIME_MULT

            # MDD sizing (h7 params)
            mdd_scale = _get_mdd_size_scale(equity, peak_equity,
                                            h7_full_below, h7_min_above, h7_min_scale)
            sm *= mdd_scale

            # H1 size reduction
            sm *= h1_mult

            # Adaptive leverage (wr_confidence)
            rolling_wr_frac = (sum(recent_results) / len(recent_results)
                               if recent_results else 0.5)
            rolling_edge_frac = (np.mean(list(recent_pnls)) / 100.0
                                 if recent_pnls else 0.0)
            entry_leverage = get_leverage_wr_confidence(rolling_wr_frac, rolling_edge_frac)
            entry_leverage = clamp(entry_leverage, MIN_LEVERAGE, MAX_LEVERAGE)

            # Aggregate risk cap
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
                    existing_exposure += (p['sl_pct'] * p_r * (1.0 / N_SLOTS)
                                          * p['leverage'] * p.get('size_mult', 1.0))

            new_r = 1.0
            if (atr_ratio is not None and sig_bar < len(atr_ratio)
                    and not np.isnan(atr_ratio[sig_bar])):
                new_r = clamp(atr_ratio[sig_bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
            new_exposure = sl_pct * new_r * (1.0 / N_SLOTS) * entry_leverage * sm
            if existing_exposure + new_exposure > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}", 'signal_bar': sig_bar,
                'entry_bar': entry_bar, 'direction': direction,
                'pattern': pat, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'size_mult': sm, 'leverage': entry_leverage,
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
        entry_lev = pos['leverage']
        fee = FEE_PCT * entry_lev
        if pos['direction'] == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * entry_lev
        else:
            pnl = (1 - exit_price / entry) * 100 * entry_lev
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': pos['entry_bar'], 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
            'leverage': entry_lev,
        })

    return trades, equity_history


def calc_stats(trades, equity_history):
    """Calculate standard stats from trades and equity curve."""
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
    avg_lev = np.mean([t.get('leverage', MAX_LEVERAGE) for t in trades])

    # Extended risk metrics
    slot_pnls = [t['pnl_slot'] for t in sorted_t]
    max_consec_loss = 0
    curr_streak = 0
    for p in slot_pnls:
        if p <= 0:
            curr_streak += 1
            if curr_streak > max_consec_loss:
                max_consec_loss = curr_streak
        else:
            curr_streak = 0

    # Worst daily PnL from equity history
    worst_daily = 0.0
    if len(equity_history) >= BARS_PER_DAY:
        for i in range(BARS_PER_DAY, len(equity_history)):
            prev_eq = equity_history[i - BARS_PER_DAY]
            if prev_eq > 0:
                daily_chg = (equity_history[i] - prev_eq) / prev_eq * 100
                if daily_chg < worst_daily:
                    worst_daily = daily_chg

    # Max DD duration (bars)
    in_dd = False
    dd_start = 0
    max_dd_dur = 0
    eq2 = 100.0
    pk2 = 100.0
    for t in sorted_t:
        eq2 += t['pnl_portfolio']
        if eq2 > pk2:
            pk2 = eq2
        dd2 = (pk2 - eq2) / pk2 * 100 if pk2 > 0 else 0
        if dd2 > 0.1:
            if not in_dd:
                dd_start = t['entry_bar']
                in_dd = True
        else:
            if in_dd:
                dur = t['exit_bar'] - dd_start
                if dur > max_dd_dur:
                    max_dd_dur = dur
                in_dd = False

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(total_pnl / mdd, 2) if mdd > 0 else 0,
        'avg_leverage': round(avg_lev, 2),
        'max_consec_loss': max_consec_loss,
        'worst_daily': round(worst_daily, 2),
        'max_dd_duration_bars': max_dd_dur,
    }


# ============================================================
# WF 3-fold expanding window
# ============================================================

def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, neutral_start, neutral_end,
           scenario_kwargs, n_folds=3):
    """3-fold expanding window WF."""
    total = neutral_end - neutral_start
    seg_size = total // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        oos_start = neutral_start + seg_size * (fold + 1)
        oos_end = (neutral_start + seg_size * (fold + 2)
                   if fold < n_folds - 1 else neutral_end)

        trades, eq_hist = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            **scenario_kwargs)
        stats = calc_stats(trades, eq_hist)
        stats['fold'] = fold + 1
        stats['oos_bars'] = int(oos_end - oos_start)
        results.append(stats)

    return results


# ============================================================
# Scenario Definitions
# ============================================================

def build_scenarios():
    """Build all hypothesis scenarios.

    Returns list of (name, kwargs_dict) tuples.
    """
    scenarios = []

    # Baseline (all production features, no new techniques)
    scenarios.append(('Baseline (production)', {}))

    # H1: Equity Curve Trading
    for ema_p in [10, 20, 30]:
        scenarios.append((f'H1_EqCurve_skip_ema{ema_p}',
                          {'h1_enabled': True, 'h1_ema_period': ema_p, 'h1_action': 'skip'}))
    for ema_p in [10, 20, 30]:
        scenarios.append((f'H1_EqCurve_half_ema{ema_p}',
                          {'h1_enabled': True, 'h1_ema_period': ema_p,
                           'h1_action': 'reduce', 'h1_size_mult': 0.5}))

    # H2: Dynamic N during drawdown
    for dd_thresh in [2, 3, 5]:
        for red_n in [5, 6, 7]:
            scenarios.append((f'H2_DynN_dd{dd_thresh}_n{red_n}',
                              {'h2_enabled': True, 'h2_dd_threshold': float(dd_thresh),
                               'h2_reduced_n': red_n}))

    # H3: Volatility Pause
    for atr_thresh in [1.5, 2.0, 2.5]:
        for pause_b in [12, 24, 48]:
            scenarios.append((f'H3_VolPause_atr{atr_thresh}_p{pause_b}',
                              {'h3_enabled': True, 'h3_atr_threshold': atr_thresh,
                               'h3_pause_bars': pause_b}))

    # H4: Intraday Loss Limit
    for daily_lim in [5, 7, 9, 11]:
        scenarios.append((f'H4_DailyLim_{daily_lim}pct',
                          {'h4_enabled': True, 'h4_daily_limit': float(daily_lim)}))

    # H5: Trailing Equity Stop (representative combos, not full 3x3x3)
    h5_combos = [
        (3, 20, 144), (3, 30, 288), (3, 50, 576),
        (5, 20, 288), (5, 30, 288), (5, 50, 576),
        (8, 20, 288), (8, 30, 576), (8, 50, 576),
    ]
    for drop_pct, lb_trades, pause_b in h5_combos:
        scenarios.append(
            (f'H5_TrailStop_d{drop_pct}_t{lb_trades}_p{pause_b}',
             {'h5_enabled': True, 'h5_drop_pct': float(drop_pct),
              'h5_lookback_trades': lb_trades, 'h5_pause_bars': pause_b}))

    # H6: Correlation-Aware Entry
    for dir_thresh in [0.7, 0.8, 0.9]:
        scenarios.append((f'H6_CorrAware_dir{int(dir_thresh*100)}',
                          {'h6_enabled': True, 'h6_dir_pct_threshold': dir_thresh}))

    # H7: Tighter MDD Sizing
    scenarios.append(('H7_MDD_tight_1.5_10_0.10',
                      {'h7_full_below': 1.5, 'h7_min_above': 10.0, 'h7_min_scale': 0.10}))
    scenarios.append(('H7_MDD_tight_2.0_8_0.15',
                      {'h7_full_below': 2.0, 'h7_min_above': 8.0, 'h7_min_scale': 0.15}))
    scenarios.append(('H7_MDD_tight_2.0_12_0.15',
                      {'h7_full_below': 2.0, 'h7_min_above': 12.0, 'h7_min_scale': 0.15}))
    scenarios.append(('H7_MDD_tight_1.0_8_0.10',
                      {'h7_full_below': 1.0, 'h7_min_above': 8.0, 'h7_min_scale': 0.10}))

    return scenarios


# ============================================================
# Main Study
# ============================================================

def main():
    print("=" * 78)
    print("MDD REDUCTION TECHNIQUES STUDY")
    print("=" * 78)
    start_time = datetime.now()

    # ---- Load Data ----
    print("\n[1] Loading data...")
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)

    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)

    neutral_start, neutral_end = find_neutral_window(closes, tol_pct=1.0)
    neutral_bars = neutral_end - neutral_start
    print(f"  Neutral window: bar {neutral_start}~{neutral_end} "
          f"({neutral_bars} bars, {neutral_bars / BARS_PER_DAY:.0f} days)")

    # ---- Load Patterns ----
    print("\n[2] Loading patterns...")
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
    print("\n[3] Building signal index...")
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
    # PHASE 1: IS Comparison (all hypotheses)
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 1: IS Full-Period Comparison (Neutral Window)")
    print("=" * 78)

    scenarios = build_scenarios()
    print(f"  Running {len(scenarios)} scenarios...")

    phase1_results = {}
    print(f"\n  {'Scenario':<40} {'Trades':>6} {'WR%':>6} {'PnL%':>9} "
          f"{'MDD%':>7} {'P/M':>7} {'ConsL':>5} {'WrstD':>7}")
    print("  " + "-" * 89)

    for name, kwargs in scenarios:
        trades, eq_hist = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            **kwargs)
        stats = calc_stats(trades, eq_hist)
        phase1_results[name] = stats
        print(f"  {name:<40} {stats['trades']:>6} {stats['wr']:>6.1f} "
              f"{stats['pnl']:>9.1f} {stats['mdd']:>7.1f} "
              f"{stats['pnl_mdd']:>7.2f} {stats['max_consec_loss']:>5} "
              f"{stats['worst_daily']:>7.1f}")

    baseline = phase1_results['Baseline (production)']

    # ================================================================
    # PHASE 2: Top Candidates (MDD reduction% > PnL reduction%)
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 2: Candidate Selection (MDD_reduction% > PnL_reduction%)")
    print("=" * 78)

    candidates = []
    for name, stats in phase1_results.items():
        if name == 'Baseline (production)':
            continue
        if baseline['pnl'] <= 0 or baseline['mdd'] <= 0:
            continue
        pnl_reduction = (baseline['pnl'] - stats['pnl']) / abs(baseline['pnl']) * 100
        mdd_reduction = (baseline['mdd'] - stats['mdd']) / baseline['mdd'] * 100
        pnl_mdd_change = ((stats['pnl_mdd'] - baseline['pnl_mdd'])
                          / baseline['pnl_mdd'] * 100 if baseline['pnl_mdd'] > 0 else 0)

        candidates.append({
            'name': name,
            'stats': stats,
            'pnl_reduction_pct': round(pnl_reduction, 2),
            'mdd_reduction_pct': round(mdd_reduction, 2),
            'pnl_mdd_change_pct': round(pnl_mdd_change, 2),
            'passes_criterion': mdd_reduction > pnl_reduction and mdd_reduction > 0,
        })

    # Filter: MDD reduction > PnL reduction AND MDD actually decreased
    passing = [c for c in candidates if c['passes_criterion']]
    passing.sort(key=lambda c: c['stats']['pnl_mdd'], reverse=True)

    print(f"\n  Baseline: PnL {baseline['pnl']:.1f}%, MDD {baseline['mdd']:.1f}%, "
          f"PnL/MDD {baseline['pnl_mdd']:.2f}")
    print(f"\n  {len(passing)} scenarios pass criterion (MDD_red% > PnL_red%):")
    print(f"  {'Scenario':<40} {'PnL%':>8} {'MDD%':>7} {'P/M':>7} "
          f"{'PnL_red':>8} {'MDD_red':>8} {'P/M_chg':>8}")
    print("  " + "-" * 89)

    for c in passing[:20]:
        s = c['stats']
        print(f"  {c['name']:<40} {s['pnl']:>8.1f} {s['mdd']:>7.1f} "
              f"{s['pnl_mdd']:>7.2f} {c['pnl_reduction_pct']:>+7.1f}% "
              f"{c['mdd_reduction_pct']:>+7.1f}% {c['pnl_mdd_change_pct']:>+7.1f}%")

    # Select top 6 by PnL/MDD for WF
    top_wf = passing[:6]
    if not top_wf:
        print("\n  WARNING: No scenarios pass criterion. Selecting top 6 by PnL/MDD anyway.")
        candidates.sort(key=lambda c: c['stats']['pnl_mdd'], reverse=True)
        top_wf = candidates[:6]

    # ================================================================
    # PHASE 3: WF 3-fold Expanding Window
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 3: Walk-Forward 3-Fold OOS Validation")
    print("=" * 78)

    # Build kwargs map from scenarios
    scenario_kwargs_map = {name: kwargs for name, kwargs in scenarios}

    wf_results = {}

    # Always include baseline
    baseline_folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, neutral_start, neutral_end,
                            scenario_kwargs_map.get('Baseline (production)', {}))
    baseline_pass = all(f['pnl'] > 0 for f in baseline_folds)
    wf_results['Baseline (production)'] = {
        'folds': baseline_folds, 'pass': baseline_pass}
    fold_str = ' | '.join([f"F{f['fold']}: {f['pnl']:+.1f}%" for f in baseline_folds])
    status = "PASS" if baseline_pass else "FAIL"
    print(f"  [{status}] {'Baseline (production)':<40} {fold_str}")

    wf_pass_names = []
    for c in top_wf:
        name = c['name']
        kwargs = scenario_kwargs_map.get(name, {})
        folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, neutral_start, neutral_end, kwargs)
        all_pass = all(f['pnl'] > 0 for f in folds)
        wf_results[name] = {'folds': folds, 'pass': all_pass}
        fold_str = ' | '.join([f"F{f['fold']}: {f['pnl']:+.1f}%" for f in folds])
        status = "PASS" if all_pass else "FAIL"
        print(f"  [{status}] {name:<40} {fold_str}")
        if all_pass:
            wf_pass_names.append(name)

    # ================================================================
    # PHASE 4: Risk Metrics for WF-PASS Candidates
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 4: Extended Risk Metrics (WF-PASS Candidates)")
    print("=" * 78)

    risk_results = {}
    risk_names = ['Baseline (production)'] + wf_pass_names

    print(f"\n  {'Scenario':<40} {'PnL%':>8} {'MDD%':>7} {'P/M':>7} "
          f"{'ConsL':>5} {'WrstD%':>7} {'DDdur':>6} {'AvgLev':>7}")
    print("  " + "-" * 89)

    for name in risk_names:
        kwargs = scenario_kwargs_map.get(name, {})
        trades, eq_hist = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            **kwargs)
        stats = calc_stats(trades, eq_hist)
        risk_results[name] = stats
        print(f"  {name:<40} {stats['pnl']:>8.1f} {stats['mdd']:>7.1f} "
              f"{stats['pnl_mdd']:>7.2f} {stats['max_consec_loss']:>5} "
              f"{stats['worst_daily']:>7.1f} {stats['max_dd_duration_bars']:>6} "
              f"{stats['avg_leverage']:>7.2f}")

    # ================================================================
    # PHASE 5: Combination Test (top 2-3 WF-PASS techniques)
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 5: Combination Test")
    print("=" * 78)

    combo_results = {}
    if len(wf_pass_names) >= 2:
        # Build combined kwargs from top 2 and top 3
        top_pass_sorted = sorted(wf_pass_names,
                                 key=lambda n: phase1_results[n]['pnl_mdd'],
                                 reverse=True)

        # Top 2 combination
        combo2_kwargs = {}
        for name in top_pass_sorted[:2]:
            combo2_kwargs.update(scenario_kwargs_map.get(name, {}))
        combo2_name = f"Combo2: {' + '.join(top_pass_sorted[:2])}"

        trades, eq_hist = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            **combo2_kwargs)
        combo2_stats = calc_stats(trades, eq_hist)

        # WF for combo2
        combo2_folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                              atr_ratio, ema_slope, neutral_start, neutral_end,
                              combo2_kwargs)
        combo2_pass = all(f['pnl'] > 0 for f in combo2_folds)

        combo_results[combo2_name] = {
            'is_stats': combo2_stats,
            'wf_folds': combo2_folds,
            'wf_pass': combo2_pass,
        }

        fold_str = ' | '.join([f"F{f['fold']}: {f['pnl']:+.1f}%" for f in combo2_folds])
        status = "PASS" if combo2_pass else "FAIL"
        print(f"\n  [{status}] {combo2_name}")
        print(f"         IS: PnL {combo2_stats['pnl']:.1f}%, MDD {combo2_stats['mdd']:.1f}%, "
              f"PnL/MDD {combo2_stats['pnl_mdd']:.2f}")
        print(f"         WF: {fold_str}")

        # Top 3 combination if available
        if len(top_pass_sorted) >= 3:
            combo3_kwargs = {}
            for name in top_pass_sorted[:3]:
                combo3_kwargs.update(scenario_kwargs_map.get(name, {}))
            combo3_name = f"Combo3: {' + '.join(top_pass_sorted[:3])}"

            trades, eq_hist = portfolio_sim(
                signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                **combo3_kwargs)
            combo3_stats = calc_stats(trades, eq_hist)

            combo3_folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                                  atr_ratio, ema_slope, neutral_start, neutral_end,
                                  combo3_kwargs)
            combo3_pass = all(f['pnl'] > 0 for f in combo3_folds)

            combo_results[combo3_name] = {
                'is_stats': combo3_stats,
                'wf_folds': combo3_folds,
                'wf_pass': combo3_pass,
            }

            fold_str = ' | '.join([f"F{f['fold']}: {f['pnl']:+.1f}%" for f in combo3_folds])
            status = "PASS" if combo3_pass else "FAIL"
            print(f"\n  [{status}] {combo3_name}")
            print(f"         IS: PnL {combo3_stats['pnl']:.1f}%, MDD {combo3_stats['mdd']:.1f}%, "
                  f"PnL/MDD {combo3_stats['pnl_mdd']:.2f}")
            print(f"         WF: {fold_str}")
    else:
        print("\n  Fewer than 2 WF-PASS candidates. Skipping combination test.")

    # ================================================================
    # PHASE 6: Verdict
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 6: VERDICT")
    print("=" * 78)

    print(f"\n  Baseline: PnL {baseline['pnl']:.1f}%, MDD {baseline['mdd']:.1f}%, "
          f"PnL/MDD {baseline['pnl_mdd']:.2f}")

    verdicts = {}

    # Individual technique verdicts
    hypothesis_groups = {}
    for c in candidates:
        h_prefix = c['name'].split('_')[0]
        if h_prefix not in hypothesis_groups:
            hypothesis_groups[h_prefix] = []
        hypothesis_groups[h_prefix].append(c)

    print("\n  --- Per-Hypothesis Verdicts ---")
    for h_prefix in sorted(hypothesis_groups.keys()):
        group = hypothesis_groups[h_prefix]
        # Find best variant by PnL/MDD that passed criterion
        passing_group = [c for c in group if c['passes_criterion']]
        wf_passing = [c for c in passing_group if c['name'] in wf_pass_names]

        if wf_passing:
            best = max(wf_passing, key=lambda c: c['stats']['pnl_mdd'])
            pnl_red = best['pnl_reduction_pct']
            mdd_red = best['mdd_reduction_pct']
            pm_chg = best['pnl_mdd_change_pct']
            verdict = "GO"
            verdicts[h_prefix] = {'verdict': verdict, 'best': best['name'],
                                  'pnl_mdd': best['stats']['pnl_mdd'],
                                  'mdd_reduction_pct': mdd_red,
                                  'pnl_reduction_pct': pnl_red}
            print(f"  [{verdict}] {h_prefix}: Best={best['name']}, "
                  f"PnL/MDD {best['stats']['pnl_mdd']:.2f} "
                  f"(MDD_red {mdd_red:+.1f}%, PnL_red {pnl_red:+.1f}%, "
                  f"P/M_chg {pm_chg:+.1f}%), WF 3/3 PASS")
        elif passing_group:
            best = max(passing_group, key=lambda c: c['stats']['pnl_mdd'])
            verdicts[h_prefix] = {'verdict': 'STOP (WF FAIL)', 'best': best['name']}
            print(f"  [STOP] {h_prefix}: Best IS={best['name']} "
                  f"(P/M {best['stats']['pnl_mdd']:.2f}), but WF FAIL")
        else:
            best = max(group, key=lambda c: c['stats']['pnl_mdd'])
            verdicts[h_prefix] = {'verdict': 'STOP (criterion fail)', 'best': best['name']}
            print(f"  [STOP] {h_prefix}: No variant passes MDD_red > PnL_red criterion")

    # Combination verdict
    if combo_results:
        print("\n  --- Combination Verdicts ---")
        for name, cr in combo_results.items():
            if cr['wf_pass']:
                s = cr['is_stats']
                pnl_red = (baseline['pnl'] - s['pnl']) / abs(baseline['pnl']) * 100
                mdd_red = (baseline['mdd'] - s['mdd']) / baseline['mdd'] * 100
                pm_chg = ((s['pnl_mdd'] - baseline['pnl_mdd'])
                          / baseline['pnl_mdd'] * 100 if baseline['pnl_mdd'] > 0 else 0)
                if mdd_red > pnl_red and mdd_red > 0:
                    print(f"  [GO] {name}")
                    print(f"       PnL {s['pnl']:.1f}%, MDD {s['mdd']:.1f}%, "
                          f"PnL/MDD {s['pnl_mdd']:.2f}")
                    print(f"       MDD_red {mdd_red:+.1f}%, PnL_red {pnl_red:+.1f}%, "
                          f"P/M_chg {pm_chg:+.1f}%")
                else:
                    print(f"  [STOP] {name} — criterion not met "
                          f"(MDD_red {mdd_red:+.1f}% vs PnL_red {pnl_red:+.1f}%)")
            else:
                print(f"  [STOP] {name} — WF FAIL")

    # Overall recommendation
    go_verdicts = {k: v for k, v in verdicts.items() if v['verdict'] == 'GO'}
    print(f"\n  === SUMMARY ===")
    print(f"  Total hypotheses tested: {len(hypothesis_groups)}")
    print(f"  Scenarios tested: {len(scenarios) - 1}")
    print(f"  Pass criterion (MDD_red > PnL_red): {len(passing)}")
    print(f"  WF 3/3 PASS: {len(wf_pass_names)}")
    print(f"  GO verdicts: {len(go_verdicts)}")

    if go_verdicts:
        print(f"\n  Recommended config changes:")
        for h, v in go_verdicts.items():
            print(f"    - {v['best']}: MDD_red {v['mdd_reduction_pct']:+.1f}%, "
                  f"PnL/MDD {v['pnl_mdd']:.2f}")
    else:
        print(f"\n  No new techniques meet all criteria. Current production stack is optimal.")

    # ---- Save Results ----
    elapsed = (datetime.now() - start_time).total_seconds()
    output = {
        'study': 'mdd_reduction_techniques',
        'date': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data_file': DATA_FILE,
        'patterns_file': PATTERNS_FILE,
        'n_patterns': len(pat_lookup),
        'neutral_window': {
            'start': int(neutral_start), 'end': int(neutral_end),
            'bars': int(neutral_bars),
        },
        'baseline': baseline,
        'phase1_is': {k: v for k, v in phase1_results.items()},
        'phase2_candidates': {
            'total_passing_criterion': len(passing),
            'top_wf_tested': [c['name'] for c in top_wf],
        },
        'phase3_wf': {k: {'pass': v['pass'],
                          'folds': v['folds']}
                      for k, v in wf_results.items()},
        'phase4_risk': risk_results,
        'phase5_combos': {k: {'is_stats': v['is_stats'],
                              'wf_pass': v['wf_pass'],
                              'wf_folds': v['wf_folds']}
                          for k, v in combo_results.items()},
        'phase6_verdicts': verdicts,
        'go_count': len(go_verdicts),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 78)


if __name__ == '__main__':
    main()
