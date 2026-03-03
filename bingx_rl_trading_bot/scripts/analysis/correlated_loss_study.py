#!/usr/bin/env python3
"""
Correlated Loss Mitigation Study
==================================

Problem: 81% of SHORT SL losses are clustered (2+ within 1hr).
6 cluster events account for 21/26 SHORT losses in live trading.
Root cause: Multiple same-direction positions with similar SL zones
hit simultaneously during strong BTC moves.

Hypotheses:
  H1: Post-SL Direction Cooldown — block same-dir NEW entries for N bars after any SL
      (prevents adding more fuel to a losing direction)
  H2: SL Proximity Guard — skip entry if new SL would cluster with existing same-dir SLs
      (prevents concentrated SL zones at entry time)
  H3: Entry Tempo Limit — max M same-dir entries per T-bar window
      (forces temporal/price dispersion of entries)
  H4: Max Cluster Exposure — cap worst-case simultaneous SL loss
      (limits max damage when cluster event does occur)
  H5: Cascading SL Tightening — after one SL exit, move remaining same-dir SLs closer
      (reduces loss when cluster is already forming)

Phases:
  1. IS Correlated Loss Diagnosis (baseline cluster metrics)
  2. Individual hypothesis testing (PnL, MDD, cluster metrics)
  3. Top candidates selection
  4. WF 3-fold validation
  5. Combination test
  6. Verdict

Baseline: Production v1.40.1 (7 active guards, G3/G4/M3 disabled)

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE adaptive (1-3x, wr_confidence w=12)
  - FEE = FEE_PCT * leverage (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - WF: 3-fold expanding window

Author: Research Agent
Date: 2026-03-03
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
# Constants — matching production v1.40.1 exactly
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

# v1.40.1: Loss Burst Brake DISABLED
LBB_ENABLED = False
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
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'correlated_loss_study.json')


# ============================================================
# Data Loading & Preprocessing (from mdd_reduction_study.py)
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
    if EXPECTED_WR <= 0:
        return MIN_LEVERAGE
    wr_conf = clamp(rolling_wr_frac / EXPECTED_WR, 0, 1)
    edge_q = clamp(rolling_edge_frac / REF_EDGE, 0, 1.5) if REF_EDGE > 0 else 0.5
    combined = clamp(wr_conf * edge_q, 0, 1)
    return MIN_LEVERAGE + (MAX_LEVERAGE - MIN_LEVERAGE) * combined


# ============================================================
# Enhanced Exit Check
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio):
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
            'entry_bar': entry_bar, 'exit_bar': bar, 'leverage': entry_lev,
            'sl_price': sl_price, 'tp_price': tp_price, 'exit_price': exit_price}


def _get_mdd_size_scale(equity, peak_equity, full_below=MDD_FULL_BELOW,
                        min_above=MDD_MIN_ABOVE, min_scale=MDD_MIN_SCALE):
    if peak_equity <= 0:
        return 1.0
    dd_pct = (peak_equity - equity) / peak_equity * 100
    if dd_pct <= full_below:
        return 1.0
    if dd_pct >= min_above:
        return min_scale
    return 1.0 - (1.0 - min_scale) * (dd_pct - full_below) / (min_above - full_below)


def _compute_sl_price(entry_price, direction, sl_pct, atr_r):
    """Compute SL price for a potential position."""
    eff_sl = max(0.1, sl_pct * atr_r - SLIPPAGE_BUFFER)
    if direction == 'LONG':
        return entry_price * (1 - eff_sl / 100)
    else:
        return entry_price * (1 + eff_sl / 100)


# ============================================================
# Enhanced Portfolio Simulator with Correlated Loss Hooks
# ============================================================

def portfolio_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                  atr_ratio, ema_slope, start_bar, end_bar,
                  # H1: Post-SL Direction Cooldown
                  h1_enabled=False, h1_cooldown_bars=12,
                  # H2: SL Proximity Guard
                  h2_enabled=False, h2_sl_cluster_pct=3.0, h2_max_cluster=2,
                  # H3: Entry Tempo Limit
                  h3_enabled=False, h3_max_entries=2, h3_window_bars=48,
                  # H4: Max Cluster Exposure
                  h4_enabled=False, h4_max_cluster_loss_pct=5.0,
                  # H5: Cascading SL Tightening (after first SL, tighten remaining)
                  h5_enabled=False, h5_tighten_pct=50.0):
    """N-pos portfolio with production v1.40.1 features + correlated loss hooks.

    Production features always ON:
      - N=9 compound, direction_cap=7, regime*0.3, agg_risk_cap(3/7%),
        momentum_guard, timeout(864 DROP), ATR-scaled TP/SL,
        adaptive leverage (wr_confidence w=12), MDD sizing
      - v1.40.1: G3/G4/M3 DISABLED
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

    # Momentum guard state
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # H1: Post-SL cooldown state
    h1_cooldown_until = {'LONG': -1, 'SHORT': -1}

    # H3: Entry tempo tracking (per-direction)
    h3_entry_times = {'LONG': [], 'SHORT': []}

    # Cluster event tracking for diagnostics
    cluster_events = []  # list of {bar, direction, n_hits, total_loss}
    sl_exit_buffer = []  # track SL exits within same bar range for cluster detection

    equity_history = []

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # --- Exits ---
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_exits = []  # track SL exits this bar for cluster detection

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

                if not is_win:
                    bar_sl_exits.append({
                        'direction': pos['direction'],
                        'pnl_portfolio': pnl_portfolio,
                        'pattern': pos['pattern'],
                    })
                    # H1: Set cooldown after SL
                    if h1_enabled:
                        h1_cooldown_until[pos['direction']] = bar + h1_cooldown_bars

                    # H5: Cascading SL tightening — move remaining same-dir SLs closer
                    if h5_enabled:
                        for other_pos in positions:
                            if (other_pos['slot'] not in closed_slots and
                                    other_pos['direction'] == pos['direction']):
                                # Tighten SL: reduce effective SL distance by h5_tighten_pct%
                                old_sl = other_pos['sl_pct']
                                other_pos['sl_pct'] = old_sl * (1 - h5_tighten_pct / 100)

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Detect cluster events (2+ same-dir SL exits in same bar)
        for direction in ['LONG', 'SHORT']:
            dir_exits = [e for e in bar_sl_exits if e['direction'] == direction]
            if len(dir_exits) >= 2:
                total_loss = sum(e['pnl_portfolio'] for e in dir_exits)
                cluster_events.append({
                    'bar': bar, 'direction': direction,
                    'n_hits': len(dir_exits), 'total_loss': total_loss
                })
        # Also track near-bar clusters (within 12 bars = 1hr)
        sl_exit_buffer.append((bar, bar_sl_exits))
        # Prune old buffer entries
        sl_exit_buffer = [(b, exs) for b, exs in sl_exit_buffer if bar - b <= 12]

        # Detect 1hr clusters
        for direction in ['LONG', 'SHORT']:
            recent_dir_exits = []
            for b, exs in sl_exit_buffer:
                recent_dir_exits.extend([e for e in exs if e['direction'] == direction])
            if len(recent_dir_exits) >= 2 and bar_sl_exits:
                dir_in_bar = [e for e in bar_sl_exits if e['direction'] == direction]
                if dir_in_bar:
                    total_loss = sum(e['pnl_portfolio'] for e in recent_dir_exits)
                    # Only record if this is a new cluster (check no duplicate)
                    if not cluster_events or cluster_events[-1]['bar'] != bar or cluster_events[-1]['direction'] != direction:
                        if len(recent_dir_exits) >= 2:
                            cluster_events.append({
                                'bar': bar, 'direction': direction,
                                'n_hits': len(recent_dir_exits),
                                'total_loss': total_loss, 'type': '1hr_cluster'
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

            # Momentum guard
            if bar < momentum_pause_until.get(direction, -1):
                continue

            # Regime sizing
            sm = 1.0
            if REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if (s > 0 and direction == 'SHORT') or (s <= 0 and direction == 'LONG'):
                    sm = REGIME_MULT

            # MDD sizing
            mdd_scale = _get_mdd_size_scale(equity, peak_equity)
            sm *= mdd_scale

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

            # ========== NEW CORRELATED LOSS GUARDS ==========

            # H1: Post-SL Direction Cooldown
            if h1_enabled and bar < h1_cooldown_until.get(direction, -1):
                continue

            # H2: SL Proximity Guard
            if h2_enabled and entry_bar < n_bars:
                proposed_entry = opens[entry_bar]
                if proposed_entry > 0:
                    proposed_sl = _compute_sl_price(proposed_entry, direction, sl_pct, new_r)
                    cluster_count = 0
                    for p in positions:
                        if p['direction'] == direction:
                            p_entry = opens[p['entry_bar']] if p['entry_bar'] < n_bars else 0
                            if p_entry > 0:
                                p_r = 1.0
                                if (atr_ratio is not None and p['signal_bar'] < len(atr_ratio)
                                        and not np.isnan(atr_ratio[p['signal_bar']])):
                                    p_r = clamp(atr_ratio[p['signal_bar']], ATR_CLAMP_LO, ATR_CLAMP_HI)
                                existing_sl = _compute_sl_price(p_entry, direction, p['sl_pct'], p_r)
                                sl_distance_pct = abs(proposed_sl - existing_sl) / proposed_sl * 100
                                if sl_distance_pct < h2_sl_cluster_pct:
                                    cluster_count += 1
                    if cluster_count >= h2_max_cluster:
                        continue

            # H3: Entry Tempo Limit
            if h3_enabled:
                # Prune old entries outside window
                h3_entry_times[direction] = [t for t in h3_entry_times[direction]
                                             if bar - t <= h3_window_bars]
                if len(h3_entry_times[direction]) >= h3_max_entries:
                    continue

            # H4: Max Cluster Exposure
            if h4_enabled:
                # Compute worst-case: if ALL same-dir SLs hit now + this new one
                worst_case = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_pnl = p['sl_pct'] * (1.0 / N_SLOTS) * p['leverage'] * p.get('size_mult', 1.0)
                        worst_case += p_pnl
                new_worst = sl_pct * new_r * (1.0 / N_SLOTS) * entry_leverage * sm
                if worst_case + new_worst > h4_max_cluster_loss_pct:
                    continue

            # ========== END CORRELATED LOSS GUARDS ==========

            positions.append({
                'slot': f"{pat}_{sig_bar}", 'signal_bar': sig_bar,
                'entry_bar': entry_bar, 'direction': direction,
                'pattern': pat, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'size_mult': sm, 'leverage': entry_leverage,
            })

            # H3: Record entry time
            if h3_enabled:
                h3_entry_times[direction].append(bar)

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

    return trades, equity_history, cluster_events


# ============================================================
# Stats Calculation (enhanced with cluster metrics)
# ============================================================

def calc_stats(trades, equity_history, cluster_events=None):
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

    # Worst daily PnL
    worst_daily = 0.0
    if len(equity_history) >= BARS_PER_DAY:
        for i in range(BARS_PER_DAY, len(equity_history)):
            prev_eq = equity_history[i - BARS_PER_DAY]
            if prev_eq > 0:
                daily_chg = (equity_history[i] - prev_eq) / prev_eq * 100
                if daily_chg < worst_daily:
                    worst_daily = daily_chg

    # Cluster metrics
    n_cluster_events = 0
    total_cluster_loss = 0.0
    max_cluster_loss = 0.0
    if cluster_events:
        # Deduplicate: keep only 1hr_cluster type or same-bar events
        seen = set()
        unique_clusters = []
        for ce in cluster_events:
            key = (ce['bar'], ce['direction'])
            if key not in seen:
                seen.add(key)
                unique_clusters.append(ce)
        n_cluster_events = len(unique_clusters)
        for ce in unique_clusters:
            total_cluster_loss += ce['total_loss']
            if ce['total_loss'] < max_cluster_loss:
                max_cluster_loss = ce['total_loss']

    # Correlated SL detection (same-bar SL exits)
    sl_trades = [t for t in sorted_t if t.get('reason') == 'SL']
    same_bar_sl_groups = {}
    for t in sl_trades:
        key = (t['exit_bar'], t['direction'])
        same_bar_sl_groups.setdefault(key, []).append(t)
    corr_sl_events = {k: v for k, v in same_bar_sl_groups.items() if len(v) >= 2}
    n_corr_sl = sum(len(v) for v in corr_sl_events.values())
    n_corr_events = len(corr_sl_events)
    corr_loss = sum(sum(t['pnl_portfolio'] for t in v) for v in corr_sl_events.values())

    # Direction breakdown
    long_trades = [t for t in trades if t['direction'] == 'LONG']
    short_trades = [t for t in trades if t['direction'] == 'SHORT']
    long_wr = sum(1 for t in long_trades if t['pnl_slot'] > 0) / max(1, len(long_trades)) * 100
    short_wr = sum(1 for t in short_trades if t['pnl_slot'] > 0) / max(1, len(short_trades)) * 100
    long_pnl = sum(t['pnl_portfolio'] for t in long_trades)
    short_pnl = sum(t['pnl_portfolio'] for t in short_trades)

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(total_pnl / mdd, 2) if mdd > 0 else 0,
        'avg_leverage': round(avg_lev, 2),
        'max_consec_loss': max_consec_loss,
        'worst_daily': round(worst_daily, 2),
        # Cluster-specific metrics
        'corr_sl_events': n_corr_events,
        'corr_sl_trades': n_corr_sl,
        'corr_sl_loss': round(corr_loss, 2),
        'max_cluster_loss': round(max_cluster_loss, 2),
        # Direction breakdown
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_wr': round(long_wr, 1),
        'short_wr': round(short_wr, 1),
        'long_pnl': round(long_pnl, 2),
        'short_pnl': round(short_pnl, 2),
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

        trades, eq_hist, cluster_events = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            **scenario_kwargs)
        stats = calc_stats(trades, eq_hist, cluster_events)
        stats['fold'] = fold + 1
        stats['oos_bars'] = int(oos_end - oos_start)
        results.append(stats)

    return results


# ============================================================
# Scenario Definitions
# ============================================================

def build_scenarios():
    scenarios = []

    # Baseline (production v1.40.1 — G3/G4/M3 disabled)
    scenarios.append(('Baseline_v1.40.1', {}))

    # H1: Post-SL Direction Cooldown
    for cooldown in [6, 12, 24, 48]:
        scenarios.append((f'H1_PostSL_cd{cooldown}',
                          {'h1_enabled': True, 'h1_cooldown_bars': cooldown}))

    # H2: SL Proximity Guard
    for sl_pct in [2.0, 3.0, 5.0]:
        for max_cl in [1, 2, 3]:
            scenarios.append((f'H2_SLProx_pct{sl_pct}_max{max_cl}',
                              {'h2_enabled': True, 'h2_sl_cluster_pct': sl_pct,
                               'h2_max_cluster': max_cl}))

    # H3: Entry Tempo Limit
    for max_ent in [2, 3]:
        for window in [24, 48, 96]:
            scenarios.append((f'H3_Tempo_max{max_ent}_w{window}',
                              {'h3_enabled': True, 'h3_max_entries': max_ent,
                               'h3_window_bars': window}))

    # H4: Max Cluster Exposure
    for max_loss in [3.0, 4.0, 5.0, 6.0]:
        scenarios.append((f'H4_ClusterCap_{max_loss}',
                          {'h4_enabled': True, 'h4_max_cluster_loss_pct': max_loss}))

    # H5: Cascading SL Tightening
    for tighten in [25, 50, 75]:
        scenarios.append((f'H5_Cascade_t{tighten}',
                          {'h5_enabled': True, 'h5_tighten_pct': float(tighten)}))

    return scenarios


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("CORRELATED LOSS MITIGATION STUDY")
    print("=" * 78)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Baseline: Production v1.40.1 (G3/G4/M3 disabled)")

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
    # PHASE 1: Baseline Cluster Diagnosis
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 1: Baseline Cluster Diagnosis (IS, Neutral Window)")
    print("=" * 78)

    trades, eq_hist, cluster_events = portfolio_sim(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end)
    baseline_stats = calc_stats(trades, eq_hist, cluster_events)

    print(f"\n  Baseline v1.40.1 IS Results:")
    print(f"    Trades: {baseline_stats['trades']}, WR: {baseline_stats['wr']}%")
    print(f"    PnL: {baseline_stats['pnl']:+.2f}%, MDD: {baseline_stats['mdd']:.2f}%")
    print(f"    PnL/MDD: {baseline_stats['pnl_mdd']:.2f}")
    print(f"    Worst daily: {baseline_stats['worst_daily']:.2f}%")
    print(f"    LONG: {baseline_stats['long_trades']}T WR {baseline_stats['long_wr']}% PnL {baseline_stats['long_pnl']:+.2f}%")
    print(f"    SHORT: {baseline_stats['short_trades']}T WR {baseline_stats['short_wr']}% PnL {baseline_stats['short_pnl']:+.2f}%")
    print(f"    --- Cluster Metrics ---")
    print(f"    Correlated SL events (2+ same-bar+dir): {baseline_stats['corr_sl_events']}")
    print(f"    Correlated SL trades: {baseline_stats['corr_sl_trades']}")
    print(f"    Correlated SL loss: {baseline_stats['corr_sl_loss']:+.2f}%")

    # ================================================================
    # PHASE 2: Individual Hypothesis Testing (IS)
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 2: Individual Hypothesis Testing (IS, Neutral Window)")
    print("=" * 78)

    scenarios = build_scenarios()
    print(f"  Running {len(scenarios)} scenarios...")

    phase2_results = {}
    header = (f"  {'Scenario':<32} {'Trades':>6} {'WR%':>6} {'PnL%':>9} "
              f"{'MDD%':>7} {'P/M':>7} {'CorrEv':>6} {'CorrL%':>7} {'WrstD':>7}")
    print(f"\n{header}")
    print("  " + "-" * 100)

    for name, kwargs in scenarios:
        trades, eq_hist, ce = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            **kwargs)
        stats = calc_stats(trades, eq_hist, ce)
        phase2_results[name] = stats
        print(f"  {name:<32} {stats['trades']:>6} {stats['wr']:>6.1f} {stats['pnl']:>+9.2f} "
              f"{stats['mdd']:>7.2f} {stats['pnl_mdd']:>7.2f} {stats['corr_sl_events']:>6} "
              f"{stats['corr_sl_loss']:>+7.2f} {stats['worst_daily']:>7.2f}")

    # ================================================================
    # PHASE 3: Select Top Candidates
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 3: Top Candidate Selection")
    print("=" * 78)

    bl = phase2_results['Baseline_v1.40.1']

    # Score: primarily reduce correlated loss + maintain PnL/MDD
    candidates = []
    for name, stats in phase2_results.items():
        if name == 'Baseline_v1.40.1':
            continue
        corr_improve = bl['corr_sl_loss'] - stats['corr_sl_loss']  # positive = less loss
        pnl_delta = stats['pnl'] - bl['pnl']
        mdd_delta = stats['mdd'] - bl['mdd']  # negative = better
        pnl_mdd_delta = stats['pnl_mdd'] - bl['pnl_mdd']
        worst_daily_improve = bl['worst_daily'] - stats['worst_daily']  # positive = better

        # Composite score: cluster reduction + PnL/MDD improvement
        score = corr_improve * 2 + pnl_mdd_delta + worst_daily_improve
        candidates.append({
            'name': name, 'score': score,
            'corr_improve': corr_improve,
            'pnl_delta': pnl_delta, 'mdd_delta': mdd_delta,
            'pnl_mdd_delta': pnl_mdd_delta,
            'worst_daily_improve': worst_daily_improve,
            'stats': stats
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  {'Rank':<5} {'Scenario':<32} {'Score':>7} {'CorrImpr':>8} "
          f"{'PnLΔ':>8} {'MDDΔ':>7} {'P/MΔ':>7} {'WrstDΔ':>7}")
    print("  " + "-" * 90)
    for i, c in enumerate(candidates[:15]):
        print(f"  {i+1:<5} {c['name']:<32} {c['score']:>7.2f} {c['corr_improve']:>+8.2f} "
              f"{c['pnl_delta']:>+8.2f} {c['mdd_delta']:>+7.2f} "
              f"{c['pnl_mdd_delta']:>+7.2f} {c['worst_daily_improve']:>+7.2f}")

    # Select top 5 for WF
    top5 = candidates[:5]
    print(f"\n  Top 5 selected for WF validation:")
    for c in top5:
        print(f"    - {c['name']} (score {c['score']:.2f})")

    # ================================================================
    # PHASE 4: WF 3-fold Validation
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 4: WF 3-fold Expanding Window Validation")
    print("=" * 78)

    # WF baseline
    bl_folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                      atr_ratio, ema_slope, neutral_start, neutral_end,
                      scenario_kwargs={})
    print(f"\n  Baseline WF:")
    all_pass = True
    for f in bl_folds:
        status = "PASS" if f['pnl'] > 0 else "FAIL"
        if f['pnl'] <= 0:
            all_pass = False
        print(f"    Fold {f['fold']}: PnL {f['pnl']:+.2f}%, WR {f['wr']:.1f}%, "
              f"MDD {f['mdd']:.2f}%, CorrEv {f['corr_sl_events']} | {status}")
    print(f"  Baseline: {'3/3 PASS' if all_pass else 'FAIL'}")

    wf_results = {}
    wf_pass_scenarios = []

    for c in top5:
        name = c['name']
        # Extract kwargs from scenario list
        kwargs = {}
        for sname, skwargs in build_scenarios():
            if sname == name:
                kwargs = skwargs
                break

        folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, neutral_start, neutral_end,
                       scenario_kwargs=kwargs)
        wf_results[name] = folds

        all_pass = all(f['pnl'] > 0 for f in folds)
        min_pnl = min(f['pnl'] for f in folds)
        total_corr = sum(f['corr_sl_events'] for f in folds)

        print(f"\n  {name}:")
        for f in folds:
            status = "PASS" if f['pnl'] > 0 else "FAIL"
            print(f"    Fold {f['fold']}: PnL {f['pnl']:+.2f}%, WR {f['wr']:.1f}%, "
                  f"MDD {f['mdd']:.2f}%, CorrEv {f['corr_sl_events']} | {status}")
        verdict = '3/3 PASS' if all_pass else 'FAIL'
        print(f"  Verdict: {verdict} (min fold {min_pnl:+.2f}%, total CorrEv {total_corr})")

        if all_pass:
            wf_pass_scenarios.append({
                'name': name, 'kwargs': kwargs,
                'is_stats': c['stats'], 'wf_folds': folds,
                'score': c['score'], 'min_fold_pnl': min_pnl,
            })

    # ================================================================
    # PHASE 5: Combination Test (if 2+ WF PASS)
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 5: Combination Test")
    print("=" * 78)

    combo_results = {}
    if len(wf_pass_scenarios) >= 2:
        # Try top 2 combined
        for i in range(min(3, len(wf_pass_scenarios))):
            for j in range(i + 1, min(4, len(wf_pass_scenarios))):
                a, b = wf_pass_scenarios[i], wf_pass_scenarios[j]
                combo_name = f"Combo_{a['name']}+{b['name']}"
                combo_kwargs = {**a['kwargs'], **b['kwargs']}

                trades, eq_hist, ce = portfolio_sim(
                    signal_tuples, opens, highs, lows, closes, n_bars,
                    atr_ratio, ema_slope, neutral_start, neutral_end,
                    **combo_kwargs)
                combo_stats = calc_stats(trades, eq_hist, ce)

                # WF
                combo_folds = run_wf(
                    signal_tuples, opens, highs, lows, closes, n_bars,
                    atr_ratio, ema_slope, neutral_start, neutral_end,
                    scenario_kwargs=combo_kwargs)
                combo_pass = all(f['pnl'] > 0 for f in combo_folds)
                combo_min = min(f['pnl'] for f in combo_folds)

                combo_results[combo_name] = {
                    'is_stats': combo_stats,
                    'wf_folds': combo_folds,
                    'wf_pass': combo_pass,
                    'min_fold_pnl': combo_min,
                }

                print(f"\n  {combo_name}:")
                print(f"    IS: Trades {combo_stats['trades']}, PnL {combo_stats['pnl']:+.2f}%, "
                      f"MDD {combo_stats['mdd']:.2f}%, P/M {combo_stats['pnl_mdd']:.2f}, "
                      f"CorrEv {combo_stats['corr_sl_events']}, CorrL {combo_stats['corr_sl_loss']:+.2f}%")
                for f in combo_folds:
                    status = "PASS" if f['pnl'] > 0 else "FAIL"
                    print(f"    Fold {f['fold']}: PnL {f['pnl']:+.2f}% | {status}")
                print(f"    Verdict: {'3/3 PASS' if combo_pass else 'FAIL'} "
                      f"(min {combo_min:+.2f}%)")
    else:
        print("  < 2 WF PASS scenarios — skipping combinations")

    # ================================================================
    # PHASE 6: Verdict
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 6: VERDICT")
    print("=" * 78)

    # Collect all WF-PASS results
    all_candidates = []
    for s in wf_pass_scenarios:
        all_candidates.append({
            'name': s['name'],
            'type': 'individual',
            'is_pnl': s['is_stats']['pnl'],
            'is_mdd': s['is_stats']['mdd'],
            'is_pnl_mdd': s['is_stats']['pnl_mdd'],
            'is_corr_events': s['is_stats']['corr_sl_events'],
            'is_corr_loss': s['is_stats']['corr_sl_loss'],
            'wf_min_fold': s['min_fold_pnl'],
            'wf_pass': True,
            'score': s['score'],
        })
    for cname, cdata in combo_results.items():
        if cdata['wf_pass']:
            all_candidates.append({
                'name': cname,
                'type': 'combo',
                'is_pnl': cdata['is_stats']['pnl'],
                'is_mdd': cdata['is_stats']['mdd'],
                'is_pnl_mdd': cdata['is_stats']['pnl_mdd'],
                'is_corr_events': cdata['is_stats']['corr_sl_events'],
                'is_corr_loss': cdata['is_stats']['corr_sl_loss'],
                'wf_min_fold': cdata['min_fold_pnl'],
                'wf_pass': True,
                'score': 0,
            })

    # Sort by cluster reduction then PnL/MDD
    all_candidates.sort(key=lambda x: (-x['is_corr_events'], x['is_pnl_mdd']), reverse=False)
    all_candidates.sort(key=lambda x: x['is_corr_loss'])  # least corr loss first (most negative = most loss)
    # Actually sort by corr_loss ascending (closer to 0 = better)
    all_candidates.sort(key=lambda x: abs(x['is_corr_loss']))

    bl_stats = phase2_results['Baseline_v1.40.1']
    print(f"\n  Baseline: PnL {bl_stats['pnl']:+.2f}%, MDD {bl_stats['mdd']:.2f}%, "
          f"P/M {bl_stats['pnl_mdd']:.2f}, CorrEv {bl_stats['corr_sl_events']}, "
          f"CorrL {bl_stats['corr_sl_loss']:+.2f}%")

    if all_candidates:
        print(f"\n  WF-PASS Candidates (sorted by cluster reduction):")
        for c in all_candidates:
            corr_delta = c['is_corr_loss'] - bl_stats['corr_sl_loss']
            pnl_delta = c['is_pnl'] - bl_stats['pnl']
            mdd_delta = c['is_mdd'] - bl_stats['mdd']
            print(f"    {c['name']}")
            print(f"      PnL {c['is_pnl']:+.2f}% ({pnl_delta:+.2f}), "
                  f"MDD {c['is_mdd']:.2f}% ({mdd_delta:+.2f}), "
                  f"P/M {c['is_pnl_mdd']:.2f}")
            print(f"      CorrEv {c['is_corr_events']} ({c['is_corr_events'] - bl_stats['corr_sl_events']:+d}), "
                  f"CorrLoss {c['is_corr_loss']:+.2f}% ({corr_delta:+.2f})")
            print(f"      WF min fold: {c['wf_min_fold']:+.2f}%")

        best = all_candidates[0]
        print(f"\n  RECOMMENDATION: {best['name']}")
        print(f"    PnL/MDD: {best['is_pnl_mdd']:.2f} (baseline {bl_stats['pnl_mdd']:.2f})")
        print(f"    CorrLoss: {best['is_corr_loss']:+.2f}% (baseline {bl_stats['corr_sl_loss']:+.2f}%)")
    else:
        print("\n  No WF-PASS candidates found. All hypotheses STOP.")
        print("  Consider: problem may require fundamentally different approach")
        print("  (e.g., reduce SHORT pattern count, or stricter direction_cap)")

    # ================================================================
    # Save Results
    # ================================================================
    output = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'baseline': 'v1.40.1 (G3/G4/M3 disabled)',
            'data_bars': int(n_bars),
            'neutral_window': [int(neutral_start), int(neutral_end)],
            'patterns': f"{len(pat_lookup)} ({n_long}L + {n_short}S)",
        },
        'phase1_baseline': baseline_stats,
        'phase2_all': {k: v for k, v in phase2_results.items()},
        'phase3_ranking': [{
            'rank': i + 1, 'name': c['name'],
            'score': round(c['score'], 2),
            'corr_improve': round(c['corr_improve'], 2),
            'pnl_delta': round(c['pnl_delta'], 2),
            'mdd_delta': round(c['mdd_delta'], 2),
        } for i, c in enumerate(candidates[:10])],
        'phase4_wf': {name: [dict(f) for f in folds]
                      for name, folds in wf_results.items()},
        'phase4_wf_baseline': [dict(f) for f in bl_folds],
        'phase5_combos': {name: {
            'is_stats': data['is_stats'],
            'wf_pass': data['wf_pass'],
            'min_fold_pnl': data['min_fold_pnl'],
        } for name, data in combo_results.items()},
        'phase6_verdict': {
            'wf_pass_candidates': all_candidates,
            'recommendation': all_candidates[0]['name'] if all_candidates else 'NONE',
        }
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
