#!/usr/bin/env python3
"""
Adaptive Leverage Risk-Focused Study — Deep Risk Analysis

v2 study verdict was STOP (PnL/MDD), but risk management properties may be
superior. This study examines:

  Phase 1: Extended risk metrics (max DD duration, worst daily loss, tail risk,
           consecutive loss severity, recovery time)
  Phase 2: Leverage dynamics during stress (leverage behavior during drawdowns,
           auto-reduction effectiveness)
  Phase 3: OOS stability analysis (min fold, fold variance, fold consistency)
  Phase 4: Monte Carlo robustness (trade-order shuffling, 1000 sims)
  Phase 5: Stress period isolation (worst 30-day windows, regime transition)
  Phase 6: Verdict (risk-adjusted)

Candidates:
  Fixed_3x (baseline)
  H1_StepTiers_w10 (v1 best PnL/MDD)
  H5_WRConf_w12 (v2 best OOS stability: min fold +23.4%)
  H5_WRConf_w8 (v2 runner-up)
  H8_Combined_w10 (v2 blend approach)
  H9_ExpDecay_w10 (v2 best IS PnL/MDD)

Standard Research Protocol applies.

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
# Constants
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

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'adaptive_leverage_risk_study.json')


# ============================================================
# Data Loading
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


# ============================================================
# Leverage Functions
# ============================================================

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def get_leverage_step_tiers(rolling_wr_pct):
    if rolling_wr_pct < 50: return 1.0
    elif rolling_wr_pct < 60: return 1.5
    elif rolling_wr_pct < 70: return 2.0
    elif rolling_wr_pct < 80: return 2.5
    else: return 3.0


def get_leverage_wr_confidence(rolling_wr, rolling_edge, expected_wr, ref_edge,
                               min_lev=MIN_LEVERAGE, max_lev=MAX_LEVERAGE):
    if expected_wr <= 0: return min_lev
    wr_conf = clamp(rolling_wr / expected_wr, 0, 1)
    edge_q = clamp(rolling_edge / ref_edge, 0, 1.5) if ref_edge > 0 else 0.5
    combined = clamp(wr_conf * edge_q, 0, 1)
    return min_lev + (max_lev - min_lev) * combined


def get_leverage_combined(rolling_wr, rolling_edge, expected_wr, ref_edge,
                          tp_pct, sl_pct, min_lev=MIN_LEVERAGE, max_lev=MAX_LEVERAGE):
    h5 = get_leverage_wr_confidence(rolling_wr, rolling_edge, expected_wr, ref_edge, 0, 1)
    if (tp_pct + sl_pct) <= 0:
        h7 = 0
    else:
        be_wr = sl_pct / (tp_pct + sl_pct)
        margin = rolling_wr - be_wr
        h7 = clamp(margin / 0.30, 0, 1) if margin > 0 else 0
    blend = 0.6 * h5 + 0.4 * h7
    return min_lev + (max_lev - min_lev) * clamp(blend, 0, 1)


def get_leverage_exp_decay(rolling_wr, expected_wr, min_lev=MIN_LEVERAGE, max_lev=MAX_LEVERAGE):
    gap = max(expected_wr - rolling_wr, 0)
    decay = math.exp(-gap / 0.10)
    return min_lev + (max_lev - min_lev) * decay


# ============================================================
# Extended Portfolio Simulator — returns detailed trade + equity data
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio):
    entry_bar = pos['entry_bar']
    if bar < entry_bar: return None
    entry = opens[entry_bar]
    if entry <= 0: return None

    sig_bar = pos['signal_bar']
    if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[sig_bar]))
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


def portfolio_detailed(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, start_bar, end_bar,
                       leverage_fn, window_size=15,
                       expected_wr=0.732, ref_edge=0.00126):
    """Portfolio sim returning detailed trade list + per-bar equity curve."""
    size_pct = 100.0 / N_SLOTS
    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0

    recent_results = deque(maxlen=window_size)
    recent_long = deque(maxlen=window_size)
    recent_short = deque(maxlen=window_size)
    recent_pnls = deque(maxlen=window_size)
    recent_win_pnls = deque(maxlen=window_size)
    recent_loss_pnls = deque(maxlen=window_size)
    n_trades_total = 0
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Per-bar equity tracking
    equity_curve = []
    leverage_at_entry = []  # (bar, leverage, equity_at_time, dd_at_time)

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
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
                if is_win:
                    recent_win_pnls.append(result['pnl_slot'])
                else:
                    recent_loss_pnls.append(abs(result['pnl_slot']))
                if pos['direction'] == 'LONG':
                    recent_long.append(1 if is_win else 0)
                else:
                    recent_short.append(1 if is_win else 0)
                n_trades_total += 1

        positions = [p for p in positions if p['slot'] not in closed_slots]
        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        equity_curve.append((bar, equity, dd))

        # Momentum guard
        if MOMENTUM_LOOKBACK > 0 and bar >= MOMENTUM_LOOKBACK:
            price_now, price_ago = closes[bar], closes[bar - MOMENTUM_LOOKBACK]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > MOMENTUM_THRESHOLD:
                    momentum_pause_until['SHORT'] = bar + MOMENTUM_COOLDOWN
                elif pct_change < -MOMENTUM_THRESHOLD:
                    momentum_pause_until['LONG'] = bar + MOMENTUM_COOLDOWN

        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= N_SLOTS: continue
            if sum(1 for p in positions if p['direction'] == direction) >= DIRECTION_CAP: continue
            if any(p['pattern'] == pat for p in positions): continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars: continue
            if bar < momentum_pause_until.get(direction, -1): continue

            sm = 1.0
            if REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if (s > 0 and direction == 'SHORT') or (s <= 0 and direction == 'LONG'):
                    sm = REGIME_MULT

            rolling_wr_pct = (sum(recent_results) / len(recent_results) * 100
                              if recent_results else 50.0)
            rolling_wr_frac = rolling_wr_pct / 100.0
            rolling_edge_pct = np.mean(list(recent_pnls)) if recent_pnls else 0.0
            rolling_edge_frac = rolling_edge_pct / 100.0
            avg_win = np.mean(list(recent_win_pnls)) if recent_win_pnls else 0.0
            avg_loss = np.mean(list(recent_loss_pnls)) if recent_loss_pnls else 1.0

            ctx = {
                'rolling_wr': rolling_wr_pct,
                'rolling_wr_long': (sum(recent_long) / len(recent_long) * 100
                                    if recent_long else 50.0),
                'rolling_wr_short': (sum(recent_short) / len(recent_short) * 100
                                     if recent_short else 50.0),
                'n_trades': n_trades_total,
                'direction': direction,
                'window_size': window_size,
                'rolling_wr_frac': rolling_wr_frac,
                'rolling_edge_frac': rolling_edge_frac,
                'rolling_edge_pct': rolling_edge_pct,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'expected_wr': expected_wr,
                'ref_edge': ref_edge,
            }
            entry_leverage = max(MIN_LEVERAGE, min(MAX_LEVERAGE, leverage_fn(ctx)))

            # Aggregate risk cap
            if AGG_RISK_COUNTER > 0 or AGG_RISK_WITH > 0:
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
                            p_r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[p_sig]))
                        existing_exposure += (p['sl_pct'] * p_r * (1.0 / N_SLOTS)
                                              * p['leverage'] * p.get('size_mult', 1.0))

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[sig_bar]))
                new_exposure = sl_pct * new_r * (1.0 / N_SLOTS) * entry_leverage * sm
                if existing_exposure + new_exposure > cap_pct:
                    continue

            leverage_at_entry.append((bar, entry_leverage, equity, dd))

            positions.append({
                'slot': f"{pat}_{sig_bar}", 'signal_bar': sig_bar,
                'entry_bar': entry_bar, 'direction': direction,
                'pattern': pat, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'size_mult': sm, 'leverage': entry_leverage,
            })

    # Force-close remaining
    for pos in positions:
        if pos['entry_bar'] >= n_bars: continue
        entry = opens[pos['entry_bar']]
        if entry <= 0: continue
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

    return trades, equity_curve, leverage_at_entry


# ============================================================
# Risk Metrics Computation
# ============================================================

def compute_risk_metrics(trades, equity_curve):
    """Compute extended risk metrics from trades and equity curve."""
    if not trades:
        return {}

    sorted_t = sorted(trades, key=lambda x: x['entry_bar'])
    pnls = [t['pnl_portfolio'] for t in sorted_t]
    slot_pnls = [t['pnl_slot'] for t in sorted_t]
    leverages = [t.get('leverage', MAX_LEVERAGE) for t in sorted_t]

    # Basic stats
    n = len(trades)
    wins = sum(1 for p in slot_pnls if p > 0)
    losses = sum(1 for p in slot_pnls if p <= 0)
    wr = wins / n * 100

    eq_values = [e[1] for e in equity_curve]
    dd_values = [e[2] for e in equity_curve]
    final_eq = eq_values[-1] if eq_values else 100.0
    total_pnl = final_eq - 100.0
    max_dd = max(dd_values) if dd_values else 0

    # 1. Max DD duration (bars)
    in_dd = False
    dd_start = 0
    max_dd_duration = 0
    for i, (bar, eq, dd) in enumerate(equity_curve):
        if dd > 0.01:  # in drawdown
            if not in_dd:
                dd_start = i
                in_dd = True
        else:
            if in_dd:
                duration = i - dd_start
                if duration > max_dd_duration:
                    max_dd_duration = duration
                in_dd = False
    if in_dd:
        duration = len(equity_curve) - dd_start
        if duration > max_dd_duration:
            max_dd_duration = duration

    # 2. Worst daily PnL (288-bar windows)
    daily_pnls = []
    for i in range(0, len(equity_curve) - BARS_PER_DAY, BARS_PER_DAY):
        start_eq = equity_curve[i][1]
        end_eq = equity_curve[i + BARS_PER_DAY][1]
        daily_pnl = (end_eq - start_eq) / start_eq * 100 if start_eq > 0 else 0
        daily_pnls.append(daily_pnl)

    worst_daily = min(daily_pnls) if daily_pnls else 0
    daily_std = np.std(daily_pnls) if daily_pnls else 0
    daily_sharpe = (np.mean(daily_pnls) / daily_std) if daily_std > 0 else 0

    # 3. Consecutive losses (max streak)
    max_consec_loss = 0
    curr_streak = 0
    consec_loss_pnl = 0
    worst_streak_pnl = 0
    for p in slot_pnls:
        if p <= 0:
            curr_streak += 1
            consec_loss_pnl += p
            if curr_streak > max_consec_loss:
                max_consec_loss = curr_streak
                worst_streak_pnl = consec_loss_pnl
        else:
            curr_streak = 0
            consec_loss_pnl = 0

    # 4. Tail risk: worst 10% of trades
    sorted_pnls = sorted(slot_pnls)
    tail_n = max(1, n // 10)
    tail_avg = np.mean(sorted_pnls[:tail_n])
    tail_sum = sum(sorted_pnls[:tail_n])

    # 5. Loss severity: average loss PnL%
    loss_pnls = [p for p in slot_pnls if p <= 0]
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    worst_single = min(slot_pnls) if slot_pnls else 0

    # 6. Leverage stats
    avg_lev = np.mean(leverages)
    lev_std = np.std(leverages)
    loss_leverages = [t.get('leverage', MAX_LEVERAGE) for t in sorted_t if t['pnl_slot'] <= 0]
    avg_loss_leverage = np.mean(loss_leverages) if loss_leverages else 0

    # 7. Recovery factor: total PnL / max DD
    recovery_factor = total_pnl / max_dd if max_dd > 0 else 0

    # 8. Calmar ratio (annualized PnL / max DD)
    n_days = len(equity_curve) / BARS_PER_DAY
    annual_pnl = total_pnl * (365 / n_days) if n_days > 0 else 0
    calmar = annual_pnl / max_dd if max_dd > 0 else 0

    # 9. Ulcer Index (RMS of DD)
    ulcer = np.sqrt(np.mean(np.array(dd_values) ** 2)) if dd_values else 0

    return {
        'trades': n, 'wr': round(wr, 1),
        'total_pnl': round(total_pnl, 2),
        'max_dd': round(max_dd, 2),
        'pnl_mdd': round(total_pnl / max_dd, 2) if max_dd > 0 else 0,
        'max_dd_duration_bars': max_dd_duration,
        'max_dd_duration_hours': round(max_dd_duration * 5 / 60, 1),
        'worst_daily_pnl': round(worst_daily, 2),
        'daily_pnl_std': round(daily_std, 3),
        'daily_sharpe': round(daily_sharpe, 3),
        'max_consec_losses': max_consec_loss,
        'worst_streak_pnl': round(worst_streak_pnl, 2),
        'tail_10pct_avg': round(tail_avg, 2),
        'tail_10pct_sum': round(tail_sum, 2),
        'avg_loss_pnl': round(avg_loss, 2),
        'worst_single_trade': round(worst_single, 2),
        'avg_leverage': round(avg_lev, 2),
        'leverage_std': round(lev_std, 2),
        'avg_loss_leverage': round(avg_loss_leverage, 2),
        'recovery_factor': round(recovery_factor, 2),
        'calmar_ratio': round(calmar, 1),
        'ulcer_index': round(ulcer, 3),
    }


def compute_leverage_during_dd(leverage_at_entry, threshold_dd=1.0):
    """Analyze leverage behavior when DD exceeds threshold."""
    if not leverage_at_entry:
        return {}

    stress_entries = [(b, lev, eq, dd) for b, lev, eq, dd in leverage_at_entry
                      if dd >= threshold_dd]
    calm_entries = [(b, lev, eq, dd) for b, lev, eq, dd in leverage_at_entry
                    if dd < threshold_dd]

    stress_avg_lev = np.mean([x[1] for x in stress_entries]) if stress_entries else 0
    calm_avg_lev = np.mean([x[1] for x in calm_entries]) if calm_entries else 0

    return {
        'stress_entries': len(stress_entries),
        'calm_entries': len(calm_entries),
        'stress_avg_leverage': round(stress_avg_lev, 2),
        'calm_avg_leverage': round(calm_avg_lev, 2),
        'leverage_reduction_in_stress': round(calm_avg_lev - stress_avg_lev, 2),
        'reduction_pct': round(
            (calm_avg_lev - stress_avg_lev) / calm_avg_lev * 100, 1
        ) if calm_avg_lev > 0 else 0,
    }


# ============================================================
# WF with risk metrics
# ============================================================

def run_wf_risk(signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                leverage_fn, window_size, expected_wr, ref_edge, n_folds=3):
    """WF returning risk metrics per fold."""
    total = neutral_end - neutral_start
    seg_size = total // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        oos_start = neutral_start + seg_size * (fold + 1)
        oos_end = neutral_start + seg_size * (fold + 2) if fold < n_folds - 1 else neutral_end

        trades, eq_curve, lev_entries = portfolio_detailed(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            leverage_fn, window_size, expected_wr, ref_edge)

        metrics = compute_risk_metrics(trades, eq_curve)
        metrics['fold'] = fold + 1
        metrics['oos_bars'] = int(oos_end - oos_start)
        results.append(metrics)

    return results


# ============================================================
# Monte Carlo robustness
# ============================================================

def mc_trade_shuffle(trades, n_sims=1000):
    """Shuffle trade order, recalculate equity curves → distribution of MDD."""
    if not trades:
        return {}

    pnl_list = [t['pnl_portfolio'] for t in trades]
    rng = np.random.RandomState(42)

    mdd_dist = []
    final_pnl_dist = []
    for _ in range(n_sims):
        shuffled = rng.permutation(pnl_list)
        eq = 100.0
        peak = 100.0
        mdd = 0.0
        for p in shuffled:
            eq += p
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            if dd > mdd:
                mdd = dd
        mdd_dist.append(mdd)
        final_pnl_dist.append(eq - 100.0)

    return {
        'mdd_mean': round(np.mean(mdd_dist), 2),
        'mdd_median': round(np.median(mdd_dist), 2),
        'mdd_95pct': round(np.percentile(mdd_dist, 95), 2),
        'mdd_99pct': round(np.percentile(mdd_dist, 99), 2),
        'mdd_max': round(max(mdd_dist), 2),
        'pnl_5pct': round(np.percentile(final_pnl_dist, 5), 2),
        'pnl_median': round(np.median(final_pnl_dist), 2),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("ADAPTIVE LEVERAGE RISK-FOCUSED STUDY")
    print("=" * 70)
    start_time = datetime.now()

    # Load data
    print("\n[1] Loading data...")
    df = load_and_classify(DATA_FILE)
    opens, highs, lows, closes = (df['open'].values, df['high'].values,
                                   df['low'].values, df['close'].values)
    n_bars = len(df)
    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)
    neutral_start, neutral_end = find_neutral_window(closes, tol_pct=1.0)
    print(f"  Neutral window: bar {neutral_start}~{neutral_end} "
          f"({neutral_end - neutral_start} bars)")

    # Load patterns
    print("\n[2] Loading patterns...")
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pats_raw = pat_data['patterns']
    tpsl = pat_data.get('patterns_tpsl', {})
    npos = pat_data.get('npos_portfolio', {})
    is_stats = npos.get('is_stats', {})
    expected_wr = is_stats.get('wr', 73.2) / 100
    ref_edge = is_stats.get('pnl_per_trade', 0.126) / 100
    print(f"  expected_wr={expected_wr:.3f}, ref_edge={ref_edge:.5f}")

    pat_lookup = {}
    for p in pats_raw.get('long', []):
        ts = tpsl.get(p, [2.0, 3.0])
        pat_lookup[p] = {'direction': 'LONG', 'tp': ts[0], 'sl': ts[1]}
    for p in pats_raw.get('short', []):
        ts = tpsl.get(p, [2.0, 3.0])
        pat_lookup[p] = {'direction': 'SHORT', 'tp': ts[0], 'sl': ts[1]}
    print(f"  {len(pat_lookup)} patterns")

    # Build signals
    rctypes = df['rctype'].values
    signal_tuples = []
    for i in range(2, n_bars):
        tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if tri in pat_lookup:
            p = pat_lookup[tri]
            signal_tuples.append((i, tri, p['direction'], p['tp'], p['sl']))
    print(f"  {len(signal_tuples)} signals")

    # ============================================================
    # Define candidates
    # ============================================================

    def fn_fixed_3x(ctx): return 3.0
    def fn_fixed_1x(ctx): return 1.0

    def fn_step_tiers(ctx):
        return get_leverage_step_tiers(ctx['rolling_wr'])

    def fn_wr_conf(ctx):
        if ctx['n_trades'] < 3: return MIN_LEVERAGE
        return get_leverage_wr_confidence(
            ctx['rolling_wr_frac'], ctx['rolling_edge_frac'],
            ctx['expected_wr'], ctx['ref_edge'])

    def fn_combined(ctx):
        if ctx['n_trades'] < 3: return MIN_LEVERAGE
        return get_leverage_combined(
            ctx['rolling_wr_frac'], ctx['rolling_edge_frac'],
            ctx['expected_wr'], ctx['ref_edge'],
            ctx['tp_pct'] / 100, ctx['sl_pct'] / 100)

    def fn_exp_decay(ctx):
        if ctx['n_trades'] < 3: return MIN_LEVERAGE
        return get_leverage_exp_decay(ctx['rolling_wr_frac'], ctx['expected_wr'])

    candidates = [
        ('Fixed_3x', fn_fixed_3x, 15),
        ('Fixed_1x', fn_fixed_1x, 15),
        ('H1_StepTiers_w10', fn_step_tiers, 10),
        ('H5_WRConf_w8', fn_wr_conf, 8),
        ('H5_WRConf_w12', fn_wr_conf, 12),
        ('H8_Combined_w10', fn_combined, 10),
        ('H9_ExpDecay_w10', fn_exp_decay, 10),
    ]

    # ============================================================
    # Phase 1: Extended IS risk metrics
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Extended IS Risk Metrics (Neutral Window)")
    print("=" * 70)

    is_results = {}
    is_trades_map = {}
    is_lev_map = {}

    for name, fn, ws in candidates:
        trades, eq_curve, lev_entries = portfolio_detailed(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            fn, ws, expected_wr, ref_edge)
        metrics = compute_risk_metrics(trades, eq_curve)
        is_results[name] = metrics
        is_trades_map[name] = trades
        is_lev_map[name] = lev_entries

    # Print comparison table
    print(f"\n{'Metric':<25}", end='')
    for name, _, _ in candidates:
        print(f" {name:>14}", end='')
    print()
    print("-" * (25 + 15 * len(candidates)))

    metrics_to_show = [
        ('Total PnL%', 'total_pnl'),
        ('Max DD%', 'max_dd'),
        ('PnL/MDD', 'pnl_mdd'),
        ('Max DD Duration (h)', 'max_dd_duration_hours'),
        ('Worst Daily PnL%', 'worst_daily_pnl'),
        ('Daily PnL Std', 'daily_pnl_std'),
        ('Daily Sharpe', 'daily_sharpe'),
        ('Max Consec Losses', 'max_consec_losses'),
        ('Worst Streak PnL%', 'worst_streak_pnl'),
        ('Tail 10% Avg PnL%', 'tail_10pct_avg'),
        ('Worst Single Trade%', 'worst_single_trade'),
        ('Avg Leverage', 'avg_leverage'),
        ('Leverage Std', 'leverage_std'),
        ('Avg Loss Leverage', 'avg_loss_leverage'),
        ('Calmar Ratio', 'calmar_ratio'),
        ('Ulcer Index', 'ulcer_index'),
    ]

    for label, key in metrics_to_show:
        print(f"{label:<25}", end='')
        for name, _, _ in candidates:
            val = is_results[name].get(key, 0)
            if isinstance(val, float):
                print(f" {val:>14.2f}", end='')
            else:
                print(f" {val:>14}", end='')
        print()

    # ============================================================
    # Phase 2: Leverage Dynamics During Stress
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Leverage Dynamics During Drawdowns")
    print("=" * 70)

    for dd_threshold in [0.5, 1.0, 2.0]:
        print(f"\n  DD threshold ≥ {dd_threshold}%:")
        print(f"  {'Scenario':<25} {'Calm Lev':>9} {'Stress Lev':>11} "
              f"{'Reduction':>10} {'Red%':>6} {'#Stress':>8}")
        print(f"  {'-' * 70}")

        for name, _, _ in candidates:
            dd_stats = compute_leverage_during_dd(is_lev_map[name], dd_threshold)
            print(f"  {name:<25} {dd_stats.get('calm_avg_leverage', 0):>9.2f} "
                  f"{dd_stats.get('stress_avg_leverage', 0):>11.2f} "
                  f"{dd_stats.get('leverage_reduction_in_stress', 0):>+10.2f} "
                  f"{dd_stats.get('reduction_pct', 0):>5.1f}% "
                  f"{dd_stats.get('stress_entries', 0):>8}")

    # ============================================================
    # Phase 3: OOS Fold Stability (WF risk metrics)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: WF OOS Fold Stability")
    print("=" * 70)

    wf_results = {}
    for name, fn, ws in candidates:
        folds = run_wf_risk(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            fn, ws, expected_wr, ref_edge)
        wf_results[name] = folds

    print(f"\n  {'Scenario':<25} {'F1 PnL':>7} {'F2 PnL':>7} {'F3 PnL':>7} "
          f"{'Min':>7} {'Std':>7} {'F1 MDD':>7} {'F2 MDD':>7} {'F3 MDD':>7} "
          f"{'MaxMDD':>7}")
    print(f"  {'-' * 93}")

    fold_summary = {}
    for name, _, _ in candidates:
        folds = wf_results[name]
        pnls = [f['total_pnl'] for f in folds]
        mdds = [f['max_dd'] for f in folds]
        min_pnl = min(pnls)
        pnl_std = np.std(pnls)
        max_mdd = max(mdds)

        print(f"  {name:<25} {pnls[0]:>+7.1f} {pnls[1]:>+7.1f} {pnls[2]:>+7.1f} "
              f"{min_pnl:>+7.1f} {pnl_std:>7.1f} {mdds[0]:>7.1f} {mdds[1]:>7.1f} "
              f"{mdds[2]:>7.1f} {max_mdd:>7.1f}")

        fold_summary[name] = {
            'fold_pnls': [round(p, 2) for p in pnls],
            'fold_mdds': [round(m, 2) for m in mdds],
            'min_fold_pnl': round(min_pnl, 2),
            'pnl_std': round(pnl_std, 2),
            'max_fold_mdd': round(max_mdd, 2),
            'all_pass': all(p > 0 for p in pnls),
        }

    # Consistency score: min_fold / avg_fold (higher = more consistent)
    print(f"\n  Consistency Score (min_fold / avg_fold, higher = better):")
    for name, _, _ in candidates:
        pnls = [f['total_pnl'] for f in wf_results[name]]
        avg = np.mean(pnls)
        consistency = min(pnls) / avg if avg > 0 else 0
        print(f"    {name:<25} {consistency:.3f}")

    # ============================================================
    # Phase 4: Monte Carlo Robustness
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Monte Carlo Trade-Order Robustness (1000 sims)")
    print("=" * 70)

    mc_results = {}
    print(f"\n  {'Scenario':<25} {'Actual MDD':>10} {'MC Mean':>8} {'MC Median':>10} "
          f"{'MC 95%':>7} {'MC 99%':>7} {'MC Max':>7} {'PnL 5%':>7}")
    print(f"  {'-' * 82}")

    for name, _, _ in candidates:
        mc = mc_trade_shuffle(is_trades_map[name], n_sims=1000)
        mc_results[name] = mc
        actual_mdd = is_results[name]['max_dd']
        print(f"  {name:<25} {actual_mdd:>10.2f} {mc['mdd_mean']:>8.2f} "
              f"{mc['mdd_median']:>10.2f} {mc['mdd_95pct']:>7.2f} "
              f"{mc['mdd_99pct']:>7.2f} {mc['mdd_max']:>7.2f} "
              f"{mc['pnl_5pct']:>7.1f}")

    # ============================================================
    # Phase 5: Stress Period Analysis (worst 30-day windows)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Worst 30-Day Window Analysis")
    print("=" * 70)

    window_days = 30
    window_bars = window_days * BARS_PER_DAY

    stress_results = {}
    for name, fn, ws in candidates:
        trades, eq_curve, _ = portfolio_detailed(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            fn, ws, expected_wr, ref_edge)

        if not eq_curve:
            stress_results[name] = {'worst_30d_pnl': 0, 'worst_30d_start': 0}
            continue

        eq_dict = {e[0]: e[1] for e in eq_curve}
        bars = sorted(eq_dict.keys())

        worst_pnl = 0
        worst_start = bars[0]
        for i in range(len(bars)):
            start_bar_w = bars[i]
            end_bar_w = start_bar_w + window_bars
            # Find closest bar to end
            start_eq = eq_dict[start_bar_w]
            end_eq = None
            for b in bars[i:]:
                if b >= end_bar_w:
                    end_eq = eq_dict[b]
                    break
            if end_eq is None and bars[-1] > start_bar_w:
                end_eq = eq_dict[bars[-1]]
            if end_eq is not None and start_eq > 0:
                w_pnl = (end_eq - start_eq) / start_eq * 100
                if w_pnl < worst_pnl:
                    worst_pnl = w_pnl
                    worst_start = start_bar_w

        stress_results[name] = {
            'worst_30d_pnl': round(worst_pnl, 2),
            'worst_30d_start_bar': int(worst_start),
        }

    print(f"\n  {'Scenario':<25} {'Worst 30d PnL%':>15} {'Start Bar':>10}")
    print(f"  {'-' * 52}")
    for name, _, _ in candidates:
        sr = stress_results[name]
        print(f"  {name:<25} {sr['worst_30d_pnl']:>+15.2f} {sr['worst_30d_start_bar']:>10}")

    # ============================================================
    # Phase 6: Verdict
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 6: Risk-Focused Verdict")
    print("=" * 70)

    baseline = is_results['Fixed_3x']
    h1 = is_results['H1_StepTiers_w10']

    print("\n  Risk metric comparison vs Fixed_3x:")
    print(f"  {'Metric':<25} {'Fixed_3x':>10} {'H1_Step':>10} {'H5_w12':>10} "
          f"{'H5_w8':>10} {'H8_w10':>10} {'H9_w10':>10}")
    print(f"  {'-' * 77}")

    compare_keys = [
        ('Max DD%', 'max_dd'),
        ('Max DD Duration(h)', 'max_dd_duration_hours'),
        ('Worst Daily%', 'worst_daily_pnl'),
        ('Max Consec Losses', 'max_consec_losses'),
        ('Tail 10% Avg', 'tail_10pct_avg'),
        ('Worst Single Trade', 'worst_single_trade'),
        ('Avg Loss Leverage', 'avg_loss_leverage'),
        ('Calmar', 'calmar_ratio'),
        ('Ulcer Index', 'ulcer_index'),
        ('MC MDD 95%', None),
    ]

    compare_scenarios = ['Fixed_3x', 'H1_StepTiers_w10', 'H5_WRConf_w12',
                         'H5_WRConf_w8', 'H8_Combined_w10', 'H9_ExpDecay_w10']

    for label, key in compare_keys:
        print(f"  {label:<25}", end='')
        for sn in compare_scenarios:
            if key is None:
                val = mc_results.get(sn, {}).get('mdd_95pct', 0)
            else:
                val = is_results.get(sn, {}).get(key, 0)
            print(f" {val:>10.2f}", end='')
        print()

    # Score card: count metrics where each beats baseline
    print("\n  Score Card — metrics BETTER than Fixed_3x:")
    risk_metrics = [
        ('max_dd', 'lower'),
        ('max_dd_duration_hours', 'lower'),
        ('worst_daily_pnl', 'higher'),  # less negative is better
        ('max_consec_losses', 'lower'),
        ('tail_10pct_avg', 'higher'),  # less negative is better
        ('worst_single_trade', 'higher'),  # less negative is better
        ('ulcer_index', 'lower'),
    ]

    for sn in compare_scenarios[1:]:
        score = 0
        for metric, direction in risk_metrics:
            baseline_val = is_results['Fixed_3x'].get(metric, 0)
            candidate_val = is_results[sn].get(metric, 0)
            if direction == 'lower' and candidate_val < baseline_val:
                score += 1
            elif direction == 'higher' and candidate_val > baseline_val:
                score += 1
        # Add MC robustness
        if mc_results.get(sn, {}).get('mdd_95pct', 999) < mc_results.get('Fixed_3x', {}).get('mdd_95pct', 0):
            score += 1
        # Add OOS min fold
        sn_min = fold_summary.get(sn, {}).get('min_fold_pnl', 0)
        bl_min = fold_summary.get('Fixed_3x', {}).get('min_fold_pnl', 0)
        if sn_min > bl_min:
            score += 1

        total_metrics = len(risk_metrics) + 2  # +MC +OOS_min
        print(f"    {sn:<25} {score}/{total_metrics} metrics superior")

    # Final recommendation
    print("\n  " + "=" * 50)

    # Find best risk-adjusted candidate
    best_risk = None
    best_risk_score = -1
    for sn in compare_scenarios[1:]:
        score = 0
        for metric, direction in risk_metrics:
            bv = is_results['Fixed_3x'].get(metric, 0)
            cv = is_results[sn].get(metric, 0)
            if direction == 'lower' and cv < bv: score += 1
            elif direction == 'higher' and cv > bv: score += 1
        if mc_results.get(sn, {}).get('mdd_95pct', 999) < mc_results.get('Fixed_3x', {}).get('mdd_95pct', 0):
            score += 1
        if fold_summary.get(sn, {}).get('min_fold_pnl', 0) > fold_summary.get('Fixed_3x', {}).get('min_fold_pnl', 0):
            score += 1

        if score > best_risk_score:
            best_risk_score = score
            best_risk = sn

    total_metrics = len(risk_metrics) + 2
    if best_risk:
        br = is_results[best_risk]
        pnl_loss = br['total_pnl'] - baseline['total_pnl']
        mdd_gain = baseline['max_dd'] - br['max_dd']

        print(f"\n  Best risk-adjusted: {best_risk} ({best_risk_score}/{total_metrics} metrics)")
        print(f"  PnL trade-off: {pnl_loss:+.1f}% (IS)")
        print(f"  MDD improvement: {mdd_gain:+.2f}%")
        print(f"  OOS min fold: {fold_summary[best_risk]['min_fold_pnl']:+.1f}% "
              f"(vs baseline {fold_summary['Fixed_3x']['min_fold_pnl']:+.1f}%)")

        if best_risk_score >= total_metrics * 0.6:
            # Extract method and window
            if '_w' in best_risk:
                parts = best_risk.rsplit('_w', 1)
                method_label = parts[0]
                rec_window = int(parts[1])
            else:
                method_label = best_risk
                rec_window = 10

            # Map to config method names
            method_config_map = {
                'H5_WRConf': 'wr_confidence',
                'H1_StepTiers': 'step_tiers',
                'H8_Combined': 'combined',
                'H9_ExpDecay': 'exp_decay',
            }
            config_method = method_config_map.get(method_label, method_label.lower())

            verdict = "GO (risk-superior)"
            print(f"\n  VERDICT: {verdict}")
            print(f"  ≥60% risk metrics superior to baseline")
            print(f"  Recommended config:")
            print(f"    adaptive_leverage.enabled: true")
            print(f"    adaptive_leverage.method: \"{config_method}\"")
            print(f"    adaptive_leverage.window: {rec_window}")
        else:
            verdict = "MARGINAL"
            print(f"\n  VERDICT: {verdict} — Risk improvement exists but <60% metrics")
    else:
        verdict = "STOP"
        print(f"\n  VERDICT: {verdict}")

    # Save
    elapsed = (datetime.now() - start_time).total_seconds()
    output = {
        'study': 'adaptive_leverage_risk',
        'date': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'expected_wr': expected_wr,
        'ref_edge': ref_edge,
        'phase1_is_risk': {k: v for k, v in is_results.items()},
        'phase3_wf_folds': fold_summary,
        'phase4_mc': mc_results,
        'phase5_stress': stress_results,
        'verdict': verdict,
        'best_risk_scenario': best_risk,
        'best_risk_score': f"{best_risk_score}/{total_metrics}",
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
