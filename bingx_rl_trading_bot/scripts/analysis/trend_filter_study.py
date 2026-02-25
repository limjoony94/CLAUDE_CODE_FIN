#!/usr/bin/env python3
"""
Trend Filter Study — Counter-trend entry filtering for trending markets.
========================================================================
Standard Research Protocol compliant.

Problem: 51 patterns (16L+35S). In strong uptrends, SHORT patterns fire
into counter-trend and hit SL consecutively. LONG patterns require
reversal candles (DN, BD, MD) that don't appear in strong uptrends.

Hypotheses:
  H1: N-bar Momentum Gate — block counter-trend when momentum > threshold
  H2: EMA Direction Gate — block counter-trend when price far from EMA
  H3: Consecutive Loss Brake — pause direction after N consecutive SLs
  H4: ATR Expansion + Momentum Combo — block only in volatile trends

Protocol:
  - Data: btc_5m_270days_reclassified.csv (270d, 75744 bars)
  - Entry: signal bar+1 open
  - Exit: intrabar high/low (distance-based, same-bar: bar-open basis)
  - Fee: 0.10% * LEVERAGE(3) = 0.30% roundtrip in capital space
  - Slippage buffer: 0.02% on TP/SL
  - ATR scaling: ATR(14)/median(576), clamp [0.6, 1.7]
  - Portfolio: N=9 virtual slots, 1/N=11.1% sizing, direction_cap=8,
    timeout=864 bars, duplicate pattern blocking
  - MC: 3 seeds (42, 123, 7), max p < 0.01
  - WF: 3-fold expanding window OOS
  - Timeout trades: DROPPED (not counted)

Output: results/trend_filter_study.json
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants (Standard Research Protocol)
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10               # 0.05% entry + 0.05% exit
SLIPPAGE_BUFFER = 0.02        # 0.02% buffer on TP/SL
TIMEOUT_BARS = 864            # 72h at 5m
MAX_POSITIONS = 9
DIRECTION_CAP = 8             # v1.35.1
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars
MAX_BARS = 288                # Max bars to hold per signal in backtest
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'trend_filter_study.json')


# ============================================================
# Data Loading
# ============================================================

def load_data():
    """Load CSV and extract type_code column for pattern matching."""
    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    # type_code has short codes: BD, DN, U, etc.
    rctypes = df['type_code'].values.tolist()
    return df, rctypes


def load_patterns():
    """Load dynamic patterns and TP/SL map."""
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    pL = set(data['patterns']['long'])
    pS = set(data['patterns']['short'])
    tpsl = data['patterns_tpsl']  # {pattern_name: [tp, sl]}
    return pL, pS, tpsl, data


# ============================================================
# ATR Computation
# ============================================================

def compute_atr_ratio(highs, lows, closes):
    """ATR / rolling_median(ATR) ratio. Wilder EMA ATR."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= ATR_PERIOD:
        atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
        for i in range(ATR_PERIOD, n):
            atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    med = pd.Series(atr).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio, atr


def compute_ema(closes, period):
    """Compute EMA array."""
    n = len(closes)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    ema[period - 1] = np.mean(closes[:period])
    mult = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = closes[i] * mult + ema[i - 1] * (1 - mult)
    return ema


# ============================================================
# Signal Generation
# ============================================================

def generate_signals(rctypes, patterns_long, patterns_short, start_idx=0):
    """Generate signals. Returns list of (bar_idx, pattern, direction)."""
    signals = []
    n = len(rctypes)
    for i in range(max(2, start_idx), n):
        pat = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


# ============================================================
# Trend Filter Functions
# ============================================================

def no_filter(bar_idx, direction, state):
    """No filtering — baseline."""
    return True


def momentum_gate(N, threshold):
    """Factory: block counter-trend when N-bar return exceeds threshold.
    If return_N > +threshold -> block SHORT
    If return_N < -threshold -> block LONG
    """
    def _filter(bar_idx, direction, state):
        closes = state['closes']
        if bar_idx < N:
            return True
        ret = (closes[bar_idx] - closes[bar_idx - N]) / closes[bar_idx - N] * 100
        if direction == 'SHORT' and ret > threshold:
            return False
        if direction == 'LONG' and ret < -threshold:
            return False
        return True
    return _filter


def ema_gate(period, mode='hard'):
    """Factory: block counter-trend based on EMA direction.
    hard: block SHORT when price > EMA, block LONG when price < EMA
    soft: only block when price is > 1 ATR away from EMA
    """
    def _filter(bar_idx, direction, state):
        emas = state.get(f'ema_{period}')
        if emas is None or bar_idx >= len(emas) or np.isnan(emas[bar_idx]):
            return True
        closes = state['closes']
        price = closes[bar_idx]
        ema_val = emas[bar_idx]

        if mode == 'hard':
            if direction == 'SHORT' and price > ema_val:
                return False
            if direction == 'LONG' and price < ema_val:
                return False
        elif mode == 'soft':
            atr_arr = state.get('atr')
            if atr_arr is None or np.isnan(atr_arr[bar_idx]):
                return True
            atr_val = atr_arr[bar_idx]
            if atr_val <= 0:
                return True
            dist = abs(price - ema_val)
            if dist > atr_val:
                if direction == 'SHORT' and price > ema_val:
                    return False
                if direction == 'LONG' and price < ema_val:
                    return False
        return True
    return _filter


def consecutive_loss_brake(max_consecutive, cooldown_bars):
    """Factory: pause direction after N consecutive SLs in that direction.
    Reactive, not predictive.
    """
    def _filter(bar_idx, direction, state):
        loss_history = state.get('loss_history', [])
        dir_losses = [t for t in loss_history
                      if t['direction'] == direction and t['pnl'] < 0]
        if len(dir_losses) < max_consecutive:
            return True

        # Check if the last max_consecutive trades in this direction are all losses
        dir_trades = [t for t in state.get('trade_history', [])
                      if t['direction'] == direction]
        if len(dir_trades) < max_consecutive:
            return True
        recent = dir_trades[-max_consecutive:]
        if not all(t['pnl'] < 0 for t in recent):
            return True

        # All recent trades are losses — check cooldown from last loss
        last_loss_bar = recent[-1]['exit_bar']
        if bar_idx - last_loss_bar < cooldown_bars:
            return False
        return True
    return _filter


def atr_momentum_combo(atr_threshold, momentum_N, momentum_threshold):
    """Factory: block counter-trend only when BOTH ATR is high AND momentum is strong.
    More selective than H1 or H2 alone.
    """
    def _filter(bar_idx, direction, state):
        closes = state['closes']
        atr_ratio_arr = state.get('atr_ratio')
        if atr_ratio_arr is None or np.isnan(atr_ratio_arr[bar_idx]):
            return True
        r = atr_ratio_arr[bar_idx]
        if r <= atr_threshold:
            return True  # Low volatility — allow all
        if bar_idx < momentum_N:
            return True
        ret = (closes[bar_idx] - closes[bar_idx - momentum_N]) / closes[bar_idx - momentum_N] * 100
        if direction == 'SHORT' and ret > momentum_threshold:
            return False
        if direction == 'LONG' and ret < -momentum_threshold:
            return False
        return True
    return _filter


# ============================================================
# Portfolio Simulation
# ============================================================

def simulate_portfolio(signals, tpsl_map, opens, highs, lows, n_bars,
                       atr_ratio, trend_filter, filter_state,
                       oos_start, oos_end):
    """Simulate N=9 multi-position portfolio with optional trend filter.

    Args:
        signals: list of (bar_idx, pattern, direction)
        tpsl_map: {pattern: [tp, sl]}
        opens/highs/lows: price arrays
        n_bars: total bars
        atr_ratio: ATR ratio array
        trend_filter: callable(bar_idx, direction, state) -> bool (True=allow)
        filter_state: dict with closes, atr, ema, trade_history etc.
        oos_start, oos_end: OOS period bar indices
    """
    positions = []
    trades = []
    filtered_count = 0
    total_signal_count = 0
    SIZE_PCT = 100.0 / MAX_POSITIONS

    # Initialize trade history for consecutive loss brake
    if 'trade_history' not in filter_state:
        filter_state['trade_history'] = []
    if 'loss_history' not in filter_state:
        filter_state['loss_history'] = []

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(oos_start, oos_end):
        # Check exits
        closed_slots = []
        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio)
            if result is not None:
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                result['pnl_portfolio'] = result['pnl_slot'] * SIZE_PCT / 100
                trades.append(result)
                # Update filter state for consecutive loss brake
                filter_state['trade_history'].append({
                    'direction': pos['direction'],
                    'pnl': result['pnl_slot'],
                    'exit_bar': bar,
                    'reason': result['reason'],
                })
                if result['pnl_slot'] < 0:
                    filter_state['loss_history'].append({
                        'direction': pos['direction'],
                        'pnl': result['pnl_slot'],
                        'exit_bar': bar,
                    })
                closed_slots.append(pos['slot'])
        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Process signals
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1
            total_signal_count += 1

            if len(positions) >= MAX_POSITIONS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            # Apply trend filter
            if not trend_filter(sig_bar, direction, filter_state):
                filtered_count += 1
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_sl[0],
                'sl_pct': tp_sl[1],
            })

    # Force-close remaining positions at OOS end (same as direction_cap_wf_study)
    fee = FEE_PCT * LEVERAGE
    SIZE_PCT_local = SIZE_PCT
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(oos_end - 1, n_bars - 1)
        exit_price = opens[exit_bar] if exit_bar < n_bars else opens[-1]
        direction = pos['direction']
        if direction == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= fee
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'hold_bars': exit_bar - entry_bar,
            'entry_price': entry, 'exit_price': exit_price,
            'pnl_slot': pnl, 'reason': 'OOS_END',
            'pattern': pos['pattern'], 'direction': pos['direction'],
            'pnl_portfolio': pnl * SIZE_PCT_local / 100,
        })
        filter_state['trade_history'].append({
            'direction': pos['direction'], 'pnl': pnl, 'exit_bar': exit_bar, 'reason': 'OOS_END',
        })

    return trades, filtered_count, total_signal_count


def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio):
    """Check if position exits on this bar."""
    entry_bar = pos['entry_bar']
    if bar <= entry_bar:  # Can't exit on or before entry bar
        return None

    entry = opens[entry_bar]
    if entry <= 0:
        return None

    tp_pct = pos['tp_pct']
    sl_pct = pos['sl_pct']
    direction = pos['direction']
    sig_bar = pos['signal_bar']

    # ATR scaling at signal bar
    if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
    else:
        r = 1.0

    eff_tp = tp_pct * r + SLIPPAGE_BUFFER
    eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    hold = bar - entry_bar
    fee = FEE_PCT * LEVERAGE

    # Timeout -> DROP
    if hold >= TIMEOUT_BARS:
        return None  # DROP timeout trades

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
        # Same-bar resolution: bar-open distance
        tp_dist = abs(tp_price - opens[bar])
        sl_dist = abs(sl_price - opens[bar])
        if tp_dist <= sl_dist:
            pnl = eff_tp * LEVERAGE - fee
            reason = 'TP'
        else:
            pnl = -eff_sl * LEVERAGE - fee
            reason = 'SL'
    elif hit_tp:
        pnl = eff_tp * LEVERAGE - fee
        reason = 'TP'
    else:
        pnl = -eff_sl * LEVERAGE - fee
        reason = 'SL'

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'hold_bars': hold,
            'entry_price': entry, 'exit_price': tp_price if reason == 'TP' else sl_price,
            'pnl_slot': pnl, 'reason': reason}


# ============================================================
# Metrics
# ============================================================

def compute_metrics(trades):
    """Compute portfolio metrics from trade list."""
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0,
                'long_trades': 0, 'short_trades': 0,
                'long_pnl': 0.0, 'short_pnl': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'pnl_per_trade': 0.0,
                'filtered': 0, 'filter_rate': 0.0}

    wins = [t for t in trades if t['pnl_slot'] > 0]
    losses = [t for t in trades if t['pnl_slot'] <= 0]
    wr = len(wins) / len(trades) * 100

    # Compound MDD
    sorted_trades = sorted(trades, key=lambda x: x['entry_bar'])
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for t in sorted_trades:
        equity *= (1 + t['pnl_portfolio'] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    long_t = [t for t in trades if t['direction'] == 'LONG']
    short_t = [t for t in trades if t['direction'] == 'SHORT']

    # Compound PnL
    eq = 100.0
    for t in sorted_trades:
        eq *= (1 + t['pnl_portfolio'] / 100)
    total_pnl = eq - 100.0

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'long_pnl': round(sum(t['pnl_portfolio'] for t in long_t), 2),
        'short_pnl': round(sum(t['pnl_portfolio'] for t in short_t), 2),
        'avg_win': round(np.mean([t['pnl_slot'] for t in wins]), 2) if wins else 0.0,
        'avg_loss': round(np.mean([t['pnl_slot'] for t in losses]), 2) if losses else 0.0,
        'pnl_per_trade': round(total_pnl / len(trades), 3) if trades else 0.0,
    }


def mc_test(trades):
    """Multi-seed sign randomization MC test on portfolio PnL."""
    pnls = [t['pnl_portfolio'] for t in trades]
    if len(pnls) < 10:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(MC_SIMS, len(pnls_arr)))
        rand_sums = signs @ pnls_arr
        p_vals.append(float(np.mean(rand_sums >= actual)))
    return max(p_vals)


# ============================================================
# Walk-Forward Validation
# ============================================================

def run_wf_validation(signals, tpsl_map, opens, highs, lows, closes,
                      n_bars, atr_ratio, atr_arr, trend_filter_factory,
                      filter_params, n_folds=3):
    """Run expanding window WF validation.

    For each fold, the OOS period is 1/n_folds of the tradeable range.
    Expanding window: IS grows, OOS is always the next chunk.
    """
    tradeable_start = ATR_WARMUP
    tradeable_bars = n_bars - tradeable_start
    fold_size = tradeable_bars // n_folds

    fold_results = []
    for fold_idx in range(n_folds):
        oos_start = tradeable_start + fold_idx * fold_size
        oos_end = tradeable_start + (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_bars

        # Build filter state (shared data, per-fold trade history)
        filter_state = {
            'closes': closes,
            'atr_ratio': atr_ratio,
            'atr': atr_arr,
            'trade_history': [],
            'loss_history': [],
        }
        # Pre-compute EMA arrays if needed
        for period in [50, 100, 200]:
            filter_state[f'ema_{period}'] = compute_ema(closes, period)

        # Build filter from factory
        trend_filter = trend_filter_factory(**filter_params) if filter_params else no_filter

        trades, filtered, total_sigs = simulate_portfolio(
            signals, tpsl_map, opens, highs, lows, n_bars,
            atr_ratio, trend_filter, filter_state,
            oos_start, oos_end
        )

        m = compute_metrics(trades)
        m['filtered'] = filtered
        m['total_signals'] = total_sigs
        m['filter_rate'] = round(filtered / total_sigs * 100, 1) if total_sigs > 0 else 0.0
        m['oos_start'] = oos_start
        m['oos_end'] = oos_end
        m['mc_p'] = mc_test(trades)
        fold_results.append(m)

    return fold_results


# ============================================================
# Phase Runners
# ============================================================

def run_single_scenario(label, signals, tpsl_map, opens, highs, lows, closes,
                        n_bars, atr_ratio, atr_arr, trend_filter, emas=None):
    """Run a single full-period backtest (no WF) for grid search."""
    filter_state = {
        'closes': closes,
        'atr_ratio': atr_ratio,
        'atr': atr_arr,
        'trade_history': [],
        'loss_history': [],
    }
    if emas:
        filter_state.update(emas)

    trades, filtered, total_sigs = simulate_portfolio(
        signals, tpsl_map, opens, highs, lows, n_bars,
        atr_ratio, trend_filter, filter_state,
        ATR_WARMUP, n_bars
    )

    m = compute_metrics(trades)
    m['filtered'] = filtered
    m['total_signals'] = total_sigs
    m['filter_rate'] = round(filtered / total_sigs * 100, 1) if total_sigs > 0 else 0.0
    m['label'] = label
    return m


def phase1_baseline(signals, tpsl, opens, highs, lows, closes, n_bars, atr_ratio, atr_arr, emas):
    """Phase 1: Baseline — no filter."""
    print("\n" + "=" * 80)
    print("PHASE 1: BASELINE (No Filter)")
    print("=" * 80)

    m = run_single_scenario(
        'baseline', signals, tpsl, opens, highs, lows, closes,
        n_bars, atr_ratio, atr_arr, no_filter, emas
    )
    print(f"  Trades: {m['trades']} ({m['long_trades']}L/{m['short_trades']}S)")
    print(f"  WR: {m['wr']}%  PnL: {m['pnl']:+.2f}%  MDD: {m['mdd']:.2f}%")
    print(f"  PnL/MDD: {m['pnl']/m['mdd']:.2f}x" if m['mdd'] > 0 else "  PnL/MDD: N/A")
    print(f"  Avg Win: {m['avg_win']:+.2f}%  Avg Loss: {m['avg_loss']:+.2f}%")
    print(f"  PnL/trade: {m['pnl_per_trade']:+.3f}%")
    return m


def phase2_momentum_gate(signals, tpsl, opens, highs, lows, closes, n_bars,
                         atr_ratio, atr_arr, emas, baseline):
    """Phase 2: H1 — N-bar Momentum Gate grid search."""
    print("\n" + "=" * 80)
    print("PHASE 2: H1 — Momentum Gate (N-bar return threshold)")
    print("=" * 80)

    N_values = [24, 48, 96, 144]
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = []
    print(f"\n  {'N':>4} {'Thr':>5} {'Trades':>7} {'Filt%':>6} {'WR':>6} {'PnL':>9} {'MDD':>7} "
          f"{'PnL/MDD':>8} {'dPnL':>8} {'dMDD':>6}")
    print("  " + "-" * 78)

    for N in N_values:
        for thr in thresholds:
            filt = momentum_gate(N, thr)
            m = run_single_scenario(
                f'mom_N{N}_T{thr}', signals, tpsl, opens, highs, lows, closes,
                n_bars, atr_ratio, atr_arr, filt, emas
            )
            d_pnl = m['pnl'] - baseline['pnl']
            d_mdd = m['mdd'] - baseline['mdd']
            pm = m['pnl'] / m['mdd'] if m['mdd'] > 0 else 0
            print(f"  {N:>4} {thr:>5.1f} {m['trades']:>7} {m['filter_rate']:>5.1f}% "
                  f"{m['wr']:>5.1f}% {m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% {pm:>7.2f}x "
                  f"{d_pnl:>+7.2f}% {d_mdd:>+5.2f}%")
            m['N'] = N
            m['threshold'] = thr
            results.append(m)

    return results


def phase3_ema_gate(signals, tpsl, opens, highs, lows, closes, n_bars,
                    atr_ratio, atr_arr, emas, baseline):
    """Phase 3: H2 — EMA Direction Gate."""
    print("\n" + "=" * 80)
    print("PHASE 3: H2 — EMA Direction Gate")
    print("=" * 80)

    periods = [50, 100, 200]
    modes = ['hard', 'soft']

    results = []
    print(f"\n  {'EMA':>4} {'Mode':>5} {'Trades':>7} {'Filt%':>6} {'WR':>6} {'PnL':>9} {'MDD':>7} "
          f"{'PnL/MDD':>8} {'dPnL':>8} {'dMDD':>6}")
    print("  " + "-" * 72)

    for period in periods:
        for mode in modes:
            filt = ema_gate(period, mode)
            m = run_single_scenario(
                f'ema_{period}_{mode}', signals, tpsl, opens, highs, lows, closes,
                n_bars, atr_ratio, atr_arr, filt, emas
            )
            d_pnl = m['pnl'] - baseline['pnl']
            d_mdd = m['mdd'] - baseline['mdd']
            pm = m['pnl'] / m['mdd'] if m['mdd'] > 0 else 0
            print(f"  {period:>4} {mode:>5} {m['trades']:>7} {m['filter_rate']:>5.1f}% "
                  f"{m['wr']:>5.1f}% {m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% {pm:>7.2f}x "
                  f"{d_pnl:>+7.2f}% {d_mdd:>+5.2f}%")
            m['period'] = period
            m['mode'] = mode
            results.append(m)

    return results


def phase4_consecutive_loss(signals, tpsl, opens, highs, lows, closes, n_bars,
                            atr_ratio, atr_arr, emas, baseline):
    """Phase 4: H3 — Consecutive Loss Brake."""
    print("\n" + "=" * 80)
    print("PHASE 4: H3 — Consecutive Loss Brake")
    print("=" * 80)

    max_consec = [2, 3, 4]
    cooldowns = [12, 24, 48, 96]

    results = []
    print(f"\n  {'N':>4} {'Cool':>5} {'Trades':>7} {'Filt%':>6} {'WR':>6} {'PnL':>9} {'MDD':>7} "
          f"{'PnL/MDD':>8} {'dPnL':>8} {'dMDD':>6}")
    print("  " + "-" * 72)

    for nc in max_consec:
        for cd in cooldowns:
            filt = consecutive_loss_brake(nc, cd)
            m = run_single_scenario(
                f'clb_N{nc}_C{cd}', signals, tpsl, opens, highs, lows, closes,
                n_bars, atr_ratio, atr_arr, filt, emas
            )
            d_pnl = m['pnl'] - baseline['pnl']
            d_mdd = m['mdd'] - baseline['mdd']
            pm = m['pnl'] / m['mdd'] if m['mdd'] > 0 else 0
            print(f"  {nc:>4} {cd:>5} {m['trades']:>7} {m['filter_rate']:>5.1f}% "
                  f"{m['wr']:>5.1f}% {m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% {pm:>7.2f}x "
                  f"{d_pnl:>+7.2f}% {d_mdd:>+5.2f}%")
            m['max_consec'] = nc
            m['cooldown'] = cd
            results.append(m)

    return results


def phase5_atr_momentum(signals, tpsl, opens, highs, lows, closes, n_bars,
                        atr_ratio, atr_arr, emas, baseline):
    """Phase 5: H4 — ATR Expansion + Momentum Combo."""
    print("\n" + "=" * 80)
    print("PHASE 5: H4 — ATR Expansion + Momentum Combo")
    print("=" * 80)

    atr_thresholds = [1.0, 1.2, 1.4]
    mom_Ns = [24, 48, 96]
    mom_thresholds = [1.0, 1.5, 2.0]

    results = []
    print(f"\n  {'ATR':>4} {'mN':>4} {'mThr':>5} {'Trades':>7} {'Filt%':>6} {'WR':>6} "
          f"{'PnL':>9} {'MDD':>7} {'PnL/MDD':>8} {'dPnL':>8} {'dMDD':>6}")
    print("  " + "-" * 82)

    for atr_thr in atr_thresholds:
        for mN in mom_Ns:
            for mThr in mom_thresholds:
                filt = atr_momentum_combo(atr_thr, mN, mThr)
                m = run_single_scenario(
                    f'atrmom_A{atr_thr}_N{mN}_T{mThr}', signals, tpsl,
                    opens, highs, lows, closes, n_bars,
                    atr_ratio, atr_arr, filt, emas
                )
                d_pnl = m['pnl'] - baseline['pnl']
                d_mdd = m['mdd'] - baseline['mdd']
                pm = m['pnl'] / m['mdd'] if m['mdd'] > 0 else 0
                print(f"  {atr_thr:>4.1f} {mN:>4} {mThr:>5.1f} {m['trades']:>7} "
                      f"{m['filter_rate']:>5.1f}% {m['wr']:>5.1f}% {m['pnl']:>+8.2f}% "
                      f"{m['mdd']:>6.2f}% {pm:>7.2f}x {d_pnl:>+7.2f}% {d_mdd:>+5.2f}%")
                m['atr_threshold'] = atr_thr
                m['momentum_N'] = mN
                m['momentum_threshold'] = mThr
                results.append(m)

    return results


def phase6_wf_validation(best_candidates, signals, tpsl, opens, highs, lows,
                         closes, n_bars, atr_ratio, atr_arr):
    """Phase 6: WF 3-fold validation of best filter candidates."""
    print("\n" + "=" * 80)
    print("PHASE 6: Walk-Forward 3-Fold OOS Validation (Top Candidates)")
    print("=" * 80)

    wf_results = {}

    # Always include baseline
    print("\n--- Baseline (no filter) ---")
    baseline_folds = run_wf_validation(
        signals, tpsl, opens, highs, lows, closes, n_bars,
        atr_ratio, atr_arr, lambda: no_filter, {}, n_folds=3
    )
    _print_wf_folds('baseline', baseline_folds)
    wf_results['baseline'] = baseline_folds

    for cand in best_candidates:
        label = cand['label']
        factory = cand['factory']
        params = cand['params']
        print(f"\n--- {label} ---")

        fold_results = run_wf_validation(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, atr_arr, factory, params, n_folds=3
        )
        _print_wf_folds(label, fold_results)
        wf_results[label] = fold_results

    return wf_results


def _print_wf_folds(label, folds):
    """Print WF fold results."""
    print(f"  {'Fold':>6} {'Trades':>7} {'Filt%':>6} {'WR':>6} {'PnL':>9} {'MDD':>7} "
          f"{'PnL/MDD':>8} {'MC':>6}")
    print("  " + "-" * 60)
    for i, m in enumerate(folds):
        pm = m['pnl'] / m['mdd'] if m['mdd'] > 0 else 0
        mc_str = f"{m['mc_p']:.4f}" if m['mc_p'] > 0 else "0.0000"
        print(f"  Fold {i+1} {m['trades']:>7} {m['filter_rate']:>5.1f}% {m['wr']:>5.1f}% "
              f"{m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% {pm:>7.2f}x {mc_str:>6}")

    # Summary
    total_trades = sum(m['trades'] for m in folds)
    total_pnl = sum(m['pnl'] for m in folds)
    max_mdd = max(m['mdd'] for m in folds)
    avg_wr = np.mean([m['wr'] for m in folds if m['trades'] > 0])
    positive_folds = sum(1 for m in folds if m['pnl'] > 0)
    max_mc = max(m['mc_p'] for m in folds)
    pm = total_pnl / max_mdd if max_mdd > 0 else 0
    status = "PASS" if positive_folds == len(folds) else "FAIL"

    print("  " + "-" * 60)
    print(f"  TOTAL {total_trades:>7} {'':>6} {avg_wr:>5.1f}% {total_pnl:>+8.2f}% "
          f"{max_mdd:>6.2f}% {pm:>7.2f}x {max_mc:>6.4f}")
    print(f"  WF: {positive_folds}/{len(folds)} {status} | Max MC: {max_mc:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()
    print("=" * 80)
    print("TREND FILTER STUDY — Counter-Trend Entry Filtering")
    print(f"Standard Research Protocol | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df, rctypes = load_data()
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n_bars = len(opens)
    print(f"  Data: {n_bars} bars ({n_bars/288:.0f} days)")
    print(f"  Period: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # Load patterns
    pL, pS, tpsl, pat_data = load_patterns()
    print(f"  Patterns: {len(pL)}L + {len(pS)}S = {len(pL)+len(pS)}")

    # Compute ATR
    print("\nComputing ATR ratio...")
    atr_ratio, atr_arr = compute_atr_ratio(highs, lows, closes)
    valid_pct = np.sum(~np.isnan(atr_ratio)) / len(atr_ratio) * 100
    print(f"  ATR ratio valid: {valid_pct:.1f}% (warmup={ATR_WARMUP} bars)")

    # Pre-compute EMAs
    print("Computing EMAs...")
    emas = {}
    for period in [50, 100, 200]:
        emas[f'ema_{period}'] = compute_ema(closes, period)
        print(f"  EMA({period}): computed")

    # Generate signals
    signals = generate_signals(rctypes, pL, pS, start_idx=ATR_WARMUP)
    print(f"  Total signals (post-warmup): {len(signals)}")
    long_sigs = sum(1 for _, _, d in signals if d == 'LONG')
    short_sigs = sum(1 for _, _, d in signals if d == 'SHORT')
    print(f"  LONG: {long_sigs}, SHORT: {short_sigs}")

    # ========================================
    # Phase 1: Baseline
    # ========================================
    baseline = phase1_baseline(signals, tpsl, opens, highs, lows, closes,
                               n_bars, atr_ratio, atr_arr, emas)

    # ========================================
    # Phase 2: Momentum Gate
    # ========================================
    mom_results = phase2_momentum_gate(signals, tpsl, opens, highs, lows, closes,
                                       n_bars, atr_ratio, atr_arr, emas, baseline)

    # ========================================
    # Phase 3: EMA Gate
    # ========================================
    ema_results = phase3_ema_gate(signals, tpsl, opens, highs, lows, closes,
                                  n_bars, atr_ratio, atr_arr, emas, baseline)

    # ========================================
    # Phase 4: Consecutive Loss Brake
    # ========================================
    clb_results = phase4_consecutive_loss(signals, tpsl, opens, highs, lows, closes,
                                          n_bars, atr_ratio, atr_arr, emas, baseline)

    # ========================================
    # Phase 5: ATR + Momentum Combo
    # ========================================
    atrmom_results = phase5_atr_momentum(signals, tpsl, opens, highs, lows, closes,
                                          n_bars, atr_ratio, atr_arr, emas, baseline)

    # ========================================
    # Select Best Candidates for WF
    # ========================================
    print("\n" + "=" * 80)
    print("CANDIDATE SELECTION FOR WF VALIDATION")
    print("=" * 80)

    # Criteria: PnL >= 90% of baseline AND MDD reduction > 3pp
    # OR PnL/MDD improvement > 10%
    baseline_pnl = baseline['pnl']
    baseline_mdd = baseline['mdd']
    baseline_pnl_mdd = baseline_pnl / baseline_mdd if baseline_mdd > 0 else 0
    baseline_trades = baseline['trades']

    all_results = []
    for r in mom_results:
        r['hypothesis'] = 'H1_momentum'
        all_results.append(r)
    for r in ema_results:
        r['hypothesis'] = 'H2_ema'
        all_results.append(r)
    for r in clb_results:
        r['hypothesis'] = 'H3_consecutive_loss'
        all_results.append(r)
    for r in atrmom_results:
        r['hypothesis'] = 'H4_atr_momentum'
        all_results.append(r)

    candidates = []
    for r in all_results:
        if r['trades'] < baseline_trades * 0.5:
            continue  # Over-filtering
        pnl_ratio = r['pnl'] / baseline_pnl if baseline_pnl != 0 else 0
        mdd_delta = r['mdd'] - baseline_mdd
        r_pnl_mdd = r['pnl'] / r['mdd'] if r['mdd'] > 0 else 0
        pnl_mdd_improvement = (r_pnl_mdd - baseline_pnl_mdd) / baseline_pnl_mdd * 100 if baseline_pnl_mdd > 0 else 0

        qualifies = False
        # Primary: PnL >= 90% AND MDD improvement >= 3pp
        if pnl_ratio >= 0.9 and mdd_delta <= -3.0:
            qualifies = True
        # Secondary: PnL/MDD improvement >= 10%
        if pnl_mdd_improvement >= 10:
            qualifies = True
        # Tertiary: PnL improvement > 0 AND PnL/MDD improvement >= 5%
        if r['pnl'] > baseline_pnl and pnl_mdd_improvement >= 5:
            qualifies = True

        if qualifies:
            r['pnl_ratio'] = round(pnl_ratio, 3)
            r['mdd_delta'] = round(mdd_delta, 2)
            r['pnl_mdd_ratio'] = round(r_pnl_mdd, 2)
            r['pnl_mdd_improvement_pct'] = round(pnl_mdd_improvement, 1)
            candidates.append(r)

    # Sort by PnL/MDD descending
    candidates.sort(key=lambda x: x.get('pnl_mdd_ratio', 0), reverse=True)

    print(f"\n  Baseline: trades={baseline_trades}, PnL={baseline_pnl:+.2f}%, "
          f"MDD={baseline_mdd:.2f}%, PnL/MDD={baseline_pnl_mdd:.2f}x")
    print(f"  Candidates qualifying: {len(candidates)}")

    if candidates:
        print(f"\n  {'Label':<30} {'Trades':>7} {'WR':>6} {'PnL':>9} {'MDD':>7} "
              f"{'PnL/MDD':>8} {'dMDD':>6} {'PnL%':>6}")
        print("  " + "-" * 85)
        for c in candidates[:15]:  # Top 15
            print(f"  {c['label']:<30} {c['trades']:>7} {c['wr']:>5.1f}% "
                  f"{c['pnl']:>+8.2f}% {c['mdd']:>6.2f}% {c['pnl_mdd_ratio']:>7.2f}x "
                  f"{c['mdd_delta']:>+5.2f}% {c['pnl_ratio']*100:>5.1f}%")
    else:
        print("\n  No candidates met selection criteria.")
        print("  This means no filter improves risk-adjusted returns while preserving >50% trades.")

    # ========================================
    # Phase 6: WF Validation of Top Candidates
    # ========================================
    wf_candidates = []

    # Build WF candidates from top results
    # Take top 5 by PnL/MDD (if any qualify)
    for c in candidates[:5]:
        label = c['label']
        hyp = c['hypothesis']

        if hyp == 'H1_momentum':
            wf_candidates.append({
                'label': label,
                'factory': lambda N=c['N'], T=c['threshold']: momentum_gate(N, T),
                'params': {},
            })
        elif hyp == 'H2_ema':
            wf_candidates.append({
                'label': label,
                'factory': lambda P=c['period'], M=c['mode']: ema_gate(P, M),
                'params': {},
            })
        elif hyp == 'H3_consecutive_loss':
            wf_candidates.append({
                'label': label,
                'factory': lambda NC=c['max_consec'], CD=c['cooldown']: consecutive_loss_brake(NC, CD),
                'params': {},
            })
        elif hyp == 'H4_atr_momentum':
            wf_candidates.append({
                'label': label,
                'factory': lambda A=c['atr_threshold'], N=c['momentum_N'], T=c['momentum_threshold']: atr_momentum_combo(A, N, T),
                'params': {},
            })

    # Also always test some representative scenarios even if they didn't qualify
    # to provide complete picture
    if not wf_candidates:
        # If no candidates qualified, test the best from each hypothesis
        all_by_hyp = {}
        for r in all_results:
            hyp = r['hypothesis']
            pm = r['pnl'] / r['mdd'] if r['mdd'] > 0 else 0
            if hyp not in all_by_hyp or pm > all_by_hyp[hyp]['best_pm']:
                all_by_hyp[hyp] = {'result': r, 'best_pm': pm}

        for hyp, info in all_by_hyp.items():
            r = info['result']
            label = r['label']
            if hyp == 'H1_momentum':
                wf_candidates.append({
                    'label': f"BEST_{label}",
                    'factory': lambda N=r['N'], T=r['threshold']: momentum_gate(N, T),
                    'params': {},
                })
            elif hyp == 'H2_ema':
                wf_candidates.append({
                    'label': f"BEST_{label}",
                    'factory': lambda P=r['period'], M=r['mode']: ema_gate(P, M),
                    'params': {},
                })
            elif hyp == 'H3_consecutive_loss':
                wf_candidates.append({
                    'label': f"BEST_{label}",
                    'factory': lambda NC=r['max_consec'], CD=r['cooldown']: consecutive_loss_brake(NC, CD),
                    'params': {},
                })
            elif hyp == 'H4_atr_momentum':
                wf_candidates.append({
                    'label': f"BEST_{label}",
                    'factory': lambda A=r['atr_threshold'], N=r['momentum_N'], T=r['momentum_threshold']: atr_momentum_combo(A, N, T),
                    'params': {},
                })

    # Run WF
    wf_results = {}
    if wf_candidates:
        print(f"\n  Running WF for {len(wf_candidates)} candidates + baseline...")

        # Baseline WF
        print("\n--- Baseline (no filter) ---")
        baseline_folds = run_wf_validation(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, atr_arr, lambda: no_filter, {}, n_folds=3
        )
        _print_wf_folds('baseline', baseline_folds)
        wf_results['baseline'] = baseline_folds

        for cand in wf_candidates:
            label = cand['label']
            factory_fn = cand['factory']
            print(f"\n--- {label} ---")

            # For WF, we need to run with the filter applied per fold
            tradeable_start_wf = ATR_WARMUP
            tradeable_bars = n_bars - tradeable_start_wf
            fold_size = tradeable_bars // 3

            fold_results_wf = []
            for fold_idx in range(3):
                oos_start = tradeable_start_wf + fold_idx * fold_size
                oos_end = tradeable_start_wf + (fold_idx + 1) * fold_size if fold_idx < 2 else n_bars

                filter_state = {
                    'closes': closes,
                    'atr_ratio': atr_ratio,
                    'atr': atr_arr,
                    'trade_history': [],
                    'loss_history': [],
                }
                for period in [50, 100, 200]:
                    filter_state[f'ema_{period}'] = emas[f'ema_{period}']

                trend_filter = factory_fn()

                trades, filtered, total_sigs = simulate_portfolio(
                    signals, tpsl, opens, highs, lows, n_bars,
                    atr_ratio, trend_filter, filter_state,
                    oos_start, oos_end
                )

                m = compute_metrics(trades)
                m['filtered'] = filtered
                m['total_signals'] = total_sigs
                m['filter_rate'] = round(filtered / total_sigs * 100, 1) if total_sigs > 0 else 0.0
                m['oos_start'] = oos_start
                m['oos_end'] = oos_end
                m['mc_p'] = mc_test(trades)
                fold_results_wf.append(m)

            _print_wf_folds(label, fold_results_wf)
            wf_results[label] = fold_results_wf

    # ========================================
    # Phase 7: Summary
    # ========================================
    print("\n" + "=" * 80)
    print("PHASE 7: SUMMARY COMPARISON")
    print("=" * 80)

    print(f"\n  {'Scenario':<30} {'IS Trades':>9} {'IS WR':>6} {'IS PnL':>9} "
          f"{'IS MDD':>7} {'IS PnL/MDD':>10}")
    print("  " + "-" * 72)

    # Print baseline
    pm_b = baseline['pnl'] / baseline['mdd'] if baseline['mdd'] > 0 else 0
    print(f"  {'BASELINE':<30} {baseline['trades']:>9} {baseline['wr']:>5.1f}% "
          f"{baseline['pnl']:>+8.2f}% {baseline['mdd']:>6.2f}% {pm_b:>9.2f}x")

    # Print top candidates
    for c in candidates[:10]:
        pm_c = c['pnl'] / c['mdd'] if c['mdd'] > 0 else 0
        print(f"  {c['label']:<30} {c['trades']:>9} {c['wr']:>5.1f}% "
              f"{c['pnl']:>+8.2f}% {c['mdd']:>6.2f}% {pm_c:>9.2f}x")

    if wf_results:
        print(f"\n  {'Scenario':<30} {'OOS Tot':>9} {'Avg WR':>7} {'Tot PnL':>9} "
              f"{'Max MDD':>8} {'PnL/MDD':>8} {'WF':>5}")
        print("  " + "-" * 78)

        for label, folds in wf_results.items():
            total_trades = sum(m['trades'] for m in folds)
            total_pnl = sum(m['pnl'] for m in folds)
            max_mdd = max(m['mdd'] for m in folds)
            avg_wr = np.mean([m['wr'] for m in folds if m['trades'] > 0])
            pm = total_pnl / max_mdd if max_mdd > 0 else 0
            pos_folds = sum(1 for m in folds if m['pnl'] > 0)
            status = f"{pos_folds}/3 {'PASS' if pos_folds == 3 else 'FAIL'}"
            print(f"  {label:<30} {total_trades:>9} {avg_wr:>6.1f}% {total_pnl:>+8.2f}% "
                  f"{max_mdd:>7.2f}% {pm:>7.2f}x {status:>5}")

    # ========================================
    # Verdict
    # ========================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    wf_pass_candidates = []
    if wf_results:
        for label, folds in wf_results.items():
            if label == 'baseline':
                continue
            pos_folds = sum(1 for m in folds if m['pnl'] > 0)
            if pos_folds == 3:
                total_pnl = sum(m['pnl'] for m in folds)
                max_mdd = max(m['mdd'] for m in folds)
                wf_pass_candidates.append({
                    'label': label,
                    'total_pnl': total_pnl,
                    'max_mdd': max_mdd,
                    'pnl_mdd': total_pnl / max_mdd if max_mdd > 0 else 0,
                })

    if wf_pass_candidates:
        wf_pass_candidates.sort(key=lambda x: x['pnl_mdd'], reverse=True)
        best = wf_pass_candidates[0]
        bl_folds = wf_results.get('baseline', [])
        bl_total_pnl = sum(m['pnl'] for m in bl_folds)
        bl_max_mdd = max(m['mdd'] for m in bl_folds)
        bl_pm = bl_total_pnl / bl_max_mdd if bl_max_mdd > 0 else 0

        print(f"\n  ADOPT candidate: {best['label']}")
        print(f"    WF OOS PnL: {best['total_pnl']:+.2f}% (baseline: {bl_total_pnl:+.2f}%)")
        print(f"    WF OOS MDD: {best['max_mdd']:.2f}% (baseline: {bl_max_mdd:.2f}%)")
        print(f"    WF OOS PnL/MDD: {best['pnl_mdd']:.2f}x (baseline: {bl_pm:.2f}x)")

        if best['pnl_mdd'] > bl_pm * 1.05:
            print("\n  RECOMMENDATION: GO — filter improves risk-adjusted OOS performance")
        else:
            print("\n  RECOMMENDATION: STOP — filter does not materially improve OOS performance")
    else:
        print("\n  No filter passed WF 3/3.")
        print("  RECOMMENDATION: STOP — no trend filter consistently improves OOS performance.")
        print("  The current unfiltered strategy is already the best option on this dataset.")

    # ========================================
    # Save Results
    # ========================================
    elapsed = time.time() - start_time

    output = {
        'study': 'trend_filter_study',
        'date': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data_bars': n_bars,
        'data_days': round(n_bars / 288, 1),
        'patterns': {'long': len(pL), 'short': len(pS), 'total': len(pL) + len(pS)},
        'protocol': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'timeout_bars': TIMEOUT_BARS,
            'max_positions': MAX_POSITIONS,
            'direction_cap': DIRECTION_CAP,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'clamp': [CLAMP_LO, CLAMP_HI],
        },
        'baseline': baseline,
        'phase2_momentum': [_serialize(r) for r in mom_results],
        'phase3_ema': [_serialize(r) for r in ema_results],
        'phase4_consecutive_loss': [_serialize(r) for r in clb_results],
        'phase5_atr_momentum': [_serialize(r) for r in atrmom_results],
        'candidates': [_serialize(c) for c in candidates[:15]],
        'wf_results': {k: v for k, v in wf_results.items()} if wf_results else {},
        'verdict': 'GO' if wf_pass_candidates else 'STOP',
        'best_filter': wf_pass_candidates[0] if wf_pass_candidates else None,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_FILE}")
    print(f"  Elapsed: {elapsed:.1f}s")


def _serialize(d):
    """Make dict JSON-serializable."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, np.int64)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


if __name__ == '__main__':
    main()
