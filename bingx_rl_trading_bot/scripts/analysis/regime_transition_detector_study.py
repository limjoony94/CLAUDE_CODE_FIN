#!/usr/bin/env python3
"""
Regime Transition Detector Study — 자동 regime 전환 감지 메커니즘 연구

핵심 질문:
  1. LONG/SHORT 방향 정확도가 수익 임계값 이하로 떨어지는 시점을 자동 감지할 수 있는가?
  2. 어떤 지표가 regime 전환을 가장 빨리 (lead time) 감지하는가?
  3. Detection → Direction Gating 적용 시 백테스트 성과는?
  4. WF 검증에서도 유효한가?

방법:
  - 297d 데이터 + 130패턴 ATR-scaled backtest
  - Breakeven WR 계산 (R:R 기반) → 임계값 설정
  - 8가지 Regime Detector 후보 테스트:
    D1. EMA(20) slope (existing regime_sizing 지표)
    D2. EMA(50) slope (longer-term trend)
    D3. Price momentum (N-bar return)
    D4. Rolling LONG WR (adaptive — 최근 N trades)
    D5. Rolling direction accuracy (fixed-horizon, no TP/SL)
    D6. BTC drawdown from local high
    D7. EMA(20) vs EMA(50) crossover
    D8. Composite (best individual detectors combined)
  - 각 detector에 대해:
    (a) Detection lead time (regime 전환 전 감지 시점)
    (b) False positive rate
    (c) Backtest: detector가 "LONG OFF" 시그널 시 LONG 진입 차단
    (d) WF 3-fold 검증

Protocol:
  - Production classify_candle() via scanner
  - LEVERAGE = 3, FEE = 0.10% × LEVERAGE
  - ATR-scaled TP/SL (scanner bt_signals_atr)
  - N=9 portfolio, 1/N=11.1% sizing
  - Timeout 864 bars (DROP)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'regime_transition_detector_study.json')
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
LEVERAGE = scanner.LEVERAGE
FEE_PCT = scanner.FEE_PCT
BARS_PER_DAY = 288
TIMEOUT_BARS = 864
MAX_POSITIONS = 9
POSITION_SIZE = 1.0 / MAX_POSITIONS  # 11.1%


def load_patterns():
    """Load current 130 patterns from dynamic_patterns.json."""
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    patterns = []
    for key, info in data.get('pattern_details', {}).items():
        patterns.append({
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': info['tp'],
            'sl': info['sl'],
        })
    # Neutral window from top level
    neutral_window = data.get('neutral_window', {})
    return patterns, neutral_window


def generate_all_trades(patterns, signal_index, opens, highs, lows, closes,
                        n_bars, use_atr=True):
    """Generate all trades using scanner backtest engine."""
    atr_ratio = None
    if use_atr and hasattr(scanner, 'compute_atr_ratio'):
        atr_ratio = scanner.compute_atr_ratio(highs, lows, closes)

    all_trades = []
    for pat in patterns:
        sigs = signal_index.get(pat['pattern'], [])
        if not sigs:
            continue
        direction = pat['direction']
        tp_pct = pat['tp']
        sl_pct = pat['sl']

        if use_atr and atr_ratio is not None:
            raw_trades = scanner.bt_signals_atr(
                sigs, direction, tp_pct, sl_pct,
                opens, highs, lows, n_bars, atr_ratio)
        else:
            raw_trades = scanner.bt_signals(
                sigs, direction, tp_pct, sl_pct,
                opens, highs, lows, n_bars)

        for entry_bar, exit_bar, pnl in raw_trades:
            all_trades.append({
                'bar': entry_bar,
                'exit_bar': exit_bar,
                'direction': direction,
                'pnl': round(pnl, 4),
                'pattern': pat['pattern'],
                'tp': tp_pct,
                'sl': sl_pct,
                'hold_bars': exit_bar - entry_bar,
            })

    all_trades.sort(key=lambda t: t['bar'])
    return all_trades


def compute_breakeven_wr(trades):
    """Compute breakeven WR from actual trade R:R distribution."""
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    if not wins or not losses:
        return 50.0
    avg_win = np.mean([t['pnl'] for t in wins])
    avg_loss = abs(np.mean([t['pnl'] for t in losses]))
    # BE WR = avg_loss / (avg_win + avg_loss)
    be_wr = avg_loss / (avg_win + avg_loss) * 100
    return round(be_wr, 1)


# ═══════════════════════════════════════════════════════════════
# REGIME DETECTORS
# ═══════════════════════════════════════════════════════════════

def detector_ema_slope(closes, period=20, lookback=5):
    """D1/D2: EMA slope direction. Returns array: +1 (uptrend), -1 (downtrend), 0 (flat)."""
    ema = pd.Series(closes).ewm(span=period, adjust=False).mean().values
    regime = np.zeros(len(closes), dtype=int)
    for i in range(lookback, len(closes)):
        slope = (ema[i] - ema[i - lookback]) / ema[i - lookback] * 100
        if slope > 0.05:   # >0.05% over lookback = uptrend
            regime[i] = 1
        elif slope < -0.05:
            regime[i] = -1
    return regime


def detector_momentum(closes, n_bars_lookback=288):
    """D3: Price momentum — N-bar return. +1 if positive, -1 if negative."""
    regime = np.zeros(len(closes), dtype=int)
    for i in range(n_bars_lookback, len(closes)):
        ret = (closes[i] - closes[i - n_bars_lookback]) / closes[i - n_bars_lookback] * 100
        if ret > 0.5:
            regime[i] = 1
        elif ret < -0.5:
            regime[i] = -1
    return regime


def detector_rolling_trade_wr(trades, n_bars_total, direction, window=30):
    """D4: Rolling WR of last N trades of this direction.
    Returns array indexed by bar: WR of last `window` trades.
    """
    dir_trades = sorted([t for t in trades if t['direction'] == direction],
                        key=lambda t: t['bar'])
    wr_array = np.full(n_bars_total, np.nan)

    for i in range(window, len(dir_trades)):
        recent = dir_trades[i - window:i]
        wr = sum(1 for t in recent if t['pnl'] > 0) / len(recent) * 100
        bar = dir_trades[i]['bar']
        wr_array[bar] = wr

    # Forward fill
    last_val = np.nan
    for i in range(n_bars_total):
        if not np.isnan(wr_array[i]):
            last_val = wr_array[i]
        else:
            wr_array[i] = last_val

    return wr_array


def detector_direction_accuracy(signal_index, patterns, opens, closes,
                                n_bars, direction, horizon=30, window=50):
    """D5: Rolling direction accuracy — did price move in predicted direction?
    Uses fixed-horizon return (no TP/SL).
    """
    # Collect direction signals sorted by bar
    dir_sigs = []
    for pat in patterns:
        if pat['direction'] != direction:
            continue
        for sig in signal_index.get(pat['pattern'], []):
            entry_bar = sig + 1
            if entry_bar + horizon >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue
            exit_price = closes[entry_bar + horizon]
            if direction == 'LONG':
                correct = 1 if exit_price > entry_price else 0
            else:
                correct = 1 if exit_price < entry_price else 0
            dir_sigs.append((sig, correct))

    dir_sigs.sort(key=lambda x: x[0])

    acc_array = np.full(n_bars, np.nan)
    for i in range(window, len(dir_sigs)):
        recent = dir_sigs[i - window:i]
        acc = sum(c for _, c in recent) / len(recent) * 100
        bar = dir_sigs[i][0]
        acc_array[bar] = acc

    # Forward fill
    last_val = np.nan
    for i in range(n_bars):
        if not np.isnan(acc_array[i]):
            last_val = acc_array[i]
        else:
            acc_array[i] = last_val
    return acc_array


def detector_drawdown(closes, lookback=288*5):
    """D6: Drawdown from rolling high. Negative = in drawdown, 0 = at peak."""
    dd = np.zeros(len(closes))
    rolling_high = closes[0]
    for i in range(len(closes)):
        start = max(0, i - lookback)
        rolling_high = max(closes[start:i + 1])
        dd[i] = (closes[i] - rolling_high) / rolling_high * 100
    return dd


def detector_ema_cross(closes, fast=20, slow=50):
    """D7: EMA crossover. +1 when fast>slow (uptrend), -1 when fast<slow (downtrend)."""
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    regime = np.zeros(len(closes), dtype=int)
    for i in range(slow, len(closes)):
        if ema_fast[i] > ema_slow[i]:
            regime[i] = 1
        elif ema_fast[i] < ema_slow[i]:
            regime[i] = -1
    return regime


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_portfolio(trades, n_bars, direction_gate=None, direction_cap=7):
    """Simulate N=9 portfolio with optional direction gating.

    direction_gate: dict mapping direction ('LONG'/'SHORT') to array of bool
                    where True = allowed, False = blocked at that bar.

    Returns: total_pnl, max_dd, trades_taken, wins, trade_list
    """
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0
    active_slots = []
    trade_log = []

    for t in trades:
        bar = t['bar']
        direction = t['direction']

        # Remove expired slots
        active_slots = [s for s in active_slots if s['exit_bar'] > bar]

        # Check direction gate
        if direction_gate and direction in direction_gate:
            gate = direction_gate[direction]
            if bar < len(gate) and not gate[bar]:
                continue

        # Direction cap
        dir_count = sum(1 for s in active_slots if s['direction'] == direction)
        if dir_count >= direction_cap:
            continue

        # Max positions
        if len(active_slots) >= MAX_POSITIONS:
            continue

        # Record trade
        pnl_portfolio = t['pnl'] * POSITION_SIZE
        equity *= (1 + pnl_portfolio / 100)
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity * 100
        max_dd = min(max_dd, dd)

        active_slots.append({
            'direction': direction,
            'exit_bar': t['exit_bar'],
        })
        trade_log.append({
            'bar': bar, 'direction': direction,
            'pnl': t['pnl'], 'pnl_port': round(pnl_portfolio, 4),
            'equity': round(equity, 2),
        })

    total_pnl = equity - 100.0
    wins = sum(1 for t in trade_log if t['pnl'] > 0)
    return total_pnl, abs(max_dd), len(trade_log), wins, trade_log


# ═══════════════════════════════════════════════════════════════
# WF VALIDATION
# ═══════════════════════════════════════════════════════════════

def wf_validate(trades, n_bars, closes, patterns, signal_index, opens,
                detector_fn, detector_kwargs, n_folds=3, direction_cap=7):
    """Walk-forward validation with detector-based gating.

    For each fold, build detector on IS data, apply gating on OOS data.
    Returns list of fold results.
    """
    # Expanding window: n_folds+1 segments, fold k uses seg 1..k+1 as IS, seg k+2 as OOS
    n_segments = n_folds + 1
    fold_size = n_bars // n_segments
    fold_results = []

    for fold in range(n_folds):
        is_end = fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, n_bars)

        if oos_end <= oos_start:
            continue

        # IS trades for detector calibration
        is_trades = [t for t in trades if t['bar'] < is_end]
        oos_trades = [t for t in trades if oos_start <= t['bar'] < oos_end]

        if not oos_trades:
            fold_results.append({
                'fold': fold + 1, 'oos_trades': 0, 'oos_wr': 0,
                'oos_pnl': 0, 'oos_mdd': 0, 'pass': False,
            })
            continue

        # Build detector — use full closes array but detector_fn handles calibration
        gate = detector_fn(
            trades=trades, is_end=is_end, n_bars=n_bars,
            closes=closes, patterns=patterns, signal_index=signal_index,
            opens=opens, **detector_kwargs)

        # Simulate OOS
        pnl, mdd, n_taken, wins, _ = simulate_portfolio(
            oos_trades, n_bars, direction_gate=gate, direction_cap=direction_cap)

        wr = wins / n_taken * 100 if n_taken > 0 else 0
        fold_results.append({
            'fold': fold + 1,
            'is_bars': is_end,
            'oos_range': [oos_start, oos_end],
            'oos_days': round((oos_end - oos_start) / BARS_PER_DAY, 0),
            'oos_trades': n_taken,
            'oos_wr': round(wr, 1),
            'oos_pnl': round(pnl, 1),
            'oos_mdd': round(mdd, 1),
            'pass': pnl > 0,
        })

    return fold_results


# ═══════════════════════════════════════════════════════════════
# DETECTOR GATE BUILDERS (for WF)
# ═══════════════════════════════════════════════════════════════

def gate_ema_slope(trades, is_end, n_bars, closes, period=20,
                   lookback=5, gate_direction='LONG', **kwargs):
    """Gate LONG when EMA slope is downtrend."""
    regime = detector_ema_slope(closes, period=period, lookback=lookback)
    gate = np.ones(n_bars, dtype=bool)
    for i in range(n_bars):
        if gate_direction == 'LONG' and regime[i] == -1:
            gate[i] = False
        elif gate_direction == 'SHORT' and regime[i] == 1:
            gate[i] = False
    return {gate_direction: gate}


def gate_momentum(trades, is_end, n_bars, closes, momentum_bars=288,
                  gate_direction='LONG', **kwargs):
    """Gate LONG when momentum is negative."""
    regime = detector_momentum(closes, n_bars_lookback=momentum_bars)
    gate = np.ones(n_bars, dtype=bool)
    for i in range(n_bars):
        if gate_direction == 'LONG' and regime[i] == -1:
            gate[i] = False
        elif gate_direction == 'SHORT' and regime[i] == 1:
            gate[i] = False
    return {gate_direction: gate}


def gate_rolling_wr(trades, is_end, n_bars, closes, wr_window=30,
                    wr_threshold=60.0, gate_direction='LONG', **kwargs):
    """Gate direction when rolling WR drops below threshold."""
    # Only use IS trades for WR computation
    is_trades = [t for t in trades if t['bar'] < is_end]
    wr_array = detector_rolling_trade_wr(is_trades, n_bars, gate_direction, window=wr_window)
    gate = np.ones(n_bars, dtype=bool)
    for i in range(n_bars):
        if not np.isnan(wr_array[i]) and wr_array[i] < wr_threshold:
            gate[i] = False
    return {gate_direction: gate}


def gate_direction_accuracy(trades, is_end, n_bars, closes, opens,
                            patterns, signal_index, acc_window=50,
                            acc_threshold=50.0, acc_horizon=30,
                            gate_direction='LONG', **kwargs):
    """Gate direction when rolling direction accuracy drops below threshold."""
    acc_array = detector_direction_accuracy(
        signal_index, patterns, opens, closes, n_bars,
        gate_direction, horizon=acc_horizon, window=acc_window)
    gate = np.ones(n_bars, dtype=bool)
    for i in range(n_bars):
        if not np.isnan(acc_array[i]) and acc_array[i] < acc_threshold:
            gate[i] = False
    return {gate_direction: gate}


def gate_drawdown(trades, is_end, n_bars, closes, dd_threshold=-3.0,
                  gate_direction='LONG', **kwargs):
    """Gate LONG when BTC in significant drawdown."""
    dd = detector_drawdown(closes)
    gate = np.ones(n_bars, dtype=bool)
    for i in range(n_bars):
        if gate_direction == 'LONG' and dd[i] < dd_threshold:
            gate[i] = False
        elif gate_direction == 'SHORT' and dd[i] > abs(dd_threshold):
            # Gate SHORT when BTC is rallying (not in drawdown)
            gate[i] = False
    return {gate_direction: gate}


def gate_ema_cross(trades, is_end, n_bars, closes, fast=20, slow=50,
                   gate_direction='LONG', **kwargs):
    """Gate LONG when fast EMA < slow EMA (downtrend)."""
    regime = detector_ema_cross(closes, fast=fast, slow=slow)
    gate = np.ones(n_bars, dtype=bool)
    for i in range(n_bars):
        if gate_direction == 'LONG' and regime[i] == -1:
            gate[i] = False
        elif gate_direction == 'SHORT' and regime[i] == 1:
            gate[i] = False
    return {gate_direction: gate}


def gate_composite(trades, is_end, n_bars, closes, opens, patterns,
                   signal_index, gate_direction='LONG', **kwargs):
    """Composite: require majority vote of top detectors."""
    # Combine EMA slope, momentum, EMA cross
    ema20 = detector_ema_slope(closes, period=20, lookback=5)
    mom = detector_momentum(closes, n_bars_lookback=288)
    cross = detector_ema_cross(closes, fast=20, slow=50)

    gate = np.ones(n_bars, dtype=bool)
    for i in range(n_bars):
        votes_against = 0
        if gate_direction == 'LONG':
            if ema20[i] == -1:
                votes_against += 1
            if mom[i] == -1:
                votes_against += 1
            if cross[i] == -1:
                votes_against += 1
        else:
            if ema20[i] == 1:
                votes_against += 1
            if mom[i] == 1:
                votes_against += 1
            if cross[i] == 1:
                votes_against += 1
        if votes_against >= 2:
            gate[i] = False
    return {gate_direction: gate}


def gate_short_only(trades, is_end, n_bars, closes, **kwargs):
    """Nuclear option: block ALL LONG trades."""
    gate = np.zeros(n_bars, dtype=bool)
    return {'LONG': gate}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    results = {'study': 'regime_transition_detector', 'sections': {}}

    print("=" * 70)
    print("REGIME TRANSITION DETECTOR — 자동 감지 메커니즘 연구")
    print("=" * 70)

    # ── Load data ──
    print("\nLoading data...")
    df = scanner.load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    signal_index = scanner.build_signal_index(types, len(df))
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    timestamps = pd.to_datetime(df['timestamp'])
    bar_to_date = timestamps.dt.date.values

    patterns, neutral_window = load_patterns()
    n_start = neutral_window.get('start_bar', 0)
    n_end = neutral_window.get('end_bar', n_bars)

    long_pats = [p for p in patterns if p['direction'] == 'LONG']
    short_pats = [p for p in patterns if p['direction'] == 'SHORT']

    print(f"  {n_bars} bars ({n_bars/BARS_PER_DAY:.0f} days)")
    print(f"  {len(patterns)} patterns ({len(long_pats)}L + {len(short_pats)}S)")
    print(f"  Neutral window: bar {n_start}→{n_end} ({(n_end-n_start)/BARS_PER_DAY:.0f}d)")

    # ── Generate all trades ──
    print("\nGenerating trades (ATR-scaled)...")
    trades = generate_all_trades(
        patterns, signal_index, opens, highs, lows, closes, n_bars, use_atr=True)
    print(f"  Total: {len(trades)} trades")

    long_trades = [t for t in trades if t['direction'] == 'LONG']
    short_trades = [t for t in trades if t['direction'] == 'SHORT']
    print(f"  LONG: {len(long_trades)} trades, WR {sum(1 for t in long_trades if t['pnl']>0)/len(long_trades)*100:.1f}%")
    print(f"  SHORT: {len(short_trades)} trades, WR {sum(1 for t in short_trades if t['pnl']>0)/len(short_trades)*100:.1f}%")

    be_wr_all = compute_breakeven_wr(trades)
    be_wr_long = compute_breakeven_wr(long_trades)
    be_wr_short = compute_breakeven_wr(short_trades)
    print(f"\n  Breakeven WR: ALL {be_wr_all}%, LONG {be_wr_long}%, SHORT {be_wr_short}%")

    results['meta'] = {
        'bars': n_bars,
        'days': round(n_bars / BARS_PER_DAY, 1),
        'patterns': len(patterns),
        'long_patterns': len(long_pats),
        'short_patterns': len(short_pats),
        'total_trades': len(trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'breakeven_wr_all': be_wr_all,
        'breakeven_wr_long': be_wr_long,
        'breakeven_wr_short': be_wr_short,
    }

    # ═══════════════════════════════════════════════════════════
    # 1. REGIME TIME SERIES — Detector Signals Over Time
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("1. REGIME DETECTOR SIGNALS OVER TIME")
    print(f"{'='*70}")

    ema20_regime = detector_ema_slope(closes, period=20, lookback=5)
    ema50_regime = detector_ema_slope(closes, period=50, lookback=10)
    mom_regime = detector_momentum(closes, n_bars_lookback=288)
    cross_regime = detector_ema_cross(closes, fast=20, slow=50)
    dd_array = detector_drawdown(closes)

    # Rolling WR (using all trades — IS analysis)
    long_wr_array = detector_rolling_trade_wr(trades, n_bars, 'LONG', window=30)
    short_wr_array = detector_rolling_trade_wr(trades, n_bars, 'SHORT', window=30)

    # Direction accuracy (30-bar horizon, 50-signal window)
    long_acc_array = detector_direction_accuracy(
        signal_index, patterns, opens, closes, n_bars, 'LONG', horizon=30, window=50)
    short_acc_array = detector_direction_accuracy(
        signal_index, patterns, opens, closes, n_bars, 'SHORT', horizon=30, window=50)

    # Monthly summary of all detectors
    period_bars = 30 * BARS_PER_DAY  # 30-day periods
    n_periods = n_bars // period_bars
    period_analysis = []

    print(f"\n  {'Period':<8} {'BTC%':>6} {'EMA20':>6} {'EMA50':>6} {'MOM':>6} "
          f"{'CROSS':>6} {'L-WR':>6} {'S-WR':>6} {'L-Acc':>6} {'S-Acc':>6} {'DD%':>6}")
    print(f"  {'-'*78}")

    for p in range(n_periods):
        start = p * period_bars
        end = min(start + period_bars, n_bars)
        btc_ret = (closes[end - 1] - closes[start]) / closes[start] * 100

        ema20_avg = np.mean(ema20_regime[start:end])
        ema50_avg = np.mean(ema50_regime[start:end])
        mom_avg = np.mean(mom_regime[start:end])
        cross_avg = np.mean(cross_regime[start:end])
        dd_min = np.min(dd_array[start:end])

        l_wr_vals = long_wr_array[start:end]
        l_wr_avg = np.nanmean(l_wr_vals) if not np.all(np.isnan(l_wr_vals)) else 0
        s_wr_vals = short_wr_array[start:end]
        s_wr_avg = np.nanmean(s_wr_vals) if not np.all(np.isnan(s_wr_vals)) else 0

        l_acc_vals = long_acc_array[start:end]
        l_acc_avg = np.nanmean(l_acc_vals) if not np.all(np.isnan(l_acc_vals)) else 0
        s_acc_vals = short_acc_array[start:end]
        s_acc_avg = np.nanmean(s_acc_vals) if not np.all(np.isnan(s_acc_vals)) else 0

        date_str = str(bar_to_date[start])[:7]
        print(f"  P{p+1:<5} {btc_ret:>+6.1f} {ema20_avg:>+6.2f} {ema50_avg:>+6.2f} "
              f"{mom_avg:>+6.2f} {cross_avg:>+6.2f} {l_wr_avg:>5.1f}% {s_wr_avg:>5.1f}% "
              f"{l_acc_avg:>5.1f}% {s_acc_avg:>5.1f}% {dd_min:>+6.1f}")

        period_analysis.append({
            'period': p + 1,
            'date': date_str,
            'btc_return': round(btc_ret, 1),
            'ema20_signal': round(ema20_avg, 2),
            'ema50_signal': round(ema50_avg, 2),
            'momentum_signal': round(mom_avg, 2),
            'cross_signal': round(cross_avg, 2),
            'long_rolling_wr': round(l_wr_avg, 1),
            'short_rolling_wr': round(s_wr_avg, 1),
            'long_direction_acc': round(l_acc_avg, 1),
            'short_direction_acc': round(s_acc_avg, 1),
            'max_drawdown': round(dd_min, 1),
        })

    results['sections']['period_signals'] = period_analysis

    # ═══════════════════════════════════════════════════════════
    # 2. DETECTOR CORRELATION WITH DIRECTION WR
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("2. DETECTOR ↔ DIRECTION WR CORRELATION")
    print(f"{'='*70}")

    # Correlate detector signals with trade outcomes
    # For each trade, get the detector value at trade entry bar
    correlation_results = {}

    detector_arrays = {
        'EMA20_slope': ema20_regime,
        'EMA50_slope': ema50_regime,
        'Momentum_24h': mom_regime,
        'EMA_Cross_20_50': cross_regime,
        'BTC_Drawdown': dd_array,
        'Rolling_LONG_WR': long_wr_array,
        'Rolling_SHORT_WR': short_wr_array,
        'DirAcc_LONG': long_acc_array,
        'DirAcc_SHORT': short_acc_array,
    }

    for det_name, det_array in detector_arrays.items():
        for direction in ['LONG', 'SHORT']:
            dir_trades = [t for t in trades if t['direction'] == direction]
            det_vals = []
            outcomes = []
            for t in dir_trades:
                bar = t['bar']
                if bar < len(det_array):
                    val = det_array[bar]
                    if not np.isnan(val):
                        det_vals.append(val)
                        outcomes.append(1 if t['pnl'] > 0 else 0)

            if len(det_vals) > 30:
                corr = np.corrcoef(det_vals, outcomes)[0, 1]
                key = f"{det_name}→{direction}"
                correlation_results[key] = round(corr, 3)

    # Sort by absolute correlation
    sorted_corr = sorted(correlation_results.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {'Detector → Direction':<35} {'Corr':>8} {'Strength':>10}")
    print(f"  {'-'*55}")
    for name, corr in sorted_corr[:15]:
        strength = "STRONG" if abs(corr) > 0.15 else "MODERATE" if abs(corr) > 0.08 else "WEAK"
        print(f"  {name:<35} {corr:>+8.3f} {strength:>10}")

    results['sections']['correlations'] = dict(sorted_corr)

    # ═══════════════════════════════════════════════════════════
    # 3. IN-SAMPLE: DETECTOR-BASED GATING VS BASELINE
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("3. DETECTOR-BASED GATING — IN-SAMPLE COMPARISON")
    print(f"{'='*70}")

    # Baseline: no gating
    bl_pnl, bl_mdd, bl_n, bl_w, _ = simulate_portfolio(trades, n_bars)
    bl_wr = bl_w / bl_n * 100 if bl_n > 0 else 0
    print(f"\n  BASELINE (no gating): {bl_n} trades, WR {bl_wr:.1f}%, "
          f"PnL {bl_pnl:+.1f}%, MDD {bl_mdd:.1f}%")

    # Test each detector
    scenarios = [
        ("D1: EMA20 slope gate LONG", gate_ema_slope,
         {'period': 20, 'lookback': 5, 'gate_direction': 'LONG'}),
        ("D2: EMA50 slope gate LONG", gate_ema_slope,
         {'period': 50, 'lookback': 10, 'gate_direction': 'LONG'}),
        ("D3a: Momentum 24h gate LONG", gate_momentum,
         {'momentum_bars': 288, 'gate_direction': 'LONG'}),
        ("D3b: Momentum 48h gate LONG", gate_momentum,
         {'momentum_bars': 576, 'gate_direction': 'LONG'}),
        ("D4: Rolling WR<60% gate LONG", gate_rolling_wr,
         {'wr_window': 30, 'wr_threshold': 60.0, 'gate_direction': 'LONG'}),
        ("D4b: Rolling WR<55% gate LONG", gate_rolling_wr,
         {'wr_window': 30, 'wr_threshold': 55.0, 'gate_direction': 'LONG'}),
        ("D5: DirAcc<50% gate LONG", gate_direction_accuracy,
         {'acc_window': 50, 'acc_threshold': 50.0, 'acc_horizon': 30, 'gate_direction': 'LONG'}),
        ("D6: DD<-3% gate LONG", gate_drawdown,
         {'dd_threshold': -3.0, 'gate_direction': 'LONG'}),
        ("D6b: DD<-5% gate LONG", gate_drawdown,
         {'dd_threshold': -5.0, 'gate_direction': 'LONG'}),
        ("D7: EMA cross gate LONG", gate_ema_cross,
         {'fast': 20, 'slow': 50, 'gate_direction': 'LONG'}),
        ("D8: Composite 2/3 gate LONG", gate_composite,
         {'gate_direction': 'LONG'}),
        ("D9: SHORT-only (all LONG off)", gate_short_only, {}),
        # Also test SHORT gating for completeness
        ("D1s: EMA20 slope gate SHORT", gate_ema_slope,
         {'period': 20, 'lookback': 5, 'gate_direction': 'SHORT'}),
        ("D7s: EMA cross gate SHORT", gate_ema_cross,
         {'fast': 20, 'slow': 50, 'gate_direction': 'SHORT'}),
    ]

    scenario_results = []
    print(f"\n  {'Scenario':<35} {'Trades':>6} {'WR%':>6} {'PnL%':>8} {'MDD%':>6} "
          f"{'PnL/MDD':>8} {'vs BL':>8}")
    print(f"  {'-'*82}")

    # Print baseline first
    bl_ratio = bl_pnl / bl_mdd if bl_mdd > 0 else 0
    print(f"  {'BASELINE':<35} {bl_n:>6} {bl_wr:>5.1f}% {bl_pnl:>+7.1f}% "
          f"{bl_mdd:>5.1f}% {bl_ratio:>7.1f}x {'---':>8}")

    for name, gate_fn, kwargs in scenarios:
        # Build gate using full data (IS analysis — WF will fix this later)
        gate = gate_fn(
            trades=trades, is_end=n_bars, n_bars=n_bars, closes=closes,
            opens=opens, patterns=patterns, signal_index=signal_index,
            **kwargs)

        pnl, mdd, n_taken, wins, _ = simulate_portfolio(
            trades, n_bars, direction_gate=gate)

        wr = wins / n_taken * 100 if n_taken > 0 else 0
        ratio = pnl / mdd if mdd > 0 else 0
        delta_pnl = pnl - bl_pnl

        blocked = bl_n - n_taken
        print(f"  {name:<35} {n_taken:>6} {wr:>5.1f}% {pnl:>+7.1f}% "
              f"{mdd:>5.1f}% {ratio:>7.1f}x {delta_pnl:>+7.1f}%")

        scenario_results.append({
            'scenario': name,
            'trades': n_taken,
            'blocked': blocked,
            'wr': round(wr, 1),
            'pnl': round(pnl, 1),
            'mdd': round(mdd, 1),
            'pnl_mdd': round(ratio, 2),
            'vs_baseline_pnl': round(delta_pnl, 1),
        })

    results['sections']['is_scenarios'] = scenario_results

    # ═══════════════════════════════════════════════════════════
    # 4. LEAD TIME ANALYSIS — How early does each detector signal?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("4. DETECTION LEAD TIME — LONG loss cluster 사전 감지")
    print(f"{'='*70}")

    # Find LONG loss clusters (3+ losses in 288 bars)
    long_losses = sorted([t for t in trades if t['direction'] == 'LONG' and t['pnl'] <= 0],
                         key=lambda t: t['bar'])
    loss_clusters = []
    i = 0
    while i < len(long_losses):
        cluster = [long_losses[i]]
        j = i + 1
        while j < len(long_losses) and long_losses[j]['bar'] - cluster[0]['bar'] <= 288:
            cluster.append(long_losses[j])
            j += 1
        if len(cluster) >= 3:
            loss_clusters.append({
                'start_bar': cluster[0]['bar'],
                'end_bar': cluster[-1]['bar'],
                'n_losses': len(cluster),
                'total_loss': sum(t['pnl'] for t in cluster),
            })
            i = j
        else:
            i += 1

    print(f"\n  Found {len(loss_clusters)} LONG loss clusters (3+ losses in 24h)")
    for c in loss_clusters:
        date = str(bar_to_date[min(c['start_bar'], n_bars - 1)])
        print(f"    Bar {c['start_bar']} ({date}): {c['n_losses']} losses, "
              f"total {c['total_loss']:+.1f}%")

    # For each cluster, check when each detector first signaled "danger"
    lead_times = {
        'EMA20': [], 'EMA50': [], 'Momentum': [],
        'EMA_Cross': [], 'Drawdown_3%': [], 'Drawdown_5%': [],
    }

    detector_checks = {
        'EMA20': (ema20_regime, lambda v: v == -1),
        'EMA50': (ema50_regime, lambda v: v == -1),
        'Momentum': (mom_regime, lambda v: v == -1),
        'EMA_Cross': (cross_regime, lambda v: v == -1),
        'Drawdown_3%': (dd_array, lambda v: v < -3.0),
        'Drawdown_5%': (dd_array, lambda v: v < -5.0),
    }

    for c in loss_clusters:
        cluster_bar = c['start_bar']
        lookback = 288 * 3  # Check 3 days before cluster
        for det_name, (det_arr, check_fn) in detector_checks.items():
            lead = None
            for b in range(max(0, cluster_bar - lookback), cluster_bar):
                if check_fn(det_arr[b]):
                    lead = cluster_bar - b  # bars before cluster
                    break
            if lead is not None:
                lead_times[det_name].append(lead)
            else:
                lead_times[det_name].append(-1)  # No warning

    print(f"\n  {'Detector':<20} {'Avg Lead':>10} {'Detected%':>10} {'Min':>6} {'Max':>6}")
    print(f"  {'-'*55}")

    lead_time_results = {}
    for det_name, leads in lead_times.items():
        valid = [l for l in leads if l >= 0]
        pct_detected = len(valid) / len(leads) * 100 if leads else 0
        avg_lead = np.mean(valid) if valid else 0
        min_lead = min(valid) if valid else 0
        max_lead = max(valid) if valid else 0
        lead_label = f"{avg_lead/BARS_PER_DAY*24:.0f}h" if avg_lead > 0 else "N/A"

        print(f"  {det_name:<20} {lead_label:>10} {pct_detected:>9.0f}% "
              f"{min_lead:>6} {max_lead:>6}")
        lead_time_results[det_name] = {
            'avg_lead_bars': round(avg_lead, 0),
            'avg_lead_hours': round(avg_lead / BARS_PER_DAY * 24, 1),
            'detection_rate': round(pct_detected, 1),
            'min_lead': min_lead,
            'max_lead': max_lead,
        }

    results['sections']['lead_times'] = lead_time_results
    results['sections']['loss_clusters'] = loss_clusters

    # ═══════════════════════════════════════════════════════════
    # 5. WF VALIDATION — Top Detectors
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("5. WALK-FORWARD VALIDATION (3-fold)")
    print(f"{'='*70}")

    # WF: baseline
    print("\n  [WF] BASELINE (no gating):")
    wf_bl = wf_validate(trades, n_bars, closes, patterns, signal_index, opens,
                        lambda **kw: None, {})
    for f in wf_bl:
        status = "PASS" if f['pass'] else "FAIL"
        print(f"    Fold {f['fold']}: {f['oos_trades']} trades, WR {f['oos_wr']}%, "
              f"PnL {f['oos_pnl']:+.1f}%, MDD {f['oos_mdd']:.1f}% → {status}")

    # WF: top scenarios
    wf_scenarios = [
        ("D1: EMA20 gate LONG", gate_ema_slope,
         {'period': 20, 'lookback': 5, 'gate_direction': 'LONG'}),
        ("D3a: Mom24h gate LONG", gate_momentum,
         {'momentum_bars': 288, 'gate_direction': 'LONG'}),
        ("D6: DD<-3% gate LONG", gate_drawdown,
         {'dd_threshold': -3.0, 'gate_direction': 'LONG'}),
        ("D7: EMA cross gate LONG", gate_ema_cross,
         {'fast': 20, 'slow': 50, 'gate_direction': 'LONG'}),
        ("D8: Composite gate LONG", gate_composite,
         {'gate_direction': 'LONG'}),
        ("D9: SHORT-only", gate_short_only, {}),
        ("D4: Rolling WR<60%", gate_rolling_wr,
         {'wr_window': 30, 'wr_threshold': 60.0, 'gate_direction': 'LONG'}),
        ("D5: DirAcc<50%", gate_direction_accuracy,
         {'acc_window': 50, 'acc_threshold': 50.0, 'acc_horizon': 30, 'gate_direction': 'LONG'}),
    ]

    wf_results = {}
    print(f"\n  {'Scenario':<30} {'F1':>12} {'F2':>12} {'F3':>12} {'Result':>8}")
    print(f"  {'-'*78}")

    for name, gate_fn, kwargs in wf_scenarios:
        wf = wf_validate(trades, n_bars, closes, patterns, signal_index, opens,
                         gate_fn, kwargs)
        passes = sum(1 for f in wf if f['pass'])
        result = f"{passes}/3 {'PASS' if passes == 3 else 'FAIL'}"

        fold_strs = []
        for f in wf:
            fold_strs.append(f"{f['oos_pnl']:+.0f}%")

        print(f"  {name:<30} {fold_strs[0] if len(fold_strs) > 0 else 'N/A':>12} "
              f"{fold_strs[1] if len(fold_strs) > 1 else 'N/A':>12} "
              f"{fold_strs[2] if len(fold_strs) > 2 else 'N/A':>12} {result:>8}")

        wf_results[name] = {
            'folds': wf,
            'passes': passes,
            'total_oos_pnl': sum(f['oos_pnl'] for f in wf),
            'avg_oos_wr': round(np.mean([f['oos_wr'] for f in wf if f['oos_trades'] > 0]), 1),
            'verdict': 'PASS' if passes == 3 else 'FAIL',
        }

    results['sections']['wf_validation'] = wf_results

    # ═══════════════════════════════════════════════════════════
    # 6. REGIME TRANSITION HEATMAP — When to disable LONG
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("6. REGIME-ADAPTIVE RECOMMENDATION")
    print(f"{'='*70}")

    # Summarize: which detector combinations protect LONG losses most effectively?
    print("\n  IS Performance Summary (sorted by PnL/MDD):")
    is_sorted = sorted(scenario_results, key=lambda x: x['pnl_mdd'], reverse=True)
    for s in is_sorted[:8]:
        print(f"    {s['scenario']:<35} PnL/MDD {s['pnl_mdd']:>6.1f}x "
              f"PnL {s['pnl']:>+7.1f}% MDD {s['mdd']:>5.1f}%")

    # WF Summary
    print("\n  WF Results:")
    wf_sorted = sorted(wf_results.items(), key=lambda x: x[1]['total_oos_pnl'], reverse=True)
    for name, wf in wf_sorted:
        v = wf['verdict']
        pnl = wf['total_oos_pnl']
        wr = wf['avg_oos_wr']
        print(f"    {name:<30} {v:<6} OOS PnL {pnl:>+7.1f}% OOS WR {wr:.1f}%")

    # Final recommendation
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    best_wf_pass = [(n, w) for n, w in wf_sorted if w['verdict'] == 'PASS']
    if best_wf_pass:
        best_name, best_wf = best_wf_pass[0]
        print(f"\n  Best WF-PASS detector: {best_name}")
        print(f"    Total OOS PnL: {best_wf['total_oos_pnl']:+.1f}%")
        print(f"    Avg OOS WR: {best_wf['avg_oos_wr']:.1f}%")
    else:
        print("\n  No detector achieved 3/3 WF PASS")

    # SHORT-only assessment
    d9_wf = wf_results.get("D9: SHORT-only", {})
    d9_is = next((s for s in scenario_results if s['scenario'] == "D9: SHORT-only (all LONG off)"), None)
    if d9_is and d9_wf:
        print(f"\n  SHORT-only assessment:")
        print(f"    IS: {d9_is['trades']} trades, WR {d9_is['wr']}%, "
              f"PnL {d9_is['pnl']:+.1f}%, MDD {d9_is['mdd']:.1f}%, "
              f"PnL/MDD {d9_is['pnl_mdd']:.1f}x")
        print(f"    WF: {d9_wf['verdict']} — OOS PnL {d9_wf['total_oos_pnl']:+.1f}%, "
              f"OOS WR {d9_wf['avg_oos_wr']:.1f}%")

    results['sections']['recommendation'] = {
        'best_wf_pass': best_wf_pass[0][0] if best_wf_pass else None,
        'short_only_verdict': d9_wf.get('verdict') if d9_wf else None,
    }

    # ── Save ──
    elapsed = time.time() - t0
    results['meta']['elapsed_seconds'] = round(elapsed, 1)
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULT_FILE}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
