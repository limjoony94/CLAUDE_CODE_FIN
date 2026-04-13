#!/usr/bin/env python3
"""
C1 Breakout v2 — Deep Validation Suite (N=2 Positions)
========================================================
10 comprehensive tests for strategy robustness.

Strategy: 15-bar channel breakout on 15m BTC + body>40% + fractal SL + ATR trail TP
N=2 simultaneous positions, 50% margin each, 1x leverage
Fee = 0.10% round trip. Emergency SL = 3.0%.
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(line_buffering=True)

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings)
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

# ─── Constants ───
FEE_PCT = 0.10       # round trip
LEVERAGE = 1
EMERGENCY_SL = 3.0   # percent
MAX_POSITIONS = 2
SIZE_PCT = 50.0       # each position uses 50% of equity
TIMEOUT_BARS = 192

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                         'data', 'btc_5m_270days_reclassified.csv')


def load_15m_data():
    """Load 5m CSV and aggregate to 15m bars."""
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['group'] = df.index // 3
    agg = df.groupby('group').agg(
        timestamp=('timestamp', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index(drop=True)
    return agg


def precompute_indicators(opens, highs, lows, closes, n,
                          channel_period=15, atr_period=14):
    """Compute all indicators from production code."""
    h_list = highs.tolist()
    l_list = lows.tolist()
    c_list = closes.tolist()

    atr = compute_atr(h_list, l_list, c_list, atr_period)
    ch_high, ch_low = compute_channel(h_list, l_list, channel_period)
    sw_low, sw_high = compute_fractal_swings(h_list, l_list, lookback=10)

    return (np.array(atr), np.array(ch_high), np.array(ch_low),
            np.array(sw_low), np.array(sw_high))


def backtest_npos(opens, highs, lows, closes, timestamps, n,
                  channel_period=15, trail_K=2.5, max_sl_atr=3.3,
                  max_positions=2, size_pct=50.0, fee_pct=0.10,
                  extra_cost_pct=0.0):
    """N-position backtest with compound equity.

    Returns list of trade dicts with keys:
        entry_bar, exit_bar, direction, pnl, reason, bars_held, entry_price, exit_price
    """
    o = opens.astype(float)
    h = highs.astype(float)
    l = lows.astype(float)
    c = closes.astype(float)

    config = {
        'channel_period': channel_period,
        'body_min_ratio': 0.4,
        'atr_period': 14,
        'trail_K': trail_K,
        'max_sl_atr': max_sl_atr,
        'emergency_sl_pct': EMERGENCY_SL,
        'max_hold_bars': TIMEOUT_BARS,
        'sl_min_pct': 0.15,
        'sl_max_pct': 3.0,
        'min_bars_between': 2,
        'trail_activation_pct': 0.05,
    }
    sig = C1BreakoutSignal(config)

    atr_vals, ch_high, ch_low, sw_low, sw_high = precompute_indicators(
        o, h, l, c, n, channel_period, 14)

    # Positions: list of dicts
    positions = []
    trades = []
    total_fee = fee_pct + extra_cost_pct
    last_entry_bar = -10

    warmup = max(channel_period + 10, 25)

    for bar in range(warmup, n):
        # ── Exit logic for each open position ──
        closed_indices = []
        for idx, pos in enumerate(positions):
            ep = pos['entry_price']
            d = pos['direction']
            bh = bar - pos['entry_bar']

            # Update best price
            if d == 'LONG':
                pos['best_price'] = max(pos['best_price'], h[bar])
            else:
                pos['best_price'] = min(pos['best_price'], l[bar])

            exit_info = sig.check_exit(
                direction=d,
                entry_price=ep,
                best_price=pos['best_price'],
                current_high=h[bar],
                current_low=l[bar],
                current_close=c[bar],
                sl_price=pos['sl_price'],
                atr_val=atr_vals[bar] if not math.isnan(atr_vals[bar]) else 0,
                bars_held=bh
            )

            if exit_info is not None:
                exit_price = exit_info['exit_price']
                if d == 'LONG':
                    raw_pnl = (exit_price / ep - 1) * 100
                else:
                    raw_pnl = (1 - exit_price / ep) * 100

                trade_pnl = raw_pnl * LEVERAGE - total_fee

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
                closed_indices.append(idx)

        # Remove closed positions (reverse order)
        for idx in sorted(closed_indices, reverse=True):
            positions.pop(idx)

        # ── Entry logic ──
        if len(positions) < max_positions and bar + 1 < n:
            if bar - last_entry_bar < 2:
                continue

            if math.isnan(atr_vals[bar]) or atr_vals[bar] <= 0:
                continue

            entry_signal = sig.check_entry(
                bar_open=o[bar], bar_high=h[bar], bar_low=l[bar], bar_close=c[bar],
                channel_high=ch_high[bar], channel_low=ch_low[bar],
                atr_val=atr_vals[bar],
                last_swing_low=sw_low[bar], last_swing_high=sw_high[bar]
            )

            if entry_signal is not None:
                entry_price = o[bar + 1]  # next bar open
                direction = entry_signal['direction']

                # Compute SL at entry price (not signal bar close)
                if direction == 'LONG':
                    sw = sw_low[bar]
                    atr_sl = entry_price - max_sl_atr * atr_vals[bar]
                    fractal_sl = sw if not math.isnan(sw) else atr_sl
                    sl_price = max(fractal_sl, atr_sl)
                else:
                    sw = sw_high[bar]
                    atr_sl = entry_price + max_sl_atr * atr_vals[bar]
                    fractal_sl = sw if not math.isnan(sw) else atr_sl
                    sl_price = min(fractal_sl, atr_sl)

                # Validate SL distance
                sl_dist = abs(entry_price - sl_price) / entry_price * 100
                if sl_dist < 0.15 or sl_dist > 3.0:
                    continue

                positions.append({
                    'entry_bar': bar + 1,
                    'entry_price': entry_price,
                    'direction': direction,
                    'sl_price': sl_price,
                    'best_price': entry_price,
                })
                last_entry_bar = bar

    return trades


def calc_equity_stats(trades, n_bars, size_pct=50.0):
    """Calculate stats with compound equity from N-pos trades."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0,
                'avg_pnl': 0, 'avg_rr': 0, 'pnl_mdd': 0, 'n_days': 0,
                'trades_per_day': 0, 'daily_pnl': 0, 'avg_bars_held': 0,
                'win_avg': 0, 'loss_avg': 0}

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Compound equity with position sizing
    eq = 100.0
    peak = 100.0
    mdd = 0.0
    for t in sorted(trades, key=lambda x: x['exit_bar']):
        # Each trade uses size_pct of equity
        contribution = t['pnl'] * (size_pct / 100) * (eq / 100)
        eq += contribution
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        mdd = max(mdd, dd)

    total_pnl = eq - 100
    n_days = n_bars * 15 / 1440
    wr = len(wins) / len(trades) * 100
    wa = np.mean(wins) if wins else 0
    la = np.mean(losses) if losses else 0
    rr = abs(wa / la) if la != 0 else float('inf')

    return {
        'trades': len(trades),
        'wr': round(wr, 2),
        'pnl': round(total_pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(total_pnl / mdd, 2) if mdd > 0 else 0,
        'avg_pnl': round(np.mean(pnls), 4),
        'avg_rr': round(rr, 2),
        'win_avg': round(wa, 4),
        'loss_avg': round(la, 4),
        'trades_per_day': round(len(trades) / n_days, 2) if n_days > 0 else 0,
        'daily_pnl': round(total_pnl / n_days, 4) if n_days > 0 else 0,
        'avg_bars_held': round(np.mean([t['bars_held'] for t in trades]), 1),
        'n_days': round(n_days, 1),
    }


# ═══════════════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════════════

def test_1_progressive_lookahead(df):
    """Progressive look-ahead test: 25%, 50%, 75%, 100% of data."""
    print("\n" + "="*70)
    print("  TEST 1: Progressive Look-Ahead")
    print("="*70)

    n_full = len(df)
    fractions = [0.25, 0.50, 0.75, 1.00]
    results = {}

    for frac in fractions:
        end = int(n_full * frac)
        sub = df.iloc[:end].copy().reset_index(drop=True)
        n = len(sub)
        trades = backtest_npos(
            sub['open'].values, sub['high'].values, sub['low'].values,
            sub['close'].values, sub['timestamp'].values, n)
        stats = calc_equity_stats(trades, n)
        results[frac] = {'trades': trades, 'stats': stats, 'n_bars': n}
        print(f"  {frac*100:5.0f}% ({n:>5} bars): T={stats['trades']:>4}  "
              f"WR={stats['wr']:>5.1f}%  R:R={stats['avg_rr']:>5.2f}  "
              f"PnL={stats['pnl']:>+8.1f}%  MDD={stats['mdd']:>5.1f}%")

    # Check consistency: trades from 25% period must match in 50%, etc.
    # Compare trades in overlapping regions
    consistent = True
    for i in range(len(fractions) - 1):
        f_small = fractions[i]
        f_large = fractions[i + 1]
        small_end = int(n_full * f_small)

        t_small = results[f_small]['trades']
        t_large = results[f_large]['trades']

        # Count trades that start in the overlapping period
        small_count = len([t for t in t_small if t['entry_bar'] < small_end])
        large_in_period = len([t for t in t_large if t['entry_bar'] < small_end])

        match = small_count == large_in_period
        if not match:
            consistent = False
        print(f"  Overlap {f_small*100:.0f}%→{f_large*100:.0f}%: "
              f"small={small_count}, large_in_period={large_in_period} "
              f"{'MATCH' if match else 'MISMATCH'}")

    passed = consistent
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (consistency={consistent})")
    return passed


def test_2_walk_forward(df):
    """Walk-Forward 5-fold expanding window."""
    print("\n" + "="*70)
    print("  TEST 2: Walk-Forward 5-Fold Expanding Window")
    print("="*70)

    n = len(df)
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    ts = df['timestamp'].values
    folds = []

    for fi in range(5):
        is_end = int(n * (fi + 1) / (5 + 1))
        oos_end = int(n * (fi + 2) / (5 + 1))

        # IS: [0, is_end)
        is_trades = backtest_npos(o[:is_end], h[:is_end], l[:is_end],
                                  c[:is_end], ts[:is_end], is_end)
        is_stats = calc_equity_stats(is_trades, is_end)

        # OOS: [is_end, oos_end)
        oos_n = oos_end - is_end
        oos_trades = backtest_npos(
            o[is_end:oos_end], h[is_end:oos_end], l[is_end:oos_end],
            c[is_end:oos_end], ts[is_end:oos_end], oos_n)
        oos_stats = calc_equity_stats(oos_trades, oos_n)

        folds.append({'is': is_stats, 'oos': oos_stats})
        p = 'P' if oos_stats['pnl'] > 0 else 'F'
        print(f"  F{fi+1}: IS [{0:>5}:{is_end:>5}] T={is_stats['trades']:>3} "
              f"WR={is_stats['wr']:>5.1f}% PnL={is_stats['pnl']:>+7.1f}% | "
              f"OOS [{is_end:>5}:{oos_end:>5}] T={oos_stats['trades']:>3} "
              f"WR={oos_stats['wr']:>5.1f}% R:R={oos_stats['avg_rr']:>5.2f} "
              f"PnL={oos_stats['pnl']:>+7.1f}% [{p}]")

    oos_pass = sum(1 for f in folds if f['oos']['pnl'] > 0)
    total_oos = sum(f['oos']['pnl'] for f in folds)
    print(f"\n  OOS Total: {total_oos:+.1f}%  |  {oos_pass}/5 PASS")

    passed = oos_pass >= 3
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} ({oos_pass}/5 >= 3 required)")
    return passed


def test_3_three_way_split(df):
    """3-Way Split: Train/Validate/Test."""
    print("\n" + "="*70)
    print("  TEST 3: 3-Way Split (Train 0-50%, Valid 50-75%, Test 75-100%)")
    print("="*70)

    n = len(df)
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    ts = df['timestamp'].values

    splits = [
        ('Train', 0, int(n * 0.50)),
        ('Valid', int(n * 0.50), int(n * 0.75)),
        ('Test', int(n * 0.75), n),
    ]

    all_positive = True
    for label, start, end in splits:
        seg_n = end - start
        trades = backtest_npos(o[start:end], h[start:end], l[start:end],
                               c[start:end], ts[start:end], seg_n)
        stats = calc_equity_stats(trades, seg_n)
        pos = stats['pnl'] > 0
        if not pos:
            all_positive = False
        print(f"  {label:>5}: [{start:>5}:{end:>5}] T={stats['trades']:>4} "
              f"WR={stats['wr']:>5.1f}% R:R={stats['avg_rr']:>5.2f} "
              f"PnL={stats['pnl']:>+8.1f}% Daily={stats['daily_pnl']:>+7.4f}% "
              f"{'PASS' if pos else 'FAIL'}")

    print(f"\n  RESULT: {'PASS' if all_positive else 'FAIL'} (all 3 must be positive)")
    return all_positive


def test_4_monte_carlo(df, baseline_trades):
    """Monte Carlo direction shuffle (999 sims)."""
    print("\n" + "="*70)
    print("  TEST 4: Monte Carlo Direction Shuffle (999 sims)")
    print("="*70)

    if not baseline_trades:
        print("  No trades to shuffle. FAIL")
        return False

    real_pnls = [t['pnl'] for t in baseline_trades]
    # Calculate real compound PnL
    eq = 100.0
    for t in sorted(baseline_trades, key=lambda x: x['exit_bar']):
        eq += t['pnl'] * (SIZE_PCT / 100) * (eq / 100)
    real_pnl = eq - 100

    better = 0
    shuffled_pnls = []
    seeds = [42, 123, 7]
    sims_per_seed = 333

    for seed in seeds:
        rng = np.random.RandomState(seed)
        for _ in range(sims_per_seed):
            eq = 100.0
            for t in sorted(baseline_trades, key=lambda x: x['exit_bar']):
                if rng.random() > 0.5:
                    p = -t['pnl'] - 2 * FEE_PCT
                else:
                    p = t['pnl']
                eq += p * (SIZE_PCT / 100) * (eq / 100)
            s_pnl = eq - 100
            shuffled_pnls.append(s_pnl)
            if s_pnl >= real_pnl:
                better += 1

    p_val = better / len(shuffled_pnls)
    print(f"  Real PnL:      {real_pnl:>+8.1f}%")
    print(f"  Shuffled mean: {np.mean(shuffled_pnls):>+8.1f}%")
    print(f"  Shuffled std:  {np.std(shuffled_pnls):>8.1f}%")
    print(f"  p-value:       {p_val:.4f}")
    print(f"  Sims >= real:  {better}/{len(shuffled_pnls)}")

    passed = p_val < 0.05
    print(f"\n  RESULT: {'PASS (DISCRIMINATING)' if passed else 'FAIL (NON-DISCRIMINATING)'}")
    return passed


def test_5_parameter_sensitivity(df):
    """Parameter sensitivity +/- 20% sweep."""
    print("\n" + "="*70)
    print("  TEST 5: Parameter Sensitivity (+/-20%)")
    print("="*70)

    n = len(df)
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    ts = df['timestamp'].values

    # Base params
    base_channel = 15
    base_trail = 2.5
    base_sl_atr = 3.3

    # Sweep ranges: base +/- 20%
    channels = [12, 13, 14, 15, 16, 17, 18]
    trail_Ks = [2.0, 2.2, 2.5, 2.8, 3.0]
    sl_atrs = [2.6, 2.8, 3.0, 3.3, 3.6, 4.0]

    print(f"\n  Channel sweep (trail_K={base_trail}, max_sl_atr={base_sl_atr}):")
    all_positive = True
    for ch in channels:
        trades = backtest_npos(o, h, l, c, ts, n, channel_period=ch)
        stats = calc_equity_stats(trades, n)
        pos = stats['pnl'] > 0
        if not pos:
            all_positive = False
        marker = " <-- BASE" if ch == base_channel else ""
        print(f"    ch={ch:>2}: T={stats['trades']:>4} WR={stats['wr']:>5.1f}% "
              f"PnL={stats['pnl']:>+8.1f}% {'OK' if pos else 'NEG'}{marker}")

    print(f"\n  Trail_K sweep (channel={base_channel}, max_sl_atr={base_sl_atr}):")
    for tk in trail_Ks:
        trades = backtest_npos(o, h, l, c, ts, n, trail_K=tk)
        stats = calc_equity_stats(trades, n)
        pos = stats['pnl'] > 0
        if not pos:
            all_positive = False
        marker = " <-- BASE" if tk == base_trail else ""
        print(f"    K={tk:.1f}: T={stats['trades']:>4} WR={stats['wr']:>5.1f}% "
              f"PnL={stats['pnl']:>+8.1f}% {'OK' if pos else 'NEG'}{marker}")

    print(f"\n  Max SL ATR sweep (channel={base_channel}, trail_K={base_trail}):")
    for sa in sl_atrs:
        trades = backtest_npos(o, h, l, c, ts, n, max_sl_atr=sa)
        stats = calc_equity_stats(trades, n)
        pos = stats['pnl'] > 0
        if not pos:
            all_positive = False
        marker = " <-- BASE" if sa == base_sl_atr else ""
        print(f"    sl={sa:.1f}: T={stats['trades']:>4} WR={stats['wr']:>5.1f}% "
              f"PnL={stats['pnl']:>+8.1f}% {'OK' if pos else 'NEG'}{marker}")

    print(f"\n  RESULT: {'PASS' if all_positive else 'FAIL'} (all combos must be positive)")
    return all_positive


def test_6_monthly_breakdown(df, trades):
    """Monthly PnL breakdown."""
    print("\n" + "="*70)
    print("  TEST 6: Monthly Breakdown")
    print("="*70)

    if not trades:
        print("  No trades. FAIL")
        return False

    timestamps = df['timestamp'].values
    monthly = defaultdict(list)

    for t in trades:
        bar_idx = min(t['exit_bar'], len(timestamps) - 1)
        ts = pd.Timestamp(timestamps[bar_idx])
        key = ts.strftime('%Y-%m')
        monthly[key].append(t['pnl'])

    n_positive = 0
    total_months = 0
    print(f"  {'Month':<10} {'Trades':>6} {'WR':>6} {'PnL':>8} {'AvgPnL':>8}")
    print(f"  {'-'*40}")
    for month in sorted(monthly.keys()):
        pnls = monthly[month]
        t_count = len(pnls)
        wr = sum(1 for p in pnls if p > 0) / t_count * 100 if t_count else 0
        total = sum(pnls)
        avg = np.mean(pnls)
        total_months += 1
        if total > 0:
            n_positive += 1
        print(f"  {month:<10} {t_count:>6} {wr:>5.1f}% {total:>+7.2f}% {avg:>+7.3f}%")

    ratio = n_positive / total_months if total_months else 0
    print(f"\n  Positive months: {n_positive}/{total_months} ({ratio*100:.0f}%)")

    passed = ratio >= 0.5
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} (>= 50% positive months required)")
    return passed


def test_7_rolling_60day(df, trades):
    """Rolling 60-day windows."""
    print("\n" + "="*70)
    print("  TEST 7: Rolling 60-Day Windows")
    print("="*70)

    if not trades:
        print("  No trades. FAIL")
        return False

    timestamps = df['timestamp'].values
    bars_per_day = 96  # 24h / 15min
    window_bars = 60 * bars_per_day
    step_bars = 30 * bars_per_day  # 30-day step

    n = len(df)
    n_windows = 0
    n_positive = 0
    window_results = []

    start = 0
    while start + window_bars <= n:
        end = start + window_bars
        # Find trades within this window
        window_trades = [t for t in trades
                         if start <= t['entry_bar'] < end and t['exit_bar'] < end]

        if window_trades:
            eq = 100.0
            for t in sorted(window_trades, key=lambda x: x['exit_bar']):
                eq += t['pnl'] * (SIZE_PCT / 100) * (eq / 100)
            w_pnl = eq - 100
        else:
            w_pnl = 0.0

        ts_start = pd.Timestamp(timestamps[start])
        ts_end = pd.Timestamp(timestamps[min(end - 1, n - 1)])
        n_windows += 1
        if w_pnl > 0:
            n_positive += 1
        window_results.append(w_pnl)

        print(f"  {ts_start.strftime('%Y-%m-%d')}~{ts_end.strftime('%Y-%m-%d')}: "
              f"T={len(window_trades):>3} PnL={w_pnl:>+7.2f}% "
              f"{'POS' if w_pnl > 0 else 'NEG'}")

        start += step_bars

    ratio = n_positive / n_windows if n_windows else 0
    print(f"\n  Positive windows: {n_positive}/{n_windows} ({ratio*100:.0f}%)")
    print(f"  Mean PnL:  {np.mean(window_results):+.2f}%")
    print(f"  Worst:     {min(window_results):+.2f}%")
    print(f"  Best:      {max(window_results):+.2f}%")

    passed = ratio >= 0.5
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} (>= 50% positive windows)")
    return passed


def test_8_fee_stress(df):
    """Fee/slippage stress test."""
    print("\n" + "="*70)
    print("  TEST 8: Fee/Slippage Stress Test")
    print("="*70)

    n = len(df)
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    ts = df['timestamp'].values

    extra_costs = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    breakeven_cost = None

    for extra in extra_costs:
        trades = backtest_npos(o, h, l, c, ts, n, extra_cost_pct=extra)
        stats = calc_equity_stats(trades, n)
        pos = stats['pnl'] > 0
        if not pos and breakeven_cost is None:
            breakeven_cost = extra
        total_fee = FEE_PCT + extra
        print(f"  Extra={extra:.2f}% (total={total_fee:.2f}%): T={stats['trades']:>4} "
              f"PnL={stats['pnl']:>+8.1f}% {'OK' if pos else 'NEG'}")

    if breakeven_cost is None:
        breakeven_cost = 0.15
        print(f"\n  Breakeven extra cost: >{breakeven_cost:.2f}% (robust)")
    else:
        print(f"\n  Breakeven extra cost: ~{breakeven_cost:.2f}%")

    passed = breakeven_cost >= 0.06
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} (breakeven >= 0.06% extra required)")
    return passed


def test_9_consecutive_losses(trades):
    """Consecutive loss analysis."""
    print("\n" + "="*70)
    print("  TEST 9: Consecutive Loss Analysis")
    print("="*70)

    if not trades:
        print("  No trades. FAIL")
        return False

    sorted_trades = sorted(trades, key=lambda x: x['exit_bar'])

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    consec_start = 0
    worst_consec_start = 0
    worst_consec_end = 0

    for i, t in enumerate(sorted_trades):
        if t['pnl'] <= 0:
            if current_consec == 0:
                consec_start = i
            current_consec += 1
            if current_consec > max_consec:
                max_consec = current_consec
                worst_consec_start = consec_start
                worst_consec_end = i
        else:
            current_consec = 0

    # Worst drawdown period (compound)
    eq = 100.0
    peak = 100.0
    mdd = 0.0
    mdd_peak_idx = 0
    mdd_trough_idx = 0
    cur_peak_idx = 0

    for i, t in enumerate(sorted_trades):
        eq += t['pnl'] * (SIZE_PCT / 100) * (eq / 100)
        if eq > peak:
            peak = eq
            cur_peak_idx = i
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > mdd:
            mdd = dd
            mdd_peak_idx = cur_peak_idx
            mdd_trough_idx = i

    print(f"  Max consecutive losses: {max_consec}")
    if max_consec > 0:
        loss_pnls = [sorted_trades[j]['pnl'] for j in range(worst_consec_start, worst_consec_end + 1)]
        print(f"    Streak PnLs: {[f'{p:+.2f}%' for p in loss_pnls]}")
        print(f"    Total damage: {sum(loss_pnls):+.2f}%")

    print(f"\n  Max Drawdown: {mdd:.2f}%")
    print(f"    Peak trade #{mdd_peak_idx} → Trough trade #{mdd_trough_idx}")
    dd_trades = mdd_trough_idx - mdd_peak_idx
    print(f"    Drawdown span: {dd_trades} trades")

    # Recovery analysis
    eq = 100.0
    peak = 100.0
    in_dd = False
    dd_start = None
    max_recovery = 0

    for i, t in enumerate(sorted_trades):
        eq += t['pnl'] * (SIZE_PCT / 100) * (eq / 100)
        if eq >= peak:
            if in_dd and dd_start is not None:
                recovery = i - dd_start
                max_recovery = max(max_recovery, recovery)
            peak = eq
            in_dd = False
            dd_start = None
        else:
            if not in_dd:
                in_dd = True
                dd_start = i

    print(f"  Longest recovery: {max_recovery} trades")

    passed = max_consec <= 15 and mdd <= 30.0
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} "
          f"(max_consec<={15}, MDD<={30}%)")
    return passed


def test_10_direction_breakdown(trades):
    """LONG vs SHORT performance."""
    print("\n" + "="*70)
    print("  TEST 10: Direction Breakdown (LONG vs SHORT)")
    print("="*70)

    if not trades:
        print("  No trades. FAIL")
        return False

    long_trades = [t for t in trades if t['direction'] == 'LONG']
    short_trades = [t for t in trades if t['direction'] == 'SHORT']

    all_positive = True
    for label, subset in [('LONG', long_trades), ('SHORT', short_trades)]:
        if not subset:
            print(f"  {label}: No trades")
            all_positive = False
            continue

        pnls = [t['pnl'] for t in subset]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(subset) * 100
        total = sum(pnls)
        wa = np.mean(wins) if wins else 0
        la = np.mean(losses) if losses else 0
        rr = abs(wa / la) if la != 0 else float('inf')
        pos = total > 0

        # Exit breakdown
        reasons = defaultdict(int)
        for t in subset:
            reasons[t['reason']] += 1

        if not pos:
            all_positive = False

        print(f"\n  {label}:")
        print(f"    Trades:   {len(subset)}")
        print(f"    WR:       {wr:.1f}%")
        print(f"    R:R:      {rr:.2f}")
        print(f"    Sum PnL:  {total:+.2f}%")
        print(f"    Avg PnL:  {np.mean(pnls):+.4f}%")
        print(f"    Win avg:  {wa:+.4f}%")
        print(f"    Loss avg: {la:+.4f}%")
        print(f"    Avg bars: {np.mean([t['bars_held'] for t in subset]):.1f}")
        exit_str = ', '.join([f"{k}={v}" for k, v in sorted(reasons.items())])
        print(f"    Exits:    {exit_str}")
        print(f"    Status:   {'POS' if pos else 'NEG'}")

    print(f"\n  RESULT: {'PASS' if all_positive else 'FAIL'} (both directions positive)")
    return all_positive


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  C1 Breakout v2 — Deep Validation Suite (N=2)")
    print("  Fee=0.10% | Leverage=1x | Size=50%/pos | Emergency SL=3.0%")
    print("=" * 70)

    # Load data
    df = load_15m_data()
    n = len(df)
    n_days = n * 15 / 1440
    print(f"\n  Data: {n} bars ({n_days:.0f} days)")
    print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # ── Baseline ──
    print("\n" + "=" * 70)
    print("  BASELINE (N=2, ch=15, trail_K=2.5, max_sl_atr=3.3)")
    print("=" * 70)

    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    ts = df['timestamp'].values

    baseline_trades = backtest_npos(o, h, l, c, ts, n)
    baseline_stats = calc_equity_stats(baseline_trades, n)

    print(f"  Trades:     {baseline_stats['trades']}")
    print(f"  WR:         {baseline_stats['wr']}%")
    print(f"  R:R:        {baseline_stats['avg_rr']}")
    print(f"  PnL:        {baseline_stats['pnl']:+.2f}%")
    print(f"  MDD:        {baseline_stats['mdd']:.2f}%")
    print(f"  PnL/MDD:    {baseline_stats['pnl_mdd']}")
    print(f"  Trades/day: {baseline_stats['trades_per_day']}")
    print(f"  Daily PnL:  {baseline_stats['daily_pnl']:+.4f}%")
    print(f"  Avg bars:   {baseline_stats['avg_bars_held']}")

    # Exit breakdown
    reasons = defaultdict(int)
    for t in baseline_trades:
        reasons[t['reason']] += 1
    print(f"  Exits: {dict(sorted(reasons.items()))}")

    # ── Run all 10 tests ──
    results = {}
    results['T1_LookAhead'] = test_1_progressive_lookahead(df)
    results['T2_WalkForward'] = test_2_walk_forward(df)
    results['T3_ThreeWay'] = test_3_three_way_split(df)
    results['T4_MonteCarlo'] = test_4_monte_carlo(df, baseline_trades)
    results['T5_ParamSensitivity'] = test_5_parameter_sensitivity(df)
    results['T6_Monthly'] = test_6_monthly_breakdown(df, baseline_trades)
    results['T7_Rolling60d'] = test_7_rolling_60day(df, baseline_trades)
    results['T8_FeeStress'] = test_8_fee_stress(df)
    results['T9_ConsecLoss'] = test_9_consecutive_losses(baseline_trades)
    results['T10_Direction'] = test_10_direction_breakdown(baseline_trades)

    # ═══════════════════════════════════════════════════════════
    #  CRITERIA CHECK
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  CRITERIA CHECK — C1 Breakout v2 Deep Validation")
    print("=" * 70)

    criteria = [
        ("T1  Progressive Look-Ahead (consistency)", results['T1_LookAhead']),
        ("T2  Walk-Forward 5-fold (>=3/5 OOS positive)", results['T2_WalkForward']),
        ("T3  3-Way Split (all positive)", results['T3_ThreeWay']),
        ("T4  Monte Carlo (p < 0.05)", results['T4_MonteCarlo']),
        ("T5  Param Sensitivity (all positive)", results['T5_ParamSensitivity']),
        ("T6  Monthly (>= 50% positive)", results['T6_Monthly']),
        ("T7  Rolling 60-day (>= 50% positive)", results['T7_Rolling60d']),
        ("T8  Fee Stress (breakeven >= 0.06%)", results['T8_FeeStress']),
        ("T9  Consecutive Losses (max<=15, MDD<=30%)", results['T9_ConsecLoss']),
        ("T10 Direction (both LONG & SHORT positive)", results['T10_Direction']),
    ]

    passed = 0
    failed = 0
    for desc, result in criteria:
        status = "PASS" if result else "FAIL"
        symbol = "[v]" if result else "[x]"
        print(f"  {symbol} {desc}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n  Score: {passed}/{len(criteria)} PASSED, {failed} FAILED")

    if passed == len(criteria):
        verdict = "ALL PASS — Strategy validated"
    elif passed >= 7:
        verdict = "MOSTLY PASS — Minor concerns"
    elif passed >= 5:
        verdict = "MIXED — Significant issues"
    else:
        verdict = "MOSTLY FAIL — Strategy needs rework"

    print(f"  Verdict: {verdict}")
    print("=" * 70)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_data = {
        'baseline': baseline_stats,
        'criteria': {desc: result for desc, result in criteria},
        'score': f"{passed}/{len(criteria)}",
        'verdict': verdict,
    }
    save_path = os.path.join(results_dir, 'c1_v2_deep_validation.json')
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {save_path}")


if __name__ == '__main__':
    main()
