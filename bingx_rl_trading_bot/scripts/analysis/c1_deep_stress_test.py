#!/usr/bin/env python3
"""
C1 Breakout v2 — Deep Stress Tests (New Angles)
=================================================
10 new verification angles that haven't been tested yet.

Strategy: 15m Channel Breakout + Fractal SL + ATR Trailing TP
N=1, additive PnL, Fee=0.10% RT, no leverage multiplier.

Tests:
  1. Slippage Sensitivity
  2. MAE/MFE Analysis
  3. Time-of-Day Analysis
  4. Day-of-Week Analysis
  5. Drawdown Duration Analysis
  6. Consecutive Loss Analysis
  7. Trade Duration Distribution
  8. Bollinger Width Filter + WF + MC
  9. Profit Factor Stability
 10. Entry Price vs Channel Distance
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(line_buffering=True)

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

# ─── Constants ───
FEE_PCT = 0.10       # round trip
LEVERAGE = 1          # additive, no leverage multiplier
EMERGENCY_SL = 3.0
TIMEOUT_BARS = 192
CHANNEL_PERIOD = 15
ATR_PERIOD = 14
TRAIL_K = 2.5
MAX_SL_ATR = 3.3
SL_MIN_PCT = 0.15
SL_MAX_PCT = 3.0
MIN_BARS_BETWEEN = 2
TRAIL_ACTIVATION_PCT = 0.05

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                         'data', 'btc_5m_270days_reclassified.csv')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                            'results', 'c1_deep_stress_test.json')


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


def precompute_indicators(highs, lows, closes, n):
    """Compute all indicators from production code."""
    h_list = highs.tolist()
    l_list = lows.tolist()
    c_list = closes.tolist()

    atr = compute_atr(h_list, l_list, c_list, ATR_PERIOD)
    ch_high, ch_low = compute_channel(h_list, l_list, CHANNEL_PERIOD)
    sw_low, sw_high = compute_fractal_swings(h_list, l_list, lookback=10)

    return (np.array(atr), np.array(ch_high), np.array(ch_low),
            np.array(sw_low), np.array(sw_high))


def compute_bollinger_width(closes, period=20, std_mult=2.0):
    """Compute Bollinger Band width = (upper - lower) / middle."""
    n = len(closes)
    bb_width = [float('nan')] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper = mean + std_mult * std
        lower = mean - std_mult * std
        if mean > 0:
            bb_width[i] = (upper - lower) / mean * 100  # as percentage
    return bb_width


def backtest_enriched(o, h, l, c, timestamps, n, atr_vals, ch_high, ch_low,
                      sw_low, sw_high, extra_cost_pct=0.0,
                      bb_width=None, bb_threshold=None):
    """Single-position backtest returning enriched trade dicts.

    Each trade includes: entry_bar, exit_bar, direction, pnl, reason,
    bars_held, entry_price, exit_price, entry_ts, mae, mfe,
    channel_distance_atr, sl_pct.
    """
    config = {
        'channel_period': CHANNEL_PERIOD,
        'body_min_ratio': 0.4,
        'atr_period': ATR_PERIOD,
        'trail_K': TRAIL_K,
        'max_sl_atr': MAX_SL_ATR,
        'emergency_sl_pct': EMERGENCY_SL,
        'max_hold_bars': TIMEOUT_BARS,
        'sl_min_pct': SL_MIN_PCT,
        'sl_max_pct': SL_MAX_PCT,
        'min_bars_between': MIN_BARS_BETWEEN,
        'trail_activation_pct': TRAIL_ACTIVATION_PCT,
    }
    sig = C1BreakoutSignal(config)

    total_fee = FEE_PCT + extra_cost_pct
    trades = []
    position = None
    last_entry_bar = -10
    warmup = max(CHANNEL_PERIOD + 10, 25)

    for bar in range(warmup, n):
        # ── Exit logic ──
        if position is not None:
            ep = position['entry_price']
            d = position['direction']
            bh = bar - position['entry_bar']

            # Update best/worst price for MAE/MFE
            if d == 'LONG':
                position['best_price'] = max(position['best_price'], h[bar])
                position['worst_price'] = min(position['worst_price'], l[bar])
                cur_mfe = (position['best_price'] / ep - 1) * 100
                cur_mae = (position['worst_price'] / ep - 1) * 100  # negative
            else:
                position['best_price'] = min(position['best_price'], l[bar])
                position['worst_price'] = max(position['worst_price'], h[bar])
                cur_mfe = (1 - position['best_price'] / ep) * 100
                cur_mae = (1 - position['worst_price'] / ep) * 100  # negative

            exit_info = sig.check_exit(
                direction=d,
                entry_price=ep,
                best_price=position['best_price'],
                current_high=h[bar],
                current_low=l[bar],
                current_close=c[bar],
                sl_price=position['sl_price'],
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
                    'entry_bar': position['entry_bar'],
                    'exit_bar': bar,
                    'direction': d,
                    'pnl': trade_pnl,
                    'reason': exit_info['reason'],
                    'bars_held': bh,
                    'entry_price': ep,
                    'exit_price': exit_price,
                    'entry_ts': str(position['entry_ts']),
                    'mae': cur_mae,  # worst unrealized (negative for adverse)
                    'mfe': cur_mfe,  # best unrealized (positive for favorable)
                    'channel_distance_atr': position.get('channel_distance_atr', 0),
                    'sl_pct': position.get('sl_pct', 0),
                })
                position = None

        # ── Entry logic (N=1) ──
        if position is None and bar + 1 < n:
            if bar - last_entry_bar < MIN_BARS_BETWEEN:
                continue

            if math.isnan(atr_vals[bar]) or atr_vals[bar] <= 0:
                continue

            # BB width filter (Test 8)
            if bb_width is not None and bb_threshold is not None:
                if math.isnan(bb_width[bar]) or bb_width[bar] < bb_threshold:
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

                # Compute SL at entry price
                if direction == 'LONG':
                    sw = sw_low[bar]
                    atr_sl = entry_price - MAX_SL_ATR * atr_vals[bar]
                    fractal_sl = sw if not math.isnan(sw) else atr_sl
                    sl_price = max(fractal_sl, atr_sl)
                else:
                    sw = sw_high[bar]
                    atr_sl = entry_price + MAX_SL_ATR * atr_vals[bar]
                    fractal_sl = sw if not math.isnan(sw) else atr_sl
                    sl_price = min(fractal_sl, atr_sl)

                sl_dist = abs(entry_price - sl_price) / entry_price * 100
                if sl_dist < SL_MIN_PCT or sl_dist > SL_MAX_PCT:
                    continue

                # Channel distance in ATR units
                if direction == 'LONG':
                    ch_dist = (c[bar] - ch_high[bar]) / atr_vals[bar] if atr_vals[bar] > 0 else 0
                else:
                    ch_dist = (ch_low[bar] - c[bar]) / atr_vals[bar] if atr_vals[bar] > 0 else 0

                position = {
                    'entry_bar': bar + 1,
                    'entry_price': entry_price,
                    'direction': direction,
                    'sl_price': sl_price,
                    'best_price': entry_price,
                    'worst_price': entry_price,
                    'entry_ts': timestamps[bar + 1],
                    'channel_distance_atr': ch_dist,
                    'sl_pct': sl_dist,
                }
                last_entry_bar = bar

    return trades


# ═══════════════════════════════════════════════════════════
# Test 1: Slippage Sensitivity
# ═══════════════════════════════════════════════════════════
def test_slippage_sensitivity(o, h, l, c, timestamps, n, atr_vals, ch_high, ch_low, sw_low, sw_high):
    print("\n" + "="*60)
    print("TEST 1: Slippage Sensitivity")
    print("="*60)

    slippage_levels = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    results = []

    for slip in slippage_levels:
        trades = backtest_enriched(o, h, l, c, timestamps, n,
                                   atr_vals, ch_high, ch_low, sw_low, sw_high,
                                   extra_cost_pct=slip)
        pnls = [t['pnl'] for t in trades]
        total_pnl = sum(pnls)
        n_trades = len(trades)
        wr = sum(1 for p in pnls if p > 0) / n_trades * 100 if n_trades > 0 else 0
        avg_pnl = total_pnl / n_trades if n_trades > 0 else 0

        results.append({
            'slippage_pct': slip,
            'total_pnl': round(total_pnl, 2),
            'trades': n_trades,
            'wr': round(wr, 2),
            'avg_pnl': round(avg_pnl, 4),
        })
        print(f"  Slip {slip:6.2f}% | PnL {total_pnl:+8.1f}% | Trades {n_trades:4d} | WR {wr:5.1f}% | Avg {avg_pnl:+.4f}%")

    # Find break-even slippage
    base_pnl = results[0]['total_pnl']
    breakeven = None
    for i in range(1, len(results)):
        if results[i]['total_pnl'] <= 0 and results[i-1]['total_pnl'] > 0:
            # Linear interpolation
            p1 = results[i-1]['total_pnl']
            p2 = results[i]['total_pnl']
            s1 = results[i-1]['slippage_pct']
            s2 = results[i]['slippage_pct']
            breakeven = s1 + (0 - p1) / (p2 - p1) * (s2 - s1)
            break
    if breakeven is None and results[-1]['total_pnl'] > 0:
        breakeven = "> 0.30%"

    print(f"\n  Break-even slippage: {breakeven}")
    return {'slippage_levels': results, 'breakeven_slippage': breakeven}


# ═══════════════════════════════════════════════════════════
# Test 2: MAE/MFE Analysis
# ═══════════════════════════════════════════════════════════
def test_mae_mfe(trades):
    print("\n" + "="*60)
    print("TEST 2: MAE/MFE Analysis")
    print("="*60)

    maes = [t['mae'] for t in trades]
    mfes = [t['mfe'] for t in trades]
    pnls = [t['pnl'] for t in trades]

    mae_arr = np.array(maes)
    mfe_arr = np.array(mfes)
    pnl_arr = np.array(pnls)

    # MAE percentiles
    mae_pcts = np.percentile(mae_arr, [5, 10, 25, 50, 75, 90, 95])
    mfe_pcts = np.percentile(mfe_arr, [5, 10, 25, 50, 75, 90, 95])

    print(f"  MAE (worst unrealized loss) percentiles:")
    for p, v in zip([5, 10, 25, 50, 75, 90, 95], mae_pcts):
        print(f"    P{p:2d}: {v:+.4f}%")

    print(f"\n  MFE (best unrealized profit) percentiles:")
    for p, v in zip([5, 10, 25, 50, 75, 90, 95], mfe_pcts):
        print(f"    P{p:2d}: {v:+.4f}%")

    # Correlation MAE vs PnL
    corr_mae_pnl = float(np.corrcoef(mae_arr, pnl_arr)[0, 1])
    corr_mfe_pnl = float(np.corrcoef(mfe_arr, pnl_arr)[0, 1])

    print(f"\n  Correlation MAE vs PnL: {corr_mae_pnl:.4f}")
    print(f"  Correlation MFE vs PnL: {corr_mfe_pnl:.4f}")

    # Split winners/losers
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]

    win_mae = np.median([t['mae'] for t in winners]) if winners else 0
    loss_mae = np.median([t['mae'] for t in losers]) if losers else 0
    win_mfe = np.median([t['mfe'] for t in winners]) if winners else 0
    loss_mfe = np.median([t['mfe'] for t in losers]) if losers else 0

    print(f"\n  Winners (n={len(winners)}) — median MAE: {win_mae:+.4f}%, median MFE: {win_mfe:+.4f}%")
    print(f"  Losers  (n={len(losers)})  — median MAE: {loss_mae:+.4f}%, median MFE: {loss_mfe:+.4f}%")

    # MFE-to-capture ratio (how much of MFE did we actually capture? — winners only)
    win_capture = []
    for t in winners:
        if t['mfe'] > 0:
            win_capture.append(t['pnl'] / t['mfe'])
    avg_win_capture = np.mean(win_capture) if win_capture else 0

    # All trades: what fraction of MFE was realized as PnL
    all_capture = []
    for t in trades:
        if t['mfe'] > 0.01:  # avoid division by near-zero
            all_capture.append(t['pnl'] / t['mfe'])
    avg_all_capture = np.median(all_capture) if all_capture else 0

    print(f"  Winners MFE capture ratio: {avg_win_capture:.3f} (1.0=perfect)")
    print(f"  All trades median MFE capture: {avg_all_capture:.3f}")

    return {
        'mae_percentiles': {f'P{p}': round(float(v), 4) for p, v in zip([5,10,25,50,75,90,95], mae_pcts)},
        'mfe_percentiles': {f'P{p}': round(float(v), 4) for p, v in zip([5,10,25,50,75,90,95], mfe_pcts)},
        'corr_mae_pnl': round(corr_mae_pnl, 4),
        'corr_mfe_pnl': round(corr_mfe_pnl, 4),
        'winners_median_mae': round(float(win_mae), 4),
        'losers_median_mae': round(float(loss_mae), 4),
        'winners_median_mfe': round(float(win_mfe), 4),
        'losers_median_mfe': round(float(loss_mfe), 4),
        'winners_mfe_capture_ratio': round(float(avg_win_capture), 3),
        'all_median_mfe_capture': round(float(avg_all_capture), 3),
    }


# ═══════════════════════════════════════════════════════════
# Test 3: Time-of-Day Analysis
# ═══════════════════════════════════════════════════════════
def test_time_of_day(trades):
    print("\n" + "="*60)
    print("TEST 3: Time-of-Day Analysis (UTC Hour)")
    print("="*60)

    hourly = defaultdict(list)
    for t in trades:
        ts = pd.Timestamp(t['entry_ts'])
        hourly[ts.hour].append(t['pnl'])

    results = {}
    print(f"  {'Hour':>4s} | {'Trades':>6s} | {'PnL':>8s} | {'WR':>6s} | {'Avg PnL':>8s}")
    print(f"  {'-'*4} | {'-'*6} | {'-'*8} | {'-'*6} | {'-'*8}")

    for hour in range(24):
        pnls = hourly.get(hour, [])
        n = len(pnls)
        total = sum(pnls) if pnls else 0
        wr = sum(1 for p in pnls if p > 0) / n * 100 if n > 0 else 0
        avg = total / n if n > 0 else 0
        results[str(hour)] = {
            'trades': n, 'pnl': round(total, 2),
            'wr': round(wr, 1), 'avg_pnl': round(avg, 4)
        }
        if n > 0:
            marker = " ***" if avg < 0 else ""
            print(f"  {hour:4d} | {n:6d} | {total:+8.2f} | {wr:5.1f}% | {avg:+.4f}%{marker}")

    # Best and worst hours
    active_hours = {h: v for h, v in results.items() if v['trades'] >= 10}
    if active_hours:
        best_h = max(active_hours, key=lambda h: active_hours[h]['avg_pnl'])
        worst_h = min(active_hours, key=lambda h: active_hours[h]['avg_pnl'])
        print(f"\n  Best hour:  {best_h} (avg {active_hours[best_h]['avg_pnl']:+.4f}%)")
        print(f"  Worst hour: {worst_h} (avg {active_hours[worst_h]['avg_pnl']:+.4f}%)")
        neg_hours = [h for h, v in active_hours.items() if v['avg_pnl'] < 0]
        print(f"  Hours with negative edge (n>=10): {neg_hours if neg_hours else 'None'}")

    return results


# ═══════════════════════════════════════════════════════════
# Test 4: Day-of-Week Analysis
# ═══════════════════════════════════════════════════════════
def test_day_of_week(trades):
    print("\n" + "="*60)
    print("TEST 4: Day-of-Week Analysis")
    print("="*60)

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = defaultdict(list)
    for t in trades:
        ts = pd.Timestamp(t['entry_ts'])
        daily[ts.dayofweek].append(t['pnl'])

    results = {}
    print(f"  {'Day':>3s} | {'Trades':>6s} | {'PnL':>8s} | {'WR':>6s} | {'Avg PnL':>8s}")
    print(f"  {'-'*3} | {'-'*6} | {'-'*8} | {'-'*6} | {'-'*8}")

    for dow in range(7):
        pnls = daily.get(dow, [])
        n = len(pnls)
        total = sum(pnls) if pnls else 0
        wr = sum(1 for p in pnls if p > 0) / n * 100 if n > 0 else 0
        avg = total / n if n > 0 else 0
        results[day_names[dow]] = {
            'trades': n, 'pnl': round(total, 2),
            'wr': round(wr, 1), 'avg_pnl': round(avg, 4)
        }
        marker = " ***" if avg < 0 and n >= 10 else ""
        print(f"  {day_names[dow]:>3s} | {n:6d} | {total:+8.2f} | {wr:5.1f}% | {avg:+.4f}%{marker}")

    return results


# ═══════════════════════════════════════════════════════════
# Test 5: Drawdown Duration Analysis
# ═══════════════════════════════════════════════════════════
def test_drawdown_duration(trades):
    print("\n" + "="*60)
    print("TEST 5: Drawdown Duration Analysis")
    print("="*60)

    # Build equity curve by exit bar
    sorted_trades = sorted(trades, key=lambda t: t['exit_bar'])
    cum_pnl = []
    running = 0
    for t in sorted_trades:
        running += t['pnl']
        cum_pnl.append({'bar': t['exit_bar'], 'equity': running})

    # Compute drawdown periods
    peak = -1e18
    dd_periods = []
    dd_start = None

    for i, pt in enumerate(cum_pnl):
        eq = pt['equity']
        if eq > peak:
            if dd_start is not None and peak > -1e17:
                # End of drawdown period
                dd_periods.append({
                    'start_idx': dd_start,
                    'end_idx': i,
                    'start_bar': cum_pnl[dd_start]['bar'],
                    'end_bar': pt['bar'],
                    'duration_bars': pt['bar'] - cum_pnl[dd_start]['bar'],
                    'depth': dd_depth,
                })
            peak = eq
            dd_start = i
            dd_depth = 0
        else:
            dd = peak - eq
            if dd > dd_depth:
                dd_depth = dd

    # If still in drawdown at end
    if dd_start is not None and dd_depth > 0:
        dd_periods.append({
            'start_idx': dd_start,
            'end_idx': len(cum_pnl) - 1,
            'start_bar': cum_pnl[dd_start]['bar'],
            'end_bar': cum_pnl[-1]['bar'],
            'duration_bars': cum_pnl[-1]['bar'] - cum_pnl[dd_start]['bar'],
            'depth': dd_depth,
            'note': 'unrecovered',
        })

    if dd_periods:
        durations = [d['duration_bars'] for d in dd_periods]
        depths = [d['depth'] for d in dd_periods]
        dur_days = [d / (4 * 24) for d in durations]  # 15m bars: 4 per hour

        max_dd_period = max(dd_periods, key=lambda d: d['depth'])
        max_dur_period = max(dd_periods, key=lambda d: d['duration_bars'])

        print(f"  Total drawdown periods: {len(dd_periods)}")
        print(f"  Max drawdown depth: {max(depths):.2f}%")
        print(f"  Max drawdown duration: {max(durations)} bars ({max(dur_days):.1f} days)")
        print(f"  Median duration: {np.median(durations):.0f} bars ({np.median(dur_days):.1f} days)")
        print(f"  Mean duration: {np.mean(durations):.0f} bars ({np.mean(dur_days):.1f} days)")
        print(f"\n  Duration percentiles (bars):")
        for p in [25, 50, 75, 90, 95]:
            v = np.percentile(durations, p)
            print(f"    P{p}: {v:.0f} bars ({v/(4*24):.1f} days)")
    else:
        print("  No drawdown periods found (monotonic increase)")

    return {
        'n_dd_periods': len(dd_periods),
        'max_dd_depth': round(max(depths), 2) if dd_periods else 0,
        'max_dd_duration_bars': max(durations) if dd_periods else 0,
        'max_dd_duration_days': round(max(dur_days), 1) if dd_periods else 0,
        'median_duration_bars': round(float(np.median(durations)), 0) if dd_periods else 0,
        'mean_duration_bars': round(float(np.mean(durations)), 0) if dd_periods else 0,
    }


# ═══════════════════════════════════════════════════════════
# Test 6: Consecutive Loss Analysis
# ═══════════════════════════════════════════════════════════
def test_consecutive_losses(trades):
    print("\n" + "="*60)
    print("TEST 6: Consecutive Loss Analysis")
    print("="*60)

    # Actual consecutive losses
    sorted_trades = sorted(trades, key=lambda t: t['entry_bar'])
    streaks = []
    cur_streak = 0
    for t in sorted_trades:
        if t['pnl'] <= 0:
            cur_streak += 1
        else:
            if cur_streak > 0:
                streaks.append(cur_streak)
            cur_streak = 0
    if cur_streak > 0:
        streaks.append(cur_streak)

    max_streak = max(streaks) if streaks else 0
    streak_counts = defaultdict(int)
    for s in streaks:
        streak_counts[s] += 1

    print(f"  Max consecutive losses: {max_streak}")
    print(f"  Total losing streaks: {len(streaks)}")
    print(f"\n  Streak distribution:")
    for length in sorted(streak_counts.keys()):
        print(f"    {length:3d} consecutive: {streak_counts[length]:3d} times")

    # Probability of long streaks
    n_trades = len(sorted_trades)
    wr = sum(1 for t in sorted_trades if t['pnl'] > 0) / n_trades if n_trades > 0 else 0
    loss_rate = 1 - wr

    # Monte Carlo: simulate 10000 sequences
    np.random.seed(42)
    n_sims = 10000
    mc_max_streaks = []
    for _ in range(n_sims):
        outcomes = np.random.random(n_trades) > wr  # True = loss
        max_s = 0
        cur_s = 0
        for o_val in outcomes:
            if o_val:
                cur_s += 1
                max_s = max(max_s, cur_s)
            else:
                cur_s = 0
        mc_max_streaks.append(max_s)

    mc_max_streaks = np.array(mc_max_streaks)
    pct_5plus = np.mean(mc_max_streaks >= 5) * 100
    pct_10plus = np.mean(mc_max_streaks >= 10) * 100
    pct_13plus = np.mean(mc_max_streaks >= 13) * 100
    pct_15plus = np.mean(mc_max_streaks >= 15) * 100

    print(f"\n  Monte Carlo ({n_sims} sims, WR={wr*100:.1f}%, {n_trades} trades):")
    print(f"    Expected max consecutive loss: {np.mean(mc_max_streaks):.1f}")
    print(f"    P(max streak >= 5):  {pct_5plus:.1f}%")
    print(f"    P(max streak >= 10): {pct_10plus:.1f}%")
    print(f"    P(max streak >= 13): {pct_13plus:.1f}%")
    print(f"    P(max streak >= 15): {pct_15plus:.1f}%")
    print(f"    MC 95th percentile:  {np.percentile(mc_max_streaks, 95):.0f}")
    print(f"    MC 99th percentile:  {np.percentile(mc_max_streaks, 99):.0f}")
    print(f"    Actual max ({max_streak}) vs expected ({np.mean(mc_max_streaks):.1f}): "
          f"{'NORMAL' if max_streak <= np.percentile(mc_max_streaks, 95) else 'ELEVATED'}")

    return {
        'actual_max_consecutive': max_streak,
        'streak_distribution': dict(streak_counts),
        'mc_expected_max': round(float(np.mean(mc_max_streaks)), 1),
        'mc_p95_max': int(np.percentile(mc_max_streaks, 95)),
        'mc_p99_max': int(np.percentile(mc_max_streaks, 99)),
        'p_5plus': round(pct_5plus, 1),
        'p_10plus': round(pct_10plus, 1),
        'p_13plus': round(pct_13plus, 1),
    }


# ═══════════════════════════════════════════════════════════
# Test 7: Trade Duration Distribution
# ═══════════════════════════════════════════════════════════
def test_trade_duration(trades):
    print("\n" + "="*60)
    print("TEST 7: Trade Duration Distribution")
    print("="*60)

    durations = [t['bars_held'] for t in trades]
    dur_arr = np.array(durations)

    print(f"  Total trades: {len(trades)}")
    print(f"  Duration percentiles (bars / hours):")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        v = np.percentile(dur_arr, p)
        print(f"    P{p:2d}: {v:6.0f} bars ({v*15/60:6.1f} hours)")

    print(f"\n  Mean: {np.mean(dur_arr):.1f} bars ({np.mean(dur_arr)*15/60:.1f} hours)")
    print(f"  Std:  {np.std(dur_arr):.1f} bars")

    # Split by exit reason
    by_reason = defaultdict(list)
    for t in trades:
        by_reason[t['reason']].append(t['bars_held'])

    print(f"\n  By exit reason:")
    for reason in sorted(by_reason.keys()):
        durs = by_reason[reason]
        n = len(durs)
        med = np.median(durs)
        mean = np.mean(durs)
        print(f"    {reason:10s}: n={n:4d}, median={med:6.0f} bars ({med*15/60:5.1f}h), "
              f"mean={mean:6.0f} bars ({mean*15/60:5.1f}h)")

    # Winners vs Losers
    win_durs = [t['bars_held'] for t in trades if t['pnl'] > 0]
    loss_durs = [t['bars_held'] for t in trades if t['pnl'] <= 0]

    if win_durs and loss_durs:
        print(f"\n  Winners:  median {np.median(win_durs):.0f} bars ({np.median(win_durs)*15/60:.1f}h), "
              f"mean {np.mean(win_durs):.0f} bars")
        print(f"  Losers:   median {np.median(loss_durs):.0f} bars ({np.median(loss_durs)*15/60:.1f}h), "
              f"mean {np.mean(loss_durs):.0f} bars")

    # Histogram buckets
    buckets = [0, 1, 2, 5, 10, 20, 50, 100, 192]
    hist = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i+1]
        count = sum(1 for d in durations if lo <= d < hi)
        hist.append({'range': f'{lo}-{hi}', 'count': count})

    return {
        'percentiles': {f'P{p}': round(float(np.percentile(dur_arr, p)), 0)
                        for p in [5,10,25,50,75,90,95,99]},
        'mean': round(float(np.mean(dur_arr)), 1),
        'std': round(float(np.std(dur_arr)), 1),
        'by_reason': {r: {'n': len(d), 'median': round(float(np.median(d)), 0),
                          'mean': round(float(np.mean(d)), 0)}
                      for r, d in by_reason.items()},
        'winners_median': round(float(np.median(win_durs)), 0) if win_durs else 0,
        'losers_median': round(float(np.median(loss_durs)), 0) if loss_durs else 0,
    }


# ═══════════════════════════════════════════════════════════
# Test 8: Bollinger Width Filter + WF + MC
# ═══════════════════════════════════════════════════════════
def test_bollinger_filter(o, h, l, c, timestamps, n, atr_vals, ch_high, ch_low, sw_low, sw_high):
    print("\n" + "="*60)
    print("TEST 8: Bollinger Width Filter")
    print("="*60)

    c_list = c.tolist()
    bb_width = compute_bollinger_width(c_list, period=20, std_mult=2.0)

    # Get valid BB width values for percentile computation
    valid_bb = [v for v in bb_width if not math.isnan(v)]
    if not valid_bb:
        print("  ERROR: No valid BB width values")
        return {}

    p25 = np.percentile(valid_bb, 25)
    p50 = np.percentile(valid_bb, 50)
    p75 = np.percentile(valid_bb, 75)

    print(f"  BB Width percentiles: P25={p25:.4f}%, P50={p50:.4f}%, P75={p75:.4f}%")

    # Baseline (no filter)
    base_trades = backtest_enriched(o, h, l, c, timestamps, n,
                                    atr_vals, ch_high, ch_low, sw_low, sw_high)
    base_pnl = sum(t['pnl'] for t in base_trades)
    base_n = len(base_trades)

    print(f"\n  Baseline (no filter): {base_n} trades, PnL {base_pnl:+.1f}%")

    # Test each threshold
    thresholds = {'P25': p25, 'P50': p50, 'P75': p75}
    threshold_results = {}

    for label, thresh in thresholds.items():
        trades = backtest_enriched(o, h, l, c, timestamps, n,
                                   atr_vals, ch_high, ch_low, sw_low, sw_high,
                                   bb_width=bb_width, bb_threshold=thresh)
        pnls = [t['pnl'] for t in trades]
        total = sum(pnls)
        nt = len(trades)
        wr = sum(1 for p in pnls if p > 0) / nt * 100 if nt > 0 else 0
        avg = total / nt if nt > 0 else 0

        threshold_results[label] = {
            'threshold': round(thresh, 4),
            'trades': nt,
            'pnl': round(total, 2),
            'wr': round(wr, 1),
            'avg_pnl': round(avg, 4),
        }
        print(f"  {label} (thresh={thresh:.4f}%): {nt} trades, PnL {total:+.1f}%, WR {wr:.1f}%, Avg {avg:+.4f}%")

    # WF 5-fold + MC for best threshold
    best_label = max(threshold_results, key=lambda k: threshold_results[k]['pnl'])
    best_thresh = thresholds[best_label]
    print(f"\n  Best threshold: {best_label} ({best_thresh:.4f}%) — running WF 5-fold + MC...")

    # WF 5-fold expanding window
    n_folds = 5
    wf_results = []
    for fi in range(n_folds):
        ie = int(n * (fi + 1) / (n_folds + 1))  # expanding IS end
        is_end = ie
        oos_end = int(n * (fi + 2) / (n_folds + 1)) if fi < n_folds - 1 else n

        # IS trades
        is_trades = backtest_enriched(o[:is_end], h[:is_end], l[:is_end], c[:is_end],
                                      timestamps[:is_end], is_end,
                                      atr_vals[:is_end], ch_high[:is_end], ch_low[:is_end],
                                      sw_low[:is_end], sw_high[:is_end],
                                      bb_width=bb_width[:is_end], bb_threshold=best_thresh)
        # OOS trades (full data run, filter to OOS range)
        oos_trades = backtest_enriched(o[:oos_end], h[:oos_end], l[:oos_end], c[:oos_end],
                                       timestamps[:oos_end], oos_end,
                                       atr_vals[:oos_end], ch_high[:oos_end], ch_low[:oos_end],
                                       sw_low[:oos_end], sw_high[:oos_end],
                                       bb_width=bb_width[:oos_end], bb_threshold=best_thresh)

        # Filter OOS trades: entry_bar >= is_end
        oos_only = [t for t in oos_trades if t['entry_bar'] >= is_end]

        is_pnl = sum(t['pnl'] for t in is_trades)
        oos_pnl = sum(t['pnl'] for t in oos_only)

        wf_results.append({
            'fold': fi + 1,
            'is_bars': is_end,
            'oos_bars': oos_end - is_end,
            'is_pnl': round(is_pnl, 2),
            'is_trades': len(is_trades),
            'oos_pnl': round(oos_pnl, 2),
            'oos_trades': len(oos_only),
            'pass': oos_pnl > 0,
        })

        status = "PASS" if oos_pnl > 0 else "FAIL"
        print(f"    Fold {fi+1}: IS {is_pnl:+.1f}% ({len(is_trades)} trades) | "
              f"OOS {oos_pnl:+.1f}% ({len(oos_only)} trades) [{status}]")

    n_pass = sum(1 for f in wf_results if f['pass'])
    oos_total = sum(f['oos_pnl'] for f in wf_results)
    print(f"    WF Result: {n_pass}/{n_folds} PASS, OOS total: {oos_total:+.1f}%")

    # MC test on filtered trades (sign randomization, 5000 sims)
    filtered_trades = backtest_enriched(o, h, l, c, timestamps, n,
                                        atr_vals, ch_high, ch_low, sw_low, sw_high,
                                        bb_width=bb_width, bb_threshold=best_thresh)
    actual_pnl = sum(t['pnl'] for t in filtered_trades)
    trade_pnls = [t['pnl'] for t in filtered_trades]
    n_sims = 5000

    mc_pvals = []
    for seed in [42, 123, 7]:
        rng = np.random.RandomState(seed)
        count_ge = 0
        for _ in range(n_sims):
            signs = rng.choice([-1, 1], size=len(trade_pnls))
            shuffled_pnl = sum(p * s for p, s in zip(trade_pnls, signs))
            if shuffled_pnl >= actual_pnl:
                count_ge += 1
        p_val = count_ge / n_sims
        mc_pvals.append(p_val)
        print(f"    MC seed={seed}: p={p_val:.4f}")

    max_p = max(mc_pvals)
    mc_pass = max_p < 0.01
    print(f"    MC max p-value: {max_p:.4f} ({'PASS' if mc_pass else 'FAIL'})")

    return {
        'bb_percentiles': {'P25': round(p25, 4), 'P50': round(p50, 4), 'P75': round(p75, 4)},
        'baseline': {'trades': base_n, 'pnl': round(base_pnl, 2)},
        'thresholds': threshold_results,
        'best_threshold': best_label,
        'wf_results': wf_results,
        'wf_pass': f'{n_pass}/{n_folds}',
        'wf_oos_total': round(oos_total, 2),
        'mc_pvalues': [round(p, 4) for p in mc_pvals],
        'mc_max_p': round(max_p, 4),
        'mc_pass': mc_pass,
    }


# ═══════════════════════════════════════════════════════════
# Test 9: Profit Factor Stability
# ═══════════════════════════════════════════════════════════
def test_profit_factor_stability(trades):
    print("\n" + "="*60)
    print("TEST 9: Profit Factor Stability (Rolling 60-trade)")
    print("="*60)

    sorted_trades = sorted(trades, key=lambda t: t['entry_bar'])
    pnls = [t['pnl'] for t in sorted_trades]
    n = len(pnls)
    window = 60

    if n < window:
        print("  Not enough trades for rolling PF analysis")
        return {}

    rolling_pf = []
    pf_below_1 = []

    for i in range(n - window + 1):
        chunk = pnls[i:i + window]
        gross_win = sum(p for p in chunk if p > 0)
        gross_loss = abs(sum(p for p in chunk if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        rolling_pf.append(pf)
        if pf < 1.0:
            pf_below_1.append((i, pf))

    pf_arr = np.array([pf for pf in rolling_pf if pf != float('inf')])

    print(f"  Rolling PF windows: {len(rolling_pf)}")
    print(f"  Min PF: {np.min(pf_arr):.3f}")
    print(f"  Max PF: {np.max(pf_arr):.3f}")
    print(f"  Mean PF: {np.mean(pf_arr):.3f}")
    print(f"  Std PF: {np.std(pf_arr):.3f}")
    print(f"  Windows with PF < 1.0: {len(pf_below_1)} ({len(pf_below_1)/len(rolling_pf)*100:.1f}%)")

    # Longest consecutive stretch of PF < 1
    max_below = 0
    cur_below = 0
    for pf in rolling_pf:
        if pf < 1.0:
            cur_below += 1
            max_below = max(max_below, cur_below)
        else:
            cur_below = 0

    print(f"  Longest consecutive PF < 1.0: {max_below} windows")

    # Overall PF
    total_win = sum(p for p in pnls if p > 0)
    total_loss = abs(sum(p for p in pnls if p <= 0))
    overall_pf = total_win / total_loss if total_loss > 0 else float('inf')
    print(f"\n  Overall Profit Factor: {overall_pf:.3f}")

    return {
        'overall_pf': round(float(overall_pf), 3),
        'rolling_min': round(float(np.min(pf_arr)), 3),
        'rolling_max': round(float(np.max(pf_arr)), 3),
        'rolling_mean': round(float(np.mean(pf_arr)), 3),
        'rolling_std': round(float(np.std(pf_arr)), 3),
        'pct_below_1': round(len(pf_below_1) / len(rolling_pf) * 100, 1),
        'longest_below_1': max_below,
    }


# ═══════════════════════════════════════════════════════════
# Test 10: Entry Price vs Channel Distance
# ═══════════════════════════════════════════════════════════
def test_channel_distance(trades):
    print("\n" + "="*60)
    print("TEST 10: Entry Price vs Channel Distance (ATR units)")
    print("="*60)

    dists = [t['channel_distance_atr'] for t in trades]
    pnls = [t['pnl'] for t in trades]

    dist_arr = np.array(dists)
    pnl_arr = np.array(pnls)

    # Quintiles
    quintile_edges = np.percentile(dist_arr, [0, 20, 40, 60, 80, 100])
    results = {}

    print(f"  {'Quintile':>8s} | {'Range (ATR)':>15s} | {'Trades':>6s} | {'PnL':>8s} | {'WR':>6s} | {'Avg PnL':>8s}")
    print(f"  {'-'*8} | {'-'*15} | {'-'*6} | {'-'*8} | {'-'*6} | {'-'*8}")

    for q in range(5):
        lo = quintile_edges[q]
        hi = quintile_edges[q + 1]
        if q == 4:
            mask = (dist_arr >= lo) & (dist_arr <= hi)
        else:
            mask = (dist_arr >= lo) & (dist_arr < hi)
        q_pnls = pnl_arr[mask]
        n_q = len(q_pnls)
        total = float(np.sum(q_pnls)) if n_q > 0 else 0
        wr = float(np.sum(q_pnls > 0)) / n_q * 100 if n_q > 0 else 0
        avg = total / n_q if n_q > 0 else 0

        label = f"Q{q+1}"
        results[label] = {
            'range': f'{lo:.3f}-{hi:.3f}',
            'trades': int(n_q),
            'pnl': round(total, 2),
            'wr': round(wr, 1),
            'avg_pnl': round(avg, 4),
        }
        print(f"  {label:>8s} | {lo:6.3f} - {hi:6.3f} | {n_q:6d} | {total:+8.2f} | {wr:5.1f}% | {avg:+.4f}%")

    # Correlation
    corr = float(np.corrcoef(dist_arr, pnl_arr)[0, 1])
    print(f"\n  Correlation (channel_dist vs PnL): {corr:.4f}")
    if corr > 0.05:
        interp = "Stronger breakouts produce better trades"
    elif corr < -0.05:
        interp = "Weaker breakouts produce better trades"
    else:
        interp = "No significant relationship between breakout strength and trade quality"
    print(f"  Interpretation: {interp}")

    return {
        'quintiles': results,
        'correlation': round(corr, 4),
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("C1 Breakout v2 — Deep Stress Tests")
    print("=" * 60)

    # Load data
    print("Loading 15m data...")
    df = load_15m_data()
    n = len(df)
    print(f"  {n} bars loaded")

    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    timestamps = df['timestamp'].values

    # Precompute indicators
    print("Computing indicators...")
    atr_vals, ch_high, ch_low, sw_low, sw_high = precompute_indicators(h, l, c, n)

    # Baseline verification
    print("\n--- BASELINE VERIFICATION ---")
    trades = backtest_enriched(o, h, l, c, timestamps, n,
                               atr_vals, ch_high, ch_low, sw_low, sw_high)
    pnls = [t['pnl'] for t in trades]
    total_pnl = sum(pnls)
    n_trades = len(trades)
    wr = sum(1 for p in pnls if p > 0) / n_trades * 100 if n_trades > 0 else 0

    by_reason = defaultdict(int)
    for t in trades:
        by_reason[t['reason']] += 1

    print(f"  Trades: {n_trades}")
    print(f"  PnL: {total_pnl:+.1f}%")
    print(f"  WR: {wr:.1f}%")
    print(f"  Exit reasons: {dict(by_reason)}")
    print(f"  Expected: ~1028 trades, ~+169.5%, ~36.6% WR")

    if abs(n_trades - 1028) > 50 or abs(total_pnl - 169.5) > 20:
        print("  WARNING: Baseline deviates significantly from expected values!")
        print("  Proceeding with tests but results may not match validated baseline.")

    all_results = {
        'metadata': {
            'script': 'c1_deep_stress_test.py',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_bars': n,
            'n_days': round(n / (4 * 24), 1),
            'baseline_trades': n_trades,
            'baseline_pnl': round(total_pnl, 2),
            'baseline_wr': round(wr, 1),
        }
    }

    # Run all tests
    all_results['test_01_slippage'] = test_slippage_sensitivity(
        o, h, l, c, timestamps, n, atr_vals, ch_high, ch_low, sw_low, sw_high)

    all_results['test_02_mae_mfe'] = test_mae_mfe(trades)

    all_results['test_03_time_of_day'] = test_time_of_day(trades)

    all_results['test_04_day_of_week'] = test_day_of_week(trades)

    all_results['test_05_drawdown_duration'] = test_drawdown_duration(trades)

    all_results['test_06_consecutive_losses'] = test_consecutive_losses(trades)

    all_results['test_07_trade_duration'] = test_trade_duration(trades)

    all_results['test_08_bollinger_filter'] = test_bollinger_filter(
        o, h, l, c, timestamps, n, atr_vals, ch_high, ch_low, sw_low, sw_high)

    all_results['test_09_profit_factor'] = test_profit_factor_stability(trades)

    all_results['test_10_channel_distance'] = test_channel_distance(trades)

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
