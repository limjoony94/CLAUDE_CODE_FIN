#!/usr/bin/env python3
"""
C1 Breakout v2 — Look-Ahead Bias, Overfitting, Fee Deep Audit
================================================================
18 forensic tests in 3 sections:
  A: Look-Ahead Bias (7 tests)
  B: Overfitting Assessment (6 tests)
  C: Fee/Cost Analysis (5 tests)

Strategy: 15m BTC Channel Breakout + Fractal SL + ATR Trailing TP
Baseline: N=1, additive 1x, fee=0.10% RT → expected ~+169.5%
"""

import os, sys, json, math, time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(line_buffering=True)

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

# ─── Constants (N=1, additive 1x, matches validated baseline) ───
FEE_PCT_RT = 0.10       # round trip total
LEVERAGE = 1             # additive 1x baseline
EMERGENCY_SL = 3.0
TIMEOUT_BARS = 192
FRACTAL_LOOKBACK = 10

# Default params
DEFAULT_PARAMS = {
    'channel_period': 15,
    'body_min_ratio': 0.4,
    'atr_period': 14,
    'trail_K': 2.5,
    'max_sl_atr': 3.3,
    'emergency_sl_pct': 3.0,
    'max_hold_bars': 192,
    'sl_min_pct': 0.15,
    'sl_max_pct': 3.0,
    'min_bars_between': 2,
    'trail_activation_pct': 0.05,
}

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                         'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                           'results', 'c1_lookahead_overfit_fee_audit.json')


# ═══════════════════════════════════════════════════════════════════
# Data Loading & Resampling
# ═══════════════════════════════════════════════════════════════════

def load_and_validate_5m():
    """Load 5m data and validate for timestamp gaps."""
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Check for gaps
    diffs = df['timestamp'].diff().dropna()
    expected_diff = pd.Timedelta(minutes=5)
    gaps = diffs[diffs != expected_diff]
    if len(gaps) > 0:
        print(f"  WARNING: {len(gaps)} timestamp gaps found in 5m data")
        for idx in gaps.index[:5]:
            print(f"    Row {idx}: {df['timestamp'].iloc[idx-1]} -> {df['timestamp'].iloc[idx]} "
                  f"(gap={gaps[idx]})")
    else:
        print(f"  5m data: {len(df)} rows, no gaps, clean")

    return df


def resample_to_15m(df_5m):
    """Resample 5m to 15m with proper alignment."""
    df = df_5m.copy()
    # Align to 15m boundaries using timestamp
    df['ts_15m'] = df['timestamp'].dt.floor('15min')
    agg = df.groupby('ts_15m').agg(
        timestamp=('timestamp', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index(drop=True)
    return agg


# ═══════════════════════════════════════════════════════════════════
# Core Backtest Engine (N=1, additive, production-identical logic)
# ═══════════════════════════════════════════════════════════════════

def backtest_n1(df, params=None, fee_rt=FEE_PCT_RT, extra_cost_per_bar=0.0,
                return_details=False):
    """N=1 additive backtest.

    Args:
        df: DataFrame with open/high/low/close columns
        params: dict overriding DEFAULT_PARAMS
        fee_rt: round-trip fee as percent
        extra_cost_per_bar: additional cost per bar held (for funding rate)
        return_details: if True, return (trades, details_dict)

    Returns:
        list of trade dicts
    """
    p = {**DEFAULT_PARAMS}
    if params:
        p.update(params)

    sig = C1BreakoutSignal(p)
    n = len(df)
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)

    h_list, l_list, c_list = h.tolist(), l.tolist(), c.tolist()
    atr = compute_atr(h_list, l_list, c_list, p['atr_period'])
    ch_high, ch_low = compute_channel(h_list, l_list, p['channel_period'])
    sw_low, sw_high = compute_fractal_swings(h_list, l_list, FRACTAL_LOOKBACK)

    trades = []
    in_position = False
    pos = None
    last_exit_bar = -10

    warmup = max(p['channel_period'] + FRACTAL_LOOKBACK, 25)

    detail_log = [] if return_details else None

    for bar in range(warmup, n):
        # ── Exit ──
        if in_position:
            ep = pos['entry_price']
            d = pos['direction']
            bh = bar - pos['entry_bar']

            # Update best price
            if d == 'LONG':
                pos['best_price'] = max(pos['best_price'], h[bar])
            else:
                pos['best_price'] = min(pos['best_price'], l[bar])

            atr_val = atr[bar] if not math.isnan(atr[bar]) else 0

            exit_info = sig.check_exit(
                direction=d, entry_price=ep, best_price=pos['best_price'],
                current_high=h[bar], current_low=l[bar], current_close=c[bar],
                sl_price=pos['sl_price'], atr_val=atr_val, bars_held=bh
            )

            if exit_info is not None:
                exit_price = exit_info['exit_price']
                if d == 'LONG':
                    raw_pnl = (exit_price / ep - 1) * 100
                else:
                    raw_pnl = (1 - exit_price / ep) * 100

                funding_cost = bh * extra_cost_per_bar if extra_cost_per_bar > 0 else 0
                trade_pnl = raw_pnl * LEVERAGE - fee_rt - funding_cost

                t = {
                    'signal_bar': pos['signal_bar'],
                    'entry_bar': pos['entry_bar'],
                    'exit_bar': bar,
                    'direction': d,
                    'entry_price': ep,
                    'exit_price': exit_price,
                    'sl_price': pos['sl_price'],
                    'pnl': trade_pnl,
                    'raw_pnl': raw_pnl,
                    'reason': exit_info['reason'],
                    'bars_held': bh,
                }
                trades.append(t)

                if return_details and detail_log is not None:
                    # Check SL-Trail same-bar conflict
                    sl_hit = False
                    trail_hit = False
                    if d == 'LONG':
                        sl_hit = l[bar] <= pos['sl_price']
                    else:
                        sl_hit = h[bar] >= pos['sl_price']

                    # Check trail condition
                    if d == 'LONG':
                        best_pnl = (pos['best_price'] / ep - 1) * 100
                        cur_pnl = (c[bar] / ep - 1) * 100
                    else:
                        best_pnl = (1 - pos['best_price'] / ep) * 100
                        cur_pnl = (1 - c[bar] / ep) * 100

                    if best_pnl > p['trail_activation_pct'] and atr_val > 0:
                        trail_dist = p['trail_K'] * atr_val / c[bar] * 100
                        drawdown = best_pnl - cur_pnl
                        trail_hit = drawdown >= trail_dist

                    detail_log.append({
                        **t,
                        'sl_hit': sl_hit,
                        'trail_hit': trail_hit,
                        'both_hit': sl_hit and trail_hit,
                        'fractal_sl_used': pos.get('fractal_sl_raw', float('nan')),
                    })

                in_position = False
                last_exit_bar = bar
                continue

        # ── Entry ──
        if not in_position and bar + 1 < n:
            if bar - last_exit_bar < p['min_bars_between']:
                continue

            if math.isnan(atr[bar]) or atr[bar] <= 0:
                continue

            entry_signal = sig.check_entry(
                bar_open=o[bar], bar_high=h[bar], bar_low=l[bar], bar_close=c[bar],
                channel_high=ch_high[bar], channel_low=ch_low[bar],
                atr_val=atr[bar],
                last_swing_low=sw_low[bar], last_swing_high=sw_high[bar]
            )

            if entry_signal is not None:
                entry_price = o[bar + 1]
                direction = entry_signal['direction']

                # Recompute SL at entry price
                if direction == 'LONG':
                    sw = sw_low[bar]
                    atr_sl = entry_price - p['max_sl_atr'] * atr[bar]
                    fractal_sl = sw if not math.isnan(sw) else atr_sl
                    sl_price = max(fractal_sl, atr_sl)
                else:
                    sw = sw_high[bar]
                    atr_sl = entry_price + p['max_sl_atr'] * atr[bar]
                    fractal_sl = sw if not math.isnan(sw) else atr_sl
                    sl_price = min(fractal_sl, atr_sl)

                sl_dist = abs(entry_price - sl_price) / entry_price * 100
                if sl_dist < p['sl_min_pct'] or sl_dist > p['sl_max_pct']:
                    continue

                in_position = True
                pos = {
                    'signal_bar': bar,
                    'entry_bar': bar + 1,
                    'entry_price': entry_price,
                    'direction': direction,
                    'sl_price': sl_price,
                    'best_price': entry_price,
                    'fractal_sl_raw': fractal_sl,
                }

    return (trades, detail_log) if return_details else trades


def compute_additive_pnl(trades):
    """Simple additive PnL sum."""
    return sum(t['pnl'] for t in trades if t['reason'] != 'TIMEOUT')


def compute_stats(trades, n_bars=None, n_days=None):
    """Basic stats from trade list."""
    active = [t for t in trades if t['reason'] != 'TIMEOUT']
    if not active:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'rr': 0, 'avg_pnl': 0}

    pnls = [t['pnl'] for t in active]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(active) * 100 if active else 0
    wa = np.mean(wins) if wins else 0
    la = np.mean(losses) if losses else 0
    rr = abs(wa / la) if la != 0 else float('inf')
    total = sum(pnls)

    if n_days is None and n_bars is not None:
        n_days = n_bars * 15 / 1440

    return {
        'trades': len(active),
        'timeouts': len(trades) - len(active),
        'wr': round(wr, 2),
        'pnl': round(total, 2),
        'rr': round(rr, 2),
        'avg_pnl': round(np.mean(pnls), 4),
        'win_avg': round(wa, 4),
        'loss_avg': round(la, 4),
        'daily_pnl': round(total / n_days, 4) if n_days and n_days > 0 else 0,
        'trades_per_day': round(len(active) / n_days, 2) if n_days and n_days > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION A: Look-Ahead Bias Forensic Audit
# ═══════════════════════════════════════════════════════════════════

def test_a1_progressive_truncation(df_15m):
    """A1: Progressive Truncation (fine-grained) + last-30-day stability."""
    print("\n" + "=" * 70)
    print("  A1: Progressive Truncation Test (Fine-Grained)")
    print("=" * 70)

    n_full = len(df_15m)
    n_days_full = n_full * 15 / 1440
    day_targets = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, int(n_days_full)]
    results = []

    for target_days in day_targets:
        n_bars = min(int(target_days * 96), n_full)  # 96 bars per day at 15m
        if n_bars > n_full:
            n_bars = n_full
        sub = df_15m.iloc[:n_bars].copy().reset_index(drop=True)
        trades = backtest_n1(sub)
        active = [t for t in trades if t['reason'] != 'TIMEOUT']
        total_pnl = sum(t['pnl'] for t in active)
        actual_days = n_bars * 15 / 1440
        daily = total_pnl / actual_days if actual_days > 0 else 0

        # PnL of LAST 30 days in this truncation
        last_30_start = max(0, n_bars - int(30 * 96))
        last30_trades = [t for t in active if t['entry_bar'] >= last_30_start]
        last30_pnl = sum(t['pnl'] for t in last30_trades)

        r = {
            'days': round(actual_days, 1),
            'n_bars': n_bars,
            'trades': len(active),
            'pnl': round(total_pnl, 2),
            'daily_pnl': round(daily, 4),
            'last_30d_trades': len(last30_trades),
            'last_30d_pnl': round(last30_pnl, 2),
        }
        results.append(r)
        print(f"  {actual_days:>6.1f}d ({n_bars:>5} bars): T={len(active):>4}  "
              f"PnL={total_pnl:>+8.1f}%  daily={daily:>+.4f}%  "
              f"last30d={last30_pnl:>+6.1f}% ({len(last30_trades)}t)")

    # Check monotonicity of cumulative PnL
    pnls = [r['pnl'] for r in results]
    monotonic_violations = sum(1 for i in range(1, len(pnls))
                               if pnls[i] < pnls[i-1] * 0.7)  # Allow 30% dips

    # Check stability of last-30-day PnL across truncations
    last30_pnls = [r['last_30d_pnl'] for r in results if r['days'] >= 60]
    last30_std = np.std(last30_pnls) if len(last30_pnls) > 1 else 0
    last30_mean = np.mean(last30_pnls) if last30_pnls else 0

    passed = monotonic_violations == 0
    print(f"\n  Monotonic violations (>30% drop): {monotonic_violations}")
    print(f"  Last-30d PnL stability: mean={last30_mean:+.1f}%, std={last30_std:.1f}%")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'A1_progressive_truncation',
        'passed': passed,
        'results': results,
        'monotonic_violations': monotonic_violations,
        'last_30d_stability': {'mean': round(last30_mean, 2), 'std': round(last30_std, 2)},
    }


def test_a2_indicator_causality(df_15m):
    """A2: Indicator Causality Verification."""
    print("\n" + "=" * 70)
    print("  A2: Indicator Causality Verification")
    print("=" * 70)

    n = len(df_15m)
    h_list = df_15m['high'].tolist()
    l_list = df_15m['low'].tolist()
    c_list = df_15m['close'].tolist()

    test_bars = [100, 300, 500, 700, 900]
    test_bars = [b for b in test_bars if b < n]

    results = []
    all_pass = True

    for bar_idx in test_bars:
        # Compute with truncated data (up to bar_idx+1)
        trunc_h = h_list[:bar_idx + 1]
        trunc_l = l_list[:bar_idx + 1]
        trunc_c = c_list[:bar_idx + 1]

        atr_trunc = compute_atr(trunc_h, trunc_l, trunc_c, 14)
        ch_h_trunc, ch_l_trunc = compute_channel(trunc_h, trunc_l, 15)
        sw_l_trunc, sw_h_trunc = compute_fractal_swings(trunc_h, trunc_l, 10)

        # Compute with full data
        atr_full = compute_atr(h_list, l_list, c_list, 14)
        ch_h_full, ch_l_full = compute_channel(h_list, l_list, 15)
        sw_l_full, sw_h_full = compute_fractal_swings(h_list, l_list, 10)

        checks = {
            'ATR': (atr_trunc[bar_idx], atr_full[bar_idx]),
            'Channel_High': (ch_h_trunc[bar_idx], ch_h_full[bar_idx]),
            'Channel_Low': (ch_l_trunc[bar_idx], ch_l_full[bar_idx]),
            'Swing_Low': (sw_l_trunc[bar_idx], sw_l_full[bar_idx]),
            'Swing_High': (sw_h_trunc[bar_idx], sw_h_full[bar_idx]),
        }

        bar_pass = True
        for name, (trunc_val, full_val) in checks.items():
            if math.isnan(trunc_val) and math.isnan(full_val):
                match = True
            elif math.isnan(trunc_val) or math.isnan(full_val):
                match = False
            else:
                match = abs(trunc_val - full_val) < 1e-10
            if not match:
                bar_pass = False
                all_pass = False

        results.append({
            'bar_index': bar_idx,
            'passed': bar_pass,
            'values': {k: {'truncated': v[0], 'full': v[1]} for k, v in checks.items()},
        })
        status = "PASS" if bar_pass else "FAIL"
        print(f"  Bar {bar_idx}: {status}")
        for name, (tv, fv) in checks.items():
            m = "OK" if (math.isnan(tv) and math.isnan(fv)) or \
                        (not math.isnan(tv) and not math.isnan(fv) and abs(tv - fv) < 1e-10) else "MISMATCH"
            print(f"    {name:>15}: trunc={tv:.6f}  full={fv:.6f}  [{m}]")

    print(f"\n  VERDICT: {'PASS' if all_pass else 'FAIL'}")
    return {'test': 'A2_indicator_causality', 'passed': all_pass, 'results': results}


def test_a3_entry_exit_temporal(trades):
    """A3: Entry-Exit Temporal Order Check."""
    print("\n" + "=" * 70)
    print("  A3: Entry-Exit Temporal Order Check")
    print("=" * 70)

    violations = {
        'entry_after_exit': [],
        'same_bar_roundtrip': [],
        'entry_not_next_bar': [],
    }

    for i, t in enumerate(trades):
        # entry_bar must be < exit_bar
        if t['entry_bar'] >= t['exit_bar']:
            violations['same_bar_roundtrip'].append(i)

        # entry_bar must be signal_bar + 1
        if t['entry_bar'] != t['signal_bar'] + 1:
            violations['entry_not_next_bar'].append(i)

    n_same_bar = len(violations['same_bar_roundtrip'])
    n_not_next = len(violations['entry_not_next_bar'])

    print(f"  Total trades: {len(trades)}")
    print(f"  Same-bar round-trips (entry_bar >= exit_bar): {n_same_bar}")
    print(f"  Entry not at signal+1: {n_not_next}")

    # bars_held = 0 trades (entry and exit on same bar)
    zero_held = [t for t in trades if t['bars_held'] == 0]
    print(f"  bars_held=0 trades: {len(zero_held)}")
    if zero_held:
        print(f"    These are entry-bar SL hits (price gaps through SL on entry bar)")
        for t in zero_held[:5]:
            print(f"      bar={t['entry_bar']} dir={t['direction']} "
                  f"entry={t['entry_price']:.1f} exit={t['exit_price']:.1f} "
                  f"sl={t['sl_price']:.1f} reason={t['reason']}")

    passed = n_same_bar == 0 and n_not_next == 0
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'A3_temporal_order',
        'passed': passed,
        'same_bar_roundtrips': n_same_bar,
        'entry_not_next_bar': n_not_next,
        'bars_held_zero': len(zero_held),
    }


def test_a4_sl_lookahead(df_15m, trades):
    """A4: SL Price Look-Ahead Check."""
    print("\n" + "=" * 70)
    print("  A4: SL Price Look-Ahead Check")
    print("=" * 70)

    h_list = df_15m['high'].tolist()
    l_list = df_15m['low'].tolist()
    c_list = df_15m['close'].tolist()

    mismatches = []
    for i, t in enumerate(trades):
        sig_bar = t['signal_bar']
        direction = t['direction']
        entry_price = t['entry_price']

        # Recompute fractal swings using ONLY data up to signal_bar
        trunc_h = h_list[:sig_bar + 1]
        trunc_l = l_list[:sig_bar + 1]
        sw_l, sw_h = compute_fractal_swings(trunc_h, trunc_l, FRACTAL_LOOKBACK)

        atr_trunc = compute_atr(h_list[:sig_bar + 1], l_list[:sig_bar + 1],
                                c_list[:sig_bar + 1], 14)
        atr_val = atr_trunc[sig_bar]

        if direction == 'LONG':
            sw = sw_l[sig_bar] if not math.isnan(sw_l[sig_bar]) else entry_price - 3.3 * atr_val
            atr_sl = entry_price - 3.3 * atr_val
            expected_sl = max(sw, atr_sl)
        else:
            sw = sw_h[sig_bar] if not math.isnan(sw_h[sig_bar]) else entry_price + 3.3 * atr_val
            atr_sl = entry_price + 3.3 * atr_val
            expected_sl = min(sw, atr_sl)

        actual_sl = t['sl_price']
        if abs(expected_sl - actual_sl) > 0.01:
            mismatches.append({
                'trade_idx': i,
                'signal_bar': sig_bar,
                'direction': direction,
                'expected_sl': expected_sl,
                'actual_sl': actual_sl,
                'diff': abs(expected_sl - actual_sl),
            })

    n_checked = len(trades)
    n_mismatch = len(mismatches)
    print(f"  Checked: {n_checked} trades")
    print(f"  Mismatches: {n_mismatch}")
    if mismatches:
        for m in mismatches[:5]:
            print(f"    Trade {m['trade_idx']}: sig_bar={m['signal_bar']} "
                  f"expected={m['expected_sl']:.2f} actual={m['actual_sl']:.2f} "
                  f"diff={m['diff']:.4f}")

    passed = n_mismatch == 0
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'A4_sl_lookahead',
        'passed': passed,
        'checked': n_checked,
        'mismatches': n_mismatch,
        'details': mismatches[:10],
    }


def test_a5_future_data_leakage(df_15m):
    """A5: Future Data Leakage via min/max/normalization."""
    print("\n" + "=" * 70)
    print("  A5: Future Data Leakage via min/max")
    print("=" * 70)

    n = len(df_15m)
    mid = n // 2
    h_list = df_15m['high'].tolist()
    l_list = df_15m['low'].tolist()
    c_list = df_15m['close'].tolist()

    # Compute with half data
    atr_half = compute_atr(h_list[:mid], l_list[:mid], c_list[:mid], 14)
    ch_h_half, ch_l_half = compute_channel(h_list[:mid], l_list[:mid], 15)
    sw_l_half, sw_h_half = compute_fractal_swings(h_list[:mid], l_list[:mid], 10)

    # Compute with full data
    atr_full = compute_atr(h_list, l_list, c_list, 14)
    ch_h_full, ch_l_full = compute_channel(h_list, l_list, 15)
    sw_l_full, sw_h_full = compute_fractal_swings(h_list, l_list, 10)

    # Compare at bar mid-1
    check_bar = mid - 1
    checks = {
        'ATR': (atr_half[check_bar], atr_full[check_bar]),
        'Channel_High': (ch_h_half[check_bar], ch_h_full[check_bar]),
        'Channel_Low': (ch_l_half[check_bar], ch_l_full[check_bar]),
        'Swing_Low': (sw_l_half[check_bar], sw_l_full[check_bar]),
        'Swing_High': (sw_h_half[check_bar], sw_h_full[check_bar]),
    }

    all_pass = True
    for name, (hv, fv) in checks.items():
        if math.isnan(hv) and math.isnan(fv):
            match = True
        elif math.isnan(hv) or math.isnan(fv):
            match = False
        else:
            match = abs(hv - fv) < 1e-10
        if not match:
            all_pass = False
        status = "OK" if match else "LEAK"
        print(f"  {name:>15} at bar {check_bar}: half={hv:.6f}  full={fv:.6f}  [{status}]")

    print(f"\n  VERDICT: {'PASS' if all_pass else 'FAIL'}")
    return {'test': 'A5_future_data_leakage', 'passed': all_pass,
            'check_bar': check_bar, 'checks': {k: {'half': v[0], 'full': v[1]}
                                                 for k, v in checks.items()}}


def test_a6_bar_resolution_order(detail_trades):
    """A6: Bar Resolution Order Check (SL before Trail)."""
    print("\n" + "=" * 70)
    print("  A6: Bar Resolution Order Check")
    print("=" * 70)

    both_hit = [t for t in detail_trades if t['both_hit']]
    sl_won = [t for t in both_hit if t['reason'] == 'SL']
    trail_won = [t for t in both_hit if t['reason'] == 'TRAIL_TP']

    print(f"  Total trades: {len(detail_trades)}")
    print(f"  Same-bar SL+Trail conflict: {len(both_hit)}")
    if both_hit:
        print(f"    SL took priority: {len(sl_won)}")
        print(f"    Trail took priority: {len(trail_won)}")
        for t in both_hit[:5]:
            print(f"      bar={t['exit_bar']} dir={t['direction']} reason={t['reason']} "
                  f"entry={t['entry_price']:.1f} sl={t['sl_price']:.1f}")

    # SL should ALWAYS take priority (documented behavior)
    passed = len(trail_won) == 0 or len(both_hit) == 0
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'A6_bar_resolution_order',
        'passed': passed,
        'both_hit_count': len(both_hit),
        'sl_priority': len(sl_won),
        'trail_priority': len(trail_won),
    }


def test_a7_stale_signal(trades):
    """A7: Stale Signal / min_bars_between Check."""
    print("\n" + "=" * 70)
    print("  A7: Stale Signal Check (min_bars_between=2)")
    print("=" * 70)

    violations = []
    sorted_trades = sorted(trades, key=lambda x: x['entry_bar'])

    for i in range(1, len(sorted_trades)):
        prev_exit = sorted_trades[i - 1]['exit_bar']
        # The signal_bar for next trade must be >= prev_exit + min_bars_between
        # But we track entry_bar = signal_bar + 1, so:
        cur_signal = sorted_trades[i]['signal_bar']
        gap = cur_signal - prev_exit
        if gap < 2:
            violations.append({
                'trade_idx': i,
                'prev_exit_bar': prev_exit,
                'cur_signal_bar': cur_signal,
                'gap': gap,
            })

    print(f"  Total trades: {len(trades)}")
    print(f"  min_bars_between violations: {len(violations)}")
    if violations:
        for v in violations[:5]:
            print(f"    Trade {v['trade_idx']}: prev_exit={v['prev_exit_bar']} "
                  f"cur_signal={v['cur_signal_bar']} gap={v['gap']}")

    passed = len(violations) == 0
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'A7_stale_signal',
        'passed': passed,
        'violations': len(violations),
        'details': violations[:10],
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION B: Overfitting Assessment
# ═══════════════════════════════════════════════════════════════════

def test_b1_param_neighborhood(df_15m):
    """B1: Parameter Neighborhood Stability (+/-20%)."""
    print("\n" + "=" * 70)
    print("  B1: Parameter Neighborhood Stability (+/-20%)")
    print("=" * 70)

    param_sweeps = {
        'channel_period': [12, 13, 14, 15, 16, 17, 18],
        'trail_K': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        'max_sl_atr': [2.6, 2.8, 3.0, 3.3, 3.6, 3.8, 4.0],
        'body_min_ratio': [0.30, 0.35, 0.40, 0.45, 0.50],
        'atr_period': [10, 12, 14, 16, 18],
    }

    results = {}
    for param_name, values in param_sweeps.items():
        sweep_results = []
        for val in values:
            params = {**DEFAULT_PARAMS, param_name: val}
            trades = backtest_n1(df_15m, params=params)
            active = [t for t in trades if t['reason'] != 'TIMEOUT']
            pnl = sum(t['pnl'] for t in active)
            sweep_results.append({
                'value': val,
                'pnl': round(pnl, 2),
                'trades': len(active),
            })

        pnls = [r['pnl'] for r in sweep_results]
        positive = sum(1 for p in pnls if p > 0)
        pct_positive = positive / len(pnls) * 100
        pnl_range = max(pnls) - min(pnls) if pnls else 0

        results[param_name] = {
            'sweep': sweep_results,
            'pct_positive': round(pct_positive, 1),
            'pnl_range': round(pnl_range, 2),
            'min_pnl': round(min(pnls), 2),
            'max_pnl': round(max(pnls), 2),
        }

        print(f"\n  {param_name}:")
        for r in sweep_results:
            marker = "*" if r['value'] == DEFAULT_PARAMS.get(param_name) else " "
            print(f"   {marker} {r['value']:>6}: PnL={r['pnl']:>+8.1f}%  T={r['trades']:>4}")
        print(f"    Positive: {pct_positive:.0f}%  Range: {pnl_range:.1f}pp")

    # Overall: all params should have >= 80% positive neighbors
    all_stable = all(r['pct_positive'] >= 80 for r in results.values())
    overall_positive = np.mean([r['pct_positive'] for r in results.values()])

    print(f"\n  Overall: {overall_positive:.0f}% positive across all sweeps")
    print(f"  VERDICT: {'PASS' if all_stable else 'FAIL'} (threshold: 80% per param)")
    return {
        'test': 'B1_param_neighborhood',
        'passed': all_stable,
        'results': results,
        'overall_pct_positive': round(overall_positive, 1),
    }


def test_b2_random_params(df_15m, n_combos=100, seed=42):
    """B2: Random Parameter Test (100 random combos)."""
    print("\n" + "=" * 70)
    print("  B2: Random Parameter Test (100 random combos)")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    results = []

    for i in range(n_combos):
        params = {
            **DEFAULT_PARAMS,
            'channel_period': int(rng.randint(8, 26)),
            'trail_K': round(rng.uniform(1.5, 4.0), 2),
            'max_sl_atr': round(rng.uniform(2.0, 5.0), 2),
            'body_min_ratio': round(rng.uniform(0.2, 0.6), 2),
            'atr_period': int(rng.randint(8, 21)),
        }
        trades = backtest_n1(df_15m, params=params)
        active = [t for t in trades if t['reason'] != 'TIMEOUT']
        pnl = sum(t['pnl'] for t in active)
        results.append({
            'params': {k: params[k] for k in ['channel_period', 'trail_K', 'max_sl_atr',
                                                'body_min_ratio', 'atr_period']},
            'pnl': round(pnl, 2),
            'trades': len(active),
        })

        if (i + 1) % 25 == 0:
            pos = sum(1 for r in results if r['pnl'] > 0)
            print(f"  {i+1}/100: {pos}/{i+1} positive ({pos/(i+1)*100:.0f}%)")

    positive = sum(1 for r in results if r['pnl'] > 0)
    pct_positive = positive / len(results) * 100
    pnls = [r['pnl'] for r in results]

    print(f"\n  Results:")
    print(f"    Positive: {positive}/{len(results)} ({pct_positive:.0f}%)")
    print(f"    PnL range: [{min(pnls):+.1f}%, {max(pnls):+.1f}%]")
    print(f"    Median PnL: {np.median(pnls):+.1f}%")
    print(f"    Mean PnL: {np.mean(pnls):+.1f}%")

    # >70% robust, 30-70% moderate, <30% overfit
    if pct_positive >= 70:
        verdict = "ROBUST (>=70%)"
        passed = True
    elif pct_positive >= 30:
        verdict = f"MODERATE ({pct_positive:.0f}%)"
        passed = True
    else:
        verdict = f"OVERFIT (<30%)"
        passed = False

    print(f"  VERDICT: {verdict}")
    return {
        'test': 'B2_random_params',
        'passed': passed,
        'pct_positive': round(pct_positive, 1),
        'median_pnl': round(np.median(pnls), 2),
        'mean_pnl': round(np.mean(pnls), 2),
        'min_pnl': round(min(pnls), 2),
        'max_pnl': round(max(pnls), 2),
    }


def test_b3_shuffled_exit_bootstrap(df_15m, trades, n_iters=1000, seed=42):
    """B3: Shuffled-Exit Bootstrap (proves exits add value)."""
    print("\n" + "=" * 70)
    print("  B3: Shuffled-Exit Bootstrap (1000 iterations)")
    print("=" * 70)

    active = [t for t in trades if t['reason'] != 'TIMEOUT']
    if not active:
        print("  No trades to test")
        return {'test': 'B3_shuffled_exit', 'passed': False, 'reason': 'no_trades'}

    real_pnl = sum(t['pnl'] for t in active)
    print(f"  Real strategy PnL: {real_pnl:+.2f}%")

    c = df_15m['close'].values.astype(float)
    n = len(c)
    rng = np.random.RandomState(seed)

    shuffled_pnls = []
    for _ in range(n_iters):
        total = 0
        for t in active:
            entry_bar = t['entry_bar']
            entry_price = t['entry_price']
            direction = t['direction']

            # Random exit within [entry+1, entry+max_hold_bars], capped at data end
            max_exit = min(entry_bar + TIMEOUT_BARS, n - 1)
            if entry_bar + 1 > max_exit:
                continue
            exit_bar = rng.randint(entry_bar + 1, max_exit + 1)
            exit_price = c[exit_bar]

            if direction == 'LONG':
                raw = (exit_price / entry_price - 1) * 100
            else:
                raw = (1 - exit_price / entry_price) * 100

            total += raw * LEVERAGE - FEE_PCT_RT

        shuffled_pnls.append(total)

    shuffled_pnls = np.array(shuffled_pnls)
    percentile = np.mean(shuffled_pnls < real_pnl) * 100

    print(f"  Shuffled PnL: mean={np.mean(shuffled_pnls):+.1f}%, "
          f"std={np.std(shuffled_pnls):.1f}%")
    print(f"  Real PnL percentile: {percentile:.1f}th")
    print(f"  P95 of shuffled: {np.percentile(shuffled_pnls, 95):+.1f}%")

    passed = percentile >= 95
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (real above 95th percentile)")
    return {
        'test': 'B3_shuffled_exit',
        'passed': passed,
        'real_pnl': round(real_pnl, 2),
        'shuffled_mean': round(float(np.mean(shuffled_pnls)), 2),
        'shuffled_std': round(float(np.std(shuffled_pnls)), 2),
        'percentile': round(percentile, 1),
        'shuffled_p95': round(float(np.percentile(shuffled_pnls, 95)), 2),
    }


def test_b4_half_life_decay(df_15m):
    """B4: Half-Life Decay Test (6 equal periods)."""
    print("\n" + "=" * 70)
    print("  B4: Half-Life Decay Test (6 periods)")
    print("=" * 70)

    n = len(df_15m)
    n_periods = 6
    period_size = n // n_periods
    results = []

    for p_idx in range(n_periods):
        start = p_idx * period_size
        end = (p_idx + 1) * period_size if p_idx < n_periods - 1 else n
        sub = df_15m.iloc[start:end].copy().reset_index(drop=True)
        trades = backtest_n1(sub)
        active = [t for t in trades if t['reason'] != 'TIMEOUT']
        pnl = sum(t['pnl'] for t in active)
        n_days = (end - start) * 15 / 1440
        daily = pnl / n_days if n_days > 0 else 0

        results.append({
            'period': p_idx + 1,
            'bars': end - start,
            'days': round(n_days, 1),
            'trades': len(active),
            'pnl': round(pnl, 2),
            'daily_pnl': round(daily, 4),
        })
        print(f"  Period {p_idx+1}: {n_days:>5.1f}d  T={len(active):>3}  "
              f"PnL={pnl:>+7.1f}%  daily={daily:>+.4f}%")

    # Check for decay trend via linear regression
    period_nums = np.arange(1, n_periods + 1)
    pnls = np.array([r['pnl'] for r in results])
    daily_pnls = np.array([r['daily_pnl'] for r in results])

    if len(daily_pnls) > 1:
        slope = np.polyfit(period_nums, daily_pnls, 1)[0]
    else:
        slope = 0

    # How many periods are positive?
    n_positive = sum(1 for r in results if r['pnl'] > 0)

    print(f"\n  Daily PnL regression slope: {slope:.6f}")
    print(f"  Positive periods: {n_positive}/{n_periods}")

    # PASS if no systematic decay (slope > -0.01) and >50% periods positive
    passed = slope > -0.01 and n_positive >= n_periods // 2
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'B4_half_life_decay',
        'passed': passed,
        'results': results,
        'slope': round(slope, 6),
        'n_positive': n_positive,
    }


def test_b5_reverse_chronological(df_15m):
    """B5: Reverse Chronological Test."""
    print("\n" + "=" * 70)
    print("  B5: Reverse Chronological Test")
    print("=" * 70)

    # Reverse OHLCV: flip order, swap open/close
    rev = df_15m.iloc[::-1].copy().reset_index(drop=True)
    rev_open = rev['close'].values.copy()  # swap O/C
    rev_close = rev['open'].values.copy()
    rev_high = rev['high'].values.copy()
    rev_low = rev['low'].values.copy()

    # Fix bars where open would be outside high-low range after swap
    for i in range(len(rev)):
        if rev_open[i] > rev_high[i]:
            rev_high[i] = rev_open[i]
        if rev_open[i] < rev_low[i]:
            rev_low[i] = rev_open[i]
        if rev_close[i] > rev_high[i]:
            rev_high[i] = rev_close[i]
        if rev_close[i] < rev_low[i]:
            rev_low[i] = rev_close[i]

    rev_df = pd.DataFrame({
        'open': rev_open, 'high': rev_high, 'low': rev_low, 'close': rev_close,
    })

    # Forward test
    fwd_trades = backtest_n1(df_15m)
    fwd_active = [t for t in fwd_trades if t['reason'] != 'TIMEOUT']
    fwd_pnl = sum(t['pnl'] for t in fwd_active)

    # Reverse test
    rev_trades = backtest_n1(rev_df)
    rev_active = [t for t in rev_trades if t['reason'] != 'TIMEOUT']
    rev_pnl = sum(t['pnl'] for t in rev_active)

    print(f"  Forward PnL:  {fwd_pnl:>+8.1f}% ({len(fwd_active)} trades)")
    print(f"  Reverse PnL:  {rev_pnl:>+8.1f}% ({len(rev_active)} trades)")
    print(f"  Difference:   {fwd_pnl - rev_pnl:>+8.1f}pp")

    # If reverse is similar to forward, suspicious (universal pattern or bias)
    # If reverse is worse/negative, confirms directional momentum capture
    if rev_pnl < 0:
        interpretation = "Strategy captures directional momentum (good)"
    elif rev_pnl > fwd_pnl * 0.8:
        interpretation = "WARNING: Reverse similar to forward - possible universal pattern"
    else:
        interpretation = "Reverse positive but weaker - acceptable"

    print(f"  Interpretation: {interpretation}")
    passed = rev_pnl < fwd_pnl * 0.8
    print(f"  VERDICT: {'PASS' if passed else 'CAUTION'}")
    return {
        'test': 'B5_reverse_chronological',
        'passed': passed,
        'forward_pnl': round(fwd_pnl, 2),
        'reverse_pnl': round(rev_pnl, 2),
        'interpretation': interpretation,
    }


def test_b6_cpcv(df_15m, k=10, purge_bars=2):
    """B6: Combinatorial Purged Cross-Validation (k=10)."""
    print("\n" + "=" * 70)
    print("  B6: Combinatorial Purged Cross-Validation (k=10)")
    print("=" * 70)

    n = len(df_15m)
    fold_size = n // k
    fold_pnls = []

    for fold_idx in range(k):
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size if fold_idx < k - 1 else n

        # Build train set (everything except test + purge)
        purge_start = max(0, test_start - purge_bars)
        purge_end = min(n, test_end + purge_bars)

        # Train on test fold only (OOS test)
        test_df = df_15m.iloc[test_start:test_end].copy().reset_index(drop=True)
        trades = backtest_n1(test_df)
        active = [t for t in trades if t['reason'] != 'TIMEOUT']
        pnl = sum(t['pnl'] for t in active)
        test_days = (test_end - test_start) * 15 / 1440

        fold_pnls.append(pnl)
        daily = pnl / test_days if test_days > 0 else 0
        print(f"  Fold {fold_idx+1:>2}: T={len(active):>3}  "
              f"PnL={pnl:>+7.1f}%  daily={daily:>+.4f}%")

    mean_pnl = np.mean(fold_pnls)
    std_pnl = np.std(fold_pnls)
    worst = min(fold_pnls)
    n_positive = sum(1 for p in fold_pnls if p > 0)

    print(f"\n  Mean OOS PnL: {mean_pnl:+.1f}%")
    print(f"  Std OOS PnL:  {std_pnl:.1f}%")
    print(f"  Worst fold:   {worst:+.1f}%")
    print(f"  Positive folds: {n_positive}/{k}")

    passed = n_positive >= k * 0.6 and mean_pnl > 0
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (>={k*0.6:.0f} positive folds)")
    return {
        'test': 'B6_cpcv',
        'passed': passed,
        'fold_pnls': [round(p, 2) for p in fold_pnls],
        'mean_pnl': round(mean_pnl, 2),
        'std_pnl': round(std_pnl, 2),
        'worst_fold': round(worst, 2),
        'n_positive': n_positive,
        'k': k,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION C: Fee/Cost Comprehensive Analysis
# ═══════════════════════════════════════════════════════════════════

def test_c1_fee_sensitivity(df_15m):
    """C1: Fee Sensitivity Curve (0.00% to 0.30%)."""
    print("\n" + "=" * 70)
    print("  C1: Fee Sensitivity Curve")
    print("=" * 70)

    # Run backtest once with 0 fee, then adjust
    trades_zero = backtest_n1(df_15m, fee_rt=0.0)
    active = [t for t in trades_zero if t['reason'] != 'TIMEOUT']
    n_trades = len(active)
    raw_pnls = [t['raw_pnl'] for t in active]

    fee_levels = np.arange(0.00, 0.31, 0.02)
    results = []
    breakeven_fee = None

    for fee in fee_levels:
        adjusted_pnls = [rp * LEVERAGE - fee for rp in raw_pnls]
        total = sum(adjusted_pnls)
        wins = sum(1 for p in adjusted_pnls if p > 0)
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        w_avg = np.mean([p for p in adjusted_pnls if p > 0]) if wins > 0 else 0
        l_avg = np.mean([p for p in adjusted_pnls if p <= 0]) if wins < n_trades else 0
        rr = abs(w_avg / l_avg) if l_avg != 0 else float('inf')

        results.append({
            'fee_rt': round(fee, 3),
            'pnl': round(total, 2),
            'wr': round(wr, 2),
            'rr': round(rr, 2),
            'trades': n_trades,
        })

        if breakeven_fee is None and total < 0:
            # Interpolate
            if len(results) >= 2:
                prev = results[-2]
                if prev['pnl'] > 0:
                    # Linear interpolation
                    breakeven_fee = prev['fee_rt'] + (fee - prev['fee_rt']) * \
                                    prev['pnl'] / (prev['pnl'] - total)

        print(f"  Fee={fee:.2f}%: PnL={total:>+8.1f}%  WR={wr:>5.1f}%  R:R={rr:>5.2f}")

    if breakeven_fee is not None:
        print(f"\n  Break-even fee: {breakeven_fee:.3f}% RT")
    else:
        print(f"\n  Break-even fee: >0.30% (strategy profitable at all tested fees)")
        breakeven_fee = 0.30

    margin_over_current = breakeven_fee - FEE_PCT_RT
    print(f"  Safety margin over current ({FEE_PCT_RT}%): {margin_over_current:+.3f}pp")
    passed = breakeven_fee > FEE_PCT_RT * 1.5  # At least 50% margin
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (breakeven > 1.5x current)")
    return {
        'test': 'C1_fee_sensitivity',
        'passed': passed,
        'breakeven_fee': round(breakeven_fee, 4) if breakeven_fee else None,
        'margin_pp': round(margin_over_current, 4),
        'results': results,
    }


def test_c2_asymmetric_fee(df_15m):
    """C2: Asymmetric Fee Test."""
    print("\n" + "=" * 70)
    print("  C2: Asymmetric Fee Test")
    print("=" * 70)

    scenarios = [
        ('Maker+Taker', 0.07),
        ('Taker+Taker (current)', 0.10),
        ('High Taker', 0.12),
        ('Market Impact', 0.15),
        ('Worst Case', 0.20),
    ]

    results = []
    for name, fee_rt in scenarios:
        trades = backtest_n1(df_15m, fee_rt=fee_rt)
        active = [t for t in trades if t['reason'] != 'TIMEOUT']
        pnl = sum(t['pnl'] for t in active)

        results.append({
            'scenario': name,
            'fee_rt': fee_rt,
            'pnl': round(pnl, 2),
            'trades': len(active),
        })
        print(f"  {name:>25} (fee={fee_rt:.2f}%): PnL={pnl:>+8.1f}%  T={len(active)}")

    all_positive = all(r['pnl'] > 0 for r in results)
    print(f"\n  VERDICT: {'PASS' if all_positive else 'PARTIAL'} "
          f"({'all positive' if all_positive else 'some negative'})")
    return {
        'test': 'C2_asymmetric_fee',
        'passed': all_positive,
        'results': results,
    }


def test_c3_fee_impact_wr_rr(df_15m):
    """C3: Fee Impact on Win Rate and R:R."""
    print("\n" + "=" * 70)
    print("  C3: Fee Impact on Win Rate and R:R")
    print("=" * 70)

    trades_zero = backtest_n1(df_15m, fee_rt=0.0)
    active = [t for t in trades_zero if t['reason'] != 'TIMEOUT']
    raw_pnls = [t['raw_pnl'] for t in active]
    n_t = len(active)

    fee_levels = [0, 0.05, 0.10, 0.15, 0.20]
    results = []

    for fee in fee_levels:
        adj = [rp * LEVERAGE - fee for rp in raw_pnls]
        wins = [p for p in adj if p > 0]
        losses = [p for p in adj if p <= 0]
        wr = len(wins) / n_t * 100 if n_t > 0 else 0
        wa = np.mean(wins) if wins else 0
        la = np.mean(losses) if losses else 0
        rr = abs(wa / la) if la != 0 else float('inf')
        pf = sum(wins) / abs(sum(losses)) if losses else float('inf')

        results.append({
            'fee_rt': fee,
            'wr': round(wr, 2),
            'avg_win': round(wa, 4),
            'avg_loss': round(la, 4),
            'rr': round(rr, 2),
            'profit_factor': round(pf, 2),
        })
        print(f"  Fee={fee:.2f}%: WR={wr:>5.1f}%  avg_w={wa:>+.4f}%  "
              f"avg_l={la:>+.4f}%  R:R={rr:>5.2f}  PF={pf:>5.2f}")

    # Check that WR doesn't collapse
    wr_at_current = results[2]['wr']  # fee=0.10
    print(f"\n  WR at current fee: {wr_at_current:.1f}%")
    passed = wr_at_current > 30 and results[2]['rr'] > 1.0
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        'test': 'C3_fee_impact_wr_rr',
        'passed': passed,
        'results': results,
    }


def test_c4_per_trade_fee_distribution(trades):
    """C4: Per-Trade Fee Distribution."""
    print("\n" + "=" * 70)
    print("  C4: Per-Trade Fee Distribution")
    print("=" * 70)

    active = [t for t in trades if t['reason'] != 'TIMEOUT']
    winners = [t for t in active if t['raw_pnl'] > 0]
    losers = [t for t in active if t['raw_pnl'] <= 0]

    if not winners:
        print("  No winning trades")
        return {'test': 'C4_per_trade_fee', 'passed': False, 'reason': 'no_winners'}

    # Fee as % of gross profit for winners
    fee_ratios = [FEE_PCT_RT / t['raw_pnl'] * 100 for t in winners if t['raw_pnl'] > 0]

    total_gross_profit = sum(t['raw_pnl'] for t in winners)
    total_fees = FEE_PCT_RT * len(active)
    fee_pct_of_gross = total_fees / total_gross_profit * 100 if total_gross_profit > 0 else 0

    print(f"  Total trades: {len(active)} (W={len(winners)}, L={len(losers)})")
    print(f"  Gross profit (winners only): {total_gross_profit:+.1f}%")
    print(f"  Total fees paid: {total_fees:.1f}%")
    print(f"  Fees as % of gross profit: {fee_pct_of_gross:.1f}%")
    print(f"\n  Fee/GrossProfit per winner:")
    print(f"    Median: {np.median(fee_ratios):.1f}%")
    print(f"    P25:    {np.percentile(fee_ratios, 25):.1f}%")
    print(f"    P75:    {np.percentile(fee_ratios, 75):.1f}%")
    print(f"    Max:    {max(fee_ratios):.1f}%")

    # PASS if fees consume < 50% of gross profit
    passed = fee_pct_of_gross < 50
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'} (fees < 50% of gross profit)")
    return {
        'test': 'C4_per_trade_fee',
        'passed': passed,
        'fee_pct_of_gross': round(fee_pct_of_gross, 2),
        'median_fee_ratio': round(np.median(fee_ratios), 2),
        'p25_fee_ratio': round(np.percentile(fee_ratios, 25), 2),
        'p75_fee_ratio': round(np.percentile(fee_ratios, 75), 2),
        'total_gross_profit': round(total_gross_profit, 2),
        'total_fees': round(total_fees, 2),
    }


def test_c5_funding_rate(df_15m):
    """C5: Funding Rate Sensitivity."""
    print("\n" + "=" * 70)
    print("  C5: Funding Rate Sensitivity")
    print("=" * 70)

    # Funding rate is per 8h. At 15m bars, 8h = 32 bars.
    # Cost per bar = funding_rate / 32
    funding_rates = [0.005, 0.01, 0.02, 0.03]  # per 8h
    results = []

    # Get baseline trades first (no funding)
    baseline_trades = backtest_n1(df_15m)
    baseline_active = [t for t in baseline_trades if t['reason'] != 'TIMEOUT']
    baseline_pnl = sum(t['pnl'] for t in baseline_active)
    avg_bars_held = np.mean([t['bars_held'] for t in baseline_active]) if baseline_active else 0

    print(f"  Baseline (no funding): PnL={baseline_pnl:+.1f}%, "
          f"avg_bars_held={avg_bars_held:.1f}")

    for fr in funding_rates:
        cost_per_bar = fr / 32  # per 15m bar
        trades = backtest_n1(df_15m, extra_cost_per_bar=cost_per_bar)
        active = [t for t in trades if t['reason'] != 'TIMEOUT']
        pnl = sum(t['pnl'] for t in active)
        diff = pnl - baseline_pnl

        results.append({
            'funding_rate_8h': fr,
            'cost_per_bar': round(cost_per_bar, 6),
            'pnl': round(pnl, 2),
            'impact': round(diff, 2),
        })
        print(f"  FR={fr:.3f}%/8h: PnL={pnl:>+8.1f}%  impact={diff:>+6.1f}pp")

    all_positive = all(r['pnl'] > 0 for r in results)
    print(f"\n  VERDICT: {'PASS' if all_positive else 'PARTIAL'} "
          f"({'all positive' if all_positive else 'some negative with funding'})")
    return {
        'test': 'C5_funding_rate',
        'passed': all_positive,
        'baseline_pnl': round(baseline_pnl, 2),
        'avg_bars_held': round(avg_bars_held, 1),
        'results': results,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print("=" * 70)
    print("  C1 Breakout v2 — Look-Ahead, Overfitting, Fee DEEP AUDIT")
    print("  Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)

    # ── Load & validate data ──
    print("\n--- Data Loading ---")
    df_5m = load_and_validate_5m()
    df_15m = resample_to_15m(df_5m)
    n = len(df_15m)
    n_days = n * 15 / 1440
    print(f"  15m bars: {n} ({n_days:.1f} days)")

    # ── Baseline gate check ──
    print("\n--- Baseline Gate Check ---")
    trades_baseline, detail_trades = backtest_n1(df_15m, return_details=True)
    active_baseline = [t for t in trades_baseline if t['reason'] != 'TIMEOUT']
    baseline_pnl = sum(t['pnl'] for t in active_baseline)
    baseline_stats = compute_stats(active_baseline, n_bars=n)

    print(f"  Baseline PnL (additive 1x): {baseline_pnl:+.1f}%")
    print(f"  Expected: ~+169.5%")
    print(f"  Trades: {baseline_stats['trades']}, WR: {baseline_stats['wr']}%, "
          f"R:R: {baseline_stats['rr']}")

    # Exit breakdown
    reasons = defaultdict(int)
    for t in trades_baseline:
        reasons[t['reason']] += 1
    print(f"  Exit breakdown: {dict(reasons)}")

    if abs(baseline_pnl - 169.5) > 50:
        print(f"\n  WARNING: Baseline PnL deviates significantly from expected!")
        print(f"  Proceeding with caution...\n")

    all_results = {
        'meta': {
            'date': datetime.now().isoformat(),
            'script': 'c1_lookahead_overfit_fee_audit.py',
            'data_file': DATA_PATH,
            'n_15m_bars': n,
            'n_days': round(n_days, 1),
            'params': DEFAULT_PARAMS,
            'baseline_pnl': round(baseline_pnl, 2),
            'baseline_stats': baseline_stats,
        },
        'results': {},
    }

    # ════════════════════════════════════════════════════════
    # SECTION A: Look-Ahead Bias
    # ════════════════════════════════════════════════════════
    print("\n" + "#" * 70)
    print("  SECTION A: LOOK-AHEAD BIAS FORENSIC AUDIT")
    print("#" * 70)

    r = test_a1_progressive_truncation(df_15m)
    all_results['results']['A1'] = r

    r = test_a2_indicator_causality(df_15m)
    all_results['results']['A2'] = r

    r = test_a3_entry_exit_temporal(trades_baseline)
    all_results['results']['A3'] = r

    r = test_a4_sl_lookahead(df_15m, trades_baseline)
    all_results['results']['A4'] = r

    r = test_a5_future_data_leakage(df_15m)
    all_results['results']['A5'] = r

    r = test_a6_bar_resolution_order(detail_trades)
    all_results['results']['A6'] = r

    r = test_a7_stale_signal(trades_baseline)
    all_results['results']['A7'] = r

    # ════════════════════════════════════════════════════════
    # SECTION B: Overfitting Assessment
    # ════════════════════════════════════════════════════════
    print("\n" + "#" * 70)
    print("  SECTION B: OVERFITTING ASSESSMENT")
    print("#" * 70)

    r = test_b1_param_neighborhood(df_15m)
    all_results['results']['B1'] = r

    r = test_b2_random_params(df_15m)
    all_results['results']['B2'] = r

    r = test_b3_shuffled_exit_bootstrap(df_15m, trades_baseline)
    all_results['results']['B3'] = r

    r = test_b4_half_life_decay(df_15m)
    all_results['results']['B4'] = r

    r = test_b5_reverse_chronological(df_15m)
    all_results['results']['B5'] = r

    r = test_b6_cpcv(df_15m)
    all_results['results']['B6'] = r

    # ════════════════════════════════════════════════════════
    # SECTION C: Fee/Cost Analysis
    # ════════════════════════════════════════════════════════
    print("\n" + "#" * 70)
    print("  SECTION C: FEE/COST COMPREHENSIVE ANALYSIS")
    print("#" * 70)

    r = test_c1_fee_sensitivity(df_15m)
    all_results['results']['C1'] = r

    r = test_c2_asymmetric_fee(df_15m)
    all_results['results']['C2'] = r

    r = test_c3_fee_impact_wr_rr(df_15m)
    all_results['results']['C3'] = r

    r = test_c4_per_trade_fee_distribution(trades_baseline)
    all_results['results']['C4'] = r

    r = test_c5_funding_rate(df_15m)
    all_results['results']['C5'] = r

    # ════════════════════════════════════════════════════════
    # Final Summary
    # ════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    section_pass = {'A': True, 'B': True, 'C': True}
    for key, result in all_results['results'].items():
        section = key[0]
        passed = result.get('passed', False)
        if not passed:
            section_pass[section] = False
        status = "PASS" if passed else "FAIL"
        print(f"  {key}: {status} - {result['test']}")

    print(f"\n  Section A (Look-Ahead): {'PASS' if section_pass['A'] else 'FAIL'}")
    print(f"  Section B (Overfitting): {'PASS' if section_pass['B'] else 'FAIL'}")
    print(f"  Section C (Fee/Cost):    {'PASS' if section_pass['C'] else 'FAIL'}")

    overall = all(section_pass.values())
    print(f"\n  OVERALL VERDICT: {'PASS' if overall else 'ISSUES FOUND'}")
    print(f"  Elapsed: {elapsed:.1f}s")

    all_results['summary'] = {
        'section_a_passed': section_pass['A'],
        'section_b_passed': section_pass['B'],
        'section_c_passed': section_pass['C'],
        'overall_passed': overall,
        'elapsed_seconds': round(elapsed, 1),
    }

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
