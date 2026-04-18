"""
C1 Breakout v2 — Last 30 Days Backtest
========================================
Runs the C1 Breakout v2 strategy on the most recent 30 days of BTC data.
Uses production signal/indicator modules directly.

Output: trade-by-trade report, summary metrics, daily/weekly breakdown,
direction analysis, comparison with full-period expectations.
"""

import sys
import os
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)

# ── Config (production-identical) ──
CONFIG = {
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
    'fractal_lookback': 10,
}

FEE_RT_PCT = 0.10  # Round-trip fee (0.05% taker x2)
LEVERAGE = 3
NUM_DAYS = 30


def load_and_resample(csv_path: str) -> pd.DataFrame:
    """Load 5m CSV, resample to 15m."""
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp')

    df15 = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open']).reset_index()

    return df15


def run_backtest(df15: pd.DataFrame, start_date: str, end_date: str):
    """Run C1 Breakout backtest on the given 15m dataframe.

    start_date/end_date: ISO date strings for the evaluation window.
    Bars before start_date serve as warmup for indicators.
    """
    signal = C1BreakoutSignal(CONFIG)

    # Extract lists for indicator computation
    opens = df15['open'].tolist()
    highs = df15['high'].tolist()
    lows = df15['low'].tolist()
    closes = df15['close'].tolist()
    timestamps = df15['timestamp'].tolist()
    n = len(closes)

    # Compute indicators on FULL data (warmup included)
    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_high, ch_low = compute_channel(highs, lows, CONFIG['channel_period'])
    sw_low, sw_high = compute_fractal_swings(highs, lows, CONFIG['fractal_lookback'])

    # Find evaluation window indices
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    eval_start_idx = None
    for i in range(n):
        if timestamps[i] >= start_dt:
            eval_start_idx = i
            break
    if eval_start_idx is None:
        print(f"ERROR: No data found after {start_date}")
        return None

    eval_end_idx = n - 1
    for i in range(n - 1, -1, -1):
        if timestamps[i] <= end_dt:
            eval_end_idx = i
            break

    print(f"Evaluation window: {timestamps[eval_start_idx]} to {timestamps[eval_end_idx]}")
    print(f"Bars in window: {eval_end_idx - eval_start_idx + 1}")
    print(f"Total 15m bars (with warmup): {n}")
    print()

    # ── Backtest loop ──
    trades = []
    in_position = False
    cooldown_until = 0

    pos_direction = None
    pos_entry_price = None
    pos_entry_time = None
    pos_entry_bar = None
    pos_sl_price = None
    pos_best_price = None
    pos_bars_held = 0

    for i in range(eval_start_idx, eval_end_idx + 1):
        if in_position:
            pos_bars_held += 1

            # Update best_price BEFORE checking exit (intrabar tracking)
            if pos_direction == 'LONG':
                pos_best_price = max(pos_best_price, highs[i])
            else:
                pos_best_price = min(pos_best_price, lows[i])

            exit_result = signal.check_exit(
                direction=pos_direction,
                entry_price=pos_entry_price,
                best_price=pos_best_price,
                current_high=highs[i],
                current_low=lows[i],
                current_close=closes[i],
                sl_price=pos_sl_price,
                atr_val=atr[i] if not math.isnan(atr[i]) else atr[i - 1],
                bars_held=pos_bars_held,
            )

            if exit_result is not None:
                exit_price = exit_result['exit_price']
                reason = exit_result['reason']

                # PnL (1x, additive)
                if pos_direction == 'LONG':
                    pnl_pct = (exit_price / pos_entry_price - 1) * 100
                else:
                    pnl_pct = (1 - exit_price / pos_entry_price) * 100
                pnl_pct -= FEE_RT_PCT  # Deduct fees

                trades.append({
                    'trade_num': len(trades) + 1,
                    'direction': pos_direction,
                    'entry_time': str(pos_entry_time),
                    'entry_price': round(pos_entry_price, 1),
                    'exit_time': str(timestamps[i]),
                    'exit_price': round(exit_price, 1),
                    'pnl_pct': round(pnl_pct, 4),
                    'pnl_pct_3x': round(pnl_pct * LEVERAGE, 4),
                    'reason': reason,
                    'bars_held': pos_bars_held,
                })

                in_position = False
                cooldown_until = i + CONFIG['min_bars_between']
                pos_direction = None

        # Check entry (only if not in position and cooldown expired)
        if not in_position and i >= cooldown_until and i < eval_end_idx:
            if math.isnan(atr[i]) or math.isnan(ch_high[i]):
                continue

            entry_signal = signal.check_entry(
                bar_open=opens[i],
                bar_high=highs[i],
                bar_low=lows[i],
                bar_close=closes[i],
                channel_high=ch_high[i],
                channel_low=ch_low[i],
                atr_val=atr[i],
                last_swing_low=sw_low[i],
                last_swing_high=sw_high[i],
            )

            if entry_signal is not None:
                next_i = i + 1
                if next_i > eval_end_idx:
                    continue

                pos_direction = entry_signal['direction']
                pos_entry_price = opens[next_i]
                pos_entry_time = timestamps[next_i]
                pos_entry_bar = next_i
                pos_sl_price = entry_signal['sl_price']
                pos_bars_held = 0
                in_position = True

                # Initialize best_price with entry-bar high/low
                if pos_direction == 'LONG':
                    pos_best_price = highs[next_i]
                else:
                    pos_best_price = lows[next_i]

                # Check immediate exit on entry bar
                exit_result = signal.check_exit(
                    direction=pos_direction,
                    entry_price=pos_entry_price,
                    best_price=pos_best_price,
                    current_high=highs[next_i],
                    current_low=lows[next_i],
                    current_close=closes[next_i],
                    sl_price=pos_sl_price,
                    atr_val=atr[next_i] if not math.isnan(atr[next_i]) else atr[i],
                    bars_held=0,
                )

                if exit_result is not None:
                    exit_price = exit_result['exit_price']
                    reason = exit_result['reason']
                    if pos_direction == 'LONG':
                        pnl_pct = (exit_price / pos_entry_price - 1) * 100
                    else:
                        pnl_pct = (1 - exit_price / pos_entry_price) * 100
                    pnl_pct -= FEE_RT_PCT

                    trades.append({
                        'trade_num': len(trades) + 1,
                        'direction': pos_direction,
                        'entry_time': str(pos_entry_time),
                        'entry_price': round(pos_entry_price, 1),
                        'exit_time': str(timestamps[next_i]),
                        'exit_price': round(exit_price, 1),
                        'pnl_pct': round(pnl_pct, 4),
                        'pnl_pct_3x': round(pnl_pct * LEVERAGE, 4),
                        'reason': reason,
                        'bars_held': 0,
                    })

                    in_position = False
                    cooldown_until = next_i + CONFIG['min_bars_between']
                    pos_direction = None

    # ── BTC price context ──
    eval_opens = [opens[i] for i in range(eval_start_idx, eval_end_idx + 1)]
    eval_highs = [highs[i] for i in range(eval_start_idx, eval_end_idx + 1)]
    eval_lows = [lows[i] for i in range(eval_start_idx, eval_end_idx + 1)]
    eval_closes = [closes[i] for i in range(eval_start_idx, eval_end_idx + 1)]

    btc_context = {
        'period_open': round(eval_opens[0], 1),
        'period_close': round(eval_closes[-1], 1),
        'period_high': round(max(eval_highs), 1),
        'period_low': round(min(eval_lows), 1),
        'btc_return_pct': round((eval_closes[-1] / eval_opens[0] - 1) * 100, 2),
        'trend': 'BULLISH' if eval_closes[-1] > eval_opens[0] else 'BEARISH',
    }

    return trades, btc_context, timestamps[eval_start_idx], timestamps[eval_end_idx]


def compute_metrics(trades):
    """Compute summary metrics from trade list.
    TIMEOUT trades are excluded from PnL aggregates per research protocol.
    """
    if not trades:
        return {}

    # Separate timeout trades
    pnl_trades = [t for t in trades if t['reason'] != 'TIMEOUT']
    timeout_trades = [t for t in trades if t['reason'] == 'TIMEOUT']

    total = len(pnl_trades)
    if total == 0:
        return {'total_trades': 0, 'timeout_trades': len(timeout_trades)}

    wins = [t for t in pnl_trades if t['pnl_pct'] > 0]
    losses = [t for t in pnl_trades if t['pnl_pct'] <= 0]
    wr = len(wins) / total * 100

    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    total_pnl_1x = sum(t['pnl_pct'] for t in pnl_trades)
    total_pnl_3x = sum(t['pnl_pct_3x'] for t in pnl_trades)

    # MDD (additive 1x equity curve)
    equity = 0
    peak = 0
    max_dd = 0
    for t in pnl_trades:
        equity += t['pnl_pct']
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Max consecutive losses
    max_consec_loss = 0
    cur_consec = 0
    for t in pnl_trades:
        if t['pnl_pct'] <= 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    # Long/Short breakdown
    longs = [t for t in pnl_trades if t['direction'] == 'LONG']
    shorts = [t for t in pnl_trades if t['direction'] == 'SHORT']
    long_wins = [t for t in longs if t['pnl_pct'] > 0]
    short_wins = [t for t in shorts if t['pnl_pct'] > 0]

    # Exit breakdown (all trades including timeout)
    exit_counts = {}
    for t in trades:
        r = t['reason']
        exit_counts[r] = exit_counts.get(r, 0) + 1

    # Best / Worst trade
    best_trade = max(pnl_trades, key=lambda t: t['pnl_pct'])
    worst_trade = min(pnl_trades, key=lambda t: t['pnl_pct'])

    return {
        'total_trades': total,
        'timeout_trades': len(timeout_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_pct': round(wr, 1),
        'avg_win_pct': round(avg_win, 4),
        'avg_loss_pct': round(avg_loss, 4),
        'risk_reward': round(rr, 2),
        'total_pnl_1x': round(total_pnl_1x, 4),
        'total_pnl_3x': round(total_pnl_3x, 4),
        'mdd_1x': round(max_dd, 4),
        'mdd_3x': round(max_dd * LEVERAGE, 4),
        'max_consecutive_losses': max_consec_loss,
        'long_count': len(longs),
        'long_wins': len(long_wins),
        'long_wr_pct': round(len(long_wins) / len(longs) * 100, 1) if longs else 0,
        'long_pnl_1x': round(sum(t['pnl_pct'] for t in longs), 4),
        'short_count': len(shorts),
        'short_wins': len(short_wins),
        'short_wr_pct': round(len(short_wins) / len(shorts) * 100, 1) if shorts else 0,
        'short_pnl_1x': round(sum(t['pnl_pct'] for t in shorts), 4),
        'exit_breakdown': exit_counts,
        'avg_bars_held': round(sum(t['bars_held'] for t in pnl_trades) / total, 1),
        'best_trade': {
            'trade_num': best_trade['trade_num'],
            'pnl_pct': best_trade['pnl_pct'],
            'pnl_pct_3x': best_trade['pnl_pct_3x'],
            'direction': best_trade['direction'],
            'entry_time': best_trade['entry_time'],
        },
        'worst_trade': {
            'trade_num': worst_trade['trade_num'],
            'pnl_pct': worst_trade['pnl_pct'],
            'pnl_pct_3x': worst_trade['pnl_pct_3x'],
            'direction': worst_trade['direction'],
            'entry_time': worst_trade['entry_time'],
        },
    }


def daily_breakdown(trades, start_date, end_date):
    """PnL per calendar day, including zero-trade days.
    TIMEOUT trades excluded from PnL.
    """
    daily = defaultdict(lambda: {'trades': 0, 'pnl_1x': 0.0, 'pnl_3x': 0.0, 'wins': 0, 'losses': 0})

    for t in trades:
        if t['reason'] == 'TIMEOUT':
            continue
        day = t['exit_time'][:10]
        daily[day]['trades'] += 1
        daily[day]['pnl_1x'] += t['pnl_pct']
        daily[day]['pnl_3x'] += t['pnl_pct_3x']
        if t['pnl_pct'] > 0:
            daily[day]['wins'] += 1
        else:
            daily[day]['losses'] += 1

    # Fill zero-trade days
    start_d = pd.Timestamp(start_date).date()
    end_d = pd.Timestamp(end_date).date()
    cur = start_d
    while cur <= end_d:
        day_str = str(cur)
        if day_str not in daily:
            daily[day_str] = {'trades': 0, 'pnl_1x': 0.0, 'pnl_3x': 0.0, 'wins': 0, 'losses': 0}
        cur += timedelta(days=1)

    # Round
    for d in daily:
        daily[d]['pnl_1x'] = round(daily[d]['pnl_1x'], 4)
        daily[d]['pnl_3x'] = round(daily[d]['pnl_3x'], 4)

    return dict(daily)


def weekly_breakdown(daily_data):
    """Group daily data into weekly chunks (Mon-Sun ISO weeks)."""
    weekly = defaultdict(lambda: {'trades': 0, 'pnl_1x': 0.0, 'pnl_3x': 0.0, 'wins': 0, 'losses': 0, 'days': 0})

    for day_str, d in sorted(daily_data.items()):
        dt = pd.Timestamp(day_str)
        iso_year, iso_week, _ = dt.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"
        weekly[week_key]['trades'] += d['trades']
        weekly[week_key]['pnl_1x'] += d['pnl_1x']
        weekly[week_key]['pnl_3x'] += d['pnl_3x']
        weekly[week_key]['wins'] += d['wins']
        weekly[week_key]['losses'] += d['losses']
        weekly[week_key]['days'] += 1

    # Round
    for w in weekly:
        weekly[w]['pnl_1x'] = round(weekly[w]['pnl_1x'], 4)
        weekly[w]['pnl_3x'] = round(weekly[w]['pnl_3x'], 4)
        total_w = weekly[w]['wins'] + weekly[w]['losses']
        weekly[w]['wr_pct'] = round(weekly[w]['wins'] / total_w * 100, 1) if total_w > 0 else 0.0

    return dict(weekly)


def main():
    csv_path = str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv')

    print("=" * 100)
    print("C1 BREAKOUT v2 — LAST 30 DAYS BACKTEST")
    print("=" * 100)
    print()

    # Load and resample
    print("Loading and resampling 5m -> 15m...")
    df15 = load_and_resample(csv_path)
    print(f"Total 15m bars: {len(df15)}")
    print(f"Data range: {df15['timestamp'].iloc[0]} to {df15['timestamp'].iloc[-1]}")
    print()

    # Determine last 30 days
    last_ts = df15['timestamp'].iloc[-1]
    start_30d = last_ts - timedelta(days=NUM_DAYS)
    print(f"Last {NUM_DAYS} days: {start_30d.strftime('%Y-%m-%d %H:%M')} to {last_ts.strftime('%Y-%m-%d %H:%M')}")
    print()

    # Run backtest
    result = run_backtest(df15, str(start_30d), str(last_ts))
    if result is None:
        print("Backtest failed.")
        return

    trades, btc_context, eval_start, eval_end = result

    # Trades excluding timeouts for PnL
    pnl_trades = [t for t in trades if t['reason'] != 'TIMEOUT']

    # ══════════════════════════════════════════════════════════════════════
    # 1. BTC PRICE CONTEXT
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("1. BTC PRICE CONTEXT (30-day period)")
    print("=" * 100)
    print(f"  Open:   ${btc_context['period_open']:,.1f}")
    print(f"  Close:  ${btc_context['period_close']:,.1f}")
    print(f"  High:   ${btc_context['period_high']:,.1f}")
    print(f"  Low:    ${btc_context['period_low']:,.1f}")
    print(f"  Return: {btc_context['btc_return_pct']:+.2f}%")
    print(f"  Trend:  {btc_context['trend']}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 2. SUMMARY METRICS
    # ══════════════════════════════════════════════════════════════════════
    metrics = compute_metrics(trades)
    print("=" * 100)
    print("2. SUMMARY METRICS")
    print("=" * 100)
    if metrics and metrics.get('total_trades', 0) > 0:
        actual_daily_1x = metrics['total_pnl_1x'] / NUM_DAYS
        actual_daily_3x = metrics['total_pnl_3x'] / NUM_DAYS
        actual_trades_per_day = metrics['total_trades'] / NUM_DAYS

        print(f"  Total Trades:      {metrics['total_trades']} (excl. {metrics['timeout_trades']} timeouts)")
        print(f"  Win Rate:          {metrics['win_rate_pct']:.1f}%")
        print(f"  Avg Win:           {metrics['avg_win_pct']:+.4f}%")
        print(f"  Avg Loss:          {metrics['avg_loss_pct']:+.4f}%")
        print(f"  Risk:Reward:       {metrics['risk_reward']:.2f}")
        print(f"  Total PnL (1x):   {metrics['total_pnl_1x']:+.4f}%")
        print(f"  Total PnL (3x):   {metrics['total_pnl_3x']:+.4f}%")
        print(f"  Daily PnL (1x):   {actual_daily_1x:+.4f}%")
        print(f"  Daily PnL (3x):   {actual_daily_3x:+.4f}%")
        print(f"  Trades/day:        {actual_trades_per_day:.1f}")
        print(f"  MDD (1x):          {metrics['mdd_1x']:.4f}%")
        print(f"  MDD (3x):          {metrics['mdd_3x']:.4f}%")
        print(f"  Avg Bars Held:     {metrics['avg_bars_held']}")
        print(f"  Max Consec Losses: {metrics['max_consecutive_losses']}")
    else:
        print("  No trades in this period.")
        actual_daily_1x = 0
        actual_daily_3x = 0
        actual_trades_per_day = 0
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 3. TRADE-BY-TRADE TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("3. TRADE-BY-TRADE REPORT")
    print("=" * 100)
    if not trades:
        print("  No trades in this period.")
    else:
        header = f"{'#':>3} {'Dir':>5} {'Entry Time':>20} {'Entry$':>10} {'Exit Time':>20} {'Exit$':>10} {'PnL%':>8} {'3xPnL%':>8} {'Reason':>10} {'Bars':>5}"
        print(header)
        print("-" * len(header))
        for t in trades:
            print(f"{t['trade_num']:>3} {t['direction']:>5} {t['entry_time']:>20} {t['entry_price']:>10,.1f} "
                  f"{t['exit_time']:>20} {t['exit_price']:>10,.1f} {t['pnl_pct']:>+8.4f} {t['pnl_pct_3x']:>+8.4f} "
                  f"{t['reason']:>10} {t['bars_held']:>5}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 4. WEEKLY BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════
    daily = daily_breakdown(trades, str(eval_start), str(eval_end))
    weekly = weekly_breakdown(daily)

    print("=" * 100)
    print("4. WEEKLY BREAKDOWN")
    print("=" * 100)
    if weekly:
        print(f"  {'Week':>10} {'Days':>5} {'Trades':>7} {'WR%':>6} {'PnL 1x':>10} {'PnL 3x':>10}")
        print(f"  {'-' * 52}")
        for week_key in sorted(weekly.keys()):
            w = weekly[week_key]
            print(f"  {week_key:>10} {w['days']:>5} {w['trades']:>7} {w['wr_pct']:>5.1f}% {w['pnl_1x']:>+10.4f} {w['pnl_3x']:>+10.4f}")

        # Best/worst week
        sorted_weeks = sorted(weekly.items(), key=lambda x: x[1]['pnl_1x'])
        best_week = sorted_weeks[-1]
        worst_week = sorted_weeks[0]
        print()
        print(f"  Best week:  {best_week[0]} PnL 1x={best_week[1]['pnl_1x']:+.4f}% ({best_week[1]['trades']} trades)")
        print(f"  Worst week: {worst_week[0]} PnL 1x={worst_week[1]['pnl_1x']:+.4f}% ({worst_week[1]['trades']} trades)")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 5. DAILY BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("5. DAILY BREAKDOWN (all 30 days)")
    print("=" * 100)
    if daily:
        print(f"  {'Date':>12} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'PnL 1x':>10} {'PnL 3x':>10}")
        print(f"  {'-' * 52}")
        for day in sorted(daily.keys()):
            d = daily[day]
            print(f"  {day:>12} {d['trades']:>7} {d['wins']:>5} {d['losses']:>5} {d['pnl_1x']:>+10.4f} {d['pnl_3x']:>+10.4f}")

        # Best/worst day
        active_days = {k: v for k, v in daily.items() if v['trades'] > 0}
        if active_days:
            sorted_days = sorted(active_days.items(), key=lambda x: x[1]['pnl_1x'])
            best_day = sorted_days[-1]
            worst_day = sorted_days[0]
            print()
            print(f"  Best day:  {best_day[0]} PnL 1x={best_day[1]['pnl_1x']:+.4f}% ({best_day[1]['trades']} trades)")
            print(f"  Worst day: {worst_day[0]} PnL 1x={worst_day[1]['pnl_1x']:+.4f}% ({worst_day[1]['trades']} trades)")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 6. DIRECTION BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("6. DIRECTION BREAKDOWN")
    print("=" * 100)
    if metrics and metrics.get('total_trades', 0) > 0:
        print(f"  {'Direction':>10} {'Count':>7} {'Wins':>6} {'WR%':>7} {'PnL 1x':>10}")
        print(f"  {'-' * 42}")
        print(f"  {'LONG':>10} {metrics['long_count']:>7} {metrics['long_wins']:>6} {metrics['long_wr_pct']:>6.1f}% {metrics['long_pnl_1x']:>+10.4f}")
        print(f"  {'SHORT':>10} {metrics['short_count']:>7} {metrics['short_wins']:>6} {metrics['short_wr_pct']:>6.1f}% {metrics['short_pnl_1x']:>+10.4f}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 7. EXIT REASON BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("7. EXIT REASON BREAKDOWN")
    print("=" * 100)
    if metrics and metrics.get('exit_breakdown'):
        total_all = sum(metrics['exit_breakdown'].values())
        for reason, count in sorted(metrics['exit_breakdown'].items()):
            pct = count / total_all * 100
            print(f"    {reason:>12}: {count:>3} ({pct:.1f}%)")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 8. BEST / WORST / EXTREMES
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("8. BEST / WORST EXTREMES")
    print("=" * 100)
    if metrics and metrics.get('total_trades', 0) > 0:
        bt = metrics['best_trade']
        wt = metrics['worst_trade']
        print(f"  Best trade:   #{bt['trade_num']} {bt['direction']} {bt['entry_time']} PnL={bt['pnl_pct']:+.4f}% (3x: {bt['pnl_pct_3x']:+.4f}%)")
        print(f"  Worst trade:  #{wt['trade_num']} {wt['direction']} {wt['entry_time']} PnL={wt['pnl_pct']:+.4f}% (3x: {wt['pnl_pct_3x']:+.4f}%)")
        print(f"  Max consec losses: {metrics['max_consecutive_losses']}")

        if active_days:
            print(f"  Best day:     {best_day[0]} PnL 1x={best_day[1]['pnl_1x']:+.4f}%")
            print(f"  Worst day:    {worst_day[0]} PnL 1x={worst_day[1]['pnl_1x']:+.4f}%")
        if weekly:
            print(f"  Best week:    {best_week[0]} PnL 1x={best_week[1]['pnl_1x']:+.4f}%")
            print(f"  Worst week:   {worst_week[0]} PnL 1x={worst_week[1]['pnl_1x']:+.4f}%")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # 9. COMPARISON WITH FULL-PERIOD EXPECTATIONS
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("9. COMPARISON WITH FULL-PERIOD EXPECTATIONS")
    print("=" * 100)
    expected = {
        'daily_pnl_1x': 0.509,
        'trades_per_day': 3.1,
        'win_rate': 36.6,
        'risk_reward': 3.36,
        'mdd_1x': 5.4,
    }
    actual_wr = metrics.get('win_rate_pct', 0) if metrics else 0
    actual_rr = metrics.get('risk_reward', 0) if metrics else 0
    actual_mdd = metrics.get('mdd_1x', 0) if metrics else 0

    print(f"  {'Metric':>20} {'Expected (333d)':>15} {'Actual (30d)':>14} {'Delta':>10}")
    print(f"  {'-' * 62}")
    print(f"  {'Daily PnL (1x)':>20} {'+0.509%':>15} {actual_daily_1x:>+13.4f}% {actual_daily_1x - 0.509:>+10.4f}")
    print(f"  {'Trades/day':>20} {'3.1':>15} {actual_trades_per_day:>13.1f}  {actual_trades_per_day - 3.1:>+10.1f}")
    print(f"  {'Win Rate':>20} {'36.6%':>15} {actual_wr:>12.1f}%  {actual_wr - 36.6:>+10.1f}")
    print(f"  {'R:R':>20} {'3.36':>15} {actual_rr:>13.2f} {actual_rr - 3.36:>+10.2f}")
    print(f"  {'MDD (1x)':>20} {'5.4%':>15} {actual_mdd:>12.4f}%  {actual_mdd - 5.4:>+10.4f}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # SAVE JSON
    # ══════════════════════════════════════════════════════════════════════
    output = {
        'metadata': {
            'script': 'c1_last_30days_backtest.py',
            'date_run': datetime.now().isoformat(),
            'data_file': csv_path,
            'eval_start': str(eval_start),
            'eval_end': str(eval_end),
            'num_days': NUM_DAYS,
            'config': CONFIG,
            'fee_rt_pct': FEE_RT_PCT,
            'leverage': LEVERAGE,
        },
        'btc_context': btc_context,
        'trades': trades,
        'metrics': metrics,
        'daily_breakdown': daily,
        'weekly_breakdown': weekly,
        'comparison': {
            'expected_daily_pnl_1x': 0.509,
            'actual_daily_pnl_1x': round(actual_daily_1x, 4),
            'expected_trades_per_day': 3.1,
            'actual_trades_per_day': round(actual_trades_per_day, 1),
            'expected_wr': 36.6,
            'actual_wr': actual_wr,
            'expected_rr': 3.36,
            'actual_rr': actual_rr,
            'expected_mdd_1x': 5.4,
            'actual_mdd_1x': actual_mdd,
        },
        'extremes': {
            'best_trade': metrics.get('best_trade') if metrics else None,
            'worst_trade': metrics.get('worst_trade') if metrics else None,
            'max_consecutive_losses': metrics.get('max_consecutive_losses', 0) if metrics else 0,
            'best_day': best_day[0] if active_days else None,
            'best_day_pnl_1x': best_day[1]['pnl_1x'] if active_days else None,
            'worst_day': worst_day[0] if active_days else None,
            'worst_day_pnl_1x': worst_day[1]['pnl_1x'] if active_days else None,
            'best_week': best_week[0] if weekly else None,
            'best_week_pnl_1x': best_week[1]['pnl_1x'] if weekly else None,
            'worst_week': worst_week[0] if weekly else None,
            'worst_week_pnl_1x': worst_week[1]['pnl_1x'] if weekly else None,
        },
    }

    results_path = str(ROOT / 'results' / 'c1_last_30days_backtest.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
