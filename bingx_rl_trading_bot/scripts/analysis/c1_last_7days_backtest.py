"""
C1 Breakout v2 — Last 7 Days Backtest
========================================
Runs the C1 Breakout v2 strategy on the most recent 7 days of BTC data.
Uses production signal/indicator modules directly.

Output: trade-by-trade report, summary metrics, daily breakdown, comparison.
"""

import sys
import os
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

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
    cooldown_until = 0  # bar index when cooldown expires

    # Position state
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
                # Entry at next bar open
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
    """Compute summary metrics from trade list."""
    if not trades:
        return {}

    total = len(trades)
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    wr = len(wins) / total * 100

    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    total_pnl_1x = sum(t['pnl_pct'] for t in trades)
    total_pnl_3x = sum(t['pnl_pct_3x'] for t in trades)

    # MDD (additive 1x equity curve)
    equity = 0
    peak = 0
    max_dd = 0
    for t in trades:
        equity += t['pnl_pct']
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Long/Short breakdown
    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t['direction'] == 'SHORT']

    # Exit breakdown
    exit_counts = {}
    for t in trades:
        r = t['reason']
        exit_counts[r] = exit_counts.get(r, 0) + 1

    return {
        'total_trades': total,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_pct': round(wr, 1),
        'avg_win_pct': round(avg_win, 4),
        'avg_loss_pct': round(avg_loss, 4),
        'risk_reward': round(rr, 2),
        'total_pnl_1x': round(total_pnl_1x, 4),
        'total_pnl_3x': round(total_pnl_3x, 4),
        'mdd_1x': round(max_dd, 4),
        'long_count': len(longs),
        'long_pnl_1x': round(sum(t['pnl_pct'] for t in longs), 4),
        'short_count': len(shorts),
        'short_pnl_1x': round(sum(t['pnl_pct'] for t in shorts), 4),
        'exit_breakdown': exit_counts,
        'avg_bars_held': round(sum(t['bars_held'] for t in trades) / total, 1),
    }


def daily_breakdown(trades):
    """PnL per calendar day."""
    daily = {}
    for t in trades:
        day = t['exit_time'][:10]
        if day not in daily:
            daily[day] = {'trades': 0, 'pnl_1x': 0, 'pnl_3x': 0}
        daily[day]['trades'] += 1
        daily[day]['pnl_1x'] += t['pnl_pct']
        daily[day]['pnl_3x'] += t['pnl_pct_3x']

    # Round
    for d in daily:
        daily[d]['pnl_1x'] = round(daily[d]['pnl_1x'], 4)
        daily[d]['pnl_3x'] = round(daily[d]['pnl_3x'], 4)

    return daily


def main():
    csv_path = str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv')

    print("=" * 80)
    print("C1 BREAKOUT v2 — LAST 7 DAYS BACKTEST")
    print("=" * 80)
    print()

    # Load and resample
    print("Loading and resampling 5m -> 15m...")
    df15 = load_and_resample(csv_path)
    print(f"Total 15m bars: {len(df15)}")
    print(f"Data range: {df15['timestamp'].iloc[0]} to {df15['timestamp'].iloc[-1]}")
    print()

    # Determine last 7 days
    last_ts = df15['timestamp'].iloc[-1]
    start_7d = last_ts - timedelta(days=7)
    print(f"Last 7 days: {start_7d.strftime('%Y-%m-%d %H:%M')} to {last_ts.strftime('%Y-%m-%d %H:%M')}")
    print()

    # Run backtest
    result = run_backtest(df15, str(start_7d), str(last_ts))
    if result is None:
        print("Backtest failed.")
        return

    trades, btc_context, eval_start, eval_end = result

    # ── BTC Price Context ──
    print("=" * 80)
    print("BTC PRICE CONTEXT (7-day period)")
    print("=" * 80)
    print(f"  Open:   ${btc_context['period_open']:,.1f}")
    print(f"  Close:  ${btc_context['period_close']:,.1f}")
    print(f"  High:   ${btc_context['period_high']:,.1f}")
    print(f"  Low:    ${btc_context['period_low']:,.1f}")
    print(f"  Return: {btc_context['btc_return_pct']:+.2f}%")
    print(f"  Trend:  {btc_context['trend']}")
    print()

    # ── Trade-by-Trade Table ──
    print("=" * 80)
    print("TRADE-BY-TRADE REPORT")
    print("=" * 80)
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

    # ── Summary Metrics ──
    metrics = compute_metrics(trades)
    print("=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)
    if metrics:
        print(f"  Total Trades:    {metrics['total_trades']}")
        print(f"  Win Rate:        {metrics['win_rate_pct']:.1f}%")
        print(f"  Avg Win:         {metrics['avg_win_pct']:+.4f}%")
        print(f"  Avg Loss:        {metrics['avg_loss_pct']:+.4f}%")
        print(f"  Risk:Reward:     {metrics['risk_reward']:.2f}")
        print(f"  Total PnL (1x):  {metrics['total_pnl_1x']:+.4f}%")
        print(f"  Total PnL (3x):  {metrics['total_pnl_3x']:+.4f}%")
        print(f"  MDD (1x):        {metrics['mdd_1x']:.4f}%")
        print(f"  Avg Bars Held:   {metrics['avg_bars_held']}")
        print()
        print(f"  LONG:  {metrics['long_count']} trades, PnL {metrics['long_pnl_1x']:+.4f}%")
        print(f"  SHORT: {metrics['short_count']} trades, PnL {metrics['short_pnl_1x']:+.4f}%")
        print()
        print("  Exit Breakdown:")
        for reason, count in sorted(metrics['exit_breakdown'].items()):
            pct = count / metrics['total_trades'] * 100
            print(f"    {reason:>12}: {count:>3} ({pct:.1f}%)")
    print()

    # ── Daily Breakdown ──
    daily = daily_breakdown(trades)
    print("=" * 80)
    print("DAILY BREAKDOWN")
    print("=" * 80)
    if daily:
        print(f"  {'Date':>12} {'Trades':>7} {'PnL 1x':>10} {'PnL 3x':>10}")
        print(f"  {'-' * 42}")
        for day in sorted(daily.keys()):
            d = daily[day]
            print(f"  {day:>12} {d['trades']:>7} {d['pnl_1x']:>+10.4f} {d['pnl_3x']:>+10.4f}")
    print()

    # ── Comparison with Full-Period Expectations ──
    num_days = 7
    if metrics:
        actual_daily_1x = metrics['total_pnl_1x'] / num_days
        actual_trades_per_day = metrics['total_trades'] / num_days
    else:
        actual_daily_1x = 0
        actual_trades_per_day = 0

    print("=" * 80)
    print("COMPARISON WITH FULL-PERIOD EXPECTATIONS")
    print("=" * 80)
    print(f"  {'Metric':>20} {'Expected':>12} {'Actual (7d)':>14} {'Delta':>10}")
    print(f"  {'-' * 58}")
    print(f"  {'Daily PnL (1x)':>20} {'+0.509%':>12} {actual_daily_1x:>+13.4f}% {actual_daily_1x - 0.509:>+10.4f}")
    print(f"  {'Trades/day':>20} {'3.1':>12} {actual_trades_per_day:>13.1f}  {actual_trades_per_day - 3.1:>+10.1f}")
    expected_wr = 36.6
    actual_wr = metrics.get('win_rate_pct', 0)
    print(f"  {'Win Rate':>20} {'36.6%':>12} {actual_wr:>12.1f}%  {actual_wr - expected_wr:>+10.1f}")
    expected_rr = 3.36
    actual_rr = metrics.get('risk_reward', 0)
    print(f"  {'R:R':>20} {'3.36':>12} {actual_rr:>13.2f} {actual_rr - expected_rr:>+10.2f}")
    print()

    # ── Save JSON ──
    output = {
        'metadata': {
            'script': 'c1_last_7days_backtest.py',
            'date_run': datetime.now().isoformat(),
            'data_file': csv_path,
            'eval_start': str(eval_start),
            'eval_end': str(eval_end),
            'config': CONFIG,
            'fee_rt_pct': FEE_RT_PCT,
            'leverage': LEVERAGE,
        },
        'btc_context': btc_context,
        'trades': trades,
        'metrics': metrics,
        'daily_breakdown': daily,
        'comparison': {
            'expected_daily_pnl_1x': 0.509,
            'actual_daily_pnl_1x': round(actual_daily_1x, 4),
            'expected_trades_per_day': 3.1,
            'actual_trades_per_day': round(actual_trades_per_day, 1),
            'expected_wr': 36.6,
            'actual_wr': actual_wr,
            'expected_rr': 3.36,
            'actual_rr': actual_rr,
        },
    }

    results_path = str(ROOT / 'results' / 'c1_last_7days_backtest.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
