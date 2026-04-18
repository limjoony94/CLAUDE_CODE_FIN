"""
C1 Breakout REVERSE v2 — 30 Days Backtest
==========================================
역방향 전략 (mean-reversion):
- close > channel_high (상승 돌파) → **SHORT** (상승 감지 시 매도)
- close < channel_low (하락 돌파) → **LONG** (하락 감지 시 매수)

Risk management (동일):
- Fractal SL → 역방향에선 무의미 (돌파 방향이 뒤바뀜)
- **대칭 ATR 캡** (max_sl_atr × ATR)만 사용 for SL
- Emergency SL 3%, Timeout 192 bars (동일)
- Trail 2.5 × ATR (동일)

Output: trade-by-trade, summary vs 정방향 baseline.
"""

import sys
import os
import json
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)

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

FEE_RT_PCT = 0.10
LEVERAGE = 3
NUM_DAYS = 30


def load_and_resample(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp')
    df15 = df.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    return df15


def reverse_check_entry(bar_open, bar_high, bar_low, bar_close,
                        channel_high, channel_low, atr_val):
    """Reverse-direction entry: breakout UP → SHORT, breakout DOWN → LONG.

    SL is symmetric ATR cap only (fractal not meaningful for inverted strategy).
    """
    if (math.isnan(channel_high) or math.isnan(channel_low)
            or math.isnan(atr_val) or atr_val <= 0):
        return None
    if channel_high <= channel_low:
        return None

    # Detect breakout direction (same as normal)
    direction_detected = None
    if bar_close > channel_high:
        direction_detected = 'UP'
    elif bar_close < channel_low:
        direction_detected = 'DOWN'
    if direction_detected is None:
        return None

    # Body filter (same threshold)
    rng = bar_high - bar_low
    if rng <= 0:
        return None
    body = bar_close - bar_open
    if abs(body) / rng < CONFIG['body_min_ratio']:
        return None

    # Body direction must match breakout (same as normal)
    if direction_detected == 'UP' and body <= 0:
        return None
    if direction_detected == 'DOWN' and body >= 0:
        return None

    # REVERSE: flip direction
    if direction_detected == 'UP':
        trade_direction = 'SHORT'  # sell on detected rise
    else:
        trade_direction = 'LONG'   # buy on detected fall

    # SL: symmetric ATR cap only (fractal swing is on wrong side for reversed)
    entry_approx = bar_close
    sl_distance = CONFIG['max_sl_atr'] * atr_val
    if trade_direction == 'LONG':
        sl_price = entry_approx - sl_distance
    else:
        sl_price = entry_approx + sl_distance

    # Validate SL distance
    sl_pct = abs(entry_approx - sl_price) / entry_approx * 100
    if sl_pct < CONFIG['sl_min_pct'] or sl_pct > CONFIG['sl_max_pct']:
        return None

    return {
        'direction': trade_direction,
        'sl_price': sl_price,
        'sl_pct': sl_pct,
    }


def run_backtest(df15: pd.DataFrame, start_date: str, end_date: str):
    """Reverse strategy backtest. Uses existing check_exit (exit logic unchanged)."""
    signal = C1BreakoutSignal(CONFIG)  # only for check_exit

    opens = df15['open'].tolist()
    highs = df15['high'].tolist()
    lows = df15['low'].tolist()
    closes = df15['close'].tolist()
    timestamps = df15['timestamp'].tolist()
    n = len(closes)

    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_high, ch_low = compute_channel(highs, lows, CONFIG['channel_period'])

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    eval_start_idx = None
    for i in range(n):
        if timestamps[i] >= start_dt:
            eval_start_idx = i
            break
    eval_end_idx = n - 1
    for i in range(n - 1, -1, -1):
        if timestamps[i] <= end_dt:
            eval_end_idx = i
            break

    print(f"REVERSE strategy evaluation window: {timestamps[eval_start_idx]} to {timestamps[eval_end_idx]}")
    print(f"Bars: {eval_end_idx - eval_start_idx + 1}")
    print()

    trades = []
    in_position = False
    cooldown_until = 0
    pos_direction = None
    pos_entry_price = None
    pos_entry_time = None
    pos_sl_price = None
    pos_best_price = None
    pos_bars_held = 0

    for i in range(eval_start_idx, eval_end_idx + 1):
        if in_position:
            pos_bars_held += 1
            if pos_direction == 'LONG':
                pos_best_price = max(pos_best_price, highs[i])
            else:
                pos_best_price = min(pos_best_price, lows[i])

            exit_result = signal.check_exit(
                direction=pos_direction, entry_price=pos_entry_price,
                best_price=pos_best_price, current_high=highs[i],
                current_low=lows[i], current_close=closes[i],
                sl_price=pos_sl_price,
                atr_val=atr[i] if not math.isnan(atr[i]) else atr[i - 1],
                bars_held=pos_bars_held,
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

        if not in_position and i >= cooldown_until and i < eval_end_idx:
            if math.isnan(atr[i]) or math.isnan(ch_high[i]):
                continue

            # USE REVERSE ENTRY LOGIC
            entry_signal = reverse_check_entry(
                opens[i], highs[i], lows[i], closes[i],
                ch_high[i], ch_low[i], atr[i],
            )

            if entry_signal is not None:
                next_i = i + 1
                if next_i > eval_end_idx:
                    continue
                pos_direction = entry_signal['direction']
                pos_entry_price = opens[next_i]
                pos_entry_time = timestamps[next_i]
                pos_sl_price = entry_signal['sl_price']
                pos_bars_held = 0
                in_position = True
                if pos_direction == 'LONG':
                    pos_best_price = highs[next_i]
                else:
                    pos_best_price = lows[next_i]

                # Immediate exit check on entry bar
                exit_result = signal.check_exit(
                    direction=pos_direction, entry_price=pos_entry_price,
                    best_price=pos_best_price, current_high=highs[next_i],
                    current_low=lows[next_i], current_close=closes[next_i],
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

    return trades, timestamps, eval_start_idx, eval_end_idx


def summarize(trades):
    if not trades:
        return {}
    total = len(trades)
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    pnl_1x = sum(t['pnl_pct'] for t in trades)
    pnl_3x = sum(t['pnl_pct_3x'] for t in trades)
    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t['direction'] == 'SHORT']
    exits = defaultdict(int)
    for t in trades:
        exits[t['reason']] += 1
    # MDD (additive 1x equity curve)
    eq = 0
    peak = 0
    max_dd = 0
    for t in trades:
        eq += t['pnl_pct']
        peak = max(peak, eq)
        dd = peak - eq
        max_dd = max(max_dd, dd)
    avg_win = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0) / max(wins, 1)
    losses = total - wins
    avg_loss = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0) / max(losses, 1)
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    return {
        'total_trades': total,
        'wins': wins, 'losses': losses,
        'win_rate_pct': round(wins / total * 100, 1),
        'total_pnl_1x': round(pnl_1x, 4),
        'total_pnl_3x': round(pnl_3x, 4),
        'mdd_1x': round(max_dd, 4),
        'avg_win_pct': round(avg_win, 4),
        'avg_loss_pct': round(avg_loss, 4),
        'risk_reward': round(rr, 2),
        'long_count': len(longs), 'short_count': len(shorts),
        'long_pnl_1x': round(sum(t['pnl_pct'] for t in longs), 4),
        'short_pnl_1x': round(sum(t['pnl_pct'] for t in shorts), 4),
        'exit_breakdown': dict(exits),
    }


def main():
    csv_path = str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: data file not found: {csv_path}")
        sys.exit(1)
    df15 = load_and_resample(csv_path)
    # 같은 30일 기간 (c1_last_30days_backtest.py와 동일)
    end_dt = df15['timestamp'].iloc[-1]
    start_dt = end_dt - pd.Timedelta(days=NUM_DAYS)
    trades, ts, si, ei = run_backtest(df15, str(start_dt), str(end_dt))

    summary = summarize(trades)
    print("=" * 100)
    print("REVERSE STRATEGY — Summary")
    print("=" * 100)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()

    # Compare with normal-direction baseline
    normal_path = ROOT / 'results' / 'c1_last_30days_backtest.json'
    if normal_path.exists():
        with open(normal_path) as f:
            normal = json.load(f)['metrics']
        print("=" * 100)
        print("COMPARISON: NORMAL vs REVERSE (same 30-day window)")
        print("=" * 100)
        rows = [
            ('trades', normal['total_trades'], summary['total_trades']),
            ('WR %', normal['win_rate_pct'], summary['win_rate_pct']),
            ('PnL 1x %', normal['total_pnl_1x'], summary['total_pnl_1x']),
            ('PnL 3x %', normal['total_pnl_3x'], summary['total_pnl_3x']),
            ('MDD 1x %', normal['mdd_1x'], summary['mdd_1x']),
            ('RR', normal['risk_reward'], summary['risk_reward']),
            ('LONG pnl 1x', normal['long_pnl_1x'], summary['long_pnl_1x']),
            ('SHORT pnl 1x', normal['short_pnl_1x'], summary['short_pnl_1x']),
        ]
        for name, nv, rv in rows:
            delta = rv - nv if isinstance(nv, (int, float)) else '-'
            print(f"  {name:15s} normal={nv:10}  reverse={rv:10}  Δ={delta}")

    out_path = ROOT / 'results' / 'c1_reverse_30days_backtest.json'
    out = {
        'metadata': {
            'script': 'c1_reverse_30days_backtest.py',
            'date_run': datetime.now().isoformat(),
            'num_days': NUM_DAYS,
            'start': str(ts[si]),
            'end': str(ts[ei]),
            'strategy': 'REVERSE — sell on up-break, buy on down-break',
        },
        'summary': summary,
        'trades': trades,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
