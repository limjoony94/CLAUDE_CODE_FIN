"""
BT vs LIVE Drawdown Comparison — 2026-04-12 to 2026-04-21
===========================================================
Fetches BingX 15m BTC data for live trading period and runs the C1 Breakout v2
backtest identically. Produces side-by-side equity curve / DD comparison.

Usage: python scripts/analysis/dd_comparison_20260421.py
"""

import sys
import os
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

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

FEE_RT_PCT = 0.10
LEVERAGE = 3
START_BALANCE = 2100.0

# Live trades (from state.json, 27 trades, 3x leverage already applied)
LIVE_TRADES = [
    {'i': 1, 'dir': 'LONG', 'entry': 71100.0, 'exit': 70610.3, 'pnl3x': -7.8875, 'reason': 'EXCHANGE_SL', 'bars': 16, 't': '2026-04-12T22:15'},
    {'i': 2, 'dir': 'SHORT', 'entry': 70560.0, 'exit': 71080.3, 'pnl3x': -2.5122, 'reason': 'EXCHANGE_SL', 'bars': 16, 't': '2026-04-13T13:00'},
    {'i': 3, 'dir': 'LONG', 'entry': 71668.6, 'exit': 74026.6, 'pnl3x': 9.5704, 'reason': 'EXCHANGE_TRAIL', 'bars': 44, 't': '2026-04-14T01:09'},
    {'i': 4, 'dir': 'SHORT', 'entry': 74226.0, 'exit': 74782.5, 'pnl3x': -2.5492, 'reason': 'EXCHANGE_SL', 'bars': 4, 't': '2026-04-14T07:15'},
    {'i': 5, 'dir': 'SHORT', 'entry': 74300.0, 'exit': 74630.0, 'pnl3x': -1.6324, 'reason': 'EXCHANGE_TRAIL', 'bars': 10, 't': '2026-04-14T13:30'},
    {'i': 6, 'dir': 'LONG', 'entry': 75112.0, 'exit': 75448.0, 'pnl3x': 1.042, 'reason': 'EXCHANGE_TRAIL', 'bars': 3, 't': '2026-04-14T14:45'},
    {'i': 7, 'dir': 'LONG', 'entry': 74521.9, 'exit': 74521.9, 'pnl3x': -0.3, 'reason': 'TRAIL_TP', 'bars': 12, 't': '2026-04-15T03:45'},
    {'i': 8, 'dir': 'SHORT', 'entry': 73948.3, 'exit': 73948.3, 'pnl3x': -0.3, 'reason': 'TRAIL_TP', 'bars': 12, 't': '2026-04-15T08:30'},
    {'i': 9, 'dir': 'LONG', 'entry': 74305.7, 'exit': 73854.8, 'pnl3x': -2.1205, 'reason': 'EXCHANGE_SL', 'bars': 3, 't': '2026-04-15T13:43'},
    {'i': 10, 'dir': 'LONG', 'entry': 74361.5, 'exit': 74622.4, 'pnl3x': 0.7526, 'reason': 'EXCHANGE_TRAIL', 'bars': 4, 't': '2026-04-15T20:11'},
    {'i': 11, 'dir': 'SHORT', 'entry': 74649.9, 'exit': 74685.6, 'pnl3x': -0.4435, 'reason': 'EXCHANGE_TRAIL', 'bars': 16, 't': '2026-04-16T12:02'},
    {'i': 12, 'dir': 'LONG', 'entry': 74958.2, 'exit': 74442.4, 'pnl3x': -2.3644, 'reason': 'EXCHANGE_TRAIL', 'bars': 0, 't': '2026-04-16T13:42'},
    {'i': 13, 'dir': 'SHORT', 'entry': 73653.8, 'exit': 74058.7, 'pnl3x': -1.9492, 'reason': 'EXCHANGE_TRAIL', 'bars': 3, 't': '2026-04-16T14:49'},
    {'i': 14, 'dir': 'LONG', 'entry': 75055.5, 'exit': 75055.5, 'pnl3x': -0.3, 'reason': 'TRAIL_TP', 'bars': 16, 't': '2026-04-16T23:00'},
    {'i': 15, 'dir': 'SHORT', 'entry': 74680.0, 'exit': 74680.0, 'pnl3x': -0.3, 'reason': 'TRAIL_TP', 'bars': 22, 't': '2026-04-17T07:00'},
    {'i': 16, 'dir': 'LONG', 'entry': 75315.2, 'exit': 75630.1, 'pnl3x': 0.9543, 'reason': 'EXCHANGE_TRAIL', 'bars': 3, 't': '2026-04-17T09:33'},
    {'i': 17, 'dir': 'LONG', 'entry': 76905.5, 'exit': 76159.5, 'pnl3x': -3.2101, 'reason': 'EXCHANGE_TRAIL', 'bars': 1, 't': '2026-04-17T13:32'},
    {'i': 18, 'dir': 'LONG', 'entry': 77448.6, 'exit': 77268.3, 'pnl3x': -0.9984, 'reason': 'EXCHANGE_TRAIL', 'bars': 13, 't': '2026-04-17T17:21'},
    {'i': 19, 'dir': 'SHORT', 'entry': 77004.1, 'exit': 76213.1, 'pnl3x': 2.7817, 'reason': 'EXCHANGE_SL', 'bars': 49, 't': '2026-04-18T12:16'},
    {'i': 20, 'dir': 'SHORT', 'entry': 75486.3, 'exit': 75672.6, 'pnl3x': -1.0404, 'reason': 'EXCHANGE_TRAIL', 'bars': 4, 't': '2026-04-19T02:42'},
    {'i': 21, 'dir': 'SHORT', 'entry': 75173.8, 'exit': 75219.4, 'pnl3x': -0.482, 'reason': 'EXCHANGE_TRAIL', 'bars': 2, 't': '2026-04-19T07:53'},
    {'i': 22, 'dir': 'LONG', 'entry': 75386.4, 'exit': 75800.0, 'pnl3x': 1.3459, 'reason': 'EXCHANGE_TRAIL', 'bars': 11, 't': '2026-04-19T14:02'},
    {'i': 23, 'dir': 'SHORT', 'entry': 75227.0, 'exit': 74278.0, 'pnl3x': 3.4845, 'reason': 'EXCHANGE_TRAIL', 'bars': 31, 't': '2026-04-20T00:51'},
    {'i': 24, 'dir': 'SHORT', 'entry': 74093.7, 'exit': 74521.4, 'pnl3x': -2.0317, 'reason': 'EXCHANGE_TRAIL', 'bars': 4, 't': '2026-04-20T06:36'},
    {'i': 25, 'dir': 'LONG', 'entry': 75227.4, 'exit': 74621.7, 'pnl3x': -2.7155, 'reason': 'EXCHANGE_TRAIL', 'bars': 2, 't': '2026-04-20T08:02'},
    {'i': 26, 'dir': 'LONG', 'entry': 75929.6, 'exit': 75850.4, 'pnl3x': -0.6129, 'reason': 'TRAIL_TP', 'bars': 15, 't': '2026-04-20T21:45'},
    {'i': 27, 'dir': 'SHORT', 'entry': 75680.0, 'exit': 76178.0, 'pnl3x': -2.2741, 'reason': 'EXCHANGE_TRAIL', 'bars': 7, 't': '2026-04-21T01:06'},
]


def fetch_candles():
    """Fetch BTC/USDT 15m from BingX. Warmup from 04-08, eval 04-12 to 04-21."""
    import ccxt
    exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})

    start_ts = int(datetime(2026, 4, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    all_c = []
    since = start_ts
    while since < end_ts:
        cs = exchange.fetch_ohlcv('BTC-USDT', '15m', since=since, limit=1000)
        if not cs:
            break
        all_c.extend(cs)
        last = cs[-1][0]
        if last <= since:
            break
        since = last + 1

    seen = set()
    unique = []
    for c in all_c:
        if c[0] not in seen:
            seen.add(c[0]); unique.append(c)
    unique = [c for c in unique if start_ts <= c[0] < end_ts]
    unique.sort(key=lambda x: x[0])
    return unique


def run_bt(candles):
    timestamps = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).replace(tzinfo=None) for c in candles]
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    n = len(closes)

    signal = C1BreakoutSignal(CONFIG)
    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, CONFIG['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, CONFIG['fractal_lookback'])

    eval_start = datetime(2026, 4, 12, 0, 0)
    eval_end = datetime(2026, 4, 22, 0, 0)
    s_idx = next((i for i, t in enumerate(timestamps) if t >= eval_start), None)
    e_idx = next((i for i in range(n-1, -1, -1) if timestamps[i] < eval_end), n-1)

    trades = []
    in_pos = False
    cd = 0
    pdir = pprice = ptime = pbar = psl = pbest = None
    pheld = 0

    for i in range(s_idx, e_idx + 1):
        if in_pos:
            pheld += 1
            if pdir == 'LONG':
                pbest = max(pbest, highs[i])
            else:
                pbest = min(pbest, lows[i])

            er = signal.check_exit(
                direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl,
                atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld,
            )
            if er:
                xp = er['exit_price']; rs = er['reason']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= FEE_RT_PCT
                trades.append({
                    'i': len(trades)+1, 'dir': pdir, 'entry': round(pprice,1),
                    'exit': round(xp,1), 'pnl1x': round(pnl,4),
                    'pnl3x': round(pnl*LEVERAGE,4), 'reason': rs, 'bars': pheld,
                    't_entry': str(ptime), 't_exit': str(timestamps[i]),
                })
                in_pos = False
                cd = i + CONFIG['min_bars_between']
                pdir = None

        if not in_pos and i >= cd and i < e_idx:
            if math.isnan(atr[i]) or math.isnan(ch_h[i]):
                continue
            es = signal.check_entry(
                bar_open=opens[i], bar_high=highs[i], bar_low=lows[i], bar_close=closes[i],
                channel_high=ch_h[i], channel_low=ch_l[i], atr_val=atr[i],
                last_swing_low=sw_l[i], last_swing_high=sw_h[i],
            )
            if es:
                ni = i + 1
                if ni > e_idx:
                    continue
                pdir = es['direction']
                pprice = opens[ni]; ptime = timestamps[ni]; pbar = ni
                psl = es['sl_price']; pheld = 0; in_pos = True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]

                er = signal.check_exit(
                    direction=pdir, entry_price=pprice, best_price=pbest,
                    current_high=highs[ni], current_low=lows[ni], current_close=closes[ni],
                    sl_price=psl,
                    atr_val=atr[ni] if not math.isnan(atr[ni]) else atr[i],
                    bars_held=0,
                )
                if er:
                    xp = er['exit_price']; rs = er['reason']
                    pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                    pnl -= FEE_RT_PCT
                    trades.append({
                        'i': len(trades)+1, 'dir': pdir, 'entry': round(pprice,1),
                        'exit': round(xp,1), 'pnl1x': round(pnl,4),
                        'pnl3x': round(pnl*LEVERAGE,4), 'reason': rs, 'bars': 0,
                        't_entry': str(ptime), 't_exit': str(timestamps[ni]),
                    })
                    in_pos = False
                    cd = ni + CONFIG['min_bars_between']
                    pdir = None

    return trades


def equity_curve(trades, start=START_BALANCE, pnl_key='pnl3x', compound=True):
    """Return list of balances after each trade."""
    eq = [start]
    cur = start
    for t in trades:
        if compound:
            cur = cur * (1 + t[pnl_key]/100)
        else:
            cur = cur + start * t[pnl_key]/100
        eq.append(cur)
    return eq


def max_dd(eq):
    peak = eq[0]; mdd = 0.0; peak_i = 0; trough_i = 0
    for i, v in enumerate(eq):
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < mdd:
            mdd = dd; trough_i = i
    return mdd


def main():
    print("Fetching BingX 15m data (04-08 ~ 04-22)...")
    candles = fetch_candles()
    print(f"Got {len(candles)} candles\n")

    print("Running BT...")
    bt = run_bt(candles)
    print(f"BT trades: {len(bt)}\n")

    # Equity curves (compound, 3x)
    bt_eq = equity_curve(bt)
    live_eq = equity_curve(LIVE_TRADES)

    bt_mdd = max_dd(bt_eq)
    live_mdd = max_dd(live_eq)

    bt_pnl_sum_3x = sum(t['pnl3x'] for t in bt)
    live_pnl_sum_3x = sum(t['pnl3x'] for t in LIVE_TRADES)
    bt_wins = sum(1 for t in bt if t['pnl3x'] > 0)
    live_wins = sum(1 for t in LIVE_TRADES if t['pnl3x'] > 0)

    print("=" * 70)
    print("  BT vs LIVE Drawdown Comparison (Apr 12 - Apr 21, 3x leverage)")
    print("=" * 70)
    print(f"{'Metric':<25} {'BT':>18} {'LIVE':>18}")
    print("-" * 70)
    print(f"{'Trades':<25} {len(bt):>18} {len(LIVE_TRADES):>18}")
    print(f"{'WR (%)':<25} {100*bt_wins/max(1,len(bt)):>18.1f} {100*live_wins/len(LIVE_TRADES):>18.1f}")
    print(f"{'Sum PnL 3x (additive)':<25} {bt_pnl_sum_3x:>17.2f}% {live_pnl_sum_3x:>17.2f}%")
    print(f"{'End balance (compound)':<25} ${bt_eq[-1]:>16.2f} ${live_eq[-1]:>16.2f}")
    print(f"{'Net % (compound)':<25} {100*(bt_eq[-1]/START_BALANCE - 1):>17.2f}% {100*(live_eq[-1]/START_BALANCE - 1):>17.2f}%")
    print(f"{'MDD % (compound)':<25} {bt_mdd:>17.2f}% {live_mdd:>17.2f}%")
    print("=" * 70)

    print("\nBT trade list:")
    print(f"{'#':>3} {'dir':<5} {'entry':>10} {'exit':>10} {'pnl3x':>8} {'reason':<18} {'t_exit':<20}")
    for t in bt:
        print(f"{t['i']:>3} {t['dir']:<5} {t['entry']:>10.1f} {t['exit']:>10.1f} {t['pnl3x']:>+7.2f}% {t['reason']:<18} {t['t_exit'][:19]}")

    # Save results
    out = {
        'bt': {'trades': bt, 'end_balance': bt_eq[-1], 'mdd_pct': bt_mdd,
               'sum_pnl_3x': bt_pnl_sum_3x, 'wr_pct': 100*bt_wins/max(1,len(bt))},
        'live': {'trades': LIVE_TRADES, 'end_balance': live_eq[-1], 'mdd_pct': live_mdd,
                 'sum_pnl_3x': live_pnl_sum_3x, 'wr_pct': 100*live_wins/len(LIVE_TRADES)},
        'start_balance': START_BALANCE, 'leverage': LEVERAGE,
        'period': '2026-04-12 to 2026-04-21',
    }
    out_path = ROOT / 'results' / f'dd_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
