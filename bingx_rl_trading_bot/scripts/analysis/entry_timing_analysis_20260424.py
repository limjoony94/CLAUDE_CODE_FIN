"""
Entry timing divergence 분석 — BT vs LIVE signal firing bars
==============================================================
Class B* 4 trades의 원인으로 추정되는 "entry slippage + signal timing divergence"를 정량화.

각 LIVE Canary trade에 대해:
1. LIVE entry time (from log ENTRY log, KST → UTC)
2. Signal price (log) vs fill price (state.json)
3. BT가 동일 신호를 어느 bar open에 잡았을지 시뮬
4. Time diff, price diff 측정
"""
import re, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import math
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / 'logs'

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

CONFIG = {
    'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
    'trail_K': 2.5, 'max_sl_atr': 3.3, 'emergency_sl_pct': 3.0,
    'max_hold_bars': 192, 'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
    'min_bars_between': 2, 'trail_activation_pct': 0.05,
    'fractal_lookback': 10,
}

# Canary LIVE entries (from state + logs)
LIVE_ENTRIES = [
    # trade#, dir, signal_price (log), fill_price (state), log_ts_utc
    (34, 'LONG',  78267.80, 78285.9, '2026-04-22 11:30:07'),
    (35, 'LONG',  79236.10, 79250.0, '2026-04-22 15:45:05'),
    (36, 'SHORT', 78546.20, 78535.7, '2026-04-22 20:30:06'),
    (37, 'LONG',  78244.50, 78260.3, '2026-04-23 06:30:06'),
    (38, 'SHORT', 77544.80, 77540.0, '2026-04-23 09:15:05'),
    (39, 'LONG',  78136.80, 78140.0, '2026-04-23 15:00:05'),
    (40, 'SHORT', 77143.70, 77059.6, '2026-04-23 17:45:05'),
    (41, 'LONG',  78272.00, 78285.0, '2026-04-23 22:45:06'),
]


def fetch_candles():
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(datetime(2026, 4, 20, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    en = int(datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    all_c, since = [], st
    while since < en:
        cs = ex.fetch_ohlcv('BTC-USDT', '15m', since=since, limit=1000)
        if not cs: break
        all_c.extend(cs)
        if cs[-1][0] <= since: break
        since = cs[-1][0] + 1
    seen, uniq = set(), []
    for c in all_c:
        if c[0] not in seen:
            seen.add(c[0]); uniq.append(c)
    uniq = [c for c in uniq if st <= c[0] < en]
    uniq.sort(key=lambda x: x[0])
    return uniq


def analyze_entry(candles, live_trade):
    """Find BT signal bar vs LIVE entry bar."""
    tnum, ldir, lsignal, lfill, lts = live_trade
    live_dt = datetime.strptime(lts, '%Y-%m-%d %H:%M:%S')

    ts = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).replace(tzinfo=None) for c in candles]
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    n = len(closes)

    signal_obj = C1BreakoutSignal(CONFIG)
    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, CONFIG['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, CONFIG['fractal_lookback'])

    # Find LIVE bar (bar close nearest to live_dt)
    live_bar = None
    for i in range(n):
        if ts[i] <= live_dt:
            live_bar = i
        else:
            break

    # BT: scan ±5 bars around live_bar for signal firing
    bt_signals = []
    if live_bar is not None:
        start = max(0, live_bar - 5)
        end = min(n, live_bar + 5)
        for i in range(start, end):
            if math.isnan(atr[i]) or math.isnan(ch_h[i]):
                continue
            sig = signal_obj.check_entry(
                bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                bar_close=closes[i], channel_high=ch_h[i], channel_low=ch_l[i],
                atr_val=atr[i], last_swing_low=sw_l[i], last_swing_high=sw_h[i],
            )
            if sig and sig['direction'] == ldir:
                # BT enters at opens[i+1] if i+1 exists
                bt_entry_price = opens[i + 1] if i + 1 < n else closes[i]
                bt_signals.append({
                    'bar': i, 'bar_ts': ts[i].isoformat(),
                    'bar_close': closes[i],
                    'entry_bar_ts': ts[i+1].isoformat() if i+1 < n else 'N/A',
                    'entry_price': bt_entry_price,
                })

    return {
        'trade': tnum, 'dir': ldir,
        'live_signal_ts': live_dt.isoformat(),
        'live_signal_price': lsignal,
        'live_fill_price': lfill,
        'live_entry_slippage_pct': (lfill / lsignal - 1) * 100 if ldir == 'LONG' else (1 - lfill / lsignal) * 100,
        'live_bar': live_bar, 'live_bar_ts': ts[live_bar].isoformat() if live_bar else None,
        'live_bar_close': closes[live_bar] if live_bar else None,
        'bt_signals_nearby': bt_signals,
    }


def main():
    print("Fetching BingX 15m (04-20 ~ 04-24 12:00 UTC)...")
    c = fetch_candles()
    print(f"Got {len(c)} candles\n")

    results = []
    for le in LIVE_ENTRIES:
        r = analyze_entry(c, le)
        results.append(r)

    print("=" * 120)
    print("Entry Timing Divergence: LIVE vs BT signal bars")
    print("=" * 120)
    print(f"{'#':>3} {'dir':<5} {'LIVE ts':<17} {'LIVE signal':>11} {'LIVE fill':>10} "
          f"{'entry_slip':>11} {'BT bars found':>14}  {'BT nearest entry'}")
    print("-" * 120)
    for r in results:
        if r['bt_signals_nearby']:
            best = r['bt_signals_nearby'][0]
            bt_info = f"{best['entry_bar_ts'][:16]} @ {best['entry_price']:.1f}"
        else:
            bt_info = 'NO SIGNAL'
        print(f"{r['trade']:>3} {r['dir']:<5} {r['live_signal_ts'][:16]:<17} "
              f"{r['live_signal_price']:>11.1f} {r['live_fill_price']:>10.1f} "
              f"{r['live_entry_slippage_pct']:>+10.3f}% {len(r['bt_signals_nearby']):>14}  {bt_info}")

    # Detailed divergence analysis
    print()
    print("=" * 120)
    print("Divergence analysis:")
    print("=" * 120)
    for r in results:
        print(f"\nTrade #{r['trade']} {r['dir']}:")
        print(f"  LIVE signal ts: {r['live_signal_ts']}, price={r['live_signal_price']}, fill={r['live_fill_price']} (slip {r['live_entry_slippage_pct']:+.3f}%)")
        for bts in r['bt_signals_nearby']:
            dt_diff = (datetime.fromisoformat(bts['entry_bar_ts']) - datetime.fromisoformat(r['live_signal_ts'])).total_seconds() / 60
            price_diff = bts['entry_price'] - r['live_fill_price']
            price_diff_pct = price_diff / r['live_fill_price'] * 100
            print(f"  BT signal bar={bts['bar_ts'][:16]}, entry_bar={bts['entry_bar_ts'][:16]} @ ${bts['entry_price']:.1f}")
            print(f"    dt_diff: {dt_diff:+.0f}min, price_diff: {price_diff:+.1f} pts ({price_diff_pct:+.3f}%)")

    out = {'date': datetime.now().isoformat(), 'results': results}
    path = ROOT / 'results' / f'entry_timing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
