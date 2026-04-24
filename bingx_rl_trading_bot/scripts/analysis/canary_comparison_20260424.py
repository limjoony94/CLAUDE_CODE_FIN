"""
Canary F-option vs BT Comparison — 2026-04-22 14:16 ~ 2026-04-24 01:32
========================================================================
F option Canary 배포 이후 완료된 8 trades를 classic BT와 비교.
"""

import sys, os, json, math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
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
    'progressive_trail': {'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5},
}

FEE_RT_PCT = 0.10
LEVERAGE = 3
START = 1880.0  # approximate balance at 04-22 when Canary trade #34 started

CANARY_TRADES = [
    {'i': 34, 'dir': 'LONG', 'entry': 78285.9, 'exit': 78905.0, 'pnl3x': 2.07, 'reason': 'EXCHANGE_SL', 't': '2026-04-22T14:16'},
    {'i': 35, 'dir': 'LONG', 'entry': 79250.0, 'exit': 78727.4, 'pnl3x': -2.28, 'reason': 'EXCHANGE_SL', 't': '2026-04-22T16:43'},
    {'i': 36, 'dir': 'SHORT','entry': 78535.7, 'exit': 78443.8, 'pnl3x': 0.05, 'reason': 'EXCHANGE_SL', 't': '2026-04-23T00:29'},
    {'i': 37, 'dir': 'LONG', 'entry': 78260.3, 'exit': 77805.8, 'pnl3x': -2.04, 'reason': 'EXCHANGE_SL', 't': '2026-04-23T08:33'},
    {'i': 38, 'dir': 'SHORT','entry': 77540.0, 'exit': 77188.2, 'pnl3x': 1.06, 'reason': 'TRAIL_TP', 't': '2026-04-23T10:15'},
    {'i': 39, 'dir': 'LONG', 'entry': 78140.0, 'exit': 77940.4, 'pnl3x': -1.07, 'reason': 'EXCHANGE_SL', 't': '2026-04-23T17:01'},
    {'i': 40, 'dir': 'SHORT','entry': 77059.6, 'exit': 77789.9, 'pnl3x': -3.14, 'reason': 'EXCHANGE_SL', 't': '2026-04-23T18:00'},
    {'i': 41, 'dir': 'LONG', 'entry': 78285.0, 'exit': 77942.0, 'pnl3x': -1.61, 'reason': 'EXCHANGE_SL', 't': '2026-04-24T01:32'},
]


def fetch_candles():
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(datetime(2026, 4, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    en = int(datetime(2026, 4, 24, 6, 0, tzinfo=timezone.utc).timestamp() * 1000)
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


def run_bt(candles):
    signal = C1BreakoutSignal(CONFIG)
    ts = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).replace(tzinfo=None) for c in candles]
    opens = [c[1] for c in candles]; highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]; closes = [c[4] for c in candles]
    n = len(closes)
    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, CONFIG['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, CONFIG['fractal_lookback'])

    s_idx = next(i for i, t in enumerate(ts) if t >= datetime(2026, 4, 22, 14, 0))
    e_idx = next(i for i in range(n-1, -1, -1) if ts[i] < datetime(2026, 4, 24, 6, 0))
    trades = []; in_pos, cd = False, 0
    pdir = pprice = psl = pbest = None; pheld = 0; ptime = None

    for i in range(s_idx, e_idx + 1):
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])
            er = signal.check_exit(direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl, atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld)
            if er:
                xp, rs = er['exit_price'], er['reason']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= FEE_RT_PCT
                trades.append({'dir': pdir, 'entry': round(pprice,1), 'exit': round(xp,1),
                    'pnl3x': round(pnl*LEVERAGE, 4), 'reason': rs, 'bars': pheld,
                    't_exit': str(ts[i])[:19]})
                in_pos, cd, pdir = False, i + CONFIG['min_bars_between'], None

        if not in_pos and i >= cd and i < e_idx:
            if math.isnan(atr[i]) or math.isnan(ch_h[i]): continue
            es = signal.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                bar_close=closes[i], channel_high=ch_h[i], channel_low=ch_l[i],
                atr_val=atr[i], last_swing_low=sw_l[i], last_swing_high=sw_h[i])
            if es:
                ni = i + 1
                if ni > e_idx: continue
                pdir = es['direction']; pprice = opens[ni]; ptime = ts[ni]
                psl = es['sl_price']; pheld = 0; in_pos = True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]
    return trades


def stats(trades):
    if not trades: return {}
    n = len(trades); wins = sum(1 for t in trades if t['pnl3x'] > 0)
    sum3x = sum(t['pnl3x'] for t in trades)
    reasons = {}
    for t in trades: reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    bal = START; peak = START; mdd = 0
    for t in trades:
        bal *= (1 + t['pnl3x']/100)
        if bal > peak: peak = bal
        dd = (bal - peak) / peak * 100
        if dd < mdd: mdd = dd
    return {'trades': n, 'wr_pct': round(100*wins/n, 1), 'sum_pnl_3x': round(sum3x, 2),
        'end_bal': round(bal, 2), 'mdd_pct': round(mdd, 2), 'reasons': reasons}


def main():
    print("Fetching BingX 15m (04-18 ~ 04-24 06:00 UTC)...")
    c = fetch_candles()
    print(f"Got {len(c)} candles\n")
    bt = run_bt(c)

    s_bt = stats(bt); s_live = stats(CANARY_TRADES)
    print("=" * 85)
    print("Canary F-option vs BT Comparison (2026-04-22 14:00 ~ 2026-04-24 06:00 UTC)")
    print("=" * 85)
    print(f"{'Metric':<25} {'BT':>18} {'LIVE Canary':>18}")
    print("-" * 85)
    print(f"{'Trades':<25} {s_bt.get('trades', 0):>18} {s_live.get('trades', 0):>18}")
    print(f"{'WR (%)':<25} {s_bt.get('wr_pct', 0):>18.1f} {s_live.get('wr_pct', 0):>18.1f}")
    print(f"{'Sum PnL 3x (add)':<25} {s_bt.get('sum_pnl_3x', 0):>+17.2f}% {s_live.get('sum_pnl_3x', 0):>+17.2f}%")
    print(f"{'End balance (cmp)':<25} ${s_bt.get('end_bal', 0):>16.2f} ${s_live.get('end_bal', 0):>16.2f}")
    print(f"{'MDD % (cmp)':<25} {s_bt.get('mdd_pct', 0):>+17.2f}% {s_live.get('mdd_pct', 0):>+17.2f}%")
    print(f"{'Reasons':<25} {str(s_bt.get('reasons', {})):>18} {str(s_live.get('reasons', {})):>18}")
    print("=" * 85)
    gap = round(s_live.get('sum_pnl_3x', 0) - s_bt.get('sum_pnl_3x', 0), 2)
    print(f"\nGap (LIVE - BT): {gap:+}pp")

    print("\nBT trade list:")
    for i, t in enumerate(bt):
        print(f"  #{i+1:>2} {t['dir']:<5} entry={t['entry']:>9.1f} exit={t['exit']:>9.1f} pnl3x={t['pnl3x']:+7.2f}% reason={t['reason']:<15} t_exit={t['t_exit']}")

    print("\nLIVE Canary trade list:")
    for t in CANARY_TRADES:
        print(f"  #{t['i']:>2} {t['dir']:<5} entry={t['entry']:>9.1f} exit={t['exit']:>9.1f} pnl3x={t['pnl3x']:+7.2f}% reason={t['reason']:<15} t_exit={t['t']}")

    out = {'bt': bt, 'bt_stats': s_bt, 'live': CANARY_TRADES, 'live_stats': s_live,
           'gap_pnl3x': gap, 'date': datetime.now().isoformat()}
    path = ROOT / 'results' / f'canary_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
