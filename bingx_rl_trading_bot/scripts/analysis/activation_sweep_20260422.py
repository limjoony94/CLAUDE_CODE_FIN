"""
Activation Threshold Sweep BT (2026-04-22)
============================================
trail_activation_pct 스윕으로 BT-LIVE gap/MDD/WR 변화 측정.
데이터 구간: 2026-04-12 ~ 2026-04-21 (dd_comparison 구간과 동일).
기존 dd_comparison_20260421.py와 동일한 CCXT fetch 사용.

Goal: activation_pct를 높이면 pre-activation TRAILING slippage 노출 시간이
      줄어들어 BT-LIVE gap 축소 가능한지 BT-only simulation으로 확인.
      (LIVE sample은 없지만 BT trail exit count/PnL 변화가 1차 신호)

Output: results/activation_sweep_{date}.json
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

BASE_CONFIG = {
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
    'trail_activation_pct': 0.05,  # swept
    'fractal_lookback': 10,
    'progressive_trail': {'enabled': False},  # hold fixed for clean comparison
}

ACTIVATION_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5]
FEE_RT_PCT = 0.10
LEVERAGE = 3
START_BALANCE = 2100.0


def fetch_candles():
    import ccxt
    exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})
    start_ts = int(datetime(2026, 4, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    all_c, since = [], start_ts
    while since < end_ts:
        cs = exchange.fetch_ohlcv('BTC-USDT', '15m', since=since, limit=1000)
        if not cs: break
        all_c.extend(cs)
        if cs[-1][0] <= since: break
        since = cs[-1][0] + 1
    seen, uniq = set(), []
    for c in all_c:
        if c[0] not in seen:
            seen.add(c[0]); uniq.append(c)
    uniq = [c for c in uniq if start_ts <= c[0] < end_ts]
    uniq.sort(key=lambda x: x[0])
    return uniq


def run_bt(candles, activation_pct):
    cfg = dict(BASE_CONFIG)
    cfg['trail_activation_pct'] = activation_pct
    signal = C1BreakoutSignal(cfg)

    ts = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).replace(tzinfo=None) for c in candles]
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    n = len(closes)

    atr = compute_atr(highs, lows, closes, cfg['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, cfg['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, cfg['fractal_lookback'])

    s_idx = next(i for i, t in enumerate(ts) if t >= datetime(2026, 4, 12))
    e_idx = next(i for i in range(n-1, -1, -1) if ts[i] < datetime(2026, 4, 22))

    trades, in_pos, cd = [], False, 0
    pdir = pprice = ptime = psl = pbest = None
    pheld = 0

    for i in range(s_idx, e_idx + 1):
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])
            er = signal.check_exit(
                direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl,
                atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld,
            )
            if er:
                xp, rs = er['exit_price'], er['reason']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= FEE_RT_PCT
                trades.append({
                    'dir': pdir, 'entry': round(pprice,1), 'exit': round(xp,1),
                    'pnl1x': round(pnl,4), 'pnl3x': round(pnl*LEVERAGE,4),
                    'reason': rs, 'bars': pheld, 't_exit': str(ts[i])[:19],
                })
                in_pos, cd, pdir = False, i + cfg['min_bars_between'], None

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
                if ni > e_idx: continue
                pdir = es['direction']
                pprice, ptime, psl = opens[ni], ts[ni], es['sl_price']
                pheld, in_pos = 0, True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]

                er = signal.check_exit(
                    direction=pdir, entry_price=pprice, best_price=pbest,
                    current_high=highs[ni], current_low=lows[ni], current_close=closes[ni],
                    sl_price=psl,
                    atr_val=atr[ni] if not math.isnan(atr[ni]) else atr[i],
                    bars_held=0,
                )
                if er:
                    xp, rs = er['exit_price'], er['reason']
                    pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                    pnl -= FEE_RT_PCT
                    trades.append({
                        'dir': pdir, 'entry': round(pprice,1), 'exit': round(xp,1),
                        'pnl1x': round(pnl,4), 'pnl3x': round(pnl*LEVERAGE,4),
                        'reason': rs, 'bars': 0, 't_exit': str(ts[ni])[:19],
                    })
                    in_pos, cd, pdir = False, ni + cfg['min_bars_between'], None

    return trades


def stats(trades, start=START_BALANCE):
    if not trades:
        return {'trades': 0, 'wr_pct': 0, 'sum_pnl_3x': 0, 'end_bal': start,
                'net_pct': 0, 'mdd_pct': 0, 'reasons': {}}
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl3x'] > 0)
    sum3x = sum(t['pnl3x'] for t in trades)
    bal = start
    peak = start; mdd = 0.0
    for t in trades:
        bal *= (1 + t['pnl3x']/100)
        if bal > peak: peak = bal
        dd = (bal - peak) / peak * 100
        if dd < mdd: mdd = dd
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    return {
        'trades': n, 'wr_pct': round(100*wins/n, 2),
        'sum_pnl_3x': round(sum3x, 4),
        'end_bal': round(bal, 2),
        'net_pct': round(100*(bal/start - 1), 4),
        'mdd_pct': round(mdd, 4),
        'reasons': reasons,
    }


def main():
    print(f"Fetching BingX 15m (04-08 ~ 04-22)...")
    candles = fetch_candles()
    print(f"Got {len(candles)} candles\n")

    results = {}
    print(f"{'act_pct':>8} {'trades':>7} {'WR%':>6} {'pnl3x':>9} {'endBal':>8} {'MDD%':>7}  reasons")
    print("-" * 80)
    for a in ACTIVATION_VALUES:
        trades = run_bt(candles, a)
        s = stats(trades)
        s['activation_pct'] = a
        s['trades_list'] = trades
        results[str(a)] = s
        reasons_str = ' '.join(f"{k}:{v}" for k, v in sorted(s['reasons'].items()))
        print(f"{a:>8.3f} {s['trades']:>7} {s['wr_pct']:>6.1f} {s['sum_pnl_3x']:>+8.2f}% ${s['end_bal']:>7.2f} {s['mdd_pct']:>+6.2f}%  {reasons_str}")

    out = ROOT / 'results' / f'activation_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'baseline': BASE_CONFIG, 'results': results}, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
