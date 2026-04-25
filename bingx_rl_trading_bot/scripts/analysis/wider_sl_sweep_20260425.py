"""
Wider SL Sweep BT — max_sl_atr 3.3 ~ 6.0 (2026-04-25)
========================================================
SL placement을 더 멀리 두면 wick exit 회피 효과 + 다른 trade-offs 측정.

Stop hunt 분석 (97.1% recovery, avg 0.615% recovery vs sl_pct 0.748%) 후속.

각 max_sl_atr 값에서:
- 04-04 ~ 04-25 (22d)
- 333d full (5m csv 가능 시)
- BT trades, PnL, MDD, SL ratio
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

BASE_CFG = {
    'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
    'trail_K': 2.5, 'max_sl_atr': 3.3,  # swept
    'emergency_sl_pct': 3.0, 'max_hold_bars': 192,
    'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
    'min_bars_between': 2, 'trail_activation_pct': 0.05,
    'fractal_lookback': 10,
    'progressive_trail': {'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5},
}

MAX_SL_ATR_VALUES = [3.3, 4.0, 4.5, 5.0, 6.0, 8.0]
FEE_RT_PCT = 0.10
LEVERAGE = 3
START = 100.0


def fetch_candles(start_dt, end_dt):
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    en = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
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


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close'], r.get('volume',0)] for _, r in df15.iterrows()]


def run_bt(candles, max_sl, eval_start, eval_end):
    cfg = dict(BASE_CFG); cfg['max_sl_atr'] = max_sl
    signal = C1BreakoutSignal(cfg)
    ts = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).replace(tzinfo=None) for c in candles]
    opens, highs = [c[1] for c in candles], [c[2] for c in candles]
    lows, closes = [c[3] for c in candles], [c[4] for c in candles]
    n = len(closes)
    atr = compute_atr(highs, lows, closes, cfg['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, cfg['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, cfg['fractal_lookback'])

    s_idx = next((i for i, t in enumerate(ts) if t >= eval_start), None)
    if s_idx is None: return []
    e_idx = next(i for i in range(n-1, -1, -1) if ts[i] < eval_end)
    trades, in_pos, cd = [], False, 0
    pdir = pprice = psl = pbest = None; pheld = 0

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
                trades.append({'dir':pdir, 'pnl1x':round(pnl,4), 'pnl3x':round(pnl*LEVERAGE,4), 'reason':rs})
                in_pos, cd, pdir = False, i + cfg['min_bars_between'], None
        if not in_pos and i >= cd and i < e_idx:
            if math.isnan(atr[i]) or math.isnan(ch_h[i]): continue
            es = signal.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                bar_close=closes[i], channel_high=ch_h[i], channel_low=ch_l[i],
                atr_val=atr[i], last_swing_low=sw_l[i], last_swing_high=sw_h[i])
            if es:
                ni = i + 1
                if ni > e_idx: continue
                pdir = es['direction']; pprice = opens[ni]; psl = es['sl_price']
                pheld = 0; in_pos = True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]
    return trades


def stats(trades, start=START):
    if not trades: return {'trades':0}
    n = len(trades); wins = sum(1 for t in trades if t['pnl3x']>0)
    sum1x = sum(t['pnl1x'] for t in trades); sum3x = sum(t['pnl3x'] for t in trades)
    bal = start; peak = start; mdd = 0.0
    for t in trades:
        bal *= (1 + t['pnl3x']/100)
        if bal>peak: peak=bal
        dd = (bal-peak)/peak*100
        if dd<mdd: mdd=dd
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'],0)+1
    sl_pct = 100 * reasons.get('SL',0) / n
    return {'trades':n, 'wr':round(100*wins/n,1), 'sum1x':round(sum1x,2), 'sum3x':round(sum3x,2),
            'end_bal':round(bal,2), 'mdd':round(mdd,2), 'sl_pct':round(sl_pct,1), 'reasons':reasons}


def main():
    print("=" * 95)
    print("Wider SL Sweep — max_sl_atr 스윕 (2026-04-25)")
    print("=" * 95)

    # Part 1: 04-04 ~ 04-25 (recent)
    print("\nPart 1: Recent 22d (04-04 ~ 04-25)")
    candles_recent = fetch_candles(datetime(2026,3,28), datetime(2026,4,25,6))
    print(f"Got {len(candles_recent)} candles\n")
    print(f"{'max_sl':>7} {'trades':>7} {'WR%':>6} {'sum1x':>9} {'sum3x':>9} {'endBal':>9} {'MDD%':>7} {'SL%':>6}")
    print("-"*85)
    p1 = {}
    for m in MAX_SL_ATR_VALUES:
        ts = run_bt(candles_recent, m, datetime(2026,4,4), datetime(2026,4,25,6))
        s = stats(ts)
        print(f"{m:>7.1f} {s.get('trades',0):>7} {s.get('wr',0):>5.1f} {s.get('sum1x',0):>+8.2f}% {s.get('sum3x',0):>+8.2f}% {s.get('end_bal',0):>9.2f} {s.get('mdd',0):>+6.2f}% {s.get('sl_pct',0):>5.1f}%")
        p1[str(m)] = s

    # Part 2: 333d full (csv)
    csv_path = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    p2 = {}
    if csv_path.exists():
        print(f"\nPart 2: 333d full (csv resample)")
        candles_full = load_csv_15m(str(csv_path))
        print(f"Got {len(candles_full)} 15m candles")
        print(f"{'max_sl':>7} {'trades':>7} {'WR%':>6} {'sum1x':>10} {'sum3x':>10} {'endBal':>10} {'MDD%':>7} {'SL%':>6}")
        print("-"*90)
        t_start = datetime.fromtimestamp(candles_full[0][0]/1000)
        t_end = datetime.fromtimestamp(candles_full[-1][0]/1000)
        for m in MAX_SL_ATR_VALUES:
            ts = run_bt(candles_full, m, t_start, t_end)
            s = stats(ts)
            print(f"{m:>7.1f} {s.get('trades',0):>7} {s.get('wr',0):>5.1f} {s.get('sum1x',0):>+9.2f}% {s.get('sum3x',0):>+9.2f}% {s.get('end_bal',0):>10.2f} {s.get('mdd',0):>+6.2f}% {s.get('sl_pct',0):>5.1f}%")
            p2[str(m)] = s
    else:
        print(f"\nPart 2 skipped — {csv_path} not found")

    out = {'date':datetime.now().isoformat(), 'part1_recent':p1, 'part2_333d':p2}
    path = ROOT / 'results' / f'wider_sl_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path,'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
