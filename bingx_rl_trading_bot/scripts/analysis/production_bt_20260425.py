"""
Production Setting BT — 04-04 ~ 04-25 (2026-04-25)
====================================================
현 운영 config 그대로 BT 실행, LIVE 실측과 비교.

CONFIG: progressive_trail enabled (threshold 0.9, K_post 0.5).
F v1/F v2는 LIVE execution layer만 변경 → BT 동일.

Output: BT trades vs LIVE trade_history (state.json) 비교.
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

# Production-identical CONFIG (c1_breakout_config.yaml)
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
    'progressive_trail': {'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5},
}

FEE_RT_PCT = 0.10
LEVERAGE = 3
START = 2100.0  # 가정 시작 자본 (compound 기준)

EVAL_START = datetime(2026, 4, 4, 0, 0)
EVAL_END = datetime(2026, 4, 25, 6, 0)


def fetch_candles():
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    # Warmup 7d before eval start
    st_dt = datetime(2026, 3, 28, 0, 0).replace(tzinfo=timezone.utc)
    en_dt = EVAL_END.replace(tzinfo=timezone.utc)
    st = int(st_dt.timestamp() * 1000)
    en = int(en_dt.timestamp() * 1000)
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

    s_idx = next(i for i, t in enumerate(ts) if t >= EVAL_START)
    e_idx = next(i for i in range(n-1, -1, -1) if ts[i] < EVAL_END)

    trades = []; in_pos, cd = False, 0
    pdir = pprice = psl = pbest = ptime = None; pheld = 0

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
                trades.append({
                    'dir': pdir, 'entry': round(pprice,1), 'exit': round(xp,1),
                    'pnl1x': round(pnl, 4), 'pnl3x': round(pnl*LEVERAGE, 4),
                    'reason': rs, 'bars': pheld,
                    't_entry': str(ptime)[:19], 't_exit': str(ts[i])[:19],
                })
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


def load_live_trades():
    """Load LIVE trades from state.json filtered to EVAL window."""
    with open(ROOT / 'results' / 'c1_breakout_state.json') as f:
        d = json.load(f)
    th = d.get('trade_history', [])
    out = []
    for t in th:
        et = t.get('exit_time', '')[:19]
        try:
            edt = datetime.strptime(et, '%Y-%m-%dT%H:%M:%S')
        except:
            continue
        if EVAL_START <= edt <= EVAL_END:
            out.append({
                'dir': t.get('direction'),
                'entry': t.get('entry_price'),
                'exit': t.get('exit_price'),
                'pnl3x': t.get('pnl_pct'),  # already 3x leveraged in state
                'reason': t.get('reason'),
                'bars': t.get('bars_held'),
                't_exit': et,
                'slip': t.get('exit_slippage_pct'),
            })
    return out


def stats(trades, leveraged=False, start=START):
    if not trades:
        return {'trades': 0}
    n = len(trades)
    if leveraged:
        sums = [t['pnl3x'] for t in trades]
    else:
        sums = [t.get('pnl3x', 0) for t in trades]
    wins = sum(1 for s in sums if s > 0)
    sum_pnl = sum(sums)
    bal = start; peak = start; mdd = 0.0
    for s in sums:
        bal *= (1 + s/100)
        if bal > peak: peak = bal
        dd = (bal - peak)/peak * 100
        if dd < mdd: mdd = dd
    reasons = {}
    for t in trades:
        r = t.get('reason', '?')
        reasons[r] = reasons.get(r, 0) + 1
    return {
        'trades': n,
        'wr_pct': round(100*wins/n, 1),
        'sum_pnl_3x': round(sum_pnl, 2),
        'end_bal': round(bal, 2),
        'net_pct': round(100*(bal/start-1), 2),
        'mdd_pct': round(mdd, 2),
        'reasons': reasons,
    }


def main():
    print("=" * 90)
    print("Production Setting BT (progressive_trail enabled) — 2026-04-04 ~ 2026-04-25")
    print("=" * 90)
    print("Fetching BingX 15m...")
    c = fetch_candles()
    print(f"Got {len(c)} candles\n")

    bt = run_bt(c)
    live = load_live_trades()

    s_bt = stats(bt)
    s_live = stats(live)

    print(f"{'Metric':<25} {'BT':>15} {'LIVE':>15} {'Gap':>10}")
    print("-" * 70)
    print(f"{'Trades':<25} {s_bt['trades']:>15} {s_live['trades']:>15}")
    print(f"{'WR (%)':<25} {s_bt.get('wr_pct',0):>15.1f} {s_live.get('wr_pct',0):>15.1f}")
    print(f"{'Sum PnL 3x (add)':<25} {s_bt.get('sum_pnl_3x',0):>+14.2f}% {s_live.get('sum_pnl_3x',0):>+14.2f}% {s_live.get('sum_pnl_3x',0)-s_bt.get('sum_pnl_3x',0):>+9.2f}pp")
    print(f"{'End balance (cmp)':<25} ${s_bt.get('end_bal',0):>13.2f} ${s_live.get('end_bal',0):>13.2f}")
    print(f"{'Net % (cmp)':<25} {s_bt.get('net_pct',0):>+14.2f}% {s_live.get('net_pct',0):>+14.2f}% {s_live.get('net_pct',0)-s_bt.get('net_pct',0):>+9.2f}pp")
    print(f"{'MDD % (cmp)':<25} {s_bt.get('mdd_pct',0):>+14.2f}% {s_live.get('mdd_pct',0):>+14.2f}%")
    print()
    print(f"BT reasons:   {s_bt.get('reasons')}")
    print(f"LIVE reasons: {s_live.get('reasons')}")
    print()

    days = (EVAL_END - EVAL_START).days + (EVAL_END - EVAL_START).seconds/86400
    print(f"Period: {days:.1f} days")
    print(f"BT daily 3x:   {s_bt.get('sum_pnl_3x',0)/days:+.2f}%/day")
    print(f"LIVE daily 3x: {s_live.get('sum_pnl_3x',0)/days:+.2f}%/day")

    out = {
        'date': datetime.now().isoformat(),
        'eval_start': EVAL_START.isoformat(), 'eval_end': EVAL_END.isoformat(),
        'days': days, 'config': CONFIG,
        'bt': {'trades': bt, 'stats': s_bt},
        'live': {'trades': live, 'stats': s_live},
    }
    path = ROOT / 'results' / f'production_bt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
