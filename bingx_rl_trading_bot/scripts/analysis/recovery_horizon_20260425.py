"""
Recovery Horizon Extended — 4 / 8 / 16 / 24 bars
==================================================
stop_hunt_analysis_20260425의 lookahead 8 bars를 multiple horizons로 확장.

목적: 회복 패턴이 단기/지속/decay 어떤지 확인.
- 단기 회복 후 재하락 → wick인지 trend reversal인지 구분
- Long horizon에서도 회복 유지되면 stop hunt 더 강력 입증
"""
import sys, json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

HORIZONS = [4, 8, 16, 24]


def fetch_candles_around(exit_dt, after_min=400):
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int((exit_dt - timedelta(minutes=15)).replace(tzinfo=timezone.utc).timestamp() * 1000)
    en = int((exit_dt + timedelta(minutes=after_min)).replace(tzinfo=timezone.utc).timestamp() * 1000)
    cs = ex.fetch_ohlcv('BTC-USDT', '15m', since=st, limit=30)
    return [c for c in cs if c[0] <= en]


def main():
    with open(ROOT / 'results' / 'c1_breakout_state.json') as f:
        d = json.load(f)
    th = d.get('trade_history', [])
    sl_trades = []
    for t in th:
        if t.get('reason') in ('EXCHANGE_SL', 'EXCHANGE_TRAIL'):
            try:
                edt = datetime.strptime(t.get('exit_time', '')[:19], '%Y-%m-%dT%H:%M:%S')
            except:
                continue
            if edt < datetime(2026, 4, 12):
                continue
            sl_trades.append({**t, 'exit_dt': edt})

    print(f"Analyzing {len(sl_trades)} SL/Trail trades — recovery at {HORIZONS} bars\n")

    rows = []
    for t in sl_trades:
        cs = fetch_candles_around(t['exit_dt'])
        if not cs or len(cs) < max(HORIZONS) + 1: continue
        exit_ts = int(t['exit_dt'].replace(tzinfo=timezone.utc).timestamp() * 1000)
        idx = next((j for j, c in enumerate(cs) if c[0] <= exit_ts < c[0]+15*60*1000), None)
        if idx is None: continue
        future = cs[idx+1:]
        if len(future) < max(HORIZONS): continue

        d_ = t.get('direction'); xp = t.get('exit_price')
        recoveries = {}
        for h in HORIZONS:
            window = future[:h]
            if d_ == 'LONG':
                m_ = max(c[2] for c in window)
                rec = (m_ - xp) / xp * 100
            else:
                m_ = min(c[3] for c in window)
                rec = (xp - m_) / xp * 100
            recoveries[h] = round(rec, 4)
        rows.append({'dir': d_, 'pnl': t.get('pnl_pct'), 'recoveries': recoveries})

    n = len(rows)
    print(f"Got {n} valid trades\n")

    # Aggregate
    print(f"{'horizon (bars)':<15} {'avg recover%':>13} {'pos rate %':>12} {'wick rate%':>12}")
    print("-"*60)
    for h in HORIZONS:
        recs = [r['recoveries'][h] for r in rows]
        pos_rate = 100 * sum(1 for r in recs if r > 0) / n
        # Wick rate = recovery > 50% of |pnl|/3
        wick_rate = 0
        for r in rows:
            sl_pct = abs(r['pnl']) / 3 if r['pnl'] else 0.5
            if r['recoveries'][h] > sl_pct * 0.5:
                wick_rate += 1
        wick_rate = 100 * wick_rate / n
        avg = sum(recs) / n
        print(f"{h:>15} {avg:>+12.3f}% {pos_rate:>11.1f}% {wick_rate:>11.1f}%")

    # Decay analysis: peak recovery vs final
    print(f"\nRecovery decay analysis (4→24 bars):")
    decays = []
    for r in rows:
        decay = r['recoveries'][24] - r['recoveries'][4]  # negative if recovered then dropped
        decays.append(decay)
    print(f"  Avg decay (24bar - 4bar): {sum(decays)/n:+.3f}%")
    print(f"  Trades that gave back recovery (decay < -0.2%): {sum(1 for d in decays if d < -0.2)}/{n}")

    out = {'date': datetime.now().isoformat(), 'n': n, 'horizons': HORIZONS, 'rows': rows}
    path = ROOT / 'results' / f'recovery_horizon_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
