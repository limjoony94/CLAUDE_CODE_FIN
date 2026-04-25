"""
Stop Hunt 가설 검증 — LIVE SL hit 후 가격 회복 분석
=======================================================
사용자 가설: "슬리피지 → 불리한 entry → SL/trail이 가까운 가격에 placed → stop hunt 공격"

검증 방법:
1. LIVE EXCHANGE_SL/EXCHANGE_TRAIL hit trades 추출 (state.json)
2. 각 exit bar 이후 N=8 bars (2시간) 가격 추적
3. 회복폭 측정: max(favorable price) - exit_price
   - LONG SL: max(high) - exit_price (= 얼마나 다시 올랐나)
   - SHORT SL: exit_price - min(low) (= 얼마나 다시 떨어졌나)
4. 평균 회복폭 > sl_pct/2 시 → wick exit (stop hunt) 시사
"""

import sys, os, json, math
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

LOOKAHEAD_BARS = 8  # 2 hours after exit


def fetch_candles_around(exit_dt, before_min=15, after_min=180):
    """Fetch BingX 15m candles around exit time."""
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int((exit_dt - timedelta(minutes=before_min)).replace(tzinfo=timezone.utc).timestamp() * 1000)
    en = int((exit_dt + timedelta(minutes=after_min)).replace(tzinfo=timezone.utc).timestamp() * 1000)
    cs = ex.fetch_ohlcv('BTC-USDT', '15m', since=st, limit=20)
    return [c for c in cs if c[0] <= en]


def analyze():
    with open(ROOT / 'results' / 'c1_breakout_state.json') as f:
        d = json.load(f)
    th = d.get('trade_history', [])

    # Filter to recent SL/Trail hits with adequate fields
    sl_trades = []
    for t in th:
        reason = t.get('reason', '')
        if reason in ('EXCHANGE_SL', 'EXCHANGE_TRAIL'):
            try:
                edt = datetime.strptime(t.get('exit_time', '')[:19], '%Y-%m-%dT%H:%M:%S')
            except:
                continue
            if edt < datetime(2026, 4, 12):
                continue
            sl_trades.append({**t, 'exit_dt': edt})

    print(f"Found {len(sl_trades)} EXCHANGE_SL/EXCHANGE_TRAIL trades since 04-12\n")

    print(f"{'#':>3} {'dir':<5} {'reason':<15} {'entry':>9} {'exit':>9} {'pnl':>7} {'recover_pct':>12} {'wick?'}")
    print("-" * 90)

    results = []
    for i, t in enumerate(sl_trades):
        dir_ = t.get('direction')
        entry = t.get('entry_price')
        exit_p = t.get('exit_price')
        pnl = t.get('pnl_pct')

        cs = fetch_candles_around(t['exit_dt'])
        if not cs or len(cs) < 2:
            continue

        exit_ts_ms = int(t['exit_dt'].replace(tzinfo=timezone.utc).timestamp() * 1000)
        # Find bar containing exit_dt
        idx = None
        for j, c in enumerate(cs):
            if c[0] <= exit_ts_ms < c[0] + 15*60*1000:
                idx = j; break
        if idx is None:
            continue

        # Look ahead LOOKAHEAD_BARS
        future = cs[idx+1:idx+1+LOOKAHEAD_BARS]
        if not future:
            continue

        # Recovery measurement
        if dir_ == 'LONG':
            # SL hit went down. Did price recover up?
            future_max = max(c[2] for c in future)  # max high
            recover_pct = (future_max - exit_p) / exit_p * 100  # positive = recovered up
        else:
            # SHORT: SL hit went up. Did price recover down?
            future_min = min(c[3] for c in future)  # min low
            recover_pct = (exit_p - future_min) / exit_p * 100  # positive = recovered down

        # Wick if recovery > 50% of pnl absolute
        sl_pct = abs(pnl) / 3 if pnl else 0  # de-leverage approx
        wick = recover_pct > sl_pct * 0.5

        results.append({
            'dir': dir_, 'reason': t.get('reason'), 'entry': entry,
            'exit': exit_p, 'pnl': pnl, 'recover_pct': recover_pct,
            'sl_pct_approx': sl_pct, 'wick': wick,
            'exit_dt': t['exit_dt'].isoformat(),
        })
        print(f"{i+1:>3} {dir_:<5} {t.get('reason'):<15} {entry:>9.1f} {exit_p:>9.1f} {pnl:>+6.2f}% {recover_pct:>+11.3f}% {'YES ⚠' if wick else 'no'}")

    # Summary
    if not results:
        print("\nNo data collected.")
        return
    print()
    n = len(results)
    n_wick = sum(1 for r in results if r['wick'])
    avg_recover = sum(r['recover_pct'] for r in results) / n
    avg_sl_pct = sum(r['sl_pct_approx'] for r in results) / n
    pos_recover = sum(1 for r in results if r['recover_pct'] > 0)
    print(f"Summary ({n} trades):")
    print(f"  Wick exits (recovery > 50% of sl_pct): {n_wick}/{n} ({100*n_wick/n:.1f}%)")
    print(f"  Positive recovery (any): {pos_recover}/{n} ({100*pos_recover/n:.1f}%)")
    print(f"  Avg recovery (favorable direction): {avg_recover:+.3f}%")
    print(f"  Avg sl_pct (1x): {avg_sl_pct:.3f}%")
    print(f"  Recovery / sl_pct ratio: {avg_recover/max(0.01,avg_sl_pct):.2f}")
    print()
    if n_wick > n * 0.4:
        print("⚠️  Wick exits >40% — stop hunt 가설 부분 지지")
    if avg_recover > avg_sl_pct * 0.3:
        print("⚠️  평균 회복 > 30% of sl_pct — 가격이 SL 후 회복하는 경향")

    # Save
    out = {
        'date': datetime.now().isoformat(),
        'lookahead_bars': LOOKAHEAD_BARS,
        'n': n, 'n_wick': n_wick, 'avg_recover_pct': avg_recover,
        'avg_sl_pct_1x': avg_sl_pct,
        'results': results,
    }
    path = ROOT / 'results' / f'stop_hunt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    analyze()
