"""
Diagnostic 1: WR 28% vs BT 40% — intrabar SL behavior 측정
============================================================
질문: LIVE SL hits에서 price wick 후 회복 비율이 BT 가정과 다른가?
방법:
  1. LIVE 14 SL hit trades 추출 (state.json reason='EXCHANGE_SL')
  2. 각 SL hit의 entry → SL hit time 사이 5m intrabar 데이터 fetch
  3. SL trigger 후 N=8 bars (2시간) 동안 가격 회복 측정
판정:
  Wick-recover 비율 ≥ 50% → SL placement issue
  Wick-recover 비율 30~50% → tighter SL 필요할 수 있음
  Wick-recover 비율 < 30% → SL은 OK, 다른 원인
"""
import sys, json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def fetch_5m(start_utc, end_utc):
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(start_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
    en = int(end_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
    out = []; cur = st
    while cur <= en:
        cs = ex.fetch_ohlcv('BTC-USDT', '5m', since=cur, limit=1000)
        if not cs: break
        out.extend(cs)
        cur = cs[-1][0] + 5 * 60 * 1000
        if len(cs) < 1000: break
    seen = set(); uniq = []
    for c in out:
        if c[0] not in seen and c[0] <= en:
            seen.add(c[0]); uniq.append(c)
    return sorted(uniq, key=lambda x: x[0])


def main():
    d = json.load(open(ROOT / 'results' / 'c1_breakout_state.json'))
    ths = d.get('trade_history', [])

    sl_trades = [t for t in ths if t.get('reason') == 'EXCHANGE_SL']
    print(f"Total LIVE SL hits: {len(sl_trades)}\n")

    # For each SL hit, look up: entry → SL exit → next 8 bars (2h)
    # Note: trade history doesn't have entry_time, so we approximate from exit_time - bars_held*15min
    print("=" * 110)
    print(f"{'#':>3} {'exit_time':<19} {'dir':<5} {'entry':>9} {'exit':>9} {'bars':>5} {'wick%':>7} {'recover_n2h':>11} {'wicked?'}")
    print("=" * 110)

    n_wicked = 0
    n_analyzed = 0
    rows = []

    for i, t in enumerate(sl_trades, 1):
        try:
            exit_dt = datetime.strptime(t['exit_time'][:19], '%Y-%m-%dT%H:%M:%S')
        except:
            continue
        bars = t.get('bars_held', 0)
        entry_dt = exit_dt - timedelta(minutes=15 * bars)

        # Fetch 5m candles from entry to exit + 2h
        try:
            cs = fetch_5m(entry_dt - timedelta(minutes=10), exit_dt + timedelta(hours=2, minutes=10))
        except Exception as e:
            print(f"#{i} fetch error: {e}")
            continue

        if not cs:
            continue

        d_ = t.get('direction')
        entry = t.get('entry_price', 0)
        exit_p = t.get('exit_price', 0)
        # Find SL hit candle (closest to exit_dt)
        exit_ts = int(exit_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        sl_idx = None
        for j, c in enumerate(cs):
            if c[0] >= exit_ts:
                sl_idx = j; break
        if sl_idx is None or sl_idx >= len(cs) - 1:
            continue

        # Wick analysis: from exit candle's close, look ahead 2h (24 5m bars)
        future = cs[sl_idx:sl_idx + 25]  # SL bar + 24 future = 2h
        if len(future) < 5:
            continue

        # Did price reverse? For LONG SL (price went down), did it go back up?
        if d_ == 'LONG':
            # SL hit at exit_p (lower than entry). Future high reaching back to entry?
            future_max = max(c[2] for c in future)
            recover_pct = (future_max - exit_p) / exit_p * 100
            crossed_back = future_max >= entry  # price went back to entry within 2h
        else:
            future_min = min(c[3] for c in future)
            recover_pct = (exit_p - future_min) / exit_p * 100
            crossed_back = future_min <= entry

        # SL distance %
        sl_pct = abs(entry - exit_p) / entry * 100

        # Wicked = recovered > 50% of SL distance OR crossed entry
        wicked = (recover_pct > sl_pct * 0.5) or crossed_back
        if wicked:
            n_wicked += 1
        n_analyzed += 1

        rows.append({
            'exit_time': t['exit_time'][:19], 'direction': d_,
            'entry': entry, 'exit': exit_p, 'bars': bars,
            'sl_pct': round(sl_pct, 4),
            'recover_pct': round(recover_pct, 4),
            'crossed_back': crossed_back,
            'wicked': wicked,
        })

        print(f"{i:>3} {t['exit_time'][:19]:<19} {d_:<5} {entry:>9.1f} {exit_p:>9.1f} "
              f"{bars:>5} {sl_pct:>6.3f}% {recover_pct:>+10.3f}% {'YES ⚠' if wicked else 'no'}"
              + (f" (crossed back)" if crossed_back else ""))

    print("=" * 110)
    print(f"\nTotal SL analyzed: {n_analyzed}")
    if n_analyzed > 0:
        print(f"Wicked (recover > 50% sl_pct OR crossed back to entry): {n_wicked}/{n_analyzed} ({100*n_wicked/n_analyzed:.1f}%)")

    # Verdict
    print("\n=== Verdict ===")
    if n_analyzed == 0:
        print("Insufficient data")
    else:
        wick_rate = 100*n_wicked/n_analyzed
        if wick_rate >= 50:
            print(f"⚠️  Wick rate {wick_rate:.1f}% — SL placement issue")
            print("    Price gets to SL intrabar then recovers. BT gets same SL hit, but crucial:")
            print("    BT also catches intrabar (uses high/low) — so this is NOT BT-LIVE divergence.")
            print("    However → SL is structurally TOO TIGHT for BTC noise. Re-design needed.")
        elif wick_rate >= 30:
            print(f"➡️  Wick rate {wick_rate:.1f}% — moderate, SL slightly tight")
        else:
            print(f"✅ Wick rate {wick_rate:.1f}% — SL OK, other factors dominate")

    out = {
        'date': datetime.now().isoformat(),
        'n_total': len(sl_trades), 'n_analyzed': n_analyzed, 'n_wicked': n_wicked,
        'wick_rate_pct': round(100*n_wicked/n_analyzed, 2) if n_analyzed > 0 else 0,
        'samples': rows,
    }
    p = ROOT / 'results' / f'diag1_intrabar_sl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
