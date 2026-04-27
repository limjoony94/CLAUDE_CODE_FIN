"""
Post-#44 신호 누락 검증 — 04-26 23:30 (Trade #44 청산) ~ 현재
=================================================================
질문: 봇이 #44 이후 거래 0건. 신호가 정말 없었나, 봇이 놓쳤나?

방법:
1. BingX 15m candles fetch (04-26 23:30 ~ now)
2. signals.py.check_entry로 모든 bar에 대해 raw signal 평가
3. signal 발생 bar 모두 출력 (direction, entry, sl, body_ratio)
4. 봇 로그의 ENTRY 라인과 cross-check
   → BT 신호 발생했는데 LIVE에 없으면: 봇 누락 (조사 필요)
   → 신호 자체가 없으면: 시장 chop, 봇 정상
"""
import sys, json, math, yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)


def fetch_15m(start_utc, end_utc):
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(start_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
    en = int(end_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
    out = []; cur = st
    while cur <= en:
        cs = ex.fetch_ohlcv('BTC-USDT', '15m', since=cur, limit=1000)
        if not cs: break
        out.extend(cs)
        cur = cs[-1][0] + 15*60*1000
        if len(cs) < 1000: break
    seen=set(); uniq=[]
    for c in out:
        if c[0] not in seen and c[0] <= en:
            seen.add(c[0]); uniq.append(c)
    return sorted(uniq, key=lambda x: x[0])


def main():
    cfg = yaml.safe_load(open(ROOT / 'config' / 'c1_breakout_config.yaml'))
    sig = C1BreakoutSignal(cfg['strategy'])

    # Fetch wide window: 1 day BEFORE #44 exit (for ATR + channel warmup) to now
    # #44 청산 = 04-26 14:21 UTC. Fetch from 04-25 12:00 UTC.
    start = datetime(2026, 4, 25, 12, 0, 0)
    end = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
    print(f"Fetching 15m: {start} ~ {end}")
    candles = fetch_15m(start, end)
    print(f"  {len(candles)} bars\n")

    # Compute indicators
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows  = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    atr = compute_atr(highs, lows, closes, sig.atr_period)
    chh, chl = compute_channel(highs, lows, sig.channel_period)
    swl, swh = compute_fractal_swings(highs, lows, 10)

    # POST_44_TS: Trade #44 청산 시점 (04-26 14:21 UTC = KST 23:21)
    # 사실상 다음 cycle 14:30 UTC부터 새 entry 가능
    post44_ts_ms = int(datetime(2026, 4, 26, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)

    print("=" * 110)
    print(f"{'#':>3} {'utc_time':<19} {'kst':<19} {'O':>8} {'H':>8} {'L':>8} {'C':>8} {'ch_H':>8} {'ch_L':>8} {'signal'}")
    print("=" * 110)

    n_signals_post = 0
    n_signals_pre  = 0
    bars_evaluated = 0
    last_print_idx = -100

    for i in range(20, len(candles)):  # skip warmup
        c = candles[i]
        ts_ms = c[0]
        utc_dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).replace(tzinfo=None)
        kst_dt = utc_dt + timedelta(hours=9)

        # Run check_entry
        if math.isnan(atr[i]) or math.isnan(chh[i]):
            continue
        es = sig.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                             bar_close=closes[i], channel_high=chh[i], channel_low=chl[i],
                             atr_val=atr[i], last_swing_low=swl[i], last_swing_high=swh[i])
        bars_evaluated += 1
        is_post44 = ts_ms >= post44_ts_ms
        if es:
            if is_post44: n_signals_post += 1
            else: n_signals_pre += 1
            mark = "🚨 SIGNAL" if is_post44 else "(pre #44)"
            print(f"{i:>3} {utc_dt.isoformat():<19} {kst_dt.strftime('%Y-%m-%d %H:%M:%S'):<19} "
                  f"{opens[i]:>8.1f} {highs[i]:>8.1f} {lows[i]:>8.1f} {closes[i]:>8.1f} "
                  f"{chh[i]:>8.1f} {chl[i]:>8.1f}  {es['direction']} {mark}")
            last_print_idx = i

    # Summary
    print("=" * 110)
    print(f"\nBars evaluated: {bars_evaluated}")
    print(f"Signals BEFORE #44 청산 (04-26 14:30 UTC 기준): {n_signals_pre}")
    print(f"Signals AFTER  #44 청산: {n_signals_post}")

    # Time since last sample
    last_c = candles[-1]
    last_dt = datetime.fromtimestamp(last_c[0]/1000, tz=timezone.utc).replace(tzinfo=None)
    print(f"\nLast bar: {last_dt.isoformat()} UTC | C={last_c[4]:.1f}")
    print(f"Last 15-bar channel: HIGH={max(c[2] for c in candles[-16:-1]):.1f} LOW={min(c[3] for c in candles[-16:-1]):.1f}")
    print(f"Current C={last_c[4]:.1f} → channel range distance:")
    chh_now = max(c[2] for c in candles[-16:-1])
    chl_now = min(c[3] for c in candles[-16:-1])
    dist_up = (chh_now - last_c[4]) / last_c[4] * 100
    dist_dn = (last_c[4] - chl_now) / last_c[4] * 100
    print(f"  to upper breakout: {dist_up:+.3f}% (~${chh_now-last_c[4]:.1f})")
    print(f"  to lower breakout: {dist_dn:+.3f}% (~${last_c[4]-chl_now:.1f})")

    # Verdict
    print()
    if n_signals_post == 0:
        print("✅ 봇 정상 — 신호 자체가 없었음 (시장 chop)")
    else:
        print(f"⚠️  BT에서 {n_signals_post}개 신호 감지 — LIVE 봇 ENTRY 로그와 비교 필요")
        print(f"    LIVE 봇 ENTRY (04-26 23:30 이후): 0건")
        print(f"    → 누락 의심. 코드/실행 환경 조사 필요.")


if __name__ == '__main__':
    main()
