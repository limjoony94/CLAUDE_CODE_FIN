"""
Stop Hunt CONTROL GROUP — Random non-SL bars 24-bar recovery rate
====================================================================
Advisor critique: stop_hunt_analysis_20260425 의 97.1% recovery는
control group이 없어 BTC 노이즈와 구분 불가. BTC 변동성 전제 시
랜덤 시점에서도 ANY 방향 회복은 trivial하게 나타날 수 있다.

검증 절차:
1. Same period (2026-04-12 ~ 현재) 15m 캔들 추출
2. 실제 SL/Trail hit ts 제외 (±8 bars 버퍼)
3. 34개 (실제 N과 동일) 랜덤 bar 추출, LONG/SHORT 17:17 균등 할당
4. 각 bar의 close = "exit_price" 가정
5. 동일한 8-bar lookahead 회복 metric 계산
6. wick% / pos_recover% / avg_recover_pct를 실제 결과(67.6% / 97.1% / +0.615%)와 비교

판정:
- Control pos_recover% ≥ 90% → 메트릭이 BTC 노이즈 trivial → 회복 narrative 약함
- Control pos_recover% << 70% → SL hit 후 회복이 정말 random보다 큼

추가: avg_recover_pct (favorable direction) — 실제는 +0.615%. Control 비교.
"""
import sys, os, json, math, random
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

LOOKAHEAD_BARS = 8
N_TRIALS = 34
EXCLUDE_BUFFER_BARS = 8
SEED = 42  # reproducibility

random.seed(SEED)


def load_actual_sl_times():
    """실제 SL/Trail hit exit_time 목록"""
    with open(ROOT / 'results' / 'c1_breakout_state.json') as f:
        d = json.load(f)
    times = []
    for t in d.get('trade_history', []):
        if t.get('reason') in ('EXCHANGE_SL', 'EXCHANGE_TRAIL'):
            try:
                edt = datetime.strptime(t.get('exit_time', '')[:19], '%Y-%m-%dT%H:%M:%S')
                if edt >= datetime(2026, 4, 12):
                    times.append(edt)
            except:
                pass
    return times


def load_15m_candles():
    """BingX 15m 04-12 ~ 현재"""
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(datetime(2026, 4, 12).replace(tzinfo=timezone.utc).timestamp() * 1000)
    en = int(datetime(2026, 4, 25, 12, 0).replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_c = []
    cur = st
    while cur < en:
        cs = ex.fetch_ohlcv('BTC-USDT', '15m', since=cur, limit=1000)
        if not cs:
            break
        all_c.extend(cs)
        cur = cs[-1][0] + 15 * 60 * 1000
        if len(cs) < 1000:
            break
    # dedupe
    seen = set(); uniq = []
    for c in all_c:
        if c[0] not in seen:
            seen.add(c[0]); uniq.append(c)
    return sorted(uniq, key=lambda x: x[0])


def main():
    print("Loading actual SL/Trail times...")
    sl_times = load_actual_sl_times()
    print(f"  Found {len(sl_times)} actual SL/Trail events\n")

    print("Loading BingX 15m candles (04-12 ~ now)...")
    candles = load_15m_candles()
    print(f"  Got {len(candles)} candles\n")

    # Build excluded indices around actual SL events
    excluded = set()
    for sdt in sl_times:
        sts = int(sdt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        for j, c in enumerate(candles):
            if c[0] <= sts < c[0] + 15 * 60 * 1000:
                for off in range(-EXCLUDE_BUFFER_BARS, EXCLUDE_BUFFER_BARS + 1):
                    if 0 <= j + off < len(candles):
                        excluded.add(j + off)
                break

    # Sample non-excluded bars (need lookahead)
    eligible = [j for j in range(len(candles) - LOOKAHEAD_BARS - 1) if j not in excluded]
    print(f"Eligible bars (excl ±{EXCLUDE_BUFFER_BARS} around SL events): {len(eligible)}")
    if len(eligible) < N_TRIALS:
        print(f"WARN: only {len(eligible)} eligible, reducing N to that")

    sample = random.sample(eligible, min(N_TRIALS, len(eligible)))

    # Match average sl_pct from actual analysis: ~0.748%, but for control
    # we use the same "any positive recovery" criterion as actual.
    print(f"\nRunning {len(sample)} random control trials (LONG/SHORT 50/50)...")
    print(f"{'#':>3} {'dir':<5} {'exit_ts':<19} {'exit_p':>9} {'recover_pct':>12} {'pos?':>5}")
    print("-" * 70)

    results = []
    # Use the actual avg sl_pct as threshold reference
    AVG_SL_PCT = 0.748  # from stop_hunt_20260425
    for i, j in enumerate(sample):
        c = candles[j]
        exit_p = c[4]  # close
        # Random direction
        dir_ = 'LONG' if random.random() < 0.5 else 'SHORT'
        future = candles[j+1:j+1+LOOKAHEAD_BARS]
        if len(future) < LOOKAHEAD_BARS:
            continue
        if dir_ == 'LONG':
            future_max = max(fc[2] for fc in future)
            recover_pct = (future_max - exit_p) / exit_p * 100
        else:
            future_min = min(fc[3] for fc in future)
            recover_pct = (exit_p - future_min) / exit_p * 100

        wick = recover_pct > AVG_SL_PCT * 0.5  # 매칭: wick = recover > 50% of avg sl_pct

        results.append({
            'dir': dir_, 'exit_ts': datetime.fromtimestamp(c[0]/1000).isoformat()[:19],
            'exit_p': exit_p, 'recover_pct': recover_pct, 'wick': wick,
        })
        if i < 15 or i >= len(sample) - 5:
            ts_str = datetime.fromtimestamp(c[0]/1000).isoformat()[:19]
            mark = "YES" if recover_pct > 0 else "no"
            print(f"{i+1:>3} {dir_:<5} {ts_str:<19} {exit_p:>9.1f} {recover_pct:>+11.3f}% {mark:>5}")
        elif i == 15:
            print(f"  ... ({len(sample) - 20} more) ...")

    n = len(results)
    n_pos = sum(1 for r in results if r['recover_pct'] > 0)
    n_wick = sum(1 for r in results if r['wick'])
    avg_recover = sum(r['recover_pct'] for r in results) / n

    print("\n" + "=" * 75)
    print("CONTROL GROUP RESULTS")
    print("=" * 75)
    print(f"Sample size: {n} (random non-SL bars)")
    print(f"Positive recovery (any favorable move): {n_pos}/{n} ({100*n_pos/n:.1f}%)")
    print(f"Wick threshold (recovery > 50% of avg sl_pct={AVG_SL_PCT:.3f}%): {n_wick}/{n} ({100*n_wick/n:.1f}%)")
    print(f"Avg recovery (favorable direction, magnitude): {avg_recover:+.3f}%")

    # Compare to actual
    print("\n" + "=" * 75)
    print("COMPARISON vs ACTUAL SL HITS")
    print("=" * 75)
    actual_n = 34
    actual_pos = 33  # 97.1%
    actual_wick = 23  # 67.6%
    actual_avg_recover = 0.615
    print(f"{'Metric':<28} {'Actual':>12} {'Control':>12} {'Δ':>10}")
    print("-" * 65)
    print(f"{'Pos recovery rate':<28} {100*actual_pos/actual_n:>11.1f}% {100*n_pos/n:>11.1f}% {100*(n_pos/n - actual_pos/actual_n):>+9.1f}pp")
    print(f"{'Wick rate (>50% sl_pct)':<28} {100*actual_wick/actual_n:>11.1f}% {100*n_wick/n:>11.1f}% {100*(n_wick/n - actual_wick/actual_n):>+9.1f}pp")
    print(f"{'Avg recovery':<28} {actual_avg_recover:>+11.3f}% {avg_recover:>+11.3f}% {avg_recover - actual_avg_recover:>+9.3f}pp")

    print("\nVerdict:")
    if n_pos / n >= 0.90:
        print("  ⚠️  Control pos_recover ≥ 90% — metric is TRIVIAL on BTC")
        print("      → 'SL hit 후 97.1% 회복' narrative는 BTC 노이즈로 설명 가능")
        print("      → stop hunt 가설 약화, max_sl 4.5 적용 근거 재검토 필요")
    elif n_pos / n >= 0.70:
        print("  ⚠️  Control pos_recover 70~90% — partial overlap")
        print("      → SL 시점 유의미 효과 있을 수 있으나 BTC 변동성 기여 큼")
    else:
        print("  ✅ Control pos_recover < 70% — SL hit 시점이 random보다 회복 강함")
        print("      → stop hunt narrative 부분 지지")

    # Wick comparison (more meaningful since it has a magnitude threshold)
    if n_wick / n >= 0.60:
        print(f"  ⚠️  Wick rate {100*n_wick/n:.1f}% ≥ 60% — 충분한 magnitude 회복도 random에서 발생")
    else:
        print(f"  ✅ Wick rate {100*n_wick/n:.1f}% — 실제 SL hit가 magnitude 회복도 더 자주 발생")

    # Save
    out = {
        'date': datetime.now().isoformat(),
        'seed': SEED,
        'lookahead_bars': LOOKAHEAD_BARS,
        'n_control': n, 'n_pos_control': n_pos, 'n_wick_control': n_wick,
        'avg_recover_control': avg_recover,
        'actual_pos_rate': actual_pos / actual_n,
        'control_pos_rate': n_pos / n,
        'actual_wick_rate': actual_wick / actual_n,
        'control_wick_rate': n_wick / n,
        'actual_avg_recover': actual_avg_recover,
        'samples': results,
    }
    path = ROOT / 'results' / f'stop_hunt_control_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
