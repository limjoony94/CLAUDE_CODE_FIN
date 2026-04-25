"""
Fold 2 vs CURRENT 추가 분석 — BingX LIVE data로 현재 환경 측정
==================================================================
fold2_regime은 CSV(04-03까지)로 5 fold 측정.
이 스크립트는 BingX에서 04-12~현재 LIVE 15m 데이터 fetch 후 동일 metric 계산.
이후 fold2_regime 결과와 결합하여 nearest fold 판정.
"""
import sys, math, statistics, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def fetch_live_15m():
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(datetime(2026, 4, 12).replace(tzinfo=timezone.utc).timestamp() * 1000)
    en = int(datetime(2026, 4, 25, 14, 0).replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_c = []
    cur = st
    while cur < en:
        cs = ex.fetch_ohlcv('BTC-USDT', '15m', since=cur, limit=1000)
        if not cs: break
        all_c.extend(cs)
        cur = cs[-1][0] + 15*60*1000
        if len(cs) < 1000: break
    seen=set(); uniq=[]
    for c in all_c:
        if c[0] not in seen:
            seen.add(c[0]); uniq.append(c)
    return sorted(uniq, key=lambda x: x[0])


def compute_metrics(candles):
    if len(candles) < 100:
        return None
    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows  = [c[3] for c in candles]
    n = len(candles)

    tr = []
    for i in range(1, n):
        tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    atr_pct = []
    for i in range(14, n):
        atr = sum(tr[i-14:i])/14
        atr_pct.append(atr/closes[i]*100)

    range_pcts = [(highs[i]-lows[i])/closes[i]*100 for i in range(n)]

    slopes = []
    for i in range(96, n):
        if closes[i-96] > 0:
            slopes.append((closes[i]-closes[i-96])/closes[i-96]*100)
    abs_slopes = [abs(s) for s in slopes]

    net_trend = (closes[-1]-closes[0])/closes[0]*100

    rets = [(closes[i]-closes[i-1])/closes[i-1]*100 for i in range(1, n)]
    ret_std = statistics.stdev(rets) if len(rets)>1 else 0

    return {
        'label': 'CURRENT (LIVE)',
        'period': f"{datetime.fromtimestamp(candles[0][0]/1000).date()} ~ {datetime.fromtimestamp(candles[-1][0]/1000).date()}",
        'n_bars': n,
        'atr_pct_mean': round(statistics.mean(atr_pct), 4) if atr_pct else 0,
        'atr_pct_median': round(statistics.median(atr_pct), 4) if atr_pct else 0,
        'atr_pct_p90': round(sorted(atr_pct)[int(0.9*len(atr_pct))], 4) if atr_pct else 0,
        'range_pct_mean': round(statistics.mean(range_pcts), 4),
        'range_pct_median': round(statistics.median(range_pcts), 4),
        'net_trend_pct': round(net_trend, 2),
        'abs_slope_24h_mean': round(statistics.mean(abs_slopes), 4) if abs_slopes else 0,
        'ret_std_15m': round(ret_std, 4),
    }


def main():
    print("Fetching LIVE 15m data (04-12 ~ now)...")
    candles = fetch_live_15m()
    print(f"  Got {len(candles)} candles\n")

    cur = compute_metrics(candles)
    if not cur:
        print("Insufficient data")
        return

    # Load fold metrics (recompute or read JSON)
    folds = [
        ("Fold 1 (4.5 ✅)", 0.2329, 0.2154, 0.3717, 0.2328,  +2.98, 1.2391, 0.1748),
        ("Fold 2 (4.5 ❌)", 0.2420, 0.2120, 0.4031, 0.2420,  -0.85, 1.3425, 0.1883),
        ("Fold 3 (4.5 ✅)", 0.3633, 0.3297, 0.6106, 0.3629, -18.09, 1.9120, 0.2642),
        ("Fold 4 (4.5 ✅)", 0.3097, 0.2480, 0.5705, 0.3107, -30.32, 1.5427, 0.2576),
        ("Fold 5 (4.5 ✅)", 0.4055, 0.3708, 0.6453, 0.4081, +11.03, 2.0852, 0.2964),
    ]

    print("=" * 130)
    print(f"{'Fold':<18} {'ATR%mean':>9} {'ATR%med':>9} {'ATR%p90':>9} {'Range%':>8} {'TrendNet%':>10} {'|slope|24h':>11} {'σ15m%':>7}")
    print("=" * 130)
    for name, am, amd, ap90, rng, tnet, sl, sd in folds:
        print(f"{name:<18} {am:>9.4f} {amd:>9.4f} {ap90:>9.4f} {rng:>8.4f} {tnet:>+9.2f}% {sl:>10.4f} {sd:>7.4f}")
    # Add CURRENT
    print(f"{cur['label']:<18} {cur['atr_pct_mean']:>9.4f} {cur['atr_pct_median']:>9.4f} {cur['atr_pct_p90']:>9.4f} "
          f"{cur['range_pct_mean']:>8.4f} {cur['net_trend_pct']:>+9.2f}% {cur['abs_slope_24h_mean']:>10.4f} {cur['ret_std_15m']:>7.4f}")
    print("=" * 130)

    # Compute Euclidean distance (normalized)
    print("\n--- Similarity to each fold (Euclidean on normalized 5 metrics) ---")
    metrics_idx = [(0,'atr_mean'), (1,'atr_med'), (3,'range_mean'), (5,'slope24h'), (6,'ret_std')]
    fold_arrays = [(name, [am, amd, ap90, rng, tnet, sl, sd]) for name, am, amd, ap90, rng, tnet, sl, sd in folds]
    cur_array = [cur['atr_pct_mean'], cur['atr_pct_median'], cur['atr_pct_p90'],
                 cur['range_pct_mean'], cur['net_trend_pct'], cur['abs_slope_24h_mean'], cur['ret_std_15m']]

    # min-max normalize each metric across all folds + CURRENT
    all_arrays = [arr for _, arr in fold_arrays] + [cur_array]
    keys_used = [0, 1, 3, 5, 6]  # atr_mean, atr_med, range_mean, slope24h, ret_std
    norms = []
    for k in keys_used:
        vals = [arr[k] for arr in all_arrays]
        mn, mx = min(vals), max(vals)
        norms.append((mn, mx-mn if mx>mn else 1))

    def n_arr(arr):
        return [(arr[k] - norms[i][0])/norms[i][1] for i, k in enumerate(keys_used)]

    cur_n = n_arr(cur_array)
    distances = []
    for name, arr in fold_arrays:
        f_n = n_arr(arr)
        d = math.sqrt(sum((cur_n[i]-f_n[i])**2 for i in range(len(cur_n))))
        distances.append((d, name, arr))
    distances.sort()

    print(f"{'Rank':>4} {'Fold':<18} {'Distance':>10}  Diff (CURRENT - fold)")
    print("-" * 90)
    for r, (d, name, arr) in enumerate(distances):
        diff = f"atrM:{cur_array[0]-arr[0]:+.4f} rng:{cur_array[3]-arr[3]:+.4f} sl24h:{cur_array[5]-arr[5]:+.4f} σ:{cur_array[6]-arr[6]:+.4f}"
        print(f"{r+1:>4} {name:<18} {d:>10.4f}  {diff}")

    nearest = distances[0][1]
    print(f"\n🎯 CURRENT 시장 = {nearest}와(과) 가장 유사")
    if "Fold 2" in nearest:
        print("  ⚠️  CRITICAL: Fold 2 (4.5 underperform)와 가장 유사 → max_sl 4.5 위험 신호")
    else:
        # check if fold 2 is in top 2
        f2_rank = next((i for i, (_, n, _) in enumerate(distances) if "Fold 2" in n), None)
        if f2_rank == 1:
            print(f"  ⚠️  Fold 2가 #2 nearest — 부분 위험")
        else:
            print(f"  ✅ Fold 2가 nearest 아님 (rank #{f2_rank+1 if f2_rank else 'NA'})")

    # Save
    out = {
        'date': datetime.now().isoformat(),
        'current_metrics': cur,
        'fold_distances': [(d, n) for d, n, _ in distances],
    }
    path = ROOT / 'results' / f'fold2_current_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
