"""
Fold 2 Regime Analysis — 왜 2025-08-28~2025-10-21이 max_sl 4.5에 불리했나?
============================================================================
Fold 2: 4.5 = +28.47% vs 3.3 = +33.60% (Δ -5.13pp, 유일한 negative fold)
다른 4개 fold는 모두 4.5 우월. 어떤 시장 환경 차이가 있었는가?

분석 차원:
1. ATR% (volatility regime) — 평균/중앙값
2. Trend strength (rolling slope of close)
3. Bar range distribution (high-low / close)
4. SL hit frequency in fold (그 fold에서 SL이 더 자주 트리거됐나?)
5. 현재 시장(04-12 이후)과의 유사도

목적: 현재 시장이 Fold 2와 유사하면 → 4.5 적용 위험 신호.
       다른 fold들과 유사하면 → 4.5 적용 정당화.
"""
import sys, os, json, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_csv():
    import pandas as pd
    p = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(p, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return df15


def stats_for_window(df, start, end, label):
    sub = df[(df['timestamp'] >= start) & (df['timestamp'] < end)].reset_index(drop=True)
    if len(sub) < 14:
        return None

    # ATR% (using 14-period True Range relative to close)
    tr = []
    for i in range(1, len(sub)):
        h, l, pc = sub.loc[i, 'high'], sub.loc[i, 'low'], sub.loc[i-1, 'close']
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr_pct = []
    for i in range(14, len(sub)):
        atr = sum(tr[i-14:i]) / 14
        atr_pct.append(atr / sub.loc[i, 'close'] * 100)

    # Range pct distribution
    range_pcts = ((sub['high'] - sub['low']) / sub['close'] * 100).tolist()

    # Trend strength: rolling 96-bar (24h) slope
    close = sub['close'].tolist()
    slopes_pct = []
    for i in range(96, len(close)):
        c0 = close[i - 96]; c1 = close[i]
        if c0 > 0:
            slopes_pct.append((c1 - c0) / c0 * 100)
    abs_slopes = [abs(s) for s in slopes_pct]

    # Net trend (start → end)
    net_trend = (close[-1] - close[0]) / close[0] * 100 if close[0] > 0 else 0

    # Volatility of returns (15m bar-to-bar)
    rets = []
    for i in range(1, len(close)):
        if close[i-1] > 0:
            rets.append((close[i] - close[i-1]) / close[i-1] * 100)
    ret_std = statistics.stdev(rets) if len(rets) > 1 else 0

    return {
        'label': label,
        'period': f"{start.date()} ~ {end.date()}",
        'n_bars': len(sub),
        'days': (end - start).days,
        'atr_pct_mean': round(statistics.mean(atr_pct), 4) if atr_pct else 0,
        'atr_pct_median': round(statistics.median(atr_pct), 4) if atr_pct else 0,
        'atr_pct_p90': round(sorted(atr_pct)[int(0.9*len(atr_pct))], 4) if atr_pct else 0,
        'range_pct_mean': round(statistics.mean(range_pcts), 4),
        'range_pct_median': round(statistics.median(range_pcts), 4),
        'net_trend_pct': round(net_trend, 2),
        'abs_slope_24h_mean': round(statistics.mean(abs_slopes), 4) if abs_slopes else 0,
        'ret_std_15m': round(ret_std, 4),
        'price_start': round(close[0], 1),
        'price_end': round(close[-1], 1),
    }


def main():
    print("Loading 15m data...")
    df = load_csv()
    print(f"  {len(df)} bars, {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}\n")

    # 5 folds from max_sl_wf
    folds = [
        ("Fold 1 (4.5 ✅)", datetime(2025, 7, 5), datetime(2025, 8, 28)),
        ("Fold 2 (4.5 ❌)", datetime(2025, 8, 28), datetime(2025, 10, 21)),
        ("Fold 3 (4.5 ✅)", datetime(2025, 10, 21), datetime(2025, 12, 14)),
        ("Fold 4 (4.5 ✅)", datetime(2025, 12, 14), datetime(2026, 2, 6)),
        ("Fold 5 (4.5 ✅)", datetime(2026, 2, 6), datetime(2026, 4, 3)),
        ("CURRENT (LIVE)", datetime(2026, 4, 12), datetime(2026, 4, 25, 12)),
    ]

    results = []
    for label, st, en in folds:
        r = stats_for_window(df, st, en, label)
        if r:
            results.append(r)

    # Print table
    print("=" * 130)
    print(f"{'Fold':<18} {'Period':<22} {'days':>5} {'ATR%mean':>9} {'ATR%med':>9} {'ATR%p90':>9} {'Range%':>8} {'TrendNet%':>10} {'|slope|24h':>11} {'σ15m%':>7}")
    print("=" * 130)
    for r in results:
        print(f"{r['label']:<18} {r['period']:<22} {r['days']:>5} "
              f"{r['atr_pct_mean']:>9.4f} {r['atr_pct_median']:>9.4f} {r['atr_pct_p90']:>9.4f} "
              f"{r['range_pct_mean']:>8.4f} {r['net_trend_pct']:>+9.2f}% "
              f"{r['abs_slope_24h_mean']:>10.4f} {r['ret_std_15m']:>7.4f}")
    print("=" * 130)

    # Compute similarity of CURRENT to each fold (cosine on key metrics)
    cur = next((r for r in results if 'CURRENT' in r['label']), None)
    if cur:
        print("\nSimilarity of CURRENT to each fold (Euclidean distance on normalized metrics):")
        print(f"{'Fold':<18} {'distance':>10}  diff (CURRENT - fold)")
        print("-" * 80)
        # Normalize each metric across folds
        keys = ['atr_pct_mean', 'atr_pct_median', 'range_pct_mean', 'abs_slope_24h_mean', 'ret_std_15m']
        fold_only = [r for r in results if 'CURRENT' not in r['label']]
        # min-max normalization
        norms = {}
        for k in keys:
            vals = [r[k] for r in fold_only + [cur]]
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx > mn else 1
            norms[k] = (mn, rng)

        def norm(v, k):
            mn, rng = norms[k]
            return (v - mn) / rng

        ranked = []
        for r in fold_only:
            d = math.sqrt(sum((norm(r[k], k) - norm(cur[k], k))**2 for k in keys))
            ranked.append((d, r))
        ranked.sort()
        for d, r in ranked:
            diffs = [(k, cur[k] - r[k]) for k in keys]
            print(f"{r['label']:<18} {d:>10.4f}  " + "  ".join(f"{k.split('_')[0][:5]}:{v:+.4f}" for k, v in diffs))

        nearest_label = ranked[0][1]['label']
        print(f"\n→ CURRENT 시장은 {nearest_label}와(과) 가장 유사")
        if "Fold 2" in nearest_label:
            print("  ⚠️  CRITICAL: Fold 2 (4.5 underperform fold)와 가장 유사!")
            print("       max_sl 4.5 production 적용은 이 환경에서 underperform 가능")
        else:
            print("  ✅ Fold 2가 아닌 다른 fold와 유사 → max_sl 4.5 적용 환경 양호")

    # Save
    out = {
        'date': datetime.now().isoformat(),
        'folds': results,
    }
    path = ROOT / 'results' / f'fold2_regime_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
