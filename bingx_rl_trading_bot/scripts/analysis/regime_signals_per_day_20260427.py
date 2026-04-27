"""
Regime conditional signals/day — Channel range별 신호 빈도 분류
==================================================================
질문: 현재 channel 0.42% 폭. 이 chop regime의 BT 평균 signals/day는?
방법: 272d eval에서 매 bar의 channel range%를 측정, 4분위로 bucket,
      각 bucket의 raw signals/day 측정.
"""
import sys, json, math, yaml
from datetime import datetime, timedelta
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close']] for _, r in df15.iterrows()]


def main():
    cfg = yaml.safe_load(open(ROOT / 'config' / 'c1_breakout_config.yaml'))
    sig = C1BreakoutSignal(cfg['strategy'])
    candles = load_csv_15m(str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv'))
    print(f"{len(candles)} bars")

    t0 = datetime.fromtimestamp(candles[0][0]/1000)
    s_idx = next(i for i, c in enumerate(candles) if c[0] >= int((t0+timedelta(days=60)).timestamp()*1000))
    e_idx = len(candles) - 1

    opens = [c[1] for c in candles]; highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]; closes = [c[4] for c in candles]
    atr = compute_atr(highs, lows, closes, sig.atr_period)
    chh, chl = compute_channel(highs, lows, sig.channel_period)
    swl, swh = compute_fractal_swings(highs, lows, 10)

    # For each bar i in eval, compute channel_range_pct, check_entry(produces signal?)
    bar_data = []  # (range_pct, has_signal)
    for i in range(s_idx, e_idx):
        if math.isnan(chh[i]) or math.isnan(chl[i]) or chh[i] <= chl[i]:
            continue
        rng_pct = (chh[i] - chl[i]) / closes[i] * 100
        es = sig.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
            bar_close=closes[i], channel_high=chh[i], channel_low=chl[i],
            atr_val=atr[i], last_swing_low=swl[i], last_swing_high=swh[i])
        bar_data.append((rng_pct, 1 if es else 0))

    n_bars = len(bar_data)
    n_signals = sum(b[1] for b in bar_data)
    print(f"\nTotal bars: {n_bars}, signals: {n_signals} ({100*n_signals/n_bars:.2f}%)")

    # Quartiles by range_pct
    sorted_rng = sorted(b[0] for b in bar_data)
    q1 = sorted_rng[int(0.25 * n_bars)]
    q2 = sorted_rng[int(0.50 * n_bars)]
    q3 = sorted_rng[int(0.75 * n_bars)]
    bars_per_day = 96  # 15m

    print(f"\nChannel range% quartiles: P25={q1:.3f}% | P50={q2:.3f}% | P75={q3:.3f}%")
    print(f"Mean: {statistics.mean(b[0] for b in bar_data):.3f}%")
    print()
    print("=" * 80)
    print(f"{'Bucket (range%)':<25} {'bars':>6} {'signals':>9} {'sig_rate':>10} {'sig/day':>8}")
    print("=" * 80)
    buckets = [
        (f"VERY LOW (<P25 ≤{q1:.3f}%)", lambda r: r <= q1),
        (f"LOW (P25~P50 ≤{q2:.3f}%)", lambda r: q1 < r <= q2),
        (f"HIGH (P50~P75 ≤{q3:.3f}%)", lambda r: q2 < r <= q3),
        (f"VERY HIGH (>P75)", lambda r: r > q3),
    ]
    for name, test in buckets:
        sub = [b for b in bar_data if test(b[0])]
        s_count = sum(b[1] for b in sub)
        rate = 100 * s_count / len(sub) if sub else 0
        sig_per_day = s_count / (len(sub)/bars_per_day) if sub else 0
        print(f"{name:<25} {len(sub):>6} {s_count:>9} {rate:>9.2f}% {sig_per_day:>7.2f}")

    # Current LIVE channel range%
    print()
    print("=" * 80)
    cur_rng_pct = 0.42  # measured earlier
    print(f"Current LIVE channel range: ~{cur_rng_pct:.2f}%")
    if cur_rng_pct <= q1: tier = "VERY LOW (<P25)"
    elif cur_rng_pct <= q2: tier = "LOW (P25~P50)"
    elif cur_rng_pct <= q3: tier = "HIGH (P50~P75)"
    else: tier = "VERY HIGH (>P75)"
    print(f"  → Regime tier: {tier}")
    sub = [b for b in bar_data if (
        (cur_rng_pct <= q1 and b[0] <= q1) or
        (q1 < cur_rng_pct <= q2 and q1 < b[0] <= q2) or
        (q2 < cur_rng_pct <= q3 and q2 < b[0] <= q3) or
        (q3 < cur_rng_pct and b[0] > q3)
    )]
    s_count = sum(b[1] for b in sub)
    sig_per_day = s_count / (len(sub)/bars_per_day) if sub else 0
    print(f"  Expected signals/day in this regime: {sig_per_day:.2f}")
    print(f"  Days/trade expected: {1/sig_per_day*0.373:.1f}h (selection 37.3% adoption)" if sig_per_day else "")
    # Note: 0.373 is N=1 adoption rate from selection_variance_20260426

    out = {
        'date': datetime.now().isoformat(),
        'quartiles_pct': {'q1': q1, 'q2': q2, 'q3': q3},
        'mean_range_pct': statistics.mean(b[0] for b in bar_data),
        'overall_signals_per_day': sum(b[1] for b in bar_data) / (n_bars/bars_per_day),
        'bucket_signals_per_day': {
            'very_low': sum(b[1] for b in bar_data if b[0] <= q1) / (sum(1 for b in bar_data if b[0] <= q1)/bars_per_day),
            'low': sum(b[1] for b in bar_data if q1 < b[0] <= q2) / max(1, sum(1 for b in bar_data if q1 < b[0] <= q2)/bars_per_day),
            'high': sum(b[1] for b in bar_data if q2 < b[0] <= q3) / max(1, sum(1 for b in bar_data if q2 < b[0] <= q3)/bars_per_day),
            'very_high': sum(b[1] for b in bar_data if b[0] > q3) / max(1, sum(1 for b in bar_data if b[0] > q3)/bars_per_day),
        },
        'current_live_range_pct': cur_rng_pct,
        'current_tier': tier,
    }
    p = ROOT / 'results' / f'regime_signals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
