"""
Pre-registered eval window (30 trades) selection variance 측정
================================================================
질문: -5pp threshold이 selection noise와 구분 가능한가?

방법:
1. 272d 전체에서 random rolling 30-trade windows 추출
2. 각 window에서 baseline + random skip p=10/20% 시뮬레이션
3. 30 trades 누적 PnL 분포 → 95% CI vs -5pp 임계 비교
"""
import sys, json, math, yaml, random, statistics
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)

FEE_RT_PCT = 0.10
SEED = 42


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close']] for _, r in df15.iterrows()]


def precompute(candles, sig):
    opens = [c[1] for c in candles]; highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]; closes = [c[4] for c in candles]
    atr = compute_atr(highs, lows, closes, sig.atr_period)
    chh, chl = compute_channel(highs, lows, sig.channel_period)
    swl, swh = compute_fractal_swings(highs, lows, 10)
    return opens, highs, lows, closes, atr, chh, chl, swl, swh


def collect_raw_signals(candles, sig, opens, highs, lows, closes, atr, chh, chl, swl, swh, s_idx, e_idx):
    signals = []
    for i in range(s_idx, e_idx):
        if math.isnan(atr[i]) or math.isnan(chh[i]):
            continue
        es = sig.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                             bar_close=closes[i], channel_high=chh[i], channel_low=chl[i],
                             atr_val=atr[i], last_swing_low=swl[i], last_swing_high=swh[i])
        if es:
            signals.append({'i': i, 'direction': es['direction'], 'sl_price': es['sl_price']})
    return signals


def simulate_n1(raw_signals, candles, sig, opens, highs, lows, closes, atr, accept_fn=None,
                start_i=0, max_trades=None):
    if accept_fn is None:
        accept_fn = lambda s: True
    in_pos = False; cd = start_i
    pdir = pprice = psl = pbest = None; pheld = 0
    trades = []
    sig_iter = iter([s for s in raw_signals if s['i'] >= start_i])
    next_sig = next(sig_iter, None)
    e_idx = len(candles) - 1
    for i in range(start_i, len(candles)):
        if max_trades and len(trades) >= max_trades:
            break
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])
            er = sig.check_exit(direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl, atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld)
            if er:
                xp, rs = er['exit_price'], er['reason']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= FEE_RT_PCT
                trades.append({'pnl1x': round(pnl, 4), 'reason': rs, 'i': i})
                in_pos, cd, pdir = False, i + 2, None
        if not in_pos and i >= cd:
            while next_sig is not None and next_sig['i'] < i:
                next_sig = next(sig_iter, None)
            if next_sig is not None and next_sig['i'] == i:
                if accept_fn(next_sig):
                    ni = i + 1
                    if ni < e_idx:
                        pdir = next_sig['direction']; pprice = opens[ni]; psl = next_sig['sl_price']
                        pheld = 0; in_pos = True
                        pbest = highs[ni] if pdir == 'LONG' else lows[ni]
                next_sig = next(sig_iter, None)
    return trades


def main():
    cfg = yaml.safe_load(open(ROOT / 'config' / 'c1_breakout_config.yaml'))
    sig = C1BreakoutSignal(cfg['strategy'])
    print(f"Config: max_sl={sig.max_sl_atr}, trail_K={sig.trail_K}, prog_trail={sig.prog_trail_enabled}\n")

    candles = load_csv_15m(str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv'))
    print(f"{len(candles)} 15m bars\n")

    t0 = datetime.fromtimestamp(candles[0][0]/1000)
    s_idx = next(i for i, c in enumerate(candles) if c[0] >= int((t0 + timedelta(days=60)).timestamp()*1000))
    e_idx = len(candles) - 1

    opens, highs, lows, closes, atr, chh, chl, swl, swh = precompute(candles, sig)

    raw = collect_raw_signals(candles, sig, opens, highs, lows, closes, atr, chh, chl, swl, swh, s_idx, e_idx)
    print(f"Raw signals: {len(raw)}\n")

    # Step 1: Run greedy N=1 to get baseline trades with their start indices
    print("Step 1: Run greedy N=1 to collect 30-trade windows...")
    base_trades = simulate_n1(raw, candles, sig, opens, highs, lows, closes, atr, start_i=s_idx)
    print(f"Total greedy trades: {len(base_trades)}\n")

    if len(base_trades) < 100:
        print("Not enough trades for window analysis"); return

    # Step 2: Rolling 30-trade window — measure PnL distribution
    WINDOW = 30
    print(f"Step 2: Rolling {WINDOW}-trade windows in greedy baseline...")
    window_pnls = []
    for k in range(len(base_trades) - WINDOW + 1):
        w = base_trades[k:k + WINDOW]
        window_pnls.append(sum(t['pnl1x'] for t in w))
    print(f"  N windows: {len(window_pnls)}")
    print(f"  Mean: {statistics.mean(window_pnls):+.2f}%")
    print(f"  Std:  {statistics.stdev(window_pnls):.2f}")
    sorted_p = sorted(window_pnls)
    print(f"  P5:   {sorted_p[int(0.05*len(sorted_p))]:+.2f}%")
    print(f"  P25:  {sorted_p[int(0.25*len(sorted_p))]:+.2f}%")
    print(f"  P50:  {sorted_p[int(0.50*len(sorted_p))]:+.2f}%")
    print(f"  P75:  {sorted_p[int(0.75*len(sorted_p))]:+.2f}%")
    print(f"  P95:  {sorted_p[int(0.95*len(sorted_p))]:+.2f}%")
    print(f"  Min:  {min(window_pnls):+.2f}%")
    print(f"  Max:  {max(window_pnls):+.2f}%")

    # Step 3: Selection-variance for fixed start index — random skip Monte Carlo
    print(f"\nStep 3: Selection variance within fixed window — random skip MC")
    print(f"  Method: pick random start index, simulate {WINDOW} trades with skip p%, measure PnL")
    print(f"{'p_skip':<8} {'mean':>8} {'std':>8} {'p5':>8} {'p95':>8} {'spread':>8} {'<-5pp':>7}")
    print("-" * 70)

    # Sample 200 random start indices
    sample_starts = []
    random.seed(SEED)
    for _ in range(50):  # 50 different start points
        start_i_rand = random.randint(s_idx, e_idx - 5000)  # ensure room for 30 trades
        sample_starts.append(start_i_rand)

    for p_skip in [0.0, 0.05, 0.10, 0.20, 0.30]:
        all_pnls = []
        for start in sample_starts:
            for trial in range(20):
                random.seed(SEED + start + trial * 1000)
                if p_skip == 0:
                    accept = lambda s: True
                else:
                    p = p_skip
                    accept = lambda s, p=p: random.random() >= p
                trades = simulate_n1(raw, candles, sig, opens, highs, lows, closes, atr,
                                     accept_fn=accept, start_i=start, max_trades=WINDOW)
                if len(trades) >= WINDOW:
                    all_pnls.append(sum(t['pnl1x'] for t in trades))
        if not all_pnls: continue
        mean_p = statistics.mean(all_pnls)
        std_p = statistics.stdev(all_pnls) if len(all_pnls) > 1 else 0
        sorted_a = sorted(all_pnls)
        p5_v = sorted_a[int(0.05 * len(sorted_a))]
        p95_v = sorted_a[int(0.95 * len(sorted_a))]
        spread = p95_v - p5_v
        n_below = sum(1 for x in all_pnls if x < -5)
        pct_below = 100 * n_below / len(all_pnls)
        print(f"  {p_skip*100:>3.0f}%   {mean_p:>+7.2f}% {std_p:>7.2f} {p5_v:>+7.2f}% {p95_v:>+7.2f}% {spread:>7.2f}pp {pct_below:>5.1f}%")

    # Verdict
    print()
    print("=" * 75)
    print("VERDICT — -5pp threshold calibration")
    print("=" * 75)
    p10_pnls = []
    for start in sample_starts:
        for trial in range(20):
            random.seed(SEED + start + trial * 1000)
            accept = lambda s: random.random() >= 0.10
            trades = simulate_n1(raw, candles, sig, opens, highs, lows, closes, atr,
                                 accept_fn=accept, start_i=start, max_trades=WINDOW)
            if len(trades) >= WINDOW:
                p10_pnls.append(sum(t['pnl1x'] for t in trades))
    n_p10_below = sum(1 for x in p10_pnls if x < -5)
    pct_p10_below = 100 * n_p10_below / len(p10_pnls) if p10_pnls else 0
    print(f"At p_skip=10% (LIVE-realistic uptime/competing): {pct_p10_below:.1f}% of windows fall below -5pp")
    print(f"This is the 'false positive rate' — if F v2 + max_sl 4.5 truly produced same dist as baseline,")
    print(f"there's still a {pct_p10_below:.1f}% chance of triggering revert by selection noise alone.")

    # Save
    out = {
        'date': datetime.now().isoformat(),
        'window_size': WINDOW,
        'rolling_baseline_pnl': {
            'n': len(window_pnls),
            'mean': statistics.mean(window_pnls),
            'std': statistics.stdev(window_pnls),
            'p5': sorted_p[int(0.05*len(sorted_p))],
            'p95': sorted_p[int(0.95*len(sorted_p))],
            'min': min(window_pnls),
            'max': max(window_pnls),
        },
        'p10_skip_below_5pp_pct': pct_p10_below,
    }
    p = ROOT / 'results' / f'eval_window_variance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
