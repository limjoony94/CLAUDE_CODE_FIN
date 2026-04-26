"""
Selection Variance — 신호 多, 포지션 N=1, 진입 타이밍 의존성 분석
====================================================================
[질문] 봇이 raw signal 풀에서 N=1 제약으로 일부만 잡음. 만약 봇이 다른
신호를 잡았다면 (downtime, race condition, hedge mode 등 외부 요인)
PnL이 얼마나 달라지는가? 즉 selection variance 크기는?

[방법]
1. Raw signal 측정: 모든 bar에 대해 check_entry 호출, position 무시 (N=∞)
   → 이론적 신호 풀
2. Greedy N=1 (현재 production BT): 첫 가용 신호 잡음 (baseline)
3. Variants — 신호 풀에서 다른 선택 시뮬레이션:
   a. greedy_skip_K: 매 K번째 가용 신호 skip (down/race 모방)
   b. random_skip_p%: 가용 신호의 p% 무작위 skip (1000 simulation)
4. 각 변종 PnL 분포 → variance 정량화

[가설] BT의 PnL 분포가 좁으면 selection 영향 작음.
       넓으면 LIVE 결과 ±20% 변동 가능 → "BT-LIVE 갭" 일부 설명 가능.
"""
import sys, os, json, math, yaml, random
from datetime import datetime, timezone, timedelta
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)

FEE_RT_PCT = 0.10
LEVERAGE = 3
SEED = 42
random.seed(SEED)


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close']] for _, r in df15.iterrows()]


def precompute(candles, sig):
    n = len(candles)
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows  = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    atr = compute_atr(highs, lows, closes, sig.atr_period)
    chh, chl = compute_channel(highs, lows, sig.channel_period)
    swl, swh = compute_fractal_swings(highs, lows, 10)  # fractal_lookback default
    return opens, highs, lows, closes, atr, chh, chl, swl, swh


def collect_raw_signals(candles, sig, opens, highs, lows, closes, atr, chh, chl, swl, swh, s_idx, e_idx):
    """Return list of (i, dir, sl_price) for every bar that produces a signal,
    independent of any position state."""
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


def simulate_with_filter(raw_signals, candles, sig, opens, highs, lows, closes, atr,
                         min_bars_between=2, accept_fn=None):
    """N=1 simulation with custom 'accept_fn(signal)' filter.
    accept_fn: callable returning True if signal should be entered, False to skip.
               Default: always accept.
    Returns trade list."""
    if accept_fn is None:
        accept_fn = lambda s: True

    in_pos = False
    cd = 0
    pdir = pprice = psl = pbest = None
    pheld = 0
    trades = []
    sig_iter = iter(raw_signals)
    next_sig = next(sig_iter, None)
    e_idx = len(candles) - 1

    for i in range(len(candles)):
        # Exit check
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
                trades.append({'pnl1x': round(pnl, 4), 'reason': rs, 'bars': pheld, 'i': i})
                in_pos, cd, pdir = False, i + min_bars_between, None

        # Entry: walk through raw_signals queue, accept first that meets filter and is in cooldown
        if not in_pos and i >= cd:
            # advance sig_iter past any signals before current i
            while next_sig is not None and next_sig['i'] < i:
                next_sig = next(sig_iter, None)
            # try the signal at current i
            if next_sig is not None and next_sig['i'] == i:
                if accept_fn(next_sig):
                    ni = i + 1
                    if ni < e_idx:
                        pdir = next_sig['direction']
                        pprice = opens[ni]
                        psl = next_sig['sl_price']
                        pheld = 0; in_pos = True
                        pbest = highs[ni] if pdir == 'LONG' else lows[ni]
                next_sig = next(sig_iter, None)
    return trades


def stats(trades):
    if not trades:
        return {'n': 0, 'sum1x': 0, 'wr': 0}
    sum1x = sum(t['pnl1x'] for t in trades)
    wins = sum(1 for t in trades if t['pnl1x'] > 0)
    return {'n': len(trades), 'sum1x': round(sum1x, 2),
            'wr': round(100*wins/len(trades), 1), 'avg': round(sum1x/len(trades), 4)}


def main():
    cfg_path = ROOT / 'config' / 'c1_breakout_config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    sig = C1BreakoutSignal(cfg['strategy'])
    print(f"Config: max_sl={sig.max_sl_atr}, trail_K={sig.trail_K}, prog_trail={sig.prog_trail_enabled}\n")

    csv_path = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    print(f"Loading {csv_path.name}...")
    candles = load_csv_15m(str(csv_path))
    print(f"  {len(candles)} 15m bars\n")

    # Warmup 60d, then full eval
    from datetime import datetime
    t0 = datetime.fromtimestamp(candles[0][0]/1000)
    t_warm = t0 + timedelta(days=60)
    s_idx = next(i for i, c in enumerate(candles) if c[0] >= int(t_warm.timestamp()*1000))
    e_idx = len(candles) - 1
    days_eval = (datetime.fromtimestamp(candles[e_idx][0]/1000) - t_warm).days
    print(f"Eval window: {days_eval}d ({t_warm.date()} ~ {datetime.fromtimestamp(candles[e_idx][0]/1000).date()})\n")

    opens, highs, lows, closes, atr, chh, chl, swl, swh = precompute(candles, sig)

    # 1. Raw signals
    print("Step 1: collecting raw signals (N=∞)...")
    raw = collect_raw_signals(candles, sig, opens, highs, lows, closes, atr, chh, chl, swl, swh, s_idx, e_idx)
    print(f"  Raw signals: {len(raw)} ({len(raw)/days_eval:.2f} signals/day)\n")

    # 2. Greedy N=1 baseline
    print("Step 2: greedy N=1 baseline...")
    base_trades = simulate_with_filter(raw, candles, sig, opens, highs, lows, closes, atr)
    base_st = stats(base_trades)
    print(f"  Baseline: {base_st['n']} trades ({base_st['n']/days_eval:.2f}/day) | sum1x={base_st['sum1x']:+.2f}% | WR={base_st['wr']}%")
    adopted_pct = 100 * base_st['n'] / len(raw) if raw else 0
    print(f"  Adoption rate: {adopted_pct:.1f}% of raw (N=1 conflict skipped: {len(raw)-base_st['n']})\n")

    # 3a. Skip every K-th signal (deterministic variant)
    print("Step 3a: deterministic skip variants...")
    print(f"{'variant':<28} {'n':>5} {'sum1x':>10} {'avg':>9} {'WR':>6}")
    print("-" * 70)
    for skip_k in [2, 3, 5, 10]:  # skip every K-th signal
        cnt = [0]
        def acc(s, k=skip_k, c=cnt):
            c[0] += 1
            return c[0] % k != 0
        ts = simulate_with_filter(raw, candles, sig, opens, highs, lows, closes, atr, accept_fn=acc)
        st = stats(ts)
        print(f"  greedy_skip_every_{skip_k}        {st['n']:>5} {st['sum1x']:>+9.2f}% {st['avg']:>+8.3f}% {st['wr']:>5.1f}%")

    # 3b. Random skip p% (Monte Carlo)
    print(f"\nStep 3b: random skip p% Monte Carlo (100 sims each)...")
    print(f"{'p_skip':<10} {'mean':>8} {'std':>8} {'p5':>8} {'p95':>8} {'min':>8} {'max':>8}")
    print("-" * 65)
    for p_skip in [0.05, 0.10, 0.20, 0.30, 0.50]:
        sums = []
        for trial in range(100):
            random.seed(SEED + trial)
            def acc_p(s, p=p_skip):
                return random.random() >= p
            ts = simulate_with_filter(raw, candles, sig, opens, highs, lows, closes, atr, accept_fn=acc_p)
            sums.append(sum(t['pnl1x'] for t in ts))
        mean = statistics.mean(sums)
        std = statistics.stdev(sums)
        sums_sorted = sorted(sums)
        p5 = sums_sorted[int(0.05*len(sums))]
        p95 = sums_sorted[int(0.95*len(sums))]
        print(f"  {p_skip*100:>3.0f}%      {mean:>+7.2f}% {std:>7.2f} {p5:>+7.2f}% {p95:>+7.2f}% {min(sums):>+7.2f}% {max(sums):>+7.2f}%")

    # 3c. Random downtime windows (봇 다운 시뮬레이션)
    print(f"\nStep 3c: random downtime window Monte Carlo (down for D days at random start, 100 sims)...")
    print(f"{'down_days':<12} {'mean':>8} {'std':>8} {'p5':>8} {'p95':>8} {'baseline_Δ':>11}")
    print("-" * 65)
    n_bars_per_day = 96  # 15m
    for down_days in [1, 2, 3, 7]:
        down_bars = down_days * n_bars_per_day
        sums = []
        for trial in range(100):
            random.seed(SEED + trial * 13)
            # random start within eval window
            max_start = e_idx - down_bars - 100
            if max_start <= s_idx + 100:
                continue
            ds = random.randint(s_idx + 100, max_start)
            de = ds + down_bars
            def acc_dt(s, ds=ds, de=de):
                return not (ds <= s['i'] <= de)
            ts = simulate_with_filter(raw, candles, sig, opens, highs, lows, closes, atr, accept_fn=acc_dt)
            sums.append(sum(t['pnl1x'] for t in ts))
        mean = statistics.mean(sums)
        std = statistics.stdev(sums)
        sums_sorted = sorted(sums)
        p5 = sums_sorted[int(0.05*len(sums))]
        p95 = sums_sorted[int(0.95*len(sums))]
        delta = mean - base_st['sum1x']
        print(f"  {down_days}d         {mean:>+7.2f}% {std:>7.2f} {p5:>+7.2f}% {p95:>+7.2f}% {delta:>+10.2f}pp")

    # Verdict
    print()
    print("=" * 75)
    print("VERDICT: Selection Variance")
    print("=" * 75)

    # Compute baseline-relative spread for 10% random skip
    random.seed(SEED)
    sums_10 = []
    for trial in range(100):
        random.seed(SEED + trial)
        def acc_p10(s):
            return random.random() >= 0.10
        ts = simulate_with_filter(raw, candles, sig, opens, highs, lows, closes, atr, accept_fn=acc_p10)
        sums_10.append(sum(t['pnl1x'] for t in ts))
    spread_10 = max(sums_10) - min(sums_10)
    rel_spread = 100 * spread_10 / abs(base_st['sum1x']) if base_st['sum1x'] else 0
    print(f"Baseline (greedy N=1, full uptime): {base_st['sum1x']:+.2f}% (1x)")
    print(f"10% random skip 100-sim spread    : {min(sums_10):+.2f}% ~ {max(sums_10):+.2f}% (range {spread_10:.2f}pp)")
    print(f"Relative spread vs baseline       : {rel_spread:.1f}%")
    print()
    if rel_spread > 30:
        print("⚠️  Selection variance LARGE (>30% of baseline) — LIVE outcomes can deviate substantially")
    elif rel_spread > 15:
        print("⚠️  Selection variance MODERATE (15~30%) — some LIVE-BT divergence expected")
    else:
        print("✅ Selection variance SMALL (<15%) — strategy robust to entry-timing perturbation")

    # Save
    out = {
        'date': datetime.now().isoformat(),
        'days_eval': days_eval,
        'raw_signals': len(raw),
        'baseline': base_st,
        'adoption_pct': round(adopted_pct, 1),
        'random_skip_10p_sims': sums_10,
        'spread_10p_pp': round(spread_10, 2),
        'rel_spread_pct_of_base': round(rel_spread, 1),
    }
    p = ROOT / 'results' / f'selection_variance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
