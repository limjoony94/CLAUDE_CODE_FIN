"""
Diagnostic 3: 272d BT rolling 14d windows 분포 — LIVE -12.86% percentile
=========================================================================
질문: LIVE 14d -12.86%가 BT 분포의 어느 위치인가?
방법:
  1. 272d production BT 실행 (max_sl 4.5, progressive_trail enabled)
  2. 매 14d window 누적 PnL 1x 계산 (rolling)
  3. LIVE -12.86% percentile 측정
판정:
  P5 미만 → unprecedented regime, BT generalization 실패
  P5~P25 → bad luck regime, 일부 변경 필요할 수 있음
  P25 이상 → 정상 분포 내, 다른 원인
"""
import sys, json, math, yaml
from datetime import datetime, timedelta
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import compute_atr, compute_channel, compute_fractal_swings


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close']] for _, r in df15.iterrows()]


def run_bt_with_times(candles, sig, s_idx, e_idx):
    """Returns trades with exit timestamps."""
    opens = [c[1] for c in candles]; highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]; closes = [c[4] for c in candles]
    times = [c[0] for c in candles]
    atr = compute_atr(highs, lows, closes, sig.atr_period)
    chh, chl = compute_channel(highs, lows, sig.channel_period)
    swl, swh = compute_fractal_swings(highs, lows, 10)
    in_pos = False; cd = 0
    pdir = pprice = psl = pbest = None; pheld = 0
    trades = []
    for i in range(s_idx, e_idx + 1):
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])
            er = sig.check_exit(direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl, atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld)
            if er:
                xp = er['exit_price']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= 0.10
                trades.append({'pnl1x': round(pnl, 4), 'reason': er['reason'],
                               'exit_ts': times[i]})
                in_pos, cd, pdir = False, i + sig.min_bars_between_trades, None
        if not in_pos and i >= cd and i < e_idx:
            if math.isnan(atr[i]) or math.isnan(chh[i]): continue
            es = sig.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                bar_close=closes[i], channel_high=chh[i], channel_low=chl[i],
                atr_val=atr[i], last_swing_low=swl[i], last_swing_high=swh[i])
            if es:
                ni = i + 1
                if ni > e_idx: continue
                pdir = es['direction']; pprice = opens[ni]; psl = es['sl_price']
                pheld = 0; in_pos = True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]
    return trades


def main():
    cfg = yaml.safe_load(open(ROOT / 'config' / 'c1_breakout_config.yaml'))
    sig = C1BreakoutSignal(cfg['strategy'])
    candles = load_csv_15m(str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv'))
    print(f"{len(candles)} bars")

    t0 = datetime.fromtimestamp(candles[0][0]/1000)
    s_idx = next(i for i, c in enumerate(candles) if c[0] >= int((t0+timedelta(days=60)).timestamp()*1000))
    e_idx = len(candles) - 1
    days = (datetime.fromtimestamp(candles[e_idx][0]/1000) - (t0 + timedelta(days=60))).days
    print(f"Eval: {days}d\n")

    print("Running 272d BT (production config)...")
    trades = run_bt_with_times(candles, sig, s_idx, e_idx)
    print(f"  {len(trades)} trades, sum={sum(t['pnl1x'] for t in trades):+.2f}% (1x)\n")

    # Rolling 14d windows: starting from each trade exit, window = next 14 days
    WINDOW_DAYS = 14
    WINDOW_MS = WINDOW_DAYS * 24 * 60 * 60 * 1000

    window_sums = []  # (start_ts, num_trades, sum_pnl1x)
    for k, t in enumerate(trades):
        end_ts = t['exit_ts'] + WINDOW_MS
        # Trades whose exit_ts ∈ [t['exit_ts'], end_ts]
        sub = [tt for tt in trades[k:] if tt['exit_ts'] <= end_ts]
        if len(sub) >= 5:  # min 5 trades for sensible window
            sum_pnl = sum(tt['pnl1x'] for tt in sub)
            window_sums.append({'start_ts': t['exit_ts'], 'n': len(sub), 'sum_pnl1x': round(sum_pnl, 4)})

    print(f"Rolling 14d windows: {len(window_sums)} (≥5 trades each)")
    if not window_sums:
        print("Insufficient data"); return

    pnls = [w['sum_pnl1x'] for w in window_sums]
    sorted_p = sorted(pnls)
    n = len(pnls)
    LIVE_PNL = -12.86

    print(f"\n{'metric':<10} {'value':>10}")
    print("-" * 30)
    print(f"{'mean':<10} {statistics.mean(pnls):>+9.2f}%")
    print(f"{'std':<10} {statistics.stdev(pnls):>9.2f}")
    for p in (5, 10, 25, 50, 75, 90, 95):
        v = sorted_p[int(p/100*n)]
        print(f"{'P'+str(p):<10} {v:>+9.2f}%")
    print(f"{'min':<10} {min(pnls):>+9.2f}%")
    print(f"{'max':<10} {max(pnls):>+9.2f}%")

    # LIVE percentile
    n_below_live = sum(1 for p in pnls if p < LIVE_PNL)
    pct_below = 100 * n_below_live / n
    print(f"\nLIVE 14d PnL 1x: {LIVE_PNL:+.2f}%")
    print(f"BT windows below LIVE: {n_below_live}/{n} ({pct_below:.2f}%)")

    print("\n=== Verdict ===")
    if pct_below < 1:
        print("🚨 LIVE -12.86% < BT 분포 P1 — UNPRECEDENTED.")
        print("    BT generalization 실패. 다른 원인 (slippage, regime, BT bug) 강력 시사.")
    elif pct_below < 5:
        print("⚠️  LIVE < BT P5 — Severe outlier.")
        print("    Bad luck or 약한 BT-LIVE divergence. 분석 필요.")
    elif pct_below < 25:
        print("⚠️  LIVE < BT P25 — Bad regime within BT distribution.")
        print("    BT 분포 내이지만 lower tail. Variance + selection 일부 설명 가능.")
    else:
        print(f"✅ LIVE within BT P{int(pct_below)} — Normal regime.")
        print("    BT 분포 내 정상. 다른 dominant 원인 (slippage, etc) 우세.")

    # Also check positive rate
    n_pos = sum(1 for p in pnls if p > 0)
    print(f"\nReference — BT windows positive: {n_pos}/{n} ({100*n_pos/n:.1f}%)")
    print(f"BT windows < 0%: {n - n_pos}/{n} ({100*(n-n_pos)/n:.1f}%)")
    print(f"BT windows < -5%: {sum(1 for p in pnls if p < -5)}/{n}")
    print(f"BT windows < -12.86%: {n_below_live}/{n}")

    out = {
        'date': datetime.now().isoformat(),
        'window_days': WINDOW_DAYS,
        'n_windows': n,
        'live_pnl': LIVE_PNL,
        'percentile_below_live': pct_below,
        'distribution': {
            'mean': statistics.mean(pnls),
            'std': statistics.stdev(pnls),
            'min': min(pnls), 'max': max(pnls),
            'p5': sorted_p[int(0.05*n)],
            'p25': sorted_p[int(0.25*n)],
            'p50': sorted_p[int(0.50*n)],
            'p75': sorted_p[int(0.75*n)],
            'p95': sorted_p[int(0.95*n)],
        },
        'pos_rate_bt': 100 * n_pos / n,
    }
    p = ROOT / 'results' / f'diag3_rolling_14d_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
