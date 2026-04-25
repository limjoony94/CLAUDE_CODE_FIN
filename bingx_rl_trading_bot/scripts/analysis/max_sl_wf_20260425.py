"""
max_sl_atr Walk-Forward 검증 (5-fold expanding) — 2026-04-25
==============================================================
max_sl 3.3 (legacy) vs 4.5 (quick win) robustness 검증.

5-fold expanding window:
- Fold k: train [day 0, day F_k_train_end], test [day F_k_test_start, day F_k_test_end]
- 333d 분할 → 각 test window ~30d (간략화: 5 contiguous test windows)

각 fold에서 max_sl 3.3 vs 4.5 BT 비교.
GO 조건: 5/5 fold에서 4.5 PnL ≥ 3.3 + 5pp (1x).
"""
import sys, os, json, math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

BASE_CFG = {
    'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
    'trail_K': 2.5, 'max_sl_atr': 3.3,  # swept
    'emergency_sl_pct': 3.0, 'max_hold_bars': 192,
    'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
    'min_bars_between': 2, 'trail_activation_pct': 0.05,
    'fractal_lookback': 10,
    'progressive_trail': {'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5},
}

FEE_RT_PCT = 0.10
LEVERAGE = 3


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close']] for _, r in df15.iterrows()]


def run_bt(candles, max_sl, eval_start, eval_end):
    cfg = dict(BASE_CFG); cfg['max_sl_atr'] = max_sl
    signal = C1BreakoutSignal(cfg)
    ts = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).replace(tzinfo=None) for c in candles]
    opens, highs = [c[1] for c in candles], [c[2] for c in candles]
    lows, closes = [c[3] for c in candles], [c[4] for c in candles]
    n = len(closes)
    atr = compute_atr(highs, lows, closes, cfg['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, cfg['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, cfg['fractal_lookback'])

    s_idx = next((i for i, t in enumerate(ts) if t >= eval_start), None)
    if s_idx is None: return []
    e_idx = next((i for i in range(n-1, -1, -1) if ts[i] < eval_end), n-1)
    trades, in_pos, cd = [], False, 0
    pdir = pprice = psl = pbest = None; pheld = 0

    for i in range(s_idx, e_idx + 1):
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])
            er = signal.check_exit(direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl, atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld)
            if er:
                xp, rs = er['exit_price'], er['reason']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= FEE_RT_PCT
                trades.append({'pnl1x':round(pnl,4), 'pnl3x':round(pnl*LEVERAGE,4), 'reason':rs})
                in_pos, cd, pdir = False, i + cfg['min_bars_between'], None
        if not in_pos and i >= cd and i < e_idx:
            if math.isnan(atr[i]) or math.isnan(ch_h[i]): continue
            es = signal.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                bar_close=closes[i], channel_high=ch_h[i], channel_low=ch_l[i],
                atr_val=atr[i], last_swing_low=sw_l[i], last_swing_high=sw_h[i])
            if es:
                ni = i + 1
                if ni > e_idx: continue
                pdir = es['direction']; pprice = opens[ni]; psl = es['sl_price']
                pheld = 0; in_pos = True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]
    return trades


def stats(trades, start=100.0):
    if not trades: return {'n':0, 'sum1x':0, 'sum3x':0, 'mdd':0, 'wr':0, 'sl_pct':0}
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl3x']>0)
    sum1x = sum(t['pnl1x'] for t in trades); sum3x = sum(t['pnl3x'] for t in trades)
    bal = start; peak = start; mdd = 0
    for t in trades:
        bal *= (1 + t['pnl3x']/100)
        if bal>peak: peak=bal
        dd = (bal-peak)/peak*100
        if dd<mdd: mdd=dd
    sls = sum(1 for t in trades if t['reason']=='SL')
    return {'n':n, 'sum1x':round(sum1x,2), 'sum3x':round(sum3x,2),
            'mdd':round(mdd,2), 'wr':round(100*wins/n,1), 'sl_pct':round(100*sls/n,1)}


def main():
    csv_path = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    if not csv_path.exists():
        print(f"ERR: {csv_path} not found"); return
    print(f"Loading {csv_path.name}...")
    candles = load_csv_15m(str(csv_path))
    print(f"Got {len(candles)} 15m candles")

    t_start = datetime.fromtimestamp(candles[0][0]/1000)
    t_end = datetime.fromtimestamp(candles[-1][0]/1000)
    total_days = (t_end - t_start).days
    print(f"Range: {t_start.date()} ~ {t_end.date()} ({total_days}d)\n")

    # 5-fold expanding window: warmup 60d, then 5 test windows of ~equal length
    from datetime import timedelta
    warmup_days = 60
    eval_start = t_start + timedelta(days=warmup_days)
    eval_total = (t_end - eval_start).days
    fold_days = eval_total // 5

    print(f"Warmup: {warmup_days}d, eval total: {eval_total}d, per-fold: {fold_days}d\n")
    print("=" * 95)
    print(f"{'fold':>4} {'period':<25} {'max_sl 3.3 sum1x':>17} {'max_sl 4.5 sum1x':>17} {'delta':>9} {'4.5_SL%':>8}")
    print("=" * 95)

    fold_results = []
    for k in range(5):
        fold_start = eval_start + timedelta(days=k * fold_days)
        fold_end = eval_start + timedelta(days=(k+1) * fold_days) if k < 4 else t_end

        t_33 = run_bt(candles, 3.3, fold_start, fold_end)
        t_45 = run_bt(candles, 4.5, fold_start, fold_end)
        s33 = stats(t_33); s45 = stats(t_45)
        delta = round(s45['sum1x'] - s33['sum1x'], 2)
        period = f"{fold_start.date()}~{fold_end.date()}"
        print(f"{k+1:>4} {period:<25} {s33['sum1x']:>+15.2f}% {s45['sum1x']:>+15.2f}% {delta:>+8.2f}% {s45['sl_pct']:>7.1f}%")
        fold_results.append({'fold':k+1, 'period':period,
            's33':s33, 's45':s45, 'delta':delta})

    # Summary
    print()
    n_pass = sum(1 for f in fold_results if f['delta'] >= 5.0)
    n_pos = sum(1 for f in fold_results if f['delta'] > 0)
    n_neg = sum(1 for f in fold_results if f['delta'] < 0)
    avg_delta = sum(f['delta'] for f in fold_results) / 5

    print(f"Folds with 4.5 ≥ 3.3 + 5pp:  {n_pass}/5")
    print(f"Folds with 4.5 > 3.3:        {n_pos}/5")
    print(f"Folds with 4.5 < 3.3:        {n_neg}/5")
    print(f"Avg delta (1x):              {avg_delta:+.2f}pp")

    print(f"\nVerdict:")
    if n_pass >= 4:
        print(f"  ✅ Robust ({n_pass}/5 GO) — max_sl 4.5 일관 우월")
    elif n_pos >= 4:
        print(f"  ⚠️  Positive ({n_pos}/5 > 0) but not all ≥ +5pp — moderate robust")
    else:
        print(f"  ❌ Inconsistent — fold variance 큼, overfitting 의심")

    out = {'date':datetime.now().isoformat(), 'folds':fold_results,
           'n_pass':n_pass, 'n_pos':n_pos, 'avg_delta':avg_delta}
    path = ROOT / 'results' / f'max_sl_wf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path,'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
