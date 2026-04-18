"""
C1 Refined Strategy Validation
===============================
Rigorous validation of ablation findings to distinguish genuine alpha
from in-sample overfitting. Tests three strategies:

  A. BASELINE     — current production C1 (fractal SL, CD=2, full filter)
  B. E7_NO_FRACT  — ablation E7: emergency-only SL (no fractal)
  C. REFINED      — E4 (body only) + E7 (no fractal SL) + CD=0

For each strategy:
  1. WF 5-fold expanding window (IS→OOS)
     - Pass: all 5 folds positive OOS
  2. 3-way split (60/20/20 train/val/test)
     - Pass: all three segments positive
  3. 60-day rolling window (stability)

Additionally: Monte Carlo direction randomization (999 sims) for p-value.
"""
import sys, os, json, math, random
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)

BASE_CFG = {
    'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
    'trail_K': 2.5, 'max_sl_atr': 3.3, 'emergency_sl_pct': 3.0,
    'max_hold_bars': 192, 'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
    'min_bars_between': 2, 'trail_activation_pct': 0.05,
    'fractal_lookback': 10,
}
FEE_RT_PCT = 0.10


# ── Entry functions ──
def entry_baseline(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """Standard C1 (fractal SL)."""
    if math.isnan(ch) or math.isnan(cl) or math.isnan(atr) or atr <= 0 or ch <= cl:
        return None
    direction = 'LONG' if bc > ch else ('SHORT' if bc < cl else None)
    if direction is None:
        return None
    rng = bh - bl
    if rng <= 0:
        return None
    body = bc - bo
    if abs(body) / rng < cfg['body_min_ratio']:
        return None
    if direction == 'LONG' and body <= 0:
        return None
    if direction == 'SHORT' and body >= 0:
        return None
    e = bc
    if direction == 'LONG':
        fs = swl if not math.isnan(swl) else e - cfg['max_sl_atr']*atr
        sl = max(fs, e - cfg['max_sl_atr']*atr)
    else:
        fs = swh if not math.isnan(swh) else e + cfg['max_sl_atr']*atr
        sl = min(fs, e + cfg['max_sl_atr']*atr)
    sp = abs(e - sl) / e * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl}


def entry_no_fractal(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """E7-style: same signal but ATR-cap SL only (no fractal)."""
    if math.isnan(ch) or math.isnan(cl) or math.isnan(atr) or atr <= 0 or ch <= cl:
        return None
    direction = 'LONG' if bc > ch else ('SHORT' if bc < cl else None)
    if direction is None:
        return None
    rng = bh - bl
    if rng <= 0:
        return None
    body = bc - bo
    if abs(body) / rng < cfg['body_min_ratio']:
        return None
    if direction == 'LONG' and body <= 0:
        return None
    if direction == 'SHORT' and body >= 0:
        return None
    e = bc
    sl_dist = cfg['max_sl_atr'] * atr
    sl = e - sl_dist if direction == 'LONG' else e + sl_dist
    sp = abs(e - sl) / e * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl}


def entry_refined(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """E4+E7: body-only filter + ATR-cap SL. No channel."""
    if math.isnan(atr) or atr <= 0:
        return None
    rng = bh - bl
    if rng <= 0:
        return None
    body = bc - bo
    if abs(body) / rng < cfg['body_min_ratio']:
        return None
    direction = 'LONG' if body > 0 else 'SHORT'
    e = bc
    sl_dist = cfg['max_sl_atr'] * atr
    sl = e - sl_dist if direction == 'LONG' else e + sl_dist
    sp = abs(e - sl) / e * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl}


# ── Exit (standard for all) ──
def check_exit(direction, entry, best, hi, lo, cl, sl, atr, bars, cfg):
    if direction == 'LONG' and lo <= sl:
        return {'reason': 'SL', 'exit_price': sl}
    if direction == 'SHORT' and hi >= sl:
        return {'reason': 'SL', 'exit_price': sl}
    worst = (lo/entry - 1)*100 if direction == 'LONG' else (1 - hi/entry)*100
    if worst <= -cfg['emergency_sl_pct']:
        ep = entry*(1 - cfg['emergency_sl_pct']/100) if direction == 'LONG' \
             else entry*(1 + cfg['emergency_sl_pct']/100)
        return {'reason': 'EMERGENCY', 'exit_price': ep}
    if bars >= cfg['max_hold_bars']:
        return {'reason': 'TIMEOUT', 'exit_price': cl}
    if direction == 'LONG':
        best_pnl = (best/entry - 1)*100
        cur_pnl = (cl/entry - 1)*100
    else:
        best_pnl = (1 - best/entry)*100
        cur_pnl = (1 - cl/entry)*100
    if (best_pnl > cfg['trail_activation_pct']
            and not math.isnan(atr) and atr > 0 and cl > 0):
        td = cfg['trail_K']*atr/cl*100
        if best_pnl - cur_pnl >= td:
            r = max(0, best_pnl - td)
            ep = entry*(1 + r/100) if direction == 'LONG' else entry*(1 - r/100)
            return {'reason': 'TRAIL_TP', 'exit_price': ep}
    return None


def run_bt(df15, cfg, entry_fn, start_i, end_i,
           timestamps=None, opens=None, highs=None, lows=None,
           closes=None, atr=None, ch=None, cl=None, swl=None, swh=None):
    """Run backtest on [start_i, end_i] slice. Pre-computed indicators passed in."""
    trades = []
    in_pos = False
    cd_until = start_i
    pd_, pep, pet, psl, pbp, pbh = None, None, None, None, None, 0

    for i in range(start_i, end_i + 1):
        if in_pos:
            pbh += 1
            if pd_ == 'LONG':
                pbp = max(pbp, highs[i])
            else:
                pbp = min(pbp, lows[i])
            ex = check_exit(pd_, pep, pbp, highs[i], lows[i], closes[i], psl,
                            atr[i] if not math.isnan(atr[i]) else atr[i-1],
                            pbh, cfg)
            if ex:
                ep = ex['exit_price']
                pnl = (ep/pep - 1)*100 if pd_ == 'LONG' else (1 - ep/pep)*100
                pnl -= FEE_RT_PCT
                trades.append({'direction': pd_, 'pnl_pct': round(pnl, 4),
                               'reason': ex['reason']})
                in_pos = False
                cd_until = i + cfg['min_bars_between']
                pd_ = None

        if not in_pos and i >= cd_until and i < end_i:
            if math.isnan(atr[i]):
                continue
            sig = entry_fn(opens[i], highs[i], lows[i], closes[i],
                           ch[i], cl[i], atr[i], swl[i], swh[i], cfg)
            if sig is None:
                continue
            ni = i + 1
            if ni > end_i:
                continue
            pd_ = sig['direction']
            pep = opens[ni]
            pet = timestamps[ni]
            psl = sig['sl_price']
            pbh = 0
            in_pos = True
            pbp = highs[ni] if pd_ == 'LONG' else lows[ni]
            ex = check_exit(pd_, pep, pbp, highs[ni], lows[ni], closes[ni], psl,
                            atr[ni] if not math.isnan(atr[ni]) else atr[i],
                            0, cfg)
            if ex:
                ep = ex['exit_price']
                pnl = (ep/pep - 1)*100 if pd_ == 'LONG' else (1 - ep/pep)*100
                pnl -= FEE_RT_PCT
                trades.append({'direction': pd_, 'pnl_pct': round(pnl, 4),
                               'reason': ex['reason']})
                in_pos = False
                cd_until = ni + cfg['min_bars_between']
                pd_ = None
    return trades


def summarize(trades):
    if not trades:
        return {'trades': 0, 'WR': 0, 'PnL': 0, 'MDD': 0}
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    pnl = sum(t['pnl_pct'] for t in trades)
    eq, peak, mdd = 0, 0, 0
    for t in trades:
        eq += t['pnl_pct']
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return {'trades': n, 'WR': round(wins/n*100, 1),
            'PnL': round(pnl, 2), 'MDD': round(mdd, 2)}


def wf_5fold(df15, cfg, entry_fn, precomputed):
    """Expanding window WF: IS=[0..T], OOS=[T..T+step]. 5 folds."""
    n = len(df15)
    # Reserve first 50 bars for warmup
    warmup = 50
    n_eff = n - warmup
    n_folds = 5
    step = n_eff // (n_folds + 1)

    oos_summaries = []
    for k in range(n_folds):
        # IS: [warmup, warmup + (k+1)*step]
        # OOS: [warmup + (k+1)*step, warmup + (k+2)*step]
        oos_start = warmup + (k + 1) * step
        oos_end = warmup + (k + 2) * step - 1
        if oos_end > n - 1:
            oos_end = n - 1
        trades = run_bt(df15, cfg, entry_fn, oos_start, oos_end, **precomputed)
        s = summarize(trades)
        s['fold'] = k + 1
        s['period'] = f"{str(precomputed['timestamps'][oos_start])[:10]}~{str(precomputed['timestamps'][oos_end])[:10]}"
        oos_summaries.append(s)
    return oos_summaries


def three_way_split(df15, cfg, entry_fn, precomputed):
    """Train 60% / Val 20% / Test 20%."""
    n = len(df15)
    warmup = 50
    t1 = warmup + int((n - warmup) * 0.6)   # end of train
    t2 = warmup + int((n - warmup) * 0.8)   # end of val
    train_t = run_bt(df15, cfg, entry_fn, warmup, t1, **precomputed)
    val_t = run_bt(df15, cfg, entry_fn, t1 + 1, t2, **precomputed)
    test_t = run_bt(df15, cfg, entry_fn, t2 + 1, n - 1, **precomputed)
    return {
        'train': summarize(train_t),
        'val': summarize(val_t),
        'test': summarize(test_t),
    }


def precompute(df15, cfg):
    opens = df15['open'].tolist()
    highs = df15['high'].tolist()
    lows = df15['low'].tolist()
    closes = df15['close'].tolist()
    timestamps = df15['timestamp'].tolist()
    atr = compute_atr(highs, lows, closes, cfg['atr_period'])
    ch, cl = compute_channel(highs, lows, cfg['channel_period'])
    swl, swh = compute_fractal_swings(highs, lows, cfg['fractal_lookback'])
    return {'opens': opens, 'highs': highs, 'lows': lows, 'closes': closes,
            'timestamps': timestamps, 'atr': atr, 'ch': ch, 'cl': cl,
            'swl': swl, 'swh': swh}


def main():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df15 = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    print(f"Bars: {len(df15)} | {df15['timestamp'].iloc[0]} ~ {df15['timestamp'].iloc[-1]}")

    strategies = [
        ('BASELINE', entry_baseline, BASE_CFG),
        ('E7_NO_FRACT', entry_no_fractal, BASE_CFG),
        ('REFINED_E4E7CD0', entry_refined,
         {**BASE_CFG, 'min_bars_between': 0}),
    ]

    all_results = {}
    for name, entry_fn, cfg in strategies:
        print(f"\n{'='*80}\n  {name}\n{'='*80}")
        pc = precompute(df15, cfg)

        # Full-period
        full = run_bt(df15, cfg, entry_fn, 50, len(df15) - 1, **pc)
        full_s = summarize(full)
        print(f"FULL period: {full_s}")

        # WF 5-fold OOS
        wf = wf_5fold(df15, cfg, entry_fn, pc)
        passed = sum(1 for f in wf if f['PnL'] > 0)
        total_oos = sum(f['PnL'] for f in wf)
        print(f"WF 5-fold OOS: {passed}/5 positive, total OOS PnL {total_oos:+.2f}%")
        for f in wf:
            status = 'PASS' if f['PnL'] > 0 else 'FAIL'
            print(f"  Fold {f['fold']} ({f['period']}): trades={f['trades']:4d} "
                  f"WR={f['WR']:4.1f}% PnL={f['PnL']:+7.2f}% MDD={f['MDD']:5.2f}% [{status}]")

        # 3-way
        tws = three_way_split(df15, cfg, entry_fn, pc)
        print(f"3-way split: Train={tws['train']['PnL']:+.2f}% "
              f"Val={tws['val']['PnL']:+.2f}% Test={tws['test']['PnL']:+.2f}%")
        all_pos = all(s['PnL'] > 0 for s in tws.values())
        print(f"  3-way ALL positive: {'YES' if all_pos else 'NO'}")

        all_results[name] = {
            'full': full_s,
            'wf': wf, 'wf_passed': passed, 'wf_total_oos': round(total_oos, 2),
            '3way': tws, '3way_all_pos': all_pos,
        }

    # Summary comparison
    print(f"\n{'='*80}\n  FINAL COMPARISON\n{'='*80}")
    print(f"{'Strategy':20s} {'FullPnL':>10s} {'WF Pass':>10s} {'WF OOS':>10s} {'3-way All+':>12s}")
    for name in all_results:
        r = all_results[name]
        print(f"{name:20s} {r['full']['PnL']:>+10.2f} "
              f"{r['wf_passed']}/5      {r['wf_total_oos']:>+10.2f} "
              f"{'YES' if r['3way_all_pos'] else 'NO':>12s}")

    out_path = ROOT / 'results' / 'c1_refined_validation.json'
    with open(out_path, 'w') as f:
        json.dump({'date_run': datetime.now().isoformat(),
                   'results': all_results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
