"""M3-R32 — R21 entry concept NATIVE on 15m + C1 production exit.

R31 had timeframe mismatch (5m entry × 15m-tuned exit). Now native 15m.

R21 concept: Pattern reversal (bullish/bearish engulfing or hammer/shooting star)
at recent extreme (20-bar swing). Native 15m means natural ATR scale, structural
SL distances align with production sl_min/max bounds.

Plus C1 production exit (progressive_trail, SL bounds, trail_activation).
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round29_c1_exact_revalidation import prepare_15m_data
from m1_bt_framework import compute_atr as compute_atr_list
from m2_round1_screening import compute_ema, load_ohlcv, merge_htf


def add_15m_extras(df):
    """Add SMA200 1h trend filter (from prepare_15m_data we have base)."""
    df = df.copy()
    # 1h aggregation from 15m
    df_1h = df.set_index('timestamp').resample('1h', label='left', closed='left').agg({
        'close': 'last'}).dropna().reset_index()
    df_1h['sma200'] = pd.Series(df_1h['close'].values).rolling(200, min_periods=200).mean().values
    df_1h['close_above_sma200'] = df_1h['close'] > df_1h['sma200']
    df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=15)
    df = merge_htf(df, df_1h.rename(columns={'close_above_sma200': 'sma200_long'})[['timestamp', 'sma200_long']],
                   60, ['sma200_long'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Volume SMA
    df['volume_sma20'] = pd.Series(df['volume'].values).rolling(20, min_periods=20).mean().values

    # Wick metrics
    op = df['open'].values; hi = df['high'].values; lo = df['low'].values; cl = df['close'].values
    rng = hi - lo
    rng_safe = np.where(rng > 0, rng, np.nan)
    body_lo = np.minimum(op, cl)
    body_hi = np.maximum(op, cl)
    df['low_wick_ratio'] = (body_lo - lo) / rng_safe
    df['up_wick_ratio'] = (hi - body_hi) / rng_safe
    df['close_pos'] = (cl - lo) / rng_safe
    return df


def detect_engulfing_at(op, cl, i):
    bull = (cl[i] > op[i]) and (op[i-1] > cl[i-1]) and (cl[i] > op[i-1]) and (op[i] < cl[i-1])
    bear = (op[i] > cl[i]) and (cl[i-1] > op[i-1]) and (op[i] > cl[i-1]) and (cl[i] < op[i-1])
    return bull, bear


def detect_hammer_at(op, cl, hi, lo, i):
    body = abs(cl[i] - op[i])
    rng = hi[i] - lo[i]
    if rng <= 0: return False, False
    body_mid = (op[i] + cl[i]) / 2
    lower_wick = min(op[i], cl[i]) - lo[i]
    upper_wick = hi[i] - max(op[i], cl[i])
    hammer = (lower_wick >= 2 * body) and ((body_mid - lo[i]) / rng > 0.5) and (rng > 0)
    star = (upper_wick >= 2 * body) and ((body_mid - lo[i]) / rng < 0.5) and (rng > 0)
    return hammer, star


def entry_r21_native_15m(df, valid, params=None):
    """R21 concept on 15m: pattern reversal at 20-bar swing extreme."""
    p = {'volume_mult': 1.2, 'lookback_extreme': 20} if params is None else params
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vol = df['volume'].values
    vol_sma = df['volume_sma20'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values

    sigs = []
    for i in range(p['lookback_extreme'] + 2, n):
        if not valid[i]: continue
        if pd.isna(vol_sma[i]) or pd.isna(vol[i]): continue
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue

        recent_lows = lo[i - p['lookback_extreme']:i]
        recent_highs = hi[i - p['lookback_extreme']:i]
        recent_min = np.min(recent_lows)
        recent_max = np.max(recent_highs)
        low_touched = (lo[i-1] == recent_min) or (lo[i-2] == recent_min)
        high_touched = (hi[i-1] == recent_max) or (hi[i-2] == recent_max)

        bull_eng, bear_eng = detect_engulfing_at(op, cl, i)
        hammer, star = detect_hammer_at(op, cl, hi, lo, i)

        if low_touched and (bull_eng or hammer) and sma_long[i]:
            sigs.append((i, 'LONG'))
        elif high_touched and (bear_eng or star) and (not sma_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_production_15m(df, sigs, friction=0.07,
                            trail_K=2.5, max_sl_atr=4.5,
                            sl_min_pct=0.15, sl_max_pct=3.0,
                            emergency_sl_pct=3.0, max_hold_bars=192,
                            trail_activation_pct=0.05,
                            prog_trail_enabled=True, prog_trail_threshold=0.9, prog_trail_K_post=0.5,
                            min_bars_between=2):
    """C1 production-style exit on 15m data."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    atr = df['atr14_15m'].values
    sw_low = df['swing_low_10'].values
    sw_high = df['swing_high_10'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}

    in_pos = False
    pdir = None; pentry = None; psl = None; pemerg = None
    pbest = None; pstart = None; patr_e = None
    cooldown = 0
    trades = []
    rejected = 0
    i = 0
    while i < n:
        if in_pos:
            atr_now = atr[i] if not np.isnan(atr[i]) else patr_e
            if pdir == 'LONG':
                pbest = max(pbest, hi[i])
            else:
                pbest = min(pbest, lo[i])
            best_pnl_pct = ((pbest / pentry - 1) * 100) if pdir == 'LONG' else ((1 - pbest / pentry) * 100)
            if prog_trail_enabled and best_pnl_pct >= prog_trail_threshold:
                effective_K = prog_trail_K_post
            else:
                effective_K = trail_K
            trail_active = best_pnl_pct >= trail_activation_pct

            exit_price = None; exit_reason = None
            if pdir == 'LONG' and lo[i] <= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            elif pdir == 'SHORT' and hi[i] >= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            if exit_price is None:
                if pdir == 'LONG' and lo[i] <= psl:
                    exit_price, exit_reason = psl, 'SL'
                elif pdir == 'SHORT' and hi[i] >= psl:
                    exit_price, exit_reason = psl, 'SL'
            if exit_price is None and trail_active:
                if pdir == 'LONG':
                    trigger = pbest - effective_K * atr_now
                    if lo[i] <= trigger and trigger > pentry:
                        exit_price, exit_reason = trigger, 'TRAIL_TP'
                else:
                    trigger = pbest + effective_K * atr_now
                    if hi[i] >= trigger and trigger < pentry:
                        exit_price, exit_reason = trigger, 'TRAIL_TP'
            held = i - pstart
            if exit_price is None and held >= max_hold_bars:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                net = gross - friction
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'direction': pdir, 'entry': float(pentry), 'exit': float(exit_price),
                                'gross_pct': round(gross, 4), 'net_pct': round(net, 4),
                                'reason': exit_reason, 'bars_held': held,
                                'best_pnl_pct': round(best_pnl_pct, 4)})
                in_pos = False
                cooldown = i + min_bars_between

        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry_c = op[ni]
                pdir_c = sig_set[i]
                patr_c = atr[i] if not np.isnan(atr[i]) else 0
                if pdir_c == 'LONG':
                    structural = sw_low[i] if not np.isnan(sw_low[i]) else pentry_c - max_sl_atr * patr_c
                    atr_sl = pentry_c - max_sl_atr * patr_c
                    psl_c = max(structural, atr_sl)
                else:
                    structural = sw_high[i] if not np.isnan(sw_high[i]) else pentry_c + max_sl_atr * patr_c
                    atr_sl = pentry_c + max_sl_atr * patr_c
                    psl_c = min(structural, atr_sl)
                sl_pct_c = abs(pentry_c - psl_c) / pentry_c * 100
                if sl_pct_c < sl_min_pct or sl_pct_c > sl_max_pct:
                    rejected += 1
                    i += 1
                    continue
                pentry = pentry_c; pdir = pdir_c; patr_e = patr_c; psl = psl_c
                if pdir == 'LONG':
                    pemerg = pentry * (1 - emergency_sl_pct / 100)
                    pbest = hi[ni]
                else:
                    pemerg = pentry * (1 + emergency_sl_pct / 100)
                    pbest = lo[ni]
                pstart = ni
                in_pos = True
                i = ni
                continue
        i += 1
    return trades, rejected


def trade_summary(trades):
    if not trades: return None
    nets = [t['net_pct'] for t in trades]
    grosses = [t['gross_pct'] for t in trades]
    days = (pd.to_datetime(trades[-1]['exit_ts']) - pd.to_datetime(trades[0]['entry_ts'])).days
    if days == 0: days = 1
    wins = sum(1 for x in nets if x > 0)
    n = len(nets)
    win_pnls = [x for x in nets if x > 0]
    loss_pnls = [x for x in nets if x <= 0]
    rr = abs((sum(win_pnls)/max(1,len(win_pnls))) / (sum(loss_pnls)/max(1,len(loss_pnls)))) if loss_pnls else float('inf')
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    return {
        'n': n, 'days': days, 'per_day': round(n/days, 3),
        'sum_net': round(sum(nets), 2), 'avg_net': round(sum(nets)/n, 4),
        'avg_gross': round(sum(grosses)/n, 4),
        'wr_pct': round(100*wins/n, 2), 'rr': round(rr, 3),
        'daily_net': round(sum(nets)/days, 4),
        'reasons': reasons, 'trades': trades,
    }


def main():
    df, valid = prepare_15m_data()
    df = add_15m_extras(df)
    valid = valid & (~df['sma200_long'].isna()).values & (~pd.isna(df['volume_sma20']).values)

    n_total = len(df)
    print(f"15m bars: {n_total:,} | days: {n_total/96:.0f}")

    sigs = entry_r21_native_15m(df, valid)
    print(f"R21 native 15m signals: {len(sigs)} → {len(sigs)/(n_total/96):.2f}/day\n")

    if len(sigs) == 0:
        print("No signals.")
        return

    # Compare 4 configs (prog_trail ON/OFF × 2 friction)
    print(f"{'='*100}")
    print(f"R21 native 15m + production exit configurations")
    print(f"{'='*100}")
    print(f"{'config':<55} {'n':>5} {'rej':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}")

    configs = [
        ('R21-native-15m + prod (prog ON)', True, 0.07),
        ('R21-native-15m + prod (prog OFF)', False, 0.07),
    ]
    results = {}
    for label, prog_on, fric in configs:
        trades, rej = run_bt_production_15m(df, sigs, friction=fric, prog_trail_enabled=prog_on)
        s = trade_summary(trades)
        results[label] = (s, rej)
        if s:
            print(f"{label:<55} {s['n']:>5} {rej:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%")

    valid_results = {k: v[0] for k, v in results.items() if v[0] is not None}
    if not valid_results: return
    best_label = max(valid_results.keys(), key=lambda k: valid_results[k]['daily_net'])
    s_best = valid_results[best_label]
    best_prog = 'ON' in best_label
    print(f"\nBest: {best_label}")

    # Friction scenarios
    print(f"\n{'='*80}\nFriction scenarios on best\n{'='*80}")
    print(f"{'scenario':<20} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}  reasons")
    fric_scenarios = {}
    for label, fric in [('A maker', 0.04), ('B mixed', 0.07), ('C taker', 0.10), ('D worst', 0.15)]:
        trades, _ = run_bt_production_15m(df, sigs, friction=fric, prog_trail_enabled=best_prog)
        s = trade_summary(trades)
        fric_scenarios[label] = s
        if s:
            print(f"{label:<20} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%  {s['reasons']}")

    # Bootstrap
    print(f"\n{'='*80}\nBootstrap 1000 × 3-day on best\n{'='*80}")
    bars_per_3day = 3 * 96
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []; bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        v_w = valid[st:en]
        sigs_w = entry_r21_native_15m(df_w, v_w)
        trades, _ = run_bt_production_15m(df_w, sigs_w, friction=0.07, prog_trail_enabled=best_prog)
        cand_pnls.append(sum(t['net_pct'] for t in trades) if trades else 0)
        bh_pnls.append((df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07)
    mean_p = float(np.mean(cand_pnls))
    pos_rate = float(np.mean(np.array(cand_pnls) > 0))
    p5 = float(np.percentile(cand_pnls, 5))
    p_better = float(np.mean(np.array(cand_pnls) > np.array(bh_pnls)))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_vs_BH={p_better:.4f}")

    # WF 5-fold
    print(f"\n{'='*80}\nWF 5-fold on best\n{'='*80}")
    fold_size = n_total // 6
    wf = []
    for fold_i in range(5):
        tr_e = (fold_i + 1) * fold_size
        te_s = tr_e
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid[te_s:te_e]
        sigs_f = entry_r21_native_15m(df_f, v_f)
        trades, _ = run_bt_production_15m(df_f, sigs_f, friction=0.07, prog_trail_enabled=best_prog)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None,
                    'rr': s_f['rr'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'} RR={s_f['rr'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

    # 333-day windows
    print(f"\n{'='*80}\n333-day window check\n{'='*80}")
    bars_333 = 333 * 96
    if n_total > bars_333:
        for label, (s_idx, e_idx) in [
            ('First 333d', (0, bars_333)),
            ('Mid 333d', ((n_total-bars_333)//2, (n_total-bars_333)//2 + bars_333)),
            ('Last 333d', (n_total-bars_333, n_total)),
        ]:
            df_c = df.iloc[s_idx:e_idx].reset_index(drop=True)
            v_c = valid[s_idx:e_idx]
            sigs_c = entry_r21_native_15m(df_c, v_c)
            trades, _ = run_bt_production_15m(df_c, sigs_c, friction=0.07, prog_trail_enabled=best_prog)
            sc = trade_summary(trades)
            if sc:
                print(f"  {label:<15} n={sc['n']} daily={sc['daily_net']:+.4f}% WR={sc['wr_pct']}% RR={sc['rr']:.2f} avg_g={sc['avg_gross']:+.4f}%")

    # Verdict
    s_b = fric_scenarios.get('B mixed')
    cond = {
        'wf_3of5': wf_pos >= 3,
        'taker_C': fric_scenarios.get('C taker') and fric_scenarios['C taker']['daily_net'] >= 0.2,
        'mixed_B': s_b is not None and s_b['daily_net'] >= 0.2,
        'maker_A': fric_scenarios.get('A maker') and fric_scenarios['A maker']['daily_net'] >= 0.3,
        'bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1 and p_better >= 0.6,
        'gross_ge_0.10': s_b is not None and s_b['avg_gross'] >= 0.10,
        'freq_ge_2': s_b is not None and s_b['per_day'] >= 2.0,
        'wr_ge_40': s_b is not None and s_b['wr_pct'] >= 40,
        'rr_ge_1': s_b is not None and s_b['rr'] >= 1.0,
    }
    print(f"\n{'='*80}\nM3-R32 VERDICT\n{'='*80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'mechanism': 'R21 native 15m + C1 production exit',
           'n_signals': len(sigs),
           'configs': {k: ({kk: vv for kk, vv in v[0].items() if kk != 'trades'} if v[0] else None, v[1]) for k, v in results.items()},
           'best_friction': {k: ({kk: vv for kk, vv in v.items() if kk != 'trades'} if v else None) for k, v in fric_scenarios.items()},
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'conditions': cond, 'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r32_native_15m_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
