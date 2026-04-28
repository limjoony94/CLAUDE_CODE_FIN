"""M3-R29 — C1 Breakout v2.6 EXACT mechanism re-validation.

Production-deployed C1 (shelved 2026-04-27 due to BT-LIVE parity gap, NOT BT criteria fail).
Replicate C1's exact spec, apply 28-round comprehensive critique suite.
User criterion relaxed: WR ≥ 40% (from 50%).

C1 spec (CLAUDE.md verbatim):
- 15m Donchian channel breakout: close > prior 15-bar high
- Body ratio filter: body_abs > 40% of (high - low)
- Fractal SL: structural swing low (10-bar lookback), max 3.3×ATR cap
- ATR Trailing TP: best_price - 2.5×ATR (LONG) / + 2.5×ATR (SHORT)
- Emergency: 3.0% hard SL
- Timeout: 192 bars (48h on 15m)
- Single asset BTC
- N=1, no concurrent positions
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m1_bt_framework import compute_atr as compute_atr_list
from m2_round1_screening import compute_ema, load_ohlcv, merge_htf


def compute_atr_arr(highs, lows, closes, period=14):
    return np.array(compute_atr_list(list(highs), list(lows), list(closes), period))


def prepare_15m_data():
    """C1 native 15m data."""
    df = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    df['atr14_15m'] = compute_atr_arr(highs, lows, closes, 14)
    df['high_15_prev'] = pd.Series(highs).rolling(15, min_periods=15).max().shift(1).values
    df['low_15_prev'] = pd.Series(lows).rolling(15, min_periods=15).min().shift(1).values
    df['swing_low_10'] = pd.Series(lows).rolling(10, min_periods=10).min().shift(1).values
    df['swing_high_10'] = pd.Series(highs).rolling(10, min_periods=10).max().shift(1).values

    valid = ((~pd.isna(df['atr14_15m']))
              & (~pd.isna(df['high_15_prev'])) & (~pd.isna(df['low_15_prev']))
              & (~pd.isna(df['swing_low_10']))).values
    return df, valid


def entry_c1_exact(df, valid, params=None):
    """C1 Breakout v2.6 exact entry: Donchian + body filter."""
    p = {'breakout_bars': 15, 'body_min_ratio': 0.40} if params is None else params
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    high_prev = df['high_15_prev'].values if p['breakout_bars'] == 15 else pd.Series(hi).rolling(p['breakout_bars'], min_periods=p['breakout_bars']).max().shift(1).values
    low_prev = df['low_15_prev'].values if p['breakout_bars'] == 15 else pd.Series(lo).rolling(p['breakout_bars'], min_periods=p['breakout_bars']).min().shift(1).values

    sigs = []
    for i in range(p['breakout_bars'] + 2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (cl[i], op[i], hi[i], lo[i], high_prev[i], low_prev[i])):
            continue

        # Body ratio filter
        body = abs(cl[i] - op[i])
        rng = hi[i] - lo[i]
        if rng <= 0: continue
        body_ratio = body / rng
        if body_ratio < p['body_min_ratio']: continue

        # Channel breakout direction
        long_break = cl[i] > high_prev[i] and cl[i] > op[i]  # bullish breakout
        short_break = cl[i] < low_prev[i] and cl[i] < op[i]  # bearish breakout

        if long_break:
            sigs.append((i, 'LONG'))
        elif short_break:
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_c1_exact(df, sigs, friction=0.10,
                     trail_k=2.5, sl_atr_cap=3.3, emergency_pct=3.0, timeout_bars=192,
                     min_bars_between=2):
    """C1 BT: Fractal SL with 3.3×ATR cap, ATR trailing TP, emergency 3%, timeout 48h."""
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
    pbest = None; pstart = None
    cooldown = 0
    trades = []
    i = 0
    while i < n:
        if in_pos:
            atr_now = atr[i] if not np.isnan(atr[i]) else (atr[i-1] if i > 0 else 0)
            # Update best
            if pdir == 'LONG':
                pbest = max(pbest, hi[i])
            else:
                pbest = min(pbest, lo[i])

            exit_price = None; exit_reason = None
            # Emergency
            if pdir == 'LONG' and lo[i] <= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            elif pdir == 'SHORT' and hi[i] >= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            # SL (structural fixed at entry)
            if exit_price is None:
                if pdir == 'LONG' and lo[i] <= psl:
                    exit_price, exit_reason = psl, 'SL'
                elif pdir == 'SHORT' and hi[i] >= psl:
                    exit_price, exit_reason = psl, 'SL'
            # Trailing TP at best - K×ATR
            if exit_price is None:
                if pdir == 'LONG':
                    trigger = pbest - trail_k * atr_now
                    if lo[i] <= trigger and trigger > pentry:  # only if profitable
                        exit_price, exit_reason = trigger, 'TRAIL_TP'
                else:
                    trigger = pbest + trail_k * atr_now
                    if hi[i] >= trigger and trigger < pentry:
                        exit_price, exit_reason = trigger, 'TRAIL_TP'
            # Timeout
            held = i - pstart
            if exit_price is None and held >= timeout_bars:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                net = gross - friction
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'direction': pdir, 'entry': float(pentry), 'exit': float(exit_price),
                                'gross_pct': round(gross, 4), 'net_pct': round(net, 4),
                                'reason': exit_reason, 'bars_held': held})
                in_pos = False
                cooldown = i + min_bars_between

        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                atr_e = atr[i] if not np.isnan(atr[i]) else 0
                # Fractal SL: structural max 3.3×ATR cap
                if pdir == 'LONG':
                    structural = sw_low[i] if not np.isnan(sw_low[i]) else pentry - sl_atr_cap * atr_e
                    atr_sl = pentry - sl_atr_cap * atr_e
                    psl = max(structural, atr_sl)  # tighter of structure or vol cap
                    pemerg = pentry * (1 - emergency_pct / 100)
                    pbest = hi[ni]
                else:
                    structural = sw_high[i] if not np.isnan(sw_high[i]) else pentry + sl_atr_cap * atr_e
                    atr_sl = pentry + sl_atr_cap * atr_e
                    psl = min(structural, atr_sl)
                    pemerg = pentry * (1 + emergency_pct / 100)
                    pbest = lo[ni]
                pstart = ni
                in_pos = True
                i = ni
                continue
        i += 1
    return trades


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
        'reasons': reasons,
        'trades': trades,
    }


def t_test(nets):
    arr = np.array(nets)
    n = len(arr)
    if n < 2: return None
    from scipy.stats import t as t_dist
    mean_x = np.mean(arr); std_x = np.std(arr, ddof=1)
    se = std_x / np.sqrt(n)
    t_stat = mean_x / se if se > 0 else 0
    p_value = 1 - t_dist.cdf(t_stat, df=n-1)
    return {'mean': float(mean_x), 'std': float(std_x), 't_stat': float(t_stat),
             'p_one_sided': float(p_value), 'sig_05': bool(p_value < 0.05)}


def main():
    df, valid = prepare_15m_data()
    n_total = len(df)
    print(f"15m bars: {n_total:,} | days: {n_total/96:.0f}")

    sigs = entry_c1_exact(df, valid)
    print(f"C1 exact signals: {len(sigs)} → {len(sigs)/(n_total/96):.2f}/day\n")

    if len(sigs) == 0:
        print("No signals.")
        return

    # Friction scenarios
    print(f"{'='*80}\nFriction scenarios (C1 exact: 15m Donchian + body + fractal + 2.5×ATR trail + 3% emerg + 192bar timeout)\n{'='*80}")
    print(f"{'scenario':<20} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}  reasons")
    scenarios = {}
    for label, fric in [('A maker (0.04)', 0.04), ('B mixed (0.07)', 0.07), ('C taker (0.10)', 0.10), ('D worst (0.15)', 0.15)]:
        trades = run_bt_c1_exact(df, sigs, friction=fric)
        s = trade_summary(trades)
        scenarios[label] = s
        if s:
            print(f"{label:<20} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%  {s['reasons']}")

    s_b = scenarios.get('B mixed (0.07)')
    if not s_b:
        print("No mixed scenario.")
        return

    # T-test
    print(f"\n{'='*80}\nT-test (per-trade returns at mixed friction)\n{'='*80}")
    nets = [t['net_pct'] for t in s_b['trades']]
    ttest = t_test(nets)
    print(f"  mean: {ttest['mean']:+.6f}%, t-stat: {ttest['t_stat']:.3f}, p (one-sided H1: mean>0): {ttest['p_one_sided']:.4f}")
    print(f"  Significant at α=0.05: {ttest['sig_05']}")

    # Bootstrap 1000 × 3-day
    print(f"\n{'='*80}\nBootstrap 1000 × 3-day windows\n{'='*80}")
    bars_per_3day = 3 * 96  # 15m
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []; bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        v_w = valid[st:en]
        sigs_w = entry_c1_exact(df_w, v_w)
        trades = run_bt_c1_exact(df_w, sigs_w, friction=0.07)
        cand_pnls.append(sum(t['net_pct'] for t in trades) if trades else 0)
        bh_pnls.append((df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07)
    mean_p = float(np.mean(cand_pnls))
    pos_rate = float(np.mean(np.array(cand_pnls) > 0))
    p5 = float(np.percentile(cand_pnls, 5))
    p_better = float(np.mean(np.array(cand_pnls) > np.array(bh_pnls)))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_vs_BH={p_better:.4f}")

    # Look-ahead
    audit_idx = random.sample([i for i, _ in sigs], min(20, len(sigs)))
    leaks = 0
    for i in audit_idx:
        df_t = df.iloc[:i+1].copy()
        v_t = valid[:i+1]
        try:
            t_sigs = entry_c1_exact(df_t, v_t)
            full_at_i = next((d for idx, d in sigs if idx == i), None)
            trunc_at_i = next((d for idx, d in t_sigs if idx == i), None)
            if full_at_i != trunc_at_i:
                leaks += 1
        except Exception:
            leaks += 1
    print(f"\nLook-ahead leaks: {leaks}/{len(audit_idx)}")

    # WF 5-fold
    print(f"\n{'='*80}\nWF 5-fold (mixed friction)\n{'='*80}")
    fold_size = n_total // 6
    wf = []
    for fold_i in range(5):
        tr_e = (fold_i + 1) * fold_size
        te_s = tr_e
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid[te_s:te_e]
        sigs_f = entry_c1_exact(df_f, v_f)
        trades = run_bt_c1_exact(df_f, sigs_f, friction=0.07)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None,
                    'rr': s_f['rr'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'} RR={s_f['rr'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

    # Per-direction
    long_t = [t for t in s_b['trades'] if t['direction'] == 'LONG']
    short_t = [t for t in s_b['trades'] if t['direction'] == 'SHORT']
    print(f"\n{'='*80}\nPer-direction breakdown\n{'='*80}")
    if long_t:
        l_nets = [t['net_pct'] for t in long_t]
        l_wins = sum(1 for x in l_nets if x > 0)
        print(f"  LONG: n={len(long_t)} avg_net={sum(l_nets)/len(long_t):+.4f}% WR={100*l_wins/len(long_t):.1f}% sum_net={sum(l_nets):+.2f}%")
    if short_t:
        s_nets = [t['net_pct'] for t in short_t]
        s_wins = sum(1 for x in s_nets if x > 0)
        print(f"  SHORT: n={len(short_t)} avg_net={sum(s_nets)/len(short_t):+.4f}% WR={100*s_wins/len(short_t):.1f}% sum_net={sum(s_nets):+.2f}%")

    # Verdict (WR ≥ 40% relaxed criterion)
    print(f"\n{'='*80}\nM3-R29 VERDICT (WR ≥ 40% relaxed)\n{'='*80}")
    cond = {
        'lookahead_leaks_0': leaks == 0,
        'wf_3of5': wf_pos >= 3,
        'taker_C_daily_positive': scenarios.get('C taker (0.10)') and scenarios['C taker (0.10)']['daily_net'] >= 0.2,
        'mixed_B_daily_positive': s_b['daily_net'] >= 0.2,
        'maker_A_daily': scenarios.get('A maker (0.04)') and scenarios['A maker (0.04)']['daily_net'] >= 0.3,
        'bootstrap_pos_rate_50': pos_rate >= 0.5 and mean_p > 0 and p5 > -1,
        'gross_ge_taker_0.10': s_b['avg_gross'] >= 0.10,
        'freq_ge_2': s_b['per_day'] >= 2.0,
        'wr_ge_40': s_b['wr_pct'] >= 40,
        'rr_ge_1': s_b['rr'] >= 1.0,
        't_test_sig_positive': ttest['sig_05'] and ttest['mean'] > 0,
    }
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'mechanism': 'C1 Breakout v2.6 EXACT (production logic, shelved)',
           'n_signals': len(sigs),
           'scenarios': {k: {kk: vv for kk, vv in v.items() if kk != 'trades'} if v else None for k, v in scenarios.items()},
           't_test': ttest,
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'lookahead_leaks': leaks,
           'conditions': cond, 'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r29_c1_exact_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
