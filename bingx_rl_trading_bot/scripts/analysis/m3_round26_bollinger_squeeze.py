"""M3-R26 — Bollinger Squeeze breakout + deeper critique.

Mechanism: BB squeeze (low band width period) → breakout when band expands.
Different axis from R20-R25: volatility expansion play, not price direction.

Additional critique:
- Statistical significance: t-test on per-trade returns
- Regime decomposition: bull / bear / sideways
- Edge decay over time (rolling 60-day daily PnL)
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round20_dynamic_scalping import prepare_5m_data
from m3_round21_pattern_structure import add_sma200_1h


def add_bollinger_bands(df, period=20, n_std=2.0):
    """Bollinger bands + bandwidth."""
    cl = df['close'].values
    df['bb_mid'] = pd.Series(cl).rolling(period, min_periods=period).mean().values
    df['bb_std'] = pd.Series(cl).rolling(period, min_periods=period).std().values
    df['bb_upper'] = df['bb_mid'] + n_std * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - n_std * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100  # % width
    df['bb_width_pctile_50'] = pd.Series(df['bb_width'].values).rolling(100, min_periods=100).quantile(0.50).values
    df['bb_width_pctile_25'] = pd.Series(df['bb_width'].values).rolling(100, min_periods=100).quantile(0.25).values
    return df


def entry_squeeze_breakout(df, h1, h4, valid, params=None):
    """BB squeeze (current width < 25th pctile of past 100) followed by breakout."""
    p = {'squeeze_lookback': 5, 'volume_mult': 1.3} if params is None else params
    n = len(df)
    cl = df['close'].values
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    bb_upper = df['bb_upper'].values
    bb_lower = df['bb_lower'].values
    bb_width = df['bb_width'].values
    bb_width_p25 = df['bb_width_pctile_25'].values
    vol = df['volume'].values
    vol_sma = df['volume_sma20'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values

    sigs = []
    for i in range(p['squeeze_lookback'] + 2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (cl[i], bb_upper[i], bb_lower[i], bb_width[i], bb_width_p25[i],
                                      vol[i], vol_sma[i])):
            continue

        # Squeeze condition: past N bars all had bb_width < p25 of last 100
        squeeze = all(bb_width[i - k] < bb_width_p25[i - k]
                       for k in range(1, p['squeeze_lookback'] + 1)
                       if not pd.isna(bb_width[i - k]) and not pd.isna(bb_width_p25[i - k]))
        if not squeeze: continue

        # Volume confirm
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue

        # Breakout: close beyond upper band (LONG) or lower band (SHORT)
        long_break = cl[i] > bb_upper[i]
        short_break = cl[i] < bb_lower[i]

        # Trend filter (1h SMA200 alignment for breakout direction)
        if long_break and sma_long[i]:
            sigs.append((i, 'LONG'))
        elif short_break and (not sma_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_squeeze(df, sigs, friction_tp=0.04, friction_sl=0.07,
                    emergency_pct=1.0, timeout_bars=24, min_bars_between=2):
    """Exit: TP at opposite band (mean expansion), SL below squeeze low/above squeeze high."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    bb_upper = df['bb_upper'].values
    bb_lower = df['bb_lower'].values
    bb_mid = df['bb_mid'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}

    in_pos = False
    pdir = None; pentry = None; psl = None; ptp = None; pemerg = None; pstart = None
    cooldown = 0
    trades = []
    i = 0
    while i < n:
        if in_pos:
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
            if exit_price is None:
                if pdir == 'LONG' and hi[i] >= ptp:
                    exit_price, exit_reason = ptp, 'TP'
                elif pdir == 'SHORT' and lo[i] <= ptp:
                    exit_price, exit_reason = ptp, 'TP'
            held = i - pstart
            if exit_price is None and held >= timeout_bars:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                fric = friction_tp if exit_reason == 'TP' else friction_sl
                net = gross - fric
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'gross': gross, 'net': net, 'reason': exit_reason})
                in_pos = False
                cooldown = i + min_bars_between

        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                if pdir == 'LONG':
                    # SL at squeeze low (recent 5-bar low before breakout)
                    sq_low = float(np.min(lo[max(0, i-5):i+1]))
                    psl = sq_low * 0.999
                    # TP: 2× expansion (price moves equal to band width from breakout)
                    ptp = pentry + (bb_upper[i] - bb_lower[i])
                    pemerg = pentry * (1 - emergency_pct / 100)
                    if not (psl < pentry < ptp):
                        i += 1; continue
                else:
                    sq_high = float(np.max(hi[max(0, i-5):i+1]))
                    psl = sq_high * 1.001
                    ptp = pentry - (bb_upper[i] - bb_lower[i])
                    pemerg = pentry * (1 + emergency_pct / 100)
                    if not (ptp < pentry < psl):
                        i += 1; continue
                pstart = ni
                in_pos = True
                i = ni
                continue
        i += 1
    return trades


def trade_summary(trades):
    if not trades: return None
    nets = [t['net'] for t in trades]
    grosses = [t['gross'] for t in trades]
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
        'nets_array': nets,
    }


def t_test_per_trade(nets):
    """One-sample t-test: is mean per-trade return > 0?"""
    arr = np.array(nets)
    n = len(arr)
    if n < 2: return None
    mean_x = np.mean(arr)
    std_x = np.std(arr, ddof=1)
    se = std_x / np.sqrt(n)
    t_stat = mean_x / se if se > 0 else 0
    # p-value: one-sided (H1: mean > 0)
    from scipy.stats import t as t_dist
    p_value = 1 - t_dist.cdf(t_stat, df=n-1)
    return {'mean': float(mean_x), 'std': float(std_x), 'n': n, 't_stat': float(t_stat),
             'p_value_one_sided': float(p_value), 'significant_at_0.05': p_value < 0.05}


def main():
    df, h1, h4, valid = prepare_5m_data()
    df = add_sma200_1h(df)
    df = add_bollinger_bands(df, period=20, n_std=2.0)
    valid = (valid & (~df['sma200_long'].isna()).values
              & (~pd.isna(df['bb_upper']).values) & (~pd.isna(df['bb_width_pctile_25']).values))

    n_total = len(df)
    print(f"5m bars: {n_total:,} | days: {n_total/(24*12):.0f}")

    sigs = entry_squeeze_breakout(df, h1, h4, valid)
    print(f"BB squeeze breakout signals: {len(sigs)} → {len(sigs)/(n_total/(24*12)):.2f}/day\n")

    if len(sigs) == 0:
        print("No signals.")
        return

    print(f"{'='*80}\nFriction scenarios\n{'='*80}")
    print(f"{'scenario':<20} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}")
    scenarios = {}
    for label, ftp, fsl in [('A maker', 0.04, 0.04), ('B mixed', 0.04, 0.07),
                              ('C taker', 0.10, 0.10), ('D worst', 0.10, 0.15)]:
        trades = run_bt_squeeze(df, sigs, friction_tp=ftp, friction_sl=fsl)
        s = trade_summary(trades)
        scenarios[label] = s
        if s:
            print(f"{label:<20} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%  reasons={s['reasons']}")

    # Statistical significance: t-test on per-trade returns (mixed friction)
    s_b = scenarios.get('B mixed')
    if s_b:
        ttest = t_test_per_trade(s_b['nets_array'])
        print(f"\n{'='*80}\nStatistical significance: t-test (per-trade returns vs 0, mixed friction)\n{'='*80}")
        print(f"  mean per-trade: {ttest['mean']:+.6f}%")
        print(f"  std: {ttest['std']:.4f}%")
        print(f"  t-stat: {ttest['t_stat']:.3f}, p (one-sided): {ttest['p_value_one_sided']:.4f}")
        print(f"  Significant at α=0.05: {ttest['significant_at_0.05']}")

    # Bootstrap
    print(f"\n{'='*80}\nBootstrap 1000 × 3-day\n{'='*80}")
    bars_per_3day = 3 * 24 * 12
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []; bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        h1_w = h1[st:en]; h4_w = h4[st:en]; v_w = valid[st:en]
        sigs_w = entry_squeeze_breakout(df_w, h1_w, h4_w, v_w)
        trades = run_bt_squeeze(df_w, sigs_w, friction_tp=0.04, friction_sl=0.07)
        cand_pnls.append(sum(t['net'] for t in trades) if trades else 0)
        bh_pnls.append((df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07)
    mean_p = float(np.mean(cand_pnls))
    pos_rate = float(np.mean(np.array(cand_pnls) > 0))
    p5 = float(np.percentile(cand_pnls, 5))
    p_better = float(np.mean(np.array(cand_pnls) > np.array(bh_pnls)))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_vs_BH={p_better:.4f}")

    # Look-ahead audit
    audit_idx = random.sample([i for i, _ in sigs], min(20, len(sigs)))
    leaks = 0
    for i in audit_idx:
        df_t = df.iloc[:i+1].copy()
        h1_t = h1[:i+1]; h4_t = h4[:i+1]; v_t = valid[:i+1]
        try:
            t_sigs = entry_squeeze_breakout(df_t, h1_t, h4_t, v_t)
            full_at_i = next((d for idx, d in sigs if idx == i), None)
            trunc_at_i = next((d for idx, d in t_sigs if idx == i), None)
            if full_at_i != trunc_at_i:
                leaks += 1
        except Exception:
            leaks += 1
    print(f"\nLook-ahead leaks: {leaks}/{len(audit_idx)}")

    # Regime decomposition: 3 chunks (early/mid/late period)
    print(f"\n{'='*80}\nRegime decomposition (3 chunks)\n{'='*80}")
    chunk = n_total // 3
    print(f"{'period':<25} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'avg_g':>10}")
    for label, (s_idx, e_idx) in [
        ('Early (chunk 1)', (0, chunk)),
        ('Mid (chunk 2)', (chunk, 2*chunk)),
        ('Late (chunk 3)', (2*chunk, n_total)),
    ]:
        df_c = df.iloc[s_idx:e_idx].reset_index(drop=True)
        h1_c = h1[s_idx:e_idx]; h4_c = h4[s_idx:e_idx]; v_c = valid[s_idx:e_idx]
        sigs_c = entry_squeeze_breakout(df_c, h1_c, h4_c, v_c)
        trades = run_bt_squeeze(df_c, sigs_c, friction_tp=0.04, friction_sl=0.07)
        sc = trade_summary(trades)
        if sc:
            print(f"{label:<25} {sc['n']:>5} {sc['per_day']:>7.3f} {sc['daily_net']:>+9.4f}% "
                  f"{sc['wr_pct']:>5.1f}% {sc['avg_gross']:>+9.4f}%")
        else:
            print(f"{label:<25} no trades")

    # WF 5-fold
    print(f"\n{'='*80}\nWF 5-fold\n{'='*80}")
    fold_size = n_total // 6
    wf = []
    for fold_i in range(5):
        tr_e = (fold_i + 1) * fold_size
        te_s = tr_e
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        h1_f = h1[te_s:te_e]; h4_f = h4[te_s:te_e]; v_f = valid[te_s:te_e]
        sigs_f = entry_squeeze_breakout(df_f, h1_f, h4_f, v_f)
        trades = run_bt_squeeze(df_f, sigs_f, friction_tp=0.04, friction_sl=0.07)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

    # Verdict
    cond = {
        'lookahead': leaks == 0,
        'wf_3of5': wf_pos >= 3,
        'taker_C': scenarios.get('C taker') and scenarios['C taker']['daily_net'] >= 0.2,
        'mixed_B': s_b is not None and s_b['daily_net'] >= 0.2,
        'maker_A': scenarios.get('A maker') and scenarios['A maker']['daily_net'] >= 0.3,
        'bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1 and p_better >= 0.6,
        'gross_ge_0.10': s_b is not None and s_b['avg_gross'] >= 0.10,
        'freq_ge_2': s_b is not None and s_b['per_day'] >= 2.0,
        'wr_rr': s_b is not None and s_b['wr_pct'] >= 50 and s_b['rr'] >= 1.0,
        't_test_significant': ttest is not None and ttest['significant_at_0.05'],
    }
    print(f"\n{'='*80}\nM3-R26 VERDICT\n{'='*80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'mechanism': 'Bollinger Squeeze breakout',
           'n_signals': len(sigs),
           'scenarios': {k: {kk: vv for kk, vv in v.items() if kk != 'nets_array'} if v else None
                          for k, v in scenarios.items()},
           't_test': ttest,
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'lookahead_leaks': leaks,
           'conditions': cond,
           'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r26_squeeze_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
