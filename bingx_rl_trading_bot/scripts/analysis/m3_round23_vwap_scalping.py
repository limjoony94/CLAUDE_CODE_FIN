"""M3-R23 — Anchored VWAP mean reversion scalping.

Professional intraday tool. Daily session anchor (00:00 UTC), z-score deviation reversion.
Pre-reg: hypothesis + tests below.
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


def compute_anchored_vwap(df):
    """Daily session VWAP, anchored at 00:00 UTC each day. Cumulative within day."""
    n = len(df)
    timestamps = pd.to_datetime(df['timestamp'])
    dates = timestamps.dt.date.values
    typical_price = (df['high'].values + df['low'].values + df['close'].values) / 3
    volume = df['volume'].values
    pv = typical_price * volume

    # Cumulative within each session (date)
    vwap = np.full(n, np.nan)
    vwap_std = np.full(n, np.nan)
    cum_pv = 0.0; cum_v = 0.0; cum_pv_sq = 0.0
    cum_pv_data = []  # for std calc
    prev_date = None
    for i in range(n):
        d = dates[i]
        if d != prev_date:
            cum_pv = 0.0; cum_v = 0.0
            cum_pv_data = []
            prev_date = d
        cum_pv += pv[i]
        cum_v += volume[i]
        if cum_v > 0:
            v = cum_pv / cum_v
            vwap[i] = v
            cum_pv_data.append(typical_price[i])
            if len(cum_pv_data) >= 5:
                vwap_std[i] = float(np.std(cum_pv_data))
    return vwap, vwap_std


def entry_vwap_reversion(df, h1, h4, valid, params=None):
    p = {'z_thresh': 2.0, 'min_session_bars': 12} if params is None else params
    n = len(df)
    cl = df['close'].values
    vwap = df['vwap'].values
    vwap_std = df['vwap_std'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values
    timestamps = pd.to_datetime(df['timestamp'])
    dates = timestamps.dt.date.values

    # Bar-of-day count (need ≥ min_session_bars before signaling)
    bar_of_day = np.zeros(n, dtype=int)
    prev_date = None
    cnt = 0
    for i in range(n):
        if dates[i] != prev_date:
            cnt = 0
            prev_date = dates[i]
        bar_of_day[i] = cnt
        cnt += 1

    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if bar_of_day[i] < p['min_session_bars']: continue
        if pd.isna(vwap[i]) or pd.isna(vwap_std[i]) or vwap_std[i] <= 0: continue
        z = (cl[i] - vwap[i]) / vwap_std[i]
        # LONG: price below VWAP by z_thresh sigmas (oversold relative to VWAP) + 1h trend permissive
        if z < -p['z_thresh'] and sma_long[i]:
            sigs.append((i, 'LONG'))
        elif z > p['z_thresh'] and (not sma_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_vwap(df, sigs, friction_tp=0.04, friction_sl=0.07,
                  emergency_pct=0.8, timeout_bars=24, min_bars_between=2):
    """Exit: TP at VWAP touch (mean reversion target), SL at recent swing, emergency."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vwap = df['vwap'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}

    in_pos = False
    pdir = None; pentry = None; psl = None; pemerg = None; pstart = None
    cooldown = 0
    trades = []
    i = 0
    while i < n:
        if in_pos:
            exit_price = None; exit_reason = None
            # Emergency
            if pdir == 'LONG' and lo[i] <= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            elif pdir == 'SHORT' and hi[i] >= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            # SL
            if exit_price is None:
                if pdir == 'LONG' and lo[i] <= psl:
                    exit_price, exit_reason = psl, 'SL'
                elif pdir == 'SHORT' and hi[i] >= psl:
                    exit_price, exit_reason = psl, 'SL'
            # TP at VWAP touch
            if exit_price is None and not pd.isna(vwap[i]):
                if pdir == 'LONG' and hi[i] >= vwap[i]:
                    exit_price, exit_reason = vwap[i], 'TP_VWAP'
                elif pdir == 'SHORT' and lo[i] <= vwap[i]:
                    exit_price, exit_reason = vwap[i], 'TP_VWAP'
            # Timeout
            held = i - pstart
            if exit_price is None and held >= timeout_bars:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                if exit_reason == 'TP_VWAP':
                    fric = friction_tp
                elif exit_reason in ('SL', 'EMERGENCY'):
                    fric = friction_sl
                else:
                    fric = friction_sl
                net = gross - fric
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'direction': pdir, 'entry': float(pentry), 'exit': float(exit_price),
                                'gross_pct': round(gross, 4), 'net_pct': round(net, 4),
                                'reason': exit_reason, 'bars_held': held, 'fric': fric})
                in_pos = False
                cooldown = i + min_bars_between

        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                # SL at recent swing (10-bar)
                if pdir == 'LONG':
                    swing_low = float(np.min(lo[max(0, i-10):i+1]))
                    psl = swing_low * 0.9995
                    pemerg = pentry * (1 - emergency_pct / 100)
                else:
                    swing_high = float(np.max(hi[max(0, i-10):i+1]))
                    psl = swing_high * 1.0005
                    pemerg = pentry * (1 + emergency_pct / 100)
                # Sanity: vwap should be on TP side
                if pd.isna(vwap[i]):
                    i += 1; continue
                if pdir == 'LONG' and not (psl < pentry < vwap[i]):
                    i += 1; continue
                if pdir == 'SHORT' and not (vwap[i] < pentry < psl):
                    i += 1; continue
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
        'sum_net': round(sum(nets), 2), 'sum_gross': round(sum(grosses), 2),
        'avg_net': round(sum(nets)/n, 4), 'avg_gross': round(sum(grosses)/n, 4),
        'wr_pct': round(100*wins/n, 2), 'rr': round(rr, 3),
        'daily_net': round(sum(nets)/days, 4),
        'reasons': reasons,
    }


def main():
    df, h1, h4, valid = prepare_5m_data()
    df = add_sma200_1h(df)
    valid = valid & (~df['sma200_long'].isna()).values

    # Compute VWAP
    print("Computing anchored VWAP (daily 00:00 UTC)...")
    vwap, vwap_std = compute_anchored_vwap(df)
    df['vwap'] = vwap
    df['vwap_std'] = vwap_std
    valid = valid & (~pd.isna(df['vwap']).values) & (~pd.isna(df['vwap_std']).values)

    n_total = len(df)
    print(f"5m bars: {n_total:,} | days: {n_total/(24*12):.0f}")

    # Test multiple z thresholds (small grid, then validate)
    print(f"\n{'='*80}\nz-threshold scan\n{'='*80}")
    print(f"{'z_thresh':>10} {'n_signals':>10} {'per_day':>8}")
    for z in (1.5, 2.0, 2.5, 3.0):
        sigs_z = entry_vwap_reversion(df, h1, h4, valid, params={'z_thresh': z, 'min_session_bars': 12})
        print(f"{z:>10.1f} {len(sigs_z):>10} {len(sigs_z)/(n_total/(24*12)):>7.2f}")

    # Use z=2.0 as base (pre-registered choice)
    z_base = 2.0
    print(f"\nUsing z_thresh = {z_base} (pre-registered base)")
    sigs = entry_vwap_reversion(df, h1, h4, valid, params={'z_thresh': z_base, 'min_session_bars': 12})

    if len(sigs) == 0:
        print("No signals.")
        return

    # Friction scenarios
    print(f"\n{'='*80}\nFriction scenarios\n{'='*80}")
    print(f"{'scenario':<25} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}")
    scenarios = {}
    for label, ftp, fsl in [('A maker', 0.04, 0.04), ('B mixed', 0.04, 0.07),
                              ('C taker', 0.10, 0.10), ('D worst', 0.10, 0.15)]:
        trades = run_bt_vwap(df, sigs, friction_tp=ftp, friction_sl=fsl)
        s = trade_summary(trades)
        scenarios[label] = s
        if s:
            print(f"{label:<25} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%  reasons={s['reasons']}")

    # Bootstrap 1000 × 3-day
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
        sigs_w = entry_vwap_reversion(df_w, h1_w, h4_w, v_w, params={'z_thresh': z_base, 'min_session_bars': 12})
        trades = run_bt_vwap(df_w, sigs_w, friction_tp=0.04, friction_sl=0.07)
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
        h1_t = h1[:i+1]; h4_t = h4[:i+1]; v_t = valid[:i+1]
        try:
            t_sigs = entry_vwap_reversion(df_t, h1_t, h4_t, v_t, params={'z_thresh': z_base, 'min_session_bars': 12})
            full_at_i = next((d for idx, d in sigs if idx == i), None)
            trunc_at_i = next((d for idx, d in t_sigs if idx == i), None)
            if full_at_i != trunc_at_i:
                leaks += 1
        except Exception:
            leaks += 1
    print(f"\n  Look-ahead audit: {len(audit_idx)} audited, leaks: {leaks}")

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
        sigs_f = entry_vwap_reversion(df_f, h1_f, h4_f, v_f, params={'z_thresh': z_base, 'min_session_bars': 12})
        trades = run_bt_vwap(df_f, sigs_f, friction_tp=0.04, friction_sl=0.07)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

    # 7-test verdict
    s_b = scenarios.get('B mixed')
    cond = {
        'test1_lookahead': leaks == 0,
        'test2_wf_3of5': wf_pos >= 3,
        'test3a_taker_C': scenarios.get('C taker') and scenarios['C taker']['daily_net'] >= 0.2,
        'test3b_mixed_B': s_b is not None and s_b['daily_net'] >= 0.2,
        'test3c_maker_A': scenarios.get('A maker') and scenarios['A maker']['daily_net'] >= 0.3,
        'test4_bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1.0 and p_better >= 0.6,
        'test5_gross_vs_fee': s_b is not None and s_b['avg_gross'] >= 0.10,
        'test6_freq_2': s_b is not None and s_b['per_day'] >= 2.0,
        'test7_wr50_rr1': s_b is not None and s_b['wr_pct'] >= 50 and s_b['rr'] >= 1.0,
    }
    print(f"\n{'='*80}\nM3-R23 VERDICT\n{'='*80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'z_base': z_base, 'n_signals': len(sigs),
           'scenarios': scenarios,
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'lookahead_leaks': leaks,
           'conditions': cond,
           'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r23_vwap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
