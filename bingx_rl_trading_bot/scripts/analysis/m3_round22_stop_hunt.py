"""M3-R22 — Stop-Hunt Liquidity Reversal scalping.

Pre-reg: claudedocs/m3_round22_stop_hunt.md

Microstructure edge: swing extreme breach + rejection + volume + cross-asset divergence.
Asymmetric R:R via tight structural SL + 2.5R TP target.
Full 7-test suite per user spec.
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


def entry_stop_hunt(df, h1, h4, valid, params=None):
    """Stop-Hunt detection: breach + rejection + volume + cross-asset divergence."""
    p = {
        'lookback_swing': 20,
        'min_lower_wick_to_body': 2.0,
        'volume_mult': 1.5,
        'eth_div_thresh': -0.3,  # ETH not below this for LONG
    } if params is None else params
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vol = df['volume'].values
    vol_sma = df['volume_sma20'].values
    eth_ret = df['eth_return_5m'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values

    sigs = []
    lb = p['lookback_swing']
    for i in range(lb + 3, n):
        if not valid[i]: continue
        if pd.isna(vol[i]) or pd.isna(vol_sma[i]): continue
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue
        if pd.isna(eth_ret[i]): continue

        # Swing reference: low_at_t_minus_2 = swing low computed from i-lb to i-2 (before current bar)
        recent_low = float(np.min(lo[i - lb:i - 1]))  # past lb bars excluding current
        recent_high = float(np.max(hi[i - lb:i - 1]))

        body = abs(cl[i] - op[i])
        rng = hi[i] - lo[i]
        if rng <= 0: continue
        lower_wick = min(op[i], cl[i]) - lo[i]
        upper_wick = hi[i] - max(op[i], cl[i])

        # LONG stop-hunt: breach + close back above + long lower wick + ETH not also down + 1h trend permissive
        long_breach = lo[i] < recent_low
        long_reject = cl[i] > recent_low
        long_wick = (body > 0) and (lower_wick >= p['min_lower_wick_to_body'] * body)
        long_eth_div = eth_ret[i] > p['eth_div_thresh']  # ETH not severely negative

        # SHORT stop-hunt
        short_breach = hi[i] > recent_high
        short_reject = cl[i] < recent_high
        short_wick = (body > 0) and (upper_wick >= p['min_lower_wick_to_body'] * body)
        short_eth_div = eth_ret[i] < -p['eth_div_thresh']

        if long_breach and long_reject and long_wick and long_eth_div and sma_long[i]:
            sigs.append((i, 'LONG'))
        elif short_breach and short_reject and short_wick and short_eth_div and (not sma_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_stop_hunt(df, sigs, friction_tp=0.04, friction_sl=0.07,
                     tp_R=2.5, emergency_pct=0.8, timeout_bars=12, min_bars_between=2):
    """Asymmetric R:R: SL = spike low/high − buffer. TP = entry + (entry-SL)×tp_R."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}
    spike_set = {idx: (lo[idx], hi[idx]) for idx, _ in sigs}  # spike's lo/hi for SL

    in_pos = False
    pdir = None; pentry = None; psl = None; ptp1 = None; ptp2 = None; pemerg = None
    pstart = None; tp1_hit = False; sig_lo = None; sig_hi = None
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

            # TP1 → trail SL to BE
            if exit_price is None and not tp1_hit:
                if pdir == 'LONG' and hi[i] >= ptp1:
                    psl = max(psl, pentry)  # trail to BE (no profit lock to maximize TP2)
                    tp1_hit = True
                elif pdir == 'SHORT' and lo[i] <= ptp1:
                    psl = min(psl, pentry)
                    tp1_hit = True

            # TP2 final
            if exit_price is None:
                if pdir == 'LONG' and hi[i] >= ptp2:
                    exit_price, exit_reason = ptp2, 'TP2'
                elif pdir == 'SHORT' and lo[i] <= ptp2:
                    exit_price, exit_reason = ptp2, 'TP2'

            # Timeout
            held = i - pstart
            if exit_price is None and held >= timeout_bars:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                if exit_reason == 'TP2':
                    fric = friction_tp
                elif exit_reason in ('SL', 'EMERGENCY'):
                    fric = friction_sl
                else:  # TIMEOUT
                    fric = friction_sl
                net = gross - fric
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'direction': pdir, 'entry': float(pentry), 'exit': float(exit_price),
                                'gross_pct': round(gross, 4), 'net_pct': round(net, 4),
                                'reason': exit_reason, 'bars_held': held, 'fric': fric})
                in_pos = False
                cooldown = i + min_bars_between
                tp1_hit = False

        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                sig_lo, sig_hi = spike_set[i]
                if pdir == 'LONG':
                    psl = sig_lo * 0.9995  # 0.05% buffer below spike low
                    risk = pentry - psl
                    ptp1 = pentry + risk * 1.0  # 1R
                    ptp2 = pentry + risk * 2.5  # 2.5R
                    pemerg = pentry * (1 - emergency_pct / 100)
                else:
                    psl = sig_hi * 1.0005
                    risk = psl - pentry
                    ptp1 = pentry - risk * 1.0
                    ptp2 = pentry - risk * 2.5
                    pemerg = pentry * (1 + emergency_pct / 100)

                # Sanity
                if pdir == 'LONG' and not (psl < pentry < ptp1 < ptp2):
                    i += 1; continue
                if pdir == 'SHORT' and not (ptp2 < ptp1 < pentry < psl):
                    i += 1; continue

                pstart = ni
                in_pos = True
                tp1_hit = False
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

    n_total = len(df)
    print(f"\n5m bars: {n_total:,} | days: {n_total/(24*12):.0f}")

    sigs = entry_stop_hunt(df, h1, h4, valid)
    print(f"Stop-hunt signals: {len(sigs)} → {len(sigs)/(n_total/(24*12)):.2f}/day\n")

    if len(sigs) == 0:
        print("No signals.")
        return

    # Friction scenarios
    print("=" * 80); print("Friction scenarios"); print("=" * 80)
    print(f"{'scenario':<25} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10} reasons")
    scenarios = {}
    for label, ftp, fsl in [
        ('A maker both', 0.04, 0.04),
        ('B mixed (TP-mkr/SL-tkr)', 0.04, 0.07),
        ('C taker both', 0.10, 0.10),
        ('D worst', 0.10, 0.15),
    ]:
        trades = run_bt_stop_hunt(df, sigs, friction_tp=ftp, friction_sl=fsl)
        s = trade_summary(trades)
        scenarios[label] = s
        if s:
            print(f"{label:<25} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}% {s['reasons']}")

    # Bootstrap 1000 × 3-day
    print(f"\n{'=' * 80}\nBootstrap 1000 × 3-day windows (mixed B friction)\n{'=' * 80}")
    bars_per_3day = 3 * 24 * 12
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []; bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        h1_w = h1[st:en]; h4_w = h4[st:en]; v_w = valid[st:en]
        sigs_w = entry_stop_hunt(df_w, h1_w, h4_w, v_w)
        trades = run_bt_stop_hunt(df_w, sigs_w, friction_tp=0.04, friction_sl=0.07)
        cand_pnl = sum(t['net_pct'] for t in trades) if trades else 0
        cand_pnls.append(cand_pnl)
        bh = (df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07
        bh_pnls.append(bh)
    mean_p = float(np.mean(cand_pnls))
    pos_rate = float(np.mean(np.array(cand_pnls) > 0))
    p5 = float(np.percentile(cand_pnls, 5))
    p_better = float(np.mean(np.array(cand_pnls) > np.array(bh_pnls)))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_vs_BH={p_better:.4f}")

    # Look-ahead
    print(f"\n{'=' * 80}\nLook-ahead audit\n{'=' * 80}")
    audit_idx = random.sample([i for i, _ in sigs], min(20, len(sigs)))
    leaks = 0
    for i in audit_idx:
        df_t = df.iloc[:i+1].copy()
        h1_t = h1[:i+1]; h4_t = h4[:i+1]; v_t = valid[:i+1]
        try:
            t_sigs = entry_stop_hunt(df_t, h1_t, h4_t, v_t)
            full_at_i = next((d for idx, d in sigs if idx == i), None)
            trunc_at_i = next((d for idx, d in t_sigs if idx == i), None)
            if full_at_i != trunc_at_i:
                leaks += 1
        except Exception:
            leaks += 1
    print(f"  Audited {len(audit_idx)}, leaks: {leaks}")

    # WF 5-fold (overfit/regime)
    print(f"\n{'=' * 80}\nWF 5-fold (overfit/regime check)\n{'=' * 80}")
    fold_size = n_total // 6
    wf = []
    for fold_i in range(5):
        tr_e = (fold_i + 1) * fold_size
        te_s = tr_e
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        h1_f = h1[te_s:te_e]; h4_f = h4[te_s:te_e]; v_f = valid[te_s:te_e]
        sigs_f = entry_stop_hunt(df_f, h1_f, h4_f, v_f)
        trades = run_bt_stop_hunt(df_f, sigs_f, friction_tp=0.04, friction_sl=0.07)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i + 1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0,
                    'wr': s_f['wr_pct'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0}, daily={s_f['daily_net'] if s_f else 'N/A'}, WR={s_f['wr_pct'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)
    print(f"  WF positive folds: {wf_pos}/5")

    # 7-test verdict (scenario B = mixed primary)
    s_b = scenarios.get('B mixed (TP-mkr/SL-tkr)')
    cond = {
        'test1_lookahead': leaks == 0,
        'test2_wf_3of5': wf_pos >= 3,
        'test3a_taker_C': scenarios.get('C taker both') and scenarios['C taker both']['daily_net'] >= 0.2,
        'test3b_mixed_B': s_b is not None and s_b['daily_net'] >= 0.2,
        'test3c_maker_A': scenarios.get('A maker both') and scenarios['A maker both']['daily_net'] >= 0.3,
        'test4_bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1.0 and p_better >= 0.6,
        'test5_gross_vs_fee': s_b is not None and s_b['avg_gross'] >= 0.10,
        'test6_freq_2': s_b is not None and s_b['per_day'] >= 2.0,
        'test7_wr50_rr1': s_b is not None and s_b['wr_pct'] >= 50 and s_b['rr'] >= 1.0,
    }
    print(f"\n{'=' * 80}\nM3-R22 VERDICT (8 tests)\n{'=' * 80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(v for v in cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round22_stop_hunt.md',
           'n_signals': len(sigs),
           'scenarios': scenarios,
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'lookahead_leaks': leaks,
           'conditions': cond,
           'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r22_stop_hunt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
