"""M3-R25 — R24 pullback continuation + TP lookback sweep.

R24 WR 47.8% best, R:R 0.69 (TP too close at 20-bar). Sweep TP lookback for balance.
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
from m3_round24_pullback_continuation import add_ema_indicators, entry_pullback


def run_bt_pullback_param_tp(df, sigs, tp_lookback=20, friction_tp=0.04, friction_sl=0.07,
                                emergency_pct=1.0, timeout_bars=24, min_bars_between=2):
    """Pullback BT with parameterized TP lookback."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
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
                    psl = lo[i] * 0.999
                    sw_hi = float(np.max(hi[max(0, i-tp_lookback):i+1]))
                    ptp = sw_hi
                    pemerg = pentry * (1 - emergency_pct / 100)
                    if not (psl < pentry < ptp):
                        i += 1; continue
                else:
                    psl = hi[i] * 1.001
                    sw_lo = float(np.min(lo[max(0, i-tp_lookback):i+1]))
                    ptp = sw_lo
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
        'sum_net': round(sum(nets), 2),
        'avg_net': round(sum(nets)/n, 4), 'avg_gross': round(sum(grosses)/n, 4),
        'wr_pct': round(100*wins/n, 2), 'rr': round(rr, 3),
        'daily_net': round(sum(nets)/days, 4),
        'reasons': reasons,
    }


def main():
    df, h1, h4, valid = prepare_5m_data()
    df = add_sma200_1h(df)
    df = add_ema_indicators(df)
    valid = valid & (~df['sma200_long'].isna()).values & (~pd.isna(df['ema20_5m']).values) & (~pd.isna(df['ema50_5m']).values)

    n_total = len(df)
    print(f"5m bars: {n_total:,} | days: {n_total/(24*12):.0f}")

    sigs = entry_pullback(df, h1, h4, valid)
    print(f"Pullback signals: {len(sigs)} → {len(sigs)/(n_total/(24*12)):.2f}/day\n")

    print("=" * 80); print("TP lookback sweep (mixed friction TP=0.04, SL=0.07)"); print("=" * 80)
    print(f"{'TP_LB':>6} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}  reasons")
    sweep_results = {}
    for tp_lb in (10, 15, 20, 30, 40, 60, 80, 100):
        trades = run_bt_pullback_param_tp(df, sigs, tp_lookback=tp_lb,
                                            friction_tp=0.04, friction_sl=0.07)
        s = trade_summary(trades)
        sweep_results[tp_lb] = s
        if s:
            print(f"{tp_lb:>6} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%  {s['reasons']}")

    # Find optimal TP_LB
    best_lb = max(sweep_results.keys(), key=lambda k: sweep_results[k]['daily_net'] if sweep_results[k] else -999)
    best = sweep_results[best_lb]
    print(f"\nBest TP_LB by daily_net: {best_lb}, daily={best['daily_net']:+.4f}%")

    # Run full 7-test on best TP_LB
    print(f"\n{'='*80}\nFull 7-test on TP_LB={best_lb}\n{'='*80}")

    # Friction scenarios
    print(f"\nFriction scenarios:")
    print(f"{'scenario':<20} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}")
    fric_scenarios = {}
    for label, ftp, fsl in [('A maker', 0.04, 0.04), ('B mixed', 0.04, 0.07),
                              ('C taker', 0.10, 0.10), ('D worst', 0.10, 0.15)]:
        trades = run_bt_pullback_param_tp(df, sigs, tp_lookback=best_lb,
                                            friction_tp=ftp, friction_sl=fsl)
        s = trade_summary(trades)
        fric_scenarios[label] = s
        if s:
            print(f"{label:<20} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%")

    # Bootstrap
    print(f"\nBootstrap 1000 × 3-day:")
    bars_per_3day = 3 * 24 * 12
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []; bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        h1_w = h1[st:en]; h4_w = h4[st:en]; v_w = valid[st:en]
        sigs_w = entry_pullback(df_w, h1_w, h4_w, v_w)
        trades = run_bt_pullback_param_tp(df_w, sigs_w, tp_lookback=best_lb,
                                            friction_tp=0.04, friction_sl=0.07)
        cand_pnls.append(sum(t['net'] for t in trades) if trades else 0)
        bh_pnls.append((df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07)
    mean_p = float(np.mean(cand_pnls))
    pos_rate = float(np.mean(np.array(cand_pnls) > 0))
    p5 = float(np.percentile(cand_pnls, 5))
    p_better = float(np.mean(np.array(cand_pnls) > np.array(bh_pnls)))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_vs_BH={p_better:.4f}")

    # WF 5-fold
    print(f"\nWF 5-fold (mixed friction):")
    fold_size = n_total // 6
    wf = []
    for fold_i in range(5):
        tr_e = (fold_i + 1) * fold_size
        te_s = tr_e
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        h1_f = h1[te_s:te_e]; h4_f = h4[te_s:te_e]; v_f = valid[te_s:te_e]
        sigs_f = entry_pullback(df_f, h1_f, h4_f, v_f)
        trades = run_bt_pullback_param_tp(df_f, sigs_f, tp_lookback=best_lb,
                                            friction_tp=0.04, friction_sl=0.07)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

    # 7-test verdict
    s_b = fric_scenarios.get('B mixed')
    cond = {
        'wf_3of5': wf_pos >= 3,
        'taker_C': fric_scenarios.get('C taker') and fric_scenarios['C taker']['daily_net'] >= 0.2,
        'mixed_B': s_b is not None and s_b['daily_net'] >= 0.2,
        'maker_A': fric_scenarios.get('A maker') and fric_scenarios['A maker']['daily_net'] >= 0.3,
        'bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1 and p_better >= 0.6,
        'gross_ge_0.10': s_b is not None and s_b['avg_gross'] >= 0.10,
        'freq_ge_2': s_b is not None and s_b['per_day'] >= 2.0,
        'wr_rr': s_b is not None and s_b['wr_pct'] >= 50 and s_b['rr'] >= 1.0,
    }
    print(f"\n{'='*80}\nM3-R25 VERDICT (TP_LB={best_lb})\n{'='*80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'sweep_results': {k: v for k, v in sweep_results.items()},
           'best_tp_lb': best_lb,
           'fric_scenarios': fric_scenarios,
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'conditions': cond,
           'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r25_pullback_tp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
