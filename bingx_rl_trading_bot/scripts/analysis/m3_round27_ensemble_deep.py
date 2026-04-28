"""M3-R27 — R21+R24 ensemble + deep critique angles.

Combine best 2 mechanisms: pattern reversal at extreme + pullback continuation.
Either trigger fires → trade taken (with structural exit).
Plus deep critique: edge decay, per-direction, time-of-day, Sharpe ratio.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round20_dynamic_scalping import prepare_5m_data
from m3_round21_pattern_structure import (add_sma200_1h, entry_psi_prime, run_bt_structural)
from m3_round24_pullback_continuation import add_ema_indicators, entry_pullback


def entry_ensemble(df, h1, h4, valid, params=None):
    """Take entries from EITHER R21 (pattern reversal at extreme) OR R24 (pullback continuation)."""
    sigs_r21 = entry_psi_prime(df, h1, h4, valid)
    sigs_r24 = entry_pullback(df, h1, h4, valid)
    # Combine and dedupe (same i, same direction)
    all_sigs = list(set(sigs_r21 + sigs_r24))
    all_sigs.sort()
    return all_sigs


def run_bt_ensemble(df, sigs, friction_tp=0.04, friction_sl=0.07,
                     emergency_pct=1.0, timeout_bars=24, min_bars_between=2):
    """Use R21's structural exit logic for both entry types."""
    return run_bt_structural(df, sigs, friction_tp=friction_tp, friction_sl=friction_sl,
                              emergency_pct=emergency_pct, timeout_bars=timeout_bars,
                              min_bars_between=min_bars_between)


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
        'nets_array': nets, 'trades': trades,
    }


def per_direction_breakdown(trades):
    long_t = [t for t in trades if t['direction'] == 'LONG']
    short_t = [t for t in trades if t['direction'] == 'SHORT']
    out = {}
    for label, t_list in [('LONG', long_t), ('SHORT', short_t)]:
        if not t_list:
            out[label] = None; continue
        nets = [t['net_pct'] for t in t_list]
        wins = sum(1 for x in nets if x > 0)
        out[label] = {
            'n': len(t_list),
            'avg_net': round(sum(nets)/len(nets), 4),
            'wr_pct': round(100*wins/len(nets), 2),
            'sum_net': round(sum(nets), 2),
        }
    return out


def time_of_day_breakdown(trades):
    """Performance by hour-of-day."""
    by_hour = {}
    for t in trades:
        ts = pd.to_datetime(t['entry_ts'])
        h = ts.hour
        by_hour.setdefault(h, []).append(t['net_pct'])
    summary = {}
    for h in sorted(by_hour.keys()):
        arr = by_hour[h]
        wins = sum(1 for x in arr if x > 0)
        summary[h] = {'n': len(arr), 'avg_net': round(sum(arr)/len(arr), 4),
                       'wr_pct': round(100*wins/len(arr), 2),
                       'sum_net': round(sum(arr), 2)}
    return summary


def edge_decay_rolling(trades, window_days=60):
    """Rolling 60-day daily PnL — does edge decay over time?"""
    if len(trades) < 10: return None
    df_t = pd.DataFrame(trades)
    df_t['exit_dt'] = pd.to_datetime(df_t['exit_ts'])
    df_t = df_t.sort_values('exit_dt')
    # Daily PnL
    df_t['date'] = df_t['exit_dt'].dt.date
    daily = df_t.groupby('date')['net_pct'].sum()
    # Reindex to all dates
    all_dates = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(all_dates.date, fill_value=0)
    # Rolling sum
    rolling = daily.rolling(window_days, min_periods=window_days//2).mean()
    # Sample at quartiles
    q1_idx = len(rolling) // 4
    q2_idx = len(rolling) // 2
    q3_idx = 3 * len(rolling) // 4
    end_idx = len(rolling) - 1
    return {
        'q1_rolling_daily': float(rolling.iloc[q1_idx]) if q1_idx < len(rolling) else None,
        'q2_rolling_daily': float(rolling.iloc[q2_idx]) if q2_idx < len(rolling) else None,
        'q3_rolling_daily': float(rolling.iloc[q3_idx]) if q3_idx < len(rolling) else None,
        'final_rolling_daily': float(rolling.iloc[end_idx]) if end_idx < len(rolling) else None,
    }


def sharpe_drawdown(trades):
    """Sharpe annualized + max drawdown."""
    if len(trades) < 10: return None
    nets = [t['net_pct'] for t in trades]
    df_t = pd.DataFrame(trades)
    df_t['exit_dt'] = pd.to_datetime(df_t['exit_ts'])
    df_t = df_t.sort_values('exit_dt')
    df_t['date'] = df_t['exit_dt'].dt.date
    daily = df_t.groupby('date')['net_pct'].sum()
    all_dates = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(all_dates.date, fill_value=0)

    if daily.std() > 0:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(365)
    else:
        sharpe = 0

    # Max drawdown (cumulative equity)
    equity = (1 + daily / 100).cumprod()
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100
    max_dd = float(dd.min())
    return {'sharpe': float(sharpe), 'max_drawdown_pct': max_dd,
             'mean_daily': float(daily.mean()), 'std_daily': float(daily.std())}


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
    df, h1, h4, valid = prepare_5m_data()
    df = add_sma200_1h(df)
    df = add_ema_indicators(df)
    valid = valid & (~df['sma200_long'].isna()).values & (~pd.isna(df['ema20_5m']).values) & (~pd.isna(df['ema50_5m']).values)

    n_total = len(df)
    print(f"5m bars: {n_total:,} | days: {n_total/(24*12):.0f}")

    # Individual entries for comparison
    sigs_r21 = entry_psi_prime(df, h1, h4, valid)
    sigs_r24 = entry_pullback(df, h1, h4, valid)
    sigs_ens = entry_ensemble(df, h1, h4, valid)
    print(f"  R21 signals: {len(sigs_r21)}")
    print(f"  R24 signals: {len(sigs_r24)}")
    print(f"  Ensemble (union): {len(sigs_ens)} ({len(sigs_ens)/(n_total/(24*12)):.2f}/day)")
    overlap = len(set([(i, d) for i, d in sigs_r21]) & set([(i, d) for i, d in sigs_r24]))
    print(f"  Overlap: {overlap}\n")

    # Friction scenarios
    print(f"{'='*80}\nFriction scenarios (ensemble entries, structural exit)\n{'='*80}")
    print(f"{'scenario':<20} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}")
    scenarios = {}
    for label, ftp, fsl in [('A maker', 0.04, 0.04), ('B mixed', 0.04, 0.07),
                              ('C taker', 0.10, 0.10), ('D worst', 0.10, 0.15)]:
        trades = run_bt_ensemble(df, sigs_ens, friction_tp=ftp, friction_sl=fsl)
        s = trade_summary(trades)
        scenarios[label] = s
        if s:
            print(f"{label:<20} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%  reasons={s['reasons']}")

    s_b = scenarios.get('B mixed')
    if not s_b:
        print("No mixed scenario trades.")
        return

    # T-test
    print(f"\n{'='*80}\nStatistical significance (t-test on per-trade returns, mixed friction)\n{'='*80}")
    ttest = t_test(s_b['nets_array'])
    print(f"  mean: {ttest['mean']:+.6f}%, std: {ttest['std']:.4f}%, t-stat: {ttest['t_stat']:.3f}")
    print(f"  p (one-sided H1: mean>0): {ttest['p_one_sided']:.4f}, sig at 0.05: {ttest['sig_05']}")

    # Per-direction
    print(f"\n{'='*80}\nPer-direction breakdown (mixed friction)\n{'='*80}")
    pd_break = per_direction_breakdown(s_b['trades'])
    for d, info in pd_break.items():
        if info:
            print(f"  {d}: n={info['n']}, avg_net={info['avg_net']:+.4f}%, WR={info['wr_pct']}%, sum={info['sum_net']:+.2f}%")

    # Time-of-day
    print(f"\n{'='*80}\nTime-of-day decomposition (mixed friction)\n{'='*80}")
    tod = time_of_day_breakdown(s_b['trades'])
    print(f"  {'hour':>5} {'n':>5} {'avg_net':>10} {'WR':>6} {'sum_net':>10}")
    pos_hours = []
    for h, info in sorted(tod.items()):
        marker = '⭐' if info['avg_net'] > 0 else '  '
        print(f"  {h:>5} {info['n']:>5} {info['avg_net']:>+9.4f}% {info['wr_pct']:>5.1f}% {info['sum_net']:>+9.2f}%  {marker}")
        if info['avg_net'] > 0:
            pos_hours.append(h)
    print(f"  Positive hours: {pos_hours}")

    # Edge decay
    print(f"\n{'='*80}\nEdge decay (rolling 60-day daily PnL)\n{'='*80}")
    decay = edge_decay_rolling(s_b['trades'], window_days=60)
    if decay:
        print(f"  Q1 (early): {decay['q1_rolling_daily']:+.4f}%/day")
        print(f"  Q2 (mid): {decay['q2_rolling_daily']:+.4f}%/day")
        print(f"  Q3 (late-mid): {decay['q3_rolling_daily']:+.4f}%/day")
        print(f"  Final: {decay['final_rolling_daily']:+.4f}%/day")

    # Sharpe + DD
    print(f"\n{'='*80}\nSharpe + Drawdown\n{'='*80}")
    sd = sharpe_drawdown(s_b['trades'])
    if sd:
        print(f"  Sharpe (annualized): {sd['sharpe']:.3f}")
        print(f"  Max drawdown: {sd['max_drawdown_pct']:.2f}%")
        print(f"  Mean daily: {sd['mean_daily']:+.4f}%, Std daily: {sd['std_daily']:.4f}%")

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
        sigs_f = entry_ensemble(df_f, h1_f, h4_f, v_f)
        trades = run_bt_ensemble(df_f, sigs_f, friction_tp=0.04, friction_sl=0.07)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

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
        sigs_w = entry_ensemble(df_w, h1_w, h4_w, v_w)
        trades = run_bt_ensemble(df_w, sigs_w, friction_tp=0.04, friction_sl=0.07)
        cand_pnls.append(sum(t['net_pct'] for t in trades) if trades else 0)
        bh_pnls.append((df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07)
    mean_p = float(np.mean(cand_pnls))
    pos_rate = float(np.mean(np.array(cand_pnls) > 0))
    p5 = float(np.percentile(cand_pnls, 5))
    p_better = float(np.mean(np.array(cand_pnls) > np.array(bh_pnls)))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_vs_BH={p_better:.4f}")

    # Verdict
    cond = {
        'wf_3of5': wf_pos >= 3,
        'taker_C': scenarios.get('C taker') and scenarios['C taker']['daily_net'] >= 0.2,
        'mixed_B': s_b['daily_net'] >= 0.2,
        'maker_A': scenarios.get('A maker') and scenarios['A maker']['daily_net'] >= 0.3,
        'bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1 and p_better >= 0.6,
        'gross_ge_0.10': s_b['avg_gross'] >= 0.10,
        'freq_ge_2': s_b['per_day'] >= 2.0,
        'wr_rr': s_b['wr_pct'] >= 50 and s_b['rr'] >= 1.0,
        't_test_sig': ttest['sig_05'],
    }
    print(f"\n{'='*80}\nM3-R27 ENSEMBLE VERDICT\n{'='*80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'mechanism': 'R21+R24 ensemble',
           'n_signals': {'r21': len(sigs_r21), 'r24': len(sigs_r24), 'ensemble': len(sigs_ens), 'overlap': overlap},
           'scenarios_summary': {k: {kk: vv for kk, vv in v.items() if kk not in ('nets_array', 'trades')} if v else None
                                    for k, v in scenarios.items()},
           't_test': ttest,
           'per_direction': pd_break,
           'time_of_day': tod,
           'positive_hours': pos_hours,
           'edge_decay': decay,
           'sharpe_dd': sd,
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'conditions': cond,
           'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r27_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
