"""M3-R31 — Apply C1 production exit to R21 + R24 + R30 (timeframe-aware).

Test if production-proven exit (progressive_trail + SL bounds + trail activation + wider SL)
improves my best mechanism entries.

Critical: ATR scale differs between 5m and 15m. Use ATR-relative parameters.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round20_dynamic_scalping import prepare_5m_data
from m3_round21_pattern_structure import (add_sma200_1h, entry_psi_prime)
from m3_round24_pullback_continuation import add_ema_indicators, entry_pullback


def run_bt_production_style(df, sigs, friction=0.10,
                              trail_K=2.5, max_sl_atr=4.5,
                              sl_min_pct=0.15, sl_max_pct=3.0,
                              emergency_sl_pct=3.0, max_hold_bars=192,
                              trail_activation_pct=0.05,
                              prog_trail_enabled=True, prog_trail_threshold=0.9, prog_trail_K_post=0.5,
                              min_bars_between=2,
                              atr_col='atr14_5m'):
    """Production C1-style exit for any entry signals.

    Key features:
    - Wider SL (max_sl_atr 4.5)
    - SL bounds rejection (signals where sl_pct outside [min, max] are skipped at entry)
    - Trail activation (no trail before profit threshold)
    - Progressive trail (tight K after profit threshold)
    - Emergency 3.0%
    - Long timeout (192 bars on 15m = 48h, on 5m = 16h)
    """
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    atr = df[atr_col].values
    timestamps = df['timestamp'].values

    # Compute swing low/high (10-bar) at each bar for SL placement
    sw_low = pd.Series(lo).rolling(10, min_periods=10).min().shift(1).values
    sw_high = pd.Series(hi).rolling(10, min_periods=10).max().shift(1).values

    sig_set = {idx: d for idx, d in sigs}

    in_pos = False
    pdir = None; pentry = None; psl = None; pemerg = None
    pbest = None; pstart = None; patr_e = None
    cooldown = 0
    trades = []
    rejected_by_sl_bounds = 0
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
                pentry_candidate = op[ni]
                pdir_candidate = sig_set[i]
                patr_candidate = atr[i] if not np.isnan(atr[i]) else 0

                # SL placement
                if pdir_candidate == 'LONG':
                    structural = sw_low[i] if not np.isnan(sw_low[i]) else pentry_candidate - max_sl_atr * patr_candidate
                    atr_sl = pentry_candidate - max_sl_atr * patr_candidate
                    psl_candidate = max(structural, atr_sl)
                else:
                    structural = sw_high[i] if not np.isnan(sw_high[i]) else pentry_candidate + max_sl_atr * patr_candidate
                    atr_sl = pentry_candidate + max_sl_atr * patr_candidate
                    psl_candidate = min(structural, atr_sl)

                # SL bounds check
                sl_pct_candidate = abs(pentry_candidate - psl_candidate) / pentry_candidate * 100
                if sl_pct_candidate < sl_min_pct or sl_pct_candidate > sl_max_pct:
                    rejected_by_sl_bounds += 1
                    i += 1
                    continue

                pentry = pentry_candidate
                pdir = pdir_candidate
                patr_e = patr_candidate
                psl = psl_candidate
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
    return trades, rejected_by_sl_bounds


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
    df, h1, h4, valid = prepare_5m_data()
    df = add_sma200_1h(df)
    df = add_ema_indicators(df)
    valid = valid & (~df['sma200_long'].isna()).values & (~pd.isna(df['ema20_5m']).values) & (~pd.isna(df['ema50_5m']).values)

    n_total = len(df)
    print(f"5m bars: {n_total:,} | days: {n_total/(24*12):.0f}\n")

    # 5m data: max_hold_bars=192 corresponds to 16h (vs 48h on 15m)
    # Adjust to 5m: 48h = 576 bars on 5m, but R21/R24 used 24 bars timeout. Try 192 (16h) to match production scale.
    max_hold_5m = 192  # 16h on 5m

    sigs_r21 = entry_psi_prime(df, h1, h4, valid)
    sigs_r24 = entry_pullback(df, h1, h4, valid)
    print(f"R21 (pattern reversal at extreme) signals: {len(sigs_r21)}")
    print(f"R24 (pullback continuation) signals: {len(sigs_r24)}\n")

    # Test 4 configurations:
    # 1. R21 entry + production exit
    # 2. R24 entry + production exit
    # 3. R21 entry + production exit (NO progressive_trail) — to isolate prog_trail effect
    # 4. R24 entry + production exit (NO progressive_trail)
    print(f"{'='*100}")
    print(f"4 mechanism × exit configurations (mixed friction 0.07)")
    print(f"{'='*100}")
    print(f"{'config':<55} {'n':>5} {'rej_sl':>7} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}")

    configs = [
        ('R21 + prod exit (prog_trail ON)', sigs_r21, True),
        ('R21 + prod exit (prog_trail OFF)', sigs_r21, False),
        ('R24 + prod exit (prog_trail ON)', sigs_r24, True),
        ('R24 + prod exit (prog_trail OFF)', sigs_r24, False),
    ]

    results = {}
    for label, sigs, prog_on in configs:
        trades, rej = run_bt_production_style(
            df, sigs, friction=0.07,
            trail_K=2.5, max_sl_atr=4.5,
            sl_min_pct=0.15, sl_max_pct=3.0,
            emergency_sl_pct=3.0, max_hold_bars=max_hold_5m,
            trail_activation_pct=0.05,
            prog_trail_enabled=prog_on, prog_trail_threshold=0.9, prog_trail_K_post=0.5,
            atr_col='atr14_5m',
        )
        s = trade_summary(trades)
        results[label] = (s, rej)
        if s:
            print(f"{label:<55} {s['n']:>5} {rej:>7} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%")

    # Pick best by daily_net for full critique
    valid_results = {k: v[0] for k, v in results.items() if v[0] is not None}
    if not valid_results: return
    best_label = max(valid_results.keys(), key=lambda k: valid_results[k]['daily_net'])
    best_summary = valid_results[best_label]
    print(f"\nBest config: {best_label}, daily={best_summary['daily_net']:+.4f}%")

    # Friction sweep on best
    print(f"\n{'='*80}\nFriction scenarios on {best_label}\n{'='*80}")
    print(f"{'scenario':<20} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}")
    best_sigs = sigs_r21 if 'R21' in best_label else sigs_r24
    best_prog = 'ON' in best_label
    fric_scenarios = {}
    for label, fric in [('A maker', 0.04), ('B mixed', 0.07), ('C taker', 0.10), ('D worst', 0.15)]:
        trades, _ = run_bt_production_style(
            df, best_sigs, friction=fric,
            max_hold_bars=max_hold_5m, prog_trail_enabled=best_prog, atr_col='atr14_5m')
        s = trade_summary(trades)
        fric_scenarios[label] = s
        if s:
            print(f"{label:<20} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%")

    # Bootstrap 1000 × 3-day on best
    print(f"\n{'='*80}\nBootstrap 1000 × 3-day on {best_label}\n{'='*80}")
    bars_per_3day = 3 * 24 * 12
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []; bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        h1_w = h1[st:en]; h4_w = h4[st:en]; v_w = valid[st:en]
        sigs_w = entry_psi_prime(df_w, h1_w, h4_w, v_w) if 'R21' in best_label else entry_pullback(df_w, h1_w, h4_w, v_w)
        trades, _ = run_bt_production_style(
            df_w, sigs_w, friction=0.07, max_hold_bars=max_hold_5m, prog_trail_enabled=best_prog, atr_col='atr14_5m')
        cand_pnls.append(sum(t['net_pct'] for t in trades) if trades else 0)
        bh_pnls.append((df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07)
    mean_p = float(np.mean(cand_pnls))
    pos_rate = float(np.mean(np.array(cand_pnls) > 0))
    p5 = float(np.percentile(cand_pnls, 5))
    p_better = float(np.mean(np.array(cand_pnls) > np.array(bh_pnls)))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_vs_BH={p_better:.4f}")

    # WF 5-fold on best
    print(f"\n{'='*80}\nWF 5-fold on {best_label}\n{'='*80}")
    fold_size = n_total // 6
    wf = []
    for fold_i in range(5):
        tr_e = (fold_i + 1) * fold_size
        te_s = tr_e
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        h1_f = h1[te_s:te_e]; h4_f = h4[te_s:te_e]; v_f = valid[te_s:te_e]
        sigs_f = entry_psi_prime(df_f, h1_f, h4_f, v_f) if 'R21' in best_label else entry_pullback(df_f, h1_f, h4_f, v_f)
        trades, _ = run_bt_production_style(
            df_f, sigs_f, friction=0.07, max_hold_bars=max_hold_5m, prog_trail_enabled=best_prog, atr_col='atr14_5m')
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None,
                    'rr': s_f['rr'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'} RR={s_f['rr'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

    # Verdict
    s_b = fric_scenarios.get('B mixed')
    cond = {
        'wf_3of5': wf_pos >= 3,
        'taker_C_daily_positive': fric_scenarios.get('C taker') and fric_scenarios['C taker']['daily_net'] >= 0.2,
        'mixed_B_daily_positive': s_b is not None and s_b['daily_net'] >= 0.2,
        'maker_A_daily': fric_scenarios.get('A maker') and fric_scenarios['A maker']['daily_net'] >= 0.3,
        'bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1 and p_better >= 0.6,
        'gross_ge_0.10': s_b is not None and s_b['avg_gross'] >= 0.10,
        'freq_ge_2': s_b is not None and s_b['per_day'] >= 2.0,
        'wr_ge_40': s_b is not None and s_b['wr_pct'] >= 40,
        'rr_ge_1': s_b is not None and s_b['rr'] >= 1.0,
    }
    print(f"\n{'='*80}\nM3-R31 VERDICT (best: {best_label})\n{'='*80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'all_4_configs': {k: ({kk: vv for kk, vv in v[0].items() if kk != 'trades'} if v[0] else None, v[1])
                                for k, v in results.items()},
           'best_config': best_label,
           'best_friction_scenarios': {k: ({kk: vv for kk, vv in v.items() if kk != 'trades'} if v else None) for k, v in fric_scenarios.items()},
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'conditions': cond, 'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r31_prod_integration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
