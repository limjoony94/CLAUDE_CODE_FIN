"""M3-R30 — C1 EXACT production spec re-validation.

After R29 + config inspection, exact spec from c1_breakout_config.yaml:
- max_sl_atr: 4.5 (NOT 3.3)
- sl_min_pct: 0.15, sl_max_pct: 3.0 (signal REJECTED if outside)
- trail_activation_pct: 0.05
- progressive_trail: enabled, threshold_pct=0.9, K_post=0.5
- emergency_sl_pct: 3.0
- max_hold_bars: 192 (48h on 15m)
- channel_period: 15, body_min_ratio: 0.4, trail_K: 2.5

User criterion: WR ≥ 40% (relaxed from 50%).
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round29_c1_exact_revalidation import prepare_15m_data


def entry_c1_production(df, valid, params=None):
    """C1 production spec: Donchian + body + body_direction + SL bounds rejection."""
    p = {
        'channel_period': 15,
        'body_min_ratio': 0.4,
        'max_sl_atr': 4.5,  # CORRECTED to production value
        'sl_min_pct': 0.15,
        'sl_max_pct': 3.0,
    } if params is None else params
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    atr = df['atr14_15m'].values
    high_prev = df['high_15_prev'].values
    low_prev = df['low_15_prev'].values
    sw_low = df['swing_low_10'].values
    sw_high = df['swing_high_10'].values

    sigs = []
    for i in range(p['channel_period'] + 2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (cl[i], op[i], hi[i], lo[i], high_prev[i], low_prev[i], atr[i])):
            continue
        if atr[i] <= 0: continue

        # Channel breakout
        long_break = cl[i] > high_prev[i]
        short_break = cl[i] < low_prev[i]
        if not (long_break or short_break): continue

        # Body filter
        body = cl[i] - op[i]
        rng = hi[i] - lo[i]
        if rng <= 0: continue
        if abs(body) / rng < p['body_min_ratio']: continue

        # Body direction matches breakout
        if long_break and body <= 0: continue
        if short_break and body >= 0: continue

        direction = 'LONG' if long_break else 'SHORT'

        # Compute SL price (fractal + ATR cap)
        entry_approx = cl[i]  # Will be next-bar open in BT, approximated
        if direction == 'LONG':
            fractal_sl = sw_low[i] if not pd.isna(sw_low[i]) else entry_approx - p['max_sl_atr'] * atr[i]
            atr_sl = entry_approx - p['max_sl_atr'] * atr[i]
            sl_price = max(fractal_sl, atr_sl)
        else:
            fractal_sl = sw_high[i] if not pd.isna(sw_high[i]) else entry_approx + p['max_sl_atr'] * atr[i]
            atr_sl = entry_approx + p['max_sl_atr'] * atr[i]
            sl_price = min(fractal_sl, atr_sl)

        # SL bounds check (PRODUCTION REJECTS signals outside bounds)
        sl_pct = abs(entry_approx - sl_price) / entry_approx * 100
        if sl_pct < p['sl_min_pct'] or sl_pct > p['sl_max_pct']:
            continue

        sigs.append((i, direction))
    return sigs


def run_bt_c1_production(df, sigs, friction=0.10,
                            trail_K=2.5, max_sl_atr=4.5, sl_min_pct=0.15, sl_max_pct=3.0,
                            emergency_sl_pct=3.0, max_hold_bars=192,
                            trail_activation_pct=0.05,
                            prog_trail_enabled=True, prog_trail_threshold=0.9, prog_trail_K_post=0.5,
                            min_bars_between=2):
    """C1 BT with progressive_trail enabled, SL bounds, trail activation."""
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
    i = 0
    while i < n:
        if in_pos:
            atr_now = atr[i] if not np.isnan(atr[i]) else patr_e
            # Update best
            if pdir == 'LONG':
                pbest = max(pbest, hi[i])
            else:
                pbest = min(pbest, lo[i])

            # Compute current best_pnl in %
            best_pnl_pct = ((pbest / pentry - 1) * 100) if pdir == 'LONG' else ((1 - pbest / pentry) * 100)

            # Determine effective trail K (progressive_trail)
            if prog_trail_enabled and best_pnl_pct >= prog_trail_threshold:
                effective_K = prog_trail_K_post
            else:
                effective_K = trail_K

            # Trail activation gate (only trail after best_pnl > activation_pct)
            trail_active = best_pnl_pct >= trail_activation_pct

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
            # Trail TP (only if active)
            if exit_price is None and trail_active:
                if pdir == 'LONG':
                    trigger = pbest - effective_K * atr_now
                    if lo[i] <= trigger and trigger > pentry:
                        exit_price, exit_reason = trigger, 'TRAIL_TP'
                else:
                    trigger = pbest + effective_K * atr_now
                    if hi[i] >= trigger and trigger < pentry:
                        exit_price, exit_reason = trigger, 'TRAIL_TP'
            # Timeout
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
                pentry = op[ni]
                pdir = sig_set[i]
                patr_e = atr[i] if not np.isnan(atr[i]) else 0
                if pdir == 'LONG':
                    structural = sw_low[i] if not np.isnan(sw_low[i]) else pentry - max_sl_atr * patr_e
                    atr_sl = pentry - max_sl_atr * patr_e
                    psl = max(structural, atr_sl)
                    pemerg = pentry * (1 - emergency_sl_pct / 100)
                    pbest = hi[ni]
                else:
                    structural = sw_high[i] if not np.isnan(sw_high[i]) else pentry + max_sl_atr * patr_e
                    atr_sl = pentry + max_sl_atr * patr_e
                    psl = min(structural, atr_sl)
                    pemerg = pentry * (1 + emergency_sl_pct / 100)
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


def main():
    df, valid = prepare_15m_data()
    n_total = len(df)
    print(f"15m bars: {n_total:,} | days: {n_total/96:.0f}")

    sigs = entry_c1_production(df, valid)
    print(f"C1 PRODUCTION signals: {len(sigs)} → {len(sigs)/(n_total/96):.2f}/day\n")

    if len(sigs) == 0:
        print("No signals.")
        return

    # Friction scenarios
    print(f"{'='*80}\nC1 PRODUCTION (max_sl_atr=4.5, progressive_trail=ON, sl_bounds, trail_activation)\n{'='*80}")
    print(f"{'scenario':<20} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}  reasons")
    scenarios = {}
    for label, fric in [('A maker (0.04)', 0.04), ('B mixed (0.07)', 0.07), ('C taker (0.10)', 0.10), ('D worst (0.15)', 0.15)]:
        trades = run_bt_c1_production(df, sigs, friction=fric)
        s = trade_summary(trades)
        scenarios[label] = s
        if s:
            print(f"{label:<20} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%  {s['reasons']}")

    s_b = scenarios.get('B mixed (0.07)')
    if not s_b: return

    # 333-day window check (CLAUDE.md claim)
    print(f"\n{'='*80}\n333-day window check (CLAUDE.md C1 BT period)\n{'='*80}")
    bars_333 = 333 * 96
    if n_total > bars_333:
        # Take last 333 days (most recent — most likely period of original BT)
        df_333 = df.iloc[-bars_333:].reset_index(drop=True)
        v_333 = valid[-bars_333:]
        sigs_333 = entry_c1_production(df_333, v_333)
        trades_333 = run_bt_c1_production(df_333, sigs_333, friction=0.07)
        s_333 = trade_summary(trades_333)
        print(f"  Last 333 days (most recent): n={s_333['n']}, daily={s_333['daily_net']:+.4f}%, WR={s_333['wr_pct']}%, RR={s_333['rr']}, avg_g={s_333['avg_gross']:+.4f}%")
        # Try first 333 days
        df_333_f = df.iloc[:bars_333].reset_index(drop=True)
        v_333_f = valid[:bars_333]
        sigs_333_f = entry_c1_production(df_333_f, v_333_f)
        trades_333_f = run_bt_c1_production(df_333_f, sigs_333_f, friction=0.07)
        s_333_f = trade_summary(trades_333_f)
        print(f"  First 333 days: n={s_333_f['n']}, daily={s_333_f['daily_net']:+.4f}%, WR={s_333_f['wr_pct']}%, RR={s_333_f['rr']}, avg_g={s_333_f['avg_gross']:+.4f}%")
        # Mid 333 days
        mid_start = (n_total - bars_333) // 2
        df_333_m = df.iloc[mid_start:mid_start+bars_333].reset_index(drop=True)
        v_333_m = valid[mid_start:mid_start+bars_333]
        sigs_333_m = entry_c1_production(df_333_m, v_333_m)
        trades_333_m = run_bt_c1_production(df_333_m, sigs_333_m, friction=0.07)
        s_333_m = trade_summary(trades_333_m)
        print(f"  Mid 333 days: n={s_333_m['n']}, daily={s_333_m['daily_net']:+.4f}%, WR={s_333_m['wr_pct']}%, RR={s_333_m['rr']}, avg_g={s_333_m['avg_gross']:+.4f}%")

    # Bootstrap 1000 × 3-day
    print(f"\n{'='*80}\nBootstrap 1000 × 3-day windows\n{'='*80}")
    bars_per_3day = 3 * 96
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []; bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        v_w = valid[st:en]
        sigs_w = entry_c1_production(df_w, v_w)
        trades = run_bt_c1_production(df_w, sigs_w, friction=0.07)
        cand_pnls.append(sum(t['net_pct'] for t in trades) if trades else 0)
        bh_pnls.append((df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07)
    mean_p = float(np.mean(cand_pnls))
    pos_rate = float(np.mean(np.array(cand_pnls) > 0))
    p5 = float(np.percentile(cand_pnls, 5))
    p_better = float(np.mean(np.array(cand_pnls) > np.array(bh_pnls)))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_vs_BH={p_better:.4f}")

    # WF 5-fold
    print(f"\n{'='*80}\nWF 5-fold (mixed)\n{'='*80}")
    fold_size = n_total // 6
    wf = []
    for fold_i in range(5):
        tr_e = (fold_i + 1) * fold_size
        te_s = tr_e
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid[te_s:te_e]
        sigs_f = entry_c1_production(df_f, v_f)
        trades = run_bt_c1_production(df_f, sigs_f, friction=0.07)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None,
                    'rr': s_f['rr'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'} RR={s_f['rr'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

    # Verdict
    cond = {
        'wf_3of5': wf_pos >= 3,
        'taker_C': scenarios.get('C taker (0.10)') and scenarios['C taker (0.10)']['daily_net'] >= 0.2,
        'mixed_B': s_b['daily_net'] >= 0.2,
        'maker_A': scenarios.get('A maker (0.04)') and scenarios['A maker (0.04)']['daily_net'] >= 0.3,
        'bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1 and p_better >= 0.6,
        'gross_ge_0.10': s_b['avg_gross'] >= 0.10,
        'freq_ge_2': s_b['per_day'] >= 2.0,
        'wr_ge_40': s_b['wr_pct'] >= 40,  # RELAXED
        'rr_ge_1': s_b['rr'] >= 1.0,
    }
    print(f"\n{'='*80}\nM3-R30 VERDICT (WR ≥ 40% relaxed)\n{'='*80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'mechanism': 'C1 PRODUCTION exact (max_sl_atr=4.5, prog_trail ON, sl_bounds, trail_activation)',
           'n_signals': len(sigs),
           'scenarios': {k: {kk: vv for kk, vv in v.items() if kk != 'trades'} if v else None for k, v in scenarios.items()},
           'window_333d_recent': {kk: vv for kk, vv in s_333.items() if kk != 'trades'} if 's_333' in dir() and s_333 else None,
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'conditions': cond, 'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r30_c1_production_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
