"""M3-R24 — Trend Pullback Continuation (MA bounce).

In strong trend, price pulls back to 5m EMA20, forms reversal candle at EMA, then continues.
This is professional "MA bounce" / "pullback entry" — opposite direction from R21-R23 mean-rev.

Same 7-test suite per user spec.
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
from m3_round20_dynamic_scalping import prepare_5m_data
from m3_round21_pattern_structure import add_sma200_1h, detect_engulfing, detect_hammer


def add_ema_indicators(df):
    """Add 5m EMA20 + 15m EMA50 alignment."""
    cl = df['close'].values
    df['ema20_5m'] = compute_ema(cl, 20)
    df['ema50_5m'] = compute_ema(cl, 50)
    return df


def entry_pullback(df, h1, h4, valid, params=None):
    """Pullback to 5m EMA20 in trending market + reversal candle at EMA."""
    p = {
        'ema_dist_pct': 0.20,  # within 0.20% of EMA20
        'volume_mult': 1.0,
        'lookback_extreme': 20,
    } if params is None else params
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vol = df['volume'].values
    vol_sma = df['volume_sma20'].values
    ema20 = df['ema20_5m'].values
    ema50 = df['ema50_5m'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values

    sigs = []
    for i in range(p['lookback_extreme'] + 2, n):
        if not valid[i]: continue
        if pd.isna(ema20[i]) or pd.isna(ema50[i]): continue
        if pd.isna(vol[i]) or pd.isna(vol_sma[i]): continue
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue

        # Trend direction (5m level using EMA20 vs EMA50)
        ema_long = ema20[i] > ema50[i]
        ema_short = ema20[i] < ema50[i]

        # Pullback: low[i] within ema_dist_pct of EMA20 (touched EMA from above for LONG)
        ema = ema20[i]
        long_pullback = (lo[i] <= ema * (1 + p['ema_dist_pct']/100)) and (cl[i] > ema)
        short_pullback = (hi[i] >= ema * (1 - p['ema_dist_pct']/100)) and (cl[i] < ema)

        # Reversal candle at EMA
        bull_eng, bear_eng = detect_engulfing(op, cl, i)
        hammer, star = detect_hammer(op, cl, hi, lo, i)

        # Trend confirm: 1h SMA200 alignment
        # LONG: 5m EMA20 > EMA50, 1h trend long, pullback to EMA, bullish reversal
        if ema_long and sma_long[i] and h1[i] and long_pullback and (bull_eng or hammer):
            sigs.append((i, 'LONG'))
        elif ema_short and (not sma_long[i]) and (not h1[i]) and short_pullback and (bear_eng or star):
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_pullback(df, sigs, friction_tp=0.04, friction_sl=0.07,
                     emergency_pct=1.0, timeout_bars=24, min_bars_between=2):
    """Exit: TP at recent swing high (LONG) / low (SHORT). SL at pullback low/high."""
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
                # SL at pullback extreme: low[i] for LONG, high[i] for SHORT
                if pdir == 'LONG':
                    psl = lo[i] * 0.999
                    # TP at recent swing high (20-bar lookback excluding current)
                    sw_hi = float(np.max(hi[max(0, i-20):i+1]))
                    ptp = sw_hi
                    pemerg = pentry * (1 - emergency_pct / 100)
                    if not (psl < pentry < ptp):
                        i += 1; continue
                else:
                    psl = hi[i] * 1.001
                    sw_lo = float(np.min(lo[max(0, i-20):i+1]))
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
    nets = [t['net_pct'] for t in trades]
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
        'avg_net': round(sum(nets)/n, 4), 'avg_gross': round(sum(t['gross_pct'] for t in trades)/n, 4),
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
    print(f"Pullback continuation signals: {len(sigs)} → {len(sigs)/(n_total/(24*12)):.2f}/day")

    if len(sigs) == 0:
        print("No signals.")
        return

    print(f"\n{'='*80}\nFriction scenarios\n{'='*80}")
    print(f"{'scenario':<20} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10}")
    scenarios = {}
    for label, ftp, fsl in [('A maker', 0.04, 0.04), ('B mixed', 0.04, 0.07),
                              ('C taker', 0.10, 0.10), ('D worst', 0.10, 0.15)]:
        trades = run_bt_pullback(df, sigs, friction_tp=ftp, friction_sl=fsl)
        s = trade_summary(trades)
        scenarios[label] = s
        if s:
            print(f"{label:<20} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}%  reasons={s['reasons']}")

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
        sigs_w = entry_pullback(df_w, h1_w, h4_w, v_w)
        trades = run_bt_pullback(df_w, sigs_w, friction_tp=0.04, friction_sl=0.07)
        cand_pnls.append(sum(t['net_pct'] for t in trades) if trades else 0)
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
            t_sigs = entry_pullback(df_t, h1_t, h4_t, v_t)
            full_at_i = next((d for idx, d in sigs if idx == i), None)
            trunc_at_i = next((d for idx, d in t_sigs if idx == i), None)
            if full_at_i != trunc_at_i:
                leaks += 1
        except Exception:
            leaks += 1
    print(f"\n  Look-ahead leaks: {leaks}/{len(audit_idx)}")

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
        sigs_f = entry_pullback(df_f, h1_f, h4_f, v_f)
        trades = run_bt_pullback(df_f, sigs_f, friction_tp=0.04, friction_sl=0.07)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i+1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0, 'wr': s_f['wr_pct'] if s_f else None})
        print(f"  fold {fold_i+1}: n={s_f['n'] if s_f else 0} daily={s_f['daily_net'] if s_f else 'N/A'} WR={s_f['wr_pct'] if s_f else 'N/A'}")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)

    # Verdict
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
    print(f"\n{'='*80}\nM3-R24 VERDICT\n{'='*80}")
    for k, v in cond.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(cond.values())
    print(f"\n  ALL PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'mechanism': 'pullback continuation MA bounce',
           'n_signals': len(sigs),
           'scenarios': scenarios,
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better},
           'wf': {'folds': wf, 'pos_count': wf_pos},
           'lookahead_leaks': leaks,
           'conditions': cond,
           'all_pass': bool(all_pass)}
    p = ROOT / 'results' / f'm3_r24_pullback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
