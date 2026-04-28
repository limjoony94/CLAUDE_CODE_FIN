"""M3-R21 — ψ': Pattern reversal entry + structure-based dynamic exit.

Pre-reg: claudedocs/m3_round21_structure_pattern.md
Different from R20: structural SL/TP (swing points), not ATR-based.
Pattern entry: bullish/bearish engulfing or hammer/shooting star at swing extreme.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round20_dynamic_scalping import prepare_5m_data


def detect_engulfing(op, cl, i):
    """Bullish engulfing at i: bull bar engulfs prev bear body."""
    bull_engulf = (cl[i] > op[i]) and (op[i-1] > cl[i-1]) and (cl[i] > op[i-1]) and (op[i] < cl[i-1])
    bear_engulf = (op[i] > cl[i]) and (cl[i-1] > op[i-1]) and (op[i] > cl[i-1]) and (cl[i] < op[i-1])
    return bull_engulf, bear_engulf


def detect_hammer(op, cl, hi, lo, i):
    """Hammer: long lower wick, small body at top of range."""
    body = abs(cl[i] - op[i])
    rng = hi[i] - lo[i]
    if rng <= 0: return False, False
    body_mid = (op[i] + cl[i]) / 2
    lower_wick = min(op[i], cl[i]) - lo[i]
    upper_wick = hi[i] - max(op[i], cl[i])
    # Hammer: lower wick ≥ 2× body AND close in upper 50% of range
    hammer = (lower_wick >= 2 * body) and ((body_mid - lo[i]) / rng > 0.5) and (rng > 0)
    # Shooting star (inverted hammer): upper wick ≥ 2× body AND close in lower 50%
    star = (upper_wick >= 2 * body) and ((body_mid - lo[i]) / rng < 0.5) and (rng > 0)
    return hammer, star


# Add SMA200 1h trend filter
def add_sma200_1h(df):
    """Add 200-bar 1h SMA for less restrictive trend filter."""
    df_1h = df.set_index('timestamp').resample('1h', label='left', closed='left').agg({
        'close': 'last'}).dropna().reset_index()
    df_1h['sma200_1h'] = pd.Series(df_1h['close'].values).rolling(200, min_periods=200).mean().values
    df_1h['close_above_sma200'] = df_1h['close'] > df_1h['sma200_1h']
    df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=5)
    from m2_round1_screening import merge_htf
    df_1h_merge = df_1h.rename(columns={'close_above_sma200': 'sma200_long'})[['timestamp', 'sma200_long']]
    df = merge_htf(df, df_1h_merge, 60, ['sma200_long'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def entry_psi_prime(df, h1, h4, valid, params=None):
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

        # Recent extreme: low[i-1] or low[i-2] = recent 20-bar low
        recent_lows = lo[i - p['lookback_extreme']:i]
        recent_highs = hi[i - p['lookback_extreme']:i]
        recent_min = np.min(recent_lows)
        recent_max = np.max(recent_highs)
        low_touched = (lo[i-1] == recent_min) or (lo[i-2] == recent_min)
        high_touched = (hi[i-1] == recent_max) or (hi[i-2] == recent_max)

        # Reversal patterns at i
        bull_eng, bear_eng = detect_engulfing(op, cl, i)
        hammer, star = detect_hammer(op, cl, hi, lo, i)

        # LONG: low touched + bullish reversal pattern + 1h trend permissive (above 200-SMA)
        if low_touched and (bull_eng or hammer) and sma_long[i]:
            sigs.append((i, 'LONG'))
        # SHORT: high touched + bearish reversal + 1h below SMA
        elif high_touched and (bear_eng or star) and (not sma_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ==================== STRUCTURE-BASED DYNAMIC EXIT ====================

def find_recent_swing_low(lows, idx, lookback=10):
    """Most recent swing low: min of past lookback bars."""
    start = max(0, idx - lookback)
    return float(np.min(lows[start:idx + 1])) if start <= idx else lows[idx]


def find_recent_swing_high(highs, idx, lookback=10):
    start = max(0, idx - lookback)
    return float(np.max(highs[start:idx + 1])) if start <= idx else highs[idx]


def find_resistance_levels(highs, idx, lookbacks=(20, 50)):
    """Two resistance levels above entry."""
    return [float(np.max(highs[max(0, idx-lb):idx+1])) for lb in lookbacks]


def find_support_levels(lows, idx, lookbacks=(20, 50)):
    return [float(np.min(lows[max(0, idx-lb):idx+1])) for lb in lookbacks]


def run_bt_structural(df, sigs, friction_tp=0.04, friction_sl=0.07,
                       emergency_pct=1.0, timeout_bars=24, min_bars_between=2):
    """Structure-based exit: tight swing SL, multi-level TP, trail to BE+ after TP1."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}

    in_pos = False
    pdir = None; pentry = None; psl = None; ptp1 = None; ptp2 = None; pemerg = None
    pstart = None; tp1_hit = False
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

            # TP1 (partial — for simplicity, full exit; in real bot would split)
            if exit_price is None and not tp1_hit:
                if pdir == 'LONG' and hi[i] >= ptp1:
                    # Trail SL to entry + 0.05% (BE+) after TP1 hit
                    psl = max(psl, pentry * 1.0005)
                    tp1_hit = True
                elif pdir == 'SHORT' and lo[i] <= ptp1:
                    psl = min(psl, pentry * 0.9995)
                    tp1_hit = True

            # TP2 (final)
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
                # Friction model: TP exit = maker (0.04 RT total), SL/EMERG = mixed (0.07 RT)
                if exit_reason in ('TP1', 'TP2'):
                    fric = friction_tp
                elif exit_reason in ('SL', 'EMERGENCY'):
                    fric = friction_sl
                else:  # TIMEOUT (mixed, conservative)
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
                # Structural SL: recent swing low/high (10-bar)
                if pdir == 'LONG':
                    swing_low = find_recent_swing_low(lo, i, lookback=10)
                    psl = swing_low * 0.9995  # 0.05% buffer below swing
                    res_levels = find_resistance_levels(hi, i)
                    ptp1 = res_levels[0]  # nearest 20-bar high
                    ptp2 = res_levels[1]  # 50-bar high (further)
                    pemerg = pentry * (1 - emergency_pct / 100)
                else:
                    swing_high = find_recent_swing_high(hi, i, lookback=10)
                    psl = swing_high * 1.0005
                    sup_levels = find_support_levels(lo, i)
                    ptp1 = sup_levels[0]
                    ptp2 = sup_levels[1]
                    pemerg = pentry * (1 + emergency_pct / 100)

                # Sanity: TP must be on right side of entry, SL on opposite
                if pdir == 'LONG':
                    if not (ptp1 > pentry and ptp2 > pentry and psl < pentry):
                        # Skip if structure doesn't make sense
                        i += 1; continue
                else:
                    if not (ptp1 < pentry and ptp2 < pentry and psl > pentry):
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
    # Exit reason breakdown
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

    sigs = entry_psi_prime(df, h1, h4, valid)
    print(f"ψ' pattern reversal signals: {len(sigs)} → {len(sigs)/(n_total/(24*12)):.2f}/day\n")

    if len(sigs) == 0:
        print("No signals. Drop spec.")
        return

    # Friction scenarios (different exit-type-aware friction model)
    print("=" * 80); print("Friction scenarios (exit-type-aware)"); print("=" * 80)
    print(f"{'scenario':<25} {'n':>5} {'per_day':>8} {'daily':>10} {'WR':>6} {'RR':>6} {'avg_g':>10} reasons")
    for label, ftp, fsl in [
        ('A maker TP/maker SL', 0.04, 0.04),
        ('B maker TP/taker SL', 0.04, 0.07),
        ('C taker both', 0.10, 0.10),
        ('D worst-case', 0.10, 0.15),
    ]:
        trades = run_bt_structural(df, sigs, friction_tp=ftp, friction_sl=fsl)
        s = trade_summary(trades)
        if s:
            print(f"{label:<25} {s['n']:>5} {s['per_day']:>7.3f} {s['daily_net']:>+9.4f}% "
                  f"{s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['avg_gross']:>+9.4f}% {s['reasons']}")

    # Bootstrap 3-day with mixed friction
    print(f"\n{'=' * 80}\nBootstrap 500 × 3-day windows (mixed friction)\n{'=' * 80}")
    bars_per_3day = 3 * 24 * 12
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(500, max_start))
    cand_pnls = []; bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        h1_w = h1[st:en]; h4_w = h4[st:en]; v_w = valid[st:en]
        sigs_w = entry_psi_prime(df_w, h1_w, h4_w, v_w)
        trades = run_bt_structural(df_w, sigs_w, friction_tp=0.04, friction_sl=0.07)
        cand_pnl = sum(t['net_pct'] for t in trades) if trades else 0
        cand_pnls.append(cand_pnl)
        bh = (df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.07
        bh_pnls.append(bh)
    mean_p = np.mean(cand_pnls)
    pos_rate = np.mean(np.array(cand_pnls) > 0)
    p5 = np.percentile(cand_pnls, 5)
    p_better = np.mean(np.array(cand_pnls) > np.array(bh_pnls))
    print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%  p_better_BH={p_better:.4f}")

    # Look-ahead audit
    print(f"\n{'=' * 80}\nLook-ahead audit\n{'=' * 80}")
    audit_idx = random.sample([i for i, _ in sigs], min(15, len(sigs)))
    leaks = 0
    for i in audit_idx:
        df_t = df.iloc[:i+1].copy()
        h1_t = h1[:i+1]; h4_t = h4[:i+1]; v_t = valid[:i+1]
        try:
            t_sigs = entry_psi_prime(df_t, h1_t, h4_t, v_t)
            full_at_i = next((d for idx, d in sigs if idx == i), None)
            trunc_at_i = next((d for idx, d in t_sigs if idx == i), None)
            if full_at_i != trunc_at_i:
                leaks += 1
        except Exception:
            leaks += 1
    print(f"  Audited {len(audit_idx)}, leaks: {leaks}")

    # 7-test verdict (using scenario B as primary, mixed friction)
    trades_b = run_bt_structural(df, sigs, friction_tp=0.04, friction_sl=0.07)
    s_b = trade_summary(trades_b)
    if s_b:
        cond = {
            'test1_lookahead': leaks == 0,
            'test3_friction_taker_C_pass': False,  # filled below
            'test3_friction_maker_A_pass': False,
            'test3_friction_mixed_B_pass': s_b['daily_net'] >= 0.2,
            'test4_bootstrap': mean_p > 0 and pos_rate >= 0.5 and p5 > -1 and p_better >= 0.6,
            'test5_gross_vs_fee': s_b['avg_gross'] >= 0.10,
            'test6_frequency': s_b['per_day'] >= 2.0,
            'test7_wr_rr': s_b['wr_pct'] >= 50 and s_b['rr'] >= 1.0,
        }
        # Test 3 specific friction scenarios
        for label_check, ftp, fsl, key in [
            ('test3_friction_maker_A_pass', 0.04, 0.04, 'test3_friction_maker_A_pass'),
            ('test3_friction_taker_C_pass', 0.10, 0.10, 'test3_friction_taker_C_pass'),
        ]:
            t = run_bt_structural(df, sigs, friction_tp=ftp, friction_sl=fsl)
            ss = trade_summary(t)
            cond[key] = ss is not None and ss['daily_net'] >= 0.2

        print(f"\n{'=' * 80}\nM3-R21 VERDICT (7 tests, primary scenario B = mixed)\n{'=' * 80}")
        for k, v in cond.items():
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
        all_pass = all(cond.values())
        print(f"\n  ALL 7 PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round21_structure_pattern.md',
           'n_signals': len(sigs),
           'scenario_B_summary': s_b,
           'bootstrap': {'mean': float(mean_p), 'pos_rate': float(pos_rate), 'p5': float(p5), 'p_better_bh': float(p_better)},
           'lookahead_leaks': leaks,
           'conditions': cond if 's_b' in dir() and s_b else None,
           'all_pass': all_pass if 's_b' in dir() and s_b else False}
    p = ROOT / 'results' / f'm3_r21_pattern_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
