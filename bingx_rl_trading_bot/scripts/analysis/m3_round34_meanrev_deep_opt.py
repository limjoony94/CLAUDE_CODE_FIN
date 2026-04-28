"""M3-R34 — Mean-Reversion strategy with C1-level deep optimization.

Real institutions use diverse strategies. C1 = momentum class (Donchian breakout + trail TP).
This R34 = mean-reversion class with class-appropriate exits.

Mechanism: Pattern reversal at extreme (R21 concept) + mean-reversion exit (VWAP target).
NOT trail TP (which is momentum-class exit).

Deep optimization angles:
1. Multi-config sweep on entry params (volume_mult, lookback_extreme, RSI threshold)
2. Multi-config sweep on exit (TP target type, SL distance, trail vs no-trail)
3. Train/test split for fair OOS
4. Class-appropriate exit: hit MEAN (VWAP/EMA20) for win, structural SL for loss
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round20_dynamic_scalping import prepare_5m_data
from m3_round21_pattern_structure import (add_sma200_1h, detect_engulfing, detect_hammer)
from m3_round24_pullback_continuation import add_ema_indicators
from m3_round23_vwap_scalping import compute_anchored_vwap


def add_indicators(df):
    """Add EMA20 5m + VWAP + RSI for mean-reversion strategy."""
    df = add_sma200_1h(df)
    df = add_ema_indicators(df)
    vwap, vwap_std = compute_anchored_vwap(df)
    df['vwap'] = vwap
    df['vwap_std'] = vwap_std
    # RSI 14 for entry filter
    cl = df['close'].values
    delta = np.diff(cl, prepend=cl[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    df['rsi14'] = 100 - 100 / (1 + rs)
    df['volume_sma20'] = pd.Series(df['volume'].values).rolling(20, min_periods=20).mean().values
    return df


def entry_meanrev_optimized(df, h1, h4, valid, params=None):
    """Mean-reversion entry: pattern reversal at extreme + RSI extreme + away from VWAP."""
    p = {
        'lookback_extreme': 20,
        'volume_mult': 1.2,
        'rsi_long_thresh': 35,  # RSI < 35 for LONG
        'vwap_z_thresh': 1.5,  # |z| > 1.5 from VWAP
    } if params is None else params
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vol = df['volume'].values
    vol_sma = df['volume_sma20'].values
    rsi = df['rsi14'].values
    vwap = df['vwap'].values
    vwap_std = df['vwap_std'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values

    sigs = []
    lb = p['lookback_extreme']
    for i in range(lb + 2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (vol[i], vol_sma[i], rsi[i], vwap[i], vwap_std[i])):
            continue
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue
        if vwap_std[i] <= 0: continue

        # Recent extreme
        recent_lows = lo[i - lb:i]
        recent_highs = hi[i - lb:i]
        recent_min = np.min(recent_lows)
        recent_max = np.max(recent_highs)
        low_touched = (lo[i-1] == recent_min) or (lo[i-2] == recent_min) or (lo[i] == recent_min)
        high_touched = (hi[i-1] == recent_max) or (hi[i-2] == recent_max) or (hi[i] == recent_max)

        # Reversal pattern
        bull_eng, bear_eng = detect_engulfing(op, cl, i)
        hammer, star = detect_hammer(op, cl, hi, lo, i)

        # VWAP z-score deviation
        vwap_z = (cl[i] - vwap[i]) / vwap_std[i]

        # LONG: low touched + bullish reversal + RSI oversold + price below VWAP + 1h trend permissive
        if (low_touched and (bull_eng or hammer)
                and rsi[i] <= p['rsi_long_thresh']
                and vwap_z < -p['vwap_z_thresh']
                and sma_long[i]):  # only LONG in uptrend bias
            sigs.append((i, 'LONG'))
        elif (high_touched and (bear_eng or star)
                and rsi[i] >= (100 - p['rsi_long_thresh'])
                and vwap_z > p['vwap_z_thresh']
                and (not sma_long[i])):
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_meanrev(df, sigs, friction_tp=0.04, friction_sl=0.07,
                     sl_buffer_pct=0.10, tp_target='vwap',  # 'vwap' or 'ema20'
                     trail_after_pct=0.20,  # trail to BE after profit > X%
                     emergency_pct=1.5, timeout_bars=24, min_bars_between=2):
    """Mean-reversion BT: TP at VWAP/EMA20 touch (not trail), structural SL, BE trail after profit."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vwap = df['vwap'].values
    ema20 = df['ema20_5m'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}

    in_pos = False
    pdir = None; pentry = None; psl = None; pemerg = None; pstart = None
    pbe_trailed = False
    cooldown = 0
    trades = []
    i = 0
    while i < n:
        if in_pos:
            current_pnl_pct = ((hi[i] / pentry - 1) * 100) if pdir == 'LONG' else ((1 - lo[i] / pentry) * 100)
            # Trail SL to BE after profit > trail_after_pct
            if not pbe_trailed and current_pnl_pct > trail_after_pct:
                if pdir == 'LONG':
                    psl = max(psl, pentry * 1.0001)  # BE+
                else:
                    psl = min(psl, pentry * 0.9999)
                pbe_trailed = True

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
            # TP at mean-reversion target
            if exit_price is None:
                target = vwap[i] if tp_target == 'vwap' and not pd.isna(vwap[i]) else ema20[i]
                if pd.isna(target): target = vwap[i] if not pd.isna(vwap[i]) else ema20[i]
                if not pd.isna(target):
                    if pdir == 'LONG' and hi[i] >= target and target > pentry:
                        exit_price, exit_reason = target, 'TP_MEAN'
                    elif pdir == 'SHORT' and lo[i] <= target and target < pentry:
                        exit_price, exit_reason = target, 'TP_MEAN'
            held = i - pstart
            if exit_price is None and held >= timeout_bars:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                fric = friction_tp if exit_reason == 'TP_MEAN' else friction_sl
                net = gross - fric
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'direction': pdir, 'entry': float(pentry), 'exit': float(exit_price),
                                'gross_pct': round(gross, 4), 'net_pct': round(net, 4),
                                'reason': exit_reason, 'bars_held': held})
                in_pos = False
                cooldown = i + min_bars_between
                pbe_trailed = False

        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                # Structural SL: signal candle low/high - buffer
                if pdir == 'LONG':
                    psl = lo[i] * (1 - sl_buffer_pct/100)
                    pemerg = pentry * (1 - emergency_pct/100)
                else:
                    psl = hi[i] * (1 + sl_buffer_pct/100)
                    pemerg = pentry * (1 + emergency_pct/100)
                # Sanity: TP target must be on right side
                target = vwap[i] if not pd.isna(vwap[i]) else ema20[i]
                if pd.isna(target):
                    i += 1; continue
                if pdir == 'LONG' and not (psl < pentry < target):
                    i += 1; continue
                if pdir == 'SHORT' and not (target < pentry < psl):
                    i += 1; continue
                pstart = ni
                in_pos = True
                pbe_trailed = False
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
        'reasons': reasons, 'trades': trades,
    }


def main():
    df, h1, h4, valid = prepare_5m_data()
    df = add_indicators(df)
    valid = (valid & (~df['sma200_long'].isna()).values
              & (~pd.isna(df['ema20_5m']).values)
              & (~pd.isna(df['vwap']).values)
              & (~pd.isna(df['vwap_std']).values)
              & (~pd.isna(df['rsi14']).values)
              & (~pd.isna(df['volume_sma20']).values))

    n_total = len(df)
    train_end = int(n_total * 0.6)
    print(f"5m bars: {n_total:,} | days: {n_total/(24*12):.0f}")
    print(f"Train/test split: {train_end} / {n_total - train_end}\n")

    # Phase 1: Train grid sweep
    print("=" * 100)
    print("Phase 1: Train grid sweep (mean-reversion deep optimization)")
    print("=" * 100)
    et_grid = (15, 25, 35)  # RSI long thresh
    vol_grid = (1.0, 1.5, 2.0)  # volume_mult
    z_grid = (1.0, 1.5, 2.0)  # vwap z thresh
    sl_buf_grid = (0.05, 0.10, 0.20)  # SL buffer pct
    tp_grid = ('vwap', 'ema20')
    timeout_grid = (12, 24, 48)  # bars (1h, 2h, 4h on 5m)

    train_results = []
    counter = 0
    for rsi_t, vm, z_t, sl_buf, tp_t, to in product(et_grid, vol_grid, z_grid, sl_buf_grid, tp_grid, timeout_grid):
        params = {'lookback_extreme': 20, 'volume_mult': vm, 'rsi_long_thresh': rsi_t, 'vwap_z_thresh': z_t}
        sigs_tr = entry_meanrev_optimized(df.iloc[:train_end].reset_index(drop=True), h1[:train_end], h4[:train_end], valid[:train_end], params=params)
        if len(sigs_tr) < 20: continue
        # Filter sigs to indices < train_end
        df_tr = df.iloc[:train_end].reset_index(drop=True)
        trades_tr = run_bt_meanrev(df_tr, sigs_tr, friction_tp=0.04, friction_sl=0.07,
                                     sl_buffer_pct=sl_buf, tp_target=tp_t, timeout_bars=to)
        s_tr = trade_summary(trades_tr)
        if s_tr is None or s_tr['n'] < 15: continue
        train_results.append({
            'rsi_t': rsi_t, 'vm': vm, 'z_t': z_t, 'sl_buf': sl_buf, 'tp_t': tp_t, 'to': to,
            'tr_daily': s_tr['daily_net'], 'tr_n': s_tr['n'], 'tr_wr': s_tr['wr_pct'], 'tr_rr': s_tr['rr'],
            'tr_avg_g': s_tr['avg_gross'],
        })
        counter += 1
    print(f"Train configs evaluated: {counter}/162")

    if not train_results:
        print("No valid train configs.")
        return

    # Sort by train daily, take top 10
    train_sorted = sorted(train_results, key=lambda r: -r['tr_daily'])
    print(f"\nTop 10 train configs:")
    print(f"{'rsi':>4} {'vm':>5} {'z':>4} {'sl':>5} {'tp':>5} {'to':>4} {'tr_d':>10} {'tr_n':>5} {'WR':>6} {'RR':>5} {'avg_g':>8}")
    for r in train_sorted[:10]:
        print(f"{r['rsi_t']:>4} {r['vm']:>5} {r['z_t']:>4} {r['sl_buf']:>5} {r['tp_t']:>5} {r['to']:>4} "
              f"{r['tr_daily']:>+9.4f} {r['tr_n']:>5} {r['tr_wr']:>5.1f}% {r['tr_rr']:>4.2f} {r['tr_avg_g']:>+7.4f}")

    # Phase 2: Test top 10 on holdout
    print(f"\n{'='*100}\nPhase 2: Top-10 by train, OOS test\n{'='*100}")
    print(f"{'rsi':>4} {'vm':>5} {'z':>4} {'sl':>5} {'tp':>5} {'to':>4} {'tr_d':>10} {'te_d':>10} {'te_n':>5} {'WR':>6} {'RR':>5} {'avg_g':>8}")

    df_te = df.iloc[train_end:].reset_index(drop=True)
    h1_te = h1[train_end:]; h4_te = h4[train_end:]; v_te = valid[train_end:]
    oos_results = []
    for r in train_sorted[:10]:
        params = {'lookback_extreme': 20, 'volume_mult': r['vm'], 'rsi_long_thresh': r['rsi_t'], 'vwap_z_thresh': r['z_t']}
        sigs_te = entry_meanrev_optimized(df_te, h1_te, h4_te, v_te, params=params)
        trades_te = run_bt_meanrev(df_te, sigs_te, friction_tp=0.04, friction_sl=0.07,
                                     sl_buffer_pct=r['sl_buf'], tp_target=r['tp_t'], timeout_bars=r['to'])
        s_te = trade_summary(trades_te)
        if s_te is None: continue
        oos_results.append({**r, 'te_daily': s_te['daily_net'], 'te_n': s_te['n'],
                              'te_wr': s_te['wr_pct'], 'te_rr': s_te['rr'], 'te_avg_g': s_te['avg_gross']})
        print(f"{r['rsi_t']:>4} {r['vm']:>5} {r['z_t']:>4} {r['sl_buf']:>5} {r['tp_t']:>5} {r['to']:>4} "
              f"{r['tr_daily']:>+9.4f} {s_te['daily_net']:>+9.4f} {s_te['n']:>5} {s_te['wr_pct']:>5.1f}% "
              f"{s_te['rr']:>4.2f} {s_te['avg_gross']:>+7.4f}")

    survivors = [r for r in oos_results if r['te_daily'] > 0]
    print(f"\nOOS survivors (test daily > 0): {len(survivors)}/10")

    if survivors:
        best = max(survivors, key=lambda r: r['te_daily'])
        print(f"\n  Best OOS: rsi={best['rsi_t']}, vm={best['vm']}, z={best['z_t']}, sl={best['sl_buf']}, tp={best['tp_t']}, to={best['to']}")
        print(f"    Train daily: {best['tr_daily']:+.4f}%, Test daily: {best['te_daily']:+.4f}%")
        print(f"    Test WR: {best['te_wr']}%, RR: {best['te_rr']}, avg_gross: {best['te_avg_g']:+.4f}%")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'mechanism': 'Mean-reversion deep optimization',
           'train_configs': len(train_results),
           'top_10_train': train_sorted[:10],
           'oos_results': oos_results,
           'survivors': len(survivors),
           'best': max(survivors, key=lambda r: r['te_daily']) if survivors else None}
    p = ROOT / 'results' / f'm3_r34_meanrev_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
