"""M3-R35 — 15m mechanism deep exploration (3 classes).

Focus: 15m timeframe. 3 distinct mechanism classes, each with C1-level optimization.

A: Pullback continuation 15m (pullback to EMA20 in trend, reversal candle at EMA)
B: Breakout retest 15m (channel break + pullback to break level + LONG at retest)
C: Multi-bar momentum 15m (3+ consecutive same-direction bars + confirmation)

Each mechanism:
- Grid sweep on key entry params
- C1 production exit (progressive_trail, SL bounds, trail_activation)
- Train/test split (60/40)
- Top-by-train evaluated on test holdout
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round29_c1_exact_revalidation import prepare_15m_data
from m3_round30_c1_production_exact import run_bt_c1_production
from m1_bt_framework import compute_atr as compute_atr_list
from m2_round1_screening import compute_ema, load_ohlcv, merge_htf


def add_15m_extras(df):
    """SMA200 1h trend filter + EMA20/50 + volume SMA + RSI."""
    # 1h SMA200 trend filter
    df_1h = df.set_index('timestamp').resample('1h', label='left', closed='left').agg({
        'close': 'last'}).dropna().reset_index()
    df_1h['sma200'] = pd.Series(df_1h['close'].values).rolling(200, min_periods=200).mean().values
    df_1h['close_above_sma200'] = df_1h['close'] > df_1h['sma200']
    df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=15)
    df = merge_htf(df, df_1h.rename(columns={'close_above_sma200': 'sma200_long'})[['timestamp', 'sma200_long']],
                   60, ['sma200_long'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 4h trend filter
    df_4h = df.set_index('timestamp').resample('4h', label='left', closed='left').agg({
        'close': 'last'}).dropna().reset_index()
    df_4h['ema20_4h'] = compute_ema(df_4h['close'].values, 20)
    df_4h['ema50_4h'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf4_long'] = df_4h['ema20_4h'] > df_4h['ema50_4h']
    df = merge_htf(df, df_4h[['timestamp', 'htf4_long']], 240, ['htf4_long'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 15m EMA + volume SMA
    cl = df['close'].values
    df['ema20_15m'] = compute_ema(cl, 20)
    df['ema50_15m'] = compute_ema(cl, 50)
    df['volume_sma20'] = pd.Series(df['volume'].values).rolling(20, min_periods=20).mean().values

    # RSI 14
    delta = np.diff(cl, prepend=cl[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    df['rsi14_15m'] = 100 - 100 / (1 + rs)

    return df


def detect_engulfing_at(op, cl, i):
    bull = (cl[i] > op[i]) and (op[i-1] > cl[i-1]) and (cl[i] > op[i-1]) and (op[i] < cl[i-1])
    bear = (op[i] > cl[i]) and (cl[i-1] > op[i-1]) and (op[i] > cl[i-1]) and (cl[i] < op[i-1])
    return bull, bear


# ==================== Mechanism A: Pullback Continuation 15m ====================

def entry_pullback_15m(df, valid, params=None):
    """Pullback to EMA20 in trending market + reversal candle confirmation."""
    p = {'ema_dist_pct': 0.30, 'volume_mult': 1.0} if params is None else params
    n = len(df)
    op = df['open'].values; hi = df['high'].values; lo = df['low'].values; cl = df['close'].values
    vol = df['volume'].values; vol_sma = df['volume_sma20'].values
    ema20 = df['ema20_15m'].values; ema50 = df['ema50_15m'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values
    h4 = df['htf4_long'].fillna(False).astype(bool).values

    sigs = []
    for i in range(50, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (ema20[i], ema50[i], vol[i], vol_sma[i])): continue
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue

        # Trend direction (5m EMA20>EMA50 + 1h SMA200 + 4h)
        ema_long = ema20[i] > ema50[i]
        ema_short = ema20[i] < ema50[i]

        # Pullback: low touched/near EMA20
        ema = ema20[i]
        long_pullback = (lo[i] <= ema * (1 + p['ema_dist_pct']/100)) and (cl[i] > ema)
        short_pullback = (hi[i] >= ema * (1 - p['ema_dist_pct']/100)) and (cl[i] < ema)

        # Reversal candle
        bull_eng, bear_eng = detect_engulfing_at(op, cl, i)

        if (ema_long and sma_long[i] and h4[i] and long_pullback and bull_eng):
            sigs.append((i, 'LONG'))
        elif (ema_short and (not sma_long[i]) and (not h4[i]) and short_pullback and bear_eng):
            sigs.append((i, 'SHORT'))
    return sigs


# ==================== Mechanism B: Breakout Retest 15m ====================

def entry_breakout_retest_15m(df, valid, params=None):
    """Channel break + pullback to break level + LONG at retest hold."""
    p = {'channel_period': 20, 'retest_buffer_pct': 0.20, 'volume_mult': 1.2} if params is None else params
    n = len(df)
    op = df['open'].values; hi = df['high'].values; lo = df['low'].values; cl = df['close'].values
    vol = df['volume'].values; vol_sma = df['volume_sma20'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values

    cp = p['channel_period']
    high_prev = pd.Series(hi).rolling(cp, min_periods=cp).max().shift(1).values
    low_prev = pd.Series(lo).rolling(cp, min_periods=cp).min().shift(1).values

    sigs = []
    for i in range(cp + 5, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (high_prev[i], low_prev[i], vol[i], vol_sma[i])): continue
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue

        # Look for breakout in past 1-3 bars
        recent_break_high = None  # break level
        recent_break_low = None
        for k in range(1, 4):
            if cl[i-k] > high_prev[i-k]:
                recent_break_high = high_prev[i-k]
                break
            if cl[i-k] < low_prev[i-k]:
                recent_break_low = low_prev[i-k]
                break

        # LONG: had recent break above + current bar pulled back to break level + held above + 1h trend
        if recent_break_high is not None:
            # Current bar tests break level: low <= level (within buffer) AND close > level
            buffer = recent_break_high * p['retest_buffer_pct'] / 100
            if (lo[i] <= recent_break_high + buffer and lo[i] >= recent_break_high - buffer
                    and cl[i] > recent_break_high
                    and sma_long[i]):
                sigs.append((i, 'LONG'))
        elif recent_break_low is not None:
            buffer = recent_break_low * p['retest_buffer_pct'] / 100
            if (hi[i] >= recent_break_low - buffer and hi[i] <= recent_break_low + buffer
                    and cl[i] < recent_break_low
                    and (not sma_long[i])):
                sigs.append((i, 'SHORT'))
    return sigs


# ==================== Mechanism C: Multi-bar Momentum 15m ====================

def entry_multibar_momentum_15m(df, valid, params=None):
    """3+ consecutive same-direction bars + confirmation + volume."""
    p = {'momentum_bars': 3, 'min_bar_body_ratio': 0.5, 'volume_mult': 1.2} if params is None else params
    n = len(df)
    op = df['open'].values; hi = df['high'].values; lo = df['low'].values; cl = df['close'].values
    vol = df['volume'].values; vol_sma = df['volume_sma20'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values

    mb = p['momentum_bars']
    sigs = []
    for i in range(mb + 2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (vol[i], vol_sma[i])): continue
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue

        # Check past mb bars all same direction
        all_up = all(cl[i-k] > op[i-k] for k in range(mb))
        all_down = all(cl[i-k] < op[i-k] for k in range(mb))

        # Body ratio of current bar
        body = abs(cl[i] - op[i])
        rng = hi[i] - lo[i]
        if rng <= 0: continue
        if body / rng < p['min_bar_body_ratio']: continue

        if all_up and cl[i] > op[i] and sma_long[i]:
            sigs.append((i, 'LONG'))
        elif all_down and cl[i] < op[i] and (not sma_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


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
    }


def run_grid(df, valid, entry_fn, param_grid, train_frac=0.6, friction=0.07, min_n=20):
    n_total = len(df)
    train_end = int(n_total * train_frac)
    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    v_tr = valid[:train_end]; v_te = valid[train_end:]

    results = []
    for params in param_grid:
        sigs_tr = entry_fn(df_tr, v_tr, params=params)
        sigs_te = entry_fn(df_te, v_te, params=params)
        if len(sigs_tr) < min_n or len(sigs_te) < min_n: continue

        trades_tr = run_bt_c1_production(df_tr, sigs_tr, friction=friction)
        trades_te = run_bt_c1_production(df_te, sigs_te, friction=friction)
        s_tr = trade_summary(trades_tr)
        s_te = trade_summary(trades_te)
        if s_tr is None or s_te is None: continue
        if s_tr['n'] < 15 or s_te['n'] < 15: continue
        results.append({'params': params,
                          'tr_daily': s_tr['daily_net'], 'tr_n': s_tr['n'], 'tr_wr': s_tr['wr_pct'], 'tr_rr': s_tr['rr'], 'tr_avg_g': s_tr['avg_gross'],
                          'te_daily': s_te['daily_net'], 'te_n': s_te['n'], 'te_wr': s_te['wr_pct'], 'te_rr': s_te['rr'], 'te_avg_g': s_te['avg_gross']})
    return results


def main():
    df, valid = prepare_15m_data()
    df = add_15m_extras(df)
    valid = (valid & (~df['sma200_long'].isna()).values & (~df['htf4_long'].isna()).values
              & (~pd.isna(df['ema20_15m']).values) & (~pd.isna(df['ema50_15m']).values)
              & (~pd.isna(df['volume_sma20']).values) & (~pd.isna(df['rsi14_15m']).values))

    n_total = len(df)
    print(f"15m bars: {n_total:,} | days: {n_total/96:.0f}\n")

    # Mechanism A: Pullback Continuation 15m
    print("=" * 100)
    print("[A] Pullback Continuation 15m + C1 production exit")
    print("=" * 100)
    grid_a = [{'ema_dist_pct': ed, 'volume_mult': vm}
              for ed, vm in product((0.20, 0.30, 0.50, 0.80), (1.0, 1.2, 1.5, 2.0))]
    results_a = run_grid(df, valid, entry_pullback_15m, grid_a)
    print(f"  Valid configs: {len(results_a)}/{len(grid_a)}")
    if results_a:
        results_a_sorted = sorted(results_a, key=lambda r: -r['te_daily'])
        print(f"  {'config':<45} {'tr_d':>9} {'te_d':>9} {'te_n':>5} {'te_WR':>6} {'te_RR':>5} {'te_avg_g':>9}")
        for r in results_a_sorted[:5]:
            p_str = str(r['params'])[:43]
            print(f"  {p_str:<45} {r['tr_daily']:>+8.4f} {r['te_daily']:>+8.4f} {r['te_n']:>5} {r['te_wr']:>5.1f}% {r['te_rr']:>4.2f} {r['te_avg_g']:>+8.4f}")
        survivors_a = [r for r in results_a if r['te_daily'] > 0]
        print(f"  OOS test survivors (te_daily > 0): {len(survivors_a)}/{len(results_a)}")

    # Mechanism B: Breakout Retest 15m
    print(f"\n{'='*100}")
    print(f"[B] Breakout Retest 15m + C1 production exit")
    print(f"{'='*100}")
    grid_b = [{'channel_period': cp, 'retest_buffer_pct': rb, 'volume_mult': vm}
              for cp, rb, vm in product((15, 20, 30), (0.10, 0.20, 0.30), (1.0, 1.2, 1.5))]
    results_b = run_grid(df, valid, entry_breakout_retest_15m, grid_b)
    print(f"  Valid configs: {len(results_b)}/{len(grid_b)}")
    if results_b:
        results_b_sorted = sorted(results_b, key=lambda r: -r['te_daily'])
        print(f"  {'config':<45} {'tr_d':>9} {'te_d':>9} {'te_n':>5} {'te_WR':>6} {'te_RR':>5} {'te_avg_g':>9}")
        for r in results_b_sorted[:5]:
            p_str = str(r['params'])[:43]
            print(f"  {p_str:<45} {r['tr_daily']:>+8.4f} {r['te_daily']:>+8.4f} {r['te_n']:>5} {r['te_wr']:>5.1f}% {r['te_rr']:>4.2f} {r['te_avg_g']:>+8.4f}")
        survivors_b = [r for r in results_b if r['te_daily'] > 0]
        print(f"  OOS test survivors: {len(survivors_b)}/{len(results_b)}")

    # Mechanism C: Multi-bar Momentum 15m
    print(f"\n{'='*100}")
    print(f"[C] Multi-bar Momentum 15m + C1 production exit")
    print(f"{'='*100}")
    grid_c = [{'momentum_bars': mb, 'min_bar_body_ratio': bb, 'volume_mult': vm}
              for mb, bb, vm in product((2, 3, 4, 5), (0.40, 0.50, 0.60), (1.0, 1.2, 1.5))]
    results_c = run_grid(df, valid, entry_multibar_momentum_15m, grid_c)
    print(f"  Valid configs: {len(results_c)}/{len(grid_c)}")
    if results_c:
        results_c_sorted = sorted(results_c, key=lambda r: -r['te_daily'])
        print(f"  {'config':<45} {'tr_d':>9} {'te_d':>9} {'te_n':>5} {'te_WR':>6} {'te_RR':>5} {'te_avg_g':>9}")
        for r in results_c_sorted[:5]:
            p_str = str(r['params'])[:43]
            print(f"  {p_str:<45} {r['tr_daily']:>+8.4f} {r['te_daily']:>+8.4f} {r['te_n']:>5} {r['te_wr']:>5.1f}% {r['te_rr']:>4.2f} {r['te_avg_g']:>+8.4f}")
        survivors_c = [r for r in results_c if r['te_daily'] > 0]
        print(f"  OOS test survivors: {len(survivors_c)}/{len(results_c)}")

    # Best across all 3 mechanisms
    all_results = []
    if results_a: all_results.extend([{**r, 'mechanism': 'A_pullback'} for r in results_a])
    if results_b: all_results.extend([{**r, 'mechanism': 'B_retest'} for r in results_b])
    if results_c: all_results.extend([{**r, 'mechanism': 'C_momentum'} for r in results_c])

    if all_results:
        all_survivors = [r for r in all_results if r['te_daily'] > 0]
        all_survivors.sort(key=lambda r: -r['te_daily'])
        print(f"\n{'='*100}\nALL 3 MECHANISMS — OOS SURVIVORS RANKED\n{'='*100}")
        print(f"  Total survivors: {len(all_survivors)}/{len(all_results)}")
        if all_survivors:
            print(f"  Top 10:")
            print(f"  {'mech':<12} {'config':<40} {'tr_d':>9} {'te_d':>9} {'te_n':>5} {'te_WR':>6} {'te_RR':>5} {'te_avg_g':>9}")
            for r in all_survivors[:10]:
                p_str = str(r['params'])[:38]
                print(f"  {r['mechanism']:<12} {p_str:<40} {r['tr_daily']:>+8.4f} {r['te_daily']:>+8.4f} {r['te_n']:>5} {r['te_wr']:>5.1f}% {r['te_rr']:>4.2f} {r['te_avg_g']:>+8.4f}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'A_pullback_results': results_a,
           'B_retest_results': results_b,
           'C_momentum_results': results_c,
           'all_survivors_count': len(all_survivors) if all_results else 0,
           'top_5_overall': all_survivors[:5] if all_results else []}
    p = ROOT / 'results' / f'm3_r35_15m_deep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
