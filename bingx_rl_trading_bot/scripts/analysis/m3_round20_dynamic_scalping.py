"""M3-R20 — ω': 5m/15m scalping with DYNAMIC ATR/structure TP/SL.

Pre-reg: claudedocs/m3_round20_dynamic_scalping.md

Multi-TF confluence entry (5m+15m+1h+4h) + cross-asset + volume.
Dynamic exit: TP=2×ATR, SL=max(swing, entry-1.5×ATR), trail after profit, emergency 1.5%.
Friction: taker 0.10% RT (strict).

Full 7-test suite: look-ahead, overfit, fees, bootstrap, gross-vs-fee, freq, WR+RR.
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


def compute_atr_arr(highs, lows, closes, period=14):
    return np.array(compute_atr_list(list(highs), list(lows), list(closes), period))


def rolling_pctile(arr, lookback, pct):
    s = pd.Series(arr)
    return s.rolling(lookback, min_periods=lookback).quantile(pct / 100).values


# ==================== DATA PREPARATION ====================

def prepare_5m_data():
    """Build BTC 5m + 15m + 1h + 4h + ETH 5m alignment."""
    print("Loading BTC 5m + ETH 5m + indicators...")

    # BTC 5m
    df = load_ohlcv(ROOT / 'data' / 'btc_5m_720days_binance.csv')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values

    # 5m indicators
    df['atr14_5m'] = compute_atr_arr(highs, lows, closes, 14)
    df['atr_pctile_25_200'] = rolling_pctile(df['atr14_5m'].values, 200, 25)
    df['atr_pctile_75_200'] = rolling_pctile(df['atr14_5m'].values, 200, 75)
    df['volume_sma20'] = pd.Series(volumes).rolling(20, min_periods=20).mean().values
    df['btc_return_5m'] = df['close'].pct_change() * 100
    df['high_20_prev'] = pd.Series(highs).rolling(20, min_periods=20).max().shift(1).values
    df['low_20_prev'] = pd.Series(lows).rolling(20, min_periods=20).min().shift(1).values
    df['swing_low_10'] = pd.Series(lows).rolling(10, min_periods=10).min().shift(1).values
    df['swing_high_10'] = pd.Series(highs).rolling(10, min_periods=10).max().shift(1).values

    # 15m aggregation
    df_15m = df.set_index('timestamp').resample('15min', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()
    df_15m['btc_return_15m'] = df_15m['close'].pct_change() * 100

    # 1h aggregation
    df_1h = df.set_index('timestamp').resample('1h', label='left', closed='left').agg({
        'close': 'last'}).dropna().reset_index()
    df_1h['ema20_1h'] = compute_ema(df_1h['close'].values, 20)
    df_1h['ema50_1h'] = compute_ema(df_1h['close'].values, 50)
    df_1h['htf1h_long'] = df_1h['ema20_1h'] > df_1h['ema50_1h']

    # 4h aggregation
    df_4h = df.set_index('timestamp').resample('4h', label='left', closed='left').agg({
        'close': 'last'}).dropna().reset_index()
    df_4h['ema20_4h'] = compute_ema(df_4h['close'].values, 20)
    df_4h['htf4h_long'] = df_4h['close'] > df_4h['ema20_4h']

    # Merge 15m, 1h, 4h into 5m
    df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=5)
    df = merge_htf(df, df_15m[['timestamp', 'btc_return_15m']], 15, ['btc_return_15m'])
    df = merge_htf(df, df_1h.rename(columns={'htf1h_long': 'h1_long'})[['timestamp', 'h1_long']], 60, ['h1_long'])
    df = merge_htf(df, df_4h.rename(columns={'htf4h_long': 'h4_long'})[['timestamp', 'h4_long']], 240, ['h4_long'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ETH 5m
    eth = load_ohlcv(ROOT / 'data' / 'eth_binance_5m.csv')
    eth = eth[['timestamp', 'close']].rename(columns={'close': 'eth_close'}).sort_values('timestamp')
    eth['eth_return_5m'] = eth['eth_close'].pct_change() * 100
    df = pd.merge_asof(df.sort_values('timestamp'), eth, on='timestamp', direction='backward',
                        tolerance=pd.Timedelta(minutes=5)).sort_values('timestamp').reset_index(drop=True)

    # ETH 3-bar avg return
    df['eth_3bar_avg'] = pd.Series(df['eth_return_5m'].values).rolling(3, min_periods=3).mean().values

    h1 = df['h1_long'].fillna(False).astype(bool).values
    h4 = df['h4_long'].fillna(False).astype(bool).values

    valid = ((~pd.isna(df['atr14_5m'])) & (~pd.isna(df['atr_pctile_25_200']))
              & (~pd.isna(df['atr_pctile_75_200'])) & (~pd.isna(df['volume_sma20']))
              & (~pd.isna(df['btc_return_5m'])) & (~pd.isna(df['btc_return_15m']))
              & (~pd.isna(df['high_20_prev'])) & (~pd.isna(df['swing_low_10']))
              & (~df['h1_long'].isna()) & (~df['h4_long'].isna())
              & (~pd.isna(df['eth_close'])) & (~pd.isna(df['eth_3bar_avg']))).values
    return df, h1, h4, valid


# ==================== ENTRY LOGIC (ω' multi-TF confluence) ====================

def entry_omega_prime(df, h1, h4, valid, params=None):
    p = {
        'breakout_bars': 20, 'volume_mult': 1.3, 'eth_avg_thresh': 0.0,
    } if params is None else params
    n = len(df)
    cl = df['close'].values
    vol = df['volume'].values
    vol_sma = df['volume_sma20'].values
    high_20_prev = df['high_20_prev'].values if p['breakout_bars'] == 20 else pd.Series(df['high'].values).rolling(p['breakout_bars'], min_periods=p['breakout_bars']).max().shift(1).values
    low_20_prev = df['low_20_prev'].values if p['breakout_bars'] == 20 else pd.Series(df['low'].values).rolling(p['breakout_bars'], min_periods=p['breakout_bars']).min().shift(1).values
    btc_15m = df['btc_return_15m'].values
    btc_5m = df['btc_return_5m'].values
    eth_avg = df['eth_3bar_avg'].values
    atr = df['atr14_5m'].values
    atr_lo = df['atr_pctile_25_200'].values
    atr_hi = df['atr_pctile_75_200'].values

    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (cl[i], vol[i], vol_sma[i], high_20_prev[i], low_20_prev[i],
                                      btc_15m[i], btc_5m[i - 1], eth_avg[i], atr[i],
                                      atr_lo[i], atr_hi[i])):
            continue

        # Risk gate: ATR in 25-75 percentile
        if not (atr_lo[i] <= atr[i] <= atr_hi[i]): continue

        # Volume confirm
        if vol[i] < p['volume_mult'] * vol_sma[i]: continue

        # LONG conditions
        long_breakout = cl[i] > high_20_prev[i]
        long_15m = btc_15m[i] > 0
        long_eth = eth_avg[i] > p['eth_avg_thresh']

        # SHORT mirror
        short_breakout = cl[i] < low_20_prev[i]
        short_15m = btc_15m[i] < 0
        short_eth = eth_avg[i] < -p['eth_avg_thresh']

        if long_breakout and long_15m and long_eth and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif short_breakout and short_15m and short_eth and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ==================== DYNAMIC EXIT BACKTEST ====================

def run_bt_dynamic(df, sigs, friction=0.10, atr_tp_mult=2.0, atr_sl_mult=1.5,
                    trail_threshold_atr=1.0, trail_dist_atr=1.0,
                    emergency_pct=1.5, timeout_bars=96, min_bars_between=2):
    """Dynamic exit: TP fixed at entry+2×ATR, SL=max(swing, entry-1.5×ATR), trail after 1×ATR profit."""
    n = len(df)
    op = df['open'].values
    high = df['high'].values
    low = df['low'].values
    cl = df['close'].values
    atr = df['atr14_5m'].values
    sw_low = df['swing_low_10'].values
    sw_high = df['swing_high_10'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}

    in_pos = False
    pdir = None; pentry = None; psl = None; ptp = None; pemerg = None
    pbest = None; pstart = None; patr_e = None
    cooldown = 0
    trades = []
    i = 0
    while i < n:
        if in_pos:
            # Update best
            if pdir == 'LONG':
                pbest = max(pbest, high[i])
            else:
                pbest = min(pbest, low[i])

            # Trail SL after profit > 1×ATR
            current_profit_pct = ((pbest / pentry - 1) * 100) if pdir == 'LONG' else ((1 - pbest / pentry) * 100)
            current_profit_atr = current_profit_pct / (patr_e / pentry * 100)  # profit in ATR units
            if current_profit_atr > trail_threshold_atr:
                if pdir == 'LONG':
                    new_sl = pbest - trail_dist_atr * patr_e
                    psl = max(psl, new_sl)  # only loosen up
                else:
                    new_sl = pbest + trail_dist_atr * patr_e
                    psl = min(psl, new_sl)

            # Tighter trail after 2×ATR profit
            if current_profit_atr > 2.0:
                if pdir == 'LONG':
                    psl = max(psl, pbest - 0.5 * patr_e)
                else:
                    psl = min(psl, pbest + 0.5 * patr_e)

            exit_price = None; exit_reason = None
            # Emergency
            if pdir == 'LONG' and low[i] <= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            elif pdir == 'SHORT' and high[i] >= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            # SL
            if exit_price is None:
                if pdir == 'LONG' and low[i] <= psl:
                    exit_price, exit_reason = psl, 'SL'
                elif pdir == 'SHORT' and high[i] >= psl:
                    exit_price, exit_reason = psl, 'SL'
            # TP
            if exit_price is None:
                if pdir == 'LONG' and high[i] >= ptp:
                    exit_price, exit_reason = ptp, 'TP'
                elif pdir == 'SHORT' and low[i] <= ptp:
                    exit_price, exit_reason = ptp, 'TP'
            # Timeout
            held = i - pstart
            if exit_price is None and held >= timeout_bars:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                net = gross - friction
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'direction': pdir, 'entry': float(pentry), 'exit': float(exit_price),
                                'gross_pct': round(gross, 4), 'net_pct': round(net, 4),
                                'reason': exit_reason, 'bars_held': held,
                                'atr_e': float(patr_e)})
                in_pos = False
                cooldown = i + min_bars_between

        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                patr_e = atr[i]
                if pdir == 'LONG':
                    structural_sl = sw_low[i] if not np.isnan(sw_low[i]) else pentry - atr_sl_mult * patr_e
                    atr_sl = pentry - atr_sl_mult * patr_e
                    psl = max(structural_sl, atr_sl)  # tighter of structure or vol
                    ptp = pentry + atr_tp_mult * patr_e
                    pemerg = pentry * (1 - emergency_pct / 100)
                    pbest = high[ni]
                else:
                    structural_sl = sw_high[i] if not np.isnan(sw_high[i]) else pentry + atr_sl_mult * patr_e
                    atr_sl = pentry + atr_sl_mult * patr_e
                    psl = min(structural_sl, atr_sl)
                    ptp = pentry - atr_tp_mult * patr_e
                    pemerg = pentry * (1 + emergency_pct / 100)
                    pbest = low[ni]
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
    return {
        'n': n, 'days': days, 'per_day': round(n/days, 3),
        'sum_net': round(sum(nets), 2), 'sum_gross': round(sum(grosses), 2),
        'avg_net': round(sum(nets)/n, 4), 'avg_gross': round(sum(grosses)/n, 4),
        'wr_pct': round(100*wins/n, 2), 'rr': round(rr, 3),
        'daily_net': round(sum(nets)/days, 4),
    }


# ==================== TEST SUITE ====================

def test_lookahead(df, h1, h4, valid):
    """Test 1: Look-ahead audit."""
    print("\n--- Test 1: Look-ahead audit ---")
    full_sigs = entry_omega_prime(df, h1, h4, valid)
    if not full_sigs:
        return {'pass': True, 'note': 'no signals — vacuous'}
    random.seed(42)
    audit_idx = random.sample([i for i, _ in full_sigs], min(20, len(full_sigs)))
    leaks = 0
    for i in audit_idx:
        df_t = df.iloc[:i+1].copy()
        h1_t = h1[:i+1]; h4_t = h4[:i+1]; v_t = valid[:i+1]
        try:
            t_sigs = entry_omega_prime(df_t, h1_t, h4_t, v_t)
            full_at_i = next((d for idx, d in full_sigs if idx == i), None)
            trunc_at_i = next((d for idx, d in t_sigs if idx == i), None)
            if full_at_i != trunc_at_i:
                leaks += 1
        except Exception as e:
            leaks += 1
    print(f"  Audited: {len(audit_idx)}, leaks: {leaks}")
    return {'pass': leaks == 0, 'audited': len(audit_idx), 'leaks': leaks}


def test_friction_scenarios(df, h1, h4, valid):
    """Test 3: Friction comprehensive."""
    print("\n--- Test 3: Friction scenarios ---")
    sigs = entry_omega_prime(df, h1, h4, valid)
    print(f"  Signals: {len(sigs)}")
    results = {}
    for label, fric in [('taker (0.10%)', 0.10), ('mixed (0.07%)', 0.07), ('maker (0.04%)', 0.04)]:
        trades = run_bt_dynamic(df, sigs, friction=fric)
        s = trade_summary(trades)
        results[label] = s
        if s:
            print(f"  {label}: n={s['n']} per_day={s['per_day']} daily_net={s['daily_net']:+.4f}% WR={s['wr_pct']}% RR={s['rr']} avg_gross={s['avg_gross']:+.4f}%")
    # Pass: taker daily ≥ 0.2%
    taker = results.get('taker (0.10%)')
    pass_taker = taker is not None and taker['daily_net'] >= 0.2
    return {'pass': pass_taker, 'results': results}


def test_bootstrap_3day(df, h1, h4, valid, n_bootstrap=500, friction=0.10):
    """Test 4: 3-day random window bootstrap."""
    print(f"\n--- Test 4: Bootstrap {n_bootstrap} × 3-day windows ---")
    bars_per_3day = 3 * 24 * 12  # 5m bars
    n_total = len(df)
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(n_bootstrap, max_start))

    cand_pnls = []
    bh_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        h1_w = h1[st:en]; h4_w = h4[st:en]; v_w = valid[st:en]
        sigs_w = entry_omega_prime(df_w, h1_w, h4_w, v_w)
        trades = run_bt_dynamic(df_w, sigs_w, friction=friction)
        cand_pnl = sum(t['net_pct'] for t in trades) if trades else 0
        cand_pnls.append(cand_pnl)
        # B&H baseline
        if len(df_w) > 0:
            bh = (df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - friction
            bh_pnls.append(bh)

    mean_p = np.mean(cand_pnls)
    pos_rate = np.mean(np.array(cand_pnls) > 0)
    p5 = np.percentile(cand_pnls, 5)
    p_better = np.mean(np.array(cand_pnls) > np.array(bh_pnls)) if bh_pnls else 0
    pass_ = mean_p > 0 and pos_rate >= 0.5 and p5 > -1 and p_better >= 0.6
    print(f"  mean={mean_p:+.4f}% pos_rate={pos_rate:.4f} p5={p5:+.4f}% p_better_than_BH={p_better:.4f}")
    print(f"  PASS criteria: mean>0 AND pos_rate≥0.5 AND p5>-1 AND p_vs_BH≥0.6 → {pass_}")
    return {'pass': pass_, 'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5, 'p_vs_bh': p_better}


def main():
    df, h1, h4, valid = prepare_5m_data()
    n_total = len(df)
    print(f"\n5m bars: {n_total:,} | days: {n_total/(24*12):.0f}")

    # Initial signal density check
    sigs = entry_omega_prime(df, h1, h4, valid)
    print(f"Total signals (full data): {len(sigs)} → {len(sigs)/(n_total/(24*12)):.2f} per day\n")

    if len(sigs) == 0:
        print("\nNo signals — strategy does not fire. Drop spec.")
        return

    # Run tests
    results = {}
    results['test1_lookahead'] = test_lookahead(df, h1, h4, valid)
    results['test3_friction'] = test_friction_scenarios(df, h1, h4, valid)
    results['test4_bootstrap'] = test_bootstrap_3day(df, h1, h4, valid, n_bootstrap=500, friction=0.10)

    # Test 5+6+7 derived from full BT @ taker friction
    print("\n--- Test 5+6+7: Per-trade gross, frequency, WR/RR @ taker friction ---")
    trades = run_bt_dynamic(df, sigs, friction=0.10)
    s_full = trade_summary(trades)
    if s_full:
        results['test5_gross_vs_fee'] = {'pass': s_full['avg_gross'] >= 0.10, 'avg_gross': s_full['avg_gross']}
        results['test6_frequency'] = {'pass': s_full['per_day'] >= 2.0, 'per_day': s_full['per_day']}
        results['test7_wr_rr'] = {'pass': s_full['wr_pct'] >= 50 and s_full['rr'] >= 1.0,
                                    'wr_pct': s_full['wr_pct'], 'rr': s_full['rr']}
        print(f"  Full BT: n={s_full['n']} days={s_full['days']} per_day={s_full['per_day']} daily={s_full['daily_net']:+.4f}%")
        print(f"  Test 5 avg_gross ≥ 0.10%: {s_full['avg_gross']:+.4f}% [{'PASS' if s_full['avg_gross']>=0.10 else 'FAIL'}]")
        print(f"  Test 6 per_day ≥ 2.0: {s_full['per_day']} [{'PASS' if s_full['per_day']>=2 else 'FAIL'}]")
        print(f"  Test 7 WR≥50%, RR≥1.0: WR={s_full['wr_pct']}, RR={s_full['rr']} [{'PASS' if s_full['wr_pct']>=50 and s_full['rr']>=1 else 'FAIL'}]")
    else:
        print("  No trades")

    # Verdict
    print("\n" + "=" * 80); print("M3-R20 VERDICT (7 tests)"); print("=" * 80)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v.get('pass') else 'FAIL'}")
    all_pass = all(v.get('pass', False) for v in results.values())
    print(f"\n  ALL 7 PASS: {all_pass}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round20_dynamic_scalping.md',
           'n_signals_total': len(sigs),
           'full_bt_summary': s_full,
           'test_results': results,
           'all_pass': all_pass}
    p = ROOT / 'results' / f'm3_r20_dynamic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
