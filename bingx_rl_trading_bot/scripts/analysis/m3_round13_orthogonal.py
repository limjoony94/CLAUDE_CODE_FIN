"""M3-R13 — Orthogonal axes: δ 1h timeframe + ε continuous regression."""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m1_bt_framework import compute_atr as compute_atr_list
from m2_round1_screening import compute_ema, compute_rsi, load_ohlcv, resample_to_4h, merge_htf


def compute_atr_arr(highs, lows, closes, period=14):
    return np.array(compute_atr_list(list(highs), list(lows), list(closes), period))


def rolling_pctile(arr, lookback, pct):
    s = pd.Series(arr)
    return s.rolling(lookback, min_periods=lookback).quantile(pct / 100).values


# ==================== δ — 1H TIMEFRAME ====================

def prepare_data_1h():
    """Build 1h BTC + ETH + 4h trend filter + 1d trend filter."""
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_15m_idx = df_15m.set_index('timestamp')
    # Aggregate 15m → 1h
    df_1h = df_15m_idx.resample('1H', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()

    # 1h indicators
    closes = df_1h['close'].values
    highs = df_1h['high'].values
    lows = df_1h['low'].values
    df_1h['ema9'] = compute_ema(closes, 9)
    df_1h['rsi14'] = compute_rsi(closes, 14)
    df_1h['atr14'] = compute_atr_arr(highs, lows, closes, 14)
    df_1h['atr_pctile_70_200'] = rolling_pctile(df_1h['atr14'].values, 200, 70)
    df_1h['btc_return'] = df_1h['close'].pct_change() * 100

    # 4h aggregate from 1h
    df_1h_idx = df_1h.set_index('timestamp')
    df_4h = df_1h_idx.resample('4H', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna(subset=['open']).reset_index()
    df_4h['ema20'] = compute_ema(df_4h['close'].values, 20)
    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf4_long'] = df_4h['ema20'] > df_4h['ema50']

    # 1d aggregate
    df_1d = df_1h_idx.resample('1D', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna(subset=['open']).reset_index()
    df_1d['ema20'] = compute_ema(df_1d['close'].values, 20)
    df_1d['htf1d_long'] = df_1d['close'] > df_1d['ema20']

    # Merge
    df_1h['close_time'] = df_1h['timestamp'] + pd.Timedelta(hours=1)
    df_1h = merge_htf(df_1h, df_4h.rename(columns={'htf4_long': 'h4_long'}), 240, ['h4_long'])
    df_1h = merge_htf(df_1h, df_1d.rename(columns={'htf1d_long': 'h1d_long'}), 1440, ['h1d_long'])
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)

    # ETH 1h aggregation
    df_eth_5m = load_ohlcv(ROOT / 'data' / 'eth_binance_5m.csv')
    df_eth_idx = df_eth_5m.set_index('timestamp')
    df_eth_1h = df_eth_idx.resample('1H', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()
    df_eth_1h['eth_return'] = df_eth_1h['close'].pct_change() * 100
    df_eth_1h = df_eth_1h.rename(columns={'close': 'eth_close'})

    df_1h = pd.merge_asof(df_1h.sort_values('timestamp'),
                           df_eth_1h[['timestamp', 'eth_close', 'eth_return']].sort_values('timestamp'),
                           on='timestamp', direction='backward', tolerance=pd.Timedelta(hours=1))
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)

    h4 = df_1h['h4_long'].fillna(False).astype(bool).values
    h1d = df_1h['h1d_long'].fillna(False).astype(bool).values

    valid = ((~pd.isna(df_1h['rsi14'])) & (~pd.isna(df_1h['atr14']))
              & (~pd.isna(df_1h['atr_pctile_70_200']))
              & (~df_1h['h4_long'].isna()) & (~df_1h['h1d_long'].isna())
              & (~pd.isna(df_1h['eth_close'])) & (~pd.isna(df_1h['eth_return']))).values
    return df_1h, h4, h1d, valid


def entry_delta(df, h4, h1d, valid, eth_thresh=0.6, btc_lag_thresh=0.2):
    n = len(df)
    btc_ret = df['btc_return'].values
    eth_ret = df['eth_return'].values
    atr = df['atr14'].values
    atr_pctile = df['atr_pctile_70_200'].values
    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (btc_ret[i - 1], eth_ret[i - 1], atr[i], atr_pctile[i])):
            continue
        if not (atr[i] > atr_pctile[i]): continue
        if eth_ret[i - 1] > eth_thresh and btc_ret[i - 1] < btc_lag_thresh and h4[i] and h1d[i]:
            sigs.append((i, 'LONG'))
        elif eth_ret[i - 1] < -eth_thresh and btc_ret[i - 1] > -btc_lag_thresh and (not h4[i]) and (not h1d[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_simple(df, sigs, N_exit, friction, emergency_pct=1.5, min_bars_between=2):
    n = len(df)
    op = df['open'].values
    high = df['high'].values
    low = df['low'].values
    cl = df['close'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}

    in_pos = False
    pdir = None; pentry = None; pemerg = None; pstart = None
    cooldown_until = 0
    trades = []
    i = 0
    while i < n:
        if in_pos:
            exit_price = None; exit_reason = None
            if pdir == 'LONG' and low[i] <= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            elif pdir == 'SHORT' and high[i] >= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            held = i - pstart
            if exit_price is None and held >= N_exit:
                exit_price, exit_reason = cl[i], 'TIMEOUT'
            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                net = gross - friction
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'direction': pdir, 'entry': float(pentry), 'exit': float(exit_price),
                                'gross_pct': round(gross, 4), 'net_pct': round(net, 4),
                                'reason': exit_reason, 'bars_held': held})
                in_pos = False
                cooldown_until = i + min_bars_between
        if not in_pos and i >= cooldown_until and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                if pdir == 'LONG':
                    pemerg = pentry * (1 - emergency_pct / 100)
                else:
                    pemerg = pentry * (1 + emergency_pct / 100)
                pstart = ni
                in_pos = True
                i = ni
                continue
        i += 1
    return trades


def trade_summary(trades):
    if not trades:
        return None
    nets = [t['net_pct'] for t in trades]
    days = (pd.to_datetime(trades[-1]['exit_ts']) - pd.to_datetime(trades[0]['entry_ts'])).days
    if days == 0: days = 1
    wins = sum(1 for x in nets if x > 0)
    n = len(nets)
    win_pnls = [x for x in nets if x > 0]
    loss_pnls = [x for x in nets if x <= 0]
    rr = abs((sum(win_pnls)/max(1, len(win_pnls))) / (sum(loss_pnls)/max(1, len(loss_pnls)))) if loss_pnls else float('inf')
    return {
        'n': n, 'days': days, 'per_day': round(n/days, 3),
        'sum_net': round(sum(nets), 2),
        'avg_net': round(sum(nets)/n, 4),
        'wr_pct': round(100 * wins / n, 2), 'rr': round(rr, 3),
        'daily_net': round(sum(nets)/days, 4),
    }


def test_delta():
    print("\n" + "=" * 80); print("δ — 1h TIMEFRAME ETH-LAG"); print("=" * 80)
    df, h4, h1d, valid = prepare_data_1h()
    print(f"  1h bars: {len(df):,} | valid: {int(valid.sum()):,} | days: {len(df)/24:.0f}")

    n_total = len(df)
    train_end = int(n_total * 0.6)
    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    h4_tr, h4_te = h4[:train_end], h4[train_end:]
    h1d_tr, h1d_te = h1d[:train_end], h1d[train_end:]
    valid_tr = valid[:train_end]
    valid_te = valid[train_end:]

    # Test base config (no parameter sweep — anti-fix-impulse)
    sigs_full = entry_delta(df, h4, h1d, valid, eth_thresh=0.6, btc_lag_thresh=0.2)
    sigs_tr = entry_delta(df_tr, h4_tr, h1d_tr, valid_tr, eth_thresh=0.6, btc_lag_thresh=0.2)
    sigs_te = entry_delta(df_te, h4_te, h1d_te, valid_te, eth_thresh=0.6, btc_lag_thresh=0.2)
    print(f"  signals: full={len(sigs_full)} train={len(sigs_tr)} test={len(sigs_te)}")

    # Sweep N_exit on full (small, only 4 values to limit multi-comp)
    print(f"\n{'N':>3} {'frict':>6} {'n_full':>7} {'daily_full':>12} {'WR':>6} {'RR':>6}")
    base_results = {}
    for N in (2, 4, 6, 8):
        trades = run_bt_simple(df, sigs_full, N_exit=N, friction=0.08)
        s = trade_summary(trades)
        base_results[N] = s
        if s:
            print(f"{N:>3} {0.08:>5.2f}% {s['n']:>7} {s['daily_net']:>+11.4f}% {s['wr_pct']:>5.1f}% {s['rr']:>5.2f}")

    # Pick N=4 as base spec (per pre-reg, no selection bias)
    N_base = 4
    print(f"\nUsing pre-registered N={N_base}, friction=0.08% (maker RT).")

    # Train/test
    trades_tr = run_bt_simple(df_tr, sigs_tr, N_exit=N_base, friction=0.08)
    trades_te = run_bt_simple(df_te, sigs_te, N_exit=N_base, friction=0.08)
    s_tr = trade_summary(trades_tr)
    s_te = trade_summary(trades_te)
    print(f"\nTrain (60%): {s_tr}")
    print(f"Test (40%): {s_te}")

    # WF 5-fold
    fold_size = n_total // 6
    wf = []
    for fold_i in range(5):
        train_end_f = (fold_i + 1) * fold_size
        test_start = train_end_f
        test_end = min(test_start + fold_size, n_total)
        df_f = df.iloc[test_start:test_end].reset_index(drop=True)
        h4_f = h4[test_start:test_end]; h1d_f = h1d[test_start:test_end]; v_f = valid[test_start:test_end]
        sigs_f = entry_delta(df_f, h4_f, h1d_f, v_f)
        trades = run_bt_simple(df_f, sigs_f, N_exit=N_base, friction=0.08)
        s_f = trade_summary(trades)
        wf.append({'fold': fold_i + 1, 'daily': s_f['daily_net'] if s_f else None,
                    'n': s_f['n'] if s_f else 0})
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)
    print(f"\nWF 5-fold (friction 0.08%): {[r['daily'] for r in wf]}")
    print(f"  Positive folds: {wf_pos}/5 [{'PASS' if wf_pos >= 3 else 'FAIL'}]")

    # Pre-reg check
    cond = {
        'oos_daily_pos': s_te is not None and s_te['daily_net'] > 0,
        'n_oos_ge_50': s_te is not None and s_te['n'] >= 50,
        'wr_ge_40': s_te is not None and s_te['wr_pct'] >= 40,
        'rr_ge_1.0': s_te is not None and s_te['rr'] >= 1.0,
        'wf_3of5': wf_pos >= 3,
    }
    all_pass = all(cond.values())
    print(f"\n  Pre-reg: {cond}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return {'name': 'δ 1h timeframe ETH-lag', 'N': N_base, 'train': s_tr, 'test': s_te,
            'wf': wf, 'wf_pos': wf_pos, 'cond': cond, 'pass': all_pass,
            'all_N_full_sample': base_results}


# ==================== ε — REGRESSION-BASED SIGNAL ====================

def test_epsilon():
    print("\n" + "=" * 80); print("ε — CONTINUOUS REGRESSION SIGNAL"); print("=" * 80)
    from sklearn.linear_model import LinearRegression
    sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
    from m3_critique_pipeline import prepare_all_data
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    print(f"  15m bars: {len(df):,} | eth_valid: {int(eth_valid.sum()):,}")

    n_total = len(df)
    train_end = int(n_total * 0.6)

    # Features (use prev bar to avoid look-ahead; predict close[i+1] to close[i+1+N])
    N_predict = 4  # predict 1 hour forward (4 bars × 15m)
    features_cols = ['eth_return', 'btc_return', 'atr14', 'rsi14', 'log_ratio', 'ratio_z']
    # Make a feature DataFrame at i (using i-1 values, to not use any forward info)
    feat_df = df[features_cols].shift(1)
    # Target: BTC return from open[i+1] to close[i+N_predict]
    op = df['open'].values
    cl = df['close'].values
    # target_pct[i] = ((cl[i+N_predict] - op[i+1]) / op[i+1]) * 100
    target = np.full(n_total, np.nan)
    for i in range(n_total - N_predict - 1):
        if pd.isna(op[i + 1]) or pd.isna(cl[i + N_predict]):
            continue
        target[i] = (cl[i + N_predict] / op[i + 1] - 1) * 100

    df['target'] = target
    df['feat_eth'] = feat_df['eth_return']
    df['feat_btc'] = feat_df['btc_return']
    df['feat_atr'] = feat_df['atr14']
    df['feat_rsi'] = feat_df['rsi14']
    df['feat_lr'] = feat_df['log_ratio']
    df['feat_z'] = feat_df['ratio_z']

    feature_cols_renamed = ['feat_eth', 'feat_btc', 'feat_atr', 'feat_rsi', 'feat_lr', 'feat_z']
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()

    train_clean = train_df.dropna(subset=feature_cols_renamed + ['target'])
    test_clean = test_df.dropna(subset=feature_cols_renamed + ['target'])
    print(f"  Train clean: {len(train_clean):,} samples | Test clean: {len(test_clean):,} samples")

    # Fit
    X_tr = train_clean[feature_cols_renamed].values
    y_tr = train_clean['target'].values
    X_te = test_clean[feature_cols_renamed].values
    y_te = test_clean['target'].values
    reg = LinearRegression().fit(X_tr, y_tr)
    pred_tr = reg.predict(X_tr)
    pred_te = reg.predict(X_te)
    print(f"  Coefficients: {dict(zip(feature_cols_renamed, [round(c, 5) for c in reg.coef_]))}")
    print(f"  Intercept: {reg.intercept_:.5f}")
    print(f"  R² train: {reg.score(X_tr, y_tr):.6f} | R² test: {reg.score(X_te, y_te):.6f}")

    # Sign agreement (forecast skill)
    sign_tr = np.mean(np.sign(pred_tr) == np.sign(y_tr))
    sign_te = np.mean(np.sign(pred_te) == np.sign(y_te))
    print(f"  Sign agreement train: {sign_tr:.4f} | test: {sign_te:.4f}")

    # Trading: use abs(pred) > X threshold
    print(f"\n  {'thresh':>8} {'n_test':>8} {'daily_test':>12} {'WR':>6} {'RR':>6}")
    test_results = {}
    test_days = (pd.to_datetime(test_clean['timestamp'].iloc[-1]) - pd.to_datetime(test_clean['timestamp'].iloc[0])).days
    if test_days == 0: test_days = 1
    for thresh in (0.10, 0.20, 0.30, 0.40):
        # Apply thresh on test predictions, simulate trades
        signals_te = []
        for j, (idx, pred, h1_v, h4_v) in enumerate(zip(test_clean.index, pred_te, test_clean['h1_long'], test_clean['h4_long'])):
            if pred > thresh and h1_v and h4_v:
                signals_te.append((idx, 'LONG'))
            elif pred < -thresh and (not h1_v) and (not h4_v):
                signals_te.append((idx, 'SHORT'))
        # Backtest signals
        # Need to use df indices (absolute), not test_clean indices
        # But test_clean indices are subset of df, so we need to adjust signals to df_te (test slice)
        # Simpler: since signals are df indices, work directly on full df but only count test trades
        sig_dict = {idx: d for idx, d in signals_te}
        n_sig = len(signals_te)
        if n_sig == 0:
            print(f"  {thresh:>7.2f}  no signals")
            continue
        # Use simple BT (open-of-next, exit at +N_predict bars or emergency)
        trades = []
        in_pos = False; pdir = None; pentry = None; pemerg = None; pstart = None
        cooldown = 0
        i = train_end
        while i < n_total:
            if in_pos:
                ep = None; er = None
                if pdir == 'LONG' and df['low'].iloc[i] <= pemerg:
                    ep, er = pemerg, 'EMERG'
                elif pdir == 'SHORT' and df['high'].iloc[i] >= pemerg:
                    ep, er = pemerg, 'EMERG'
                held = i - pstart
                if ep is None and held >= N_predict:
                    ep, er = df['close'].iloc[i], 'TIMEOUT'
                if ep is not None:
                    gross = ((ep / pentry - 1) * 100) if pdir == 'LONG' else ((1 - ep / pentry) * 100)
                    net = gross - 0.08
                    trades.append({'gross': gross, 'net': net, 'reason': er,
                                    'entry_ts': str(df['timestamp'].iloc[pstart]), 'exit_ts': str(df['timestamp'].iloc[i])})
                    in_pos = False
                    cooldown = i + 2
            if not in_pos and i >= cooldown and i in sig_dict:
                ni = i + 1
                if ni < n_total:
                    pentry = df['open'].iloc[ni]
                    pdir = sig_dict[i]
                    pemerg = pentry * (0.985 if pdir == 'LONG' else 1.015)
                    pstart = ni
                    in_pos = True
                    i = ni
                    continue
            i += 1
        if not trades:
            test_results[thresh] = None
            print(f"  {thresh:>7.2f}  no trades after BT")
            continue
        nets = [t['net'] for t in trades]
        wins = sum(1 for x in nets if x > 0)
        n_trades = len(nets)
        win_pnls = [x for x in nets if x > 0]
        loss_pnls = [x for x in nets if x <= 0]
        rr = abs((sum(win_pnls)/max(1, len(win_pnls))) / (sum(loss_pnls)/max(1, len(loss_pnls)))) if loss_pnls else float('inf')
        daily = sum(nets) / test_days
        test_results[thresh] = {
            'n': n_trades, 'daily_net': round(daily, 4),
            'wr_pct': round(100 * wins / n_trades, 2), 'rr': round(rr, 3),
            'sum_net': round(sum(nets), 2),
        }
        print(f"  {thresh:>7.2f} {n_trades:>8} {daily:>+11.4f}% {test_results[thresh]['wr_pct']:>5.1f}% {test_results[thresh]['rr']:>5.2f}")

    # Pre-reg check on best threshold
    valid_test = {t: r for t, r in test_results.items() if r is not None}
    best = max(valid_test.items(), key=lambda kv: kv[1]['daily_net']) if valid_test else None
    if best:
        bt = best[1]
        cond = {
            'oos_daily_pos': bt['daily_net'] > 0,
            'n_oos_ge_50': bt['n'] >= 50,
            'wr_ge_40': bt['wr_pct'] >= 40,
            'rr_ge_1.0': bt['rr'] >= 1.0,
            'sign_agreement_gt_50': sign_te > 0.50,
        }
        all_pass = all(cond.values())
        print(f"\n  Best threshold {best[0]}: {bt}")
        print(f"  Pre-reg: {cond}")
        print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    else:
        cond = None; all_pass = False
        print(f"\n  No valid test results.")
    return {'name': 'ε continuous regression', 'r2_train': reg.score(X_tr, y_tr),
            'r2_test': reg.score(X_te, y_te), 'sign_agree_train': sign_tr,
            'sign_agree_test': sign_te,
            'coef': dict(zip(feature_cols_renamed, [round(c, 5) for c in reg.coef_])),
            'test_results': test_results,
            'best': best, 'cond': cond, 'pass': all_pass}


def main():
    delta_results = test_delta()
    epsilon_results = test_epsilon()

    print("\n" + "=" * 100)
    print("M3-R13 — VERDICT (orthogonal axes)")
    print("=" * 100)
    print(f"  δ (1h timeframe)       : {'PASS' if delta_results['pass'] else 'FAIL'}")
    print(f"  ε (regression-based)   : {'PASS' if epsilon_results['pass'] else 'FAIL'}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round13_orthogonal_axes.md',
           'delta': delta_results,
           'epsilon': epsilon_results,
           'delta_pass': delta_results['pass'],
           'epsilon_pass': epsilon_results['pass']}
    p = ROOT / 'results' / f'm3_r13_orthogonal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
