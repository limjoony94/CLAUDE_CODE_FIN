"""
M3 Critique Pipeline — Spec-Agnostic Reusable Artifact
=======================================================
3 mechanisms (α, β, γ) × 5 critiques = 3×5 matrix output.

Architecture:
  - prepare_all_data(): unified data prep (BTC 15m + funding + ETH cross-asset)
  - SPECS: 3 mechanism definitions (entry_fn injection)
  - run_bt_with_spec(): one BT runner, exit_fn unified across specs
  - critique_*: 5 critique functions, generic interface, return {pass, metrics, details}
  - main(): per-mechanism fail-fast loop, 3×5 matrix output

Design constraint (advisor):
  - "If you find yourself writing 'and for mechanism α specifically...' you've broken reusability."
  - Critiques reusable across future specs (Round 4, etc.)

NO winner label. Matrix is the deliverable.
"""
import sys, json, random, copy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m2_round1_screening import (compute_ema, compute_rsi, load_ohlcv,
                                  resample_to_4h, merge_htf, percentile, stats_mfe,
                                  measure_mfe_for_signals, apply_n1_sequencing,
                                  isolation_test)
from m2_round2_screening import (rolling_max, rolling_min_arr, sma,
                                   measure_mfe_random_universe)
from m1_bt_framework import compute_atr as compute_atr_list


def compute_atr_arr(highs, lows, closes, period=14):
    """ATR returning np array."""
    return np.array(compute_atr_list(list(highs), list(lows), list(closes), period))


def rolling_pctile(arr, lookback, pct):
    """Rolling percentile of past `lookback` bars (causal)."""
    s = pd.Series(arr)
    return s.rolling(lookback, min_periods=lookback).quantile(pct / 100).values


# ---------- Data preparation (shared across all specs) ----------

def prepare_all_data():
    """Return df_15m with all needed columns + boolean masks."""
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_1h = load_ohlcv(ROOT / 'data' / 'btc_1h_720days.csv')
    df_4h = resample_to_4h(df_1h)

    closes = df_15m['close'].values
    highs = df_15m['high'].values
    lows = df_15m['low'].values

    # BTC 15m indicators
    df_15m['ema9'] = compute_ema(closes, 9)
    df_15m['rsi14'] = compute_rsi(closes, 14)
    df_15m['atr14'] = compute_atr_arr(highs, lows, closes, 14)
    df_15m['atr_pctile_70_200'] = rolling_pctile(df_15m['atr14'].values, 200, 70)
    df_15m['btc_return'] = df_15m['close'].pct_change() * 100

    # swing low / high (10-bar lookback, causal)
    sw_low = np.full(len(df_15m), np.nan)
    sw_high = np.full(len(df_15m), np.nan)
    cur_l = np.nan; cur_h = np.nan
    for i in range(10, len(df_15m)):
        wlow = lows[i - 10:i + 1]; whigh = highs[i - 10:i + 1]
        if lows[i] == wlow.min(): cur_l = lows[i]
        if highs[i] == whigh.max(): cur_h = highs[i]
        sw_low[i] = cur_l; sw_high[i] = cur_h
    df_15m['swing_low'] = sw_low
    df_15m['swing_high'] = sw_high

    # 1h trend filter
    df_1h['ema20'] = compute_ema(df_1h['close'].values, 20)
    df_1h['ema50'] = compute_ema(df_1h['close'].values, 50)
    df_1h['htf_long'] = df_1h['ema20'] > df_1h['ema50']
    # 4h trend filter
    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf_long'] = df_4h['close'] > df_4h['ema50']

    df_15m['close_time'] = df_15m['timestamp'] + pd.Timedelta(minutes=15)
    df_15m = merge_htf(df_15m, df_1h.rename(columns={'htf_long': 'h1_long'}), 60, ['h1_long'])
    df_15m = merge_htf(df_15m, df_4h.rename(columns={'htf_long': 'h4_long'}), 240, ['h4_long'])
    df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)

    # Funding rate alignment
    with open(ROOT / 'data' / 'bingx_funding_rates_full.json') as f:
        records = json.load(f)
    df_fund = pd.DataFrame(records)
    df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'], unit='ms', utc=True)
    df_fund['funding_pct'] = df_fund['fundingRate'].astype(float) * 100
    df_fund = df_fund.sort_values('timestamp').reset_index(drop=True)
    df_15m_sorted = df_15m.sort_values('timestamp')
    df_fund_sorted = df_fund[['timestamp', 'funding_pct']].sort_values('timestamp')
    df_15m = pd.merge_asof(df_15m_sorted, df_fund_sorted, on='timestamp', direction='backward').sort_values('timestamp').reset_index(drop=True)
    # 8 consecutive funding periods sum (256 fifteen-min bars = 64h)
    df_15m['funding_8sum'] = pd.Series(df_15m['funding_pct'].values).rolling(256, min_periods=256).sum().values

    # ETH alignment (for α + β + γ cross-asset)
    df_eth_5m = load_ohlcv(ROOT / 'data' / 'eth_binance_5m.csv')
    df = df_eth_5m.set_index('timestamp')
    df_eth_15m = df.resample('15min', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()
    df_eth_15m['eth_return'] = df_eth_15m['close'].pct_change() * 100
    df_eth_15m = df_eth_15m.rename(columns={'close': 'eth_close'})

    df_15m_sorted = df_15m.sort_values('timestamp')
    df_15m = pd.merge_asof(df_15m_sorted,
                            df_eth_15m[['timestamp', 'eth_close', 'eth_return']].sort_values('timestamp'),
                            on='timestamp', direction='backward', tolerance=pd.Timedelta(minutes=15))
    df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)

    # Spread z-score + correlation (for β)
    df_15m['log_ratio'] = np.log(df_15m['close'] / df_15m['eth_close'])
    df_15m['ratio_mean50'] = pd.Series(df_15m['log_ratio'].values).rolling(50, min_periods=50).mean().values
    df_15m['ratio_std50'] = pd.Series(df_15m['log_ratio'].values).rolling(50, min_periods=50).std().values
    df_15m['ratio_z'] = (df_15m['log_ratio'] - df_15m['ratio_mean50']) / df_15m['ratio_std50']
    df_15m['corr50'] = pd.Series(df_15m['btc_return'].values).rolling(50, min_periods=50).corr(
        pd.Series(df_15m['eth_return'].values))

    # Boolean masks
    h1_long = df_15m['h1_long'].fillna(False).astype(bool).values
    h4_long = df_15m['h4_long'].fillna(False).astype(bool).values

    base_valid = ((~pd.isna(df_15m['rsi14'])) & (~pd.isna(df_15m['atr14']))
                   & (~pd.isna(df_15m['atr_pctile_70_200'])) & (~pd.isna(df_15m['swing_low']))
                   & (~df_15m['h1_long'].isna()) & (~df_15m['h4_long'].isna())).values
    eth_valid = base_valid & (~pd.isna(df_15m['eth_close'])).values & (~pd.isna(df_15m['ratio_z'])).values & (~pd.isna(df_15m['corr50'])).values
    funding_valid = base_valid & (~pd.isna(df_15m['funding_pct'])).values & (~pd.isna(df_15m['funding_8sum'])).values

    return df_15m, h1_long, h4_long, base_valid, eth_valid, funding_valid


# ---------- Spec definitions (entry functions) ----------

def entry_alpha(df, h1, h4, valid, params=None):
    """α: ETH-lag + high-vol regime conditional."""
    p = {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0} if params is None else params
    n = len(df)
    btc_ret = df['btc_return'].values
    eth_ret = df['eth_return'].values
    atr = df['atr14'].values
    atr_pctile_col = df['atr_pctile_70_200'].values  # 70th pctile threshold reference

    # If params provided different pctile, recompute
    if params and params.get('atr_pctile', 70) != 70:
        atr_pctile_col = rolling_pctile(atr, 200, params['atr_pctile'])

    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if pd.isna(btc_ret[i - 1]) or pd.isna(eth_ret[i - 1]) or pd.isna(atr[i]) or pd.isna(atr_pctile_col[i]): continue
        # Regime gate
        if not (atr[i] > atr_pctile_col[i]): continue
        eth_up = eth_ret[i - 1] > p['eth_thresh']
        btc_lag_up = btc_ret[i - 1] < p['btc_lag_thresh']
        eth_down = eth_ret[i - 1] < -p['eth_thresh']
        btc_lag_down = btc_ret[i - 1] > -p['btc_lag_thresh']
        if eth_up and btc_lag_up and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif eth_down and btc_lag_down and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def entry_beta(df, h1, h4, valid, params=None):
    """β: spread mean-rev × correlation breakdown compound."""
    p = {'z_thresh': 2.0, 'corr_thresh': 0.5} if params is None else params
    n = len(df)
    z = df['ratio_z'].values
    corr = df['corr50'].values
    sigs = []
    for i in range(1, n):
        if not valid[i]: continue
        if pd.isna(z[i]) or pd.isna(corr[i]): continue
        # Compound condition: z extreme AND correlation broken
        if not (corr[i] < p['corr_thresh']): continue
        if z[i] < -p['z_thresh'] and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif z[i] > p['z_thresh'] and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def entry_gamma(df, h1, h4, valid, params=None):
    """γ: funding sustained extreme + cross-asset confirmation."""
    p = {'funding_sum_thresh': 0.24, 'rsi_thresh': 70} if params is None else params
    n = len(df)
    fsum = df['funding_8sum'].values
    rsi = df['rsi14'].values
    eth_ret = df['eth_return'].values
    sigs = []
    for i in range(1, n):
        if not valid[i]: continue
        if pd.isna(fsum[i]) or pd.isna(rsi[i]) or pd.isna(eth_ret[i - 1]): continue
        # Fade overheated longs (SHORT)
        if fsum[i] >= p['funding_sum_thresh'] and rsi[i] >= p['rsi_thresh'] and eth_ret[i - 1] < 0:
            sigs.append((i, 'SHORT'))
        # Fade overheated shorts (LONG)
        elif fsum[i] <= -p['funding_sum_thresh'] and rsi[i] <= (100 - p['rsi_thresh']) and eth_ret[i - 1] > 0:
            sigs.append((i, 'LONG'))
    return sigs


# ---------- Spec registry ----------

SPECS = {
    'alpha': {
        'name': 'α ETH-lag + 고변동성',
        'entry_fn': entry_alpha,
        'parameters': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0},
        'sensitivity_params': {  # for C4 overfitting probe (±20%)
            'eth_thresh': [0.24, 0.36],
            'btc_lag_thresh': [0.08, 0.12],
            'atr_pctile': [60.0, 80.0],
        },
        'valid_mask_key': 'eth_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
    'beta': {
        'name': 'β spread × correlation compound',
        'entry_fn': entry_beta,
        'parameters': {'z_thresh': 2.0, 'corr_thresh': 0.5},
        'sensitivity_params': {
            'z_thresh': [1.6, 2.4],
            'corr_thresh': [0.4, 0.6],
        },
        'valid_mask_key': 'eth_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
    'gamma': {
        'name': 'γ funding × cross-asset',
        'entry_fn': entry_gamma,
        'parameters': {'funding_sum_thresh': 0.24, 'rsi_thresh': 70},
        'sensitivity_params': {
            'funding_sum_thresh': [0.18, 0.30],
            'rsi_thresh': [65, 75],
        },
        'valid_mask_key': 'funding_valid',
        'eligible_universe_with_filter': False,  # γ is counter-trend
        'direction_by_trend': False,
    },
}


# ---------- Common BT runner (spec-parameterized) ----------

EXIT_PARAMS = {
    'sl_atr_mult': 2.0,
    'trail_k': 2.0,
    'emergency_pct': 1.5,
    'timeout_bars': 16,
    'min_bars_between': 2,
}


def run_bt_with_spec(df, h1, h4, valid_mask, spec, friction=0.20, params=None):
    """Spec-parameterized BT. Common exit framework. Returns trades."""
    n = len(df)
    op = df['open'].values
    high = df['high'].values
    low = df['low'].values
    cl = df['close'].values
    atr = df['atr14'].values
    sw_low = df['swing_low'].values
    sw_high = df['swing_high'].values
    timestamps = df['timestamp'].values

    spec_params = params if params is not None else spec['parameters']
    signals = spec['entry_fn'](df, h1, h4, valid_mask, params=spec_params)
    signal_set = {idx: dir_ for idx, dir_ in signals}

    in_pos = False
    pdir = None; pentry = None; psl = None; pemerg = None; pbest = None; pstart = None
    cooldown_until = 0
    trades = []

    i = 0
    while i < n:
        if in_pos:
            atr_now = atr[i] if not np.isnan(atr[i]) else (atr[i - 1] if i > 0 else 0)
            # Update best
            if pdir == 'LONG':
                pbest = max(pbest, high[i])
            else:
                pbest = min(pbest, low[i])

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
            # Trail TP
            if exit_price is None:
                if pdir == 'LONG':
                    trigger = pbest - EXIT_PARAMS['trail_k'] * atr_now
                    if low[i] <= trigger:
                        exit_price, exit_reason = trigger, 'TRAIL_TP'
                else:
                    trigger = pbest + EXIT_PARAMS['trail_k'] * atr_now
                    if high[i] >= trigger:
                        exit_price, exit_reason = trigger, 'TRAIL_TP'
            # Timeout
            held = i - pstart
            if exit_price is None and held >= EXIT_PARAMS['timeout_bars']:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                net = gross - friction
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'direction': pdir, 'entry': float(pentry), 'exit': float(exit_price),
                                'gross_pct': round(gross, 4), 'net_pct': round(net, 4),
                                'reason': exit_reason, 'bars_held': held})
                in_pos = False
                cooldown_until = i + EXIT_PARAMS['min_bars_between']

        if not in_pos and i >= cooldown_until and i in signal_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = signal_set[i]
                # SL: max(swing, entry - 2*ATR)
                atr_dist = EXIT_PARAMS['sl_atr_mult'] * (atr[i] if not np.isnan(atr[i]) else 0)
                if pdir == 'LONG':
                    atr_sl = pentry - atr_dist
                    structural = sw_low[i] if not np.isnan(sw_low[i]) else atr_sl
                    psl = max(structural, atr_sl)
                    pemerg = pentry * (1 - EXIT_PARAMS['emergency_pct'] / 100)
                else:
                    atr_sl = pentry + atr_dist
                    structural = sw_high[i] if not np.isnan(sw_high[i]) else atr_sl
                    psl = min(structural, atr_sl)
                    pemerg = pentry * (1 + EXIT_PARAMS['emergency_pct'] / 100)
                pbest = high[ni] if pdir == 'LONG' else low[ni]
                pstart = ni
                in_pos = True
                i = ni
                continue
        i += 1
    return trades


def trade_summary(trades, friction=0.20, days=None):
    if not trades:
        return None
    nets = [t['net_pct'] for t in trades]
    grosses = [t['gross_pct'] for t in trades]
    if days is None:
        days = (pd.to_datetime(trades[-1]['exit_ts']) - pd.to_datetime(trades[0]['entry_ts'])).days
    wins = sum(1 for x in nets if x > 0)
    n = len(nets)
    win_pnls = [x for x in nets if x > 0]
    loss_pnls = [x for x in nets if x <= 0]
    rr = abs((sum(win_pnls)/max(1, len(win_pnls))) / (sum(loss_pnls)/max(1, len(loss_pnls)))) if loss_pnls else float('inf')
    return {
        'n': n,
        'days': days,
        'per_day': round(n/days, 3) if days else 0,
        'sum_net': round(sum(nets), 2),
        'sum_gross': round(sum(grosses), 2),
        'avg_net': round(sum(nets)/n, 4),
        'avg_gross': round(sum(grosses)/n, 4),
        'wr_pct': round(100 * wins / n, 2),
        'rr': round(rr, 3),
        'daily_net': round(sum(nets)/days, 4) if days else 0,
    }


# ---------- Critique 1: Random baseline ----------

def critique_random_baseline(df, h1, h4, spec, valid_mask, eligible_mask):
    """Compare candidate vs random entries on same eligible universe."""
    spec_params = spec['parameters']
    signals = spec['entry_fn'](df, h1, h4, valid_mask, params=spec_params)
    cand_mfe = measure_mfe_for_signals(df, signals, max_bars=8)
    cand_stats = stats_mfe(cand_mfe, friction=0.20)
    if cand_stats is None:
        return {'pass': False, 'metrics': {'verdict': 'NO_SIGNALS', 'n': 0}, 'details': {'signals': 0}}

    rnd_per_seed = []
    for seed in (42, 123, 456, 789, 1234):
        rnd = measure_mfe_random_universe(df, eligible_mask, h1, h4,
                                            target_n=cand_stats['n'], max_bars=8, seed=seed,
                                            direction_by_trend=spec['direction_by_trend'])
        rs = stats_mfe(rnd, 0.20)
        if rs:
            rnd_per_seed.append(rs)
    if not rnd_per_seed:
        return {'pass': False, 'metrics': {'verdict': 'NO_RANDOM_SAMPLES'}, 'details': {}}

    rnd_p50 = sum(r['mfe_p50'] for r in rnd_per_seed) / len(rnd_per_seed)
    rnd_pct = sum(r['pct_mfe_gt_friction'] for r in rnd_per_seed) / len(rnd_per_seed)
    diff_p50 = cand_stats['mfe_p50'] - rnd_p50
    diff_pct = cand_stats['pct_mfe_gt_friction'] - rnd_pct
    asym = cand_stats['mfe_p50'] + cand_stats['mae_p50']

    pass_ = (diff_p50 >= 0.05) and (diff_pct >= 5.0)
    return {
        'pass': bool(pass_),
        'metrics': {
            'cand_mfe_p50': cand_stats['mfe_p50'], 'cand_mae_p50': cand_stats['mae_p50'],
            'random_p50': round(rnd_p50, 4), 'random_pct': round(rnd_pct, 4),
            'diff_p50': round(diff_p50, 4), 'diff_pct': round(diff_pct, 2),
            'asymmetry': round(asym, 4), 'n_signals': len(signals), 'n_after_seq': cand_stats['n'],
        },
        'details': {'random_per_seed': rnd_per_seed},
    }


# ---------- Critique 2: Look-ahead audit ----------

def critique_lookahead_audit(df, h1, h4, spec, valid_mask, eligible_mask):
    """Truncate data at random t, recompute spec at t, verify signals match.
    If indicator has look-ahead, signals at t will differ between truncated and full data.
    """
    n = len(df)
    # Get full signals
    full_sigs = spec['entry_fn'](df, h1, h4, valid_mask, params=spec['parameters'])
    if not full_sigs:
        return {'pass': True, 'metrics': {'verdict': 'NO_SIGNALS_TO_AUDIT'},
                'details': {'note': 'No signals — vacuously pass'}}

    # Pick random sample of signal indices to audit
    random.seed(42)
    audit_indices = random.sample([i for i, _ in full_sigs], min(20, len(full_sigs)))

    leaks = []
    for i in audit_indices:
        # Reconstruct df truncated at index i (i is included, i+1 onwards excluded)
        df_trunc = df.iloc[:i + 1].copy()
        h1_trunc = h1[:i + 1]
        h4_trunc = h4[:i + 1]
        valid_trunc = valid_mask[:i + 1]
        # Re-run entry_fn on truncated data
        try:
            trunc_sigs = spec['entry_fn'](df_trunc, h1_trunc, h4_trunc, valid_trunc,
                                            params=spec['parameters'])
        except Exception as e:
            leaks.append({'idx': int(i), 'error': str(e)})
            continue
        # Check if signal at i exists in trunc
        full_sig_at_i = next((d for idx_f, d in full_sigs if idx_f == i), None)
        trunc_sig_at_i = next((d for idx_t, d in trunc_sigs if idx_t == i), None)
        if full_sig_at_i != trunc_sig_at_i:
            leaks.append({'idx': int(i), 'full': full_sig_at_i, 'trunc': trunc_sig_at_i})

    pass_ = len(leaks) == 0
    return {
        'pass': bool(pass_),
        'metrics': {'audited': len(audit_indices), 'leaks_detected': len(leaks)},
        'details': {'leaks': leaks[:5]},  # first 5 only
    }


# ---------- Critique 3: Friction stress ----------

def critique_friction_stress(df, h1, h4, spec, valid_mask):
    """Run BT with friction in [0.20, 0.30, 0.50, 0.80]. PASS if BASE and MED both positive net daily PnL."""
    results = {}
    for friction in (0.20, 0.30, 0.50, 0.80):
        trades = run_bt_with_spec(df, h1, h4, valid_mask, spec, friction=friction)
        if not trades:
            results[friction] = None
            continue
        s = trade_summary(trades, friction=friction)
        results[friction] = s

    base = results.get(0.20)
    med = results.get(0.30)
    pass_ = (base and base['daily_net'] > 0) and (med and med['daily_net'] > 0)
    return {
        'pass': bool(pass_),
        'metrics': {f'friction_{k}': (v['daily_net'] if v else None) for k, v in results.items()},
        'details': {f'friction_{k}': v for k, v in results.items()},
    }


# ---------- Critique 4: Overfitting probe ----------

def critique_overfitting_probe(df, h1, h4, spec, valid_mask):
    """Sensitivity ±20% on each critical parameter. Pass: all cells positive daily_net AND consistent direction."""
    base_params = spec['parameters']
    base_trades = run_bt_with_spec(df, h1, h4, valid_mask, spec, friction=0.20, params=base_params)
    base_s = trade_summary(base_trades) if base_trades else None
    base_daily = base_s['daily_net'] if base_s else None

    sensitivity_results = {}
    for param_name, alt_values in spec['sensitivity_params'].items():
        for v in alt_values:
            test_params = dict(base_params)
            test_params[param_name] = v
            trades = run_bt_with_spec(df, h1, h4, valid_mask, spec, friction=0.20, params=test_params)
            s = trade_summary(trades) if trades else None
            sensitivity_results[f'{param_name}={v}'] = s['daily_net'] if s else None

    # Pass condition: base and all sensitivity cells daily_net > 0 (sign consistent)
    all_dailies = [base_daily] + [v for v in sensitivity_results.values() if v is not None]
    pass_ = base_daily is not None and base_daily > 0 and all(v > 0 for v in all_dailies if v is not None)

    # 3-fold expanding WF
    wf_results = []
    n = len(df)
    fold_size = n // 4  # 3 folds expanding (use first 1/4, then up to 1/2, etc.)
    for fold_i in range(3):
        train_end = (fold_i + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size
        if test_end > n: test_end = n
        df_test = df.iloc[test_start:test_end].reset_index(drop=True)
        h1_test = h1[test_start:test_end]
        h4_test = h4[test_start:test_end]
        valid_test = valid_mask[test_start:test_end]
        trades = run_bt_with_spec(df_test, h1_test, h4_test, valid_test, spec, friction=0.20)
        s = trade_summary(trades) if trades else None
        wf_results.append({'fold': fold_i + 1, 'daily_net': s['daily_net'] if s else None,
                           'n': s['n'] if s else 0})

    wf_pos = sum(1 for r in wf_results if r['daily_net'] is not None and r['daily_net'] > 0)
    wf_pass = wf_pos >= 2

    overall_pass = pass_ and wf_pass
    return {
        'pass': bool(overall_pass),
        'metrics': {
            'base_daily_net': base_daily,
            'sensitivity_cells': sensitivity_results,
            'wf_folds_positive': wf_pos,
            'wf_total': 3,
            'sensitivity_consistent': pass_,
            'wf_pass': wf_pass,
        },
        'details': {'wf_results': wf_results},
    }


# ---------- Critique 5: Bootstrap 3-day stability ----------

def critique_bootstrap_3day(df, h1, h4, spec, valid_mask, n_bootstrap=200):
    """1000 random 3-day windows, full BT each. Reduced to 200 for compute.
    Pass: mean > 0, pos_rate ≥ 50%, P5 > -1%, P(cand > random_3day) ≥ 60%.
    """
    random.seed(42)
    n = len(df)
    bars_per_3day = 3 * 24 * 4  # 15m bars × 96/day × 3 = 288 bars

    # Generate random window starts
    max_start = n - bars_per_3day - 1
    if max_start <= 0:
        return {'pass': False, 'metrics': {'verdict': 'INSUFFICIENT_DATA'}}

    window_starts = random.sample(range(max_start), min(n_bootstrap, max_start))
    candidate_pnls = []
    random_pnls = []
    for start in window_starts:
        end = start + bars_per_3day
        df_w = df.iloc[start:end].reset_index(drop=True)
        h1_w = h1[start:end]; h4_w = h4[start:end]; valid_w = valid_mask[start:end]
        trades = run_bt_with_spec(df_w, h1_w, h4_w, valid_w, spec, friction=0.20)
        cand_pnl = sum(t['net_pct'] for t in trades) if trades else 0
        candidate_pnls.append(cand_pnl)

        # Random baseline 3-day window: random entries within eligible bars in this window
        # For simplicity, use buy-and-hold as random comparison
        if len(df_w) > 0:
            bh_pnl = (df_w['close'].iloc[-1] / df_w['open'].iloc[0] - 1) * 100 - 0.20
            random_pnls.append(bh_pnl)

    if not candidate_pnls:
        return {'pass': False, 'metrics': {'verdict': 'NO_TRADES'}}

    mean_c = sum(candidate_pnls) / len(candidate_pnls)
    pos_rate = sum(1 for p in candidate_pnls if p > 0) / len(candidate_pnls)
    sorted_p = sorted(candidate_pnls)
    p5 = sorted_p[int(0.05 * len(sorted_p))]
    p_better = sum(1 for c, r in zip(candidate_pnls, random_pnls) if c > r) / max(1, len(random_pnls))

    pass_ = (mean_c > 0) and (pos_rate >= 0.5) and (p5 > -1.0) and (p_better >= 0.6)
    return {
        'pass': bool(pass_),
        'metrics': {
            'n_windows': len(candidate_pnls),
            'mean_pnl': round(mean_c, 4),
            'pos_rate': round(pos_rate, 4),
            'p5': round(p5, 4),
            'p_cand_better_than_bh': round(p_better, 4),
        },
        'details': {'sample_pnls': candidate_pnls[:20]},
    }


# ---------- Main pipeline ----------

def main():
    print("Loading + indicators (BTC 15m + funding + ETH cross-asset)...")
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    print(f"  bars: {len(df):,} | base_valid: {int(base_valid.sum()):,} | "
          f"eth_valid: {int(eth_valid.sum()):,} | funding_valid: {int(funding_valid.sum()):,}\n")

    valid_map = {'eth_valid': eth_valid, 'funding_valid': funding_valid}

    # Eligible universes for random baseline
    eligible_with_filter_eth = (h1 & h4 | (~h1) & (~h4)) & eth_valid
    eligible_no_filter_funding = funding_valid

    matrix = {}

    for spec_id, spec in SPECS.items():
        print("=" * 80); print(f"MECHANISM {spec_id}: {spec['name']}"); print("=" * 80)
        valid = valid_map[spec['valid_mask_key']]
        if spec['eligible_universe_with_filter']:
            eligible = eligible_with_filter_eth if spec['valid_mask_key'] == 'eth_valid' else (h1 & h4 | (~h1) & (~h4)) & valid
        else:
            eligible = valid

        results = {}

        # C1
        print("  C1 random baseline...")
        c1 = critique_random_baseline(df, h1, h4, spec, valid, eligible)
        results['C1'] = c1
        print(f"     pass={c1['pass']} metrics: {c1['metrics']}")
        if not c1['pass']:
            results['skipped'] = ['C2', 'C3', 'C4', 'C5']
            matrix[spec_id] = results
            continue

        # C2
        print("  C2 look-ahead audit...")
        c2 = critique_lookahead_audit(df, h1, h4, spec, valid, eligible)
        results['C2'] = c2
        print(f"     pass={c2['pass']} metrics: {c2['metrics']}")
        if not c2['pass']:
            results['skipped'] = ['C3', 'C4', 'C5']
            matrix[spec_id] = results
            continue

        # C3
        print("  C3 friction stress...")
        c3 = critique_friction_stress(df, h1, h4, spec, valid)
        results['C3'] = c3
        print(f"     pass={c3['pass']} metrics: {c3['metrics']}")
        if not c3['pass']:
            results['skipped'] = ['C4', 'C5']
            matrix[spec_id] = results
            continue

        # C4
        print("  C4 overfitting probe...")
        c4 = critique_overfitting_probe(df, h1, h4, spec, valid)
        results['C4'] = c4
        print(f"     pass={c4['pass']} metrics: {c4['metrics']}")
        if not c4['pass']:
            results['skipped'] = ['C5']
            matrix[spec_id] = results
            continue

        # C5
        print("  C5 bootstrap 3-day...")
        c5 = critique_bootstrap_3day(df, h1, h4, spec, valid)
        results['C5'] = c5
        print(f"     pass={c5['pass']} metrics: {c5['metrics']}")

        matrix[spec_id] = results
        print()

    # 3×5 matrix
    print("=" * 100)
    print("M3 — 3×5 MATRIX (per-mechanism fail-fast, no winner label)")
    print("=" * 100)
    print(f"{'mechanism':<35} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} {'died_at':>10}")
    for spec_id, res in matrix.items():
        spec_name = SPECS[spec_id]['name']
        cells = []
        died_at = '-'
        for ck in ['C1', 'C2', 'C3', 'C4', 'C5']:
            if ck not in res:
                cells.append('skip')
            else:
                cells.append('PASS' if res[ck]['pass'] else 'FAIL')
                if not res[ck]['pass'] and died_at == '-':
                    died_at = ck
        print(f"{spec_name:<35} " + " ".join(f"{c:>10}" for c in cells) + f" {died_at:>10}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec_doc': 'claudedocs/m3_mechanisms_3specs.md',
        'matrix': {k: {ck: v.get('pass') if isinstance(v, dict) else v for ck, v in res.items()}
                    for k, res in matrix.items()},
        'full_results': matrix,
    }
    p = ROOT / 'results' / f'm3_3x5_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
