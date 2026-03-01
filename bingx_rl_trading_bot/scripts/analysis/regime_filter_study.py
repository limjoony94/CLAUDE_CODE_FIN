#!/usr/bin/env python3
"""
Regime Filter Study — Portfolio-Level Direction Filters for Pattern 5m Bot

Live WR 63.4% vs In-sample WR 87.2% (23.8pp gap). Breakeven WR ~63.0%.
Recent loss analysis: 3/4 SL hits were counter-trend entries.

This study develops 6 regime indicators and backtests them as portfolio-level
direction filters to improve WR and PnL by avoiding counter-trend entries.

Indicators:
  1. SER  — Signed Efficiency Ratio
  2. MRT  — Multi-Resolution Trend (EMA-based)
  3. CBF  — Candle Body Flow (classify_candle weight-based)
  4. VND  — Volatility-Normalized Displacement
  5. PPP  — Price Percentile Position
  6. Hurst — Variance Ratio approximation of Hurst exponent

Protocol compliance:
  - classify_candle from production indicators.py
  - LEVERAGE = 3, FEE = FEE_PCT * LEVERAGE
  - Timeout trades: DROP
  - Same-bar TP/SL: bar open distance-based
  - Entry: signal bar + 1 open
  - Compound returns for equity

Usage:
  cd bingx_rl_trading_bot
  python scripts/analysis/regime_filter_study.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('regime_filter_study')

# ============================================================
# Constants
# ============================================================
FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 288
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'regime_filter_study.json')

# IS/OOS split: 70/30 of 77,760 bars
IS_END_BAR = 54432
# WF folds for top-10 validation
WF_FOLDS = [
    (0, 25920, 38880),     # Fold 1: 90d IS, 45d OOS
    (0, 38880, 51840),     # Fold 2: 135d IS, 45d OOS
    (0, 51840, 77760),     # Fold 3: 180d IS, 90d OOS
]

# Candle Body Flow weights
CBF_WEIGHTS = {
    'BU': 1.0, 'MU': 0.8, 'H': 0.5, 'DF': 0.3, 'U': 0.2,
    'DN': -0.2, 'ST': 0.0, 'D': 0.0,
    'BD': -1.0, 'MD': -0.8, 'IH': -0.5, 'GS': -0.3,
}


# ============================================================
# Data Loading
# ============================================================

def load_and_classify(data_file):
    """Load CSV and classify candles using production classify_candle()."""
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW).mean()

    types = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body']
        if pd.isna(avg_b):
            avg_b = 1.0
        ct = classify_candle(row, avg_b)
        types.append(ct.value)

    df['candle_type'] = types
    logger.info(f"Classified {len(df)} candles into 12 types")
    return df


def load_patterns(patterns_file):
    """Load dynamic patterns and return list of (pattern, direction, tp, sl)."""
    with open(patterns_file) as f:
        data = json.load(f)
    patterns = []
    for key, info in data['pattern_details'].items():
        patterns.append({
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': info['tp'],
            'sl': info['sl'],
        })
    logger.info(f"Loaded {len(patterns)} patterns from {patterns_file}")
    return patterns


def build_signal_index(types, n):
    """Build index: pattern_name -> list of signal bar indices."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


# ============================================================
# Backtest Engine (from scanner, with entry_bars output)
# ============================================================

def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest with TP/SL. Returns list of (entry_bar, exit_bar, pnl_pct)."""
    fee = FEE_PCT * LEVERAGE
    trades = []
    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1

        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if direction == 'LONG':
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (tp_pct if dist_tp <= dist_sl else -sl_pct) * LEVERAGE - fee
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - fee))
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - fee))
                break
        # Timeout: DROP

    return trades


def portfolio_1pos(all_trades):
    """1-position-at-a-time filter: sort by entry, skip overlapping."""
    if not all_trades:
        return []
    all_trades = sorted(all_trades, key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def calc_stats(trades):
    """Calculate portfolio statistics (compound equity)."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'edge_per_trade': 0}
    pnls = [t[2] for t in trades]
    # Compound equity
    equity = 1.0
    peak = 1.0
    mdd = 0
    wins = []
    losses = []
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
        else:
            losses.append(p)
    total_pnl = (equity - 1) * 100
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(total_pnl, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'edge_per_trade': round(sum(pnls) / len(pnls), 2),
    }


# ============================================================
# Generate All Signals (with entry bar for regime filter)
# ============================================================

def generate_all_signals(patterns, signal_index, opens, highs, lows, n_bars, bar_limit=None):
    """Generate all trades for the 59-pattern portfolio.

    Returns list of (entry_bar, exit_bar, pnl, direction, signal_bar).
    signal_bar is the pattern detection bar (entry_bar = signal_bar + 1).
    """
    all_trades = []
    effective_n = bar_limit if bar_limit else n_bars
    for pinfo in patterns:
        pat = pinfo['pattern']
        direction = pinfo['direction']
        tp = pinfo['tp']
        sl = pinfo['sl']
        sigs = signal_index.get(pat, [])
        if bar_limit:
            sigs = [s for s in sigs if s < bar_limit]
        trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, effective_n)
        for eb, xb, pnl in trades:
            all_trades.append((eb, xb, pnl, direction, eb - 1))  # signal_bar = eb - 1
    return all_trades


def portfolio_1pos_extended(all_trades):
    """1-pos filter keeping direction and signal_bar metadata."""
    if not all_trades:
        return []
    all_trades = sorted(all_trades, key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl, direction, sig_bar in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl, direction, sig_bar))
            last_exit = xb
    return filtered


# ============================================================
# Regime Indicators
# ============================================================

def compute_ser(closes, lookbacks):
    """Signed Efficiency Ratio: (C[0]-C[n]) / Σ|C[i]-C[i-1]|."""
    n = len(closes)
    results = {}
    for lb in lookbacks:
        arr = np.full(n, np.nan)
        for i in range(lb, n):
            net = closes[i] - closes[i - lb]
            path = sum(abs(closes[i - lb + k + 1] - closes[i - lb + k]) for k in range(lb))
            arr[i] = net / path if path > 0 else 0
        results[lb] = arr
    return results


def compute_ser_vectorized(closes, lookbacks):
    """Vectorized SER computation."""
    n = len(closes)
    diffs = np.abs(np.diff(closes))  # |C[i] - C[i-1]|, length n-1
    results = {}
    for lb in lookbacks:
        net = closes[lb:] - closes[:-lb]  # length n-lb
        # Cumulative path sums using rolling window
        cum_diffs = np.cumsum(diffs)
        path = np.empty(n - lb)
        for i in range(n - lb):
            start = i  # diffs[i] = |closes[i+1] - closes[i]|
            end = i + lb
            path[i] = cum_diffs[end - 1] - (cum_diffs[start - 1] if start > 0 else 0)
        ser = np.where(path > 0, net / path, 0)
        arr = np.full(n, np.nan)
        arr[lb:] = ser
        results[lb] = arr
    return results


def compute_mrt(closes, ema_periods=(96, 384, 2304), weights=(0.2, 0.3, 0.5)):
    """Multi-Resolution Trend: weighted sum of sign(close - EMA(p))."""
    n = len(closes)
    arr = np.full(n, np.nan)
    emas = []
    for p in ema_periods:
        s = pd.Series(closes)
        emas.append(s.ewm(span=p, min_periods=p).mean().values)

    max_period = max(ema_periods)
    for i in range(max_period, n):
        score = 0.0
        for k, p in enumerate(ema_periods):
            if not np.isnan(emas[k][i]):
                score += np.sign(closes[i] - emas[k][i]) * weights[k]
        arr[i] = score
    return arr


def compute_cbf(candle_types, lookbacks):
    """Candle Body Flow: weighted average of candle type scores."""
    n = len(candle_types)
    weights_arr = np.array([CBF_WEIGHTS.get(ct, 0) for ct in candle_types])
    results = {}
    for lb in lookbacks:
        arr = np.full(n, np.nan)
        cum = np.cumsum(weights_arr)
        for i in range(lb, n):
            s = cum[i] - (cum[i - lb] if i >= lb else 0)
            arr[i] = s / lb
        results[lb] = arr
    return results


def compute_vnd(closes, highs, lows, lookbacks, atr_period=14):
    """Volatility-Normalized Displacement: (C[0]-C[n]) / ATR(14)."""
    n = len(closes)
    # ATR calculation
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    tr = np.concatenate([[highs[0] - lows[0]], tr])
    atr = pd.Series(tr).rolling(atr_period).mean().values

    results = {}
    for lb in lookbacks:
        arr = np.full(n, np.nan)
        for i in range(max(lb, atr_period), n):
            if atr[i] > 0:
                arr[i] = (closes[i] - closes[i - lb]) / atr[i]
            else:
                arr[i] = 0
        results[lb] = arr
    return results


def compute_ppp(closes, lookbacks):
    """Price Percentile Position: rank of close in recent window."""
    n = len(closes)
    results = {}
    for lb in lookbacks:
        arr = np.full(n, np.nan)
        for i in range(lb, n):
            window = closes[i - lb:i + 1]
            rank = np.sum(window <= closes[i])
            arr[i] = rank / len(window)
        results[lb] = arr
    return results


def compute_hurst(closes, lags, window_mult=4):
    """Hurst exponent via Variance Ratio approximation."""
    n = len(closes)
    rets_1 = np.diff(np.log(np.maximum(closes, 1e-10)))  # 1-period log returns
    results = {}
    for lag in lags:
        arr = np.full(n, np.nan)
        window = lag * window_mult
        for i in range(window + lag, n):
            r1 = rets_1[i - window:i]
            # Lag-period returns
            log_c = np.log(np.maximum(closes[i - window:i + 1], 1e-10))
            r_lag = log_c[lag:] - log_c[:-lag]
            var_1 = np.var(r1) if len(r1) > 1 else 1e-10
            var_lag = np.var(r_lag) if len(r_lag) > 1 else 1e-10
            if var_1 > 0 and lag > 1:
                vr = var_lag / (lag * var_1)
                vr = max(vr, 1e-10)  # avoid log(0)
                h = 0.5 + 0.5 * np.log2(vr) / np.log2(lag)
                arr[i] = h
        results[lag] = arr
    return results


# ============================================================
# Filter Application
# ============================================================

def apply_regime_filter(portfolio_trades, indicator_values, threshold,
                        filter_type, indicator_bar_offset=0):
    """Apply regime filter to portfolio trades.

    Args:
        portfolio_trades: list of (eb, xb, pnl, direction, signal_bar)
        indicator_values: numpy array of indicator values
        threshold: filter threshold (meaning depends on filter_type)
        filter_type: 'directional' | 'ppp' | 'hurst' | 'hurst_directional'
        indicator_bar_offset: use signal_bar + offset for indicator lookup

    Returns:
        filtered trades list (same format)
    """
    filtered = []
    for eb, xb, pnl, direction, sig_bar in portfolio_trades:
        bar = sig_bar + indicator_bar_offset
        if bar < 0 or bar >= len(indicator_values):
            continue
        val = indicator_values[bar]
        if np.isnan(val):
            continue  # Skip if indicator not ready

        if filter_type == 'directional':
            # LONG: val > threshold, SHORT: val < -threshold
            if direction == 'LONG' and val > threshold:
                filtered.append((eb, xb, pnl, direction, sig_bar))
            elif direction == 'SHORT' and val < -threshold:
                filtered.append((eb, xb, pnl, direction, sig_bar))

        elif filter_type == 'ppp':
            # threshold is (low_thr, high_thr)
            low_thr, high_thr = threshold
            if direction == 'LONG' and val > low_thr:
                filtered.append((eb, xb, pnl, direction, sig_bar))
            elif direction == 'SHORT' and val < high_thr:
                filtered.append((eb, xb, pnl, direction, sig_bar))

        elif filter_type == 'hurst':
            # Only enter in trending markets: H > threshold
            if val > threshold:
                filtered.append((eb, xb, pnl, direction, sig_bar))

        elif filter_type == 'hurst_directional':
            # Trending market + direction check via SER
            h_val, ser_val = val  # packed tuple — handled separately
            # This case is handled in the Hurst scenario loop directly
            pass

    return filtered


# ============================================================
# Scenario Definition
# ============================================================

def build_scenarios(closes, highs_arr, lows_arr, candle_types):
    """Build all indicator arrays and scenario configurations."""
    logger.info("Computing regime indicators...")
    t0 = time.time()

    # 1. SER
    ser_data = compute_ser_vectorized(closes, [12, 24, 48])
    # 2. MRT
    mrt_data = compute_mrt(closes)
    # 3. CBF
    cbf_data = compute_cbf(candle_types, [12, 24, 48])
    # 4. VND
    vnd_data = compute_vnd(closes, highs_arr, lows_arr, [24, 48, 96])
    # 5. PPP
    ppp_data = compute_ppp(closes, [96, 288, 576])
    # 6. Hurst
    hurst_data = compute_hurst(closes, [16, 32, 64])

    elapsed = time.time() - t0
    logger.info(f"Indicators computed in {elapsed:.1f}s")

    scenarios = []

    # --- SER scenarios ---
    for lb in [12, 24, 48]:
        for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            scenarios.append({
                'name': f'SER_lb{lb}_thr{thr}',
                'indicator': 'SER',
                'values': ser_data[lb],
                'threshold': thr,
                'filter_type': 'directional',
                'params': {'lookback': lb, 'threshold': thr},
            })

    # --- MRT scenarios ---
    for thr in [0.0, 0.3, 0.5, 0.7]:
        scenarios.append({
            'name': f'MRT_thr{thr}',
            'indicator': 'MRT',
            'values': mrt_data,
            'threshold': thr,
            'filter_type': 'directional',
            'params': {'threshold': thr},
        })

    # --- CBF scenarios ---
    for lb in [12, 24, 48]:
        for thr in [0.0, 0.05, 0.10, 0.15, 0.20]:
            scenarios.append({
                'name': f'CBF_lb{lb}_thr{thr}',
                'indicator': 'CBF',
                'values': cbf_data[lb],
                'threshold': thr,
                'filter_type': 'directional',
                'params': {'lookback': lb, 'threshold': thr},
            })

    # --- VND scenarios ---
    for lb in [24, 48, 96]:
        for thr in [0.3, 0.5, 1.0, 1.5, 2.0]:
            scenarios.append({
                'name': f'VND_lb{lb}_thr{thr}',
                'indicator': 'VND',
                'values': vnd_data[lb],
                'threshold': thr,
                'filter_type': 'directional',
                'params': {'lookback': lb, 'threshold': thr},
            })

    # --- PPP scenarios ---
    for lb in [96, 288, 576]:
        for low_thr, high_thr in [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5)]:
            scenarios.append({
                'name': f'PPP_lb{lb}_lo{low_thr}_hi{high_thr}',
                'indicator': 'PPP',
                'values': ppp_data[lb],
                'threshold': (low_thr, high_thr),
                'filter_type': 'ppp',
                'params': {'lookback': lb, 'low_threshold': low_thr, 'high_threshold': high_thr},
            })

    # --- Hurst scenarios (trending only) ---
    for lag in [16, 32, 64]:
        for thr in [0.50, 0.52, 0.55, 0.58]:
            scenarios.append({
                'name': f'Hurst_lag{lag}_thr{thr}',
                'indicator': 'Hurst',
                'values': hurst_data[lag],
                'threshold': thr,
                'filter_type': 'hurst',
                'params': {'lag': lag, 'threshold': thr},
            })

    # --- Hurst + Direction (SER-24 for direction) ---
    ser_24 = ser_data[24]
    for lag in [16, 32, 64]:
        for h_thr in [0.52, 0.55]:
            scenarios.append({
                'name': f'Hurst+Dir_lag{lag}_h{h_thr}',
                'indicator': 'Hurst+SER',
                'values': (hurst_data[lag], ser_24),  # tuple
                'threshold': h_thr,
                'filter_type': 'hurst_directional',
                'params': {'lag': lag, 'hurst_threshold': h_thr, 'direction_indicator': 'SER_24'},
            })

    logger.info(f"Built {len(scenarios)} scenarios")
    return scenarios


def apply_hurst_directional_filter(portfolio_trades, hurst_arr, ser_arr, h_threshold):
    """Hurst trending + SER direction filter."""
    filtered = []
    for eb, xb, pnl, direction, sig_bar in portfolio_trades:
        if sig_bar < 0 or sig_bar >= len(hurst_arr) or sig_bar >= len(ser_arr):
            continue
        h_val = hurst_arr[sig_bar]
        s_val = ser_arr[sig_bar]
        if np.isnan(h_val) or np.isnan(s_val):
            continue
        if h_val > h_threshold:
            if direction == 'LONG' and s_val > 0:
                filtered.append((eb, xb, pnl, direction, sig_bar))
            elif direction == 'SHORT' and s_val < 0:
                filtered.append((eb, xb, pnl, direction, sig_bar))
    return filtered


# ============================================================
# IS Backtest
# ============================================================

def run_is_backtest(portfolio_trades_is, scenarios, baseline_stats):
    """Run IS backtest for all scenarios. Returns sorted results."""
    results = []
    for sc in scenarios:
        name = sc['name']

        if sc['filter_type'] == 'hurst_directional':
            h_arr, s_arr = sc['values']
            filtered = apply_hurst_directional_filter(
                portfolio_trades_is, h_arr, s_arr, sc['threshold'])
        else:
            filtered = apply_regime_filter(
                portfolio_trades_is, sc['values'], sc['threshold'], sc['filter_type'])

        # Convert back to (eb, xb, pnl) for stats
        trades_3 = [(eb, xb, pnl) for eb, xb, pnl, _, _ in filtered]
        stats = calc_stats(trades_3)

        # Skip if too few trades
        if stats['trades'] < 10:
            continue

        filtered_pct = round((1 - stats['trades'] / baseline_stats['trades']) * 100, 1) if baseline_stats['trades'] > 0 else 0

        results.append({
            'name': name,
            'indicator': sc['indicator'],
            'params': sc['params'],
            'trades': stats['trades'],
            'wr': stats['wr'],
            'pnl': stats['pnl'],
            'mdd': stats['mdd'],
            'pf': stats['pf'],
            'edge_per_trade': stats['edge_per_trade'],
            'filtered_pct': filtered_pct,
            'wr_delta': round(stats['wr'] - baseline_stats['wr'], 1),
            'pnl_delta': round(stats['pnl'] - baseline_stats['pnl'], 1),
            'pnl_mdd': round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 999,
        })

    # Sort by PnL/MDD descending
    results.sort(key=lambda x: x['pnl_mdd'], reverse=True)
    return results


# ============================================================
# Walk-Forward Validation
# ============================================================

def run_wf_validation(patterns, signal_index, opens, highs_arr, lows_arr,
                      closes, candle_types, top_scenarios, n_bars):
    """Run expanding window WF on top scenarios."""
    logger.info(f"Running WF validation on {len(top_scenarios)} scenarios...")
    wf_results = []

    for sc_info in top_scenarios:
        sc_name = sc_info['name']
        sc_params = sc_info['params']
        indicator = sc_info['indicator']
        logger.info(f"  WF: {sc_name}")

        fold_results = []
        for fold_idx, (is_start, is_end, oos_end) in enumerate(WF_FOLDS):
            # Generate trades for IS and OOS ranges
            all_trades_is = generate_all_signals(patterns, signal_index, opens, highs_arr, lows_arr, n_bars, bar_limit=is_end)
            all_trades_oos_full = generate_all_signals(patterns, signal_index, opens, highs_arr, lows_arr, n_bars, bar_limit=oos_end)

            # OOS trades = trades with entry_bar in [is_end, oos_end)
            all_trades_oos = [t for t in all_trades_oos_full if t[0] >= is_end]

            port_is = portfolio_1pos_extended(all_trades_is)
            port_oos = portfolio_1pos_extended(all_trades_oos)

            # Recompute indicators on IS range for filter calibration
            # (but apply to OOS trades using full indicator array)
            # Since indicators are computed on the entire series,
            # we use the global indicator values — no future leak
            # because indicator at bar X only uses data <= X

            # Rebuild indicator for this scenario
            ind_values = _rebuild_indicator(indicator, sc_params, closes, highs_arr, lows_arr, candle_types)

            # Apply filter to OOS trades
            if 'Hurst+' in indicator:
                hurst_lag = sc_params['lag']
                hurst_arr = compute_hurst(closes, [hurst_lag])[hurst_lag]
                ser_arr = compute_ser_vectorized(closes, [24])[24]
                filtered_oos = apply_hurst_directional_filter(
                    port_oos, hurst_arr, ser_arr, sc_params['hurst_threshold'])
            else:
                ft = _get_filter_type(indicator)
                threshold = _get_threshold(sc_params, indicator)
                filtered_oos = apply_regime_filter(port_oos, ind_values, threshold, ft)

            # Baseline OOS (no filter)
            base_oos = [(eb, xb, pnl) for eb, xb, pnl, _, _ in port_oos]
            filt_oos = [(eb, xb, pnl) for eb, xb, pnl, _, _ in filtered_oos]

            base_stats = calc_stats(base_oos)
            filt_stats = calc_stats(filt_oos)

            fold_results.append({
                'fold': fold_idx + 1,
                'is_bars': f"{is_start}-{is_end}",
                'oos_bars': f"{is_end}-{oos_end}",
                'baseline_oos': base_stats,
                'filtered_oos': filt_stats,
                'wr_delta': round(filt_stats['wr'] - base_stats['wr'], 1),
                'pnl_delta': round(filt_stats['pnl'] - base_stats['pnl'], 1),
            })

        # Aggregate WF results
        avg_wr_delta = np.mean([f['wr_delta'] for f in fold_results])
        avg_pnl_delta = np.mean([f['pnl_delta'] for f in fold_results])
        total_oos_pnl_base = sum(f['baseline_oos']['pnl'] for f in fold_results)
        total_oos_pnl_filt = sum(f['filtered_oos']['pnl'] for f in fold_results)
        all_positive = all(f['filtered_oos']['pnl'] > 0 for f in fold_results)
        folds_improved = sum(1 for f in fold_results if f['pnl_delta'] > 0)

        wf_results.append({
            'name': sc_name,
            'indicator': indicator,
            'params': sc_params,
            'folds': fold_results,
            'summary': {
                'avg_wr_delta': round(avg_wr_delta, 1),
                'avg_pnl_delta': round(avg_pnl_delta, 1),
                'total_oos_pnl_baseline': round(total_oos_pnl_base, 1),
                'total_oos_pnl_filtered': round(total_oos_pnl_filt, 1),
                'all_oos_positive': all_positive,
                'folds_improved': folds_improved,
                'verdict': 'PASS' if folds_improved >= 2 and all_positive else 'FAIL',
            }
        })

    return wf_results


def _rebuild_indicator(indicator, params, closes, highs_arr, lows_arr, candle_types):
    """Rebuild a single indicator array from params."""
    if indicator == 'SER':
        return compute_ser_vectorized(closes, [params['lookback']])[params['lookback']]
    elif indicator == 'MRT':
        return compute_mrt(closes)
    elif indicator == 'CBF':
        return compute_cbf(candle_types, [params['lookback']])[params['lookback']]
    elif indicator == 'VND':
        return compute_vnd(closes, highs_arr, lows_arr, [params['lookback']])[params['lookback']]
    elif indicator == 'PPP':
        return compute_ppp(closes, [params['lookback']])[params['lookback']]
    elif indicator == 'Hurst':
        return compute_hurst(closes, [params['lag']])[params['lag']]
    return np.full(len(closes), np.nan)


def _get_filter_type(indicator):
    """Map indicator name to filter type."""
    if indicator == 'PPP':
        return 'ppp'
    elif indicator == 'Hurst':
        return 'hurst'
    return 'directional'


def _get_threshold(params, indicator):
    """Extract threshold from params."""
    if indicator == 'PPP':
        return (params['low_threshold'], params['high_threshold'])
    return params.get('threshold', params.get('hurst_threshold', 0.5))


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Regime Filter Study — 6 Indicators × Multi-Parameter")
    logger.info("=" * 60)

    # 1. Load data
    df = load_and_classify(DATA_FILE)
    patterns = load_patterns(PATTERNS_FILE)
    n_bars = len(df)
    logger.info(f"Data: {n_bars} bars, IS: 0-{IS_END_BAR}, OOS: {IS_END_BAR}-{n_bars}")

    opens = df['open'].values
    highs_arr = df['high'].values
    lows_arr = df['low'].values
    closes = df['close'].values
    candle_types = df['candle_type'].values

    # 2. Build signal index + generate baseline
    signal_index = build_signal_index(candle_types.tolist(), n_bars)

    # Full-period baseline (for reference)
    all_trades_full = generate_all_signals(patterns, signal_index, opens, highs_arr, lows_arr, n_bars)
    port_full = portfolio_1pos_extended(all_trades_full)
    baseline_full = calc_stats([(eb, xb, pnl) for eb, xb, pnl, _, _ in port_full])
    logger.info(f"Full baseline: {baseline_full['trades']} trades, WR {baseline_full['wr']}%, PnL {baseline_full['pnl']}%")

    # IS baseline
    all_trades_is_raw = generate_all_signals(patterns, signal_index, opens, highs_arr, lows_arr, n_bars, bar_limit=IS_END_BAR)
    port_is = portfolio_1pos_extended(all_trades_is_raw)
    baseline_is = calc_stats([(eb, xb, pnl) for eb, xb, pnl, _, _ in port_is])
    logger.info(f"IS baseline: {baseline_is['trades']} trades, WR {baseline_is['wr']}%, PnL {baseline_is['pnl']}%")

    # OOS baseline
    all_trades_oos_raw = generate_all_signals(patterns, signal_index, opens, highs_arr, lows_arr, n_bars)
    port_oos = portfolio_1pos_extended(all_trades_oos_raw)
    port_oos_only = [(eb, xb, pnl, d, s) for eb, xb, pnl, d, s in port_oos if eb >= IS_END_BAR]
    baseline_oos = calc_stats([(eb, xb, pnl) for eb, xb, pnl, _, _ in port_oos_only])
    logger.info(f"OOS baseline: {baseline_oos['trades']} trades, WR {baseline_oos['wr']}%, PnL {baseline_oos['pnl']}%")

    # 3. Build scenarios and compute indicators
    scenarios = build_scenarios(closes, highs_arr, lows_arr, candle_types)

    # 4. IS backtest all scenarios
    logger.info(f"\nRunning IS backtest for {len(scenarios)} scenarios...")
    is_results = run_is_backtest(port_is, scenarios, baseline_is)
    logger.info(f"IS results: {len(is_results)} valid scenarios (≥10 trades)")

    # Print IS top-20
    print("\n" + "=" * 100)
    print(f"{'Rank':>4} {'Scenario':<35} {'Trades':>7} {'WR%':>6} {'PnL%':>8} {'MDD%':>7} {'PnL/MDD':>8} {'PF':>6} {'Filt%':>6} {'WR Δ':>6} {'PnL Δ':>8}")
    print("-" * 100)
    for i, r in enumerate(is_results[:20]):
        print(f"{i+1:>4} {r['name']:<35} {r['trades']:>7} {r['wr']:>6.1f} {r['pnl']:>8.1f} {r['mdd']:>7.1f} {r['pnl_mdd']:>8.2f} {r['pf']:>6.2f} {r['filtered_pct']:>6.1f} {r['wr_delta']:>+6.1f} {r['pnl_delta']:>+8.1f}")
    print("=" * 100)

    # Baseline row for reference
    bl_pnl_mdd = round(baseline_is['pnl'] / baseline_is['mdd'], 2) if baseline_is['mdd'] > 0 else 999
    print(f"{'BL':>4} {'BASELINE (no filter)':<35} {baseline_is['trades']:>7} {baseline_is['wr']:>6.1f} {baseline_is['pnl']:>8.1f} {baseline_is['mdd']:>7.1f} {bl_pnl_mdd:>8.2f} {baseline_is['pf']:>6.2f} {'0.0':>6} {'+0.0':>6} {'+0.0':>8}")

    # 5. Select top-10 for WF validation
    top_10 = is_results[:10]
    logger.info(f"\nTop-10 for WF validation:")
    for r in top_10:
        logger.info(f"  {r['name']}: PnL/MDD {r['pnl_mdd']}, WR {r['wr']}%, PnL {r['pnl']}%")

    # 6. WF validation
    wf_results = run_wf_validation(
        patterns, signal_index, opens, highs_arr, lows_arr,
        closes, candle_types, top_10, n_bars)

    # Print WF summary
    print("\n" + "=" * 100)
    print("WALK-FORWARD VALIDATION (3-fold Expanding Window)")
    print("=" * 100)
    for wf in wf_results:
        v = wf['summary']
        print(f"\n{wf['name']}  [{v['verdict']}]")
        print(f"  Avg WR Δ: {v['avg_wr_delta']:+.1f}pp  |  Avg PnL Δ: {v['avg_pnl_delta']:+.1f}%")
        print(f"  Total OOS PnL: Baseline {v['total_oos_pnl_baseline']:.1f}% → Filtered {v['total_oos_pnl_filtered']:.1f}%")
        print(f"  Folds improved: {v['folds_improved']}/3  |  All OOS positive: {v['all_oos_positive']}")
        for f in wf['folds']:
            b = f['baseline_oos']
            fi = f['filtered_oos']
            print(f"    Fold {f['fold']}: OOS {f['oos_bars']} | Base: {b['trades']}t WR {b['wr']}% PnL {b['pnl']}% | Filt: {fi['trades']}t WR {fi['wr']}% PnL {fi['pnl']}% | Δ WR {f['wr_delta']:+.1f} PnL {f['pnl_delta']:+.1f}")

    # 7. OOS test for top-10 (simple holdout)
    print("\n" + "=" * 100)
    print("OOS HOLDOUT TEST (30% holdout)")
    print("=" * 100)
    oos_results = []
    for sc_info in top_10:
        indicator = sc_info['indicator']
        sc_params = sc_info['params']
        ind_values = _rebuild_indicator(indicator, sc_params, closes, highs_arr, lows_arr, candle_types)

        if 'Hurst+' in indicator:
            hurst_lag = sc_params['lag']
            hurst_arr = compute_hurst(closes, [hurst_lag])[hurst_lag]
            ser_arr = compute_ser_vectorized(closes, [24])[24]
            filtered = apply_hurst_directional_filter(
                port_oos_only, hurst_arr, ser_arr, sc_params['hurst_threshold'])
        else:
            ft = _get_filter_type(indicator)
            threshold = _get_threshold(sc_params, indicator)
            filtered = apply_regime_filter(port_oos_only, ind_values, threshold, ft)

        filt_stats = calc_stats([(eb, xb, pnl) for eb, xb, pnl, _, _ in filtered])
        oos_results.append({
            'name': sc_info['name'],
            'baseline': baseline_oos,
            'filtered': filt_stats,
            'wr_delta': round(filt_stats['wr'] - baseline_oos['wr'], 1),
            'pnl_delta': round(filt_stats['pnl'] - baseline_oos['pnl'], 1),
        })
        print(f"  {sc_info['name']:<35} | Base: {baseline_oos['trades']}t WR {baseline_oos['wr']}% PnL {baseline_oos['pnl']}% | Filt: {filt_stats['trades']}t WR {filt_stats['wr']}% PnL {filt_stats['pnl']}% | Δ WR {oos_results[-1]['wr_delta']:+.1f} PnL {oos_results[-1]['pnl_delta']:+.1f}")

    # 8. Save results
    elapsed = time.time() - t_start
    output = {
        'study': 'regime_filter_study',
        'version': '1.0',
        'generated_at': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data': {
            'file': os.path.basename(DATA_FILE),
            'bars': n_bars,
            'is_bars': f"0-{IS_END_BAR}",
            'oos_bars': f"{IS_END_BAR}-{n_bars}",
        },
        'patterns': {
            'count': len(patterns),
            'source': os.path.basename(PATTERNS_FILE),
        },
        'baseline': {
            'full': baseline_full,
            'is': baseline_is,
            'oos': baseline_oos,
        },
        'scenarios_tested': len(scenarios),
        'is_results_top20': is_results[:20],
        'wf_results': wf_results,
        'oos_holdout_results': oos_results,
        'conclusion': _generate_conclusion(wf_results, oos_results, baseline_oos),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {OUTPUT_FILE}")
    logger.info(f"Total time: {elapsed:.1f}s")

    # Final summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    passed = [w for w in wf_results if w['summary']['verdict'] == 'PASS']
    if passed:
        print(f"  {len(passed)} filters PASSED WF validation:")
        for p in passed:
            s = p['summary']
            print(f"    {p['name']}: OOS PnL {s['total_oos_pnl_filtered']:.1f}% (baseline {s['total_oos_pnl_baseline']:.1f}%), WR Δ {s['avg_wr_delta']:+.1f}pp")
    else:
        print("  No filters passed WF validation (≥2/3 folds improved + all OOS positive).")
        print("  Regime filters do not reliably improve the portfolio on out-of-sample data.")
    print("=" * 60)


def _generate_conclusion(wf_results, oos_results, baseline_oos):
    """Generate structured conclusion."""
    passed_wf = [w for w in wf_results if w['summary']['verdict'] == 'PASS']
    improved_oos = [o for o in oos_results if o['pnl_delta'] > 0]

    return {
        'wf_passed': len(passed_wf),
        'wf_total': len(wf_results),
        'oos_improved': len(improved_oos),
        'oos_total': len(oos_results),
        'best_filter': passed_wf[0]['name'] if passed_wf else None,
        'recommendation': (
            f"Use {passed_wf[0]['name']} as portfolio regime filter"
            if passed_wf else
            "No regime filter recommended — none passed WF validation"
        ),
    }


if __name__ == '__main__':
    main()
