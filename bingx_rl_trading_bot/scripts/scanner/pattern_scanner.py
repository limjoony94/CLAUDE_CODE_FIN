#!/usr/bin/env python3
"""
Pattern Scanner CLI v2.3 — Dynamic Walk-Forward Pattern Selection

Standalone offline tool that scans historical data to select patterns
with genuine edge. Supports Universal and Per-Pattern TP/SL discovery.

Discovery methods:
- universal: Fixed TP/SL for all patterns (v1.28.0 original)
- per_pattern: Grid search optimal TP/SL per pattern (v1.28.6, default)
  PP discovery: +487% avg OOS vs Universal +18% (fair_discovery_comparison.py)
- mae_mfe: Derive TP/SL from MAE/MFE percentiles per pattern (v1.28.39)
  MAE/MFE discovery: WF OOS 2.4x vs grid search (mae_mfe_discovery.py)

v2.3 Enhancements:
- Neutral window discovery (default, --no-neutral to disable) — direction-balanced patterns
  Neutral window auto-finds longest price-flat window to avoid directional bias (v1.36.3)
- ATR-scaled TP/SL (default, --no-atr to disable) — Scanner-Production alignment
  ATR scanner vs Fixed: +64% OOS PnL (atr_scanner_alignment_study.py)
- Multiple testing correction (BH FDR / Bonferroni)
- Expanding window walk-forward validation
- Parallel grid search (ProcessPoolExecutor)
- Progress bars (tqdm) and timing

Usage:
  python scripts/scanner/pattern_scanner.py                              # PP + ATR + Neutral (default)
  python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe   # MAE/MFE + ATR + Neutral
  python scripts/scanner/pattern_scanner.py --no-neutral --is-days 270   # Disable neutral window
  python scripts/scanner/pattern_scanner.py --neutral-tol 2.0            # Wider price tolerance
  python scripts/scanner/pattern_scanner.py --no-atr                     # Fixed TP/SL (no ATR)
  python scripts/scanner/pattern_scanner.py --correction bh --wf-folds 3  # With corrections + WF
  python scripts/scanner/pattern_scanner.py --concurrency 4 -v           # 4 workers, verbose
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None

# Add project root to path for production imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

logger = logging.getLogger('pattern_scanner')

# ============================================================
# Constants (from v7 True Walk-Forward, verified)
# ============================================================
FEE_PCT = 0.10          # 0.05% entry + 0.05% exit
LEVERAGE = 3            # Position leverage
MAX_BARS = 288          # Max bars to hold trade (24h at 5m); v1.28.24: 500→288 (24h study)
MC_SIMS = 5000          # Monte Carlo simulations
DEFAULT_UNI_TP = 2.0    # Universal TP % (v1.28.2: WF frontier optimal)
DEFAULT_UNI_SL = 3.0    # Universal SL %
DEFAULT_EDGE_THRESHOLD = 10.0  # Minimum edge in pp (v1.28.4: 5->10 for statistical rigor)
DEFAULT_MC_THRESHOLD = 0.01    # MC p-value cutoff
DEFAULT_MIN_TRADES = 25        # v1.28.6: 20->25 (scanner_param_study: OOS PnL/MDD 2.38 best)

# Per-pattern discovery grid (v1.28.6)
TP_GRID = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.5, 3.0]
SL_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
MAX_BASELINE_WR = 70.0  # Skip combos where distance effect dominates (R:R >= 0.43)
MC_SEEDS = [42, 123, 7777]  # Multi-seed for robustness

# MAE/MFE percentile grids (v1.28.39: MAE/MFE discovery method)
TP_PERCENTILES = [40, 50, 60, 70, 80]
SL_PERCENTILES = [70, 75, 80, 85, 90, 95]

SLIPPAGE_BUFFER = 0.02  # 0.02% slippage buffer for ATR-scaled TP/SL

# ATR scaling defaults (v1.28.42 production params, a14/w576/0.6-1.7)
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_WINDOW = 576
DEFAULT_ATR_CLAMP_LO = 0.6
DEFAULT_ATR_CLAMP_HI = 1.7

DEFAULT_DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
DEFAULT_OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')


# ============================================================
# Core Functions
# ============================================================

def load_and_classify(data_file: str) -> pd.DataFrame:
    """Load CSV and classify candles using production classify_candle()."""
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)

    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate avg_body for classification
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW).mean()

    # Classify each candle using production function (match NaN handling)
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


def find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=288):
    """Find longest window where start_price ≈ end_price within tolerance.

    Scans all possible start points and finds the farthest end point where
    the price is within ±tol_pct of the start price. Returns the longest
    such window to maximize data utilization while ensuring directional balance.

    Args:
        closes: Array of close prices.
        tol_pct: Price tolerance in percent (default 1.0 = ±1%).
        min_days: Minimum window length in days (default 90).
        bars_per_day: Bars per day (288 for 5m, 96 for 15m).

    Returns:
        (start_bar, end_bar) tuple, or None if no valid window found.
    """
    n = len(closes)
    min_bars = min_days * bars_per_day
    best_len, best_start, best_end = 0, 0, 0

    for i in range(0, n - min_bars, bars_per_day):
        start_price = closes[i]
        lo = start_price * (1 - tol_pct / 100)
        hi = start_price * (1 + tol_pct / 100)
        for j in range(n - 1, i + min_bars - 1, -bars_per_day):
            if lo <= closes[j] <= hi:
                length = j - i
                if length > best_len:
                    best_len = length
                    best_start, best_end = i, j
                break

    if best_len == 0:
        return None
    return best_start, best_end


def build_signal_index(types, n):
    """Build index: pattern_name -> list of signal bar indices."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """
    Backtest given signal bars with TP/SL.

    Entry: next bar open after signal.
    Exit: intrabar high/low check (distance-based).
    Timeout trades are DROPPED (not counted).
    """
    # Fee as % of capital: price-space fee × leverage (fee is on notional)
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
                # Both hit same bar — bar open distance-based resolution
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
        # Timeout trades are dropped (no append)

    return trades


def compute_atr_ratio(highs, lows, closes, atr_period=14, window=576):
    """ATR / rolling_median(ATR) ratio time series.

    Uses EMA-style ATR (Wilder smoothing) for consistency with production.
    Returns array of same length as input; early values are NaN.
    """
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    # Wilder EMA ATR
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period - 1] = tr[:atr_period].mean()
        for i in range(atr_period, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                    opens, highs, lows, n_bars,
                    atr_ratio, clamp_lo=0.6, clamp_hi=1.7):
    """Backtest with ATR-scaled TP/SL per signal.

    Same protocol as bt_signals() but applies ATR ratio scaling
    to TP/SL at each signal bar. Timeout trades are DROPPED.
    """
    fee = FEE_PCT * LEVERAGE
    is_long = (direction == 'LONG')
    trades = []

    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1

        # ATR scaling at signal bar
        if (atr_ratio is not None and idx < len(atr_ratio)
                and not np.isnan(atr_ratio[idx])):
            r = max(clamp_lo, min(clamp_hi, atr_ratio[idx]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE_BUFFER
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

        if is_long:
            tpp = entry * (1 + eff_tp / 100)
            slp = entry * (1 - eff_sl / 100)
        else:
            tpp = entry * (1 - eff_tp / 100)
            slp = entry * (1 + eff_sl / 100)

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if is_long:
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (eff_tp if dist_tp <= dist_sl else -eff_sl) * LEVERAGE - fee
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, eff_tp * LEVERAGE - fee))
                break
            elif hs:
                trades.append((eb, j, -eff_sl * LEVERAGE - fee))
                break
        # Timeout → DROP

    return trades


def mc_test(pnls, n_sims=MC_SIMS):
    """Multi-seed sign randomization MC test. Returns max p-value (conservative)."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(n_sims, len(pnls_arr)))
        rand_sums = signs @ pnls_arr
        p_vals.append(float(np.mean(rand_sums >= actual)))
    return max(p_vals)


def apply_multiple_testing_correction(selected, n_tested, method='none',
                                      fdr_q=0.05, alpha=0.01):
    """Apply multiple testing correction to selected patterns.

    Args:
        selected: dict of pattern_key -> pattern_info (must have 'mc_p')
        n_tested: total number of pattern-direction combos tested
        method: 'none' | 'bh' | 'bonferroni'
        fdr_q: FDR q-value for BH correction (default 0.05)
        alpha: significance level for Bonferroni (default 0.01)

    Returns:
        (filtered_selected, correction_meta)
    """
    if method == 'none' or not selected:
        return selected, {
            'correction_method': 'none', 'n_tested': n_tested,
            'n_before_correction': len(selected),
            'n_after_correction': len(selected),
        }

    # Sort by p-value ascending
    sorted_items = sorted(selected.items(), key=lambda x: x[1]['mc_p'])

    if method == 'bonferroni':
        threshold = alpha / n_tested if n_tested > 0 else alpha
        filtered = {k: v for k, v in sorted_items if v['mc_p'] <= threshold}
        meta = {
            'correction_method': 'bonferroni', 'n_tested': n_tested,
            'bonf_threshold': threshold, 'alpha': alpha,
            'n_before_correction': len(selected),
            'n_after_correction': len(filtered),
        }
        logger.info(f"Bonferroni: threshold={threshold:.6f}, "
                     f"{len(selected)} -> {len(filtered)} patterns")
        return filtered, meta

    elif method == 'bh':
        # Benjamini-Hochberg step-up: find largest k where p(k) <= q*k/m
        # m = total hypotheses tested (not just selected), for proper FDR control
        m = max(n_tested, len(sorted_items))
        last_pass_idx = -1
        for i in range(len(sorted_items) - 1, -1, -1):
            rank = i + 1
            bh_threshold = fdr_q * rank / m
            if sorted_items[i][1]['mc_p'] <= bh_threshold:
                last_pass_idx = i
                break

        filtered = dict(sorted_items[:last_pass_idx + 1]) if last_pass_idx >= 0 else {}
        meta = {
            'correction_method': 'bh', 'n_tested': n_tested,
            'fdr_q': fdr_q, 'm_hypotheses': m,
            'n_before_correction': len(selected),
            'n_after_correction': len(filtered),
        }
        logger.info(f"BH FDR: q={fdr_q}, "
                     f"{len(selected)} -> {len(filtered)} patterns")
        return filtered, meta

    else:
        raise ValueError(f"Unknown correction method: {method}")


def grid_search_best(signal_bars, direction, opens, highs, lows, n_bars,
                     min_tr=20, max_baseline_wr=MAX_BASELINE_WR,
                     atr_ratio=None, clamp_lo=0.6, clamp_hi=1.7):
    """Grid search for best TP/SL by PnL/MDD. Skips combos with baseline WR > max_baseline_wr."""
    best = None
    best_score = -9999
    for tp in TP_GRID:
        for sl in SL_GRID:
            if sl < 0.5:
                continue
            bwr = sl / (tp + sl) * 100
            if bwr > max_baseline_wr:
                continue
            if atr_ratio is not None:
                trades = bt_signals_atr(signal_bars, direction, tp, sl,
                                        opens, highs, lows, n_bars,
                                        atr_ratio, clamp_lo, clamp_hi)
            else:
                trades = bt_signals(signal_bars, direction, tp, sl, opens, highs, lows, n_bars)
            if len(trades) < min_tr:
                continue
            pnls = [t[2] for t in trades]
            cum = 0; peak = 0; mdd_val = 0; w = 0
            for p in pnls:
                cum += p
                if cum > peak: peak = cum
                dd = peak - cum
                if dd > mdd_val: mdd_val = dd
                if p > 0: w += 1
            wr = w / len(pnls) * 100
            score = (cum / mdd_val) if mdd_val > 0 else cum
            if score > best_score:
                best_score = score
                best = {'tp': tp, 'sl': sl, 'trades': len(trades),
                        'wr': round(wr, 1), 'pnl': round(cum, 1), 'mdd': round(mdd_val, 1)}
    return best


def compute_excursions(signal_bars, direction, opens, highs, lows, n_bars,
                       max_bars=MAX_BARS):
    """Compute MFE and MAE for each trade (without TP/SL, natural path).

    Returns list of dicts with mfe_pct, mae_pct, bars_to_mfe, final_pnl_pct.
    """
    excursions = []
    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue

        mfe = 0.0
        mae = 0.0
        bars_to_mfe = 0

        end_bar = min(idx + 2 + max_bars, n_bars)
        for j in range(idx + 2, end_bar):
            if direction == 'LONG':
                favorable = highs[j] - entry
                adverse = entry - lows[j]
            else:
                favorable = entry - lows[j]
                adverse = highs[j] - entry

            if favorable > mfe:
                mfe = favorable
                bars_to_mfe = j - (idx + 1)
            if adverse > mae:
                mae = adverse

        mfe_pct = mfe / entry * 100
        mae_pct = mae / entry * 100

        last_bar = min(idx + 1 + max_bars, n_bars - 1)
        if direction == 'LONG':
            final_pnl = (opens[last_bar] - entry) / entry * 100
        else:
            final_pnl = (entry - opens[last_bar]) / entry * 100

        excursions.append({
            'signal_bar': idx,
            'entry': entry,
            'mfe_pct': round(mfe_pct, 4),
            'mae_pct': round(mae_pct, 4),
            'bars_to_mfe': bars_to_mfe,
            'final_pnl_pct': round(final_pnl, 4),
        })

    return excursions


def derive_tp_sl(excursions, tp_percentile, sl_percentile):
    """Derive TP/SL from MFE/MAE percentiles."""
    if not excursions:
        return None, None
    mfes = [e['mfe_pct'] for e in excursions]
    maes = [e['mae_pct'] for e in excursions]
    tp = round(float(np.percentile(mfes, tp_percentile)), 2)
    sl = round(float(np.percentile(maes, sl_percentile)), 2)
    return tp, sl


def grid_search_mae_mfe(signal_bars, direction, opens, highs, lows, n_bars,
                         max_bars=MAX_BARS, min_tr=20,
                         max_baseline_wr=MAX_BASELINE_WR,
                         atr_ratio=None, clamp_lo=0.6, clamp_hi=1.7,
                         tp_max=None, sl_max=None):
    """Grid search over MFE/MAE percentiles to find best TP/SL by PnL/MDD."""
    excursions = compute_excursions(signal_bars, direction, opens, highs, lows,
                                    n_bars, max_bars)
    if len(excursions) < min_tr:
        return None, None

    best = None
    best_score = -9999

    for tp_p in TP_PERCENTILES:
        for sl_p in SL_PERCENTILES:
            tp, sl = derive_tp_sl(excursions, tp_p, sl_p)
            if tp is None or sl is None:
                continue
            if tp < 0.1 or sl < 0.3:
                continue
            if tp_max is not None and tp > tp_max:
                continue
            if sl_max is not None and sl > sl_max:
                continue

            bwr = sl / (tp + sl) * 100
            if bwr > max_baseline_wr:
                continue

            if atr_ratio is not None:
                trades = bt_signals_atr(signal_bars, direction, tp, sl,
                                        opens, highs, lows, n_bars,
                                        atr_ratio, clamp_lo, clamp_hi)
            else:
                trades = bt_signals(signal_bars, direction, tp, sl,
                                    opens, highs, lows, n_bars)
            if len(trades) < min_tr:
                continue

            pnls = [t[2] for t in trades]
            cum = 0; peak = 0; mdd_val = 0; w = 0
            for p in pnls:
                cum += p
                if cum > peak: peak = cum
                dd = peak - cum
                if dd > mdd_val: mdd_val = dd
                if p > 0: w += 1

            wr = w / len(pnls) * 100
            score = (cum / mdd_val) if mdd_val > 0 else cum

            if score > best_score:
                best_score = score
                best = {
                    'tp': tp, 'sl': sl,
                    'tp_percentile': tp_p, 'sl_percentile': sl_p,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'pnl': round(cum, 1), 'mdd': round(mdd_val, 1),
                }

    # Excursion stats for profiling
    mfes = np.array([e['mfe_pct'] for e in excursions])
    maes = np.array([e['mae_pct'] for e in excursions])
    bars_mfe = np.array([e['bars_to_mfe'] for e in excursions])

    exc_stats = {
        'n_signals': len(excursions),
        'mfe_median': round(float(np.median(mfes)), 3),
        'mfe_mean': round(float(np.mean(mfes)), 3),
        'mae_median': round(float(np.median(maes)), 3),
        'mae_mean': round(float(np.mean(maes)), 3),
        'mae_p90': round(float(np.percentile(maes, 90)), 3),
        'bars_to_mfe_median': round(float(np.median(bars_mfe)), 1),
        'mfe_mae_ratio': round(float(np.median(mfes) / np.median(maes))
                               if np.median(maes) > 0 else 0, 3),
    }

    return best, exc_stats


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
    """Calculate portfolio statistics from trade list."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    pnls = [t[2] for t in trades]
    cum = 0
    peak = 0
    mdd = 0
    wins = []
    losses = []
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
        else:
            losses.append(p)
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(cum, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
    }


# ============================================================
# Parallel PP Worker (module-level for pickling)
# ============================================================

_shared_data = {}


def _pp_init(opens, highs, lows, n_bars,
             atr_ratio=None, clamp_lo=0.6, clamp_hi=1.7,
             tp_max=None, sl_max=None):
    """Initialize shared data in worker processes."""
    _shared_data['opens'] = opens
    _shared_data['highs'] = highs
    _shared_data['lows'] = lows
    _shared_data['n_bars'] = n_bars
    _shared_data['atr_ratio'] = atr_ratio
    _shared_data['clamp_lo'] = clamp_lo
    _shared_data['clamp_hi'] = clamp_hi
    _shared_data['tp_max'] = tp_max
    _shared_data['sl_max'] = sl_max


def _pp_worker(args_tuple):
    """Worker for parallel per-pattern grid search.

    Returns result dict or None if pattern doesn't pass filters.
    """
    (pat_name, direction, sigs_list,
     min_trades, edge_threshold, mc_threshold, max_baseline_wr) = args_tuple

    opens = _shared_data['opens']
    highs = _shared_data['highs']
    lows = _shared_data['lows']
    n_bars = _shared_data['n_bars']
    atr_ratio = _shared_data.get('atr_ratio')
    clamp_lo = _shared_data.get('clamp_lo', 0.6)
    clamp_hi = _shared_data.get('clamp_hi', 1.7)

    if len(sigs_list) < min_trades:
        return None

    opt = grid_search_best(sigs_list, direction, opens, highs, lows, n_bars,
                           min_tr=min_trades, max_baseline_wr=max_baseline_wr,
                           atr_ratio=atr_ratio, clamp_lo=clamp_lo, clamp_hi=clamp_hi)
    if opt is None:
        return None

    tp, sl = opt['tp'], opt['sl']
    if atr_ratio is not None:
        trades = bt_signals_atr(sigs_list, direction, tp, sl,
                                opens, highs, lows, n_bars,
                                atr_ratio, clamp_lo, clamp_hi)
    else:
        trades = bt_signals(sigs_list, direction, tp, sl, opens, highs, lows, n_bars)
    if len(trades) < min_trades:
        return None

    pnls = [t[2] for t in trades]
    wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
    baseline_wr = sl / (tp + sl) * 100
    edge = wr - baseline_wr

    if edge < edge_threshold:
        return None

    p = mc_test(pnls)
    if p >= mc_threshold:
        return None

    key = f"{pat_name}_{direction}"
    return {
        'key': key,
        'pattern': pat_name, 'direction': direction,
        'tp': tp, 'sl': sl,
        'trades': len(trades), 'wr': round(wr, 1),
        'edge': round(edge, 1), 'mc_p': round(p, 4),
        'baseline_wr': round(baseline_wr, 1),
        'trades_raw': trades,
    }


def _mae_mfe_worker(args_tuple):
    """Worker for parallel MAE/MFE grid search.

    Returns result dict or None if pattern doesn't pass filters.
    """
    (pat_name, direction, sigs_list,
     min_trades, edge_threshold, mc_threshold, max_baseline_wr) = args_tuple

    opens = _shared_data['opens']
    highs = _shared_data['highs']
    lows = _shared_data['lows']
    n_bars = _shared_data['n_bars']
    atr_ratio = _shared_data.get('atr_ratio')
    clamp_lo = _shared_data.get('clamp_lo', 0.6)
    clamp_hi = _shared_data.get('clamp_hi', 1.7)
    tp_max = _shared_data.get('tp_max')
    sl_max = _shared_data.get('sl_max')

    if len(sigs_list) < min_trades:
        return None

    opt, exc_stats = grid_search_mae_mfe(
        sigs_list, direction, opens, highs, lows, n_bars,
        max_bars=MAX_BARS, min_tr=min_trades,
        max_baseline_wr=max_baseline_wr,
        atr_ratio=atr_ratio, clamp_lo=clamp_lo, clamp_hi=clamp_hi,
        tp_max=tp_max, sl_max=sl_max)
    if opt is None:
        return None

    tp, sl = opt['tp'], opt['sl']
    if atr_ratio is not None:
        trades = bt_signals_atr(sigs_list, direction, tp, sl,
                                opens, highs, lows, n_bars,
                                atr_ratio, clamp_lo, clamp_hi)
    else:
        trades = bt_signals(sigs_list, direction, tp, sl, opens, highs, lows, n_bars)
    if len(trades) < min_trades:
        return None

    pnls = [t[2] for t in trades]
    wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
    baseline_wr = sl / (tp + sl) * 100
    edge = wr - baseline_wr

    if edge < edge_threshold:
        return None

    p = mc_test(pnls)
    if p >= mc_threshold:
        return None

    key = f"{pat_name}_{direction}"
    return {
        'key': key,
        'pattern': pat_name, 'direction': direction,
        'tp': tp, 'sl': sl,
        'tp_percentile': opt['tp_percentile'],
        'sl_percentile': opt['sl_percentile'],
        'trades': len(trades), 'wr': round(wr, 1),
        'edge': round(edge, 1), 'mc_p': round(p, 4),
        'baseline_wr': round(baseline_wr, 1),
        'exc_stats': exc_stats,
        'trades_raw': trades,
    }


# ============================================================
# Range-Based Scanning (for Walk-Forward)
# ============================================================

def scan_universe_range(signal_index, opens, highs, lows, n_bars,
                        bar_start, bar_end, mode,
                        uni_tp=None, uni_sl=None,
                        min_trades=DEFAULT_MIN_TRADES,
                        edge_threshold=DEFAULT_EDGE_THRESHOLD,
                        mc_threshold=DEFAULT_MC_THRESHOLD,
                        max_baseline_wr=MAX_BASELINE_WR,
                        atr_ratio=None, clamp_lo=0.6, clamp_hi=1.7,
                        tp_max=None, sl_max=None):
    """Scan patterns within [bar_start, bar_end) signal range.

    Supports both 'universal' and 'per_pattern' modes.
    Returns list of selected pattern dicts with trades_raw.
    Uses bar_end as trade boundary to prevent data leakage beyond the range.
    """
    selected = []
    # Use bar_end as boundary for bt_signals to prevent data leakage
    trade_boundary = bar_end if bar_end is not None else n_bars

    for pat_name, all_sigs in signal_index.items():
        sigs = [s for s in all_sigs if bar_start <= s < bar_end]

        for direction in ['LONG', 'SHORT']:
            if len(sigs) < min_trades:
                continue

            # Select bt function based on ATR availability
            def _bt(sig_bars, d, tp_v, sl_v):
                if atr_ratio is not None:
                    return bt_signals_atr(sig_bars, d, tp_v, sl_v,
                                          opens, highs, lows, trade_boundary,
                                          atr_ratio, clamp_lo, clamp_hi)
                return bt_signals(sig_bars, d, tp_v, sl_v,
                                  opens, highs, lows, trade_boundary)

            if mode == 'universal':
                tp, sl = uni_tp, uni_sl
                trades = _bt(sigs, direction, tp, sl)
                if len(trades) < min_trades:
                    continue
            elif mode == 'per_pattern':
                opt = grid_search_best(sigs, direction, opens, highs, lows,
                                       trade_boundary, min_tr=min_trades,
                                       max_baseline_wr=max_baseline_wr,
                                       atr_ratio=atr_ratio, clamp_lo=clamp_lo,
                                       clamp_hi=clamp_hi)
                if opt is None:
                    continue
                tp, sl = opt['tp'], opt['sl']
                trades = _bt(sigs, direction, tp, sl)
                if len(trades) < min_trades:
                    continue
            elif mode == 'mae_mfe':
                opt, _ = grid_search_mae_mfe(
                    sigs, direction, opens, highs, lows,
                    trade_boundary, max_bars=MAX_BARS,
                    min_tr=min_trades, max_baseline_wr=max_baseline_wr,
                    atr_ratio=atr_ratio, clamp_lo=clamp_lo, clamp_hi=clamp_hi,
                    tp_max=tp_max, sl_max=sl_max)
                if opt is None:
                    continue
                tp, sl = opt['tp'], opt['sl']
                trades = _bt(sigs, direction, tp, sl)
                if len(trades) < min_trades:
                    continue
            else:
                raise ValueError(f"Unknown mode: {mode}")

            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            baseline_wr = sl / (tp + sl) * 100
            edge = wr - baseline_wr

            if edge < edge_threshold:
                continue

            p = mc_test(pnls)
            if p >= mc_threshold:
                continue

            selected.append({
                'pattern': pat_name, 'direction': direction,
                'tp': tp, 'sl': sl,
                'trades': len(trades), 'wr': round(wr, 1),
                'edge': round(edge, 1), 'mc_p': round(p, 4),
                'trades_raw': trades,
            })

    return selected


def expanding_window_wf(signal_index, opens, highs, lows, n_bars,
                        n_folds, mode,
                        uni_tp=None, uni_sl=None,
                        min_trades=DEFAULT_MIN_TRADES,
                        edge_threshold=DEFAULT_EDGE_THRESHOLD,
                        mc_threshold=DEFAULT_MC_THRESHOLD,
                        max_baseline_wr=MAX_BASELINE_WR,
                        closes=None, atr_period=14, atr_window=576,
                        clamp_lo=0.6, clamp_hi=1.7, use_atr=True,
                        tp_max=None, sl_max=None):
    """True expanding window walk-forward validation.

    Splits data into n_folds+1 equal segments.
    Fold f: IS=[0, (f+1)*seg), OOS=[(f+1)*seg, (f+2)*seg)

    Returns dict with folds, positive_folds, total_oos_pnl, stable_patterns.
    """
    seg_size = n_bars // (n_folds + 1)
    folds = []
    pattern_appearances = Counter()

    for fold in range(n_folds):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        logger.info(f"  WF Fold {fold+1}/{n_folds}: "
                     f"IS=[0, {is_end}), OOS=[{oos_start}, {oos_end})")

        # Compute ATR ratio on IS data only (prevent look-ahead bias)
        fold_atr = None
        if use_atr and closes is not None:
            fold_atr = compute_atr_ratio(
                highs[:is_end], lows[:is_end], closes[:is_end],
                atr_period, window=atr_window)

        # Fresh scan on IS range
        is_patterns = scan_universe_range(
            signal_index, opens, highs, lows, n_bars,
            bar_start=0, bar_end=is_end, mode=mode,
            uni_tp=uni_tp, uni_sl=uni_sl,
            min_trades=min_trades, edge_threshold=edge_threshold,
            mc_threshold=mc_threshold, max_baseline_wr=max_baseline_wr,
            atr_ratio=fold_atr, clamp_lo=clamp_lo, clamp_hi=clamp_hi,
            tp_max=tp_max, sl_max=sl_max,
        )

        # Track pattern stability
        for r in is_patterns:
            pattern_appearances[(r['pattern'], r['direction'])] += 1

        # OOS backtest with IS-discovered patterns and their TP/SL
        # Compute ATR on full data up to oos_end for OOS evaluation
        oos_atr = None
        if use_atr and closes is not None:
            oos_atr = compute_atr_ratio(
                highs[:oos_end], lows[:oos_end], closes[:oos_end],
                atr_period, window=atr_window)

        oos_trades = []
        for r in is_patterns:
            oos_sigs = [s for s in signal_index[r['pattern']]
                        if oos_start <= s < oos_end]
            if oos_atr is not None:
                oos_trades.extend(bt_signals_atr(
                    oos_sigs, r['direction'], r['tp'], r['sl'],
                    opens, highs, lows, oos_end,
                    oos_atr, clamp_lo, clamp_hi
                ))
            else:
                oos_trades.extend(bt_signals(
                    oos_sigs, r['direction'], r['tp'], r['sl'],
                    opens, highs, lows, oos_end
                ))

        oos_port = portfolio_1pos(oos_trades)
        oos_stats = calc_stats(oos_port)
        n_long = sum(1 for r in is_patterns if r['direction'] == 'LONG')
        n_short = sum(1 for r in is_patterns if r['direction'] == 'SHORT')

        fold_result = {
            'fold': fold + 1,
            'is_bars': is_end,
            'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_patterns),
            'is_long': n_long, 'is_short': n_short,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_mdd': oos_stats['mdd'],
            'oos_pf': oos_stats['pf'],
            'oos_positive': oos_stats['pnl'] > 0,
        }
        folds.append(fold_result)

        logger.info(f"    IS: {len(is_patterns)} pat ({n_long}L+{n_short}S) | "
                     f"OOS: {oos_stats['trades']} tr, "
                     f"WR {oos_stats['wr']}%, PnL {oos_stats['pnl']}%")

    # Stable patterns = appear in ALL folds
    stable = sorted(
        f"{pat}_{d}"
        for (pat, d), cnt in pattern_appearances.items()
        if cnt == n_folds
    )
    positive_folds = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_pnl'] for f in folds), 1)
    total_oos_trades = sum(f['oos_trades'] for f in folds)

    return {
        'n_folds': n_folds,
        'folds': folds,
        'positive_folds': positive_folds,
        'total_oos_pnl': total_oos_pnl,
        'total_oos_trades': total_oos_trades,
        'stable_pattern_count': len(stable),
        'stable_patterns': stable,
    }


# ============================================================
# Main Scanner Logic
# ============================================================

def scan_patterns(
    df: pd.DataFrame,
    uni_tp: float = DEFAULT_UNI_TP,
    uni_sl: float = DEFAULT_UNI_SL,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    mc_threshold: float = DEFAULT_MC_THRESHOLD,
    min_trades: int = DEFAULT_MIN_TRADES,
    signal_index: dict = None,
    correction_method: str = 'none',
    fdr_q: float = 0.05,
    require_portfolio_mc: bool = False,
) -> dict:
    """
    Scan all 3456 patterns (12^3 x 2 directions) and filter by edge + MC.

    Returns dict with selected patterns and backtest summary.
    """
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    # Build signal index if not provided
    if signal_index is None:
        signal_index = build_signal_index(types, n)
    logger.info(f"Built signal index: {len(signal_index)} unique patterns found")

    # Baseline WR for random walk
    baseline_wr = uni_sl / (uni_tp + uni_sl) * 100

    # Scan all patterns x directions
    selected = {}
    pattern_details = []
    n_tested = 0

    for pat_name in signal_index:
        for direction in ['LONG', 'SHORT']:
            sigs = signal_index[pat_name]
            if len(sigs) < min_trades:
                continue

            trades = bt_signals(sigs, direction, uni_tp, uni_sl,
                                opens, highs, lows, n)
            if len(trades) < min_trades:
                continue

            n_tested += 1

            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            edge = wr - baseline_wr

            if edge < edge_threshold:
                continue

            # MC test
            p = mc_test(pnls)
            if p >= mc_threshold:
                continue

            key = f"{pat_name}_{direction}"
            selected[key] = {
                'pattern': pat_name,
                'direction': direction,
                'trades': len(trades),
                'wr': round(wr, 1),
                'edge': round(edge, 1),
                'mc_p': round(p, 4),
            }
            pattern_details.append({
                'key': key,
                'trades_raw': trades,
            })

    logger.info(f"Patterns passing filters: {len(selected)} (tested: {n_tested})")

    # Apply multiple testing correction
    selected, correction_meta = apply_multiple_testing_correction(
        selected, n_tested, method=correction_method, fdr_q=fdr_q
    )

    # Rebuild pattern_details after correction
    surviving_keys = set(selected.keys())
    pattern_details = [pd_item for pd_item in pattern_details
                       if pd_item['key'] in surviving_keys]

    # Build portfolio and run 1-pos filter
    all_trades = []
    for pd_item in pattern_details:
        all_trades.extend(pd_item['trades_raw'])
    portfolio_trades = portfolio_1pos(all_trades)
    portfolio_stats = calc_stats(portfolio_trades)

    # MC test on portfolio
    if portfolio_trades:
        portfolio_pnls = [t[2] for t in portfolio_trades]
        portfolio_mc = mc_test(portfolio_pnls)
    else:
        portfolio_mc = 1.0

    portfolio_mc_pass = portfolio_mc < mc_threshold
    if require_portfolio_mc and not portfolio_mc_pass:
        logger.warning(f"Portfolio MC FAILED: p={portfolio_mc:.4f} >= {mc_threshold}")

    # Organize by direction
    long_patterns = sorted([v['pattern'] for v in selected.values()
                            if v['direction'] == 'LONG'])
    short_patterns = sorted([v['pattern'] for v in selected.values()
                             if v['direction'] == 'SHORT'])

    logger.info(f"Selected: {len(long_patterns)}L + {len(short_patterns)}S "
                f"= {len(selected)} patterns")
    logger.info(f"Portfolio: {portfolio_stats['trades']} trades, "
                f"WR {portfolio_stats['wr']}%, PnL {portfolio_stats['pnl']}%")

    return {
        'long_patterns': long_patterns,
        'short_patterns': short_patterns,
        'portfolio_stats': portfolio_stats,
        'portfolio_mc': round(portfolio_mc, 4),
        'portfolio_mc_pass': portfolio_mc_pass,
        'pattern_details': {k: v for k, v in selected.items()},
        'correction_meta': correction_meta,
    }


def scan_patterns_pp(
    df: pd.DataFrame,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    mc_threshold: float = DEFAULT_MC_THRESHOLD,
    min_trades: int = DEFAULT_MIN_TRADES,
    signal_index: dict = None,
    concurrency: int = 0,
    max_baseline_wr: float = MAX_BASELINE_WR,
    correction_method: str = 'none',
    fdr_q: float = 0.05,
    require_portfolio_mc: bool = False,
    atr_ratio=None, clamp_lo=0.6, clamp_hi=1.7,
) -> dict:
    """Per-Pattern Discovery: grid search optimal TP/SL per pattern.

    Args:
        concurrency: 0=auto (min(cpu_count, 8)), 1=sequential, N=N workers
    """
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    if signal_index is None:
        signal_index = build_signal_index(types, n)
    logger.info(f"Built signal index: {len(signal_index)} unique patterns found")

    # Build work items
    work_items = []
    for pat_name in signal_index:
        for direction in ['LONG', 'SHORT']:
            sigs = signal_index[pat_name]
            work_items.append(
                (pat_name, direction, sigs, min_trades,
                 edge_threshold, mc_threshold, max_baseline_wr)
            )

    # n_tested = combos with enough signals to be tested
    n_tested = sum(1 for item in work_items if len(item[2]) >= min_trades)
    logger.info(f"Pattern-direction combos to test: {n_tested} "
                f"(of {len(work_items)} total)")

    selected = {}
    pattern_details = []

    # Determine concurrency
    if concurrency == 0:
        n_workers = min(os.cpu_count() or 1, 8)
    else:
        n_workers = concurrency

    if n_workers > 1:
        # Parallel mode
        logger.info(f"Parallel grid search: {n_workers} workers")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pp_init,
            initargs=(opens, highs, lows, n, atr_ratio, clamp_lo, clamp_hi),
        ) as executor:
            futures = {executor.submit(_pp_worker, item): item
                       for item in work_items}
            iterator = as_completed(futures)
            if _tqdm_cls:
                iterator = _tqdm_cls(iterator, total=len(futures),
                                     desc="PP Grid Search")
            for future in iterator:
                result = future.result()
                if result is not None:
                    key = result['key']
                    trades_raw = result.pop('trades_raw')
                    selected[key] = result
                    pattern_details.append({
                        'key': key, 'trades_raw': trades_raw,
                    })
    else:
        # Sequential mode
        logger.info("Sequential grid search")
        _shared_data['opens'] = opens
        _shared_data['highs'] = highs
        _shared_data['lows'] = lows
        _shared_data['n_bars'] = n
        _shared_data['atr_ratio'] = atr_ratio
        _shared_data['clamp_lo'] = clamp_lo
        _shared_data['clamp_hi'] = clamp_hi

        iterator = work_items
        if _tqdm_cls:
            iterator = _tqdm_cls(iterator, desc="PP Grid Search")
        for item in iterator:
            result = _pp_worker(item)
            if result is not None:
                key = result['key']
                trades_raw = result.pop('trades_raw')
                selected[key] = result
                pattern_details.append({
                    'key': key, 'trades_raw': trades_raw,
                })

    logger.info(f"Patterns passing filters: {len(selected)} "
                f"(tested: {n_tested})")

    # Apply multiple testing correction
    selected, correction_meta = apply_multiple_testing_correction(
        selected, n_tested, method=correction_method, fdr_q=fdr_q
    )

    # Rebuild pattern_details after correction
    surviving_keys = set(selected.keys())
    pattern_details = [pd_item for pd_item in pattern_details
                       if pd_item['key'] in surviving_keys]

    # Deduplicate dual-direction patterns: keep direction with better edge
    pat_by_name = {}
    for k, v in selected.items():
        pn = v['pattern']
        if pn not in pat_by_name or v['edge'] > pat_by_name[pn]['edge']:
            pat_by_name[pn] = v
        else:
            logger.info(f"Dedup: {pn} keeping {pat_by_name[pn]['direction']} "
                        f"(edge {pat_by_name[pn]['edge']}pp) over "
                        f"{v['direction']} (edge {v['edge']}pp)")
    deduped = {f"{v['pattern']}_{v['direction']}": v
               for v in pat_by_name.values()}
    n_removed = len(selected) - len(deduped)
    if n_removed:
        logger.info(f"Removed {n_removed} dual-direction duplicates")
        selected = deduped
        # Rebuild pattern_details after dedup
        surviving_keys = set(selected.keys())
        pattern_details = [pd_item for pd_item in pattern_details
                           if pd_item['key'] in surviving_keys]

    # Portfolio 1-pos filter + stats (computed AFTER dedup for accuracy)
    all_trades = []
    for pd_item in pattern_details:
        all_trades.extend(pd_item['trades_raw'])
    portfolio_trades = portfolio_1pos(all_trades)
    portfolio_stats = calc_stats(portfolio_trades)
    portfolio_mc = (mc_test([t[2] for t in portfolio_trades])
                    if portfolio_trades else 1.0)

    portfolio_mc_pass = portfolio_mc < mc_threshold
    if require_portfolio_mc and not portfolio_mc_pass:
        logger.warning(f"Portfolio MC FAILED: p={portfolio_mc:.4f} "
                       f">= {mc_threshold}")

    # Organize by direction
    long_patterns = sorted([v['pattern'] for v in selected.values()
                            if v['direction'] == 'LONG'])
    short_patterns = sorted([v['pattern'] for v in selected.values()
                             if v['direction'] == 'SHORT'])

    # Build patterns_tpsl dict
    patterns_tpsl = {}
    for v in selected.values():
        patterns_tpsl[v['pattern']] = [v['tp'], v['sl']]

    # TP/SL distribution
    tps = [v['tp'] for v in selected.values()]
    sls = [v['sl'] for v in selected.values()]

    logger.info(f"Selected: {len(long_patterns)}L + {len(short_patterns)}S "
                f"= {len(selected)} patterns")
    logger.info(f"Portfolio: {portfolio_stats['trades']} trades, "
                f"WR {portfolio_stats['wr']}%, PnL {portfolio_stats['pnl']}%")

    return {
        'long_patterns': long_patterns,
        'short_patterns': short_patterns,
        'patterns_tpsl': patterns_tpsl,
        'portfolio_stats': portfolio_stats,
        'portfolio_mc': round(portfolio_mc, 4),
        'portfolio_mc_pass': portfolio_mc_pass,
        'pattern_details': {k: v for k, v in selected.items()},
        'tp_distribution': {
            'min': min(tps), 'median': round(float(np.median(tps)), 1),
            'mean': round(float(np.mean(tps)), 1), 'max': max(tps),
        } if tps else {},
        'sl_distribution': {
            'min': min(sls), 'median': round(float(np.median(sls)), 1),
            'mean': round(float(np.mean(sls)), 1), 'max': max(sls),
        } if sls else {},
        'correction_meta': correction_meta,
    }


def scan_patterns_mae_mfe(
    df: pd.DataFrame,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    mc_threshold: float = DEFAULT_MC_THRESHOLD,
    min_trades: int = DEFAULT_MIN_TRADES,
    signal_index: dict = None,
    concurrency: int = 0,
    max_baseline_wr: float = MAX_BASELINE_WR,
    correction_method: str = 'none',
    fdr_q: float = 0.05,
    require_portfolio_mc: bool = False,
    atr_ratio=None, clamp_lo=0.6, clamp_hi=1.7,
    tp_max=None, sl_max=None,
) -> dict:
    """MAE/MFE Discovery: derive TP/SL from excursion percentiles per pattern.

    Uses MFE percentile -> TP, MAE percentile -> SL instead of fixed grid search.
    Output format identical to scan_patterns_pp() for bot compatibility.
    """
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    if signal_index is None:
        signal_index = build_signal_index(types, n)
    logger.info(f"Built signal index: {len(signal_index)} unique patterns found")

    # Build work items (same structure as PP)
    work_items = []
    for pat_name in signal_index:
        for direction in ['LONG', 'SHORT']:
            sigs = signal_index[pat_name]
            work_items.append(
                (pat_name, direction, sigs, min_trades,
                 edge_threshold, mc_threshold, max_baseline_wr)
            )

    n_tested = sum(1 for item in work_items if len(item[2]) >= min_trades)
    logger.info(f"Pattern-direction combos to test: {n_tested} "
                f"(of {len(work_items)} total)")

    selected = {}
    pattern_details = []

    # Determine concurrency
    if concurrency == 0:
        n_workers = min(os.cpu_count() or 1, 8)
    else:
        n_workers = concurrency

    if n_workers > 1:
        logger.info(f"Parallel MAE/MFE search: {n_workers} workers")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pp_init,
            initargs=(opens, highs, lows, n, atr_ratio, clamp_lo, clamp_hi,
                      tp_max, sl_max),
        ) as executor:
            futures = {executor.submit(_mae_mfe_worker, item): item
                       for item in work_items}
            iterator = as_completed(futures)
            if _tqdm_cls:
                iterator = _tqdm_cls(iterator, total=len(futures),
                                     desc="MAE/MFE Search")
            for future in iterator:
                result = future.result()
                if result is not None:
                    key = result['key']
                    trades_raw = result.pop('trades_raw')
                    selected[key] = result
                    pattern_details.append({
                        'key': key, 'trades_raw': trades_raw,
                    })
    else:
        logger.info("Sequential MAE/MFE search")
        _shared_data['opens'] = opens
        _shared_data['highs'] = highs
        _shared_data['lows'] = lows
        _shared_data['n_bars'] = n
        _shared_data['atr_ratio'] = atr_ratio
        _shared_data['clamp_lo'] = clamp_lo
        _shared_data['clamp_hi'] = clamp_hi

        iterator = work_items
        if _tqdm_cls:
            iterator = _tqdm_cls(iterator, desc="MAE/MFE Search")
        for item in iterator:
            result = _mae_mfe_worker(item)
            if result is not None:
                key = result['key']
                trades_raw = result.pop('trades_raw')
                selected[key] = result
                pattern_details.append({
                    'key': key, 'trades_raw': trades_raw,
                })

    logger.info(f"Patterns passing filters: {len(selected)} "
                f"(tested: {n_tested})")

    # Apply multiple testing correction
    selected, correction_meta = apply_multiple_testing_correction(
        selected, n_tested, method=correction_method, fdr_q=fdr_q
    )

    # Rebuild pattern_details after correction
    surviving_keys = set(selected.keys())
    pattern_details = [pd_item for pd_item in pattern_details
                       if pd_item['key'] in surviving_keys]

    # Deduplicate dual-direction patterns: keep direction with better edge
    pat_by_name = {}
    for k, v in selected.items():
        pn = v['pattern']
        if pn not in pat_by_name or v['edge'] > pat_by_name[pn]['edge']:
            pat_by_name[pn] = v
        else:
            logger.info(f"Dedup: {pn} keeping {pat_by_name[pn]['direction']} "
                        f"(edge {pat_by_name[pn]['edge']}pp) over "
                        f"{v['direction']} (edge {v['edge']}pp)")
    deduped = {f"{v['pattern']}_{v['direction']}": v
               for v in pat_by_name.values()}
    n_removed = len(selected) - len(deduped)
    if n_removed:
        logger.info(f"Removed {n_removed} dual-direction duplicates")
        selected = deduped
        surviving_keys = set(selected.keys())
        pattern_details = [pd_item for pd_item in pattern_details
                           if pd_item['key'] in surviving_keys]

    # Portfolio 1-pos filter + stats
    all_trades = []
    for pd_item in pattern_details:
        all_trades.extend(pd_item['trades_raw'])
    portfolio_trades = portfolio_1pos(all_trades)
    portfolio_stats = calc_stats(portfolio_trades)
    portfolio_mc = (mc_test([t[2] for t in portfolio_trades])
                    if portfolio_trades else 1.0)

    portfolio_mc_pass = portfolio_mc < mc_threshold
    if require_portfolio_mc and not portfolio_mc_pass:
        logger.warning(f"Portfolio MC FAILED: p={portfolio_mc:.4f} "
                       f">= {mc_threshold}")

    # Organize by direction
    long_patterns = sorted([v['pattern'] for v in selected.values()
                            if v['direction'] == 'LONG'])
    short_patterns = sorted([v['pattern'] for v in selected.values()
                             if v['direction'] == 'SHORT'])

    # Build patterns_tpsl dict (same format as PP for bot compatibility)
    patterns_tpsl = {}
    for v in selected.values():
        patterns_tpsl[v['pattern']] = [v['tp'], v['sl']]

    # TP/SL distribution
    tps = [v['tp'] for v in selected.values()]
    sls = [v['sl'] for v in selected.values()]

    logger.info(f"Selected: {len(long_patterns)}L + {len(short_patterns)}S "
                f"= {len(selected)} patterns")
    logger.info(f"Portfolio: {portfolio_stats['trades']} trades, "
                f"WR {portfolio_stats['wr']}%, PnL {portfolio_stats['pnl']}%")

    return {
        'long_patterns': long_patterns,
        'short_patterns': short_patterns,
        'patterns_tpsl': patterns_tpsl,
        'portfolio_stats': portfolio_stats,
        'portfolio_mc': round(portfolio_mc, 4),
        'portfolio_mc_pass': portfolio_mc_pass,
        'pattern_details': {k: v for k, v in selected.items()},
        'tp_distribution': {
            'min': min(tps), 'median': round(float(np.median(tps)), 1),
            'mean': round(float(np.mean(tps)), 1), 'max': max(tps),
        } if tps else {},
        'sl_distribution': {
            'min': min(sls), 'median': round(float(np.median(sls)), 1),
            'mean': round(float(np.mean(sls)), 1), 'max': max(sls),
        } if sls else {},
        'correction_meta': correction_meta,
    }


def build_output_json(
    scan_result: dict,
    data_file: str,
    data_bars: int,
    discovery_method: str,
    edge_threshold: float,
    mc_threshold: float,
    min_trades: int,
    uni_tp: float = None,
    uni_sl: float = None,
    correction_meta: dict = None,
    wf_result: dict = None,
    timing: dict = None,
    is_days: int = 0,
    holdout_result: dict = None,
    holdout_days: int = 0,
    clean_mode: bool = False,
    atr_config: dict = None,
    neutral_meta: dict = None,
) -> dict:
    """Build the output JSON structure. Supports both universal and per_pattern modes."""
    output = {
        'version': '3.0' if clean_mode else '2.1',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'data_file': os.path.basename(data_file),
        'data_bars': data_bars,
        'selection_criteria': {
            'discovery_method': discovery_method,
            'edge_threshold_pp': edge_threshold,
            'mc_threshold': mc_threshold,
            'min_trades': min_trades,
            'max_baseline_wr': (MAX_BASELINE_WR
                                if discovery_method in ('per_pattern', 'mae_mfe')
                                else None),
            'is_days': is_days if is_days > 0 else None,
        },
        'patterns': {
            'long': scan_result['long_patterns'],
            'short': scan_result['short_patterns'],
        },
        'pattern_count': {
            'long': len(scan_result['long_patterns']),
            'short': len(scan_result['short_patterns']),
        },
        'backtest_summary': {
            'total_trades': scan_result['portfolio_stats']['trades'],
            'win_rate': scan_result['portfolio_stats']['wr'],
            'pnl_pct': scan_result['portfolio_stats']['pnl'],
            'max_drawdown_pct': scan_result['portfolio_stats']['mdd'],
            'profit_factor': scan_result['portfolio_stats']['pf'],
            'mc_pvalue': scan_result['portfolio_mc'],
            'portfolio_mc_pass': scan_result.get('portfolio_mc_pass', True),
        },
        'pattern_details': scan_result['pattern_details'],
    }

    # Correction metadata
    if correction_meta:
        output['selection_criteria'].update(correction_meta)
    elif 'correction_meta' in scan_result:
        output['selection_criteria'].update(scan_result['correction_meta'])

    # TP/SL mode specifics
    if discovery_method == 'universal':
        output['tp_sl_mode'] = 'universal'
        output['universal_tp'] = uni_tp
        output['universal_sl'] = uni_sl
    elif discovery_method == 'per_pattern':
        output['tp_sl_mode'] = 'per_pattern'
        output['patterns_tpsl'] = scan_result['patterns_tpsl']
        output['tp_distribution'] = scan_result['tp_distribution']
        output['sl_distribution'] = scan_result['sl_distribution']
    elif discovery_method == 'mae_mfe':
        output['tp_sl_mode'] = 'per_pattern'  # Bot-compatible output
        output['patterns_tpsl'] = scan_result['patterns_tpsl']
        output['tp_distribution'] = scan_result['tp_distribution']
        output['sl_distribution'] = scan_result['sl_distribution']
        output['selection_criteria']['tp_percentile_grid'] = TP_PERCENTILES
        output['selection_criteria']['sl_percentile_grid'] = SL_PERCENTILES

    # Walk-forward results
    if wf_result:
        output['walk_forward'] = wf_result

    # Holdout validation (v1.34.0)
    if holdout_result:
        output['holdout_validation'] = {
            'holdout_days': holdout_days,
            'results': holdout_result,
        }
        output['selection_criteria']['holdout_days'] = holdout_days

    # ATR config metadata
    if atr_config:
        output['atr_config'] = atr_config

    # Neutral window metadata (v1.36.3)
    if neutral_meta:
        output['neutral_window'] = neutral_meta

    # Timing
    if timing:
        output['timing'] = timing

    return output


def holdout_validate(df_holdout, scan_result, signal_index_holdout, n_holdout,
                     atr_ratio=None, clamp_lo=0.6, clamp_hi=1.7):
    """Validate selected patterns on holdout data (v1.34.0).

    For each selected pattern, backtest on holdout period and check
    WR Excess > 0 (genuine edge on unseen data).

    Returns:
        dict: pattern_key -> {status, wr, rw_wr, wr_excess, trades}
    """
    opens_h = df_holdout['open'].values
    highs_h = df_holdout['high'].values
    lows_h = df_holdout['low'].values

    # Merge long + short patterns with their direction and TP/SL
    patterns_tpsl = scan_result.get('patterns_tpsl', {})
    default_tp = scan_result.get('universal_tp', 2.0)
    default_sl = scan_result.get('universal_sl', 3.0)

    pattern_schedule = {}
    for pat_name in scan_result.get('long_patterns', []):
        pat_key = f"{pat_name}_LONG"
        tpsl = patterns_tpsl.get(pat_name)
        if isinstance(tpsl, (list, tuple)) and len(tpsl) >= 2:
            tp_pct, sl_pct = tpsl[0], tpsl[1]
        else:
            tp_pct, sl_pct = default_tp, default_sl
        pattern_schedule[pat_key] = {
            'direction': 'LONG', 'pattern': pat_name,
            'tp_pct': tp_pct, 'sl_pct': sl_pct,
        }
    for pat_name in scan_result.get('short_patterns', []):
        pat_key = f"{pat_name}_SHORT"
        tpsl = patterns_tpsl.get(pat_name)
        if isinstance(tpsl, (list, tuple)) and len(tpsl) >= 2:
            tp_pct, sl_pct = tpsl[0], tpsl[1]
        else:
            tp_pct, sl_pct = default_tp, default_sl
        pattern_schedule[pat_key] = {
            'direction': 'SHORT', 'pattern': pat_name,
            'tp_pct': tp_pct, 'sl_pct': sl_pct,
        }

    results = {}
    for pat_key, info in pattern_schedule.items():
        pat_name = info['pattern']
        direction = info['direction']
        tp_pct = info['tp_pct']
        sl_pct = info['sl_pct']

        # Get signal bars in holdout range
        signal_bars = signal_index_holdout.get(pat_name, [])
        if not signal_bars:
            results[pat_key] = {'status': 'SKIP', 'reason': 'no_signals', 'trades': 0}
            continue

        if atr_ratio is not None:
            trades = bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                                    opens_h, highs_h, lows_h, n_holdout,
                                    atr_ratio, clamp_lo, clamp_hi)
        else:
            trades = bt_signals(signal_bars, direction, tp_pct, sl_pct,
                                opens_h, highs_h, lows_h, n_holdout)
        n_trades = len(trades)
        if n_trades < 3:
            results[pat_key] = {'status': 'SKIP', 'reason': 'insufficient_trades',
                                'trades': n_trades}
            continue

        wins = sum(1 for _, _, pnl in trades if pnl > 0)
        wr = wins / n_trades
        rw_wr = sl_pct / (tp_pct + sl_pct)  # Random Walk WR
        wr_excess = wr - rw_wr

        results[pat_key] = {
            'status': 'PASS' if wr_excess > 0 else 'FAIL',
            'wr': round(wr * 100, 1),
            'rw_wr': round(rw_wr * 100, 1),
            'wr_excess': round(wr_excess * 100, 1),
            'trades': n_trades,
        }

    return results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pattern Scanner v2.3 — Dynamic Walk-Forward Pattern Selection (ATR + Neutral default)'
    )
    parser.add_argument('--data', default=DEFAULT_DATA_FILE,
                        help='Path to OHLCV CSV file')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_FILE,
                        help='Output JSON file path')
    parser.add_argument('--discovery-method', choices=['universal', 'per_pattern', 'mae_mfe'],
                        default='per_pattern',
                        help='Discovery method (default: per_pattern)')
    parser.add_argument('--tp', type=float, default=DEFAULT_UNI_TP,
                        help=f'Universal TP %% (default: {DEFAULT_UNI_TP})')
    parser.add_argument('--sl', type=float, default=DEFAULT_UNI_SL,
                        help=f'Universal SL %% (default: {DEFAULT_UNI_SL})')
    parser.add_argument('--edge-threshold', type=float, default=DEFAULT_EDGE_THRESHOLD,
                        help=f'Min edge in pp (default: {DEFAULT_EDGE_THRESHOLD})')
    parser.add_argument('--mc-threshold', type=float, default=DEFAULT_MC_THRESHOLD,
                        help=f'MC p-value cutoff (default: {DEFAULT_MC_THRESHOLD})')
    parser.add_argument('--min-trades', type=int, default=DEFAULT_MIN_TRADES,
                        help=f'Min trades required (default: {DEFAULT_MIN_TRADES})')
    # v2.1 new args
    parser.add_argument('--correction', choices=['none', 'bh', 'bonferroni'],
                        default='none',
                        help='Multiple testing correction (default: none)')
    parser.add_argument('--fdr-q', type=float, default=0.05,
                        help='FDR q-value for BH correction (default: 0.05)')
    parser.add_argument('--max-baseline-wr', type=float, default=MAX_BASELINE_WR,
                        help=f'Max baseline WR filter (default: {MAX_BASELINE_WR})')
    parser.add_argument('--require-portfolio-mc', action='store_true',
                        help='Log warning if portfolio MC test fails')
    parser.add_argument('--wf-folds', type=int, default=0,
                        help='Walk-forward folds (0=disabled, 3=typical)')
    parser.add_argument('--concurrency', type=int, default=0,
                        help='Parallel workers: 0=auto, 1=sequential, N=N workers')
    parser.add_argument('--is-days', type=int, default=0,
                        help='Limit IS training window to N most recent days '
                             '(0=use all data, e.g. 135 for Edge Decay optimal)')
    parser.add_argument('--holdout-days', type=int, default=7,
                        help='Reserve last N days as holdout OOS '
                             '(default: 7, 0=disabled)')
    # Neutral window args (v1.36.3: direction-balanced discovery)
    parser.add_argument('--neutral', action='store_true', default=True,
                        help='Auto-find price-neutral window (default: enabled)')
    parser.add_argument('--no-neutral', dest='neutral', action='store_false',
                        help='Disable neutral window (use full data or --is-days)')
    parser.add_argument('--neutral-tol', type=float, default=1.0,
                        help='Neutral window price tolerance %% (default: 1.0)')
    # ATR scaling args (default: enabled with production params)
    parser.add_argument('--no-atr', action='store_true', default=False,
                        help='Disable ATR scaling (Fixed TP/SL mode)')
    parser.add_argument('--atr-period', type=int, default=DEFAULT_ATR_PERIOD,
                        help=f'ATR calculation period (default: {DEFAULT_ATR_PERIOD})')
    parser.add_argument('--atr-window', type=int, default=DEFAULT_ATR_WINDOW,
                        help=f'ATR rolling median window (default: {DEFAULT_ATR_WINDOW})')
    parser.add_argument('--atr-clamp-lo', type=float, default=DEFAULT_ATR_CLAMP_LO,
                        help=f'ATR ratio min clamp (default: {DEFAULT_ATR_CLAMP_LO})')
    parser.add_argument('--atr-clamp-hi', type=float, default=DEFAULT_ATR_CLAMP_HI,
                        help=f'ATR ratio max clamp (default: {DEFAULT_ATR_CLAMP_HI})')
    parser.add_argument('--max-bars', type=int, default=MAX_BARS,
                        help=f'Max bars to hold trade (default: {MAX_BARS}, '
                             f'288=24h@5m, 96=24h@15m)')
    parser.add_argument('--tp-max', type=float, default=None,
                        help='Max TP %% (filter out wider TPs, e.g. 3.0 for 15m)')
    parser.add_argument('--sl-max', type=float, default=None,
                        help='Max SL %% (filter out wider SLs, e.g. 4.0 for 15m)')
    parser.add_argument('--clean', action='store_true',
                        help='Clean single-pass protocol: BH FDR as primary filter, '
                             'theory-derived thresholds, pre-registration manifest')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    # Clean mode: override parameters to theory-derived values
    CLEAN_MIN_EDGE_PP = 5.0   # Practical minimum (cost-based)
    CLEAN_MIN_SL_PCT = 1.0    # Execution risk floor
    if args.clean:
        args.edge_threshold = 0.0   # Let BH handle statistical significance
        args.mc_threshold = 1.0     # No per-pattern pre-filter
        args.correction = 'bh'      # Mandatory BH FDR
        if args.holdout_days == 0:
            args.holdout_days = 7   # Default holdout if not specified

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # Override module-level MAX_BARS if --max-bars specified
    _this = sys.modules[__name__]
    if args.max_bars != _this.MAX_BARS:
        logger.info(f"MAX_BARS override: {_this.MAX_BARS} → {args.max_bars}")
        _this.MAX_BARS = args.max_bars

    logger.info("=" * 60)
    if args.clean:
        logger.info("Pattern Scanner v3.0 — Clean Single-Pass Protocol")
    else:
        logger.info("Pattern Scanner v2.3 — Dynamic Walk-Forward Selection")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")

    # Clean mode: write pre-registration manifest BEFORE any data analysis
    if args.clean:
        manifest = {
            'protocol': 'clean_v3.0',
            'pre_registered_at': datetime.now().isoformat(timespec='seconds'),
            'rationale': 'Theory-derived thresholds, BH FDR primary filter, '
                         'single-pass execution',
            'parameters': {
                'discovery_method': args.discovery_method,
                'correction': 'bh',
                'fdr_q': args.fdr_q,
                'edge_threshold': 0.0,
                'mc_threshold': 1.0,
                'min_trades': args.min_trades,
                'max_baseline_wr': args.max_baseline_wr,
                'holdout_days': args.holdout_days,
                'wf_folds': args.wf_folds,
                'post_bh_min_edge_pp': CLEAN_MIN_EDGE_PP,
                'post_bh_min_sl_pct': CLEAN_MIN_SL_PCT,
            },
            'theory_basis': {
                'edge_5pp': 'Cost-based minimum: fees(0.30%) + slippage(0.06%) + '
                            'safety margin -> ~5pp WR excess needed for positive EV',
                'sl_1pct': 'Execution risk: SL < 1.0% cannot be reliably filled '
                           'given BTC spread + slippage',
                'bh_fdr': 'Controls false discovery rate at q=0.05 across all '
                          'tested hypotheses simultaneously',
            },
        }
        manifest_file = os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            'clean_scan_manifest.json')
        os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"Pre-registration manifest: {manifest_file}")
        logger.info(f"  BH FDR q={args.fdr_q}, post-BH: edge>={CLEAN_MIN_EDGE_PP}pp, "
                     f"SL>={CLEAN_MIN_SL_PCT}%")
    if args.is_days > 0:
        logger.info(f"IS Window: {args.is_days} days (most recent)")
    logger.info(f"Discovery: {args.discovery_method}")
    if args.discovery_method == 'universal':
        logger.info(f"TP: {args.tp}% | SL: {args.sl}% | "
                     f"Edge >= {args.edge_threshold}pp | MC < {args.mc_threshold}")
    elif args.discovery_method == 'mae_mfe':
        logger.info(f"MAE/MFE: TP pctiles {TP_PERCENTILES} | "
                     f"SL pctiles {SL_PERCENTILES} | "
                     f"Max baseline WR: {args.max_baseline_wr}%")
        logger.info(f"Edge >= {args.edge_threshold}pp | "
                     f"MC < {args.mc_threshold} (3-seed)")
    else:
        logger.info(f"Grid: TP {TP_GRID} | SL {SL_GRID} | "
                     f"Max baseline WR: {args.max_baseline_wr}%")
        logger.info(f"Edge >= {args.edge_threshold}pp | "
                     f"MC < {args.mc_threshold} (3-seed)")
    if args.correction != 'none':
        logger.info(f"Correction: {args.correction} (FDR q={args.fdr_q})")
    if args.wf_folds > 0:
        logger.info(f"Walk-Forward: {args.wf_folds} expanding folds")
    if args.holdout_days > 0:
        logger.info(f"Holdout: {args.holdout_days}d OOS validation")
    if not args.no_atr:
        logger.info(f"ATR scaling: period={args.atr_period}, window={args.atr_window}, "
                     f"clamp=[{args.atr_clamp_lo}, {args.atr_clamp_hi}]")
    else:
        logger.info("ATR scaling: DISABLED (--no-atr)")
    if args.neutral:
        logger.info(f"Neutral window: tol +/-{args.neutral_tol}% (auto-detect)")
    else:
        logger.info("Neutral window: disabled")
    if args.tp_max is not None or args.sl_max is not None:
        logger.info(f"TP/SL caps: TP max={args.tp_max}%, SL max={args.sl_max}%")
    if args.concurrency != 1:
        n_w = (min(os.cpu_count() or 1, 8)
               if args.concurrency == 0 else args.concurrency)
        logger.info(f"Concurrency: {n_w} workers")

    timing = {}

    # Load and classify
    t0 = time.time()
    df = load_and_classify(args.data)
    timing['classify_sec'] = round(time.time() - t0, 1)

    # Neutral window: find price-balanced window (v1.36.3)
    neutral_meta = None
    if args.neutral:
        closes_all = df['close'].values
        result_nw = find_neutral_window(closes_all, tol_pct=args.neutral_tol)
        if result_nw is not None:
            ns, ne = result_nw
            drift = (closes_all[ne] / closes_all[ns] - 1) * 100
            neutral_days = (ne - ns) / 288
            df = df.iloc[ns:ne + 1].reset_index(drop=True)
            neutral_meta = {
                'enabled': True,
                'tol_pct': args.neutral_tol,
                'window_days': round(neutral_days, 1),
                'start_bar': int(ns),
                'end_bar': int(ne),
                'start_price': round(float(closes_all[ns]), 2),
                'end_price': round(float(closes_all[ne]), 2),
                'drift_pct': round(drift, 2),
            }
            logger.info(f"Neutral window: {neutral_days:.0f}d (bars {ns}-{ne}), "
                        f"price {closes_all[ns]:.0f}->{closes_all[ne]:.0f} "
                        f"(drift {drift:+.2f}%, tol +/-{args.neutral_tol}%)")
        else:
            logger.warning(f"No neutral window found (tol +/-{args.neutral_tol}%, min 90d). "
                           f"Falling back to full data.")
            neutral_meta = {'enabled': False, 'reason': 'no_window_found'}
    else:
        neutral_meta = {'enabled': False}

    # Slice to IS window if --is-days specified (skip if neutral window is active)
    if args.is_days > 0 and not (args.neutral and neutral_meta and neutral_meta.get('enabled')):
        bars_per_day = 288  # 5m candles per day
        max_bars = args.is_days * bars_per_day
        if max_bars < len(df):
            original_len = len(df)
            df = df.iloc[-max_bars:].reset_index(drop=True)
            logger.info(f"IS window: {args.is_days}d -> "
                        f"kept last {len(df)} bars "
                        f"(trimmed {original_len - len(df)} from start)")
        else:
            logger.info(f"IS window: {args.is_days}d -> "
                        f"all {len(df)} bars used (data < requested window)")

    # Holdout split (v1.34.0): reserve last N days for OOS validation
    df_holdout = None
    if args.holdout_days > 0:
        holdout_bars = args.holdout_days * 288
        if holdout_bars < len(df) - 288:  # ensure IS has at least 1 day
            df_holdout = df.iloc[-holdout_bars:].reset_index(drop=True)
            df = df.iloc[:-holdout_bars].reset_index(drop=True)
            logger.info(f"Holdout split: {args.holdout_days}d ({holdout_bars} bars) reserved, "
                        f"IS: {len(df)} bars, Holdout: {len(df_holdout)} bars")
        else:
            logger.warning(f"Holdout {args.holdout_days}d too large for data ({len(df)} bars), disabled")

    # Build signal index (shared between scan and WF)
    types = df['candle_type'].tolist()
    n = len(types)
    signal_index = build_signal_index(types, n)

    # Compute ATR ratio for IS data (--no-atr disables)
    highs_arr = df['high'].values
    lows_arr = df['low'].values
    closes_arr = df['close'].values
    atr_ratio = None
    if not args.no_atr:
        atr_ratio = compute_atr_ratio(
            highs_arr, lows_arr, closes_arr,
            atr_period=args.atr_period, window=args.atr_window
        )
        valid_pct = np.sum(~np.isnan(atr_ratio)) / len(atr_ratio) * 100
        logger.info(f"ATR ratio computed: period={args.atr_period}, "
                     f"window={args.atr_window}, "
                     f"clamp=[{args.atr_clamp_lo}, {args.atr_clamp_hi}], "
                     f"valid={valid_pct:.1f}%")

    # Scan patterns
    t1 = time.time()
    if args.discovery_method == 'universal':
        result = scan_patterns(
            df,
            uni_tp=args.tp,
            uni_sl=args.sl,
            edge_threshold=args.edge_threshold,
            mc_threshold=args.mc_threshold,
            min_trades=args.min_trades,
            signal_index=signal_index,
            correction_method=args.correction,
            fdr_q=args.fdr_q,
            require_portfolio_mc=args.require_portfolio_mc,
        )
    elif args.discovery_method == 'mae_mfe':
        result = scan_patterns_mae_mfe(
            df,
            edge_threshold=args.edge_threshold,
            mc_threshold=args.mc_threshold,
            min_trades=args.min_trades,
            signal_index=signal_index,
            concurrency=args.concurrency,
            max_baseline_wr=args.max_baseline_wr,
            correction_method=args.correction,
            fdr_q=args.fdr_q,
            require_portfolio_mc=args.require_portfolio_mc,
            atr_ratio=atr_ratio, clamp_lo=args.atr_clamp_lo,
            clamp_hi=args.atr_clamp_hi,
            tp_max=args.tp_max, sl_max=args.sl_max,
        )
    else:
        result = scan_patterns_pp(
            df,
            edge_threshold=args.edge_threshold,
            mc_threshold=args.mc_threshold,
            min_trades=args.min_trades,
            signal_index=signal_index,
            concurrency=args.concurrency,
            max_baseline_wr=args.max_baseline_wr,
            correction_method=args.correction,
            fdr_q=args.fdr_q,
            require_portfolio_mc=args.require_portfolio_mc,
            atr_ratio=atr_ratio, clamp_lo=args.atr_clamp_lo,
            clamp_hi=args.atr_clamp_hi,
        )
    timing['scan_sec'] = round(time.time() - t1, 1)

    # Post-BH practical filters (clean mode only)
    if args.clean and result.get('pattern_details'):
        pre_count = len(result['pattern_details'])
        to_remove = []
        for key, info in result['pattern_details'].items():
            edge_val = info.get('edge', 0)
            sl_val = info.get('sl', 0)
            if edge_val < CLEAN_MIN_EDGE_PP:
                to_remove.append((key, f'edge={edge_val:.1f}pp < {CLEAN_MIN_EDGE_PP}pp'))
            elif sl_val < CLEAN_MIN_SL_PCT:
                to_remove.append((key, f'SL={sl_val:.1f}% < {CLEAN_MIN_SL_PCT}%'))

        for key, reason in to_remove:
            result['pattern_details'].pop(key, None)
            pat_name = key.rsplit('_', 1)[0]
            direction = key.rsplit('_', 1)[1]
            pat_list = (result['long_patterns'] if direction == 'LONG'
                        else result['short_patterns'])
            if pat_name in pat_list:
                pat_list.remove(pat_name)
            if 'patterns_tpsl' in result:
                result['patterns_tpsl'].pop(pat_name, None)
            logger.info(f"  Post-BH filter: {key} removed ({reason})")

        post_count = len(result['pattern_details'])
        logger.info(f"Post-BH practical filter: {pre_count} -> {post_count} patterns "
                    f"(edge >= {CLEAN_MIN_EDGE_PP}pp, SL >= {CLEAN_MIN_SL_PCT}%)")

    # Holdout validation (v1.34.0)
    holdout_result = None
    if df_holdout is not None:
        logger.info("")
        logger.info(f"Holdout validation ({args.holdout_days}d)...")
        types_h = df_holdout['candle_type'].tolist()
        n_h = len(types_h)
        signal_index_h = build_signal_index(types_h, n_h)
        # Compute ATR on holdout data (uses full data up to holdout for warm-up)
        holdout_atr = None
        if atr_ratio is not None:
            holdout_atr = compute_atr_ratio(
                df_holdout['high'].values, df_holdout['low'].values,
                df_holdout['close'].values,
                args.atr_period, window=args.atr_window)
        holdout_result = holdout_validate(
            df_holdout, result, signal_index_h, n_h,
            atr_ratio=holdout_atr, clamp_lo=args.atr_clamp_lo,
            clamp_hi=args.atr_clamp_hi)

        # Remove FAIL patterns from result
        fail_keys = [k for k, v in holdout_result.items() if v['status'] == 'FAIL']
        pass_count = sum(1 for v in holdout_result.values() if v['status'] == 'PASS')
        skip_count = sum(1 for v in holdout_result.values() if v['status'] == 'SKIP')

        for pat_key in fail_keys:
            pat_name = pat_key.rsplit('_', 1)[0]
            direction = pat_key.rsplit('_', 1)[1]
            logger.warning(f"Holdout FAIL: {pat_key} "
                           f"(WR={holdout_result[pat_key]['wr']}%, "
                           f"Excess={holdout_result[pat_key]['wr_excess']}pp)")
            pat_list = result['long_patterns'] if direction == 'LONG' else result['short_patterns']
            if pat_name in pat_list:
                pat_list.remove(pat_name)
            # Remove from pattern_details and patterns_tpsl
            if 'pattern_details' in result:
                result['pattern_details'].pop(pat_key, None)
            if 'patterns_tpsl' in result:
                result['patterns_tpsl'].pop(pat_key, None)

        logger.info(f"Holdout: {pass_count} PASS, {len(fail_keys)} FAIL, {skip_count} SKIP")
        if fail_keys:
            logger.info(f"Removed {len(fail_keys)} patterns, "
                        f"remaining: {len(result['long_patterns'])}L + "
                        f"{len(result['short_patterns'])}S")

    # Walk-forward validation (optional)
    wf_result = None
    if args.wf_folds > 0:
        logger.info("")
        logger.info(f"Walk-Forward validation ({args.wf_folds} folds)...")
        t2 = time.time()
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        wf_result = expanding_window_wf(
            signal_index, opens, highs, lows, n,
            n_folds=args.wf_folds,
            mode=args.discovery_method,
            uni_tp=args.tp if args.discovery_method == 'universal' else None,
            uni_sl=args.sl if args.discovery_method == 'universal' else None,
            min_trades=args.min_trades,
            edge_threshold=args.edge_threshold,
            mc_threshold=args.mc_threshold,
            max_baseline_wr=args.max_baseline_wr,
            closes=closes_arr if not args.no_atr else None,
            atr_period=args.atr_period, atr_window=args.atr_window,
            clamp_lo=args.atr_clamp_lo, clamp_hi=args.atr_clamp_hi,
            use_atr=not args.no_atr,
            tp_max=args.tp_max, sl_max=args.sl_max,
        )
        timing['wf_sec'] = round(time.time() - t2, 1)
        logger.info(f"WF Complete: {wf_result['positive_folds']}/{args.wf_folds} "
                     f"positive, OOS PnL {wf_result['total_oos_pnl']}%, "
                     f"{wf_result['stable_pattern_count']} stable patterns")

    timing['total_sec'] = round(time.time() - t0, 1)

    # Build output
    output = build_output_json(
        result,
        data_file=args.data,
        data_bars=len(df),
        discovery_method=args.discovery_method,
        edge_threshold=args.edge_threshold,
        mc_threshold=args.mc_threshold,
        min_trades=args.min_trades,
        uni_tp=args.tp if args.discovery_method == 'universal' else None,
        uni_sl=args.sl if args.discovery_method == 'universal' else None,
        correction_meta=result.get('correction_meta'),
        wf_result=wf_result,
        timing=timing,
        is_days=args.is_days,
        holdout_result=holdout_result,
        holdout_days=args.holdout_days,
        clean_mode=args.clean,
        atr_config={
            'enabled': not args.no_atr,
            'period': args.atr_period,
            'window': args.atr_window,
            'clamp_lo': args.atr_clamp_lo,
            'clamp_hi': args.atr_clamp_hi,
        } if not args.no_atr else {'enabled': False},
        neutral_meta=neutral_meta,
    )

    # Write JSON (numpy types → native Python for JSON compatibility)
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    logger.info(f"Output written to {args.output}")
    logger.info("=" * 60)

    # Print summary
    bs = output['backtest_summary']
    pc = output['pattern_count']
    print(f"\nScan Complete ({args.discovery_method}):")
    print(f"  Patterns: {pc['long']}L + {pc['short']}S = {pc['long'] + pc['short']}")
    print(f"  Trades: {bs['total_trades']} | WR: {bs['win_rate']}% "
          f"| PnL: {bs['pnl_pct']}%")
    print(f"  MDD: {bs['max_drawdown_pct']}% | PF: {bs['profit_factor']} "
          f"| MC p: {bs['mc_pvalue']}")
    if args.correction != 'none':
        sc = output['selection_criteria']
        print(f"  Correction: {args.correction} "
              f"({sc.get('n_before_correction', '?')} "
              f"-> {sc.get('n_after_correction', '?')})")
    if args.discovery_method in ('per_pattern', 'mae_mfe') and 'tp_distribution' in output:
        tp_d = output['tp_distribution']
        sl_d = output['sl_distribution']
        if tp_d:
            print(f"  TP: min={tp_d['min']} med={tp_d['median']} "
                  f"max={tp_d['max']}")
            print(f"  SL: min={sl_d['min']} med={sl_d['median']} "
                  f"max={sl_d['max']}")
    if not args.no_atr:
        print(f"  ATR: period={args.atr_period}, window={args.atr_window}, "
              f"clamp=[{args.atr_clamp_lo}, {args.atr_clamp_hi}]")
    else:
        print("  ATR: disabled")
    if neutral_meta and neutral_meta.get('enabled'):
        print(f"  Neutral: {neutral_meta['window_days']}d, "
              f"drift {neutral_meta['drift_pct']:+.2f}%")
    elif args.neutral:
        print("  Neutral: no window found (fallback to full data)")
    else:
        print("  Neutral: disabled")
    if wf_result:
        print(f"  Walk-Forward: {wf_result['positive_folds']}/"
              f"{wf_result['n_folds']} positive, "
              f"OOS PnL {wf_result['total_oos_pnl']}%, "
              f"{wf_result['stable_pattern_count']} stable patterns")
    wf_time = f", wf {timing['wf_sec']}s" if 'wf_sec' in timing else ""
    print(f"  Timing: classify {timing['classify_sec']}s, "
          f"scan {timing['scan_sec']}s{wf_time}, "
          f"total {timing['total_sec']}s")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
