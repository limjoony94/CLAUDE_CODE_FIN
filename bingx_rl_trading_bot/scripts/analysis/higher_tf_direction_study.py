#!/usr/bin/env python3
"""
Higher Timeframe Direction Study
================================
Research question: Does the 50/50 MFE/MAE symmetry observed on 5m break
on higher timeframes (15m, 1h, 4h)?

If higher TFs show directional edge (R:R > 1 for some patterns), this
could be used as a direction filter for 5m execution.

Phases:
  1. Resample 5m data → 15m, 1h, 4h; classify candles per TF
  2. MFE/MAE exhaustive analysis per TF (all 3-candle patterns x 2 directions)
  3. Symmetry test: binomial test on R:R > 1 proportion
  4. R:R=1 backtest on promising TFs (WR excess over 50%)
  5. R:R>1 backtest using MFE/MAE-derived TP/SL
  6. 5m direction filter strategy derivation

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE=3, FEE=FEE_PCT*LEVERAGE, slippage 0.02%
  - MC: seeds [42, 123, 7], max p < 0.01
  - Entry: next bar open, Exit: intrabar distance-based
  - Same-bar: abs(tp - opens[j]), NOT abs(tp - entry)
  - Expanding Window WF only
  - Timeout = DROP

Output: results/higher_tf_direction_study.json
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW, CandleType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('higher_tf_direction')

# ============================================================
# Constants — Standard Research Protocol
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10          # 0.05% x 2 sides
SLIPPAGE_BUFFER = 0.02  # 0.02%
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]
MIN_TRADES = 20         # Relaxed from 25 for higher TFs (fewer data points)

# Max holding bars by TF (24h equivalent)
MAX_BARS_BY_TF = {
    '5m':  288,   # 24h
    '15m': 96,    # 24h
    '1h':  24,    # 24h
    '4h':  6,     # 24h
}

# Also test 72h holding period
MAX_BARS_72H_BY_TF = {
    '5m':  864,   # 72h
    '15m': 288,   # 72h
    '1h':  72,    # 72h
    '4h':  18,    # 72h
}

# R:R=1 TP/SL levels to test per TF
RR1_LEVELS = {
    '15m': [1.0, 1.5, 2.0, 2.5, 3.0],
    '1h':  [1.5, 2.0, 3.0, 4.0, 5.0],
    '4h':  [2.0, 3.0, 4.0, 5.0, 7.0],
}

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'higher_tf_direction_study.json')


# ============================================================
# Data Loading & Resampling
# ============================================================

def load_5m_data():
    """Load 5m data CSV."""
    logger.info(f"Loading 5m data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} bars, "
                f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df


def resample_to_tf(df_5m, tf_str):
    """Resample 5m OHLCV to higher timeframe.

    Args:
        df_5m: DataFrame with timestamp, open, high, low, close, volume columns
        tf_str: pandas resample string ('15min', '1h', '4h')

    Returns:
        Resampled DataFrame with OHLCV columns
    """
    df = df_5m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.set_index('timestamp')

    # Use 'min' suffix for pandas resample compatibility
    resample_str = tf_str
    if tf_str == '15m':
        resample_str = '15min'

    df_tf = df.resample(resample_str).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    df_tf = df_tf.reset_index()
    logger.info(f"Resampled to {tf_str}: {len(df_tf)} bars")
    return df_tf


def classify_dataframe(df):
    """Classify candles using production classify_candle and build patterns.

    Args:
        df: DataFrame with open, high, low, close columns.

    Returns:
        DataFrame with candle_type, type_code, pattern_3 columns added.
    """
    df = df.copy()
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW).mean()

    types = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body']
        if pd.isna(avg_b):
            avg_b = 1.0
        ct = classify_candle(row, avg_b)
        types.append(ct.value)

    df['type_code'] = types

    # Build 3-candle patterns
    tc = df['type_code'].values
    patterns = [None, None] + [
        f"{tc[i-2]}-{tc[i-1]}-{tc[i]}"
        for i in range(2, len(tc))
    ]
    df['pattern_3'] = patterns

    # Distribution summary
    dist = Counter(types)
    logger.info(f"  Type distribution: {dict(sorted(dist.items(), key=lambda x: -x[1]))}")

    return df


# ============================================================
# MFE/MAE Analysis
# ============================================================

def compute_excursions(signal_bars, direction, opens, highs, lows, n_bars, max_bars):
    """Compute MFE and MAE for each trade (no TP/SL, natural path).

    Returns list of dicts with mfe_pct, mae_pct.
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
            if adverse > mae:
                mae = adverse

        mfe_pct = mfe / entry * 100
        mae_pct = mae / entry * 100

        excursions.append({
            'mfe_pct': mfe_pct,
            'mae_pct': mae_pct,
        })

    return excursions


def build_signal_index(types, n):
    """Build index: pattern_name -> list of signal bar indices."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def analyze_mfe_mae_for_tf(df_tf, tf_name, max_bars):
    """Exhaustive MFE/MAE analysis for all pattern-direction combos in a TF.

    Returns dict with summary stats per pattern-direction.
    """
    opens = df_tf['open'].values
    highs = df_tf['high'].values
    lows = df_tf['low'].values
    types = df_tf['type_code'].values
    n = len(df_tf)

    sig_idx = build_signal_index(types, n)

    results = {}
    for pat, bars in sig_idx.items():
        for direction in ['LONG', 'SHORT']:
            exc = compute_excursions(bars, direction, opens, highs, lows, n, max_bars)
            if len(exc) < 5:
                continue

            mfes = np.array([e['mfe_pct'] for e in exc])
            maes = np.array([e['mae_pct'] for e in exc])

            # Avoid division by zero: use MAE + epsilon
            med_mfe = float(np.median(mfes))
            med_mae = float(np.median(maes))
            rr = med_mfe / med_mae if med_mae > 0.001 else (999.0 if med_mfe > 0 else 1.0)

            # Fraction where MFE > MAE (direction accuracy)
            mfe_gt_mae = float(np.mean(mfes > maes))

            key = f"{pat}_{direction}"
            results[key] = {
                'pattern': pat,
                'direction': direction,
                'n_signals': len(exc),
                'median_mfe': round(med_mfe, 4),
                'median_mae': round(med_mae, 4),
                'rr_ratio': round(rr, 4),
                'pct_mfe_gt_mae': round(mfe_gt_mae * 100, 2),
                'mean_mfe': round(float(np.mean(mfes)), 4),
                'mean_mae': round(float(np.mean(maes)), 4),
            }

    return results


# ============================================================
# Backtest Engine (Standard Protocol)
# ============================================================

def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars, max_bars):
    """Backtest with TP/SL.

    Entry: next bar open after signal.
    Exit: intrabar high/low distance-based resolution.
    Timeout: DROP.
    Fee: FEE_PCT * LEVERAGE (BingX notional basis).
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

        if is_long:
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        for j in range(idx + 2, min(idx + 2 + max_bars, n_bars)):
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
                pnl = (tp_pct if dist_tp <= dist_sl else -sl_pct) * LEVERAGE - fee
                trades.append(pnl)
                break
            elif ht:
                trades.append(tp_pct * LEVERAGE - fee)
                break
            elif hs:
                trades.append(-sl_pct * LEVERAGE - fee)
                break
        # Timeout → DROP

    return trades


def mc_test(pnls):
    """Multi-seed sign randomization MC test. Returns max p-value (conservative)."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(MC_SIMS, len(pnls_arr)))
        rand_sums = signs @ pnls_arr
        p_vals.append(float(np.mean(rand_sums >= actual)))
    return max(p_vals)


def compute_compound_pnl(pnls):
    """Compound (multiplicative) returns from additive PnL list."""
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
    total_pnl = (equity - 1) * 100
    return total_pnl, mdd


def expanding_window_wf(signal_bars, direction, tp_pct, sl_pct,
                         opens, highs, lows, n_bars, max_bars, n_folds=3):
    """Expanding window walk-forward validation.

    Splits data into n_folds+1 equal segments.
    Fold k: IS = segments[0..k], OOS = segment[k+1].
    Returns list of fold results.
    """
    segment_size = n_bars // (n_folds + 1)
    if segment_size < 100:
        return []

    fold_results = []
    for fold in range(n_folds):
        is_end = segment_size * (fold + 1)
        oos_start = is_end
        oos_end = segment_size * (fold + 2)
        if fold == n_folds - 1:
            oos_end = n_bars  # Last fold gets remainder

        # OOS signal bars
        oos_bars = [b for b in signal_bars if oos_start <= b < oos_end]
        if len(oos_bars) < 3:
            fold_results.append({
                'fold': fold + 1,
                'oos_trades': 0,
                'oos_wr': 0.0,
                'oos_pnl': 0.0,
                'pass': False,
            })
            continue

        pnls = bt_signals(oos_bars, direction, tp_pct, sl_pct,
                          opens, highs, lows, n_bars, max_bars)

        if len(pnls) == 0:
            fold_results.append({
                'fold': fold + 1,
                'oos_trades': 0,
                'oos_wr': 0.0,
                'oos_pnl': 0.0,
                'pass': False,
            })
            continue

        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        total_pnl, mdd = compute_compound_pnl(pnls)

        fold_results.append({
            'fold': fold + 1,
            'oos_trades': len(pnls),
            'oos_wr': round(wr, 1),
            'oos_pnl': round(total_pnl, 2),
            'oos_mdd': round(mdd, 2),
            'pass': total_pnl > 0,
        })

    return fold_results


# ============================================================
# Phase 1: Load & Resample
# ============================================================

def phase1_resample(df_5m):
    """Resample 5m to 15m, 1h, 4h and classify candles."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Resampling & Classification")
    logger.info("=" * 60)

    tf_data = {}

    # 5m — use existing classification from the CSV
    df_5m_work = df_5m.copy()
    # Ensure type_code uses short codes (not full enum repr)
    if 'type_code' in df_5m_work.columns:
        tc = df_5m_work['type_code'].values
        # Check for full enum repr like "CandleType.BIG_DOWN"
        if len(tc) > 0 and isinstance(tc[2], str) and 'CandleType.' in str(tc[2]):
            logger.info("  5m: Re-classifying (full enum repr detected in type_code)")
            df_5m_work = classify_dataframe(df_5m_work)
        else:
            logger.info(f"  5m: Using existing classification ({len(df_5m_work)} bars)")
    else:
        logger.info("  5m: Classifying from scratch")
        df_5m_work = classify_dataframe(df_5m_work)

    # Rebuild pattern_3 if not present or type_code was fixed
    if 'pattern_3' not in df_5m_work.columns:
        tc = df_5m_work['type_code'].values
        patterns = [None, None] + [
            f"{tc[i-2]}-{tc[i-1]}-{tc[i]}"
            for i in range(2, len(tc))
        ]
        df_5m_work['pattern_3'] = patterns

    tf_data['5m'] = df_5m_work

    # Higher timeframes
    for tf_label, tf_resample in [('15m', '15m'), ('1h', '1h'), ('4h', '4h')]:
        logger.info(f"  Resampling to {tf_label}...")
        df_tf = resample_to_tf(df_5m, tf_resample)
        df_tf = classify_dataframe(df_tf)
        tf_data[tf_label] = df_tf
        logger.info(f"  {tf_label}: {len(df_tf)} bars, "
                     f"{df_tf['timestamp'].iloc[0]} to {df_tf['timestamp'].iloc[-1]}")

    return tf_data


# ============================================================
# Phase 2: MFE/MAE Exhaustive Analysis
# ============================================================

def phase2_mfe_mae(tf_data):
    """Analyze MFE/MAE for all pattern-direction combos across all TFs."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: MFE/MAE Exhaustive Analysis")
    logger.info("=" * 60)

    results = {}

    for tf_name, df_tf in tf_data.items():
        max_bars = MAX_BARS_BY_TF[tf_name]
        logger.info(f"\n  {tf_name} (max_bars={max_bars}, {len(df_tf)} bars)...")

        mfe_mae = analyze_mfe_mae_for_tf(df_tf, tf_name, max_bars)

        # Aggregate stats
        rr_ratios = [v['rr_ratio'] for v in mfe_mae.values() if v['rr_ratio'] < 100]
        rr_gt1 = [v for v in mfe_mae.values() if v['rr_ratio'] > 1.0 and v['rr_ratio'] < 100]
        total = len([v for v in mfe_mae.values() if v['rr_ratio'] < 100])

        rr_gt1_pct = len(rr_gt1) / total * 100 if total > 0 else 0
        mean_rr = float(np.mean(rr_ratios)) if rr_ratios else 0

        # Also compute with min_trades filter
        rr_ratios_filt = [v['rr_ratio'] for v in mfe_mae.values()
                          if v['rr_ratio'] < 100 and v['n_signals'] >= MIN_TRADES]
        rr_gt1_filt = [v for v in mfe_mae.values()
                       if v['rr_ratio'] > 1.0 and v['rr_ratio'] < 100
                       and v['n_signals'] >= MIN_TRADES]
        total_filt = len([v for v in mfe_mae.values()
                          if v['rr_ratio'] < 100 and v['n_signals'] >= MIN_TRADES])

        rr_gt1_pct_filt = len(rr_gt1_filt) / total_filt * 100 if total_filt > 0 else 0
        mean_rr_filt = float(np.mean(rr_ratios_filt)) if rr_ratios_filt else 0

        logger.info(f"    Total pattern-direction combos: {total}")
        logger.info(f"    R:R > 1.0: {len(rr_gt1)} ({rr_gt1_pct:.1f}%)")
        logger.info(f"    Mean R:R: {mean_rr:.4f}")
        logger.info(f"    Filtered (>={MIN_TRADES} trades): {total_filt} combos, "
                     f"R:R > 1.0: {len(rr_gt1_filt)} ({rr_gt1_pct_filt:.1f}%), "
                     f"Mean R:R: {mean_rr_filt:.4f}")

        # Top 10 by R:R
        top10 = sorted(mfe_mae.values(), key=lambda x: -x['rr_ratio'])[:10]
        logger.info(f"    Top 10 R:R patterns:")
        for t in top10:
            logger.info(f"      {t['pattern']}_{t['direction']}: R:R={t['rr_ratio']:.3f}, "
                         f"MFE={t['median_mfe']:.3f}%, MAE={t['median_mae']:.3f}%, "
                         f"n={t['n_signals']}, MFE>MAE={t['pct_mfe_gt_mae']:.1f}%")

        results[tf_name] = {
            'total_combos': total,
            'rr_gt1_count': len(rr_gt1),
            'rr_gt1_pct': round(rr_gt1_pct, 2),
            'mean_rr': round(mean_rr, 4),
            'median_rr': round(float(np.median(rr_ratios)), 4) if rr_ratios else 0,
            'std_rr': round(float(np.std(rr_ratios)), 4) if rr_ratios else 0,
            'filtered_total': total_filt,
            'filtered_rr_gt1_count': len(rr_gt1_filt),
            'filtered_rr_gt1_pct': round(rr_gt1_pct_filt, 2),
            'filtered_mean_rr': round(mean_rr_filt, 4),
            'per_pattern': {k: v for k, v in mfe_mae.items()},
            'top10': [{k: v for k, v in t.items()} for t in top10],
        }

    return results


# ============================================================
# Phase 3: Symmetry Test
# ============================================================

def phase3_symmetry_test(phase2_results):
    """Test if R:R > 1 proportion significantly deviates from 50%."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 3: Symmetry Test (Binomial, H0: p=0.5)")
    logger.info("=" * 60)

    results = {}

    for tf_name, tf_data in phase2_results.items():
        total = tf_data['total_combos']
        rr_gt1 = tf_data['rr_gt1_count']

        if total == 0:
            continue

        # Binomial test: H0: p(R:R > 1) = 0.5
        binom_p = stats.binomtest(rr_gt1, total, p=0.5).pvalue

        # Also test with filtered (min_trades)
        total_f = tf_data['filtered_total']
        rr_gt1_f = tf_data['filtered_rr_gt1_count']
        binom_p_f = stats.binomtest(rr_gt1_f, total_f, p=0.5).pvalue if total_f > 0 else 1.0

        # One-sample t-test on R:R ratios (H0: mean = 1.0)
        rr_vals = [v['rr_ratio'] for v in tf_data['per_pattern'].values()
                   if v['rr_ratio'] < 100]
        ttest_stat, ttest_p = stats.ttest_1samp(rr_vals, 1.0) if len(rr_vals) > 2 else (0, 1)

        is_symmetric = binom_p > 0.05 and binom_p_f > 0.05

        logger.info(f"\n  {tf_name}:")
        logger.info(f"    R:R > 1: {rr_gt1}/{total} = {rr_gt1/total*100:.1f}%")
        logger.info(f"    Binomial p-value: {binom_p:.6f} "
                     f"{'(SYMMETRIC)' if binom_p > 0.05 else '*** ASYMMETRIC ***'}")
        logger.info(f"    Filtered R:R > 1: {rr_gt1_f}/{total_f} = "
                     f"{rr_gt1_f/total_f*100:.1f}%" if total_f > 0 else "    N/A")
        logger.info(f"    Filtered binomial p: {binom_p_f:.6f}")
        logger.info(f"    Mean R:R: {tf_data['mean_rr']:.4f}, "
                     f"t-test p: {ttest_p:.6f}")
        logger.info(f"    Verdict: {'SYMMETRIC' if is_symmetric else 'ASYMMETRIC'}")

        results[tf_name] = {
            'total': total,
            'rr_gt1': rr_gt1,
            'rr_gt1_pct': round(rr_gt1 / total * 100, 2),
            'binomial_p': round(binom_p, 6),
            'filtered_total': total_f,
            'filtered_rr_gt1': rr_gt1_f,
            'filtered_rr_gt1_pct': round(rr_gt1_f / total_f * 100, 2) if total_f > 0 else 0,
            'filtered_binomial_p': round(binom_p_f, 6),
            'mean_rr': tf_data['mean_rr'],
            'ttest_stat': round(float(ttest_stat), 4),
            'ttest_p': round(float(ttest_p), 6),
            'symmetric': is_symmetric,
        }

    return results


# ============================================================
# Phase 4: R:R = 1 Backtest
# ============================================================

def phase4_rr1_backtest(tf_data, phase2_results):
    """Backtest with R:R = 1 (TP = SL) on each TF.

    At R:R = 1, baseline WR = 50%. Any WR > 50% represents genuine edge.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 4: R:R = 1 Backtest (TP = SL, baseline WR = 50%)")
    logger.info("=" * 60)

    results = {}

    for tf_name in ['15m', '1h', '4h']:
        df_tf = tf_data[tf_name]
        opens = df_tf['open'].values
        highs = df_tf['high'].values
        lows = df_tf['low'].values
        types = df_tf['type_code'].values
        n = len(df_tf)
        max_bars = MAX_BARS_BY_TF[tf_name]

        sig_idx = build_signal_index(types, n)
        levels = RR1_LEVELS[tf_name]

        logger.info(f"\n  {tf_name} ({n} bars, max_bars={max_bars}):")

        tf_results = {}

        for level in levels:
            tp_pct = level
            sl_pct = level

            edge_patterns = []
            total_tested = 0
            total_trades_all = 0

            for pat, bars in sig_idx.items():
                for direction in ['LONG', 'SHORT']:
                    total_tested += 1
                    pnls = bt_signals(bars, direction, tp_pct, sl_pct,
                                      opens, highs, lows, n, max_bars)

                    if len(pnls) < MIN_TRADES:
                        continue

                    total_trades_all += len(pnls)
                    wins = sum(1 for p in pnls if p > 0)
                    wr = wins / len(pnls) * 100

                    # Baseline WR at R:R=1 is 50%
                    wr_excess = wr - 50.0

                    if wr_excess >= 5.0:  # At least 5pp edge
                        mc_p = mc_test(pnls)
                        if mc_p < 0.01:
                            total_pnl, mdd = compute_compound_pnl(pnls)
                            edge_patterns.append({
                                'pattern': pat,
                                'direction': direction,
                                'trades': len(pnls),
                                'wr': round(wr, 1),
                                'wr_excess': round(wr_excess, 1),
                                'mc_p': round(mc_p, 4),
                                'pnl': round(total_pnl, 2),
                                'mdd': round(mdd, 2),
                            })

            logger.info(f"    TP=SL={level}%: {total_tested} tested, "
                         f"{len(edge_patterns)} with edge>=5pp & MC<0.01")

            if edge_patterns:
                for ep in sorted(edge_patterns, key=lambda x: -x['wr_excess'])[:5]:
                    logger.info(f"      {ep['pattern']}_{ep['direction']}: "
                                 f"WR={ep['wr']:.1f}%, excess={ep['wr_excess']:.1f}pp, "
                                 f"trades={ep['trades']}, MC={ep['mc_p']:.4f}, "
                                 f"PnL={ep['pnl']:.1f}%")

            tf_results[f"level_{level}"] = {
                'tp_sl_pct': level,
                'total_tested': total_tested,
                'total_trades': total_trades_all,
                'edge_patterns': len(edge_patterns),
                'patterns': edge_patterns,
            }

        results[tf_name] = tf_results

    return results


# ============================================================
# Phase 5: MFE/MAE-Based Backtest (R:R > 1)
# ============================================================

def phase5_mfe_mae_backtest(tf_data, phase2_results, phase3_results):
    """For TFs with asymmetry, backtest using MFE/MAE-derived TP/SL where TP > SL.

    Uses MFE percentiles for TP and MAE percentiles for SL.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 5: MFE/MAE-Based Backtest (TP > SL)")
    logger.info("=" * 60)

    TP_PTILES = [50, 60, 70]
    SL_PTILES = [70, 80, 90]

    results = {}

    for tf_name in ['15m', '1h', '4h']:
        df_tf = tf_data[tf_name]
        opens = df_tf['open'].values
        highs = df_tf['high'].values
        lows = df_tf['low'].values
        types = df_tf['type_code'].values
        n = len(df_tf)
        max_bars = MAX_BARS_BY_TF[tf_name]

        sig_idx = build_signal_index(types, n)

        logger.info(f"\n  {tf_name} ({n} bars, max_bars={max_bars}):")

        tf_edge_patterns = []

        for pat, bars in sig_idx.items():
            for direction in ['LONG', 'SHORT']:
                # Compute excursions
                exc = compute_excursions(bars, direction, opens, highs, lows, n, max_bars)
                if len(exc) < MIN_TRADES:
                    continue

                mfes = np.array([e['mfe_pct'] for e in exc])
                maes = np.array([e['mae_pct'] for e in exc])

                # Grid search over MFE/MAE percentile combinations
                best_score = -9999
                best_config = None

                for tp_p in TP_PTILES:
                    for sl_p in SL_PTILES:
                        tp_val = float(np.percentile(mfes, tp_p))
                        sl_val = float(np.percentile(maes, sl_p))

                        # Enforce TP > SL (R:R > 1) and SL >= 1.0%
                        if tp_val <= sl_val or sl_val < 1.0:
                            continue

                        pnls = bt_signals(bars, direction, tp_val, sl_val,
                                          opens, highs, lows, n, max_bars)

                        if len(pnls) < MIN_TRADES:
                            continue

                        wins = sum(1 for p in pnls if p > 0)
                        wr = wins / len(pnls) * 100
                        baseline_wr = sl_val / (tp_val + sl_val) * 100
                        wr_excess = wr - baseline_wr

                        total_pnl, mdd = compute_compound_pnl(pnls)
                        score = total_pnl / mdd if mdd > 0 else total_pnl

                        if score > best_score and wr_excess >= 5.0:
                            best_score = score
                            best_config = {
                                'tp_percentile': tp_p,
                                'sl_percentile': sl_p,
                                'tp_pct': round(tp_val, 2),
                                'sl_pct': round(sl_val, 2),
                                'rr': round(tp_val / sl_val, 3),
                                'trades': len(pnls),
                                'wr': round(wr, 1),
                                'baseline_wr': round(baseline_wr, 1),
                                'wr_excess': round(wr_excess, 1),
                                'pnl': round(total_pnl, 2),
                                'mdd': round(mdd, 2),
                                'pnl_mdd': round(score, 2),
                            }

                if best_config is not None:
                    # MC test on best config
                    pnls = bt_signals(bars, direction,
                                      best_config['tp_pct'], best_config['sl_pct'],
                                      opens, highs, lows, n, max_bars)
                    mc_p = mc_test(pnls)
                    best_config['mc_p'] = round(mc_p, 4)

                    if mc_p < 0.01:
                        best_config['pattern'] = pat
                        best_config['direction'] = direction

                        # WF validation
                        wf_results = expanding_window_wf(
                            bars, direction,
                            best_config['tp_pct'], best_config['sl_pct'],
                            opens, highs, lows, n, max_bars, n_folds=3
                        )
                        wf_pass = sum(1 for f in wf_results if f.get('pass', False))
                        best_config['wf_folds'] = wf_results
                        best_config['wf_pass'] = f"{wf_pass}/{len(wf_results)}"
                        best_config['wf_all_pass'] = (wf_pass == len(wf_results))

                        tf_edge_patterns.append(best_config)

        # Sort by PnL/MDD
        tf_edge_patterns.sort(key=lambda x: -x.get('pnl_mdd', 0))

        logger.info(f"    Found {len(tf_edge_patterns)} patterns with edge>=5pp, MC<0.01")

        wf_pass_count = sum(1 for p in tf_edge_patterns if p.get('wf_all_pass', False))
        logger.info(f"    WF 3/3 PASS: {wf_pass_count}/{len(tf_edge_patterns)}")

        for p in tf_edge_patterns[:10]:
            logger.info(f"      {p['pattern']}_{p['direction']}: "
                         f"TP={p['tp_pct']}%/SL={p['sl_pct']}% (R:R={p['rr']}), "
                         f"WR={p['wr']:.1f}% (excess={p['wr_excess']:.1f}pp), "
                         f"trades={p['trades']}, PnL={p['pnl']:.1f}%, "
                         f"MC={p['mc_p']:.4f}, WF={p['wf_pass']}")

        results[tf_name] = {
            'total_edge_patterns': len(tf_edge_patterns),
            'wf_all_pass': wf_pass_count,
            'patterns': tf_edge_patterns,
        }

    return results


# ============================================================
# Phase 6: 5m Direction Filter Strategy
# ============================================================

def phase6_direction_filter(tf_data, phase4_results, phase5_results):
    """If higher TF shows directional edge, test as 5m direction filter.

    Strategy: When higher TF pattern gives LONG signal, only take LONG entries
    on 5m for the next N bars (= 1 higher TF bar). Vice versa for SHORT.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 6: 5m Direction Filter Strategy")
    logger.info("=" * 60)

    # Find best TF with genuine edge
    best_tf = None
    best_count = 0

    for tf_name in ['1h', '4h', '15m']:
        p5 = phase5_results.get(tf_name, {})
        wf_count = p5.get('wf_all_pass', 0)
        if wf_count > best_count:
            best_count = wf_count
            best_tf = tf_name

    if best_count == 0:
        # Check Phase 4 as fallback
        for tf_name in ['1h', '4h', '15m']:
            p4 = phase4_results.get(tf_name, {})
            for level_key, level_data in p4.items():
                n_edge = level_data.get('edge_patterns', 0)
                if n_edge > best_count:
                    best_count = n_edge
                    best_tf = tf_name

    if best_tf is None or best_count == 0:
        logger.info("  No TF showed significant directional edge.")
        logger.info("  Direction filter NOT recommended.")
        return {
            'recommendation': 'NO_FILTER',
            'reason': 'No higher TF showed statistically significant directional edge',
            'best_tf': None,
        }

    logger.info(f"  Best TF: {best_tf} ({best_count} edge patterns)")

    # Build direction filter from higher TF patterns
    # For each higher-TF bar, determine allowed direction based on the
    # pattern that completed at that bar (if it's one of the edge patterns)

    # Collect edge patterns from Phase 5
    edge_patterns = {}
    p5 = phase5_results.get(best_tf, {})
    for p in p5.get('patterns', []):
        if p.get('wf_all_pass', False):
            key = p['pattern']
            edge_patterns[key] = p['direction']

    if not edge_patterns:
        # Fallback to Phase 4
        p4 = phase4_results.get(best_tf, {})
        for level_key, level_data in p4.items():
            for p in level_data.get('patterns', []):
                key = p['pattern']
                if key not in edge_patterns:
                    edge_patterns[key] = p['direction']

    logger.info(f"  Edge patterns from {best_tf}: {len(edge_patterns)}")

    if not edge_patterns:
        logger.info("  No actionable edge patterns found.")
        return {
            'recommendation': 'NO_FILTER',
            'reason': 'No WF-validated edge patterns found on any higher TF',
            'best_tf': best_tf,
        }

    # Map: higher TF bar timestamp → allowed direction
    df_htf = tf_data[best_tf]
    htf_patterns = df_htf['pattern_3'].values
    htf_timestamps = df_htf['timestamp'].values

    direction_map = {}  # timestamp → direction
    for i in range(len(htf_patterns)):
        pat = htf_patterns[i]
        if pat in edge_patterns:
            direction_map[htf_timestamps[i]] = edge_patterns[pat]

    logger.info(f"  Direction signals in {best_tf}: {len(direction_map)} "
                f"({len(direction_map)/len(df_htf)*100:.1f}% of bars)")

    # Now test on 5m: only allow entries matching the higher TF direction
    df_5m = tf_data['5m']
    opens_5m = df_5m['open'].values
    highs_5m = df_5m['high'].values
    lows_5m = df_5m['low'].values
    types_5m = df_5m['type_code'].values
    ts_5m = pd.to_datetime(df_5m['timestamp']).values
    n_5m = len(df_5m)
    max_bars_5m = MAX_BARS_BY_TF['5m']

    # Map each 5m bar to its higher TF bar
    bars_per_htf = {'15m': 3, '1h': 12, '4h': 48}
    bars_ratio = bars_per_htf[best_tf]

    # For each 5m pattern signal, check if its higher TF period has
    # a direction signal, and if so, whether it matches
    sig_idx_5m = build_signal_index(types_5m, n_5m)

    # Baseline: all 5m patterns (no filter) with R:R=1 TP=SL=2%
    baseline_trades_all = 0
    filtered_trades_all = 0
    baseline_pnls = []
    filtered_pnls = []

    tp_test = 2.0
    sl_test = 2.0

    for pat, bars in sig_idx_5m.items():
        for direction in ['LONG', 'SHORT']:
            pnls = bt_signals(bars, direction, tp_test, sl_test,
                              opens_5m, highs_5m, lows_5m, n_5m, max_bars_5m)
            baseline_pnls.extend(pnls)
            baseline_trades_all += len(pnls)

            # Filter: only take trades where higher TF direction matches
            filtered_bars = []
            for b in bars:
                # Find corresponding higher TF bar
                bar_ts = ts_5m[b]
                htf_bar_idx = b // bars_ratio
                if htf_bar_idx < len(htf_timestamps):
                    htf_ts = htf_timestamps[htf_bar_idx]
                    if htf_ts in direction_map and direction_map[htf_ts] == direction:
                        filtered_bars.append(b)

            if filtered_bars:
                f_pnls = bt_signals(filtered_bars, direction, tp_test, sl_test,
                                    opens_5m, highs_5m, lows_5m, n_5m, max_bars_5m)
                filtered_pnls.extend(f_pnls)
                filtered_trades_all += len(f_pnls)

    baseline_wr = (sum(1 for p in baseline_pnls if p > 0) / len(baseline_pnls) * 100
                   if baseline_pnls else 0)
    filtered_wr = (sum(1 for p in filtered_pnls if p > 0) / len(filtered_pnls) * 100
                   if filtered_pnls else 0)

    baseline_pnl_total, baseline_mdd = compute_compound_pnl(baseline_pnls) if baseline_pnls else (0, 0)
    filtered_pnl_total, filtered_mdd = compute_compound_pnl(filtered_pnls) if filtered_pnls else (0, 0)

    logger.info(f"\n  Direction Filter Test (5m, TP=SL={tp_test}%):")
    logger.info(f"    Baseline:  {baseline_trades_all} trades, WR={baseline_wr:.1f}%, "
                 f"PnL={baseline_pnl_total:.1f}%")
    logger.info(f"    Filtered:  {filtered_trades_all} trades, WR={filtered_wr:.1f}%, "
                 f"PnL={filtered_pnl_total:.1f}%")
    logger.info(f"    WR improvement: {filtered_wr - baseline_wr:.1f}pp")

    recommendation = 'INVESTIGATE' if filtered_wr - baseline_wr > 3.0 else 'NO_FILTER'
    logger.info(f"    Recommendation: {recommendation}")

    return {
        'best_tf': best_tf,
        'edge_patterns_count': len(edge_patterns),
        'edge_patterns': {k: v for k, v in list(edge_patterns.items())[:20]},
        'direction_signals': len(direction_map),
        'baseline': {
            'trades': baseline_trades_all,
            'wr': round(baseline_wr, 1),
            'pnl': round(baseline_pnl_total, 1),
            'mdd': round(baseline_mdd, 1),
        },
        'filtered': {
            'trades': filtered_trades_all,
            'wr': round(filtered_wr, 1),
            'pnl': round(filtered_pnl_total, 1),
            'mdd': round(filtered_mdd, 1),
        },
        'wr_improvement_pp': round(filtered_wr - baseline_wr, 1),
        'recommendation': recommendation,
    }


# ============================================================
# Main
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        return super().default(obj)


def main():
    start_time = time.time()

    logger.info("Higher Timeframe Direction Study")
    logger.info(f"Data: {DATA_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info(f"Protocol: LEVERAGE={LEVERAGE}, FEE={FEE_PCT}%*{LEVERAGE}="
                f"{FEE_PCT*LEVERAGE:.2f}%, MC seeds={MC_SEEDS}")
    logger.info(f"MIN_TRADES={MIN_TRADES}")
    logger.info("")

    # Phase 1: Load & Resample
    df_5m = load_5m_data()
    tf_data = phase1_resample(df_5m)

    # Phase 2: MFE/MAE Analysis
    phase2 = phase2_mfe_mae(tf_data)

    # Phase 3: Symmetry Test
    phase3 = phase3_symmetry_test(phase2)

    # Phase 4: R:R=1 Backtest
    phase4 = phase4_rr1_backtest(tf_data, phase2)

    # Phase 5: MFE/MAE-Based Backtest (R:R > 1)
    phase5 = phase5_mfe_mae_backtest(tf_data, phase2, phase3)

    # Phase 6: Direction Filter
    phase6 = phase6_direction_filter(tf_data, phase4, phase5)

    # Compile verdict
    symmetry_status = {}
    for tf in ['5m', '15m', '1h', '4h']:
        if tf in phase3:
            symmetry_status[tf] = {
                'symmetric': phase3[tf]['symmetric'],
                'rr_gt1_pct': phase3[tf]['rr_gt1_pct'],
                'binomial_p': phase3[tf]['binomial_p'],
                'mean_rr': phase3[tf]['mean_rr'],
            }

    # Find which TFs break symmetry
    asymmetric_tfs = [tf for tf, s in symmetry_status.items() if not s.get('symmetric', True)]

    # Count total WF-passing patterns across TFs
    total_wf_pass = 0
    best_tf_for_edge = None
    best_wf_count = 0
    for tf in ['15m', '1h', '4h']:
        p5 = phase5.get(tf, {})
        wf_count = p5.get('wf_all_pass', 0)
        total_wf_pass += wf_count
        if wf_count > best_wf_count:
            best_wf_count = wf_count
            best_tf_for_edge = tf

    verdict = {
        'symmetry_breaks_at': asymmetric_tfs if asymmetric_tfs else 'NONE',
        'all_tfs_symmetric': len(asymmetric_tfs) == 0,
        'best_tf': best_tf_for_edge,
        'total_wf_pass_patterns': total_wf_pass,
        'direction_filter_recommendation': phase6.get('recommendation', 'NO_FILTER'),
        'conclusion': '',
    }

    if len(asymmetric_tfs) == 0 and total_wf_pass == 0:
        verdict['conclusion'] = (
            "All timeframes show symmetric MFE/MAE (R:R ~ 1.0). "
            "3-candle patterns do not provide directional edge on any TF tested. "
            "Higher TF direction filter NOT recommended."
        )
    elif total_wf_pass > 0:
        verdict['conclusion'] = (
            f"Found {total_wf_pass} WF-validated edge patterns on {best_tf_for_edge}. "
            f"Further investigation warranted but must survive OOS before deployment."
        )
    else:
        verdict['conclusion'] = (
            f"Symmetry breaks detected on {asymmetric_tfs} but no patterns survived "
            f"WF validation. Likely noise, not genuine edge."
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("VERDICT")
    logger.info("=" * 60)
    logger.info(f"  Symmetry breaks at: {verdict['symmetry_breaks_at']}")
    logger.info(f"  Best TF for edge: {verdict['best_tf']}")
    logger.info(f"  Total WF-pass patterns: {verdict['total_wf_pass_patterns']}")
    logger.info(f"  Direction filter: {verdict['direction_filter_recommendation']}")
    logger.info(f"  Conclusion: {verdict['conclusion']}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    # Save results (minimize per_pattern data to keep JSON manageable)
    # Trim per_pattern from Phase 2 to reduce file size
    phase2_compact = {}
    for tf, data in phase2.items():
        phase2_compact[tf] = {
            k: v for k, v in data.items() if k != 'per_pattern'
        }

    output = {
        'study': 'higher_tf_direction',
        'timestamp': datetime.now().isoformat(),
        'data_file': DATA_FILE,
        'protocol': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'fee_effective': FEE_PCT * LEVERAGE,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'mc_seeds': MC_SEEDS,
            'mc_sims': MC_SIMS,
            'min_trades': MIN_TRADES,
            'max_bars_24h': MAX_BARS_BY_TF,
        },
        'data_summary': {
            tf: {'n_bars': len(df), 'n_days': round(len(df) * {'5m': 5, '15m': 15, '1h': 60, '4h': 240}[tf] / 1440, 1)}
            for tf, df in tf_data.items()
        },
        'phase2_mfe_mae_summary': phase2_compact,
        'phase3_symmetry_test': phase3,
        'phase4_rr1_backtest': phase4,
        'phase5_mfe_mae_backtest': phase5,
        'phase6_direction_filter': phase6,
        'verdict': verdict,
        'elapsed_seconds': round(elapsed, 1),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
