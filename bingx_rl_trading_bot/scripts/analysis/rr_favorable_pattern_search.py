#!/usr/bin/env python3
"""
R:R Favorable Pattern Search
==============================
Search for 3-candle patterns where MFE > MAE (natural R:R > 1).

Previous research showed that forcing R:R > 1 via fixed grid kills all patterns.
This study takes the opposite approach: find patterns where price naturally moves
MORE in the predicted direction than against it, then derive TP/SL from that
natural behavior.

Phase 1: MFE/MAE census of all pattern-direction combos (12^3 x 2 = 3,456)
Phase 2: Filter to R:R > 1 patterns (median MFE > median MAE)
Phase 3: Derive TP/SL from MFE/MAE percentiles, backtest with R:R > 1 constraint
Phase 4: Expanding Window WF validation (3-fold)
Phase 5: Random baseline comparison
Phase 6: Extended search (2-candle, 4-candle) if 3-candle yields few results

Standard Research Protocol:
- Production classify_candle import
- LEVERAGE = 3, FEE = 0.10% x LEVERAGE (notional)
- Slippage buffer 0.02%
- Entry: signal next-bar open
- MAX_BARS = 288 (24h timeout -> DROP)
- ATR scaling: ATR(14)/median(576), clamp [0.6, 1.7]
- MC: 3 seeds (42, 123, 7), max p < 0.01
- Same-bar resolution: abs(tp - bar_open) vs abs(sl - bar_open)
- Expanding Window WF only (never cross-validation)
- Neutral window (start ~ end price, +/- 1% tolerance)
- min_trades >= 25
- SL >= 1.0%
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    bt_signals, bt_signals_atr, compute_atr_ratio, compute_excursions,
    derive_tp_sl, mc_test, portfolio_1pos, calc_stats,
    FEE_PCT, LEVERAGE, MAX_BARS, MC_SEEDS, MC_SIMS,
    DEFAULT_MIN_TRADES, MAX_BASELINE_WR, SLIPPAGE_BUFFER,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# --- Paths ---
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
RESULTS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'rr_favorable_pattern_search.json')

# --- Parameters ---
EDGE_THRESHOLD = 0.0       # For R:R > 1, even small positive edge is interesting
WR_EXCESS_MIN = 5.0        # WR - baseline WR >= 5pp (genuine edge)
MC_THRESHOLD = 0.01
MIN_TRADES = 25
SL_MIN = 1.0               # Execution risk floor
N_FOLDS = 3
BARS_PER_DAY = 288
NEUTRAL_TOL = 1.0           # Neutral window tolerance %

# MFE/MAE percentile configs for R:R > 1 TP/SL derivation
# We need TP > SL, so we use lower MFE percentiles (conservative TP)
# and higher MAE percentiles (generous SL)
RR_CONFIGS = [
    {'name': 'conservative',  'tp_p': 25, 'sl_p': 50},
    {'name': 'balanced_25_75', 'tp_p': 25, 'sl_p': 75},
    {'name': 'balanced_40_60', 'tp_p': 40, 'sl_p': 60},
    {'name': 'balanced_50_50', 'tp_p': 50, 'sl_p': 50},
    {'name': 'balanced_50_75', 'tp_p': 50, 'sl_p': 75},
    {'name': 'aggressive',    'tp_p': 60, 'sl_p': 50},
    {'name': 'max_rr',        'tp_p': 25, 'sl_p': 75},
    {'name': 'tight',         'tp_p': 30, 'sl_p': 40},
]


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
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


def build_signal_index_ncand(types, n, n_candles=3):
    """Build signal index for variable-length candle patterns.

    Args:
        types: Array of candle type strings.
        n: Length of types array.
        n_candles: Number of candles in pattern (2, 3, or 4).

    Returns:
        dict: pattern_name -> list of signal bar indices
    """
    idx = {}
    for i in range(n_candles - 1, n):
        parts = [types[i - (n_candles - 1 - k)] for k in range(n_candles)]
        pat = '-'.join(parts)
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def compute_excursions_for_signals(signal_bars, direction, opens, highs, lows,
                                   n_bars, max_bars=MAX_BARS):
    """Compute MFE and MAE for each signal (no TP/SL, natural path).

    Reuses scanner logic but returns raw arrays for statistical analysis.
    """
    mfes = []
    maes = []
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
        mfes.append(mfe_pct)
        maes.append(mae_pct)

    return np.array(mfes), np.array(maes)


def backtest_with_atr(signal_bars, direction, tp_pct, sl_pct,
                      opens, highs, lows, n_bars, atr_ratio,
                      clamp_lo=DEFAULT_ATR_CLAMP_LO,
                      clamp_hi=DEFAULT_ATR_CLAMP_HI):
    """Backtest with ATR scaling. Wrapper for bt_signals_atr."""
    if atr_ratio is not None:
        return bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                              opens, highs, lows, n_bars,
                              atr_ratio, clamp_lo, clamp_hi)
    else:
        return bt_signals(signal_bars, direction, tp_pct, sl_pct,
                          opens, highs, lows, n_bars)


def phase1_mfe_mae_census(signal_index, opens, highs, lows, n_bars,
                          bar_start, bar_end):
    """Phase 1: Compute MFE/MAE statistics for ALL pattern-direction combos.

    Returns:
        list of dicts with pattern stats, sorted by mfe_mae_ratio descending.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: MFE/MAE Census of All Pattern-Direction Combos")
    logger.info("=" * 70)

    results = []
    trade_boundary = bar_end if bar_end is not None else n_bars
    total_combos = 0
    with_min_trades = 0

    for pat_name, all_sigs in signal_index.items():
        sigs = [s for s in all_sigs if bar_start <= s < bar_end]
        for direction in ['LONG', 'SHORT']:
            total_combos += 1
            if len(sigs) < MIN_TRADES:
                continue
            with_min_trades += 1

            mfes, maes = compute_excursions_for_signals(
                sigs, direction, opens, highs, lows, trade_boundary)

            if len(mfes) < MIN_TRADES:
                continue

            median_mfe = float(np.median(mfes))
            median_mae = float(np.median(maes))
            ratio = median_mfe / median_mae if median_mae > 0 else 999.0
            pct_mfe_gt_mae = float(np.mean(mfes > maes) * 100)

            results.append({
                'pattern': pat_name,
                'direction': direction,
                'n_signals': len(mfes),
                'median_mfe': round(median_mfe, 4),
                'p25_mfe': round(float(np.percentile(mfes, 25)), 4),
                'p50_mfe': round(float(np.percentile(mfes, 50)), 4),
                'p75_mfe': round(float(np.percentile(mfes, 75)), 4),
                'mean_mfe': round(float(np.mean(mfes)), 4),
                'median_mae': round(median_mae, 4),
                'p25_mae': round(float(np.percentile(maes, 25)), 4),
                'p50_mae': round(float(np.percentile(maes, 50)), 4),
                'p75_mae': round(float(np.percentile(maes, 75)), 4),
                'mean_mae': round(float(np.mean(maes)), 4),
                'mfe_mae_ratio': round(ratio, 4),
                'pct_mfe_gt_mae': round(pct_mfe_gt_mae, 1),
            })

    # Sort by ratio descending
    results.sort(key=lambda x: x['mfe_mae_ratio'], reverse=True)

    n_rr_gt1 = sum(1 for r in results if r['mfe_mae_ratio'] > 1.0)
    ratios = [r['mfe_mae_ratio'] for r in results]

    logger.info(f"Total combos tested: {total_combos}")
    logger.info(f"With >= {MIN_TRADES} trades: {with_min_trades}")
    logger.info(f"With valid excursions: {len(results)}")
    logger.info(f"R:R > 1.0 (median MFE > median MAE): {n_rr_gt1} "
                f"({n_rr_gt1/len(results)*100:.1f}%)" if results else "")
    if ratios:
        logger.info(f"MFE/MAE ratio: mean={np.mean(ratios):.3f}, "
                     f"median={np.median(ratios):.3f}, "
                     f"p25={np.percentile(ratios, 25):.3f}, "
                     f"p75={np.percentile(ratios, 75):.3f}")

    # Top 20 by ratio
    logger.info("\nTop 20 patterns by MFE/MAE ratio:")
    for i, r in enumerate(results[:20]):
        logger.info(f"  {i+1:3d}. {r['pattern']:20s} {r['direction']:5s} "
                     f"R:R={r['mfe_mae_ratio']:.3f}  "
                     f"MFE={r['median_mfe']:.3f}%  MAE={r['median_mae']:.3f}%  "
                     f"MFE>MAE={r['pct_mfe_gt_mae']:.0f}%  "
                     f"n={r['n_signals']}")

    summary = {
        'total_combos': total_combos,
        'with_min_trades': with_min_trades,
        'with_valid_excursions': len(results),
        'n_rr_gt1': n_rr_gt1,
        'pct_rr_gt1': round(n_rr_gt1 / len(results) * 100, 1) if results else 0,
        'ratio_mean': round(float(np.mean(ratios)), 4) if ratios else 0,
        'ratio_median': round(float(np.median(ratios)), 4) if ratios else 0,
        'ratio_p25': round(float(np.percentile(ratios, 25)), 4) if ratios else 0,
        'ratio_p75': round(float(np.percentile(ratios, 75)), 4) if ratios else 0,
        'ratio_max': round(float(np.max(ratios)), 4) if ratios else 0,
    }

    return results, summary


def phase2_filter_rr_gt1(phase1_results):
    """Phase 2: Filter to patterns with natural R:R > 1.

    Returns filtered list and summary statistics.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: Filter R:R > 1 Patterns")
    logger.info("=" * 70)

    rr_gt1 = [r for r in phase1_results if r['mfe_mae_ratio'] > 1.0]

    n_long = sum(1 for r in rr_gt1 if r['direction'] == 'LONG')
    n_short = sum(1 for r in rr_gt1 if r['direction'] == 'SHORT')

    logger.info(f"R:R > 1 patterns: {len(rr_gt1)} ({n_long}L + {n_short}S)")

    if rr_gt1:
        ratios = [r['mfe_mae_ratio'] for r in rr_gt1]
        pct_vals = [r['pct_mfe_gt_mae'] for r in rr_gt1]
        logger.info(f"  MFE/MAE ratio: mean={np.mean(ratios):.3f}, "
                     f"range=[{np.min(ratios):.3f}, {np.max(ratios):.3f}]")
        logger.info(f"  Pct MFE>MAE per trade: mean={np.mean(pct_vals):.1f}%, "
                     f"range=[{np.min(pct_vals):.1f}%, {np.max(pct_vals):.1f}%]")

        # Also filter by pct_mfe_gt_mae > 50%
        # (majority of individual trades have MFE > MAE)
        rr_strong = [r for r in rr_gt1 if r['pct_mfe_gt_mae'] > 50]
        logger.info(f"\n  Strong (>50% of trades have MFE>MAE): {len(rr_strong)}")

    # Distribution analysis
    all_ratios = [r['mfe_mae_ratio'] for r in phase1_results]
    bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 999]
    hist, _ = np.histogram(all_ratios, bins=bins)
    logger.info("\nMFE/MAE ratio distribution:")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        label = f"  [{lo:.1f}, {hi:.1f})" if hi < 999 else f"  [{lo:.1f}, inf)"
        logger.info(f"{label}: {hist[i]}")

    summary = {
        'total_rr_gt1': len(rr_gt1),
        'long_count': n_long,
        'short_count': n_short,
        'strong_count': len([r for r in rr_gt1 if r['pct_mfe_gt_mae'] > 50]),
        'ratio_distribution': {
            f'{bins[i]:.1f}-{bins[i+1]:.1f}': int(hist[i])
            for i in range(len(bins) - 1)
        },
    }

    return rr_gt1, summary


def phase3_backtest_rr_gt1(rr_gt1_patterns, signal_index, opens, highs, lows,
                            n_bars, atr_ratio, bar_start, bar_end):
    """Phase 3: Derive TP/SL from MFE/MAE percentiles with R:R > 1 constraint.

    For each R:R > 1 pattern, test multiple percentile configs.
    Only keep configs where TP > SL (R:R > 1 in TP/SL space too).
    Apply full quality filters: MC, WR Excess, SL >= 1.0%.
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: Backtest R:R > 1 Patterns with MFE/MAE-Derived TP/SL")
    logger.info("=" * 70)

    trade_boundary = bar_end if bar_end is not None else n_bars
    config_results = {}

    for cfg in RR_CONFIGS:
        cfg_name = cfg['name']
        tp_p, sl_p = cfg['tp_p'], cfg['sl_p']
        passing = []
        tested = 0
        skipped_rr = 0
        skipped_sl = 0
        skipped_trades = 0
        skipped_wr_excess = 0
        skipped_mc = 0

        for pat_info in rr_gt1_patterns:
            pat_name = pat_info['pattern']
            direction = pat_info['direction']

            sigs = [s for s in signal_index[pat_name] if bar_start <= s < bar_end]
            if len(sigs) < MIN_TRADES:
                continue
            tested += 1

            # Compute excursions for this pattern-direction
            excursions = compute_excursions(
                sigs, direction, opens, highs, lows, trade_boundary, MAX_BARS)
            if len(excursions) < MIN_TRADES:
                skipped_trades += 1
                continue

            # Derive TP/SL from percentiles
            tp, sl = derive_tp_sl(excursions, tp_p, sl_p)
            if tp is None or sl is None:
                continue

            # R:R > 1 constraint: TP must be > SL
            if tp <= sl:
                skipped_rr += 1
                continue

            # SL floor
            if sl < SL_MIN:
                skipped_sl += 1
                continue

            # Baseline WR for this R:R
            baseline_wr = sl / (tp + sl) * 100

            # Backtest with ATR scaling
            trades = backtest_with_atr(
                sigs, direction, tp, sl,
                opens, highs, lows, trade_boundary,
                atr_ratio)

            if len(trades) < MIN_TRADES:
                skipped_trades += 1
                continue

            pnls = [t[2] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100
            wr_excess = wr - baseline_wr

            if wr_excess < WR_EXCESS_MIN:
                skipped_wr_excess += 1
                continue

            # MC test
            mc_p = mc_test(pnls)
            if mc_p >= MC_THRESHOLD:
                skipped_mc += 1
                continue

            # Compute PnL stats
            cum = 0
            peak = 0
            mdd_val = 0
            for p in pnls:
                cum += p
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > mdd_val:
                    mdd_val = dd

            rr_actual = tp / sl

            passing.append({
                'pattern': pat_name,
                'direction': direction,
                'tp': tp,
                'sl': sl,
                'rr': round(rr_actual, 3),
                'trades': len(trades),
                'wr': round(wr, 1),
                'baseline_wr': round(baseline_wr, 1),
                'wr_excess': round(wr_excess, 1),
                'pnl': round(cum, 1),
                'mdd': round(mdd_val, 1),
                'mc_p': round(mc_p, 4),
                'tp_percentile': tp_p,
                'sl_percentile': sl_p,
                'mfe_mae_ratio': pat_info['mfe_mae_ratio'],
                'pct_mfe_gt_mae': pat_info['pct_mfe_gt_mae'],
                'n_signals': pat_info['n_signals'],
                'trades_raw': trades,
            })

        logger.info(f"\nConfig '{cfg_name}' (TP=p{tp_p}_MFE, SL=p{sl_p}_MAE):")
        logger.info(f"  Tested: {tested}, Passing: {len(passing)}")
        logger.info(f"  Skipped: RR<=1={skipped_rr}, SL<1%={skipped_sl}, "
                     f"trades<{MIN_TRADES}={skipped_trades}, "
                     f"WR_excess<{WR_EXCESS_MIN}pp={skipped_wr_excess}, "
                     f"MC>={MC_THRESHOLD}={skipped_mc}")

        if passing:
            wrs = [p['wr'] for p in passing]
            rrs = [p['rr'] for p in passing]
            edges = [p['wr_excess'] for p in passing]
            logger.info(f"  WR: mean={np.mean(wrs):.1f}%, "
                         f"range=[{np.min(wrs):.1f}%, {np.max(wrs):.1f}%]")
            logger.info(f"  R:R: mean={np.mean(rrs):.2f}, "
                         f"range=[{np.min(rrs):.2f}, {np.max(rrs):.2f}]")
            logger.info(f"  WR Excess: mean={np.mean(edges):.1f}pp")
            for p in passing:
                logger.info(f"    {p['pattern']:20s} {p['direction']:5s} "
                             f"TP={p['tp']:.2f}% SL={p['sl']:.2f}% "
                             f"R:R={p['rr']:.2f} WR={p['wr']:.1f}% "
                             f"WRx={p['wr_excess']:.1f}pp "
                             f"PnL={p['pnl']:.1f}% MC={p['mc_p']:.4f}")

        config_results[cfg_name] = {
            'tp_percentile': tp_p,
            'sl_percentile': sl_p,
            'tested': tested,
            'passing': len(passing),
            'skipped_rr_le1': skipped_rr,
            'skipped_sl_low': skipped_sl,
            'skipped_trades': skipped_trades,
            'skipped_wr_excess': skipped_wr_excess,
            'skipped_mc': skipped_mc,
            'patterns': [{k: v for k, v in p.items() if k != 'trades_raw'}
                         for p in passing],
            'patterns_with_trades': passing,
        }

    # Find best config by total passing patterns, then by total PnL
    best_cfg = None
    best_count = 0
    best_pnl = -9999
    for cfg_name, cr in config_results.items():
        if cr['passing'] > best_count or (
                cr['passing'] == best_count and
                sum(p['pnl'] for p in cr['patterns']) > best_pnl):
            best_count = cr['passing']
            best_pnl = sum(p['pnl'] for p in cr['patterns'])
            best_cfg = cfg_name

    logger.info(f"\nBest config: '{best_cfg}' with {best_count} patterns")

    # Also create a MERGED set: best TP/SL per pattern across all configs
    merged_best = {}
    for cfg_name, cr in config_results.items():
        for p in cr['patterns_with_trades']:
            key = f"{p['pattern']}_{p['direction']}"
            if key not in merged_best or p['wr_excess'] > merged_best[key]['wr_excess']:
                merged_best[key] = p

    merged_list = sorted(merged_best.values(),
                         key=lambda x: x['wr_excess'], reverse=True)
    logger.info(f"Merged best (unique patterns, best config each): {len(merged_list)}")

    summary = {
        'config_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'patterns_with_trades'}
            for k, v in config_results.items()
        },
        'best_config': best_cfg,
        'best_config_passing': best_count,
        'merged_unique_patterns': len(merged_list),
        'merged_patterns': [{k: v for k, v in p.items() if k != 'trades_raw'}
                            for p in merged_list],
    }

    return config_results, merged_list, summary


def phase4_wf_validation(signal_index, opens, highs, lows, closes, n_bars,
                          merged_patterns, n_folds=N_FOLDS, bar_start=0,
                          bar_end=None):
    """Phase 4: Expanding Window WF validation.

    For each fold, re-discover R:R > 1 patterns from IS data only,
    then test on OOS.
    """
    logger.info("=" * 70)
    logger.info("PHASE 4: Expanding Window Walk-Forward Validation")
    logger.info("=" * 70)

    if bar_end is None:
        bar_end = n_bars

    # We operate on the neutral window subset
    total_bars = bar_end - bar_start
    seg_size = total_bars // (n_folds + 1)
    folds = []

    for fold in range(n_folds):
        is_end = bar_start + (fold + 1) * seg_size
        oos_start = is_end
        oos_end = bar_start + (fold + 2) * seg_size if fold < n_folds - 1 else bar_end

        logger.info(f"\n  WF Fold {fold+1}/{n_folds}: "
                     f"IS=[{bar_start}, {is_end}), OOS=[{oos_start}, {oos_end})")

        # Compute ATR on IS data only
        is_atr = compute_atr_ratio(
            highs[:is_end], lows[:is_end], closes[:is_end],
            DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)

        # Phase 1-3 on IS data: find R:R > 1 patterns
        is_mfe_results = []
        for pat_name, all_sigs in signal_index.items():
            sigs = [s for s in all_sigs if bar_start <= s < is_end]
            for direction in ['LONG', 'SHORT']:
                if len(sigs) < MIN_TRADES:
                    continue
                mfes, maes = compute_excursions_for_signals(
                    sigs, direction, opens, highs, lows, is_end)
                if len(mfes) < MIN_TRADES:
                    continue
                median_mfe = float(np.median(mfes))
                median_mae = float(np.median(maes))
                if median_mae <= 0:
                    continue
                ratio = median_mfe / median_mae
                if ratio <= 1.0:
                    continue
                is_mfe_results.append({
                    'pattern': pat_name,
                    'direction': direction,
                    'mfe_mae_ratio': ratio,
                    'n_signals': len(mfes),
                })

        # For each R:R > 1 pattern, find best TP/SL config
        is_passing = []
        for pat_info in is_mfe_results:
            pat_name = pat_info['pattern']
            direction = pat_info['direction']
            sigs = [s for s in signal_index[pat_name] if bar_start <= s < is_end]

            best_pat_result = None
            best_wr_excess = -999

            for cfg in RR_CONFIGS:
                excursions = compute_excursions(
                    sigs, direction, opens, highs, lows, is_end, MAX_BARS)
                if len(excursions) < MIN_TRADES:
                    continue
                tp, sl = derive_tp_sl(excursions, cfg['tp_p'], cfg['sl_p'])
                if tp is None or sl is None or tp <= sl or sl < SL_MIN:
                    continue

                baseline_wr = sl / (tp + sl) * 100
                trades = backtest_with_atr(
                    sigs, direction, tp, sl,
                    opens, highs, lows, is_end, is_atr)
                if len(trades) < MIN_TRADES:
                    continue

                pnls = [t[2] for t in trades]
                wins = sum(1 for p in pnls if p > 0)
                wr = wins / len(pnls) * 100
                wr_excess = wr - baseline_wr

                if wr_excess < WR_EXCESS_MIN:
                    continue

                mc_p = mc_test(pnls)
                if mc_p >= MC_THRESHOLD:
                    continue

                if wr_excess > best_wr_excess:
                    best_wr_excess = wr_excess
                    best_pat_result = {
                        'pattern': pat_name,
                        'direction': direction,
                        'tp': tp,
                        'sl': sl,
                    }

            if best_pat_result is not None:
                is_passing.append(best_pat_result)

        logger.info(f"    IS discovered: {len(is_passing)} R:R>1 patterns")

        # OOS backtest with IS-discovered patterns
        oos_atr = compute_atr_ratio(
            highs[:oos_end], lows[:oos_end], closes[:oos_end],
            DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)

        oos_trades = []
        for p in is_passing:
            oos_sigs = [s for s in signal_index[p['pattern']]
                        if oos_start <= s < oos_end]
            trades = backtest_with_atr(
                oos_sigs, p['direction'], p['tp'], p['sl'],
                opens, highs, lows, oos_end, oos_atr)
            oos_trades.extend(trades)

        oos_port = portfolio_1pos(oos_trades)
        oos_stats = calc_stats(oos_port)
        n_long = sum(1 for p in is_passing if p['direction'] == 'LONG')
        n_short = sum(1 for p in is_passing if p['direction'] == 'SHORT')

        fold_result = {
            'fold': fold + 1,
            'is_bars': is_end - bar_start,
            'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_passing),
            'is_long': n_long,
            'is_short': n_short,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_mdd': oos_stats['mdd'],
            'oos_pf': oos_stats['pf'],
            'oos_positive': oos_stats['pnl'] > 0,
        }
        folds.append(fold_result)

        logger.info(f"    OOS: {oos_stats['trades']} trades, "
                     f"WR {oos_stats['wr']}%, "
                     f"PnL {oos_stats['pnl']}%, "
                     f"MDD {oos_stats['mdd']}%")

    positive_folds = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_pnl'] for f in folds), 1)
    total_oos_trades = sum(f['oos_trades'] for f in folds)
    wf_pass = positive_folds == n_folds

    logger.info(f"\nWF Result: {positive_folds}/{n_folds} positive folds "
                f"({'PASS' if wf_pass else 'FAIL'})")
    logger.info(f"Total OOS PnL: {total_oos_pnl}%, Trades: {total_oos_trades}")

    return {
        'n_folds': n_folds,
        'folds': folds,
        'positive_folds': positive_folds,
        'total_oos_pnl': total_oos_pnl,
        'total_oos_trades': total_oos_trades,
        'wf_pass': wf_pass,
    }


def phase5_random_baseline(signal_index, opens, highs, lows, n_bars,
                            atr_ratio, bar_start, bar_end,
                            merged_patterns, n_iterations=10):
    """Phase 5: Random baseline comparison.

    Pick random pattern-direction combos with similar TP/SL from the R:R > 1
    parameter space and compare against the discovered patterns.
    """
    logger.info("=" * 70)
    logger.info("PHASE 5: Random Baseline Comparison")
    logger.info("=" * 70)

    if not merged_patterns:
        logger.info("No patterns to compare -- skipping.")
        return {'skipped': True, 'reason': 'no_patterns'}

    trade_boundary = bar_end if bar_end is not None else n_bars
    n_patterns = len(merged_patterns)

    # Actual portfolio performance
    all_actual_trades = []
    for p in merged_patterns:
        sigs = [s for s in signal_index[p['pattern']] if bar_start <= s < bar_end]
        trades = backtest_with_atr(
            sigs, p['direction'], p['tp'], p['sl'],
            opens, highs, lows, trade_boundary, atr_ratio)
        all_actual_trades.extend(trades)
    actual_port = portfolio_1pos(all_actual_trades)
    actual_stats = calc_stats(actual_port)

    logger.info(f"Actual ({n_patterns} patterns): "
                f"PnL={actual_stats['pnl']}%, WR={actual_stats['wr']}%, "
                f"MDD={actual_stats['mdd']}%, trades={actual_stats['trades']}")

    # Random baseline: sample random patterns with random TP/SL from the
    # same parameter space (TP > SL, SL >= 1.0%, TP from [0.5, 4.0])
    all_pats = list(signal_index.keys())
    directions = ['LONG', 'SHORT']
    rng = np.random.RandomState(42)

    random_pnls = []
    for it in range(n_iterations):
        # Pick n_patterns random combos
        rand_trades = []
        for _ in range(n_patterns):
            pat = rng.choice(all_pats)
            d = rng.choice(directions)
            # Random TP/SL with R:R > 1
            sl = rng.uniform(1.0, 3.0)
            tp = rng.uniform(sl + 0.1, sl + 2.0)  # Ensure TP > SL
            sigs = [s for s in signal_index[pat] if bar_start <= s < bar_end]
            trades = backtest_with_atr(
                sigs, d, tp, sl,
                opens, highs, lows, trade_boundary, atr_ratio)
            rand_trades.extend(trades)
        rand_port = portfolio_1pos(rand_trades)
        rand_stats = calc_stats(rand_port)
        random_pnls.append(rand_stats['pnl'])

    mean_random = float(np.mean(random_pnls))
    logger.info(f"Random baseline ({n_iterations} iterations, {n_patterns} patterns each):")
    logger.info(f"  Mean PnL: {mean_random:.1f}%")
    logger.info(f"  Range: [{np.min(random_pnls):.1f}%, {np.max(random_pnls):.1f}%]")
    logger.info(f"  Actual vs Random: {actual_stats['pnl']:.1f}% vs {mean_random:.1f}%")

    edge_over_random = actual_stats['pnl'] - mean_random

    return {
        'actual_pnl': actual_stats['pnl'],
        'actual_wr': actual_stats['wr'],
        'actual_trades': actual_stats['trades'],
        'random_mean_pnl': round(mean_random, 1),
        'random_pnls': [round(p, 1) for p in random_pnls],
        'edge_over_random': round(edge_over_random, 1),
        'genuine': edge_over_random > 0,
    }


def phase6_extended_search(df, opens, highs, lows, closes, n_bars,
                            bar_start, bar_end, types):
    """Phase 6: Search 2-candle and 4-candle patterns for R:R > 1.

    Only runs if 3-candle results are scarce.
    """
    logger.info("=" * 70)
    logger.info("PHASE 6: Extended Search (2-candle and 4-candle)")
    logger.info("=" * 70)

    trade_boundary = bar_end if bar_end is not None else n_bars
    results = {}

    for n_candles in [2, 4]:
        logger.info(f"\n--- {n_candles}-candle patterns ---")
        sig_index = build_signal_index_ncand(types, n_bars, n_candles)
        logger.info(f"  Unique patterns: {len(sig_index)}")

        total_tested = 0
        rr_gt1_count = 0
        rr_gt1_strong = 0
        top_patterns = []

        for pat_name, all_sigs in sig_index.items():
            sigs = [s for s in all_sigs if bar_start <= s < bar_end]
            for direction in ['LONG', 'SHORT']:
                if len(sigs) < MIN_TRADES:
                    continue
                total_tested += 1

                mfes, maes = compute_excursions_for_signals(
                    sigs, direction, opens, highs, lows, trade_boundary)
                if len(mfes) < MIN_TRADES:
                    continue

                median_mfe = float(np.median(mfes))
                median_mae = float(np.median(maes))
                if median_mae <= 0:
                    continue
                ratio = median_mfe / median_mae
                pct_gt = float(np.mean(mfes > maes) * 100)

                if ratio > 1.0:
                    rr_gt1_count += 1
                    if pct_gt > 50:
                        rr_gt1_strong += 1
                    top_patterns.append({
                        'pattern': pat_name,
                        'direction': direction,
                        'mfe_mae_ratio': round(ratio, 4),
                        'pct_mfe_gt_mae': round(pct_gt, 1),
                        'n_signals': len(mfes),
                        'median_mfe': round(median_mfe, 4),
                        'median_mae': round(median_mae, 4),
                    })

        top_patterns.sort(key=lambda x: x['mfe_mae_ratio'], reverse=True)

        logger.info(f"  Tested: {total_tested}")
        logger.info(f"  R:R > 1: {rr_gt1_count} "
                     f"({rr_gt1_count/total_tested*100:.1f}%)" if total_tested > 0 else "")
        logger.info(f"  Strong (>50% MFE>MAE): {rr_gt1_strong}")

        if top_patterns:
            logger.info(f"  Top 10:")
            for i, p in enumerate(top_patterns[:10]):
                logger.info(f"    {i+1}. {p['pattern']:30s} {p['direction']:5s} "
                             f"R:R={p['mfe_mae_ratio']:.3f} "
                             f"MFE>MAE={p['pct_mfe_gt_mae']:.0f}% "
                             f"n={p['n_signals']}")

        results[f'{n_candles}_candle'] = {
            'unique_patterns': len(sig_index),
            'tested': total_tested,
            'rr_gt1': rr_gt1_count,
            'pct_rr_gt1': round(rr_gt1_count / total_tested * 100, 1) if total_tested > 0 else 0,
            'strong_rr_gt1': rr_gt1_strong,
            'top_10': top_patterns[:10],
        }

    return results


def main():
    t0 = time.time()

    # ====================================================================
    # Load data
    # ====================================================================
    logger.info("Loading and classifying data...")
    df = load_and_classify(DATA_FILE)
    n_total = len(df)
    logger.info(f"Total bars: {n_total}")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values

    # Find neutral window
    nw = find_neutral_window(closes, tol_pct=NEUTRAL_TOL, min_days=90,
                              bars_per_day=BARS_PER_DAY)
    if nw is not None:
        bar_start, bar_end = nw
        nw_days = (bar_end - bar_start) / BARS_PER_DAY
        logger.info(f"Neutral window: [{bar_start}, {bar_end}] "
                     f"= {nw_days:.0f} days "
                     f"(price: {closes[bar_start]:.0f} -> {closes[bar_end]:.0f})")
    else:
        logger.warning("No neutral window found, using full data")
        bar_start, bar_end = 0, n_total

    # Build signal index (3-candle)
    signal_index = build_signal_index(types, n_total)
    logger.info(f"Unique 3-candle patterns: {len(signal_index)}")

    # Compute ATR ratio for full range
    atr_ratio = compute_atr_ratio(
        highs[:bar_end], lows[:bar_end], closes[:bar_end],
        DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)

    # ====================================================================
    # Phase 1: MFE/MAE Census
    # ====================================================================
    phase1_results, phase1_summary = phase1_mfe_mae_census(
        signal_index, opens, highs, lows, n_total, bar_start, bar_end)

    # ====================================================================
    # Phase 2: Filter R:R > 1
    # ====================================================================
    rr_gt1, phase2_summary = phase2_filter_rr_gt1(phase1_results)

    # ====================================================================
    # Phase 3: Backtest with R:R > 1 constraint
    # ====================================================================
    phase3_configs, merged_patterns, phase3_summary = phase3_backtest_rr_gt1(
        rr_gt1, signal_index, opens, highs, lows, n_total, atr_ratio,
        bar_start, bar_end)

    # ====================================================================
    # Phase 4: WF Validation
    # ====================================================================
    if merged_patterns:
        phase4_result = phase4_wf_validation(
            signal_index, opens, highs, lows, closes, n_total,
            merged_patterns, n_folds=N_FOLDS,
            bar_start=bar_start, bar_end=bar_end)
    else:
        logger.info("\nNo patterns passed Phase 3 -- skipping WF.")
        phase4_result = {'skipped': True, 'reason': 'no_patterns',
                         'wf_pass': False}

    # ====================================================================
    # Phase 5: Random Baseline
    # ====================================================================
    if merged_patterns:
        phase5_result = phase5_random_baseline(
            signal_index, opens, highs, lows, n_total, atr_ratio,
            bar_start, bar_end, merged_patterns)
    else:
        phase5_result = {'skipped': True, 'reason': 'no_patterns'}

    # ====================================================================
    # Phase 6: Extended Search (if few results)
    # ====================================================================
    few_results = len(merged_patterns) < 5
    if few_results:
        phase6_result = phase6_extended_search(
            df, opens, highs, lows, closes, n_total,
            bar_start, bar_end, types)
    else:
        logger.info(f"\n{len(merged_patterns)} patterns found -- "
                     "skipping extended search.")
        phase6_result = {'skipped': True, 'reason': 'sufficient_3candle_results'}

    # ====================================================================
    # Verdict
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("VERDICT")
    logger.info("=" * 70)

    n_rr_gt1_total = phase2_summary['total_rr_gt1']
    n_passing = phase3_summary['merged_unique_patterns']
    wf_pass = phase4_result.get('wf_pass', False)
    random_genuine = phase5_result.get('genuine', False) if not phase5_result.get('skipped') else False

    verdict = {
        'rr_gt1_patterns_exist': n_rr_gt1_total > 0,
        'n_rr_gt1_raw': n_rr_gt1_total,
        'n_passing_all_filters': n_passing,
        'wf_pass': wf_pass,
        'genuine_edge_vs_random': random_genuine,
        'recommendation': '',
    }

    if n_passing == 0:
        verdict['recommendation'] = (
            'NO_GO: No 3-candle patterns with natural R:R > 1 passed quality filters. '
            'BTC 5m price action does not exhibit systematic directional advantage '
            '(MFE > MAE) that survives statistical testing.'
        )
    elif not wf_pass:
        verdict['recommendation'] = (
            f'WEAK: {n_passing} patterns found in-sample but WF FAIL. '
            'R:R > 1 patterns are likely in-sample artifacts.'
        )
    elif not random_genuine:
        verdict['recommendation'] = (
            f'WEAK: {n_passing} patterns WF PASS but no edge over random. '
            'R:R > 1 effect may be noise.'
        )
    else:
        verdict['recommendation'] = (
            f'GO: {n_passing} R:R > 1 patterns found, WF PASS, edge over random. '
            'Consider integration but note R:R > 1 strategies typically have '
            'lower WR and higher variance.'
        )

    logger.info(f"R:R > 1 patterns exist (raw): {n_rr_gt1_total}")
    logger.info(f"Pass all quality filters: {n_passing}")
    logger.info(f"WF: {'PASS' if wf_pass else 'FAIL' if not phase4_result.get('skipped') else 'SKIPPED'}")
    logger.info(f"Genuine edge vs random: "
                f"{'YES' if random_genuine else 'NO' if not phase5_result.get('skipped') else 'SKIPPED'}")
    logger.info(f"Recommendation: {verdict['recommendation']}")

    # ====================================================================
    # Save results
    # ====================================================================
    elapsed = time.time() - t0

    output = {
        'study': 'rr_favorable_pattern_search',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'script': os.path.basename(__file__),
        'data_file': DATA_FILE,
        'elapsed_seconds': round(elapsed, 1),
        'parameters': {
            'min_trades': MIN_TRADES,
            'mc_threshold': MC_THRESHOLD,
            'wr_excess_min': WR_EXCESS_MIN,
            'sl_min': SL_MIN,
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'max_bars': MAX_BARS,
            'neutral_tol': NEUTRAL_TOL,
            'atr_period': DEFAULT_ATR_PERIOD,
            'atr_window': DEFAULT_ATR_WINDOW,
            'atr_clamp': [DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI],
            'wf_folds': N_FOLDS,
            'rr_configs': RR_CONFIGS,
        },
        'neutral_window': {
            'start_bar': bar_start,
            'end_bar': bar_end,
            'days': round((bar_end - bar_start) / BARS_PER_DAY, 1),
            'start_price': round(float(closes[bar_start]), 1),
            'end_price': round(float(closes[bar_end - 1]), 1),
        },
        'phase1_mfe_mae_census': {
            'summary': phase1_summary,
            'top_20_rr': [
                {k: v for k, v in r.items()
                 if k in ('pattern', 'direction', 'mfe_mae_ratio',
                          'pct_mfe_gt_mae', 'n_signals', 'median_mfe',
                          'median_mae')}
                for r in phase1_results[:20]
            ],
        },
        'phase2_rr_gt1_filter': phase2_summary,
        'phase3_backtest': phase3_summary,
        'phase4_wf': phase4_result,
        'phase5_random_baseline': {k: v for k, v in phase5_result.items()
                                    if k != 'random_pnls'},
        'phase6_extended': phase6_result,
        'verdict': verdict,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved to {RESULTS_FILE}")
    logger.info(f"Total elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
