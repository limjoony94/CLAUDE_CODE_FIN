#!/usr/bin/env python3
"""
MAE/MFE-Based Pattern Discovery
=================================
Instead of grid-searching TP/SL, derive them from actual trade excursions:
  - MFE (Max Favorable Excursion): how far price moves IN our favor
  - MAE (Max Adverse Excursion): how far price moves AGAINST us

A good pattern: high MFE, low MAE (price moves cleanly in predicted direction)
A bad pattern:  low MFE, high MAE (price wanders randomly)

Phase 1: MAE/MFE profiling of current 112 patterns (how "clean" are they?)
Phase 2: MAE/MFE-based TP/SL discovery (TP=MFE percentile, SL=MAE percentile)
Phase 3: MAE quality filter (reject noisy patterns)
Phase 4: WF validation of best config on 720d
"""

import sys
import os
import json
import time
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, bt_signals,
    portfolio_1pos, calc_stats, mc_test,
    FEE_PCT, LEVERAGE, MAX_BARS, MC_SEEDS, MC_SIMS,
    DEFAULT_MIN_TRADES, MAX_BASELINE_WR,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# --- Paths ---
DATA_270 = os.path.join(os.path.dirname(__file__), '..', '..',
                        'data', 'btc_5m_270days_reclassified.csv')
DATA_720 = os.path.join(os.path.dirname(__file__), '..', '..',
                        'data', 'btc_5m_720days_binance.csv')
DYN_PAT = os.path.join(os.path.dirname(__file__), '..', '..',
                       'results', 'dynamic_patterns.json')
RESULTS = os.path.join(os.path.dirname(__file__), '..', '..',
                       'results', 'mae_mfe_discovery.json')

# --- Parameters ---
EDGE_THRESHOLD = 21.8
MC_THRESHOLD = 0.01
MIN_TRADES = 25
POST_WR_MIN = 60.0
POST_SL_MIN = 1.0
N_FOLDS = 3
BARS_PER_DAY = 288

# MAE/MFE percentile grids for TP/SL derivation
TP_PERCENTILES = [40, 50, 60, 70, 80]   # TP = this percentile of MFE
SL_PERCENTILES = [70, 75, 80, 85, 90, 95]  # SL = this percentile of MAE


def compute_excursions(signal_bars, direction, opens, highs, lows, n_bars,
                       max_bars=288):
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

        mfe = 0.0  # max favorable excursion (price units)
        mae = 0.0  # max adverse excursion (price units)
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

        # Final PnL (close of last bar)
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
                         max_bars=288, min_tr=25, max_baseline_wr=70.0):
    """Grid search over MFE/MAE percentiles to find best TP/SL."""
    excursions = compute_excursions(signal_bars, direction, opens, highs, lows,
                                    n_bars, max_bars)
    if len(excursions) < min_tr:
        return None, None

    best = None
    best_score = -9999
    best_exc_stats = None

    for tp_p in TP_PERCENTILES:
        for sl_p in SL_PERCENTILES:
            tp, sl = derive_tp_sl(excursions, tp_p, sl_p)
            if tp is None or sl is None:
                continue
            if tp < 0.1 or sl < 0.3:
                continue

            bwr = sl / (tp + sl) * 100
            if bwr > max_baseline_wr:
                continue

            trades = bt_signals(signal_bars, direction, tp, sl,
                                opens, highs, lows, n_bars)
            if len(trades) < min_tr:
                continue

            pnls = [t[2] for t in trades]
            cum = 0
            peak = 0
            mdd_val = 0
            w = 0
            for p in pnls:
                cum += p
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > mdd_val:
                    mdd_val = dd
                if p > 0:
                    w += 1

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
        'mfe_p25': round(float(np.percentile(mfes, 25)), 3),
        'mfe_p75': round(float(np.percentile(mfes, 75)), 3),
        'mae_median': round(float(np.median(maes)), 3),
        'mae_mean': round(float(np.mean(maes)), 3),
        'mae_p25': round(float(np.percentile(maes, 25)), 3),
        'mae_p75': round(float(np.percentile(maes, 75)), 3),
        'mae_p90': round(float(np.percentile(maes, 90)), 3),
        'bars_to_mfe_median': round(float(np.median(bars_mfe)), 1),
        'mfe_mae_ratio': round(float(np.median(mfes) / np.median(maes)) if np.median(maes) > 0 else 0, 3),
        'clean_score': round(float(np.median(mfes) / (np.median(maes) + 0.001)), 3),
    }

    return best, exc_stats


def scan_mae_mfe(signal_index, opens, highs, lows, n_bars,
                  bar_start, bar_end, max_bars=288,
                  min_trades=25, edge_threshold=21.8, mc_threshold=0.01):
    """Pattern discovery using MAE/MFE-derived TP/SL."""
    selected = []
    trade_boundary = bar_end if bar_end is not None else n_bars

    combos = []
    for pat_key, all_sigs in signal_index.items():
        filtered_sigs = [s for s in all_sigs if bar_start <= s < bar_end]
        if len(filtered_sigs) < 5:
            continue
        for direction in ['LONG', 'SHORT']:
            combos.append((pat_key, direction, filtered_sigs))

    for pat_key, direction, sigs in combos:
        result, exc_stats = grid_search_mae_mfe(
            sigs, direction, opens, highs, lows,
            trade_boundary, max_bars,
            min_tr=min_trades, max_baseline_wr=MAX_BASELINE_WR)

        if result is None:
            continue

        bwr = result['sl'] / (result['tp'] + result['sl']) * 100
        edge = result['wr'] - bwr
        if edge < edge_threshold:
            continue

        # MC test
        trades = bt_signals(sigs, direction, result['tp'], result['sl'],
                            opens, highs, lows, trade_boundary)
        pnls = [t[2] for t in trades]
        mc_p = mc_test(pnls)
        if mc_p >= mc_threshold:
            continue

        selected.append({
            'pattern': pat_key,
            'direction': direction,
            'tp': result['tp'],
            'sl': result['sl'],
            'tp_percentile': result['tp_percentile'],
            'sl_percentile': result['sl_percentile'],
            'trades': result['trades'],
            'wr': result['wr'],
            'edge': round(edge, 1),
            'pnl': result['pnl'],
            'mdd': result['mdd'],
            'mc_p': round(mc_p, 4),
            'baseline_wr': round(bwr, 1),
            'exc_stats': exc_stats,
            'trades_raw': trades,
        })

    return selected


def post_filter(patterns):
    """WR>=60% + SL>=1.0%."""
    return [p for p in patterns
            if p['wr'] >= POST_WR_MIN and p['sl'] >= POST_SL_MIN]


def oos_backtest(signal_index, opens, highs, lows, patterns, oos_start, oos_end):
    """Standard OOS backtest."""
    all_trades = []
    for p in patterns:
        oos_sigs = [s for s in signal_index[p['pattern']]
                    if oos_start <= s < oos_end]
        trades = bt_signals(oos_sigs, p['direction'], p['tp'], p['sl'],
                            opens, highs, lows, oos_end)
        all_trades.extend(trades)
    port = portfolio_1pos(all_trades)
    stats = calc_stats(port)
    return stats, all_trades


def main():
    t0 = time.time()

    # ============================================================
    # Phase 1: MAE/MFE profiling of current 112 patterns
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: MAE/MFE Profile of Current 112 Patterns")
    logger.info("=" * 60)

    df = load_and_classify(DATA_270)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    types = df['candle_type'].values
    n_bars = len(opens)
    logger.info(f"Loaded {n_bars} bars")

    signal_index = build_signal_index(types, n_bars)

    # Load current 112 patterns
    with open(DYN_PAT) as f:
        dyn = json.load(f)

    current_patterns = list(dyn.get('pattern_details', {}).values())
    logger.info(f"Current patterns: {len(current_patterns)}")

    # Profile each pattern's excursion behavior
    profiles = []
    clean_patterns = []
    noisy_patterns = []

    for pat in current_patterns:
        key = pat['pattern']
        direction = pat['direction']
        sigs = signal_index.get(key, [])
        if len(sigs) < 10:
            continue

        exc = compute_excursions(sigs, direction, opens, highs, lows,
                                  n_bars, MAX_BARS)
        if not exc:
            continue

        mfes = np.array([e['mfe_pct'] for e in exc])
        maes = np.array([e['mae_pct'] for e in exc])
        bars_mfe = np.array([e['bars_to_mfe'] for e in exc])

        mfe_med = float(np.median(mfes))
        mae_med = float(np.median(maes))
        ratio = mfe_med / (mae_med + 0.001)
        bars_med = float(np.median(bars_mfe))

        profile = {
            'key': f"{key}_{direction}",
            'tp': pat['tp'],
            'sl': pat['sl'],
            'mfe_median': round(mfe_med, 3),
            'mae_median': round(mae_med, 3),
            'mfe_mae_ratio': round(ratio, 3),
            'bars_to_mfe_median': round(bars_med, 1),
            'mae_p90': round(float(np.percentile(maes, 90)), 3),
            'mfe_p25': round(float(np.percentile(mfes, 25)), 3),
        }
        profiles.append(profile)

        if ratio >= 1.5:  # MFE > 1.5× MAE = "clean" pattern
            clean_patterns.append(profile)
        else:
            noisy_patterns.append(profile)

    profiles.sort(key=lambda x: x['mfe_mae_ratio'], reverse=True)

    logger.info(f"\nMAE/MFE Profile Summary:")
    logger.info(f"  Total profiled: {len(profiles)}")
    logger.info(f"  Clean (MFE/MAE >= 1.5): {len(clean_patterns)}")
    logger.info(f"  Noisy (MFE/MAE < 1.5): {len(noisy_patterns)}")

    ratios = [p['mfe_mae_ratio'] for p in profiles]
    logger.info(f"  MFE/MAE ratio: min={min(ratios):.2f}, median={np.median(ratios):.2f}, "
                f"max={max(ratios):.2f}")

    mae_meds = [p['mae_median'] for p in profiles]
    mfe_meds = [p['mfe_median'] for p in profiles]
    bars_meds = [p['bars_to_mfe_median'] for p in profiles]
    logger.info(f"  MAE median: {np.median(mae_meds):.3f}%")
    logger.info(f"  MFE median: {np.median(mfe_meds):.3f}%")
    logger.info(f"  Bars to MFE median: {np.median(bars_meds):.0f} bars "
                f"({np.median(bars_meds)*5/60:.1f} hours)")

    # Top 10 cleanest
    logger.info(f"\n  Top 10 CLEANEST patterns (high MFE/MAE ratio):")
    for p in profiles[:10]:
        logger.info(f"    {p['key']:<25} MFE/MAE={p['mfe_mae_ratio']:.2f}  "
                     f"MFE={p['mfe_median']:.2f}%  MAE={p['mae_median']:.2f}%  "
                     f"TP={p['tp']}%  SL={p['sl']}%  "
                     f"bars_to_MFE={p['bars_to_mfe_median']:.0f}")

    # Bottom 10 noisiest
    logger.info(f"\n  Top 10 NOISIEST patterns (low MFE/MAE ratio):")
    for p in profiles[-10:]:
        logger.info(f"    {p['key']:<25} MFE/MAE={p['mfe_mae_ratio']:.2f}  "
                     f"MFE={p['mfe_median']:.2f}%  MAE={p['mae_median']:.2f}%  "
                     f"TP={p['tp']}%  SL={p['sl']}%  "
                     f"bars_to_MFE={p['bars_to_mfe_median']:.0f}")

    # ============================================================
    # Phase 2: MAE/MFE-based discovery on 270d
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: MAE/MFE-Based Discovery (270d)")
    logger.info(f"{'='*60}")
    logger.info(f"TP percentile grid: {TP_PERCENTILES}")
    logger.info(f"SL percentile grid: {SL_PERCENTILES}")

    t1 = time.time()
    raw = scan_mae_mfe(
        signal_index, opens, highs, lows, n_bars,
        bar_start=0, bar_end=n_bars, max_bars=MAX_BARS,
        min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
        mc_threshold=MC_THRESHOLD)
    filtered = post_filter(raw)
    elapsed_disc = time.time() - t1

    n_long = sum(1 for p in filtered if p['direction'] == 'LONG')
    n_short = sum(1 for p in filtered if p['direction'] == 'SHORT')
    logger.info(f"  {len(raw)} raw → {len(filtered)} filtered "
                 f"({n_long}L+{n_short}S) in {elapsed_disc:.1f}s")

    # Portfolio stats
    if filtered:
        all_trades = []
        for p in filtered:
            all_trades.extend(p['trades_raw'])
        port = portfolio_1pos(all_trades)
        stats = calc_stats(port)

        tps = [p['tp'] for p in filtered]
        sls = [p['sl'] for p in filtered]
        edges = [p['edge'] for p in filtered]
        ratios_d = [p['exc_stats']['mfe_mae_ratio'] for p in filtered]

        logger.info(f"  Portfolio: {stats['trades']}t, WR {stats['wr']}%, "
                     f"PnL {stats['pnl']}%, MDD {stats['mdd']}%, PF {stats['pf']}")
        logger.info(f"  TP: {min(tps):.1f}~{max(tps):.1f} (med {np.median(tps):.1f})")
        logger.info(f"  SL: {min(sls):.1f}~{max(sls):.1f} (med {np.median(sls):.1f})")
        logger.info(f"  Edge: {np.mean(edges):.1f}pp mean")
        logger.info(f"  MFE/MAE ratio: {np.mean(ratios_d):.2f} mean")
    else:
        stats = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}

    # Derived TP/SL percentile distribution
    if filtered:
        tp_ps = [p['tp_percentile'] for p in filtered]
        sl_ps = [p['sl_percentile'] for p in filtered]
        logger.info(f"  TP percentile choices: {dict(zip(*np.unique(tp_ps, return_counts=True)))}")
        logger.info(f"  SL percentile choices: {dict(zip(*np.unique(sl_ps, return_counts=True)))}")

    discovery_result = {
        'n_raw': len(raw),
        'n_filtered': len(filtered),
        'n_long': n_long,
        'n_short': n_short,
        'portfolio_stats': stats,
        'patterns': [{
            'key': f"{p['pattern']}_{p['direction']}",
            'pattern': p['pattern'],
            'direction': p['direction'],
            'tp': p['tp'],
            'sl': p['sl'],
            'tp_percentile': p['tp_percentile'],
            'sl_percentile': p['sl_percentile'],
            'trades': p['trades'],
            'wr': p['wr'],
            'edge': p['edge'],
            'mfe_mae_ratio': p['exc_stats']['mfe_mae_ratio'],
            'mfe_median': p['exc_stats']['mfe_median'],
            'mae_median': p['exc_stats']['mae_median'],
            'bars_to_mfe': p['exc_stats']['bars_to_mfe_median'],
        } for p in filtered],
        'elapsed': round(elapsed_disc, 1),
    }

    # ============================================================
    # Phase 3: Clean-pattern filtered portfolio (from 112)
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 3: Clean-Pattern Filter (MFE/MAE >= threshold)")
    logger.info(f"{'='*60}")

    thresholds = [1.0, 1.25, 1.5, 2.0]
    clean_results = {}

    for thresh in thresholds:
        clean_keys = set(p['key'] for p in profiles if p['mfe_mae_ratio'] >= thresh)
        clean_pats = [pat for pat in current_patterns
                      if f"{pat['pattern']}_{pat['direction']}" in clean_keys]
        n_clean = len(clean_pats)

        # Backtest clean subset
        all_trades = []
        for pat in clean_pats:
            sigs = signal_index.get(pat['pattern'], [])
            trades = bt_signals(sigs, pat['direction'], pat['tp'], pat['sl'],
                                opens, highs, lows, n_bars)
            all_trades.extend(trades)

        port = portfolio_1pos(all_trades)
        s = calc_stats(port)

        n_l = sum(1 for p in clean_pats if p['direction'] == 'LONG')
        n_s = sum(1 for p in clean_pats if p['direction'] == 'SHORT')

        logger.info(f"  MFE/MAE >= {thresh}: {n_clean} patterns ({n_l}L+{n_s}S), "
                     f"{s['trades']}t, WR {s['wr']}%, PnL {s['pnl']}%, "
                     f"MDD {s['mdd']}%, PF {s['pf']}")

        clean_results[f"ratio_{thresh}"] = {
            'threshold': thresh,
            'n_patterns': n_clean,
            'n_long': n_l,
            'n_short': n_s,
            'portfolio_stats': s,
        }

    # ============================================================
    # Phase 4: WF validation on 720d
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 4: Walk-Forward on 720d (MAE/MFE discovery)")
    logger.info(f"{'='*60}")

    df_720 = load_and_classify(DATA_720)
    opens_720 = df_720['open'].values.astype(np.float64)
    highs_720 = df_720['high'].values.astype(np.float64)
    lows_720 = df_720['low'].values.astype(np.float64)
    types_720 = df_720['candle_type'].values
    n_bars_720 = len(opens_720)
    signal_index_720 = build_signal_index(types_720, n_bars_720)

    seg_size = n_bars_720 // (N_FOLDS + 1)

    folds = []
    all_stable = {}

    for fold in range(N_FOLDS):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < N_FOLDS - 1 else n_bars_720
        is_days = is_end // BARS_PER_DAY
        oos_days = (oos_end - oos_start) // BARS_PER_DAY

        logger.info(f"\nFold {fold+1}: IS=[0,{is_days}d) OOS=[{is_days}d,{is_days+oos_days}d)")

        t1 = time.time()
        is_raw = scan_mae_mfe(
            signal_index_720, opens_720, highs_720, lows_720, n_bars_720,
            bar_start=0, bar_end=is_end, max_bars=MAX_BARS,
            min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
            mc_threshold=MC_THRESHOLD)
        is_filtered = post_filter(is_raw)
        t_disc = time.time() - t1

        n_long = sum(1 for p in is_filtered if p['direction'] == 'LONG')
        n_short = sum(1 for p in is_filtered if p['direction'] == 'SHORT')
        logger.info(f"  IS: {len(is_raw)} raw → {len(is_filtered)} "
                     f"({n_long}L+{n_short}S) in {t_disc:.1f}s")

        for p in is_filtered:
            key = f"{p['pattern']}_{p['direction']}"
            all_stable[key] = all_stable.get(key, 0) + 1

        if is_filtered:
            oos_stats, _ = oos_backtest(signal_index_720, opens_720, highs_720,
                                         lows_720, is_filtered, oos_start, oos_end)
        else:
            oos_stats = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}

        status = "PASS" if oos_stats['pnl'] > 0 else "FAIL"
        logger.info(f"  OOS: {oos_stats['trades']}t, WR {oos_stats['wr']}%, "
                     f"PnL {oos_stats['pnl']}%, MDD {oos_stats['mdd']}% → {status}")

        folds.append({
            'fold': fold + 1,
            'is_days': is_days,
            'oos_days': oos_days,
            'is_patterns': len(is_filtered),
            'is_long': n_long,
            'is_short': n_short,
            'oos_stats': oos_stats,
            'oos_positive': oos_stats['pnl'] > 0,
        })

    stable_pats = sorted(k for k, v in all_stable.items() if v == N_FOLDS)
    n_pos = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_stats']['pnl'] for f in folds), 1)
    avg_oos_wr = round(float(np.mean([f['oos_stats']['wr'] for f in folds])), 1) if folds else 0
    avg_oos_mdd = round(float(np.mean([f['oos_stats']['mdd'] for f in folds])), 1) if folds else 0
    verdict = "PASS" if n_pos >= 2 else "FAIL"

    logger.info(f"\n{'='*60}")
    logger.info(f"WF SUMMARY (MAE/MFE-based)")
    logger.info(f"{'='*60}")
    logger.info(f"Positive folds: {n_pos}/{N_FOLDS}")
    logger.info(f"Total OOS PnL: {total_oos_pnl}%")
    logger.info(f"Avg OOS WR: {avg_oos_wr}%")
    logger.info(f"Verdict: {verdict}")

    wf_result = {
        'folds': folds,
        'summary': {
            'positive_folds': n_pos,
            'total_oos_pnl': total_oos_pnl,
            'avg_oos_wr': avg_oos_wr,
            'avg_oos_mdd': avg_oos_mdd,
            'stable_patterns': stable_pats[:30],
            'verdict': verdict,
        }
    }

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    # Save
    output = {
        'study': 'MAE/MFE-Based Pattern Discovery',
        'date': '2026-02-18',
        'phase1_profiles': {
            'total_profiled': len(profiles),
            'clean_count': len(clean_patterns),
            'noisy_count': len(noisy_patterns),
            'top_10_cleanest': profiles[:10],
            'top_10_noisiest': profiles[-10:],
            'all_profiles': profiles,
        },
        'phase2_mae_mfe_discovery': discovery_result,
        'phase3_clean_filter': clean_results,
        'phase4_wf_720d': wf_result,
        'elapsed_seconds': round(elapsed, 1),
    }

    with open(RESULTS, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {RESULTS}")


if __name__ == '__main__':
    main()
