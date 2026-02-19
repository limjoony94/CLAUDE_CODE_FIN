#!/usr/bin/env python3
"""
Tight TP/SL Discovery Study
============================
MAX_BARS=288 (24h DROP) + tight TP/SL grid:
  TP: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.5]
  SL: [1.0, 1.2, 1.5, 1.8, 2.0]

Phase 1: Discovery on 270d data (IS stats)
Phase 2: 3-fold Expanding Window WF on 720d data (OOS validation)

Compares with current 112-pattern wide-grid results.
"""

import sys
import os
import json
import time
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import scripts.scanner.pattern_scanner as scanner
from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index,
    scan_universe_range, bt_signals, portfolio_1pos, calc_stats, mc_test,
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
RESULTS = os.path.join(os.path.dirname(__file__), '..', '..',
                       'results', 'tight_tpsl_discovery.json')

# --- Tight grid ---
TIGHT_TP_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.5]
TIGHT_SL_GRID = [1.0, 1.2, 1.5, 1.8, 2.0]

# --- Wide grid (current, for comparison) ---
WIDE_TP_GRID = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.5, 3.0]
WIDE_SL_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# --- Scanner parameters ---
EDGE_THRESHOLD = 21.8
MC_THRESHOLD = 0.01
MIN_TRADES = 25
MAX_BASELINE_WR = 70.0
POST_WR_MIN = 60.0
POST_SL_MIN = 1.0
N_FOLDS = 3
BARS_PER_DAY = 288


def set_grid(tp_grid, sl_grid):
    """Override scanner module-level grid constants."""
    scanner.TP_GRID = tp_grid
    scanner.SL_GRID = sl_grid


def post_filter(patterns):
    """Apply SL>=1.0% + WR>=60% post-filter."""
    return [p for p in patterns
            if p['sl'] >= POST_SL_MIN and p['wr'] >= POST_WR_MIN]


def oos_backtest(signal_index, opens, highs, lows, patterns, oos_start, oos_end):
    """Backtest IS-discovered patterns on OOS period."""
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


def run_discovery(signal_index, opens, highs, lows, n_bars, label, tp_grid, sl_grid):
    """Run pattern discovery with given grid."""
    set_grid(tp_grid, sl_grid)
    t0 = time.time()
    raw = scan_universe_range(
        signal_index, opens, highs, lows, n_bars,
        bar_start=0, bar_end=n_bars, mode='per_pattern',
        min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
        mc_threshold=MC_THRESHOLD, max_baseline_wr=MAX_BASELINE_WR,
    )
    elapsed = time.time() - t0

    filtered = post_filter(raw)
    n_long = sum(1 for p in filtered if p['direction'] == 'LONG')
    n_short = sum(1 for p in filtered if p['direction'] == 'SHORT')

    logger.info(f"  [{label}] {len(raw)} raw → {len(filtered)} filtered "
                f"({n_long}L+{n_short}S) in {elapsed:.1f}s")

    # Portfolio stats
    all_trades = []
    for p in filtered:
        sigs = signal_index[p['pattern']]
        trades = bt_signals(sigs, p['direction'], p['tp'], p['sl'],
                            opens, highs, lows, n_bars)
        all_trades.extend(trades)
    port = portfolio_1pos(all_trades)
    stats = calc_stats(port)

    # TP/SL distribution
    if filtered:
        tps = [p['tp'] for p in filtered]
        sls = [p['sl'] for p in filtered]
        edges = [p['edge'] for p in filtered]
        wrs = [p['wr'] for p in filtered]
        tp_dist = {'min': min(tps), 'median': round(float(np.median(tps)), 2),
                   'mean': round(float(np.mean(tps)), 2), 'max': max(tps)}
        sl_dist = {'min': min(sls), 'median': round(float(np.median(sls)), 2),
                   'mean': round(float(np.mean(sls)), 2), 'max': max(sls)}
        edge_dist = {'min': round(min(edges), 1), 'mean': round(float(np.mean(edges)), 1),
                     'max': round(max(edges), 1)}
        wr_dist = {'min': round(min(wrs), 1), 'mean': round(float(np.mean(wrs)), 1),
                   'max': round(max(wrs), 1)}
        be_wrs = [s / (t + s) * 100 for t, s in zip(tps, sls)]
        be_dist = {'min': round(min(be_wrs), 1), 'mean': round(float(np.mean(be_wrs)), 1),
                   'max': round(max(be_wrs), 1)}
    else:
        tp_dist = sl_dist = edge_dist = wr_dist = be_dist = {}

    logger.info(f"  [{label}] Portfolio: {stats['trades']} trades, WR {stats['wr']}%, "
                f"PnL {stats['pnl']}%, MDD {stats['mdd']}%, PF {stats['pf']}")
    logger.info(f"  [{label}] TP: {tp_dist} | SL: {sl_dist}")
    logger.info(f"  [{label}] Edge: {edge_dist} | BE WR: {be_dist}")

    return {
        'label': label,
        'tp_grid': tp_grid,
        'sl_grid': sl_grid,
        'n_raw': len(raw),
        'n_filtered': len(filtered),
        'n_long': n_long,
        'n_short': n_short,
        'portfolio_stats': stats,
        'tp_dist': tp_dist,
        'sl_dist': sl_dist,
        'edge_dist': edge_dist,
        'wr_dist': wr_dist,
        'be_wr_dist': be_dist,
        'patterns': [{
            'key': f"{p['pattern']}_{p['direction']}",
            'pattern': p['pattern'],
            'direction': p['direction'],
            'tp': p['tp'],
            'sl': p['sl'],
            'trades': p['trades'],
            'wr': p['wr'],
            'edge': p['edge'],
        } for p in filtered],
        'elapsed': round(elapsed, 1),
    }


def run_wf(data_path, tp_grid, sl_grid, label):
    """Run 3-fold expanding window WF."""
    logger.info(f"\n{'='*60}")
    logger.info(f"WF Validation: {label}")
    logger.info(f"{'='*60}")

    df = load_and_classify(data_path)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    types = df['candle_type'].values
    n_bars = len(opens)
    total_days = n_bars // BARS_PER_DAY
    logger.info(f"Loaded {n_bars} bars ({total_days} days)")

    signal_index = build_signal_index(types, n_bars)
    set_grid(tp_grid, sl_grid)

    seg_size = n_bars // (N_FOLDS + 1)
    seg_days = seg_size // BARS_PER_DAY

    folds = []
    all_stable = {}

    for fold in range(N_FOLDS):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < N_FOLDS - 1 else n_bars
        is_days = is_end // BARS_PER_DAY
        oos_days = (oos_end - oos_start) // BARS_PER_DAY

        logger.info(f"\nFold {fold+1}/{N_FOLDS}: IS=[0, {is_days}d), OOS=[{is_days}d, {is_days+oos_days}d)")

        # Discovery on IS
        t1 = time.time()
        is_raw = scan_universe_range(
            signal_index, opens, highs, lows, n_bars,
            bar_start=0, bar_end=is_end, mode='per_pattern',
            min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
            mc_threshold=MC_THRESHOLD, max_baseline_wr=MAX_BASELINE_WR,
        )
        t_disc = time.time() - t1

        is_filtered = post_filter(is_raw)
        n_long = sum(1 for p in is_filtered if p['direction'] == 'LONG')
        n_short = sum(1 for p in is_filtered if p['direction'] == 'SHORT')
        logger.info(f"  Discovery: {len(is_raw)} raw → {len(is_filtered)} filtered "
                     f"({n_long}L+{n_short}S) in {t_disc:.1f}s")

        for p in is_filtered:
            key = f"{p['pattern']}_{p['direction']}"
            all_stable[key] = all_stable.get(key, 0) + 1

        # IS backtest
        is_stats, _ = oos_backtest(signal_index, opens, highs, lows,
                                   is_filtered, 0, is_end)
        logger.info(f"  IS:  {is_stats['trades']} trades, WR {is_stats['wr']}%, "
                     f"PnL {is_stats['pnl']}%, MDD {is_stats['mdd']}%, PF {is_stats['pf']}")

        # OOS backtest
        oos_stats, oos_trades_raw = oos_backtest(
            signal_index, opens, highs, lows,
            is_filtered, oos_start, oos_end)
        status = "PASS" if oos_stats['pnl'] > 0 else "FAIL"
        logger.info(f"  OOS: {oos_stats['trades']} trades, WR {oos_stats['wr']}%, "
                     f"PnL {oos_stats['pnl']}%, MDD {oos_stats['mdd']}%, PF {oos_stats['pf']} "
                     f"→ {status}")

        folds.append({
            'fold': fold + 1,
            'is_days': is_days,
            'oos_days': oos_days,
            'is_patterns_raw': len(is_raw),
            'is_patterns_filtered': len(is_filtered),
            'is_long': n_long,
            'is_short': n_short,
            'is_stats': is_stats,
            'oos_stats': oos_stats,
            'oos_positive': oos_stats['pnl'] > 0,
        })

    stable_pats = sorted(k for k, v in all_stable.items() if v == N_FOLDS)
    n_positive = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_stats']['pnl'] for f in folds), 1)
    total_oos_trades = sum(f['oos_stats']['trades'] for f in folds)
    avg_oos_wr = round(float(np.mean([f['oos_stats']['wr'] for f in folds])), 1)
    avg_oos_mdd = round(float(np.mean([f['oos_stats']['mdd'] for f in folds])), 1)

    verdict = "PASS" if n_positive >= 2 else "FAIL"

    logger.info(f"\n{'='*60}")
    logger.info(f"WF SUMMARY [{label}]")
    logger.info(f"{'='*60}")
    logger.info(f"Positive OOS folds: {n_positive}/{N_FOLDS}")
    logger.info(f"Total OOS PnL: {total_oos_pnl}%")
    logger.info(f"Total OOS trades: {total_oos_trades}")
    logger.info(f"Avg OOS WR: {avg_oos_wr}%")
    logger.info(f"Avg OOS MDD: {avg_oos_mdd}%")
    logger.info(f"Stable patterns (all {N_FOLDS} folds): {len(stable_pats)}")
    if stable_pats:
        logger.info(f"  {stable_pats[:30]}")
    logger.info(f"Verdict: {verdict}")

    return {
        'label': label,
        'data_days': n_bars // BARS_PER_DAY,
        'n_folds': N_FOLDS,
        'folds': folds,
        'summary': {
            'positive_folds': n_positive,
            'total_oos_pnl': total_oos_pnl,
            'total_oos_trades': total_oos_trades,
            'avg_oos_wr': avg_oos_wr,
            'avg_oos_mdd': avg_oos_mdd,
            'stable_pattern_count': len(stable_pats),
            'stable_patterns': stable_pats[:50],
            'verdict': verdict,
        }
    }


def main():
    t0 = time.time()

    # ============================================================
    # Phase 1: Discovery comparison on 270d data
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Discovery on 270d data")
    logger.info("=" * 60)

    logger.info(f"\nLoading 270d data...")
    df_270 = load_and_classify(DATA_270)
    opens = df_270['open'].values.astype(np.float64)
    highs = df_270['high'].values.astype(np.float64)
    lows = df_270['low'].values.astype(np.float64)
    types = df_270['candle_type'].values
    n_bars = len(opens)
    logger.info(f"Loaded {n_bars} bars ({n_bars // BARS_PER_DAY} days)")

    signal_index = build_signal_index(types, n_bars)

    # A: Tight grid
    logger.info("\n--- A: TIGHT grid (TP 0.3~1.5, SL 1.0~2.0) ---")
    tight_result = run_discovery(signal_index, opens, highs, lows, n_bars,
                                 "TIGHT", TIGHT_TP_GRID, TIGHT_SL_GRID)

    # B: Wide grid (current, for comparison)
    logger.info("\n--- B: WIDE grid (TP 0.5~3.0, SL 0.5~4.0) ---")
    wide_result = run_discovery(signal_index, opens, highs, lows, n_bars,
                                "WIDE", WIDE_TP_GRID, WIDE_SL_GRID)

    # Comparison
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1 COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"{'Metric':<25} {'TIGHT':<20} {'WIDE':<20}")
    logger.info(f"{'-'*65}")
    for key in ['n_filtered', 'n_long', 'n_short']:
        logger.info(f"{key:<25} {tight_result[key]:<20} {wide_result[key]:<20}")
    for key in ['pnl', 'trades', 'wr', 'mdd', 'pf']:
        t_val = tight_result['portfolio_stats'][key]
        w_val = wide_result['portfolio_stats'][key]
        logger.info(f"{key:<25} {t_val:<20} {w_val:<20}")

    # ============================================================
    # Phase 2: WF on 720d data
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: Walk-Forward on 720d Binance data")
    logger.info(f"{'='*60}")

    tight_wf = run_wf(DATA_720, TIGHT_TP_GRID, TIGHT_SL_GRID, "TIGHT_720d")
    wide_wf = run_wf(DATA_720, WIDE_TP_GRID, WIDE_SL_GRID, "WIDE_720d")

    # WF Comparison
    logger.info(f"\n{'='*60}")
    logger.info("FINAL WF COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"{'Metric':<25} {'TIGHT':<20} {'WIDE':<20}")
    logger.info(f"{'-'*65}")
    for key in ['positive_folds', 'total_oos_pnl', 'total_oos_trades',
                'avg_oos_wr', 'avg_oos_mdd', 'stable_pattern_count', 'verdict']:
        t_val = tight_wf['summary'][key]
        w_val = wide_wf['summary'][key]
        logger.info(f"{key:<25} {t_val:<20} {w_val:<20}")

    elapsed = time.time() - t0

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'study': 'Tight TP/SL Discovery Study',
        'date': '2026-02-18',
        'tight_grid': {'tp': TIGHT_TP_GRID, 'sl': TIGHT_SL_GRID},
        'wide_grid': {'tp': WIDE_TP_GRID, 'sl': WIDE_SL_GRID},
        'parameters': {
            'edge_threshold': EDGE_THRESHOLD,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
            'max_baseline_wr': MAX_BASELINE_WR,
            'post_filter': f'SL>={POST_SL_MIN}% + WR>={POST_WR_MIN}%',
            'max_bars': 288,
        },
        'phase1_270d': {
            'tight': tight_result,
            'wide': wide_result,
        },
        'phase2_wf_720d': {
            'tight': tight_wf,
            'wide': wide_wf,
        },
        'elapsed_seconds': round(elapsed, 1),
    }

    with open(RESULTS, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {RESULTS}")
    logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
