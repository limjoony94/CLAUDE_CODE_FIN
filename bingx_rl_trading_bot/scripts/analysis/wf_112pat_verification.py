#!/usr/bin/env python3
"""
WF OOS Verification — 112-pattern discovery pipeline
=====================================================
720d Binance data, 3-fold expanding window, per_pattern mode.
Matches current scanner parameters + SL>=1.0% post-filter.

Tests: "Does the discovery pipeline (edge>=21.8pp, MC<0.01, SL>=1.0%)
        produce profitable patterns in OOS?"
"""

import sys
import os
import json
import time
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
DATA_720 = os.path.join(os.path.dirname(__file__), '..', '..',
                        'data', 'btc_5m_720days_binance.csv')
RESULTS = os.path.join(os.path.dirname(__file__), '..', '..',
                       'results', 'wf_112pat_verification.json')

# --- Scanner parameters (matching current dynamic_patterns.json) ---
EDGE_THRESHOLD = 21.8    # post-filter edge
MC_THRESHOLD = 0.01
MIN_TRADES = 25
MAX_BASELINE_WR = 70.0
POST_SL_MIN = 1.0        # SL >= 1.0% (v1.28.11 lesson)
POST_WR_MIN = 60.0       # WR >= 60%
N_FOLDS = 3
BARS_PER_DAY = 288


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


def main():
    t0 = time.time()

    # --- Load data ---
    logger.info("Loading 720d Binance data...")
    df = load_and_classify(DATA_720)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows  = df['low'].values.astype(np.float64)
    types = df['candle_type'].values
    n_bars = len(opens)
    total_days = n_bars // BARS_PER_DAY
    logger.info(f"Loaded {n_bars} bars ({total_days} days)")

    signal_index = build_signal_index(types, n_bars)

    # --- Define folds ---
    # n_folds+1 equal segments; fold f: IS=[0, (f+1)*seg), OOS=[(f+1)*seg, (f+2)*seg)
    seg_size = n_bars // (N_FOLDS + 1)
    seg_days = seg_size // BARS_PER_DAY

    logger.info(f"\n{'='*60}")
    logger.info(f"3-fold Expanding Window WF | Segment = {seg_days} days")
    logger.info(f"Parameters: per_pattern, edge>={EDGE_THRESHOLD}pp, MC<{MC_THRESHOLD}, "
                f"min_trades>={MIN_TRADES}, SL>={POST_SL_MIN}%")
    logger.info(f"{'='*60}")

    folds = []
    all_stable = {}  # pattern_key -> appearance count

    for fold in range(N_FOLDS):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < N_FOLDS - 1 else n_bars
        is_days = is_end // BARS_PER_DAY
        oos_days = (oos_end - oos_start) // BARS_PER_DAY

        logger.info(f"\nFold {fold+1}/{N_FOLDS}: IS=[0, {is_days}d), OOS=[{is_days}d, {is_days+oos_days}d)")

        # Step 1: Discover on IS
        t1 = time.time()
        is_raw = scan_universe_range(
            signal_index, opens, highs, lows, n_bars,
            bar_start=0, bar_end=is_end, mode='per_pattern',
            min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
            mc_threshold=MC_THRESHOLD, max_baseline_wr=MAX_BASELINE_WR,
        )
        t_disc = time.time() - t1

        # Step 2: Post-filter
        is_filtered = post_filter(is_raw)
        n_long = sum(1 for p in is_filtered if p['direction'] == 'LONG')
        n_short = sum(1 for p in is_filtered if p['direction'] == 'SHORT')
        logger.info(f"  Discovery: {len(is_raw)} raw → {len(is_filtered)} filtered "
                     f"({n_long}L+{n_short}S) in {t_disc:.1f}s")

        # Track stability
        for p in is_filtered:
            key = f"{p['pattern']}_{p['direction']}"
            all_stable[key] = all_stable.get(key, 0) + 1

        # Step 3: IS backtest (for reference)
        is_stats, _ = oos_backtest(signal_index, opens, highs, lows,
                                   is_filtered, 0, is_end)
        logger.info(f"  IS:  {is_stats['trades']} trades, WR {is_stats['wr']}%, "
                     f"PnL {is_stats['pnl']}%, MDD {is_stats['mdd']}%, PF {is_stats['pf']}")

        # Step 4: OOS backtest
        oos_stats, oos_trades_raw = oos_backtest(
            signal_index, opens, highs, lows,
            is_filtered, oos_start, oos_end)
        status = "PASS" if oos_stats['pnl'] > 0 else "FAIL"
        logger.info(f"  OOS: {oos_stats['trades']} trades, WR {oos_stats['wr']}%, "
                     f"PnL {oos_stats['pnl']}%, MDD {oos_stats['mdd']}%, PF {oos_stats['pf']} "
                     f"→ {status}")

        # OOS MC test
        oos_pnls = [t[2] for t in portfolio_1pos(
            sorted([(t[0], t[1], t[2]) for t in oos_trades_raw], key=lambda x: x[0])
        )]
        oos_mc_p = mc_test(oos_pnls) if len(oos_pnls) >= 5 else 1.0

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
            'oos_mc_p': round(oos_mc_p, 4),
            'oos_positive': oos_stats['pnl'] > 0,
        })

    # --- Summary ---
    stable_pats = sorted(k for k, v in all_stable.items() if v == N_FOLDS)
    n_positive = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_stats']['pnl'] for f in folds), 1)
    total_oos_trades = sum(f['oos_stats']['trades'] for f in folds)
    avg_oos_wr = round(np.mean([f['oos_stats']['wr'] for f in folds]), 1)
    avg_oos_mdd = round(np.mean([f['oos_stats']['mdd'] for f in folds]), 1)

    verdict = "PASS" if n_positive >= 2 else "FAIL"
    elapsed = time.time() - t0

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Positive OOS folds: {n_positive}/{N_FOLDS}")
    logger.info(f"Total OOS PnL: {total_oos_pnl}%")
    logger.info(f"Total OOS trades: {total_oos_trades}")
    logger.info(f"Avg OOS WR: {avg_oos_wr}%")
    logger.info(f"Avg OOS MDD: {avg_oos_mdd}%")
    logger.info(f"Stable patterns (all {N_FOLDS} folds): {len(stable_pats)}")
    if stable_pats:
        logger.info(f"  {stable_pats[:30]}")
    logger.info(f"\nVerdict: {verdict}")
    logger.info(f"Time: {elapsed:.1f}s")

    # --- Save ---
    output = {
        'description': '112-pattern WF OOS verification (expanding window, 720d Binance)',
        'parameters': {
            'data': 'btc_5m_720days_binance.csv',
            'n_folds': N_FOLDS,
            'mode': 'per_pattern',
            'edge_threshold': EDGE_THRESHOLD,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
            'max_baseline_wr': MAX_BASELINE_WR,
            'post_filter': f'SL>={POST_SL_MIN}% + WR>={POST_WR_MIN}%',
        },
        'folds': folds,
        'summary': {
            'positive_folds': n_positive,
            'total_oos_pnl': total_oos_pnl,
            'total_oos_trades': total_oos_trades,
            'avg_oos_wr': avg_oos_wr,
            'avg_oos_mdd': avg_oos_mdd,
            'stable_pattern_count': len(stable_pats),
            'stable_patterns': stable_pats,
            'verdict': verdict,
        },
        'elapsed_seconds': round(elapsed, 1),
    }

    with open(RESULTS, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {RESULTS}")


if __name__ == '__main__':
    main()
