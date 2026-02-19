#!/usr/bin/env python3
"""
ATR-Normalized TP/SL Discovery Study
=====================================
Instead of fixed TP/SL %, use ATR multiples:
  TP = tp_mult × ATR(period) as % of price
  SL = sl_mult × ATR(period) as % of price

This adapts TP/SL to current volatility:
  - High vol → wider TP/SL (resolves in similar # of bars)
  - Low vol  → tighter TP/SL (still resolves in similar # of bars)

Also tests SHORTER MAX_BARS (3h, 6h, 12h) to match
the pattern's expected predictive horizon.

Phase 1: ATR analysis (what does BTC 5m ATR look like?)
Phase 2: ATR-normalized discovery on 270d
Phase 3: Multiple MAX_BARS sweep
Phase 4: WF validation on 720d for best config
"""

import sys
import os
import json
import time
import logging
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index,
    portfolio_1pos, calc_stats, mc_test,
    FEE_PCT, LEVERAGE, MC_SEEDS, MC_SIMS,
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
RESULTS = os.path.join(os.path.dirname(__file__), '..', '..',
                       'results', 'atr_normalized_discovery.json')

# --- ATR parameters ---
ATR_PERIOD = 60  # 60 bars = 5 hours

# --- ATR multiplier grids ---
TP_MULT_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SL_MULT_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

# --- MAX_BARS sweep ---
MAX_BARS_OPTIONS = [36, 72, 144, 288]  # 3h, 6h, 12h, 24h

# --- Scanner parameters ---
EDGE_THRESHOLD = 21.8
MC_THRESHOLD = 0.01
MIN_TRADES = 25
POST_WR_MIN = 60.0
POST_SL_MIN = 1.0  # minimum effective avg SL
N_FOLDS = 3
BARS_PER_DAY = 288


def compute_atr(highs, lows, closes, period=ATR_PERIOD):
    """Compute ATR as percentage of price. Returns array of ATR%."""
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = (highs[0] - lows[0])

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # SMA-based ATR
    atr = np.zeros(n)
    for i in range(n):
        start = max(0, i - period + 1)
        atr[i] = np.mean(tr[start:i + 1])

    # Convert to percentage of close price
    atr_pct = np.zeros(n)
    for i in range(n):
        if closes[i] > 0:
            atr_pct[i] = atr[i] / closes[i] * 100
    return atr_pct


def bt_signals_atr(signal_bars, direction, tp_mult, sl_mult,
                   opens, highs, lows, atr_pct, n_bars, max_bars=288):
    """Backtest with ATR-normalized TP/SL.

    tp_mult/sl_mult are multipliers of ATR% at signal time.
    """
    fee = FEE_PCT * LEVERAGE
    trades = []
    tp_pcts_used = []
    sl_pcts_used = []

    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue

        # ATR at signal bar
        atr_at_signal = atr_pct[idx]
        if atr_at_signal <= 0:
            continue

        # Compute actual TP/SL % for this trade
        tp_pct = tp_mult * atr_at_signal
        sl_pct = sl_mult * atr_at_signal

        # Skip if SL too small (spread/slip would dominate)
        if sl_pct < 0.3:
            continue

        tp_pcts_used.append(tp_pct)
        sl_pcts_used.append(sl_pct)

        eb = idx + 1

        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        for j in range(idx + 2, min(idx + 2 + max_bars, n_bars)):
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
        # Timeout → DROP

    avg_tp = round(float(np.mean(tp_pcts_used)), 3) if tp_pcts_used else 0
    avg_sl = round(float(np.mean(sl_pcts_used)), 3) if sl_pcts_used else 0
    return trades, avg_tp, avg_sl


def grid_search_atr(signal_bars, direction, opens, highs, lows,
                    atr_pct, n_bars, max_bars=288,
                    min_tr=25, max_baseline_wr=70.0):
    """Grid search for best ATR-mult TP/SL by PnL/MDD."""
    best = None
    best_score = -9999

    for tp_m in TP_MULT_GRID:
        for sl_m in SL_MULT_GRID:
            # Estimate baseline WR from median ATR-based TP/SL
            # Approximate: median_tp / (median_tp + median_sl)
            bwr = sl_m / (tp_m + sl_m) * 100
            if bwr > max_baseline_wr:
                continue

            trades, avg_tp, avg_sl = bt_signals_atr(
                signal_bars, direction, tp_m, sl_m,
                opens, highs, lows, atr_pct, n_bars, max_bars)

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
                    'tp_mult': tp_m, 'sl_mult': sl_m,
                    'avg_tp_pct': avg_tp, 'avg_sl_pct': avg_sl,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'pnl': round(cum, 1), 'mdd': round(mdd_val, 1),
                }
    return best


def scan_atr(signal_index, opens, highs, lows, atr_pct, n_bars,
             bar_start, bar_end, max_bars=288,
             min_trades=25, edge_threshold=21.8, mc_threshold=0.01):
    """Discover patterns using ATR-normalized TP/SL."""
    selected = []
    trade_boundary = bar_end if bar_end is not None else n_bars

    # Filter signals within range
    combos = []
    for pat_key, all_sigs in signal_index.items():
        filtered_sigs = [s for s in all_sigs if bar_start <= s < bar_end]
        if len(filtered_sigs) < 5:
            continue
        for direction in ['LONG', 'SHORT']:
            combos.append((pat_key, direction, filtered_sigs))

    for pat_key, direction, sigs in combos:
        result = grid_search_atr(
            sigs, direction, opens, highs, lows,
            atr_pct, trade_boundary, max_bars,
            min_tr=min_trades, max_baseline_wr=MAX_BASELINE_WR)

        if result is None:
            continue

        # Edge calculation
        bwr = result['sl_mult'] / (result['tp_mult'] + result['sl_mult']) * 100
        edge = result['wr'] - bwr
        if edge < edge_threshold:
            continue

        # MC test
        trades, _, _ = bt_signals_atr(
            sigs, direction, result['tp_mult'], result['sl_mult'],
            opens, highs, lows, atr_pct, trade_boundary, max_bars)
        pnls = [t[2] for t in trades]
        mc_p = mc_test(pnls)
        if mc_p >= mc_threshold:
            continue

        selected.append({
            'pattern': pat_key,
            'direction': direction,
            'tp_mult': result['tp_mult'],
            'sl_mult': result['sl_mult'],
            'avg_tp_pct': result['avg_tp_pct'],
            'avg_sl_pct': result['avg_sl_pct'],
            'trades': result['trades'],
            'wr': result['wr'],
            'edge': round(edge, 1),
            'pnl': result['pnl'],
            'mdd': result['mdd'],
            'mc_p': round(mc_p, 4),
            'baseline_wr': round(bwr, 1),
            'trades_raw': trades,
        })

    return selected


def post_filter(patterns):
    """WR>=60% + avg SL>=1.0%."""
    return [p for p in patterns
            if p['wr'] >= POST_WR_MIN and p['avg_sl_pct'] >= POST_SL_MIN]


def oos_backtest_atr(signal_index, opens, highs, lows, atr_pct,
                     patterns, oos_start, oos_end, max_bars=288):
    """OOS backtest with ATR-normalized TP/SL."""
    all_trades = []
    for p in patterns:
        oos_sigs = [s for s in signal_index[p['pattern']]
                    if oos_start <= s < oos_end]
        trades, _, _ = bt_signals_atr(
            oos_sigs, p['direction'], p['tp_mult'], p['sl_mult'],
            opens, highs, lows, atr_pct, oos_end, max_bars)
        all_trades.extend(trades)
    port = portfolio_1pos(all_trades)
    stats = calc_stats(port)
    return stats, all_trades


def main():
    t0 = time.time()

    # ============================================================
    # Phase 1: ATR Analysis
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: ATR Analysis on 270d data")
    logger.info("=" * 60)

    df = load_and_classify(DATA_270)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    types = df['candle_type'].values
    n_bars = len(opens)
    logger.info(f"Loaded {n_bars} bars ({n_bars // BARS_PER_DAY} days)")

    atr_pct = compute_atr(highs, lows, closes, ATR_PERIOD)
    signal_index = build_signal_index(types, n_bars)

    # ATR statistics
    valid_atr = atr_pct[atr_pct > 0]
    atr_stats = {
        'period': ATR_PERIOD,
        'min': round(float(np.min(valid_atr)), 4),
        'p25': round(float(np.percentile(valid_atr, 25)), 4),
        'median': round(float(np.median(valid_atr)), 4),
        'mean': round(float(np.mean(valid_atr)), 4),
        'p75': round(float(np.percentile(valid_atr, 75)), 4),
        'p95': round(float(np.percentile(valid_atr, 95)), 4),
        'max': round(float(np.max(valid_atr)), 4),
    }
    logger.info(f"ATR({ATR_PERIOD}) stats: {atr_stats}")
    logger.info(f"  Interpretation: ATR median {atr_stats['median']}% → "
                f"tp_mult=2.0 → TP={2.0*atr_stats['median']:.2f}%, "
                f"sl_mult=3.0 → SL={3.0*atr_stats['median']:.2f}%")

    # ============================================================
    # Phase 2: ATR-normalized discovery on 270d (multiple MAX_BARS)
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: ATR-Normalized Discovery (270d, multiple MAX_BARS)")
    logger.info(f"{'='*60}")
    logger.info(f"TP_MULT grid: {TP_MULT_GRID}")
    logger.info(f"SL_MULT grid: {SL_MULT_GRID}")

    discovery_results = {}

    for mb in MAX_BARS_OPTIONS:
        hours = mb / 12
        label = f"ATR_MB{mb}_{hours:.0f}h"
        logger.info(f"\n--- {label}: MAX_BARS={mb} ({hours}h) ---")

        t1 = time.time()
        raw = scan_atr(
            signal_index, opens, highs, lows, atr_pct, n_bars,
            bar_start=0, bar_end=n_bars, max_bars=mb,
            min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
            mc_threshold=MC_THRESHOLD)
        elapsed = time.time() - t1

        filtered = post_filter(raw)
        n_long = sum(1 for p in filtered if p['direction'] == 'LONG')
        n_short = sum(1 for p in filtered if p['direction'] == 'SHORT')

        logger.info(f"  {len(raw)} raw → {len(filtered)} filtered "
                     f"({n_long}L+{n_short}S) in {elapsed:.1f}s")

        # Portfolio
        if filtered:
            all_trades = []
            for p in filtered:
                all_trades.extend(p['trades_raw'])
            port = portfolio_1pos(all_trades)
            stats = calc_stats(port)

            avg_tps = [p['avg_tp_pct'] for p in filtered]
            avg_sls = [p['avg_sl_pct'] for p in filtered]
            edges = [p['edge'] for p in filtered]

            logger.info(f"  Portfolio: {stats['trades']} trades, WR {stats['wr']}%, "
                         f"PnL {stats['pnl']}%, MDD {stats['mdd']}%, PF {stats['pf']}")
            logger.info(f"  Avg TP%: {np.mean(avg_tps):.2f} ({np.min(avg_tps):.2f}~{np.max(avg_tps):.2f})")
            logger.info(f"  Avg SL%: {np.mean(avg_sls):.2f} ({np.min(avg_sls):.2f}~{np.max(avg_sls):.2f})")
            logger.info(f"  Edge: mean {np.mean(edges):.1f}pp")
        else:
            stats = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}

        discovery_results[label] = {
            'max_bars': mb,
            'hours': hours,
            'n_raw': len(raw),
            'n_filtered': len(filtered),
            'n_long': n_long,
            'n_short': n_short,
            'portfolio_stats': stats,
            'patterns': [{
                'key': f"{p['pattern']}_{p['direction']}",
                'pattern': p['pattern'],
                'direction': p['direction'],
                'tp_mult': p['tp_mult'],
                'sl_mult': p['sl_mult'],
                'avg_tp_pct': p['avg_tp_pct'],
                'avg_sl_pct': p['avg_sl_pct'],
                'trades': p['trades'],
                'wr': p['wr'],
                'edge': p['edge'],
            } for p in filtered],
            'elapsed': round(elapsed, 1),
        }

    # ============================================================
    # Phase 3: WF on 720d for best config(s)
    # ============================================================
    # Pick configs with >= 5 patterns for WF
    wf_candidates = {k: v for k, v in discovery_results.items()
                     if v['n_filtered'] >= 5}

    wf_results = {}
    if wf_candidates:
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 3: Walk-Forward on 720d")
        logger.info(f"{'='*60}")
        logger.info(f"WF candidates: {list(wf_candidates.keys())}")

        df_720 = load_and_classify(DATA_720)
        opens_720 = df_720['open'].values.astype(np.float64)
        highs_720 = df_720['high'].values.astype(np.float64)
        lows_720 = df_720['low'].values.astype(np.float64)
        closes_720 = df_720['close'].values.astype(np.float64)
        types_720 = df_720['candle_type'].values
        n_bars_720 = len(opens_720)
        atr_pct_720 = compute_atr(highs_720, lows_720, closes_720, ATR_PERIOD)
        signal_index_720 = build_signal_index(types_720, n_bars_720)

        seg_size = n_bars_720 // (N_FOLDS + 1)

        for label, config in wf_candidates.items():
            mb = config['max_bars']
            logger.info(f"\n--- WF: {label} (MAX_BARS={mb}) ---")

            folds = []
            all_stable = {}

            for fold in range(N_FOLDS):
                is_end = (fold + 1) * seg_size
                oos_start = is_end
                oos_end = (fold + 2) * seg_size if fold < N_FOLDS - 1 else n_bars_720
                is_days = is_end // BARS_PER_DAY
                oos_days = (oos_end - oos_start) // BARS_PER_DAY

                logger.info(f"  Fold {fold+1}: IS=[0,{is_days}d) OOS=[{is_days}d,{is_days+oos_days}d)")

                t1 = time.time()
                is_raw = scan_atr(
                    signal_index_720, opens_720, highs_720, lows_720,
                    atr_pct_720, n_bars_720,
                    bar_start=0, bar_end=is_end, max_bars=mb,
                    min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
                    mc_threshold=MC_THRESHOLD)
                is_filtered = post_filter(is_raw)
                t_disc = time.time() - t1

                n_long = sum(1 for p in is_filtered if p['direction'] == 'LONG')
                n_short = sum(1 for p in is_filtered if p['direction'] == 'SHORT')
                logger.info(f"    IS: {len(is_raw)} raw → {len(is_filtered)} "
                             f"({n_long}L+{n_short}S) in {t_disc:.1f}s")

                for p in is_filtered:
                    key = f"{p['pattern']}_{p['direction']}"
                    all_stable[key] = all_stable.get(key, 0) + 1

                # OOS
                if is_filtered:
                    oos_stats, _ = oos_backtest_atr(
                        signal_index_720, opens_720, highs_720, lows_720,
                        atr_pct_720, is_filtered, oos_start, oos_end, mb)
                else:
                    oos_stats = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}

                status = "PASS" if oos_stats['pnl'] > 0 else "FAIL"
                logger.info(f"    OOS: {oos_stats['trades']}t, WR {oos_stats['wr']}%, "
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
            verdict = "PASS" if n_pos >= 2 else "FAIL"

            logger.info(f"  WF Summary: {n_pos}/{N_FOLDS} PASS, OOS PnL {total_oos_pnl}%, "
                         f"avg WR {avg_oos_wr}%, verdict: {verdict}")

            wf_results[label] = {
                'folds': folds,
                'summary': {
                    'positive_folds': n_pos,
                    'total_oos_pnl': total_oos_pnl,
                    'avg_oos_wr': avg_oos_wr,
                    'stable_patterns': stable_pats[:30],
                    'verdict': verdict,
                }
            }
    else:
        logger.info("\nNo config with >=5 patterns. Skipping WF.")

    # ============================================================
    # Final Summary
    # ============================================================
    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Config':<25} {'Pat':>4} {'PnL':>8} {'WR':>6} {'MDD':>6} {'WF':>6}")
    logger.info(f"{'-'*60}")
    for label, res in discovery_results.items():
        s = res['portfolio_stats']
        wf_v = wf_results.get(label, {}).get('summary', {}).get('verdict', 'N/A')
        logger.info(f"{label:<25} {res['n_filtered']:>4} {s['pnl']:>8} "
                     f"{s['wr']:>6} {s['mdd']:>6} {wf_v:>6}")

    logger.info(f"\nTotal time: {elapsed:.1f}s")

    # Save
    output = {
        'study': 'ATR-Normalized TP/SL Discovery',
        'date': '2026-02-18',
        'atr_period': ATR_PERIOD,
        'atr_stats': atr_stats,
        'tp_mult_grid': TP_MULT_GRID,
        'sl_mult_grid': SL_MULT_GRID,
        'max_bars_options': MAX_BARS_OPTIONS,
        'parameters': {
            'edge_threshold': EDGE_THRESHOLD,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
            'post_filter': f'WR>={POST_WR_MIN}% + avg_SL>={POST_SL_MIN}%',
        },
        'discovery_270d': discovery_results,
        'wf_720d': wf_results,
        'elapsed_seconds': round(elapsed, 1),
    }

    with open(RESULTS, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved to {RESULTS}")


if __name__ == '__main__':
    main()
