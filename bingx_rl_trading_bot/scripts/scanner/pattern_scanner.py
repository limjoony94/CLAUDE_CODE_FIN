#!/usr/bin/env python3
"""
Pattern Scanner CLI — Dynamic Walk-Forward Pattern Selection

Standalone offline tool that scans historical data to select patterns
with genuine edge. Supports Universal and Per-Pattern TP/SL discovery.

Discovery methods:
- universal: Fixed TP/SL for all patterns (v1.28.0 original)
- per_pattern: Grid search optimal TP/SL per pattern (v1.28.6, default)
  PP discovery: +487% avg OOS vs Universal +18% (fair_discovery_comparison.py)

Features:
- Multi-seed MC test (3 seeds, conservative max p-value)
- MAX_BASELINE_WR filter to exclude distance-dominated combos
- 1-position-at-a-time portfolio constraint

Usage:
  python scripts/scanner/pattern_scanner.py                              # PP discovery (default)
  python scripts/scanner/pattern_scanner.py --discovery-method universal  # Universal mode
  python scripts/scanner/pattern_scanner.py --data data/custom.csv -v
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

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
MAX_BARS = 500          # Max bars to hold trade before timeout
MC_SIMS = 5000          # Monte Carlo simulations
DEFAULT_UNI_TP = 2.0    # Universal TP % (v1.28.2: WF frontier optimal)
DEFAULT_UNI_SL = 3.0    # Universal SL %
DEFAULT_EDGE_THRESHOLD = 10.0  # Minimum edge in pp (v1.28.4: 5→10 for statistical rigor)
DEFAULT_MC_THRESHOLD = 0.01    # MC p-value cutoff
DEFAULT_MIN_TRADES = 25        # v1.28.6: 20→25 (scanner_param_study: OOS PnL/MDD 2.38 best)

# Per-pattern discovery grid (v1.28.6)
TP_GRID = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.5, 3.0]
SL_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
MAX_BASELINE_WR = 70.0  # Skip combos where distance effect dominates (R:R >= 0.43)
MC_SEEDS = [42, 123, 7777]  # Multi-seed for robustness

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
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()

    # Classify each candle using production function
    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, df.at[i, 'avg_body'])
        types.append(ct.value)

    df['candle_type'] = types
    logger.info(f"Classified {len(df)} candles into 12 types")
    return df


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
    Backtest given signal bars with Universal TP/SL.

    Entry: next bar open after signal.
    Exit: intrabar high/low check (distance-based).
    Timeout trades are DROPPED (not counted).
    """
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
                # Both hit same bar — bar open 기준 distance-based resolution
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (tp_pct if dist_tp <= dist_sl else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE_PCT))
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT))
                break
        # Timeout trades are dropped (no append)

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


def grid_search_best(signal_bars, direction, opens, highs, lows, n_bars, min_tr=20):
    """Grid search for best TP/SL by PnL/MDD. Skips combos with baseline WR > MAX_BASELINE_WR."""
    best = None
    best_score = -9999
    for tp in TP_GRID:
        for sl in SL_GRID:
            if sl < 0.5:
                continue
            bwr = sl / (tp + sl) * 100
            if bwr > MAX_BASELINE_WR:
                continue
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


def portfolio_1pos(all_trades):
    """1-position-at-a-time filter: sort by entry, skip overlapping."""
    if not all_trades:
        return []
    all_trades.sort(key=lambda x: x[0])
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
# Main Scanner Logic
# ============================================================

def scan_patterns(
    df: pd.DataFrame,
    uni_tp: float = DEFAULT_UNI_TP,
    uni_sl: float = DEFAULT_UNI_SL,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    mc_threshold: float = DEFAULT_MC_THRESHOLD,
    min_trades: int = DEFAULT_MIN_TRADES,
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

    # Build signal index for all patterns
    signal_index = build_signal_index(types, n)
    logger.info(f"Built signal index: {len(signal_index)} unique patterns found")

    # Baseline WR for random walk
    baseline_wr = uni_sl / (uni_tp + uni_sl) * 100

    # Scan all patterns x directions
    selected = {}
    pattern_details = []

    for pat_name in signal_index:
        for direction in ['LONG', 'SHORT']:
            sigs = signal_index[pat_name]
            if len(sigs) < min_trades:
                continue

            trades = bt_signals(sigs, direction, uni_tp, uni_sl, opens, highs, lows, n)
            if len(trades) < min_trades:
                continue

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

    logger.info(f"Patterns passing filters: {len(selected)}")

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

    # Organize by direction
    long_patterns = sorted([v['pattern'] for v in selected.values() if v['direction'] == 'LONG'])
    short_patterns = sorted([v['pattern'] for v in selected.values() if v['direction'] == 'SHORT'])

    logger.info(f"Selected: {len(long_patterns)}L + {len(short_patterns)}S = {len(selected)} patterns")
    logger.info(f"Portfolio: {portfolio_stats['trades']} trades, WR {portfolio_stats['wr']}%, PnL {portfolio_stats['pnl']}%")

    return {
        'long_patterns': long_patterns,
        'short_patterns': short_patterns,
        'portfolio_stats': portfolio_stats,
        'portfolio_mc': round(portfolio_mc, 4),
        'pattern_details': {k: {kk: vv for kk, vv in v.items()} for k, v in selected.items()},
    }


def scan_patterns_pp(
    df: pd.DataFrame,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    mc_threshold: float = DEFAULT_MC_THRESHOLD,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> dict:
    """Per-Pattern Discovery: grid search optimal TP/SL per pattern → edge+MC filter."""
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    signal_index = build_signal_index(types, n)
    logger.info(f"Built signal index: {len(signal_index)} unique patterns found")

    selected = {}
    pattern_details = []

    for pat_name in signal_index:
        for direction in ['LONG', 'SHORT']:
            sigs = signal_index[pat_name]
            if len(sigs) < min_trades:
                continue

            opt = grid_search_best(sigs, direction, opens, highs, lows, n, min_tr=min_trades)
            if opt is None:
                continue

            tp, sl = opt['tp'], opt['sl']
            # Re-backtest with optimal TP/SL
            trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n)
            if len(trades) < min_trades:
                continue

            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            baseline_wr = sl / (tp + sl) * 100
            edge = wr - baseline_wr

            if edge < edge_threshold:
                continue

            p = mc_test(pnls)
            if p >= mc_threshold:
                continue

            key = f"{pat_name}_{direction}"
            selected[key] = {
                'pattern': pat_name, 'direction': direction,
                'tp': tp, 'sl': sl,
                'trades': len(trades), 'wr': round(wr, 1),
                'edge': round(edge, 1), 'mc_p': round(p, 4),
                'baseline_wr': round(baseline_wr, 1),
            }
            pattern_details.append({'key': key, 'trades_raw': trades})

    logger.info(f"Patterns passing filters: {len(selected)}")

    # Portfolio 1-pos filter + stats
    all_trades = []
    for pd_item in pattern_details:
        all_trades.extend(pd_item['trades_raw'])
    portfolio_trades = portfolio_1pos(all_trades)
    portfolio_stats = calc_stats(portfolio_trades)
    portfolio_mc = mc_test([t[2] for t in portfolio_trades]) if portfolio_trades else 1.0

    # Organize by direction
    long_patterns = sorted([v['pattern'] for v in selected.values() if v['direction'] == 'LONG'])
    short_patterns = sorted([v['pattern'] for v in selected.values() if v['direction'] == 'SHORT'])

    # Build patterns_tpsl dict
    patterns_tpsl = {}
    for v in selected.values():
        patterns_tpsl[v['pattern']] = [v['tp'], v['sl']]

    # TP/SL distribution
    tps = [v['tp'] for v in selected.values()]
    sls = [v['sl'] for v in selected.values()]

    logger.info(f"Selected: {len(long_patterns)}L + {len(short_patterns)}S = {len(selected)} patterns")
    logger.info(f"Portfolio: {portfolio_stats['trades']} trades, WR {portfolio_stats['wr']}%, PnL {portfolio_stats['pnl']}%")

    return {
        'long_patterns': long_patterns,
        'short_patterns': short_patterns,
        'patterns_tpsl': patterns_tpsl,
        'portfolio_stats': portfolio_stats,
        'portfolio_mc': round(portfolio_mc, 4),
        'pattern_details': {k: v for k, v in selected.items()},
        'tp_distribution': {'min': min(tps), 'median': round(float(np.median(tps)), 1),
                            'mean': round(float(np.mean(tps)), 1), 'max': max(tps)} if tps else {},
        'sl_distribution': {'min': min(sls), 'median': round(float(np.median(sls)), 1),
                            'mean': round(float(np.mean(sls)), 1), 'max': max(sls)} if sls else {},
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
) -> dict:
    """Build the output JSON structure. Supports both universal and per_pattern modes."""
    output = {
        'version': '2.0',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'data_file': os.path.basename(data_file),
        'data_bars': data_bars,
        'selection_criteria': {
            'discovery_method': discovery_method,
            'edge_threshold_pp': edge_threshold,
            'mc_threshold': mc_threshold,
            'min_trades': min_trades,
            'max_baseline_wr': MAX_BASELINE_WR if discovery_method == 'per_pattern' else None,
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
        },
        'pattern_details': scan_result['pattern_details'],
    }

    if discovery_method == 'universal':
        output['tp_sl_mode'] = 'universal'
        output['universal_tp'] = uni_tp
        output['universal_sl'] = uni_sl
    elif discovery_method == 'per_pattern':
        output['tp_sl_mode'] = 'per_pattern'
        output['patterns_tpsl'] = scan_result['patterns_tpsl']
        output['tp_distribution'] = scan_result['tp_distribution']
        output['sl_distribution'] = scan_result['sl_distribution']

    return output


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pattern Scanner — Dynamic Walk-Forward Pattern Selection'
    )
    parser.add_argument('--data', default=DEFAULT_DATA_FILE,
                        help='Path to OHLCV CSV file')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_FILE,
                        help='Output JSON file path')
    parser.add_argument('--discovery-method', choices=['universal', 'per_pattern'],
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
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    logger.info("=" * 60)
    logger.info("Pattern Scanner — Dynamic Walk-Forward Selection")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
    logger.info(f"Discovery: {args.discovery_method}")
    if args.discovery_method == 'universal':
        logger.info(f"TP: {args.tp}% | SL: {args.sl}% | Edge >= {args.edge_threshold}pp | MC < {args.mc_threshold}")
    else:
        logger.info(f"Grid: TP {TP_GRID} | SL {SL_GRID} | Max baseline WR: {MAX_BASELINE_WR}%")
        logger.info(f"Edge >= {args.edge_threshold}pp | MC < {args.mc_threshold} (3-seed)")

    # Load and classify
    df = load_and_classify(args.data)

    # Scan patterns
    if args.discovery_method == 'universal':
        result = scan_patterns(
            df,
            uni_tp=args.tp,
            uni_sl=args.sl,
            edge_threshold=args.edge_threshold,
            mc_threshold=args.mc_threshold,
            min_trades=args.min_trades,
        )
    else:
        result = scan_patterns_pp(
            df,
            edge_threshold=args.edge_threshold,
            mc_threshold=args.mc_threshold,
            min_trades=args.min_trades,
        )

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
    )

    # Write JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Output written to {args.output}")
    logger.info("=" * 60)

    # Print summary
    bs = output['backtest_summary']
    pc = output['pattern_count']
    print(f"\nScan Complete ({args.discovery_method}):")
    print(f"  Patterns: {pc['long']}L + {pc['short']}S = {pc['long'] + pc['short']}")
    print(f"  Trades: {bs['total_trades']} | WR: {bs['win_rate']}% | PnL: {bs['pnl_pct']}%")
    print(f"  MDD: {bs['max_drawdown_pct']}% | PF: {bs['profit_factor']} | MC p: {bs['mc_pvalue']}")
    if args.discovery_method == 'per_pattern' and 'tp_distribution' in output:
        tp_d = output['tp_distribution']
        sl_d = output['sl_distribution']
        print(f"  TP: min={tp_d['min']} med={tp_d['median']} max={tp_d['max']}")
        print(f"  SL: min={sl_d['min']} med={sl_d['median']} max={sl_d['max']}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
