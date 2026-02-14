#!/usr/bin/env python3
"""
TP 0.80 / SL 1.20 Comprehensive Validation Study
=================================================
Validates the new tight TP/SL configuration with:

1. Walk-Forward (3-fold): Train → discover patterns → test OOS
2. Slippage Sensitivity: 0.00% ~ 0.10% in 0.01% steps
3. MC Drawdown Simulation: 10k bootstrap runs for MDD distribution
4. Quarterly Edge Stability: WR & edge by quarter
5. Comparison vs TP 2.10/SL 3.00 (previous)

Uses production classify_candle() — no self-classification.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10          # 0.05% entry + 0.05% exit
MAX_BARS = 500
MC_SIMS = 5000
EDGE_THRESHOLD = 5.0
MC_THRESHOLD = 0.01
MIN_TRADES = 10

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'tpsl_080_120_validation.json')

# TP/SL configs to compare
CONFIGS = {
    'NEW_080_120': {'tp': 0.80, 'sl': 1.20},
    'OLD_210_300': {'tp': 2.10, 'sl': 3.00},
}


# ============================================================
# Data Loading
# ============================================================
def load_and_classify(data_file):
    """Load CSV and classify candles using production classify_candle()."""
    print(f"Loading data from {os.path.basename(data_file)}...")
    df = pd.read_csv(data_file)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()

    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, df.at[i, 'avg_body'])
        types.append(ct.value)
    df['candle_type'] = types
    print(f"  Classified {len(df)} candles")
    return df


# ============================================================
# Core Backtest Functions
# ============================================================
def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars,
               extra_fee=0.0):
    """Backtest with optional extra slippage fee."""
    total_fee = FEE_PCT + extra_fee
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
                pnl = (tp_pct if dist_tp <= dist_sl else -sl_pct) * LEVERAGE - total_fee
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - total_fee))
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - total_fee))
                break
        # Timeout trades dropped

    return trades


def mc_test(pnls, n_sims=MC_SIMS):
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    rand_sums = signs @ pnls_arr
    return float(np.mean(rand_sums >= actual))


def portfolio_1pos(all_trades):
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
        'pnl': round(cum, 2),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 2),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
    }


def compound_stats(trades):
    """Compound (multiplicative) returns."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    equity = 100.0
    peak = 100.0
    mdd = 0
    wins = 0
    losses_n = 0
    for _, _, pnl_pct in trades:
        equity *= (1 + pnl_pct / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
        if pnl_pct > 0:
            wins += 1
        else:
            losses_n += 1
    total = wins + losses_n
    return {
        'pnl': round(equity - 100, 2),
        'trades': total,
        'wr': round(wins / total * 100, 1) if total else 0,
        'mdd': round(mdd, 2),
    }


def scan_patterns_full(df, tp, sl, extra_fee=0.0, start_idx=0, end_idx=None):
    """Scan all patterns on a data slice. Returns (selected_patterns, all_portfolio_trades)."""
    types = df['candle_type'].tolist()
    n = len(types) if end_idx is None else end_idx
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    signal_index = build_signal_index(types, n)
    baseline_wr = sl / (tp + sl) * 100

    selected = {}
    all_raw_trades = []

    for pat_name in signal_index:
        for direction in ['LONG', 'SHORT']:
            sigs = [s for s in signal_index[pat_name] if s >= start_idx and s < n]
            if len(sigs) < MIN_TRADES:
                continue
            trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n,
                                extra_fee=extra_fee)
            if len(trades) < MIN_TRADES:
                continue

            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            edge = wr - baseline_wr

            if edge < EDGE_THRESHOLD:
                continue

            p = mc_test(pnls)
            if p >= MC_THRESHOLD:
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
            all_raw_trades.extend(trades)

    portfolio = portfolio_1pos(all_raw_trades)
    return selected, portfolio


def scan_with_predefined(df, patterns_dict, tp, sl, start_idx=0, end_idx=None, extra_fee=0.0):
    """Backtest predefined patterns on a data slice (for OOS testing)."""
    types = df['candle_type'].tolist()
    n = len(types) if end_idx is None else end_idx
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    signal_index = build_signal_index(types, n)
    all_raw_trades = []

    for key, info in patterns_dict.items():
        pat_name = info['pattern']
        direction = info['direction']
        if pat_name not in signal_index:
            continue
        sigs = [s for s in signal_index[pat_name] if s >= start_idx and s < n]
        if not sigs:
            continue
        trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n,
                            extra_fee=extra_fee)
        all_raw_trades.extend(trades)

    portfolio = portfolio_1pos(all_raw_trades)
    return portfolio


# ============================================================
# Phase 1: Walk-Forward Validation (3-fold)
# ============================================================
def run_walkforward(df, tp, sl, n_folds=3):
    """Split data into folds, train on (n-1) folds, test on held-out fold."""
    n = len(df)
    fold_size = n // n_folds
    results = []

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION ({n_folds}-fold) — TP {tp}% / SL {sl}%")
    print(f"{'='*60}")

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

        # Train: everything except test fold
        # Discover patterns on train data
        train_types = df['candle_type'].tolist()
        train_n = len(train_types)
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values

        signal_index = build_signal_index(train_types, train_n)
        baseline_wr = sl / (tp + sl) * 100

        # Train: scan on non-test bars
        selected = {}
        for pat_name in signal_index:
            for direction in ['LONG', 'SHORT']:
                train_sigs = [s for s in signal_index[pat_name]
                              if s < test_start or s >= test_end]
                if len(train_sigs) < MIN_TRADES:
                    continue
                trades = bt_signals(train_sigs, direction, tp, sl, opens, highs, lows, train_n)
                if len(trades) < MIN_TRADES:
                    continue

                pnls = [t[2] for t in trades]
                wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                edge = wr - baseline_wr
                if edge < EDGE_THRESHOLD:
                    continue
                p = mc_test(pnls)
                if p >= MC_THRESHOLD:
                    continue
                key = f"{pat_name}_{direction}"
                selected[key] = {'pattern': pat_name, 'direction': direction,
                                 'wr': round(wr, 1), 'edge': round(edge, 1)}

        # Test: apply discovered patterns to test fold only
        test_trades = []
        for key, info in selected.items():
            pat_name = info['pattern']
            direction = info['direction']
            if pat_name not in signal_index:
                continue
            test_sigs = [s for s in signal_index[pat_name]
                         if s >= test_start and s < test_end]
            if not test_sigs:
                continue
            trades = bt_signals(test_sigs, direction, tp, sl, opens, highs, lows, train_n)
            test_trades.extend(trades)

        test_portfolio = portfolio_1pos(test_trades)
        test_stats = compound_stats(test_portfolio)

        # Train stats (for comparison)
        train_all_trades = []
        for key, info in selected.items():
            pat_name = info['pattern']
            direction = info['direction']
            if pat_name not in signal_index:
                continue
            train_sigs = [s for s in signal_index[pat_name]
                          if s < test_start or s >= test_end]
            trades = bt_signals(train_sigs, direction, tp, sl, opens, highs, lows, train_n)
            train_all_trades.extend(trades)
        train_portfolio = portfolio_1pos(train_all_trades)
        train_stats = compound_stats(train_portfolio)

        fold_result = {
            'fold': fold + 1,
            'test_range': f"bar {test_start}-{test_end}",
            'patterns_discovered': len(selected),
            'train': train_stats,
            'test': test_stats,
            'wr_drop': round(train_stats['wr'] - test_stats['wr'], 1),
        }
        results.append(fold_result)

        print(f"\nFold {fold+1}: Test bars {test_start}-{test_end}")
        print(f"  Discovered: {len(selected)} patterns")
        print(f"  Train: {train_stats['trades']} trades, WR {train_stats['wr']}%, PnL {train_stats['pnl']}%")
        print(f"  Test:  {test_stats['trades']} trades, WR {test_stats['wr']}%, PnL {test_stats['pnl']}%")
        print(f"  WR drop: {fold_result['wr_drop']}pp")

    # Summary
    avg_test_wr = np.mean([r['test']['wr'] for r in results])
    avg_test_pnl = np.mean([r['test']['pnl'] for r in results])
    avg_wr_drop = np.mean([r['wr_drop'] for r in results])
    oos_positive = sum(1 for r in results if r['test']['pnl'] > 0)

    summary = {
        'folds': results,
        'avg_test_wr': round(avg_test_wr, 1),
        'avg_test_pnl': round(avg_test_pnl, 1),
        'avg_wr_drop': round(avg_wr_drop, 1),
        'oos_positive_folds': f"{oos_positive}/{n_folds}",
    }

    print(f"\n--- WF Summary ---")
    print(f"  Avg OOS WR: {avg_test_wr:.1f}%")
    print(f"  Avg OOS PnL: {avg_test_pnl:.1f}%")
    print(f"  Avg WR drop: {avg_wr_drop:.1f}pp")
    print(f"  OOS positive: {oos_positive}/{n_folds}")

    return summary


# ============================================================
# Phase 2: Slippage Sensitivity
# ============================================================
def run_slippage_sensitivity(df, tp, sl):
    """Test with various slippage levels."""
    print(f"\n{'='*60}")
    print(f"SLIPPAGE SENSITIVITY — TP {tp}% / SL {sl}%")
    print(f"{'='*60}")

    slippage_levels = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    results = []

    for slip in slippage_levels:
        extra_fee = slip * 2  # entry + exit slippage
        selected, portfolio = scan_patterns_full(df, tp, sl, extra_fee=extra_fee)
        stats = compound_stats(portfolio)
        simple_stats = calc_stats(portfolio)

        result = {
            'slippage_pct': slip,
            'extra_fee_pct': round(extra_fee, 3),
            'patterns': len(selected),
            'trades': stats['trades'],
            'wr': stats['wr'],
            'pnl_compound': stats['pnl'],
            'mdd': stats['mdd'],
            'pf': simple_stats['pf'],
        }
        results.append(result)

        print(f"  Slip {slip:.2f}%: {len(selected)} pat, {stats['trades']} trades, "
              f"WR {stats['wr']}%, PnL {stats['pnl']:.1f}%, MDD {stats['mdd']:.1f}%")

    # Calculate degradation rate
    base_pnl = results[0]['pnl_compound']
    for r in results:
        r['pnl_degradation_pct'] = round((r['pnl_compound'] - base_pnl) / abs(base_pnl) * 100, 1) if base_pnl != 0 else 0

    # Break-even slippage (where PnL turns negative)
    be_slip = None
    for r in results:
        if r['pnl_compound'] <= 0:
            be_slip = r['slippage_pct']
            break

    summary = {
        'levels': results,
        'breakeven_slippage': be_slip,
        'pnl_at_007': next((r['pnl_compound'] for r in results if r['slippage_pct'] == 0.07), None),
    }

    print(f"\n  Break-even slippage: {be_slip if be_slip else '>0.10%'}")
    print(f"  PnL at realistic 0.07%: {summary['pnl_at_007']:.1f}%")

    return summary


# ============================================================
# Phase 3: MC Drawdown Simulation
# ============================================================
def run_mc_drawdown(df, tp, sl, n_sims=10000):
    """Bootstrap MC simulation for MDD distribution."""
    print(f"\n{'='*60}")
    print(f"MC DRAWDOWN SIMULATION ({n_sims} runs) — TP {tp}% / SL {sl}%")
    print(f"{'='*60}")

    # Get actual trade PnLs
    selected, portfolio = scan_patterns_full(df, tp, sl)
    pnls = np.array([t[2] for t in portfolio])
    n_trades = len(pnls)

    if n_trades < 10:
        print("  Insufficient trades for MC simulation")
        return {'error': 'insufficient_trades'}

    print(f"  Base: {n_trades} trades, WR {np.mean(pnls > 0)*100:.1f}%")

    # Bootstrap: resample trades with replacement
    np.random.seed(42)
    mdds = []
    final_pnls = []

    for _ in range(n_sims):
        # Resample trade PnLs
        sample = np.random.choice(pnls, size=n_trades, replace=True)
        # Compound equity
        equity = 100.0
        peak = 100.0
        max_dd = 0
        for p in sample:
            equity *= (1 + p / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        mdds.append(max_dd)
        final_pnls.append(equity - 100)

    mdds = np.array(mdds)
    final_pnls = np.array(final_pnls)

    result = {
        'n_sims': n_sims,
        'n_trades': n_trades,
        'mdd_mean': round(float(np.mean(mdds)), 1),
        'mdd_median': round(float(np.median(mdds)), 1),
        'mdd_p90': round(float(np.percentile(mdds, 90)), 1),
        'mdd_p95': round(float(np.percentile(mdds, 95)), 1),
        'mdd_p99': round(float(np.percentile(mdds, 99)), 1),
        'mdd_max': round(float(np.max(mdds)), 1),
        'pnl_mean': round(float(np.mean(final_pnls)), 1),
        'pnl_median': round(float(np.median(final_pnls)), 1),
        'pnl_p5': round(float(np.percentile(final_pnls, 5)), 1),
        'pnl_p10': round(float(np.percentile(final_pnls, 10)), 1),
        'loss_probability': round(float(np.mean(final_pnls < 0)) * 100, 1),
    }

    print(f"  MDD: mean {result['mdd_mean']}%, median {result['mdd_median']}%")
    print(f"  MDD: P90 {result['mdd_p90']}%, P95 {result['mdd_p95']}%, P99 {result['mdd_p99']}%")
    print(f"  PnL: mean {result['pnl_mean']}%, median {result['pnl_median']}%")
    print(f"  PnL: P5 {result['pnl_p5']}%, P10 {result['pnl_p10']}%")
    print(f"  Loss probability: {result['loss_probability']}%")

    return result


# ============================================================
# Phase 4: Quarterly Edge Stability
# ============================================================
def run_quarterly_stability(df, tp, sl):
    """Analyze edge stability across quarters."""
    print(f"\n{'='*60}")
    print(f"QUARTERLY EDGE STABILITY — TP {tp}% / SL {sl}%")
    print(f"{'='*60}")

    # Discover patterns on full data
    selected, portfolio = scan_patterns_full(df, tp, sl)

    # Always use bar index for quarterly split (simpler, no timestamp parsing issues)
    n = len(df)
    q_size = n // 4
    quarters = []
    for q in range(4):
        q_start = q * q_size
        q_end = (q + 1) * q_size if q < 3 else n
        quarters.append((f"Q{q+1}", q_start, q_end))

    if True:  # Always execute quarterly analysis

        results = []
        baseline_wr = sl / (tp + sl) * 100

        for q_name, q_start, q_end in quarters:
            # Apply discovered patterns to this quarter
            q_trades = []
            types = df['candle_type'].tolist()
            n_total = len(types)
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            signal_index = build_signal_index(types, n_total)

            for key, info in selected.items():
                pat_name = info['pattern']
                direction = info['direction']
                if pat_name not in signal_index:
                    continue
                sigs = [s for s in signal_index[pat_name] if s >= q_start and s < q_end]
                if not sigs:
                    continue
                trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n_total)
                q_trades.extend(trades)

            q_portfolio = portfolio_1pos(q_trades)
            q_stats = compound_stats(q_portfolio)

            # Calculate WR and edge for this quarter
            q_pnls = [t[2] for t in q_portfolio]
            q_wr = len([p for p in q_pnls if p > 0]) / len(q_pnls) * 100 if q_pnls else 0
            q_edge = q_wr - baseline_wr

            result = {
                'quarter': q_name,
                'bars': f"{q_start}-{q_end}",
                'trades': q_stats['trades'],
                'wr': round(q_wr, 1),
                'edge': round(q_edge, 1),
                'pnl': q_stats['pnl'],
                'mdd': q_stats['mdd'],
            }
            results.append(result)

            print(f"  {q_name}: {q_stats['trades']} trades, WR {q_wr:.1f}%, "
                  f"Edge {q_edge:.1f}pp, PnL {q_stats['pnl']:.1f}%, MDD {q_stats['mdd']:.1f}%")

        # Stability metrics
        edges = [r['edge'] for r in results if r['trades'] > 0]
        wrs = [r['wr'] for r in results if r['trades'] > 0]
        positive_q = sum(1 for r in results if r['pnl'] > 0)

        summary = {
            'quarters': results,
            'edge_mean': round(np.mean(edges), 1) if edges else 0,
            'edge_std': round(np.std(edges), 1) if edges else 0,
            'edge_min': round(min(edges), 1) if edges else 0,
            'wr_mean': round(np.mean(wrs), 1) if wrs else 0,
            'wr_std': round(np.std(wrs), 1) if wrs else 0,
            'positive_quarters': f"{positive_q}/{len(results)}",
        }

        print(f"\n  Edge: mean {summary['edge_mean']}pp, std {summary['edge_std']}pp, min {summary['edge_min']}pp")
        print(f"  WR: mean {summary['wr_mean']}%, std {summary['wr_std']}%")
        print(f"  Positive quarters: {positive_q}/{len(results)}")

        return summary


# ============================================================
# Phase 5: Head-to-Head Comparison
# ============================================================
def run_comparison(df):
    """Compare NEW 0.80/1.20 vs OLD 2.10/3.00."""
    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD COMPARISON")
    print(f"{'='*60}")

    results = {}
    for name, cfg in CONFIGS.items():
        tp, sl = cfg['tp'], cfg['sl']
        selected, portfolio = scan_patterns_full(df, tp, sl)
        stats = compound_stats(portfolio)
        simple = calc_stats(portfolio)

        # Per-trade expected value
        avg_win = tp * LEVERAGE - FEE_PCT
        avg_loss = sl * LEVERAGE + FEE_PCT
        expected = stats['wr'] / 100 * avg_win - (1 - stats['wr'] / 100) * avg_loss

        # Slippage impact at 0.07%
        _, portfolio_slip = scan_patterns_full(df, tp, sl, extra_fee=0.07 * 2)
        stats_slip = compound_stats(portfolio_slip)

        long_count = sum(1 for v in selected.values() if v['direction'] == 'LONG')
        short_count = sum(1 for v in selected.values() if v['direction'] == 'SHORT')

        results[name] = {
            'tp': tp,
            'sl': sl,
            'patterns': len(selected),
            'long': long_count,
            'short': short_count,
            'trades': stats['trades'],
            'wr': stats['wr'],
            'pnl_compound': stats['pnl'],
            'mdd': stats['mdd'],
            'pnl_mdd': round(stats['pnl'] / stats['mdd'], 1) if stats['mdd'] > 0 else 999,
            'pf': simple['pf'],
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expected_per_trade': round(expected, 3),
            'pnl_with_slippage': stats_slip['pnl'],
            'slippage_impact_pct': round((stats_slip['pnl'] - stats['pnl']) / abs(stats['pnl']) * 100, 1) if stats['pnl'] != 0 else 0,
        }

        print(f"\n  {name}: TP {tp}% / SL {sl}%")
        print(f"    Patterns: {long_count}L + {short_count}S = {len(selected)}")
        print(f"    Trades: {stats['trades']} | WR: {stats['wr']}%")
        print(f"    PnL: {stats['pnl']:.1f}% | MDD: {stats['mdd']:.1f}% | PnL/MDD: {results[name]['pnl_mdd']}x")
        print(f"    PF: {simple['pf']} | Expected/trade: {expected:.3f}%")
        print(f"    With 0.07% slippage: PnL {stats_slip['pnl']:.1f}% (impact: {results[name]['slippage_impact_pct']:.1f}%)")

    return results


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("TP 0.80 / SL 1.20 — COMPREHENSIVE VALIDATION")
    print("=" * 60)

    df = load_and_classify(DATA_FILE)

    output = {}

    # Phase 1: Walk-Forward
    output['walkforward_new'] = run_walkforward(df, 0.80, 1.20)
    output['walkforward_old'] = run_walkforward(df, 2.10, 3.00)

    # Phase 2: Slippage Sensitivity
    output['slippage_new'] = run_slippage_sensitivity(df, 0.80, 1.20)
    output['slippage_old'] = run_slippage_sensitivity(df, 2.10, 3.00)

    # Phase 3: MC Drawdown
    output['mc_drawdown_new'] = run_mc_drawdown(df, 0.80, 1.20)
    output['mc_drawdown_old'] = run_mc_drawdown(df, 2.10, 3.00)

    # Phase 4: Quarterly Stability
    output['quarterly_new'] = run_quarterly_stability(df, 0.80, 1.20)
    output['quarterly_old'] = run_quarterly_stability(df, 2.10, 3.00)

    # Phase 5: Head-to-Head
    output['comparison'] = run_comparison(df)

    # Save
    elapsed = time.time() - t0
    output['meta'] = {
        'data_file': os.path.basename(DATA_FILE),
        'data_bars': len(df),
        'elapsed_seconds': round(elapsed, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE — {elapsed:.1f}s")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
