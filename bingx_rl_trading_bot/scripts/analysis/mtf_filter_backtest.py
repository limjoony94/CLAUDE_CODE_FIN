#!/usr/bin/env python3
"""
MTF Filter Backtest — Phase 3 & 4
==================================
Multi-Timeframe Direction Filter: Backtest + Tight TP Optimization

Phase 3: Compare baseline (no filter) vs MTF-filtered signals
Phase 4: Grid search tight TP/SL with MTF filter

Depends on Phase 1-2 results: results/mtf_direction_study.json

Standard Research Protocol:
- Production classify_candle import
- LEVERAGE=3, FEE=0.10% (FEE_TOTAL=0.30%)
- Entry at next bar open
- Intrabar TP/SL check (distance-based same-bar resolution)
- Timeout 288 bars (24h) = DROP
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Production imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW
from scripts.analysis.mtf_direction_study import (
    load_5m_data, aggregate_to_higher_tf, classify_dataframe,
    merge_1h_pattern_onto_5m, measure_directional_accuracy,
    analyze_directional_results
)

# ============================================================
# Constants
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10  # 0.05% * 2 sides
MAX_BARS = 288   # 24h timeout at 5m
FEE_TOTAL = FEE_PCT * LEVERAGE  # 0.30% capital-space

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

# Phase 4 grid
TP_GRID = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
SL_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]


# ============================================================
# Data Preparation
# ============================================================

def prepare_data():
    """Load 5m data, classify, merge 1h directional patterns."""
    # Load raw 5m
    df_5m = load_5m_data()

    # Classify 5m candles (production classify_candle)
    print("[Data] Classifying 5m candles...")
    df_5m = classify_dataframe(df_5m)
    n_5m_pats = df_5m['pattern_3'].notna().sum()
    print(f"[Data] 5m patterns: {n_5m_pats:,} valid")

    # Load pre-classified 1h (from Phase 1) or create
    df_1h_file = DATA_DIR / 'btc_1h_720days.csv'
    if df_1h_file.exists():
        print("[Data] Loading pre-classified 1h...")
        df_1h = pd.read_csv(df_1h_file, parse_dates=['timestamp'])
    else:
        print("[Data] Aggregating 5m → 1h + classifying...")
        df_1h = aggregate_to_higher_tf(df_5m, 60)
        df_1h = classify_dataframe(df_1h)

    # Merge 1h pattern onto 5m
    df = merge_1h_pattern_onto_5m(df_5m, df_1h)
    return df


def load_production_patterns():
    """Load 35 production patterns from dynamic_patterns.json."""
    dp_file = RESULTS_DIR / 'dynamic_patterns.json'
    with open(dp_file) as f:
        dp = json.load(f)

    patterns = {}
    for key, info in dp['pattern_details'].items():
        patterns[key] = {
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': info['tp'],
            'sl': info['sl'],
        }

    print(f"[Data] Loaded {len(patterns)} production patterns "
          f"(TP {min(p['tp'] for p in patterns.values()):.2f}"
          f"-{max(p['tp'] for p in patterns.values()):.2f}%, "
          f"SL {min(p['sl'] for p in patterns.values()):.2f}"
          f"-{max(p['sl'] for p in patterns.values()):.2f}%)")
    return patterns


def load_strong_1h_patterns():
    """Load strong 1h directional patterns from Phase 2 results."""
    result_file = RESULTS_DIR / 'mtf_direction_study.json'
    with open(result_file) as f:
        data = json.load(f)

    strong = {}
    for pat, info in data['phase_2']['unique_strong_detail'].items():
        strong[pat] = info['direction']

    n_long = sum(1 for d in strong.values() if d == 'LONG')
    n_short = sum(1 for d in strong.values() if d == 'SHORT')
    print(f"[Data] Loaded {len(strong)} strong 1h patterns "
          f"({n_long}L + {n_short}S)")
    return strong


def build_expanded_1h_patterns(df_1h, min_occ=30, horizons=None):
    """Build 1h directional pattern sets at multiple accuracy thresholds.

    Returns:
        tiers: {threshold_pct: {pattern: direction}}
        best_per_pattern: {pattern: {accuracy, direction, horizon, ...}}
    """
    if horizons is None:
        horizons = [1, 3, 6, 12, 24]

    print(f"\n[Expanded] Computing ALL 1h pattern directions "
          f"(min_occ={min_occ})...")
    dir_results = measure_directional_accuracy(df_1h, horizons)

    # Analyze with very relaxed thresholds: acc>50%, p<1.0 (accept all)
    all_analysis = analyze_directional_results(
        dir_results, min_occ, 0.50, 1.0)

    # Best direction per pattern (highest accuracy across horizons)
    best_per_pattern = {}
    for a in all_analysis:
        pat = a['pattern']
        if pat not in best_per_pattern or a['accuracy'] > best_per_pattern[pat]['accuracy']:
            best_per_pattern[pat] = a

    print(f"[Expanded] Total patterns with N >= {min_occ}: "
          f"{len(best_per_pattern)}")

    # Create tiers at different accuracy thresholds
    thresholds = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0]
    tiers = {}
    for thr in thresholds:
        tier = {}
        for pat, info in best_per_pattern.items():
            if info['accuracy'] >= thr:
                tier[pat] = info['direction']
        tiers[thr] = tier
        n_l = sum(1 for d in tier.values() if d == 'LONG')
        n_s = sum(1 for d in tier.values() if d == 'SHORT')
        print(f"  acc >= {thr:.0f}%: {len(tier)} patterns ({n_l}L + {n_s}S)")

    return tiers, best_per_pattern


# ============================================================
# Signal Generation
# ============================================================

def build_signal_index(df, production_patterns):
    """Find all bars where a production 5m pattern matches.

    Returns list of (bar_idx, pattern_key, direction, tp, sl).
    """
    patterns_5m = df['pattern_3'].values

    # Build lookup: pattern_name → list of (key, direction, tp, sl)
    lookup = {}
    for key, info in production_patterns.items():
        pat = info['pattern']
        if pat not in lookup:
            lookup[pat] = []
        lookup[pat].append((key, info['direction'], info['tp'], info['sl']))

    signals = []
    for i in range(len(patterns_5m)):
        p = patterns_5m[i]
        if p and not pd.isna(p) and p in lookup:
            for key, direction, tp, sl in lookup[p]:
                signals.append((i, key, direction, tp, sl))

    return signals


# ============================================================
# MTF Filter
# ============================================================

def apply_mtf_filter(signals, df, strong_1h, mode='permissive'):
    """Filter signals based on 1h directional bias.

    Modes:
        permissive: Strong 1h must match; neutral allows both
        strict: Only enter when strong 1h matches direction
    """
    pattern_1h = df['pattern_3_1h'].values
    n = len(pattern_1h)

    filtered = []
    stats = {
        'total': 0, 'passed': 0,
        'blocked_mismatch': 0, 'blocked_no_1h': 0,
        'passed_neutral': 0, 'passed_match': 0
    }

    for bar_idx, key, direction, tp, sl in signals:
        stats['total'] += 1
        cur_1h = pattern_1h[bar_idx] if bar_idx < n else None

        if cur_1h and not pd.isna(cur_1h) and cur_1h in strong_1h:
            bias = strong_1h[cur_1h]
            if bias == direction:
                filtered.append((bar_idx, key, direction, tp, sl))
                stats['passed'] += 1
                stats['passed_match'] += 1
            else:
                stats['blocked_mismatch'] += 1
        else:
            if mode == 'permissive':
                filtered.append((bar_idx, key, direction, tp, sl))
                stats['passed'] += 1
                stats['passed_neutral'] += 1
            else:
                stats['blocked_no_1h'] += 1

    return filtered, stats


# ============================================================
# Backtest Engine
# ============================================================

def bt_signals_from_list(signal_list, opens, highs, lows, n_bars,
                         override_tp=None, override_sl=None):
    """Backtest signals. Matches scanner bt_signals logic exactly.

    Returns list of (entry_bar, exit_bar, pnl_pct, direction, key, hold_bars).
    Timeout = DROP (not counted).
    """
    fee = FEE_PCT * LEVERAGE
    trades = []

    for bar_idx, key, direction, tp, sl in signal_list:
        tp_pct = override_tp if override_tp is not None else tp
        sl_pct = override_sl if override_sl is not None else sl

        entry_bar = bar_idx + 1
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue

        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        for j in range(entry_bar + 1, min(entry_bar + 1 + MAX_BARS, n_bars)):
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
                trades.append((entry_bar, j, pnl, direction, key, j - entry_bar))
                break
            elif ht:
                trades.append((entry_bar, j, tp_pct * LEVERAGE - fee, direction, key,
                               j - entry_bar))
                break
            elif hs:
                trades.append((entry_bar, j, -sl_pct * LEVERAGE - fee, direction, key,
                               j - entry_bar))
                break
            # else: timeout → DROP

    return trades


# ============================================================
# Statistics
# ============================================================

def compute_trade_stats(trades, total_bars):
    """Per-trade statistics."""
    if not trades:
        return {
            'trades': 0, 'wr': 0, 'edge': 0, 'avg_win': 0, 'avg_loss': 0,
            'pnl_simple': 0, 'trades_per_day': 0, 'median_hold_bars': 0,
            'long_trades': 0, 'short_trades': 0, 'long_wr': 0, 'short_wr': 0
        }

    pnls = [t[2] for t in trades]
    dirs = [t[3] for t in trades]
    holds = [t[5] for t in trades]

    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(pnls) * 100

    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = abs(np.mean(loss_pnls)) if loss_pnls else 0
    edge = np.mean(pnls)

    total_days = total_bars / 288
    trades_per_day = len(pnls) / total_days if total_days > 0 else 0

    long_t = [t for t in trades if t[3] == 'LONG']
    short_t = [t for t in trades if t[3] == 'SHORT']
    long_wins = sum(1 for t in long_t if t[2] > 0)
    short_wins = sum(1 for t in short_t if t[2] > 0)

    return {
        'trades': len(pnls),
        'wr': round(wr, 1),
        'edge': round(edge, 3),
        'avg_win': round(avg_win, 3),
        'avg_loss': round(avg_loss, 3),
        'pnl_simple': round(sum(pnls), 1),
        'trades_per_day': round(trades_per_day, 2),
        'median_hold_bars': int(np.median(holds)),
        'median_hold_hours': round(int(np.median(holds)) * 5 / 60, 1),
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'long_wr': round(long_wins / len(long_t) * 100, 1) if long_t else 0,
        'short_wr': round(short_wins / len(short_t) * 100, 1) if short_t else 0,
    }


def compute_portfolio_pnl(trades, total_bars):
    """Compound portfolio PnL + MDD (sequential, no multi-position)."""
    if not trades:
        return {'pnl_pct': 0, 'mdd_pct': 0, 'pnl_mdd': 0}

    sorted_trades = sorted(trades, key=lambda t: t[0])

    equity = 100.0
    peak = equity
    max_dd = 0

    for t in sorted_trades:
        equity *= (1 + t[2] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_pnl = (equity / 100 - 1) * 100
    pnl_mdd = total_pnl / max_dd if max_dd > 0 else float('inf')

    return {
        'pnl_pct': round(total_pnl, 1),
        'mdd_pct': round(max_dd, 1),
        'pnl_mdd': round(pnl_mdd, 2)
    }


# ============================================================
# Phase 3: MTF Filter Effect
# ============================================================

def run_phase3(df, signals, strong_1h, opens, highs, lows, n_bars, total_bars):
    """Phase 3: Baseline vs MTF-filtered backtest comparison."""
    print("\n" + "=" * 80)
    print("PHASE 3: MTF FILTER EFFECT")
    print("=" * 80)

    results = {}

    # --- A. Baseline (no filter) ---
    print("\n[Phase 3A] Baseline: all 35-pattern signals, current TP/SL...")
    trades_base = bt_signals_from_list(signals, opens, highs, lows, n_bars)
    stats_base = compute_trade_stats(trades_base, total_bars)
    port_base = compute_portfolio_pnl(trades_base, total_bars)
    results['baseline'] = {**stats_base, **port_base}

    # --- B. MTF Permissive (strong 1h match required; neutral → pass) ---
    print("[Phase 3B] MTF Permissive: strong 1h match + neutral pass...")
    filtered_perm, fstats_perm = apply_mtf_filter(
        signals, df, strong_1h, mode='permissive')
    trades_perm = bt_signals_from_list(filtered_perm, opens, highs, lows, n_bars)
    stats_perm = compute_trade_stats(trades_perm, total_bars)
    port_perm = compute_portfolio_pnl(trades_perm, total_bars)
    results['mtf_permissive'] = {
        **stats_perm, **port_perm, 'filter': fstats_perm}

    # --- C. MTF Strict (only strong 1h matches allowed) ---
    print("[Phase 3C] MTF Strict: only strong 1h direction matches...")
    filtered_strict, fstats_strict = apply_mtf_filter(
        signals, df, strong_1h, mode='strict')
    trades_strict = bt_signals_from_list(filtered_strict, opens, highs, lows, n_bars)
    stats_strict = compute_trade_stats(trades_strict, total_bars)
    port_strict = compute_portfolio_pnl(trades_strict, total_bars)
    results['mtf_strict'] = {
        **stats_strict, **port_strict, 'filter': fstats_strict}

    # --- Print comparison ---
    print("\n" + "-" * 80)
    print(f"{'Metric':>20} | {'Baseline':>12} | {'MTF Perm':>12} | {'MTF Strict':>12}")
    print("-" * 80)

    for metric in ['trades', 'wr', 'edge', 'avg_win', 'avg_loss',
                    'trades_per_day', 'median_hold_hours',
                    'long_trades', 'short_trades', 'long_wr', 'short_wr',
                    'pnl_pct', 'mdd_pct', 'pnl_mdd']:
        v_b = results['baseline'].get(metric, '-')
        v_p = results['mtf_permissive'].get(metric, '-')
        v_s = results['mtf_strict'].get(metric, '-')
        fmt = '.1f' if isinstance(v_b, float) else ''
        if fmt:
            print(f"{metric:>20} | {v_b:>12{fmt}} | {v_p:>12{fmt}} | {v_s:>12{fmt}}")
        else:
            print(f"{metric:>20} | {v_b:>12} | {v_p:>12} | {v_s:>12}")

    # --- Filter statistics ---
    print("\n--- Filter Stats ---")
    for name, fst in [('Permissive', fstats_perm), ('Strict', fstats_strict)]:
        pct = fst['passed'] / fst['total'] * 100 if fst['total'] > 0 else 0
        print(f"  {name}: {fst['passed']}/{fst['total']} passed ({pct:.1f}%) "
              f"| match={fst['passed_match']} neutral={fst['passed_neutral']} "
              f"| blocked: mismatch={fst['blocked_mismatch']} "
              f"no_1h={fst['blocked_no_1h']}")

    # --- Comparison summary ---
    wr_imp_perm = stats_perm['wr'] - stats_base['wr']
    wr_imp_strict = stats_strict['wr'] - stats_base['wr']
    edge_imp_perm = stats_perm['edge'] - stats_base['edge']
    edge_imp_strict = stats_strict['edge'] - stats_base['edge']

    results['comparison'] = {
        'wr_improvement_permissive': round(wr_imp_perm, 1),
        'wr_improvement_strict': round(wr_imp_strict, 1),
        'edge_improvement_permissive': round(edge_imp_perm, 3),
        'edge_improvement_strict': round(edge_imp_strict, 3),
        'trade_reduction_permissive': round(
            (1 - stats_perm['trades'] / stats_base['trades']) * 100, 1)
        if stats_base['trades'] > 0 else 0,
        'trade_reduction_strict': round(
            (1 - stats_strict['trades'] / stats_base['trades']) * 100, 1)
        if stats_base['trades'] > 0 else 0,
    }

    print(f"\n--- Phase 3 Summary ---")
    print(f"  Permissive: WR {wr_imp_perm:+.1f}pp, edge {edge_imp_perm:+.3f}%/trade, "
          f"trades {results['comparison']['trade_reduction_permissive']:.1f}% reduction")
    print(f"  Strict:     WR {wr_imp_strict:+.1f}pp, edge {edge_imp_strict:+.3f}%/trade, "
          f"trades {results['comparison']['trade_reduction_strict']:.1f}% reduction")

    # Go/No-Go for Phase 4
    go_phase4 = wr_imp_perm >= 3.0 or wr_imp_strict >= 3.0
    results['go_phase4'] = go_phase4
    if go_phase4:
        print(f"\n  → GO for Phase 4 (WR improvement >= 3pp)")
    else:
        print(f"\n  → Phase 4 will run regardless for completeness, "
              f"but WR improvement < 3pp")

    return results, filtered_perm, trades_perm


# ============================================================
# Phase 4: Tight TP Optimization
# ============================================================

def run_phase4(filtered_signals, opens, highs, lows, n_bars, total_bars,
               baseline_stats):
    """Phase 4: Grid search tight TP/SL with MTF-filtered signals."""
    print("\n" + "=" * 80)
    print("PHASE 4: TIGHT TP/SL OPTIMIZATION (MTF-filtered signals)")
    print(f"TP grid: {TP_GRID}")
    print(f"SL grid: {SL_GRID}")
    print(f"Fee constraint: TP >= 0.40% (FEE_TOTAL={FEE_TOTAL:.2f}%)")
    print("=" * 80)

    grid_results = []

    for tp in TP_GRID:
        for sl in SL_GRID:
            trades = bt_signals_from_list(
                filtered_signals, opens, highs, lows, n_bars,
                override_tp=tp, override_sl=sl)
            stats = compute_trade_stats(trades, total_bars)
            port = compute_portfolio_pnl(trades, total_bars)

            rw_wr = sl / (tp + sl) * 100
            wr_excess = stats['wr'] - rw_wr

            grid_results.append({
                'tp': tp, 'sl': sl,
                'rw_wr': round(rw_wr, 1),
                'wr_excess': round(wr_excess, 1),
                **stats, **port,
            })

    # Sort by WR Excess (primary), then PnL/MDD (secondary)
    grid_results.sort(key=lambda x: (-x['wr_excess'], -x['pnl_mdd']))

    # --- Print top combinations ---
    print(f"\n{'TP':>5} {'SL':>5} | {'WR%':>6} {'RW%':>6} {'Exc':>6} "
          f"| {'Edge':>7} {'Trades':>7} {'T/day':>6} "
          f"| {'PnL%':>8} {'MDD%':>6} {'P/M':>7} {'Hold_h':>7}")
    print("-" * 95)

    for r in grid_results[:20]:
        print(f"{r['tp']:>5.2f} {r['sl']:>5.2f} | {r['wr']:>5.1f}% {r['rw_wr']:>5.1f}% "
              f"{r['wr_excess']:>+5.1f} | {r['edge']:>+6.3f}% {r['trades']:>7} "
              f"{r['trades_per_day']:>6.2f} | {r['pnl_pct']:>+7.1f}% "
              f"{r['mdd_pct']:>5.1f}% {r['pnl_mdd']:>7.2f} "
              f"{r['median_hold_hours']:>6.1f}h")

    # Best by WR Excess (trades >= 100)
    viable = [r for r in grid_results if r['trades'] >= 100 and r['wr_excess'] > 0]
    best = viable[0] if viable else grid_results[0]

    print(f"\n--- Best viable (WR Excess, trades >= 100) ---")
    print(f"  TP={best['tp']:.2f}%, SL={best['sl']:.2f}%")
    print(f"  WR={best['wr']:.1f}%, WR Excess={best['wr_excess']:+.1f}pp, "
          f"Edge={best['edge']:+.3f}%/trade")
    print(f"  Trades={best['trades']}, {best['trades_per_day']:.2f}/day, "
          f"hold={best['median_hold_hours']:.1f}h")
    print(f"  PnL={best['pnl_pct']:+.1f}%, MDD={best['mdd_pct']:.1f}%, "
          f"PnL/MDD={best['pnl_mdd']:.2f}")

    # Compare best tight TP vs baseline per-pattern TP/SL
    print(f"\n--- Tight TP vs Baseline (per-pattern TP/SL) ---")
    print(f"  {'':>18} | {'Baseline':>10} | {'Tight TP':>10} | {'Delta':>10}")
    print(f"  {'-'*55}")
    for m in ['wr', 'edge', 'trades', 'trades_per_day',
              'median_hold_hours', 'pnl_pct', 'mdd_pct', 'pnl_mdd']:
        b = baseline_stats.get(m, 0)
        t = best.get(m, 0)
        delta = t - b if isinstance(t, (int, float)) else '-'
        fmt = '.1f' if isinstance(b, float) else ''
        if isinstance(delta, (int, float)):
            print(f"  {m:>18} | {b:>10{fmt}} | {t:>10{fmt}} | {delta:>+10{fmt}}")
        else:
            print(f"  {m:>18} | {b:>10{fmt}} | {t:>10{fmt}} | {'':>10}")

    results = {
        'grid_results': grid_results,
        'best': best,
        'viable_count': len(viable),
    }

    return results


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='MTF Filter Backtest — Phase 3 & 4')
    parser.add_argument('--skip-phase4', action='store_true',
                        help='Skip Phase 4 tight TP optimization')
    parser.add_argument('--tp-grid', type=str,
                        default=','.join(str(x) for x in TP_GRID),
                        help=f'Comma-separated TP grid (default: {TP_GRID})')
    parser.add_argument('--sl-grid', type=str,
                        default=','.join(str(x) for x in SL_GRID),
                        help=f'Comma-separated SL grid (default: {SL_GRID})')
    return parser.parse_args()


def main():
    args = parse_args()

    # Override grids if specified
    global TP_GRID, SL_GRID
    TP_GRID = [float(x) for x in args.tp_grid.split(',')]
    SL_GRID = [float(x) for x in args.sl_grid.split(',')]

    print("=" * 80)
    print("MTF FILTER BACKTEST — Phase 3 & 4")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Protocol: LEVERAGE={LEVERAGE}, FEE={FEE_PCT}% "
          f"(total={FEE_TOTAL:.2f}%), MAX_BARS={MAX_BARS}")
    print("=" * 80)

    # ================================================================
    # DATA PREPARATION
    # ================================================================
    df = prepare_data()
    prod_patterns = load_production_patterns()
    strong_1h = load_strong_1h_patterns()

    # Pre-extract arrays for fast backtest
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)
    total_bars = n_bars
    total_days = total_bars / 288

    # Build signal index
    signals = build_signal_index(df, prod_patterns)
    print(f"\n[Signals] Total 5m signals: {len(signals):,} "
          f"({len(signals)/total_days:.1f}/day) over {total_days:.0f} days")

    # ================================================================
    # PHASE 3: MTF FILTER EFFECT (Original 15 strong patterns)
    # ================================================================
    phase3, filtered_perm, trades_perm = run_phase3(
        df, signals, strong_1h, opens, highs, lows, n_bars, total_bars)

    # ================================================================
    # PHASE 3X: EXPANDED THRESHOLD SWEEP
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 3X: EXPANDED 1h PATTERN THRESHOLD SWEEP")
    print("Testing accuracy thresholds: 50%, 51%, 52%, 53%, 54%, 55%")
    print("min_occurrences = 30 (relaxed from 50)")
    print("=" * 80)

    # Load 1h data for expanded analysis
    df_1h_file = DATA_DIR / 'btc_1h_720days.csv'
    if df_1h_file.exists():
        df_1h = pd.read_csv(df_1h_file, parse_dates=['timestamp'])
    else:
        df_1h = aggregate_to_higher_tf(df, 60)
        df_1h = classify_dataframe(df_1h)

    tiers, best_per_pattern = build_expanded_1h_patterns(df_1h, min_occ=30)

    # Compute time coverage for each tier
    pattern_1h_vals = df['pattern_3_1h'].values
    total_5m_rows = len(df)

    # Run sweep: for each tier, apply filter (permissive + strict)
    baseline_stats = phase3['baseline']
    sweep_results = {}

    print(f"\n{'Thr':>4} {'#1hPat':>7} | {'Mode':>10} "
          f"| {'Trades':>7} {'WR%':>6} {'Edge':>7} {'T/day':>6} "
          f"| {'PnL%':>8} {'MDD%':>6} {'P/M':>7} "
          f"| {'WRdelta':>8} {'Edgedelta':>10} {'TrdPct':>7}")
    print("-" * 120)

    # Baseline row
    print(f"{'BASE':>4} {'---':>7} | {'no_filt':>10} "
          f"| {baseline_stats['trades']:>7} {baseline_stats['wr']:>5.1f}% "
          f"{baseline_stats['edge']:>+6.3f}% {baseline_stats['trades_per_day']:>6.2f} "
          f"| {baseline_stats['pnl_pct']:>+7.1f}% {baseline_stats['mdd_pct']:>5.1f}% "
          f"{baseline_stats['pnl_mdd']:>7.2f} | {'---':>8} {'---':>10} {'100%':>7}")

    for thr in sorted(tiers.keys()):
        tier_1h = tiers[thr]
        if not tier_1h:
            continue

        # Calculate time coverage of this tier
        covered = sum(1 for v in pattern_1h_vals
                      if v and not pd.isna(v) and v in tier_1h)
        coverage_pct = covered / total_5m_rows * 100

        for mode in ['permissive', 'strict']:
            filtered, fstats = apply_mtf_filter(
                signals, df, tier_1h, mode=mode)
            trades = bt_signals_from_list(
                filtered, opens, highs, lows, n_bars)
            stats = compute_trade_stats(trades, total_bars)
            port = compute_portfolio_pnl(trades, total_bars)

            wr_d = stats['wr'] - baseline_stats['wr']
            edge_d = stats['edge'] - baseline_stats['edge']
            trd_pct = (stats['trades'] / baseline_stats['trades'] * 100
                       if baseline_stats['trades'] > 0 else 0)

            label = f"{thr:.0f}%"
            print(f"{label:>4} {len(tier_1h):>4}({coverage_pct:>3.0f}%) "
                  f"| {mode:>10} "
                  f"| {stats['trades']:>7} {stats['wr']:>5.1f}% "
                  f"{stats['edge']:>+6.3f}% {stats['trades_per_day']:>6.2f} "
                  f"| {port['pnl_pct']:>+7.1f}% {port['mdd_pct']:>5.1f}% "
                  f"{port['pnl_mdd']:>7.2f} "
                  f"| {wr_d:>+7.1f}pp {edge_d:>+9.3f}% {trd_pct:>6.1f}%")

            sweep_results[f"thr{thr:.0f}_{mode}"] = {
                'threshold': thr,
                'mode': mode,
                'n_1h_patterns': len(tier_1h),
                'coverage_pct': round(coverage_pct, 1),
                **stats, **port,
                'wr_delta': round(wr_d, 1),
                'edge_delta': round(edge_d, 3),
                'trade_retention_pct': round(trd_pct, 1),
            }

    # Find best sweep result by edge (with reasonable trade count)
    viable_sweep = {k: v for k, v in sweep_results.items()
                    if v['trades'] >= 200 and v['edge'] > baseline_stats['edge']}
    if viable_sweep:
        best_sweep_key = max(viable_sweep, key=lambda k: viable_sweep[k]['edge'])
        bs = viable_sweep[best_sweep_key]
        print(f"\n--- Best Sweep (edge, trades>=200) ---")
        print(f"  Config: {best_sweep_key}")
        print(f"  WR={bs['wr']:.1f}% ({bs['wr_delta']:+.1f}pp), "
              f"Edge={bs['edge']:+.3f}% ({bs['edge_delta']:+.3f}%), "
              f"Trades={bs['trades']} ({bs['trade_retention_pct']:.0f}% of baseline)")
        print(f"  PnL={bs['pnl_pct']:+.1f}%, MDD={bs['mdd_pct']:.1f}%, "
              f"PnL/MDD={bs['pnl_mdd']:.2f}")
    else:
        best_sweep_key = None
        print(f"\n--- No sweep config beats baseline with trades>=200 ---")

    # Also check best by PnL/MDD
    viable_pnlmdd = {k: v for k, v in sweep_results.items()
                     if v['trades'] >= 200 and v['pnl_mdd'] > baseline_stats.get('pnl_mdd', 0)}
    if viable_pnlmdd:
        best_pm_key = max(viable_pnlmdd, key=lambda k: viable_pnlmdd[k]['pnl_mdd'])
        bpm = viable_pnlmdd[best_pm_key]
        print(f"\n--- Best Sweep (PnL/MDD, trades>=200) ---")
        print(f"  Config: {best_pm_key}")
        print(f"  PnL/MDD={bpm['pnl_mdd']:.2f} (baseline={baseline_stats.get('pnl_mdd', 0):.2f})")
        print(f"  WR={bpm['wr']:.1f}%, Edge={bpm['edge']:+.3f}%, Trades={bpm['trades']}")

    # ================================================================
    # PHASE 4: TIGHT TP OPTIMIZATION (on best sweep if found)
    # ================================================================
    phase4 = None
    phase4_input = filtered_perm  # default: original permissive

    # If expanded sweep found improvement, use that tier for Phase 4
    if best_sweep_key and viable_sweep:
        bs_cfg = viable_sweep[best_sweep_key]
        thr_val = bs_cfg['threshold']
        mode_val = bs_cfg['mode']
        best_tier_1h = tiers[thr_val]
        phase4_input, _ = apply_mtf_filter(
            signals, df, best_tier_1h, mode=mode_val)
        print(f"\n[Phase 4] Using best sweep config: {best_sweep_key}")
    else:
        print(f"\n[Phase 4] Using original permissive filter (no sweep improvement)")

    if not args.skip_phase4:
        phase4 = run_phase4(
            phase4_input, opens, highs, lows, n_bars, total_bars,
            baseline_stats)

    # ================================================================
    # DECISION
    # ================================================================
    print("\n" + "=" * 80)
    print("OVERALL DECISION")
    print("=" * 80)

    # Check original Phase 3
    wr_imp = phase3['comparison']['wr_improvement_permissive']
    edge_imp = phase3['comparison']['edge_improvement_permissive']
    mtf_effective_orig = wr_imp >= 3.0

    # Check expanded sweep
    sweep_effective = best_sweep_key is not None
    if sweep_effective and viable_sweep:
        bs = viable_sweep[best_sweep_key]
        sweep_wr_imp = bs['wr_delta']
        sweep_edge_imp = bs['edge_delta']
    else:
        sweep_wr_imp = 0
        sweep_edge_imp = 0

    mtf_effective = mtf_effective_orig or (sweep_wr_imp >= 3.0)

    tight_tp_viable = False
    if phase4:
        best = phase4['best']
        tight_tp_viable = (best['wr_excess'] > 5.0
                           and best['edge'] > 0.10
                           and best['trades'] >= 100)

    decision = {
        'mtf_filter_effective_original': mtf_effective_orig,
        'mtf_filter_effective_expanded': sweep_wr_imp >= 3.0 if sweep_effective else False,
        'mtf_filter_effective': mtf_effective,
        'wr_improvement_original': wr_imp,
        'wr_improvement_expanded': sweep_wr_imp,
        'edge_improvement_original': edge_imp,
        'edge_improvement_expanded': sweep_edge_imp,
        'best_sweep_config': best_sweep_key,
        'tight_tp_viable': tight_tp_viable,
    }

    if tight_tp_viable and phase4:
        decision['recommended'] = (
            f"MTF + Tight TP={phase4['best']['tp']:.2f}%/"
            f"SL={phase4['best']['sl']:.2f}%")
        decision['recommended_wr'] = phase4['best']['wr']
        decision['recommended_edge'] = phase4['best']['edge']
    elif mtf_effective:
        decision['recommended'] = (
            f"MTF ({best_sweep_key}) + current per-pattern TP/SL"
            if sweep_effective else "MTF Permissive + current per-pattern TP/SL")
    else:
        decision['recommended'] = "STOP — Current v1.33.0 (MTF filter not effective)"

    print(f"  Original (15 strong): WR {wr_imp:+.1f}pp")
    print(f"  Expanded sweep best:  WR {sweep_wr_imp:+.1f}pp "
          f"(config: {best_sweep_key or 'none'})")
    print(f"  MTF effective: {mtf_effective}")
    print(f"  Tight TP viable: {tight_tp_viable}")
    print(f"  Recommendation: {decision['recommended']}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    output = {
        'study': 'mtf_filter_backtest',
        'date': datetime.now().isoformat(),
        'phases': ['phase_3'] + (['phase_4'] if phase4 else []),
        'parameters': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'fee_total': FEE_TOTAL,
            'max_bars': MAX_BARS,
            'production_patterns': len(prod_patterns),
            'strong_1h_patterns': len(strong_1h),
            'total_5m_bars': n_bars,
            'total_days': round(total_days, 1),
            'total_signals': len(signals),
        },
        'phase_3': phase3,
        'phase_3x_sweep': sweep_results,
        'decision': decision,
    }

    if phase4:
        # Save only top 20 grid results to keep JSON manageable
        phase4_save = {
            'best': phase4['best'],
            'viable_count': phase4['viable_count'],
            'tp_grid': TP_GRID,
            'sl_grid': SL_GRID,
            'top_20': phase4['grid_results'][:20],
        }
        output['phase_4'] = phase4_save

    out_file = RESULTS_DIR / 'mtf_filter_backtest.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return output


if __name__ == '__main__':
    main()
