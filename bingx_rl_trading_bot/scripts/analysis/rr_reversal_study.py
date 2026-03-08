#!/usr/bin/env python3
"""
R:R Reversal Study — Genuine Edge Detection with Symmetric/Favorable R:R
=========================================================================

Core Question:
  Does the pattern discovery pipeline find genuine predictive edge when
  R:R >= 1:1, where Random Walk WR = 50%?

Background:
  Current strategy uses R:R ~0.5:1 (TP ~2%, SL ~3.5%), producing Random Walk
  WR ~63.6%. Live WR 52.2% vs backtest 80.6% -- the gap between production
  filter pass rate (10.0%) and random filter pass rate (9.9%) is negligible.
  The R:R structure inflates WR making genuine edge hard to distinguish.

Phases:
  1. Symmetric R:R Scan (TP = SL): Grid over SL = 1.5-4.0% with TP = SL.
     Random Walk WR is exactly 50%, so any WR > 50% is genuine edge.
  2. Favorable R:R Scan (TP > SL): R:R = 1.5:1 and 2.0:1.
     Random Walk WR < 50%, lower WR required for profitability.
  3. Per-Pattern Optimal R:R: For each pattern, find best (TP, SL) from
     Phase 1+2 that maximizes WR Excess.
  4. Portfolio Backtest + WF: Hedge N=9 portfolio with 3-fold expanding
     window walk-forward validation.
  5. Random Baseline Comparison: Are R:R >= 1:1 results distinguishable
     from random patterns? Compare filter pass rates and OOS performance.

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based, same-bar resolution from bar OPEN)
  - Fee: FEE_PCT * LEVERAGE = 0.30% capital-space
  - Slippage: 0.02%
  - ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  - Timeout: 288 bars (24h) -> DROP
  - MC: 3 seeds (42, 123, 7), max p < 0.01
  - WF: 3-fold expanding window
  - classify_candle: production import (NEVER self-implement)

Data: btc_5m_270days_reclassified.csv (with neutral window)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# CONSTANTS (standard research protocol)
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10             # 0.05% entry + 0.05% exit
SLIPPAGE_BUFFER = 0.02     # 0.02% slippage buffer
MAX_BARS = 288             # 24h timeout -> DROP
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]
MC_THRESHOLD = 0.01
MIN_TRADES = 25
BARS_PER_DAY = 288

# ATR scaling params (production: a14/w576/clamp [0.6, 1.7])
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7

# Phase 1: Symmetric R:R grid (TP = SL)
SYMMETRIC_SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Phase 2: Favorable R:R (TP > SL)
FAVORABLE_RATIOS = [1.5, 2.0]
FAVORABLE_SL_GRID = [1.5, 2.0, 2.5, 3.0]

# Edge threshold lowered for R:R >= 1 (Random Walk WR = 50%)
EDGE_THRESHOLD_RR1 = 5.0   # 5pp above 50% baseline

# Neutral window
NEUTRAL_TOL_PCT = 1.0
MIN_NEUTRAL_DAYS = 90

# Portfolio params
MAX_POSITIONS = 9
DIRECTION_CAP = 7
TIMEOUT_BARS = 864          # 72h for portfolio sim

# File paths
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'rr_reversal_study.json')


# ============================================================
# DATA LOADING
# ============================================================

def load_and_classify(data_file):
    """Load CSV and classify candles using production classify_candle()."""
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)

    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW).mean()

    types = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body']
        if pd.isna(avg_b):
            avg_b = 1.0
        ct = classify_candle(row, avg_b)
        types.append(ct.value)

    df['candle_type'] = types
    print(f"Classified {len(df)} candles into 12 types")
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


def find_neutral_window(closes, tol_pct=NEUTRAL_TOL_PCT, min_days=MIN_NEUTRAL_DAYS,
                         bars_per_day=BARS_PER_DAY):
    """Find longest window where start_price ~ end_price within tolerance."""
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


# ============================================================
# ATR COMPUTATION
# ============================================================

def compute_atr_ratio(highs, lows, closes, atr_period=ATR_PERIOD, window=ATR_WINDOW):
    """ATR / rolling_median(ATR) ratio time series (Wilder smoothing)."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
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


# ============================================================
# BACKTESTING ENGINE
# ============================================================

def bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                    opens, highs, lows, n_bars,
                    atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI):
    """Backtest with ATR-scaled TP/SL. Timeout trades are DROPPED."""
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
        # Timeout -> DROP (no append)

    return trades


# ============================================================
# MC TEST
# ============================================================

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


# ============================================================
# PORTFOLIO SIMULATION (1-pos filter)
# ============================================================

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
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0}
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
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(np.mean(losses), 2) if losses else 0,
    }


# ============================================================
# COMPOUND PORTFOLIO (N=9 Hedge with direction cap)
# ============================================================

def compound_portfolio(pattern_trades_list, n_bars,
                       max_pos=MAX_POSITIONS, dir_cap=DIRECTION_CAP,
                       timeout_bars=TIMEOUT_BARS):
    """
    Simulate Hedge N=9 compound portfolio.

    pattern_trades_list: list of dicts with 'direction', 'trades' (list of (eb, xb, pnl))
    Returns compound PnL, MDD, trade count, WR.
    """
    # Build signal schedule: sorted list of (entry_bar, exit_bar, pnl, direction)
    all_signals = []
    for pt in pattern_trades_list:
        direction = pt['direction']
        for (eb, xb, pnl) in pt['trades']:
            # Apply timeout
            if xb - eb > timeout_bars:
                continue  # DROP
            all_signals.append((eb, xb, pnl, direction))

    if not all_signals:
        return {'compound_pnl': 0, 'mdd': 0, 'trades': 0, 'wr': 0}

    all_signals.sort(key=lambda x: x[0])

    # Simulate with N slots
    equity = 100.0
    peak_eq = 100.0
    max_dd = 0.0
    active = []  # list of (exit_bar, direction)
    wins = 0
    total_trades = 0

    for eb, xb, pnl, direction in all_signals:
        # Close expired positions
        active = [(axb, ad) for axb, ad in active if axb > eb]

        if len(active) >= max_pos:
            continue

        # Direction cap
        long_count = sum(1 for _, ad in active if ad == 'LONG')
        short_count = sum(1 for _, ad in active if ad == 'SHORT')
        if direction == 'LONG' and long_count >= dir_cap:
            continue
        if direction == 'SHORT' and short_count >= dir_cap:
            continue

        # Execute trade with 1/N sizing (compound)
        pos_frac = 1.0 / max_pos
        trade_return = pnl / 100.0  # pnl is already in % with leverage
        equity *= (1.0 + pos_frac * trade_return)
        total_trades += 1
        if pnl > 0:
            wins += 1

        if equity > peak_eq:
            peak_eq = equity
        dd = (peak_eq - equity) / peak_eq * 100
        if dd > max_dd:
            max_dd = dd

        active.append((xb, direction))

    compound_pnl = (equity / 100.0 - 1.0) * 100.0
    wr = wins / total_trades * 100 if total_trades > 0 else 0

    return {
        'compound_pnl': round(compound_pnl, 1),
        'mdd': round(max_dd, 1),
        'trades': total_trades,
        'wr': round(wr, 1),
    }


# ============================================================
# EXPANDING WINDOW WF (for R:R >= 1 patterns)
# ============================================================

def expanding_window_wf(signal_index, opens, highs, lows, closes, n_bars,
                        selected_patterns, n_folds=3):
    """
    3-fold expanding window WF for a set of selected patterns.

    selected_patterns: list of dicts with 'pattern', 'direction', 'tp', 'sl'
    Returns WF results dict.
    """
    seg_size = n_bars // (n_folds + 1)
    folds = []

    for fold in range(n_folds):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        # Re-scan patterns on IS data
        is_atr = compute_atr_ratio(
            highs[:is_end], lows[:is_end], closes[:is_end],
            ATR_PERIOD, ATR_WINDOW
        )

        # For each selected pattern: re-evaluate on IS, test on OOS
        oos_pattern_trades = []

        for sp in selected_patterns:
            pat_name = sp['pattern']
            direction = sp['direction']
            tp = sp['tp']
            sl = sp['sl']

            # Re-check edge on IS
            is_sigs = [s for s in signal_index.get(pat_name, []) if s < is_end]
            if len(is_sigs) < MIN_TRADES:
                continue

            is_trades = bt_signals_atr(is_sigs, direction, tp, sl,
                                        opens, highs, lows, is_end,
                                        is_atr, CLAMP_LO, CLAMP_HI)
            if len(is_trades) < MIN_TRADES:
                continue

            is_pnls = [t[2] for t in is_trades]
            is_wr = sum(1 for p in is_pnls if p > 0) / len(is_pnls) * 100
            baseline_wr = sl / (tp + sl) * 100
            is_edge = is_wr - baseline_wr

            if is_edge < EDGE_THRESHOLD_RR1:
                continue

            is_mc = mc_test(is_pnls)
            if is_mc >= MC_THRESHOLD:
                continue

            # OOS backtest
            oos_atr = compute_atr_ratio(
                highs[:oos_end], lows[:oos_end], closes[:oos_end],
                ATR_PERIOD, ATR_WINDOW
            )
            oos_sigs = [s for s in signal_index.get(pat_name, [])
                        if oos_start <= s < oos_end]
            if not oos_sigs:
                continue

            oos_trades = bt_signals_atr(oos_sigs, direction, tp, sl,
                                         opens, highs, lows, oos_end,
                                         oos_atr, CLAMP_LO, CLAMP_HI)
            oos_pattern_trades.append({
                'direction': direction,
                'trades': oos_trades,
            })

        # Portfolio stats on OOS trades
        all_oos = []
        for pt in oos_pattern_trades:
            all_oos.extend(pt['trades'])
        oos_port = portfolio_1pos(all_oos)
        oos_stats = calc_stats(oos_port)

        # Compound portfolio
        oos_compound = compound_portfolio(oos_pattern_trades, oos_end)

        folds.append({
            'fold': fold + 1,
            'is_bars': is_end,
            'oos_bars': oos_end - oos_start,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_mdd': oos_stats['mdd'],
            'oos_compound_pnl': oos_compound['compound_pnl'],
            'oos_compound_mdd': oos_compound['mdd'],
            'oos_compound_trades': oos_compound['trades'],
            'oos_positive': oos_stats['pnl'] > 0,
        })

        print(f"    Fold {fold+1}/{n_folds}: IS=[0,{is_end}) OOS=[{oos_start},{oos_end}) | "
              f"OOS trades={oos_stats['trades']}, WR={oos_stats['wr']}%, "
              f"PnL={oos_stats['pnl']}%, compound={oos_compound['compound_pnl']}%")

    positive_folds = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_pnl'] for f in folds), 1)
    total_oos_trades = sum(f['oos_trades'] for f in folds)

    return {
        'n_folds': n_folds,
        'folds': folds,
        'positive_folds': positive_folds,
        'total_oos_pnl': total_oos_pnl,
        'total_oos_trades': total_oos_trades,
        'wf_pass': positive_folds == n_folds,
    }


# ============================================================
# PHASE 1: Symmetric R:R Scan (TP = SL)
# ============================================================

def phase1_symmetric_scan(signal_index, opens, highs, lows, closes, n_bars,
                           atr_ratio, bar_start, bar_end):
    """
    Scan all patterns with symmetric TP = SL.
    Random Walk WR = 50% (exactly), so genuine edge is clearly visible.
    """
    print("\n" + "=" * 90)
    print("  PHASE 1: SYMMETRIC R:R SCAN (TP = SL)")
    print("  Random Walk WR = 50.0% for all symmetric levels")
    print("  Edge threshold: >= 5pp above 50%")
    print("=" * 90)

    grid_results = []

    for sl_val in SYMMETRIC_SL_GRID:
        tp_val = sl_val  # Symmetric: TP = SL
        baseline_wr = 50.0  # Exact for TP = SL

        n_pass = 0
        n_tested = 0
        all_wrs = []
        all_edges = []
        passing_patterns = []

        for pat_name, all_sigs in signal_index.items():
            sigs = [s for s in all_sigs if bar_start <= s < bar_end]

            for direction in ['LONG', 'SHORT']:
                if len(sigs) < MIN_TRADES:
                    continue

                trades = bt_signals_atr(sigs, direction, tp_val, sl_val,
                                         opens, highs, lows, bar_end,
                                         atr_ratio, CLAMP_LO, CLAMP_HI)
                if len(trades) < MIN_TRADES:
                    continue

                n_tested += 1
                pnls = [t[2] for t in trades]
                wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                edge = wr - baseline_wr
                all_wrs.append(wr)
                all_edges.append(edge)

                if edge >= EDGE_THRESHOLD_RR1:
                    p = mc_test(pnls)
                    if p < MC_THRESHOLD:
                        n_pass += 1
                        passing_patterns.append({
                            'pattern': pat_name,
                            'direction': direction,
                            'tp': tp_val,
                            'sl': sl_val,
                            'trades': len(trades),
                            'wr': round(wr, 1),
                            'edge': round(edge, 1),
                            'mc_p': round(p, 4),
                            'baseline_wr': baseline_wr,
                        })

        result = {
            'tp': tp_val,
            'sl': sl_val,
            'rr': 1.0,
            'baseline_wr': baseline_wr,
            'n_tested': n_tested,
            'n_passing': n_pass,
            'pass_rate': round(n_pass / n_tested * 100, 2) if n_tested > 0 else 0,
            'mean_wr': round(float(np.mean(all_wrs)), 1) if all_wrs else 0,
            'mean_edge': round(float(np.mean(all_edges)), 1) if all_edges else 0,
            'max_edge': round(float(np.max(all_edges)), 1) if all_edges else 0,
            'patterns': passing_patterns,
        }
        grid_results.append(result)

        print(f"  SL/TP={sl_val:.1f}%: tested={n_tested}, pass={n_pass} "
              f"({result['pass_rate']:.1f}%), mean_WR={result['mean_wr']:.1f}%, "
              f"mean_edge={result['mean_edge']:.1f}pp, max_edge={result['max_edge']:.1f}pp")

    # Summary
    total_passing = sum(r['n_passing'] for r in grid_results)
    all_passing = []
    for r in grid_results:
        all_passing.extend(r['patterns'])
    print(f"\n  PHASE 1 TOTAL: {total_passing} pattern-direction combos with "
          f"edge >= {EDGE_THRESHOLD_RR1}pp and MC < {MC_THRESHOLD}")

    return {
        'grid_results': grid_results,
        'total_passing': total_passing,
        'all_passing_patterns': all_passing,
    }


# ============================================================
# PHASE 2: Favorable R:R Scan (TP > SL)
# ============================================================

def phase2_favorable_scan(signal_index, opens, highs, lows, closes, n_bars,
                           atr_ratio, bar_start, bar_end):
    """
    Scan all patterns with R:R > 1 (favorable).
    Random Walk WR = SL/(TP+SL) < 50%.
    """
    print("\n" + "=" * 90)
    print("  PHASE 2: FAVORABLE R:R SCAN (TP > SL)")
    print("  Random Walk WR < 50% -- lower WR but higher per-trade expectancy")
    print("=" * 90)

    grid_results = []

    for rr_ratio in FAVORABLE_RATIOS:
        for sl_val in FAVORABLE_SL_GRID:
            tp_val = round(sl_val * rr_ratio, 2)
            baseline_wr = sl_val / (tp_val + sl_val) * 100

            # Skip if SL < 1.0% (execution risk)
            if sl_val < 1.0:
                continue

            n_pass = 0
            n_tested = 0
            all_wrs = []
            all_edges = []
            passing_patterns = []

            for pat_name, all_sigs in signal_index.items():
                sigs = [s for s in all_sigs if bar_start <= s < bar_end]

                for direction in ['LONG', 'SHORT']:
                    if len(sigs) < MIN_TRADES:
                        continue

                    trades = bt_signals_atr(sigs, direction, tp_val, sl_val,
                                             opens, highs, lows, bar_end,
                                             atr_ratio, CLAMP_LO, CLAMP_HI)
                    if len(trades) < MIN_TRADES:
                        continue

                    n_tested += 1
                    pnls = [t[2] for t in trades]
                    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                    edge = wr - baseline_wr
                    all_wrs.append(wr)
                    all_edges.append(edge)

                    if edge >= EDGE_THRESHOLD_RR1:
                        p = mc_test(pnls)
                        if p < MC_THRESHOLD:
                            n_pass += 1
                            passing_patterns.append({
                                'pattern': pat_name,
                                'direction': direction,
                                'tp': tp_val,
                                'sl': sl_val,
                                'rr': rr_ratio,
                                'trades': len(trades),
                                'wr': round(wr, 1),
                                'edge': round(edge, 1),
                                'mc_p': round(p, 4),
                                'baseline_wr': round(baseline_wr, 1),
                            })

            result = {
                'tp': tp_val,
                'sl': sl_val,
                'rr': rr_ratio,
                'baseline_wr': round(baseline_wr, 1),
                'n_tested': n_tested,
                'n_passing': n_pass,
                'pass_rate': round(n_pass / n_tested * 100, 2) if n_tested > 0 else 0,
                'mean_wr': round(float(np.mean(all_wrs)), 1) if all_wrs else 0,
                'mean_edge': round(float(np.mean(all_edges)), 1) if all_edges else 0,
                'patterns': passing_patterns,
            }
            grid_results.append(result)

            print(f"  R:R={rr_ratio}:1, TP={tp_val:.2f}%, SL={sl_val:.1f}%: "
                  f"tested={n_tested}, pass={n_pass} ({result['pass_rate']:.1f}%), "
                  f"baseline_WR={baseline_wr:.1f}%, mean_WR={result['mean_wr']:.1f}%")

    total_passing = sum(r['n_passing'] for r in grid_results)
    all_passing = []
    for r in grid_results:
        all_passing.extend(r['patterns'])
    print(f"\n  PHASE 2 TOTAL: {total_passing} pattern-direction combos")

    return {
        'grid_results': grid_results,
        'total_passing': total_passing,
        'all_passing_patterns': all_passing,
    }


# ============================================================
# PHASE 3: Per-Pattern Optimal R:R
# ============================================================

def phase3_per_pattern_optimal(signal_index, opens, highs, lows, closes, n_bars,
                                atr_ratio, bar_start, bar_end,
                                phase1_results, phase2_results):
    """
    For each pattern, select the best (TP, SL) from Phase 1+2 that maximizes
    WR Excess while maintaining edge >= 5pp and MC < 0.01.
    """
    print("\n" + "=" * 90)
    print("  PHASE 3: PER-PATTERN OPTIMAL R:R")
    print("  Select best TP/SL per pattern from Phase 1+2 by WR Excess")
    print("=" * 90)

    # Combine all passing patterns from Phase 1 and Phase 2
    all_candidates = {}  # key -> list of candidate configs

    for pat in phase1_results['all_passing_patterns']:
        key = f"{pat['pattern']}_{pat['direction']}"
        if key not in all_candidates:
            all_candidates[key] = []
        all_candidates[key].append(pat)

    for pat in phase2_results['all_passing_patterns']:
        key = f"{pat['pattern']}_{pat['direction']}"
        if key not in all_candidates:
            all_candidates[key] = []
        all_candidates[key].append(pat)

    print(f"  Unique pattern-direction combos with at least 1 passing config: "
          f"{len(all_candidates)}")

    # For each pattern-direction, select config with highest WR Excess
    selected = []
    for key, candidates in sorted(all_candidates.items()):
        # WR Excess = WR - baseline_WR = edge (already computed)
        best = max(candidates, key=lambda c: c['edge'])
        selected.append(best)

    # Dedup: if same pattern appears in both LONG and SHORT, keep better edge
    pat_by_name = {}
    for s in selected:
        pn = s['pattern']
        if pn not in pat_by_name or s['edge'] > pat_by_name[pn]['edge']:
            pat_by_name[pn] = s

    deduped = list(pat_by_name.values())
    n_removed = len(selected) - len(deduped)
    if n_removed > 0:
        print(f"  Removed {n_removed} dual-direction duplicates (kept better edge)")
    selected = deduped

    # Statistics
    n_long = sum(1 for s in selected if s['direction'] == 'LONG')
    n_short = sum(1 for s in selected if s['direction'] == 'SHORT')
    mean_wr = np.mean([s['wr'] for s in selected]) if selected else 0
    mean_edge = np.mean([s['edge'] for s in selected]) if selected else 0
    mean_trades = np.mean([s['trades'] for s in selected]) if selected else 0

    rr_values = [s['tp'] / s['sl'] if s['sl'] > 0 else 0 for s in selected]
    rr_dist = {
        'min': round(min(rr_values), 2) if rr_values else 0,
        'max': round(max(rr_values), 2) if rr_values else 0,
        'mean': round(float(np.mean(rr_values)), 2) if rr_values else 0,
        'median': round(float(np.median(rr_values)), 2) if rr_values else 0,
    }

    tp_range = [round(min(s['tp'] for s in selected), 2),
                round(max(s['tp'] for s in selected), 2)] if selected else [0, 0]
    sl_range = [round(min(s['sl'] for s in selected), 2),
                round(max(s['sl'] for s in selected), 2)] if selected else [0, 0]

    print(f"\n  PHASE 3 RESULT: {len(selected)} patterns ({n_long}L + {n_short}S)")
    print(f"  Mean WR: {mean_wr:.1f}%, Mean WR Excess: {mean_edge:.1f}pp")
    print(f"  Mean trades/pattern: {mean_trades:.0f}")
    print(f"  R:R distribution: min={rr_dist['min']}, median={rr_dist['median']}, "
          f"max={rr_dist['max']}")
    print(f"  TP range: {tp_range[0]}-{tp_range[1]}%")
    print(f"  SL range: {sl_range[0]}-{sl_range[1]}%")

    # Print individual patterns
    print(f"\n  {'Pattern':<20} {'Dir':<6} {'TP%':<6} {'SL%':<6} {'R:R':<6} "
          f"{'Trades':<7} {'WR%':<7} {'Edge':<7} {'MC_p':<8}")
    print("  " + "-" * 80)
    for s in sorted(selected, key=lambda x: -x['edge']):
        rr = s['tp'] / s['sl'] if s['sl'] > 0 else 0
        print(f"  {s['pattern']:<20} {s['direction']:<6} {s['tp']:<6.2f} {s['sl']:<6.2f} "
              f"{rr:<6.2f} {s['trades']:<7} {s['wr']:<7.1f} {s['edge']:<7.1f} "
              f"{s['mc_p']:<8.4f}")

    return {
        'total_patterns': len(selected),
        'long_count': n_long,
        'short_count': n_short,
        'mean_wr': round(float(mean_wr), 1),
        'mean_wr_excess': round(float(mean_edge), 1),
        'mean_trades': round(float(mean_trades), 0),
        'rr_distribution': rr_dist,
        'tp_range': tp_range,
        'sl_range': sl_range,
        'selected_patterns': selected,
    }


# ============================================================
# PHASE 4: Portfolio Backtest + WF
# ============================================================

def phase4_portfolio_wf(signal_index, opens, highs, lows, closes, n_bars,
                         selected_patterns, bar_start, bar_end, atr_ratio):
    """
    Full IS portfolio backtest + 3-fold expanding window WF.
    """
    print("\n" + "=" * 90)
    print("  PHASE 4: PORTFOLIO BACKTEST + WF")
    print(f"  {len(selected_patterns)} patterns, Hedge N={MAX_POSITIONS}, "
          f"DirCap={DIRECTION_CAP}, T={TIMEOUT_BARS}")
    print("=" * 90)

    if not selected_patterns:
        print("  NO PATTERNS -- skipping Phase 4")
        return {
            'is_stats': {},
            'wf_folds': [],
            'wf_pass': False,
        }

    # IS backtest (all data within neutral window)
    pattern_trades_list = []
    all_is_trades = []

    for sp in selected_patterns:
        sigs = [s for s in signal_index.get(sp['pattern'], [])
                if bar_start <= s < bar_end]
        if not sigs:
            continue
        trades = bt_signals_atr(sigs, sp['direction'], sp['tp'], sp['sl'],
                                 opens, highs, lows, bar_end,
                                 atr_ratio, CLAMP_LO, CLAMP_HI)
        if trades:
            pattern_trades_list.append({
                'direction': sp['direction'],
                'trades': trades,
            })
            all_is_trades.extend(trades)

    # IS 1-pos stats
    is_port = portfolio_1pos(all_is_trades)
    is_stats = calc_stats(is_port)

    # IS compound
    is_compound = compound_portfolio(pattern_trades_list, bar_end)

    print(f"\n  IS Results (1-pos): trades={is_stats['trades']}, "
          f"WR={is_stats['wr']}%, PnL={is_stats['pnl']}%, MDD={is_stats['mdd']}%")
    print(f"  IS Results (compound N={MAX_POSITIONS}): "
          f"trades={is_compound['trades']}, WR={is_compound['wr']}%, "
          f"PnL={is_compound['compound_pnl']}%, MDD={is_compound['mdd']}%")

    # WF
    print(f"\n  --- Walk-Forward Validation (3-fold expanding window) ---")
    wf_results = expanding_window_wf(
        signal_index, opens, highs, lows, closes, bar_end,
        selected_patterns, n_folds=3
    )

    wf_pass = wf_results['wf_pass']
    print(f"\n  WF RESULT: {wf_results['positive_folds']}/3 positive folds "
          f"({'PASS' if wf_pass else 'FAIL'})")
    print(f"  Total OOS PnL: {wf_results['total_oos_pnl']}%, "
          f"Total OOS trades: {wf_results['total_oos_trades']}")

    return {
        'is_stats': {
            '1pos': is_stats,
            'compound': is_compound,
        },
        'wf_results': wf_results,
        'wf_pass': wf_pass,
    }


# ============================================================
# PHASE 5: Random Baseline Comparison
# ============================================================

def phase5_random_comparison(signal_index, opens, highs, lows, closes, n_bars,
                              selected_patterns, bar_start, bar_end, atr_ratio):
    """
    Compare R:R >= 1:1 results against random patterns.

    Key question: do production patterns pass filters at higher rate than random?
    With R:R >= 1:1, random WR = 50%, so genuine edge should be more visible.
    """
    print("\n" + "=" * 90)
    print("  PHASE 5: RANDOM BASELINE COMPARISON")
    print("  Testing if R:R >= 1:1 pipeline distinguishes from random patterns")
    print("=" * 90)

    # All available pattern-direction combos
    all_patterns = list(signal_index.keys())
    all_combos = []
    for pat in all_patterns:
        for d in ['LONG', 'SHORT']:
            all_combos.append((pat, d))

    # Production patterns' TP/SL distribution
    if selected_patterns:
        prod_tps = [s['tp'] for s in selected_patterns]
        prod_sls = [s['sl'] for s in selected_patterns]
    else:
        prod_tps = [2.0]
        prod_sls = [2.0]

    n_prod = len(selected_patterns)
    n_trials = 10
    random_trials = []

    # Production: how many combos pass edge >= 5pp + MC < 0.01?
    prod_pass = 0
    prod_tested = 0
    for pat_name, all_sigs in signal_index.items():
        sigs = [s for s in all_sigs if bar_start <= s < bar_end]
        for direction in ['LONG', 'SHORT']:
            if len(sigs) < MIN_TRADES:
                continue

            # Use median TP/SL from production set for fair comparison
            median_tp = float(np.median(prod_tps))
            median_sl = float(np.median(prod_sls))

            trades = bt_signals_atr(sigs, direction, median_tp, median_sl,
                                     opens, highs, lows, bar_end,
                                     atr_ratio, CLAMP_LO, CLAMP_HI)
            if len(trades) < MIN_TRADES:
                continue
            prod_tested += 1
            pnls = [t[2] for t in trades]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            baseline_wr = median_sl / (median_tp + median_sl) * 100
            edge = wr - baseline_wr
            if edge >= EDGE_THRESHOLD_RR1:
                p = mc_test(pnls)
                if p < MC_THRESHOLD:
                    prod_pass += 1

    prod_pass_rate = prod_pass / prod_tested * 100 if prod_tested > 0 else 0

    print(f"  Production filter pass rate (edge>={EDGE_THRESHOLD_RR1}pp + MC<{MC_THRESHOLD}): "
          f"{prod_pass}/{prod_tested} = {prod_pass_rate:.1f}%")

    # Random: same test with random TP/SL assignments
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 7 + 42)

        # Sample random patterns (same count as production)
        sample_size = min(n_prod if n_prod > 0 else 50, len(all_combos))
        sample_idx = rng.choice(len(all_combos), size=sample_size, replace=False)
        sampled = [all_combos[i] for i in sample_idx]

        # Assign random TP/SL from production distribution (R:R >= 1 guaranteed)
        tp_samples = rng.choice(prod_tps, size=len(sampled))
        sl_samples = rng.choice(prod_sls, size=len(sampled))

        # Backtest sampled patterns (IS only within neutral window)
        is_trades_all = []
        pattern_trades_list = []
        for i, (pat, direction) in enumerate(sampled):
            sigs = [s for s in signal_index.get(pat, [])
                    if bar_start <= s < bar_end]
            if len(sigs) < 5:
                continue
            trades = bt_signals_atr(
                sigs, direction, tp_samples[i], sl_samples[i],
                opens, highs, lows, bar_end,
                atr_ratio, CLAMP_LO, CLAMP_HI
            )
            is_trades_all.extend(trades)
            pattern_trades_list.append({
                'direction': direction,
                'trades': trades,
            })

        is_port = portfolio_1pos(is_trades_all)
        is_stats = calc_stats(is_port)
        compound = compound_portfolio(pattern_trades_list, bar_end)

        # Count how many pass filters
        rand_pass = 0
        rand_tested = 0
        for i, (pat, direction) in enumerate(sampled):
            sigs = [s for s in signal_index.get(pat, [])
                    if bar_start <= s < bar_end]
            if len(sigs) < MIN_TRADES:
                continue
            trades = bt_signals_atr(
                sigs, direction, tp_samples[i], sl_samples[i],
                opens, highs, lows, bar_end,
                atr_ratio, CLAMP_LO, CLAMP_HI
            )
            if len(trades) < MIN_TRADES:
                continue
            rand_tested += 1
            pnls = [t[2] for t in trades]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            baseline_wr = sl_samples[i] / (tp_samples[i] + sl_samples[i]) * 100
            edge = wr - baseline_wr
            if edge >= EDGE_THRESHOLD_RR1:
                p = mc_test(pnls)
                if p < MC_THRESHOLD:
                    rand_pass += 1

        rand_pass_rate = rand_pass / rand_tested * 100 if rand_tested > 0 else 0

        random_trials.append({
            'trial': trial + 1,
            'n_sampled': len(sampled),
            'is_trades': is_stats['trades'],
            'is_wr': is_stats['wr'],
            'is_pnl': is_stats['pnl'],
            'compound_pnl': compound['compound_pnl'],
            'filter_pass': rand_pass,
            'filter_tested': rand_tested,
            'filter_pass_rate': round(rand_pass_rate, 1),
        })

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{n_trials}: IS WR={is_stats['wr']}%, "
                  f"PnL={is_stats['pnl']}%, compound={compound['compound_pnl']}%, "
                  f"filter pass={rand_pass_rate:.1f}%")

    # Summary
    random_is_wrs = [t['is_wr'] for t in random_trials]
    random_is_pnls = [t['is_pnl'] for t in random_trials]
    random_compound_pnls = [t['compound_pnl'] for t in random_trials]
    random_pass_rates = [t['filter_pass_rate'] for t in random_trials]

    print(f"\n  --- RANDOM BASELINE SUMMARY ---")
    print(f"  Random IS WR: {np.mean(random_is_wrs):.1f}% "
          f"(range {np.min(random_is_wrs):.1f} - {np.max(random_is_wrs):.1f})")
    print(f"  Random IS PnL: {np.mean(random_is_pnls):.1f}% "
          f"(range {np.min(random_is_pnls):.1f} - {np.max(random_is_pnls):.1f})")
    print(f"  Random compound PnL: {np.mean(random_compound_pnls):.1f}% "
          f"(range {np.min(random_compound_pnls):.1f} - {np.max(random_compound_pnls):.1f})")
    print(f"  Random filter pass rate: {np.mean(random_pass_rates):.1f}% "
          f"(prod: {prod_pass_rate:.1f}%)")
    print(f"  Edge over random: {prod_pass_rate - np.mean(random_pass_rates):.1f}pp")

    return {
        'prod_filter_pass': prod_pass,
        'prod_filter_tested': prod_tested,
        'prod_filter_pass_rate': round(prod_pass_rate, 1),
        'random_trials': random_trials,
        'random_mean_is_wr': round(float(np.mean(random_is_wrs)), 1),
        'random_mean_is_pnl': round(float(np.mean(random_is_pnls)), 1),
        'random_mean_compound_pnl': round(float(np.mean(random_compound_pnls)), 1),
        'random_mean_pass_rate': round(float(np.mean(random_pass_rates)), 1),
        'edge_over_random_pp': round(prod_pass_rate - float(np.mean(random_pass_rates)), 1),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print("=" * 90)
    print("  R:R REVERSAL STUDY")
    print("  Testing genuine edge with symmetric/favorable R:R (>= 1:1)")
    print("=" * 90)

    # Load data
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].tolist()

    # Build signal index
    signal_index = build_signal_index(types, n_bars)
    print(f"Signal index: {len(signal_index)} unique patterns")

    # Find neutral window
    nw = find_neutral_window(closes)
    if nw is None:
        print("WARNING: No neutral window found, using full data")
        bar_start, bar_end = 0, n_bars
    else:
        bar_start, bar_end = nw
        nw_days = (bar_end - bar_start) / BARS_PER_DAY
        price_start = closes[bar_start]
        price_end = closes[bar_end]
        price_change = (price_end / price_start - 1) * 100
        print(f"Neutral window: bars [{bar_start}, {bar_end}], "
              f"{nw_days:.0f} days, price change: {price_change:+.2f}%")

    # Compute ATR for neutral window
    atr_ratio = compute_atr_ratio(
        highs[:bar_end], lows[:bar_end], closes[:bar_end],
        ATR_PERIOD, ATR_WINDOW
    )
    print(f"ATR ratio computed: {np.sum(~np.isnan(atr_ratio))} valid bars")

    # ---- PHASE 1: Symmetric R:R ----
    p1 = phase1_symmetric_scan(signal_index, opens, highs, lows, closes,
                                n_bars, atr_ratio, bar_start, bar_end)

    # ---- PHASE 2: Favorable R:R ----
    p2 = phase2_favorable_scan(signal_index, opens, highs, lows, closes,
                                n_bars, atr_ratio, bar_start, bar_end)

    # ---- PHASE 3: Per-Pattern Optimal R:R ----
    p3 = phase3_per_pattern_optimal(signal_index, opens, highs, lows, closes,
                                     n_bars, atr_ratio, bar_start, bar_end, p1, p2)

    # ---- PHASE 4: Portfolio + WF ----
    p4 = phase4_portfolio_wf(signal_index, opens, highs, lows, closes, n_bars,
                              p3['selected_patterns'], bar_start, bar_end, atr_ratio)

    # ---- PHASE 5: Random Comparison ----
    p5 = phase5_random_comparison(signal_index, opens, highs, lows, closes, n_bars,
                                   p3['selected_patterns'], bar_start, bar_end, atr_ratio)

    # ---- COMPARISON WITH CURRENT R:R < 1 STRATEGY ----
    print("\n" + "=" * 90)
    print("  COMPARISON: R:R >= 1:1 vs Current R:R < 1 Strategy")
    print("=" * 90)

    # Load current production patterns for comparison
    try:
        with open(PATTERNS_FILE) as f:
            prod_data = json.load(f)
        prod_tpsl = prod_data.get('patterns_tpsl', {})
        prod_tps = [v[0] for v in prod_tpsl.values()]
        prod_sls = [v[1] for v in prod_tpsl.values()]
        prod_rrs = [tp / sl if sl > 0 else 0 for tp, sl in prod_tpsl.values()]
        print(f"  Current R:R<1 strategy: {len(prod_tpsl)} patterns")
        print(f"    TP range: {min(prod_tps):.2f} - {max(prod_tps):.2f}%")
        print(f"    SL range: {min(prod_sls):.2f} - {max(prod_sls):.2f}%")
        print(f"    R:R range: {min(prod_rrs):.2f} - {max(prod_rrs):.2f}")
        print(f"    Mean R:R: {np.mean(prod_rrs):.2f}")
        current_baseline_wr = np.mean([sl / (tp + sl) * 100
                                       for tp, sl in prod_tpsl.values()])
        print(f"    Mean baseline WR (random walk): {current_baseline_wr:.1f}%")
    except Exception as e:
        print(f"  Could not load production patterns: {e}")

    print(f"\n  R:R >= 1:1 study results:")
    print(f"    Patterns found: {p3['total_patterns']} "
          f"({p3['long_count']}L + {p3['short_count']}S)")
    print(f"    Mean WR: {p3['mean_wr']:.1f}%")
    print(f"    Mean WR Excess: {p3['mean_wr_excess']:.1f}pp "
          f"(baseline = 50% for symmetric)")
    print(f"    R:R distribution: {p3['rr_distribution']}")
    if p4.get('wf_results'):
        print(f"    WF: {'PASS' if p4['wf_pass'] else 'FAIL'} "
              f"({p4['wf_results']['positive_folds']}/3)")
        print(f"    Total OOS PnL: {p4['wf_results']['total_oos_pnl']}%")
    print(f"    Edge over random: {p5['edge_over_random_pp']:.1f}pp "
          f"(prod pass rate {p5['prod_filter_pass_rate']:.1f}% vs "
          f"random {p5['random_mean_pass_rate']:.1f}%)")

    # ---- VERDICT ----
    rr1_viable = (p3['total_patterns'] >= 10 and
                  p4.get('wf_pass', False) and
                  p5['edge_over_random_pp'] > 3.0)

    genuine_edge = p5['edge_over_random_pp']

    if rr1_viable:
        recommendation = (
            f"R:R >= 1:1 is VIABLE. {p3['total_patterns']} patterns with "
            f"genuine edge {genuine_edge:.1f}pp over random. "
            f"WF 3/3 PASS. Consider switching from R:R<1 to R:R>=1 strategy."
        )
    elif p3['total_patterns'] >= 5:
        recommendation = (
            f"R:R >= 1:1 shows SOME genuine edge ({genuine_edge:.1f}pp) "
            f"with {p3['total_patterns']} patterns, but "
            f"{'WF FAIL' if not p4.get('wf_pass', False) else 'insufficient differentiation from random'}. "
            f"Current R:R<1 strategy may still be better for stability."
        )
    else:
        recommendation = (
            f"R:R >= 1:1 is NOT VIABLE. Only {p3['total_patterns']} patterns pass "
            f"filters. Edge over random: {genuine_edge:.1f}pp. "
            f"The pattern pipeline relies on R:R<1 structure for apparent edge."
        )

    print(f"\n  VERDICT: {recommendation}")

    verdict = {
        'rr1_viable': rr1_viable,
        'genuine_edge_pp': round(genuine_edge, 1),
        'recommendation': recommendation,
    }

    # ---- SAVE ----
    # Remove raw trade data from selected_patterns to keep JSON manageable
    clean_patterns = []
    for sp in p3['selected_patterns']:
        cp = {k: v for k, v in sp.items() if k != 'trades_raw'}
        clean_patterns.append(cp)

    # Clean Phase 1/2 grid results (remove individual pattern lists for size)
    p1_clean = {
        'grid_summary': [{
            'tp': r['tp'], 'sl': r['sl'], 'rr': r['rr'],
            'baseline_wr': r['baseline_wr'], 'n_tested': r['n_tested'],
            'n_passing': r['n_passing'], 'pass_rate': r['pass_rate'],
            'mean_wr': r['mean_wr'], 'mean_edge': r['mean_edge'],
        } for r in p1['grid_results']],
        'total_passing': p1['total_passing'],
    }
    p2_clean = {
        'grid_summary': [{
            'tp': r['tp'], 'sl': r['sl'], 'rr': r['rr'],
            'baseline_wr': r['baseline_wr'], 'n_tested': r['n_tested'],
            'n_passing': r['n_passing'], 'pass_rate': r['pass_rate'],
            'mean_wr': r['mean_wr'], 'mean_edge': r['mean_edge'],
        } for r in p2['grid_results']],
        'total_passing': p2['total_passing'],
    }

    output = {
        'study': 'rr_reversal',
        'generated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'data_file': DATA_FILE,
        'data_bars': n_bars,
        'neutral_window': {
            'bar_start': bar_start,
            'bar_end': bar_end,
            'days': round((bar_end - bar_start) / BARS_PER_DAY, 0),
        },
        'constants': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'max_bars': MAX_BARS,
            'timeout_bars': TIMEOUT_BARS,
            'edge_threshold_rr1': EDGE_THRESHOLD_RR1,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'clamp': [CLAMP_LO, CLAMP_HI],
        },
        'phase1_symmetric': p1_clean,
        'phase2_favorable': p2_clean,
        'phase3_per_pattern_optimal': {
            'total_patterns': p3['total_patterns'],
            'long_count': p3['long_count'],
            'short_count': p3['short_count'],
            'mean_wr': p3['mean_wr'],
            'mean_wr_excess': p3['mean_wr_excess'],
            'rr_distribution': p3['rr_distribution'],
            'tp_range': p3['tp_range'],
            'sl_range': p3['sl_range'],
            'selected_patterns': clean_patterns,
        },
        'phase4_portfolio_wf': p4,
        'phase5_random_comparison': p5,
        'verdict': verdict,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    return output


if __name__ == '__main__':
    main()
