#!/usr/bin/env python3
"""
Regime-Robust Pattern Study
============================
Find patterns profitable across ALL market regimes (bull/bear/sideways).

Concern: 35 current patterns (74% SHORT) learned on mostly bearish data.
         If bull market comes, massive losses expected.
Solution: Filter for patterns that have genuine edge in EVERY regime.

Phases:
  1. Regime classification (rolling 7d return → BULL/BEAR/SIDE)
  2. Current 35pat — per-regime breakdown (identify vulnerable patterns)
  3. Full 59pat universe — per-regime compact grid search
  4. Regime-robust portfolio construction
  5. WF 3-fold validation
  6. Regime stress test + recommendation

Standard Research Protocol:
  Entry: next bar open | Exit: intrabar high/low | Compound sizing
  Fee: 0.10% × 3x = 0.30% capital-space | Timeout: 864 bars → DROP
  SL >= 1.0% | min_trades >= 25 | WF: 3-fold expanding window
"""

import json
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from multi_position_diversification_study import (
    load_and_classify, build_signal_index, compute_atr_ratio,
    calc_stats, find_overlap_bar,
)
from tpsl_regime_bias_study import (
    backtest_pattern_trades, build_custom_schedule,
    load_patterns_from_file,
)
from long_restore_and_direction_cap_study import (
    simulate_hedge_with_cap, run_wf_3fold_with_cap,
)

# ============================================================
# CONSTANTS
# ============================================================
BASE = os.path.join(SCRIPT_DIR, '..', '..')
DATA_720 = os.path.join(BASE, 'data', 'btc_5m_720days_binance.csv')
RESULTS_DIR = os.path.join(BASE, 'results')
CURRENT_PAT = os.path.join(RESULTS_DIR, 'dynamic_patterns.json')
BACKUP_59 = os.path.join(RESULTS_DIR, 'dynamic_patterns_59pat_backup.json')

LEVERAGE = 3
ATR_PERIOD = 14
ATR_WINDOW = 576
TIMEOUT_T864 = 864
MAX_POSITIONS = 9
DIRECTION_CAP = 6
BARS_PER_DAY = 288

# Regime classification
REGIME_WINDOW = 2016   # 7 days of 5m bars
REGIME_BULL_THR = 2.0  # +2% rolling return = BULL
REGIME_BEAR_THR = -2.0 # -2% rolling return = BEAR

# Compact TP/SL grids
TP_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
SL_GRID = [1.00, 1.25, 1.50, 1.75, 2.00, 2.50]

# Min trades per regime for reliability
MIN_TRADES_PER_REGIME = 5
MIN_TRADES_TOTAL = 25


# ============================================================
# REGIME CLASSIFICATION
# ============================================================
def classify_regimes(closes, window=REGIME_WINDOW):
    """Classify each bar into BULL/BEAR/SIDE based on rolling return."""
    n = len(closes)
    regimes = np.full(n, 1, dtype=np.int8)  # 1=SIDE default

    for i in range(window, n):
        ret = (closes[i] - closes[i - window]) / closes[i - window] * 100
        if ret > REGIME_BULL_THR:
            regimes[i] = 2   # BULL
        elif ret < REGIME_BEAR_THR:
            regimes[i] = 0   # BEAR
        else:
            regimes[i] = 1   # SIDE

    return regimes


REGIME_NAMES = {0: 'BEAR', 1: 'SIDE', 2: 'BULL'}


# ============================================================
# PER-REGIME BACKTEST
# ============================================================
def backtest_per_regime(sig_bars, direction_int, tp_pct, sl_pct,
                        opens, highs, lows, n_bars,
                        atr_ratio, regimes, lo_bar=0, hi_bar=None):
    """Backtest and split results by regime of entry bar."""
    trades = backtest_pattern_trades(
        sig_bars, direction_int, tp_pct, sl_pct,
        opens, highs, lows, n_bars,
        atr_ratio=atr_ratio, use_atr=True,
        lo_bar=lo_bar, hi_bar=hi_bar,
        timeout_bars=TIMEOUT_T864)

    regime_trades = {0: [], 1: [], 2: []}  # BEAR, SIDE, BULL
    for t in trades:
        entry_bar = t['entry_bar']
        if entry_bar < len(regimes):
            r = regimes[entry_bar]
            regime_trades[r].append(t)

    return trades, regime_trades


def compute_regime_stats(regime_trades, tp_pct, sl_pct):
    """Compute WR, WR Excess, PnL stats per regime."""
    rw_wr = sl_pct / (tp_pct + sl_pct) * 100  # Random Walk WR
    stats = {}
    for r_id, trades in regime_trades.items():
        n = len(trades)
        if n == 0:
            stats[r_id] = {'trades': 0, 'wr': 0, 'wr_excess': 0, 'avg_pnl': 0}
            continue
        wins = sum(1 for t in trades if t['pnl'] > 0)
        wr = wins / n * 100
        wr_excess = wr - rw_wr
        avg_pnl = np.mean([t['pnl'] for t in trades])
        stats[r_id] = {
            'trades': n, 'wr': wr, 'wr_excess': wr_excess, 'avg_pnl': avg_pnl
        }
    return stats


def compact_grid_search_per_regime(sig_bars, direction_int, opens, highs, lows,
                                    n_bars, atr_ratio, regimes, overlap_bar):
    """Find compact TP/SL that maximizes min(WR Excess across regimes)."""
    best_tp, best_sl = None, None
    best_min_excess = -999
    best_stats = None
    best_total_trades = 0

    for tp in TP_GRID:
        for sl in SL_GRID:
            if sl < 1.0:
                continue
            trades, rt = backtest_per_regime(
                sig_bars, direction_int, tp, sl,
                opens, highs, lows, n_bars,
                atr_ratio, regimes, lo_bar=overlap_bar)

            total_trades = len(trades)
            if total_trades < MIN_TRADES_TOTAL:
                continue

            rs = compute_regime_stats(rt, tp, sl)

            # Count regimes with sufficient trades
            active_regimes = [r for r in [0, 1, 2] if rs[r]['trades'] >= MIN_TRADES_PER_REGIME]
            if len(active_regimes) < 2:
                continue

            # Min WR Excess across active regimes
            min_excess = min(rs[r]['wr_excess'] for r in active_regimes)

            # Prefer higher min_excess, break ties by total trades
            if (min_excess > best_min_excess or
                (min_excess == best_min_excess and total_trades > best_total_trades)):
                best_min_excess = min_excess
                best_tp = tp
                best_sl = sl
                best_stats = rs
                best_total_trades = total_trades

    return best_tp, best_sl, best_min_excess, best_stats


# ============================================================
# PHASES
# ============================================================
def phase1_regime_distribution(closes, regimes, overlap_bar, n_bars):
    print("\n" + "=" * 110)
    print("  PHASE 1: REGIME CLASSIFICATION")
    print(f"  Window: {REGIME_WINDOW} bars ({REGIME_WINDOW/BARS_PER_DAY:.0f} days)")
    print(f"  BULL: > +{REGIME_BULL_THR}%, BEAR: < {REGIME_BEAR_THR}%, SIDE: between")
    print("=" * 110)

    # Full 720d
    for label, lo, hi in [("Full 720d", REGIME_WINDOW, n_bars),
                           ("Post-overlap 270d", overlap_bar, n_bars)]:
        r_slice = regimes[lo:hi]
        total = len(r_slice)
        bear = np.sum(r_slice == 0)
        side = np.sum(r_slice == 1)
        bull = np.sum(r_slice == 2)
        print(f"\n  {label} ({total} bars = {total/BARS_PER_DAY:.0f} days):")
        print(f"    BULL: {bull:6d} ({bull/total*100:5.1f}%)")
        print(f"    SIDE: {side:6d} ({side/total*100:5.1f}%)")
        print(f"    BEAR: {bear:6d} ({bear/total*100:5.1f}%)")

    # Price trajectory
    start_price = closes[overlap_bar]
    end_price = closes[n_bars - 1]
    total_ret = (end_price - start_price) / start_price * 100
    print(f"\n  Post-overlap price: ${start_price:.0f} → ${end_price:.0f} ({total_ret:+.1f}%)")

    return {
        'regime_window': REGIME_WINDOW,
        'bull_thr': REGIME_BULL_THR,
        'bear_thr': REGIME_BEAR_THR,
    }


def phase2_current_patterns(current_patterns, sig_index, opens, highs, lows,
                             n_bars, atr_ratio, regimes, overlap_bar):
    print("\n" + "=" * 110)
    print("  PHASE 2: CURRENT 35 PATTERNS — PER-REGIME BREAKDOWN")
    print("=" * 110)

    print(f"\n  {'Pattern':<18} {'Dir':>5} | {'TP':>5} {'SL':>5} | "
          f"{'BULL_N':>6} {'BULL_WRx':>8} | {'SIDE_N':>6} {'SIDE_WRx':>8} | "
          f"{'BEAR_N':>6} {'BEAR_WRx':>8} | {'Robust':>6}")
    print("  " + "-" * 105)

    results = []
    robust_count = 0
    vulnerable = {'bull_only': [], 'bear_only': [], 'mixed': []}

    for p in current_patterns:
        pat = p['pattern']
        direction = p['direction']
        tp = p['tp']
        sl = p['sl']
        dir_int = 1 if direction == 'LONG' else -1

        bars = sig_index.get(pat, [])
        if not bars:
            continue

        _, rt = backtest_per_regime(
            bars, dir_int, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio, regimes, lo_bar=overlap_bar)
        rs = compute_regime_stats(rt, tp, sl)

        # Check robustness: WR Excess > 0 in all regimes with sufficient trades
        active = [r for r in [0, 1, 2] if rs[r]['trades'] >= MIN_TRADES_PER_REGIME]
        is_robust = (len(active) >= 2 and
                     all(rs[r]['wr_excess'] > 0 for r in active))
        if is_robust:
            robust_count += 1

        # Classify vulnerability
        if not is_robust and len(active) >= 2:
            pos_regimes = [r for r in active if rs[r]['wr_excess'] > 0]
            neg_regimes = [r for r in active if rs[r]['wr_excess'] <= 0]
            neg_names = [REGIME_NAMES[r] for r in neg_regimes]
            result_cat = 'vulnerable_in_' + '_'.join(neg_names)
        else:
            result_cat = 'robust' if is_robust else 'insufficient_data'

        r_tag = "YES" if is_robust else "no"
        print(f"  {pat:<18} {direction:>5} | {tp:5.2f} {sl:5.2f} | "
              f"{rs[2]['trades']:6d} {rs[2]['wr_excess']:+7.1f}pp | "
              f"{rs[1]['trades']:6d} {rs[1]['wr_excess']:+7.1f}pp | "
              f"{rs[0]['trades']:6d} {rs[0]['wr_excess']:+7.1f}pp | "
              f"{r_tag:>6}")

        results.append({
            'pattern': pat, 'direction': direction, 'tp': tp, 'sl': sl,
            'regime_stats': {REGIME_NAMES[k]: v for k, v in rs.items()},
            'is_robust': is_robust, 'category': result_cat,
        })

    print(f"\n  Regime-robust: {robust_count}/{len(current_patterns)}")
    print(f"  Vulnerable: {len(current_patterns) - robust_count}/{len(current_patterns)}")

    return results


def phase3_universe_screening(all_patterns, sig_index, opens, highs, lows,
                               n_bars, atr_ratio, regimes, overlap_bar):
    print("\n" + "=" * 110)
    print("  PHASE 3: 59-PATTERN UNIVERSE — REGIME-ROBUST COMPACT GRID SEARCH")
    print(f"  TP grid: {TP_GRID}")
    print(f"  SL grid: {SL_GRID}")
    print(f"  Criterion: min(WR Excess across regimes) > 0, min {MIN_TRADES_PER_REGIME} trades/regime")
    print("=" * 110)

    print(f"\n  {'Pattern':<18} {'Dir':>5} | {'TP':>5} {'SL':>5} {'minExc':>7} | "
          f"{'BULL_N':>6} {'BULL_WRx':>8} | {'SIDE_N':>6} {'SIDE_WRx':>8} | "
          f"{'BEAR_N':>6} {'BEAR_WRx':>8} | {'Robust':>6}")
    print("  " + "-" * 115)

    robust_patterns = []
    all_results = []

    for p in all_patterns:
        pat = p['pattern']
        direction = p['direction']
        dir_int = 1 if direction == 'LONG' else -1

        bars = sig_index.get(pat, [])
        if not bars:
            continue

        tp, sl, min_exc, rs = compact_grid_search_per_regime(
            bars, dir_int, opens, highs, lows,
            n_bars, atr_ratio, regimes, overlap_bar)

        if tp is None:
            continue

        is_robust = min_exc > 0

        r_tag = "YES" if is_robust else "no"
        bull_s = rs[2] if rs else {'trades': 0, 'wr_excess': 0}
        side_s = rs[1] if rs else {'trades': 0, 'wr_excess': 0}
        bear_s = rs[0] if rs else {'trades': 0, 'wr_excess': 0}

        print(f"  {pat:<18} {direction:>5} | {tp:5.2f} {sl:5.2f} {min_exc:+6.1f}pp | "
              f"{bull_s['trades']:6d} {bull_s['wr_excess']:+7.1f}pp | "
              f"{side_s['trades']:6d} {side_s['wr_excess']:+7.1f}pp | "
              f"{bear_s['trades']:6d} {bear_s['wr_excess']:+7.1f}pp | "
              f"{r_tag:>6}")

        entry = {
            'pattern': pat, 'direction': direction,
            'tp': tp, 'sl': sl, 'min_wr_excess': min_exc,
            'regime_stats': {REGIME_NAMES[k]: v for k, v in rs.items()},
            'is_robust': is_robust,
        }
        all_results.append(entry)
        if is_robust:
            robust_patterns.append(entry)

    n_long = sum(1 for p in robust_patterns if p['direction'] == 'LONG')
    n_short = sum(1 for p in robust_patterns if p['direction'] == 'SHORT')
    print(f"\n  Regime-robust: {len(robust_patterns)}/{len(all_results)} "
          f"({n_long}L + {n_short}S)")

    return robust_patterns, all_results


def phase4_portfolio(current_pats, robust_pats, sig_index, opens, highs, lows,
                     n_bars, atr_ratio, overlap_bar):
    print("\n" + "=" * 110)
    print("  PHASE 4: PORTFOLIO COMPARISON (Hedge N=9, cap=6, T864)")
    print("=" * 110)

    configs = {}

    # Current 35pat
    current_list = [(p['pattern'], p['direction'], p['tp'], p['sl']) for p in current_pats]
    configs['current_35pat'] = current_list

    # Regime-robust subset of current
    robust_current = [p for p in current_pats
                      if any(rp['pattern'] == p['pattern'] and rp['direction'] == p['direction']
                             for rp in robust_pats)]
    if robust_current:
        configs['robust_current'] = [(p['pattern'], p['direction'], p['tp'], p['sl'])
                                      for p in robust_current]

    # Full regime-robust (from 59pat universe, new compact TP/SL)
    configs['robust_59'] = [(p['pattern'], p['direction'], p['tp'], p['sl'])
                             for p in robust_pats]

    results = {}
    for label, pat_list in configs.items():
        if not pat_list:
            continue

        # Build schedule
        sched = {}
        for pat, direction, tp, sl in pat_list:
            dir_int = 1 if direction == 'LONG' else -1
            bars = sig_index.get(pat, [])
            for b in bars:
                entry_bar = b + 1
                if entry_bar >= n_bars:
                    continue
                entry_price = opens[entry_bar]
                if entry_price <= 0:
                    continue
                if entry_bar not in sched:
                    sched[entry_bar] = []
                sched[entry_bar].append((pat, dir_int, tp, sl, entry_price, b))

        # Simulate
        trades = simulate_hedge_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched,
            opens, highs, lows, overlap_bar, n_bars, TIMEOUT_T864)

        port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in trades]
        stats = calc_stats(port)

        n_long = sum(1 for p in pat_list if p[1] == 'LONG')
        n_short = sum(1 for p in pat_list if p[1] == 'SHORT')
        n_days = (n_bars - overlap_bar) / BARS_PER_DAY
        tpd = stats['trades'] / n_days if n_days > 0 else 0
        ratio = stats['cmp_pnl'] / stats['cmp_mdd'] if stats['cmp_mdd'] > 0 else 0

        print(f"\n  [{label}] {len(pat_list)}pat ({n_long}L+{n_short}S)")
        print(f"    Trades: {stats['trades']} ({tpd:.2f}/day)")
        print(f"    WR: {stats['wr']:.1f}%, PnL: {stats['cmp_pnl']:+.1f}%, "
              f"MDD: {stats['cmp_mdd']:.1f}%")
        print(f"    PnL/MDD: {ratio:.2f}")

        results[label] = {
            'patterns': len(pat_list), 'long': n_long, 'short': n_short,
            'trades': stats['trades'], 'trades_per_day': round(tpd, 2),
            'wr': round(stats['wr'], 1), 'pnl': round(stats['cmp_pnl'], 1),
            'mdd': round(stats['cmp_mdd'], 1), 'pnl_mdd': round(ratio, 2),
            'pat_list': [(p[0], p[1], p[2], p[3]) for p in pat_list],
        }

    return results


def phase5_wf(portfolio_results, sig_index, opens, highs, lows,
              n_bars, atr_ratio, overlap_bar):
    print("\n" + "=" * 110)
    print("  PHASE 5: WF 3-FOLD VALIDATION")
    print("=" * 110)

    wf_results = {}
    for label, data in portfolio_results.items():
        pat_list = data['pat_list']
        # Build schedule
        sched = {}
        for pat, direction, tp, sl in pat_list:
            dir_int = 1 if direction == 'LONG' else -1
            bars = sig_index.get(pat, [])
            for b in bars:
                entry_bar = b + 1
                if entry_bar >= n_bars:
                    continue
                entry_price = opens[entry_bar]
                if entry_price <= 0:
                    continue
                if entry_bar not in sched:
                    sched[entry_bar] = []
                sched[entry_bar].append((pat, dir_int, tp, sl, entry_price, b))

        folds, verdict = run_wf_3fold_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched,
            opens, highs, lows, overlap_bar, n_bars, TIMEOUT_T864)

        print(f"\n  [{label}] {verdict}")
        for f in folds:
            print(f"    Fold {f['fold']}: trades={f['oos_trades']}, "
                  f"WR={f['oos_wr']:.1f}%, PnL={f['oos_pnl']:+.1f}%, "
                  f"MDD={f['oos_mdd']:.1f}%")

        wf_results[label] = {
            'folds': [{k: v for k, v in f.items()} for f in folds],
            'verdict': verdict,
        }

    return wf_results


def phase6_stress_test(current_pats, robust_pats, sig_index, opens, highs, lows,
                        n_bars, atr_ratio, regimes, overlap_bar):
    print("\n" + "=" * 110)
    print("  PHASE 6: REGIME STRESS TEST")
    print("  What happens to each portfolio in pure BULL vs pure BEAR periods?")
    print("=" * 110)

    # Find pure regime periods (>80% same regime in 2000-bar windows)
    window_size = 2000
    regime_periods = {'BULL': [], 'BEAR': [], 'SIDE': []}

    for start in range(overlap_bar, n_bars - window_size, window_size // 2):
        end = start + window_size
        r_window = regimes[start:end]
        for r_id, r_name in REGIME_NAMES.items():
            pct = np.sum(r_window == r_id) / len(r_window) * 100
            if pct > 70:
                regime_periods[r_name].append((start, end, pct))

    for r_name, periods in regime_periods.items():
        print(f"\n  {r_name} periods (>70%): {len(periods)} windows")

    # Test portfolios in each regime period
    portfolios = {
        'current_35': [(p['pattern'], p['direction'], p['tp'], p['sl']) for p in current_pats],
        'robust': [(p['pattern'], p['direction'], p['tp'], p['sl']) for p in robust_pats],
    }

    for r_name in ['BULL', 'BEAR']:
        periods = regime_periods.get(r_name, [])
        if not periods:
            print(f"\n  No {r_name} periods found → cannot stress test")
            continue

        print(f"\n  === {r_name} REGIME STRESS TEST ===")
        for port_name, pat_list in portfolios.items():
            if not pat_list:
                continue

            total_trades = 0
            total_wins = 0
            total_pnl = 0.0

            for start, end, _ in periods:
                sched = {}
                for pat, direction, tp, sl in pat_list:
                    dir_int = 1 if direction == 'LONG' else -1
                    bars = sig_index.get(pat, [])
                    for b in bars:
                        entry_bar = b + 1
                        if start <= entry_bar < end and entry_bar < n_bars:
                            entry_price = opens[entry_bar]
                            if entry_price <= 0:
                                continue
                            if entry_bar not in sched:
                                sched[entry_bar] = []
                            sched[entry_bar].append((pat, dir_int, tp, sl, entry_price, b))

                trades = simulate_hedge_with_cap(
                    MAX_POSITIONS, DIRECTION_CAP, sched,
                    opens, highs, lows, start, end, TIMEOUT_T864)

                for t in trades:
                    total_trades += 1
                    if t['scaled_pnl'] > 0:
                        total_wins += 1
                    total_pnl += t['scaled_pnl']

            wr = total_wins / total_trades * 100 if total_trades > 0 else 0
            print(f"    [{port_name}] {r_name}: {total_trades} trades, "
                  f"WR={wr:.1f}%, total_pnl_sum={total_pnl:+.2f}%")

    return regime_periods


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    # --- Load data ---
    print("Loading 720-day data...")
    df = load_and_classify(DATA_720)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n_bars = len(df)
    print(f"  {n_bars} bars loaded")

    sig_index = build_signal_index(df['rctype'].values, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)
    overlap_bar = find_overlap_bar(df)
    print(f"  Overlap bar: {overlap_bar}")

    # Regime classification
    regimes = classify_regimes(closes)
    print(f"  Regimes classified ({REGIME_WINDOW}-bar window)")

    # Load current 35pat
    with open(CURRENT_PAT) as f:
        cur_data = json.load(f)
    current_patterns = []
    pd = cur_data.get('pattern_details', {})
    for key, info in pd.items():
        current_patterns.append({
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': info['tp'],
            'sl': info['sl'],
        })
    print(f"  Current: {len(current_patterns)} patterns")

    # Load 59pat universe
    all_59_patterns = []
    if os.path.exists(BACKUP_59):
        with open(BACKUP_59) as f:
            data59 = json.load(f)
        pd59 = data59.get('pattern_details', {})
        for key, info in pd59.items():
            all_59_patterns.append({
                'pattern': info['pattern'],
                'direction': info['direction'],
                'tp': info['tp'],
                'sl': info['sl'],
            })
        print(f"  59pat universe: {len(all_59_patterns)} patterns")
    else:
        print(f"  WARNING: 59pat backup not found, using current only")
        all_59_patterns = current_patterns

    # === PHASE 1 ===
    p1 = phase1_regime_distribution(closes, regimes, overlap_bar, n_bars)

    # === PHASE 2 ===
    p2 = phase2_current_patterns(
        current_patterns, sig_index, opens, highs, lows,
        n_bars, atr_ratio, regimes, overlap_bar)

    # === PHASE 3 ===
    robust_pats, all_screened = phase3_universe_screening(
        all_59_patterns, sig_index, opens, highs, lows,
        n_bars, atr_ratio, regimes, overlap_bar)

    # === PHASE 4 ===
    p4 = phase4_portfolio(current_patterns, robust_pats, sig_index,
                          opens, highs, lows, n_bars, atr_ratio, overlap_bar)

    # === PHASE 5 ===
    p5 = phase5_wf(p4, sig_index, opens, highs, lows, n_bars, atr_ratio, overlap_bar)

    # === PHASE 6 ===
    p6 = phase6_stress_test(current_patterns, robust_pats, sig_index,
                            opens, highs, lows, n_bars, atr_ratio, regimes, overlap_bar)

    # === SUMMARY ===
    print("\n" + "=" * 110)
    print("  FINAL SUMMARY")
    print("=" * 110)

    # Best config
    best_label = None
    best_ratio = 0
    for label, data in p4.items():
        wf = p5.get(label, {}).get('verdict', 'FAIL')
        if 'PASS' in wf and data['pnl_mdd'] > best_ratio:
            best_ratio = data['pnl_mdd']
            best_label = label

    if best_label:
        d = p4[best_label]
        print(f"\n  BEST: {best_label}")
        print(f"    Patterns: {d['patterns']} ({d['long']}L+{d['short']}S)")
        print(f"    Trades: {d['trades']} ({d['trades_per_day']}/day)")
        print(f"    WR: {d['wr']}%, PnL: {d['pnl']:+.1f}%, MDD: {d['mdd']:.1f}%")
        print(f"    PnL/MDD: {d['pnl_mdd']:.2f}")
        print(f"    WF: {p5[best_label]['verdict']}")
    else:
        print("\n  No configuration passed WF 3/3")

    # Robust pattern count
    n_robust = len(robust_pats)
    n_robust_l = sum(1 for p in robust_pats if p['direction'] == 'LONG')
    n_robust_s = sum(1 for p in robust_pats if p['direction'] == 'SHORT')
    print(f"\n  Regime-robust patterns: {n_robust} ({n_robust_l}L+{n_robust_s}S)")

    # Current vulnerability
    vulnerable = [p for p in p2 if not p['is_robust']]
    print(f"  Current vulnerable patterns: {len(vulnerable)}/{len(current_patterns)}")
    for v in vulnerable:
        rs = v['regime_stats']
        worst = min(rs.items(), key=lambda x: x[1]['wr_excess'])
        print(f"    {v['pattern']:18s} {v['direction']:>5} — worst: {worst[0]} "
              f"WRx={worst[1]['wr_excess']:+.1f}pp ({worst[1]['trades']} trades)")

    # Save results
    output = {
        'phase1_regime': p1,
        'phase2_current': [{k: v for k, v in p.items()} for p in p2],
        'phase3_robust': [{k: v for k, v in p.items()} for p in robust_pats],
        'phase3_all': [{k: v for k, v in p.items()} for p in all_screened],
        'phase4_portfolios': {k: {kk: vv for kk, vv in v.items() if kk != 'pat_list'}
                              for k, v in p4.items()},
        'phase5_wf': {k: v for k, v in p5.items()},
        'best': best_label,
    }

    out_path = os.path.join(RESULTS_DIR, 'regime_robust_pattern_study.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
