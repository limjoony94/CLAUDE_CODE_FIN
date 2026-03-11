#!/usr/bin/env python3
"""
Natural Adaptation Study
========================
Analyze whether signal frequency naturally shifts with market regime.

User insight: "강제로 균형은 이상한데? 상승장이면 long이 자연스럽게 많아져야 할거고
하락장이면 short이 자연스럽게 많을거야. 문제는, 단기 상승장 단기 하락장에서도
제대로 동작해야 하는거지"

Phases:
  1. Per-pattern signal frequency by regime (BULL/SIDE/BEAR)
  2. Natural adaptation score — does signal frequency correlate with direction?
  3. Problem pattern identification — SHORT patterns firing in BULL
  4. Short-term (1-week) regime window simulation
  5. Adaptive portfolio construction — naturally adapting patterns only
  6. WF validation + recommendation

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
from regime_robust_pattern_study import classify_regimes, REGIME_NAMES

# ============================================================
# CONSTANTS
# ============================================================
BASE = os.path.join(SCRIPT_DIR, '..', '..')
DATA_720 = os.path.join(BASE, 'data', 'btc_5m_720days_binance.csv')
RESULTS_DIR = os.path.join(BASE, 'results')
CURRENT_PAT = os.path.join(RESULTS_DIR, 'dynamic_patterns.json')
BACKUP_59 = os.path.join(RESULTS_DIR, 'dynamic_patterns_59pat_backup.json')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'natural_adaptation_study.json')

LEVERAGE = 3
ATR_PERIOD = 14
ATR_WINDOW = 576
TIMEOUT_T864 = 864
MAX_POSITIONS = 9
DIRECTION_CAP = 6
BARS_PER_DAY = 288

REGIME_WINDOW = 2016
REGIME_BULL_THR = 2.0
REGIME_BEAR_THR = -2.0

# Short-term regime: 1-week = 2016 bars
SHORT_REGIME_WINDOW = 2016
# But we also look at 3-day windows for really short-term
VERY_SHORT_WINDOW = 864  # 3 days

# Compact TP/SL grids
TP_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
SL_GRID = [1.00, 1.25, 1.50, 1.75, 2.00, 2.50]
MIN_TRADES_TOTAL = 25


# ============================================================
# PHASE 1: SIGNAL FREQUENCY BY REGIME
# ============================================================
def phase1_signal_frequency(patterns, sig_index, regimes, overlap_bar, n_bars):
    """Count how many signals each pattern fires in each regime."""
    print("\n" + "=" * 110)
    print("  PHASE 1: PER-PATTERN SIGNAL FREQUENCY BY REGIME")
    print("  Question: Do LONG patterns fire more in BULL? SHORT in BEAR?")
    print("=" * 110)

    # Count regime bars (post-overlap) for normalization
    post_regimes = regimes[overlap_bar:n_bars]
    regime_bar_counts = {
        0: int(np.sum(post_regimes == 0)),
        1: int(np.sum(post_regimes == 1)),
        2: int(np.sum(post_regimes == 2)),
    }
    total_bars = len(post_regimes)
    print(f"\n  Post-overlap regime bars: "
          f"BULL={regime_bar_counts[2]} ({regime_bar_counts[2]/total_bars*100:.1f}%), "
          f"SIDE={regime_bar_counts[1]} ({regime_bar_counts[1]/total_bars*100:.1f}%), "
          f"BEAR={regime_bar_counts[0]} ({regime_bar_counts[0]/total_bars*100:.1f}%)")

    print(f"\n  {'Pattern':<18} {'Dir':>5} | {'Total':>5} | "
          f"{'BULL':>5} {'B_rate':>7} | {'SIDE':>5} {'S_rate':>7} | "
          f"{'BEAR':>5} {'BR_rate':>7} | {'Adapt':>7}")
    print("  " + "-" * 105)

    results = []
    long_totals = {0: 0, 1: 0, 2: 0}
    short_totals = {0: 0, 1: 0, 2: 0}

    for p in patterns:
        pat = p['pattern']
        direction = p['direction']
        bars = sig_index.get(pat, [])

        # Count signals per regime (post-overlap only)
        regime_counts = {0: 0, 1: 0, 2: 0}
        total = 0
        for b in bars:
            if overlap_bar <= b < n_bars:
                r = regimes[b]
                regime_counts[r] += 1
                total += 1
                if direction == 'LONG':
                    long_totals[r] += 1
                else:
                    short_totals[r] += 1

        # Signals per 1000 regime bars (normalized rate)
        rates = {}
        for r in [0, 1, 2]:
            if regime_bar_counts[r] > 0:
                rates[r] = regime_counts[r] / regime_bar_counts[r] * 1000
            else:
                rates[r] = 0

        # Natural adaptation score:
        # LONG: should fire MORE in BULL, LESS in BEAR → bull_rate - bear_rate
        # SHORT: should fire MORE in BEAR, LESS in BULL → bear_rate - bull_rate
        if direction == 'LONG':
            adapt_score = rates[2] - rates[0]  # bull - bear
        else:
            adapt_score = rates[0] - rates[2]  # bear - bull

        print(f"  {pat:<18} {direction:>5} | {total:5d} | "
              f"{regime_counts[2]:5d} {rates[2]:6.2f}‰ | "
              f"{regime_counts[1]:5d} {rates[1]:6.2f}‰ | "
              f"{regime_counts[0]:5d} {rates[0]:6.2f}‰ | "
              f"{adapt_score:+6.2f}")

        results.append({
            'pattern': pat, 'direction': direction,
            'tp': p.get('tp', 0), 'sl': p.get('sl', 0),
            'total_signals': total,
            'bull_signals': regime_counts[2],
            'side_signals': regime_counts[1],
            'bear_signals': regime_counts[0],
            'bull_rate': round(rates[2], 3),
            'side_rate': round(rates[1], 3),
            'bear_rate': round(rates[0], 3),
            'adaptation_score': round(adapt_score, 3),
        })

    # Aggregate LONG vs SHORT
    print(f"\n  === AGGREGATE SIGNAL FREQUENCY ===")
    for label, totals in [("LONG", long_totals), ("SHORT", short_totals)]:
        total_all = sum(totals.values())
        if total_all == 0:
            continue
        pct = {r: totals[r] / total_all * 100 for r in [0, 1, 2]}
        norm = {r: totals[r] / regime_bar_counts[r] * 1000 if regime_bar_counts[r] > 0 else 0
                for r in [0, 1, 2]}
        print(f"  {label:>6} signals: {total_all} total | "
              f"BULL {totals[2]} ({pct[2]:.1f}%, {norm[2]:.2f}‰) | "
              f"SIDE {totals[1]} ({pct[1]:.1f}%, {norm[1]:.2f}‰) | "
              f"BEAR {totals[0]} ({pct[0]:.1f}%, {norm[0]:.2f}‰)")

    # Key test: do LONG signals proportionally increase in BULL?
    long_bull_share = long_totals[2] / sum(long_totals.values()) * 100 if sum(long_totals.values()) > 0 else 0
    short_bear_share = short_totals[0] / sum(short_totals.values()) * 100 if sum(short_totals.values()) > 0 else 0
    bull_bar_share = regime_bar_counts[2] / total_bars * 100
    bear_bar_share = regime_bar_counts[0] / total_bars * 100

    print(f"\n  Natural Adaptation Test:")
    print(f"    BULL bars: {bull_bar_share:.1f}% of time")
    print(f"    LONG signals in BULL: {long_bull_share:.1f}% of all LONG signals "
          f"({'ADAPTED' if long_bull_share > bull_bar_share else 'NOT adapted'})")
    print(f"    BEAR bars: {bear_bar_share:.1f}% of time")
    print(f"    SHORT signals in BEAR: {short_bear_share:.1f}% of all SHORT signals "
          f"({'ADAPTED' if short_bear_share > bear_bar_share else 'NOT adapted'})")

    return results, regime_bar_counts


# ============================================================
# PHASE 2: PROBLEM PATTERN IDENTIFICATION
# ============================================================
def phase2_problem_patterns(phase1_results, regime_bar_counts):
    """Identify SHORT patterns that fire disproportionately in BULL (= dangerous)."""
    print("\n" + "=" * 110)
    print("  PHASE 2: PROBLEM PATTERN IDENTIFICATION")
    print("  Dangerous = SHORT pattern that fires MORE in BULL than in BEAR")
    print("=" * 110)

    total_bars = sum(regime_bar_counts.values())
    bull_share = regime_bar_counts[2] / total_bars

    dangerous = []
    safe = []
    neutral = []

    for p in phase1_results:
        direction = p['direction']
        bull_rate = p['bull_rate']
        bear_rate = p['bear_rate']
        adapt = p['adaptation_score']

        if direction == 'SHORT':
            # SHORT pattern that fires MORE in BULL than BEAR = dangerous
            if bull_rate > bear_rate * 1.2:  # 20% more in BULL
                dangerous.append(p)
            elif bear_rate > bull_rate * 1.2:  # 20% more in BEAR
                safe.append(p)
            else:
                neutral.append(p)
        else:
            # LONG pattern that fires MORE in BEAR than BULL = dangerous
            if bear_rate > bull_rate * 1.2:
                dangerous.append(p)
            elif bull_rate > bear_rate * 1.2:
                safe.append(p)
            else:
                neutral.append(p)

    print(f"\n  Naturally adapted (safe): {len(safe)} patterns")
    for p in sorted(safe, key=lambda x: x['adaptation_score'], reverse=True):
        print(f"    {p['pattern']:<18} {p['direction']:>5}  adapt={p['adaptation_score']:+.2f}  "
              f"BULL={p['bull_rate']:.2f}‰  BEAR={p['bear_rate']:.2f}‰")

    print(f"\n  Regime-neutral: {len(neutral)} patterns")
    for p in neutral:
        print(f"    {p['pattern']:<18} {p['direction']:>5}  adapt={p['adaptation_score']:+.2f}  "
              f"BULL={p['bull_rate']:.2f}‰  BEAR={p['bear_rate']:.2f}‰")

    print(f"\n  Counter-adapted (DANGEROUS): {len(dangerous)} patterns")
    for p in sorted(dangerous, key=lambda x: x['adaptation_score']):
        print(f"    {p['pattern']:<18} {p['direction']:>5}  adapt={p['adaptation_score']:+.2f}  "
              f"BULL={p['bull_rate']:.2f}‰  BEAR={p['bear_rate']:.2f}‰")

    return {'safe': safe, 'neutral': neutral, 'dangerous': dangerous}


# ============================================================
# PHASE 3: SHORT-TERM REGIME WINDOW ANALYSIS
# ============================================================
def phase3_short_term_windows(patterns, sig_index, opens, highs, lows,
                               n_bars, atr_ratio, regimes, overlap_bar):
    """Simulate portfolio in short (1-week) bull/bear windows."""
    print("\n" + "=" * 110)
    print("  PHASE 3: SHORT-TERM REGIME WINDOW SIMULATION")
    print("  1-week (2016 bars) windows, >70% same regime")
    print("=" * 110)

    # Find short-term regime windows
    window_size = SHORT_REGIME_WINDOW
    step = window_size // 4  # overlapping windows

    windows = {'BULL': [], 'BEAR': [], 'SIDE': []}
    for start in range(overlap_bar, n_bars - window_size, step):
        end = start + window_size
        r_window = regimes[start:end]
        for r_id, r_name in REGIME_NAMES.items():
            pct = np.sum(r_window == r_id) / len(r_window) * 100
            if pct > 70:
                windows[r_name].append((start, end, float(pct)))

    for r_name, wins in windows.items():
        print(f"  {r_name} windows (>70%): {len(wins)}")

    # Build schedule from all patterns
    sched_all = _build_schedule(patterns, sig_index, opens, n_bars)

    # Test in each window
    results = {}
    for r_name in ['BULL', 'BEAR', 'SIDE']:
        wins = windows[r_name]
        if not wins:
            print(f"\n  No {r_name} windows → skip")
            results[r_name] = {'windows': 0, 'avg_pnl': 0}
            continue

        window_pnls = []
        window_details = []

        for start, end, regime_pct in wins:
            trades = simulate_hedge_with_cap(
                MAX_POSITIONS, DIRECTION_CAP, sched_all,
                opens, highs, lows, start, end, TIMEOUT_T864)

            if not trades:
                continue

            total_pnl = sum(t['scaled_pnl'] for t in trades)
            n_trades = len(trades)
            long_trades = sum(1 for t in trades if t.get('direction', 0) == 1)
            short_trades = n_trades - long_trades
            wins_count = sum(1 for t in trades if t['scaled_pnl'] > 0)
            wr = wins_count / n_trades * 100 if n_trades > 0 else 0

            window_pnls.append(total_pnl)
            window_details.append({
                'start': int(start), 'end': int(end),
                'trades': n_trades, 'long': long_trades, 'short': short_trades,
                'wr': round(wr, 1), 'pnl': round(total_pnl, 2),
            })

        avg_pnl = np.mean(window_pnls) if window_pnls else 0
        median_pnl = np.median(window_pnls) if window_pnls else 0
        pct_positive = sum(1 for p in window_pnls if p > 0) / len(window_pnls) * 100 if window_pnls else 0

        # Direction composition in this regime's windows
        total_long = sum(d['long'] for d in window_details)
        total_short = sum(d['short'] for d in window_details)
        total_trades = total_long + total_short
        long_pct = total_long / total_trades * 100 if total_trades > 0 else 0

        print(f"\n  === {r_name} windows ({len(window_details)} with trades) ===")
        print(f"    Avg PnL/window: {avg_pnl:+.2f}%, Median: {median_pnl:+.2f}%")
        print(f"    Positive windows: {pct_positive:.1f}%")
        print(f"    Direction mix: {total_long}L + {total_short}S "
              f"({long_pct:.1f}% LONG)")
        if r_name == 'BULL':
            print(f"    → LONG signals naturally increase in BULL? "
                  f"{'YES' if long_pct > 30 else 'NO — still SHORT dominated'}")
        elif r_name == 'BEAR':
            print(f"    → SHORT signals naturally increase in BEAR? "
                  f"{'YES' if long_pct < 20 else 'Similar ratio'}")

        results[r_name] = {
            'windows': len(window_details),
            'avg_pnl': round(avg_pnl, 2),
            'median_pnl': round(median_pnl, 2),
            'pct_positive': round(pct_positive, 1),
            'long_pct': round(long_pct, 1),
            'total_long': total_long,
            'total_short': total_short,
            'window_details': window_details[:10],  # save top 10
        }

    return results


# ============================================================
# PHASE 4: ADAPTIVE PORTFOLIO — remove counter-adapted patterns
# ============================================================
def phase4_adaptive_portfolio(phase1_results, phase2_cats, sig_index,
                               opens, highs, lows, n_bars, atr_ratio,
                               regimes, overlap_bar):
    """Build portfolio excluding counter-adapted (dangerous) patterns."""
    print("\n" + "=" * 110)
    print("  PHASE 4: ADAPTIVE PORTFOLIO COMPARISON")
    print("=" * 110)

    # Portfolio configs
    all_patterns = phase1_results
    safe_neutral = phase2_cats['safe'] + phase2_cats['neutral']
    safe_only = phase2_cats['safe']

    configs = {
        'current_all': all_patterns,
        'safe_neutral': safe_neutral,
        'safe_only': safe_only,
    }

    results = {}
    for label, pat_list in configs.items():
        if not pat_list:
            print(f"\n  [{label}] No patterns → skip")
            continue

        n_long = sum(1 for p in pat_list if p['direction'] == 'LONG')
        n_short = sum(1 for p in pat_list if p['direction'] == 'SHORT')

        # Build schedule
        sched = _build_schedule_from_phase1(pat_list, sig_index, opens, n_bars)

        # Full simulation
        trades = simulate_hedge_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched,
            opens, highs, lows, overlap_bar, n_bars, TIMEOUT_T864)

        port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in trades]
        stats = calc_stats(port)
        n_days = (n_bars - overlap_bar) / BARS_PER_DAY
        tpd = stats['trades'] / n_days if n_days > 0 else 0
        ratio = stats['cmp_pnl'] / stats['cmp_mdd'] if stats['cmp_mdd'] > 0 else 0

        print(f"\n  [{label}] {len(pat_list)}pat ({n_long}L+{n_short}S)")
        print(f"    Trades: {stats['trades']} ({tpd:.2f}/day)")
        print(f"    WR: {stats['wr']:.1f}%, PnL: {stats['cmp_pnl']:+.1f}%, "
              f"MDD: {stats['cmp_mdd']:.1f}%")
        print(f"    PnL/MDD: {ratio:.2f}")

        # Test in regime windows
        window_size = SHORT_REGIME_WINDOW
        step = window_size // 4
        for r_name in ['BULL', 'BEAR']:
            r_id = 2 if r_name == 'BULL' else 0
            window_pnls = []
            for start in range(overlap_bar, n_bars - window_size, step):
                end = start + window_size
                r_window = regimes[start:end]
                pct = np.sum(r_window == r_id) / len(r_window) * 100
                if pct < 70:
                    continue
                w_trades = simulate_hedge_with_cap(
                    MAX_POSITIONS, DIRECTION_CAP, sched,
                    opens, highs, lows, start, end, TIMEOUT_T864)
                if w_trades:
                    window_pnls.append(sum(t['scaled_pnl'] for t in w_trades))

            if window_pnls:
                avg = np.mean(window_pnls)
                pct_pos = sum(1 for p in window_pnls if p > 0) / len(window_pnls) * 100
                print(f"    {r_name} windows: avg_pnl={avg:+.2f}%, positive={pct_pos:.0f}%")

        results[label] = {
            'patterns': len(pat_list), 'long': n_long, 'short': n_short,
            'trades': stats['trades'], 'trades_per_day': round(tpd, 2),
            'wr': round(stats['wr'], 1), 'pnl': round(stats['cmp_pnl'], 1),
            'mdd': round(stats['cmp_mdd'], 1), 'pnl_mdd': round(ratio, 2),
            'pat_list': [(p['pattern'], p['direction'], p['tp'], p['sl']) for p in pat_list],
        }

    return results


# ============================================================
# PHASE 5: ADAPTIVE PORTFOLIO FROM 59-PAT UNIVERSE
# ============================================================
def phase5_universe_adaptive(all_59_patterns, sig_index, opens, highs, lows,
                              n_bars, atr_ratio, regimes, overlap_bar):
    """Screen 59-pat universe for naturally adaptive patterns + grid search."""
    print("\n" + "=" * 110)
    print("  PHASE 5: 59-PAT UNIVERSE — NATURAL ADAPTATION SCREENING")
    print("  Filter: adaptation_score > 0 (signal frequency matches direction)")
    print("  + compact grid search for optimal TP/SL")
    print("=" * 110)

    regime_bar_counts = {
        0: int(np.sum(regimes[overlap_bar:n_bars] == 0)),
        1: int(np.sum(regimes[overlap_bar:n_bars] == 1)),
        2: int(np.sum(regimes[overlap_bar:n_bars] == 2)),
    }

    adaptive_patterns = []
    all_screened = []

    print(f"\n  {'Pattern':<18} {'Dir':>5} | {'Adapt':>7} | "
          f"{'TP':>5} {'SL':>5} {'WRx':>6} | {'Trades':>6} | {'Status':>10}")
    print("  " + "-" * 90)

    for p in all_59_patterns:
        pat = p['pattern']
        direction = p['direction']
        dir_int = 1 if direction == 'LONG' else -1
        bars = sig_index.get(pat, [])

        # Compute adaptation score
        regime_counts = {0: 0, 1: 0, 2: 0}
        for b in bars:
            if overlap_bar <= b < n_bars:
                regime_counts[regimes[b]] += 1

        rates = {}
        for r in [0, 1, 2]:
            if regime_bar_counts[r] > 0:
                rates[r] = regime_counts[r] / regime_bar_counts[r] * 1000
            else:
                rates[r] = 0

        if direction == 'LONG':
            adapt_score = rates[2] - rates[0]
        else:
            adapt_score = rates[0] - rates[2]

        # Compact grid search for best WR Excess
        best_tp, best_sl = None, None
        best_wrx = -999
        best_trades = 0

        for tp in TP_GRID:
            for sl in SL_GRID:
                if sl < 1.0:
                    continue
                trades = backtest_pattern_trades(
                    bars, dir_int, tp, sl,
                    opens, highs, lows, n_bars,
                    atr_ratio=atr_ratio, use_atr=True,
                    lo_bar=overlap_bar,
                    timeout_bars=TIMEOUT_T864)
                n = len(trades)
                if n < MIN_TRADES_TOTAL:
                    continue
                wins = sum(1 for t in trades if t['pnl'] > 0)
                wr = wins / n * 100
                rw_wr = sl / (tp + sl) * 100
                wrx = wr - rw_wr
                if wrx > best_wrx or (wrx == best_wrx and n > best_trades):
                    best_wrx = wrx
                    best_tp = tp
                    best_sl = sl
                    best_trades = n

        entry = {
            'pattern': pat, 'direction': direction,
            'adaptation_score': round(adapt_score, 3),
            'tp': best_tp, 'sl': best_sl,
            'wr_excess': round(best_wrx, 1) if best_wrx > -999 else None,
            'trades': best_trades,
        }
        all_screened.append(entry)

        # Criteria: adaptation_score > 0 AND WR Excess > 5pp AND min_trades
        is_adaptive = (adapt_score > 0 and best_wrx > 5 and best_trades >= MIN_TRADES_TOTAL
                       and best_tp is not None)

        status = "ADAPTIVE" if is_adaptive else "skip"
        if best_tp is not None:
            print(f"  {pat:<18} {direction:>5} | {adapt_score:+6.2f} | "
                  f"{best_tp:5.2f} {best_sl:5.2f} {best_wrx:+5.1f}pp | "
                  f"{best_trades:6d} | {status:>10}")
        else:
            print(f"  {pat:<18} {direction:>5} | {adapt_score:+6.2f} | "
                  f"{'--':>5} {'--':>5} {'--':>6} | {'--':>6} | no valid TP/SL")

        if is_adaptive:
            adaptive_patterns.append(entry)

    n_long = sum(1 for p in adaptive_patterns if p['direction'] == 'LONG')
    n_short = sum(1 for p in adaptive_patterns if p['direction'] == 'SHORT')
    print(f"\n  Naturally adaptive: {len(adaptive_patterns)}/59 "
          f"({n_long}L + {n_short}S)")

    return adaptive_patterns, all_screened


# ============================================================
# PHASE 6: WF VALIDATION + BULL/BEAR STRESS TEST
# ============================================================
def phase6_wf_and_stress(adaptive_pats, current_pats_phase1, sig_index,
                          opens, highs, lows, n_bars, atr_ratio,
                          regimes, overlap_bar):
    """WF validation + regime stress test for adaptive portfolio."""
    print("\n" + "=" * 110)
    print("  PHASE 6: WF VALIDATION + REGIME STRESS TEST")
    print("=" * 110)

    configs = {
        'current_35': [(p['pattern'], p['direction'], p['tp'], p['sl'])
                        for p in current_pats_phase1],
        'adaptive_59': [(p['pattern'], p['direction'], p['tp'], p['sl'])
                         for p in adaptive_pats if p['tp'] is not None],
    }

    results = {}
    for label, pat_list in configs.items():
        if not pat_list:
            print(f"\n  [{label}] No patterns → skip")
            continue

        n_long = sum(1 for p in pat_list if p[1] == 'LONG')
        n_short = sum(1 for p in pat_list if p[1] == 'SHORT')
        print(f"\n  === [{label}] {len(pat_list)}pat ({n_long}L+{n_short}S) ===")

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

        # Full IS performance
        trades = simulate_hedge_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched,
            opens, highs, lows, overlap_bar, n_bars, TIMEOUT_T864)
        port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in trades]
        stats = calc_stats(port)
        ratio = stats['cmp_pnl'] / stats['cmp_mdd'] if stats['cmp_mdd'] > 0 else 0
        print(f"  Full IS: trades={stats['trades']}, WR={stats['wr']:.1f}%, "
              f"PnL={stats['cmp_pnl']:+.1f}%, MDD={stats['cmp_mdd']:.1f}%, "
              f"PnL/MDD={ratio:.2f}")

        # WF
        folds, verdict = run_wf_3fold_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched,
            opens, highs, lows, overlap_bar, n_bars, TIMEOUT_T864)
        print(f"  WF: {verdict}")
        for f in folds:
            print(f"    Fold {f['fold']}: trades={f['oos_trades']}, "
                  f"WR={f['oos_wr']:.1f}%, PnL={f['oos_pnl']:+.1f}%, "
                  f"MDD={f['oos_mdd']:.1f}%")

        # Regime stress test
        window_size = SHORT_REGIME_WINDOW
        step = window_size // 4
        stress = {}
        for r_name in ['BULL', 'BEAR']:
            r_id = 2 if r_name == 'BULL' else 0
            window_pnls = []
            window_wr = []
            for start in range(overlap_bar, n_bars - window_size, step):
                end = start + window_size
                r_window = regimes[start:end]
                pct = np.sum(r_window == r_id) / len(r_window) * 100
                if pct < 70:
                    continue
                w_trades = simulate_hedge_with_cap(
                    MAX_POSITIONS, DIRECTION_CAP, sched,
                    opens, highs, lows, start, end, TIMEOUT_T864)
                if w_trades:
                    pnl = sum(t['scaled_pnl'] for t in w_trades)
                    window_pnls.append(pnl)
                    wins = sum(1 for t in w_trades if t['scaled_pnl'] > 0)
                    window_wr.append(wins / len(w_trades) * 100)

            if window_pnls:
                avg_pnl = np.mean(window_pnls)
                avg_wr = np.mean(window_wr)
                pct_pos = sum(1 for p in window_pnls if p > 0) / len(window_pnls) * 100
                print(f"  {r_name}: {len(window_pnls)} windows, "
                      f"avg_pnl={avg_pnl:+.2f}%, avg_WR={avg_wr:.1f}%, "
                      f"positive={pct_pos:.0f}%")
                stress[r_name] = {
                    'windows': len(window_pnls),
                    'avg_pnl': round(avg_pnl, 2),
                    'avg_wr': round(avg_wr, 1),
                    'pct_positive': round(pct_pos, 1),
                }
            else:
                stress[r_name] = {'windows': 0}

        results[label] = {
            'patterns': len(pat_list), 'long': n_long, 'short': n_short,
            'trades': stats['trades'], 'wr': round(stats['wr'], 1),
            'pnl': round(stats['cmp_pnl'], 1), 'mdd': round(stats['cmp_mdd'], 1),
            'pnl_mdd': round(ratio, 2),
            'wf_verdict': verdict,
            'wf_folds': [{k: v for k, v in f.items()} for f in folds],
            'stress': stress,
        }

    return results


# ============================================================
# HELPERS
# ============================================================
def _build_schedule(patterns, sig_index, opens, n_bars):
    """Build signal schedule from pattern list (dict with pattern/direction/tp/sl)."""
    sched = {}
    for p in patterns:
        pat = p['pattern']
        direction = p['direction']
        tp = p.get('tp', 1.5)
        sl = p.get('sl', 2.0)
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
    return sched


def _build_schedule_from_phase1(phase1_pats, sig_index, opens, n_bars):
    """Build schedule from phase1 results (same as _build_schedule)."""
    return _build_schedule(phase1_pats, sig_index, opens, n_bars)


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

    # Load current patterns (35pat v1.33.0)
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

    # === PHASE 1: Signal frequency by regime ===
    p1_results, regime_bar_counts = phase1_signal_frequency(
        current_patterns, sig_index, regimes, overlap_bar, n_bars)

    # === PHASE 2: Problem pattern identification ===
    p2_cats = phase2_problem_patterns(p1_results, regime_bar_counts)

    # === PHASE 3: Short-term regime windows ===
    p3_results = phase3_short_term_windows(
        current_patterns, sig_index, opens, highs, lows,
        n_bars, atr_ratio, regimes, overlap_bar)

    # === PHASE 4: Adaptive portfolio from current patterns ===
    p4_results = phase4_adaptive_portfolio(
        p1_results, p2_cats, sig_index,
        opens, highs, lows, n_bars, atr_ratio, regimes, overlap_bar)

    # === PHASE 5: Universe screening with adaptation filter ===
    p5_adaptive, p5_all = phase5_universe_adaptive(
        all_59_patterns, sig_index, opens, highs, lows,
        n_bars, atr_ratio, regimes, overlap_bar)

    # === PHASE 6: WF + Stress test ===
    p6_results = phase6_wf_and_stress(
        p5_adaptive, p1_results, sig_index,
        opens, highs, lows, n_bars, atr_ratio, regimes, overlap_bar)

    # === SUMMARY ===
    elapsed = time.time() - t_start
    print("\n" + "=" * 110)
    print(f"  STUDY COMPLETE ({elapsed:.1f}s)")
    print("=" * 110)

    # Key findings
    print("\n  KEY FINDINGS:")

    # Natural adaptation test
    long_total = sum(p['total_signals'] for p in p1_results if p['direction'] == 'LONG')
    short_total = sum(p['total_signals'] for p in p1_results if p['direction'] == 'SHORT')
    long_bull = sum(p['bull_signals'] for p in p1_results if p['direction'] == 'LONG')
    short_bear = sum(p['bear_signals'] for p in p1_results if p['direction'] == 'SHORT')
    long_bull_pct = long_bull / long_total * 100 if long_total > 0 else 0
    short_bear_pct = short_bear / short_total * 100 if short_total > 0 else 0
    total_bars = sum(regime_bar_counts.values())
    bull_pct = regime_bar_counts[2] / total_bars * 100
    bear_pct = regime_bar_counts[0] / total_bars * 100

    print(f"    1. Natural adaptation: LONG signals in BULL = {long_bull_pct:.1f}% "
          f"(baseline {bull_pct:.1f}%) → "
          f"{'YES' if long_bull_pct > bull_pct + 3 else 'WEAK/NO'}")
    print(f"       SHORT signals in BEAR = {short_bear_pct:.1f}% "
          f"(baseline {bear_pct:.1f}%) → "
          f"{'YES' if short_bear_pct > bear_pct + 3 else 'WEAK/NO'}")
    print(f"    2. Problem patterns: {len(p2_cats['dangerous'])} counter-adapted "
          f"(fire AGAINST their direction's regime)")
    print(f"    3. Safe patterns: {len(p2_cats['safe'])} naturally adapted")

    if 'adaptive_59' in p6_results:
        a = p6_results['adaptive_59']
        print(f"    4. Adaptive portfolio (59-universe): {a['patterns']}pat "
              f"({a['long']}L+{a['short']}S)")
        print(f"       PnL={a['pnl']:+.1f}%, MDD={a['mdd']:.1f}%, "
              f"PnL/MDD={a['pnl_mdd']:.2f}, WF={a['wf_verdict']}")
        if 'BULL' in a.get('stress', {}):
            bull = a['stress']['BULL']
            if bull.get('windows', 0) > 0:
                print(f"       BULL stress: avg_pnl={bull['avg_pnl']:+.2f}%, "
                      f"positive={bull['pct_positive']:.0f}%")

    if 'current_35' in p6_results:
        c = p6_results['current_35']
        print(f"    5. Current 35pat: PnL={c['pnl']:+.1f}%, PnL/MDD={c['pnl_mdd']:.2f}")
        if 'BULL' in c.get('stress', {}):
            bull = c['stress']['BULL']
            if bull.get('windows', 0) > 0:
                print(f"       BULL stress: avg_pnl={bull['avg_pnl']:+.2f}%, "
                      f"positive={bull['pct_positive']:.0f}%")

    # Save results
    output = {
        'study': 'Natural Adaptation Study',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phase1_signal_frequency': p1_results,
        'phase1_regime_bar_counts': {REGIME_NAMES[k]: v for k, v in regime_bar_counts.items()},
        'phase2_categories': {
            'safe': [p['pattern'] for p in p2_cats['safe']],
            'neutral': [p['pattern'] for p in p2_cats['neutral']],
            'dangerous': [p['pattern'] for p in p2_cats['dangerous']],
        },
        'phase3_short_term': p3_results,
        'phase4_adaptive': p4_results,
        'phase5_universe': {
            'adaptive_count': len(p5_adaptive),
            'adaptive_patterns': p5_adaptive,
            'all_screened': p5_all,
        },
        'phase6_wf_stress': p6_results,
        'elapsed_seconds': round(elapsed, 1),
    }

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=_convert)
    print(f"\n  Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
