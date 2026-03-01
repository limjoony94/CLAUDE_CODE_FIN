"""
Strategy Reform Study — 3 Hypotheses
=====================================
H1: R:R Reversal — Can we find profitable patterns with R:R >= 1.0/1.5/2.0?
H2: Position Count Reduction — What N and direction_cap minimize correlated damage?
H3: Live Data Regime Analysis — What market conditions predict win/loss?

Standard Research Protocol: compound sizing, LEVERAGE=3, fee=0.10% (notional),
entry=next bar open, ATR-scaled TP/SL, 3-fold expanding WF, MC 3-seed.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
SCANNER_DIR = os.path.join(PROJECT_ROOT, "scripts", "scanner")
sys.path.insert(0, SCANNER_DIR)

from pattern_scanner import (
    load_and_classify, build_signal_index, bt_signals_atr, bt_signals,
    compute_atr_ratio, mc_test, calc_stats, find_neutral_window,
    expanding_window_wf, scan_universe_range,
    FEE_PCT, LEVERAGE, MAX_BARS,
)

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "btc_5m_270days_reclassified.csv")
DYNAMIC_PATTERNS_FILE = os.path.join(PROJECT_ROOT, "results", "dynamic_patterns.json")
METRICS_FILE = os.path.join(PROJECT_ROOT, "results", "pattern_5m_metrics.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "strategy_reform_study.json")

# --- R:R Reversal Grid ---
RR_TP_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
RR_SL_GRID = [0.8, 1.0, 1.2, 1.5, 2.0]
MIN_RR_LEVELS = [1.0, 1.5, 2.0]

# Quality thresholds
EDGE_THRESHOLD = 10  # pp (lower for R:R>1 since baseline WR is lower)
MC_THRESHOLD = 0.01
MIN_TRADES = 25

# H2 scenarios
N_VALUES = [2, 3, 4, 5, 9]
CAP_VALUES = [1, 2, 3, 4, 7]


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# =========================================================================
# H1: R:R REVERSAL
# =========================================================================

def rr_constrained_scan(signal_index, opens, highs, lows, n_bars,
                        atr_ratio, min_rr, bar_start=0, bar_end=None):
    """Scan all patterns with R:R >= min_rr constraint."""
    if bar_end is None:
        bar_end = n_bars

    results = []
    for pat_name, all_sigs in signal_index.items():
        for direction in ["LONG", "SHORT"]:
            # Filter signals within bar range
            sigs = [s for s in all_sigs if bar_start <= s < bar_end]
            if len(sigs) < MIN_TRADES:
                continue

            best = None
            best_edge = -999

            for tp in RR_TP_GRID:
                for sl in RR_SL_GRID:
                    if tp / sl < min_rr:
                        continue

                    trades = bt_signals_atr(
                        sigs, direction, tp, sl,
                        opens, highs, lows, n_bars,
                        atr_ratio, clamp_lo=0.6, clamp_hi=1.7
                    )
                    if len(trades) < MIN_TRADES:
                        continue

                    stats = calc_stats(trades)
                    baseline_wr = sl / (tp + sl) * 100
                    edge = stats['wr'] - baseline_wr

                    if edge > best_edge:
                        best = {
                            'pattern': pat_name,
                            'direction': direction,
                            'tp': tp, 'sl': sl,
                            'rr': round(tp / sl, 2),
                            'trades': stats['trades'],
                            'wr': round(stats['wr'], 1),
                            'baseline_wr': round(baseline_wr, 1),
                            'edge': round(edge, 1),
                            'pnl': round(stats['pnl'], 1),
                            'mdd': round(stats['mdd'], 1),
                            'pf': round(stats['pf'], 2),
                            'pnls': [t[2] for t in trades],
                            'trade_list': trades,
                        }
                        best_edge = edge

            if best and best_edge >= EDGE_THRESHOLD:
                mc_p = mc_test(best['pnls'])
                if mc_p < MC_THRESHOLD:
                    best['mc_p'] = round(mc_p, 4)
                    results.append(best)

    return results


def run_h1(df, signal_index, opens, highs, lows, closes, n_bars, atr_ratio,
           bar_start, bar_end):
    """H1: R:R Reversal Study."""
    print_header("H1: R:R REVERSAL STUDY")

    h1_results = {}

    for min_rr in MIN_RR_LEVELS:
        print_section(f"H1-{min_rr}: R:R >= {min_rr}")
        t0 = time.time()

        patterns = rr_constrained_scan(
            signal_index, opens, highs, lows, n_bars,
            atr_ratio, min_rr, bar_start, bar_end
        )
        elapsed = time.time() - t0
        print(f"  Scan time: {elapsed:.1f}s")
        print(f"  Patterns found: {len(patterns)}")

        if not patterns:
            print("  No patterns found with these constraints.")
            h1_results[f'rr_{min_rr}'] = {'count': 0}
            continue

        # Direction breakdown
        n_long = sum(1 for p in patterns if p['direction'] == 'LONG')
        n_short = sum(1 for p in patterns if p['direction'] == 'SHORT')
        print(f"  LONG: {n_long} | SHORT: {n_short}")

        # Statistics
        wrs = [p['wr'] for p in patterns]
        edges = [p['edge'] for p in patterns]
        rrs = [p['rr'] for p in patterns]
        trades_list = [p['trades'] for p in patterns]

        print(f"  WR:    mean {np.mean(wrs):.1f}% | range [{min(wrs):.1f}, {max(wrs):.1f}]%")
        print(f"  Edge:  mean {np.mean(edges):.1f}pp | range [{min(edges):.1f}, {max(edges):.1f}]pp")
        print(f"  R:R:   mean {np.mean(rrs):.2f} | range [{min(rrs):.2f}, {max(rrs):.2f}]")
        print(f"  Trades: mean {np.mean(trades_list):.0f} | total {sum(trades_list)}")

        # Breakeven analysis
        for p in patterns:
            p['breakeven_wr'] = p['sl'] / (p['tp'] + p['sl']) * 100
        bev_wrs = [p['breakeven_wr'] for p in patterns]
        print(f"  Breakeven WR: mean {np.mean(bev_wrs):.1f}%")
        print(f"  WR - Breakeven: mean {np.mean(wrs) - np.mean(bev_wrs):+.1f}pp")

        # Portfolio (1-position, no overlap)
        all_trades = []
        for p in patterns:
            all_trades.extend(p['trade_list'])
        all_trades.sort(key=lambda t: t[0])

        # Simple no-overlap portfolio
        portfolio_trades = []
        last_exit = -1
        for t in all_trades:
            if t[0] > last_exit:
                portfolio_trades.append(t)
                last_exit = t[1]

        if portfolio_trades:
            pstats = calc_stats(portfolio_trades)
            print(f"\n  Portfolio (1-pos, no overlap):")
            print(f"    Trades: {pstats['trades']} | WR: {pstats['wr']:.1f}% | "
                  f"PnL: {pstats['pnl']:+.1f}% | MDD: {pstats['mdd']:.1f}% | "
                  f"PF: {pstats['pf']:.2f}")
            if pstats['mdd'] > 0:
                print(f"    PnL/MDD: {pstats['pnl']/pstats['mdd']:.2f}x")

        # TP/SL distribution
        tps = [p['tp'] for p in patterns]
        sls = [p['sl'] for p in patterns]
        print(f"\n  TP range: {min(tps):.1f}-{max(tps):.1f}% (median {np.median(tps):.1f}%)")
        print(f"  SL range: {min(sls):.1f}-{max(sls):.1f}% (median {np.median(sls):.1f}%)")

        # Top 10 patterns
        patterns.sort(key=lambda p: p['edge'], reverse=True)
        print(f"\n  Top 10 patterns (by edge):")
        print(f"  {'Pattern':<14} {'Dir':<6} {'TP':>4} {'SL':>4} {'R:R':>5} "
              f"{'WR':>5} {'Edge':>5} {'Trd':>4} {'PnL':>7} {'MC':>6}")
        for p in patterns[:10]:
            print(f"  {p['pattern']:<14} {p['direction']:<6} {p['tp']:>4.1f} {p['sl']:>4.1f} "
                  f"{p['rr']:>5.2f} {p['wr']:>5.1f} {p['edge']:>5.1f} {p['trades']:>4} "
                  f"{p['pnl']:>+7.1f} {p['mc_p']:>6.3f}")

        h1_results[f'rr_{min_rr}'] = {
            'count': len(patterns),
            'n_long': n_long,
            'n_short': n_short,
            'mean_wr': round(np.mean(wrs), 1),
            'mean_edge': round(np.mean(edges), 1),
            'mean_rr': round(np.mean(rrs), 2),
            'portfolio_trades': pstats['trades'] if portfolio_trades else 0,
            'portfolio_pnl': round(pstats['pnl'], 1) if portfolio_trades else 0,
            'portfolio_mdd': round(pstats['mdd'], 1) if portfolio_trades else 0,
            'top_patterns': [
                {k: v for k, v in p.items() if k not in ('pnls', 'trade_list')}
                for p in patterns[:20]
            ],
        }

    # WF for best R:R level
    best_level = max(h1_results.items(), key=lambda x: x[1].get('portfolio_pnl', 0))
    print_section(f"H1: Walk-Forward Validation (best level: {best_level[0]})")

    min_rr_best = float(best_level[0].split('_')[1])

    # Custom WF: 3-fold expanding window
    n_folds = 3
    seg = n_bars // (n_folds + 1)
    wf_results = []

    for fold in range(n_folds):
        is_end = (fold + 1) * seg
        oos_start = is_end
        oos_end = min((fold + 2) * seg, n_bars)

        # IS scan
        is_patterns = rr_constrained_scan(
            signal_index, opens, highs, lows, n_bars,
            atr_ratio, min_rr_best, bar_start, is_end
        )

        # OOS test
        oos_trades = []
        for p in is_patterns:
            sigs = [s for s in signal_index[p['pattern']] if oos_start <= s < oos_end]
            if sigs:
                trades = bt_signals_atr(
                    sigs, p['direction'], p['tp'], p['sl'],
                    opens, highs, lows, n_bars,
                    atr_ratio, 0.6, 1.7
                )
                oos_trades.extend(trades)

        # No-overlap portfolio
        oos_trades.sort(key=lambda t: t[0])
        portfolio = []
        last_exit = -1
        for t in oos_trades:
            if t[0] > last_exit:
                portfolio.append(t)
                last_exit = t[1]

        oos_stats = calc_stats(portfolio) if portfolio else {'pnl': 0, 'trades': 0, 'wr': 0}
        wf_results.append({
            'fold': fold + 1,
            'is_patterns': len(is_patterns),
            'oos_trades': oos_stats['trades'],
            'oos_wr': round(oos_stats['wr'], 1),
            'oos_pnl': round(oos_stats['pnl'], 1),
        })
        print(f"  Fold {fold+1}: IS {len(is_patterns)} pat | "
              f"OOS {oos_stats['trades']} trades, WR {oos_stats['wr']:.1f}%, "
              f"PnL {oos_stats['pnl']:+.1f}%")

    positive_folds = sum(1 for f in wf_results if f['oos_pnl'] > 0)
    total_oos_pnl = sum(f['oos_pnl'] for f in wf_results)
    print(f"\n  WF Result: {positive_folds}/{n_folds} PASS | "
          f"Total OOS PnL: {total_oos_pnl:+.1f}%")
    h1_results['wf'] = {
        'folds': wf_results,
        'pass': positive_folds == n_folds,
        'total_oos_pnl': round(total_oos_pnl, 1),
    }

    return h1_results


# =========================================================================
# H2: POSITION COUNT REDUCTION
# =========================================================================

def portfolio_hedge_sim(all_trades, n_max, dir_cap, timeout_bars=MAX_BARS):
    """
    Simulate hedge portfolio with N max positions and direction cap.

    all_trades: list of dict with keys:
        entry_bar, exit_bar, pnl_pct, direction, pattern
    Returns: dict with pnl, mdd, trades, wr, trade_list
    """
    all_trades = sorted(all_trades, key=lambda t: t['entry_bar'])

    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    open_positions = []  # list of {entry_bar, exit_bar, pnl_pct, direction}
    taken_trades = []

    for trade in all_trades:
        eb = trade['entry_bar']

        # Close expired positions
        open_positions = [p for p in open_positions if p['exit_bar'] > eb]

        # Check capacity
        if len(open_positions) >= n_max:
            continue

        # Check direction cap
        dir_count = sum(1 for p in open_positions if p['direction'] == trade['direction'])
        if dir_count >= dir_cap:
            continue

        # Open position with 1/N sizing
        size_pct = 1.0 / n_max
        portfolio_pnl = trade['pnl_pct'] * size_pct
        equity *= (1 + portfolio_pnl / 100)
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

        open_positions.append(trade)
        taken_trades.append({**trade, 'portfolio_pnl': portfolio_pnl})

    total_pnl = equity - 100.0
    wins = sum(1 for t in taken_trades if t['pnl_pct'] >= 0)
    wr = wins / len(taken_trades) * 100 if taken_trades else 0

    return {
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'trades': len(taken_trades),
        'wr': round(wr, 1),
        'wins': wins,
        'pnl_per_mdd': round(total_pnl / max_dd, 2) if max_dd > 0 else 0,
    }


def run_h2(df, signal_index, opens, highs, lows, n_bars, atr_ratio,
           bar_start, bar_end):
    """H2: Position Count Reduction Study."""
    print_header("H2: POSITION COUNT REDUCTION")

    # Load current 130 patterns
    with open(DYNAMIC_PATTERNS_FILE) as f:
        dyn = json.load(f)

    patterns_tpsl = dyn.get('patterns_tpsl', {})
    patterns_dict = dyn.get('patterns', {})
    long_patterns = patterns_dict.get('long', [])
    short_patterns = patterns_dict.get('short', [])

    # Generate all trades
    all_trades = []
    for pat_name in long_patterns:
        # Try direction-suffixed key first, then plain key
        tp_sl = patterns_tpsl.get(f"{pat_name}_LONG") or patterns_tpsl.get(pat_name)
        if not tp_sl:
            continue
        tp, sl = tp_sl
        sigs = [s for s in signal_index.get(pat_name, []) if bar_start <= s < bar_end]
        if not sigs:
            continue
        trades = bt_signals_atr(sigs, "LONG", tp, sl, opens, highs, lows, n_bars,
                                atr_ratio, 0.6, 1.7)
        for t in trades:
            all_trades.append({
                'entry_bar': t[0], 'exit_bar': t[1], 'pnl_pct': t[2],
                'direction': 'LONG', 'pattern': pat_name,
            })

    for pat_name in short_patterns:
        tp_sl = patterns_tpsl.get(f"{pat_name}_SHORT") or patterns_tpsl.get(pat_name)
        if not tp_sl:
            continue
        tp, sl = tp_sl
        sigs = [s for s in signal_index.get(pat_name, []) if bar_start <= s < bar_end]
        if not sigs:
            continue
        trades = bt_signals_atr(sigs, "SHORT", tp, sl, opens, highs, lows, n_bars,
                                atr_ratio, 0.6, 1.7)
        for t in trades:
            all_trades.append({
                'entry_bar': t[0], 'exit_bar': t[1], 'pnl_pct': t[2],
                'direction': 'SHORT', 'pattern': pat_name,
            })

    print(f"  Total individual trades generated: {len(all_trades)}")
    print(f"  Long: {sum(1 for t in all_trades if t['direction'] == 'LONG')}")
    print(f"  Short: {sum(1 for t in all_trades if t['direction'] == 'SHORT')}")

    # Run scenarios
    print_section("H2: N × Cap Scenario Matrix")
    print(f"  {'N':>3} {'Cap':>4} {'Trades':>7} {'WR':>6} {'PnL':>8} {'MDD':>6} {'PnL/MDD':>8}")

    h2_results = {}
    for n_max in N_VALUES:
        for cap in CAP_VALUES:
            if cap > n_max:
                continue
            result = portfolio_hedge_sim(all_trades, n_max, cap)
            key = f"N{n_max}_C{cap}"
            h2_results[key] = result
            print(f"  {n_max:>3} {cap:>4} {result['trades']:>7} {result['wr']:>5.1f}% "
                  f"{result['pnl']:>+8.1f} {result['mdd']:>5.1f}% {result['pnl_per_mdd']:>8.2f}")

    # Best by PnL/MDD
    best = max(h2_results.items(), key=lambda x: x[1]['pnl_per_mdd'])
    print(f"\n  Best PnL/MDD: {best[0]} = {best[1]['pnl_per_mdd']:.2f}x "
          f"(PnL {best[1]['pnl']:+.1f}%, MDD {best[1]['mdd']:.1f}%)")

    # Compare correlated loss impact
    print_section("H2: Correlated Loss Analysis")
    if not all_trades:
        print("  No trades generated — skipping correlated loss analysis")
    else:
        for key in ['N9_C7', 'N4_C2', 'N3_C2', 'N2_C1']:
            if key not in h2_results:
                continue
            r = h2_results[key]
            n_val = int(key.split('_')[0][1:])
            cap_val = int(key.split('_')[1][1:])
            losses_per_trade = [t['pnl_pct'] / n_val
                               for t in all_trades if t['pnl_pct'] < 0]
            if not losses_per_trade:
                print(f"  {key}: No losses")
                continue
            worst_single = abs(min(losses_per_trade))
            max_concurrent_loss = cap_val * worst_single / 100
            print(f"  {key}: Max concurrent same-dir loss (worst case) = "
                  f"{cap_val} x {worst_single:.1f}% / {n_val} = "
                  f"{max_concurrent_loss*100:.1f}%")

    return h2_results


# =========================================================================
# H3: LIVE DATA REGIME ANALYSIS
# =========================================================================

def run_h3():
    """H3: Live Trade Analysis — What predicts win/loss?"""
    print_header("H3: LIVE DATA REGIME ANALYSIS")

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    trades = metrics.get('trade_history', [])
    pattern_trades = [t for t in trades if t['pattern'] != 'N/A']

    if not pattern_trades:
        print("  No pattern trades in history.")
        return {}

    # Compute derived features for each trade
    for t in pattern_trades:
        entry = t['entry_price']
        tp = t['tp_price']
        sl = t['sl_price']
        direction = t['direction']

        if direction == 'LONG':
            t['tp_dist_pct'] = (tp - entry) / entry * 100
            t['sl_dist_pct'] = (entry - sl) / entry * 100
        else:
            t['tp_dist_pct'] = (entry - tp) / entry * 100
            t['sl_dist_pct'] = (sl - entry) / entry * 100

        t['rr'] = t['tp_dist_pct'] / t['sl_dist_pct'] if t['sl_dist_pct'] > 0 else 0
        t['is_win'] = t['pnl_slot'] >= 0

    wins = [t for t in pattern_trades if t['is_win']]
    losses = [t for t in pattern_trades if not t['is_win']]

    print(f"  Pattern trades: {len(pattern_trades)} ({len(wins)}W / {len(losses)}L)")

    # 1. TP/SL distance comparison
    print_section("H3-1: TP/SL Distance — Wins vs Losses")
    for label, group in [("Wins", wins), ("Losses", losses)]:
        if not group:
            continue
        tp_dists = [t['tp_dist_pct'] for t in group]
        sl_dists = [t['sl_dist_pct'] for t in group]
        rrs = [t['rr'] for t in group]
        print(f"  {label} (n={len(group)}):")
        print(f"    TP dist: {np.mean(tp_dists):.2f}% (med {np.median(tp_dists):.2f}%)")
        print(f"    SL dist: {np.mean(sl_dists):.2f}% (med {np.median(sl_dists):.2f}%)")
        print(f"    R:R:     {np.mean(rrs):.3f} (med {np.median(rrs):.3f})")

    # 2. Holding time analysis
    print_section("H3-2: Holding Time — Wins vs Losses")
    for label, group in [("Wins", wins), ("Losses", losses)]:
        if not group:
            continue
        holds = [t['hold_minutes'] for t in group]
        print(f"  {label}: mean {np.mean(holds):.0f}min | "
              f"median {np.median(holds):.0f}min | "
              f"max {max(holds)}min")

    # 3. Direction analysis
    print_section("H3-3: Direction Performance")
    for direction in ['LONG', 'SHORT']:
        dir_trades = [t for t in pattern_trades if t['direction'] == direction]
        dir_wins = [t for t in dir_trades if t['is_win']]
        if dir_trades:
            wr = len(dir_wins) / len(dir_trades) * 100
            avg_pnl = np.mean([t['pnl_slot'] for t in dir_trades])
            print(f"  {direction}: {len(dir_trades)} trades, WR {wr:.1f}%, "
                  f"Avg slot PnL {avg_pnl:+.2f}%")

    # 4. BTC price move during trade
    print_section("H3-4: Price Move During Trade (proxy for regime)")
    for t in pattern_trades:
        # Price move = (exit - entry) / entry
        t['price_move_pct'] = (t['exit_price'] - t['entry_price']) / t['entry_price'] * 100

    for label, group in [("Wins", wins), ("Losses", losses)]:
        if not group:
            continue
        moves = [t['price_move_pct'] for t in group]
        print(f"  {label}: mean BTC move {np.mean(moves):+.2f}% | "
              f"range [{min(moves):+.2f}%, {max(moves):+.2f}%]")

    # 5. Temporal patterns
    print_section("H3-5: Time-of-Day Analysis")
    by_hour = defaultdict(lambda: {'wins': 0, 'losses': 0})
    for t in pattern_trades:
        hour = datetime.fromisoformat(t['timestamp']).hour
        if t['is_win']:
            by_hour[hour]['wins'] += 1
        else:
            by_hour[hour]['losses'] += 1

    print(f"  {'Hour':>4} {'W':>3} {'L':>3} {'WR':>6}")
    for hour in sorted(by_hour.keys()):
        h = by_hour[hour]
        total = h['wins'] + h['losses']
        wr = h['wins'] / total * 100 if total > 0 else 0
        print(f"  {hour:>4} {h['wins']:>3} {h['losses']:>3} {wr:>5.0f}%")

    # 6. Pattern frequency in wins vs losses
    print_section("H3-6: Most Profitable/Unprofitable Patterns")

    by_pattern = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
    for t in pattern_trades:
        key = f"{t['pattern']}_{t['direction']}"
        if t['is_win']:
            by_pattern[key]['wins'] += 1
        else:
            by_pattern[key]['losses'] += 1
        by_pattern[key]['pnl'] += t['pnl_portfolio']

    sorted_pats = sorted(by_pattern.items(), key=lambda x: x[1]['pnl'])

    print("  Worst:")
    for pat, info in sorted_pats[:5]:
        total = info['wins'] + info['losses']
        wr = info['wins'] / total * 100
        print(f"    {pat:<20} {total}t ({info['wins']}W/{info['losses']}L) "
              f"WR {wr:.0f}% PnL {info['pnl']:+.2f}%")

    print("  Best:")
    for pat, info in sorted_pats[-5:]:
        total = info['wins'] + info['losses']
        wr = info['wins'] / total * 100
        print(f"    {pat:<20} {total}t ({info['wins']}W/{info['losses']}L) "
              f"WR {wr:.0f}% PnL {info['pnl']:+.2f}%")

    # 7. Key insight: direction accuracy
    print_section("H3-7: Direction Prediction Accuracy")
    for t in pattern_trades:
        if t['direction'] == 'LONG':
            t['direction_correct'] = t['exit_price'] > t['entry_price']
        else:
            t['direction_correct'] = t['exit_price'] < t['entry_price']

    correct = sum(1 for t in pattern_trades if t['direction_correct'])
    total = len(pattern_trades)
    print(f"  Direction correct: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  (50% = random, >50% = some directional edge)")

    for direction in ['LONG', 'SHORT']:
        dir_trades = [t for t in pattern_trades if t['direction'] == direction]
        dir_correct = sum(1 for t in dir_trades if t['direction_correct'])
        if dir_trades:
            print(f"  {direction}: {dir_correct}/{len(dir_trades)} = "
                  f"{dir_correct/len(dir_trades)*100:.1f}% direction accuracy")

    return {'n_trades': len(pattern_trades)}


# =========================================================================
# MAIN
# =========================================================================

def main():
    t_start = time.time()

    print_header("STRATEGY REFORM STUDY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Data: {DATA_FILE}")
    print(f"Protocol: Compound, Lev={LEVERAGE}, Fee={FEE_PCT*2:.2f}%, "
          f"ATR(14/576, [0.6,1.7]), MC<{MC_THRESHOLD}, MinTr>={MIN_TRADES}")

    # Load data
    print("\nLoading data...")
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    print(f"  Bars: {n_bars}")

    # Find neutral window
    closes = df['close'].values
    nw = find_neutral_window(closes, tol_pct=1.0)
    if nw:
        bar_start, bar_end = nw
        start_price = closes[bar_start]
        end_price = closes[bar_end - 1]
        days = (bar_end - bar_start) / 288
        print(f"  Neutral window: bars [{bar_start}, {bar_end}) = {days:.0f} days")
        print(f"  Price: ${start_price:.0f} -> ${end_price:.0f} "
              f"({(end_price/start_price-1)*100:+.1f}%)")
    else:
        bar_start, bar_end = 0, n_bars
        print("  No neutral window found, using full data")

    # Prepare arrays
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    # Build signal index
    types = df['candle_type'].values  # short codes (BD, BU, IH, etc.)
    signal_index = build_signal_index(types, n_bars)
    print(f"  Unique patterns: {len(signal_index)}")

    # ATR ratio
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    print(f"  ATR ratio: mean {np.nanmean(atr_ratio):.3f}, "
          f"std {np.nanstd(atr_ratio):.3f}")

    # Run H1
    h1_results = run_h1(df, signal_index, opens, highs, lows, closes, n_bars,
                        atr_ratio, bar_start, bar_end)

    # Run H2
    h2_results = run_h2(df, signal_index, opens, highs, lows, n_bars,
                        atr_ratio, bar_start, bar_end)

    # Run H3
    h3_results = run_h3()

    # Summary
    print_header("SUMMARY & RECOMMENDATIONS")

    print("\nH1 (R:R Reversal):")
    for level in MIN_RR_LEVELS:
        key = f'rr_{level}'
        r = h1_results.get(key, {})
        count = r.get('count', 0)
        pnl = r.get('portfolio_pnl', 0)
        mdd = r.get('portfolio_mdd', 0)
        print(f"  R:R >= {level}: {count} patterns | "
              f"PnL {pnl:+.1f}% | MDD {mdd:.1f}%")

    if 'wf' in h1_results:
        wf = h1_results['wf']
        status = "PASS" if wf['pass'] else "FAIL"
        print(f"  WF: {status} | OOS PnL {wf['total_oos_pnl']:+.1f}%")

    print("\nH2 (Position Reduction):")
    current = h2_results.get('N9_C7', {})
    reduced = h2_results.get('N3_C2', {})
    if current and reduced:
        print(f"  Current (N9/C7): PnL {current['pnl']:+.1f}%, "
              f"MDD {current['mdd']:.1f}%, PnL/MDD {current['pnl_per_mdd']:.2f}x")
        print(f"  Reduced (N3/C2): PnL {reduced['pnl']:+.1f}%, "
              f"MDD {reduced['mdd']:.1f}%, PnL/MDD {reduced['pnl_per_mdd']:.2f}x")

    print("\nH3 (Live Analysis):")
    print("  See detailed analysis above")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'h1': {k: v for k, v in h1_results.items()
               if k != 'wf' or not isinstance(v, dict)},
        'h2': h2_results,
        'h3': h3_results,
    }
    # Serialize numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
