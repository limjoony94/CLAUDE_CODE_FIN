#!/usr/bin/env python3
"""
Timeout Forced-Close WF Research
Scanner-Production Gap 해소 — MAX_BARS 강제청산 도입 시 WF OOS 성과 검증

핵심 질문:
  현재 Scanner(DROP)와 프로덕션(HOLD) 간 괴리를 강제청산으로 해소할 수 있는가?
  최적 MAX_BARS는 무엇인가?

Phase 1: DROP vs FC vs HOLD 비교 (현재 288바, 112패턴)
Phase 2: MAX_BARS 최적화 (144/216/288/432바, PP scanner → DROP/FC)
Phase 3: WF 검증 (IS=DROP, OOS=DROP+FC, 상위 3개 MAX_BARS)
Phase 4: 고 타임아웃 패턴 필터링 + WF (최적 MAX_BARS)
Phase 5: FC-aware 패턴 발굴 + WF (핵심 — DROP 발굴 vs FC 발굴 비교)

Protocol:
  - Production classify_candle() via scanner (indicators.py)
  - LEVERAGE = 3, FEE = 0.10%
  - Expanding Window WF (3-fold)
  - Multi-seed MC test
"""

import os
import sys
import json
import time
import numpy as np
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'timeout_forced_close_study.json')
LEVERAGE = scanner.LEVERAGE      # 3
FEE_PCT = scanner.FEE_PCT        # 0.10
EDGE_THRESHOLD = 21.8            # Current post-filter


# ============================================================
# Forced-Close Backtest Engine
# ============================================================

def bt_signals_fc(signal_bars, direction, tp_pct, sl_pct,
                  opens, highs, lows, closes, n_bars, max_bars):
    """
    bt_signals variant: timeout trades get forced-close PnL (market close at MAX_BARS)
    instead of being dropped.

    Returns list of (entry_bar, exit_bar, pnl, is_timeout).
    Matches scanner.bt_signals logic exactly for resolved trades.
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

        resolved = False
        end = min(idx + 2 + max_bars, n_bars)

        for j in range(idx + 2, end):
            if direction == 'LONG':
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append((eb, j, round(pnl, 4), False))
                resolved = True
                break
            elif ht:
                trades.append((eb, j, round(tp_pct * LEVERAGE - FEE_PCT, 4), False))
                resolved = True
                break
            elif hs:
                trades.append((eb, j, round(-sl_pct * LEVERAGE - FEE_PCT, 4), False))
                resolved = True
                break

        if not resolved and end > idx + 2:
            # Forced close at last bar's close price
            xb = end - 1
            xp = closes[xb]
            if direction == 'LONG':
                raw = (xp - entry) / entry * 100
            else:
                raw = (entry - xp) / entry * 100
            pnl = raw * LEVERAGE - FEE_PCT
            trades.append((eb, xb, round(pnl, 4), True))

    return trades


def portfolio_1pos_4t(all_trades):
    """1-pos-at-a-time for 4-tuple trades."""
    if not all_trades:
        return []
    all_trades = sorted(all_trades, key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for t in all_trades:
        if t[0] > last_exit:
            filtered.append(t)
            last_exit = t[1]
    return filtered


def calc_stats_fc(trades):
    """calc_stats for 4-tuple trades. Adds timeout breakdown."""
    empty = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
             'timeout_count': 0, 'timeout_pct': 0, 'timeout_avg_pnl': 0, 'timeout_wr': 0}
    if not trades:
        return empty

    pnls = [t[2] for t in trades]
    timeouts = [t for t in trades if t[3]]

    cum = 0.0
    peak = 0.0
    mdd = 0.0
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
    tp_pnls = [t[2] for t in timeouts]

    return {
        'pnl': round(cum, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1),
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'timeout_count': len(timeouts),
        'timeout_pct': round(len(timeouts) / len(trades) * 100, 1),
        'timeout_avg_pnl': round(float(np.mean(tp_pnls)), 2) if tp_pnls else 0,
        'timeout_wr': round(sum(1 for p in tp_pnls if p > 0) / len(tp_pnls) * 100, 1) if tp_pnls else 0,
    }


# ============================================================
# Portfolio Backtest Helpers (pattern list format)
# ============================================================

def _backtest_drop(pat_list, signal_index, opens, highs, lows, n_bars, max_bars,
                   bar_start=0, bar_end=None):
    """DROP backtest from a pattern list (scan_universe_range format)."""
    if bar_end is None:
        bar_end = n_bars
    original = scanner.MAX_BARS
    scanner.MAX_BARS = max_bars
    try:
        all_trades = []
        for r in pat_list:
            sigs = [s for s in signal_index[r['pattern']] if bar_start <= s < bar_end]
            trades = scanner.bt_signals(sigs, r['direction'], r['tp'], r['sl'],
                                        opens, highs, lows, bar_end)
            all_trades.extend(trades)
        port = scanner.portfolio_1pos(all_trades)
        stats = scanner.calc_stats(port)
        pnls = [t[2] for t in port]
        mc_p = scanner.mc_test(pnls) if len(pnls) >= 5 else 1.0
        return {'stats': stats, 'mc_p': round(mc_p, 4)}
    finally:
        scanner.MAX_BARS = original


def _backtest_fc(pat_list, signal_index, opens, highs, lows, closes, n_bars, max_bars,
                 bar_start=0, bar_end=None):
    """Forced-Close backtest from a pattern list."""
    if bar_end is None:
        bar_end = n_bars
    all_trades = []
    for r in pat_list:
        sigs = [s for s in signal_index[r['pattern']] if bar_start <= s < bar_end]
        trades = bt_signals_fc(sigs, r['direction'], r['tp'], r['sl'],
                               opens, highs, lows, closes, bar_end, max_bars)
        all_trades.extend(trades)
    port = portfolio_1pos_4t(all_trades)
    stats = calc_stats_fc(port)
    pnls = [t[2] for t in port]
    mc_p = scanner.mc_test(pnls) if len(pnls) >= 5 else 1.0
    return {'stats': stats, 'mc_p': round(mc_p, 4)}


def _per_pattern_timeout_rate(pat_list, signal_index, opens, highs, lows, closes,
                              n_bars, max_bars, bar_start=0, bar_end=None):
    """Compute timeout rate per pattern (non-portfolio, individual)."""
    if bar_end is None:
        bar_end = n_bars
    rates = {}
    for r in pat_list:
        key = f"{r['pattern']}_{r['direction']}"
        sigs = [s for s in signal_index[r['pattern']] if bar_start <= s < bar_end]
        fc_trades = bt_signals_fc(sigs, r['direction'], r['tp'], r['sl'],
                                  opens, highs, lows, closes, bar_end, max_bars)
        total = len(fc_trades)
        timeouts = sum(1 for t in fc_trades if t[3])
        tp_pnls = [t[2] for t in fc_trades if t[3]]
        rates[key] = {
            'total': total,
            'timeouts': timeouts,
            'timeout_pct': round(timeouts / total * 100, 1) if total > 0 else 0,
            'timeout_avg_pnl': round(float(np.mean(tp_pnls)), 2) if tp_pnls else 0,
        }
    return rates


# ============================================================
# FC-aware Discovery Engine (Phase 5)
# ============================================================

def grid_search_best_fc(signal_bars, direction, opens, highs, lows, closes,
                        n_bars, max_bars, min_tr=20,
                        max_baseline_wr=scanner.MAX_BASELINE_WR):
    """Grid search optimized by FC PnL/MDD (includes timeout PnL).
    Key difference from scanner.grid_search_best: uses bt_signals_fc
    so timeout trades affect the scoring, not just resolved ones.
    """
    best = None
    best_score = -9999
    for tp in scanner.TP_GRID:
        for sl in scanner.SL_GRID:
            if sl < 0.5:
                continue
            bwr = sl / (tp + sl) * 100
            if bwr > max_baseline_wr:
                continue
            trades = bt_signals_fc(signal_bars, direction, tp, sl,
                                   opens, highs, lows, closes, n_bars, max_bars)
            if len(trades) < min_tr:
                continue
            pnls = [t[2] for t in trades]
            cum = 0.0; peak = 0.0; mdd_val = 0.0; w = 0
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


def scan_universe_range_fc(signal_index, opens, highs, lows, closes, n_bars,
                           bar_start, bar_end, max_bars,
                           min_trades=scanner.DEFAULT_MIN_TRADES,
                           edge_threshold=EDGE_THRESHOLD,
                           mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
                           max_baseline_wr=scanner.MAX_BASELINE_WR):
    """FC-aware pattern scan: grid_search_best_fc + FC backtest for scoring.
    Patterns discovered here are inherently optimized for FC performance.
    """
    selected = []
    trade_boundary = bar_end if bar_end is not None else n_bars

    for pat_name, all_sigs in signal_index.items():
        sigs = [s for s in all_sigs if bar_start <= s < bar_end]

        for direction in ['LONG', 'SHORT']:
            if len(sigs) < min_trades:
                continue

            opt = grid_search_best_fc(sigs, direction, opens, highs, lows, closes,
                                      trade_boundary, max_bars,
                                      min_tr=min_trades,
                                      max_baseline_wr=max_baseline_wr)
            if opt is None:
                continue
            tp, sl = opt['tp'], opt['sl']

            # Re-run FC backtest to get full trades for edge/MC check
            trades = bt_signals_fc(sigs, direction, tp, sl,
                                   opens, highs, lows, closes,
                                   trade_boundary, max_bars)
            if len(trades) < min_trades:
                continue

            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            baseline_wr = sl / (tp + sl) * 100
            edge = wr - baseline_wr

            if edge < edge_threshold:
                continue

            p = scanner.mc_test(pnls)
            if p >= mc_threshold:
                continue

            selected.append({
                'pattern': pat_name, 'direction': direction,
                'tp': tp, 'sl': sl,
                'trades': len(trades), 'wr': round(wr, 1),
                'edge': round(edge, 1), 'mc_p': round(p, 4),
            })

    return selected


def wf_fc_aware(signal_index, opens, highs, lows, closes, n_bars,
                max_bars, edge_threshold=EDGE_THRESHOLD, n_folds=3):
    """
    FC-aware Expanding Window WF.
    IS: scan_universe_range_fc (FC-aware grid search).
    OOS: FC backtest.
    Key difference from wf_drop_vs_fc: discovery itself is FC-aware.
    """
    seg_size = n_bars // (n_folds + 1)
    folds = []
    pattern_appearances = Counter()

    for fold_idx in range(n_folds):
        is_end = (fold_idx + 1) * seg_size
        oos_start = is_end
        oos_end = (fold_idx + 2) * seg_size if fold_idx < n_folds - 1 else n_bars

        # IS: FC-aware grid search
        is_patterns = scan_universe_range_fc(
            signal_index, opens, highs, lows, closes, n_bars,
            bar_start=0, bar_end=is_end, max_bars=max_bars,
            min_trades=scanner.DEFAULT_MIN_TRADES,
            edge_threshold=edge_threshold,
            mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
            max_baseline_wr=scanner.MAX_BASELINE_WR,
        )

        # Dedup: keep best edge per pattern
        best_by_pat = {}
        for r in is_patterns:
            key = r['pattern']
            if key not in best_by_pat or r['edge'] > best_by_pat[key]['edge']:
                best_by_pat[key] = r
        is_patterns = list(best_by_pat.values())

        if not is_patterns:
            folds.append({
                'fold': fold_idx + 1, 'is_bars': is_end, 'oos_bars': oos_end - oos_start,
                'is_patterns': 0,
                'oos_fc': {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                           'timeout_count': 0, 'timeout_pct': 0, 'timeout_avg_pnl': 0, 'timeout_wr': 0},
                'oos_drop': {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0},
            })
            continue

        for r in is_patterns:
            pattern_appearances[(r['pattern'], r['direction'])] += 1

        n_long = sum(1 for r in is_patterns if r['direction'] == 'LONG')
        n_short = sum(1 for r in is_patterns if r['direction'] == 'SHORT')

        # OOS FC
        fc_r = _backtest_fc(is_patterns, signal_index, opens, highs, lows, closes,
                            n_bars, max_bars, bar_start=oos_start, bar_end=oos_end)
        # OOS DROP (for comparison)
        drop_r = _backtest_drop(is_patterns, signal_index, opens, highs, lows,
                                n_bars, max_bars, bar_start=oos_start, bar_end=oos_end)

        folds.append({
            'fold': fold_idx + 1,
            'is_bars': is_end, 'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_patterns), 'is_long': n_long, 'is_short': n_short,
            'oos_fc': fc_r['stats'],
            'oos_drop': drop_r['stats'],
        })

    fc_pnls = [f['oos_fc']['pnl'] for f in folds]
    drop_pnls = [f['oos_drop']['pnl'] for f in folds]

    stable = sorted(
        f"{pat}_{d}" for (pat, d), cnt in pattern_appearances.items()
        if cnt == n_folds
    )

    return {
        'folds': folds,
        'fc_total_oos': round(sum(fc_pnls), 1),
        'drop_total_oos': round(sum(drop_pnls), 1),
        'fc_positive': sum(1 for p in fc_pnls if p > 0),
        'drop_positive': sum(1 for p in drop_pnls if p > 0),
        'n_folds': len(folds),
        'stable_patterns': stable,
        'fc_avg_timeout_pct': round(float(np.mean(
            [f['oos_fc'].get('timeout_pct', 0) for f in folds if f['oos_fc'].get('trades', 0) > 0]
        )), 1) if folds else 0,
    }


# ============================================================
# WF Engine (IS: DROP discovery, OOS: DROP + FC)
# ============================================================

def wf_drop_vs_fc(signal_index, opens, highs, lows, closes, n_bars,
                  max_bars, edge_threshold=EDGE_THRESHOLD, n_folds=3,
                  timeout_filter_pct=None):
    """
    Expanding Window WF.
    IS: scan_universe_range (DROP mode, per_pattern grid search).
    OOS: both DROP and FC backtests.
    Optional: filter IS patterns by timeout rate on IS data.
    """
    seg_size = n_bars // (n_folds + 1)
    folds = []
    pattern_appearances = Counter()

    for fold_idx in range(n_folds):
        is_end = (fold_idx + 1) * seg_size
        oos_start = is_end
        oos_end = (fold_idx + 2) * seg_size if fold_idx < n_folds - 1 else n_bars

        # IS: scanner PP grid search (DROP mode)
        original = scanner.MAX_BARS
        scanner.MAX_BARS = max_bars
        is_patterns = scanner.scan_universe_range(
            signal_index, opens, highs, lows, n_bars,
            bar_start=0, bar_end=is_end, mode='per_pattern',
            min_trades=scanner.DEFAULT_MIN_TRADES,
            edge_threshold=edge_threshold,
            mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
            max_baseline_wr=scanner.MAX_BASELINE_WR,
        )
        scanner.MAX_BARS = original

        # Dedup: keep best edge per pattern
        best_by_pat = {}
        for r in is_patterns:
            key = r['pattern']
            if key not in best_by_pat or r['edge'] > best_by_pat[key]['edge']:
                best_by_pat[key] = r
        is_patterns = list(best_by_pat.values())

        # Optional: timeout filter on IS data
        if timeout_filter_pct is not None and is_patterns:
            t_rates = _per_pattern_timeout_rate(
                is_patterns, signal_index, opens, highs, lows, closes,
                n_bars, max_bars, bar_start=0, bar_end=is_end
            )
            is_patterns = [
                r for r in is_patterns
                if f"{r['pattern']}_{r['direction']}" in t_rates
                and t_rates[f"{r['pattern']}_{r['direction']}"]['timeout_pct'] <= timeout_filter_pct
            ]

        if not is_patterns:
            folds.append({
                'fold': fold_idx + 1, 'is_bars': is_end, 'oos_bars': oos_end - oos_start,
                'is_patterns': 0, 'oos_drop': {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0},
                'oos_fc': {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0},
            })
            continue

        for r in is_patterns:
            pattern_appearances[(r['pattern'], r['direction'])] += 1

        # OOS DROP
        drop_r = _backtest_drop(is_patterns, signal_index, opens, highs, lows,
                                n_bars, max_bars, bar_start=oos_start, bar_end=oos_end)
        # OOS FC
        fc_r = _backtest_fc(is_patterns, signal_index, opens, highs, lows, closes,
                            n_bars, max_bars, bar_start=oos_start, bar_end=oos_end)

        n_long = sum(1 for r in is_patterns if r['direction'] == 'LONG')
        n_short = sum(1 for r in is_patterns if r['direction'] == 'SHORT')

        folds.append({
            'fold': fold_idx + 1,
            'is_bars': is_end, 'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_patterns), 'is_long': n_long, 'is_short': n_short,
            'oos_drop': drop_r['stats'],
            'oos_fc': fc_r['stats'],
        })

    # Summary
    drop_pnls = [f['oos_drop']['pnl'] for f in folds]
    fc_pnls = [f['oos_fc']['pnl'] for f in folds]
    fc_stats_list = [f['oos_fc'] for f in folds]

    stable = sorted(
        f"{pat}_{d}" for (pat, d), cnt in pattern_appearances.items()
        if cnt == n_folds
    )

    return {
        'folds': folds,
        'drop_total_oos': round(sum(drop_pnls), 1),
        'fc_total_oos': round(sum(fc_pnls), 1),
        'drop_positive': sum(1 for p in drop_pnls if p > 0),
        'fc_positive': sum(1 for p in fc_pnls if p > 0),
        'n_folds': len(folds),
        'stable_patterns': stable,
        'gap_pnl': round(sum(drop_pnls) - sum(fc_pnls), 1),
        'fc_avg_timeout_pct': round(float(np.mean(
            [f['oos_fc'].get('timeout_pct', 0) for f in folds if f['oos_fc'].get('trades', 0) > 0]
        )), 1) if folds else 0,
    }


# ============================================================
# Main Study
# ============================================================

def main():
    t0 = time.time()
    results = {}

    # Load data
    print("Loading data...")
    df = scanner.load_and_classify(scanner.DEFAULT_DATA_FILE)
    types = df['candle_type'].tolist()
    signal_index = scanner.build_signal_index(types, len(df))
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    print(f"  {n_bars} bars, {n_bars * 5 / 60 / 24:.0f} days")

    # ========================================
    # Phase 1: DROP vs FC vs HOLD (288 bars)
    # ========================================
    print("\n" + "=" * 60)
    print("Phase 1: DROP vs FORCED_CLOSE vs HOLD (288 bars, edge>=21.8)")
    print("=" * 60)

    # Get current patterns via PP scanner
    original_max = scanner.MAX_BARS
    scanner.MAX_BARS = 288
    pp = scanner.scan_patterns_pp(df, edge_threshold=EDGE_THRESHOLD, signal_index=signal_index)
    scanner.MAX_BARS = original_max

    # Convert pattern_details dict to list format for consistency
    pat_list = []
    for key, v in pp['pattern_details'].items():
        pat_list.append({
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': v['tp'], 'sl': v['sl'],
            'trades': v['trades'], 'wr': v['wr'], 'edge': v['edge'],
        })

    n_long = len(pp['long_patterns'])
    n_short = len(pp['short_patterns'])
    print(f"  Patterns: {len(pat_list)} ({n_long}L+{n_short}S)")

    # 1a. DROP (scanner default)
    drop_r = _backtest_drop(pat_list, signal_index, opens, highs, lows, n_bars, 288)
    print(f"  DROP:  PnL {drop_r['stats']['pnl']:+.1f}%, WR {drop_r['stats']['wr']}%, "
          f"MDD {drop_r['stats']['mdd']}%, trades {drop_r['stats']['trades']}")

    # 1b. FORCED_CLOSE at 288 bars
    fc_r = _backtest_fc(pat_list, signal_index, opens, highs, lows, closes, n_bars, 288)
    print(f"  FC:    PnL {fc_r['stats']['pnl']:+.1f}%, WR {fc_r['stats']['wr']}%, "
          f"MDD {fc_r['stats']['mdd']}%, trades {fc_r['stats']['trades']}, "
          f"timeout {fc_r['stats']['timeout_pct']}% (avg PnL {fc_r['stats']['timeout_avg_pnl']:+.2f}%)")

    # 1c. HOLD (effectively no timeout, 50000 bars ~173 days)
    hold_r = _backtest_fc(pat_list, signal_index, opens, highs, lows, closes, n_bars, 50000)
    print(f"  HOLD:  PnL {hold_r['stats']['pnl']:+.1f}%, WR {hold_r['stats']['wr']}%, "
          f"MDD {hold_r['stats']['mdd']}%, trades {hold_r['stats']['trades']}, "
          f"timeout {hold_r['stats']['timeout_pct']}%")

    # Per-pattern timeout analysis
    t_rates = _per_pattern_timeout_rate(pat_list, signal_index, opens, highs, lows, closes,
                                        n_bars, 288)
    timeout_pcts = [v['timeout_pct'] for v in t_rates.values()]
    top_timeout = sorted(t_rates.items(), key=lambda x: x[1]['timeout_pct'], reverse=True)[:10]

    results['phase1'] = {
        'max_bars': 288, 'patterns': len(pat_list), 'long': n_long, 'short': n_short,
        'drop': {'stats': drop_r['stats'], 'mc_p': drop_r['mc_p']},
        'forced_close': {'stats': fc_r['stats'], 'mc_p': fc_r['mc_p']},
        'hold': {'stats': hold_r['stats'], 'mc_p': hold_r['mc_p']},
        'gap_drop_fc': {
            'pnl': round(drop_r['stats']['pnl'] - fc_r['stats']['pnl'], 1),
            'wr': round(drop_r['stats']['wr'] - fc_r['stats']['wr'], 1),
            'mdd': round(fc_r['stats']['mdd'] - drop_r['stats']['mdd'], 1),
        },
        'gap_drop_hold': {
            'pnl': round(drop_r['stats']['pnl'] - hold_r['stats']['pnl'], 1),
            'wr': round(drop_r['stats']['wr'] - hold_r['stats']['wr'], 1),
            'mdd': round(hold_r['stats']['mdd'] - drop_r['stats']['mdd'], 1),
        },
        'timeout_distribution': {
            'mean': round(float(np.mean(timeout_pcts)), 1),
            'median': round(float(np.median(timeout_pcts)), 1),
            'max': round(float(np.max(timeout_pcts)), 1),
            'pct_above_50': sum(1 for p in timeout_pcts if p > 50),
            'pct_above_70': sum(1 for p in timeout_pcts if p > 70),
        },
        'top_timeout_patterns': [
            {'pattern': k, 'timeout_pct': v['timeout_pct'], 'total': v['total'],
             'timeout_avg_pnl': v['timeout_avg_pnl']}
            for k, v in top_timeout
        ],
    }

    print(f"\n  Scanner-Production Gap:")
    print(f"    DROP→FC:   PnL {results['phase1']['gap_drop_fc']['pnl']:+.1f}%, "
          f"WR {results['phase1']['gap_drop_fc']['wr']:+.1f}pp, "
          f"MDD +{results['phase1']['gap_drop_fc']['mdd']:.1f}pp")
    print(f"    DROP→HOLD: PnL {results['phase1']['gap_drop_hold']['pnl']:+.1f}%, "
          f"WR {results['phase1']['gap_drop_hold']['wr']:+.1f}pp, "
          f"MDD +{results['phase1']['gap_drop_hold']['mdd']:.1f}pp")
    print(f"\n  Timeout >50%: {results['phase1']['timeout_distribution']['pct_above_50']}/{len(pat_list)} patterns")
    print(f"  Timeout >70%: {results['phase1']['timeout_distribution']['pct_above_70']}/{len(pat_list)} patterns")

    # ========================================
    # Phase 2: MAX_BARS Optimization
    # ========================================
    print("\n" + "=" * 60)
    print("Phase 2: MAX_BARS Optimization (PP scanner → DROP/FC)")
    print("=" * 60)

    mb_candidates = [144, 216, 288, 432]
    phase2 = {}

    for mb in mb_candidates:
        hours = mb * 5 / 60
        print(f"\n  MAX_BARS={mb} ({hours:.0f}h)...")

        scanner.MAX_BARS = mb
        pp_mb = scanner.scan_patterns_pp(df, edge_threshold=EDGE_THRESHOLD,
                                          signal_index=signal_index)
        scanner.MAX_BARS = original_max

        det_mb = pp_mb['pattern_details']
        pl_mb = []
        for key, v in det_mb.items():
            pl_mb.append({
                'pattern': v['pattern'], 'direction': v['direction'],
                'tp': v['tp'], 'sl': v['sl'],
                'trades': v['trades'], 'wr': v['wr'], 'edge': v['edge'],
            })

        n_l = len(pp_mb['long_patterns'])
        n_s = len(pp_mb['short_patterns'])
        print(f"    Patterns: {len(pl_mb)} ({n_l}L+{n_s}S)")

        drop_mb = _backtest_drop(pl_mb, signal_index, opens, highs, lows, n_bars, mb)
        fc_mb = _backtest_fc(pl_mb, signal_index, opens, highs, lows, closes, n_bars, mb)

        gap_pnl = drop_mb['stats']['pnl'] - fc_mb['stats']['pnl']
        gap_wr = drop_mb['stats']['wr'] - fc_mb['stats']['wr']
        fc_pnl_mdd = (fc_mb['stats']['pnl'] / fc_mb['stats']['mdd']
                       if fc_mb['stats']['mdd'] > 0 else 0)

        print(f"    DROP: PnL {drop_mb['stats']['pnl']:+.1f}%, WR {drop_mb['stats']['wr']}%, "
              f"MDD {drop_mb['stats']['mdd']}%")
        print(f"    FC:   PnL {fc_mb['stats']['pnl']:+.1f}%, WR {fc_mb['stats']['wr']}%, "
              f"MDD {fc_mb['stats']['mdd']}%, timeout {fc_mb['stats']['timeout_pct']}%")
        print(f"    Gap:  PnL {gap_pnl:+.1f}%, WR {gap_wr:+.1f}pp")
        print(f"    FC PnL/MDD: {fc_pnl_mdd:.1f}x")

        # Timeout distribution
        t_rates_mb = _per_pattern_timeout_rate(pl_mb, signal_index, opens, highs, lows,
                                               closes, n_bars, mb)
        tp_pcts = [v['timeout_pct'] for v in t_rates_mb.values()]

        phase2[f'MB_{mb}'] = {
            'max_bars': mb, 'hours': round(hours, 1),
            'patterns': len(pl_mb), 'long': n_l, 'short': n_s,
            'drop': {'stats': drop_mb['stats'], 'mc_p': drop_mb['mc_p']},
            'fc': {'stats': fc_mb['stats'], 'mc_p': fc_mb['mc_p']},
            'fc_pnl_mdd': round(fc_pnl_mdd, 1),
            'gap': {'pnl': round(gap_pnl, 1), 'wr': round(gap_wr, 1)},
            'timeout_dist': {
                'mean': round(float(np.mean(tp_pcts)), 1) if tp_pcts else 0,
                'median': round(float(np.median(tp_pcts)), 1) if tp_pcts else 0,
                'above_50': sum(1 for p in tp_pcts if p > 50),
                'above_70': sum(1 for p in tp_pcts if p > 70),
            },
        }

    results['phase2'] = phase2

    # Rank by FC PnL/MDD
    print("\n  === Phase 2 Ranking (by FC PnL/MDD) ===")
    ranked = sorted(phase2.items(), key=lambda x: x[1]['fc_pnl_mdd'], reverse=True)
    for rank, (k, v) in enumerate(ranked, 1):
        print(f"    #{rank} MB={v['max_bars']} ({v['hours']:.0f}h): "
              f"FC PnL/MDD {v['fc_pnl_mdd']:.1f}x, "
              f"FC PnL {v['fc']['stats']['pnl']:+.1f}%, "
              f"Gap PnL {v['gap']['pnl']:+.1f}%, "
              f"timeout {v['fc']['stats']['timeout_pct']}%")

    # ========================================
    # Phase 3: WF Validation (top 3 MAX_BARS)
    # ========================================
    print("\n" + "=" * 60)
    print("Phase 3: WF Validation (IS=DROP discovery, OOS=DROP+FC)")
    print("=" * 60)

    # Top 3 by FC PnL/MDD
    top_mbs = [v['max_bars'] for _, v in ranked[:3]]
    print(f"  Testing MAX_BARS: {top_mbs}")

    phase3 = {}
    for mb in top_mbs:
        hours = mb * 5 / 60
        print(f"\n  WF for MAX_BARS={mb} ({hours:.0f}h)...")
        wf_r = wf_drop_vs_fc(signal_index, opens, highs, lows, closes, n_bars,
                              max_bars=mb, edge_threshold=EDGE_THRESHOLD)

        for f in wf_r['folds']:
            drop_s = f.get('oos_drop', {})
            fc_s = f.get('oos_fc', {})
            print(f"    Fold {f['fold']}: "
                  f"DROP {drop_s.get('pnl', 0):+.1f}% | "
                  f"FC {fc_s.get('pnl', 0):+.1f}% (timeout {fc_s.get('timeout_pct', 0):.0f}%) | "
                  f"patterns {f['is_patterns']}")
        print(f"    Total DROP OOS: {wf_r['drop_total_oos']:+.1f}% "
              f"({wf_r['drop_positive']}/{wf_r['n_folds']} positive)")
        print(f"    Total FC OOS:   {wf_r['fc_total_oos']:+.1f}% "
              f"({wf_r['fc_positive']}/{wf_r['n_folds']} positive)")
        print(f"    Gap (DROP-FC):  {wf_r['gap_pnl']:+.1f}%")

        phase3[f'MB_{mb}'] = wf_r

    results['phase3'] = phase3

    # Find best MAX_BARS by FC OOS
    best_mb_key = max(phase3.keys(), key=lambda k: phase3[k]['fc_total_oos'])
    best_mb = int(best_mb_key.split('_')[1])
    print(f"\n  === Best MAX_BARS by FC OOS: {best_mb} ({best_mb*5/60:.0f}h) ===")

    # ========================================
    # Phase 4: Timeout Pattern Filtering + WF
    # ========================================
    print("\n" + "=" * 60)
    print(f"Phase 4: Timeout Filtering + WF (MAX_BARS={best_mb})")
    print("=" * 60)

    # Already have baseline from Phase 3
    baseline_wf = phase3[best_mb_key]
    print(f"\n  Baseline (no filter): FC OOS {baseline_wf['fc_total_oos']:+.1f}% "
          f"({baseline_wf['fc_positive']}/{baseline_wf['n_folds']} positive)")

    timeout_thresholds = [40, 50, 60, 70]
    phase4 = {
        'best_max_bars': best_mb,
        'baseline': baseline_wf,
    }

    for thr in timeout_thresholds:
        print(f"\n  Timeout filter <= {thr}%...")
        wf_filtered = wf_drop_vs_fc(
            signal_index, opens, highs, lows, closes, n_bars,
            max_bars=best_mb, edge_threshold=EDGE_THRESHOLD,
            timeout_filter_pct=thr
        )

        for f in wf_filtered['folds']:
            fc_s = f.get('oos_fc', {})
            print(f"    Fold {f['fold']}: FC OOS {fc_s.get('pnl', 0):+.1f}%, "
                  f"timeout {fc_s.get('timeout_pct', 0):.0f}%, patterns {f['is_patterns']}")
        print(f"    Total FC OOS: {wf_filtered['fc_total_oos']:+.1f}% "
              f"({wf_filtered['fc_positive']}/{wf_filtered['n_folds']} positive)")

        phase4[f'timeout_le_{thr}'] = wf_filtered

    results['phase4'] = phase4

    # ========================================
    # Phase 4 Ranking
    # ========================================
    print("\n  === Phase 4 Ranking (by FC OOS PnL) ===")
    p4_items = [('no_filter', baseline_wf)]
    for thr in timeout_thresholds:
        p4_items.append((f'timeout_le_{thr}', phase4[f'timeout_le_{thr}']))
    p4_items.sort(key=lambda x: x[1]['fc_total_oos'], reverse=True)
    for rank, (name, wf) in enumerate(p4_items, 1):
        avg_pat = round(np.mean([f['is_patterns'] for f in wf['folds']]), 0)
        print(f"    #{rank} {name}: FC OOS {wf['fc_total_oos']:+.1f}%, "
              f"positive {wf['fc_positive']}/{wf['n_folds']}, "
              f"avg patterns {avg_pat:.0f}")

    # ========================================
    # Phase 5: FC-aware Pattern Discovery + WF
    # ========================================
    print("\n" + "=" * 60)
    print(f"Phase 5: FC-aware Pattern Discovery + WF")
    print("  → grid_search_best_fc (timeout PnL included in optimization)")
    print("  → FC discovery vs DROP discovery comparison")
    print("=" * 60)

    phase5 = {
        'method': 'fc_aware_discovery',
        'description': 'Patterns discovered using FC-aware grid search (timeout PnL included in optimization)',
        'fc_aware_wf': {},
        'comparison': {},
    }

    for mb in top_mbs:
        hours = mb * 5 / 60
        print(f"\n  FC-aware WF (MAX_BARS={mb}, {hours:.0f}h)...")
        wf_fc = wf_fc_aware(signal_index, opens, highs, lows, closes, n_bars,
                            max_bars=mb, edge_threshold=EDGE_THRESHOLD)

        for f in wf_fc['folds']:
            fc_s = f.get('oos_fc', {})
            drop_s = f.get('oos_drop', {})
            print(f"    Fold {f['fold']}: "
                  f"FC OOS {fc_s.get('pnl', 0):+.1f}% (WR {fc_s.get('wr', 0):.1f}%, "
                  f"timeout {fc_s.get('timeout_pct', 0):.0f}%) | "
                  f"DROP OOS {drop_s.get('pnl', 0):+.1f}% | "
                  f"patterns {f['is_patterns']}")
        print(f"    Total FC OOS: {wf_fc['fc_total_oos']:+.1f}% "
              f"({wf_fc['fc_positive']}/{wf_fc['n_folds']} positive)")
        print(f"    Stable patterns: {len(wf_fc['stable_patterns'])}")

        phase5['fc_aware_wf'][f'MB_{mb}'] = wf_fc

    # Comparison table
    print("\n  === Phase 5: DROP-discovery vs FC-discovery (FC OOS) ===")
    for mb in top_mbs:
        k = f'MB_{mb}'
        drop_disc = phase3.get(k, {}).get('fc_total_oos', 0)
        fc_disc = phase5['fc_aware_wf'].get(k, {}).get('fc_total_oos', 0)
        drop_pos = phase3.get(k, {}).get('fc_positive', 0)
        fc_pos = phase5['fc_aware_wf'].get(k, {}).get('fc_positive', 0)

        delta = fc_disc - drop_disc
        winner = 'FC-aware' if fc_disc > drop_disc else 'DROP'

        # Also compare DROP OOS of both discovery modes
        drop_disc_drop_oos = phase3.get(k, {}).get('drop_total_oos', 0)
        fc_disc_drop_oos = phase5['fc_aware_wf'].get(k, {}).get('drop_total_oos', 0)

        print(f"    MB={mb}:")
        print(f"      DROP-disc → FC OOS:  {drop_disc:+.1f}% ({drop_pos}/3)")
        print(f"      FC-disc   → FC OOS:  {fc_disc:+.1f}% ({fc_pos}/3)")
        print(f"      Delta: {delta:+.1f}% → Winner: {winner}")
        print(f"      (DROP-disc → DROP OOS: {drop_disc_drop_oos:+.1f}%, "
              f"FC-disc → DROP OOS: {fc_disc_drop_oos:+.1f}%)")

        phase5['comparison'][k] = {
            'drop_discovery_fc_oos': round(drop_disc, 1),
            'fc_discovery_fc_oos': round(fc_disc, 1),
            'delta': round(delta, 1),
            'winner': winner,
            'drop_discovery_drop_oos': round(drop_disc_drop_oos, 1),
            'fc_discovery_drop_oos': round(fc_disc_drop_oos, 1),
        }

    results['phase5'] = phase5

    # ========================================
    # Summary
    # ========================================
    elapsed = time.time() - t0
    results['total_time_seconds'] = round(elapsed, 1)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Phase 1 — Scanner-Production Gap at 288 bars:")
    print(f"    DROP:  PnL {results['phase1']['drop']['stats']['pnl']:+.1f}%, "
          f"WR {results['phase1']['drop']['stats']['wr']}%")
    print(f"    FC:    PnL {results['phase1']['forced_close']['stats']['pnl']:+.1f}%, "
          f"WR {results['phase1']['forced_close']['stats']['wr']}%")
    print(f"    HOLD:  PnL {results['phase1']['hold']['stats']['pnl']:+.1f}%, "
          f"WR {results['phase1']['hold']['stats']['wr']}%")
    print(f"    FC is {results['phase1']['gap_drop_fc']['pnl']:+.1f}% vs DROP, "
          f"but {results['phase1']['forced_close']['stats']['pnl'] - results['phase1']['hold']['stats']['pnl']:+.1f}% vs HOLD")

    print(f"\n  Phase 2 — Best MAX_BARS by FC PnL/MDD: {ranked[0][1]['max_bars']} "
          f"({ranked[0][1]['fc_pnl_mdd']:.1f}x)")

    print(f"\n  Phase 3 — Best MAX_BARS by WF FC OOS: {best_mb} "
          f"({baseline_wf['fc_total_oos']:+.1f}%)")

    best_p4 = p4_items[0]
    print(f"\n  Phase 4 — Best filter: {best_p4[0]} "
          f"(FC OOS {best_p4[1]['fc_total_oos']:+.1f}%)")

    # Phase 5 summary
    best_fc_mb = max(phase5['fc_aware_wf'].keys(),
                     key=lambda k: phase5['fc_aware_wf'][k]['fc_total_oos'])
    best_fc_oos = phase5['fc_aware_wf'][best_fc_mb]['fc_total_oos']
    print(f"\n  Phase 5 — FC-aware discovery best: {best_fc_mb} "
          f"(FC OOS {best_fc_oos:+.1f}%)")

    # Overall verdict
    best_drop_fc = baseline_wf['fc_total_oos']
    print(f"\n  === VERDICT ===")
    print(f"    DROP-discovery + FC-OOS (Phase 3 best): {best_drop_fc:+.1f}%")
    print(f"    FC-discovery + FC-OOS (Phase 5 best):   {best_fc_oos:+.1f}%")
    delta_overall = best_fc_oos - best_drop_fc
    print(f"    FC-aware discovery delta: {delta_overall:+.1f}%")
    if delta_overall > 0:
        print(f"    → FC-aware discovery IMPROVES production-realistic performance")
    else:
        print(f"    → DROP discovery remains better even for FC evaluation")

    print(f"\n  Total time: {elapsed / 60:.1f} minutes")

    # Save
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=json_safe)
    print(f"\n  Results saved to {RESULT_FILE}")


if __name__ == '__main__':
    main()
