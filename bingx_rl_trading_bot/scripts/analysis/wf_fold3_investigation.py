#!/usr/bin/env python3
"""
WF Fold 3 = 0.0% Investigation
================================
All strategies show Fold 3 OOS PnL = 0.0% in WF validation.
This is extremely suspicious — investigate root cause.

Hypotheses:
  H1: Zero trades in Fold 3 OOS window (no pattern signals in that period)
  H2: Data boundary issue (OOS window extends beyond data)
  H3: Market regime shift making all patterns unprofitable
  H4: Signal construction bug (patterns don't match in last segment)
  H5: portfolio_npos returns empty/zero for that window

Tests:
  1. Examine exact fold boundaries (IS/OOS bar ranges)
  2. Count signals per fold OOS window
  3. Run 1-pos backtest on each fold OOS to compare
  4. Check if N-pos simulation has trades in Fold 3
  5. Vary fold count (4, 5) to see if issue persists
  6. Check data completeness in last segment
"""

import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, portfolio_npos,
    compute_atr_ratio, compute_ema_slope, calc_stats_compound,
)

DATA_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_JSON = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')

# Config matching production
N_SLOTS = 9
DIR_CAP = 7
AGG_COUNTER = 8.0
AGG_WITH = 15.0
TIMEOUT = 288
CASCADE_PCT = 85
MOMENTUM_LB = 3
MOMENTUM_TH = 1.5
MOMENTUM_CD = 12
TP_SCALE = 0.5
SL_SCALE = 1.1

def load_data():
    df = load_and_classify(DATA_CSV)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values
    return df, opens, highs, lows, closes, types

def load_patterns():
    with open(PATTERNS_JSON) as f:
        data = json.load(f)
    patterns = {}
    for key, det in data.get('pattern_details', {}).items():
        pat = det['pattern']
        direction = det['direction']
        tp = round(max(0.3, det['tp'] * TP_SCALE), 3)
        sl = round(max(0.5, det['sl'] * SL_SCALE), 3)
        patterns[f"{pat}_{direction}"] = {'tp': tp, 'sl': sl, 'direction': direction, 'pattern': pat}
    return patterns

def build_signals(patterns, signal_index):
    tuples = []
    for key, info in patterns.items():
        parts = key.rsplit('_', 1)
        pat_name, direction = parts[0], parts[1]
        if pat_name in signal_index:
            for sig_bar in signal_index[pat_name]:
                tuples.append((sig_bar, pat_name, direction, info['tp'], info['sl']))
    return sorted(tuples, key=lambda x: x[0])

def run_npos(signal_tuples, opens, highs, lows, closes, atr_ratio, ema_slope, start_bar, end_bar,
             cascade_pct=CASCADE_PCT, momentum_enabled=True):
    if not signal_tuples:
        return [], {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}
    n_bars = len(opens)
    trades, raw_stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
        n_slots=N_SLOTS, direction_cap=DIR_CAP,
        agg_risk_counter=AGG_COUNTER, agg_risk_with=AGG_WITH,
        timeout_bars=TIMEOUT,
        cascade_tighten_pct=cascade_pct,
        momentum_lookback=MOMENTUM_LB if momentum_enabled else 3,
        momentum_threshold=MOMENTUM_TH if momentum_enabled else 99.0,
        momentum_cooldown=MOMENTUM_CD if momentum_enabled else 0,
    )
    # Filter non-drop trades and compute compound stats
    active_trades = [t for t in trades if not t.get('drop', False)]
    stats = calc_stats_compound(active_trades)
    stats['mdd_mtm'] = raw_stats.get('mdd_mtm', 0)
    stats['blocked'] = raw_stats.get('blocked', {})
    return active_trades, stats

def expanding_window_folds(total_bars, n_folds=3, min_is_pct=0.4):
    """Generate expanding window fold boundaries."""
    folds = []
    oos_size = total_bars // (n_folds + 1)
    for i in range(n_folds):
        is_end = total_bars - (n_folds - i) * oos_size
        oos_start = is_end
        oos_end = is_end + oos_size
        if i == n_folds - 1:
            oos_end = total_bars
        folds.append({
            'fold': i + 1,
            'is_start': 0,
            'is_end': is_end,
            'oos_start': oos_start,
            'oos_end': oos_end,
        })
    return folds

def main():
    t0 = time.time()
    df, opens, highs, lows, closes, types = load_data()
    total_bars = len(df)
    patterns = load_patterns()
    signal_index = build_signal_index(types, len(types))
    all_signals = build_signals(patterns, signal_index)

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    print(f"Data: {total_bars} bars, {total_bars/288:.0f} days")
    print(f"Patterns: {len(patterns)}, Signals: {len(all_signals)}")
    print()

    results = {}

    # ================================================================
    # TEST 1: Examine fold boundaries (3-fold expanding window)
    # ================================================================
    print("=" * 60)
    print("TEST 1: Fold Boundaries (3-fold Expanding Window)")
    print("=" * 60)

    folds = expanding_window_folds(total_bars, n_folds=3)
    for fold in folds:
        is_days = (fold['is_end'] - fold['is_start']) / 288
        oos_days = (fold['oos_end'] - fold['oos_start']) / 288
        oos_start_date = df.iloc[fold['oos_start']]['open_time'] if 'open_time' in df.columns else f"bar {fold['oos_start']}"
        oos_end_idx = min(fold['oos_end'] - 1, len(df) - 1)
        oos_end_date = df.iloc[oos_end_idx]['open_time'] if 'open_time' in df.columns else f"bar {oos_end_idx}"

        # Count signals in OOS window
        oos_signals = [s for s in all_signals if fold['oos_start'] <= s[0] < fold['oos_end']]

        print(f"  Fold {fold['fold']}:")
        print(f"    IS:  bar {fold['is_start']:6d} → {fold['is_end']:6d} ({is_days:.0f} days)")
        print(f"    OOS: bar {fold['oos_start']:6d} → {fold['oos_end']:6d} ({oos_days:.0f} days)")
        print(f"    OOS dates: {oos_start_date} → {oos_end_date}")
        print(f"    OOS signals: {len(oos_signals)}")
        print()

    results['fold_boundaries'] = folds

    # ================================================================
    # TEST 2: N-pos simulation per fold OOS
    # ================================================================
    print("=" * 60)
    print("TEST 2: N-pos Simulation Per Fold")
    print("=" * 60)

    fold_results = []
    for fold in folds:
        # IS simulation
        is_signals = [s for s in all_signals if s[0] < fold['is_end']]
        is_trades, is_stats = run_npos(is_signals, opens, highs, lows, closes,
                                        atr_ratio, ema_slope, 0, fold['is_end'])

        # OOS simulation
        oos_signals = [s for s in all_signals if fold['oos_start'] <= s[0] < fold['oos_end']]
        oos_trades, oos_stats = run_npos(oos_signals, opens, highs, lows, closes,
                                          atr_ratio, ema_slope, fold['oos_start'], fold['oos_end'])

        is_pnl = is_stats.get('pnl', 0)
        oos_pnl = oos_stats.get('pnl', 0)
        is_trades_n = is_stats.get('trades', 0)
        oos_trades_n = oos_stats.get('trades', 0)
        oos_wr = oos_stats.get('wr', 0)

        print(f"  Fold {fold['fold']}:")
        print(f"    IS:  {is_trades_n} trades, PnL={is_pnl:+.1f}%")
        print(f"    OOS: {oos_trades_n} trades, PnL={oos_pnl:+.1f}%, WR={oos_wr:.1f}%")

        fold_results.append({
            'fold': fold['fold'],
            'is_trades': is_trades_n, 'is_pnl': round(is_pnl, 2),
            'oos_trades': oos_trades_n, 'oos_pnl': round(oos_pnl, 2),
            'oos_wr': round(oos_wr, 2),
            'oos_signals': len(oos_signals),
        })

    print()
    results['fold_npos'] = fold_results

    # ================================================================
    # TEST 3: Fold 3 OOS deep dive — per-direction breakdown
    # ================================================================
    print("=" * 60)
    print("TEST 3: Fold 3 OOS Deep Dive")
    print("=" * 60)

    fold3 = folds[2]
    oos3_signals = [s for s in all_signals if fold3['oos_start'] <= s[0] < fold3['oos_end']]

    # Direction breakdown
    long_sigs = [s for s in oos3_signals if s[2] == 'LONG']
    short_sigs = [s for s in oos3_signals if s[2] == 'SHORT']
    print(f"  Fold 3 OOS signals: {len(oos3_signals)} (LONG={len(long_sigs)}, SHORT={len(short_sigs)})")

    # Run with relaxed settings to isolate cause
    print(f"\n  Relaxed settings test (no cascade, no momentum, no agg_risk):")
    if oos3_signals:
        n_bars = len(opens)
        trades_relaxed, stats_relaxed = portfolio_npos(
            oos3_signals, opens, highs, lows, closes, n_bars,
            atr_ratio=atr_ratio, ema_slope=ema_slope,
            start_bar=fold3['oos_start'], end_bar=fold3['oos_end'],
            n_slots=N_SLOTS, direction_cap=DIR_CAP,
            agg_risk_counter=99.0, agg_risk_with=99.0,
            timeout_bars=TIMEOUT,
            cascade_tighten_pct=0,
            momentum_lookback=3, momentum_threshold=99.0, momentum_cooldown=0,
        )
        relaxed_active = [t for t in trades_relaxed if not t.get('drop', False)]
        stats_relaxed = calc_stats_compound(relaxed_active)
        print(f"    Relaxed: {stats_relaxed.get('trades', 0)} trades, PnL={stats_relaxed.get('pnl', 0):+.1f}%")
    else:
        print(f"    No signals — cannot run simulation")
        stats_relaxed = {}

    # 1-pos simple backtest on Fold 3 OOS
    print(f"\n  1-pos simple backtest (no portfolio):")
    wins, losses, pnls = 0, 0, []
    for sig_bar, pat, direction, tp, sl in oos3_signals:
        entry_bar = sig_bar + 1
        if entry_bar >= fold3['oos_end'] or entry_bar >= len(opens):
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        tp_price = entry * (1 + tp/100) if direction == 'LONG' else entry * (1 - tp/100)
        sl_price = entry * (1 - sl/100) if direction == 'LONG' else entry * (1 + sl/100)

        hit_tp, hit_sl = False, False
        for bar in range(entry_bar, min(entry_bar + TIMEOUT, fold3['oos_end'])):
            if bar >= len(highs):
                break
            if direction == 'LONG':
                if lows[bar] <= sl_price:
                    hit_sl = True
                    break
                if highs[bar] >= tp_price:
                    hit_tp = True
                    break
            else:
                if highs[bar] >= sl_price:
                    hit_sl = True
                    break
                if lows[bar] <= tp_price:
                    hit_tp = True
                    break

        if hit_tp:
            pnl = tp - 0.10  # fee
            wins += 1
        elif hit_sl:
            pnl = -sl - 0.10
            losses += 1
        else:
            # Timeout — use price at timeout bar
            timeout_bar = min(entry_bar + TIMEOUT - 1, fold3['oos_end'] - 1, len(closes) - 1)
            exit_price = closes[timeout_bar]
            if direction == 'LONG':
                pnl = (exit_price / entry - 1) * 100 - 0.10
            else:
                pnl = (1 - exit_price / entry) * 100 - 0.10
            if pnl >= 0:
                wins += 1
            else:
                losses += 1
        pnls.append(pnl)

    total_1pos = wins + losses
    wr_1pos = wins / total_1pos * 100 if total_1pos > 0 else 0
    cum_pnl_1pos = sum(pnls) if pnls else 0
    print(f"    1-pos: {total_1pos} trades, WR={wr_1pos:.1f}%, cumPnL={cum_pnl_1pos:+.1f}%")

    results['fold3_deep'] = {
        'oos_signals': len(oos3_signals),
        'long_signals': len(long_sigs),
        'short_signals': len(short_sigs),
        'relaxed_trades': stats_relaxed.get('trades', 0),
        'relaxed_pnl': round(stats_relaxed.get('pnl', 0), 2),
        '1pos_trades': total_1pos,
        '1pos_wr': round(wr_1pos, 2),
        '1pos_cum_pnl': round(cum_pnl_1pos, 2),
    }

    # ================================================================
    # TEST 4: Market regime in Fold 3 OOS
    # ================================================================
    print(f"\n{'='*60}")
    print("TEST 4: Market Regime in Fold 3 OOS")
    print("=" * 60)

    oos3_closes = closes[fold3['oos_start']:fold3['oos_end']]
    if len(oos3_closes) > 1:
        period_return = (oos3_closes[-1] / oos3_closes[0] - 1) * 100
        period_volatility = np.std(np.diff(oos3_closes) / oos3_closes[:-1]) * 100
        max_dd_period = 0
        peak = oos3_closes[0]
        for c in oos3_closes:
            if c > peak:
                peak = c
            dd = (peak - c) / peak * 100
            if dd > max_dd_period:
                max_dd_period = dd

        print(f"  Period return: {period_return:+.2f}%")
        print(f"  5m bar volatility: {period_volatility:.4f}%")
        print(f"  Max drawdown: {max_dd_period:.2f}%")
        print(f"  Start price: {oos3_closes[0]:.0f}, End price: {oos3_closes[-1]:.0f}")

        results['fold3_regime'] = {
            'period_return': round(period_return, 2),
            'volatility': round(period_volatility, 4),
            'max_dd': round(max_dd_period, 2),
        }

    # ================================================================
    # TEST 5: Vary fold count (4, 5 folds)
    # ================================================================
    print(f"\n{'='*60}")
    print("TEST 5: Different Fold Counts")
    print("=" * 60)

    vary_results = {}
    for n_folds in [3, 4, 5]:
        var_folds = expanding_window_folds(total_bars, n_folds=n_folds)
        oos_pnls = []
        for fold in var_folds:
            oos_sigs = [s for s in all_signals if fold['oos_start'] <= s[0] < fold['oos_end']]
            if oos_sigs:
                _, stats = run_npos(oos_sigs, opens, highs, lows, closes,
                                    atr_ratio, ema_slope, fold['oos_start'], fold['oos_end'])
                oos_pnls.append(round(stats.get('pnl', 0), 1))
            else:
                oos_pnls.append(0.0)

        total_oos = sum(oos_pnls)
        passes = sum(1 for p in oos_pnls if p > 0)
        print(f"  {n_folds}-fold: {oos_pnls} → Total={total_oos:+.1f}% ({passes}/{n_folds} PASS)")
        vary_results[f'{n_folds}_fold'] = {'oos_pnls': oos_pnls, 'total': round(total_oos, 1), 'passes': passes}

    results['vary_folds'] = vary_results

    # ================================================================
    # TEST 6: Fold 3 IS training → does it find patterns profitable?
    # ================================================================
    print(f"\n{'='*60}")
    print("TEST 6: IS Profitability by Data Segment")
    print("=" * 60)

    # Split data into quarters
    q_size = total_bars // 4
    for qi in range(4):
        q_start = qi * q_size
        q_end = (qi + 1) * q_size if qi < 3 else total_bars
        q_signals = [s for s in all_signals if q_start <= s[0] < q_end]
        if q_signals:
            _, q_stats = run_npos(q_signals, opens, highs, lows, closes,
                                  atr_ratio, ema_slope, q_start, q_end)
            q_days = (q_end - q_start) / 288
            print(f"  Q{qi+1} (bar {q_start}-{q_end}, {q_days:.0f}d): "
                  f"{q_stats.get('trades', 0)} trades, PnL={q_stats.get('pnl', 0):+.1f}%, "
                  f"WR={q_stats.get('wr', 0):.1f}%")
        else:
            print(f"  Q{qi+1}: 0 signals")

    # ================================================================
    # TEST 7: Check if issue is OOS-only vs full window
    # ================================================================
    print(f"\n{'='*60}")
    print("TEST 7: Full Window vs OOS-Only for Fold 3")
    print("=" * 60)

    fold3 = folds[2]
    # Full simulation from start to end of Fold 3 OOS (IS+OOS)
    full_sigs = [s for s in all_signals if s[0] < fold3['oos_end']]
    full_trades, full_stats = run_npos(full_sigs, opens, highs, lows, closes,
                                        atr_ratio, ema_slope, 0, fold3['oos_end'])

    # Count how many trades exit within OOS window
    oos_exit_trades = []
    for t in full_trades:
        if isinstance(t, dict):
            exit_bar = t.get('exit_bar', 0)
            entry_bar = t.get('entry_bar', 0)
        else:
            # Tuple format
            entry_bar = t[0] if len(t) > 0 else 0
            exit_bar = t[1] if len(t) > 1 else 0

        if exit_bar >= fold3['oos_start']:
            oos_exit_trades.append(t)

    print(f"  Full window (0→{fold3['oos_end']}): {full_stats.get('trades', 0)} trades, "
          f"PnL={full_stats.get('pnl', 0):+.1f}%")
    print(f"  Trades exiting in OOS window: {len(oos_exit_trades)}")
    print(f"  OOS-only (isolated): {fold_results[2]['oos_trades']} trades, "
          f"PnL={fold_results[2]['oos_pnl']:+.1f}%")

    # The key insight: OOS-only simulation starts fresh with no positions
    # vs full simulation has positions carried over from IS
    print(f"\n  ⚠ KEY QUESTION: Does portfolio_npos handle OOS-only window differently?")
    print(f"    Full sim includes IS-period signals that may still be open at OOS start")
    print(f"    OOS-only sim starts fresh with 0 positions at OOS start bar")

    results['fold3_full_vs_oos'] = {
        'full_trades': full_stats.get('trades', 0),
        'full_pnl': round(full_stats.get('pnl', 0), 2),
        'oos_exit_trades': len(oos_exit_trades),
        'oos_only_trades': fold_results[2]['oos_trades'],
        'oos_only_pnl': fold_results[2]['oos_pnl'],
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)

    f3 = fold_results[2]
    print(f"\n  Fold 3 OOS:")
    print(f"    Signals: {f3['oos_signals']}")
    print(f"    N-pos trades: {f3['oos_trades']}, PnL: {f3['oos_pnl']:+.1f}%")
    print(f"    1-pos trades: {results['fold3_deep']['1pos_trades']}, "
          f"WR: {results['fold3_deep']['1pos_wr']:.1f}%, "
          f"PnL: {results['fold3_deep']['1pos_cum_pnl']:+.1f}%")

    if f3['oos_trades'] == 0 and f3['oos_signals'] > 0:
        print(f"\n  🔴 ROOT CAUSE: Signals exist ({f3['oos_signals']}) but N-pos produces 0 trades")
        print(f"     → portfolio_npos filtering out all signals in this window")
        print(f"     Check: start_bar/end_bar boundary, signal_bar >= start_bar filter")
    elif f3['oos_trades'] == 0 and f3['oos_signals'] == 0:
        print(f"\n  🔴 ROOT CAUSE: Zero signals in Fold 3 OOS window")
        print(f"     → No patterns match in the last data segment")
    elif f3['oos_pnl'] == 0:
        print(f"\n  🟡 ISSUE: Trades exist but PnL rounds to 0.0%")
        print(f"     → Possible compound equity calculation issue")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'wf_fold3_investigation.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {os.path.abspath(out)}")
    print(f"Elapsed: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
