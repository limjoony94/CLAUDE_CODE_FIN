#!/usr/bin/env python3
"""
Direction Cap Sweep Study — Cluster Damage Mitigation Analysis

Sweeps direction_cap from 3 to 9 (no cap) to evaluate impact on:
- IS metrics (WR, PnL, MDD, PnL/MDD)
- Cluster damage (max_corr_loss)
- Direction distribution and slot utilization
- WF OOS performance (3-fold expanding window)
- Entry rejections from cap

Standard Research Protocol enforced:
- Entry: next bar Open after signal
- Exit: Intrabar High/Low (distance-based)
- Sizing: Compound (via portfolio_npos)
- Fee: 0.10% (0.05% x 2), applied as FEE_PCT * LEVERAGE
- Slippage: 0.02% buffer (in ATR scaling)
- Leverage: 3x
- Timeout: 288 bars (DROP)
- Production classify_candle import
- tp_scale_factor: 0.72
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import Counter

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, portfolio_npos,
    compute_atr_ratio, compute_ema_slope, bt_signals_atr,
    calc_stats_compound, mc_test, find_neutral_window,
    FEE_PCT, LEVERAGE, MAX_BARS,
    DEFAULT_N_SLOTS, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    TIMEOUT_BARS,
)
from scripts.production.pattern_5m.indicators import classify_candle

# ============================================================
# Study Constants
# ============================================================
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'direction_cap_sweep.json')

N_SLOTS = 9
TP_SCALE_FACTOR = 0.72
BASELINE_CAP = 7
SWEEP_CAPS = [3, 4, 5, 6, 7, 8, 9]
WF_FOLDS = 3
MC_SEEDS = [42, 123, 7]


def load_patterns(patterns_file):
    """Load dynamic patterns and apply tp_scale_factor."""
    with open(patterns_file) as f:
        data = json.load(f)

    pattern_details = data['pattern_details']
    long_pats = data['patterns']['long']
    short_pats = data['patterns']['short']

    # Build signal tuples format: {pattern: (direction, tp, sl)}
    # pattern_details keys are "PATTERN_DIRECTION" format (e.g., "BD-BD-BU_LONG")
    pat_config = {}
    for pat in long_pats:
        key = f"{pat}_LONG"
        if key in pattern_details:
            d = pattern_details[key]
            tp = d['tp'] * TP_SCALE_FACTOR
            tp = max(tp, 0.3)  # floor
            sl = d['sl']
            pat_config[pat] = ('LONG', tp, sl)
    for pat in short_pats:
        key = f"{pat}_SHORT"
        if key in pattern_details:
            d = pattern_details[key]
            tp = d['tp'] * TP_SCALE_FACTOR
            tp = max(tp, 0.3)
            sl = d['sl']
            pat_config[pat] = ('SHORT', tp, sl)

    return pat_config


def build_signal_tuples(signal_index, pat_config, n_bars):
    """Build signal tuples for portfolio_npos: (signal_bar, pattern, direction, tp, sl)."""
    tuples = []
    for pat, bars in signal_index.items():
        if pat not in pat_config:
            continue
        direction, tp, sl = pat_config[pat]
        for bar in bars:
            if bar + 1 < n_bars:
                tuples.append((bar, pat, direction, tp, sl))
    tuples.sort(key=lambda x: x[0])
    return tuples


def run_npos_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                 atr_ratio, ema_slope, start_bar, end_bar, direction_cap):
    """Run N-pos simulation with given direction_cap, return trades + stats."""
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        n_slots=N_SLOTS,
        direction_cap=direction_cap,
        regime_mult=DEFAULT_REGIME_MULT,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
        agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
        momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        clamp_lo=DEFAULT_ATR_CLAMP_LO,
        clamp_hi=DEFAULT_ATR_CLAMP_HI,
        timeout_bars=TIMEOUT_BARS,
        cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
    )
    return trades, stats


def compute_direction_stats(trades):
    """Compute direction distribution from trades."""
    if not trades:
        return {'long_trades': 0, 'short_trades': 0, 'long_pct': 0, 'short_pct': 0}
    long_count = sum(1 for t in trades if t['direction'] == 'LONG')
    short_count = sum(1 for t in trades if t['direction'] == 'SHORT')
    total = len(trades)
    return {
        'long_trades': long_count,
        'short_trades': short_count,
        'long_pct': round(long_count / total * 100, 1) if total else 0,
        'short_pct': round(short_count / total * 100, 1) if total else 0,
    }


def compute_rr_ratio(trades):
    """Average win / average loss ratio."""
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [abs(t['pnl_slot']) for t in trades if t['pnl_slot'] < 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 1
    return round(avg_win / avg_loss, 3) if avg_loss > 0 else 0


def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, start_bar, end_bar, direction_cap, n_folds=WF_FOLDS):
    """3-fold expanding window WF. Returns fold PnLs and verdict."""
    n = end_bar - start_bar
    fold_results = []

    for fi in range(n_folds):
        is_end = start_bar + int(n * (fi + 1) / (n_folds + 1))
        oos_start = is_end
        oos_end = start_bar + int(n * (fi + 2) / (n_folds + 1))

        if oos_start >= oos_end:
            fold_results.append({'oos_pnl': 0.0, 'oos_trades': 0})
            continue

        # OOS simulation
        oos_trades, oos_stats = run_npos_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end, direction_cap
        )
        oos_metrics = calc_stats_compound(oos_trades)

        fold_results.append({
            'fold': fi + 1,
            'oos_pnl': oos_metrics['pnl'],
            'oos_wr': oos_metrics['wr'],
            'oos_trades': oos_metrics['trades'],
            'oos_mdd': oos_metrics['mdd'],
        })

    total_oos = sum(f['oos_pnl'] for f in fold_results)
    all_pass = all(f['oos_pnl'] > 0 for f in fold_results)

    return fold_results, total_oos, 'PASS' if all_pass else 'FAIL'


def main():
    print("=" * 70)
    print("Direction Cap Sweep Study")
    print(f"Caps: {SWEEP_CAPS} | Baseline: {BASELINE_CAP}")
    print(f"N-slots: {N_SLOTS} | TP scale: {TP_SCALE_FACTOR}")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading data and classifying candles...")
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values
    n_bars = len(df)
    print(f"  Bars: {n_bars} ({n_bars / 288:.0f} days)")

    # 2. Compute ATR ratio and EMA slope
    print("\n[2/5] Computing ATR ratio and EMA slope...")
    atr_ratio = compute_atr_ratio(highs, lows, closes, atr_period=14, window=576)
    ema_slope = compute_ema_slope(closes)

    # 3. Find neutral window
    print("\n[3/5] Finding neutral window...")
    nw = find_neutral_window(closes, tol_pct=1.0)
    if nw is None:
        print("  WARNING: No neutral window found, using full data")
        start_bar, end_bar = 0, n_bars
    else:
        start_bar, end_bar = nw
        days = (end_bar - start_bar) / 288
        print(f"  Neutral window: bars {start_bar}-{end_bar} ({days:.0f} days)")

    # 4. Build signal index and tuples
    print("\n[4/5] Building signal index...")
    signal_index = build_signal_index(types, n_bars)
    pat_config = load_patterns(PATTERNS_FILE)
    signal_tuples = build_signal_tuples(signal_index, pat_config, n_bars)
    print(f"  Patterns: {len(pat_config)} | Total signals: {len(signal_tuples)}")

    # Filter signals to neutral window
    signals_in_window = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                         if start_bar <= s < end_bar]
    print(f"  Signals in neutral window: {len(signals_in_window)}")

    # 5. Sweep
    print("\n[5/5] Running direction cap sweep...")
    print("-" * 70)

    sweep_results = []

    for cap in SWEEP_CAPS:
        print(f"\n  Cap={cap}:")

        # IS simulation (full neutral window)
        trades, stats = run_npos_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar, cap
        )

        is_metrics = calc_stats_compound(trades)
        dir_stats = compute_direction_stats(trades)
        rr = compute_rr_ratio(trades)

        # Slot utilization: approximate from trades timing
        # Count average occupied slots per bar
        if trades:
            bar_slots = Counter()
            for t in trades:
                for b in range(t['entry_bar'], t['exit_bar'] + 1):
                    bar_slots[b] += 1
            active_bars = end_bar - start_bar
            avg_slots = sum(bar_slots.values()) / active_bars if active_bars > 0 else 0
            slot_util = avg_slots / N_SLOTS
        else:
            slot_util = 0

        # WF
        wf_folds, wf_total, wf_verdict = run_wf(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar, cap
        )

        # MC test on IS trades
        if trades:
            pnls = [t['pnl_portfolio'] for t in trades]
            mc_p = mc_test(pnls)
        else:
            mc_p = 1.0

        result = {
            'direction_cap': cap,
            'is_wr': is_metrics['wr'],
            'is_pnl': is_metrics['pnl'],
            'is_mdd': stats['mdd_mtm'],
            'is_pnl_mdd': round(is_metrics['pnl'] / stats['mdd_mtm'], 2) if stats['mdd_mtm'] > 0 else 0,
            'trades': is_metrics['trades'],
            'trades_per_day': round(is_metrics['trades'] / ((end_bar - start_bar) / 288), 2),
            'max_corr_loss': stats['max_corr_loss'],
            'corr_events': stats['corr_events'],
            'rr_ratio': rr,
            'long_trades': dir_stats['long_trades'],
            'short_trades': dir_stats['short_trades'],
            'long_pct': dir_stats['long_pct'],
            'short_pct': dir_stats['short_pct'],
            'entry_rejections_dir_cap': stats['blocked']['dir_cap'],
            'entry_rejections_all': stats['blocked'],
            'slot_utilization': round(slot_util, 3),
            'max_sim_positions': stats['max_sim_positions'],
            'mc_pvalue': round(mc_p, 4),
            'wf_oos_folds': wf_folds,
            'wf_oos_total': round(wf_total, 2),
            'wf_verdict': wf_verdict,
        }

        sweep_results.append(result)

        marker = " <-- BASELINE" if cap == BASELINE_CAP else ""
        print(f"    IS: WR {is_metrics['wr']}%, PnL +{is_metrics['pnl']:.1f}%, "
              f"MDD(MTM) {stats['mdd_mtm']:.2f}%, PnL/MDD {result['is_pnl_mdd']:.1f}x")
        print(f"    Trades: {is_metrics['trades']} ({result['trades_per_day']}/day), "
              f"MaxCorrLoss: {stats['max_corr_loss']:.2f}%, CorrEvents: {stats['corr_events']}")
        print(f"    DirCap rejections: {stats['blocked']['dir_cap']}, "
              f"Slot util: {slot_util:.3f}, MaxSimPos: {stats['max_sim_positions']}")
        fold_strs = [f"{f['oos_pnl']:.1f}" for f in wf_folds]
        print(f"    WF OOS: {wf_total:.1f}% ({wf_verdict}), "
              f"Folds: [{', '.join(fold_strs)}]")
        print(f"    R:R {rr}, L/S: {dir_stats['long_trades']}/{dir_stats['short_trades']}, "
              f"MC p={mc_p:.4f}{marker}")

    # Recommendation
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find best by PnL/MDD among WF PASS candidates
    passing = [r for r in sweep_results if r['wf_verdict'] == 'PASS']
    if passing:
        best = max(passing, key=lambda r: r['is_pnl_mdd'])
        baseline_r = next(r for r in sweep_results if r['direction_cap'] == BASELINE_CAP)

        # Compare best vs baseline
        if best['direction_cap'] == BASELINE_CAP:
            recommendation = (
                f"KEEP_BASELINE (cap={BASELINE_CAP}). "
                f"No WF-passing cap improves over baseline PnL/MDD {baseline_r['is_pnl_mdd']:.1f}x."
            )
        else:
            pnl_mdd_delta = best['is_pnl_mdd'] - baseline_r['is_pnl_mdd']
            corr_delta = best['max_corr_loss'] - baseline_r['max_corr_loss']
            recommendation = (
                f"Cap={best['direction_cap']} is best WF-passing candidate: "
                f"PnL/MDD {best['is_pnl_mdd']:.1f}x (vs baseline {baseline_r['is_pnl_mdd']:.1f}x, "
                f"delta {pnl_mdd_delta:+.1f}x), "
                f"MaxCorrLoss {best['max_corr_loss']:.2f}% (vs {baseline_r['max_corr_loss']:.2f}%, "
                f"delta {corr_delta:+.2f}pp), "
                f"WF OOS {best['wf_oos_total']:.1f}% (vs {baseline_r['wf_oos_total']:.1f}%)."
            )
            # Check if cluster damage improved
            if corr_delta < 0:
                recommendation += f" Cluster damage reduced by {abs(corr_delta):.2f}pp."
            else:
                recommendation += f" WARNING: Cluster damage NOT reduced."
    else:
        recommendation = "NO WF-PASSING candidates found. All caps failed WF."

    print(f"\nRecommendation: {recommendation}")

    # Print comparison table
    print(f"\n{'Cap':>4} | {'WR':>5} | {'PnL':>8} | {'MDD':>6} | {'P/M':>6} | "
          f"{'CorrL':>6} | {'DirRej':>6} | {'Trades':>6} | {'WF_OOS':>8} | {'WF':>4}")
    print("-" * 85)
    for r in sweep_results:
        marker = "*" if r['direction_cap'] == BASELINE_CAP else " "
        print(f"{r['direction_cap']:>3}{marker} | {r['is_wr']:>5.1f} | {r['is_pnl']:>+8.1f} | "
              f"{r['is_mdd']:>6.2f} | {r['is_pnl_mdd']:>6.1f} | "
              f"{r['max_corr_loss']:>6.2f} | {r['entry_rejections_dir_cap']:>6} | "
              f"{r['trades']:>6} | {r['wf_oos_total']:>+8.1f} | {r['wf_verdict']:>4}")

    # Save results
    output = {
        'study': 'direction_cap_sweep',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'baseline_cap': BASELINE_CAP,
        'tp_scale_factor': TP_SCALE_FACTOR,
        'n_slots': N_SLOTS,
        'data_bars': n_bars,
        'neutral_window': {'start': start_bar, 'end': end_bar,
                           'days': round((end_bar - start_bar) / 288, 1)},
        'patterns': len(pat_config),
        'signals_in_window': len(signals_in_window),
        'protocol': {
            'entry': 'next_bar_open',
            'exit': 'intrabar_high_low_distance',
            'sizing': 'compound',
            'fee_pct': FEE_PCT,
            'leverage': LEVERAGE,
            'timeout_bars': TIMEOUT_BARS,
            'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
            'agg_risk': f'{DEFAULT_AGG_RISK_COUNTER}/{DEFAULT_AGG_RISK_WITH}',
            'momentum': f'{DEFAULT_MOMENTUM_THRESHOLD}%/{DEFAULT_MOMENTUM_LOOKBACK}b/{DEFAULT_MOMENTUM_COOLDOWN}b',
            'atr_clamp': f'[{DEFAULT_ATR_CLAMP_LO}, {DEFAULT_ATR_CLAMP_HI}]',
            'regime_mult': DEFAULT_REGIME_MULT,
            'wf_folds': WF_FOLDS,
            'wf_method': 'expanding_window',
        },
        'sweep_results': sweep_results,
        'recommendation': recommendation,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
