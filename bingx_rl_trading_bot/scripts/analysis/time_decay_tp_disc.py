#!/usr/bin/env python3
"""
Time-Decay TP Discrimination Test
===================================
Question: Is linear_144's OOS +11.6% improvement over baseline genuine,
or would random pattern subsets also show similar improvement?

Method: For N random seeds, randomly subsample 70% of patterns,
run WF for both baseline and linear_144, count how often linear_144 wins.
If linear_144 wins in most random subsets, the mechanism has genuine
structural value (not pattern-dependent).

Additional check: Shuffle pattern directions to test if improvement
persists even with random directions.

Protocol: Standard Research Protocol
Output: results/time_decay_tp_disc.json
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Reuse everything from the main study
from scripts.analysis.time_decay_tp import (
    load_data_and_patterns,
    portfolio_npos_decay,
    decay_none,
    decay_linear_144,
    TP_SCALE_FACTOR,
    DECAY_VARIANTS,
)
from scripts.scanner.pattern_scanner import (
    compute_atr_ratio,
    compute_ema_slope,
    calc_stats_compound,
    FEE_PCT,
    LEVERAGE,
)

OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'time_decay_tp_disc.json')

N_SIMS = 30
SUBSAMPLE_RATIO = 0.70
N_FOLDS = 3
SEEDS = list(range(42, 42 + N_SIMS))


def run_wf_for_variant(data, decay_fn, pattern_keys, n_folds=3):
    """Run WF validation for a given decay function with specific pattern subset.
    Returns total OOS PnL."""
    nw_start = data['nw_start']
    nw_end = data['nw_end']
    nw_len = nw_end - nw_start
    seg_size = nw_len // (n_folds + 1)

    total_oos = 0.0

    for fold in range(n_folds):
        is_end = nw_start + (fold + 1) * seg_size
        oos_start = is_end
        oos_end = nw_start + (fold + 2) * seg_size if fold < n_folds - 1 else nw_end

        if oos_start >= oos_end:
            continue

        oos_signal_tuples = []
        for key in pattern_keys:
            info = data['pattern_details'][key]
            pat = info['pattern']
            direction = info['direction']
            tp = info['tp'] * TP_SCALE_FACTOR
            sl = info['sl']
            if pat not in data['signal_index']:
                continue
            for sig_bar in data['signal_index'][pat]:
                if oos_start <= sig_bar < oos_end:
                    oos_signal_tuples.append((sig_bar, pat, direction, tp, sl))

        if not oos_signal_tuples:
            continue

        oos_atr = compute_atr_ratio(
            data['highs'][:oos_end], data['lows'][:oos_end], data['closes'][:oos_end])
        oos_ema = compute_ema_slope(data['closes'][:oos_end])

        trades, npos_stats = portfolio_npos_decay(
            oos_signal_tuples,
            data['opens'], data['highs'], data['lows'], data['closes'],
            oos_end, oos_atr, oos_ema,
            oos_start, oos_end,
            decay_fn=decay_fn,
        )
        stats = calc_stats_compound(trades)
        total_oos += stats['pnl']

    return total_oos


def run_wf_shuffled_directions(data, decay_fn, n_folds=3, rng=None):
    """Run WF with ALL patterns but randomly shuffled directions.
    Tests if improvement is direction-dependent."""
    nw_start = data['nw_start']
    nw_end = data['nw_end']
    nw_len = nw_end - nw_start
    seg_size = nw_len // (n_folds + 1)

    all_keys = list(data['pattern_details'].keys())
    # Create shuffled direction mapping
    directions = [data['pattern_details'][k]['direction'] for k in all_keys]
    rng.shuffle(directions)
    dir_map = {k: d for k, d in zip(all_keys, directions)}

    total_oos = 0.0

    for fold in range(n_folds):
        is_end = nw_start + (fold + 1) * seg_size
        oos_start = is_end
        oos_end = nw_start + (fold + 2) * seg_size if fold < n_folds - 1 else nw_end

        if oos_start >= oos_end:
            continue

        oos_signal_tuples = []
        for key in all_keys:
            info = data['pattern_details'][key]
            pat = info['pattern']
            direction = dir_map[key]  # shuffled
            tp = info['tp'] * TP_SCALE_FACTOR
            sl = info['sl']
            if pat not in data['signal_index']:
                continue
            for sig_bar in data['signal_index'][pat]:
                if oos_start <= sig_bar < oos_end:
                    oos_signal_tuples.append((sig_bar, pat, direction, tp, sl))

        if not oos_signal_tuples:
            continue

        oos_atr = compute_atr_ratio(
            data['highs'][:oos_end], data['lows'][:oos_end], data['closes'][:oos_end])
        oos_ema = compute_ema_slope(data['closes'][:oos_end])

        trades, npos_stats = portfolio_npos_decay(
            oos_signal_tuples,
            data['opens'], data['highs'], data['lows'], data['closes'],
            oos_end, oos_atr, oos_ema,
            oos_start, oos_end,
            decay_fn=decay_fn,
        )
        stats = calc_stats_compound(trades)
        total_oos += stats['pnl']

    return total_oos


def main():
    t_start = time.time()
    print("Time-Decay TP Discrimination Test")
    print("=" * 60)

    data = load_data_and_patterns()
    all_keys = list(data['pattern_details'].keys())
    n_pat = len(all_keys)
    subset_size = int(n_pat * SUBSAMPLE_RATIO)

    print(f"  Patterns: {n_pat} | Subset: {subset_size} ({SUBSAMPLE_RATIO*100:.0f}%)")
    print(f"  Simulations: {N_SIMS} | WF folds: {N_FOLDS}")

    # =========================================================
    # Test 1: Random pattern subsets
    # =========================================================
    print(f"\n{'='*60}")
    print("TEST 1: Random Pattern Subsets (70%)")
    print(f"{'='*60}")

    subset_results = []
    decay_wins = 0

    for i, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed)
        chosen = rng.choice(all_keys, size=subset_size, replace=False).tolist()

        oos_baseline = run_wf_for_variant(data, decay_none, chosen, N_FOLDS)
        oos_decay = run_wf_for_variant(data, decay_linear_144, chosen, N_FOLDS)
        gap = oos_decay - oos_baseline
        win = oos_decay > oos_baseline

        if win:
            decay_wins += 1

        subset_results.append({
            'seed': seed,
            'oos_baseline': round(oos_baseline, 1),
            'oos_decay': round(oos_decay, 1),
            'gap': round(gap, 1),
            'decay_wins': win,
        })

        marker = "✓" if win else "✗"
        print(f"  [{i+1:2d}/{N_SIMS}] seed={seed:3d} | "
              f"base={oos_baseline:+7.1f}% decay={oos_decay:+7.1f}% "
              f"gap={gap:+6.1f}% {marker}")

    win_rate = decay_wins / N_SIMS * 100
    gaps = [r['gap'] for r in subset_results]
    avg_gap = np.mean(gaps)
    med_gap = np.median(gaps)

    print(f"\n  Decay wins: {decay_wins}/{N_SIMS} ({win_rate:.0f}%)")
    print(f"  Avg gap: {avg_gap:+.1f}% | Median gap: {med_gap:+.1f}%")

    # Verdict
    if win_rate >= 80:
        subset_verdict = "DISCRIMINATING"
    elif win_rate >= 60:
        subset_verdict = "MARGINAL"
    else:
        subset_verdict = "NON-DISCRIMINATING"
    print(f"  Verdict: {subset_verdict}")

    # =========================================================
    # Test 2: Shuffled directions (all patterns)
    # =========================================================
    print(f"\n{'='*60}")
    print("TEST 2: Shuffled Directions (mechanical check)")
    print(f"{'='*60}")

    shuffle_results = []
    shuffle_decay_wins = 0

    for i, seed in enumerate(SEEDS[:15]):  # 15 sims for speed
        rng = np.random.RandomState(seed)

        oos_baseline = run_wf_shuffled_directions(data, decay_none, N_FOLDS, rng)
        rng2 = np.random.RandomState(seed)  # same shuffle
        oos_decay = run_wf_shuffled_directions(data, decay_linear_144, N_FOLDS, rng2)
        gap = oos_decay - oos_baseline
        win = oos_decay > oos_baseline

        if win:
            shuffle_decay_wins += 1

        shuffle_results.append({
            'seed': seed,
            'oos_baseline': round(oos_baseline, 1),
            'oos_decay': round(oos_decay, 1),
            'gap': round(gap, 1),
            'decay_wins': win,
        })

        marker = "✓" if win else "✗"
        print(f"  [{i+1:2d}/15] seed={seed:3d} | "
              f"base={oos_baseline:+7.1f}% decay={oos_decay:+7.1f}% "
              f"gap={gap:+6.1f}% {marker}")

    shuffle_win_rate = shuffle_decay_wins / 15 * 100
    shuffle_gaps = [r['gap'] for r in shuffle_results]

    print(f"\n  Decay wins: {shuffle_decay_wins}/15 ({shuffle_win_rate:.0f}%)")
    print(f"  Avg gap: {np.mean(shuffle_gaps):+.1f}% | Median: {np.median(shuffle_gaps):+.1f}%")

    if shuffle_win_rate >= 80:
        shuffle_verdict = "MECHANICAL_IMPROVEMENT"
        print(f"  Verdict: {shuffle_verdict} (decay works regardless of direction → structural)")
    elif shuffle_win_rate >= 60:
        shuffle_verdict = "PARTIAL_MECHANICAL"
        print(f"  Verdict: {shuffle_verdict}")
    else:
        shuffle_verdict = "DIRECTION_DEPENDENT"
        print(f"  Verdict: {shuffle_verdict} (improvement depends on correct patterns)")

    # =========================================================
    # Overall Assessment
    # =========================================================
    print(f"\n{'='*60}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*60}")

    if subset_verdict == "DISCRIMINATING" and shuffle_verdict == "MECHANICAL_IMPROVEMENT":
        overall = "GO — genuine structural improvement"
        print(f"  {overall}")
        print("  linear_144 consistently beats baseline across random subsets")
        print("  AND works with shuffled directions → mechanism is robust")
    elif subset_verdict == "DISCRIMINATING":
        overall = "GO — pattern-robust improvement"
        print(f"  {overall}")
        print("  linear_144 consistently beats baseline across pattern subsets")
    elif subset_verdict == "NON-DISCRIMINATING":
        overall = "STOP — improvement not robust"
        print(f"  {overall}")
        print("  Random pattern subsets don't consistently show improvement")
    else:
        overall = "CAUTION — marginal discrimination"
        print(f"  {overall}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save results
    results = {
        'study': 'time_decay_tp_discrimination',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'variant': 'linear_144',
        'tp_scale_factor': TP_SCALE_FACTOR,
        'protocol': {
            'n_sims': N_SIMS,
            'subsample_ratio': SUBSAMPLE_RATIO,
            'n_folds': N_FOLDS,
            'total_patterns': n_pat,
            'subset_size': subset_size,
        },
        'test1_pattern_subsets': {
            'decay_wins': decay_wins,
            'total_sims': N_SIMS,
            'win_rate_pct': round(win_rate, 1),
            'avg_gap_pct': round(avg_gap, 1),
            'median_gap_pct': round(float(med_gap), 1),
            'verdict': subset_verdict,
            'details': subset_results,
        },
        'test2_shuffled_directions': {
            'decay_wins': shuffle_decay_wins,
            'total_sims': 15,
            'win_rate_pct': round(shuffle_win_rate, 1),
            'avg_gap_pct': round(float(np.mean(shuffle_gaps)), 1),
            'median_gap_pct': round(float(np.median(shuffle_gaps)), 1),
            'verdict': shuffle_verdict,
            'details': shuffle_results,
        },
        'overall': overall,
        'elapsed_seconds': round(elapsed, 1),
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
