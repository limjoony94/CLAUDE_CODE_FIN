#!/usr/bin/env python3
"""
Phase 5 Only: FC-aware Pattern Discovery + WF
Loads Phase 1-4 results from timeout_forced_close_study.json,
runs only Phase 5, then merges results.

User insight: patterns discovered with DROP mode perform poorly under FC
because they never optimized for timeout trade PnL. FC-aware discovery
includes timeout PnL in grid search optimization.
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
LEVERAGE = scanner.LEVERAGE
FEE_PCT = scanner.FEE_PCT
EDGE_THRESHOLD = 21.8

# Import functions from the main study
from scripts.analysis.timeout_forced_close_study import (
    bt_signals_fc,
    grid_search_best_fc,
    scan_universe_range_fc,
    wf_fc_aware,
    _backtest_fc,
    _backtest_drop,
)


def main():
    t0 = time.time()

    # Load existing results
    if not os.path.exists(RESULT_FILE):
        print("ERROR: No existing results found. Run timeout_forced_close_study.py first.")
        return

    with open(RESULT_FILE) as f:
        results = json.load(f)

    if not all(f'phase{i}' in results for i in range(1, 5)):
        print("ERROR: Phases 1-4 not complete in existing results.")
        return

    print("=" * 60)
    print("Phase 5 Only: FC-aware Pattern Discovery + WF")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = scanner.load_and_classify(scanner.DEFAULT_DATA_FILE)
    types = df['candle_type'].tolist()
    signal_index = scanner.build_signal_index(types, len(df))
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    print(f"  {n_bars} bars, {n_bars * 5 / 60 / 24:.0f} days")

    # Reconstruct key variables from existing results
    phase3 = results['phase3']
    top_mbs = sorted([int(k.split('_')[1]) for k in phase3.keys()])
    best_mb = int(results['phase4']['best_max_bars'])
    baseline_wf = phase3[f'MB_{best_mb}']

    print(f"\n  Existing Phase 3 results:")
    for mb in top_mbs:
        k = f'MB_{mb}'
        fc_oos = phase3[k]['fc_total_oos']
        drop_oos = phase3[k]['drop_total_oos']
        fc_pos = phase3[k]['fc_positive']
        print(f"    MB={mb}: DROP OOS {drop_oos:+.1f}%, FC OOS {fc_oos:+.1f}% ({fc_pos}/3)")

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

    # Merge and save
    results['phase5'] = phase5

    elapsed = time.time() - t0
    # Update total time to include Phase 5
    results['total_time_seconds'] = round(
        results.get('total_time_seconds', 0) + elapsed, 1
    )

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 5 SUMMARY")
    print("=" * 60)

    best_fc_mb = max(phase5['fc_aware_wf'].keys(),
                     key=lambda k: phase5['fc_aware_wf'][k]['fc_total_oos'])
    best_fc_oos = phase5['fc_aware_wf'][best_fc_mb]['fc_total_oos']
    best_fc_pos = phase5['fc_aware_wf'][best_fc_mb]['fc_positive']
    print(f"\n  FC-aware discovery best: {best_fc_mb} "
          f"(FC OOS {best_fc_oos:+.1f}%, {best_fc_pos}/3 positive)")

    best_drop_fc = baseline_wf['fc_total_oos']
    print(f"\n  === VERDICT ===")
    print(f"    DROP-discovery + FC-OOS (Phase 3 best, MB={best_mb}): {best_drop_fc:+.1f}%")
    print(f"    FC-discovery + FC-OOS (Phase 5 best, {best_fc_mb}):   {best_fc_oos:+.1f}%")
    delta_overall = best_fc_oos - best_drop_fc
    print(f"    FC-aware discovery delta: {delta_overall:+.1f}%")
    if delta_overall > 0:
        print(f"    → FC-aware discovery IMPROVES production-realistic performance")
    else:
        print(f"    → DROP discovery remains better even for FC evaluation")

    print(f"\n  Phase 5 time: {elapsed / 60:.1f} minutes")

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
