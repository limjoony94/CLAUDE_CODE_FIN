"""MDD Verification: Compare old (realized-only) vs new (mark-to-market) MDD.

Runs portfolio_npos with current 131 patterns and reports both MDD values
to quantify the underestimation from the old calculation.
"""
import json
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.scanner.pattern_scanner import (
    portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    load_and_classify, build_signal_index,
    find_neutral_window,
    LEVERAGE, FEE_PCT,
)

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')

def main():
    t0 = time.time()
    print("=== MDD Verification: Realized-only vs Mark-to-Market ===\n")

    # Load and classify data
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values
    n_bars = len(df)
    print(f"Data: {n_bars} bars ({n_bars/288:.0f} days)")

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    # Build signal index
    signal_index = build_signal_index(types, n_bars)

    # Load pattern set
    with open(PATTERNS_FILE) as f:
        pdata = json.load(f)
    patterns = pdata['pattern_details']

    # Build signal tuples from pattern details + signal index
    signal_tuples = []
    for pat_key, pat_info in patterns.items():
        pat_name = pat_info['pattern']
        direction = pat_info['direction']
        tp = pat_info['tp']
        sl = pat_info['sl']
        if pat_name in signal_index:
            for sb in signal_index[pat_name]:
                signal_tuples.append((sb, pat_key, direction, tp, sl))

    print(f"Patterns: {len(patterns)}, Signals: {len(signal_tuples)}")

    # Neutral window
    meta = pdata.get('metadata', {})
    nw_raw = meta.get('neutral_window')
    if nw_raw:
        start_bar, end_bar = nw_raw
    else:
        nw = find_neutral_window(closes)
        if nw:
            start_bar, end_bar = nw
        else:
            start_bar, end_bar = 0, n_bars
    print(f"Neutral window: [{start_bar}, {end_bar}] ({(end_bar-start_bar)/288:.1f} days)")

    # Run portfolio_npos (new: includes mdd_mtm)
    print("\nRunning portfolio_npos (Cascade ON, full config)...")
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
    )
    print(f"  Trades: {len(trades)}, Max sim positions: {stats['max_sim_positions']}")
    print(f"  Max corr loss: {stats['max_corr_loss']}%")
    print(f"  Mark-to-Market MDD: {stats['mdd_mtm']}%")

    # calc_stats_compound (now sorts by exit_bar)
    cstats = calc_stats_compound(trades)
    print(f"\n  calc_stats_compound (exit_bar sorted):")
    print(f"    WR: {cstats['wr']}%")
    print(f"    PnL: {cstats['pnl']}%")
    print(f"    Realized MDD: {cstats['mdd']}%")
    print(f"    PnL/MDD (realized): {cstats['pnl_mdd']}")

    # Also compute old-style (entry_bar sorted) for reference
    sorted_by_entry = sorted(trades, key=lambda x: x['entry_bar'])
    eq = 100.0
    pk = eq
    old_mdd = 0.0
    for t in sorted_by_entry:
        eq += t['pnl_portfolio']
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk * 100 if pk > 0 else 0
        if dd > old_mdd:
            old_mdd = dd

    # Compare
    mdd_mtm = stats['mdd_mtm']
    mdd_real = cstats['mdd']
    pnl = cstats['pnl']
    pm_mtm = round(pnl / mdd_mtm, 2) if mdd_mtm > 0 else 0

    print(f"\n{'='*60}")
    print(f"  MDD COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'-'*42}")
    print(f"  {'OLD (entry_bar sorted)':<30} {old_mdd:>11.2f}%")
    print(f"  {'NEW (exit_bar sorted)':<30} {mdd_real:>11.2f}%")
    print(f"  {'MARK-TO-MARKET (bar-level)':<30} {mdd_mtm:>11.2f}%")
    print(f"  {'-'*42}")
    print(f"  {'PnL':<30} {pnl:>11.2f}%")
    print(f"  {'PnL/MDD (old)':<30} {round(pnl/old_mdd, 2) if old_mdd > 0 else 0:>12.2f}")
    print(f"  {'PnL/MDD (exit sorted)':<30} {cstats['pnl_mdd']:>12.2f}")
    print(f"  {'PnL/MDD (MTM)':<30} {pm_mtm:>12.2f}")
    print(f"  {'-'*42}")
    ratio = mdd_mtm / old_mdd if old_mdd > 0 else float('inf')
    print(f"  {'MTM/Old ratio':<30} {ratio:>11.1f}x")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save results
    result = {
        'metadata': {
            'script': 'mdd_verification.py',
            'date': time.strftime('%Y-%m-%d %H:%M'),
            'data_bars': n_bars,
            'patterns': len(patterns),
            'trades': len(trades),
        },
        'mdd_comparison': {
            'old_entry_sorted': round(old_mdd, 2),
            'new_exit_sorted': round(mdd_real, 2),
            'mark_to_market': round(mdd_mtm, 2),
            'underestimation_factor': round(ratio, 1),
        },
        'stats': {
            'wr': cstats['wr'],
            'pnl': cstats['pnl'],
            'pnl_mdd_old': round(pnl / old_mdd, 2) if old_mdd > 0 else 0,
            'pnl_mdd_exit_sorted': cstats['pnl_mdd'],
            'pnl_mdd_mtm': pm_mtm,
            'max_corr_loss': stats['max_corr_loss'],
            'max_sim_positions': stats['max_sim_positions'],
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'mdd_verification.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
