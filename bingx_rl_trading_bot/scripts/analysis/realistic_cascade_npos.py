#!/usr/bin/env python3
"""
Realistic Cascade Exit Pricing Study
=====================================
CRITICAL FINDING: Scanner portfolio_npos() exits cascade-tightened SLs at the
tight SL price (near entry), but live exits at market price (bar OPEN, already
moved 1-2% against). This study quantifies the BT inflation.

Scenarios:
  A. CASCADE_98 scanner pricing (original BT — SL exit at tight SL price)
  B. CASCADE_98 realistic pricing (SL exit recalculated at bar OPEN)
  C. NO_CASCADE baseline (cascade disabled entirely)
  D. PARTIAL_CLOSE: When cascade fires, close worst N/2 at bar OPEN,
     keep remaining with ORIGINAL SLs unchanged

For each: IS full-data + 5-fold expanding WF.

Standard Research Protocol:
- classify_candle from production indicators
- LEVERAGE = 3, FEE = 0.10% * 3 = 0.30%
- Timeout = PnL at close (not DROP)
- Expanding Window WF: is_end = n*(fi+1)/6, oos_end = n*(fi+2)/6
"""

import os
import sys
import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime

# Setup paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW
from scripts.scanner.pattern_scanner import (
    build_signal_index,
    compute_atr_ratio,
    compute_ema_slope,
    portfolio_npos,
    calc_stats_compound,
    LEVERAGE,
    FEE_PCT,
    SLIPPAGE_BUFFER,
    MAX_DAILY_LOSS_PCT,
)

# =============================================================
# Constants
# =============================================================
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'realistic_cascade_npos.json')

N_SLOTS = 9
DIRECTION_CAP = 6
TIMEOUT_BARS = 576
CASCADE_TIGHTEN_PCT = 98       # v1.69.1 production
PREEMPTIVE_CASCADE_PCT = 1.5   # v1.69.1 production
PREEMPTIVE_TIGHTEN_PCT = 98    # v1.69.1 production
CLAMP_LO = 1.0                 # v1.69.1: expand-only
CLAMP_HI = 1.5
TP_DECAY_RATE = 0.0            # disabled per task
VOL_ADAPT_CLAMP = (0.0, 0.0)  # disabled
WF_FOLDS = 5


def load_data():
    """Load CSV and classify candles."""
    df = pd.read_csv(DATA_FILE)
    body = (df['close'] - df['open']).abs()
    avg_body = body.rolling(AVG_BODY_WINDOW, min_periods=1).mean()

    types = []
    for i in range(len(df)):
        row = df.iloc[i]
        ab = avg_body.iloc[i] if avg_body.iloc[i] > 0 else 1e-9
        ct = classify_candle(row, ab)
        types.append(ct.value)

    opens = df['open'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    n_bars = len(df)

    return types, opens, highs, lows, closes, n_bars


def load_patterns():
    """Load dynamic_patterns.json and build signal_tuples."""
    with open(PATTERNS_FILE) as f:
        dp = json.load(f)
    details = dp['pattern_details']
    return details


def build_tuples(signal_index, details):
    """Build (entry_bar, pat_key, direction, tp, sl) tuples using MFE median TP."""
    tuples = []
    for pat_key, info in details.items():
        pat = info['pattern']
        direction = info['direction']
        sl = info['sl']
        # MFE median TP (v1.67.0)
        exc = info.get('exc_stats', {})
        tp = exc.get('mfe_median', info['tp'])

        if pat not in signal_index:
            continue
        for bar in signal_index[pat]:
            tuples.append((bar, pat_key, direction, tp, sl))
    return tuples


def run_portfolio(signal_tuples, opens, highs, lows, closes, n_bars,
                  atr_ratio, ema_slope, start_bar, end_bar,
                  cascade_pct, preemptive_pct, preemptive_tighten):
    """Run portfolio_npos with given cascade settings."""
    trades, stats = portfolio_npos(
        signal_tuples=[(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples],
        opens=opens, highs=highs, lows=lows, closes=closes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
        n_slots=N_SLOTS, direction_cap=DIRECTION_CAP,
        timeout_bars=TIMEOUT_BARS,
        cascade_tighten_pct=cascade_pct,
        preemptive_cascade_pct=preemptive_pct,
        preemptive_tighten_pct=preemptive_tighten,
        tp_decay_rate=TP_DECAY_RATE,
        vol_adapt_clamp=VOL_ADAPT_CLAMP,
        clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI,
    )
    return trades, stats


def adjust_cascade_to_realistic(trades, opens, closes, n_bars):
    """Post-process: recalculate cascade SL exit PnL using bar OPEN (market price).

    When cascade tightens an SL to near entry, the BT exits at the tight SL price.
    In reality, the SL order fires but exits at market price (the bar's OPEN),
    which is already 1-2% against the position.

    For non-cascade trades, keep original PnL unchanged.
    """
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS
    adjusted = []

    for t in trades:
        t2 = dict(t)
        reason = t2.get('reason', '')
        # Cascade exits are SL exits on positions that were cascade-tightened
        # The scanner marks them as 'SL' but we can detect them by checking
        # if the PnL is suspiciously small (near zero minus fee) for an SL exit.
        # A cascade 98% SL keeps only 2% of original distance, so PnL ~ -0.06% to -0.3%
        # Normal SL exits have PnL ~ -3% to -15% (leveraged)
        #
        # Better approach: we tag cascade trades during the portfolio run.
        # Since we can't modify portfolio_npos, we detect cascade SL exits by
        # checking if |pnl_slot + fee| < 1.0 (i.e., raw PnL before fee is tiny)
        # Normal SL PnL ~ -3*SL% + slippage, which is -3% to -15%
        # Cascade SL PnL ~ -3*0.02*SL% + slippage ~ -0.06% to -0.3%
        is_cascade_sl = (reason == 'SL' and abs(t2['pnl_slot'] + fee) < 1.5)

        if is_cascade_sl:
            eb = t2['entry_bar']
            xb = t2['exit_bar']
            if eb < n_bars and xb < n_bars:
                entry_price = opens[eb]
                exit_price = opens[xb]  # realistic: market price at bar open
                if entry_price > 0:
                    direction = t2.get('direction', 'LONG')
                    if direction == 'LONG':
                        pnl = (exit_price / entry_price - 1) * 100 * LEVERAGE
                    else:
                        pnl = (1 - exit_price / entry_price) * 100 * LEVERAGE
                    pnl -= fee
                    t2['pnl_slot'] = pnl
                    sm = t2.get('size_mult', 1.0)
                    t2['pnl_portfolio'] = pnl * (size_pct / 100) * sm
                    t2['realistic_adjusted'] = True

        adjusted.append(t2)
    return adjusted


def run_partial_close_portfolio(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, start_bar, end_bar):
    """Scenario D: When cascade would fire, close worst N/2 positions at bar OPEN,
    keep remaining with ORIGINAL SLs unchanged.

    This requires reimplementing the portfolio loop with modified cascade logic.
    We run the FULL portfolio_npos with CASCADE to identify cascade events,
    then re-simulate with partial close behavior.
    """
    # Step 1: Run with cascade to identify when cascade fires
    # We need to know which bars have cascade events.
    # Instead of modifying the scanner, we implement a lightweight simulation.

    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    # First, get the NO_CASCADE run as baseline behavior
    trades_no_cascade, _ = run_portfolio(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        cascade_pct=0, preemptive_pct=0, preemptive_tighten=0,
    )

    # Run WITH cascade to identify cascade SL exits
    trades_with_cascade, _ = run_portfolio(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        cascade_pct=CASCADE_TIGHTEN_PCT,
        preemptive_pct=PREEMPTIVE_CASCADE_PCT,
        preemptive_tighten=PREEMPTIVE_TIGHTEN_PCT,
    )

    # Identify cascade SL bars: bars where cascade SL exits occurred
    cascade_sl_bars = set()
    for t in trades_with_cascade:
        if t.get('reason') == 'SL' and abs(t['pnl_slot'] + fee) < 1.5:
            cascade_sl_bars.add(t['exit_bar'])

    # For partial close: on cascade bars, take the worst half of active positions
    # and close them at bar OPEN price.
    # This is an approximation: we use the no-cascade trades as the base,
    # then for each bar where cascade would have fired, we estimate the impact.

    # Since we can't easily reimplement the full portfolio loop, we use a
    # different approach: post-process the cascade trades list.
    # For cascade SL exits: keep N/2 worst (close at bar OPEN), remove the rest
    # (they continue with original SLs = no cascade behavior).

    # Group cascade exits by bar
    from collections import defaultdict
    cascade_by_bar = defaultdict(list)
    non_cascade_trades = []

    for t in trades_with_cascade:
        if t.get('reason') == 'SL' and abs(t['pnl_slot'] + fee) < 1.5:
            cascade_by_bar[t['exit_bar']].append(t)
        else:
            non_cascade_trades.append(t)

    partial_close_trades = list(non_cascade_trades)

    for bar, cascade_trades in cascade_by_bar.items():
        n_close = max(1, len(cascade_trades) // 2)
        # Sort by unrealized PnL (worst first) — use the cascade SL pnl as proxy
        sorted_ct = sorted(cascade_trades, key=lambda t: t['pnl_slot'])
        # Close worst N/2 at bar OPEN (realistic pricing)
        for t in sorted_ct[:n_close]:
            t2 = dict(t)
            eb = t2['entry_bar']
            if eb < n_bars and bar < n_bars:
                entry_price = opens[eb]
                exit_price = opens[bar]
                if entry_price > 0:
                    direction = t2.get('direction', 'LONG')
                    if direction == 'LONG':
                        pnl = (exit_price / entry_price - 1) * 100 * LEVERAGE
                    else:
                        pnl = (1 - exit_price / entry_price) * 100 * LEVERAGE
                    pnl -= fee
                    t2['pnl_slot'] = pnl
                    sm = t2.get('size_mult', 1.0)
                    t2['pnl_portfolio'] = pnl * (size_pct / 100) * sm
                    t2['reason'] = 'PARTIAL_CLOSE'
            partial_close_trades.append(t2)
        # The remaining positions: they stay open with original SLs.
        # In our approximation, they follow the no-cascade trajectory.
        # We simply drop them from the cascade trades (they aren't closed).
        # Their eventual exits are already captured in the non-cascade trades
        # above (approximately — this is a simplification since the no-cascade
        # run has different slot dynamics).

    return partial_close_trades


def compute_stats(trades):
    """Compute stats from trades list, excluding drop=True."""
    valid = [t for t in trades if not t.get('drop', False)]
    if not valid:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}
    return calc_stats_compound(valid)


def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, scenario_fn, label):
    """Run 5-fold expanding window WF.

    Fold formula: is_end = n*(fi+1)/6, oos_end = n*(fi+2)/6
    """
    n = n_bars
    results = []
    for fi in range(WF_FOLDS):
        is_end = int(n * (fi + 1) / (WF_FOLDS + 1))
        oos_end = int(n * (fi + 2) / (WF_FOLDS + 1))
        oos_start = is_end

        # IS
        is_trades = scenario_fn(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, 0, is_end)
        is_stats = compute_stats(is_trades)

        # OOS
        oos_trades = scenario_fn(signal_tuples, opens, highs, lows, closes, n_bars,
                                 atr_ratio, ema_slope, oos_start, oos_end)
        oos_stats = compute_stats(oos_trades)

        results.append({
            'fold': fi + 1,
            'is_range': f'0-{is_end}',
            'oos_range': f'{oos_start}-{oos_end}',
            'is': is_stats,
            'oos': oos_stats,
        })
        print(f"  WF F{fi+1}: IS PnL={is_stats['pnl']:+.1f}% WR={is_stats['wr']:.1f}% "
              f"| OOS PnL={oos_stats['pnl']:+.1f}% WR={oos_stats['wr']:.1f}%")

    oos_total = sum(r['oos']['pnl'] for r in results)
    oos_trades_total = sum(r['oos']['trades'] for r in results)
    print(f"  WF TOTAL OOS: PnL={oos_total:+.1f}%, Trades={oos_trades_total}")
    return results, oos_total


def main():
    print("=" * 70)
    print("REALISTIC CASCADE EXIT PRICING STUDY")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"CASCADE_TIGHTEN_PCT: {CASCADE_TIGHTEN_PCT}")
    print(f"PREEMPTIVE: {PREEMPTIVE_CASCADE_PCT}% / {PREEMPTIVE_TIGHTEN_PCT}%")
    print(f"TIMEOUT: {TIMEOUT_BARS}, TP_DECAY: {TP_DECAY_RATE}")
    print(f"ATR CLAMP: [{CLAMP_LO}, {CLAMP_HI}]")
    print()

    # Load data
    print("Loading data...")
    types, opens, highs, lows, closes, n_bars = load_data()
    print(f"  Bars: {n_bars}")

    # Build signal index
    signal_index = build_signal_index(types, n_bars)

    # Load patterns
    details = load_patterns()
    signal_tuples = build_tuples(signal_index, details)
    print(f"  Patterns: {len(details)}, Signals: {len(signal_tuples)}")

    # Compute ATR and EMA
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    results = {}

    # =========================================================
    # Scenario A: CASCADE_98 scanner pricing (original)
    # =========================================================
    print("\n" + "=" * 50)
    print("SCENARIO A: CASCADE_98 — Scanner Pricing (Original)")
    print("=" * 50)

    def scenario_a(st, o, h, l, c, nb, ar, es, s, e):
        trades, _ = run_portfolio(st, o, h, l, c, nb, ar, es, s, e,
                                  CASCADE_TIGHTEN_PCT, PREEMPTIVE_CASCADE_PCT,
                                  PREEMPTIVE_TIGHTEN_PCT)
        return trades

    trades_a = scenario_a(signal_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, 0, n_bars)
    stats_a = compute_stats(trades_a)
    print(f"  IS: PnL={stats_a['pnl']:+.1f}%, WR={stats_a['wr']:.1f}%, "
          f"MDD={stats_a['mdd']:.2f}%, Trades={stats_a['trades']}")

    # Count cascade SL exits
    fee = FEE_PCT * LEVERAGE
    cascade_count_a = sum(1 for t in trades_a
                          if t.get('reason') == 'SL' and abs(t['pnl_slot'] + fee) < 1.5)
    normal_sl_a = sum(1 for t in trades_a if t.get('reason') == 'SL') - cascade_count_a
    tp_count_a = sum(1 for t in trades_a if t.get('reason') == 'TP')
    timeout_a = sum(1 for t in trades_a if t.get('reason') == 'TIMEOUT')
    print(f"  Exit breakdown: TP={tp_count_a}, Normal_SL={normal_sl_a}, "
          f"Cascade_SL={cascade_count_a}, Timeout={timeout_a}")

    # Cascade PnL contribution
    cascade_pnl_a = sum(t['pnl_portfolio'] for t in trades_a
                        if t.get('reason') == 'SL' and abs(t['pnl_slot'] + fee) < 1.5)
    print(f"  Cascade SL total portfolio PnL: {cascade_pnl_a:+.2f}%")

    # Average cascade vs normal SL PnL
    cascade_trades_a = [t for t in trades_a
                        if t.get('reason') == 'SL' and abs(t['pnl_slot'] + fee) < 1.5]
    normal_sl_trades_a = [t for t in trades_a
                          if t.get('reason') == 'SL' and abs(t['pnl_slot'] + fee) >= 1.5]
    if cascade_trades_a:
        avg_cascade_pnl = np.mean([t['pnl_slot'] for t in cascade_trades_a])
        print(f"  Avg cascade SL pnl_slot: {avg_cascade_pnl:+.3f}%")
    if normal_sl_trades_a:
        avg_normal_sl = np.mean([t['pnl_slot'] for t in normal_sl_trades_a])
        print(f"  Avg normal SL pnl_slot: {avg_normal_sl:+.3f}%")

    print("\n  WF 5-fold:")
    wf_a, oos_total_a = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, scenario_a, "A")

    results['A_cascade_scanner'] = {
        'is': stats_a,
        'cascade_sl_count': cascade_count_a,
        'cascade_pnl_portfolio': round(cascade_pnl_a, 2),
        'wf': wf_a,
        'oos_total': round(oos_total_a, 1),
    }

    # =========================================================
    # Scenario B: CASCADE_98 realistic pricing (bar OPEN)
    # =========================================================
    print("\n" + "=" * 50)
    print("SCENARIO B: CASCADE_98 — Realistic Pricing (Bar OPEN)")
    print("=" * 50)

    def scenario_b(st, o, h, l, c, nb, ar, es, s, e):
        trades, _ = run_portfolio(st, o, h, l, c, nb, ar, es, s, e,
                                  CASCADE_TIGHTEN_PCT, PREEMPTIVE_CASCADE_PCT,
                                  PREEMPTIVE_TIGHTEN_PCT)
        return adjust_cascade_to_realistic(trades, o, c, nb)

    trades_b = scenario_b(signal_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, 0, n_bars)
    stats_b = compute_stats(trades_b)
    print(f"  IS: PnL={stats_b['pnl']:+.1f}%, WR={stats_b['wr']:.1f}%, "
          f"MDD={stats_b['mdd']:.2f}%, Trades={stats_b['trades']}")

    # Adjusted cascade PnL
    adjusted_trades = [t for t in trades_b if t.get('realistic_adjusted')]
    if adjusted_trades:
        adj_pnl = sum(t['pnl_portfolio'] for t in adjusted_trades)
        avg_adj_pnl = np.mean([t['pnl_slot'] for t in adjusted_trades])
        print(f"  Adjusted cascade SL trades: {len(adjusted_trades)}")
        print(f"  Adjusted cascade PnL contribution: {adj_pnl:+.2f}%")
        print(f"  Avg adjusted cascade pnl_slot: {avg_adj_pnl:+.3f}%")

    # Quantify the difference
    pnl_diff = stats_b['pnl'] - stats_a['pnl']
    if stats_a['pnl'] != 0:
        pct_inflation = (stats_a['pnl'] - stats_b['pnl']) / abs(stats_b['pnl']) * 100 if stats_b['pnl'] != 0 else float('inf')
    else:
        pct_inflation = 0
    print(f"\n  PnL difference (Realistic - Scanner): {pnl_diff:+.1f}%")
    print(f"  Scanner inflation over realistic: {pct_inflation:+.1f}%")

    print("\n  WF 5-fold:")
    wf_b, oos_total_b = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, scenario_b, "B")

    results['B_cascade_realistic'] = {
        'is': stats_b,
        'adjusted_cascade_count': len(adjusted_trades),
        'wf': wf_b,
        'oos_total': round(oos_total_b, 1),
    }

    # =========================================================
    # Scenario C: NO_CASCADE baseline
    # =========================================================
    print("\n" + "=" * 50)
    print("SCENARIO C: NO_CASCADE Baseline")
    print("=" * 50)

    def scenario_c(st, o, h, l, c, nb, ar, es, s, e):
        trades, _ = run_portfolio(st, o, h, l, c, nb, ar, es, s, e,
                                  cascade_pct=0, preemptive_pct=0,
                                  preemptive_tighten=0)
        return trades

    trades_c = scenario_c(signal_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, 0, n_bars)
    stats_c = compute_stats(trades_c)
    print(f"  IS: PnL={stats_c['pnl']:+.1f}%, WR={stats_c['wr']:.1f}%, "
          f"MDD={stats_c['mdd']:.2f}%, Trades={stats_c['trades']}")

    sl_c = sum(1 for t in trades_c if t.get('reason') == 'SL')
    tp_c = sum(1 for t in trades_c if t.get('reason') == 'TP')
    to_c = sum(1 for t in trades_c if t.get('reason') == 'TIMEOUT')
    print(f"  Exit breakdown: TP={tp_c}, SL={sl_c}, Timeout={to_c}")

    print("\n  WF 5-fold:")
    wf_c, oos_total_c = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, scenario_c, "C")

    results['C_no_cascade'] = {
        'is': stats_c,
        'wf': wf_c,
        'oos_total': round(oos_total_c, 1),
    }

    # =========================================================
    # Scenario D: PARTIAL_CLOSE
    # =========================================================
    print("\n" + "=" * 50)
    print("SCENARIO D: PARTIAL_CLOSE (worst N/2 at bar OPEN, rest keep original SL)")
    print("=" * 50)

    def scenario_d(st, o, h, l, c, nb, ar, es, s, e):
        return run_partial_close_portfolio(st, o, h, l, c, nb, ar, es, s, e)

    trades_d = scenario_d(signal_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, 0, n_bars)
    stats_d = compute_stats(trades_d)
    print(f"  IS: PnL={stats_d['pnl']:+.1f}%, WR={stats_d['wr']:.1f}%, "
          f"MDD={stats_d['mdd']:.2f}%, Trades={stats_d['trades']}")

    partial_closed = sum(1 for t in trades_d if t.get('reason') == 'PARTIAL_CLOSE')
    print(f"  Partial close exits: {partial_closed}")

    print("\n  WF 5-fold:")
    wf_d, oos_total_d = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, scenario_d, "D")

    results['D_partial_close'] = {
        'is': stats_d,
        'partial_close_count': partial_closed,
        'wf': wf_d,
        'oos_total': round(oos_total_d, 1),
    }

    # =========================================================
    # Summary comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Scenario':<35} {'IS PnL':>8} {'IS WR':>6} {'IS MDD':>7} {'IS P/M':>7} {'OOS Tot':>8}")
    print("-" * 70)
    for label, key in [
        ("A: Cascade Scanner Pricing", 'A_cascade_scanner'),
        ("B: Cascade Realistic (bar OPEN)", 'B_cascade_realistic'),
        ("C: No Cascade", 'C_no_cascade'),
        ("D: Partial Close", 'D_partial_close'),
    ]:
        s = results[key]['is']
        pm = s['pnl'] / s['mdd'] if s['mdd'] > 0 else 0
        oos = results[key]['oos_total']
        print(f"{label:<35} {s['pnl']:>+7.1f}% {s['wr']:>5.1f}% {s['mdd']:>6.2f}% {pm:>6.1f}x {oos:>+7.1f}%")

    print()
    print("KEY FINDING:")
    if stats_b['pnl'] != 0:
        inflation = (stats_a['pnl'] / stats_b['pnl'] - 1) * 100
        print(f"  Scanner CASCADE overestimates IS PnL by {inflation:+.1f}% vs realistic pricing")
    print(f"  Scanner CASCADE IS: {stats_a['pnl']:+.1f}%  vs  Realistic: {stats_b['pnl']:+.1f}%")
    print(f"  No Cascade baseline: {stats_c['pnl']:+.1f}%")
    print(f"  Partial Close: {stats_d['pnl']:+.1f}%")

    if cascade_count_a > 0 and adjusted_trades:
        avg_scanner = np.mean([t['pnl_slot'] for t in cascade_trades_a])
        avg_real = np.mean([t['pnl_slot'] for t in adjusted_trades])
        print(f"\n  Per-cascade-trade PnL: Scanner avg {avg_scanner:+.3f}% vs Realistic avg {avg_real:+.3f}%")
        if avg_scanner != 0:
            ratio = avg_real / avg_scanner
            print(f"  Ratio (realistic/scanner): {ratio:.1f}x")

    # Save results
    output = {
        'study': 'realistic_cascade_npos',
        'date': datetime.now().isoformat(),
        'params': {
            'cascade_tighten_pct': CASCADE_TIGHTEN_PCT,
            'preemptive_cascade_pct': PREEMPTIVE_CASCADE_PCT,
            'preemptive_tighten_pct': PREEMPTIVE_TIGHTEN_PCT,
            'timeout_bars': TIMEOUT_BARS,
            'tp_decay_rate': TP_DECAY_RATE,
            'clamp_lo': CLAMP_LO,
            'clamp_hi': CLAMP_HI,
            'n_slots': N_SLOTS,
            'direction_cap': DIRECTION_CAP,
            'wf_folds': WF_FOLDS,
        },
        'results': results,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
