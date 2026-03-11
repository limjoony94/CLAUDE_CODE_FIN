#!/usr/bin/env python3
"""
WR Gap Study: Live TP+SL WR (61.6%) vs Scanner IS WR (72.8%) — 11.2pp gap
Cross-validated multi-phase investigation.

Hypotheses:
  H1: Slippage/execution — Live exits at worse prices than ideal
  H2: Temporal mismatch — Scanner 252d vs Live 16d (different regime)
  H3: Pattern mix — Different patterns dominate live vs scanner
  H4: ATR regime — Live period ATR differs from scanner average
  H5: Sample size — 146 trades may be statistically compatible with 72.8%

Standard Research Protocol: entry next-bar open, intrabar exit, fee 0.10%,
  slippage 0.02%, leverage 3x, compound, expanding window WF.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import sys
import time

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope, _check_exit_npos,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
)

def run_npos(signals, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope,
             start_bar, end_bar):
    """Run portfolio_npos and return (trades, stats) with calc_stats_compound."""
    trades, raw_stats = portfolio_npos(
        signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
        regime_mult=None,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
        clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
        timeout_bars=TIMEOUT_BARS,
    )
    stats = calc_stats_compound(trades)
    stats.update(raw_stats)
    return trades, stats

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
METRICS_FILE = "results/pattern_5m_metrics.json"

# ─── Load data ───
print("=" * 60)
print("WR GAP STUDY: Live 61.6% vs Scanner 72.8% (11.2pp)")
print("=" * 60)

df = load_and_classify(DATA_FILE)
opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
n_bars = len(df)
type_codes = df['candle_type'].values

atr_ratio = compute_atr_ratio(highs, lows, closes)
ema_slope = compute_ema_slope(closes)

with open(PATTERNS_FILE) as f:
    pat_data = json.load(f)
with open(METRICS_FILE) as f:
    metrics = json.load(f)

signal_index = build_signal_index(type_codes, n_bars)
neutral = pat_data.get('neutral_window', {})
ns = neutral.get('start_bar', 0)
ne = neutral.get('end_bar', n_bars)

# Build signal tuples for full IS
pattern_details = pat_data.get('pattern_details', {})
all_patterns = {}
for p in pat_data['patterns']['long']:
    all_patterns[p] = 'LONG'
for p in pat_data['patterns']['short']:
    all_patterns[p] = 'SHORT'

# Build signals for FULL range (ignore neutral window for live period comparison)
signal_tuples = []
signal_tuples_neutral = []
for triplet, direction in all_patterns.items():
    if triplet in signal_index:
        det = pattern_details.get(f"{triplet}_{direction}", {})
        tp = det.get('tp', 1.5)
        sl = det.get('sl', 3.0)
        for bar in signal_index[triplet]:
            sig = (bar, triplet, direction, tp, sl)
            if ns <= bar < ne:
                signal_tuples_neutral.append(sig)
            signal_tuples.append(sig)
signal_tuples.sort(key=lambda x: x[0])
signal_tuples_neutral.sort(key=lambda x: x[0])
print(f"Total signals (all bars): {len(signal_tuples)}")
print(f"Total signals (neutral): {len(signal_tuples_neutral)}")

# Live trades
live_history = metrics['trade_history']
live_tp_sl = [t for t in live_history if t.get('exit_reason') in ('TP', 'SL')]
print(f"Live TP+SL trades: {len(live_tp_sl)}")

# ─── Phase 1: Statistical Baseline ───
print("\n" + "=" * 60)
print("PHASE 1: STATISTICAL BASELINE — Is 11.2pp statistically significant?")
print("=" * 60)

live_wins = sum(1 for t in live_tp_sl if t['pnl_portfolio'] > 0)
live_n = len(live_tp_sl)
live_wr = live_wins / live_n * 100

scanner_wr = pat_data['npos_portfolio']['is_stats']['wr']

# Binomial test: is live WR compatible with scanner WR?
from scipy import stats as sp_stats
# H0: live WR = scanner WR
binom_result = sp_stats.binomtest(live_wins, live_n, scanner_wr / 100, alternative='less')
binom_p = binom_result.pvalue

# Bootstrap CI for live WR
rng = np.random.RandomState(42)
n_boot = 10000
boot_wrs = []
live_results = [1 if t['pnl_portfolio'] > 0 else 0 for t in live_tp_sl]
for _ in range(n_boot):
    sample = rng.choice(live_results, size=live_n, replace=True)
    boot_wrs.append(sample.mean() * 100)
ci_lo = np.percentile(boot_wrs, 2.5)
ci_hi = np.percentile(boot_wrs, 97.5)

print(f"Live WR:    {live_wr:.1f}% ({live_wins}/{live_n})")
print(f"Scanner WR: {scanner_wr:.1f}%")
print(f"Gap:        {scanner_wr - live_wr:.1f}pp")
print(f"Binomial p: {binom_p:.4f} (H0: live=scanner)")
print(f"Bootstrap 95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]")
print(f"Scanner WR in CI: {'YES' if ci_lo <= scanner_wr <= ci_hi else 'NO'}")
sig = "SIGNIFICANT" if binom_p < 0.05 else "NOT significant"
print(f"Result: Gap is {sig} at α=0.05")

phase1 = {
    'live_wr': live_wr, 'scanner_wr': scanner_wr,
    'gap_pp': scanner_wr - live_wr,
    'binom_p': binom_p, 'ci_95': [ci_lo, ci_hi],
    'significant': binom_p < 0.05
}

# ─── Phase 2: Temporal Match — Scanner on same 16-day period ───
print("\n" + "=" * 60)
print("PHASE 2: TEMPORAL MATCH — Scanner on live's exact date range")
print("=" * 60)

# Find bar range matching live period — data ends 2026-03-04, live starts 02-24
df['timestamp'] = pd.to_datetime(df['timestamp'])
live_start = pd.Timestamp('2026-02-24')
live_end = df['timestamp'].iloc[-1]  # Use data end
live_mask = (df['timestamp'] >= live_start) & (df['timestamp'] <= live_end)
live_bar_start = df.index[live_mask].min()
live_bar_end = df.index[live_mask].max()
print(f"Live period bars: {live_bar_start} - {live_bar_end} ({live_bar_end - live_bar_start} bars = {(live_bar_end - live_bar_start)/288:.1f} days)")
print(f"Data covers: {df['timestamp'].iloc[live_bar_start]} ~ {df['timestamp'].iloc[live_bar_end]}")
print(f"NOTE: Data ends 03-04, live continues to 03-11. Overlap ~8.8 days.")

# Run scanner on just this period (use ALL signals, not neutral-filtered)
live_signals = [(b, t, d, tp, sl) for b, t, d, tp, sl in signal_tuples
                if live_bar_start <= b < live_bar_end]
print(f"NOTE: Using full signal set (not neutral-filtered) for live period match")
print(f"Signals in live period: {len(live_signals)}")

trades_period, stats_period = run_npos(
    live_signals, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, live_bar_start, live_bar_end,
)

scanner_period_trades = stats_period.get('trades', 0)
scanner_period_wr = stats_period.get('wr', 0) if scanner_period_trades > 0 else 0
print(f"Scanner (same period): WR {scanner_period_wr:.1f}%, {scanner_period_trades} trades")
print(f"Live (same period):    WR {live_wr:.1f}%, {live_n} trades")
print(f"Period gap: {scanner_period_wr - live_wr:.1f}pp")

# Breakdown by direction
period_tp = sum(1 for t in trades_period if t.get('reason') == 'TP')
period_sl = sum(1 for t in trades_period if t.get('reason') == 'SL')
print(f"Scanner period exits: TP={period_tp}, SL={period_sl}")

# Scanner full IS direction breakdown
live_tp = sum(1 for t in live_tp_sl if t['exit_reason'] == 'TP')
live_sl = sum(1 for t in live_tp_sl if t['exit_reason'] == 'SL')
print(f"Live exits:           TP={live_tp}, SL={live_sl}")
if period_tp + period_sl > 0:
    print(f"TP rate: Scanner {period_tp/(period_tp+period_sl)*100:.1f}% vs Live {live_tp/(live_tp+live_sl)*100:.1f}%")
else:
    print(f"TP rate: Scanner N/A (0 trades) vs Live {live_tp/(live_tp+live_sl)*100:.1f}%")

phase2 = {
    'scanner_period_wr': scanner_period_wr,
    'scanner_period_trades': scanner_period_trades,
    'live_wr': live_wr, 'live_trades': live_n,
    'period_gap': scanner_period_wr - live_wr,
    'scanner_tp_sl': [period_tp, period_sl],
    'live_tp_sl': [live_tp, live_sl],
}

# ─── Phase 3: Slippage/Execution Analysis ───
print("\n" + "=" * 60)
print("PHASE 3: SLIPPAGE & EXECUTION ANALYSIS")
print("=" * 60)

# Analyze entry/exit price deviations
# For each live TP/SL trade, compute ideal exit price and compare
tp_slippages = []
sl_slippages = []
entry_slippages = []

for t in live_tp_sl:
    entry = t['entry_price']
    exit_p = t['exit_price']
    tp_p = t['tp_price']
    sl_p = t['sl_price']
    d = t['direction']
    reason = t['exit_reason']

    if reason == 'TP':
        # TP hit: ideal exit = tp_price, actual exit = exit_price
        if d == 'LONG':
            slip = (tp_p - exit_p) / entry * 100  # positive = favorable
        else:
            slip = (exit_p - tp_p) / entry * 100
        tp_slippages.append(slip)
    elif reason == 'SL':
        # SL hit: ideal exit = sl_price, actual exit = exit_price
        if d == 'LONG':
            slip = (exit_p - sl_p) / entry * 100  # positive = favorable (less loss)
        else:
            slip = (sl_p - exit_p) / entry * 100
        sl_slippages.append(slip)

print(f"TP Slippage ({len(tp_slippages)} trades):")
if tp_slippages:
    tp_arr = np.array(tp_slippages)
    print(f"  Mean: {tp_arr.mean():+.4f}%  Median: {np.median(tp_arr):+.4f}%")
    print(f"  Std:  {tp_arr.std():.4f}%  [Min: {tp_arr.min():+.4f}%, Max: {tp_arr.max():+.4f}%]")
    print(f"  Favorable (positive): {(tp_arr >= 0).sum()}/{len(tp_arr)} = {(tp_arr >= 0).mean()*100:.0f}%")

print(f"\nSL Slippage ({len(sl_slippages)} trades):")
if sl_slippages:
    sl_arr = np.array(sl_slippages)
    print(f"  Mean: {sl_arr.mean():+.4f}%  Median: {np.median(sl_arr):+.4f}%")
    print(f"  Std:  {sl_arr.std():.4f}%  [Min: {sl_arr.min():+.4f}%, Max: {sl_arr.max():+.4f}%]")
    print(f"  Favorable (positive): {(sl_arr >= 0).sum()}/{len(sl_arr)} = {(sl_arr >= 0).mean()*100:.0f}%")

# Analyze TP price vs actual exit — are TPs executing at TP price or better?
tp_exact = sum(1 for t in live_tp_sl if t['exit_reason'] == 'TP' and abs(t['exit_price'] - t['tp_price']) / t['entry_price'] < 0.001)
sl_exact = sum(1 for t in live_tp_sl if t['exit_reason'] == 'SL' and abs(t['exit_price'] - t['sl_price']) / t['entry_price'] < 0.001)
print(f"\nExact TP price hits: {tp_exact}/{live_tp} = {tp_exact/live_tp*100:.0f}%")
print(f"Exact SL price hits: {sl_exact}/{live_sl} = {sl_exact/live_sl*100:.0f}%")

# Compute effective edge lost to slippage
if tp_slippages and sl_slippages:
    total_slip = sum(tp_slippages) + sum(sl_slippages)
    avg_slip = total_slip / live_n
    print(f"\nTotal slippage impact: {total_slip:+.3f}% over {live_n} trades")
    print(f"Avg slippage per trade: {avg_slip:+.4f}%")

phase3 = {
    'tp_slippage_mean': float(np.mean(tp_slippages)) if tp_slippages else 0,
    'sl_slippage_mean': float(np.mean(sl_slippages)) if sl_slippages else 0,
    'tp_exact_rate': tp_exact / live_tp if live_tp else 0,
    'sl_exact_rate': sl_exact / live_sl if live_sl else 0,
    'total_slippage_pct': float(sum(tp_slippages) + sum(sl_slippages)),
    'avg_slippage_per_trade': float((sum(tp_slippages) + sum(sl_slippages)) / live_n) if live_n else 0,
}

# ─── Phase 4: Pattern-Level WR Comparison ───
print("\n" + "=" * 60)
print("PHASE 4: PATTERN-LEVEL WR COMPARISON")
print("=" * 60)

# Live WR by pattern
live_pat_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0})
for t in live_tp_sl:
    key = f"{t['pattern']}_{t['direction']}"
    live_pat_stats[key]['total'] += 1
    if t['pnl_portfolio'] > 0:
        live_pat_stats[key]['wins'] += 1
    live_pat_stats[key]['pnl'] += t.get('pnl_portfolio', 0)

# Scanner WR by pattern (from period simulation)
scanner_pat_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0})
for t in trades_period:
    key = f"{t['pattern']}_{t['direction']}"
    scanner_pat_stats[key]['total'] += 1
    if t.get('pnl_slot', 0) > 0:
        scanner_pat_stats[key]['wins'] += 1
    scanner_pat_stats[key]['pnl'] += t.get('pnl_slot', 0)

# Compare patterns that exist in both
common_patterns = set(live_pat_stats.keys()) & set(scanner_pat_stats.keys())
print(f"Patterns in live TP+SL: {len(live_pat_stats)}")
print(f"Patterns in scanner (period): {len(scanner_pat_stats)}")
print(f"Common patterns: {len(common_patterns)}")

pattern_gaps = []
print(f"\n{'Pattern':<25} {'Live WR':>8} {'Scan WR':>8} {'Gap':>6} {'L_n':>4} {'S_n':>4}")
for pat in sorted(common_patterns):
    l = live_pat_stats[pat]
    s = scanner_pat_stats[pat]
    if l['total'] >= 2 and s['total'] >= 2:
        l_wr = l['wins'] / l['total'] * 100
        s_wr = s['wins'] / s['total'] * 100
        gap = s_wr - l_wr
        pattern_gaps.append({'pattern': pat, 'live_wr': l_wr, 'scanner_wr': s_wr, 'gap': gap,
                             'live_n': l['total'], 'scanner_n': s['total']})
        print(f"  {pat:<25} {l_wr:6.1f}% {s_wr:6.1f}% {gap:+5.1f} {l['total']:4d} {s['total']:4d}")

if pattern_gaps:
    avg_gap = np.mean([p['gap'] for p in pattern_gaps])
    weighted_gap = sum(p['gap'] * p['live_n'] for p in pattern_gaps) / sum(p['live_n'] for p in pattern_gaps)
    print(f"\nAvg pattern gap:      {avg_gap:+.1f}pp")
    print(f"Weighted pattern gap: {weighted_gap:+.1f}pp")

# Patterns only in live (not in scanner period) — might indicate signal mismatch
live_only = set(live_pat_stats.keys()) - set(scanner_pat_stats.keys())
if live_only:
    print(f"\nPatterns in live but not scanner period: {len(live_only)}")
    for pat in sorted(live_only):
        s = live_pat_stats[pat]
        wr = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  {pat:<25} WR {wr:.0f}% ({s['total']}t)")

phase4 = {
    'common_patterns': len(common_patterns),
    'pattern_gaps': pattern_gaps,
    'avg_gap': float(avg_gap) if pattern_gaps else 0,
    'weighted_gap': float(weighted_gap) if pattern_gaps else 0,
    'live_only_patterns': len(live_only),
}

# ─── Phase 5: ATR & Volatility Regime Analysis ───
print("\n" + "=" * 60)
print("PHASE 5: ATR & VOLATILITY REGIME DURING LIVE PERIOD")
print("=" * 60)

# Compare ATR during live period vs full dataset
full_atr = atr_ratio[~np.isnan(atr_ratio)]
period_atr = atr_ratio[live_bar_start:live_bar_end+1]
period_atr = period_atr[~np.isnan(period_atr)]

print(f"Full dataset ATR ratio: mean={full_atr.mean():.3f}, std={full_atr.std():.3f}")
print(f"Live period ATR ratio:  mean={period_atr.mean():.3f}, std={period_atr.std():.3f}")
print(f"ATR ratio delta: {(period_atr.mean() - full_atr.mean()):+.3f}")

# ATR distribution: % of time in low/mid/high ATR
for label, lo, hi in [('Low (<0.7)', 0, 0.7), ('Mid (0.7-1.3)', 0.7, 1.3), ('High (>1.3)', 1.3, 99)]:
    full_pct = ((full_atr >= lo) & (full_atr < hi)).mean() * 100
    period_pct = ((period_atr >= lo) & (period_atr < hi)).mean() * 100
    print(f"  {label:<15}: Full {full_pct:.1f}%  Live {period_pct:.1f}%  Δ{period_pct-full_pct:+.1f}pp")

# Effect on TP/SL: higher ATR → wider TP/SL → harder to hit TP
# Scanner uses ATR clamp [0.5, 1.5], so effective ATR impact is bounded
# But skew matters: if ATR > 1.0 more often during live, SLs hit more

# Check per-trade ATR at signal time
live_atr_at_trade = []
for t in live_tp_sl:
    # Find the entry bar from hold_minutes
    exit_ts = pd.Timestamp(t['timestamp'])
    hold_min = t.get('hold_minutes', 0)
    entry_ts = exit_ts - pd.Timedelta(minutes=hold_min)
    entry_bars = df.index[df['timestamp'] >= entry_ts]
    if len(entry_bars) > 0:
        ebar = entry_bars[0]
        if ebar < len(atr_ratio) and not np.isnan(atr_ratio[ebar]):
            atr_val = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[ebar]))
            live_atr_at_trade.append((atr_val, t['exit_reason'], t['pnl_portfolio']))

if live_atr_at_trade:
    atr_vals = [x[0] for x in live_atr_at_trade]
    tp_atr = [x[0] for x in live_atr_at_trade if x[1] == 'TP']
    sl_atr = [x[0] for x in live_atr_at_trade if x[1] == 'SL']
    print(f"\nATR at trade entry:")
    print(f"  Overall: mean={np.mean(atr_vals):.3f}")
    print(f"  TP exits: mean={np.mean(tp_atr):.3f}" if tp_atr else "  TP exits: N/A")
    print(f"  SL exits: mean={np.mean(sl_atr):.3f}" if sl_atr else "  SL exits: N/A")
    if tp_atr and sl_atr:
        print(f"  SL-TP ATR diff: {np.mean(sl_atr) - np.mean(tp_atr):+.3f}")
        print(f"  (positive = SL happens at higher ATR = wider SL, harder to avoid)")

phase5 = {
    'full_atr_mean': float(full_atr.mean()),
    'period_atr_mean': float(period_atr.mean()),
    'atr_delta': float(period_atr.mean() - full_atr.mean()),
}

# ─── Phase 6: Cross-Validation — Sliding Window WR ───
print("\n" + "=" * 60)
print("PHASE 6: CROSS-VALIDATION — Scanner WR stability over time")
print("=" * 60)

# Split full IS into 16-day windows and compute WR in each
window_bars = int(16 * 288)  # 16 days
window_results = []

for w_start in range(ns, ne - window_bars, window_bars // 2):  # 50% overlap
    w_end = w_start + window_bars
    w_signals = [(b, t, d, tp, sl) for b, t, d, tp, sl in signal_tuples if w_start <= b < w_end]
    if len(w_signals) < 20:
        continue
    w_trades, w_stats = run_npos(
        w_signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, w_start, w_end,
    )
    if w_stats.get('trades', 0) >= 10:
        window_results.append({
            'start_bar': w_start,
            'end_bar': w_end,
            'trades': w_stats['trades'],
            'wr': w_stats['wr'],
            'pnl': w_stats['pnl'],
        })

wrs = [w['wr'] for w in window_results]
print(f"16-day windows (50% overlap): {len(window_results)}")
print(f"WR distribution: mean={np.mean(wrs):.1f}%, std={np.std(wrs):.1f}%, "
      f"min={np.min(wrs):.1f}%, max={np.max(wrs):.1f}%")
print(f"WR range: [{np.min(wrs):.1f}%, {np.max(wrs):.1f}%]")
print(f"Live WR (61.6%) percentile: {(np.array(wrs) <= live_wr).mean()*100:.0f}th")

# How many windows have WR <= live WR?
below_live = sum(1 for w in wrs if w <= live_wr)
print(f"Windows with WR ≤ {live_wr:.1f}%: {below_live}/{len(window_results)} ({below_live/len(window_results)*100:.0f}%)")

# Histogram
bins = [50, 55, 60, 65, 70, 75, 80, 85, 90]
print(f"\nWR Histogram:")
for i in range(len(bins) - 1):
    count = sum(1 for w in wrs if bins[i] <= w < bins[i+1])
    bar = '#' * count
    marker = ' ←LIVE' if bins[i] <= live_wr < bins[i+1] else ''
    print(f"  {bins[i]:2d}-{bins[i+1]:2d}%: {count:3d} {bar}{marker}")

phase6 = {
    'n_windows': len(window_results),
    'wr_mean': float(np.mean(wrs)),
    'wr_std': float(np.std(wrs)),
    'wr_min': float(np.min(wrs)),
    'wr_max': float(np.max(wrs)),
    'live_wr_percentile': float((np.array(wrs) <= live_wr).mean() * 100),
    'windows_below_live': below_live,
}

# ─── Phase 7: Scanner Simulation Fidelity Check ───
print("\n" + "=" * 60)
print("PHASE 7: SCANNER vs LIVE — Same-bar exit resolution check")
print("=" * 60)

# The scanner uses distance-based resolution when both TP and SL are hit on same bar
# Live uses exchange-side TP/SL orders → exchange decides which fills first
# This could cause systematic bias

# Count same-bar TP+SL hits in scanner (period simulation)
both_hit_count = 0
both_hit_tp = 0
both_hit_sl = 0
for t in trades_period:
    # We can't directly check from trades output, but we can approximate:
    # If hold_bars is very short and the pnl is very close to TP or SL, it may be same-bar
    pass

# Instead, check live: how many trades have exit_price != tp_price and != sl_price?
tp_not_exact = 0
sl_not_exact = 0
for t in live_tp_sl:
    if t['exit_reason'] == 'TP':
        if abs(t['exit_price'] - t['tp_price']) / t['entry_price'] > 0.002:
            tp_not_exact += 1
    elif t['exit_reason'] == 'SL':
        if abs(t['exit_price'] - t['sl_price']) / t['entry_price'] > 0.002:
            sl_not_exact += 1

print(f"TP exits with price deviation > 0.2%: {tp_not_exact}/{live_tp} ({tp_not_exact/live_tp*100:.0f}%)")
print(f"SL exits with price deviation > 0.2%: {sl_not_exact}/{live_sl} ({sl_not_exact/live_sl*100:.0f}%)")

# Cascade SL effect on live WR
# Cascade tightens SL → more SL hits → lower WR
# Check if live has cascaded SL trades
cascade_candidates = []
for i, t in enumerate(live_tp_sl):
    if t['exit_reason'] == 'SL' and i > 0:
        # Check if previous trade was also SL in same direction
        prev = live_tp_sl[i-1]
        if prev['exit_reason'] == 'SL' and prev['direction'] == t['direction']:
            cascade_candidates.append(t)

print(f"\nPossible cascade SL sequences: {len(cascade_candidates)} trades")
if cascade_candidates:
    cascade_wr_effect = len(cascade_candidates) / live_n * 100
    print(f"Cascade SL impact on WR: ~{cascade_wr_effect:.1f}pp (all are losses)")

phase7 = {
    'tp_price_deviation_rate': tp_not_exact / live_tp if live_tp else 0,
    'sl_price_deviation_rate': sl_not_exact / live_sl if live_sl else 0,
    'cascade_sl_sequences': len(cascade_candidates),
}

# ─── Phase 8: Direction Breakdown Cross-Validation ───
print("\n" + "=" * 60)
print("PHASE 8: DIRECTION BREAKDOWN — LONG vs SHORT WR gap")
print("=" * 60)

for direction in ['LONG', 'SHORT']:
    live_d = [t for t in live_tp_sl if t['direction'] == direction]
    live_d_wins = sum(1 for t in live_d if t['pnl_portfolio'] > 0)
    live_d_wr = live_d_wins / len(live_d) * 100 if live_d else 0

    scan_d = [t for t in trades_period if t['direction'] == direction]
    scan_d_wins = sum(1 for t in scan_d if t.get('pnl_slot', 0) > 0)
    scan_d_wr = scan_d_wins / len(scan_d) * 100 if scan_d else 0

    print(f"{direction}:")
    print(f"  Live:    WR {live_d_wr:.1f}% ({len(live_d)}t)")
    print(f"  Scanner: WR {scan_d_wr:.1f}% ({len(scan_d)}t)")
    print(f"  Gap:     {scan_d_wr - live_d_wr:+.1f}pp")

phase8 = {}
for direction in ['LONG', 'SHORT']:
    live_d = [t for t in live_tp_sl if t['direction'] == direction]
    scan_d = [t for t in trades_period if t['direction'] == direction]
    live_d_wr = sum(1 for t in live_d if t['pnl_portfolio'] > 0) / len(live_d) * 100 if live_d else 0
    scan_d_wr = sum(1 for t in scan_d if t.get('pnl_slot', 0) > 0) / len(scan_d) * 100 if scan_d else 0
    phase8[direction] = {'live_wr': live_d_wr, 'scanner_wr': scan_d_wr,
                         'gap': scan_d_wr - live_d_wr,
                         'live_n': len(live_d), 'scanner_n': len(scan_d)}

# ─── Synthesis ───
print("\n" + "=" * 60)
print("SYNTHESIS")
print("=" * 60)

print(f"""
Overall gap: {phase1['gap_pp']:.1f}pp (Live {live_wr:.1f}% vs Scanner IS {scanner_wr:.1f}%)
Temporal match gap: {phase2['period_gap']:.1f}pp (Scanner on same 16d period)
Statistical significance: {'YES' if phase1['significant'] else 'NO'} (p={binom_p:.4f})
Live WR in scanner distribution: {phase6['live_wr_percentile']:.0f}th percentile

Gap Decomposition:
  Slippage contribution:  ~{abs(phase3['avg_slippage_per_trade']):.3f}% per trade
  ATR regime difference:  {phase5['atr_delta']:+.3f} (live vs full)
  Scanner WR variance:    σ={phase6['wr_std']:.1f}pp across 16d windows

Verdict: {'NORMAL VARIANCE' if phase6['live_wr_percentile'] >= 10 else 'INVESTIGATE FURTHER'}
  (Live WR is at {phase6['live_wr_percentile']:.0f}th percentile of scanner 16d windows)
""")

runtime = time.time() - start_time
results = {
    'metadata': {
        'script': 'wr_gap_study.py',
        'date': pd.Timestamp.now().isoformat(),
        'runtime_sec': round(runtime, 1),
        'live_period': '2026-02-24 ~ 2026-03-11',
        'live_tp_sl_trades': live_n,
    },
    'phase1_statistics': phase1,
    'phase2_temporal': phase2,
    'phase3_slippage': phase3,
    'phase4_patterns': phase4,
    'phase5_atr': phase5,
    'phase6_cross_validation': phase6,
    'phase7_fidelity': phase7,
    'phase8_direction': phase8,
}

out_path = Path("results/wr_gap_study.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path} ({runtime:.1f}s)")
