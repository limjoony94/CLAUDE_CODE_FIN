#!/usr/bin/env python3
"""
Intrabar Trail Trigger Impact Analysis
========================================
Quantify how bar-close trail checks (backtest) vs tick-by-tick trail
(live exchange TRAILING_STOP_MARKET) differ in outcomes.

Tests:
  1. Bar-Close Baseline (current backtest)
  2. Intrabar Trail using bar LOW/HIGH as worst case
  3. 5-Minute Resolution trail checks
  4. Exchange Trail Multiplier sweep
  5. Premature Exit Impact quantification

Data: 5m -> 15m resampled. Fee = 0.10% RT (1x additive).
"""
import os, sys, math, json
import pandas as pd
import numpy as np
from datetime import datetime

os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '.')

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

# ─── Constants ──────────────────────────────────────────────────────
FEE = 0.10  # round trip % (1x)
WARMUP = 26

# ─── Load & Resample Data ──────────────────────────────────────────
print("Loading data...")
df5 = pd.read_csv('data/btc_5m_270days_reclassified.csv')
df5['timestamp'] = pd.to_datetime(df5['timestamp'])

# 15m resampling via index // 3
df5['group'] = df5.index // 3
agg15 = df5.groupby('group').agg(
    open=('open', 'first'), high=('high', 'max'),
    low=('low', 'min'), close=('close', 'last'),
    ts=('timestamp', 'first')
).reset_index(drop=True)

o15 = agg15['open'].tolist()
h15 = agg15['high'].tolist()
l15 = agg15['low'].tolist()
c15 = agg15['close'].tolist()
n15 = len(c15)

# Keep 5m arrays for Test 3
o5 = df5['open'].tolist()
h5 = df5['high'].tolist()
l5 = df5['low'].tolist()
c5 = df5['close'].tolist()
n5 = len(c5)

# ─── Compute Indicators (15m) ──────────────────────────────────────
cfg = load_config()
strat = cfg['strategy']
sig = C1BreakoutSignal(strat)

atr14 = compute_atr(h15, l15, c15, strat.get('atr_period', 14))
ch_h, ch_l = compute_channel(h15, l15, strat.get('channel_period', 15))
sw_l, sw_h = compute_fractal_swings(h15, l15, strat.get('fractal_lookback', 10))

trail_K = strat.get('trail_K', 2.5)
trail_act = strat.get('trail_activation_pct', 0.05)
emergency_sl = strat.get('emergency_sl_pct', 3.0)
max_hold = strat.get('max_hold_bars', 192)
min_bars_between = strat.get('min_bars_between', 2)


# ─── Core Backtest Engine ──────────────────────────────────────────
def run_backtest(mode='bar_close', trail_K_override=None):
    """
    Unified backtest engine.

    mode:
      'bar_close' — Test 1: check trail at bar close only (current backtest)
      'intrabar'  — Test 2: check trail using bar worst price (LOW for LONG, HIGH for SHORT)
      '5m'        — Test 3: check trail on every 5m sub-bar within each 15m bar

    trail_K_override:
      If set, overrides trail_K for the exit check (Test 4 multiplier sweep)
    """
    tk = trail_K_override if trail_K_override is not None else trail_K
    trades = []
    pos = None
    last_exit_bar = -3  # 15m bar index

    for bar in range(WARMUP, n15 - 1):
        # ── Exit check ──
        if pos is not None:
            # Update best price FIRST — but NOT for 5m mode
            # (5m mode handles best_price granularly per sub-bar)
            if mode != '5m':
                if pos['d'] == 'LONG':
                    pos['bp'] = max(pos['bp'], h15[bar])
                else:
                    pos['bp'] = min(pos['bp'], l15[bar])
            pos['bh'] += 1

            exit_result = None

            if mode == '5m':
                # Check SL/trail on each 5m sub-bar
                exit_result = _check_exit_5m(pos, bar, tk)
            elif mode == 'intrabar':
                # Check SL first (same as production — uses bar high/low)
                exit_result = _check_exit_intrabar(pos, bar, tk)
            else:
                # bar_close: use production check_exit as-is but with overridden trail_K
                exit_result = _check_exit_bar_close(pos, bar, tk)

            if exit_result:
                xp = exit_result['exit_price']
                if pos['d'] == 'LONG':
                    raw = (xp / pos['ep'] - 1) * 100
                else:
                    raw = (1 - xp / pos['ep']) * 100
                trades.append({
                    'raw': raw, 'net': raw - FEE,
                    'reason': exit_result['reason'],
                    'bh': pos['bh'], 'd': pos['d'],
                    'entry_bar': pos['entry_bar'],
                    'exit_bar': bar,
                    'entry_price': pos['ep'],
                    'exit_price': xp,
                })
                pos = None
                last_exit_bar = bar
                continue

        # ── Entry check ──
        if pos is None and bar - last_exit_bar >= min_bars_between:
            e = sig.check_entry(
                o15[bar], h15[bar], l15[bar], c15[bar],
                ch_h[bar], ch_l[bar], atr14[bar],
                sw_l[bar], sw_h[bar]
            )
            if e and bar + 1 < n15:
                pos = {
                    'd': e['direction'],
                    'ep': o15[bar + 1],
                    'sl': e['sl_price'],
                    'bp': o15[bar + 1],
                    'bh': 0,
                    'entry_bar': bar + 1,
                }

    return trades


def _check_exit_bar_close(pos, bar, tk):
    """Bar-close trail check (Test 1 / Test 4 baseline)."""
    d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
    bh = pos['bh']

    # 1. Fractal SL
    if d == 'LONG' and l15[bar] <= sl:
        return {'reason': 'SL', 'exit_price': sl}
    elif d == 'SHORT' and h15[bar] >= sl:
        return {'reason': 'SL', 'exit_price': sl}

    # 2. Emergency SL
    if d == 'LONG':
        worst_pnl = (l15[bar] / ep - 1) * 100
    else:
        worst_pnl = (1 - h15[bar] / ep) * 100
    if worst_pnl <= -emergency_sl:
        if d == 'LONG':
            return {'reason': 'EMERGENCY', 'exit_price': ep * (1 - emergency_sl / 100)}
        else:
            return {'reason': 'EMERGENCY', 'exit_price': ep * (1 + emergency_sl / 100)}

    # 3. Timeout
    if bh >= max_hold:
        return {'reason': 'TIMEOUT', 'exit_price': c15[bar]}

    # 4. Trail TP — at bar CLOSE
    if d == 'LONG':
        best_pnl = (bp / ep - 1) * 100
        cur_pnl = (c15[bar] / ep - 1) * 100
    else:
        best_pnl = (1 - bp / ep) * 100
        cur_pnl = (1 - c15[bar] / ep) * 100

    atr_val = atr14[bar]
    if best_pnl > trail_act and not math.isnan(atr_val) and atr_val > 0:
        trail_dist_pct = tk * atr_val / c15[bar] * 100
        drawdown = best_pnl - cur_pnl
        if drawdown >= trail_dist_pct:
            realized = max(0, best_pnl - trail_dist_pct)
            if d == 'LONG':
                return {'reason': 'TRAIL_TP', 'exit_price': ep * (1 + realized / 100)}
            else:
                return {'reason': 'TRAIL_TP', 'exit_price': ep * (1 - realized / 100)}

    return None


def _check_exit_intrabar(pos, bar, tk):
    """Intrabar trail: uses bar LOW/HIGH as worst case (Test 2)."""
    d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
    bh = pos['bh']

    # 1. Fractal SL (same — already uses bar high/low)
    if d == 'LONG' and l15[bar] <= sl:
        return {'reason': 'SL', 'exit_price': sl}
    elif d == 'SHORT' and h15[bar] >= sl:
        return {'reason': 'SL', 'exit_price': sl}

    # 2. Emergency SL
    if d == 'LONG':
        worst_pnl = (l15[bar] / ep - 1) * 100
    else:
        worst_pnl = (1 - h15[bar] / ep) * 100
    if worst_pnl <= -emergency_sl:
        if d == 'LONG':
            return {'reason': 'EMERGENCY', 'exit_price': ep * (1 - emergency_sl / 100)}
        else:
            return {'reason': 'EMERGENCY', 'exit_price': ep * (1 + emergency_sl / 100)}

    # 3. Timeout
    if bh >= max_hold:
        return {'reason': 'TIMEOUT', 'exit_price': c15[bar]}

    # 4. Trail TP — using bar WORST price (intrabar simulation)
    # best_price was already updated with bar high/low before this call
    if d == 'LONG':
        best_pnl = (bp / ep - 1) * 100
        # Use bar LOW as worst case within bar
        worst_cur_pnl = (l15[bar] / ep - 1) * 100
    else:
        best_pnl = (1 - bp / ep) * 100
        # Use bar HIGH as worst case within bar
        worst_cur_pnl = (1 - h15[bar] / ep) * 100

    atr_val = atr14[bar]
    if best_pnl > trail_act and not math.isnan(atr_val) and atr_val > 0:
        # Use close for ATR normalization (consistent with production)
        trail_dist_pct = tk * atr_val / c15[bar] * 100
        drawdown_worst = best_pnl - worst_cur_pnl
        if drawdown_worst >= trail_dist_pct:
            # Exit at trail boundary, not at worst price
            realized = max(0, best_pnl - trail_dist_pct)
            if d == 'LONG':
                return {'reason': 'TRAIL_TP', 'exit_price': ep * (1 + realized / 100)}
            else:
                return {'reason': 'TRAIL_TP', 'exit_price': ep * (1 - realized / 100)}

    return None


def _check_exit_5m(pos, bar15, tk):
    """5-minute resolution trail check (Test 3).

    Checks SL and trail on each of the 3 sub-5m bars within the 15m bar.
    Indicators (ATR) stay at 15m resolution.
    bars_held counts 15m bars (for timeout).
    """
    d, ep, sl = pos['d'], pos['ep'], pos['sl']
    bh = pos['bh']

    # Map 15m bar index to 5m bar indices
    start_5m = bar15 * 3
    end_5m = min(start_5m + 3, n5)

    atr_val = atr14[bar15]

    for i5 in range(start_5m, end_5m):
        # Update best_price with this 5m bar's extreme FIRST
        if d == 'LONG':
            pos['bp'] = max(pos['bp'], h5[i5])
        else:
            pos['bp'] = min(pos['bp'], l5[i5])

        bp = pos['bp']

        # 1. Fractal SL
        if d == 'LONG' and l5[i5] <= sl:
            return {'reason': 'SL', 'exit_price': sl}
        elif d == 'SHORT' and h5[i5] >= sl:
            return {'reason': 'SL', 'exit_price': sl}

        # 2. Emergency SL
        if d == 'LONG':
            worst_pnl = (l5[i5] / ep - 1) * 100
        else:
            worst_pnl = (1 - h5[i5] / ep) * 100
        if worst_pnl <= -emergency_sl:
            if d == 'LONG':
                return {'reason': 'EMERGENCY', 'exit_price': ep * (1 - emergency_sl / 100)}
            else:
                return {'reason': 'EMERGENCY', 'exit_price': ep * (1 + emergency_sl / 100)}

        # 3. Timeout (still in 15m bars)
        if bh >= max_hold:
            return {'reason': 'TIMEOUT', 'exit_price': c5[i5]}

        # 4. Trail TP on 5m bar
        if d == 'LONG':
            best_pnl = (bp / ep - 1) * 100
            cur_pnl = (c5[i5] / ep - 1) * 100
        else:
            best_pnl = (1 - bp / ep) * 100
            cur_pnl = (1 - c5[i5] / ep) * 100

        if best_pnl > trail_act and not math.isnan(atr_val) and atr_val > 0:
            trail_dist_pct = tk * atr_val / c5[i5] * 100
            drawdown = best_pnl - cur_pnl
            if drawdown >= trail_dist_pct:
                realized = max(0, best_pnl - trail_dist_pct)
                if d == 'LONG':
                    return {'reason': 'TRAIL_TP', 'exit_price': ep * (1 + realized / 100)}
                else:
                    return {'reason': 'TRAIL_TP', 'exit_price': ep * (1 - realized / 100)}

    return None


# ─── Statistics Helper ──────────────────────────────────────────────
def compute_stats(trades, label):
    pnls = [t['net'] for t in trades]
    if not pnls:
        return {'label': label, 'trades': 0, 'pnl': 0, 'wr': 0, 'mdd': 0}
    wins = [p for p in pnls if p > 0]
    total = sum(pnls)
    eq = 0; pk = 0; mdd = 0
    for p in pnls:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    wr = len(wins) / len(pnls) * 100
    wa = np.mean(wins) if wins else 0
    losses = [p for p in pnls if p <= 0]
    la = np.mean(losses) if losses else -1
    rr = abs(wa / la) if la else 0
    nd = n15 * 15 / 1440

    reasons = {}
    for t in trades:
        r = t['reason']
        if r not in reasons:
            reasons[r] = 0
        reasons[r] += 1

    return {
        'label': label, 'trades': len(pnls), 'pnl': total,
        'wr': wr, 'mdd': mdd, 'rr': rr, 'daily': total / nd,
        'reasons': reasons,
    }


def print_stats(s):
    print(f"  PnL:    {s['pnl']:+.1f}%")
    print(f"  Trades: {s['trades']}")
    print(f"  WR:     {s['wr']:.1f}%")
    print(f"  MDD:    {s['mdd']:.1f}%")
    print(f"  R:R:    {s['rr']:.2f}")
    print(f"  Daily:  {s['daily']:+.4f}%")
    if s.get('reasons'):
        for r, cnt in sorted(s['reasons'].items()):
            print(f"    {r:>10}: {cnt}")


# ─── Run Tests ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  INTRABAR TRAIL TRIGGER IMPACT ANALYSIS")
print("=" * 70)

# Test 1: Baseline
print(f"\n{'=' * 70}")
print("  Test 1: Bar-Close Baseline (current backtest)")
print(f"{'=' * 70}")
trades1 = run_backtest(mode='bar_close')
s1 = compute_stats(trades1, 'bar_close')
print_stats(s1)

# Test 2: Intrabar Trail
print(f"\n{'=' * 70}")
print("  Test 2: Intrabar Trail (bar LOW/HIGH as worst case)")
print(f"{'=' * 70}")
trades2 = run_backtest(mode='intrabar')
s2 = compute_stats(trades2, 'intrabar')
print_stats(s2)
print(f"\n  Delta vs Baseline:")
print(f"    PnL:    {s2['pnl'] - s1['pnl']:+.1f}pp")
print(f"    Trades: {s2['trades'] - s1['trades']:+d}")
print(f"    WR:     {s2['wr'] - s1['wr']:+.1f}pp")

# Test 3: 5-Min Resolution
print(f"\n{'=' * 70}")
print("  Test 3: 5-Minute Resolution Trail Check")
print(f"{'=' * 70}")
trades3 = run_backtest(mode='5m')
s3 = compute_stats(trades3, '5m_resolution')
print_stats(s3)
print(f"\n  Delta vs Baseline:")
print(f"    PnL:    {s3['pnl'] - s1['pnl']:+.1f}pp")
print(f"    Trades: {s3['trades'] - s1['trades']:+d}")
print(f"    WR:     {s3['wr'] - s1['wr']:+.1f}pp")

# Test 4: Exchange Trail Multiplier
print(f"\n{'=' * 70}")
print("  Test 4: Exchange Trail Multiplier Sweep")
print(f"{'=' * 70}")
multipliers = [1.0, 1.3, 1.5, 1.7, 2.0]
mult_results = []
print(f"  {'Mult':>6} | {'Intrabar PnL':>14} | {'vs Baseline':>12} | {'Trades':>7} | {'WR':>6}")
print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}")

for m in multipliers:
    tk_override = trail_K * m
    t4 = run_backtest(mode='intrabar', trail_K_override=tk_override)
    s4 = compute_stats(t4, f'mult_{m}')
    delta = s4['pnl'] - s1['pnl']
    print(f"  {m:>6.1f} | {s4['pnl']:>+13.1f}% | {delta:>+11.1f}pp | {s4['trades']:>7} | {s4['wr']:>5.1f}%")
    mult_results.append({
        'multiplier': m,
        'trail_K_effective': tk_override,
        'pnl': s4['pnl'],
        'trades': s4['trades'],
        'wr': s4['wr'],
        'mdd': s4['mdd'],
        'delta_vs_baseline': delta,
    })

# Find best multiplier (closest to baseline PnL)
best_mult = min(mult_results, key=lambda x: abs(x['delta_vs_baseline']))
print(f"\n  Best match to baseline: mult={best_mult['multiplier']:.1f} "
      f"(delta={best_mult['delta_vs_baseline']:+.1f}pp)")

# Test 5: Premature Exit Analysis
print(f"\n{'=' * 70}")
print("  Test 5: Premature Exit Impact Analysis")
print(f"{'=' * 70}")

# Build trade maps by entry_bar for comparison
map1 = {t['entry_bar']: t for t in trades1}
map2 = {t['entry_bar']: t for t in trades2}

# Trades that exist in both but exit differently
same_entry_bars = set(map1.keys()) & set(map2.keys())
different_exits = []
for eb in sorted(same_entry_bars):
    t1 = map1[eb]
    t2 = map2[eb]
    if t2['exit_bar'] != t1['exit_bar'] or t2['reason'] != t1['reason']:
        different_exits.append({
            'entry_bar': eb,
            'bar_close_exit': t1['exit_bar'],
            'intrabar_exit': t2['exit_bar'],
            'bar_close_reason': t1['reason'],
            'intrabar_reason': t2['reason'],
            'bar_close_pnl': t1['net'],
            'intrabar_pnl': t2['net'],
            'pnl_diff': t2['net'] - t1['net'],
            'earlier_exit': t2['exit_bar'] < t1['exit_bar'],
        })

# Trades only in one set (cascading effects)
only_in_baseline = set(map1.keys()) - set(map2.keys())
only_in_intrabar = set(map2.keys()) - set(map1.keys())

# Premature exits (intrabar exited earlier)
premature = [d for d in different_exits if d['earlier_exit']]
delayed = [d for d in different_exits if not d['earlier_exit']]

print(f"  Shared entry bars:         {len(same_entry_bars)}")
print(f"  Different exit outcomes:   {len(different_exits)}")
print(f"    Premature (earlier):     {len(premature)}")
print(f"    Delayed (later):         {len(delayed)}")
print(f"  Only in baseline:          {len(only_in_baseline)}")
print(f"  Only in intrabar:          {len(only_in_intrabar)}")

if premature:
    avg_pnl_diff = np.mean([d['pnl_diff'] for d in premature])
    # How many premature exits lost money vs would-have-been-profitable
    would_profit = sum(1 for d in premature if d['bar_close_pnl'] > 0 and d['intrabar_pnl'] <= 0)
    reduced_profit = sum(1 for d in premature if d['bar_close_pnl'] > 0 and d['intrabar_pnl'] > 0 and d['intrabar_pnl'] < d['bar_close_pnl'])
    improved = sum(1 for d in premature if d['intrabar_pnl'] > d['bar_close_pnl'])
    total_pnl_lost = sum(d['pnl_diff'] for d in premature)

    print(f"\n  Premature Exit Details:")
    print(f"    Avg PnL diff per trade:  {avg_pnl_diff:+.4f}%")
    print(f"    Total PnL impact:        {total_pnl_lost:+.1f}pp")
    print(f"    Flipped profit->loss:    {would_profit}")
    print(f"    Reduced profit:          {reduced_profit}")
    print(f"    Actually improved:       {improved}")

    # Distribution of PnL differences
    diffs = [d['pnl_diff'] for d in premature]
    print(f"    PnL diff p25/p50/p75:    {np.percentile(diffs,25):+.3f} / {np.percentile(diffs,50):+.3f} / {np.percentile(diffs,75):+.3f}")

if different_exits:
    # Overall impact
    total_diff = sum(d['pnl_diff'] for d in different_exits)
    print(f"\n  Total PnL impact (all changed trades): {total_diff:+.1f}pp")

    # Reason transition table
    print(f"\n  Exit Reason Transitions (bar_close -> intrabar):")
    transitions = {}
    for d in different_exits:
        key = f"{d['bar_close_reason']} -> {d['intrabar_reason']}"
        if key not in transitions:
            transitions[key] = 0
        transitions[key] += 1
    for k, v in sorted(transitions.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")


# ─── Save Results ──────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  SUMMARY")
print(f"{'=' * 70}")
print(f"  {'Test':>25} | {'PnL':>10} | {'Trades':>7} | {'WR':>6} | {'MDD':>6}")
print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")
for s in [s1, s2, s3]:
    print(f"  {s['label']:>25} | {s['pnl']:>+9.1f}% | {s['trades']:>7} | {s['wr']:>5.1f}% | {s['mdd']:>5.1f}%")

results = {
    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'script': 'scripts/analysis/intrabar_trail_impact.py',
    'data': 'data/btc_5m_270days_reclassified.csv',
    'data_bars_5m': n5,
    'data_bars_15m': n15,
    'parameters': {
        'channel_period': strat.get('channel_period', 15),
        'trail_K': trail_K,
        'max_sl_atr': strat.get('max_sl_atr', 3.3),
        'atr_period': strat.get('atr_period', 14),
        'fee_pct': FEE,
    },
    'test1_bar_close': {
        'pnl': round(s1['pnl'], 1),
        'trades': s1['trades'],
        'wr': round(s1['wr'], 1),
        'mdd': round(s1['mdd'], 1),
        'reasons': s1.get('reasons', {}),
    },
    'test2_intrabar': {
        'pnl': round(s2['pnl'], 1),
        'trades': s2['trades'],
        'wr': round(s2['wr'], 1),
        'mdd': round(s2['mdd'], 1),
        'delta_pnl_pp': round(s2['pnl'] - s1['pnl'], 1),
        'delta_trades': s2['trades'] - s1['trades'],
        'reasons': s2.get('reasons', {}),
    },
    'test3_5m_resolution': {
        'pnl': round(s3['pnl'], 1),
        'trades': s3['trades'],
        'wr': round(s3['wr'], 1),
        'mdd': round(s3['mdd'], 1),
        'delta_pnl_pp': round(s3['pnl'] - s1['pnl'], 1),
        'delta_trades': s3['trades'] - s1['trades'],
        'reasons': s3.get('reasons', {}),
    },
    'test4_multiplier_sweep': mult_results,
    'test4_best_multiplier': best_mult['multiplier'],
    'test5_premature_exits': {
        'shared_entry_bars': len(same_entry_bars),
        'different_exit_outcomes': len(different_exits),
        'premature_exits': len(premature),
        'delayed_exits': len(delayed),
        'only_in_baseline': len(only_in_baseline),
        'only_in_intrabar': len(only_in_intrabar),
        'avg_pnl_diff_premature': round(np.mean([d['pnl_diff'] for d in premature]), 4) if premature else 0,
        'total_pnl_impact_premature': round(sum(d['pnl_diff'] for d in premature), 1) if premature else 0,
    },
}

out_path = 'results/intrabar_trail_impact.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Results saved to {out_path}")
