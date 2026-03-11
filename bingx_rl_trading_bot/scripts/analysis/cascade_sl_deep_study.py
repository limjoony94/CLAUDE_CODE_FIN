#!/usr/bin/env python3
"""
Cascade SL Deep Cross-Validation & Optimization Study

Current: tighten_pct=85% (SL distance × 0.15 after same-dir SL exit)

Phases:
  1. Baseline ablation: Cascade ON vs OFF (IS + WF)
  2. Tighten-pct sweep: 50-95% in 5% steps (IS + WF)
  3. Trigger scope: SL-only vs TP+SL vs ALL exits
  4. Direction isolation: same-dir vs both-dir vs counter-dir
  5. Delay/debounce: immediate vs 1-bar delay vs 2-bar delay
  6. Progressive tightening: linear vs exponential vs step
  7. MC discrimination: 5-seed random baseline (are improvements genuine?)
  8. WF final validation: best config 3-fold expanding window
  9. Summary & recommendation

Standard Research Protocol: next-bar entry, intrabar exit, 0.10% fee,
  0.02% slippage, 3x leverage, compound sizing.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import sys
import time

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope, _check_exit_npos,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/cascade_sl_deep_study.json"

NPOS_DEFAULTS = dict(
    n_slots=DEFAULT_N_SLOTS,
    direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
    agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
    momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO,
    clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS,
)


def run_npos(signals, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope,
             start_bar, end_bar, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
             **extra_kwargs):
    """Run portfolio_npos with optional cascade override."""
    kwargs = {**NPOS_DEFAULTS}
    kwargs.update(extra_kwargs)
    kwargs['cascade_tighten_pct'] = cascade_tighten_pct

    trades, raw_stats = portfolio_npos(
        signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        **kwargs
    )
    stats = calc_stats_compound(trades)
    # Merge raw_stats (mdd_mtm etc), prefer MTM MDD
    if raw_stats.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw_stats['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw_stats.items() if k not in stats})
    return trades, stats


def run_wf(signals, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope,
           ns, ne, n_folds=3, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
           **extra_kwargs):
    """Run expanding-window WF validation within neutral window [ns, ne)."""
    total = ne - ns
    fold_size = total // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        train_end = ns + fold_size * (fold + 2)
        test_start = train_end
        test_end = min(train_end + fold_size, ne)
        if test_start >= test_end:
            continue

        trades, stats = run_npos(
            signals, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, test_start, test_end,
            cascade_tighten_pct=cascade_tighten_pct,
            **extra_kwargs
        )
        results.append({
            'fold': fold + 1,
            'oos_start': int(test_start),
            'oos_end': int(test_end),
            'pnl': stats.get('pnl', 0),
            'wr': stats.get('wr', 0),
            'trades': stats.get('trades', 0),
            'mdd': stats.get('mdd', 0),
        })

    oos_total_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results)
    return results, oos_total_pnl, all_pass


def fmt_stats(stats):
    """Format stats for printing."""
    return (f"PnL {stats.get('pnl', 0):+.1f}%, "
            f"WR {stats.get('wr', 0):.1f}%, "
            f"Trades {stats.get('trades', 0)}, "
            f"MDD {stats.get('mdd', 0):.2f}%")


# ─── Load data ───
print("=" * 70)
print("CASCADE SL DEEP CROSS-VALIDATION & OPTIMIZATION STUDY")
print("=" * 70)

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
pattern_details = pat_data.get('pattern_details', {})
print(f"Loaded {len(pattern_details)} patterns, {n_bars} bars")

# Build signal index (pattern_name -> [bar indices])
signal_index = build_signal_index(type_codes, n_bars)

# Build signal tuples: (sig_bar, pat_key, direction, tp_pct, sl_pct)
def build_signal_tuples(patterns, sig_idx):
    tuples = []
    for k, v in patterns.items():
        pat_name = v.get('pattern') or k.rsplit('_', 1)[0]
        if pat_name in sig_idx:
            for bar in sig_idx[pat_name]:
                tuples.append((bar, k, v['direction'], v['tp'], v['sl']))
    return tuples

signals = build_signal_tuples(pattern_details, signal_index)
ns, ne = find_neutral_window(closes)
n_signals_range = len([s for s in signals if ns <= s[0] < ne])
print(f"Built {len(signals)} signals ({n_signals_range} in neutral window {ns}-{ne}, "
      f"{(ne-ns)//288:.0f}d)")

all_results = {
    "study": "cascade_sl_deep_cross_validation",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "data_bars": n_bars,
    "patterns": len(pattern_details),
    "signals": len(signals),
    "baseline_cascade_pct": DEFAULT_CASCADE_TIGHTEN_PCT,
}


# ═══════════════════════════════════════════════════════════
# PHASE 1: Baseline Ablation — Cascade ON vs OFF
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: Baseline Ablation — Cascade ON (85%) vs OFF (0%)")
print("=" * 70)

_, stats_on = run_npos(signals, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)
_, stats_off = run_npos(signals, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=0)

print(f"  Cascade ON  (85%): {fmt_stats(stats_on)}")
print(f"  Cascade OFF  (0%): {fmt_stats(stats_off)}")

pnl_diff = stats_on['pnl'] - stats_off['pnl']
mdd_diff = stats_on.get('mdd', 0) - stats_off.get('mdd', 0)
pnl_mdd_on = stats_on['pnl'] / max(stats_on.get('mdd', 1), 0.01)
pnl_mdd_off = stats_off['pnl'] / max(stats_off.get('mdd', 1), 0.01)
print(f"  ΔPnL: {pnl_diff:+.1f}%, ΔMDD: {mdd_diff:+.2f}%")
print(f"  PnL/MDD ON: {pnl_mdd_on:.1f}x, OFF: {pnl_mdd_off:.1f}x")

# WF validation for both
wf_on, oos_pnl_on, pass_on = run_wf(signals, opens, highs, lows, closes, n_bars,
                                      atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)
wf_off, oos_pnl_off, pass_off = run_wf(signals, opens, highs, lows, closes, n_bars,
                                         atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=0)

wf_on_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_on)
wf_off_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_off)
print(f"\n  WF ON:  OOS {oos_pnl_on:+.1f}% {'PASS' if pass_on else 'FAIL'} [{wf_on_str}]")
print(f"  WF OFF: OOS {oos_pnl_off:+.1f}% {'PASS' if pass_off else 'FAIL'} [{wf_off_str}]")

all_results["phase1_ablation"] = {
    "cascade_on": {"pnl": stats_on['pnl'], "wr": stats_on.get('wr', 0),
                   "mdd": stats_on.get('mdd', 0), "trades": stats_on.get('trades', 0),
                   "pnl_mdd": pnl_mdd_on,
                   "wf_folds": wf_on, "oos_pnl": oos_pnl_on, "wf_pass": pass_on},
    "cascade_off": {"pnl": stats_off['pnl'], "wr": stats_off.get('wr', 0),
                    "mdd": stats_off.get('mdd', 0), "trades": stats_off.get('trades', 0),
                    "pnl_mdd": pnl_mdd_off,
                    "wf_folds": wf_off, "oos_pnl": oos_pnl_off, "wf_pass": pass_off},
    "delta_pnl": pnl_diff, "delta_mdd": mdd_diff,
}


# ═══════════════════════════════════════════════════════════
# PHASE 2: Tighten-Pct Sweep (50-95%, 5% steps)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Tighten-Pct Sweep (50-95%)")
print("=" * 70)

sweep_pcts = list(range(50, 100, 5))
sweep_results = []

for pct in sweep_pcts:
    _, stats = run_npos(signals, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=pct)
    wf_folds, oos_pnl, wf_pass = run_wf(signals, opens, highs, lows, closes, n_bars,
                                          atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=pct)
    pnl_mdd = stats['pnl'] / max(stats.get('mdd', 1), 0.01)

    row = {
        "pct": pct,
        "is_pnl": stats['pnl'],
        "is_wr": stats.get('wr', 0),
        "is_mdd": stats.get('mdd', 0),
        "is_trades": stats.get('trades', 0),
        "is_pnl_mdd": pnl_mdd,
        "oos_pnl": oos_pnl,
        "wf_pass": wf_pass,
        "wf_folds": wf_folds,
    }
    sweep_results.append(row)
    marker = "←CURRENT" if pct == 85 else ("★BEST" if pnl_mdd == max(r['is_pnl_mdd'] for r in sweep_results) else "")
    print(f"  {pct}%: IS PnL {stats['pnl']:+.1f}%, WR {stats.get('wr',0):.1f}%, "
          f"MDD {stats.get('mdd',0):.2f}%, PnL/MDD {pnl_mdd:.1f}x, "
          f"OOS {oos_pnl:+.1f}% {'PASS' if wf_pass else 'FAIL'} {marker}")

# Find best by PnL/MDD among WF-PASS configs
passing = [r for r in sweep_results if r['wf_pass']]
if passing:
    best_sweep = max(passing, key=lambda r: r['is_pnl_mdd'])
else:
    best_sweep = max(sweep_results, key=lambda r: r['is_pnl_mdd'])
print(f"\n  Best (WF-PASS, PnL/MDD): {best_sweep['pct']}% — "
      f"IS PnL/MDD {best_sweep['is_pnl_mdd']:.1f}x, OOS {best_sweep['oos_pnl']:+.1f}%")

all_results["phase2_sweep"] = {
    "configs": sweep_results,
    "best_pct": best_sweep['pct'],
    "best_pnl_mdd": best_sweep['is_pnl_mdd'],
}


# ═══════════════════════════════════════════════════════════
# PHASE 3: Trigger Scope — Which exits trigger cascade?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Trigger Scope — SL-only vs All-exits vs TP+SL")
print("=" * 70)
print("  (Current production: SL-only triggers cascade)")

# We need to modify the cascade logic for this.
# Current: cascade only on SL exits (reason == 'SL')
# Variant A: trigger on ALL exits (TP + SL)
# Variant B: trigger on SL + TIMEOUT (non-TP)

# Since we can't easily modify portfolio_npos for trigger scope,
# we'll build a custom mini-simulator for this phase
# that wraps the existing exit check with different cascade triggers.

def portfolio_npos_custom_cascade(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    cascade_tighten_pct=85,
    cascade_trigger='SL',  # 'SL', 'ALL', 'SL+TIMEOUT'
):
    """Custom N-pos simulator with configurable cascade triggers."""
    size_pct = 100.0 / DEFAULT_N_SLOTS
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_eq = 100.0
    max_dd_mtm = 0.0
    momentum_pause = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_cascade_dirs = set()

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                       atr_ratio, fee,
                                       DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
                                       TIMEOUT_BARS)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    # Check if timeout should trigger cascade
                    if cascade_trigger == 'SL+TIMEOUT':
                        bar_cascade_dirs.add(pos['direction'])
                    elif cascade_trigger == 'ALL':
                        bar_cascade_dirs.add(pos['direction'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio

                # Determine cascade trigger
                reason = result['reason']
                if cascade_trigger == 'SL' and reason == 'SL':
                    bar_cascade_dirs.add(result['direction'])
                elif cascade_trigger == 'ALL':
                    bar_cascade_dirs.add(result['direction'])
                elif cascade_trigger == 'SL+TIMEOUT' and reason == 'SL':
                    bar_cascade_dirs.add(result['direction'])

        # Apply cascade
        if cascade_tighten_pct > 0 and bar_cascade_dirs:
            keep_ratio = 1.0 - cascade_tighten_pct / 100.0
            for sl_dir in bar_cascade_dirs:
                for pos in positions:
                    if pos['slot'] in closed_slots:
                        continue
                    if pos['direction'] != sl_dir:
                        continue
                    sig = pos['signal_bar']
                    if (atr_ratio is not None and sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[sig])):
                        r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sig]))
                    else:
                        r = 1.0
                    cur_eff_sl = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]
        equity += bar_pnl_sum
        if equity > peak_eq:
            peak_eq = equity

        # Momentum guard
        if DEFAULT_MOMENTUM_LOOKBACK > 0 and DEFAULT_MOMENTUM_THRESHOLD > 0 and bar >= DEFAULT_MOMENTUM_LOOKBACK:
            pn = closes[bar]
            pa = closes[bar - DEFAULT_MOMENTUM_LOOKBACK]
            if pa > 0:
                pc = (pn / pa - 1) * 100
                if pc > DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['SHORT'] = bar + DEFAULT_MOMENTUM_COOLDOWN
                elif pc < -DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['LONG'] = bar + DEFAULT_MOMENTUM_COOLDOWN

        # Entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= DEFAULT_N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DEFAULT_DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            if DEFAULT_MOMENTUM_LOOKBACK > 0 and bar < momentum_pause.get(direction, -1):
                continue

            # Regime sizing
            sm = 1.0
            if DEFAULT_REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = DEFAULT_REGIME_MULT
                elif s <= 0 and direction == 'LONG':
                    sm = DEFAULT_REGIME_MULT

            # AggRisk check
            is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
            is_counter = ((is_uptrend and direction == 'SHORT') or
                          (not is_uptrend and direction == 'LONG'))
            cap_pct = DEFAULT_AGG_RISK_COUNTER if is_counter else DEFAULT_AGG_RISK_WITH

            existing_exp = 0.0
            for p in positions:
                if p['direction'] == direction:
                    p_sl = p['sl_pct']
                    p_sig = p['signal_bar']
                    if (atr_ratio is not None and p_sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[p_sig])):
                        p_r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[p_sig]))
                    else:
                        p_r = 1.0
                    p_eff = p_sl * p_r
                    p_sm = p.get('size_mult', 1.0)
                    existing_exp += p_eff * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * p_sm

            new_r = 1.0
            if (atr_ratio is not None and sig_bar < len(atr_ratio)
                    and not np.isnan(atr_ratio[sig_bar])):
                new_r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sig_bar]))
            new_exp = sl_pct * new_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * sm

            if existing_exp + new_exp > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
            })

    stats = calc_stats_compound(trades)
    return trades, stats


trigger_modes = ['SL', 'ALL', 'SL+TIMEOUT']
p3_results = {}

for trigger in trigger_modes:
    trades, stats = portfolio_npos_custom_cascade(
        signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        cascade_tighten_pct=85, cascade_trigger=trigger
    )
    pnl_mdd = stats['pnl'] / max(stats.get('mdd', 1), 0.01)
    marker = "←CURRENT" if trigger == 'SL' else ""
    print(f"  Trigger={trigger:12s}: PnL {stats['pnl']:+.1f}%, "
          f"WR {stats.get('wr',0):.1f}%, MDD {stats.get('mdd',0):.2f}%, "
          f"PnL/MDD {pnl_mdd:.1f}x, Trades {stats.get('trades',0)} {marker}")
    p3_results[trigger] = {
        "pnl": stats['pnl'], "wr": stats.get('wr', 0),
        "mdd": stats.get('mdd', 0), "trades": stats.get('trades', 0),
        "pnl_mdd": pnl_mdd,
    }

all_results["phase3_trigger_scope"] = p3_results


# ═══════════════════════════════════════════════════════════
# PHASE 4: Direction Isolation — same-dir vs both-dir vs counter-dir
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Direction Isolation")
print("=" * 70)

def portfolio_npos_dir_cascade(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    cascade_tighten_pct=85,
    cascade_dir_mode='same',  # 'same', 'both', 'counter'
):
    """N-pos simulator with direction-based cascade targeting."""
    size_pct = 100.0 / DEFAULT_N_SLOTS
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_eq = 100.0
    momentum_pause = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        closed_slots = []
        bar_pnl_sum = 0.0
        sl_exit_dirs = set()

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                       atr_ratio, fee,
                                       DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
                                       TIMEOUT_BARS)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    sl_exit_dirs.add(result['direction'])

        # Apply cascade with direction mode
        if cascade_tighten_pct > 0 and sl_exit_dirs:
            keep_ratio = 1.0 - cascade_tighten_pct / 100.0
            for sl_dir in sl_exit_dirs:
                for pos in positions:
                    if pos['slot'] in closed_slots:
                        continue

                    # Direction filter
                    if cascade_dir_mode == 'same' and pos['direction'] != sl_dir:
                        continue
                    elif cascade_dir_mode == 'counter' and pos['direction'] == sl_dir:
                        continue
                    # 'both' → no filter, apply to all

                    sig = pos['signal_bar']
                    if (atr_ratio is not None and sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[sig])):
                        r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sig]))
                    else:
                        r = 1.0
                    cur_eff_sl = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]
        equity += bar_pnl_sum
        if equity > peak_eq:
            peak_eq = equity

        # Momentum guard
        if DEFAULT_MOMENTUM_LOOKBACK > 0 and DEFAULT_MOMENTUM_THRESHOLD > 0 and bar >= DEFAULT_MOMENTUM_LOOKBACK:
            pn = closes[bar]
            pa = closes[bar - DEFAULT_MOMENTUM_LOOKBACK]
            if pa > 0:
                pc = (pn / pa - 1) * 100
                if pc > DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['SHORT'] = bar + DEFAULT_MOMENTUM_COOLDOWN
                elif pc < -DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['LONG'] = bar + DEFAULT_MOMENTUM_COOLDOWN

        # Entries (same as standard)
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= DEFAULT_N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DEFAULT_DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            if DEFAULT_MOMENTUM_LOOKBACK > 0 and bar < momentum_pause.get(direction, -1):
                continue

            sm = 1.0
            if DEFAULT_REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = DEFAULT_REGIME_MULT
                elif s <= 0 and direction == 'LONG':
                    sm = DEFAULT_REGIME_MULT

            is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
            is_counter = ((is_uptrend and direction == 'SHORT') or
                          (not is_uptrend and direction == 'LONG'))
            cap_pct = DEFAULT_AGG_RISK_COUNTER if is_counter else DEFAULT_AGG_RISK_WITH

            existing_exp = 0.0
            for p in positions:
                if p['direction'] == direction:
                    p_sl = p['sl_pct']
                    p_sig = p['signal_bar']
                    if (atr_ratio is not None and p_sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[p_sig])):
                        p_r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[p_sig]))
                    else:
                        p_r = 1.0
                    existing_exp += p_sl * p_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * p.get('size_mult', 1.0)

            new_r = 1.0
            if (atr_ratio is not None and sig_bar < len(atr_ratio)
                    and not np.isnan(atr_ratio[sig_bar])):
                new_r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sig_bar]))
            new_exp = sl_pct * new_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * sm

            if existing_exp + new_exp > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
            })

    stats = calc_stats_compound(trades)
    return trades, stats


dir_modes = ['same', 'both', 'counter']
p4_results = {}

for mode in dir_modes:
    trades, stats = portfolio_npos_dir_cascade(
        signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        cascade_tighten_pct=85, cascade_dir_mode=mode
    )
    pnl_mdd = stats['pnl'] / max(stats.get('mdd', 1), 0.01)
    marker = "←CURRENT" if mode == 'same' else ""
    print(f"  Dir={mode:8s}: PnL {stats['pnl']:+.1f}%, "
          f"WR {stats.get('wr',0):.1f}%, MDD {stats.get('mdd',0):.2f}%, "
          f"PnL/MDD {pnl_mdd:.1f}x, Trades {stats.get('trades',0)} {marker}")
    p4_results[mode] = {
        "pnl": stats['pnl'], "wr": stats.get('wr', 0),
        "mdd": stats.get('mdd', 0), "trades": stats.get('trades', 0),
        "pnl_mdd": pnl_mdd,
    }

all_results["phase4_direction"] = p4_results


# ═══════════════════════════════════════════════════════════
# PHASE 5: Progressive Tightening Modes
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: Progressive Tightening — Linear vs Fixed vs Step")
print("=" * 70)
print("  (Current: fixed 85% per cascade event, compounds multiplicatively)")

# Variant A: Fixed 85% (current) — already tested
# Variant B: Linear step — each cascade reduces SL by fixed amount (e.g., 30% of original)
# Variant C: Diminishing — first cascade 85%, second 50%, third 25%
# Variant D: Hard cap — cascade to minimum 0.5% SL distance (no compounding)

def portfolio_npos_progressive_cascade(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    cascade_mode='fixed',  # 'fixed', 'linear', 'diminishing', 'hard_cap'
    base_tighten_pct=85,
):
    """N-pos simulator with different cascade progression modes."""
    size_pct = 100.0 / DEFAULT_N_SLOTS
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_eq = 100.0
    momentum_pause = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0
        sl_exit_dirs = set()

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                       atr_ratio, fee,
                                       DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
                                       TIMEOUT_BARS)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1
                    sl_exit_dirs.add(result['direction'])

        # Apply cascade with progression mode
        if base_tighten_pct > 0 and bar_sl_count > 0 and sl_exit_dirs:
            for sl_dir in sl_exit_dirs:
                for pos in positions:
                    if pos['slot'] in closed_slots:
                        continue
                    if pos['direction'] != sl_dir:
                        continue

                    sig = pos['signal_bar']
                    if (atr_ratio is not None and sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[sig])):
                        r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sig]))
                    else:
                        r = 1.0

                    original_sl = pos['sl_pct'] * r  # Original ATR-scaled SL
                    cur_eff_sl = pos.get('eff_sl_override') or original_sl
                    cascade_count = pos.get('cascade_count', 0)

                    if cascade_mode == 'fixed':
                        # Current: multiply by keep_ratio each time
                        keep = 1.0 - base_tighten_pct / 100.0
                        new_sl = cur_eff_sl * keep

                    elif cascade_mode == 'linear':
                        # Subtract fixed % of ORIGINAL each time
                        step = original_sl * (base_tighten_pct / 100.0) * 0.3  # 30% of original per step
                        new_sl = max(0.1, cur_eff_sl - step)

                    elif cascade_mode == 'diminishing':
                        # Diminishing tighten: 85%, 50%, 25% for 1st, 2nd, 3rd cascade
                        diminish_schedule = [base_tighten_pct, 50, 25, 10]
                        pct = diminish_schedule[min(cascade_count, len(diminish_schedule) - 1)]
                        keep = 1.0 - pct / 100.0
                        new_sl = cur_eff_sl * keep

                    elif cascade_mode == 'hard_cap':
                        # Tighten but never below 0.5% SL distance
                        keep = 1.0 - base_tighten_pct / 100.0
                        new_sl = max(0.5, cur_eff_sl * keep)

                    pos['eff_sl_override'] = new_sl
                    pos['cascade_count'] = cascade_count + 1

        positions = [p for p in positions if p['slot'] not in closed_slots]
        equity += bar_pnl_sum
        if equity > peak_eq:
            peak_eq = equity

        # Momentum guard
        if DEFAULT_MOMENTUM_LOOKBACK > 0 and DEFAULT_MOMENTUM_THRESHOLD > 0 and bar >= DEFAULT_MOMENTUM_LOOKBACK:
            pn = closes[bar]
            pa = closes[bar - DEFAULT_MOMENTUM_LOOKBACK]
            if pa > 0:
                pc = (pn / pa - 1) * 100
                if pc > DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['SHORT'] = bar + DEFAULT_MOMENTUM_COOLDOWN
                elif pc < -DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['LONG'] = bar + DEFAULT_MOMENTUM_COOLDOWN

        # Entries (standard)
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= DEFAULT_N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DEFAULT_DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            if DEFAULT_MOMENTUM_LOOKBACK > 0 and bar < momentum_pause.get(direction, -1):
                continue

            sm = 1.0
            if DEFAULT_REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = DEFAULT_REGIME_MULT
                elif s <= 0 and direction == 'LONG':
                    sm = DEFAULT_REGIME_MULT

            is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
            is_counter = ((is_uptrend and direction == 'SHORT') or
                          (not is_uptrend and direction == 'LONG'))
            cap_pct = DEFAULT_AGG_RISK_COUNTER if is_counter else DEFAULT_AGG_RISK_WITH

            existing_exp = 0.0
            for p in positions:
                if p['direction'] == direction:
                    p_sl = p['sl_pct']
                    p_sig = p['signal_bar']
                    if (atr_ratio is not None and p_sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[p_sig])):
                        p_r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[p_sig]))
                    else:
                        p_r = 1.0
                    existing_exp += p_sl * p_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * p.get('size_mult', 1.0)

            new_r = 1.0
            if (atr_ratio is not None and sig_bar < len(atr_ratio)
                    and not np.isnan(atr_ratio[sig_bar])):
                new_r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sig_bar]))
            new_exp = sl_pct * new_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * sm

            if existing_exp + new_exp > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
            })

    stats = calc_stats_compound(trades)
    return trades, stats


progress_modes = ['fixed', 'linear', 'diminishing', 'hard_cap']
p5_results = {}

for mode in progress_modes:
    trades, stats = portfolio_npos_progressive_cascade(
        signals, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        cascade_mode=mode, base_tighten_pct=85
    )
    pnl_mdd = stats['pnl'] / max(stats.get('mdd', 1), 0.01)
    marker = "←CURRENT" if mode == 'fixed' else ""
    print(f"  Mode={mode:13s}: PnL {stats['pnl']:+.1f}%, "
          f"WR {stats.get('wr',0):.1f}%, MDD {stats.get('mdd',0):.2f}%, "
          f"PnL/MDD {pnl_mdd:.1f}x, Trades {stats.get('trades',0)} {marker}")
    p5_results[mode] = {
        "pnl": stats['pnl'], "wr": stats.get('wr', 0),
        "mdd": stats.get('mdd', 0), "trades": stats.get('trades', 0),
        "pnl_mdd": pnl_mdd,
    }

all_results["phase5_progression"] = p5_results


# ═══════════════════════════════════════════════════════════
# PHASE 6: MC Discrimination Test (5 seeds)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: MC Discrimination — Random baseline (5 seeds)")
print("=" * 70)

mc_seeds = [42, 123, 7, 999, 314]
mc_results = []

# Randomize signal directions and check if cascade still "helps"
rng_base = np.random.RandomState(42)

for seed in mc_seeds:
    rng = np.random.RandomState(seed)
    # Shuffle signal directions randomly
    random_signals = []
    for sig_bar, pat, direction, tp_pct, sl_pct in signals:
        rand_dir = rng.choice(['LONG', 'SHORT'])
        random_signals.append((sig_bar, pat, rand_dir, tp_pct, sl_pct))

    _, stats_on = run_npos(random_signals, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)
    _, stats_off = run_npos(random_signals, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=0)

    delta_pnl = stats_on['pnl'] - stats_off['pnl']
    wf_folds, oos_pnl, wf_pass = run_wf(random_signals, opens, highs, lows, closes, n_bars,
                                          atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85)

    mc_results.append({
        "seed": seed,
        "cascade_on_pnl": stats_on['pnl'],
        "cascade_off_pnl": stats_off['pnl'],
        "delta_pnl": delta_pnl,
        "wf_pass": wf_pass,
        "oos_pnl": oos_pnl,
    })
    print(f"  Seed {seed}: ON {stats_on['pnl']:+.1f}%, OFF {stats_off['pnl']:+.1f}%, "
          f"Δ{delta_pnl:+.1f}%, WF {'PASS' if wf_pass else 'FAIL'}")

# Discrimination: how many random seeds show cascade "helping"?
random_helps = sum(1 for r in mc_results if r['delta_pnl'] > 0)
random_wf_pass = sum(1 for r in mc_results if r['wf_pass'])
print(f"\n  Random cascade helps: {random_helps}/{len(mc_seeds)}")
print(f"  Random WF PASS: {random_wf_pass}/{len(mc_seeds)}")

discriminating = random_wf_pass < len(mc_seeds) // 2  # <50% random pass = discriminating
print(f"  Discrimination: {'YES (genuine edge)' if discriminating else 'NO (non-discriminating)'}")

all_results["phase6_mc"] = {
    "seeds": mc_results,
    "random_helps": random_helps,
    "random_wf_pass": random_wf_pass,
    "discriminating": discriminating,
}


# ═══════════════════════════════════════════════════════════
# PHASE 7: Optimal Config — Fine-grained sweep around best
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 7: Fine-Grained Sweep Around Best Config")
print("=" * 70)

# From Phase 2, get best pct and sweep ±5% in 1% steps
best_pct_phase2 = best_sweep['pct']
fine_range = list(range(max(50, best_pct_phase2 - 10), min(96, best_pct_phase2 + 11)))
fine_results = []

for pct in fine_range:
    _, stats = run_npos(signals, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=pct)
    pnl_mdd = stats['pnl'] / max(stats.get('mdd', 1), 0.01)
    fine_results.append({
        "pct": pct,
        "is_pnl": stats['pnl'],
        "is_wr": stats.get('wr', 0),
        "is_mdd": stats.get('mdd', 0),
        "is_pnl_mdd": pnl_mdd,
    })

# Find top 3
fine_sorted = sorted(fine_results, key=lambda r: r['is_pnl_mdd'], reverse=True)
print(f"  Top 5 by PnL/MDD:")
for r in fine_sorted[:5]:
    marker = "←CURRENT" if r['pct'] == 85 else ""
    print(f"    {r['pct']}%: PnL {r['is_pnl']:+.1f}%, MDD {r['is_mdd']:.2f}%, "
          f"PnL/MDD {r['is_pnl_mdd']:.1f}x {marker}")

# WF validate top 3 (if different from current 85%)
top3_pcts = list(set([r['pct'] for r in fine_sorted[:3]] + [85]))
p7_wf = {}
for pct in top3_pcts:
    wf_folds, oos_pnl, wf_pass = run_wf(signals, opens, highs, lows, closes, n_bars,
                                          atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=pct)
    p7_wf[pct] = {"oos_pnl": oos_pnl, "wf_pass": wf_pass, "folds": wf_folds}
    marker = "←CURRENT" if pct == 85 else ""
    print(f"  WF {pct}%: OOS {oos_pnl:+.1f}% {'PASS' if wf_pass else 'FAIL'} {marker}")

all_results["phase7_fine_sweep"] = {
    "range": [fine_range[0], fine_range[-1]],
    "top5": fine_sorted[:5],
    "wf_validation": {str(k): v for k, v in p7_wf.items()},
}


# ═══════════════════════════════════════════════════════════
# PHASE 8: Cascade Event Analysis
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 8: Cascade Event Deep Analysis")
print("=" * 70)

# Run with cascade ON and track cascade events
def portfolio_npos_with_cascade_tracking(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    cascade_tighten_pct=85,
):
    """Track cascade events for analysis."""
    size_pct = 100.0 / DEFAULT_N_SLOTS
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    cascade_events = []
    momentum_pause = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        closed_slots = []
        bar_sl_count = 0
        sl_exit_dirs = set()

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                       atr_ratio, fee,
                                       DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
                                       TIMEOUT_BARS)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                if result['reason'] == 'SL':
                    bar_sl_count += 1
                    sl_exit_dirs.add(result['direction'])

        if cascade_tighten_pct > 0 and bar_sl_count > 0 and sl_exit_dirs:
            keep_ratio = 1.0 - cascade_tighten_pct / 100.0
            affected_count = 0
            for sl_dir in sl_exit_dirs:
                for pos in positions:
                    if pos['slot'] in closed_slots:
                        continue
                    if pos['direction'] != sl_dir:
                        continue

                    sig = pos['signal_bar']
                    if (atr_ratio is not None and sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[sig])):
                        r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sig]))
                    else:
                        r = 1.0
                    cur_eff_sl = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio
                    pos['cascade_count'] = pos.get('cascade_count', 0) + 1
                    affected_count += 1

            if affected_count > 0:
                cascade_events.append({
                    'bar': int(bar),
                    'sl_exits': bar_sl_count,
                    'directions': list(sl_exit_dirs),
                    'affected_positions': affected_count,
                })

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Momentum guard
        if DEFAULT_MOMENTUM_LOOKBACK > 0 and DEFAULT_MOMENTUM_THRESHOLD > 0 and bar >= DEFAULT_MOMENTUM_LOOKBACK:
            pn = closes[bar]
            pa = closes[bar - DEFAULT_MOMENTUM_LOOKBACK]
            if pa > 0:
                pc = (pn / pa - 1) * 100
                if pc > DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['SHORT'] = bar + DEFAULT_MOMENTUM_COOLDOWN
                elif pc < -DEFAULT_MOMENTUM_THRESHOLD:
                    momentum_pause['LONG'] = bar + DEFAULT_MOMENTUM_COOLDOWN

        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= DEFAULT_N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DEFAULT_DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            if DEFAULT_MOMENTUM_LOOKBACK > 0 and bar < momentum_pause.get(direction, -1):
                continue

            sm = 1.0
            if DEFAULT_REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = DEFAULT_REGIME_MULT
                elif s <= 0 and direction == 'LONG':
                    sm = DEFAULT_REGIME_MULT

            is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
            is_counter = ((is_uptrend and direction == 'SHORT') or
                          (not is_uptrend and direction == 'LONG'))
            cap_pct = DEFAULT_AGG_RISK_COUNTER if is_counter else DEFAULT_AGG_RISK_WITH

            existing_exp = 0.0
            for p in positions:
                if p['direction'] == direction:
                    p_sl = p['sl_pct']
                    p_sig = p['signal_bar']
                    if (atr_ratio is not None and p_sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[p_sig])):
                        p_r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[p_sig]))
                    else:
                        p_r = 1.0
                    existing_exp += p_sl * p_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * p.get('size_mult', 1.0)

            new_r = 1.0
            if (atr_ratio is not None and sig_bar < len(atr_ratio)
                    and not np.isnan(atr_ratio[sig_bar])):
                new_r = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sig_bar]))
            new_exp = sl_pct * new_r * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * sm

            if existing_exp + new_exp > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
            })

    stats = calc_stats_compound(trades)
    return trades, stats, cascade_events


trades_tracked, stats_tracked, cascade_events = portfolio_npos_with_cascade_tracking(
    signals, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne, cascade_tighten_pct=85
)

total_cascade = len(cascade_events)
total_affected = sum(e['affected_positions'] for e in cascade_events)
multi_sl_bars = sum(1 for e in cascade_events if e['sl_exits'] >= 2)
avg_affected = total_affected / total_cascade if total_cascade > 0 else 0

# Track cascade outcomes: what happens to positions AFTER cascade tightening
# (this requires matching cascade events to subsequent trades)
cascade_subsequent_exits = {'TP': 0, 'SL': 0, 'TIMEOUT': 0, 'CASCADE_SL': 0, 'other': 0}
# Count trades where a cascade-tightened position exits
for t in trades_tracked:
    reason = t.get('reason', '')
    # Can't directly track cascade state here, but we can check cascade_count
    # Unfortunately our simplified sim doesn't propagate that to trades
    # So we'll just report the event stats

print(f"  Total cascade events: {total_cascade}")
print(f"  Total positions affected: {total_affected}")
print(f"  Avg positions affected per event: {avg_affected:.1f}")
print(f"  Multi-SL bars (≥2 SL same bar): {multi_sl_bars}")
if cascade_events:
    dir_dist = defaultdict(int)
    for e in cascade_events:
        for d in e['directions']:
            dir_dist[d] += 1
    print(f"  Direction distribution: LONG {dir_dist.get('LONG', 0)}, SHORT {dir_dist.get('SHORT', 0)}")

    # Temporal distribution
    bars_per_quarter = n_bars // 4
    quarters = [0, 0, 0, 0]
    for e in cascade_events:
        q = min(e['bar'] // bars_per_quarter, 3)
        quarters[q] += 1
    print(f"  Temporal distribution (Q1-Q4): {quarters}")

all_results["phase8_events"] = {
    "total_cascade_events": total_cascade,
    "total_affected": total_affected,
    "avg_affected_per_event": avg_affected,
    "multi_sl_bars": multi_sl_bars,
}


# ═══════════════════════════════════════════════════════════
# PHASE 9: Final Summary & Recommendation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 9: SUMMARY & RECOMMENDATION")
print("=" * 70)

# Compile all findings
print(f"\n  Phase 1 (Ablation): Cascade ON vs OFF")
print(f"    ΔPnL: {all_results['phase1_ablation']['delta_pnl']:+.1f}%")
print(f"    PnL/MDD: {pnl_mdd_on:.1f}x vs {pnl_mdd_off:.1f}x")
print(f"    WF: ON={oos_pnl_on:+.1f}% {'PASS' if pass_on else 'FAIL'}, "
      f"OFF={oos_pnl_off:+.1f}% {'PASS' if pass_off else 'FAIL'}")

print(f"\n  Phase 2 (Sweep): Best={best_sweep['pct']}% "
      f"(PnL/MDD {best_sweep['is_pnl_mdd']:.1f}x)")
print(f"    Current 85%: PnL/MDD {next((r['is_pnl_mdd'] for r in sweep_results if r['pct']==85), 0):.1f}x")

print(f"\n  Phase 3 (Trigger): Best=", end="")
best_trigger = max(p3_results.items(), key=lambda x: x[1]['pnl_mdd'])
print(f"{best_trigger[0]} (PnL/MDD {best_trigger[1]['pnl_mdd']:.1f}x)")

print(f"\n  Phase 4 (Direction): Best=", end="")
best_dir = max(p4_results.items(), key=lambda x: x[1]['pnl_mdd'])
print(f"{best_dir[0]} (PnL/MDD {best_dir[1]['pnl_mdd']:.1f}x)")

print(f"\n  Phase 5 (Progression): Best=", end="")
best_prog = max(p5_results.items(), key=lambda x: x[1]['pnl_mdd'])
print(f"{best_prog[0]} (PnL/MDD {best_prog[1]['pnl_mdd']:.1f}x)")

print(f"\n  Phase 6 (MC): {'DISCRIMINATING' if discriminating else 'NON-DISCRIMINATING'}")
print(f"    Random helps: {random_helps}/{len(mc_seeds)}, Random WF PASS: {random_wf_pass}/{len(mc_seeds)}")

# Final recommendation
current_pnl_mdd = next((r['is_pnl_mdd'] for r in sweep_results if r['pct'] == 85), 0)
best_overall_pnl_mdd = fine_sorted[0]['is_pnl_mdd'] if fine_sorted else 0
best_overall_pct = fine_sorted[0]['pct'] if fine_sorted else 85

# Check if best overall passes WF
best_wf = p7_wf.get(best_overall_pct, {})
best_wf_pass = best_wf.get('wf_pass', False)

if best_overall_pct != 85 and best_wf_pass and best_overall_pnl_mdd > current_pnl_mdd * 1.05:
    recommendation = f"CHANGE to {best_overall_pct}%"
    explanation = (f"PnL/MDD {best_overall_pnl_mdd:.1f}x > current {current_pnl_mdd:.1f}x "
                   f"(+{(best_overall_pnl_mdd/current_pnl_mdd-1)*100:.0f}%), WF PASS")
else:
    recommendation = "KEEP_BASELINE (85%)"
    if best_overall_pct == 85:
        explanation = "Current 85% is already optimal"
    elif not best_wf_pass:
        explanation = f"Best {best_overall_pct}% fails WF validation"
    else:
        explanation = f"Best {best_overall_pct}% improvement < 5% threshold"

print(f"\n  ★ RECOMMENDATION: {recommendation}")
print(f"    Reason: {explanation}")

all_results["recommendation"] = {
    "action": recommendation,
    "reason": explanation,
    "current_pct": 85,
    "current_pnl_mdd": current_pnl_mdd,
    "best_pct": best_overall_pct,
    "best_pnl_mdd": best_overall_pnl_mdd,
    "best_wf_pass": best_wf_pass,
}

elapsed = time.time() - start_time
all_results["elapsed_seconds"] = elapsed
print(f"\n  Elapsed: {elapsed:.1f}s")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Results saved to {OUTPUT_FILE}")
