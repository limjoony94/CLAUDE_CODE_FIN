#!/usr/bin/env python3
"""
TP/SL Methodology — N-pos Portfolio Validation

Phase 1 study found:
  - Per-pattern 1-pos: TP should be lower, SL should be wider
  - Fixed Grid slightly beat MAE/MFE (20p sample)
  - ATR ON confirmed

This script validates whether these findings hold at N-pos portfolio level:
  1. Current baseline (N-pos)
  2. Per-pattern re-optimized TP/SL (1-pos optimal → N-pos test)
  3. Full re-discovery with Fixed Grid method (N-pos evaluation)
  4. Systematic TP/SL shift study (portfolio-level sweep)
  5. Cross-validation with WF
"""

import json
import numpy as np
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    bt_signals_atr, grid_search_best,
    LEVERAGE, FEE_PCT, MAX_BARS,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT, DEFAULT_MIN_TRADES,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/tpsl_npos_validation.json"

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
    cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
)


def load_patterns(filepath=PATTERNS_FILE):
    with open(filepath) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': v['tp'], 'sl': v['sl'],
        }
    return result


def build_signal_tuples(patterns, sig_idx):
    tuples = []
    for k, v in patterns.items():
        pat_name = v.get('pattern') or k.rsplit('_', 1)[0]
        if pat_name in sig_idx:
            for bar in sig_idx[pat_name]:
                tuples.append((bar, k, v['direction'], v['tp'], v['sl']))
    return tuples


def run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, start_bar, end_bar, **extra):
    kwargs = {**NPOS_DEFAULTS}
    kwargs.update(extra)
    trades, raw = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar, **kwargs
    )
    stats = calc_stats_compound(trades)
    if raw.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw.items() if k not in stats})
    return trades, stats


def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, ns, ne, n_folds=3, **extra):
    total = ne - ns
    min_train = total // 3
    fold_size = total // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        train_end = ns + min_train + fold_size * fold
        test_start = train_end
        test_end = min(train_end + fold_size, ne)
        if test_start >= ne or test_end <= test_start:
            continue
        _, stats = run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, test_start, test_end, **extra)
        results.append({'fold': fold + 1, 'pnl': stats.get('pnl', 0),
                        'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0)})
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


# ─── Load data ───
print("=" * 70)
print("TP/SL METHODOLOGY — N-POS PORTFOLIO VALIDATION")
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

signal_index = build_signal_index(type_codes, n_bars)
ns, ne = find_neutral_window(closes)
print(f"Data: {n_bars} bars, neutral window: {ns}-{ne} ({(ne-ns)//288:.0f}d)")

current_patterns = load_patterns()
current_tuples = build_signal_tuples(current_patterns, signal_index)
print(f"Current patterns: {len(current_patterns)}, signals: {len(current_tuples)}")

all_results = {
    "study": "tpsl_npos_validation",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}


# ═══════════════════════════════════════════════════════════
# PHASE 1: Current Baseline (N-pos)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: Current Baseline (N-pos)")
print("=" * 70)

_, baseline = run_npos(current_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, ns, ne)
wf_folds, wf_oos, wf_pass = run_wf(current_tuples, opens, highs, lows, closes,
                                      n_bars, atr_ratio, ema_slope, ns, ne)

print(f"  IS: PnL {baseline['pnl']:+.1f}%, WR {baseline['wr']:.1f}%, "
      f"MDD {baseline['mdd']:.2f}%, PnL/MDD {baseline.get('pnl_mdd', 0):.1f}x, "
      f"Trades {baseline['trades']}")
wf_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_folds)
print(f"  WF: OOS {wf_oos:+.1f}% {'PASS' if wf_pass else 'FAIL'} [{wf_str}]")

all_results["phase1_baseline"] = {
    "pnl": baseline['pnl'], "wr": baseline['wr'], "mdd": baseline['mdd'],
    "pnl_mdd": baseline.get('pnl_mdd', 0), "trades": baseline['trades'],
    "oos_pnl": wf_oos, "wf_pass": wf_pass, "wf_folds": wf_folds,
}


# ═══════════════════════════════════════════════════════════
# PHASE 2: Per-pattern 1-pos re-optimized TP/SL → N-pos test
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Re-optimized TP/SL (per-pattern 2D grid) → N-pos")
print("=" * 70)
print("  Finding optimal TP/SL for each pattern via 1-pos grid search...")

tp_range = np.arange(0.5, 3.0, 0.15)
sl_range = np.arange(1.0, 5.5, 0.2)

optimized_patterns = {}
changes = {'tp_up': 0, 'tp_down': 0, 'sl_up': 0, 'sl_down': 0, 'unchanged': 0}

for key, info in current_patterns.items():
    pat_name = info['pattern']
    direction = info['direction']
    current_tp = info['tp']
    current_sl = info['sl']

    sig_bars = signal_index.get(pat_name, [])
    sig_bars = [s for s in sig_bars if ns <= s < ne]
    if len(sig_bars) < 25:
        # Keep current for patterns with few signals
        optimized_patterns[key] = dict(info)
        changes['unchanged'] += 1
        continue

    best_score = -1
    best_tp_sl = (current_tp, current_sl)

    for tp in tp_range:
        for sl in sl_range:
            trades = bt_signals_atr(
                sig_bars, direction, tp, sl,
                opens, highs, lows, n_bars,
                atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI
            )
            if len(trades) < 20:
                continue
            # Simple PnL/MDD scoring
            pnls = [t[2] for t in trades]
            cum = 0; peak = 0; mdd_val = 0; w = 0
            for p in pnls:
                cum += p
                if cum > peak: peak = cum
                dd = peak - cum
                if dd > mdd_val: mdd_val = dd
                if p > 0: w += 1
            wr = w / len(pnls) * 100
            bwr = sl / (tp + sl) * 100
            edge = wr - bwr
            if edge < 18:
                continue
            if sl < 1.0:
                continue
            score = (cum / mdd_val) if mdd_val > 0 else cum
            if score > best_score:
                best_score = score
                best_tp_sl = (round(tp, 2), round(sl, 2))

    optimized_patterns[key] = {
        'pattern': pat_name, 'direction': direction,
        'tp': best_tp_sl[0], 'sl': best_tp_sl[1],
    }

    tp_diff = best_tp_sl[0] - current_tp
    sl_diff = best_tp_sl[1] - current_sl
    if abs(tp_diff) < 0.2 and abs(sl_diff) < 0.3:
        changes['unchanged'] += 1
    else:
        if tp_diff < -0.2: changes['tp_down'] += 1
        elif tp_diff > 0.2: changes['tp_up'] += 1
        if sl_diff > 0.3: changes['sl_up'] += 1
        elif sl_diff < -0.3: changes['sl_down'] += 1

print(f"  Changes: TP↑{changes['tp_up']} TP↓{changes['tp_down']} "
      f"SL↑{changes['sl_up']} SL↓{changes['sl_down']} "
      f"Unchanged: {changes['unchanged']}")

# Evaluate in N-pos
opt_tuples = build_signal_tuples(optimized_patterns, signal_index)
_, opt_stats = run_npos(opt_tuples, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne)
opt_wf, opt_oos, opt_pass = run_wf(opt_tuples, opens, highs, lows, closes,
                                     n_bars, atr_ratio, ema_slope, ns, ne)

print(f"\n  N-pos Re-optimized:")
print(f"    IS: PnL {opt_stats['pnl']:+.1f}%, WR {opt_stats['wr']:.1f}%, "
      f"MDD {opt_stats['mdd']:.2f}%, PnL/MDD {opt_stats.get('pnl_mdd', 0):.1f}x, "
      f"Trades {opt_stats['trades']}")
wf_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in opt_wf)
print(f"    WF: OOS {opt_oos:+.1f}% {'PASS' if opt_pass else 'FAIL'} [{wf_str}]")

delta_pnl = opt_stats['pnl'] - baseline['pnl']
delta_mdd = opt_stats['mdd'] - baseline['mdd']
print(f"\n  vs Baseline: PnL {delta_pnl:+.1f}%, MDD {delta_mdd:+.2f}%")
print(f"  PnL/MDD: {baseline.get('pnl_mdd', 0):.1f}x → {opt_stats.get('pnl_mdd', 0):.1f}x")

all_results["phase2_reoptimized"] = {
    "pnl": opt_stats['pnl'], "wr": opt_stats['wr'], "mdd": opt_stats['mdd'],
    "pnl_mdd": opt_stats.get('pnl_mdd', 0), "trades": opt_stats['trades'],
    "oos_pnl": opt_oos, "wf_pass": opt_pass, "wf_folds": opt_wf,
    "changes": changes,
    "delta_pnl": delta_pnl,
}


# ═══════════════════════════════════════════════════════════
# PHASE 3: Systematic TP/SL Shift (portfolio-level)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Systematic TP/SL Shift — Portfolio-level Sweep")
print("=" * 70)
print("  Applying uniform TP/SL multipliers to all patterns...")

shift_results = {}
tp_mults = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
sl_mults = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

for tp_m in tp_mults:
    for sl_m in sl_mults:
        shifted_pats = {}
        for k, v in current_patterns.items():
            new_tp = max(0.3, v['tp'] * tp_m)
            new_sl = max(1.0, v['sl'] * sl_m)
            shifted_pats[k] = {
                'pattern': v['pattern'], 'direction': v['direction'],
                'tp': round(new_tp, 3), 'sl': round(new_sl, 3),
            }
        shifted_tuples = build_signal_tuples(shifted_pats, signal_index)
        _, stats = run_npos(shifted_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, ns, ne)
        pm = stats.get('pnl_mdd', 0)
        key_label = f"tp{tp_m:.1f}_sl{sl_m:.1f}"
        shift_results[key_label] = {
            "tp_mult": tp_m, "sl_mult": sl_m,
            "pnl": stats['pnl'], "wr": stats['wr'],
            "mdd": stats['mdd'], "pnl_mdd": pm,
            "trades": stats['trades'],
        }

# Find best combination
best_shift = max(shift_results.values(), key=lambda x: x['pnl_mdd'])
current_shift = shift_results.get('tp1.0_sl1.0', {})

print(f"\n  Best shift: TP×{best_shift['tp_mult']:.1f}, SL×{best_shift['sl_mult']:.1f}")
print(f"    PnL {best_shift['pnl']:+.1f}%, MDD {best_shift['mdd']:.2f}%, "
      f"PnL/MDD {best_shift['pnl_mdd']:.1f}x")
print(f"  Current (1.0×1.0):")
print(f"    PnL {current_shift.get('pnl', 0):+.1f}%, MDD {current_shift.get('mdd', 0):.2f}%, "
      f"PnL/MDD {current_shift.get('pnl_mdd', 0):.1f}x")

# Show top-5 by PnL/MDD
sorted_shifts = sorted(shift_results.values(), key=lambda x: -x['pnl_mdd'])
print(f"\n  Top-5 TP/SL shifts by PnL/MDD:")
for i, s in enumerate(sorted_shifts[:5]):
    marker = " ←CURRENT" if s['tp_mult'] == 1.0 and s['sl_mult'] == 1.0 else ""
    print(f"    {i+1}. TP×{s['tp_mult']:.1f} SL×{s['sl_mult']:.1f}: "
          f"PnL {s['pnl']:+.1f}%, MDD {s['mdd']:.2f}%, "
          f"PnL/MDD {s['pnl_mdd']:.1f}x{marker}")

# Current rank
current_pm = current_shift.get('pnl_mdd', 0)
rank = sum(1 for s in sorted_shifts if s['pnl_mdd'] > current_pm) + 1
print(f"\n  Current (1.0×1.0) rank: {rank}/{len(sorted_shifts)}")

all_results["phase3_shift_sweep"] = {
    "best": best_shift,
    "current": current_shift,
    "current_rank": rank,
    "total_combos": len(sorted_shifts),
    "top5": sorted_shifts[:5],
}

# WF validate best shift if different from current
if best_shift['tp_mult'] != 1.0 or best_shift['sl_mult'] != 1.0:
    shifted_pats = {}
    for k, v in current_patterns.items():
        shifted_pats[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(max(0.3, v['tp'] * best_shift['tp_mult']), 3),
            'sl': round(max(1.0, v['sl'] * best_shift['sl_mult']), 3),
        }
    shifted_tuples = build_signal_tuples(shifted_pats, signal_index)
    wf_f, wf_o, wf_p = run_wf(shifted_tuples, opens, highs, lows, closes,
                                n_bars, atr_ratio, ema_slope, ns, ne)
    wf_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_f)
    print(f"\n  Best shift WF: OOS {wf_o:+.1f}% {'PASS' if wf_p else 'FAIL'} [{wf_str}]")
    all_results["phase3_best_wf"] = {
        "oos_pnl": wf_o, "wf_pass": wf_p, "wf_folds": wf_f,
    }


# ═══════════════════════════════════════════════════════════
# PHASE 4: Fixed Grid Full Re-discovery (N-pos)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Fixed Grid Full Re-discovery → N-pos Evaluation")
print("=" * 70)

grid_patterns = {}
for pat_name, sig_bars in signal_index.items():
    sigs = [s for s in sig_bars if ns <= s < ne]
    if len(sigs) < 25:
        continue
    for direction in ['LONG', 'SHORT']:
        result = grid_search_best(
            sigs, direction, opens, highs, lows, n_bars,
            min_tr=25, max_baseline_wr=70.0,
            atr_ratio=atr_ratio,
            clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
        )
        if result is None:
            continue
        tp, sl = result['tp'], result['sl']
        wr = result.get('wr', 0)
        bwr = sl / (tp + sl) * 100
        edge = wr - bwr
        if edge < 18 or sl < 1.0:
            continue
        key = f"{pat_name}_{direction}"
        grid_patterns[key] = {
            'pattern': pat_name, 'direction': direction,
            'tp': tp, 'sl': sl,
        }

print(f"  Grid discovery: {len(grid_patterns)} patterns (vs {len(current_patterns)} current)")

if grid_patterns:
    grid_tuples = build_signal_tuples(grid_patterns, signal_index)
    _, grid_stats = run_npos(grid_tuples, opens, highs, lows, closes, n_bars,
                             atr_ratio, ema_slope, ns, ne)
    grid_wf, grid_oos, grid_pass = run_wf(grid_tuples, opens, highs, lows, closes,
                                            n_bars, atr_ratio, ema_slope, ns, ne)

    print(f"  N-pos Fixed Grid:")
    print(f"    IS: PnL {grid_stats['pnl']:+.1f}%, WR {grid_stats['wr']:.1f}%, "
          f"MDD {grid_stats['mdd']:.2f}%, PnL/MDD {grid_stats.get('pnl_mdd', 0):.1f}x, "
          f"Trades {grid_stats['trades']}")
    wf_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in grid_wf)
    print(f"    WF: OOS {grid_oos:+.1f}% {'PASS' if grid_pass else 'FAIL'} [{wf_str}]")

    all_results["phase4_grid"] = {
        "n_patterns": len(grid_patterns),
        "pnl": grid_stats['pnl'], "wr": grid_stats['wr'],
        "mdd": grid_stats['mdd'], "pnl_mdd": grid_stats.get('pnl_mdd', 0),
        "trades": grid_stats['trades'],
        "oos_pnl": grid_oos, "wf_pass": grid_pass, "wf_folds": grid_wf,
    }


# ═══════════════════════════════════════════════════════════
# PHASE 5: Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: SUMMARY & RECOMMENDATION")
print("=" * 70)

print(f"\n  {'Method':<30s} {'IS PnL':>8s} {'IS MDD':>8s} {'PnL/MDD':>8s} {'OOS PnL':>8s} {'WF':>5s}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

# Baseline
b = all_results["phase1_baseline"]
print(f"  {'Current MAE/MFE':<30s} {b['pnl']:>+8.1f} {b['mdd']:>8.2f} {b['pnl_mdd']:>8.1f} "
      f"{b['oos_pnl']:>+8.1f} {'PASS' if b['wf_pass'] else 'FAIL':>5s}")

# Re-optimized
r = all_results["phase2_reoptimized"]
print(f"  {'Per-pattern re-opt':<30s} {r['pnl']:>+8.1f} {r['mdd']:>8.2f} {r['pnl_mdd']:>8.1f} "
      f"{r['oos_pnl']:>+8.1f} {'PASS' if r['wf_pass'] else 'FAIL':>5s}")

# Best shift
s = all_results["phase3_shift_sweep"]["best"]
bwf = all_results.get("phase3_best_wf", {})
bwf_pnl = bwf.get("oos_pnl", 0)
bwf_pass = bwf.get("wf_pass", False)
label = f"Shift TP×{s['tp_mult']:.1f} SL×{s['sl_mult']:.1f}"
print(f"  {label:<30s} {s['pnl']:>+8.1f} {s['mdd']:>8.2f} {s['pnl_mdd']:>8.1f} "
      f"{bwf_pnl:>+8.1f} {'PASS' if bwf_pass else 'FAIL':>5s}")

# Grid
if 'phase4_grid' in all_results:
    g = all_results["phase4_grid"]
    print(f"  {'Fixed Grid rediscovery':<30s} {g['pnl']:>+8.1f} {g['mdd']:>8.2f} {g['pnl_mdd']:>8.1f} "
          f"{g['oos_pnl']:>+8.1f} {'PASS' if g['wf_pass'] else 'FAIL':>5s}")

# Verdict
methods = [
    ("Current MAE/MFE", b['pnl_mdd'], b['oos_pnl'], b['wf_pass']),
    ("Per-pattern re-opt", r['pnl_mdd'], r['oos_pnl'], r['wf_pass']),
    (f"Shift TP×{s['tp_mult']:.1f} SL×{s['sl_mult']:.1f}", s['pnl_mdd'], bwf_pnl, bwf_pass),
]
if 'phase4_grid' in all_results:
    g = all_results["phase4_grid"]
    methods.append(("Fixed Grid", g['pnl_mdd'], g['oos_pnl'], g['wf_pass']))

# Find best by OOS PnL (with WF pass preference)
passing = [m for m in methods if m[3]]
if passing:
    best_method = max(passing, key=lambda x: x[2])
else:
    best_method = max(methods, key=lambda x: x[2])

print(f"\n  ★ BEST METHOD: {best_method[0]}")
print(f"    PnL/MDD {best_method[1]:.1f}x, OOS {best_method[2]:+.1f}%, WF {'PASS' if best_method[3] else 'FAIL'}")

if best_method[0] == "Current MAE/MFE":
    verdict = "KEEP_CURRENT — Current methodology is portfolio-optimal"
else:
    verdict = f"INVESTIGATE — {best_method[0]} may improve portfolio performance"

print(f"    → {verdict}")
all_results["verdict"] = verdict
all_results["best_method"] = best_method[0]

elapsed = time.time() - start_time
all_results["elapsed_seconds"] = elapsed
print(f"\n  Elapsed: {elapsed:.1f}s")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Results saved to {OUTPUT_FILE}")
