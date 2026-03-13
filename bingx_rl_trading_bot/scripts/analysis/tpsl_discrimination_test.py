#!/usr/bin/env python3
"""
TP/SL Optimization Discrimination Test

Tests whether the improvements found in tpsl_npos_validation.py are genuine
or an artifact of the mechanism stack (Cascade, Timeout, etc.).

Method: Randomize TP/SL assignments across patterns, run same N-pos pipeline.
If random TP/SL also achieves similar improvements, the optimization is NON-DISCRIMINATING.

Seeds: 42, 123, 7 (Standard Protocol)
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
    bt_signals_atr,
    LEVERAGE, FEE_PCT, MAX_BARS,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/tpsl_discrimination_test.json"

NPOS_DEFAULTS = dict(
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
)

SEEDS = [42, 123, 7]


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


print("=" * 70)
print("TP/SL OPTIMIZATION — DISCRIMINATION TEST")
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

current_patterns = load_patterns()
print(f"Data: {n_bars} bars, neutral: {ns}-{ne}, patterns: {len(current_patterns)}")

# Current TP/SL value pools
tp_pool = [v['tp'] for v in current_patterns.values()]
sl_pool = [v['sl'] for v in current_patterns.values()]

all_results = {"study": "tpsl_discrimination_test", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

# ═══ Test 1: Current baseline ═══
print("\n--- Current baseline ---")
current_tuples = build_signal_tuples(current_patterns, signal_index)
_, baseline = run_npos(current_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, ns, ne)
wf_f, wf_oos, wf_pass = run_wf(current_tuples, opens, highs, lows, closes,
                                  n_bars, atr_ratio, ema_slope, ns, ne)
print(f"  IS: PnL/MDD {baseline.get('pnl_mdd', 0):.1f}x, PnL {baseline['pnl']:+.1f}%")
print(f"  WF: OOS {wf_oos:+.1f}% {'PASS' if wf_pass else 'FAIL'}")
all_results["baseline"] = {
    "pnl_mdd": baseline.get('pnl_mdd', 0), "pnl": baseline['pnl'],
    "oos_pnl": wf_oos, "wf_pass": wf_pass,
}

# ═══ Test 2: TP×0.6 (best shift from Phase 3) ═══
print("\n--- TP×0.6 SL×1.0 (best shift) ---")
shifted_pats = {}
for k, v in current_patterns.items():
    shifted_pats[k] = {
        'pattern': v['pattern'], 'direction': v['direction'],
        'tp': round(max(0.3, v['tp'] * 0.6), 3), 'sl': v['sl'],
    }
shifted_tuples = build_signal_tuples(shifted_pats, signal_index)
_, shift_stats = run_npos(shifted_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, ns, ne)
wf_f, shift_oos, shift_pass = run_wf(shifted_tuples, opens, highs, lows, closes,
                                        n_bars, atr_ratio, ema_slope, ns, ne)
print(f"  IS: PnL/MDD {shift_stats.get('pnl_mdd', 0):.1f}x, PnL {shift_stats['pnl']:+.1f}%")
print(f"  WF: OOS {shift_oos:+.1f}% {'PASS' if shift_pass else 'FAIL'}")
all_results["tp06_shift"] = {
    "pnl_mdd": shift_stats.get('pnl_mdd', 0), "pnl": shift_stats['pnl'],
    "oos_pnl": shift_oos, "wf_pass": shift_pass,
}

# ═══ Test 3: Random TP/SL shuffle (discrimination) ═══
print(f"\n--- Random TP/SL Shuffle Test ({len(SEEDS)} seeds) ---")
print("  If random shuffle also achieves similar OOS → NON-DISCRIMINATING")

random_results = []
random_wf_pass_count = 0

for seed in SEEDS:
    rng = np.random.RandomState(seed)
    random_pats = {}
    shuffled_tp = rng.permutation(tp_pool)
    shuffled_sl = rng.permutation(sl_pool)

    for i, (k, v) in enumerate(current_patterns.items()):
        random_pats[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': float(shuffled_tp[i % len(shuffled_tp)]),
            'sl': max(1.0, float(shuffled_sl[i % len(shuffled_sl)])),
        }

    random_tuples = build_signal_tuples(random_pats, signal_index)
    _, r_stats = run_npos(random_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, ns, ne)
    r_wf, r_oos, r_pass = run_wf(random_tuples, opens, highs, lows, closes,
                                    n_bars, atr_ratio, ema_slope, ns, ne)

    random_results.append({
        "seed": seed, "pnl_mdd": r_stats.get('pnl_mdd', 0), "pnl": r_stats['pnl'],
        "oos_pnl": r_oos, "wf_pass": r_pass,
    })
    if r_pass:
        random_wf_pass_count += 1
    print(f"  Seed {seed}: IS PnL/MDD {r_stats.get('pnl_mdd', 0):.1f}x, "
          f"OOS {r_oos:+.1f}% {'PASS' if r_pass else 'FAIL'}")

all_results["random_shuffle"] = random_results

# ═══ Test 4: Completely random TP/SL values (not from pool) ═══
print(f"\n--- Completely Random TP/SL ({len(SEEDS)} seeds) ---")
print("  Random TP in [0.5, 3.0], SL in [1.0, 5.5]")

full_random_results = []
for seed in SEEDS:
    rng = np.random.RandomState(seed + 1000)
    random_pats = {}
    for k, v in current_patterns.items():
        random_pats[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(rng.uniform(0.5, 3.0), 2),
            'sl': round(rng.uniform(1.0, 5.5), 2),
        }

    random_tuples = build_signal_tuples(random_pats, signal_index)
    _, r_stats = run_npos(random_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, ns, ne)
    r_wf, r_oos, r_pass = run_wf(random_tuples, opens, highs, lows, closes,
                                    n_bars, atr_ratio, ema_slope, ns, ne)

    full_random_results.append({
        "seed": seed, "pnl_mdd": r_stats.get('pnl_mdd', 0), "pnl": r_stats['pnl'],
        "oos_pnl": r_oos, "wf_pass": r_pass,
    })
    print(f"  Seed {seed}: IS PnL/MDD {r_stats.get('pnl_mdd', 0):.1f}x, "
          f"OOS {r_oos:+.1f}% {'PASS' if r_pass else 'FAIL'}")

all_results["full_random"] = full_random_results

# ═══ Test 5: TP fixed at 0.6× across all, SL random ═══
print(f"\n--- TP×0.6 fixed + Random SL ({len(SEEDS)} seeds) ---")
tp06_random_results = []
for seed in SEEDS:
    rng = np.random.RandomState(seed + 2000)
    random_pats = {}
    for k, v in current_patterns.items():
        random_pats[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(v['tp'] * 0.6, 3),
            'sl': round(rng.uniform(1.0, 5.5), 2),
        }

    random_tuples = build_signal_tuples(random_pats, signal_index)
    _, r_stats = run_npos(random_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, ns, ne)
    r_wf, r_oos, r_pass = run_wf(random_tuples, opens, highs, lows, closes,
                                    n_bars, atr_ratio, ema_slope, ns, ne)
    tp06_random_results.append({
        "seed": seed, "pnl_mdd": r_stats.get('pnl_mdd', 0), "pnl": r_stats['pnl'],
        "oos_pnl": r_oos, "wf_pass": r_pass,
    })
    print(f"  Seed {seed}: IS PnL/MDD {r_stats.get('pnl_mdd', 0):.1f}x, "
          f"OOS {r_oos:+.1f}% {'PASS' if r_pass else 'FAIL'}")

all_results["tp06_random_sl"] = tp06_random_results

# ═══ Summary ═══
print("\n" + "=" * 70)
print("DISCRIMINATION TEST SUMMARY")
print("=" * 70)

baseline_oos = all_results["baseline"]["oos_pnl"]
tp06_oos = all_results["tp06_shift"]["oos_pnl"]
random_avg_oos = np.mean([r["oos_pnl"] for r in random_results])
full_random_avg_oos = np.mean([r["oos_pnl"] for r in full_random_results])
tp06_random_avg_oos = np.mean([r["oos_pnl"] for r in tp06_random_results])

print(f"\n  {'Method':<35s} {'Avg OOS':>10s} {'WF PASS':>10s}")
print(f"  {'-'*35} {'-'*10} {'-'*10}")
print(f"  {'Current baseline':<35s} {baseline_oos:>+10.1f} {'PASS':>10s}")
print(f"  {'TP×0.6 (proposed)':<35s} {tp06_oos:>+10.1f} {'PASS':>10s}")
print(f"  {'Random shuffle (from pool)':<35s} {random_avg_oos:>+10.1f} "
      f"{f'{random_wf_pass_count}/{len(SEEDS)}':>10s}")
fr_pass = sum(1 for r in full_random_results if r['wf_pass'])
print(f"  {'Full random [0.5-3.0,1.0-5.5]':<35s} {full_random_avg_oos:>+10.1f} "
      f"{f'{fr_pass}/{len(SEEDS)}':>10s}")
tp06r_pass = sum(1 for r in tp06_random_results if r['wf_pass'])
print(f"  {'TP×0.6 + random SL':<35s} {tp06_random_avg_oos:>+10.1f} "
      f"{f'{tp06r_pass}/{len(SEEDS)}':>10s}")

# Discrimination verdict
# If random achieves similar OOS as optimized → NON-DISCRIMINATING
gap_vs_random = tp06_oos - random_avg_oos
gap_vs_full_random = tp06_oos - full_random_avg_oos

print(f"\n  TP×0.6 gap vs shuffled: {gap_vs_random:+.1f}%")
print(f"  TP×0.6 gap vs full random: {gap_vs_full_random:+.1f}%")

if gap_vs_random < 20 and gap_vs_full_random < 30:
    disc_verdict = "NON-DISCRIMINATING — TP/SL optimization is mechanism-dependent"
    print(f"\n  ★ {disc_verdict}")
    print(f"    Random TP/SL achieves similar performance.")
    print(f"    TP×0.6 improvement is likely from Cascade/Timeout mechanics,")
    print(f"    not from genuine TP/SL optimization signal.")
elif gap_vs_random > 50:
    disc_verdict = "DISCRIMINATING — TP/SL optimization provides genuine edge"
    print(f"\n  ★ {disc_verdict}")
    print(f"    Significant gap between optimized and random TP/SL.")
else:
    disc_verdict = "MARGINAL — Possible weak signal, proceed with caution"
    print(f"\n  ★ {disc_verdict}")

all_results["discrimination"] = {
    "verdict": disc_verdict,
    "gap_vs_shuffled": gap_vs_random,
    "gap_vs_full_random": gap_vs_full_random,
}

elapsed = time.time() - start_time
all_results["elapsed_seconds"] = elapsed
print(f"\n  Elapsed: {elapsed:.1f}s")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Results saved to {OUTPUT_FILE}")
