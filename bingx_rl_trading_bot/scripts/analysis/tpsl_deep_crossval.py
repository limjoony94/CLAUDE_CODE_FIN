#!/usr/bin/env python3
"""
TP/SL Deep Cross-Validation — Final Decision Study

Prior findings:
  - TP×0.6 improves IS PnL/MDD 88.4→104.0x, OOS +242→+276%
  - DISCRIMINATING (not mechanism artifact)
  - Per-pattern re-opt also improves (99.6x)

This study answers remaining questions:
  1. Fine-grained TP multiplier sweep (0.4-1.0, step 0.05) — exact optimal
  2. Joint TP×SL 2D sweep with finer resolution
  3. Multi-seed MC robustness (10 seeds for IS + OOS stability)
  4. Cascade interaction test — does TP×0.6 improvement survive Cascade OFF?
  5. Per-fold stability — is improvement consistent across all WF folds?
  6. Direction analysis — does TP×0.6 help LONG, SHORT, or both?
  7. Pattern-level analysis — which patterns benefit most from TP reduction?
  8. Comparison: TP×0.6 uniform vs per-pattern re-opt → which is more robust?
  9. Final recommendation with confidence interval

Standard Research Protocol: compound, 0.10% fee, 0.02% slippage, 3x leverage.
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
OUTPUT_FILE = "results/tpsl_deep_crossval.json"

NPOS_DEFAULTS = dict(
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
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


def apply_tp_sl_mult(patterns, tp_mult=1.0, sl_mult=1.0, direction_filter=None):
    """Apply uniform TP/SL multiplier, optionally only to one direction."""
    result = {}
    for k, v in patterns.items():
        if direction_filter and v['direction'] != direction_filter:
            result[k] = dict(v)
        else:
            result[k] = {
                'pattern': v['pattern'], 'direction': v['direction'],
                'tp': round(max(0.3, v['tp'] * tp_mult), 3),
                'sl': round(max(1.0, v['sl'] * sl_mult), 3),
            }
    return result


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
                        'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0),
                        'mdd': stats.get('mdd', 0)})
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


def fmt_wf(wf_folds):
    return ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_folds)


# ─── Load data ───
print("=" * 70)
print("TP/SL DEEP CROSS-VALIDATION — FINAL DECISION STUDY")
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
print(f"Data: {n_bars} bars, neutral: {ns}-{ne} ({(ne-ns)//288:.0f}d)")

current_patterns = load_patterns()
print(f"Patterns: {len(current_patterns)}")

all_results = {
    "study": "tpsl_deep_crossval",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}


# ═══════════════════════════════════════════════════════════
# PHASE 1: Fine-grained TP multiplier sweep
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: Fine TP Multiplier Sweep (0.40-1.00, step 0.05)")
print("=" * 70)

tp_sweep = {}
tp_mults = np.arange(0.40, 1.05, 0.05)

for tp_m in tp_mults:
    tp_m = round(tp_m, 2)
    pats = apply_tp_sl_mult(current_patterns, tp_mult=tp_m)
    tuples = build_signal_tuples(pats, signal_index)
    _, stats = run_npos(tuples, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne)
    wf_f, wf_oos, wf_pass = run_wf(tuples, opens, highs, lows, closes,
                                      n_bars, atr_ratio, ema_slope, ns, ne)
    tp_sweep[tp_m] = {
        "pnl": stats['pnl'], "wr": stats['wr'], "mdd": stats['mdd'],
        "pnl_mdd": stats.get('pnl_mdd', 0), "trades": stats['trades'],
        "oos_pnl": wf_oos, "wf_pass": wf_pass, "wf_folds": wf_f,
    }
    marker = " ←CURRENT" if tp_m == 1.0 else ""
    wf_tag = "PASS" if wf_pass else "FAIL"
    print(f"  TP×{tp_m:.2f}: IS PnL {stats['pnl']:+.1f}%, PnL/MDD {stats.get('pnl_mdd',0):.1f}x, "
          f"OOS {wf_oos:+.1f}% {wf_tag} [{fmt_wf(wf_f)}]{marker}")

# Find optimum
best_tp = max(tp_sweep.items(), key=lambda x: x[1]['pnl_mdd'])
best_oos_tp = max(tp_sweep.items(), key=lambda x: x[1]['oos_pnl'])
print(f"\n  Best IS PnL/MDD: TP×{best_tp[0]:.2f} ({best_tp[1]['pnl_mdd']:.1f}x)")
print(f"  Best OOS PnL:    TP×{best_oos_tp[0]:.2f} ({best_oos_tp[1]['oos_pnl']:+.1f}%)")

all_results["phase1_tp_sweep"] = {
    "sweep": {str(k): v for k, v in tp_sweep.items()},
    "best_is_tp_mult": best_tp[0],
    "best_oos_tp_mult": best_oos_tp[0],
}


# ═══════════════════════════════════════════════════════════
# PHASE 2: Joint TP×SL 2D sweep (finer, around optimal)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Joint TP×SL 2D Sweep (finer resolution)")
print("=" * 70)

tp_range_2d = np.arange(0.50, 0.85, 0.05)
sl_range_2d = np.arange(0.8, 1.6, 0.1)

joint_sweep = {}
for tp_m in tp_range_2d:
    for sl_m in sl_range_2d:
        tp_m_r = round(tp_m, 2)
        sl_m_r = round(sl_m, 1)
        pats = apply_tp_sl_mult(current_patterns, tp_mult=tp_m_r, sl_mult=sl_m_r)
        tuples = build_signal_tuples(pats, signal_index)
        _, stats = run_npos(tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, ns, ne)
        key = f"tp{tp_m_r:.2f}_sl{sl_m_r:.1f}"
        joint_sweep[key] = {
            "tp_mult": tp_m_r, "sl_mult": sl_m_r,
            "pnl": stats['pnl'], "mdd": stats['mdd'],
            "pnl_mdd": stats.get('pnl_mdd', 0), "trades": stats['trades'],
        }

best_joint = max(joint_sweep.values(), key=lambda x: x['pnl_mdd'])
print(f"  Best joint: TP×{best_joint['tp_mult']:.2f} SL×{best_joint['sl_mult']:.1f} "
      f"→ PnL/MDD {best_joint['pnl_mdd']:.1f}x, PnL {best_joint['pnl']:+.1f}%")

# Show top-5
sorted_joint = sorted(joint_sweep.values(), key=lambda x: -x['pnl_mdd'])
print(f"\n  Top-5 joint combos (by PnL/MDD):")
for i, j in enumerate(sorted_joint[:5]):
    print(f"    {i+1}. TP×{j['tp_mult']:.2f} SL×{j['sl_mult']:.1f}: "
          f"PnL {j['pnl']:+.1f}%, MDD {j['mdd']:.2f}%, PnL/MDD {j['pnl_mdd']:.1f}x")

# WF validate best joint
best_tp_m = best_joint['tp_mult']
best_sl_m = best_joint['sl_mult']
pats = apply_tp_sl_mult(current_patterns, tp_mult=best_tp_m, sl_mult=best_sl_m)
tuples = build_signal_tuples(pats, signal_index)
wf_f, wf_oos, wf_pass = run_wf(tuples, opens, highs, lows, closes,
                                  n_bars, atr_ratio, ema_slope, ns, ne)
print(f"\n  Best joint WF: OOS {wf_oos:+.1f}% {'PASS' if wf_pass else 'FAIL'} [{fmt_wf(wf_f)}]")

all_results["phase2_joint_sweep"] = {
    "best": best_joint,
    "best_wf": {"oos_pnl": wf_oos, "wf_pass": wf_pass, "wf_folds": wf_f},
    "top5": sorted_joint[:5],
}


# ═══════════════════════════════════════════════════════════
# PHASE 3: Multi-seed MC Robustness (10 seeds)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Multi-Seed MC Robustness — TP×0.6 vs Current (10 seeds)")
print("=" * 70)

SEEDS = [42, 123, 7, 999, 314, 271, 628, 137, 515, 808]

# We test if TP×0.6 consistently beats current by bootstrapping signal subsets
mc_results = {"current": [], "tp06": []}

for seed in SEEDS:
    rng = np.random.RandomState(seed)
    # Bootstrap: sample 80% of patterns with replacement
    n_sample = int(len(current_patterns) * 0.8)
    keys = list(current_patterns.keys())
    sampled_keys = rng.choice(keys, size=n_sample, replace=True)

    for label, tp_m in [("current", 1.0), ("tp06", 0.6)]:
        sampled_pats = {}
        for k in sampled_keys:
            v = current_patterns[k]
            sampled_pats[k] = {
                'pattern': v['pattern'], 'direction': v['direction'],
                'tp': round(max(0.3, v['tp'] * tp_m), 3), 'sl': v['sl'],
            }

        tuples = build_signal_tuples(sampled_pats, signal_index)
        _, stats = run_npos(tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, ns, ne)
        wf_f, wf_oos, wf_pass = run_wf(tuples, opens, highs, lows, closes,
                                          n_bars, atr_ratio, ema_slope, ns, ne)
        mc_results[label].append({
            "seed": seed, "pnl_mdd": stats.get('pnl_mdd', 0),
            "pnl": stats['pnl'], "oos_pnl": wf_oos, "wf_pass": wf_pass,
        })

current_oos = [r['oos_pnl'] for r in mc_results['current']]
tp06_oos = [r['oos_pnl'] for r in mc_results['tp06']]
current_pm = [r['pnl_mdd'] for r in mc_results['current']]
tp06_pm = [r['pnl_mdd'] for r in mc_results['tp06']]

print(f"  {'Metric':<20s} {'Current':>12s} {'TP×0.6':>12s} {'Delta':>12s}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'OOS PnL (mean)':<20s} {np.mean(current_oos):>+12.1f} {np.mean(tp06_oos):>+12.1f} "
      f"{np.mean(tp06_oos)-np.mean(current_oos):>+12.1f}")
print(f"  {'OOS PnL (median)':<20s} {np.median(current_oos):>+12.1f} {np.median(tp06_oos):>+12.1f} "
      f"{np.median(tp06_oos)-np.median(current_oos):>+12.1f}")
print(f"  {'OOS PnL (min)':<20s} {min(current_oos):>+12.1f} {min(tp06_oos):>+12.1f}")
print(f"  {'OOS PnL (std)':<20s} {np.std(current_oos):>12.1f} {np.std(tp06_oos):>12.1f}")
print(f"  {'IS PnL/MDD (mean)':<20s} {np.mean(current_pm):>12.1f} {np.mean(tp06_pm):>12.1f}")

# Win count: how many seeds does TP×0.6 beat current?
wins = sum(1 for a, b in zip(tp06_oos, current_oos) if a > b)
print(f"\n  TP×0.6 wins: {wins}/{len(SEEDS)} ({wins/len(SEEDS)*100:.0f}%)")

# WF pass count
current_wf_pass = sum(1 for r in mc_results['current'] if r['wf_pass'])
tp06_wf_pass = sum(1 for r in mc_results['tp06'] if r['wf_pass'])
print(f"  WF PASS: Current {current_wf_pass}/{len(SEEDS)}, TP×0.6 {tp06_wf_pass}/{len(SEEDS)}")

all_results["phase3_mc"] = {
    "n_seeds": len(SEEDS),
    "current_oos_mean": float(np.mean(current_oos)),
    "tp06_oos_mean": float(np.mean(tp06_oos)),
    "tp06_win_rate": wins / len(SEEDS),
    "current_wf_pass_rate": current_wf_pass / len(SEEDS),
    "tp06_wf_pass_rate": tp06_wf_pass / len(SEEDS),
    "current_oos_std": float(np.std(current_oos)),
    "tp06_oos_std": float(np.std(tp06_oos)),
}


# ═══════════════════════════════════════════════════════════
# PHASE 4: Cascade Interaction — does improvement survive Cascade OFF?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Cascade SL Interaction Test")
print("=" * 70)

for cascade_pct, label in [(DEFAULT_CASCADE_TIGHTEN_PCT, "Cascade ON (85%)"),
                            (0, "Cascade OFF")]:
    for tp_m, tp_label in [(1.0, "Current"), (0.6, "TP×0.6")]:
        pats = apply_tp_sl_mult(current_patterns, tp_mult=tp_m)
        tuples = build_signal_tuples(pats, signal_index)
        _, stats = run_npos(tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, ns, ne,
                            cascade_tighten_pct=cascade_pct)
        wf_f, wf_oos, wf_pass = run_wf(tuples, opens, highs, lows, closes,
                                          n_bars, atr_ratio, ema_slope, ns, ne,
                                          cascade_tighten_pct=cascade_pct)
        print(f"  {label:20s} + {tp_label:10s}: IS PnL/MDD {stats.get('pnl_mdd',0):6.1f}x, "
              f"OOS {wf_oos:+.1f}% {'PASS' if wf_pass else 'FAIL'}")

        rkey = f"{label}_{tp_label}".replace(" ", "_").replace("(", "").replace(")", "")
        all_results.setdefault("phase4_cascade", {})[rkey] = {
            "pnl_mdd": stats.get('pnl_mdd', 0), "pnl": stats['pnl'],
            "oos_pnl": wf_oos, "wf_pass": wf_pass,
        }

# Check: does TP×0.6 gap persist without Cascade?
p4 = all_results["phase4_cascade"]
cascade_on_gap = (p4.get("Cascade_ON_85%_TP×0.6", {}).get("oos_pnl", 0) -
                  p4.get("Cascade_ON_85%_Current", {}).get("oos_pnl", 0))
cascade_off_gap = (p4.get("Cascade_OFF_TP×0.6", {}).get("oos_pnl", 0) -
                   p4.get("Cascade_OFF_Current", {}).get("oos_pnl", 0))
print(f"\n  TP×0.6 gap with Cascade ON:  {cascade_on_gap:+.1f}%")
print(f"  TP×0.6 gap with Cascade OFF: {cascade_off_gap:+.1f}%")
if cascade_off_gap > 10:
    print(f"  → TP×0.6 improvement is INDEPENDENT of Cascade")
else:
    print(f"  → TP×0.6 improvement is Cascade-DEPENDENT")


# ═══════════════════════════════════════════════════════════
# PHASE 5: Per-fold Stability
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: Per-Fold Consistency (5-fold expanded)")
print("=" * 70)

for tp_m, label in [(1.0, "Current"), (0.6, "TP×0.6"), (best_tp[0], f"TP×{best_tp[0]:.2f}")]:
    pats = apply_tp_sl_mult(current_patterns, tp_mult=tp_m)
    tuples = build_signal_tuples(pats, signal_index)
    wf_f, wf_oos, wf_pass = run_wf(tuples, opens, highs, lows, closes,
                                      n_bars, atr_ratio, ema_slope, ns, ne, n_folds=5)
    fold_pnls = [r['pnl'] for r in wf_f]
    min_fold = min(fold_pnls) if fold_pnls else 0
    consistency = sum(1 for p in fold_pnls if p > 0) / len(fold_pnls) * 100 if fold_pnls else 0
    print(f"  {label:12s}: OOS {wf_oos:+.1f}%, min fold {min_fold:+.1f}%, "
          f"consistency {consistency:.0f}% [{fmt_wf(wf_f)}]")
    all_results.setdefault("phase5_folds", {})[label] = {
        "oos_pnl": wf_oos, "wf_folds": wf_f, "min_fold": min_fold,
        "consistency": consistency,
    }


# ═══════════════════════════════════════════════════════════
# PHASE 6: Direction Analysis — LONG vs SHORT
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: Direction Analysis — LONG/SHORT Separate Impact")
print("=" * 70)

for direction in ['LONG', 'SHORT', None]:
    d_label = direction or "ALL"
    for tp_m, tp_label in [(1.0, "Current"), (0.6, "TP×0.6")]:
        pats = apply_tp_sl_mult(current_patterns, tp_mult=tp_m,
                                direction_filter=direction if direction else None)
        # If direction filter, only count affected patterns
        if direction:
            n_affected = sum(1 for v in current_patterns.values() if v['direction'] == direction)
        else:
            n_affected = len(current_patterns)

        tuples = build_signal_tuples(pats, signal_index)
        _, stats = run_npos(tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, ns, ne)
        wf_f, wf_oos, wf_pass = run_wf(tuples, opens, highs, lows, closes,
                                          n_bars, atr_ratio, ema_slope, ns, ne)
        print(f"  {d_label:5s} {tp_label:10s} ({n_affected:3d}p): "
              f"IS PnL/MDD {stats.get('pnl_mdd',0):6.1f}x, OOS {wf_oos:+.1f}% "
              f"{'PASS' if wf_pass else 'FAIL'}")
        rkey = f"{d_label}_{tp_label}"
        all_results.setdefault("phase6_direction", {})[rkey] = {
            "n_patterns": n_affected,
            "pnl_mdd": stats.get('pnl_mdd', 0), "oos_pnl": wf_oos, "wf_pass": wf_pass,
        }


# ═══════════════════════════════════════════════════════════
# PHASE 7: Pattern-level Impact — Which benefit most?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 7: Pattern-Level — Top benefiting patterns from TP reduction")
print("=" * 70)

pattern_impact = []
fee = FEE_PCT * LEVERAGE

for key, info in current_patterns.items():
    pat_name = info['pattern']
    direction = info['direction']
    sig_bars = signal_index.get(pat_name, [])
    sig_bars = [s for s in sig_bars if ns <= s < ne]
    if len(sig_bars) < 25:
        continue

    # Current TP
    trades_current = bt_signals_atr(
        sig_bars, direction, info['tp'], info['sl'],
        opens, highs, lows, n_bars,
        atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI
    )
    # TP×0.6
    trades_tp06 = bt_signals_atr(
        sig_bars, direction, info['tp'] * 0.6, info['sl'],
        opens, highs, lows, n_bars,
        atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI
    )

    if len(trades_current) < 20 or len(trades_tp06) < 20:
        continue

    def simple_pnl_mdd(trades):
        pnls = [t[2] for t in trades]
        cum = 0; peak = 0; mdd = 0
        for p in pnls:
            cum += p
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > mdd: mdd = dd
        return cum, mdd

    pnl_c, mdd_c = simple_pnl_mdd(trades_current)
    pnl_t, mdd_t = simple_pnl_mdd(trades_tp06)
    pm_c = pnl_c / mdd_c if mdd_c > 0 else pnl_c
    pm_t = pnl_t / mdd_t if mdd_t > 0 else pnl_t

    wr_c = sum(1 for t in trades_current if t[2] > 0) / len(trades_current) * 100
    wr_t = sum(1 for t in trades_tp06 if t[2] > 0) / len(trades_tp06) * 100

    pattern_impact.append({
        'key': key, 'direction': direction,
        'current_tp': info['tp'], 'current_sl': info['sl'],
        'pnl_mdd_current': round(pm_c, 1), 'pnl_mdd_tp06': round(pm_t, 1),
        'wr_current': round(wr_c, 1), 'wr_tp06': round(wr_t, 1),
        'delta_pnl_mdd': round(pm_t - pm_c, 1),
        'delta_wr': round(wr_t - wr_c, 1),
        'trades': len(trades_current),
    })

# Sort by impact
pattern_impact.sort(key=lambda x: -x['delta_pnl_mdd'])
improved = [p for p in pattern_impact if p['delta_pnl_mdd'] > 0]
degraded = [p for p in pattern_impact if p['delta_pnl_mdd'] < 0]
neutral = [p for p in pattern_impact if p['delta_pnl_mdd'] == 0]

print(f"  Analyzed: {len(pattern_impact)} patterns")
print(f"    Improved by TP×0.6: {len(improved)} ({len(improved)/len(pattern_impact)*100:.0f}%)")
print(f"    Degraded:           {len(degraded)} ({len(degraded)/len(pattern_impact)*100:.0f}%)")
print(f"    Neutral:            {len(neutral)}")
print(f"    Avg WR change:      {np.mean([p['delta_wr'] for p in pattern_impact]):+.1f}pp")

if improved:
    print(f"\n  Top-5 most improved patterns:")
    for p in improved[:5]:
        print(f"    {p['key']:25s}: PnL/MDD {p['pnl_mdd_current']:6.1f}→{p['pnl_mdd_tp06']:6.1f} "
              f"(Δ{p['delta_pnl_mdd']:+.1f}), WR {p['wr_current']:.0f}→{p['wr_tp06']:.0f}%, "
              f"TP {p['current_tp']:.2f}→{p['current_tp']*0.6:.2f}")

if degraded:
    print(f"\n  Top-5 most degraded patterns:")
    for p in sorted(degraded, key=lambda x: x['delta_pnl_mdd'])[:5]:
        print(f"    {p['key']:25s}: PnL/MDD {p['pnl_mdd_current']:6.1f}→{p['pnl_mdd_tp06']:6.1f} "
              f"(Δ{p['delta_pnl_mdd']:+.1f}), WR {p['wr_current']:.0f}→{p['wr_tp06']:.0f}%, "
              f"TP {p['current_tp']:.2f}→{p['current_tp']*0.6:.2f}")

all_results["phase7_pattern_impact"] = {
    "n_analyzed": len(pattern_impact),
    "improved_count": len(improved),
    "degraded_count": len(degraded),
    "improved_pct": len(improved) / len(pattern_impact) * 100 if pattern_impact else 0,
    "avg_wr_change": float(np.mean([p['delta_wr'] for p in pattern_impact])) if pattern_impact else 0,
    "top5_improved": improved[:5],
    "top5_degraded": sorted(degraded, key=lambda x: x['delta_pnl_mdd'])[:5] if degraded else [],
}


# ═══════════════════════════════════════════════════════════
# PHASE 8: Uniform TP×0.6 vs Per-Pattern Re-opt Robustness
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 8: Uniform TP×0.6 vs Per-Pattern Re-opt — Robustness Comparison")
print("=" * 70)

# Per-pattern re-opt (from previous study: use 2D grid to find per-pattern optimal)
tp_grid = np.arange(0.5, 3.0, 0.2)
sl_grid = np.arange(1.0, 5.5, 0.3)

reopt_patterns = {}
for key, info in current_patterns.items():
    pat_name = info['pattern']
    direction = info['direction']
    sig_bars = signal_index.get(pat_name, [])
    sig_bars = [s for s in sig_bars if ns <= s < ne]
    if len(sig_bars) < 25:
        reopt_patterns[key] = dict(info)
        continue

    best_score = -1
    best_tp_sl = (info['tp'], info['sl'])

    for tp in tp_grid:
        for sl in sl_grid:
            trades = bt_signals_atr(
                sig_bars, direction, tp, sl,
                opens, highs, lows, n_bars,
                atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI
            )
            if len(trades) < 20:
                continue
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
            if (wr - bwr) < 18 or sl < 1.0:
                continue
            score = (cum / mdd_val) if mdd_val > 0 else cum
            if score > best_score:
                best_score = score
                best_tp_sl = (round(tp, 2), round(sl, 2))

    reopt_patterns[key] = {
        'pattern': pat_name, 'direction': direction,
        'tp': best_tp_sl[0], 'sl': best_tp_sl[1],
    }

# Evaluate both at 5-fold WF
for label, pats in [("Uniform TP×0.6", apply_tp_sl_mult(current_patterns, tp_mult=0.6)),
                    ("Per-pattern re-opt", reopt_patterns),
                    ("Current", current_patterns)]:
    tuples = build_signal_tuples(pats, signal_index)
    _, is_stats = run_npos(tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne)
    wf_f, wf_oos, wf_pass = run_wf(tuples, opens, highs, lows, closes,
                                      n_bars, atr_ratio, ema_slope, ns, ne, n_folds=5)
    fold_pnls = [r['pnl'] for r in wf_f]
    min_fold = min(fold_pnls) if fold_pnls else 0
    print(f"  {label:25s}: IS PnL/MDD {is_stats.get('pnl_mdd',0):6.1f}x, "
          f"OOS {wf_oos:+.1f}%, min_fold {min_fold:+.1f}%, "
          f"{'PASS' if wf_pass else 'FAIL'}")
    all_results.setdefault("phase8_comparison", {})[label] = {
        "pnl_mdd": is_stats.get('pnl_mdd', 0), "oos_pnl": wf_oos,
        "min_fold": min_fold, "wf_pass": wf_pass, "wf_folds": wf_f,
    }


# ═══════════════════════════════════════════════════════════
# PHASE 9: Final Recommendation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 9: FINAL RECOMMENDATION")
print("=" * 70)

# Gather all evidence
p1 = all_results["phase1_tp_sweep"]
p2 = all_results["phase2_joint_sweep"]
p3 = all_results["phase3_mc"]
p4 = all_results["phase4_cascade"]
p5 = all_results["phase5_folds"]
p7 = all_results["phase7_pattern_impact"]
p8 = all_results["phase8_comparison"]

print(f"\n  Evidence Summary:")
print(f"    Phase 1: Best IS TP mult = {p1['best_is_tp_mult']:.2f}, "
      f"Best OOS TP mult = {p1['best_oos_tp_mult']:.2f}")
print(f"    Phase 2: Best joint = TP×{p2['best']['tp_mult']:.2f} SL×{p2['best']['sl_mult']:.1f}")
print(f"    Phase 3: TP×0.6 wins {p3['tp06_win_rate']*100:.0f}% of {p3['n_seeds']} bootstrap seeds")
print(f"    Phase 4: Cascade-independent = {cascade_off_gap > 10}")
print(f"    Phase 7: {p7['improved_pct']:.0f}% patterns improve, avg WR {p7['avg_wr_change']:+.1f}pp")

# Decision criteria:
# 1. IS PnL/MDD improvement > 10%
# 2. OOS improvement > 0
# 3. WF PASS maintained
# 4. MC win rate > 60%
# 5. Cascade-independent
# 6. Per-fold consistency maintained
# 7. Majority of patterns benefit

optimal_tp = p1['best_is_tp_mult']
current_ref = tp_sweep[1.0]
optimal_ref = tp_sweep[optimal_tp]

is_improvement = (optimal_ref['pnl_mdd'] / max(current_ref.get('pnl_mdd', 1), 1) - 1) * 100
oos_improvement = optimal_ref['oos_pnl'] - current_ref['oos_pnl']
mc_win_rate = p3['tp06_win_rate'] * 100
cascade_independent = cascade_off_gap > 10
pattern_benefit_pct = p7['improved_pct']

print(f"\n  Decision Criteria:")
print(f"    IS PnL/MDD improvement: {is_improvement:+.1f}% {'✓' if is_improvement > 10 else '✗'}")
print(f"    OOS improvement: {oos_improvement:+.1f}% {'✓' if oos_improvement > 0 else '✗'}")
print(f"    WF maintained: {'✓' if optimal_ref['wf_pass'] else '✗'}")
print(f"    MC win rate: {mc_win_rate:.0f}% {'✓' if mc_win_rate > 60 else '✗'}")
print(f"    Cascade-independent: {'✓' if cascade_independent else '✗'}")
print(f"    Pattern benefit: {pattern_benefit_pct:.0f}% {'✓' if pattern_benefit_pct > 50 else '✗'}")

criteria_met = sum([
    is_improvement > 10,
    oos_improvement > 0,
    optimal_ref['wf_pass'],
    mc_win_rate > 60,
    cascade_independent,
    pattern_benefit_pct > 50,
])

print(f"\n  Criteria met: {criteria_met}/6")

if criteria_met >= 5:
    verdict = f"GO — Apply TP×{optimal_tp:.2f} (uniform scaling)"
    print(f"\n  ★ {verdict}")
    print(f"    Recommended: Multiply all pattern TP values by {optimal_tp:.2f}")
    print(f"    Expected improvement: IS PnL/MDD {is_improvement:+.1f}%, OOS {oos_improvement:+.1f}%")

    # Compare uniform vs per-pattern
    uniform_oos = p8.get("Uniform TP×0.6", {}).get("oos_pnl", 0)
    reopt_oos = p8.get("Per-pattern re-opt", {}).get("oos_pnl", 0)
    if reopt_oos > uniform_oos * 1.05:
        print(f"    ALSO: Per-pattern re-opt may further improve (OOS {reopt_oos:+.1f}% vs {uniform_oos:+.1f}%)")
        print(f"    → Consider re-running scanner after uniform TP scaling")
elif criteria_met >= 3:
    verdict = f"CAUTIOUS_GO — Apply TP×{optimal_tp:.2f} with monitoring"
    print(f"\n  ★ {verdict}")
    print(f"    Some criteria not met — apply with close monitoring")
else:
    verdict = "KEEP_CURRENT — Insufficient evidence for change"
    print(f"\n  ★ {verdict}")

all_results["verdict"] = verdict
all_results["optimal_tp_mult"] = optimal_tp
all_results["criteria_met"] = criteria_met

elapsed = time.time() - start_time
all_results["elapsed_seconds"] = elapsed
print(f"\n  Elapsed: {elapsed:.1f}s")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Results saved to {OUTPUT_FILE}")
