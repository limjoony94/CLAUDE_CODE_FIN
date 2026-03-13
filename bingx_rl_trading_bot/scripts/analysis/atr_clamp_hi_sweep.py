#!/usr/bin/env python3
"""
ATR clamp_hi Sweep — Cluster Damage Mitigation Study

Motivation:
  - Current clamp_hi=1.5 → max SL ~4.31% → worst cluster slot loss -13% (x3 lev)
  - Live: 11 SL cluster events in 18 days, 75% of all SLs, -43.88% portfolio damage
  - 31% of cluster SLs have SL > 4%, producing worst damage
  - Previous v1.50.0 change: clamp_hi 1.7→1.5 improved IS PnL/MDD +52%

Sweep: clamp_hi in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5(baseline)]
Fixed: clamp_lo=0.5, tp_scale_factor=0.72, ATR a14/w576

Standard Research Protocol: compound, 0.10% fee, 0.02% slippage, 3x leverage.
N-pos: 9 slots, dir_cap 7, agg_risk 8/15, cascade 85%, momentum 1.5%/15min/1h, timeout 288.
WF: 3-fold expanding window.
MC: 3 seeds (42, 123, 7), max p < 0.01.
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
    portfolio_npos, calc_stats_compound, mc_test,
    compute_atr_ratio, compute_ema_slope,
    LEVERAGE, FEE_PCT, MAX_BARS, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT, DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/atr_clamp_hi_sweep.json"

TP_SCALE_FACTOR = 0.72
CLAMP_LO = 0.5  # fixed
SWEEP_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def make_npos_kwargs(clamp_hi):
    return dict(
        n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
        regime_mult=DEFAULT_REGIME_MULT,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        clamp_lo=CLAMP_LO, clamp_hi=clamp_hi,
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


def apply_tp_scale(patterns, tp_scale=TP_SCALE_FACTOR):
    """Apply TP scale factor (min floor 0.3%)."""
    result = {}
    for k, v in patterns.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(max(0.3, v['tp'] * tp_scale), 3),
            'sl': v['sl'],
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
             atr_ratio, ema_slope, start_bar, end_bar, clamp_hi):
    kwargs = make_npos_kwargs(clamp_hi)
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
           atr_ratio, ema_slope, ns, ne, clamp_hi, n_folds=3):
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
                            atr_ratio, ema_slope, test_start, test_end, clamp_hi)
        results.append({'fold': fold + 1, 'pnl': stats.get('pnl', 0),
                        'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0),
                        'mdd': stats.get('mdd', 0)})
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


def compute_sl_distribution(patterns, atr_ratio, signal_index, clamp_hi):
    """Compute effective SL distribution across all pattern signals.

    Returns stats about the actual SL distances after ATR scaling + clamping.
    Also counts how many patterns have their SL clamped by clamp_hi.
    """
    all_eff_sls = []
    patterns_clamped = set()
    pattern_max_sls = {}

    for k, v in patterns.items():
        pat_name = v.get('pattern') or k.rsplit('_', 1)[0]
        if pat_name not in signal_index:
            continue
        base_sl = v['sl']
        bars = signal_index[pat_name]
        was_clamped = False
        sls_for_pattern = []

        for bar in bars:
            if bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                r = max(CLAMP_LO, min(clamp_hi, atr_ratio[bar]))
                # Check if clamp_hi is the binding constraint
                if atr_ratio[bar] > clamp_hi:
                    was_clamped = True
            else:
                r = 1.0
            # Daily loss cap (production parity)
            if base_sl > 0:
                r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / base_sl)
            eff_sl = max(0.1, base_sl * r - SLIPPAGE_BUFFER)
            sls_for_pattern.append(eff_sl)
            all_eff_sls.append(eff_sl)

        if was_clamped:
            patterns_clamped.add(k)
        if sls_for_pattern:
            pattern_max_sls[k] = max(sls_for_pattern)

    if not all_eff_sls:
        return {}

    arr = np.array(all_eff_sls)
    n_gt3 = sum(1 for k, mx in pattern_max_sls.items() if mx > 3.0)
    n_gt4 = sum(1 for k, mx in pattern_max_sls.items() if mx > 4.0)

    return {
        'max_sl_pct': round(float(np.max(arr)), 3),
        'mean_sl_pct': round(float(np.mean(arr)), 3),
        'median_sl_pct': round(float(np.median(arr)), 3),
        'p95_sl_pct': round(float(np.percentile(arr, 95)), 3),
        'patterns_sl_gt3': n_gt3,
        'patterns_sl_gt4': n_gt4,
        'affected_patterns': len(patterns_clamped),
        'total_patterns': len(patterns),
    }


def compute_rr_ratio(trades):
    """Compute average win / average loss ratio."""
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [abs(t['pnl_slot']) for t in trades if t['pnl_slot'] < 0]
    if not wins or not losses:
        return 0.0
    return round(np.mean(wins) / np.mean(losses), 3)


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("ATR CLAMP_HI SWEEP — CLUSTER DAMAGE MITIGATION STUDY")
print("=" * 70)

df = load_and_classify(DATA_FILE)
opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
n_bars = len(df)
type_codes = df['candle_type'].values

atr_ratio = compute_atr_ratio(highs, lows, closes,
                                atr_period=DEFAULT_ATR_PERIOD,
                                window=DEFAULT_ATR_WINDOW)
ema_slope = compute_ema_slope(closes)

signal_index = build_signal_index(type_codes, n_bars)
nw = find_neutral_window(closes)
ns, ne = nw
n_days = (ne - ns) / 288
print(f"Data: {n_bars} bars, neutral window: {ns}-{ne} ({n_days:.0f}d)")

# Load patterns and apply TP scale
raw_patterns = load_patterns()
patterns = apply_tp_scale(raw_patterns, TP_SCALE_FACTOR)
signal_tuples = build_signal_tuples(patterns, signal_index)
print(f"Patterns: {len(patterns)} (TP x{TP_SCALE_FACTOR})")
print(f"Signal tuples: {len(signal_tuples)}")

# ═══════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"SWEEP: clamp_hi in {SWEEP_VALUES}")
print(f"Fixed: clamp_lo={CLAMP_LO}, tp_scale={TP_SCALE_FACTOR}")
print("=" * 70)

sweep_results = []

for ch in SWEEP_VALUES:
    t0 = time.time()
    is_baseline = (ch == 1.5)
    marker = " (BASELINE)" if is_baseline else ""
    print(f"\n--- clamp_hi = {ch}{marker} ---")

    # IS run
    trades, stats = run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                             atr_ratio, ema_slope, ns, ne, clamp_hi=ch)

    # R:R ratio
    rr = compute_rr_ratio(trades)

    # SL distribution
    sl_dist = compute_sl_distribution(patterns, atr_ratio, signal_index, clamp_hi=ch)

    # Trades per day
    tpd = round(stats['trades'] / n_days, 2) if n_days > 0 else 0

    # WF
    wf_folds, wf_oos, wf_pass = run_wf(signal_tuples, opens, highs, lows, closes,
                                         n_bars, atr_ratio, ema_slope, ns, ne,
                                         clamp_hi=ch)
    wf_verdict = "PASS" if wf_pass else "FAIL"

    # MC test on IS trades
    is_pnls = [t['pnl_portfolio'] for t in trades]
    mc_p = mc_test(is_pnls) if len(is_pnls) >= 25 else 1.0

    elapsed = time.time() - t0

    result = {
        'clamp_hi': ch,
        'is_wr': stats.get('wr', 0),
        'is_pnl': stats.get('pnl', 0),
        'is_mdd': stats.get('mdd', 0),
        'is_pnl_mdd': stats.get('pnl_mdd', 0),
        'trades': stats.get('trades', 0),
        'trades_per_day': tpd,
        'max_sl_pct': sl_dist.get('max_sl_pct', 0),
        'mean_sl_pct': sl_dist.get('mean_sl_pct', 0),
        'median_sl_pct': sl_dist.get('median_sl_pct', 0),
        'p95_sl_pct': sl_dist.get('p95_sl_pct', 0),
        'patterns_sl_gt3': sl_dist.get('patterns_sl_gt3', 0),
        'patterns_sl_gt4': sl_dist.get('patterns_sl_gt4', 0),
        'rr_ratio': rr,
        'max_corr_loss': stats.get('max_corr_loss', 0),
        'corr_events': stats.get('corr_events', 0),
        'affected_patterns': sl_dist.get('affected_patterns', 0),
        'mc_p': mc_p,
        'wf_oos_folds': wf_folds,
        'wf_oos_total': round(wf_oos, 2),
        'wf_verdict': wf_verdict,
        'blocked': stats.get('blocked', {}),
    }
    sweep_results.append(result)

    # Print summary
    wf_detail = ', '.join(f"F{f['fold']}:{f['pnl']:+.1f}%" for f in wf_folds)
    print(f"  IS: WR={stats['wr']:.1f}%, PnL={stats['pnl']:+.1f}%, "
          f"MDD={stats['mdd']:.2f}%, PnL/MDD={stats.get('pnl_mdd',0):.1f}")
    print(f"  Trades: {stats['trades']} ({tpd}/day), R:R={rr:.3f}")
    print(f"  SL: max={sl_dist.get('max_sl_pct',0):.2f}%, mean={sl_dist.get('mean_sl_pct',0):.2f}%, "
          f"p95={sl_dist.get('p95_sl_pct',0):.2f}%")
    print(f"  Patterns: SL>3%={sl_dist.get('patterns_sl_gt3',0)}, "
          f"SL>4%={sl_dist.get('patterns_sl_gt4',0)}, "
          f"clamped={sl_dist.get('affected_patterns',0)}/{len(patterns)}")
    print(f"  Cluster: max_corr_loss={stats.get('max_corr_loss',0):.2f}%, "
          f"events={stats.get('corr_events',0)}")
    print(f"  WF {wf_verdict}: {wf_detail} (total={wf_oos:+.1f}%)")
    print(f"  MC p={mc_p:.4f}, time={elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)
print(f"{'clamp_hi':>8} {'WR':>6} {'PnL':>8} {'MDD':>6} {'P/M':>7} {'Trd':>5} "
      f"{'maxSL':>6} {'meanSL':>7} {'>3%':>4} {'>4%':>4} {'R:R':>5} "
      f"{'CorrL':>6} {'ClmpN':>5} {'WF':>4} {'OOS':>8}")
print("-" * 105)

baseline_idx = None
for i, r in enumerate(sweep_results):
    if r['clamp_hi'] == 1.5:
        baseline_idx = i
    tag = " *" if r['clamp_hi'] == 1.5 else ""
    print(f"{r['clamp_hi']:>8.1f} {r['is_wr']:>5.1f}% {r['is_pnl']:>+7.1f}% "
          f"{r['is_mdd']:>5.2f}% {r['is_pnl_mdd']:>6.1f} {r['trades']:>5} "
          f"{r['max_sl_pct']:>5.2f}% {r['mean_sl_pct']:>6.2f}% "
          f"{r['patterns_sl_gt3']:>4} {r['patterns_sl_gt4']:>4} {r['rr_ratio']:>5.3f} "
          f"{r['max_corr_loss']:>5.2f}% {r['affected_patterns']:>5} "
          f"{r['wf_verdict']:>4} {r['wf_oos_total']:>+7.1f}%{tag}")

# ═══════════════════════════════════════════════════════════
# DELTA vs BASELINE
# ═══════════════════════════════════════════════════════════
if baseline_idx is not None:
    bl = sweep_results[baseline_idx]
    print("\n" + "=" * 70)
    print("DELTA vs BASELINE (clamp_hi=1.5)")
    print("=" * 70)
    print(f"{'clamp_hi':>8} {'dPnL':>8} {'dMDD':>7} {'dP/M':>7} {'dTrd':>5} "
          f"{'dMaxSL':>7} {'dCorrL':>7} {'dOOS':>8} {'WF':>4}")
    print("-" * 75)
    for r in sweep_results:
        if r['clamp_hi'] == 1.5:
            continue
        d_pnl = r['is_pnl'] - bl['is_pnl']
        d_mdd = r['is_mdd'] - bl['is_mdd']
        d_pm = r['is_pnl_mdd'] - bl['is_pnl_mdd']
        d_trd = r['trades'] - bl['trades']
        d_maxsl = r['max_sl_pct'] - bl['max_sl_pct']
        d_corr = r['max_corr_loss'] - bl['max_corr_loss']
        d_oos = r['wf_oos_total'] - bl['wf_oos_total']
        print(f"{r['clamp_hi']:>8.1f} {d_pnl:>+7.1f}% {d_mdd:>+6.2f}% "
              f"{d_pm:>+6.1f} {d_trd:>+5} {d_maxsl:>+6.2f}% "
              f"{d_corr:>+6.2f}% {d_oos:>+7.1f}% {r['wf_verdict']:>4}")


# ═══════════════════════════════════════════════════════════
# RECOMMENDATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Find best by PnL/MDD among WF-passing candidates
passing = [r for r in sweep_results if r['wf_verdict'] == 'PASS' and r['mc_p'] < 0.01]
if passing:
    best = max(passing, key=lambda r: r['is_pnl_mdd'])
    bl = sweep_results[baseline_idx] if baseline_idx is not None else None

    if bl and best['clamp_hi'] != bl['clamp_hi']:
        pnl_delta = best['is_pnl'] - bl['is_pnl']
        mdd_delta = best['is_mdd'] - bl['is_mdd']
        corr_delta = best['max_corr_loss'] - bl['max_corr_loss']
        oos_delta = best['wf_oos_total'] - bl['wf_oos_total']
        maxsl_delta = best['max_sl_pct'] - bl['max_sl_pct']
        recommendation = (
            f"GO clamp_hi={best['clamp_hi']}: "
            f"PnL/MDD {best['is_pnl_mdd']:.1f} (vs baseline {bl['is_pnl_mdd']:.1f}), "
            f"max_corr_loss {best['max_corr_loss']:.2f}% (delta {corr_delta:+.2f}%), "
            f"max_SL {best['max_sl_pct']:.2f}% (delta {maxsl_delta:+.2f}%), "
            f"OOS {best['wf_oos_total']:+.1f}% (delta {oos_delta:+.1f}%), "
            f"WF 3/3 PASS, MC p={best['mc_p']:.4f}"
        )
    else:
        recommendation = (
            f"KEEP BASELINE clamp_hi=1.5: "
            f"PnL/MDD {bl['is_pnl_mdd']:.1f}, "
            f"max_corr_loss {bl['max_corr_loss']:.2f}%, "
            f"OOS {bl['wf_oos_total']:+.1f}%, WF 3/3 PASS"
        )
else:
    recommendation = "NO CANDIDATE PASSES (all WF FAIL or MC>0.01)"

print(recommendation)


# ═══════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════
output = {
    "study": "atr_clamp_hi_sweep",
    "date": "2026-03-13",
    "baseline_clamp_hi": 1.5,
    "tp_scale_factor": TP_SCALE_FACTOR,
    "clamp_lo": CLAMP_LO,
    "atr_period": DEFAULT_ATR_PERIOD,
    "atr_window": DEFAULT_ATR_WINDOW,
    "data_bars": n_bars,
    "neutral_window": [int(ns), int(ne)],
    "neutral_days": round(n_days, 1),
    "n_patterns": len(patterns),
    "n_signal_tuples": len(signal_tuples),
    "npos_config": {
        "n_slots": DEFAULT_N_SLOTS,
        "direction_cap": DEFAULT_DIRECTION_CAP,
        "regime_mult": DEFAULT_REGIME_MULT,
        "agg_risk_counter": DEFAULT_AGG_RISK_COUNTER,
        "agg_risk_with": DEFAULT_AGG_RISK_WITH,
        "momentum_lookback": DEFAULT_MOMENTUM_LOOKBACK,
        "momentum_threshold": DEFAULT_MOMENTUM_THRESHOLD,
        "momentum_cooldown": DEFAULT_MOMENTUM_COOLDOWN,
        "cascade_tighten_pct": DEFAULT_CASCADE_TIGHTEN_PCT,
        "timeout_bars": TIMEOUT_BARS,
    },
    "sweep_results": sweep_results,
    "recommendation": recommendation,
    "elapsed_seconds": round(time.time() - start_time, 1),
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUTPUT_FILE}")
print(f"Total time: {time.time() - start_time:.1f}s")
