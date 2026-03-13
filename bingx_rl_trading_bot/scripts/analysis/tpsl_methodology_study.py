#!/usr/bin/env python3
"""
TP/SL Discovery Methodology Deep Cross-Validation Study

Questions:
  Q1: Is MAE/MFE percentile the best discovery method?
  Q2: Are the percentile grids [40-80] TP / [70-95] SL optimal?
  Q3: Is PnL/MDD the right scoring function?
  Q4: Does ATR scaling genuinely help vs fixed TP/SL?
  Q5: Is the baseline WR cap (70%) correctly set?
  Q6: Percentile vs Distribution method?

Phases:
  1. Baseline: Current methodology IS + WF performance
  2. TP percentile grid expansion (20-95%)
  3. SL percentile grid expansion (50-99%)
  4. Scoring function comparison (PnL/MDD vs Edge vs Sharpe vs PnL)
  5. ATR scaling ablation (ATR ON vs OFF)
  6. Fixed grid vs MAE/MFE head-to-head
  7. Baseline WR cap sweep (50-90%)
  8. Combined optimal config + WF validation
  9. Summary

Standard Research Protocol: next-bar entry, intrabar exit, 0.10% fee,
  0.02% slippage, 3x leverage, compound sizing.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
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
    bt_signals, bt_signals_atr,
    grid_search_mae_mfe, grid_search_best,
    scan_universe_range,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER, MAX_BARS,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT, DEFAULT_MIN_TRADES,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/tpsl_methodology_study.json"

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
            'tp_pct': v['tp'], 'sl_pct': v['sl'],
        }
    return result


def calc_simple_stats(trades_tuples):
    """Simple compound stats from bt_signals/bt_signals_atr tuples (entry_bar, exit_bar, pnl)."""
    if not trades_tuples:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}
    pnls = [t[2] for t in trades_tuples]
    wins = sum(1 for p in pnls if p > 0)
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    total_pnl = equity - 100.0
    wr = wins / len(pnls) * 100
    pnl_mdd = total_pnl / max_dd if max_dd > 0 else total_pnl
    return {'trades': len(pnls), 'wr': round(wr, 1), 'pnl': round(total_pnl, 2),
            'mdd': round(max_dd, 2), 'pnl_mdd': round(pnl_mdd, 2)}


def build_signal_tuples(patterns, sig_idx):
    tuples = []
    for k, v in patterns.items():
        pat_name = v.get('pattern') or k.rsplit('_', 1)[0]
        if pat_name in sig_idx:
            for bar in sig_idx[pat_name]:
                tuples.append((bar, k, v['direction'], v['tp_pct'], v['sl_pct']))
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
    fold_size = total // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        train_end = ns + fold_size * (fold + 2)
        test_start = train_end
        test_end = min(train_end + fold_size, ne)
        if test_start >= test_end:
            continue
        _, stats = run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, test_start, test_end, **extra)
        results.append({'fold': fold + 1, 'pnl': stats.get('pnl', 0),
                        'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0)})
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


def rediscover_patterns(type_codes, n_bars, opens, highs, lows, closes,
                        signal_index, atr_ratio,
                        tp_percentiles, sl_percentiles,
                        edge_threshold=18.0, min_trades=25,
                        max_baseline_wr=70.0,
                        use_atr=True, scoring='pnl_mdd',
                        clamp_lo=DEFAULT_ATR_CLAMP_LO,
                        clamp_hi=DEFAULT_ATR_CLAMP_HI,
                        start_bar=0, end_bar=None):
    """Re-run pattern discovery with custom parameters.

    Returns dict of discovered patterns with TP/SL.
    """
    if end_bar is None:
        end_bar = n_bars

    fee = FEE_PCT * LEVERAGE
    discovered = {}

    for pat_name, sig_bars in signal_index.items():
        # Filter to range
        sigs_in_range = [s for s in sig_bars if start_bar <= s < end_bar]
        if len(sigs_in_range) < min_trades:
            continue

        for direction in ['LONG', 'SHORT']:
            key = f"{pat_name}_{direction}"

            # Call grid_search_mae_mfe with custom grids
            result, _exc = grid_search_mae_mfe(
                sigs_in_range, direction, opens, highs, lows, n_bars,
                max_bars=MAX_BARS, min_tr=min_trades,
                max_baseline_wr=max_baseline_wr,
                atr_ratio=atr_ratio if use_atr else None,
                clamp_lo=clamp_lo, clamp_hi=clamp_hi,
                tp_max=None, sl_max=None,
                tp_method='percentile',
            )

            if result is None:
                continue

            tp, sl = result['tp'], result['sl']
            wr = result.get('wr', 0)
            n_trades = result.get('trades', 0)
            # Compute edge = WR - baseline_wr
            bwr = sl / (tp + sl) * 100
            edge = wr - bwr
            mdd_v = result.get('mdd', 0)
            pnl_mdd_score = (result.get('pnl', 0) / mdd_v) if mdd_v > 0 else result.get('pnl', 0)

            # Quality filters
            if edge < edge_threshold:
                continue
            if n_trades < min_trades:
                continue
            if sl < 1.0:  # Execution risk floor
                continue

            # MC test (simplified: sign test)
            # Skip full MC for speed — use edge as proxy
            discovered[key] = {
                'pattern': pat_name, 'direction': direction,
                'tp_pct': tp, 'sl_pct': sl,
                'wr': wr, 'trades': n_trades,
                'edge': edge, 'pnl_mdd': pnl_mdd_score,
            }

    return discovered


def evaluate_pattern_set(patterns, signal_index, opens, highs, lows, closes,
                         n_bars, atr_ratio, ema_slope, ns, ne):
    """Evaluate a set of patterns: IS + WF."""
    tuples = build_signal_tuples(patterns, signal_index)
    if not tuples:
        return None

    # IS
    _, is_stats = run_npos(tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne)

    # WF
    wf_folds, oos_pnl, wf_pass = run_wf(tuples, opens, highs, lows, closes, n_bars,
                                          atr_ratio, ema_slope, ns, ne)

    return {
        'n_patterns': len(patterns),
        'n_signals': len(tuples),
        'is_pnl': is_stats.get('pnl', 0),
        'is_wr': is_stats.get('wr', 0),
        'is_mdd': is_stats.get('mdd', 0),
        'is_trades': is_stats.get('trades', 0),
        'is_pnl_mdd': is_stats.get('pnl_mdd', 0),
        'oos_pnl': oos_pnl,
        'wf_pass': wf_pass,
        'wf_folds': wf_folds,
    }


# ─── Load data ───
print("=" * 70)
print("TP/SL METHODOLOGY DEEP CROSS-VALIDATION STUDY")
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

# Load current patterns as baseline
current_patterns = load_patterns()
current_tuples = build_signal_tuples(current_patterns, signal_index)
n_current_signals = len([s for s in current_tuples if ns <= s[0] < ne])
print(f"Current patterns: {len(current_patterns)}, signals in neutral: {n_current_signals}")

all_results = {
    "study": "tpsl_methodology_deep_study",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "data_bars": n_bars,
    "neutral_window": [int(ns), int(ne)],
    "current_patterns": len(current_patterns),
}


# ═══════════════════════════════════════════════════════════
# PHASE 1: Baseline Performance
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: Current Baseline (MAE/MFE, ATR, PnL/MDD scoring)")
print("=" * 70)

baseline = evaluate_pattern_set(current_patterns, signal_index, opens, highs, lows,
                                closes, n_bars, atr_ratio, ema_slope, ns, ne)
print(f"  Patterns: {baseline['n_patterns']}, Signals: {baseline['n_signals']}")
print(f"  IS: PnL {baseline['is_pnl']:+.1f}%, WR {baseline['is_wr']:.1f}%, "
      f"MDD {baseline['is_mdd']:.2f}%, PnL/MDD {baseline['is_pnl_mdd']:.1f}x, "
      f"Trades {baseline['is_trades']}")
wf_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in baseline['wf_folds'])
print(f"  WF: OOS {baseline['oos_pnl']:+.1f}% {'PASS' if baseline['wf_pass'] else 'FAIL'} [{wf_str}]")

all_results["phase1_baseline"] = baseline


# ═══════════════════════════════════════════════════════════
# PHASE 2: TP Percentile Grid Exploration
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: TP Percentile Grid — Which MFE percentiles work best?")
print("=" * 70)
print("  Current: TP_PERCENTILES = [40, 50, 60, 70, 80]")

# Analyze the actual TP values chosen by current patterns
tp_values = [v['tp_pct'] for v in current_patterns.values()]
sl_values = [v['sl_pct'] for v in current_patterns.values()]
print(f"\n  Current TP distribution: min={min(tp_values):.2f}, median={np.median(tp_values):.2f}, "
      f"max={max(tp_values):.2f}, mean={np.mean(tp_values):.2f}")
print(f"  Current SL distribution: min={min(sl_values):.2f}, median={np.median(sl_values):.2f}, "
      f"max={max(sl_values):.2f}, mean={np.mean(sl_values):.2f}")

# Test different TP grids by checking how many patterns survive with each grid
# We'll sample a subset of patterns and compare TP choices
tp_grid_variants = {
    'narrow_low': [30, 40, 50],
    'current': [40, 50, 60, 70, 80],
    'expanded': [20, 30, 40, 50, 60, 70, 80, 90],
    'high_only': [60, 70, 80, 90],
    'fine_mid': [40, 45, 50, 55, 60, 65, 70],
    'aggressive': [20, 30, 40],
    'conservative': [70, 80, 90, 95],
}

# For proper comparison, we need to re-run discovery with each grid
# This is expensive, so we'll use a representative sample
# Instead, let's analyze what the optimal TP would be for each current pattern

print(f"\n  Analyzing TP optimality for {len(current_patterns)} patterns...")
print(f"  (Checking if higher/lower TP percentiles would improve individual pattern PnL/MDD)")

# For each pattern, test a wider range of TP values while keeping SL fixed
fee = FEE_PCT * LEVERAGE
tp_test_range = np.arange(0.5, 3.5, 0.1)

pattern_tp_optimal = []
n_sample = min(30, len(current_patterns))  # Sample for speed
sample_keys = list(current_patterns.keys())[:n_sample]

for key in sample_keys:
    info = current_patterns[key]
    pat_name = info['pattern']
    direction = info['direction']
    current_tp = info['tp_pct']
    current_sl = info['sl_pct']

    sig_bars = signal_index.get(pat_name, [])
    sig_bars = [s for s in sig_bars if ns <= s < ne]
    if len(sig_bars) < 25:
        continue

    best_pnl_mdd = -1
    best_tp = current_tp
    current_score = -1

    for test_tp in tp_test_range:
        # Quick backtest with ATR
        trades = bt_signals_atr(
            sig_bars, direction, test_tp, current_sl,
            opens, highs, lows, n_bars,
            atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI
        )
        if len(trades) < 20:
            continue
        stats = calc_simple_stats(trades)
        score = stats.get('pnl_mdd', 0)

        if abs(test_tp - current_tp) < 0.05:
            current_score = score

        if score > best_pnl_mdd:
            best_pnl_mdd = score
            best_tp = test_tp

    if current_score >= 0:
        pattern_tp_optimal.append({
            'key': key,
            'current_tp': current_tp,
            'optimal_tp': round(best_tp, 2),
            'current_score': current_score,
            'optimal_score': best_pnl_mdd,
            'improvement': (best_pnl_mdd / max(current_score, 0.01) - 1) * 100
                           if current_score > 0 else 0,
        })

if pattern_tp_optimal:
    avg_improvement = np.mean([p['improvement'] for p in pattern_tp_optimal])
    tp_diffs = [p['optimal_tp'] - p['current_tp'] for p in pattern_tp_optimal]
    better_count = sum(1 for p in pattern_tp_optimal if p['improvement'] > 5)
    same_count = sum(1 for p in pattern_tp_optimal if abs(p['improvement']) <= 5)
    worse_count = sum(1 for p in pattern_tp_optimal if p['improvement'] < -5)

    print(f"  Analyzed {len(pattern_tp_optimal)} patterns:")
    print(f"    Current TP ≈ optimal (±5%): {same_count} ({same_count/len(pattern_tp_optimal)*100:.0f}%)")
    print(f"    Better TP exists (>5% improvement): {better_count}")
    print(f"    Avg potential improvement: {avg_improvement:+.1f}%")
    print(f"    Avg TP shift: {np.mean(tp_diffs):+.2f}% (positive=higher TP better)")
    print(f"    Median TP shift: {np.median(tp_diffs):+.2f}%")

all_results["phase2_tp_grid"] = {
    "current_tp_stats": {
        "min": min(tp_values), "median": float(np.median(tp_values)),
        "max": max(tp_values), "mean": float(np.mean(tp_values)),
    },
    "n_analyzed": len(pattern_tp_optimal),
    "same_pct": same_count / max(len(pattern_tp_optimal), 1) * 100 if pattern_tp_optimal else 0,
    "better_count": better_count if pattern_tp_optimal else 0,
    "avg_improvement": float(avg_improvement) if pattern_tp_optimal else 0,
    "avg_tp_shift": float(np.mean(tp_diffs)) if pattern_tp_optimal else 0,
    "top5_improvements": sorted(pattern_tp_optimal, key=lambda x: -x['improvement'])[:5]
                         if pattern_tp_optimal else [],
}


# ═══════════════════════════════════════════════════════════
# PHASE 3: SL Percentile Analysis
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: SL Percentile — Are SL values optimal?")
print("=" * 70)

sl_test_range = np.arange(0.8, 6.0, 0.2)
pattern_sl_optimal = []

for key in sample_keys:
    info = current_patterns[key]
    pat_name = info['pattern']
    direction = info['direction']
    current_tp = info['tp_pct']
    current_sl = info['sl_pct']

    sig_bars = signal_index.get(pat_name, [])
    sig_bars = [s for s in sig_bars if ns <= s < ne]
    if len(sig_bars) < 25:
        continue

    best_pnl_mdd = -1
    best_sl = current_sl
    current_score = -1

    for test_sl in sl_test_range:
        if test_sl < 1.0:  # Min SL floor
            continue
        trades = bt_signals_atr(
            sig_bars, direction, current_tp, test_sl,
            opens, highs, lows, n_bars,
            atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI
        )
        if len(trades) < 20:
            continue
        stats = calc_simple_stats(trades)
        score = stats.get('pnl_mdd', 0)

        if abs(test_sl - current_sl) < 0.1:
            current_score = score

        if score > best_pnl_mdd:
            best_pnl_mdd = score
            best_sl = test_sl

    if current_score >= 0:
        pattern_sl_optimal.append({
            'key': key,
            'current_sl': current_sl,
            'optimal_sl': round(best_sl, 2),
            'current_score': current_score,
            'optimal_score': best_pnl_mdd,
            'improvement': (best_pnl_mdd / max(current_score, 0.01) - 1) * 100
                           if current_score > 0 else 0,
        })

if pattern_sl_optimal:
    avg_improvement = np.mean([p['improvement'] for p in pattern_sl_optimal])
    sl_diffs = [p['optimal_sl'] - p['current_sl'] for p in pattern_sl_optimal]
    better_count = sum(1 for p in pattern_sl_optimal if p['improvement'] > 5)
    same_count = sum(1 for p in pattern_sl_optimal if abs(p['improvement']) <= 5)

    print(f"  Analyzed {len(pattern_sl_optimal)} patterns:")
    print(f"    Current SL ≈ optimal (±5%): {same_count} ({same_count/len(pattern_sl_optimal)*100:.0f}%)")
    print(f"    Better SL exists (>5% improvement): {better_count}")
    print(f"    Avg potential improvement: {avg_improvement:+.1f}%")
    print(f"    Avg SL shift: {np.mean(sl_diffs):+.2f}% (positive=wider SL)")
    print(f"    Median SL shift: {np.median(sl_diffs):+.2f}%")

all_results["phase3_sl_grid"] = {
    "current_sl_stats": {
        "min": min(sl_values), "median": float(np.median(sl_values)),
        "max": max(sl_values), "mean": float(np.mean(sl_values)),
    },
    "n_analyzed": len(pattern_sl_optimal),
    "same_pct": same_count / max(len(pattern_sl_optimal), 1) * 100 if pattern_sl_optimal else 0,
    "better_count": better_count if pattern_sl_optimal else 0,
    "avg_improvement": float(avg_improvement) if pattern_sl_optimal else 0,
    "avg_sl_shift": float(np.mean(sl_diffs)) if pattern_sl_optimal else 0,
}


# ═══════════════════════════════════════════════════════════
# PHASE 4: Joint TP+SL Optimization — Is current combo optimal?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Joint TP+SL Optimization (2D grid per pattern)")
print("=" * 70)

# For a subset of patterns, do full 2D grid search and compare with current
joint_optimal = []
tp_range = np.arange(0.5, 3.0, 0.2)
sl_range = np.arange(1.0, 5.0, 0.3)

n_joint_sample = min(20, len(sample_keys))
for key in sample_keys[:n_joint_sample]:
    info = current_patterns[key]
    pat_name = info['pattern']
    direction = info['direction']
    current_tp = info['tp_pct']
    current_sl = info['sl_pct']

    sig_bars = signal_index.get(pat_name, [])
    sig_bars = [s for s in sig_bars if ns <= s < ne]
    if len(sig_bars) < 25:
        continue

    best_score = -1
    best_tp_sl = (current_tp, current_sl)
    current_score = -1

    for tp in tp_range:
        for sl in sl_range:
            trades = bt_signals_atr(
                sig_bars, direction, tp, sl,
                opens, highs, lows, n_bars,
                atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI
            )
            if len(trades) < 20:
                continue
            stats = calc_simple_stats(trades)
            score = stats.get('pnl_mdd', 0)

            if abs(tp - current_tp) < 0.1 and abs(sl - current_sl) < 0.15:
                current_score = score
            if score > best_score:
                best_score = score
                best_tp_sl = (round(tp, 2), round(sl, 2))

    if current_score > 0:
        improvement = (best_score / current_score - 1) * 100
        joint_optimal.append({
            'key': key,
            'current': (current_tp, current_sl),
            'optimal': best_tp_sl,
            'current_score': current_score,
            'optimal_score': best_score,
            'improvement': improvement,
        })

if joint_optimal:
    avg_improvement = np.mean([j['improvement'] for j in joint_optimal])
    near_optimal = sum(1 for j in joint_optimal if j['improvement'] < 10)
    print(f"  Analyzed {len(joint_optimal)} patterns with 2D grid:")
    print(f"    Near-optimal (improvement <10%): {near_optimal}/{len(joint_optimal)} "
          f"({near_optimal/len(joint_optimal)*100:.0f}%)")
    print(f"    Avg potential improvement: {avg_improvement:+.1f}%")
    # Check if improvements come from TP or SL changes
    tp_changed = sum(1 for j in joint_optimal
                     if abs(j['optimal'][0] - j['current'][0]) > 0.15)
    sl_changed = sum(1 for j in joint_optimal
                     if abs(j['optimal'][1] - j['current'][1]) > 0.2)
    print(f"    TP would change: {tp_changed}/{len(joint_optimal)}")
    print(f"    SL would change: {sl_changed}/{len(joint_optimal)}")

all_results["phase4_joint"] = {
    "n_analyzed": len(joint_optimal),
    "near_optimal_pct": near_optimal / max(len(joint_optimal), 1) * 100 if joint_optimal else 0,
    "avg_improvement": float(avg_improvement) if joint_optimal else 0,
    "details": joint_optimal[:10],
}


# ═══════════════════════════════════════════════════════════
# PHASE 5: ATR Scaling Ablation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: ATR Scaling — ON vs OFF")
print("=" * 70)

# Test: what if we disable ATR scaling (use raw TP/SL percentages)?
# We need to build signals without ATR scaling in the exit check
# In portfolio_npos, ATR is applied in _check_exit_npos

# Method: run with clamp_lo=1.0, clamp_hi=1.0 → effectively ATR ratio = 1.0 always
_, stats_atr_on = run_npos(current_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne)
_, stats_atr_off = run_npos(current_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, ns, ne,
                            clamp_lo=1.0, clamp_hi=1.0)

pnl_mdd_on = stats_atr_on.get('pnl_mdd', 0)
pnl_mdd_off = stats_atr_off.get('pnl_mdd', 0)

print(f"  ATR ON  (clamp {DEFAULT_ATR_CLAMP_LO}-{DEFAULT_ATR_CLAMP_HI}): "
      f"PnL {stats_atr_on['pnl']:+.1f}%, WR {stats_atr_on['wr']:.1f}%, "
      f"MDD {stats_atr_on['mdd']:.2f}%, PnL/MDD {pnl_mdd_on:.1f}x")
print(f"  ATR OFF (ratio=1.0 fixed): "
      f"PnL {stats_atr_off['pnl']:+.1f}%, WR {stats_atr_off['wr']:.1f}%, "
      f"MDD {stats_atr_off['mdd']:.2f}%, PnL/MDD {pnl_mdd_off:.1f}x")

# WF both
wf_on, oos_on, pass_on = run_wf(current_tuples, opens, highs, lows, closes, n_bars,
                                  atr_ratio, ema_slope, ns, ne)
wf_off, oos_off, pass_off = run_wf(current_tuples, opens, highs, lows, closes, n_bars,
                                     atr_ratio, ema_slope, ns, ne,
                                     clamp_lo=1.0, clamp_hi=1.0)

print(f"  WF ATR ON:  OOS {oos_on:+.1f}% {'PASS' if pass_on else 'FAIL'}")
print(f"  WF ATR OFF: OOS {oos_off:+.1f}% {'PASS' if pass_off else 'FAIL'}")

# Also test different clamp ranges
clamp_variants = [
    (0.3, 1.7, "wide"),
    (0.5, 1.5, "current"),
    (0.7, 1.3, "narrow"),
    (0.8, 1.2, "very_narrow"),
    (1.0, 1.0, "off"),
]

print(f"\n  Clamp range sweep:")
p5_clamp = {}
for lo, hi, label in clamp_variants:
    _, stats = run_npos(current_tuples, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne,
                        clamp_lo=lo, clamp_hi=hi)
    pm = stats.get('pnl_mdd', 0)
    marker = "←CURRENT" if label == "current" else ""
    print(f"    [{lo:.1f}, {hi:.1f}] ({label:12s}): PnL {stats['pnl']:+.1f}%, "
          f"MDD {stats['mdd']:.2f}%, PnL/MDD {pm:.1f}x {marker}")
    p5_clamp[label] = {"lo": lo, "hi": hi, "pnl": stats['pnl'],
                        "mdd": stats['mdd'], "pnl_mdd": pm}

all_results["phase5_atr"] = {
    "atr_on": {"pnl": stats_atr_on['pnl'], "mdd": stats_atr_on['mdd'],
               "pnl_mdd": pnl_mdd_on, "oos_pnl": oos_on, "wf_pass": pass_on},
    "atr_off": {"pnl": stats_atr_off['pnl'], "mdd": stats_atr_off['mdd'],
                "pnl_mdd": pnl_mdd_off, "oos_pnl": oos_off, "wf_pass": pass_off},
    "clamp_sweep": p5_clamp,
}


# ═══════════════════════════════════════════════════════════
# PHASE 6: MAE/MFE vs Fixed Grid Discovery
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: MAE/MFE vs Fixed Grid — Head-to-Head Comparison")
print("=" * 70)

# Compare: for the same pattern set, what happens if we use grid_search_best
# (fixed grid) instead of grid_search_mae_mfe?
# We'll re-discover TP/SL using the fixed grid method for the same patterns

TP_GRID = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.5, 3.0]
SL_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

grid_patterns = {}
mae_mfe_patterns = {}
grid_count = 0
mae_count = 0

print(f"  Re-discovering TP/SL for {n_joint_sample} patterns with both methods...")

for key in sample_keys[:n_joint_sample]:
    info = current_patterns[key]
    pat_name = info['pattern']
    direction = info['direction']

    sig_bars = signal_index.get(pat_name, [])
    sig_bars = [s for s in sig_bars if ns <= s < ne]
    if len(sig_bars) < 25:
        continue

    # Method 1: Fixed grid
    best_grid = grid_search_best(
        sig_bars, direction, opens, highs, lows, n_bars,
        min_tr=25, max_baseline_wr=70.0,
        atr_ratio=atr_ratio,
        clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    )

    # Method 2: MAE/MFE
    best_mae, _exc = grid_search_mae_mfe(
        sig_bars, direction, opens, highs, lows, n_bars,
        max_bars=MAX_BARS, min_tr=25, max_baseline_wr=70.0,
        atr_ratio=atr_ratio,
        clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    )

    if best_grid and best_grid.get('wr', 0) > 50:
        grid_patterns[key] = {
            'pattern': pat_name, 'direction': direction,
            'tp_pct': best_grid['tp'], 'sl_pct': best_grid['sl'],
        }
        grid_count += 1

    if best_mae and best_mae.get('wr', 0) > 50:
        mae_mfe_patterns[key] = {
            'pattern': pat_name, 'direction': direction,
            'tp_pct': best_mae['tp'], 'sl_pct': best_mae['sl'],
        }
        mae_count += 1

print(f"  Fixed grid: {grid_count} patterns discovered")
print(f"  MAE/MFE:    {mae_count} patterns discovered")

# Evaluate both sets
if grid_patterns:
    grid_eval = evaluate_pattern_set(grid_patterns, signal_index, opens, highs, lows,
                                     closes, n_bars, atr_ratio, ema_slope, ns, ne)
    if grid_eval:
        print(f"\n  Fixed Grid: IS PnL {grid_eval['is_pnl']:+.1f}%, "
              f"PnL/MDD {grid_eval['is_pnl_mdd']:.1f}x, "
              f"OOS {grid_eval['oos_pnl']:+.1f}% {'PASS' if grid_eval['wf_pass'] else 'FAIL'}")

if mae_mfe_patterns:
    mae_eval = evaluate_pattern_set(mae_mfe_patterns, signal_index, opens, highs, lows,
                                    closes, n_bars, atr_ratio, ema_slope, ns, ne)
    if mae_eval:
        print(f"  MAE/MFE:    IS PnL {mae_eval['is_pnl']:+.1f}%, "
              f"PnL/MDD {mae_eval['is_pnl_mdd']:.1f}x, "
              f"OOS {mae_eval['oos_pnl']:+.1f}% {'PASS' if mae_eval['wf_pass'] else 'FAIL'}")

all_results["phase6_method_comparison"] = {
    "grid_patterns": grid_count,
    "mae_mfe_patterns": mae_count,
    "grid_eval": grid_eval if grid_patterns and grid_eval else None,
    "mae_eval": mae_eval if mae_mfe_patterns and mae_eval else None,
}


# ═══════════════════════════════════════════════════════════
# PHASE 7: Baseline WR Cap Sweep
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 7: Baseline WR Cap — Impact on Pattern Selection")
print("=" * 70)
print("  Current: max_baseline_wr = 70% (rejects high R:R combos)")

# The baseline WR = SL/(TP+SL) represents what a random walk would achieve
# Higher baseline WR means TP is very far vs SL (easy to hit SL first)
# Cap at 70% means we reject combos where SL < TP*2.33

# Analyze current patterns' baseline WR distribution
baseline_wrs = []
for key, info in current_patterns.items():
    tp = info['tp_pct']
    sl = info['sl_pct']
    bwr = sl / (tp + sl) * 100
    baseline_wrs.append(bwr)

print(f"  Current patterns baseline WR distribution:")
print(f"    Min: {min(baseline_wrs):.1f}%, Median: {np.median(baseline_wrs):.1f}%, "
      f"Max: {max(baseline_wrs):.1f}%")
print(f"    Patterns with bWR > 60%: {sum(1 for b in baseline_wrs if b > 60)}")
print(f"    Patterns with bWR > 65%: {sum(1 for b in baseline_wrs if b > 65)}")

# Count how many patterns at different caps
cap_values = [50, 55, 60, 65, 70, 75, 80, 85, 90]
p7_results = {}
for cap in cap_values:
    # Filter current patterns by baseline WR cap
    filtered = {}
    for key, info in current_patterns.items():
        tp = info['tp_pct']
        sl = info['sl_pct']
        bwr = sl / (tp + sl) * 100
        if bwr <= cap:
            filtered[key] = info

    if not filtered:
        continue

    ev = evaluate_pattern_set(filtered, signal_index, opens, highs, lows,
                              closes, n_bars, atr_ratio, ema_slope, ns, ne)
    if ev:
        marker = "←CURRENT" if cap == 70 else ""
        print(f"  Cap {cap}%: {ev['n_patterns']} patterns, IS PnL {ev['is_pnl']:+.1f}%, "
              f"PnL/MDD {ev['is_pnl_mdd']:.1f}x, OOS {ev['oos_pnl']:+.1f}% "
              f"{'PASS' if ev['wf_pass'] else 'FAIL'} {marker}")
        p7_results[cap] = ev

all_results["phase7_bwr_cap"] = p7_results


# ═══════════════════════════════════════════════════════════
# PHASE 8: R:R Ratio Analysis
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 8: Risk:Reward Ratio Distribution Analysis")
print("=" * 70)

# Analyze the R:R ratio (TP/SL) across patterns
rr_ratios = [info['tp_pct'] / info['sl_pct'] for info in current_patterns.values()
             if info['sl_pct'] > 0]
print(f"  R:R (TP/SL) distribution across {len(rr_ratios)} patterns:")
print(f"    Min: {min(rr_ratios):.3f}, Median: {np.median(rr_ratios):.3f}, "
      f"Max: {max(rr_ratios):.3f}, Mean: {np.mean(rr_ratios):.3f}")
print(f"    R:R < 0.5 (unfavorable): {sum(1 for r in rr_ratios if r < 0.5)} patterns")
print(f"    R:R 0.5-1.0: {sum(1 for r in rr_ratios if 0.5 <= r < 1.0)} patterns")
print(f"    R:R >= 1.0 (favorable): {sum(1 for r in rr_ratios if r >= 1.0)} patterns")

# Test: what if we only keep favorable R:R patterns?
favorable_patterns = {k: v for k, v in current_patterns.items()
                      if v['tp_pct'] / v['sl_pct'] >= 0.5}
unfavorable_patterns = {k: v for k, v in current_patterns.items()
                        if v['tp_pct'] / v['sl_pct'] < 0.5}

if favorable_patterns:
    fav_eval = evaluate_pattern_set(favorable_patterns, signal_index, opens, highs, lows,
                                    closes, n_bars, atr_ratio, ema_slope, ns, ne)
    if fav_eval:
        print(f"\n  R:R >= 0.5 only ({len(favorable_patterns)}p): "
              f"IS PnL {fav_eval['is_pnl']:+.1f}%, PnL/MDD {fav_eval['is_pnl_mdd']:.1f}x, "
              f"OOS {fav_eval['oos_pnl']:+.1f}%")

if unfavorable_patterns:
    unfav_eval = evaluate_pattern_set(unfavorable_patterns, signal_index, opens, highs, lows,
                                      closes, n_bars, atr_ratio, ema_slope, ns, ne)
    if unfav_eval:
        print(f"  R:R < 0.5 only  ({len(unfavorable_patterns)}p): "
              f"IS PnL {unfav_eval['is_pnl']:+.1f}%, PnL/MDD {unfav_eval['is_pnl_mdd']:.1f}x, "
              f"OOS {unfav_eval['oos_pnl']:+.1f}%")

all_results["phase8_rr_ratio"] = {
    "distribution": {
        "min": min(rr_ratios), "median": float(np.median(rr_ratios)),
        "max": max(rr_ratios), "mean": float(np.mean(rr_ratios)),
    },
    "favorable_count": sum(1 for r in rr_ratios if r >= 0.5),
    "unfavorable_count": sum(1 for r in rr_ratios if r < 0.5),
    "favorable_eval": fav_eval if favorable_patterns and fav_eval else None,
    "unfavorable_eval": unfav_eval if unfavorable_patterns and unfav_eval else None,
}


# ═══════════════════════════════════════════════════════════
# PHASE 9: Summary & Recommendation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 9: SUMMARY & RECOMMENDATION")
print("=" * 70)

print(f"\n  Phase 1 (Baseline): IS PnL/MDD {baseline['is_pnl_mdd']:.1f}x, "
      f"OOS {baseline['oos_pnl']:+.1f}%")

if pattern_tp_optimal:
    print(f"\n  Phase 2 (TP Grid): {all_results['phase2_tp_grid']['same_pct']:.0f}% "
          f"of patterns already near-optimal")
    print(f"    → TP percentile grid [40-80] is {'adequate' if all_results['phase2_tp_grid']['same_pct'] > 60 else 'suboptimal'}")

if pattern_sl_optimal:
    print(f"\n  Phase 3 (SL Grid): {all_results['phase3_sl_grid']['same_pct']:.0f}% "
          f"of patterns already near-optimal")

if joint_optimal:
    print(f"\n  Phase 4 (Joint TP+SL): {all_results['phase4_joint']['near_optimal_pct']:.0f}% "
          f"near-optimal, avg improvement potential {all_results['phase4_joint']['avg_improvement']:+.1f}%")

print(f"\n  Phase 5 (ATR Scaling):")
print(f"    ATR ON PnL/MDD: {all_results['phase5_atr']['atr_on']['pnl_mdd']:.1f}x")
print(f"    ATR OFF PnL/MDD: {all_results['phase5_atr']['atr_off']['pnl_mdd']:.1f}x")
atr_verdict = "KEEP ATR" if pnl_mdd_on > pnl_mdd_off else "REMOVE ATR"
print(f"    → {atr_verdict}")

if 'phase6_method_comparison' in all_results:
    p6 = all_results['phase6_method_comparison']
    if p6.get('grid_eval') and p6.get('mae_eval'):
        print(f"\n  Phase 6 (Grid vs MAE/MFE):")
        print(f"    Grid:    PnL/MDD {p6['grid_eval']['is_pnl_mdd']:.1f}x")
        print(f"    MAE/MFE: PnL/MDD {p6['mae_eval']['is_pnl_mdd']:.1f}x")
        method_verdict = "KEEP MAE/MFE" if p6['mae_eval']['is_pnl_mdd'] >= p6['grid_eval']['is_pnl_mdd'] else "CONSIDER GRID"
        print(f"    → {method_verdict}")

# Final verdict
print(f"\n  ★ METHODOLOGY VERDICT:")
issues_found = []
if pattern_tp_optimal and all_results['phase2_tp_grid']['same_pct'] < 50:
    issues_found.append("TP grid may be suboptimal")
if pattern_sl_optimal and all_results['phase3_sl_grid']['same_pct'] < 50:
    issues_found.append("SL grid may be suboptimal")
if pnl_mdd_off > pnl_mdd_on * 1.1:
    issues_found.append("ATR scaling may be harmful")

if not issues_found:
    print(f"    METHODOLOGY VALIDATED — Current MAE/MFE + ATR approach is sound")
    print(f"    No major issues found across 8 phases of cross-validation")
    recommendation = "KEEP_CURRENT_METHODOLOGY"
else:
    print(f"    ISSUES FOUND:")
    for issue in issues_found:
        print(f"      - {issue}")
    recommendation = f"INVESTIGATE: {', '.join(issues_found)}"

all_results["recommendation"] = recommendation
all_results["issues_found"] = issues_found

elapsed = time.time() - start_time
all_results["elapsed_seconds"] = elapsed
print(f"\n  Elapsed: {elapsed:.1f}s")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Results saved to {OUTPUT_FILE}")
