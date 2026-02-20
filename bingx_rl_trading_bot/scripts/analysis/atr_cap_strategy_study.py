#!/usr/bin/env python3
"""
ATR-Scaled TP/SL Cap Strategy Deep Analysis

Questions:
1. ATR ratio 분포 특성 (선형? 정규? 편향?)
2. 캡 전략 비교: hard cap / adaptive clamp / sqrt / asymmetric / sigmoid
3. Daily loss limit (13%) 준수하면서 edge 보존하는 최적 전략
4. WF 3-fold 검증

Evaluated strategies:
A) NO_CAP: 현재 구현 (clamp 0.6-1.7, 캡 없음)
B) HARD_SL_CAP: scaled_sl = min(base_sl × mult, 4.33%)  [13%/3x]
C) ADAPTIVE_CLAMP: pattern별 clamp_hi = min(1.7, 4.33/base_sl)
D) SQRT_SCALE: ratio = sqrt(atr_ratio) — 비선형 압축
E) ASYMMETRIC: TP clamp [0.6, 1.7], SL clamp [0.8, 1.25]
F) SIGMOID: ratio = 1 + tanh(k*(raw_ratio-1)) * amplitude
G) HARD_SL_CAP_TIGHT: scaled_sl = min(base_sl × mult, 3.5%)  [tighter]
"""

import os, sys, json, time, math
import numpy as np
import pandas as pd
from collections import namedtuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# Constants
FEE_PCT = 0.10
LEVERAGE = 3
FEE = FEE_PCT * LEVERAGE  # 0.30%
MAX_BARS = 288
BARS_PER_DAY = 288
SLIPPAGE_BUFFER = 0.02
MAX_DAILY_LOSS_PCT = 13.0
MAX_SL_CAP = MAX_DAILY_LOSS_PCT / LEVERAGE  # 4.333%

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'atr_cap_strategy_study.json')

OVERLAP_TIMESTAMP = '2025-05-05 15:00:00'
_CandleRow = namedtuple('_CandleRow', ['open', 'high', 'low', 'close'])

ATR_PERIOD = 14
MEDIAN_WINDOW = 576
BASE_CLAMP_LO = 0.6
BASE_CLAMP_HI = 1.7


# ===================================================================
# DATA LOADING
# ===================================================================

def load_and_classify(data_file):
    print(f"Loading {os.path.basename(data_file)}")
    df = pd.read_csv(data_file)
    n = len(df)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows  = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)

    body_abs = np.abs(closes - opens)
    avg_body = pd.Series(body_abs).rolling(AVG_BODY_WINDOW).mean().values

    types = []
    for i in range(n):
        ab = avg_body[i] if not np.isnan(avg_body[i]) else 1.0
        types.append(classify_candle(_CandleRow(opens[i], highs[i], lows[i], closes[i]), ab).value)
    df['rctype'] = types
    print(f"  {n} bars classified")
    return df


def find_overlap_bar(df):
    if 'timestamp' in df.columns:
        mask = df['timestamp'] >= OVERLAP_TIMESTAMP
        if mask.any():
            return int(mask.values.argmax())
    return max(0, len(df) - 270 * BARS_PER_DAY)


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    details = data.get('pattern_details', {})
    return [{'pattern': v['pattern'], 'direction': v['direction'],
             'tp': v['tp'], 'sl': v['sl']}
            for v in details.values()]


# ===================================================================
# ATR COMPUTATION
# ===================================================================

def compute_atr_series(highs, lows, closes):
    """Compute ATR(14) for the full series."""
    n = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(ATR_PERIOD).mean().values
    return atr


def compute_atr_ratio_series(atr):
    """Compute ATR ratio = ATR / rolling_median(ATR, MEDIAN_WINDOW)."""
    med = pd.Series(atr).rolling(MEDIAN_WINDOW, min_periods=MEDIAN_WINDOW).median().values
    ratio = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if not np.isnan(atr[i]) and not np.isnan(med[i]) and med[i] > 0:
            ratio[i] = atr[i] / med[i]
    return ratio


# ===================================================================
# CAP STRATEGIES
# ===================================================================

def apply_strategy(raw_ratio, base_tp, base_sl, strategy_name):
    """
    Apply a capping strategy to get (scaled_tp, scaled_sl).
    Returns (tp_mult, sl_mult) — multipliers applied to base TP/SL.
    """
    if np.isnan(raw_ratio):
        return 1.0, 1.0

    if strategy_name == 'A_NO_CAP':
        # Current: clamp [0.6, 1.7], no SL cap
        m = max(BASE_CLAMP_LO, min(BASE_CLAMP_HI, raw_ratio))
        return m, m

    elif strategy_name == 'B_HARD_SL_CAP':
        # Clamp [0.6, 1.7], then cap scaled_sl ≤ 4.33%
        m = max(BASE_CLAMP_LO, min(BASE_CLAMP_HI, raw_ratio))
        tp_m = m
        sl_m = m
        if base_sl * sl_m > MAX_SL_CAP:
            sl_m = MAX_SL_CAP / base_sl
        return tp_m, sl_m

    elif strategy_name == 'C_ADAPTIVE_CLAMP':
        # Per-pattern clamp_hi = min(1.7, 4.33/base_sl)
        adaptive_hi = min(BASE_CLAMP_HI, MAX_SL_CAP / base_sl)
        m = max(BASE_CLAMP_LO, min(adaptive_hi, raw_ratio))
        return m, m

    elif strategy_name == 'D_SQRT_SCALE':
        # Non-linear: sqrt compression (keeps direction, reduces magnitude)
        # For ratio > 1: sqrt grows slower. For ratio < 1: sqrt shrinks slower.
        clamped = max(BASE_CLAMP_LO, min(BASE_CLAMP_HI, raw_ratio))
        if clamped >= 1.0:
            m = 1.0 + math.sqrt(clamped - 1.0)  # e.g. 1.7 → 1.837 — still too high
        else:
            m = 1.0 - math.sqrt(1.0 - clamped)  # e.g. 0.6 → 0.368
        # Apply same SL cap safety
        tp_m = m
        sl_m = m
        if base_sl * sl_m > MAX_SL_CAP:
            sl_m = MAX_SL_CAP / base_sl
        return tp_m, sl_m

    elif strategy_name == 'E_ASYMMETRIC':
        # TP: full range [0.6, 1.7], SL: conservative [0.8, 1.25]
        tp_m = max(BASE_CLAMP_LO, min(BASE_CLAMP_HI, raw_ratio))
        sl_m = max(0.8, min(1.25, raw_ratio))
        return tp_m, sl_m

    elif strategy_name == 'F_SIGMOID':
        # Sigmoid-like: 1 + tanh(k * (ratio-1)) * amplitude
        # Smooth, bounded, centered at 1.0
        k = 2.0  # steepness
        amplitude = 0.4  # max deviation from 1.0
        m = 1.0 + math.tanh(k * (raw_ratio - 1.0)) * amplitude
        # Clamp to safety bounds
        m = max(0.6, min(1.4, m))
        tp_m = m
        sl_m = m
        if base_sl * sl_m > MAX_SL_CAP:
            sl_m = MAX_SL_CAP / base_sl
        return tp_m, sl_m

    elif strategy_name == 'G_HARD_SL_CAP_TIGHT':
        # Tighter: cap scaled_sl ≤ 3.5%
        tight_cap = 3.5
        m = max(BASE_CLAMP_LO, min(BASE_CLAMP_HI, raw_ratio))
        tp_m = m
        sl_m = m
        if base_sl * sl_m > tight_cap:
            sl_m = tight_cap / base_sl
        return tp_m, sl_m

    return 1.0, 1.0


STRATEGIES = [
    'A_NO_CAP',
    'B_HARD_SL_CAP',
    'C_ADAPTIVE_CLAMP',
    'D_SQRT_SCALE',
    'E_ASYMMETRIC',
    'F_SIGMOID',
    'G_HARD_SL_CAP_TIGHT',
]


# ===================================================================
# BACKTEST ENGINE
# ===================================================================

def bt_signals(df, patterns, atr_ratio_arr, strategy_name, start_idx, end_idx):
    """Backtest with ATR-scaled TP/SL using a specific cap strategy."""
    opens = df['open'].values
    highs = df['high'].values
    lows  = df['low'].values
    closes = df['close'].values
    types = df['rctype'].values
    n = len(df)

    pat_map = {}
    for p in patterns:
        key = (p['pattern'], p['direction'])
        pat_map[key] = p

    trades = []
    i = max(start_idx, 2)
    while i < min(end_idx, n - 1):
        pat3 = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        matched = None
        for direction in ('LONG', 'SHORT'):
            key = (pat3, direction)
            if key in pat_map:
                matched = pat_map[key]
                break
        if matched is None:
            i += 1
            continue

        entry = opens[i + 1]
        if entry <= 0:
            i += 1
            continue

        base_tp = matched['tp']
        base_sl = matched['sl']
        direction = matched['direction']

        # Get ATR ratio at signal bar
        raw_ratio = atr_ratio_arr[i] if i < len(atr_ratio_arr) else 1.0
        tp_mult, sl_mult = apply_strategy(raw_ratio, base_tp, base_sl, strategy_name)

        tp_pct = (base_tp * tp_mult) + SLIPPAGE_BUFFER
        sl_pct = (base_sl * sl_mult) - SLIPPAGE_BUFFER

        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)

        # Scan forward
        exited = False
        for j in range(i + 1, min(i + 1 + MAX_BARS, n)):
            if direction == 'LONG':
                hit_tp = highs[j] >= tp_price
                hit_sl = lows[j] <= sl_price
            else:
                hit_tp = lows[j] <= tp_price
                hit_sl = highs[j] >= sl_price

            if hit_tp and hit_sl:
                # Same-bar: use bar open distance
                dist_tp = abs(tp_price - opens[j])
                dist_sl = abs(sl_price - opens[j])
                if dist_tp <= dist_sl:
                    hit_sl = False
                else:
                    hit_tp = False

            if hit_tp:
                pnl = tp_pct * LEVERAGE - FEE
                trades.append({
                    'bar': i, 'pattern': pat3, 'direction': direction,
                    'entry': entry, 'pnl': pnl, 'win': True,
                    'tp_mult': tp_mult, 'sl_mult': sl_mult,
                    'raw_ratio': float(raw_ratio) if not np.isnan(raw_ratio) else 1.0,
                    'scaled_sl_pct': base_sl * sl_mult,
                    'max_loss_pct': base_sl * sl_mult * LEVERAGE,
                })
                i = j + 1
                exited = True
                break
            elif hit_sl:
                pnl = -(sl_pct * LEVERAGE + FEE)
                trades.append({
                    'bar': i, 'pattern': pat3, 'direction': direction,
                    'entry': entry, 'pnl': pnl, 'win': False,
                    'tp_mult': tp_mult, 'sl_mult': sl_mult,
                    'raw_ratio': float(raw_ratio) if not np.isnan(raw_ratio) else 1.0,
                    'scaled_sl_pct': base_sl * sl_mult,
                    'max_loss_pct': base_sl * sl_mult * LEVERAGE,
                })
                i = j + 1
                exited = True
                break

        if not exited:
            i += 1
            continue

    return trades


def calc_stats(trades):
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'edge': 0, 'mdd': 0,
                'max_single_loss': 0, 'daily_limit_violations': 0}

    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for t in trades if t['win'])
    n = len(trades)

    # Compound PnL
    equity = 1.0
    peak = 1.0
    max_dd = 0
    for p in pnls:
        equity *= (1 + p / 100)
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    # Max single loss (leverage-adjusted)
    max_loss = max((t['max_loss_pct'] for t in trades), default=0)

    # Daily limit violations
    violations = sum(1 for t in trades if t['max_loss_pct'] > MAX_DAILY_LOSS_PCT)

    return {
        'trades': n,
        'wr': round(wins / n * 100, 1) if n else 0,
        'pnl': round((equity - 1) * 100, 1),
        'edge': round(sum(pnls) / n, 3) if n else 0,
        'mdd': round(max_dd, 1),
        'max_single_loss': round(max_loss, 2),
        'daily_limit_violations': violations,
    }


# ===================================================================
# ATR RATIO DISTRIBUTION ANALYSIS
# ===================================================================

def analyze_atr_distribution(atr_ratio_arr):
    """Deep analysis of ATR ratio distribution."""
    valid = atr_ratio_arr[~np.isnan(atr_ratio_arr)]
    print(f"\n{'='*60}")
    print("PHASE 1: ATR Ratio Distribution Analysis")
    print(f"{'='*60}")
    print(f"  Valid observations: {len(valid)}")
    print(f"  Mean:   {np.mean(valid):.4f}")
    print(f"  Median: {np.median(valid):.4f}")
    print(f"  Std:    {np.std(valid):.4f}")
    print(f"  Min:    {np.min(valid):.4f}")
    print(f"  Max:    {np.max(valid):.4f}")
    print(f"  Skewness: {pd.Series(valid).skew():.3f}")
    print(f"  Kurtosis: {pd.Series(valid).kurtosis():.3f}")

    print("\n  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:02d}: {np.percentile(valid, p):.4f}")

    # Distribution buckets
    print("\n  Distribution buckets:")
    buckets = [(0, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2),
               (1.2, 1.5), (1.5, 1.7), (1.7, 2.0), (2.0, 999)]
    for lo, hi in buckets:
        count = np.sum((valid >= lo) & (valid < hi))
        pct = count / len(valid) * 100
        label = f"{lo:.1f}-{hi:.1f}" if hi < 999 else f"{lo:.1f}+"
        print(f"    [{label:>8s}]: {count:6d} ({pct:5.1f}%)")

    return {
        'count': int(len(valid)),
        'mean': round(float(np.mean(valid)), 4),
        'median': round(float(np.median(valid)), 4),
        'std': round(float(np.std(valid)), 4),
        'skewness': round(float(pd.Series(valid).skew()), 3),
        'kurtosis': round(float(pd.Series(valid).kurtosis()), 3),
        'percentiles': {str(p): round(float(np.percentile(valid, p)), 4) for p in [1,5,10,25,50,75,90,95,99]},
    }


# ===================================================================
# RISK ANALYSIS PER STRATEGY
# ===================================================================

def analyze_risk_profile(trades, strategy_name):
    """Analyze risk characteristics of a strategy's trades."""
    if not trades:
        return {}

    sl_pcts = [t['scaled_sl_pct'] for t in trades]
    max_losses = [t['max_loss_pct'] for t in trades]
    ratios = [t['raw_ratio'] for t in trades]
    sl_mults = [t['sl_mult'] for t in trades]
    tp_mults = [t['tp_mult'] for t in trades]

    return {
        'sl_pct_mean': round(np.mean(sl_pcts), 3),
        'sl_pct_max': round(np.max(sl_pcts), 3),
        'sl_pct_p95': round(np.percentile(sl_pcts, 95), 3),
        'max_loss_pct_mean': round(np.mean(max_losses), 2),
        'max_loss_pct_max': round(np.max(max_losses), 2),
        'max_loss_pct_p95': round(np.percentile(max_losses, 95), 2),
        'over_daily_limit': sum(1 for m in max_losses if m > MAX_DAILY_LOSS_PCT),
        'over_daily_limit_pct': round(sum(1 for m in max_losses if m > MAX_DAILY_LOSS_PCT) / len(trades) * 100, 1),
        'ratio_mean': round(np.mean(ratios), 3),
        'sl_mult_mean': round(np.mean(sl_mults), 3),
        'sl_mult_max': round(np.max(sl_mults), 3),
        'tp_mult_mean': round(np.mean(tp_mults), 3),
        'tp_mult_max': round(np.max(tp_mults), 3),
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    # Load data
    df = load_and_classify(DATA_FILE)
    overlap_bar = find_overlap_bar(df)
    patterns = load_patterns()
    print(f"  Patterns: {len(patterns)}, Overlap bar: {overlap_bar}")

    # Compute ATR ratio series
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    atr = compute_atr_series(highs, lows, closes)
    atr_ratio = compute_atr_ratio_series(atr)

    # Phase 1: ATR distribution analysis
    dist_info = analyze_atr_distribution(atr_ratio)

    # Phase 2: Strategy comparison
    print(f"\n{'='*60}")
    print("PHASE 2: Strategy Comparison (Full 720d)")
    print(f"{'='*60}")

    full_results = {}
    for strat in STRATEGIES:
        trades = bt_signals(df, patterns, atr_ratio, strat, 0, len(df))
        stats = calc_stats(trades)
        risk = analyze_risk_profile(trades, strat)
        full_results[strat] = {'stats': stats, 'risk': risk}
        viol = risk.get('over_daily_limit', 0)
        viol_pct = risk.get('over_daily_limit_pct', 0)
        print(f"  {strat:25s} | Trades={stats['trades']:4d} | WR={stats['wr']:5.1f}% | "
              f"PnL={stats['pnl']:>8.1f}% | MDD={stats['mdd']:5.1f}% | "
              f"Edge={stats['edge']:+.3f} | MaxLoss={risk.get('max_loss_pct_max',0):5.2f}% | "
              f"Violations={viol} ({viol_pct:.1f}%)")

    # Phase 3: WF 3-fold validation
    print(f"\n{'='*60}")
    print("PHASE 3: Walk-Forward 3-Fold Validation")
    print(f"{'='*60}")

    # WF folds: expanding window
    # Fold 1: IS=[0, 90d], OOS=[90d, 270d]
    # Fold 2: IS=[0, 270d], OOS=[270d, 450d]
    # Fold 3: IS=[0, 450d], OOS=[450d, 720d]
    fold_boundaries = [
        (0, 90 * BARS_PER_DAY, 270 * BARS_PER_DAY),
        (0, 270 * BARS_PER_DAY, 450 * BARS_PER_DAY),
        (0, 450 * BARS_PER_DAY, 720 * BARS_PER_DAY),
    ]

    # Baseline WF
    baseline_folds = []
    for fold_idx, (is_start, oos_start, oos_end) in enumerate(fold_boundaries):
        oos_end_actual = min(oos_end, len(df))
        trades = bt_signals(df, patterns, atr_ratio, 'A_NO_CAP', oos_start, oos_end_actual)
        # Baseline = no scaling (mult=1.0 always)
        # Actually need a "no scaling" baseline
        pass

    # For each strategy, compute WF OOS edge
    wf_results = {}
    # First get baseline (no ATR scaling at all)
    print("\n  Baseline (no ATR scaling):")
    bl_edges = []
    for fold_idx, (is_start, oos_start, oos_end) in enumerate(fold_boundaries):
        oos_end_actual = min(oos_end, len(df))
        # Create a "baseline" ratio array of all 1.0
        baseline_ratio = np.ones_like(atr_ratio)
        trades = bt_signals(df, patterns, baseline_ratio, 'A_NO_CAP', oos_start, oos_end_actual)
        stats = calc_stats(trades)
        bl_edges.append(stats['edge'])
        print(f"    Fold {fold_idx+1}: Trades={stats['trades']:3d} | WR={stats['wr']:5.1f}% | "
              f"Edge={stats['edge']:+.3f} | PnL={stats['pnl']:>7.1f}%")
    wf_results['BASELINE'] = {'fold_edges': bl_edges, 'avg_edge': round(np.mean(bl_edges), 3)}

    for strat in STRATEGIES:
        print(f"\n  {strat}:")
        fold_edges = []
        fold_details = []
        for fold_idx, (is_start, oos_start, oos_end) in enumerate(fold_boundaries):
            oos_end_actual = min(oos_end, len(df))
            trades = bt_signals(df, patterns, atr_ratio, strat, oos_start, oos_end_actual)
            stats = calc_stats(trades)
            risk = analyze_risk_profile(trades, strat)
            fold_edges.append(stats['edge'])
            edge_d = stats['edge'] - bl_edges[fold_idx]
            fold_details.append({
                'fold': fold_idx + 1,
                'stats': stats,
                'risk': risk,
                'bl_edge': bl_edges[fold_idx],
                'edge_d': round(edge_d, 3),
            })
            viol = risk.get('over_daily_limit', 0)
            print(f"    Fold {fold_idx+1}: Trades={stats['trades']:3d} | WR={stats['wr']:5.1f}% | "
                  f"Edge={stats['edge']:+.3f} (Δ{edge_d:+.3f}) | MaxLoss={risk.get('max_loss_pct_max',0):5.2f}% | "
                  f"Violations={viol}")

        all_positive = all(fe > ble for fe, ble in zip(fold_edges, bl_edges))
        avg_delta = round(np.mean([fe - ble for fe, ble in zip(fold_edges, bl_edges)]), 3)
        wf_pass = sum(1 for fe, ble in zip(fold_edges, bl_edges) if fe > ble)
        print(f"    → WF {wf_pass}/3 {'PASS' if wf_pass == 3 else 'FAIL'} | avg Δedge={avg_delta:+.3f}")

        wf_results[strat] = {
            'fold_edges': fold_edges,
            'avg_edge': round(np.mean(fold_edges), 3),
            'avg_delta': avg_delta,
            'wf_pass': wf_pass,
            'wf_verdict': 'PASS' if wf_pass == 3 else 'FAIL',
            'folds': fold_details,
        }

    # Phase 4: Summary & Recommendation
    print(f"\n{'='*60}")
    print("PHASE 4: Summary & Recommendation")
    print(f"{'='*60}")

    print(f"\n  {'Strategy':25s} | {'WF':5s} | {'Avg Δedge':>10s} | {'Max Loss%':>10s} | {'Violations':>10s} | {'720d PnL':>10s}")
    print(f"  {'-'*25}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for strat in STRATEGIES:
        wf = wf_results[strat]
        fr = full_results[strat]
        print(f"  {strat:25s} | {wf['wf_pass']}/3 {wf['wf_verdict']:4s} | "
              f"{wf['avg_delta']:>+10.3f} | "
              f"{fr['risk'].get('max_loss_pct_max',0):>10.2f}% | "
              f"{fr['risk'].get('over_daily_limit',0):>10d} | "
              f"{fr['stats']['pnl']:>10.1f}%")

    # Find best: WF PASS + no violations + highest avg_delta
    safe_pass = [(s, wf_results[s]) for s in STRATEGIES
                 if wf_results[s]['wf_verdict'] == 'PASS'
                 and full_results[s]['risk'].get('over_daily_limit', 0) == 0]

    if safe_pass:
        best_name, best_wf = max(safe_pass, key=lambda x: x[1]['avg_delta'])
        print(f"\n  ★ RECOMMENDED: {best_name}")
        print(f"    WF {best_wf['wf_pass']}/3 PASS | Avg Δedge={best_wf['avg_delta']:+.3f} | "
              f"Max loss={full_results[best_name]['risk']['max_loss_pct_max']:.2f}% | "
              f"0 daily limit violations")
    else:
        # Fallback: WF PASS + fewest violations
        pass_only = [(s, wf_results[s]) for s in STRATEGIES if wf_results[s]['wf_verdict'] == 'PASS']
        if pass_only:
            best_name, best_wf = min(pass_only, key=lambda x: full_results[x[0]]['risk'].get('over_daily_limit', 999))
            print(f"\n  ★ BEST AVAILABLE (has violations): {best_name}")
        else:
            best_name = None
            print("\n  ⚠ No strategy passed WF with acceptable risk")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # Save results
    output = {
        'study': 'atr_cap_strategy_study',
        'generated_at': pd.Timestamp.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'atr_distribution': dist_info,
        'strategies': {s: {
            'full': full_results[s],
            'wf': wf_results[s],
        } for s in STRATEGIES},
        'baseline_wf': wf_results['BASELINE'],
        'recommendation': best_name,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
