"""
Strategy Deep Review v1.0 — v1.26.2 Portfolio Validation
=========================================================
Comprehensive analysis triggered by live performance gap (v1.25.4 WR 90.4% → live 65%).

Phase 1: Distance Bias Verification
  - Same-candle TP/SL co-hit frequency for v1.25.4 patterns
  - WR under 3 resolution methods: distance (current), random 50/50, SL-first
  - Quantify how much WR inflation distance bias explains

Phase 2: ST-DN-U / U-ST-DN Mirror Pattern Correlation
  - Signal overlap frequency measurement
  - Simultaneous failure MDD impact simulation
  - Recommendation: keep both or remove one

Phase 3: Low-Sample (<20 trades) Pattern Validation
  - Bootstrap confidence intervals for 12 thin patterns
  - Expected trade frequency estimation
  - Risk tier classification

Phase 4: R:R < 1.0 Pattern TP/SL Re-optimization
  - Test if R:R >= 1.0 achievable for 14 patterns
  - Compare expectancy and WR resilience vs current settings

Phase 5: 60-Trade Monitoring Protocol Design
  - Statistical thresholds for live intervention
  - Sequential testing boundaries
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_STATS, PATTERN_OPTIMAL_TPSL
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
SLIPPAGE = 0.02
LEVERAGE = 3
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
SAFETY_MARGIN = 15

# v1.25.4 portfolio (for distance bias analysis)
V1254_PORTFOLIO = {
    "MD-BU-U": ("LONG", 0.5, 2.0), "MU-MU-U": ("LONG", 0.7, 2.0),
    "MU-U-MU": ("LONG", 0.5, 2.0), "BU-BU-BD": ("LONG", 0.7, 2.0),
    "ST-MU-U": ("LONG", 0.7, 2.0), "IH-DN-DN": ("LONG", 0.7, 2.0),
    "MD-ST-ST": ("SHORT", 0.5, 2.0), "U-MU-BU": ("SHORT", 0.5, 2.0),
    "MU-BU-DN": ("SHORT", 0.7, 2.0), "ST-H-U": ("SHORT", 0.5, 2.0),
    "ST-DN-H": ("SHORT", 0.7, 2.0), "MD-MU-U": ("SHORT", 1.0, 2.0),
    "BU-U-ST": ("SHORT", 0.7, 2.0), "H-DN-ST": ("SHORT", 0.7, 2.0),
    "DN-BD-BU": ("SHORT", 0.7, 2.0), "DN-BU-U": ("SHORT", 1.0, 2.0),
    "DN-U-U": ("SHORT", 0.5, 2.0), "ST-DN-U": ("SHORT", 0.7, 2.0),
    "ST-DN-DN": ("SHORT", 0.7, 2.0), "U-ST-DN": ("SHORT", 0.7, 2.0),
    "U-U-ST": ("SHORT", 0.7, 2.0), "ST-U-DN": ("SHORT", 0.7, 2.0),
    "ST-ST-U": ("SHORT", 0.7, 2.0),
}

# Build v1.26.2 portfolio from constants
V1262_PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    if pat in PATTERN_OPTIMAL_TPSL:
        tp, sl = PATTERN_OPTIMAL_TPSL[pat]
        V1262_PORTFOLIO[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    if pat in PATTERN_OPTIMAL_TPSL:
        tp, sl = PATTERN_OPTIMAL_TPSL[pat]
        V1262_PORTFOLIO[pat] = ("SHORT", tp, sl)

print("=" * 70)
print("Strategy Deep Review v1.0 — v1.26.2 Portfolio Validation")
print(f"  v1.25.4 patterns: {len(V1254_PORTFOLIO)}")
print(f"  v1.26.2 patterns: {len(V1262_PORTFOLIO)}")
print("=" * 70)

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} bars ({len(df)/288:.0f} days)")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

print("  Done.\n")


# ============================================================
# Enhanced backtest with co-hit tracking
# ============================================================
def bt_with_cohit_tracking(indices, direction, tp_pct, sl_pct):
    """
    Backtest that tracks same-candle co-hits separately.
    Returns: (pnls_distance, pnls_random, pnls_sl_first, cohit_count, total_count)
    """
    pnls_distance = []   # Current: distance-based (TP wins if closer)
    pnls_random = []     # Alternative: 50/50 random when both hit
    pnls_sl_first = []   # Worst case: SL always wins when both hit
    cohit_count = 0
    total_count = 0

    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)

            if ht and hs:
                # Both hit in same candle
                cohit_count += 1
                total_count += 1
                tp_pnl = tp_pct * LEVERAGE - FEE_PCT
                sl_pnl = -sl_pct * LEVERAGE - FEE_PCT

                # Distance-based (current method)
                if abs(tpp - entry) <= abs(slp - entry):
                    pnls_distance.append(tp_pnl)
                else:
                    pnls_distance.append(sl_pnl)

                # Random 50/50
                if np.random.random() < 0.5:
                    pnls_random.append(tp_pnl)
                else:
                    pnls_random.append(sl_pnl)

                # SL-first (worst case)
                pnls_sl_first.append(sl_pnl)
                break

            elif ht:
                total_count += 1
                tp_pnl = tp_pct * LEVERAGE - FEE_PCT
                pnls_distance.append(tp_pnl)
                pnls_random.append(tp_pnl)
                pnls_sl_first.append(tp_pnl)
                break

            elif hs:
                total_count += 1
                sl_pnl = -sl_pct * LEVERAGE - FEE_PCT
                pnls_distance.append(sl_pnl)
                pnls_random.append(sl_pnl)
                pnls_sl_first.append(sl_pnl)
                break

    return pnls_distance, pnls_random, pnls_sl_first, cohit_count, total_count


def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Standard backtest (distance-based)."""
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnls.append((tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT)
                break
            elif ht:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hs:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls


def mc_test(pnl_list, n_sims=MC_SIMS):
    """Monte Carlo sign randomization test."""
    if len(pnl_list) < 5:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def walk_forward_test(indices, direction, tp_pct, sl_pct, n_folds=5):
    """Walk-forward: count folds with positive PnL."""
    si = np.sort(indices)
    fs = len(si) // n_folds
    if fs < 3:
        return 0
    profitable = 0
    for f in range(n_folds):
        s = f * fs
        e = s + fs if f < n_folds - 1 else len(si)
        pnls = bt_fixed(si[s:e], direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


# ============================================================
# PHASE 1: Distance Bias Verification
# ============================================================
print("=" * 70)
print("PHASE 1: Distance Bias Verification")
print("  Question: Does intrabar distance bias explain v1.25.4's WR gap?")
print("=" * 70)

np.random.seed(42)

# Phase 1a: v1.25.4 patterns (the ones that showed 90.4% backtest → 65% live)
print(f"\n--- v1.25.4 Portfolio ({len(V1254_PORTFOLIO)} patterns) ---")
print(f"{'Pattern':<12} {'Dir':>5} {'TP/SL':>6} {'R:R':>5} | {'Trades':>6} {'CoHit':>6} {'%CoHit':>7} | {'WR_dist':>8} {'WR_rand':>8} {'WR_SL1st':>8} | {'Gap':>6}")
print("-" * 105)

v1254_totals = {'trades': 0, 'cohits': 0, 'wins_dist': 0, 'wins_rand': 0, 'wins_sl': 0}
phase1_details = {}

for pat, (direction, tp, sl) in sorted(V1254_PORTFOLIO.items()):
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 5:
        continue

    pnls_d, pnls_r, pnls_s, cohits, total = bt_with_cohit_tracking(indices, direction, tp, sl)
    if total == 0:
        continue

    rr = tp / sl
    wr_d = sum(1 for p in pnls_d if p > 0) / len(pnls_d) * 100 if pnls_d else 0
    wr_r = sum(1 for p in pnls_r if p > 0) / len(pnls_r) * 100 if pnls_r else 0
    wr_s = sum(1 for p in pnls_s if p > 0) / len(pnls_s) * 100 if pnls_s else 0
    cohit_pct = cohits / total * 100 if total > 0 else 0
    gap = wr_d - wr_r

    v1254_totals['trades'] += total
    v1254_totals['cohits'] += cohits
    v1254_totals['wins_dist'] += sum(1 for p in pnls_d if p > 0)
    v1254_totals['wins_rand'] += sum(1 for p in pnls_r if p > 0)
    v1254_totals['wins_sl'] += sum(1 for p in pnls_s if p > 0)

    phase1_details[pat] = {
        'direction': direction, 'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'trades': total, 'cohits': cohits, 'cohit_pct': round(cohit_pct, 1),
        'wr_distance': round(wr_d, 1), 'wr_random': round(wr_r, 1), 'wr_sl_first': round(wr_s, 1),
        'bias_gap': round(gap, 1),
    }

    print(f"{pat:<12} {direction:>5} {tp}/{sl:>3.1f} {rr:>5.2f} | {total:>6} {cohits:>6} {cohit_pct:>6.1f}% | {wr_d:>7.1f}% {wr_r:>7.1f}% {wr_s:>7.1f}% | {gap:>+5.1f}%")

# Aggregate
t = v1254_totals
if t['trades'] > 0:
    agg_wr_d = t['wins_dist'] / t['trades'] * 100
    agg_wr_r = t['wins_rand'] / t['trades'] * 100
    agg_wr_s = t['wins_sl'] / t['trades'] * 100
    agg_cohit = t['cohits'] / t['trades'] * 100
    print("-" * 105)
    print(f"{'AGGREGATE':<12} {'':>5} {'':>6} {'':>5} | {t['trades']:>6} {t['cohits']:>6} {agg_cohit:>6.1f}% | {agg_wr_d:>7.1f}% {agg_wr_r:>7.1f}% {agg_wr_s:>7.1f}% | {agg_wr_d - agg_wr_r:>+5.1f}%")

    print(f"\n  Key Findings:")
    print(f"  - Same-candle co-hits: {t['cohits']}/{t['trades']} = {agg_cohit:.1f}% of all trades")
    print(f"  - WR inflation from distance bias: {agg_wr_d:.1f}% → {agg_wr_r:.1f}% (random) = {agg_wr_d - agg_wr_r:+.1f}pp")
    print(f"  - Worst case (SL-first): {agg_wr_s:.1f}% (gap: {agg_wr_d - agg_wr_s:+.1f}pp)")
    print(f"  - Live WR: ~65%")
    if abs(agg_wr_r - 65) < 10:
        print(f"  >>> Random-resolution WR ({agg_wr_r:.1f}%) is CLOSE to live WR (65%)")
        print(f"  >>> Distance bias likely explains most of the backtest-live gap!")
    else:
        print(f"  >>> Random-resolution WR ({agg_wr_r:.1f}%) differs from live WR (65%) by {abs(agg_wr_r - 65):.1f}pp")

# Phase 1b: v1.26.2 patterns (check if bias still exists)
print(f"\n--- v1.26.2 Portfolio ({len(V1262_PORTFOLIO)} patterns) ---")
print(f"{'Pattern':<12} {'Dir':>5} {'TP/SL':>6} {'R:R':>5} | {'Trades':>6} {'CoHit':>6} {'%CoHit':>7} | {'WR_dist':>8} {'WR_rand':>8} {'WR_SL1st':>8} | {'Gap':>6}")
print("-" * 105)

v1262_totals = {'trades': 0, 'cohits': 0, 'wins_dist': 0, 'wins_rand': 0, 'wins_sl': 0}
v1262_cohit_details = {}

# Only show patterns with >0 co-hits or R:R != 1.0 (where bias matters)
for pat, (direction, tp, sl) in sorted(V1262_PORTFOLIO.items()):
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 5:
        continue

    pnls_d, pnls_r, pnls_s, cohits, total = bt_with_cohit_tracking(indices, direction, tp, sl)
    if total == 0:
        continue

    rr = tp / sl
    wr_d = sum(1 for p in pnls_d if p > 0) / len(pnls_d) * 100 if pnls_d else 0
    wr_r = sum(1 for p in pnls_r if p > 0) / len(pnls_r) * 100 if pnls_r else 0
    wr_s = sum(1 for p in pnls_s if p > 0) / len(pnls_s) * 100 if pnls_s else 0
    gap = wr_d - wr_r

    v1262_totals['trades'] += total
    v1262_totals['cohits'] += cohits
    v1262_totals['wins_dist'] += sum(1 for p in pnls_d if p > 0)
    v1262_totals['wins_rand'] += sum(1 for p in pnls_r if p > 0)
    v1262_totals['wins_sl'] += sum(1 for p in pnls_s if p > 0)

    v1262_cohit_details[pat] = {
        'direction': direction, 'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'trades': total, 'cohits': cohits, 'wr_distance': round(wr_d, 1),
        'wr_random': round(wr_r, 1), 'bias_gap': round(gap, 1),
    }

    if cohits > 0 or abs(gap) > 0.5:
        print(f"{pat:<12} {direction:>5} {tp}/{sl:>3.1f} {rr:>5.2f} | {total:>6} {cohits:>6} {cohits/total*100:>6.1f}% | {wr_d:>7.1f}% {wr_r:>7.1f}% {wr_s:>7.1f}% | {gap:>+5.1f}%")

t = v1262_totals
if t['trades'] > 0:
    agg_wr_d = t['wins_dist'] / t['trades'] * 100
    agg_wr_r = t['wins_rand'] / t['trades'] * 100
    agg_wr_s = t['wins_sl'] / t['trades'] * 100
    agg_cohit = t['cohits'] / t['trades'] * 100
    print("-" * 105)
    print(f"{'AGGREGATE':<12} {'':>5} {'':>6} {'':>5} | {t['trades']:>6} {t['cohits']:>6} {agg_cohit:>6.1f}% | {agg_wr_d:>7.1f}% {agg_wr_r:>7.1f}% {agg_wr_s:>7.1f}% | {agg_wr_d - agg_wr_r:>+5.1f}%")
    print(f"\n  v1.26.2 bias: {agg_wr_d - agg_wr_r:+.1f}pp (vs v1.25.4 bias: {v1254_totals['wins_dist']/v1254_totals['trades']*100 - v1254_totals['wins_rand']/v1254_totals['trades']*100:+.1f}pp)")
    print(f"  Higher R:R in v1.26.2 reduces distance bias significantly")


# ============================================================
# PHASE 2: ST-DN-U / U-ST-DN Mirror Pattern Correlation
# ============================================================
print(f"\n{'='*70}")
print("PHASE 2: ST-DN-U / U-ST-DN Mirror Pattern Correlation")
print("  Both: SHORT, 3.0/3.0, R:R=1.0, excess_wr=+7.6% (weakest edge)")
print("=" * 70)

idx_stdu = pattern_indices.get("ST-DN-U", np.array([]))
idx_ustd = pattern_indices.get("U-ST-DN", np.array([]))

# Signal overlap: count how many signal bars are within N bars of each other
overlap_windows = [0, 1, 3, 5, 10]
print(f"\n  Signal Overlap Analysis:")
print(f"  ST-DN-U signals: {len(idx_stdu)}")
print(f"  U-ST-DN signals: {len(idx_ustd)}")

overlap_results = {}
for window in overlap_windows:
    overlaps = 0
    for idx_s in idx_stdu:
        for idx_u in idx_ustd:
            if abs(int(idx_s) - int(idx_u)) <= window:
                overlaps += 1
                break
    pct = overlaps / len(idx_stdu) * 100 if len(idx_stdu) > 0 else 0
    overlap_results[window] = {'count': overlaps, 'pct': round(pct, 1)}
    print(f"  Window +/- {window} bars: {overlaps}/{len(idx_stdu)} ST-DN-U signals overlap ({pct:.1f}%)")

# Exact same bar overlap
same_bar = len(set(idx_stdu.tolist()) & set(idx_ustd.tolist()))
print(f"\n  Exact same bar signals: {same_bar}")

# Trade outcome correlation
print(f"\n  Trade Outcome Correlation:")
if "ST-DN-U" in V1262_PORTFOLIO and "U-ST-DN" in V1262_PORTFOLIO:
    _, tp_s, sl_s = V1262_PORTFOLIO["ST-DN-U"]
    _, tp_u, sl_u = V1262_PORTFOLIO["U-ST-DN"]

    # Get per-trade outcomes for both patterns
    def get_trade_outcomes(indices, direction, tp, sl):
        outcomes = {}
        for idx in indices:
            if idx + 1 >= n_bars:
                continue
            entry = opens[idx + 1]
            if entry <= 0:
                continue
            if direction == 'LONG':
                tpp = entry * (1 + tp / 100)
                slp = entry * (1 - sl / 100)
            else:
                tpp = entry * (1 - tp / 100)
                slp = entry * (1 + sl / 100)
            for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
                if ht and hs:
                    win = abs(tpp - entry) <= abs(slp - entry)
                    outcomes[int(idx)] = 1 if win else 0
                    break
                elif ht:
                    outcomes[int(idx)] = 1
                    break
                elif hs:
                    outcomes[int(idx)] = 0
                    break
        return outcomes

    outcomes_s = get_trade_outcomes(idx_stdu, "SHORT", tp_s, sl_s)
    outcomes_u = get_trade_outcomes(idx_ustd, "SHORT", tp_u, sl_u)

    # Find simultaneous trades (within 10 bars)
    simultaneous = []
    for idx_s in outcomes_s:
        for idx_u in outcomes_u:
            if abs(idx_s - idx_u) <= 10:
                simultaneous.append((idx_s, idx_u, outcomes_s[idx_s], outcomes_u[idx_u]))
                break

    if simultaneous:
        both_win = sum(1 for _, _, w1, w2 in simultaneous if w1 == 1 and w2 == 1)
        both_lose = sum(1 for _, _, w1, w2 in simultaneous if w1 == 0 and w2 == 0)
        s_win_u_lose = sum(1 for _, _, w1, w2 in simultaneous if w1 == 1 and w2 == 0)
        s_lose_u_win = sum(1 for _, _, w1, w2 in simultaneous if w1 == 0 and w2 == 1)

        print(f"  Simultaneous trades (within 10 bars): {len(simultaneous)}")
        print(f"  Both WIN:  {both_win} ({both_win/len(simultaneous)*100:.1f}%)")
        print(f"  Both LOSE: {both_lose} ({both_lose/len(simultaneous)*100:.1f}%)")
        print(f"  ST-DN-U win, U-ST-DN lose: {s_win_u_lose}")
        print(f"  ST-DN-U lose, U-ST-DN win: {s_lose_u_win}")

        # Correlation coefficient
        if len(simultaneous) >= 5:
            s_outcomes = np.array([w1 for _, _, w1, _ in simultaneous])
            u_outcomes = np.array([w2 for _, _, _, w2 in simultaneous])
            if np.std(s_outcomes) > 0 and np.std(u_outcomes) > 0:
                corr = np.corrcoef(s_outcomes, u_outcomes)[0, 1]
                print(f"  Win/Loss correlation: {corr:.3f}")
                if corr > 0.5:
                    print(f"  >>> HIGH correlation — removing one would reduce correlated drawdowns")
                elif corr > 0.2:
                    print(f"  >>> MODERATE correlation — some diversification benefit exists")
                else:
                    print(f"  >>> LOW correlation — patterns provide independent signals")

    # MDD impact: portfolio with both vs portfolio with only one
    print(f"\n  MDD Impact Simulation:")

    def simulate_1pos(portfolio_map):
        """1-position-at-a-time simulation, returns (trades, pnl, mdd)."""
        raw_trades = []
        for i in range(2, n_bars):
            pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
            if pat not in portfolio_map:
                continue
            direction, tp, sl = portfolio_map[pat]
            if i + 1 >= n_bars:
                continue
            entry = opens[i + 1]
            if entry <= 0:
                continue
            if direction == 'LONG':
                tpp = entry * (1 + tp / 100)
                slp = entry * (1 - sl / 100)
            else:
                tpp = entry * (1 - tp / 100)
                slp = entry * (1 + sl / 100)
            for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
                ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
                if ht and hs:
                    pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                    raw_trades.append((i + 1, j, pnl))
                    break
                elif ht:
                    raw_trades.append((i + 1, j, tp * LEVERAGE - FEE_PCT))
                    break
                elif hs:
                    raw_trades.append((i + 1, j, -sl * LEVERAGE - FEE_PCT))
                    break

        raw_trades.sort(key=lambda x: x[0])
        filtered = []
        last_exit = -1
        for entry_bar, exit_bar, pnl in raw_trades:
            if entry_bar > last_exit:
                filtered.append(pnl)
                last_exit = exit_bar

        cum = 0
        peak = 0
        mdd = 0
        for p in filtered:
            cum += p
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > mdd:
                mdd = dd
        wr = sum(1 for p in filtered if p > 0) / len(filtered) * 100 if filtered else 0
        return len(filtered), cum, mdd, wr

    # Full portfolio
    full_trades, full_pnl, full_mdd, full_wr = simulate_1pos(V1262_PORTFOLIO)

    # Without ST-DN-U
    no_stdu = {k: v for k, v in V1262_PORTFOLIO.items() if k != "ST-DN-U"}
    t1, p1, m1, w1 = simulate_1pos(no_stdu)

    # Without U-ST-DN
    no_ustd = {k: v for k, v in V1262_PORTFOLIO.items() if k != "U-ST-DN"}
    t2, p2, m2, w2 = simulate_1pos(no_ustd)

    # Without both
    no_both = {k: v for k, v in V1262_PORTFOLIO.items() if k not in ("ST-DN-U", "U-ST-DN")}
    t3, p3, m3, w3 = simulate_1pos(no_both)

    print(f"  {'Scenario':<25} {'Pats':>4} {'Trades':>6} {'WR':>6} {'PnL':>10} {'MDD':>8} {'PnL/MDD':>8}")
    print("  " + "-" * 75)
    print(f"  {'Full portfolio':<25} {len(V1262_PORTFOLIO):>4} {full_trades:>6} {full_wr:>5.1f}% {full_pnl:>+9.1f}% {full_mdd:>7.1f}% {full_pnl/full_mdd if full_mdd > 0 else 999:>7.2f}x")
    print(f"  {'Without ST-DN-U':<25} {len(no_stdu):>4} {t1:>6} {w1:>5.1f}% {p1:>+9.1f}% {m1:>7.1f}% {p1/m1 if m1 > 0 else 999:>7.2f}x")
    print(f"  {'Without U-ST-DN':<25} {len(no_ustd):>4} {t2:>6} {w2:>5.1f}% {p2:>+9.1f}% {m2:>7.1f}% {p2/m2 if m2 > 0 else 999:>7.2f}x")
    print(f"  {'Without both':<25} {len(no_both):>4} {t3:>6} {w3:>5.1f}% {p3:>+9.1f}% {m3:>7.1f}% {p3/m3 if m3 > 0 else 999:>7.2f}x")

    # Recommendation
    print(f"\n  Recommendation:")
    if full_pnl / full_mdd > p3 / m3 if m3 > 0 else 0:
        print(f"  >>> Keep both: PnL/MDD ratio ({full_pnl/full_mdd:.2f}x) BETTER than without both ({p3/m3:.2f}x)")
    else:
        print(f"  >>> Consider removing both: PnL/MDD ratio improves from {full_pnl/full_mdd:.2f}x to {p3/m3:.2f}x")


# ============================================================
# PHASE 3: Low-Sample (<20 trades) Pattern Validation
# ============================================================
print(f"\n{'='*70}")
print("PHASE 3: Low-Sample (<20 trades) Pattern Validation")
print("=" * 70)

low_sample_patterns = []
for pat, (direction, tp, sl) in V1262_PORTFOLIO.items():
    indices = pattern_indices.get(pat, np.array([]))
    pnls = bt_fixed(indices, direction, tp, sl)
    if len(pnls) < 20:
        low_sample_patterns.append({
            'pattern': pat, 'direction': direction, 'tp': tp, 'sl': sl,
            'trades': len(pnls), 'indices_count': len(indices),
        })

print(f"\n  Patterns with < 20 trades: {len(low_sample_patterns)}")
print(f"\n  {'Pattern':<12} {'Dir':>5} {'TP/SL':>6} {'Trades':>6} {'Signals':>7} | {'WR':>6} {'95%CI':>14} | {'Boot_Exp':>9} {'Risk':>6}")
print("  " + "-" * 90)

phase3_results = {}
for entry in sorted(low_sample_patterns, key=lambda x: x['trades']):
    pat = entry['pattern']
    direction = entry['direction']
    tp, sl = entry['tp'], entry['sl']
    indices = pattern_indices.get(pat, np.array([]))
    pnls = bt_fixed(indices, direction, tp, sl)

    if len(pnls) == 0:
        continue

    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    wins = sum(1 for p in pnls if p > 0)

    # Bootstrap confidence interval for WR
    n_boot = 10000
    boot_wrs = []
    pnl_arr = np.array(pnls)
    for _ in range(n_boot):
        sample = np.random.choice(pnl_arr, size=len(pnl_arr), replace=True)
        boot_wrs.append(sum(1 for p in sample if p > 0) / len(sample) * 100)

    ci_low = np.percentile(boot_wrs, 2.5)
    ci_high = np.percentile(boot_wrs, 97.5)
    boot_exp_mean = np.mean(boot_wrs)

    # Expected trades per day
    trades_per_day = len(pnls) / 270

    # Break-even WR for this TP/SL
    win_pnl = tp * LEVERAGE - FEE_PCT
    loss_pnl = sl * LEVERAGE + FEE_PCT
    be_wr = loss_pnl / (win_pnl + loss_pnl) * 100

    # Risk tier
    wr_margin = ci_low - be_wr
    if wr_margin > 15:
        risk = "LOW"
    elif wr_margin > 5:
        risk = "MED"
    elif wr_margin > 0:
        risk = "HIGH"
    else:
        risk = "CRIT"

    phase3_results[pat] = {
        'direction': direction, 'tp': tp, 'sl': sl,
        'trades': len(pnls), 'signals': len(indices),
        'wr': round(wr, 1), 'ci_low': round(ci_low, 1), 'ci_high': round(ci_high, 1),
        'be_wr': round(be_wr, 1), 'wr_margin_at_95ci': round(wr_margin, 1),
        'risk_tier': risk, 'trades_per_day': round(trades_per_day, 2),
    }

    print(f"  {pat:<12} {direction:>5} {tp}/{sl:>3.1f} {len(pnls):>6} {len(indices):>7} | {wr:>5.1f}% [{ci_low:>5.1f}-{ci_high:>5.1f}] | {boot_exp_mean:>8.1f}% {risk:>6}")

# Summary
crit_count = sum(1 for v in phase3_results.values() if v['risk_tier'] == 'CRIT')
high_count = sum(1 for v in phase3_results.values() if v['risk_tier'] == 'HIGH')
print(f"\n  Summary: CRIT={crit_count}, HIGH={high_count}, MED={sum(1 for v in phase3_results.values() if v['risk_tier'] == 'MED')}, LOW={sum(1 for v in phase3_results.values() if v['risk_tier'] == 'LOW')}")
if crit_count > 0:
    crit_pats = [k for k, v in phase3_results.items() if v['risk_tier'] == 'CRIT']
    print(f"  CRITICAL patterns (95% CI lower bound below BE WR): {', '.join(crit_pats)}")
    print(f"  >>> These patterns need additional data or removal consideration")


# ============================================================
# PHASE 4: R:R < 1.0 Pattern TP/SL Re-optimization
# ============================================================
print(f"\n{'='*70}")
print("PHASE 4: R:R < 1.0 Pattern TP/SL Re-optimization")
print("  Goal: Find R:R >= 1.0 alternatives that maintain edge")
print("=" * 70)

low_rr_patterns = []
for pat, (direction, tp, sl) in V1262_PORTFOLIO.items():
    rr = tp / sl
    if rr < 1.0:
        low_rr_patterns.append((pat, direction, tp, sl, rr))

print(f"\n  Patterns with R:R < 1.0: {len(low_rr_patterns)}")
print(f"\n  {'Pattern':<12} {'Dir':>5} {'Cur TP/SL':>9} {'R:R':>5} | {'Best TP/SL':>10} {'New R:R':>7} {'WR':>6} {'Exp':>7} {'MC':>7} {'WF':>4} | {'Status':>8}")
print("  " + "-" * 100)

phase4_results = {}
for pat, direction, cur_tp, cur_sl, cur_rr in sorted(low_rr_patterns, key=lambda x: x[4]):
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 10:
        continue

    # Current performance
    cur_pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    cur_wr = sum(1 for p in cur_pnls if p > 0) / len(cur_pnls) * 100 if cur_pnls else 0
    cur_exp = None
    if cur_pnls:
        wins = [p for p in cur_pnls if p > 0]
        losses = [abs(p) for p in cur_pnls if p <= 0]
        avg_w = np.mean(wins) if wins else 0
        avg_l = np.mean(losses) if losses else 0
        cur_exp = (cur_wr / 100) * avg_w - (1 - cur_wr / 100) * avg_l

    # Search R:R >= 1.0 alternatives
    best = None
    best_score = -999

    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            rr = tp / sl
            if rr < 1.0:
                continue

            pnls = bt_fixed(indices, direction, tp, sl)
            if len(pnls) < 10:
                continue

            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            total_pnl = sum(pnls)
            if total_pnl <= 0:
                continue

            # Break-even WR check
            win_pnl = tp * LEVERAGE - FEE_PCT
            loss_pnl = sl * LEVERAGE + FEE_PCT
            be_wr = loss_pnl / (win_pnl + loss_pnl) * 100
            if wr < be_wr + SAFETY_MARGIN:
                continue

            # Quick MC
            mc = mc_test(pnls, 3000)
            if mc >= 0.05:
                continue

            wf = walk_forward_test(indices, direction, tp, sl)
            if wf < 3:
                continue

            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p <= 0]
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 0
            exp = (wr / 100) * avg_w - (1 - wr / 100) * avg_l

            score = exp * len(pnls) * (wf / 5)
            if score > best_score:
                best_score = score
                best = {
                    'tp': tp, 'sl': sl, 'rr': round(rr, 2),
                    'trades': len(pnls), 'wr': round(wr, 1),
                    'mc': round(mc, 4), 'wf': wf,
                    'expectancy': round(exp, 3), 'total_pnl': round(total_pnl, 1),
                }

    phase4_results[pat] = {
        'direction': direction,
        'current': {'tp': cur_tp, 'sl': cur_sl, 'rr': round(cur_rr, 2),
                    'wr': round(cur_wr, 1), 'exp': round(cur_exp, 3) if cur_exp else None},
        'best_rr1': best,
    }

    if best:
        status = "FOUND" if best['expectancy'] > 0 else "WEAK"
        print(f"  {pat:<12} {direction:>5} {cur_tp}/{cur_sl:>3.1f} {cur_rr:>5.2f} | {best['tp']}/{best['sl']:>3.1f} {best['rr']:>7.2f} {best['wr']:>5.1f}% {best['expectancy']:>+6.3f}% {best['mc']:>7.4f} {best['wf']:>3}/5 | {status:>8}")
    else:
        print(f"  {pat:<12} {direction:>5} {cur_tp}/{cur_sl:>3.1f} {cur_rr:>5.2f} | {'---':>10} {'---':>7} {'---':>6} {'---':>7} {'---':>7} {'---':>4} | {'NONE':>8}")

found = sum(1 for v in phase4_results.values() if v['best_rr1'] is not None)
print(f"\n  R:R >= 1.0 alternatives found: {found}/{len(low_rr_patterns)}")
if found > 0:
    better = sum(1 for v in phase4_results.values()
                 if v['best_rr1'] and v['current']['exp'] and
                 v['best_rr1']['expectancy'] > v['current']['exp'])
    print(f"  With better expectancy than current: {better}/{found}")


# ============================================================
# PHASE 5: 60-Trade Monitoring Protocol Design
# ============================================================
print(f"\n{'='*70}")
print("PHASE 5: 60-Trade Monitoring Protocol Design")
print("=" * 70)

# Calculate statistical thresholds
expected_wr = 72.0  # v1.26.2 expected
be_wr_portfolio = 44.1  # from T5 analysis

# Binomial test: what WR is statistically concerning at different trade counts?
from scipy.stats import binom

print(f"\n  Expected WR: {expected_wr}%")
print(f"  Break-even WR: {be_wr_portfolio}%")
print(f"\n  Binomial Alarm Thresholds (p < 0.05 that true WR >= expected):")
print(f"  {'Trades':>6} {'Min Wins':>8} {'Min WR':>7} {'Margin':>8} {'Action':>15}")
print("  " + "-" * 55)

monitoring_thresholds = {}
for n_trades in [10, 20, 30, 40, 50, 60, 80, 100]:
    # Find min wins where P(X <= wins | p=expected_wr) < 0.05
    for wins in range(n_trades + 1):
        p_value = binom.cdf(wins, n_trades, expected_wr / 100)
        if p_value >= 0.05:
            break
    alarm_wr = (wins - 1) / n_trades * 100 if wins > 0 else 0
    margin = alarm_wr - be_wr_portfolio

    if n_trades <= 20:
        action = "OBSERVE"
    elif margin < 5:
        action = "HALT"
    elif alarm_wr < 55:
        action = "REDUCE SIZE"
    else:
        action = "MONITOR"

    monitoring_thresholds[n_trades] = {
        'min_wins': wins - 1, 'alarm_wr': round(alarm_wr, 1),
        'margin_over_be': round(margin, 1), 'action': action,
    }
    print(f"  {n_trades:>6} {wins-1:>8} {alarm_wr:>6.1f}% {margin:>+7.1f}pp {action:>15}")

# Consecutive loss analysis
print(f"\n  Consecutive Loss Probabilities (WR = {expected_wr}%):")
print(f"  {'Streak':>6} {'P(streak)':>10} {'Expected per 60':>16}")
print("  " + "-" * 40)

loss_prob = 1 - expected_wr / 100
for streak in [3, 4, 5, 6, 7, 8]:
    p_streak = loss_prob ** streak
    expected_in_60 = 60 * p_streak
    print(f"  {streak:>6} {p_streak:>9.4f} {expected_in_60:>15.2f}")

# Design monitoring protocol
print(f"\n  === RECOMMENDED MONITORING PROTOCOL ===")
print(f"  ")
print(f"  Level 1 - OBSERVE (trades 1-20):")
print(f"    - Track all outcomes, no automated intervention")
print(f"    - Flag if WR < 50% (10+ losses out of 20)")
print(f"  ")
print(f"  Level 2 - MONITOR (trades 21-40):")
print(f"    - Alert if WR < 55%")
print(f"    - Alert on 5+ consecutive losses")
print(f"    - Review if cumulative PnL < 0")
print(f"  ")
print(f"  Level 3 - ACTIVE (trades 41-60):")
print(f"    - REDUCE position size 50% if WR < 60%")
print(f"    - HALT trading if WR < 50% or MDD > 35%")
print(f"    - Review strategy if 6+ consecutive losses")
print(f"  ")
print(f"  Level 4 - DECISION (after 60 trades):")
print(f"    - If WR >= 65%: CONTINUE (strategy validated)")
print(f"    - If 55% <= WR < 65%: CONTINUE with caution (within tolerance)")
print(f"    - If 45% <= WR < 55%: REDUCE to 50% size, extend to 100 trades")
print(f"    - If WR < 45%: HALT (below break-even)")
print(f"  ")
print(f"  Emergency Stops (any time):")
print(f"    - MDD > 35%: Immediate halt")
print(f"    - 7+ consecutive losses: Halt + manual review")
print(f"    - Daily loss > 10%: Halt for 24 hours")


# ============================================================
# Save Results
# ============================================================
print(f"\n{'='*70}")
print("Saving Results")
print("=" * 70)

output = {
    'timestamp': datetime.now().isoformat(),
    'research': 'Strategy Deep Review v1.0 — v1.26.2 Portfolio Validation',
    'phase1_distance_bias': {
        'v1254': {
            'aggregate': {
                'total_trades': v1254_totals['trades'],
                'total_cohits': v1254_totals['cohits'],
                'cohit_pct': round(v1254_totals['cohits'] / v1254_totals['trades'] * 100, 1) if v1254_totals['trades'] > 0 else 0,
                'wr_distance': round(v1254_totals['wins_dist'] / v1254_totals['trades'] * 100, 1) if v1254_totals['trades'] > 0 else 0,
                'wr_random': round(v1254_totals['wins_rand'] / v1254_totals['trades'] * 100, 1) if v1254_totals['trades'] > 0 else 0,
                'wr_sl_first': round(v1254_totals['wins_sl'] / v1254_totals['trades'] * 100, 1) if v1254_totals['trades'] > 0 else 0,
            },
            'per_pattern': phase1_details,
        },
        'v1262': {
            'aggregate': {
                'total_trades': v1262_totals['trades'],
                'total_cohits': v1262_totals['cohits'],
                'cohit_pct': round(v1262_totals['cohits'] / v1262_totals['trades'] * 100, 1) if v1262_totals['trades'] > 0 else 0,
                'wr_distance': round(v1262_totals['wins_dist'] / v1262_totals['trades'] * 100, 1) if v1262_totals['trades'] > 0 else 0,
                'wr_random': round(v1262_totals['wins_rand'] / v1262_totals['trades'] * 100, 1) if v1262_totals['trades'] > 0 else 0,
                'wr_sl_first': round(v1262_totals['wins_sl'] / v1262_totals['trades'] * 100, 1) if v1262_totals['trades'] > 0 else 0,
            },
            'per_pattern': v1262_cohit_details,
        },
    },
    'phase2_mirror_correlation': {
        'signal_overlap': overlap_results,
        'same_bar_signals': same_bar,
        'portfolio_impact': {
            'full': {'trades': full_trades, 'pnl': round(full_pnl, 1), 'mdd': round(full_mdd, 1), 'wr': round(full_wr, 1)},
            'no_stdu': {'trades': t1, 'pnl': round(p1, 1), 'mdd': round(m1, 1), 'wr': round(w1, 1)},
            'no_ustd': {'trades': t2, 'pnl': round(p2, 1), 'mdd': round(m2, 1), 'wr': round(w2, 1)},
            'no_both': {'trades': t3, 'pnl': round(p3, 1), 'mdd': round(m3, 1), 'wr': round(w3, 1)},
        },
    },
    'phase3_low_sample': phase3_results,
    'phase4_rr_reopt': phase4_results,
    'phase5_monitoring': {
        'expected_wr': expected_wr,
        'be_wr': be_wr_portfolio,
        'thresholds': monitoring_thresholds,
    },
}

out_path = Path(__file__).resolve().parent / '../../results/strategy_deep_review.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("\nDone.")
