"""
Pattern Combination Optimization Research
==========================================
52패턴 중 최적 부분집합 탐색.
1-pos-at-a-time 제약에서 패턴 간 신호 경쟁이 발생하므로,
패턴 수 ≠ 성과. 최적 조합을 찾는 심층 연구.

Phase 1: Leave-One-Out (LOO) Marginal Contribution
  - 각 패턴 제거 시 포트폴리오 영향 측정
  - 음의 기여 패턴 식별

Phase 2: Greedy Backward Elimination (WF)
  - 훈련 데이터에서 가장 해로운 패턴을 순차 제거
  - 제거 중단 시점 = 최적 부분집합 크기
  - OOS 검증

Phase 3: Greedy Forward Selection (WF)
  - 빈 포트폴리오에서 시작, 가장 기여도 높은 패턴 순차 추가
  - 추가 중단 시점 = 최적 부분집합 크기
  - OOS 검증

Phase 4: Portfolio Size Sweep
  - 랜덤 서브셋 (크기 10~50) 각 200회 샘플링
  - 크기별 PnL/MDD 분포 분석

Phase 5: WF Comparison of Best Strategies
  - Phase 2-3에서 발견된 최적 조합들의 WF OOS 비교

Standard Research Protocol: next-bar open entry, distance-based exit,
  0.10% fee, 3x leverage, 10k MC sims
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
    PATTERN_OPTIMAL_TPSL,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
TP_SCALE = 0.70
N_PERIODS = 5
RANDOM_SUBSETS = 200  # per size level

np.random.seed(42)

print("=" * 70)
print("Pattern Combination Optimization Research")
print("=" * 70)

# ============================================================
# Data loading & classification
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)
print(f"Data: {n_bars} bars")

print("Classifying candles...")
body_abs = np.abs(closes - opens)
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values
types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20[i] if not pd.isna(avg_body_20[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)
print("Done.\n")

# Build full portfolio
PORTFOLIO = {}
ALL_PATTERNS = []
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", round(tp * TP_SCALE, 2), sl)
    ALL_PATTERNS.append(pat)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", round(tp * TP_SCALE, 2), sl)
    ALL_PATTERNS.append(pat)

print(f"Portfolio: {len(PORTFOLIO)} patterns ({len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S)")

# ============================================================
# Precompute all raw trades per pattern (for speed)
# ============================================================
print("Precomputing per-pattern trades...")

def simulate_trade_raw(direction, tp, sl, signal_bar):
    eb = signal_bar + 1
    if eb >= n_bars:
        return None
    entry = opens[eb]
    if entry <= 0:
        return None
    if direction == 'LONG':
        tpp = entry * (1 + tp / 100)
        slp = entry * (1 - sl / 100)
    else:
        tpp = entry * (1 - tp / 100)
        slp = entry * (1 + sl / 100)
    for j in range(eb + 1, min(eb + MAX_BARS, n_bars)):
        ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
        hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
        if ht and hs:
            pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
            return (eb, j, pnl)
        elif ht:
            return (eb, j, tp * LEVERAGE - FEE_PCT)
        elif hs:
            return (eb, j, -sl * LEVERAGE - FEE_PCT)
    return None

# pattern_trades[pat] = list of (signal_bar, entry_bar, exit_bar, pnl)
pattern_trades = {}
for pat, (direction, tp, sl) in PORTFOLIO.items():
    trades = []
    for i in range(2, n_bars):
        p = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if p != pat:
            continue
        result = simulate_trade_raw(direction, tp, sl, i)
        if result is not None:
            trades.append((i, *result))  # (signal_bar, entry_bar, exit_bar, pnl)
    pattern_trades[pat] = trades

total_raw = sum(len(v) for v in pattern_trades.values())
print(f"Total raw trades: {total_raw}")


def portfolio_from_patterns(pat_list, bar_min=0, bar_max=None):
    """Build 1-pos portfolio from pattern list, using precomputed trades."""
    if bar_max is None:
        bar_max = n_bars - 1
    # Merge all trades in time order
    all_trades = []
    for pat in pat_list:
        for sig, eb, xb, pnl in pattern_trades[pat]:
            if bar_min <= sig <= bar_max:
                all_trades.append((eb, xb, pnl, pat))
    all_trades.sort(key=lambda x: x[0])
    # 1-pos filter
    filtered = []
    last_exit = -1
    for eb, xb, pnl, pat in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl, pat))
            last_exit = xb
    return filtered


def eval_pnl(trades):
    """Quick PnL sum."""
    return sum(t[2] for t in trades) if trades else 0


def eval_full(trades):
    """Full stats."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0}
    pnls = [t[2] for t in trades]
    cum = peak = max_dd = win_sum = loss_sum = 0
    wins = 0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_sum += p
        else:
            loss_sum += abs(p)
    return {
        'pnl': round(cum, 2),
        'trades': len(pnls),
        'wr': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(max_dd, 2),
        'pf': round(win_sum / loss_sum, 2) if loss_sum > 0 else 999,
        'pnl_mdd': round(cum / max_dd, 1) if max_dd > 0 else 999,
    }


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 8:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


# WF periods
period_size = n_bars // N_PERIODS
period_bounds = [(i * period_size, (i + 1) * period_size - 1) for i in range(N_PERIODS)]
period_bounds[-1] = (period_bounds[-1][0], n_bars - 1)

results = {}

# ============================================================
# Phase 1: Leave-One-Out Marginal Contribution (full sample)
# ============================================================
print()
print("=" * 70)
print("Phase 1: Leave-One-Out Marginal Contribution (270d)")
print("=" * 70)
print()

baseline_trades = portfolio_from_patterns(ALL_PATTERNS)
baseline_stats = eval_full(baseline_trades)
print(f"  Baseline (52): PnL={baseline_stats['pnl']:+.1f}% T={baseline_stats['trades']} "
      f"WR={baseline_stats['wr']}% MDD={baseline_stats['mdd']:.1f}% PnL/MDD={baseline_stats['pnl_mdd']:.1f}x")
print()

loo_results = []
for pat in ALL_PATTERNS:
    reduced = [p for p in ALL_PATTERNS if p != pat]
    trades = portfolio_from_patterns(reduced)
    stats = eval_full(trades)
    # Marginal contribution = baseline - reduced (positive = pattern helps)
    pnl_contrib = baseline_stats['pnl'] - stats['pnl']
    mdd_contrib = stats['mdd'] - baseline_stats['mdd']  # positive = removing worsens MDD
    trade_diff = baseline_stats['trades'] - stats['trades']

    loo_results.append({
        'pattern': pat,
        'direction': PORTFOLIO[pat][0],
        'pnl_contribution': round(pnl_contrib, 2),
        'mdd_impact': round(mdd_contrib, 2),  # positive = pattern helps MDD
        'trade_diff': trade_diff,
        'reduced_pnl': stats['pnl'],
        'reduced_mdd': stats['mdd'],
        'reduced_wr': stats['wr'],
        'reduced_trades': stats['trades'],
        'reduced_pnl_mdd': stats['pnl_mdd'],
    })

# Sort by PnL contribution
loo_results.sort(key=lambda x: x['pnl_contribution'])

# Negative contributors (removing HELPS PnL)
neg_contrib = [r for r in loo_results if r['pnl_contribution'] < 0]
pos_contrib = [r for r in loo_results if r['pnl_contribution'] >= 0]

print(f"  Negative PnL contributors (removing helps): {len(neg_contrib)}")
print(f"  Positive PnL contributors (removing hurts):  {len(pos_contrib)}")
print()

print("  Patterns HURTING portfolio (sorted by impact, worst first):")
print(f"  {'Pattern':<14} {'Dir':<6} {'PnL_Contrib':>12} {'MDD_Impact':>11} {'Without_PnL':>12} {'Without_MDD':>12} {'PnL/MDD':>8}")
print("  " + "-" * 85)
for r in loo_results[:15]:
    print(f"  {r['pattern']:<14} {r['direction']:<6} {r['pnl_contribution']:>+11.1f}% "
          f"{r['mdd_impact']:>+10.1f}% {r['reduced_pnl']:>+11.1f}% {r['reduced_mdd']:>11.1f}% "
          f"{r['reduced_pnl_mdd']:>7.1f}x")

print()
print("  Patterns HELPING portfolio most (sorted by contribution):")
print(f"  {'Pattern':<14} {'Dir':<6} {'PnL_Contrib':>12} {'MDD_Impact':>11}")
print("  " + "-" * 50)
for r in sorted(loo_results, key=lambda x: x['pnl_contribution'], reverse=True)[:10]:
    print(f"  {r['pattern']:<14} {r['direction']:<6} {r['pnl_contribution']:>+11.1f}% "
          f"{r['mdd_impact']:>+10.1f}%")

results['phase1_loo'] = loo_results
results['phase1_baseline'] = baseline_stats

# ============================================================
# Phase 2: Greedy Backward Elimination
# ============================================================
print()
print("=" * 70)
print("Phase 2: Greedy Backward Elimination (full sample + WF)")
print("=" * 70)
print()

# 2a. Full sample backward elimination
print("  2a. Full sample backward elimination path:")
print()

current_patterns = list(ALL_PATTERNS)
elimination_path = []

for step in range(len(ALL_PATTERNS) - 1):
    best_remove = None
    best_pnl = -999999

    for pat in current_patterns:
        reduced = [p for p in current_patterns if p != pat]
        trades = portfolio_from_patterns(reduced)
        pnl = eval_pnl(trades)
        if pnl > best_pnl:
            best_pnl = pnl
            best_remove = pat

    current_patterns.remove(best_remove)
    trades = portfolio_from_patterns(current_patterns)
    stats = eval_full(trades)

    elimination_path.append({
        'step': step + 1,
        'removed': best_remove,
        'remaining': len(current_patterns),
        'pnl': stats['pnl'],
        'trades': stats['trades'],
        'wr': stats['wr'],
        'mdd': stats['mdd'],
        'pnl_mdd': stats['pnl_mdd'],
    })

    if step < 20 or step % 10 == 0 or len(current_patterns) <= 5:
        print(f"  Step {step+1:>2}: Remove {best_remove:<14} → {len(current_patterns)} pats, "
              f"PnL={stats['pnl']:>+7.1f}% MDD={stats['mdd']:>5.1f}% PnL/MDD={stats['pnl_mdd']:>5.1f}x")

# Find optimal size
best_pnl_idx = max(range(len(elimination_path)), key=lambda i: elimination_path[i]['pnl'])
best_pnlmdd_idx = max(range(len(elimination_path)), key=lambda i: elimination_path[i]['pnl_mdd'])

print()
print(f"  Best PnL at step {best_pnl_idx + 1}: {elimination_path[best_pnl_idx]['remaining']} patterns, "
      f"PnL={elimination_path[best_pnl_idx]['pnl']:+.1f}%")
print(f"  Best PnL/MDD at step {best_pnlmdd_idx + 1}: {elimination_path[best_pnlmdd_idx]['remaining']} patterns, "
      f"PnL/MDD={elimination_path[best_pnlmdd_idx]['pnl_mdd']:.1f}x")

# Determine patterns removed for best PnL and best PnL/MDD
removed_for_best_pnl = [ep['removed'] for ep in elimination_path[:best_pnl_idx + 1]]
removed_for_best_pnlmdd = [ep['removed'] for ep in elimination_path[:best_pnlmdd_idx + 1]]
best_pnl_set = [p for p in ALL_PATTERNS if p not in removed_for_best_pnl]
best_pnlmdd_set = [p for p in ALL_PATTERNS if p not in removed_for_best_pnlmdd]

results['phase2_elimination_path'] = elimination_path
results['phase2_best_pnl'] = {
    'step': best_pnl_idx + 1,
    'n_patterns': len(best_pnl_set),
    'removed': removed_for_best_pnl,
    'remaining': best_pnl_set,
    'stats': elimination_path[best_pnl_idx],
}
results['phase2_best_pnlmdd'] = {
    'step': best_pnlmdd_idx + 1,
    'n_patterns': len(best_pnlmdd_set),
    'removed': removed_for_best_pnlmdd,
    'remaining': best_pnlmdd_set,
    'stats': elimination_path[best_pnlmdd_idx],
}

# ============================================================
# Phase 3: Greedy Forward Selection
# ============================================================
print()
print("=" * 70)
print("Phase 3: Greedy Forward Selection (full sample)")
print("=" * 70)
print()

available = list(ALL_PATTERNS)
selected = []
forward_path = []

for step in range(len(ALL_PATTERNS)):
    best_add = None
    best_pnl = -999999

    for pat in available:
        candidate = selected + [pat]
        trades = portfolio_from_patterns(candidate)
        pnl = eval_pnl(trades)
        if pnl > best_pnl:
            best_pnl = pnl
            best_add = pat

    selected.append(best_add)
    available.remove(best_add)
    trades = portfolio_from_patterns(selected)
    stats = eval_full(trades)

    forward_path.append({
        'step': step + 1,
        'added': best_add,
        'n_patterns': len(selected),
        'pnl': stats['pnl'],
        'trades': stats['trades'],
        'wr': stats['wr'],
        'mdd': stats['mdd'],
        'pnl_mdd': stats['pnl_mdd'],
    })

    if step < 20 or step % 10 == 0:
        print(f"  Step {step+1:>2}: Add {best_add:<14} → {len(selected)} pats, "
              f"PnL={stats['pnl']:>+7.1f}% MDD={stats['mdd']:>5.1f}% PnL/MDD={stats['pnl_mdd']:>5.1f}x")

# Find optimal size
fwd_best_pnl_idx = max(range(len(forward_path)), key=lambda i: forward_path[i]['pnl'])
fwd_best_pnlmdd_idx = max(range(len(forward_path)), key=lambda i: forward_path[i]['pnl_mdd'])

fwd_best_pnl_set = [fp['added'] for fp in forward_path[:fwd_best_pnl_idx + 1]]
fwd_best_pnlmdd_set = [fp['added'] for fp in forward_path[:fwd_best_pnlmdd_idx + 1]]

print()
print(f"  Best PnL at step {fwd_best_pnl_idx + 1}: {len(fwd_best_pnl_set)} patterns, "
      f"PnL={forward_path[fwd_best_pnl_idx]['pnl']:+.1f}%")
print(f"  Best PnL/MDD at step {fwd_best_pnlmdd_idx + 1}: {len(fwd_best_pnlmdd_set)} patterns, "
      f"PnL/MDD={forward_path[fwd_best_pnlmdd_idx]['pnl_mdd']:.1f}x")

results['phase3_forward_path'] = forward_path
results['phase3_best_pnl'] = {
    'n_patterns': len(fwd_best_pnl_set),
    'patterns': fwd_best_pnl_set,
    'stats': forward_path[fwd_best_pnl_idx],
}
results['phase3_best_pnlmdd'] = {
    'n_patterns': len(fwd_best_pnlmdd_set),
    'patterns': fwd_best_pnlmdd_set,
    'stats': forward_path[fwd_best_pnlmdd_idx],
}

# ============================================================
# Phase 4: Portfolio Size Sweep (Random Subsets)
# ============================================================
print()
print("=" * 70)
print("Phase 4: Portfolio Size Sweep (Random Subsets)")
print("=" * 70)
print()

SIZE_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 52]
size_results = {}

for size in SIZE_LEVELS:
    pnls = []
    mdds = []
    wrs = []
    pnl_mdds = []

    n_samples = 1 if size == 52 else RANDOM_SUBSETS
    for _ in range(n_samples):
        if size == 52:
            subset = ALL_PATTERNS
        else:
            subset = list(np.random.choice(ALL_PATTERNS, size=size, replace=False))
        trades = portfolio_from_patterns(subset)
        stats = eval_full(trades)
        pnls.append(stats['pnl'])
        mdds.append(stats['mdd'])
        wrs.append(stats['wr'])
        pnl_mdds.append(stats['pnl_mdd'])

    size_results[size] = {
        'pnl_mean': round(np.mean(pnls), 1),
        'pnl_std': round(np.std(pnls), 1),
        'pnl_p25': round(np.percentile(pnls, 25), 1),
        'pnl_p75': round(np.percentile(pnls, 75), 1),
        'pnl_max': round(np.max(pnls), 1),
        'mdd_mean': round(np.mean(mdds), 1),
        'mdd_std': round(np.std(mdds), 1),
        'wr_mean': round(np.mean(wrs), 1),
        'pnl_mdd_mean': round(np.mean(pnl_mdds), 1),
        'pnl_mdd_max': round(np.max(pnl_mdds), 1),
    }

print(f"  {'Size':>5} {'PnL_Mean':>9} {'PnL_Std':>9} {'PnL_P25':>9} {'PnL_P75':>9} {'PnL_Max':>9} {'MDD_Mean':>9} {'PnL/MDD':>8}")
print("  " + "-" * 75)
for size in SIZE_LEVELS:
    r = size_results[size]
    print(f"  {size:>5} {r['pnl_mean']:>+8.1f}% {r['pnl_std']:>8.1f}% "
          f"{r['pnl_p25']:>+8.1f}% {r['pnl_p75']:>+8.1f}% {r['pnl_max']:>+8.1f}% "
          f"{r['mdd_mean']:>8.1f}% {r['pnl_mdd_mean']:>7.1f}x")

results['phase4_size_sweep'] = size_results

# ============================================================
# Phase 5: WF Comparison of Best Strategies
# ============================================================
print()
print("=" * 70)
print("Phase 5: Walk-Forward Comparison of Best Strategies")
print("=" * 70)
print()

# Define strategies to compare
strategies = {
    'A_all_52': ALL_PATTERNS,
    'B_bwd_bestPnL': best_pnl_set,
    'C_bwd_bestPnLMDD': best_pnlmdd_set,
    'D_fwd_bestPnL': fwd_best_pnl_set,
    'E_fwd_bestPnLMDD': fwd_best_pnlmdd_set,
}

# Remove duplicates (if same set)
unique_strategies = {}
seen_sets = set()
for name, pats in strategies.items():
    key = frozenset(pats)
    if key not in seen_sets:
        unique_strategies[name] = pats
        seen_sets.add(key)
    else:
        print(f"  Skipping {name} (identical to previous)")

# 5a. WF with FIXED pattern sets (in-sample optimized, test OOS)
print(f"\n  5a. Fixed pattern sets → WF OOS (potential overfit)")
print()

phase5_fixed = {}
for strat_name, pat_list in unique_strategies.items():
    fold_results = []
    all_oos_trades = []

    for fold_idx in range(1, N_PERIODS):
        oos_start = period_bounds[fold_idx][0]
        oos_end = period_bounds[fold_idx][1]
        oos_trades = portfolio_from_patterns(pat_list, bar_min=oos_start, bar_max=oos_end)
        stats = eval_full(oos_trades)
        fold_results.append(stats)
        all_oos_trades.extend(oos_trades)

    agg = eval_full(all_oos_trades)
    oos_mc = mc_test([t[2] for t in all_oos_trades])
    phase5_fixed[strat_name] = {
        'n_patterns': len(pat_list),
        'folds': fold_results,
        'aggregate': agg,
        'oos_mc': round(oos_mc, 4),
    }

print(f"  {'Strategy':<22} {'N':>4} {'OOS_PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8} {'MC':>8}")
print("  " + "-" * 85)
for name, data in phase5_fixed.items():
    a = data['aggregate']
    print(f"  {name:<22} {data['n_patterns']:>3} {a['pnl']:>+8.1f}% {a['trades']:>6} "
          f"{a['wr']:>5.1f}% {a['mdd']:>6.1f}% {a['pf']:>5.2f} {a['pnl_mdd']:>7.1f}x {data['oos_mc']:>7.4f}")

# 5b. Per-fold backward elimination (TRUE WF — no future info)
print(f"\n  5b. Per-fold backward elimination (TRUE WF — train-time optimization)")
print()

true_wf_oos = []
true_wf_details = []

for fold_idx in range(1, N_PERIODS):
    train_end = period_bounds[fold_idx - 1][1]
    oos_start = period_bounds[fold_idx][0]
    oos_end = period_bounds[fold_idx][1]

    # Backward elimination on TRAINING data only
    train_patterns = list(ALL_PATTERNS)
    best_train_pnl = eval_pnl(portfolio_from_patterns(train_patterns, bar_max=train_end))
    removed_in_fold = []

    for step in range(20):  # max 20 removals per fold
        best_remove = None
        best_pnl_after = -999999

        for pat in train_patterns:
            reduced = [p for p in train_patterns if p != pat]
            trades = portfolio_from_patterns(reduced, bar_max=train_end)
            pnl = eval_pnl(trades)
            if pnl > best_pnl_after:
                best_pnl_after = pnl
                best_remove = pat

        if best_pnl_after <= best_train_pnl:
            break  # No more improvement
        train_patterns.remove(best_remove)
        removed_in_fold.append(best_remove)
        best_train_pnl = best_pnl_after

    # Test on OOS with optimized pattern set
    oos_trades = portfolio_from_patterns(train_patterns, bar_min=oos_start, bar_max=oos_end)
    oos_stats = eval_full(oos_trades)

    # Baseline OOS (all 52)
    base_oos = portfolio_from_patterns(ALL_PATTERNS, bar_min=oos_start, bar_max=oos_end)
    base_stats = eval_full(base_oos)

    true_wf_oos.extend(oos_trades)
    true_wf_details.append({
        'fold': fold_idx,
        'removed': removed_in_fold,
        'remaining': len(train_patterns),
        'oos_pnl': oos_stats['pnl'],
        'oos_trades': oos_stats['trades'],
        'oos_wr': oos_stats['wr'],
        'oos_mdd': oos_stats['mdd'],
        'base_pnl': base_stats['pnl'],
        'base_trades': base_stats['trades'],
        'diff': round(oos_stats['pnl'] - base_stats['pnl'], 2),
    })

    print(f"  Fold {fold_idx}: removed {len(removed_in_fold)} → {len(train_patterns)} pats | "
          f"OOS: {oos_stats['pnl']:+.1f}% vs base {base_stats['pnl']:+.1f}% (diff {oos_stats['pnl'] - base_stats['pnl']:+.1f}%)")
    if removed_in_fold:
        print(f"         Removed: {', '.join(removed_in_fold[:10])}")

# True WF aggregate
true_wf_agg = eval_full(true_wf_oos)
true_wf_mc = mc_test([t[2] for t in true_wf_oos])

# Baseline WF aggregate
base_all_oos = []
for fold_idx in range(1, N_PERIODS):
    oos_start = period_bounds[fold_idx][0]
    oos_end = period_bounds[fold_idx][1]
    base_all_oos.extend(portfolio_from_patterns(ALL_PATTERNS, bar_min=oos_start, bar_max=oos_end))
base_wf_agg = eval_full(base_all_oos)

print(f"\n  True WF Aggregate:")
print(f"    Baseline 52:    PnL={base_wf_agg['pnl']:+.1f}% T={base_wf_agg['trades']} WR={base_wf_agg['wr']}% MDD={base_wf_agg['mdd']:.1f}%")
print(f"    BWD-optimized:  PnL={true_wf_agg['pnl']:+.1f}% T={true_wf_agg['trades']} WR={true_wf_agg['wr']}% MDD={true_wf_agg['mdd']:.1f}% MC={true_wf_mc:.4f}")
print(f"    Diff:           PnL={true_wf_agg['pnl'] - base_wf_agg['pnl']:+.1f}%")

folds_improved = sum(1 for d in true_wf_details if d['diff'] > 0)
folds_worse = sum(1 for d in true_wf_details if d['diff'] < 0)
print(f"    Folds improved: {folds_improved}/{N_PERIODS - 1}")
print(f"    Folds worse:    {folds_worse}/{N_PERIODS - 1}")

results['phase5_fixed'] = phase5_fixed
results['phase5_true_wf'] = {
    'details': true_wf_details,
    'aggregate': {**eval_full(true_wf_oos), 'mc': round(true_wf_mc, 4)},
    'baseline_aggregate': base_wf_agg,
}

# ============================================================
# Save
# ============================================================
out_path = Path(__file__).resolve().parent / '../../results/pattern_combination_research.json'
results['timestamp'] = datetime.now().isoformat()
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n\u2705 Saved: {out_path}")

# ============================================================
# Final Summary
# ============================================================
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print()
print(f"  Phase 1: LOO — {len(neg_contrib)} negative contributors, {len(pos_contrib)} positive")
print(f"  Phase 2: BWD elimination — Best PnL at {len(best_pnl_set)} patterns ({baseline_stats['pnl']:+.1f}→{elimination_path[best_pnl_idx]['pnl']:+.1f}%)")
print(f"           BWD elimination — Best PnL/MDD at {len(best_pnlmdd_set)} patterns ({baseline_stats['pnl_mdd']:.1f}→{elimination_path[best_pnlmdd_idx]['pnl_mdd']:.1f}x)")
print(f"  Phase 3: FWD selection — Best PnL at {len(fwd_best_pnl_set)} patterns ({forward_path[fwd_best_pnl_idx]['pnl']:+.1f}%)")
print(f"           FWD selection — Best PnL/MDD at {len(fwd_best_pnlmdd_set)} patterns ({forward_path[fwd_best_pnlmdd_idx]['pnl_mdd']:.1f}x)")
print(f"  Phase 4: Size sweep — optimal range around {SIZE_LEVELS[np.argmax([size_results[s]['pnl_mdd_mean'] for s in SIZE_LEVELS])]} patterns")
print(f"  Phase 5: TRUE WF BWD — OOS PnL {true_wf_agg['pnl']:+.1f}% vs baseline {base_wf_agg['pnl']:+.1f}% (diff {true_wf_agg['pnl'] - base_wf_agg['pnl']:+.1f}%)")
print(f"           Folds: {folds_improved} improved, {folds_worse} worse")
print("=" * 70)
