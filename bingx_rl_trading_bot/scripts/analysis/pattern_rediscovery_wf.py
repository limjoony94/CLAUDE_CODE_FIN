"""
Pattern-Rediscovery Walk-Forward
================================
270일 데이터로 발굴한 51패턴이 과적합된 것인지 검증.

핵심 논리적 결함:
  - 기존 WF는 패턴 목록을 270일 전체에서 고정하고 WF 수행 → pattern selection look-ahead
  - 270일 데이터에서 발굴한 패턴이 270일에서 잘 작동하는 건 당연

이 연구의 해결:
  - 각 fold의 훈련 기간에서 전체 3456 유니버스를 처음부터 재스캔
  - MC p < 0.01 통과 + edge > 5pp → "재발견"으로 인정
  - 재발견 패턴만으로 OOS 포트폴리오 테스트
  - 진정한 pattern-selection-unbiased OOS 성과 측정

Phases:
  1. Pre-computation: 패턴 인스턴스, 가격 데이터
  2. Per-fold rediscovery: 3456 유니버스 스캔 + grid search + MC
  3. Rediscovery analysis: 재발견 비율, watchlist vs non-watchlist
  4. Portfolio WF: 재발견 패턴만으로 OOS 성과
  5. Verdict: 진정한 미래 수익 기대치
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL, VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
MIN_TRADES = 10
MIN_EDGE_PP = 5.0  # excess WR over random walk
MC_THRESHOLD = 0.01

TP_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
CANDLE_TYPES = ['D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN']

# Watchlist from individual validation
WATCHLIST = ['MD-H-MD', 'MD-DN-MU', 'BD-BU-DN', 'U-MD-MD', 'BD-MU-BD', 'DN-D-BD', 'DN-MD-DN']
wl_set = set(WATCHLIST)

print("=" * 80)
print("Pattern-Rediscovery Walk-Forward")
print("  핵심: 각 fold에서 패턴을 처음부터 재발굴하여 진정한 OOS 검증")
print("=" * 80)

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
n_bars = len(df)
bars_per_day = 288
total_days = n_bars // bars_per_day
print(f"\nLoaded: {n_bars} bars ({total_days} days)")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)
print("Classification complete.")

# ============================================================
# Build current pattern maps
# ============================================================
current_patterns = {}
for p in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[p]
    current_patterns[p] = ('LONG', tp, sl)
for p in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[p]
    current_patterns[p] = ('SHORT', tp, sl)

current_pat_set = set(current_patterns.keys())
current_pat_directions = {p: v[0] for p, v in current_patterns.items()}
print(f"Current patterns: {len(current_patterns)} (v1.27.2)")

# ============================================================
# Phase 1: Pre-compute all pattern instances
# ============================================================
print("\n" + "=" * 80)
print("PHASE 1: Pre-computing pattern instances")
print("=" * 80)

t0 = time.time()

# Map: pattern_key → [signal_bar_indices]
all_instances = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    all_instances[pat].append(i)

unique_patterns = len(all_instances)
total_instances = sum(len(v) for v in all_instances.values())
print(f"  Unique 3-candle patterns in data: {unique_patterns}")
print(f"  Total instances: {total_instances}")
print(f"  Pre-computation: {time.time()-t0:.1f}s")


# ============================================================
# Engine functions
# ============================================================
def evaluate_pattern_grid(instances, direction, start_bar, end_bar):
    """
    Grid search TP/SL for a pattern on given data range.
    Pre-computes favorable/adverse price moves per instance for efficiency.
    Returns best TP/SL and trade data, or None if insufficient trades.
    """
    valid = [i for i in instances if start_bar <= i < end_bar and i + 1 < n_bars]
    if len(valid) < MIN_TRADES:
        return None

    # For each grid point, collect trade PnLs
    grid_results = {}  # (tp, sl) → {'pnls': [], 'exit_bars': []}

    for tp in TP_GRID:
        for sl in SL_GRID:
            grid_results[(tp, sl)] = {'pnls': [], 'exit_bars': []}

    for idx in valid:
        entry = opens[idx + 1]
        if entry <= 0:
            continue

        end = min(idx + 2 + MAX_BARS, n_bars)
        if idx + 2 >= end:
            continue

        # Pre-compute favorable/adverse moves
        bar_range = slice(idx + 2, end)
        if direction == 'LONG':
            fav = (highs[bar_range] - entry) / entry * 100
            adv = (entry - lows[bar_range]) / entry * 100
        else:
            fav = (entry - lows[bar_range]) / entry * 100
            adv = (highs[bar_range] - entry) / entry * 100

        # Pre-compute first hit bar for each TP and SL level
        tp_bars = {}
        for tp in TP_GRID:
            mask = fav >= tp
            tp_bars[tp] = int(np.argmax(mask)) if np.any(mask) else MAX_BARS + 1

        sl_bars = {}
        for sl in SL_GRID:
            mask = adv >= sl
            sl_bars[sl] = int(np.argmax(mask)) if np.any(mask) else MAX_BARS + 1

        # Evaluate all grid combinations
        for tp in TP_GRID:
            tb = tp_bars[tp]
            for sl in SL_GRID:
                sb = sl_bars[sl]
                if tb < sb:
                    pnl = tp * LEVERAGE - FEE_PCT
                    exit_bar = idx + 2 + tb
                elif sb < tb:
                    pnl = -sl * LEVERAGE - FEE_PCT
                    exit_bar = idx + 2 + sb
                elif tb <= MAX_BARS:
                    # Same bar — distance-based
                    if tp <= sl:
                        pnl = tp * LEVERAGE - FEE_PCT
                    else:
                        pnl = -sl * LEVERAGE - FEE_PCT
                    exit_bar = idx + 2 + tb
                else:
                    pnl = 0
                    exit_bar = idx + 2 + MAX_BARS
                grid_results[(tp, sl)]['pnls'].append(pnl)
                grid_results[(tp, sl)]['exit_bars'].append(exit_bar)

    # Find best TP/SL by total PnL (with minimum trades filter)
    best_pnl = -999
    best_config = None
    for (tp, sl), data in grid_results.items():
        if len(data['pnls']) < MIN_TRADES:
            continue
        total = sum(data['pnls'])
        if total > best_pnl:
            best_pnl = total
            best_config = (tp, sl, data['pnls'], data['exit_bars'])

    if best_config is None:
        return None

    tp, sl, pnls, exit_bars = best_config
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    rw_wr = sl / (tp + sl) * 100  # random walk WR
    excess_wr = wr - rw_wr

    return {
        'tp': tp, 'sl': sl,
        'pnl': round(best_pnl, 2),
        'trades': len(pnls),
        'wr': round(wr, 1),
        'rw_wr': round(rw_wr, 1),
        'excess_wr': round(excess_wr, 1),
        'trade_pnls': pnls,
        'exit_bars': exit_bars,
    }


def mc_test(pnls, n_sims=MC_SIMS):
    """Sign randomization MC test."""
    pnls = np.array(pnls)
    real = np.sum(pnls)
    if real <= 0:
        return 1.0
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls)))
    sims = np.sum(pnls * signs, axis=1)
    return float(np.mean(sims >= real))


def collect_trades_1pos(pattern_map, start_bar=0, end_bar=None):
    """1-position-at-a-time portfolio simulation."""
    if end_bar is None:
        end_bar = n_bars
    raw_trades = []
    for i in range(max(2, start_bar), min(end_bar, n_bars)):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
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
                raw_trades.append((i + 1, j, pnl, pat))
                break
            elif ht:
                raw_trades.append((i + 1, j, tp * LEVERAGE - FEE_PCT, pat))
                break
            elif hs:
                raw_trades.append((i + 1, j, -sl * LEVERAGE - FEE_PCT, pat))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl, pat in raw_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl, pat))
            last_exit = xb
    return filtered


def eval_portfolio(trades):
    """Calculate portfolio metrics."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wps = []; lps = []
    for p in pnl_list:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0: wins += 1; wps.append(p)
        else: lps.append(abs(p))
    tw = sum(wps); tl = sum(lps)
    return {
        'pnl': round(cum, 2), 'trades': len(pnl_list),
        'wr': round(wins / len(pnl_list) * 100, 2) if pnl_list else 0,
        'mdd': round(mdd, 2),
        'pf': round(tw / tl, 2) if tl > 0 else 999,
        'pnl_mdd': round(cum / mdd, 2) if mdd > 0 else 999,
    }


# ============================================================
# Phase 2: Per-Fold Rediscovery Scan
# ============================================================
print("\n" + "=" * 80)
print("PHASE 2: Per-Fold Pattern Rediscovery (Full 3456 Universe Scan)")
print("=" * 80)

np.random.seed(42)

fold_days = total_days // 5  # ~54 days
n_folds = 4

# Build full universe: all 1728 patterns × 2 directions
universe = []
for t1 in CANDLE_TYPES:
    for t2 in CANDLE_TYPES:
        for t3 in CANDLE_TYPES:
            pat = f"{t1}-{t2}-{t3}"
            for direction in ['LONG', 'SHORT']:
                universe.append((pat, direction))

print(f"  Universe size: {len(universe)} pattern-directions")
print(f"  Fold size: {fold_days} days, OOS folds: {n_folds}")

# Track rediscovery across folds
# rediscovery_matrix[pat] = {fold_i: {discovered: bool, tp, sl, mc_p, ...}}
rediscovery_matrix = defaultdict(dict)
fold_results = []

# Track ALL discovered patterns per fold (including new ones not in our 51)
all_discovered_per_fold = []

for fold_i in range(n_folds):
    train_end_bar = (fold_i + 1) * fold_days * bars_per_day
    oos_start_bar = train_end_bar
    oos_end_bar = min((fold_i + 2) * fold_days * bars_per_day, n_bars)
    train_days = (fold_i + 1) * fold_days

    print(f"\n  Fold {fold_i+1}: Train {train_days}d (bars 0-{train_end_bar}) → "
          f"OOS {fold_days}d (bars {oos_start_bar}-{oos_end_bar})")

    t_fold = time.time()
    discovered = {}  # pat → {direction, tp, sl, mc_p, ...}
    scanned = 0
    passed_grid = 0
    passed_edge = 0
    passed_mc = 0

    for pat, direction in universe:
        instances = all_instances.get(pat, [])
        if not instances:
            continue
        scanned += 1

        # Grid search on training data
        result = evaluate_pattern_grid(instances, direction, 0, train_end_bar)
        if result is None:
            continue
        passed_grid += 1

        # Edge check: excess WR > 5pp
        if result['excess_wr'] < MIN_EDGE_PP:
            continue
        passed_edge += 1

        # MC test
        mc_p = mc_test(result['trade_pnls'])
        if mc_p >= MC_THRESHOLD:
            continue
        passed_mc += 1

        key = f"{pat}_{direction}"
        discovered[key] = {
            'pattern': pat,
            'direction': direction,
            'tp': result['tp'],
            'sl': result['sl'],
            'pnl': result['pnl'],
            'trades': result['trades'],
            'wr': result['wr'],
            'excess_wr': result['excess_wr'],
            'mc_p': round(mc_p, 6),
        }

        # Check if this is one of our current 51
        if pat in current_pat_directions and current_pat_directions[pat] == direction:
            rediscovery_matrix[pat][fold_i] = {
                'discovered': True,
                'tp': result['tp'], 'sl': result['sl'],
                'mc_p': round(mc_p, 6),
                'excess_wr': result['excess_wr'],
                'trades': result['trades'],
            }

    # Mark non-discovered current patterns
    for pat, direction in current_patterns.items():
        dir_str = direction[0] if isinstance(direction, tuple) else direction
        if isinstance(direction, tuple):
            dir_str = direction[0]
        else:
            dir_str = current_patterns[pat][0]
        if fold_i not in rediscovery_matrix.get(pat, {}):
            rediscovery_matrix[pat][fold_i] = {'discovered': False}

    all_discovered_per_fold.append(discovered)

    # Count how many of our 51 were rediscovered
    n_current_rediscovered = sum(
        1 for p in current_patterns
        if rediscovery_matrix[p].get(fold_i, {}).get('discovered', False)
    )

    # Count new patterns (not in our 51)
    current_keys = set()
    for p, (d, tp, sl) in current_patterns.items():
        current_keys.add(f"{p}_{d}")
    n_new = sum(1 for k in discovered if k not in current_keys)

    elapsed = time.time() - t_fold
    print(f"    Scanned: {scanned}, Grid pass: {passed_grid}, "
          f"Edge pass: {passed_edge}, MC pass: {passed_mc}")
    print(f"    Current 51 rediscovered: {n_current_rediscovered}/51 "
          f"({n_current_rediscovered/51*100:.1f}%)")
    print(f"    New patterns (not in 51): {n_new}")
    print(f"    Time: {elapsed:.1f}s")

    fold_results.append({
        'fold': fold_i + 1,
        'train_days': train_days,
        'scanned': scanned,
        'passed_grid': passed_grid,
        'passed_edge': passed_edge,
        'passed_mc': passed_mc,
        'current_rediscovered': n_current_rediscovered,
        'new_patterns': n_new,
        'total_discovered': passed_mc,
    })


# ============================================================
# Phase 3: Rediscovery Analysis
# ============================================================
print("\n" + "=" * 80)
print("PHASE 3: Rediscovery Analysis")
print("=" * 80)

# Per-pattern rediscovery rate
pattern_rediscovery = {}
for pat in current_patterns:
    folds_discovered = [
        fold_i for fold_i in range(n_folds)
        if rediscovery_matrix[pat].get(fold_i, {}).get('discovered', False)
    ]
    rate = len(folds_discovered) / n_folds
    pattern_rediscovery[pat] = {
        'direction': current_patterns[pat][0],
        'tp': current_patterns[pat][1],
        'sl': current_patterns[pat][2],
        'folds_discovered': folds_discovered,
        'rediscovery_rate': rate,
        'is_watchlist': pat in wl_set,
    }

# Sort by rediscovery rate
sorted_patterns = sorted(pattern_rediscovery.items(), key=lambda x: x[1]['rediscovery_rate'])

# Never-rediscovered patterns
never = [p for p, d in sorted_patterns if d['rediscovery_rate'] == 0]
partial = [p for p, d in sorted_patterns if 0 < d['rediscovery_rate'] < 1.0]
always = [p for p, d in sorted_patterns if d['rediscovery_rate'] == 1.0]

print(f"\n  Rediscovery Summary:")
print(f"    Never rediscovered (0/4 folds):  {len(never)} patterns")
print(f"    Partially rediscovered (1-3/4):  {len(partial)} patterns")
print(f"    Always rediscovered (4/4 folds): {len(always)} patterns")

# Watchlist vs non-watchlist comparison
wl_rates = [d['rediscovery_rate'] for p, d in pattern_rediscovery.items() if d['is_watchlist']]
non_wl_rates = [d['rediscovery_rate'] for p, d in pattern_rediscovery.items() if not d['is_watchlist']]

print(f"\n  Watchlist ({len(wl_rates)}):     avg rediscovery rate = {np.mean(wl_rates):.2f}")
print(f"  Non-watchlist ({len(non_wl_rates)}): avg rediscovery rate = {np.mean(non_wl_rates):.2f}")

# Detailed per-pattern table
print(f"\n  {'Pattern':<14} {'Dir':<6} {'TP/SL':<8} {'Rate':>5} {'Folds':<12} {'WL':>3}")
print(f"  {'-'*14} {'-'*6} {'-'*8} {'-'*5} {'-'*12} {'-'*3}")

for pat, d in sorted_patterns:
    folds_str = ','.join(str(f+1) for f in d['folds_discovered']) if d['folds_discovered'] else '-'
    wl_mark = '***' if d['is_watchlist'] else ''
    rate_str = f"{d['rediscovery_rate']:.0%}"
    tp_sl = f"{d['tp']}/{d['sl']}"
    print(f"  {pat:<14} {d['direction']:<6} {tp_sl:<8} {rate_str:>5} {folds_str:<12} {wl_mark:>3}")

# Look-ahead dependency
n_lookahead = sum(1 for p, d in pattern_rediscovery.items() if d['rediscovery_rate'] < 1.0)
print(f"\n  Look-ahead dependent (rate < 100%): {n_lookahead}/51 ({n_lookahead/51*100:.1f}%)")

# Fold-by-fold TP/SL comparison for rediscovered patterns
print(f"\n  TP/SL Stability Check (rediscovered patterns, training-optimized vs current):")
tp_sl_matches = 0
tp_sl_total = 0
for pat in current_patterns:
    for fold_i in range(n_folds):
        fold_data = rediscovery_matrix[pat].get(fold_i, {})
        if fold_data.get('discovered', False):
            tp_sl_total += 1
            current_tp = current_patterns[pat][1]
            current_sl = current_patterns[pat][2]
            fold_tp = fold_data['tp']
            fold_sl = fold_data['sl']
            if fold_tp == current_tp and fold_sl == current_sl:
                tp_sl_matches += 1

if tp_sl_total > 0:
    print(f"    TP/SL exact matches: {tp_sl_matches}/{tp_sl_total} ({tp_sl_matches/tp_sl_total*100:.1f}%)")
    print(f"    (Low match rate → TP/SL optimization also has look-ahead)")


# ============================================================
# Phase 4: Portfolio WF with Rediscovered Patterns
# ============================================================
print("\n" + "=" * 80)
print("PHASE 4: Portfolio Walk-Forward — Rediscovered vs All-51")
print("=" * 80)

# For each fold, build 3 portfolios and test on OOS:
#   A. All 51 with current TP/SL (baseline — has look-ahead)
#   B. Rediscovered only with TRAINING-optimized TP/SL (no look-ahead)
#   C. Rediscovered only with CURRENT TP/SL (partial look-ahead: pattern selection clean, TP/SL from 270d)

wf_oos_results = []

oos_pnl_A = 0
oos_pnl_B = 0
oos_pnl_C = 0

for fold_i in range(n_folds):
    train_end_bar = (fold_i + 1) * fold_days * bars_per_day
    oos_start_bar = train_end_bar
    oos_end_bar = min((fold_i + 2) * fold_days * bars_per_day, n_bars)
    train_days = (fold_i + 1) * fold_days

    # A: All 51 current TP/SL
    map_A = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items()}
    trades_A = collect_trades_1pos(map_A, oos_start_bar, oos_end_bar)
    eval_A = eval_portfolio(trades_A)

    # B: Rediscovered only, training-optimized TP/SL
    map_B = {}
    for pat, (direction, current_tp, current_sl) in current_patterns.items():
        fold_data = rediscovery_matrix[pat].get(fold_i, {})
        if fold_data.get('discovered', False):
            map_B[pat] = (direction, fold_data['tp'], fold_data['sl'])
    trades_B = collect_trades_1pos(map_B, oos_start_bar, oos_end_bar)
    eval_B = eval_portfolio(trades_B)

    # C: Rediscovered only, current TP/SL
    map_C = {}
    for pat, (direction, current_tp, current_sl) in current_patterns.items():
        fold_data = rediscovery_matrix[pat].get(fold_i, {})
        if fold_data.get('discovered', False):
            map_C[pat] = (direction, current_tp, current_sl)
    trades_C = collect_trades_1pos(map_C, oos_start_bar, oos_end_bar)
    eval_C = eval_portfolio(trades_C)

    n_rediscovered = len(map_B)

    print(f"\n  Fold {fold_i+1} (Train {train_days}d → OOS {fold_days}d): "
          f"Rediscovered {n_rediscovered}/51")
    print(f"    A (all 51, current TP/SL):    PnL {eval_A['pnl']:>+8.1f}%, "
          f"Trades {eval_A['trades']:>3}, WR {eval_A['wr']:>5.1f}%, MDD {eval_A['mdd']:>5.1f}%")
    print(f"    B (rediscovered, train TP/SL): PnL {eval_B['pnl']:>+8.1f}%, "
          f"Trades {eval_B['trades']:>3}, WR {eval_B['wr']:>5.1f}%, MDD {eval_B['mdd']:>5.1f}%")
    print(f"    C (rediscovered, curr TP/SL):  PnL {eval_C['pnl']:>+8.1f}%, "
          f"Trades {eval_C['trades']:>3}, WR {eval_C['wr']:>5.1f}%, MDD {eval_C['mdd']:>5.1f}%")

    delta_BA = eval_B['pnl'] - eval_A['pnl']
    delta_CA = eval_C['pnl'] - eval_A['pnl']
    print(f"    B vs A: {delta_BA:+.1f}%  |  C vs A: {delta_CA:+.1f}%")

    oos_pnl_A += eval_A['pnl']
    oos_pnl_B += eval_B['pnl']
    oos_pnl_C += eval_C['pnl']

    wf_oos_results.append({
        'fold': fold_i + 1,
        'train_days': train_days,
        'n_rediscovered': n_rediscovered,
        'A_all51': eval_A,
        'B_rediscovered_train_tpsl': eval_B,
        'C_rediscovered_curr_tpsl': eval_C,
    })

print(f"\n  === OOS TOTALS ===")
print(f"  A (all 51, current TP/SL):    {oos_pnl_A:+.1f}%")
print(f"  B (rediscovered, train TP/SL): {oos_pnl_B:+.1f}%")
print(f"  C (rediscovered, curr TP/SL):  {oos_pnl_C:+.1f}%")
print(f"  Look-ahead gap (A - C): {oos_pnl_A - oos_pnl_C:+.1f}% "
      f"({(oos_pnl_A - oos_pnl_C)/oos_pnl_A*100:.1f}% of A)")
print(f"  TP/SL look-ahead gap (C - B): {oos_pnl_C - oos_pnl_B:+.1f}%")


# ============================================================
# Phase 5: Watchlist-Specific Analysis
# ============================================================
print("\n" + "=" * 80)
print("PHASE 5: Watchlist Pattern Forward Survival")
print("=" * 80)

print(f"\n  {'Pattern':<14} {'Dir':<6} {'Rate':>5} {'Fold1':>6} {'Fold2':>6} {'Fold3':>6} {'Fold4':>6}")
print(f"  {'-'*14} {'-'*6} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

for pat in WATCHLIST:
    d = pattern_rediscovery[pat]
    row = f"  {pat:<14} {d['direction']:<6} {d['rediscovery_rate']:.0%}"
    for fold_i in range(n_folds):
        fold_data = rediscovery_matrix[pat].get(fold_i, {})
        if fold_data.get('discovered', False):
            row += f"  {'PASS':>5}"
        else:
            row += f"  {'FAIL':>5}"
    print(row)

# Compare: what if we only used non-watchlist rediscovered patterns?
print(f"\n  Counterfactual: Remove watchlist that are never/rarely rediscovered")
rare_wl = [p for p in WATCHLIST if pattern_rediscovery[p]['rediscovery_rate'] <= 0.25]
print(f"  Rarely rediscovered watchlist (rate <= 25%): {len(rare_wl)}")
for p in rare_wl:
    print(f"    {p}: rate {pattern_rediscovery[p]['rediscovery_rate']:.0%}")


# ============================================================
# Phase 6: Cross-Validation of Look-Ahead Impact
# ============================================================
print("\n" + "=" * 80)
print("PHASE 6: Look-Ahead Quantification")
print("=" * 80)

# For each fold, compare # patterns available in A vs B/C
# And the OOS contribution of look-ahead-only patterns
for fold_i in range(n_folds):
    fold_data = wf_oos_results[fold_i]
    n_rd = fold_data['n_rediscovered']
    n_lookahead_only = 51 - n_rd

    pnl_A = fold_data['A_all51']['pnl']
    pnl_C = fold_data['C_rediscovered_curr_tpsl']['pnl']
    la_contribution = pnl_A - pnl_C

    print(f"  Fold {fold_i+1}: {n_rd} rediscovered + {n_lookahead_only} look-ahead-only "
          f"→ LA contribution: {la_contribution:+.1f}%")

# Average look-ahead pattern count
avg_rediscovered = np.mean([f['n_rediscovered'] for f in wf_oos_results])
avg_lookahead = 51 - avg_rediscovered
pct_lookahead = avg_lookahead / 51 * 100

print(f"\n  Average rediscovered per fold: {avg_rediscovered:.1f}/51 ({avg_rediscovered/51*100:.1f}%)")
print(f"  Average look-ahead dependent:  {avg_lookahead:.1f}/51 ({pct_lookahead:.1f}%)")

# Pattern selection look-ahead vs TP/SL look-ahead
total_la_pattern = oos_pnl_A - oos_pnl_C  # due to pattern selection
total_la_tpsl = oos_pnl_C - oos_pnl_B     # due to TP/SL optimization
total_genuine = oos_pnl_B                   # genuine forward value

print(f"\n  === LOOK-AHEAD DECOMPOSITION ===")
print(f"  Total OOS PnL (A, all 51): {oos_pnl_A:+.1f}%")
print(f"    ├─ Pattern selection look-ahead: {total_la_pattern:+.1f}% ({total_la_pattern/oos_pnl_A*100:.1f}%)")
print(f"    ├─ TP/SL optimization look-ahead: {total_la_tpsl:+.1f}% ({total_la_tpsl/oos_pnl_A*100:.1f}%)")
print(f"    └─ Genuine forward value: {total_genuine:+.1f}% ({total_genuine/oos_pnl_A*100:.1f}%)")


# ============================================================
# Final Verdict
# ============================================================
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

# Key metrics
genuine_pct = total_genuine / oos_pnl_A * 100 if oos_pnl_A > 0 else 0
pattern_la_pct = total_la_pattern / oos_pnl_A * 100 if oos_pnl_A > 0 else 0
tpsl_la_pct = total_la_tpsl / oos_pnl_A * 100 if oos_pnl_A > 0 else 0

# Are watchlist patterns more look-ahead dependent?
wl_avg_rate = np.mean(wl_rates)
non_wl_avg_rate = np.mean(non_wl_rates)
rate_diff = non_wl_avg_rate - wl_avg_rate

print(f"  1. Genuine forward value: {total_genuine:+.1f}% ({genuine_pct:.1f}% of total)")
print(f"     → {'POSITIVE' if total_genuine > 0 else 'NEGATIVE'}: "
      f"{'Strategy has real edge even without look-ahead' if total_genuine > 0 else 'Strategy depends entirely on look-ahead'}")

print(f"  2. Pattern selection look-ahead: {total_la_pattern:+.1f}% ({pattern_la_pct:.1f}%)")
print(f"  3. TP/SL look-ahead: {total_la_tpsl:+.1f}% ({tpsl_la_pct:.1f}%)")

print(f"  4. Watchlist rediscovery: {wl_avg_rate:.2f} vs Non-WL: {non_wl_avg_rate:.2f} "
      f"(diff: {rate_diff:+.2f})")
if rate_diff > 0.15:
    print(f"     → WATCHLIST MORE LOOK-AHEAD DEPENDENT (차이 {rate_diff:.2f})")
elif rate_diff > 0:
    print(f"     → Watchlist slightly more look-ahead dependent")
else:
    print(f"     → No systematic difference")

# Recommendations
print(f"\n  === RECOMMENDATIONS ===")

# Never-rediscovered patterns
if never:
    print(f"  REMOVAL CANDIDATES ({len(never)} never-rediscovered patterns):")
    for p in never:
        d = pattern_rediscovery[p]
        wl_mark = " [WATCHLIST]" if p in wl_set else ""
        print(f"    {p} ({d['direction']}, TP/SL {d['tp']}/{d['sl']}){wl_mark}")

# Rarely rediscovered (1/4)
rarely = [p for p, d in sorted_patterns if d['rediscovery_rate'] == 0.25]
if rarely:
    print(f"  WATCH CANDIDATES ({len(rarely)} patterns, 1/4 folds only):")
    for p in rarely:
        d = pattern_rediscovery[p]
        wl_mark = " [WATCHLIST]" if p in wl_set else ""
        print(f"    {p} ({d['direction']}, TP/SL {d['tp']}/{d['sl']}){wl_mark}")

# Always rediscovered
print(f"  ROBUST PATTERNS ({len(always)} always-rediscovered):")
print(f"    These {len(always)} patterns have genuine forward edge confirmed across all folds.")


# ============================================================
# Save results
# ============================================================
results = {
    'meta': {
        'script': 'pattern_rediscovery_wf.py',
        'version': 'v1.27.2',
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Pattern-Rediscovery Walk-Forward: re-scan full 3456 universe per fold',
        'parameters': {
            'mc_sims': MC_SIMS,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
            'min_edge_pp': MIN_EDGE_PP,
            'tp_grid': TP_GRID,
            'sl_grid': SL_GRID,
            'n_folds': n_folds,
            'fold_days': fold_days,
        },
    },
    'fold_scan_summary': fold_results,
    'rediscovery': {
        'per_pattern': {
            p: {
                'direction': d['direction'],
                'tp': d['tp'], 'sl': d['sl'],
                'rediscovery_rate': d['rediscovery_rate'],
                'folds_discovered': d['folds_discovered'],
                'is_watchlist': d['is_watchlist'],
            }
            for p, d in pattern_rediscovery.items()
        },
        'summary': {
            'never_rediscovered': never,
            'partially_rediscovered': partial,
            'always_rediscovered': always,
            'watchlist_avg_rate': round(float(wl_avg_rate), 4),
            'non_watchlist_avg_rate': round(float(non_wl_avg_rate), 4),
            'rate_difference': round(float(rate_diff), 4),
        },
        'tp_sl_stability': {
            'exact_matches': tp_sl_matches,
            'total_comparisons': tp_sl_total,
            'match_rate': round(tp_sl_matches / tp_sl_total * 100, 1) if tp_sl_total > 0 else 0,
        },
    },
    'portfolio_wf': {
        'folds': wf_oos_results,
        'totals': {
            'A_all51_current_tpsl': round(oos_pnl_A, 1),
            'B_rediscovered_train_tpsl': round(oos_pnl_B, 1),
            'C_rediscovered_current_tpsl': round(oos_pnl_C, 1),
        },
        'look_ahead_decomposition': {
            'pattern_selection_la': round(total_la_pattern, 1),
            'tpsl_optimization_la': round(total_la_tpsl, 1),
            'genuine_forward_value': round(total_genuine, 1),
            'genuine_pct': round(genuine_pct, 1),
        },
    },
    'verdict': {
        'genuine_forward_positive': total_genuine > 0,
        'genuine_forward_pnl': round(total_genuine, 1),
        'genuine_forward_pct': round(genuine_pct, 1),
        'watchlist_more_la_dependent': rate_diff > 0.15,
        'never_rediscovered_count': len(never),
        'never_rediscovered_patterns': never,
        'removal_candidates': [p for p in never if p in wl_set],
    },
}

out_path = Path(__file__).resolve().parent / '../../results/pattern_rediscovery_wf.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved: {out_path}")
print("Done.")
