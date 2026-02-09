"""
v1.26.4 Deep TP/SL Validation
===============================
32개 변경안에 대한 심층 검증:
  Phase 1: Leave-One-Fold-Out CV — 안정적으로 같은 TP/SL이 선택되는지
  Phase 2: Neighborhood Stability — 최적 주변 grid도 수익인지
  Phase 3: SL Widening Bias Test — 랜덤 진입 대비 패턴 edge 확인
  Phase 4: Train(70%)/Test(30%) OOS Portfolio Validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL, VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
)

MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

MC_THRESHOLD = 0.01
WF_THRESHOLD = 4
PERIOD_THRESHOLD = 2
MIN_TRADES = 10

print("=" * 80)
print("v1.26.4 Deep TP/SL Validation")
print("=" * 80)

# ============================================================
# Load and classify
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values

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

# Build maps
current_map = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('LONG', tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('SHORT', tp, sl)

# Load optimization results
opt_path = Path(__file__).resolve().parent / '../../results/tp_sl_optimization_v1264.json'
with open(opt_path) as f:
    opt_data = json.load(f)

changes = opt_data['changes']
change_map = {c['pattern']: c for c in changes}

print(f"Changes to validate: {len(changes)}")


# ============================================================
# Engine
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
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
    if len(pnl_list) < 5:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def find_best_tpsl(indices, direction, min_trades=5):
    """Find best TP/SL by total PnL with MC<0.01 constraint."""
    best = None
    best_pnl = -999
    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            pnls = bt_fixed(indices, direction, tp, sl)
            if len(pnls) < min_trades:
                continue
            total = sum(pnls)
            if total <= 0:
                continue
            mc = mc_test(pnls, 3000)
            if mc >= 0.03:
                continue
            mc_full = mc_test(pnls, MC_SIMS)
            if mc_full >= MC_THRESHOLD:
                continue
            if total > best_pnl:
                best_pnl = total
                best = (tp, sl, total, len(pnls), mc_full)
    return best


def collect_trades_1pos(pattern_map, start_bar=0, end_bar=None):
    if end_bar is None:
        end_bar = n_bars
    raw_trades = []
    for i in range(max(2, start_bar), end_bar):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
        entry_bar = i + 1
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
                raw_trades.append((entry_bar, j, pnl))
                break
            elif ht:
                raw_trades.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                raw_trades.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in raw_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def eval_portfolio(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'exp': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wps = []; lps = []
    for p in pnl_list:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0: wins += 1; wps.append(p)
        else: lps.append(abs(p))
    aw = np.mean(wps) if wps else 0
    al = np.mean(lps) if lps else 0
    tw = sum(wps); tl = sum(lps)
    wr = wins / len(pnl_list) * 100
    exp = (wr / 100) * aw - (1 - wr / 100) * al
    return {
        'pnl': cum, 'trades': len(pnl_list), 'wr': wr, 'mdd': mdd,
        'pf': tw / tl if tl > 0 else 999, 'exp': exp,
        'pnl_mdd': cum / mdd if mdd > 0 else 999,
    }


# ============================================================
# Phase 1: Leave-One-Fold-Out CV
# ============================================================
print(f"\n{'='*80}")
print("Phase 1: Leave-One-Fold-Out Cross-Validation (5-fold)")
print("  For each pattern, optimize on 4 folds, check if same TP/SL selected")
print(f"{'='*80}")

t0 = time.time()
n_folds = 5

cv_results = {}

for pat, chg in sorted(change_map.items()):
    direction = chg['direction']
    indices = pattern_indices.get(pat, np.array([]))
    si = np.sort(indices)
    fs = len(si) // n_folds

    if fs < 3:
        cv_results[pat] = {'stable': False, 'reason': 'too_few', 'selections': [], 'oos_pnls': []}
        continue

    selections = []
    oos_pnls = []

    for held_out in range(n_folds):
        # Build train indices (all folds except held_out)
        train_idx = []
        test_idx = []
        for f in range(n_folds):
            s = f * fs
            e = s + fs if f < n_folds - 1 else len(si)
            if f == held_out:
                test_idx = si[s:e]
            else:
                train_idx.extend(si[s:e])
        train_idx = np.array(train_idx)

        # Find best on train
        best = find_best_tpsl(train_idx, direction, min_trades=3)
        if best:
            tp, sl, train_pnl, train_n, train_mc = best
            selections.append((tp, sl))
            # Test on held-out fold
            test_pnls = bt_fixed(test_idx, direction, tp, sl)
            oos_pnl = sum(test_pnls) if test_pnls else 0
            oos_pnls.append(oos_pnl)
        else:
            selections.append(None)
            oos_pnls.append(0)

    # Check stability: how many folds selected the proposed TP/SL?
    proposed = (chg['new_tp'], chg['new_sl'])
    n_match = sum(1 for s in selections if s == proposed)
    n_valid = sum(1 for s in selections if s is not None)

    # Check consistency: most common selection
    valid_sels = [s for s in selections if s is not None]
    if valid_sels:
        most_common = Counter(valid_sels).most_common(1)[0]
        consensus_tpsl = most_common[0]
        consensus_count = most_common[1]
    else:
        consensus_tpsl = None
        consensus_count = 0

    # OOS profitability: how many folds profitable OOS?
    oos_positive = sum(1 for p in oos_pnls if p > 0)

    cv_results[pat] = {
        'proposed': proposed,
        'selections': selections,
        'consensus_tpsl': consensus_tpsl,
        'consensus_count': consensus_count,
        'n_match_proposed': n_match,
        'n_valid': n_valid,
        'oos_pnls': [round(p, 1) for p in oos_pnls],
        'oos_positive': oos_positive,
        'oos_total': round(sum(oos_pnls), 1),
        'stable': consensus_count >= 3,  # majority consensus
    }

elapsed = time.time() - t0
print(f"  Complete in {elapsed:.1f}s\n")

print(f"{'Pattern':<12} {'Proposed':>8} {'Consensus':>9} {'Match':>5} {'OOS+':>4} {'OOS PnL':>8} {'Stable':>6}")
print("-" * 60)

stable_count = 0
for pat in sorted(cv_results.keys()):
    r = cv_results[pat]
    if 'reason' in r:
        print(f"{pat:<12} {'---':>8} {'---':>9} {'---':>5} {'---':>4} {'---':>8} {'SKIP':>6}")
        continue
    prop_str = f"{r['proposed'][0]}/{r['proposed'][1]}"
    cons_str = f"{r['consensus_tpsl'][0]}/{r['consensus_tpsl'][1]}" if r['consensus_tpsl'] else "---"
    stable_str = "YES" if r['stable'] else "NO"
    if r['stable']:
        stable_count += 1
    print(f"{pat:<12} {prop_str:>8} {cons_str:>9} {r['n_match_proposed']:>3}/5 {r['oos_positive']:>2}/5 {r['oos_total']:>+7.1f}% {stable_str:>6}")

print(f"\nStable (consensus >= 3/5): {stable_count}/{len(cv_results)}")


# ============================================================
# Phase 2: Neighborhood Stability
# ============================================================
print(f"\n{'='*80}")
print("Phase 2: Neighborhood Stability (proposed TP/SL ± 1 grid step)")
print(f"{'='*80}\n")

neighbor_results = {}

for pat, chg in sorted(change_map.items()):
    direction = chg['direction']
    indices = pattern_indices.get(pat, np.array([]))
    new_tp, new_sl = chg['new_tp'], chg['new_sl']

    tp_idx = TP_SL_GRID.index(new_tp) if new_tp in TP_SL_GRID else -1
    sl_idx = TP_SL_GRID.index(new_sl) if new_sl in TP_SL_GRID else -1

    if tp_idx < 0 or sl_idx < 0:
        continue

    # Test 3x3 neighborhood (or smaller at edges)
    neighbors = []
    for dtp in [-1, 0, 1]:
        for dsl in [-1, 0, 1]:
            ti = tp_idx + dtp
            si_n = sl_idx + dsl
            if 0 <= ti < len(TP_SL_GRID) and 0 <= si_n < len(TP_SL_GRID):
                ntp = TP_SL_GRID[ti]
                nsl = TP_SL_GRID[si_n]
                pnls = bt_fixed(indices, direction, ntp, nsl)
                if len(pnls) >= 5:
                    total = sum(pnls)
                    wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
                    is_center = (dtp == 0 and dsl == 0)
                    neighbors.append({
                        'tp': ntp, 'sl': nsl, 'pnl': round(total, 1),
                        'trades': len(pnls), 'wr': round(wr, 1),
                        'center': is_center,
                    })

    # Count profitable neighbors
    n_profitable = sum(1 for n in neighbors if n['pnl'] > 0)
    n_total = len(neighbors)

    neighbor_results[pat] = {
        'proposed': (new_tp, new_sl),
        'neighbors': neighbors,
        'n_profitable': n_profitable,
        'n_total': n_total,
        'plateau': n_profitable >= n_total * 0.6,  # 60%+ neighbors profitable
    }

print(f"{'Pattern':<12} {'TP/SL':>7} {'Neighbors':>10} {'Profitable':>10} {'Plateau':>8}")
print("-" * 52)

plateau_count = 0
for pat in sorted(neighbor_results.keys()):
    r = neighbor_results[pat]
    tpsl = f"{r['proposed'][0]}/{r['proposed'][1]}"
    pl = "YES" if r['plateau'] else "NO"
    if r['plateau']:
        plateau_count += 1
    print(f"{pat:<12} {tpsl:>7} {r['n_total']:>10} {r['n_profitable']:>10} {pl:>8}")

print(f"\nPlateau (60%+ neighbors profitable): {plateau_count}/{len(neighbor_results)}")


# ============================================================
# Phase 3: SL Widening Bias Test
# ============================================================
print(f"\n{'='*80}")
print("Phase 3: SL Widening Bias Test")
print("  Random entry baseline WR for different SL widths")
print(f"{'='*80}\n")

# Sample random entries
np.random.seed(42)
sample_idx = np.random.choice(np.arange(22, n_bars - MAX_BARS), size=500, replace=False)

print("Random entry WR by TP/SL (LONG direction, 500 random entries):")
print(f"{'TP\\SL':>6}", end='')
for sl in TP_SL_GRID:
    print(f" {sl:>5}", end='')
print()

baseline_wr = {}
for tp in TP_SL_GRID:
    print(f"{tp:>6}", end='')
    for sl in TP_SL_GRID:
        pnls = bt_fixed(sample_idx, 'LONG', tp, sl)
        if len(pnls) > 0:
            wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
            baseline_wr[(tp, sl, 'LONG')] = wr
            print(f" {wr:>4.0f}%", end='')
        else:
            baseline_wr[(tp, sl, 'LONG')] = 50
            print(f"   --", end='')
    print()

# Compare: pattern WR - baseline WR = excess WR (edge)
print(f"\nExcess WR (pattern WR - random baseline WR) for changed patterns:")
print(f"{'Pattern':<12} {'Dir':>5} {'TP/SL':>7} {'Pat WR':>7} {'Base WR':>8} {'Excess':>7} {'Edge':>5}")
print("-" * 58)

edge_results = {}
for pat, chg in sorted(change_map.items()):
    direction = chg['direction']
    indices = pattern_indices.get(pat, np.array([]))
    new_tp, new_sl = chg['new_tp'], chg['new_sl']

    pnls = bt_fixed(indices, direction, new_tp, new_sl)
    if len(pnls) == 0:
        continue
    pat_wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100

    # Get baseline for this direction
    base_pnls = bt_fixed(sample_idx, direction, new_tp, new_sl)
    if len(base_pnls) > 0:
        base_wr = sum(1 for x in base_pnls if x > 0) / len(base_pnls) * 100
    else:
        base_wr = 50

    excess = pat_wr - base_wr
    has_edge = excess > 5  # 5pp minimum excess

    edge_results[pat] = {
        'pat_wr': round(pat_wr, 1), 'base_wr': round(base_wr, 1),
        'excess': round(excess, 1), 'has_edge': has_edge,
    }

    edge_str = "YES" if has_edge else "NO"
    print(f"{pat:<12} {direction:>5} {new_tp}/{new_sl:>3} {pat_wr:>6.1f}% {base_wr:>7.1f}% {excess:>+6.1f}% {edge_str:>5}")

edge_count = sum(1 for r in edge_results.values() if r['has_edge'])
print(f"\nPatterns with genuine edge (excess > 5pp): {edge_count}/{len(edge_results)}")


# ============================================================
# Phase 4: Train(70%) / Test(30%) OOS Portfolio
# ============================================================
print(f"\n{'='*80}")
print("Phase 4: Train(70%) / Test(30%) OOS Portfolio Validation")
print(f"{'='*80}\n")

train_end = int(n_bars * 0.7)
test_start = train_end

print(f"  Train: bars 0-{train_end} ({train_end} bars)")
print(f"  Test:  bars {test_start}-{n_bars} ({n_bars - test_start} bars)")

# Build optimized map on TRAIN data only
print("\n  Optimizing on train data...")
train_opt_map = {}
for pat, (direction, cur_tp, cur_sl) in current_map.items():
    indices = pattern_indices.get(pat, np.array([]))
    train_indices = indices[indices < train_end]

    if len(train_indices) < MIN_TRADES:
        # Keep current
        train_opt_map[pat] = (direction, cur_tp, cur_sl)
        continue

    best = find_best_tpsl(train_indices, direction, min_trades=5)
    if best:
        tp, sl, _, _, _ = best
        train_opt_map[pat] = (direction, tp, sl)
    else:
        train_opt_map[pat] = (direction, cur_tp, cur_sl)

# Count changes
train_changes = sum(1 for pat in current_map
                    if train_opt_map[pat][1] != current_map[pat][1]
                    or train_opt_map[pat][2] != current_map[pat][2])
print(f"  Changes from train optimization: {train_changes}")

# Evaluate on TEST set
print(f"\n  TEST set results (OOS):")
# Current on test
t_cur_test = collect_trades_1pos(current_map, test_start, n_bars)
r_cur_test = eval_portfolio(t_cur_test)

# Train-optimized on test
t_opt_test = collect_trades_1pos(train_opt_map, test_start, n_bars)
r_opt_test = eval_portfolio(t_opt_test)

# Full-data optimized on test (the 32-change map)
full_opt_map = {}
for pat, (direction, cur_tp, cur_sl) in current_map.items():
    if pat in change_map:
        c = change_map[pat]
        full_opt_map[pat] = (direction, c['new_tp'], c['new_sl'])
    else:
        full_opt_map[pat] = (direction, cur_tp, cur_sl)

t_full_test = collect_trades_1pos(full_opt_map, test_start, n_bars)
r_full_test = eval_portfolio(t_full_test)

print(f"\n  {'':>20} {'v1.26.3':>12} {'Train-Opt':>12} {'Full-Opt':>12}")
print(f"  {'-'*60}")
for k, label in [('trades', 'Trades'), ('wr', 'WR (%)'), ('pnl', 'PnL (%)'),
                  ('mdd', 'MDD (%)'), ('pf', 'PF'), ('exp', 'Exp (%)')]:
    vals = []
    for r in [r_cur_test, r_opt_test, r_full_test]:
        v = r[k]
        if k in ('wr', 'pnl', 'mdd', 'exp'):
            vals.append(f"{v:>11.1f}%")
        elif k == 'pf':
            vals.append(f"{v:>12.2f}")
        else:
            vals.append(f"{v:>12}")
    print(f"  {label:>20} {vals[0]} {vals[1]} {vals[2]}")

# Also show TRAIN set for reference
print(f"\n  TRAIN set results (in-sample, for reference):")
t_cur_train = collect_trades_1pos(current_map, 0, train_end)
r_cur_train = eval_portfolio(t_cur_train)
t_opt_train = collect_trades_1pos(train_opt_map, 0, train_end)
r_opt_train = eval_portfolio(t_opt_train)
t_full_train = collect_trades_1pos(full_opt_map, 0, train_end)
r_full_train = eval_portfolio(t_full_train)

print(f"\n  {'':>20} {'v1.26.3':>12} {'Train-Opt':>12} {'Full-Opt':>12}")
print(f"  {'-'*60}")
for k, label in [('trades', 'Trades'), ('wr', 'WR (%)'), ('pnl', 'PnL (%)'),
                  ('mdd', 'MDD (%)'), ('pf', 'PF'), ('exp', 'Exp (%)')]:
    vals = []
    for r in [r_cur_train, r_opt_train, r_full_train]:
        v = r[k]
        if k in ('wr', 'pnl', 'mdd', 'exp'):
            vals.append(f"{v:>11.1f}%")
        elif k == 'pf':
            vals.append(f"{v:>12.2f}")
        else:
            vals.append(f"{v:>12}")
    print(f"  {label:>20} {vals[0]} {vals[1]} {vals[2]}")


# ============================================================
# Phase 5: Final Composite Score
# ============================================================
print(f"\n{'='*80}")
print("Phase 5: Composite Validation Score per Pattern")
print(f"{'='*80}\n")

# For each change, score based on all validations
print(f"{'Pattern':<12} {'CV Stable':>9} {'Plateau':>8} {'Edge':>5} {'OOS+':>4} {'Score':>6} {'Verdict':>8}")
print("-" * 60)

final_accept = []
final_reject = []

for pat in sorted(change_map.keys()):
    cv = cv_results.get(pat, {})
    nb = neighbor_results.get(pat, {})
    eg = edge_results.get(pat, {})

    # Scoring (0-4 points)
    score = 0
    cv_ok = cv.get('stable', False)
    if cv_ok:
        score += 1
    pl_ok = nb.get('plateau', False)
    if pl_ok:
        score += 1
    eg_ok = eg.get('has_edge', False)
    if eg_ok:
        score += 1
    oos_ok = cv.get('oos_positive', 0) >= 3
    if oos_ok:
        score += 1

    verdict = "ACCEPT" if score >= 3 else ("MAYBE" if score == 2 else "REJECT")

    if score >= 3:
        final_accept.append(pat)
    elif score == 2:
        final_accept.append(pat)  # borderline accept
    else:
        final_reject.append(pat)

    cv_str = "YES" if cv_ok else "NO"
    pl_str = "YES" if pl_ok else "NO"
    eg_str = "YES" if eg_ok else "NO"
    oos_str = f"{cv.get('oos_positive', 0)}/5"

    print(f"{pat:<12} {cv_str:>9} {pl_str:>8} {eg_str:>5} {oos_str:>4} {score:>4}/4 {verdict:>8}")

print(f"\nACCEPT (score >= 2): {len(final_accept)}")
print(f"REJECT (score < 2): {len(final_reject)}")

if final_reject:
    print(f"\nRejected changes (keep current TP/SL):")
    for pat in final_reject:
        chg = change_map[pat]
        print(f"  {pat}: keep {chg['old_tp']}/{chg['old_sl']} (not {chg['new_tp']}/{chg['new_sl']})")

# Build final recommended map
final_map = {}
for pat, (direction, cur_tp, cur_sl) in current_map.items():
    if pat in change_map and pat in final_accept:
        c = change_map[pat]
        final_map[pat] = (direction, c['new_tp'], c['new_sl'])
    else:
        final_map[pat] = (direction, cur_tp, cur_sl)

# Evaluate final map
print(f"\n--- Final Recommended Portfolio ---")
t_final = collect_trades_1pos(final_map)
r_final = eval_portfolio(t_final)
n_changes_final = sum(1 for pat in current_map
                      if final_map[pat][1] != current_map[pat][1]
                      or final_map[pat][2] != current_map[pat][2])

print(f"  Changes: {n_changes_final}")
print(f"\n  {'':>16} {'v1.26.3':>12} {'Recommended':>12} {'Delta':>10}")
print(f"  {'-'*54}")
r_cur_full = eval_portfolio(collect_trades_1pos(current_map))
for k, label in [('trades', 'Trades'), ('wr', 'WR'), ('pnl', 'PnL'),
                  ('mdd', 'MDD'), ('pf', 'PF'), ('exp', 'Exp'), ('pnl_mdd', 'PnL/MDD')]:
    o = r_cur_full[k]; n = r_final[k]; d = n - o
    if k in ('wr', 'pnl', 'mdd', 'exp'):
        print(f"  {label:>16} {o:>11.1f}% {n:>11.1f}% {d:>+9.1f}%")
    elif k in ('pf', 'pnl_mdd'):
        print(f"  {label:>16} {o:>12.2f} {n:>12.2f} {d:>+10.2f}")
    else:
        print(f"  {label:>16} {o:>12} {n:>12} {d:>+10}")


# ============================================================
# Save
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'research': 'v1.26.4 Deep TP/SL Validation',
    'phase1_cv': cv_results,
    'phase2_neighbors': {p: {k: v for k, v in r.items() if k != 'neighbors'} for p, r in neighbor_results.items()},
    'phase3_edge': edge_results,
    'phase4_oos': {
        'test_current': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_cur_test.items()},
        'test_train_opt': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_opt_test.items()},
        'test_full_opt': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_full_test.items()},
    },
    'phase5_final': {
        'accepted': final_accept,
        'rejected': final_reject,
        'portfolio': {k: round(v, 2) if isinstance(v, float) else v for k, v in r_final.items()},
        'changes': [
            {'pattern': pat, 'direction': final_map[pat][0],
             'tp': final_map[pat][1], 'sl': final_map[pat][2]}
            for pat in final_accept if pat in change_map
        ],
    },
}

out_path = Path(__file__).resolve().parent / '../../results/tp_sl_deep_validation.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done.")
