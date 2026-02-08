"""
TP/SL Bias Research — Random Entry Baseline Analysis
=====================================================
핵심 질문: 패턴의 높은 WR이 실제 예측력인가, 아니면 TP < SL 구조에 의한
           랜덤 워크 편향인가?

방법:
  1. 각 고유 (TP, SL, direction) 조합에 대해 전체 바에서 랜덤 진입 시뮬레이션
  2. 랜덤 진입의 경험적 WR (baseline) 산출
  3. 패턴 실제 WR과 비교 → excess WR 계산
  4. 이항 검정으로 excess WR의 통계적 유의성 판정
  5. excess WR 기반으로 "진짜 에지" 패턴만 선별
  6. 진짜 에지 패턴으로 포트폴리오 재평가

이론:
  - 대칭 랜덤 워크에서 P(TP hit) = SL / (TP + SL)
    예) TP=1.5, SL=3.0 → P = 66.7% (TP 유리)
    예) TP=3.0, SL=0.3 → P = 9.1% (SL 유리)
  - 실제 시장은 비대칭이므로 경험적 baseline 필요
  - excess_wr = actual_wr - random_wr > 0 이어야 진짜 에지
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import PATTERN_OPTIMAL_TPSL, PATTERN_STATS

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500          # max holding period
FEE_PCT = 0.10          # 0.05% * 2 (round trip)
LEVERAGE = 3
SAMPLE_STEP = 5         # random entry every Nth bar (5 = ~15,500 entries)
EXCESS_WR_MIN = 3.0     # minimum excess WR (pp) for "genuine edge"
P_VALUE_MAX = 0.05      # max p-value for statistical significance

print("=" * 70)
print("TP/SL Bias Research — Random Entry Baseline Analysis")
print("=" * 70)

# ============================================================
# Load pruning results (T5 pattern list + per-pattern data)
# ============================================================
results_dir = Path(__file__).resolve().parent / '../../results'
pruning_path = results_dir / 'portfolio_pruning_v4.json'
with open(pruning_path) as f:
    pruning = json.load(f)

# All 78 patterns and their data
pattern_data = pruning['pattern_data']
# T5 optimized list
t5_patterns = set(pruning['tier_results']['T5_Optimized']['pattern_list'])

# Build portfolio maps
ALL_78 = {}
for pat, info in pattern_data.items():
    ALL_78[pat] = (info['direction'], info['tp'], info['sl'])

print(f"Loaded: {len(ALL_78)} patterns, T5: {len(t5_patterns)} patterns")

# ============================================================
# Load and classify data (reuse same logic as pruning script)
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
n_bars = len(df)
print(f"Data: {n_bars} bars")

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

# Build pattern occurrence indices
pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

print("Classification done.")


# ============================================================
# Backtest function (same as pruning script)
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Backtest with fixed TP/SL. Returns list of per-trade PnL."""
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


# ============================================================
# Phase 1: Random Entry Baseline per (TP, SL, Direction)
# ============================================================
print()
print("=" * 70)
print("Phase 1: Random Entry Baseline Simulation")
print(f"  Entry every {SAMPLE_STEP} bars → ~{n_bars // SAMPLE_STEP} entries per config")
print("=" * 70)

# Collect unique (tp, sl) pairs
unique_configs = set()
for pat, (d, tp, sl) in ALL_78.items():
    unique_configs.add((tp, sl))

print(f"  Unique (TP, SL) pairs: {len(unique_configs)}")

# Generate random entry indices (evenly spaced across dataset)
random_indices = np.arange(20, n_bars - MAX_BARS, SAMPLE_STEP)  # skip first 20 for rolling avg
print(f"  Random entry points: {len(random_indices)}")

t0 = time.time()
random_baselines = {}  # key: (tp, sl, direction) → {'wr': float, 'exp': float, 'trades': int}

for tp, sl in sorted(unique_configs):
    for direction in ['LONG', 'SHORT']:
        pnls = bt_fixed(random_indices, direction, tp, sl)
        if not pnls:
            random_baselines[(tp, sl, direction)] = {'wr': 0.0, 'exp': 0.0, 'trades': 0}
            continue

        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100
        exp = np.mean(pnls)

        random_baselines[(tp, sl, direction)] = {
            'wr': wr,
            'exp': exp,
            'trades': len(pnls),
        }

elapsed = time.time() - t0
print(f"  Simulation complete in {elapsed:.1f}s")

# Theoretical baseline comparison
print()
print(f"  {'TP':>5} {'SL':>5}  {'Dir':>5} | {'Theory':>7} {'Empir':>7} {'Diff':>6} | {'Exp':>7} | {'N':>6}")
print(f"  {'-'*70}")
for (tp, sl, direction), bl in sorted(random_baselines.items()):
    theory_wr = sl / (tp + sl) * 100  # symmetric random walk
    diff = bl['wr'] - theory_wr
    print(f"  {tp:5.1f} {sl:5.1f}  {direction:>5} | {theory_wr:6.1f}% {bl['wr']:6.1f}% {diff:+5.1f}% | {bl['exp']:+6.3f}% | {bl['trades']:6d}")


# ============================================================
# Phase 2: Per-Pattern Actual vs Random Comparison
# ============================================================
print()
print("=" * 70)
print("Phase 2: Pattern Actual WR vs Random Baseline")
print(f"  Genuine Edge: excess WR >= {EXCESS_WR_MIN}pp AND p-value < {P_VALUE_MAX}")
print("=" * 70)

results = []
for pat, (direction, tp, sl) in sorted(ALL_78.items()):
    # Actual backtest
    indices = pattern_indices.get(pat, np.array([], dtype=int))
    if len(indices) == 0:
        continue

    pnls = bt_fixed(indices, direction, tp, sl)
    if not pnls:
        continue

    actual_wins = sum(1 for p in pnls if p > 0)
    actual_trades = len(pnls)
    actual_wr = actual_wins / actual_trades * 100
    actual_exp = np.mean(pnls)

    # Random baseline
    bl = random_baselines.get((tp, sl, direction), {'wr': 50.0, 'exp': 0.0})
    random_wr = bl['wr']
    random_exp = bl['exp']

    # Excess WR
    excess_wr = actual_wr - random_wr

    # Binomial test: H0: pattern WR = random WR, H1: pattern WR > random WR
    # Under H0, each trade is a Bernoulli trial with p = random_wr/100
    if random_wr > 0:
        btest = scipy_stats.binomtest(actual_wins, actual_trades, random_wr / 100, alternative='greater')
        p_value = btest.pvalue
    else:
        p_value = 0.0

    # Excess expectancy
    excess_exp = actual_exp - random_exp

    # Theoretical baseline
    theory_wr = sl / (tp + sl) * 100
    theory_excess = actual_wr - theory_wr

    # Verdict
    has_edge = excess_wr >= EXCESS_WR_MIN and p_value < P_VALUE_MAX
    in_t5 = pat in t5_patterns

    results.append({
        'pattern': pat,
        'direction': direction,
        'tp': tp,
        'sl': sl,
        'rr': round(tp / sl, 2) if sl > 0 else 99.9,
        'trades': actual_trades,
        'actual_wr': round(actual_wr, 1),
        'random_wr': round(random_wr, 1),
        'excess_wr': round(excess_wr, 1),
        'theory_wr': round(theory_wr, 1),
        'theory_excess': round(theory_excess, 1),
        'p_value': round(p_value, 4),
        'actual_exp': round(actual_exp, 3),
        'random_exp': round(random_exp, 3),
        'excess_exp': round(excess_exp, 3),
        'has_edge': has_edge,
        'in_t5': in_t5,
    })

# Sort by excess WR descending
results.sort(key=lambda x: x['excess_wr'], reverse=True)

# Print table
print()
print(f"  {'#':>3} {'Pattern':<12} {'Dir':>5} {'TP/SL':>7} {'R:R':>5} {'Tr':>4} | "
      f"{'ActWR':>6} {'RndWR':>6} {'ExcWR':>6} | {'ThryWR':>6} | "
      f"{'p-val':>7} | {'ActExp':>7} {'RndExp':>7} {'ExcExp':>7} | {'Edge':>6} {'T5':>3}")
print(f"  {'-'*125}")

genuine_count = 0
no_edge_count = 0
for i, r in enumerate(results, 1):
    edge_str = "YES" if r['has_edge'] else "NO"
    t5_str = "T5" if r['in_t5'] else ""
    p_str = f"{r['p_value']:.4f}" if r['p_value'] >= 0.0001 else "<.0001"

    # Highlight markers
    marker = ""
    if r['has_edge']:
        genuine_count += 1
        marker = " ***"
    else:
        no_edge_count += 1

    print(f"  {i:3d} {r['pattern']:<12} {r['direction']:>5} {r['tp']:3.1f}/{r['sl']:3.1f} {r['rr']:5.2f} {r['trades']:4d} | "
          f"{r['actual_wr']:5.1f}% {r['random_wr']:5.1f}% {r['excess_wr']:+5.1f}% | {r['theory_wr']:5.1f}% | "
          f"{p_str:>7} | {r['actual_exp']:+6.3f}% {r['random_exp']:+6.3f}% {r['excess_exp']:+6.3f}% | "
          f"{edge_str:>6} {t5_str:>3}{marker}")

print()
print(f"  Genuine Edge: {genuine_count}/{len(results)} patterns")
print(f"  No Genuine Edge: {no_edge_count}/{len(results)} patterns")


# ============================================================
# Phase 3: Categorize patterns by edge quality
# ============================================================
print()
print("=" * 70)
print("Phase 3: Edge Quality Categories")
print("=" * 70)

cat_strong = [r for r in results if r['excess_wr'] >= 10.0 and r['p_value'] < 0.01]
cat_moderate = [r for r in results if r['excess_wr'] >= EXCESS_WR_MIN and r['p_value'] < P_VALUE_MAX and r not in cat_strong]
cat_weak = [r for r in results if 0 < r['excess_wr'] < EXCESS_WR_MIN or (r['excess_wr'] >= EXCESS_WR_MIN and r['p_value'] >= P_VALUE_MAX)]
cat_none = [r for r in results if r['excess_wr'] <= 0]

print(f"\n  STRONG edge (excess >= 10pp, p < 0.01): {len(cat_strong)}")
for r in sorted(cat_strong, key=lambda x: x['excess_wr'], reverse=True):
    print(f"    {r['pattern']:<12} {r['direction']:>5} TP={r['tp']}/SL={r['sl']} "
          f"actual={r['actual_wr']:.1f}% random={r['random_wr']:.1f}% excess={r['excess_wr']:+.1f}pp p={r['p_value']:.4f}")

print(f"\n  MODERATE edge (excess >= {EXCESS_WR_MIN}pp, p < {P_VALUE_MAX}): {len(cat_moderate)}")
for r in sorted(cat_moderate, key=lambda x: x['excess_wr'], reverse=True):
    print(f"    {r['pattern']:<12} {r['direction']:>5} TP={r['tp']}/SL={r['sl']} "
          f"actual={r['actual_wr']:.1f}% random={r['random_wr']:.1f}% excess={r['excess_wr']:+.1f}pp p={r['p_value']:.4f}")

print(f"\n  WEAK/INSIGNIFICANT (0 < excess < {EXCESS_WR_MIN}pp OR p >= {P_VALUE_MAX}): {len(cat_weak)}")
for r in sorted(cat_weak, key=lambda x: x['excess_wr'], reverse=True):
    print(f"    {r['pattern']:<12} {r['direction']:>5} TP={r['tp']}/SL={r['sl']} "
          f"actual={r['actual_wr']:.1f}% random={r['random_wr']:.1f}% excess={r['excess_wr']:+.1f}pp p={r['p_value']:.4f}")

print(f"\n  NO EDGE (excess <= 0): {len(cat_none)}")
for r in sorted(cat_none, key=lambda x: x['excess_wr'], reverse=True):
    print(f"    {r['pattern']:<12} {r['direction']:>5} TP={r['tp']}/SL={r['sl']} "
          f"actual={r['actual_wr']:.1f}% random={r['random_wr']:.1f}% excess={r['excess_wr']:+.1f}pp p={r['p_value']:.4f}")


# ============================================================
# Phase 4: Portfolio Comparison — All vs Genuine-Edge-Only
# ============================================================
print()
print("=" * 70)
print("Phase 4: Portfolio Comparison (1-pos-at-a-time, simple returns)")
print("=" * 70)

genuine_patterns = set(r['pattern'] for r in results if r['has_edge'])
genuine_and_t5 = genuine_patterns & t5_patterns

# Build portfolio maps
portfolios = {
    'T0_All78': ALL_78,
    'T5_Optimized': {p: ALL_78[p] for p in t5_patterns if p in ALL_78},
    'Genuine_Edge': {p: ALL_78[p] for p in genuine_patterns if p in ALL_78},
    'T5_AND_Genuine': {p: ALL_78[p] for p in genuine_and_t5 if p in ALL_78},
}


def simulate_portfolio(pattern_map):
    """1-position-at-a-time simulation with simple returns."""
    raw_trades = []
    for i in range(2, n_bars):
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
        pnl_val = None
        exit_bar = None
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl_val = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                exit_bar = j
                break
            elif ht:
                pnl_val = tp * LEVERAGE - FEE_PCT
                exit_bar = j
                break
            elif hs:
                pnl_val = -sl * LEVERAGE - FEE_PCT
                exit_bar = j
                break
        if pnl_val is not None:
            raw_trades.append((i, exit_bar, pnl_val))

    if not raw_trades:
        return {'trades': 0}

    # 1-position-at-a-time filter
    raw_trades.sort(key=lambda x: x[0])
    trades = []
    last_exit = -1
    for entry_bar, exit_bar, pnl_val in raw_trades:
        if entry_bar > last_exit:
            trades.append(pnl_val)
            last_exit = exit_bar

    if not trades:
        return {'trades': 0}

    ta = np.array(trades)
    wins = np.sum(ta > 0)
    total = len(ta)
    wr = wins / total * 100
    exp = np.mean(ta)
    pnl = np.sum(ta)

    equity = np.cumsum(ta)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    pnl_mdd = pnl / mdd if mdd > 0 else float('inf')

    avg_w = np.mean(ta[ta > 0]) if wins > 0 else 0
    avg_l = abs(np.mean(ta[ta <= 0])) if (total - wins) > 0 else 0
    pf = (np.sum(ta[ta > 0]) / abs(np.sum(ta[ta <= 0]))) if np.sum(ta[ta <= 0]) != 0 else float('inf')

    # Walk-forward
    fold_size = total // 5
    wf_pass = 0
    wf_details = []
    for f in range(5):
        s = f * fold_size
        e = s + fold_size if f < 4 else total
        fp = np.sum(ta[s:e])
        passed = fp > 0
        if passed:
            wf_pass += 1
        wf_details.append(f"[F{f+1}: {e-s}t {fp:+.1f}% {'OK' if passed else 'FAIL'}]")

    return {
        'trades': total,
        'wr': round(wr, 1),
        'exp': round(exp, 3),
        'pnl': round(pnl, 1),
        'mdd': round(mdd, 1),
        'pnl_mdd': round(pnl_mdd, 2),
        'pf': round(pf, 2),
        'avg_win': round(avg_w, 2),
        'avg_loss': round(avg_l, 2),
        'wf': wf_pass,
        'wf_details': ' '.join(wf_details),
        'trades_per_day': round(total / 270, 1),
    }


print()
header = f"{'Portfolio':<18} {'Pats':>4} {'Trades':>6} {'Tr/d':>5} {'WR':>6} {'Exp/tr':>8} {'PnL':>10} {'MDD':>8} {'P/M':>8} {'PF':>6} {'AvgW':>6} {'AvgL':>6} {'WF':>5}"
print(header)
print("-" * len(header))

portfolio_results = {}
for name, pmap in portfolios.items():
    n_long = sum(1 for _, (d, _, _) in pmap.items() if d == 'LONG')
    n_short = len(pmap) - n_long
    stats = simulate_portfolio(pmap)
    portfolio_results[name] = stats

    if stats['trades'] == 0:
        print(f"{name:<18} {len(pmap):>4} -- no trades --")
        continue

    print(f"{name:<18} {len(pmap):>4} {stats['trades']:>6} {stats['trades_per_day']:>5.1f} "
          f"{stats['wr']:>5.1f}% {stats['exp']:>+7.3f}% {stats['pnl']:>+9.1f}% {stats['mdd']:>7.1f}% "
          f"{stats['pnl_mdd']:>7.2f}x {stats['pf']:>5.2f} {stats['avg_win']:>5.2f} {stats['avg_loss']:>5.2f} "
          f"{stats['wf']:>3d}/5")
    print(f"  WF: {stats['wf_details']}")


# ============================================================
# Phase 5: WR Resilience of genuine-edge portfolios
# ============================================================
print()
print("=" * 70)
print("Phase 5: WR Drop Resilience")
print("=" * 70)

for name, pmap in portfolios.items():
    if not pmap:
        continue

    # Get pattern-level results for this portfolio
    pat_results = [r for r in results if r['pattern'] in pmap]
    avg_excess = np.mean([r['excess_wr'] for r in pat_results]) if pat_results else 0

    # Avg actual WR and random WR
    avg_actual = np.mean([r['actual_wr'] for r in pat_results]) if pat_results else 0
    avg_random = np.mean([r['random_wr'] for r in pat_results]) if pat_results else 0

    # Portfolio-level stats
    ps = portfolio_results.get(name, {})
    pnl = ps.get('pnl', 0)
    exp = ps.get('exp', 0)

    print(f"\n  {name} ({len(pmap)} patterns):")
    print(f"    Avg actual WR: {avg_actual:.1f}%")
    print(f"    Avg random WR: {avg_random:.1f}%")
    print(f"    Avg excess WR: {avg_excess:+.1f}pp")
    print(f"    Portfolio Exp:  {exp:+.3f}%")
    print(f"    Portfolio PnL:  {pnl:+.1f}%")


# ============================================================
# Phase 6: Summary Table — Excess WR by TP/SL structure
# ============================================================
print()
print("=" * 70)
print("Phase 6: TP/SL Structure Analysis")
print("=" * 70)

# Group by R:R range
rr_groups = {
    'R:R < 1.0 (TP < SL, biased)': [r for r in results if r['rr'] < 1.0],
    'R:R = 1.0 (TP = SL, fair)': [r for r in results if r['rr'] == 1.0],
    'R:R > 1.0 (TP > SL, unbiased)': [r for r in results if r['rr'] > 1.0],
}

for label, group in rr_groups.items():
    if not group:
        continue
    avg_excess = np.mean([r['excess_wr'] for r in group])
    genuine = sum(1 for r in group if r['has_edge'])
    avg_actual = np.mean([r['actual_wr'] for r in group])
    avg_random = np.mean([r['random_wr'] for r in group])
    avg_exp_excess = np.mean([r['excess_exp'] for r in group])

    print(f"\n  {label} ({len(group)} patterns):")
    print(f"    Avg actual WR:  {avg_actual:.1f}%")
    print(f"    Avg random WR:  {avg_random:.1f}%")
    print(f"    Avg excess WR:  {avg_excess:+.1f}pp")
    print(f"    Avg excess Exp: {avg_exp_excess:+.3f}%")
    print(f"    Genuine edge:   {genuine}/{len(group)} ({genuine/len(group)*100:.0f}%)")
    print(f"    Patterns: ", end="")
    for r in group:
        marker = "*" if r['has_edge'] else " "
        print(f"{marker}{r['pattern']}", end="  ")
    print()


# ============================================================
# Phase 7: Random Entry Expectancy Analysis
# ============================================================
print()
print("=" * 70)
print("Phase 7: Random Entry Expectancy (key insight)")
print("  If random entry already has positive exp, TP/SL config is biased")
print("=" * 70)

print(f"\n  {'TP':>5} {'SL':>5} {'R:R':>5} {'Dir':>5} | {'RndWR':>6} {'ThryWR':>7} | {'RndExp':>8} | {'Bias':>10}")
print(f"  {'-'*75}")

for (tp, sl, direction), bl in sorted(random_baselines.items()):
    rr = tp / sl if sl > 0 else 99.9
    theory_wr = sl / (tp + sl) * 100
    wr_diff = bl['wr'] - theory_wr

    # In fair market, random exp should be ~0
    bias = "BIASED" if abs(bl['exp']) > 0.1 else "FAIR"
    if bl['exp'] > 0.1:
        bias = "LONG-BIASED" if direction == 'LONG' else "SHORT-BIASED"
    elif bl['exp'] < -0.1:
        bias = "ANTI-" + direction

    print(f"  {tp:5.1f} {sl:5.1f} {rr:5.2f} {direction:>5} | {bl['wr']:5.1f}% {theory_wr:6.1f}% | {bl['exp']:+7.3f}% | {bias:>10}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Count edge patterns in T5
t5_edge = [r for r in results if r['in_t5'] and r['has_edge']]
t5_no_edge = [r for r in results if r['in_t5'] and not r['has_edge']]

print(f"\n  T5_Optimized (58 patterns):")
print(f"    With genuine edge:    {len(t5_edge)} patterns")
print(f"    Without genuine edge: {len(t5_no_edge)} patterns")

if t5_no_edge:
    print(f"\n  T5 patterns WITHOUT genuine edge (candidates for removal):")
    for r in sorted(t5_no_edge, key=lambda x: x['excess_wr']):
        print(f"    {r['pattern']:<12} {r['direction']:>5} R:R={r['rr']:.2f} "
              f"actual={r['actual_wr']:.1f}% random={r['random_wr']:.1f}% excess={r['excess_wr']:+.1f}pp "
              f"p={r['p_value']:.4f}")

# Recommended portfolio
if len(t5_edge) > 0:
    rec_name = "T5_AND_Genuine"
    rec_pats = genuine_and_t5
else:
    rec_name = "T5_Optimized"
    rec_pats = t5_patterns

rec_stats = portfolio_results.get(rec_name, {})

print(f"\n  --- Recommendation: {rec_name} ---")
if rec_stats.get('trades', 0) > 0:
    print(f"    Patterns:   {len(rec_pats)} ({sum(1 for p in rec_pats if ALL_78[p][0]=='LONG')}L + "
          f"{sum(1 for p in rec_pats if ALL_78[p][0]=='SHORT')}S)")
    print(f"    Trades:     {rec_stats['trades']} ({rec_stats['trades_per_day']}/day)")
    print(f"    Expectancy: {rec_stats['exp']:+.3f}%/trade")
    print(f"    PnL:        {rec_stats['pnl']:+.1f}%")
    print(f"    MDD:        {rec_stats['mdd']:.1f}%")
    print(f"    PnL/MDD:    {rec_stats['pnl_mdd']:.2f}x")
    print(f"    PF:         {rec_stats['pf']:.2f}")
    print(f"    WF:         {rec_stats['wf']}/5")


# ============================================================
# Save results
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'research': 'TP/SL Bias Research — Random Entry Baseline',
    'parameters': {
        'sample_step': SAMPLE_STEP,
        'max_bars': MAX_BARS,
        'fee_pct': FEE_PCT,
        'leverage': LEVERAGE,
        'excess_wr_min': EXCESS_WR_MIN,
        'p_value_max': P_VALUE_MAX,
    },
    'random_baselines': {
        f"{tp}/{sl}/{d}": bl for (tp, sl, d), bl in random_baselines.items()
    },
    'pattern_results': results,
    'portfolio_comparison': portfolio_results,
    'edge_summary': {
        'genuine_count': genuine_count,
        'no_edge_count': no_edge_count,
        'strong': [r['pattern'] for r in cat_strong],
        'moderate': [r['pattern'] for r in cat_moderate],
        'weak': [r['pattern'] for r in cat_weak],
        'none': [r['pattern'] for r in cat_none],
    },
    't5_edge_analysis': {
        'with_edge': [r['pattern'] for r in t5_edge],
        'without_edge': [r['pattern'] for r in t5_no_edge],
    },
    'recommendation': rec_name,
}

out_path = results_dir / 'tp_sl_bias_research.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Results saved to {out_path}")
print("\nDone.")
