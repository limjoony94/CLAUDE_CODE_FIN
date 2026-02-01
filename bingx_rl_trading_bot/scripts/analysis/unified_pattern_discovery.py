"""
Unified Pattern Discovery - Production-Consistent Classification
================================================================
270일 데이터, 프로덕션과 동일한 캔들 분류, 전체 1,728 패턴 스캔.

검증 기준:
  - MC p-value < 0.05 (sign randomization, 10k sims)
  - WF >= 3/5 (walk-forward OOS)
  - 기간별 안정성 >= 2/3 (H1/H2/H3)
  - MIN_TRADES >= 15
  - Excess WR > baseline

출력: 유효 패턴 목록 + 균일 1.0/1.0 vs per-pattern TP/SL 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

# Import canonical classify_candle from production
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.production.pattern_5m.indicators import classify_candle

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MIN_TRADES = 15
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# ============================================================
# Load data
# ============================================================
data_path = Path(__file__).parent / '../../data/btc_5m_270days.csv'
df = pd.read_csv(data_path)
print(f"Loaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values

period_masks = {
    'H1_May_Jul': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'H2_Aug_Oct': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'H3_Nov_Jan': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

# ============================================================
# Candle Classification - PRODUCTION IDENTICAL
# Thresholds from constants.py:
#   DOJI_BODY_RATIO_THRESHOLD = 0.10
#   WICK_DOMINANCE_THRESHOLD = 0.70
#   MARUBOZU_WICK_RATIO_THRESHOLD = 0.15
#   HAMMER_WICK_TO_BODY_RATIO = 2.0
#   HAMMER_OPPOSITE_WICK_RATIO = 0.3
#   SPINNING_TOP_BODY_NORM = 0.5
#   SPINNING_TOP_WICK_RATIO = 0.5
#   BIG_CANDLE_NORM_THRESHOLD = 1.5
#   AVG_BODY_WINDOW = 20
# ============================================================

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values  # NaN for bars 0-19
# For early bars: use default avg_body_20=1.0 to preserve range-based classification
# (DOJI, HAMMER, DRAGONFLY, GRAVESTONE, MARUBOZU) while norm_body defaults conservatively
types = [classify_candle(opens[i], highs[i], lows[i], closes[i],
                         avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0)
         for i in range(n_bars)]

# Count type distribution
from collections import Counter
type_counts = Counter(types)
print("Candle type distribution:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:>3}: {c:>6} ({c/n_bars*100:.1f}%)")

# Build pattern index
pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

total_patterns = len(pattern_indices)
print(f"\nTotal unique 3-candle patterns: {total_patterns}")
print(f"Patterns with >= {MIN_TRADES} trades: {sum(1 for v in pattern_indices.values() if len(v) >= MIN_TRADES)}")

# ============================================================
# Baseline win rates (random entry benchmark)
# ============================================================
print("\nComputing baselines...")
sample_idx = np.arange(2, n_bars, 20)

baselines = {}
for tp in TP_SL_GRID:
    for sl in TP_SL_GRID:
        wins_l, wins_s, total_l, total_s = 0, 0, 0, 0
        for idx in sample_idx:
            if idx + 1 >= n_bars:
                continue
            entry = opens[idx + 1]
            if entry <= 0:
                continue
            for d, w_ref, t_ref in [('LONG', 'wl', 'tl'), ('SHORT', 'ws', 'ts')]:
                if d == 'LONG':
                    tpp = entry * (1 + tp / 100)
                    slp = entry * (1 - sl / 100)
                else:
                    tpp = entry * (1 - tp / 100)
                    slp = entry * (1 + sl / 100)
                for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                    ht = (highs[j] >= tpp) if d == 'LONG' else (lows[j] <= tpp)
                    hs = (lows[j] <= slp) if d == 'LONG' else (highs[j] >= slp)
                    if ht and hs:
                        if abs(tpp - entry) <= abs(slp - entry):
                            if d == 'LONG':
                                wins_l += 1
                            else:
                                wins_s += 1
                        if d == 'LONG':
                            total_l += 1
                        else:
                            total_s += 1
                        break
                    elif ht:
                        if d == 'LONG':
                            wins_l += 1
                            total_l += 1
                        else:
                            wins_s += 1
                            total_s += 1
                        break
                    elif hs:
                        if d == 'LONG':
                            total_l += 1
                        else:
                            total_s += 1
                        break
        baselines[(tp, sl)] = {
            'LONG': wins_l / total_l * 100 if total_l else 50,
            'SHORT': wins_s / total_s * 100 if total_s else 50,
        }
print("  Baselines computed.")

# ============================================================
# Backtest engine
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Returns list of per-trade PnL percentages."""
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


def walk_forward_test(indices, direction, tp_pct, sl_pct, n_folds=5):
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


def period_test(indices, direction, tp_pct, sl_pct):
    profitable = 0
    for mask in period_masks.values():
        pidx = indices[mask[indices]]
        if len(pidx) < 3:
            continue
        pnls = bt_fixed(pidx, direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


# ============================================================
# Phase 1: Full pattern scan (all TP/SL combos)
# ============================================================
print(f"\n{'='*70}")
print("Phase 1: Full Pattern Scan (1,728 patterns × {len(TP_SL_GRID)}² TP/SL × 2 directions)")
print(f"{'='*70}")

t0 = time.time()
valid_patterns = {}  # key -> (direction, tp, sl, n_trades, wr, excess, mc, wf, periods, total_pnl)

for pat, indices in pattern_indices.items():
    if len(indices) < MIN_TRADES:
        continue
    for direction in ['LONG', 'SHORT']:
        best = None
        for tp in TP_SL_GRID:
            for sl in TP_SL_GRID:
                pnls = bt_fixed(indices, direction, tp, sl)
                if len(pnls) < MIN_TRADES:
                    continue
                wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
                base = baselines.get((tp, sl), {}).get(direction, 50)
                excess = wr - base
                if excess < 10:
                    continue
                total_pnl = sum(pnls)
                if total_pnl <= 0:
                    continue
                # Quick MC
                mc = mc_test(pnls, 5000)
                if mc >= 0.05:
                    continue
                # WF
                wf = walk_forward_test(indices, direction, tp, sl)
                if wf < 3:
                    continue
                # Period
                pp = period_test(indices, direction, tp, sl)
                if pp < 2:
                    continue
                # Full MC
                mc_full = mc_test(pnls, MC_SIMS)
                score = total_pnl * (wf / 5) * (1 - mc_full)
                if best is None or score > best[-1]:
                    best = (direction, tp, sl, len(pnls), wr, excess, mc_full, wf, pp, total_pnl, score)
        if best:
            key = f"{pat}_{best[0]}"
            valid_patterns[key] = (pat,) + best[:-1]  # drop score

elapsed = time.time() - t0
print(f"  Scan complete in {elapsed:.1f}s")
print(f"  Raw valid patterns: {len(valid_patterns)}")

# Remove bidirectional (keep best direction per pattern)
from collections import Counter as Ctr
pat_names = [v[0] for v in valid_patterns.values()]
bidir = {p for p, c in Ctr(pat_names).items() if c > 1}
if bidir:
    # Keep the one with higher total_pnl
    to_remove = []
    for p in bidir:
        keys = [k for k, v in valid_patterns.items() if v[0] == p]
        keys.sort(key=lambda k: valid_patterns[k][10], reverse=True)  # sort by total_pnl
        to_remove.extend(keys[1:])
    for k in to_remove:
        del valid_patterns[k]
    print(f"  After removing bidirectional: {len(valid_patterns)}")

# ============================================================
# Phase 2: Results table
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: Validated Patterns")
print(f"{'='*70}\n")

# Sort by total_pnl
sorted_pats = sorted(valid_patterns.items(), key=lambda x: -x[1][10])

longs = [(k, v) for k, v in sorted_pats if v[1] == 'LONG']
shorts = [(k, v) for k, v in sorted_pats if v[1] == 'SHORT']

print(f"LONG patterns: {len(longs)}")
print(f"{'Pattern':<15} {'Dir':>5} {'TP':>4} {'SL':>4} {'Trades':>6} {'WR':>6} {'Excess':>7} {'MC':>8} {'WF':>4} {'Per':>4} {'TotPnL':>8}")
print('-' * 80)
for k, v in longs:
    pat, d, tp, sl, nt, wr, exc, mc, wf, pp, tpnl = v
    print(f"{pat:<15} {d:>5} {tp:>4.1f} {sl:>4.1f} {nt:>6} {wr:>5.1f}% {exc:>+6.1f}% {mc:>8.4f} {wf:>3}/5 {pp:>3}/3 {tpnl:>+7.1f}%")

print(f"\nSHORT patterns: {len(shorts)}")
print(f"{'Pattern':<15} {'Dir':>5} {'TP':>4} {'SL':>4} {'Trades':>6} {'WR':>6} {'Excess':>7} {'MC':>8} {'WF':>4} {'Per':>4} {'TotPnL':>8}")
print('-' * 80)
for k, v in shorts:
    pat, d, tp, sl, nt, wr, exc, mc, wf, pp, tpnl = v
    print(f"{pat:<15} {d:>5} {tp:>4.1f} {sl:>4.1f} {nt:>6} {wr:>5.1f}% {exc:>+6.1f}% {mc:>8.4f} {wf:>3}/5 {pp:>3}/3 {tpnl:>+7.1f}%")

# ============================================================
# Phase 3: Uniform 1.0/1.0 test for all valid patterns
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Uniform 1.0/1.0 TP/SL Test")
print(f"{'='*70}\n")

uniform_valid = []
print(f"{'Pattern':<15} {'Dir':>5} {'Trades':>6} {'WR':>6} {'Excess':>7} {'MC':>8} {'WF':>4} {'Per':>4} {'TotPnL':>8}")
print('-' * 80)

for k, v in sorted_pats:
    pat, d, _, _, _, _, _, _, _, _, _ = v
    indices = pattern_indices[pat]
    pnls = bt_fixed(indices, d, 1.0, 1.0)
    if len(pnls) < MIN_TRADES:
        continue
    wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
    base = baselines.get((1.0, 1.0), {}).get(d, 50)
    excess = wr - base
    total_pnl = sum(pnls)
    mc = mc_test(pnls, MC_SIMS)
    wf = walk_forward_test(indices, d, 1.0, 1.0)
    pp = period_test(indices, d, 1.0, 1.0)
    status = "✅" if (mc < 0.05 and wf >= 3 and pp >= 2 and excess > 10 and total_pnl > 0) else "❌"
    print(f"{pat:<15} {d:>5} {len(pnls):>6} {wr:>5.1f}% {excess:>+6.1f}% {mc:>8.4f} {wf:>3}/5 {pp:>3}/3 {total_pnl:>+7.1f}% {status}")
    if mc < 0.05 and wf >= 3 and pp >= 2 and excess > 10 and total_pnl > 0:
        uniform_valid.append({
            'pattern': pat, 'direction': d,
            'trades': len(pnls), 'wr': wr, 'excess': excess,
            'mc': mc, 'wf': wf, 'periods': pp, 'total_pnl': total_pnl,
        })

print(f"\nUniform 1.0/1.0 valid: {len(uniform_valid)} patterns")

# Tier 1 (strict): WF>=4, MC<0.01, excess>15
uniform_tier1 = [p for p in uniform_valid if p['wf'] >= 4 and p['mc'] < 0.01 and p['excess'] > 15]
print(f"Uniform 1.0/1.0 Tier 1 (WF>=4, MC<0.01, excess>15): {len(uniform_tier1)} patterns")
for p in uniform_tier1:
    print(f"  {p['pattern']:<15} {p['direction']:>5} {p['trades']:>4}t {p['wr']:>5.1f}% exc{p['excess']:>+6.1f}% MC={p['mc']:.4f} WF={p['wf']}/5 PP={p['periods']}/3")

# ============================================================
# Phase 4: Portfolio backtest comparison
# ============================================================
print(f"\n{'='*70}")
print("Phase 4: Portfolio Backtest (Compound Equity)")
print(f"{'='*70}\n")

def collect_trades(pattern_map, bar_range=None):
    """Collect trades for a set of patterns."""
    trades = []
    start = bar_range[0] if bar_range else 2
    end = bar_range[1] if bar_range else n_bars
    for i in range(max(start, 2), end):
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
                trades.append((i + 1, (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT))
                break
            elif ht:
                trades.append((i + 1, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                trades.append((i + 1, -sl * LEVERAGE - FEE_PCT))
                break
    trades.sort(key=lambda x: x[0])
    return trades


def eval_trades(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pnl_mdd': 0, 'pf': 0}
    pnl_list = [t[1] for t in trades]
    equity = 100.0
    peak = 100.0
    max_dd = 0
    wins = 0
    total_win = 0
    total_loss = 0
    for p in pnl_list:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            total_win += p
        else:
            total_loss += abs(p)
    return {
        'pnl': (equity / 100 - 1) * 100,
        'trades': len(pnl_list),
        'wr': wins / len(pnl_list) * 100,
        'mdd': max_dd,
        'pnl_mdd': (equity / 100 - 1) * 100 / max_dd if max_dd > 0 else 999,
        'pf': total_win / total_loss if total_loss > 0 else 999,
    }


def mc_portfolio(collect_fn, n_sims=MC_SIMS):
    trades = collect_fn()
    if len(trades) < 5:
        return 1.0
    pnl_list = np.array([t[1] for t in trades])
    real_eq = np.prod(1 + pnl_list / 100)
    count = sum(1 for _ in range(n_sims)
                if np.prod(1 + pnl_list * np.random.choice([-1, 1], len(pnl_list)) / 100) >= real_eq)
    return count / n_sims


# Build pattern maps
# A: Per-pattern optimal TP/SL
pp_map = {}
for k, v in valid_patterns.items():
    pat, d, tp, sl = v[0], v[1], v[2], v[3]
    pp_map[pat] = (d, tp, sl)

# B: Uniform 1.0/1.0 (only patterns that pass uniform validation)
uniform_map = {}
for p in uniform_valid:
    uniform_map[p['pattern']] = (p['direction'], 1.0, 1.0)

uniform_t1_map = {}
for p in uniform_tier1:
    uniform_t1_map[p['pattern']] = (p['direction'], 1.0, 1.0)

print(f"A: Per-pattern optimal: {len(pp_map)} patterns")
print(f"B: Uniform 1.0/1.0:    {len(uniform_map)} patterns")
print(f"C: Uniform Tier 1:     {len(uniform_t1_map)} patterns\n")

for label, pmap in [('A: Per-Pattern Optimal', pp_map), ('B: Uniform 1.0/1.0', uniform_map), ('C: Uniform Tier 1 (strict)', uniform_t1_map)]:
    fn = lambda br=None, pm=pmap: collect_trades(pm, br)
    trades = fn()
    res = eval_trades(trades)
    mc = mc_portfolio(fn, MC_SIMS)

    # WF
    fold_size = n_bars // 5
    wf_count = 0
    wf_detail = []
    for f in range(5):
        s = f * fold_size
        e = s + fold_size if f < 4 else n_bars
        ft = fn(br=(s, e))
        fr = eval_trades(ft)
        wf_detail.append(fr)
        if fr['pnl'] > 0:
            wf_count += 1

    # Period
    period_results = {}
    for pname, mask in period_masks.items():
        bidx = np.where(mask)[0]
        if len(bidx) == 0:
            continue
        pt = fn(br=(bidx[0], bidx[-1] + 1))
        period_results[pname] = eval_trades(pt)

    pp_count = sum(1 for r in period_results.values() if r['pnl'] > 0)

    print(f"--- {label} ({len(pmap)} patterns) ---")
    print(f"  Trades: {res['trades']}, WR: {res['wr']:.1f}%, PnL: {res['pnl']:+.1f}%")
    print(f"  MDD: {res['mdd']:.1f}%, PnL/MDD: {res['pnl_mdd']:.1f}, PF: {res['pf']:.2f}")
    print(f"  MC p-value: {mc:.4f}")
    print(f"  Walk-Forward: {wf_count}/5")
    for i, fr in enumerate(wf_detail):
        s = "✅" if fr['pnl'] > 0 else "❌"
        print(f"    Fold {i+1}: {fr['trades']:>4}t, {fr['wr']:>5.1f}% WR, {fr['pnl']:>+10.1f}% PnL, {fr['mdd']:>5.1f}% MDD {s}")
    print(f"  Period stability: {pp_count}/3")
    for pn, pr in period_results.items():
        s = "✅" if pr['pnl'] > 0 else "❌"
        print(f"    {pn}: {pr['trades']:>4}t, {pr['wr']:>5.1f}% WR, {pr['pnl']:>+10.1f}% PnL {s}")
    print()

# ============================================================
# Phase 5: Export results
# ============================================================
results = {
    'timestamp': datetime.now().isoformat(),
    'data_bars': n_bars,
    'classification': 'production_unified (avg_body_20 based)',
    'per_pattern_optimal': [],
    'uniform_1_1': [],
}

for k, v in sorted_pats:
    pat, d, tp, sl, nt, wr, exc, mc, wf, pp, tpnl = v
    results['per_pattern_optimal'].append({
        'pattern': pat, 'direction': d, 'tp': tp, 'sl': sl,
        'trades': nt, 'wr': round(wr, 1), 'excess': round(exc, 1),
        'mc': round(mc, 4), 'wf': wf, 'periods': pp, 'total_pnl': round(tpnl, 1),
    })

for p in uniform_valid:
    results['uniform_1_1'].append(p)
results['uniform_tier1'] = uniform_tier1

out_path = Path(__file__).parent / '../../results/unified_pattern_discovery.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results saved to {out_path}")
print("\nDone.")
