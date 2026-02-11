"""
MC Filter Walk-Forward Validation
==================================
v1.27.0에서 개별 MC 실패하는 15/52 패턴 제거가 OOS에서 유효한지 검증.

Question: "MC 실패 패턴을 제거하면 미래 성과가 개선되는가?"

Strategies:
  A: All 52 patterns (v1.27.0 baseline)
  B: Adaptive MC filter strict (train MC p>=0.01 제거)
  C: Adaptive MC filter loose (train MC p>=0.05 제거)
  D: Fixed removal of 3 worst (MD-DN-MU, MD-ST-BD, U-MD-GS)

Methodology:
  Expanding-window Walk-Forward (train on past, test on future)
  270일 → 5 periods (~54일) → 4 WF iterations:
    WF1: Train[P1]       → Test[P2]
    WF2: Train[P1+P2]    → Test[P3]
    WF3: Train[P1+P2+P3] → Test[P4]
    WF4: Train[P1..P4]   → Test[P5]

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar distance-based (TP/SL)
  - Fees: 0.10% round-trip
  - Leverage: 3x
  - MC: 10,000 sign randomization sims
  - 1-position-at-a-time portfolio constraint
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
MIN_TRADES_FOR_MC = 8  # MC requires minimum trades for reliability
FIXED_REMOVE = ['MD-DN-MU', 'MD-ST-BD', 'U-MD-GS']

np.random.seed(42)

print("=" * 70)
print("MC Filter Walk-Forward Validation")
print("=" * 70)

# ============================================================
# Build 70% TP portfolio
# ============================================================
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", round(tp * TP_SCALE, 2), sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", round(tp * TP_SCALE, 2), sl)

print(f"Portfolio: {len(PORTFOLIO)} patterns "
      f"({len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S)")

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Data: {n_bars} bars ({n_bars // 288} days)")

print("Classifying candles...")
body_abs = np.abs(closes - opens)
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20[i] if not pd.isna(avg_body_20[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

print("Done.\n")


# ============================================================
# Core engine
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Single-pattern backtest → list of per-trade PnLs."""
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
    """Sign randomization MC test → p-value."""
    if len(pnl_list) < MIN_TRADES_FOR_MC:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def collect_trades_1pos(port, bar_min=0, bar_max=None):
    """Portfolio backtest with 1-position-at-a-time.
    Only considers signals with signal_bar in [bar_min, bar_max]."""
    if bar_max is None:
        bar_max = n_bars - 1
    raw = []
    for i in range(max(2, bar_min), min(bar_max + 1, n_bars)):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in port:
            continue
        direction, tp, sl = port[pat]
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
                raw.append((entry_bar, j, pnl))
                break
            elif ht:
                raw.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                raw.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                break
    raw.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in raw:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def eval_trades(trades):
    """Compute portfolio stats from trade list."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0}
    pnls = [t[2] for t in trades]
    cum = 0
    peak = 0
    max_dd = 0
    win_sum = 0
    loss_sum = 0
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
        'wr': round(wins / len(pnls) * 100, 1),
        'mdd': round(max_dd, 2),
        'pf': round(win_sum / loss_sum, 2) if loss_sum > 0 else 999,
        'pnl_mdd': round(cum / max_dd, 1) if max_dd > 0 else 999,
    }


# ============================================================
# Temporal period boundaries
# ============================================================
period_size = n_bars // N_PERIODS
periods = []
for p in range(N_PERIODS):
    s = p * period_size
    e = (p + 1) * period_size - 1 if p < N_PERIODS - 1 else n_bars - 1
    periods.append((s, e))
    d0 = df['timestamp'].iloc[s].strftime('%Y-%m-%d')
    d1 = df['timestamp'].iloc[e].strftime('%Y-%m-%d')
    print(f"  P{p+1}: bars {s:>6}-{e:>6}  ({d0} ~ {d1}, {(e-s)//288}d)")


# ============================================================
# Phase 1: Expanding-Window Walk-Forward
# ============================================================
print("\n" + "=" * 70)
print("Phase 1: Expanding-Window Walk-Forward")
print("=" * 70)

wf_results = []

for wf_i in range(N_PERIODS - 1):
    test_idx = wf_i + 1
    train_start = periods[0][0]
    train_end = periods[wf_i][1]
    test_start = periods[test_idx][0]
    test_end = periods[test_idx][1]
    train_days = (train_end - train_start) // 288
    test_days = (test_end - test_start) // 288

    print(f"\n--- WF {wf_i+1}: Train P1~P{wf_i+1} ({train_days}d) → Test P{test_idx+1} ({test_days}d) ---")

    # Step 1: Per-pattern MC on train data
    train_mc = {}
    for pat, (direction, tp, sl) in PORTFOLIO.items():
        if pat not in pattern_indices:
            continue
        idx = pattern_indices[pat]
        ti = idx[(idx >= train_start) & (idx <= train_end)]
        pnls = bt_fixed(ti, direction, tp, sl)
        mc_p = mc_test(pnls)
        train_mc[pat] = {'mc_p': mc_p, 'n_trades': len(pnls)}

    # Step 2: Build filtered portfolios
    mc_fail_001 = [p for p, r in train_mc.items() if r['mc_p'] >= 0.01]
    mc_fail_005 = [p for p, r in train_mc.items() if r['mc_p'] >= 0.05]

    port_a = PORTFOLIO  # All 52
    port_b = {p: v for p, v in PORTFOLIO.items() if p not in mc_fail_001}
    port_c = {p: v for p, v in PORTFOLIO.items() if p not in mc_fail_005}
    port_d = {p: v for p, v in PORTFOLIO.items() if p not in FIXED_REMOVE}

    print(f"  MC fail (train p>=0.01): {len(mc_fail_001)}")
    print(f"  MC fail (train p>=0.05): {len(mc_fail_005)}")
    print(f"  Portfolios: A={len(port_a)}, B={len(port_b)}, C={len(port_c)}, D={len(port_d)}")

    # Step 3: Portfolio backtest on test period
    trades_a = collect_trades_1pos(port_a, test_start, test_end)
    trades_b = collect_trades_1pos(port_b, test_start, test_end)
    trades_c = collect_trades_1pos(port_c, test_start, test_end)
    trades_d = collect_trades_1pos(port_d, test_start, test_end)

    sa = eval_trades(trades_a)
    sb = eval_trades(trades_b)
    sc = eval_trades(trades_c)
    sd = eval_trades(trades_d)

    print(f"\n  OOS Results (P{test_idx+1}):")
    print(f"  {'Strategy':<25} {'PnL':>8} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8}")
    print(f"  {'-'*73}")
    for name, st in [('A: All 52', sa), ('B: MC<0.01 filter', sb),
                      ('C: MC<0.05 filter', sc), ('D: Fixed-3 removed', sd)]:
        print(f"  {name:<25} {st['pnl']:>+8.1f} {st['trades']:>7} "
              f"{st['wr']:>5.1f}% {st['mdd']:>6.1f}% {st['pf']:>6.2f} {st['pnl_mdd']:>7.1f}x")

    # Step 4: Track which patterns were removed by B
    b_removed_that_traded = []
    for pat in mc_fail_001:
        if pat in pattern_indices:
            ti = pattern_indices[pat]
            test_signals = ti[(ti >= test_start) & (ti <= test_end)]
            if len(test_signals) > 0:
                b_removed_that_traded.append(pat)

    wf_results.append({
        'iteration': wf_i + 1,
        'train_periods': f"P1~P{wf_i+1}",
        'test_period': f"P{test_idx+1}",
        'train_days': train_days,
        'test_days': test_days,
        'mc_fail_001': mc_fail_001,
        'mc_fail_005': mc_fail_005,
        'mc_fail_001_count': len(mc_fail_001),
        'mc_fail_005_count': len(mc_fail_005),
        'b_removed_with_test_signals': b_removed_that_traded,
        'strategy_a': sa,
        'strategy_b': sb,
        'strategy_c': sc,
        'strategy_d': sd,
    })


# ============================================================
# Phase 2: Aggregate OOS Results
# ============================================================
print("\n" + "=" * 70)
print("Phase 2: Aggregate OOS Results")
print("=" * 70)

agg = {}
for sk, sn in [('strategy_a', 'A'), ('strategy_b', 'B'),
               ('strategy_c', 'C'), ('strategy_d', 'D')]:
    total_pnl = sum(r[sk]['pnl'] for r in wf_results)
    total_trades = sum(r[sk]['trades'] for r in wf_results)
    total_wins = sum(r[sk]['trades'] * r[sk]['wr'] / 100 for r in wf_results)
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    max_mdd = max(r[sk]['mdd'] for r in wf_results)
    profitable_folds = sum(1 for r in wf_results if r[sk]['pnl'] > 0)
    avg_pf = float(np.mean([r[sk]['pf'] for r in wf_results if r[sk]['pf'] < 999]))
    per_fold_pnl = [r[sk]['pnl'] for r in wf_results]

    agg[sn] = {
        'total_pnl': round(total_pnl, 2),
        'total_trades': total_trades,
        'avg_wr': round(avg_wr, 1),
        'max_mdd': round(max_mdd, 2),
        'avg_pf': round(avg_pf, 2),
        'profitable_folds': profitable_folds,
        'per_fold_pnl': [round(p, 2) for p in per_fold_pnl],
    }

labels = {'A': 'All 52', 'B': 'MC<0.01 filter', 'C': 'MC<0.05 filter', 'D': 'Fixed-3 removed'}
print(f"\n  {'Strategy':<25} {'OOS PnL':>8} {'Trades':>7} {'WR':>6} {'MaxMDD':>7} "
      f"{'AvgPF':>6} {'Folds+':>7}")
print(f"  {'-'*73}")
for key in ['A', 'B', 'C', 'D']:
    a = agg[key]
    print(f"  {key}: {labels[key]:<20} {a['total_pnl']:>+8.1f} {a['total_trades']:>7} "
          f"{a['avg_wr']:>5.1f}% {a['max_mdd']:>6.1f}% {a['avg_pf']:>6.2f} "
          f"{a['profitable_folds']}/{len(wf_results)}")

# Per-fold comparison A vs B
print(f"\n  Per-fold PnL comparison (A vs B):")
print(f"  {'Fold':<10} {'A PnL':>8} {'B PnL':>8} {'B-A':>8} {'B wins?':>8}")
print(f"  {'-'*45}")
b_wins = 0
for i, r in enumerate(wf_results):
    a_pnl = r['strategy_a']['pnl']
    b_pnl = r['strategy_b']['pnl']
    diff = b_pnl - a_pnl
    winner = 'B' if diff > 0 else 'A'
    if diff > 0:
        b_wins += 1
    print(f"  WF{i+1:<7} {a_pnl:>+8.1f} {b_pnl:>+8.1f} {diff:>+8.1f} {'  ✓' if diff > 0 else '  ✗':>8}")
print(f"  B wins: {b_wins}/{len(wf_results)} folds")


# ============================================================
# Phase 3: MC Failure Stability Across Folds
# ============================================================
print("\n" + "=" * 70)
print("Phase 3: MC Failure Stability")
print("=" * 70)

fail_freq = defaultdict(int)
for r in wf_results:
    for pat in r['mc_fail_001']:
        fail_freq[pat] += 1

n_iters = len(wf_results)
always_fail = sorted([p for p, c in fail_freq.items() if c == n_iters])
mostly_fail = sorted([p for p, c in fail_freq.items() if c >= n_iters - 1 and c < n_iters])
sometimes_fail = sorted([p for p, c in fail_freq.items() if c < n_iters - 1])
never_fail = sorted([p for p in PORTFOLIO if p not in fail_freq])

print(f"\nAlways fail (all {n_iters} folds): {len(always_fail)}")
for p in always_fail:
    print(f"  {p:<16} (fail {fail_freq[p]}/{n_iters})")

print(f"\nMostly fail ({n_iters-1}/{n_iters}): {len(mostly_fail)}")
for p in mostly_fail:
    print(f"  {p:<16} (fail {fail_freq[p]}/{n_iters})")

print(f"\nSometimes fail: {len(sometimes_fail)}")
for p in sometimes_fail:
    print(f"  {p:<16} (fail {fail_freq[p]}/{n_iters})")

print(f"\nNever fail: {len(never_fail)}/{len(PORTFOLIO)}")


# ============================================================
# Phase 4: "Always-Fail" Portfolio Test
# ============================================================
print("\n" + "=" * 70)
print("Phase 4: Always-Fail Removal — Full Portfolio Comparison")
print("=" * 70)

# Strategy E: Remove only "always fail" patterns (most conservative filter)
always_fail_set = set(always_fail)
port_e = {p: v for p, v in PORTFOLIO.items() if p not in always_fail_set}
print(f"\nStrategy E: Remove {len(always_fail)} always-fail patterns → {len(port_e)} remain")

# Rerun WF for strategy E
e_folds = []
for wf_i in range(N_PERIODS - 1):
    test_idx = wf_i + 1
    test_start = periods[test_idx][0]
    test_end = periods[test_idx][1]
    trades_e = collect_trades_1pos(port_e, test_start, test_end)
    se = eval_trades(trades_e)
    e_folds.append(se)

total_e_pnl = sum(s['pnl'] for s in e_folds)
total_e_trades = sum(s['trades'] for s in e_folds)
total_e_wins = sum(s['trades'] * s['wr'] / 100 for s in e_folds)
avg_e_wr = total_e_wins / total_e_trades * 100 if total_e_trades > 0 else 0
max_e_mdd = max(s['mdd'] for s in e_folds)
e_profitable = sum(1 for s in e_folds if s['pnl'] > 0)

agg['E'] = {
    'total_pnl': round(total_e_pnl, 2),
    'total_trades': total_e_trades,
    'avg_wr': round(avg_e_wr, 1),
    'max_mdd': round(max_e_mdd, 2),
    'avg_pf': round(float(np.mean([s['pf'] for s in e_folds if s['pf'] < 999])), 2),
    'profitable_folds': e_profitable,
    'per_fold_pnl': [round(s['pnl'], 2) for s in e_folds],
}

print(f"\n  {'Strategy':<30} {'OOS PnL':>8} {'Trades':>7} {'WR':>6} {'MaxMDD':>7}")
print(f"  {'-'*60}")
print(f"  {'A: All 52':<30} {agg['A']['total_pnl']:>+8.1f} {agg['A']['total_trades']:>7} "
      f"{agg['A']['avg_wr']:>5.1f}% {agg['A']['max_mdd']:>6.1f}%")
print(f"  {'E: Always-fail removed':<30} {agg['E']['total_pnl']:>+8.1f} {agg['E']['total_trades']:>7} "
      f"{agg['E']['avg_wr']:>5.1f}% {agg['E']['max_mdd']:>6.1f}%")
print(f"  {'B: All MC<0.01 filter':<30} {agg['B']['total_pnl']:>+8.1f} {agg['B']['total_trades']:>7} "
      f"{agg['B']['avg_wr']:>5.1f}% {agg['B']['max_mdd']:>6.1f}%")


# ============================================================
# Phase 5: Full-Sample Reference (In-Sample, for context)
# ============================================================
print("\n" + "=" * 70)
print("Phase 5: Full-Sample Reference (In-Sample)")
print("=" * 70)

full_mc = {}
for pat, (direction, tp, sl) in PORTFOLIO.items():
    if pat not in pattern_indices:
        continue
    pnls = bt_fixed(pattern_indices[pat], direction, tp, sl)
    mc_p = mc_test(pnls)
    full_mc[pat] = mc_p

full_fail = sorted([p for p, mc in full_mc.items() if mc >= 0.01])
print(f"Full-sample MC failures: {len(full_fail)}/{len(PORTFOLIO)}")

# Cross-reference: full-sample fails vs WF always-fails
always_and_full = sorted(set(always_fail) & set(full_fail))
full_not_always = sorted(set(full_fail) - set(always_fail))
always_not_full = sorted(set(always_fail) - set(full_fail))

print(f"\n  Full-sample fail AND always WF fail: {len(always_and_full)}")
for p in always_and_full:
    print(f"    {p}")
print(f"  Full-sample fail but NOT always WF fail: {len(full_not_always)}")
for p in full_not_always:
    print(f"    {p} (WF fail {fail_freq.get(p, 0)}/{n_iters})")
print(f"  Always WF fail but NOT full-sample fail: {len(always_not_full)}")
for p in always_not_full:
    print(f"    {p} (full MC p={full_mc.get(p, 'N/A')})")

# Full-sample portfolio comparison
port_e_full = {p: v for p, v in PORTFOLIO.items() if p not in always_fail_set}
trades_a_f = collect_trades_1pos(PORTFOLIO)
trades_b_f = collect_trades_1pos({p: v for p, v in PORTFOLIO.items() if full_mc.get(p, 1) < 0.01})
trades_d_f = collect_trades_1pos({p: v for p, v in PORTFOLIO.items() if p not in FIXED_REMOVE})
trades_e_f = collect_trades_1pos(port_e_full)

sa_f = eval_trades(trades_a_f)
sb_f = eval_trades(trades_b_f)
sd_f = eval_trades(trades_d_f)
se_f = eval_trades(trades_e_f)

print(f"\n  Full-Sample (In-Sample, 270d):")
print(f"  {'Strategy':<30} {'PnL':>8} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8}")
print(f"  {'-'*78}")
for name, st in [('A: All 52', sa_f), ('B: MC<0.01 filter', sb_f),
                  ('D: Fixed-3 removed', sd_f), ('E: Always-fail removed', se_f)]:
    print(f"  {name:<30} {st['pnl']:>+8.1f} {st['trades']:>7} "
          f"{st['wr']:>5.1f}% {st['mdd']:>6.1f}% {st['pf']:>6.2f} {st['pnl_mdd']:>7.1f}x")

full_sample = {
    'strategy_a': sa_f, 'strategy_b': sb_f,
    'strategy_d': sd_f, 'strategy_e': se_f,
    'mc_fail_patterns': full_fail,
    'always_fail_overlap': always_and_full,
}


# ============================================================
# Phase 6: Portfolio MC on OOS Trades
# ============================================================
print("\n" + "=" * 70)
print("Phase 6: Portfolio-Level MC on OOS Trades")
print("=" * 70)

# Collect all OOS trades for each strategy and run portfolio MC
oos_mc = {}
for wf_i in range(N_PERIODS - 1):
    test_idx = wf_i + 1
    test_start = periods[test_idx][0]
    test_end = periods[test_idx][1]

    # Only rebuild B for this fold's specific filter
    mc_fail_this = wf_results[wf_i]['mc_fail_001']
    port_b_this = {p: v for p, v in PORTFOLIO.items() if p not in mc_fail_this}

    for label, port in [('A', PORTFOLIO), ('B', port_b_this),
                        ('D', {p: v for p, v in PORTFOLIO.items() if p not in FIXED_REMOVE}),
                        ('E', port_e)]:
        if label not in oos_mc:
            oos_mc[label] = []
        trades = collect_trades_1pos(port, test_start, test_end)
        oos_mc[label].extend(trades)

for label in ['A', 'B', 'D', 'E']:
    pnl_arr = np.array([t[2] for t in oos_mc[label]])
    if len(pnl_arr) < 5:
        print(f"  {label}: too few trades for MC")
        continue
    real_sum = np.sum(pnl_arr)
    signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_arr)))
    mc_p = float(np.mean(np.sum(pnl_arr * signs, axis=1) >= real_sum))
    print(f"  {label}: {len(pnl_arr)} OOS trades, PnL={real_sum:+.1f}%, MC p={mc_p:.4f} "
          f"{'✓ PASS' if mc_p < 0.01 else '✗ FAIL'}")


# ============================================================
# Save results
# ============================================================
results = {
    'meta': {
        'script': 'mc_filter_wf_validation.py',
        'timestamp': datetime.now().isoformat(),
        'n_bars': n_bars,
        'n_periods': N_PERIODS,
        'n_wf_iterations': n_iters,
        'tp_scale': TP_SCALE,
        'mc_sims': MC_SIMS,
        'min_trades_for_mc': MIN_TRADES_FOR_MC,
        'random_seed': 42,
        'strategies': {
            'A': 'All 52 patterns (v1.27.0 baseline)',
            'B': 'WF-adaptive MC filter (train p>=0.01 removed)',
            'C': 'WF-adaptive MC filter (train p>=0.05 removed)',
            'D': f'Fixed removal of {FIXED_REMOVE}',
            'E': f'Always-fail removal ({len(always_fail)} patterns)',
        },
    },
    'wf_iterations': wf_results,
    'aggregate_oos': agg,
    'mc_stability': {
        'always_fail': always_fail,
        'mostly_fail': mostly_fail,
        'sometimes_fail': sometimes_fail,
        'never_fail_count': len(never_fail),
        'fail_frequency': dict(sorted(fail_freq.items(), key=lambda x: -x[1])),
    },
    'full_sample': full_sample,
}

out_path = Path(__file__).resolve().parent / '../../results/mc_filter_wf_validation.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✅ Saved: {out_path}")


# ============================================================
# Final Verdict
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

best_pnl = max(agg.items(), key=lambda x: x[1]['total_pnl'])
pnl_b_vs_a = agg['B']['total_pnl'] - agg['A']['total_pnl']
pnl_e_vs_a = agg['E']['total_pnl'] - agg['A']['total_pnl']
pnl_d_vs_a = agg['D']['total_pnl'] - agg['A']['total_pnl']

print(f"\n  Best OOS PnL: Strategy {best_pnl[0]} ({best_pnl[1]['total_pnl']:+.1f}%)")
print(f"\n  B (MC<0.01 filter) vs A: {pnl_b_vs_a:+.1f}% {'HELPS' if pnl_b_vs_a > 0 else 'HURTS'}")
print(f"  D (Fixed-3 removed) vs A: {pnl_d_vs_a:+.1f}% {'HELPS' if pnl_d_vs_a > 0 else 'HURTS'}")
print(f"  E (Always-fail removed) vs A: {pnl_e_vs_a:+.1f}% {'HELPS' if pnl_e_vs_a > 0 else 'HURTS'}")

if pnl_b_vs_a <= 0 and pnl_e_vs_a <= 0:
    print("\n  → MC 필터링은 OOS에서 개선 효과 없음. 52개 전체 유지 권장.")
elif pnl_e_vs_a > 0:
    print(f"\n  → Always-fail {len(always_fail)}개 제거가 OOS에서 유효. 추가 검증 권장.")
else:
    print(f"\n  → 결과가 혼합적. 추가 분석 필요.")

print("\n" + "=" * 70)
