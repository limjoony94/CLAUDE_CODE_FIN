"""
New Pattern WF Validation
=========================
reverse_edge_research.py에서 발견된 63개 신규 MC<0.01 패턴의
TP/SL 최적화 + Walk-Forward 검증 + 포트폴리오 통합 테스트.

Phase 1: TP/SL Grid Search (개별 패턴)
  - 각 후보에 대해 TP/SL 그리드 탐색
  - MC < 0.01 + edge >= 5pp 필터

Phase 2: Walk-Forward Validation (개별)
  - Expanding-window WF (5 periods, 4 iterations)
  - 개별 패턴 OOS WR, PnL 검증

Phase 3: Portfolio Integration (52 + new)
  - 기존 52패턴 + WF 통과 패턴 통합
  - 1-pos-at-a-time 포트폴리오 WF OOS 비교

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
MIN_TRADES_GRID = 10
MIN_EDGE_PP = 5.0

np.random.seed(42)

print("=" * 70)
print("New Pattern WF Validation")
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

# Build current 52-pattern portfolio
PORTFOLIO_52 = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO_52[pat] = ("LONG", round(tp * TP_SCALE, 2), sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO_52[pat] = ("SHORT", round(tp * TP_SCALE, 2), sl)

# Load 63 candidates
with open(Path(__file__).resolve().parent / '../../results/reverse_edge_research.json') as f:
    rev_data = json.load(f)
candidates = rev_data['phase2_new_significant']
print(f"Candidates from reverse_edge_research: {len(candidates)}")

# ============================================================
# Shared helpers
# ============================================================
def simulate_trade(direction, tp, sl, signal_bar):
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


def get_signal_bars(pat_name, bar_min=0, bar_max=None):
    """Get all signal bars for a pattern."""
    if bar_max is None:
        bar_max = n_bars - 1
    bars = []
    for i in range(max(2, bar_min), min(bar_max + 1, n_bars)):
        p = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if p == pat_name:
            bars.append(i)
    return bars


def backtest_pattern(pat_name, direction, tp, sl, bar_min=0, bar_max=None):
    """Backtest single pattern, return list of (entry_bar, exit_bar, pnl)."""
    sig_bars = get_signal_bars(pat_name, bar_min, bar_max)
    trades = []
    for sb in sig_bars:
        result = simulate_trade(direction, tp, sl, sb)
        if result is not None:
            trades.append(result)
    return trades


def collect_portfolio_trades(port, bar_min=0, bar_max=None):
    """Collect all trades for portfolio, apply 1-pos filter."""
    if bar_max is None:
        bar_max = n_bars - 1
    raw = []
    for i in range(max(2, bar_min), min(bar_max + 1, n_bars)):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in port:
            continue
        direction, tp, sl = port[pat]
        result = simulate_trade(direction, tp, sl, i)
        if result is not None:
            raw.append((*result, pat))
    raw.sort(key=lambda x: x[0])
    # 1-pos filter
    filtered = []
    last_exit = -1
    for eb, xb, pnl, *rest in raw:
        if eb > last_exit:
            filtered.append((eb, xb, pnl, *rest))
            last_exit = xb
    return filtered


def eval_trades(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    pnls = [t[2] for t in trades]
    cum = peak = max_dd = 0
    win_sum = loss_sum = wins = 0
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
    }


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 8:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def distance_wr(tp, sl):
    return sl / (tp + sl) * 100 if (tp + sl) > 0 else 50.0


# WF periods
period_size = n_bars // N_PERIODS
period_bounds = [(i * period_size, (i + 1) * period_size - 1) for i in range(N_PERIODS)]
period_bounds[-1] = (period_bounds[-1][0], n_bars - 1)  # extend last

results = {}

# ============================================================
# Phase 1: TP/SL Grid Search for 63 candidates
# ============================================================
print("=" * 70)
print("Phase 1: TP/SL Grid Search Optimization")
print("=" * 70)
print()

TP_GRID = [0.35, 0.5, 0.7, 1.0, 1.4, 1.75, 2.1]
SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

phase1 = []
optimized = []

for ci, cand in enumerate(candidates):
    pat = cand['pattern']
    direction = cand['direction']

    best_score = -999
    best_tp = best_sl = best_wr = best_mc = best_trades = best_pnl = None
    best_edge = 0

    for tp in TP_GRID:
        for sl in SL_GRID:
            trades = backtest_pattern(pat, direction, tp, sl)
            pnls = [t[2] for t in trades]
            if len(pnls) < MIN_TRADES_GRID:
                continue
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            d_wr = distance_wr(tp, sl)
            edge = wr - d_wr
            total_pnl = sum(pnls)
            # Score: edge * sqrt(trades) — balance edge strength with sample size
            score = edge * np.sqrt(len(pnls))
            if score > best_score and edge >= MIN_EDGE_PP and total_pnl > 0:
                best_score = score
                best_tp = tp
                best_sl = sl
                best_wr = round(wr, 1)
                best_trades = len(pnls)
                best_pnl = round(total_pnl, 2)
                best_edge = round(edge, 1)

    if best_tp is None:
        continue

    # MC test at best TP/SL
    trades = backtest_pattern(pat, direction, best_tp, best_sl)
    pnls = [t[2] for t in trades]
    mc = mc_test(pnls)

    entry = {
        'pattern': pat,
        'direction': direction,
        'tp': best_tp,
        'sl': best_sl,
        'trades': best_trades,
        'wr': best_wr,
        'edge': best_edge,
        'pnl': best_pnl,
        'mc': round(mc, 4),
        'score': round(best_score, 1),
    }
    phase1.append(entry)

    if mc < 0.01:
        optimized.append(entry)

    if (ci + 1) % 10 == 0:
        print(f"  Processed {ci + 1}/{len(candidates)}...")

phase1.sort(key=lambda x: x['score'], reverse=True)
optimized.sort(key=lambda x: x['score'], reverse=True)

print(f"\n  Grid search complete:")
print(f"  Candidates with edge >= {MIN_EDGE_PP}pp: {len(phase1)}")
print(f"  MC < 0.01 after optimization: {len(optimized)}")
print()

if optimized:
    print(f"  Optimized patterns (MC < 0.01, edge >= {MIN_EDGE_PP}pp):")
    print(f"  {'Pattern':<14} {'Dir':<6} {'TP/SL':>8} {'Trades':>7} {'WR':>6} {'Edge':>8} {'MC':>8} {'PnL':>8}")
    print("  " + "-" * 75)
    for p in optimized:
        print(f"  {p['pattern']:<14} {p['direction']:<6} {p['tp']:.1f}/{p['sl']:.1f} "
              f"{p['trades']:>6} {p['wr']:>5.1f}% {p['edge']:>+7.1f}pp {p['mc']:>7.4f} {p['pnl']:>+7.1f}%")

results['phase1'] = phase1
results['phase1_optimized'] = optimized

# ============================================================
# Phase 2: Walk-Forward Validation (individual patterns)
# ============================================================
print()
print("=" * 70)
print("Phase 2: Walk-Forward Validation (Individual Patterns)")
print("=" * 70)
print()

phase2 = []
wf_passed = []

for entry in optimized:
    pat = entry['pattern']
    direction = entry['direction']
    tp = entry['tp']
    sl = entry['sl']

    fold_results = []
    oos_pnls = []

    for fold_idx in range(1, N_PERIODS):
        # Training: periods 0..fold_idx-1
        train_end = period_bounds[fold_idx - 1][1]
        # OOS: period fold_idx
        oos_start = period_bounds[fold_idx][0]
        oos_end = period_bounds[fold_idx][1]

        # Train: check edge on training data
        train_trades = backtest_pattern(pat, direction, tp, sl, bar_min=0, bar_max=train_end)
        train_pnls = [t[2] for t in train_trades]
        train_wr = sum(1 for p in train_pnls if p > 0) / len(train_pnls) * 100 if train_pnls else 0
        train_mc = mc_test(train_pnls) if len(train_pnls) >= 8 else 1.0
        d_wr = distance_wr(tp, sl)
        train_edge = train_wr - d_wr

        # OOS: test performance
        oos_trades = backtest_pattern(pat, direction, tp, sl, bar_min=oos_start, bar_max=oos_end)
        oos_pnl_list = [t[2] for t in oos_trades]
        oos_wr = sum(1 for p in oos_pnl_list if p > 0) / len(oos_pnl_list) * 100 if oos_pnl_list else 0
        oos_pnl_total = sum(oos_pnl_list)
        oos_edge = oos_wr - d_wr

        fold_results.append({
            'fold': fold_idx,
            'train_trades': len(train_pnls),
            'train_wr': round(train_wr, 1),
            'train_mc': round(train_mc, 4),
            'train_edge': round(train_edge, 1),
            'oos_trades': len(oos_pnl_list),
            'oos_wr': round(oos_wr, 1),
            'oos_pnl': round(oos_pnl_total, 2),
            'oos_edge': round(oos_edge, 1),
        })
        oos_pnls.extend(oos_pnl_list)

    # Aggregate OOS stats
    total_oos_pnl = sum(oos_pnls)
    total_oos_trades = len(oos_pnls)
    oos_wr_agg = sum(1 for p in oos_pnls if p > 0) / total_oos_trades * 100 if total_oos_trades else 0
    oos_edge_agg = oos_wr_agg - distance_wr(tp, sl)
    oos_mc = mc_test(oos_pnls) if len(oos_pnls) >= 8 else 1.0

    # Pass criteria: OOS edge >= 3pp AND OOS PnL > 0 AND OOS MC < 0.10
    passed = oos_edge_agg >= 3.0 and total_oos_pnl > 0 and oos_mc < 0.10

    wf_entry = {
        **entry,
        'folds': fold_results,
        'oos_total_pnl': round(total_oos_pnl, 2),
        'oos_total_trades': total_oos_trades,
        'oos_wr': round(oos_wr_agg, 1),
        'oos_edge': round(oos_edge_agg, 1),
        'oos_mc': round(oos_mc, 4),
        'wf_passed': passed,
    }
    phase2.append(wf_entry)
    if passed:
        wf_passed.append(wf_entry)

print(f"  WF tested: {len(phase2)} patterns")
print(f"  WF passed (OOS edge >= 3pp, PnL > 0, MC < 0.10): {len(wf_passed)}")
print()

if wf_passed:
    wf_passed.sort(key=lambda x: x['oos_total_pnl'], reverse=True)
    print(f"  WF-passed patterns:")
    print(f"  {'Pattern':<14} {'Dir':<6} {'TP/SL':>8} {'InSmp':>6} {'OOS_T':>6} {'OOS_WR':>7} {'OOS_Edge':>9} {'OOS_MC':>8} {'OOS_PnL':>8}")
    print("  " + "-" * 85)
    for p in wf_passed:
        print(f"  {p['pattern']:<14} {p['direction']:<6} {p['tp']:.1f}/{p['sl']:.1f} "
              f"{p['trades']:>5} {p['oos_total_trades']:>5} {p['oos_wr']:>6.1f}% "
              f"{p['oos_edge']:>+8.1f}pp {p['oos_mc']:>7.4f} {p['oos_total_pnl']:>+7.1f}%")
    print()

    # Show fold detail for top patterns
    print("  Top 5 WF-passed fold details:")
    for p in wf_passed[:5]:
        print(f"\n  {p['pattern']} ({p['direction']}) TP={p['tp']}/SL={p['sl']}:")
        for fold in p['folds']:
            status = "OK" if fold['oos_edge'] >= 0 else "FAIL"
            print(f"    Fold {fold['fold']}: train {fold['train_trades']}t WR={fold['train_wr']}% "
                  f"MC={fold['train_mc']:.3f} | OOS {fold['oos_trades']}t WR={fold['oos_wr']}% "
                  f"edge={fold['oos_edge']:+.1f}pp pnl={fold['oos_pnl']:+.1f}% [{status}]")

# Also show WF-failed patterns
wf_failed = [p for p in phase2 if not p['wf_passed']]
if wf_failed:
    print(f"\n  WF-failed patterns ({len(wf_failed)}):")
    print(f"  {'Pattern':<14} {'Dir':<6} {'TP/SL':>8} {'OOS_T':>6} {'OOS_WR':>7} {'OOS_Edge':>9} {'OOS_MC':>8} {'OOS_PnL':>8} {'Reason':<15}")
    print("  " + "-" * 95)
    for p in sorted(wf_failed, key=lambda x: x['oos_total_pnl'], reverse=True):
        reasons = []
        if p['oos_edge'] < 3.0:
            reasons.append("low edge")
        if p['oos_total_pnl'] <= 0:
            reasons.append("neg PnL")
        if p['oos_mc'] >= 0.10:
            reasons.append("MC fail")
        print(f"  {p['pattern']:<14} {p['direction']:<6} {p['tp']:.1f}/{p['sl']:.1f} "
              f"{p['oos_total_trades']:>5} {p['oos_wr']:>6.1f}% "
              f"{p['oos_edge']:>+8.1f}pp {p['oos_mc']:>7.4f} {p['oos_total_pnl']:>+7.1f}% {', '.join(reasons):<15}")

results['phase2'] = phase2
results['phase2_passed'] = [p['pattern'] + '_' + p['direction'] for p in wf_passed]

# ============================================================
# Phase 3: Portfolio Integration Test
# ============================================================
print()
print("=" * 70)
print("Phase 3: Portfolio Integration WF Test")
print("=" * 70)
print()

if not wf_passed:
    print("  No WF-passed patterns to integrate. Skipping Phase 3.\n")
    results['phase3'] = {'skipped': True, 'reason': 'no wf-passed patterns'}
else:
    # Build extended portfolio
    PORTFOLIO_EXTENDED = dict(PORTFOLIO_52)
    new_additions = []
    for p in wf_passed:
        key = p['pattern']
        if key in PORTFOLIO_EXTENDED:
            # Already exists (shouldn't happen but safety check)
            continue
        # Apply TP_SCALE to new patterns too for consistency
        PORTFOLIO_EXTENDED[key] = (p['direction'], round(p['tp'] * TP_SCALE, 2), p['sl'])
        new_additions.append(key)

    print(f"  Baseline portfolio: {len(PORTFOLIO_52)} patterns")
    print(f"  New additions: {len(new_additions)} patterns")
    print(f"  Extended portfolio: {len(PORTFOLIO_EXTENDED)} patterns")
    print()

    # WF comparison: baseline vs extended
    strategies = {
        'A_baseline_52': PORTFOLIO_52,
        'B_extended': PORTFOLIO_EXTENDED,
    }

    # Also test: only new patterns (to see standalone value)
    PORTFOLIO_NEW_ONLY = {}
    for p in wf_passed:
        key = p['pattern']
        if key not in PORTFOLIO_52:
            PORTFOLIO_NEW_ONLY[key] = (p['direction'], round(p['tp'] * TP_SCALE, 2), p['sl'])

    if PORTFOLIO_NEW_ONLY:
        strategies['C_new_only'] = PORTFOLIO_NEW_ONLY

    phase3_wf = {}

    for strat_name, port in strategies.items():
        fold_results = []
        all_oos_trades = []

        for fold_idx in range(1, N_PERIODS):
            oos_start = period_bounds[fold_idx][0]
            oos_end = period_bounds[fold_idx][1]

            oos_trades = collect_portfolio_trades(port, bar_min=oos_start, bar_max=oos_end)
            stats = eval_trades(oos_trades)
            fold_results.append(stats)
            all_oos_trades.extend(oos_trades)

        agg = eval_trades(all_oos_trades)
        oos_mc = mc_test([t[2] for t in all_oos_trades])

        phase3_wf[strat_name] = {
            'folds': fold_results,
            'aggregate': agg,
            'oos_mc': round(oos_mc, 4),
            'n_patterns': len(port),
        }

    print(f"  WF OOS Portfolio Comparison:")
    print(f"  {'Strategy':<20} {'Patterns':>9} {'OOS_PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'MC':>8}")
    print("  " + "-" * 80)
    for name, data in phase3_wf.items():
        a = data['aggregate']
        print(f"  {name:<20} {data['n_patterns']:>8} {a['pnl']:>+8.1f}% {a['trades']:>6} "
              f"{a['wr']:>5.1f}% {a['mdd']:>6.1f}% {a['pf']:>5.2f} {data['oos_mc']:>7.4f}")

    print()
    print("  Per-fold comparison:")
    for fold_idx in range(N_PERIODS - 1):
        print(f"\n  Fold {fold_idx + 1}:")
        for name, data in phase3_wf.items():
            f = data['folds'][fold_idx]
            print(f"    {name:<20} PnL={f['pnl']:>+7.1f}% T={f['trades']:>3} WR={f['wr']:>5.1f}% MDD={f['mdd']:>5.1f}%")

    # Check which folds improved
    if 'A_baseline_52' in phase3_wf and 'B_extended' in phase3_wf:
        base_folds = phase3_wf['A_baseline_52']['folds']
        ext_folds = phase3_wf['B_extended']['folds']
        folds_improved = sum(1 for b, e in zip(base_folds, ext_folds) if e['pnl'] > b['pnl'])
        folds_worse = sum(1 for b, e in zip(base_folds, ext_folds) if e['pnl'] < b['pnl'])
        pnl_diff = phase3_wf['B_extended']['aggregate']['pnl'] - phase3_wf['A_baseline_52']['aggregate']['pnl']
        mdd_diff = phase3_wf['B_extended']['aggregate']['mdd'] - phase3_wf['A_baseline_52']['aggregate']['mdd']

        print(f"\n  Integration impact:")
        print(f"  Folds improved: {folds_improved}/{N_PERIODS - 1}")
        print(f"  Folds worse:    {folds_worse}/{N_PERIODS - 1}")
        print(f"  PnL diff:       {pnl_diff:+.1f}%")
        print(f"  MDD diff:       {mdd_diff:+.1f}%")
        verdict = "BENEFICIAL" if pnl_diff > 0 and mdd_diff <= 5 and folds_improved > folds_worse else "NOT BENEFICIAL"
        print(f"  Verdict:        {verdict}")

    results['phase3'] = phase3_wf
    results['phase3_new_additions'] = new_additions

# ============================================================
# Phase 4: Full-Sample Reference (통합)
# ============================================================
print()
print("=" * 70)
print("Phase 4: Full-Sample Reference")
print("=" * 70)
print()

# Baseline 52
base_trades = collect_portfolio_trades(PORTFOLIO_52)
base_stats = eval_trades(base_trades)
base_mc = mc_test([t[2] for t in base_trades])

print(f"  Baseline (52 patterns): PnL={base_stats['pnl']:+.1f}% T={base_stats['trades']} "
      f"WR={base_stats['wr']}% MDD={base_stats['mdd']:.1f}% PF={base_stats['pf']:.2f} MC={base_mc:.4f}")

if wf_passed:
    ext_trades = collect_portfolio_trades(PORTFOLIO_EXTENDED)
    ext_stats = eval_trades(ext_trades)
    ext_mc = mc_test([t[2] for t in ext_trades])

    print(f"  Extended ({len(PORTFOLIO_EXTENDED)} patterns): PnL={ext_stats['pnl']:+.1f}% T={ext_stats['trades']} "
          f"WR={ext_stats['wr']}% MDD={ext_stats['mdd']:.1f}% PF={ext_stats['pf']:.2f} MC={ext_mc:.4f}")

    pnl_diff = ext_stats['pnl'] - base_stats['pnl']
    trade_diff = ext_stats['trades'] - base_stats['trades']
    mdd_diff = ext_stats['mdd'] - base_stats['mdd']
    print(f"  Diff: PnL {pnl_diff:+.1f}%, Trades {trade_diff:+d}, MDD {mdd_diff:+.1f}%")

    results['phase4'] = {
        'baseline': {**base_stats, 'mc': round(base_mc, 4), 'n_patterns': len(PORTFOLIO_52)},
        'extended': {**ext_stats, 'mc': round(ext_mc, 4), 'n_patterns': len(PORTFOLIO_EXTENDED)},
    }
else:
    results['phase4'] = {
        'baseline': {**base_stats, 'mc': round(base_mc, 4), 'n_patterns': len(PORTFOLIO_52)},
    }

# ============================================================
# Save
# ============================================================
out_path = Path(__file__).resolve().parent / '../../results/new_pattern_wf_validation.json'
results['timestamp'] = datetime.now().isoformat()
results['config'] = {
    'min_trades_grid': MIN_TRADES_GRID,
    'min_edge_pp': MIN_EDGE_PP,
    'tp_scale': TP_SCALE,
    'mc_sims': MC_SIMS,
    'n_periods': N_PERIODS,
    'tp_grid': TP_GRID,
    'sl_grid': SL_GRID,
}
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
print(f"  Phase 1: TP/SL grid search")
print(f"    Candidates tested: {len(candidates)}")
print(f"    Edge >= {MIN_EDGE_PP}pp: {len(phase1)}")
print(f"    MC < 0.01: {len(optimized)}")
print()
print(f"  Phase 2: Walk-Forward validation")
print(f"    WF tested: {len(phase2)}")
print(f"    WF passed: {len(wf_passed)}")
if wf_passed:
    print(f"    Passed list: {', '.join(p['pattern'] + '(' + p['direction'][0] + ')' for p in wf_passed)}")
print()
print(f"  Phase 3: Portfolio integration")
if wf_passed:
    a = phase3_wf.get('A_baseline_52', {}).get('aggregate', {})
    b = phase3_wf.get('B_extended', {}).get('aggregate', {})
    if a and b:
        print(f"    Baseline OOS: PnL={a['pnl']:+.1f}% WR={a['wr']}%")
        print(f"    Extended OOS: PnL={b['pnl']:+.1f}% WR={b['wr']}%")
        print(f"    Diff:         PnL={b['pnl'] - a['pnl']:+.1f}%")
else:
    print(f"    Skipped (no WF-passed patterns)")
print("=" * 70)
