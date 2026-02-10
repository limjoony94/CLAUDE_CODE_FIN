"""
Uniform TP 70% Deep Validation
===============================
Microstructure 연구에서 발견한 D_Uniform_70pct 전략의 심층 검증

Phase 1: Per-Pattern Validation
  - 52개 패턴 각각 TP*0.7로 MC test + WF test
  - 현재 TP vs 70% TP 비교 (패턴별 WR, PnL, EV)

Phase 2: Robustness Sweep (65%~80%)
  - 포트폴리오 레벨에서 TP 축소 비율 65/67/70/73/75/80% 비교
  - 70%가 "최적점"인지 vs "넓은 plateau" 위인지 확인

Phase 3: Portfolio Monte Carlo (Trade Order Shuffle)
  - 거래 순서 10,000회 셔플 → MDD 분포
  - P50/P90/P95/P99 MDD 산출
  - Current vs 70% TP MDD 분포 비교

Phase 4: Walk-Forward Portfolio Test
  - 270일을 5등분 → 각 구간에서 포트폴리오 성과
  - 전 구간 수익 여부 확인

Phase 5: Edge Preservation Check
  - 70% TP에서도 genuine edge가 유지되는지
  - Random walk WR vs actual WR 비교 (distance-edge decomposition 재검증)

Phase 6: Consecutive Loss Analysis
  - 현재 vs 70% TP에서 최대 연속 손실 비교
  - 연속 손실 발생 빈도 분포

Phase 7: Final Recommendation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL, PATTERN_STATS, BOT_VERSION,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
TP_SCALE = 0.70  # Primary target: 70% of current TP

print("=" * 70)
print(f"Uniform TP {int(TP_SCALE*100)}% Deep Validation (from v{BOT_VERSION})")
print("=" * 70)

# ============================================================
# Build portfolio maps
# ============================================================
PORTFOLIO_CURRENT = {}
PORTFOLIO_70 = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO_CURRENT[pat] = ("LONG", tp, sl)
    PORTFOLIO_70[pat] = ("LONG", round(tp * TP_SCALE, 2), sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO_CURRENT[pat] = ("SHORT", tp, sl)
    PORTFOLIO_70[pat] = ("SHORT", round(tp * TP_SCALE, 2), sl)

print(f"Patterns: {len(PORTFOLIO_CURRENT)} ({len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S)")

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"Loaded: {len(df)} bars")

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

print("Done.\n")


# ============================================================
# Backtest engine (reused from portfolio_pruning_v4)
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


def collect_trades_1pos(pattern_map):
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
    for entry_bar, exit_bar, pnl in raw_trades:
        if entry_bar > last_exit:
            filtered.append((entry_bar, exit_bar, pnl))
            last_exit = exit_bar
    return filtered


def eval_trades_simple(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] for t in trades]
    cum_pnl = 0
    peak_pnl = 0
    max_dd = 0
    wins = 0
    win_pnls = []
    loss_pnls = []
    for p in pnl_list:
        cum_pnl += p
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_pnls.append(p)
        else:
            loss_pnls.append(abs(p))
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    total_win = sum(win_pnls)
    total_loss = sum(loss_pnls)
    wr_pct = wins / len(pnl_list) * 100
    expectancy = (wr_pct / 100 * avg_win) - ((1 - wr_pct / 100) * avg_loss)
    return {
        'pnl': cum_pnl,
        'trades': len(pnl_list),
        'wr': wr_pct,
        'mdd': max_dd,
        'pf': total_win / total_loss if total_loss > 0 else 999,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'pnl_mdd': cum_pnl / max_dd if max_dd > 0 else 999,
    }


# ============================================================
# Phase 1: Per-Pattern Validation
# ============================================================
print("=" * 70)
print("Phase 1: Per-Pattern Validation — Current TP vs 70% TP")
print("=" * 70)

phase1 = []
mc_fail_count = 0
wf_degrade_count = 0

print(f"\n  {'Pattern':<16} {'Dir':<6} {'TP_cur':>6} {'TP_70':>6} {'SL':>5}  "
      f"{'WR_cur':>6} {'WR_70':>6} {'MC_cur':>7} {'MC_70':>7} {'WF_cur':>6} {'WF_70':>6}")
print("  " + "-" * 100)

for pat in sorted(PORTFOLIO_CURRENT.keys()):
    direction, tp_cur, sl = PORTFOLIO_CURRENT[pat]
    tp_70 = round(tp_cur * TP_SCALE, 2)

    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 5:
        continue

    # Current TP
    pnls_cur = bt_fixed(indices, direction, tp_cur, sl)
    mc_cur = mc_test(pnls_cur)
    wf_cur = walk_forward_test(indices, direction, tp_cur, sl)
    wr_cur = sum(1 for p in pnls_cur if p > 0) / len(pnls_cur) * 100 if pnls_cur else 0

    # 70% TP
    pnls_70 = bt_fixed(indices, direction, tp_70, sl)
    mc_70 = mc_test(pnls_70)
    wf_70 = walk_forward_test(indices, direction, tp_70, sl)
    wr_70 = sum(1 for p in pnls_70 if p > 0) / len(pnls_70) * 100 if pnls_70 else 0

    flag = ""
    if mc_70 >= 0.01:
        flag += " MC_FAIL"
        mc_fail_count += 1
    if wf_70 < wf_cur:
        flag += " WF_DOWN"
        wf_degrade_count += 1

    print(f"  {pat:<16} {direction:<6} {tp_cur:>6.1f} {tp_70:>6.2f} {sl:>5.1f}  "
          f"{wr_cur:>5.1f}% {wr_70:>5.1f}% {mc_cur:>7.4f} {mc_70:>7.4f} {wf_cur:>4d}/5 {wf_70:>4d}/5{flag}")

    phase1.append({
        'pattern': pat, 'direction': direction,
        'tp_cur': tp_cur, 'tp_70': tp_70, 'sl': sl,
        'wr_cur': wr_cur, 'wr_70': wr_70,
        'mc_cur': mc_cur, 'mc_70': mc_70,
        'wf_cur': wf_cur, 'wf_70': wf_70,
        'pnl_cur': sum(pnls_cur), 'pnl_70': sum(pnls_70),
    })

print(f"\n  Summary:")
print(f"    MC failures (mc >= 0.01 at 70% TP): {mc_fail_count}/{len(phase1)}")
print(f"    WF degraded (fewer profitable folds): {wf_degrade_count}/{len(phase1)}")
avg_wr_change = np.mean([p['wr_70'] - p['wr_cur'] for p in phase1])
print(f"    Average WR change: {avg_wr_change:+.1f}pp")


# ============================================================
# Phase 2: Robustness Sweep (65% ~ 80%)
# ============================================================
print("\n" + "=" * 70)
print("Phase 2: Robustness Sweep — Portfolio PnL at Various TP Scales")
print("=" * 70)

scales = [0.60, 0.65, 0.67, 0.70, 0.73, 0.75, 0.80, 0.85, 0.90, 1.00]
phase2 = []

print(f"\n  {'Scale':>6}  {'PnL':>10}  {'Trades':>7}  {'WR':>7}  {'MDD':>7}  {'PF':>6}  {'PnL/MDD':>8}")
print("  " + "-" * 65)

for scale in scales:
    port = {}
    for pat in PORTFOLIO_CURRENT:
        direction, tp, sl = PORTFOLIO_CURRENT[pat]
        port[pat] = (direction, round(tp * scale, 2), sl)

    trades = collect_trades_1pos(port)
    stats = eval_trades_simple(trades)

    marker = " <<<" if scale == TP_SCALE else (" [current]" if scale == 1.0 else "")
    print(f"  {scale*100:>5.0f}%  {stats['pnl']:>+9.1f}%  {stats['trades']:>7d}  "
          f"{stats['wr']:>6.1f}%  {stats['mdd']:>6.1f}%  {stats['pf']:>5.2f}  {stats['pnl_mdd']:>7.1f}x{marker}")

    phase2.append({
        'scale': scale,
        'pnl': stats['pnl'], 'trades': stats['trades'],
        'wr': stats['wr'], 'mdd': stats['mdd'],
        'pf': stats['pf'], 'pnl_mdd': stats['pnl_mdd'],
    })

# Find optimal scale
best_pnl_mdd = max(phase2, key=lambda x: x['pnl_mdd'])
best_pnl = max(phase2, key=lambda x: x['pnl'])
print(f"\n  Best PnL/MDD: {best_pnl_mdd['scale']*100:.0f}% ({best_pnl_mdd['pnl_mdd']:.1f}x)")
print(f"  Best PnL:     {best_pnl['scale']*100:.0f}% ({best_pnl['pnl']:+.1f}%)")

# Check plateau width
plateau_scores = [p for p in phase2 if p['pnl_mdd'] >= best_pnl_mdd['pnl_mdd'] * 0.90]
print(f"  90% of best PnL/MDD: {len(plateau_scores)} scale points "
      f"({min(p['scale'] for p in plateau_scores)*100:.0f}%~{max(p['scale'] for p in plateau_scores)*100:.0f}%)")


# ============================================================
# Phase 3: Portfolio Monte Carlo — MDD Distribution
# ============================================================
print("\n" + "=" * 70)
print("Phase 3: Portfolio Monte Carlo — MDD Distribution (Trade Order Shuffle)")
print("=" * 70)

def portfolio_mc_mdd(trades, n_sims=10000):
    """Shuffle trade order → compute MDD for each shuffle."""
    pnl_list = np.array([t[2] for t in trades])
    mdds = []
    for _ in range(n_sims):
        shuffled = np.random.permutation(pnl_list)
        cum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        mdds.append(np.max(dd))
    return np.array(mdds)

t0 = time.time()

# Current portfolio
trades_current = collect_trades_1pos(PORTFOLIO_CURRENT)
stats_current = eval_trades_simple(trades_current)
mdds_current = portfolio_mc_mdd(trades_current)

# 70% TP portfolio
trades_70 = collect_trades_1pos(PORTFOLIO_70)
stats_70 = eval_trades_simple(trades_70)
mdds_70 = portfolio_mc_mdd(trades_70)

elapsed = time.time() - t0

print(f"\n  (10,000 sims x 2 portfolios completed in {elapsed:.1f}s)\n")
print(f"  {'Metric':<20} {'Current':>12} {'TP 70%':>12}")
print("  " + "-" * 50)
print(f"  {'Actual MDD':<20} {stats_current['mdd']:>11.1f}% {stats_70['mdd']:>11.1f}%")
for pctl, label in [(50, 'P50'), (75, 'P75'), (90, 'P90'), (95, 'P95'), (99, 'P99')]:
    v_cur = np.percentile(mdds_current, pctl)
    v_70 = np.percentile(mdds_70, pctl)
    print(f"  MC MDD {label:<12} {v_cur:>11.1f}% {v_70:>11.1f}%")

print(f"\n  MDD < 20% probability:")
print(f"    Current: {np.mean(mdds_current < 20) * 100:.1f}%")
print(f"    TP 70%:  {np.mean(mdds_70 < 20) * 100:.1f}%")
print(f"  MDD < 30% probability:")
print(f"    Current: {np.mean(mdds_current < 30) * 100:.1f}%")
print(f"    TP 70%:  {np.mean(mdds_70 < 30) * 100:.1f}%")

phase3 = {
    'current': {
        'actual_mdd': stats_current['mdd'],
        'p50': float(np.percentile(mdds_current, 50)),
        'p90': float(np.percentile(mdds_current, 90)),
        'p95': float(np.percentile(mdds_current, 95)),
        'p99': float(np.percentile(mdds_current, 99)),
    },
    'tp70': {
        'actual_mdd': stats_70['mdd'],
        'p50': float(np.percentile(mdds_70, 50)),
        'p90': float(np.percentile(mdds_70, 90)),
        'p95': float(np.percentile(mdds_70, 95)),
        'p99': float(np.percentile(mdds_70, 99)),
    },
}


# ============================================================
# Phase 4: Walk-Forward Portfolio Test (5-fold temporal)
# ============================================================
print("\n" + "=" * 70)
print("Phase 4: Walk-Forward Portfolio Test (5-fold temporal)")
print("=" * 70)

def wf_portfolio(trades, n_folds=5):
    """Split trades temporally into n_folds, check each fold profitable."""
    if not trades:
        return 0, []
    sorted_trades = sorted(trades, key=lambda x: x[0])
    fold_size = len(sorted_trades) // n_folds
    results = []
    for f in range(n_folds):
        s = f * fold_size
        e = s + fold_size if f < n_folds - 1 else len(sorted_trades)
        fold_trades = sorted_trades[s:e]
        fold_pnl = sum(t[2] for t in fold_trades)
        fold_wr = sum(1 for t in fold_trades if t[2] > 0) / len(fold_trades) * 100
        results.append({'fold': f + 1, 'trades': len(fold_trades), 'pnl': fold_pnl, 'wr': fold_wr})
    profitable_folds = sum(1 for r in results if r['pnl'] > 0)
    return profitable_folds, results

wf_cur, wf_cur_detail = wf_portfolio(trades_current)
wf_70, wf_70_detail = wf_portfolio(trades_70)

print(f"\n  Current Portfolio WF: {wf_cur}/5 profitable folds")
for r in wf_cur_detail:
    marker = " +" if r['pnl'] > 0 else " -"
    print(f"    Fold {r['fold']}: {r['trades']:>3d} trades, PnL {r['pnl']:>+8.1f}%, WR {r['wr']:>5.1f}%{marker}")

print(f"\n  TP 70% Portfolio WF: {wf_70}/5 profitable folds")
for r in wf_70_detail:
    marker = " +" if r['pnl'] > 0 else " -"
    print(f"    Fold {r['fold']}: {r['trades']:>3d} trades, PnL {r['pnl']:>+8.1f}%, WR {r['wr']:>5.1f}%{marker}")

phase4 = {
    'current': {'profitable_folds': wf_cur, 'folds': wf_cur_detail},
    'tp70': {'profitable_folds': wf_70, 'folds': wf_70_detail},
}


# ============================================================
# Phase 5: Edge Preservation Check
# ============================================================
print("\n" + "=" * 70)
print("Phase 5: Edge Preservation — Does genuine edge survive at 70% TP?")
print("=" * 70)

edge_current = []
edge_70 = []

for pat in sorted(PORTFOLIO_CURRENT.keys()):
    direction, tp_cur, sl = PORTFOLIO_CURRENT[pat]
    tp_70 = round(tp_cur * TP_SCALE, 2)

    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 5:
        continue

    # Random walk WR at current TP
    rw_wr_cur = sl / (tp_cur + sl) * 100
    pnls_cur = bt_fixed(indices, direction, tp_cur, sl)
    act_wr_cur = sum(1 for p in pnls_cur if p > 0) / len(pnls_cur) * 100 if pnls_cur else 0
    edge_cur = act_wr_cur - rw_wr_cur

    # Random walk WR at 70% TP (TP closer → higher RW WR)
    rw_wr_70 = sl / (tp_70 + sl) * 100
    pnls_70 = bt_fixed(indices, direction, tp_70, sl)
    act_wr_70 = sum(1 for p in pnls_70 if p > 0) / len(pnls_70) * 100 if pnls_70 else 0
    edge_70_val = act_wr_70 - rw_wr_70

    edge_current.append({'pattern': pat, 'rw_wr': rw_wr_cur, 'act_wr': act_wr_cur, 'edge': edge_cur})
    edge_70.append({'pattern': pat, 'rw_wr': rw_wr_70, 'act_wr': act_wr_70, 'edge': edge_70_val})

avg_edge_cur = np.mean([e['edge'] for e in edge_current])
avg_edge_70 = np.mean([e['edge'] for e in edge_70])
min_edge_cur = min(e['edge'] for e in edge_current)
min_edge_70 = min(e['edge'] for e in edge_70)
strong_cur = sum(1 for e in edge_current if e['edge'] >= 10)
strong_70 = sum(1 for e in edge_70 if e['edge'] >= 10)
thin_70 = sum(1 for e in edge_70 if e['edge'] < 5)

print(f"\n  {'Metric':<30} {'Current TP':>12} {'70% TP':>12}")
print("  " + "-" * 60)
print(f"  {'Avg genuine edge':<30} {avg_edge_cur:>+11.1f}pp {avg_edge_70:>+11.1f}pp")
print(f"  {'Min edge':<30} {min_edge_cur:>+11.1f}pp {min_edge_70:>+11.1f}pp")
print(f"  {'Edge >= 10pp (strong)':<30} {strong_cur:>12d} {strong_70:>12d}")
print(f"  {'Edge < 5pp (thin)':<30} {0:>12d} {thin_70:>12d}")
print(f"  {'Avg RW WR (distance only)':<30} {np.mean([e['rw_wr'] for e in edge_current]):>11.1f}% "
      f"{np.mean([e['rw_wr'] for e in edge_70]):>11.1f}%")

# Show patterns where edge drops most
edge_changes = []
for ec, e7 in zip(edge_current, edge_70):
    edge_changes.append({
        'pattern': ec['pattern'],
        'edge_cur': ec['edge'],
        'edge_70': e7['edge'],
        'change': e7['edge'] - ec['edge'],
    })
edge_changes.sort(key=lambda x: x['change'])

print(f"\n  Top 10 largest edge DROPS at 70% TP:")
print(f"    {'Pattern':<16} {'Edge_cur':>9} {'Edge_70':>9} {'Change':>9}")
for ec in edge_changes[:10]:
    print(f"    {ec['pattern']:<16} {ec['edge_cur']:>+8.1f}pp {ec['edge_70']:>+8.1f}pp {ec['change']:>+8.1f}pp")

phase5 = {
    'avg_edge_current': avg_edge_cur,
    'avg_edge_70': avg_edge_70,
    'min_edge_current': min_edge_cur,
    'min_edge_70': min_edge_70,
    'strong_edge_current': strong_cur,
    'strong_edge_70': strong_70,
    'thin_edge_70': thin_70,
}


# ============================================================
# Phase 6: Consecutive Loss Analysis
# ============================================================
print("\n" + "=" * 70)
print("Phase 6: Consecutive Loss Analysis — Current vs 70% TP")
print("=" * 70)

def analyze_consecutive_losses(trades):
    """Analyze consecutive loss sequences."""
    pnl_list = [t[2] for t in trades]
    streaks = []
    current_streak = 0
    current_loss_sum = 0.0
    for p in pnl_list:
        if p < 0:
            current_streak += 1
            current_loss_sum += p
        else:
            if current_streak > 0:
                streaks.append((current_streak, current_loss_sum))
            current_streak = 0
            current_loss_sum = 0.0
    if current_streak > 0:
        streaks.append((current_streak, current_loss_sum))

    max_streak = max((s[0] for s in streaks), default=0)
    max_loss_streak = min((s[1] for s in streaks), default=0)

    # Distribution
    streak_counts = defaultdict(int)
    for s in streaks:
        streak_counts[s[0]] += 1

    return {
        'max_streak': max_streak,
        'max_loss_sum': max_loss_streak,
        'streak_dist': dict(streak_counts),
        'total_loss_events': len(streaks),
    }

consec_cur = analyze_consecutive_losses(trades_current)
consec_70 = analyze_consecutive_losses(trades_70)

print(f"\n  {'Metric':<35} {'Current':>12} {'TP 70%':>12}")
print("  " + "-" * 65)
print(f"  {'Max consecutive losses':<35} {consec_cur['max_streak']:>12d} {consec_70['max_streak']:>12d}")
print(f"  {'Worst streak PnL':<35} {consec_cur['max_loss_sum']:>+11.1f}% {consec_70['max_loss_sum']:>+11.1f}%")
print(f"  {'Total loss events':<35} {consec_cur['total_loss_events']:>12d} {consec_70['total_loss_events']:>12d}")

for length in [1, 2, 3, 4, 5]:
    c_cur = consec_cur['streak_dist'].get(length, 0)
    c_70 = consec_70['streak_dist'].get(length, 0)
    print(f"  {f'{length}x consecutive losses':<35} {c_cur:>12d} {c_70:>12d}")

phase6 = {
    'current': consec_cur,
    'tp70': consec_70,
}


# ============================================================
# Phase 7: Portfolio MC (Sign Randomization) — is portfolio edge real?
# ============================================================
print("\n" + "=" * 70)
print("Phase 7: Portfolio-Level MC Test (Sign Randomization)")
print("=" * 70)

pnl_arr_cur = np.array([t[2] for t in trades_current])
pnl_arr_70 = np.array([t[2] for t in trades_70])

real_sum_cur = np.sum(pnl_arr_cur)
real_sum_70 = np.sum(pnl_arr_70)

signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_arr_cur)))
mc_sums_cur = np.sum(pnl_arr_cur * signs, axis=1)
mc_p_cur = float(np.mean(mc_sums_cur >= real_sum_cur))

signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_arr_70)))
mc_sums_70 = np.sum(pnl_arr_70 * signs, axis=1)
mc_p_70 = float(np.mean(mc_sums_70 >= real_sum_70))

print(f"\n  {'Portfolio':<20} {'Trades':>8} {'PnL':>10} {'MC p-value':>12} {'Result':>10}")
print("  " + "-" * 65)
print(f"  {'Current':<20} {len(pnl_arr_cur):>8d} {real_sum_cur:>+9.1f}% {mc_p_cur:>12.4f} "
      f"{'PASS' if mc_p_cur < 0.01 else 'FAIL':>10}")
print(f"  {'TP 70%':<20} {len(pnl_arr_70):>8d} {real_sum_70:>+9.1f}% {mc_p_70:>12.4f} "
      f"{'PASS' if mc_p_70 < 0.01 else 'FAIL':>10}")


# ============================================================
# Phase 8: Final Summary & Recommendation
# ============================================================
print("\n" + "=" * 70)
print("Phase 8: Final Summary & Recommendation")
print("=" * 70)

print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Portfolio Comparison: Current vs TP 70%                           │
  │                                                                     │
  │  Metric              Current         TP 70%        Change          │
  │  ─────────────────────────────────────────────────────────────────  │
  │  PnL                {stats_current['pnl']:>+8.1f}%       {stats_70['pnl']:>+8.1f}%      {stats_70['pnl'] - stats_current['pnl']:>+7.1f}%       │
  │  Trades             {stats_current['trades']:>8d}        {stats_70['trades']:>8d}       {stats_70['trades'] - stats_current['trades']:>+7d}          │
  │  Win Rate           {stats_current['wr']:>7.1f}%        {stats_70['wr']:>7.1f}%       {stats_70['wr'] - stats_current['wr']:>+6.1f}pp       │
  │  MDD                {stats_current['mdd']:>7.1f}%        {stats_70['mdd']:>7.1f}%       {stats_70['mdd'] - stats_current['mdd']:>+6.1f}pp       │
  │  PF                 {stats_current['pf']:>8.2f}         {stats_70['pf']:>8.2f}        {stats_70['pf'] - stats_current['pf']:>+6.2f}          │
  │  PnL/MDD            {stats_current['pnl_mdd']:>7.1f}x        {stats_70['pnl_mdd']:>7.1f}x       {stats_70['pnl_mdd'] - stats_current['pnl_mdd']:>+6.1f}x        │
  │  Max Consec Loss    {consec_cur['max_streak']:>8d}        {consec_70['max_streak']:>8d}       {consec_70['max_streak'] - consec_cur['max_streak']:>+7d}          │
  │  Worst Streak PnL   {consec_cur['max_loss_sum']:>+7.1f}%       {consec_70['max_loss_sum']:>+7.1f}%      {consec_70['max_loss_sum'] - consec_cur['max_loss_sum']:>+6.1f}%       │
  │  WF (portfolio)     {wf_cur:>6d}/5         {wf_70:>6d}/5                        │
  │  MC p-value         {mc_p_cur:>8.4f}        {mc_p_70:>8.4f}                       │
  │  Avg Edge           {avg_edge_cur:>+7.1f}pp       {avg_edge_70:>+7.1f}pp      {avg_edge_70 - avg_edge_cur:>+6.1f}pp       │
  │                                                                     │
  │  MC MDD P90         {phase3['current']['p90']:>7.1f}%        {phase3['tp70']['p90']:>7.1f}%       {phase3['tp70']['p90'] - phase3['current']['p90']:>+6.1f}pp       │
  │  MC MDD P99         {phase3['current']['p99']:>7.1f}%        {phase3['tp70']['p99']:>7.1f}%       {phase3['tp70']['p99'] - phase3['current']['p99']:>+6.1f}pp       │
  └─────────────────────────────────────────────────────────────────────┘
""")

# Decision logic
issues = []
advantages = []

if mc_p_70 >= 0.01:
    issues.append("Portfolio MC FAIL at 70% TP")
if wf_70 < 4:
    issues.append(f"WF only {wf_70}/5 at 70% TP")
if mc_fail_count > len(phase1) * 0.20:
    issues.append(f"{mc_fail_count}/{len(phase1)} patterns fail MC at 70% TP")
if avg_edge_70 < 10:
    issues.append(f"Average edge dropped to {avg_edge_70:.1f}pp")
if thin_70 > 5:
    issues.append(f"{thin_70} patterns with edge < 5pp")

if stats_70['pnl'] > stats_current['pnl']:
    advantages.append(f"PnL improved: {stats_current['pnl']:+.1f}% → {stats_70['pnl']:+.1f}%")
if stats_70['mdd'] < stats_current['mdd']:
    advantages.append(f"MDD reduced: {stats_current['mdd']:.1f}% → {stats_70['mdd']:.1f}%")
if stats_70['pnl_mdd'] > stats_current['pnl_mdd']:
    advantages.append(f"PnL/MDD improved: {stats_current['pnl_mdd']:.1f}x → {stats_70['pnl_mdd']:.1f}x")
if stats_70['wr'] > stats_current['wr']:
    advantages.append(f"WR improved: {stats_current['wr']:.1f}% → {stats_70['wr']:.1f}%")
if consec_70['max_streak'] < consec_cur['max_streak']:
    advantages.append(f"Max consecutive losses reduced: {consec_cur['max_streak']} → {consec_70['max_streak']}")
if phase3['tp70']['p99'] < phase3['current']['p99']:
    advantages.append(f"MC P99 MDD reduced: {phase3['current']['p99']:.1f}% → {phase3['tp70']['p99']:.1f}%")

print(f"  Validation Issues: {len(issues)}")
for iss in issues:
    print(f"    - {iss}")

print(f"\n  Advantages: {len(advantages)}")
for adv in advantages:
    print(f"    + {adv}")

if not issues:
    print(f"\n  RECOMMENDATION: TP 70% 전략 적용 가능")
    print(f"    - 모든 검증 통과")
    print(f"    - constants.py의 PATTERN_OPTIMAL_TPSL에서 모든 TP에 * 0.7 적용")
    print(f"    - EXPECTED_WIN_RATE: {stats_current['wr']:.0f} → {stats_70['wr']:.0f} 업데이트")
elif len(issues) <= 2 and len(advantages) >= 4:
    print(f"\n  RECOMMENDATION: 조건부 적용 — 이슈 해결 후 적용")
    for iss in issues:
        print(f"    해결 필요: {iss}")
else:
    print(f"\n  RECOMMENDATION: 현재 TP 유지 — 70% TP 적용 위험")
    for iss in issues:
        print(f"    위험 요인: {iss}")


# ============================================================
# Save results
# ============================================================
results = {
    'meta': {
        'script': 'uniform_tp_validation.py',
        'version': BOT_VERSION,
        'tp_scale': TP_SCALE,
        'timestamp': datetime.now().isoformat(),
        'patterns': len(PORTFOLIO_CURRENT),
    },
    'phase1_per_pattern': phase1,
    'phase2_robustness': phase2,
    'phase3_mc_mdd': phase3,
    'phase4_walk_forward': phase4,
    'phase5_edge': phase5,
    'phase6_consecutive': {
        'current': {k: v for k, v in consec_cur.items()},
        'tp70': {k: v for k, v in consec_70.items()},
    },
    'phase7_portfolio_mc': {
        'current': {'pnl': float(real_sum_cur), 'mc_p': mc_p_cur},
        'tp70': {'pnl': float(real_sum_70), 'mc_p': mc_p_70},
    },
    'summary': {
        'current': {
            'pnl': stats_current['pnl'], 'trades': stats_current['trades'],
            'wr': stats_current['wr'], 'mdd': stats_current['mdd'],
            'pf': stats_current['pf'], 'pnl_mdd': stats_current['pnl_mdd'],
        },
        'tp70': {
            'pnl': stats_70['pnl'], 'trades': stats_70['trades'],
            'wr': stats_70['wr'], 'mdd': stats_70['mdd'],
            'pf': stats_70['pf'], 'pnl_mdd': stats_70['pnl_mdd'],
        },
        'issues': issues,
        'advantages': advantages,
    },
}

out_path = Path(__file__).resolve().parent / '../../results/uniform_tp_validation.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved: {out_path}")
