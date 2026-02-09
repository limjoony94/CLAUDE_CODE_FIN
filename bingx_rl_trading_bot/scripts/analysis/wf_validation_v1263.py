"""
v1.26.3 Walk-Forward Validation
================================
52개 전체 패턴에 대한 종합 WF 검증:
  Phase 1: 개별 패턴 WF 검증 (52개) — WF, MC, 기본 통계, 3-period 안정성
  Phase 2: 프로덕션 포트폴리오 WF 검증 — 1-pos-at-a-time, 5-fold WF
  Phase 3: 요약 리포트 — FAIL 목록, 포트폴리오 결과, v1.26.3 재최적화 하이라이트
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
    PATTERN_OPTIMAL_TPSL, PATTERN_STATS,
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
)

# ============================================================
# Parameters (same as tp_ge_sl_research_v3.py)
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10          # 0.05% * 2 (round trip)
LEVERAGE = 3
MC_SIMS = 10000

# v1.26.3 re-optimized patterns
REOPT_PATTERNS = {"D-MU-U", "GS-ST-ST", "MU-BD-ST", "D-BD-ST", "ST-DN-BU"}

print("=" * 70)
print("v1.26.3 Walk-Forward Validation")
print("  52 patterns (32L+20S), v1.26.3 TP/SL settings")
print("=" * 70)

# ============================================================
# Load and classify data
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

# 3-period masks (270일을 3등분)
period_masks = {
    'P1_May_Jul': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'P2_Aug_Oct': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'P3_Nov_Jan': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

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

print(f"Classification complete. Total patterns: {len(pattern_indices)}")


# ============================================================
# Backtest engine (identical to tp_ge_sl_research_v3.py)
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Backtest: returns list of per-trade PnL (leveraged, fee-adjusted)."""
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


def period_test(indices, direction, tp_pct, sl_pct):
    """Period stability: count periods with positive PnL."""
    profitable = 0
    for mask in period_masks.values():
        pidx = indices[mask[indices]]
        if len(pidx) < 3:
            continue
        pnls = bt_fixed(pidx, direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


def collect_trades_1pos(pattern_map):
    """Collect trades with 1-position-at-a-time constraint."""
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
    """Evaluate trades using simple returns (sum of PnL %)."""
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
# Build v1.26.3 pattern map from constants
# ============================================================
v1263_map = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    v1263_map[pat] = ('LONG', tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    v1263_map[pat] = ('SHORT', tp, sl)

assert len(v1263_map) == 52, f"Expected 52 patterns, got {len(v1263_map)}"
print(f"v1.26.3 portfolio: {len(v1263_map)} patterns ({len(VALIDATED_LONG_PATTERNS)}L+{len(VALIDATED_SHORT_PATTERNS)}S)")


# ============================================================
# Phase 1: 개별 패턴 WF 검증 (52개)
# ============================================================
print(f"\n{'='*70}")
print("Phase 1: Individual Pattern WF Validation (52 patterns)")
print(f"{'='*70}")

t0 = time.time()

pattern_results = {}

for pat, (direction, tp, sl) in sorted(v1263_map.items()):
    indices = pattern_indices.get(pat, np.array([]))
    rr = tp / sl if sl > 0 else 999

    # Full backtest
    pnls = bt_fixed(indices, direction, tp, sl)
    n_trades = len(pnls)

    if n_trades == 0:
        pattern_results[pat] = {
            'direction': direction, 'tp': tp, 'sl': sl, 'rr': round(rr, 2),
            'trades': 0, 'wr': 0, 'total_pnl': 0, 'expectancy': 0,
            'mc': 1.0, 'wf': 0, 'periods': 0,
            'pass': False, 'fail_reason': 'no_trades',
            'reopt': pat in REOPT_PATTERNS,
        }
        continue

    wr = sum(1 for x in pnls if x > 0) / n_trades * 100
    total_pnl = sum(pnls)
    avg_win = np.mean([x for x in pnls if x > 0]) if any(x > 0 for x in pnls) else 0
    avg_loss = np.mean([abs(x) for x in pnls if x <= 0]) if any(x <= 0 for x in pnls) else 0
    exp = (wr / 100) * avg_win - (1 - wr / 100) * avg_loss

    # WF test
    wf = walk_forward_test(indices, direction, tp, sl)

    # MC test
    mc = mc_test(pnls, MC_SIMS)

    # 3-period stability
    periods = period_test(indices, direction, tp, sl)

    # PASS/FAIL judgment
    fail_reasons = []
    if wf < 4:
        fail_reasons.append(f'wf={wf}<4')
    if mc >= 0.01:
        fail_reasons.append(f'mc={mc:.4f}>=0.01')
    if total_pnl <= 0:
        fail_reasons.append(f'pnl={total_pnl:.1f}<=0')

    passed = len(fail_reasons) == 0

    pattern_results[pat] = {
        'direction': direction, 'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'trades': n_trades, 'wr': round(wr, 1), 'total_pnl': round(total_pnl, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'expectancy': round(exp, 3),
        'mc': round(mc, 4), 'wf': wf, 'periods': periods,
        'pass': passed,
        'fail_reason': ', '.join(fail_reasons) if fail_reasons else '',
        'reopt': pat in REOPT_PATTERNS,
    }

elapsed = time.time() - t0
print(f"  Complete in {elapsed:.1f}s\n")

# Print table
pass_count = sum(1 for r in pattern_results.values() if r['pass'])
fail_count = sum(1 for r in pattern_results.values() if not r['pass'])

header = f"{'Pattern':<12} {'Dir':>5} {'TP/SL':>7} {'R:R':>5} {'Tr':>4} {'WR':>6} {'PnL':>8} {'Exp':>7} {'MC':>7} {'WF':>4} {'Per':>4} {'Result':>6}"
print(header)
print("-" * len(header))

for pat in sorted(pattern_results.keys()):
    r = pattern_results[pat]
    tpsl = f"{r['tp']}/{r['sl']}"
    mark = " *" if r['reopt'] else "  "
    status = "PASS" if r['pass'] else "FAIL"
    print(f"{pat:<12} {r['direction']:>5} {tpsl:>7} {r['rr']:>5.2f} {r['trades']:>4} {r['wr']:>5.1f}% {r['total_pnl']:>+7.1f}% {r['expectancy']:>+6.3f}% {r['mc']:>7.4f} {r['wf']:>2}/5 {r['periods']:>2}/3 {status:>4}{mark}")

print(f"\nSummary: {pass_count} PASS, {fail_count} FAIL (out of 52)")

if fail_count > 0:
    print("\nFAIL details:")
    for pat in sorted(pattern_results.keys()):
        r = pattern_results[pat]
        if not r['pass']:
            print(f"  {pat}: {r['fail_reason']}")


# ============================================================
# Phase 2: 프로덕션 포트폴리오 WF 검증
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: Production Portfolio WF Validation (1-pos-at-a-time)")
print(f"{'='*70}\n")

# Full portfolio evaluation
full_trades = collect_trades_1pos(v1263_map)
full_stats = eval_trades_simple(full_trades)

print("Full Portfolio (270 days):")
print(f"  Trades: {full_stats['trades']}")
print(f"  WR: {full_stats['wr']:.1f}%")
print(f"  PnL: {full_stats['pnl']:+.1f}%")
print(f"  MDD: {full_stats['mdd']:.1f}%")
print(f"  PnL/MDD: {full_stats['pnl_mdd']:.2f}x")
print(f"  PF: {full_stats['pf']:.2f}")
print(f"  Exp/trade: {full_stats['expectancy']:+.3f}%")
print(f"  Avg Win: {full_stats['avg_win']:.2f}%")
print(f"  Avg Loss: {full_stats['avg_loss']:.2f}%")

# 5-fold portfolio WF
print(f"\n5-Fold Portfolio Walk-Forward:")
fold_size = n_bars // 5
wf_portfolio_count = 0
wf_fold_results = []

for f in range(5):
    s = f * fold_size
    e = s + fold_size if f < 4 else n_bars

    # Collect 1-pos-at-a-time trades within this fold
    ft = []
    last_exit = -1
    for i in range(max(s, 2), e):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in v1263_map:
            continue
        direction, tp, sl = v1263_map[pat]
        if i + 1 >= n_bars:
            continue
        entry_bar = i + 1
        if entry_bar <= last_exit:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp / 100)
            slp = entry * (1 - sl / 100)
        else:
            tpp = entry * (1 - tp / 100)
            slp = entry * (1 + sl / 100)
        for j in range(entry_bar + 1, min(entry_bar + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                ft.append((entry_bar, j, pnl))
                last_exit = j
                break
            elif ht:
                ft.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                last_exit = j
                break
            elif hs:
                ft.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                last_exit = j
                break

    fr = eval_trades_simple(ft)
    wf_fold_results.append(fr)
    if fr['pnl'] > 0:
        wf_portfolio_count += 1

    status = "OK" if fr['pnl'] > 0 else "FAIL"
    print(f"  Fold {f+1}: {fr['trades']:>4} trades, WR {fr['wr']:>5.1f}%, PnL {fr['pnl']:>+8.1f}%, MDD {fr['mdd']:>5.1f}%, PF {fr['pf']:>5.2f}  [{status}]")

portfolio_wf_pass = wf_portfolio_count >= 4
print(f"\nPortfolio WF: {wf_portfolio_count}/5  {'PASS' if portfolio_wf_pass else 'FAIL'}")

# WR resilience
pnl_list = [t[2] for t in full_trades]
actual_wr = full_stats['wr']
avg_win = full_stats['avg_win']
avg_loss = full_stats['avg_loss']
be_wr = avg_loss / (avg_win + avg_loss) * 100 if (avg_win + avg_loss) > 0 else 0
max_tolerable_drop = actual_wr - be_wr

print(f"\nWR Resilience:")
print(f"  Actual WR: {actual_wr:.1f}%")
print(f"  Break-even WR: {be_wr:.1f}%")
print(f"  Max tolerable drop: {max_tolerable_drop:+.1f}pp")


# ============================================================
# Phase 3: 요약 리포트
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Summary Report")
print(f"{'='*70}\n")

# FAIL patterns
fail_patterns = {p: r for p, r in pattern_results.items() if not r['pass']}
if fail_patterns:
    print(f"FAILED Patterns ({len(fail_patterns)}):")
    for p, r in sorted(fail_patterns.items()):
        reopt_mark = " [RE-OPT]" if r['reopt'] else ""
        print(f"  {p}: {r['fail_reason']}{reopt_mark}")
else:
    print("All 52 patterns PASSED!")

# Re-optimized patterns highlight
print(f"\nv1.26.3 Re-optimized Patterns (5):")
print(f"{'Pattern':<12} {'Dir':>5} {'TP/SL':>7} {'Tr':>4} {'WR':>6} {'PnL':>8} {'Exp':>7} {'MC':>7} {'WF':>4} {'Result':>6}")
print("-" * 78)
for pat in sorted(REOPT_PATTERNS):
    r = pattern_results[pat]
    tpsl = f"{r['tp']}/{r['sl']}"
    status = "PASS" if r['pass'] else "FAIL"
    print(f"{pat:<12} {r['direction']:>5} {tpsl:>7} {r['trades']:>4} {r['wr']:>5.1f}% {r['total_pnl']:>+7.1f}% {r['expectancy']:>+6.3f}% {r['mc']:>7.4f} {r['wf']:>2}/5 {status:>4}")

# Overall verdict
print(f"\n--- Overall Verdict ---")
print(f"  Individual: {pass_count}/52 PASS ({pass_count/52*100:.0f}%)")
print(f"  Portfolio WF: {wf_portfolio_count}/5 ({'PASS' if portfolio_wf_pass else 'FAIL'})")
print(f"  Portfolio PnL: {full_stats['pnl']:+.1f}%")
print(f"  Portfolio MDD: {full_stats['mdd']:.1f}%")
print(f"  Portfolio PnL/MDD: {full_stats['pnl_mdd']:.2f}x")
print(f"  WR Resilience: {max_tolerable_drop:+.1f}pp")


# ============================================================
# Save results
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v1.26.3',
    'validation': 'Walk-Forward Validation',
    'portfolio_size': len(v1263_map),
    'phase1_individual': {},
    'phase2_portfolio': {
        'full_stats': {k: round(v, 3) if isinstance(v, float) else v for k, v in full_stats.items()},
        'wf_score': wf_portfolio_count,
        'wf_pass': portfolio_wf_pass,
        'wf_folds': [
            {k: round(v, 3) if isinstance(v, float) else v for k, v in fr.items()}
            for fr in wf_fold_results
        ],
        'wr_resilience': {
            'actual_wr': round(actual_wr, 1),
            'be_wr': round(be_wr, 1),
            'max_tolerable_drop_pp': round(max_tolerable_drop, 1),
        },
    },
    'phase3_summary': {
        'individual_pass': pass_count,
        'individual_fail': fail_count,
        'failed_patterns': list(fail_patterns.keys()),
        'reopt_results': {},
    },
}

for pat, r in pattern_results.items():
    output['phase1_individual'][pat] = r

for pat in REOPT_PATTERNS:
    output['phase3_summary']['reopt_results'][pat] = pattern_results[pat]

out_path = Path(__file__).resolve().parent / '../../results/wf_validation_v1263.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done.")
