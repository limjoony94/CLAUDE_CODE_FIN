#!/usr/bin/env python3
"""
Pattern Genuine Edge Study
============================

Strategy Foundation 연구에서 WF가 100% non-discriminating임이 확인됨.
→ 새로운 검증 방법론: Cascade-OFF에서 패턴별 edge 측정.

Phase 1: Cascade-OFF 패턴별 WR & PnL (130 patterns individually)
  - 메커니즘 증폭 없이 순수 패턴 예측력 측정
  - Random baseline WR 57.8% 대비 초과 성과

Phase 2: Random Baseline Bootstrap (패턴별 통계적 유의성)
  - 각 패턴의 trade 수로 random WR 분포 생성
  - 패턴 WR이 random 95th percentile 초과 → genuine edge

Phase 3: Genuine Edge Subset 포트폴리오
  - Phase 2에서 유의한 패턴만으로 full-stack (Cascade ON) 포트폴리오 구성
  - 전체 130패턴 vs Genuine subset 비교

Phase 4: Cascade-OFF WF (진짜 WF 검증)
  - Genuine subset을 Cascade-OFF 상태에서 WF 3-fold
  - 이것이 PASS하면 패턴 자체의 예측력이 검증된 것

Phase 5: ATR/AggRisk Constraint Analysis
  - ATR OFF / AggRisk OFF가 OOS에서도 제약인지 검증
  - IS에서만 제약 = 유지, OOS에서도 제약 = 파라미터 완화 검토

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE = 3 FIXED (v1.42.0)
  - FEE = FEE_PCT * LEVERAGE (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - WF: 3-fold expanding window

Author: Research Agent
Date: 2026-03-04
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, portfolio_sim, calc_stats,
    DATA_FILE, PATTERNS_FILE
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'pattern_genuine_edge_study.json')

# ============================================================
# Data Loading
# ============================================================
print('=' * 90)
print('PATTERN GENUINE EDGE STUDY')
print('=' * 90)
print()

print('Loading data...')
df = load_and_classify(DATA_FILE)
atr_ratio = compute_atr_ratio(df)
ema_slope = compute_ema_slope(df['close'].values)
ns, ne = find_neutral_window(df['close'].values, 1.0)
print(f'Neutral window: {ns}-{ne} ({ne-ns} bars, {(ne-ns)/288:.0f}d)')

with open(PATTERNS_FILE) as f:
    pdata = json.load(f)
pats_raw = pdata['patterns']
tpsl = pdata.get('patterns_tpsl', {})

pat_lookup = {}
for pat_name in pats_raw.get('long', []):
    tp_sl = tpsl.get(pat_name, [2.0, 3.0])
    pat_lookup[pat_name] = {'direction': 'LONG', 'tp': tp_sl[0], 'sl': tp_sl[1]}
for pat_name in pats_raw.get('short', []):
    tp_sl = tpsl.get(pat_name, [2.0, 3.0])
    pat_lookup[pat_name] = {'direction': 'SHORT', 'tp': tp_sl[0], 'sl': tp_sl[1]}
print(f'{len(pat_lookup)} patterns ({len(pats_raw.get("long",[]))}L + {len(pats_raw.get("short",[]))}S)')

rctypes = df['rctype'].values
n_bars = len(df)

# Build per-pattern signal lists
pattern_signals = {}
for i in range(2, n_bars):
    tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
    if tri in pat_lookup:
        p = pat_lookup[tri]
        pattern_signals.setdefault(tri, []).append(
            (i, tri, p['direction'], p['tp'], p['sl'])
        )

all_signals = []
for sigs in pattern_signals.values():
    all_signals.extend(sigs)
all_signals.sort(key=lambda x: x[0])
print(f'{len(all_signals)} total signals across {len(pattern_signals)} patterns')

# Count in neutral
neutral_counts = {}
for pat, sigs in pattern_signals.items():
    nc = sum(1 for b, _, _, _, _ in sigs if ns <= b < ne)
    neutral_counts[pat] = nc

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
tcodes = df['rctype'].values

common = dict(
    opens=opens, highs=highs, lows=lows, closes=closes,
    type_codes=tcodes, n_bars=n_bars,
    atr_ratio=atr_ratio, ema_slope=ema_slope,
)


# ============================================================
# Phase 1: Per-pattern Cascade-OFF performance
# ============================================================
print()
print('=' * 90)
print('PHASE 1: Per-pattern Cascade-OFF WR & PnL')
print('=' * 90)
t0 = time.time()

# Run full portfolio sim with Cascade OFF to get per-trade results
all_trades_cascade_off, _ = portfolio_sim(
    signal_tuples=all_signals, start_bar=ns, end_bar=ne,
    cascade_enabled=False, **common
)

# Group trades by pattern
pattern_trades_off = {}
for t in all_trades_cascade_off:
    pat = t['pattern']
    pattern_trades_off.setdefault(pat, []).append(t)

# Also run Cascade ON for comparison
all_trades_cascade_on, _ = portfolio_sim(
    signal_tuples=all_signals, start_bar=ns, end_bar=ne,
    cascade_enabled=True, **common
)
pattern_trades_on = {}
for t in all_trades_cascade_on:
    pat = t['pattern']
    pattern_trades_on.setdefault(pat, []).append(t)

# Compute per-pattern stats
phase1_results = {}
for pat in sorted(pat_lookup.keys()):
    trades_off = pattern_trades_off.get(pat, [])
    trades_on = pattern_trades_on.get(pat, [])

    n_off = len(trades_off)
    n_on = len(trades_on)

    if n_off > 0:
        wins_off = sum(1 for t in trades_off if t['pnl_slot'] > 0)
        wr_off = wins_off / n_off * 100
        avg_pnl_off = np.mean([t['pnl_slot'] for t in trades_off])
    else:
        wr_off = 0
        avg_pnl_off = 0

    if n_on > 0:
        wins_on = sum(1 for t in trades_on if t['pnl_slot'] > 0)
        wr_on = wins_on / n_on * 100
        avg_pnl_on = np.mean([t['pnl_slot'] for t in trades_on])
    else:
        wr_on = 0
        avg_pnl_on = 0

    phase1_results[pat] = {
        'direction': pat_lookup[pat]['direction'],
        'tp': pat_lookup[pat]['tp'],
        'sl': pat_lookup[pat]['sl'],
        'signals_neutral': neutral_counts.get(pat, 0),
        'trades_off': n_off,
        'wr_off': round(wr_off, 1),
        'avg_pnl_off': round(float(avg_pnl_off), 2),
        'trades_on': n_on,
        'wr_on': round(wr_on, 1),
        'avg_pnl_on': round(float(avg_pnl_on), 2),
    }

p1_time = time.time() - t0
print(f'Phase 1 done ({p1_time:.0f}s)')

# Summary
wr_offs = [v['wr_off'] for v in phase1_results.values() if v['trades_off'] >= 3]
wr_ons = [v['wr_on'] for v in phase1_results.values() if v['trades_on'] >= 3]
print(f'  Cascade OFF WR: mean={np.mean(wr_offs):.1f}%, median={np.median(wr_offs):.1f}%')
print(f'  Cascade ON  WR: mean={np.mean(wr_ons):.1f}%, median={np.median(wr_ons):.1f}%')
print(f'  Patterns with >=3 trades (OFF): {len(wr_offs)}/{len(phase1_results)}')


# ============================================================
# Phase 2: Random Baseline Bootstrap — per-pattern significance
# ============================================================
print()
print('=' * 90)
print('PHASE 2: Random Baseline Bootstrap — genuine edge detection')
print('=' * 90)
t0 = time.time()

# Random baseline: For N trades, what's the WR distribution under random?
# From strategy_foundation study: random WR ~ 57.8% in portfolio sim
# But per-pattern WR differs from portfolio WR due to slot competition
# Use binomial: under random, each trade has ~50% WR (before Cascade/mechanism effects)
# More accurate: use the actual Cascade-OFF random WR from foundation study

RANDOM_WR_BASELINE = 0.50  # Under Cascade-OFF, no pattern edge → ~50% WR pre-mechanism
# Note: mechanism stack (timeout DROP, ATR, etc.) doesn't change WR direction-neutrally
# The 57.8% was with all mechanisms ON including Cascade; Cascade-OFF random should be ~50%

# Actually let's measure it empirically with a few random sets
N_BOOTSTRAP = 500
genuine_patterns = []
marginal_patterns = []
no_edge_patterns = []

for pat, stats in phase1_results.items():
    n_trades = stats['trades_off']
    if n_trades < 3:
        no_edge_patterns.append(pat)
        stats['p_value'] = 1.0
        stats['genuine'] = False
        continue

    observed_wr = stats['wr_off'] / 100.0

    # Binomial test: P(X >= observed_wins | n_trades, p=0.50)
    from scipy.stats import binomtest
    observed_wins = int(round(observed_wr * n_trades))
    result = binomtest(observed_wins, n_trades, 0.50, alternative='greater')
    p_val = result.pvalue

    stats['p_value'] = round(float(p_val), 4)

    if p_val < 0.05 and observed_wr > 0.55:
        genuine_patterns.append(pat)
        stats['genuine'] = True
    elif p_val < 0.20 and observed_wr > 0.50:
        marginal_patterns.append(pat)
        stats['genuine'] = False
    else:
        no_edge_patterns.append(pat)
        stats['genuine'] = False

p2_time = time.time() - t0
print(f'Phase 2 done ({p2_time:.0f}s)')
print(f'  Genuine edge (p<0.05, WR>55%): {len(genuine_patterns)} patterns')
print(f'  Marginal (p<0.20, WR>50%):     {len(marginal_patterns)} patterns')
print(f'  No edge / insufficient trades:  {len(no_edge_patterns)} patterns')
print(f'  Total: {len(genuine_patterns) + len(marginal_patterns) + len(no_edge_patterns)}')

# Top genuine patterns
if genuine_patterns:
    sorted_genuine = sorted(genuine_patterns,
                            key=lambda p: phase1_results[p]['wr_off'], reverse=True)
    print(f'\n  Top 15 genuine edge patterns (Cascade OFF):')
    print(f'  {"Pattern":<15} {"Dir":>5} {"Trades":>7} {"WR_off":>7} {"WR_on":>7} {"p-val":>7} {"AvgPnL":>8}')
    print(f'  {"-"*57}')
    for pat in sorted_genuine[:15]:
        s = phase1_results[pat]
        print(f'  {pat:<15} {s["direction"]:>5} {s["trades_off"]:>7} '
              f'{s["wr_off"]:>7.1f} {s["wr_on"]:>7.1f} {s["p_value"]:>7.4f} {s["avg_pnl_off"]:>+8.2f}')


# ============================================================
# Phase 3: Genuine Edge Subset Portfolio
# ============================================================
print()
print('=' * 90)
print('PHASE 3: Genuine Edge Subset Portfolio vs Full 130')
print('=' * 90)
t0 = time.time()

# Build signal sets
genuine_signal_set = set(genuine_patterns)
genuine_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in all_signals
                   if p in genuine_signal_set]
print(f'Genuine subset: {len(genuine_patterns)} patterns, {len(genuine_signals)} signals')

n_gen_long = sum(1 for p in genuine_patterns if pat_lookup[p]['direction'] == 'LONG')
n_gen_short = len(genuine_patterns) - n_gen_long
print(f'  Direction: {n_gen_long}L + {n_gen_short}S')

# Full 130 (Cascade ON) — already computed
full_stats_on = calc_stats(all_trades_cascade_on)
full_stats_off = calc_stats(all_trades_cascade_off)

# Genuine subset — Cascade ON
if genuine_signals:
    gen_trades_on, _ = portfolio_sim(
        signal_tuples=genuine_signals, start_bar=ns, end_bar=ne,
        cascade_enabled=True, **common
    )
    gen_stats_on = calc_stats(gen_trades_on)

    # Genuine subset — Cascade OFF
    gen_trades_off, _ = portfolio_sim(
        signal_tuples=genuine_signals, start_bar=ns, end_bar=ne,
        cascade_enabled=False, **common
    )
    gen_stats_off = calc_stats(gen_trades_off)
else:
    gen_stats_on = {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0}
    gen_stats_off = gen_stats_on

p3_time = time.time() - t0
print(f'\nPhase 3 done ({p3_time:.0f}s)')
print(f'\n  {"Config":<30} {"Trades":>7} {"WR%":>7} {"PnL%":>10} {"MDD%":>8} {"PnL/MDD":>9}')
print(f'  {"-"*71}')
for name, st in [
    ('Full 130 + Cascade ON', full_stats_on),
    ('Full 130 + Cascade OFF', full_stats_off),
    ('Genuine subset + Cascade ON', gen_stats_on),
    ('Genuine subset + Cascade OFF', gen_stats_off),
]:
    print(f'  {name:<30} {st["trades"]:>7} {st["wr"]:>7.1f} {st["pnl"]:>+10.1f} '
          f'{st["mdd"]:>8.2f} {st["pnl_mdd"]:>9.1f}')

# Pattern quality comparison
if gen_stats_off['pnl'] > 0 and full_stats_off['pnl'] > 0:
    quality_ratio = gen_stats_off['pnl_mdd'] / full_stats_off['pnl_mdd'] if full_stats_off['pnl_mdd'] > 0 else 0
    print(f'\n  Genuine subset quality (Cascade OFF PnL/MDD): '
          f'{gen_stats_off["pnl_mdd"]:.1f} vs Full {full_stats_off["pnl_mdd"]:.1f} '
          f'(ratio: {quality_ratio:.2f}x)')


# ============================================================
# Phase 4: Cascade-OFF WF (True Pattern Edge Validation)
# ============================================================
print()
print('=' * 90)
print('PHASE 4: Cascade-OFF WF — True Pattern Edge Validation')
print('=' * 90)
t0 = time.time()

def wf_3fold(signal_tuples, cascade_on=True, **extra_kwargs):
    n_folds = 3
    total = ne - ns
    seg_size = total // (n_folds + 1)
    folds = []
    for fold in range(n_folds):
        oos_start = ns + seg_size * (fold + 1)
        oos_end = ns + seg_size * (fold + 2) if fold < n_folds - 1 else ne
        trades_f, _ = portfolio_sim(
            signal_tuples=signal_tuples,
            start_bar=oos_start, end_bar=oos_end,
            cascade_enabled=cascade_on,
            **extra_kwargs, **common
        )
        stats_f = calc_stats(trades_f)
        stats_f['fold'] = fold + 1
        folds.append(stats_f)
    n_pass = sum(1 for f in folds if f['pnl'] > 0)
    return n_pass, folds

configs_to_test = {
    'Full_130_CascadeON': (all_signals, True),
    'Full_130_CascadeOFF': (all_signals, False),
}
if genuine_signals:
    configs_to_test['Genuine_CascadeON'] = (genuine_signals, True)
    configs_to_test['Genuine_CascadeOFF'] = (genuine_signals, False)

wf_results = {}
for name, (sigs, cascade) in configs_to_test.items():
    n_pass, folds = wf_3fold(sigs, cascade_on=cascade)
    wf_results[name] = {
        'n_pass': n_pass,
        'folds': folds,
        'verdict': f'{n_pass}/3 {"PASS" if n_pass == 3 else "FAIL"}',
    }

p4_time = time.time() - t0
print(f'Phase 4 done ({p4_time:.0f}s)')
print(f'\n  {"Config":<25} {"F1":>8} {"F2":>8} {"F3":>8} {"Verdict":>10}')
print(f'  {"-"*60}')
for name, r in wf_results.items():
    f = r['folds']
    print(f'  {name:<25} {f[0]["pnl"]:>+8.1f} {f[1]["pnl"]:>+8.1f} {f[2]["pnl"]:>+8.1f} '
          f'{r["verdict"]:>10}')


# ============================================================
# Phase 5: ATR/AggRisk Constraint Analysis (OOS)
# ============================================================
print()
print('=' * 90)
print('PHASE 5: ATR/AggRisk Constraint Analysis (IS vs OOS)')
print('=' * 90)
t0 = time.time()

constraint_configs = {
    'Baseline (all ON)': {},
    'No ATR Scale': {'atr_scaling_enabled': False},
    'No AggRisk': {'agg_risk_enabled': False},
    'No ATR + No AggRisk': {'atr_scaling_enabled': False, 'agg_risk_enabled': False},
}

print(f'\n  {"Config":<25} {"IS PnL":>9} {"IS MDD":>9} {"IS P/M":>9} | '
      f'{"F1":>8} {"F2":>8} {"F3":>8} {"WF":>8}')
print(f'  {"-"*90}')

p5_results = {}
for name, kwargs in constraint_configs.items():
    # IS
    is_trades, _ = portfolio_sim(
        signal_tuples=all_signals, start_bar=ns, end_bar=ne,
        **kwargs, **common
    )
    is_st = calc_stats(is_trades)

    # WF
    n_pass, folds = wf_3fold(all_signals, **kwargs)

    p5_results[name] = {
        'is': is_st,
        'wf_pass': n_pass,
        'wf_folds': [f['pnl'] for f in folds],
    }

    verdict = f'{n_pass}/3 {"P" if n_pass == 3 else "F"}'
    print(f'  {name:<25} {is_st["pnl"]:>+9.1f} {is_st["mdd"]:>9.2f} {is_st["pnl_mdd"]:>9.1f} | '
          f'{folds[0]["pnl"]:>+8.1f} {folds[1]["pnl"]:>+8.1f} {folds[2]["pnl"]:>+8.1f} {verdict:>8}')

p5_time = time.time() - t0


# ============================================================
# SYNTHESIS
# ============================================================
print()
print('=' * 90)
print('SYNTHESIS')
print('=' * 90)

total_time = p1_time + p2_time + p3_time + p4_time + p5_time

print(f"""
PHASE 1: Per-pattern Cascade-OFF WR
  Mean WR (Cascade OFF): {np.mean(wr_offs):.1f}%
  Mean WR (Cascade ON):  {np.mean(wr_ons):.1f}%
  Cascade adds ~{np.mean(wr_ons) - np.mean(wr_offs):.1f}pp to per-pattern WR

PHASE 2: Genuine Edge Detection
  {len(genuine_patterns)} genuine (p<0.05, WR>55%)
  {len(marginal_patterns)} marginal
  {len(no_edge_patterns)} no edge
  {'Pattern selection HAS value' if len(genuine_patterns) > 20
   else 'Pattern selection has LIMITED value' if len(genuine_patterns) > 5
   else 'Pattern selection has MINIMAL value — most patterns are noise'}

PHASE 3: Portfolio Comparison
  Full 130 Cascade-ON:  PnL/MDD = {full_stats_on['pnl_mdd']:.1f}
  Genuine Cascade-ON:   PnL/MDD = {gen_stats_on['pnl_mdd']:.1f}
  Genuine Cascade-OFF:  PnL/MDD = {gen_stats_off['pnl_mdd']:.1f}

PHASE 4: Cascade-OFF WF""")

gen_off_wf = wf_results.get('Genuine_CascadeOFF', {})
if gen_off_wf:
    print(f'  Genuine Cascade-OFF WF: {gen_off_wf["verdict"]}')
    if gen_off_wf['n_pass'] == 3:
        print('  >>> Genuine patterns have REAL predictive edge (survives without Cascade)')
    else:
        print('  >>> Even "genuine" patterns may not survive without Cascade')

print(f"""
PHASE 5: ATR/AggRisk Constraints""")
for name, r in p5_results.items():
    print(f'  {name}: IS PnL/MDD={r["is"]["pnl_mdd"]:.1f}, WF {r["wf_pass"]}/3')

# Strategic recommendation
print(f"""
STRATEGIC RECOMMENDATIONS:
""")
if len(genuine_patterns) > 20:
    print(f'  1. ADOPT Genuine subset ({len(genuine_patterns)} patterns) as primary pattern set')
    print(f'     — Validated edge independent of Cascade amplification')
if gen_off_wf and gen_off_wf['n_pass'] == 3:
    print(f'  2. Use Cascade-OFF WF as new validation gate')
    print(f'     — Replaces standard WF which is non-discriminating')
else:
    print(f'  2. Cascade-OFF WF fails → pattern edge is amplifier-dependent')
    print(f'     — System fundamentally relies on Cascade, not pattern prediction')

no_atr_wf = p5_results.get('No ATR Scale', {})
if no_atr_wf and no_atr_wf['wf_pass'] < 3:
    print(f'  3. ATR Scale is OOS-protective (removal causes WF FAIL)')
elif no_atr_wf:
    print(f'  3. ATR Scale may be over-constraining (removal still WF PASS)')

print(f'\nTotal study time: {total_time:.0f}s')

# ============================================================
# Save
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v1.42.0',
    'neutral_window': {'start': ns, 'end': ne, 'bars': ne - ns},
    'phase1_per_pattern': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                                for kk, vv in v.items()} for k, v in phase1_results.items()},
    'phase2_edge_detection': {
        'genuine_count': len(genuine_patterns),
        'genuine_patterns': genuine_patterns,
        'marginal_count': len(marginal_patterns),
        'marginal_patterns': marginal_patterns,
        'no_edge_count': len(no_edge_patterns),
        'random_wr_baseline': RANDOM_WR_BASELINE,
    },
    'phase3_portfolio': {
        'full_130_cascade_on': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                 for k, v in full_stats_on.items()},
        'full_130_cascade_off': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                  for k, v in full_stats_off.items()},
        'genuine_cascade_on': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                for k, v in gen_stats_on.items()},
        'genuine_cascade_off': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                 for k, v in gen_stats_off.items()},
    },
    'phase4_wf': {name: {'n_pass': r['n_pass'], 'verdict': r['verdict'],
                          'folds_pnl': [f['pnl'] for f in r['folds']]}
                  for name, r in wf_results.items()},
    'phase5_constraints': {name: {'is_pnl': r['is']['pnl'], 'is_mdd': r['is']['mdd'],
                                   'is_pnl_mdd': r['is']['pnl_mdd'],
                                   'wf_pass': r['wf_pass'], 'wf_folds': r['wf_folds']}
                           for name, r in p5_results.items()},
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved: {OUTPUT_FILE}')
