#!/usr/bin/env python3
"""
Strategy Foundation Critical Study
=====================================

v1.43.0 ROLLBACK에서 도출된 3가지 핵심 질문에 대한 체계적 검증.

Q1: WF Discrimination — WF 3/3 PASS가 진짜 엣지를 검증하는가?
  - 동일 빈도/방향 비율의 랜덤 신호도 WF 3/3 PASS하면 WF는 무의미
  - 100개 랜덤 신호 세트 × WF 3-fold → PASS 비율 측정

Q2: Pattern Independence — 패턴 없이 메커니즘 스택만으로 수익 가능한가?
  - 랜덤 진입 + 실제 TP/SL 분포 + full mechanism stack → PnL
  - 패턴 예측력 vs 메커니즘 관리력 분리 정량화

Q3: Cascade Dependency — CascadeSL이 없으면 시스템이 성립하는가?
  - Cascade ON vs OFF, 랜덤 vs 실제 패턴 2×2 매트릭스
  - Cascade가 "패턴 엣지 증폭기"인지 "독립 수익원"인지 판별

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

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'strategy_foundation_critical_study.json')


# ============================================================
# Data Loading
# ============================================================
print('=' * 90)
print('STRATEGY FOUNDATION CRITICAL STUDY')
print('=' * 90)
print()

print('Loading data...')
df = load_and_classify(DATA_FILE)
atr_ratio = compute_atr_ratio(df)
ema_slope = compute_ema_slope(df['close'].values)
ns, ne = find_neutral_window(df['close'].values, 1.0)
print(f'Neutral window: {ns}-{ne} ({ne-ns} bars, {(ne-ns)/288:.0f}d)')

# Load patterns
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

# Build real signal tuples
rctypes = df['rctype'].values
n_bars = len(df)
real_signals = []
for i in range(2, n_bars):
    tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
    if tri in pat_lookup:
        p = pat_lookup[tri]
        real_signals.append((i, tri, p['direction'], p['tp'], p['sl']))
print(f'{len(real_signals)} real signals')

# Count signals in neutral window
real_in_neutral = [(b, p, d, tp, sl) for b, p, d, tp, sl in real_signals if ns <= b < ne]
n_real_neutral = len(real_in_neutral)
n_long = sum(1 for _, _, d, _, _ in real_in_neutral if d == 'LONG')
n_short = n_real_neutral - n_long
long_ratio = n_long / n_real_neutral if n_real_neutral > 0 else 0.5
print(f'In neutral: {n_real_neutral} signals ({n_long}L + {n_short}S, L ratio={long_ratio:.3f})')

# Collect TP/SL distributions from real patterns
real_tp_vals = [tp for _, _, _, tp, _ in real_in_neutral]
real_sl_vals = [sl for _, _, _, _, sl in real_in_neutral]

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
# Helper: Generate random signal set
# ============================================================
def generate_random_signals(rng, n_signals, bar_range_start, bar_range_end,
                            long_ratio_target, tp_pool, sl_pool):
    """Generate random signals with same frequency/direction/TP-SL distribution as real."""
    bars = rng.integers(bar_range_start + 2, bar_range_end, size=n_signals)
    bars = np.sort(np.unique(bars))  # Deduplicate and sort
    signals = []
    for b in bars:
        is_long = rng.random() < long_ratio_target
        d = 'LONG' if is_long else 'SHORT'
        tp = float(rng.choice(tp_pool))
        sl = float(rng.choice(sl_pool))
        signals.append((int(b), f'RND_{b}', d, tp, sl))
    return signals


# ============================================================
# Helper: WF 3-fold expanding window
# ============================================================
def wf_3fold(signal_tuples, cascade_on=True, **extra_kwargs):
    """Run 3-fold expanding window WF. Returns (n_pass, folds_stats)."""
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


# ============================================================
# Q1: WF Discrimination Test
# ============================================================
print()
print('=' * 90)
print('Q1: WF DISCRIMINATION — Do random signals also pass WF 3/3?')
print('=' * 90)
t0 = time.time()

N_RANDOM_SETS = 30
wf_results_random = []
wf_pass_count = 0
random_is_stats = []

for trial in range(N_RANDOM_SETS):
    rng = np.random.default_rng(trial)
    rsig = generate_random_signals(
        rng, n_signals=n_real_neutral,
        bar_range_start=ns, bar_range_end=ne,
        long_ratio_target=long_ratio,
        tp_pool=real_tp_vals, sl_pool=real_sl_vals
    )

    # IS full
    is_trades, _ = portfolio_sim(
        signal_tuples=rsig, start_bar=ns, end_bar=ne, **common
    )
    is_st = calc_stats(is_trades)
    random_is_stats.append(is_st)

    # WF
    n_pass, folds = wf_3fold(rsig)
    passed = n_pass == 3
    wf_results_random.append({
        'trial': trial, 'n_pass': n_pass, 'passed': passed,
        'is_pnl': is_st['pnl'], 'is_wr': is_st['wr'],
        'oos_pnls': [f['pnl'] for f in folds],
    })
    if passed:
        wf_pass_count += 1

    if (trial + 1) % 10 == 0:
        print(f'  Trial {trial+1}/{N_RANDOM_SETS} done... ({wf_pass_count} pass so far)')

wf_pass_rate = wf_pass_count / N_RANDOM_SETS * 100
q1_time = time.time() - t0
print(f'\nQ1 Results ({q1_time:.0f}s):')
print(f'  Random signal WF 3/3 PASS rate: {wf_pass_count}/{N_RANDOM_SETS} = {wf_pass_rate:.1f}%')
print(f'  Random IS PnL: mean={np.mean([s["pnl"] for s in random_is_stats]):.1f}%, '
      f'median={np.median([s["pnl"] for s in random_is_stats]):.1f}%')
print(f'  Random IS WR: mean={np.mean([s["wr"] for s in random_is_stats]):.1f}%')

# Compare with real signals
real_is_trades, _ = portfolio_sim(signal_tuples=real_signals, start_bar=ns, end_bar=ne, **common)
real_is_stats = calc_stats(real_is_trades)
real_n_pass, real_folds = wf_3fold(real_signals)
print(f'\n  Real signals IS: PnL={real_is_stats["pnl"]:+.1f}%, WR={real_is_stats["wr"]:.1f}%, '
      f'MDD={real_is_stats["mdd"]:.2f}%')
print(f'  Real signals WF: {real_n_pass}/3 PASS')

q1_verdict = 'DISCRIMINATING' if wf_pass_rate < 20 else ('WEAK' if wf_pass_rate < 50 else 'NON-DISCRIMINATING')
print(f'\n  >>> Q1 VERDICT: WF is {q1_verdict} (random pass rate {wf_pass_rate:.1f}%)')
print(f'      (Threshold: <20% = discriminating, 20-50% = weak, >50% = non-discriminating)')


# ============================================================
# Q2: Pattern Independence — Random entries + mechanism stack
# ============================================================
print()
print('=' * 90)
print('Q2: PATTERN INDEPENDENCE — Can mechanism stack profit with random entries?')
print('=' * 90)
t0 = time.time()

N_RANDOM_Q2 = 20
random_pnls = []
random_wrs = []
random_mdds = []
random_pnl_mdds = []

for trial in range(N_RANDOM_Q2):
    rng = np.random.default_rng(1000 + trial)
    rsig = generate_random_signals(
        rng, n_signals=n_real_neutral,
        bar_range_start=ns, bar_range_end=ne,
        long_ratio_target=long_ratio,
        tp_pool=real_tp_vals, sl_pool=real_sl_vals
    )
    trades, _ = portfolio_sim(
        signal_tuples=rsig, start_bar=ns, end_bar=ne, **common
    )
    st = calc_stats(trades)
    random_pnls.append(st['pnl'])
    random_wrs.append(st['wr'])
    random_mdds.append(st['mdd'])
    if st['mdd'] > 0:
        random_pnl_mdds.append(st['pnl'] / st['mdd'])
    else:
        random_pnl_mdds.append(0)

q2_time = time.time() - t0
random_pnl_mean = np.mean(random_pnls)
random_pnl_median = np.median(random_pnls)
random_pnl_positive = sum(1 for p in random_pnls if p > 0) / N_RANDOM_Q2 * 100
random_wr_mean = np.mean(random_wrs)

print(f'\nQ2 Results ({q2_time:.0f}s):')
print(f'  Random entries (N={N_RANDOM_Q2}):')
print(f'    PnL: mean={random_pnl_mean:+.1f}%, median={random_pnl_median:+.1f}%, '
      f'positive={random_pnl_positive:.0f}%')
print(f'    WR: mean={random_wr_mean:.1f}%')
print(f'    MDD: mean={np.mean(random_mdds):.2f}%')
print(f'    PnL/MDD: mean={np.mean(random_pnl_mdds):.1f}')
print(f'\n  Real patterns:')
print(f'    PnL: {real_is_stats["pnl"]:+.1f}%')
print(f'    WR: {real_is_stats["wr"]:.1f}%')
print(f'    MDD: {real_is_stats["mdd"]:.2f}%')
print(f'    PnL/MDD: {real_is_stats["pnl_mdd"]:.1f}')

pattern_lift_pnl = real_is_stats['pnl'] - random_pnl_mean
pattern_contribution = (pattern_lift_pnl / real_is_stats['pnl'] * 100) if real_is_stats['pnl'] != 0 else 0

print(f'\n  Pattern lift: {pattern_lift_pnl:+.1f}% PnL ({pattern_contribution:.1f}% of total)')
print(f'  Mechanism-only PnL: {random_pnl_mean:+.1f}% ({100-pattern_contribution:.1f}% of total)')

q2_verdict = 'PATTERN-DEPENDENT' if pattern_contribution > 60 else (
    'HYBRID' if pattern_contribution > 30 else 'MECHANISM-DEPENDENT')
print(f'\n  >>> Q2 VERDICT: System is {q2_verdict} (pattern contribution {pattern_contribution:.1f}%)')


# ============================================================
# Q3: Cascade Dependency — 2×2 Matrix (Real/Random × Cascade ON/OFF)
# ============================================================
print()
print('=' * 90)
print('Q3: CASCADE DEPENDENCY — 2×2 Matrix (Signal Type × Cascade)')
print('=' * 90)
t0 = time.time()

# 3a. Real patterns + Cascade ON (already computed)
real_cascade_on = real_is_stats

# 3b. Real patterns + Cascade OFF
real_off_trades, _ = portfolio_sim(
    signal_tuples=real_signals, start_bar=ns, end_bar=ne,
    cascade_enabled=False, **common
)
real_cascade_off = calc_stats(real_off_trades)

# 3c. Random entries + Cascade ON (use Q2 results mean — recompute a few for comparison)
N_Q3 = 15
random_on_pnls = []
random_off_pnls = []
random_on_mdds = []
random_off_mdds = []

for trial in range(N_Q3):
    rng = np.random.default_rng(2000 + trial)
    rsig = generate_random_signals(
        rng, n_signals=n_real_neutral,
        bar_range_start=ns, bar_range_end=ne,
        long_ratio_target=long_ratio,
        tp_pool=real_tp_vals, sl_pool=real_sl_vals
    )

    # Cascade ON
    trades_on, _ = portfolio_sim(
        signal_tuples=rsig, start_bar=ns, end_bar=ne,
        cascade_enabled=True, **common
    )
    st_on = calc_stats(trades_on)
    random_on_pnls.append(st_on['pnl'])
    random_on_mdds.append(st_on['mdd'])

    # Cascade OFF
    trades_off, _ = portfolio_sim(
        signal_tuples=rsig, start_bar=ns, end_bar=ne,
        cascade_enabled=False, **common
    )
    st_off = calc_stats(trades_off)
    random_off_pnls.append(st_off['pnl'])
    random_off_mdds.append(st_off['mdd'])

q3_time = time.time() - t0

# 2×2 Matrix
matrix = {
    'Real+CascadeON': {'pnl': real_cascade_on['pnl'], 'mdd': real_cascade_on['mdd'],
                        'pnl_mdd': real_cascade_on['pnl_mdd']},
    'Real+CascadeOFF': {'pnl': real_cascade_off['pnl'], 'mdd': real_cascade_off['mdd'],
                         'pnl_mdd': real_cascade_off['pnl_mdd']},
    'Random+CascadeON': {'pnl': round(float(np.mean(random_on_pnls)), 1),
                          'mdd': round(float(np.mean(random_on_mdds)), 2),
                          'pnl_mdd': round(float(np.mean(
                              [p/m if m > 0 else 0 for p, m in zip(random_on_pnls, random_on_mdds)]
                          )), 1)},
    'Random+CascadeOFF': {'pnl': round(float(np.mean(random_off_pnls)), 1),
                           'mdd': round(float(np.mean(random_off_mdds)), 2),
                           'pnl_mdd': round(float(np.mean(
                               [p/m if m > 0 else 0 for p, m in zip(random_off_pnls, random_off_mdds)]
                           )), 1)},
}

print(f'\nQ3 Results ({q3_time:.0f}s):')
print(f'\n  {"Config":<22} {"PnL%":>10} {"MDD%":>10} {"PnL/MDD":>10}')
print(f'  {"-"*52}')
for name, vals in matrix.items():
    print(f'  {name:<22} {vals["pnl"]:>+10.1f} {vals["mdd"]:>10.2f} {vals["pnl_mdd"]:>10.1f}')

# Cascade effect decomposition
cascade_lift_real = real_cascade_on['pnl'] - real_cascade_off['pnl']
cascade_lift_random = np.mean(random_on_pnls) - np.mean(random_off_pnls)
pattern_lift_cascade_on = real_cascade_on['pnl'] - np.mean(random_on_pnls)
pattern_lift_cascade_off = real_cascade_off['pnl'] - np.mean(random_off_pnls)

print(f'\n  Effect Decomposition:')
print(f'    Cascade lift (real patterns):  {cascade_lift_real:+.1f}%')
print(f'    Cascade lift (random entries): {cascade_lift_random:+.1f}%')
print(f'    Pattern lift (cascade ON):     {pattern_lift_cascade_on:+.1f}%')
print(f'    Pattern lift (cascade OFF):    {pattern_lift_cascade_off:+.1f}%')

# Cascade independence test
if cascade_lift_random > 0 and cascade_lift_real > 0:
    cascade_independence = cascade_lift_random / cascade_lift_real * 100
    print(f'\n    Cascade independence: {cascade_independence:.0f}% '
          f'(random cascade lift / real cascade lift)')
    if cascade_independence > 60:
        q3_verdict = 'CASCADE-INDEPENDENT (works with any signal)'
    elif cascade_independence > 30:
        q3_verdict = 'CASCADE-SYNERGISTIC (amplified by patterns)'
    else:
        q3_verdict = 'CASCADE-DEPENDENT (requires pattern edge)'
elif cascade_lift_real > 0 and cascade_lift_random <= 0:
    cascade_independence = 0
    q3_verdict = 'CASCADE-DEPENDENT (only works with real patterns)'
else:
    cascade_independence = 100
    q3_verdict = 'CASCADE-INDEPENDENT (works without patterns)'

print(f'\n  >>> Q3 VERDICT: System is {q3_verdict}')


# ============================================================
# Q4 (Bonus): Mechanism Ablation with Random Signals
# ============================================================
print()
print('=' * 90)
print('Q4 (BONUS): Which mechanisms drive random signal profitability?')
print('=' * 90)
t0 = time.time()

# Use 20 random sets, test each mechanism OFF
mechanisms = {
    'Baseline (all ON)': {},
    'No Cascade': {'cascade_enabled': False},
    'No MDD Sizing': {'mdd_sizing_enabled': False},
    'No Momentum': {'momentum_enabled': False},
    'No AggRisk': {'agg_risk_enabled': False},
    'No DirCap': {'direction_cap_enabled': False},
    'No ATR Scale': {'atr_scaling_enabled': False},
    'No Timeout': {'timeout_enabled': False},
    'No EarlyExit': {'early_exit_enabled': False},
    'BARE (no mechanisms)': {
        'cascade_enabled': False, 'mdd_sizing_enabled': False,
        'momentum_enabled': False, 'agg_risk_enabled': False,
        'direction_cap_enabled': False, 'atr_scaling_enabled': False,
        'timeout_enabled': False, 'early_exit_enabled': False,
    },
}

N_Q4 = 10
q4_results = {}
for mech_name, mech_kwargs in mechanisms.items():
    pnls = []
    for trial in range(N_Q4):
        rng = np.random.default_rng(3000 + trial)
        rsig = generate_random_signals(
            rng, n_signals=n_real_neutral,
            bar_range_start=ns, bar_range_end=ne,
            long_ratio_target=long_ratio,
            tp_pool=real_tp_vals, sl_pool=real_sl_vals
        )
        trades, _ = portfolio_sim(
            signal_tuples=rsig, start_bar=ns, end_bar=ne,
            **mech_kwargs, **common
        )
        st = calc_stats(trades)
        pnls.append(st['pnl'])
    avg_pnl = np.mean(pnls)
    q4_results[mech_name] = {'avg_pnl': round(float(avg_pnl), 1),
                              'median_pnl': round(float(np.median(pnls)), 1)}

q4_time = time.time() - t0
baseline_pnl = q4_results['Baseline (all ON)']['avg_pnl']

print(f'\nQ4 Results ({q4_time:.0f}s) — Random signals, N={N_Q4} trials each:')
print(f'\n  {"Mechanism Config":<25} {"Avg PnL%":>10} {"Δ from base":>12}')
print(f'  {"-"*47}')
for name, vals in sorted(q4_results.items(), key=lambda x: x[1]['avg_pnl'], reverse=True):
    delta = vals['avg_pnl'] - baseline_pnl
    marker = ' ***' if abs(delta) > 20 else ''
    print(f'  {name:<25} {vals["avg_pnl"]:>+10.1f} {delta:>+12.1f}{marker}')


# ============================================================
# SYNTHESIS
# ============================================================
print()
print('=' * 90)
print('SYNTHESIS')
print('=' * 90)

print(f"""
Q1: WF Discrimination = {q1_verdict}
    Random PASS rate: {wf_pass_rate:.1f}%
    Implication: {'WF 3/3 PASS is a necessary but insufficient validation gate' if wf_pass_rate > 20
                  else 'WF 3/3 PASS has genuine discriminating power'}

Q2: Pattern Contribution = {q2_verdict}
    Pattern lift: {pattern_lift_pnl:+.1f}% ({pattern_contribution:.1f}% of total PnL)
    Mechanism-only: {random_pnl_mean:+.1f}%
    Implication: {'Patterns provide genuine predictive edge' if pattern_contribution > 60
                  else 'Mechanism stack is the primary profit driver, patterns are secondary' if pattern_contribution < 30
                  else 'Both patterns and mechanisms contribute significantly'}

Q3: Cascade Role = {q3_verdict}
    Cascade lift (real): {cascade_lift_real:+.1f}%
    Cascade lift (random): {cascade_lift_random:+.1f}%
    Implication: {'Cascade works independently of signal quality — it is a risk management tool, not a pattern amplifier' if cascade_independence > 60
                  else 'Cascade amplifies pattern edge — system needs both' if cascade_independence > 30
                  else 'Cascade only works with genuine pattern edge — dependent on signal quality'}

STRATEGIC IMPLICATIONS:
""")

if wf_pass_rate > 50 and pattern_contribution < 30:
    print('  >>> CRITICAL: System is fundamentally a mechanism-driven portfolio manager.')
    print('  >>> Patterns serve as entry timing (any timing works), WF cannot discriminate.')
    print('  >>> Priority: Improve mechanism stack, not pattern selection.')
elif wf_pass_rate > 50 and pattern_contribution > 60:
    print('  >>> PARADOX: Patterns contribute significantly but WF cannot discriminate.')
    print('  >>> WF validation methodology needs fundamental redesign.')
elif wf_pass_rate < 20 and pattern_contribution > 60:
    print('  >>> HEALTHY: Patterns provide real edge, WF can detect it.')
    print('  >>> Current approach is sound — continue pattern research.')
else:
    print(f'  >>> MIXED: WF pass rate {wf_pass_rate:.0f}%, pattern contribution {pattern_contribution:.0f}%.')
    print('  >>> Consider: (1) Stricter WF criteria, (2) Additional validation layers.')

# ============================================================
# Save results
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v1.42.0',
    'neutral_window': {'start': ns, 'end': ne, 'bars': ne - ns},
    'real_signals_neutral': n_real_neutral,
    'q1_wf_discrimination': {
        'n_random_sets': N_RANDOM_SETS,
        'wf_pass_count': wf_pass_count,
        'wf_pass_rate_pct': round(wf_pass_rate, 1),
        'random_is_pnl_mean': round(float(np.mean([s['pnl'] for s in random_is_stats])), 1),
        'random_is_wr_mean': round(float(np.mean([s['wr'] for s in random_is_stats])), 1),
        'real_is_pnl': real_is_stats['pnl'],
        'real_is_wr': real_is_stats['wr'],
        'real_wf_pass': real_n_pass,
        'verdict': q1_verdict,
    },
    'q2_pattern_independence': {
        'n_random_trials': N_RANDOM_Q2,
        'random_pnl_mean': round(random_pnl_mean, 1),
        'random_pnl_median': round(float(random_pnl_median), 1),
        'random_pnl_positive_pct': round(random_pnl_positive, 1),
        'random_wr_mean': round(float(random_wr_mean), 1),
        'real_pnl': real_is_stats['pnl'],
        'pattern_lift_pnl': round(pattern_lift_pnl, 1),
        'pattern_contribution_pct': round(pattern_contribution, 1),
        'verdict': q2_verdict,
    },
    'q3_cascade_dependency': {
        'matrix': matrix,
        'cascade_lift_real': round(cascade_lift_real, 1),
        'cascade_lift_random': round(float(cascade_lift_random), 1),
        'pattern_lift_cascade_on': round(float(pattern_lift_cascade_on), 1),
        'pattern_lift_cascade_off': round(float(pattern_lift_cascade_off), 1),
        'cascade_independence_pct': round(float(cascade_independence), 1),
        'verdict': q3_verdict,
    },
    'q4_mechanism_ablation_random': q4_results,
    'synthesis': {
        'q1': q1_verdict,
        'q2': q2_verdict,
        'q3': q3_verdict,
    },
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved: {OUTPUT_FILE}')

total_time = time.time() - t0
print(f'\nTotal study time: {q1_time + q2_time + q3_time + q4_time:.0f}s')
