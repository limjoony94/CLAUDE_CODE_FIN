#!/usr/bin/env python3
"""
AggRisk × Cascade Cross-Validation Study
==========================================

v1.43.0 ROLLBACK 교훈 적용:
  1. WF 3/3 PASS는 비차별적 (100% 랜덤 신호 PASS)
  2. Cascade 상호작용이 독립적 효과를 과대평가할 수 있음

검증 질문:
  Q1: AggRisk 완화가 Cascade 독립적인가? (Cascade-OFF에서도 개선?)
  Q2: AggRisk OFF + 랜덤 신호도 WF PASS하는가? (판별력 검증)
  Q3: 최악 correlated loss 시나리오에서 daily loss 허용 범위인가?

Standard Research Protocol: LEVERAGE=3, FEE×LEV, Timeout DROP, ATR-scaled, Compound

Author: Research Agent
Date: 2026-03-04
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, calc_stats,
    DATA_FILE, PATTERNS_FILE,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER, TIMEOUT_BARS, N_SLOTS, DIRECTION_CAP,
    MOMENTUM_LOOKBACK, MOMENTUM_THRESHOLD, MOMENTUM_COOLDOWN,
    ATR_PERIOD, ATR_WINDOW, ATR_CLAMP_LO, ATR_CLAMP_HI,
    EMA_PERIOD, EMA_LOOKBACK, BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
    clamp,
)

# Reuse custom sim from aggrisk study
from scripts.analysis.aggrisk_relaxation_study import portfolio_sim_custom

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'aggrisk_cascade_validation.json')


def generate_random_signals(rng, n_signals, bar_start, bar_end, tp_pool, sl_pool):
    """Generate random signals matching real signal count and TP/SL distribution."""
    bars = rng.integers(bar_start + 2, bar_end, size=n_signals * 2)
    bars = np.sort(np.unique(bars))[:n_signals]
    signals = []
    for b in bars:
        d = 'LONG' if rng.random() < 0.47 else 'SHORT'  # match 61L/69S ratio
        tp = float(rng.choice(tp_pool))
        sl = float(rng.choice(sl_pool))
        signals.append((int(b), f'RND_{b}', d, tp, sl))
    return signals


def analyze_correlated_loss(trades):
    """Analyze worst-case correlated loss patterns."""
    if not trades:
        return {}

    # Group SL exits by day
    sl_trades = [t for t in trades if t['reason'] == 'SL']
    if not sl_trades:
        return {'sl_count': 0, 'max_same_dir_sl_day': 0, 'worst_burst_loss': 0.0}

    # Group by day
    daily_sl = {}
    for t in sl_trades:
        day = t['exit_bar'] // BARS_PER_DAY
        daily_sl.setdefault(day, []).append(t)

    # Worst same-direction SL cluster
    max_same_dir = 0
    worst_burst = 0.0
    for day, day_trades in daily_sl.items():
        long_sl = [t for t in day_trades if t['direction'] == 'LONG']
        short_sl = [t for t in day_trades if t['direction'] == 'SHORT']
        max_dir = max(len(long_sl), len(short_sl))
        if max_dir > max_same_dir:
            max_same_dir = max_dir
            worst_cluster = long_sl if len(long_sl) >= len(short_sl) else short_sl
            worst_burst = sum(t['pnl_portfolio'] for t in worst_cluster)

    # Max consecutive SL exits (any direction)
    sorted_sl = sorted(sl_trades, key=lambda t: t['exit_bar'])
    max_consec = 1
    cur_consec = 1
    for i in range(1, len(sorted_sl)):
        if sorted_sl[i]['exit_bar'] - sorted_sl[i-1]['exit_bar'] <= 12:  # within 1h
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 1

    return {
        'sl_count': len(sl_trades),
        'max_same_dir_sl_day': max_same_dir,
        'worst_burst_loss': round(worst_burst, 2),
        'max_consecutive_sl_1h': max_consec,
        'sl_days': len(daily_sl),
        'avg_sl_per_sl_day': round(len(sl_trades) / max(1, len(daily_sl)), 1),
    }


# ============================================================
# Main
# ============================================================
print('=' * 90)
print('AGGRISK × CASCADE CROSS-VALIDATION STUDY')
print('=' * 90)
print()

print('Loading data...')
df = load_and_classify(DATA_FILE)
atr_ratio = compute_atr_ratio(df)
ema_slope = compute_ema_slope(df['close'].values)
ns, ne = find_neutral_window(df['close'].values, 1.0)
print(f'Neutral: {ns}-{ne} ({ne-ns} bars, {(ne-ns)/288:.0f}d)')

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

rctypes = df['rctype'].values
n_bars = len(df)
all_signals = []
for i in range(2, n_bars):
    tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
    if tri in pat_lookup:
        p = pat_lookup[tri]
        all_signals.append((i, tri, p['direction'], p['tp'], p['sl']))

tp_pool = np.array([v['tp'] for v in pat_lookup.values()])
sl_pool = np.array([v['sl'] for v in pat_lookup.values()])

print(f'{len(all_signals)} signals, {len(pat_lookup)} patterns')

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

# Configs to test (top 3 from relaxation + baseline)
configs = {
    'A_baseline_3_7':   {'agg_counter_cap': 3.0, 'agg_with_cap': 7.0},
    'D_relax_5_15':     {'agg_counter_cap': 5.0, 'agg_with_cap': 15.0},
    'H_counter_5':      {'agg_counter_cap': 5.0, 'agg_with_cap': 999.0},
    'J_OFF':            {'agg_risk_enabled': False},
}

# ============================================================
# Q1: Cascade Independence Test
# ============================================================
print()
print('=' * 90)
print('Q1: AggRisk × Cascade Cross-Test')
print('    Does AggRisk relaxation benefit independently of Cascade?')
print('=' * 90)

header = f'{"Config":<30} {"PnL%":>9} {"MDD%":>8} {"P/M":>8} {"WR%":>7} {"Trades":>7} {"MaxDL":>8}'
print(header)
print('-' * 80)

q1_results = {}
for name, kwargs in configs.items():
    for cascade in [True, False]:
        label = f'{name}_{"CascON" if cascade else "CascOFF"}'
        trades, max_dl, _ = portfolio_sim_custom(
            signal_tuples=all_signals, start_bar=ns, end_bar=ne,
            cascade_enabled=cascade, **kwargs, **common
        )
        st = calc_stats(trades)
        st['max_daily_loss'] = round(max_dl, 2)
        q1_results[label] = st
        print(f'{label:<30} {st["pnl"]:>+9.1f} {st["mdd"]:>8.2f} {st["pnl_mdd"]:>8.1f} '
              f'{st["wr"]:>7.1f} {st["trades"]:>7} {st["max_daily_loss"]:>+8.2f}')

# Cascade dependency ratio for each config
print(f'\n--- Cascade Dependency Analysis ---')
for name in configs:
    on = q1_results[f'{name}_CascON']
    off = q1_results[f'{name}_CascOFF']
    cascade_lift = on['pnl'] - off['pnl']
    if on['pnl'] != 0:
        dependency = cascade_lift / on['pnl'] * 100
    else:
        dependency = 0
    print(f'  {name:<20} Cascade lift: {cascade_lift:+.1f}%, dependency: {dependency:.0f}%, '
          f'CascOFF PnL/MDD: {off["pnl_mdd"]:.1f}')

# Key test: Does relaxation improve even WITHOUT Cascade?
baseline_off = q1_results['A_baseline_3_7_CascOFF']['pnl_mdd']
improved_off = {k: v['pnl_mdd'] for k, v in q1_results.items() if 'CascOFF' in k}
print(f'\n  Baseline CascOFF PnL/MDD: {baseline_off:.1f}')
for label, pm in improved_off.items():
    if 'baseline' not in label:
        delta = pm - baseline_off
        print(f'  {label}: {pm:.1f} ({delta:+.1f} vs baseline)')


# ============================================================
# Q2: Random Signal Discrimination Test
# ============================================================
print()
print('=' * 90)
print('Q2: Random Signal Discrimination (AggRisk OFF + Cascade ON)')
print('    If random signals also pass WF with AggRisk OFF, it is non-discriminating')
print('=' * 90)

N_RANDOM = 15  # random signal sets to test
n_folds = 3
total = ne - ns
seg_size = total // (n_folds + 1)

n_real_signals = len([s for s in all_signals if ns <= s[0] < ne])
rng = np.random.default_rng(42)

random_wf_pass = 0
random_results = []
for trial in range(N_RANDOM):
    rand_sigs = generate_random_signals(
        rng, n_real_signals, ns, ne, tp_pool, sl_pool
    )

    # WF test with AggRisk OFF + Cascade ON
    fold_pnls = []
    for fold in range(n_folds):
        oos_start = ns + seg_size * (fold + 1)
        oos_end = ns + seg_size * (fold + 2) if fold < n_folds - 1 else ne
        trades_f, _, _ = portfolio_sim_custom(
            signal_tuples=rand_sigs, start_bar=oos_start, end_bar=oos_end,
            agg_risk_enabled=False, cascade_enabled=True, **common
        )
        st_f = calc_stats(trades_f)
        fold_pnls.append(st_f['pnl'])

    n_pass = sum(1 for p in fold_pnls if p > 0)
    is_pass = n_pass == 3
    if is_pass:
        random_wf_pass += 1
    random_results.append({
        'folds': fold_pnls, 'n_pass': n_pass, 'pass': is_pass
    })
    if (trial + 1) % 5 == 0:
        print(f'  Trial {trial+1}/{N_RANDOM}: {random_wf_pass}/{trial+1} pass so far')

random_pass_rate = random_wf_pass / N_RANDOM * 100
print(f'\nQ2 Result: {random_wf_pass}/{N_RANDOM} random sets WF 3/3 PASS ({random_pass_rate:.0f}%)')
if random_pass_rate >= 80:
    print('  >>> NON-DISCRIMINATING: AggRisk OFF + Cascade ON passes random signals too')
    print('  >>> Improvement is likely Cascade-driven, not genuine edge release')
else:
    print(f'  >>> PARTIALLY DISCRIMINATING: {100-random_pass_rate:.0f}% rejection rate')

# Also test AggRisk OFF + Cascade OFF with random
random_cascade_off_pass = 0
for trial in range(N_RANDOM):
    rand_sigs = generate_random_signals(
        np.random.default_rng(42 + trial + 100), n_real_signals, ns, ne, tp_pool, sl_pool
    )
    fold_pnls = []
    for fold in range(n_folds):
        oos_start = ns + seg_size * (fold + 1)
        oos_end = ns + seg_size * (fold + 2) if fold < n_folds - 1 else ne
        trades_f, _, _ = portfolio_sim_custom(
            signal_tuples=rand_sigs, start_bar=oos_start, end_bar=oos_end,
            agg_risk_enabled=False, cascade_enabled=False, **common
        )
        st_f = calc_stats(trades_f)
        fold_pnls.append(st_f['pnl'])
    n_pass = sum(1 for p in fold_pnls if p > 0)
    if n_pass == 3:
        random_cascade_off_pass += 1

rc_off_rate = random_cascade_off_pass / N_RANDOM * 100
print(f'\nAggRisk OFF + Cascade OFF random: {random_cascade_off_pass}/{N_RANDOM} PASS ({rc_off_rate:.0f}%)')
print(f'  Cascade contributes {random_pass_rate - rc_off_rate:.0f}pp to random WF pass rate')


# ============================================================
# Q3: Correlated Loss Stress Test
# ============================================================
print()
print('=' * 90)
print('Q3: Correlated Loss Stress Test')
print('    Max same-direction SL cluster analysis per AggRisk config')
print('=' * 90)

header = f'{"Config":<25} {"SLs":>5} {"MaxDirSL/d":>11} {"WorstBurst%":>12} {"MaxConsec1h":>12} {"SL/day":>8}'
print(header)
print('-' * 80)

q3_results = {}
for name, kwargs in configs.items():
    trades, max_dl, _ = portfolio_sim_custom(
        signal_tuples=all_signals, start_bar=ns, end_bar=ne,
        cascade_enabled=True, **kwargs, **common
    )
    cl = analyze_correlated_loss(trades)
    q3_results[name] = cl
    print(f'{name:<25} {cl.get("sl_count",0):>5} {cl.get("max_same_dir_sl_day",0):>11} '
          f'{cl.get("worst_burst_loss",0):>+12.2f} {cl.get("max_consecutive_sl_1h",0):>12} '
          f'{cl.get("avg_sl_per_sl_day",0):>8.1f}')

# Cascade OFF stress test
print(f'\n--- Same analysis with Cascade OFF ---')
print(header)
print('-' * 80)
for name, kwargs in configs.items():
    trades, max_dl, _ = portfolio_sim_custom(
        signal_tuples=all_signals, start_bar=ns, end_bar=ne,
        cascade_enabled=False, **kwargs, **common
    )
    cl = analyze_correlated_loss(trades)
    q3_results[f'{name}_CascOFF'] = cl
    print(f'{name+"_CascOFF":<25} {cl.get("sl_count",0):>5} {cl.get("max_same_dir_sl_day",0):>11} '
          f'{cl.get("worst_burst_loss",0):>+12.2f} {cl.get("max_consecutive_sl_1h",0):>12} '
          f'{cl.get("avg_sl_per_sl_day",0):>8.1f}')


# ============================================================
# Q4: WF with Cascade-OFF (the real test)
# ============================================================
print()
print('=' * 90)
print('Q4: WF 3-fold with Cascade-OFF (true edge test)')
print('    If AggRisk relaxation improves WF even without Cascade, it is a real gain')
print('=' * 90)

wf_cascade_off = {}
for name, kwargs in configs.items():
    folds = []
    for fold in range(n_folds):
        oos_start = ns + seg_size * (fold + 1)
        oos_end = ns + seg_size * (fold + 2) if fold < n_folds - 1 else ne
        trades_f, _, _ = portfolio_sim_custom(
            signal_tuples=all_signals, start_bar=oos_start, end_bar=oos_end,
            cascade_enabled=False, **kwargs, **common
        )
        st_f = calc_stats(trades_f)
        folds.append(st_f)
    n_pass = sum(1 for f in folds if f['pnl'] > 0)
    avg_oos = np.mean([f['pnl'] for f in folds])
    wf_cascade_off[name] = {
        'n_pass': n_pass,
        'avg_oos': round(float(avg_oos), 1),
        'folds_pnl': [round(f['pnl'], 1) for f in folds],
    }

print(f'\n{"Config":<25} {"F1":>8} {"F2":>8} {"F3":>8} {"Avg":>8} {"WF":>8}')
print('-' * 65)
for name in configs:
    r = wf_cascade_off[name]
    verdict = f'{r["n_pass"]}/3 {"P" if r["n_pass"]==3 else "F"}'
    print(f'{name:<25} {r["folds_pnl"][0]:>+8.1f} {r["folds_pnl"][1]:>+8.1f} '
          f'{r["folds_pnl"][2]:>+8.1f} {r["avg_oos"]:>+8.1f} {verdict:>8}')


# ============================================================
# SYNTHESIS
# ============================================================
print()
print('=' * 90)
print('SYNTHESIS: Production Decision')
print('=' * 90)

# Cascade dependency
baseline_dep = (q1_results['A_baseline_3_7_CascON']['pnl'] -
                q1_results['A_baseline_3_7_CascOFF']['pnl'])
off_dep = (q1_results['J_OFF_CascON']['pnl'] -
           q1_results['J_OFF_CascOFF']['pnl'])

print(f'\n1. Cascade Dependency:')
print(f'   Baseline (3/7): Cascade lift = {baseline_dep:+.1f}%')
print(f'   AggRisk OFF:    Cascade lift = {off_dep:+.1f}%')
if abs(off_dep) > abs(baseline_dep) * 1.5:
    print('   >>> AggRisk OFF INCREASES Cascade dependency (concerning)')
elif abs(off_dep) < abs(baseline_dep) * 0.5:
    print('   >>> AggRisk OFF is Cascade-INDEPENDENT (good)')
else:
    print('   >>> Similar Cascade dependency across configs')

print(f'\n2. Random Signal Discrimination:')
print(f'   AggRisk OFF + Cascade ON:  {random_pass_rate:.0f}% random WF pass')
print(f'   AggRisk OFF + Cascade OFF: {rc_off_rate:.0f}% random WF pass')
if random_pass_rate >= 80:
    print('   >>> WF is NON-DISCRIMINATING for AggRisk changes (with Cascade ON)')
else:
    print(f'   >>> WF has {100-random_pass_rate:.0f}% discrimination power')

print(f'\n3. Cascade-OFF WF (true edge):')
for name in configs:
    r = wf_cascade_off[name]
    label = 'PASS' if r['n_pass'] == 3 else 'FAIL'
    print(f'   {name:<20} {label} (avg {r["avg_oos"]:+.1f}%)')

print(f'\n4. Correlated Loss Risk:')
for name in ['A_baseline_3_7', 'D_relax_5_15', 'H_counter_5', 'J_OFF']:
    cl = q3_results[name]
    print(f'   {name:<20} max same-dir SL/day: {cl.get("max_same_dir_sl_day",0)}, '
          f'worst burst: {cl.get("worst_burst_loss",0):+.2f}%')

# Final recommendation
print(f'\n5. FINAL RECOMMENDATION:')

# Check if Cascade-OFF WF improves with relaxation
off_baseline_avg = wf_cascade_off['A_baseline_3_7']['avg_oos']
off_best = max(wf_cascade_off.items(), key=lambda x: x[1]['avg_oos'] if x[1]['n_pass'] == 3 else -999)
off_best_name, off_best_data = off_best

cascade_off_improves = off_best_data['avg_oos'] > off_baseline_avg * 1.1  # >10% improvement

if random_pass_rate >= 80 and not cascade_off_improves:
    print('   >>> HOLD: AggRisk relaxation is Cascade-amplified, not genuine edge release')
    print('   >>> Keep current 3/7 baseline (safe)')
    recommendation = 'HOLD_3_7'
elif cascade_off_improves and random_pass_rate < 80:
    print(f'   >>> RELAX: {off_best_name} shows genuine improvement even without Cascade')
    recommendation = off_best_name
elif cascade_off_improves and random_pass_rate >= 80:
    print(f'   >>> CAUTIOUS RELAX: {off_best_name} improves Cascade-OFF, but WF non-discriminating')
    print('   >>> Consider moderate relaxation (D_relax_5_15) over full removal')
    recommendation = 'CAUTIOUS_' + off_best_name
else:
    print('   >>> HOLD: No clear evidence for relaxation')
    recommendation = 'HOLD_3_7'


# ============================================================
# Save
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v1.42.0',
    'q1_cascade_cross': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                              for kk, vv in v.items()} for k, v in q1_results.items()},
    'q2_random_discrimination': {
        'n_trials': N_RANDOM,
        'aggrisk_off_cascade_on_pass_rate': random_pass_rate,
        'aggrisk_off_cascade_off_pass_rate': rc_off_rate,
    },
    'q3_correlated_loss': {k: v for k, v in q3_results.items()},
    'q4_cascade_off_wf': wf_cascade_off,
    'recommendation': recommendation,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
print(f'\nSaved: {OUTPUT_FILE}')
