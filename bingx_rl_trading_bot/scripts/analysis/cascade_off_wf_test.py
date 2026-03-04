#!/usr/bin/env python3
"""Quick WF test: Cascade OFF + various mechanism combos.
Tests 8 configurations with IS full + WF 3-fold expanding window.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, portfolio_sim, calc_stats,
    DATA_FILE, PATTERNS_FILE
)

print('Loading data...')
df = load_and_classify(DATA_FILE)
atr_ratio = compute_atr_ratio(df)
ema_slope = compute_ema_slope(df['close'].values)
ns, ne = find_neutral_window(df['close'].values, 1.0)
print(f'Neutral: {ns}-{ne} ({ne-ns} bars, {(ne-ns)/288:.0f}d)')

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

# Build signal tuples
rctypes = df['rctype'].values
n_bars = len(df)
signal_tuples = []
for i in range(2, n_bars):
    tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
    if tri in pat_lookup:
        p = pat_lookup[tri]
        signal_tuples.append((i, tri, p['direction'], p['tp'], p['sl']))
print(f'{len(signal_tuples)} signals')

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
tcodes = df['rctype'].values

# Common args
common = dict(
    opens=opens, highs=highs, lows=lows, closes=closes,
    type_codes=tcodes, n_bars=n_bars,
    atr_ratio=atr_ratio, ema_slope=ema_slope,
)

# Configs to test
configs = {
    'A_full_8_current': dict(),
    'B_cascade_OFF': dict(cascade_enabled=False),
    'C_cascade+mom_OFF': dict(cascade_enabled=False, momentum_enabled=False),
    'D_cascade+early_OFF': dict(cascade_enabled=False, early_exit_enabled=False),
    'E_cascade+mdd_OFF': dict(cascade_enabled=False, mdd_sizing_enabled=False),
    'F_cascade+dircap_OFF': dict(cascade_enabled=False, direction_cap_enabled=False),
    'G_g5+atr+to': dict(cascade_enabled=False, direction_cap_enabled=False,
                         momentum_enabled=False, mdd_sizing_enabled=False,
                         early_exit_enabled=False),
    'H_g5+atr+to+mdd': dict(cascade_enabled=False, direction_cap_enabled=False,
                              momentum_enabled=False, early_exit_enabled=False),
}

# ============================================================
# PHASE A: IS Full Neutral Window
# ============================================================
print()
print('=' * 95)
print('PHASE A: IS Full Neutral Window')
print('=' * 95)
header = f'{"Config":<25} {"PnL%":>8} {"MDD%":>8} {"P/M":>8} {"WR%":>8} {"Trades":>8}'
print(header)
print('-' * 70)

is_results = {}
for name, kwargs in configs.items():
    trades, _ = portfolio_sim(signal_tuples=signal_tuples,
                              start_bar=ns, end_bar=ne,
                              **kwargs, **common)
    stats = calc_stats(trades)
    is_results[name] = stats
    print(f'{name:<25} {stats["pnl"]:>+8.1f} {stats["mdd"]:>8.2f} '
          f'{stats["pnl_mdd"]:>8.1f} {stats["wr"]:>8.1f} {stats["trades"]:>8}')

# ============================================================
# PHASE B: WF 3-fold Expanding Window
# ============================================================
print()
print('=' * 95)
print('PHASE B: WF 3-fold Expanding Window (OOS only)')
print('=' * 95)

n_folds = 3
total = ne - ns
seg_size = total // (n_folds + 1)

wf_results = {}
for name, kwargs in configs.items():
    folds = []
    for fold in range(n_folds):
        oos_start = ns + seg_size * (fold + 1)
        oos_end = ns + seg_size * (fold + 2) if fold < n_folds - 1 else ne
        trades_f, _ = portfolio_sim(signal_tuples=signal_tuples,
                                    start_bar=oos_start, end_bar=oos_end,
                                    **kwargs, **common)
        stats_f = calc_stats(trades_f)
        stats_f['fold'] = fold + 1
        folds.append(stats_f)

    n_pass = sum(1 for f in folds if f['pnl'] > 0)
    avg_oos = np.mean([f['pnl'] for f in folds])
    min_oos = min(f['pnl'] for f in folds)
    verdict = f'{n_pass}/3 PASS' if n_pass == 3 else f'{n_pass}/3 FAIL'
    wf_results[name] = {'folds': folds, 'n_pass': n_pass,
                         'avg_oos': round(float(avg_oos), 2),
                         'min_oos': round(float(min_oos), 2),
                         'verdict': verdict}

# Print WF summary
print(f'\n{"Config":<25} {"F1":>8} {"F2":>8} {"F3":>8} {"Avg":>8} {"Min":>8} {"Verdict":>10}')
print('-' * 80)
for name in configs:
    r = wf_results[name]
    f = r['folds']
    print(f'{name:<25} {f[0]["pnl"]:>+8.1f} {f[1]["pnl"]:>+8.1f} {f[2]["pnl"]:>+8.1f} '
          f'{r["avg_oos"]:>+8.1f} {r["min_oos"]:>+8.1f} {r["verdict"]:>10}')

# Detailed fold stats
print(f'\nDetailed fold stats:')
for name in configs:
    r = wf_results[name]
    print(f'\n  {name}:')
    for f in r['folds']:
        print(f'    Fold {f["fold"]}: PnL={f["pnl"]:+.1f}%, WR={f["wr"]:.1f}%, '
              f'MDD={f["mdd"]:.2f}%, Trades={f["trades"]}')

# ============================================================
# PHASE C: Decision Matrix
# ============================================================
print()
print('=' * 95)
print('PHASE C: Decision Matrix')
print('=' * 95)

# Rank by: WF all pass first, then IS PnL/MDD
ranked = sorted(configs.keys(), key=lambda n: (
    wf_results[n]['n_pass'] == 3,  # WF pass first
    is_results[n]['pnl_mdd'],       # then PnL/MDD
), reverse=True)

print(f'\n{"Rank":<6} {"Config":<25} {"IS P/M":>8} {"IS PnL":>8} {"IS MDD":>8} '
      f'{"OOS Avg":>8} {"OOS Min":>8} {"WF":>10}')
print('-' * 90)
for i, name in enumerate(ranked, 1):
    isr = is_results[name]
    wfr = wf_results[name]
    print(f'{i:<6} {name:<25} {isr["pnl_mdd"]:>8.1f} {isr["pnl"]:>+8.1f} {isr["mdd"]:>8.2f} '
          f'{wfr["avg_oos"]:>+8.1f} {wfr["min_oos"]:>+8.1f} {wfr["verdict"]:>10}')

# Best WF-passing config
best_wf = [n for n in ranked if wf_results[n]['n_pass'] == 3]
if best_wf:
    print(f'\n>>> RECOMMENDED: {best_wf[0]} (best PnL/MDD among WF 3/3 PASS configs)')
else:
    print(f'\n>>> WARNING: No config achieved WF 3/3 PASS!')

# Save results
output = {
    'is_results': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                        for kk, vv in v.items()} for k, v in is_results.items()},
    'wf_results': {k: {'n_pass': v['n_pass'], 'avg_oos': v['avg_oos'],
                        'min_oos': v['min_oos'], 'verdict': v['verdict'],
                        'folds': [{kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                                    for kk, vv in f.items()} for f in v['folds']]}
                   for k, v in wf_results.items()},
    'ranking': ranked,
    'recommended': best_wf[0] if best_wf else None,
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'cascade_off_wf_test.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved: {out_path}')
