#!/usr/bin/env python3
"""
Cascade tighten_pct Tuning + Pattern Quality Filter Study
============================================================

Two independent optimizations tested with v1.44.0 baseline (AggRisk 5/15):

Phase 1: Cascade tighten_pct sweep (50/60/70/75/80/85/90%)
  - 75% was chosen from correlated_loss_study without alternative comparison
  - Cascade is #1 profit driver — parameter tuning has outsized impact

Phase 2: Pattern quality filter (25 genuine / 65 genuine+marginal / 130 full)
  - pattern_genuine_edge_study: 25 genuine (p<0.05), 40 marginal (p<0.10), 65 no-edge
  - Fewer patterns = better signal quality but less Cascade fuel

Phase 3: Best combination WF validation

Standard Research Protocol: LEVERAGE=3, FEE×LEV, Timeout DROP, ATR-scaled, Compound

Author: Research Agent
Date: 2026-03-05
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
    ATR_CLAMP_LO, ATR_CLAMP_HI, BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'cascade_pct_pattern_filter_study.json')

# v1.44.0 AggRisk settings
AGG_COUNTER_CAP = 5.0
AGG_WITH_CAP = 15.0


def portfolio_sim_v144(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    tighten_pct=75,
):
    """Portfolio sim matching v1.44.0 production (AggRisk 5/15)."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS
    cascade_mult = 1.0 - tighten_pct / 100.0  # e.g., 75% → 0.25

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    sig_by_bar = {}
    for s_bar, pat, direction, tp, sl in signal_tuples:
        if start_bar <= s_bar < end_bar:
            sig_by_bar.setdefault(s_bar, []).append((pat, direction, tp, sl))

    for bar in range(start_bar, end_bar):
        if bar >= n_bars - 1:
            break

        closed_slots = set()
        sl_exits = []

        h_bar = highs[bar] if bar < n_bars else 0
        l_bar = lows[bar] if bar < n_bars else 0
        o_bar = opens[bar] if bar < n_bars else 0

        for pos in positions:
            entry_bar = pos['entry_bar']
            if bar < entry_bar:
                continue
            entry_p = pos['entry_price']
            if entry_p <= 0:
                continue

            direction = pos['direction']
            eff_tp = pos['eff_tp_pct']
            eff_sl = pos['eff_sl_pct']
            sm = pos['size_mult']
            hold = bar - entry_bar

            exit_price = None
            reason = None

            if hold >= TIMEOUT_BARS:
                closed_slots.add(pos['slot'])
                continue

            if hold >= EARLY_CONFIRM:
                exit_types = ['BD'] if direction == 'LONG' else ['BU']
                candle_ok = True
                for k in range(EARLY_CONFIRM):
                    cb = bar - 1 - k
                    if cb < 0 or cb >= len(type_codes) or type_codes[cb] not in exit_types:
                        candle_ok = False
                        break
                if candle_ok:
                    cp = closes[bar - 1] if bar - 1 < n_bars else entry_p
                    if direction == 'LONG':
                        unr = (cp / entry_p - 1) * 100 * LEVERAGE
                    else:
                        unr = (1 - cp / entry_p) * 100 * LEVERAGE
                    if unr >= EARLY_MIN_PROFIT:
                        exit_price = cp
                        reason = 'EARLY'

            if reason is None:
                if direction == 'LONG':
                    tp_p = entry_p * (1 + eff_tp / 100)
                    sl_p = entry_p * (1 - eff_sl / 100)
                    hit_tp = h_bar >= tp_p
                    hit_sl = l_bar <= sl_p
                else:
                    tp_p = entry_p * (1 - eff_tp / 100)
                    sl_p = entry_p * (1 + eff_sl / 100)
                    hit_tp = l_bar <= tp_p
                    hit_sl = h_bar >= sl_p

                if hit_tp and hit_sl:
                    if abs(tp_p - o_bar) <= abs(sl_p - o_bar):
                        exit_price, reason = tp_p, 'TP'
                    else:
                        exit_price, reason = sl_p, 'SL'
                elif hit_tp:
                    exit_price, reason = tp_p, 'TP'
                elif hit_sl:
                    exit_price, reason = sl_p, 'SL'

            if reason is None:
                continue

            if direction == 'LONG':
                pnl = (exit_price / entry_p - 1) * 100 * LEVERAGE
            else:
                pnl = (1 - exit_price / entry_p) * 100 * LEVERAGE
            pnl -= fee

            pnl_portfolio = pnl * (size_pct / 100) * sm
            trades.append({
                'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
                'reason': reason, 'pattern': pos['pattern'],
                'direction': direction, 'pnl_portfolio': pnl_portfolio,
            })
            closed_slots.add(pos['slot'])

            if reason == 'SL':
                sl_exits.append(pos)

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Cascade SL tightening
        if sl_exits:
            for sl_pos in sl_exits:
                sl_dir = sl_pos['direction']
                for pos in positions:
                    if pos['direction'] == sl_dir:
                        pos['eff_sl_pct'] *= cascade_mult

        equity_delta = sum(t['pnl_portfolio'] for t in trades if t['exit_bar'] == bar)
        equity *= (1 + equity_delta / 100)
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard
        if bar >= MOMENTUM_LOOKBACK:
            pa = closes[bar - MOMENTUM_LOOKBACK]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > MOMENTUM_THRESHOLD:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'], bar + MOMENTUM_COOLDOWN)
                elif pct < -MOMENTUM_THRESHOLD:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'], bar + MOMENTUM_COOLDOWN)

        if bar not in sig_by_bar:
            continue

        for pat, direction, tp_pct, sl_pct in sig_by_bar[bar]:
            if len(positions) >= N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            if bar < mom_pause_until.get(direction, -1):
                continue

            sm = 1.0
            if peak_equity > 0:
                dd_pct = (peak_equity - equity) / peak_equity * 100
                if dd_pct <= MDD_FULL_BELOW:
                    mdd_scale = 1.0
                elif dd_pct >= MDD_MIN_ABOVE:
                    mdd_scale = MDD_MIN_SCALE
                else:
                    mdd_scale = 1.0 - (1.0 - MDD_MIN_SCALE) * (
                        dd_pct - MDD_FULL_BELOW) / (MDD_MIN_ABOVE - MDD_FULL_BELOW)
                sm *= mdd_scale

            if bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                r = clamp(atr_ratio[bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # AggRisk cap (v1.44.0: 5/15)
            slope = ema_slope[bar] if bar < len(ema_slope) else 0
            is_uptrend = slope > 0
            is_counter = ((direction == 'SHORT' and is_uptrend) or
                          (direction == 'LONG' and not is_uptrend))
            cap_pct = AGG_COUNTER_CAP if is_counter else AGG_WITH_CAP

            existing = sum(
                p['eff_sl_pct'] * (1.0 / N_SLOTS) * LEVERAGE * p['size_mult']
                for p in positions if p['direction'] == direction
            )
            new_exp = eff_sl * (1.0 / N_SLOTS) * LEVERAGE * sm
            if existing + new_exp > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{bar}",
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'direction': direction,
                'pattern': pat,
                'eff_tp_pct': eff_tp,
                'eff_sl_pct': eff_sl,
                'size_mult': sm,
            })

    # Force-close remaining
    for pos in positions:
        eb = pos['entry_bar']
        if eb >= n_bars:
            continue
        ep = pos['entry_price']
        if ep <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / ep - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / ep) * 100 * LEVERAGE
        pnl -= fee
        sm = pos['size_mult']
        trades.append({
            'entry_bar': eb, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'pnl_portfolio': pnl * (size_pct / 100) * sm,
        })

    return trades


# ============================================================
# Main
# ============================================================
print('=' * 90)
print('CASCADE TIGHTEN_PCT + PATTERN FILTER STUDY (v1.44.0 baseline)')
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

# Load genuine/marginal from previous study
with open(os.path.join(_PROJECT_ROOT, 'results', 'pattern_genuine_edge_study.json')) as f:
    edge_data = json.load(f)
genuine_pats = set(edge_data['phase2_edge_detection']['genuine_patterns'])
marginal_pats = set(edge_data['phase2_edge_detection']['marginal_patterns'])

rctypes = df['rctype'].values
n_bars = len(df)

# Build all signals
all_signals = []
for i in range(2, n_bars):
    tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
    if tri in pat_lookup:
        p = pat_lookup[tri]
        all_signals.append((i, tri, p['direction'], p['tp'], p['sl']))

# Build filtered signal sets
genuine_signals = [s for s in all_signals if s[1] in genuine_pats]
gm_signals = [s for s in all_signals if s[1] in genuine_pats or s[1] in marginal_pats]

print(f'Signals: full={len(all_signals)}, genuine+marginal={len(gm_signals)}, genuine={len(genuine_signals)}')

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
# Phase 1: Cascade tighten_pct sweep (full 130 patterns)
# ============================================================
print()
print('=' * 90)
print('PHASE 1: Cascade tighten_pct Sweep (full 130 patterns, v1.44.0 AggRisk 5/15)')
print('=' * 90)

tighten_values = [50, 60, 70, 75, 80, 85, 90]

header = f'{"tighten%":<10} {"PnL%":>9} {"MDD%":>8} {"P/M":>8} {"WR%":>7} {"Trades":>7}'
print(header)
print('-' * 55)

phase1_results = {}
for t in tighten_values:
    trades = portfolio_sim_v144(
        signal_tuples=all_signals, start_bar=ns, end_bar=ne,
        tighten_pct=t, **common
    )
    st = calc_stats(trades)
    phase1_results[f't{t}'] = st
    marker = ' ← current' if t == 75 else ''
    print(f't{t:<9} {st["pnl"]:>+9.1f} {st["mdd"]:>8.2f} {st["pnl_mdd"]:>8.1f} '
          f'{st["wr"]:>7.1f} {st["trades"]:>7}{marker}')


# ============================================================
# Phase 2: Pattern quality filter (with current t75)
# ============================================================
print()
print('=' * 90)
print('PHASE 2: Pattern Quality Filter (tighten_pct=75, v1.44.0 AggRisk 5/15)')
print('=' * 90)

signal_sets = {
    'full_130': all_signals,
    'genuine_marginal_65': gm_signals,
    'genuine_25': genuine_signals,
}

header = f'{"Filter":<25} {"PnL%":>9} {"MDD%":>8} {"P/M":>8} {"WR%":>7} {"Trades":>7} {"Sig/day":>8}'
print(header)
print('-' * 75)

phase2_results = {}
for name, sigs in signal_sets.items():
    trades = portfolio_sim_v144(
        signal_tuples=sigs, start_bar=ns, end_bar=ne,
        tighten_pct=75, **common
    )
    st = calc_stats(trades)
    n_days = (ne - ns) / BARS_PER_DAY
    sig_per_day = len([s for s in sigs if ns <= s[0] < ne]) / n_days
    st['sig_per_day'] = round(sig_per_day, 1)
    phase2_results[name] = st
    marker = ' ← current' if name == 'full_130' else ''
    print(f'{name:<25} {st["pnl"]:>+9.1f} {st["mdd"]:>8.2f} {st["pnl_mdd"]:>8.1f} '
          f'{st["wr"]:>7.1f} {st["trades"]:>7} {sig_per_day:>8.1f}{marker}')


# ============================================================
# Phase 3: Best tighten_pct × pattern filter combination
# ============================================================
print()
print('=' * 90)
print('PHASE 3: Tighten × Filter Grid (top combos)')
print('=' * 90)

# Test promising combos
combos = {}
for t in [70, 75, 80, 85]:
    for fname, fsigs in signal_sets.items():
        label = f't{t}_{fname}'
        trades = portfolio_sim_v144(
            signal_tuples=fsigs, start_bar=ns, end_bar=ne,
            tighten_pct=t, **common
        )
        st = calc_stats(trades)
        combos[label] = st

# Rank by PnL/MDD
ranked = sorted(combos.items(), key=lambda x: x[1]['pnl_mdd'], reverse=True)

header = f'{"Rank":<5} {"Combo":<35} {"PnL%":>9} {"MDD%":>8} {"P/M":>8} {"WR%":>7} {"Trades":>7}'
print(header)
print('-' * 80)
for i, (name, st) in enumerate(ranked[:10], 1):
    marker = ' ← current' if name == 't75_full_130' else ''
    print(f'{i:<5} {name:<35} {st["pnl"]:>+9.1f} {st["mdd"]:>8.2f} {st["pnl_mdd"]:>8.1f} '
          f'{st["wr"]:>7.1f} {st["trades"]:>7}{marker}')

best_combo = ranked[0][0]
print(f'\nBest IS combo: {best_combo} (PnL/MDD {ranked[0][1]["pnl_mdd"]:.1f})')


# ============================================================
# Phase 4: WF validation of top combos
# ============================================================
print()
print('=' * 90)
print('PHASE 4: WF 3-fold Validation')
print('=' * 90)

n_folds = 3
total = ne - ns
seg_size = total // (n_folds + 1)

# Test top 5 combos + current baseline
wf_candidates = [name for name, _ in ranked[:5]]
if 't75_full_130' not in wf_candidates:
    wf_candidates.append('t75_full_130')

wf_results = {}
for label in wf_candidates:
    parts = label.split('_', 1)
    t = int(parts[0][1:])
    fname = parts[1]
    sigs = signal_sets[fname]

    folds = []
    for fold in range(n_folds):
        oos_start = ns + seg_size * (fold + 1)
        oos_end = ns + seg_size * (fold + 2) if fold < n_folds - 1 else ne
        trades_f = portfolio_sim_v144(
            signal_tuples=sigs, start_bar=oos_start, end_bar=oos_end,
            tighten_pct=t, **common
        )
        st_f = calc_stats(trades_f)
        folds.append(st_f)

    n_pass = sum(1 for f in folds if f['pnl'] > 0)
    avg_oos = np.mean([f['pnl'] for f in folds])
    min_oos = min(f['pnl'] for f in folds)
    wf_results[label] = {
        'n_pass': n_pass,
        'avg_oos': round(float(avg_oos), 1),
        'min_oos': round(float(min_oos), 1),
        'folds_pnl': [round(f['pnl'], 1) for f in folds],
    }

print(f'\n{"Combo":<35} {"F1":>8} {"F2":>8} {"F3":>8} {"Avg":>8} {"Min":>8} {"WF":>8}')
print('-' * 85)
for label in wf_candidates:
    r = wf_results[label]
    verdict = f'{r["n_pass"]}/3 {"P" if r["n_pass"]==3 else "F"}'
    marker = ' ← current' if label == 't75_full_130' else ''
    print(f'{label:<35} {r["folds_pnl"][0]:>+8.1f} {r["folds_pnl"][1]:>+8.1f} '
          f'{r["folds_pnl"][2]:>+8.1f} {r["avg_oos"]:>+8.1f} {r["min_oos"]:>+8.1f} {verdict:>8}{marker}')


# ============================================================
# SYNTHESIS
# ============================================================
print()
print('=' * 90)
print('SYNTHESIS')
print('=' * 90)

# Best WF-passing combo
wf_pass = {k: v for k, v in wf_results.items() if v['n_pass'] == 3}
if wf_pass:
    best_wf = max(wf_pass.items(), key=lambda x: combos[x[0]]['pnl_mdd'])
    curr = combos.get('t75_full_130', {})
    best = combos[best_wf[0]]
    print(f'\nCurrent (t75_full_130):')
    print(f'  IS PnL/MDD: {curr.get("pnl_mdd", "?"):.1f}, OOS avg: {wf_results.get("t75_full_130", {}).get("avg_oos", "?")}')
    print(f'\nBest WF-passing ({best_wf[0]}):')
    print(f'  IS PnL/MDD: {best["pnl_mdd"]:.1f}, OOS avg: {best_wf[1]["avg_oos"]}')

    if best_wf[0] != 't75_full_130':
        delta_pm = best['pnl_mdd'] - curr.get('pnl_mdd', 0)
        delta_oos = best_wf[1]['avg_oos'] - wf_results.get('t75_full_130', {}).get('avg_oos', 0)
        print(f'\n  IS PnL/MDD delta: {delta_pm:+.1f}')
        print(f'  OOS avg delta: {delta_oos:+.1f}%')

        if delta_pm > 0 and delta_oos > 0:
            print(f'\n>>> RECOMMEND: Switch to {best_wf[0]}')
        elif delta_pm > 0 and delta_oos < 0:
            print(f'\n>>> HOLD: IS better but OOS worse — overfitting risk')
        else:
            print(f'\n>>> HOLD: Current baseline is competitive')
    else:
        print(f'\n>>> HOLD: Current t75_full_130 is already best')
else:
    print('\n>>> WARNING: No WF 3/3 PASS combos!')


# ============================================================
# Save
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v1.44.0',
    'phase1_tighten_sweep': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                                  for kk, vv in v.items()} for k, v in phase1_results.items()},
    'phase2_pattern_filter': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                                   for kk, vv in v.items()} for k, v in phase2_results.items()},
    'phase3_top_combos': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                               for kk, vv in v.items()} for k, v in combos.items()},
    'phase4_wf': wf_results,
    'ranking': [name for name, _ in ranked],
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved: {OUTPUT_FILE}')
