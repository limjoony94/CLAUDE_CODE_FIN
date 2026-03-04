#!/usr/bin/env python3
"""
AggRisk Relaxation Study
===========================

Pattern Genuine Edge Study Phase 5에서 AggRisk 제거 시 IS+OOS 모두 대폭 향상 확인.
→ 완전 제거는 위험 (correlated loss 방어). 파라미터 완화 연구.

현행: counter_cap=3.0%, with_cap=7.0% (v1.35.5)
테스트: counter 3~10%, with 5~OFF, 총 12개 시나리오

평가 기준:
  1. WF 3-fold OOS (3/3 PASS 필수)
  2. IS PnL/MDD
  3. Max daily loss (correlated loss 방어 핵심)
  4. Cascade-OFF에서도 WF PASS 유지 (robustness)

Standard Research Protocol: LEVERAGE=3, FEE×LEV, Timeout DROP, ATR-scaled, Compound

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

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'aggrisk_relaxation_study.json')


# ============================================================
# Custom portfolio_sim with configurable agg_risk caps
# ============================================================
def portfolio_sim_custom(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    agg_counter_cap=3.0, agg_with_cap=7.0,
    agg_risk_enabled=True,
    cascade_enabled=True,
):
    """Portfolio sim with custom agg risk caps."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS
    timeout_bars = TIMEOUT_BARS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    # Daily loss tracking
    daily_pnl = 0.0
    current_day = -1
    max_daily_loss = 0.0
    daily_losses = []

    sig_by_bar = {}
    for s_bar, pat, direction, tp, sl in signal_tuples:
        if start_bar <= s_bar < end_bar:
            sig_by_bar.setdefault(s_bar, []).append((pat, direction, tp, sl))

    for bar in range(start_bar, end_bar):
        if bar >= n_bars - 1:
            break

        # Day tracking
        day = bar // BARS_PER_DAY
        if day != current_day:
            if current_day >= 0:
                daily_losses.append(daily_pnl)
                if daily_pnl < max_daily_loss:
                    max_daily_loss = daily_pnl
            daily_pnl = 0.0
            current_day = day

        closed_slots = set()
        sl_exits = []
        bar_pnl_sum = 0.0

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

            # Timeout
            if hold >= timeout_bars:
                closed_slots.add(pos['slot'])
                continue

            # Early exit
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

            # TP/SL
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
                'direction': direction, 'size_mult': sm,
                'pnl_portfolio': pnl_portfolio, 'leverage': LEVERAGE,
            })
            closed_slots.add(pos['slot'])
            bar_pnl_sum += pnl_portfolio
            daily_pnl += pnl_portfolio

            if reason == 'SL':
                sl_exits.append(pos)

        positions = [p for p in positions if p['slot'] not in closed_slots]

        if cascade_enabled and sl_exits:
            for sl_pos in sl_exits:
                sl_dir = sl_pos['direction']
                for pos in positions:
                    if pos['direction'] == sl_dir:
                        pos['eff_sl_pct'] *= 0.25

        equity *= (1 + bar_pnl_sum / 100)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # Momentum
        if bar >= MOMENTUM_LOOKBACK:
            pa = closes[bar - MOMENTUM_LOOKBACK]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > MOMENTUM_THRESHOLD:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'], bar + MOMENTUM_COOLDOWN)
                elif pct < -MOMENTUM_THRESHOLD:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'], bar + MOMENTUM_COOLDOWN)

        # New entries
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

            # Aggregate risk cap (custom)
            if agg_risk_enabled:
                slope = ema_slope[bar] if bar < len(ema_slope) else 0
                is_uptrend = slope > 0
                is_counter = ((direction == 'SHORT' and is_uptrend) or
                              (direction == 'LONG' and not is_uptrend))
                cap_pct = agg_counter_cap if is_counter else agg_with_cap

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
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
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
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm, 'leverage': LEVERAGE,
        })

    if current_day >= 0:
        daily_losses.append(daily_pnl)
        if daily_pnl < max_daily_loss:
            max_daily_loss = daily_pnl

    return trades, max_daily_loss, daily_losses


# ============================================================
# Main
# ============================================================
print('=' * 90)
print('AGGRISK RELAXATION STUDY')
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

rctypes = df['rctype'].values
n_bars = len(df)
all_signals = []
for i in range(2, n_bars):
    tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
    if tri in pat_lookup:
        p = pat_lookup[tri]
        all_signals.append((i, tri, p['direction'], p['tp'], p['sl']))
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

# ============================================================
# Scenario Grid
# ============================================================
scenarios = {
    'A_baseline_3_7':       {'agg_counter_cap': 3.0, 'agg_with_cap': 7.0},     # current
    'B_relax_5_7':          {'agg_counter_cap': 5.0, 'agg_with_cap': 7.0},
    'C_relax_5_10':         {'agg_counter_cap': 5.0, 'agg_with_cap': 10.0},
    'D_relax_5_15':         {'agg_counter_cap': 5.0, 'agg_with_cap': 15.0},
    'E_relax_7_10':         {'agg_counter_cap': 7.0, 'agg_with_cap': 10.0},
    'F_relax_7_15':         {'agg_counter_cap': 7.0, 'agg_with_cap': 15.0},
    'G_relax_10_15':        {'agg_counter_cap': 10.0, 'agg_with_cap': 15.0},
    'H_counter_only_5':     {'agg_counter_cap': 5.0, 'agg_with_cap': 999.0},   # with uncapped
    'I_counter_only_10':    {'agg_counter_cap': 10.0, 'agg_with_cap': 999.0},
    'J_OFF':                {'agg_risk_enabled': False},
}

# ============================================================
# Phase 1: IS Full
# ============================================================
print()
print('=' * 90)
print('PHASE 1: IS Full Neutral Window')
print('=' * 90)

header = f'{"Scenario":<22} {"PnL%":>9} {"MDD%":>8} {"P/M":>8} {"WR%":>7} {"Trades":>7} {"MaxDayLoss":>11}'
print(header)
print('-' * 75)

is_results = {}
for name, kwargs in scenarios.items():
    trades, max_dl, dailies = portfolio_sim_custom(
        signal_tuples=all_signals, start_bar=ns, end_bar=ne,
        **kwargs, **common
    )
    st = calc_stats(trades)
    st['max_daily_loss'] = round(max_dl, 2)
    is_results[name] = st
    print(f'{name:<22} {st["pnl"]:>+9.1f} {st["mdd"]:>8.2f} {st["pnl_mdd"]:>8.1f} '
          f'{st["wr"]:>7.1f} {st["trades"]:>7} {st["max_daily_loss"]:>+11.2f}')

# ============================================================
# Phase 2: WF 3-fold
# ============================================================
print()
print('=' * 90)
print('PHASE 2: WF 3-fold Expanding Window (OOS only)')
print('=' * 90)

n_folds = 3
total = ne - ns
seg_size = total // (n_folds + 1)

wf_results = {}
for name, kwargs in scenarios.items():
    folds = []
    for fold in range(n_folds):
        oos_start = ns + seg_size * (fold + 1)
        oos_end = ns + seg_size * (fold + 2) if fold < n_folds - 1 else ne
        trades_f, _, _ = portfolio_sim_custom(
            signal_tuples=all_signals, start_bar=oos_start, end_bar=oos_end,
            **kwargs, **common
        )
        stats_f = calc_stats(trades_f)
        folds.append(stats_f)
    n_pass = sum(1 for f in folds if f['pnl'] > 0)
    avg_oos = np.mean([f['pnl'] for f in folds])
    min_oos = min(f['pnl'] for f in folds)
    wf_results[name] = {
        'n_pass': n_pass,
        'avg_oos': round(float(avg_oos), 1),
        'min_oos': round(float(min_oos), 1),
        'folds': folds,
    }

print(f'\n{"Scenario":<22} {"F1":>8} {"F2":>8} {"F3":>8} {"Avg":>8} {"Min":>8} {"WF":>8}')
print('-' * 65)
for name in scenarios:
    r = wf_results[name]
    f = r['folds']
    verdict = f'{r["n_pass"]}/3 {"P" if r["n_pass"]==3 else "F"}'
    print(f'{name:<22} {f[0]["pnl"]:>+8.1f} {f[1]["pnl"]:>+8.1f} {f[2]["pnl"]:>+8.1f} '
          f'{r["avg_oos"]:>+8.1f} {r["min_oos"]:>+8.1f} {verdict:>8}')

# ============================================================
# Phase 3: Decision Matrix
# ============================================================
print()
print('=' * 90)
print('PHASE 3: Decision Matrix')
print('=' * 90)

# Rank: WF 3/3 first, then PnL/MDD, then max_daily_loss (less negative = better)
ranked = sorted(scenarios.keys(), key=lambda n: (
    wf_results[n]['n_pass'] == 3,
    is_results[n]['pnl_mdd'],
    is_results[n].get('max_daily_loss', -999),
), reverse=True)

print(f'\n{"Rank":<5} {"Scenario":<22} {"IS P/M":>8} {"IS PnL":>9} {"IS MDD":>8} '
      f'{"MaxDL":>8} {"OOS Avg":>9} {"OOS Min":>9} {"WF":>6}')
print('-' * 90)
for i, name in enumerate(ranked, 1):
    isr = is_results[name]
    wfr = wf_results[name]
    verdict = f'{wfr["n_pass"]}/3'
    print(f'{i:<5} {name:<22} {isr["pnl_mdd"]:>8.1f} {isr["pnl"]:>+9.1f} {isr["mdd"]:>8.2f} '
          f'{isr["max_daily_loss"]:>+8.2f} {wfr["avg_oos"]:>+9.1f} {wfr["min_oos"]:>+9.1f} {verdict:>6}')

# Best WF passing with acceptable daily loss
wf_pass_configs = [n for n in ranked if wf_results[n]['n_pass'] == 3]
baseline_max_dl = is_results['A_baseline_3_7']['max_daily_loss']
safe_configs = [n for n in wf_pass_configs
                if is_results[n]['max_daily_loss'] >= baseline_max_dl * 1.5]  # max 50% worse

print(f'\nBaseline max daily loss: {baseline_max_dl:+.2f}%')
print(f'WF 3/3 PASS configs: {len(wf_pass_configs)}/{len(scenarios)}')
print(f'Safe configs (max daily loss <= 1.5x baseline): {len(safe_configs)}')

if safe_configs:
    best = safe_configs[0]
    print(f'\n>>> RECOMMENDED: {best}')
    print(f'    IS PnL/MDD: {is_results[best]["pnl_mdd"]:.1f} '
          f'(vs baseline {is_results["A_baseline_3_7"]["pnl_mdd"]:.1f})')
    print(f'    Max daily loss: {is_results[best]["max_daily_loss"]:+.2f}% '
          f'(vs baseline {baseline_max_dl:+.2f}%)')
    print(f'    OOS avg: {wf_results[best]["avg_oos"]:+.1f}% '
          f'(vs baseline {wf_results["A_baseline_3_7"]["avg_oos"]:+.1f}%)')
else:
    print('\n>>> WARNING: No safe relaxation found. Keep current 3/7.')

# ============================================================
# Save
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v1.42.0',
    'is_results': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                        for kk, vv in v.items()} for k, v in is_results.items()},
    'wf_results': {k: {'n_pass': v['n_pass'], 'avg_oos': v['avg_oos'], 'min_oos': v['min_oos'],
                        'folds_pnl': [f['pnl'] for f in v['folds']]}
                   for k, v in wf_results.items()},
    'ranking': ranked,
    'recommended': safe_configs[0] if safe_configs else None,
    'baseline_max_daily_loss': baseline_max_dl,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved: {OUTPUT_FILE}')
