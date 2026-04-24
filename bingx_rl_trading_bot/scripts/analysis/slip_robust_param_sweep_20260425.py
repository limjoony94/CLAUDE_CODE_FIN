#!/usr/bin/env python3
"""
Slip-Robust Parameter Sweep (2026-04-25)
==========================================
`slippage_model.py`와 `slip_adjusted_baseline_20260425`의 발견(F v2 slip risk)에
대응. Production param (trail_K=2.5, max_sl_atr=3.3)이 slip-adjusted 기준에서도
pareto-optimal인가? 더 slip-robust한 config이 있나?

## 가설

**H1**: 더 넓은 trail_K (3.0+)는 TRAIL_TP 빈도↓ → 총 trail slip cost↓
     → MED/HIGH scenario에서 수익 더 보존

**H2**: max_sl_atr 4.0 (candidate_C 값)은 slip-adjusted view에서 재평가 필요
     — 기존 bootstrap relative 21.1% 기각이 slip 무시 기준이었음

**H3**: tighter trail_K (1.5~2.0)는 slip 민감도↑ (작은 winner가 slip에 쉽게 먹힘)

## Grid

- trail_K ∈ {1.5, 2.0, 2.5, 3.0, 3.5}
- max_sl_atr ∈ {2.5, 3.0, 3.3, 4.0, 5.0}
- 5×5 = 25 configs
- 각 config를 5 slip scenario에 걸쳐 평가

## GO gate (9-flag baseline-relative)

모든 scenario에서 **current production(2.5, 3.3)을 beat**:
- ZERO: candidate_daily > +0.514
- MED:  candidate_daily > +0.078
- HIGH: candidate_daily > -0.570

+ stability flags: 3-way 전부 양수 (train_not_degraded 포함)

## Output

- 25 configs × 5 scenarios 매트릭스
- Top 3 slip-robust config 상세
- PDCA 승격 후보 식별
- `results/slip_robust_param_sweep_20260425.json`
"""

import sys
import os
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.analysis.slippage_model import (
    apply_slip_to_trades, compare_scenarios, SCENARIOS
)

# ═══════════════════════════════════════════════════════════════════════
FEE_RT_PCT = 0.10
WARMUP_BARS = 50
DATA_CSV = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
OUTPUT_JSON = ROOT / 'results' / 'slip_robust_param_sweep_20260425.json'

# Grid
TRAIL_K_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]
MAX_SL_ATR_GRID = [2.5, 3.0, 3.3, 4.0, 5.0]

PRODUCTION_TRAIL_K = 2.5
PRODUCTION_MAX_SL_ATR = 3.3

BASE_CONFIG = {
    'channel_period': 15,
    'body_min_ratio': 0.4,
    'atr_period': 14,
    'emergency_sl_pct': 3.0,
    'max_hold_bars': 192,
    'sl_min_pct': 0.15,
    'sl_max_pct': 3.0,
    'min_bars_between': 2,
    'trail_activation_pct': 0.05,
    'fractal_lookback': 10,
    'progressive_trail': {'enabled': False},
}


def load_and_resample(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp')
    df15 = df.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    return df15


def run_bt(df15, trail_k, max_sl_atr):
    config = {**BASE_CONFIG, 'trail_K': trail_k, 'max_sl_atr': max_sl_atr}
    signal = C1BreakoutSignal(config)

    opens  = df15['open'].tolist()
    highs  = df15['high'].tolist()
    lows   = df15['low'].tolist()
    closes = df15['close'].tolist()
    ts     = df15['timestamp'].tolist()
    n = len(closes)

    atr = compute_atr(highs, lows, closes, config['atr_period'])
    ch_high, ch_low = compute_channel(highs, lows, config['channel_period'])
    sw_low, sw_high = compute_fractal_swings(highs, lows,
                                              config['fractal_lookback'])

    trades = []
    in_pos = False
    pos = None
    cooldown_until = WARMUP_BARS

    for i in range(WARMUP_BARS, n):
        if in_pos:
            pos['bars_held'] += 1
            if pos['direction'] == 'LONG':
                pos['best_price'] = max(pos['best_price'], highs[i])
            else:
                pos['best_price'] = min(pos['best_price'], lows[i])
            atr_now = atr[i] if not math.isnan(atr[i]) else atr[i - 1]
            exit_result = signal.check_exit(
                direction=pos['direction'],
                entry_price=pos['entry_price'],
                best_price=pos['best_price'],
                current_high=highs[i],
                current_low=lows[i],
                current_close=closes[i],
                sl_price=pos['sl_price'],
                atr_val=atr_now,
                bars_held=pos['bars_held'],
            )
            if exit_result is not None:
                ep = pos['entry_price']
                xp = exit_result['exit_price']
                if pos['direction'] == 'LONG':
                    pnl_raw = (xp / ep - 1) * 100
                else:
                    pnl_raw = (1 - xp / ep) * 100
                pnl_net = pnl_raw - FEE_RT_PCT
                trades.append({
                    'direction':   pos['direction'],
                    'entry_time':  str(ts[pos['entry_bar']]),
                    'entry_price': round(ep, 2),
                    'exit_time':   str(ts[i]),
                    'exit_price':  round(xp, 2),
                    'sl_price':    round(pos['sl_price'], 2),
                    'reason':      exit_result['reason'],
                    'pnl_pct':     round(pnl_net, 4),
                    'bars_held':   pos['bars_held'],
                })
                in_pos = False
                pos = None
                cooldown_until = i + 1 + config['min_bars_between']
                continue

        if not in_pos and i >= cooldown_until and i + 1 < n:
            if math.isnan(atr[i]) or math.isnan(ch_high[i]):
                continue
            sig = signal.check_entry(
                bar_open=opens[i], bar_high=highs[i],
                bar_low=lows[i], bar_close=closes[i],
                channel_high=ch_high[i], channel_low=ch_low[i],
                atr_val=atr[i],
                last_swing_low=sw_low[i], last_swing_high=sw_high[i],
            )
            if sig is not None and i + 1 < n:
                pos = {
                    'direction':   sig['direction'],
                    'entry_price': opens[i + 1],
                    'sl_price':    sig['sl_price'],
                    'entry_bar':   i + 1,
                    'bars_held':   0,
                }
                if pos['direction'] == 'LONG':
                    pos['best_price'] = max(opens[i + 1], highs[i + 1])
                else:
                    pos['best_price'] = min(opens[i + 1], lows[i + 1])
                in_pos = True

    return trades


def three_way(trades):
    if len(trades) < 30:
        return None
    n = len(trades)
    third = n // 3
    parts = [trades[0:third], trades[third:2*third], trades[2*third:]]
    pnls = [sum(t['pnl_pct'] for t in p) for p in parts]
    return {
        'train': round(pnls[0], 2),
        'val':   round(pnls[1], 2),
        'test':  round(pnls[2], 2),
        'all_positive': all(p > 0 for p in pnls),
    }


def main():
    print('=' * 100)
    print('SLIP-ROBUST PARAMETER SWEEP — 2026-04-25')
    print('=' * 100)
    df15 = load_and_resample(DATA_CSV)
    t_start = df15['timestamp'].iloc[WARMUP_BARS]
    t_end = df15['timestamp'].iloc[-1]
    days = (t_end - t_start).total_seconds() / 86400
    print(f'{len(df15)} 15m bars  |  {days:.1f} days')
    print(f'Grid: trail_K {TRAIL_K_GRID}  ×  max_sl_atr {MAX_SL_ATR_GRID}')
    print(f'→ {len(TRAIL_K_GRID) * len(MAX_SL_ATR_GRID)} configs × 5 scenarios')
    print()

    # Reference: current production under each scenario
    print('Running baseline (2.5, 3.3)...')
    baseline_trades = run_bt(df15, PRODUCTION_TRAIL_K, PRODUCTION_MAX_SL_ATR)
    baseline_cmp = compare_scenarios(baseline_trades)
    baseline_daily = {sc: baseline_cmp[sc]['total_pnl'] / days
                        for sc in SCENARIOS}
    print(f'  trades={len(baseline_trades)}  '
          f'ZERO={baseline_daily["ZERO"]:+.4f}  MED={baseline_daily["MED"]:+.4f}  '
          f'HIGH={baseline_daily["HIGH"]:+.4f}')
    print()

    # Sweep
    all_results = []
    print('─' * 100)
    print(f'{"trail_K":>7} {"sl_atr":>7} {"trades":>7} '
          f'{"ZERO":>8} {"LOW":>8} {"MED":>8} {"HIGH":>8} {"STRESS":>8} '
          f'{"3-way":>10}')
    print('─' * 100)

    for tk in TRAIL_K_GRID:
        for sa in MAX_SL_ATR_GRID:
            trades = run_bt(df15, tk, sa)
            cmp = compare_scenarios(trades)
            dailies = {sc: cmp[sc]['total_pnl'] / days for sc in SCENARIOS}
            split = three_way(trades)

            # GO gate check
            beats_zero = dailies['ZERO'] > baseline_daily['ZERO']
            beats_med  = dailies['MED']  > baseline_daily['MED']
            beats_high = dailies['HIGH'] > baseline_daily['HIGH']
            all_3way_pos = split['all_positive'] if split else False

            config_key = f'({tk}, {sa})'
            is_prod = (tk == PRODUCTION_TRAIL_K and sa == PRODUCTION_MAX_SL_ATR)
            marker = '*' if is_prod else ' '

            passed = beats_zero + beats_med + beats_high + all_3way_pos

            result = {
                'trail_K': tk, 'max_sl_atr': sa,
                'trades': len(trades),
                'ZERO':  round(dailies['ZERO'], 4),
                'LOW':   round(dailies['LOW'], 4),
                'MED':   round(dailies['MED'], 4),
                'HIGH':  round(dailies['HIGH'], 4),
                'STRESS': round(dailies['STRESS'], 4),
                'three_way':     split,
                'beats_baseline': {
                    'ZERO':    beats_zero,
                    'MED':     beats_med,
                    'HIGH':    beats_high,
                    '3way_all_pos': all_3way_pos,
                },
                'passed_flags':  passed,
                'is_production': is_prod,
            }
            all_results.append(result)

            split_s = (f'T={split["train"]:+.0f}/V={split["val"]:+.0f}/'
                       f'T={split["test"]:+.0f}' if split else '—')
            print(f'{marker}{tk:>6} {sa:>7} {len(trades):>7} '
                  f'{dailies["ZERO"]:>+8.4f} '
                  f'{dailies["LOW"]:>+8.4f} '
                  f'{dailies["MED"]:>+8.4f} '
                  f'{dailies["HIGH"]:>+8.4f} '
                  f'{dailies["STRESS"]:>+8.4f} '
                  f'{split_s:>10} '
                  f'{"[PASS]" if passed == 4 else f"({passed}/4)"}')
    print('─' * 100)
    print('(* = current production)')
    print()

    # Best by scenario
    print('─' * 100)
    print('BEST CONFIG PER SCENARIO')
    print('─' * 100)
    for sc in ['ZERO', 'MED', 'HIGH']:
        best = max(all_results, key=lambda r: r[sc])
        if best['is_production']:
            print(f'  {sc:>6}: production (2.5, 3.3) @ {best[sc]:+.4f}/day  ← 현재 최적')
        else:
            print(f'  {sc:>6}: ({best["trail_K"]}, {best["max_sl_atr"]}) @ '
                  f'{best[sc]:+.4f}/day  [vs prod {baseline_daily[sc]:+.4f}, '
                  f'Δ {best[sc]-baseline_daily[sc]:+.4f}]')
    print()

    # GO gate pass
    print('─' * 100)
    print('GO GATE PASS (all 4 flags: beats_baseline ZERO+MED+HIGH + 3way_all_pos)')
    print('─' * 100)
    passes = [r for r in all_results if r['passed_flags'] == 4 and not r['is_production']]
    if not passes:
        print('  ❌ 0 configs 통과 — production (2.5, 3.3)이 slip-aware pareto-optimal')
    else:
        passes.sort(key=lambda r: r['MED'], reverse=True)
        for r in passes:
            print(f'  ✅ ({r["trail_K"]}, {r["max_sl_atr"]}) — '
                  f'ZERO {r["ZERO"]:+.4f}, MED {r["MED"]:+.4f}, HIGH {r["HIGH"]:+.4f}')
    print()

    # Save
    out = {
        'timestamp': datetime.utcnow().isoformat(),
        'data_window': {'start': str(t_start), 'end': str(t_end),
                          'days': round(days, 1)},
        'grid': {
            'trail_K': TRAIL_K_GRID,
            'max_sl_atr': MAX_SL_ATR_GRID,
        },
        'production_config': {
            'trail_K': PRODUCTION_TRAIL_K,
            'max_sl_atr': PRODUCTION_MAX_SL_ATR,
        },
        'baseline_daily': baseline_daily,
        'results':        all_results,
        'go_gate_passed': [
            {'trail_K': r['trail_K'], 'max_sl_atr': r['max_sl_atr'],
             'ZERO': r['ZERO'], 'MED': r['MED'], 'HIGH': r['HIGH']}
            for r in all_results
            if r['passed_flags'] == 4 and not r['is_production']
        ],
    }
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'JSON saved → {OUTPUT_JSON.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
