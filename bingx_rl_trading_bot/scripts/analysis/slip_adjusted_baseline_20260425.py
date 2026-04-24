#!/usr/bin/env python3
"""
Slip-Adjusted Baseline Analysis (2026-04-25)
================================================
C1 Breakout v2 baseline 전구간 BT 실행 후, 5개 slip 시나리오 적용하여
daily PnL, WR, R:R, MDD 변화 정량화.

## 목적

향후 모든 파라미터 연구의 **slip-aware reference point** 확립.
"Candidate가 BT에서 daily +0.55%면 LIVE에서 얼마?"라는 질문에 답 가능.

## 재사용 가능한 출력

`results/slip_adjusted_baseline_20260425.json`에 5개 시나리오 각각의 metrics 저장.
향후 candidate 평가 시 "baseline-relative 기준"을 slip-adjusted로 쉽게 계산 가능.

## Scenarios

- ZERO: no slip (theoretical BT)
- LOW:  ~10th percentile optimistic LIVE
- MED:  pre-F-v2 typical LIVE (median observed)
- HIGH: 90th percentile conservative (F v2 MARKET close expectation)
- STRESS: F v2 first-trade observed (0.64%)
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
C1_CONFIG = {
    'channel_period': 15,
    'body_min_ratio': 0.4,
    'atr_period': 14,
    'trail_K': 2.5,
    'max_sl_atr': 3.3,
    'emergency_sl_pct': 3.0,
    'max_hold_bars': 192,
    'sl_min_pct': 0.15,
    'sl_max_pct': 3.0,
    'min_bars_between': 2,
    'trail_activation_pct': 0.05,
    'fractal_lookback': 10,
    'progressive_trail': {'enabled': False},  # clean baseline (no progressive)
}

FEE_RT_PCT = 0.10
WARMUP_BARS = 50
DATA_CSV = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
OUTPUT_JSON = ROOT / 'results' / 'slip_adjusted_baseline_20260425.json'


def load_and_resample(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp')
    df15 = df.resample('15min').agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    return df15


def run_c1_baseline(df15):
    """Full-period C1 baseline BT. Returns trade list."""
    signal = C1BreakoutSignal(C1_CONFIG)

    opens  = df15['open'].tolist()
    highs  = df15['high'].tolist()
    lows   = df15['low'].tolist()
    closes = df15['close'].tolist()
    ts     = df15['timestamp'].tolist()
    n = len(closes)

    atr = compute_atr(highs, lows, closes, C1_CONFIG['atr_period'])
    ch_high, ch_low = compute_channel(highs, lows, C1_CONFIG['channel_period'])
    sw_low, sw_high = compute_fractal_swings(highs, lows,
                                              C1_CONFIG['fractal_lookback'])

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
                cooldown_until = i + 1 + C1_CONFIG['min_bars_between']
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

    return trades, ts[WARMUP_BARS], ts[-1]


def main():
    print('=' * 90)
    print('SLIP-ADJUSTED BASELINE ANALYSIS — 2026-04-25')
    print('=' * 90)

    df15 = load_and_resample(DATA_CSV)
    t_start = df15['timestamp'].iloc[WARMUP_BARS]
    t_end = df15['timestamp'].iloc[-1]
    days = (t_end - t_start).total_seconds() / 86400
    print(f'Loaded {len(df15)} 15m bars  |  {t_start} → {t_end}  ({days:.1f} days)')
    print()

    print('Running C1 baseline BT (full period, progressive=off)...')
    trades, t0, t1 = run_c1_baseline(df15)
    print(f'  Total trades: {len(trades)}')

    # Reason breakdown
    reason_cnt = {}
    for t in trades:
        reason_cnt[t['reason']] = reason_cnt.get(t['reason'], 0) + 1
    print(f'  Reason breakdown: {reason_cnt}')
    print()

    # Apply all scenarios
    print('─' * 90)
    print(f'{"Scenario":<10} {"Total":>8} {"Daily":>8} {"WR":>6} {"R:R":>6} '
          f'{"MDD":>6} {"Δ_zero":>8} {"slip_cost":>10}')
    print('─' * 90)

    results = compare_scenarios(trades, fee_rt_pct=FEE_RT_PCT)
    zero_daily = results['ZERO']['total_pnl'] / days

    scenario_records = {}
    for sc in ['ZERO', 'LOW', 'MED', 'HIGH', 'STRESS']:
        r = results[sc]
        daily = r['total_pnl'] / days if days > 0 else 0
        delta_zero = daily - zero_daily
        slip_cost_daily = zero_daily - daily  # how much slip ate
        slip_cost_pct = (slip_cost_daily / zero_daily * 100) if zero_daily > 0 else 0

        print(f'{sc:<10} {r["total_pnl"]:>+8.2f} {daily:>+8.4f} '
              f'{r["WR"]:>5.1f}% {r["RR"] or 0:>5.2f} '
              f'{r["MDD"]:>5.2f} {delta_zero:>+8.4f} '
              f'{slip_cost_pct:>9.1f}%')

        scenario_records[sc] = {
            'total_pnl':       r['total_pnl'],
            'daily_pnl':       round(daily, 4),
            'WR':              r['WR'],
            'RR':              r['RR'],
            'MDD':             r['MDD'],
            'avg_win':         r.get('avg_win'),
            'avg_loss':        r.get('avg_loss'),
            'delta_vs_zero':   round(delta_zero, 4),
            'slip_cost_daily': round(slip_cost_daily, 4),
            'slip_cost_pct':   round(slip_cost_pct, 1),
        }
    print('─' * 90)
    print()

    # Interpretation
    print('─' * 90)
    print('INTERPRETATION')
    print('─' * 90)
    med_daily = scenario_records['MED']['daily_pnl']
    high_daily = scenario_records['HIGH']['daily_pnl']
    stress_daily = scenario_records['STRESS']['daily_pnl']

    print(f'  - Clean BT (ZERO):     {zero_daily:+.4f}%/day  ← current reference')
    print(f'  - Pre-F v2 LIVE (MED): {med_daily:+.4f}%/day  ← expected LIVE under MED slip')
    print(f'  - F v2 LIVE (HIGH):    {high_daily:+.4f}%/day  ← expected F v2 with MARKET close slip')
    print(f'  - Stress (STRESS):     {stress_daily:+.4f}%/day  ← worst-case 0.64% F v2 trail slip')
    print()
    print(f'  Slip cost MED:   {zero_daily - med_daily:.4f}%/day   '
          f'({scenario_records["MED"]["slip_cost_pct"]:.1f}% of gross)')
    print(f'  Slip cost HIGH:  {zero_daily - high_daily:.4f}%/day   '
          f'({scenario_records["HIGH"]["slip_cost_pct"]:.1f}% of gross)')
    print(f'  Slip cost STRESS:{zero_daily - stress_daily:.4f}%/day   '
          f'({scenario_records["STRESS"]["slip_cost_pct"]:.1f}% of gross)')
    print()

    # Guideline for future candidate evaluation
    print('─' * 90)
    print('FUTURE PDCA GO GATE REFERENCE (baseline-relative delta)')
    print('─' * 90)
    print('  Candidate이 PRODUCTION-WORTH 하려면:')
    print(f'  - Scenario ZERO:  candidate daily > {zero_daily:+.4f}%')
    print(f'  - Scenario MED:   candidate daily > {med_daily:+.4f}%  ← typical LIVE reference')
    print(f'  - Scenario HIGH:  candidate daily > {high_daily:+.4f}%  ← F v2 conservative')
    print(f'  - Stability: delta consistent across scenarios (not just ZERO)')
    print()

    # Save
    out = {
        'timestamp':     datetime.utcnow().isoformat(),
        'data_window':   {'start': str(t_start), 'end': str(t_end),
                           'days':  round(days, 1)},
        'config':        C1_CONFIG,
        'total_trades':  len(trades),
        'reason_counts': reason_cnt,
        'scenarios':     scenario_records,
        'slip_scenarios_def': SCENARIOS,
    }
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'JSON saved → {OUTPUT_JSON.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
