#!/usr/bin/env python3
"""
New Candidate Bootstrap Relative Test (2026-04-25)
=====================================================
`slip_robust_param_sweep_20260425`가 찾은 두 pass candidate:
  - (trail_K=2.5, max_sl_atr=4.0)  = today's rejected candidate_C
  - (trail_K=2.5, max_sl_atr=5.0)  = NEW, bootstrap relative 미테스트

`candidate_c_full_validation_20260424`가 candidate_C를 `c6: bootstrap relative
P(cand>base) ≥ 55%`에서 21.1%로 기각. Slip-adjusted sweep은 aggregate daily
PnL 기준이라 per-window consistency 미반영.

## 테스트

(2.5, 5.0)이 candidate_C와 같은 per-window consistency 문제를 가지는지 확인.

### Bootstrap relative test
1. (2.5, 5.0) trades + baseline (2.5, 3.3) trades 생성 (full 332d)
2. 3-day 무작위 window 1000개 추출
3. 각 window에서 sum_pnl(cand) > sum_pnl(base) 비율 계산
4. >= 55% = PASS (memory research_protocol_3day_bootstrap)

### Additional slip-adjusted bootstrap relative
5. 각 scenario(MED, HIGH)에서도 bootstrap relative 계산
6. Slip 환경에서 per-window 승률 변화 확인

### Also (2.5, 4.0) 재테스트 (sanity)
Today's candidate_c_full_validation 결과 재현 확인
"""

import sys
import os
import json
import math
import random
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.analysis.slippage_model import (
    apply_slip_to_trades, SCENARIOS
)

FEE_RT_PCT = 0.10
WARMUP_BARS = 50
DATA_CSV = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
OUTPUT_JSON = ROOT / 'results' / 'new_candidate_bootstrap_20260425.json'

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

# Candidates to test
CANDIDATES = [
    ('baseline', 2.5, 3.3),
    ('cand_25_40', 2.5, 4.0),  # today's rejected candidate_C
    ('cand_25_50', 2.5, 5.0),  # NEW
]

WINDOW_DAYS = 3
WINDOW_BARS = 96 * WINDOW_DAYS  # 96 = 15m bars per day
N_BOOTSTRAP = 1000
SEED = 42


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
    """Full BT. Returns list of trade dicts with entry_bar index."""
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
                    'entry_bar':   pos['entry_bar'],
                    'exit_bar':    i,
                    'direction':   pos['direction'],
                    'entry_price': ep,
                    'exit_price':  xp,
                    'sl_price':    pos['sl_price'],
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

    return trades, n


def bootstrap_relative(trades_a, trades_b, n_bars, n_boot=N_BOOTSTRAP,
                        window_bars=WINDOW_BARS, seed=SEED,
                        pnl_key='pnl_pct'):
    """
    For n_boot random windows of window_bars, compute:
      - sum_pnl_a, sum_pnl_b for trades with entry_bar in window
      - P(sum_a > sum_b), P(sum_a == sum_b)
    """
    rng = random.Random(seed)
    start_min = WARMUP_BARS + 10
    start_max = n_bars - window_bars - 1

    cnt_a_wins = 0
    cnt_ties = 0
    diffs = []
    for _ in range(n_boot):
        s = rng.randint(start_min, start_max)
        e = s + window_bars
        sum_a = sum(t[pnl_key] for t in trades_a if s <= t['entry_bar'] < e)
        sum_b = sum(t[pnl_key] for t in trades_b if s <= t['entry_bar'] < e)
        diff = sum_a - sum_b
        diffs.append(diff)
        if diff > 1e-6:
            cnt_a_wins += 1
        elif abs(diff) < 1e-6:
            cnt_ties += 1

    return {
        'n_boot':       n_boot,
        'window_days':  WINDOW_DAYS,
        'p_a_beats_b':  round(cnt_a_wins / n_boot * 100, 2),
        'p_tie':        round(cnt_ties / n_boot * 100, 2),
        'p_b_beats_a':  round((n_boot - cnt_a_wins - cnt_ties) / n_boot * 100, 2),
        'mean_diff':    round(mean(diffs), 4),
        'median_diff':  round(median(diffs), 4),
    }


def main():
    print('=' * 95)
    print('NEW CANDIDATE BOOTSTRAP RELATIVE TEST — 2026-04-25')
    print('=' * 95)
    df15 = load_and_resample(DATA_CSV)
    n_bars = len(df15)
    print(f'{n_bars} bars  |  {df15["timestamp"].iloc[0]} → {df15["timestamp"].iloc[-1]}')
    print()

    # Generate trades for all candidates
    all_trades = {}
    for name, tk, sa in CANDIDATES:
        trades, _ = run_bt(df15, tk, sa)
        total_pnl = sum(t['pnl_pct'] for t in trades)
        print(f'  {name:15s} ({tk}, {sa}): {len(trades)} trades, '
              f'total PnL {total_pnl:+.2f}%')
        all_trades[name] = trades
    print()

    # Bootstrap relative — clean BT (no slip)
    print('─' * 95)
    print('BOOTSTRAP RELATIVE (3-day window, 1000 samples, clean BT)')
    print('─' * 95)

    baseline = all_trades['baseline']
    go_threshold = 55.0  # memory research_protocol_3day_bootstrap

    results = {}
    for name in ['cand_25_40', 'cand_25_50']:
        cand = all_trades[name]
        r = bootstrap_relative(cand, baseline, n_bars)
        results[f'{name}_clean'] = r
        flag = '✅ PASS' if r['p_a_beats_b'] >= go_threshold else '❌ FAIL'
        print(f'  {name:15s}: P(cand > base) = {r["p_a_beats_b"]:.1f}%  '
              f'(tie {r["p_tie"]:.1f}%, base > cand {r["p_b_beats_a"]:.1f}%) '
              f'{flag}')
    print()

    # Sanity check: candidate_c (2.5, 4.0) today's validation said 21.1%.
    # If our result is close, method is consistent.
    print(f'  Note: today candidate_c_full_validation reported 21.1% for '
          f'(4.0, 2.5) — expect similar for cand_25_40 here.')
    print()

    # Bootstrap relative under slip scenarios
    print('─' * 95)
    print('BOOTSTRAP RELATIVE WITH SLIP (MED, HIGH)')
    print('─' * 95)

    for scenario in ['MED', 'HIGH']:
        print(f'  {scenario} scenario:')
        base_slip = apply_slip_to_trades(baseline, scenario=scenario)
        # Copy entry_bar from original (apply_slip preserves it)
        for orig, slipped in zip(baseline, base_slip):
            slipped['entry_bar'] = orig['entry_bar']

        for name in ['cand_25_40', 'cand_25_50']:
            cand = all_trades[name]
            cand_slip = apply_slip_to_trades(cand, scenario=scenario)
            for orig, slipped in zip(cand, cand_slip):
                slipped['entry_bar'] = orig['entry_bar']

            r = bootstrap_relative(cand_slip, base_slip, n_bars,
                                    pnl_key='pnl_pct_slipped')
            results[f'{name}_{scenario.lower()}'] = r
            flag = '✅ PASS' if r['p_a_beats_b'] >= go_threshold else '❌ FAIL'
            print(f'    {name:15s}: P(cand > base) = {r["p_a_beats_b"]:.1f}%  {flag}')
    print()

    # Summary verdict per candidate
    print('─' * 95)
    print('VERDICT SUMMARY')
    print('─' * 95)
    for name in ['cand_25_40', 'cand_25_50']:
        pass_clean = results[f'{name}_clean']['p_a_beats_b'] >= go_threshold
        pass_med = results[f'{name}_med']['p_a_beats_b'] >= go_threshold
        pass_high = results[f'{name}_high']['p_a_beats_b'] >= go_threshold
        passes = sum([pass_clean, pass_med, pass_high])
        verdict = 'STRONG PASS' if passes == 3 else \
                  'PARTIAL PASS' if passes >= 1 else 'FAIL'
        print(f'  {name:15s}: clean={results[f"{name}_clean"]["p_a_beats_b"]:>5.1f}%  '
              f'MED={results[f"{name}_med"]["p_a_beats_b"]:>5.1f}%  '
              f'HIGH={results[f"{name}_high"]["p_a_beats_b"]:>5.1f}%  '
              f'({passes}/3) [{verdict}]')
    print()

    # Save
    out = {
        'timestamp':   datetime.utcnow().isoformat(),
        'n_bootstrap': N_BOOTSTRAP,
        'window_days': WINDOW_DAYS,
        'seed':        SEED,
        'candidates':  CANDIDATES,
        'trade_counts': {name: len(all_trades[name])
                          for name in all_trades},
        'total_pnl_clean': {name: round(sum(t['pnl_pct'] for t in all_trades[name]), 2)
                             for name in all_trades},
        'bootstrap_results': results,
        'go_threshold_pct':  go_threshold,
    }
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'JSON saved → {OUTPUT_JSON.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
