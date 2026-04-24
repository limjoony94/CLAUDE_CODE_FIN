#!/usr/bin/env python3
"""
Pullback Entry Feasibility BT (Phase 1A)
==========================================
C1 breakout 방향을 유지하면서, 돌파 후 즉시 진입 대신 N% pullback 대기 후 진입.

  Entry flow:
    1. C1.check_entry() 신호 발생 (direction + sl_price)
    2. 즉시 진입 대신 LIMIT 설정:
         LONG  → close × (1 - pullback_pct)
         SHORT → close × (1 + pullback_pct)
    3. TTL 내 price가 limit 터치 시 → FILL at trigger
    4. TTL 경과 시 → CANCEL
  Exit flow:
    - Fill 후 C1.check_exit() 그대로 사용 (fractal SL + ATR trail + timeout)
    - 단, entry_price/best_price는 NEW entry 기준

  Safety filters (signal 단계):
    - trigger가 channel 반대편이면 skip (LONG: trigger ≤ channel_high)
    - trigger가 SL 반대편이면 skip (LONG: trigger ≤ sl_price)

  Sweep: pullback_pct ∈ {0.1, 0.2, 0.3, 0.4, 0.5} × ttl_bars ∈ {4, 8, 16}
  = 15 configs, each ~10-15s. Total ~3분.

Reference — C1 baseline (v2.5, 333d, additive 1x):
  daily +0.509%  WR 36.6%  R:R 3.36  MDD 5.4%  trades/day 3.1  PnL +169.5%

GO gate per config:
  G1: daily PnL > 0
  G2: WR >= 30 (lower bar — pullback filter may shift WR)
  G3: MDD <= baseline × 2 (10.8%)
  G4: trades/day >= 1
  G5: 3-way split all positive (best config only)

Output:
  - console: full sweep matrix + best config detail + 3-way
  - results/pullback_entry_feasibility.json

2026-04-24 | Phase 1A
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

# ═══════════════════════════════════════════════════════════════════════
# Constants
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
    # Progressive trail disabled for feasibility (clean comparison with C1 v2.5)
    'progressive_trail': {'enabled': False},
}

FEE_RT_PCT = 0.10
WARMUP_BARS = 50
COOLDOWN_BARS = 2
DATA_CSV = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
OUTPUT_JSON = ROOT / 'results' / 'pullback_entry_feasibility.json'

# Sweep grid
PULLBACK_GRID = [0.001, 0.002, 0.003, 0.004, 0.005]
TTL_GRID = [4, 8, 16]

# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Pullback-entry BT engine
# ═══════════════════════════════════════════════════════════════════════

def run_pullback_backtest(df15, pullback_pct, ttl_bars):
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
    events = {'signals':         0,
              'placed':          0,
              'skipped_chan':    0,   # trigger in channel
              'skipped_sl':      0,   # trigger past SL
              'filled':          0,
              'cancelled_ttl':   0}

    state = 'IDLE'
    pending = None
    position = None
    cooldown_until = WARMUP_BARS

    for i in range(WARMUP_BARS, n):
        # ── OPEN: check C1 exit ──
        if state == 'OPEN':
            position['bars_held'] += 1
            if position['direction'] == 'LONG':
                position['best_price'] = max(position['best_price'], highs[i])
            else:
                position['best_price'] = min(position['best_price'], lows[i])

            atr_now = atr[i] if not math.isnan(atr[i]) else atr[i - 1]
            exit_result = signal.check_exit(
                direction=position['direction'],
                entry_price=position['entry_price'],
                best_price=position['best_price'],
                current_high=highs[i],
                current_low=lows[i],
                current_close=closes[i],
                sl_price=position['sl_price'],
                atr_val=atr_now,
                bars_held=position['bars_held'],
            )

            if exit_result is not None:
                ep = position['entry_price']
                xp = exit_result['exit_price']
                if position['direction'] == 'LONG':
                    pnl_raw = (xp / ep - 1) * 100
                else:
                    pnl_raw = (1 - xp / ep) * 100
                pnl_net = pnl_raw - FEE_RT_PCT

                trades.append({
                    'signal_bar':  position['signal_bar'],
                    'direction':   position['direction'],
                    'entry_bar':   position['entry_bar'],
                    'entry_time':  str(ts[position['entry_bar']]),
                    'entry_price': round(ep, 2),
                    'exit_bar':    i,
                    'exit_time':   str(ts[i]),
                    'exit_price':  round(xp, 2),
                    'sl_price':    round(position['sl_price'], 2),
                    'reason':      exit_result['reason'],
                    'pnl_pct':     round(pnl_net, 4),
                    'pnl_raw':     round(pnl_raw, 4),
                    'bars_waited': position['bars_waited'],
                    'bars_held':   position['bars_held'],
                    'sig_close':   round(position['sig_close'], 2),
                })
                state = 'IDLE'
                position = None
                cooldown_until = i + 1 + COOLDOWN_BARS
                continue

        # ── PENDING: check fill or TTL ──
        if state == 'PENDING':
            pending['bars_waited'] += 1

            # Fill check
            fill = False
            if pending['direction'] == 'LONG':
                if lows[i] <= pending['trigger']:
                    fill = True
            else:
                if highs[i] >= pending['trigger']:
                    fill = True

            if fill:
                events['filled'] += 1
                entry_price = pending['trigger']  # limit fill

                position = {
                    'direction':   pending['direction'],
                    'entry_price': entry_price,
                    'sl_price':    pending['sl_price'],
                    'sig_close':   pending['sig_close'],
                    'entry_bar':   i,
                    'signal_bar':  pending['signal_bar'],
                    'bars_waited': pending['bars_waited'],
                    'bars_held':   0,
                    'best_price':  entry_price,  # will update below
                }
                # Initialize best_price from current bar extremes (intrabar)
                if position['direction'] == 'LONG':
                    position['best_price'] = max(entry_price, highs[i])
                else:
                    position['best_price'] = min(entry_price, lows[i])

                state = 'OPEN'
                pending = None

                # Same-bar exit check (C1 logic, bars_held=0)
                atr_now = atr[i] if not math.isnan(atr[i]) else atr[i - 1]
                exit_result = signal.check_exit(
                    direction=position['direction'],
                    entry_price=position['entry_price'],
                    best_price=position['best_price'],
                    current_high=highs[i],
                    current_low=lows[i],
                    current_close=closes[i],
                    sl_price=position['sl_price'],
                    atr_val=atr_now,
                    bars_held=0,
                )
                if exit_result is not None:
                    ep = position['entry_price']
                    xp = exit_result['exit_price']
                    if position['direction'] == 'LONG':
                        pnl_raw = (xp / ep - 1) * 100
                    else:
                        pnl_raw = (1 - xp / ep) * 100
                    pnl_net = pnl_raw - FEE_RT_PCT

                    trades.append({
                        'signal_bar':  position['signal_bar'],
                        'direction':   position['direction'],
                        'entry_bar':   position['entry_bar'],
                        'entry_time':  str(ts[position['entry_bar']]),
                        'entry_price': round(ep, 2),
                        'exit_bar':    i,
                        'exit_time':   str(ts[i]),
                        'exit_price':  round(xp, 2),
                        'sl_price':    round(position['sl_price'], 2),
                        'reason':      exit_result['reason'] + '_SAMEBAR',
                        'pnl_pct':     round(pnl_net, 4),
                        'pnl_raw':     round(pnl_raw, 4),
                        'bars_waited': position['bars_waited'],
                        'bars_held':   0,
                        'sig_close':   round(position['sig_close'], 2),
                    })
                    state = 'IDLE'
                    position = None
                    cooldown_until = i + 1 + COOLDOWN_BARS
                continue

            # TTL cancel
            if pending['bars_waited'] >= ttl_bars:
                events['cancelled_ttl'] += 1
                state = 'IDLE'
                pending = None
                cooldown_until = i + 1  # minimal

        # ── IDLE: look for new signal ──
        if state == 'IDLE' and i >= cooldown_until and i + 1 < n:
            if math.isnan(atr[i]) or math.isnan(ch_high[i]) or math.isnan(ch_low[i]):
                continue
            entry_signal = signal.check_entry(
                bar_open=opens[i], bar_high=highs[i],
                bar_low=lows[i], bar_close=closes[i],
                channel_high=ch_high[i], channel_low=ch_low[i],
                atr_val=atr[i],
                last_swing_low=sw_low[i], last_swing_high=sw_high[i],
            )
            if entry_signal is None:
                continue

            events['signals'] += 1
            direction = entry_signal['direction']
            sl_price = entry_signal['sl_price']
            sig_close = closes[i]

            # Compute pullback trigger
            if direction == 'LONG':
                trigger = sig_close * (1 - pullback_pct)
                # Safety: trigger must stay above channel_high
                if trigger <= ch_high[i]:
                    events['skipped_chan'] += 1
                    continue
                # Safety: trigger must stay above SL
                if trigger <= sl_price:
                    events['skipped_sl'] += 1
                    continue
            else:
                trigger = sig_close * (1 + pullback_pct)
                if trigger >= ch_low[i]:
                    events['skipped_chan'] += 1
                    continue
                if trigger >= sl_price:
                    events['skipped_sl'] += 1
                    continue

            pending = {
                'direction':   direction,
                'trigger':     trigger,
                'sl_price':    sl_price,
                'sig_close':   sig_close,
                'signal_bar':  i,
                'bars_waited': 0,
            }
            state = 'PENDING'
            events['placed'] += 1

    return trades, events


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_mdd(trades):
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for t in trades:
        cum += t['pnl_pct']
        peak = max(peak, cum)
        dd = peak - cum
        mdd = max(mdd, dd)
    return mdd


def summarize(trades, events, days):
    if not trades:
        return {
            'trades': 0, 'daily_pnl': 0.0, 'total_pnl_1x': 0.0,
            'WR': 0.0, 'RR': 0.0, 'MDD_1x': 0.0, 'trades_per_day': 0.0,
            'events': events,
        }
    total_pnl = sum(t['pnl_pct'] for t in trades)
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    wr = len(wins) / len(trades) * 100
    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t['pnl_pct'] for t in losses) / len(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
    mdd = compute_mdd(trades)
    daily = total_pnl / max(days, 1)
    tpd = len(trades) / max(days, 1)

    reasons = {}
    for t in trades:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1

    return {
        'trades':         len(trades),
        'daily_pnl':      round(daily, 4),
        'total_pnl_1x':   round(total_pnl, 2),
        'WR':             round(wr, 2),
        'RR':             round(rr, 3) if rr != float('inf') else 999.0,
        'avg_win':        round(avg_win, 4),
        'avg_loss':       round(avg_loss, 4),
        'MDD_1x':         round(mdd, 2),
        'trades_per_day': round(tpd, 2),
        'exit_reasons':   reasons,
        'events':         events,
    }


def three_way_split(trades, days):
    if len(trades) < 30:
        return {'valid': False}
    n = len(trades)
    third = n // 3
    parts = [trades[0:third], trades[third:2*third], trades[2*third:]]
    pnls = [sum(t['pnl_pct'] for t in p) for p in parts]
    wrs = [sum(1 for t in p if t['pnl_pct'] > 0) / len(p) * 100 if p else 0
           for p in parts]
    return {
        'valid':          True,
        'train_pnl':      round(pnls[0], 2),
        'val_pnl':        round(pnls[1], 2),
        'test_pnl':       round(pnls[2], 2),
        'train_wr':       round(wrs[0], 1),
        'val_wr':         round(wrs[1], 1),
        'test_wr':        round(wrs[2], 1),
        'all_positive':   all(p > 0 for p in pnls),
        'train_not_degraded': (
            pnls[2] >= 0.5 * pnls[0] if pnls[0] > 0 else pnls[2] > 0
        ),
    }


def evaluate_go_gate(summary, split):
    g1 = summary['daily_pnl'] > 0
    g2 = summary['WR'] >= 30
    g3 = summary['MDD_1x'] <= 10.8  # C1 baseline × 2
    g4 = summary['trades_per_day'] >= 1
    g5 = split.get('all_positive', False) if split.get('valid') else False
    passed = sum([g1, g2, g3, g4, g5])
    return {
        'G1_daily>0':       g1,
        'G2_WR>=30':        g2,
        'G3_MDD<=10.8':     g3,
        'G4_tpd>=1':        g4,
        'G5_3way_positive': g5,
        'passed':           f'{passed}/5',
        'verdict':          'PASS' if passed == 5 else 'FAIL',
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print('=' * 90)
    print('PULLBACK ENTRY FEASIBILITY — Phase 1A')
    print('=' * 90)
    print(f'Data: {DATA_CSV.name}')
    print(f'Grid: pullback_pct ∈ {[f"{p*100:.1f}%" for p in PULLBACK_GRID]}  '
          f'× TTL ∈ {TTL_GRID}  = {len(PULLBACK_GRID)*len(TTL_GRID)} configs')
    print()

    df15 = load_and_resample(DATA_CSV)
    t_start = df15['timestamp'].iloc[WARMUP_BARS]
    t_end = df15['timestamp'].iloc[-1]
    days = (t_end - t_start).total_seconds() / 86400
    print(f'Loaded {len(df15)} 15m bars  |  {t_start} → {t_end}  ({days:.1f} days)')
    print()

    # Run sweep
    all_results = []
    print('─' * 90)
    print(f'{"pullback":>10} {"TTL":>5} | {"trades":>7} {"tpd":>5} {"daily":>7} '
          f'{"WR":>6} {"R:R":>6} {"MDD":>6} {"fill%":>6} | verdict')
    print('─' * 90)

    for p in PULLBACK_GRID:
        for ttl in TTL_GRID:
            trades, events = run_pullback_backtest(df15, p, ttl)
            summary = summarize(trades, events, days)
            fill_rate = (events['filled'] / max(events['placed'], 1)) * 100

            config = {'pullback_pct': p, 'ttl_bars': ttl,
                      'fill_rate': round(fill_rate, 1)}
            result = {**config, 'summary': summary,
                      'events': events, 'trades': trades}
            all_results.append(result)

            # Quick verdict row (skip 3-way for perf)
            rr_s = f'{summary["RR"]:.2f}' if summary['RR'] < 10 else '>10'
            vstring = '?'
            if summary['daily_pnl'] > 0 and summary['WR'] >= 30 \
                    and summary['MDD_1x'] <= 10.8 and summary['trades_per_day'] >= 1:
                vstring = '4/5?'
            else:
                fails = []
                if not summary['daily_pnl'] > 0: fails.append('d')
                if not summary['WR'] >= 30: fails.append('W')
                if not summary['MDD_1x'] <= 10.8: fails.append('M')
                if not summary['trades_per_day'] >= 1: fails.append('t')
                vstring = 'FAIL:' + ''.join(fails)

            print(f'{p*100:>9.1f}% {ttl:>5} | '
                  f'{summary["trades"]:>7} '
                  f'{summary["trades_per_day"]:>5.2f} '
                  f'{summary["daily_pnl"]:>+7.3f} '
                  f'{summary["WR"]:>5.1f}% '
                  f'{rr_s:>6} '
                  f'{summary["MDD_1x"]:>5.1f}% '
                  f'{fill_rate:>5.1f}% | {vstring}')
    print('─' * 90)
    print()

    # Find best by daily PnL among configs passing core gates (G1, G3, G4)
    viable = [r for r in all_results
              if r['summary']['daily_pnl'] > 0
              and r['summary']['MDD_1x'] <= 10.8
              and r['summary']['trades_per_day'] >= 1]

    if not viable:
        print('❌ NO viable config — all 15 configs FAIL core gates.')
        best = max(all_results, key=lambda r: r['summary']['daily_pnl'])
        print(f'   Best-of-failed: pullback={best["pullback_pct"]*100:.1f}% '
              f'ttl={best["ttl_bars"]} daily={best["summary"]["daily_pnl"]:+.3f}%')
    else:
        viable.sort(key=lambda r: r['summary']['daily_pnl'], reverse=True)
        best = viable[0]
        print(f'✅ {len(viable)}/15 configs viable. BEST config:')
        print(f'   pullback={best["pullback_pct"]*100:.1f}% ttl={best["ttl_bars"]}')

    # Detail on best config
    print()
    print('─' * 90)
    print(f'BEST CONFIG DETAIL — pullback {best["pullback_pct"]*100:.1f}%, '
          f'TTL {best["ttl_bars"]} bars')
    print('─' * 90)
    s = best['summary']
    for k, v in s.items():
        if k in ('exit_reasons', 'events'):
            continue
        print(f'  {k:20s}: {v}')
    print(f'  exit_reasons        : {s["exit_reasons"]}')
    print(f'  events              : {s["events"]}')
    print()

    split = three_way_split(best['trades'], days)
    print('─' * 90)
    print('3-WAY SPLIT (best config)')
    print('─' * 90)
    for k, v in split.items():
        print(f'  {k:25s}: {v}')
    print()

    gate = evaluate_go_gate(s, split)
    print('─' * 90)
    print('GO GATE (best config)')
    print('─' * 90)
    for k, v in gate.items():
        print(f'  {k:25s}: {v}')
    print()

    # Comparison vs C1
    print('─' * 90)
    print('COMPARISON vs C1 Baseline (v2.5, 333d, additive 1x)')
    print('─' * 90)
    c1 = {'daily_pnl': 0.509, 'total_pnl_1x': 169.5, 'WR': 36.6,
          'RR': 3.36, 'MDD_1x': 5.4, 'trades_per_day': 3.1}
    for k in c1:
        pb_v = s.get(k, 0) or 0
        c1_v = c1[k]
        delta = pb_v - c1_v
        sign = '+' if delta >= 0 else ''
        print(f'  {k:20s}: pullback={pb_v:>10}  c1={c1_v:>10}  Δ={sign}{delta:.3f}')
    print()

    # ── Save JSON ──
    # Strip trade-level detail from JSON for readability — save separately
    lean_results = []
    for r in all_results:
        lean_results.append({
            'pullback_pct': r['pullback_pct'],
            'ttl_bars':     r['ttl_bars'],
            'fill_rate':    r['fill_rate'],
            'summary':      r['summary'],
        })

    out = {
        'strategy':  'pullback_entry_v1',
        'phase':     '1A',
        'timestamp': datetime.utcnow().isoformat(),
        'data_window': {'start': str(t_start), 'end': str(t_end),
                         'days': round(days, 1)},
        'sweep_grid': {'pullback_pct': PULLBACK_GRID, 'ttl_bars': TTL_GRID},
        'sweep_results': lean_results,
        'best_config': {
            'pullback_pct': best['pullback_pct'],
            'ttl_bars':     best['ttl_bars'],
            'summary':      best['summary'],
            'three_way':    split,
            'go_gate':      gate,
        },
        'c1_baseline': c1,
    }
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'JSON saved → {OUTPUT_JSON.relative_to(ROOT)}')

    # Best trades for later inspection
    best_trades_json = OUTPUT_JSON.with_name('pullback_entry_best_trades.json')
    with open(best_trades_json, 'w') as f:
        json.dump(best['trades'], f, indent=2, default=str)
    print(f'Best trades → {best_trades_json.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
