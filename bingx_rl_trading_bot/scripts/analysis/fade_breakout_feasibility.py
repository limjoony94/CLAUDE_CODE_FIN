#!/usr/bin/env python3
"""
Fade Breakout Feasibility BT (Phase 1)
========================================
Inverted entry vs C1 Breakout — "fade the breakout" hypothesis:

  - On C1 breakout signal at bar i, place LIMIT in OPPOSITE direction at
    channel_high × 1.005  (LONG breakout → SHORT limit, 0.5% above channel)
    channel_low  × 0.995  (SHORT breakout → LONG limit, 0.5% below channel)
  - Limit waits up to 16 bars; cancel if price re-enters channel before fill
  - Once filled:
      Entry price = limit trigger (fixed)
      SL  = channel_high × 1.015 (SHORT) or channel_low × 0.985 (LONG)  → 1.0% beyond entry
      TP  = channel_high (SHORT) or channel_low (LONG)                   → 0.5% profit from entry
      Timeout = 96 bars (24h) exit at market close
  - Same-bar SL+TP ambiguity: CONSERVATIVE → SL wins (worst case)
  - Same-bar fill + exit: check SL/TP on fill bar, SL wins if both touchable
  - Cooldown 2 bars after any exit
  - N=1 (no pyramiding)
  - Fee 0.10% RT additive
  - PnL = additive (not compound)

Reference — C1 baseline (v2.5, 333d, additive 1x):
  daily +0.509%  |  WR 36.6%  |  R:R 3.36  |  MDD 5.4%
  trades/day 3.1  |  total PnL +169.5%  |  trades 1028

GO gate (Phase 1 feasibility):
  G1: daily PnL > 0
  G2: WR >= 33%
  G3: MDD <= 10%
  G4: trades/day >= 1
  G5: train_not_degraded (3-way split all positive)

If ALL 5 PASS → proceed to Phase 2 (Exit variants sweep + overfit guards)
If ANY FAIL  → document failure mode, align with MR-path lessons

Output: results/fade_breakout_feasibility.json + console summary

2026-04-24 | Phase 1 feasibility
"""

import sys
import os
import json
import math
from datetime import datetime
from pathlib import Path

# Project root
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
}

# Fade strategy parameters
EXTENSION_PCT     = 0.005   # 0.5% beyond channel → limit trigger
SL_OFFSET_PCT     = 0.015   # 1.5% beyond channel → SL (= 1.0% beyond entry)
LIMIT_TTL_BARS    = 16      # limit cancels after this many bars
TIMEOUT_BARS      = 96      # 24h at 15m
COOLDOWN_BARS     = 2       # bars between exit and next signal consideration

FEE_RT_PCT        = 0.10    # 0.10% round-trip additive
WARMUP_BARS       = 50
DATA_CSV          = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
OUTPUT_JSON       = ROOT / 'results' / 'fade_breakout_feasibility.json'

# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_and_resample(csv_path: Path) -> pd.DataFrame:
    """Load 5m CSV and resample to 15m."""
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
# Fade BT engine
# ═══════════════════════════════════════════════════════════════════════

def check_exit_in_bar(pos, bar_high, bar_low, bar_close, bars_held):
    """Check if an OPEN position exits within this bar.

    Priority (conservative):
      1. SL (if bar range touches SL level) — worst case for fade thesis
      2. TP (only if SL NOT touchable)
      3. TIMEOUT (if bars_held >= TIMEOUT_BARS)

    Returns {'exit_price', 'reason'} or None (hold).
    """
    if pos['direction'] == 'SHORT':
        sl_hit = bar_high >= pos['sl_price']
        tp_hit = bar_low  <= pos['tp_price']
    else:  # LONG
        sl_hit = bar_low  <= pos['sl_price']
        tp_hit = bar_high >= pos['tp_price']

    if sl_hit:
        return {'exit_price': pos['sl_price'], 'reason': 'SL'}
    if tp_hit:
        return {'exit_price': pos['tp_price'], 'reason': 'TP'}
    if bars_held >= TIMEOUT_BARS:
        return {'exit_price': bar_close, 'reason': 'TIMEOUT'}
    return None


def run_fade_backtest(df15: pd.DataFrame):
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
    limit_events = {'placed': 0, 'filled': 0, 'cancelled_reentry': 0,
                    'cancelled_ttl': 0}

    # State machine
    state = 'IDLE'          # IDLE | PENDING | OPEN
    pending = None          # dict when PENDING
    position = None         # dict when OPEN
    cooldown_until = WARMUP_BARS

    for i in range(WARMUP_BARS, n):
        # ── STATE: OPEN ── check exit
        if state == 'OPEN':
            position['bars_held'] += 1
            exit_result = check_exit_in_bar(
                position, highs[i], lows[i], closes[i], position['bars_held']
            )
            if exit_result is not None:
                ep = position['entry_price']
                xp = exit_result['exit_price']
                if position['direction'] == 'SHORT':
                    pnl_raw = (1 - xp / ep) * 100
                else:
                    pnl_raw = (xp / ep - 1) * 100
                pnl_net = pnl_raw - FEE_RT_PCT

                trades.append({
                    'signal_bar':     position['signal_bar'],
                    'signal_time':    str(ts[position['signal_bar']]),
                    'c1_direction':   position['c1_direction'],
                    'fade_direction': position['direction'],
                    'channel_level':  round(position['channel_level'], 2),
                    'entry_bar':      position['entry_bar'],
                    'entry_time':     str(ts[position['entry_bar']]),
                    'entry_price':    round(ep, 2),
                    'exit_bar':       i,
                    'exit_time':      str(ts[i]),
                    'exit_price':     round(xp, 2),
                    'sl_price':       round(position['sl_price'], 2),
                    'tp_price':       round(position['tp_price'], 2),
                    'reason':         exit_result['reason'],
                    'pnl_pct':        round(pnl_net, 4),
                    'pnl_raw':        round(pnl_raw, 4),
                    'bars_waited':    position['bars_waited'],
                    'bars_held':      position['bars_held'],
                })
                state = 'IDLE'
                position = None
                cooldown_until = i + 1 + COOLDOWN_BARS
                # Do NOT process new signal on same bar as exit
                continue

        # ── STATE: PENDING ── check fill or cancel
        if state == 'PENDING':
            pending['bars_waited'] += 1
            fill_happened = False
            cancel_reason = None

            # Check fill first (limit order touch)
            if pending['direction'] == 'SHORT':
                # fading LONG breakout — SHORT limit at channel_high × 1.005
                if highs[i] >= pending['trigger']:
                    fill_happened = True
            else:  # LONG limit, fading SHORT breakout
                if lows[i] <= pending['trigger']:
                    fill_happened = True

            if fill_happened:
                limit_events['filled'] += 1
                entry_price = pending['trigger']  # limit fills at exactly trigger
                position = {
                    'direction':     pending['direction'],
                    'c1_direction':  pending['c1_direction'],
                    'channel_level': pending['channel_level'],
                    'entry_price':   entry_price,
                    'sl_price':      pending['sl_price'],
                    'tp_price':      pending['tp_price'],
                    'entry_bar':     i,
                    'signal_bar':    pending['signal_bar'],
                    'bars_waited':   pending['bars_waited'],
                    'bars_held':     0,
                }
                state = 'OPEN'
                pending = None

                # Same-bar exit check (conservative SL-first)
                exit_result = check_exit_in_bar(
                    position, highs[i], lows[i], closes[i], 0
                )
                if exit_result is not None:
                    ep = position['entry_price']
                    xp = exit_result['exit_price']
                    if position['direction'] == 'SHORT':
                        pnl_raw = (1 - xp / ep) * 100
                    else:
                        pnl_raw = (xp / ep - 1) * 100
                    pnl_net = pnl_raw - FEE_RT_PCT

                    trades.append({
                        'signal_bar':     position['signal_bar'],
                        'signal_time':    str(ts[position['signal_bar']]),
                        'c1_direction':   position['c1_direction'],
                        'fade_direction': position['direction'],
                        'channel_level':  round(position['channel_level'], 2),
                        'entry_bar':      position['entry_bar'],
                        'entry_time':     str(ts[position['entry_bar']]),
                        'entry_price':    round(ep, 2),
                        'exit_bar':       i,
                        'exit_time':      str(ts[i]),
                        'exit_price':     round(xp, 2),
                        'sl_price':       round(position['sl_price'], 2),
                        'tp_price':       round(position['tp_price'], 2),
                        'reason':         exit_result['reason'] + '_SAMEBAR',
                        'pnl_pct':        round(pnl_net, 4),
                        'pnl_raw':        round(pnl_raw, 4),
                        'bars_waited':    position['bars_waited'],
                        'bars_held':      0,
                    })
                    state = 'IDLE'
                    position = None
                    cooldown_until = i + 1 + COOLDOWN_BARS
                continue

            # Check cancel — price re-enters channel before fill
            if pending['direction'] == 'SHORT':
                # LONG breakout; cancel if bar_low dips back below channel_high
                if lows[i] < pending['channel_level']:
                    cancel_reason = 'reentry'
            else:
                if highs[i] > pending['channel_level']:
                    cancel_reason = 'reentry'

            # TTL cancel
            if cancel_reason is None and pending['bars_waited'] >= LIMIT_TTL_BARS:
                cancel_reason = 'ttl'

            if cancel_reason is not None:
                if cancel_reason == 'reentry':
                    limit_events['cancelled_reentry'] += 1
                else:
                    limit_events['cancelled_ttl'] += 1
                state = 'IDLE'
                pending = None
                cooldown_until = i + 1  # minimal cooldown after cancel
                continue

        # ── STATE: IDLE ── check for new signal
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

            c1_dir = entry_signal['direction']
            # Flip to fade
            if c1_dir == 'LONG':
                # fade → SHORT limit at channel_high × 1.005
                fade_dir = 'SHORT'
                level = ch_high[i]
                trigger = level * (1 + EXTENSION_PCT)
                sl      = level * (1 + SL_OFFSET_PCT)
                tp      = level
            else:
                fade_dir = 'LONG'
                level = ch_low[i]
                trigger = level * (1 - EXTENSION_PCT)
                sl      = level * (1 - SL_OFFSET_PCT)
                tp      = level

            pending = {
                'direction':     fade_dir,
                'c1_direction':  c1_dir,
                'channel_level': level,
                'trigger':       trigger,
                'sl_price':      sl,
                'tp_price':      tp,
                'signal_bar':    i,
                'bars_waited':   0,
            }
            state = 'PENDING'
            limit_events['placed'] += 1

    return trades, limit_events, ts[WARMUP_BARS], ts[-1]


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_mdd(trades):
    """Max drawdown on cumulative additive PnL (1x, net of fee)."""
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for t in trades:
        cum += t['pnl_pct']
        peak = max(peak, cum)
        dd = peak - cum
        mdd = max(mdd, dd)
    return mdd


def summarize(trades, limit_events, t_start, t_end):
    days = (pd.Timestamp(t_end) - pd.Timestamp(t_start)).total_seconds() / 86400

    if not trades:
        return {
            'trades': 0, 'days': round(days, 1),
            'daily_pnl': 0.0, 'total_pnl_1x': 0.0, 'WR': 0.0,
            'RR': 0.0, 'MDD_1x': 0.0, 'trades_per_day': 0.0,
            'limit_events': limit_events,
            'note': 'no trades — limit never filled',
        }

    total_pnl = sum(t['pnl_pct'] for t in trades)
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0

    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t['pnl_pct'] for t in losses) / len(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else float('inf')

    mdd = compute_mdd(trades)
    daily = total_pnl / max(days, 1)
    tpd = len(trades) / max(days, 1)

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1

    # Direction split
    dirs = {'LONG_fade': 0, 'SHORT_fade': 0}
    for t in trades:
        if t['fade_direction'] == 'LONG':
            dirs['LONG_fade'] += 1
        else:
            dirs['SHORT_fade'] += 1

    # Bars waited stats
    waits = [t['bars_waited'] for t in trades]
    holds = [t['bars_held'] for t in trades]

    return {
        'trades':           len(trades),
        'days':             round(days, 1),
        'daily_pnl':        round(daily, 4),
        'total_pnl_1x':     round(total_pnl, 2),
        'WR':               round(wr, 2),
        'RR':               round(rr, 3) if rr != float('inf') else None,
        'avg_win':          round(avg_win, 4),
        'avg_loss':         round(avg_loss, 4),
        'MDD_1x':           round(mdd, 2),
        'trades_per_day':   round(tpd, 2),
        'exit_reasons':     reasons,
        'direction_split':  dirs,
        'bars_waited_avg':  round(sum(waits) / len(waits), 1),
        'bars_held_avg':    round(sum(holds) / len(holds), 1),
        'bars_waited_max':  max(waits),
        'bars_held_max':    max(holds),
        'limit_events':     limit_events,
    }


def three_way_split_check(trades):
    """Split trades into 3 chronological thirds and compute PnL in each."""
    if len(trades) < 30:
        return {'valid': False, 'note': 'too few trades'}
    n = len(trades)
    third = n // 3
    parts = [
        trades[0:third],
        trades[third:2 * third],
        trades[2 * third:],
    ]
    pnls = [sum(t['pnl_pct'] for t in p) for p in parts]
    wr3 = [
        (sum(1 for t in p if t['pnl_pct'] > 0) / len(p) * 100) if p else 0
        for p in parts
    ]
    return {
        'valid': True,
        'train_pnl':     round(pnls[0], 2),
        'val_pnl':       round(pnls[1], 2),
        'test_pnl':      round(pnls[2], 2),
        'train_wr':      round(wr3[0], 1),
        'val_wr':        round(wr3[1], 1),
        'test_wr':       round(wr3[2], 1),
        'all_positive':  all(p > 0 for p in pnls),
        'train_not_degraded': pnls[2] >= 0.5 * pnls[0] if pnls[0] > 0 else pnls[2] > 0,
    }


def evaluate_go_gate(summary, split):
    g1 = summary['daily_pnl'] > 0
    g2 = summary['WR'] >= 33
    g3 = summary['MDD_1x'] <= 10
    g4 = summary['trades_per_day'] >= 1
    g5 = split.get('all_positive', False) if split.get('valid') else False

    passed = sum([g1, g2, g3, g4, g5])
    return {
        'G1_daily_pnl>0':           g1,
        'G2_WR>=33':                g2,
        'G3_MDD<=10':               g3,
        'G4_trades_per_day>=1':     g4,
        'G5_3way_all_positive':     g5,
        'total_passed':             f'{passed}/5',
        'verdict':                  'PASS' if passed == 5 else 'FAIL',
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print('=' * 78)
    print('FADE BREAKOUT FEASIBILITY BT — Phase 1')
    print('=' * 78)
    print(f'Data: {DATA_CSV.name}')
    print(f'Params: ext=+/-{EXTENSION_PCT*100:.2f}%  sl=+/-{SL_OFFSET_PCT*100:.2f}%  '
          f'ttl={LIMIT_TTL_BARS}  timeout={TIMEOUT_BARS}  fee={FEE_RT_PCT}% RT')
    print()

    df15 = load_and_resample(DATA_CSV)
    print(f'Loaded {len(df15)} 15m bars  |  '
          f'{df15["timestamp"].iloc[0]} → {df15["timestamp"].iloc[-1]}')
    print()

    trades, limit_events, t_start, t_end = run_fade_backtest(df15)
    summary = summarize(trades, limit_events, t_start, t_end)
    split = three_way_split_check(trades)
    gate = evaluate_go_gate(summary, split)

    # ── Console report ──
    print('─' * 78)
    print('LIMIT ORDER EVENTS')
    print('─' * 78)
    for k, v in limit_events.items():
        print(f'  {k:25s}: {v}')
    print()

    print('─' * 78)
    print('SUMMARY (additive 1x, net of fee)')
    print('─' * 78)
    for k, v in summary.items():
        if k in ('exit_reasons', 'direction_split', 'limit_events'):
            continue
        print(f'  {k:25s}: {v}')
    print(f'  exit_reasons             : {summary["exit_reasons"]}')
    print(f'  direction_split          : {summary["direction_split"]}')
    print()

    print('─' * 78)
    print('3-WAY SPLIT (train / val / test)')
    print('─' * 78)
    for k, v in split.items():
        print(f'  {k:25s}: {v}')
    print()

    print('─' * 78)
    print('GO GATE EVALUATION')
    print('─' * 78)
    for k, v in gate.items():
        print(f'  {k:30s}: {v}')
    print()

    print('─' * 78)
    print('COMPARISON vs C1 Baseline (v2.5, 333d, additive 1x)')
    print('─' * 78)
    c1 = {
        'daily_pnl':      0.509,
        'total_pnl_1x':   169.5,
        'WR':             36.6,
        'RR':             3.36,
        'MDD_1x':         5.4,
        'trades_per_day': 3.1,
    }
    for k in c1:
        fade_v = summary.get(k, 0)
        c1_v = c1[k]
        if fade_v is None:
            fade_v = 0
        delta = fade_v - c1_v
        sign = '+' if delta >= 0 else ''
        print(f'  {k:20s}: fade={fade_v:>10}  c1={c1_v:>10}  Δ={sign}{delta:.3f}')
    print()

    # ── Save JSON ──
    out = {
        'strategy':       'fade_breakout_v1',
        'phase':          1,
        'timestamp':      datetime.utcnow().isoformat(),
        'params': {
            'extension_pct':  EXTENSION_PCT,
            'sl_offset_pct':  SL_OFFSET_PCT,
            'limit_ttl_bars': LIMIT_TTL_BARS,
            'timeout_bars':   TIMEOUT_BARS,
            'cooldown_bars':  COOLDOWN_BARS,
            'fee_rt_pct':     FEE_RT_PCT,
            'warmup_bars':    WARMUP_BARS,
        },
        'summary':        summary,
        'three_way_split': split,
        'go_gate':        gate,
        'c1_baseline':    c1,
        'trades_count':   len(trades),
        'data_window':    {'start': str(t_start), 'end': str(t_end)},
    }
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'JSON saved → {OUTPUT_JSON.relative_to(ROOT)}')

    # Also save trade-level detail for later inspection
    trades_json = OUTPUT_JSON.with_name('fade_breakout_feasibility_trades.json')
    with open(trades_json, 'w') as f:
        json.dump(trades, f, indent=2, default=str)
    print(f'Trades saved → {trades_json.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
