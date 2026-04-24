#!/usr/bin/env python3
"""
Pullback Entry Ablation + Neighborhood (Phase 1B)
===================================================
Phase 1A에서 pullback 0.1% / TTL 4가 5/5 GO gate PASS였으나
절대 daily PnL(+0.443%)은 C1 baseline(+0.509%)보다 낮음.

핵심 질문: 0.1% 결과가 진짜 "entry timing edge"인지,
아니면 "skip-on-channel-violation" 내재 필터 효과인지?

## Ablation (4-way)

| Test | min_margin | pullback | invalid | ttl_expiry | 의미 |
|------|-----------|----------|---------|------------|-----|
| T0   | 0         | 0        | —       | —          | C1 pure baseline |
| T1   | 0.001     | 0        | —       | —          | Filter-only (close > ch × 1.001) |
| T2   | 0         | 0.001    | skip    | skip       | Phase 1A replicate (skip-mode) |
| T3   | 0         | 0.001    | immed   | immed      | Timing-only with full fallback |

Decision tree:
  - If T1 ≈ T2 → filter가 주 driver, pullback timing 허상 → 기각
  - If T3 > T0 (meaningful) → 진짜 timing edge 존재 → 추가 탐구
  - If T2 > max(T1, T3) → 필터+타이밍 synergy
  - If ALL similar → 노이즈, 기각

## Neighborhood (overfit check)

Pullback ∈ {0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25}% × TTL=4, skip-mode.
0.1% peak이 isolated spike(overfit)인지 smooth plateau인지 판정.

Output: console sweep + results/pullback_entry_ablation.json
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
    'progressive_trail': {'enabled': False},
}

FEE_RT_PCT = 0.10
WARMUP_BARS = 50
COOLDOWN_BARS = 2
DATA_CSV = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
OUTPUT_JSON = ROOT / 'results' / 'pullback_entry_ablation.json'


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
# Parameterized BT engine
# ═══════════════════════════════════════════════════════════════════════

def run_variant(df15,
                min_margin_pct=0.0,
                pullback_pct=0.0,
                ttl_bars=0,
                on_invalid='skip',     # 'skip' | 'immediate'
                on_ttl='skip'):        # 'skip' | 'immediate'
    """
    Unified entry-variant BT.

    min_margin_pct > 0:
        Require close > channel_high × (1 + margin) for LONG
              close < channel_low  × (1 - margin) for SHORT
        Applied BEFORE pullback logic.

    pullback_pct > 0:
        LONG:  limit at close × (1 - pullback_pct)
        SHORT: limit at close × (1 + pullback_pct)
        Require trigger on correct side of channel AND valid vs SL.

    on_invalid:
        'skip'      — trigger violates channel/SL → skip signal entirely
        'immediate' — trigger violates → enter at next-bar-open instead (C1-style)

    on_ttl:
        'skip'      — TTL expires without fill → cancel, no trade
        'immediate' — TTL expires → enter at the expiry bar's close as fallback
    """
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
    events = {
        'signals':           0,
        'filter_rejected':   0,   # min_margin_pct filter
        'pullback_placed':   0,
        'pullback_filled':   0,
        'pullback_ttl_skip': 0,
        'pullback_ttl_fb':   0,   # TTL fallback entered
        'invalid_skip':      0,
        'invalid_fb':        0,   # invalid fallback entered
        'immediate':         0,   # pullback_pct=0 direct entries
    }

    state = 'IDLE'
    pending = None
    position = None
    cooldown_until = WARMUP_BARS

    def finalize_trade(pos, exit_result, exit_bar):
        ep = pos['entry_price']
        xp = exit_result['exit_price']
        if pos['direction'] == 'LONG':
            pnl_raw = (xp / ep - 1) * 100
        else:
            pnl_raw = (1 - xp / ep) * 100
        pnl_net = pnl_raw - FEE_RT_PCT
        trades.append({
            'signal_bar':  pos['signal_bar'],
            'direction':   pos['direction'],
            'entry_bar':   pos['entry_bar'],
            'entry_time':  str(ts[pos['entry_bar']]),
            'entry_price': round(ep, 2),
            'exit_bar':    exit_bar,
            'exit_time':   str(ts[exit_bar]),
            'exit_price':  round(xp, 2),
            'sl_price':    round(pos['sl_price'], 2),
            'reason':      exit_result['reason'],
            'pnl_pct':     round(pnl_net, 4),
            'pnl_raw':     round(pnl_raw, 4),
            'bars_waited': pos.get('bars_waited', 0),
            'bars_held':   pos['bars_held'],
            'entry_mode':  pos.get('entry_mode', 'unknown'),
        })

    def open_position_at(i, direction, entry_price, sl_price, signal_bar,
                          bars_waited, entry_mode):
        """Open position at bar i with given entry_price, then check same-bar exit."""
        nonlocal state, position, cooldown_until
        position = {
            'direction':   direction,
            'entry_price': entry_price,
            'sl_price':    sl_price,
            'signal_bar':  signal_bar,
            'entry_bar':   i,
            'bars_waited': bars_waited,
            'bars_held':   0,
            'entry_mode':  entry_mode,
        }
        if direction == 'LONG':
            position['best_price'] = max(entry_price, highs[i])
        else:
            position['best_price'] = min(entry_price, lows[i])
        state = 'OPEN'

        atr_now = atr[i] if not math.isnan(atr[i]) else atr[i - 1]
        exit_result = signal.check_exit(
            direction=direction,
            entry_price=entry_price,
            best_price=position['best_price'],
            current_high=highs[i],
            current_low=lows[i],
            current_close=closes[i],
            sl_price=sl_price,
            atr_val=atr_now,
            bars_held=0,
        )
        if exit_result is not None:
            reason = exit_result['reason']
            if '_SAMEBAR' not in reason:
                reason = reason + '_SAMEBAR'
            exit_result = {**exit_result, 'reason': reason}
            finalize_trade(position, exit_result, i)
            state = 'IDLE'
            position = None
            cooldown_until = i + 1 + COOLDOWN_BARS
            return True
        return False

    for i in range(WARMUP_BARS, n):
        # ── OPEN ──
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
                finalize_trade(position, exit_result, i)
                state = 'IDLE'
                position = None
                cooldown_until = i + 1 + COOLDOWN_BARS
                continue

        # ── PENDING (only when pullback_pct > 0) ──
        if state == 'PENDING':
            pending['bars_waited'] += 1
            fill = False
            if pending['direction'] == 'LONG':
                if lows[i] <= pending['trigger']:
                    fill = True
            else:
                if highs[i] >= pending['trigger']:
                    fill = True

            if fill:
                events['pullback_filled'] += 1
                open_position_at(
                    i, pending['direction'], pending['trigger'],
                    pending['sl_price'], pending['signal_bar'],
                    pending['bars_waited'], entry_mode='pullback_fill',
                )
                pending = None
                if state == 'OPEN':
                    continue
                # same-bar exit already handled
                continue

            # TTL expired
            if pending['bars_waited'] >= ttl_bars:
                if on_ttl == 'immediate':
                    events['pullback_ttl_fb'] += 1
                    # Enter at current bar close
                    entry_price = closes[i]
                    # Validate entry vs SL
                    valid = ((pending['direction'] == 'LONG' and entry_price > pending['sl_price']) or
                             (pending['direction'] == 'SHORT' and entry_price < pending['sl_price']))
                    if valid:
                        open_position_at(
                            i, pending['direction'], entry_price,
                            pending['sl_price'], pending['signal_bar'],
                            pending['bars_waited'], entry_mode='ttl_fallback',
                        )
                        pending = None
                        continue
                    # else fall through to skip
                events['pullback_ttl_skip'] += 1
                state = 'IDLE'
                pending = None
                cooldown_until = i + 1

        # ── IDLE: look for signal ──
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

            # Apply min_margin filter
            if min_margin_pct > 0:
                if direction == 'LONG':
                    needed = ch_high[i] * (1 + min_margin_pct)
                    if sig_close <= needed:
                        events['filter_rejected'] += 1
                        continue
                else:
                    needed = ch_low[i] * (1 - min_margin_pct)
                    if sig_close >= needed:
                        events['filter_rejected'] += 1
                        continue

            # No pullback → immediate entry at next bar open (C1 baseline behavior)
            if pullback_pct == 0:
                if i + 1 >= n:
                    continue
                entry_price = opens[i + 1]
                valid = ((direction == 'LONG' and entry_price > sl_price) or
                         (direction == 'SHORT' and entry_price < sl_price))
                if not valid:
                    continue
                events['immediate'] += 1
                # Advance to next bar
                open_position_at(
                    i + 1, direction, entry_price, sl_price,
                    signal_bar=i, bars_waited=0, entry_mode='immediate',
                )
                continue

            # Pullback logic
            if direction == 'LONG':
                trigger = sig_close * (1 - pullback_pct)
                in_channel = trigger <= ch_high[i]
                past_sl = trigger <= sl_price
            else:
                trigger = sig_close * (1 + pullback_pct)
                in_channel = trigger >= ch_low[i]
                past_sl = trigger >= sl_price

            invalid = in_channel or past_sl
            if invalid:
                if on_invalid == 'immediate':
                    # Fallback to immediate entry at next bar open
                    if i + 1 >= n:
                        continue
                    entry_price = opens[i + 1]
                    valid = ((direction == 'LONG' and entry_price > sl_price) or
                             (direction == 'SHORT' and entry_price < sl_price))
                    if not valid:
                        continue
                    events['invalid_fb'] += 1
                    open_position_at(
                        i + 1, direction, entry_price, sl_price,
                        signal_bar=i, bars_waited=0,
                        entry_mode='invalid_fallback',
                    )
                    continue
                else:
                    events['invalid_skip'] += 1
                    continue

            # Valid pullback setup
            pending = {
                'direction':   direction,
                'trigger':     trigger,
                'sl_price':    sl_price,
                'signal_bar':  i,
                'bars_waited': 0,
            }
            state = 'PENDING'
            events['pullback_placed'] += 1

    return trades, events


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_mdd(trades):
    cum = 0.0; peak = 0.0; mdd = 0.0
    for t in trades:
        cum += t['pnl_pct']
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return mdd


def summarize(trades, events, days):
    if not trades:
        return {'trades': 0, 'daily_pnl': 0, 'WR': 0, 'RR': 0,
                'MDD_1x': 0, 'trades_per_day': 0, 'events': events}
    total_pnl = sum(t['pnl_pct'] for t in trades)
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    wr = len(wins) / len(trades) * 100
    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t['pnl_pct'] for t in losses) / len(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 999
    mdd = compute_mdd(trades)

    reasons = {}
    for t in trades:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1

    modes = {}
    for t in trades:
        m = t.get('entry_mode', 'unknown')
        modes[m] = modes.get(m, 0) + 1

    return {
        'trades':         len(trades),
        'daily_pnl':      round(total_pnl / days, 4),
        'total_pnl_1x':   round(total_pnl, 2),
        'WR':             round(wr, 2),
        'RR':             round(rr, 3),
        'avg_win':        round(avg_win, 4),
        'avg_loss':       round(avg_loss, 4),
        'MDD_1x':         round(mdd, 2),
        'trades_per_day': round(len(trades) / days, 2),
        'per_trade':      round(total_pnl / len(trades), 4),
        'exit_reasons':   reasons,
        'entry_modes':    modes,
        'events':         events,
    }


def three_way(trades, days):
    if len(trades) < 30:
        return {'valid': False}
    n = len(trades)
    third = n // 3
    parts = [trades[0:third], trades[third:2*third], trades[2*third:]]
    pnls = [sum(t['pnl_pct'] for t in p) for p in parts]
    wrs = [sum(1 for t in p if t['pnl_pct'] > 0) / len(p) * 100
           for p in parts]
    return {
        'valid':              True,
        'train_pnl':          round(pnls[0], 2),
        'val_pnl':            round(pnls[1], 2),
        'test_pnl':           round(pnls[2], 2),
        'train_wr':           round(wrs[0], 1),
        'val_wr':             round(wrs[1], 1),
        'test_wr':            round(wrs[2], 1),
        'all_positive':       all(p > 0 for p in pnls),
        'train_not_degraded': (pnls[2] >= 0.5 * pnls[0] if pnls[0] > 0 else pnls[2] > 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print('=' * 100)
    print('PULLBACK ENTRY ABLATION + NEIGHBORHOOD — Phase 1B')
    print('=' * 100)
    df15 = load_and_resample(DATA_CSV)
    t_start = df15['timestamp'].iloc[WARMUP_BARS]
    t_end = df15['timestamp'].iloc[-1]
    days = (t_end - t_start).total_seconds() / 86400
    print(f'Loaded {len(df15)} 15m bars  |  {t_start} → {t_end}  ({days:.1f} days)')
    print()

    # ──── Ablation ────
    tests = [
        ('T0_C1_baseline',       dict(min_margin_pct=0,     pullback_pct=0)),
        ('T1_filter_only',       dict(min_margin_pct=0.001, pullback_pct=0)),
        ('T2_pullback_skip',     dict(min_margin_pct=0,     pullback_pct=0.001,
                                      ttl_bars=4, on_invalid='skip',
                                      on_ttl='skip')),
        ('T3_pullback_immediate', dict(min_margin_pct=0,     pullback_pct=0.001,
                                       ttl_bars=4, on_invalid='immediate',
                                       on_ttl='immediate')),
    ]

    print('─' * 100)
    print('ABLATION — 4 tests (pullback_pct=0.1%, TTL=4)')
    print('─' * 100)
    print(f'{"test":<28} {"trades":>7} {"tpd":>5} {"daily":>8} {"WR":>6} '
          f'{"R:R":>6} {"MDD":>6} {"per-trade":>10}')
    print('─' * 100)

    ablation = {}
    for name, params in tests:
        trades, events = run_variant(df15, **params)
        summary = summarize(trades, events, days)
        split = three_way(trades, days)
        ablation[name] = {'params': params, 'summary': summary,
                           'three_way': split, 'trades_n': len(trades)}
        rr_s = f'{summary["RR"]:.2f}' if summary['RR'] < 10 else '>10'
        pt = summary.get('per_trade', 0) or 0
        print(f'{name:<28} {summary["trades"]:>7} '
              f'{summary["trades_per_day"]:>5.2f} '
              f'{summary["daily_pnl"]:>+8.3f} '
              f'{summary["WR"]:>5.1f}% '
              f'{rr_s:>6} '
              f'{summary["MDD_1x"]:>5.1f}% '
              f'{pt:>+10.4f}')
    print()

    # Ablation interpretation
    t0 = ablation['T0_C1_baseline']['summary']['daily_pnl']
    t1 = ablation['T1_filter_only']['summary']['daily_pnl']
    t2 = ablation['T2_pullback_skip']['summary']['daily_pnl']
    t3 = ablation['T3_pullback_immediate']['summary']['daily_pnl']

    print('─' * 100)
    print('ABLATION INTERPRETATION')
    print('─' * 100)
    print(f'  T0 (C1 pure):           daily = {t0:+.4f}%/day')
    print(f'  T1 (filter only):       daily = {t1:+.4f}%/day   Δvs T0 = {t1-t0:+.4f}')
    print(f'  T2 (phase1a, skip):     daily = {t2:+.4f}%/day   Δvs T0 = {t2-t0:+.4f}')
    print(f'  T3 (timing, fallback):  daily = {t3:+.4f}%/day   Δvs T0 = {t3-t0:+.4f}')
    print()
    filter_effect = t1 - t0
    timing_effect = t3 - t0
    synergy = t2 - max(t1, t3)
    print(f'  Filter effect  (T1-T0):    {filter_effect:+.4f}%/day')
    print(f'  Timing effect  (T3-T0):    {timing_effect:+.4f}%/day')
    print(f'  Synergy        (T2-max):   {synergy:+.4f}%/day')
    print()

    # Verdict logic
    verdict = []
    if abs(filter_effect) > abs(timing_effect) * 2:
        verdict.append('✗ Filter dominant — timing 허상 or 약함')
    elif timing_effect > 0.02 and timing_effect > filter_effect:
        verdict.append('✓ Timing edge 확인 (vs baseline > 0.02%/day)')
    elif synergy > 0.02:
        verdict.append('✓ Filter+Timing synergy 확인')
    else:
        verdict.append('△ 효과 모호 — 노이즈 수준')

    # Compare with C1 baseline for GO consideration
    for name, a in ablation.items():
        delta_vs_c1 = a['summary']['daily_pnl'] - t0
        if delta_vs_c1 > 0:
            verdict.append(f'  {name}: daily {a["summary"]["daily_pnl"]:+.4f} '
                          f'(BEATS C1 by {delta_vs_c1:+.4f})')
    for v in verdict:
        print(f'  {v}')
    print()

    # ──── Neighborhood ────
    print('─' * 100)
    print('NEIGHBORHOOD GRID — pullback_pct fine sweep, TTL=4, skip-mode (T2-style)')
    print('─' * 100)
    neighborhood_grid = [0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175,
                          0.002, 0.0025]
    print(f'{"pullback":>10} | {"trades":>7} {"tpd":>5} {"daily":>8} '
          f'{"WR":>6} {"MDD":>6} {"fill%":>6}')
    print('─' * 100)

    neighborhood = {}
    for p in neighborhood_grid:
        trades, events = run_variant(df15, min_margin_pct=0, pullback_pct=p,
                                      ttl_bars=4, on_invalid='skip', on_ttl='skip')
        s = summarize(trades, events, days)
        fill_rate = events['pullback_filled'] / max(events['pullback_placed'], 1) * 100
        neighborhood[f'{p*100:.3f}%'] = {'summary': s, 'fill_rate': round(fill_rate, 1)}
        print(f'{p*100:>9.3f}% | {s["trades"]:>7} '
              f'{s["trades_per_day"]:>5.2f} '
              f'{s["daily_pnl"]:>+8.4f} '
              f'{s["WR"]:>5.1f}% '
              f'{s["MDD_1x"]:>5.1f}% '
              f'{fill_rate:>5.1f}%')
    print()

    # Neighborhood analysis: sharpness of 0.1% peak
    daily_values = [neighborhood[k]['summary']['daily_pnl'] for k in neighborhood]
    peak_idx = daily_values.index(max(daily_values))
    peak_key = list(neighborhood.keys())[peak_idx]
    peak_val = daily_values[peak_idx]
    print(f'Peak: {peak_key} @ daily {peak_val:+.4f}')
    # Measure smoothness around peak
    if 0 < peak_idx < len(daily_values) - 1:
        left = daily_values[peak_idx - 1]
        right = daily_values[peak_idx + 1]
        peak_drop = peak_val - max(left, right)
        print(f'Drop to nearest neighbors: {peak_drop:+.4f}')
        if peak_drop > 0.1:
            print('  ⚠ Sharp peak — overfit suspicion')
        else:
            print('  ✓ Smooth neighborhood — stable region')
    print()

    # ──── Save JSON ────
    out = {
        'phase':     '1B',
        'timestamp': datetime.utcnow().isoformat(),
        'data_window': {'start': str(t_start), 'end': str(t_end),
                         'days': round(days, 1)},
        'ablation':  {k: {'summary': v['summary'],
                          'three_way': v['three_way'],
                          'trades_n': v['trades_n']}
                      for k, v in ablation.items()},
        'ablation_effects': {
            'T0_daily':      t0,
            'T1_daily':      t1,
            'T2_daily':      t2,
            'T3_daily':      t3,
            'filter_effect': round(filter_effect, 4),
            'timing_effect': round(timing_effect, 4),
            'synergy':       round(synergy, 4),
        },
        'neighborhood': neighborhood,
    }
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'JSON saved → {OUTPUT_JSON.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
