#!/usr/bin/env python3
"""
Direction Cap Re-sweep Study (v1.46.0 baseline)
=================================================

Direction Cap was set to 7 in v1.36.1 (portfolio study with old mechanism stack).
Since then: AggRisk relaxed (3/7→5/15), Cascade t85, Momentum lb3/cd12.
These changes affect directional concentration risk — cap may need re-tuning.

Sweep: cap 3-9 (full range) + uncapped
Phase 1: IS sweep with v1.46.0 full stack
Phase 2: Top configs WF 3-fold validation

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
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, calc_stats,
    DATA_FILE, PATTERNS_FILE,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER, TIMEOUT_BARS, N_SLOTS,
    ATR_CLAMP_LO, ATR_CLAMP_HI, BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'direction_cap_resweep_study.json')

# v1.46.0 settings
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
AGG_COUNTER_CAP = 5.0
AGG_WITH_CAP = 15.0
MOM_THRESHOLD = 1.0
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12

CAPS_TO_TEST = [3, 4, 5, 6, 7, 8, 9, 99]  # 99 = effectively uncapped


def portfolio_sim_cap(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    direction_cap=7,
):
    """Portfolio sim v1.46.0 with configurable direction cap."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    # Track correlated losses for analysis
    daily_losses = {}  # day -> list of (direction, pnl)

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

            # Track daily losses
            if pnl < 0:
                day = bar // BARS_PER_DAY
                daily_losses.setdefault(day, []).append((direction, pnl_portfolio))

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Cascade SL tightening (v1.45.0: t85)
        if sl_exits:
            for sl_pos in sl_exits:
                sl_dir = sl_pos['direction']
                for pos in positions:
                    if pos['direction'] == sl_dir:
                        pos['eff_sl_pct'] *= CASCADE_MULT

        equity_delta = sum(t['pnl_portfolio'] for t in trades if t['exit_bar'] == bar)
        equity *= (1 + equity_delta / 100)
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard (v1.46.0: lb3/cd12)
        if bar >= MOM_LOOKBACK:
            pa = closes[bar - MOM_LOOKBACK]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > MOM_THRESHOLD:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'], bar + MOM_COOLDOWN)
                elif pct < -MOM_THRESHOLD:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'], bar + MOM_COOLDOWN)

        if bar not in sig_by_bar:
            continue

        for pat, direction, tp_pct, sl_pct in sig_by_bar[bar]:
            if len(positions) >= N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
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
            'reason': 'FORCE_CLOSE', 'pattern': pos['pattern'],
            'direction': pos['direction'],
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
        })

    # Compute correlated loss metrics
    worst_burst = 0.0
    max_same_dir_sl = 0
    for day, losses in daily_losses.items():
        day_total = sum(pnl for _, pnl in losses)
        if day_total < worst_burst:
            worst_burst = day_total
        for d in ['LONG', 'SHORT']:
            cnt = sum(1 for dr, _ in losses if dr == d)
            if cnt > max_same_dir_sl:
                max_same_dir_sl = cnt

    return trades, worst_burst, max_same_dir_sl


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        direction_cap, n_folds=3):
    """3-fold expanding window WF."""
    total_bars = neutral_end - neutral_start
    fold_size = total_bars // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        is_end = neutral_start + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, neutral_end)
        if oos_start >= oos_end:
            continue

        oos_trades, _, _ = portfolio_sim_cap(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            direction_cap=direction_cap,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("DIRECTION CAP RE-SWEEP STUDY (v1.46.0 baseline)")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    type_codes = df['rctype'].values
    n_bars = len(df)
    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)
    neutral_start, neutral_end = find_neutral_window(closes)
    print(f"  Neutral: {neutral_start}-{neutral_end} ({neutral_end - neutral_start} bars)")

    # Load patterns
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pats_raw = pat_data['patterns']
    tpsl = pat_data.get('patterns_tpsl', {})
    patterns = {}
    for pat_name in pats_raw.get('long', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        patterns[pat_name] = {'direction': 'LONG', 'tp_pct': tp_sl[0], 'sl_pct': tp_sl[1]}
    for pat_name in pats_raw.get('short', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        patterns[pat_name] = {'direction': 'SHORT', 'tp_pct': tp_sl[0], 'sl_pct': tp_sl[1]}
    print(f"  Loaded {len(patterns)} patterns")

    # Build signals
    signal_tuples = []
    for i in range(2, n_bars):
        triplet = f"{type_codes[i-2]}-{type_codes[i-1]}-{type_codes[i]}"
        if triplet in patterns:
            info = patterns[triplet]
            signal_tuples.append((i, triplet, info['direction'], info['tp_pct'], info['sl_pct']))
    neutral_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in signal_tuples
                       if neutral_start <= b < neutral_end]
    print(f"  Signals: {len(neutral_signals)} in neutral window")

    # ==================== PHASE 1: IS sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: Direction Cap IS Sweep (v1.46.0)")
    print("=" * 90)

    results_is = {}
    print(f"\n{'Cap':<8} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7} "
          f"{'Worst Burst':>12} {'Max Dir SL':>11}")
    print("-" * 85)

    for cap in CAPS_TO_TEST:
        trades, worst_burst, max_dir_sl = portfolio_sim_cap(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            direction_cap=cap,
        )
        stats = calc_stats(trades)
        label = str(cap) if cap < 99 else 'OFF'
        results_is[label] = {
            'pnl': round(stats['pnl'], 1),
            'mdd': round(stats['mdd'], 2),
            'pnl_mdd': round(stats['pnl'] / max(stats['mdd'], 0.01), 1),
            'wr': round(stats['wr'], 1),
            'trades': stats['trades'],
            'worst_burst': round(worst_burst, 2),
            'max_dir_sl': max_dir_sl,
        }

        marker = ' ← current' if cap == 7 else ''
        print(f"{label:<8} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{stats['pnl'] / max(stats['mdd'], 0.01):>8.1f} {stats['wr']:>6.1f} "
              f"{stats['trades']:>7} {worst_burst:>+12.2f} {max_dir_sl:>11}{marker}")

    # ==================== PHASE 2: WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 2: WF 3-fold Validation (all caps)")
    print("=" * 90)

    wf_results = {}
    print(f"\n{'Cap':<8} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} {'WF':>6}")
    print("-" * 55)

    for cap in CAPS_TO_TEST:
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            direction_cap=cap,
        )
        label = str(cap) if cap < 99 else 'OFF'

        if len(folds) == 3:
            avg = np.mean(folds)
            mn = min(folds)
            n_pass = sum(1 for f in folds if f > 0)
            wf_str = f"{n_pass}/3 {'P' if n_pass == 3 else 'F'}"
        else:
            avg = mn = 0
            n_pass = 0
            wf_str = "N/A"

        wf_results[label] = {
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1),
            'min': round(mn, 1),
            'n_pass': n_pass,
        }

        marker = ' ← current' if cap == 7 else ''
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:<8} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    current = results_is['7']
    current_wf = wf_results['7']
    print(f"\nCurrent (cap=7):")
    print(f"  IS PnL/MDD: {current['pnl_mdd']}, MDD: {current['mdd']}%")
    print(f"  OOS avg: {current_wf['avg']}, min: {current_wf['min']}")
    print(f"  Worst burst: {current['worst_burst']}%, Max dir SL: {current['max_dir_sl']}")

    # Find best WF-passing
    best_label = None
    best_avg = -999
    for label in [str(c) if c < 99 else 'OFF' for c in CAPS_TO_TEST]:
        r = wf_results.get(label, {})
        if r.get('n_pass', 0) == 3 and r['avg'] > best_avg:
            best_avg = r['avg']
            best_label = label

    if best_label:
        best_is = results_is[best_label]
        best_wf = wf_results[best_label]
        print(f"\nBest WF-passing (cap={best_label}):")
        print(f"  IS PnL/MDD: {best_is['pnl_mdd']}, MDD: {best_is['mdd']}%")
        print(f"  OOS avg: {best_wf['avg']}, min: {best_wf['min']}")
        print(f"  Worst burst: {best_is['worst_burst']}%, Max dir SL: {best_is['max_dir_sl']}")

        if best_label == '7':
            print("\n>>> RECOMMEND: KEEP current cap=7")
        else:
            delta_oos = best_wf['avg'] - current_wf['avg']
            delta_burst = best_is['worst_burst'] - current['worst_burst']
            print(f"\n  OOS avg delta: {delta_oos:+.1f}%")
            print(f"  Worst burst delta: {delta_burst:+.2f}%")
            if delta_oos > 5 and delta_burst > -2:
                print(f"\n>>> RECOMMEND: Switch to cap={best_label}")
            elif delta_oos > 5:
                print(f"\n>>> CAUTION: OOS improves but burst risk increases")
            else:
                print(f"\n>>> RECOMMEND: Marginal improvement, keep current")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.46.0',
        'is_results': results_is,
        'wf_results': wf_results,
        'recommended': best_label,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
