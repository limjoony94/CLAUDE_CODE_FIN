#!/usr/bin/env python3
"""
Timeout Bars Parameter Sweep Study (v1.47.0 baseline)
=====================================================

Position timeout: bars after which position is DROP (force-closed at market).
Currently 864 bars = 72h. Never re-swept since v1.31.0.

v1.43.0 tried 432 (36h) as part of entry optimization → ROLLED BACK.
Now sweep independently under v1.47.0 stack.

Phase 1: Timeout sweep (288-1440, step 144 = 12h increments)
Phase 2: Fine-grained sweep around best
Phase 3: Include NO_TIMEOUT
Phase 4: WF 3-fold validation

Standard Research Protocol: LEVERAGE=3, FEE×LEV, ATR-scaled, Compound
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
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER, N_SLOTS,
    BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'timeout_sweep_study.json')

# v1.47.0 fixed params
DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
AGG_COUNTER_CAP = 5.0
AGG_WITH_CAP = 15.0
MOM_THRESHOLD = 1.0
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12
ATR_LO = 0.5
ATR_HI = 1.7
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3


def portfolio_sim_timeout(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    timeout_bars=864, timeout_enabled=True,
):
    """Portfolio sim v1.47.0 with configurable timeout."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}
    timeout_exits = 0

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

            # Timeout (configurable)
            if timeout_enabled and hold >= timeout_bars:
                closed_slots.add(pos['slot'])
                timeout_exits += 1
                continue

            # Early exit (fixed v1.47.0)
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
                        pos['eff_sl_pct'] *= CASCADE_MULT

        equity_delta = sum(t['pnl_portfolio'] for t in trades if t['exit_bar'] == bar)
        equity *= (1 + equity_delta / 100)
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard
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
                r = clamp(atr_ratio[bar], ATR_LO, ATR_HI)
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

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

    return trades, timeout_exits


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        timeout_bars, timeout_enabled=True, n_folds=3):
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

        oos_trades, _ = portfolio_sim_timeout(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            timeout_bars=timeout_bars, timeout_enabled=timeout_enabled,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("TIMEOUT BARS PARAMETER SWEEP STUDY (v1.47.0 baseline)")
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
    print(f"  Loaded {n_bars} bars, neutral: {neutral_start}-{neutral_end} ({neutral_end - neutral_start} bars)")

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

    # ==================== PHASE 1: Coarse sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: Coarse timeout sweep (12h increments)")
    print("=" * 90)

    # 288=24h, 432=36h, 576=48h, 720=60h, 864=72h(current), 1008=84h, 1152=96h, 1440=5d
    timeout_vals = [288, 432, 576, 720, 864, 1008, 1152, 1440]
    print(f"\n{'Timeout':>8} {'Hours':>6} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'TO_exit':>8}")
    print("-" * 65)

    p1_results = {}
    for to in timeout_vals:
        trades, n_to = portfolio_sim_timeout(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            timeout_bars=to,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        hours = to * 5 / 60
        p1_results[to] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                          'wr': stats['wr'], 'trades': stats['trades'], 'to_exits': n_to,
                          'hours': hours}
        marker = ' ← current' if to == 864 else ''
        print(f"{to:>8} {hours:>6.0f}h {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {n_to:>8}{marker}")

    # ==================== PHASE 2: Fine-grained around best ====================
    print("\n" + "=" * 90)
    print("PHASE 2: Fine-grained sweep around top region")
    print("=" * 90)

    # Find top 2 coarse values
    sorted_coarse = sorted(p1_results.keys(), key=lambda k: p1_results[k]['pm'], reverse=True)
    best_coarse = sorted_coarse[0]

    # Fine sweep ±2 steps around best (72-bar = 6h increments)
    fine_center = best_coarse
    fine_vals = sorted(set([fine_center + i * 72 for i in range(-3, 4)
                           if fine_center + i * 72 >= 144]))
    # Remove already-tested coarse values
    fine_vals = [v for v in fine_vals if v not in p1_results]

    if fine_vals:
        print(f"\n  Fine sweep around {best_coarse} (±{3*72} bars)")
        print(f"\n{'Timeout':>8} {'Hours':>6} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
              f"{'Trades':>7} {'TO_exit':>8}")
        print("-" * 65)

        for to in fine_vals:
            trades, n_to = portfolio_sim_timeout(
                neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                timeout_bars=to,
            )
            stats = calc_stats(trades)
            pm = stats['pnl'] / max(stats['mdd'], 0.01)
            hours = to * 5 / 60
            p1_results[to] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                              'wr': stats['wr'], 'trades': stats['trades'], 'to_exits': n_to,
                              'hours': hours}
            print(f"{to:>8} {hours:>6.0f}h {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
                  f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {n_to:>8}")
    else:
        print("\n  No additional fine points needed (best at boundary)")

    # ==================== PHASE 3: Include NO_TIMEOUT ====================
    print("\n" + "=" * 90)
    print("PHASE 3: Compare with NO_TIMEOUT")
    print("=" * 90)

    trades_off, n_to_off = portfolio_sim_timeout(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end,
        timeout_bars=864, timeout_enabled=False,
    )
    stats_off = calc_stats(trades_off)
    pm_off = stats_off['pnl'] / max(stats_off['mdd'], 0.01)

    print(f"\n  NO_TIMEOUT: PnL {stats_off['pnl']:+.1f}%, MDD {stats_off['mdd']:.2f}%, "
          f"PnL/MDD {pm_off:.1f}, WR {stats_off['wr']:.1f}%, Trades {stats_off['trades']}")

    current = p1_results.get(864, {})
    print(f"  CURRENT(864): PnL {current.get('pnl', 0):+.1f}%, MDD {current.get('mdd', 0):.2f}%, "
          f"PnL/MDD {current.get('pm', 0):.1f}, WR {current.get('wr', 0):.1f}%")

    # ==================== PHASE 4: WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 4: WF 3-fold validation")
    print("=" * 90)

    # Top 5 by IS PnL/MDD + current + OFF
    all_configs = [(to, r) for to, r in sorted(p1_results.items(), key=lambda x: x[1]['pm'], reverse=True)]
    wf_candidates = all_configs[:5]

    # Ensure current is included
    current_in = any(to == 864 for to, _ in wf_candidates)
    if not current_in:
        wf_candidates.append((864, p1_results[864]))

    # Add NO_TIMEOUT
    wf_candidates.append(('OFF', {'pnl': stats_off['pnl'], 'mdd': stats_off['mdd'],
                                  'pm': pm_off, 'wr': stats_off['wr'],
                                  'trades': stats_off['trades'], 'to_exits': 0}))

    print(f"\n{'Config':>12} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} "
          f"{'IS P/M':>8} {'WF':>6}")
    print("-" * 75)

    wf_results = {}
    for to, r in wf_candidates:
        if to == 'OFF':
            folds = expanding_window_wf(
                neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                timeout_bars=864, timeout_enabled=False,
            )
            label = 'OFF'
            is_pm = pm_off
        else:
            folds = expanding_window_wf(
                neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                timeout_bars=to,
            )
            label = str(to)
            is_pm = r['pm']

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
            'timeout': to if to != 'OFF' else None,
            'enabled': to != 'OFF',
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pnl_mdd': round(is_pm, 1),
        }

        marker = ''
        if to == 864:
            marker = ' ← current'
        elif to == 'OFF':
            marker = ' ← OFF'
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:>12} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} "
              f"{is_pm:>8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    # Holding time analysis
    print("\n--- Holding time distribution by timeout ---")
    for to_val in [864, all_configs[0][0]]:
        if to_val == 864:
            trades_ht, _ = portfolio_sim_timeout(
                neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                timeout_bars=864,
            )
        else:
            trades_ht, _ = portfolio_sim_timeout(
                neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                timeout_bars=to_val,
            )
        holding = [t['exit_bar'] - t['entry_bar'] for t in trades_ht]
        if holding:
            arr = np.array(holding)
            hours_arr = arr * 5 / 60
            print(f"\n  timeout={to_val} ({to_val*5/60:.0f}h):")
            print(f"    P25={np.percentile(hours_arr, 25):.0f}h, "
                  f"P50={np.percentile(hours_arr, 50):.0f}h, "
                  f"P75={np.percentile(hours_arr, 75):.0f}h, "
                  f"P95={np.percentile(hours_arr, 95):.0f}h, "
                  f"max={np.max(hours_arr):.0f}h")

    # Summary
    current_wf = wf_results.get('864', {})
    print(f"\nCurrent (864 bars = 72h):")
    print(f"  IS PnL/MDD: {p1_results.get(864, {}).get('pm', 'N/A'):.1f}")
    print(f"  OOS avg: {current_wf.get('avg', 'N/A')}, min: {current_wf.get('min', 'N/A')}")
    print(f"  Timeout exits: {p1_results.get(864, {}).get('to_exits', 0)}")

    best_label = None
    best_min = -999
    for label, wr in wf_results.items():
        if wr.get('n_pass', 0) == 3 and wr['min'] > best_min:
            best_min = wr['min']
            best_label = label

    if best_label:
        best_wf = wf_results[best_label]
        print(f"\nBest WF-passing by min fold: {best_label}")
        if best_label != 'OFF':
            best_is = p1_results.get(int(best_label), {})
            print(f"  IS PnL/MDD: {best_is.get('pm', 'N/A'):.1f}")
        print(f"  OOS avg: {best_wf['avg']}, min: {best_wf['min']}")

        delta_min = best_wf['min'] - current_wf.get('min', 0)
        if best_label == '864':
            print("\n>>> RECOMMEND: KEEP current timeout (864 bars = 72h)")
        elif delta_min > 5:
            print(f"\n  Min fold delta: {delta_min:+.1f}%")
            print(f"\n>>> RECOMMEND: Switch to timeout={best_label}")
        else:
            print(f"\n  Min fold delta: {delta_min:+.1f}%")
            print(f"\n>>> RECOMMEND: Marginal, consider keeping current")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.47.0',
        'all_results': {str(k): v for k, v in p1_results.items()},
        'no_timeout': {'pnl': stats_off['pnl'], 'mdd': stats_off['mdd'],
                       'pm': pm_off, 'wr': stats_off['wr'], 'trades': stats_off['trades']},
        'wf_results': wf_results,
        'recommended': best_label,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
