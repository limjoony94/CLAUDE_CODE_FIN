#!/usr/bin/env python3
"""
ATR Clamp Parameter Sweep Study (v1.46.0 baseline)
====================================================

ATR scaling clamps (lo=0.6, hi=1.7) were set in v1.28.42 and never re-swept
under the current mechanism stack (Cascade SL t85, AggRisk 5/15, etc.).

ATR ratio = ATR(14) / rolling_median(ATR(14), 576). Clamped to [lo, hi].
Affects both TP and SL proportionally, preserving R:R ratio.

Phase 1: clamp_lo sweep (0.3-0.9)
Phase 2: clamp_hi sweep (1.2-2.5)
Phase 3: 2D grid search top combos
Phase 4: WF 3-fold validation
Also test: ATR OFF (ratio fixed at 1.0)

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
    BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'atr_clamp_sweep_study.json')

# v1.46.0 fixed params
DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
AGG_COUNTER_CAP = 5.0
AGG_WITH_CAP = 15.0
MOM_THRESHOLD = 1.0
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12


def portfolio_sim_atr(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    clamp_lo=0.6, clamp_hi=1.7, atr_enabled=True,
):
    """Portfolio sim v1.46.0 with configurable ATR clamp params."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

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

            # MDD sizing
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

            # ATR scaling (configurable clamps)
            if atr_enabled and bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                r = clamp(atr_ratio[bar], clamp_lo, clamp_hi)
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

    return trades


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        clamp_lo, clamp_hi, atr_enabled=True, n_folds=3):
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

        oos_trades = portfolio_sim_atr(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            clamp_lo=clamp_lo, clamp_hi=clamp_hi, atr_enabled=atr_enabled,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("ATR CLAMP PARAMETER SWEEP STUDY (v1.46.0 baseline)")
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

    # ATR ratio distribution
    valid_atr = atr_ratio[~np.isnan(atr_ratio)]
    print(f"\n  ATR ratio stats: mean={np.mean(valid_atr):.3f}, "
          f"std={np.std(valid_atr):.3f}, "
          f"min={np.min(valid_atr):.3f}, max={np.max(valid_atr):.3f}")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{pct}: {np.percentile(valid_atr, pct):.3f}")

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

    all_results = {}

    # ==================== BASELINE + OFF ====================
    print("\n" + "=" * 90)
    print("BASELINE: Current (lo=0.6, hi=1.7) vs ATR OFF")
    print("=" * 90)

    base_trades = portfolio_sim_atr(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end,
        clamp_lo=0.6, clamp_hi=1.7, atr_enabled=True,
    )
    base_stats = calc_stats(base_trades)
    print(f"  Current (0.6/1.7): PnL={base_stats['pnl']:+.1f}%, MDD={base_stats['mdd']:.2f}%, "
          f"PnL/MDD={base_stats['pnl'] / max(base_stats['mdd'], 0.01):.1f}, "
          f"WR={base_stats['wr']:.1f}%, Trades={base_stats['trades']}")

    off_trades = portfolio_sim_atr(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end,
        clamp_lo=0.6, clamp_hi=1.7, atr_enabled=False,
    )
    off_stats = calc_stats(off_trades)
    print(f"  ATR OFF:           PnL={off_stats['pnl']:+.1f}%, MDD={off_stats['mdd']:.2f}%, "
          f"PnL/MDD={off_stats['pnl'] / max(off_stats['mdd'], 0.01):.1f}, "
          f"WR={off_stats['wr']:.1f}%, Trades={off_stats['trades']}")

    # ==================== PHASE 1: clamp_lo sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: clamp_lo sweep (hi=1.7 fixed)")
    print("=" * 90)

    lo_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"\n{'lo':>6} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7}")
    print("-" * 50)

    p1_results = {}
    for lo in lo_vals:
        trades = portfolio_sim_atr(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            clamp_lo=lo, clamp_hi=1.7,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p1_results[lo] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                          'wr': stats['wr'], 'trades': stats['trades']}
        marker = ' ← current' if lo == 0.6 else ''
        print(f"{lo:>6.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    # ==================== PHASE 2: clamp_hi sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 2: clamp_hi sweep (lo=0.6 fixed)")
    print("=" * 90)

    hi_vals = [1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]
    print(f"\n{'hi':>6} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7}")
    print("-" * 50)

    p2_results = {}
    for hi in hi_vals:
        trades = portfolio_sim_atr(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            clamp_lo=0.6, clamp_hi=hi,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p2_results[hi] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                          'wr': stats['wr'], 'trades': stats['trades']}
        marker = ' ← current' if hi == 1.7 else ''
        print(f"{hi:>6.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    # ==================== PHASE 3: 2D Grid ====================
    print("\n" + "=" * 90)
    print("PHASE 3: 2D Grid (top candidates)")
    print("=" * 90)

    top_lo = sorted(p1_results.keys(), key=lambda k: p1_results[k]['pm'], reverse=True)[:4]
    top_hi = sorted(p2_results.keys(), key=lambda k: p2_results[k]['pm'], reverse=True)[:4]

    print(f"\n  Top lo: {top_lo}")
    print(f"  Top hi: {top_hi}")

    grid_results = {}
    configs = []
    for lo in top_lo:
        for hi in top_hi:
            if lo >= hi:
                continue
            configs.append((lo, hi))
    configs.append((0.6, 1.7))  # current
    configs = list(set(configs))
    configs.sort()

    print(f"\n  Grid: {len(configs)} configs")
    print(f"\n{'Config':>12} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7}")
    print("-" * 55)

    for lo, hi in configs:
        trades = portfolio_sim_atr(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            clamp_lo=lo, clamp_hi=hi,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        label = f"{lo:.1f}/{hi:.1f}"
        grid_results[label] = {
            'lo': lo, 'hi': hi,
            'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
            'pnl_mdd': round(pm, 1), 'wr': round(stats['wr'], 1),
            'trades': stats['trades'],
        }
        marker = ' ← current' if lo == 0.6 and hi == 1.7 else ''
        print(f"{label:>12} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    # ==================== PHASE 4: WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 4: WF 3-fold validation (top 5 + current + OFF)")
    print("=" * 90)

    ranked = sorted(grid_results.items(), key=lambda x: x[1]['pnl_mdd'], reverse=True)
    wf_candidates = []
    seen = set()
    for label, r in ranked:
        key = (r['lo'], r['hi'])
        if key not in seen:
            seen.add(key)
            wf_candidates.append((label, r))
        if len(wf_candidates) >= 5:
            break

    # Ensure current is included
    cur_key = (0.6, 1.7)
    if cur_key not in seen:
        seen.add(cur_key)
        wf_candidates.append(("0.6/1.7", grid_results.get("0.6/1.7", {})))

    # Add OFF
    wf_candidates.append(("OFF", {'lo': 0.6, 'hi': 1.7}))

    wf_results = {}
    print(f"\n{'Config':>12} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} {'WF':>6}")
    print("-" * 60)

    for label, r in wf_candidates:
        is_off = label == 'OFF'
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            clamp_lo=r.get('lo', 0.6), clamp_hi=r.get('hi', 1.7),
            atr_enabled=not is_off,
        )

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
            'lo': r.get('lo', 0.6), 'hi': r.get('hi', 1.7),
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pnl_mdd': r.get('pnl_mdd', 0),
        }

        marker = ''
        if label == '0.6/1.7':
            marker = ' ← current'
        elif label == 'OFF':
            marker = ' ← OFF'
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:>12} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    cur_label = "0.6/1.7"
    current_is = grid_results.get(cur_label, {})
    current_wf = wf_results.get(cur_label, {})
    print(f"\nCurrent (0.6/1.7):")
    print(f"  IS PnL/MDD: {current_is.get('pnl_mdd', 'N/A')}")
    print(f"  OOS avg: {current_wf.get('avg', 'N/A')}, min: {current_wf.get('min', 'N/A')}")

    off_wf = wf_results.get('OFF', {})
    print(f"\nATR OFF:")
    print(f"  IS PnL/MDD: {off_stats['pnl'] / max(off_stats['mdd'], 0.01):.1f}")
    print(f"  OOS avg: {off_wf.get('avg', 'N/A')}, min: {off_wf.get('min', 'N/A')}")

    # Find best WF-passing by min fold
    best_label = None
    best_min = -999
    for label, wr in wf_results.items():
        if wr.get('n_pass', 0) == 3 and wr['min'] > best_min:
            best_min = wr['min']
            best_label = label

    if best_label:
        best_wf = wf_results[best_label]
        best_is = grid_results.get(best_label, {})
        print(f"\nBest WF-passing by min fold: {best_label}")
        print(f"  IS PnL/MDD: {best_is.get('pnl_mdd', 'N/A')}")
        print(f"  OOS avg: {best_wf['avg']}, min: {best_wf['min']}")

        if best_label == cur_label:
            print("\n>>> RECOMMEND: KEEP current ATR clamps (0.6/1.7)")
        elif best_label == 'OFF':
            print("\n>>> RECOMMEND: DISABLE ATR scaling")
        else:
            delta_min = best_wf['min'] - current_wf.get('min', 0)
            delta_is = best_is.get('pnl_mdd', 0) - current_is.get('pnl_mdd', 0)
            print(f"\n  Min fold delta: {delta_min:+.1f}%")
            print(f"  IS PnL/MDD delta: {delta_is:+.1f}")
            if delta_min > 5 or delta_is > 20:
                print(f"\n>>> RECOMMEND: Switch to {best_label}")
            else:
                print(f"\n>>> RECOMMEND: Marginal improvement, consider keeping current")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.46.0',
        'atr_distribution': {
            'mean': round(float(np.mean(valid_atr)), 3),
            'std': round(float(np.std(valid_atr)), 3),
            'p5': round(float(np.percentile(valid_atr, 5)), 3),
            'p95': round(float(np.percentile(valid_atr, 95)), 3),
        },
        'baseline': {
            'pnl': round(base_stats['pnl'], 1), 'mdd': round(base_stats['mdd'], 2),
            'pnl_mdd': round(base_stats['pnl'] / max(base_stats['mdd'], 0.01), 1),
        },
        'off': {
            'pnl': round(off_stats['pnl'], 1), 'mdd': round(off_stats['mdd'], 2),
            'pnl_mdd': round(off_stats['pnl'] / max(off_stats['mdd'], 0.01), 1),
        },
        'phase1_lo': {str(k): v for k, v in p1_results.items()},
        'phase2_hi': {str(k): v for k, v in p2_results.items()},
        'grid_results': grid_results,
        'wf_results': wf_results,
        'recommended': best_label,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
