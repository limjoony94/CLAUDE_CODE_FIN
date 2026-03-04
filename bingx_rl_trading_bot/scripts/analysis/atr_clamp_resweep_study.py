#!/usr/bin/env python3
"""
ATR Clamp Re-sweep Study (v1.53.0, 303d data)
===============================================

Re-validate ATR clamp bounds with extended 303d data.
Current: clamp_lo=0.5, clamp_hi=1.5 (v1.47.0 + v1.50.0).

Phase 1: clamp_lo sweep (0.3-0.8, hi=1.5 fixed)
Phase 2: clamp_hi sweep (1.0-2.0, lo=0.5 fixed)
Phase 3: 2D grid (top lo × top hi)
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
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    ATR_PERIOD, ATR_WINDOW,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'atr_clamp_resweep_study.json')

DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
MOM_THRESHOLD = 1.5
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12
TIMEOUT_BARS = 288
AGG_COUNTER = 8.0
AGG_WITH = 15.0
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def portfolio_sim(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    atr_lo=0.5, atr_hi=1.5,
):
    """Portfolio sim v1.53.0 with configurable ATR clamps."""
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
                r = clamp(atr_ratio[bar], atr_lo, atr_hi)
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            slope = ema_slope[bar] if bar < len(ema_slope) else 0
            is_uptrend = slope > 0
            is_counter = ((direction == 'SHORT' and is_uptrend) or
                          (direction == 'LONG' and not is_uptrend))
            cap_pct = AGG_COUNTER if is_counter else AGG_WITH

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
                        atr_lo=0.5, atr_hi=1.5, n_folds=3):
    total_bars = neutral_end - neutral_start
    fold_size = total_bars // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        is_end = neutral_start + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, neutral_end)
        if oos_start >= oos_end:
            continue
        oos_trades = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            atr_lo=atr_lo, atr_hi=atr_hi,
        )
        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])
    return results


def main():
    print("=" * 90)
    print("ATR CLAMP RE-SWEEP STUDY (v1.53.0, 303d data)")
    print("=" * 90)

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
    print(f"  Loaded {n_bars} bars, neutral: {neutral_start}-{neutral_end}")

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

    signal_tuples = []
    for i in range(2, n_bars):
        triplet = f"{type_codes[i-2]}-{type_codes[i-1]}-{type_codes[i]}"
        if triplet in patterns:
            info = patterns[triplet]
            signal_tuples.append((i, triplet, info['direction'], info['tp_pct'], info['sl_pct']))
    neutral_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in signal_tuples
                       if neutral_start <= b < neutral_end]
    print(f"  Signals: {len(neutral_signals)} in neutral window")

    # ATR ratio distribution
    valid_atr = atr_ratio[neutral_start:neutral_end]
    valid_atr = valid_atr[~np.isnan(valid_atr)]
    print(f"\n  ATR ratio distribution (neutral window):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{p}: {np.percentile(valid_atr, p):.3f}")

    # ==================== PHASE 1: clamp_lo sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: clamp_lo sweep (clamp_hi=1.5 fixed)")
    print("=" * 90)

    lo_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    print(f"\n{'Lo':>6} {'PnL%':>10} {'MDD%':>8} {'P/M':>10} {'WR%':>6} {'Trades':>7}")
    print("-" * 55)

    p1 = {}
    for lo in lo_vals:
        trades = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            atr_lo=lo, atr_hi=1.5,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p1[lo] = {'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
                   'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
                   'trades': stats['trades']}
        marker = ' ← current' if lo == 0.5 else ''
        print(f"{lo:>6.1f} {stats['pnl']:>+10.1f} {stats['mdd']:>8.2f} "
              f"{pm:>10.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    # ==================== PHASE 2: clamp_hi sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 2: clamp_hi sweep (clamp_lo=0.5 fixed)")
    print("=" * 90)

    hi_vals = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    print(f"\n{'Hi':>6} {'PnL%':>10} {'MDD%':>8} {'P/M':>10} {'WR%':>6} {'Trades':>7}")
    print("-" * 55)

    p2 = {}
    for hi in hi_vals:
        trades = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            atr_lo=0.5, atr_hi=hi,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p2[hi] = {'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
                   'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
                   'trades': stats['trades']}
        marker = ' ← current' if hi == 1.5 else ''
        print(f"{hi:>6.1f} {stats['pnl']:>+10.1f} {stats['mdd']:>8.2f} "
              f"{pm:>10.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    # ==================== PHASE 3: 2D grid ====================
    print("\n" + "=" * 90)
    print("PHASE 3: 2D Grid (top 3 lo × top 3 hi)")
    print("=" * 90)

    top_lo = sorted(p1.keys(), key=lambda k: p1[k]['pm'], reverse=True)[:3]
    top_hi = sorted(p2.keys(), key=lambda k: p2[k]['pm'], reverse=True)[:3]
    print(f"  Top lo: {top_lo}")
    print(f"  Top hi: {top_hi}")

    grid = {}
    configs = [(lo, hi) for lo in top_lo for hi in top_hi if hi > lo]
    configs.append((0.5, 1.5))  # current
    configs = list(set(configs))
    configs.sort()

    print(f"\n{'Config':>12} {'PnL%':>10} {'MDD%':>8} {'P/M':>10} {'WR%':>6} {'Trades':>7}")
    print("-" * 60)

    for lo, hi in configs:
        trades = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            atr_lo=lo, atr_hi=hi,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        label = f"{lo:.1f}/{hi:.1f}"
        grid[label] = {
            'lo': lo, 'hi': hi,
            'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
            'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
            'trades': stats['trades'],
        }
        marker = ' ← current' if (lo == 0.5 and hi == 1.5) else ''
        print(f"{label:>12} {stats['pnl']:>+10.1f} {stats['mdd']:>8.2f} "
              f"{pm:>10.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    # ==================== PHASE 4: WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 4: WF 3-fold validation")
    print("=" * 90)

    ranked = sorted(grid.items(), key=lambda x: x[1]['pm'], reverse=True)
    wf_candidates = []
    seen = set()
    for label, r in ranked:
        key = (r['lo'], r['hi'])
        if key not in seen:
            seen.add(key)
            wf_candidates.append((label, r))
        if len(wf_candidates) >= 8:
            break

    wf_results = {}
    print(f"\n{'Config':>12} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} "
          f"{'IS P/M':>10} {'WF':>6}")
    print("-" * 80)

    for label, r in wf_candidates:
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            atr_lo=r['lo'], atr_hi=r['hi'],
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
            'lo': r['lo'], 'hi': r['hi'],
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pnl_mdd': r['pm'],
        }
        marker = ' ← current' if label == '0.5/1.5' else ''
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:>12} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} "
              f"{r['pm']:>10.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    current_wf = wf_results.get('0.5/1.5', {})
    current_min = current_wf.get('min', -999)
    print(f"\nCurrent (0.5/1.5):")
    print(f"  IS PnL/MDD: {p1.get(0.5, {}).get('pm', 'N/A')}")
    print(f"  OOS folds: {current_wf.get('folds', [])}")

    best_label = None
    best_min = current_min
    for label, wr in wf_results.items():
        if label == '0.5/1.5':
            continue
        if wr.get('n_pass', 0) == 3 and wr['min'] > best_min:
            best_min = wr['min']
            best_label = label

    if best_label:
        best_wf = wf_results[best_label]
        delta = best_wf['min'] - current_min
        print(f"\nBest: {best_label}, min fold delta: {delta:+.1f}%")
        if delta > 5:
            print(f">>> RECOMMEND: Switch to {best_label}")
        else:
            print(f">>> RECOMMEND: Marginal ({delta:+.1f}%), KEEP current")
    else:
        print("\n>>> RECOMMEND: KEEP current (0.5/1.5)")

    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.53.0',
        'atr_distribution': {
            f'p{p}': round(float(np.percentile(valid_atr, p)), 3) for p in [5,10,25,50,75,90,95]
        },
        'phase1_lo': {str(k): v for k, v in p1.items()},
        'phase2_hi': {str(k): v for k, v in p2.items()},
        'grid': grid,
        'wf_results': wf_results,
        'recommended': best_label or '0.5/1.5',
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
