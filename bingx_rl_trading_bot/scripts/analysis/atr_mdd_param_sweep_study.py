#!/usr/bin/env python3
"""
ATR clamp_hi + MDD Sizing Parameter Sweep (v1.49.0 baseline)
=============================================================

Two independent parameter groups swept under v1.49.0 stack:
  1) ATR clamp_hi: upper clamp for ATR ratio (currently 1.7)
  2) MDD sizing: full_below / min_above / min_scale thresholds

Phase 1: ATR clamp_hi sweep (1.2-3.0)
Phase 2: MDD full_below sweep (1.0-8.0, others fixed)
Phase 3: MDD min_above sweep (8.0-25.0, others fixed)
Phase 4: MDD min_scale sweep (0.1-0.5, others fixed)
Phase 5: Best combos + WF 3-fold validation

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
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'atr_mdd_param_sweep_study.json')

# v1.49.0 fixed params
DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
MOM_THRESHOLD = 1.0
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12
ATR_LO = 0.5
TIMEOUT_BARS = 288
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3
AGG_COUNTER_CAP = 8.0
AGG_WITH_CAP = 15.0


def portfolio_sim(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    atr_hi=1.7, mdd_full_below=3.0, mdd_min_above=15.0, mdd_min_scale=0.25,
):
    """Portfolio sim v1.49.0 with configurable ATR clamp_hi and MDD sizing."""
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

            # Timeout
            if hold >= TIMEOUT_BARS:
                closed_slots.add(pos['slot'])
                continue

            # Early exit
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
                if dd_pct <= mdd_full_below:
                    mdd_scale = 1.0
                elif dd_pct >= mdd_min_above:
                    mdd_scale = mdd_min_scale
                else:
                    mdd_scale = 1.0 - (1.0 - mdd_min_scale) * (
                        dd_pct - mdd_full_below) / (mdd_min_above - mdd_full_below)
                sm *= mdd_scale

            if bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                r = clamp(atr_ratio[bar], ATR_LO, atr_hi)
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # AggRisk cap
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
                        atr_hi=1.7, mdd_full_below=3.0, mdd_min_above=15.0,
                        mdd_min_scale=0.25, n_folds=3):
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
            atr_hi=atr_hi, mdd_full_below=mdd_full_below,
            mdd_min_above=mdd_min_above, mdd_min_scale=mdd_min_scale,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("ATR clamp_hi + MDD SIZING PARAMETER SWEEP (v1.49.0)")
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
    print(f"  Loaded {n_bars} bars, neutral: {neutral_start}-{neutral_end}")

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

    # Baseline
    print("\n--- Baseline (v1.49.0: atr_hi=1.7, mdd_full=3, mdd_above=15, mdd_scale=0.25) ---")
    base_trades = portfolio_sim(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end,
    )
    base_stats = calc_stats(base_trades)
    base_pm = base_stats['pnl'] / max(base_stats['mdd'], 0.01)
    print(f"  PnL: {base_stats['pnl']:+.1f}%, MDD: {base_stats['mdd']:.2f}%, "
          f"P/M: {base_pm:.1f}, WR: {base_stats['wr']:.1f}%, Trades: {base_stats['trades']}")

    all_results = {'baseline': {
        'pnl': round(base_stats['pnl'], 1), 'mdd': round(base_stats['mdd'], 2),
        'pnl_mdd': round(base_pm, 1), 'wr': round(base_stats['wr'], 1),
        'trades': base_stats['trades'],
    }}

    # ==================== PHASE 1: ATR clamp_hi sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: ATR clamp_hi sweep (1.2-3.0)")
    print("=" * 90)

    atr_hi_vals = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.3, 2.5, 3.0]
    print(f"\n{'clamp_hi':>10} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7}")
    print("-" * 55)

    p1_results = {}
    for hi in atr_hi_vals:
        trades = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            atr_hi=hi,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p1_results[hi] = {'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
                           'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
                           'trades': stats['trades']}
        marker = ' ← current' if hi == 1.7 else ''
        print(f"{hi:>10.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    all_results['phase1_atr_hi'] = {str(k): v for k, v in p1_results.items()}

    # ==================== PHASE 2: MDD full_below sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 2: MDD full_below sweep (min_above=15, min_scale=0.25 fixed)")
    print("=" * 90)

    full_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    print(f"\n{'full_below':>12} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7}")
    print("-" * 55)

    p2_results = {}
    for fb in full_vals:
        trades = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mdd_full_below=fb,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p2_results[fb] = {'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
                           'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
                           'trades': stats['trades']}
        marker = ' ← current' if fb == 3.0 else ''
        print(f"{fb:>12.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    all_results['phase2_mdd_full_below'] = {str(k): v for k, v in p2_results.items()}

    # ==================== PHASE 3: MDD min_above sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 3: MDD min_above sweep (full_below=3, min_scale=0.25 fixed)")
    print("=" * 90)

    above_vals = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
    print(f"\n{'min_above':>12} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7}")
    print("-" * 55)

    p3_results = {}
    for ma in above_vals:
        trades = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mdd_min_above=ma,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p3_results[ma] = {'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
                           'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
                           'trades': stats['trades']}
        marker = ' ← current' if ma == 15.0 else ''
        print(f"{ma:>12.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    all_results['phase3_mdd_min_above'] = {str(k): v for k, v in p3_results.items()}

    # ==================== PHASE 4: MDD min_scale sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 4: MDD min_scale sweep (full_below=3, min_above=15 fixed)")
    print("=" * 90)

    scale_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    print(f"\n{'min_scale':>12} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7}")
    print("-" * 55)

    p4_results = {}
    for ms in scale_vals:
        trades = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mdd_min_scale=ms,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p4_results[ms] = {'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
                           'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
                           'trades': stats['trades']}
        marker = ' ← current' if ms == 0.25 else ''
        print(f"{ms:>12.2f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    all_results['phase4_mdd_min_scale'] = {str(round(k, 2)): v for k, v in p4_results.items()}

    # ==================== PHASE 5: WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 5: WF 3-fold validation — top candidates + current")
    print("=" * 90)

    # Pick best from each phase + current baseline
    wf_configs = []

    # Current baseline
    wf_configs.append(('CURRENT', {'atr_hi': 1.7, 'mdd_full_below': 3.0,
                                     'mdd_min_above': 15.0, 'mdd_min_scale': 0.25}))

    # Best ATR clamp_hi by PnL/MDD
    best_hi = max(p1_results, key=lambda k: p1_results[k]['pm'])
    if best_hi != 1.7:
        wf_configs.append((f'atr_hi={best_hi}', {'atr_hi': best_hi, 'mdd_full_below': 3.0,
                                                    'mdd_min_above': 15.0, 'mdd_min_scale': 0.25}))
    # 2nd best ATR
    sorted_hi = sorted(p1_results, key=lambda k: p1_results[k]['pm'], reverse=True)
    if len(sorted_hi) > 1 and sorted_hi[1] != 1.7:
        hi2 = sorted_hi[1]
        wf_configs.append((f'atr_hi={hi2}', {'atr_hi': hi2, 'mdd_full_below': 3.0,
                                                'mdd_min_above': 15.0, 'mdd_min_scale': 0.25}))

    # Best MDD full_below
    best_fb = max(p2_results, key=lambda k: p2_results[k]['pm'])
    if best_fb != 3.0:
        wf_configs.append((f'full_below={best_fb}', {'atr_hi': 1.7, 'mdd_full_below': best_fb,
                                                       'mdd_min_above': 15.0, 'mdd_min_scale': 0.25}))

    # Best MDD min_above
    best_ma = max(p3_results, key=lambda k: p3_results[k]['pm'])
    if best_ma != 15.0:
        wf_configs.append((f'min_above={best_ma}', {'atr_hi': 1.7, 'mdd_full_below': 3.0,
                                                      'mdd_min_above': best_ma, 'mdd_min_scale': 0.25}))

    # Best MDD min_scale
    best_ms = max(p4_results, key=lambda k: p4_results[k]['pm'])
    if best_ms != 0.25:
        wf_configs.append((f'min_scale={best_ms:.2f}', {'atr_hi': 1.7, 'mdd_full_below': 3.0,
                                                          'mdd_min_above': 15.0, 'mdd_min_scale': best_ms}))

    # Combined best from independent sweeps
    combo_params = {'atr_hi': best_hi, 'mdd_full_below': best_fb,
                    'mdd_min_above': best_ma, 'mdd_min_scale': best_ms}
    combo_label = f"combo(hi={best_hi},fb={best_fb},ma={best_ma},ms={best_ms:.2f})"
    if combo_params != {'atr_hi': 1.7, 'mdd_full_below': 3.0, 'mdd_min_above': 15.0, 'mdd_min_scale': 0.25}:
        wf_configs.append((combo_label, combo_params))

    # MDD OFF (all at full size)
    wf_configs.append(('MDD_OFF', {'atr_hi': 1.7, 'mdd_full_below': 999.0,
                                     'mdd_min_above': 999.0, 'mdd_min_scale': 1.0}))

    # Deduplicate
    seen = set()
    unique_configs = []
    for label, params in wf_configs:
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            unique_configs.append((label, params))

    print(f"\n  WF candidates: {len(unique_configs)}")
    print(f"\n{'Config':>35} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} {'WF':>6}")
    print("-" * 85)

    wf_results = {}
    for label, params in unique_configs:
        # IS stats
        is_trades = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end, **params,
        )
        is_stats = calc_stats(is_trades)
        is_pm = is_stats['pnl'] / max(is_stats['mdd'], 0.01)

        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end, **params,
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
            'params': params,
            'is_pnl_mdd': round(is_pm, 1),
            'is_pnl': round(is_stats['pnl'], 1),
            'is_mdd': round(is_stats['mdd'], 2),
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
        }

        marker = ' ← current' if label == 'CURRENT' else ''
        if label == 'MDD_OFF':
            marker = ' ← MDD OFF'
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:>35} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} {wf_str:>6}{marker}")

    all_results['phase5_wf'] = wf_results

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    current_wf = wf_results.get('CURRENT', {})
    print(f"\nCurrent: IS P/M {current_wf.get('is_pnl_mdd', 'N/A')}, "
          f"OOS avg {current_wf.get('avg', 'N/A')}, min {current_wf.get('min', 'N/A')}")

    # Find best WF-passing by min fold
    best_label = None
    best_min = -999
    for label, wr in wf_results.items():
        if wr.get('n_pass', 0) == 3 and wr['min'] > best_min:
            best_min = wr['min']
            best_label = label

    if best_label:
        best_wf = wf_results[best_label]
        print(f"\nBest WF-passing (by min fold): {best_label}")
        print(f"  IS P/M: {best_wf['is_pnl_mdd']}, OOS avg: {best_wf['avg']}, min: {best_wf['min']}")
        print(f"  Params: {best_wf['params']}")

        current_min = current_wf.get('min', 0)
        delta = best_wf['min'] - current_min
        print(f"\n  Min fold delta vs current: {delta:+.1f}%")

        if best_label == 'CURRENT':
            print("\n>>> RECOMMEND: KEEP all current parameters")
        elif delta > 5:
            print(f"\n>>> RECOMMEND: APPLY {best_label}")
        else:
            print(f"\n>>> RECOMMEND: Marginal ({delta:+.1f}%), KEEP current")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.49.0',
        'results': all_results,
        'wf_results': wf_results,
        'recommendation': best_label if best_label else 'KEEP',
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
