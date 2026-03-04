#!/usr/bin/env python3
"""
Momentum Guard Parameter Sweep Study (v1.45.0 baseline)
=========================================================

Current Momentum Guard (v1.36.2):
  threshold_pct=1.0%, lookback_bars=6, cooldown_bars=6
  Chosen from a single config in entry_improvement_study — never grid-searched.

This study sweeps the 3D parameter space:
  - threshold_pct: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] (trigger sensitivity)
  - lookback_bars: [3, 6, 12] (detection window: 15m/30m/1h)
  - cooldown_bars: [3, 6, 12, 24] (block duration: 15m/30m/1h/2h)
  + OFF baseline

Phase 1: Full grid IS sweep (6×3×4 = 72 configs + OFF = 73)
Phase 2: Top-5 WF 3-fold validation
Phase 3: Ablation impact (ON vs OFF delta)

Standard Research Protocol: LEVERAGE=3, FEE×LEV, Timeout DROP, ATR-scaled, Compound

Author: Research Agent
Date: 2026-03-05
"""

import os
import sys
import json
import warnings
from datetime import datetime
from itertools import product

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, calc_stats,
    DATA_FILE, PATTERNS_FILE,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER, TIMEOUT_BARS, N_SLOTS, DIRECTION_CAP,
    ATR_CLAMP_LO, ATR_CLAMP_HI, BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'momentum_guard_sweep_study.json')

# v1.45.0 settings
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0  # 0.15
AGG_COUNTER_CAP = 5.0
AGG_WITH_CAP = 15.0

# Sweep grid
THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
LOOKBACKS = [3, 6, 12]
COOLDOWNS = [3, 6, 12, 24]


def portfolio_sim_momentum(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    mom_threshold=1.0, mom_lookback=6, mom_cooldown=6,
    momentum_enabled=True,
):
    """Portfolio sim v1.45.0 with configurable momentum guard parameters."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}
    blocked_count = 0

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

        # Momentum guard (configurable)
        if momentum_enabled and bar >= mom_lookback:
            pa = closes[bar - mom_lookback]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > mom_threshold:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'], bar + mom_cooldown)
                elif pct < -mom_threshold:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'], bar + mom_cooldown)

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

            if momentum_enabled and bar < mom_pause_until.get(direction, -1):
                blocked_count += 1
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

            # AggRisk cap (v1.45.0: 5/15)
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

    return trades, blocked_count


def build_signal_index(opens, type_codes, n_bars, atr_ratio, patterns):
    """Build signal tuples from pattern matches."""
    type_str = [None] * n_bars
    for i in range(n_bars):
        type_str[i] = type_codes[i] if i < len(type_codes) else ''

    signal_tuples = []
    for i in range(2, n_bars):
        triplet = f"{type_str[i-2]}-{type_str[i-1]}-{type_str[i]}"
        if triplet in patterns:
            info = patterns[triplet]
            signal_tuples.append((i, triplet, info['direction'], info['tp_pct'], info['sl_pct']))

    return signal_tuples


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        mom_threshold, mom_lookback, mom_cooldown, momentum_enabled,
                        n_folds=3):
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

        oos_trades, _ = portfolio_sim_momentum(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            mom_threshold=mom_threshold,
            mom_lookback=mom_lookback,
            mom_cooldown=mom_cooldown,
            momentum_enabled=momentum_enabled,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("MOMENTUM GUARD PARAMETER SWEEP STUDY (v1.45.0 baseline)")
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
    print(f"  Loaded {n_bars} bars, neutral window: {neutral_start}-{neutral_end} "
          f"({neutral_end - neutral_start} bars, {(neutral_end - neutral_start) // BARS_PER_DAY}d)")

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

    # Build signal index
    signal_tuples = build_signal_index(opens, type_codes, n_bars, atr_ratio, patterns)
    neutral_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in signal_tuples
                       if neutral_start <= b < neutral_end]
    print(f"  Signals: {len(neutral_signals)} in neutral window")

    # ==================== PHASE 1: Full grid IS sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: Momentum Guard Parameter Sweep (IS, full 130 patterns)")
    print("=" * 90)

    configs = []
    # OFF baseline
    configs.append(('OFF', None, None, None, False))
    # Grid
    for th, lb, cd in product(THRESHOLDS, LOOKBACKS, COOLDOWNS):
        label = f"th{th}_lb{lb}_cd{cd}"
        configs.append((label, th, lb, cd, True))

    results_is = {}
    print(f"\n{'Config':<25} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7} {'Blocked':>8}")
    print("-" * 80)

    for label, th, lb, cd, enabled in configs:
        trades, blocked = portfolio_sim_momentum(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mom_threshold=th if th else 1.0,
            mom_lookback=lb if lb else 6,
            mom_cooldown=cd if cd else 6,
            momentum_enabled=enabled,
        )
        stats = calc_stats(trades)
        results_is[label] = {
            'pnl': round(stats['pnl'], 1),
            'mdd': round(stats['mdd'], 2),
            'pnl_mdd': round(stats['pnl'] / max(stats['mdd'], 0.01), 1),
            'wr': round(stats['wr'], 1),
            'trades': stats['trades'],
            'blocked': blocked,
        }

        marker = ' ← current' if label == 'th1.0_lb6_cd6' else ''
        marker = marker or (' ← OFF' if label == 'OFF' else '')
        print(f"{label:<25} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{stats['pnl'] / max(stats['mdd'], 0.01):>8.1f} {stats['wr']:>6.1f} "
              f"{stats['trades']:>7} {blocked:>8}{marker}")

    # Rank by PnL/MDD
    ranked = sorted(results_is.items(), key=lambda x: -x[1]['pnl_mdd'])

    print(f"\n{'Rank':<6} {'Config':<25} {'PnL/MDD':>8} {'PnL%':>8} {'MDD%':>8} {'Blocked':>8}")
    print("-" * 65)
    for i, (label, r) in enumerate(ranked[:15]):
        marker = ' ← current' if label == 'th1.0_lb6_cd6' else ''
        marker = marker or (' ← OFF' if label == 'OFF' else '')
        print(f"{i+1:<6} {label:<25} {r['pnl_mdd']:>8.1f} {r['pnl']:>+8.1f} {r['mdd']:>8.2f} "
              f"{r['blocked']:>8}{marker}")

    # Current rank
    current_rank = next(i+1 for i, (l, _) in enumerate(ranked) if l == 'th1.0_lb6_cd6')
    off_rank = next(i+1 for i, (l, _) in enumerate(ranked) if l == 'OFF')
    best_label = ranked[0][0]
    best_r = ranked[0][1]

    print(f"\nCurrent (th1.0_lb6_cd6) rank: {current_rank}/{len(ranked)}")
    print(f"OFF rank: {off_rank}/{len(ranked)}")
    print(f"Best: {best_label} (PnL/MDD {best_r['pnl_mdd']})")

    # ==================== PHASE 2: Top-5 WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 2: WF 3-fold Validation (top configs + current + OFF)")
    print("=" * 90)

    # Select top 5 unique + current + OFF
    wf_candidates = []
    seen = set()
    for label, r in ranked:
        if label not in seen and len(wf_candidates) < 5:
            wf_candidates.append(label)
            seen.add(label)
    if 'th1.0_lb6_cd6' not in seen:
        wf_candidates.append('th1.0_lb6_cd6')
    if 'OFF' not in seen:
        wf_candidates.append('OFF')

    wf_results = {}
    print(f"\n{'Config':<25} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} {'WF':>6}")
    print("-" * 75)

    for label in wf_candidates:
        cfg = next((c for c in configs if c[0] == label), None)
        if cfg is None:
            continue
        _, th, lb, cd, enabled = cfg

        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mom_threshold=th if th else 1.0,
            mom_lookback=lb if lb else 6,
            mom_cooldown=cd if cd else 6,
            momentum_enabled=enabled,
        )

        if len(folds) == 3:
            avg = np.mean(folds)
            mn = min(folds)
            n_pass = sum(1 for f in folds if f > 0)
            wf_str = f"{n_pass}/3 {'P' if n_pass == 3 else 'F'}"
        else:
            avg = mn = 0
            wf_str = "N/A"

        wf_results[label] = {
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1),
            'min': round(mn, 1),
            'n_pass': n_pass if len(folds) == 3 else 0,
        }

        marker = ' ← current' if label == 'th1.0_lb6_cd6' else ''
        marker = marker or (' ← OFF' if label == 'OFF' else '')
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:<25} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    current_is = results_is['th1.0_lb6_cd6']
    current_wf = wf_results.get('th1.0_lb6_cd6', {})
    off_is = results_is['OFF']
    off_wf = wf_results.get('OFF', {})

    print(f"\nCurrent (th1.0/lb6/cd6):")
    print(f"  IS PnL/MDD: {current_is['pnl_mdd']}, blocked: {current_is['blocked']}")
    print(f"  OOS avg: {current_wf.get('avg', 'N/A')}, min: {current_wf.get('min', 'N/A')}")

    print(f"\nOFF baseline:")
    print(f"  IS PnL/MDD: {off_is['pnl_mdd']}, blocked: {off_is['blocked']}")
    print(f"  OOS avg: {off_wf.get('avg', 'N/A')}, min: {off_wf.get('min', 'N/A')}")

    # Find best WF-passing config
    best_wf = None
    best_wf_label = None
    for label in wf_candidates:
        r = wf_results.get(label, {})
        if r.get('n_pass', 0) == 3:
            if best_wf is None or r['avg'] > best_wf['avg']:
                best_wf = r
                best_wf_label = label

    if best_wf_label:
        best_is_r = results_is[best_wf_label]
        print(f"\nBest WF-passing: {best_wf_label}")
        print(f"  IS PnL/MDD: {best_is_r['pnl_mdd']}, blocked: {best_is_r['blocked']}")
        print(f"  OOS avg: {best_wf['avg']}, min: {best_wf['min']}")

        if best_wf_label == 'th1.0_lb6_cd6':
            print("\n>>> RECOMMEND: KEEP current parameters (already optimal)")
        elif best_wf_label == 'OFF':
            print("\n>>> RECOMMEND: Consider disabling Momentum Guard (OFF is best)")
        else:
            delta_is = best_is_r['pnl_mdd'] - current_is['pnl_mdd']
            delta_oos = best_wf['avg'] - current_wf.get('avg', 0)
            print(f"\n  IS PnL/MDD delta: {delta_is:+.1f}")
            print(f"  OOS avg delta: {delta_oos:+.1f}%")
            if delta_oos > 5:
                print(f"\n>>> RECOMMEND: Switch to {best_wf_label}")
            else:
                print(f"\n>>> RECOMMEND: Marginal improvement ({delta_oos:+.1f}%), keep current")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.45.0',
        'is_results': results_is,
        'is_ranking': [label for label, _ in ranked],
        'wf_results': wf_results,
        'current_rank': current_rank,
        'off_rank': off_rank,
        'best_wf_label': best_wf_label,
        'recommended': best_wf_label,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
