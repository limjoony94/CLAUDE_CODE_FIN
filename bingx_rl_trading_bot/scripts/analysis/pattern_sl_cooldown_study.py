#!/usr/bin/env python3
"""
Pattern SL Re-entry Cooldown Study (v1.53.0 baseline)
=====================================================

After pattern X exits via SL, block re-entry of the SAME pattern for N bars.
Hypothesis: cluster losses from the same pattern in trending markets can be
reduced by imposing a cooldown after SL exits.

Phase 1: Cooldown sweep (0/12/24/48/72/144/288 bars)
Phase 2: Direction-specific cooldown (only block same-direction re-entry)
Phase 3: Any-SL cooldown (any SL exit blocks ALL patterns in that direction)
Phase 4: WF 3-fold validation of top candidates

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
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'pattern_sl_cooldown_study.json')

# v1.53.0 production params
DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
MOM_THRESHOLD = 1.5
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12
ATR_LO = 0.5
ATR_HI = 1.5
TIMEOUT_BARS = 288
AGG_COUNTER = 8.0
AGG_WITH = 15.0
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3


def portfolio_sim_cooldown(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    pattern_cooldown_bars=0,       # cooldown after pattern SL (0=off)
    dir_only_cooldown=False,       # True: only block same-direction re-entry
    any_sl_dir_cooldown_bars=0,    # >0: any SL blocks ALL patterns in that direction
):
    """Portfolio sim v1.53.0 with configurable pattern SL cooldown."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    # Cooldown trackers
    # pattern_cooldown_until[pattern_key] = bar when cooldown expires
    # (or pattern_cooldown_until[(pattern_key, direction)] for dir_only)
    pattern_cooldown_until = {}
    # dir_cooldown_until[direction] = bar when any-SL direction cooldown expires
    dir_cooldown_until = {'LONG': -1, 'SHORT': -1}

    cooldown_blocks = 0
    dir_cooldown_blocks = 0

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
                # Apply pattern cooldown
                if pattern_cooldown_bars > 0:
                    if dir_only_cooldown:
                        key = (pos['pattern'], direction)
                    else:
                        key = pos['pattern']
                    pattern_cooldown_until[key] = max(
                        pattern_cooldown_until.get(key, -1),
                        bar + pattern_cooldown_bars
                    )
                # Apply any-SL direction cooldown
                if any_sl_dir_cooldown_bars > 0:
                    dir_cooldown_until[direction] = max(
                        dir_cooldown_until[direction],
                        bar + any_sl_dir_cooldown_bars
                    )

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

            # Pattern cooldown check
            if pattern_cooldown_bars > 0:
                if dir_only_cooldown:
                    key = (pat, direction)
                else:
                    key = pat
                if pattern_cooldown_until.get(key, -1) > bar:
                    cooldown_blocks += 1
                    continue

            # Any-SL direction cooldown check
            if any_sl_dir_cooldown_bars > 0:
                if dir_cooldown_until.get(direction, -1) > bar:
                    dir_cooldown_blocks += 1
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

            if bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                r = clamp(atr_ratio[bar], ATR_LO, ATR_HI)
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # AggRisk cap
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

    return trades, cooldown_blocks, dir_cooldown_blocks


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        pattern_cooldown_bars=0, dir_only_cooldown=False,
                        any_sl_dir_cooldown_bars=0, n_folds=3):
    total_bars = neutral_end - neutral_start
    fold_size = total_bars // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        is_end = neutral_start + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, neutral_end)
        if oos_start >= oos_end:
            continue

        oos_trades, _, _ = portfolio_sim_cooldown(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            pattern_cooldown_bars=pattern_cooldown_bars,
            dir_only_cooldown=dir_only_cooldown,
            any_sl_dir_cooldown_bars=any_sl_dir_cooldown_bars,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("PATTERN SL RE-ENTRY COOLDOWN STUDY (v1.53.0 baseline)")
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
    print(f"  Loaded {n_bars} bars, neutral: {neutral_start}-{neutral_end} "
          f"({neutral_end - neutral_start} bars)")

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
    print(f"  Loaded {len(patterns)} patterns ({len(pats_raw.get('long',[]))}L + "
          f"{len(pats_raw.get('short',[]))}S)")

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

    # ==================== BASELINE ====================
    print("\n" + "=" * 90)
    print("BASELINE (no cooldown)")
    print("=" * 90)

    trades_bl, _, _ = portfolio_sim_cooldown(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end,
    )
    bl_stats = calc_stats(trades_bl)
    bl_pm = bl_stats['pnl'] / max(bl_stats['mdd'], 0.01)
    print(f"  PnL: {bl_stats['pnl']:+.1f}%, MDD: {bl_stats['mdd']:.2f}%, "
          f"PnL/MDD: {bl_pm:.1f}, WR: {bl_stats['wr']:.1f}%, "
          f"Trades: {bl_stats['trades']}")

    # Count SL exits by pattern for context
    sl_trades = [t for t in trades_bl if t['reason'] == 'SL']
    pat_sl_counts = {}
    for t in sl_trades:
        pat_sl_counts[t['pattern']] = pat_sl_counts.get(t['pattern'], 0) + 1
    top_sl_pats = sorted(pat_sl_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n  SL exits: {len(sl_trades)} total")
    print(f"  Top SL patterns: {top_sl_pats[:5]}")

    # Cluster analysis: same pattern SL within N bars
    sl_by_pat = {}
    for t in sl_trades:
        sl_by_pat.setdefault(t['pattern'], []).append(t['exit_bar'])
    cluster_counts = {24: 0, 48: 0, 72: 0, 144: 0, 288: 0}
    for pat, bars in sl_by_pat.items():
        bars.sort()
        for i in range(1, len(bars)):
            gap = bars[i] - bars[i - 1]
            for thresh in cluster_counts:
                if gap <= thresh:
                    cluster_counts[thresh] += 1
    print(f"  Same-pattern SL clusters: {cluster_counts}")

    # ==================== PHASE 1: Pattern cooldown sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: Pattern Cooldown Sweep (same pattern blocked after SL)")
    print("=" * 90)

    cooldown_vals = [0, 6, 12, 24, 48, 72, 144, 288]
    print(f"\n{'Cooldown':>10} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Blocks':>7} {'ΔP/M':>8}")
    print("-" * 70)

    p1_results = {}
    for cd in cooldown_vals:
        trades, blocks, _ = portfolio_sim_cooldown(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            pattern_cooldown_bars=cd,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        delta_pm = pm - bl_pm
        p1_results[cd] = {
            'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
            'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
            'trades': stats['trades'], 'blocks': blocks,
        }
        marker = ' ← baseline' if cd == 0 else ''
        print(f"{cd:>10} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} "
              f"{blocks:>7} {delta_pm:>+8.1f}{marker}")

    # ==================== PHASE 2: Direction-specific cooldown ====================
    print("\n" + "=" * 90)
    print("PHASE 2: Direction-Specific Cooldown (same pattern+direction blocked)")
    print("=" * 90)

    print(f"\n{'Cooldown':>10} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Blocks':>7} {'ΔP/M':>8}")
    print("-" * 70)

    p2_results = {}
    for cd in cooldown_vals:
        if cd == 0:
            continue  # same as baseline
        trades, blocks, _ = portfolio_sim_cooldown(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            pattern_cooldown_bars=cd,
            dir_only_cooldown=True,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        delta_pm = pm - bl_pm
        p2_results[cd] = {
            'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
            'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
            'trades': stats['trades'], 'blocks': blocks,
        }
        print(f"{cd:>10} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} "
              f"{blocks:>7} {delta_pm:>+8.1f}")

    # ==================== PHASE 3: Any-SL direction cooldown ====================
    print("\n" + "=" * 90)
    print("PHASE 3: Any-SL Direction Cooldown (ANY SL blocks entire direction)")
    print("=" * 90)

    dir_cd_vals = [0, 6, 12, 24, 48, 72]
    print(f"\n{'DirCD':>10} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'DirBlks':>7} {'ΔP/M':>8}")
    print("-" * 70)

    p3_results = {}
    for cd in dir_cd_vals:
        if cd == 0:
            continue
        trades, _, dir_blocks = portfolio_sim_cooldown(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            any_sl_dir_cooldown_bars=cd,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        delta_pm = pm - bl_pm
        p3_results[cd] = {
            'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
            'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
            'trades': stats['trades'], 'dir_blocks': dir_blocks,
        }
        print(f"{cd:>10} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} "
              f"{dir_blocks:>7} {delta_pm:>+8.1f}")

    # ==================== PHASE 3b: Combo — pattern cd + dir cd ====================
    print("\n" + "=" * 90)
    print("PHASE 3b: Combo (pattern cooldown + any-SL dir cooldown)")
    print("=" * 90)

    # Select top pattern cooldowns from P1
    top_pat_cds = sorted(p1_results.keys(), key=lambda k: p1_results[k]['pm'], reverse=True)[:3]
    top_pat_cds = [c for c in top_pat_cds if c > 0]
    if not top_pat_cds:
        top_pat_cds = [48]
    top_dir_cds = sorted(p3_results.keys(), key=lambda k: p3_results[k]['pm'], reverse=True)[:3]
    if not top_dir_cds:
        top_dir_cds = [12]

    combo_results = {}
    print(f"\n{'Config':>16} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'P_Blk':>7} {'D_Blk':>7} {'ΔP/M':>8}")
    print("-" * 85)

    for pcd in top_pat_cds:
        for dcd in top_dir_cds:
            trades, p_blocks, d_blocks = portfolio_sim_cooldown(
                neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                pattern_cooldown_bars=pcd,
                any_sl_dir_cooldown_bars=dcd,
            )
            stats = calc_stats(trades)
            pm = stats['pnl'] / max(stats['mdd'], 0.01)
            delta_pm = pm - bl_pm
            label = f"p{pcd}_d{dcd}"
            combo_results[label] = {
                'pat_cd': pcd, 'dir_cd': dcd,
                'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
                'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
                'trades': stats['trades'], 'pat_blocks': p_blocks, 'dir_blocks': d_blocks,
            }
            print(f"{label:>16} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
                  f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} "
                  f"{p_blocks:>7} {d_blocks:>7} {delta_pm:>+8.1f}")

    # ==================== PHASE 4: WF 3-fold validation ====================
    print("\n" + "=" * 90)
    print("PHASE 4: WF 3-fold validation")
    print("=" * 90)

    # Collect all configs with PM > baseline
    wf_candidates = []

    # Baseline
    wf_candidates.append(('BASELINE', 0, False, 0))

    # Top P1 (any better)
    for cd in sorted(p1_results.keys(), key=lambda k: p1_results[k]['pm'], reverse=True):
        if cd > 0 and p1_results[cd]['pm'] > bl_pm - 5:
            wf_candidates.append((f'pat_{cd}', cd, False, 0))
        if len(wf_candidates) >= 4:
            break

    # Top P2
    for cd in sorted(p2_results.keys(), key=lambda k: p2_results[k]['pm'], reverse=True):
        if p2_results[cd]['pm'] > bl_pm - 5:
            wf_candidates.append((f'patdir_{cd}', cd, True, 0))
        if len(wf_candidates) >= 7:
            break

    # Top P3
    for cd in sorted(p3_results.keys(), key=lambda k: p3_results[k]['pm'], reverse=True):
        if p3_results[cd]['pm'] > bl_pm - 5:
            wf_candidates.append((f'anysl_{cd}', 0, False, cd))
        if len(wf_candidates) >= 9:
            break

    # Top combos
    for label in sorted(combo_results.keys(), key=lambda k: combo_results[k]['pm'], reverse=True):
        r = combo_results[label]
        if r['pm'] > bl_pm - 5:
            wf_candidates.append((label, r['pat_cd'], False, r['dir_cd']))
        if len(wf_candidates) >= 12:
            break

    # Deduplicate
    seen = set()
    unique_candidates = []
    for label, pcd, donly, dcd in wf_candidates:
        key = (pcd, donly, dcd)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((label, pcd, donly, dcd))

    wf_results = {}
    print(f"\n{'Config':>16} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} "
          f"{'IS P/M':>8} {'WF':>6}")
    print("-" * 80)

    for label, pcd, donly, dcd in unique_candidates:
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            pattern_cooldown_bars=pcd,
            dir_only_cooldown=donly,
            any_sl_dir_cooldown_bars=dcd,
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

        # Get IS PnL/MDD
        if label == 'BASELINE':
            is_pm = bl_pm
        elif label in combo_results:
            is_pm = combo_results[label]['pm']
        elif label.startswith('pat_'):
            is_pm = p1_results.get(pcd, {}).get('pm', 0)
        elif label.startswith('patdir_'):
            is_pm = p2_results.get(pcd, {}).get('pm', 0)
        elif label.startswith('anysl_'):
            is_pm = p3_results.get(dcd, {}).get('pm', 0)
        else:
            is_pm = 0

        wf_results[label] = {
            'pat_cd': pcd, 'dir_only': donly, 'dir_cd': dcd,
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pnl_mdd': round(is_pm, 1),
        }

        fold_strs = [f"{f:>+8.1f}" for f in folds]
        marker = ' ← baseline' if label == 'BASELINE' else ''
        print(f"{label:>16} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} "
              f"{is_pm:>8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    bl_wf = wf_results.get('BASELINE', {})
    print(f"\nBaseline:")
    print(f"  IS PnL/MDD: {bl_pm:.1f}")
    print(f"  OOS folds: {bl_wf.get('folds', [])}")
    print(f"  OOS avg: {bl_wf.get('avg', 'N/A')}, min: {bl_wf.get('min', 'N/A')}")

    # Best WF-passing by min fold (better than baseline min)
    bl_min = bl_wf.get('min', -999)
    best_label = None
    best_min = bl_min
    for label, wr in wf_results.items():
        if label == 'BASELINE':
            continue
        if wr.get('n_pass', 0) == 3 and wr['min'] > best_min:
            best_min = wr['min']
            best_label = label

    if best_label:
        best_wf = wf_results[best_label]
        print(f"\nBest WF-passing by min fold (above baseline): {best_label}")
        print(f"  IS PnL/MDD: {best_wf['is_pnl_mdd']}")
        print(f"  OOS avg: {best_wf['avg']}, min: {best_wf['min']}")
        delta_min = best_wf['min'] - bl_min
        print(f"  Min fold delta vs baseline: {delta_min:+.1f}%")

        if delta_min > 5:
            print(f"\n>>> RECOMMEND: ADOPT {best_label} (significant OOS improvement)")
            verdict = 'ADOPT'
        elif delta_min > 2:
            print(f"\n>>> RECOMMEND: MARGINAL improvement, consider {best_label}")
            verdict = 'MARGINAL'
        else:
            print(f"\n>>> RECOMMEND: KEEP baseline (no meaningful OOS improvement)")
            verdict = 'KEEP'
    else:
        print("\n>>> No config improves on baseline min fold")
        print(">>> RECOMMEND: KEEP baseline (no cooldown)")
        verdict = 'KEEP'
        best_label = None

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.53.0',
        'baseline': {
            'pnl': round(bl_stats['pnl'], 1), 'mdd': round(bl_stats['mdd'], 2),
            'pnl_mdd': round(bl_pm, 1), 'wr': round(bl_stats['wr'], 1),
            'trades': bl_stats['trades'],
            'sl_exits': len(sl_trades),
            'sl_clusters': cluster_counts,
        },
        'phase1_pattern_cooldown': p1_results,
        'phase2_dir_specific': p2_results,
        'phase3_any_sl_dir': p3_results,
        'phase3b_combo': combo_results,
        'wf_results': wf_results,
        'verdict': verdict,
        'recommended': best_label,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
