#!/usr/bin/env python3
"""
Position Sizing Formula Study (v1.53.0 baseline)
==================================================

Compare sizing approaches for N=9 multi-position portfolio:
  H0: 1/N uniform (current) — each slot gets 100/9 = 11.1%
  H1: Edge-weighted — size proportional to pattern's historical edge
  H2: WR-weighted — size proportional to pattern's WR excess
  H3: Inverse-SL — size inversely proportional to SL distance (risk-parity)
  H4: Half Kelly — f* = (WR/R - (1-WR)) / 2, clamp [min_size, 2/N]
  H5: Tiered — 3 tiers based on edge rank (top=1.5x, mid=1.0x, low=0.6x)

Phase 1: IS comparison (full neutral window)
Phase 2: Sensitivity to WR degradation (simulate live-like WR)
Phase 3: WF 3-fold validation

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

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'position_sizing_study.json')

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


def compute_pattern_stats(signal_tuples, opens, highs, lows, n_bars,
                          atr_ratio, start_bar, end_bar):
    """Compute per-pattern WR and edge from IS data for sizing calibration."""
    from collections import defaultdict
    pat_trades = defaultdict(list)

    for s_bar, pat, direction, tp_pct, sl_pct in signal_tuples:
        if not (start_bar <= s_bar < end_bar):
            continue
        entry_bar = s_bar + 1
        if entry_bar >= n_bars:
            continue
        entry_p = opens[entry_bar]
        if entry_p <= 0:
            continue

        # ATR scaling
        if s_bar < len(atr_ratio) and not np.isnan(atr_ratio[s_bar]):
            r = clamp(atr_ratio[s_bar], ATR_LO, ATR_HI)
        else:
            r = 1.0
        eff_tp = tp_pct * r + SLIPPAGE_BUFFER
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

        # Simple single-trade backtest (no portfolio effects)
        for bar in range(entry_bar, min(entry_bar + TIMEOUT_BARS, end_bar)):
            if bar >= n_bars:
                break
            h_bar = highs[bar]
            l_bar = lows[bar]
            o_bar = opens[bar]

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
                    pnl = eff_tp * LEVERAGE - FEE_PCT * LEVERAGE
                else:
                    pnl = -eff_sl * LEVERAGE - FEE_PCT * LEVERAGE
                pat_trades[pat].append(pnl)
                break
            elif hit_tp:
                pnl = eff_tp * LEVERAGE - FEE_PCT * LEVERAGE
                pat_trades[pat].append(pnl)
                break
            elif hit_sl:
                pnl = -eff_sl * LEVERAGE - FEE_PCT * LEVERAGE
                pat_trades[pat].append(pnl)
                break

    # Compute stats
    stats = {}
    for pat, pnls in pat_trades.items():
        if len(pnls) < 5:
            continue
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls)
        avg_win = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
        avg_loss = abs(np.mean([p for p in pnls if p <= 0])) if wins < len(pnls) else 0
        edge = np.mean(pnls)
        stats[pat] = {
            'wr': wr, 'edge': edge, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'trades': len(pnls),
        }
    return stats


def compute_raw_mults(method, pat_stats, patterns_info):
    """Compute raw multipliers for all patterns, then normalize so mean = 1.0."""
    raw = {}

    for pat, info in pat_stats.items():
        if method == 'uniform':
            raw[pat] = 1.0

        elif method == 'edge_weighted':
            all_edges = [s['edge'] for s in pat_stats.values()]
            mean_edge = np.mean(all_edges) if all_edges else 1.0
            raw[pat] = info['edge'] / mean_edge if mean_edge > 0 else 1.0

        elif method == 'wr_weighted':
            wr_excess = info['wr'] - 0.5
            raw[pat] = max(0.1, wr_excess * 5.0) if wr_excess > 0 else 0.1

        elif method == 'inverse_sl':
            p_info = patterns_info.get(pat)
            if p_info and p_info['sl_pct'] > 0:
                all_sls = [pi['sl_pct'] for pi in patterns_info.values()]
                med_sl = np.median(all_sls) if all_sls else 2.5
                raw[pat] = med_sl / p_info['sl_pct']
            else:
                raw[pat] = 1.0

        elif method == 'half_kelly':
            avg_loss = info['avg_loss']
            if avg_loss > 0:
                rr = info['avg_win'] / avg_loss
                kelly = info['wr'] - (1 - info['wr']) / rr
                raw[pat] = max(0.1, kelly / 2 * 3.0)
            else:
                raw[pat] = 1.0

        elif method == 'tiered':
            all_edges = sorted([(p, s['edge']) for p, s in pat_stats.items()],
                              key=lambda x: -x[1])
            n = len(all_edges)
            rank = next((i for i, (p, _) in enumerate(all_edges) if p == pat), n)
            if rank < n // 3:
                raw[pat] = 1.5
            elif rank < 2 * n // 3:
                raw[pat] = 1.0
            else:
                raw[pat] = 0.6

    if not raw:
        return {}

    # Normalize so mean = 1.0 (fair comparison: same total allocation)
    mean_raw = np.mean(list(raw.values()))
    if mean_raw <= 0:
        return {p: 1.0 for p in raw}
    normalized = {p: clamp(v / mean_raw, 0.3, 2.5) for p, v in raw.items()}

    # Re-normalize after clamping
    mean_norm = np.mean(list(normalized.values()))
    if mean_norm > 0:
        normalized = {p: v / mean_norm for p, v in normalized.items()}

    return normalized


# Cache for precomputed multipliers
_cached_mults = {}


def get_size_mult(method, pat, pat_stats, patterns_info, base_size_pct):
    """Return normalized size multiplier for a pattern."""
    if method == 'uniform':
        return 1.0

    # Use cached multipliers
    cache_key = id(pat_stats)  # changes when pat_stats dict changes
    if (method, cache_key) not in _cached_mults:
        _cached_mults[(method, cache_key)] = compute_raw_mults(
            method, pat_stats, patterns_info
        )

    mults = _cached_mults.get((method, cache_key), {})
    return mults.get(pat, 1.0)


def portfolio_sim_sizing(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    sizing_method='uniform', pat_stats=None, patterns_info=None,
):
    """Portfolio sim v1.53.0 with configurable position sizing."""
    fee = FEE_PCT * LEVERAGE
    base_size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    if pat_stats is None:
        pat_stats = {}
    if patterns_info is None:
        patterns_info = {}

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

            pnl_portfolio = pnl * (base_size_pct / 100) * sm
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

            # Position sizing
            sizing_mult = get_size_mult(sizing_method, pat, pat_stats,
                                        patterns_info, base_size_pct)

            # MDD sizing
            sm = sizing_mult
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
            'pnl_portfolio': pnl * (base_size_pct / 100) * sm,
        })

    return trades


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        sizing_method='uniform', patterns_info=None, n_folds=3):
    """WF with IS-calibrated pattern stats for each fold."""
    total_bars = neutral_end - neutral_start
    fold_size = total_bars // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        is_start = neutral_start
        is_end = neutral_start + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, neutral_end)
        if oos_start >= oos_end:
            continue

        # Calibrate pattern stats from IS period
        is_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in signal_tuples
                      if is_start <= b < is_end]
        pat_stats = compute_pattern_stats(
            is_signals, opens, highs, lows, n_bars, atr_ratio, is_start, is_end
        )

        # Run OOS with calibrated stats
        oos_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in signal_tuples
                       if oos_start <= b < oos_end]
        oos_trades = portfolio_sim_sizing(
            oos_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            sizing_method=sizing_method, pat_stats=pat_stats,
            patterns_info=patterns_info,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("POSITION SIZING FORMULA STUDY (v1.53.0 baseline)")
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
    patterns_info = {}
    for pat_name in pats_raw.get('long', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        patterns_info[pat_name] = {'direction': 'LONG', 'tp_pct': tp_sl[0], 'sl_pct': tp_sl[1]}
    for pat_name in pats_raw.get('short', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        patterns_info[pat_name] = {'direction': 'SHORT', 'tp_pct': tp_sl[0], 'sl_pct': tp_sl[1]}
    print(f"  Loaded {len(patterns_info)} patterns")

    # Build signals
    signal_tuples = []
    for i in range(2, n_bars):
        triplet = f"{type_codes[i-2]}-{type_codes[i-1]}-{type_codes[i]}"
        if triplet in patterns_info:
            info = patterns_info[triplet]
            signal_tuples.append((i, triplet, info['direction'], info['tp_pct'], info['sl_pct']))
    neutral_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in signal_tuples
                       if neutral_start <= b < neutral_end]
    print(f"  Signals: {len(neutral_signals)} in neutral window")

    # Compute full-period pattern stats for IS comparison
    print("\nComputing pattern statistics...")
    pat_stats = compute_pattern_stats(
        neutral_signals, opens, highs, lows, n_bars, atr_ratio,
        neutral_start, neutral_end
    )
    print(f"  {len(pat_stats)} patterns with sufficient trades")

    # Show sizing distribution for each method
    methods = ['uniform', 'edge_weighted', 'wr_weighted', 'inverse_sl',
               'half_kelly', 'tiered']

    print("\nSizing multiplier distribution by method:")
    for method in methods:
        mults = [get_size_mult(method, p, pat_stats, patterns_info, 100/N_SLOTS)
                 for p in pat_stats]
        if mults:
            print(f"  {method:>15}: mean={np.mean(mults):.2f}, "
                  f"std={np.std(mults):.2f}, "
                  f"range=[{min(mults):.2f}, {max(mults):.2f}]")

    # ==================== PHASE 1: IS Comparison ====================
    print("\n" + "=" * 90)
    print("PHASE 1: IS Comparison (full neutral window)")
    print("=" * 90)

    print(f"\n{'Method':>15} {'PnL%':>10} {'MDD%':>8} {'P/M':>10} {'WR%':>6} "
          f"{'Trades':>7}")
    print("-" * 65)

    p1_results = {}
    for method in methods:
        trades = portfolio_sim_sizing(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            sizing_method=method, pat_stats=pat_stats, patterns_info=patterns_info,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p1_results[method] = {
            'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
            'pm': round(pm, 1), 'wr': round(stats['wr'], 1),
            'trades': stats['trades'],
        }
        marker = ' ← current' if method == 'uniform' else ''
        print(f"{method:>15} {stats['pnl']:>+10.1f} {stats['mdd']:>8.2f} "
              f"{pm:>10.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    # ==================== PHASE 2: WR degradation sensitivity ====================
    print("\n" + "=" * 90)
    print("PHASE 2: Does sizing help when WR drops? (theoretical)")
    print("=" * 90)

    # Analyze per-direction PnL contribution
    print("\nPer-direction analysis (uniform baseline):")
    bl_trades = portfolio_sim_sizing(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end,
        sizing_method='uniform', pat_stats=pat_stats, patterns_info=patterns_info,
    )

    long_pnl = sum(t['pnl_portfolio'] for t in bl_trades if t['direction'] == 'LONG')
    short_pnl = sum(t['pnl_portfolio'] for t in bl_trades if t['direction'] == 'SHORT')
    long_trades = [t for t in bl_trades if t['direction'] == 'LONG']
    short_trades = [t for t in bl_trades if t['direction'] == 'SHORT']
    long_wr = (sum(1 for t in long_trades if t['pnl_slot'] > 0) / len(long_trades) * 100
               if long_trades else 0)
    short_wr = (sum(1 for t in short_trades if t['pnl_slot'] > 0) / len(short_trades) * 100
                if short_trades else 0)

    print(f"  LONG:  {len(long_trades)} trades, WR {long_wr:.1f}%, PnL {long_pnl:+.1f}%")
    print(f"  SHORT: {len(short_trades)} trades, WR {short_wr:.1f}%, PnL {short_pnl:+.1f}%")

    # Compare size allocation to high-edge vs low-edge patterns
    print("\nEdge distribution of pattern stats:")
    edges = sorted([(p, s['edge']) for p, s in pat_stats.items()], key=lambda x: -x[1])
    n_pats = len(edges)
    if n_pats > 0:
        top_third = edges[:n_pats//3]
        bot_third = edges[2*n_pats//3:]
        print(f"  Top 1/3 edge: {np.mean([e for _,e in top_third]):.2f}%/trade "
              f"({len(top_third)} patterns)")
        print(f"  Bot 1/3 edge: {np.mean([e for _,e in bot_third]):.2f}%/trade "
              f"({len(bot_third)} patterns)")

    # ==================== PHASE 3: WF Validation ====================
    print("\n" + "=" * 90)
    print("PHASE 3: WF 3-fold validation")
    print("=" * 90)

    wf_results = {}
    bl_pm = p1_results['uniform']['pm']

    print(f"\n{'Method':>15} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} "
          f"{'IS P/M':>10} {'WF':>6}")
    print("-" * 80)

    for method in methods:
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            sizing_method=method, patterns_info=patterns_info,
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

        is_pm = p1_results[method]['pm']
        wf_results[method] = {
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pnl_mdd': is_pm,
        }

        fold_strs = [f"{f:>+8.1f}" for f in folds]
        marker = ' ← current' if method == 'uniform' else ''
        print(f"{method:>15} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} "
              f"{is_pm:>10.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    bl_wf = wf_results.get('uniform', {})
    bl_min = bl_wf.get('min', -999)
    print(f"\nBaseline (uniform 1/N):")
    print(f"  IS PnL/MDD: {bl_pm:.1f}")
    print(f"  OOS folds: {bl_wf.get('folds', [])}")
    print(f"  OOS avg: {bl_wf.get('avg', 'N/A')}, min: {bl_wf.get('min', 'N/A')}")

    best_label = None
    best_min = bl_min
    for method, wr in wf_results.items():
        if method == 'uniform':
            continue
        if wr.get('n_pass', 0) == 3 and wr['min'] > best_min:
            best_min = wr['min']
            best_label = method

    if best_label:
        best_wf = wf_results[best_label]
        delta_min = best_wf['min'] - bl_min
        print(f"\nBest WF-passing above baseline: {best_label}")
        print(f"  IS PnL/MDD: {best_wf['is_pnl_mdd']}")
        print(f"  OOS avg: {best_wf['avg']}, min: {best_wf['min']}")
        print(f"  Min fold delta: {delta_min:+.1f}%")

        if delta_min > 5:
            verdict = 'ADOPT'
            print(f"\n>>> RECOMMEND: ADOPT {best_label}")
        elif delta_min > 2:
            verdict = 'MARGINAL'
            print(f"\n>>> RECOMMEND: MARGINAL, consider {best_label}")
        else:
            verdict = 'KEEP'
            print(f"\n>>> RECOMMEND: KEEP uniform (no meaningful improvement)")
    else:
        verdict = 'KEEP'
        print("\n>>> No method improves on uniform min fold")
        print(">>> RECOMMEND: KEEP uniform 1/N sizing")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.53.0',
        'phase1_is': p1_results,
        'wf_results': wf_results,
        'verdict': verdict,
        'recommended': best_label or 'uniform',
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
