#!/usr/bin/env python3
"""
Counter-Regime Direction Cap Study (v1.52.0 baseline)
=====================================================

Live performance: SHORT WR 41.2% (-21.70%), 7 SHORT loss clusters in 8.6 days.
Root cause: BTC rally → multiple counter-regime SHORTs → simultaneous SL.

Current: direction_cap=7 applies uniformly regardless of regime.
Hypothesis: Separate caps for with-regime vs counter-regime positions.

Phase 1: Counter-regime cap sweep (1-7, with-regime=7 fixed)
Phase 2: With-regime cap interaction
Phase 3: Combined with momentum guard tightening
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
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'counter_regime_cap_study.json')

# v1.52.0 production constants
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
MOM_THRESHOLD = 1.5
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12
ATR_LO = 0.5
ATR_HI = 1.5
TIMEOUT_BARS = 288
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3
AGG_COUNTER_CAP = 8.0
AGG_WITH_CAP = 15.0


def portfolio_sim(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    with_dir_cap=7, counter_dir_cap=7,
    mom_threshold=MOM_THRESHOLD, mom_lookback=MOM_LOOKBACK,
    mom_cooldown=MOM_COOLDOWN,
):
    """Portfolio sim with separate with/counter regime direction caps."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    # Cluster tracking
    sl_cluster_events = []
    current_cluster = []

    sig_by_bar = {}
    for s_bar, pat, direction, tp, sl in signal_tuples:
        if start_bar <= s_bar < end_bar:
            sig_by_bar.setdefault(s_bar, []).append((pat, direction, tp, sl))

    for bar in range(start_bar, end_bar):
        if bar >= n_bars - 1:
            break

        closed_slots = set()
        sl_exits = []

        h_bar = highs[bar]
        l_bar = lows[bar]
        o_bar = opens[bar]

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

        # Cascade SL tightening
        if sl_exits:
            for sl_pos in sl_exits:
                sl_dir = sl_pos['direction']
                for pos in positions:
                    if pos['direction'] == sl_dir:
                        pos['eff_sl_pct'] *= CASCADE_MULT

        # Track SL clusters (3+ same-bar same-direction)
        if len(sl_exits) >= 3:
            dirs = [s['direction'] for s in sl_exits]
            for d in set(dirs):
                cnt = dirs.count(d)
                if cnt >= 3:
                    loss_sum = sum(t['pnl_portfolio'] for t in trades
                                  if t['exit_bar'] == bar and t['reason'] == 'SL'
                                  and t['direction'] == d)
                    sl_cluster_events.append({
                        'bar': bar, 'direction': d, 'count': cnt,
                        'loss_pct': round(loss_sum, 2),
                    })

        equity_delta = sum(t['pnl_portfolio'] for t in trades if t['exit_bar'] == bar)
        equity *= (1 + equity_delta / 100)
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard
        if bar >= mom_lookback:
            pa = closes[bar - mom_lookback]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > mom_threshold:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'], bar + mom_cooldown)
                elif pct < -mom_threshold:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'], bar + mom_cooldown)

        if bar not in sig_by_bar:
            continue

        # Determine regime
        slope = ema_slope[bar] if bar < len(ema_slope) else 0
        is_uptrend = slope > 0

        for pat, direction, tp_pct, sl_pct in sig_by_bar[bar]:
            if len(positions) >= N_SLOTS:
                continue

            # Regime-aware direction cap
            is_counter = ((direction == 'SHORT' and is_uptrend) or
                          (direction == 'LONG' and not is_uptrend))
            dir_cap = counter_dir_cap if is_counter else with_dir_cap

            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= dir_cap:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar_i = bar + 1
            if entry_bar_i >= n_bars:
                continue
            entry_price = opens[entry_bar_i]
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

            # AggRisk cap
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
                'entry_bar': entry_bar_i,
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

    return trades, sl_cluster_events


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, ns, ne,
                        with_dc, counter_dc,
                        mom_threshold=MOM_THRESHOLD,
                        mom_lookback=MOM_LOOKBACK,
                        mom_cooldown=MOM_COOLDOWN,
                        n_folds=3):
    total_bars = ne - ns
    fold_size = total_bars // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        oos_start = ns + fold_size * (fold + 1)
        oos_end = min(oos_start + fold_size, ne)
        if oos_start >= oos_end:
            continue
        trades, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            with_dir_cap=with_dc, counter_dir_cap=counter_dc,
            mom_threshold=mom_threshold, mom_lookback=mom_lookback,
            mom_cooldown=mom_cooldown,
        )
        stats = calc_stats(trades)
        results.append(stats['pnl'])
    return results


def run_sim(label, signal_tuples, opens, highs, lows, closes, type_codes,
            n_bars, atr_ratio, ema_slope, ns, ne,
            with_dc=7, counter_dc=7, **kwargs):
    trades, clusters = portfolio_sim(
        signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        with_dir_cap=with_dc, counter_dir_cap=counter_dc, **kwargs,
    )
    stats = calc_stats(trades)
    pm = stats['pnl'] / max(stats['mdd'], 0.01)

    # Direction breakdown
    long_trades = [t for t in trades if t['direction'] == 'LONG' and t['reason'] != 'FORCE_CLOSE']
    short_trades = [t for t in trades if t['direction'] == 'SHORT' and t['reason'] != 'FORCE_CLOSE']
    long_wr = (sum(1 for t in long_trades if t['pnl_slot'] >= 0) / len(long_trades) * 100) if long_trades else 0
    short_wr = (sum(1 for t in short_trades if t['pnl_slot'] >= 0) / len(short_trades) * 100) if short_trades else 0
    long_pnl = sum(t['pnl_portfolio'] for t in long_trades)
    short_pnl = sum(t['pnl_portfolio'] for t in short_trades)

    return {
        'label': label, 'pnl': round(stats['pnl'], 1),
        'mdd': round(stats['mdd'], 2), 'pnl_mdd': round(pm, 1),
        'wr': round(stats['wr'], 1), 'trades': stats['trades'],
        'long_trades': len(long_trades), 'short_trades': len(short_trades),
        'long_wr': round(long_wr, 1), 'short_wr': round(short_wr, 1),
        'long_pnl': round(long_pnl, 1), 'short_pnl': round(short_pnl, 1),
        'clusters': len(clusters),
        'cluster_loss': round(sum(c['loss_pct'] for c in clusters), 1),
    }


def main():
    print("=" * 100)
    print("COUNTER-REGIME DIRECTION CAP STUDY (v1.52.0 baseline)")
    print("=" * 100)

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
    ns, ne = find_neutral_window(closes)
    print(f"  Loaded {n_bars} bars, neutral: {ns}-{ne}")

    # Load patterns
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pats_raw = pat_data['patterns']
    tpsl = pat_data.get('patterns_tpsl', {})
    patterns = {}
    for pn in pats_raw.get('long', []):
        tp_sl = tpsl.get(pn, [2.0, 3.0])
        patterns[pn] = {'direction': 'LONG', 'tp_pct': tp_sl[0], 'sl_pct': tp_sl[1]}
    for pn in pats_raw.get('short', []):
        tp_sl = tpsl.get(pn, [2.0, 3.0])
        patterns[pn] = {'direction': 'SHORT', 'tp_pct': tp_sl[0], 'sl_pct': tp_sl[1]}
    print(f"  Loaded {len(patterns)} patterns")

    # Build signals
    signal_tuples = []
    for i in range(2, n_bars):
        triplet = f"{type_codes[i-2]}-{type_codes[i-1]}-{type_codes[i]}"
        if triplet in patterns:
            info = patterns[triplet]
            signal_tuples.append((i, triplet, info['direction'], info['tp_pct'], info['sl_pct']))
    neutral_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in signal_tuples
                       if ns <= b < ne]
    print(f"  Signals: {len(neutral_signals)} in neutral window")

    args = (neutral_signals, opens, highs, lows, closes, type_codes,
            n_bars, atr_ratio, ema_slope, ns, ne)

    # ==================== PHASE 1: Counter-regime cap sweep ====================
    print("\n" + "=" * 100)
    print("PHASE 1: Counter-regime direction cap sweep (with_cap=7 fixed)")
    print("=" * 100)

    counter_caps = [1, 2, 3, 4, 5, 6, 7]
    print(f"\n{'CtrCap':>7} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'L_WR':>6} {'S_WR':>6} {'L_PnL':>8} {'S_PnL':>8} "
          f"{'Clust':>6} {'CLoss':>7}")
    print("-" * 110)

    p1_results = {}
    for cc in counter_caps:
        r = run_sim(f"cc{cc}", *args, with_dc=7, counter_dc=cc)
        p1_results[cc] = r
        marker = ' ← current' if cc == 7 else ''
        print(f"{cc:>7} {r['pnl']:>+8.1f} {r['mdd']:>8.2f} {r['pnl_mdd']:>8.1f} "
              f"{r['wr']:>6.1f} {r['trades']:>7} {r['long_wr']:>6.1f} {r['short_wr']:>6.1f} "
              f"{r['long_pnl']:>+8.1f} {r['short_pnl']:>+8.1f} "
              f"{r['clusters']:>6} {r['cluster_loss']:>+7.1f}{marker}")

    # ==================== PHASE 2: With-regime cap interaction ====================
    print("\n" + "=" * 100)
    print("PHASE 2: 2D Grid — with_cap × counter_cap")
    print("=" * 100)

    # Top counter caps from Phase 1
    ranked_cc = sorted(p1_results.items(), key=lambda x: x[1]['pnl_mdd'], reverse=True)
    top_cc = [k for k, _ in ranked_cc[:4]]
    with_caps = [5, 6, 7, 9]

    print(f"\n  Counter caps to test: {top_cc}")
    print(f"  With caps to test: {with_caps}")

    print(f"\n{'Config':>12} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'L_PnL':>8} {'S_PnL':>8} {'Clust':>6}")
    print("-" * 85)

    p2_results = {}
    for wc in with_caps:
        for cc in top_cc:
            if cc > wc:
                continue  # counter_cap should be <= with_cap
            label = f"w{wc}_c{cc}"
            r = run_sim(label, *args, with_dc=wc, counter_dc=cc)
            p2_results[label] = {**r, 'with_dc': wc, 'counter_dc': cc}
            marker = ' ← current' if wc == 7 and cc == 7 else ''
            print(f"{label:>12} {r['pnl']:>+8.1f} {r['mdd']:>8.2f} {r['pnl_mdd']:>8.1f} "
                  f"{r['wr']:>6.1f} {r['trades']:>7} "
                  f"{r['long_pnl']:>+8.1f} {r['short_pnl']:>+8.1f} "
                  f"{r['clusters']:>6}{marker}")

    # ==================== PHASE 3: Combined with momentum guard ====================
    print("\n" + "=" * 100)
    print("PHASE 3: Combined — top counter cap + momentum guard tightening")
    print("=" * 100)

    # Get best from Phase 2
    best_p2 = max(p2_results.items(), key=lambda x: x[1]['pnl_mdd'])
    best_wc = best_p2[1]['with_dc']
    best_cc = best_p2[1]['counter_dc']
    print(f"\n  Best Phase 2: {best_p2[0]} (PnL/MDD {best_p2[1]['pnl_mdd']})")

    # Momentum guard variations
    mom_configs = [
        ('current',    1.5, 3, 12),
        ('tighter',    1.0, 3, 12),
        ('fast_cool',  1.5, 3, 18),
        ('aggressive', 1.0, 3, 18),
        ('wide_look',  1.5, 6, 12),
        ('ultra',      0.8, 3, 24),
    ]

    print(f"\n{'Config':>18} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'S_WR':>6} {'S_PnL':>8} {'Clust':>6}")
    print("-" * 90)

    p3_results = {}
    for name, mt, ml, mc in mom_configs:
        # Use best counter cap from Phase 2
        label = f"{name}_w{best_wc}c{best_cc}"
        r = run_sim(label, *args,
                    with_dc=best_wc, counter_dc=best_cc,
                    mom_threshold=mt, mom_lookback=ml, mom_cooldown=mc)
        p3_results[label] = {
            **r, 'with_dc': best_wc, 'counter_dc': best_cc,
            'mom_threshold': mt, 'mom_lookback': ml, 'mom_cooldown': mc,
        }
        print(f"{label:>18} {r['pnl']:>+8.1f} {r['mdd']:>8.2f} {r['pnl_mdd']:>8.1f} "
              f"{r['wr']:>6.1f} {r['trades']:>7} {r['short_wr']:>6.1f} "
              f"{r['short_pnl']:>+8.1f} {r['clusters']:>6}")

    # Also test with current direction caps (w7/c7) for comparison
    print(f"\n  --- Same momentum configs with current caps (w7/c7) ---")
    for name, mt, ml, mc in mom_configs:
        if name == 'current':
            continue  # already tested as baseline
        label = f"{name}_w7c7"
        r = run_sim(label, *args,
                    with_dc=7, counter_dc=7,
                    mom_threshold=mt, mom_lookback=ml, mom_cooldown=mc)
        p3_results[label] = {
            **r, 'with_dc': 7, 'counter_dc': 7,
            'mom_threshold': mt, 'mom_lookback': ml, 'mom_cooldown': mc,
        }
        print(f"{label:>18} {r['pnl']:>+8.1f} {r['mdd']:>8.2f} {r['pnl_mdd']:>8.1f} "
              f"{r['wr']:>6.1f} {r['trades']:>7} {r['short_wr']:>6.1f} "
              f"{r['short_pnl']:>+8.1f} {r['clusters']:>6}")

    # ==================== PHASE 4: WF Validation ====================
    print("\n" + "=" * 100)
    print("PHASE 4: WF 3-fold Validation")
    print("=" * 100)

    # Collect candidates: top-3 from each phase + current
    all_configs = {}
    all_configs['CURRENT_w7c7'] = {'with_dc': 7, 'counter_dc': 7,
                                    'mom_threshold': MOM_THRESHOLD,
                                    'mom_lookback': MOM_LOOKBACK,
                                    'mom_cooldown': MOM_COOLDOWN}

    # Best from Phase 1
    best_p1 = max(p1_results.items(), key=lambda x: x[1]['pnl_mdd'])
    all_configs[f'P1_best_cc{best_p1[0]}'] = {
        'with_dc': 7, 'counter_dc': best_p1[0],
        'mom_threshold': MOM_THRESHOLD, 'mom_lookback': MOM_LOOKBACK,
        'mom_cooldown': MOM_COOLDOWN,
    }

    # Best from Phase 2
    all_configs[f'P2_best_{best_p2[0]}'] = {
        'with_dc': best_wc, 'counter_dc': best_cc,
        'mom_threshold': MOM_THRESHOLD, 'mom_lookback': MOM_LOOKBACK,
        'mom_cooldown': MOM_COOLDOWN,
    }

    # Top-3 from Phase 3
    p3_ranked = sorted(p3_results.items(), key=lambda x: x[1]['pnl_mdd'], reverse=True)
    for i, (label, r) in enumerate(p3_ranked[:3]):
        all_configs[f'P3_{label}'] = {
            'with_dc': r['with_dc'], 'counter_dc': r['counter_dc'],
            'mom_threshold': r['mom_threshold'], 'mom_lookback': r['mom_lookback'],
            'mom_cooldown': r['mom_cooldown'],
        }

    # Also test: counter_dc=3 (strong constraint)
    if 3 not in [v['counter_dc'] for v in all_configs.values()]:
        all_configs['cc3_w7'] = {'with_dc': 7, 'counter_dc': 3,
                                  'mom_threshold': MOM_THRESHOLD,
                                  'mom_lookback': MOM_LOOKBACK,
                                  'mom_cooldown': MOM_COOLDOWN}

    print(f"\n  {len(all_configs)} configs to validate")
    print(f"\n{'Config':>25} {'F1':>8} {'F2':>8} {'F3':>8} "
          f"{'Avg':>8} {'Min':>8} {'WF':>6} {'IS P/M':>8}")
    print("-" * 90)

    p4_results = {}
    for label, cfg in all_configs.items():
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, ns, ne,
            with_dc=cfg['with_dc'], counter_dc=cfg['counter_dc'],
            mom_threshold=cfg['mom_threshold'],
            mom_lookback=cfg['mom_lookback'],
            mom_cooldown=cfg['mom_cooldown'],
        )

        # Get IS stats
        is_trades, _ = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, ns, ne,
            with_dir_cap=cfg['with_dc'], counter_dir_cap=cfg['counter_dc'],
            mom_threshold=cfg['mom_threshold'],
            mom_lookback=cfg['mom_lookback'],
            mom_cooldown=cfg['mom_cooldown'],
        )
        is_stats = calc_stats(is_trades)
        is_pm = is_stats['pnl'] / max(is_stats['mdd'], 0.01)

        if len(folds) == 3:
            avg = np.mean(folds)
            mn = min(folds)
            n_pass = sum(1 for f in folds if f > 0)
            wf_str = f"{n_pass}/3 {'P' if n_pass == 3 else 'F'}"
        else:
            avg = mn = 0
            n_pass = 0
            wf_str = "N/A"

        p4_results[label] = {
            **cfg,
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pnl_mdd': round(is_pm, 1),
        }

        marker = ' ← current' if 'CURRENT' in label else ''
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:>25} {''.join(fold_strs)} "
              f"{avg:>+8.1f} {mn:>+8.1f} {wf_str:>6} {is_pm:>8.1f}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 100)
    print("SYNTHESIS")
    print("=" * 100)

    current = p1_results[7]
    print(f"\n  Current (w7/c7): PnL {current['pnl']:+.1f}%, MDD {current['mdd']:.2f}%, "
          f"P/M {current['pnl_mdd']:.1f}, Clusters {current['clusters']}")

    # Best overall (WF PASS, highest min fold)
    wf_pass = {k: v for k, v in p4_results.items() if v.get('n_pass', 0) == 3}
    if wf_pass:
        best = max(wf_pass.items(), key=lambda x: x[1]['min'])
        print(f"\n  Best WF-passing (min fold): {best[0]}")
        print(f"    with_dc={best[1]['with_dc']}, counter_dc={best[1]['counter_dc']}")
        print(f"    mom: threshold={best[1]['mom_threshold']}, "
              f"lookback={best[1]['mom_lookback']}, cooldown={best[1]['mom_cooldown']}")
        print(f"    IS PnL/MDD: {best[1]['is_pnl_mdd']}")
        print(f"    OOS: avg {best[1]['avg']:+.1f}%, min {best[1]['min']:+.1f}%")

        current_wf = p4_results.get('CURRENT_w7c7', {})
        current_min = current_wf.get('min', 0)
        delta_min = best[1]['min'] - current_min
        print(f"    vs current: min fold {delta_min:+.1f}%")

        if best[0] == 'CURRENT_w7c7':
            print("\n>>> RECOMMEND: KEEP current (w7/c7)")
        elif delta_min > 5:
            print(f"\n>>> RECOMMEND: ADOPT {best[0]} — min fold +{delta_min:.1f}%")
        else:
            print(f"\n>>> RECOMMEND: Marginal ({delta_min:+.1f}%), KEEP current")
    else:
        print("\n  No configs passed WF 3/3")
        print("\n>>> RECOMMEND: KEEP current (w7/c7)")

    # Cluster reduction analysis
    print(f"\n  === Cluster Reduction Analysis ===")
    for cc in [7, 5, 4, 3, 2, 1]:
        if cc in p1_results:
            r = p1_results[cc]
            print(f"    counter_cap={cc}: {r['clusters']} clusters, "
                  f"cluster_loss {r['cluster_loss']:+.1f}%, "
                  f"SHORT WR {r['short_wr']:.1f}%, SHORT PnL {r['short_pnl']:+.1f}%")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.52.0',
        'phase1_counter_sweep': {str(k): v for k, v in p1_results.items()},
        'phase2_grid': p2_results,
        'phase3_combined': p3_results,
        'phase4_wf': p4_results,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
