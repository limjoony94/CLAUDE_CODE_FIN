#!/usr/bin/env python3
"""
AggRisk Re-sweep Study (v1.48.0 baseline)
==========================================

Aggregate Risk Cap limits directional SL exposure.
Current: counter_cap=5.0%, with_cap=15.0% (v1.44.0).
Re-validate under v1.48.0 stack (ATR clamp_lo=0.5, timeout=288).

Phase 1: Counter cap sweep (2-8%, with=15 fixed)
Phase 2: With cap sweep (8-25%, counter=5 fixed)
Phase 3: 2D grid + OFF
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

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'aggrisk_resweep_study.json')

# v1.48.0 fixed params
DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
MOM_THRESHOLD = 1.0
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12
ATR_LO = 0.5
ATR_HI = 1.7
TIMEOUT_BARS = 288  # v1.48.0
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3


def portfolio_sim_agg(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    counter_cap=5.0, with_cap=15.0, agg_enabled=True,
):
    """Portfolio sim v1.48.0 with configurable AggRisk caps."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}
    agg_blocks = 0

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

            # Timeout (v1.48.0: 288 bars)
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

            # AggRisk cap (configurable)
            if agg_enabled:
                slope = ema_slope[bar] if bar < len(ema_slope) else 0
                is_uptrend = slope > 0
                is_counter = ((direction == 'SHORT' and is_uptrend) or
                              (direction == 'LONG' and not is_uptrend))
                cap_pct = counter_cap if is_counter else with_cap

                existing = sum(
                    p['eff_sl_pct'] * (1.0 / N_SLOTS) * LEVERAGE * p['size_mult']
                    for p in positions if p['direction'] == direction
                )
                new_exp = eff_sl * (1.0 / N_SLOTS) * LEVERAGE * sm
                if existing + new_exp > cap_pct:
                    agg_blocks += 1
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

    return trades, agg_blocks


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        counter_cap, with_cap, agg_enabled=True, n_folds=3):
    total_bars = neutral_end - neutral_start
    fold_size = total_bars // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        is_end = neutral_start + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, neutral_end)
        if oos_start >= oos_end:
            continue

        oos_trades, _ = portfolio_sim_agg(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            counter_cap=counter_cap, with_cap=with_cap, agg_enabled=agg_enabled,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("AGGRISK RE-SWEEP STUDY (v1.48.0 baseline)")
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

    # ==================== PHASE 1: Counter cap sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: Counter cap sweep (with_cap=15.0 fixed)")
    print("=" * 90)

    counter_vals = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    print(f"\n{'Counter':>8} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Blocks':>7}")
    print("-" * 58)

    p1_results = {}
    for c in counter_vals:
        trades, blocks = portfolio_sim_agg(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            counter_cap=c, with_cap=15.0,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p1_results[c] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                          'wr': stats['wr'], 'trades': stats['trades'], 'blocks': blocks}
        marker = ' ← current' if c == 5.0 else ''
        print(f"{c:>8.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {blocks:>7}{marker}")

    # ==================== PHASE 2: With cap sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 2: With cap sweep (counter_cap=5.0 fixed)")
    print("=" * 90)

    with_vals = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 50.0]
    print(f"\n{'With':>8} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Blocks':>7}")
    print("-" * 58)

    p2_results = {}
    for w in with_vals:
        trades, blocks = portfolio_sim_agg(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            counter_cap=5.0, with_cap=w,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p2_results[w] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                          'wr': stats['wr'], 'trades': stats['trades'], 'blocks': blocks}
        marker = ' ← current' if w == 15.0 else ''
        print(f"{w:>8.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {blocks:>7}{marker}")

    # ==================== PHASE 3: 2D Grid + OFF ====================
    print("\n" + "=" * 90)
    print("PHASE 3: 2D Grid + OFF")
    print("=" * 90)

    top_c = sorted(p1_results.keys(), key=lambda k: p1_results[k]['pm'], reverse=True)[:4]
    top_w = sorted(p2_results.keys(), key=lambda k: p2_results[k]['pm'], reverse=True)[:4]

    print(f"\n  Top counter: {top_c}")
    print(f"  Top with: {top_w}")

    grid_results = {}
    configs = []
    for c in top_c:
        for w in top_w:
            if w > c:  # with_cap must be > counter_cap
                configs.append((c, w, True))
    configs.append((5.0, 15.0, True))  # current
    configs.append((5.0, 15.0, False))  # OFF
    configs = list(set(configs))
    configs.sort()

    print(f"\n  Grid: {len(configs)} configs")
    print(f"\n{'Config':>16} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Blocks':>7}")
    print("-" * 72)

    for c, w, enabled in configs:
        trades, blocks = portfolio_sim_agg(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            counter_cap=c, with_cap=w, agg_enabled=enabled,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        if not enabled:
            label = "OFF"
        elif c == 5.0 and w == 15.0:
            label = "CURRENT"
        else:
            label = f"c{c:.0f}_w{w:.0f}"
        grid_results[label] = {
            'counter': c, 'with_cap': w, 'enabled': enabled,
            'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
            'pnl_mdd': round(pm, 1), 'wr': round(stats['wr'], 1),
            'trades': stats['trades'], 'blocks': blocks,
        }
        marker = ''
        if not enabled:
            marker = ' ← OFF'
        elif c == 5.0 and w == 15.0:
            marker = ' ← current'
        print(f"{label:>16} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {blocks:>7}{marker}")

    # ==================== PHASE 4: WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 4: WF 3-fold validation")
    print("=" * 90)

    ranked = sorted(grid_results.items(), key=lambda x: x[1]['pnl_mdd'], reverse=True)
    wf_candidates = []
    seen = set()
    for label, r in ranked:
        key = (r['counter'], r['with_cap'], r['enabled'])
        if key not in seen:
            seen.add(key)
            wf_candidates.append((label, r))
        if len(wf_candidates) >= 6:
            break
    for label in ('CURRENT', 'OFF'):
        r = grid_results.get(label, {})
        if r:
            key = (r['counter'], r['with_cap'], r['enabled'])
            if key not in seen:
                seen.add(key)
                wf_candidates.append((label, r))

    wf_results = {}
    print(f"\n{'Config':>16} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} "
          f"{'IS P/M':>8} {'WF':>6}")
    print("-" * 80)

    for label, r in wf_candidates:
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            counter_cap=r['counter'], with_cap=r['with_cap'], agg_enabled=r['enabled'],
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
            'counter': r['counter'], 'with_cap': r['with_cap'],
            'enabled': r['enabled'],
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pnl_mdd': r['pnl_mdd'],
        }

        marker = ''
        if label == 'CURRENT':
            marker = ' ← current'
        elif label == 'OFF':
            marker = ' ← OFF'
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:>16} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} "
              f"{r['pnl_mdd']:>8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    current_is = grid_results.get('CURRENT', {})
    current_wf = wf_results.get('CURRENT', {})
    print(f"\nCurrent (c5/w15):")
    print(f"  IS PnL/MDD: {current_is.get('pnl_mdd', 'N/A')}, Blocks: {current_is.get('blocks', 0)}")
    print(f"  OOS avg: {current_wf.get('avg', 'N/A')}, min: {current_wf.get('min', 'N/A')}")

    off_wf = wf_results.get('OFF', {})
    print(f"\nOFF:")
    print(f"  OOS avg: {off_wf.get('avg', 'N/A')}, min: {off_wf.get('min', 'N/A')}")

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

        delta_min = best_wf['min'] - current_wf.get('min', 0)
        if best_label == 'CURRENT':
            print("\n>>> RECOMMEND: KEEP current AggRisk (c5/w15)")
        elif delta_min > 5:
            print(f"\n  Min fold delta: {delta_min:+.1f}%")
            print(f"\n>>> RECOMMEND: Switch to {best_label}")
        else:
            print(f"\n  Min fold delta: {delta_min:+.1f}%")
            print(f"\n>>> RECOMMEND: Marginal improvement, KEEP current")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.48.0',
        'phase1_counter': {str(k): v for k, v in p1_results.items()},
        'phase2_with': {str(k): v for k, v in p2_results.items()},
        'grid_results': grid_results,
        'wf_results': wf_results,
        'recommended': best_label,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
