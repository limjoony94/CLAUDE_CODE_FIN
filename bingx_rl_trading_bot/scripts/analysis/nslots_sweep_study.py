#!/usr/bin/env python3
"""
N-Slots Sweep Study (v1.52.0 baseline)
========================================

Sweep max_positions (N_SLOTS) to find optimal slot count.
Current: N=9. Test N=3,5,7,9,11,13,15.
Sizing = 100/N per slot (compound).

Phase 1: IS sweep (N=3..15)
Phase 2: Direction cap interaction (dir_cap = N-2, N-1, N)
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
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'nslots_sweep_study.json')

# v1.52.0 fixed params
DIRECTION_CAP = 7
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
    n_slots=9, direction_cap=7,
):
    """Portfolio sim v1.52.0 with configurable N_SLOTS and direction_cap."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / n_slots

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}
    blocked = {'dir_cap': 0, 'agg_risk': 0, 'max_pos': 0, 'momentum': 0, 'dup_pat': 0}

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
            if len(positions) >= n_slots:
                blocked['max_pos'] += 1
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                blocked['dir_cap'] += 1
                continue
            if any(p['pattern'] == pat for p in positions):
                blocked['dup_pat'] += 1
                continue

            entry_bar = bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            if bar < mom_pause_until.get(direction, -1):
                blocked['momentum'] += 1
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
            slope = ema_slope[bar] if bar < len(ema_slope) else 0
            is_uptrend = slope > 0
            is_counter = ((direction == 'SHORT' and is_uptrend) or
                          (direction == 'LONG' and not is_uptrend))
            cap_pct = AGG_COUNTER_CAP if is_counter else AGG_WITH_CAP

            existing = sum(
                p['eff_sl_pct'] * (1.0 / n_slots) * LEVERAGE * p['size_mult']
                for p in positions if p['direction'] == direction
            )
            new_exp = eff_sl * (1.0 / n_slots) * LEVERAGE * sm
            if existing + new_exp > cap_pct:
                blocked['agg_risk'] += 1
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

    return trades, blocked


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        n_slots=9, direction_cap=7, n_folds=3):
    total_bars = neutral_end - neutral_start
    fold_size = total_bars // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        is_end = neutral_start + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, neutral_end)
        if oos_start >= oos_end:
            continue

        oos_trades, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            n_slots=n_slots, direction_cap=direction_cap,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("N-SLOTS SWEEP STUDY (v1.52.0 baseline)")
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

    # ==================== PHASE 1: N_SLOTS sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: N_SLOTS sweep (direction_cap=7 or min(7,N))")
    print("=" * 90)

    n_values = [3, 5, 7, 9, 11, 13, 15]
    print(f"\n{'N':>4} {'DirCap':>7} {'SizePct':>8} {'PnL%':>8} {'MDD%':>8} "
          f"{'P/M':>8} {'WR%':>6} {'Trades':>7} {'DirBlk':>7} {'AggBlk':>7} {'MaxBlk':>7}")
    print("-" * 100)

    p1_results = {}
    for n in n_values:
        dc = min(DIRECTION_CAP, n)
        trades, blk = portfolio_sim(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            n_slots=n, direction_cap=dc,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p1_results[n] = {
            'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
            'wr': stats['wr'], 'trades': stats['trades'],
            'dir_cap': dc, 'blocked': blk,
        }
        marker = ' ← current' if n == 9 else ''
        print(f"{n:>4} {dc:>7} {100/n:>8.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} "
              f"{blk['dir_cap']:>7} {blk['agg_risk']:>7} {blk['max_pos']:>7}{marker}")

    # ==================== PHASE 2: Direction cap interaction ====================
    print("\n" + "=" * 90)
    print("PHASE 2: Direction cap interaction (top N values)")
    print("=" * 90)

    # Pick top 3 N by PnL/MDD + current
    ranked_n = sorted(p1_results.keys(), key=lambda k: p1_results[k]['pm'], reverse=True)
    top_n = []
    for n in ranked_n:
        if n not in top_n:
            top_n.append(n)
        if len(top_n) >= 3:
            break
    if 9 not in top_n:
        top_n.append(9)
    top_n.sort()

    print(f"\n  Testing N={top_n} with direction_cap variants")
    print(f"\n{'N':>4} {'DirCap':>7} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} "
          f"{'WR%':>6} {'Trades':>7}")
    print("-" * 55)

    p2_results = {}
    for n in top_n:
        for dc in [max(2, n - 2), n - 1, n]:
            if dc < 1 or dc > n:
                continue
            trades, blk = portfolio_sim(
                neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                n_slots=n, direction_cap=dc,
            )
            stats = calc_stats(trades)
            pm = stats['pnl'] / max(stats['mdd'], 0.01)
            label = f"N{n}_dc{dc}"
            p2_results[label] = {
                'n_slots': n, 'dir_cap': dc,
                'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                'wr': stats['wr'], 'trades': stats['trades'],
            }
            marker = ' ← current' if n == 9 and dc == 7 else ''
            print(f"{n:>4} {dc:>7} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
                  f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7}{marker}")

    # ==================== PHASE 3: WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 3: WF 3-fold validation")
    print("=" * 90)

    # Rank by PnL/MDD, take top 6 + current
    all_configs = {}
    for n, r in p1_results.items():
        label = f"N{n}_dc{min(DIRECTION_CAP, n)}"
        all_configs[label] = {'n_slots': n, 'dir_cap': min(DIRECTION_CAP, n), 'pm': r['pm']}
    for label, r in p2_results.items():
        all_configs[label] = {'n_slots': r['n_slots'], 'dir_cap': r['dir_cap'], 'pm': r['pm']}

    ranked = sorted(all_configs.items(), key=lambda x: x[1]['pm'], reverse=True)
    wf_candidates = []
    seen = set()
    for label, cfg in ranked:
        key = (cfg['n_slots'], cfg['dir_cap'])
        if key not in seen:
            seen.add(key)
            wf_candidates.append((label, cfg))
        if len(wf_candidates) >= 8:
            break
    # Ensure current is included
    current_key = (9, 7)
    if current_key not in seen:
        wf_candidates.append(('N9_dc7', {'n_slots': 9, 'dir_cap': 7, 'pm': 0}))

    wf_results = {}
    print(f"\n{'Config':>12} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} "
          f"{'IS P/M':>8} {'WF':>6}")
    print("-" * 76)

    for label, cfg in wf_candidates:
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            n_slots=cfg['n_slots'], direction_cap=cfg['dir_cap'],
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
            'n_slots': cfg['n_slots'], 'dir_cap': cfg['dir_cap'],
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pm': round(cfg['pm'], 1),
        }

        marker = ' ← current' if cfg['n_slots'] == 9 and cfg['dir_cap'] == 7 else ''
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:>12} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} "
              f"{cfg['pm']:>8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    current_wf = wf_results.get('N9_dc7', {})
    print(f"\nCurrent (N=9, dc=7):")
    print(f"  IS PnL/MDD: {current_wf.get('is_pm', 'N/A')}")
    print(f"  OOS avg: {current_wf.get('avg', 'N/A')}, min: {current_wf.get('min', 'N/A')}")

    best_label = None
    best_min = -999
    for label, wr in wf_results.items():
        if wr.get('n_pass', 0) == 3 and wr['min'] > best_min:
            best_min = wr['min']
            best_label = label

    if best_label:
        best_wf = wf_results[best_label]
        print(f"\nBest WF-passing by min fold: {best_label}")
        print(f"  N={best_wf['n_slots']}, dir_cap={best_wf['dir_cap']}")
        print(f"  IS PnL/MDD: {best_wf['is_pm']}")
        print(f"  OOS avg: {best_wf['avg']}, min: {best_wf['min']}")

        delta_min = best_wf['min'] - current_wf.get('min', 0)
        if best_label == 'N9_dc7':
            print("\n>>> RECOMMEND: KEEP current N=9/dc=7")
        elif delta_min > 5:
            print(f"\n  Min fold delta: {delta_min:+.1f}%")
            print(f"\n>>> RECOMMEND: Switch to {best_label}")
        else:
            print(f"\n  Min fold delta: {delta_min:+.1f}%")
            print(f"\n>>> RECOMMEND: Marginal improvement, KEEP current")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.52.0',
        'phase1_nslots': {str(k): {
            'pnl': round(v['pnl'], 1), 'mdd': round(v['mdd'], 2),
            'pnl_mdd': round(v['pm'], 1), 'wr': round(v['wr'], 1),
            'trades': v['trades'], 'dir_cap': v['dir_cap'],
        } for k, v in p1_results.items()},
        'phase2_interaction': {k: {
            'n_slots': v['n_slots'], 'dir_cap': v['dir_cap'],
            'pnl': round(v['pnl'], 1), 'mdd': round(v['mdd'], 2),
            'pnl_mdd': round(v['pm'], 1),
        } for k, v in p2_results.items()},
        'wf_results': wf_results,
        'recommended': best_label,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
