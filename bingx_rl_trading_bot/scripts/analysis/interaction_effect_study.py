#!/usr/bin/env python3
"""
Interaction Effect Study (v1.45.0→v1.52.0)
============================================

Verify that 7 parameter changes from v1.45.0~v1.51.0 don't cancel each other:

  v1.45.0: cascade_tighten_pct 75→85
  v1.46.0: momentum lookback 6→3, cooldown 6→12
  v1.47.0: atr_clamp_lo 0.6→0.5
  v1.48.0: timeout 864→288
  v1.49.0: aggrisk counter_cap 5→8
  v1.50.0: atr_clamp_hi 1.7→1.5
  v1.51.0: momentum threshold 1.0→1.5

Phase 1: Individual Contribution — Each param NEW vs OLD, rest at v1.52.0
Phase 2: Sequential Stacking — Apply changes one by one in order
Phase 3: Ablation — Full v1.52.0 minus one change at a time
Phase 4: WF validation of interesting configs

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

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'interaction_effect_study.json')

# ── Parameter definitions ──
# Each param: (name, old_value, new_value)
PARAMS = [
    ('cascade_pct',     75,  85),    # v1.45.0
    ('mom_lookback',     6,   3),    # v1.46.0
    ('mom_cooldown',     6,  12),    # v1.46.0
    ('atr_lo',         0.6, 0.5),    # v1.47.0
    ('timeout',        864, 288),    # v1.48.0
    ('agg_counter',    5.0, 8.0),    # v1.49.0
    ('atr_hi',         1.7, 1.5),    # v1.50.0
    ('mom_threshold',  1.0, 1.5),    # v1.51.0
]

# v1.52.0 (current) — all new values
V152 = {p[0]: p[2] for p in PARAMS}
# v1.44.0 baseline — all old values
V144 = {p[0]: p[1] for p in PARAMS}

# Fixed params (unchanged in v1.45~v1.51)
DIRECTION_CAP = 7
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3
AGG_WITH_CAP = 15.0  # unchanged


def portfolio_sim(signal_tuples, opens, highs, lows, closes, type_codes,
                  n_bars, atr_ratio, ema_slope, start_bar, end_bar, cfg):
    """Portfolio sim with configurable params from cfg dict."""
    cascade_mult = 1.0 - cfg['cascade_pct'] / 100.0
    timeout_bars = cfg['timeout']
    mom_lookback = cfg['mom_lookback']
    mom_cooldown = cfg['mom_cooldown']
    mom_threshold = cfg['mom_threshold']
    atr_lo = cfg['atr_lo']
    atr_hi = cfg['atr_hi']
    agg_counter = cfg['agg_counter']

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

            # Timeout
            if hold >= timeout_bars:
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
                        pos['eff_sl_pct'] *= cascade_mult

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

        for pat, direction, tp_pct, sl_pct in sig_by_bar[bar]:
            if len(positions) >= N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
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
                r = clamp(atr_ratio[bar], atr_lo, atr_hi)
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # AggRisk cap
            slope = ema_slope[bar] if bar < len(ema_slope) else 0
            is_uptrend = slope > 0
            is_counter = ((direction == 'SHORT' and is_uptrend) or
                          (direction == 'LONG' and not is_uptrend))
            cap_pct = agg_counter if is_counter else AGG_WITH_CAP

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

    return trades


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, ns, ne, cfg, n_folds=3):
    total_bars = ne - ns
    fold_size = total_bars // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        oos_start = ns + fold_size * (fold + 1)
        oos_end = min(oos_start + fold_size, ne)
        if oos_start >= oos_end:
            continue
        trades = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end, cfg,
        )
        stats = calc_stats(trades)
        results.append(stats['pnl'])
    return results


def run_sim(label, cfg, neutral_signals, opens, highs, lows, closes,
            type_codes, n_bars, atr_ratio, ema_slope, ns, ne):
    trades = portfolio_sim(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, ns, ne, cfg,
    )
    stats = calc_stats(trades)
    pm = stats['pnl'] / max(stats['mdd'], 0.01)
    return {
        'label': label, 'pnl': round(stats['pnl'], 1),
        'mdd': round(stats['mdd'], 2), 'pnl_mdd': round(pm, 1),
        'wr': round(stats['wr'], 1), 'trades': stats['trades'],
    }


def main():
    print("=" * 90)
    print("INTERACTION EFFECT STUDY (v1.45.0 → v1.52.0)")
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

    # ==================== Baselines ====================
    print("\n" + "=" * 90)
    print("BASELINES")
    print("=" * 90)

    base_v152 = run_sim("v1.52.0 (ALL NEW)", V152, *args)
    base_v144 = run_sim("v1.44.0 (ALL OLD)", V144, *args)

    print(f"\n{'Config':>25} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} {'Trades':>7}")
    print("-" * 70)
    for r in [base_v152, base_v144]:
        print(f"{r['label']:>25} {r['pnl']:>+8.1f} {r['mdd']:>8.2f} "
              f"{r['pnl_mdd']:>8.1f} {r['wr']:>6.1f} {r['trades']:>7}")

    delta_pnl = base_v152['pnl'] - base_v144['pnl']
    delta_mdd = base_v152['mdd'] - base_v144['mdd']
    print(f"\n  Total delta: PnL {delta_pnl:+.1f}%, MDD {delta_mdd:+.2f}%")

    # ==================== PHASE 1: Individual Contribution ====================
    print("\n" + "=" * 90)
    print("PHASE 1: Individual Contribution (flip one param to OLD, rest at v1.52.0)")
    print("=" * 90)

    print(f"\n{'Param':>18} {'OLD':>6} {'NEW':>6} "
          f"{'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'dPnL':>7} {'dMDD':>7}")
    print("-" * 95)

    p1_results = []
    for name, old_val, new_val in PARAMS:
        cfg = dict(V152)
        cfg[name] = old_val  # revert this one param
        r = run_sim(f"revert_{name}", cfg, *args)
        d_pnl = base_v152['pnl'] - r['pnl']  # positive = new is better
        d_mdd = base_v152['mdd'] - r['mdd']
        p1_results.append({
            'param': name, 'old': old_val, 'new': new_val,
            **r, 'delta_pnl': round(d_pnl, 1), 'delta_mdd': round(d_mdd, 2),
        })
        print(f"{name:>18} {old_val:>6} {new_val:>6} "
              f"{r['pnl']:>+8.1f} {r['mdd']:>8.2f} {r['pnl_mdd']:>8.1f} {r['wr']:>6.1f} "
              f"{d_pnl:>+7.1f} {d_mdd:>+7.2f}")

    # Rank by contribution (delta_pnl)
    ranked = sorted(p1_results, key=lambda x: x['delta_pnl'], reverse=True)
    print(f"\n  Contribution ranking (higher = new value more beneficial):")
    for i, r in enumerate(ranked):
        sign = "+" if r['delta_pnl'] > 0 else ""
        print(f"    {i+1}. {r['param']:18s}: PnL {sign}{r['delta_pnl']:.1f}%, MDD {r['delta_mdd']:+.2f}%")

    # Check for super-additivity
    sum_individual = sum(r['delta_pnl'] for r in p1_results)
    print(f"\n  Sum of individual contributions: {sum_individual:+.1f}%")
    print(f"  Actual total delta: {delta_pnl:+.1f}%")
    interaction = delta_pnl - sum_individual
    print(f"  Interaction effect: {interaction:+.1f}% "
          f"({'synergy' if interaction > 0 else 'cancellation'})")

    # ==================== PHASE 2: Sequential Stacking ====================
    print("\n" + "=" * 90)
    print("PHASE 2: Sequential Stacking (apply changes one by one)")
    print("=" * 90)

    print(f"\n{'Step':>3} {'Param applied':>18} "
          f"{'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Marginal PnL':>13}")
    print("-" * 80)

    cfg = dict(V144)  # start from old baseline
    r_prev = base_v144
    print(f"{'0':>3} {'baseline (v1.44)':>18} "
          f"{r_prev['pnl']:>+8.1f} {r_prev['mdd']:>8.2f} {r_prev['pnl_mdd']:>8.1f} "
          f"{r_prev['wr']:>6.1f} {'--':>13}")

    p2_results = [{'step': 0, 'param': 'baseline', **base_v144}]
    for step, (name, old_val, new_val) in enumerate(PARAMS, 1):
        cfg[name] = new_val  # apply this change
        r = run_sim(f"step{step}_{name}", cfg, *args)
        marginal = r['pnl'] - r_prev['pnl']
        p2_results.append({
            'step': step, 'param': name, **r,
            'marginal_pnl': round(marginal, 1),
        })
        print(f"{step:>3} {name:>18} "
              f"{r['pnl']:>+8.1f} {r['mdd']:>8.2f} {r['pnl_mdd']:>8.1f} "
              f"{r['wr']:>6.1f} {marginal:>+13.1f}")
        r_prev = r

    # ==================== PHASE 3: Leave-One-Out Ablation ====================
    print("\n" + "=" * 90)
    print("PHASE 3: Leave-One-Out Ablation (v1.52.0 minus one change)")
    print("=" * 90)

    print(f"\n{'Removed param':>18} "
          f"{'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'vs v1.52':>9} {'Verdict':>10}")
    print("-" * 75)

    p3_results = []
    for name, old_val, new_val in PARAMS:
        cfg = dict(V152)
        cfg[name] = old_val  # revert just this one
        r = run_sim(f"no_{name}", cfg, *args)
        delta = base_v152['pnl'] - r['pnl']
        verdict = "KEEP NEW" if delta > 0 else "REVERT" if delta < -2 else "NEUTRAL"
        p3_results.append({
            'param': name, 'old': old_val, 'new': new_val,
            **r, 'delta_vs_v152': round(delta, 1), 'verdict': verdict,
        })
        print(f"{name:>18} "
              f"{r['pnl']:>+8.1f} {r['mdd']:>8.2f} {r['pnl_mdd']:>8.1f} "
              f"{r['wr']:>6.1f} {delta:>+9.1f} {verdict:>10}")

    # ==================== PHASE 4: WF Validation ====================
    print("\n" + "=" * 90)
    print("PHASE 4: WF 3-fold Validation")
    print("=" * 90)

    wf_configs = [
        ("v1.52.0 (current)", V152),
        ("v1.44.0 (all old)", V144),
    ]
    # Add any configs where reverting improved PnL
    for r in p3_results:
        if r['delta_vs_v152'] < -2:
            cfg_revert = dict(V152)
            cfg_revert[r['param']] = r['old']
            wf_configs.append((f"revert_{r['param']}", cfg_revert))

    print(f"\n{'Config':>25} {'F1':>8} {'F2':>8} {'F3':>8} "
          f"{'Avg':>8} {'Min':>8} {'WF':>6}")
    print("-" * 75)

    p4_results = {}
    for label, cfg in wf_configs:
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, ns, ne, cfg,
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

        p4_results[label] = {
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
        }
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        marker = ' ← current' if 'v1.52' in label else ''
        print(f"{label:>25} {''.join(fold_strs)} "
              f"{avg:>+8.1f} {mn:>+8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    print(f"\n  v1.44.0 → v1.52.0 total delta: PnL {delta_pnl:+.1f}%, MDD {delta_mdd:+.2f}%")
    print(f"  Sum of individual contributions: {sum_individual:+.1f}%")
    print(f"  Interaction effect: {interaction:+.1f}%")

    if abs(interaction) < 5:
        print("\n  Interaction effect is SMALL — changes are mostly independent.")
    elif interaction > 5:
        print("\n  Interaction effect is SYNERGISTIC — changes amplify each other.")
    else:
        print("\n  Interaction effect is CANCELLING — some changes conflict.")

    # Harmful changes
    harmful = [r for r in p3_results if r['delta_vs_v152'] < -2]
    if harmful:
        print(f"\n  Potentially harmful changes (reverting improves PnL >2%):")
        for r in harmful:
            print(f"    {r['param']}: {r['old']}→{r['new']}, revert gains +{-r['delta_vs_v152']:.1f}%")
    else:
        print("\n  No harmful changes detected.")

    # WF verdicts
    v152_wf = p4_results.get("v1.52.0 (current)", {})
    print(f"\n  v1.52.0 WF: {v152_wf.get('n_pass', 0)}/3, "
          f"min fold {v152_wf.get('min', 'N/A')}%")

    if v152_wf.get('n_pass', 0) == 3 and not harmful:
        print("\n>>> VERDICT: v1.52.0 stack is VALID — no harmful interactions detected.")
    elif harmful:
        wf_any_pass = any(p4_results.get(f"revert_{r['param']}", {}).get('n_pass', 0) == 3
                          for r in harmful)
        if wf_any_pass:
            print("\n>>> VERDICT: Consider reverting harmful param(s) — WF validates revert.")
        else:
            print("\n>>> VERDICT: Harmful interaction detected but revert fails WF — KEEP v1.52.0.")
    else:
        print("\n>>> VERDICT: KEEP v1.52.0 stack.")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.52.0',
        'baselines': {'v152': base_v152, 'v144': base_v144},
        'total_delta_pnl': round(delta_pnl, 1),
        'sum_individual': round(sum_individual, 1),
        'interaction_effect': round(interaction, 1),
        'phase1_individual': p1_results,
        'phase2_sequential': p2_results,
        'phase3_ablation': p3_results,
        'phase4_wf': p4_results,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
