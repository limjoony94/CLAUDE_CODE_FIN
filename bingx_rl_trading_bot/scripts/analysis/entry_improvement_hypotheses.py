#!/usr/bin/env python3
"""
Entry Improvement Hypotheses — WF Validated
=============================================
entry_behavior_critical_study.py 발견 기반 개선 가설 WF 검증.

발견 → 가설:
[F2] 보유시간 ↑ = WR ↓ → H1: Timeout 864→432 (36h 단축)
[F3] Counter-regime WR 50% → H2: Counter-regime 진입 차단 (momentum guard 강화)
[F5] 밀도 체감 N>6 → H3: MaxSlots 9→7
[F4] Guard 과차단 6,812%  → H4: AggRisk cap 완화 (counter 3→5%, with 7→10%)
[F2] 빠른 체결 유리 → H5: TP 축소 (75% of current, 더 빠른 체결)
[F5+F2] 복합 → H6: N=7 + Timeout 432
[F2] 복합 → H7: TP 75% + Timeout 432

Standard Research Protocol: 3-fold WF expanding window, compound, LEVERAGE=3.
"""
import sys, os, json, time
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, calc_stats, clamp,
    DATA_FILE, PATTERNS_FILE,
    N_SLOTS, DIRECTION_CAP, LEVERAGE, FEE_PCT,
    TIMEOUT_BARS, ATR_CLAMP_LO, ATR_CLAMP_HI, SLIPPAGE_BUFFER,
    AGG_RISK_COUNTER, AGG_RISK_WITH, BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    MOMENTUM_LOOKBACK, MOMENTUM_THRESHOLD, MOMENTUM_COOLDOWN,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
)


def portfolio_sim_extended(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    # Hypothesis overrides
    n_slots=None,
    direction_cap=None,
    timeout_bars=None,
    agg_counter_cap=None,
    agg_with_cap=None,
    tp_mult=1.0,  # multiply TP by this factor
    counter_regime_block=False,  # block counter-regime entries entirely
    momentum_lookback=None,
    momentum_threshold=None,
):
    """Extended sim with hypothesis-testable parameters."""
    n_sl = n_slots or N_SLOTS
    d_cap = direction_cap or DIRECTION_CAP
    t_bars = timeout_bars or TIMEOUT_BARS
    a_counter = agg_counter_cap if agg_counter_cap is not None else AGG_RISK_COUNTER
    a_with = agg_with_cap if agg_with_cap is not None else AGG_RISK_WITH
    m_lookback = momentum_lookback or MOMENTUM_LOOKBACK
    m_threshold = momentum_threshold or MOMENTUM_THRESHOLD

    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / n_sl

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

        # EXITS
        closed_slots = set()
        sl_exits = []
        h_bar, l_bar, o_bar = highs[bar], lows[bar], opens[bar]

        for pos in positions:
            if bar < pos['entry_bar']:
                continue
            entry_p = pos['entry_price']
            if entry_p <= 0:
                continue

            direction = pos['direction']
            eff_tp = pos['eff_tp_pct']
            eff_sl = pos['eff_sl_pct']
            sm = pos['size_mult']
            hold = bar - pos['entry_bar']

            exit_price = reason = None

            # Timeout
            if hold >= t_bars:
                closed_slots.add(pos['slot'])
                continue  # DROP

            # Early exit
            if hold >= EARLY_CONFIRM:
                exit_types = ['BD'] if direction == 'LONG' else ['BU']
                candle_ok = all(
                    0 <= bar-1-k < len(type_codes) and type_codes[bar-1-k] in exit_types
                    for k in range(EARLY_CONFIRM)
                )
                if candle_ok:
                    cp = closes[bar-1] if bar-1 < n_bars else entry_p
                    unr = ((cp/entry_p - 1) if direction == 'LONG' else (1 - cp/entry_p)) * 100 * LEVERAGE
                    if unr >= EARLY_MIN_PROFIT:
                        exit_price, reason = cp, 'EARLY'

            # TP/SL
            if reason is None:
                if direction == 'LONG':
                    tp_p = entry_p * (1 + eff_tp / 100)
                    sl_p = entry_p * (1 - eff_sl / 100)
                    hit_tp, hit_sl = h_bar >= tp_p, l_bar <= sl_p
                else:
                    tp_p = entry_p * (1 - eff_tp / 100)
                    sl_p = entry_p * (1 + eff_sl / 100)
                    hit_tp, hit_sl = l_bar <= tp_p, h_bar >= sl_p

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

            pnl = ((exit_price/entry_p - 1) if direction == 'LONG' else (1 - exit_price/entry_p)) * 100 * LEVERAGE - fee
            pnl_port = pnl * (size_pct / 100) * sm
            trades.append({
                'entry_bar': pos['entry_bar'], 'exit_bar': bar,
                'pnl_slot': pnl, 'pnl_portfolio': pnl_port,
                'reason': reason, 'pattern': pos['pattern'],
                'direction': direction, 'size_mult': sm,
            })
            closed_slots.add(pos['slot'])
            if reason == 'SL':
                sl_exits.append(pos)

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Cascade
        if sl_exits:
            for sl_pos in sl_exits:
                for pos in positions:
                    if pos['direction'] == sl_pos['direction']:
                        pos['eff_sl_pct'] *= 0.25

        bar_pnl = sum(t['pnl_portfolio'] for t in trades if t['exit_bar'] == bar)
        equity *= (1 + bar_pnl / 100)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0

        # Momentum
        if bar >= m_lookback:
            pa = closes[bar - m_lookback]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > m_threshold:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'], bar + MOMENTUM_COOLDOWN)
                elif pct < -m_threshold:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'], bar + MOMENTUM_COOLDOWN)

        # ENTRIES
        if bar not in sig_by_bar:
            continue

        for pat, direction, tp_pct, sl_pct in sig_by_bar[bar]:
            if len(positions) >= n_sl:
                continue
            if sum(1 for p in positions if p['direction'] == direction) >= d_cap:
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

            # Counter-regime block
            if counter_regime_block:
                slope = ema_slope[bar] if bar < len(ema_slope) else 0
                is_counter = ((direction == 'SHORT' and slope > 0) or
                              (direction == 'LONG' and slope <= 0))
                if is_counter:
                    continue

            # MDD sizing
            sm = 1.0
            if peak_equity > 0:
                dd_pct = (peak_equity - equity) / peak_equity * 100
                if dd_pct <= MDD_FULL_BELOW:
                    sm = 1.0
                elif dd_pct >= MDD_MIN_ABOVE:
                    sm = MDD_MIN_SCALE
                else:
                    sm = 1.0 - (1.0 - MDD_MIN_SCALE) * (dd_pct - MDD_FULL_BELOW) / (MDD_MIN_ABOVE - MDD_FULL_BELOW)

            # ATR
            r = clamp(atr_ratio[bar], ATR_CLAMP_LO, ATR_CLAMP_HI) if bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]) else 1.0
            eff_tp = tp_pct * tp_mult * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # Agg risk
            slope = ema_slope[bar] if bar < len(ema_slope) else 0
            is_counter = ((direction == 'SHORT' and slope > 0) or
                          (direction == 'LONG' and slope <= 0))
            cap_pct = a_counter if is_counter else a_with
            existing = sum(p['eff_sl_pct'] * (1.0 / n_sl) * LEVERAGE * p['size_mult']
                           for p in positions if p['direction'] == direction)
            new_exp = eff_sl * (1.0 / n_sl) * LEVERAGE * sm
            if existing + new_exp > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{bar}",
                'entry_bar': entry_bar, 'entry_price': entry_price,
                'direction': direction, 'pattern': pat,
                'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'eff_tp_pct': eff_tp, 'eff_sl_pct': eff_sl,
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
        exit_p = opens[exit_bar]
        pnl = ((exit_p/ep - 1) if pos['direction'] == 'LONG' else (1 - exit_p/ep)) * 100 * LEVERAGE - fee
        sm = pos['size_mult']
        trades.append({
            'entry_bar': eb, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
            'reason': 'END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
        })

    return trades


def run_wf(signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
           atr_ratio, ema_slope, ns, ne, n_folds=3, **kwargs):
    """3-fold expanding window WF."""
    total = ne - ns
    seg = total // (n_folds + 1)

    folds = []
    for fold in range(n_folds):
        oos_start = ns + seg * (fold + 1)
        oos_end = ns + seg * (fold + 2) if fold < n_folds - 1 else ne
        trades = portfolio_sim_extended(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end, **kwargs)
        stats = calc_stats(trades)
        stats['fold'] = fold + 1
        folds.append(stats)

    return folds


def main():
    t_start = time.time()
    print("=" * 85)
    print("ENTRY IMPROVEMENT HYPOTHESES — WF VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 85)

    # Load
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    type_codes = df['rctype'].values
    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)
    ns, ne = find_neutral_window(closes)
    print(f"Neutral: {ns}-{ne} ({(ne-ns)/BARS_PER_DAY:.0f}d)")

    # Load patterns
    with open(PATTERNS_FILE) as f:
        pdata = json.load(f)
    pats_raw = pdata['patterns']
    tpsl = pdata.get('patterns_tpsl', {})
    pat_lookup = {}
    for pn in pats_raw.get('long', []):
        ts = tpsl.get(pn, [2.0, 3.0])
        pat_lookup[pn] = {'direction': 'LONG', 'tp': ts[0], 'sl': ts[1]}
    for pn in pats_raw.get('short', []):
        ts = tpsl.get(pn, [2.0, 3.0])
        pat_lookup[pn] = {'direction': 'SHORT', 'tp': ts[0], 'sl': ts[1]}

    rctypes = df['rctype'].values
    signal_tuples = []
    for i in range(2, n_bars):
        tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if tri in pat_lookup:
            p = pat_lookup[tri]
            signal_tuples.append((i, tri, p['direction'], p['tp'], p['sl']))
    print(f"{len(pat_lookup)} patterns, {len(signal_tuples)} signals")

    common = dict(
        signal_tuples=signal_tuples, opens=opens, highs=highs, lows=lows,
        closes=closes, type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope, ns=ns, ne=ne,
    )

    # ============================================================
    # Hypotheses
    # ============================================================
    hypotheses = {
        'H0_baseline': {},
        'H1_timeout_432': {'timeout_bars': 432},
        'H2_counter_block': {'counter_regime_block': True},
        'H3_n7': {'n_slots': 7, 'direction_cap': 5},
        'H4_agg_relax': {'agg_counter_cap': 5.0, 'agg_with_cap': 10.0},
        'H5_tp75': {'tp_mult': 0.75},
        'H6_n7_to432': {'n_slots': 7, 'direction_cap': 5, 'timeout_bars': 432},
        'H7_tp75_to432': {'tp_mult': 0.75, 'timeout_bars': 432},
        'H8_tp85': {'tp_mult': 0.85},
        'H9_agg_relax_mod': {'agg_counter_cap': 4.0, 'agg_with_cap': 8.0},
        'H10_counter_block_tp85': {'counter_regime_block': True, 'tp_mult': 0.85},
    }

    # ============================================================
    # Phase A: IS Full Neutral
    # ============================================================
    print(f"\n{'='*85}")
    print("PHASE A: IS Full Neutral Window")
    print(f"{'='*85}")
    print(f"{'Config':<28} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>8} {'Trades':>8}")
    print(f"{'-'*72}")

    is_results = {}
    for name, kwargs in hypotheses.items():
        trades = portfolio_sim_extended(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, ns, ne, **kwargs)
        stats = calc_stats(trades)
        is_results[name] = stats
        print(f"{name:<28} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{stats['pnl_mdd']:>8.1f} {stats['wr']:>8.1f} {stats['trades']:>8}")

    # ============================================================
    # Phase B: WF 3-fold
    # ============================================================
    print(f"\n{'='*85}")
    print("PHASE B: WF 3-fold Expanding Window")
    print(f"{'='*85}")
    print(f"{'Config':<28} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} {'Verdict':>10}")
    print(f"{'-'*85}")

    wf_results = {}
    for name, kwargs in hypotheses.items():
        folds = run_wf(**common, **kwargs)
        n_pass = sum(1 for f in folds if f['pnl'] > 0)
        avg_oos = np.mean([f['pnl'] for f in folds])
        min_oos = min(f['pnl'] for f in folds)
        verdict = f"{n_pass}/3 {'PASS' if n_pass == 3 else 'FAIL'}"
        wf_results[name] = {
            'folds': folds, 'n_pass': n_pass,
            'avg_oos': round(float(avg_oos), 2),
            'min_oos': round(float(min_oos), 2),
            'verdict': verdict,
        }
        print(f"{name:<28} {folds[0]['pnl']:>+8.1f} {folds[1]['pnl']:>+8.1f} "
              f"{folds[2]['pnl']:>+8.1f} {avg_oos:>+8.1f} {min_oos:>+8.1f} {verdict:>10}")

    # ============================================================
    # Phase C: Decision Matrix
    # ============================================================
    print(f"\n{'='*85}")
    print("PHASE C: Decision Matrix (WF 3/3 PASS only)")
    print(f"{'='*85}")

    passed = [(name, is_results[name], wf_results[name])
              for name in hypotheses if wf_results[name]['n_pass'] == 3]
    passed.sort(key=lambda x: x[1]['pnl_mdd'], reverse=True)

    print(f"{'Rank':<6} {'Config':<28} {'IS P/M':>8} {'IS PnL':>8} {'IS MDD':>8} "
          f"{'OOS Avg':>8} {'OOS Min':>8}")
    print(f"{'-'*85}")

    for i, (name, isr, wfr) in enumerate(passed, 1):
        delta_pm = isr['pnl_mdd'] - is_results['H0_baseline']['pnl_mdd']
        print(f"{i:<6} {name:<28} {isr['pnl_mdd']:>8.1f} {isr['pnl']:>+8.1f} {isr['mdd']:>8.2f} "
              f"{wfr['avg_oos']:>+8.1f} {wfr['min_oos']:>+8.1f}  (Δ P/M={delta_pm:+.1f})")

    failed = [name for name in hypotheses if wf_results[name]['n_pass'] < 3]
    if failed:
        print(f"\n  WF FAILED: {', '.join(failed)}")

    if passed:
        best = passed[0]
        print(f"\n>>> BEST: {best[0]} (IS PnL/MDD={best[1]['pnl_mdd']:.1f}, "
              f"OOS avg={best[2]['avg_oos']:+.1f}%)")

    # Detailed fold stats for top 3
    print(f"\n{'='*85}")
    print("Detailed fold stats (top 5):")
    for name, _, wfr in passed[:5]:
        print(f"\n  {name}:")
        for f in wfr['folds']:
            print(f"    Fold {f['fold']}: PnL={f['pnl']:+.1f}%, WR={f['wr']:.1f}%, "
                  f"MDD={f['mdd']:.2f}%, Trades={f['trades']}")

    # Save
    out = {
        'is_results': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                            for kk, vv in v.items()} for k, v in is_results.items()},
        'wf_results': {k: {'n_pass': v['n_pass'], 'avg_oos': v['avg_oos'],
                            'min_oos': v['min_oos'], 'verdict': v['verdict'],
                            'folds': [{kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                                        for kk, vv in f.items()} for f in v['folds']]}
                       for k, v in wf_results.items()},
        'passed_ranking': [p[0] for p in passed],
        'failed': failed,
        'best': passed[0][0] if passed else None,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'results', 'entry_improvement_hypotheses.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Runtime: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
