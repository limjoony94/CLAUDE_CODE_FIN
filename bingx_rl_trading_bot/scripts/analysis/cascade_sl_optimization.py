#!/usr/bin/env python3
"""Cascade SL Optimization Study.

Research question:
  현재 Cascade SL(85% tighten)이 최선인가?
  다른 tighten%, 또는 완전히 다른 메커니즘이 더 나은가?

Variants:
  1. Tighten % sweep: 0(OFF), 50, 60, 70, 75, 80, 85(current), 90, 95
  2. Immediate close: SL hit → close ALL same-direction at market
  3. Breakeven move: SL hit → move remaining same-dir SLs to entry price
  4. Partial close + tighten: SL hit → close oldest same-dir + tighten rest

Protocol: N-pos portfolio, Entry next-bar open, intrabar exit,
  Fee 0.10%, Slippage 0.02%, Lev 3x, compound.
  WF 3-fold expanding window. Random discrimination 10 seeds.
"""
import os, sys, json, time, logging
import numpy as np, pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))

from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    calc_stats_compound, scan_universe_range,
    compute_atr_ratio, compute_ema_slope,
    _check_exit_npos,
    FEE_PCT, LEVERAGE, SLIPPAGE_BUFFER,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_REGIME_MULT, TIMEOUT_BARS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'cascade_sl_optimization.json')

# Cascade modes
MODE_TIGHTEN = 'tighten'         # Standard: tighten SL distance by X%
MODE_IMMEDIATE = 'immediate'     # Close all same-dir positions at market
MODE_BREAKEVEN = 'breakeven'     # Move SL to entry price (breakeven)
MODE_PARTIAL = 'partial_close'   # Close oldest same-dir + tighten rest


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    return {k: {'pattern': v['pattern'], 'direction': v['direction'],
                'tp_pct': v['tp'], 'sl_pct': v['sl']} for k, v in details.items()}


def portfolio_npos_cascade(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, start_bar, end_bar,
                           cascade_mode=MODE_TIGHTEN, cascade_pct=85,
                           n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
                           regime_mult=DEFAULT_REGIME_MULT,
                           agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
                           agg_risk_with=DEFAULT_AGG_RISK_WITH,
                           momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
                           momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
                           momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
                           clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
                           timeout_bars=TIMEOUT_BARS):
    """Portfolio simulator with configurable cascade SL mechanism."""
    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0, 'max_pos': 0}
    cascade_events = 0

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_sorted = sorted(
        [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples if start_bar <= s < end_bar],
        key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check exits
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0
        sl_directions = set()

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                       atr_ratio, fee, clamp_lo, clamp_hi, timeout_bars)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1
                    sl_directions.add(pos['direction'])

        # 2. Cascade mechanism (only if SL exit happened)
        if bar_sl_count > 0 and (cascade_mode != MODE_TIGHTEN or cascade_pct > 0):
            for sl_dir in sl_directions:
                same_dir_alive = [p for p in positions
                                  if p['slot'] not in closed_slots and p['direction'] == sl_dir]
                if not same_dir_alive:
                    continue

                cascade_events += 1

                if cascade_mode == MODE_TIGHTEN:
                    # Standard tighten
                    keep_ratio = 1.0 - cascade_pct / 100.0
                    for pos in same_dir_alive:
                        sig = pos['signal_bar']
                        if (atr_ratio is not None and sig < len(atr_ratio)
                                and not np.isnan(atr_ratio[sig])):
                            r = max(clamp_lo, min(clamp_hi, atr_ratio[sig]))
                        else:
                            r = 1.0
                        cur_eff_sl = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                        pos['eff_sl_override'] = cur_eff_sl * keep_ratio

                elif cascade_mode == MODE_IMMEDIATE:
                    # Close all same-direction at market
                    for pos in same_dir_alive:
                        entry_price = opens[pos['entry_bar']]
                        if entry_price <= 0:
                            continue
                        current_price = closes[bar]
                        if pos['direction'] == 'LONG':
                            pnl = (current_price / entry_price - 1) * 100 * LEVERAGE
                        else:
                            pnl = (1 - current_price / entry_price) * 100 * LEVERAGE
                        pnl -= fee
                        sm = pos.get('size_mult', 1.0)
                        pnl_portfolio = pnl * (size_pct / 100) * sm
                        trades.append({
                            'entry_bar': pos['entry_bar'], 'exit_bar': bar,
                            'pnl_slot': pnl, 'reason': 'CASCADE_CLOSE',
                            'pattern': pos['pattern'], 'direction': pos['direction'],
                            'size_mult': sm, 'pnl_portfolio': pnl_portfolio,
                        })
                        closed_slots.append(pos['slot'])
                        bar_pnl_sum += pnl_portfolio

                elif cascade_mode == MODE_BREAKEVEN:
                    # Move SL to entry price (breakeven)
                    for pos in same_dir_alive:
                        # Set eff_sl_override to 0 (SL at entry = 0% distance)
                        # Actually need small buffer for slippage
                        pos['eff_sl_override'] = SLIPPAGE_BUFFER * 2  # ~0.04%

                elif cascade_mode == MODE_PARTIAL:
                    # Close oldest same-dir + tighten rest
                    # Sort by entry_bar (oldest first)
                    sorted_same = sorted(same_dir_alive, key=lambda p: p['entry_bar'])
                    if sorted_same:
                        oldest = sorted_same[0]
                        entry_price = opens[oldest['entry_bar']]
                        if entry_price > 0:
                            current_price = closes[bar]
                            if oldest['direction'] == 'LONG':
                                pnl = (current_price / entry_price - 1) * 100 * LEVERAGE
                            else:
                                pnl = (1 - current_price / entry_price) * 100 * LEVERAGE
                            pnl -= fee
                            sm = oldest.get('size_mult', 1.0)
                            pnl_portfolio = pnl * (size_pct / 100) * sm
                            trades.append({
                                'entry_bar': oldest['entry_bar'], 'exit_bar': bar,
                                'pnl_slot': pnl, 'reason': 'CASCADE_PARTIAL',
                                'pattern': oldest['pattern'], 'direction': oldest['direction'],
                                'size_mult': sm, 'pnl_portfolio': pnl_portfolio,
                            })
                            closed_slots.append(oldest['slot'])
                            bar_pnl_sum += pnl_portfolio

                        # Tighten rest (use 85% default for partial mode)
                        keep_ratio = 0.15
                        for pos in sorted_same[1:]:
                            sig = pos['signal_bar']
                            if (atr_ratio is not None and sig < len(atr_ratio)
                                    and not np.isnan(atr_ratio[sig])):
                                r = max(clamp_lo, min(clamp_hi, atr_ratio[sig]))
                            else:
                                r = 1.0
                            cur_eff_sl = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                            pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]

        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum)
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard
        if momentum_lookback > 0 and momentum_threshold > 0 and bar >= momentum_lookback:
            price_now = closes[bar]
            price_ago = closes[bar - momentum_lookback]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > momentum_threshold:
                    momentum_pause_until['SHORT'] = bar + momentum_cooldown
                elif pct_change < -momentum_threshold:
                    momentum_pause_until['LONG'] = bar + momentum_cooldown

        # 3. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= n_slots:
                total_blocked['max_pos'] += 1
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                total_blocked['dir_cap'] += 1
                continue
            if any(p['pattern'] == pat for p in positions):
                total_blocked['dup_pat'] += 1
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            if momentum_lookback > 0 and bar < momentum_pause_until.get(direction, -1):
                total_blocked['momentum'] += 1
                continue

            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            if agg_risk_counter > 0 or agg_risk_with > 0:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                cap_pct = agg_risk_counter if is_counter else agg_risk_with
                existing_exposure = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_sl = p['sl_pct']
                        p_sig = p['signal_bar']
                        if (atr_ratio is not None and p_sig < len(atr_ratio)
                                and not np.isnan(atr_ratio[p_sig])):
                            p_r = max(clamp_lo, min(clamp_hi, atr_ratio[p_sig]))
                        else:
                            p_r = 1.0
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / n_slots) * LEVERAGE * p_sm
                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
                new_exposure = sl_pct * new_r * (1.0 / n_slots) * LEVERAGE * sm
                if existing_exposure + new_exposure > cap_pct:
                    total_blocked['agg_risk'] += 1
                    continue

            positions.append({
                'slot': f"{pat}_{sig_bar}", 'signal_bar': sig_bar,
                'entry_bar': entry_bar, 'direction': direction,
                'pattern': pat, 'tp_pct': tp_pct, 'sl_pct': sl_pct, 'size_mult': sm,
            })

        if len(positions) > max_sim_positions:
            max_sim_positions = len(positions)

    # Force-close remaining
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
        })

    return trades, {'max_corr_loss': round(max_corr_loss, 2),
                    'max_sim_positions': max_sim_positions,
                    'cascade_events': cascade_events, 'blocked': total_blocked}


def build_signal_tuples(patterns, signal_index, bar_start, bar_end):
    tuples = []
    for pk, info in patterns.items():
        pn = info.get('pattern') or pk.rsplit('_', 1)[0]
        if pn in signal_index:
            for sb in signal_index[pn]:
                if bar_start <= sb < bar_end:
                    tuples.append((sb, pk, info['direction'],
                                   info['tp_pct'], info['sl_pct']))
    return sorted(tuples, key=lambda x: x[0])


def run_variant(label, signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, ns, ne, cascade_mode, cascade_pct=85):
    trades, meta = portfolio_npos_cascade(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        cascade_mode=cascade_mode, cascade_pct=cascade_pct)
    stats = calc_stats_compound(trades)
    pnl = stats.get('pnl', 0)
    mdd = max(stats.get('mdd', 0), 0.01)

    exit_counts = {}
    for t in trades:
        r = t.get('reason', 'UNKNOWN')
        exit_counts[r] = exit_counts.get(r, 0) + 1

    return {
        'label': label, 'mode': cascade_mode, 'cascade_pct': cascade_pct,
        'trades': stats.get('trades', 0), 'wr': round(stats.get('wr', 0), 1),
        'pnl': round(pnl, 1), 'mdd': round(mdd, 2),
        'pnl_mdd': round(pnl / mdd, 1),
        'cascade_events': meta.get('cascade_events', 0),
        'exit_counts': exit_counts,
    }


def wf_validate(signal_tuples, opens, highs, lows, closes, n_bars, types,
                atr_ratio, ema_slope, ns, ne, cascade_mode, cascade_pct, label):
    """3-fold expanding window WF."""
    logger.info(f"  WF {label}...")
    total_bars = ne - ns
    n_folds = 3
    fs = total_bars // (n_folds + 1)
    folds = []

    for fold in range(n_folds):
        is_end = ns + (fold + 2) * fs
        oos_s = is_end
        oos_e = min(is_end + fs, ne)
        if oos_s >= ne:
            break

        is_si = build_signal_index(types, n_bars)
        is_pats = scan_universe_range(
            is_si, opens, highs, lows, n_bars, ns, is_end, 'per_pattern',
            min_trades=25, edge_threshold=18.0, mc_threshold=0.01,
            atr_ratio=atr_ratio, clamp_lo=DEFAULT_ATR_CLAMP_LO,
            clamp_hi=DEFAULT_ATR_CLAMP_HI)

        full_si = build_signal_index(types, n_bars)
        oos_tups = []
        for pi in is_pats:
            pn, d = pi['pattern'], pi['direction']
            pk = f"{pn}_{d}"
            tp = pi.get('tp_pct', pi.get('tp'))
            sl = pi.get('sl_pct', pi.get('sl'))
            if pn in full_si:
                oos_tups.extend((s, pk, d, tp, sl)
                                for s in full_si[pn] if oos_s <= s < oos_e)
        if not oos_tups:
            folds.append({'fold': fold + 1, 'oos_trades': 0, 'oos_pnl': 0, 'status': 'NO_TRADES'})
            continue

        oos_tups.sort(key=lambda x: x[0])
        trades, meta = portfolio_npos_cascade(
            oos_tups, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_s, oos_e,
            cascade_mode=cascade_mode, cascade_pct=cascade_pct)
        stats = calc_stats_compound(trades)
        pnl = stats.get('pnl', 0)

        folds.append({
            'fold': fold + 1, 'is_patterns': len(is_pats),
            'oos_trades': stats.get('trades', 0),
            'oos_wr': round(stats.get('wr', 0), 1),
            'oos_pnl': round(pnl, 1),
            'oos_mdd': round(stats.get('mdd', 0), 2),
            'status': 'PASS' if pnl > 0 else 'FAIL',
        })

    pc = sum(1 for f in folds if f.get('status') == 'PASS')
    verdict = 'PASS' if pc == len(folds) and len(folds) > 0 else 'FAIL'
    logger.info(f"    {label}: {verdict} ({pc}/{len(folds)}) "
                + " | ".join(f"F{f['fold']}:{f.get('oos_pnl',0):+.1f}%" for f in folds))
    return {'label': label, 'folds': folds, 'pass_count': pc,
            'total_folds': len(folds), 'verdict': verdict}


def random_disc(signal_tuples, opens, highs, lows, closes, n_bars, types,
                atr_ratio, ema_slope, ns, ne, cascade_mode, cascade_pct, n_seeds=10):
    """Random discrimination test."""
    total_bars = ne - ns
    n_folds = 3
    fs = total_bars // (n_folds + 1)
    pass_count = 0

    for seed in range(1, n_seeds + 1):
        rng = np.random.RandomState(seed)
        all_bars = list(range(ns, ne - 1))
        n_sigs = min(len(signal_tuples), len(all_bars))
        rand_bars = sorted(rng.choice(all_bars, size=n_sigs, replace=False))

        pat_keys = list(set(s[1] for s in signal_tuples))
        pat_lookup = {s[1]: s for s in signal_tuples if s[1] not in {}}
        for s in signal_tuples:
            pat_lookup.setdefault(s[1], s)

        rand_tups = []
        for rb in rand_bars:
            pk = rng.choice(pat_keys)
            ref = pat_lookup[pk]
            rand_tups.append((rb, pk, ref[2], ref[3], ref[4]))

        all_pass = True
        for fold in range(n_folds):
            is_end = ns + (fold + 2) * fs
            oos_s = is_end
            oos_e = min(is_end + fs, ne)
            if oos_s >= ne:
                all_pass = False
                break
            oos_tups = sorted(
                [(s, pk, d, tp, sl) for s, pk, d, tp, sl in rand_tups if oos_s <= s < oos_e],
                key=lambda x: x[0])
            if not oos_tups:
                all_pass = False
                break
            trades, _ = portfolio_npos_cascade(
                oos_tups, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, oos_s, oos_e,
                cascade_mode=cascade_mode, cascade_pct=cascade_pct)
            stats = calc_stats_compound(trades)
            if stats.get('pnl', 0) <= 0:
                all_pass = False
                break
        if all_pass:
            pass_count += 1

    pct = pass_count / n_seeds * 100
    return {'n_seeds': n_seeds, 'pass_count': pass_count, 'pass_pct': pct,
            'verdict': 'NON-DISCRIMINATING' if pct >= 80 else 'DISCRIMINATING'}


def main():
    t0 = time.time()
    logger.info("=== Cascade SL Optimization Study ===")

    df = load_and_classify(DATA_FILE)
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    n_bars = len(df)
    types = df['candle_type'].values
    ns, ne = find_neutral_window(closes)
    logger.info(f"Neutral window: {ns}-{ne} ({(ne-ns)/288:.0f}d)")

    atr_ratio = compute_atr_ratio(highs, lows, closes,
                                   atr_period=DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
    ema_slope = compute_ema_slope(closes)

    signal_index = build_signal_index(types, n_bars)
    patterns = load_patterns()
    signal_tuples = build_signal_tuples(patterns, signal_index, ns, ne)
    logger.info(f"{len(patterns)} patterns, {len(signal_tuples)} signals")

    results = {'metadata': {
        'script': 'cascade_sl_optimization.py',
        'date': pd.Timestamp.now().isoformat(),
        'n_patterns': len(patterns), 'n_signals': len(signal_tuples),
        'neutral_window': [int(ns), int(ne)],
    }}

    # Phase 1: IS sweep
    logger.info("=== Phase 1: IS Sweep ===")
    variants = []

    # Tighten % sweep
    for pct in [0, 50, 60, 70, 75, 80, 85, 90, 95]:
        label = f'tighten_{pct}' if pct > 0 else 'cascade_off'
        v = run_variant(label, signal_tuples, opens, highs, lows, closes,
                        n_bars, atr_ratio, ema_slope, ns, ne,
                        MODE_TIGHTEN, pct)
        variants.append(v)
        logger.info(f"  {label}: {v['trades']}t WR{v['wr']:.1f}% "
                    f"PnL{v['pnl']:+.1f}% MDD{v['mdd']:.2f}% PM{v['pnl_mdd']:.1f} "
                    f"cascades={v['cascade_events']}")

    # Alternative mechanisms
    for mode, label in [(MODE_IMMEDIATE, 'immediate_close'),
                        (MODE_BREAKEVEN, 'breakeven_move'),
                        (MODE_PARTIAL, 'partial_close')]:
        v = run_variant(label, signal_tuples, opens, highs, lows, closes,
                        n_bars, atr_ratio, ema_slope, ns, ne, mode)
        variants.append(v)
        logger.info(f"  {label}: {v['trades']}t WR{v['wr']:.1f}% "
                    f"PnL{v['pnl']:+.1f}% MDD{v['mdd']:.2f}% PM{v['pnl_mdd']:.1f} "
                    f"cascades={v['cascade_events']}")

    results['phase1_sweep'] = {v['label']: v for v in variants}

    # Sort by PnL/MDD
    ranked = sorted(variants, key=lambda x: x['pnl_mdd'], reverse=True)
    logger.info(f"\n  Ranking by PnL/MDD:")
    for i, v in enumerate(ranked[:5]):
        logger.info(f"    #{i+1} {v['label']}: PM{v['pnl_mdd']:.1f} "
                    f"PnL{v['pnl']:+.1f}% MDD{v['mdd']:.2f}%")

    # Phase 2: WF for top-5
    logger.info("\n=== Phase 2: WF Validation (top-5) ===")
    results['phase2_wf'] = {}
    for v in ranked[:5]:
        wf = wf_validate(signal_tuples, opens, highs, lows, closes, n_bars, types,
                         atr_ratio, ema_slope, ns, ne,
                         v['mode'], v['cascade_pct'], v['label'])
        results['phase2_wf'][v['label']] = wf

    # Phase 3: Random discrimination for top-3
    logger.info("\n=== Phase 3: Random Discrimination (top-3) ===")
    results['phase3_random'] = {}
    for v in ranked[:3]:
        rd = random_disc(signal_tuples, opens, highs, lows, closes, n_bars, types,
                         atr_ratio, ema_slope, ns, ne,
                         v['mode'], v['cascade_pct'])
        results['phase3_random'][v['label']] = rd
        logger.info(f"  {v['label']}: {rd['pass_count']}/{rd['n_seeds']} pass "
                    f"({rd['pass_pct']:.0f}%) → {rd['verdict']}")

    elapsed = time.time() - t0
    results['metadata']['runtime_sec'] = round(elapsed, 1)

    # Verdict
    current = next(v for v in variants if v['label'] == 'tighten_85')
    best = ranked[0]

    logger.info(f"\n{'='*70}")
    logger.info(f"CASCADE SL OPTIMIZATION — VERDICT")
    logger.info(f"{'='*70}")
    logger.info(f"Current (85%): PM{current['pnl_mdd']:.1f} PnL{current['pnl']:+.1f}% MDD{current['mdd']:.2f}%")
    logger.info(f"Best ({best['label']}): PM{best['pnl_mdd']:.1f} PnL{best['pnl']:+.1f}% MDD{best['mdd']:.2f}%")

    wf_best = results['phase2_wf'].get(best['label'], {}).get('verdict', 'N/A')
    wf_current = results['phase2_wf'].get('tighten_85', {}).get('verdict', 'N/A')
    rand_best = results['phase3_random'].get(best['label'], {}).get('verdict', 'N/A')

    delta_pm = best['pnl_mdd'] - current['pnl_mdd']
    if best['label'] == 'tighten_85':
        verdict = 'KEEP_CURRENT'
        reason = '현행 85%가 이미 최선'
    elif delta_pm > current['pnl_mdd'] * 0.1 and wf_best == 'PASS':
        verdict = 'GO' if rand_best != 'NON-DISCRIMINATING' else 'STOP_NON_DISC'
        reason = f'{best["label"]} PM +{delta_pm:.0f} WF {wf_best} Random {rand_best}'
    else:
        verdict = 'KEEP_CURRENT'
        reason = f'개선 미미 또는 WF FAIL ({best["label"]} PM{best["pnl_mdd"]:.1f} WF {wf_best})'

    results['verdict'] = {
        'decision': verdict, 'reason': reason,
        'current': 'tighten_85', 'best': best['label'],
        'current_pm': current['pnl_mdd'], 'best_pm': best['pnl_mdd'],
        'delta_pm': round(delta_pm, 1),
    }
    logger.info(f"Verdict: {verdict} — {reason}")
    logger.info(f"Runtime: {elapsed:.1f}s")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
