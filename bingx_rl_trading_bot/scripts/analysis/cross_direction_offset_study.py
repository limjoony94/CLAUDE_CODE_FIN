#!/usr/bin/env python3
"""Cross-Direction Offset Study.

Research question:
  When entering a position opposite to existing positions, does the new entry's SL
  cancel out existing positions' TP, resulting in net loss?

Approach:
  Phase 1: Analyze baseline N-pos trades for opposite-direction overlap events
  Phase 2: Pre-filter signals (remove entries where SL > opposite TP on equity basis)
           and re-run portfolio_npos
  Phase 3: Threshold sweep for optimal filter strength
  Phase 4: WF validation
  Phase 5: Random discrimination

Protocol: Entry next-bar open, intrabar exit, Fee 0.10%, Slippage 0.02%, Lev 3x, compound.
"""
import os, sys, json, logging, time, copy
import numpy as np, pandas as pd
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))

from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    portfolio_npos, calc_stats_compound, scan_universe_range,
    compute_atr_ratio, compute_ema_slope,
    FEE_PCT, LEVERAGE, MAX_BARS, SLIPPAGE_BUFFER,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    TIMEOUT_BARS, _check_exit_npos,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'cross_direction_offset_study.json')


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'],
            'direction': v['direction'],
            'tp_pct': v['tp'],
            'sl_pct': v['sl'],
        }
    return result


def build_signal_tuples(patterns, signal_index, bar_start, bar_end):
    tuples = []
    for pat_key, info in patterns.items():
        pat_name = info.get('pattern') or pat_key.rsplit('_', 1)[0]
        if pat_name in signal_index:
            for sig_bar in signal_index[pat_name]:
                if bar_start <= sig_bar < bar_end:
                    tuples.append((sig_bar, pat_key, info['direction'],
                                   info['tp_pct'], info['sl_pct']))
    return sorted(tuples, key=lambda x: x[0])


def run_portfolio(signal_tuples, opens, highs, lows, closes, n_bars,
                  atr_ratio, ema_slope, ns, ne, label=''):
    """Run standard portfolio_npos and return trades, stats."""
    trades, _meta = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne)
    stats = calc_stats_compound(trades)
    return trades, stats


def phase1_overlap_analysis(trades):
    """Phase 1: Analyze co-existing opposite-direction positions from trade data."""
    logger.info("=== Phase 1: Opposite-Direction Overlap Analysis ===")

    if not trades:
        return {'error': 'no trades'}

    # Build timeline of active positions
    # Each trade has entry_bar, exit_bar, direction, pnl_slot
    trade_list = []
    for t in trades:
        trade_list.append({
            'entry': t['entry_bar'],
            'exit': t['exit_bar'],
            'dir': t['direction'],
            'pnl': t['pnl_slot'],
            'reason': t['reason'],
            'pattern': t['pattern'],
            'tp_pct': None,  # not in trade result, but we have pnl
            'sl_pct': None,
        })

    # Find overlapping opposite-direction pairs
    overlap_events = []
    for i, t1 in enumerate(trade_list):
        for j, t2 in enumerate(trade_list):
            if i >= j:
                continue
            if t1['dir'] == t2['dir']:
                continue
            # Check temporal overlap
            overlap_start = max(t1['entry'], t2['entry'])
            overlap_end = min(t1['exit'], t2['exit'])
            if overlap_start < overlap_end:
                # These two positions were co-existing with opposite directions
                later = t1 if t1['entry'] > t2['entry'] else t2
                earlier = t2 if later is t1 else t1
                overlap_events.append({
                    'earlier_dir': earlier['dir'],
                    'earlier_pnl': earlier['pnl'],
                    'earlier_reason': earlier['reason'],
                    'later_dir': later['dir'],
                    'later_pnl': later['pnl'],
                    'later_reason': later['reason'],
                    'overlap_bars': overlap_end - overlap_start,
                    'combined_pnl': earlier['pnl'] + later['pnl'],
                })

    if not overlap_events:
        logger.info("No opposite-direction overlaps found")
        return {'total_overlaps': 0}

    # Statistics
    combined_pnls = [e['combined_pnl'] for e in overlap_events]
    net_negative = [e for e in overlap_events if e['combined_pnl'] < 0]
    net_positive = [e for e in overlap_events if e['combined_pnl'] > 0]

    # Cases where later entry's SL wipes earlier's TP
    sl_wipes_tp = [e for e in overlap_events
                   if (e['later_reason'] == 'SL' and e['earlier_reason'] == 'TP'
                       and e['combined_pnl'] < 0)]
    tp_wipes_sl = [e for e in overlap_events
                   if (e['later_reason'] == 'TP' and e['earlier_reason'] == 'SL'
                       and e['combined_pnl'] > 0)]

    result = {
        'total_overlaps': len(overlap_events),
        'net_negative': len(net_negative),
        'net_positive': len(net_positive),
        'net_zero': len(overlap_events) - len(net_negative) - len(net_positive),
        'avg_combined_pnl': round(np.mean(combined_pnls), 3),
        'median_combined_pnl': round(np.median(combined_pnls), 3),
        'total_combined_pnl': round(sum(combined_pnls), 1),
        'sl_wipes_tp_count': len(sl_wipes_tp),
        'tp_wipes_sl_count': len(tp_wipes_sl),
        'pct_negative': round(len(net_negative) / len(overlap_events) * 100, 1),
    }

    if sl_wipes_tp:
        result['sl_wipes_tp_avg_loss'] = round(np.mean([e['combined_pnl'] for e in sl_wipes_tp]), 3)

    logger.info(f"  Overlaps: {len(overlap_events)} total, "
                f"{len(net_negative)} net-negative ({result['pct_negative']:.1f}%), "
                f"{len(net_positive)} net-positive")
    logger.info(f"  Avg combined PnL: {result['avg_combined_pnl']:.3f}%")
    logger.info(f"  SL-wipes-TP events: {len(sl_wipes_tp)}")

    return result


def phase2_filter_test(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, ns, ne, patterns):
    """Phase 2: Compare baseline vs filtered (remove unfavorable cross-dir entries).

    Filter logic: Before entering, check if new entry's SL (equity %) >
    average opposite-direction TP (equity %). If so, skip.
    This is done by pre-computing per-signal the SL/TP equity impact and
    simulating with the filter embedded.
    """
    logger.info("=== Phase 2: Cross-Direction Offset Filter Test ===")

    # Build pattern lookup for TP/SL
    pat_info = {}
    for pk, info in patterns.items():
        pat_info[pk] = {'tp': info['tp_pct'], 'sl': info['sl_pct'], 'dir': info['direction']}

    # Run baseline
    trades_bl, _m = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne)
    stats_bl = calc_stats_compound(trades_bl)

    bl_pnl = stats_bl.get('pnl', 0)
    bl_mdd = max(stats_bl.get('mdd', 0), 0.01)
    bl_wr = stats_bl.get('wr', 0)

    logger.info(f"  Baseline: {len(trades_bl)}t WR{bl_wr:.1f}% PnL{bl_pnl:+.1f}% MDD{bl_mdd:.2f}%")

    # Now we need to test the filter. Since portfolio_npos doesn't support
    # custom entry filters, we'll use a signal pre-filtering approach:
    # Remove signals that are "opposing" to the dominant direction at that time.
    # This is an approximation — true filtering needs position state awareness.

    # Approach: simulate position state externally, mark signals to keep/skip,
    # then run portfolio_npos with filtered signals.
    # To approximate: for each signal, estimate how many same/opposite positions
    # would exist based on recent signal history.

    # Simpler approximation: group signals by time window and direction.
    # If within a window, the dominant direction has more signals,
    # filter opposing signals where SL > dominant TP.

    # Actually, the most accurate approach: modify signal_tuples by removing
    # entries where the SL equity impact > opposite direction's average TP equity impact.

    slot_size = 100.0 / DEFAULT_N_SLOTS  # percent
    results = {}

    for ratio_name, ratio_thresh in [('strict', 1.0), ('relaxed', 1.5), ('loose', 2.0)]:
        # For each signal, check if SL > ratio_thresh * average opposite TP
        # Use per-pattern ATR to scale
        filtered_sigs = []
        blocked = 0

        # Group signals by bar to process sequentially
        sig_by_bar = defaultdict(list)
        for s in signal_tuples:
            sig_by_bar[s[0]].append(s)

        # Track "active" signals (approximate, window-based)
        recent_window = 288  # ~24h
        active_long_tps = []  # rolling window of LONG TP values
        active_short_tps = []

        for bar in sorted(sig_by_bar.keys()):
            # Age out old entries
            active_long_tps = [(b, tp) for b, tp in active_long_tps if bar - b < recent_window]
            active_short_tps = [(b, tp) for b, tp in active_short_tps if bar - b < recent_window]

            for sig in sig_by_bar[bar]:
                sig_bar, pat_key, direction, tp_pct, sl_pct = sig

                # ATR scaling
                nr = 1.0
                entry_bar = sig_bar + 1
                if atr_ratio is not None and entry_bar < len(atr_ratio):
                    r = atr_ratio[entry_bar]
                    if not np.isnan(r):
                        nr = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, r))

                new_sl_eq = sl_pct * nr * (slot_size / 100) * LEVERAGE
                new_tp_eq = tp_pct * nr * (slot_size / 100) * LEVERAGE

                # Check opposite direction TPs
                opp_tps = active_short_tps if direction == 'LONG' else active_long_tps
                if opp_tps:
                    avg_opp_tp = np.mean([tp for _, tp in opp_tps])
                    # Filter: skip if my SL loss > ratio × opposite avg TP gain
                    if new_sl_eq > ratio_thresh * avg_opp_tp:
                        blocked += 1
                        continue

                filtered_sigs.append(sig)
                # Track this signal
                if direction == 'LONG':
                    active_long_tps.append((bar, new_tp_eq))
                else:
                    active_short_tps.append((bar, new_tp_eq))

        # Run portfolio with filtered signals
        trades_f, _meta_f = portfolio_npos(
            filtered_sigs, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, ns, ne)
        stats_f = calc_stats_compound(trades_f)

        f_pnl = stats_f.get('pnl', 0)
        f_mdd = max(stats_f.get('mdd', 0), 0.01)
        f_wr = stats_f.get('wr', 0)
        f_pm = f_pnl / f_mdd

        results[ratio_name] = {
            'ratio': ratio_thresh,
            'trades': len(trades_f), 'wr': round(f_wr, 1),
            'pnl': round(f_pnl, 1), 'mdd': round(f_mdd, 2),
            'pnl_mdd': round(f_pm, 1),
            'blocked': blocked,
            'blocked_pct': round(blocked / max(len(signal_tuples), 1) * 100, 1),
            'delta_pnl': round(f_pnl - bl_pnl, 1),
            'delta_mdd': round(f_mdd - bl_mdd, 2),
        }
        logger.info(f"  {ratio_name}(×{ratio_thresh}): {len(trades_f)}t WR{f_wr:.1f}% "
                    f"PnL{f_pnl:+.1f}% MDD{f_mdd:.2f}% blocked={blocked} "
                    f"dPnL{f_pnl-bl_pnl:+.1f}%")

    results['baseline'] = {
        'trades': len(trades_bl), 'wr': round(bl_wr, 1),
        'pnl': round(bl_pnl, 1), 'mdd': round(bl_mdd, 2),
        'pnl_mdd': round(bl_pnl / bl_mdd, 1),
    }

    return results


def phase3_wf(signal_tuples_full, opens, highs, lows, closes, n_bars, types,
              atr_ratio, ema_slope, ns, ne, patterns, filter_ratio, label):
    """Phase 3: WF validation with cross-direction filter."""
    logger.info(f"=== Phase 3: WF — {label} (ratio={filter_ratio}) ===")

    total_bars = ne - ns
    n_folds = 3
    fs = total_bars // (n_folds + 1)
    folds = []

    pat_info = {pk: {'tp': v['tp_pct'], 'sl': v['sl_pct'], 'dir': v['direction']}
                for pk, v in patterns.items()}
    slot_size = 100.0 / DEFAULT_N_SLOTS

    for fold in range(n_folds):
        is_end = ns + (fold + 2) * fs
        oos_s = is_end
        oos_e = min(is_end + fs, ne)
        if oos_s >= ne:
            break

        # IS pattern discovery
        is_si = build_signal_index(types, n_bars)
        is_pats = scan_universe_range(
            is_si, opens, highs, lows, n_bars, ns, is_end, 'per_pattern',
            min_trades=25, edge_threshold=18.0, mc_threshold=0.01,
            atr_ratio=atr_ratio, clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI)

        # Build OOS signals
        full_si = build_signal_index(types, n_bars)
        oos_tups = []
        for pi in is_pats:
            pn, d = pi['pattern'], pi['direction']
            pk = f"{pn}_{d}"
            tp = pi.get('tp_pct', pi.get('tp'))
            sl = pi.get('sl_pct', pi.get('sl'))
            if pn in full_si:
                oos_tups.extend(
                    (s, pk, d, tp, sl)
                    for s in full_si[pn] if oos_s <= s < oos_e)

        if not oos_tups:
            folds.append({'fold': fold + 1, 'oos_trades': 0, 'oos_pnl': 0, 'status': 'NO_TRADES'})
            continue

        # Apply cross-direction filter if ratio specified
        if filter_ratio is not None:
            oos_tups = _apply_offset_filter(oos_tups, atr_ratio, filter_ratio, slot_size)

        oos_tups.sort(key=lambda x: x[0])
        trades, _meta = portfolio_npos(
            oos_tups, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_s, oos_e)
        stats = calc_stats_compound(trades)

        pnl = stats.get('pnl', 0)
        folds.append({
            'fold': fold + 1, 'is_patterns': len(is_pats),
            'oos_trades': len(trades), 'oos_wr': round(stats.get('wr', 0), 1),
            'oos_pnl': round(pnl, 1),
            'status': 'PASS' if pnl > 0 else 'FAIL',
        })
        logger.info(f"  {label} F{fold+1}: OOS {len(trades)}t WR{stats.get('wr',0):.1f}% "
                    f"PnL{pnl:+.1f}% {'PASS' if pnl > 0 else 'FAIL'}")

    pc = sum(1 for f in folds if f.get('status') == 'PASS')
    verdict = 'PASS' if pc == len(folds) and len(folds) > 0 else 'FAIL'
    logger.info(f"  WF {label}: {verdict} ({pc}/{len(folds)})")
    return {'label': label, 'filter_ratio': filter_ratio, 'folds': folds,
            'pass_count': pc, 'total_folds': len(folds), 'verdict': verdict}


def _apply_offset_filter(signal_tuples, atr_ratio, ratio_thresh, slot_size):
    """Pre-filter signals by cross-direction offset criterion."""
    filtered = []
    active_long_tps = []
    active_short_tps = []
    recent_window = 288

    sig_by_bar = defaultdict(list)
    for s in signal_tuples:
        sig_by_bar[s[0]].append(s)

    for bar in sorted(sig_by_bar.keys()):
        active_long_tps = [(b, tp) for b, tp in active_long_tps if bar - b < recent_window]
        active_short_tps = [(b, tp) for b, tp in active_short_tps if bar - b < recent_window]

        for sig in sig_by_bar[bar]:
            sig_bar, pat_key, direction, tp_pct, sl_pct = sig
            nr = 1.0
            entry_bar = sig_bar + 1
            if atr_ratio is not None and entry_bar < len(atr_ratio):
                r = atr_ratio[entry_bar]
                if not np.isnan(r):
                    nr = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, r))

            new_sl_eq = sl_pct * nr * (slot_size / 100) * LEVERAGE
            new_tp_eq = tp_pct * nr * (slot_size / 100) * LEVERAGE

            opp_tps = active_short_tps if direction == 'LONG' else active_long_tps
            if opp_tps:
                avg_opp_tp = np.mean([tp for _, tp in opp_tps])
                if new_sl_eq > ratio_thresh * avg_opp_tp:
                    continue

            filtered.append(sig)
            if direction == 'LONG':
                active_long_tps.append((bar, new_tp_eq))
            else:
                active_short_tps.append((bar, new_tp_eq))

    return filtered


def phase4_random(signal_tuples, opens, highs, lows, closes, n_bars, types,
                  atr_ratio, ema_slope, ns, ne, filter_ratio, n_seeds=10):
    """Phase 4: Random discrimination."""
    logger.info(f"=== Phase 4: Random Discrimination (ratio={filter_ratio}) ===")

    total_bars = ne - ns
    n_folds = 3
    fs = total_bars // (n_folds + 1)
    slot_size = 100.0 / DEFAULT_N_SLOTS
    pass_count = 0

    for seed in range(1, n_seeds + 1):
        rng = np.random.RandomState(seed)
        all_bars = list(range(ns, ne - 1))
        n_sigs = min(len(signal_tuples), len(all_bars))
        rand_bars = sorted(rng.choice(all_bars, size=n_sigs, replace=False))

        pat_keys = list(set(s[1] for s in signal_tuples))
        pat_lookup = {}
        for s in signal_tuples:
            if s[1] not in pat_lookup:
                pat_lookup[s[1]] = s

        rand_tups = []
        for rb in rand_bars:
            pk = rng.choice(pat_keys)
            ref = pat_lookup[pk]
            rand_tups.append((rb, pk, ref[2], ref[3], ref[4]))

        if filter_ratio is not None:
            rand_tups = _apply_offset_filter(rand_tups, atr_ratio, filter_ratio, slot_size)

        all_pass = True
        for fold in range(n_folds):
            is_end = ns + (fold + 2) * fs
            oos_s = is_end
            oos_e = min(is_end + fs, ne)
            if oos_s >= ne:
                all_pass = False
                break
            oos_tups = [(s, pk, d, tp, sl) for s, pk, d, tp, sl in rand_tups
                        if oos_s <= s < oos_e]
            if not oos_tups:
                all_pass = False
                break
            oos_tups.sort(key=lambda x: x[0])
            trades, _meta = portfolio_npos(
                oos_tups, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, oos_s, oos_e)
            stats = calc_stats_compound(trades)
            if stats.get('pnl', 0) <= 0:
                all_pass = False
                break
        if all_pass:
            pass_count += 1

    pct = pass_count / n_seeds * 100
    logger.info(f"  Random: {pass_count}/{n_seeds} pass ({pct:.0f}%)")
    return {'n_seeds': n_seeds, 'pass_count': pass_count, 'pass_pct': pct,
            'verdict': 'NON-DISCRIMINATING' if pct >= 80 else 'DISCRIMINATING'}


def main():
    t0 = time.time()
    logger.info("=== Cross-Direction Offset Study ===")

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
        'script': 'cross_direction_offset_study.py',
        'date': pd.Timestamp.now().isoformat(),
        'n_patterns': len(patterns),
        'n_signals': len(signal_tuples),
        'neutral_window': [int(ns), int(ne)],
    }}

    # Phase 1: Overlap analysis from baseline trades
    trades_bl, _meta_bl = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne)
    stats_bl = calc_stats_compound(trades_bl)
    bl_pnl = stats_bl.get('pnl', 0)
    bl_mdd = max(stats_bl.get('mdd', 0), 0.01)
    logger.info(f"Baseline: {len(trades_bl)}t WR{stats_bl.get('wr',0):.1f}% "
                f"PnL{bl_pnl:+.1f}% MDD{bl_mdd:.2f}% PnL/MDD{bl_pnl/bl_mdd:.1f}")

    results['baseline'] = {
        'trades': len(trades_bl), 'wr': round(stats_bl.get('wr', 0), 1),
        'pnl': round(bl_pnl, 1), 'mdd': round(bl_mdd, 2),
        'pnl_mdd': round(bl_pnl / bl_mdd, 1),
    }

    results['phase1'] = phase1_overlap_analysis(trades_bl)

    # Phase 2: Filter test with different ratios
    results['phase2'] = phase2_filter_test(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne, patterns)

    # Find best filter from phase 2
    best_ratio = None
    best_pm = bl_pnl / bl_mdd
    for name, v in results['phase2'].items():
        if name == 'baseline':
            continue
        pm = v.get('pnl_mdd', 0)
        if pm > best_pm:
            best_pm = pm
            best_ratio = v['ratio']

    # Phase 3: WF validation
    wf_bl = phase3_wf(signal_tuples, opens, highs, lows, closes, n_bars, types,
                      atr_ratio, ema_slope, ns, ne, patterns, None, 'baseline')

    results['phase3'] = {'baseline': wf_bl}

    if best_ratio is not None:
        wf_best = phase3_wf(signal_tuples, opens, highs, lows, closes, n_bars, types,
                            atr_ratio, ema_slope, ns, ne, patterns, best_ratio,
                            f'filter_{best_ratio}')
        results['phase3']['best_filter'] = wf_best

    # Phase 4: Random discrimination
    results['phase4'] = {
        'baseline': phase4_random(signal_tuples, opens, highs, lows, closes, n_bars, types,
                                  atr_ratio, ema_slope, ns, ne, None),
    }
    if best_ratio is not None:
        results['phase4']['best_filter'] = phase4_random(
            signal_tuples, opens, highs, lows, closes, n_bars, types,
            atr_ratio, ema_slope, ns, ne, best_ratio)

    elapsed = time.time() - t0
    results['metadata']['runtime_sec'] = round(elapsed, 1)

    # Verdict
    p1 = results['phase1']
    p2 = results['phase2']
    logger.info(f"\n{'='*70}")
    logger.info(f"CROSS-DIRECTION OFFSET STUDY — SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Baseline: {results['baseline']['trades']}t "
                f"PnL{results['baseline']['pnl']:+.1f}% PnL/MDD{results['baseline']['pnl_mdd']:.1f}")

    if p1.get('total_overlaps', 0) > 0:
        logger.info(f"Overlaps: {p1['total_overlaps']} total, "
                    f"{p1['pct_negative']:.1f}% net-negative, "
                    f"SL-wipes-TP: {p1.get('sl_wipes_tp_count', 0)}")

    for name in ['strict', 'relaxed', 'loose']:
        if name in p2:
            v = p2[name]
            logger.info(f"  {name}(×{v['ratio']}): dPnL{v['delta_pnl']:+.1f}% "
                        f"PnL/MDD{v['pnl_mdd']:.1f} blocked={v['blocked']}")

    if best_ratio is not None:
        logger.info(f"\nBest filter: ×{best_ratio} (PnL/MDD {best_pm:.1f})")
        logger.info(f"WF: {results['phase3'].get('best_filter', {}).get('verdict', 'N/A')}")
        logger.info(f"Random: {results['phase4'].get('best_filter', {}).get('verdict', 'N/A')}")
        verdict = 'GO' if best_pm > bl_pnl / bl_mdd * 1.1 else 'MARGINAL'
    else:
        logger.info("\nNo filter improved PnL/MDD over baseline")
        verdict = 'STOP'

    results['verdict'] = {
        'decision': verdict,
        'best_ratio': best_ratio,
        'baseline_pnl_mdd': round(bl_pnl / bl_mdd, 1),
        'best_pnl_mdd': round(best_pm, 1),
    }
    logger.info(f"Verdict: {verdict}")
    logger.info(f"Runtime: {elapsed:.1f}s")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
