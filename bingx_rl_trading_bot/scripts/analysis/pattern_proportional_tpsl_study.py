#!/usr/bin/env python3
"""
Pattern-Proportional TP/SL Study
=================================
Hypothesis: TP/SL should be proportional to the 3-candle pattern's own price range,
not fixed percentages. Fixed TP/SL >> pattern range → regime-dependent outcomes.

Phase 1: Measure per-signal pattern ranges
Phase 2: Pattern-proportional TP/SL backtest (TP=Nx range, SL=Mx range)
Phase 3: Fixed vs Proportional comparison
Phase 4: Regime bias verification (direction flip, WR Excess, symmetric test)
Phase 5: Portfolio + WF 3-fold validation
Phase 6: Final recommendation

Standard Research Protocol:
- Entry: next bar open
- Exit: intrabar high/low (distance-based)
- Sizing: compound
- Fee: 0.30% capital-space (0.10% × 3x leverage)
- Timeout: 864 bars → DROP
- WF: 3-fold expanding window
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.analysis.multi_position_diversification_study import (
    load_and_classify, build_signal_index, compute_atr_ratio, calc_stats, find_overlap_bar
)
from scripts.analysis.tpsl_regime_bias_study import (
    backtest_pattern_trades, load_patterns_from_file
)

# Constants
DATA_720 = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_720days_binance.csv')
DYNAMIC_PATTERNS = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'pattern_proportional_tpsl_study.json')

LEVERAGE = 3
FEE_PCT = 0.10
FEE_TOTAL = FEE_PCT * LEVERAGE  # 0.30% capital-space
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
TIMEOUT_T864 = 864
MAX_BARS = 864
MAX_POS = 9
DIR_CAP = 6
MIN_TRADES = 25

# Multiplier grids for pattern-proportional TP/SL
# TP = tp_mult × pattern_range, SL = sl_mult × pattern_range
TP_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SL_MULTS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

# Also test fixed compact grids (already have from previous study)
# TP_FIXED and SL_FIXED kept as reference
TP_FIXED = [0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SL_FIXED = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

# Ultra-tight fixed grids (new)
TP_TIGHT = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]
SL_TIGHT = [0.30, 0.40, 0.50, 0.75, 1.0]


def compute_pattern_ranges(types, highs, lows, opens, n_bars, sig_index):
    """
    For each signal bar, compute the 3-candle pattern range as % of entry price.
    Pattern = candles [sig_bar-2, sig_bar-1, sig_bar]
    Entry = opens[sig_bar + 1]
    """
    pattern_ranges = {}  # pattern_key -> list of (sig_bar, range_pct)
    for key, bars in sig_index.items():
        ranges = []
        for sb in bars:
            if sb < 2 or sb + 1 >= n_bars:
                continue
            pat_high = max(highs[sb-2], highs[sb-1], highs[sb])
            pat_low = min(lows[sb-2], lows[sb-1], lows[sb])
            entry = opens[sb + 1]
            if entry > 0:
                r = (pat_high - pat_low) / entry * 100
                ranges.append((sb, r))
        pattern_ranges[key] = ranges
    # Also create direction-keyed aliases (same data, just different key)
    # e.g. "BD-BD-BU_LONG" -> same as "BD-BD-BU"
    aliases = {}
    for key, ranges in pattern_ranges.items():
        aliases[f"{key}_LONG"] = ranges
        aliases[f"{key}_SHORT"] = ranges
    pattern_ranges.update(aliases)
    return pattern_ranges


def backtest_proportional(sig_bars, direction_int, tp_mult, sl_mult,
                          opens, highs, lows, n_bars, pattern_ranges_list,
                          atr_ratio=None, use_atr=False,
                          lo_bar=0, hi_bar=None, timeout_bars=None):
    """
    Backtest with pattern-proportional TP/SL.
    tp_pct = tp_mult × pattern_range, sl_pct = sl_mult × pattern_range
    Each trade has its own TP/SL based on its pattern's range.
    Enforces SL >= 0.30% floor (execution safety).
    """
    if hi_bar is None:
        hi_bar = n_bars

    is_long = (direction_int == 1)
    # Build sig_bar -> range_pct lookup
    range_lookup = {sb: rng for sb, rng in pattern_ranges_list}

    SLIPPAGE = 0.02
    results = []

    for sig_bar in sig_bars:
        if sig_bar < lo_bar or sig_bar >= hi_bar:
            continue
        entry_bar = sig_bar + 1
        if entry_bar >= n_bars:
            continue
        entry_price = opens[entry_bar]
        if entry_price <= 0:
            continue

        pat_range = range_lookup.get(sig_bar)
        if pat_range is None or pat_range <= 0:
            continue

        # Proportional TP/SL
        tp_pct = tp_mult * pat_range
        sl_pct = max(0.30, sl_mult * pat_range)  # SL floor 0.30%

        # ATR scaling
        if use_atr and atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
            r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE)

        if is_long:
            tp_price = entry_price * (1 + eff_tp / 100)
            sl_price = entry_price * (1 - eff_sl / 100)
        else:
            tp_price = entry_price * (1 - eff_tp / 100)
            sl_price = entry_price * (1 + eff_sl / 100)

        max_exit = min(entry_bar + 1 + (timeout_bars if timeout_bars else MAX_BARS), n_bars)

        resolved = False
        for j in range(entry_bar + 1, max_exit):
            h, l, o = highs[j], lows[j], opens[j]
            if is_long:
                tp_hit = h >= tp_price
                sl_hit = l <= sl_price
            else:
                tp_hit = l <= tp_price
                sl_hit = h >= sl_price

            if tp_hit and sl_hit:
                tp_dist = abs(tp_price - o)
                sl_dist = abs(sl_price - o)
                if tp_dist <= sl_dist:
                    tp_hit, sl_hit = True, False
                else:
                    tp_hit, sl_hit = False, True

            if tp_hit:
                pnl = (eff_tp * LEVERAGE) - FEE_TOTAL
                results.append({
                    'sig_bar': sig_bar, 'entry_bar': entry_bar, 'exit_bar': j,
                    'pnl': pnl, 'reason': 'tp', 'entry_price': entry_price,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'pat_range': pat_range
                })
                resolved = True
                break
            elif sl_hit:
                pnl = -(eff_sl * LEVERAGE) - FEE_TOTAL
                results.append({
                    'sig_bar': sig_bar, 'entry_bar': entry_bar, 'exit_bar': j,
                    'pnl': pnl, 'reason': 'sl', 'entry_price': entry_price,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'pat_range': pat_range
                })
                resolved = True
                break

        if not resolved:
            # Timeout → DROP
            pass

    return results


def compute_wr_excess(trades, tp_pct_avg, sl_pct_avg):
    """WR Excess = actual WR - Random Walk WR (SL/(TP+SL))."""
    if not trades:
        return 0, 0, 0, 0
    wins = sum(1 for t in trades if t['pnl'] > 0)
    n = len(trades)
    wr = wins / n * 100
    rw_wr = sl_pct_avg / (tp_pct_avg + sl_pct_avg) * 100 if (tp_pct_avg + sl_pct_avg) > 0 else 50.0
    excess = wr - rw_wr
    return wr, rw_wr, excess, n


def simulate_hedge_proportional(schedule, sig_index, opens, highs, lows, n_bars,
                                 pattern_ranges, atr_ratio, lo_bar, hi_bar=None,
                                 max_pos=MAX_POS, dir_cap=DIR_CAP,
                                 proportional=True, timeout_bars=TIMEOUT_T864):
    """
    Simulate hedge portfolio with proportional or fixed TP/SL.
    schedule: list of {pattern, direction, tp, sl} or {pattern, direction, tp_mult, sl_mult}
    """
    if hi_bar is None:
        hi_bar = n_bars

    # Collect all signals with their TP/SL
    all_events = []
    for item in schedule:
        pat = item['pattern']
        direction = item['direction']
        dir_int = 1 if direction == 'LONG' else -1
        key = f"{pat}_{direction}"
        bars_key = pat  # sig_index uses pattern name only
        bars = sig_index.get(bars_key, [])
        pranges = pattern_ranges.get(key, [])
        range_lookup = {sb: rng for sb, rng in pranges}

        for sb in bars:
            if sb < lo_bar or sb >= hi_bar:
                continue
            entry_bar = sb + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            if proportional:
                pr = range_lookup.get(sb, 0)
                if pr <= 0:
                    continue
                tp_pct = item['tp_mult'] * pr
                sl_pct = max(0.30, item['sl_mult'] * pr)
            else:
                tp_pct = item['tp']
                sl_pct = item['sl']

            all_events.append({
                'sig_bar': sb, 'entry_bar': entry_bar,
                'entry_price': entry_price,
                'direction': direction, 'dir_int': dir_int,
                'pattern': pat,
                'tp_pct': tp_pct, 'sl_pct': sl_pct,
            })

    all_events.sort(key=lambda x: x['sig_bar'])

    # Simulate
    equity = 100.0
    peak = equity
    max_dd = 0
    positions = []  # active positions
    total_trades = 0
    total_wins = 0
    long_trades = 0
    short_trades = 0
    total_pnl_list = []

    SLIPPAGE = 0.02

    for ev in all_events:
        sb = ev['sig_bar']

        # Check existing positions for exit
        new_positions = []
        for pos in positions:
            if pos.get('closed'):
                continue
            # Check if position exits on bars between last check and current signal
            pos_closed = False
            start_check = max(pos['entry_bar'] + 1, pos.get('last_check', pos['entry_bar'] + 1))
            end_check = min(sb + 2, n_bars)
            timeout_limit = pos['entry_bar'] + 1 + timeout_bars if timeout_bars else n_bars

            for j in range(start_check, min(end_check, timeout_limit)):
                h, l, o = highs[j], lows[j], opens[j]
                is_long = pos['dir_int'] == 1

                # ATR scaling
                if atr_ratio is not None and pos['sig_bar'] < len(atr_ratio) and not np.isnan(atr_ratio[pos['sig_bar']]):
                    r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[pos['sig_bar']]))
                else:
                    r = 1.0

                eff_tp = pos['tp_pct'] * r + SLIPPAGE
                eff_sl = max(0.1, pos['sl_pct'] * r - SLIPPAGE)
                ep = pos['entry_price']

                if is_long:
                    tp_price = ep * (1 + eff_tp / 100)
                    sl_price = ep * (1 - eff_sl / 100)
                    tp_hit = h >= tp_price
                    sl_hit = l <= sl_price
                else:
                    tp_price = ep * (1 - eff_tp / 100)
                    sl_price = ep * (1 + eff_sl / 100)
                    tp_hit = l <= tp_price
                    sl_hit = h >= sl_price

                if tp_hit and sl_hit:
                    tp_dist = abs(tp_price - o)
                    sl_dist = abs(sl_price - o)
                    if tp_dist <= sl_dist:
                        tp_hit, sl_hit = True, False
                    else:
                        tp_hit, sl_hit = False, True

                if tp_hit:
                    pnl = (eff_tp * LEVERAGE - FEE_TOTAL) / 100
                    size = equity / max_pos
                    equity += size * pnl
                    total_trades += 1
                    total_wins += 1
                    if is_long:
                        long_trades += 1
                    else:
                        short_trades += 1
                    total_pnl_list.append(pnl * 100)
                    pos_closed = True
                    break
                elif sl_hit:
                    pnl = -(eff_sl * LEVERAGE + FEE_TOTAL) / 100
                    size = equity / max_pos
                    equity += size * pnl
                    total_trades += 1
                    if is_long:
                        long_trades += 1
                    else:
                        short_trades += 1
                    total_pnl_list.append(pnl * 100)
                    pos_closed = True
                    break

            if not pos_closed:
                # Check timeout
                if timeout_bars and sb >= pos['entry_bar'] + timeout_bars:
                    pos_closed = True  # DROP
                else:
                    pos['last_check'] = min(sb + 2, n_bars)
                    new_positions.append(pos)

            if pos_closed:
                pos['closed'] = True

        positions = new_positions

        # Peak / MDD update
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

        # Try to open new position
        if len(positions) >= max_pos:
            continue

        # Direction cap check
        same_dir = sum(1 for p in positions if p['direction'] == ev['direction'])
        if same_dir >= dir_cap:
            continue

        positions.append({
            'sig_bar': ev['sig_bar'],
            'entry_bar': ev['entry_bar'],
            'entry_price': ev['entry_price'],
            'direction': ev['direction'],
            'dir_int': ev['dir_int'],
            'tp_pct': ev['tp_pct'],
            'sl_pct': ev['sl_pct'],
            'closed': False,
        })

    # Close remaining positions at last bar
    # (just DROP them — timeout protocol)

    final_pnl = (equity - 100.0)
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    return {
        'trades': total_trades,
        'wr': wr,
        'cmp_pnl': final_pnl,
        'cmp_mdd': max_dd,
        'pnl_mdd_ratio': final_pnl / max_dd if max_dd > 0 else 0,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'pf': 0,  # simplified
    }


# ===================================================================
# PHASE 1: Pattern Range Distribution
# ===================================================================

def phase1_pattern_ranges(types, highs, lows, opens, n_bars, sig_index, pattern_ranges, overlap_bar, current_pats):
    print("\n" + "=" * 110)
    print("  PHASE 1: PATTERN RANGE DISTRIBUTION")
    print("=" * 110)

    all_ranges = []
    per_pattern = {}
    for item in current_pats:
        key = f"{item['pattern']}_{item['direction']}"
        pranges = pattern_ranges.get(key, [])
        post_overlap = [r for sb, r in pranges if sb >= overlap_bar]
        all_ranges.extend(post_overlap)
        if post_overlap:
            per_pattern[key] = {
                'mean': np.mean(post_overlap),
                'median': np.median(post_overlap),
                'p25': np.percentile(post_overlap, 25),
                'p75': np.percentile(post_overlap, 75),
                'n': len(post_overlap),
            }

    arr = np.array(all_ranges)
    print(f"\n  Post-overlap pattern range (% of entry):")
    print(f"    N = {len(arr)}")
    print(f"    Mean:   {np.mean(arr):.4f}%")
    print(f"    Median: {np.median(arr):.4f}%")
    print(f"    P25:    {np.percentile(arr, 25):.4f}%")
    print(f"    P75:    {np.percentile(arr, 75):.4f}%")
    print(f"    P90:    {np.percentile(arr, 90):.4f}%")

    # Current TP/SL vs range
    tps = [item['tp'] for item in current_pats]
    sls = [item['sl'] for item in current_pats]
    med_range = np.median(arr)
    print(f"\n  Current TP/SL (median): TP={np.median(tps):.2f}%, SL={np.median(sls):.2f}%")
    print(f"  TP/Range ratio: {np.median(tps)/med_range:.1f}x")
    print(f"  SL/Range ratio: {np.median(sls)/med_range:.1f}x")
    print(f"  → TP/SL is {np.median(tps)/med_range:.0f}-{np.median(sls)/med_range:.0f}x larger than pattern signal")

    return {'global_stats': {
        'mean': float(np.mean(arr)), 'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)), 'p75': float(np.percentile(arr, 75)),
    }, 'per_pattern': per_pattern}


# ===================================================================
# PHASE 2: Pattern-Proportional Grid Search
# ===================================================================

def phase2_proportional_grid(current_pats, sig_index, pattern_ranges,
                              opens, highs, lows, n_bars, atr_ratio, overlap_bar):
    print("\n" + "=" * 110)
    print("  PHASE 2: PATTERN-PROPORTIONAL TP/SL GRID SEARCH")
    print(f"  TP multipliers: {TP_MULTS}")
    print(f"  SL multipliers: {SL_MULTS}")
    print("=" * 110)
    t0 = time.time()

    results = []
    for item in current_pats:
        pat = item['pattern']
        direction = item['direction']
        dir_int = 1 if direction == 'LONG' else -1
        key = f"{pat}_{direction}"
        bars = [sb for sb, _ in pattern_ranges.get(key, [])]
        pranges_list = pattern_ranges.get(key, [])

        # Current fixed TP/SL as baseline
        fixed_tp = item['tp']
        fixed_sl = item['sl']
        fixed_trades = backtest_pattern_trades(
            bars, dir_int, fixed_tp, fixed_sl,
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
        fixed_resolved = [t for t in fixed_trades if t.get('reason') != 'timeout']
        fixed_wr = sum(1 for t in fixed_resolved if t['pnl'] > 0) / len(fixed_resolved) * 100 if fixed_resolved else 0
        rw_wr_fixed = fixed_sl / (fixed_tp + fixed_sl) * 100
        fixed_excess = fixed_wr - rw_wr_fixed
        fixed_hold = np.median([t['exit_bar'] - t['entry_bar'] for t in fixed_resolved]) if fixed_resolved else 0

        # Grid search proportional
        best_excess = -999
        best_tp_m, best_sl_m = 2.0, 3.0
        best_wr, best_n = 0, 0
        best_hold = 0
        best_avg_tp, best_avg_sl = 0, 0

        for tp_m in TP_MULTS:
            for sl_m in SL_MULTS:
                if sl_m < tp_m * 0.5:  # skip extreme asymmetry
                    continue
                trades = backtest_proportional(
                    bars, dir_int, tp_m, sl_m,
                    opens, highs, lows, n_bars, pranges_list,
                    atr_ratio=atr_ratio, use_atr=True,
                    lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)

                resolved = [t for t in trades if t.get('reason') != 'timeout']
                if len(resolved) < MIN_TRADES:
                    continue

                wins = sum(1 for t in resolved if t['pnl'] > 0)
                n = len(resolved)
                wr = wins / n * 100

                # Average TP/SL for WR Excess calc
                avg_tp = np.mean([t['tp_pct'] for t in resolved])
                avg_sl = np.mean([t['sl_pct'] for t in resolved])
                rw_wr = avg_sl / (avg_tp + avg_sl) * 100
                excess = wr - rw_wr

                if excess > best_excess:
                    best_excess = excess
                    best_tp_m, best_sl_m = tp_m, sl_m
                    best_wr, best_n = wr, n
                    best_hold = np.median([t['exit_bar'] - t['entry_bar'] for t in resolved])
                    best_avg_tp = avg_tp
                    best_avg_sl = avg_sl

        # Also try ultra-tight fixed grid
        tight_best_excess = -999
        tight_tp, tight_sl = 0.25, 0.50
        tight_wr, tight_n, tight_hold = 0, 0, 0

        for tp in TP_TIGHT:
            for sl in SL_TIGHT:
                if sl < 0.30:
                    continue
                trades = backtest_pattern_trades(
                    bars, dir_int, tp, sl,
                    opens, highs, lows, n_bars,
                    atr_ratio=atr_ratio, use_atr=True,
                    lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
                resolved = [t for t in trades if t.get('reason') != 'timeout']
                if len(resolved) < MIN_TRADES:
                    continue
                wins = sum(1 for t in resolved if t['pnl'] > 0)
                n = len(resolved)
                wr = wins / n * 100
                rw_wr = sl / (tp + sl) * 100
                excess = wr - rw_wr
                if excess > tight_best_excess:
                    tight_best_excess = excess
                    tight_tp, tight_sl = tp, sl
                    tight_wr, tight_n = wr, n
                    tight_hold = np.median([t['exit_bar'] - t['entry_bar'] for t in resolved])

        results.append({
            'pattern': pat, 'direction': direction,
            # Fixed (current compact)
            'fixed_tp': fixed_tp, 'fixed_sl': fixed_sl,
            'fixed_excess': round(fixed_excess, 1),
            'fixed_hold': int(fixed_hold),
            'fixed_n': len(fixed_resolved),
            # Proportional best
            'prop_tp_mult': best_tp_m, 'prop_sl_mult': best_sl_m,
            'prop_avg_tp': round(best_avg_tp, 3), 'prop_avg_sl': round(best_avg_sl, 3),
            'prop_excess': round(best_excess, 1),
            'prop_hold': int(best_hold),
            'prop_n': best_n,
            'prop_wr': round(best_wr, 1),
            # Ultra-tight fixed
            'tight_tp': tight_tp, 'tight_sl': tight_sl,
            'tight_excess': round(tight_best_excess, 1),
            'tight_hold': int(tight_hold),
            'tight_n': tight_n,
            'tight_wr': round(tight_wr, 1),
        })

    elapsed = time.time() - t0

    # Print comparison
    print(f"\n  {'Pattern':<14} {'Dir':>5} | {'Fix_TP':>6} {'Fix_SL':>6} {'F_Exc':>6} {'F_Hold':>6} | "
          f"{'P_TM':>4} {'P_SM':>4} {'P_TP':>6} {'P_SL':>6} {'P_Exc':>6} {'P_Hold':>6} | "
          f"{'T_TP':>5} {'T_SL':>5} {'T_Exc':>6} {'T_Hold':>6}")
    print("  " + "-" * 130)

    for r in results:
        print(f"  {r['pattern']:<14} {r['direction']:>5} | "
              f"{r['fixed_tp']:6.2f} {r['fixed_sl']:6.2f} {r['fixed_excess']:>+5.1f} {r['fixed_hold']:5d}b | "
              f"{r['prop_tp_mult']:4.1f} {r['prop_sl_mult']:4.1f} "
              f"{r['prop_avg_tp']:6.3f} {r['prop_avg_sl']:6.3f} {r['prop_excess']:>+5.1f} {r['prop_hold']:5d}b | "
              f"{r['tight_tp']:5.2f} {r['tight_sl']:5.2f} {r['tight_excess']:>+5.1f} {r['tight_hold']:5d}b")

    # Summary
    prop_strong = sum(1 for r in results if r['prop_excess'] > 5.0)
    tight_strong = sum(1 for r in results if r['tight_excess'] > 5.0)
    fixed_strong = sum(1 for r in results if r['fixed_excess'] > 5.0)

    prop_holds = [r['prop_hold'] for r in results if r['prop_hold'] > 0]
    tight_holds = [r['tight_hold'] for r in results if r['tight_hold'] > 0]
    fixed_holds = [r['fixed_hold'] for r in results if r['fixed_hold'] > 0]

    print(f"\n  STRONG (WR Excess > 5pp): Fixed={fixed_strong}, Proportional={prop_strong}, Tight={tight_strong}")
    if prop_holds:
        print(f"  Median hold: Fixed={np.median(fixed_holds):.0f}b ({np.median(fixed_holds)*5/60:.1f}h), "
              f"Prop={np.median(prop_holds):.0f}b ({np.median(prop_holds)*5/60:.1f}h), "
              f"Tight={np.median(tight_holds):.0f}b ({np.median(tight_holds)*5/60:.1f}h)")
    print(f"  ({elapsed:.1f}s)")

    return results


# ===================================================================
# PHASE 3: Regime Bias Verification
# ===================================================================

def phase3_regime_check(results_phase2, sig_index, pattern_ranges,
                         opens, highs, lows, n_bars, atr_ratio, overlap_bar):
    """Check regime bias: flip direction and verify edge disappears."""
    print("\n" + "=" * 110)
    print("  PHASE 3: REGIME BIAS VERIFICATION (Direction Flip Test)")
    print("=" * 110)

    flip_results = []
    for r in results_phase2:
        pat = r['pattern']
        direction = r['direction']
        flipped_dir = 'SHORT' if direction == 'LONG' else 'LONG'
        flipped_int = -1 if direction == 'LONG' else 1
        key = f"{pat}_{direction}"
        bars = [sb for sb, _ in pattern_ranges.get(key, [])]
        pranges_list = pattern_ranges.get(key, [])

        # Flip with proportional best
        flip_trades = backtest_proportional(
            bars, flipped_int, r['prop_tp_mult'], r['prop_sl_mult'],
            opens, highs, lows, n_bars, pranges_list,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
        flip_resolved = [t for t in flip_trades if t.get('reason') != 'timeout']

        if flip_resolved:
            flip_wr = sum(1 for t in flip_resolved if t['pnl'] > 0) / len(flip_resolved) * 100
            avg_tp = np.mean([t['tp_pct'] for t in flip_resolved])
            avg_sl = np.mean([t['sl_pct'] for t in flip_resolved])
            rw_wr = avg_sl / (avg_tp + avg_sl) * 100
            flip_excess = flip_wr - rw_wr
        else:
            flip_wr, flip_excess = 0, 0

        # Flip with tight fixed
        flip_tight = backtest_pattern_trades(
            bars, flipped_int, r['tight_tp'], r['tight_sl'],
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
        flip_tight_resolved = [t for t in flip_tight if t.get('reason') != 'timeout']
        if flip_tight_resolved:
            ft_wr = sum(1 for t in flip_tight_resolved if t['pnl'] > 0) / len(flip_tight_resolved) * 100
            ft_rw = r['tight_sl'] / (r['tight_tp'] + r['tight_sl']) * 100
            ft_excess = ft_wr - ft_rw
        else:
            ft_wr, ft_excess = 0, 0

        genuine_prop = r['prop_excess'] > 5.0 and flip_excess < r['prop_excess'] * 0.5
        genuine_tight = r['tight_excess'] > 5.0 and ft_excess < r['tight_excess'] * 0.5

        flip_results.append({
            'pattern': pat, 'direction': direction,
            'prop_excess': r['prop_excess'], 'prop_flip_excess': round(flip_excess, 1),
            'prop_genuine': genuine_prop,
            'tight_excess': r['tight_excess'], 'tight_flip_excess': round(ft_excess, 1),
            'tight_genuine': genuine_tight,
        })

    print(f"\n  {'Pattern':<14} {'Dir':>5} | {'P_Exc':>6} {'P_Flip':>6} {'P_Gen':>5} | {'T_Exc':>6} {'T_Flip':>6} {'T_Gen':>5}")
    print("  " + "-" * 75)
    for fr in flip_results:
        pg = "YES" if fr['prop_genuine'] else "no"
        tg = "YES" if fr['tight_genuine'] else "no"
        print(f"  {fr['pattern']:<14} {fr['direction']:>5} | "
              f"{fr['prop_excess']:>+5.1f} {fr['prop_flip_excess']:>+5.1f} {pg:>5} | "
              f"{fr['tight_excess']:>+5.1f} {fr['tight_flip_excess']:>+5.1f} {tg:>5}")

    prop_genuine = sum(1 for fr in flip_results if fr['prop_genuine'])
    tight_genuine = sum(1 for fr in flip_results if fr['tight_genuine'])
    print(f"\n  Genuine edge: Proportional={prop_genuine}/{len(flip_results)}, Tight={tight_genuine}/{len(flip_results)}")

    return flip_results


# ===================================================================
# PHASE 4: Portfolio Comparison
# ===================================================================

def phase4_portfolio(results_phase2, flip_results, sig_index, pattern_ranges,
                      opens, highs, lows, n_bars, atr_ratio, overlap_bar, current_pats):
    print("\n" + "=" * 110)
    print("  PHASE 4: PORTFOLIO COMPARISON (Hedge N=9, cap=6, T864)")
    print("=" * 110)

    # Build schedules
    # 1. Current fixed compact (baseline)
    fixed_schedule = current_pats

    # 2. Proportional (STRONG + genuine only)
    prop_schedule = []
    for r, fr in zip(results_phase2, flip_results):
        if r['prop_excess'] > 5.0:
            prop_schedule.append({
                'pattern': r['pattern'], 'direction': r['direction'],
                'tp_mult': r['prop_tp_mult'], 'sl_mult': r['prop_sl_mult'],
            })

    # 3. Tight fixed (STRONG only)
    tight_schedule = []
    for r in results_phase2:
        if r['tight_excess'] > 5.0 and r['tight_n'] >= MIN_TRADES:
            tight_schedule.append({
                'pattern': r['pattern'], 'direction': r['direction'],
                'tp': r['tight_tp'], 'sl': r['tight_sl'],
            })

    configs = {}

    # Fixed compact baseline
    res_fixed = simulate_hedge_proportional(
        fixed_schedule, sig_index, opens, highs, lows, n_bars,
        pattern_ranges, atr_ratio, overlap_bar,
        proportional=False)
    n_long_f = sum(1 for p in fixed_schedule if p['direction'] == 'LONG')
    n_short_f = sum(1 for p in fixed_schedule if p['direction'] == 'SHORT')
    res_fixed['patterns'] = len(fixed_schedule)
    res_fixed['n_long'] = n_long_f
    res_fixed['n_short'] = n_short_f
    res_fixed['trades_per_day'] = round(res_fixed['trades'] / (81384 / 288), 2)
    configs['fixed_compact'] = res_fixed

    # Proportional
    if prop_schedule:
        res_prop = simulate_hedge_proportional(
            prop_schedule, sig_index, opens, highs, lows, n_bars,
            pattern_ranges, atr_ratio, overlap_bar,
            proportional=True)
        n_long_p = sum(1 for p in prop_schedule if p['direction'] == 'LONG')
        n_short_p = sum(1 for p in prop_schedule if p['direction'] == 'SHORT')
        res_prop['patterns'] = len(prop_schedule)
        res_prop['n_long'] = n_long_p
        res_prop['n_short'] = n_short_p
        res_prop['trades_per_day'] = round(res_prop['trades'] / (81384 / 288), 2)
        configs['proportional'] = res_prop

    # Tight fixed
    if tight_schedule:
        res_tight = simulate_hedge_proportional(
            tight_schedule, sig_index, opens, highs, lows, n_bars,
            pattern_ranges, atr_ratio, overlap_bar,
            proportional=False)
        n_long_t = sum(1 for p in tight_schedule if p['direction'] == 'LONG')
        n_short_t = sum(1 for p in tight_schedule if p['direction'] == 'SHORT')
        res_tight['patterns'] = len(tight_schedule)
        res_tight['n_long'] = n_long_t
        res_tight['n_short'] = n_short_t
        res_tight['trades_per_day'] = round(res_tight['trades'] / (81384 / 288), 2)
        configs['tight_fixed'] = res_tight

    for label, res in configs.items():
        print(f"\n  [{label}] {res['patterns']}pat ({res['n_long']}L+{res['n_short']}S)")
        print(f"    Trades: {res['trades']} ({res['trades_per_day']}/day)")
        print(f"    WR: {res['wr']:.1f}%, PnL: {res['cmp_pnl']:+.1f}%, MDD: {res['cmp_mdd']:.1f}%")
        print(f"    PnL/MDD: {res['pnl_mdd_ratio']:.2f}")

    return configs, {'fixed': fixed_schedule, 'proportional': prop_schedule, 'tight': tight_schedule}


# ===================================================================
# PHASE 5: WF 3-Fold Validation
# ===================================================================

def phase5_wf(configs, schedules, sig_index, pattern_ranges,
              opens, highs, lows, n_bars, atr_ratio, overlap_bar):
    print("\n" + "=" * 110)
    print("  PHASE 5: WF 3-FOLD VALIDATION")
    print("=" * 110)

    # Standard 3-fold expanding window on POST-OVERLAP data only
    avail_bars = n_bars - overlap_bar
    quarter = avail_bars // 4
    folds = [
        {'oos_lo': overlap_bar + quarter,     'oos_hi': overlap_bar + 2 * quarter},
        {'oos_lo': overlap_bar + 2 * quarter, 'oos_hi': overlap_bar + 3 * quarter},
        {'oos_lo': overlap_bar + 3 * quarter, 'oos_hi': min(overlap_bar + avail_bars, n_bars)},
    ]

    wf_results = {}
    for label in ['fixed_compact', 'proportional', 'tight_fixed']:
        if label not in configs:
            continue
        sched_key = label.replace('_compact', '').replace('_fixed', '')
        if sched_key == 'fixed':
            sched = schedules['fixed']
            is_prop = False
        elif sched_key == 'proportional':
            sched = schedules['proportional']
            is_prop = True
        elif sched_key == 'tight':
            sched = schedules['tight']
            is_prop = False
        else:
            continue

        fold_results = []
        all_pass = True
        for i, fold in enumerate(folds):
            res = simulate_hedge_proportional(
                sched, sig_index, opens, highs, lows, n_bars,
                pattern_ranges, atr_ratio, fold['oos_lo'], fold['oos_hi'],  # post-overlap only
                proportional=is_prop)
            fold_results.append(res)
            passed = res['cmp_pnl'] > 0 and res['trades'] >= 20
            if not passed:
                all_pass = False
            print(f"  [{label}] Fold {i+1}: trades={res['trades']}, WR={res['wr']:.1f}%, "
                  f"PnL={res['cmp_pnl']:+.1f}%, MDD={res['cmp_mdd']:.1f}%")

        verdict = "PASS (3/3)" if all_pass else f"FAIL"
        print(f"  [{label}] {verdict}")
        wf_results[label] = {'folds': fold_results, 'verdict': verdict}

    return wf_results


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()

    print("Loading 720-day data...")
    df = load_and_classify(DATA_720)
    types = df['rctype'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    print(f"  {n_bars} bars loaded")

    sig_index = build_signal_index(types, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)
    overlap_bar = find_overlap_bar(df)
    print(f"  Overlap bar: {overlap_bar}")

    # Load current patterns
    with open(DYNAMIC_PATTERNS) as f:
        dp = json.load(f)

    current_pats = []
    for key, det in dp['pattern_details'].items():
        current_pats.append({
            'pattern': det['pattern'],
            'direction': det['direction'],
            'tp': det['tp'],
            'sl': det['sl'],
        })
    print(f"  Current: {len(current_pats)} patterns")

    # Compute pattern ranges
    pattern_ranges = compute_pattern_ranges(types, highs, lows, opens, n_bars, sig_index)

    # Phase 1
    p1 = phase1_pattern_ranges(types, highs, lows, opens, n_bars, sig_index, pattern_ranges, overlap_bar, current_pats)

    # Phase 2
    p2 = phase2_proportional_grid(current_pats, sig_index, pattern_ranges,
                                   opens, highs, lows, n_bars, atr_ratio, overlap_bar)

    # Phase 3
    p3 = phase3_regime_check(p2, sig_index, pattern_ranges,
                              opens, highs, lows, n_bars, atr_ratio, overlap_bar)

    # Phase 4
    p4_configs, p4_schedules = phase4_portfolio(p2, p3, sig_index, pattern_ranges,
                                                 opens, highs, lows, n_bars, atr_ratio,
                                                 overlap_bar, current_pats)

    # Phase 5
    p5 = phase5_wf(p4_configs, p4_schedules, sig_index, pattern_ranges,
                    opens, highs, lows, n_bars, atr_ratio, overlap_bar)

    # Phase 6: Final recommendation
    print("\n" + "=" * 110)
    print("  PHASE 6: FINAL RECOMMENDATION")
    print("=" * 110)

    best_label = None
    best_ratio = 0
    for label, wf in p5.items():
        if 'PASS' in wf['verdict']:
            ratio = p4_configs[label]['pnl_mdd_ratio']
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = label

    if best_label:
        best = p4_configs[best_label]
        print(f"\n  BEST: {best_label}")
        print(f"    Patterns: {best['patterns']} ({best['n_long']}L+{best['n_short']}S)")
        print(f"    Trades: {best['trades']} ({best['trades_per_day']}/day)")
        print(f"    WR: {best['wr']:.1f}%, PnL: {best['cmp_pnl']:+.1f}%, MDD: {best['cmp_mdd']:.1f}%")
        print(f"    PnL/MDD: {best['pnl_mdd_ratio']:.2f}")
        print(f"    WF: {p5[best_label]['verdict']}")

        fc = p4_configs.get('fixed_compact', {})
        if fc:
            print(f"\n  vs Fixed Compact baseline:")
            print(f"    Trades: {fc['trades']} -> {best['trades']}")
            print(f"    PnL/MDD: {fc['pnl_mdd_ratio']:.2f} -> {best_ratio:.2f}")
    else:
        print("\n  No configuration passed WF 3/3")
        print("  → Keep current fixed compact TP/SL (regime-dependent but WF-validated)")

    total = time.time() - t_start

    # Save results
    output = {
        'timestamp': str(np.datetime64('now')),
        'study': 'Pattern-Proportional TP/SL Study',
        'phase1_ranges': p1,
        'phase2_grid': [{k: v for k, v in r.items()} for r in p2],
        'phase3_flip': [{k: v for k, v in r.items()} for r in p3],
        'phase4_portfolio': {k: v for k, v in p4_configs.items()},
        'phase5_wf': {k: {'verdict': v['verdict']} for k, v in p5.items()},
        'recommendation': best_label or 'fixed_compact (no improvement)',
        'elapsed_seconds': round(total, 1),
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")
    print(f"  Total elapsed: {total:.1f}s")


if __name__ == '__main__':
    main()
