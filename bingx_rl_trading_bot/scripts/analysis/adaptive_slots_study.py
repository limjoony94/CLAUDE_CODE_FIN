#!/usr/bin/env python3
"""
Adaptive Slot Count Study — Reduce active slots during high-volatility periods.

Concept:
  - When ATR ratio exceeds a threshold, reduce max_positions from 9 to a lower value
  - When volatility normalizes, restore to 9
  - This limits the NUMBER of positions that can cluster-SL simultaneously
  - Don't force-close existing positions — just block new entries
  - Position sizing adjusts: equity / effective_n_slots (larger per-position when fewer slots)

Sweep:
  vol_threshold: [1.0, 1.2, 1.4, 1.6]  (ATR ratio above which we reduce)
  reduced_slots: [3, 4, 5, 6, 7]
  => 20 combos + 1 baseline = 21 tests

Standard Research Protocol: compound, 0.10% fee, 0.02% slippage, 3x leverage.
WF: 3-fold expanding window. MC: 3 seeds (42, 123, 7).
"""

import json
import numpy as np
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    _check_exit_npos,
    LEVERAGE, FEE_PCT, MAX_BARS, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/adaptive_slots_study.json"
TP_SCALE_FACTOR = 0.72  # v1.61.0

# Sweep grid
VOL_THRESHOLDS = [1.0, 1.2, 1.4, 1.6]
REDUCED_SLOTS = [3, 4, 5, 6, 7]

# MC seeds
MC_SEEDS = [42, 123, 7]


def load_patterns(filepath=PATTERNS_FILE):
    with open(filepath) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(max(0.3, v['tp'] * TP_SCALE_FACTOR), 3),
            'sl': v['sl'],
        }
    return result


def build_signal_tuples(patterns, sig_idx):
    tuples = []
    for k, v in patterns.items():
        pat_name = v.get('pattern') or k.rsplit('_', 1)[0]
        if pat_name in sig_idx:
            for bar in sig_idx[pat_name]:
                tuples.append((bar, k, v['direction'], v['tp'], v['sl']))
    return tuples


def portfolio_npos_adaptive(signal_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, start_bar, end_bar,
                            n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
                            regime_mult=DEFAULT_REGIME_MULT,
                            agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
                            agg_risk_with=DEFAULT_AGG_RISK_WITH,
                            momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
                            momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
                            momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
                            clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
                            timeout_bars=TIMEOUT_BARS,
                            cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
                            vol_threshold=None, reduced_slots=None):
    """N-position portfolio simulator with adaptive slot count.

    When vol_threshold is not None and atr_ratio[bar] > vol_threshold,
    effective_n_slots = reduced_slots. Otherwise effective_n_slots = n_slots.
    Existing positions are never force-closed; only new entries are blocked.
    Position sizing uses effective_n_slots: size_pct = 100 / effective_n_slots.
    Direction cap scales: min(original_dir_cap, effective_n_slots - 2).
    """
    adaptive = (vol_threshold is not None and reduced_slots is not None)
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd_mtm = 0.0

    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0,
                     'dup_pat': 0, 'max_pos': 0, 'vol_reduce': 0}
    corr_events = []

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Adaptive slot tracking
    bars_in_reduced = 0
    total_bars_counted = 0
    total_slot_utilization = 0.0

    # Filter and sort signals in range
    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # Determine effective slots for this bar
        if adaptive and bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
            if atr_ratio[bar] > vol_threshold:
                eff_n_slots = reduced_slots
                bars_in_reduced += 1
            else:
                eff_n_slots = n_slots
        else:
            eff_n_slots = n_slots

        eff_dir_cap = min(direction_cap, max(1, eff_n_slots - 2))
        eff_size_pct = 100.0 / eff_n_slots

        total_bars_counted += 1
        total_slot_utilization += len(positions) / eff_n_slots if eff_n_slots > 0 else 0

        # 1. Check exits (use the size_pct at entry time, stored in position)
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                      atr_ratio, fee, clamp_lo, clamp_hi,
                                      timeout_bars)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                entry_size_pct = pos.get('entry_size_pct', 100.0 / n_slots)
                pnl_portfolio = result['pnl_slot'] * (entry_size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1

        # Cascade SL tightening
        if cascade_tighten_pct > 0 and bar_sl_count > 0:
            keep_ratio = 1.0 - cascade_tighten_pct / 100.0
            sl_directions = set()
            for t in trades[len(trades) - len(closed_slots):]:
                if t.get('reason') == 'SL':
                    sl_directions.add(t['direction'])
            for sl_dir in sl_directions:
                for pos in positions:
                    if pos['slot'] in closed_slots:
                        continue
                    if pos['direction'] != sl_dir:
                        continue
                    sig = pos['signal_bar']
                    if (atr_ratio is not None and sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[sig])):
                        r = max(clamp_lo, min(clamp_hi, atr_ratio[sig]))
                    else:
                        r = 1.0
                    p_sl = pos['sl_pct']
                    if p_sl > 0:
                        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                    cur_eff_sl = pos.get('eff_sl_override') or (p_sl * r)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]

        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum)
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct
            corr_events.append((bar, bar_sl_count, loss_pct))

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

        # 2. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= eff_n_slots:
                total_blocked['max_pos'] += 1
                if adaptive and eff_n_slots < n_slots:
                    total_blocked['vol_reduce'] += 1
                continue

            # Direction cap (using effective cap)
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= eff_dir_cap:
                total_blocked['dir_cap'] += 1
                continue

            # Duplicate pattern check
            if any(p['pattern'] == pat for p in positions):
                total_blocked['dup_pat'] += 1
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            # Momentum guard check
            if momentum_lookback > 0 and bar < momentum_pause_until.get(direction, -1):
                total_blocked['momentum'] += 1
                continue

            # Regime sizing
            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            # Aggregate risk cap check (use eff_n_slots for sizing)
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
                        if p_sl > 0:
                            p_r = min(p_r, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        # Use position's entry-time sizing for existing exposure
                        p_entry_size = p.get('entry_size_pct', 100.0 / n_slots) / 100.0
                        existing_exposure += p_eff_sl * p_entry_size * LEVERAGE * p_sm

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
                if sl_pct > 0:
                    new_r = min(new_r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)
                new_eff_sl = sl_pct * new_r
                new_exposure = new_eff_sl * (1.0 / eff_n_slots) * LEVERAGE * sm

                if existing_exposure + new_exposure > cap_pct:
                    total_blocked['agg_risk'] += 1
                    continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
                'entry_size_pct': eff_size_pct,  # lock sizing at entry time
            })

        if len(positions) > max_sim_positions:
            max_sim_positions = len(positions)

        # Mark-to-market MDD
        if positions and bar < n_bars:
            mtm_equity = equity
            for pos in positions:
                eb = pos['entry_bar']
                if eb >= n_bars or bar < eb:
                    continue
                entry_price = opens[eb]
                if entry_price <= 0:
                    continue
                if pos['direction'] == 'LONG':
                    unr = (closes[bar] / entry_price - 1) * 100 * LEVERAGE
                else:
                    unr = (1 - closes[bar] / entry_price) * 100 * LEVERAGE
                sm = pos.get('size_mult', 1.0)
                entry_size_pct = pos.get('entry_size_pct', 100.0 / n_slots)
                mtm_equity += unr * (entry_size_pct / 100) * sm
            if mtm_equity > peak_equity:
                peak_equity = mtm_equity
            dd = (peak_equity - mtm_equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd
        elif not positions:
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd

    # Force-close remaining
    for pos in positions:
        entry_bar_p = pos['entry_bar']
        if entry_bar_p >= n_bars:
            continue
        entry = opens[entry_bar_p]
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
        entry_size_pct = pos.get('entry_size_pct', 100.0 / n_slots)
        trades.append({
            'entry_bar': entry_bar_p, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (entry_size_pct / 100) * sm,
        })

    pct_reduced = round(bars_in_reduced / total_bars_counted * 100, 1) if total_bars_counted > 0 else 0
    avg_util = round(total_slot_utilization / total_bars_counted, 3) if total_bars_counted > 0 else 0

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'corr_events': len(corr_events),
        'blocked': total_blocked,
        'mdd_mtm': round(max_dd_mtm, 2),
        'pct_bars_reduced': pct_reduced,
        'avg_slot_utilization': avg_util,
    }
    return trades, stats


def run_npos_adaptive(signal_tuples, opens, highs, lows, closes, n_bars,
                      atr_ratio, ema_slope, start_bar, end_bar,
                      vol_threshold=None, reduced_slots=None, **extra):
    kwargs = dict(
        n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
        regime_mult=DEFAULT_REGIME_MULT,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
        timeout_bars=TIMEOUT_BARS, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
    )
    kwargs.update(extra)
    trades, raw = portfolio_npos_adaptive(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        vol_threshold=vol_threshold, reduced_slots=reduced_slots, **kwargs
    )
    stats = calc_stats_compound(trades)
    if raw.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw.items() if k not in stats})
    return trades, stats


def run_mc(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, ns, ne,
           vol_threshold=None, reduced_slots=None, seeds=MC_SEEDS):
    """MC sign randomization test: shuffle trade signs, compare PnL."""
    # Real run
    real_trades, real_stats = run_npos_adaptive(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        vol_threshold=vol_threshold, reduced_slots=reduced_slots)
    real_pnl = real_stats.get('pnl', 0)

    max_p = 0.0
    for seed in seeds:
        rng = np.random.RandomState(seed)
        n_sims = 5000
        count_ge = 0
        for _ in range(n_sims):
            shuffled_pnl = 0.0
            for t in real_trades:
                sign = 1 if rng.random() < 0.5 else -1
                shuffled_pnl += t['pnl_portfolio'] * sign
            if shuffled_pnl >= real_pnl:
                count_ge += 1
        p = count_ge / n_sims
        if p > max_p:
            max_p = p

    return max_p


def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, ns, ne, n_folds=3,
           vol_threshold=None, reduced_slots=None):
    total = ne - ns
    min_train = total // 3
    fold_size = total // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        train_end = ns + min_train + fold_size * fold
        test_start = train_end
        test_end = min(train_end + fold_size, ne)
        if test_start >= ne or test_end <= test_start:
            continue
        _, stats = run_npos_adaptive(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, test_start, test_end,
            vol_threshold=vol_threshold, reduced_slots=reduced_slots)
        results.append({'fold': fold + 1, 'pnl': stats.get('pnl', 0),
                        'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0),
                        'mdd': stats.get('mdd', 0)})
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

print("=" * 70)
print("ADAPTIVE SLOT COUNT STUDY")
print("Reduce active slots during high-volatility periods")
print("=" * 70)

df = load_and_classify(DATA_FILE)
opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
n_bars = len(df)
type_codes = df['candle_type'].values

atr_ratio = compute_atr_ratio(highs, lows, closes)
ema_slope = compute_ema_slope(closes)

signal_index = build_signal_index(type_codes, n_bars)
nw = find_neutral_window(closes)
ns, ne = nw
print(f"Data: {n_bars} bars, neutral: {ns}-{ne} ({(ne-ns)//288:.0f}d)")

patterns = load_patterns()
print(f"Patterns: {len(patterns)} (TP x{TP_SCALE_FACTOR})")

signal_tuples = build_signal_tuples(patterns, signal_index)
print(f"Signal tuples: {len(signal_tuples)}")

# ATR ratio distribution for context
atr_valid = atr_ratio[ns:ne]
atr_valid = atr_valid[~np.isnan(atr_valid)]
print(f"\nATR ratio in neutral window:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {np.percentile(atr_valid, p):.3f}")

all_results = {
    "study": "adaptive_slots_study",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "config": {
        "n_slots_base": DEFAULT_N_SLOTS,
        "direction_cap_base": DEFAULT_DIRECTION_CAP,
        "tp_scale_factor": TP_SCALE_FACTOR,
        "vol_thresholds": VOL_THRESHOLDS,
        "reduced_slots": REDUCED_SLOTS,
        "neutral_window": [int(ns), int(ne)],
        "n_bars": n_bars,
    },
    "atr_distribution": {
        f"p{p}": round(float(np.percentile(atr_valid, p)), 3)
        for p in [25, 50, 75, 90, 95, 99]
    },
}


# ═══════════════════════════════════════════════════════════
# PHASE 1: Baseline (no adaptive reduction)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: BASELINE (always 9 slots)")
print("=" * 70)

_, baseline_stats = run_npos_adaptive(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne)

print(f"  Trades: {baseline_stats['trades']}")
print(f"  WR: {baseline_stats['wr']:.1f}%")
print(f"  PnL: {baseline_stats['pnl']:+.1f}%")
print(f"  MDD(MTM): {baseline_stats['mdd']:.2f}%")
print(f"  PnL/MDD: {baseline_stats.get('pnl_mdd', 0):.1f}x")
print(f"  Max Corr Loss: {baseline_stats['max_corr_loss']:.2f}%")
print(f"  Max Sim Positions: {baseline_stats['max_sim_positions']}")
print(f"  Blocked: {baseline_stats.get('blocked', {})}")

all_results['baseline'] = baseline_stats


# ═══════════════════════════════════════════════════════════
# PHASE 2: 2D Grid Sweep
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: 2D GRID SWEEP (vol_threshold x reduced_slots)")
print("=" * 70)

grid_results = []
header = f"{'VT':>5} {'RS':>3} | {'Trades':>6} {'WR':>6} {'PnL':>8} {'MDD':>6} {'P/M':>7} {'CorrL':>6} {'%Red':>5} {'AvgUt':>5} | {'vs BL':>7}"
print(header)
print("-" * len(header))

for vt in VOL_THRESHOLDS:
    for rs in REDUCED_SLOTS:
        _, stats = run_npos_adaptive(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, ns, ne,
            vol_threshold=vt, reduced_slots=rs)

        pnl_delta = stats['pnl'] - baseline_stats['pnl']
        pm = stats.get('pnl_mdd', 0)

        print(f"{vt:5.1f} {rs:3d} | {stats['trades']:6d} {stats['wr']:5.1f}% "
              f"{stats['pnl']:+7.1f}% {stats['mdd']:5.2f}% {pm:6.1f}x "
              f"{stats['max_corr_loss']:5.2f}% {stats.get('pct_bars_reduced', 0):4.1f}% "
              f"{stats.get('avg_slot_utilization', 0):5.3f} | {pnl_delta:+6.1f}%")

        grid_results.append({
            'vol_threshold': vt,
            'reduced_slots': rs,
            'trades': stats['trades'],
            'wr': stats['wr'],
            'pnl': stats['pnl'],
            'mdd': stats['mdd'],
            'pnl_mdd': pm,
            'max_corr_loss': stats['max_corr_loss'],
            'max_sim_positions': stats['max_sim_positions'],
            'pct_bars_reduced': stats.get('pct_bars_reduced', 0),
            'avg_slot_utilization': stats.get('avg_slot_utilization', 0),
            'blocked': stats.get('blocked', {}),
            'pnl_delta': round(pnl_delta, 2),
            'mdd_delta': round(stats['mdd'] - baseline_stats['mdd'], 2),
            'corr_loss_delta': round(stats['max_corr_loss'] - baseline_stats['max_corr_loss'], 2),
        })

all_results['grid_sweep'] = grid_results


# ═══════════════════════════════════════════════════════════
# PHASE 3: Top candidates — WF + MC validation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: TOP CANDIDATES — WF + MC VALIDATION")
print("=" * 70)

# Select top 5 by PnL/MDD (must improve over baseline in at least one dimension)
baseline_pm = baseline_stats.get('pnl_mdd', 0)
baseline_corr = baseline_stats['max_corr_loss']

# Rank by PnL/MDD
ranked = sorted(grid_results, key=lambda x: x['pnl_mdd'], reverse=True)

# Take top 5 candidates (include baseline-beaters and best PnL/MDD)
top_candidates = ranked[:5]

print(f"\nTop 5 candidates (by PnL/MDD):")
print(f"  Baseline PnL/MDD: {baseline_pm:.1f}x, Max Corr Loss: {baseline_corr:.2f}%")
print()

# Baseline WF
print("Baseline WF:")
wf_folds, wf_oos, wf_pass = run_wf(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne)
wf_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_folds)
mc_p = run_mc(signal_tuples, opens, highs, lows, closes, n_bars,
              atr_ratio, ema_slope, ns, ne)
print(f"  WF: {wf_str} = {wf_oos:+.1f}% {'PASS' if wf_pass else 'FAIL'}")
print(f"  MC max-p: {mc_p:.4f} {'<0.01' if mc_p < 0.01 else '>=0.01 FAIL'}")

all_results['baseline_wf'] = {
    'folds': wf_folds, 'oos_pnl': round(wf_oos, 1),
    'all_pass': wf_pass, 'mc_max_p': round(mc_p, 4),
}

validated = []
for i, cand in enumerate(top_candidates):
    vt = cand['vol_threshold']
    rs = cand['reduced_slots']
    print(f"\nCandidate {i+1}: VT={vt}, RS={rs}")
    print(f"  IS: PnL={cand['pnl']:+.1f}%, MDD={cand['mdd']:.2f}%, "
          f"PnL/MDD={cand['pnl_mdd']:.1f}x, CorrL={cand['max_corr_loss']:.2f}%, "
          f"%Red={cand['pct_bars_reduced']:.1f}%")

    wf_folds_c, wf_oos_c, wf_pass_c = run_wf(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        vol_threshold=vt, reduced_slots=rs)
    wf_str_c = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_folds_c)

    mc_p_c = run_mc(signal_tuples, opens, highs, lows, closes, n_bars,
                    atr_ratio, ema_slope, ns, ne,
                    vol_threshold=vt, reduced_slots=rs)

    print(f"  WF: {wf_str_c} = {wf_oos_c:+.1f}% {'PASS' if wf_pass_c else 'FAIL'}")
    print(f"  MC max-p: {mc_p_c:.4f} {'<0.01' if mc_p_c < 0.01 else '>=0.01 FAIL'}")

    cand_result = {
        'vol_threshold': vt,
        'reduced_slots': rs,
        'is': {
            'pnl': cand['pnl'], 'mdd': cand['mdd'], 'pnl_mdd': cand['pnl_mdd'],
            'wr': cand['wr'], 'trades': cand['trades'],
            'max_corr_loss': cand['max_corr_loss'],
            'pct_bars_reduced': cand['pct_bars_reduced'],
        },
        'wf': {
            'folds': wf_folds_c, 'oos_pnl': round(wf_oos_c, 1),
            'all_pass': wf_pass_c,
        },
        'mc_max_p': round(mc_p_c, 4),
    }
    validated.append(cand_result)

all_results['validated_candidates'] = validated


# ═══════════════════════════════════════════════════════════
# PHASE 4: Discrimination Test
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: DISCRIMINATION TEST")
print("Compare best candidate vs baseline WF OOS gap")
print("=" * 70)

# Best candidate = highest WF OOS PnL among PASS candidates
pass_candidates = [c for c in validated if c['wf']['all_pass']]
if pass_candidates:
    best = max(pass_candidates, key=lambda x: x['wf']['oos_pnl'])
    gap = best['wf']['oos_pnl'] - wf_oos
    print(f"\nBest PASS candidate: VT={best['vol_threshold']}, RS={best['reduced_slots']}")
    print(f"  Baseline WF OOS: {wf_oos:+.1f}%")
    print(f"  Candidate WF OOS: {best['wf']['oos_pnl']:+.1f}%")
    print(f"  Gap: {gap:+.1f}%")
    print(f"  IS PnL/MDD: {best['is']['pnl_mdd']:.1f}x vs baseline {baseline_pm:.1f}x")
    print(f"  IS Max Corr Loss: {best['is']['max_corr_loss']:.2f}% vs baseline {baseline_corr:.2f}%")
    print(f"  IS MDD: {best['is']['mdd']:.2f}% vs baseline {baseline_stats['mdd']:.2f}%")

    # DISCRIMINATING requires: OOS gap > +10%, IS PnL/MDD >= baseline, MC < 0.01
    # A marginal OOS gain with worse IS metrics = NON-DISCRIMINATING (noise)
    is_pm_ok = best['is']['pnl_mdd'] >= baseline_pm
    disc = gap > 10.0 and is_pm_ok and best['mc_max_p'] < 0.01
    disc_str = "DISCRIMINATING" if disc else "NON-DISCRIMINATING"
    print(f"\n  IS PnL/MDD >= baseline? {is_pm_ok} ({best['is']['pnl_mdd']:.1f}x vs {baseline_pm:.1f}x)")
    print(f"  OOS gap > +10%? {gap > 10.0} ({gap:+.1f}%)")
    print(f"  Verdict: {disc_str}")

    all_results['best_candidate'] = {
        'vol_threshold': best['vol_threshold'],
        'reduced_slots': best['reduced_slots'],
        'wf_oos_gap': round(gap, 1),
        'discriminating': disc,
    }
else:
    print("\nNo candidate passed WF 3/3.")
    # Still find best by PnL/MDD
    if validated:
        best = max(validated, key=lambda x: x['is']['pnl_mdd'])
        print(f"  Best IS PnL/MDD: VT={best['vol_threshold']}, RS={best['reduced_slots']}: "
              f"{best['is']['pnl_mdd']:.1f}x")
    all_results['best_candidate'] = {'discriminating': False, 'note': 'No WF 3/3 PASS'}


# ═══════════════════════════════════════════════════════════
# PHASE 5: Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Overall assessment
any_improves_pm = any(g['pnl_mdd'] > baseline_pm for g in grid_results)
any_improves_corr = any(g['max_corr_loss'] < baseline_corr for g in grid_results)
any_improves_mdd = any(g['mdd'] < baseline_stats['mdd'] for g in grid_results)

print(f"\nBaseline: PnL/MDD {baseline_pm:.1f}x, MDD {baseline_stats['mdd']:.2f}%, "
      f"CorrL {baseline_corr:.2f}%, PnL {baseline_stats['pnl']:+.1f}%")
print(f"\nAny combo improves PnL/MDD: {any_improves_pm}")
print(f"Any combo reduces MDD: {any_improves_mdd}")
print(f"Any combo reduces Max Corr Loss: {any_improves_corr}")

if pass_candidates:
    best_pm = max(grid_results, key=lambda x: x['pnl_mdd'])
    best_mdd = min(grid_results, key=lambda x: x['mdd'])
    best_corr = min(grid_results, key=lambda x: x['max_corr_loss'])
    print(f"\nBest PnL/MDD: VT={best_pm['vol_threshold']}, RS={best_pm['reduced_slots']}: "
          f"{best_pm['pnl_mdd']:.1f}x ({best_pm['pnl_mdd']-baseline_pm:+.1f}x)")
    print(f"Best MDD: VT={best_mdd['vol_threshold']}, RS={best_mdd['reduced_slots']}: "
          f"{best_mdd['mdd']:.2f}% ({best_mdd['mdd']-baseline_stats['mdd']:+.2f}%)")
    print(f"Best CorrL: VT={best_corr['vol_threshold']}, RS={best_corr['reduced_slots']}: "
          f"{best_corr['max_corr_loss']:.2f}% ({best_corr['max_corr_loss']-baseline_corr:+.2f}%)")

    # Final recommendation
    recommendation = "KEEP_BASELINE"
    if all_results.get('best_candidate', {}).get('discriminating', False):
        recommendation = "GO"
    print(f"\nRecommendation: {recommendation}")
    all_results['recommendation'] = recommendation
else:
    # No WF passes, check if any grid result is strictly better
    best_pm = max(grid_results, key=lambda x: x['pnl_mdd'])
    best_mdd = min(grid_results, key=lambda x: x['mdd'])
    best_corr = min(grid_results, key=lambda x: x['max_corr_loss'])
    print(f"\nBest PnL/MDD: VT={best_pm['vol_threshold']}, RS={best_pm['reduced_slots']}: "
          f"{best_pm['pnl_mdd']:.1f}x ({best_pm['pnl_mdd']-baseline_pm:+.1f}x)")
    print(f"Best MDD: VT={best_mdd['vol_threshold']}, RS={best_mdd['reduced_slots']}: "
          f"{best_mdd['mdd']:.2f}% ({best_mdd['mdd']-baseline_stats['mdd']:+.2f}%)")
    print(f"Best CorrL: VT={best_corr['vol_threshold']}, RS={best_corr['reduced_slots']}: "
          f"{best_corr['max_corr_loss']:.2f}% ({best_corr['max_corr_loss']-baseline_corr:+.2f}%)")

    recommendation = "KEEP_BASELINE"
    print(f"\nRecommendation: {recommendation}")
    all_results['recommendation'] = recommendation

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.1f}s")
all_results['elapsed_s'] = round(elapsed, 1)

# Save results (convert numpy types for JSON)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)
print(f"\nResults saved to {OUTPUT_FILE}")
