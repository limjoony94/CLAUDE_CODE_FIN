#!/usr/bin/env python3
"""Timeout 288-bar (24h) Cross-Validation Study.

Previous research summary:
  - timeout_sweep_study: v1.47.0 stack → 288 best IS, WF PASS (but old params)
  - timeout_cause_decomposition: 1-pos sim → direction is primary cause (MAE r=0.813)
  - r2_timeout_pnl_decomposition: breakeven_192 promising but NON-DISC from R1
  - timeout_impact_study: old data (720d)

Key question: Does timeout genuinely contribute to PnL, or is it Cascade-dependent?

Phase 1: Ablation — Timeout ON(288) vs OFF, current v1.55.0 full stack
Phase 2: Cascade × Timeout interaction — 2×2 matrix (Cascade ON/OFF × Timeout ON/OFF)
Phase 3: Timeout sweep (144-576) under current stack
Phase 4: Random discrimination — do random patterns also benefit from timeout?
Phase 5: WF 3-fold validation of all variants
Phase 6: Per-trade analysis — timeout exits profitability vs TP/SL

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE = 3 FIXED
  - FEE = FEE_PCT * LEVERAGE (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - WF: 3-fold expanding window
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))

from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    compute_atr_ratio, compute_ema_slope,
    portfolio_npos, calc_stats_compound,
    FEE_PCT, LEVERAGE, SLIPPAGE_BUFFER,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_REGIME_MULT,
    TIMEOUT_BARS,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'timeout_cross_validation.json')

RANDOM_SEEDS = [42, 123, 7777, 2024, 9999]
N_RANDOM_PATTERNS = 131  # same count as real


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp_pct': v['tp'], 'sl_pct': v['sl'],
        }
    return result


def build_signal_tuples(patterns, signal_index):
    tuples = []
    for pat_key, info in patterns.items():
        pat_name = info.get('pattern') or pat_key.rsplit('_', 1)[0]
        if pat_name in signal_index:
            for sig_bar in signal_index[pat_name]:
                tuples.append((sig_bar, pat_key, info['direction'],
                               info['tp_pct'], info['sl_pct']))
    return tuples


def run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            timeout_bars=TIMEOUT_BARS, cascade_pct=DEFAULT_CASCADE_TIGHTEN_PCT):
    """Run portfolio simulation with given timeout and cascade settings."""
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        n_slots=DEFAULT_N_SLOTS,
        direction_cap=DEFAULT_DIRECTION_CAP,
        regime_mult=DEFAULT_REGIME_MULT,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
        agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
        momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        clamp_lo=DEFAULT_ATR_CLAMP_LO,
        clamp_hi=DEFAULT_ATR_CLAMP_HI,
        timeout_bars=timeout_bars,
        cascade_tighten_pct=cascade_pct,
    )
    cstats = calc_stats_compound(trades)
    cstats['mdd_mtm'] = stats.get('mdd_mtm', cstats['mdd'])
    cstats['blocked'] = stats.get('blocked', {})

    # Count exit reasons
    reasons = {}
    for t in trades:
        r = t.get('reason', 'UNKNOWN')
        reasons[r] = reasons.get(r, 0) + 1
    cstats['exit_reasons'] = reasons
    return cstats, trades


def wf_3fold(signal_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, ns, ne,
             timeout_bars=TIMEOUT_BARS, cascade_pct=DEFAULT_CASCADE_TIGHTEN_PCT):
    """3-fold expanding window WF. Returns list of fold PnL values."""
    total = ne - ns
    fold_size = total // 4  # 4 segments → 3 OOS folds

    results = []
    for fold in range(3):
        is_end = ns + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, ne)
        if oos_start >= oos_end:
            continue
        stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, oos_start, oos_end,
                           timeout_bars=timeout_bars, cascade_pct=cascade_pct)
        results.append({
            'fold': fold + 1,
            'oos_bars': oos_end - oos_start,
            'pnl': stats['pnl'],
            'wr': stats['wr'],
            'mdd': stats['mdd'],
            'trades': stats['trades'],
        })
    return results


def generate_random_patterns(signal_index, n_patterns, seed):
    """Generate random pattern set from available triplets."""
    rng = np.random.RandomState(seed)
    all_triplets = list(signal_index.keys())
    if len(all_triplets) < n_patterns:
        chosen = all_triplets
    else:
        chosen = list(rng.choice(all_triplets, n_patterns, replace=False))

    patterns = {}
    for i, trip in enumerate(chosen):
        direction = 'LONG' if rng.random() < 0.5 else 'SHORT'
        tp = rng.uniform(0.8, 2.8)
        sl = rng.uniform(1.5, 5.5)
        key = f"{trip}_{direction}"
        patterns[key] = {
            'pattern': trip, 'direction': direction,
            'tp_pct': round(tp, 2), 'sl_pct': round(sl, 2),
        }
    return patterns


def main():
    t0 = time.time()
    print("=" * 90)
    print("TIMEOUT 288-BAR (24H) CROSS-VALIDATION STUDY")
    print("v1.55.0 stack | Standard Research Protocol")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)
    ns, ne = find_neutral_window(closes)
    print(f"  {n_bars} bars, neutral window: {ns}-{ne} ({ne-ns} bars, {(ne-ns)/288:.0f} days)")

    # Build signal index
    type_codes = df['candle_type'].values
    signal_index = build_signal_index(type_codes, n_bars)

    # Load real patterns
    patterns = load_patterns()
    signal_tuples = build_signal_tuples(patterns, signal_index)
    n_signals = len([s for s in signal_tuples if ns <= s[0] < ne])
    print(f"  {len(patterns)} patterns, {n_signals} signals in neutral window")

    output = {
        'metadata': {
            'script': 'timeout_cross_validation_study.py',
            'date': datetime.now().isoformat(),
            'stack': 'v1.55.0',
            'n_patterns': len(patterns),
            'n_signals': n_signals,
            'neutral_window': [ns, ne],
            'params': {
                'n_slots': DEFAULT_N_SLOTS,
                'direction_cap': DEFAULT_DIRECTION_CAP,
                'cascade_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
                'agg_risk': [DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH],
                'momentum': [DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN],
                'atr_clamp': [DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI],
                'timeout_bars': TIMEOUT_BARS,
            },
        },
    }

    # ==================== PHASE 1: Ablation ====================
    print("\n" + "=" * 90)
    print("PHASE 1: Timeout Ablation (ON=288 vs OFF=99999)")
    print("=" * 90)

    configs_p1 = [
        ('ON_288', TIMEOUT_BARS, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('OFF', 99999, DEFAULT_CASCADE_TIGHTEN_PCT),
    ]

    print(f"\n{'Config':>12} {'PnL%':>8} {'MDD%':>8} {'MTM%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'TP':>6} {'SL':>6} {'DROP':>6} {'OOS':>6}")
    print("-" * 95)

    p1_results = {}
    for label, to, casc in configs_p1:
        stats, trades = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, ns, ne,
                                timeout_bars=to, cascade_pct=casc)
        pm = stats['pnl'] / max(stats['mdd_mtm'], 0.01)
        reasons = stats['exit_reasons']
        p1_results[label] = {**stats, 'pnl_mdd_mtm': round(pm, 2)}
        print(f"{label:>12} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{stats['mdd_mtm']:>8.2f} {pm:>8.1f} {stats['wr']:>6.1f} "
              f"{stats['trades']:>7} {reasons.get('TP',0):>6} {reasons.get('SL',0):>6} "
              f"{reasons.get('DROP',0):>6} {reasons.get('OOS_END',0):>6}")

    delta_pnl = p1_results['ON_288']['pnl'] - p1_results['OFF']['pnl']
    delta_mdd = p1_results['OFF']['mdd_mtm'] - p1_results['ON_288']['mdd_mtm']
    print(f"\n  Timeout ON vs OFF delta: PnL {delta_pnl:+.1f}%, MDD diff {delta_mdd:+.1f}% (positive = ON better)")

    output['phase1_ablation'] = p1_results

    # ==================== PHASE 2: Cascade × Timeout 2×2 ====================
    print("\n" + "=" * 90)
    print("PHASE 2: Cascade × Timeout Interaction (2×2 Matrix)")
    print("=" * 90)

    configs_p2 = [
        ('Casc85_TO288', TIMEOUT_BARS, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('Casc85_TOoff', 99999, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('Casc0_TO288', TIMEOUT_BARS, 0),
        ('Casc0_TOoff', 99999, 0),
    ]

    print(f"\n{'Config':>16} {'PnL%':>8} {'MDD%':>8} {'MTM%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7}")
    print("-" * 70)

    p2_results = {}
    for label, to, casc in configs_p2:
        stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne,
                           timeout_bars=to, cascade_pct=casc)
        pm = stats['pnl'] / max(stats['mdd_mtm'], 0.01)
        p2_results[label] = {**stats, 'pnl_mdd_mtm': round(pm, 2)}
        print(f"{label:>16} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{stats['mdd_mtm']:>8.2f} {pm:>8.1f} {stats['wr']:>6.1f} "
              f"{stats['trades']:>7}")

    # Interaction analysis
    casc_on_to_on = p2_results['Casc85_TO288']['pnl']
    casc_on_to_off = p2_results['Casc85_TOoff']['pnl']
    casc_off_to_on = p2_results['Casc0_TO288']['pnl']
    casc_off_to_off = p2_results['Casc0_TOoff']['pnl']

    timeout_effect_with_cascade = casc_on_to_on - casc_on_to_off
    timeout_effect_without_cascade = casc_off_to_on - casc_off_to_off
    cascade_effect_with_timeout = casc_on_to_on - casc_off_to_on
    cascade_effect_without_timeout = casc_on_to_off - casc_off_to_off
    interaction = timeout_effect_with_cascade - timeout_effect_without_cascade

    print(f"\n  Timeout effect (Cascade ON):  {timeout_effect_with_cascade:+.1f}%")
    print(f"  Timeout effect (Cascade OFF): {timeout_effect_without_cascade:+.1f}%")
    print(f"  Cascade effect (Timeout ON):  {cascade_effect_with_timeout:+.1f}%")
    print(f"  Cascade effect (Timeout OFF): {cascade_effect_without_timeout:+.1f}%")
    print(f"  Interaction term:             {interaction:+.1f}%")

    if abs(timeout_effect_without_cascade) < 1.0 and abs(timeout_effect_with_cascade) > 5.0:
        print("  ⚠ Timeout effect is CASCADE-DEPENDENT (only works with Cascade)")
    elif abs(timeout_effect_without_cascade) > 5.0:
        print("  ✓ Timeout has INDEPENDENT effect (works without Cascade too)")
    else:
        print("  ? Timeout effect is marginal or mixed")

    output['phase2_interaction'] = {
        'matrix': p2_results,
        'timeout_effect_cascade_on': round(timeout_effect_with_cascade, 2),
        'timeout_effect_cascade_off': round(timeout_effect_without_cascade, 2),
        'cascade_effect_timeout_on': round(cascade_effect_with_timeout, 2),
        'cascade_effect_timeout_off': round(cascade_effect_without_timeout, 2),
        'interaction_term': round(interaction, 2),
    }

    # ==================== PHASE 3: Timeout Sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 3: Timeout Sweep (current stack, Cascade 85%)")
    print("=" * 90)

    timeout_vals = [144, 192, 240, 288, 360, 432, 576, 864, 99999]
    timeout_labels = {
        144: '12h', 192: '16h', 240: '20h', 288: '24h', 360: '30h',
        432: '36h', 576: '48h', 864: '72h', 99999: 'OFF',
    }

    print(f"\n{'Timeout':>8} {'Hours':>6} {'PnL%':>8} {'MDD%':>8} {'MTM%':>8} {'P/M':>8} "
          f"{'WR%':>6} {'Trades':>7} {'DROPs':>7}")
    print("-" * 75)

    p3_results = {}
    for to in timeout_vals:
        stats, trades = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, ns, ne,
                                timeout_bars=to)
        pm = stats['pnl'] / max(stats['mdd_mtm'], 0.01)
        drops = stats['exit_reasons'].get('DROP', 0)
        label = timeout_labels.get(to, str(to))
        marker = ' ← current' if to == 288 else ''
        p3_results[str(to)] = {**stats, 'pnl_mdd_mtm': round(pm, 2)}
        print(f"{to:>8} {label:>6} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{stats['mdd_mtm']:>8.2f} {pm:>8.1f} {stats['wr']:>6.1f} "
              f"{stats['trades']:>7} {drops:>7}{marker}")

    output['phase3_sweep'] = p3_results

    # ==================== PHASE 4: Random Discrimination ====================
    print("\n" + "=" * 90)
    print("PHASE 4: Random Pattern Discrimination Test")
    print("  Does timeout improve random patterns too? (NON-DISC check)")
    print("=" * 90)

    rand_on_wins = 0
    rand_off_wins = 0
    rand_wf_on_pass = 0
    rand_wf_off_pass = 0

    print(f"\n{'Seed':>8} {'ON_PnL':>8} {'OFF_PnL':>8} {'ON>OFF':>8} "
          f"{'ON_WF':>8} {'OFF_WF':>8}")
    print("-" * 55)

    p4_results = []
    for seed in RANDOM_SEEDS:
        rand_pats = generate_random_patterns(signal_index, N_RANDOM_PATTERNS, seed)
        rand_tuples = build_signal_tuples(rand_pats, signal_index)

        stats_on, _ = run_sim(rand_tuples, opens, highs, lows, closes, n_bars,
                              atr_ratio, ema_slope, ns, ne,
                              timeout_bars=TIMEOUT_BARS)
        stats_off, _ = run_sim(rand_tuples, opens, highs, lows, closes, n_bars,
                               atr_ratio, ema_slope, ns, ne,
                               timeout_bars=99999)

        on_better = stats_on['pnl'] > stats_off['pnl']
        if on_better:
            rand_on_wins += 1
        else:
            rand_off_wins += 1

        # Quick WF for random
        wf_on = wf_3fold(rand_tuples, opens, highs, lows, closes, n_bars,
                         atr_ratio, ema_slope, ns, ne, timeout_bars=TIMEOUT_BARS)
        wf_off = wf_3fold(rand_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, ns, ne, timeout_bars=99999)

        on_pass = sum(1 for f in wf_on if f['pnl'] > 0)
        off_pass = sum(1 for f in wf_off if f['pnl'] > 0)
        on_wf_str = f"{on_pass}/3"
        off_wf_str = f"{off_pass}/3"
        if on_pass == 3:
            rand_wf_on_pass += 1
        if off_pass == 3:
            rand_wf_off_pass += 1

        p4_results.append({
            'seed': seed,
            'on_pnl': stats_on['pnl'], 'off_pnl': stats_off['pnl'],
            'on_better': on_better,
            'on_wf_pass': on_pass, 'off_wf_pass': off_pass,
        })
        print(f"{seed:>8} {stats_on['pnl']:>+8.1f} {stats_off['pnl']:>+8.1f} "
              f"{'YES' if on_better else 'NO':>8} {on_wf_str:>8} {off_wf_str:>8}")

    print(f"\n  Random ON wins: {rand_on_wins}/{len(RANDOM_SEEDS)}")
    print(f"  Random WF 3/3 PASS (ON): {rand_wf_on_pass}/{len(RANDOM_SEEDS)}")
    print(f"  Random WF 3/3 PASS (OFF): {rand_wf_off_pass}/{len(RANDOM_SEEDS)}")

    if rand_wf_on_pass >= len(RANDOM_SEEDS) * 0.8:
        print("  ⚠ Timeout ON is NON-DISCRIMINATING (random patterns also WF PASS)")
    elif rand_wf_on_pass == 0:
        print("  ✓ Timeout ON is DISCRIMINATING (no random WF PASS)")
    else:
        print(f"  ? Partial discrimination ({rand_wf_on_pass}/{len(RANDOM_SEEDS)})")

    output['phase4_discrimination'] = {
        'results': p4_results,
        'on_wins': rand_on_wins,
        'off_wins': rand_off_wins,
        'wf_on_pass': rand_wf_on_pass,
        'wf_off_pass': rand_wf_off_pass,
    }

    # ==================== PHASE 5: WF 3-fold Validation ====================
    print("\n" + "=" * 90)
    print("PHASE 5: WF 3-fold Validation (Real Patterns)")
    print("=" * 90)

    # Select top IS candidates + current + OFF
    wf_candidates = [
        ('288_current', 288, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('192', 192, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('240', 240, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('360', 360, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('432', 432, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('576', 576, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('864', 864, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('OFF', 99999, DEFAULT_CASCADE_TIGHTEN_PCT),
        ('288_noCasc', 288, 0),
        ('OFF_noCasc', 99999, 0),
    ]

    print(f"\n{'Config':>14} {'F1_PnL':>8} {'F2_PnL':>8} {'F3_PnL':>8} {'Avg':>8} "
          f"{'Min':>8} {'WF':>6} {'IS_PnL':>8}")
    print("-" * 80)

    p5_results = {}
    for label, to, casc in wf_candidates:
        folds = wf_3fold(signal_tuples, opens, highs, lows, closes, n_bars,
                         atr_ratio, ema_slope, ns, ne,
                         timeout_bars=to, cascade_pct=casc)
        is_stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                              atr_ratio, ema_slope, ns, ne,
                              timeout_bars=to, cascade_pct=casc)

        fold_pnls = [f['pnl'] for f in folds]
        avg_pnl = np.mean(fold_pnls) if fold_pnls else 0
        min_pnl = min(fold_pnls) if fold_pnls else 0
        n_pass = sum(1 for p in fold_pnls if p > 0)
        wf_str = f"{n_pass}/3 {'P' if n_pass == 3 else 'F'}"

        p5_results[label] = {
            'folds': [{'pnl': round(f['pnl'], 2), 'wr': f['wr'],
                       'trades': f['trades']} for f in folds],
            'avg_pnl': round(avg_pnl, 2),
            'min_pnl': round(min_pnl, 2),
            'n_pass': n_pass,
            'is_pnl': is_stats['pnl'],
        }

        fold_strs = [f"{p:>+8.1f}" for p in fold_pnls]
        marker = ' ← current' if label == '288_current' else ''
        print(f"{label:>14} {''.join(fold_strs)} {avg_pnl:>+8.1f} "
              f"{min_pnl:>+8.1f} {wf_str:>6} {is_stats['pnl']:>+8.1f}{marker}")

    output['phase5_wf'] = p5_results

    # ==================== PHASE 6: Per-Trade & Slot Analysis ====================
    print("\n" + "=" * 90)
    print("PHASE 6: Per-Trade Exit Analysis + Timeout Slot Liberation")
    print("  NOTE: Timeout exits are DROP (PnL=0, slot freed), NOT recorded as trades")
    print("=" * 90)

    stats_on, trades_on = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                                  atr_ratio, ema_slope, ns, ne,
                                  timeout_bars=TIMEOUT_BARS)
    stats_off, trades_off = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                                    atr_ratio, ema_slope, ns, ne,
                                    timeout_bars=99999)

    # Exit reason breakdown for both
    for label, trades in [('TO_ON', trades_on), ('TO_OFF', trades_off)]:
        by_reason = {}
        for t in trades:
            reason = t.get('reason', 'UNKNOWN')
            if reason not in by_reason:
                by_reason[reason] = {'count': 0, 'total_pnl': 0, 'win': 0}
            by_reason[reason]['count'] += 1
            by_reason[reason]['total_pnl'] += t['pnl_portfolio']
            if t['pnl_slot'] > 0:
                by_reason[reason]['win'] += 1

        print(f"\n  [{label}] Exit reasons:")
        print(f"  {'Reason':>12} {'Count':>7} {'PnL%':>8} {'AvgPnL':>8} {'WR%':>6}")
        print(f"  {'-'*50}")
        for reason in sorted(by_reason.keys()):
            d = by_reason[reason]
            avg = d['total_pnl'] / d['count'] if d['count'] > 0 else 0
            wr = d['win'] / d['count'] * 100 if d['count'] > 0 else 0
            print(f"  {reason:>12} {d['count']:>7} {d['total_pnl']:>+8.1f} "
                  f"{avg:>+8.3f} {wr:>6.1f}")

    # Trade count difference analysis
    n_on = len(trades_on)
    n_off = len(trades_off)
    print(f"\n  Trade count: ON={n_on}, OFF={n_off} (delta={n_on - n_off})")
    print(f"  PnL: ON={stats_on['pnl']:+.1f}%, OFF={stats_off['pnl']:+.1f}%")

    # Slot turnover analysis: timeout frees slots → more entries possible
    blocked_on = stats_on.get('blocked', {})
    blocked_off = stats_off.get('blocked', {})
    print(f"\n  Blocked entries (max_pos): ON={blocked_on.get('max_pos',0)}, OFF={blocked_off.get('max_pos',0)}")
    print(f"  Blocked entries (total): ON={sum(blocked_on.values())}, OFF={sum(blocked_off.values())}")
    print(f"  → Timeout frees {blocked_off.get('max_pos',0) - blocked_on.get('max_pos',0)} more slot opportunities")

    # Holding time distribution comparison
    for label, trades in [('TO_ON', trades_on), ('TO_OFF', trades_off)]:
        holds = [t['exit_bar'] - t['entry_bar'] for t in trades]
        if holds:
            arr = np.array(holds)
            print(f"\n  [{label}] Holding time: "
                  f"P50={np.median(arr)*5/60:.0f}h, P75={np.percentile(arr,75)*5/60:.0f}h, "
                  f"P95={np.percentile(arr,95)*5/60:.0f}h, max={np.max(arr)*5/60:.0f}h")

    p6_results = {
        'trades_on': n_on, 'trades_off': n_off,
        'pnl_on': stats_on['pnl'], 'pnl_off': stats_off['pnl'],
        'blocked_on': blocked_on, 'blocked_off': blocked_off,
    }

    output['phase6_per_trade'] = p6_results

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS & VERDICT")
    print("=" * 90)

    # 1. Does timeout improve PnL? (Phase 1)
    on_pnl = p1_results['ON_288']['pnl']
    off_pnl = p1_results['OFF']['pnl']
    pnl_improve = on_pnl > off_pnl

    # 2. Is it Cascade-dependent? (Phase 2)
    cascade_dep = abs(timeout_effect_without_cascade) < abs(timeout_effect_with_cascade) * 0.3

    # 3. Is it non-discriminating? (Phase 4)
    non_disc = rand_wf_on_pass >= len(RANDOM_SEEDS) * 0.6

    # 4. Does it WF? (Phase 5)
    current_wf = p5_results.get('288_current', {})
    wf_pass = current_wf.get('n_pass', 0) == 3

    # 5. Best alternative
    best_alt = None
    best_alt_min = -999
    for label, data in p5_results.items():
        if data['n_pass'] == 3 and data['min_pnl'] > best_alt_min:
            best_alt = label
            best_alt_min = data['min_pnl']

    print(f"\n  1. PnL improvement (ON vs OFF): {'YES' if pnl_improve else 'NO'} "
          f"({on_pnl:+.1f}% vs {off_pnl:+.1f}%)")
    print(f"  2. Cascade-dependent: {'YES' if cascade_dep else 'NO'} "
          f"(TO effect w/Cascade={timeout_effect_with_cascade:+.1f}%, "
          f"w/o={timeout_effect_without_cascade:+.1f}%)")
    print(f"  3. Non-discriminating: {'YES' if non_disc else 'NO'} "
          f"(random WF PASS: {rand_wf_on_pass}/{len(RANDOM_SEEDS)})")
    print(f"  4. WF 3/3 PASS: {'YES' if wf_pass else 'NO'}")
    print(f"  5. Best WF candidate: {best_alt} (min fold: {best_alt_min:+.1f}%)")

    # Verdict
    if pnl_improve and wf_pass and not cascade_dep:
        verdict = "KEEP_TIMEOUT_288"
        explanation = "Timeout independently improves PnL, WF validated"
    elif pnl_improve and wf_pass and cascade_dep:
        verdict = "KEEP_TIMEOUT_288_WITH_CAVEAT"
        explanation = "Timeout improves PnL but depends on Cascade interaction"
    elif not pnl_improve:
        if best_alt and best_alt != '288_current':
            verdict = f"CONSIDER_{best_alt.upper()}"
            explanation = f"288 does not improve PnL, {best_alt} may be better"
        else:
            verdict = "REMOVE_TIMEOUT"
            explanation = "Timeout does not improve PnL"
    else:
        verdict = "INCONCLUSIVE"
        explanation = "Mixed results, needs further investigation"

    if non_disc:
        verdict += "_NON_DISC"
        explanation += ". WARNING: Non-discriminating (random patterns also pass)"

    print(f"\n  >>> VERDICT: {verdict}")
    print(f"  >>> {explanation}")

    output['synthesis'] = {
        'pnl_improvement': pnl_improve,
        'cascade_dependent': cascade_dep,
        'non_discriminating': non_disc,
        'wf_pass': wf_pass,
        'best_alternative': best_alt,
        'verdict': verdict,
        'explanation': explanation,
    }

    runtime = time.time() - t0
    output['metadata']['runtime_sec'] = round(runtime, 1)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Runtime: {runtime:.1f}s")


if __name__ == '__main__':
    main()
