#!/usr/bin/env python3
"""Mechanism Stack Cross-Validation Study.

Comprehensive ablation + interaction analysis of all active position management
mechanisms in v1.55.0 stack.

Active mechanisms (scanner portfolio_npos):
  M1: Cascade SL Tightening (85%)
  M2: Aggregate Risk Cap (counter 8% / with 15%)
  M3: Direction Cap (7/9)
  M4: Momentum Guard (1.5%/15min, 1h cooldown)
  M5: Position Timeout (288 bars = 24h)
  M6: Regime Sizing (counter-trend × 0.3)

Previous research:
  - Guard ablation (03-03): M3(old)+G4+G3 harmful → disabled
  - Mechanism stack (03-03): M2(regime)+M4(adaptive_lev) redundant → disabled
  - Strategy foundation (03-04): WF 100% NON-DISC, mechanism 86% contribution
  - AggRisk-Cascade validation (03-04): Random 0/15 WF PASS (discriminating)
  - Timeout cross-validation (03-11): Timeout independent of Cascade, KEEP

Phase 1: Individual Ablation (leave-one-out) — 8 configs
Phase 2: Pairwise Interaction Matrix — key pairs
Phase 3: Progressive Stack Build-Up (bare → full)
Phase 4: Random Discrimination (5 seeds)
Phase 5: WF 3-fold Validation (top candidates)
Phase 6: Entry Logic Contribution (pattern quality vs mechanism)

Standard Research Protocol:
  LEVERAGE=3, FEE×LEV, ATR-scaled, Compound, Expanding WF
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from itertools import combinations

import numpy as np

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

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'mechanism_cross_validation.json')

RANDOM_SEEDS = [42, 123, 7777, 2024, 9999]
N_RANDOM = 131


# ============================================================
# Mechanism Configuration
# ============================================================
# Each mechanism: (param_name, on_value, off_value)
MECHANISMS = {
    'M1_Cascade':   {'cascade_tighten_pct': (DEFAULT_CASCADE_TIGHTEN_PCT, 0)},
    'M2_AggRisk':   {'agg_risk_counter': (DEFAULT_AGG_RISK_COUNTER, 999.0),
                     'agg_risk_with': (DEFAULT_AGG_RISK_WITH, 999.0)},
    'M3_DirCap':    {'direction_cap': (DEFAULT_DIRECTION_CAP, DEFAULT_N_SLOTS)},
    'M4_Momentum':  {'momentum_lookback': (DEFAULT_MOMENTUM_LOOKBACK, 0)},
    'M5_Timeout':   {'timeout_bars': (TIMEOUT_BARS, 99999)},
    'M6_Regime':    {'regime_mult': (DEFAULT_REGIME_MULT, None)},
}

# Baseline = all ON
BASELINE_PARAMS = {
    'n_slots': DEFAULT_N_SLOTS,
    'direction_cap': DEFAULT_DIRECTION_CAP,
    'regime_mult': DEFAULT_REGIME_MULT,
    'agg_risk_counter': DEFAULT_AGG_RISK_COUNTER,
    'agg_risk_with': DEFAULT_AGG_RISK_WITH,
    'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK,
    'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD,
    'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN,
    'clamp_lo': DEFAULT_ATR_CLAMP_LO,
    'clamp_hi': DEFAULT_ATR_CLAMP_HI,
    'timeout_bars': TIMEOUT_BARS,
    'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
}


def make_params(disabled_mechanisms=None):
    """Create param dict with specified mechanisms disabled."""
    params = dict(BASELINE_PARAMS)
    if disabled_mechanisms:
        for mech_name in disabled_mechanisms:
            if mech_name in MECHANISMS:
                for param, (on_val, off_val) in MECHANISMS[mech_name].items():
                    params[param] = off_val
    return params


def run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar, params):
    """Run portfolio simulation with given params."""
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        **params,
    )
    cstats = calc_stats_compound(trades)
    cstats['mdd_mtm'] = stats.get('mdd_mtm', cstats['mdd'])
    cstats['blocked'] = stats.get('blocked', {})

    reasons = {}
    for t in trades:
        r = t.get('reason', 'UNK')
        reasons[r] = reasons.get(r, 0) + 1
    cstats['exit_reasons'] = reasons
    return cstats, trades


def wf_3fold(signal_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, ns, ne, params):
    """3-fold expanding window WF."""
    total = ne - ns
    fold_size = total // 4

    results = []
    for fold in range(3):
        is_end = ns + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, ne)
        if oos_start >= oos_end:
            continue
        stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, oos_start, oos_end, params)
        results.append(stats)
    return results


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


def generate_random_patterns(signal_index, n_patterns, seed):
    rng = np.random.RandomState(seed)
    all_triplets = list(signal_index.keys())
    chosen = list(rng.choice(all_triplets, min(n_patterns, len(all_triplets)), replace=False))
    patterns = {}
    for trip in chosen:
        direction = 'LONG' if rng.random() < 0.5 else 'SHORT'
        tp = rng.uniform(0.8, 2.8)
        sl = rng.uniform(1.5, 5.5)
        key = f"{trip}_{direction}"
        patterns[key] = {
            'pattern': trip, 'direction': direction,
            'tp_pct': round(tp, 2), 'sl_pct': round(sl, 2),
        }
    return patterns


def fmt_stats(s, label='', marker=''):
    pm = s['pnl'] / max(s.get('mdd_mtm', s['mdd']), 0.01)
    return (f"{label:>20} {s['pnl']:>+8.1f} {s['mdd']:>8.2f} "
            f"{s.get('mdd_mtm', s['mdd']):>8.2f} {pm:>8.1f} {s['wr']:>6.1f} "
            f"{s['trades']:>7}{marker}")


def print_header():
    print(f"{'Config':>20} {'PnL%':>8} {'MDD%':>8} {'MTM%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7}")
    print("-" * 75)


def main():
    t0 = time.time()
    print("=" * 90)
    print("MECHANISM STACK CROSS-VALIDATION STUDY")
    print("v1.55.0 | 6 Active Mechanisms | Standard Research Protocol")
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
    type_codes = df['candle_type'].values
    signal_index = build_signal_index(type_codes, n_bars)
    patterns = load_patterns()
    signal_tuples = build_signal_tuples(patterns, signal_index)
    n_signals = len([s for s in signal_tuples if ns <= s[0] < ne])
    print(f"  {n_bars} bars, neutral {ns}-{ne} ({(ne-ns)/288:.0f}d), "
          f"{len(patterns)} patterns, {n_signals} signals")

    output = {
        'metadata': {
            'script': 'mechanism_cross_validation_study.py',
            'date': datetime.now().isoformat(),
            'stack': 'v1.55.0',
            'n_patterns': len(patterns),
            'n_signals': n_signals,
            'neutral_window': [ns, ne],
            'mechanisms': {k: {p: {'on': on, 'off': off}
                               for p, (on, off) in v.items()}
                           for k, v in MECHANISMS.items()},
        },
    }

    # ==================== PHASE 1: Individual Ablation ====================
    print("\n" + "=" * 90)
    print("PHASE 1: Individual Ablation (Leave-One-Out)")
    print("  Disable one mechanism at a time; measure PnL/MDD impact")
    print("=" * 90)

    print()
    print_header()

    baseline_params = make_params()
    baseline_stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, ns, ne, baseline_params)
    print(fmt_stats(baseline_stats, 'BASELINE (all ON)'))

    p1_results = {'BASELINE': baseline_stats}
    ablation_impacts = {}

    for mech_name in MECHANISMS:
        params = make_params(disabled_mechanisms=[mech_name])
        stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne, params)
        label = f"-{mech_name}"
        p1_results[label] = stats
        delta_pnl = stats['pnl'] - baseline_stats['pnl']
        pm_base = baseline_stats['pnl'] / max(baseline_stats.get('mdd_mtm', baseline_stats['mdd']), 0.01)
        pm_this = stats['pnl'] / max(stats.get('mdd_mtm', stats['mdd']), 0.01)
        ablation_impacts[mech_name] = {
            'delta_pnl': round(delta_pnl, 2),
            'delta_pm': round(pm_this - pm_base, 2),
        }
        marker = f"  ΔPnL={delta_pnl:+.1f}%"
        print(fmt_stats(stats, label, marker))

    # All OFF
    all_off_params = make_params(disabled_mechanisms=list(MECHANISMS.keys()))
    all_off_stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                               atr_ratio, ema_slope, ns, ne, all_off_params)
    p1_results['ALL_OFF'] = all_off_stats
    delta_all = all_off_stats['pnl'] - baseline_stats['pnl']
    print(fmt_stats(all_off_stats, 'ALL_OFF (bare)', f"  ΔPnL={delta_all:+.1f}%"))

    # Rank by impact
    print("\n  Mechanism contribution ranking (IS PnL lost when removed):")
    ranked = sorted(ablation_impacts.items(), key=lambda x: x[1]['delta_pnl'])
    for rank, (mech, impact) in enumerate(ranked, 1):
        direction = "HARMFUL to remove" if impact['delta_pnl'] < -5 else \
                    "BENEFICIAL to remove" if impact['delta_pnl'] > 5 else "MARGINAL"
        print(f"    {rank}. {mech:>15}: ΔPnL={impact['delta_pnl']:>+7.1f}%, "
              f"ΔP/M={impact['delta_pm']:>+7.1f}  [{direction}]")

    # Sum of individual effects vs total
    sum_individual = sum(v['delta_pnl'] for v in ablation_impacts.values())
    total_effect = all_off_stats['pnl'] - baseline_stats['pnl']
    synergy = total_effect - sum_individual
    print(f"\n  Sum of individual effects: {sum_individual:+.1f}%")
    print(f"  Total (all OFF): {total_effect:+.1f}%")
    print(f"  Synergy/Interaction: {synergy:+.1f}% "
          f"({'positive synergy' if synergy < 0 else 'negative interaction'})")

    output['phase1_ablation'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'blocked'}
                    for k, v in p1_results.items()},
        'impacts': ablation_impacts,
        'synergy': round(synergy, 2),
    }

    # ==================== PHASE 2: Pairwise Interaction ====================
    print("\n" + "=" * 90)
    print("PHASE 2: Pairwise Interaction Matrix")
    print("  Disable pairs; measure interaction term (non-additive effect)")
    print("=" * 90)

    mech_names = list(MECHANISMS.keys())
    # Focus on top-impact pairs (skip trivially marginal mechanisms)
    top_mechs = [m for m, imp in ablation_impacts.items() if abs(imp['delta_pnl']) > 3]
    if len(top_mechs) < 3:
        top_mechs = mech_names[:4]  # fallback: first 4

    print(f"\n  Testing pairs from: {', '.join(top_mechs)}")

    p2_results = {}
    print(f"\n{'Pair':>30} {'PnL%':>8} {'MTM%':>8} {'P/M':>8} {'Interaction':>12}")
    print("-" * 75)

    for m1, m2 in combinations(top_mechs, 2):
        params_pair = make_params(disabled_mechanisms=[m1, m2])
        stats_pair, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, ns, ne, params_pair)

        # Individual effects
        e1 = ablation_impacts[m1]['delta_pnl']
        e2 = ablation_impacts[m2]['delta_pnl']
        pair_delta = stats_pair['pnl'] - baseline_stats['pnl']
        interaction = pair_delta - (e1 + e2)

        pm = stats_pair['pnl'] / max(stats_pair.get('mdd_mtm', stats_pair['mdd']), 0.01)
        label = f"-{m1} -{m2}"

        p2_results[label] = {
            'pnl': stats_pair['pnl'],
            'mdd_mtm': stats_pair.get('mdd_mtm', stats_pair['mdd']),
            'pm': round(pm, 2),
            'pair_delta': round(pair_delta, 2),
            'e1': round(e1, 2), 'e2': round(e2, 2),
            'interaction': round(interaction, 2),
        }
        int_label = f"{interaction:+.1f}%"
        if abs(interaction) > 20:
            int_label += " ⚠STRONG"
        elif abs(interaction) > 10:
            int_label += " ~MODERATE"

        print(f"{label:>30} {stats_pair['pnl']:>+8.1f} "
              f"{stats_pair.get('mdd_mtm', stats_pair['mdd']):>8.2f} "
              f"{pm:>8.1f} {int_label:>12}")

    # Find strongest interactions
    if p2_results:
        strongest = max(p2_results.items(), key=lambda x: abs(x[1]['interaction']))
        print(f"\n  Strongest interaction: {strongest[0]} "
              f"(interaction={strongest[1]['interaction']:+.1f}%)")

    output['phase2_interaction'] = p2_results

    # ==================== PHASE 3: Progressive Build-Up ====================
    print("\n" + "=" * 90)
    print("PHASE 3: Progressive Stack Build-Up")
    print("  Add mechanisms one at a time (largest impact first)")
    print("=" * 90)

    # Sort mechanisms by individual impact (most negative delta = most important)
    sorted_mechs = sorted(ablation_impacts.items(), key=lambda x: x[1]['delta_pnl'])
    build_order = [m for m, _ in sorted_mechs]

    print(f"\n  Build order (most impactful first): {' → '.join(build_order)}")
    print()
    print_header()

    # Start from all OFF
    current_disabled = set(mech_names)
    build_stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                             atr_ratio, ema_slope, ns, ne,
                             make_params(disabled_mechanisms=list(current_disabled)))
    print(fmt_stats(build_stats, 'BARE (all off)'))

    p3_results = [{'config': 'BARE', **{k: v for k, v in build_stats.items() if k != 'blocked'}}]

    for mech in build_order:
        current_disabled.discard(mech)
        params = make_params(disabled_mechanisms=list(current_disabled))
        stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne, params)
        delta = stats['pnl'] - p3_results[-1]['pnl']
        marker = f"  +{mech}: ΔPnL={delta:+.1f}%"
        print(fmt_stats(stats, f"+{mech}", marker))
        p3_results.append({
            'config': f"+{mech}",
            'added': mech,
            'marginal_delta': round(delta, 2),
            **{k: v for k, v in stats.items() if k != 'blocked'},
        })

    # Check diminishing returns
    print("\n  Marginal contribution at each step:")
    for step in p3_results[1:]:
        pct_of_total = step['marginal_delta'] / max(abs(total_effect), 0.01) * 100
        bar = '█' * max(1, int(abs(pct_of_total) / 5))
        print(f"    {step['config']:>20}: {step['marginal_delta']:>+7.1f}% "
              f"({pct_of_total:>+5.1f}% of total) {bar}")

    output['phase3_buildup'] = p3_results

    # ==================== PHASE 4: Random Discrimination ====================
    print("\n" + "=" * 90)
    print("PHASE 4: Random Pattern Discrimination")
    print("  Compare baseline vs all_off with random patterns")
    print("=" * 90)

    rand_results = []
    rand_base_wins = 0
    rand_wf_base_pass = 0
    rand_wf_bare_pass = 0

    print(f"\n{'Seed':>8} {'Base_PnL':>9} {'Bare_PnL':>9} {'Base>Bare':>10} "
          f"{'Base_WF':>8} {'Bare_WF':>8}")
    print("-" * 60)

    for seed in RANDOM_SEEDS:
        rand_pats = generate_random_patterns(signal_index, N_RANDOM, seed)
        rand_tuples = build_signal_tuples(rand_pats, signal_index)

        stats_base, _ = run_sim(rand_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, ns, ne, baseline_params)
        stats_bare, _ = run_sim(rand_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, ns, ne, all_off_params)

        base_wins = stats_base['pnl'] > stats_bare['pnl']
        if base_wins:
            rand_base_wins += 1

        # WF for both
        wf_base = wf_3fold(rand_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne, baseline_params)
        wf_bare = wf_3fold(rand_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne, all_off_params)

        base_pass = sum(1 for f in wf_base if f['pnl'] > 0)
        bare_pass = sum(1 for f in wf_bare if f['pnl'] > 0)
        if base_pass == 3:
            rand_wf_base_pass += 1
        if bare_pass == 3:
            rand_wf_bare_pass += 1

        rand_results.append({
            'seed': seed,
            'base_pnl': stats_base['pnl'], 'bare_pnl': stats_bare['pnl'],
            'base_wins': base_wins,
            'base_wf': base_pass, 'bare_wf': bare_pass,
        })

        print(f"{seed:>8} {stats_base['pnl']:>+9.1f} {stats_bare['pnl']:>+9.1f} "
              f"{'YES' if base_wins else 'NO':>10} "
              f"{base_pass}/3{' P' if base_pass==3 else ' F':>5} "
              f"{bare_pass}/3{' P' if bare_pass==3 else ' F':>5}")

    print(f"\n  Stack wins vs bare: {rand_base_wins}/{len(RANDOM_SEEDS)}")
    print(f"  Random WF PASS (stack): {rand_wf_base_pass}/{len(RANDOM_SEEDS)}")
    print(f"  Random WF PASS (bare):  {rand_wf_bare_pass}/{len(RANDOM_SEEDS)}")

    if rand_wf_base_pass >= 4:
        print("  ⚠ Full stack is NON-DISCRIMINATING (random also WF PASS)")
    if rand_wf_bare_pass == 0 and rand_wf_base_pass >= 3:
        print("  ✓ Mechanism stack IS the discriminator (bare fails, stack passes)")
    if rand_wf_bare_pass >= 3:
        print("  ⚠ Even bare config WF PASS with random — structure is very favorable")

    output['phase4_discrimination'] = {
        'results': rand_results,
        'base_wins': rand_base_wins,
        'wf_base_pass': rand_wf_base_pass,
        'wf_bare_pass': rand_wf_bare_pass,
    }

    # ==================== PHASE 5: WF Validation ====================
    print("\n" + "=" * 90)
    print("PHASE 5: WF 3-fold Validation (Real Patterns)")
    print("=" * 90)

    # Select key configs
    wf_configs = [
        ('BASELINE', baseline_params),
        ('ALL_OFF', all_off_params),
    ]

    # Add top-3 single ablation configs (most impactful to remove)
    for mech, imp in sorted_mechs[:3]:
        wf_configs.append((f'-{mech}', make_params(disabled_mechanisms=[mech])))

    # Add top-3 single ablation configs (least impactful = potential simplification)
    for mech, imp in sorted_mechs[-3:]:
        label = f'-{mech}'
        if not any(l == label for l, _ in wf_configs):
            wf_configs.append((label, make_params(disabled_mechanisms=[mech])))

    # Add strongest interaction pair disabled
    if p2_results:
        strongest_pair = max(p2_results.items(), key=lambda x: abs(x[1]['interaction']))
        pair_mechs = [m.strip() for m in strongest_pair[0].replace('-', '').split()]
        # Parse: "-M1_Cascade -M2_AggRisk" → ['M1_Cascade', 'M2_AggRisk']
        pair_mechs_parsed = []
        for part in strongest_pair[0].split(' -'):
            part = part.strip().lstrip('-')
            if part:
                pair_mechs_parsed.append(part)
        if pair_mechs_parsed:
            wf_configs.append((strongest_pair[0],
                              make_params(disabled_mechanisms=pair_mechs_parsed)))

    # Halfway point: top 3 only
    top3 = [m for m, _ in sorted_mechs[:3]]
    others = [m for m in mech_names if m not in top3]
    wf_configs.append(('TOP3_only', make_params(disabled_mechanisms=others)))

    print(f"\n{'Config':>25} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} "
          f"{'WF':>6} {'IS_PnL':>8}")
    print("-" * 90)

    p5_results = {}
    for label, params in wf_configs:
        folds = wf_3fold(signal_tuples, opens, highs, lows, closes, n_bars,
                         atr_ratio, ema_slope, ns, ne, params)
        is_stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                              atr_ratio, ema_slope, ns, ne, params)

        fold_pnls = [f['pnl'] for f in folds]
        avg_pnl = np.mean(fold_pnls) if fold_pnls else 0
        min_pnl = min(fold_pnls) if fold_pnls else 0
        n_pass = sum(1 for p in fold_pnls if p > 0)
        wf_str = f"{n_pass}/3 {'P' if n_pass == 3 else 'F'}"

        p5_results[label] = {
            'folds': [round(f, 2) for f in fold_pnls],
            'avg': round(avg_pnl, 2), 'min': round(min_pnl, 2),
            'n_pass': n_pass, 'is_pnl': is_stats['pnl'],
        }

        fold_strs = [f"{p:>+8.1f}" for p in fold_pnls]
        marker = ' ← baseline' if label == 'BASELINE' else ''
        print(f"{label:>25} {''.join(fold_strs)} {avg_pnl:>+8.1f} {min_pnl:>+8.1f} "
              f"{wf_str:>6} {is_stats['pnl']:>+8.1f}{marker}")

    output['phase5_wf'] = p5_results

    # ==================== PHASE 6: Entry Logic Contribution ====================
    print("\n" + "=" * 90)
    print("PHASE 6: Entry Logic Contribution (Pattern Quality)")
    print("  How much does pattern selection contribute vs mechanism stack?")
    print("=" * 90)

    # Real patterns + full stack
    real_full, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne, baseline_params)
    # Real patterns + bare
    real_bare, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne, all_off_params)

    # Random patterns + full stack (avg of seeds)
    rand_full_pnls = []
    rand_bare_pnls = []
    for seed in RANDOM_SEEDS:
        rp = generate_random_patterns(signal_index, N_RANDOM, seed)
        rt = build_signal_tuples(rp, signal_index)
        rf, _ = run_sim(rt, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne, baseline_params)
        rb, _ = run_sim(rt, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne, all_off_params)
        rand_full_pnls.append(rf['pnl'])
        rand_bare_pnls.append(rb['pnl'])

    avg_rand_full = np.mean(rand_full_pnls)
    avg_rand_bare = np.mean(rand_bare_pnls)

    print(f"\n  {'':>20} {'Full Stack':>12} {'Bare':>12} {'Stack Effect':>14}")
    print(f"  {'-'*60}")
    print(f"  {'Real patterns':>20} {real_full['pnl']:>+12.1f} {real_bare['pnl']:>+12.1f} "
          f"{real_full['pnl'] - real_bare['pnl']:>+14.1f}")
    print(f"  {'Random (avg)':>20} {avg_rand_full:>+12.1f} {avg_rand_bare:>+12.1f} "
          f"{avg_rand_full - avg_rand_bare:>+14.1f}")
    print(f"  {'Pattern Effect':>20} {real_full['pnl'] - avg_rand_full:>+12.1f} "
          f"{real_bare['pnl'] - avg_rand_bare:>+12.1f}")

    stack_effect = real_full['pnl'] - real_bare['pnl']
    pattern_effect = real_full['pnl'] - avg_rand_full
    total_vs_rand_bare = real_full['pnl'] - avg_rand_bare
    interaction_entry = total_vs_rand_bare - stack_effect - pattern_effect

    print(f"\n  Decomposition (vs random+bare baseline):")
    print(f"    Total improvement:   {total_vs_rand_bare:>+.1f}%")
    print(f"    Stack contribution:  {stack_effect:>+.1f}% "
          f"({stack_effect/max(abs(total_vs_rand_bare),0.01)*100:.0f}%)")
    print(f"    Pattern contribution: {pattern_effect:>+.1f}% "
          f"({pattern_effect/max(abs(total_vs_rand_bare),0.01)*100:.0f}%)")
    print(f"    Interaction:         {interaction_entry:>+.1f}%")

    output['phase6_entry'] = {
        'real_full': real_full['pnl'],
        'real_bare': real_bare['pnl'],
        'rand_full_avg': round(avg_rand_full, 2),
        'rand_bare_avg': round(avg_rand_bare, 2),
        'stack_effect': round(stack_effect, 2),
        'pattern_effect': round(pattern_effect, 2),
        'interaction': round(interaction_entry, 2),
    }

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS & RECOMMENDATIONS")
    print("=" * 90)

    # 1. Essential mechanisms (removing causes >10% PnL loss)
    essential = [m for m, imp in ablation_impacts.items() if imp['delta_pnl'] < -10]
    # 2. Beneficial but not essential (5-10%)
    beneficial = [m for m, imp in ablation_impacts.items() if -10 <= imp['delta_pnl'] < -5]
    # 3. Marginal (-5 to +5%)
    marginal = [m for m, imp in ablation_impacts.items() if -5 <= imp['delta_pnl'] <= 5]
    # 4. Potentially harmful (>+5% when removed = keeping it hurts)
    harmful = [m for m, imp in ablation_impacts.items() if imp['delta_pnl'] > 5]

    print(f"\n  ESSENTIAL (>10% PnL loss when removed): {essential or 'none'}")
    print(f"  BENEFICIAL (5-10%):                      {beneficial or 'none'}")
    print(f"  MARGINAL (<5%):                          {marginal or 'none'}")
    print(f"  POTENTIALLY HARMFUL (gains when removed): {harmful or 'none'}")

    # Stack vs entry contribution
    if abs(stack_effect) > abs(pattern_effect) * 2:
        print(f"\n  System is MECHANISM-DOMINATED ({stack_effect:+.1f}% vs {pattern_effect:+.1f}%)")
    elif abs(pattern_effect) > abs(stack_effect) * 2:
        print(f"\n  System is PATTERN-DOMINATED ({pattern_effect:+.1f}% vs {stack_effect:+.1f}%)")
    else:
        print(f"\n  System is BALANCED (stack={stack_effect:+.1f}%, pattern={pattern_effect:+.1f}%)")

    # Simplification opportunity
    if marginal or harmful:
        simpl = marginal + harmful
        print(f"\n  SIMPLIFICATION CANDIDATES: {simpl}")
        params_simplified = make_params(disabled_mechanisms=simpl)
        s_stats, _ = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                             atr_ratio, ema_slope, ns, ne, params_simplified)
        s_wf = wf_3fold(signal_tuples, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, ns, ne, params_simplified)
        s_pass = sum(1 for f in s_wf if f['pnl'] > 0)
        pm_s = s_stats['pnl'] / max(s_stats.get('mdd_mtm', s_stats['mdd']), 0.01)
        print(f"    Simplified: PnL={s_stats['pnl']:+.1f}%, P/M={pm_s:.1f}, WF={s_pass}/3")
        print(f"    Baseline:   PnL={baseline_stats['pnl']:+.1f}%, "
              f"P/M={baseline_stats['pnl']/max(baseline_stats.get('mdd_mtm',baseline_stats['mdd']),0.01):.1f}")
        if s_stats['pnl'] >= baseline_stats['pnl'] * 0.95 and s_pass == 3:
            print("    → RECOMMENDED: Remove marginal/harmful mechanisms for simplicity")
        else:
            print("    → KEEP all: simplified config underperforms or fails WF")

    output['synthesis'] = {
        'essential': essential,
        'beneficial': beneficial,
        'marginal': marginal,
        'harmful': harmful,
        'system_type': 'mechanism-dominated' if abs(stack_effect) > abs(pattern_effect) * 2
                       else 'pattern-dominated' if abs(pattern_effect) > abs(stack_effect) * 2
                       else 'balanced',
        'stack_effect_pct': round(stack_effect, 2),
        'pattern_effect_pct': round(pattern_effect, 2),
    }

    runtime = time.time() - t0
    output['metadata']['runtime_sec'] = round(runtime, 1)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Runtime: {runtime:.1f}s")


if __name__ == '__main__':
    main()
