#!/usr/bin/env python3
"""Follow-up: Random discrimination check for promising configs.

From mechanism_cross_validation_study.py:
  - TOP3_only (Timeout+Cascade+Momentum): IS +398.7%, WF OOS +102.1%
  - -M6_Regime: IS +357.0%, WF OOS +87.0%
  - BASELINE: IS +235.3%, WF OOS +58.7%

Need to verify: Are TOP3/noRegime gains genuine or do random patterns also benefit?

Phase 1: 15-seed random discrimination for TOP3 vs BASELINE vs noRegime
Phase 2: Regime Sizing effect decomposition (why does it hurt?)
Phase 3: AggRisk + DirCap ablation cross-check with random
"""

import os, sys, json, time, warnings
from datetime import datetime
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))

from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    compute_atr_ratio, compute_ema_slope,
    portfolio_npos, calc_stats_compound,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_REGIME_MULT, TIMEOUT_BARS,
)

warnings.filterwarnings('ignore')

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'mechanism_disc_followup.json')

RANDOM_SEEDS = list(range(1, 16))  # 15 seeds for stronger signal

CONFIGS = {
    'BASELINE': {
        'n_slots': DEFAULT_N_SLOTS, 'direction_cap': DEFAULT_DIRECTION_CAP,
        'regime_mult': DEFAULT_REGIME_MULT,
        'agg_risk_counter': DEFAULT_AGG_RISK_COUNTER, 'agg_risk_with': DEFAULT_AGG_RISK_WITH,
        'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK, 'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD,
        'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN,
        'clamp_lo': DEFAULT_ATR_CLAMP_LO, 'clamp_hi': DEFAULT_ATR_CLAMP_HI,
        'timeout_bars': TIMEOUT_BARS, 'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
    },
    'NO_REGIME': {
        'n_slots': DEFAULT_N_SLOTS, 'direction_cap': DEFAULT_DIRECTION_CAP,
        'regime_mult': None,
        'agg_risk_counter': DEFAULT_AGG_RISK_COUNTER, 'agg_risk_with': DEFAULT_AGG_RISK_WITH,
        'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK, 'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD,
        'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN,
        'clamp_lo': DEFAULT_ATR_CLAMP_LO, 'clamp_hi': DEFAULT_ATR_CLAMP_HI,
        'timeout_bars': TIMEOUT_BARS, 'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
    },
    'TOP3': {  # Timeout + Cascade + Momentum only
        'n_slots': DEFAULT_N_SLOTS, 'direction_cap': DEFAULT_N_SLOTS,  # DirCap OFF
        'regime_mult': None,  # Regime OFF
        'agg_risk_counter': 999.0, 'agg_risk_with': 999.0,  # AggRisk OFF
        'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK, 'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD,
        'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN,
        'clamp_lo': DEFAULT_ATR_CLAMP_LO, 'clamp_hi': DEFAULT_ATR_CLAMP_HI,
        'timeout_bars': TIMEOUT_BARS, 'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
    },
    'NO_REGIME_NO_DIRCAP': {
        'n_slots': DEFAULT_N_SLOTS, 'direction_cap': DEFAULT_N_SLOTS,
        'regime_mult': None,
        'agg_risk_counter': DEFAULT_AGG_RISK_COUNTER, 'agg_risk_with': DEFAULT_AGG_RISK_WITH,
        'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK, 'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD,
        'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN,
        'clamp_lo': DEFAULT_ATR_CLAMP_LO, 'clamp_hi': DEFAULT_ATR_CLAMP_HI,
        'timeout_bars': TIMEOUT_BARS, 'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
    },
    'NO_REGIME_NO_AGGRISK': {
        'n_slots': DEFAULT_N_SLOTS, 'direction_cap': DEFAULT_DIRECTION_CAP,
        'regime_mult': None,
        'agg_risk_counter': 999.0, 'agg_risk_with': 999.0,
        'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK, 'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD,
        'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN,
        'clamp_lo': DEFAULT_ATR_CLAMP_LO, 'clamp_hi': DEFAULT_ATR_CLAMP_HI,
        'timeout_bars': TIMEOUT_BARS, 'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
    },
}


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {'pattern': v['pattern'], 'direction': v['direction'],
                     'tp_pct': v['tp'], 'sl_pct': v['sl']}
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


def generate_random_patterns(signal_index, n, seed):
    rng = np.random.RandomState(seed)
    trips = list(signal_index.keys())
    chosen = list(rng.choice(trips, min(n, len(trips)), replace=False))
    pats = {}
    for t in chosen:
        d = 'LONG' if rng.random() < 0.5 else 'SHORT'
        pats[f"{t}_{d}"] = {'pattern': t, 'direction': d,
                            'tp_pct': round(rng.uniform(0.8, 2.8), 2),
                            'sl_pct': round(rng.uniform(1.5, 5.5), 2)}
    return pats


def run_sim(sig_tuples, opens, highs, lows, closes, n_bars, ar, es, s, e, params):
    trades, stats = portfolio_npos(sig_tuples, opens, highs, lows, closes, n_bars,
                                   ar, es, s, e, **params)
    cs = calc_stats_compound(trades)
    cs['mdd_mtm'] = stats.get('mdd_mtm', cs['mdd'])
    return cs


def wf_3fold(sig_tuples, opens, highs, lows, closes, n_bars, ar, es, ns, ne, params):
    total = ne - ns
    fold_size = total // 4
    folds = []
    for fold in range(3):
        oos_s = ns + fold_size * (fold + 1)
        oos_e = min(oos_s + fold_size, ne)
        if oos_s >= oos_e:
            continue
        s = run_sim(sig_tuples, opens, highs, lows, closes, n_bars, ar, es, oos_s, oos_e, params)
        folds.append(s['pnl'])
    return folds


def main():
    t0 = time.time()
    print("=" * 90)
    print("MECHANISM DISCRIMINATION FOLLOW-UP (15-seed)")
    print("=" * 90)

    df = load_and_classify(DATA_FILE)
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    n_bars = len(df)
    ar = compute_atr_ratio(highs, lows, closes)
    es = compute_ema_slope(closes)
    ns, ne = find_neutral_window(closes)
    si = build_signal_index(df['candle_type'].values, n_bars)
    patterns = load_patterns()
    real_tuples = build_signal_tuples(patterns, si)
    print(f"  {n_bars} bars, {len(patterns)} patterns, neutral {ns}-{ne}")

    output = {'metadata': {'date': datetime.now().isoformat(), 'seeds': RANDOM_SEEDS}}

    # ==================== Real patterns: IS + WF for all configs ====================
    print("\n" + "=" * 90)
    print("REAL PATTERNS: IS + WF")
    print("=" * 90)

    print(f"\n{'Config':>25} {'IS_PnL':>8} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} {'WF':>6}")
    print("-" * 85)

    real_results = {}
    for name, params in CONFIGS.items():
        is_s = run_sim(real_tuples, opens, highs, lows, closes, n_bars, ar, es, ns, ne, params)
        wf = wf_3fold(real_tuples, opens, highs, lows, closes, n_bars, ar, es, ns, ne, params)
        n_pass = sum(1 for f in wf if f > 0)
        real_results[name] = {'is_pnl': is_s['pnl'], 'wf': wf, 'n_pass': n_pass}
        wf_str = f"{n_pass}/3 {'P' if n_pass==3 else 'F'}"
        folds = [f"{f:>+8.1f}" for f in wf]
        avg = np.mean(wf)
        mn = min(wf)
        print(f"{name:>25} {is_s['pnl']:>+8.1f} {''.join(folds)} {avg:>+8.1f} {mn:>+8.1f} {wf_str:>6}")

    output['real_patterns'] = real_results

    # ==================== Random discrimination: 15 seeds ====================
    print("\n" + "=" * 90)
    print("RANDOM DISCRIMINATION (15 seeds)")
    print("=" * 90)

    rand_data = {name: {'wf_pass': 0, 'is_pnls': [], 'oos_avgs': []} for name in CONFIGS}

    for seed in RANDOM_SEEDS:
        rp = generate_random_patterns(si, 131, seed)
        rt = build_signal_tuples(rp, si)
        for name, params in CONFIGS.items():
            is_s = run_sim(rt, opens, highs, lows, closes, n_bars, ar, es, ns, ne, params)
            wf = wf_3fold(rt, opens, highs, lows, closes, n_bars, ar, es, ns, ne, params)
            n_pass = sum(1 for f in wf if f > 0)
            rand_data[name]['is_pnls'].append(is_s['pnl'])
            rand_data[name]['oos_avgs'].append(np.mean(wf))
            if n_pass == 3:
                rand_data[name]['wf_pass'] += 1

    print(f"\n{'Config':>25} {'WF_PASS':>10} {'Rand_IS_avg':>12} {'Rand_OOS_avg':>13} {'DISC?':>8}")
    print("-" * 75)

    for name in CONFIGS:
        d = rand_data[name]
        wf_rate = f"{d['wf_pass']}/15"
        is_avg = np.mean(d['is_pnls'])
        oos_avg = np.mean(d['oos_avgs'])
        disc = 'NO' if d['wf_pass'] >= 12 else 'PARTIAL' if d['wf_pass'] >= 6 else 'YES'
        print(f"{name:>25} {wf_rate:>10} {is_avg:>+12.1f} {oos_avg:>+13.1f} {disc:>8}")

    output['random_disc'] = {name: {
        'wf_pass': d['wf_pass'], 'is_avg': round(np.mean(d['is_pnls']), 2),
        'oos_avg': round(np.mean(d['oos_avgs']), 2),
    } for name, d in rand_data.items()}

    # ==================== Regime Sizing Decomposition ====================
    print("\n" + "=" * 90)
    print("REGIME SIZING EFFECT DECOMPOSITION")
    print("  Why does regime_mult=0.3 hurt PnL by -121.7%?")
    print("=" * 90)

    # Run with regime ON, track size_mult per trade
    params_regime = dict(CONFIGS['BASELINE'])
    trades_regime, _ = portfolio_npos(real_tuples, opens, highs, lows, closes, n_bars,
                                      ar, es, ns, ne, **params_regime)
    params_no_regime = dict(CONFIGS['NO_REGIME'])
    trades_no_regime, _ = portfolio_npos(real_tuples, opens, highs, lows, closes, n_bars,
                                         ar, es, ns, ne, **params_no_regime)

    # Analyze size_mult distribution
    regime_sizes = [t.get('size_mult', 1.0) for t in trades_regime]
    full_size = sum(1 for s in regime_sizes if s >= 0.9)
    reduced = sum(1 for s in regime_sizes if s < 0.9)

    print(f"\n  Total trades with Regime ON: {len(trades_regime)}")
    print(f"    Full size (≥0.9):  {full_size} ({full_size/len(trades_regime)*100:.1f}%)")
    print(f"    Reduced (<0.9):    {reduced} ({reduced/len(trades_regime)*100:.1f}%)")

    if reduced > 0:
        reduced_trades = [t for t in trades_regime if t.get('size_mult', 1.0) < 0.9]
        avg_sm = np.mean([t.get('size_mult', 1.0) for t in reduced_trades])
        avg_pnl = np.mean([t['pnl_slot'] for t in reduced_trades])
        wr = sum(1 for t in reduced_trades if t['pnl_slot'] > 0) / len(reduced_trades) * 100
        print(f"    Avg size_mult:     {avg_sm:.2f}")
        print(f"    Avg PnL (reduced): {avg_pnl:+.2f}%")
        print(f"    WR (reduced):      {wr:.1f}%")

        full_trades = [t for t in trades_regime if t.get('size_mult', 1.0) >= 0.9]
        if full_trades:
            avg_pnl_full = np.mean([t['pnl_slot'] for t in full_trades])
            wr_full = sum(1 for t in full_trades if t['pnl_slot'] > 0) / len(full_trades) * 100
            print(f"    Avg PnL (full):    {avg_pnl_full:+.2f}%")
            print(f"    WR (full):         {wr_full:.1f}%")

        # Counter-trend trades are actually profitable — regime undersizes winners
        if avg_pnl > 0 or wr > 55:
            print("\n  ⚠ COUNTER-TREND TRADES ARE PROFITABLE but undersized!")
            print("    Regime Sizing is SUPPRESSING profitable counter-trend trades")
        else:
            print("\n  Counter-trend trades are net negative — Regime should help")
            print("  But overall effect is still negative — interaction with other mechanisms?")

    # Direction breakdown
    for direction in ['LONG', 'SHORT']:
        with_r = [t for t in trades_regime if t['direction'] == direction]
        no_r = [t for t in trades_no_regime if t['direction'] == direction]
        pnl_w = sum(t['pnl_portfolio'] for t in with_r)
        pnl_n = sum(t['pnl_portfolio'] for t in no_r)
        print(f"\n  {direction}: Regime ON PnL={pnl_w:+.1f}% ({len(with_r)} trades), "
              f"OFF PnL={pnl_n:+.1f}% ({len(no_r)} trades), delta={pnl_n-pnl_w:+.1f}%")

    output['regime_decomposition'] = {
        'total_trades': len(trades_regime),
        'full_size': full_size, 'reduced': reduced,
    }

    # ==================== VERDICT ====================
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    # Check if TOP3 is genuinely better or just more aggressive
    base_disc = rand_data['BASELINE']['wf_pass']
    top3_disc = rand_data['TOP3']['wf_pass']
    no_regime_disc = rand_data['NO_REGIME']['wf_pass']

    print(f"\n  Discrimination (random WF PASS / 15 seeds):")
    print(f"    BASELINE:        {base_disc}/15")
    print(f"    NO_REGIME:       {no_regime_disc}/15")
    print(f"    TOP3:            {top3_disc}/15")

    if top3_disc >= 12 and base_disc >= 12:
        print("\n  Both are NON-DISCRIMINATING — improvement is from mechanism, not pattern")
        print("  TOP3 amplifies mechanism effect (more trades, no size reduction)")
    elif top3_disc > base_disc + 3:
        print("\n  ⚠ TOP3 is MORE non-discriminating — simplification helps random too")
        print("  Improvement may be from increased risk, not genuine edge")
    else:
        print("\n  Similar discrimination — improvement likely genuine")

    # Final recommendation
    real_base = real_results['BASELINE']
    real_top3 = real_results['TOP3']
    real_no_r = real_results['NO_REGIME']

    if real_no_r['n_pass'] == 3 and real_no_r['is_pnl'] > real_base['is_pnl'] * 1.3:
        print(f"\n  >>> STRONG RECOMMENDATION: Remove Regime Sizing (M6)")
        print(f"      IS: {real_no_r['is_pnl']:+.1f}% vs {real_base['is_pnl']:+.1f}% (+{real_no_r['is_pnl']-real_base['is_pnl']:.0f}%)")
        print(f"      OOS avg: {np.mean(real_no_r['wf']):+.1f}% vs {np.mean(real_base['wf']):+.1f}%")

    if real_top3['n_pass'] == 3 and real_top3['is_pnl'] > real_base['is_pnl'] * 1.5:
        if top3_disc <= base_disc + 3:
            print(f"\n  >>> CONSIDER: TOP3 config (Timeout+Cascade+Momentum only)")
            print(f"      IS: {real_top3['is_pnl']:+.1f}%, OOS avg: {np.mean(real_top3['wf']):+.1f}%")
        else:
            print(f"\n  >>> CAUTION: TOP3 is better but less discriminating")

    runtime = time.time() - t0
    output['metadata']['runtime_sec'] = round(runtime, 1)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Runtime: {runtime:.1f}s")


if __name__ == '__main__':
    main()
