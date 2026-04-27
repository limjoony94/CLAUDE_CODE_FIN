"""
M2 Round 3 — STRICT re-run (artifact verification)
====================================================
Same 9 variants as Round 3, but:
  - Δp50 ≥ 0.10pp (was 0.05)
  - Δ%>fr ≥ 10.0pp (was 5.0)
  - 10 random seeds (was 5) — variance more precise
  - Per-seed PASS rate: of 10 seed measurements, how many independently pass strict?

advisor 권고: "After 24 negative cells, a 25th cell passing is more likely a
measurement artifact than a real signal until proven otherwise."

"Robust PASS" = strict thresholds AND ≥7/10 seeds individually pass
(robust to random baseline noise).
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from m2_round1_screening import (apply_n1_sequencing, isolation_test,
                                  measure_mfe_for_signals, percentile, stats_mfe)
from m2_round2_screening import measure_mfe_random_universe
from m2_round3_screening import (prepare_btc_15m_with_filter, prepare_funding_aligned,
                                   prepare_eth_aligned,
                                   signals_a1_extreme_funding_fade, signals_a2_funding_cross_zero,
                                   signals_a3_sustained_extreme, signals_b1_volume_spike_break,
                                   signals_b2_volume_divergence, signals_b3_vwap_bounce,
                                   signals_c1_spread_mean_rev, signals_c2_correlation_breakdown,
                                   signals_c3_eth_leads_btc)

STRICT_DELTA_P50 = 0.10
STRICT_DELTA_PCT = 10.0
N_SEEDS = 10


def strict_screen(df, h1_long, h4_long, eligible, signals, horizons,
                  label, direction_by_trend=True, friction=0.20):
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    per_day = len(signals) / days if days else 0

    iso = {f'h{h}': isolation_test(df, signals, h, friction) for h in horizons}

    cand_mfe = measure_mfe_for_signals(df, signals, max_bars=horizons[1])
    cand_stats = stats_mfe(cand_mfe, friction)
    if cand_stats is None:
        return {'variant': label, 'raw_signals': len(signals), 'per_day': round(per_day, 3),
                'verdict': 'NO_SIGNALS', 'cand_mfe_p50': None,
                'random_seeds': [], 'random_avg_p50': None, 'random_avg_pct': None,
                'gate5_horizons_pos': 0, 'gate5_pass': False,
                'gate6_diff_p50': None, 'gate6_diff_pct': None,
                'gate6_strict_pass': False, 'seeds_passing_strict': 0,
                'asymmetry': None}

    seeds = list(range(42, 42 + N_SEEDS * 100, 100))[:N_SEEDS]  # 10 seeds: 42, 142, 242, ...
    rnd_per_seed = []
    seeds_passing = 0
    for seed in seeds:
        rnd_samples = measure_mfe_random_universe(df, eligible, h1_long, h4_long,
                                                    target_n=cand_stats['n'],
                                                    max_bars=horizons[1], seed=seed,
                                                    direction_by_trend=direction_by_trend)
        rs = stats_mfe(rnd_samples, friction)
        if rs is None:
            rnd_per_seed.append(None)
            continue
        rnd_per_seed.append(rs)
        # per-seed strict check
        seed_dp = cand_stats['mfe_p50'] - rs['mfe_p50']
        seed_dpct = cand_stats['pct_mfe_gt_friction'] - rs['pct_mfe_gt_friction']
        if seed_dp >= STRICT_DELTA_P50 and seed_dpct >= STRICT_DELTA_PCT:
            seeds_passing += 1

    valid_seeds = [r for r in rnd_per_seed if r]
    if not valid_seeds:
        return {'variant': label, 'raw_signals': len(signals), 'per_day': round(per_day, 3),
                'verdict': 'INSUFFICIENT', 'gate5_pass': False, 'gate6_strict_pass': False}

    rnd_p50_avg = sum(r['mfe_p50'] for r in valid_seeds) / len(valid_seeds)
    rnd_pct_avg = sum(r['pct_mfe_gt_friction'] for r in valid_seeds) / len(valid_seeds)
    rnd_p50_std = float(np.std([r['mfe_p50'] for r in valid_seeds]))
    rnd_pct_std = float(np.std([r['pct_mfe_gt_friction'] for r in valid_seeds]))

    diff_p50 = cand_stats['mfe_p50'] - rnd_p50_avg
    diff_pct = cand_stats['pct_mfe_gt_friction'] - rnd_pct_avg

    asym = cand_stats['mfe_p50'] + cand_stats['mae_p50']

    gate5_pos = sum(1 for r in iso.values() if r and r['gross_sum'] > 0)
    gate5_pass = gate5_pos >= 2
    gate6_strict_pass = (diff_p50 >= STRICT_DELTA_P50) and (diff_pct >= STRICT_DELTA_PCT)
    robust_pass = gate5_pass and gate6_strict_pass and seeds_passing >= 7

    if robust_pass:
        verdict = 'ROBUST_PASS'
    elif gate5_pass and gate6_strict_pass:
        verdict = 'STRICT_PASS_BUT_FRAGILE'  # 평균 PASS but seeds <7
    elif gate5_pass and not gate6_strict_pass:
        verdict = 'FAIL_G6_STRICT'
    elif not gate5_pass and gate6_strict_pass:
        verdict = 'FAIL_G5'
    else:
        verdict = 'FAIL_BOTH'

    return {
        'variant': label,
        'raw_signals': len(signals), 'per_day': round(per_day, 3),
        'cand_mfe_p50': cand_stats['mfe_p50'],
        'cand_mae_p50': cand_stats['mae_p50'],
        'cand_pct_above': cand_stats['pct_mfe_gt_friction'],
        'asymmetry': round(asym, 4),
        'isolation': iso,
        'gate5_horizons_pos': gate5_pos,
        'gate5_pass': bool(gate5_pass),
        'random_seeds_count': len(valid_seeds),
        'random_avg_p50': round(rnd_p50_avg, 4),
        'random_p50_std': round(rnd_p50_std, 4),
        'random_avg_pct': round(rnd_pct_avg, 4),
        'random_pct_std': round(rnd_pct_std, 4),
        'gate6_diff_p50': round(diff_p50, 4),
        'gate6_diff_pct': round(diff_pct, 2),
        'gate6_strict_pass': bool(gate6_strict_pass),
        'seeds_passing_strict': seeds_passing,
        'seeds_total': N_SEEDS,
        'verdict': verdict,
    }


def main():
    print("Loading + indicators (BTC 15m + 1h + 4h)...")
    df_btc, h1, h4, valid_btc = prepare_btc_15m_with_filter()
    df_funding = prepare_funding_aligned(df_btc)
    valid_funding = valid_btc & (~pd.isna(df_funding['funding_pct'])).values
    df_cross = prepare_eth_aligned(df_btc)
    valid_cross = (valid_btc & (~pd.isna(df_cross['eth_close']))
                    & (~pd.isna(df_cross['ratio_z'])) & (~pd.isna(df_cross['corr50']))).values

    H = [4, 8, 16]
    elig_btc = (h1 & h4 | (~h1) & (~h4)) & valid_btc
    elig_funding = (h1 & h4 | (~h1) & (~h4)) & valid_funding
    elig_funding_no_filter = valid_funding
    elig_cross = (h1 & h4 | (~h1) & (~h4)) & valid_cross

    config = [
        ('A.1_extreme_funding_fade', signals_a1_extreme_funding_fade, df_funding, valid_funding, elig_funding_no_filter, False),
        ('A.2_funding_cross_zero', signals_a2_funding_cross_zero, df_funding, valid_funding, elig_funding, True),
        ('A.3_sustained_extreme', signals_a3_sustained_extreme, df_funding, valid_funding, elig_funding_no_filter, False),
        ('B.1_volume_spike_break', signals_b1_volume_spike_break, df_btc, valid_btc, elig_btc, True),
        ('B.2_volume_divergence_fade', signals_b2_volume_divergence, df_btc, valid_btc, elig_btc, True),
        ('B.3_vwap_bounce', signals_b3_vwap_bounce, df_btc, valid_btc, elig_btc, True),
        ('C.1_spread_mean_rev', signals_c1_spread_mean_rev, df_cross, valid_cross, elig_cross, True),
        ('C.2_correlation_breakdown', signals_c2_correlation_breakdown, df_cross, valid_cross, elig_cross, True),
        ('C.3_eth_leads_btc', signals_c3_eth_leads_btc, df_cross, valid_cross, elig_cross, True),
    ]

    results = []
    print(f"\nStrict criteria: Δp50 ≥ {STRICT_DELTA_P50}pp AND Δ%>fr ≥ {STRICT_DELTA_PCT}pp")
    print(f"Random baseline: {N_SEEDS} seeds")
    print(f"Robust PASS = strict + ≥7/10 seeds individually pass\n")

    print("=" * 100)
    for label, fn, df_use, valid, elig, dir_by_trend in config:
        sigs = fn(df_use, h1, h4, valid)
        r = strict_screen(df_use, h1, h4, elig, sigs, H, label,
                           direction_by_trend=dir_by_trend)
        results.append(r)
        sps = r.get('seeds_passing_strict', 0)
        st = r.get('seeds_total', N_SEEDS)
        rnd_p50 = r.get('random_avg_p50')
        rnd_std = r.get('random_p50_std')
        cand_p50 = r.get('cand_mfe_p50')
        diff_p50 = r.get('gate6_diff_p50')
        diff_pct = r.get('gate6_diff_pct')
        verdict = r.get('verdict', 'FAIL')
        print(f"{label:<30} sigs={r['raw_signals']:>5} ({r['per_day']:.2f}/day) "
              f"cand_p50={cand_p50 if cand_p50 is None else f'{cand_p50:+.4f}'} "
              f"rnd_p50={rnd_p50 if rnd_p50 is None else f'{rnd_p50:+.4f}±{rnd_std:.4f}'} "
              f"Δp50={diff_p50 if diff_p50 is None else f'{diff_p50:+.4f}'} "
              f"Δ%>fr={diff_pct if diff_pct is None else f'{diff_pct:+.2f}'} "
              f"seeds={sps}/{st} → {verdict}")

    print("\n" + "=" * 100)
    print("STRICT SUMMARY")
    print("=" * 100)
    print(f"{'cell':<32} {'Δp50':>9} {'Δ%>fr':>8} {'seeds':>7} {'asym':>9} {'verdict':>26}")
    for r in results:
        p50 = r.get('gate6_diff_p50')
        pct = r.get('gate6_diff_pct')
        sd = r.get('seeds_passing_strict', 0)
        asym = r.get('asymmetry')
        p50_s = f"{p50:+.4f}" if p50 is not None else "  N/A"
        pct_s = f"{pct:+.2f}" if pct is not None else " N/A"
        asym_s = f"{asym:+.4f}" if asym is not None else "  N/A"
        seed_s = f"{sd}/{N_SEEDS}"
        print(f"{r['variant']:<32} {p50_s:>9} {pct_s:>8} {seed_s:>7} {asym_s:>9} {r.get('verdict','FAIL'):>26}")

    n_robust = sum(1 for r in results if r.get('verdict') == 'ROBUST_PASS')
    n_fragile = sum(1 for r in results if r.get('verdict') == 'STRICT_PASS_BUT_FRAGILE')
    print(f"\nROBUST PASS: {n_robust}/9")
    print(f"STRICT PASS but FRAGILE (seeds<7): {n_fragile}/9")

    # Find C.3 specifically
    c3 = next((r for r in results if r['variant'] == 'C.3_eth_leads_btc'), None)
    if c3:
        print(f"\nC.3 (was Round 3 PASS) under strict:")
        print(f"  Δp50: {c3.get('gate6_diff_p50'):+.4f} (strict ≥ 0.10) → {'PASS' if c3.get('gate6_diff_p50', 0) >= 0.10 else 'FAIL'}")
        print(f"  Δ%>fr: {c3.get('gate6_diff_pct'):+.2f} (strict ≥ 10.0) → {'PASS' if c3.get('gate6_diff_pct', 0) >= 10.0 else 'FAIL'}")
        print(f"  Seeds passing: {c3.get('seeds_passing_strict')}/{N_SEEDS} → {'ROBUST' if c3.get('seeds_passing_strict', 0) >= 7 else 'FRAGILE'}")
        print(f"  Final: {c3.get('verdict')}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'criteria': {
            'strict_delta_p50_pp': STRICT_DELTA_P50,
            'strict_delta_pct_pp': STRICT_DELTA_PCT,
            'n_random_seeds': N_SEEDS,
            'robust_pass_seeds_threshold': 7,
        },
        'results': results,
        'n_robust_pass': n_robust,
        'n_strict_fragile': n_fragile,
    }
    p = ROOT / 'results' / f'm2_round3_strict_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
