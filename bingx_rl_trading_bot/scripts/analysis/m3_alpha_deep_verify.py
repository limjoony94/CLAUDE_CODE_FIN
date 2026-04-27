"""
M3 — Alpha Mechanism Deep Verification
========================================
α (ETH-lag + 고변동성) was the only spec to PASS C1 strict thresholds
(Δp50 +0.16 > 0.10, Δ%>fr +13.5 > 10) with meaningful sample (n=173).

Question: is this real or measurement artifact?

Tests:
  1. 10-seed strict verify (per-seed PASS count) — measurement noise
  2. Friction breakdown — at what friction does α monetize?
  3. Per-horizon gross PnL — entry alpha capture by simple horizons
  4. Compare to random with EXACT same exit logic — entry vs exit attribution
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from m3_critique_pipeline import (prepare_all_data, SPECS, run_bt_with_spec,
                                    trade_summary, EXIT_PARAMS)
from m2_round1_screening import (measure_mfe_for_signals, stats_mfe, isolation_test,
                                  apply_n1_sequencing)
from m2_round2_screening import measure_mfe_random_universe


def main():
    print("Loading data...")
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()

    spec = SPECS['alpha']
    valid = eth_valid
    eligible = (h1 & h4 | (~h1) & (~h4)) & eth_valid

    print("\n" + "=" * 80)
    print("Test 1: α C1 strict verify — 10 seeds, per-seed PASS count")
    print("=" * 80)
    signals = spec['entry_fn'](df, h1, h4, valid, params=spec['parameters'])
    cand_mfe = measure_mfe_for_signals(df, signals, max_bars=8)
    cand_stats = stats_mfe(cand_mfe, friction=0.20)
    print(f"  signals: {len(signals)} → after seq: {cand_stats['n']}")
    print(f"  cand MFE_p50: {cand_stats['mfe_p50']:+.4f} | %>fr: {cand_stats['pct_mfe_gt_friction']}%")

    seeds = list(range(42, 42 + 10 * 100, 100))
    per_seed = []
    for seed in seeds:
        rnd = measure_mfe_random_universe(df, eligible, h1, h4, target_n=cand_stats['n'],
                                            max_bars=8, seed=seed,
                                            direction_by_trend=spec['direction_by_trend'])
        rs = stats_mfe(rnd, 0.20)
        if rs:
            per_seed.append({
                'seed': seed,
                'random_p50': rs['mfe_p50'],
                'random_pct': rs['pct_mfe_gt_friction'],
                'diff_p50': cand_stats['mfe_p50'] - rs['mfe_p50'],
                'diff_pct': cand_stats['pct_mfe_gt_friction'] - rs['pct_mfe_gt_friction'],
            })
    seeds_strict_pass = sum(1 for r in per_seed if r['diff_p50'] >= 0.10 and r['diff_pct'] >= 10.0)
    seeds_relaxed_pass = sum(1 for r in per_seed if r['diff_p50'] >= 0.05 and r['diff_pct'] >= 5.0)

    print(f"\n  Per-seed strict (Δp50≥0.10 AND Δ%>fr≥10):")
    for r in per_seed:
        mark = '✓' if (r['diff_p50'] >= 0.10 and r['diff_pct'] >= 10.0) else ('~' if (r['diff_p50'] >= 0.05 and r['diff_pct'] >= 5.0) else '✗')
        print(f"    seed={r['seed']:>4} Δp50={r['diff_p50']:+.4f} Δ%>fr={r['diff_pct']:+.2f} [{mark}]")
    print(f"\n  Strict PASS: {seeds_strict_pass}/10 seeds")
    print(f"  Relaxed PASS: {seeds_relaxed_pass}/10 seeds")
    avg_dp50 = sum(r['diff_p50'] for r in per_seed) / len(per_seed)
    std_dp50 = float(np.std([r['diff_p50'] for r in per_seed]))
    print(f"  Avg Δp50: {avg_dp50:+.4f} ± {std_dp50:.4f}")

    print("\n" + "=" * 80)
    print("Test 2: Friction breakdown — at what friction does α monetize?")
    print("=" * 80)
    friction_grid = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.80]
    friction_results = []
    for f in friction_grid:
        trades = run_bt_with_spec(df, h1, h4, valid, spec, friction=f)
        s = trade_summary(trades) if trades else None
        if s:
            friction_results.append({
                'friction': f, 'n_trades': s['n'], 'daily_net': s['daily_net'],
                'sum_net': s['sum_net'], 'sum_gross': s['sum_gross'],
                'wr_pct': s['wr_pct'], 'rr': s['rr'],
            })
            print(f"  friction={f:.2f}%: n={s['n']:>3} daily_net={s['daily_net']:+.4f}% "
                  f"WR={s['wr_pct']:.1f}% RR={s['rr']:.2f} sum_gross={s['sum_gross']:+.2f}%")

    # Find break-even friction
    pos_frictions = [r for r in friction_results if r['daily_net'] > 0]
    if pos_frictions:
        max_pos = max(pos_frictions, key=lambda r: r['friction'])
        print(f"\n  Maximum friction at which α profitable: {max_pos['friction']:.2f}% (daily {max_pos['daily_net']:+.4f}%)")
    else:
        print(f"\n  α NOT profitable at any friction tested (incl. 0%) — entry alpha < exit framework drag")

    print("\n" + "=" * 80)
    print("Test 3: Per-horizon gross PnL (fixed-N-bar exit, simplest exit)")
    print("=" * 80)
    for h_bars in [4, 8, 12, 16, 24]:
        iso = isolation_test(df, signals, h_bars, friction=0.20)
        if iso:
            print(f"  N={h_bars:>2} bars (fixed exit): n={iso['n_trades']:>3} "
                  f"gross_sum={iso['gross_sum']:+.2f}% gross_avg={iso['gross_avg']:+.4f}% "
                  f"gross_WR={iso['gross_wr_pct']}%")

    print("\n" + "=" * 80)
    print("Test 4: Random with SAME exit logic (entry vs exit attribution)")
    print("=" * 80)
    print("  α entries vs random entries, both using same trail/SL/timeout exit framework.")
    print("  If α exit BT gross > random exit BT gross → entry alpha real.")
    print("  If α gross ≈ random gross → exit framework is the issue.\n")

    # α BT zero-friction
    alpha_trades_zerof = run_bt_with_spec(df, h1, h4, valid, spec, friction=0.0)
    alpha_s = trade_summary(alpha_trades_zerof)
    print(f"  α (zero friction): n={alpha_s['n']} sum_gross={alpha_s['sum_gross']:+.2f}% "
          f"daily_gross={alpha_s['daily_net']:+.4f}% WR={alpha_s['wr_pct']:.1f}%")

    # Random entries with SAME exit — generate random entries, run through BT
    n_alpha = alpha_s['n']
    random.seed(42)
    op = df['open'].values
    high = df['high'].values
    low = df['low'].values
    cl = df['close'].values

    eligible_idx = np.where(eligible)[0]
    eligible_idx = eligible_idx[(eligible_idx > 0) & (eligible_idx < len(df) - 50)]

    rand_results = []
    for trial_seed in (42, 123, 456):
        random.seed(trial_seed)
        # Pick random entry indices, similar count to alpha
        sampled = sorted(random.sample(eligible_idx.tolist(), min(n_alpha * 3, len(eligible_idx))))
        # Apply N=1 sequencing with same timeout
        rand_signals = []
        last_exit = -1
        for idx in sampled:
            if idx > last_exit:
                # direction by trend
                if h1[idx] and h4[idx]:
                    rand_signals.append((idx, 'LONG'))
                elif (not h1[idx]) and (not h4[idx]):
                    rand_signals.append((idx, 'SHORT'))
                else:
                    continue
                last_exit = idx + EXIT_PARAMS['timeout_bars'] + EXIT_PARAMS['min_bars_between']
        # Build a synthetic spec for these random signals
        rand_spec = dict(spec)
        rand_spec['entry_fn'] = lambda df_, h1_, h4_, v_, params=None: rand_signals
        rand_trades_zerof = run_bt_with_spec(df, h1, h4, valid, rand_spec, friction=0.0)
        rand_s = trade_summary(rand_trades_zerof) if rand_trades_zerof else None
        if rand_s:
            rand_results.append({
                'seed': trial_seed, 'n': rand_s['n'],
                'sum_gross': rand_s['sum_gross'],
                'daily_gross': rand_s['daily_net'],
                'wr': rand_s['wr_pct'],
            })

    avg_rand_daily = sum(r['daily_gross'] for r in rand_results) / max(1, len(rand_results))
    print(f"\n  Random entries × same exit (3 seeds avg):")
    for r in rand_results:
        print(f"    seed={r['seed']} n={r['n']} daily_gross={r['daily_gross']:+.4f}% WR={r['wr']:.1f}%")
    print(f"  avg random daily_gross: {avg_rand_daily:+.4f}%")
    print(f"  α daily_gross (zero friction): {alpha_s['daily_net']:+.4f}%")
    print(f"  diff (α − random): {alpha_s['daily_net'] - avg_rand_daily:+.4f}%")

    if alpha_s['daily_net'] - avg_rand_daily > 0.05:
        print("  → α entry alpha REAL (BT-level positive vs random with same exit)")
    elif alpha_s['daily_net'] - avg_rand_daily > 0:
        print("  → α entry alpha marginal vs random (close to noise)")
    else:
        print("  → α entry advantage NOT preserved through BT exit logic — exit framework destroys edge")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec': 'α ETH-lag + 고변동성',
        'test1_strict_verify': {
            'seeds_strict_pass': seeds_strict_pass,
            'seeds_relaxed_pass': seeds_relaxed_pass,
            'per_seed': per_seed,
            'avg_diff_p50': avg_dp50,
            'std_diff_p50': std_dp50,
        },
        'test2_friction_breakdown': friction_results,
        'test4_entry_vs_exit_attribution': {
            'alpha_zero_friction_daily_gross': alpha_s['daily_net'],
            'random_zero_friction_daily_gross': avg_rand_daily,
            'diff': alpha_s['daily_net'] - avg_rand_daily,
        },
    }
    p = ROOT / 'results' / f'm3_alpha_deep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
