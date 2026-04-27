"""
Phase 2.7 Diagnostic: Random Entry Baseline C — MFE/MAE Comparison
===================================================================
Pre-registered Baseline C (claudedocs/m1_baseline_definition.md).
M1-A entry filter (RSI cross + body + EMA9 + 15m D3) vs random entry on same
trend-filtered (1h+4h aligned) universe.

비교:
  - 같은 trend-filter universe (1h+4h aligned)
  - 같은 N=1, 2-bar cooldown
  - 같은 sample size (~1565 from M1-A)
  - 같은 24-bar window MFE/MAE
  - Random direction (LONG/SHORT)는 trend filter 방향 따름

해석 (advisor 권고):
  - Random MFE P50 ≈ M1-A MFE P50 → entry filter 기여 zero, M1-A shelve + paradigm shift
  - Random MFE P50 < M1-A MFE P50 → filter는 volatility-rich 선택은 함, but no directional edge
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from m1_bt_framework import M1ABot, prepare_data
from m1_mfe_distribution import percentile


def measure_mfe_random(df_5m, h1_long, h4_long, valid_mask, target_n=1565,
                       max_bars=24, seed=42):
    """Random entry within trend-filtered (1h+4h aligned) universe.
    Direction = trend direction (LONG if 1h_long & 4h_long, else SHORT).
    N=1, 2-bar cooldown.
    """
    random.seed(seed)
    n = len(df_5m)
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    closes = df_5m['close'].values

    # Eligible bars: trend-aligned (LONG or SHORT) AND valid
    eligible_long = h1_long & h4_long & valid_mask
    eligible_short = (~h1_long) & (~h4_long) & valid_mask
    eligible = eligible_long | eligible_short

    eligible_indices = np.where(eligible)[0]
    eligible_indices = eligible_indices[(eligible_indices > 0) & (eligible_indices < n - max_bars - 1)]
    if len(eligible_indices) < target_n:
        print(f"WARN: only {len(eligible_indices)} eligible bars < target {target_n}")
        target_n = len(eligible_indices)

    # Oversample then enforce N=1 spacing greedily (mirror M1-A sequencing).
    pool = eligible_indices.tolist()
    needed = min(target_n * 5, len(pool))
    random_samples = sorted(random.sample(pool, needed))
    samples = []
    last_exit = -1
    for idx in random_samples:
        if idx > last_exit:
            samples.append(idx)
            last_exit = idx + max_bars + 2  # entry occupies up to entry+max_bars, then 2-bar cooldown
            if len(samples) >= target_n:
                break

    # Compute MFE/MAE per sample
    bot = M1ABot()
    out = []
    for entry_signal_bar in samples:
        ni = entry_signal_bar + 1
        if ni >= n:
            continue
        # direction by trend
        if h1_long[entry_signal_bar] and h4_long[entry_signal_bar]:
            pdir = 'LONG'
        else:
            pdir = 'SHORT'
        pentry = opens[ni]
        pstart = ni
        pend = min(ni + max_bars, n - 1)

        if pdir == 'LONG':
            mfe_idx = max(range(pstart, pend + 1), key=lambda k: highs[k])
            mae_idx = min(range(pstart, pend + 1), key=lambda k: lows[k])
            mfe_pct = (highs[mfe_idx] / pentry - 1) * 100
            mae_pct = (lows[mae_idx] / pentry - 1) * 100
            final_pct = (closes[pend] / pentry - 1) * 100
        else:
            mfe_idx = min(range(pstart, pend + 1), key=lambda k: lows[k])
            mae_idx = max(range(pstart, pend + 1), key=lambda k: highs[k])
            mfe_pct = (1 - lows[mfe_idx] / pentry) * 100
            mae_pct = (1 - highs[mae_idx] / pentry) * 100
            final_pct = (1 - closes[pend] / pentry) * 100

        out.append({
            'direction': pdir,
            'entry_bar': pstart,
            'mfe_pct': round(mfe_pct, 4),
            'mae_pct': round(mae_pct, 4),
            'final_pct': round(final_pct, 4),
            'mfe_offset': mfe_idx - pstart,
        })
    return out


def stats(label, samples):
    if not samples:
        return None
    mfes = [s['mfe_pct'] for s in samples]
    maes = [s['mae_pct'] for s in samples]
    finals = [s['final_pct'] for s in samples]
    return {
        'label': label,
        'n': len(samples),
        'mfe_mean': round(sum(mfes)/len(mfes), 4),
        'mfe_p25': round(percentile(mfes, 25), 4),
        'mfe_p50': round(percentile(mfes, 50), 4),
        'mfe_p75': round(percentile(mfes, 75), 4),
        'mfe_p90': round(percentile(mfes, 90), 4),
        'mae_mean': round(sum(maes)/len(maes), 4),
        'mae_p25': round(percentile(maes, 25), 4),
        'mae_p50': round(percentile(maes, 50), 4),
        'mae_p75': round(percentile(maes, 75), 4),
        'mae_p10': round(percentile(maes, 10), 4),
        'final_mean': round(sum(finals)/len(finals), 4),
        'final_p50': round(percentile(finals, 50), 4),
        'pct_mfe_gt_friction_0.20': round(100 * sum(1 for x in mfes if x > 0.20) / len(mfes), 1),
        'pct_mfe_gt_2x_friction_0.40': round(100 * sum(1 for x in mfes if x > 0.40) / len(mfes), 1),
    }


def main():
    print("Loading + indicators...")
    df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask = prepare_data(
        ROOT / 'data' / 'btc_5m_720days_binance.csv',
        ROOT / 'data' / 'btc_15m_720days.csv',
        ROOT / 'data' / 'btc_1h_720days.csv',
    )
    print(f"  5m: {len(df_5m):,}, valid: {int(valid_mask.sum()):,}\n")

    # Load M1-A MFE for comparison
    m1_results = sorted(Path(ROOT / 'results').glob('m1_mfe_dist_*.json'))
    if m1_results:
        with open(m1_results[-1]) as f:
            m1a = json.load(f)
    else:
        print("ERROR: M1-A MFE results not found")
        return

    print(f"M1-A reference: {m1_results[-1].name}")
    print(f"  MFE P50 = {m1a['mfe']['p50']:+.4f}%")
    print(f"  MAE P50 = {m1a['mae']['p50']:+.4f}%")
    print(f"  Final P50 = {m1a['final']['p50']:+.4f}%\n")

    # Run random baselines with multiple seeds
    print("Running random-entry baselines (5 seeds)...")
    seeds = [42, 123, 456, 789, 1234]
    all_random = []
    for seed in seeds:
        samples = measure_mfe_random(df_5m, h1_long, h4_long, valid_mask,
                                      target_n=1565, max_bars=24, seed=seed)
        s = stats(f'random_seed_{seed}', samples)
        all_random.append(s)
        print(f"  seed={seed}: n={s['n']} MFE_p50={s['mfe_p50']:+.4f}% MAE_p50={s['mae_p50']:+.4f}% Final_p50={s['final_p50']:+.4f}%")

    # Aggregate (mean of medians across seeds)
    avg_mfe_p50 = sum(r['mfe_p50'] for r in all_random) / len(all_random)
    avg_mae_p50 = sum(r['mae_p50'] for r in all_random) / len(all_random)
    avg_final_p50 = sum(r['final_p50'] for r in all_random) / len(all_random)
    avg_pct_mfe_gt_0_20 = sum(r['pct_mfe_gt_friction_0.20'] for r in all_random) / len(all_random)

    print(f"\n=== Random baseline aggregate (mean across 5 seeds) ===")
    print(f"  MFE P50 (avg): {avg_mfe_p50:+.4f}%")
    print(f"  MAE P50 (avg): {avg_mae_p50:+.4f}%")
    print(f"  Final P50 (avg): {avg_final_p50:+.4f}%")
    print(f"  % MFE > 0.20% (avg): {avg_pct_mfe_gt_0_20:.1f}%")

    print(f"\n=== Side-by-side ===")
    print(f"  {'metric':<25} {'M1-A':>10} {'random':>10} {'diff':>10}")
    print(f"  {'MFE P50':<25} {m1a['mfe']['p50']:>+10.4f} {avg_mfe_p50:>+10.4f} {m1a['mfe']['p50']-avg_mfe_p50:>+10.4f}")
    print(f"  {'MAE P50':<25} {m1a['mae']['p50']:>+10.4f} {avg_mae_p50:>+10.4f} {m1a['mae']['p50']-avg_mae_p50:>+10.4f}")
    print(f"  {'Final close P50':<25} {m1a['final']['p50']:>+10.4f} {avg_final_p50:>+10.4f} {m1a['final']['p50']-avg_final_p50:>+10.4f}")
    print(f"  {'% MFE > 0.20%':<25} {m1a['pct_mfe_gt_friction']:>10.1f} {avg_pct_mfe_gt_0_20:>10.1f} {m1a['pct_mfe_gt_friction']-avg_pct_mfe_gt_0_20:>+10.1f}")

    print(f"\n=== Verdict ===")
    diff_mfe = m1a['mfe']['p50'] - avg_mfe_p50
    diff_pct_above = m1a['pct_mfe_gt_friction'] - avg_pct_mfe_gt_0_20
    if abs(diff_mfe) < 0.05 and abs(diff_pct_above) < 5:
        print("Random ≈ M1-A. RSI/body/EMA9/15m filter는 directional edge 무력.")
        print("=> M1-A 폐기. paradigm shift 필요.")
        verdict = 'NO_FILTER_EDGE'
    elif diff_mfe > 0.10 or diff_pct_above > 10:
        print("M1-A > random in MFE — filter는 volatility-rich 선택. 단 directional edge는 약함.")
        print("=> 사용자 보고. monetization 방법 재고 필요.")
        verdict = 'WEAK_VOLATILITY_FILTER'
    else:
        print("M1-A marginally > random. Filter 효과 minimal.")
        print("=> 사용자 보고. 추가 진단 vs paradigm shift.")
        verdict = 'MARGINAL'

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'baseline_C_definition': 'Random entry within 1h+4h trend-filtered universe, N=1, 2-bar cooldown',
        'm1a_reference': {'mfe_p50': m1a['mfe']['p50'], 'mae_p50': m1a['mae']['p50'],
                          'final_p50': m1a['final']['p50'],
                          'pct_mfe_gt_0.20': m1a['pct_mfe_gt_friction']},
        'random_per_seed': all_random,
        'random_aggregate': {
            'mfe_p50_avg': avg_mfe_p50,
            'mae_p50_avg': avg_mae_p50,
            'final_p50_avg': avg_final_p50,
            'pct_mfe_gt_0.20_avg': avg_pct_mfe_gt_0_20,
        },
        'comparison': {
            'mfe_p50_diff': diff_mfe,
            'pct_mfe_gt_0.20_diff': diff_pct_above,
        },
        'verdict': verdict,
    }
    p = ROOT / 'results' / f'm1_random_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
