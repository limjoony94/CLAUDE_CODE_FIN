"""M3-R9b — Fixed N exit at low friction (exit framework drag elimination probe).

Top 3 candidates: κ, ι, α
For each: fixed N exit (no trail, no SL, only emergency + N timeout) × friction grid.
Find ANY combination producing positive daily.
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import (prepare_all_data, run_bt_with_spec, trade_summary,
                                    entry_alpha)
from m3_round2_critique import (prepare_data_with_eth_break, entry_iota,
                                  ALPHA_PRIME_EXIT_PARAMS)
from m3_round8_critique import entry_kappa, prepare_data_r8


def make_fixed_exit_params(N):
    return {
        'use_sl': False, 'use_trail': False,
        'sl_atr_mult': 0.0, 'trail_k': 0.0,
        'emergency_pct': 1.5,
        'timeout_bars': N,
        'min_bars_between': 2,
    }


def main():
    print("Loading data...")
    df, h1, h4, base_valid, eth_valid_ext, funding_valid, kappa_valid, _ = prepare_data_r8()
    eligible_with_filter = (h1 & h4 | (~h1) & (~h4))

    candidates = [
        {
            'name': 'κ (ι + MID-vol regime)',
            'entry_fn': entry_kappa,
            'params': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'eth_break_lookback': 24},
            'valid': eth_valid_ext & (~pd.isna(df['atr_pctile_30'])).values & (~pd.isna(df['atr_pctile_70'])).values,
            'direction_by_trend': True,
        },
        {
            'name': 'ι (α + ETH 24-bar break)',
            'entry_fn': entry_iota,
            'params': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0, 'eth_break_lookback': 24},
            'valid': eth_valid_ext,
            'direction_by_trend': True,
        },
        {
            'name': 'α (ETH-lag + 고변동성)',
            'entry_fn': entry_alpha,
            'params': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0},
            'valid': eth_valid_ext,
            'direction_by_trend': True,
        },
    ]

    N_grid = (2, 4, 6, 8, 12, 16, 24)
    friction_grid = (0.00, 0.02, 0.04, 0.06, 0.10, 0.20)

    results = {}
    print(f"\n{'mechanism':<30} {'N':>3} {'fric':>6} {'daily_net':>12} {'WR':>6} {'RR':>6} {'n':>5} {'gross_sum':>12}")
    print("-" * 95)
    for cand in candidates:
        name = cand['name']
        results[name] = {}
        for N in N_grid:
            for friction in friction_grid:
                spec = {
                    'name': f'{name}_N{N}_f{friction}',
                    'entry_fn': cand['entry_fn'],
                    'parameters': cand['params'],
                    'direction_by_trend': cand['direction_by_trend'],
                    'exit_params': make_fixed_exit_params(N),
                }
                trades = run_bt_with_spec(df, h1, h4, cand['valid'], spec, friction=friction)
                if not trades:
                    continue
                s = trade_summary(trades, friction=friction)
                key = f'N={N}_f={friction}'
                results[name][key] = {
                    'N': N, 'friction': friction,
                    'daily_net': s['daily_net'], 'wr_pct': s['wr_pct'],
                    'rr': s['rr'], 'n': s['n'], 'sum_gross': s['sum_gross'],
                    'avg_gross': s['avg_gross'],
                }
                # Print only if positive or near-zero
                if s['daily_net'] > -0.005:
                    print(f"  {name:<28} {N:>3} {friction:>5.2f}% {s['daily_net']:>+11.4f}% {s['wr_pct']:>5.1f}% {s['rr']:>5.2f} {s['n']:>5} {s['sum_gross']:>+11.2f}%")

    # Identify any positive daily configurations
    print("\n" + "=" * 100)
    print("M3-R9b — POSITIVE DAILY CONFIGURATIONS (any N × friction combo)")
    print("=" * 100)
    any_positive = False
    for name, configs in results.items():
        positive = [(k, c) for k, c in configs.items() if c['daily_net'] > 0]
        if positive:
            any_positive = True
            print(f"\n  {name}:")
            for k, c in sorted(positive, key=lambda x: -x[1]['daily_net'])[:5]:
                print(f"    {k}: daily_net={c['daily_net']:+.4f}% WR={c['wr_pct']:.1f}% RR={c['rr']:.2f} n={c['n']} gross_sum={c['sum_gross']:+.2f}%")
    if not any_positive:
        print("\n  *** ZERO POSITIVE DAILY ACROSS ALL N × FRICTION COMBINATIONS ***")
        print("  → Entry alpha < exit framework drag at every friction level (incl. zero)")
        print("  → Mechanism exhaustion confirmed at gross level")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'note': 'Fixed N exit at low friction probe — exit framework drag elimination',
           'N_grid': list(N_grid), 'friction_grid': list(friction_grid),
           'results': results,
           'any_positive_daily': any_positive}
    p = ROOT / 'results' / f'm3_r9b_fixed_exit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
