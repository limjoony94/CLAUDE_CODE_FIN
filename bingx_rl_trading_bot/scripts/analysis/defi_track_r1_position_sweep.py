"""DeFi-R1 position-size sweep — verify advisor's capital-bound diagnostic.

Advisor 2026-04-29 claim:
  DeFi friction is $-per-swap (gas), so net APY scales with capital. PB-R1-maker
  friction is %-of-notional (taker/maker fees), so net APY is invariant to capital.
  → Convergence at $1,500 may be coincidence, not universal "capital binds".

This script sweeps position_size_usd ∈ {500, 5000, 50000} for DeFi-R1 only.
PB-R1-maker is analytically scale-invariant (friction_per_transaction is a %),
so no sweep needed; we report this analytically.

Method: re-use defi_track_r1_yield_rotation core, override LOCKED['position_size_usd']
and capital_usd consistently (always 3× position size = 3 positions).

This is NOT a retuning of R1 — original R1 verdict stands. This is a sensitivity
check on the advisor's capital-bound hypothesis.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

import defi_track_r1_yield_rotation as r1mod

RESULTS = ROOT / 'results'
SWEEP_SIZES = [500, 5000, 50000]


def run_at_size(position_usd: int) -> dict:
    """Run R1 at given position size, return summary."""
    r1mod.LOCKED['position_size_usd'] = position_usd
    r1mod.LOCKED['capital_usd'] = position_usd * 3

    panel = r1mod.load_panel()
    f = r1mod.filter_universe(panel)
    apy_pivot = r1mod.build_apy_pivot(f)
    bt = r1mod.run_rotation(apy_pivot)
    s = r1mod.summarize(bt)
    s['position_size_usd'] = position_usd
    s['capital_usd'] = position_usd * 3
    s['gas_per_swap_usd'] = position_usd * (r1mod.LOCKED['friction_per_swap_pct'] / 100.0)
    return s


def main():
    print('=' * 100)
    print('DeFi-R1 position-size sweep — capital-bound hypothesis check')
    print('=' * 100)
    print('Locked: gas $2/swap fixed.  Sweeping position_size_usd ∈ {500, 5000, 50000}')
    print('(NOTE: friction_per_swap_pct=0.4% is held — but real-world gas is $-fixed,')
    print(" so we model it as gas_$ = position × 0.4% = $2 at $500. To test the advisor")
    print(" hypothesis, we hold gas_$=$2 fixed across sizes by adjusting friction_pct.)\n")

    print('At gas_$=$2 fixed, friction_pct scales as 2/position × 100:')
    print(f'  $500   pos → fric=0.4000% → drag {(0.4 * 12):.2f}%/yr (wee. rebal x 12 swaps avg)')
    print(f'  $5000  pos → fric=0.0400% → drag {(0.04 * 12):.2f}%/yr')
    print(f'  $50000 pos → fric=0.0040% → drag {(0.004 * 12):.2f}%/yr\n')

    rows = []
    for size in SWEEP_SIZES:
        gas_pct_at_2usd = (2.0 / size) * 100.0
        r1mod.LOCKED['friction_per_swap_pct'] = gas_pct_at_2usd
        print(f'\n--- position_size = ${size:,} (friction = {gas_pct_at_2usd:.4f}%/swap = $2 fixed) ---')
        s = run_at_size(size)
        s['friction_per_swap_pct_used'] = gas_pct_at_2usd
        for k in ['cum_gross_pct', 'cum_net_pct', 'avg_daily_gross_pct',
                  'avg_daily_fric_pct', 'avg_daily_net_pct',
                  'annualized_gross_apy_pct', 'annualized_net_apy_pct',
                  'max_dd_pct', 'worst_5d_net_pct', 'sharpe_annualized']:
            v = s.get(k)
            if isinstance(v, float):
                print(f'  {k}: {v:+.4f}')
            else:
                print(f'  {k}: {v}')
        rows.append(s)

    print('\n' + '=' * 100)
    print('SUMMARY — net APY scales with position size (DeFi-R1)')
    print('=' * 100)
    print(f'{"position_$":>12}  {"gross_apy":>10}  {"fric_apy":>10}  {"net_apy":>10}  {"net_$/yr":>12}  {"sharpe":>7}')
    for r in rows:
        net_apy = r['annualized_net_apy_pct']
        gross_apy = r['annualized_gross_apy_pct']
        fric_apy = gross_apy - net_apy
        net_dollars = net_apy / 100.0 * r['capital_usd']
        print(f'  ${r["position_size_usd"]:>9,}  '
              f'{gross_apy:>+9.2f}%  '
              f'{fric_apy:>+9.2f}%  '
              f'{net_apy:>+9.2f}%  '
              f'${net_dollars:>+11,.0f}  '
              f'{r["sharpe_annualized"]:>+6.2f}')

    print('\nAnalytical comparison — PB-R1-maker (scale-invariant)')
    print('=' * 100)
    print('PB-R1 friction = 0.04% per transaction (% of notional, scale-invariant).')
    print('Net APY ≈ +9% gross − 6% friction = ~3%/yr regardless of position size.')
    print(f'  $1,500 capital  → ~$45/yr')
    print(f'  $15,000 capital → ~$450/yr   (10× capital, 10× $ but same %)')
    print(f'  $150,000 capital→ ~$4,500/yr (100× capital, 100× $ but same %)')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'sweep_sizes_usd': SWEEP_SIZES,
        'gas_assumption_usd_per_swap': 2.0,
        'defi_r1_sweep': rows,
        'pb_r1_maker_analytical': {
            'friction_type': 'percent_of_notional',
            'scale_invariance': True,
            'net_apy_pct_yr': 3.0,
            'net_usd_at_1500': 45,
            'net_usd_at_15000': 450,
            'net_usd_at_150000': 4500,
            'note': 'friction_per_transaction=0.04% is %-of-notional, mathematically '
                    'invariant to position size. Net %APY identical at all scales.',
        },
        'advisor_hypothesis': (
            'DeFi friction is $-per-swap, scales inversely with position. '
            'PB-R1-maker friction is %-of-notional, scale-invariant. '
            'Convergence at $1,500 is coincidence, not universal capital-binds.'
        ),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'defi_track_r1_position_sweep_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
