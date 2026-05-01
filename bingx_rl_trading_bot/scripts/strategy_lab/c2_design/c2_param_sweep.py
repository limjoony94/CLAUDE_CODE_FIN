"""C2 parameter sweep — answers user push back: "극과적합 모델 도출 가능 vs OOS robust 0".

Sweep grid:
  z_entry:     [1.5, 2.0, 2.5, 3.0]
  z_exit:      [0.3, 0.5, 0.7]
  max_hold:    [10, 21, 42, 84] periods
  z_lookback:  [15, 30, 60] days

= 4 × 3 × 4 × 3 = 144 configs.

For each config:
  - In-sample full-period BT
  - Save: cum_net_pct, n_trades, wr, apy

Output:
  - Configs sorted by APY (cherry-pick best)
  - Visible: "if in-sample positive exists" → answer to user push back
"""
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse first-pass logic
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_first_pass_bt import load_funding, compute_zscores


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'results'


def simulate_config(df_z: pd.DataFrame, z_entry: float, z_exit: float,
                     max_hold: int, capital: float = 1000,
                     fric_per_side: float = 0.06) -> dict:
    """Simulate with given (z_entry, z_exit, max_hold)."""
    pivot_z = df_z.pivot(index='timestamp', columns='symbol', values='zscore')
    pivot_f = df_z.pivot(index='timestamp', columns='symbol', values='fund_pct')
    timestamps = pivot_z.index.sort_values()
    leg_notional = capital * 1.0 / 4

    active = None
    trades = []
    cum_funding = cum_friction = 0.0

    for i, ts in enumerate(timestamps):
        z_row = pivot_z.loc[ts]
        f_row = pivot_f.loc[ts]
        if z_row.isna().all():
            continue
        valid = z_row.dropna()
        if len(valid) < 4:
            continue

        if active is None:
            mz = valid.max(); mn = valid.min()
            if mz > z_entry and mn < -z_entry:
                short_c = valid.idxmax(); long_c = valid.idxmin()
                if short_c != long_c:
                    cum_friction += 4 * fric_per_side / 100 * leg_notional
                    active = {
                        'long': long_c, 'short': short_c,
                        'enter_idx': i, 'enter_ts': ts,
                        'cum_f': 0.0,
                    }

        if active is not None:
            f_long = f_row.get(active['long'], np.nan)
            f_short = f_row.get(active['short'], np.nan)
            if not np.isnan(f_long) and not np.isnan(f_short):
                active['cum_f'] += (-f_long + f_short) / 100 * leg_notional

            held = i - active['enter_idx']
            cz_long = z_row.get(active['long'], np.nan)
            cz_short = z_row.get(active['short'], np.nan)
            should_exit = False
            if held >= max_hold:
                should_exit = True
            elif (not np.isnan(cz_long) and not np.isnan(cz_short)
                  and abs(cz_long) < z_exit and abs(cz_short) < z_exit):
                should_exit = True

            if should_exit:
                cum_friction += 4 * fric_per_side / 100 * leg_notional
                cum_funding += active['cum_f']
                trades.append({
                    'enter_ts': str(active['enter_ts']), 'exit_ts': str(ts),
                    'long': active['long'], 'short': active['short'],
                    'periods': held, 'gross_funding': active['cum_f'],
                })
                active = None

    cum_net = cum_funding - cum_friction
    return {
        'cum_funding_usd': cum_funding,
        'cum_friction_usd': cum_friction,
        'cum_net_usd': cum_net,
        'cum_net_pct': cum_net / capital * 100,
        'n_trades': len(trades),
        'trades_per_30d': len(trades) / ((pd.to_datetime(timestamps.max()) - pd.to_datetime(timestamps.min())).total_seconds() / 86400 / 30) if len(timestamps) > 1 else 0,
    }


def main():
    print('=' * 100)
    print('C2 Parameter Sweep — answer to user push back ("극과적합 도출 가능?")')
    print('=' * 100)

    df = load_funding()
    span_days = (df.timestamp.max() - df.timestamp.min()).total_seconds() / 86400
    print(f'Data span: {span_days:.1f} days')

    grid = list(product(
        [1.5, 2.0, 2.5, 3.0],     # z_entry
        [0.3, 0.5, 0.7],          # z_exit
        [10, 21, 42, 84],         # max_hold
        [15, 30, 60],             # z_lookback
    ))
    print(f'Configs: {len(grid)}')
    print()

    # Pre-compute z-scores per lookback (reuse)
    z_cache = {}
    for lb in {g[3] for g in grid}:
        z_cache[lb] = compute_zscores(df, lb)

    results = []
    for i, (ze, zx, mh, lb) in enumerate(grid):
        df_z = z_cache[lb]
        if zx >= ze:
            continue   # nonsensical
        r = simulate_config(df_z, ze, zx, mh)
        r.update({'z_entry': ze, 'z_exit': zx, 'max_hold': mh, 'z_lookback': lb})
        # APY extrapolation
        r['apy_pct'] = r['cum_net_pct'] / span_days * 365 if span_days > 0 else 0
        results.append(r)
        if (i+1) % 30 == 0:
            print(f'  {i+1}/{len(grid)}...')

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('apy_pct', ascending=False).reset_index(drop=True)

    print()
    print('=' * 100)
    print('TOP 10 (in-sample best — cherry-picked)')
    print('=' * 100)
    print(df_res[['z_entry', 'z_exit', 'max_hold', 'z_lookback',
                   'n_trades', 'cum_net_pct', 'apy_pct']].head(10).to_string(index=False))

    print()
    print('=' * 100)
    print('Distribution stats')
    print('=' * 100)
    print(f'  Configs with positive APY:  {(df_res["apy_pct"] > 0).sum():>3}/{len(df_res)}')
    print(f'  Configs APY > 1%:           {(df_res["apy_pct"] > 1).sum():>3}/{len(df_res)}')
    print(f'  Configs APY > 5%:           {(df_res["apy_pct"] > 5).sum():>3}/{len(df_res)}')
    print(f'  Configs APY > 10%:          {(df_res["apy_pct"] > 10).sum():>3}/{len(df_res)}')
    print(f'  Best APY:                   {df_res["apy_pct"].max():+.4f}%')
    print(f'  Median APY:                 {df_res["apy_pct"].median():+.4f}%')
    print(f'  Worst APY:                  {df_res["apy_pct"].min():+.4f}%')
    print()
    print('Answer to user push back:')
    n_pos = (df_res["apy_pct"] > 0).sum()
    if n_pos > 0:
        print(f'  ✅ {n_pos} configs with in-sample APY > 0 (cherry-pick possible)')
        print(f'  → "극과적합 모델 도출 가능" 정량 증명')
    else:
        print(f'  ❌ 0 configs with in-sample APY > 0 even after sweep')
        print(f'  → friction floor structurally binds C2 mechanism')

    out_path = RESULTS / f'c2_param_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    df_res.to_json(out_path, orient='records', indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
