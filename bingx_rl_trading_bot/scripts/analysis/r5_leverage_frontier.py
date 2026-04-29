"""R5 + Leverage Frontier — Ruin Probability Characterization.

Pre-reg: claudedocs/r5_leverage_frontier_prereg.md (commit 14748b1)

Mechanism:
  R5 single-coin BTC cash-and-carry at leverage L ∈ {1, 2, 3, 5, 7, 10, 15, 20, 30}.
  Bootstrap 1000 × 365-day paths from BingX funding (800d) + spot/perp daily basis (250d).
  Liquidation rule: intraday |basis_swing| > (1 - 0.005) / L → ruin.

Output: per-leverage net APY mean, ruin probability per year, adjusted yield,
verdict against (T4 daily ≥ 0.20% AND ruin_prob ≤ 1%/year).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

FUNDING_FILE = DATA / 'funding_history.parquet'
SPOT_FILE = DATA / 'btc_spot_daily_bingx.parquet'
PERP_FILE = DATA / 'btc_perp_daily_bingx.parquet'

LOCKED = {
    'capital_usd': 1500,
    'leverage_levels': [1, 2, 3, 5, 7, 10, 15, 20, 30],
    'maintenance_margin_pct': 0.50,
    'spot_friction_per_side_pct': 0.10,
    'perp_friction_per_side_pct': 0.04,
    'entry_threshold_apy_pct': 3.0,
    'exit_threshold_apy_pct': 0.0,
    'lookback_funding_days': 7,
    'sim_paths': 1000,
    'sim_path_length_days': 365,
    'ruin_threshold_per_year': 0.01,
    't4_daily_threshold_pct': 0.20,
    'baseline_1x_apy_pct': 3.28,
}


def load_data():
    fund = pd.read_parquet(FUNDING_FILE)
    btc_fund = fund[fund['symbol'] == 'BTC/USDT'].copy()
    btc_fund['datetime'] = pd.to_datetime(btc_fund['datetime'])
    btc_fund['date'] = btc_fund['datetime'].dt.tz_localize(None).dt.floor('D')
    daily_fund = btc_fund.groupby('date')['funding_rate'].sum().sort_index()

    spot = pd.read_parquet(SPOT_FILE)
    perp = pd.read_parquet(PERP_FILE)
    spot['date'] = pd.to_datetime(spot['date'])
    perp['date'] = pd.to_datetime(perp['date'])
    m = pd.merge(
        spot[['date', 'high', 'low', 'close']].rename(
            columns={'high': 'h_s', 'low': 'l_s', 'close': 'c_s'}),
        perp[['date', 'high', 'low', 'close']].rename(
            columns={'high': 'h_p', 'low': 'l_p', 'close': 'c_p'}),
        on='date'
    )
    m['daily_basis_pct'] = (m['c_p'] - m['c_s']) / m['c_s'] * 100
    # Intraday max-min basis swing (worst case for liquidation)
    m['intraday_max_basis'] = (m['h_p'] - m['l_s']) / m['c_s'] * 100
    m['intraday_min_basis'] = (m['l_p'] - m['h_s']) / m['c_s'] * 100
    m['intraday_swing'] = m['intraday_max_basis'] - m['intraday_min_basis']

    # Merge funding with basis on overlap dates
    df = pd.DataFrame({'date': daily_fund.index, 'funding_daily': daily_fund.values})
    out = pd.merge(df, m[['date', 'daily_basis_pct', 'intraday_swing']],
                   on='date', how='inner')
    return out


def simulate_path(df: pd.DataFrame, L: float, rng: np.random.Generator,
                  start_idx: int, length: int) -> dict:
    """Simulate single path of R5 at leverage L for `length` days starting at start_idx.

    Returns: {final_capital_pct, ruin_event, n_entries, n_exits, daily_returns}
    """
    cap = 1.0  # normalized to 1.0 = $1500
    in_pos = False
    daily_returns = []
    n_entries = 0
    n_exits = 0
    n_days = 0
    ruin = False

    spot_fric_per_side = LOCKED['spot_friction_per_side_pct'] / 100
    perp_fric_per_side = LOCKED['perp_friction_per_side_pct'] / 100
    liquidation_threshold_pct = (1.0 - LOCKED['maintenance_margin_pct'] / 100) / L * 100

    for offset in range(length):
        i = start_idx + offset
        if i >= len(df):
            break
        row = df.iloc[i]
        n_days += 1
        funding_today = row['funding_daily']
        basis_swing = row['intraday_swing']

        # 7-day trailing APY for regime gate
        if i < LOCKED['lookback_funding_days']:
            rolling_apy = np.nan
        else:
            rolling_apy = (df['funding_daily']
                           .iloc[i - LOCKED['lookback_funding_days']:i]
                           .mean() * 365 * 100)

        daily_pct = 0.0
        if not in_pos:
            if pd.notna(rolling_apy) and rolling_apy >= LOCKED['entry_threshold_apy_pct']:
                in_pos = True
                n_entries += 1
                # Friction at entry: both legs × L
                # Spot: 0.10% × notional / capital × L = 0.10% × L of capital (one side)
                fric = (spot_fric_per_side + perp_fric_per_side) * L * 100
                daily_pct -= fric
        else:
            # Funding accrues on perp short × L
            funding_pnl = funding_today * L * 100  # in pct of capital
            # Basis drift: assume hedged so 0 expected, but liquidation if swing too big
            if basis_swing > liquidation_threshold_pct:
                ruin = True
                cap = 0.0
                daily_returns.append(-100.0)
                break
            daily_pct += funding_pnl
            if pd.notna(rolling_apy) and rolling_apy <= LOCKED['exit_threshold_apy_pct']:
                in_pos = False
                n_exits += 1
                fric = (spot_fric_per_side + perp_fric_per_side) * L * 100
                daily_pct -= fric

        cap *= (1 + daily_pct / 100)
        daily_returns.append(daily_pct)

    return {
        'final_cap_pct': float((cap - 1.0) * 100),
        'ruin': ruin,
        'n_entries': n_entries,
        'n_exits': n_exits,
        'n_days': n_days,
        'mean_daily_pct': float(np.mean(daily_returns)) if daily_returns else 0.0,
        'std_daily_pct': float(np.std(daily_returns)) if len(daily_returns) > 1 else 0.0,
    }


def run_leverage(df: pd.DataFrame, L: float, rng: np.random.Generator) -> dict:
    n_data = len(df)
    path_len = LOCKED['sim_path_length_days']
    if n_data < path_len:
        return {'leverage': L, 'error': 'insufficient data'}

    n_paths = LOCKED['sim_paths']
    starts = rng.integers(0, n_data - path_len, size=n_paths)
    sims = [simulate_path(df, L, rng, int(s), path_len) for s in starts]

    final_caps = np.array([s['final_cap_pct'] for s in sims])
    ruined = np.array([s['ruin'] for s in sims])
    mean_dailies = np.array([s['mean_daily_pct'] for s in sims])

    ruin_prob = float(ruined.mean())
    # Survivor mean (excluding ruined paths)
    survivor_mask = ~ruined
    if survivor_mask.sum() > 0:
        survivor_mean_apy = float(final_caps[survivor_mask].mean())  # 365-day cum %
        survivor_daily = float(mean_dailies[survivor_mask].mean())
    else:
        survivor_mean_apy = -100.0
        survivor_daily = -100.0

    # Adjusted yield: E(yield) × P(survive)
    expected_apy = float(final_caps.mean())  # includes ruined as -100
    expected_daily = float(mean_dailies.mean())
    adjusted_yield = expected_apy

    return {
        'leverage': L,
        'n_paths': n_paths,
        'ruin_prob': ruin_prob,
        'survivor_apy_pct': survivor_mean_apy,
        'survivor_daily_pct': survivor_daily,
        'expected_apy_pct': expected_apy,
        'expected_daily_pct': expected_daily,
        'adjusted_yield_pct': adjusted_yield,
        'final_cap_p5': float(np.percentile(final_caps, 5)),
        'final_cap_p50': float(np.percentile(final_caps, 50)),
        'final_cap_p95': float(np.percentile(final_caps, 95)),
        'liquidation_threshold_pct': (1.0 - LOCKED['maintenance_margin_pct'] / 100) / L * 100,
    }


def evaluate(result: dict) -> str:
    daily = result['expected_daily_pct']
    ruin = result['ruin_prob']
    if daily >= LOCKED['t4_daily_threshold_pct'] and ruin <= LOCKED['ruin_threshold_per_year']:
        return 'DEPLOYABLE'
    elif 0.05 <= daily < LOCKED['t4_daily_threshold_pct'] and ruin <= LOCKED['ruin_threshold_per_year']:
        return 'SUB_DEPLOYABLE'
    elif ruin > LOCKED['ruin_threshold_per_year']:
        return 'RUIN_BOUND'
    else:
        return 'YIELD_INSUFFICIENT'


def main():
    print('=' * 100)
    print('R5 + Leverage Frontier — Ruin Probability Characterization')
    print('=' * 100)
    print('Pre-reg: claudedocs/r5_leverage_frontier_prereg.md (14748b1)')
    print(f'Locked: {LOCKED}\n')

    df = load_data()
    print(f'Data: {len(df)} days with funding + basis overlap')
    print(f'Date range: {df["date"].min().date()} → {df["date"].max().date()}')
    print(f'Funding daily mean: {df["funding_daily"].mean()*100:.4f}% '
          f'(={df["funding_daily"].mean()*365*100:.2f}%/yr)')
    print(f'Daily basis pct: mean {df["daily_basis_pct"].mean():+.3f}%, '
          f'std {df["daily_basis_pct"].std():.3f}%, min {df["daily_basis_pct"].min():.3f}%, '
          f'max {df["daily_basis_pct"].max():.3f}%')
    print(f'Intraday swing: mean {df["intraday_swing"].mean():.2f}%, '
          f'std {df["intraday_swing"].std():.2f}%, max {df["intraday_swing"].max():.2f}%')
    print(f'Intraday swing percentiles: p50={df["intraday_swing"].quantile(0.5):.2f}, '
          f'p95={df["intraday_swing"].quantile(0.95):.2f}, '
          f'p99={df["intraday_swing"].quantile(0.99):.2f}\n')

    rng = np.random.default_rng(42)
    results = []
    for L in LOCKED['leverage_levels']:
        print(f'Running L={L}× ...')
        r = run_leverage(df, L, rng)
        r['verdict'] = evaluate(r)
        results.append(r)

    print()
    print('=' * 100)
    print(f'{"L":>4} {"mean_daily%":>13} {"ruin_prob":>11} {"E[apy]%":>10} '
          f'{"survivor_apy%":>15} {"liq_thresh%":>13} {"verdict":>20}')
    print('-' * 100)
    for r in results:
        print(f'{r["leverage"]:>4}× {r["expected_daily_pct"]:>+12.4f} '
              f'{r["ruin_prob"]:>11.4f} {r["expected_apy_pct"]:>+9.2f} '
              f'{r["survivor_apy_pct"]:>+14.2f} {r["liquidation_threshold_pct"]:>12.2f} '
              f'{r["verdict"]:>20}')
    print('=' * 100)

    deployable = [r for r in results if r['verdict'] == 'DEPLOYABLE']
    sub_deployable = [r for r in results if r['verdict'] == 'SUB_DEPLOYABLE']
    print()
    if deployable:
        best = max(deployable, key=lambda r: r['expected_daily_pct'])
        print(f'BEST DEPLOYABLE: L={best["leverage"]}× → daily {best["expected_daily_pct"]:.4f}%, '
              f'ruin {best["ruin_prob"]:.4f}/yr, APY {best["expected_apy_pct"]:.2f}%')
    elif sub_deployable:
        best = max(sub_deployable, key=lambda r: r['expected_daily_pct'])
        print(f'BEST SUB-DEPLOYABLE: L={best["leverage"]}× → daily {best["expected_daily_pct"]:.4f}%, '
              f'ruin {best["ruin_prob"]:.4f}/yr (below 0.20% target)')
    else:
        print('NO DEPLOYABLE LEVERAGE — yield-insufficient at safe levels, '
              'ruin-bound at high levels.')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '14748b1',
        'locked': LOCKED,
        'data_summary': {
            'n_days': len(df),
            'date_min': str(df['date'].min().date()),
            'date_max': str(df['date'].max().date()),
            'funding_apy_pct': float(df['funding_daily'].mean() * 365 * 100),
            'intraday_swing_p99': float(df['intraday_swing'].quantile(0.99)),
            'intraday_swing_max': float(df['intraday_swing'].max()),
        },
        'frontier': results,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'r5_leverage_frontier_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
