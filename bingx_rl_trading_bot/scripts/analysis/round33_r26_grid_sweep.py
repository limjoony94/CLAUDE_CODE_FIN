"""Round 33 — R26 Parameter Sweep with Train/Test Split.

Pre-reg: claudedocs/round33_r26_grid_sweep_prereg.md (commit 840b160)

3×3×3 = 27 configs on R26 baseline.
Train (60%) winner selection → Test (40%) OOS validation.
"""
import json
import random
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

DATA_FILE = DATA / 'btc_1h_720days.csv'

GRID = {
    'grid_spacing_pct': [0.20, 0.30, 0.50],
    'grid_levels_each_side': [3, 5, 7],
    'trend_exit_distance_pct': [1.0, 1.5, 2.5],
}

FIXED = {
    'asset': 'BTC/USDT',
    'capital_usd': 1500,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
    'train_test_split': 0.60,
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_ranging_filter(df: pd.DataFrame) -> pd.Series:
    atr = compute_atr(df, FIXED['atr_period'])
    atr_pct = atr / df['close']
    median_30d = atr_pct.rolling(FIXED['atr_pct_median_lookback_bars'],
                                  min_periods=240).median()
    return (atr_pct < median_30d).fillna(False)


def simulate_grid(df: pd.DataFrame, spacing_pct: float, levels: int,
                   trend_exit_pct: float, start_idx: int = 0,
                   end_idx: int = None) -> dict:
    """Vectorized R26 grid simulation for given config and slice."""
    n = len(df)
    if end_idx is None:
        end_idx = n
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    is_ranging = compute_ranging_filter(df).values

    spacing = spacing_pct / 100
    per_level_usd = FIXED['capital_usd'] / (2 * levels)
    capital = FIXED['capital_usd']
    maker_fric = FIXED['maker_friction_per_side_pct'] / 100
    taker_fric = FIXED['taker_friction_per_side_pct'] / 100
    trend_exit_dist = trend_exit_pct / 100
    max_lifetime = FIXED['max_grid_lifetime_bars']

    active_grid = None
    cum_harvest = 0.0
    cum_drift = 0.0
    cum_friction = 0.0
    n_cycles = 0
    n_grid_setups = 0
    n_trend_exits = 0
    daily_returns = {}  # date → cum net for the day
    timestamps = df['timestamp'].values
    n_bars_processed = 0

    start = max(FIXED['atr_period'] + 50, FIXED['atr_pct_median_lookback_bars'])
    sim_start = max(start, start_idx)
    for i in range(sim_start, end_idx):
        n_bars_processed += 1

        if active_grid is None and is_ranging[i]:
            init_mid = close[i]
            buy_levels = [init_mid * (1 - spacing * (k + 1)) for k in range(levels)]
            sell_levels = [init_mid * (1 + spacing * (k + 1)) for k in range(levels)]
            active_grid = {
                'init_mid': init_mid, 'init_idx': i,
                'buy_levels': buy_levels, 'sell_levels': sell_levels,
                'buy_filled': [False] * levels, 'sell_filled': [False] * levels,
                'open_positions': [],
            }
            n_grid_setups += 1

        if active_grid is None:
            continue

        elapsed = i - active_grid['init_idx']
        price_dist = abs(close[i] - active_grid['init_mid']) / active_grid['init_mid']
        force_exit = False
        if elapsed >= max_lifetime:
            force_exit = True
        elif price_dist > trend_exit_dist and not is_ranging[i]:
            force_exit = True
            n_trend_exits += 1

        if force_exit:
            for p in active_grid['open_positions']:
                if p['side'] == 'LONG':
                    pnl_pct = (close[i] - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl_pct = (p['entry_price'] - close[i]) / p['entry_price'] * 100
                fric_pct = taker_fric * 100
                net_pct = pnl_pct - fric_pct
                contrib = net_pct * (p['qty_usd'] / capital)
                cum_drift += contrib
                cum_friction += fric_pct * (p['qty_usd'] / capital)
                # Daily aggregate
                d = pd.to_datetime(timestamps[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            active_grid = None
            continue

        # New fills
        for k in range(levels):
            if not active_grid['buy_filled'][k] and low[i] <= active_grid['buy_levels'][k]:
                buy_price = active_grid['buy_levels'][k]
                tp_price = buy_price * (1 + spacing)
                cum_friction += maker_fric * 100 * (per_level_usd / capital)
                active_grid['open_positions'].append({
                    'side': 'LONG', 'entry_price': buy_price,
                    'tp_price': tp_price, 'qty_usd': per_level_usd,
                    'open_idx': i,
                })
                active_grid['buy_filled'][k] = True
            if not active_grid['sell_filled'][k] and high[i] >= active_grid['sell_levels'][k]:
                sell_price = active_grid['sell_levels'][k]
                tp_price = sell_price * (1 - spacing)
                cum_friction += maker_fric * 100 * (per_level_usd / capital)
                active_grid['open_positions'].append({
                    'side': 'SHORT', 'entry_price': sell_price,
                    'tp_price': tp_price, 'qty_usd': per_level_usd,
                    'open_idx': i,
                })
                active_grid['sell_filled'][k] = True

        # TP fills
        new_open = []
        for p in active_grid['open_positions']:
            tp_hit = False
            if p['side'] == 'LONG' and high[i] >= p['tp_price']:
                tp_hit = True
                exit_price = p['tp_price']
            elif p['side'] == 'SHORT' and low[i] <= p['tp_price']:
                tp_hit = True
                exit_price = p['tp_price']
            if tp_hit:
                if p['side'] == 'LONG':
                    pnl_pct = (exit_price - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl_pct = (p['entry_price'] - exit_price) / p['entry_price'] * 100
                fric_pct = maker_fric * 100
                net_pct = pnl_pct - fric_pct
                contrib = net_pct * (p['qty_usd'] / capital)
                cum_harvest += contrib
                cum_friction += fric_pct * (p['qty_usd'] / capital)
                n_cycles += 1
                d = pd.to_datetime(timestamps[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            else:
                new_open.append(p)
        active_grid['open_positions'] = new_open

    cum_net = cum_harvest + cum_drift
    n_days = n_bars_processed / 24
    daily_pct = cum_net / n_days if n_days > 0 else 0

    # Bootstrap
    daily_arr = np.array(list(daily_returns.values()))
    if len(daily_arr) > 3:
        random.seed(42)
        n_iter = min(1000, max(100, len(daily_arr) - 3))
        starts = [random.randint(0, len(daily_arr) - 3 - 1) for _ in range(n_iter)] if len(daily_arr) > 4 else []
        if starts:
            cums = [daily_arr[s:s + 3].sum() for s in starts]
            arr = np.array(cums)
            bs_pos = float((arr > 0).mean())
        else:
            bs_pos = 0
    else:
        bs_pos = 0

    return {
        'cum_net_pct': cum_net,
        'cum_harvest_pct': cum_harvest,
        'cum_drift_pct': cum_drift,
        'cum_friction_pct': cum_friction,
        'daily_pct': daily_pct,
        'n_cycles': n_cycles,
        'n_grid_setups': n_grid_setups,
        'n_trend_exits': n_trend_exits,
        'n_days': n_days,
        'bs_pos_rate': bs_pos,
    }


def main():
    print('=' * 100)
    print('Round 33 — R26 Parameter Sweep (3×3×3 = 27 configs) with Train/Test Split')
    print('=' * 100)
    print('Pre-reg: claudedocs/round33_r26_grid_sweep_prereg.md (840b160)\n')

    df = load_data()
    n = len(df)
    n_days_full = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    train_end_idx = int(n * FIXED['train_test_split'])
    train_days = (df['timestamp'].iloc[train_end_idx] -
                   df['timestamp'].iloc[0]).total_seconds() / 86400
    test_days = n_days_full - train_days
    print(f'Total: {n_days_full:.1f}d | Train: 0-{train_end_idx} ({train_days:.1f}d) | '
          f'Test: {train_end_idx}-{n} ({test_days:.1f}d)\n')

    # TRAIN: run all 27 configs
    print('Running 27 configs on TRAIN...')
    train_results = []
    config_idx = 0
    for spacing, levels, te_dist in product(
        GRID['grid_spacing_pct'],
        GRID['grid_levels_each_side'],
        GRID['trend_exit_distance_pct']
    ):
        config_idx += 1
        r = simulate_grid(df, spacing, levels, te_dist,
                           start_idx=0, end_idx=train_end_idx)
        train_results.append({
            'config': {'spacing': spacing, 'levels': levels, 'te_dist': te_dist},
            **r,
        })
        print(f'  [{config_idx}/27] spacing={spacing}, levels={levels}, '
              f'te={te_dist}: train_daily {r["daily_pct"]:+.4f}%, '
              f'cycles {r["n_cycles"]}, BS_pos {r["bs_pos_rate"]:.4f}')

    # Sort by train daily descending
    train_results.sort(key=lambda r: r['daily_pct'], reverse=True)

    print('\n=== TRAIN results (sorted by daily_pct) ===')
    print(f'{"#":>3} {"spc":>5} {"lev":>4} {"te":>5} {"cycles":>7} '
          f'{"trades":>7} {"daily%":>10} {"BS_pos":>8}')
    print('-' * 80)
    for idx, r in enumerate(train_results):
        c = r['config']
        is_baseline = (c['spacing'] == 0.30 and c['levels'] == 5 and c['te_dist'] == 1.5)
        marker = ' *' if is_baseline else '  '
        print(f'{idx+1:>3} {c["spacing"]:>5} {c["levels"]:>4} {c["te_dist"]:>5} '
              f'{r["n_cycles"]:>7} {r["n_trend_exits"]:>7} '
              f'{r["daily_pct"]:>+9.4f}% {r["bs_pos_rate"]:>8.4f}{marker}')
    print('  * = R26 baseline\n')

    # WINNER
    winner = train_results[0]
    print(f'=== TRAIN WINNER ===')
    print(f'  config: spacing={winner["config"]["spacing"]}%, '
          f'levels={winner["config"]["levels"]}+{winner["config"]["levels"]}, '
          f'trend_exit={winner["config"]["te_dist"]}%')
    print(f'  train daily: {winner["daily_pct"]:+.4f}%, '
          f'cycles: {winner["n_cycles"]}, '
          f'BS_pos: {winner["bs_pos_rate"]:.4f}\n')

    # TEST
    print('Running WINNER on TEST...')
    test_r = simulate_grid(df, winner['config']['spacing'],
                            winner['config']['levels'],
                            winner['config']['te_dist'],
                            start_idx=train_end_idx, end_idx=n)
    print(f'  test daily: {test_r["daily_pct"]:+.4f}%, '
          f'cycles: {test_r["n_cycles"]}, '
          f'BS_pos: {test_r["bs_pos_rate"]:.4f}\n')

    print('=== Train vs Test Overfit Check ===')
    train_d = winner['daily_pct']
    test_d = test_r['daily_pct']
    print(f'  train daily: {train_d:+.4f}%')
    print(f'  test daily:  {test_d:+.4f}%')
    if train_d > 0:
        retention = test_d / train_d
        print(f'  test/train ratio: {retention:.4f}')
    else:
        retention = 0

    if test_d >= 0.20 and (train_d == 0 or test_d / train_d > 0.5):
        verdict = 'GENUINE_SIGNAL_TARGET_MET'
    elif test_d >= 0.10:
        verdict = 'POSITIVE_SUB_TARGET'
    elif test_d >= 0.03:
        verdict = 'PARETO_BOUNDARY_R26_BASELINE'
    elif test_d >= 0:
        verdict = 'BORDERLINE_NEAR_ZERO'
    else:
        verdict = 'CATASTROPHIC_OVERFIT'

    print(f'  → VERDICT: {verdict}')
    print()

    # R26 baseline test for comparison
    print('Running R26 BASELINE on TEST for comparison...')
    baseline_test = simulate_grid(df, 0.30, 5, 1.5, start_idx=train_end_idx, end_idx=n)
    print(f'  baseline test daily: {baseline_test["daily_pct"]:+.4f}%, '
          f'cycles: {baseline_test["n_cycles"]}\n')

    print('=' * 100)
    print(f'WINNER test daily: {test_d:+.4f}%')
    print(f'BASELINE test daily: {baseline_test["daily_pct"]:+.4f}%')
    print(f'Improvement over baseline: {test_d - baseline_test["daily_pct"]:+.4f}%')
    print(f'User target: 0.20%/day')
    print(f'Verdict: {verdict}')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '840b160',
        'grid': GRID, 'fixed': FIXED,
        'train_days': train_days, 'test_days': test_days,
        'train_results_sorted': [
            {**r, 'config': r['config']} for r in train_results
        ],
        'winner_config': winner['config'],
        'winner_train_daily': train_d,
        'winner_test_daily': test_d,
        'winner_test_full': test_r,
        'baseline_test_daily': baseline_test['daily_pct'],
        'baseline_test_full': baseline_test,
        'verdict': verdict,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round33_r26_grid_sweep_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
