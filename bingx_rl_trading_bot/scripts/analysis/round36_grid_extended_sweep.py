"""Round 36 — Grid spacing × levels EXTENDED sweep (30 configs).

Pre-reg: claudedocs/round36_grid_extended_sweep_prereg.md (commit 9147da4)
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
    'grid_spacing_pct': [0.10, 0.15, 0.20, 0.30, 0.50, 0.80],
    'grid_levels_each_side': [3, 5, 7, 10, 15],
}

FIXED = {
    'capital_usd': 1500,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'trend_exit_distance_pct': 1.5,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
    'train_test_split': 0.60,
}

CRITERIA = {
    'stability_gate_bs_pos_min': 0.85,
    'switch_test_bs_pos_min': 0.85,
    'switch_daily_improvement_pct': 0.02,
    'switch_retention_min': 0.60,
    'wf_min_pos_folds': 4,
    'wf_min_bs_pos_per_fold': 0.80,
    'wf_total_folds': 5,
}

BASELINE = {'spacing': 0.30, 'levels': 5}


def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_atr(df, period):
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_ranging_filter(df):
    atr = compute_atr(df, FIXED['atr_period'])
    atr_pct = atr / df['close']
    median = atr_pct.rolling(FIXED['atr_pct_median_lookback_bars'], min_periods=240).median()
    return (atr_pct < median).fillna(False)


def simulate_grid(df, spacing_pct, levels, start_idx=0, end_idx=None):
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
    trend_exit_dist = FIXED['trend_exit_distance_pct'] / 100
    max_lifetime = FIXED['max_grid_lifetime_bars']

    active_grid = None
    cum_harvest = 0.0
    cum_drift = 0.0
    n_cycles = 0
    n_grid_setups = 0
    n_trend_exits = 0
    daily_returns = {}
    timestamps = df['timestamp'].values

    start = max(FIXED['atr_period'] + 50, FIXED['atr_pct_median_lookback_bars'])
    sim_start = max(start, start_idx)

    for i in range(sim_start, end_idx):
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
                contrib = (pnl_pct - fric_pct) * (per_level_usd / capital)
                cum_drift += contrib
                d = pd.to_datetime(timestamps[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            active_grid = None
            continue

        for k in range(levels):
            if not active_grid['buy_filled'][k] and low[i] <= active_grid['buy_levels'][k]:
                buy_price = active_grid['buy_levels'][k]
                tp_price = buy_price * (1 + spacing)
                active_grid['open_positions'].append({
                    'side': 'LONG', 'entry_price': buy_price, 'tp_price': tp_price,
                    'open_idx': i,
                })
                active_grid['buy_filled'][k] = True
            if not active_grid['sell_filled'][k] and high[i] >= active_grid['sell_levels'][k]:
                sell_price = active_grid['sell_levels'][k]
                tp_price = sell_price * (1 - spacing)
                active_grid['open_positions'].append({
                    'side': 'SHORT', 'entry_price': sell_price, 'tp_price': tp_price,
                    'open_idx': i,
                })
                active_grid['sell_filled'][k] = True

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
                contrib = (pnl_pct - fric_pct) * (per_level_usd / capital)
                cum_harvest += contrib
                n_cycles += 1
                d = pd.to_datetime(timestamps[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            else:
                new_open.append(p)
        active_grid['open_positions'] = new_open

    cum_net = cum_harvest + cum_drift
    n_days = (end_idx - sim_start) / 24
    daily_pct = cum_net / n_days if n_days > 0 else 0

    daily_arr = np.array(list(daily_returns.values()))
    if len(daily_arr) > 4:
        random.seed(42)
        n_iter = min(1000, len(daily_arr) - 3)
        starts = random.sample(range(len(daily_arr) - 3), n_iter)
        cums = [daily_arr[s:s + 3].sum() for s in starts]
        bs_pos = float((np.array(cums) > 0).mean())
    else:
        bs_pos = 0

    grid_extent_pct = spacing_pct * levels  # max grid extent (one side)
    return {
        'cum_net_pct': cum_net,
        'cum_harvest_pct': cum_harvest,
        'cum_drift_pct': cum_drift,
        'daily_pct': daily_pct,
        'n_cycles': n_cycles,
        'n_grid_setups': n_grid_setups,
        'n_trend_exits': n_trend_exits,
        'bs_pos_rate': bs_pos,
        'grid_extent_pct': grid_extent_pct,
        'extent_vs_trend_exit': 'WITHIN' if grid_extent_pct <= FIXED['trend_exit_distance_pct'] else 'BEYOND',
    }


def main():
    print('=' * 100)
    print('Round 36 — Grid spacing × levels EXTENDED Sweep (30 configs)')
    print('=' * 100)
    print('Pre-reg: 9147da4\n')

    df = load_data()
    n = len(df)
    train_end_idx = int(n * FIXED['train_test_split'])
    train_days = (df['timestamp'].iloc[train_end_idx] -
                   df['timestamp'].iloc[0]).total_seconds() / 86400
    test_days = ((df['timestamp'].iloc[-1] - df['timestamp'].iloc[train_end_idx]).total_seconds() / 86400)
    print(f'Train: {train_days:.1f}d | Test: {test_days:.1f}d\n')

    print('Running 30 configs on TRAIN...')
    train_results = []
    for spacing, levels in product(GRID['grid_spacing_pct'], GRID['grid_levels_each_side']):
        r = simulate_grid(df, spacing, levels, 0, train_end_idx)
        train_results.append({'config': {'spacing': spacing, 'levels': levels}, **r})
        is_baseline = (spacing == BASELINE['spacing'] and levels == BASELINE['levels'])
        marker = ' *' if is_baseline else '  '
        ext_marker = '⚠' if r['extent_vs_trend_exit'] == 'BEYOND' else ' '
        print(f'  {ext_marker} spc={spacing}, lev={levels}, extent={r["grid_extent_pct"]:.2f}%: '
              f'daily={r["daily_pct"]:+.4f}%, BS_pos={r["bs_pos_rate"]:.4f}, '
              f'cycles={r["n_cycles"]}{marker}')
    print('  * = R26 baseline (0.30/5), ⚠ = grid extent > trend_exit 1.5%\n')

    stability_gated = [r for r in train_results
                        if r['bs_pos_rate'] >= CRITERIA['stability_gate_bs_pos_min']]
    print(f'Stability gate (BS_pos ≥ 0.85): {len(stability_gated)}/{len(train_results)}\n')

    if not stability_gated:
        print('NO config passed stability gate. R26 baseline retained.')
        return

    stability_gated.sort(key=lambda r: r['daily_pct'], reverse=True)
    print('Top 10 stability-gated by train daily:')
    for i, r in enumerate(stability_gated[:10]):
        c = r['config']
        is_baseline = (c['spacing'] == BASELINE['spacing'] and c['levels'] == BASELINE['levels'])
        marker = ' *' if is_baseline else '  '
        print(f'  {i+1}. spc={c["spacing"]}, lev={c["levels"]}: '
              f'daily={r["daily_pct"]:+.4f}%, BS_pos={r["bs_pos_rate"]:.4f}, '
              f'cycles={r["n_cycles"]}{marker}')
    print()

    winner = stability_gated[0]
    baseline_train = next(r for r in train_results
                           if r['config']['spacing'] == BASELINE['spacing']
                           and r['config']['levels'] == BASELINE['levels'])

    print(f'TRAIN WINNER:')
    print(f'  spc={winner["config"]["spacing"]}, lev={winner["config"]["levels"]}')
    print(f'  daily={winner["daily_pct"]:+.4f}%, BS_pos={winner["bs_pos_rate"]:.4f}\n')
    print(f'BASELINE: daily={baseline_train["daily_pct"]:+.4f}%, '
          f'BS_pos={baseline_train["bs_pos_rate"]:.4f}\n')

    winner_test = simulate_grid(df, winner['config']['spacing'],
                                  winner['config']['levels'], train_end_idx, n)
    baseline_test = simulate_grid(df, BASELINE['spacing'], BASELINE['levels'],
                                    train_end_idx, n)
    print(f'TEST validation:')
    print(f'  Winner test: daily={winner_test["daily_pct"]:+.4f}%, BS_pos={winner_test["bs_pos_rate"]:.4f}')
    print(f'  Baseline test: daily={baseline_test["daily_pct"]:+.4f}%, BS_pos={baseline_test["bs_pos_rate"]:.4f}\n')

    test_bs_ok = winner_test['bs_pos_rate'] >= CRITERIA['switch_test_bs_pos_min']
    daily_imp = winner_test['daily_pct'] - baseline_test['daily_pct']
    daily_imp_ok = daily_imp >= CRITERIA['switch_daily_improvement_pct']
    if winner['daily_pct'] > 0:
        retention = winner_test['daily_pct'] / winner['daily_pct']
    else:
        retention = 0
    retention_ok = retention >= CRITERIA['switch_retention_min']

    print(f'Switch criterion:')
    print(f'  Test BS_pos ≥ 0.85: {winner_test["bs_pos_rate"]:.4f} → {"PASS" if test_bs_ok else "FAIL"}')
    print(f'  Daily imp ≥ 0.02%: +{daily_imp:.4f}% → {"PASS" if daily_imp_ok else "FAIL"}')
    print(f'  Retention ≥ 60%: {retention:.4f} → {"PASS" if retention_ok else "FAIL"}')

    folds = CRITERIA['wf_total_folds']
    fold_size = n // (folds + 1)
    fold_results = []
    print(f'\nWF 5-fold (winner):')
    for f in range(folds):
        s = (f + 1) * fold_size
        e = min(s + fold_size, n)
        r = simulate_grid(df, winner['config']['spacing'],
                           winner['config']['levels'], s, e)
        fold_results.append(r)
        print(f'  Fold {f+1}: daily={r["daily_pct"]:+.4f}%, BS_pos={r["bs_pos_rate"]:.4f}')

    n_pos = sum(1 for r in fold_results if r['daily_pct'] > 0)
    n_bs_ok = sum(1 for r in fold_results
                    if r['bs_pos_rate'] >= CRITERIA['wf_min_bs_pos_per_fold'])
    wf_pos_ok = n_pos >= CRITERIA['wf_min_pos_folds']
    wf_bs_ok = n_bs_ok >= CRITERIA['wf_min_pos_folds']
    print(f'  Folds positive: {n_pos}/5 → {"PASS" if wf_pos_ok else "FAIL"}')
    print(f'  Folds BS_pos ≥ 0.80: {n_bs_ok}/5 → {"PASS" if wf_bs_ok else "FAIL"}\n')

    all_pass = test_bs_ok and daily_imp_ok and retention_ok and wf_pos_ok and wf_bs_ok

    print('=' * 100)
    if all_pass:
        print(f'VERDICT: SWITCH RECOMMENDED — winner spc={winner["config"]["spacing"]}, '
              f'lev={winner["config"]["levels"]}')
    else:
        print('VERDICT: KEEP R26 BASELINE')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '9147da4',
        'grid': GRID, 'fixed': FIXED, 'criteria': CRITERIA, 'baseline': BASELINE,
        'all_train_results': train_results,
        'baseline_train': baseline_train,
        'baseline_test': baseline_test,
        'winner_train': winner,
        'winner_test': winner_test,
        'wf_folds': fold_results,
        'switch_recommended': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round36_grid_extended_sweep_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
