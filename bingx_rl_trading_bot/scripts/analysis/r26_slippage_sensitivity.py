"""R26 Slippage Sensitivity — Taker Market Exit Realism.

User confirmation (2026-04-30): Entry/SL market (taker, slippage), TP limit (maker, no slippage).
R26 already conforms: grid fills + TP cycles = maker (no slippage).
Only forced trend exits = market (taker, slippage applies).

Test: re-run R26 with slippage = {0, 0.02, 0.05, 0.10, 0.20}% applied to each
forced exit. Quantify cum_net / daily / BS_pos degradation.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

DATA_FILE = DATA / 'btc_1h_720days.csv'

LOCKED = {
    'capital_usd': 1500,
    'grid_spacing_pct': 0.30,
    'grid_levels_each_side': 5,
    'per_level_usd': 150,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'trend_exit_distance_pct': 1.5,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
}

SLIPPAGE_SCENARIOS = [0.0, 0.02, 0.05, 0.10, 0.20]  # % per market exit


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
    atr = compute_atr(df, LOCKED['atr_period'])
    atr_pct = atr / df['close']
    median_30d = atr_pct.rolling(LOCKED['atr_pct_median_lookback_bars'],
                                  min_periods=240).median()
    return (atr_pct < median_30d).fillna(False)


def simulate_grid(df: pd.DataFrame, slippage_pct: float) -> dict:
    """R26 simulation with explicit slippage on forced market exits."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    ts = df['timestamp'].values
    is_ranging = compute_ranging_filter(df).values

    spacing = LOCKED['grid_spacing_pct'] / 100
    levels = LOCKED['grid_levels_each_side']
    per_level_usd = LOCKED['per_level_usd']
    capital = LOCKED['capital_usd']
    maker_fric = LOCKED['maker_friction_per_side_pct'] / 100
    taker_fric = LOCKED['taker_friction_per_side_pct'] / 100
    trend_exit_dist = LOCKED['trend_exit_distance_pct'] / 100
    max_lifetime = LOCKED['max_grid_lifetime_bars']
    slip = slippage_pct / 100

    active_grid = None
    cum_harvest = 0.0
    cum_drift = 0.0
    cum_friction = 0.0
    cum_slippage = 0.0
    n_cycles = 0
    n_grid_setups = 0
    n_trend_exits = 0
    daily_returns = {}

    start = max(LOCKED['atr_period'] + 50, LOCKED['atr_pct_median_lookback_bars'])

    for i in range(start, n):
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
                # Apply slippage: adverse to position direction
                if p['side'] == 'LONG':
                    # LONG forced exit: filled lower (slippage worse)
                    exit_price = close[i] * (1 - slip)
                    pnl_pct = (exit_price - p['entry_price']) / p['entry_price'] * 100
                else:
                    # SHORT forced exit: filled higher (slippage worse)
                    exit_price = close[i] * (1 + slip)
                    pnl_pct = (p['entry_price'] - exit_price) / p['entry_price'] * 100
                fric_pct = taker_fric * 100
                slippage_cost = slip * 100
                net_pct = pnl_pct - fric_pct
                contrib = net_pct * (p['qty_usd'] / capital)
                cum_drift += contrib
                cum_friction += fric_pct * (p['qty_usd'] / capital)
                cum_slippage += slippage_cost * (p['qty_usd'] / capital)
                d = pd.to_datetime(ts[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            active_grid = None
            continue

        for k in range(levels):
            if not active_grid['buy_filled'][k] and low[i] <= active_grid['buy_levels'][k]:
                buy_price = active_grid['buy_levels'][k]
                tp_price = buy_price * (1 + spacing)
                cum_friction += maker_fric * 100 * (per_level_usd / capital)
                active_grid['open_positions'].append({
                    'side': 'LONG', 'entry_price': buy_price,
                    'tp_price': tp_price, 'qty_usd': per_level_usd, 'open_idx': i,
                })
                active_grid['buy_filled'][k] = True
            if not active_grid['sell_filled'][k] and high[i] >= active_grid['sell_levels'][k]:
                sell_price = active_grid['sell_levels'][k]
                tp_price = sell_price * (1 - spacing)
                cum_friction += maker_fric * 100 * (per_level_usd / capital)
                active_grid['open_positions'].append({
                    'side': 'SHORT', 'entry_price': sell_price,
                    'tp_price': tp_price, 'qty_usd': per_level_usd, 'open_idx': i,
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
                net_pct = pnl_pct - fric_pct
                contrib = net_pct * (p['qty_usd'] / capital)
                cum_harvest += contrib
                cum_friction += fric_pct * (p['qty_usd'] / capital)
                n_cycles += 1
                d = pd.to_datetime(ts[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            else:
                new_open.append(p)
        active_grid['open_positions'] = new_open

    cum_net = cum_harvest + cum_drift
    n_days = (n - start) / 24
    daily_pct = cum_net / n_days

    # Bootstrap
    daily_arr = np.array(list(daily_returns.values()))
    if len(daily_arr) > 4:
        random.seed(42)
        n_iter = min(1000, len(daily_arr) - 3)
        starts = random.sample(range(len(daily_arr) - 3), n_iter)
        cums = [daily_arr[s:s + 3].sum() for s in starts]
        bs_pos = float((np.array(cums) > 0).mean())
    else:
        bs_pos = 0

    return {
        'slippage_pct': slippage_pct,
        'cum_net_pct': cum_net,
        'cum_harvest_pct': cum_harvest,
        'cum_drift_pct': cum_drift,
        'cum_friction_pct': cum_friction,
        'cum_slippage_pct': cum_slippage,
        'daily_pct': daily_pct,
        'n_cycles': n_cycles,
        'n_trend_exits': n_trend_exits,
        'bs_pos_rate': bs_pos,
    }


def main():
    print('=' * 100)
    print('R26 Slippage Sensitivity — Taker Market Exit Realism')
    print('=' * 100)
    print('User: Entry/SL market (taker, slippage), TP limit (maker, no slippage)')
    print('R26 conforms: grid fills + TP cycles = maker. Forced exits = taker market.\n')

    df = load_data()
    n_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars, {n_days:.1f} days\n')

    print('Running R26 across slippage scenarios...\n')
    results = []
    for slip_pct in SLIPPAGE_SCENARIOS:
        r = simulate_grid(df, slip_pct)
        results.append(r)
        print(f'Slippage {slip_pct:.2f}%:')
        print(f'  cum_net: {r["cum_net_pct"]:+8.4f}%, daily: {r["daily_pct"]:+.4f}%')
        print(f'  harvest: {r["cum_harvest_pct"]:+7.2f}%, drift: {r["cum_drift_pct"]:+7.2f}%, '
              f'friction: {r["cum_friction_pct"]:+6.2f}%, slippage: {r["cum_slippage_pct"]:+6.2f}%')
        print(f'  cycles: {r["n_cycles"]}, trend_exits: {r["n_trend_exits"]}, '
              f'BS_pos: {r["bs_pos_rate"]:.4f}\n')

    print('=' * 100)
    print('Summary table — R26 across slippage scenarios:')
    print(f'{"Slip%":>6} {"cum_net%":>10} {"daily%":>10} {"BS_pos":>8} '
          f'{"vs 0% slip":>12} {"target met":>12}')
    print('-' * 80)
    for r in results:
        delta = r['daily_pct'] - results[0]['daily_pct']
        target = 'PASS' if r['daily_pct'] >= 0.20 else 'FAIL'
        print(f'{r["slippage_pct"]:>6.2f} {r["cum_net_pct"]:>+9.4f}% '
              f'{r["daily_pct"]:>+9.4f}% {r["bs_pos_rate"]:>8.4f} '
              f'{delta:>+11.4f}% {target:>12}')
    print('=' * 100)

    # Robustness verdict
    daily_at_5bps = next((r['daily_pct'] for r in results if r['slippage_pct'] == 0.05), None)
    daily_at_10bps = next((r['daily_pct'] for r in results if r['slippage_pct'] == 0.10), None)
    daily_at_20bps = next((r['daily_pct'] for r in results if r['slippage_pct'] == 0.20), None)

    print()
    print('Slippage robustness verdict:')
    print(f'  At 0.05% slippage (typical retail): daily {daily_at_5bps:+.4f}% '
          f'({"still positive" if daily_at_5bps > 0 else "turns negative"})')
    print(f'  At 0.10% slippage (volatile periods): daily {daily_at_10bps:+.4f}% '
          f'({"still positive" if daily_at_10bps > 0 else "turns negative"})')
    print(f'  At 0.20% slippage (extreme): daily {daily_at_20bps:+.4f}% '
          f'({"still positive" if daily_at_20bps > 0 else "turns negative"})')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'locked': LOCKED,
        'slippage_scenarios': SLIPPAGE_SCENARIOS,
        'results': results,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'r26_slippage_sensitivity_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
