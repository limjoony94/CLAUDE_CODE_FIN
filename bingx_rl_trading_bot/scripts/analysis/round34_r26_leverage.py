"""Round 34 — R26 + Leverage Frontier.

Pre-reg: claudedocs/round34_r26_leverage_prereg.md (commit eb1a271)

R26 grid simulation with leverage scaling + per-position liquidation check.
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
    'asset': 'BTC/USDT',
    'tf': '1h',
    'capital_usd': 1500,
    'grid_spacing_pct': 0.30,
    'grid_levels_each_side': 5,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'trend_exit_distance_pct': 1.5,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
    'maintenance_margin_pct': 0.50,
}

LEVERAGE_LEVELS = [1, 2, 3, 4, 5, 7, 10]


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


def simulate_grid_with_leverage(df: pd.DataFrame, L: float) -> dict:
    """R26 simulation with leverage scaling + per-position liquidation check."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    ts = df['timestamp'].values
    is_ranging = compute_ranging_filter(df).values

    spacing = LOCKED['grid_spacing_pct'] / 100
    levels = LOCKED['grid_levels_each_side']
    per_level_notional = 150  # at L×, position notional = $150 (margin = $150 / L per pos)
    capital = LOCKED['capital_usd']
    maker_fric = LOCKED['maker_friction_per_side_pct'] / 100
    taker_fric = LOCKED['taker_friction_per_side_pct'] / 100
    trend_exit_dist = LOCKED['trend_exit_distance_pct'] / 100
    max_lifetime = LOCKED['max_grid_lifetime_bars']
    mm_pct = LOCKED['maintenance_margin_pct']
    # Liquidation: adverse move × L > (1 - mm/100)
    # adverse_pct_threshold = (1 - mm/100) / L * 100
    liq_threshold_pct = (1 - mm_pct / 100) / L * 100 if L > 0 else 100

    # Simulation: per-position p&l × L
    # Position notional = per_level_notional × L
    # Position contribution to capital: net_pct × (notional × L) / capital
    # Note: at L×, max simultaneous exposure = 10 × per_level × L = $1500 × L

    active_grid = None
    cum_harvest = 0.0
    cum_drift = 0.0
    cum_friction = 0.0
    cum_liquidation_loss = 0.0
    n_cycles = 0
    n_grid_setups = 0
    n_trend_exits = 0
    n_liquidation_events = 0
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

        # --- Per-position liquidation check ---
        # For each open position, compute current adverse % and check if exceeds threshold
        new_open = []
        for p in active_grid['open_positions']:
            if p['side'] == 'LONG':
                # LONG: adverse move = entry - current_low (worst intraday low)
                adverse_pct = (p['entry_price'] - low[i]) / p['entry_price'] * 100
            else:
                adverse_pct = (high[i] - p['entry_price']) / p['entry_price'] * 100

            if adverse_pct > liq_threshold_pct:
                # Liquidation event
                # Loss: position notional - margin available remainder
                # In leveraged accounts: margin lost = position_notional / L (= per_level_notional fixed at $150)
                liq_loss_pct = liq_threshold_pct  # adverse % at liquidation
                loss_to_capital = liq_loss_pct * (per_level_notional * L) / capital
                cum_liquidation_loss += loss_to_capital
                cum_drift += -loss_to_capital
                n_liquidation_events += 1
                d = pd.to_datetime(ts[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) - loss_to_capital
                # Position closed at liquidation, do not add to new_open
            else:
                new_open.append(p)
        active_grid['open_positions'] = new_open

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
                # Net P&L: scale by (notional × L / capital)
                # pnl_pct is on entry price basis, position notional contribution
                contrib = (pnl_pct - fric_pct) * (per_level_notional * L) / capital
                cum_drift += contrib
                cum_friction += fric_pct * (per_level_notional * L) / capital
                d = pd.to_datetime(ts[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            active_grid = None
            continue

        for k in range(levels):
            if not active_grid['buy_filled'][k] and low[i] <= active_grid['buy_levels'][k]:
                buy_price = active_grid['buy_levels'][k]
                tp_price = buy_price * (1 + spacing)
                cum_friction += maker_fric * 100 * (per_level_notional * L) / capital
                active_grid['open_positions'].append({
                    'side': 'LONG', 'entry_price': buy_price,
                    'tp_price': tp_price,
                    'open_idx': i,
                })
                active_grid['buy_filled'][k] = True
            if not active_grid['sell_filled'][k] and high[i] >= active_grid['sell_levels'][k]:
                sell_price = active_grid['sell_levels'][k]
                tp_price = sell_price * (1 - spacing)
                cum_friction += maker_fric * 100 * (per_level_notional * L) / capital
                active_grid['open_positions'].append({
                    'side': 'SHORT', 'entry_price': sell_price,
                    'tp_price': tp_price,
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
                contrib = (pnl_pct - fric_pct) * (per_level_notional * L) / capital
                cum_harvest += contrib
                cum_friction += fric_pct * (per_level_notional * L) / capital
                n_cycles += 1
                d = pd.to_datetime(ts[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            else:
                new_open.append(p)
        active_grid['open_positions'] = new_open

    cum_net = cum_harvest + cum_drift
    n_days = (n - start) / 24
    daily_pct = cum_net / n_days if n_days > 0 else 0

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

    # Max DD
    daily_sorted = sorted(daily_returns.keys())
    nav = 1.0
    peak = 1.0
    max_dd = 0.0
    for d in daily_sorted:
        nav *= (1 + daily_returns[d] / 100)
        peak = max(peak, nav)
        dd = (peak - nav) / peak
        max_dd = max(max_dd, dd)

    return {
        'leverage': L,
        'cum_net_pct': cum_net,
        'cum_harvest_pct': cum_harvest,
        'cum_drift_pct': cum_drift,
        'cum_friction_pct': cum_friction,
        'cum_liquidation_loss_pct': cum_liquidation_loss,
        'daily_pct': daily_pct,
        'n_cycles': n_cycles,
        'n_trend_exits': n_trend_exits,
        'n_liquidation_events': n_liquidation_events,
        'bs_pos_rate': bs_pos,
        'max_dd_pct': max_dd * 100,
        'liq_threshold_pct': liq_threshold_pct,
    }


def evaluate(r: dict) -> str:
    daily = r['daily_pct']
    n_liq = r['n_liquidation_events']
    # 720d data, 1/year = 720/365 events
    liq_per_yr = n_liq / (720 / 365)

    if daily >= 0.20 and liq_per_yr <= 1:
        return 'DEPLOYABLE'
    elif 0.10 <= daily < 0.20 and liq_per_yr <= 1:
        return 'SUB_DEPLOYABLE'
    elif liq_per_yr > 1:
        return 'RUIN_BOUND'
    else:
        return 'YIELD_INSUFFICIENT'


def main():
    print('=' * 100)
    print('Round 34 — R26 + Leverage Frontier')
    print('=' * 100)
    print('Pre-reg: claudedocs/round34_r26_leverage_prereg.md (eb1a271)')
    print(f'Leverage levels: {LEVERAGE_LEVELS}')
    print(f'Maintenance margin: {LOCKED["maintenance_margin_pct"]}%\n')

    df = load_data()
    print(f'Data: {len(df):,} bars, '
          f'{(df["timestamp"].max() - df["timestamp"].min()).days:.0f} days\n')

    print('Running R26 across leverage levels...\n')
    results = []
    for L in LEVERAGE_LEVELS:
        print(f'L = {L}× ...')
        r = simulate_grid_with_leverage(df, L)
        r['verdict'] = evaluate(r)
        results.append(r)
        liq_per_yr = r['n_liquidation_events'] / (720 / 365)
        print(f'  daily: {r["daily_pct"]:+.4f}%, harvest: {r["cum_harvest_pct"]:+7.2f}%, '
              f'drift: {r["cum_drift_pct"]:+7.2f}%, friction: {r["cum_friction_pct"]:+6.2f}%')
        print(f'  liq events: {r["n_liquidation_events"]} ({liq_per_yr:.2f}/yr), '
              f'liq threshold: {r["liq_threshold_pct"]:.2f}% adverse')
        print(f'  cum_net: {r["cum_net_pct"]:+8.2f}%, max_dd: {r["max_dd_pct"]:.2f}%, '
              f'BS_pos: {r["bs_pos_rate"]:.4f}')
        print(f'  → {r["verdict"]}\n')

    print('=' * 100)
    print('Summary table:')
    print(f'{"L":>4} {"daily%":>10} {"cum_net%":>10} {"max_dd%":>9} '
          f'{"liq/yr":>8} {"BS_pos":>8} {"verdict":>20}')
    print('-' * 100)
    for r in results:
        liq_per_yr = r['n_liquidation_events'] / (720 / 365)
        print(f'{r["leverage"]:>3}× {r["daily_pct"]:>+9.4f}% {r["cum_net_pct"]:>+9.2f}% '
              f'{r["max_dd_pct"]:>8.2f}% {liq_per_yr:>8.2f} {r["bs_pos_rate"]:>8.4f} '
              f'{r["verdict"]:>20}')
    print('=' * 100)

    # Optimal leverage
    deployable = [r for r in results if r['verdict'] == 'DEPLOYABLE']
    if deployable:
        # Pick max L (max yield) within DEPLOYABLE
        optimal = max(deployable, key=lambda r: r['leverage'])
        print(f'\nOptimal leverage: L={optimal["leverage"]}×')
        print(f'  Daily: {optimal["daily_pct"]:+.4f}%')
        print(f'  Cum 720d: {optimal["cum_net_pct"]:+.2f}%')
        print(f'  Max DD: {optimal["max_dd_pct"]:.2f}%')
        print(f'  Liquidation events/yr: {optimal["n_liquidation_events"] / (720 / 365):.2f}')
    else:
        sub = [r for r in results if r['verdict'] == 'SUB_DEPLOYABLE']
        if sub:
            best = max(sub, key=lambda r: r['daily_pct'])
            print(f'\nNo DEPLOYABLE leverage. Best SUB_DEPLOYABLE:')
            print(f'  L={best["leverage"]}×: daily {best["daily_pct"]:+.4f}% '
                  f'(target 0.20%, {0.20 - best["daily_pct"]:+.4f}% gap)')
        else:
            print('\nNo viable leverage in tested range.')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'eb1a271',
        'locked': LOCKED,
        'leverage_levels': LEVERAGE_LEVELS,
        'results': results,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round34_r26_leverage_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
