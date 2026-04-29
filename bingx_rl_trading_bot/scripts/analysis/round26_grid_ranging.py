"""Round 26 — Grid Trading on ATR-Based Ranging Regime.

Pre-reg: claudedocs/round26_grid_ranging_prereg.md (commit bd9a233)

Mechanism:
  Range filter: ATR(20)/close < 30d trailing median → ranging
  Grid: 5 buys + 5 sells at +/- 0.30% × k from init_mid
  Per-level: $150 ($1500 / 10 levels)
  TP cycle: BUY fill → place SELL limit at +0.30% (maker)
            SELL fill → place BUY limit at -0.30% (maker)
  Trend exit: |close - init_mid| > 1.5% AND ranging=False → close all (taker)

LOCKED. NO TUNING.
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
RESULTS.mkdir(exist_ok=True)

DATA_FILE = DATA / 'btc_1h_720days.csv'

LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'capital_usd': 1500,
    'grid_spacing_pct': 0.30,
    'grid_levels_each_side': 5,
    'per_level_usd': 150,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,  # 30d × 24h
    'trend_exit_distance_pct': 1.5,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
}

GATES = {
    'gate_A_min_ranging_fraction': 0.30,
    'gate_B_random_p95_required': True,
    'c1_daily_pct_min': 0.20,
    'c2_per_trade_gross_min': 0.07,
    'c3_min_trades': 100,
    'c4_bs_window_days': 3,
    'c4_bs_n_iter': 1000,
    'c4_min_pos_rate': 0.50,
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
    """Returns boolean Series: True if ATR(20)/close < 30d rolling median."""
    atr = compute_atr(df, LOCKED['atr_period'])
    atr_pct = atr / df['close']
    median_30d = atr_pct.rolling(LOCKED['atr_pct_median_lookback_bars'],
                                  min_periods=240).median()
    return (atr_pct < median_30d).fillna(False)


def simulate_grid(df: pd.DataFrame) -> dict:
    """Walk through bars, set up grids during ranging, simulate fills."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
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

    # Grid state
    active_grid = None  # dict with 'init_mid', 'init_idx', 'buy_levels', 'sell_levels',
                       # 'open_positions' (list of dicts with 'side', 'price', 'tp_price', 'qty_usd')

    trades = []  # closed cycles
    drift_exits = []  # forced trend-exit events

    cum_harvest_pct = 0.0  # sum of profitable cycle nets
    cum_friction_pct = 0.0  # all friction
    cum_drift_pct = 0.0  # losses from forced exits

    n_grid_setups = 0
    n_full_cycles = 0
    n_trend_exits = 0
    n_max_lifetime_resets = 0
    ranging_count = 0

    start = max(LOCKED['atr_period'] + 50, LOCKED['atr_pct_median_lookback_bars'])

    for i in range(start, n):
        if is_ranging[i]:
            ranging_count += 1

        # --- Grid setup ---
        if active_grid is None and is_ranging[i]:
            # Initialize grid
            init_mid = close[i]
            buy_levels = [init_mid * (1 - spacing * (k + 1)) for k in range(levels)]
            sell_levels = [init_mid * (1 + spacing * (k + 1)) for k in range(levels)]
            active_grid = {
                'init_mid': init_mid,
                'init_idx': i,
                'buy_levels': buy_levels,    # prices to buy at
                'sell_levels': sell_levels,  # prices to sell at
                'buy_filled': [False] * levels,
                'sell_filled': [False] * levels,
                'open_positions': [],  # each: {'side', 'entry_price', 'tp_price', 'qty_usd', 'open_idx'}
            }
            n_grid_setups += 1

        if active_grid is None:
            continue

        # --- Check trend exit ---
        elapsed = i - active_grid['init_idx']
        price_dist_from_mid = abs(close[i] - active_grid['init_mid']) / active_grid['init_mid']
        force_exit = False
        if elapsed >= max_lifetime:
            force_exit = True
            exit_reason = 'MAX_LIFETIME'
            n_max_lifetime_resets += 1
        elif price_dist_from_mid > trend_exit_dist and not is_ranging[i]:
            force_exit = True
            exit_reason = 'TREND_EXIT'
            n_trend_exits += 1

        if force_exit:
            # Close all open positions at current market (taker)
            net_dist = 0.0
            for p in active_grid['open_positions']:
                if p['side'] == 'LONG':
                    pnl_pct = (close[i] - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl_pct = (p['entry_price'] - close[i]) / p['entry_price'] * 100
                # Friction: entry already paid maker. Now exit taker.
                fric_pct = taker_fric * 100
                net_pct = pnl_pct - fric_pct
                # Scale to capital: position is qty_usd / capital
                contrib_to_capital = net_pct * (p['qty_usd'] / capital)
                cum_drift_pct += contrib_to_capital  # could be neg or pos
                cum_friction_pct += fric_pct * (p['qty_usd'] / capital)
                drift_exits.append({
                    'ts': ts[i], 'side': p['side'], 'reason': exit_reason,
                    'entry_price': p['entry_price'], 'exit_price': close[i],
                    'gross_pct': pnl_pct, 'net_pct': net_pct,
                    'contrib_pct': contrib_to_capital,
                    'qty_usd': p['qty_usd'],
                })
                trades.append({
                    'open_ts': ts[p['open_idx']], 'close_ts': ts[i],
                    'side': p['side'], 'entry_price': p['entry_price'],
                    'exit_price': close[i], 'gross_pct': pnl_pct,
                    'net_pct': net_pct, 'contrib_pct': contrib_to_capital,
                    'reason': exit_reason, 'qty_usd': p['qty_usd'],
                })
            active_grid = None
            continue

        # --- Process intrabar fills ---
        # Order: check buy fills, sell fills (independent)
        # We approximate intrabar by checking if low touches buy levels and high touches sell levels
        # For each fill, place TP order at +/-0.30% from fill price (maker)

        for k in range(levels):
            # Buy level k
            if not active_grid['buy_filled'][k] and low[i] <= active_grid['buy_levels'][k]:
                buy_price = active_grid['buy_levels'][k]
                tp_price = buy_price * (1 + spacing)
                # Pay maker friction at fill
                cum_friction_pct += maker_fric * 100 * (per_level_usd / capital)
                active_grid['open_positions'].append({
                    'side': 'LONG', 'entry_price': buy_price,
                    'tp_price': tp_price, 'qty_usd': per_level_usd,
                    'open_idx': i,
                })
                active_grid['buy_filled'][k] = True

            # Sell level k
            if not active_grid['sell_filled'][k] and high[i] >= active_grid['sell_levels'][k]:
                sell_price = active_grid['sell_levels'][k]
                tp_price = sell_price * (1 - spacing)
                cum_friction_pct += maker_fric * 100 * (per_level_usd / capital)
                active_grid['open_positions'].append({
                    'side': 'SHORT', 'entry_price': sell_price,
                    'tp_price': tp_price, 'qty_usd': per_level_usd,
                    'open_idx': i,
                })
                active_grid['sell_filled'][k] = True

        # --- Process TP fills on open positions ---
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
                # Maker friction at TP exit
                fric_pct = maker_fric * 100
                net_pct = pnl_pct - fric_pct
                contrib_to_capital = net_pct * (p['qty_usd'] / capital)
                cum_harvest_pct += contrib_to_capital
                cum_friction_pct += fric_pct * (p['qty_usd'] / capital)
                n_full_cycles += 1
                trades.append({
                    'open_ts': ts[p['open_idx']], 'close_ts': ts[i],
                    'side': p['side'], 'entry_price': p['entry_price'],
                    'exit_price': exit_price, 'gross_pct': pnl_pct,
                    'net_pct': net_pct, 'contrib_pct': contrib_to_capital,
                    'reason': 'TP_CYCLE', 'qty_usd': p['qty_usd'],
                })
            else:
                new_open.append(p)
        active_grid['open_positions'] = new_open

    cum_net_pct = cum_harvest_pct + cum_drift_pct  # cum_friction is already netted in each
    ranging_fraction = ranging_count / (n - start)

    return {
        'n_bars_processed': n - start,
        'ranging_fraction': float(ranging_fraction),
        'n_grid_setups': n_grid_setups,
        'n_full_cycles': n_full_cycles,
        'n_trend_exits': n_trend_exits,
        'n_max_lifetime_resets': n_max_lifetime_resets,
        'cum_harvest_pct': float(cum_harvest_pct),
        'cum_drift_pct': float(cum_drift_pct),
        'cum_friction_pct': float(cum_friction_pct),
        'cum_net_pct': float(cum_net_pct),
        'trades': pd.DataFrame(trades),
        'drift_exits': drift_exits,
    }


def summarize(result: dict, n_days: float) -> dict:
    trades = result['trades']
    n = len(trades)
    cum_net = result['cum_net_pct']
    daily_pct = cum_net / n_days

    if n > 0:
        avg_gross_per_trade = float(trades['gross_pct'].mean())
        avg_net_per_trade = float(trades['net_pct'].mean())
        wr = float((trades['net_pct'] > 0).mean())
        trades_per_day = n / n_days
        # Worst 5d aggregated by close date
        trades_copy = trades.copy()
        trades_copy['close_date'] = pd.to_datetime(trades_copy['close_ts']).dt.floor('D')
        # contrib aggregated as actual capital % per day
        daily_contrib = trades_copy.groupby('close_date')['contrib_pct'].sum()
        if len(daily_contrib) >= 5:
            worst_5d = float(daily_contrib.rolling(5).sum().min())
        else:
            worst_5d = float(daily_contrib.min())
    else:
        avg_gross_per_trade = 0
        avg_net_per_trade = 0
        wr = 0
        trades_per_day = 0
        worst_5d = 0

    mean_drift_per_event = (
        result['cum_drift_pct'] / result['n_trend_exits']
        if result['n_trend_exits'] > 0 else 0
    )

    return {
        'n_trades': n,
        'cum_net_pct': cum_net,
        'cum_harvest_pct': result['cum_harvest_pct'],
        'cum_drift_pct': result['cum_drift_pct'],
        'cum_friction_pct': result['cum_friction_pct'],
        'avg_gross_per_trade_pct': avg_gross_per_trade,
        'avg_net_per_trade_pct': avg_net_per_trade,
        'wr': wr,
        'daily_pct': daily_pct,
        'trades_per_day': trades_per_day,
        'worst_5d_pct': worst_5d,
        'n_grid_setups': result['n_grid_setups'],
        'n_full_cycles': result['n_full_cycles'],
        'n_trend_exits': result['n_trend_exits'],
        'mean_drift_per_event_pct': mean_drift_per_event,
        'ranging_fraction': result['ranging_fraction'],
    }


def main():
    print('=' * 100)
    print('Round 26 — Grid Trading on ATR-Based Ranging Regime')
    print('=' * 100)
    print('Pre-reg: claudedocs/round26_grid_ranging_prereg.md (bd9a233)')
    print(f'Locked: spacing {LOCKED["grid_spacing_pct"]}%, '
          f'{LOCKED["grid_levels_each_side"]}+{LOCKED["grid_levels_each_side"]} levels, '
          f'${LOCKED["per_level_usd"]}/level')
    print(f'Friction: maker {LOCKED["maker_friction_per_side_pct"]}%/side, '
          f'taker {LOCKED["taker_friction_per_side_pct"]}%/side')
    print(f'Trend exit: > {LOCKED["trend_exit_distance_pct"]}% from init mid AND '
          f'ranging=False\n')

    df = load_data()
    n_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars, {n_days:.1f} days\n')

    print('Simulating grid...')
    result = simulate_grid(df)
    summ = summarize(result, n_days)

    print('=== Decomposition ===')
    print(f'  ranging fraction: {summ["ranging_fraction"]:.4f}')
    print(f'  grid setups: {summ["n_grid_setups"]}')
    print(f'  full cycles (TP fills): {summ["n_full_cycles"]}')
    print(f'  trend exits: {summ["n_trend_exits"]}')
    print(f'  cum harvest: {summ["cum_harvest_pct"]:+.4f}%')
    print(f'  cum drift drawdown: {summ["cum_drift_pct"]:+.4f}%')
    print(f'  cum friction: {summ["cum_friction_pct"]:+.4f}%')
    print(f'  cum net: {summ["cum_net_pct"]:+.4f}%')
    print(f'  mean drift per trend exit: {summ["mean_drift_per_event_pct"]:+.4f}%\n')

    print('=== Trade summary ===')
    for k, v in summ.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    # Gate A
    gA_pass = summ['ranging_fraction'] >= GATES['gate_A_min_ranging_fraction']
    print(f'=== Gate A — Ranging fraction ≥ 30% ===')
    print(f'  ranging frac: {summ["ranging_fraction"]:.4f}  '
          f'→ {"PASS" if gA_pass else "FAIL"}\n')

    daily = summ['daily_pct']
    pt_gross = summ['avg_gross_per_trade_pct']
    n_trades = summ['n_trades']
    c1_pass = daily >= GATES['c1_daily_pct_min']
    c2_pass = pt_gross >= GATES['c2_per_trade_gross_min']
    c3_pass = n_trades >= GATES['c3_min_trades']

    print(f'=== C1 (HARD) Daily ≥ 0.20% ===')
    print(f'  daily: {daily:+.4f}%  → {"PASS" if c1_pass else "FAIL"}\n')
    print(f'=== C2 Per-trade gross > 0.07% ===')
    print(f'  per-trade gross: {pt_gross:+.4f}%  → {"PASS" if c2_pass else "FAIL"}\n')
    print(f'=== C3 Trade count ≥ 100 ===')
    print(f'  n_trades: {n_trades}  → {"PASS" if c3_pass else "FAIL"}\n')

    # C4 bootstrap
    if n_trades >= 5:
        trades_copy = result['trades'].copy()
        trades_copy['close_date'] = pd.to_datetime(trades_copy['close_ts']).dt.floor('D')
        daily_contrib = trades_copy.groupby('close_date')['contrib_pct'].sum()
        daily_contrib = daily_contrib.reindex(
            pd.date_range(daily_contrib.index.min(), daily_contrib.index.max(), freq='D'),
            fill_value=0
        )
        nets = daily_contrib.values
        n = len(nets)
        win = GATES['c4_bs_window_days']
        if n > win:
            random.seed(42)
            starts = random.sample(range(n - win), min(GATES['c4_bs_n_iter'], n - win))
            cums = [nets[s:s + win].sum() for s in starts]
            arr = np.array(cums)
            pos_rate = float((arr > 0).mean())
            c4_pass = pos_rate >= GATES['c4_min_pos_rate']
        else:
            pos_rate = 0
            c4_pass = False
    else:
        pos_rate = 0
        c4_pass = False
    print(f'=== C4 Bootstrap 1000 × 3-day ≥ 50% pos_rate ===')
    print(f'  pos_rate: {pos_rate:.4f}  → {"PASS" if c4_pass else "FAIL"}\n')

    n_pass = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    print('=' * 100)
    print(f'VERDICT: {n_pass}/4 user criteria PASS')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'bd9a233',
        'locked': LOCKED, 'gates': GATES,
        'summary': summ,
        'gate_A': {'ranging_fraction': summ['ranging_fraction'], 'pass': bool(gA_pass)},
        'c1_daily': {'daily_pct': daily, 'pass': bool(c1_pass)},
        'c2_per_trade_gross': {'gross_pct': pt_gross, 'pass': bool(c2_pass)},
        'c3_trade_count': {'n_trades': n_trades, 'pass': bool(c3_pass)},
        'c4_bootstrap': {'pos_rate': pos_rate, 'pass': bool(c4_pass)},
        'verdict_pass': n_pass,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round26_grid_ranging_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
