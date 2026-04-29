"""R26 Behavioral Analysis — FOMO/panic vs Market-Making Pattern Detection.

User question (2026-04-30): "R26 거래 방식에 대한 추가 분석 - 개미의 입장에서 거래를
진행하지는 않는지? 즉, 올랐을 때 FOMO로 사고 내렸을 때 패닉셀 하지는 않는지?"

Re-run R26 with detailed trade logging.
For each trade: entry context (prior 1h/4h momentum), exit context (distance from
init_mid), categorize as:
  - Anti-FOMO SHORT: SELL when momentum was UP (selling into rallies)
  - Anti-panic LONG: BUY when momentum was DOWN (buying into dips)
  - FOMO LONG: BUY when momentum was UP (chasing rallies)
  - Panic SHORT: SELL when momentum was DOWN (selling into drops)

Trend exits: distribution of distance from init_mid at forced exit time.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

# Reuse R26 logic with detailed logging
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
    median_30d = atr_pct.rolling(LOCKED['atr_pct_median_lookback_bars'], min_periods=240).median()
    return (atr_pct < median_30d).fillna(False)


def simulate_with_logging(df: pd.DataFrame) -> tuple:
    """Re-run R26 logic with per-trade behavioral metadata."""
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

    active_grid = None
    cycle_trades = []  # successful TP cycles
    forced_trades = []  # forced trend exits

    start = max(LOCKED['atr_period'] + 50, LOCKED['atr_pct_median_lookback_bars'])

    for i in range(start, n):
        # --- Grid setup ---
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

        if active_grid is None:
            continue

        elapsed = i - active_grid['init_idx']
        price_dist_from_mid = (close[i] - active_grid['init_mid']) / active_grid['init_mid']
        force_exit = False
        if elapsed >= max_lifetime:
            force_exit = True
            exit_reason = 'MAX_LIFETIME'
        elif abs(price_dist_from_mid) > trend_exit_dist and not is_ranging[i]:
            force_exit = True
            exit_reason = 'TREND_EXIT'

        if force_exit:
            for p in active_grid['open_positions']:
                if p['side'] == 'LONG':
                    pnl_pct = (close[i] - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl_pct = (p['entry_price'] - close[i]) / p['entry_price'] * 100
                fric_pct = taker_fric * 100
                net_pct = pnl_pct - fric_pct

                # Behavioral context: at entry, what was prior 1h/4h momentum?
                entry_idx = p['open_idx']
                if entry_idx >= 4:
                    mom_1h = (close[entry_idx] - close[entry_idx - 1]) / close[entry_idx - 1] * 100
                    mom_4h = (close[entry_idx] - close[entry_idx - 4]) / close[entry_idx - 4] * 100
                else:
                    mom_1h = 0
                    mom_4h = 0

                forced_trades.append({
                    'entry_ts': ts[p['open_idx']], 'exit_ts': ts[i],
                    'side': p['side'], 'entry_price': p['entry_price'],
                    'exit_price': close[i], 'init_mid': active_grid['init_mid'],
                    'distance_at_exit_pct': price_dist_from_mid * 100,
                    'gross_pct': pnl_pct, 'net_pct': net_pct,
                    'reason': exit_reason,
                    'mom_1h_pct': mom_1h, 'mom_4h_pct': mom_4h,
                })
            active_grid = None
            continue

        # Process new fills
        for k in range(levels):
            if not active_grid['buy_filled'][k] and low[i] <= active_grid['buy_levels'][k]:
                buy_price = active_grid['buy_levels'][k]
                tp_price = buy_price * (1 + spacing)
                # Entry context (prior momentum)
                if i >= 4:
                    mom_1h = (close[i] - close[i - 1]) / close[i - 1] * 100
                    mom_4h = (close[i] - close[i - 4]) / close[i - 4] * 100
                else:
                    mom_1h = 0
                    mom_4h = 0
                active_grid['open_positions'].append({
                    'side': 'LONG', 'entry_price': buy_price,
                    'tp_price': tp_price, 'qty_usd': per_level_usd,
                    'open_idx': i, 'mom_1h_pct': mom_1h, 'mom_4h_pct': mom_4h,
                })
                active_grid['buy_filled'][k] = True

            if not active_grid['sell_filled'][k] and high[i] >= active_grid['sell_levels'][k]:
                sell_price = active_grid['sell_levels'][k]
                tp_price = sell_price * (1 - spacing)
                if i >= 4:
                    mom_1h = (close[i] - close[i - 1]) / close[i - 1] * 100
                    mom_4h = (close[i] - close[i - 4]) / close[i - 4] * 100
                else:
                    mom_1h = 0
                    mom_4h = 0
                active_grid['open_positions'].append({
                    'side': 'SHORT', 'entry_price': sell_price,
                    'tp_price': tp_price, 'qty_usd': per_level_usd,
                    'open_idx': i, 'mom_1h_pct': mom_1h, 'mom_4h_pct': mom_4h,
                })
                active_grid['sell_filled'][k] = True

        # Process TP fills
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
                cycle_trades.append({
                    'entry_ts': ts[p['open_idx']], 'exit_ts': ts[i],
                    'side': p['side'], 'entry_price': p['entry_price'],
                    'exit_price': exit_price, 'gross_pct': pnl_pct, 'net_pct': net_pct,
                    'reason': 'TP_CYCLE',
                    'mom_1h_pct': p['mom_1h_pct'], 'mom_4h_pct': p['mom_4h_pct'],
                })
            else:
                new_open.append(p)
        active_grid['open_positions'] = new_open

    return pd.DataFrame(cycle_trades), pd.DataFrame(forced_trades)


def behavioral_analysis(cycles: pd.DataFrame, forced: pd.DataFrame) -> dict:
    """Categorize trades by entry context."""
    # Cycles
    n_cycles = len(cycles)
    cycles['is_long'] = cycles['side'] == 'LONG'
    cycles['mom_up_4h'] = cycles['mom_4h_pct'] > 0
    cycles['mom_down_4h'] = cycles['mom_4h_pct'] < 0

    # 4-quadrant categorization
    long_into_dip = ((cycles['side'] == 'LONG') & (cycles['mom_4h_pct'] < 0)).sum()
    long_into_rally = ((cycles['side'] == 'LONG') & (cycles['mom_4h_pct'] > 0)).sum()
    short_into_rally = ((cycles['side'] == 'SHORT') & (cycles['mom_4h_pct'] > 0)).sum()
    short_into_dip = ((cycles['side'] == 'SHORT') & (cycles['mom_4h_pct'] < 0)).sum()

    # Anti-FOMO / anti-panic = SHORT into rally + LONG into dip = market making behavior
    market_making = short_into_rally + long_into_dip
    fomo_panic = long_into_rally + short_into_dip

    # PnL by category
    def pnl_for_category(filter_):
        s = cycles.loc[filter_, 'net_pct']
        return {'count': len(s), 'mean': float(s.mean()) if len(s) > 0 else 0,
                'sum': float(s.sum()) if len(s) > 0 else 0}

    cat_long_dip = pnl_for_category((cycles['side'] == 'LONG') & (cycles['mom_4h_pct'] < 0))
    cat_long_rally = pnl_for_category((cycles['side'] == 'LONG') & (cycles['mom_4h_pct'] > 0))
    cat_short_rally = pnl_for_category((cycles['side'] == 'SHORT') & (cycles['mom_4h_pct'] > 0))
    cat_short_dip = pnl_for_category((cycles['side'] == 'SHORT') & (cycles['mom_4h_pct'] < 0))

    # Forced exits (panic-equivalent?)
    n_forced = len(forced)
    if n_forced > 0:
        avg_forced_dist = float(forced['distance_at_exit_pct'].mean())
        avg_forced_pnl = float(forced['net_pct'].mean())
        # Panic patterns: side == LONG and exit at -%, side == SHORT and exit at +%
        panic_long = ((forced['side'] == 'LONG') &
                      (forced['distance_at_exit_pct'] < 0)).sum()
        panic_short = ((forced['side'] == 'SHORT') &
                       (forced['distance_at_exit_pct'] > 0)).sum()
    else:
        avg_forced_dist = 0
        avg_forced_pnl = 0
        panic_long = 0
        panic_short = 0

    return {
        'total_cycle_trades': int(n_cycles),
        'total_forced_trades': int(n_forced),
        'category_distribution': {
            'long_into_dip (anti-panic)': long_into_dip,
            'short_into_rally (anti-FOMO)': short_into_rally,
            'long_into_rally (FOMO chase)': long_into_rally,
            'short_into_dip (panic chase)': short_into_dip,
        },
        'market_making_count': int(market_making),
        'fomo_panic_count': int(fomo_panic),
        'market_making_pct': float(market_making / n_cycles * 100) if n_cycles > 0 else 0,
        'fomo_panic_pct': float(fomo_panic / n_cycles * 100) if n_cycles > 0 else 0,
        'pnl_by_category': {
            'long_into_dip': cat_long_dip,
            'long_into_rally': cat_long_rally,
            'short_into_rally': cat_short_rally,
            'short_into_dip': cat_short_dip,
        },
        'forced_exits': {
            'count': n_forced,
            'avg_distance_pct': avg_forced_dist,
            'avg_net_pnl_pct': avg_forced_pnl,
            'panic_long_count': int(panic_long),
            'panic_short_count': int(panic_short),
        },
    }


def main():
    print('=' * 100)
    print('R26 Behavioral Analysis — FOMO/Panic vs Market-Making Pattern')
    print('=' * 100)
    print('User question: "개미의 입장에서 거래를 진행하지는 않는지?"')
    print('Method: Re-run R26 with per-trade momentum context logging\n')

    df = load_data()
    print(f'Data: {len(df):,} bars (1h), '
          f'{(df["timestamp"].max() - df["timestamp"].min()).days:.0f} days\n')

    print('Re-running R26 with logging...')
    cycles, forced = simulate_with_logging(df)
    print(f'  Cycle TP trades: {len(cycles)}')
    print(f'  Forced exits: {len(forced)}\n')

    analysis = behavioral_analysis(cycles, forced)

    print('=== Trade Categorization (by 4h prior momentum at entry) ===')
    print(f'  Total TP cycles: {analysis["total_cycle_trades"]}')
    cat = analysis['category_distribution']
    for k, v in cat.items():
        print(f'    {k}: {v} ({v / analysis["total_cycle_trades"] * 100:.2f}%)')
    print()

    print('=== Behavior Pattern Summary ===')
    print(f'  Market-making (anti-FOMO + anti-panic): '
          f'{analysis["market_making_count"]} ({analysis["market_making_pct"]:.2f}%)')
    print(f'  FOMO/panic chase: '
          f'{analysis["fomo_panic_count"]} ({analysis["fomo_panic_pct"]:.2f}%)')
    print()

    print('=== PnL by Behavior Category (mean net %) ===')
    pbc = analysis['pnl_by_category']
    print(f'  LONG into DIP (anti-panic):    n={pbc["long_into_dip"]["count"]}, '
          f'mean={pbc["long_into_dip"]["mean"]:+.4f}%, sum={pbc["long_into_dip"]["sum"]:+.2f}%')
    print(f'  SHORT into RALLY (anti-FOMO):  n={pbc["short_into_rally"]["count"]}, '
          f'mean={pbc["short_into_rally"]["mean"]:+.4f}%, sum={pbc["short_into_rally"]["sum"]:+.2f}%')
    print(f'  LONG into RALLY (FOMO chase): n={pbc["long_into_rally"]["count"]}, '
          f'mean={pbc["long_into_rally"]["mean"]:+.4f}%, sum={pbc["long_into_rally"]["sum"]:+.2f}%')
    print(f'  SHORT into DIP (panic chase): n={pbc["short_into_dip"]["count"]}, '
          f'mean={pbc["short_into_dip"]["mean"]:+.4f}%, sum={pbc["short_into_dip"]["sum"]:+.2f}%')
    print()

    print('=== Forced Trend Exits (panic-equivalent?) ===')
    fe = analysis['forced_exits']
    print(f'  Total forced exits: {fe["count"]}')
    print(f'  Avg distance from init_mid at exit: {fe["avg_distance_pct"]:+.4f}%')
    print(f'  Avg net PnL on forced exits: {fe["avg_net_pnl_pct"]:+.4f}%')
    print(f'  Panic-LONG (long held during downward trend exit): {fe["panic_long_count"]}')
    print(f'  Panic-SHORT (short held during upward trend exit): {fe["panic_short_count"]}')
    print()

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'analysis': analysis,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'r26_behavioral_analysis_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
