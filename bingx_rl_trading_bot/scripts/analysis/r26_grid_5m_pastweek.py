"""R26 Grid — 5m Intrabar Backtest (LIVE-parity, advisor-validated).

Windows: past week (2026-04-24 ~ 05-01) + past month (2026-03-31 ~ 05-01) UTC.

LIVE parity model (advisor 2026-05-01 review):
  - Re-arm fill: when SL/TP closes a position, the grid level is re-armed.
    Next bar, if open[i] is unfavorable to level (marketable), entry fills
    at open[i] with TAKER friction. Otherwise passive maker fill on touch.
    (Fixes the prior infinite-SL spiral that produced -3158% with naive re-arm.)
  - Halt: daily NAV drop > halt_daily_loss_pct → force close + simulation stop.
  - Lookahead: grid setup at bar i → fills evaluated from bar i+1 onward.
  - Funding: drag every 8h on open notional × funding_fee_per_8h_pct.

Variants compared:
  A) legacy_no_rearm    — round26 BT replica (single-fill per level)
  B) live_parity        — re-arm + marketable LIMIT (this is the LIVE model)
  C) live_parity_with_halt — variant B + halt + funding (production realism)
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

DATA_5M_WEEK = DATA / 'btc_5m_recent_week.csv'
DATA_5M_MONTH = DATA / 'btc_5m_recent_month.csv'
DATA_1H_45D = DATA / 'btc_1h_recent_45d.csv'
DATA_1H_65D = DATA / 'btc_1h_recent_65d.csv'

CFG = {
    'asset': 'BTC/USDT:USDT',
    'balance_usd': 500.0,
    'balance_utilization_pct': 100.0,
    'trading_leverage': 4,
    'exchange_leverage': 10,
    'grid_spacing_pct': 0.30,
    'grid_levels_each_side': 5,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'trend_exit_distance_pct': 1.5,
    'max_grid_lifetime_bars': 168,
    'per_position_stop_loss_pct': 2.0,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'halt_daily_loss_pct': 3.0,
    'halt_total_loss_pct': 10.0,
    'funding_fee_per_8h_pct': 0.01,
}

WINDOWS = [
    ('PAST_WEEK',  pd.Timestamp('2026-04-24 00:00:00', tz='UTC'),
                   pd.Timestamp('2026-05-01 00:00:00', tz='UTC'),
                   DATA_5M_WEEK, DATA_1H_45D),
    ('PAST_MONTH', pd.Timestamp('2026-03-31 00:00:00', tz='UTC'),
                   pd.Timestamp('2026-05-01 00:00:00', tz='UTC'),
                   DATA_5M_MONTH, DATA_1H_65D),
]


def load(p5, p1):
    df5 = pd.read_csv(p5, parse_dates=['timestamp'])
    df1 = pd.read_csv(p1, parse_dates=['timestamp'])
    df5['timestamp'] = pd.to_datetime(df5['timestamp'], utc=True)
    df1['timestamp'] = pd.to_datetime(df1['timestamp'], utc=True)
    return (df5.sort_values('timestamp').reset_index(drop=True),
            df1.sort_values('timestamp').reset_index(drop=True))


def compute_atr(df, period):
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_ranging_1h(df1):
    atr = compute_atr(df1, CFG['atr_period'])
    atr_pct = atr / df1['close']
    median_30d = atr_pct.rolling(CFG['atr_pct_median_lookback_bars'],
                                  min_periods=240).median()
    s = (atr_pct < median_30d).fillna(False)
    s.index = df1['timestamp']
    return s


def map_ranging_to_5m(df5, ranging_1h):
    """Use most recently CLOSED 1h bar (no lookahead)."""
    floors = df5['timestamp'].dt.floor('1h') - pd.Timedelta(hours=1)
    return ranging_1h.reindex(floors).fillna(False).values.astype(bool)


def simulate(df5, ranging_5m, mode='live_parity'):
    """Modes:
      'legacy_no_rearm'        — single-fill per level (round26 replica)
      'live_parity'            — re-arm + marketable LIMIT (LIVE model)
      'live_parity_with_halt'  — live_parity + daily halt + funding
    """
    n = len(df5)
    open_p = df5['open'].values
    high = df5['high'].values
    low = df5['low'].values
    close = df5['close'].values
    ts = df5['timestamp'].values
    ts_pd = pd.to_datetime(ts, utc=True)

    balance = CFG['balance_usd']
    lev = CFG['trading_leverage']
    util = CFG['balance_utilization_pct'] / 100
    levels = CFG['grid_levels_each_side']
    per_level_notional = balance * util * lev / (levels * 2)
    spacing = CFG['grid_spacing_pct'] / 100
    sl_pct = CFG['per_position_stop_loss_pct'] / 100
    maker_fric = CFG['maker_friction_per_side_pct'] / 100
    taker_fric = CFG['taker_friction_per_side_pct'] / 100
    trend_exit_dist = CFG['trend_exit_distance_pct'] / 100
    max_lifetime_5m = CFG['max_grid_lifetime_bars'] * 12
    halt_daily = CFG['halt_daily_loss_pct']
    funding_per_8h = CFG['funding_fee_per_8h_pct'] / 100  # decimal

    rearm_enabled = mode in ('live_parity', 'live_parity_with_halt')
    halt_enabled = (mode == 'live_parity_with_halt')

    active = None
    trades = []
    cum_harvest = cum_drift = cum_friction = cum_funding = 0.0
    n_setups = n_tp = n_sl = n_trend = n_max = 0
    n_marketable = n_passive_open = 0
    ranging_count = 0
    halt_triggered = False; halt_info = None
    nav_pct = 0.0   # session-start anchor (always 0; we measure cumulative drop)

    def total_realized():
        return cum_harvest + cum_drift + cum_funding

    def equity_equivalent_pct(idx, active_state):
        """LIVE-parity: equity = realized + unrealized MTM (used for halt check)."""
        unrealized = 0.0
        if active_state is not None:
            cur_close = close[idx]
            for p in active_state['open_positions']:
                if p['side'] == 'LONG':
                    pnl = (cur_close - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl = (p['entry_price'] - cur_close) / p['entry_price'] * 100
                unrealized += pnl * (p['qty_usd'] / balance)
        return total_realized() + unrealized

    def force_close(reason, idx, exit_price_override=None):
        nonlocal cum_drift, cum_friction
        for p in active['open_positions']:
            xp = exit_price_override if exit_price_override is not None else close[idx]
            if p['side'] == 'LONG':
                pnl_pct = (xp - p['entry_price']) / p['entry_price'] * 100
            else:
                pnl_pct = (p['entry_price'] - xp) / p['entry_price'] * 100
            fric_pct = taker_fric * 100
            net_pct = pnl_pct - fric_pct
            contrib = net_pct * (p['qty_usd'] / balance)
            cum_drift += contrib
            cum_friction += fric_pct * (p['qty_usd'] / balance)
            trades.append({
                'open_ts': str(ts[p['open_idx']]), 'close_ts': str(ts[idx]),
                'side': p['side'], 'reason': reason,
                'entry_price': p['entry_price'], 'exit_price': float(xp),
                'gross_pct': pnl_pct, 'net_pct': net_pct,
                'contrib_pct': contrib, 'qty_usd': p['qty_usd'],
            })

    for i in range(n):
        if ranging_5m[i]:
            ranging_count += 1

        # --- Halt check (LIVE parity: session-start anchor + equity_equivalent) ---
        # LIVE bot.py uses self.start_capital set on first cycle (NOT daily reset),
        # and equity = realized + unrealized.
        if halt_enabled and not halt_triggered:
            equity_pct = equity_equivalent_pct(i, active)
            cum_loss = -equity_pct  # session_start_nav (=0) - equity
            if cum_loss > halt_daily:
                if active is not None:
                    force_close(f'HALT_DAILY_{cum_loss:.2f}', i)
                    active = None
                halt_triggered = True
                halt_info = {'idx': int(i), 'ts': str(ts[i]),
                              'reason': f'CUM_LOSS_{cum_loss:.2f}_PCT',
                              'equity_at_halt_pct': equity_pct,
                              'realized_at_halt_pct': total_realized()}
                break

        # --- Funding fee (8h interval) ---
        if (mode == 'live_parity_with_halt' and active is not None
                and active['open_positions']
                and ts_pd[i].hour in (0, 8, 16)
                and ts_pd[i].minute < 5):
            open_notional = sum(p['qty_usd'] for p in active['open_positions'])
            # Symmetric assumption: net longs vs shorts get/pay funding.
            # Conservative: treat as drag on |net| × funding rate.
            net_long = sum(p['qty_usd'] if p['side']=='LONG' else -p['qty_usd']
                            for p in active['open_positions'])
            funding_drag = abs(net_long) * funding_per_8h / balance * 100
            cum_funding -= funding_drag

        # --- Setup new grid ---
        if active is None and ranging_5m[i]:
            init_mid = close[i]
            active = {
                'init_mid': init_mid, 'init_idx': i,
                'buy_levels':  [init_mid * (1 - spacing * (k + 1)) for k in range(levels)],
                'sell_levels': [init_mid * (1 + spacing * (k + 1)) for k in range(levels)],
                'buy_filled':  [False] * levels,
                'sell_filled': [False] * levels,
                'open_positions': [],
            }
            n_setups += 1
            continue   # NO fills on setup bar (lookahead-1 shift)

        if active is None:
            continue

        # --- Forced exits ---
        elapsed = i - active['init_idx']
        dist = abs(close[i] - active['init_mid']) / active['init_mid']
        force_exit_reason = None
        if elapsed >= max_lifetime_5m:
            force_exit_reason = 'MAX_LIFETIME'; n_max += 1
        elif dist > trend_exit_dist and not ranging_5m[i]:
            force_exit_reason = 'TREND_EXIT'; n_trend += 1
        if force_exit_reason:
            force_close(force_exit_reason, i)
            active = None
            continue

        # --- Entry phase (LIVE parity: marketable LIMIT vs passive) ---
        for k in range(levels):
            # BUY level k
            if not active['buy_filled'][k]:
                lvl = active['buy_levels'][k]
                if rearm_enabled and open_p[i] < lvl:
                    # Marketable BUY LIMIT — taker @ open
                    entry = open_p[i]
                    fric_per = taker_fric
                    n_marketable += 1
                    fill_kind = 'taker'
                elif low[i] <= lvl:
                    entry = lvl
                    fric_per = maker_fric
                    n_passive_open += 1
                    fill_kind = 'maker'
                else:
                    entry = None
                if entry is not None:
                    cum_friction += fric_per * 100 * (per_level_notional / balance)
                    active['open_positions'].append({
                        'side': 'LONG', 'entry_price': entry,
                        'tp_price': entry * (1 + spacing),
                        'sl_price': entry * (1 - sl_pct),
                        'qty_usd': per_level_notional, 'open_idx': i,
                        'level_idx': k, 'level_side': 'buy',
                        'entry_kind': fill_kind,
                    })
                    active['buy_filled'][k] = True
            # SELL level k
            if not active['sell_filled'][k]:
                lvl = active['sell_levels'][k]
                if rearm_enabled and open_p[i] > lvl:
                    entry = open_p[i]
                    fric_per = taker_fric
                    n_marketable += 1
                    fill_kind = 'taker'
                elif high[i] >= lvl:
                    entry = lvl
                    fric_per = maker_fric
                    n_passive_open += 1
                    fill_kind = 'maker'
                else:
                    entry = None
                if entry is not None:
                    cum_friction += fric_per * 100 * (per_level_notional / balance)
                    active['open_positions'].append({
                        'side': 'SHORT', 'entry_price': entry,
                        'tp_price': entry * (1 - spacing),
                        'sl_price': entry * (1 + sl_pct),
                        'qty_usd': per_level_notional, 'open_idx': i,
                        'level_idx': k, 'level_side': 'sell',
                        'entry_kind': fill_kind,
                    })
                    active['sell_filled'][k] = True

        # --- Exit phase (TP / SL on open positions) ---
        new_open = []
        for p in active['open_positions']:
            if p['open_idx'] == i:
                # Same-bar exit not allowed: entry just opened this bar
                new_open.append(p)
                continue
            tp_hit = sl_hit = False
            if p['side'] == 'LONG':
                tp_hit = high[i] >= p['tp_price']
                sl_hit = low[i] <= p['sl_price']
            else:
                tp_hit = low[i] <= p['tp_price']
                sl_hit = high[i] >= p['sl_price']
            if tp_hit and sl_hit:
                tp_hit = False  # SL wins on same bar (worst case)
            if sl_hit:
                xp = p['sl_price']
                if p['side'] == 'LONG':
                    pnl_pct = (xp - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl_pct = (p['entry_price'] - xp) / p['entry_price'] * 100
                fric_pct = taker_fric * 100
                net_pct = pnl_pct - fric_pct
                contrib = net_pct * (p['qty_usd'] / balance)
                cum_drift += contrib
                cum_friction += fric_pct * (p['qty_usd'] / balance)
                n_sl += 1
                trades.append({
                    'open_ts': str(ts[p['open_idx']]), 'close_ts': str(ts[i]),
                    'side': p['side'], 'reason': 'PER_POS_SL',
                    'entry_price': p['entry_price'], 'exit_price': xp,
                    'gross_pct': pnl_pct, 'net_pct': net_pct,
                    'contrib_pct': contrib, 'qty_usd': p['qty_usd'],
                    'entry_kind': p.get('entry_kind'),
                })
                if rearm_enabled:
                    if p['level_side'] == 'buy':
                        active['buy_filled'][p['level_idx']] = False
                    else:
                        active['sell_filled'][p['level_idx']] = False
                continue
            if tp_hit:
                xp = p['tp_price']
                if p['side'] == 'LONG':
                    pnl_pct = (xp - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl_pct = (p['entry_price'] - xp) / p['entry_price'] * 100
                fric_pct = maker_fric * 100
                net_pct = pnl_pct - fric_pct
                contrib = net_pct * (p['qty_usd'] / balance)
                cum_harvest += contrib
                cum_friction += fric_pct * (p['qty_usd'] / balance)
                n_tp += 1
                trades.append({
                    'open_ts': str(ts[p['open_idx']]), 'close_ts': str(ts[i]),
                    'side': p['side'], 'reason': 'TP_CYCLE',
                    'entry_price': p['entry_price'], 'exit_price': xp,
                    'gross_pct': pnl_pct, 'net_pct': net_pct,
                    'contrib_pct': contrib, 'qty_usd': p['qty_usd'],
                    'entry_kind': p.get('entry_kind'),
                })
                if rearm_enabled:
                    if p['level_side'] == 'buy':
                        active['buy_filled'][p['level_idx']] = False
                    else:
                        active['sell_filled'][p['level_idx']] = False
                continue
            new_open.append(p)
        active['open_positions'] = new_open

    # MTM at end
    final_unrealized = 0.0
    if active is not None and active['open_positions']:
        last_close = close[-1]
        for p in active['open_positions']:
            if p['side'] == 'LONG':
                pnl = (last_close - p['entry_price']) / p['entry_price'] * 100
            else:
                pnl = (p['entry_price'] - last_close) / p['entry_price'] * 100
            final_unrealized += pnl * (p['qty_usd'] / balance)

    return {
        'mode': mode,
        'n_5m_bars': n,
        'ranging_fraction_5m': float(ranging_count / n) if n else 0.0,
        'per_level_notional_usd': per_level_notional,
        'n_setups': n_setups, 'n_tp': n_tp, 'n_sl': n_sl,
        'n_trend_exits': n_trend, 'n_max_lifetime': n_max,
        'n_marketable_fills': n_marketable, 'n_passive_fills': n_passive_open,
        'cum_harvest_pct': float(cum_harvest),
        'cum_drift_pct': float(cum_drift),
        'cum_friction_pct': float(cum_friction),
        'cum_funding_pct': float(cum_funding),
        'cum_realized_net_pct': float(cum_harvest + cum_drift + cum_funding),
        'final_unrealized_pct': float(final_unrealized),
        'cum_total_pct': float(cum_harvest + cum_drift + cum_funding + final_unrealized),
        'open_positions_at_end': len(active['open_positions']) if active else 0,
        'halt_triggered': halt_triggered, 'halt_info': halt_info,
        'trades': trades,
    }


def summarize(res, n_days):
    trades = pd.DataFrame(res['trades']) if res['trades'] else pd.DataFrame()
    n = len(trades)
    if n > 0:
        avg_gross = float(trades['gross_pct'].mean())
        avg_net = float(trades['net_pct'].mean())
        wr = float((trades['net_pct'] > 0).mean())
        n_long = int((trades['side'] == 'LONG').sum())
        n_short = int((trades['side'] == 'SHORT').sum())
        trades['close_dt'] = pd.to_datetime(trades['close_ts'])
        daily = trades.groupby(trades['close_dt'].dt.floor('D'))['contrib_pct'].sum()
        worst = float(daily.min()) if len(daily) else 0.0
        best = float(daily.max()) if len(daily) else 0.0
    else:
        avg_gross = avg_net = wr = 0.0
        n_long = n_short = 0; worst = best = 0.0
    return {
        'mode': res['mode'],
        'n_trades': n, 'n_long': n_long, 'n_short': n_short,
        'cum_realized_net_pct': res['cum_realized_net_pct'],
        'cum_harvest_pct': res['cum_harvest_pct'],
        'cum_drift_pct': res['cum_drift_pct'],
        'cum_friction_pct': res['cum_friction_pct'],
        'cum_funding_pct': res['cum_funding_pct'],
        'final_unrealized_pct': res['final_unrealized_pct'],
        'cum_total_pct': res['cum_total_pct'],
        'avg_gross_per_trade_pct': avg_gross,
        'avg_net_per_trade_pct': avg_net,
        'wr': wr,
        'daily_total_pct': res['cum_total_pct'] / n_days,
        'best_day_pct': best, 'worst_day_pct': worst,
        'trades_per_day': n / n_days,
        'n_setups': res['n_setups'], 'n_tp': res['n_tp'], 'n_sl': res['n_sl'],
        'n_trend_exits': res['n_trend_exits'], 'n_max_lifetime': res['n_max_lifetime'],
        'n_marketable_fills': res['n_marketable_fills'],
        'n_passive_fills': res['n_passive_fills'],
        'open_positions_at_end': res['open_positions_at_end'],
        'ranging_fraction_5m': res['ranging_fraction_5m'],
        'per_level_notional_usd': res['per_level_notional_usd'],
        'halt_triggered': res['halt_triggered'], 'halt_info': res['halt_info'],
    }


def print_block(s):
    print(f'  setups/TP/SL/trend/maxLife/openEnd: '
          f'{s["n_setups"]}/{s["n_tp"]}/{s["n_sl"]}/{s["n_trend_exits"]}/'
          f'{s["n_max_lifetime"]}/{s["open_positions_at_end"]}')
    print(f'  marketable/passive fills: {s["n_marketable_fills"]}/{s["n_passive_fills"]}')
    print(f'  cum total: {s["cum_total_pct"]:+.4f}%  '
          f'(harvest {s["cum_harvest_pct"]:+.4f}%, drift {s["cum_drift_pct"]:+.4f}%, '
          f'funding {s["cum_funding_pct"]:+.4f}%)')
    print(f'  daily: {s["daily_total_pct"]:+.4f}%/day  '
          f'best/worst: {s["best_day_pct"]:+.4f}% / {s["worst_day_pct"]:+.4f}%')
    print(f'  n_trades: {s["n_trades"]} ({s["n_long"]}L/{s["n_short"]}S), '
          f'WR {s["wr"]:.3f}, avg_net {s["avg_gross_per_trade_pct"]:+.4f}% gross / '
          f'{s["avg_net_per_trade_pct"]:+.4f}% net')
    if s['halt_triggered']:
        print(f'  >>> HALT: {s["halt_info"]}')


def run_window(label, bt_start, bt_end, p5, p1):
    print('=' * 100)
    print(f'R26 Grid — 5m BT — {label}    Window: {bt_start} → {bt_end}')
    print('=' * 100)
    df5, df1 = load(p5, p1)
    print(f'5m: {len(df5):,} | 1h: {len(df1):,}')
    ranging_1h = compute_ranging_1h(df1)
    rf = ranging_1h[(ranging_1h.index >= bt_start) & (ranging_1h.index <= bt_end)]
    print(f'1h ranging fraction in window: {rf.mean():.3f}  (long-run ≈ 0.50)')

    mask = (df5['timestamp'] >= bt_start) & (df5['timestamp'] <= bt_end)
    df5w = df5.loc[mask].reset_index(drop=True)
    ranging_5m = map_ranging_to_5m(df5w, ranging_1h)
    n_days = (df5w['timestamp'].max() - df5w['timestamp'].min()).total_seconds() / 86400
    print(f'5m bars in window: {len(df5w):,}, span {n_days:.2f} days\n')

    out = {'label': label, 'window_utc': [str(bt_start), str(bt_end)],
            'span_days': n_days, 'ranging_fraction_1h': float(rf.mean())}

    for mode, title in [
        ('legacy_no_rearm',     'A) legacy (round26 replica, no re-arm)'),
        ('live_parity',         'B) live_parity (re-arm + marketable LIMIT)'),
        ('live_parity_with_halt', 'C) live_parity + halt + funding (production)'),
    ]:
        print(f'--- {title} ---')
        res = simulate(df5w, ranging_5m, mode=mode)
        s = summarize(res, n_days)
        print_block(s)
        print()
        out[mode] = {k: v for k, v in s.items() if k != 'mode'}
    return out


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_out = {'date': datetime.now(timezone.utc).isoformat(), 'config': CFG}
    for label, bt_start, bt_end, p5, p1 in WINDOWS:
        all_out[label] = run_window(label, bt_start, bt_end, p5, p1)
        print()

    print('=' * 100)
    print('Summary across modes (cum_total_pct)')
    print('=' * 100)
    print(f'{"Window":<12} {"A_legacy":>12} {"B_live_parity":>15} {"C_with_halt":>15} {"ranging%":>10}')
    for label, *_ in WINDOWS:
        w = all_out[label]
        a = w['legacy_no_rearm']['cum_total_pct']
        b = w['live_parity']['cum_total_pct']
        c = w['live_parity_with_halt']['cum_total_pct']
        r = w['ranging_fraction_1h']
        print(f'{label:<12} {a:>+11.4f}% {b:>+14.4f}% {c:>+14.4f}% {r:>9.3f}')
    print()

    p = RESULTS / f'r26_grid_5m_multi_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
