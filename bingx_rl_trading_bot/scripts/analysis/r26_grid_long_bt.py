"""R26 Grid — long-period BT with auto-restart, path metrics, halt-anchor reset.

Plan (advisor 2026-05-01 narrowed):
  - Variant A: legacy (no re-arm, no halt) — sanity-check vs round26 BT (+0.21%/day claim)
  - Variant D: live_parity + halt + auto-restart — LIVE-realistic (user redeploys after halt)

Datasets:
  - BingX 100d 5m (max BingX historical depth) — exchange-matched, primary
  - Binance 720d 5m — longer span, data-source caveat (5-30bps gap vs BingX)

Path metrics (variant D):
  - Equity curve (daily snapshots)
  - Max drawdown (peak-to-trough %)
  - Halt count + temporal distribution
  - Longest consecutive losing-day streak
  - Cumulative PnL %

Halt-anchor reset on auto-restart:
  - LIVE bot.py:201 sets `start_capital = equity` ONCE on first cycle.
  - Auto-restart simulates user redeploying: new session_anchor = current equity_pct.
  - balance stays $500 (PnL %는 cumulative).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from r26_grid_5m_pastweek import (
    CFG, compute_ranging_1h, map_ranging_to_5m,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

DATASETS = [
    ('BingX_100d',   DATA / 'btc_5m_bingx_100d.csv',  DATA / 'btc_1h_bingx_long.csv'),
    ('Binance_720d', DATA / 'btc_5m_720days_binance.csv', DATA / 'btc_1h_720days.csv'),
]


def simulate_long(df5, ranging_5m, mode='live_parity_with_restart'):
    """Long BT with auto-restart on halt and full path metrics.

    mode:
      'legacy_no_rearm'             — no re-arm, no halt (round26 BT replica)
      'live_parity_with_restart'    — re-arm + marketable LIMIT + halt + auto-restart

    Balance compounding (LIVE auto_size_from_balance=true parity):
      - balance_usd tracked throughout sim
      - per_level_notional recomputed (a) on every grid setup, (b) on every TP/SL close
        → matches LIVE grid.py:_replace_grid_level notional_callback
      - balance ≤ ruin_threshold ($25 = 5% of start) → sim halts (RUIN)
    """
    n = len(df5)
    open_p = df5['open'].values
    high = df5['high'].values
    low = df5['low'].values
    close = df5['close'].values
    ts = df5['timestamp'].values
    ts_pd = pd.to_datetime(ts, utc=True)

    starting_balance = CFG['balance_usd']
    balance = starting_balance   # tracked dynamically (compound)
    lev = CFG['trading_leverage']
    util = CFG['balance_utilization_pct'] / 100
    levels = CFG['grid_levels_each_side']
    total_lvl = levels * 2

    def per_level_now():
        return max(balance * util * lev / total_lvl, 0.0)

    per_level_notional = per_level_now()
    ruin_threshold = starting_balance * 0.05   # 5% of starting → ruin
    ruin_event = None
    spacing = CFG['grid_spacing_pct'] / 100
    sl_pct = CFG['per_position_stop_loss_pct'] / 100
    maker_fric = CFG['maker_friction_per_side_pct'] / 100
    taker_fric = CFG['taker_friction_per_side_pct'] / 100
    trend_exit_dist = CFG['trend_exit_distance_pct'] / 100
    max_lifetime_5m = CFG['max_grid_lifetime_bars'] * 12
    halt_daily = CFG['halt_daily_loss_pct']
    funding_per_8h = CFG['funding_fee_per_8h_pct'] / 100

    rearm = mode == 'live_parity_with_restart'
    halt_enabled = mode == 'live_parity_with_restart'

    active = None
    trades = []
    n_setups = n_tp = n_sl = n_trend = n_max = 0
    n_marketable = n_passive_open = 0
    ranging_count = 0
    cum_friction_usd = 0.0
    cum_funding_usd = 0.0

    # Halt + auto-restart state — anchor in $ terms (LIVE bot.py:start_capital)
    session_anchor_balance = balance
    halt_events = []

    daily_snapshots = []   # (date, equity_usd, balance_usd)
    last_snapshot_day = None

    def equity_usd(idx, active_state):
        """Realized + unrealized equity in USD."""
        u = 0.0
        if active_state is not None:
            cc = close[idx]
            for p in active_state['open_positions']:
                if p['side'] == 'LONG':
                    u += (cc - p['entry_price']) / p['entry_price'] * p['qty_usd']
                else:
                    u += (p['entry_price'] - cc) / p['entry_price'] * p['qty_usd']
        return balance + u

    def apply_pnl_to_balance(net_pct, qty_usd):
        """Update balance USD; recompute per_level for next fills."""
        nonlocal balance, per_level_notional
        pnl_usd = net_pct / 100.0 * qty_usd
        balance += pnl_usd
        per_level_notional = per_level_now()
        return pnl_usd

    def apply_friction_to_balance(fric_pct, qty_usd):
        nonlocal balance, per_level_notional, cum_friction_usd
        fric_usd = fric_pct / 100.0 * qty_usd
        balance -= fric_usd
        cum_friction_usd += fric_usd
        per_level_notional = per_level_now()

    def force_close(reason, idx):
        if active is None: return
        for p in active['open_positions']:
            xp = close[idx]
            if p['side'] == 'LONG':
                pnl_pct = (xp - p['entry_price']) / p['entry_price'] * 100
            else:
                pnl_pct = (p['entry_price'] - xp) / p['entry_price'] * 100
            fric = taker_fric * 100
            net = pnl_pct - fric
            pnl_usd = apply_pnl_to_balance(net, p['qty_usd'])
            trades.append({
                'open_ts': str(ts[p['open_idx']]), 'close_ts': str(ts[idx]),
                'side': p['side'], 'reason': reason,
                'entry_price': p['entry_price'], 'exit_price': float(xp),
                'gross_pct': pnl_pct, 'net_pct': net,
                'pnl_usd': pnl_usd, 'qty_usd': p['qty_usd'],
                'balance_after': balance,
            })

    for i in range(n):
        if ranging_5m[i]:
            ranging_count += 1

        # Daily snapshot
        d = ts_pd[i].date()
        if last_snapshot_day != d:
            daily_snapshots.append((str(d), equity_usd(i, active), balance))
            last_snapshot_day = d

        # RUIN check
        cur_eq = equity_usd(i, active)
        if cur_eq <= ruin_threshold:
            ruin_event = {
                'idx': int(i), 'ts': str(ts[i]),
                'equity_at_ruin_usd': cur_eq,
                'balance_at_ruin_usd': balance,
                'pct_remaining': cur_eq / starting_balance * 100,
            }
            force_close('RUIN', i)
            active = None
            break

        # Halt check (LIVE-parity: session_anchor in USD, equity = realized + unrealized)
        if halt_enabled:
            cum_loss_pct = (session_anchor_balance - cur_eq) / session_anchor_balance * 100
            if cum_loss_pct > halt_daily:
                force_close(f'HALT_{cum_loss_pct:.2f}', i)
                halt_events.append({
                    'idx': int(i), 'ts': str(ts[i]),
                    'cum_loss_pct_at_halt': cum_loss_pct,
                    'equity_at_halt_usd': cur_eq,
                    'balance_at_halt_usd': balance,
                    'session_anchor_usd': session_anchor_balance,
                })
                active = None
                # Auto-restart: reset anchor to current equity (LIVE redeploy)
                session_anchor_balance = balance

        # Funding (8h) — drag in USD
        if (active is not None and active['open_positions']
                and ts_pd[i].hour in (0, 8, 16) and ts_pd[i].minute < 5):
            net_long = sum(p['qty_usd'] if p['side']=='LONG' else -p['qty_usd']
                            for p in active['open_positions'])
            funding_usd = abs(net_long) * funding_per_8h
            balance -= funding_usd
            cum_funding_usd += funding_usd
            per_level_notional = per_level_now()

        # Setup
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
            continue

        if active is None:
            continue

        # Force exits
        elapsed = i - active['init_idx']
        dist = abs(close[i] - active['init_mid']) / active['init_mid']
        fer = None
        if elapsed >= max_lifetime_5m:
            fer = 'MAX_LIFETIME'; n_max += 1
        elif dist > trend_exit_dist and not ranging_5m[i]:
            fer = 'TREND_EXIT'; n_trend += 1
        if fer:
            force_close(fer, i)
            active = None
            continue

        # Entry phase (LIVE-parity, balance-aware per_level)
        for k in range(levels):
            if not active['buy_filled'][k]:
                lvl = active['buy_levels'][k]
                if rearm and open_p[i] < lvl:
                    entry, fric_per, kind = open_p[i], taker_fric, 'taker'
                    n_marketable += 1
                elif low[i] <= lvl:
                    entry, fric_per, kind = lvl, maker_fric, 'maker'
                    n_passive_open += 1
                else:
                    entry = None
                if entry is not None and per_level_notional > 0:
                    qty = per_level_notional   # snapshot at fill time
                    apply_friction_to_balance(fric_per * 100, qty)
                    active['open_positions'].append({
                        'side': 'LONG', 'entry_price': entry,
                        'tp_price': entry * (1 + spacing),
                        'sl_price': entry * (1 - sl_pct),
                        'qty_usd': qty, 'open_idx': i,
                        'level_idx': k, 'level_side': 'buy', 'entry_kind': kind,
                    })
                    active['buy_filled'][k] = True
            if not active['sell_filled'][k]:
                lvl = active['sell_levels'][k]
                if rearm and open_p[i] > lvl:
                    entry, fric_per, kind = open_p[i], taker_fric, 'taker'
                    n_marketable += 1
                elif high[i] >= lvl:
                    entry, fric_per, kind = lvl, maker_fric, 'maker'
                    n_passive_open += 1
                else:
                    entry = None
                if entry is not None and per_level_notional > 0:
                    qty = per_level_notional
                    apply_friction_to_balance(fric_per * 100, qty)
                    active['open_positions'].append({
                        'side': 'SHORT', 'entry_price': entry,
                        'tp_price': entry * (1 - spacing),
                        'sl_price': entry * (1 + sl_pct),
                        'qty_usd': qty, 'open_idx': i,
                        'level_idx': k, 'level_side': 'sell', 'entry_kind': kind,
                    })
                    active['sell_filled'][k] = True

        # Exit phase
        new_open = []
        for p in active['open_positions']:
            if p['open_idx'] == i:
                new_open.append(p); continue
            if p['side'] == 'LONG':
                tp_hit = high[i] >= p['tp_price']
                sl_hit = low[i] <= p['sl_price']
            else:
                tp_hit = low[i] <= p['tp_price']
                sl_hit = high[i] >= p['sl_price']
            if tp_hit and sl_hit: tp_hit = False
            if sl_hit:
                xp = p['sl_price']
                if p['side'] == 'LONG':
                    pnl = (xp - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl = (p['entry_price'] - xp) / p['entry_price'] * 100
                fric = taker_fric * 100; net = pnl - fric
                pnl_usd = apply_pnl_to_balance(net, p['qty_usd'])
                n_sl += 1
                trades.append({'open_ts':str(ts[p['open_idx']]),'close_ts':str(ts[i]),
                                'side':p['side'],'reason':'PER_POS_SL',
                                'entry_price':p['entry_price'],'exit_price':xp,
                                'gross_pct':pnl,'net_pct':net,
                                'pnl_usd':pnl_usd,'qty_usd':p['qty_usd'],
                                'balance_after':balance})
                if rearm:
                    if p['level_side']=='buy': active['buy_filled'][p['level_idx']] = False
                    else: active['sell_filled'][p['level_idx']] = False
                continue
            if tp_hit:
                xp = p['tp_price']
                if p['side'] == 'LONG':
                    pnl = (xp - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl = (p['entry_price'] - xp) / p['entry_price'] * 100
                fric = maker_fric * 100; net = pnl - fric
                pnl_usd = apply_pnl_to_balance(net, p['qty_usd'])
                n_tp += 1
                trades.append({'open_ts':str(ts[p['open_idx']]),'close_ts':str(ts[i]),
                                'side':p['side'],'reason':'TP_CYCLE',
                                'entry_price':p['entry_price'],'exit_price':xp,
                                'gross_pct':pnl,'net_pct':net,
                                'pnl_usd':pnl_usd,'qty_usd':p['qty_usd'],
                                'balance_after':balance})
                if rearm:
                    if p['level_side']=='buy': active['buy_filled'][p['level_idx']] = False
                    else: active['sell_filled'][p['level_idx']] = False
                continue
            new_open.append(p)
        active['open_positions'] = new_open

    # Final equity USD (realized balance + unrealized of remaining open positions)
    final_unrealized_usd = 0.0
    if active is not None:
        cc = close[-1]
        for p in active['open_positions']:
            if p['side']=='LONG':
                final_unrealized_usd += (cc - p['entry_price']) / p['entry_price'] * p['qty_usd']
            else:
                final_unrealized_usd += (p['entry_price'] - cc) / p['entry_price'] * p['qty_usd']
    final_equity_usd = balance + final_unrealized_usd

    # Path metrics — equity in USD terms, then % from starting
    eq_arr = np.array([e for _, e, _ in daily_snapshots])
    if len(eq_arr) > 1:
        running_max = np.maximum.accumulate(eq_arr)
        dd_pct = (eq_arr - running_max) / running_max * 100
        max_drawdown_pct = float(dd_pct.min())
        daily_diff = np.diff(eq_arr)
        cur_streak = max_streak = 0
        for di in daily_diff:
            if di < 0:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0
    else:
        max_drawdown_pct = 0.0
        max_streak = 0

    cum_total_pct = (final_equity_usd / starting_balance - 1) * 100

    return {
        'mode': mode,
        'n_5m_bars': n,
        'span_days': (ts_pd[-1] - ts_pd[0]).total_seconds() / 86400,
        'ranging_fraction_5m': float(ranging_count / n) if n else 0.0,
        'starting_balance_usd': starting_balance,
        'final_balance_usd': float(balance),
        'final_equity_usd': float(final_equity_usd),
        'final_unrealized_usd': float(final_unrealized_usd),
        'final_per_level_notional_usd': float(per_level_notional),
        'cum_total_pct': float(cum_total_pct),
        'cum_friction_usd': float(cum_friction_usd),
        'cum_funding_usd': float(cum_funding_usd),
        'n_setups': n_setups, 'n_tp': n_tp, 'n_sl': n_sl,
        'n_trend_exits': n_trend, 'n_max_lifetime': n_max,
        'n_marketable_fills': n_marketable, 'n_passive_fills': n_passive_open,
        'open_positions_at_end': len(active['open_positions']) if active else 0,
        'n_halts': len(halt_events),
        'halt_events': halt_events,
        'ruin_event': ruin_event,
        'max_drawdown_pct': max_drawdown_pct,
        'longest_losing_day_streak': int(max_streak),
        'daily_snapshots_count': len(daily_snapshots),
        'daily_equity_curve': daily_snapshots,
        'n_trades': len(trades),
    }


def summarize(res):
    span = res['span_days']
    cum = res['cum_total_pct']
    return {
        'mode': res['mode'],
        'span_days': span,
        'n_trades': res['n_trades'],
        'n_setups': res['n_setups'],
        'n_tp': res['n_tp'], 'n_sl': res['n_sl'],
        'n_trend_exits': res['n_trend_exits'],
        'n_marketable_fills': res['n_marketable_fills'],
        'starting_balance_usd': res['starting_balance_usd'],
        'final_balance_usd': res['final_balance_usd'],
        'final_equity_usd': res['final_equity_usd'],
        'final_per_level_notional_usd': res['final_per_level_notional_usd'],
        'cum_total_pct': cum,
        'daily_pct': cum / span if span > 0 else 0.0,
        'max_drawdown_pct': res['max_drawdown_pct'],
        'longest_losing_day_streak': res['longest_losing_day_streak'],
        'n_halts': res['n_halts'],
        'halts_per_30d': res['n_halts'] / span * 30 if span > 0 else 0.0,
        'ruin_event': res['ruin_event'],
        'cum_friction_usd': res['cum_friction_usd'],
        'cum_funding_usd': res['cum_funding_usd'],
        'ranging_fraction_5m': res['ranging_fraction_5m'],
    }


def print_block(s):
    print(f'  span: {s["span_days"]:.1f} days  | ranging: {s["ranging_fraction_5m"]:.3f}')
    print(f'  setups/TP/SL/trend: {s["n_setups"]}/{s["n_tp"]}/{s["n_sl"]}/{s["n_trend_exits"]}'
          f'  marketable: {s["n_marketable_fills"]}')
    print(f'  starting balance: ${s["starting_balance_usd"]:.2f}  '
          f'→ final equity: ${s["final_equity_usd"]:.2f}  '
          f'(balance ${s["final_balance_usd"]:.2f})')
    print(f'  per_level final: ${s["final_per_level_notional_usd"]:.2f}  '
          f'(start ${s["starting_balance_usd"] * 4 * 1.0 / 10:.2f})')
    print(f'  cum total: {s["cum_total_pct"]:+.4f}%   daily: {s["daily_pct"]:+.4f}%/day')
    print(f'  Max drawdown: {s["max_drawdown_pct"]:+.4f}%   '
          f'Longest losing streak: {s["longest_losing_day_streak"]} days')
    print(f'  Halts: {s["n_halts"]} total ({s["halts_per_30d"]:.2f}/30d)')
    if s['ruin_event']:
        print(f'  >>> RUIN: {s["ruin_event"]["ts"]}  '
              f'equity ${s["ruin_event"]["equity_at_ruin_usd"]:.2f} '
              f'({s["ruin_event"]["pct_remaining"]:.2f}% of start)')


def run_dataset(label, p5, p1):
    print('=' * 100)
    print(f'Dataset: {label}')
    print('=' * 100)
    df5 = pd.read_csv(p5, parse_dates=['timestamp'])
    df1 = pd.read_csv(p1, parse_dates=['timestamp'])
    df5['timestamp'] = pd.to_datetime(df5['timestamp'], utc=True)
    df1['timestamp'] = pd.to_datetime(df1['timestamp'], utc=True)
    df5 = df5.sort_values('timestamp').reset_index(drop=True)
    df1 = df1.sort_values('timestamp').reset_index(drop=True)
    print(f'5m: {len(df5):,} bars, {df5.timestamp.min()} → {df5.timestamp.max()}')
    print(f'1h: {len(df1):,} bars, {df1.timestamp.min()} → {df1.timestamp.max()}')

    ranging_1h = compute_ranging_1h(df1)
    # Use only 5m bars where lookback is satisfied (drop first 30d after 1h start)
    cutoff = df1['timestamp'].iloc[0] + pd.Timedelta(days=31)
    df5w = df5[df5['timestamp'] >= cutoff].reset_index(drop=True)
    print(f'5m (after 30d lookback warmup): {len(df5w):,} bars, '
          f'{df5w.timestamp.min()} → {df5w.timestamp.max()}\n')

    ranging_5m = map_ranging_to_5m(df5w, ranging_1h)

    out = {'label': label,
            'span_5m': str(df5w['timestamp'].min()) + ' → ' + str(df5w['timestamp'].max())}
    for mode, title in [
        ('legacy_no_rearm',          'A) legacy (no re-arm, no halt)'),
        ('live_parity_with_restart', 'D) live_parity + halt + auto-restart (LIVE-realistic)'),
    ]:
        print(f'--- {title} ---')
        res = simulate_long(df5w, ranging_5m, mode=mode)
        s = summarize(res)
        print_block(s); print()
        out[mode] = s
        # Persist halt events + equity curve in larger key
        out[mode + '_halts'] = res['halt_events']
        out[mode + '_equity_curve'] = res['daily_equity_curve']
    return out


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_out = {'date': datetime.now(timezone.utc).isoformat(), 'config': CFG}
    for label, p5, p1 in DATASETS:
        all_out[label] = run_dataset(label, p5, p1)
        print()

    # Cross-dataset summary
    print('=' * 110)
    print('Cross-dataset summary (balance-aware)')
    print('=' * 110)
    print(f'{"Dataset":<14} {"Mode":<28} {"Span":>6} {"Total %":>10} {"Daily %":>9} '
          f'{"MaxDD %":>9} {"FinalEq$":>10} {"#Halts":>7} {"Ruin?":>7}')
    for label, *_ in DATASETS:
        for mode in ['legacy_no_rearm', 'live_parity_with_restart']:
            s = all_out[label][mode]
            ruin = 'YES' if s['ruin_event'] else 'no'
            print(f'{label:<14} {mode:<28} {s["span_days"]:>6.1f} '
                  f'{s["cum_total_pct"]:>+9.2f}% {s["daily_pct"]:>+8.4f}% '
                  f'{s["max_drawdown_pct"]:>+8.2f}% ${s["final_equity_usd"]:>8.2f} '
                  f'{s["n_halts"]:>7d} {ruin:>7}')
    print()

    p = RESULTS / f'r26_grid_long_bt_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
