"""C2 first-pass BT — funding z-score spread on 8 coins.

Mechanism:
  Every 8h funding tick:
    1. compute 30d rolling z-score per coin
    2. find: top z-score coin (high funding → short perp + long spot)
             bottom z-score coin (low funding → long perp + short spot)
    3. if |max_z| > z_threshold AND |min_z| > z_threshold → enter both legs
    4. hold until z-score normalize (|z| < exit_z) OR max_hold reached
  Each leg: spot + perp hedge → market neutral, harvest funding.

Position: per_leg_notional = capital × util / (n_legs × 2 instruments)
  e.g., $1000 capital × 1.0 util / (2 legs × 2) = $250 per instrument

Friction (LIVE-realistic):
  - Maker open + Maker close = 0.04% × 2 = 0.08% RT per leg
  - Funding fee per period × position size (collected/paid based on side)
  - Slippage 0.02% per side (top-of-book)

This is FIRST-PASS — Gate 1+2+3+4 not all applied yet. Statistical viability check.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)


# Strategy parameters (LOCKED for first-pass)
LOCKED = {
    'z_lookback_days': 30,
    'z_entry_threshold': 1.5,    # |z| > 1.5 to enter (start liberal)
    'z_exit_threshold': 0.5,     # |z| < 0.5 to exit
    'max_hold_periods': 21,      # 21 × 8h = 7d max hold
    'capital_usd': 1000,
    'util_pct': 1.0,
    'maker_friction_pct': 0.04,  # per side
    'slippage_pct': 0.02,        # per side
}


def load_funding():
    df = pd.read_csv(DATA / 'c2_funding_history.csv', parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def compute_zscores(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Per-symbol rolling z-score of funding rate."""
    lookback_periods = lookback_days * 3   # 3 funding events per day
    df = df.copy()
    df['fund_pct'] = df['funding_rate'] * 100   # convert to pct
    df['rolling_mean'] = df.groupby('symbol')['fund_pct'].transform(
        lambda x: x.rolling(lookback_periods, min_periods=lookback_periods // 2).mean()
    )
    df['rolling_std'] = df.groupby('symbol')['fund_pct'].transform(
        lambda x: x.rolling(lookback_periods, min_periods=lookback_periods // 2).std()
    )
    df['zscore'] = (df['fund_pct'] - df['rolling_mean']) / df['rolling_std'].replace(0, np.nan)
    return df


def simulate(df_z: pd.DataFrame) -> dict:
    """Per-tick: find top/bottom z-score coins, enter pair, accumulate PnL."""
    # Pivot: timestamp × symbol = z-score, fund_pct
    pivot_z = df_z.pivot(index='timestamp', columns='symbol', values='zscore')
    pivot_f = df_z.pivot(index='timestamp', columns='symbol', values='fund_pct')

    timestamps = pivot_z.index.sort_values()
    z_thresh = LOCKED['z_entry_threshold']
    z_exit = LOCKED['z_exit_threshold']
    max_hold = LOCKED['max_hold_periods']
    capital = LOCKED['capital_usd']
    fric_per_side = LOCKED['maker_friction_pct'] + LOCKED['slippage_pct']  # 0.06% per side
    leg_notional = capital * LOCKED['util_pct'] / 4   # 2 legs × 2 instruments
    annual_per_pos = leg_notional * 4   # gross capital deployed per cycle (4 instruments)

    active_pair = None   # {'long_coin', 'short_coin', 'enter_idx', 'enter_z_long', 'enter_z_short'}
    trades = []
    cum_funding_pct = 0.0
    cum_friction_pct = 0.0
    n_entries = 0
    n_exits = 0

    for i, ts in enumerate(timestamps):
        z_row = pivot_z.loc[ts]
        f_row = pivot_f.loc[ts]
        if z_row.isna().all():
            continue

        # Identify candidates with valid z-scores
        valid = z_row.dropna()
        if len(valid) < 4:
            continue

        if active_pair is None:
            # Look for entry: highest z (short perp) and lowest z (long perp)
            max_z = valid.max()
            min_z = valid.min()
            if max_z > z_thresh and min_z < -z_thresh:
                short_coin = valid.idxmax()  # high funding → short perp
                long_coin = valid.idxmin()   # low funding → long perp
                if short_coin != long_coin:
                    # Pay friction: 4 instruments × per-side friction
                    entry_friction = 4 * fric_per_side / 100 * leg_notional
                    cum_friction_pct += entry_friction / capital * 100
                    active_pair = {
                        'long_coin': long_coin,
                        'short_coin': short_coin,
                        'enter_idx': i,
                        'enter_ts': ts,
                        'enter_z_long': float(min_z),
                        'enter_z_short': float(max_z),
                        'cum_funding_usd': 0.0,
                    }
                    n_entries += 1

        if active_pair is not None:
            # Collect funding (one period)
            f_long = f_row.get(active_pair['long_coin'], np.nan)
            f_short = f_row.get(active_pair['short_coin'], np.nan)
            # We hold: long perp on long_coin (pays funding if rate > 0)
            #          short perp on short_coin (receives funding if rate > 0)
            # Spot legs: receive nothing (just hedge)
            if not np.isnan(f_long) and not np.isnan(f_short):
                # Funding amount = rate × position notional
                # We're long perp on low-funding coin: pay if rate > 0, receive if rate < 0
                # We're short perp on high-funding coin: receive if rate > 0, pay if rate < 0
                period_funding = (-f_long + f_short) / 100 * leg_notional
                active_pair['cum_funding_usd'] += period_funding

            # Check exit
            held = i - active_pair['enter_idx']
            cur_z_long = z_row.get(active_pair['long_coin'], np.nan)
            cur_z_short = z_row.get(active_pair['short_coin'], np.nan)
            should_exit = False
            exit_reason = None
            if held >= max_hold:
                should_exit, exit_reason = True, 'MAX_HOLD'
            elif (not np.isnan(cur_z_long) and not np.isnan(cur_z_short)
                  and abs(cur_z_long) < z_exit and abs(cur_z_short) < z_exit):
                should_exit, exit_reason = True, 'Z_NORMALIZE'

            if should_exit:
                # Pay exit friction
                exit_friction = 4 * fric_per_side / 100 * leg_notional
                cum_friction_pct += exit_friction / capital * 100
                # Calculate trade PnL (funding only, ignoring price drift since hedged)
                trade_pnl = active_pair['cum_funding_usd'] - exit_friction - (4 * fric_per_side / 100 * leg_notional)
                # entry friction already added globally; recompute net for trade record
                gross_funding = active_pair['cum_funding_usd']
                total_friction = entry_friction + exit_friction if False else (8 * fric_per_side / 100 * leg_notional)
                net_pnl = gross_funding - total_friction
                trades.append({
                    'enter_ts': str(active_pair['enter_ts']),
                    'exit_ts': str(ts),
                    'long_coin': active_pair['long_coin'],
                    'short_coin': active_pair['short_coin'],
                    'enter_z_long': active_pair['enter_z_long'],
                    'enter_z_short': active_pair['enter_z_short'],
                    'periods_held': held,
                    'reason': exit_reason,
                    'gross_funding_usd': gross_funding,
                    'total_friction_usd': total_friction,
                    'net_pnl_usd': net_pnl,
                    'net_pnl_pct': net_pnl / capital * 100,
                })
                cum_funding_pct += gross_funding / capital * 100
                active_pair = None
                n_exits += 1

    cum_net_pct = cum_funding_pct - cum_friction_pct
    return {
        'n_periods': len(timestamps),
        'n_entries': n_entries,
        'n_exits': n_exits,
        'cum_funding_pct': float(cum_funding_pct),
        'cum_friction_pct': float(cum_friction_pct),
        'cum_net_pct': float(cum_net_pct),
        'open_at_end': active_pair is not None,
        'trades': trades,
    }


def summarize(res, n_days):
    trades = pd.DataFrame(res['trades']) if res['trades'] else pd.DataFrame()
    n = len(trades)
    if n > 0:
        avg_pnl_pct = float(trades['net_pnl_pct'].mean())
        wr = float((trades['net_pnl_pct'] > 0).mean())
        avg_periods = float(trades['periods_held'].mean())
    else:
        avg_pnl_pct = wr = avg_periods = 0.0
    return {
        'span_days': n_days,
        'n_entries': res['n_entries'],
        'n_exits': res['n_exits'],
        'n_completed_trades': n,
        'cum_funding_pct': res['cum_funding_pct'],
        'cum_friction_pct': res['cum_friction_pct'],
        'cum_net_pct': res['cum_net_pct'],
        'apy_extrapolated_pct': res['cum_net_pct'] / n_days * 365 if n_days > 0 else 0,
        'usd_per_year_on_1000': res['cum_net_pct'] / n_days * 365 / 100 * 1000 if n_days > 0 else 0,
        'avg_pnl_per_trade_pct': avg_pnl_pct,
        'wr': wr,
        'avg_periods_held': avg_periods,
        'trades_per_30d': n / n_days * 30 if n_days > 0 else 0,
    }


def main():
    print('=' * 100)
    print('C2 First-Pass BT — Funding Z-Score Spread (8 coins)')
    print('=' * 100)
    print(f'Locked: z_lookback {LOCKED["z_lookback_days"]}d, z_entry ±{LOCKED["z_entry_threshold"]}, '
          f'z_exit ±{LOCKED["z_exit_threshold"]}, max_hold {LOCKED["max_hold_periods"]} periods (~7d)')
    print(f'Capital ${LOCKED["capital_usd"]}, friction {LOCKED["maker_friction_pct"]+LOCKED["slippage_pct"]:.2f}% per side')
    print()

    df = load_funding()
    coins = sorted(df['symbol'].unique())
    print(f'Coins: {coins}')
    print(f'Records: {len(df):,}')
    span = (df.timestamp.max() - df.timestamp.min()).total_seconds() / 86400
    print(f'Span: {span:.1f} days, {df.timestamp.min()} → {df.timestamp.max()}\n')

    df_z = compute_zscores(df, LOCKED['z_lookback_days'])
    n_valid_z = df_z['zscore'].notna().sum()
    print(f'Valid z-scores after rolling lookback: {n_valid_z:,}')
    print(f'Z-score distribution: mean={df_z["zscore"].mean():.3f}, std={df_z["zscore"].std():.3f}, '
          f'min={df_z["zscore"].min():.3f}, max={df_z["zscore"].max():.3f}\n')

    print('Simulating...')
    res = simulate(df_z)
    summ = summarize(res, span)

    print('=' * 100)
    print('Results')
    print('=' * 100)
    for k, v in summ.items():
        if isinstance(v, float):
            print(f'  {k:<28} {v:+.4f}')
        else:
            print(f'  {k:<28} {v}')

    if summ['n_completed_trades'] > 0:
        print()
        print('Top 5 trades by PnL:')
        df_t = pd.DataFrame(res['trades'])
        for _, t in df_t.sort_values('net_pnl_pct', ascending=False).head(5).iterrows():
            print(f'  {t["enter_ts"][:10]} → {t["exit_ts"][:10]} '
                  f'(LONG {t["long_coin"][:3]} / SHORT {t["short_coin"][:3]}) '
                  f'{t["periods_held"]}p, net {t["net_pnl_pct"]:+.4f}%, reason {t["reason"]}')
        print()
        print('Bottom 5 trades:')
        for _, t in df_t.sort_values('net_pnl_pct').head(5).iterrows():
            print(f'  {t["enter_ts"][:10]} → {t["exit_ts"][:10]} '
                  f'(LONG {t["long_coin"][:3]} / SHORT {t["short_coin"][:3]}) '
                  f'{t["periods_held"]}p, net {t["net_pnl_pct"]:+.4f}%, reason {t["reason"]}')

    out_path = RESULTS / f'c2_first_pass_bt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'date': datetime.now(timezone.utc).isoformat(),
            'locked': LOCKED, 'summary': summ, 'trades': res['trades'][:50],
        }, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
