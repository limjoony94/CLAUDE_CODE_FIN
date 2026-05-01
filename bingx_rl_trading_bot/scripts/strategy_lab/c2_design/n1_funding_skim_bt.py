"""N1 First-Pass BT — Just-in-time Funding Skim (selective high-funding entry).

Mechanism:
  - For each coin, monitor funding rate per 8h tick
  - Enter (long perp + short spot OR short perp + long spot) when |rate| > threshold
    - High positive rate → short perp (collect funding) + long spot (hedge)
    - High negative rate → long perp (collect funding) + short spot (hedge)
  - Hold until rate drops below exit_threshold OR max_hold reached
  - Exit at next funding tick

Key insight:
  - Continuous R5 carry harvests average funding (~0.005%/8h)
  - N1 only enters when funding > threshold → higher per-trade gross
  - Trade-off: lower frequency, higher quality

Apply bootstrap_validator: 사용자 success criteria.
"""
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap_validator import bootstrap_validate, report as bootstrap_report


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


# Locked params for first-pass
LOCKED = {
    'capital_usd': 1000,
    'util_pct': 1.0,
    'entry_threshold_pct': 0.04,   # |rate| > 0.04% per 8h to enter
    'exit_threshold_pct': 0.01,    # exit when |rate| < 0.01%
    'max_hold_periods': 21,        # 7d max
    'maker_friction_pct': 0.04,    # per side
    'slippage_pct': 0.02,          # per side
    'leg_count': 2,                # spot + perp
    'side_count': 2,               # entry + exit
}


def load_funding():
    df = pd.read_csv(DATA / 'c2_funding_history.csv', parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['fund_pct'] = df['funding_rate'] * 100   # convert to pct
    return df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def simulate(df: pd.DataFrame, params: dict = None) -> dict:
    """Per-coin: enter when |rate| > threshold, exit when normalize."""
    p = {**LOCKED, **(params or {})}
    capital = p['capital_usd']
    leg_notional = capital * p['util_pct'] / p['leg_count']  # per leg
    fric_per_side = p['maker_friction_pct'] + p['slippage_pct']
    fric_per_trade = p['leg_count'] * p['side_count'] * fric_per_side / 100 * leg_notional

    trades = []

    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        active = None

        for i, row in group.iterrows():
            rate_pct = row['fund_pct']
            ts = row['timestamp']

            if active is None:
                # Entry check
                if abs(rate_pct) > p['entry_threshold_pct']:
                    side = 'short_perp_long_spot' if rate_pct > 0 else 'long_perp_short_spot'
                    active = {
                        'symbol': symbol,
                        'side': side,
                        'enter_ts': ts,
                        'enter_idx': i,
                        'enter_rate_pct': rate_pct,
                        'cum_funding_usd': 0.0,
                    }
            else:
                # Collect funding (one tick × leg_notional)
                # If short perp + long spot: receive funding when rate > 0
                # If long perp + short spot: receive funding when rate < 0
                # Direction: collect if (rate > 0 and side=short_perp) or (rate < 0 and side=long_perp)
                if active['side'] == 'short_perp_long_spot':
                    period_pnl = rate_pct / 100 * leg_notional   # rate × notional
                else:
                    period_pnl = -rate_pct / 100 * leg_notional   # absorb negative
                active['cum_funding_usd'] += period_pnl

                # Exit check
                held = i - active['enter_idx']
                exit_now = False; reason = None
                if held >= p['max_hold_periods']:
                    exit_now, reason = True, 'MAX_HOLD'
                elif abs(rate_pct) < p['exit_threshold_pct']:
                    exit_now, reason = True, 'NORMALIZE'

                if exit_now:
                    net_pnl = active['cum_funding_usd'] - fric_per_trade
                    trades.append({
                        'symbol': symbol, 'side': active['side'],
                        'enter_ts': str(active['enter_ts']),
                        'close_ts': str(ts),
                        'enter_rate_pct': active['enter_rate_pct'],
                        'periods_held': held,
                        'reason': reason,
                        'gross_funding_usd': active['cum_funding_usd'],
                        'net_pnl_usd': net_pnl,
                        'net_pnl_pct': net_pnl / capital * 100,
                        'gross_pct': active['cum_funding_usd'] / capital * 100,
                    })
                    active = None

    return trades


def main():
    print('=' * 100)
    print('N1 First-Pass BT — Just-in-time Funding Skim')
    print('=' * 100)
    print(f'Locked: entry_threshold ±{LOCKED["entry_threshold_pct"]}%, '
          f'exit ±{LOCKED["exit_threshold_pct"]}%, max_hold {LOCKED["max_hold_periods"]} periods')
    print(f'Capital ${LOCKED["capital_usd"]}, '
          f'friction {LOCKED["maker_friction_pct"]+LOCKED["slippage_pct"]:.2f}%/side × '
          f'{LOCKED["leg_count"]} legs × {LOCKED["side_count"]} sides = '
          f'{LOCKED["leg_count"] * LOCKED["side_count"] * (LOCKED["maker_friction_pct"]+LOCKED["slippage_pct"]):.2f}% RT total\n')

    df = load_funding()
    span_start = df.timestamp.min()
    span_end = df.timestamp.max()
    span_days = (span_end - span_start).total_seconds() / 86400
    print(f'Funding history: {len(df):,} records, {span_days:.1f} days, {span_start} → {span_end}')

    # Distribution check
    print(f'\n=== Funding rate distribution per coin ===')
    for sym, g in df.groupby('symbol'):
        ext_rate = (g['fund_pct'].abs() > LOCKED['entry_threshold_pct']).sum()
        print(f'  {sym}: n={len(g)}, |rate|>{LOCKED["entry_threshold_pct"]}%: {ext_rate} ({ext_rate/len(g)*100:.1f}%)')

    # First-pass simulate
    print('\n=== First-pass BT (entry_threshold 0.04%) ===')
    trades = simulate(df)
    if not trades:
        print('  No trades generated')
        return
    df_t = pd.DataFrame(trades)
    cum_pct = float(df_t['net_pnl_pct'].sum())
    apy = cum_pct / span_days * 365
    n = len(df_t)
    print(f'  n_trades: {n}')
    print(f'  cum_net_pct: {cum_pct:+.4f}%')
    print(f'  APY extrapolated: {apy:+.4f}%')
    print(f'  avg_per_trade_net: {df_t["net_pnl_pct"].mean():+.4f}%')
    print(f'  avg_per_trade_gross: {df_t["gross_pct"].mean():+.4f}%')
    print(f'  WR: {(df_t["net_pnl_pct"] > 0).mean():.3f}')
    print(f'  avg_periods_held: {df_t["periods_held"].mean():.1f}')

    # Bootstrap validate
    print('\n=== Bootstrap Validator (사용자 criteria) ===')
    res = bootstrap_validate(df_t, span_start, span_end)
    bootstrap_report(res, 'N1 first-pass')

    # Quick threshold sweep — does any threshold pass user criteria?
    print('\n=== Threshold sweep ===')
    thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    sweep_results = []
    for th in thresholds:
        params = {'entry_threshold_pct': th, 'exit_threshold_pct': th * 0.25}
        ts_trades = simulate(df, params)
        if not ts_trades: continue
        ts_df = pd.DataFrame(ts_trades)
        ts_cum = float(ts_df['net_pnl_pct'].sum())
        ts_apy = ts_cum / span_days * 365
        avg_pt = float(ts_df['net_pnl_pct'].mean())
        avg_pt_gross = float(ts_df['gross_pct'].mean())
        bs = bootstrap_validate(ts_df, span_start, span_end)
        sweep_results.append({
            'threshold': th, 'n_trades': len(ts_df),
            'cum_pct': ts_cum, 'apy': ts_apy,
            'avg_per_trade_net': avg_pt,
            'avg_per_trade_gross': avg_pt_gross,
            'mean_daily': bs.mean_daily_pct,
            'p5_daily': bs.p5_daily_pct,
            'pos_rate': bs.pos_rate,
            'overall_pass': bs.overall_pass,
        })
    print(f'{"thresh":>7} {"n":>5} {"cum%":>9} {"APY%":>9} {"avg_net%":>10} {"avg_gross%":>11} {"mean_d%":>9} {"p5_d%":>8} {"pos_rate":>8} {"PASS":>6}')
    for s in sweep_results:
        pass_mark = '✅' if s['overall_pass'] else '🔴'
        print(f'{s["threshold"]:>7.3f} {s["n_trades"]:>5} '
              f'{s["cum_pct"]:>+8.3f} {s["apy"]:>+8.3f} '
              f'{s["avg_per_trade_net"]:>+9.4f} {s["avg_per_trade_gross"]:>+10.4f} '
              f'{s["mean_daily"]:>+8.4f} {s["p5_daily"]:>+7.4f} '
              f'{s["pos_rate"]:>8.3f} {pass_mark:>6}')

    # Save
    out_path = RESULTS / f'n1_first_pass_bt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'locked': LOCKED,
        'first_pass': {
            'n_trades': n, 'cum_pct': cum_pct, 'apy': apy,
            'avg_per_trade_net': float(df_t['net_pnl_pct'].mean()),
            'avg_per_trade_gross': float(df_t['gross_pct'].mean()),
            'wr': float((df_t['net_pnl_pct'] > 0).mean()),
            'mean_daily_bootstrap': res.mean_daily_pct,
            'p5_daily_bootstrap': res.p5_daily_pct,
            'overall_pass': res.overall_pass,
            'pass_criteria': res.pass_criteria,
        },
        'threshold_sweep': sweep_results,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
