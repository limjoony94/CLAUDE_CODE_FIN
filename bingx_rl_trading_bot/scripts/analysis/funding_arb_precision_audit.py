"""Funding Rate Arbitrage Precision Audit.

R5 published +3.28%/yr vs Paper +19.26%/yr 6× gap 정밀 측정.

Variants:
  V1: R5 baseline (entry APY ≥ 3%, exit ≤ 0%)
  V2: Continuous capture (always on, no threshold)
  V3: Continuous + 2025-only period (funding spike year)
  V4: Continuous + low-friction (maker/maker both legs)
  V5: Continuous + leverage 2× (perp full $1500 with margin)

Metrics: net APY, MaxDD, Sharpe, daily mean, bootstrap user 6-criteria
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
from bootstrap_validator import bootstrap_validate

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def load_btc_funding():
    df = pd.read_parquet(DATA / 'funding_history.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    btc = df[df['symbol'] == 'BTC/USDT'].copy()
    btc = btc.sort_values('datetime').reset_index(drop=True)
    btc['rate_pct'] = btc['funding_rate'] * 100  # %
    return btc


def simulate_continuous(df, capital_usd=1500, spot_alloc=0.5, perp_alloc=0.5,
                        spot_friction_pct=0.10, perp_friction_pct=0.04,
                        leverage=1.0, period_filter=None):
    """Continuous delta-neutral capture.

    Spot LONG (spot_alloc * capital), Perp SHORT (perp_alloc * capital × leverage).
    Funding: every 8h, rate_pct * perp_position.
    Entry/exit friction: one-time setup + closeout.
    """
    if period_filter is not None:
        df = df[(df['datetime'] >= period_filter[0]) & (df['datetime'] <= period_filter[1])].copy()

    if len(df) == 0:
        return None

    spot_pos = capital_usd * spot_alloc
    perp_pos = capital_usd * perp_alloc * leverage

    # Funding income per 8h period (perp short receives positive funding)
    # rate is from longs paying shorts, so short collects when rate > 0
    df = df.copy()
    df['funding_income_usd'] = df['rate_pct'] / 100 * perp_pos

    # Entry/exit friction (one-time)
    entry_fric = (spot_friction_pct / 100 * spot_pos) + (perp_friction_pct / 100 * perp_pos)
    exit_fric = entry_fric

    # Total cumulative
    cum_funding = df['funding_income_usd'].sum()
    net_pnl = cum_funding - entry_fric - exit_fric
    net_pnl_pct = net_pnl / capital_usd * 100

    span_start = df['datetime'].min()
    span_end = df['datetime'].max()
    span_days = (span_end - span_start).total_seconds() / 86400
    apy = net_pnl_pct / span_days * 365

    # Daily PnL
    df['date'] = df['datetime'].dt.date
    daily_funding_pct = df.groupby('date')['funding_income_usd'].sum() / capital_usd * 100
    daily_mean = daily_funding_pct.mean()
    daily_std = daily_funding_pct.std()

    # Drawdown
    cum_series = daily_funding_pct.cumsum()
    peak = cum_series.cummax()
    dd = (cum_series - peak)
    max_dd = dd.min()

    # Bootstrap (treat each day with funding as 1 trade for stat)
    nonzero = daily_funding_pct[daily_funding_pct != 0]
    if len(nonzero) >= 5:
        trades_df = pd.DataFrame({
            'close_ts': pd.to_datetime(nonzero.index),
            'gross_pct': nonzero.values + 0.001,  # gross approx (friction tiny)
            'net_pnl_pct': nonzero.values,
        })
        ts_min = trades_df['close_ts'].min()
        ts_max = trades_df['close_ts'].max()
        try:
            res = bootstrap_validate(trades_df, ts_min, ts_max)
            bs = {
                'mean_daily': float(res.mean_daily_pct),
                'p5_daily': float(res.p5_daily_pct),
                'pos_rate': float(res.pos_rate),
                'avg_per_trade': float(res.avg_per_trade_pct),
                'pass_count_6': sum(res.pass_criteria.values()),
                'overall_pass': res.overall_pass,
            }
        except Exception as e:
            bs = {'error': str(e)}
    else:
        bs = None

    return {
        'span_days': float(span_days),
        'cum_funding_pct': float(cum_funding / capital_usd * 100),
        'net_pnl_pct': float(net_pnl_pct),
        'net_apy_pct': float(apy),
        'entry_fric_pct': float(entry_fric / capital_usd * 100),
        'exit_fric_pct': float(exit_fric / capital_usd * 100),
        'daily_mean_pct': float(daily_mean),
        'daily_std_pct': float(daily_std),
        'sharpe_ann': float(daily_mean / max(daily_std, 1e-9) * np.sqrt(365)),
        'max_drawdown_pct': float(max_dd),
        'n_periods': int(len(df)),
        'positive_period_pct': float((df['rate_pct'] > 0).mean() * 100),
        'bootstrap': bs,
    }


def simulate_threshold(df, entry_apy_pct=3.0, exit_apy_pct=0.0, capital_usd=1500,
                        spot_friction_pct=0.10, perp_friction_pct=0.04, leverage=1.0):
    """R5 baseline — threshold-based entry/exit."""
    df = df.copy().sort_values('datetime').reset_index(drop=True)
    spot_pos = capital_usd * 0.5
    perp_pos = capital_usd * 0.5 * leverage

    # 7d trailing APY estimate (3 periods/day × 7d = 21 periods)
    df['trail_apy_pct'] = df['rate_pct'].rolling(21).mean() * 3 * 365
    df['funding_income_usd'] = df['rate_pct'] / 100 * perp_pos

    entry_thr = entry_apy_pct
    exit_thr = exit_apy_pct
    entry_fric = (spot_friction_pct / 100 * spot_pos) + (perp_friction_pct / 100 * perp_pos)

    in_pos = False
    cum_funding = 0
    cum_fric = 0
    n_entries = 0
    daily_pnl_records = []
    span_start = df['datetime'].min()

    for i, row in df.iterrows():
        trail = row['trail_apy_pct']
        date = row['datetime'].date()
        if pd.isna(trail):
            continue
        if not in_pos:
            if trail >= entry_thr:
                in_pos = True
                cum_fric += entry_fric * 2  # entry + future exit
                n_entries += 1
        else:
            cum_funding += row['funding_income_usd']
            if trail <= exit_thr:
                in_pos = False
        daily_pnl_records.append({
            'date': date,
            'funding': row['funding_income_usd'] if in_pos else 0,
        })

    daily_df = pd.DataFrame(daily_pnl_records)
    daily_pnl = daily_df.groupby('date')['funding'].sum() / capital_usd * 100

    span_end = df['datetime'].max()
    span_days = (span_end - span_start).total_seconds() / 86400
    net_pnl = cum_funding - cum_fric
    apy = (net_pnl / capital_usd * 100) / span_days * 365

    return {
        'span_days': float(span_days),
        'n_entries': int(n_entries),
        'cum_funding_usd': float(cum_funding),
        'cum_fric_usd': float(cum_fric),
        'net_pnl_pct': float(net_pnl / capital_usd * 100),
        'net_apy_pct': float(apy),
        'daily_mean_pct': float(daily_pnl.mean()),
        'sharpe_ann': float(daily_pnl.mean() / max(daily_pnl.std(), 1e-9) * np.sqrt(365)),
    }


def main():
    print('=' * 100)
    print('Funding Rate Arbitrage Precision Audit')
    print('=' * 100)
    print('R5 published +3.28%/yr vs Paper +19.26%/yr 6× gap')
    print()

    df = load_btc_funding()
    print(f'BTC funding records: {len(df)}, {df["datetime"].min()} → {df["datetime"].max()}')
    print(f'Mean rate: {df["rate_pct"].mean():.6f}%/8h, '
          f'positive: {(df["rate_pct"] > 0).mean()*100:.1f}%')
    print(f'Theoretical APY (mean × 3/day × 365, no friction):  '
          f'{df["rate_pct"].mean() * 3 * 365:.2f}%/yr\n')

    results = {}

    # V1: R5 baseline
    print('=== V1: R5 baseline (entry APY ≥ 3%, exit ≤ 0%) ===')
    v1 = simulate_threshold(df, entry_apy_pct=3.0, exit_apy_pct=0.0)
    print(f'  net APY: {v1["net_apy_pct"]:+.2f}%/yr')
    print(f'  Daily mean: {v1["daily_mean_pct"]:+.4f}%')
    print(f'  Entries: {v1["n_entries"]}')
    print(f'  Sharpe (ann): {v1["sharpe_ann"]:.3f}')
    results['V1_R5_baseline'] = v1

    # V2: Continuous capture (no threshold)
    print('\n=== V2: Continuous capture (always on, no threshold) ===')
    v2 = simulate_continuous(df, leverage=1.0)
    print(f'  net APY: {v2["net_apy_pct"]:+.2f}%/yr')
    print(f'  cum funding: {v2["cum_funding_pct"]:+.2f}%, friction: -{v2["entry_fric_pct"]+v2["exit_fric_pct"]:.4f}%')
    print(f'  Daily mean: {v2["daily_mean_pct"]:+.4f}%')
    print(f'  MaxDD: {v2["max_drawdown_pct"]:+.2f}%')
    print(f'  Sharpe (ann): {v2["sharpe_ann"]:.3f}')
    if v2['bootstrap']:
        print(f'  Bootstrap pass: {v2["bootstrap"]["pass_count_6"]}/6, '
              f'mean_daily={v2["bootstrap"]["mean_daily"]:+.4f}%')
    results['V2_continuous'] = v2

    # V3: Continuous + 2025-only period
    print('\n=== V3: Continuous + 2025-only ===')
    p_2025 = (pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-12-31', tz='UTC'))
    v3 = simulate_continuous(df, leverage=1.0, period_filter=p_2025)
    if v3:
        print(f'  net APY: {v3["net_apy_pct"]:+.2f}%/yr (2025 only)')
        print(f'  cum: {v3["cum_funding_pct"]:+.2f}%, span: {v3["span_days"]:.0f}d')
        print(f'  Sharpe (ann): {v3["sharpe_ann"]:.3f}')
    results['V3_continuous_2025'] = v3

    # V4: Continuous + low-friction (maker/maker)
    print('\n=== V4: Continuous + maker/maker (0.04% both legs) ===')
    v4 = simulate_continuous(df, leverage=1.0,
                              spot_friction_pct=0.04, perp_friction_pct=0.04)
    print(f'  net APY: {v4["net_apy_pct"]:+.2f}%/yr')
    print(f'  Daily mean: {v4["daily_mean_pct"]:+.4f}%')
    results['V4_maker_maker'] = v4

    # V5: Leverage 2x (perp 2× notional)
    print('\n=== V5: Continuous + leverage 2× (perp $1500 with $750 margin) ===')
    # Capital $1500: spot $750 (long, no leverage) + perp short $1500 (with $750 margin → 2× lev)
    v5 = simulate_continuous(df, capital_usd=1500, spot_alloc=0.5,
                              perp_alloc=1.0, leverage=1.0)  # perp_alloc=1.0 means $1500 notional on $750 margin
    print(f'  net APY: {v5["net_apy_pct"]:+.2f}%/yr (perp 2× leverage)')
    print(f'  Daily mean: {v5["daily_mean_pct"]:+.4f}%')
    results['V5_leverage_2x'] = v5

    # V6: Continuous full notional (capital $1500 fully deployed each leg)
    print('\n=== V6: Full notional ($1500 spot + $1500 perp = $3000 capital) ===')
    # If "capital" means margin only ($1500), and full position = $3000
    # spot $1500 + perp $1500 → return per cycle = rate × $1500
    v6 = simulate_continuous(df, capital_usd=3000, spot_alloc=0.5, perp_alloc=0.5)
    print(f'  net APY (on $3000 capital): {v6["net_apy_pct"]:+.2f}%/yr')
    print(f'  Daily mean: {v6["daily_mean_pct"]:+.4f}%')
    results['V6_full_notional'] = v6

    # ============================================================
    # VERDICT
    # ============================================================
    print('\n' + '=' * 100)
    print('VERDICT — Funding Rate Arb Precision')
    print('=' * 100)
    print(f'\nUser target: +0.20%/day (≈ +73%/yr) OR bank-interest baseline (3-5%/yr)\n')
    print(f'  V1 R5 baseline (threshold 3%/0%):              {v1["net_apy_pct"]:+.2f}%/yr')
    print(f'  V2 Continuous (always on):                      {v2["net_apy_pct"]:+.2f}%/yr')
    if v3:
        print(f'  V3 Continuous + 2025-only:                      {v3["net_apy_pct"]:+.2f}%/yr')
    print(f'  V4 Continuous + maker/maker (low friction):     {v4["net_apy_pct"]:+.2f}%/yr')
    print(f'  V5 Continuous + leverage 2× perp:               {v5["net_apy_pct"]:+.2f}%/yr')
    print(f'  V6 Full notional ($3000 capital):              {v6["net_apy_pct"]:+.2f}%/yr')

    # Reconcile with paper
    print(f'\n  Paper claim 2025: +19.26%/yr')
    print(f'  Best ours: V{max(results, key=lambda k: results[k]["net_apy_pct"] if results[k] else -999)}')

    # Daily target check
    target_daily = 0.20
    print(f'\n  vs +{target_daily}%/day target:')
    for k, v in results.items():
        if v is None:
            continue
        if 'daily_mean_pct' in v:
            d = v['daily_mean_pct']
            ratio = d / target_daily * 100
            status = '✅' if d >= target_daily else '🔴'
            print(f'    {k}: {d:+.4f}%/day = {ratio:.1f}% of target {status}')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'funding rate arb precision audit (R5 vs paper)',
        'btc_funding_stats': {
            'n_records': int(len(df)),
            'mean_rate_8h_pct': float(df['rate_pct'].mean()),
            'positive_pct': float((df['rate_pct'] > 0).mean() * 100),
            'theoretical_apy_no_fric': float(df['rate_pct'].mean() * 3 * 365),
        },
        'variants': results,
    }
    out_path = RESULTS / f'funding_arb_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
