"""Absolute Ceiling — Per-bar perfect timing (mechanism-free).

사용자 질문 답: "왜 과적합인데 +2.87%/day밖에? 발산해야 하는 것 아닌가?"

L1-L8까지의 결과는 모두 "8 mechanism × best-IS config × cherry-pick" — mechanism-bound.
진짜 데이터-level absolute ceiling은 mechanism free:

Bar-level perfect timing:
  매 1h bar에서 next bar return이 양수면 LONG, 음수면 SHORT (perfect look-ahead)
  Daily PnL = sum of |hourly returns| × 24h - friction × 24trades

이건 BTC 1h volatility의 absolute upper bound.
"Why not infinite": 한 timestep의 perfect timing은 그 bar의 return으로 capped.

Multiple timeframes:
  1h: 17,280 bars / 720d
  15m: 69,120 bars
  5m: 207,360 bars

더 짧은 timeframe = more bars = more cherry-pick = higher ceiling.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def perfect_bar_timing(df, friction_per_trade_pct=0.10):
    """매 bar마다 next bar return이 양수면 LONG, 음수면 SHORT.

    Returns:
        gross_daily_mean: friction 없는 daily mean
        net_daily_mean: friction 있는 daily mean (각 bar마다 trade 1번)
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    close = df['close'].values
    # next-bar return
    next_ret = np.diff(close) / close[:-1] * 100  # %
    # Perfect timing: take |next_ret| at each bar (always profitable direction)
    perfect_pnl = np.abs(next_ret)
    # Friction per trade
    net_pnl = perfect_pnl - friction_per_trade_pct

    span_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    gross_daily = perfect_pnl.sum() / span_days
    net_daily = net_pnl.sum() / span_days
    n_bars = len(perfect_pnl)
    avg_per_bar = perfect_pnl.mean()

    return {
        'span_days': float(span_days),
        'n_bars': int(n_bars),
        'avg_per_bar_gross': float(avg_per_bar),
        'avg_per_bar_net': float(avg_per_bar - friction_per_trade_pct),
        'gross_daily_mean': float(gross_daily),
        'net_daily_mean': float(net_daily),
        'cum_gross': float(perfect_pnl.sum()),
        'cum_net': float(net_pnl.sum()),
    }


def main():
    print('=' * 100)
    print('Absolute Ceiling — Per-bar Perfect Timing (mechanism-free)')
    print('=' * 100)
    print('사용자 질문: "왜 과적합인데 +2.87%/day밖에? 발산해야 하는 것 아닌가?"')
    print()

    results = {}

    # 1h
    print('=== 1h timeframe ===')
    df_1h = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    r_1h = perfect_bar_timing(df_1h, friction_per_trade_pct=0.10)
    print(f'  bars: {r_1h["n_bars"]:,}, span: {r_1h["span_days"]:.1f}d')
    print(f'  avg per-bar |return|: {r_1h["avg_per_bar_gross"]:.4f}%')
    print(f'  Perfect gross daily: {r_1h["gross_daily_mean"]:+.4f}%')
    print(f'  Perfect net daily (-0.10% per trade): {r_1h["net_daily_mean"]:+.4f}%')
    results['1h'] = r_1h

    # 15m
    print('\n=== 15m timeframe ===')
    df_15m = pd.read_csv(DATA / 'btc_15m_720days.csv', parse_dates=['timestamp'])
    r_15m = perfect_bar_timing(df_15m, friction_per_trade_pct=0.10)
    print(f'  bars: {r_15m["n_bars"]:,}, span: {r_15m["span_days"]:.1f}d')
    print(f'  avg per-bar |return|: {r_15m["avg_per_bar_gross"]:.4f}%')
    print(f'  Perfect gross daily: {r_15m["gross_daily_mean"]:+.4f}%')
    print(f'  Perfect net daily (-0.10% per trade): {r_15m["net_daily_mean"]:+.4f}%')
    results['15m'] = r_15m

    # 5m
    print('\n=== 5m timeframe ===')
    df_5m = pd.read_csv(DATA / 'btc_5m_720days_binance.csv', parse_dates=['timestamp'])
    r_5m = perfect_bar_timing(df_5m, friction_per_trade_pct=0.10)
    print(f'  bars: {r_5m["n_bars"]:,}, span: {r_5m["span_days"]:.1f}d')
    print(f'  avg per-bar |return|: {r_5m["avg_per_bar_gross"]:.4f}%')
    print(f'  Perfect gross daily: {r_5m["gross_daily_mean"]:+.4f}%')
    print(f'  Perfect net daily (-0.10% per trade): {r_5m["net_daily_mean"]:+.4f}%')
    results['5m'] = r_5m

    # Daily
    print('\n=== Daily timeframe ===')
    df_d = df_1h.copy()
    df_d['date'] = df_d['timestamp'].dt.normalize()
    df_d_agg = df_d.groupby('date').agg({'close': 'last'}).reset_index()
    df_d_agg.rename(columns={'date': 'timestamp'}, inplace=True)
    r_d = perfect_bar_timing(df_d_agg, friction_per_trade_pct=0.10)
    print(f'  bars: {r_d["n_bars"]:,}, span: {r_d["span_days"]:.1f}d')
    print(f'  avg per-bar |return|: {r_d["avg_per_bar_gross"]:.4f}%')
    print(f'  Perfect gross daily: {r_d["gross_daily_mean"]:+.4f}%')
    print(f'  Perfect net daily: {r_d["net_daily_mean"]:+.4f}%')
    results['1d'] = r_d

    # Comparison
    print('\n' + '=' * 100)
    print('FULL CEILING SPECTRUM — 사용자 질문 답')
    print('=' * 100)
    print('\nUser target: +0.20%/day')
    print()
    print('  Realistic (causal, no look-ahead):')
    print('    Online learning rolling weight:                    +0.0906%/day  (4.7% of L2)')
    print()
    print('  Weak overfit (32 sweep best-IS configs):')
    print('    L1 single mechanism in-sample best:               ~+0.30%/day')
    print('    L3a fixed-weight max-mean:                         +0.2338%/day  (R2b 100%)')
    print('    L3b max-Sharpe:                                    +0.0555%/day')
    print('    L3c long-short:                                    +0.4745%/day  (extreme leverage)')
    print()
    print('  Medium overfit (mechanism-level cherry-pick):')
    print('    L4 weekly best-mech hindsight:                     +0.9182%/day')
    print('    L2 per-day mech switcher:                          +1.8975%/day')
    print()
    print('  Strong overfit (per-trade cherry-pick within 8 mech):')
    print('    L5 per-mech winners-only sum across 8:             +2.8726%/day')
    print('    L6 per-day BEST winner across all 8:               +2.0373%/day')
    print('    L7 per-day SUM all winners across all 8:           +2.8726%/day')
    print()
    print('  Absolute ceiling (mechanism-free, perfect bar timing):')
    print(f'    Daily timeframe (perfect direction call/day):      {results["1d"]["gross_daily_mean"]:+.2f}%/day gross,  {results["1d"]["net_daily_mean"]:+.2f}% net')
    print(f'    1h timeframe   (perfect direction call/hour):      {results["1h"]["gross_daily_mean"]:+.2f}%/day gross,  {results["1h"]["net_daily_mean"]:+.2f}% net')
    print(f'    15m timeframe  (perfect direction call/15min):     {results["15m"]["gross_daily_mean"]:+.2f}%/day gross, {results["15m"]["net_daily_mean"]:+.2f}% net')
    print(f'    5m timeframe   (perfect direction call/5min):      {results["5m"]["gross_daily_mean"]:+.2f}%/day gross, {results["5m"]["net_daily_mean"]:+.2f}% net')

    print('\n=== 답: 왜 발산 안 하는가? ===')
    print()
    print('  1. BTC 1h volatility (per-bar |return|)이 자연스럽게 ceiling 결정')
    print(f'     - 1h avg per-bar |return|: {r_1h["avg_per_bar_gross"]:.4f}%')
    print(f'     - 1d 24bar × {r_1h["avg_per_bar_gross"]:.3f}% = {r_1h["avg_per_bar_gross"] * 24:.2f}%/day (perfect 1h timing)')
    print(f'     - 5m: 288bar × {r_5m["avg_per_bar_gross"]:.3f}% = {r_5m["avg_per_bar_gross"] * 288:.2f}%/day (perfect 5m timing)')
    print()
    print('  2. 진짜 무한대는 timeframe 1초 또는 tick-level perfect timing')
    print('     - 더 짧은 timeframe = 더 많은 trades × 더 작은 |return| × cumulative > sum')
    print()
    print('  3. 실용적 friction은 모든 timeframe에서 ceiling을 깎음:')
    for tf in ['1d', '1h', '15m', '5m']:
        r = results[tf]
        gross = r['gross_daily_mean']
        net = r['net_daily_mean']
        decay = (gross - net) / gross * 100 if gross > 0 else 0
        print(f'     - {tf}: gross {gross:+.2f}% → net {net:+.2f}% ({decay:.1f}% decay)')
    print()
    print('  4. Mechanism-bound (L1-L7) vs mechanism-free (perfect bar) gap:')
    l7 = 2.8726
    l_5m_gross = results['5m']['gross_daily_mean']
    print(f'     L7 (mechanism cherry-pick): +{l7:.2f}%/day')
    print(f'     5m perfect bar timing (gross): +{l_5m_gross:.2f}%/day')
    print(f'     Gap = {l_5m_gross - l7:.2f}%/day = mechanism이 못 잡는 timing')
    print()
    print('  5. 결론: 사용자 직관 부분 맞음 — "더 강한 overfit으로 더 큰 ceiling 가능"')
    print(f'     Strong mechanism overfit (L7) = +2.87%/day')
    print(f'     Absolute bar-timing ceiling (1h gross) = {r_1h["gross_daily_mean"]:+.2f}%/day')
    print(f'     1초/tick perfect = theoretically infinite (with infinite trade count)')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'absolute ceiling per-bar perfect timing — 사용자 질문 답',
        'results_by_timeframe': results,
    }
    out_path = RESULTS / f'absolute_ceiling_bar_timing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
