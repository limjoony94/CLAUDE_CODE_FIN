"""Risk-Adjusted Potential Evaluation — 사용자 요청.

사용자: "전략의 포텐셜 측정 (잠재 수익률) + 잠재 리스크 측정 → 최적값 평가"

Framework:
  Upside (잠재 수익률):
    - avg_gross_per_trade (mechanism edge)
    - daily_mean_pnl (in-sample)
    - cum_pnl_total (720d)
    - bootstrap p95_daily (best 5% windows)
    - max single-day return

  Downside (잠재 리스크):
    - bootstrap p5_daily (worst 5%)
    - max drawdown
    - max single-day loss
    - WR + R:R asymmetry
    - distribution stability fail count

  Risk-adjusted (composite):
    - Sharpe ratio (annualized)
    - Sortino ratio (downside vol)
    - Calmar ratio (annual return / max DD)
    - Profit factor
    - User 6-criteria pass count

  Composite score = weighted combination → optimal recommendation
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))

from bootstrap_validator import bootstrap_validate

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

ANNUAL_DAYS = 365


def load_pnl_df():
    """8 mechanism daily PnL DataFrame."""
    from d3_portfolio_simulation import (
        BEST_CONFIGS, get_btc_1h, sim_r8b, sim_r37b, sim_r40b,
        simulate_with_signals, range_expansion_signals, volume_spike_signals,
        run_xs_momentum, load_pivot, sim_n8b, get_macro_data, run_xs_reversal,
        trades_to_daily_pnl,
    )
    df_1h = get_btc_1h()
    span_min = df_1h['timestamp'].min().normalize()
    span_max = df_1h['timestamp'].max().normalize()
    if span_min.tz is None:
        span_min = span_min.tz_localize('UTC')
        span_max = span_max.tz_localize('UTC')
    date_index = pd.date_range(span_min, span_max, freq='1D', tz='UTC')

    s = {}
    s['R8b'] = trades_to_daily_pnl(sim_r8b(df_1h, BEST_CONFIGS['R8b']), date_index)
    s['R37b'] = trades_to_daily_pnl(sim_r37b(df_1h, BEST_CONFIGS['R37b']), date_index)
    s['R40b'] = trades_to_daily_pnl(sim_r40b(df_1h, BEST_CONFIGS['R40b']), date_index)
    s['Range'] = trades_to_daily_pnl(simulate_with_signals(df_1h, range_expansion_signals(df_1h, BEST_CONFIGS['Range']), BEST_CONFIGS['Range']), date_index)
    s['VolSpike'] = trades_to_daily_pnl(simulate_with_signals(df_1h, volume_spike_signals(df_1h, BEST_CONFIGS['VolSpike']), BEST_CONFIGS['VolSpike']), date_index)
    prices = load_pivot()
    s['R1b'] = trades_to_daily_pnl(run_xs_momentum(prices, BEST_CONFIGS['R1b']), date_index)
    s['R2b'] = trades_to_daily_pnl(run_xs_reversal(prices, BEST_CONFIGS['R2b']), date_index)
    macro_full = get_macro_data()
    s['N8b'] = trades_to_daily_pnl(sim_n8b(macro_full, BEST_CONFIGS['N8b']), date_index)
    return pd.DataFrame(s)


def compute_metrics(pnl_series, name):
    """Comprehensive risk-adjusted metrics for a single mechanism/portfolio."""
    s = pnl_series.dropna()
    n_days = len(s)
    nonzero = s[s != 0]
    n_active = len(nonzero)

    if n_active < 10:
        return {'name': name, 'n_active_days': n_active, 'insufficient': True}

    daily_mean = s.mean()
    daily_std = s.std()
    cum = s.sum()
    annual_return = daily_mean * ANNUAL_DAYS

    # Drawdown
    eq = (1 + s / 100).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe
    sharpe_ann = daily_mean / max(daily_std, 1e-9) * np.sqrt(ANNUAL_DAYS)

    # Sortino (downside vol)
    downside = s[s < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-9
    sortino_ann = daily_mean / max(downside_std, 1e-9) * np.sqrt(ANNUAL_DAYS)

    # Calmar
    calmar = annual_return / max(abs(max_dd), 1e-9) if max_dd < 0 else 0

    # Tail risk
    p5 = np.percentile(s, 5)
    p95 = np.percentile(s, 95)
    p1 = np.percentile(s, 1)
    p99 = np.percentile(s, 99)

    # WR + profit factor
    winners = nonzero[nonzero > 0]
    losers = nonzero[nonzero < 0]
    wr = len(winners) / n_active if n_active > 0 else 0
    sum_w = winners.sum()
    sum_l = losers.sum()
    profit_factor = abs(sum_w / sum_l) if sum_l < 0 else float('inf')
    avg_w = winners.mean() if len(winners) > 0 else 0
    avg_l = losers.mean() if len(losers) > 0 else 0
    rr = abs(avg_w / avg_l) if avg_l < 0 else float('inf')

    # Bootstrap 6-criteria (only on nonzero days)
    trades_df = pd.DataFrame({
        'close_ts': nonzero.index,
        'gross_pct': nonzero.values + 0.07,
        'net_pnl_pct': nonzero.values,
    })
    trades_df['close_ts'] = pd.to_datetime(trades_df['close_ts'])
    span_min = trades_df['close_ts'].min()
    span_max = trades_df['close_ts'].max()
    res = bootstrap_validate(trades_df, span_min, span_max)
    bs_pass_count = sum(res.pass_criteria.values()) if res.pass_criteria else 0

    # Composite score (사용자 framing: 잠재 수익 / 잠재 리스크 균형)
    # Higher is better
    # - Sharpe weight 0.3 (risk-adjusted return)
    # - Sortino weight 0.2 (downside-adjusted)
    # - Calmar weight 0.2 (DD-adjusted)
    # - Profit factor weight 0.15
    # - Bootstrap pass count weight 0.15
    composite = (
        0.30 * np.tanh(sharpe_ann / 2) +    # bound to (-1, 1)
        0.20 * np.tanh(sortino_ann / 2) +
        0.20 * np.tanh(calmar / 2) +
        0.15 * np.tanh((profit_factor - 1) / 2) +   # 1.0 = neutral
        0.15 * (bs_pass_count / 6)
    )

    return {
        'name': name,
        'n_days': int(n_days),
        'n_active_days': int(n_active),
        # Upside
        'daily_mean_pct': float(daily_mean),
        'annual_return_pct': float(annual_return),
        'cum_pct': float(cum),
        'p95_daily': float(p95),
        'p99_daily': float(p99),
        'max_day_pct': float(s.max()),
        'WR': float(wr),
        'avg_winner_pct': float(avg_w),
        'sum_winners_pct': float(sum_w),
        # Downside
        'daily_std_pct': float(daily_std),
        'p5_daily': float(p5),
        'p1_daily': float(p1),
        'min_day_pct': float(s.min()),
        'max_drawdown_pct': float(max_dd),
        'avg_loser_pct': float(avg_l),
        'sum_losers_pct': float(sum_l),
        # Risk-adjusted
        'sharpe_ann': float(sharpe_ann),
        'sortino_ann': float(sortino_ann),
        'calmar': float(calmar),
        'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.0,
        'rr_ratio': float(rr) if rr != float('inf') else 999.0,
        # Bootstrap
        'bootstrap_mean_daily': float(res.mean_daily_pct),
        'bootstrap_pos_rate': float(res.pos_rate),
        'bootstrap_p5_daily': float(res.p5_daily_pct),
        'bootstrap_avg_per_trade': float(res.avg_per_trade_pct),
        'bootstrap_pass_count_6': int(bs_pass_count),
        'bootstrap_overall_pass': bool(res.overall_pass),
        # Composite
        'composite_score': float(composite),
    }


def main():
    print('=' * 100)
    print('Risk-Adjusted Potential Evaluation — 사용자 framing')
    print('=' * 100)
    print('Upside (잠재 수익률) + Downside (잠재 리스크) + Risk-adjusted composite')
    print()

    pnl_df = load_pnl_df()
    print(f'Daily PnL DataFrame: {pnl_df.shape}')

    # Per-mechanism
    print('\n=== Computing metrics for 8 mechanisms ===')
    results = []
    for col in pnl_df.columns:
        m = compute_metrics(pnl_df[col], col)
        results.append(m)
        if not m.get('insufficient'):
            print(f'  {col} done')

    # Portfolios
    print('\n=== Computing metrics for portfolios ===')
    # Equal-weight
    ew = pnl_df.mean(axis=1)
    results.append(compute_metrics(ew, 'Portfolio_EW'))
    print('  Portfolio_EW done')

    # Risk-parity
    vols = pnl_df.std()
    inv_vol = 1.0 / vols.replace(0, np.nan)
    weights_rp = inv_vol / inv_vol.sum()
    rp = (pnl_df * weights_rp).sum(axis=1)
    results.append(compute_metrics(rp, 'Portfolio_RP'))
    print('  Portfolio_RP done')

    # Top-3 low corr (R40b/Range/R1b)
    top3 = pnl_df[['R40b', 'Range', 'R1b']].mean(axis=1)
    results.append(compute_metrics(top3, 'Portfolio_Top3LowCorr'))
    print('  Portfolio_Top3LowCorr done')

    # Sort by composite
    results.sort(key=lambda r: r.get('composite_score', -999), reverse=True)

    # Print summary table
    print('\n' + '=' * 100)
    print('UPSIDE METRICS (잠재 수익률)')
    print('=' * 100)
    print(f"{'Strategy':<22} {'Daily':>9} {'Annual':>9} {'Cum720d':>10} {'WR':>6} {'AvgW':>8} {'P95':>8} {'MaxDay':>8}")
    print('-' * 100)
    for r in results:
        if r.get('insufficient'):
            continue
        print(f"{r['name']:<22} "
              f"{r['daily_mean_pct']:>+8.4f}% "
              f"{r['annual_return_pct']:>+8.2f}% "
              f"{r['cum_pct']:>+9.2f}% "
              f"{r['WR']:>6.3f} "
              f"{r['avg_winner_pct']:>+7.3f}% "
              f"{r['p95_daily']:>+7.3f}% "
              f"{r['max_day_pct']:>+7.2f}%")

    print('\n' + '=' * 100)
    print('DOWNSIDE METRICS (잠재 리스크)')
    print('=' * 100)
    print(f"{'Strategy':<22} {'DailyStd':>9} {'MaxDD':>8} {'P5':>8} {'P1':>8} {'MinDay':>8} {'AvgL':>8} {'L/W ratio':>10}")
    print('-' * 100)
    for r in results:
        if r.get('insufficient'):
            continue
        lw = r['avg_loser_pct'] / r['avg_winner_pct'] if r['avg_winner_pct'] > 0 else float('inf')
        print(f"{r['name']:<22} "
              f"{r['daily_std_pct']:>8.4f}% "
              f"{r['max_drawdown_pct']:>+7.2f}% "
              f"{r['p5_daily']:>+7.3f}% "
              f"{r['p1_daily']:>+7.3f}% "
              f"{r['min_day_pct']:>+7.2f}% "
              f"{r['avg_loser_pct']:>+7.3f}% "
              f"{lw:>10.3f}")

    print('\n' + '=' * 100)
    print('RISK-ADJUSTED METRICS')
    print('=' * 100)
    print(f"{'Strategy':<22} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'PF':>6} {'R:R':>6} {'BS pass':>8} {'Composite':>10}")
    print('-' * 100)
    for r in results:
        if r.get('insufficient'):
            continue
        print(f"{r['name']:<22} "
              f"{r['sharpe_ann']:>8.3f} "
              f"{r['sortino_ann']:>9.3f} "
              f"{r['calmar']:>8.3f} "
              f"{r['profit_factor']:>6.2f} "
              f"{r['rr_ratio']:>6.2f} "
              f"{r['bootstrap_pass_count_6']:>2d}/6   "
              f"{r['composite_score']:>+10.4f}")

    # Top recommendation
    print('\n' + '=' * 100)
    print('TOP RECOMMENDATION (가장 균형 잡힌 strategy)')
    print('=' * 100)
    top = results[0]
    print(f"\n  🥇 #1: {top['name']}")
    print(f"     Composite score: {top['composite_score']:+.4f}")
    print(f"     Daily mean: {top['daily_mean_pct']:+.4f}% (annual {top['annual_return_pct']:+.2f}%)")
    print(f"     Sharpe (ann): {top['sharpe_ann']:.3f}, Sortino: {top['sortino_ann']:.3f}, Calmar: {top['calmar']:.3f}")
    print(f"     MaxDD: {top['max_drawdown_pct']:+.2f}%, P5 daily: {top['p5_daily']:+.3f}%")
    print(f"     Profit factor: {top['profit_factor']:.2f}, WR: {top['WR']:.3f}")
    bs_overall_str = "✅" if top["bootstrap_overall_pass"] else "🔴"
    print(f"     Bootstrap 6-criteria: {top['bootstrap_pass_count_6']}/6 PASS, Overall: {bs_overall_str}")

    if len(results) >= 2:
        rk2 = results[1]
        print(f"\n  🥈 #2: {rk2['name']} (composite {rk2['composite_score']:+.4f})")
    if len(results) >= 3:
        rk3 = results[2]
        print(f"  🥉 #3: {rk3['name']} (composite {rk3['composite_score']:+.4f})")

    # Best in each category
    print('\n=== Best in each category ===')
    best_upside = max(results, key=lambda r: r.get('annual_return_pct', -999) if not r.get('insufficient') else -999)
    best_sharpe = max(results, key=lambda r: r.get('sharpe_ann', -999) if not r.get('insufficient') else -999)
    best_dd = max(results, key=lambda r: r.get('max_drawdown_pct', -999) if not r.get('insufficient') else -999)
    best_pf = max(results, key=lambda r: r.get('profit_factor', -999) if not r.get('insufficient') else -999)
    print(f'  Highest upside (annual return): {best_upside["name"]} = {best_upside["annual_return_pct"]:+.2f}%')
    print(f'  Best Sharpe ratio:              {best_sharpe["name"]} = {best_sharpe["sharpe_ann"]:.3f}')
    print(f'  Lowest MaxDD (least risk):      {best_dd["name"]} = {best_dd["max_drawdown_pct"]:+.2f}%')
    print(f'  Best profit factor:             {best_pf["name"]} = {best_pf["profit_factor"]:.2f}')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'risk-adjusted potential evaluation',
        'composite_weights': {
            'sharpe': 0.30, 'sortino': 0.20, 'calmar': 0.20,
            'profit_factor': 0.15, 'bootstrap_pass_6': 0.15,
        },
        'rankings_by_composite': [
            {'rank': i+1, 'name': r['name'], 'composite_score': r['composite_score']}
            for i, r in enumerate([rr for rr in results if not rr.get('insufficient')])
        ],
        'detailed_metrics': results,
        'category_winners': {
            'highest_upside': best_upside['name'],
            'best_sharpe': best_sharpe['name'],
            'lowest_drawdown': best_dd['name'],
            'best_profit_factor': best_pf['name'],
        },
    }
    out_path = RESULTS / f'risk_adjusted_potential_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
