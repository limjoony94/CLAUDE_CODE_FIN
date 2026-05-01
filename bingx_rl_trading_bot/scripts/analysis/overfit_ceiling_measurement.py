"""Overfit Ceiling Measurement — L1/L2/L3 (사용자 mandate 2026-05-01).

User: "극과적합 모델 develop해서 potential 측정"

Diagnostic question: envelope 한계인가, generalization framework가 너무 strict한가?

Three levels of in-sample maximization:

L1: Naive in-sample best
  - 32 sweep 결과 best-IS daily means
  - 이미 done — R2b/N8b ~+0.30%/day 등
  - Sweep 자체가 already in-sample optimization, 이게 baseline

L2: Hindsight per-day mechanism switcher (PERFECT LOOK-AHEAD)
  - 매일 8 mechanism 중 PnL 가장 높은 것 선택 (look-ahead 허용)
  - 데이터 자체가 허용하는 ABSOLUTE CEILING
  - L2 < +0.20%/day → envelope 진짜 empty at data level
  - L2 > +0.50%/day → mechanism mixing potential 있음, framework 재검토

L3: Full-sample weight optimization (REALISTIC OVERFIT)
  - 8 mechanism의 weight를 in-sample에 직접 optimize (mean 또는 Sharpe)
  - Per-day switching은 안 함, fixed weight
  - Realistic overfit ceiling (heavy in-sample fitting)

8 mechanism daily PnL DataFrame from D-3 simulation reuse.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def load_d3_pnl():
    """Load latest D-3 simulation result + reconstruct daily PnL DataFrame."""
    # Re-run D-3 simulation to get pnl_df (faster than rebuilding)
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

    daily_pnl_series = {}
    print('Building 8-mechanism daily PnL...')
    daily_pnl_series['R8b'] = trades_to_daily_pnl(sim_r8b(df_1h, BEST_CONFIGS['R8b']), date_index)
    daily_pnl_series['R37b'] = trades_to_daily_pnl(sim_r37b(df_1h, BEST_CONFIGS['R37b']), date_index)
    daily_pnl_series['R40b'] = trades_to_daily_pnl(sim_r40b(df_1h, BEST_CONFIGS['R40b']), date_index)
    daily_pnl_series['Range'] = trades_to_daily_pnl(
        simulate_with_signals(df_1h, range_expansion_signals(df_1h, BEST_CONFIGS['Range']), BEST_CONFIGS['Range']),
        date_index)
    daily_pnl_series['VolSpike'] = trades_to_daily_pnl(
        simulate_with_signals(df_1h, volume_spike_signals(df_1h, BEST_CONFIGS['VolSpike']), BEST_CONFIGS['VolSpike']),
        date_index)
    prices = load_pivot()
    daily_pnl_series['R1b'] = trades_to_daily_pnl(run_xs_momentum(prices, BEST_CONFIGS['R1b']), date_index)
    daily_pnl_series['R2b'] = trades_to_daily_pnl(run_xs_reversal(prices, BEST_CONFIGS['R2b']), date_index)
    macro_full = get_macro_data()
    daily_pnl_series['N8b'] = trades_to_daily_pnl(sim_n8b(macro_full, BEST_CONFIGS['N8b']), date_index)

    return pd.DataFrame(daily_pnl_series)


def main():
    print('=' * 100)
    print('Overfit Ceiling Measurement — L1/L2/L3')
    print('=' * 100)

    pnl_df = load_d3_pnl()
    print(f'\nDaily PnL DataFrame: {pnl_df.shape}')
    print(f'Span: {pnl_df.index.min()} → {pnl_df.index.max()}')
    print(f'Mechanisms: {list(pnl_df.columns)}\n')

    # Per-mechanism summary
    print('=== L1: Naive in-sample best per mechanism (already done in 32 sweep) ===')
    for col in pnl_df.columns:
        s = pnl_df[col]
        nonzero = s[s != 0]
        print(f'  {col}: daily mean={s.mean():+.4f}%, std={s.std():.4f}%, '
              f'active days={len(nonzero)}, '
              f'mean on active days={nonzero.mean() if len(nonzero) > 0 else 0:+.4f}%')

    n_days = len(pnl_df)
    print(f'\nTotal days: {n_days}\n')

    # ============================================================
    # L2: Hindsight per-day mechanism switcher
    # ============================================================
    print('=' * 100)
    print('L2: HINDSIGHT PER-DAY MECHANISM SWITCHER (perfect look-ahead)')
    print('=' * 100)

    # Per day, pick the best mechanism
    l2_daily_max = pnl_df.max(axis=1)  # best PnL each day
    l2_mean = l2_daily_max.mean()
    l2_std = l2_daily_max.std()
    l2_pos_rate = (l2_daily_max > 0).mean()
    l2_p5 = np.percentile(l2_daily_max, 5)
    l2_cum = l2_daily_max.sum()
    print(f'  L2 daily mean: {l2_mean:+.4f}%')
    print(f'  L2 daily std:  {l2_std:.4f}%')
    print(f'  L2 pos_rate:   {l2_pos_rate:.3f}')
    print(f'  L2 p5_daily:   {l2_p5:+.4f}%')
    print(f'  L2 cum:        {l2_cum:+.2f}% over {n_days}d')
    print(f'  L2 Sharpe (ann): {l2_mean / max(l2_std, 1e-9) * np.sqrt(365):.3f}')

    # Per-day winner distribution
    winners = pnl_df.idxmax(axis=1)
    winner_counts = winners.value_counts()
    print(f'\n  Winner distribution (which mechanism best each day):')
    for m, c in winner_counts.items():
        print(f'    {m}: {c} days ({c/n_days*100:.1f}%)')

    # Friction adjustment for L2 (hindsight switcher)
    # Each day switching mechanism = trade exit + new entry. Conservative friction 0.10% per switch
    switches = (winners != winners.shift(1)).sum() - 1
    fric_per_switch = 0.10  # conservative 0.10% switch cost
    fric_total = switches * fric_per_switch
    fric_per_day = fric_total / n_days
    print(f'\n  Switching friction adjustment:')
    print(f'    Switches: {switches} / {n_days} days ({switches/n_days*100:.1f}%)')
    print(f'    Total friction: -{fric_total:.2f}%')
    print(f'    Per-day friction: -{fric_per_day:.4f}%')
    l2_mean_fric = l2_mean - fric_per_day
    print(f'    L2 daily mean (post-switching friction): {l2_mean_fric:+.4f}%')

    # ============================================================
    # L3: Full-sample weight optimization
    # ============================================================
    print('\n' + '=' * 100)
    print('L3: FULL-SAMPLE WEIGHT OPTIMIZATION (realistic overfit)')
    print('=' * 100)

    pnl_arr = pnl_df.values
    n_mech = pnl_arr.shape[1]

    def neg_mean(w):
        port = (pnl_arr * w).sum(axis=1)
        return -port.mean()

    def neg_sharpe(w):
        port = (pnl_arr * w).sum(axis=1)
        std = port.std()
        if std < 1e-9:
            return 0
        return -(port.mean() / std)

    # Constraints: weights sum to 1, all >= 0 (long-only weight)
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n_mech
    x0 = np.ones(n_mech) / n_mech

    print('\n--- L3a: Maximize daily mean ---')
    res_mean = minimize(neg_mean, x0, method='SLSQP', constraints=cons, bounds=bounds)
    w_mean = res_mean.x
    port_mean = (pnl_arr * w_mean).sum(axis=1)
    print(f'  Weights:')
    for col, w in zip(pnl_df.columns, w_mean):
        print(f'    {col}: {w:.4f}')
    print(f'  Portfolio daily mean: {port_mean.mean():+.4f}%')
    print(f'  Portfolio daily std:  {port_mean.std():.4f}%')
    print(f'  Portfolio Sharpe (ann): {port_mean.mean() / max(port_mean.std(), 1e-9) * np.sqrt(365):.3f}')
    print(f'  Portfolio cum: {port_mean.sum():+.2f}% over {n_days}d')

    print('\n--- L3b: Maximize Sharpe (annualized) ---')
    res_sh = minimize(neg_sharpe, x0, method='SLSQP', constraints=cons, bounds=bounds)
    w_sh = res_sh.x
    port_sh = (pnl_arr * w_sh).sum(axis=1)
    print(f'  Weights:')
    for col, w in zip(pnl_df.columns, w_sh):
        print(f'    {col}: {w:.4f}')
    print(f'  Portfolio daily mean: {port_sh.mean():+.4f}%')
    print(f'  Portfolio daily std:  {port_sh.std():.4f}%')
    print(f'  Portfolio Sharpe (ann): {port_sh.mean() / max(port_sh.std(), 1e-9) * np.sqrt(365):.3f}')

    # L3c: Long-short (allow negative weights)
    print('\n--- L3c: Maximize daily mean with long-short (-1 ≤ w ≤ 1, sum=1) ---')
    bounds_ls = [(-1, 1)] * n_mech
    res_ls = minimize(neg_mean, x0, method='SLSQP', constraints=cons, bounds=bounds_ls)
    w_ls = res_ls.x
    port_ls = (pnl_arr * w_ls).sum(axis=1)
    print(f'  Weights:')
    for col, w in zip(pnl_df.columns, w_ls):
        print(f'    {col}: {w:+.4f}')
    print(f'  Portfolio daily mean: {port_ls.mean():+.4f}%')
    print(f'  Portfolio daily std:  {port_ls.std():.4f}%')
    print(f'  Portfolio Sharpe (ann): {port_ls.mean() / max(port_ls.std(), 1e-9) * np.sqrt(365):.3f}')

    # L4 (extra): Per-week dynamic re-weighting (perfect hindsight)
    print('\n--- L4 (extra): Per-week dynamic re-weighting (perfect hindsight) ---')
    pnl_df_indexed = pnl_df.copy()
    pnl_df_indexed['week'] = pnl_df_indexed.index.isocalendar().week + pnl_df_indexed.index.year * 100
    weeks = pnl_df_indexed['week'].unique()
    l4_pnl = []
    for w in weeks:
        mask = pnl_df_indexed['week'] == w
        if mask.sum() < 1:
            continue
        sub = pnl_df.loc[mask, [c for c in pnl_df.columns]].values
        if sub.shape[0] == 0:
            continue
        # For each week, pick the single best-mean mechanism (perfect look-ahead)
        means = sub.mean(axis=0)
        best_idx = means.argmax()
        l4_pnl.extend(sub[:, best_idx].tolist())
    l4_arr = np.array(l4_pnl)
    print(f'  Weekly best-mech daily mean: {l4_arr.mean():+.4f}%')
    print(f'  Weekly best-mech daily std:  {l4_arr.std():.4f}%')

    # ============================================================
    # VERDICT
    # ============================================================
    print('\n' + '=' * 100)
    print('VERDICT — Data-level overfit ceiling')
    print('=' * 100)

    target = 0.20
    print(f'\nUser target: +{target}%/day')
    print(f'\n  L1 (naive sweep best, e.g., R2b/N8b):  ~+0.30%/day in-sample mean')
    print(f'  L2 (per-day hindsight, no fric):       {l2_mean:+.4f}%/day')
    print(f'  L2 (per-day hindsight, post-fric):     {l2_mean_fric:+.4f}%/day')
    print(f'  L3a (full-sample weight max-mean):     {port_mean.mean():+.4f}%/day')
    print(f'  L3b (full-sample weight max-Sharpe):   {port_sh.mean():+.4f}%/day')
    print(f'  L3c (long-short weight max-mean):      {port_ls.mean():+.4f}%/day')
    print(f'  L4 (weekly best-mech hindsight):       {l4_arr.mean():+.4f}%/day')

    l2_passes = l2_mean > target
    l2_fric_passes = l2_mean_fric > target
    l3_passes = port_mean.mean() > target

    print(f'\n  Interpretation:')
    if not l2_passes:
        print(f'  🔴 L2 (data-level absolute ceiling, no friction) DOES NOT reach +{target}%/day target.')
        print(f'     → Envelope confirmed at DATA LEVEL. Even with perfect look-ahead per-day mechanism switching,')
        print(f'        these 8 mechanisms cannot produce +{target}%/day.')
        print(f'     → No overfit, no in-sample fitting can escape this. ENVELOPE IS HARD LIMIT.')
    elif not l2_fric_passes:
        print(f'  🟡 L2 (no friction) reaches target but post-switching friction does NOT.')
        print(f'     → 데이터에는 potential 있으나 friction model에 의해 차단.')
    else:
        print(f'  🟢 L2 (post-friction) reaches target.')
        print(f'     → Per-day hindsight switching이 target 도달 가능. Generalization 문제.')

    if l3_passes:
        print(f'  🟢 L3 (realistic overfit) reaches target → fixed-weight portfolio in-sample achievable.')
    else:
        print(f'  🔴 L3 (realistic overfit) does NOT reach target → fixed-weight cannot escape envelope.')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'overfit ceiling diagnostic',
        'n_days': int(n_days),
        'mechanisms': list(pnl_df.columns),
        'L1_naive_sweep_best_per_mechanism': {col: float(pnl_df[col].mean()) for col in pnl_df.columns},
        'L2_hindsight_per_day_max': {
            'daily_mean': float(l2_mean),
            'daily_std': float(l2_std),
            'daily_mean_post_fric': float(l2_mean_fric),
            'pos_rate': float(l2_pos_rate),
            'p5_daily': float(l2_p5),
            'cum_pct': float(l2_cum),
            'switches': int(switches),
            'switch_fric_pct': float(fric_per_switch),
            'winner_distribution': {m: int(c) for m, c in winner_counts.items()},
        },
        'L3a_full_sample_max_mean': {
            'weights': {col: float(w) for col, w in zip(pnl_df.columns, w_mean)},
            'daily_mean': float(port_mean.mean()),
            'daily_std': float(port_mean.std()),
            'sharpe_ann': float(port_mean.mean() / max(port_mean.std(), 1e-9) * np.sqrt(365)),
        },
        'L3b_full_sample_max_sharpe': {
            'weights': {col: float(w) for col, w in zip(pnl_df.columns, w_sh)},
            'daily_mean': float(port_sh.mean()),
            'daily_std': float(port_sh.std()),
            'sharpe_ann': float(port_sh.mean() / max(port_sh.std(), 1e-9) * np.sqrt(365)),
        },
        'L3c_long_short_max_mean': {
            'weights': {col: float(w) for col, w in zip(pnl_df.columns, w_ls)},
            'daily_mean': float(port_ls.mean()),
            'daily_std': float(port_ls.std()),
        },
        'L4_weekly_best_mech_hindsight': {
            'daily_mean': float(l4_arr.mean()),
            'daily_std': float(l4_arr.std()),
        },
        'verdict': {
            'L2_passes_target_no_fric': bool(l2_passes),
            'L2_passes_target_post_fric': bool(l2_fric_passes),
            'L3_passes_target': bool(l3_passes),
        },
    }
    out_path = RESULTS / f'overfit_ceiling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
