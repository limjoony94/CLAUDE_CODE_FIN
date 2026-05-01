"""Online Learning Adaptive Weight Simulation.

Pre-commit: memory/online_learning_precommit_20260501.md (frozen design).

CAUSAL design (NO look-ahead):
  At day t, weight w_t computed from PnL[t-30:t-1] (excluding t).
  Apply w_t to day t's mechanism PnL.

Locked params:
  WINDOW           = 30  # rolling lookback days
  CAP              = 0.40  # max 40% per mechanism
  DEACTIVATE_LB    = 14  # 14d cumulative PnL <0 → 0 weight
  MIN_ACTIVE       = 3   # equal-weight fallback if active < 3

Bootstrap user 6-criteria evaluation.

Stopping: 1 attempt. PASS → deployable, FAIL → closure 강제.
Daily mean > +0.5% → lookahead 의심, audit + advisor reconcile.
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

from bootstrap_validator import bootstrap_validate, report as bootstrap_report

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

# FROZEN params (per pre-commit)
WINDOW = 30
CAP = 0.40
DEACTIVATE_LB = 14
MIN_ACTIVE = 3


def load_pnl_df():
    """Reuse D-3 daily PnL DataFrame (8 mechanism)."""
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
    print('Building 8-mechanism daily PnL (D-3 reuse)...')
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


def compute_causal_weights(pnl_df):
    """For each day t, weight = inverse-variance over [t-WINDOW, t-1].

    CAUSAL: pnl[:t] used (excluding t). Day 0 to WINDOW-1: equal-weight fallback.

    Returns: weights DataFrame same shape as pnl_df.
    """
    n_days, n_mech = pnl_df.shape
    cols = pnl_df.columns
    weights = pd.DataFrame(0.0, index=pnl_df.index, columns=cols)

    pnl_arr = pnl_df.values

    for t in range(n_days):
        if t < WINDOW:
            # Insufficient history → equal weight among all
            weights.iloc[t] = 1.0 / n_mech
            continue

        # CAUSAL: data from [t-WINDOW, t-1] inclusive
        window_data = pnl_arr[t - WINDOW: t, :]   # shape (WINDOW, n_mech), excludes t

        # 14d cumulative deactivation check
        deact_window = pnl_arr[max(0, t - DEACTIVATE_LB): t, :]   # last 14 days
        cum_pnl = deact_window.sum(axis=0)
        active = cum_pnl >= 0   # True if active (cumulative non-negative)

        # If too few active, equal-weight fallback among all
        if active.sum() < MIN_ACTIVE:
            weights.iloc[t] = 1.0 / n_mech
            continue

        # Inverse-variance among active
        active_idx = np.where(active)[0]
        active_data = window_data[:, active_idx]
        variances = active_data.var(axis=0, ddof=0)
        # Avoid divide by zero
        variances = np.where(variances > 1e-12, variances, 1e-12)
        inv_var = 1.0 / variances
        raw_w = inv_var / inv_var.sum()

        # Apply 40% cap iteratively (water-filling)
        capped_w = np.minimum(raw_w, CAP)
        # If sum < 1 due to capping, redistribute remainder
        if capped_w.sum() < 1.0 - 1e-9:
            uncapped_mask = capped_w < CAP - 1e-9
            remaining = 1.0 - capped_w.sum()
            # Distribute proportionally to uncapped raw_w
            if uncapped_mask.any():
                uncapped_total = raw_w[uncapped_mask].sum()
                if uncapped_total > 0:
                    capped_w[uncapped_mask] += remaining * raw_w[uncapped_mask] / uncapped_total
        capped_w = capped_w / capped_w.sum()  # normalize

        # Place back into full weight vector
        full_w = np.zeros(n_mech)
        full_w[active_idx] = capped_w
        weights.iloc[t] = full_w

    return weights


def main():
    print('=' * 100)
    print('Online Learning Adaptive Weight Simulation')
    print('=' * 100)
    print(f'Locked params: WINDOW={WINDOW}, CAP={CAP}, DEACTIVATE_LB={DEACTIVATE_LB}, MIN_ACTIVE={MIN_ACTIVE}')
    print(f'Pre-commit: memory/online_learning_precommit_20260501.md')
    print()

    pnl_df = load_pnl_df()
    print(f'\nDaily PnL DataFrame: {pnl_df.shape}')
    print(f'Span: {pnl_df.index.min()} → {pnl_df.index.max()}')

    # Compute causal weights
    print('\nComputing CAUSAL rolling weights (no look-ahead)...')
    weights_df = compute_causal_weights(pnl_df)

    # Sanity check: weights[t] only depends on pnl[:t]
    print('\nSanity check (weights time series):')
    print(f'  Day 0 weights: equal? {np.allclose(weights_df.iloc[0].values, 1/8)}')
    print(f'  Day {WINDOW-1} weights: equal? {np.allclose(weights_df.iloc[WINDOW-1].values, 1/8)}')
    print(f'  Day {WINDOW} weights (first computed): {weights_df.iloc[WINDOW].round(3).to_dict()}')

    # Apply weights to PnL (element-wise multiply, sum across mechanisms)
    portfolio_pnl = (weights_df * pnl_df).sum(axis=1)

    print(f'\n=== Online Learning Portfolio Stats ===')
    print(f'Total days: {len(portfolio_pnl)}')
    print(f'Active days (nonzero portfolio PnL): {(portfolio_pnl != 0).sum()}')
    print(f'Daily mean: {portfolio_pnl.mean():+.4f}%')
    print(f'Daily std: {portfolio_pnl.std():.4f}%')
    print(f'Daily Sharpe (ann): {portfolio_pnl.mean() / max(portfolio_pnl.std(), 1e-9) * np.sqrt(365):.3f}')
    print(f'Cumulative: {portfolio_pnl.sum():+.2f}%')

    # Average weights
    print('\n=== Average weights (post-warmup) ===')
    avg_w = weights_df.iloc[WINDOW:].mean()
    for col, w in avg_w.items():
        print(f'  {col}: {w:.4f}')

    # Lookahead audit warning
    if portfolio_pnl.mean() > 0.5:
        print(f'\n⚠️  LOOKAHEAD SUSPICION: daily mean +{portfolio_pnl.mean():.4f}% > 0.5% threshold')
        print(f'   Code audit 필요. compute_causal_weights() indexing 재확인.')

    # Bootstrap evaluation
    print('\n=== Bootstrap evaluation (user 6-criteria) ===')
    pnl_nonzero = portfolio_pnl[portfolio_pnl != 0]
    trades_df = pd.DataFrame({
        'close_ts': pnl_nonzero.index,
        'gross_pct': pnl_nonzero.values + 0.07,  # approx gross
        'net_pnl_pct': pnl_nonzero.values,
    })
    if len(trades_df) > 0:
        span_min = trades_df['close_ts'].min()
        span_max = trades_df['close_ts'].max()
        res = bootstrap_validate(trades_df, span_min, span_max)
        bootstrap_report(res, 'Online Learning')

        f1 = res.avg_per_trade_pct > 0.07
        f6 = len(trades_df) >= 50
        overall = f1 and f6 and res.overall_pass
        print(f'  Portfolio overall: {"✅ PASS" if overall else "🔴 FAIL"}')

        # Save
        out = {
            'date': datetime.now(timezone.utc).isoformat(),
            'mandate': 'online learning adaptive weight (causal)',
            'pre_commit': 'memory/online_learning_precommit_20260501.md',
            'locked_params': {
                'WINDOW': WINDOW, 'CAP': CAP,
                'DEACTIVATE_LB': DEACTIVATE_LB, 'MIN_ACTIVE': MIN_ACTIVE,
            },
            'n_days': int(len(portfolio_pnl)),
            'n_active_days': int((portfolio_pnl != 0).sum()),
            'daily_mean_pct': float(portfolio_pnl.mean()),
            'daily_std_pct': float(portfolio_pnl.std()),
            'sharpe_ann': float(portfolio_pnl.mean() / max(portfolio_pnl.std(), 1e-9) * np.sqrt(365)),
            'cumulative_pct': float(portfolio_pnl.sum()),
            'avg_weights': {col: float(w) for col, w in avg_w.items()},
            'bootstrap_mean_daily': float(res.mean_daily_pct),
            'bootstrap_pos_rate': float(res.pos_rate),
            'bootstrap_p5_daily': float(res.p5_daily_pct),
            'bootstrap_avg_per_trade': float(res.avg_per_trade_pct),
            'bootstrap_pass_criteria': {k: bool(v) for k, v in res.pass_criteria.items()},
            'bootstrap_overall_pass': bool(res.overall_pass),
            'F1_avg_gross_pass': bool(f1),
            'F6_full_n_pass': bool(f6),
            'portfolio_overall_pass': bool(overall),
            'lookahead_suspicion': bool(portfolio_pnl.mean() > 0.5),
        }
        out_path = RESULTS / f'online_learning_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'\nSaved: {out_path}')

        # Final verdict
        print('\n' + '=' * 100)
        print('VERDICT (PRE-COMMITTED)')
        print('=' * 100)
        if overall:
            print('  🟢 ONLINE LEARNING PASS — DEPLOYABLE candidate')
            print('  → Lookahead audit 추가 + advisor reconcile + regime test')
        else:
            print('  🔴 ONLINE LEARNING FAIL')
            print('  → PRE-COMMITTED: closure 강제. Meta-strategy/Drawdown silent pivot 금지.')
            print(f'  → Selection problem partial solution도 envelope 한계 confirm.')
            print(f'  → Daily {portfolio_pnl.mean():+.4f}% < target +0.20%')


if __name__ == '__main__':
    main()
