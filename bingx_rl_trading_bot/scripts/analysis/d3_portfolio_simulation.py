"""D-3 Portfolio Simulation — single attempt (pre-committed).

Pre-commit: D-3 PASS → deployable, FAIL → closure (silent pivot 금지).
Memory: memory/d3_portfolio_precommit_20260501.md

Methodology:
  1. Extract top borderline configs from 32 sweep results
  2. Build daily-aligned trade timeline per mechanism
  3. Correlation matrix
  4. Equal-weight + Risk-parity (inverse-vol) portfolio
  5. Bootstrap user 6-criteria

Borderline mechanisms (best config from 32 sweep):
  - N8b macro regime: lookback=14, on_thr=0.3, off_thr=0.4, usd_thr=-0.3
  - R2b XS reversal: lookback=7, long_top_n=2, short=0, rebal=1
  - R1b XS momentum: lookback=60, long_top_n=4, short=0, rebal=7
  - R37b compression: NR-10 + BB squeeze pctile<0.10
  - R40b volume absorption: vol_lookback=20, vol_mult=3.0
  - Range expansion: lookback=20, mult=3.0
  - Volume spike: vol_lookback=50, vol_mult=3.0

Each mechanism's best-IS config simulated on full 720d → daily PnL series →
combine into portfolio → measure portfolio metrics.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from bootstrap_validator import bootstrap_validate, report as bootstrap_report

# Import simulators from sweep files
from r8b_donchian_sweep import simulate_trades as sim_r8b
from r41b_macd_cross_sweep import simulate_trades as sim_r41b
from r37b_compression_breakout_sweep import simulate_trades as sim_r37b
from r39b_orb_sweep import simulate_trades as sim_r39b
from r1b_xs_momentum_sweep import run_xs_momentum, load_pivot
from n8b_macro_regime_sweep import run_strategy as sim_n8b, get_macro_data
from r2b_xs_reversal_sweep import run_xs_reversal
from r40b_volume_absorption_sweep import simulate_trades as sim_r40b
from multi_indicator_batch_sweep import simulate_with_signals, range_expansion_signals, volume_spike_signals

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


# Best configs from 32 sweep (extracted from result JSONs)
BEST_CONFIGS = {
    'N8b': {
        'corr_lookback_days': 14, 'risk_on_thresh': 0.3,
        'risk_off_thresh': 0.4, 'usd_strong_thresh': -0.3,
    },
    'R2b': {
        'lookback_days': 7, 'long_top_n': 2,
        'short_bottom_n': 0, 'rebalance_days': 1,
    },
    'R1b': {
        'lookback_days': 60, 'long_top_n': 4,
        'short_bottom_n': 0, 'rebalance_days': 7,
    },
    'R8b': {
        'channel_lookback': 48, 'body_min_ratio': 0.3,
        'sl_atr_mult': 1.5, 'tp_atr_mult': 5.0,
        'max_hold_bars': 48, 'cooldown_bars': 4,
    },
    'R37b': {
        'compression_lookback': 10, 'bandwidth_lookback': 20,
        'bandwidth_pctile_max': 0.10, 'bb_period': 20, 'bb_std': 1.5,
        'body_min_ratio': 0.40, 'sl_atr_mult': 2.0,
        'tp_atr_mult': 2.0, 'max_hold_bars': 48,
    },
    'R40b': {
        'vol_lookback': 20, 'vol_mult': 3.0,
        'body_ratio_max': 0.30, 'conf_body_min': 0.30,
        'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0,
        'max_hold_bars': 24,
    },
    'Range': {
        'range_lookback': 20, 'range_mult': 3.0,
        'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0,
        'max_hold_bars': 12,
    },
    'VolSpike': {
        'vol_lookback': 50, 'vol_mult': 3.0,
        'sl_atr_mult': 2.0, 'tp_atr_mult': 2.0,
        'max_hold_bars': 12,
    },
}


def get_btc_1h():
    return pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])


def get_btc_5m_15m():
    """For mechanisms needing 5m + 1h MTF — but R21b excluded as it failed avg_gross."""
    pass


def trades_to_daily_pnl(trades_df, date_index):
    """Convert trade list to daily PnL series, indexed by date_index."""
    if len(trades_df) == 0:
        return pd.Series(0.0, index=date_index)
    trades_df = trades_df.copy()
    trades_df['close_ts'] = pd.to_datetime(trades_df['close_ts'])
    trades_df['date'] = trades_df['close_ts'].dt.normalize()
    daily_pnl = trades_df.groupby('date')['net_pnl_pct'].sum()
    # Make sure tz matches
    if daily_pnl.index.tz is None:
        daily_pnl.index = daily_pnl.index.tz_localize('UTC')
    if date_index.tz is None:
        date_index = date_index.tz_localize('UTC')
    return daily_pnl.reindex(date_index, fill_value=0.0)


def main():
    print('=' * 100)
    print('D-3 Portfolio Simulation — single attempt (pre-committed)')
    print('=' * 100)

    # Common date index
    df_1h = get_btc_1h()
    span_min = df_1h['timestamp'].min().normalize()
    span_max = df_1h['timestamp'].max().normalize()
    if span_min.tz is None:
        span_min = span_min.tz_localize('UTC')
        span_max = span_max.tz_localize('UTC')
    date_index = pd.date_range(span_min, span_max, freq='1D', tz='UTC')
    print(f'Date index: {len(date_index)} days, {date_index[0]} → {date_index[-1]}\n')

    # Run each mechanism
    print('Running each mechanism best-IS config on full 720d...')
    daily_pnl_series = {}

    # 1. R8b
    print('  R8b 1h Donchian...')
    trades = sim_r8b(df_1h, BEST_CONFIGS['R8b'])
    daily_pnl_series['R8b'] = trades_to_daily_pnl(trades, date_index)
    print(f'    n={len(trades)}, daily mean={daily_pnl_series["R8b"].mean():.4f}%')

    # 2. R37b
    print('  R37b compression breakout...')
    trades = sim_r37b(df_1h, BEST_CONFIGS['R37b'])
    daily_pnl_series['R37b'] = trades_to_daily_pnl(trades, date_index)
    print(f'    n={len(trades)}, daily mean={daily_pnl_series["R37b"].mean():.4f}%')

    # 3. R40b
    print('  R40b volume absorption...')
    trades = sim_r40b(df_1h, BEST_CONFIGS['R40b'])
    daily_pnl_series['R40b'] = trades_to_daily_pnl(trades, date_index)
    print(f'    n={len(trades)}, daily mean={daily_pnl_series["R40b"].mean():.4f}%')

    # 4. Range expansion
    print('  Range expansion...')
    sig = range_expansion_signals(df_1h, BEST_CONFIGS['Range'])
    trades = simulate_with_signals(df_1h, sig, BEST_CONFIGS['Range'])
    daily_pnl_series['Range'] = trades_to_daily_pnl(trades, date_index)
    print(f'    n={len(trades)}, daily mean={daily_pnl_series["Range"].mean():.4f}%')

    # 5. Volume spike
    print('  Volume spike directional...')
    sig = volume_spike_signals(df_1h, BEST_CONFIGS['VolSpike'])
    trades = simulate_with_signals(df_1h, sig, BEST_CONFIGS['VolSpike'])
    daily_pnl_series['VolSpike'] = trades_to_daily_pnl(trades, date_index)
    print(f'    n={len(trades)}, daily mean={daily_pnl_series["VolSpike"].mean():.4f}%')

    # 6. R1b XS momentum (10-coin)
    print('  R1b XS momentum 10coin...')
    prices = load_pivot()
    trades = run_xs_momentum(prices, BEST_CONFIGS['R1b'])
    daily_pnl_series['R1b'] = trades_to_daily_pnl(trades, date_index)
    print(f'    n={len(trades)}, daily mean={daily_pnl_series["R1b"].mean():.4f}%')

    # 7. R2b XS reversal (10-coin)
    print('  R2b XS reversal 10coin...')
    trades = run_xs_reversal(prices, BEST_CONFIGS['R2b'])
    daily_pnl_series['R2b'] = trades_to_daily_pnl(trades, date_index)
    print(f'    n={len(trades)}, daily mean={daily_pnl_series["R2b"].mean():.4f}%')

    # 8. N8b macro regime (data has its own range)
    print('  N8b macro regime BTC vs DXY/SPY/GLD...')
    macro_full = get_macro_data()
    trades = sim_n8b(macro_full, BEST_CONFIGS['N8b'])
    daily_pnl_series['N8b'] = trades_to_daily_pnl(trades, date_index)
    print(f'    n={len(trades)}, daily mean={daily_pnl_series["N8b"].mean():.4f}%')

    # Build daily PnL DataFrame
    pnl_df = pd.DataFrame(daily_pnl_series)
    print(f'\nDaily PnL DataFrame: {pnl_df.shape}')

    # Correlation matrix
    print('\n=== Correlation matrix (daily PnL) ===')
    corr = pnl_df.corr()
    print(corr.round(3).to_string())

    # Off-diagonal mean
    n = len(corr.columns)
    mask = ~np.eye(n, dtype=bool)
    rho_avg = corr.values[mask].mean()
    print(f'\nρ_avg off-diagonal: {rho_avg:.4f}')
    N_eff = n / (1 + (n - 1) * max(rho_avg, 0))
    print(f'Effective N (8 mechanisms → {N_eff:.2f} independent)')

    # Equal-weight portfolio
    print('\n=== Equal-weight portfolio ===')
    ew_pnl = pnl_df.mean(axis=1)
    print(f'EW daily mean: {ew_pnl.mean():+.4f}%')
    print(f'EW daily std:  {ew_pnl.std():.4f}%')
    print(f'EW Sharpe (ann): {ew_pnl.mean() / max(ew_pnl.std(), 1e-9) * np.sqrt(365):.3f}')

    # Risk-parity (inverse vol)
    print('\n=== Risk-parity (inverse-vol) portfolio ===')
    vols = pnl_df.std()
    inv_vol = 1.0 / vols.replace(0, np.nan)
    weights = inv_vol / inv_vol.sum()
    print('Weights:')
    for k, v in weights.items():
        print(f'  {k}: {v:.4f}')
    rp_pnl = (pnl_df * weights).sum(axis=1)
    print(f'RP daily mean: {rp_pnl.mean():+.4f}%')
    print(f'RP daily std:  {rp_pnl.std():.4f}%')
    print(f'RP Sharpe (ann): {rp_pnl.mean() / max(rp_pnl.std(), 1e-9) * np.sqrt(365):.3f}')

    # Bootstrap evaluation per portfolio
    def evaluate_portfolio(pnl_series, name):
        """Convert daily PnL to trade-like format for bootstrap."""
        pnl_nonzero = pnl_series[pnl_series != 0]
        trades_df = pd.DataFrame({
            'close_ts': pnl_nonzero.index,
            'gross_pct': pnl_nonzero.values + 0.07,  # approximate gross (assume 0.07 friction baked in)
            'net_pnl_pct': pnl_nonzero.values,
        })
        if len(trades_df) == 0:
            print(f'\n{name}: no nonzero days')
            return None
        span_min = trades_df['close_ts'].min()
        span_max = trades_df['close_ts'].max()
        res = bootstrap_validate(trades_df, span_min, span_max)
        bootstrap_report(res, name)

        f1 = res.avg_per_trade_pct > 0.07
        f6 = len(trades_df) >= 50
        overall = f1 and f6 and res.overall_pass
        print(f'  Portfolio overall: {"✅ PASS" if overall else "🔴 FAIL"}')
        return {
            'name': name,
            'n_active_days': int(len(trades_df)),
            'mean_daily_pct': pnl_series.mean(),
            'std_daily_pct': pnl_series.std(),
            'sharpe_ann': pnl_series.mean() / max(pnl_series.std(), 1e-9) * np.sqrt(365),
            'bootstrap_mean_daily': res.mean_daily_pct,
            'bootstrap_pos_rate': res.pos_rate,
            'bootstrap_p5_daily': res.p5_daily_pct,
            'bootstrap_avg_per_trade': res.avg_per_trade_pct,
            'bootstrap_pass_criteria': {k: bool(v) for k, v in res.pass_criteria.items()},
            'bootstrap_overall_pass': bool(res.overall_pass),
            'F1_avg_gross_pass': bool(f1),
            'F6_full_n_pass': bool(f6),
            'portfolio_overall_pass': bool(overall),
        }

    print('\n' + '=' * 100)
    print('Portfolio Bootstrap Evaluation')
    print('=' * 100)
    ew_eval = evaluate_portfolio(ew_pnl, 'Equal-weight portfolio')
    rp_eval = evaluate_portfolio(rp_pnl, 'Risk-parity portfolio')

    # Top-3 lowest-correlation subset
    print('\n=== Top-3 lowest correlation subset ===')
    if len(corr) >= 3:
        # Find triple with lowest avg pairwise corr
        from itertools import combinations
        best_triple = None
        best_avg_corr = float('inf')
        for combo in combinations(corr.columns, 3):
            sub = corr.loc[list(combo), list(combo)].values
            mask3 = ~np.eye(3, dtype=bool)
            avg_corr = sub[mask3].mean()
            if avg_corr < best_avg_corr:
                best_avg_corr = avg_corr
                best_triple = combo
        print(f'Best triple: {best_triple}, avg ρ={best_avg_corr:.3f}')
        triple_pnl = pnl_df[list(best_triple)].mean(axis=1)
        triple_eval = evaluate_portfolio(triple_pnl, f'Top-3 low-corr ({", ".join(best_triple)})')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_commit': 'memory/d3_portfolio_precommit_20260501.md',
        'mechanisms': list(BEST_CONFIGS.keys()),
        'best_configs': BEST_CONFIGS,
        'correlation_matrix': corr.to_dict(),
        'rho_avg': float(rho_avg),
        'n_effective': float(N_eff),
        'equal_weight': ew_eval,
        'risk_parity': rp_eval,
        'top3_low_corr': {
            'triple': list(best_triple) if best_triple else None,
            'avg_rho': float(best_avg_corr) if best_triple else None,
            'eval': triple_eval if best_triple else None,
        },
    }
    out_path = RESULTS / f'd3_portfolio_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')

    # Final verdict
    print('\n' + '=' * 100)
    print('D-3 VERDICT (PRE-COMMITTED)')
    print('=' * 100)
    any_pass = (ew_eval and ew_eval['portfolio_overall_pass']) or \
               (rp_eval and rp_eval['portfolio_overall_pass']) or \
               (best_triple and triple_eval and triple_eval['portfolio_overall_pass'])
    if any_pass:
        print('  🟢 D-3 PORTFOLIO PASS — DEPLOYABLE candidate')
        print('  → Additional verification (regime test, BT-LIVE parity) needed')
    else:
        print('  🔴 D-3 PORTFOLIO FAIL')
        print('  → PRE-COMMITTED: Closure 강제. D-1/D-2/E silent pivot 금지.')
        print('  → Final synthesis: retail BingX 1× envelope empty for +0.20%/day target.')


if __name__ == '__main__':
    main()
