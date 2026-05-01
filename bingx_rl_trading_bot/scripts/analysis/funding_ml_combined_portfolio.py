"""Funding Arb V5 + Strict ML thr=0.15 Combined Portfolio.

Pre-commit: memory/funding_ml_combined_precommit_20260501.md (frozen).

Two deployable mechanisms:
  - Funding Arb V5 (leverage 2× perp): +7%/yr, Sharpe 26, MaxDD -0.17%
  - Strict ML thr=0.15 (1h ML + on-chain): +23%/yr WF, hit 66.6%, 5/5+2/2 robust

Combined daily PnL → portfolio simulation:
  V1: 50/50 equal-weight
  V2: Risk-parity (inverse vol)
  V3: Correlation-optimal (minimum variance)

Bootstrap user 6-criteria.
"""
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
from h1_direction_prediction import build_features
from h1_ml_with_onchain_features import load_onchain_daily
from bootstrap_validator import bootstrap_validate, report as bootstrap_report

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

# Frozen params
FRICTION_PCT = 0.10
THRESHOLD = 0.15

# Funding params (V5)
CAPITAL_USD = 1500
SPOT_POS = 750  # $750 spot LONG
PERP_POS_LEV = 1500  # $1500 perp SHORT (with $750 margin = 2× leverage)
SPOT_FRIC_PCT = 0.10
PERP_FRIC_PCT = 0.04


def build_funding_daily_pnl():
    """V5 leverage 2× perp daily PnL."""
    df = pd.read_parquet(DATA / 'funding_history.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    btc = df[df['symbol'] == 'BTC/USDT'].copy().sort_values('datetime').reset_index(drop=True)
    btc['rate_pct'] = btc['funding_rate'] * 100

    # 8h funding × 3/day → daily
    btc['date'] = btc['datetime'].dt.tz_convert('UTC').dt.normalize().dt.tz_localize(None)
    btc['funding_income_usd'] = btc['rate_pct'] / 100 * PERP_POS_LEV  # perp short receives positive funding
    daily_funding_pnl_pct = btc.groupby('date')['funding_income_usd'].sum() / CAPITAL_USD * 100

    # Subtract one-time entry friction over span (amortized)
    span_days = (btc['date'].max() - btc['date'].min()).days
    entry_fric = (SPOT_FRIC_PCT / 100 * SPOT_POS) + (PERP_FRIC_PCT / 100 * PERP_POS_LEV)
    exit_fric = entry_fric
    total_fric_pct = (entry_fric + exit_fric) / CAPITAL_USD * 100
    daily_fric = total_fric_pct / span_days
    daily_funding_pnl_pct = daily_funding_pnl_pct - daily_fric

    return daily_funding_pnl_pct


def build_ml_daily_pnl():
    """Strict ML thr=0.15 daily PnL (using train 50% only - test/val window 50%).

    For combined portfolio, use IN-SAMPLE train+val period as base for robust evaluation
    (matches funding span). Conservative: use trained-on-50% model applied to ALL data
    causally would create lookahead, so use only test 50% with proper train.
    """
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    feats = build_features(df)
    feats = feats.merge(df[['timestamp']], left_index=True, right_index=True, how='left')
    feats = feats.dropna(subset=['timestamp'])

    onchain = load_onchain_daily()
    feats['date'] = pd.to_datetime(feats['timestamp']).dt.normalize()
    if feats['date'].dt.tz is not None:
        feats['date'] = feats['date'].dt.tz_localize(None)
    if onchain['date'].dt.tz is not None:
        onchain['date'] = onchain['date'].dt.tz_localize(None)
    feats = pd.merge_asof(
        feats.sort_values('date'),
        onchain.sort_values('date'),
        on='date', direction='backward',
    )
    feats = feats.dropna(subset=['fng_value', 'mvrv'])
    feats = feats.sort_values('timestamp').reset_index(drop=True)

    tech_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']
    onchain_cols = ['fng_value', 'fng_z90', 'mvrv', 'mvrv_z90',
                     'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']
    all_cols = tech_cols + onchain_cols

    n = len(feats)
    i_split = int(n * 0.50)
    train = feats.iloc[:i_split]
    test_full = feats.iloc[i_split:]  # val+test combined as "OOS"

    X_tr = train[all_cols].values
    y_tr = train['target'].values
    X_te = test_full[all_cols].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_tr_s, y_tr)
    p_te = model.predict_proba(X_te_s)[:, 1]

    direction = np.where(p_te > 0.5 + THRESHOLD, 1,
                          np.where(p_te < 0.5 - THRESHOLD, -1, 0))
    next_ret = test_full['next_ret_pct'].values
    ts = test_full['timestamp'].values

    trades = []
    for i in range(len(p_te)):
        if direction[i] == 0 or pd.isna(next_ret[i]):
            continue
        gross = next_ret[i] * direction[i]
        net = gross - FRICTION_PCT
        trades.append({
            'date': pd.Timestamp(ts[i]).normalize(),
            'net_pnl_pct': net,
        })
    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return pd.Series(dtype=float)

    if trades_df['date'].dt.tz is not None:
        trades_df['date'] = trades_df['date'].dt.tz_localize(None)

    daily_ml_pnl = trades_df.groupby('date')['net_pnl_pct'].sum()
    return daily_ml_pnl


def evaluate_portfolio(pnl_series, name):
    """Compute portfolio metrics + bootstrap user 6-criteria."""
    s = pnl_series.dropna()
    n_days = len(s)
    nonzero = s[s != 0]

    daily_mean = float(s.mean())
    daily_std = float(s.std())
    cum = float(s.sum())
    annual = daily_mean * 365
    sharpe_ann = daily_mean / max(daily_std, 1e-9) * np.sqrt(365)

    # Drawdown
    eq = (1 + s / 100).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100
    max_dd = float(dd.min())

    # Bootstrap (only nonzero days)
    if len(nonzero) >= 5:
        trades_df = pd.DataFrame({
            'close_ts': pd.to_datetime(nonzero.index),
            'gross_pct': nonzero.values + 0.001,
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
                'pass_criteria': {k: bool(v) for k, v in res.pass_criteria.items()},
                'overall_pass': bool(res.overall_pass),
            }
        except Exception as e:
            bs = {'error': str(e)}
    else:
        bs = None

    return {
        'name': name,
        'n_days': n_days,
        'n_nonzero_days': len(nonzero),
        'daily_mean_pct': daily_mean,
        'daily_std_pct': daily_std,
        'cum_pct': cum,
        'annual_return_pct': annual,
        'sharpe_ann': sharpe_ann,
        'max_drawdown_pct': max_dd,
        'bootstrap': bs,
    }


def main():
    print('=' * 100)
    print('Funding Arb V5 + Strict ML thr=0.15 Combined Portfolio')
    print('=' * 100)

    funding_daily = build_funding_daily_pnl()
    print(f'\nFunding Arb V5 daily PnL: {len(funding_daily)} days, '
          f'mean={funding_daily.mean():+.4f}%, std={funding_daily.std():.4f}%')

    ml_daily = build_ml_daily_pnl()
    print(f'Strict ML daily PnL: {len(ml_daily)} days, '
          f'mean={ml_daily.mean():+.4f}%, std={ml_daily.std():.4f}%')

    # Align on common dates
    common_dates = funding_daily.index.intersection(ml_daily.index)
    funding_aligned = funding_daily.loc[common_dates]
    ml_aligned = ml_daily.loc[common_dates]

    # Forward-fill ML zeros (no-trade days)
    pnl_df = pd.DataFrame({
        'Funding': funding_aligned,
        'ML_thr015': ml_aligned,
    }).fillna(0)

    print(f'\nAligned PnL DataFrame: {pnl_df.shape}')
    print(f'Span: {pnl_df.index.min()} → {pnl_df.index.max()}')

    # Correlation
    print('\n=== Correlation ===')
    corr = pnl_df.corr()
    print(corr.round(4).to_string())
    rho = corr.iloc[0, 1]
    print(f'\nFunding ↔ ML correlation: {rho:+.4f}')

    # Per-mechanism stats
    print('\n=== Per-mechanism ===')
    funding_eval = evaluate_portfolio(pnl_df['Funding'], 'Funding Arb V5')
    ml_eval = evaluate_portfolio(pnl_df['ML_thr015'], 'Strict ML thr=0.15')
    for r in [funding_eval, ml_eval]:
        print(f'\n{r["name"]}:')
        print(f'  Daily mean: {r["daily_mean_pct"]:+.4f}%, Annual: {r["annual_return_pct"]:+.2f}%')
        print(f'  Sharpe: {r["sharpe_ann"]:.3f}, MaxDD: {r["max_drawdown_pct"]:+.2f}%')
        if r['bootstrap']:
            print(f'  Bootstrap pass: {r["bootstrap"]["pass_count_6"]}/6')

    # Portfolio variants
    print('\n=== Portfolio Variants ===')
    # V1: 50/50 equal-weight
    pnl_ew = (pnl_df['Funding'] + pnl_df['ML_thr015']) / 2
    ew_eval = evaluate_portfolio(pnl_ew, 'V1_EqualWeight_50_50')

    # V2: Risk-parity
    vols = pnl_df.std()
    inv_vol = 1.0 / vols.replace(0, np.nan)
    weights_rp = inv_vol / inv_vol.sum()
    pnl_rp = (pnl_df * weights_rp).sum(axis=1)
    rp_eval = evaluate_portfolio(pnl_rp, 'V2_RiskParity')

    # V3: Minimum variance (correlation-optimal)
    cov = pnl_df.cov().values
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(2)
    weights_mv = inv_cov @ ones / (ones @ inv_cov @ ones)
    weights_mv = np.clip(weights_mv, 0, 1)
    weights_mv = weights_mv / weights_mv.sum()
    pnl_mv = (pnl_df.values @ weights_mv)
    pnl_mv = pd.Series(pnl_mv, index=pnl_df.index)
    mv_eval = evaluate_portfolio(pnl_mv, 'V3_MinVariance')

    print('\n--- V1 Equal-weight 50/50 ---')
    print(f'  Daily mean: {ew_eval["daily_mean_pct"]:+.4f}%, Annual: {ew_eval["annual_return_pct"]:+.2f}%')
    print(f'  Sharpe: {ew_eval["sharpe_ann"]:.3f}, MaxDD: {ew_eval["max_drawdown_pct"]:+.2f}%')
    if ew_eval['bootstrap']:
        print(f'  Bootstrap pass: {ew_eval["bootstrap"]["pass_count_6"]}/6, overall: {"✅" if ew_eval["bootstrap"]["overall_pass"] else "🔴"}')

    print(f'\n--- V2 Risk-parity (Funding {weights_rp[0]:.3f}, ML {weights_rp[1]:.3f}) ---')
    print(f'  Daily mean: {rp_eval["daily_mean_pct"]:+.4f}%, Annual: {rp_eval["annual_return_pct"]:+.2f}%')
    print(f'  Sharpe: {rp_eval["sharpe_ann"]:.3f}, MaxDD: {rp_eval["max_drawdown_pct"]:+.2f}%')
    if rp_eval['bootstrap']:
        print(f'  Bootstrap pass: {rp_eval["bootstrap"]["pass_count_6"]}/6, overall: {"✅" if rp_eval["bootstrap"]["overall_pass"] else "🔴"}')

    print(f'\n--- V3 Min-variance (Funding {weights_mv[0]:.3f}, ML {weights_mv[1]:.3f}) ---')
    print(f'  Daily mean: {mv_eval["daily_mean_pct"]:+.4f}%, Annual: {mv_eval["annual_return_pct"]:+.2f}%')
    print(f'  Sharpe: {mv_eval["sharpe_ann"]:.3f}, MaxDD: {mv_eval["max_drawdown_pct"]:+.2f}%')
    if mv_eval['bootstrap']:
        print(f'  Bootstrap pass: {mv_eval["bootstrap"]["pass_count_6"]}/6, overall: {"✅" if mv_eval["bootstrap"]["overall_pass"] else "🔴"}')

    # ============================================================
    # Pre-committed VERDICT
    # ============================================================
    print('\n' + '=' * 100)
    print('VERDICT (PRE-COMMITTED)')
    print('=' * 100)

    any_pass = (ew_eval['bootstrap'] and ew_eval['bootstrap']['overall_pass']) or \
               (rp_eval['bootstrap'] and rp_eval['bootstrap']['overall_pass']) or \
               (mv_eval['bootstrap'] and mv_eval['bootstrap']['overall_pass'])

    if any_pass:
        print('  🟢 COMBINED PORTFOLIO PASS — DEPLOYABLE candidate')
        print('  → Lookahead audit obligation + LIVE execution은 별도 mandate')
    else:
        print('  🟡 COMBINED FAIL strict 6-criteria → 2-MECHANISM SEPARATE DEPLOY LIST')
        print('  (closure 아닌 valid final state per pre-commit)')
        print()
        print('  Deployable mechanisms (각각 단독 valid):')
        print(f'    1. Funding Arb V5: +{funding_eval["annual_return_pct"]:.2f}%/yr, Sharpe {funding_eval["sharpe_ann"]:.2f}, MaxDD {funding_eval["max_drawdown_pct"]:+.2f}%')
        print(f'    2. Strict ML thr=0.15: +{ml_eval["annual_return_pct"]:.2f}%/yr, Sharpe {ml_eval["sharpe_ann"]:.2f}')
        print(f'    Combined EW: +{ew_eval["annual_return_pct"]:.2f}%/yr, Sharpe {ew_eval["sharpe_ann"]:.2f}, MaxDD {ew_eval["max_drawdown_pct"]:+.2f}%')
        print(f'    → 사용자 결정: 단일 / 합산 / 모두 deploy')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'Funding+ML combined portfolio (자율 mandate)',
        'pre_commit': 'memory/funding_ml_combined_precommit_20260501.md',
        'correlation': float(rho),
        'funding_eval': funding_eval,
        'ml_eval': ml_eval,
        'V1_equal_weight': ew_eval,
        'V2_risk_parity': rp_eval,
        'V2_weights': {'funding': float(weights_rp[0]), 'ml': float(weights_rp[1])},
        'V3_min_variance': mv_eval,
        'V3_weights': {'funding': float(weights_mv[0]), 'ml': float(weights_mv[1])},
        'any_pass': bool(any_pass),
    }
    out_path = RESULTS / f'funding_ml_combined_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
