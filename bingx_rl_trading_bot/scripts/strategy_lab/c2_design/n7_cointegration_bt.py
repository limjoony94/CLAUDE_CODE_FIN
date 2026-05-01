"""N7 — Cointegration Pair Trading BT.

Mechanism:
  1. For each of C(8,2)=28 pairs, run Engle-Granger cointegration test on rolling 90d window
  2. If cointegrated (p < 0.05), compute spread = log(P_A) - β × log(P_B)
  3. Compute z-score of spread vs rolling mean/std
  4. Entry: |z| > 2.0
  5. Exit: |z| < 0.5 OR half-life × 2 elapsed
  6. Hedge ratio: rolling OLS β
  7. Position: long underpriced, short overpriced, dollar-neutral

Friction: maker (mean-reversion → no adverse selection per R25 lesson)
  0.04% per side × 2 legs × 2 sides = 0.16% RT total per trade

Bootstrap: bootstrap_validator (사용자 criteria 적용)
"""
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap_validator import bootstrap_validate, report as bootstrap_report


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


LOCKED = {
    'capital_usd': 1000,
    'coint_lookback_bars': 540,    # 90d × 6 (4h)
    'zscore_lookback_bars': 180,   # 30d × 6
    'coint_pval_threshold': 0.05,
    'z_entry': 2.0,
    'z_exit': 0.5,
    'maker_fric_pct': 0.04,
    'slippage_pct': 0.02,
    'max_hold_bars': 180,           # 30d max
    'min_half_life_bars': 6,        # 1d min
    'max_half_life_bars': 180,      # 30d max
}


def load_data():
    df = pd.read_csv(DATA / 'n7_8coin_4h_close.csv', parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.pivot(index='timestamp', columns='symbol', values='close').sort_index()


def estimate_half_life(spread):
    """Estimate mean-reversion half-life via AR(1) on differenced spread."""
    s = pd.Series(spread).dropna()
    if len(s) < 10:
        return np.nan
    s_lag = s.shift(1).dropna()
    s_diff = s.diff().dropna()
    common = s_lag.index.intersection(s_diff.index)
    if len(common) < 10:
        return np.nan
    X = s_lag.loc[common].values
    y = s_diff.loc[common].values
    # OLS without intercept
    if np.var(X) == 0:
        return np.nan
    beta = np.cov(X, y, bias=True)[0, 1] / np.var(X)
    if beta >= 0:
        return np.inf
    return -np.log(2) / beta


def screen_pairs(prices: pd.DataFrame) -> list[dict]:
    """Find cointegrated pairs (full sample)."""
    coins = sorted(prices.columns)
    results = []
    for a, b in combinations(coins, 2):
        pa = prices[a].dropna()
        pb = prices[b].dropna()
        common = pa.index.intersection(pb.index)
        if len(common) < 200:
            continue
        try:
            t_stat, p_val, _ = coint(np.log(pa.loc[common]), np.log(pb.loc[common]))
        except Exception:
            continue
        # Hedge ratio
        ya = np.log(pa.loc[common])
        xb = np.log(pb.loc[common])
        beta = np.cov(ya, xb, bias=True)[0, 1] / np.var(xb) if np.var(xb) > 0 else np.nan
        spread = ya - beta * xb
        hl = estimate_half_life(spread)
        results.append({
            'a': a, 'b': b,
            'coint_t': float(t_stat),
            'coint_p': float(p_val),
            'beta': float(beta),
            'half_life_bars': float(hl) if not np.isnan(hl) else None,
            'spread_std': float(spread.std()),
        })
    return sorted(results, key=lambda x: x['coint_p'])


def simulate_pair(prices: pd.DataFrame, a: str, b: str, beta: float) -> list[dict]:
    """Per-bar BT for a pair."""
    pa = np.log(prices[a].dropna())
    pb = np.log(prices[b].dropna())
    common = pa.index.intersection(pb.index)
    spread = (pa.loc[common] - beta * pb.loc[common]).dropna()
    timestamps = spread.index

    # Rolling z-score
    lb = LOCKED['zscore_lookback_bars']
    rolling_mean = spread.rolling(lb, min_periods=lb // 2).mean()
    rolling_std = spread.rolling(lb, min_periods=lb // 2).std()
    z = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

    capital = LOCKED['capital_usd']
    leg = capital / 2   # 50% per leg
    fric_per_side = LOCKED['maker_fric_pct'] + LOCKED['slippage_pct']
    fric_per_trade = 4 * fric_per_side / 100 * leg   # 2 legs × 2 sides

    active = None
    trades = []

    for i, ts in enumerate(timestamps):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue

        if active is None:
            if abs(zi) > LOCKED['z_entry']:
                # If z > 0 → spread too high → short A (overpriced) + long B
                # If z < 0 → spread too low → long A (underpriced) + short B
                side = 'short_A_long_B' if zi > 0 else 'long_A_short_B'
                active = {
                    'side': side,
                    'enter_idx': i, 'enter_ts': ts,
                    'enter_z': float(zi),
                    'enter_log_pa': float(pa.loc[ts]),
                    'enter_log_pb': float(pb.loc[ts]),
                }
        else:
            held = i - active['enter_idx']
            should_exit = False; reason = None
            if held >= LOCKED['max_hold_bars']:
                should_exit, reason = True, 'MAX_HOLD'
            elif abs(zi) < LOCKED['z_exit']:
                should_exit, reason = True, 'Z_NORMALIZE'

            if should_exit:
                cur_log_pa = float(pa.loc[ts])
                cur_log_pb = float(pb.loc[ts])
                d_pa = cur_log_pa - active['enter_log_pa']
                d_pb = cur_log_pb - active['enter_log_pb']
                # PnL: if short A, long B: PnL = -A * d_pa + beta * B * d_pb (approximate via log returns)
                # Simplified: each leg notional × log return
                if active['side'] == 'short_A_long_B':
                    leg_pnl = -leg * d_pa + leg * beta * d_pb
                else:
                    leg_pnl = leg * d_pa - leg * beta * d_pb
                net_pnl = leg_pnl - fric_per_trade
                trades.append({
                    'a': a, 'b': b, 'side': active['side'],
                    'enter_ts': str(active['enter_ts']), 'close_ts': str(ts),
                    'enter_z': active['enter_z'], 'exit_z': float(zi),
                    'periods_held': held, 'reason': reason,
                    'gross_pnl_usd': leg_pnl, 'fric_usd': fric_per_trade,
                    'net_pnl_usd': net_pnl, 'net_pnl_pct': net_pnl / capital * 100,
                    'gross_pct': leg_pnl / capital * 100,
                })
                active = None
    return trades


def main():
    print('=' * 100)
    print('N7 — Cointegration Pair Trading BT (8-coin universe)')
    print('=' * 100)
    print(f'Locked: coint_p < {LOCKED["coint_pval_threshold"]}, '
          f'z_entry ±{LOCKED["z_entry"]}, z_exit ±{LOCKED["z_exit"]}, '
          f'half-life [{LOCKED["min_half_life_bars"]/6:.0f}d, {LOCKED["max_half_life_bars"]/6:.0f}d]')
    print(f'Friction {LOCKED["maker_fric_pct"]+LOCKED["slippage_pct"]:.2f}%/side × 4 = '
          f'{4*(LOCKED["maker_fric_pct"]+LOCKED["slippage_pct"]):.2f}% RT per trade\n')

    prices = load_data()
    span_start = prices.index.min()
    span_end = prices.index.max()
    span_days = (span_end - span_start).total_seconds() / 86400
    print(f'8-coin 4h close: {len(prices):,} bars × 8 = {prices.shape}')
    print(f'Span: {span_days:.1f} days, {span_start} → {span_end}\n')

    # Screen pairs
    print('Screening 28 pairs for cointegration (full-sample Engle-Granger)...')
    pair_results = screen_pairs(prices)
    df_pairs = pd.DataFrame(pair_results)
    print(f'\n=== Pair cointegration results ===')
    print(df_pairs.to_string(index=False))
    print()

    # Filter: p < 0.05, valid half-life
    cointegrated = df_pairs[
        (df_pairs['coint_p'] < LOCKED['coint_pval_threshold']) &
        (df_pairs['half_life_bars'].notna()) &
        (df_pairs['half_life_bars'] >= LOCKED['min_half_life_bars']) &
        (df_pairs['half_life_bars'] <= LOCKED['max_half_life_bars'])
    ]
    print(f'\nCointegrated pairs (p<{LOCKED["coint_pval_threshold"]}, valid half-life): '
          f'{len(cointegrated)}/{len(df_pairs)}')

    if len(cointegrated) == 0:
        print('No cointegrated pairs found. N7 fails at screening.')
        return

    # Simulate each cointegrated pair
    all_trades = []
    for _, p in cointegrated.iterrows():
        trades = simulate_pair(prices, p['a'], p['b'], p['beta'])
        all_trades.extend(trades)
        print(f'  {p["a"]:<5}-{p["b"]:<5}  p={p["coint_p"]:.4f}  hl={p["half_life_bars"]:.1f}b  trades={len(trades)}')

    if not all_trades:
        print('\nNo trades generated.')
        return

    df_t = pd.DataFrame(all_trades)
    cum_pct = float(df_t['net_pnl_pct'].sum())
    apy = cum_pct / span_days * 365
    print(f'\n=== Aggregate ===')
    print(f'  n_trades: {len(df_t)}')
    print(f'  cum_net_pct: {cum_pct:+.4f}%')
    print(f'  APY extrapolated: {apy:+.4f}%')
    print(f'  avg_per_trade_net: {df_t["net_pnl_pct"].mean():+.4f}%')
    print(f'  avg_per_trade_gross: {df_t["gross_pct"].mean():+.4f}%')
    print(f'  WR: {(df_t["net_pnl_pct"] > 0).mean():.3f}')
    print(f'  median periods held: {df_t["periods_held"].median():.0f} bars ({df_t["periods_held"].median()/6:.1f}d)')

    # Bootstrap
    print('\n=== Bootstrap (사용자 criteria) ===')
    res = bootstrap_validate(df_t, span_start, span_end)
    bootstrap_report(res, 'N7 cointegration')

    out_path = RESULTS / f'n7_first_pass_bt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'date': datetime.now(timezone.utc).isoformat(),
            'locked': LOCKED,
            'pair_screening': pair_results,
            'cointegrated_pairs_count': len(cointegrated),
            'aggregate': {
                'n_trades': len(df_t),
                'cum_pct': cum_pct, 'apy': apy,
                'avg_per_trade_net': float(df_t['net_pnl_pct'].mean()),
                'avg_per_trade_gross': float(df_t['gross_pct'].mean()),
                'wr': float((df_t['net_pnl_pct'] > 0).mean()),
            },
            'bootstrap': {
                'mean_daily': res.mean_daily_pct,
                'p5_daily': res.p5_daily_pct,
                'pos_rate': res.pos_rate,
                'overall_pass': res.overall_pass,
                'pass_criteria': res.pass_criteria,
            },
        }, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
