"""Path B R2 — Cross-Sectional Reversal (10 crypto, daily, weekly rebalance) LOCKED OOS.

Pre-registered (claudedocs/path_b_r2_xs_reversal_prereg.md, commit 6cb07f0).
Theory: De Bondt-Thaler 1985 + Lehmann 1990 short-term reversal.

Locked params:
  lookback_days=7, long_bottom_n=3, short_top_n=3 (REVERSED from R1)
  rebalance=7d, friction=0.07% per transaction
"""
import json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data' / 'multi_asset_daily.parquet'

LOCKED = {
    'lookback_days': 7,
    'long_bottom_n': 3,
    'short_top_n': 3,
    'rebalance_frequency_days': 7,
    'friction_per_transaction': 0.07,
}
UNIVERSE = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'LINK/USDT']
DISPERSION_FLOOR = 5.0


def load_pivot() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df['date'] = pd.to_datetime(df['date'])
    pivot = df.pivot(index='date', columns='symbol', values='close').sort_index()
    pivot = pivot[[c for c in UNIVERSE if c in pivot.columns]]
    return pivot.dropna(how='any')


def run_xs_reversal(prices: pd.DataFrame, friction_pct: float = 0.07) -> pd.DataFrame:
    """REVERSAL — long bottom (worst recent), short top (best recent)."""
    look = LOCKED['lookback_days']
    rebal = LOCKED['rebalance_frequency_days']
    n_long = LOCKED['long_bottom_n']
    n_short = LOCKED['short_top_n']

    dates = prices.index
    n_dates = len(dates)
    if n_dates < look + 7:
        return pd.DataFrame()

    daily_ret = prices.pct_change().fillna(0)
    trail_ret = (prices / prices.shift(look) - 1) * 100

    pos = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    last_rebal = -1
    cur_long = []
    cur_short = []
    for i in range(look, n_dates):
        if i - last_rebal >= rebal:
            ranks = trail_ret.iloc[i].dropna().sort_values(ascending=False)
            if len(ranks) < n_long + n_short:
                continue
            # REVERSAL: top of ranks = winners → SHORT; bottom = losers → LONG
            cur_short = ranks.head(n_short).index.tolist()
            cur_long = ranks.tail(n_long).index.tolist()
            last_rebal = i
        for s in cur_long:
            pos.iat[i, prices.columns.get_loc(s)] = 1.0 / n_long
        for s in cur_short:
            pos.iat[i, prices.columns.get_loc(s)] = -1.0 / n_short

    pos_lag = pos.shift(1).fillna(0)
    port_ret_pct = (pos_lag * daily_ret).sum(axis=1) * 100
    turnover = (pos - pos_lag).abs().sum(axis=1)
    fric_pct = turnover * friction_pct
    out = pd.DataFrame({
        'date': dates,
        'port_ret_pct': port_ret_pct.values,
        'turnover': turnover.values,
        'friction_pct': fric_pct.values,
        'net_pct': (port_ret_pct - fric_pct).values,
    })
    return out


def summarize(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {'n_days': 0}
    n = len(bt)
    avg_daily = bt['net_pct'].mean()
    avg_weekly = avg_daily * 7
    avg_gross_weekly = bt['port_ret_pct'].mean() * 7
    avg_friction_weekly = bt['friction_pct'].mean() * 7
    pos_days = (bt['net_pct'] > 0).sum()
    wr = pos_days / n * 100 if n > 0 else 0
    daily_std = bt['net_pct'].std()
    sharpe = avg_daily / max(daily_std, 1e-9) * (365 ** 0.5)
    eq = (1 + bt['net_pct'] / 100).cumprod()
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max * 100
    max_dd = dd.min()
    cum_net = float(eq.iloc[-1] - 1) * 100
    return {
        'n_days': n, 'cum_net_pct': round(cum_net, 2),
        'avg_daily_pct': round(avg_daily, 4),
        'avg_weekly_gross_pct': round(avg_gross_weekly, 4),
        'avg_weekly_friction_pct': round(avg_friction_weekly, 4),
        'avg_weekly_net_pct': round(avg_weekly, 4),
        'wr_daily_pct': round(wr, 2),
        'sharpe_annualized': round(sharpe, 3),
        'max_dd_pct': round(max_dd, 2),
    }


def main():
    print("=" * 100)
    print("Path B R2 — Cross-Sectional REVERSAL (10 Crypto, Daily, Weekly) LOCKED OOS")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Pre-reg: claudedocs/path_b_r2_xs_reversal_prereg.md (commit 6cb07f0)")

    prices = load_pivot()
    n_dates = len(prices)
    print(f'\nDaily prices: {n_dates} days × {prices.shape[1]} coins\n')

    look = LOCKED['lookback_days']
    trail_ret = (prices / prices.shift(look) - 1) * 100
    dispersion = trail_ret.std(axis=1)
    median_disp = dispersion.median()
    print(f"Dispersion gate (trailing {look}d): {median_disp:.2f}% (floor {DISPERSION_FLOOR}%)")
    if median_disp < DISPERSION_FLOOR:
        print(f"  ⚠️  DISPERSION FAIL — INCONCLUSIVE")
        return
    print('  ✓ Dispersion PASS\n')

    bt_full = run_xs_reversal(prices, friction_pct=LOCKED['friction_per_transaction'])
    sumf = summarize(bt_full)
    print("=== Full-sample BT ===")
    for k, v in sumf.items():
        print(f"  {k}: {v}")
    print(f"\n  Edge above friction: {sumf['avg_weekly_gross_pct'] - sumf['avg_weekly_friction_pct']:+.4f}%/wk")

    print(f"\n{'='*100}\nTEST 1 — WF 5-fold expanding\n{'='*100}")
    print(f"  {'fold':>4} {'days':>5} {'avg_wk_gross':>13} {'avg_wk_fric':>12} {'avg_wk_net':>11} {'sharpe':>8} {'max_dd':>9}")
    fold_size = n_dates // 6
    wf_results = []
    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_dates)
        sub = prices.iloc[te_s:te_e]
        bt = run_xs_reversal(sub, friction_pct=LOCKED['friction_per_transaction'])
        s = summarize(bt)
        if s.get('n_days', 0) == 0:
            wf_results.append({'fold': fold_i+1, 'n_days': 0, 'avg_weekly_net_pct': None})
            continue
        wf_results.append({'fold': fold_i+1, **s})
        gate = '✓' if s['avg_weekly_net_pct'] > 0 else '✗'
        print(f"  {fold_i+1:>4} {s['n_days']:>5} {s['avg_weekly_gross_pct']:>+12.4f}% {s['avg_weekly_friction_pct']:>+11.4f}% {s['avg_weekly_net_pct']:>+10.4f}% {gate} {s['sharpe_annualized']:>+7.2f} {s['max_dd_pct']:>+8.2f}%")
    wf_pos = sum(1 for r in wf_results if r.get('avg_weekly_net_pct') is not None and r['avg_weekly_net_pct'] > 0)
    wf_pass = wf_pos >= 3
    print(f"  WF positive folds: {wf_pos}/5 → {'PASS' if wf_pass else 'FAIL'}")

    # Bootstrap
    print(f"\n{'='*100}\nTEST 2 — Bootstrap 1000 × 30-day NET-return windows (from full BT)\n{'='*100}")
    net_returns = bt_full['net_pct'].values
    n_returns = len(net_returns)
    win = 30
    max_start = n_returns - win - 1
    random.seed(42)
    starts = random.sample(range(max(1, max_start)), min(1000, max(1, max_start)))
    pnls = []
    for st in starts:
        slice_ = net_returns[st:st + win]
        cum = float((1 + slice_ / 100).prod() - 1) * 100
        pnls.append(cum)
    arr = np.array(pnls)
    bs_mean = float(arr.mean())
    bs_pos_rate = float((arr > 0).mean())
    bs_p5 = float(np.percentile(arr, 5))
    bs_pass = bs_pos_rate >= 0.5
    print(f"  mean (30d cum): {bs_mean:+.4f}% pos_rate={bs_pos_rate:.4f} p5={bs_p5:+.4f}%")
    print(f"  Pos rate ≥ 0.50: {'PASS' if bs_pass else 'FAIL'}")

    # Train/Test
    print(f"\n{'='*100}\nTEST 3 — Train/Test 60/40\n{'='*100}")
    train_end = int(n_dates * 0.6)
    bt_tr = run_xs_reversal(prices.iloc[:train_end], friction_pct=LOCKED['friction_per_transaction'])
    bt_te = run_xs_reversal(prices.iloc[train_end:], friction_pct=LOCKED['friction_per_transaction'])
    s_tr = summarize(bt_tr)
    s_te = summarize(bt_te)
    print(f"  train: days={s_tr['n_days']} avg_wk_net={s_tr['avg_weekly_net_pct']:+.4f}% (gross={s_tr['avg_weekly_gross_pct']:+.4f}%, fric={s_tr['avg_weekly_friction_pct']:.4f}%) sharpe={s_tr['sharpe_annualized']:+.2f} dd={s_tr['max_dd_pct']:+.2f}%")
    print(f"  test:  days={s_te['n_days']} avg_wk_net={s_te['avg_weekly_net_pct']:+.4f}% (gross={s_te['avg_weekly_gross_pct']:+.4f}%, fric={s_te['avg_weekly_friction_pct']:.4f}%) sharpe={s_te['sharpe_annualized']:+.2f} dd={s_te['max_dd_pct']:+.2f}%")
    tt_pass = (s_tr['avg_weekly_net_pct'] > 0) and (s_te['avg_weekly_net_pct'] > 0)
    print(f"  Train+Test both > 0: {'PASS' if tt_pass else 'FAIL'}")

    print(f"\n{'='*100}\nPath B R2 FINAL VERDICT\n{'='*100}")
    print(f"  Dispersion gate:                 PASS ({median_disp:.2f}%)")
    print(f"  Test 1 (WF ≥3/5):                {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap ≥50%):         {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):  {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"  OVERALL: {'ALL 3 PASS' if all_pass else 'FAIL'}")
    edge_above_friction = sumf['avg_weekly_gross_pct'] - sumf['avg_weekly_friction_pct']
    print(f"\n  Edge above friction: {edge_above_friction:+.4f}%/wk → {'gross > friction' if edge_above_friction > 0 else 'GROSS BELOW FRICTION (broken econ)'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '6cb07f0',
        'locked_params': LOCKED,
        'dispersion_median_pct': float(median_disp),
        'full_sample': sumf,
        'test_1_wf': {'folds': wf_results, 'pos_count': wf_pos, 'pass': wf_pass},
        'test_2_bootstrap_30d': {'mean_pct': bs_mean, 'pos_rate': bs_pos_rate, 'p5_pct': bs_p5, 'pass': bs_pass},
        'test_3_train_test': {'train': s_tr, 'test': s_te, 'pass': tt_pass},
        'all_pass': bool(all_pass),
        'edge_above_friction_pct_per_week': edge_above_friction,
    }
    p = ROOT / 'results' / f'path_b_r2_xs_reversal_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
