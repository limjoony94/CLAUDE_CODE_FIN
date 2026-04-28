"""Path B R1 — Cross-Sectional Momentum (10 crypto, daily, weekly rebalance) LOCKED OOS.

Pre-registered (claudedocs/path_b_r1_xs_momentum_prereg.md, commit a597b1e).
Theory: Jegadeesh-Titman 1993 + Liu-Tsyvinski 2021 crypto momentum factor.

Locked params:
  universe={10 coins}, lookback_days=30, long_top_n=3, short_bottom_n=3
  rebalance=7d, friction=0.07% per transaction, equal-weight

Tests: dispersion gate, WF 5-fold, bootstrap 1000×30d, train/test 60/40.
"""
import json, random
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data' / 'multi_asset_daily.parquet'

LOCKED = {
    'lookback_days': 30,
    'long_top_n': 3,
    'short_bottom_n': 3,
    'rebalance_frequency_days': 7,
    'friction_per_transaction': 0.07,  # %
}
UNIVERSE = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'LINK/USDT']
DISPERSION_FLOOR = 5.0  # %


def load_pivot() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df['date'] = pd.to_datetime(df['date'])
    pivot_close = df.pivot(index='date', columns='symbol', values='close').sort_index()
    # Restrict to universe in case extra symbols
    pivot_close = pivot_close[[c for c in UNIVERSE if c in pivot_close.columns]]
    pivot_close = pivot_close.dropna(how='any')  # drop dates missing any coin
    return pivot_close


def run_xs_momentum(prices: pd.DataFrame, friction_pct: float = 0.07) -> pd.DataFrame:
    """Backtest cross-sectional momentum.
    Returns DataFrame with daily returns + portfolio info.
    """
    look = LOCKED['lookback_days']
    rebal = LOCKED['rebalance_frequency_days']
    n_long = LOCKED['long_top_n']
    n_short = LOCKED['short_bottom_n']

    dates = prices.index
    n_dates = len(dates)
    if n_dates < look + 7:
        return pd.DataFrame()

    # Daily returns per coin
    daily_ret = prices.pct_change().fillna(0)

    # Trailing N-day returns
    trail_ret = (prices / prices.shift(look) - 1) * 100

    # Position vector per date (1 = long, -1 = short, 0 = flat)
    n_coins = prices.shape[1]
    pos = pd.DataFrame(0.0, index=dates, columns=prices.columns)

    # Iterate by day; rebalance every `rebal` days starting after `look` warmup
    last_rebal = -1
    cur_long = []
    cur_short = []
    for i in range(look, n_dates):
        if i - last_rebal >= rebal:
            ranks = trail_ret.iloc[i].dropna().sort_values(ascending=False)
            if len(ranks) < n_long + n_short:
                continue
            cur_long = ranks.head(n_long).index.tolist()
            cur_short = ranks.tail(n_short).index.tolist()
            last_rebal = i
        # Apply position
        for s in cur_long:
            pos.iat[i, prices.columns.get_loc(s)] = 1.0 / n_long
        for s in cur_short:
            pos.iat[i, prices.columns.get_loc(s)] = -1.0 / n_short

    # Daily portfolio returns: sum(pos_t * daily_ret_t) — pos applied to current day's return
    # We use lag-1: position at end of day t-1 earns return on day t (no look-ahead)
    pos_lag = pos.shift(1).fillna(0)
    port_ret_pct = (pos_lag * daily_ret).sum(axis=1) * 100

    # Friction: compute turnover per day = sum(|pos_t - pos_{t-1}|)
    turnover = (pos - pos_lag).abs().sum(axis=1)
    # Friction cost per day (per side, 0.07% × turnover units)
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
    """Summary stats."""
    if bt.empty:
        return {'n': 0}
    n = len(bt)
    days = n
    weeks = n / 7
    cum_net = (1 + bt['net_pct'] / 100).cumprod().iloc[-1] - 1
    avg_daily = bt['net_pct'].mean()
    avg_weekly = avg_daily * 7
    avg_gross_weekly = bt['port_ret_pct'].mean() * 7
    avg_friction_weekly = bt['friction_pct'].mean() * 7
    pos_days = (bt['net_pct'] > 0).sum()
    wr = pos_days / n * 100
    daily_std = bt['net_pct'].std()
    sharpe = avg_daily / max(daily_std, 1e-9) * (365 ** 0.5)
    eq = (1 + bt['net_pct'] / 100).cumprod()
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max * 100
    max_dd = dd.min()
    return {
        'n_days': n, 'n_weeks': round(weeks, 1),
        'cum_net_pct': round(cum_net * 100, 2),
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
    print("Path B R1 — Cross-Sectional Momentum (10 Crypto, Daily, Weekly Rebalance) LOCKED OOS")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Pre-reg: claudedocs/path_b_r1_xs_momentum_prereg.md (commit a597b1e)")

    prices = load_pivot()
    n_dates = len(prices)
    n_coins = prices.shape[1]
    print(f'\nDaily prices: {n_dates} days × {n_coins} coins, range: {prices.index.min().date()} → {prices.index.max().date()}')
    print(f'Universe: {sorted(prices.columns.tolist())}\n')

    # Dispersion gate
    look = LOCKED['lookback_days']
    trail_ret = (prices / prices.shift(look) - 1) * 100
    dispersion = trail_ret.std(axis=1)
    median_disp = dispersion.median()
    print(f"Dispersion gate (median std of trailing {look}d returns): {median_disp:.2f}% (floor {DISPERSION_FLOOR}%)")
    if median_disp < DISPERSION_FLOOR:
        print(f"  ⚠️  DISPERSION FAIL — R1 INCONCLUSIVE")
        out = {'date': datetime.now(timezone.utc).isoformat(), 'pre_reg_commit': 'a597b1e',
               'locked_params': LOCKED, 'dispersion_median_pct': median_disp,
               'dispersion_floor': DISPERSION_FLOOR, 'verdict': 'INCONCLUSIVE — dispersion below floor'}
        p = ROOT / 'results' / f'path_b_r1_xs_mom_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
        print(f'\nSaved: {p}')
        return
    print('  ✓ Dispersion PASS\n')

    # Full-sample backtest
    bt_full = run_xs_momentum(prices, friction_pct=LOCKED['friction_per_transaction'])
    sumf = summarize(bt_full)
    print(f"=== Full-sample BT ===")
    for k, v in sumf.items():
        print(f"  {k}: {v}")
    breakeven_weekly_friction = sumf['avg_weekly_friction_pct']
    print(f"\n  Breakeven gross/week needed: {breakeven_weekly_friction:.4f}%")
    print(f"  Actual avg gross/week: {sumf['avg_weekly_gross_pct']:.4f}%")
    print(f"  Edge above friction: {sumf['avg_weekly_gross_pct'] - breakeven_weekly_friction:+.4f}%")

    # TEST 1: WF 5-fold expanding
    print(f"\n{'='*100}\nTEST 1 — WF 5-fold expanding (avg_weekly_net prominent)\n{'='*100}")
    print(f"  {'fold':>4} {'days':>5} {'avg_wk_gross':>13} {'avg_wk_fric':>12} {'avg_wk_net':>11} {'sharpe':>8} {'max_dd':>9}")
    print(f"  {'----':>4} {'-----':>5} {'-------------':>13} {'------------':>12} {'-----------':>11} {'--------':>8} {'---------':>9}")
    fold_size = n_dates // 6
    wf_results = []
    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_dates)
        sub = prices.iloc[te_s:te_e]
        bt = run_xs_momentum(sub, friction_pct=LOCKED['friction_per_transaction'])
        s = summarize(bt)
        if s.get('n_days', 0) == 0:
            wf_results.append({'fold': fold_i+1, 'n_days': 0, 'avg_weekly_net_pct': None})
            continue
        wf_results.append({'fold': fold_i+1, **s})
        gate = '✓' if s['avg_weekly_net_pct'] > 0 else '✗'
        print(f"  {fold_i+1:>4} {s['n_days']:>5} {s['avg_weekly_gross_pct']:>+12.4f}% {s['avg_weekly_friction_pct']:>+11.4f}% {s['avg_weekly_net_pct']:>+10.4f}% {gate} {s['sharpe_annualized']:>+7.2f} {s['max_dd_pct']:>+8.2f}%")

    wf_pos = sum(1 for r in wf_results if r.get('avg_weekly_net_pct') is not None and r['avg_weekly_net_pct'] > 0)
    wf_pass = wf_pos >= 3
    print(f"\n  WF positive folds: {wf_pos}/5 → {'PASS' if wf_pass else 'FAIL'}")

    # TEST 2: Bootstrap 1000 × 30-day windows of NET returns (proper portfolio bootstrap)
    # NOTE: bootstrap on returns from full-sample BT preserves lookback context.
    # Slicing prices to 30d would zero out the strategy (lookback=30d).
    print(f"\n{'='*100}\nTEST 2 — Bootstrap 1000 × 30-day NET-return windows (from full BT)\n{'='*100}")
    net_returns = bt_full['net_pct'].values  # daily net % from full-sample BT
    n_returns = len(net_returns)
    win = 30
    max_start = n_returns - win - 1
    random.seed(42)
    starts = random.sample(range(max(1, max_start)), min(1000, max(1, max_start)))
    pnls = []
    for st in starts:
        en = st + win
        slice_ = net_returns[st:en]
        cum = float((1 + slice_ / 100).prod() - 1) * 100
        pnls.append(cum)
    arr = np.array(pnls)
    bs_mean = float(arr.mean())
    bs_pos_rate = float((arr > 0).mean())
    bs_p5 = float(np.percentile(arr, 5))
    bs_pass = bs_pos_rate >= 0.5
    print(f"  mean (30d cum %): {bs_mean:+.4f}%")
    print(f"  pos_rate: {bs_pos_rate:.4f} ({int(bs_pos_rate*1000)}/1000)")
    print(f"  p5: {bs_p5:+.4f}%")
    print(f"  Pos rate ≥ 0.50: {'PASS' if bs_pass else 'FAIL'}")

    # TEST 3: Train/Test 60/40
    print(f"\n{'='*100}\nTEST 3 — Train/Test 60/40\n{'='*100}")
    train_end = int(n_dates * 0.6)
    p_tr = prices.iloc[:train_end]
    p_te = prices.iloc[train_end:]
    bt_tr = run_xs_momentum(p_tr, friction_pct=LOCKED['friction_per_transaction'])
    bt_te = run_xs_momentum(p_te, friction_pct=LOCKED['friction_per_transaction'])
    s_tr = summarize(bt_tr)
    s_te = summarize(bt_te)
    print(f"  train: days={s_tr['n_days']} avg_wk_net={s_tr['avg_weekly_net_pct']:+.4f}% (gross={s_tr['avg_weekly_gross_pct']:+.4f}%, fric={s_tr['avg_weekly_friction_pct']:.4f}%) sharpe={s_tr['sharpe_annualized']:+.2f} dd={s_tr['max_dd_pct']:+.2f}%")
    print(f"  test:  days={s_te['n_days']} avg_wk_net={s_te['avg_weekly_net_pct']:+.4f}% (gross={s_te['avg_weekly_gross_pct']:+.4f}%, fric={s_te['avg_weekly_friction_pct']:.4f}%) sharpe={s_te['sharpe_annualized']:+.2f} dd={s_te['max_dd_pct']:+.2f}%")
    tt_pass = (s_tr['avg_weekly_net_pct'] > 0) and (s_te['avg_weekly_net_pct'] > 0)
    print(f"  Train+Test both > 0: {'PASS' if tt_pass else 'FAIL'}")

    # FINAL
    print(f"\n{'='*100}\nPath B R1 FINAL VERDICT\n{'='*100}")
    print(f"  Dispersion gate:                 PASS ({median_disp:.2f}%)")
    print(f"  Test 1 (WF ≥3/5):                {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap ≥50%):         {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):  {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"  OVERALL: {'ALL 3 PASS — call advisor before any claim' if all_pass else 'FAIL — Path B R1 negative'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'a597b1e',
        'locked_params': LOCKED, 'universe': sorted(prices.columns.tolist()),
        'dispersion_median_pct': float(median_disp), 'dispersion_pass': True,
        'full_sample': sumf,
        'test_1_wf': {'folds': wf_results, 'pos_count': wf_pos, 'pass': wf_pass},
        'test_2_bootstrap_30d': {'mean_pct': bs_mean, 'pos_rate': bs_pos_rate,
                                   'p5_pct': bs_p5, 'pass': bs_pass},
        'test_3_train_test': {
            'train': s_tr, 'test': s_te, 'pass': tt_pass,
        },
        'all_pass': bool(all_pass),
    }
    p = ROOT / 'results' / f'path_b_r1_xs_mom_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
