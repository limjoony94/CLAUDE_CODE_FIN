"""Path B R13 — Multi-Coin Cash-and-Carry Portfolio LOCKED OOS.

Pre-reg: claudedocs/path_b_r13_multicoin_cash_and_carry_prereg.md (commit 825684d)

Mechanism:
  Parallel cash-and-carry on 8 coins: LINK, DOGE, ADA, ETH, BTC, XRP, SOL, AVAX.
  Each coin: long spot ($93.75) + short perp ($93.75), delta-neutral, $187.50/pair.
  Excluded: BNB (-0.72%/yr historical funding), TRX (-0.12%/yr).

  Per-coin regime filter: enter when 7d trailing funding APY >= 3%, exit when <= 0%.
  Daily portfolio NAV = sum across coins of (funding_pnl - friction).

Data: data/funding_history.parquet — 10 coins, 2024-02-19 to 2026-04-29 (~800 days).

Tests:
  Gate A: >= 5/8 coins have >= 100 days where regime filter active.
  Gate B: random-baseline anti-fix-impulse.
  T1 WF 5-fold (>=3/5 positive)
  T2 Bootstrap 1000 x 30d (pos_rate >= 50%)
  T3 Train/Test 60/40 (BOTH positive)
  T4 (HARD) daily >= 0.2%
  T5/T6 carry-implicit (rebalance hit-rate / natural ratio)
  T7 (HARD) trades/day >= 2
  T8 (HARD) per-trade gross > 0.07%
  T9 worst 5d >= -15%
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

FUNDING_FILE = DATA / 'funding_history.parquet'

LOCKED = {
    'capital_usd': 1500,
    'coin_universe': ['LINK/USDT', 'DOGE/USDT', 'ADA/USDT', 'ETH/USDT',
                      'BTC/USDT', 'XRP/USDT', 'SOL/USDT', 'AVAX/USDT'],
    'pair_capital_usd': 187.50,        # 1500 / 8
    'per_leg_usd': 93.75,              # 187.50 / 2
    'spot_friction_per_side_pct': 0.10,
    'perp_friction_per_side_pct': 0.04,
    'entry_threshold_apy_pct': 3.0,
    'exit_threshold_apy_pct': 0.0,
    'lookback_funding_periods_for_regime': 21,  # 7d x 3/day
}

GATES = {
    'gate_A_min_active_coins': 5,
    'gate_A_min_active_days_per_coin': 100,
    'wf_min_pos_folds': 3,
    'wf_total_folds': 5,
    'bs_n_iter': 1000,
    'bs_window_days': 30,
    'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_t4_min_daily_pct': 0.20,
    'tail_t9_max_5d_dd_pct': 15.0,
    't7_min_trades_per_day': 2.0,
    't8_min_per_trade_gross_pct': 0.07,
}

N = len(LOCKED['coin_universe'])


def load_funding() -> pd.DataFrame:
    df = pd.read_parquet(FUNDING_FILE)
    df = df[df['symbol'].isin(LOCKED['coin_universe'])].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    return df


def daily_funding_per_coin(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to date x coin matrix of summed daily funding rates."""
    fund_df = fund_df.copy()
    fund_df['date'] = fund_df['datetime'].dt.tz_localize(None).dt.floor('D')
    daily = (fund_df.groupby(['date', 'symbol'])['funding_rate']
             .sum()
             .unstack('symbol')
             .sort_index())
    return daily


def run_per_coin_carry(daily_fund_per_coin: pd.Series, coin: str) -> pd.DataFrame:
    """Single-coin carry backtest at coin-pair level (P&L in $ on $187.50 pair capital)."""
    df = pd.DataFrame({'date': daily_fund_per_coin.index,
                       'daily_funding': daily_fund_per_coin.values})
    df['rolling_7d_apy'] = df['daily_funding'].rolling(7).mean() * 365 * 100

    in_pos = False
    rows = []
    pair_cap = LOCKED['pair_capital_usd']
    leg_usd = LOCKED['per_leg_usd']

    for _, row in df.iterrows():
        action = 'hold' if in_pos else 'flat'
        friction_pct_of_pair = 0.0
        funding_pnl_pct_of_pair = 0.0
        rolling_apy = row['rolling_7d_apy']

        if not in_pos:
            if pd.notna(rolling_apy) and rolling_apy >= LOCKED['entry_threshold_apy_pct']:
                in_pos = True
                action = 'enter'
                friction_pct_of_pair = (
                    (LOCKED['spot_friction_per_side_pct'] * leg_usd / pair_cap) +
                    (LOCKED['perp_friction_per_side_pct'] * leg_usd / pair_cap)
                )
        else:
            funding_pnl_pct_of_pair = (
                row['daily_funding'] * leg_usd / pair_cap
            ) * 100
            if pd.notna(rolling_apy) and rolling_apy <= LOCKED['exit_threshold_apy_pct']:
                in_pos = False
                action = 'exit'
                friction_pct_of_pair = (
                    (LOCKED['spot_friction_per_side_pct'] * leg_usd / pair_cap) +
                    (LOCKED['perp_friction_per_side_pct'] * leg_usd / pair_cap)
                )

        net_pct_pair = funding_pnl_pct_of_pair - friction_pct_of_pair
        rows.append({
            'date': row['date'],
            'coin': coin,
            'daily_funding': row['daily_funding'],
            'rolling_7d_apy': rolling_apy,
            'in_position': in_pos,
            'action': action,
            'pair_funding_pnl_pct': funding_pnl_pct_of_pair,
            'pair_friction_pct': friction_pct_of_pair,
            'pair_net_pct': net_pct_pair,
        })
    return pd.DataFrame(rows)


def run_portfolio(daily_fund_per_coin: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-coin backtests into portfolio NAV."""
    coin_bts = []
    for coin in LOCKED['coin_universe']:
        if coin not in daily_fund_per_coin.columns:
            continue
        bt = run_per_coin_carry(daily_fund_per_coin[coin], coin)
        coin_bts.append(bt)

    long_df = pd.concat(coin_bts, ignore_index=True)

    # Each pair capital is 1/N of portfolio.
    # Per-pair pct contributes (1/N) * pair_pct to portfolio pct.
    long_df['portfolio_funding_pnl_pct'] = long_df['pair_funding_pnl_pct'] / N
    long_df['portfolio_friction_pct'] = long_df['pair_friction_pct'] / N
    long_df['portfolio_net_pct'] = long_df['pair_net_pct'] / N

    # Aggregate by date
    agg = long_df.groupby('date').agg(
        portfolio_funding_pnl_pct=('portfolio_funding_pnl_pct', 'sum'),
        portfolio_friction_pct=('portfolio_friction_pct', 'sum'),
        portfolio_net_pct=('portfolio_net_pct', 'sum'),
        coins_in_position=('in_position', 'sum'),
        n_entries=('action', lambda s: (s == 'enter').sum()),
        n_exits=('action', lambda s: (s == 'exit').sum()),
    ).reset_index()
    agg['n_trades'] = agg['n_entries'] + agg['n_exits']

    return agg, long_df


def gate_A(long_df: pd.DataFrame) -> dict:
    """>=5/8 coins must have >=100 days where rolling 7d APY >= 3%."""
    counts = (long_df.groupby('coin')
              .apply(lambda d: int((d['rolling_7d_apy'] >= LOCKED['entry_threshold_apy_pct']).sum())))
    active = (counts >= GATES['gate_A_min_active_days_per_coin']).sum()
    return {
        'per_coin_active_days': counts.to_dict(),
        'active_coins': int(active),
        'gate': GATES['gate_A_min_active_coins'],
        'pass': int(active) >= GATES['gate_A_min_active_coins'],
    }


def gate_B_random_baseline(daily_fund_per_coin: pd.DataFrame, agg_actual: pd.DataFrame,
                           n_iter: int = 1000) -> dict:
    """Random-entry baseline: shuffle each coin's funding series independently."""
    rng = np.random.default_rng(42)
    actual_cum = float((1 + agg_actual['portfolio_net_pct'] / 100).prod() - 1) * 100

    # Build per-coin daily series, shuffle, re-run portfolio
    cums = []
    for _ in range(n_iter):
        shuffled = pd.DataFrame(index=daily_fund_per_coin.index)
        for coin in daily_fund_per_coin.columns:
            arr = daily_fund_per_coin[coin].values.copy()
            rng.shuffle(arr)
            shuffled[coin] = arr
        agg_shuf, _ = run_portfolio(shuffled)
        cums.append(float((1 + agg_shuf['portfolio_net_pct'] / 100).prod() - 1) * 100)
    arr = np.array(cums)
    p95 = float(np.percentile(arr, 95))
    return {
        'actual_cum_pct': actual_cum,
        'random_p95_pct': p95,
        'random_mean_pct': float(arr.mean()),
        'random_min_pct': float(arr.min()),
        'random_max_pct': float(arr.max()),
        'pass': actual_cum > p95,
    }


def summarize(agg: pd.DataFrame, long_df: pd.DataFrame) -> dict:
    if agg.empty:
        return {'n_days': 0}
    n = len(agg)
    cum_net = float((1 + agg['portfolio_net_pct'] / 100).prod() - 1) * 100
    cum_funding = float(agg['portfolio_funding_pnl_pct'].sum())
    cum_friction = float(agg['portfolio_friction_pct'].sum())

    avg_daily_net = float(agg['portfolio_net_pct'].mean())
    annualized_net_apy = avg_daily_net * 365

    nav = (1 + agg['portfolio_net_pct'].values / 100).cumprod()
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    max_dd = float(dd.max()) * 100

    rolling_5d = pd.Series(agg['portfolio_net_pct'].values).rolling(5).apply(
        lambda x: (1 + x / 100).prod() - 1
    ) * 100
    worst_5d = float(rolling_5d.min())

    daily_std = float(agg['portfolio_net_pct'].std())
    sharpe = (avg_daily_net / daily_std * (365 ** 0.5)) if daily_std > 0 else 0.0

    n_trades_total = int(agg['n_trades'].sum())
    avg_trades_per_day = n_trades_total / n if n > 0 else 0.0

    # Per-trade gross: gross funding earned divided by total trades
    # Total funding pnl pct of capital, n trades, per-trade-gross = total / n
    per_trade_gross = (cum_funding / n_trades_total) if n_trades_total > 0 else 0.0

    avg_coins_in_pos = float(agg['coins_in_position'].mean())

    return {
        'n_days': int(n),
        'cum_net_pct': cum_net,
        'cum_funding_pct': cum_funding,
        'cum_friction_pct': cum_friction,
        'avg_daily_net_pct': avg_daily_net,
        'annualized_net_apy_pct': annualized_net_apy,
        'avg_coins_in_position': avg_coins_in_pos,
        'n_trades_total': n_trades_total,
        'avg_trades_per_day': avg_trades_per_day,
        'per_trade_gross_pct': per_trade_gross,
        'max_dd_pct': max_dd,
        'worst_5d_net_pct': worst_5d,
        'daily_std_pct': daily_std,
        'sharpe_annualized': sharpe,
    }


def test_1_walk_forward(daily_fund_per_coin: pd.DataFrame) -> dict:
    folds = GATES['wf_total_folds']
    n = len(daily_fund_per_coin)
    fold_size = n // (folds + 1)
    results = []
    for i in range(folds):
        s = (i + 1) * fold_size
        e = min(s + fold_size, n)
        sub = daily_fund_per_coin.iloc[s:e]
        agg, long_df = run_portfolio(sub)
        summ = summarize(agg, long_df)
        results.append({'fold': i + 1, **summ})
    pos_count = sum(1 for r in results if r.get('cum_net_pct', 0) > 0)
    return {'folds': results, 'pos_count': pos_count,
            'pass': pos_count >= GATES['wf_min_pos_folds']}


def test_2_bootstrap(agg: pd.DataFrame) -> dict:
    nets = agg['portfolio_net_pct'].values
    n = len(nets)
    win = GATES['bs_window_days']
    if n <= win:
        return {'pass': False, 'reason': 'panel too short'}
    random.seed(42)
    starts = random.sample(range(n - win), min(GATES['bs_n_iter'], n - win))
    cums = [(1 + nets[s:s+win] / 100).prod() - 1 for s in starts]
    arr = np.array(cums) * 100
    pos_rate = float((arr > 0).mean())
    return {
        'n_iter': len(arr),
        'mean_cum_pct': float(arr.mean()),
        'pos_rate': pos_rate,
        'p5': float(np.percentile(arr, 5)),
        'p95': float(np.percentile(arr, 95)),
        'pass': pos_rate >= GATES['bs_min_pos_rate'],
    }


def test_3_train_test(daily_fund_per_coin: pd.DataFrame) -> dict:
    n = len(daily_fund_per_coin)
    split = int(n * GATES['tt_split'])
    agg_tr, ld_tr = run_portfolio(daily_fund_per_coin.iloc[:split])
    agg_te, ld_te = run_portfolio(daily_fund_per_coin.iloc[split:])
    s_tr = summarize(agg_tr, ld_tr)
    s_te = summarize(agg_te, ld_te)
    return {
        'train': s_tr, 'test': s_te,
        'pass': (s_tr.get('cum_net_pct', 0) > 0) and (s_te.get('cum_net_pct', 0) > 0),
    }


def main():
    print('=' * 100)
    print('Path B R13 — Multi-Coin Cash-and-Carry Portfolio (8 coins)')
    print('=' * 100)
    print('Pre-reg: claudedocs/path_b_r13_multicoin_cash_and_carry_prereg.md (825684d)')
    print(f'Universe: {LOCKED["coin_universe"]}')
    print(f'Capital: ${LOCKED["capital_usd"]} / N=8 = ${LOCKED["pair_capital_usd"]}/pair, '
          f'${LOCKED["per_leg_usd"]}/leg\n')

    fund = load_funding()
    print(f'Funding rows: {len(fund):,} across {fund["symbol"].nunique()} symbols')
    daily = daily_funding_per_coin(fund)
    print(f'Daily aggregated: {daily.shape[0]} days x {daily.shape[1]} coins')
    print(f'Date range: {daily.index.min().date()} -> {daily.index.max().date()}\n')

    print('Per-coin daily mean APY%:')
    for c in daily.columns:
        apy = daily[c].mean() * 365 * 100
        print(f'  {c}: {apy:+.2f}%/yr')
    print()

    print('=== Full-sample portfolio backtest ===')
    agg_full, long_full = run_portfolio(daily)
    s_full = summarize(agg_full, long_full)
    for k, v in s_full.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    print('=== Gate A — Coin universe sufficiency ===')
    gA = gate_A(long_full)
    for c, n in gA['per_coin_active_days'].items():
        print(f'  {c}: {n} active days')
    print(f'  active coins (>={gA["gate"]} required): {gA["active_coins"]}/8'
          f'  -> {"PASS" if gA["pass"] else "FAIL"}\n')

    print('=== Gate B — Random-baseline (1000 shuffled) ===')
    gB = gate_B_random_baseline(daily, agg_full, n_iter=1000)
    print(f'  actual cum: {gB["actual_cum_pct"]:+.4f}%')
    print(f'  random p95:  {gB["random_p95_pct"]:+.4f}%  '
          f'(mean {gB["random_mean_pct"]:+.4f}, min {gB["random_min_pct"]:+.4f}, '
          f'max {gB["random_max_pct"]:+.4f})')
    print(f'  -> {"PASS" if gB["pass"] else "FAIL"}\n')

    print('=== T1 — WF 5-fold ===')
    t1 = test_1_walk_forward(daily)
    for f_ in t1['folds']:
        print(f'  fold {f_["fold"]}: cum_net={f_.get("cum_net_pct", 0):+.4f}%  '
              f'apy={f_.get("annualized_net_apy_pct", 0):+.2f}%  '
              f'trades/d={f_.get("avg_trades_per_day", 0):.2f}')
    print(f'  -> {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/5)\n')

    print('=== T2 — Bootstrap 1000 x 30d ===')
    t2 = test_2_bootstrap(agg_full)
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  mean: {t2.get("mean_cum_pct", 0):+.4f}%  '
          f'p5: {t2.get("p5", 0):+.4f}%  p95: {t2.get("p95", 0):+.4f}%')
    print(f'  -> {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== T3 — Train/Test 60/40 ===')
    t3 = test_3_train_test(daily)
    print(f'  train: cum_net={t3["train"].get("cum_net_pct", 0):+.4f}%  '
          f'apy={t3["train"].get("annualized_net_apy_pct", 0):+.2f}%')
    print(f'  test:  cum_net={t3["test"].get("cum_net_pct", 0):+.4f}%  '
          f'apy={t3["test"].get("annualized_net_apy_pct", 0):+.2f}%')
    print(f'  -> {"PASS" if t3["pass"] else "FAIL"}\n')

    avg_daily = s_full['avg_daily_net_pct']
    t4_pass = avg_daily >= GATES['magnitude_t4_min_daily_pct']
    print(f'=== T4 (HARD) Daily >= {GATES["magnitude_t4_min_daily_pct"]}% ===')
    print(f'  avg daily net: {avg_daily:+.4f}%  -> {"PASS" if t4_pass else "FAIL"}\n')

    avg_tpd = s_full['avg_trades_per_day']
    t7_pass = avg_tpd >= GATES['t7_min_trades_per_day']
    print(f'=== T7 (HARD) Trades/day >= {GATES["t7_min_trades_per_day"]} ===')
    print(f'  avg trades/day: {avg_tpd:.4f}  -> {"PASS" if t7_pass else "FAIL"}\n')

    pt_gross = s_full['per_trade_gross_pct']
    t8_pass = pt_gross > GATES['t8_min_per_trade_gross_pct']
    print(f'=== T8 (HARD) Per-trade gross > {GATES["t8_min_per_trade_gross_pct"]}% ===')
    print(f'  per-trade gross: {pt_gross:+.4f}%  -> {"PASS" if t8_pass else "FAIL"}\n')

    worst = s_full['worst_5d_net_pct']
    t9_pass = worst >= -GATES['tail_t9_max_5d_dd_pct']
    print(f'=== T9 Worst 5d >= -{GATES["tail_t9_max_5d_dd_pct"]}% ===')
    print(f'  worst 5d: {worst:+.4f}%  -> {"PASS" if t9_pass else "FAIL"}\n')

    hard_passes = [t4_pass, t7_pass, t8_pass]
    all_passes = [gA['pass'], gB['pass'], t1['pass'], t2['pass'], t3['pass'],
                  t4_pass, t7_pass, t8_pass, t9_pass]
    n_hard_pass = sum(hard_passes)
    n_total_pass = sum(all_passes)

    print('=' * 100)
    print(f'VERDICT: {n_total_pass}/9 PASS  |  HARD: {n_hard_pass}/3'
          f'  (T4 daily, T7 freq, T8 gross)')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '825684d',
        'locked': LOCKED,
        'gates': GATES,
        'full_sample': s_full,
        'gate_A': gA,
        'gate_B': gB,
        'test_1_wf': t1,
        'test_2_bootstrap': t2,
        'test_3_train_test': t3,
        't4_daily': {'avg_daily_pct': avg_daily, 'pass': bool(t4_pass)},
        't7_trades_per_day': {'avg_tpd': avg_tpd, 'pass': bool(t7_pass)},
        't8_per_trade_gross': {'gross_pct': pt_gross, 'pass': bool(t8_pass)},
        't9_worst_5d': {'worst_5d_pct': worst, 'pass': bool(t9_pass)},
        'verdict_total_pass': n_total_pass,
        'verdict_hard_pass': n_hard_pass,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r13_multicoin_carry_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
