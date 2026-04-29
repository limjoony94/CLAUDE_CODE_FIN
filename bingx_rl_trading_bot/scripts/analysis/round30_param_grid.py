"""Round 30 — Parameter Grid Search (Entry / SL / TP) with Train/Test Split.

Pre-reg: claudedocs/round30_param_grid_prereg.md (commit 7c9a4f5)

27 configs (3 entry body × 3 SL mult × 3 TP mult) on R29 baseline (15m fade).
Train (first 60%) winner selection → Test (last 40%) OOS validation.
"""
import json
import random
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

DATA_FILE = DATA / 'btc_15m_720days.csv'

GRID = {
    'entry_body_filter_pct': [30, 50, 70],
    'sl_period_range_multiple': [0.5, 1.0, 1.5],
    'tp_period_range_multiple': [1.5, 2.5, 4.0],
}

FIXED = {
    'asset': 'BTC/USDT',
    'tf': '15m',
    'period_lookback_bars': 16,
    'direction': 'fade',
    'max_hold_bars': 96,
    'taker_per_side_pct': 0.05,
    'maker_per_side_pct': 0.02,
    'capital_usd': 1500,
    'train_test_split': 0.60,
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def detect_signals_for_config(df: pd.DataFrame, body_pct: float) -> list:
    """Detect with given body filter %."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    lookback = FIXED['period_lookback_bars']
    body_min = body_pct / 100.0
    signals = []
    for i in range(lookback + 1, n - 2):
        period_high = high[i - lookback:i].max()
        period_low = low[i - lookback:i].min()
        period_range = period_high - period_low
        if period_range <= 0:
            continue
        bar1_body = abs(close[i - 1] - open_[i - 1])
        bar0_body = abs(close[i] - open_[i])
        bar1_range = high[i - 1] - low[i - 1]
        bar0_range = high[i] - low[i]
        combined_range = bar1_range + bar0_range
        if combined_range <= 0:
            continue
        combined_body_pct = (bar1_body + bar0_body) / combined_range
        if combined_body_pct < body_min:
            continue
        if close[i] > period_high and close[i] > open_[i]:
            signals.append({'idx': i, 'fade_side': 'SHORT',
                            'period_range': period_range})
        elif close[i] < period_low and close[i] < open_[i]:
            signals.append({'idx': i, 'fade_side': 'LONG',
                            'period_range': period_range})
    return signals


def simulate_config(df: pd.DataFrame, signals: list, sl_mult: float, tp_mult: float,
                     start_idx: int = 0, end_idx: int = None) -> pd.DataFrame:
    """Simulate trades for given SL/TP multipliers within [start_idx, end_idx)."""
    n = len(df)
    if end_idx is None:
        end_idx = n
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    ts = df['timestamp'].values

    taker = FIXED['taker_per_side_pct']
    maker = FIXED['maker_per_side_pct']
    max_hold = FIXED['max_hold_bars']

    trades = []
    cooldown_until = -1

    for sig in signals:
        i = sig['idx']
        if i < start_idx or i >= end_idx:
            continue
        if i < cooldown_until:
            continue
        if i + 1 >= n:
            continue
        entry_idx = i + 1
        entry_price = open_[entry_idx]
        side = sig['fade_side']
        pr = sig['period_range']
        if side == 'SHORT':
            sl = entry_price + sl_mult * pr
            tp = entry_price - tp_mult * pr
        else:
            sl = entry_price - sl_mult * pr
            tp = entry_price + tp_mult * pr
        exit_price = None
        exit_friction = None
        for j in range(entry_idx, min(entry_idx + max_hold + 1, n)):
            if side == 'LONG':
                if low[j] <= sl:
                    exit_price = sl
                    exit_friction = taker
                    break
                if high[j] >= tp:
                    exit_price = tp
                    exit_friction = maker
                    break
            else:
                if high[j] >= sl:
                    exit_price = sl
                    exit_friction = taker
                    break
                if low[j] <= tp:
                    exit_price = tp
                    exit_friction = maker
                    break
        if exit_price is None:
            timeout_idx = min(entry_idx + max_hold, n - 1)
            exit_price = close[timeout_idx]
            exit_friction = taker
            j = timeout_idx
        if side == 'LONG':
            gross_pct = (exit_price - entry_price) / entry_price * 100
        else:
            gross_pct = (entry_price - exit_price) / entry_price * 100
        friction_pct = taker + exit_friction
        net_pct = gross_pct - friction_pct
        trades.append({
            'entry_ts': ts[entry_idx], 'exit_ts': ts[j],
            'side': side, 'gross_pct': gross_pct, 'net_pct': net_pct,
            'friction_pct': friction_pct,
        })
        cooldown_until = j + 1
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, n_days: float) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0, 'daily_pct': 0,
                'avg_gross_per_trade_pct': 0, 'wr': 0,
                'rr_realized': 0, 'bs_pos_rate': 0}
    cum_net = float((1 + trades['net_pct'] / 100).prod() - 1) * 100
    n = len(trades)
    avg_gross = float(trades['gross_pct'].mean())
    wr = float((trades['net_pct'] > 0).mean())
    aw = trades.loc[trades['net_pct'] > 0, 'net_pct'].mean()
    al = trades.loc[trades['net_pct'] < 0, 'net_pct'].mean()
    rr = abs(aw / al) if pd.notna(al) and al != 0 else float('inf')
    daily_pct = float(trades['net_pct'].sum() / n_days)
    # Bootstrap pos_rate
    trades['exit_date'] = pd.to_datetime(trades['exit_ts']).dt.floor('D')
    daily_returns = trades.groupby('exit_date')['net_pct'].sum()
    if len(daily_returns) > 3:
        daily_returns = daily_returns.reindex(
            pd.date_range(daily_returns.index.min(), daily_returns.index.max(), freq='D'),
            fill_value=0
        )
        nets = daily_returns.values
        if len(nets) > 3:
            random.seed(42)
            n_iter = min(1000, len(nets) - 3)
            starts = random.sample(range(len(nets) - 3), n_iter)
            cums = [nets[s:s + 3].sum() for s in starts]
            arr = np.array(cums)
            bs_pos_rate = float((arr > 0).mean())
        else:
            bs_pos_rate = 0
    else:
        bs_pos_rate = 0
    return {
        'n_trades': n, 'cum_net_pct': cum_net,
        'avg_gross_per_trade_pct': avg_gross,
        'wr': wr, 'rr_realized': rr,
        'daily_pct': daily_pct,
        'bs_pos_rate': bs_pos_rate,
    }


def main():
    print('=' * 100)
    print('Round 30 — Parameter Grid (Entry / SL / TP) with Train/Test Split')
    print('=' * 100)
    print('Pre-reg: claudedocs/round30_param_grid_prereg.md (7c9a4f5)')
    print(f'Grid: 3 entry × 3 SL × 3 TP = 27 configs')
    print(f'Train: 60%, Test: 40%\n')

    df = load_data()
    n = len(df)
    n_days_full = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400

    train_end_idx = int(n * FIXED['train_test_split'])
    train_end_ts = df['timestamp'].iloc[train_end_idx]
    train_days = (train_end_ts - df['timestamp'].iloc[0]).total_seconds() / 86400
    test_days = n_days_full - train_days
    print(f'Total: {n_days_full:.1f}d, Train: {train_days:.1f}d (idx 0-{train_end_idx}), '
          f'Test: {test_days:.1f}d (idx {train_end_idx}-{n})\n')

    # Pre-detect signals for each body filter (3 levels), use across SL/TP variants
    signals_by_body = {}
    for body in GRID['entry_body_filter_pct']:
        sigs = detect_signals_for_config(df, body)
        signals_by_body[body] = sigs
        print(f'Body {body}%: {len(sigs)} signals')
    print()

    # Run all 27 configs on TRAIN
    train_results = []
    for body, sl, tp in product(GRID['entry_body_filter_pct'],
                                  GRID['sl_period_range_multiple'],
                                  GRID['tp_period_range_multiple']):
        sigs = signals_by_body[body]
        trades = simulate_config(df, sigs, sl, tp,
                                  start_idx=0, end_idx=train_end_idx)
        summ = summarize(trades, train_days)
        train_results.append({
            'config': {'body': body, 'sl_mult': sl, 'tp_mult': tp},
            **summ
        })

    # Sort by train daily_pct descending
    train_results.sort(key=lambda r: r['daily_pct'], reverse=True)

    print('=== TRAIN results (sorted by daily_pct descending) ===')
    print(f'{"body":>5} {"sl":>5} {"tp":>5} {"n_tr":>6} {"daily%":>10} '
          f'{"gross%":>10} {"WR":>6} {"R:R":>6} {"BS_pos":>8}')
    print('-' * 80)
    for r in train_results:
        c = r['config']
        print(f'{c["body"]:>5} {c["sl_mult"]:>5} {c["tp_mult"]:>5} '
              f'{r["n_trades"]:>6} {r["daily_pct"]:>+9.4f}% '
              f'{r["avg_gross_per_trade_pct"]:>+9.4f}% '
              f'{r["wr"]:>6.3f} {r["rr_realized"]:>6.3f} {r["bs_pos_rate"]:>8.4f}')
    print()

    # WINNER selection by train daily
    winner = train_results[0]
    print(f'=== TRAIN WINNER ===')
    print(f'  config: body={winner["config"]["body"]}%, '
          f'SL={winner["config"]["sl_mult"]}× pr, '
          f'TP={winner["config"]["tp_mult"]}× pr')
    print(f'  train daily: {winner["daily_pct"]:+.4f}%, '
          f'gross/trade: {winner["avg_gross_per_trade_pct"]:+.4f}%, '
          f'WR: {winner["wr"]:.4f}, '
          f'BS_pos: {winner["bs_pos_rate"]:.4f}\n')

    # TEST validation on winner
    winner_body = winner['config']['body']
    winner_sl = winner['config']['sl_mult']
    winner_tp = winner['config']['tp_mult']
    winner_signals = signals_by_body[winner_body]
    test_trades = simulate_config(df, winner_signals, winner_sl, winner_tp,
                                    start_idx=train_end_idx, end_idx=n)
    test_summ = summarize(test_trades, test_days)

    print(f'=== TEST validation (winner only) ===')
    print(f'  test daily: {test_summ["daily_pct"]:+.4f}%')
    print(f'  test gross/trade: {test_summ["avg_gross_per_trade_pct"]:+.4f}%')
    print(f'  test WR: {test_summ["wr"]:.4f}')
    print(f'  test BS_pos: {test_summ["bs_pos_rate"]:.4f}')
    print(f'  test n_trades: {test_summ["n_trades"]}\n')

    # Verdict
    train_daily = winner['daily_pct']
    test_daily = test_summ['daily_pct']
    test_target_pct = 0.20

    print('=== Train vs Test Overfit Check ===')
    print(f'  train daily: {train_daily:+.4f}%')
    print(f'  test daily:  {test_daily:+.4f}%')
    if train_daily > 0:
        retention = test_daily / train_daily
        print(f'  test/train ratio: {retention:.4f}')
    else:
        retention = 0

    if test_daily >= test_target_pct and (train_daily == 0 or test_daily / train_daily > 0.5):
        verdict = 'GENUINE_SIGNAL'
    elif test_daily >= 0.10:
        verdict = 'PARTIAL_POSITIVE_BELOW_TARGET'
    elif test_daily >= 0:
        verdict = 'BORDERLINE_NEAR_ZERO'
    else:
        verdict = 'CATASTROPHIC_OVERFIT'

    print(f'  → VERDICT: {verdict}')
    if verdict == 'GENUINE_SIGNAL':
        print(f'  Test daily ≥ 0.20% AND retention > 50% → CANDIDATE FOUND')
    else:
        print(f'  Test daily {test_daily:+.4f}% does NOT clear 0.20% target')
    print()

    print('=' * 100)
    print(f'27-config grid: TRAIN winner gives test daily {test_daily:+.4f}%')
    print(f'User target: +0.20%/day')
    print(f'Result: {verdict}')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '7c9a4f5',
        'grid': GRID, 'fixed': FIXED,
        'train_days': train_days, 'test_days': test_days,
        'train_results_sorted': train_results,
        'winner_config': winner['config'],
        'winner_train': winner,
        'winner_test': test_summ,
        'train_daily_pct': train_daily,
        'test_daily_pct': test_daily,
        'verdict': verdict,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round30_param_grid_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
