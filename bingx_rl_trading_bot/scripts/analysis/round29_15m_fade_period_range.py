"""Round 29 — 15m Fade + Period-Range TP/SL + 2-Bar Body Filter.

Pre-reg: claudedocs/round29_15m_fade_period_range_prereg.md (commit 55c6e0e)

Mechanism (5 LOCKED user-specified changes):
  TF=15m, lookback=16 bars (4h), 2-bar body sum > 50%,
  fade direction, TP=2×period_range, SL=1×period_range, max hold 96 bars (24h).
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

DATA_FILE = DATA / 'btc_15m_720days.csv'

LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '15m',
    'period_lookback_bars': 16,
    'body_combined_min_pct_of_range': 0.50,
    'tp_period_range_multiple': 2.0,
    'sl_period_range_multiple': 1.0,
    'max_hold_bars': 96,
    'taker_per_side_pct': 0.05,
    'maker_per_side_pct': 0.02,
    'capital_usd': 1500,
}

GATES = {
    'gate_A_min_setups': 100,
    'c1_daily_pct_min': 0.20,
    'c2_per_trade_gross_min': 0.07,
    'c3_min_trades': 100,
    'c4_bs_window_days': 3,
    'c4_bs_n_iter': 1000,
    'c4_min_pos_rate': 0.50,
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def detect_signals(df: pd.DataFrame) -> list:
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    lookback = LOCKED['period_lookback_bars']
    body_min = LOCKED['body_combined_min_pct_of_range']
    signals = []
    for i in range(lookback + 1, n - 2):
        period_high = high[i - lookback:i].max()
        period_low = low[i - lookback:i].min()
        period_range = period_high - period_low
        if period_range <= 0:
            continue

        # 2-bar combined body
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

        # Detection
        if close[i] > period_high and close[i] > open_[i]:
            signals.append({
                'idx': i, 'fade_side': 'SHORT',
                'period_range': period_range,
                'period_high': period_high, 'period_low': period_low,
            })
        elif close[i] < period_low and close[i] < open_[i]:
            signals.append({
                'idx': i, 'fade_side': 'LONG',
                'period_range': period_range,
                'period_high': period_high, 'period_low': period_low,
            })
    return signals


def simulate_fade(df: pd.DataFrame, signals: list) -> pd.DataFrame:
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    ts = df['timestamp'].values

    taker = LOCKED['taker_per_side_pct']
    maker = LOCKED['maker_per_side_pct']
    tp_mult = LOCKED['tp_period_range_multiple']
    sl_mult = LOCKED['sl_period_range_multiple']
    max_hold = LOCKED['max_hold_bars']

    trades = []
    cooldown_until = -1

    for sig in signals:
        i = sig['idx']
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
        exit_reason = None
        exit_friction = None
        for j in range(entry_idx, min(entry_idx + max_hold + 1, n)):
            if side == 'LONG':
                if low[j] <= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    exit_friction = taker
                    break
                if high[j] >= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    exit_friction = maker
                    break
            else:
                if high[j] >= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    exit_friction = taker
                    break
                if low[j] <= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    exit_friction = maker
                    break

        if exit_price is None:
            timeout_idx = min(entry_idx + max_hold, n - 1)
            exit_price = close[timeout_idx]
            exit_reason = 'TIMEOUT'
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
            'side': side, 'entry_price': entry_price, 'exit_price': exit_price,
            'sl': sl, 'tp': tp, 'period_range': pr,
            'gross_pct': gross_pct, 'friction_pct': friction_pct, 'net_pct': net_pct,
            'exit_reason': exit_reason, 'hold_bars': j - entry_idx,
        })
        cooldown_until = j + 1

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, n_days: float) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0}
    cum_net = float((1 + trades['net_pct'] / 100).prod() - 1) * 100
    n = len(trades)
    avg_gross = float(trades['gross_pct'].mean())
    avg_friction = float(trades['friction_pct'].mean())
    avg_net = float(trades['net_pct'].mean())
    wr = float((trades['net_pct'] > 0).mean())
    avg_win = float(trades.loc[trades['net_pct'] > 0, 'net_pct'].mean()) if (trades['net_pct'] > 0).any() else 0
    avg_loss = float(trades.loc[trades['net_pct'] < 0, 'net_pct'].mean()) if (trades['net_pct'] < 0).any() else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    daily_pct = float(trades['net_pct'].sum() / n_days)
    trades_per_day = n / n_days

    trades_copy = trades.copy()
    trades_copy['exit_date'] = pd.to_datetime(trades_copy['exit_ts']).dt.floor('D')
    daily_returns = trades_copy.groupby('exit_date')['net_pct'].sum()
    if len(daily_returns) >= 5:
        worst_5d = float(daily_returns.rolling(5).sum().min())
    else:
        worst_5d = float(daily_returns.min())

    exit_breakdown = trades['exit_reason'].value_counts().to_dict()

    return {
        'n_trades': n, 'cum_net_pct': cum_net,
        'avg_gross_per_trade_pct': avg_gross,
        'avg_friction_per_trade_pct': avg_friction,
        'avg_net_per_trade_pct': avg_net,
        'wr': wr, 'rr_realized': rr,
        'daily_pct': daily_pct, 'trades_per_day': trades_per_day,
        'worst_5d_pct': worst_5d,
        'exit_breakdown': exit_breakdown,
    }


def main():
    print('=' * 100)
    print('Round 29 — 15m Fade + Period-Range TP/SL + 2-Bar Body Filter')
    print('=' * 100)
    print('Pre-reg: claudedocs/round29_15m_fade_period_range_prereg.md (55c6e0e)')
    print(f'Locked: TF=15m, lookback={LOCKED["period_lookback_bars"]} bars (4h), '
          f'body 2-bar sum > {LOCKED["body_combined_min_pct_of_range"]*100}%, '
          f'TP {LOCKED["tp_period_range_multiple"]}× pr, SL {LOCKED["sl_period_range_multiple"]}× pr\n')

    df = load_data()
    n_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars (15m), {n_days:.1f} days\n')

    print('Detecting signals...')
    signals = detect_signals(df)
    n_short = sum(1 for s in signals if s['fade_side'] == 'SHORT')
    n_long = sum(1 for s in signals if s['fade_side'] == 'LONG')
    print(f'  signals: {len(signals)} (bullish→SHORT {n_short}, bearish→LONG {n_long})\n')

    print('Simulating fade entries...')
    trades = simulate_fade(df, signals)
    print(f'  trades: {len(trades)}\n')

    summ = summarize(trades, n_days)
    print('=== Full-sample summary ===')
    for k, v in summ.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    daily = summ.get('daily_pct', 0)
    pt_gross = summ.get('avg_gross_per_trade_pct', 0)
    n_trades = summ.get('n_trades', 0)

    c1_pass = daily >= GATES['c1_daily_pct_min']
    c2_pass = pt_gross >= GATES['c2_per_trade_gross_min']
    c3_pass = n_trades >= GATES['c3_min_trades']

    print(f'=== C1 (HARD) Daily ≥ 0.20% ===')
    print(f'  daily: {daily:+.4f}%  → {"PASS" if c1_pass else "FAIL"}\n')
    print(f'=== C2 Per-trade gross > 0.07% ===')
    print(f'  per-trade gross: {pt_gross:+.4f}%  → {"PASS" if c2_pass else "FAIL"}\n')
    print(f'=== C3 Trade count ≥ 100 ===')
    print(f'  n_trades: {n_trades}  → {"PASS" if c3_pass else "FAIL"}\n')

    if n_trades >= 5:
        trades_copy = trades.copy()
        trades_copy['exit_date'] = pd.to_datetime(trades_copy['exit_ts']).dt.floor('D')
        daily_returns = trades_copy.groupby('exit_date')['net_pct'].sum()
        daily_returns = daily_returns.reindex(
            pd.date_range(daily_returns.index.min(), daily_returns.index.max(), freq='D'),
            fill_value=0
        )
        nets = daily_returns.values
        win = GATES['c4_bs_window_days']
        n = len(nets)
        if n > win:
            random.seed(42)
            starts = random.sample(range(n - win), min(GATES['c4_bs_n_iter'], n - win))
            cums = [nets[s:s + win].sum() for s in starts]
            arr = np.array(cums)
            pos_rate = float((arr > 0).mean())
            c4_pass = pos_rate >= GATES['c4_min_pos_rate']
        else:
            pos_rate = 0
            c4_pass = False
    else:
        pos_rate = 0
        c4_pass = False
    print(f'=== C4 Bootstrap pos_rate ≥ 50% ===')
    print(f'  pos_rate: {pos_rate:.4f}  → {"PASS" if c4_pass else "FAIL"}\n')

    n_pass = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    print('=' * 100)
    print(f'VERDICT: {n_pass}/4 user criteria PASS')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '55c6e0e',
        'locked': LOCKED, 'gates': GATES,
        'summary': summ,
        'c1_daily': {'daily_pct': daily, 'pass': bool(c1_pass)},
        'c2_per_trade_gross': {'gross_pct': pt_gross, 'pass': bool(c2_pass)},
        'c3_trade_count': {'n_trades': n_trades, 'pass': bool(c3_pass)},
        'c4_bootstrap': {'pos_rate': pos_rate, 'pass': bool(c4_pass)},
        'verdict_pass': n_pass,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round29_15m_fade_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
