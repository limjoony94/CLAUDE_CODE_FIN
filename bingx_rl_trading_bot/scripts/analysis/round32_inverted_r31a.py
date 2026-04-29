"""Round 32 — Inverted R31a (trend direction + swapped TP/SL).

User insight: R31a WR 7.34% — flip everything.
  Direction: LONG on bullish breakout (was fade SHORT)
  TP = period_high + 0.5×ATR (was R31a SL, close to entry)
  SL = period_low (was R31a TP, far from entry)
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

DATA_FILE = DATA / 'btc_15m_720days.csv'

LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '15m',
    'period_lookback_bars': 16,
    'body_combined_min_pct_of_range': 0.50,
    'direction': 'TREND',  # INVERTED from fade
    'tp_atr_buffer_mult': 0.5,
    'atr_period': 14,
    'max_hold_bars': 96,
    'taker_per_side_pct': 0.05,
    'maker_per_side_pct': 0.02,
    'capital_usd': 1500,
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_signals(df: pd.DataFrame) -> list:
    """Same detection as R31a — bullish/bearish breakouts."""
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
        bar1_body = abs(close[i - 1] - open_[i - 1])
        bar0_body = abs(close[i] - open_[i])
        bar1_range = high[i - 1] - low[i - 1]
        bar0_range = high[i] - low[i]
        combined_range = bar1_range + bar0_range
        if combined_range <= 0:
            continue
        if (bar1_body + bar0_body) / combined_range < body_min:
            continue
        if close[i] > period_high and close[i] > open_[i]:
            # Bullish breakout — trend direction = LONG (INVERTED from fade)
            signals.append({'idx': i, 'side': 'LONG',
                            'period_high': period_high, 'period_low': period_low,
                            'period_range': period_range})
        elif close[i] < period_low and close[i] < open_[i]:
            # Bearish breakdown — trend direction = SHORT (INVERTED from fade)
            signals.append({'idx': i, 'side': 'SHORT',
                            'period_high': period_high, 'period_low': period_low,
                            'period_range': period_range})
    return signals


def simulate(df: pd.DataFrame, signals: list) -> pd.DataFrame:
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    ts = df['timestamp'].values
    atr = compute_atr(df, LOCKED['atr_period']).values

    taker = LOCKED['taker_per_side_pct']
    maker = LOCKED['maker_per_side_pct']
    max_hold = LOCKED['max_hold_bars']
    buf_mult = LOCKED['tp_atr_buffer_mult']

    trades = []
    cooldown_until = -1

    for sig in signals:
        i = sig['idx']
        if i < cooldown_until or i + 1 >= n:
            continue
        entry_idx = i + 1
        entry_price = open_[entry_idx]
        side = sig['side']
        atr_now = atr[i] if not np.isnan(atr[i]) else 0
        if atr_now <= 0:
            continue
        buffer = buf_mult * atr_now

        # INVERTED: TP at small buffer beyond breakout side, SL at far period extreme
        if side == 'LONG':
            tp = sig['period_high'] + buffer  # small distance above entry
            sl = sig['period_low']             # far below entry
            if tp <= entry_price or sl >= entry_price:
                continue
        else:  # SHORT
            tp = sig['period_low'] - buffer    # small distance below entry
            sl = sig['period_high']            # far above entry
            if tp >= entry_price or sl <= entry_price:
                continue

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
            'tp': tp, 'sl': sl, 'entry_price': entry_price,
        })
        cooldown_until = j + 1
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, n_days: float) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0, 'daily_pct': 0}
    cum_net = float((1 + trades['net_pct'] / 100).prod() - 1) * 100
    n = len(trades)
    avg_gross = float(trades['gross_pct'].mean())
    avg_friction = float(trades['friction_pct'].mean())
    avg_net = float(trades['net_pct'].mean())
    wr = float((trades['net_pct'] > 0).mean())
    aw = trades.loc[trades['net_pct'] > 0, 'net_pct'].mean()
    al = trades.loc[trades['net_pct'] < 0, 'net_pct'].mean()
    rr = abs(aw / al) if pd.notna(al) and al != 0 else float('inf')
    daily_pct = float(trades['net_pct'].sum() / n_days)
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
            bs_pos = float((arr > 0).mean())
        else:
            bs_pos = 0
    else:
        bs_pos = 0

    # Worst 5d
    if len(daily_returns) >= 5:
        worst_5d = float(daily_returns.rolling(5).sum().min())
    else:
        worst_5d = float(daily_returns.min())

    return {
        'n_trades': n, 'cum_net_pct': cum_net,
        'avg_gross_per_trade_pct': avg_gross,
        'avg_friction_per_trade_pct': avg_friction,
        'avg_net_per_trade_pct': avg_net,
        'wr': wr, 'rr_realized': rr,
        'daily_pct': daily_pct,
        'worst_5d_pct': worst_5d,
        'bs_pos_rate': bs_pos,
    }


def main():
    print('=' * 100)
    print('Round 32 — Inverted R31a (TREND instead of FADE)')
    print('=' * 100)
    print('User insight: R31a fade had WR 7.34% — invert direction + swap TP/SL.')
    print(f'Direction: bullish breakout → LONG, bearish breakdown → SHORT')
    print(f'TP = period extreme on signal-side + 0.5×ATR(14) (small)')
    print(f'SL = period extreme on opposite side (far)\n')

    df = load_data()
    n_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars (15m), {n_days:.1f} days\n')

    signals = detect_signals(df)
    n_long = sum(1 for s in signals if s['side'] == 'LONG')
    n_short = sum(1 for s in signals if s['side'] == 'SHORT')
    print(f'Signals: {len(signals)} (bullish→LONG {n_long}, bearish→SHORT {n_short})\n')

    trades = simulate(df, signals)
    print(f'Trades: {len(trades)}\n')
    summ = summarize(trades, n_days)
    print('=== Summary ===')
    for k, v in summ.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    daily = summ['daily_pct']
    gross = summ['avg_gross_per_trade_pct']
    bs = summ['bs_pos_rate']
    nt = summ['n_trades']
    c1 = daily >= 0.20
    c2 = gross >= 0.07
    c3 = nt >= 100
    c4 = bs >= 0.50
    n_pass = sum([c1, c2, c3, c4])
    print(f'  C1 daily≥0.20%: {"PASS" if c1 else "FAIL"} ({daily:+.4f}%)')
    print(f'  C2 gross>0.07%: {"PASS" if c2 else "FAIL"} ({gross:+.4f}%)')
    print(f'  C3 trades≥100:  {"PASS" if c3 else "FAIL"} ({nt})')
    print(f'  C4 BS_pos≥50%:  {"PASS" if c4 else "FAIL"} ({bs:.4f})')
    print(f'  → {n_pass}/4\n')

    # Comparison with R31a
    print('=== Comparison vs R31a (fade direction) ===')
    print(f'  R31a daily: -0.30%, R32 daily: {daily:+.4f}%')
    print(f'  R31a gross: +0.006%, R32 gross: {gross:+.4f}%')
    print(f'  R31a WR: 7.34%, R32 WR: {summ["wr"]*100:.2f}%')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'inline_r32',
        'locked': LOCKED,
        'summary': summ,
        'c1_pass': bool(c1), 'c2_pass': bool(c2),
        'c3_pass': bool(c3), 'c4_pass': bool(c4),
        'n_pass': n_pass,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round32_inverted_r31a_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
