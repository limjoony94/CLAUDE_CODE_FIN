"""Round 28 — C1-Fade-Fixed: Reverse Direction + Range-Based Fixed Targets.

Pre-reg: claudedocs/round28_c1_fade_fixed_prereg.md (commit 95663d4)

Mechanism:
  C1 detection (15-bar Donchian + 40% body) → REVERSE direction.
  Bullish breakout → SHORT (fade upward move).
  Bearish breakdown → LONG (fade downward move).
  Entry: market at next bar open (taker 0.05%).
  TP: 2 × candle_range[t] from entry (limit, maker 0.02%).
  SL: 1 × candle_range[t] from entry (market, taker 0.05%).
  Max hold: 192 bars (8 days at 1h).
  R:R = 2.0 LOCKED.
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

DATA_FILE = DATA / 'btc_1h_720days.csv'

LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'channel_lookback_bars': 15,
    'body_min_pct_of_range': 0.40,
    'tp_candle_range_multiple': 2.0,
    'sl_candle_range_multiple': 1.0,
    'max_hold_bars': 192,
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


def detect_c1_signals(df: pd.DataFrame) -> list:
    """C1 detection: close > 15-bar high AND body > 40% range AND bullish bar.
    Or close < 15-bar low AND body > 40% AND bearish bar."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    lookback = LOCKED['channel_lookback_bars']
    body_min = LOCKED['body_min_pct_of_range']
    signals = []
    for i in range(lookback, n - 2):
        prior_high = high[i - lookback:i].max()
        prior_low = low[i - lookback:i].min()
        bar_range = high[i] - low[i]
        if bar_range <= 0:
            continue
        body = abs(close[i] - open_[i])
        body_pct = body / bar_range
        if close[i] > prior_high and body_pct > body_min and close[i] > open_[i]:
            # Bullish breakout → fade SHORT
            signals.append({
                'idx': i, 'fade_side': 'SHORT',
                'breakout_close': close[i], 'candle_range': bar_range,
            })
        elif close[i] < prior_low and body_pct > body_min and close[i] < open_[i]:
            # Bearish breakdown → fade LONG
            signals.append({
                'idx': i, 'fade_side': 'LONG',
                'breakout_close': close[i], 'candle_range': bar_range,
            })
    return signals


def simulate_fade(df: pd.DataFrame, signals: list) -> pd.DataFrame:
    """Simulate fade entries with fixed range-based targets."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    ts = df['timestamp'].values

    taker = LOCKED['taker_per_side_pct']
    maker = LOCKED['maker_per_side_pct']
    tp_mult = LOCKED['tp_candle_range_multiple']
    sl_mult = LOCKED['sl_candle_range_multiple']
    max_hold = LOCKED['max_hold_bars']

    trades = []
    in_pos = False
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
        cr = sig['candle_range']

        if side == 'SHORT':
            sl = entry_price + sl_mult * cr
            tp = entry_price - tp_mult * cr
        else:
            sl = entry_price - sl_mult * cr
            tp = entry_price + tp_mult * cr

        # Walk forward looking for SL/TP/timeout
        exit_price = None
        exit_reason = None
        exit_friction = None
        for j in range(entry_idx, min(entry_idx + max_hold + 1, n)):
            bar_high = high[j]
            bar_low = low[j]
            if side == 'LONG':
                if bar_low <= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    exit_friction = taker
                    break
                if bar_high >= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    exit_friction = maker
                    break
            else:  # SHORT
                if bar_high >= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    exit_friction = taker
                    break
                if bar_low <= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    exit_friction = maker
                    break

        if exit_price is None:
            # Timeout
            timeout_idx = min(entry_idx + max_hold, n - 1)
            exit_price = close[timeout_idx]
            exit_reason = 'TIMEOUT'
            exit_friction = taker
            j = timeout_idx

        if side == 'LONG':
            gross_pct = (exit_price - entry_price) / entry_price * 100
        else:
            gross_pct = (entry_price - exit_price) / entry_price * 100

        # Friction: entry taker + exit per reason
        friction_pct = taker + exit_friction
        net_pct = gross_pct - friction_pct

        trades.append({
            'entry_ts': ts[entry_idx], 'exit_ts': ts[j],
            'side': side, 'entry_price': entry_price, 'exit_price': exit_price,
            'sl': sl, 'tp': tp, 'candle_range': cr,
            'gross_pct': gross_pct, 'friction_pct': friction_pct, 'net_pct': net_pct,
            'exit_reason': exit_reason, 'hold_bars': j - entry_idx,
        })
        cooldown_until = j + 1  # no overlap

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
        'n_trades': n,
        'cum_net_pct': cum_net,
        'avg_gross_per_trade_pct': avg_gross,
        'avg_friction_per_trade_pct': avg_friction,
        'avg_net_per_trade_pct': avg_net,
        'wr': wr,
        'rr_realized': rr,
        'daily_pct': daily_pct,
        'trades_per_day': trades_per_day,
        'worst_5d_pct': worst_5d,
        'exit_breakdown': exit_breakdown,
    }


def main():
    print('=' * 100)
    print('Round 28 — C1-Fade-Fixed: Reverse Direction + Range-Based Fixed Targets')
    print('=' * 100)
    print('Pre-reg: claudedocs/round28_c1_fade_fixed_prereg.md (95663d4)')
    print(f'Locked: TP {LOCKED["tp_candle_range_multiple"]}× cr, '
          f'SL {LOCKED["sl_candle_range_multiple"]}× cr, '
          f'R:R {LOCKED["tp_candle_range_multiple"]/LOCKED["sl_candle_range_multiple"]}\n')

    df = load_data()
    n_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars, {n_days:.1f} days\n')

    print('Detecting C1 breakout signals...')
    signals = detect_c1_signals(df)
    print(f'  signals: {len(signals)} '
          f'({sum(1 for s in signals if s["fade_side"]=="SHORT")} bullish→SHORT, '
          f'{sum(1 for s in signals if s["fade_side"]=="LONG")} bearish→LONG)\n')

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
        'pre_reg_commit': '95663d4',
        'locked': LOCKED, 'gates': GATES,
        'summary': summ,
        'c1_daily': {'daily_pct': daily, 'pass': bool(c1_pass)},
        'c2_per_trade_gross': {'gross_pct': pt_gross, 'pass': bool(c2_pass)},
        'c3_trade_count': {'n_trades': n_trades, 'pass': bool(c3_pass)},
        'c4_bootstrap': {'pos_rate': pos_rate, 'pass': bool(c4_pass)},
        'verdict_pass': n_pass,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round28_c1_fade_fixed_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
