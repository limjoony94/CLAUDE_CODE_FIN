"""Round 27 — Opening Range Breakout (Session-Local Formation).

Pre-reg: claudedocs/round27_orb_session_local_prereg.md (commit 4c6305e)

Mechanism:
  UTC day 00:00-04:00 = opening range (OR) formation (4 hours).
  04:00-22:00 = trade window. First breakout of OR_high → LONG (stop entry, taker).
  First breakdown of OR_low → SHORT.
  SL: opposite side − 0.1×ATR(14). TP: 1.5×OR_range. Max 18h hold.
  Max 1 trade per UTC day. LOCKED.
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
    'or_start_utc': 0,
    'or_end_utc': 4,
    'trade_window_start_utc': 4,
    'trade_window_end_utc': 22,
    'sl_atr_buffer_mult': 0.1,
    'tp_or_multiple': 1.5,
    'max_hold_bars': 18,
    'max_trades_per_day': 1,
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
    df['hour_utc'] = df['timestamp'].dt.hour
    df['date_utc'] = df['timestamp'].dt.floor('D')
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def simulate_orb(df: pd.DataFrame) -> pd.DataFrame:
    """Walk per-UTC-day, build OR, then trade first breakout in window."""
    df = df.copy()
    df['atr14'] = compute_atr(df, 14)

    trades = []
    grouped = df.groupby('date_utc')
    days = sorted(df['date_utc'].unique())

    taker_fric = LOCKED['taker_per_side_pct']
    maker_fric = LOCKED['maker_per_side_pct']
    or_start = LOCKED['or_start_utc']
    or_end = LOCKED['or_end_utc']
    tw_start = LOCKED['trade_window_start_utc']
    tw_end = LOCKED['trade_window_end_utc']
    sl_buf = LOCKED['sl_atr_buffer_mult']
    tp_mult = LOCKED['tp_or_multiple']
    max_hold = LOCKED['max_hold_bars']

    for d in days:
        day_df = df[df['date_utc'] == d].sort_values('timestamp').reset_index(drop=True)
        if len(day_df) < 18:  # need OR + at least some trade window bars
            continue

        # OR window
        or_df = day_df[(day_df['hour_utc'] >= or_start) &
                       (day_df['hour_utc'] < or_end)]
        if len(or_df) < 4:
            continue
        or_high = float(or_df['high'].max())
        or_low = float(or_df['low'].min())
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        # ATR at OR end
        or_end_idx = or_df.index[-1]
        atr_at_or_end = or_df['atr14'].iloc[-1]
        if pd.isna(atr_at_or_end) or atr_at_or_end <= 0:
            continue

        # Trade window
        tw_df = day_df[(day_df['hour_utc'] >= tw_start) &
                       (day_df['hour_utc'] < tw_end)].reset_index(drop=True)
        if len(tw_df) == 0:
            continue

        # Walk forward; first bar that breaks OR triggers entry
        triggered = False
        side = None
        entry_idx = None
        entry_price = None
        for k in range(len(tw_df)):
            bar = tw_df.iloc[k]
            high_breaks = bar['high'] > or_high
            low_breaks = bar['low'] < or_low
            if high_breaks and low_breaks:
                # ambiguous — skip day
                break
            if high_breaks:
                side = 'LONG'
                entry_price = or_high  # stop fills at OR_high
                entry_idx = k
                triggered = True
                break
            if low_breaks:
                side = 'SHORT'
                entry_price = or_low
                entry_idx = k
                triggered = True
                break

        if not triggered:
            continue

        # SL
        if side == 'LONG':
            sl = or_low - sl_buf * atr_at_or_end
            tp = entry_price + tp_mult * or_range
        else:
            sl = or_high + sl_buf * atr_at_or_end
            tp = entry_price - tp_mult * or_range

        # Walk from entry_idx forward, max_hold bars, looking for SL/TP/timeout
        exit_price = None
        exit_reason = None
        exit_friction_pct = None
        for j in range(entry_idx, min(entry_idx + max_hold + 1, len(tw_df))):
            bar = tw_df.iloc[j]
            if side == 'LONG':
                if bar['low'] <= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    exit_friction_pct = taker_fric  # market exit
                    break
                if bar['high'] >= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    exit_friction_pct = maker_fric  # limit fill
                    break
            else:
                if bar['high'] >= sl:
                    exit_price = sl
                    exit_reason = 'SL'
                    exit_friction_pct = taker_fric
                    break
                if bar['low'] <= tp:
                    exit_price = tp
                    exit_reason = 'TP'
                    exit_friction_pct = maker_fric
                    break

        if exit_price is None:
            # Timeout exit at last bar of window
            last_bar = tw_df.iloc[min(entry_idx + max_hold, len(tw_df) - 1)]
            exit_price = last_bar['close']
            exit_reason = 'TIMEOUT'
            exit_friction_pct = taker_fric

        # P&L
        if side == 'LONG':
            gross_pct = (exit_price - entry_price) / entry_price * 100
        else:
            gross_pct = (entry_price - exit_price) / entry_price * 100

        # Friction: entry taker (stop) + exit per reason
        friction_pct = taker_fric + exit_friction_pct
        net_pct = gross_pct - friction_pct

        trades.append({
            'date_utc': d,
            'side': side,
            'or_high': or_high, 'or_low': or_low, 'or_range': or_range,
            'entry_price': entry_price, 'exit_price': exit_price,
            'sl': sl, 'tp': tp,
            'gross_pct': gross_pct, 'friction_pct': friction_pct, 'net_pct': net_pct,
            'exit_reason': exit_reason,
        })

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

    daily_returns = trades.groupby('date_utc')['net_pct'].sum()
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


def gate_B_random_baseline(df: pd.DataFrame, actual_trades: pd.DataFrame, n_iter: int = 1000) -> dict:
    """Random-OR-window: pick random 4h window per day, simulate same logic."""
    if actual_trades.empty:
        return {'pass': False, 'reason': 'no actual trades'}
    rng = np.random.default_rng(42)
    actual_cum = float((1 + actual_trades['net_pct'] / 100).prod() - 1) * 100

    cum_arr = []
    for _ in range(n_iter):
        # Random OR start hour (0-19, ensuring trade window of at least 4 hours fits)
        random_start = rng.integers(0, 20)
        # Substitute LOCKED OR window with random one
        cum_arr.append(simulate_random_orb(df, random_start))
    arr = np.array(cum_arr)
    p95 = float(np.percentile(arr, 95))
    return {
        'actual_cum_pct': actual_cum,
        'random_p95_pct': p95,
        'random_mean_pct': float(arr.mean()),
        'pass': actual_cum > p95,
    }


def simulate_random_orb(df: pd.DataFrame, start_hour: int) -> float:
    """Simplified random-OR replay returning cum_net_pct."""
    end_hour = (start_hour + 4) % 24
    or_start = start_hour
    or_end = end_hour if end_hour > start_hour else 24

    days = sorted(df['date_utc'].unique())
    cum_net_factor = 1.0
    for d in days:
        day_df = df[df['date_utc'] == d].sort_values('timestamp').reset_index(drop=True)
        if len(day_df) < 18:
            continue

        if end_hour > start_hour:
            or_df = day_df[(day_df['hour_utc'] >= or_start) & (day_df['hour_utc'] < or_end)]
            tw_df = day_df[(day_df['hour_utc'] >= or_end) | (day_df['hour_utc'] < or_start)].sort_values('timestamp').reset_index(drop=True)
        else:
            continue  # skip wraparound for simplicity

        if len(or_df) < 3 or len(tw_df) == 0:
            continue
        or_high = float(or_df['high'].max())
        or_low = float(or_df['low'].min())
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        atr = or_df['atr14'].iloc[-1] if 'atr14' in or_df.columns else None
        if atr is None or pd.isna(atr) or atr <= 0:
            continue

        triggered = False
        side = None
        entry_price = None
        for k in range(len(tw_df)):
            bar = tw_df.iloc[k]
            if bar['high'] > or_high and bar['low'] >= or_low:
                side, entry_price, triggered = 'LONG', or_high, True
                entry_idx = k
                break
            if bar['low'] < or_low and bar['high'] <= or_high:
                side, entry_price, triggered = 'SHORT', or_low, True
                entry_idx = k
                break
        if not triggered:
            continue

        if side == 'LONG':
            sl = or_low - 0.1 * atr
            tp = entry_price + 1.5 * or_range
        else:
            sl = or_high + 0.1 * atr
            tp = entry_price - 1.5 * or_range

        exit_price = None
        for j in range(entry_idx, min(entry_idx + 18, len(tw_df))):
            bar = tw_df.iloc[j]
            if side == 'LONG':
                if bar['low'] <= sl:
                    exit_price, exit_friction = sl, 0.05
                    break
                if bar['high'] >= tp:
                    exit_price, exit_friction = tp, 0.02
                    break
            else:
                if bar['high'] >= sl:
                    exit_price, exit_friction = sl, 0.05
                    break
                if bar['low'] <= tp:
                    exit_price, exit_friction = tp, 0.02
                    break
        if exit_price is None:
            exit_price = tw_df.iloc[min(entry_idx + 17, len(tw_df) - 1)]['close']
            exit_friction = 0.05

        if side == 'LONG':
            gross = (exit_price - entry_price) / entry_price * 100
        else:
            gross = (entry_price - exit_price) / entry_price * 100
        net = gross - 0.05 - exit_friction
        cum_net_factor *= (1 + net / 100)

    return (cum_net_factor - 1) * 100


def main():
    print('=' * 100)
    print('Round 27 — Opening Range Breakout (Session-Local Formation)')
    print('=' * 100)
    print('Pre-reg: claudedocs/round27_orb_session_local_prereg.md (4c6305e)')
    print(f'Locked: OR {LOCKED["or_start_utc"]:02d}:00-{LOCKED["or_end_utc"]:02d}:00 UTC, '
          f'TP {LOCKED["tp_or_multiple"]}× OR_range, '
          f'SL OR_opposite − 0.1×ATR(14)\n')

    df = load_data()
    n_days_full = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars, {n_days_full:.1f} days\n')

    print('Simulating ORB...')
    trades = simulate_orb(df)
    print(f'  trades: {len(trades)}\n')

    summ = summarize(trades, n_days_full)
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

    # C4 bootstrap
    if n_trades >= 5:
        daily_returns = trades.groupby('date_utc')['net_pct'].sum()
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
        'pre_reg_commit': '4c6305e',
        'locked': LOCKED, 'gates': GATES,
        'summary': summ,
        'c1_daily': {'daily_pct': daily, 'pass': bool(c1_pass)},
        'c2_per_trade_gross': {'gross_pct': pt_gross, 'pass': bool(c2_pass)},
        'c3_trade_count': {'n_trades': n_trades, 'pass': bool(c3_pass)},
        'c4_bootstrap': {'pos_rate': pos_rate, 'pass': bool(c4_pass)},
        'verdict_pass': n_pass,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round27_orb_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
