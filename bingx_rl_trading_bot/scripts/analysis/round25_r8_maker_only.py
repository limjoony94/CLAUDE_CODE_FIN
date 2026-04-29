"""Round 25 — R8 1h Donchian Breakout + Maker-Only Execution.

Pre-reg: claudedocs/round25_r8_maker_only_prereg.md (commit 5b9c193)

Mechanism:
  Identical R8 1h Donchian breakout signal logic.
  Entry: limit at signal_close - 0.05% (long) / + 0.05% (short), 1-bar max wait.
  TP exit: limit at trailing 2.5×ATR (maker).
  SL exit: market at fractal swing point (taker, capped 3.3×ATR).
  Timeout 192 bars at market.

Locked: limit_offset 0.05%, max_wait 1 bar. NO TUNING.
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
    'atr_period': 14,
    'fractal_lookback': 5,
    'fractal_atr_mult_cap': 3.3,
    'trail_atr_mult': 2.5,
    'max_hold_bars': 192,
    'limit_entry_offset_pct': 0.05,
    'limit_max_wait_bars': 1,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'capital_usd': 1500,
}

GATES = {
    'gate_A_min_setups': 100,
    'wf_min_pos_folds': 3,
    'wf_total_folds': 5,
    'bs_n_iter': 1000,
    'bs_window_days': 3,
    'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    't4_min_daily_pct': 0.20,
    't5_min_wr': 0.30,
    't7_min_trades_per_day': 2.0,
    't8_min_per_trade_gross_pct': 0.07,
    't9_max_5d_dd_pct': 15.0,
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


def find_fractal_sl(df: pd.DataFrame, i: int, side: str, atr_now: float) -> float:
    """Find swing high/low in last fractal_lookback bars; capped by atr × mult."""
    fl = LOCKED['fractal_lookback']
    cap = LOCKED['fractal_atr_mult_cap']
    if i < fl:
        return None
    window = df.iloc[i - fl:i]
    if side == 'LONG':
        sl = window['low'].min()
        max_sl_distance = cap * atr_now
        if (df.iloc[i]['close'] - sl) > max_sl_distance:
            sl = df.iloc[i]['close'] - max_sl_distance
        return sl
    else:
        sl = window['high'].max()
        max_sl_distance = cap * atr_now
        if (sl - df.iloc[i]['close']) > max_sl_distance:
            sl = df.iloc[i]['close'] + max_sl_distance
        return sl


def detect_breakouts(df: pd.DataFrame) -> list:
    """R8 logic: close > 15-bar high AND body > 40% of range → breakout signal."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    lookback = LOCKED['channel_lookback_bars']
    body_min_pct = LOCKED['body_min_pct_of_range']
    signals = []
    for i in range(lookback + LOCKED['atr_period'], n - 2):
        prior_high = high[i - lookback:i].max()
        prior_low = low[i - lookback:i].min()
        bar_range = high[i] - low[i]
        if bar_range <= 0:
            continue
        body = abs(close[i] - open_[i])
        body_pct = body / bar_range
        if close[i] > prior_high and body_pct > body_min_pct and close[i] > open_[i]:
            signals.append({'idx': i, 'side': 'LONG', 'close': close[i]})
        elif close[i] < prior_low and body_pct > body_min_pct and close[i] < open_[i]:
            signals.append({'idx': i, 'side': 'SHORT', 'close': close[i]})
    return signals


def simulate_maker_trades(df: pd.DataFrame, signals: list) -> tuple:
    """Apply maker-entry / maker-TP / taker-SL semantics. Return trades + diagnostics."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    atr = compute_atr(df, LOCKED['atr_period']).values
    ts = df['timestamp'].values

    offset = LOCKED['limit_entry_offset_pct'] / 100
    max_wait = LOCKED['limit_max_wait_bars']
    maker_fric = LOCKED['maker_friction_per_side_pct']
    taker_fric = LOCKED['taker_friction_per_side_pct']
    trail_mult = LOCKED['trail_atr_mult']
    max_hold = LOCKED['max_hold_bars']

    trades = []
    in_pos = False
    pos = None
    n_signals = len(signals)
    n_filled = 0
    n_unfilled = 0

    sig_idx = 0
    for i in range(n):
        # Check open position management first
        if in_pos:
            # Update trailing TP based on extreme since entry
            if pos['side'] == 'LONG':
                pos['best_price'] = max(pos['best_price'], high[i])
                trail_tp = pos['best_price'] - trail_mult * pos['atr_at_entry']
                if low[i] <= pos['sl']:
                    exit_price = pos['sl']
                    exit_friction = taker_fric / 100
                    exit_reason = 'SL'
                elif low[i] <= trail_tp and pos['best_price'] > pos['entry_price']:
                    # TP fills as limit (maker) when trailing level is touched
                    exit_price = trail_tp
                    exit_friction = maker_fric / 100
                    exit_reason = 'TRAIL_TP'
                elif (i - pos['entry_idx']) >= max_hold:
                    exit_price = close[i]
                    exit_friction = taker_fric / 100
                    exit_reason = 'TIMEOUT'
                else:
                    exit_price = None
                    exit_friction = None
                    exit_reason = None
            else:  # SHORT
                pos['best_price'] = min(pos['best_price'], low[i])
                trail_tp = pos['best_price'] + trail_mult * pos['atr_at_entry']
                if high[i] >= pos['sl']:
                    exit_price = pos['sl']
                    exit_friction = taker_fric / 100
                    exit_reason = 'SL'
                elif high[i] >= trail_tp and pos['best_price'] < pos['entry_price']:
                    exit_price = trail_tp
                    exit_friction = maker_fric / 100
                    exit_reason = 'TRAIL_TP'
                elif (i - pos['entry_idx']) >= max_hold:
                    exit_price = close[i]
                    exit_friction = taker_fric / 100
                    exit_reason = 'TIMEOUT'
                else:
                    exit_price = None
                    exit_friction = None
                    exit_reason = None

            if exit_price is not None:
                gross_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                if pos['side'] == 'SHORT':
                    gross_pct = -gross_pct
                # Friction: entry maker (always 0.02%) + exit (varies by reason)
                fric_pct = (LOCKED['maker_friction_per_side_pct'] + exit_friction * 100)
                net_pct = gross_pct - fric_pct
                trades.append({
                    'entry_ts': pos['entry_ts'],
                    'exit_ts': ts[i],
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'gross_pct': gross_pct,
                    'net_pct': net_pct,
                    'friction_pct': fric_pct,
                    'exit_reason': exit_reason,
                    'hold_bars': i - pos['entry_idx'],
                })
                in_pos = False
                pos = None

        # Check next signal at i: limit entry to be filled at i+1 (within max_wait bars)
        if not in_pos and sig_idx < n_signals and signals[sig_idx]['idx'] == i:
            sig = signals[sig_idx]
            sig_idx += 1
            limit_price = sig['close'] * (1 - offset) if sig['side'] == 'LONG' else sig['close'] * (1 + offset)
            filled = False
            fill_idx = None
            fill_price = None
            for j in range(1, max_wait + 1):
                if i + j >= n:
                    break
                if sig['side'] == 'LONG' and low[i + j] <= limit_price:
                    filled = True
                    fill_idx = i + j
                    fill_price = limit_price
                    break
                elif sig['side'] == 'SHORT' and high[i + j] >= limit_price:
                    filled = True
                    fill_idx = i + j
                    fill_price = limit_price
                    break

            if filled:
                n_filled += 1
                atr_now = atr[i] if not np.isnan(atr[i]) else atr[i - 1]
                # SL based on fractal at signal bar
                temp_df = df
                sl = find_fractal_sl(temp_df, i, sig['side'], atr_now)
                if sl is None:
                    continue
                pos = {
                    'side': sig['side'],
                    'entry_idx': fill_idx,
                    'entry_ts': ts[fill_idx],
                    'entry_price': fill_price,
                    'sl': sl,
                    'atr_at_entry': atr_now,
                    'best_price': fill_price,
                }
                in_pos = True
            else:
                n_unfilled += 1

        # Skip past signals at i not yet processed (in case sig_idx behind)
        while sig_idx < n_signals and signals[sig_idx]['idx'] < i:
            sig_idx += 1

    return pd.DataFrame(trades), {'n_signals': n_signals,
                                  'n_filled': n_filled,
                                  'n_unfilled': n_unfilled,
                                  'fill_rate': n_filled / n_signals if n_signals > 0 else 0}


def summarize(trades: pd.DataFrame, n_days: float) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0}
    cum_net = float((1 + trades['net_pct'] / 100).prod() - 1) * 100
    n = len(trades)
    avg_gross = float(trades['gross_pct'].mean())
    avg_net = float(trades['net_pct'].mean())
    avg_friction = float(trades['friction_pct'].mean())
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

    return {
        'n_trades': n,
        'cum_net_pct': cum_net,
        'avg_gross_per_trade_pct': avg_gross,
        'avg_net_per_trade_pct': avg_net,
        'avg_friction_per_trade_pct': avg_friction,
        'wr': wr,
        'rr_realized': rr,
        'daily_pct': daily_pct,
        'trades_per_day': trades_per_day,
        'worst_5d_pct': worst_5d,
    }


def main():
    print('=' * 100)
    print('Round 25 — R8 1h Donchian + Maker-Only Execution')
    print('=' * 100)
    print('Pre-reg: claudedocs/round25_r8_maker_only_prereg.md (5b9c193)')
    print(f'Locked: limit_offset {LOCKED["limit_entry_offset_pct"]}%, '
          f'max_wait {LOCKED["limit_max_wait_bars"]} bars, '
          f'maker {LOCKED["maker_friction_per_side_pct"]}%, '
          f'taker {LOCKED["taker_friction_per_side_pct"]}%\n')

    df = load_data()
    n_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars, {n_days:.1f} days\n')

    print('Detecting R8 breakout signals...')
    signals = detect_breakouts(df)
    print(f'  signals: {len(signals)}\n')

    print('Simulating maker-only execution...')
    trades, fill_diag = simulate_maker_trades(df, signals)
    print(f'  signals filled: {fill_diag["n_filled"]}/{fill_diag["n_signals"]} '
          f'(fill rate {fill_diag["fill_rate"]:.4f})')
    print(f'  trades executed: {len(trades)}\n')

    summ = summarize(trades, n_days)
    print('=== Full-sample summary ===')
    for k, v in summ.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    daily = summ.get('daily_pct', 0)
    tpd = summ.get('trades_per_day', 0)
    pt_gross = summ.get('avg_gross_per_trade_pct', 0)
    avg_fric = summ.get('avg_friction_per_trade_pct', 0)
    wr = summ.get('wr', 0)
    rr = summ.get('rr_realized', 0)
    worst5 = summ.get('worst_5d_pct', 0)

    t4_pass = daily >= GATES['t4_min_daily_pct']
    t7_pass = tpd >= GATES['t7_min_trades_per_day']
    # T8: per-trade gross > weighted friction
    t8_pass_taker_baseline = pt_gross >= GATES['t8_min_per_trade_gross_pct']
    t8_pass_maker_realized = pt_gross > avg_fric

    print(f'=== T4 (HARD) Daily ≥ 0.20% ===')
    print(f'  daily: {daily:+.4f}%  → {"PASS" if t4_pass else "FAIL"}\n')
    print(f'=== T5 WR ≥ 0.30 ===')
    print(f'  WR: {wr:.4f}  → {"PASS" if wr >= GATES["t5_min_wr"] else "FAIL"}\n')
    print(f'=== T6 R:R ≥ 1.0 ===')
    print(f'  R:R: {rr:.4f}  → {"PASS" if rr >= 1.0 else "FAIL"}\n')
    print(f'=== T7 (HARD) Trades/day ≥ 2.0 ===')
    print(f'  trades/day: {tpd:.4f}  → {"PASS" if t7_pass else "FAIL"}\n')
    print(f'=== T8 — per-trade gross vs friction ===')
    print(f'  per-trade gross: {pt_gross:+.4f}%')
    print(f'  avg friction: {avg_fric:+.4f}%')
    print(f'  T8 (taker baseline 0.07%): {"PASS" if t8_pass_taker_baseline else "FAIL"}')
    print(f'  T8 (maker realized friction): {"PASS" if t8_pass_maker_realized else "FAIL"}\n')
    print(f'=== T9 Worst 5d ≥ -15.0% ===')
    print(f'  worst 5d: {worst5:+.4f}%  → {"PASS" if worst5 >= -GATES["t9_max_5d_dd_pct"] else "FAIL"}\n')

    n_hard = sum([t4_pass, t7_pass, t8_pass_taker_baseline])

    print('=' * 100)
    print(f'VERDICT: HARD {n_hard}/3  |  fill_rate {fill_diag["fill_rate"]:.3f}')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '5b9c193',
        'locked': LOCKED, 'gates': GATES,
        'fill_diagnostics': fill_diag,
        'full_summary': summ,
        't4': {'daily_pct': daily, 'pass': bool(t4_pass)},
        't7': {'trades_per_day': tpd, 'pass': bool(t7_pass)},
        't8_taker_baseline': {'gross_pct': pt_gross, 'pass': bool(t8_pass_taker_baseline)},
        't8_maker_realized': {'gross_pct': pt_gross, 'avg_friction': avg_fric,
                              'pass': bool(t8_pass_maker_realized)},
        'verdict_hard': n_hard,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round25_r8_maker_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
