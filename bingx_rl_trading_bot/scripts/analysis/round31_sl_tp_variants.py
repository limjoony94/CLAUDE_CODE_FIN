"""Round 31 — Two SL/TP Mechanism Variants.

Pre-reg: claudedocs/round31_sl_tp_variants_prereg.md (commit e1ba7dc)

Variant A: TP = period_low/high level, SL = period extreme + 0.5×ATR buffer
Variant B: TP = SL = 1.5×entry_candle_range (symmetric R:R 1.0)

Both on R29 baseline (15m, lookback 16, fade, 50% body filter).
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
    'direction': 'fade',
    'max_hold_bars': 96,
    'taker_per_side_pct': 0.05,
    'maker_per_side_pct': 0.02,
    'capital_usd': 1500,
    'a_sl_atr_buffer_mult': 0.5,
    'a_atr_period': 14,
    'b_candle_range_multiple': 1.5,
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
        candle_range = high[i] - low[i]
        if close[i] > period_high and close[i] > open_[i]:
            signals.append({'idx': i, 'fade_side': 'SHORT',
                            'period_high': period_high, 'period_low': period_low,
                            'period_range': period_range, 'candle_range': candle_range})
        elif close[i] < period_low and close[i] < open_[i]:
            signals.append({'idx': i, 'fade_side': 'LONG',
                            'period_high': period_high, 'period_low': period_low,
                            'period_range': period_range, 'candle_range': candle_range})
    return signals


def simulate(df: pd.DataFrame, signals: list, variant: str) -> pd.DataFrame:
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    ts = df['timestamp'].values
    atr = compute_atr(df, LOCKED['a_atr_period']).values

    taker = LOCKED['taker_per_side_pct']
    maker = LOCKED['maker_per_side_pct']
    max_hold = LOCKED['max_hold_bars']

    trades = []
    cooldown_until = -1
    for sig in signals:
        i = sig['idx']
        if i < cooldown_until or i + 1 >= n:
            continue
        entry_idx = i + 1
        entry_price = open_[entry_idx]
        side = sig['fade_side']

        if variant == 'A':
            # Period-level absolute targets
            atr_now = atr[i] if not np.isnan(atr[i]) else 0
            if atr_now <= 0:
                continue
            buffer = LOCKED['a_sl_atr_buffer_mult'] * atr_now
            if side == 'SHORT':
                tp = sig['period_low']
                sl = sig['period_high'] + buffer
            else:
                tp = sig['period_high']
                sl = sig['period_low'] - buffer
        elif variant == 'B':
            # Symmetric TP/SL around entry candle
            cr = sig['candle_range']
            dist = LOCKED['b_candle_range_multiple'] * cr
            if side == 'SHORT':
                tp = entry_price - dist
                sl = entry_price + dist
            else:
                tp = entry_price + dist
                sl = entry_price - dist
        else:
            continue

        # Check sl/tp validity
        if side == 'SHORT' and (tp >= entry_price or sl <= entry_price):
            continue
        if side == 'LONG' and (tp <= entry_price or sl >= entry_price):
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
    return {
        'n_trades': n, 'cum_net_pct': cum_net,
        'avg_gross_per_trade_pct': avg_gross,
        'wr': wr, 'rr_realized': rr,
        'daily_pct': daily_pct,
        'bs_pos_rate': bs_pos,
    }


def main():
    print('=' * 100)
    print('Round 31 — Two SL/TP Mechanism Variants')
    print('=' * 100)
    print('Pre-reg: claudedocs/round31_sl_tp_variants_prereg.md (e1ba7dc)\n')

    df = load_data()
    n_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars (15m), {n_days:.1f} days\n')

    signals = detect_signals(df)
    print(f'Signals: {len(signals)}\n')

    for variant in ['A', 'B']:
        print(f'=== Variant {variant} ===')
        if variant == 'A':
            print('  TP = period_low/high (absolute level)')
            print(f'  SL = period extreme + {LOCKED["a_sl_atr_buffer_mult"]}× ATR(14) buffer')
        else:
            print(f'  TP = SL = {LOCKED["b_candle_range_multiple"]}× entry_candle_range (symmetric R:R 1.0)')

        trades = simulate(df, signals, variant)
        print(f'  trades: {len(trades)}')
        summ = summarize(trades, n_days)
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

        out_var = {
            'variant': variant,
            'summary': summ,
            'c1_pass': bool(c1), 'c2_pass': bool(c2),
            'c3_pass': bool(c3), 'c4_pass': bool(c4),
            'n_pass': n_pass,
        }
        if variant == 'A':
            out_a = out_var
        else:
            out_b = out_var

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'e1ba7dc',
        'locked': LOCKED,
        'variant_a': out_a,
        'variant_b': out_b,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round31_sl_tp_variants_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
