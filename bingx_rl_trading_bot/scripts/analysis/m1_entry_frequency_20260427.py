"""
Phase 0.2: M1 Scalping — Entry Frequency Sanity Check
=====================================================
M1-A spec **그대로** entry condition pass count (BT 아님, 1초 query).

진행:
  1. 5m + 15m + 1h + 4h(1h resample) MTF 동시 evaluation
  2. 각 5m 캔들에서 LONG/SHORT entry 조건 단계별 pass count:
     A: Trend filter (1h+4h alignment)
     B: A + 5m RSI cross
     C: B + 5m body + EMA9
     D: C + 15m EMA9>EMA21 alignment  ← 최종 entry candidate
  3. 일평균 entries 계산 (LONG + SHORT 합)

GO 조건: D / days ≥ 2 entries/day
FAIL 시: spec 너무 strict → 사용자 보고

NO modifications to spec — 측정만.
"""
import sys, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def load_ohlcv(path):
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    return df


def compute_ema(values, period):
    """Standard EMA. Returns np array, NaN for warmup < period."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = arr[:period].mean()
    for i in range(period, n):
        out[i] = arr[i] * alpha + out[i - 1] * (1 - alpha)
    return out


def compute_rsi(closes, period=14):
    """Wilder RSI. Returns np array."""
    arr = np.asarray(closes, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    diffs = np.diff(arr)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def resample_to_4h(df_1h):
    df = df_1h.set_index('timestamp')
    df4 = df.resample('4h', origin='epoch', label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()
    return df4


def main():
    print("Loading...")
    df_5m = load_ohlcv(ROOT / 'data' / 'btc_5m_720days_binance.csv')
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_1h = load_ohlcv(ROOT / 'data' / 'btc_1h_720days.csv')
    df_4h = resample_to_4h(df_1h)
    print(f"  5m={len(df_5m):,} 15m={len(df_15m):,} 1h={len(df_1h):,} 4h={len(df_4h):,}")

    # --- Compute indicators ---
    print("Computing indicators...")
    # 5m
    df_5m['ema9'] = compute_ema(df_5m['close'].values, 9)
    df_5m['rsi14'] = compute_rsi(df_5m['close'].values, 14)
    df_5m['body'] = (df_5m['close'] - df_5m['open']).abs()
    df_5m['range'] = (df_5m['high'] - df_5m['low']).replace(0, np.nan)
    df_5m['body_ratio'] = df_5m['body'] / df_5m['range']

    # 15m
    df_15m['ema9'] = compute_ema(df_15m['close'].values, 9)
    df_15m['ema21'] = compute_ema(df_15m['close'].values, 21)
    df_15m['ema9_above_ema21'] = df_15m['ema9'] > df_15m['ema21']

    # 1h
    df_1h['ema20'] = compute_ema(df_1h['close'].values, 20)
    df_1h['ema50'] = compute_ema(df_1h['close'].values, 50)
    df_1h['ema20_above_ema50'] = df_1h['ema20'] > df_1h['ema50']

    # 4h
    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['close_above_ema50'] = df_4h['close'] > df_4h['ema50']

    # --- Causal merge: for each 5m bar, attach floor(15m), floor(1h), floor(4h) values ---
    # IMPORTANT: when 5m bar at timestamp t closes, we use the LAST CLOSED 15m/1h/4h bar.
    # The floor of t to 15m/1h/4h frequency identifies the bar that just closed at boundaries,
    # but for non-boundary 5m bars, we still want the most recent CLOSED higher TF.
    # Simplest causal rule: for 5m bar at t, use higher TF bar with timestamp <= t - 5min
    # (i.e., previous closed bar). But our 'timestamp' represents bar OPEN time typically.
    # Convention: timestamp = bar OPEN. Bar [t, t+5min) closes at t+5min. So at moment t+5min
    # we know everything about [t, t+5min). For HTF, same rule. Higher TF bar at timestamp T
    # closes at T+TF. So at 5m bar [t, t+5min), the most recent CLOSED HTF bar has T + TF <= t.
    # For trading decision at 5m bar t (after it closes, we trade at t+5min open), we use
    # HTF bars with T + TF <= t + 5min, i.e., T <= t + 5min - TF.
    #
    # Simplification: use merge_asof on the 5m close time.
    print("Merging MTF (causal)...")
    df_5m['close_time'] = df_5m['timestamp'] + pd.Timedelta(minutes=5)

    # 1h merge
    df_1h_x = df_1h[['timestamp', 'ema20_above_ema50']].copy()
    df_1h_x['close_time_1h'] = df_1h_x['timestamp'] + pd.Timedelta(minutes=60)
    df_1h_x = df_1h_x.sort_values('close_time_1h')
    df_5m = pd.merge_asof(df_5m.sort_values('close_time'),
                          df_1h_x[['close_time_1h', 'ema20_above_ema50']].rename(
                              columns={'close_time_1h': 'close_time', 'ema20_above_ema50': '1h_ema20_above_ema50'}),
                          on='close_time', direction='backward')

    # 4h merge
    df_4h_x = df_4h[['timestamp', 'close_above_ema50']].copy()
    df_4h_x['close_time_4h'] = df_4h_x['timestamp'] + pd.Timedelta(minutes=240)
    df_4h_x = df_4h_x.sort_values('close_time_4h')
    df_5m = pd.merge_asof(df_5m.sort_values('close_time'),
                          df_4h_x[['close_time_4h', 'close_above_ema50']].rename(
                              columns={'close_time_4h': 'close_time', 'close_above_ema50': '4h_close_above_ema50'}),
                          on='close_time', direction='backward')

    # 15m merge
    df_15m_x = df_15m[['timestamp', 'ema9_above_ema21']].copy()
    df_15m_x['close_time_15m'] = df_15m_x['timestamp'] + pd.Timedelta(minutes=15)
    df_15m_x = df_15m_x.sort_values('close_time_15m')
    df_5m = pd.merge_asof(df_5m.sort_values('close_time'),
                          df_15m_x[['close_time_15m', 'ema9_above_ema21']].rename(
                              columns={'close_time_15m': 'close_time', 'ema9_above_ema21': '15m_ema9_above_ema21'}),
                          on='close_time', direction='backward')

    df_5m = df_5m.sort_values('timestamp').reset_index(drop=True)

    # --- Entry condition evaluation ---
    print("Evaluating entry conditions...")
    n = len(df_5m)
    rsi = df_5m['rsi14'].values
    close = df_5m['close'].values
    ema9_5m = df_5m['ema9'].values
    body_ratio = df_5m['body_ratio'].values
    htf_1h_raw = df_5m['1h_ema20_above_ema50']
    htf_4h_raw = df_5m['4h_close_above_ema50']
    ltf_15m_raw = df_5m['15m_ema9_above_ema21']
    htf_1h = htf_1h_raw.fillna(False).astype(bool).values
    htf_4h = htf_4h_raw.fillna(False).astype(bool).values
    ltf_15m = ltf_15m_raw.fillna(False).astype(bool).values
    htf_1h_known = ~htf_1h_raw.isna()
    htf_4h_known = ~htf_4h_raw.isna()
    ltf_15m_known = ~ltf_15m_raw.isna()

    # RSI cross above 40: rsi[i] > 40 AND any(rsi[i-3:i]) <= 40
    # RSI cross below 60: rsi[i] < 60 AND any(rsi[i-3:i]) >= 60
    rsi_min_lookback = np.array([np.nan if i < 3 else rsi[i - 3:i].min() for i in range(n)])
    rsi_max_lookback = np.array([np.nan if i < 3 else rsi[i - 3:i].max() for i in range(n)])

    long_rsi_cross = (rsi > 40) & (rsi_min_lookback <= 40)
    short_rsi_cross = (rsi < 60) & (rsi_max_lookback >= 60)

    body_ok = body_ratio > 0.4
    long_close_above_ema9 = close > ema9_5m
    short_close_below_ema9 = close < ema9_5m

    # LONG counts
    A_long = pd.Series(htf_1h & htf_4h, dtype=bool)
    B_long = A_long & long_rsi_cross
    C_long = B_long & body_ok & long_close_above_ema9
    D_long = C_long & ltf_15m

    # SHORT counts (mirror)
    htf_1h_short = (~htf_1h) & htf_1h_known.values
    htf_4h_short = (~htf_4h) & htf_4h_known.values
    ltf_15m_short = (~ltf_15m) & ltf_15m_known.values

    A_short = pd.Series(htf_1h_short & htf_4h_short, dtype=bool)
    B_short = A_short & short_rsi_cross
    C_short = B_short & body_ok & short_close_below_ema9
    D_short = C_short & ltf_15m_short

    # Total bars usable (after warmup)
    usable_mask = pd.Series((~pd.isna(rsi)) & (~pd.isna(ema9_5m))).values & \
                  htf_1h_known.values & htf_4h_known.values & ltf_15m_known.values
    n_usable = int(usable_mask.sum())
    days = n_usable * 5 / 60 / 24

    # Counts (only on usable rows)
    counts = {
        'total_5m_bars': n,
        'usable_bars': n_usable,
        'usable_days': round(days, 2),

        'A_long_pct': round(100 * (A_long & usable_mask).sum() / n_usable, 2),
        'A_short_pct': round(100 * (A_short & usable_mask).sum() / n_usable, 2),
        'A_total_pct': round(100 * ((A_long | A_short) & usable_mask).sum() / n_usable, 2),

        'B_long_n': int((B_long & usable_mask).sum()),
        'B_short_n': int((B_short & usable_mask).sum()),

        'C_long_n': int((C_long & usable_mask).sum()),
        'C_short_n': int((C_short & usable_mask).sum()),

        'D_long_n': int((D_long & usable_mask).sum()),
        'D_short_n': int((D_short & usable_mask).sum()),
        'D_total_n': int(((D_long | D_short) & usable_mask).sum()),

        'D_long_per_day': round(int((D_long & usable_mask).sum()) / days, 3),
        'D_short_per_day': round(int((D_short & usable_mask).sum()) / days, 3),
        'D_total_per_day': round(int(((D_long | D_short) & usable_mask).sum()) / days, 3),
    }

    print("\n=== Entry Frequency Funnel (M1-A spec as-is) ===")
    print(f"Usable: {counts['usable_bars']:,} bars / {counts['usable_days']:.1f} days\n")
    print(f"  A. Trend filter (1h+4h aligned):")
    print(f"     LONG  : {counts['A_long_pct']:>6.2f}% of bars")
    print(f"     SHORT : {counts['A_short_pct']:>6.2f}% of bars")
    print(f"     Total : {counts['A_total_pct']:>6.2f}% of bars (LONG OR SHORT)\n")

    print(f"  B. + 5m RSI cross (40↑ for LONG / 60↓ for SHORT):")
    print(f"     LONG  : {counts['B_long_n']:,}")
    print(f"     SHORT : {counts['B_short_n']:,}\n")

    print(f"  C. + body>40% AND close vs EMA9:")
    print(f"     LONG  : {counts['C_long_n']:,}")
    print(f"     SHORT : {counts['C_short_n']:,}\n")

    print(f"  D. + 15m EMA9 vs EMA21 alignment (FINAL entry candidates):")
    print(f"     LONG  : {counts['D_long_n']:,} ({counts['D_long_per_day']:.2f}/day)")
    print(f"     SHORT : {counts['D_short_n']:,} ({counts['D_short_per_day']:.2f}/day)")
    print(f"     TOTAL : {counts['D_total_n']:,} ({counts['D_total_per_day']:.2f}/day)\n")

    print("=== Verdict (Criterion 7: ≥ 2 trades/day) ===")
    if counts['D_total_per_day'] >= 2.0:
        print(f"PASS — {counts['D_total_per_day']:.2f}/day raw entries (before min_bars_between_trades=2 dedupe)")
        print("       실제 BT에서 dedupe로 약간 감소하지만 margin 충분")
    elif counts['D_total_per_day'] >= 1.0:
        print(f"BORDERLINE — {counts['D_total_per_day']:.2f}/day. min_bars_between=2 dedupe 후 < 2 가능성. 사용자 보고 필요.")
    else:
        print(f"FAIL — {counts['D_total_per_day']:.2f}/day < 2.0. spec 너무 strict.")
        print("       사용자 confirm 필요: filter 완화 vs 다른 paradigm")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec': 'M1-A as-is (Plan §3)',
        **counts,
        'criterion_7_threshold': 2.0,
        'verdict': 'PASS' if counts['D_total_per_day'] >= 2.0 else (
            'BORDERLINE' if counts['D_total_per_day'] >= 1.0 else 'FAIL'),
    }
    p = ROOT / 'results' / f'm1_entry_frequency_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
