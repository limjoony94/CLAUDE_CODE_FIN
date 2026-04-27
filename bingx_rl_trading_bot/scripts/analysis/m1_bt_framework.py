"""
Phase 1: M1 Scalping BT Framework — M1-A spec (with Phase 0.2 D3 buffered 15m)
==============================================================================
Friction-aware BT (sl 0.10% RT fee + 0.05% × 2 entry/exit slippage = 0.20%/trade).

Spec (Plan §3, post Phase 0.2):
  Trend filter:
    1h: EMA20 > EMA50 (LONG bias)  /  EMA20 < EMA50 (SHORT bias)
    4h: close > 4h EMA50 (LONG)   /  close < 4h EMA50 (SHORT)
    BOTH 1h AND 4h aligned required
  Entry trigger (5m + 15m):
    5m: RSI(14) crossed above 40 (LONG, 직전 3봉 ≤40 후 회복) [SHORT: 60↓]
    5m: close > EMA9 (LONG)  [SHORT: close < EMA9]
    5m: body / range > 0.4
    15m buffered: EMA9_15m ≥ EMA21_15m × 0.999 (LONG) / ≤ × 1.001 (SHORT)
  Exit:
    SL: max(직전 5m swing_low, entry − 1.5 × 5m_ATR)  [SHORT: min/symmetric]
    Emergency_SL: −1.5% hard
    TP_trail: best_price − 2.0 × 5m_ATR  [trailing, ratchet]
    Timeout: 24 bars (= 2h)
  Position:
    N=1, leverage 1x (criterion 1x 평가)
  Frequency:
    min_bars_between_trades = 2
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------- Indicators ----------

def compute_atr(highs, lows, closes, period=14):
    n = len(closes)
    tr = [0.0] * n
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    atr = [float('nan')] * n
    if n >= period:
        atr[period - 1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_ema(values, period):
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
    out[period] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return out


def compute_swings_5m(lows, highs, lookback=10):
    """Causal swing low/high (rolling min/max past `lookback` bars including current)."""
    n = len(lows)
    sw_low = [float('nan')] * n
    sw_high = [float('nan')] * n
    cur_l = float('nan'); cur_h = float('nan')
    for i in range(lookback, n):
        wlow = lows[i - lookback:i + 1]
        whigh = highs[i - lookback:i + 1]
        if lows[i] == min(wlow): cur_l = lows[i]
        if highs[i] == max(whigh): cur_h = highs[i]
        sw_low[i] = cur_l; sw_high[i] = cur_h
    return sw_low, sw_high


# ---------- Data ----------

def load_ohlcv(path):
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    return df


def resample_to_4h(df_1h):
    df = df_1h.set_index('timestamp')
    df4 = df.resample('4h', origin='epoch', label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()
    return df4


def merge_htf(df_target, df_htf, htf_minutes, cols):
    df_htf = df_htf.copy()
    df_htf['close_time'] = df_htf['timestamp'] + pd.Timedelta(minutes=htf_minutes)
    df_htf = df_htf[['close_time'] + cols].sort_values('close_time')
    return pd.merge_asof(df_target.sort_values('close_time'), df_htf, on='close_time', direction='backward')


# ---------- BT logic ----------

class M1ABot:
    def __init__(self, friction_per_trade=0.20):
        self.friction_per_trade = friction_per_trade
        self.atr_period = 14
        self.rsi_period = 14
        self.ema_5m_period = 9
        self.swing_lookback = 10
        self.rsi_long_threshold = 40
        self.rsi_short_threshold = 60
        self.rsi_lookback_bars = 3
        self.body_min_ratio = 0.4
        self.sl_atr_mult = 1.5
        self.trail_k = 2.0
        self.emergency_pct = 1.5  # %
        self.timeout_bars = 24
        self.min_bars_between = 2

    def check_entry(self, idx, df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask):
        """Return entry dict or None.
        idx = current 5m bar index (closing decision)."""
        if not valid_mask[idx]:
            return None
        rsi = df_5m['rsi14'].values
        close = df_5m['close'].values
        ema9 = df_5m['ema9'].values
        body_ratio = df_5m['body_ratio'].values

        # RSI cross detection
        if idx < self.rsi_lookback_bars:
            return None
        rsi_min = rsi[idx - self.rsi_lookback_bars:idx].min()
        rsi_max = rsi[idx - self.rsi_lookback_bars:idx].max()

        # body + EMA9
        if not (body_ratio[idx] > self.body_min_ratio):
            return None

        # LONG check
        if (h1_long[idx] and h4_long[idx] and d3_long[idx]
                and rsi[idx] > self.rsi_long_threshold and rsi_min <= self.rsi_long_threshold
                and close[idx] > ema9[idx]):
            return {'direction': 'LONG'}
        # SHORT check
        if ((not h1_long[idx]) and (not h4_long[idx]) and d3_short[idx]
                and rsi[idx] < self.rsi_short_threshold and rsi_max >= self.rsi_short_threshold
                and close[idx] < ema9[idx]):
            return {'direction': 'SHORT'}
        return None

    def compute_sl(self, direction, entry_price, swing_low, swing_high, atr):
        """SL = max(swing_low_ref, entry − 1.5×ATR) for LONG; mirror for SHORT."""
        atr_dist = self.sl_atr_mult * atr
        if direction == 'LONG':
            atr_sl = entry_price - atr_dist
            structural = swing_low if not np.isnan(swing_low) else atr_sl
            return max(structural, atr_sl)
        else:
            atr_sl = entry_price + atr_dist
            structural = swing_high if not np.isnan(swing_high) else atr_sl
            return min(structural, atr_sl)

    def compute_trail_trigger(self, direction, entry_price, best_price, atr):
        """TP trigger = best - 2×ATR (LONG) / best + 2×ATR (SHORT)."""
        if direction == 'LONG':
            return best_price - self.trail_k * atr
        else:
            return best_price + self.trail_k * atr

    def emergency_price(self, direction, entry_price):
        if direction == 'LONG':
            return entry_price * (1 - self.emergency_pct / 100)
        else:
            return entry_price * (1 + self.emergency_pct / 100)


def run_bt(df_5m, h1_long_arr, h4_long_arr, d3_long_arr, d3_short_arr, valid_mask,
           swing_low, swing_high, atr_5m, bot=None, friction=0.20):
    """Single-pass BT. N=1, additive 1x return.
    Entry decided at end of bar i, executed at open of bar i+1.
    Exit checked intrabar at bar j (j > i): SL first, then trail, then timeout.
    """
    if bot is None:
        bot = M1ABot(friction_per_trade=friction)

    n = len(df_5m)
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    closes = df_5m['close'].values
    timestamps = df_5m['timestamp'].values

    in_pos = False
    pdir = None; pentry = None; psl = None; pbest = None; pstart_idx = None
    pemerg = None
    cooldown_until = 0
    trades = []

    i = 0
    while i < n:
        if in_pos:
            # exit check at bar i (intrabar)
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            atr_now = atr_5m[i] if not np.isnan(atr_5m[i]) else (atr_5m[i - 1] if i > 0 else 0)

            # update best_price (running) BEFORE exit checks, simulating intrabar drift
            if pdir == 'LONG':
                pbest = max(pbest, bar_high)
            else:
                pbest = min(pbest, bar_low)

            # priority: emergency_sl > sl > trail_tp > timeout
            exit_price = None
            exit_reason = None

            # Emergency 1.5%
            if pdir == 'LONG':
                if bar_low <= pemerg:
                    exit_price = pemerg
                    exit_reason = 'EMERGENCY'
            else:
                if bar_high >= pemerg:
                    exit_price = pemerg
                    exit_reason = 'EMERGENCY'

            # SL (structural + ATR)
            if exit_price is None:
                if pdir == 'LONG' and bar_low <= psl:
                    exit_price = psl
                    exit_reason = 'SL'
                elif pdir == 'SHORT' and bar_high >= psl:
                    exit_price = psl
                    exit_reason = 'SL'

            # Trail TP — trigger evaluated each bar from updated best
            if exit_price is None:
                trail_trigger = bot.compute_trail_trigger(pdir, pentry, pbest, atr_now)
                if pdir == 'LONG' and bar_low <= trail_trigger:
                    exit_price = trail_trigger
                    exit_reason = 'TRAIL_TP'
                elif pdir == 'SHORT' and bar_high >= trail_trigger:
                    exit_price = trail_trigger
                    exit_reason = 'TRAIL_TP'

            # Timeout
            held = i - pstart_idx
            if exit_price is None and held >= bot.timeout_bars:
                exit_price = bar_close
                exit_reason = 'TIMEOUT'

            if exit_price is not None:
                if pdir == 'LONG':
                    gross_pct = (exit_price / pentry - 1) * 100
                else:
                    gross_pct = (1 - exit_price / pentry) * 100
                net_pct = gross_pct - friction
                trades.append({
                    'entry_ts': str(timestamps[pstart_idx]),
                    'exit_ts': str(timestamps[i]),
                    'direction': pdir,
                    'entry': float(pentry),
                    'exit': float(exit_price),
                    'sl': float(psl),
                    'gross_pct': round(gross_pct, 4),
                    'net_pct': round(net_pct, 4),
                    'reason': exit_reason,
                    'bars_held': held,
                })
                in_pos = False
                cooldown_until = i + bot.min_bars_between

        if not in_pos and i >= cooldown_until:
            sig = bot.check_entry(i, df_5m, h1_long_arr, h4_long_arr, d3_long_arr, d3_short_arr, valid_mask)
            if sig:
                ni = i + 1
                if ni < n:
                    pentry = opens[ni]
                    psl = bot.compute_sl(sig['direction'], pentry, swing_low[i], swing_high[i],
                                         atr_5m[i] if not np.isnan(atr_5m[i]) else 0)
                    pemerg = bot.emergency_price(sig['direction'], pentry)
                    pdir = sig['direction']
                    pbest = highs[ni] if pdir == 'LONG' else lows[ni]
                    pstart_idx = ni
                    in_pos = True
                    i = ni  # advance to entry bar
                    continue
        i += 1

    return trades


def prepare_data(data_path_5m, data_path_15m, data_path_1h):
    """Load all TFs + indicators + merged signal flags."""
    df_5m = load_ohlcv(data_path_5m)
    df_15m = load_ohlcv(data_path_15m)
    df_1h = load_ohlcv(data_path_1h)
    df_4h = resample_to_4h(df_1h)

    # 5m indicators
    df_5m['ema9'] = compute_ema(df_5m['close'].values, 9)
    df_5m['rsi14'] = compute_rsi(df_5m['close'].values, 14)
    df_5m['atr14'] = compute_atr(df_5m['high'].tolist(), df_5m['low'].tolist(),
                                  df_5m['close'].tolist(), 14)
    sw_low, sw_high = compute_swings_5m(df_5m['low'].tolist(), df_5m['high'].tolist(), 10)
    df_5m['swing_low'] = sw_low
    df_5m['swing_high'] = sw_high
    df_5m['body_ratio'] = (df_5m['close'] - df_5m['open']).abs() / \
        (df_5m['high'] - df_5m['low']).replace(0, np.nan)

    # 15m indicators
    df_15m['ema9'] = compute_ema(df_15m['close'].values, 9)
    df_15m['ema21'] = compute_ema(df_15m['close'].values, 21)
    df_15m['D3_long'] = df_15m['ema9'] >= df_15m['ema21'] * 0.999
    df_15m['D3_short'] = df_15m['ema9'] <= df_15m['ema21'] * 1.001

    # 1h indicators
    df_1h['ema20'] = compute_ema(df_1h['close'].values, 20)
    df_1h['ema50'] = compute_ema(df_1h['close'].values, 50)
    df_1h['htf_long'] = df_1h['ema20'] > df_1h['ema50']

    # 4h indicators
    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf_long'] = df_4h['close'] > df_4h['ema50']

    # MTF causal merge
    df_5m['close_time'] = df_5m['timestamp'] + pd.Timedelta(minutes=5)
    df_5m = merge_htf(df_5m, df_1h.rename(columns={'htf_long': 'h1_long'}), 60, ['h1_long'])
    df_5m = merge_htf(df_5m, df_4h.rename(columns={'htf_long': 'h4_long'}), 240, ['h4_long'])
    df_5m = merge_htf(df_5m, df_15m, 15, ['D3_long', 'D3_short'])
    df_5m = df_5m.sort_values('timestamp').reset_index(drop=True)

    # Boolean arrays (NaN → False with known mask)
    h1_long = df_5m['h1_long'].fillna(False).astype(bool).values
    h4_long = df_5m['h4_long'].fillna(False).astype(bool).values
    d3_long = df_5m['D3_long'].fillna(False).astype(bool).values
    d3_short = df_5m['D3_short'].fillna(False).astype(bool).values

    valid_mask = ((~pd.isna(df_5m['rsi14'])) & (~pd.isna(df_5m['ema9']))
                   & (~pd.isna(df_5m['atr14'])) & (~pd.isna(df_5m['swing_low']))
                   & (~df_5m['h1_long'].isna()) & (~df_5m['h4_long'].isna())
                   & (~df_5m['D3_long'].isna()) & (~df_5m['D3_short'].isna())).values

    return df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask


def main():
    print("Loading data + indicators...")
    df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask = prepare_data(
        ROOT / 'data' / 'btc_5m_720days_binance.csv',
        ROOT / 'data' / 'btc_15m_720days.csv',
        ROOT / 'data' / 'btc_1h_720days.csv',
    )
    print(f"  5m: {len(df_5m):,} bars, valid: {int(valid_mask.sum()):,}")

    print("\nRunning BT (M1-A spec, friction 0.20%/trade)...")
    bot = M1ABot()
    trades = run_bt(df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask,
                    df_5m['swing_low'].values, df_5m['swing_high'].values,
                    df_5m['atr14'].values, bot=bot, friction=0.20)
    print(f"  trades: {len(trades)}")

    if not trades:
        print("No trades — abort.")
        return

    # Quick stats
    nets = [t['net_pct'] for t in trades]
    grosses = [t['gross_pct'] for t in trades]
    sum_net = sum(nets); sum_gross = sum(grosses)
    wins = sum(1 for x in nets if x > 0)
    wr = 100 * wins / len(nets)
    avg_net = sum_net / len(nets)
    days = (pd.to_datetime(trades[-1]['exit_ts']) - pd.to_datetime(trades[0]['entry_ts'])).days
    daily = sum_net / days if days > 0 else 0

    avg_win = sum(x for x in nets if x > 0) / max(1, wins)
    avg_loss = sum(x for x in nets if x <= 0) / max(1, len(nets) - wins)
    rr = abs(avg_win / avg_loss) if avg_loss < 0 else float('inf')

    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1

    print(f"\n=== M1-A BT (friction 0.20% / trade) ===")
    print(f"  Period       : {trades[0]['entry_ts']} ~ {trades[-1]['exit_ts']} ({days} days)")
    print(f"  Trades       : {len(trades)}")
    print(f"  Trades/day   : {len(trades)/days:.2f}")
    print(f"  WR           : {wr:.1f}%")
    print(f"  Sum net 1x   : {sum_net:+.2f}%  (gross {sum_gross:+.2f}%)")
    print(f"  Avg net/tr   : {avg_net:+.4f}%")
    print(f"  R:R          : {rr:.2f}")
    print(f"  Daily 1x     : {daily:+.4f}%")
    print(f"  Reasons      : {reasons}")

    print(f"\n=== Criterion check ===")
    crit = {
        '1_WR_50pct': wr >= 50,
        '2_RR_1.0': rr >= 1.0,
        '5_daily_0.2pct': daily >= 0.2,
        '6_avg_gt_friction': avg_net > 0,  # net > 0 = gross > friction
        '7_2trades_per_day': len(trades) / days >= 2.0,
    }
    for k, v in crit.items():
        print(f"  {k:<24}: {'PASS' if v else 'FAIL'}")
    print(f"  Hard criteria pass: {sum(crit.values())}/5")

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec': 'M1-A (Plan §3 + Phase 0.2 D3 buffered)',
        'period': {'start': trades[0]['entry_ts'], 'end': trades[-1]['exit_ts'], 'days': days},
        'n_trades': len(trades),
        'trades_per_day': round(len(trades) / days, 3),
        'wr_pct': round(wr, 2),
        'sum_net_1x_pct': round(sum_net, 2),
        'sum_gross_1x_pct': round(sum_gross, 2),
        'avg_net_per_trade_pct': round(avg_net, 4),
        'rr': round(rr, 3),
        'daily_1x_pct': round(daily, 4),
        'reasons': reasons,
        'criterion_pass': crit,
        'criterion_pass_count': sum(crit.values()),
    }
    p = ROOT / 'results' / f'm1_bt_phase2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    # Save trades separately
    pt = ROOT / 'results' / f'm1_bt_phase2_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(pt, 'w') as f: json.dump(trades, f, indent=2, default=str)
    print(f"\nSaved: {p}")
    print(f"Saved: {pt}")


if __name__ == '__main__':
    main()
