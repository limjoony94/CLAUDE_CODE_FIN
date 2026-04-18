#!/usr/bin/env python3
"""
Mean-Reversion vs Breakout + Regime Detection Research
========================================================
Compare three strategies on BTC 15m:

  S1: C1 Breakout (baseline) — LONG when close > ch_high, SHORT when close < ch_low
  S2: Mean-Reversion — LONG when close < ch_low (buy the dip), SHORT when close > ch_high
  S3: Regime-Adaptive — switch between S1 and S2 based on ADX regime

Entry: signal bar[i] -> next bar open[i+1]
Exit variants:
  - Breakout: Fractal SL + ATR trail (production C1 logic)
  - Mean-reversion: Fixed ATR SL + Fixed ATR TP
Fee: 0.10% RT (additive)
PnL: Additive (not compound)
Validation: WF 5-fold, MC 999 sims, Progressive look-ahead

Output: results/mean_reversion_vs_breakout_research.json
"""
import os, sys, math, json, time
from datetime import datetime

import pandas as pd
import numpy as np

os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '.')

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
FEE = 0.10          # 0.10% RT (taker 0.05% x 2)
WARMUP = 30         # bars needed for indicators to settle
MIN_BARS_BETWEEN = 2

# ═══════════════════════════════════════════════════════════════
# Data: load 5m -> resample 15m
# ═══════════════════════════════════════════════════════════════
print("Loading and resampling data...")
df = pd.read_csv('data/btc_5m_270days_reclassified.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['group'] = df.index // 3
agg = df.groupby('group').agg(
    timestamp=('timestamp', 'first'),
    open=('open', 'first'),
    high=('high', 'max'),
    low=('low', 'min'),
    close=('close', 'last'),
    volume=('volume', 'sum')
).reset_index(drop=True)

timestamps = agg['timestamp'].tolist()
o = agg['open'].tolist()
h = agg['high'].tolist()
l = agg['low'].tolist()
c = agg['close'].tolist()
vol = agg['volume'].tolist()
N = len(c)
print(f"  {N} bars (15m), from {timestamps[0]} to {timestamps[-1]}")
days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
print(f"  ~{days:.0f} days")


# ═══════════════════════════════════════════════════════════════
# Additional Indicators: ADX, Bollinger Width
# ═══════════════════════════════════════════════════════════════
def compute_adx(highs, lows, closes, period=14):
    """Wilder-smoothed ADX. Returns (adx, plus_di, minus_di) lists."""
    n = len(closes)
    adx = [float('nan')] * n
    plus_di = [float('nan')] * n
    minus_di = [float('nan')] * n

    # True Range
    tr = [0.0] * n
    tr[0] = highs[0] - lows[0]
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n

    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    if n < period + 1:
        return adx, plus_di, minus_di

    # Wilder smoothing for ATR14, +DM14, -DM14
    atr_s = sum(tr[1:period+1]) / period
    pdm_s = sum(plus_dm[1:period+1]) / period
    mdm_s = sum(minus_dm[1:period+1]) / period

    for i in range(period, n):
        if i == period:
            atr_s = sum(tr[1:period+1]) / period
            pdm_s = sum(plus_dm[1:period+1]) / period
            mdm_s = sum(minus_dm[1:period+1]) / period
        else:
            atr_s = (atr_s * (period - 1) + tr[i]) / period
            pdm_s = (pdm_s * (period - 1) + plus_dm[i]) / period
            mdm_s = (mdm_s * (period - 1) + minus_dm[i]) / period

        if atr_s > 0:
            plus_di[i] = 100 * pdm_s / atr_s
            minus_di[i] = 100 * mdm_s / atr_s
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0

    # DX and ADX (Wilder smoothed)
    dx_vals = []
    first_adx_bar = None
    for i in range(period, n):
        if not math.isnan(plus_di[i]) and not math.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
            else:
                dx = 0.0
            dx_vals.append((i, dx))
            if len(dx_vals) == period:
                first_adx_bar = i
                adx[i] = sum(d[1] for d in dx_vals) / period
            elif len(dx_vals) > period:
                adx[i] = (adx[i-1] * (period - 1) + dx) / period

    return adx, plus_di, minus_di


def compute_bollinger_width(closes, period=20):
    """Bollinger Band width = (upper - lower) / middle."""
    n = len(closes)
    bw = [float('nan')] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        mu = sum(window) / period
        std = (sum((x - mu) ** 2 for x in window) / period) ** 0.5
        if mu > 0:
            bw[i] = (4 * std) / mu * 100  # as percentage
    return bw


# ═══════════════════════════════════════════════════════════════
# Precompute all indicators on full dataset
# ═══════════════════════════════════════════════════════════════
print("Computing indicators...")
atr14 = compute_atr(h, l, c, 14)
ch_h, ch_l = compute_channel(h, l, 15)
sw_l, sw_h = compute_fractal_swings(h, l, 10)
adx_vals, pdi, mdi = compute_adx(h, l, c, 14)
boll_width = compute_bollinger_width(c, 20)


# ═══════════════════════════════════════════════════════════════
# Strategy 1: C1 Breakout (baseline) — production logic
# ═══════════════════════════════════════════════════════════════
cfg = load_config()
sig_cfg = cfg['strategy']


def bt_breakout(oo, hh, ll, cc, atr, ch_hi, ch_lo, sw_lo, sw_hi,
                start=WARMUP, end=None):
    """C1 Breakout backtest — production logic."""
    nn = len(cc) if end is None else end
    sig = C1BreakoutSignal(sig_cfg)
    trades = []
    pos = None
    last_exit = start - MIN_BARS_BETWEEN - 1

    for bar in range(start, nn - 1):
        if pos is not None:
            if pos['d'] == 'LONG':
                pos['bp'] = max(pos['bp'], hh[bar])
            else:
                pos['bp'] = min(pos['bp'], ll[bar])
            pos['bh'] += 1
            ex = sig.check_exit(pos['d'], pos['ep'], pos['bp'],
                                hh[bar], ll[bar], cc[bar],
                                pos['sl'], atr[bar], pos['bh'])
            if ex:
                if pos['d'] == 'LONG':
                    raw = (ex['exit_price'] / pos['ep'] - 1) * 100
                else:
                    raw = (1 - ex['exit_price'] / pos['ep']) * 100
                net = raw - FEE
                trades.append({
                    'pnl': net, 'raw': raw, 'reason': ex['reason'],
                    'entry_bar': pos['ebar'], 'exit_bar': bar,
                    'direction': pos['d'], 'bars_held': pos['bh']
                })
                pos = None
                last_exit = bar
                continue

        if pos is None and bar - last_exit >= MIN_BARS_BETWEEN:
            if (math.isnan(ch_hi[bar]) or math.isnan(ch_lo[bar]) or
                    math.isnan(atr[bar]) or math.isnan(sw_lo[bar]) or
                    math.isnan(sw_hi[bar])):
                continue
            e = sig.check_entry(oo[bar], hh[bar], ll[bar], cc[bar],
                                ch_hi[bar], ch_lo[bar], atr[bar],
                                sw_lo[bar], sw_hi[bar])
            if e and bar + 1 < nn:
                pos = {
                    'd': e['direction'], 'ep': oo[bar + 1],
                    'sl': e['sl_price'], 'bp': oo[bar + 1],
                    'bh': 0, 'ebar': bar + 1
                }

    return trades


# ═══════════════════════════════════════════════════════════════
# Strategy 2: Mean-Reversion — buy dip / sell rally
# ═══════════════════════════════════════════════════════════════
def bt_mean_reversion(oo, hh, ll, cc, atr, ch_hi, ch_lo,
                      sl_atr_k=2.0, tp_atr_k=2.0, body_filter=True,
                      channel_period=15, max_hold=192,
                      start=WARMUP, end=None):
    """
    Mean-reversion: LONG when close < ch_low (buy dip), SHORT when close > ch_high.
    Fixed ATR SL and fixed ATR TP.

    Args:
        sl_atr_k: SL distance in ATR multiples
        tp_atr_k: TP distance in ATR multiples
        body_filter: require reversal body (bullish for LONG, bearish for SHORT)
    """
    nn = len(cc) if end is None else end
    trades = []
    pos = None
    last_exit = start - MIN_BARS_BETWEEN - 1
    body_min_ratio = 0.4  # same filter strength as C1

    for bar in range(start, nn - 1):
        # Exit logic for open position
        if pos is not None:
            pos['bh'] += 1

            # 1. SL check (fixed ATR SL)
            if pos['d'] == 'LONG' and ll[bar] <= pos['sl']:
                raw = (pos['sl'] / pos['ep'] - 1) * 100
                net = raw - FEE
                trades.append({
                    'pnl': net, 'raw': raw, 'reason': 'SL',
                    'entry_bar': pos['ebar'], 'exit_bar': bar,
                    'direction': pos['d'], 'bars_held': pos['bh']
                })
                pos = None; last_exit = bar; continue
            elif pos['d'] == 'SHORT' and hh[bar] >= pos['sl']:
                raw = (1 - pos['sl'] / pos['ep']) * 100
                net = raw - FEE
                trades.append({
                    'pnl': net, 'raw': raw, 'reason': 'SL',
                    'entry_bar': pos['ebar'], 'exit_bar': bar,
                    'direction': pos['d'], 'bars_held': pos['bh']
                })
                pos = None; last_exit = bar; continue

            # 2. TP check (fixed ATR TP)
            if pos['d'] == 'LONG' and hh[bar] >= pos['tp']:
                raw = (pos['tp'] / pos['ep'] - 1) * 100
                net = raw - FEE
                trades.append({
                    'pnl': net, 'raw': raw, 'reason': 'TP',
                    'entry_bar': pos['ebar'], 'exit_bar': bar,
                    'direction': pos['d'], 'bars_held': pos['bh']
                })
                pos = None; last_exit = bar; continue
            elif pos['d'] == 'SHORT' and ll[bar] <= pos['tp']:
                raw = (1 - pos['tp'] / pos['ep']) * 100
                net = raw - FEE
                trades.append({
                    'pnl': net, 'raw': raw, 'reason': 'TP',
                    'entry_bar': pos['ebar'], 'exit_bar': bar,
                    'direction': pos['d'], 'bars_held': pos['bh']
                })
                pos = None; last_exit = bar; continue

            # 3. Timeout
            if pos['bh'] >= max_hold:
                # Exclude timeout trades from PnL (drop, not force-close)
                trades.append({
                    'pnl': 0.0, 'raw': 0.0, 'reason': 'TIMEOUT',
                    'entry_bar': pos['ebar'], 'exit_bar': bar,
                    'direction': pos['d'], 'bars_held': pos['bh']
                })
                pos = None; last_exit = bar; continue

        # Entry logic
        if pos is None and bar - last_exit >= MIN_BARS_BETWEEN:
            if (math.isnan(ch_hi[bar]) or math.isnan(ch_lo[bar]) or
                    math.isnan(atr[bar]) or atr[bar] <= 0):
                continue

            direction = None
            # Mean-reversion: OPPOSITE of breakout
            if cc[bar] < ch_lo[bar]:
                direction = 'LONG'   # buy the dip
            elif cc[bar] > ch_hi[bar]:
                direction = 'SHORT'  # sell the rally

            if direction is None:
                continue

            # Body filter: require reversal body
            rng = hh[bar] - ll[bar]
            if rng <= 0:
                continue
            body = cc[bar] - oo[bar]
            if body_filter:
                if abs(body) / rng < body_min_ratio:
                    continue
                # Reversal body: LONG needs bullish (close > open)
                if direction == 'LONG' and body <= 0:
                    continue
                if direction == 'SHORT' and body >= 0:
                    continue

            entry_price = oo[bar + 1]
            if bar + 1 >= nn:
                continue

            # Compute SL and TP
            sl_dist = sl_atr_k * atr[bar]
            tp_dist = tp_atr_k * atr[bar]
            if direction == 'LONG':
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

            # Validate SL distance (same range as C1)
            sl_pct = sl_dist / entry_price * 100
            if sl_pct < 0.15 or sl_pct > 3.0:
                continue

            pos = {
                'd': direction, 'ep': entry_price,
                'sl': sl_price, 'tp': tp_price,
                'bh': 0, 'ebar': bar + 1
            }

    return trades


# ═══════════════════════════════════════════════════════════════
# Strategy 3: Regime-Adaptive — ADX switch
# ═══════════════════════════════════════════════════════════════
def bt_regime_adaptive(oo, hh, ll, cc, atr, ch_hi, ch_lo, sw_lo, sw_hi,
                       adx, adx_threshold=25.0,
                       mr_sl_k=2.0, mr_tp_k=2.0,
                       start=WARMUP, end=None):
    """
    Use breakout when ADX >= threshold (trending), mean-reversion when ADX < threshold.
    """
    nn = len(cc) if end is None else end
    sig = C1BreakoutSignal(sig_cfg)
    trades = []
    pos = None
    last_exit = start - MIN_BARS_BETWEEN - 1
    body_min_ratio = 0.4

    for bar in range(start, nn - 1):
        # Exit logic
        if pos is not None:
            pos['bh'] += 1

            if pos['mode'] == 'breakout':
                # Breakout exit (production C1)
                if pos['d'] == 'LONG':
                    pos['bp'] = max(pos['bp'], hh[bar])
                else:
                    pos['bp'] = min(pos['bp'], ll[bar])
                ex = sig.check_exit(pos['d'], pos['ep'], pos['bp'],
                                    hh[bar], ll[bar], cc[bar],
                                    pos['sl'], atr[bar], pos['bh'])
                if ex:
                    if pos['d'] == 'LONG':
                        raw = (ex['exit_price'] / pos['ep'] - 1) * 100
                    else:
                        raw = (1 - ex['exit_price'] / pos['ep']) * 100
                    if ex['reason'] == 'TIMEOUT':
                        trades.append({
                            'pnl': 0.0, 'raw': 0.0, 'reason': 'TIMEOUT',
                            'entry_bar': pos['ebar'], 'exit_bar': bar,
                            'direction': pos['d'], 'bars_held': pos['bh'],
                            'mode': pos['mode']
                        })
                    else:
                        net = raw - FEE
                        trades.append({
                            'pnl': net, 'raw': raw, 'reason': ex['reason'],
                            'entry_bar': pos['ebar'], 'exit_bar': bar,
                            'direction': pos['d'], 'bars_held': pos['bh'],
                            'mode': pos['mode']
                        })
                    pos = None; last_exit = bar; continue
            else:
                # Mean-reversion exit (fixed SL/TP)
                if pos['d'] == 'LONG' and ll[bar] <= pos['sl']:
                    raw = (pos['sl'] / pos['ep'] - 1) * 100
                    trades.append({
                        'pnl': raw - FEE, 'raw': raw, 'reason': 'SL',
                        'entry_bar': pos['ebar'], 'exit_bar': bar,
                        'direction': pos['d'], 'bars_held': pos['bh'],
                        'mode': pos['mode']
                    })
                    pos = None; last_exit = bar; continue
                elif pos['d'] == 'SHORT' and hh[bar] >= pos['sl']:
                    raw = (1 - pos['sl'] / pos['ep']) * 100
                    trades.append({
                        'pnl': raw - FEE, 'raw': raw, 'reason': 'SL',
                        'entry_bar': pos['ebar'], 'exit_bar': bar,
                        'direction': pos['d'], 'bars_held': pos['bh'],
                        'mode': pos['mode']
                    })
                    pos = None; last_exit = bar; continue
                if pos['d'] == 'LONG' and hh[bar] >= pos['tp']:
                    raw = (pos['tp'] / pos['ep'] - 1) * 100
                    trades.append({
                        'pnl': raw - FEE, 'raw': raw, 'reason': 'TP',
                        'entry_bar': pos['ebar'], 'exit_bar': bar,
                        'direction': pos['d'], 'bars_held': pos['bh'],
                        'mode': pos['mode']
                    })
                    pos = None; last_exit = bar; continue
                elif pos['d'] == 'SHORT' and ll[bar] <= pos['tp']:
                    raw = (1 - pos['tp'] / pos['ep']) * 100
                    trades.append({
                        'pnl': raw - FEE, 'raw': raw, 'reason': 'TP',
                        'entry_bar': pos['ebar'], 'exit_bar': bar,
                        'direction': pos['d'], 'bars_held': pos['bh'],
                        'mode': pos['mode']
                    })
                    pos = None; last_exit = bar; continue
                if pos['bh'] >= 192:
                    trades.append({
                        'pnl': 0.0, 'raw': 0.0, 'reason': 'TIMEOUT',
                        'entry_bar': pos['ebar'], 'exit_bar': bar,
                        'direction': pos['d'], 'bars_held': pos['bh'],
                        'mode': pos['mode']
                    })
                    pos = None; last_exit = bar; continue

        # Entry logic
        if pos is None and bar - last_exit >= MIN_BARS_BETWEEN:
            if (math.isnan(ch_hi[bar]) or math.isnan(ch_lo[bar]) or
                    math.isnan(atr[bar]) or atr[bar] <= 0):
                continue

            cur_adx = adx[bar] if not math.isnan(adx[bar]) else 0.0

            if cur_adx >= adx_threshold:
                # Trending regime -> breakout
                if math.isnan(sw_lo[bar]) or math.isnan(sw_hi[bar]):
                    continue
                e = sig.check_entry(oo[bar], hh[bar], ll[bar], cc[bar],
                                    ch_hi[bar], ch_lo[bar], atr[bar],
                                    sw_lo[bar], sw_hi[bar])
                if e and bar + 1 < nn:
                    pos = {
                        'd': e['direction'], 'ep': oo[bar + 1],
                        'sl': e['sl_price'], 'bp': oo[bar + 1],
                        'bh': 0, 'ebar': bar + 1, 'mode': 'breakout'
                    }
            else:
                # Ranging regime -> mean-reversion
                direction = None
                if cc[bar] < ch_lo[bar]:
                    direction = 'LONG'
                elif cc[bar] > ch_hi[bar]:
                    direction = 'SHORT'

                if direction is None:
                    continue

                # No body direction filter for MR in regime mode
                # (body filter kills all MR trades — see sweep results)
                entry_price = oo[bar + 1]
                if bar + 1 >= nn:
                    continue

                sl_dist = mr_sl_k * atr[bar]
                tp_dist = mr_tp_k * atr[bar]
                sl_pct = sl_dist / entry_price * 100
                if sl_pct < 0.15 or sl_pct > 3.0:
                    continue

                if direction == 'LONG':
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                pos = {
                    'd': direction, 'ep': entry_price,
                    'sl': sl_price, 'tp': tp_price,
                    'bh': 0, 'ebar': bar + 1, 'mode': 'mr'
                }

    return trades


# ═══════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════
def compute_metrics(trades, label="", total_days=None):
    """Compute metrics from trade list. Excludes TIMEOUT from PnL."""
    active = [t for t in trades if t['reason'] != 'TIMEOUT']
    if not active:
        return {'label': label, 'trades': 0, 'pnl_pct': 0.0, 'mdd_pct': 0.0,
                'wr_pct': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0, 'rr': 0.0,
                'trades_per_day': 0.0, 'daily_pnl': 0.0, 'exit_reasons': {},
                'days': 0, 'total_trades_incl_timeout': len(trades),
                'note': 'no trades'}

    pnls = [t['pnl'] for t in active]
    n_trades = len(active)
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n_trades * 100 if n_trades else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # MDD (additive)
    eq = [0.0]
    for p in pnls:
        eq.append(eq[-1] + p)
    peak = eq[0]
    mdd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > mdd:
            mdd = dd

    if total_days is None:
        if active:
            total_days = (active[-1]['exit_bar'] - active[0]['entry_bar']) / 96
        else:
            total_days = 1

    tpd = n_trades / max(total_days, 1)
    dpd = total_pnl / max(total_days, 1)

    # Exit reasons
    reasons = {}
    for t in trades:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1

    return {
        'label': label,
        'trades': n_trades,
        'total_trades_incl_timeout': len(trades),
        'pnl_pct': round(total_pnl, 2),
        'mdd_pct': round(mdd, 2),
        'wr_pct': round(wr, 1),
        'avg_win': round(avg_win, 3),
        'avg_loss': round(avg_loss, 3),
        'rr': round(rr, 2),
        'trades_per_day': round(tpd, 2),
        'daily_pnl': round(dpd, 3),
        'exit_reasons': reasons,
        'days': round(total_days, 1)
    }


# ═══════════════════════════════════════════════════════════════
# MC Direction Test (sign randomization)
# ═══════════════════════════════════════════════════════════════
def mc_direction_test(trades, n_sims=999, seed=42):
    """Sign randomization MC test. Returns p-value."""
    active = [t for t in trades if t['reason'] != 'TIMEOUT']
    if len(active) < 10:
        return {'p_value': 1.0, 'n_sims': n_sims, 'n_trades': len(active),
                'note': 'too few trades'}
    pnls = np.array([t['pnl'] for t in active])
    real_pnl = pnls.sum()
    rng = np.random.RandomState(seed)
    count_ge = 0
    for _ in range(n_sims):
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled_pnl = (pnls * signs).sum()
        if shuffled_pnl >= real_pnl:
            count_ge += 1
    p = (count_ge + 1) / (n_sims + 1)
    return {
        'p_value': round(p, 4),
        'n_sims': n_sims,
        'n_trades': len(active),
        'real_pnl': round(real_pnl, 2),
        'disc': 'DISC' if p < 0.01 else 'NOT_DISC'
    }


# ═══════════════════════════════════════════════════════════════
# Walk-Forward (5-fold expanding window)
# ═══════════════════════════════════════════════════════════════
def walk_forward_test(bt_func, n_folds=5, label=""):
    """
    Expanding window WF. IS = [0..ie], OOS = [ie..oe].
    WF formula: ie = int(N * (fi + 1) / (n_folds + 1))
    """
    results = []
    for fi in range(n_folds):
        ie = int(N * (fi + 1) / (n_folds + 1))
        oe = int(N * (fi + 2) / (n_folds + 1)) if fi < n_folds - 1 else N

        is_trades = bt_func(start=WARMUP, end=ie)
        oos_trades = bt_func(start=ie, end=oe)

        is_m = compute_metrics(is_trades, f"IS_fold{fi}")
        oos_m = compute_metrics(oos_trades, f"OOS_fold{fi}")

        oos_active = [t for t in oos_trades if t['reason'] != 'TIMEOUT']
        oos_pnl = sum(t['pnl'] for t in oos_active)

        results.append({
            'fold': fi,
            'is_bars': f"0-{ie}",
            'oos_bars': f"{ie}-{oe}",
            'is_pnl': is_m['pnl_pct'],
            'is_trades': is_m['trades'],
            'oos_pnl': round(oos_pnl, 2),
            'oos_trades': oos_m['trades'],
            'oos_wr': oos_m.get('wr_pct', 0),
            'pass': oos_pnl > 0
        })

    n_pass = sum(1 for r in results if r['pass'])
    total_oos = sum(r['oos_pnl'] for r in results)
    return {
        'label': label,
        'folds': results,
        'pass_count': f"{n_pass}/{n_folds}",
        'total_oos_pnl': round(total_oos, 2),
        'verdict': 'PASS' if n_pass >= 3 else 'FAIL'
    }


# ═══════════════════════════════════════════════════════════════
# Progressive Look-Ahead Test
# ═══════════════════════════════════════════════════════════════
def progressive_test(bt_func, n_cuts=10, label=""):
    """
    Progressive truncation: backtest on first 10%, 20%, ..., 100%.
    If PnL changes drastically with more data = potential bias.
    """
    results = []
    for cut in range(1, n_cuts + 1):
        end_bar = int(N * cut / n_cuts)
        if end_bar < WARMUP + 50:
            continue
        trades = bt_func(start=WARMUP, end=end_bar)
        m = compute_metrics(trades, f"cut_{cut}")
        results.append({
            'cut': f"{cut * 10}%",
            'bars': end_bar,
            'pnl': m['pnl_pct'],
            'trades': m['trades'],
            'wr': m.get('wr_pct', 0)
        })
    return {'label': label, 'cuts': results}


# ═══════════════════════════════════════════════════════════════
# Run S1: Breakout (baseline)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("S1: C1 BREAKOUT (BASELINE)")
print("=" * 70)

s1_trades = bt_breakout(o, h, l, c, atr14, ch_h, ch_l, sw_l, sw_h)
s1_metrics = compute_metrics(s1_trades, "S1_Breakout", days)
print(f"  Trades: {s1_metrics['trades']}, PnL: {s1_metrics['pnl_pct']}%, "
      f"WR: {s1_metrics['wr_pct']}%, R:R: {s1_metrics['rr']}, MDD: {s1_metrics['mdd_pct']}%")
print(f"  Daily: {s1_metrics['daily_pnl']}%, Trades/day: {s1_metrics['trades_per_day']}")
print(f"  Exits: {s1_metrics['exit_reasons']}")


# ═══════════════════════════════════════════════════════════════
# Run S2: Mean-Reversion — sweep SL/TP parameters
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("S2: MEAN-REVERSION — PARAMETER SWEEP")
print("=" * 70)

mr_sweep_results = []
best_mr = None
best_mr_pnl = -float('inf')

for sl_k in [1.5, 2.0, 2.5, 3.0]:
    for tp_k in [1.0, 1.5, 2.0, 3.0]:
        for body_f in [True, False]:
            bf_str = "body" if body_f else "nobody"
            label = f"MR_sl{sl_k}_tp{tp_k}_{bf_str}"
            trades = bt_mean_reversion(o, h, l, c, atr14, ch_h, ch_l,
                                       sl_atr_k=sl_k, tp_atr_k=tp_k,
                                       body_filter=body_f)
            m = compute_metrics(trades, label, days)
            mr_sweep_results.append(m)
            active_count = m['trades']
            pnl = m['pnl_pct']
            print(f"  {label}: trades={active_count}, PnL={pnl:.1f}%, "
                  f"WR={m['wr_pct']:.1f}%, R:R={m['rr']:.2f}")
            if pnl > best_mr_pnl and active_count >= 25:
                best_mr_pnl = pnl
                best_mr = m
                best_mr_params = {'sl_k': sl_k, 'tp_k': tp_k, 'body_filter': body_f}

if best_mr:
    print(f"\n  BEST MR: {best_mr['label']}")
    print(f"    PnL={best_mr['pnl_pct']}%, WR={best_mr['wr_pct']}%, R:R={best_mr['rr']}")
    print(f"    Params: {best_mr_params}")
else:
    print("  No MR configuration met min trade threshold (25)")
    best_mr_params = {'sl_k': 2.0, 'tp_k': 2.0, 'body_filter': True}


# ═══════════════════════════════════════════════════════════════
# Regime Analysis: ADX-based breakdown
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("REGIME ANALYSIS: ADX-Based")
print("=" * 70)

# Compute ADX median for split
valid_adx = [v for v in adx_vals if not math.isnan(v)]
adx_median = sorted(valid_adx)[len(valid_adx) // 2] if valid_adx else 25.0
print(f"  ADX median: {adx_median:.1f}")

# Assign regime to each bar
regime = []  # 'trending' or 'ranging' for each bar
for i in range(N):
    if math.isnan(adx_vals[i]):
        regime.append('unknown')
    elif adx_vals[i] >= adx_median:
        regime.append('trending')
    else:
        regime.append('ranging')

# Split S1 breakout trades by regime at entry bar
s1_trending = [t for t in s1_trades if t['entry_bar'] < N and regime[t['entry_bar']] == 'trending']
s1_ranging = [t for t in s1_trades if t['entry_bar'] < N and regime[t['entry_bar']] == 'ranging']

print(f"\n  S1 Breakout in TRENDING regime ({len(s1_trending)} trades):")
s1_trend_m = compute_metrics(s1_trending, "S1_trending", days / 2)
print(f"    PnL={s1_trend_m['pnl_pct']}%, WR={s1_trend_m['wr_pct']}%, R:R={s1_trend_m['rr']}")

print(f"  S1 Breakout in RANGING regime ({len(s1_ranging)} trades):")
s1_range_m = compute_metrics(s1_ranging, "S1_ranging", days / 2)
print(f"    PnL={s1_range_m['pnl_pct']}%, WR={s1_range_m['wr_pct']}%, R:R={s1_range_m['rr']}")

# Best MR by regime
mr_best_sl = best_mr_params['sl_k']
mr_best_tp = best_mr_params['tp_k']
mr_best_body = best_mr_params['body_filter']
s2_all = bt_mean_reversion(o, h, l, c, atr14, ch_h, ch_l,
                           sl_atr_k=mr_best_sl, tp_atr_k=mr_best_tp,
                           body_filter=mr_best_body)
s2_trending = [t for t in s2_all if t['entry_bar'] < N and regime[t['entry_bar']] == 'trending']
s2_ranging = [t for t in s2_all if t['entry_bar'] < N and regime[t['entry_bar']] == 'ranging']

print(f"\n  S2 Mean-Rev in TRENDING regime ({len(s2_trending)} trades):")
s2_trend_m = compute_metrics(s2_trending, "S2_trending", days / 2)
print(f"    PnL={s2_trend_m['pnl_pct']}%, WR={s2_trend_m['wr_pct']}%, R:R={s2_trend_m['rr']}")

print(f"  S2 Mean-Rev in RANGING regime ({len(s2_ranging)} trades):")
s2_range_m = compute_metrics(s2_ranging, "S2_ranging", days / 2)
print(f"    PnL={s2_range_m['pnl_pct']}%, WR={s2_range_m['wr_pct']}%, R:R={s2_range_m['rr']}")


# ═══════════════════════════════════════════════════════════════
# S3: Regime-Adaptive — sweep ADX thresholds
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("S3: REGIME-ADAPTIVE — ADX THRESHOLD SWEEP")
print("=" * 70)

regime_sweep_results = []
best_regime = None
best_regime_pnl = -float('inf')

for adx_th in [15, 20, 25, 30, 35, 40]:
    trades = bt_regime_adaptive(o, h, l, c, atr14, ch_h, ch_l, sw_l, sw_h,
                                adx_vals, adx_threshold=adx_th,
                                mr_sl_k=mr_best_sl, mr_tp_k=mr_best_tp)
    m = compute_metrics(trades, f"Regime_ADX{adx_th}", days)
    regime_sweep_results.append(m)

    bo_count = sum(1 for t in trades if t.get('mode') == 'breakout' and t['reason'] != 'TIMEOUT')
    mr_count = sum(1 for t in trades if t.get('mode') == 'mr' and t['reason'] != 'TIMEOUT')
    print(f"  ADX_th={adx_th}: trades={m['trades']} (BO:{bo_count} MR:{mr_count}), "
          f"PnL={m['pnl_pct']:.1f}%, WR={m['wr_pct']:.1f}%, R:R={m['rr']:.2f}")

    if m['pnl_pct'] > best_regime_pnl and m['trades'] >= 25:
        best_regime_pnl = m['pnl_pct']
        best_regime = m
        best_regime_adx = adx_th

if best_regime:
    print(f"\n  BEST REGIME: ADX_th={best_regime_adx}")
    print(f"    PnL={best_regime['pnl_pct']}%, WR={best_regime['wr_pct']}%, R:R={best_regime['rr']}")


# ═══════════════════════════════════════════════════════════════
# Additional regime indicators
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ADDITIONAL REGIME INDICATORS")
print("=" * 70)

# Bollinger width regime
valid_bw = [v for v in boll_width if not math.isnan(v)]
bw_median = sorted(valid_bw)[len(valid_bw) // 2] if valid_bw else 2.0
print(f"\n  Bollinger Width median: {bw_median:.2f}%")

bw_regime = []
for i in range(N):
    if math.isnan(boll_width[i]):
        bw_regime.append('unknown')
    elif boll_width[i] >= bw_median:
        bw_regime.append('wide')  # trending
    else:
        bw_regime.append('narrow')  # ranging

s1_bw_wide = [t for t in s1_trades if t['entry_bar'] < N and bw_regime[t['entry_bar']] == 'wide']
s1_bw_narrow = [t for t in s1_trades if t['entry_bar'] < N and bw_regime[t['entry_bar']] == 'narrow']

print(f"  S1 Breakout in WIDE Boll ({len(s1_bw_wide)} trades):")
m_w = compute_metrics(s1_bw_wide, "S1_BW_wide", days / 2)
print(f"    PnL={m_w['pnl_pct']}%, WR={m_w['wr_pct']}%, R:R={m_w['rr']}")

print(f"  S1 Breakout in NARROW Boll ({len(s1_bw_narrow)} trades):")
m_n = compute_metrics(s1_bw_narrow, "S1_BW_narrow", days / 2)
print(f"    PnL={m_n['pnl_pct']}%, WR={m_n['wr_pct']}%, R:R={m_n['rr']}")

# Channel width / ATR ratio regime
ch_atr_ratio = [float('nan')] * N
for i in range(N):
    if not math.isnan(ch_h[i]) and not math.isnan(ch_l[i]) and not math.isnan(atr14[i]) and atr14[i] > 0:
        ch_atr_ratio[i] = (ch_h[i] - ch_l[i]) / atr14[i]

valid_car = [v for v in ch_atr_ratio if not math.isnan(v)]
car_median = sorted(valid_car)[len(valid_car) // 2] if valid_car else 3.0
print(f"\n  Channel/ATR ratio median: {car_median:.2f}")

car_regime = []
for i in range(N):
    if math.isnan(ch_atr_ratio[i]):
        car_regime.append('unknown')
    elif ch_atr_ratio[i] >= car_median:
        car_regime.append('wide')
    else:
        car_regime.append('narrow')

s1_car_wide = [t for t in s1_trades if t['entry_bar'] < N and car_regime[t['entry_bar']] == 'wide']
s1_car_narrow = [t for t in s1_trades if t['entry_bar'] < N and car_regime[t['entry_bar']] == 'narrow']

print(f"  S1 Breakout in WIDE Channel/ATR ({len(s1_car_wide)} trades):")
m_cw = compute_metrics(s1_car_wide, "S1_CAR_wide", days / 2)
print(f"    PnL={m_cw['pnl_pct']}%, WR={m_cw['wr_pct']}%, R:R={m_cw['rr']}")

print(f"  S1 Breakout in NARROW Channel/ATR ({len(s1_car_narrow)} trades):")
m_cn = compute_metrics(s1_car_narrow, "S1_CAR_narrow", days / 2)
print(f"    PnL={m_cn['pnl_pct']}%, WR={m_cn['wr_pct']}%, R:R={m_cn['rr']}")


# ═══════════════════════════════════════════════════════════════
# Validation: WF, MC, Progressive for all 3 strategies
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

# --- S1 Breakout WF ---
print("\n  S1 Breakout — Walk-Forward 5-fold:")
s1_wf = walk_forward_test(
    lambda start=WARMUP, end=None: bt_breakout(o, h, l, c, atr14, ch_h, ch_l, sw_l, sw_h, start=start, end=end),
    label="S1_Breakout"
)
for f in s1_wf['folds']:
    print(f"    Fold {f['fold']}: IS={f['is_pnl']:.1f}% ({f['is_trades']}t) "
          f"OOS={f['oos_pnl']:.1f}% ({f['oos_trades']}t) {'PASS' if f['pass'] else 'FAIL'}")
print(f"    Verdict: {s1_wf['verdict']} ({s1_wf['pass_count']}), OOS total: {s1_wf['total_oos_pnl']:.1f}%")

# --- S2 Mean-Reversion WF ---
print(f"\n  S2 Mean-Reversion (best params: sl={mr_best_sl}, tp={mr_best_tp}, body={mr_best_body}) — WF:")
s2_wf = walk_forward_test(
    lambda start=WARMUP, end=None: bt_mean_reversion(
        o, h, l, c, atr14, ch_h, ch_l,
        sl_atr_k=mr_best_sl, tp_atr_k=mr_best_tp,
        body_filter=mr_best_body, start=start, end=end),
    label="S2_MeanRev"
)
for f in s2_wf['folds']:
    print(f"    Fold {f['fold']}: IS={f['is_pnl']:.1f}% ({f['is_trades']}t) "
          f"OOS={f['oos_pnl']:.1f}% ({f['oos_trades']}t) {'PASS' if f['pass'] else 'FAIL'}")
print(f"    Verdict: {s2_wf['verdict']} ({s2_wf['pass_count']}), OOS total: {s2_wf['total_oos_pnl']:.1f}%")

# --- S3 Regime-Adaptive WF ---
if best_regime:
    print(f"\n  S3 Regime-Adaptive (ADX_th={best_regime_adx}) — WF:")
    s3_wf = walk_forward_test(
        lambda start=WARMUP, end=None: bt_regime_adaptive(
            o, h, l, c, atr14, ch_h, ch_l, sw_l, sw_h, adx_vals,
            adx_threshold=best_regime_adx,
            mr_sl_k=mr_best_sl, mr_tp_k=mr_best_tp,
            start=start, end=end),
        label="S3_Regime"
    )
    for f in s3_wf['folds']:
        print(f"    Fold {f['fold']}: IS={f['is_pnl']:.1f}% ({f['is_trades']}t) "
              f"OOS={f['oos_pnl']:.1f}% ({f['oos_trades']}t) {'PASS' if f['pass'] else 'FAIL'}")
    print(f"    Verdict: {s3_wf['verdict']} ({s3_wf['pass_count']}), OOS total: {s3_wf['total_oos_pnl']:.1f}%")
else:
    s3_wf = {'verdict': 'SKIP', 'note': 'no viable regime config'}

# --- MC Direction Tests ---
print("\n  MC Direction Tests (999 sims):")
s1_mc = mc_direction_test(s1_trades, n_sims=999, seed=42)
print(f"    S1 Breakout: p={s1_mc['p_value']:.4f} {s1_mc['disc']}")

s2_mc = mc_direction_test(s2_all, n_sims=999, seed=42)
print(f"    S2 Mean-Rev: p={s2_mc['p_value']:.4f} {s2_mc['disc']}")

if best_regime:
    best_regime_trades = bt_regime_adaptive(
        o, h, l, c, atr14, ch_h, ch_l, sw_l, sw_h, adx_vals,
        adx_threshold=best_regime_adx,
        mr_sl_k=mr_best_sl, mr_tp_k=mr_best_tp)
    s3_mc = mc_direction_test(best_regime_trades, n_sims=999, seed=42)
    print(f"    S3 Regime:   p={s3_mc['p_value']:.4f} {s3_mc['disc']}")
else:
    s3_mc = {'p_value': 1.0, 'disc': 'SKIP'}

# --- Progressive Look-Ahead ---
print("\n  Progressive Look-Ahead Tests:")
s1_prog = progressive_test(
    lambda start=WARMUP, end=None: bt_breakout(o, h, l, c, atr14, ch_h, ch_l, sw_l, sw_h, start=start, end=end),
    label="S1_Breakout"
)
print(f"    S1 Breakout:")
for cut in s1_prog['cuts']:
    print(f"      {cut['cut']}: PnL={cut['pnl']:.1f}%, trades={cut['trades']}, WR={cut['wr']:.1f}%")

s2_prog = progressive_test(
    lambda start=WARMUP, end=None: bt_mean_reversion(
        o, h, l, c, atr14, ch_h, ch_l,
        sl_atr_k=mr_best_sl, tp_atr_k=mr_best_tp,
        body_filter=mr_best_body, start=start, end=end),
    label="S2_MeanRev"
)
print(f"    S2 Mean-Rev:")
for cut in s2_prog['cuts']:
    print(f"      {cut['cut']}: PnL={cut['pnl']:.1f}%, trades={cut['trades']}, WR={cut['wr']:.1f}%")


# ═══════════════════════════════════════════════════════════════
# Monthly Breakdown
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MONTHLY BREAKDOWN")
print("=" * 70)


def monthly_breakdown(trades, label=""):
    """Group trades by month of entry."""
    monthly = {}
    for t in trades:
        if t['reason'] == 'TIMEOUT':
            continue
        bar_idx = t['entry_bar']
        if bar_idx < len(timestamps):
            month = timestamps[bar_idx].strftime('%Y-%m')
            if month not in monthly:
                monthly[month] = []
            monthly[month].append(t['pnl'])

    result = []
    for month in sorted(monthly.keys()):
        pnls = monthly[month]
        total = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        result.append({
            'month': month,
            'pnl': round(total, 2),
            'trades': len(pnls),
            'wr': round(wr, 1)
        })
    return result


s1_monthly = monthly_breakdown(s1_trades, "S1")
s2_monthly = monthly_breakdown(s2_all, "S2")

print("\n  S1 Breakout monthly:")
for m in s1_monthly:
    print(f"    {m['month']}: PnL={m['pnl']:+.1f}%, trades={m['trades']}, WR={m['wr']:.0f}%")

print("\n  S2 Mean-Rev monthly:")
for m in s2_monthly:
    print(f"    {m['month']}: PnL={m['pnl']:+.1f}%, trades={m['trades']}, WR={m['wr']:.0f}%")


# ═══════════════════════════════════════════════════════════════
# Correlation Analysis: Are S1 and S2 complementary?
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPLEMENTARITY ANALYSIS")
print("=" * 70)

# Check if S1 and S2 monthly PnLs are negatively correlated
s1_month_dict = {m['month']: m['pnl'] for m in s1_monthly}
s2_month_dict = {m['month']: m['pnl'] for m in s2_monthly}
common_months = sorted(set(s1_month_dict.keys()) & set(s2_month_dict.keys()))

if len(common_months) >= 3:
    s1_vals = [s1_month_dict[m] for m in common_months]
    s2_vals = [s2_month_dict[m] for m in common_months]
    combined = [s1_month_dict.get(m, 0) + s2_month_dict.get(m, 0) for m in common_months]

    # Simple correlation
    s1_arr = np.array(s1_vals)
    s2_arr = np.array(s2_vals)
    if s1_arr.std() > 0 and s2_arr.std() > 0:
        corr = np.corrcoef(s1_arr, s2_arr)[0, 1]
    else:
        corr = 0.0

    combined_total = sum(combined)
    combined_mdd_months = 0
    peak = 0
    max_dd = 0
    eq = 0
    for v in combined:
        eq += v
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    print(f"  Monthly correlation (S1 vs S2): {corr:.3f}")
    print(f"  Combined monthly PnL: {combined_total:.1f}%")
    print(f"  Combined monthly max drawdown: {max_dd:.1f}%")
    if corr < -0.3:
        print("  -> NEGATIVE correlation: strategies are complementary")
    elif corr < 0.3:
        print("  -> LOW correlation: some diversification benefit")
    else:
        print("  -> POSITIVE correlation: limited diversification benefit")
else:
    corr = float('nan')
    print("  Not enough common months for correlation analysis")


# ═══════════════════════════════════════════════════════════════
# Save Results
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'metadata': {
        'script': 'mean_reversion_vs_breakout_research.py',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_bars': N,
        'data_days': round(days, 1),
        'data_range': f"{timestamps[0]} to {timestamps[-1]}",
        'fee_pct': FEE,
        'pnl_mode': 'additive',
    },
    's1_breakout': {
        'metrics': s1_metrics,
        'wf': s1_wf,
        'mc': s1_mc,
        'progressive': s1_prog,
        'monthly': s1_monthly,
        'by_regime_adx': {
            'trending': compute_metrics(s1_trending, "S1_trending"),
            'ranging': compute_metrics(s1_ranging, "S1_ranging"),
        },
        'by_regime_bollinger': {
            'wide': m_w,
            'narrow': m_n,
        },
        'by_regime_channel_atr': {
            'wide': m_cw,
            'narrow': m_cn,
        }
    },
    's2_mean_reversion': {
        'best_params': best_mr_params,
        'best_metrics': best_mr,
        'all_sweep': [{'label': r['label'], 'pnl': r['pnl_pct'], 'trades': r['trades'],
                       'wr': r['wr_pct'], 'rr': r['rr']} for r in mr_sweep_results],
        'wf': s2_wf,
        'mc': s2_mc,
        'progressive': s2_prog,
        'monthly': s2_monthly,
        'by_regime_adx': {
            'trending': compute_metrics(s2_trending, "S2_trending"),
            'ranging': compute_metrics(s2_ranging, "S2_ranging"),
        }
    },
    's3_regime_adaptive': {
        'best_adx_threshold': best_regime_adx if best_regime else None,
        'best_metrics': best_regime,
        'all_sweep': [{'label': r['label'], 'pnl': r['pnl_pct'], 'trades': r['trades'],
                       'wr': r['wr_pct'], 'rr': r['rr']} for r in regime_sweep_results],
        'wf': s3_wf,
        'mc': s3_mc,
    },
    'complementarity': {
        'monthly_correlation': round(corr, 3) if not math.isnan(corr) else None,
        'adx_median': round(adx_median, 1),
        'bollinger_width_median': round(bw_median, 2),
        'channel_atr_ratio_median': round(car_median, 2),
    },
    'conclusions': {}  # Filled after analysis
}

# Save
output_path = 'results/mean_reversion_vs_breakout_research.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"  Saved to {output_path}")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  S1 Breakout:     PnL={s1_metrics['pnl_pct']}%, WR={s1_metrics['wr_pct']}%, "
      f"R:R={s1_metrics['rr']}, WF={s1_wf['verdict']}, MC={s1_mc['disc']}")
if best_mr:
    print(f"  S2 Mean-Rev:     PnL={best_mr['pnl_pct']}%, WR={best_mr['wr_pct']}%, "
          f"R:R={best_mr['rr']}, WF={s2_wf['verdict']}, MC={s2_mc['disc']}")
if best_regime:
    print(f"  S3 Regime:       PnL={best_regime['pnl_pct']}%, WR={best_regime['wr_pct']}%, "
          f"R:R={best_regime['rr']}, WF={s3_wf['verdict']}, MC={s3_mc['disc']}")
print(f"\n  Monthly corr(S1,S2): {corr:.3f}" if not math.isnan(corr) else "")
print("\nDone.")
