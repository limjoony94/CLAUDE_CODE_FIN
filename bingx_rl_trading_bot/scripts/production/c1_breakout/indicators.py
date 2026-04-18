"""
C1 Breakout v2 — Technical Indicators
=======================================
ATR, Channel, Fractal swing points.
All computed on past data only (no look-ahead).
"""

import numpy as np


def compute_atr(highs: list, lows: list, closes: list, period: int = 14) -> list:
    """Wilder-smoothed ATR. Returns list of floats (NaN for warmup)."""
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


def compute_channel(highs: list, lows: list, period: int = 15):
    """N-bar high/low channel (excludes current bar).

    Returns (channel_high, channel_low) lists.
    """
    n = len(highs)
    ch_high = [float('nan')] * n
    ch_low = [float('nan')] * n
    for i in range(period, n):
        ch_high[i] = max(highs[i - period:i])
        ch_low[i] = min(lows[i - period:i])
    return ch_high, ch_low


def compute_fractal_swings(highs: list, lows: list, lookback: int = 10):
    """Causal fractal swing highs/lows — past data only, no look-ahead.

    A bar is a swing low if it's the lowest of the past `lookback` bars
    (including itself). No future bars used.

    Returns (last_swing_low, last_swing_high) lists.
    """
    n = len(highs)
    last_sw_low = [float('nan')] * n
    last_sw_high = [float('nan')] * n

    cur_sl = float('nan')
    cur_sh = float('nan')

    for i in range(lookback, n):
        window_low = lows[i - lookback:i + 1]    # past only
        window_high = highs[i - lookback:i + 1]   # past only
        if lows[i] == min(window_low):
            cur_sl = lows[i]
        if highs[i] == max(window_high):
            cur_sh = highs[i]

        last_sw_low[i] = cur_sl
        last_sw_high[i] = cur_sh

    return last_sw_low, last_sw_high
