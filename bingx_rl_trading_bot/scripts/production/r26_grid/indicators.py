"""R26 Grid Bot — Indicators: ATR + Ranging Filter.

Mirrors R26 backtest exactly (claudedocs/round26_grid_ranging_prereg.md).
"""
import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Average True Range, classical Wilder smoothing approximated as SMA over period.

    R26 BT used SMA(TR, period). For LIVE parity, MUST use SMA same way.
    """
    h = df['high']
    l = df['low']
    c = df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_ranging_filter(df: pd.DataFrame,
                            atr_period: int = 20,
                            lookback_bars: int = 720) -> pd.Series:
    """Ranging regime filter: ATR(period)/close < 30d trailing median.

    Returns pd.Series[bool] aligned with df.
    """
    atr = compute_atr(df, atr_period)
    atr_pct = atr / df['close']
    median = atr_pct.rolling(lookback_bars, min_periods=lookback_bars // 3).median()
    return (atr_pct < median).fillna(False)


def compute_grid_levels(init_mid: float, spacing_pct: float, levels_each_side: int) -> tuple:
    """Compute buy and sell limit price levels around init_mid.

    R26 BT: buy_levels[k] = init_mid * (1 - spacing * (k+1)) for k in 0..levels-1
            sell_levels[k] = init_mid * (1 + spacing * (k+1))

    Returns: (buy_levels: list[float], sell_levels: list[float])
    """
    spacing = spacing_pct / 100
    buys = [init_mid * (1 - spacing * (k + 1)) for k in range(levels_each_side)]
    sells = [init_mid * (1 + spacing * (k + 1)) for k in range(levels_each_side)]
    return buys, sells


def is_ranging_now(df: pd.DataFrame, atr_period: int, lookback_bars: int) -> bool:
    """Check if the LATEST bar in df is in ranging regime.

    Returns: True/False, or False if insufficient warmup data.
    """
    if len(df) < lookback_bars // 3:
        return False
    series = compute_ranging_filter(df, atr_period, lookback_bars)
    if series.empty or pd.isna(series.iloc[-1]):
        return False
    return bool(series.iloc[-1])


def compute_trend_exit_signal(current_price: float, init_mid: float,
                                trend_exit_distance_pct: float, is_ranging: bool) -> bool:
    """R26 trend exit logic: |price - init_mid| > 1.5% AND ranging filter off.

    Returns: True if should force-close all positions.
    """
    if init_mid <= 0:
        return False
    distance_pct = abs(current_price - init_mid) / init_mid * 100
    return distance_pct > trend_exit_distance_pct and not is_ranging
