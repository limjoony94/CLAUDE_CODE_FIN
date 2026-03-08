#!/usr/bin/env python3
"""
Non-Candle Signal Study — Exploratory search for directional edge beyond 3-bar patterns.

Background:
  3-bar candle patterns have NO directional predictive power at R:R=1 (WR=49.1%).
  Their value comes entirely from asymmetric TP/SL (R:R < 1 + high WR).
  This study explores whether non-candle signals carry genuine directional edge.

Signals tested (15 total, each LONG + SHORT):
  Category 1 - Volume:  S1 Spike, S2 DryUp, S3 TrendDivergence
  Category 2 - Volatility: S4 BB Squeeze, S5 ATR Expansion, S6 Vol MeanReversion
  Category 3 - Momentum: S7 RSI Extreme, S8 RSI Divergence, S9 MACD Cross, S10 MomBurst
  Category 4 - MeanReversion: S11 BB Touch, S12 EMA Distance, S13 ConsecCandles
  Category 5 - Composite: S14 Vol+Mom, S15 BB+Vol

Protocol:
  - Entry: signal bar + 1 open
  - Exit: intrabar high/low distance from bar OPEN (same-bar rule)
  - LEVERAGE = 3, FEE = 0.10% x LEVERAGE, slippage 0.02%
  - MAX_BARS = 288 (24h timeout, DROP)
  - ATR scaling (14/576/0.6-1.7) for R:R>1 tests
  - MC: 3 seeds (42, 123, 7), max p < 0.01 for PASS (< 0.05 exploratory)
  - All indicators shift(1) applied (no look-ahead)
  - Neutral window for data selection

Output: results/non_candle_signal_study.json
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle  # NEVER self-implement

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('non_candle_signal_study')

# ============================================================
# Standard Research Protocol Constants
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10          # 0.05% x 2 sides
MAX_BARS = 288           # 24h timeout
MC_SIMS = 10000
MC_SEEDS = [42, 123, 7]
SLIPPAGE_BUFFER = 0.02   # %

# ATR scaling
ATR_PERIOD = 14
ATR_WINDOW = 576
ATR_CLAMP_LO = 0.6
ATR_CLAMP_HI = 1.7

# Quality thresholds (exploratory — relaxed from standard)
MIN_SIGNALS = 50
WR_EXCESS_THRESHOLD = 5.0   # pp at R:R=1
MC_THRESHOLD = 0.05         # exploratory
COST_PER_TRADE = FEE_PCT * LEVERAGE + SLIPPAGE_BUFFER * LEVERAGE  # ~0.36%

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'non_candle_signal_study.json')


# ============================================================
# Neutral Window (from scanner v2.3)
# ============================================================
def find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=288):
    """Find longest window where start_price ~ end_price within tolerance."""
    n = len(closes)
    min_bars = min_days * bars_per_day
    best_len, best_start, best_end = 0, 0, 0

    for i in range(0, n - min_bars, bars_per_day):
        start_price = closes[i]
        lo = start_price * (1 - tol_pct / 100)
        hi = start_price * (1 + tol_pct / 100)
        for j in range(n - 1, i + min_bars - 1, -bars_per_day):
            if lo <= closes[j] <= hi:
                length = j - i
                if length > best_len:
                    best_len = length
                    best_start, best_end = i, j
                break

    if best_len == 0:
        return None
    return best_start, best_end


# ============================================================
# ATR Ratio (from scanner v2.3)
# ============================================================
def compute_atr_ratio(highs, lows, closes, atr_period=ATR_PERIOD, window=ATR_WINDOW):
    """ATR / rolling_median(ATR) ratio time series."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period - 1] = tr[:atr_period].mean()
        for i in range(atr_period, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio, atr


# ============================================================
# Backtest Engine (from scanner — protocol compliant)
# ============================================================
def bt_signals(signal_bars, direction, tp_pct, sl_pct,
               opens, highs, lows, n_bars,
               atr_ratio=None, use_atr=False):
    """
    Backtest given signal bars with TP/SL.

    Entry: next bar open after signal.
    Exit: intrabar high/low (distance from bar OPEN for same-bar).
    Timeout trades DROPPED.
    ATR scaling optional.
    """
    fee = FEE_PCT * LEVERAGE
    is_long = (direction == 'LONG')
    trades = []

    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1

        # ATR scaling
        if use_atr and atr_ratio is not None and idx < len(atr_ratio) and not np.isnan(atr_ratio[idx]):
            r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[idx]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE_BUFFER
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

        if is_long:
            tpp = entry * (1 + eff_tp / 100)
            slp = entry * (1 - eff_sl / 100)
        else:
            tpp = entry * (1 - eff_tp / 100)
            slp = entry * (1 + eff_sl / 100)

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if is_long:
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (eff_tp if dist_tp <= dist_sl else -eff_sl) * LEVERAGE - fee
                trades.append(pnl)
                break
            elif ht:
                trades.append(eff_tp * LEVERAGE - fee)
                break
            elif hs:
                trades.append(-eff_sl * LEVERAGE - fee)
                break
        # Timeout -> DROP

    return trades


def mc_test(pnls, n_sims=MC_SIMS):
    """Multi-seed sign randomization MC. Returns max p-value (conservative)."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(n_sims, len(pnls_arr)))
        sim_sums = (signs * pnls_arr).sum(axis=1)
        p = np.mean(sim_sums >= actual)
        p_vals.append(p)
    return max(p_vals)


# ============================================================
# MFE / MAE Computation
# ============================================================
def compute_mfe_mae(signal_bars, direction, opens, highs, lows, n_bars):
    """Compute MFE and MAE for each signal within MAX_BARS window."""
    is_long = (direction == 'LONG')
    mfe_list = []
    mae_list = []

    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue

        best_favor = 0.0
        worst_against = 0.0

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if is_long:
                favor = (highs[j] - entry) / entry * 100
                against = (entry - lows[j]) / entry * 100
            else:
                favor = (entry - lows[j]) / entry * 100
                against = (highs[j] - entry) / entry * 100

            best_favor = max(best_favor, favor)
            worst_against = max(worst_against, against)

        mfe_list.append(best_favor)
        mae_list.append(worst_against)

    return np.array(mfe_list), np.array(mae_list)


# ============================================================
# Indicator Computations (all shift(1) applied = no look-ahead)
# ============================================================
def compute_indicators(df):
    """Compute all technical indicators needed for the 15 signals.

    All indicators are shifted by 1 to prevent look-ahead bias.
    The value at bar i reflects information available up to bar i-1.
    """
    n = len(df)
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values

    ind = {}

    # --- Volume indicators ---
    vol_ema20 = pd.Series(volumes).ewm(span=20, adjust=False).mean().values
    ind['vol_ema20'] = np.roll(vol_ema20, 1)
    ind['vol_ema20'][:1] = np.nan

    # Candle direction (1 = bullish, -1 = bearish)
    candle_dir = np.where(closes > opens, 1, np.where(closes < opens, -1, 0))
    ind['candle_dir'] = np.roll(candle_dir, 1)
    ind['candle_dir'][:1] = 0

    # 3-bar price direction (for S2, S3)
    price_chg_3 = pd.Series(closes).diff(3).values
    ind['price_dir_3'] = np.roll(np.sign(price_chg_3), 1)
    ind['price_dir_3'][:4] = 0

    # 3-bar volume direction
    vol_chg_3 = pd.Series(volumes).diff(3).values
    ind['vol_dir_3'] = np.roll(np.sign(vol_chg_3), 1)
    ind['vol_dir_3'][:4] = 0

    # --- Volatility indicators ---
    # Bollinger Bands (20, 2)
    bb_sma = pd.Series(closes).rolling(20).mean().values
    bb_std = pd.Series(closes).rolling(20).std().values
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_sma * 100  # as % of sma

    ind['bb_sma'] = np.roll(bb_sma, 1)
    ind['bb_upper'] = np.roll(bb_upper, 1)
    ind['bb_lower'] = np.roll(bb_lower, 1)
    ind['bb_width'] = np.roll(bb_width, 1)
    ind['bb_sma'][:1] = np.nan
    ind['bb_upper'][:1] = np.nan
    ind['bb_lower'][:1] = np.nan
    ind['bb_width'][:1] = np.nan

    # BB width percentile (lookback 100)
    bb_w_pct = pd.Series(bb_width).rolling(100).rank(pct=True).values
    ind['bb_width_pct'] = np.roll(bb_w_pct, 1)
    ind['bb_width_pct'][:1] = np.nan

    # ATR(14) — Wilder smoothing
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr14 = np.full(n, np.nan)
    if n >= 14:
        atr14[13] = tr[:14].mean()
        for i in range(14, n):
            atr14[i] = (atr14[i - 1] * 13 + tr[i]) / 14

    atr_ema20 = pd.Series(atr14).ewm(span=20, adjust=False).mean().values
    atr_median50 = pd.Series(atr14).rolling(50, min_periods=50).median().values

    ind['atr14'] = np.roll(atr14, 1)
    ind['atr_ema20'] = np.roll(atr_ema20, 1)
    ind['atr_median50'] = np.roll(atr_median50, 1)
    ind['atr14'][:1] = np.nan
    ind['atr_ema20'][:1] = np.nan
    ind['atr_median50'][:1] = np.nan

    # --- Momentum indicators ---
    # RSI(14)
    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    ind['rsi14'] = np.roll(rsi.values, 1)
    ind['rsi14'][:1] = np.nan

    # For RSI divergence: price lows/highs and RSI lows over 3-bar windows
    # We compute rolling min/max over 5 bars for divergence detection
    price_low_5 = pd.Series(lows).rolling(5).min().values
    rsi_low_5 = rsi.rolling(5).min().values
    price_high_5 = pd.Series(highs).rolling(5).max().values
    rsi_high_5 = rsi.rolling(5).max().values

    ind['price_low_5'] = np.roll(price_low_5, 1)
    ind['rsi_low_5'] = np.roll(rsi_low_5, 1)
    ind['price_high_5'] = np.roll(price_high_5, 1)
    ind['rsi_high_5'] = np.roll(rsi_high_5, 1)
    ind['price_low_5'][:1] = np.nan
    ind['rsi_low_5'][:1] = np.nan
    ind['price_high_5'][:1] = np.nan
    ind['rsi_high_5'][:1] = np.nan

    # Previous 5-bar window (shifted another 5) for divergence comparison
    price_low_5_prev = np.roll(price_low_5, 5)
    rsi_low_5_prev = np.roll(rsi_low_5, 5)
    price_high_5_prev = np.roll(price_high_5, 5)
    rsi_high_5_prev = np.roll(rsi_high_5, 5)

    ind['price_low_5_prev'] = np.roll(price_low_5_prev, 1)
    ind['rsi_low_5_prev'] = np.roll(rsi_low_5_prev, 1)
    ind['price_high_5_prev'] = np.roll(price_high_5_prev, 1)
    ind['rsi_high_5_prev'] = np.roll(rsi_high_5_prev, 1)
    ind['price_low_5_prev'][:11] = np.nan
    ind['rsi_low_5_prev'][:11] = np.nan
    ind['price_high_5_prev'][:11] = np.nan
    ind['rsi_high_5_prev'][:11] = np.nan

    # MACD (12, 26, 9)
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values

    ind['macd_line'] = np.roll(macd_line, 1)
    ind['macd_signal'] = np.roll(macd_signal, 1)
    ind['macd_line_prev'] = np.roll(macd_line, 2)
    ind['macd_signal_prev'] = np.roll(macd_signal, 2)
    ind['macd_line'][:1] = np.nan
    ind['macd_signal'][:1] = np.nan
    ind['macd_line_prev'][:2] = np.nan
    ind['macd_signal_prev'][:2] = np.nan

    # ROC(5) = (close - close[5]) / close[5] * 100
    roc5 = pd.Series(closes).pct_change(5).values * 100
    ind['roc5'] = np.roll(roc5, 1)
    ind['roc5'][:6] = np.nan

    # --- Mean Reversion indicators ---
    # EMA(20)
    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
    ind['ema20'] = np.roll(ema20, 1)
    ind['ema20'][:1] = np.nan

    # Distance from EMA20 in ATR units
    dist_ema = np.full(n, np.nan)
    valid = ~np.isnan(atr14) & (atr14 > 0)
    dist_ema[valid] = (closes[valid] - ema20[valid]) / atr14[valid]
    ind['dist_ema_atr'] = np.roll(dist_ema, 1)
    ind['dist_ema_atr'][:1] = np.nan

    # Consecutive same-direction candles
    consec = np.zeros(n, dtype=int)
    for i in range(1, n):
        d = candle_dir[i]
        if d != 0 and d == candle_dir[i - 1]:
            consec[i] = consec[i - 1] + 1
        elif d != 0:
            consec[i] = 1
        else:
            consec[i] = 0
    ind['consec_dir'] = np.roll(consec, 1)
    ind['consec_dir'][:1] = 0
    ind['consec_candle_dir'] = np.roll(candle_dir, 1)
    ind['consec_candle_dir'][:1] = 0

    # Store raw (unshifted) for signal generation reference
    ind['volume_raw'] = volumes
    ind['close_raw'] = closes
    ind['candle_dir_raw'] = candle_dir

    return ind


# ============================================================
# Signal Generators
# ============================================================
def generate_signals(df, ind, n):
    """Generate all 15 signal types. Returns dict of signal_name -> {LONG: [bars], SHORT: [bars]}."""
    volumes = ind['volume_raw']
    closes = ind['close_raw']

    # Use shifted indicators for signal decisions (look-ahead free)
    signals = {}

    # Helper: safe check
    def ok(val):
        return not (val is None or (isinstance(val, float) and np.isnan(val)))

    # S1: Volume Spike — vol > 2x EMA(vol,20), direction from candle
    long_bars, short_bars = [], []
    for i in range(1, n):
        v = volumes[i]
        ve = ind['vol_ema20'][i]
        cd = ind['candle_dir'][i]  # shifted: reflects bar i-1's direction
        if ok(ve) and ve > 0 and v > 2 * ve:
            # Use current bar's candle direction for direction assignment
            cur_dir = 1 if closes[i] > df['open'].values[i] else (-1 if closes[i] < df['open'].values[i] else 0)
            if cur_dir == 1:
                long_bars.append(i)
            elif cur_dir == -1:
                short_bars.append(i)
    signals['S01_vol_spike'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S2: Volume Dry-Up — vol < 0.5x EMA(vol,20), mean reversion (reverse 3-bar direction)
    long_bars, short_bars = [], []
    for i in range(1, n):
        v = volumes[i]
        ve = ind['vol_ema20'][i]
        pd3 = ind['price_dir_3'][i]
        if ok(ve) and ve > 0 and v < 0.5 * ve:
            if pd3 > 0:      # price was going up -> SHORT (mean reversion)
                short_bars.append(i)
            elif pd3 < 0:    # price was going down -> LONG (mean reversion)
                long_bars.append(i)
    signals['S02_vol_dryup'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S3: Volume Trend Divergence — price up + vol down = SHORT, price down + vol down = LONG
    long_bars, short_bars = [], []
    for i in range(1, n):
        pd3 = ind['price_dir_3'][i]
        vd3 = ind['vol_dir_3'][i]
        if pd3 > 0 and vd3 < 0:   # price up + vol down = bearish divergence
            short_bars.append(i)
        elif pd3 < 0 and vd3 < 0:  # price down + vol down = bullish divergence
            long_bars.append(i)
    signals['S03_vol_trend_div'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S4: Bollinger Squeeze -> Breakout — BB width < p10, direction = current bar
    long_bars, short_bars = [], []
    for i in range(1, n):
        bwp = ind['bb_width_pct'][i]
        if ok(bwp) and bwp < 0.10:  # bottom 10th percentile
            cur_dir = 1 if closes[i] > df['open'].values[i] else (-1 if closes[i] < df['open'].values[i] else 0)
            if cur_dir == 1:
                long_bars.append(i)
            elif cur_dir == -1:
                short_bars.append(i)
    signals['S04_bb_squeeze'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S5: ATR Expansion — ATR(14) > 1.5x EMA(ATR,20), direction = expansion bar direction
    long_bars, short_bars = [], []
    for i in range(1, n):
        a = ind['atr14'][i]
        ae = ind['atr_ema20'][i]
        if ok(a) and ok(ae) and ae > 0 and a > 1.5 * ae:
            cur_dir = 1 if closes[i] > df['open'].values[i] else (-1 if closes[i] < df['open'].values[i] else 0)
            if cur_dir == 1:
                long_bars.append(i)
            elif cur_dir == -1:
                short_bars.append(i)
    signals['S05_atr_expansion'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S6: Volatility Mean Reversion — ATR > 2x median(ATR,50), counter-direction
    long_bars, short_bars = [], []
    for i in range(1, n):
        a = ind['atr14'][i]
        am = ind['atr_median50'][i]
        if ok(a) and ok(am) and am > 0 and a > 2 * am:
            cur_dir = 1 if closes[i] > df['open'].values[i] else (-1 if closes[i] < df['open'].values[i] else 0)
            # Counter-direction (mean reversion after vol spike)
            if cur_dir == 1:
                short_bars.append(i)
            elif cur_dir == -1:
                long_bars.append(i)
    signals['S06_vol_meanrev'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S7: RSI Extreme — RSI < 30 = LONG, RSI > 70 = SHORT
    long_bars, short_bars = [], []
    for i in range(1, n):
        r = ind['rsi14'][i]
        if ok(r):
            if r < 30:
                long_bars.append(i)
            elif r > 70:
                short_bars.append(i)
    signals['S07_rsi_extreme'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S8: RSI Divergence — price lower low + RSI higher low = LONG (bullish div)
    #                       price higher high + RSI lower high = SHORT (bearish div)
    long_bars, short_bars = [], []
    for i in range(1, n):
        pl = ind['price_low_5'][i]
        rl = ind['rsi_low_5'][i]
        plp = ind['price_low_5_prev'][i]
        rlp = ind['rsi_low_5_prev'][i]
        ph = ind['price_high_5'][i]
        rh = ind['rsi_high_5'][i]
        php = ind['price_high_5_prev'][i]
        rhp = ind['rsi_high_5_prev'][i]

        if ok(pl) and ok(rl) and ok(plp) and ok(rlp):
            if pl < plp and rl > rlp:  # bullish divergence
                long_bars.append(i)
        if ok(ph) and ok(rh) and ok(php) and ok(rhp):
            if ph > php and rh < rhp:  # bearish divergence
                short_bars.append(i)
    signals['S08_rsi_divergence'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S9: MACD Cross — MACD crosses above signal = LONG, below = SHORT
    long_bars, short_bars = [], []
    for i in range(1, n):
        ml = ind['macd_line'][i]
        ms = ind['macd_signal'][i]
        mlp = ind['macd_line_prev'][i]
        msp = ind['macd_signal_prev'][i]
        if ok(ml) and ok(ms) and ok(mlp) and ok(msp):
            if mlp <= msp and ml > ms:   # cross above
                long_bars.append(i)
            elif mlp >= msp and ml < ms:  # cross below
                short_bars.append(i)
    signals['S09_macd_cross'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S10: Momentum Burst — ROC(5) > 1% = LONG continuation, < -1% = SHORT continuation
    long_bars, short_bars = [], []
    for i in range(1, n):
        roc = ind['roc5'][i]
        if ok(roc):
            if roc > 1.0:
                long_bars.append(i)
            elif roc < -1.0:
                short_bars.append(i)
    signals['S10_mom_burst'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S11: Bollinger Band Touch — close < lower BB = LONG, close > upper BB = SHORT
    long_bars, short_bars = [], []
    for i in range(1, n):
        c = closes[i]
        bl = ind['bb_lower'][i]
        bu = ind['bb_upper'][i]
        if ok(bl) and ok(bu):
            if c < bl:
                long_bars.append(i)
            elif c > bu:
                short_bars.append(i)
    signals['S11_bb_touch'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S12: Distance from EMA — dist > 2 ATR = SHORT, dist < -2 ATR = LONG
    long_bars, short_bars = [], []
    for i in range(1, n):
        d = ind['dist_ema_atr'][i]
        if ok(d):
            if d > 2.0:
                short_bars.append(i)
            elif d < -2.0:
                long_bars.append(i)
    signals['S12_ema_distance'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S13: Consecutive Candles — 3+ same direction -> reversal
    long_bars, short_bars = [], []
    for i in range(1, n):
        cc = ind['consec_dir'][i]
        cd = ind['consec_candle_dir'][i]
        if cc >= 3:
            if cd == 1:    # 3+ bullish -> SHORT (reversal)
                short_bars.append(i)
            elif cd == -1: # 3+ bearish -> LONG (reversal)
                long_bars.append(i)
    signals['S13_consec_candles'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S14: Volume + Momentum composite — vol spike + momentum burst (same direction)
    long_bars, short_bars = [], []
    s1_long_set = set(signals['S01_vol_spike']['LONG'])
    s1_short_set = set(signals['S01_vol_spike']['SHORT'])
    s10_long_set = set(signals['S10_mom_burst']['LONG'])
    s10_short_set = set(signals['S10_mom_burst']['SHORT'])
    for i in s1_long_set & s10_long_set:
        long_bars.append(i)
    for i in s1_short_set & s10_short_set:
        short_bars.append(i)
    long_bars.sort()
    short_bars.sort()
    signals['S14_vol_mom'] = {'LONG': long_bars, 'SHORT': short_bars}

    # S15: BB Touch + Volume — BB touch + vol above average
    long_bars, short_bars = [], []
    s11_long_set = set(signals['S11_bb_touch']['LONG'])
    s11_short_set = set(signals['S11_bb_touch']['SHORT'])
    for i in s11_long_set:
        v = volumes[i]
        ve = ind['vol_ema20'][i]
        if ok(ve) and ve > 0 and v > ve:
            long_bars.append(i)
    for i in s11_short_set:
        v = volumes[i]
        ve = ind['vol_ema20'][i]
        if ok(ve) and ve > 0 and v > ve:
            short_bars.append(i)
    long_bars.sort()
    short_bars.sort()
    signals['S15_bb_vol'] = {'LONG': long_bars, 'SHORT': short_bars}

    return signals


# ============================================================
# Analysis Pipeline
# ============================================================
def analyze_signal(name, direction, bars, opens, highs, lows, n_bars, atr_ratio):
    """Full analysis for one signal + direction."""
    n_sig = len(bars)
    result = {
        'n_signals': n_sig,
        'direction': direction,
    }

    if n_sig < MIN_SIGNALS:
        result['skip'] = True
        result['skip_reason'] = f'n_signals={n_sig} < {MIN_SIGNALS}'
        return result

    result['skip'] = False

    # --- MFE/MAE Analysis ---
    mfe, mae = compute_mfe_mae(bars, direction, opens, highs, lows, n_bars)
    result['mfe_median'] = float(np.median(mfe))
    result['mae_median'] = float(np.median(mae))
    result['mfe_p25'] = float(np.percentile(mfe, 25))
    result['mfe_p50'] = float(np.percentile(mfe, 50))
    result['mfe_p75'] = float(np.percentile(mfe, 75))
    result['mae_p25'] = float(np.percentile(mae, 25))
    result['mae_p50'] = float(np.percentile(mae, 50))
    result['mae_p75'] = float(np.percentile(mae, 75))
    mfe_mae_ratio = float(np.median(mfe) / np.median(mae)) if np.median(mae) > 0 else 0.0
    result['mfe_mae_ratio'] = mfe_mae_ratio

    # Directional accuracy: % of trades where MFE > MAE
    pct_mfe_gt_mae = float(np.mean(mfe > mae) * 100)
    result['pct_mfe_gt_mae'] = pct_mfe_gt_mae

    # --- R:R = 1:1 Backtests (fixed TP=SL, no ATR) ---
    rr1_results = {}
    best_rr1_wr = 0
    best_rr1_excess = -100
    best_rr1_sl = 0

    for sl in [1.5, 2.0, 2.5, 3.0]:
        tp = sl  # R:R = 1
        baseline_wr = sl / (tp + sl) * 100  # = 50% for R:R=1
        trades = bt_signals(bars, direction, tp, sl, opens, highs, lows, n_bars,
                            atr_ratio=None, use_atr=False)
        if len(trades) < 10:
            continue
        n_t = len(trades)
        wins = sum(1 for t in trades if t > 0)
        wr = wins / n_t * 100
        wr_excess = wr - baseline_wr
        total_pnl = sum(trades)
        ev = total_pnl / n_t if n_t > 0 else 0

        rr1_results[f'sl_{sl}'] = {
            'trades': n_t,
            'wr': round(wr, 2),
            'baseline_wr': round(baseline_wr, 2),
            'wr_excess': round(wr_excess, 2),
            'total_pnl': round(total_pnl, 3),
            'ev_per_trade': round(ev, 4),
        }

        if wr_excess > best_rr1_excess:
            best_rr1_excess = wr_excess
            best_rr1_wr = wr
            best_rr1_sl = sl

    result['rr1_tests'] = rr1_results
    result['rr1_best_wr'] = round(best_rr1_wr, 2)
    result['rr1_best_wr_excess'] = round(best_rr1_excess, 2)
    result['rr1_best_sl'] = best_rr1_sl

    # --- R:R > 1 Backtests (ATR-scaled) ---
    rr_gt1_results = {}
    best_rr_gt1_excess = -100

    for tp, sl in [(3.0, 2.0), (4.0, 2.0), (2.5, 1.5), (3.0, 1.5)]:
        baseline_wr = sl / (tp + sl) * 100
        trades = bt_signals(bars, direction, tp, sl, opens, highs, lows, n_bars,
                            atr_ratio=atr_ratio, use_atr=True)
        if len(trades) < 10:
            continue
        n_t = len(trades)
        wins = sum(1 for t in trades if t > 0)
        wr = wins / n_t * 100
        wr_excess = wr - baseline_wr
        total_pnl = sum(trades)
        ev = total_pnl / n_t if n_t > 0 else 0

        label = f'tp{tp}_sl{sl}'
        rr_gt1_results[label] = {
            'trades': n_t,
            'wr': round(wr, 2),
            'baseline_wr': round(baseline_wr, 2),
            'wr_excess': round(wr_excess, 2),
            'rr': round(tp / sl, 2),
            'total_pnl': round(total_pnl, 3),
            'ev_per_trade': round(ev, 4),
        }

        if wr_excess > best_rr_gt1_excess:
            best_rr_gt1_excess = wr_excess

    result['rr_gt1_tests'] = rr_gt1_results
    result['rr_gt1_best_wr_excess'] = round(best_rr_gt1_excess, 2)

    # --- Optimal TP/SL from MFE/MAE percentiles ---
    opt_tp = float(np.percentile(mfe, 40))  # Conservative TP (p40 of MFE)
    opt_sl = float(np.percentile(mae, 70))  # Conservative SL (p70 of MAE)
    if opt_tp < 0.3:
        opt_tp = 0.5
    if opt_sl < 1.0:
        opt_sl = 1.0

    opt_rr = round(opt_tp / opt_sl, 3) if opt_sl > 0 else 0
    result['optimal_tp'] = round(opt_tp, 3)
    result['optimal_sl'] = round(opt_sl, 3)
    result['optimal_rr'] = opt_rr

    # Backtest with optimal TP/SL (ATR-scaled)
    opt_trades = bt_signals(bars, direction, opt_tp, opt_sl, opens, highs, lows, n_bars,
                            atr_ratio=atr_ratio, use_atr=True)
    if len(opt_trades) >= 10:
        n_ot = len(opt_trades)
        opt_wins = sum(1 for t in opt_trades if t > 0)
        opt_wr = opt_wins / n_ot * 100
        opt_baseline = opt_sl / (opt_tp + opt_sl) * 100
        opt_total = sum(opt_trades)
        opt_ev = opt_total / n_ot

        result['optimal_bt'] = {
            'trades': n_ot,
            'wr': round(opt_wr, 2),
            'baseline_wr': round(opt_baseline, 2),
            'wr_excess': round(opt_wr - opt_baseline, 2),
            'total_pnl': round(opt_total, 3),
            'ev_per_trade': round(opt_ev, 4),
        }
    else:
        result['optimal_bt'] = {'trades': len(opt_trades), 'skip': True}

    # --- MC Test on best R:R=1 config ---
    if best_rr1_sl > 0:
        mc_trades = bt_signals(bars, direction, best_rr1_sl, best_rr1_sl,
                               opens, highs, lows, n_bars,
                               atr_ratio=None, use_atr=False)
        mc_p = mc_test(mc_trades)
        result['mc_p_rr1'] = round(mc_p, 6)
    else:
        result['mc_p_rr1'] = 1.0

    # --- MC Test on optimal TP/SL ---
    if len(opt_trades) >= 10:
        mc_p_opt = mc_test(opt_trades)
        result['mc_p_optimal'] = round(mc_p_opt, 6)
    else:
        result['mc_p_optimal'] = 1.0

    # --- Pass/Fail ---
    passes_wr = best_rr1_excess >= WR_EXCESS_THRESHOLD
    passes_mc = result['mc_p_rr1'] < MC_THRESHOLD or result['mc_p_optimal'] < MC_THRESHOLD
    ev_ok = False
    if result.get('optimal_bt') and not result['optimal_bt'].get('skip'):
        ev_ok = result['optimal_bt']['ev_per_trade'] > COST_PER_TRADE
    result['pass_wr'] = passes_wr
    result['pass_mc'] = passes_mc
    result['pass_ev'] = ev_ok
    result['pass'] = passes_wr and passes_mc

    return result


# ============================================================
# Compound PnL for ranking
# ============================================================
def compound_pnl(trades):
    """Compute compound PnL from list of per-trade % returns."""
    equity = 1.0
    for t in trades:
        equity *= (1 + t / 100)
    return (equity - 1) * 100


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Non-Candle Signal Study — Exploratory Edge Discovery")
    logger.info("=" * 70)

    # Load data
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    logger.info(f"Loaded {len(df)} bars ({len(df)/288:.0f} days)")

    # Check volume
    if 'volume' not in df.columns or df['volume'].isna().all():
        logger.error("No volume data available. Cannot proceed with volume-based signals.")
        return

    vol_zeros = (df['volume'] == 0).sum()
    logger.info(f"Volume column: {df['volume'].notna().sum()} valid, {vol_zeros} zeros")

    # Neutral window
    closes = df['close'].values
    nw = find_neutral_window(closes)
    if nw:
        start_bar, end_bar = nw
        nw_days = (end_bar - start_bar) / 288
        start_price = closes[start_bar]
        end_price = closes[end_bar]
        pct_change = (end_price - start_price) / start_price * 100
        logger.info(f"Neutral window: bars {start_bar}-{end_bar} ({nw_days:.0f} days), "
                     f"price {start_price:.0f}->{end_price:.0f} ({pct_change:+.2f}%)")
        df = df.iloc[start_bar:end_bar + 1].reset_index(drop=True)
    else:
        logger.warning("No neutral window found, using full data")
        start_bar, end_bar = 0, len(df) - 1
        nw_days = len(df) / 288

    n = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # ATR ratio
    atr_ratio, _ = compute_atr_ratio(highs, lows, closes)
    valid_atr = np.sum(~np.isnan(atr_ratio))
    logger.info(f"ATR ratio computed: {valid_atr} valid bars")

    # Compute indicators
    logger.info("Computing indicators (all shift(1) applied)...")
    ind = compute_indicators(df)

    # Generate signals
    logger.info("Generating signals...")
    signals = generate_signals(df, ind, n)

    # Summary
    logger.info("\n--- Signal Counts ---")
    for name, dirs in sorted(signals.items()):
        nl = len(dirs['LONG'])
        ns = len(dirs['SHORT'])
        logger.info(f"  {name}: LONG={nl}, SHORT={ns}, total={nl+ns}")

    # Analyze each signal
    logger.info("\n" + "=" * 70)
    logger.info("Analyzing signals...")
    logger.info("=" * 70)

    all_results = {}
    for sig_name, dirs in sorted(signals.items()):
        for direction in ['LONG', 'SHORT']:
            bars = dirs[direction]
            key = f"{sig_name}_{direction}"
            logger.info(f"\n--- {key} ({len(bars)} signals) ---")

            result = analyze_signal(sig_name, direction, bars,
                                    opens, highs, lows, n, atr_ratio)

            if result.get('skip'):
                logger.info(f"  SKIP: {result['skip_reason']}")
            else:
                logger.info(f"  MFE/MAE ratio: {result['mfe_mae_ratio']:.3f}")
                logger.info(f"  MFE>MAE: {result['pct_mfe_gt_mae']:.1f}%")
                logger.info(f"  R:R=1 best WR excess: {result['rr1_best_wr_excess']:+.2f}pp "
                             f"(WR={result['rr1_best_wr']:.1f}%, SL={result['rr1_best_sl']}%)")
                logger.info(f"  R:R>1 best WR excess: {result['rr_gt1_best_wr_excess']:+.2f}pp")
                logger.info(f"  Optimal TP/SL: {result['optimal_tp']:.2f}/{result['optimal_sl']:.2f} "
                             f"(R:R={result['optimal_rr']:.3f})")
                if result.get('optimal_bt') and not result['optimal_bt'].get('skip'):
                    obt = result['optimal_bt']
                    logger.info(f"  Optimal BT: WR={obt['wr']:.1f}%, excess={obt['wr_excess']:+.1f}pp, "
                                 f"EV={obt['ev_per_trade']:.4f}%")
                logger.info(f"  MC(R:R=1): p={result['mc_p_rr1']:.4f}, "
                             f"MC(optimal): p={result['mc_p_optimal']:.4f}")
                verdict = "PASS" if result['pass'] else "FAIL"
                logger.info(f"  --> {verdict} (WR:{result['pass_wr']}, MC:{result['pass_mc']}, EV:{result['pass_ev']})")

            all_results[key] = result

    # --- Ranking ---
    logger.info("\n" + "=" * 70)
    logger.info("RANKING — Sorted by WR Excess (R:R=1)")
    logger.info("=" * 70)

    ranking = []
    for key, r in all_results.items():
        if r.get('skip'):
            continue
        ranking.append({
            'signal': key,
            'n_signals': r['n_signals'],
            'mfe_mae_ratio': r['mfe_mae_ratio'],
            'pct_mfe_gt_mae': r['pct_mfe_gt_mae'],
            'rr1_best_wr': r['rr1_best_wr'],
            'rr1_best_wr_excess': r['rr1_best_wr_excess'],
            'rr_gt1_best_wr_excess': r['rr_gt1_best_wr_excess'],
            'optimal_rr': r['optimal_rr'],
            'mc_p_rr1': r['mc_p_rr1'],
            'mc_p_optimal': r['mc_p_optimal'],
            'pass': r['pass'],
        })

    ranking.sort(key=lambda x: x['rr1_best_wr_excess'], reverse=True)

    for i, r in enumerate(ranking):
        tag = "PASS" if r['pass'] else "fail"
        logger.info(f"  {i+1:2d}. [{tag}] {r['signal']:30s} "
                     f"WR_excess={r['rr1_best_wr_excess']:+6.2f}pp  "
                     f"MFE/MAE={r['mfe_mae_ratio']:.3f}  "
                     f"MFE>MAE={r['pct_mfe_gt_mae']:.1f}%  "
                     f"MC={r['mc_p_rr1']:.4f}  "
                     f"n={r['n_signals']}")

    passing = [r for r in ranking if r['pass']]
    logger.info(f"\nPassing signals: {len(passing)}/{len(ranking)}")

    # --- Verdict ---
    verdict = {
        'any_genuine_edge': len(passing) > 0,
        'n_tested': len(ranking),
        'n_passing': len(passing),
        'best_signal': passing[0]['signal'] if passing else None,
        'best_wr_excess': passing[0]['rr1_best_wr_excess'] if passing else None,
    }

    if passing:
        logger.info(f"\nBest signal: {verdict['best_signal']} "
                     f"(WR excess: {verdict['best_wr_excess']:+.2f}pp)")
        verdict['recommendation'] = (
            f"Signal {verdict['best_signal']} shows exploratory edge. "
            f"Needs WF validation before any deployment consideration."
        )
    else:
        logger.info("\nNo signals passed exploratory thresholds.")
        verdict['recommendation'] = (
            "No non-candle signals show genuine directional edge above thresholds. "
            "Current 3-bar pattern approach (directional edge via asymmetric TP/SL) remains valid."
        )

    # --- Save results ---
    output = {
        'study': 'non_candle_signal_study',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': DATA_FILE,
        'data_bars': n,
        'data_days': round(n / 288, 1),
        'neutral_window': {
            'start_bar': int(start_bar),
            'end_bar': int(end_bar),
            'days': round(nw_days, 1),
        },
        'protocol': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'max_bars': MAX_BARS,
            'mc_sims': MC_SIMS,
            'mc_seeds': MC_SEEDS,
            'atr_params': f'a{ATR_PERIOD}/w{ATR_WINDOW}/clamp[{ATR_CLAMP_LO},{ATR_CLAMP_HI}]',
            'min_signals': MIN_SIGNALS,
            'wr_excess_threshold': WR_EXCESS_THRESHOLD,
            'mc_threshold': MC_THRESHOLD,
        },
        'signals': all_results,
        'ranking': ranking,
        'passing_signals': [r['signal'] for r in passing],
        'verdict': verdict,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {OUTPUT_FILE}")

    elapsed = time.time() - t0
    logger.info(f"Total runtime: {elapsed:.1f}s")

    return output


if __name__ == '__main__':
    main()
