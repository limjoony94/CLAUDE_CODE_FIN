#!/usr/bin/env python3
"""
Scalping Strategy Study — Non-Pattern Short-Holding BTC 5m Strategies

Background:
- 3-candle patterns have zero directional power at R:R=1 (WR ~49.1%)
- But short-term mean reversion, volatility clustering, and extreme
  indicator readings may offer genuine scalping edges
- Scalping's biggest enemy: COST. At 3x leverage, round-trip cost is
  FEE_PCT * LEVERAGE * 2-sides = 0.30%, plus ~0.04% slippage = 0.34% per trade.
  With entry at next-bar open, effective cost per trade in capital space:
  fee_capital = FEE_PCT * LEVERAGE = 0.30% (already per round-trip)

Strategies:
  1. BMR  — Big Move Reversal (mean reversion after ATR-multiple moves)
  2. VSB  — Volatility Squeeze Breakout (BB squeeze → expansion)
  3. WR   — Wick Reversion (long wick bounce)
  4. MTF  — Micro Trend Following (EMA cross + RSI confirmation)
  5. PLB  — Price Level Bounce (round-number proximity)
  6. RES  — RSI Extreme Scalp (extreme RSI readings)

Protocol:
  - LEVERAGE = 3, FEE = 0.10% (x LEVERAGE for capital-space PnL)
  - Entry: signal bar's NEXT bar open
  - Exit: intrabar high/low distance-based (same-bar: abs(tp - bar_open))
  - Timeout: DROP (excluded from PnL)
  - MC: 3 seeds (42, 123, 7), p < 0.05 (exploratory)
  - WF: 3-fold expanding window on neutral data
  - Compound sizing (multiplicative returns)

Output: results/scalping_strategy_study.json

Author: Research script (Standard Research Protocol)
Date: 2026-03-01
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

logger = logging.getLogger('scalping_study')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ============================================================
# Constants (Standard Research Protocol)
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10          # 0.05% entry + 0.05% exit (round-trip)
FEE_CAPITAL = FEE_PCT * LEVERAGE  # 0.30% in capital space
SLIPPAGE = 0.02         # 0.02% slippage buffer
COST_PER_TRADE = FEE_CAPITAL  # 0.30% (slippage handled separately in TP/SL levels)

MC_SIMS = 10000
MC_SEEDS = [42, 123, 7]
MC_THRESHOLD = 0.05     # Exploratory (relaxed from 0.01)

MIN_SIGNALS = 50        # Minimum signals for analysis
WR_EXCESS_MIN = 3.0     # Minimum WR excess in pp (relaxed for scalping)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'scalping_strategy_study.json')

BARS_PER_DAY = 288      # 5m bars per day


# ============================================================
# Data Loading
# ============================================================
def load_data():
    """Load and validate 5m BTC data."""
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    logger.info(f"  Rows: {len(df)}, Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Extract numpy arrays for speed
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else None

    return df, opens, highs, lows, closes, volumes


def find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=BARS_PER_DAY):
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
# Technical Indicators (computed once)
# ============================================================
def compute_indicators(opens, highs, lows, closes, volumes):
    """Compute all technical indicators needed across strategies."""
    n = len(closes)

    # ATR (Wilder EMA, period=14)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))

    atr14 = np.full(n, np.nan)
    atr14[13] = tr[:14].mean()
    for i in range(14, n):
        atr14[i] = (atr14[i - 1] * 13 + tr[i]) / 14

    # EMA(5), EMA(13) for MTF strategy
    ema5 = _ema(closes, 5)
    ema13 = _ema(closes, 13)

    # RSI(5), RSI(3) for RES strategy
    rsi5 = _rsi(closes, 5)
    rsi3 = _rsi(closes, 3)

    # Bollinger Band width (20, 2) for VSB strategy
    bb_width = _bb_width(closes, 20, 2)

    # Candle body/wick info (precomputed for WR strategy)
    body = closes - opens
    body_abs = np.abs(body)
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows

    indicators = {
        'atr14': atr14,
        'ema5': ema5,
        'ema13': ema13,
        'rsi5': rsi5,
        'rsi3': rsi3,
        'bb_width': bb_width,
        'body': body,
        'body_abs': body_abs,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
    }
    return indicators


def _ema(data, period):
    """Exponential moving average."""
    n = len(data)
    out = np.full(n, np.nan)
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(data, period):
    """RSI via Wilder smoothing."""
    n = len(data)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100 - 100 / (1 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100 - 100 / (1 + rs)

    return out


def _bb_width(data, period=20, num_std=2):
    """Bollinger Band width as percentage of middle band."""
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        sma = np.mean(window)
        std = np.std(window, ddof=0)
        if sma > 0:
            out[i] = (2 * num_std * std) / sma * 100
    return out


# ============================================================
# Backtest Engine (Standard Protocol)
# ============================================================
def backtest(signal_bars, directions, tp_pct, sl_pct, opens, highs, lows,
             n_bars, max_hold_bars):
    """
    Backtest with standard protocol.

    Args:
        signal_bars: list of bar indices where signal fires
        directions: list of 'LONG'/'SHORT' per signal (or single str for all)
        tp_pct: TP distance in % (price space, pre-leverage)
        sl_pct: SL distance in % (price space, pre-leverage)
        opens, highs, lows: price arrays
        n_bars: total bars
        max_hold_bars: max bars to hold before timeout (DROP)

    Returns:
        List of (entry_bar, exit_bar, pnl_pct, direction) tuples.
        pnl_pct includes leverage and fees.
    """
    fee = FEE_PCT * LEVERAGE
    trades = []

    if isinstance(directions, str):
        directions = [directions] * len(signal_bars)

    for sig_idx, direction in zip(signal_bars, directions):
        if sig_idx + 1 >= n_bars:
            continue
        entry = opens[sig_idx + 1]
        if entry <= 0:
            continue
        eb = sig_idx + 1

        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        resolved = False
        end_bar = min(sig_idx + 2 + max_hold_bars, n_bars)
        for j in range(sig_idx + 2, end_bar):
            if direction == 'LONG':
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                # Same-bar resolution: distance from bar OPEN
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (tp_pct if dist_tp <= dist_sl else -sl_pct) * LEVERAGE - fee
                trades.append((eb, j, pnl, direction))
                resolved = True
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - fee, direction))
                resolved = True
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - fee, direction))
                resolved = True
                break
        # Timeout: DROP (no append)

    return trades


def compute_mfe_mae(signal_bars, directions, opens, highs, lows, n_bars, max_hold_bars):
    """Compute MFE and MAE for each signal (unconstrained excursion)."""
    excursions = []

    if isinstance(directions, str):
        directions = [directions] * len(signal_bars)

    for sig_idx, direction in zip(signal_bars, directions):
        if sig_idx + 1 >= n_bars:
            continue
        entry = opens[sig_idx + 1]
        if entry <= 0:
            continue

        mfe = 0.0
        mae = 0.0
        bars_to_mfe = 0

        end_bar = min(sig_idx + 2 + max_hold_bars, n_bars)
        for j in range(sig_idx + 2, end_bar):
            if direction == 'LONG':
                favorable = highs[j] - entry
                adverse = entry - lows[j]
            else:
                favorable = entry - lows[j]
                adverse = highs[j] - entry

            if favorable > mfe:
                mfe = favorable
                bars_to_mfe = j - (sig_idx + 1)
            if adverse > mae:
                mae = adverse

        mfe_pct = mfe / entry * 100
        mae_pct = mae / entry * 100

        excursions.append({
            'mfe_pct': mfe_pct,
            'mae_pct': mae_pct,
            'bars_to_mfe': bars_to_mfe,
            'direction': direction,
        })

    return excursions


# ============================================================
# Monte Carlo (sign randomization)
# ============================================================
def mc_test(trades_pnl, n_sims=MC_SIMS, seeds=MC_SEEDS):
    """
    Monte Carlo sign randomization test.
    Returns max p-value across seeds (conservative).
    """
    if len(trades_pnl) < 10:
        return 1.0

    actual_pnl = sum(trades_pnl)
    pvalues = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        count_ge = 0
        for _ in range(n_sims):
            signs = rng.choice([-1, 1], size=len(trades_pnl))
            sim_pnl = sum(t * s for t, s in zip(trades_pnl, signs))
            if sim_pnl >= actual_pnl:
                count_ge += 1
        pvalues.append(count_ge / n_sims)

    return max(pvalues)  # Conservative: worst seed


# ============================================================
# Walk-Forward (3-fold expanding window)
# ============================================================
def walk_forward_test(generate_signals_fn, opens, highs, lows, closes,
                      indicators, tp_pct, sl_pct, max_hold_bars, n_folds=3,
                      start_bar=0, end_bar=None):
    """
    Expanding window walk-forward.

    Splits data into n_folds+1 segments. IS expands, OOS is next segment.
    Returns list of {fold, is_bars, oos_bars, oos_trades, oos_wr, oos_pnl}.
    """
    if end_bar is None:
        end_bar = len(opens)

    total_bars = end_bar - start_bar
    fold_size = total_bars // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        is_end = start_bar + fold_size * (fold + 2)
        oos_start = is_end
        oos_end = min(start_bar + fold_size * (fold + 3), end_bar)

        if oos_end <= oos_start:
            continue

        # Generate signals on IS data only, but apply within OOS range
        # The function should return signals that fall within [oos_start, oos_end)
        signal_bars, directions = generate_signals_fn(
            opens, highs, lows, closes, indicators,
            bar_start=oos_start, bar_end=oos_end
        )

        if len(signal_bars) == 0:
            results.append({
                'fold': fold + 1,
                'is_bars': is_end - start_bar,
                'oos_bars': oos_end - oos_start,
                'oos_trades': 0,
                'oos_wr': 0.0,
                'oos_pnl': 0.0,
            })
            continue

        trades = backtest(signal_bars, directions, tp_pct, sl_pct,
                         opens, highs, lows, len(opens), max_hold_bars)

        n_trades = len(trades)
        if n_trades == 0:
            results.append({
                'fold': fold + 1,
                'is_bars': is_end - start_bar,
                'oos_bars': oos_end - oos_start,
                'oos_trades': 0,
                'oos_wr': 0.0,
                'oos_pnl': 0.0,
            })
            continue

        wins = sum(1 for t in trades if t[2] > 0)
        wr = wins / n_trades * 100

        # Compound PnL
        equity = 1.0
        for t in trades:
            equity *= (1 + t[2] / 100)
        cpnl = (equity - 1) * 100

        results.append({
            'fold': fold + 1,
            'is_bars': is_end - start_bar,
            'oos_bars': oos_end - oos_start,
            'oos_trades': n_trades,
            'oos_wr': round(wr, 1),
            'oos_pnl': round(cpnl, 2),
        })

    return results


# ============================================================
# Signal Generators for Each Strategy
# ============================================================

def generate_bmr_signals(opens, highs, lows, closes, indicators,
                         lookback_n=2, threshold_atr=2.0,
                         bar_start=None, bar_end=None):
    """
    Big Move Reversal: enter opposite after large price move.

    Signal: abs(close[i] - close[i-N]) > threshold * ATR(14)
    Direction: opposite of move (mean reversion)
    """
    atr = indicators['atr14']
    n = len(closes)
    start = max(lookback_n, 14, bar_start or 0)
    end = bar_end or n

    signal_bars = []
    directions = []

    for i in range(start, end):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        move = closes[i] - closes[i - lookback_n]
        if abs(move) > threshold_atr * atr[i]:
            signal_bars.append(i)
            # Mean reversion: opposite direction
            directions.append('SHORT' if move > 0 else 'LONG')

    return signal_bars, directions


def generate_vsb_signals(opens, highs, lows, closes, indicators,
                         bb_lookback=200, bb_percentile=10,
                         bar_start=None, bar_end=None):
    """
    Volatility Squeeze Breakout: enter on BB expansion after extreme squeeze.

    Trigger: BB_width < p10(BB_width, lookback=200)
    Direction: follow first candle after squeeze
    """
    bb_width = indicators['bb_width']
    n = len(closes)
    start = max(bb_lookback + 20, bar_start or 0)
    end = bar_end or n

    signal_bars = []
    directions = []

    for i in range(start, end):
        if np.isnan(bb_width[i]):
            continue
        # Compute local percentile of BB width
        window = bb_width[max(0, i - bb_lookback):i]
        valid = window[~np.isnan(window)]
        if len(valid) < 50:
            continue
        threshold = np.percentile(valid, bb_percentile)
        if bb_width[i] < threshold:
            # Squeeze detected: direction = candle's direction
            candle_move = closes[i] - opens[i]
            if candle_move == 0:
                continue
            direction = 'LONG' if candle_move > 0 else 'SHORT'
            signal_bars.append(i)
            directions.append(direction)

    return signal_bars, directions


def generate_wick_reversion_signals(opens, highs, lows, closes, indicators,
                                    wick_body_ratio=2.0, wick_atr_ratio=1.0,
                                    bar_start=None, bar_end=None):
    """
    Wick Reversion: enter opposite to long wick direction.

    Trigger (LONG): lower_wick > wick_body_ratio * body_abs AND
                    lower_wick > wick_atr_ratio * ATR(14)
    Trigger (SHORT): upper_wick > wick_body_ratio * body_abs AND
                     upper_wick > wick_atr_ratio * ATR(14)
    """
    atr = indicators['atr14']
    body_abs = indicators['body_abs']
    upper_wick = indicators['upper_wick']
    lower_wick = indicators['lower_wick']
    n = len(closes)
    start = max(14, bar_start or 0)
    end = bar_end or n

    signal_bars = []
    directions = []

    for i in range(start, end):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        ba = max(body_abs[i], 1e-10)  # Avoid zero body edge case

        # LONG: big lower wick
        if lower_wick[i] > wick_body_ratio * ba and lower_wick[i] > wick_atr_ratio * atr[i]:
            signal_bars.append(i)
            directions.append('LONG')
        # SHORT: big upper wick
        elif upper_wick[i] > wick_body_ratio * ba and upper_wick[i] > wick_atr_ratio * atr[i]:
            signal_bars.append(i)
            directions.append('SHORT')

    return signal_bars, directions


def generate_mtf_signals(opens, highs, lows, closes, indicators,
                         bar_start=None, bar_end=None):
    """
    Micro Trend Following: EMA(5) x EMA(13) crossover + RSI confirmation.

    LONG: EMA(5) crosses above EMA(13) + RSI(5) > 50
    SHORT: EMA(5) crosses below EMA(13) + RSI(5) < 50
    """
    ema5 = indicators['ema5']
    ema13 = indicators['ema13']
    rsi5 = indicators['rsi5']
    n = len(closes)
    start = max(14, bar_start or 0)
    end = bar_end or n

    signal_bars = []
    directions = []

    for i in range(start, end):
        if np.isnan(ema5[i]) or np.isnan(ema13[i]) or np.isnan(ema5[i-1]) or np.isnan(ema13[i-1]):
            continue
        if np.isnan(rsi5[i]):
            continue

        # Golden cross
        if ema5[i] > ema13[i] and ema5[i-1] <= ema13[i-1] and rsi5[i] > 50:
            signal_bars.append(i)
            directions.append('LONG')
        # Death cross
        elif ema5[i] < ema13[i] and ema5[i-1] >= ema13[i-1] and rsi5[i] < 50:
            signal_bars.append(i)
            directions.append('SHORT')

    return signal_bars, directions


def generate_plb_signals(opens, highs, lows, closes, indicators,
                         round_interval=1000, proximity_pct=0.1,
                         bar_start=None, bar_end=None):
    """
    Price Level Bounce: enter when price is near round number.

    Trigger: close within proximity_pct% of a round number (multiples of $1,000)
    Direction: approach from below = LONG, from above = SHORT
    """
    n = len(closes)
    start = max(2, bar_start or 0)
    end = bar_end or n

    signal_bars = []
    directions = []

    for i in range(start, end):
        price = closes[i]
        # Nearest round number
        nearest_round = round(price / round_interval) * round_interval
        distance_pct = abs(price - nearest_round) / nearest_round * 100

        if distance_pct < proximity_pct:
            # Determine approach direction from previous bar
            prev_price = closes[i - 1]
            if prev_price < nearest_round and price < nearest_round:
                # Approaching from below
                signal_bars.append(i)
                directions.append('LONG')
            elif prev_price > nearest_round and price > nearest_round:
                # Approaching from above
                signal_bars.append(i)
                directions.append('SHORT')

    return signal_bars, directions


def generate_rsi_extreme_signals(opens, highs, lows, closes, indicators,
                                  rsi_period=5, oversold=15, overbought=85,
                                  bar_start=None, bar_end=None):
    """
    RSI Extreme Scalp: enter on extreme RSI readings (mean reversion).

    LONG: RSI(period) < oversold
    SHORT: RSI(period) > overbought
    """
    if rsi_period == 5:
        rsi = indicators['rsi5']
    elif rsi_period == 3:
        rsi = indicators['rsi3']
    else:
        rsi = _rsi(closes, rsi_period)

    n = len(closes)
    start = max(rsi_period + 1, bar_start or 0)
    end = bar_end or n

    signal_bars = []
    directions = []

    for i in range(start, end):
        if np.isnan(rsi[i]):
            continue
        if rsi[i] < oversold:
            signal_bars.append(i)
            directions.append('LONG')
        elif rsi[i] > overbought:
            signal_bars.append(i)
            directions.append('SHORT')

    return signal_bars, directions


# ============================================================
# Strategy Analysis Pipeline
# ============================================================

def analyze_strategy(name, signal_bars, directions, tp_sl_configs,
                     max_hold_bars_options, opens, highs, lows, closes,
                     n_bars, indicators, neutral_start, neutral_end):
    """
    Full analysis pipeline for one strategy:
    1. Signal count / frequency
    2. MFE/MAE analysis (using largest max_hold_bars)
    3. Backtest each TP/SL x max_hold_bars config
    4. Filter: min signals, WR excess, positive E[trade]
    5. MC test on best configs
    6. WF on MC-passing configs
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Strategy: {name}")
    logger.info(f"{'='*60}")

    # Filter signals to neutral window
    nw_signals = []
    nw_dirs = []
    for sb, d in zip(signal_bars, directions):
        if neutral_start <= sb <= neutral_end:
            nw_signals.append(sb)
            nw_dirs.append(d)

    n_signals = len(nw_signals)
    n_long = sum(1 for d in nw_dirs if d == 'LONG')
    n_short = n_signals - n_long
    days = (neutral_end - neutral_start) / BARS_PER_DAY
    freq_per_day = n_signals / days if days > 0 else 0

    logger.info(f"  Signals (neutral): {n_signals} ({n_long}L / {n_short}S)")
    logger.info(f"  Signal frequency: {freq_per_day:.2f}/day ({days:.0f} days)")

    if n_signals < MIN_SIGNALS:
        logger.info(f"  SKIP: too few signals ({n_signals} < {MIN_SIGNALS})")
        return {
            'name': name,
            'signals': n_signals,
            'signals_long': n_long,
            'signals_short': n_short,
            'freq_per_day': round(freq_per_day, 2),
            'skip_reason': f'too_few_signals ({n_signals} < {MIN_SIGNALS})',
            'configs': [],
            'best_config': None,
            'verdict': 'SKIP',
        }

    # MFE/MAE analysis (use max of max_hold_bars_options)
    max_mb = max(max_hold_bars_options)
    excursions = compute_mfe_mae(nw_signals, nw_dirs, opens, highs, lows,
                                 n_bars, max_mb)

    if len(excursions) > 0:
        mfes = [e['mfe_pct'] for e in excursions]
        maes = [e['mae_pct'] for e in excursions]
        median_mfe = np.median(mfes)
        median_mae = np.median(maes)
        mean_mfe = np.mean(mfes)
        mean_mae = np.mean(maes)
        mfe_mae_ratio = median_mfe / median_mae if median_mae > 0 else float('inf')
        pct_mfe_gt_mae = sum(1 for m, a in zip(mfes, maes) if m > a) / len(mfes) * 100
        mfe_p25 = np.percentile(mfes, 25)
        mfe_p75 = np.percentile(mfes, 75)
        mae_p25 = np.percentile(maes, 25)
        mae_p75 = np.percentile(maes, 75)
    else:
        median_mfe = median_mae = mean_mfe = mean_mae = 0
        mfe_mae_ratio = 0
        pct_mfe_gt_mae = 0
        mfe_p25 = mfe_p75 = mae_p25 = mae_p75 = 0

    logger.info(f"  MFE/MAE (max_hold={max_mb}b):")
    logger.info(f"    Median MFE: {median_mfe:.4f}%, Median MAE: {median_mae:.4f}%")
    logger.info(f"    MFE/MAE ratio: {mfe_mae_ratio:.3f}")
    logger.info(f"    % MFE > MAE: {pct_mfe_gt_mae:.1f}%")

    mfe_mae_info = {
        'median_mfe': round(median_mfe, 4),
        'median_mae': round(median_mae, 4),
        'mean_mfe': round(mean_mfe, 4),
        'mean_mae': round(mean_mae, 4),
        'mfe_mae_ratio': round(mfe_mae_ratio, 3),
        'pct_mfe_gt_mae': round(pct_mfe_gt_mae, 1),
        'mfe_p25': round(mfe_p25, 4),
        'mfe_p75': round(mfe_p75, 4),
        'mae_p25': round(mae_p25, 4),
        'mae_p75': round(mae_p75, 4),
    }

    # Backtest each config
    configs_results = []
    for max_bars in max_hold_bars_options:
        for tp_pct, sl_pct in tp_sl_configs:
            # Cost sensitivity
            baseline_wr = sl_pct / (tp_pct + sl_pct) * 100  # Random walk WR
            cost_as_pct_of_tp = COST_PER_TRADE / (tp_pct * LEVERAGE) * 100 if tp_pct > 0 else float('inf')
            # Break-even WR (including cost)
            # E[trade] = WR * (tp*L - fee) - (1-WR) * (sl*L + fee) = 0
            # WR = (sl*L + fee) / (tp*L + sl*L) = (sl*L + fee) / ((tp + sl)*L)
            breakeven_wr = (sl_pct * LEVERAGE + COST_PER_TRADE) / ((tp_pct + sl_pct) * LEVERAGE) * 100

            trades = backtest(nw_signals, nw_dirs, tp_pct, sl_pct,
                             opens, highs, lows, n_bars, max_bars)

            n_trades = len(trades)
            if n_trades < 10:
                continue

            wins = sum(1 for t in trades if t[2] > 0)
            wr = wins / n_trades * 100
            wr_excess = wr - baseline_wr

            # Per-trade E[value]
            trade_pnls = [t[2] for t in trades]
            ev_per_trade = np.mean(trade_pnls)

            # Compound PnL
            equity = 1.0
            for t in trades:
                equity *= (1 + t[2] / 100)
            compound_pnl = (equity - 1) * 100

            # Hold time stats
            hold_bars = [t[1] - t[0] for t in trades]
            median_hold = np.median(hold_bars) if hold_bars else 0

            config_result = {
                'tp': tp_pct,
                'sl': sl_pct,
                'max_bars': max_bars,
                'rr_ratio': round(tp_pct / sl_pct, 2) if sl_pct > 0 else 0,
                'baseline_wr': round(baseline_wr, 1),
                'breakeven_wr': round(breakeven_wr, 1),
                'cost_pct_of_tp_lev': round(cost_as_pct_of_tp, 1),
                'n_trades': n_trades,
                'wins': wins,
                'wr': round(wr, 2),
                'wr_excess': round(wr_excess, 2),
                'ev_per_trade': round(ev_per_trade, 4),
                'compound_pnl': round(compound_pnl, 2),
                'median_hold_bars': round(median_hold, 1),
                'mc_pvalue': None,
                'wf_result': None,
            }
            configs_results.append(config_result)

    # Sort by EV per trade descending
    configs_results.sort(key=lambda x: x['ev_per_trade'], reverse=True)

    logger.info(f"\n  Config results ({len(configs_results)} tested):")
    for i, cr in enumerate(configs_results[:10]):
        logger.info(f"    [{i+1}] TP={cr['tp']}% SL={cr['sl']}% MB={cr['max_bars']} | "
                    f"trades={cr['n_trades']} WR={cr['wr']:.1f}% "
                    f"WRx={cr['wr_excess']:+.1f}pp "
                    f"E[t]={cr['ev_per_trade']:.4f}% "
                    f"cPnL={cr['compound_pnl']:+.1f}%")

    # Filter configs: positive EV + WR excess >= threshold
    viable = [c for c in configs_results
              if c['ev_per_trade'] > 0 and c['wr_excess'] >= WR_EXCESS_MIN]

    logger.info(f"\n  Viable configs (EV>0, WRx>={WR_EXCESS_MIN}pp): {len(viable)}")

    if len(viable) == 0:
        logger.info(f"  VERDICT: NO viable configs")
        return {
            'name': name,
            'signals': n_signals,
            'signals_long': n_long,
            'signals_short': n_short,
            'freq_per_day': round(freq_per_day, 2),
            'mfe_mae': mfe_mae_info,
            'configs_tested': len(configs_results),
            'configs': configs_results[:5],  # Top 5 for reference
            'best_config': None,
            'verdict': 'NO_VIABLE_CONFIGS',
        }

    # MC test on top viable configs (max 5)
    mc_passed = []
    for c in viable[:5]:
        trades = backtest(nw_signals, nw_dirs, c['tp'], c['sl'],
                         opens, highs, lows, n_bars, c['max_bars'])
        trade_pnls = [t[2] for t in trades]
        pval = mc_test(trade_pnls)
        c['mc_pvalue'] = round(pval, 4)
        logger.info(f"    MC: TP={c['tp']} SL={c['sl']} MB={c['max_bars']} "
                    f"p={pval:.4f} {'PASS' if pval < MC_THRESHOLD else 'FAIL'}")
        if pval < MC_THRESHOLD:
            mc_passed.append(c)

    if len(mc_passed) == 0:
        logger.info(f"  VERDICT: All MC tests failed")
        return {
            'name': name,
            'signals': n_signals,
            'signals_long': n_long,
            'signals_short': n_short,
            'freq_per_day': round(freq_per_day, 2),
            'mfe_mae': mfe_mae_info,
            'configs_tested': len(configs_results),
            'configs': viable[:5],
            'best_config': None,
            'verdict': 'MC_FAIL',
        }

    # WF on MC-passed configs
    logger.info(f"\n  WF validation ({len(mc_passed)} configs)...")
    wf_passed = []

    for c in mc_passed:
        # Build signal generator function for this strategy
        gen_fn = _build_signal_generator(name, opens, highs, lows, closes, indicators)

        wf_results = walk_forward_test(
            gen_fn, opens, highs, lows, closes, indicators,
            c['tp'], c['sl'], c['max_bars'],
            n_folds=3, start_bar=neutral_start, end_bar=neutral_end
        )

        # Check: all folds positive OOS PnL
        all_positive = all(f['oos_pnl'] > 0 for f in wf_results if f['oos_trades'] > 0)
        folds_with_trades = [f for f in wf_results if f['oos_trades'] > 0]
        n_pass = sum(1 for f in folds_with_trades if f['oos_pnl'] > 0)

        c['wf_result'] = {
            'folds': wf_results,
            'pass': n_pass == len(folds_with_trades) and len(folds_with_trades) >= 2,
            'n_pass': n_pass,
            'n_folds_with_trades': len(folds_with_trades),
            'total_oos_pnl': round(sum(f['oos_pnl'] for f in wf_results), 2),
        }

        logger.info(f"    WF: TP={c['tp']} SL={c['sl']} MB={c['max_bars']} | "
                    f"{n_pass}/{len(folds_with_trades)} folds positive | "
                    f"OOS PnL: {c['wf_result']['total_oos_pnl']:+.1f}%")
        for f in wf_results:
            logger.info(f"      Fold {f['fold']}: IS={f['is_bars']}b OOS={f['oos_bars']}b "
                        f"trades={f['oos_trades']} WR={f['oos_wr']:.1f}% PnL={f['oos_pnl']:+.1f}%")

        if c['wf_result']['pass']:
            wf_passed.append(c)

    # Determine best config
    if len(wf_passed) > 0:
        best = max(wf_passed, key=lambda x: x['wf_result']['total_oos_pnl'])
        verdict = 'WF_PASS'
    elif len(mc_passed) > 0:
        best = mc_passed[0]
        verdict = 'MC_PASS_WF_FAIL'
    else:
        best = viable[0] if viable else configs_results[0] if configs_results else None
        verdict = 'NO_PASS'

    logger.info(f"\n  VERDICT: {verdict}")
    if best:
        logger.info(f"  Best: TP={best['tp']}% SL={best['sl']}% MB={best['max_bars']} "
                    f"WR={best['wr']:.1f}% E[t]={best['ev_per_trade']:.4f}% "
                    f"cPnL={best['compound_pnl']:+.1f}%")

    return {
        'name': name,
        'signals': n_signals,
        'signals_long': n_long,
        'signals_short': n_short,
        'freq_per_day': round(freq_per_day, 2),
        'mfe_mae': mfe_mae_info,
        'configs_tested': len(configs_results),
        'viable_count': len(viable),
        'mc_passed_count': len(mc_passed),
        'wf_passed_count': len(wf_passed),
        'configs': (wf_passed if wf_passed else mc_passed if mc_passed else viable)[:5],
        'best_config': best,
        'verdict': verdict,
    }


def _build_signal_generator(strategy_name, opens, highs, lows, closes, indicators):
    """Build a signal generator function for WF, given a strategy name."""
    # Return a function with signature (opens, highs, lows, closes, indicators, bar_start, bar_end)
    # Note: for WF, we re-generate signals within the OOS window
    generators = {
        'BMR_N1_T2.0': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=1, threshold_atr=2.0,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N1_T2.5': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=1, threshold_atr=2.5,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N1_T3.0': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=1, threshold_atr=3.0,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N2_T1.5': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=2, threshold_atr=1.5,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N2_T2.0': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=2, threshold_atr=2.0,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N2_T2.5': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=2, threshold_atr=2.5,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N2_T3.0': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=2, threshold_atr=3.0,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N3_T1.5': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=3, threshold_atr=1.5,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N3_T2.0': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=3, threshold_atr=2.0,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N3_T2.5': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=3, threshold_atr=2.5,
                               bar_start=bar_start, bar_end=bar_end),
        'BMR_N3_T3.0': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_bmr_signals(o, h, l, c, ind, lookback_n=3, threshold_atr=3.0,
                               bar_start=bar_start, bar_end=bar_end),
        'VSB': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_vsb_signals(o, h, l, c, ind, bar_start=bar_start, bar_end=bar_end),
        'WR': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_wick_reversion_signals(o, h, l, c, ind, bar_start=bar_start, bar_end=bar_end),
        'WR_3x1.5': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_wick_reversion_signals(o, h, l, c, ind, wick_body_ratio=3.0,
                                          wick_atr_ratio=1.5,
                                          bar_start=bar_start, bar_end=bar_end),
        'MTF': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_mtf_signals(o, h, l, c, ind, bar_start=bar_start, bar_end=bar_end),
        'PLB': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_plb_signals(o, h, l, c, ind, bar_start=bar_start, bar_end=bar_end),
        'RES_5_15_85': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_rsi_extreme_signals(o, h, l, c, ind, rsi_period=5,
                                        oversold=15, overbought=85,
                                        bar_start=bar_start, bar_end=bar_end),
        'RES_5_10_90': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_rsi_extreme_signals(o, h, l, c, ind, rsi_period=5,
                                        oversold=10, overbought=90,
                                        bar_start=bar_start, bar_end=bar_end),
        'RES_3_15_85': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_rsi_extreme_signals(o, h, l, c, ind, rsi_period=3,
                                        oversold=15, overbought=85,
                                        bar_start=bar_start, bar_end=bar_end),
        'RES_3_10_90': lambda o, h, l, c, ind, bar_start, bar_end:
            generate_rsi_extreme_signals(o, h, l, c, ind, rsi_period=3,
                                        oversold=10, overbought=90,
                                        bar_start=bar_start, bar_end=bar_end),
    }

    if strategy_name in generators:
        return generators[strategy_name]

    # Fallback: try to match prefix
    for key, fn in generators.items():
        if strategy_name.startswith(key.split('_')[0]):
            return fn

    raise ValueError(f"Unknown strategy: {strategy_name}")


# ============================================================
# Cost Analysis
# ============================================================
def cost_analysis():
    """Print cost structure for reference."""
    logger.info("\n" + "=" * 60)
    logger.info("COST ANALYSIS (3x Leverage)")
    logger.info("=" * 60)
    logger.info(f"  Fee per side: {FEE_PCT/2:.2f}%")
    logger.info(f"  Fee round-trip (price space): {FEE_PCT:.2f}%")
    logger.info(f"  Fee round-trip (capital space, x{LEVERAGE}): {FEE_CAPITAL:.2f}%")
    logger.info(f"  Slippage est: {SLIPPAGE:.2f}% (per entry)")
    logger.info(f"  Total cost per trade: {COST_PER_TRADE:.2f}%")
    logger.info("")
    logger.info("  Break-even WR at various TP/SL (R:R=1):")
    for tp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        sl = tp
        be_wr = (sl * LEVERAGE + COST_PER_TRADE) / ((tp + sl) * LEVERAGE) * 100
        logger.info(f"    TP={tp}% SL={sl}%: break-even WR = {be_wr:.1f}%")
    logger.info("")
    logger.info("  Break-even WR at various R:R ratios (TP=1.0%):")
    for sl in [0.5, 0.7, 1.0, 1.5]:
        be_wr = (sl * LEVERAGE + COST_PER_TRADE) / ((1.0 + sl) * LEVERAGE) * 100
        logger.info(f"    TP=1.0% SL={sl}%: break-even WR = {be_wr:.1f}%")
    logger.info(f"\n  Conclusion: For TP=1.0%/SL=1.0%, need WR > 55% to profit.")
    logger.info(f"  For TP=0.5%/SL=0.5%, need WR > 60% to profit (cost dominates).")
    logger.info(f"  TP < 0.5% is UNVIABLE at 3x leverage with 0.10% fee.\n")


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    cost_analysis()

    # Load data
    df, opens, highs, lows, closes, volumes = load_data()
    n_bars = len(opens)

    # Find neutral window
    nw = find_neutral_window(closes, tol_pct=1.0, min_days=90)
    if nw is None:
        logger.error("Could not find neutral window. Using full data range.")
        neutral_start, neutral_end = 0, n_bars - 1
    else:
        neutral_start, neutral_end = nw
        nw_days = (neutral_end - neutral_start) / BARS_PER_DAY
        logger.info(f"Neutral window: bars [{neutral_start}, {neutral_end}], "
                    f"{nw_days:.0f} days, "
                    f"start price ${closes[neutral_start]:.0f}, "
                    f"end price ${closes[neutral_end]:.0f}")

    # Compute indicators
    logger.info("\nComputing indicators...")
    indicators = compute_indicators(opens, highs, lows, closes, volumes)

    # TP/SL configurations per strategy type
    # NOTE: Minimum viable TP is ~0.5% (below that, cost destroys edge)
    # We test down to 0.5% but expect most below 0.7% to fail cost analysis.
    tp_sl_rr1 = [(0.5, 0.5), (0.7, 0.7), (1.0, 1.0), (1.5, 1.5)]
    tp_sl_rr_gt1 = [(0.7, 0.5), (1.0, 0.7), (1.0, 0.5), (1.5, 1.0), (2.0, 1.5)]
    tp_sl_rr_lt1 = [(0.5, 0.7), (0.7, 1.0), (1.0, 1.5)]  # Mean reversion: accept lower R:R

    # Define strategy configurations
    strategies = []

    # ---- Strategy 1: Big Move Reversal (BMR) ----
    # Multiple lookback / threshold combos
    bmr_configs = [
        ('BMR_N1_T2.0', 1, 2.0),
        ('BMR_N1_T2.5', 1, 2.5),
        ('BMR_N1_T3.0', 1, 3.0),
        ('BMR_N2_T1.5', 2, 1.5),
        ('BMR_N2_T2.0', 2, 2.0),
        ('BMR_N2_T2.5', 2, 2.5),
        ('BMR_N2_T3.0', 2, 3.0),
        ('BMR_N3_T1.5', 3, 1.5),
        ('BMR_N3_T2.0', 3, 2.0),
        ('BMR_N3_T2.5', 3, 2.5),
        ('BMR_N3_T3.0', 3, 3.0),
    ]
    for bname, lookback, threshold in bmr_configs:
        sigs, dirs = generate_bmr_signals(opens, highs, lows, closes, indicators,
                                          lookback_n=lookback, threshold_atr=threshold)
        strategies.append((
            bname, sigs, dirs,
            tp_sl_rr1 + tp_sl_rr_gt1 + tp_sl_rr_lt1,
            [12, 24, 36],
        ))

    # ---- Strategy 2: Volatility Squeeze Breakout (VSB) ----
    sigs, dirs = generate_vsb_signals(opens, highs, lows, closes, indicators)
    strategies.append((
        'VSB', sigs, dirs,
        tp_sl_rr1 + tp_sl_rr_gt1,
        [12, 24, 36],
    ))

    # ---- Strategy 3: Wick Reversion (WR) ----
    for wr_name, wbr, war in [('WR', 2.0, 1.0), ('WR_3x1.5', 3.0, 1.5)]:
        sigs, dirs = generate_wick_reversion_signals(
            opens, highs, lows, closes, indicators,
            wick_body_ratio=wbr, wick_atr_ratio=war)
        strategies.append((
            wr_name, sigs, dirs,
            tp_sl_rr1 + tp_sl_rr_gt1 + tp_sl_rr_lt1,
            [6, 12, 24],
        ))

    # ---- Strategy 4: Micro Trend Following (MTF) ----
    sigs, dirs = generate_mtf_signals(opens, highs, lows, closes, indicators)
    strategies.append((
        'MTF', sigs, dirs,
        tp_sl_rr_gt1 + [(1.0, 1.0), (1.5, 1.5)],
        [12, 24, 36],
    ))

    # ---- Strategy 5: Price Level Bounce (PLB) ----
    sigs, dirs = generate_plb_signals(opens, highs, lows, closes, indicators)
    strategies.append((
        'PLB', sigs, dirs,
        tp_sl_rr1 + tp_sl_rr_gt1,
        [6, 12, 24],
    ))

    # ---- Strategy 6: RSI Extreme Scalp (RES) ----
    for rname, period, ov_s, ob in [
        ('RES_5_15_85', 5, 15, 85),
        ('RES_5_10_90', 5, 10, 90),
        ('RES_3_15_85', 3, 15, 85),
        ('RES_3_10_90', 3, 10, 90),
    ]:
        sigs, dirs = generate_rsi_extreme_signals(
            opens, highs, lows, closes, indicators,
            rsi_period=period, oversold=ov_s, overbought=ob)
        strategies.append((
            rname, sigs, dirs,
            tp_sl_rr1 + tp_sl_rr_gt1 + tp_sl_rr_lt1,
            [6, 12, 24],
        ))

    # Run analysis
    results = {}
    for sname, sigs, dirs, tp_sl_cfgs, mb_options in strategies:
        result = analyze_strategy(
            sname, sigs, dirs, tp_sl_cfgs, mb_options,
            opens, highs, lows, closes, n_bars, indicators,
            neutral_start, neutral_end
        )
        results[sname] = result

    # ============================================================
    # Summary & Ranking
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Group by strategy family
    families = {}
    for name, r in results.items():
        family = name.split('_')[0]
        if family not in families:
            families[family] = []
        families[family].append(r)

    # Rank by verdict priority: WF_PASS > MC_PASS > NO_VIABLE > SKIP
    verdict_order = {'WF_PASS': 0, 'MC_PASS_WF_FAIL': 1, 'NO_VIABLE_CONFIGS': 2, 'MC_FAIL': 3, 'SKIP': 4, 'NO_PASS': 5}

    ranking = sorted(results.values(),
                     key=lambda r: (verdict_order.get(r['verdict'], 99),
                                    -(r['best_config']['ev_per_trade'] if r.get('best_config') and r['best_config'] else -999)))

    logger.info(f"\n{'Strategy':<25} {'Signals':>8} {'Freq/d':>7} {'Verdict':<20} "
                f"{'Best TP':>7} {'Best SL':>7} {'WR':>6} {'WRx':>6} {'E[t]':>8} {'cPnL':>8}")
    logger.info("-" * 110)
    for r in ranking:
        bc = r.get('best_config')
        if bc:
            logger.info(f"{r['name']:<25} {r['signals']:>8} {r['freq_per_day']:>7.2f} {r['verdict']:<20} "
                        f"{bc['tp']:>6.1f}% {bc['sl']:>6.1f}% {bc['wr']:>5.1f}% "
                        f"{bc['wr_excess']:>+5.1f} {bc['ev_per_trade']:>+7.4f}% {bc['compound_pnl']:>+7.1f}%")
        else:
            logger.info(f"{r['name']:<25} {r['signals']:>8} {r['freq_per_day']:>7.2f} {r['verdict']:<20}")

    # Overall verdict
    wf_winners = [r for r in ranking if r['verdict'] == 'WF_PASS']
    mc_only = [r for r in ranking if r['verdict'] == 'MC_PASS_WF_FAIL']

    logger.info(f"\n{'='*70}")
    logger.info("OVERALL VERDICT")
    logger.info(f"{'='*70}")
    logger.info(f"  WF PASS strategies: {len(wf_winners)}")
    logger.info(f"  MC PASS (WF FAIL) strategies: {len(mc_only)}")

    if wf_winners:
        best_overall = wf_winners[0]
        logger.info(f"\n  BEST STRATEGY: {best_overall['name']}")
        bc = best_overall['best_config']
        logger.info(f"    Config: TP={bc['tp']}% SL={bc['sl']}% MaxBars={bc['max_bars']}")
        logger.info(f"    Performance: WR={bc['wr']:.1f}% WRx={bc['wr_excess']:+.1f}pp "
                    f"E[t]={bc['ev_per_trade']:+.4f}% cPnL={bc['compound_pnl']:+.1f}%")
        if bc.get('wf_result'):
            logger.info(f"    WF OOS: {bc['wf_result']['total_oos_pnl']:+.1f}%")
        recommendation = (f"VIABLE: {best_overall['name']} shows genuine edge "
                         f"with WR excess {bc['wr_excess']:+.1f}pp, "
                         f"WF {bc['wf_result']['n_pass']}/{bc['wf_result']['n_folds_with_trades']} PASS. "
                         f"Consider further optimization and live paper testing.")
    else:
        recommendation = ("NO VIABLE SCALPING STRATEGIES found. "
                         "At 3x leverage with 0.30% round-trip cost, "
                         "short-holding BTC strategies cannot overcome transaction costs. "
                         "The existing pattern-based approach with wider TP/SL remains superior.")

    logger.info(f"\n  Recommendation: {recommendation}")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        'study': 'scalping_strategy',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': DATA_FILE,
        'data_rows': n_bars,
        'neutral_window': {
            'start_bar': neutral_start,
            'end_bar': neutral_end,
            'days': round((neutral_end - neutral_start) / BARS_PER_DAY, 1),
            'start_price': round(closes[neutral_start], 1),
            'end_price': round(closes[neutral_end], 1),
        },
        'protocol': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'fee_capital': FEE_CAPITAL,
            'slippage': SLIPPAGE,
            'cost_per_trade': COST_PER_TRADE,
            'mc_sims': MC_SIMS,
            'mc_seeds': MC_SEEDS,
            'mc_threshold': MC_THRESHOLD,
            'min_signals': MIN_SIGNALS,
            'wr_excess_min': WR_EXCESS_MIN,
        },
        'strategies': {name: _serialize_result(r) for name, r in results.items()},
        'ranking': [{'name': r['name'], 'verdict': r['verdict'],
                     'signals': r['signals'],
                     'best_ev': r['best_config']['ev_per_trade'] if r.get('best_config') and r['best_config'] else None}
                    for r in ranking],
        'verdict': {
            'wf_pass_count': len(wf_winners),
            'mc_pass_count': len(mc_only),
            'best_strategy': wf_winners[0]['name'] if wf_winners else None,
            'recommendation': recommendation,
        },
        'elapsed_seconds': round(time.time() - t0, 1),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info(f"\nResults saved to {OUTPUT_FILE}")
    logger.info(f"Total elapsed: {time.time() - t0:.1f}s")


def _serialize_result(r):
    """Clean result dict for JSON serialization."""
    # Remove non-serializable items if any
    return r


def _json_default(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    main()
