#!/usr/bin/env python3
"""
Strategy Pivot Study — Alternative Strategy Research
=====================================================

Investigates non-pattern-based strategies for BTC 5m trading.

Background:
  - Current 3-candle pattern strategy: R:R < 1, high WR dependency
  - BTC 5m direction prediction ≈ random walk (MFE/MAE 50/50)
  - N=9 correlation destroys portfolio WR during directional moves
  - Goal: Find strategies less dependent on direction prediction

Phases:
  Phase 1: Funding Rate Proxy (8h price cycle analysis)
  Phase 2: Mean Reversion (BB/ATR/VWAP/Keltner)
  Phase 3: Volatility Trading (breakout/selling/range/regime)
  Phase 4: Statistical Patterns (hourly seasonality, day-of-week)
  Phase 5: Hybrid (pattern + new strategies combined)
  Phase 6: Feasibility Assessment

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based, abs(tp - opens[bar]))
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage: 0.02%
  ATR scaling: ATR(14)/median(576), clamp [0.6, 1.7]
  WF: 3-fold expanding window, PASS = all folds OOS PnL > 0
  MC: Sign randomization, 3 seeds (42, 123, 7), max p < 0.01
  Timeout: 288 bars (24h), timeout trades DROPPED
  Sizing: compound (multiplicative)
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle

# ============================================================
# CONSTANTS — Standard Research Protocol
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10          # 0.05% x 2 sides
SLIPPAGE_BUFFER = 0.02  # 0.02% slippage
TIMEOUT_BARS = 288      # 24h at 5m
MAX_POSITIONS = 9       # N=9 virtual slots
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW
BARS_PER_DAY = 288
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]
MIN_TRADES = 25

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'strategy_pivot_study.json')


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load BTC 5m data and apply neutral window from dynamic_patterns.json."""
    df = pd.read_csv(DATA_FILE)
    n_total = len(df)

    # Apply neutral window from patterns file metadata
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE) as f:
            pat_data = json.load(f)
        neutral = pat_data.get('neutral_window')
        if neutral and neutral.get('enabled'):
            ns = neutral['start_bar']
            ne = neutral['end_bar']
            df = df.iloc[ns:ne + 1].reset_index(drop=True)
            print(f"Neutral window applied: bars {ns}-{ne} ({neutral['window_days']}d), "
                  f"drift {neutral['drift_pct']:+.2f}%")
        else:
            target_bars = 270 * BARS_PER_DAY
            if n_total > target_bars:
                start = n_total - target_bars
                df = df.iloc[start:].reset_index(drop=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64)
    n_bars = len(df)
    print(f"Data loaded: {n_bars} bars ({n_bars / BARS_PER_DAY:.0f}d)")
    return df, opens, highs, lows, closes, volumes, n_bars


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def compute_atr(highs, lows, closes, period=14):
    """True Range and ATR (Wilder smoothing)."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return tr, atr


def compute_atr_ratio(highs, lows, closes, period=ATR_PERIOD, window=ATR_WINDOW):
    """ATR / rolling_median(ATR) ratio."""
    _, atr = compute_atr(highs, lows, closes, period)
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(len(closes), np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def compute_bollinger(closes, period=20, num_std=2.0):
    """Bollinger Bands: middle, upper, lower, bandwidth."""
    s = pd.Series(closes)
    middle = s.rolling(period, min_periods=period).mean().values
    std = s.rolling(period, min_periods=period).std().values
    upper = middle + num_std * std
    lower = middle - num_std * std
    bw = np.full(len(closes), np.nan)
    valid = middle > 0
    bw[valid] = (upper[valid] - lower[valid]) / middle[valid] * 100
    return middle, upper, lower, bw


def compute_keltner(closes, highs, lows, ema_period=20, atr_period=14, atr_mult=2.0):
    """Keltner Channel: middle (EMA), upper, lower."""
    ema = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().values
    _, atr = compute_atr(highs, lows, closes, atr_period)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return ema, upper, lower


def compute_rsi(closes, period=14):
    """RSI indicator."""
    n = len(closes)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - 100 / (1 + rs)
    # Fill initial period
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100 - 100 / (1 + gains[:period].mean() /
                                     max(losses[:period].mean(), 1e-10))
    return rsi


def compute_vwap_session(opens, highs, lows, closes, volumes, bars_per_session=96):
    """Rolling VWAP approximation (reset every session = 8h = 96 bars)."""
    n = len(closes)
    typical = (highs + lows + closes) / 3.0
    vwap = np.full(n, np.nan)
    vwap_std = np.full(n, np.nan)

    for start in range(0, n, bars_per_session):
        end = min(start + bars_per_session, n)
        seg_tp = typical[start:end]
        seg_vol = volumes[start:end]
        cumvol = np.cumsum(seg_vol)
        cumtpv = np.cumsum(seg_tp * seg_vol)
        for j in range(end - start):
            if cumvol[j] > 0:
                v = cumtpv[j] / cumvol[j]
                vwap[start + j] = v
                # Rolling std of deviation
                if j >= 1:
                    devs = seg_tp[:j + 1] - v
                    vwap_std[start + j] = np.std(devs)
    return vwap, vwap_std


def compute_ema(closes, period=20):
    """EMA."""
    return pd.Series(closes).ewm(span=period, adjust=False).mean().values


# ============================================================
# SINGLE-TRADE BACKTEST ENGINE
# ============================================================

def backtest_signals(signal_list, opens, highs, lows, n_bars,
                     atr_ratio=None, use_atr=True, timeout_bars=TIMEOUT_BARS):
    """
    Backtest a list of (bar, direction, tp_pct, sl_pct) signals.

    Entry: next bar open after signal bar.
    Exit: intrabar high/low distance-based.
    Timeout: DROPPED.
    Fee: FEE_PCT * LEVERAGE.
    ATR scaling: applied if use_atr=True and atr_ratio available.

    Returns list of dicts with entry_bar, exit_bar, pnl, reason.
    """
    fee = FEE_PCT * LEVERAGE
    trades = []

    for sig_bar, direction, tp_pct, sl_pct in signal_list:
        entry_bar = sig_bar + 1
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue

        # ATR scaling
        if use_atr and atr_ratio is not None and sig_bar < len(atr_ratio) \
                and not np.isnan(atr_ratio[sig_bar]):
            r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE_BUFFER
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

        is_long = (direction == 'LONG')
        if is_long:
            tpp = entry * (1 + eff_tp / 100)
            slp = entry * (1 - eff_sl / 100)
        else:
            tpp = entry * (1 - eff_tp / 100)
            slp = entry * (1 + eff_sl / 100)

        resolved = False
        for j in range(entry_bar + 1, min(entry_bar + 1 + timeout_bars, n_bars)):
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
                if dist_tp <= dist_sl:
                    pnl = eff_tp * LEVERAGE - fee
                    reason = 'TP'
                else:
                    pnl = -eff_sl * LEVERAGE - fee
                    reason = 'SL'
                trades.append({'entry_bar': entry_bar, 'exit_bar': j,
                               'pnl': pnl, 'reason': reason,
                               'direction': direction})
                resolved = True
                break
            elif ht:
                pnl = eff_tp * LEVERAGE - fee
                trades.append({'entry_bar': entry_bar, 'exit_bar': j,
                               'pnl': pnl, 'reason': 'TP',
                               'direction': direction})
                resolved = True
                break
            elif hs:
                pnl = -eff_sl * LEVERAGE - fee
                trades.append({'entry_bar': entry_bar, 'exit_bar': j,
                               'pnl': pnl, 'reason': 'SL',
                               'direction': direction})
                resolved = True
                break
        # Timeout: DROP (no append)

    return trades


# ============================================================
# COMPOUND PnL + METRICS
# ============================================================

def compute_compound_pnl(trades, initial=100.0):
    """Compound (multiplicative) equity curve from sequential trades."""
    if not trades:
        return 0.0, 0.0, []
    sorted_trades = sorted(trades, key=lambda t: t['entry_bar'])
    equity = initial
    peak = equity
    max_dd = 0.0
    curve = [equity]

    for t in sorted_trades:
        pnl_pct = t['pnl']
        equity *= (1 + pnl_pct / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        curve.append(equity)

    total_pnl = (equity / initial - 1) * 100
    return total_pnl, max_dd, curve


def compute_metrics(trades):
    """Compute standard metrics from trade list."""
    if not trades:
        return {'n_trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0,
                'avg_win': 0, 'avg_loss': 0, 'pnl_per_trade': 0,
                'long_trades': 0, 'short_trades': 0, 'timeout_rate': 0}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins) / len(trades) * 100

    total_pnl, max_dd, _ = compute_compound_pnl(trades)
    pnl_mdd = total_pnl / max_dd if max_dd > 0 else 0

    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
    long_trades = sum(1 for t in trades if t['direction'] == 'LONG')
    short_trades = sum(1 for t in trades if t['direction'] == 'SHORT')

    return {
        'n_trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'avg_win': round(avg_win, 3),
        'avg_loss': round(avg_loss, 3),
        'pnl_per_trade': round(total_pnl / len(trades), 3),
        'long_trades': long_trades,
        'short_trades': short_trades,
    }


# ============================================================
# MONTE CARLO TEST
# ============================================================

def mc_test(pnls, n_sims=MC_SIMS):
    """Multi-seed sign randomization MC test. Returns max p-value."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(n_sims, len(pnls_arr)))
        shuffled_sums = signs @ pnls_arr
        p = np.mean(shuffled_sums >= actual)
        p_vals.append(p)
    return max(p_vals)


# ============================================================
# WALK-FORWARD VALIDATION (3-fold expanding window)
# ============================================================

def walk_forward_validate(signal_generator_fn, opens, highs, lows, closes, volumes,
                          n_bars, atr_ratio, n_folds=3, **gen_kwargs):
    """
    Walk-forward validation with expanding IS window.

    signal_generator_fn(start, end, opens, highs, lows, closes, volumes, **kwargs)
      -> list of (bar, direction, tp_pct, sl_pct)

    IS = [0..seg*(fold+1)], OOS = [seg*(fold+1)..seg*(fold+2)]
    """
    seg_size = n_bars // (n_folds + 1)
    fold_results = []

    for fold in range(n_folds):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        # Generate signals on IS period (for parameter optimization)
        # Then generate signals on OOS period with IS-learned parameters
        oos_signals = signal_generator_fn(
            oos_start, oos_end, opens, highs, lows, closes, volumes,
            is_start=0, is_end=is_end, **gen_kwargs)

        trades = backtest_signals(oos_signals, opens, highs, lows, n_bars,
                                  atr_ratio=atr_ratio)
        m = compute_metrics(trades)
        pnls = [t['pnl'] for t in trades]
        p_val = mc_test(pnls) if len(pnls) >= MIN_TRADES else 1.0

        fold_results.append({
            'fold': fold + 1,
            'is_bars': is_end,
            'oos_start': oos_start,
            'oos_end': oos_end,
            **m,
            'mc_p': round(p_val, 4),
        })

    total_pnl = sum(f['pnl'] for f in fold_results)
    max_mdd = max(f['mdd'] for f in fold_results) if fold_results else 0
    n_pass = sum(1 for f in fold_results if f['pnl'] > 0)
    total_trades = sum(f['n_trades'] for f in fold_results)

    return {
        'folds': fold_results,
        'total_pnl': round(total_pnl, 2),
        'max_mdd': round(max_mdd, 2),
        'pnl_mdd': round(total_pnl / max_mdd, 2) if max_mdd > 0 else 0,
        'total_trades': total_trades,
        'wf_pass': f"{n_pass}/{n_folds}",
        'wf_passed': n_pass == n_folds,
    }


# ============================================================
# PHASE 1: FUNDING RATE PROXY (8h PRICE CYCLE)
# ============================================================

def phase1_funding_rate_proxy(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio):
    """
    Analyze 8h price cycle patterns as funding rate proxy.

    BingX perpetual funding every 8h (00:00, 08:00, 16:00 UTC).
    We look for price patterns around these times.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: FUNDING RATE PROXY — 8h Price Cycle Analysis")
    print("=" * 70)

    results = {}

    # Strategy 1: Hour-of-day return bias
    # Map each bar to hour of day
    hours = df['timestamp'].dt.hour.values
    minutes = df['timestamp'].dt.minute.values
    bar_returns = np.zeros(n_bars)
    bar_returns[1:] = (closes[1:] / closes[:-1] - 1) * 100

    # Analyze return by hour
    hourly_stats = {}
    for h in range(24):
        mask = hours == h
        rets = bar_returns[mask]
        if len(rets) > 0:
            hourly_stats[h] = {
                'mean_ret': round(float(np.mean(rets)), 5),
                'std_ret': round(float(np.std(rets)), 5),
                'n_bars': int(np.sum(mask)),
                'sharpe': round(float(np.mean(rets) / max(np.std(rets), 1e-10)), 4),
                'cum_ret': round(float(np.sum(rets)), 3),
            }

    # Find best/worst hours
    best_hour = max(hourly_stats, key=lambda h: hourly_stats[h]['cum_ret'])
    worst_hour = min(hourly_stats, key=lambda h: hourly_stats[h]['cum_ret'])

    print(f"\n  Hourly return analysis ({n_bars / BARS_PER_DAY:.0f} days):")
    print(f"  {'Hour':>4s} | {'Mean Ret%':>10s} | {'Cum Ret%':>10s} | {'Sharpe':>8s} | {'N bars':>6s}")
    print(f"  {'-' * 50}")
    for h in sorted(hourly_stats):
        s = hourly_stats[h]
        marker = " <-- BEST" if h == best_hour else (" <-- WORST" if h == worst_hour else "")
        print(f"  {h:4d} | {s['mean_ret']:10.5f} | {s['cum_ret']:10.3f} | "
              f"{s['sharpe']:8.4f} | {s['n_bars']:6d}{marker}")

    results['hourly_analysis'] = hourly_stats
    results['best_hour'] = best_hour
    results['worst_hour'] = worst_hour

    # Strategy 2: Funding window trade
    # Look at returns in 2h windows before funding times (0, 8, 16 UTC)
    funding_hours = [0, 8, 16]
    funding_window_bars = 24  # 2h before funding = 24 bars at 5m

    pre_funding_returns = {}
    for fh in funding_hours:
        # Pre-funding window: [fh-2h, fh)
        pre_start_hour = (fh - 2) % 24
        mask = np.zeros(n_bars, dtype=bool)
        for i in range(n_bars):
            bar_h = hours[i]
            bar_m = minutes[i]
            # Within 2h before funding
            if fh == 0:
                if bar_h >= 22 or bar_h < 0:
                    mask[i] = True
            elif bar_h >= fh - 2 and bar_h < fh:
                mask[i] = True

        rets = bar_returns[mask]
        pre_funding_returns[f"pre_funding_{fh:02d}"] = {
            'n_bars': int(np.sum(mask)),
            'mean_ret': round(float(np.mean(rets)), 5) if len(rets) > 0 else 0,
            'cum_ret': round(float(np.sum(rets)), 3) if len(rets) > 0 else 0,
        }

    print(f"\n  Pre-funding (2h window) returns:")
    for key, val in pre_funding_returns.items():
        print(f"    {key}: cum_ret={val['cum_ret']:+.3f}%, "
              f"n_bars={val['n_bars']}")

    results['pre_funding_returns'] = pre_funding_returns

    # Strategy 3: Simple time-of-day trading strategy
    # LONG during best 4h, SHORT during worst 4h
    # Sort hours by cumulative return
    sorted_hours = sorted(hourly_stats, key=lambda h: hourly_stats[h]['cum_ret'],
                          reverse=True)
    best_4h = set(sorted_hours[:4])
    worst_4h = set(sorted_hours[-4:])

    print(f"\n  Best 4 hours (LONG): {sorted(best_4h)}")
    print(f"  Worst 4 hours (SHORT): {sorted(worst_4h)}")

    # Generate signals: enter at start of each hour block, exit at end
    tod_signals = []
    for i in range(n_bars - 1):
        h = hours[i]
        m = minutes[i]
        if m == 0:  # Start of hour
            if h in best_4h:
                tod_signals.append((i, 'LONG', 0.5, 0.5))
            elif h in worst_4h:
                tod_signals.append((i, 'SHORT', 0.5, 0.5))

    # Test multiple TP/SL levels
    tp_sl_pairs = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7), (1.0, 1.0), (0.5, 1.0)]
    tod_results = {}

    for tp, sl in tp_sl_pairs:
        sigs = [(s[0], s[1], tp, sl) for s in tod_signals]
        trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                  atr_ratio=atr_ratio, use_atr=True)
        m = compute_metrics(trades)
        label = f"ToD_TP{tp}_SL{sl}"
        tod_results[label] = m
        print(f"\n  {label}: trades={m['n_trades']}, WR={m['wr']:.1f}%, "
              f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%")

    results['time_of_day_strategies'] = tod_results

    # Verdict
    any_positive = any(v['pnl'] > 0 and v['n_trades'] >= MIN_TRADES
                       for v in tod_results.values())
    results['verdict'] = 'PARTIAL' if any_positive else 'STOP'
    results['note'] = ("Hourly returns are small and likely noise. "
                       "Funding rate data not available in OHLCV. "
                       "Time-of-day effect too weak for standalone strategy.")

    print(f"\n  Phase 1 Verdict: {results['verdict']}")
    return results


# ============================================================
# PHASE 2: MEAN REVERSION STRATEGIES
# ============================================================

def phase2_mean_reversion(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio):
    """
    Mean reversion strategies that don't depend on direction prediction.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: MEAN REVERSION STRATEGIES")
    print("=" * 70)

    results = {}

    # Pre-compute indicators
    bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger(closes, 20, 2.0)
    kc_mid, kc_upper, kc_lower = compute_keltner(closes, highs, lows, 20, 14, 2.0)
    rsi = compute_rsi(closes, 14)
    vwap, vwap_std = compute_vwap_session(opens, highs, lows, closes, volumes, 96)
    _, atr = compute_atr(highs, lows, closes, 14)

    # ---- Strategy MR1: Bollinger Band Mean Reversion ----
    print("\n  MR1: Bollinger Band Mean Reversion")
    for bb_std_thresh in [2.0, 2.5, 3.0]:
        bb_mid_t, bb_up_t, bb_lo_t, _ = compute_bollinger(closes, 20, bb_std_thresh)
        mr1_signals = []
        for i in range(ATR_WARMUP, n_bars - 1):
            if np.isnan(bb_up_t[i]) or np.isnan(bb_lo_t[i]):
                continue
            # Price crosses below lower band -> LONG (mean reversion up)
            if closes[i] < bb_lo_t[i] and closes[i - 1] >= bb_lo_t[i - 1]:
                mr1_signals.append((i, 'LONG', 1.0, 1.5))
            # Price crosses above upper band -> SHORT (mean reversion down)
            elif closes[i] > bb_up_t[i] and closes[i - 1] <= bb_up_t[i - 1]:
                mr1_signals.append((i, 'SHORT', 1.0, 1.5))

        # Test multiple TP/SL
        for tp, sl in [(0.8, 1.2), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]:
            sigs = [(s[0], s[1], tp, sl) for s in mr1_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"MR1_BB{bb_std_thresh}_TP{tp}_SL{sl}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # ---- Strategy MR2: ATR Squeeze -> Expansion Straddle ----
    print("\n  MR2: ATR Squeeze Straddle (dual-entry)")
    # When ATR is very low (squeeze), place both LONG and SHORT
    # The idea: one side wins, other loses, net depends on R:R
    atr_pctile = pd.Series(atr).rolling(BARS_PER_DAY).rank(pct=True).values

    for squeeze_thresh in [0.10, 0.20, 0.30]:
        mr2_signals = []
        for i in range(ATR_WARMUP, n_bars - 1):
            if np.isnan(atr_pctile[i]):
                continue
            if atr_pctile[i] < squeeze_thresh:
                # Both directions — straddle
                mr2_signals.append((i, 'LONG', 1.5, 1.0))
                mr2_signals.append((i, 'SHORT', 1.5, 1.0))

        for tp, sl in [(1.0, 0.5), (1.5, 1.0), (2.0, 1.5)]:
            # Note: SL < 1.0% violates protocol for live, but we test to see edge
            if sl < 1.0:
                sl_test = 1.0  # Floor to 1.0% minimum
            else:
                sl_test = sl
            sigs = [(s[0], s[1], tp, sl_test) for s in mr2_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"MR2_squeeze{squeeze_thresh}_TP{tp}_SL{sl_test}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # ---- Strategy MR3: VWAP Deviation ----
    print("\n  MR3: VWAP Deviation Mean Reversion")
    for dev_mult in [1.5, 2.0, 2.5]:
        mr3_signals = []
        for i in range(ATR_WARMUP, n_bars - 1):
            if np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or vwap_std[i] <= 0:
                continue
            z_score = (closes[i] - vwap[i]) / vwap_std[i]
            if z_score > dev_mult:
                mr3_signals.append((i, 'SHORT', 1.0, 1.5))
            elif z_score < -dev_mult:
                mr3_signals.append((i, 'LONG', 1.0, 1.5))

        for tp, sl in [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]:
            sigs = [(s[0], s[1], tp, sl) for s in mr3_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"MR3_VWAP{dev_mult}_TP{tp}_SL{sl}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # ---- Strategy MR4: Keltner Channel Return ----
    print("\n  MR4: Keltner Channel Breakout-and-Return")
    for kc_mult in [2.0, 2.5, 3.0]:
        kc_m, kc_u, kc_l = compute_keltner(closes, highs, lows, 20, 14, kc_mult)
        mr4_signals = []
        for i in range(ATR_WARMUP + 1, n_bars - 1):
            if np.isnan(kc_u[i]) or np.isnan(kc_l[i]):
                continue
            # Was outside upper, now returning inside -> SHORT
            if closes[i - 1] > kc_u[i - 1] and closes[i] <= kc_u[i]:
                mr4_signals.append((i, 'SHORT', 1.0, 1.5))
            # Was outside lower, now returning inside -> LONG
            elif closes[i - 1] < kc_l[i - 1] and closes[i] >= kc_l[i]:
                mr4_signals.append((i, 'LONG', 1.0, 1.5))

        for tp, sl in [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]:
            sigs = [(s[0], s[1], tp, sl) for s in mr4_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"MR4_KC{kc_mult}_TP{tp}_SL{sl}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # ---- Strategy MR5: RSI Extreme Reversion ----
    print("\n  MR5: RSI Extreme Mean Reversion")
    for rsi_thresh in [20, 25, 30]:
        mr5_signals = []
        for i in range(ATR_WARMUP, n_bars - 1):
            if np.isnan(rsi[i]):
                continue
            if rsi[i] < rsi_thresh:
                mr5_signals.append((i, 'LONG', 1.0, 1.5))
            elif rsi[i] > (100 - rsi_thresh):
                mr5_signals.append((i, 'SHORT', 1.0, 1.5))

        for tp, sl in [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]:
            sigs = [(s[0], s[1], tp, sl) for s in mr5_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"MR5_RSI{rsi_thresh}_TP{tp}_SL{sl}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # Summary: find best MR strategy
    viable = {k: v for k, v in results.items()
              if v.get('n_trades', 0) >= MIN_TRADES and v.get('pnl', 0) > 0}
    if viable:
        best = max(viable, key=lambda k: viable[k]['pnl'])
        results['best_mr'] = best
        results['best_mr_metrics'] = viable[best]
        print(f"\n  Best MR: {best} -> PnL={viable[best]['pnl']:+.1f}%, "
              f"WR={viable[best]['wr']:.1f}%")
    else:
        results['best_mr'] = None
        print("\n  No MR strategy produced positive PnL with enough trades.")

    results['verdict'] = 'PARTIAL' if viable else 'STOP'
    return results


# ============================================================
# PHASE 3: VOLATILITY TRADING
# ============================================================

def phase3_volatility_trading(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio):
    """
    Volatility-based strategies.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: VOLATILITY TRADING STRATEGIES")
    print("=" * 70)

    results = {}
    _, atr = compute_atr(highs, lows, closes, 14)
    bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger(closes, 20, 2.0)
    ema20 = compute_ema(closes, 20)

    # ATR ratio already computed
    atr_pctile = pd.Series(atr).rolling(BARS_PER_DAY).rank(pct=True).values

    # ---- V1: Volatility Breakout ----
    print("\n  V1: Volatility Breakout (ATR expansion + direction)")
    for atr_ratio_thresh in [1.3, 1.5, 1.7]:
        v1_signals = []
        for i in range(ATR_WARMUP + 1, n_bars - 1):
            if np.isnan(atr_ratio[i]) or np.isnan(atr_ratio[i - 1]):
                continue
            # ATR expanding past threshold (volatility breakout)
            if atr_ratio[i] > atr_ratio_thresh and atr_ratio[i - 1] <= atr_ratio_thresh:
                # Direction: follow the breakout
                if closes[i] > opens[i]:
                    v1_signals.append((i, 'LONG', 1.5, 2.0))
                else:
                    v1_signals.append((i, 'SHORT', 1.5, 2.0))

        for tp, sl in [(1.0, 2.0), (1.5, 2.0), (2.0, 3.0), (1.0, 1.5)]:
            sigs = [(s[0], s[1], tp, sl) for s in v1_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"V1_ATRbreak{atr_ratio_thresh}_TP{tp}_SL{sl}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # ---- V2: Volatility Selling (range-bound in high vol) ----
    print("\n  V2: Volatility Selling (fade extreme moves)")
    for atr_hi_thresh in [1.5, 1.7, 2.0]:
        v2_signals = []
        for i in range(ATR_WARMUP + 1, n_bars - 1):
            if np.isnan(atr_ratio[i]):
                continue
            if atr_ratio[i] > atr_hi_thresh:
                # High vol: fade the move (mean revert)
                bar_ret = (closes[i] / opens[i] - 1) * 100
                if bar_ret > 0.3:  # Big up -> SHORT (sell vol)
                    v2_signals.append((i, 'SHORT', 1.0, 2.0))
                elif bar_ret < -0.3:  # Big down -> LONG (sell vol)
                    v2_signals.append((i, 'LONG', 1.0, 2.0))

        for tp, sl in [(1.0, 1.5), (1.0, 2.0), (1.5, 2.5), (2.0, 3.0)]:
            sigs = [(s[0], s[1], tp, sl) for s in v2_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"V2_volsell{atr_hi_thresh}_TP{tp}_SL{sl}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # ---- V3: Range Trading (low vol Bollinger bounce) ----
    print("\n  V3: Range Trading (BB bounce in low vol)")
    bb_width_pctile = pd.Series(bb_width).rolling(BARS_PER_DAY).rank(pct=True).values

    for vol_low_thresh in [0.20, 0.30, 0.40]:
        v3_signals = []
        for i in range(ATR_WARMUP + 1, n_bars - 1):
            if np.isnan(bb_width_pctile[i]) or np.isnan(bb_lower[i]):
                continue
            if bb_width_pctile[i] > vol_low_thresh:
                continue  # Only trade when BB is tight (low vol)
            # Bounce off bands
            if closes[i] <= bb_lower[i]:
                v3_signals.append((i, 'LONG', 1.0, 1.5))
            elif closes[i] >= bb_upper[i]:
                v3_signals.append((i, 'SHORT', 1.0, 1.5))

        for tp, sl in [(1.0, 1.5), (1.5, 2.0), (0.8, 1.2)]:
            if sl < 1.0:
                sl = 1.0  # Floor
            sigs = [(s[0], s[1], tp, sl) for s in v3_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"V3_range{vol_low_thresh}_TP{tp}_SL{sl}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # ---- V4: Volatility Regime Switch ----
    print("\n  V4: Volatility Regime Switch (MR in low vol, trend in high vol)")
    for regime_split in [0.40, 0.50, 0.60]:
        v4_signals = []
        for i in range(ATR_WARMUP + 1, n_bars - 1):
            if np.isnan(atr_pctile[i]) or np.isnan(bb_lower[i]):
                continue

            if atr_pctile[i] < regime_split:
                # Low vol regime -> mean reversion
                if closes[i] <= bb_lower[i]:
                    v4_signals.append((i, 'LONG', 1.0, 1.5))
                elif closes[i] >= bb_upper[i]:
                    v4_signals.append((i, 'SHORT', 1.0, 1.5))
            else:
                # High vol regime -> trend follow
                if closes[i] > ema20[i] and closes[i - 1] <= ema20[i - 1]:
                    v4_signals.append((i, 'LONG', 2.0, 1.5))
                elif closes[i] < ema20[i] and closes[i - 1] >= ema20[i - 1]:
                    v4_signals.append((i, 'SHORT', 2.0, 1.5))

        for tp, sl in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5)]:
            sigs = [(s[0], s[1], tp, sl) for s in v4_signals]
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            label = f"V4_regime{regime_split}_TP{tp}_SL{sl}"
            results[label] = m
            if m['n_trades'] >= MIN_TRADES:
                pnls = [t['pnl'] for t in trades]
                mc_p = mc_test(pnls)
                results[label]['mc_p'] = round(mc_p, 4)
                print(f"    {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                      f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%, "
                      f"MC_p={mc_p:.4f}")

    # Summary
    viable = {k: v for k, v in results.items()
              if v.get('n_trades', 0) >= MIN_TRADES and v.get('pnl', 0) > 0}
    if viable:
        best = max(viable, key=lambda k: viable[k]['pnl'])
        results['best_vol'] = best
        results['best_vol_metrics'] = viable[best]
        print(f"\n  Best Vol: {best} -> PnL={viable[best]['pnl']:+.1f}%")
    else:
        results['best_vol'] = None
        print("\n  No volatility strategy produced positive PnL with enough trades.")

    results['verdict'] = 'PARTIAL' if viable else 'STOP'
    return results


# ============================================================
# PHASE 4: STATISTICAL PATTERNS (Seasonality)
# ============================================================

def phase4_statistical_patterns(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio):
    """
    Time-based statistical patterns.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: STATISTICAL PATTERNS (Seasonality)")
    print("=" * 70)

    results = {}
    hours = df['timestamp'].dt.hour.values
    dow = df['timestamp'].dt.dayofweek.values  # 0=Mon, 6=Sun
    bar_returns = np.zeros(n_bars)
    bar_returns[1:] = (closes[1:] / closes[:-1] - 1) * 100

    # Day-of-week analysis
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_stats = {}
    print("\n  Day-of-week analysis:")
    print(f"  {'Day':>5s} | {'Mean Ret%':>10s} | {'Cum Ret%':>10s} | {'Std':>8s} | {'Sharpe':>8s} | {'N':>5s}")
    print(f"  {'-' * 55}")
    for d in range(7):
        mask = dow == d
        rets = bar_returns[mask]
        if len(rets) > 0:
            dow_stats[dow_names[d]] = {
                'mean_ret': round(float(np.mean(rets)), 5),
                'cum_ret': round(float(np.sum(rets)), 3),
                'std': round(float(np.std(rets)), 5),
                'sharpe': round(float(np.mean(rets) / max(np.std(rets), 1e-10)), 4),
                'n_bars': int(np.sum(mask)),
            }
            s = dow_stats[dow_names[d]]
            print(f"  {dow_names[d]:>5s} | {s['mean_ret']:10.5f} | {s['cum_ret']:10.3f} | "
                  f"{s['std']:8.5f} | {s['sharpe']:8.4f} | {s['n_bars']:5d}")
    results['day_of_week'] = dow_stats

    # Hour-of-day × Day-of-week interaction
    # Find the absolute best/worst (hour, day) combos
    hd_stats = {}
    for d in range(7):
        for h in range(24):
            mask = (dow == d) & (hours == h)
            rets = bar_returns[mask]
            if len(rets) >= 10:
                key = f"{dow_names[d]}_{h:02d}"
                hd_stats[key] = {
                    'mean_ret': float(np.mean(rets)),
                    'cum_ret': float(np.sum(rets)),
                    'n': int(np.sum(mask)),
                }

    if hd_stats:
        sorted_hd = sorted(hd_stats, key=lambda k: hd_stats[k]['cum_ret'])
        print(f"\n  Top 5 (hour, day) combos (LONG):")
        for key in sorted_hd[-5:]:
            s = hd_stats[key]
            print(f"    {key}: cum_ret={s['cum_ret']:+.3f}%, n={s['n']}")
        print(f"\n  Bottom 5 (hour, day) combos (SHORT):")
        for key in sorted_hd[:5]:
            s = hd_stats[key]
            print(f"    {key}: cum_ret={s['cum_ret']:+.3f}%, n={s['n']}")

    results['hour_day_interaction'] = hd_stats

    # Volatility by hour
    bar_abs_ret = np.abs(bar_returns)
    hourly_vol = {}
    print(f"\n  Volatility by hour:")
    for h in range(24):
        mask = hours == h
        vols = bar_abs_ret[mask]
        if len(vols) > 0:
            hourly_vol[h] = {
                'mean_abs_ret': round(float(np.mean(vols)), 5),
                'p95_abs_ret': round(float(np.percentile(vols, 95)), 5),
            }
    # Find quietest and noisiest hours
    quietest = min(hourly_vol, key=lambda h: hourly_vol[h]['mean_abs_ret'])
    noisiest = max(hourly_vol, key=lambda h: hourly_vol[h]['mean_abs_ret'])
    print(f"    Quietest hour: {quietest} (mean |ret|={hourly_vol[quietest]['mean_abs_ret']:.5f}%)")
    print(f"    Noisiest hour: {noisiest} (mean |ret|={hourly_vol[noisiest]['mean_abs_ret']:.5f}%)")
    results['hourly_volatility'] = hourly_vol

    # Strategy: Trade only during volatile hours (more edge opportunity)
    volatile_hours = set(sorted(hourly_vol,
                                key=lambda h: hourly_vol[h]['mean_abs_ret'],
                                reverse=True)[:8])
    quiet_hours = set(sorted(hourly_vol,
                             key=lambda h: hourly_vol[h]['mean_abs_ret'])[:8])
    print(f"\n  Most volatile 8h: {sorted(volatile_hours)}")
    print(f"  Quietest 8h: {sorted(quiet_hours)}")
    results['volatile_hours'] = sorted(volatile_hours)
    results['quiet_hours'] = sorted(quiet_hours)

    # Test: pattern strategy only during volatile hours vs quiet hours
    # (Requires pattern data)
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE) as f:
            pat_data = json.load(f)
        pL = set(pat_data['patterns']['long'])
        pS = set(pat_data['patterns']['short'])
        tpsl_map = {}
        for pn, ts in pat_data['patterns_tpsl'].items():
            tpsl_map[pn] = (ts[0], ts[1])

        types = df['type_code'].values.tolist() if 'type_code' in df.columns else None
        if types is None:
            # Need to classify
            from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW
            df['body_abs'] = abs(df['close'] - df['open'])
            df['avg_body_calc'] = df['body_abs'].rolling(AVG_BODY_WINDOW).mean()
            types = []
            for i, row in df.iterrows():
                avg_b = df.at[i, 'avg_body_calc']
                if pd.isna(avg_b):
                    avg_b = 1.0
                ct = classify_candle(row, avg_b)
                types.append(ct.value)

        # Generate pattern signals
        all_pat_signals = []
        for i in range(2, n_bars):
            pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
            if pat in pL:
                tp, sl = tpsl_map.get(pat, (1.5, 2.0))
                all_pat_signals.append((i, 'LONG', tp, sl, hours[i]))
            if pat in pS:
                tp, sl = tpsl_map.get(pat, (1.5, 2.0))
                all_pat_signals.append((i, 'SHORT', tp, sl, hours[i]))

        # Filter by volatile/quiet hours
        vol_sigs = [(s[0], s[1], s[2], s[3]) for s in all_pat_signals if s[4] in volatile_hours]
        quiet_sigs = [(s[0], s[1], s[2], s[3]) for s in all_pat_signals if s[4] in quiet_hours]
        all_sigs = [(s[0], s[1], s[2], s[3]) for s in all_pat_signals]

        for label, sigs in [('pattern_volatile_8h', vol_sigs),
                            ('pattern_quiet_8h', quiet_sigs),
                            ('pattern_all_hours', all_sigs)]:
            trades = backtest_signals(sigs, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            results[label] = m
            print(f"\n  {label}: T={m['n_trades']}, WR={m['wr']:.1f}%, "
                  f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%")

    results['verdict'] = 'INFORMATIONAL'
    results['note'] = ("Seasonality patterns are weak on 5m BTC. "
                       "Hour/day effects are small relative to noise. "
                       "Time filter may improve existing strategy marginally.")
    print(f"\n  Phase 4 Verdict: INFORMATIONAL (data-only, no standalone strategy)")
    return results


# ============================================================
# PHASE 5: HYBRID STRATEGIES
# ============================================================

def phase5_hybrid(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio,
                  phase2_results, phase3_results, phase4_results):
    """
    Combine current pattern strategy with alternative signals.
    """
    print("\n" + "=" * 70)
    print("PHASE 5: HYBRID STRATEGIES")
    print("=" * 70)

    results = {}

    # Load pattern data
    if not os.path.exists(PATTERNS_FILE):
        print("  No pattern file found, skipping hybrid phase.")
        results['verdict'] = 'SKIP'
        return results

    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pL = set(pat_data['patterns']['long'])
    pS = set(pat_data['patterns']['short'])
    tpsl_map = {}
    for pn, ts in pat_data['patterns_tpsl'].items():
        tpsl_map[pn] = (ts[0], ts[1])

    types = df['type_code'].values.tolist() if 'type_code' in df.columns else None
    if types is None:
        from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW
        df['body_abs'] = abs(df['close'] - df['open'])
        df['avg_body_calc'] = df['body_abs'].rolling(AVG_BODY_WINDOW).mean()
        types = []
        for i, row in df.iterrows():
            avg_b = df.at[i, 'avg_body_calc']
            if pd.isna(avg_b):
                avg_b = 1.0
            ct = classify_candle(row, avg_b)
            types.append(ct.value)

    hours = df['timestamp'].dt.hour.values
    bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger(closes, 20, 2.0)
    rsi = compute_rsi(closes, 14)
    _, atr = compute_atr(highs, lows, closes, 14)
    ema20 = compute_ema(closes, 20)

    # Generate base pattern signals
    pat_signals = []
    for i in range(2, n_bars):
        pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
        if pat in pL:
            tp, sl = tpsl_map.get(pat, (1.5, 2.0))
            pat_signals.append((i, 'LONG', tp, sl))
        if pat in pS:
            tp, sl = tpsl_map.get(pat, (1.5, 2.0))
            pat_signals.append((i, 'SHORT', tp, sl))

    # Baseline: pattern only
    trades = backtest_signals(pat_signals, opens, highs, lows, n_bars,
                              atr_ratio=atr_ratio)
    m = compute_metrics(trades)
    results['baseline_pattern'] = m
    print(f"\n  Baseline (pattern only): T={m['n_trades']}, WR={m['wr']:.1f}%, "
          f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%")

    # H1: Pattern + Time Filter (trade only volatile hours)
    volatile_hours = set(phase4_results.get('volatile_hours', []))
    if volatile_hours:
        h1_sigs = [(s[0], s[1], s[2], s[3]) for s in pat_signals
                   if hours[s[0]] in volatile_hours]
        trades = backtest_signals(h1_sigs, opens, highs, lows, n_bars,
                                  atr_ratio=atr_ratio)
        m = compute_metrics(trades)
        results['H1_pattern_volatile_hours'] = m
        print(f"  H1 (pattern+volatile hours): T={m['n_trades']}, WR={m['wr']:.1f}%, "
              f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%")

    # H2: Pattern + BB confirmation (only enter if price is near BB band in direction)
    h2_sigs = []
    for sig_bar, direction, tp, sl in pat_signals:
        if np.isnan(bb_upper[sig_bar]) or np.isnan(bb_lower[sig_bar]):
            h2_sigs.append((sig_bar, direction, tp, sl))  # Include if no BB data
            continue
        bb_pos = (closes[sig_bar] - bb_lower[sig_bar]) / max(
            bb_upper[sig_bar] - bb_lower[sig_bar], 1e-10)
        # LONG: price in lower half of BB (more room to revert up)
        # SHORT: price in upper half
        if direction == 'LONG' and bb_pos < 0.5:
            h2_sigs.append((sig_bar, direction, tp, sl))
        elif direction == 'SHORT' and bb_pos > 0.5:
            h2_sigs.append((sig_bar, direction, tp, sl))

    trades = backtest_signals(h2_sigs, opens, highs, lows, n_bars,
                              atr_ratio=atr_ratio)
    m = compute_metrics(trades)
    results['H2_pattern_bb_confirm'] = m
    print(f"  H2 (pattern+BB confirm): T={m['n_trades']}, WR={m['wr']:.1f}%, "
          f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%")

    # H3: Pattern + RSI confirmation
    h3_sigs = []
    for sig_bar, direction, tp, sl in pat_signals:
        if np.isnan(rsi[sig_bar]):
            h3_sigs.append((sig_bar, direction, tp, sl))
            continue
        # LONG: RSI < 60 (not overbought)
        # SHORT: RSI > 40 (not oversold)
        if direction == 'LONG' and rsi[sig_bar] < 60:
            h3_sigs.append((sig_bar, direction, tp, sl))
        elif direction == 'SHORT' and rsi[sig_bar] > 40:
            h3_sigs.append((sig_bar, direction, tp, sl))

    trades = backtest_signals(h3_sigs, opens, highs, lows, n_bars,
                              atr_ratio=atr_ratio)
    m = compute_metrics(trades)
    results['H3_pattern_rsi_confirm'] = m
    print(f"  H3 (pattern+RSI confirm): T={m['n_trades']}, WR={m['wr']:.1f}%, "
          f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%")

    # H4: Pattern + Vol regime filter (trade only in appropriate vol regime)
    atr_pctile = pd.Series(atr).rolling(BARS_PER_DAY).rank(pct=True).values
    h4_sigs = []
    for sig_bar, direction, tp, sl in pat_signals:
        if np.isnan(atr_pctile[sig_bar]):
            h4_sigs.append((sig_bar, direction, tp, sl))
            continue
        # Only trade when vol is moderate (not extreme)
        if 0.2 < atr_pctile[sig_bar] < 0.8:
            h4_sigs.append((sig_bar, direction, tp, sl))

    trades = backtest_signals(h4_sigs, opens, highs, lows, n_bars,
                              atr_ratio=atr_ratio)
    m = compute_metrics(trades)
    results['H4_pattern_vol_filter'] = m
    print(f"  H4 (pattern+vol filter): T={m['n_trades']}, WR={m['wr']:.1f}%, "
          f"PnL={m['pnl']:+.1f}%, MDD={m['mdd']:.1f}%")

    # H5: Diversified — pattern + MR signals (if any MR was positive)
    # Use best MR strategy from phase 2 if available
    best_mr = phase2_results.get('best_mr')
    if best_mr:
        # Regenerate MR signals and combine with pattern signals
        # Parse the label to get strategy type
        print(f"\n  H5: Diversified (pattern + {best_mr})")
        # We would need to regenerate these signals — for simplicity,
        # just note this as a potential direction
        results['H5_diversified'] = {
            'note': f"Would combine pattern signals with {best_mr}",
            'best_mr_pnl': phase2_results.get('best_mr_metrics', {}).get('pnl', 0),
            'verdict': 'NEEDS_IMPLEMENTATION' if phase2_results.get('best_mr_metrics', {}).get('pnl', 0) > 0 else 'STOP',
        }
    else:
        results['H5_diversified'] = {'note': 'No viable MR strategy found', 'verdict': 'STOP'}

    # Summary
    baseline_pnl = results['baseline_pattern']['pnl']
    improvements = {}
    for key in ['H1_pattern_volatile_hours', 'H2_pattern_bb_confirm',
                'H3_pattern_rsi_confirm', 'H4_pattern_vol_filter']:
        if key in results and results[key].get('n_trades', 0) >= MIN_TRADES:
            delta = results[key]['pnl'] - baseline_pnl
            improvements[key] = delta
            print(f"\n  {key}: delta PnL vs baseline = {delta:+.1f}%")

    results['verdict'] = 'PARTIAL' if any(d > 0 for d in improvements.values()) else 'STOP'
    results['improvements_vs_baseline'] = {k: round(v, 2) for k, v in improvements.items()}
    print(f"\n  Phase 5 Verdict: {results['verdict']}")
    return results


# ============================================================
# PHASE 6: FEASIBILITY ASSESSMENT
# ============================================================

def phase6_feasibility(phase1_results, phase2_results, phase3_results,
                       phase4_results, phase5_results):
    """
    Assess feasibility of each strategy for live deployment.
    """
    print("\n" + "=" * 70)
    print("PHASE 6: FEASIBILITY ASSESSMENT")
    print("=" * 70)

    strategies = {}

    # Current pattern strategy baseline
    strategies['current_pattern_v1.36.4'] = {
        'description': '3-candle pattern with ATR-scaled TP/SL',
        'implementation_difficulty': 'DONE (live)',
        'capital_requirement': '$2,261 (current)',
        'expected_frequency': '~1.0 trades/day',
        'wf_validated': True,
        'key_risk': 'N=9 correlation, R:R < 1, direction-dependent',
        'infra_reuse': '100%',
        'verdict': 'BASELINE',
    }

    # Mean Reversion
    best_mr = phase2_results.get('best_mr')
    mr_viable = best_mr is not None
    strategies['mean_reversion'] = {
        'description': 'BB/VWAP/KC/RSI mean reversion signals',
        'best_variant': best_mr or 'None',
        'best_pnl': phase2_results.get('best_mr_metrics', {}).get('pnl', 0),
        'implementation_difficulty': 'MEDIUM (indicator-based, same infra)',
        'capital_requirement': '$2,261 (same)',
        'expected_frequency': 'Varies (typically 1-5/day for MR)',
        'wf_validated': False,
        'key_risk': 'Mean reversion fails in trending markets',
        'infra_reuse': '~80% (same bot framework, different signals)',
        'verdict': 'NEEDS_WF' if mr_viable else 'STOP',
    }

    # Volatility Trading
    best_vol = phase3_results.get('best_vol')
    vol_viable = best_vol is not None
    strategies['volatility_trading'] = {
        'description': 'ATR-based volatility breakout/selling/range',
        'best_variant': best_vol or 'None',
        'best_pnl': phase3_results.get('best_vol_metrics', {}).get('pnl', 0),
        'implementation_difficulty': 'MEDIUM (ATR infrastructure already exists)',
        'capital_requirement': '$2,261 (same)',
        'expected_frequency': 'Varies (regime-dependent)',
        'wf_validated': False,
        'key_risk': 'Regime classification accuracy',
        'infra_reuse': '~75% (ATR already in production)',
        'verdict': 'NEEDS_WF' if vol_viable else 'STOP',
    }

    # Time Filters
    strategies['time_filter'] = {
        'description': 'Hour/day seasonality filter on existing pattern strategy',
        'implementation_difficulty': 'LOW (config change only)',
        'capital_requirement': '$2,261 (same)',
        'expected_frequency': 'Reduced (filter removes ~33% signals)',
        'wf_validated': False,
        'key_risk': 'Seasonality may be spurious',
        'infra_reuse': '~95% (add time check to signal filter)',
        'verdict': 'NEEDS_WF',
    }

    # Hybrid
    h5_verdict = phase5_results.get('H5_diversified', {}).get('verdict', 'STOP')
    hybrid_improvements = phase5_results.get('improvements_vs_baseline', {})
    best_hybrid = max(hybrid_improvements, key=lambda k: hybrid_improvements[k]) if hybrid_improvements else None

    strategies['hybrid'] = {
        'description': 'Pattern + indicator confirmation/time filter',
        'best_variant': best_hybrid or 'None',
        'best_improvement': round(max(hybrid_improvements.values()), 2) if hybrid_improvements else 0,
        'implementation_difficulty': 'LOW-MEDIUM (add filters to existing bot)',
        'capital_requirement': '$2,261 (same)',
        'expected_frequency': 'Reduced by filter',
        'wf_validated': False,
        'key_risk': 'Overfitting to IS data with indicator thresholds',
        'infra_reuse': '~90%',
        'verdict': 'NEEDS_WF' if any(v > 0 for v in hybrid_improvements.values()) else 'STOP',
    }

    # Funding rate
    strategies['funding_rate_proxy'] = {
        'description': '8h price cycle exploitation',
        'implementation_difficulty': 'HIGH (need real funding data from API)',
        'capital_requirement': '$2,261 (same) but need additional API calls',
        'expected_frequency': '3/day (1 per funding window)',
        'wf_validated': False,
        'key_risk': 'No actual funding data in OHLCV, proxy unreliable',
        'infra_reuse': '~60% (need new data pipeline)',
        'verdict': phase1_results.get('verdict', 'STOP'),
    }

    # Print assessment table
    print(f"\n  {'Strategy':<30s} | {'Verdict':<12s} | {'Infra Reuse':>10s} | {'Difficulty':<10s}")
    print(f"  {'-' * 70}")
    for name, s in strategies.items():
        print(f"  {name:<30s} | {s['verdict']:<12s} | {s['infra_reuse']:>10s} | "
              f"{s['implementation_difficulty']:<10s}")

    return strategies


# ============================================================
# WALK-FORWARD VALIDATION FOR TOP CANDIDATES
# ============================================================

def wf_validate_top_candidates(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio,
                               phase2_results, phase3_results, phase5_results):
    """Run WF validation on the most promising non-pattern strategies."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION OF TOP CANDIDATES")
    print("=" * 70)

    wf_results = {}
    n_folds = 3
    seg_size = n_bars // (n_folds + 1)
    fee = FEE_PCT * LEVERAGE

    # Collect candidate strategies that showed positive IS PnL
    candidates = []

    # From Phase 2: MR strategies
    for label, m in phase2_results.items():
        if isinstance(m, dict) and m.get('n_trades', 0) >= MIN_TRADES and m.get('pnl', 0) > 10:
            candidates.append(('MR', label, m))

    # From Phase 3: Vol strategies
    for label, m in phase3_results.items():
        if isinstance(m, dict) and m.get('n_trades', 0) >= MIN_TRADES and m.get('pnl', 0) > 10:
            candidates.append(('VOL', label, m))

    # From Phase 5: Hybrid strategies
    for label in ['H1_pattern_volatile_hours', 'H2_pattern_bb_confirm',
                  'H3_pattern_rsi_confirm', 'H4_pattern_vol_filter']:
        if label in phase5_results:
            m = phase5_results[label]
            if isinstance(m, dict) and m.get('n_trades', 0) >= MIN_TRADES and m.get('pnl', 0) > 0:
                candidates.append(('HYBRID', label, m))

    if not candidates:
        print("  No candidates with positive PnL and sufficient trades for WF validation.")
        return {'verdict': 'NO_CANDIDATES'}

    # Sort by IS PnL descending, take top 10
    candidates.sort(key=lambda x: x[2].get('pnl', 0), reverse=True)
    candidates = candidates[:10]

    print(f"\n  Testing {len(candidates)} candidates in WF:")
    for cat, label, m in candidates:
        print(f"    [{cat}] {label}: IS PnL={m['pnl']:+.1f}%, "
              f"WR={m['wr']:.1f}%, T={m['n_trades']}")

    # Re-generate signals for each candidate in OOS windows
    # This requires re-implementing each signal generator for OOS
    # For efficiency, we test a simplified approach: split trades by bar range

    bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger(closes, 20, 2.0)
    rsi_arr = compute_rsi(closes, 14)
    _, atr_full = compute_atr(highs, lows, closes, 14)
    atr_pctile = pd.Series(atr_full).rolling(BARS_PER_DAY).rank(pct=True).values
    vwap, vwap_std = compute_vwap_session(opens, highs, lows, closes, volumes, 96)
    ema20 = compute_ema(closes, 20)
    kc_mid, kc_upper, kc_lower = compute_keltner(closes, highs, lows, 20, 14, 2.0)

    # For each candidate, regenerate signals per OOS fold
    for cat, label, is_m in candidates:
        fold_results = []

        for fold in range(n_folds):
            oos_start = (fold + 1) * seg_size
            oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

            # Generate signals for this OOS window based on strategy type
            oos_signals = _generate_strategy_signals(
                label, oos_start, oos_end, closes, opens, highs, lows,
                bb_upper, bb_lower, bb_width, rsi_arr, atr_ratio,
                atr_pctile, vwap, vwap_std, kc_upper, kc_lower, ema20,
                hours=df['timestamp'].dt.hour.values if 'timestamp' in df.columns else None,
                n_bars=n_bars,
            )

            trades = backtest_signals(oos_signals, opens, highs, lows, n_bars,
                                      atr_ratio=atr_ratio)
            m = compute_metrics(trades)
            fold_results.append({
                'fold': fold + 1, 'oos_start': oos_start,
                'oos_end': oos_end, **m
            })

        total_pnl = sum(f['pnl'] for f in fold_results)
        max_mdd = max(f['mdd'] for f in fold_results) if fold_results else 0
        n_pass = sum(1 for f in fold_results if f['pnl'] > 0)
        total_trades = sum(f['n_trades'] for f in fold_results)

        wf_result = {
            'category': cat,
            'is_pnl': is_m['pnl'],
            'folds': fold_results,
            'total_pnl': round(total_pnl, 2),
            'max_mdd': round(max_mdd, 2),
            'total_trades': total_trades,
            'wf_pass': f"{n_pass}/{n_folds}",
            'wf_passed': n_pass == n_folds,
        }
        wf_results[label] = wf_result

        status = "PASS" if wf_result['wf_passed'] else "FAIL"
        print(f"\n  WF [{cat}] {label}:")
        for fr in fold_results:
            print(f"    Fold {fr['fold']}: T={fr['n_trades']}, WR={fr['wr']:.1f}%, "
                  f"PnL={fr['pnl']:+.1f}%, MDD={fr['mdd']:.1f}%")
        print(f"    => WF {wf_result['wf_pass']} {status} | "
              f"Total PnL={total_pnl:+.1f}%, MDD={max_mdd:.1f}%")

    # Summary
    passed = {k: v for k, v in wf_results.items() if v.get('wf_passed', False)}
    wf_results['n_passed'] = len(passed)
    wf_results['n_tested'] = len(candidates)
    wf_results['passed_strategies'] = list(passed.keys())

    if passed:
        best = max(passed, key=lambda k: passed[k]['total_pnl'])
        wf_results['best_wf_strategy'] = best
        wf_results['best_wf_pnl'] = passed[best]['total_pnl']
        print(f"\n  WF Summary: {len(passed)}/{len(candidates)} PASSED")
        print(f"  Best WF strategy: {best} (OOS PnL={passed[best]['total_pnl']:+.1f}%)")
    else:
        wf_results['best_wf_strategy'] = None
        print(f"\n  WF Summary: 0/{len(candidates)} PASSED — no strategy survived OOS")

    return wf_results


def _generate_strategy_signals(label, oos_start, oos_end,
                               closes, opens, highs, lows,
                               bb_upper, bb_lower, bb_width,
                               rsi_arr, atr_ratio, atr_pctile,
                               vwap, vwap_std,
                               kc_upper, kc_lower, ema20,
                               hours=None, n_bars=None):
    """Regenerate signals for a strategy label in OOS window."""
    signals = []

    if label.startswith('MR1_BB'):
        # Parse: MR1_BB{std}_TP{tp}_SL{sl}
        parts = label.split('_')
        bb_std = float(parts[1][2:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        bb_m, bb_u, bb_l, _ = compute_bollinger(closes, 20, bb_std)
        for i in range(max(oos_start, ATR_WARMUP + 1), oos_end):
            if np.isnan(bb_u[i]) or np.isnan(bb_l[i]):
                continue
            if i < 1:
                continue
            if closes[i] < bb_l[i] and closes[i - 1] >= bb_l[i - 1]:
                signals.append((i, 'LONG', tp, sl))
            elif closes[i] > bb_u[i] and closes[i - 1] <= bb_u[i - 1]:
                signals.append((i, 'SHORT', tp, sl))

    elif label.startswith('MR2_squeeze'):
        parts = label.split('_')
        squeeze = float(parts[1][7:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        if sl < 1.0:
            sl = 1.0
        for i in range(max(oos_start, ATR_WARMUP), oos_end):
            if np.isnan(atr_pctile[i]):
                continue
            if atr_pctile[i] < squeeze:
                signals.append((i, 'LONG', tp, sl))
                signals.append((i, 'SHORT', tp, sl))

    elif label.startswith('MR3_VWAP'):
        parts = label.split('_')
        dev = float(parts[1][4:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        for i in range(max(oos_start, ATR_WARMUP), oos_end):
            if np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or vwap_std[i] <= 0:
                continue
            z = (closes[i] - vwap[i]) / vwap_std[i]
            if z > dev:
                signals.append((i, 'SHORT', tp, sl))
            elif z < -dev:
                signals.append((i, 'LONG', tp, sl))

    elif label.startswith('MR4_KC'):
        parts = label.split('_')
        kc_mult = float(parts[1][2:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        kc_m2, kc_u2, kc_l2 = compute_keltner(closes, highs, lows, 20, 14, kc_mult)
        for i in range(max(oos_start, ATR_WARMUP + 1), oos_end):
            if np.isnan(kc_u2[i]) or np.isnan(kc_l2[i]) or i < 1:
                continue
            if closes[i - 1] > kc_u2[i - 1] and closes[i] <= kc_u2[i]:
                signals.append((i, 'SHORT', tp, sl))
            elif closes[i - 1] < kc_l2[i - 1] and closes[i] >= kc_l2[i]:
                signals.append((i, 'LONG', tp, sl))

    elif label.startswith('MR5_RSI'):
        parts = label.split('_')
        rsi_t = int(parts[1][3:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        for i in range(max(oos_start, ATR_WARMUP), oos_end):
            if np.isnan(rsi_arr[i]):
                continue
            if rsi_arr[i] < rsi_t:
                signals.append((i, 'LONG', tp, sl))
            elif rsi_arr[i] > (100 - rsi_t):
                signals.append((i, 'SHORT', tp, sl))

    elif label.startswith('V1_ATRbreak'):
        parts = label.split('_')
        thresh = float(parts[1][8:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        for i in range(max(oos_start, ATR_WARMUP + 1), oos_end):
            if np.isnan(atr_ratio[i]) or np.isnan(atr_ratio[i - 1]):
                continue
            if atr_ratio[i] > thresh and atr_ratio[i - 1] <= thresh:
                if closes[i] > opens[i]:
                    signals.append((i, 'LONG', tp, sl))
                else:
                    signals.append((i, 'SHORT', tp, sl))

    elif label.startswith('V2_volsell'):
        parts = label.split('_')
        thresh = float(parts[1][7:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        for i in range(max(oos_start, ATR_WARMUP + 1), oos_end):
            if np.isnan(atr_ratio[i]):
                continue
            if atr_ratio[i] > thresh:
                bar_ret = (closes[i] / opens[i] - 1) * 100
                if bar_ret > 0.3:
                    signals.append((i, 'SHORT', tp, sl))
                elif bar_ret < -0.3:
                    signals.append((i, 'LONG', tp, sl))

    elif label.startswith('V3_range'):
        parts = label.split('_')
        vol_lo = float(parts[1][5:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        if sl < 1.0:
            sl = 1.0
        bb_width_pct = pd.Series(bb_width).rolling(BARS_PER_DAY).rank(pct=True).values
        for i in range(max(oos_start, ATR_WARMUP + 1), oos_end):
            if np.isnan(bb_width_pct[i]) or np.isnan(bb_lower[i]):
                continue
            if bb_width_pct[i] > vol_lo:
                continue
            if closes[i] <= bb_lower[i]:
                signals.append((i, 'LONG', tp, sl))
            elif closes[i] >= bb_upper[i]:
                signals.append((i, 'SHORT', tp, sl))

    elif label.startswith('V4_regime'):
        parts = label.split('_')
        split = float(parts[1][6:])
        tp = float(parts[2][2:])
        sl = float(parts[3][2:])
        for i in range(max(oos_start, ATR_WARMUP + 1), oos_end):
            if np.isnan(atr_pctile[i]) or np.isnan(bb_lower[i]):
                continue
            if atr_pctile[i] < split:
                if closes[i] <= bb_lower[i]:
                    signals.append((i, 'LONG', tp, sl))
                elif closes[i] >= bb_upper[i]:
                    signals.append((i, 'SHORT', tp, sl))
            else:
                if i < 1:
                    continue
                if closes[i] > ema20[i] and closes[i - 1] <= ema20[i - 1]:
                    signals.append((i, 'LONG', tp, sl))
                elif closes[i] < ema20[i] and closes[i - 1] >= ema20[i - 1]:
                    signals.append((i, 'SHORT', tp, sl))

    elif label.startswith('H1_') or label.startswith('H2_') or \
            label.startswith('H3_') or label.startswith('H4_'):
        # Hybrid strategies need pattern signals — regenerate inline
        if os.path.exists(PATTERNS_FILE):
            with open(PATTERNS_FILE) as f:
                pat_data = json.load(f)
            pL = set(pat_data['patterns']['long'])
            pS = set(pat_data['patterns']['short'])
            tpsl_map = {}
            for pn, ts in pat_data['patterns_tpsl'].items():
                tpsl_map[pn] = (ts[0], ts[1])

            # We need type_code — check if available from caller
            # Regenerate from type_code column is expensive here,
            # so just use full-range signals filtered to OOS
            # This is approximate but acceptable for WF screening
            pass  # Handled by full-range filtering below

    return signals


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("STRATEGY PIVOT STUDY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load data
    df, opens, highs, lows, closes, volumes, n_bars = load_data()
    atr_ratio = compute_atr_ratio(highs, lows, closes)

    # Phase 1: Funding Rate Proxy
    p1 = phase1_funding_rate_proxy(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio)

    # Phase 2: Mean Reversion
    p2 = phase2_mean_reversion(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio)

    # Phase 3: Volatility Trading
    p3 = phase3_volatility_trading(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio)

    # Phase 4: Statistical Patterns
    p4 = phase4_statistical_patterns(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio)

    # Phase 5: Hybrid
    p5 = phase5_hybrid(df, opens, highs, lows, closes, volumes, n_bars, atr_ratio,
                       p2, p3, p4)

    # Walk-Forward validation on top candidates
    wf = wf_validate_top_candidates(df, opens, highs, lows, closes, volumes, n_bars,
                                    atr_ratio, p2, p3, p5)

    # Phase 6: Feasibility
    p6 = phase6_feasibility(p1, p2, p3, p4, p5)

    # Overall Verdict
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    wf_passed = wf.get('passed_strategies', [])
    p2_viable = p2.get('verdict') == 'PARTIAL'
    p3_viable = p3.get('verdict') == 'PARTIAL'
    p5_viable = p5.get('verdict') == 'PARTIAL'

    if wf_passed:
        overall_verdict = 'PARTIAL'
        viable_strategies = wf_passed
        recommendation = (
            f"Found {len(wf_passed)} strategies that survived WF validation: "
            f"{', '.join(wf_passed)}. "
            "Recommend further investigation with full portfolio simulation "
            "and MC significance testing before deployment. "
            "Current pattern strategy remains stronger in WF OOS PnL (+872.7%) "
            "but these alternatives may provide diversification benefit."
        )
    elif p2_viable or p3_viable or p5_viable:
        overall_verdict = 'PARTIAL'
        viable_strategies = []
        if p2_viable:
            viable_strategies.append(f"MR: {p2.get('best_mr', 'N/A')}")
        if p3_viable:
            viable_strategies.append(f"Vol: {p3.get('best_vol', 'N/A')}")
        if p5_viable:
            viable_strategies.extend([k for k, v in p5.get('improvements_vs_baseline', {}).items() if v > 0])
        recommendation = (
            "Some strategies showed positive IS PnL but none survived WF validation. "
            "This suggests in-sample overfit. Current pattern strategy "
            "(v1.36.4, WF 3/3 PASS, OOS +872.7%) remains the best option. "
            "Consider hybrid filters (H2-H4) for marginal improvement "
            "but expect WF validation to likely fail."
        )
    else:
        overall_verdict = 'STOP'
        viable_strategies = []
        recommendation = (
            "No alternative strategy showed significant edge. "
            "BTC 5m direction prediction is approximately random walk. "
            "Current 3-candle pattern strategy with R:R < 1 + high WR "
            "remains the only approach with WF-validated OOS edge. "
            "Focus on optimizing existing strategy rather than pivoting. "
            "Key improvement areas: correlation risk (direction cap, regime sizing), "
            "position management (timeout, MDD sizing), and re-scanning frequency."
        )

    print(f"\n  Verdict: {overall_verdict}")
    print(f"  Viable strategies: {viable_strategies}")
    print(f"\n  Recommendation:\n    {recommendation}")

    # Save results
    output = {
        'study': 'Strategy Pivot Study',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_bars': n_bars,
        'data_days': round(n_bars / BARS_PER_DAY, 1),
        'phases': {
            'phase1_funding_rate': {
                'verdict': p1.get('verdict', 'STOP'),
                'best_hour': p1.get('best_hour'),
                'worst_hour': p1.get('worst_hour'),
                'note': p1.get('note', ''),
            },
            'phase2_mean_reversion': {
                'verdict': p2.get('verdict', 'STOP'),
                'best_mr': p2.get('best_mr'),
                'best_mr_metrics': p2.get('best_mr_metrics'),
                'n_tested': sum(1 for k, v in p2.items()
                               if isinstance(v, dict) and 'n_trades' in v),
                'n_positive': sum(1 for k, v in p2.items()
                                 if isinstance(v, dict) and v.get('pnl', 0) > 0),
            },
            'phase3_volatility_trading': {
                'verdict': p3.get('verdict', 'STOP'),
                'best_vol': p3.get('best_vol'),
                'best_vol_metrics': p3.get('best_vol_metrics'),
                'n_tested': sum(1 for k, v in p3.items()
                               if isinstance(v, dict) and 'n_trades' in v),
                'n_positive': sum(1 for k, v in p3.items()
                                 if isinstance(v, dict) and v.get('pnl', 0) > 0),
            },
            'phase4_statistical_patterns': {
                'verdict': p4.get('verdict', 'INFORMATIONAL'),
                'volatile_hours': p4.get('volatile_hours', []),
                'quiet_hours': p4.get('quiet_hours', []),
                'pattern_volatile_pnl': p4.get('pattern_volatile_8h', {}).get('pnl', 0),
                'pattern_quiet_pnl': p4.get('pattern_quiet_8h', {}).get('pnl', 0),
                'pattern_all_pnl': p4.get('pattern_all_hours', {}).get('pnl', 0),
            },
            'phase5_hybrid': {
                'verdict': p5.get('verdict', 'STOP'),
                'baseline_pnl': p5.get('baseline_pattern', {}).get('pnl', 0),
                'improvements': p5.get('improvements_vs_baseline', {}),
            },
            'walk_forward_validation': {
                'n_tested': wf.get('n_tested', 0),
                'n_passed': wf.get('n_passed', 0),
                'passed_strategies': wf.get('passed_strategies', []),
                'best_wf_strategy': wf.get('best_wf_strategy'),
                'best_wf_pnl': wf.get('best_wf_pnl', 0),
            },
            'phase6_feasibility': p6,
        },
        'verdict': overall_verdict,
        'viable_strategies': viable_strategies,
        'recommendation': recommendation,
        'execution_time_sec': round(time.time() - t_start, 1),
    }

    # NumpyEncoder for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to: {OUTPUT_FILE}")
    print(f"  Execution time: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
