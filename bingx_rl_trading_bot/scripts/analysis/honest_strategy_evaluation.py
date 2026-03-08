#!/usr/bin/env python3
"""
Honest Strategy Evaluation — 5-Phase Reality Check

Brutally honest assessment of the current 130-pattern strategy:
  Phase 1: Cost structure analysis (4 cost scenarios)
  Phase 2: Per-trade expected value decomposition
  Phase 3: Edge decay over rolling windows
  Phase 4: Buy & Hold comparison (1x and 3x)
  Phase 5: WR scenario analysis (IS / True OOS / Live)

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE = 3, FEE_PCT * LEVERAGE for notional fees
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - MC: seeds [42, 123, 7], max p < 0.01

Author: Research Agent
Date: 2026-03-01
"""

import os
import sys
import json
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

warnings.filterwarnings('ignore')

# ============================================================
# Constants — matching scanner exactly
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10           # 0.05% entry + 0.05% exit (price space)
SLIPPAGE_PCT = 0.02      # slippage buffer (price space)
SPREAD_PCT = 0.025       # BingX BTC spread ~5bps total, half per side
MAX_BARS = 288            # 24h timeout
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]
BARS_PER_DAY = 288

# ATR scaling defaults
ATR_PERIOD = 14
ATR_WINDOW = 576
ATR_CLAMP_LO = 0.6
ATR_CLAMP_HI = 1.7

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'honest_strategy_evaluation.json')


# ============================================================
# Data Loading
# ============================================================

def load_and_classify(data_file):
    """Load CSV and classify candles using production classify_candle()."""
    df = pd.read_csv(data_file)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW).mean()

    types = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body']
        if pd.isna(avg_b):
            avg_b = 1.0
        ct = classify_candle(row, avg_b)
        types.append(ct.value)
    df['candle_type'] = types
    return df


def load_patterns(patterns_file):
    """Load dynamic patterns JSON."""
    with open(patterns_file) as f:
        data = json.load(f)
    return data


def compute_atr_ratio(highs, lows, closes, atr_period=ATR_PERIOD, window=ATR_WINDOW):
    """ATR / rolling_median(ATR) ratio. Production-matching."""
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
    return ratio


def build_signal_index(types, n):
    """Build index: pattern_name -> list of signal bar indices."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


# ============================================================
# Backtest Engine
# ============================================================

def bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                   opens, highs, lows, n_bars,
                   atr_ratio, clamp_lo=ATR_CLAMP_LO, clamp_hi=ATR_CLAMP_HI,
                   extra_cost_pct=0.0):
    """
    Backtest with ATR-scaled TP/SL. Timeout = DROP.
    extra_cost_pct: additional cost per trade in capital space (already leverage-adjusted).
    Returns list of (entry_bar, exit_bar, pnl_pct, is_win).
    """
    fee = FEE_PCT * LEVERAGE  # 0.30% per trade in capital space
    is_long = (direction == 'LONG')
    trades = []

    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1

        # ATR scaling at signal bar
        if (atr_ratio is not None and idx < len(atr_ratio)
                and not np.isnan(atr_ratio[idx])):
            r = max(clamp_lo, min(clamp_hi, atr_ratio[idx]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE_PCT
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_PCT)

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
                pnl = (eff_tp if dist_tp <= dist_sl else -eff_sl) * LEVERAGE - fee - extra_cost_pct
                is_win = dist_tp <= dist_sl
                trades.append((eb, j, pnl, is_win))
                break
            elif ht:
                pnl = eff_tp * LEVERAGE - fee - extra_cost_pct
                trades.append((eb, j, pnl, True))
                break
            elif hs:
                pnl = -eff_sl * LEVERAGE - fee - extra_cost_pct
                trades.append((eb, j, pnl, False))
                break
        # Timeout → DROP

    return trades


def compute_portfolio_stats(all_trades):
    """
    Compute ADDITIVE portfolio stats from sorted trades.

    Uses additive PnL (sum of per-trade %) for portfolio-level metrics.
    This is the standard for multi-pattern backtests where trades overlap
    and share 1/N sizing. Compound is unrealistic with 5,500+ sequential
    trades on a single equity pool.

    Returns dict with PnL, MDD, WR, PF, Sharpe, trade_count, etc.
    """
    if not all_trades:
        return {
            'total_trades': 0, 'win_rate': 0.0, 'pnl_pct': 0.0,
            'max_drawdown_pct': 0.0, 'profit_factor': 0.0,
            'avg_win_pct': 0.0, 'avg_loss_pct': 0.0,
            'per_trade_expectancy': 0.0, 'sharpe': 0.0,
            'calmar': 0.0,
        }

    # Sort by entry bar
    all_trades.sort(key=lambda x: x[0])

    pnls = [t[2] for t in all_trades]
    wins = sum(1 for t in all_trades if t[3])
    losses = len(all_trades) - wins

    # ADDITIVE equity curve (cumulative sum of per-trade PnL %)
    cum_pnl = np.cumsum(pnls)
    total_pnl = cum_pnl[-1]

    # MDD on additive curve
    peak = 0.0
    max_dd = 0.0
    for cp in cum_pnl:
        if cp > peak:
            peak = cp
        dd = peak - cp
        if dd > max_dd:
            max_dd = dd

    # Win/loss stats
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]
    avg_win = np.mean(win_pnls) if win_pnls else 0.0
    avg_loss = abs(np.mean(loss_pnls)) if loss_pnls else 0.0

    # Profit factor
    gross_profit = sum(win_pnls) if win_pnls else 0.0
    gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0.001
    pf = gross_profit / gross_loss

    wr = wins / len(all_trades) * 100
    expectancy = np.mean(pnls)

    # Daily returns for Sharpe (additive: sum of trade PnLs per day)
    entry_bars = [t[0] for t in all_trades]
    daily_returns = []
    if entry_bars:
        min_bar = min(entry_bars)
        max_bar = max(t[1] for t in all_trades)
        n_days = max(1, (max_bar - min_bar) // BARS_PER_DAY)
        # Group trades by day
        day_pnl = {}
        for t in all_trades:
            day = (t[0] - min_bar) // BARS_PER_DAY
            if day not in day_pnl:
                day_pnl[day] = 0.0
            day_pnl[day] += t[2]
        # Fill zero days
        for d in range(n_days + 1):
            daily_returns.append(day_pnl.get(d, 0.0))

    sharpe = 0.0
    if daily_returns and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)

    # Calmar (annualized additive return / max dd)
    calmar = 0.0
    if max_dd > 0 and entry_bars:
        n_bars_total = max(1, max(t[1] for t in all_trades) - min(t[0] for t in all_trades))
        n_years = n_bars_total / (BARS_PER_DAY * 365)
        if n_years > 0:
            annual_return = total_pnl / n_years
            calmar = annual_return / max_dd

    return {
        'total_trades': len(all_trades),
        'wins': wins,
        'losses': losses,
        'win_rate': round(wr, 2),
        'pnl_pct': round(float(total_pnl), 2),
        'max_drawdown_pct': round(float(max_dd), 2),
        'profit_factor': round(pf, 3),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'per_trade_expectancy': round(expectancy, 4),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
    }


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
        rand_sums = signs @ pnls_arr
        p_vals.append(float(np.mean(rand_sums >= actual)))
    return max(p_vals)


# ============================================================
# Phase 1: Cost Structure Analysis
# ============================================================

def phase1_cost_analysis(pattern_details, signal_index, opens, highs, lows,
                          n_bars, atr_ratio, start_bar, end_bar):
    """Run backtest under 4 cost scenarios."""
    print("\n" + "=" * 70)
    print("PHASE 1: Cost Structure Analysis")
    print("=" * 70)

    # Cost scenarios (in capital space, after leverage)
    # Note: fee is already handled inside bt_signals_atr as FEE_PCT * LEVERAGE = 0.30%
    # So extra_cost_pct is ADDITIONAL cost beyond the base fee
    scenarios = {
        'A_gross': {
            'label': 'Gross (zero cost)',
            'description': 'No fees, no slippage, no spread — pure edge measurement',
            'extra_cost': 0.0,
            'disable_base_fee': True,  # Special: zero ALL costs
        },
        'B_fees_only': {
            'label': 'Fees only',
            'description': f'Commission {FEE_PCT*2:.2f}% round-trip x {LEVERAGE}x leverage = {FEE_PCT*LEVERAGE*2:.2f}% capital',
            'extra_cost': 0.0,
            'disable_base_fee': False,
        },
        'C_fees_slip': {
            'label': 'Fees + slippage',
            'description': f'+ slippage {SLIPPAGE_PCT:.2f}% x 2 sides x {LEVERAGE}x = {SLIPPAGE_PCT*2*LEVERAGE:.2f}% capital',
            'extra_cost': SLIPPAGE_PCT * 2 * LEVERAGE,  # 0.12% in capital space
            'disable_base_fee': False,
        },
        'D_all_costs': {
            'label': 'All costs (realistic)',
            'description': f'+ spread {SPREAD_PCT:.3f}% x 2 x {LEVERAGE}x = {SPREAD_PCT*2*LEVERAGE:.3f}% capital',
            'extra_cost': (SLIPPAGE_PCT * 2 + SPREAD_PCT * 2) * LEVERAGE,  # 0.27% in capital space
            'disable_base_fee': False,
        },
    }

    results = {}

    for scenario_key, scenario in scenarios.items():
        print(f"\n--- Scenario {scenario_key}: {scenario['label']} ---")
        print(f"    {scenario['description']}")

        all_trades = []
        for pkey, pinfo in pattern_details.items():
            pat = pinfo['pattern']
            direction = pinfo['direction']
            tp = pinfo['tp']
            sl = pinfo['sl']

            if pat not in signal_index:
                continue

            # Filter signals to neutral window
            sig_bars = [s for s in signal_index[pat] if start_bar <= s <= end_bar]
            if not sig_bars:
                continue

            if scenario.get('disable_base_fee'):
                # Zero-cost backtest: modify the function to not charge fees
                trades = _bt_zero_cost(sig_bars, direction, tp, sl,
                                       opens, highs, lows, n_bars, atr_ratio)
            else:
                trades = bt_signals_atr(sig_bars, direction, tp, sl,
                                        opens, highs, lows, n_bars,
                                        atr_ratio, extra_cost_pct=scenario['extra_cost'])
            all_trades.extend(trades)

        stats = compute_portfolio_stats(all_trades)
        results[scenario_key] = {
            'label': scenario['label'],
            'description': scenario['description'],
            **{k: v for k, v in stats.items() if k != 'equity_curve'},
        }

        print(f"    Trades: {stats['total_trades']}, WR: {stats['win_rate']:.1f}%")
        print(f"    PnL: {stats['pnl_pct']:+.1f}%, MDD: {stats['max_drawdown_pct']:.1f}%")
        print(f"    PF: {stats['profit_factor']:.2f}, E[trade]: {stats['per_trade_expectancy']:+.4f}%")
        print(f"    AvgWin: {stats['avg_win_pct']:.3f}%, AvgLoss: {stats['avg_loss_pct']:.3f}%")

    # Edge erosion summary (per-trade EV is the meaningful metric)
    print("\n--- Cost Impact Summary (per-trade EV) ---")
    gross_ev = results['A_gross']['per_trade_expectancy']
    for k in ['B_fees_only', 'C_fees_slip', 'D_all_costs']:
        cost_ev = results[k]['per_trade_expectancy']
        erosion = gross_ev - cost_ev
        pct_erosion = (erosion / gross_ev * 100) if gross_ev > 0 else 0
        print(f"    {k}: E[trade] {cost_ev:+.4f}% (cost: {erosion:.4f}%, {pct_erosion:.0f}% of gross)")

    return results


def _bt_zero_cost(signal_bars, direction, tp_pct, sl_pct,
                  opens, highs, lows, n_bars, atr_ratio):
    """Zero-cost backtest (no fees, but ATR scaling + slippage buffer still applied to TP/SL distances)."""
    is_long = (direction == 'LONG')
    trades = []

    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1

        if (atr_ratio is not None and idx < len(atr_ratio)
                and not np.isnan(atr_ratio[idx])):
            r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[idx]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE_PCT
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_PCT)

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
                pnl = (eff_tp if dist_tp <= dist_sl else -eff_sl) * LEVERAGE
                is_win = dist_tp <= dist_sl
                trades.append((eb, j, pnl, is_win))
                break
            elif ht:
                trades.append((eb, j, eff_tp * LEVERAGE, True))
                break
            elif hs:
                trades.append((eb, j, -eff_sl * LEVERAGE, False))
                break

    return trades


# ============================================================
# Phase 2: Per-Trade Expected Value
# ============================================================

def phase2_expected_value(pattern_details, signal_index, opens, highs, lows,
                           n_bars, atr_ratio, start_bar, end_bar):
    """Compute per-pattern and portfolio expected value with full costs."""
    print("\n" + "=" * 70)
    print("PHASE 2: Per-Trade Expected Value Decomposition")
    print("=" * 70)

    # Full realistic cost per trade in capital space
    # Fee: 0.30% (already in bt engine)
    # Additional: slippage + spread
    extra_cost = (SLIPPAGE_PCT * 2 + SPREAD_PCT * 2) * LEVERAGE

    pattern_evs = []
    total_trades_all = 0
    total_weighted_ev = 0.0

    for pkey, pinfo in pattern_details.items():
        pat = pinfo['pattern']
        direction = pinfo['direction']
        tp = pinfo['tp']
        sl = pinfo['sl']

        if pat not in signal_index:
            continue

        sig_bars = [s for s in signal_index[pat] if start_bar <= s <= end_bar]
        if not sig_bars:
            continue

        trades = bt_signals_atr(sig_bars, direction, tp, sl,
                                opens, highs, lows, n_bars,
                                atr_ratio, extra_cost_pct=extra_cost)
        if not trades:
            continue

        pnls = [t[2] for t in trades]
        wins = sum(1 for t in trades if t[3])
        wr = wins / len(trades) * 100
        win_pnls = [p for p in pnls if p > 0]
        loss_pnls = [p for p in pnls if p <= 0]
        avg_win = np.mean(win_pnls) if win_pnls else 0.0
        avg_loss = abs(np.mean(loss_pnls)) if loss_pnls else 0.0

        ev = np.mean(pnls)
        baseline_wr = pinfo.get('baseline_wr', sl / (tp + sl) * 100)
        edge_over_rw = wr - baseline_wr

        pattern_evs.append({
            'pattern': pkey,
            'direction': direction,
            'tp': tp,
            'sl': sl,
            'rr_ratio': round(tp / sl, 3),
            'trades': len(trades),
            'wr': round(wr, 1),
            'baseline_wr': round(baseline_wr, 1),
            'edge_over_random': round(edge_over_rw, 1),
            'avg_win': round(avg_win, 3),
            'avg_loss': round(avg_loss, 3),
            'ev_per_trade': round(ev, 4),
            'is_positive_ev': ev > 0,
        })

        total_trades_all += len(trades)
        total_weighted_ev += ev * len(trades)

    # Sort by EV
    pattern_evs.sort(key=lambda x: x['ev_per_trade'], reverse=True)

    positive_ev = sum(1 for p in pattern_evs if p['is_positive_ev'])
    negative_ev = len(pattern_evs) - positive_ev
    weighted_ev = total_weighted_ev / total_trades_all if total_trades_all > 0 else 0.0

    print(f"\nTotal patterns with trades: {len(pattern_evs)}")
    print(f"Positive EV patterns: {positive_ev} ({positive_ev/len(pattern_evs)*100:.0f}%)")
    print(f"Negative EV patterns: {negative_ev} ({negative_ev/len(pattern_evs)*100:.0f}%)")
    print(f"Trade-weighted portfolio EV: {weighted_ev:+.4f}% per trade")
    print(f"Total trades: {total_trades_all}")

    # EV distribution
    evs = [p['ev_per_trade'] for p in pattern_evs]
    print(f"\nEV distribution:")
    print(f"  Min:    {min(evs):+.4f}%")
    print(f"  Q25:    {np.percentile(evs, 25):+.4f}%")
    print(f"  Median: {np.median(evs):+.4f}%")
    print(f"  Q75:    {np.percentile(evs, 75):+.4f}%")
    print(f"  Max:    {max(evs):+.4f}%")

    # Top 5 and Bottom 5
    print(f"\nTop 5 EV patterns:")
    for p in pattern_evs[:5]:
        print(f"  {p['pattern']}: EV={p['ev_per_trade']:+.4f}%, WR={p['wr']:.0f}%, "
              f"trades={p['trades']}, edge={p['edge_over_random']:+.1f}pp")

    print(f"\nBottom 5 EV patterns:")
    for p in pattern_evs[-5:]:
        print(f"  {p['pattern']}: EV={p['ev_per_trade']:+.4f}%, WR={p['wr']:.0f}%, "
              f"trades={p['trades']}, edge={p['edge_over_random']:+.1f}pp")

    # Break-even WR calculation for mean R:R
    mean_tp_lev = np.mean([p['avg_win'] for p in pattern_evs if p['avg_win'] > 0])
    mean_sl_lev = np.mean([p['avg_loss'] for p in pattern_evs if p['avg_loss'] > 0])
    # E[trade] = WR * avg_win - (1-WR) * avg_loss = 0
    # WR = avg_loss / (avg_win + avg_loss)
    breakeven_wr = mean_sl_lev / (mean_tp_lev + mean_sl_lev) * 100 if (mean_tp_lev + mean_sl_lev) > 0 else 50.0

    print(f"\nMean AvgWin (leverage-included): {mean_tp_lev:.3f}%")
    print(f"Mean AvgLoss (leverage-included): {mean_sl_lev:.3f}%")
    print(f"Break-even WR (for mean R:R): {breakeven_wr:.1f}%")

    return {
        'total_patterns_with_trades': len(pattern_evs),
        'positive_ev_patterns': positive_ev,
        'negative_ev_patterns': negative_ev,
        'weighted_portfolio_ev': round(weighted_ev, 4),
        'total_trades': total_trades_all,
        'ev_distribution': {
            'min': round(min(evs), 4),
            'q25': round(float(np.percentile(evs, 25)), 4),
            'median': round(float(np.median(evs)), 4),
            'q75': round(float(np.percentile(evs, 75)), 4),
            'max': round(max(evs), 4),
        },
        'breakeven_wr': round(breakeven_wr, 1),
        'mean_avg_win_lev': round(mean_tp_lev, 3),
        'mean_avg_loss_lev': round(mean_sl_lev, 3),
        'pattern_details': pattern_evs[:10] + pattern_evs[-10:],  # Top/bottom 10
        'mc_portfolio_pvalue': None,  # filled below
    }


# ============================================================
# Phase 3: Edge Decay
# ============================================================

def phase3_edge_decay(df, pattern_details, atr_ratio, start_bar, end_bar):
    """Rolling window edge decay analysis."""
    print("\n" + "=" * 70)
    print("PHASE 3: Edge Decay Over Time")
    print("=" * 70)

    types = df['candle_type'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)

    # Parameters
    is_window = 60 * BARS_PER_DAY   # 60 days IS
    oos_window = 60 * BARS_PER_DAY  # 60 days OOS
    step = 30 * BARS_PER_DAY        # 30 day step

    extra_cost = (SLIPPAGE_PCT * 2 + SPREAD_PCT * 2) * LEVERAGE

    windows = []
    window_id = 0

    current_start = start_bar
    while current_start + is_window + oos_window <= end_bar:
        is_end = current_start + is_window
        oos_end = is_end + oos_window

        # IS: discover patterns (just use the known 130 patterns but test IS WR)
        # OOS: test performance
        is_signal_index = {}
        oos_signal_index = {}

        for i in range(max(2, current_start), min(oos_end, n_bars)):
            pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
            if i < is_end:
                if pat not in is_signal_index:
                    is_signal_index[pat] = []
                is_signal_index[pat].append(i)
            else:
                if pat not in oos_signal_index:
                    oos_signal_index[pat] = []
                oos_signal_index[pat].append(i)

        # Test each pattern in OOS
        oos_trades = []
        is_trades = []
        patterns_tested = 0

        for pkey, pinfo in pattern_details.items():
            pat = pinfo['pattern']
            direction = pinfo['direction']
            tp = pinfo['tp']
            sl = pinfo['sl']

            # IS performance
            is_sigs = is_signal_index.get(pat, [])
            if is_sigs:
                is_t = bt_signals_atr(is_sigs, direction, tp, sl,
                                      opens, highs, lows, n_bars,
                                      atr_ratio, extra_cost_pct=extra_cost)
                is_trades.extend(is_t)

            # OOS performance
            oos_sigs = oos_signal_index.get(pat, [])
            if oos_sigs:
                oos_t = bt_signals_atr(oos_sigs, direction, tp, sl,
                                       opens, highs, lows, n_bars,
                                       atr_ratio, extra_cost_pct=extra_cost)
                oos_trades.extend(oos_t)
                patterns_tested += 1

        # Compute stats
        is_pnls = [t[2] for t in is_trades]
        oos_pnls = [t[2] for t in oos_trades]

        is_ev = np.mean(is_pnls) if is_pnls else 0.0
        oos_ev = np.mean(oos_pnls) if oos_pnls else 0.0
        is_wr = sum(1 for t in is_trades if t[3]) / len(is_trades) * 100 if is_trades else 0.0
        oos_wr = sum(1 for t in oos_trades if t[3]) / len(oos_trades) * 100 if oos_trades else 0.0

        is_day_start = current_start // BARS_PER_DAY
        oos_day_start = is_end // BARS_PER_DAY

        window_result = {
            'window_id': window_id,
            'is_start_day': is_day_start,
            'oos_start_day': oos_day_start,
            'is_trades': len(is_trades),
            'oos_trades': len(oos_trades),
            'is_wr': round(is_wr, 1),
            'oos_wr': round(oos_wr, 1),
            'is_ev': round(is_ev, 4),
            'oos_ev': round(oos_ev, 4),
            'oos_positive_ev': oos_ev > 0,
            'patterns_in_oos': patterns_tested,
        }
        windows.append(window_result)

        print(f"  Window {window_id}: IS d{is_day_start}-d{oos_day_start} | "
              f"OOS d{oos_day_start}-d{oos_day_start+60} | "
              f"IS EV={is_ev:+.4f}% WR={is_wr:.0f}% ({len(is_trades)}t) | "
              f"OOS EV={oos_ev:+.4f}% WR={oos_wr:.0f}% ({len(oos_trades)}t)")

        window_id += 1
        current_start += step

    # Summary
    positive_windows = sum(1 for w in windows if w['oos_positive_ev'])
    total_windows = len(windows)
    pct_positive = positive_windows / total_windows * 100 if total_windows > 0 else 0

    oos_evs = [w['oos_ev'] for w in windows if w['oos_trades'] > 0]
    mean_oos_ev = np.mean(oos_evs) if oos_evs else 0.0

    # Edge decay: correlation of OOS EV with time
    if len(oos_evs) >= 3:
        x = np.arange(len(oos_evs))
        slope, intercept = np.polyfit(x, oos_evs, 1)
        # Half-life: when does EV reach half of initial?
        if slope < 0 and oos_evs[0] > 0:
            half_ev = oos_evs[0] / 2
            # y = slope * x + intercept => x = (y - intercept) / slope
            half_x = (half_ev - intercept) / slope
            halflife_days = half_x * 30  # Each step is 30 days
        else:
            halflife_days = None
    else:
        slope = 0.0
        halflife_days = None

    print(f"\n  Summary:")
    print(f"    Windows: {total_windows}")
    print(f"    OOS positive EV: {positive_windows}/{total_windows} ({pct_positive:.0f}%)")
    print(f"    Mean OOS EV: {mean_oos_ev:+.4f}%")
    print(f"    EV trend slope: {slope:+.6f} per window")
    if halflife_days is not None:
        print(f"    Edge half-life: ~{halflife_days:.0f} days")
    else:
        print(f"    Edge half-life: N/A (no decay detected or insufficient data)")

    return {
        'windows': windows,
        'total_windows': total_windows,
        'positive_ev_windows': positive_windows,
        'pct_positive_ev': round(pct_positive, 1),
        'mean_oos_ev': round(mean_oos_ev, 4),
        'ev_trend_slope': round(slope, 6),
        'edge_halflife_days': round(halflife_days, 0) if halflife_days else None,
    }


# ============================================================
# Phase 4: Buy & Hold Comparison
# ============================================================

def phase4_vs_buyhold(df, all_trades_full_cost, start_bar, end_bar):
    """Compare strategy against BTC Buy & Hold (1x and 3x)."""
    print("\n" + "=" * 70)
    print("PHASE 4: Strategy vs Buy & Hold Comparison")
    print("=" * 70)

    opens = df['open'].values
    closes = df['close'].values

    # BTC price data for the period
    start_price = opens[start_bar]
    end_price = closes[end_bar]
    btc_return = (end_price / start_price - 1) * 100
    n_bars_period = end_bar - start_bar
    n_days = n_bars_period / BARS_PER_DAY
    n_years = n_days / 365

    # BTC B&H 1x
    bh_1x_return = btc_return
    bh_1x_annual = ((1 + btc_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # BTC B&H 3x (simplified: 3x daily return, rebalanced daily)
    daily_closes = []
    for d in range(int(n_days) + 1):
        bar_idx = start_bar + d * BARS_PER_DAY
        if bar_idx < len(closes):
            daily_closes.append(closes[bar_idx])

    bh_3x_equity = 100.0
    bh_3x_curve = [100.0]
    bh_1x_equity_curve = [100.0]
    daily_returns_bh = []
    daily_returns_3x = []

    for i in range(1, len(daily_closes)):
        daily_ret = (daily_closes[i] / daily_closes[i - 1] - 1)
        daily_returns_bh.append(daily_ret * 100)
        daily_returns_3x.append(daily_ret * 3 * 100)
        bh_3x_equity *= (1 + daily_ret * 3)
        bh_3x_curve.append(bh_3x_equity)
        bh_1x_eq = 100 * (daily_closes[i] / daily_closes[0])
        bh_1x_equity_curve.append(bh_1x_eq)

    bh_3x_return = (bh_3x_equity / 100 - 1) * 100
    bh_3x_annual = ((bh_3x_equity / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # BH MDD
    bh_1x_mdd = _compute_mdd(bh_1x_equity_curve)
    bh_3x_mdd = _compute_mdd(bh_3x_curve)

    # BH Sharpe
    bh_1x_sharpe = _compute_sharpe(daily_returns_bh)
    bh_3x_sharpe = _compute_sharpe(daily_returns_3x)

    # Strategy stats (additive PnL — sum of per-trade % on 1/N=11.1% sizing)
    strat_stats = compute_portfolio_stats(all_trades_full_cost)
    # Scale additive PnL by 1/N sizing to get portfolio-level return
    N_POSITIONS = 9
    sizing = 1.0 / N_POSITIONS  # 11.1% per trade
    strat_return_raw = strat_stats['pnl_pct']  # raw sum of trade PnLs (as if 100% sizing)
    strat_return = strat_return_raw * sizing  # scaled to portfolio level
    strat_mdd = strat_stats['max_drawdown_pct'] * sizing  # approx MDD scaling
    strat_annual = strat_return / n_years if n_years > 0 else 0
    strat_sharpe = strat_stats['sharpe']  # Sharpe is ratio, scale-invariant

    # Calmar ratios
    bh_1x_calmar = bh_1x_annual / bh_1x_mdd if bh_1x_mdd > 0 else 0
    bh_3x_calmar = bh_3x_annual / bh_3x_mdd if bh_3x_mdd > 0 else 0
    strat_calmar = strat_annual / strat_mdd if strat_mdd > 0 else 0

    print(f"\nPeriod: {n_days:.0f} days ({n_years:.2f} years)")
    print(f"BTC: {start_price:.0f} -> {end_price:.0f}")
    print(f"")
    print(f"{'Metric':<25} {'Strategy':>12} {'BH 1x':>12} {'BH 3x':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'Total Return %':<25} {strat_return:>+11.1f}% {bh_1x_return:>+11.1f}% {bh_3x_return:>+11.1f}%")
    print(f"{'Annual Return %':<25} {strat_annual:>+11.1f}% {bh_1x_annual:>+11.1f}% {bh_3x_annual:>+11.1f}%")
    print(f"{'Max Drawdown %':<25} {strat_mdd:>11.1f}% {bh_1x_mdd:>11.1f}% {bh_3x_mdd:>11.1f}%")
    print(f"{'Sharpe (ann.)':<25} {strat_sharpe:>12.3f} {bh_1x_sharpe:>12.3f} {bh_3x_sharpe:>12.3f}")
    print(f"{'Calmar':<25} {strat_calmar:>12.3f} {bh_1x_calmar:>12.3f} {bh_3x_calmar:>12.3f}")
    print(f"{'Trades':<25} {strat_stats['total_trades']:>12} {'N/A':>12} {'N/A':>12}")
    print(f"{'Win Rate':<25} {strat_stats['win_rate']:>11.1f}% {'N/A':>12} {'N/A':>12}")

    # Strategy alpha over B&H
    alpha_1x = strat_return - bh_1x_return
    alpha_3x = strat_return - bh_3x_return

    print(f"\nStrategy alpha vs BH 1x: {alpha_1x:+.1f}%")
    print(f"Strategy alpha vs BH 3x: {alpha_3x:+.1f}%")

    return {
        'period_days': round(n_days, 1),
        'period_years': round(n_years, 2),
        'btc_start_price': round(start_price, 0),
        'btc_end_price': round(end_price, 0),
        'strategy': {
            'total_return': round(strat_return, 1),
            'annual_return': round(strat_annual, 1),
            'max_drawdown': round(strat_mdd, 1),
            'sharpe': round(strat_sharpe, 3),
            'calmar': round(strat_calmar, 3),
            'trades': strat_stats['total_trades'],
            'win_rate': strat_stats['win_rate'],
        },
        'buyhold_1x': {
            'total_return': round(bh_1x_return, 1),
            'annual_return': round(bh_1x_annual, 1),
            'max_drawdown': round(bh_1x_mdd, 1),
            'sharpe': round(bh_1x_sharpe, 3),
            'calmar': round(bh_1x_calmar, 3),
        },
        'buyhold_3x': {
            'total_return': round(bh_3x_return, 1),
            'annual_return': round(bh_3x_annual, 1),
            'max_drawdown': round(bh_3x_mdd, 1),
            'sharpe': round(bh_3x_sharpe, 3),
            'calmar': round(bh_3x_calmar, 3),
        },
        'alpha_vs_bh1x': round(alpha_1x, 1),
        'alpha_vs_bh3x': round(alpha_3x, 1),
    }


def _compute_mdd(equity_curve):
    """Max drawdown from equity curve list."""
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def _compute_sharpe(daily_returns):
    """Annualized Sharpe from daily returns (in %)."""
    if not daily_returns or np.std(daily_returns) == 0:
        return 0.0
    return round(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365), 3)


# ============================================================
# Phase 5: Scenario Analysis
# ============================================================

def phase5_scenarios(pattern_details, signal_index, opens, highs, lows,
                     n_bars, atr_ratio, start_bar, end_bar):
    """WR scenario analysis: IS, True OOS, Live, Break-even."""
    print("\n" + "=" * 70)
    print("PHASE 5: WR Scenario Analysis")
    print("=" * 70)

    extra_cost = (SLIPPAGE_PCT * 2 + SPREAD_PCT * 2) * LEVERAGE

    # Collect all full-cost trades
    all_trades = []
    for pkey, pinfo in pattern_details.items():
        pat = pinfo['pattern']
        direction = pinfo['direction']
        tp = pinfo['tp']
        sl = pinfo['sl']
        if pat not in signal_index:
            continue
        sig_bars = [s for s in signal_index[pat] if start_bar <= s <= end_bar]
        if not sig_bars:
            continue
        trades = bt_signals_atr(sig_bars, direction, tp, sl,
                                opens, highs, lows, n_bars,
                                atr_ratio, extra_cost_pct=extra_cost)
        all_trades.extend(trades)

    all_trades.sort(key=lambda x: x[0])
    pnls = [t[2] for t in all_trades]
    wins = sum(1 for t in all_trades if t[3])
    actual_wr = wins / len(all_trades) * 100 if all_trades else 0

    # Measured values from backtest
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]
    avg_win = np.mean(win_pnls) if win_pnls else 0.0
    avg_loss = abs(np.mean(loss_pnls)) if loss_pnls else 0.0

    n_days = (end_bar - start_bar) / BARS_PER_DAY
    trades_per_day = len(all_trades) / n_days if n_days > 0 else 0

    # Break-even WR
    breakeven_wr = avg_loss / (avg_win + avg_loss) * 100 if (avg_win + avg_loss) > 0 else 50.0

    scenarios = {
        'best_case_IS': {
            'label': f'Best case (IS backtest WR={actual_wr:.1f}%)',
            'wr': actual_wr,
        },
        'true_oos': {
            'label': 'True OOS (WF OOS avg WR=88.8%)',
            'wr': 88.8,
        },
        'realistic_oos': {
            'label': 'Realistic OOS (production true_oos study WR=80.6%)',
            'wr': 80.6,
        },
        'live_actual': {
            'label': 'Live actual (114 trades, WR=52.2%)',
            'wr': 52.2,
        },
        'breakeven': {
            'label': f'Break-even (WR={breakeven_wr:.1f}%)',
            'wr': breakeven_wr,
        },
    }

    print(f"\nMeasured from backtest (full costs, N=1, compound):")
    print(f"  AvgWin:  {avg_win:+.3f}% (capital space, post-fees)")
    print(f"  AvgLoss: {avg_loss:.3f}%")
    print(f"  R:R (capital): {avg_win/avg_loss:.3f}" if avg_loss > 0 else "  R:R: N/A")
    print(f"  Trades/day: {trades_per_day:.2f}")
    print(f"  Break-even WR: {breakeven_wr:.1f}%")

    # Real sizing: each trade uses 1/N of equity (N=9)
    N_POSITIONS = 9
    sizing = 1.0 / N_POSITIONS  # 11.1%

    results = {}
    for skey, scenario in scenarios.items():
        wr = scenario['wr']
        # Expected per-trade PnL (on the trade's allocated capital)
        ev_full = (wr / 100) * avg_win - (1 - wr / 100) * avg_loss
        # Portfolio-level EV: scale by sizing
        ev_portfolio = ev_full * sizing
        # Daily portfolio EV = trades_per_day * ev_portfolio
        daily_ev = trades_per_day * ev_portfolio
        # Monthly additive
        monthly_return = daily_ev * 30
        # Annual additive
        annual_return = daily_ev * 365

        results[skey] = {
            'label': scenario['label'],
            'wr': round(wr, 1),
            'ev_per_trade_full': round(ev_full, 4),
            'ev_per_trade_portfolio': round(ev_portfolio, 4),
            'trades_per_day': round(trades_per_day, 2),
            'daily_portfolio_ev': round(daily_ev, 4),
            'monthly_return_additive': round(monthly_return, 1),
            'annual_return_additive': round(annual_return, 1),
        }

        status = "PROFITABLE" if ev_full > 0 else "LOSING"
        print(f"\n  {scenario['label']}:")
        print(f"    E[trade]: {ev_full:+.4f}% (full slot), {ev_portfolio:+.4f}% (portfolio)")
        print(f"    Daily portfolio EV: {daily_ev:+.3f}%  [{status}]")
        print(f"    Monthly (additive): {monthly_return:+.1f}%")
        print(f"    Annual (additive): {annual_return:+.1f}%")

    # Edge margin analysis
    print(f"\n--- Edge Margin Analysis ---")
    margin_over_breakeven_is = actual_wr - breakeven_wr
    margin_over_breakeven_oos = 88.8 - breakeven_wr
    margin_over_breakeven_live = 52.2 - breakeven_wr

    print(f"  Break-even WR: {breakeven_wr:.1f}%")
    print(f"  IS WR margin:   {margin_over_breakeven_is:+.1f}pp")
    print(f"  WF OOS margin:  {margin_over_breakeven_oos:+.1f}pp")
    print(f"  Live margin:    {margin_over_breakeven_live:+.1f}pp")

    if margin_over_breakeven_live < 0:
        print(f"\n  *** WARNING: Live WR is BELOW break-even by {abs(margin_over_breakeven_live):.1f}pp ***")
        print(f"  *** Strategy is LOSING MONEY at live WR ***")

    return {
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'rr_capital_space': round(avg_win / avg_loss, 3) if avg_loss > 0 else 0,
        'breakeven_wr': round(breakeven_wr, 1),
        'trades_per_day': round(trades_per_day, 2),
        'scenarios': results,
        'edge_margins': {
            'is_margin_pp': round(margin_over_breakeven_is, 1),
            'wf_oos_margin_pp': round(margin_over_breakeven_oos, 1),
            'live_margin_pp': round(margin_over_breakeven_live, 1),
        },
    }


# ============================================================
# Verdict
# ============================================================

def generate_verdict(p1, p2, p3, p4, p5):
    """Generate honest verdict from all phases."""
    print("\n" + "=" * 70)
    print("VERDICT: Honest Strategy Assessment")
    print("=" * 70)

    # 1. Edge after costs
    full_cost_ev = p2['weighted_portfolio_ev']
    edge_positive = full_cost_ev > 0

    # 2. Sustainable?
    pct_positive_oos = p3.get('pct_positive_ev', 0)
    halflife = p3.get('edge_halflife_days')
    sustainable = pct_positive_oos >= 60  # At least 60% of windows profitable

    # 3. vs Buy & Hold
    alpha_1x = p4['alpha_vs_bh1x']
    alpha_3x = p4['alpha_vs_bh3x']
    beats_bh1x = alpha_1x > 0
    beats_bh3x = alpha_3x > 0

    # 4. Live viability
    live_margin = p5['edge_margins']['live_margin_pp']
    live_viable = live_margin > 0

    # 5. Break-even buffer
    breakeven_wr = p5['breakeven_wr']
    oos_wr = 88.8
    buffer = oos_wr - breakeven_wr

    print(f"\n1. EDGE AFTER ALL COSTS:")
    print(f"   Portfolio E[trade] = {full_cost_ev:+.4f}% {'(POSITIVE)' if edge_positive else '(NEGATIVE)'}")
    print(f"   Positive EV patterns: {p2['positive_ev_patterns']}/{p2['total_patterns_with_trades']}")

    print(f"\n2. SUSTAINABILITY:")
    print(f"   OOS positive windows: {p3.get('positive_ev_windows', 'N/A')}/{p3.get('total_windows', 'N/A')} ({pct_positive_oos:.0f}%)")
    print(f"   Edge half-life: {halflife if halflife else 'N/A'} days")
    print(f"   {'SUSTAINABLE' if sustainable else 'QUESTIONABLE'}")

    print(f"\n3. VS BUY & HOLD:")
    print(f"   Alpha vs BH 1x: {alpha_1x:+.1f}% {'(BEATS)' if beats_bh1x else '(LOSES)'}")
    print(f"   Alpha vs BH 3x: {alpha_3x:+.1f}% {'(BEATS)' if beats_bh3x else '(LOSES)'}")

    print(f"\n4. LIVE VIABILITY:")
    print(f"   Break-even WR: {breakeven_wr:.1f}%")
    print(f"   WF OOS WR: {oos_wr:.1f}% (buffer: {buffer:+.1f}pp)")
    print(f"   Live WR: 52.2% (margin: {live_margin:+.1f}pp)")
    if not live_viable:
        print(f"   *** LIVE WR IS BELOW BREAKEVEN — LOSING MONEY ***")

    print(f"\n5. COST BURDEN:")
    gross_ev = p1.get('A_gross', {}).get('per_trade_expectancy', 0)
    net_ev = p1.get('D_all_costs', {}).get('per_trade_expectancy', 0)
    if gross_ev > 0:
        cost_burden = (gross_ev - net_ev) / gross_ev * 100
        print(f"   Gross E[trade]: {gross_ev:+.4f}%")
        print(f"   Net E[trade] (all costs): {net_ev:+.4f}%")
        print(f"   Cost erosion: {cost_burden:.0f}% of gross edge consumed by costs")
    else:
        print(f"   Gross E[trade]: {gross_ev:+.4f}% (negative even before costs)")

    # Recommendation
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION:")
    if edge_positive and sustainable and beats_bh1x:
        if live_viable:
            rec = "CONTINUE — Edge exists, sustainable, beats B&H, live-viable"
        else:
            rec = "CAUTION — Backtest edge exists but live WR is below breakeven. Investigate execution gap."
    elif edge_positive and not sustainable:
        rec = "REDUCE — Edge exists but decays rapidly. Shorten rescan interval, reduce position size."
    elif not edge_positive:
        rec = "STOP — No edge after realistic costs. Strategy is a net loser."
    else:
        rec = "MONITOR — Mixed signals. Continue with reduced size and frequent monitoring."

    print(f"   {rec}")
    print(f"{'='*70}")

    return {
        'edge_after_costs_positive': edge_positive,
        'ev_per_trade': full_cost_ev,
        'sustainable': sustainable,
        'pct_positive_oos_windows': round(pct_positive_oos, 1),
        'edge_halflife_days': halflife,
        'beats_buyhold_1x': beats_bh1x,
        'beats_buyhold_3x': beats_bh3x,
        'alpha_vs_bh1x': alpha_1x,
        'alpha_vs_bh3x': alpha_3x,
        'live_viable': live_viable,
        'breakeven_wr': breakeven_wr,
        'oos_buffer_pp': round(buffer, 1),
        'live_margin_pp': round(live_margin, 1),
        'recommendation': rec,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("HONEST STRATEGY EVALUATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load data
    print("\nLoading and classifying data...")
    df = load_and_classify(DATA_FILE)
    print(f"  Loaded {len(df)} bars")

    # Load patterns
    pdata = load_patterns(PATTERNS_FILE)
    pattern_details = pdata.get('pattern_details', {})
    neutral = pdata.get('neutral_window', {})
    start_bar = neutral.get('start_bar', 0)
    end_bar = neutral.get('end_bar', len(df) - 1)

    print(f"  Patterns: {len(pattern_details)}")
    print(f"  Neutral window: bar {start_bar} to {end_bar} ({(end_bar-start_bar)/BARS_PER_DAY:.0f} days)")
    print(f"  Start price: {neutral.get('start_price', 'N/A')}, End price: {neutral.get('end_price', 'N/A')}")
    print(f"  Drift: {neutral.get('drift_pct', 'N/A')}%")

    # Prepare arrays
    types = df['candle_type'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)

    # Build signal index
    signal_index = build_signal_index(types, n_bars)

    # ATR ratio
    print("  Computing ATR ratios...")
    atr_ratio = compute_atr_ratio(highs, lows, closes)

    # ========================================
    # Run all phases
    # ========================================

    # Phase 1: Cost analysis
    p1 = phase1_cost_analysis(pattern_details, signal_index, opens, highs, lows,
                               n_bars, atr_ratio, start_bar, end_bar)

    # Phase 2: Expected value
    p2 = phase2_expected_value(pattern_details, signal_index, opens, highs, lows,
                                n_bars, atr_ratio, start_bar, end_bar)

    # MC test on full-cost portfolio
    extra_cost = (SLIPPAGE_PCT * 2 + SPREAD_PCT * 2) * LEVERAGE
    all_trades_full = []
    for pkey, pinfo in pattern_details.items():
        pat = pinfo['pattern']
        direction = pinfo['direction']
        tp = pinfo['tp']
        sl = pinfo['sl']
        if pat not in signal_index:
            continue
        sig_bars = [s for s in signal_index[pat] if start_bar <= s <= end_bar]
        if not sig_bars:
            continue
        trades = bt_signals_atr(sig_bars, direction, tp, sl,
                                opens, highs, lows, n_bars,
                                atr_ratio, extra_cost_pct=extra_cost)
        all_trades_full.extend(trades)
    all_trades_full.sort(key=lambda x: x[0])

    portfolio_pnls = [t[2] for t in all_trades_full]
    mc_p = mc_test(portfolio_pnls)
    p2['mc_portfolio_pvalue'] = mc_p
    print(f"\nPortfolio MC test (full costs): p = {mc_p:.4f} {'PASS' if mc_p < 0.01 else 'FAIL'}")

    # Phase 3: Edge decay
    p3 = phase3_edge_decay(df, pattern_details, atr_ratio, start_bar, end_bar)

    # Phase 4: vs Buy & Hold
    p4 = phase4_vs_buyhold(df, all_trades_full, start_bar, end_bar)

    # Phase 5: Scenarios
    p5 = phase5_scenarios(pattern_details, signal_index, opens, highs, lows,
                           n_bars, atr_ratio, start_bar, end_bar)

    # Verdict
    verdict = generate_verdict(p1, p2, p3, p4, p5)

    # Save results
    output = {
        'study': 'honest_strategy_evaluation',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': DATA_FILE,
        'patterns_file': PATTERNS_FILE,
        'n_patterns': len(pattern_details),
        'neutral_window': neutral,
        'cost_assumptions': {
            'leverage': LEVERAGE,
            'fee_pct_price': FEE_PCT,
            'fee_pct_capital': FEE_PCT * LEVERAGE,
            'slippage_pct_price': SLIPPAGE_PCT,
            'slippage_pct_capital': SLIPPAGE_PCT * 2 * LEVERAGE,
            'spread_pct_price': SPREAD_PCT,
            'spread_pct_capital': SPREAD_PCT * 2 * LEVERAGE,
            'total_cost_per_trade_capital': FEE_PCT * LEVERAGE + SLIPPAGE_PCT * 2 * LEVERAGE + SPREAD_PCT * 2 * LEVERAGE,
            'note': 'Fee is on notional (x leverage). Slippage/spread x2 for round-trip x leverage.',
        },
        'phase1_cost_analysis': p1,
        'phase2_expected_value': p2,
        'phase3_edge_decay': p3,
        'phase4_vs_buyhold': p4,
        'phase5_scenarios': p5,
        'verdict': verdict,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    return output


if __name__ == '__main__':
    main()
