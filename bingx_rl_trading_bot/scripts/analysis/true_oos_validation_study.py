#!/usr/bin/env python3
"""
True OOS Overfitting Validation Study
=======================================
The bot shows 88.8% WR in WF OOS but only 53.5% WR live (114 trades).
This study answers: is the entire pattern discovery approach overfitting?

Phase 1: True Forward OOS Test
  - IS: data before 2026-01-22 (bot go-live date)
  - Scan patterns with MAE/MFE + ATR, edge>=18pp, MC<0.01, min_trades>=25
  - Test on 2026-01-22 ~ 2026-02-26 (true OOS, ~36 days)
  - Compare True OOS WR vs WF internal OOS WR (88.8%)

Phase 2: Rolling Forward Test
  - 30-day rolling IS windows + 30-day forward OOS
  - For each: scan patterns -> test next 30d
  - Show how WR degrades as we move forward from IS
  - Tests if "pattern discovery -> forward test" reliably produces edge

Phase 3: Live Trade Replay
  - Load 59 actual live trades from metrics.json trade_history
  - For each trade: check if pattern+direction in production pattern set
  - Replay what backtest predicted vs actual outcome
  - Directly measures backtest-to-live accuracy

Phase 4: Randomization Benchmark
  - Generate 130 RANDOM pattern-direction pairs (same count as production)
  - Backtest with same TP/SL distribution
  - Run WF on random patterns
  - If random patterns show high WF OOS WR, the approach is fundamentally flawed

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based, abs(tp - opens[bar]))
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage: 0.02%
  ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  Timeout: 864 bars (72h) -> DROP (not counted)
  WF: 3-fold expanding window
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# CONSTANTS (standard research protocol)
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10
SLIPPAGE_BUFFER = 0.02
MAX_BARS = 288            # 24h timeout for scanner
TIMEOUT_BARS = 864        # 72h for portfolio sim
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]
MAX_BASELINE_WR = 70.0
EDGE_THRESHOLD = 18.0
MC_THRESHOLD = 0.01
MIN_TRADES = 25
BARS_PER_DAY = 288

# ATR scaling
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7

# MAE/MFE percentile grids (same as scanner)
TP_PERCENTILES = [40, 50, 60, 70, 80]
SL_PERCENTILES = [70, 75, 80, 85, 90, 95]

# Live trading start date
LIVE_START_DATE = '2026-01-22'

# Rolling window params (Phase 2)
ROLLING_IS_DAYS = 90      # 90-day IS windows
ROLLING_OOS_DAYS = 30     # 30-day forward OOS
ROLLING_STEP_DAYS = 30    # step by 30 days

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
METRICS_FILE = os.path.join(PROJECT_ROOT, 'results', 'pattern_5m_metrics.json')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'true_oos_validation_study.json')


# ============================================================
# DATA LOADING
# ============================================================

def load_and_classify(data_file):
    """Load CSV and classify candles using production classify_candle()."""
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)

    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

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
    print(f"Classified {len(df)} candles into 12 types")
    return df


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
# ATR COMPUTATION
# ============================================================

def compute_atr_ratio(highs, lows, closes, atr_period=ATR_PERIOD, window=ATR_WINDOW):
    """ATR / rolling_median(ATR) ratio time series (Wilder smoothing)."""
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


# ============================================================
# BACKTESTING ENGINE
# ============================================================

def bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                    opens, highs, lows, n_bars,
                    atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI):
    """Backtest with ATR-scaled TP/SL. Timeout trades are DROPPED."""
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

        # ATR scaling at signal bar
        if (atr_ratio is not None and idx < len(atr_ratio)
                and not np.isnan(atr_ratio[idx])):
            r = max(clamp_lo, min(clamp_hi, atr_ratio[idx]))
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
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, eff_tp * LEVERAGE - fee))
                break
            elif hs:
                trades.append((eb, j, -eff_sl * LEVERAGE - fee))
                break
        # Timeout -> DROP (no append)

    return trades


# ============================================================
# MC TEST
# ============================================================

def mc_test(pnls, n_sims=MC_SIMS):
    """Multi-seed sign randomization MC test. Returns max p-value (conservative)."""
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
# MAE/MFE DISCOVERY (from scanner)
# ============================================================

def compute_excursions(signal_bars, direction, opens, highs, lows, n_bars,
                       max_bars=MAX_BARS):
    """Compute MFE and MAE for each trade (natural path, no TP/SL)."""
    excursions = []
    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        mfe = 0.0
        mae = 0.0
        end_bar = min(idx + 2 + max_bars, n_bars)
        for j in range(idx + 2, end_bar):
            if direction == 'LONG':
                favorable = highs[j] - entry
                adverse = entry - lows[j]
            else:
                favorable = entry - lows[j]
                adverse = highs[j] - entry
            if favorable > mfe:
                mfe = favorable
            if adverse > mae:
                mae = adverse
        mfe_pct = mfe / entry * 100
        mae_pct = mae / entry * 100
        excursions.append({'mfe_pct': mfe_pct, 'mae_pct': mae_pct})
    return excursions


def derive_tp_sl(excursions, tp_percentile, sl_percentile):
    """Derive TP/SL from MFE/MAE percentiles."""
    if not excursions:
        return None, None
    mfes = [e['mfe_pct'] for e in excursions]
    maes = [e['mae_pct'] for e in excursions]
    tp = round(float(np.percentile(mfes, tp_percentile)), 2)
    sl = round(float(np.percentile(maes, sl_percentile)), 2)
    return tp, sl


def grid_search_mae_mfe(signal_bars, direction, opens, highs, lows, n_bars,
                         atr_ratio, min_tr=MIN_TRADES):
    """Grid search over MFE/MAE percentiles. Returns best (tp, sl, stats) or None."""
    excursions = compute_excursions(signal_bars, direction, opens, highs, lows,
                                    n_bars, MAX_BARS)
    if len(excursions) < min_tr:
        return None

    best = None
    best_score = -9999

    for tp_p in TP_PERCENTILES:
        for sl_p in SL_PERCENTILES:
            tp, sl = derive_tp_sl(excursions, tp_p, sl_p)
            if tp is None or sl is None:
                continue
            if tp < 0.1 or sl < 0.3:
                continue
            bwr = sl / (tp + sl) * 100
            if bwr > MAX_BASELINE_WR:
                continue

            trades = bt_signals_atr(signal_bars, direction, tp, sl,
                                    opens, highs, lows, n_bars,
                                    atr_ratio, CLAMP_LO, CLAMP_HI)
            if len(trades) < min_tr:
                continue

            pnls = [t[2] for t in trades]
            cum = 0
            peak = 0
            mdd_val = 0
            w = 0
            for p in pnls:
                cum += p
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > mdd_val:
                    mdd_val = dd
                if p > 0:
                    w += 1

            wr = w / len(pnls) * 100
            score = (cum / mdd_val) if mdd_val > 0 else cum

            if score > best_score:
                best_score = score
                best = {
                    'tp': tp, 'sl': sl,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'pnl': round(cum, 1), 'mdd': round(mdd_val, 1),
                    'trades_raw': trades,
                }

    return best


# ============================================================
# PATTERN SCANNER (simplified, single-threaded)
# ============================================================

def scan_patterns_is(signal_index, opens, highs, lows, n_bars,
                      bar_start, bar_end, atr_ratio):
    """Scan patterns within [bar_start, bar_end) range using MAE/MFE + ATR.

    Returns dict of {pattern_direction: {pattern, direction, tp, sl, trades, wr, edge, mc_p}}.
    """
    trade_boundary = bar_end if bar_end is not None else n_bars
    selected = {}

    for pat_name, all_sigs in signal_index.items():
        sigs = [s for s in all_sigs if bar_start <= s < bar_end]

        for direction in ['LONG', 'SHORT']:
            if len(sigs) < MIN_TRADES:
                continue

            # MAE/MFE grid search within IS range
            excursions = compute_excursions(sigs, direction, opens, highs, lows,
                                            trade_boundary, MAX_BARS)
            if len(excursions) < MIN_TRADES:
                continue

            best = None
            best_score = -9999

            for tp_p in TP_PERCENTILES:
                for sl_p in SL_PERCENTILES:
                    tp, sl = derive_tp_sl(excursions, tp_p, sl_p)
                    if tp is None or sl is None:
                        continue
                    if tp < 0.1 or sl < 0.3:
                        continue
                    bwr = sl / (tp + sl) * 100
                    if bwr > MAX_BASELINE_WR:
                        continue

                    trades = bt_signals_atr(sigs, direction, tp, sl,
                                            opens, highs, lows, trade_boundary,
                                            atr_ratio, CLAMP_LO, CLAMP_HI)
                    if len(trades) < MIN_TRADES:
                        continue

                    pnls = [t[2] for t in trades]
                    cum = 0
                    peak = 0
                    mdd_val = 0
                    w = 0
                    for p in pnls:
                        cum += p
                        if cum > peak:
                            peak = cum
                        dd = peak - cum
                        if dd > mdd_val:
                            mdd_val = dd
                        if p > 0:
                            w += 1

                    wr = w / len(pnls) * 100
                    score = (cum / mdd_val) if mdd_val > 0 else cum

                    if score > best_score:
                        best_score = score
                        best = {
                            'tp': tp, 'sl': sl,
                            'trades': len(trades), 'wr': round(wr, 1),
                            'pnl': round(cum, 1),
                            'trades_raw': trades,
                        }

            if best is None:
                continue

            tp, sl = best['tp'], best['sl']
            pnls = [t[2] for t in best['trades_raw']]
            wr = best['wr']
            baseline_wr = sl / (tp + sl) * 100
            edge = wr - baseline_wr

            if edge < EDGE_THRESHOLD:
                continue

            p = mc_test(pnls)
            if p >= MC_THRESHOLD:
                continue

            # SL >= 1.0% filter
            if sl < 1.0:
                continue

            key = f"{pat_name}_{direction}"
            selected[key] = {
                'pattern': pat_name,
                'direction': direction,
                'tp': tp,
                'sl': sl,
                'trades': best['trades'],
                'wr': wr,
                'edge': round(edge, 1),
                'mc_p': round(p, 4),
                'baseline_wr': round(baseline_wr, 1),
            }

    return selected


def portfolio_1pos(all_trades):
    """1-position-at-a-time filter: sort by entry, skip overlapping."""
    if not all_trades:
        return []
    all_trades = sorted(all_trades, key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def calc_stats(trades):
    """Calculate portfolio statistics from trade list."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0}
    pnls = [t[2] for t in trades]
    cum = 0
    peak = 0
    mdd = 0
    wins = []
    losses = []
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
        else:
            losses.append(p)
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(cum, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'avg_win': round(sum(wins) / len(wins), 2) if wins else 0,
        'avg_loss': round(sum(abs(x) for x in losses) / len(losses), 2) if losses else 0,
    }


# ============================================================
# PHASE 1: TRUE FORWARD OOS TEST
# ============================================================

def phase1_true_forward_oos(df, signal_index, opens, highs, lows, closes, n_bars):
    """Scan on pre-live data, test on post-live data."""
    print("\n" + "=" * 70)
    print("PHASE 1: TRUE FORWARD OOS TEST")
    print("=" * 70)

    # Find cutoff bar
    timestamps = pd.to_datetime(df['timestamp'])
    cutoff = pd.Timestamp(LIVE_START_DATE)
    is_mask = timestamps < cutoff
    is_end_bar = is_mask.sum()
    oos_start_bar = is_end_bar

    print(f"IS period: bar 0 to {is_end_bar} ({is_end_bar / BARS_PER_DAY:.1f} days)")
    print(f"OOS period: bar {oos_start_bar} to {n_bars} ({(n_bars - oos_start_bar) / BARS_PER_DAY:.1f} days)")
    print(f"IS date range: {timestamps.iloc[0].date()} to {timestamps.iloc[is_end_bar - 1].date()}")
    print(f"OOS date range: {timestamps.iloc[oos_start_bar].date()} to {timestamps.iloc[-1].date()}")

    # Compute ATR on IS data only (no look-ahead)
    is_atr = compute_atr_ratio(
        highs[:is_end_bar], lows[:is_end_bar], closes[:is_end_bar],
        ATR_PERIOD, ATR_WINDOW
    )

    # Scan patterns on IS
    t0 = time.time()
    is_patterns = scan_patterns_is(
        signal_index, opens, highs, lows, n_bars,
        bar_start=0, bar_end=is_end_bar, atr_ratio=is_atr
    )
    scan_time = time.time() - t0

    n_long = sum(1 for v in is_patterns.values() if v['direction'] == 'LONG')
    n_short = sum(1 for v in is_patterns.values() if v['direction'] == 'SHORT')
    print(f"\nIS scan: {len(is_patterns)} patterns ({n_long}L + {n_short}S) "
          f"in {scan_time:.1f}s")

    if not is_patterns:
        print("NO PATTERNS FOUND IN IS -- cannot proceed")
        return {
            'status': 'NO_PATTERNS',
            'is_bars': is_end_bar,
            'oos_bars': n_bars - oos_start_bar,
            'is_patterns': 0,
        }

    # Print IS summary
    is_wrs = [v['wr'] for v in is_patterns.values()]
    is_edges = [v['edge'] for v in is_patterns.values()]
    print(f"IS WR range: {min(is_wrs):.1f}% - {max(is_wrs):.1f}% (mean {np.mean(is_wrs):.1f}%)")
    print(f"IS Edge range: {min(is_edges):.1f}pp - {max(is_edges):.1f}pp (mean {np.mean(is_edges):.1f}pp)")

    # Compute full ATR for OOS evaluation (IS data + OOS data available at each point)
    full_atr = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    # Backtest IS-discovered patterns on OOS
    oos_trades_all = []
    pattern_oos_results = {}

    for key, pinfo in is_patterns.items():
        oos_sigs = [s for s in signal_index[pinfo['pattern']]
                    if oos_start_bar <= s < n_bars]

        oos_trades = bt_signals_atr(
            oos_sigs, pinfo['direction'], pinfo['tp'], pinfo['sl'],
            opens, highs, lows, n_bars,
            full_atr, CLAMP_LO, CLAMP_HI
        )

        oos_trades_all.extend(oos_trades)

        if oos_trades:
            oos_pnls = [t[2] for t in oos_trades]
            oos_w = sum(1 for p in oos_pnls if p > 0)
            oos_wr = oos_w / len(oos_pnls) * 100
        else:
            oos_wr = 0.0

        pattern_oos_results[key] = {
            'pattern': pinfo['pattern'],
            'direction': pinfo['direction'],
            'tp': pinfo['tp'],
            'sl': pinfo['sl'],
            'is_wr': pinfo['wr'],
            'is_edge': pinfo['edge'],
            'oos_trades': len(oos_trades),
            'oos_wr': round(oos_wr, 1),
            'oos_signals': len(oos_sigs),
        }

    # Portfolio-level OOS stats (1-pos filter)
    oos_portfolio = portfolio_1pos(oos_trades_all)
    oos_stats = calc_stats(oos_portfolio)

    # Compound PnL
    equity = 1.0
    for t in oos_portfolio:
        equity *= (1 + t[2] / 100)
    compound_pnl = (equity - 1) * 100

    # Per-pattern WR degradation analysis
    wr_degradation = []
    for key, res in pattern_oos_results.items():
        if res['oos_trades'] > 0:
            wr_degradation.append({
                'pattern': res['pattern'],
                'direction': res['direction'],
                'is_wr': res['is_wr'],
                'oos_wr': res['oos_wr'],
                'delta_wr': round(res['oos_wr'] - res['is_wr'], 1),
                'oos_trades': res['oos_trades'],
            })
    wr_degradation.sort(key=lambda x: x['delta_wr'])

    # Aggregate IS WR for comparison
    is_mean_wr = np.mean([v['wr'] for v in is_patterns.values()])

    print(f"\n--- TRUE OOS RESULTS ---")
    print(f"Patterns discovered in IS: {len(is_patterns)} ({n_long}L + {n_short}S)")
    print(f"OOS trades (1-pos portfolio): {oos_stats['trades']}")
    print(f"OOS WR: {oos_stats['wr']}%")
    print(f"OOS PnL (additive): {oos_stats['pnl']}%")
    print(f"OOS PnL (compound): {compound_pnl:.1f}%")
    print(f"OOS MDD: {oos_stats['mdd']}%")
    print(f"OOS PF: {oos_stats['pf']}")
    print(f"IS mean WR: {is_mean_wr:.1f}%")
    print(f"WR drop: {is_mean_wr:.1f}% -> {oos_stats['wr']}% = {oos_stats['wr'] - is_mean_wr:.1f}pp")

    # Pattern-level analysis
    oos_active = [r for r in pattern_oos_results.values() if r['oos_trades'] > 0]
    oos_inactive = [r for r in pattern_oos_results.values() if r['oos_trades'] == 0]
    print(f"\nPatterns with OOS trades: {len(oos_active)}/{len(is_patterns)}")
    print(f"Patterns with 0 OOS trades: {len(oos_inactive)} (no signals in OOS period)")

    if wr_degradation:
        print(f"\nWR Degradation (IS -> OOS):")
        print(f"  Worst 5:")
        for item in wr_degradation[:5]:
            print(f"    {item['pattern']}_{item['direction']}: "
                  f"{item['is_wr']}% -> {item['oos_wr']}% ({item['delta_wr']:+.1f}pp, "
                  f"{item['oos_trades']} trades)")
        print(f"  Best 5:")
        for item in wr_degradation[-5:]:
            print(f"    {item['pattern']}_{item['direction']}: "
                  f"{item['is_wr']}% -> {item['oos_wr']}% ({item['delta_wr']:+.1f}pp, "
                  f"{item['oos_trades']} trades)")

    result = {
        'is_bars': is_end_bar,
        'is_days': round(is_end_bar / BARS_PER_DAY, 1),
        'oos_bars': n_bars - oos_start_bar,
        'oos_days': round((n_bars - oos_start_bar) / BARS_PER_DAY, 1),
        'is_patterns': len(is_patterns),
        'is_long': n_long,
        'is_short': n_short,
        'is_mean_wr': round(is_mean_wr, 1),
        'is_mean_edge': round(np.mean(is_edges), 1),
        'oos_stats': oos_stats,
        'oos_compound_pnl': round(compound_pnl, 1),
        'wr_drop_pp': round(oos_stats['wr'] - is_mean_wr, 1),
        'patterns_with_oos_trades': len(oos_active),
        'patterns_without_oos_trades': len(oos_inactive),
        'wr_degradation_top10': wr_degradation[:10],
        'wr_degradation_bottom10': wr_degradation[-10:],
        'per_pattern_oos': pattern_oos_results,
        'scan_time_s': round(scan_time, 1),
    }

    return result


# ============================================================
# PHASE 2: ROLLING FORWARD TEST
# ============================================================

def phase2_rolling_forward(df, signal_index, opens, highs, lows, closes, n_bars):
    """Rolling IS windows + forward OOS to test edge persistence."""
    print("\n" + "=" * 70)
    print("PHASE 2: ROLLING FORWARD TEST")
    print(f"IS window: {ROLLING_IS_DAYS}d, OOS: {ROLLING_OOS_DAYS}d, step: {ROLLING_STEP_DAYS}d")
    print("=" * 70)

    is_bars = ROLLING_IS_DAYS * BARS_PER_DAY
    oos_bars_len = ROLLING_OOS_DAYS * BARS_PER_DAY
    step_bars = ROLLING_STEP_DAYS * BARS_PER_DAY

    timestamps = pd.to_datetime(df['timestamp'])
    windows = []
    window_idx = 0

    start = 0
    while start + is_bars + oos_bars_len <= n_bars:
        is_end = start + is_bars
        oos_start = is_end
        oos_end = min(oos_start + oos_bars_len, n_bars)

        window_idx += 1
        is_start_date = timestamps.iloc[start].strftime('%Y-%m-%d')
        is_end_date = timestamps.iloc[is_end - 1].strftime('%Y-%m-%d')
        oos_start_date = timestamps.iloc[oos_start].strftime('%Y-%m-%d')
        oos_end_date = timestamps.iloc[oos_end - 1].strftime('%Y-%m-%d')

        print(f"\n  Window {window_idx}: IS [{is_start_date} -> {is_end_date}] "
              f"OOS [{oos_start_date} -> {oos_end_date}]")

        # ATR on IS only
        is_atr = compute_atr_ratio(
            highs[start:is_end], lows[start:is_end], closes[start:is_end],
            ATR_PERIOD, ATR_WINDOW
        )
        # Pad to full array indices (signals use global indices)
        full_is_atr = np.full(n_bars, np.nan)
        full_is_atr[start:is_end] = is_atr

        # Scan IS range
        is_patterns = scan_patterns_is(
            signal_index, opens, highs, lows, n_bars,
            bar_start=start, bar_end=is_end, atr_ratio=full_is_atr
        )

        n_long = sum(1 for v in is_patterns.values() if v['direction'] == 'LONG')
        n_short = sum(1 for v in is_patterns.values() if v['direction'] == 'SHORT')
        print(f"    IS: {len(is_patterns)} patterns ({n_long}L + {n_short}S)")

        if not is_patterns:
            print(f"    No patterns found -- skip")
            start += step_bars
            continue

        # ATR for OOS (can use data up to oos_end)
        oos_atr = compute_atr_ratio(
            highs[:oos_end], lows[:oos_end], closes[:oos_end],
            ATR_PERIOD, ATR_WINDOW
        )

        # OOS backtest
        oos_trades_all = []
        for key, pinfo in is_patterns.items():
            oos_sigs = [s for s in signal_index[pinfo['pattern']]
                        if oos_start <= s < oos_end]
            oos_trades = bt_signals_atr(
                oos_sigs, pinfo['direction'], pinfo['tp'], pinfo['sl'],
                opens, highs, lows, oos_end,
                oos_atr, CLAMP_LO, CLAMP_HI
            )
            oos_trades_all.extend(oos_trades)

        oos_portfolio = portfolio_1pos(oos_trades_all)
        oos_stats = calc_stats(oos_portfolio)

        is_mean_wr = np.mean([v['wr'] for v in is_patterns.values()])

        print(f"    OOS: {oos_stats['trades']} trades, WR {oos_stats['wr']}%, "
              f"PnL {oos_stats['pnl']}%")
        print(f"    IS mean WR: {is_mean_wr:.1f}% -> OOS WR: {oos_stats['wr']}% "
              f"(delta {oos_stats['wr'] - is_mean_wr:.1f}pp)")

        windows.append({
            'window': window_idx,
            'is_start_bar': start,
            'is_end_bar': is_end,
            'oos_start_bar': oos_start,
            'oos_end_bar': oos_end,
            'is_start_date': is_start_date,
            'oos_end_date': oos_end_date,
            'is_patterns': len(is_patterns),
            'is_long': n_long,
            'is_short': n_short,
            'is_mean_wr': round(is_mean_wr, 1),
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_mdd': oos_stats['mdd'],
            'oos_positive': oos_stats['pnl'] > 0,
        })

        start += step_bars

    # Summary
    if windows:
        positive_count = sum(1 for w in windows if w['oos_positive'])
        avg_oos_wr = np.mean([w['oos_wr'] for w in windows if w['oos_trades'] > 0])
        avg_oos_pnl = np.mean([w['oos_pnl'] for w in windows])
        avg_is_wr = np.mean([w['is_mean_wr'] for w in windows])
        avg_wr_drop = avg_oos_wr - avg_is_wr

        print(f"\n--- ROLLING FORWARD SUMMARY ---")
        print(f"Total windows: {len(windows)}")
        print(f"Positive OOS windows: {positive_count}/{len(windows)} "
              f"({positive_count / len(windows) * 100:.0f}%)")
        print(f"Average IS WR: {avg_is_wr:.1f}%")
        print(f"Average OOS WR: {avg_oos_wr:.1f}%")
        print(f"Average WR drop: {avg_wr_drop:.1f}pp")
        print(f"Average OOS PnL: {avg_oos_pnl:.1f}%")
    else:
        positive_count = 0
        avg_oos_wr = 0
        avg_oos_pnl = 0
        avg_is_wr = 0
        avg_wr_drop = 0

    result = {
        'rolling_is_days': ROLLING_IS_DAYS,
        'rolling_oos_days': ROLLING_OOS_DAYS,
        'rolling_step_days': ROLLING_STEP_DAYS,
        'total_windows': len(windows),
        'positive_windows': positive_count,
        'positive_pct': round(positive_count / len(windows) * 100, 1) if windows else 0,
        'avg_is_wr': round(avg_is_wr, 1),
        'avg_oos_wr': round(avg_oos_wr, 1),
        'avg_wr_drop_pp': round(avg_wr_drop, 1),
        'avg_oos_pnl': round(avg_oos_pnl, 1),
        'windows': windows,
    }

    return result


# ============================================================
# PHASE 3: LIVE TRADE REPLAY
# ============================================================

def phase3_live_trade_replay(df, signal_index, opens, highs, lows, closes, n_bars):
    """Compare actual live trades against backtest predictions."""
    print("\n" + "=" * 70)
    print("PHASE 3: LIVE TRADE REPLAY")
    print("=" * 70)

    # Load metrics
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    trade_history = metrics.get('trade_history', [])

    # Load production patterns
    with open(PATTERNS_FILE) as f:
        prod_patterns = json.load(f)

    # Build production pattern lookup
    prod_lookup = {}
    patterns_tpsl = prod_patterns.get('patterns_tpsl', {})
    for pat_name in prod_patterns.get('patterns', {}).get('long', []):
        prod_lookup[f"{pat_name}_LONG"] = {
            'tp': patterns_tpsl.get(pat_name, [1.0, 1.0])[0],
            'sl': patterns_tpsl.get(pat_name, [1.0, 1.0])[1],
        }
    for pat_name in prod_patterns.get('patterns', {}).get('short', []):
        prod_lookup[f"{pat_name}_SHORT"] = {
            'tp': patterns_tpsl.get(pat_name, [1.0, 1.0])[0],
            'sl': patterns_tpsl.get(pat_name, [1.0, 1.0])[1],
        }

    print(f"Live trades in history: {len(trade_history)}")
    print(f"Production patterns: {len(prod_lookup)}")

    timestamps = pd.to_datetime(df['timestamp'])

    # Full ATR for replay
    full_atr = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    replay_results = []
    matched_count = 0
    unmatched_count = 0
    na_count = 0
    bt_agrees = 0
    bt_disagrees = 0

    for trade in trade_history:
        pattern = trade.get('pattern', 'N/A')
        direction = trade.get('direction', 'UNKNOWN')
        entry_price = trade.get('entry_price', 0)
        exit_reason = trade.get('exit_reason', 'UNKNOWN')
        pnl_slot = trade.get('pnl_slot', 0)
        timestamp = trade.get('timestamp', '')

        if pattern == 'N/A':
            na_count += 1
            continue

        key = f"{pattern}_{direction}"
        in_prod = key in prod_lookup

        # Find the closest bar to this trade's entry
        trade_ts = pd.Timestamp(timestamp)
        # Find the signal bar: look for the pattern in the data near this timestamp
        bar_diffs = abs(timestamps - trade_ts)
        closest_bar = bar_diffs.idxmin()

        # Determine if backtest would predict TP or SL for this entry
        bt_prediction = None
        if in_prod and entry_price > 0:
            tp_pct = prod_lookup[key]['tp']
            sl_pct = prod_lookup[key]['sl']

            # Get ATR ratio at signal bar
            if closest_bar < len(full_atr) and not np.isnan(full_atr[closest_bar]):
                r = max(CLAMP_LO, min(CLAMP_HI, full_atr[closest_bar]))
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            if direction == 'LONG':
                tpp = entry_price * (1 + eff_tp / 100)
                slp = entry_price * (1 - eff_sl / 100)
            else:
                tpp = entry_price * (1 - eff_tp / 100)
                slp = entry_price * (1 + eff_sl / 100)

            # Simulate forward from entry
            entry_bar = closest_bar + 1
            if entry_bar < n_bars:
                for j in range(entry_bar + 1, min(entry_bar + 1 + MAX_BARS, n_bars)):
                    if direction == 'LONG':
                        ht = highs[j] >= tpp
                        hs = lows[j] <= slp
                    else:
                        ht = lows[j] <= tpp
                        hs = highs[j] >= slp

                    if ht and hs:
                        bo = opens[j]
                        bt_prediction = 'TP' if abs(tpp - bo) <= abs(slp - bo) else 'SL'
                        break
                    elif ht:
                        bt_prediction = 'TP'
                        break
                    elif hs:
                        bt_prediction = 'SL'
                        break
                    # If loop ends without break: TIMEOUT

        # Normalize exit_reason to TP/SL
        live_outcome = None
        if exit_reason == 'TP':
            live_outcome = 'TP'
        elif exit_reason == 'SL':
            live_outcome = 'SL'
        elif exit_reason == 'MARKET':
            # Market close -- classify by PnL
            live_outcome = 'TP' if pnl_slot > 0 else 'SL'

        agrees = None
        if bt_prediction is not None and live_outcome is not None:
            agrees = (bt_prediction == live_outcome)
            if agrees:
                bt_agrees += 1
            else:
                bt_disagrees += 1

        if in_prod:
            matched_count += 1
        else:
            unmatched_count += 1

        replay_results.append({
            'timestamp': timestamp,
            'pattern': pattern,
            'direction': direction,
            'entry_price': entry_price,
            'exit_reason': exit_reason,
            'pnl_slot': round(pnl_slot, 2),
            'in_production_set': in_prod,
            'bt_prediction': bt_prediction,
            'live_outcome': live_outcome,
            'agrees': agrees,
        })

    # Classify live trades
    live_wins = sum(1 for r in replay_results if r['live_outcome'] == 'TP')
    live_losses = sum(1 for r in replay_results if r['live_outcome'] == 'SL')
    live_total = live_wins + live_losses
    live_wr = live_wins / live_total * 100 if live_total > 0 else 0

    # Agreement rate
    total_compared = bt_agrees + bt_disagrees
    agreement_rate = bt_agrees / total_compared * 100 if total_compared > 0 else 0

    # Matched vs unmatched WR
    matched_trades = [r for r in replay_results if r['in_production_set'] and r['live_outcome']]
    unmatched_trades = [r for r in replay_results if not r['in_production_set'] and r['live_outcome']]
    matched_wr = (sum(1 for r in matched_trades if r['live_outcome'] == 'TP')
                  / len(matched_trades) * 100 if matched_trades else 0)
    unmatched_wr = (sum(1 for r in unmatched_trades if r['live_outcome'] == 'TP')
                    / len(unmatched_trades) * 100 if unmatched_trades else 0)

    print(f"\n--- LIVE TRADE REPLAY RESULTS ---")
    print(f"Total trades in history: {len(trade_history)}")
    print(f"N/A pattern trades (excluded): {na_count}")
    print(f"Analyzed trades: {len(replay_results)}")
    print(f"In production pattern set: {matched_count}")
    print(f"NOT in production set: {unmatched_count}")
    print(f"\nLive WR: {live_wr:.1f}% ({live_wins}W/{live_losses}L)")
    print(f"Matched pattern WR: {matched_wr:.1f}% ({len(matched_trades)} trades)")
    print(f"Unmatched pattern WR: {unmatched_wr:.1f}% ({len(unmatched_trades)} trades)")
    print(f"\nBacktest vs Live agreement: {agreement_rate:.1f}% ({bt_agrees}/{total_compared})")
    print(f"  Backtest predicted TP, was TP: {sum(1 for r in replay_results if r.get('bt_prediction') == 'TP' and r.get('live_outcome') == 'TP')}")
    print(f"  Backtest predicted TP, was SL: {sum(1 for r in replay_results if r.get('bt_prediction') == 'TP' and r.get('live_outcome') == 'SL')}")
    print(f"  Backtest predicted SL, was SL: {sum(1 for r in replay_results if r.get('bt_prediction') == 'SL' and r.get('live_outcome') == 'SL')}")
    print(f"  Backtest predicted SL, was TP: {sum(1 for r in replay_results if r.get('bt_prediction') == 'SL' and r.get('live_outcome') == 'TP')}")
    print(f"  Timeout/no-prediction: {len(replay_results) - total_compared}")

    result = {
        'total_live_trades': len(trade_history),
        'na_excluded': na_count,
        'analyzed_trades': len(replay_results),
        'matched_in_prod': matched_count,
        'unmatched': unmatched_count,
        'live_wr': round(live_wr, 1),
        'matched_wr': round(matched_wr, 1),
        'unmatched_wr': round(unmatched_wr, 1),
        'bt_agreement_rate': round(agreement_rate, 1),
        'bt_agrees': bt_agrees,
        'bt_disagrees': bt_disagrees,
        'bt_no_prediction': len(replay_results) - total_compared,
        'confusion_matrix': {
            'bt_tp_live_tp': sum(1 for r in replay_results if r.get('bt_prediction') == 'TP' and r.get('live_outcome') == 'TP'),
            'bt_tp_live_sl': sum(1 for r in replay_results if r.get('bt_prediction') == 'TP' and r.get('live_outcome') == 'SL'),
            'bt_sl_live_sl': sum(1 for r in replay_results if r.get('bt_prediction') == 'SL' and r.get('live_outcome') == 'SL'),
            'bt_sl_live_tp': sum(1 for r in replay_results if r.get('bt_prediction') == 'SL' and r.get('live_outcome') == 'TP'),
        },
        'replay_details': replay_results,
    }

    return result


# ============================================================
# PHASE 4: RANDOMIZATION BENCHMARK
# ============================================================

def phase4_randomization_benchmark(df, signal_index, opens, highs, lows, closes, n_bars):
    """Test if random patterns also show high IS WR and OOS performance."""
    print("\n" + "=" * 70)
    print("PHASE 4: RANDOMIZATION BENCHMARK")
    print("=" * 70)

    # Load production patterns for TP/SL distribution reference
    with open(PATTERNS_FILE) as f:
        prod_patterns = json.load(f)
    patterns_tpsl = prod_patterns.get('patterns_tpsl', {})

    # Get TP/SL distribution from production
    prod_tps = [v[0] for v in patterns_tpsl.values()]
    prod_sls = [v[1] for v in patterns_tpsl.values()]
    n_prod = len(patterns_tpsl)

    print(f"Production patterns: {n_prod}")
    print(f"Production TP range: {min(prod_tps):.2f} - {max(prod_tps):.2f}")
    print(f"Production SL range: {min(prod_sls):.2f} - {max(prod_sls):.2f}")

    # Get all available pattern names
    all_patterns = list(signal_index.keys())
    directions = ['LONG', 'SHORT']

    # Build list of all possible pattern-direction combos
    all_combos = []
    for pat in all_patterns:
        for d in directions:
            all_combos.append((pat, d))

    # Full ATR
    full_atr = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    # Use Phase 1's IS/OOS split for comparison
    timestamps = pd.to_datetime(df['timestamp'])
    cutoff = pd.Timestamp(LIVE_START_DATE)
    is_end_bar = (timestamps < cutoff).sum()

    print(f"Total available pattern-direction combos: {len(all_combos)}")
    print(f"Random sampling: {n_prod} combos per trial, 20 trials")
    print(f"IS: bars 0-{is_end_bar}, OOS: bars {is_end_bar}-{n_bars}")

    # IS ATR
    is_atr = compute_atr_ratio(
        highs[:is_end_bar], lows[:is_end_bar], closes[:is_end_bar],
        ATR_PERIOD, ATR_WINDOW
    )

    random_trials = []
    n_trials = 20

    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 7 + 42)

        # Sample random patterns
        sample_idx = rng.choice(len(all_combos), size=min(n_prod, len(all_combos)),
                                replace=False)
        sampled = [all_combos[i] for i in sample_idx]

        # Assign random TP/SL from production distribution
        tp_samples = rng.choice(prod_tps, size=len(sampled))
        sl_samples = rng.choice(prod_sls, size=len(sampled))

        # IS backtest
        is_trades_all = []
        is_patterns_tested = 0
        for i, (pat, direction) in enumerate(sampled):
            sigs = [s for s in signal_index.get(pat, []) if s < is_end_bar]
            if len(sigs) < 5:  # Relaxed minimum for random test
                continue
            is_patterns_tested += 1
            trades = bt_signals_atr(
                sigs, direction, tp_samples[i], sl_samples[i],
                opens, highs, lows, is_end_bar,
                is_atr, CLAMP_LO, CLAMP_HI
            )
            is_trades_all.extend(trades)

        is_portfolio = portfolio_1pos(is_trades_all)
        is_stats = calc_stats(is_portfolio)

        # OOS backtest (same random patterns)
        oos_trades_all = []
        for i, (pat, direction) in enumerate(sampled):
            oos_sigs = [s for s in signal_index.get(pat, [])
                        if is_end_bar <= s < n_bars]
            if not oos_sigs:
                continue
            trades = bt_signals_atr(
                oos_sigs, direction, tp_samples[i], sl_samples[i],
                opens, highs, lows, n_bars,
                full_atr, CLAMP_LO, CLAMP_HI
            )
            oos_trades_all.extend(trades)

        oos_portfolio = portfolio_1pos(oos_trades_all)
        oos_stats = calc_stats(oos_portfolio)

        random_trials.append({
            'trial': trial + 1,
            'n_sampled': len(sampled),
            'is_patterns_with_trades': is_patterns_tested,
            'is_trades': is_stats['trades'],
            'is_wr': is_stats['wr'],
            'is_pnl': is_stats['pnl'],
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
        })

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial + 1}/{n_trials}: IS WR {is_stats['wr']}%, "
                  f"OOS WR {oos_stats['wr']}%, OOS PnL {oos_stats['pnl']}%")

    # Summary
    random_is_wrs = [t['is_wr'] for t in random_trials]
    random_oos_wrs = [t['oos_wr'] for t in random_trials if t['oos_trades'] > 0]
    random_oos_pnls = [t['oos_pnl'] for t in random_trials]

    print(f"\n--- RANDOMIZATION BENCHMARK RESULTS ---")
    print(f"Trials: {n_trials}")
    print(f"Random IS WR: {np.mean(random_is_wrs):.1f}% "
          f"(range {np.min(random_is_wrs):.1f} - {np.max(random_is_wrs):.1f})")
    print(f"Random OOS WR: {np.mean(random_oos_wrs):.1f}% "
          f"(range {np.min(random_oos_wrs):.1f} - {np.max(random_oos_wrs):.1f})")
    print(f"Random OOS PnL: {np.mean(random_oos_pnls):.1f}% "
          f"(range {np.min(random_oos_pnls):.1f} - {np.max(random_oos_pnls):.1f})")
    print(f"Positive OOS trials: {sum(1 for p in random_oos_pnls if p > 0)}/{n_trials}")

    # Additional test: scan random patterns through the FULL discovery pipeline
    # to see if they pass edge/MC filters at the same rate
    print("\n--- FILTER PASS RATE TEST ---")
    print("Testing if random pattern-direction combos pass edge>=18pp + MC<0.01...")

    n_filter_trials = 5
    filter_pass_rates = []

    for trial in range(n_filter_trials):
        rng = np.random.RandomState(trial * 13 + 99)
        sample_idx = rng.choice(len(all_combos), size=min(500, len(all_combos)),
                                replace=False)
        sampled_500 = [all_combos[i] for i in sample_idx]

        passed = 0
        tested = 0
        for pat, direction in sampled_500:
            sigs = [s for s in signal_index.get(pat, []) if s < is_end_bar]
            if len(sigs) < MIN_TRADES:
                continue
            tested += 1

            # Run MAE/MFE discovery on this random combo
            best = grid_search_mae_mfe(sigs, direction, opens, highs, lows,
                                        is_end_bar, is_atr, min_tr=MIN_TRADES)
            if best is None:
                continue

            pnls = [t[2] for t in best['trades_raw']]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            baseline_wr = best['sl'] / (best['tp'] + best['sl']) * 100
            edge = wr - baseline_wr

            if edge >= EDGE_THRESHOLD and mc_test(pnls) < MC_THRESHOLD and best['sl'] >= 1.0:
                passed += 1

        pass_rate = passed / tested * 100 if tested > 0 else 0
        filter_pass_rates.append(pass_rate)
        print(f"  Trial {trial + 1}: {passed}/{tested} passed ({pass_rate:.1f}%)")

    avg_filter_pass_rate = np.mean(filter_pass_rates) if filter_pass_rates else 0
    print(f"Average filter pass rate: {avg_filter_pass_rate:.1f}%")

    # Compare against production pass rate
    # Production: ~130 out of ~1294 tested = 10.0%
    prod_pass_rate = 130 / 1294 * 100  # from dynamic_patterns.json metadata
    print(f"Production pass rate: {prod_pass_rate:.1f}% (130/1294)")
    print(f"Random pass rate: {avg_filter_pass_rate:.1f}%")

    if avg_filter_pass_rate > prod_pass_rate * 0.5:
        print("WARNING: Random patterns pass filters at a substantial rate!")
        print("This suggests the filters may not discriminate genuine edge well.")
    else:
        print("Random patterns pass at lower rate than production -- filters have some discriminating power.")

    result = {
        'n_trials': n_trials,
        'random_is_wr_mean': round(np.mean(random_is_wrs), 1),
        'random_is_wr_range': [round(np.min(random_is_wrs), 1),
                                round(np.max(random_is_wrs), 1)],
        'random_oos_wr_mean': round(np.mean(random_oos_wrs), 1),
        'random_oos_wr_range': [round(np.min(random_oos_wrs), 1),
                                 round(np.max(random_oos_wrs), 1)],
        'random_oos_pnl_mean': round(np.mean(random_oos_pnls), 1),
        'random_oos_pnl_range': [round(np.min(random_oos_pnls), 1),
                                  round(np.max(random_oos_pnls), 1)],
        'positive_oos_trials': sum(1 for p in random_oos_pnls if p > 0),
        'trials': random_trials,
        'filter_pass_rate_test': {
            'n_trials': n_filter_trials,
            'pass_rates': [round(r, 1) for r in filter_pass_rates],
            'avg_random_pass_rate': round(avg_filter_pass_rate, 1),
            'prod_pass_rate': round(prod_pass_rate, 1),
        },
    }

    return result


# ============================================================
# FINAL VERDICT
# ============================================================

def generate_verdict(p1, p2, p3, p4):
    """Synthesize all phases into a final verdict."""
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    evidence = []
    overfit_score = 0  # Higher = more evidence of overfitting

    # Phase 1: True OOS
    if p1.get('status') == 'NO_PATTERNS':
        evidence.append("[P1] Could not find patterns in IS -- inconclusive")
    else:
        true_oos_wr = p1['oos_stats']['wr']
        wr_drop = p1['wr_drop_pp']

        if true_oos_wr < 55:
            evidence.append(f"[P1] True OOS WR = {true_oos_wr}% (near random) -- STRONG overfit signal")
            overfit_score += 3
        elif true_oos_wr < 65:
            evidence.append(f"[P1] True OOS WR = {true_oos_wr}% (weak edge) -- moderate overfit signal")
            overfit_score += 2
        elif true_oos_wr < 75:
            evidence.append(f"[P1] True OOS WR = {true_oos_wr}% (decent edge survives) -- mild overfit signal")
            overfit_score += 1
        else:
            evidence.append(f"[P1] True OOS WR = {true_oos_wr}% (strong edge survives) -- genuine edge signal")
            overfit_score -= 1

        if abs(wr_drop) > 30:
            evidence.append(f"[P1] WR drop = {wr_drop:.1f}pp (massive degradation)")
            overfit_score += 2
        elif abs(wr_drop) > 15:
            evidence.append(f"[P1] WR drop = {wr_drop:.1f}pp (significant degradation)")
            overfit_score += 1

    # Phase 2: Rolling Forward
    if p2.get('total_windows', 0) > 0:
        positive_pct = p2['positive_pct']
        avg_oos_wr = p2['avg_oos_wr']
        avg_wr_drop = p2['avg_wr_drop_pp']

        if positive_pct < 50:
            evidence.append(f"[P2] Only {positive_pct:.0f}% positive OOS windows -- STRONG overfit signal")
            overfit_score += 3
        elif positive_pct < 70:
            evidence.append(f"[P2] {positive_pct:.0f}% positive OOS windows -- moderate overfit signal")
            overfit_score += 1
        else:
            evidence.append(f"[P2] {positive_pct:.0f}% positive OOS windows -- edge persists forward")
            overfit_score -= 1

        if avg_oos_wr < 55:
            evidence.append(f"[P2] Average OOS WR = {avg_oos_wr:.1f}% (near random)")
            overfit_score += 2
        elif avg_oos_wr < 65:
            evidence.append(f"[P2] Average OOS WR = {avg_oos_wr:.1f}% (weak)")
            overfit_score += 1

        evidence.append(f"[P2] Average IS->OOS WR drop = {avg_wr_drop:.1f}pp")

    # Phase 3: Live Replay
    if p3.get('analyzed_trades', 0) > 0:
        agreement = p3['bt_agreement_rate']
        matched_wr = p3['matched_wr']

        if agreement < 55:
            evidence.append(f"[P3] BT-Live agreement = {agreement:.1f}% (backtest not predictive)")
            overfit_score += 2
        elif agreement < 70:
            evidence.append(f"[P3] BT-Live agreement = {agreement:.1f}% (weak predictive power)")
            overfit_score += 1
        else:
            evidence.append(f"[P3] BT-Live agreement = {agreement:.1f}% (backtest has predictive value)")
            overfit_score -= 1

        if matched_wr < 55:
            evidence.append(f"[P3] Matched pattern live WR = {matched_wr:.1f}% (patterns not performing)")
            overfit_score += 2

    # Phase 4: Random Benchmark
    if p4.get('n_trials', 0) > 0:
        random_oos_wr = p4['random_oos_wr_mean']
        random_pass_rate = p4.get('filter_pass_rate_test', {}).get('avg_random_pass_rate', 0)
        prod_pass_rate = p4.get('filter_pass_rate_test', {}).get('prod_pass_rate', 10)

        if random_pass_rate > prod_pass_rate * 0.7:
            evidence.append(f"[P4] Random filter pass rate = {random_pass_rate:.1f}% vs prod {prod_pass_rate:.1f}% "
                            f"(filters barely discriminate)")
            overfit_score += 3
        elif random_pass_rate > prod_pass_rate * 0.3:
            evidence.append(f"[P4] Random filter pass rate = {random_pass_rate:.1f}% vs prod {prod_pass_rate:.1f}% "
                            f"(moderate discrimination)")
            overfit_score += 1
        else:
            evidence.append(f"[P4] Random filter pass rate = {random_pass_rate:.1f}% vs prod {prod_pass_rate:.1f}% "
                            f"(good discrimination)")
            overfit_score -= 1

        positive_random = p4['positive_oos_trials']
        n_trials = p4['n_trials']
        if positive_random > n_trials * 0.5:
            evidence.append(f"[P4] {positive_random}/{n_trials} random trials have positive OOS PnL "
                            f"-- market has inherent direction bias")

    # Final determination
    print("\nEvidence summary:")
    for e in evidence:
        print(f"  {e}")

    print(f"\nOverfit score: {overfit_score} (higher = more evidence of overfitting)")

    if overfit_score >= 6:
        verdict = "OVERFIT"
        explanation = ("Strong evidence that the pattern discovery approach is overfitting. "
                       "The high IS WR does not persist in true OOS testing, and "
                       "the live trading results confirm this.")
    elif overfit_score >= 3:
        verdict = "LIKELY_OVERFIT"
        explanation = ("Moderate evidence of overfitting. Some edge may exist but it is "
                       "substantially inflated by the discovery process. The live WR "
                       "degradation is significant.")
    elif overfit_score >= 1:
        verdict = "PARTIAL_OVERFIT"
        explanation = ("Mixed evidence. There may be genuine edge that is partially inflated "
                       "by the discovery process. Edge decay and regime shift could explain "
                       "some of the live degradation.")
    elif overfit_score >= -1:
        verdict = "EDGE_DECAY"
        explanation = ("The evidence suggests genuine edge exists but is decaying over time. "
                       "The discovery process captures real patterns but their predictive power "
                       "diminishes, possibly due to market regime changes.")
    else:
        verdict = "GENUINE_EDGE"
        explanation = ("The evidence supports genuine edge in the pattern discovery approach. "
                       "Live degradation is consistent with normal statistical variation "
                       "or market regime shift rather than fundamental overfitting.")

    print(f"\n*** VERDICT: {verdict} ***")
    print(f"\n{explanation}")

    # Key question answer
    print(f"\nKey Question: '52.7% live WR is the result of overfitting or edge decay + regime shift?'")
    if verdict in ['OVERFIT', 'LIKELY_OVERFIT']:
        print("Answer: Primarily overfitting. The discovery pipeline inflates IS metrics "
              "beyond what survives in truly unseen data.")
    elif verdict == 'PARTIAL_OVERFIT':
        print("Answer: Both. Some genuine edge exists but it is partially overfitted "
              "and partially decayed. The 88.8% WF OOS is inflated.")
    else:
        print("Answer: Primarily edge decay and/or regime shift. The patterns had genuine "
              "predictive power that has weakened over time.")

    return {
        'verdict': verdict,
        'overfit_score': overfit_score,
        'explanation': explanation,
        'evidence': evidence,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("TRUE OOS OVERFITTING VALIDATION STUDY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {DATA_FILE}")
    print(f"Live start: {LIVE_START_DATE}")
    print("=" * 70)

    t0 = time.time()

    # Load data
    df = load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    n_bars = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    signal_index = build_signal_index(types, n_bars)

    print(f"Data: {n_bars} bars ({n_bars / BARS_PER_DAY:.1f} days)")
    print(f"Unique patterns: {len(signal_index)}")

    # Run all phases
    p1 = phase1_true_forward_oos(df, signal_index, opens, highs, lows, closes, n_bars)
    p2 = phase2_rolling_forward(df, signal_index, opens, highs, lows, closes, n_bars)
    p3 = phase3_live_trade_replay(df, signal_index, opens, highs, lows, closes, n_bars)
    p4 = phase4_randomization_benchmark(df, signal_index, opens, highs, lows, closes, n_bars)

    # Generate final verdict
    verdict = generate_verdict(p1, p2, p3, p4)

    total_time = time.time() - t0

    # Save results
    results = {
        'study': 'true_oos_validation',
        'date': datetime.now().isoformat(),
        'data_file': os.path.basename(DATA_FILE),
        'data_bars': n_bars,
        'live_start_date': LIVE_START_DATE,
        'parameters': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'max_bars': MAX_BARS,
            'edge_threshold': EDGE_THRESHOLD,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'clamp_lo': CLAMP_LO,
            'clamp_hi': CLAMP_HI,
            'rolling_is_days': ROLLING_IS_DAYS,
            'rolling_oos_days': ROLLING_OOS_DAYS,
            'rolling_step_days': ROLLING_STEP_DAYS,
        },
        'phase1_true_forward_oos': p1,
        'phase2_rolling_forward': p2,
        'phase3_live_trade_replay': p3,
        'phase4_randomization_benchmark': p4,
        'verdict': verdict,
        'total_time_s': round(total_time, 1),
    }

    # Remove non-serializable items (per_pattern_oos has nested dicts)
    # and replay_details for size
    p1_save = {k: v for k, v in p1.items() if k != 'per_pattern_oos'}
    p3_save = {k: v for k, v in p3.items() if k != 'replay_details'}
    results['phase1_true_forward_oos'] = p1_save
    results['phase3_live_trade_replay'] = p3_save

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == '__main__':
    main()
