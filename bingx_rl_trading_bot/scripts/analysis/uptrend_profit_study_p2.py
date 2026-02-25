#!/usr/bin/env python3
"""
Uptrend Profit Study Phase 2: Direction-Specific Thresholds, Data Extension, Regime Sizing
============================================================================================
Follow-up to Phase 1 (uptrend_profit_study.py) where all 5 hypotheses returned STOP.

Phase 1 findings:
  - LONG edge avg 8.8pp vs SHORT 14.3pp (5.5pp structural gap)
  - 270d data: uptrend 55% / downtrend 45%, net change -11.5% (bearish bias)
  - H1 bullish LONG 3 patterns -> WF FAIL (IS edge not reproducible)
  - Trend filter 4 hypotheses -> all STOP
  - Current: 51 patterns (16L+35S), edge>=21.8pp uniform threshold

Phase 2 Hypotheses:
  H1: Direction-Specific Edge Threshold
      - LONG patterns have structurally lower edge -> lower threshold may capture genuine LONGs
      - Test thresholds: 15pp, 17pp, 19pp, 21.8pp (current) for LONG only
      - Combine with existing 35 SHORT at 21.8pp, run WF 3-fold portfolio validation
      - Key: does adding lower-threshold LONGs improve portfolio, or introduce noise?

  H2: Data Extension with API
      - 270d CSV ends ~2026-01; fetch ~26d API data (2026-01~02) to get 296d
      - Recent period may include bullish regime -> LONG edge distribution improves
      - Re-scan full 296d with current criteria; compare LONG/SHORT pattern counts
      - WF 3-fold on extended data

  H3: Regime-Aware Position Sizing
      - Don't change patterns, change position SIZE by regime
      - UP regime (EMA slope > 0): SHORT size x0.5, DOWN: LONG size x0.5
      - Portfolio sim: regime-aware sizing vs uniform sizing
      - WF 3-fold validation

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based)
  Same-bar resolution: abs(tp - opens[j]) vs abs(sl - opens[j])
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage buffer: 0.02%
  ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  Timeout: 864 bars -> DROP
  N=9 multi-position (virtual slots), direction_cap=8, Hedge mode
  MC: sign randomization 10k sims, seeds [42,123,7], max p < 0.01
  WF: 3-fold expanding window (re-scan IS per fold)
  Quality: edge>=21.8pp, WR>=60%, SL>=1.0%, min_trades>=25

Output:
  - results/uptrend_profit_study_p2.json
  - Console report with per-hypothesis GO/STOP
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
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# CONSTANTS
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10          # roundtrip fee in price-space
SLIPPAGE_BUFFER = 0.02  # 0.02% slippage buffer for ATR
MAX_BARS = 288           # 24h timeout for scanner-level trades
TIMEOUT_BARS = 864       # 72h portfolio timeout
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]
MAX_POSITIONS = 9
DIRECTION_CAP = 8        # v1.35.1 current
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars
BARS_PER_DAY = 288

# Quality thresholds
EDGE_THRESHOLD = 21.8
MC_THRESHOLD = 0.01
MIN_TRADES = 25
MAX_BASELINE_WR = 70.0
SL_MIN = 1.0

# TP/SL grids (MAE/MFE percentile-based)
TP_PERCENTILES = [40, 50, 60, 70, 80]
SL_PERCENTILES = [70, 75, 80, 85, 90, 95]

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'uptrend_profit_study_p2.json')


# ============================================================
# CORE FUNCTIONS (from scanner protocol)
# ============================================================

def load_data():
    """Load and classify 270d CSV data."""
    df = pd.read_csv(DATA_FILE)
    types = df['type_code'].values.tolist()
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n_bars = len(df)
    print(f"Loaded {n_bars} bars ({n_bars / BARS_PER_DAY:.0f}d)", flush=True)
    print(f"Period: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}", flush=True)
    return df, types, opens, highs, lows, closes, n_bars


def fetch_api_data(df_csv):
    """Fetch recent BingX API data beyond CSV end. Returns (df_api, rctypes) or (None, None)."""
    try:
        import yaml
        import ccxt
    except ImportError:
        print("  WARNING: ccxt/yaml not available, skipping API fetch")
        return None, None

    cfg_path = os.path.join(PROJECT_ROOT, 'config', 'api_keys.yaml')
    if not os.path.exists(cfg_path):
        print("  WARNING: api_keys.yaml not found, skipping API fetch")
        return None, None

    with open(cfg_path) as f:
        keys = yaml.safe_load(f)
    bingx_keys = keys['bingx']['mainnet']
    ex = ccxt.bingx({
        'apiKey': bingx_keys['api_key'],
        'secret': bingx_keys['secret_key'],
        'options': {'defaultType': 'swap'},
        'enableRateLimit': True,
    })

    csv_end = pd.to_datetime(df_csv['timestamp'].iloc[-1])
    since_ts = int(csv_end.timestamp() * 1000) + 300000  # +5min

    all_bars = []
    remaining = 10000  # ~35 days max
    while remaining > 0:
        batch = ex.fetch_ohlcv('BTC-USDT', '5m', since=since_ts, limit=min(1000, remaining))
        if not batch:
            break
        all_bars.extend(batch)
        since_ts = batch[-1][0] + 300000
        remaining -= len(batch)
        if len(batch) < 100:
            break

    if not all_bars:
        print("  WARNING: No API data fetched")
        return None, None

    df_api = pd.DataFrame(all_bars, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    df_api = df_api.drop_duplicates(subset='ts_ms').sort_values('ts_ms').reset_index(drop=True)
    df_api['datetime'] = pd.to_datetime(df_api['ts_ms'], unit='ms')
    df_api['timestamp'] = df_api['datetime'].astype(str)

    # Classify candles using production classify_candle
    df_api['body'] = (df_api['close'] - df_api['open']).abs()
    df_api['avg_body'] = df_api['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()
    rctypes = []
    for _, row in df_api.iterrows():
        ct = classify_candle(row, row['avg_body'])
        rctypes.append(ct.value)

    print(f"  API data: {len(df_api)} bars ({len(df_api) / BARS_PER_DAY:.1f}d), "
          f"{df_api['datetime'].iloc[0]} ~ {df_api['datetime'].iloc[-1]}")
    return df_api, rctypes


def merge_csv_api(df_csv, types_csv, df_api, types_api):
    """Merge CSV and API data into a single dataset."""
    # Build combined arrays
    opens_csv = df_csv['open'].values.astype(np.float64)
    highs_csv = df_csv['high'].values.astype(np.float64)
    lows_csv = df_csv['low'].values.astype(np.float64)
    closes_csv = df_csv['close'].values.astype(np.float64)

    opens_api = df_api['open'].values.astype(np.float64)
    highs_api = df_api['high'].values.astype(np.float64)
    lows_api = df_api['low'].values.astype(np.float64)
    closes_api = df_api['close'].values.astype(np.float64)

    opens = np.concatenate([opens_csv, opens_api])
    highs = np.concatenate([highs_csv, highs_api])
    lows = np.concatenate([lows_csv, lows_api])
    closes = np.concatenate([closes_csv, closes_api])
    types = types_csv + types_api
    n_bars = len(types)

    print(f"  Merged: {len(df_csv)} + {len(df_api)} = {n_bars} bars "
          f"({n_bars / BARS_PER_DAY:.0f}d)")
    return types, opens, highs, lows, closes, n_bars


def compute_atr_ratio(highs, lows, closes):
    """ATR / rolling_median(ATR) ratio time series (Wilder EMA)."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= ATR_PERIOD:
        atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
        for i in range(ATR_PERIOD, n):
            atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    med = pd.Series(atr).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def build_signal_index(types, n):
    """Build index: pattern_name -> list of signal bar indices."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                    opens, highs, lows, n_bars,
                    atr_ratio, max_bars=MAX_BARS):
    """Backtest with ATR-scaled TP/SL. Timeout trades DROPPED."""
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
        if atr_ratio is not None and idx < len(atr_ratio) and not np.isnan(atr_ratio[idx]):
            r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[idx]))
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
        for j in range(idx + 2, min(idx + 2 + max_bars, n_bars)):
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
        # Timeout -> DROP
    return trades


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
                fav = highs[j] - entry
                adv = entry - lows[j]
            else:
                fav = entry - lows[j]
                adv = highs[j] - entry
            if fav > mfe:
                mfe = fav
            if adv > mae:
                mae = adv
        excursions.append({
            'mfe_pct': mfe / entry * 100,
            'mae_pct': mae / entry * 100,
        })
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


def scan_patterns_for_direction(signal_index, direction, edge_threshold,
                                 opens, highs, lows, n_bars, atr_ratio,
                                 bar_end=None):
    """Scan all patterns for a given direction with specified edge threshold.

    Uses MAE/MFE grid search (same as scanner v2.2).
    Returns list of pattern dicts passing all quality filters.
    """
    if bar_end is None:
        bar_end = n_bars

    selected = []
    for pat_name, all_sigs in signal_index.items():
        sigs = [s for s in all_sigs if s < bar_end]
        if len(sigs) < MIN_TRADES:
            continue

        excursions = compute_excursions(sigs, direction, opens, highs, lows, bar_end)
        if len(excursions) < MIN_TRADES:
            continue

        best = None
        best_score = -9999
        for tp_p in TP_PERCENTILES:
            for sl_p in SL_PERCENTILES:
                tp, sl = derive_tp_sl(excursions, tp_p, sl_p)
                if tp is None or sl is None or tp < 0.1 or sl < SL_MIN:
                    continue
                bwr = sl / (tp + sl) * 100
                if bwr > MAX_BASELINE_WR:
                    continue
                trades = bt_signals_atr(sigs, direction, tp, sl,
                                        opens, highs, lows, bar_end, atr_ratio)
                if len(trades) < MIN_TRADES:
                    continue
                pnls = [t[2] for t in trades]
                cum = 0
                pk = 0
                md = 0
                w = 0
                for p in pnls:
                    cum += p
                    if cum > pk:
                        pk = cum
                    dd = pk - cum
                    if dd > md:
                        md = dd
                    if p > 0:
                        w += 1
                wr = w / len(pnls) * 100
                score = (cum / md) if md > 0 else cum
                if score > best_score:
                    best_score = score
                    best = {
                        'tp': tp, 'sl': sl, 'tp_p': tp_p, 'sl_p': sl_p,
                        'trades': len(trades), 'wr': round(wr, 1),
                        'pnl': round(cum, 1), 'mdd': round(md, 1),
                        'baseline_wr': round(bwr, 1),
                        'edge': round(wr - bwr, 1),
                    }

        if best is None:
            continue
        if best['sl'] < SL_MIN:
            continue
        if best['edge'] < edge_threshold:
            continue

        # MC test
        trades_mc = bt_signals_atr(sigs, direction, best['tp'], best['sl'],
                                    opens, highs, lows, bar_end, atr_ratio)
        pnls_mc = [t[2] for t in trades_mc]
        mc_p = mc_test(pnls_mc)
        if mc_p >= MC_THRESHOLD:
            continue

        selected.append({
            'pattern': pat_name, 'direction': direction,
            'tp': best['tp'], 'sl': best['sl'],
            'tp_p': best['tp_p'], 'sl_p': best['sl_p'],
            'trades': best['trades'], 'wr': best['wr'],
            'edge': best['edge'], 'mc_p': round(mc_p, 4),
            'baseline_wr': best['baseline_wr'],
        })

    return selected


# ============================================================
# Portfolio Simulation (N=9, direction_cap=8, timeout=864)
# ============================================================

def generate_signals(types, patterns_long, patterns_short, n_bars, start_idx=0):
    """Generate signals from type codes. Returns list of (bar_idx, pattern, direction)."""
    signals = []
    for i in range(max(2, start_idx), n_bars):
        pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee,
                size_mult=1.0):
    """Check if position should exit at given bar. Returns trade dict or None."""
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None
    entry = opens[entry_bar]
    if entry <= 0:
        return None

    tp_pct = pos['tp_pct']
    sl_pct = pos['sl_pct']
    direction = pos['direction']
    sig_bar = pos['signal_bar']

    if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
    else:
        r = 1.0

    eff_tp = tp_pct * r + SLIPPAGE_BUFFER
    eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    hold = bar - entry_bar
    if hold >= TIMEOUT_BARS:
        # Timeout -> DROP (do not count as a trade)
        return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': 0,
                'reason': 'TIMEOUT', 'drop': True}

    h, l = highs[bar], lows[bar]
    if direction == 'LONG':
        hit_tp = h >= tp_price
        hit_sl = l <= sl_price
    else:
        hit_tp = l <= tp_price
        hit_sl = h >= sl_price

    if not hit_tp and not hit_sl:
        return None

    if hit_tp and hit_sl:
        tp_dist = abs(tp_price - opens[bar])
        sl_dist = abs(sl_price - opens[bar])
        if tp_dist <= sl_dist:
            exit_price, reason = tp_price, 'TP'
        else:
            exit_price, reason = sl_price, 'SL'
    elif hit_tp:
        exit_price, reason = tp_price, 'TP'
    else:
        exit_price, reason = sl_price, 'SL'

    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * LEVERAGE
    else:
        pnl = (1 - exit_price / entry) * 100 * LEVERAGE
    pnl -= fee

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
            'reason': reason, 'drop': False}


def simulate_portfolio(signals, tpsl_map, opens, highs, lows, n_bars,
                       atr_ratio, direction_cap, oos_start, oos_end,
                       size_mult_fn=None):
    """N=9 multi-position portfolio simulation.

    Args:
        size_mult_fn: optional function(bar, direction) -> float multiplier for position
                      sizing. Default None = uniform 1.0 for all positions.
                      Used by H3 for regime-aware sizing.

    Returns list of completed trade dicts. Timeout trades are DROPPED (not counted).
    """
    positions = []
    trades = []
    SIZE_PCT = 100.0 / MAX_POSITIONS
    fee = FEE_PCT * LEVERAGE

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(oos_start, oos_end):
        # Check exits
        closed_slots = []
        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee)
            if result is not None:
                if result.get('drop', False):
                    # Timeout -> DROP: remove position, don't count as trade
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                result['size_mult'] = pos.get('size_mult', 1.0)
                result['pnl_portfolio'] = result['pnl_slot'] * SIZE_PCT / 100 * result['size_mult']
                trades.append(result)
                closed_slots.append(pos['slot'])
        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Process signals
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= MAX_POSITIONS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                continue

            sm = 1.0
            if size_mult_fn is not None:
                sm = size_mult_fn(bar, direction)

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_sl[0],
                'sl_pct': tp_sl[1],
                'size_mult': sm,
            })

    # Force-close remaining at OOS end (as OOS_END, counted as trade)
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(oos_end - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        direction = pos['direction']
        if direction == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'pnl_slot': pnl, 'reason': 'OOS_END',
            'pattern': pos['pattern'], 'direction': direction,
            'size_mult': sm,
            'pnl_portfolio': pnl * SIZE_PCT / 100 * sm,
        })

    return trades


def compute_portfolio_metrics(trades):
    """Compute portfolio-level metrics from trade list."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0,
                'long_trades': 0, 'short_trades': 0,
                'long_pnl': 0, 'short_pnl': 0}

    wins = [t for t in trades if t['pnl_slot'] > 0]
    sorted_trades = sorted(trades, key=lambda x: x['entry_bar'])
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for t in sorted_trades:
        equity += t['pnl_portfolio']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    long_t = [t for t in trades if t['direction'] == 'LONG']
    short_t = [t for t in trades if t['direction'] == 'SHORT']
    return {
        'trades': len(trades),
        'wr': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'pnl': round(sum(t['pnl_portfolio'] for t in trades), 2),
        'mdd': round(max_dd, 2),
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'long_pnl': round(sum(t['pnl_portfolio'] for t in long_t), 2),
        'short_pnl': round(sum(t['pnl_portfolio'] for t in short_t), 2),
    }


def run_wf_portfolio(types, opens, highs, lows, closes, n_bars, atr_ratio,
                     patterns_long, patterns_short, tpsl_map,
                     direction_cap=DIRECTION_CAP, size_mult_fn=None,
                     label=""):
    """Run 3-fold expanding window WF on a given pattern set.

    Fold structure (expanding IS, equal OOS):
      n_folds+1 segments of equal size.
      Fold f: IS = [0, (f+1)*seg), OOS = [(f+1)*seg, (f+2)*seg)

    NOTE: This does NOT re-scan IS. It uses fixed pattern/TP-SL sets
    and validates OOS performance only. For pattern discovery WF,
    use scanner's expanding_window_wf().

    Returns dict with folds and summary.
    """
    n_folds = 3
    seg_size = n_bars // (n_folds + 1)

    all_signals = generate_signals(types, patterns_long, patterns_short, n_bars,
                                    start_idx=ATR_WARMUP)

    fold_results = []
    for fold in range(n_folds):
        oos_start = (fold + 1) * seg_size
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        trades = simulate_portfolio(
            all_signals, tpsl_map, opens, highs, lows, n_bars,
            atr_ratio, direction_cap, oos_start, oos_end,
            size_mult_fn=size_mult_fn,
        )
        m = compute_portfolio_metrics(trades)
        fold_results.append({
            'fold': fold + 1,
            'oos_start': oos_start,
            'oos_end': oos_end,
            'oos_bars': oos_end - oos_start,
            **m,
        })

    total_pnl = sum(f['pnl'] for f in fold_results)
    max_mdd = max(f['mdd'] for f in fold_results) if fold_results else 0
    all_positive = all(f['pnl'] > 0 for f in fold_results if f['trades'] > 0)
    n_positive = sum(1 for f in fold_results if f['pnl'] > 0)
    pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0

    return {
        'label': label,
        'n_folds': n_folds,
        'folds': fold_results,
        'total_pnl': round(total_pnl, 2),
        'max_mdd': round(max_mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'pass_3_3': all_positive,
        'n_positive': n_positive,
    }


def run_wf_with_rescanning(signal_index, types, opens, highs, lows, closes,
                            n_bars, long_edge_threshold, short_edge_threshold,
                            label=""):
    """Run 3-fold expanding window WF WITH re-scanning on each IS fold.

    This is the proper WF: each fold independently discovers patterns
    on IS data, then tests on OOS. This avoids look-ahead bias from
    using a fixed pattern set discovered on the full dataset.

    Returns dict with folds and summary.
    """
    n_folds = 3
    seg_size = n_bars // (n_folds + 1)

    fold_results = []
    for fold in range(n_folds):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        # Compute ATR on IS data only
        fold_atr = compute_atr_ratio(highs[:is_end], lows[:is_end], closes[:is_end])

        # Scan LONG patterns on IS with potentially different threshold
        long_pats = scan_patterns_for_direction(
            signal_index, 'LONG', long_edge_threshold,
            opens, highs, lows, n_bars, fold_atr, bar_end=is_end)

        # Scan SHORT patterns on IS with standard threshold
        short_pats = scan_patterns_for_direction(
            signal_index, 'SHORT', short_edge_threshold,
            opens, highs, lows, n_bars, fold_atr, bar_end=is_end)

        # Build tpsl map and pattern sets
        tpsl_map = {}
        patterns_long = set()
        patterns_short = set()
        for p in long_pats:
            tpsl_map[p['pattern']] = (p['tp'], p['sl'])
            patterns_long.add(p['pattern'])
        for p in short_pats:
            tpsl_map[p['pattern']] = (p['tp'], p['sl'])
            patterns_short.add(p['pattern'])

        # OOS ATR (up to oos_end)
        oos_atr = compute_atr_ratio(highs[:oos_end], lows[:oos_end], closes[:oos_end])

        # Generate signals and simulate on OOS
        all_signals = generate_signals(
            types, patterns_long, patterns_short, n_bars, start_idx=ATR_WARMUP)

        trades = simulate_portfolio(
            all_signals, tpsl_map, opens, highs, lows, n_bars,
            oos_atr, DIRECTION_CAP, oos_start, oos_end,
        )
        m = compute_portfolio_metrics(trades)

        fold_results.append({
            'fold': fold + 1,
            'is_end': is_end,
            'oos_start': oos_start,
            'oos_end': oos_end,
            'oos_bars': oos_end - oos_start,
            'is_long_patterns': len(long_pats),
            'is_short_patterns': len(short_pats),
            'long_edge_threshold': long_edge_threshold,
            'short_edge_threshold': short_edge_threshold,
            **m,
        })

        pnl_str = f"{m['pnl']:+.2f}%"
        print(f"    Fold {fold + 1}: IS=[0,{is_end}) OOS=[{oos_start},{oos_end}) | "
              f"{len(long_pats)}L+{len(short_pats)}S | "
              f"trades={m['trades']} WR={m['wr']}% PnL={pnl_str} MDD={m['mdd']:.1f}%")

    total_pnl = sum(f['pnl'] for f in fold_results)
    max_mdd = max(f['mdd'] for f in fold_results) if fold_results else 0
    all_positive = all(f['pnl'] > 0 for f in fold_results if f['trades'] > 0)
    n_positive = sum(1 for f in fold_results if f['pnl'] > 0)
    pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0

    return {
        'label': label,
        'n_folds': n_folds,
        'folds': fold_results,
        'total_pnl': round(total_pnl, 2),
        'max_mdd': round(max_mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'pass_3_3': all_positive,
        'n_positive': n_positive,
    }


# ============================================================
# H1: Direction-Specific Edge Threshold
# ============================================================

def run_h1(types, opens, highs, lows, closes, n_bars, atr_ratio, signal_index):
    """Test whether lowering edge threshold for LONG patterns improves portfolio.

    Method:
    1. Full scan of all patterns for LONG at thresholds 15, 17, 19, 21.8pp
    2. Count LONG patterns passing each threshold
    3. For each threshold, run 3-fold WF WITH re-scanning
       (each fold independently discovers LONG at lower threshold + SHORT at 21.8pp)
    4. Compare vs baseline (both at 21.8pp)
    """
    print("\n" + "=" * 80)
    print("H1: DIRECTION-SPECIFIC EDGE THRESHOLD")
    print("=" * 80)

    # Phase 1: Distribution analysis (full data, informational only)
    # Use a fast single-percentile probe (p60/p85) to estimate edge distribution
    print("\n--- Phase 1: LONG/SHORT Edge Distribution (270d IS, informational) ---",
          flush=True)

    long_edges = []
    short_edges = []
    pat_count = 0
    total_pats = len(signal_index)

    for pat_name, sigs in signal_index.items():
        pat_count += 1
        if pat_count % 200 == 0:
            print(f"  Scanning {pat_count}/{total_pats}...", flush=True)
        if len(sigs) < MIN_TRADES:
            continue
        for direction in ['LONG', 'SHORT']:
            excursions = compute_excursions(sigs, direction, opens, highs, lows, n_bars)
            if len(excursions) < MIN_TRADES:
                continue
            # Fast single-point edge estimate (p60 TP, p85 SL)
            tp, sl = derive_tp_sl(excursions, 60, 85)
            if tp is None or sl is None or tp < 0.1 or sl < SL_MIN:
                continue
            bwr = sl / (tp + sl) * 100
            if bwr > MAX_BASELINE_WR:
                continue
            trades = bt_signals_atr(sigs, direction, tp, sl,
                                    opens, highs, lows, n_bars, atr_ratio)
            if len(trades) < MIN_TRADES:
                continue
            pnls = [t[2] for t in trades]
            w = sum(1 for p in pnls if p > 0)
            wr = w / len(pnls) * 100
            edge = wr - bwr
            if edge > 0:
                if direction == 'LONG':
                    long_edges.append(edge)
                else:
                    short_edges.append(edge)

    long_edges = sorted(long_edges, reverse=True)
    short_edges = sorted(short_edges, reverse=True)

    print(f"LONG:  {len(long_edges)} patterns with positive edge")
    print(f"  >21.8pp: {sum(1 for e in long_edges if e >= 21.8)}")
    print(f"  >19pp:   {sum(1 for e in long_edges if e >= 19)}")
    print(f"  >17pp:   {sum(1 for e in long_edges if e >= 17)}")
    print(f"  >15pp:   {sum(1 for e in long_edges if e >= 15)}")
    if long_edges:
        print(f"  Mean: {np.mean(long_edges):.1f}pp  Median: {np.median(long_edges):.1f}pp")

    print(f"SHORT: {len(short_edges)} patterns with positive edge")
    print(f"  >21.8pp: {sum(1 for e in short_edges if e >= 21.8)}")
    if short_edges:
        print(f"  Mean: {np.mean(short_edges):.1f}pp  Median: {np.median(short_edges):.1f}pp")

    # Phase 2: WF with re-scanning at different LONG thresholds
    print("\n--- Phase 2: WF 3-fold with re-scanning ---")
    thresholds_to_test = [15.0, 17.0, 19.0, EDGE_THRESHOLD]
    wf_results = {}

    for long_thr in thresholds_to_test:
        label = f"LONG_{long_thr:.0f}pp_SHORT_{EDGE_THRESHOLD:.0f}pp"
        print(f"\n  Scenario: {label}")
        result = run_wf_with_rescanning(
            signal_index, types, opens, highs, lows, closes, n_bars,
            long_edge_threshold=long_thr,
            short_edge_threshold=EDGE_THRESHOLD,
            label=label,
        )
        wf_results[label] = result

    # Print comparison table
    print(f"\n{'Scenario':<35} {'Trades':>7} {'PnL':>9} {'MDD':>8} "
          f"{'PnL/MDD':>8} {'L_PnL':>8} {'S_PnL':>8} {'3/3?':>5}")
    print("-" * 100)

    baseline_label = f"LONG_{EDGE_THRESHOLD:.0f}pp_SHORT_{EDGE_THRESHOLD:.0f}pp"
    baseline_pnl = wf_results[baseline_label]['total_pnl']

    for long_thr in thresholds_to_test:
        label = f"LONG_{long_thr:.0f}pp_SHORT_{EDGE_THRESHOLD:.0f}pp"
        r = wf_results[label]
        total_trades = sum(f['trades'] for f in r['folds'])
        total_L = sum(f['long_pnl'] for f in r['folds'])
        total_S = sum(f['short_pnl'] for f in r['folds'])
        status = "PASS" if r['pass_3_3'] else "FAIL"
        fold_str = " | ".join(f"{f['pnl']:+.1f}" for f in r['folds'])
        print(f"{label:<35} {total_trades:>7} {r['total_pnl']:>+8.2f}% "
              f"{r['max_mdd']:>7.2f}% {r['pnl_mdd']:>7.2f}x {total_L:>+7.2f}% "
              f"{total_S:>+7.2f}% {r['n_positive']}/3 {status}  [{fold_str}]")

    # Delta analysis
    print(f"\n--- Delta vs Baseline ({EDGE_THRESHOLD:.0f}pp) ---")
    for long_thr in thresholds_to_test:
        if long_thr == EDGE_THRESHOLD:
            continue
        label = f"LONG_{long_thr:.0f}pp_SHORT_{EDGE_THRESHOLD:.0f}pp"
        delta = wf_results[label]['total_pnl'] - baseline_pnl
        status = "PASS" if wf_results[label]['pass_3_3'] else "FAIL"
        total_L = sum(f['long_pnl'] for f in wf_results[label]['folds'])
        total_L_base = sum(f['long_pnl'] for f in wf_results[baseline_label]['folds'])
        delta_L = total_L - total_L_base
        print(f"  LONG_{long_thr:.0f}pp: PnL delta={delta:+.2f}%  "
              f"LONG delta={delta_L:+.2f}%  {status}")

    # Verdict
    better = [f"LONG_{t:.0f}pp" for t in thresholds_to_test
              if t != EDGE_THRESHOLD
              and wf_results[f"LONG_{t:.0f}pp_SHORT_{EDGE_THRESHOLD:.0f}pp"]['pass_3_3']
              and wf_results[f"LONG_{t:.0f}pp_SHORT_{EDGE_THRESHOLD:.0f}pp"]['total_pnl'] > baseline_pnl]

    verdict = 'GO' if better else 'STOP'
    print(f"\nH1 Verdict: {verdict}")
    if better:
        print(f"  Better scenarios: {', '.join(better)}")
    else:
        print("  No relaxed LONG threshold improved OOS portfolio performance.")

    return {
        'hypothesis': 'H1_direction_specific_threshold',
        'verdict': verdict,
        'better_scenarios': better,
        'edge_distribution': {
            'long_count': len(long_edges),
            'short_count': len(short_edges),
            'long_above_21_8': sum(1 for e in long_edges if e >= 21.8),
            'long_above_19': sum(1 for e in long_edges if e >= 19),
            'long_above_17': sum(1 for e in long_edges if e >= 17),
            'long_above_15': sum(1 for e in long_edges if e >= 15),
            'long_mean': round(float(np.mean(long_edges)), 1) if long_edges else 0,
            'short_above_21_8': sum(1 for e in short_edges if e >= 21.8),
            'short_mean': round(float(np.mean(short_edges)), 1) if short_edges else 0,
        },
        'wf_results': {k: {kk: vv for kk, vv in v.items()}
                       for k, v in wf_results.items()},
    }


# ============================================================
# H2: Data Extension with API
# ============================================================

def run_h2(df_csv, types_csv, opens_csv, highs_csv, lows_csv, closes_csv,
           n_bars_csv, signal_index_csv):
    """Extend data with API, re-scan, compare LONG/SHORT distribution and WF.

    1. Fetch ~26d API data beyond 270d CSV
    2. Merge datasets -> 296d
    3. Scan all patterns at 21.8pp threshold on extended data
    4. Compare LONG/SHORT counts vs 270d-only
    5. Run WF 3-fold on extended data with fixed pattern set
    """
    print("\n" + "=" * 80)
    print("H2: DATA EXTENSION WITH API")
    print("=" * 80)

    # Phase 1: Fetch API data
    print("\n--- Phase 1: Fetch API data ---")
    df_api, types_api = fetch_api_data(df_csv)

    if df_api is None or types_api is None:
        print("  SKIP: Could not fetch API data")
        return {
            'hypothesis': 'H2_data_extension',
            'verdict': 'SKIP',
            'reason': 'API data fetch failed',
        }

    # Phase 2: Merge and analyze
    print("\n--- Phase 2: Merge CSV + API ---")
    types_ext, opens_ext, highs_ext, lows_ext, closes_ext, n_bars_ext = \
        merge_csv_api(df_csv, types_csv, df_api, types_api)

    atr_ext = compute_atr_ratio(highs_ext, lows_ext, closes_ext)
    si_ext = build_signal_index(types_ext, n_bars_ext)

    # Phase 3: Full scan on extended data (informational)
    print("\n--- Phase 3: Full scan comparison ---")

    # Scan 270d
    atr_270 = compute_atr_ratio(highs_csv, lows_csv, closes_csv)
    long_270 = scan_patterns_for_direction(
        signal_index_csv, 'LONG', EDGE_THRESHOLD,
        opens_csv, highs_csv, lows_csv, n_bars_csv, atr_270)
    short_270 = scan_patterns_for_direction(
        signal_index_csv, 'SHORT', EDGE_THRESHOLD,
        opens_csv, highs_csv, lows_csv, n_bars_csv, atr_270)

    # Scan extended
    long_ext = scan_patterns_for_direction(
        si_ext, 'LONG', EDGE_THRESHOLD,
        opens_ext, highs_ext, lows_ext, n_bars_ext, atr_ext)
    short_ext = scan_patterns_for_direction(
        si_ext, 'SHORT', EDGE_THRESHOLD,
        opens_ext, highs_ext, lows_ext, n_bars_ext, atr_ext)

    print(f"  270d: {len(long_270)}L + {len(short_270)}S = {len(long_270) + len(short_270)}")
    print(f"  {n_bars_ext // BARS_PER_DAY}d:  {len(long_ext)}L + {len(short_ext)}S = "
          f"{len(long_ext) + len(short_ext)}")

    delta_L = len(long_ext) - len(long_270)
    delta_S = len(short_ext) - len(short_270)
    print(f"  Delta: LONG {delta_L:+d}, SHORT {delta_S:+d}")

    # New LONG patterns in extended that weren't in 270d
    long_270_names = set(p['pattern'] for p in long_270)
    long_ext_names = set(p['pattern'] for p in long_ext)
    new_longs = long_ext_names - long_270_names
    lost_longs = long_270_names - long_ext_names
    if new_longs:
        print(f"  New LONGs in extended: {sorted(new_longs)}")
    if lost_longs:
        print(f"  Lost LONGs in extended: {sorted(lost_longs)}")

    # Phase 4: WF 3-fold on extended data
    print("\n--- Phase 4: WF 3-fold on extended data ---")

    # Build tpsl and pattern sets from extended scan
    tpsl_ext = {}
    pL_ext = set()
    pS_ext = set()
    for p in long_ext:
        tpsl_ext[p['pattern']] = (p['tp'], p['sl'])
        pL_ext.add(p['pattern'])
    for p in short_ext:
        tpsl_ext[p['pattern']] = (p['tp'], p['sl'])
        pS_ext.add(p['pattern'])

    wf_ext = run_wf_portfolio(
        types_ext, opens_ext, highs_ext, lows_ext, closes_ext, n_bars_ext,
        atr_ext, pL_ext, pS_ext, tpsl_ext,
        label=f"extended_{n_bars_ext // BARS_PER_DAY}d",
    )

    # Also WF with re-scanning on extended data (proper WF)
    print("\n  WF with re-scanning on extended data:")
    wf_ext_rescan = run_wf_with_rescanning(
        si_ext, types_ext, opens_ext, highs_ext, lows_ext, closes_ext,
        n_bars_ext,
        long_edge_threshold=EDGE_THRESHOLD,
        short_edge_threshold=EDGE_THRESHOLD,
        label=f"extended_rescan_{n_bars_ext // BARS_PER_DAY}d",
    )

    # Print summary
    print(f"\n  Fixed patterns WF: PnL={wf_ext['total_pnl']:+.2f}% "
          f"MDD={wf_ext['max_mdd']:.2f}% "
          f"{'PASS' if wf_ext['pass_3_3'] else 'FAIL'}")
    print(f"  Re-scanning WF:    PnL={wf_ext_rescan['total_pnl']:+.2f}% "
          f"MDD={wf_ext_rescan['max_mdd']:.2f}% "
          f"{'PASS' if wf_ext_rescan['pass_3_3'] else 'FAIL'}")

    # Verdict
    improved_long = delta_L >= 3  # at least 3 more LONG patterns
    wf_pass = wf_ext_rescan['pass_3_3']
    verdict = 'GO' if (improved_long and wf_pass) else 'STOP'
    print(f"\nH2 Verdict: {verdict}")
    if improved_long:
        print(f"  LONG count improved by {delta_L}")
    else:
        print(f"  LONG count delta: {delta_L} (need >= 3 for GO)")
    if not wf_pass:
        print(f"  WF 3/3 FAIL on extended data")

    return {
        'hypothesis': 'H2_data_extension',
        'verdict': verdict,
        'csv_bars': n_bars_csv,
        'extended_bars': n_bars_ext,
        'extended_days': round(n_bars_ext / BARS_PER_DAY, 1),
        'scan_comparison': {
            '270d_long': len(long_270),
            '270d_short': len(short_270),
            'ext_long': len(long_ext),
            'ext_short': len(short_ext),
            'delta_long': delta_L,
            'delta_short': delta_S,
            'new_longs': sorted(new_longs) if new_longs else [],
            'lost_longs': sorted(lost_longs) if lost_longs else [],
        },
        'wf_fixed': wf_ext,
        'wf_rescan': wf_ext_rescan,
    }


# ============================================================
# H3: Regime-Aware Position Sizing
# ============================================================

def run_h3(types, opens, highs, lows, closes, n_bars, atr_ratio, signal_index):
    """Test regime-aware position sizing without changing patterns.

    Keep existing 51 patterns + TP/SL unchanged.
    In UP regime (EMA slope > 0): SHORT size x multiplier
    In DOWN regime (EMA slope <= 0): LONG size x multiplier
    NEUTRAL: uniform sizing

    Test multiple EMA windows and sizing multipliers.
    Compare against baseline (uniform sizing).
    """
    print("\n" + "=" * 80)
    print("H3: REGIME-AWARE POSITION SIZING")
    print("=" * 80)

    # Load current patterns
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pL = set(pat_data['patterns']['long'])
    pS = set(pat_data['patterns']['short'])
    tpsl = {}
    for pat_name, tp_sl_list in pat_data['patterns_tpsl'].items():
        tpsl[pat_name] = (tp_sl_list[0], tp_sl_list[1])

    # Test configurations
    ema_windows = [20, 60, 120]  # ~1.7h, ~5h, ~10h in bars
    size_mults = [0.3, 0.5, 0.7]  # reduce counter-regime direction to this fraction
    slope_lookbacks = [5, 20]  # bars to compute slope

    # Precompute all EMA slopes
    ema_slopes = {}
    for ew in ema_windows:
        ema = pd.Series(closes).ewm(span=ew, adjust=False).mean().values
        for slb in slope_lookbacks:
            key = f"ema{ew}_lb{slb}"
            slope = np.full(n_bars, 0.0)
            for i in range(slb, n_bars):
                if ema[i - slb] > 0:
                    slope[i] = (ema[i] / ema[i - slb] - 1) * 100
            ema_slopes[key] = slope

    # Scenarios
    scenarios = [('baseline_uniform', None, None, None)]

    for ew in ema_windows:
        for slb in slope_lookbacks:
            for sm in size_mults:
                label = f"ema{ew}_lb{slb}_mult{sm}"
                scenarios.append((label, f"ema{ew}_lb{slb}", sm, ema_slopes[f"ema{ew}_lb{slb}"]))

    # Run WF for each scenario
    print(f"\nTesting {len(scenarios)} scenarios...")
    results = {}

    for label, slope_key, sm, slope in scenarios:
        if slope_key is None:
            # Baseline: uniform sizing
            wf = run_wf_portfolio(
                types, opens, highs, lows, closes, n_bars, atr_ratio,
                pL, pS, tpsl, label=label,
            )
        else:
            # Regime-aware sizing
            def make_size_fn(slope_arr, mult):
                def size_fn(bar, direction):
                    if bar >= len(slope_arr):
                        return 1.0
                    s = slope_arr[bar]
                    if s > 0 and direction == 'SHORT':
                        return mult  # Reduce SHORT in uptrend
                    elif s <= 0 and direction == 'LONG':
                        return mult  # Reduce LONG in downtrend
                    return 1.0
                return size_fn

            sfn = make_size_fn(slope, sm)
            wf = run_wf_portfolio(
                types, opens, highs, lows, closes, n_bars, atr_ratio,
                pL, pS, tpsl, label=label, size_mult_fn=sfn,
            )

        results[label] = wf

    # Print comparison table
    print(f"\n{'Scenario':<30} {'Trades':>7} {'PnL':>9} {'MDD':>8} "
          f"{'PnL/MDD':>8} {'L_PnL':>8} {'S_PnL':>8} {'3/3?':>5}")
    print("-" * 100)

    baseline_pnl = results['baseline_uniform']['total_pnl']

    for label, _, _, _ in scenarios:
        r = results[label]
        total_trades = sum(f['trades'] for f in r['folds'])
        total_L = sum(f['long_pnl'] for f in r['folds'])
        total_S = sum(f['short_pnl'] for f in r['folds'])
        status = "PASS" if r['pass_3_3'] else "FAIL"
        fold_str = " | ".join(f"{f['pnl']:+.1f}" for f in r['folds'])
        print(f"{label:<30} {total_trades:>7} {r['total_pnl']:>+8.2f}% "
              f"{r['max_mdd']:>7.2f}% {r['pnl_mdd']:>7.2f}x {total_L:>+7.2f}% "
              f"{total_S:>+7.2f}% {r['n_positive']}/3 {status}  [{fold_str}]")

    # Delta analysis
    print(f"\n--- Delta vs Baseline (uniform) ---")
    for label, slope_key, sm, _ in scenarios:
        if slope_key is None:
            continue
        delta = results[label]['total_pnl'] - baseline_pnl
        delta_mdd = results[label]['max_mdd'] - results['baseline_uniform']['max_mdd']
        status = "PASS" if results[label]['pass_3_3'] else "FAIL"
        print(f"  {label:<30} PnL delta={delta:+.2f}%  MDD delta={delta_mdd:+.2f}%  {status}")

    # Verdict: any regime scenario beat baseline on PnL/MDD AND pass 3/3?
    better = []
    for label, slope_key, sm, _ in scenarios:
        if slope_key is None:
            continue
        r = results[label]
        if r['pass_3_3'] and r['pnl_mdd'] > results['baseline_uniform']['pnl_mdd']:
            better.append(label)

    verdict = 'GO' if better else 'STOP'
    print(f"\nH3 Verdict: {verdict}")
    if better:
        best_label = max(better, key=lambda l: results[l]['pnl_mdd'])
        print(f"  Best scenario: {best_label} "
              f"(PnL/MDD={results[best_label]['pnl_mdd']:.2f}x vs "
              f"baseline {results['baseline_uniform']['pnl_mdd']:.2f}x)")
    else:
        print("  No regime-aware sizing improved PnL/MDD with WF 3/3 PASS.")

    # Compact results for JSON (avoid excessive size)
    compact_results = {}
    for label, _, _, _ in scenarios:
        r = results[label]
        total_L = sum(f['long_pnl'] for f in r['folds'])
        total_S = sum(f['short_pnl'] for f in r['folds'])
        compact_results[label] = {
            'total_pnl': r['total_pnl'],
            'max_mdd': r['max_mdd'],
            'pnl_mdd': r['pnl_mdd'],
            'pass_3_3': r['pass_3_3'],
            'long_pnl': round(total_L, 2),
            'short_pnl': round(total_S, 2),
            'fold_pnls': [round(f['pnl'], 2) for f in r['folds']],
        }

    return {
        'hypothesis': 'H3_regime_aware_sizing',
        'verdict': verdict,
        'better_scenarios': better,
        'wf_results': compact_results,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    # Force unbuffered output for progress visibility
    print("=" * 80, flush=True)
    print("UPTREND PROFIT STUDY -- Phase 2 (3 Hypotheses)", flush=True)
    print(f"Date: {datetime.now().isoformat()}", flush=True)
    print("=" * 80, flush=True)

    # Load 270d data
    df, types, opens, highs, lows, closes, n_bars = load_data()
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    signal_index = build_signal_index(types, n_bars)

    print(f"\nATR ratio: {np.sum(~np.isnan(atr_ratio)) / n_bars * 100:.1f}% valid", flush=True)
    print(f"Signal patterns: {len(signal_index)}", flush=True)

    # Run hypotheses
    h1_result = run_h1(types, opens, highs, lows, closes, n_bars, atr_ratio, signal_index)
    h2_result = run_h2(df, types, opens, highs, lows, closes, n_bars, signal_index)
    h3_result = run_h3(types, opens, highs, lows, closes, n_bars, atr_ratio, signal_index)

    elapsed = time.time() - t0

    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 2 FINAL SUMMARY")
    print("=" * 80)

    verdicts = {
        'H1_direction_specific_threshold': h1_result['verdict'],
        'H2_data_extension': h2_result['verdict'],
        'H3_regime_aware_sizing': h3_result['verdict'],
    }
    for name, v in verdicts.items():
        print(f"  {name:<40} {v}")

    go_count = sum(1 for v in verdicts.values() if v == 'GO')
    if go_count == 0:
        recommendation = (
            "STOP: No Phase 2 hypothesis produced actionable results. "
            "The LONG/SHORT structural asymmetry cannot be solved by threshold "
            "relaxation, data extension, or regime-aware sizing. "
            "Recommendation: Accept the SHORT-dominant structure. "
            "Current MDD sizing and direction_cap=8 already manage risk."
        )
    elif go_count >= 2:
        go_hyps = [k for k, v in verdicts.items() if v == 'GO']
        recommendation = (
            f"FURTHER RESEARCH: {go_count} hypotheses show promise "
            f"({', '.join(go_hyps)}). "
            "Validate on live OOS data before production deployment."
        )
    else:
        go_hyps = [k for k, v in verdicts.items() if v == 'GO']
        recommendation = (
            f"FURTHER RESEARCH: 1 hypothesis shows promise ({go_hyps[0]}). "
            "Needs extended live OOS validation before deployment."
        )

    print(f"\n  Recommendation: {recommendation}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save results
    output = {
        'study': 'uptrend_profit_study_p2',
        'date': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data_bars': n_bars,
        'data_days': round(n_bars / BARS_PER_DAY, 1),
        'protocol': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'timeout_bars': TIMEOUT_BARS,
            'max_positions': MAX_POSITIONS,
            'direction_cap': DIRECTION_CAP,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'clamp': [CLAMP_LO, CLAMP_HI],
            'edge_threshold': EDGE_THRESHOLD,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
        },
        'verdicts': verdicts,
        'recommendation': recommendation,
        'h1_direction_threshold': h1_result,
        'h2_data_extension': h2_result,
        'h3_regime_sizing': h3_result,
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
