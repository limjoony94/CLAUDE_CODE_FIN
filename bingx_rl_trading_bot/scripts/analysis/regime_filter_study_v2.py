#!/usr/bin/env python3
"""
Regime Filter Study v2 — Market Quality Filters (Non-Directional)

Key changes from v1:
1. Non-directional market quality filters (not trend-following)
2. Signal-level filtering (before 1-pos constraint)
3. Additive PnL as primary metric + Compound as secondary
4. 720-day Binance data with true OOS pre-overlap period
5. Per-trade edge comparison

6 Filters:
  1. ATR Ratio Band (volatility regime)       — 8 scenarios
  2. ADX Trend Strength (non-directional)     — 8 scenarios
  3. Choppiness Index (market noise)          — 8 scenarios
  4. Hour-of-Day (session WR)                 — 3 scenarios
  5. Volume Ratio (liquidity)                 — 3 scenarios
  6. Drawdown Regime (adaptive, path-dep.)    — 5 scenarios

5 Phases:
  P1: 270d IS parameter tuning
  P2: 270d OOS holdout
  P3: 720d pre-overlap true OOS
  P4: 720d 3-fold expanding WF
  P5: Combinations + ATR-scaled TP/SL
"""

import os
import sys
import json
import time
import logging
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEE_PCT = 0.10            # Round-trip fee in price space (0.05% x 2)
LEVERAGE = 3
FEE = FEE_PCT * LEVERAGE  # 0.30% capital-space per trade
MAX_BARS = 288            # 24h timeout
BARS_PER_DAY = 288

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'regime_filter_study_v2.json')

OVERLAP_TIMESTAMP = '2025-05-05 15:00:00'
MIN_TRADES_OOS = 50

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('regime_filter_v2')

# Lightweight row for classify_candle (uses attribute access: row.open etc.)
_CandleRow = namedtuple('_CandleRow', ['open', 'high', 'low', 'close'])


# ===================================================================
# DATA LOADING
# ===================================================================

def load_and_classify(data_file):
    """Load 720d Binance CSV and reclassify with production classify_candle()."""
    logger.info(f"Loading {os.path.basename(data_file)}")
    df = pd.read_csv(data_file)
    n = len(df)
    logger.info(f"  {n} bars loaded")

    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)

    body_abs = np.abs(closes - opens)
    avg_body = pd.Series(body_abs).rolling(AVG_BODY_WINDOW).mean().values

    types = []
    for i in range(n):
        ab = avg_body[i]
        if np.isnan(ab):
            ab = 1.0
        row = _CandleRow(opens[i], highs[i], lows[i], closes[i])
        types.append(classify_candle(row, ab).value)
        if i > 0 and i % 50000 == 0:
            logger.info(f"  Classified {i}/{n} ({i*100//n}%)")

    df['rctype'] = types
    logger.info(f"  Classification done")

    # Verify overlap-period match with existing type_code
    if 'type_code' in df.columns and 'timestamp' in df.columns:
        mask = df['timestamp'] >= OVERLAP_TIMESTAMP
        if mask.any():
            ov = df.loc[mask]
            match_rate = (ov['rctype'] == ov['type_code']).mean()
            logger.info(f"  Overlap classification match: {match_rate*100:.1f}%")
            if match_rate < 0.99:
                logger.warning("  >1% mismatch — investigate!")

    return df


def find_overlap_bar(df):
    """Find bar index where 270d training data starts."""
    if 'timestamp' in df.columns:
        mask = df['timestamp'] >= OVERLAP_TIMESTAMP
        if mask.any():
            return int(mask.values.argmax())
    return max(0, len(df) - 270 * BARS_PER_DAY)


def load_patterns():
    """Load dynamic patterns from results/dynamic_patterns.json."""
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    pats = []
    for info in data['pattern_details'].values():
        pats.append({
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': float(info['tp']),
            'sl': float(info['sl']),
        })
    return pats


# ===================================================================
# SIGNAL & BACKTEST ENGINE
# ===================================================================

def build_signal_index(types, n):
    """Pattern -> list of signal bar indices."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals_full(signal_bars, direction, tp_pct, sl_pct,
                    opens, highs, lows, n_bars):
    """Backtest all signals WITHOUT 1-pos constraint.

    Returns list of (entry_bar, exit_bar, pnl%, signal_bar, direction).
    Timeout trades are DROPPED.
    """
    trades = []
    is_long = direction == 'LONG'
    for idx in signal_bars:
        eb = idx + 1
        if eb >= n_bars:
            continue
        entry = opens[eb]
        if entry <= 0:
            continue

        if is_long:
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        end_bar = min(idx + 2 + MAX_BARS, n_bars)
        for j in range(idx + 2, end_bar):
            if is_long:
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE
                trades.append((eb, j, pnl, idx, direction))
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE, idx, direction))
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE, idx, direction))
                break
        # timeout -> DROP
    return trades


def generate_all_trades(patterns, sig_index, opens, highs, lows, n_bars):
    """Generate ALL trades for all patterns (no 1-pos)."""
    all_t = []
    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue
        all_t.extend(bt_signals_full(bars, p['direction'], p['tp'], p['sl'],
                                     opens, highs, lows, n_bars))
    return all_t


def portfolio_1pos(trades):
    """Sort by entry bar, enforce single position at a time."""
    if not trades:
        return []
    s = sorted(trades, key=lambda t: t[0])
    out = []
    last_exit = -1
    for t in s:
        if t[0] > last_exit:
            out.append(t)
            last_exit = t[1]
    return out


# ===================================================================
# METRICS
# ===================================================================

def calc_stats(trades):
    """Additive + compound stats."""
    if not trades:
        return dict(trades=0, wr=0, pf=0,
                    add_pnl=0, add_edge=0, add_mdd=0,
                    cmp_pnl=0, cmp_mdd=0)
    pnls = [t[2] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p >= 0]
    losses = [p for p in pnls if p < 0]

    # Additive
    add_pnl = sum(pnls)
    add_edge = add_pnl / n
    cum = peak_a = add_mdd = 0.0
    for p in pnls:
        cum += p
        if cum > peak_a:
            peak_a = cum
        dd = peak_a - cum
        if dd > add_mdd:
            add_mdd = dd

    # Compound
    eq = peak_c = 1.0
    cmp_mdd = 0.0
    for p in pnls:
        eq *= (1 + p / 100)
        if eq > peak_c:
            peak_c = eq
        dd = (peak_c - eq) / peak_c * 100
        if dd > cmp_mdd:
            cmp_mdd = dd
    cmp_pnl = (eq - 1) * 100

    wsum = sum(wins) if wins else 0
    lsum = sum(abs(x) for x in losses) if losses else 0

    return dict(
        trades=n,
        wr=round(len(wins) / n * 100, 1),
        pf=round(wsum / lsum, 2) if lsum > 0 else 999.0,
        add_pnl=round(add_pnl, 1),
        add_edge=round(add_edge, 3),
        add_mdd=round(add_mdd, 1),
        cmp_pnl=round(cmp_pnl, 1),
        cmp_mdd=round(cmp_mdd, 1),
    )


def filter_by_range(trades, lo, hi):
    """Keep trades where entry_bar in [lo, hi)."""
    return [t for t in trades if lo <= t[0] < hi]


def compute_baseline(all_trades, lo, hi):
    return calc_stats(portfolio_1pos(filter_by_range(all_trades, lo, hi)))


# ===================================================================
# INDICATORS
# ===================================================================

def _true_range(highs, lows, closes):
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    return tr


def compute_atr(highs, lows, closes, period=14):
    n = len(highs)
    tr = _true_range(highs, lows, closes)
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_atr_ratio(highs, lows, closes, atr_period=14, window=288):
    atr = compute_atr(highs, lows, closes, atr_period)
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(len(atr), np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def compute_adx(highs, lows, closes, period=14):
    n = len(highs)
    adx = np.full(n, np.nan)
    if n < 2 * period + 1:
        return adx

    tr = _true_range(highs, lows, closes)
    pdm = np.zeros(n)
    ndm = np.zeros(n)
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            pdm[i] = up
        if down > up and down > 0:
            ndm[i] = down

    # Wilder smoothing — initial sums (bars 1..period)
    s_tr = tr[1:period + 1].sum()
    s_pdm = pdm[1:period + 1].sum()
    s_ndm = ndm[1:period + 1].sum()

    dx_arr = np.full(n, np.nan)
    for i in range(period, n):
        if i > period:
            s_tr = s_tr - s_tr / period + tr[i]
            s_pdm = s_pdm - s_pdm / period + pdm[i]
            s_ndm = s_ndm - s_ndm / period + ndm[i]
        if s_tr > 0:
            pdi = 100 * s_pdm / s_tr
            ndi = 100 * s_ndm / s_tr
            denom = pdi + ndi
            dx_arr[i] = 100 * abs(pdi - ndi) / denom if denom > 0 else 0
        else:
            dx_arr[i] = 0

    # Collect first `period` valid DX for initial ADX
    first_valid = []
    for i in range(period, n):
        if not np.isnan(dx_arr[i]):
            first_valid.append(i)
            if len(first_valid) == period:
                break
    if len(first_valid) < period:
        return adx

    start = first_valid[-1]
    adx_val = np.mean([dx_arr[j] for j in first_valid])
    adx[start] = adx_val
    for i in range(start + 1, n):
        if not np.isnan(dx_arr[i]):
            adx_val = (adx_val * (period - 1) + dx_arr[i]) / period
        adx[i] = adx_val
    return adx


def compute_choppiness(highs, lows, closes, period=14):
    n = len(highs)
    tr = _true_range(highs, lows, closes)
    ci = np.full(n, np.nan)
    log_p = np.log10(period)
    for i in range(period - 1, n):
        s = i - period + 1
        sum_tr = tr[s:i + 1].sum()
        hh = highs[s:i + 1].max()
        ll = lows[s:i + 1].min()
        rng = hh - ll
        if rng > 0 and sum_tr > 0:
            ci[i] = 100 * np.log10(sum_tr / rng) / log_p
    return ci


def compute_volume_ratio(volumes, window=288):
    rm = pd.Series(volumes).rolling(window, min_periods=window).mean().values
    ratio = np.full(len(volumes), np.nan)
    valid = (~np.isnan(rm)) & (rm > 0)
    ratio[valid] = volumes[valid] / rm[valid]
    return ratio


# ===================================================================
# FILTER APPLICATION
# ===================================================================

def _filt_band(trades, arr, lo, hi):
    """Keep trades where arr[signal_bar] in [lo, hi]."""
    out = []
    na = len(arr)
    for t in trades:
        sb = t[3]
        if sb < na:
            v = arr[sb]
            if not np.isnan(v) and lo <= v <= hi:
                out.append(t)
    return out


def _filt_above(trades, arr, thresh):
    """Keep trades where arr[signal_bar] >= thresh."""
    out = []
    na = len(arr)
    for t in trades:
        sb = t[3]
        if sb < na:
            v = arr[sb]
            if not np.isnan(v) and v >= thresh:
                out.append(t)
    return out


def _filt_below(trades, arr, thresh):
    """Keep trades where arr[signal_bar] <= thresh."""
    out = []
    na = len(arr)
    for t in trades:
        sb = t[3]
        if sb < na:
            v = arr[sb]
            if not np.isnan(v) and v <= thresh:
                out.append(t)
    return out


def _compute_hod_wr(trades, timestamps):
    """WR per 2-hour block from trade list."""
    blocks = {}
    for t in trades:
        sb = t[3]
        if sb < len(timestamps):
            hour = int(str(timestamps[sb])[11:13])
            blk = hour // 2
            if blk not in blocks:
                blocks[blk] = [0, 0]
            blocks[blk][1] += 1
            if t[2] >= 0:
                blocks[blk][0] += 1
    wr = {}
    for blk, (w, tot) in blocks.items():
        wr[blk] = w / tot if tot >= 5 else 0.0
    return wr


def _filt_hod(trades, timestamps, allowed_blocks):
    out = []
    for t in trades:
        sb = t[3]
        if sb < len(timestamps):
            hour = int(str(timestamps[sb])[11:13])
            if hour // 2 in allowed_blocks:
                out.append(t)
    return out


def _filt_drawdown(trades, dd_pct, pause_bars):
    """Path-dependent drawdown pause. Applied AFTER 1-pos."""
    if not trades:
        return []
    s = sorted(trades, key=lambda t: t[0])
    out = []
    cum = peak = 0.0
    pause_until = -1
    for t in s:
        if t[0] <= pause_until:
            continue
        out.append(t)
        cum += t[2]
        if cum > peak:
            peak = cum
        if peak - cum >= dd_pct:
            pause_until = t[0] + pause_bars
    return out


# ===================================================================
# SCENARIO DEFINITIONS
# ===================================================================

def build_scenarios():
    scenarios = []
    # 1. ATR Ratio Band: 2 windows x 4 bands = 8
    for w in [288, 576]:
        for lo, hi in [(0.5, 1.5), (0.6, 1.4), (0.7, 1.3), (0.5, 2.0)]:
            scenarios.append(dict(
                name=f'ATR_w{w}_{lo}-{hi}', ftype='atr_ratio',
                params=dict(window=w, lo=lo, hi=hi)))

    # 2. ADX Trend Strength: 2 periods x 4 thresholds = 8
    for p in [14, 20]:
        for th in [15, 20, 25, 30]:
            scenarios.append(dict(
                name=f'ADX_p{p}_t{th}', ftype='adx',
                params=dict(period=p, threshold=th)))

    # 3. Choppiness Index: 2 periods x 4 thresholds = 8
    for p in [14, 28]:
        for th in [50, 55, 60, 65]:
            scenarios.append(dict(
                name=f'CHOP_p{p}_t{th}', ftype='choppiness',
                params=dict(period=p, threshold=th)))

    # 4. Hour-of-Day: 3 scenarios
    for top_n in [8, 10, 12]:
        scenarios.append(dict(
            name=f'HOD_top{top_n}', ftype='hod',
            params=dict(top_n=top_n)))

    # 5. Volume Ratio: 3 scenarios
    for th in [0.3, 0.5, 0.7]:
        scenarios.append(dict(
            name=f'VOL_t{th}', ftype='volume',
            params=dict(threshold=th)))

    # 6. Drawdown Regime: 5 scenarios
    for dd, pb in [(5, 144), (5, 288), (10, 144), (10, 288), (15, 288)]:
        scenarios.append(dict(
            name=f'DD_{dd}pct_{pb}b', ftype='drawdown',
            params=dict(dd_pct=dd, pause_bars=pb)))

    return scenarios


# ===================================================================
# EVALUATION ENGINE
# ===================================================================

def apply_filter(trades, sc, indic, timestamps, is_trades_hod=None):
    """Apply one filter scenario (except drawdown which is post-1pos)."""
    ft = sc['ftype']
    p = sc['params']

    if ft == 'atr_ratio':
        return _filt_band(trades, indic[f"atr_w{p['window']}"], p['lo'], p['hi'])
    if ft == 'adx':
        return _filt_above(trades, indic[f"adx_p{p['period']}"], p['threshold'])
    if ft == 'choppiness':
        return _filt_below(trades, indic[f"chop_p{p['period']}"], p['threshold'])
    if ft == 'hod':
        ref = is_trades_hod if is_trades_hod is not None else trades
        wr = _compute_hod_wr(ref, timestamps)
        top = set(b for b, _ in sorted(wr.items(), key=lambda x: -x[1])[:p['top_n']])
        return _filt_hod(trades, timestamps, top)
    if ft == 'volume':
        return _filt_above(trades, indic['vol_ratio'], p['threshold'])
    if ft == 'drawdown':
        return trades  # handled post-1pos
    return trades


def evaluate(sc, all_trades, indic, timestamps, lo, hi, is_hod=None):
    """Evaluate a scenario on bar range [lo, hi).

    Flow: range-filter -> scenario-filter -> 1-pos -> stats
    Exception: drawdown applied after 1-pos.
    """
    rng = filter_by_range(all_trades, lo, hi)
    if sc['ftype'] == 'drawdown':
        pos = portfolio_1pos(rng)
        filt = _filt_drawdown(pos, sc['params']['dd_pct'], sc['params']['pause_bars'])
        return calc_stats(filt)
    filt = apply_filter(rng, sc, indic, timestamps, is_trades_hod=is_hod)
    return calc_stats(portfolio_1pos(filt))


# ===================================================================
# ATR-SCALED TP/SL (Phase 5 special)
# ===================================================================

def generate_atr_scaled_trades(patterns, sig_index, opens, highs, lows,
                               n_bars, atr_ratio):
    """Generate trades with TP/SL scaled by ATR ratio."""
    trades = []
    for pat in patterns:
        bars = sig_index.get(pat['pattern'])
        if not bars:
            continue
        is_long = pat['direction'] == 'LONG'
        base_tp, base_sl = pat['tp'], pat['sl']

        for idx in bars:
            eb = idx + 1
            if eb >= n_bars:
                continue
            entry = opens[eb]
            if entry <= 0:
                continue

            r = atr_ratio[idx] if idx < len(atr_ratio) and not np.isnan(atr_ratio[idx]) else 1.0
            r = max(0.5, min(2.0, r))
            tp_s = base_tp * r
            sl_s = base_sl * r

            if is_long:
                tpp = entry * (1 + tp_s / 100)
                slp = entry * (1 - sl_s / 100)
            else:
                tpp = entry * (1 - tp_s / 100)
                slp = entry * (1 + sl_s / 100)

            end_bar = min(idx + 2 + MAX_BARS, n_bars)
            for j in range(idx + 2, end_bar):
                if is_long:
                    ht = highs[j] >= tpp
                    hs = lows[j] <= slp
                else:
                    ht = lows[j] <= tpp
                    hs = highs[j] >= slp

                if ht and hs:
                    bo = opens[j]
                    pnl = (tp_s if abs(tpp - bo) <= abs(slp - bo) else -sl_s) * LEVERAGE - FEE
                    trades.append((eb, j, pnl, idx, pat['direction']))
                    break
                elif ht:
                    trades.append((eb, j, tp_s * LEVERAGE - FEE, idx, pat['direction']))
                    break
                elif hs:
                    trades.append((eb, j, -sl_s * LEVERAGE - FEE, idx, pat['direction']))
                    break
    return trades


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    if not os.path.isfile(DATA_FILE):
        logger.error(f"Data file not found: {DATA_FILE}")
        return

    # ---- Load & classify ----
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64)
    types = df['rctype'].values
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.array([''] * n_bars)

    # ---- Overlap detection ----
    ov_bar = find_overlap_bar(df)
    ov_len = n_bars - ov_bar
    is_end = ov_bar + int(ov_len * 0.7)

    logger.info(f"Bars: {n_bars}  Overlap bar: {ov_bar} ({ov_bar // BARS_PER_DAY}d)")
    logger.info(f"Pre-overlap [0, {ov_bar}) = {ov_bar // BARS_PER_DAY}d")
    logger.info(f"Overlap IS  [{ov_bar}, {is_end}) = {(is_end - ov_bar) // BARS_PER_DAY}d")
    logger.info(f"Overlap OOS [{is_end}, {n_bars}) = {(n_bars - is_end) // BARS_PER_DAY}d")

    # ---- Patterns & signals ----
    patterns = load_patterns()
    logger.info(f"Patterns: {len(patterns)}")
    sig_index = build_signal_index(types, n_bars)

    # ---- Generate all trades ----
    all_trades = generate_all_trades(patterns, sig_index, opens, highs, lows, n_bars)
    logger.info(f"Raw trades (no 1-pos): {len(all_trades)}")

    # ---- Compute indicators ----
    logger.info("Computing indicators...")
    indic = {}
    for w in [288, 576]:
        indic[f'atr_w{w}'] = compute_atr_ratio(highs, lows, closes, 14, w)
    for p in [14, 20]:
        indic[f'adx_p{p}'] = compute_adx(highs, lows, closes, p)
    for p in [14, 28]:
        indic[f'chop_p{p}'] = compute_choppiness(highs, lows, closes, p)
    indic['vol_ratio'] = compute_volume_ratio(volumes, 288)
    logger.info("Indicators done")

    # ---- Baselines ----
    bl = {
        'pre': compute_baseline(all_trades, 0, ov_bar),
        'ov_is': compute_baseline(all_trades, ov_bar, is_end),
        'ov_oos': compute_baseline(all_trades, is_end, n_bars),
        'full': compute_baseline(all_trades, 0, n_bars),
    }
    logger.info(f"Baseline pre: WR={bl['pre']['wr']}%, edge={bl['pre']['add_edge']}, "
                f"trades={bl['pre']['trades']}")
    logger.info(f"Baseline IS:  WR={bl['ov_is']['wr']}%, edge={bl['ov_is']['add_edge']}, "
                f"trades={bl['ov_is']['trades']}")

    scenarios = build_scenarios()
    logger.info(f"Scenarios: {len(scenarios)}")

    # IS 1-pos trades for HoD reference
    is_hod_ref = portfolio_1pos(filter_by_range(all_trades, ov_bar, is_end))

    # ==============================================================
    # PHASE 1: IS Parameter Tuning
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: IS Parameter Tuning (overlap IS)")

    p1_results = []
    for sc in scenarios:
        st = evaluate(sc, all_trades, indic, timestamps, ov_bar, is_end,
                      is_hod=is_hod_ref)
        b = bl['ov_is']
        p1_results.append({
            'name': sc['name'], 'ftype': sc['ftype'],
            'params': sc['params'], **st,
            'edge_d': round(st['add_edge'] - b['add_edge'], 3),
            'wr_d': round(st['wr'] - b['wr'], 1),
            'trade_pct': round(st['trades'] / b['trades'] * 100, 1) if b['trades'] else 0,
        })

    p1_results.sort(key=lambda r: -r['add_edge'])
    top10_names = [r['name'] for r in p1_results[:10]]

    logger.info("Top-10 by add_edge:")
    for r in p1_results[:10]:
        logger.info(f"  {r['name']}: edge={r['add_edge']:.3f} (d={r['edge_d']:+.3f}) "
                     f"WR={r['wr']}% trades={r['trades']} ({r['trade_pct']:.0f}%)")

    top10_sc = [sc for sc in scenarios if sc['name'] in top10_names]

    # ==============================================================
    # PHASE 2: OOS Holdout
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: OOS Holdout (overlap OOS)")

    p2_results = []
    for sc in top10_sc:
        st = evaluate(sc, all_trades, indic, timestamps, is_end, n_bars,
                      is_hod=is_hod_ref)
        b = bl['ov_oos']
        p2_results.append({
            'name': sc['name'], **st,
            'edge_d': round(st['add_edge'] - b['add_edge'], 3),
            'wr_d': round(st['wr'] - b['wr'], 1),
        })
        logger.info(f"  {sc['name']}: edge={st['add_edge']:.3f} "
                     f"(d={st['add_edge'] - b['add_edge']:+.3f}) WR={st['wr']}%")

    p2_results.sort(key=lambda r: -r['add_edge'])

    # ==============================================================
    # PHASE 3: True OOS (pre-overlap ~437d)
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: True OOS (pre-overlap)")

    p3_results = []
    for sc in top10_sc:
        st = evaluate(sc, all_trades, indic, timestamps, 0, ov_bar,
                      is_hod=is_hod_ref)
        b = bl['pre']
        if st['trades'] < MIN_TRADES_OOS:
            logger.info(f"  {sc['name']}: SKIP (trades={st['trades']} < {MIN_TRADES_OOS})")
            continue
        p3_results.append({
            'name': sc['name'], **st,
            'edge_d': round(st['add_edge'] - b['add_edge'], 3),
            'wr_d': round(st['wr'] - b['wr'], 1),
        })
        logger.info(f"  {sc['name']}: edge={st['add_edge']:.3f} "
                     f"(d={st['add_edge'] - b['add_edge']:+.3f}) "
                     f"WR={st['wr']}% trades={st['trades']}")

    p3_results.sort(key=lambda r: -r['add_edge'])

    # ==============================================================
    # PHASE 4: Walk-Forward (720d, 3-fold expanding)
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: Walk-Forward")

    d90 = 90 * BARS_PER_DAY
    d270 = 270 * BARS_PER_DAY
    d450 = 450 * BARS_PER_DAY
    d720 = min(720 * BARS_PER_DAY, n_bars)

    wf_folds = [
        (0, d90, d270),
        (0, d270, d450),
        (0, d450, d720),
    ]

    p4_results = {}
    for sc in top10_sc:
        folds = []
        for fi, (is_s, is_e, oos_e) in enumerate(wf_folds):
            # Fold-specific IS reference for HoD
            is_ref = portfolio_1pos(filter_by_range(all_trades, is_s, is_e))
            oos_st = evaluate(sc, all_trades, indic, timestamps, is_e, oos_e,
                              is_hod=is_ref)
            bl_oos = compute_baseline(all_trades, is_e, oos_e)
            ed = round(oos_st['add_edge'] - bl_oos['add_edge'], 3) if bl_oos['trades'] else 0
            folds.append({
                'fold': fi + 1,
                'is': f"{is_s // BARS_PER_DAY}-{is_e // BARS_PER_DAY}d",
                'oos': f"{is_e // BARS_PER_DAY}-{oos_e // BARS_PER_DAY}d",
                'stats': oos_st, 'bl': bl_oos, 'edge_d': ed,
                'wr_d': round(oos_st['wr'] - bl_oos['wr'], 1) if bl_oos['trades'] else 0,
            })

        pos = sum(1 for f in folds if f['edge_d'] > 0)
        avg_ed = round(np.mean([f['edge_d'] for f in folds]), 3)
        verdict = 'PASS' if pos >= 2 else 'FAIL'

        p4_results[sc['name']] = {
            'folds': folds, 'positive': pos,
            'avg_edge_d': avg_ed, 'verdict': verdict,
        }
        logger.info(f"  {sc['name']}: {pos}/3 {verdict}  avg_ed={avg_ed:+.3f}")

    # ==============================================================
    # PHASE 5: Combinations + ATR-scaled TP/SL
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Combinations + ATR-scaled TP/SL")

    wf_pass = [n for n, r in p4_results.items() if r['verdict'] == 'PASS']
    logger.info(f"WF-passing: {wf_pass}")

    # Select best WF-passing filter per type for cross-type combinations
    type_best = {}
    for name in wf_pass:
        sc = next(s for s in scenarios if s['name'] == name)
        ft = sc['ftype']
        ed = p4_results[name]['avg_edge_d']
        if ft not in type_best or ed > type_best[ft][1]:
            type_best[ft] = (name, ed)
    diverse_pass = [v[0] for v in type_best.values()]
    logger.info(f"Diverse WF-pass (best per type): {diverse_pass}")

    p5_combos = []
    if len(diverse_pass) >= 2:
        from itertools import combinations
        for n1, n2 in combinations(diverse_pass, 2):
            sc1 = next(s for s in scenarios if s['name'] == n1)
            sc2 = next(s for s in scenarios if s['name'] == n2)

            # Skip drawdown combos (path-dependent, no mixing)
            if sc1['ftype'] == 'drawdown' or sc2['ftype'] == 'drawdown':
                continue

            rng = filter_by_range(all_trades, 0, ov_bar)
            f1 = apply_filter(rng, sc1, indic, timestamps, is_trades_hod=is_hod_ref)
            f2 = apply_filter(f1, sc2, indic, timestamps, is_trades_hod=is_hod_ref)
            st = calc_stats(portfolio_1pos(f2))
            b = bl['pre']
            p5_combos.append({
                'combo': f"{n1} + {n2}", **st,
                'edge_d': round(st['add_edge'] - b['add_edge'], 3) if b['trades'] else 0,
            })
            logger.info(f"  {n1} + {n2}: edge={st['add_edge']:.3f} "
                         f"WR={st['wr']}% trades={st['trades']}")

    # ATR-scaled TP/SL
    logger.info("ATR-scaled TP/SL...")
    atr_arr = indic['atr_w288']
    atr_trades = generate_atr_scaled_trades(patterns, sig_index, opens, highs, lows,
                                            n_bars, atr_arr)
    atr_pre = calc_stats(portfolio_1pos(filter_by_range(atr_trades, 0, ov_bar)))
    atr_is = calc_stats(portfolio_1pos(filter_by_range(atr_trades, ov_bar, is_end)))
    p5_atr = {
        'name': 'ATR_SCALED_TPSL',
        'pre': atr_pre, 'is': atr_is,
        'pre_edge_d': round(atr_pre['add_edge'] - bl['pre']['add_edge'], 3) if bl['pre']['trades'] else 0,
        'is_edge_d': round(atr_is['add_edge'] - bl['ov_is']['add_edge'], 3) if bl['ov_is']['trades'] else 0,
    }
    logger.info(f"  ATR-scaled: pre edge={atr_pre['add_edge']:.3f} "
                f"(d={p5_atr['pre_edge_d']:+.3f})  "
                f"IS edge={atr_is['add_edge']:.3f} (d={p5_atr['is_edge_d']:+.3f})")

    # ==============================================================
    # CONCLUSION
    # ==============================================================
    best = wf_pass[0] if wf_pass else 'NONE'
    conclusion = {
        'wf_pass_count': len(wf_pass),
        'wf_pass_filters': wf_pass,
        'best_filter': best,
        'recommendation': (
            f"Best WF-passing filter: {best}" if wf_pass
            else "No filter passed WF. Maintain current unfiltered strategy."
        ),
    }
    logger.info("=" * 60)
    logger.info(f"CONCLUSION: {conclusion['recommendation']}")
    logger.info("=" * 60)

    # ==============================================================
    # OUTPUT
    # ==============================================================
    elapsed = time.time() - t0
    output = {
        'study': 'regime_filter_study_v2',
        'version': '2.0',
        'generated_at': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data': {
            'file': os.path.basename(DATA_FILE),
            'bars': n_bars,
            'overlap_bar': ov_bar,
            'pre_overlap_days': round(ov_bar / BARS_PER_DAY, 1),
        },
        'patterns': {'count': len(patterns), 'source': os.path.basename(PATTERNS_FILE)},
        'baseline': bl,
        'phase1_is': p1_results,
        'phase2_oos': p2_results,
        'phase3_pre': p3_results,
        'phase4_wf': p4_results,
        'phase5_combos': p5_combos,
        'phase5_atr_scaled': p5_atr,
        'conclusion': conclusion,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved: {OUTPUT_FILE}")
    logger.info(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
