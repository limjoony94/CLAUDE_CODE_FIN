#!/usr/bin/env python3
"""
ATR-Scaled TP/SL Walk-Forward Validation Study

Validates regime_filter_study_v2 finding: ATR-scaling TP/SL improves
per-trade edge by ~33% on true OOS (pre-overlap 437d).

Concept: scale each pattern's TP/SL by ATR_ratio = ATR(14) / median(ATR(14), window).
High volatility → wider TP/SL (match market conditions).
Low volatility  → tighter TP/SL (tighter targets).

Grid: 3 median windows × 4 clamp ranges × 2 ATR periods = 24 scenarios
WF: 3-fold expanding window on 720d Binance data
Also: TP-only / SL-only scaling (asymmetric) diagnostic
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
FEE_PCT = 0.10
LEVERAGE = 3
FEE = FEE_PCT * LEVERAGE   # 0.30% capital-space
MAX_BARS = 288
BARS_PER_DAY = 288

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'atr_scaled_tpsl_wf_study.json')

OVERLAP_TIMESTAMP = '2025-05-05 15:00:00'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('atr_scaled_wf')

_CandleRow = namedtuple('_CandleRow', ['open', 'high', 'low', 'close'])


# ===================================================================
# DATA (same as regime_filter_study_v2)
# ===================================================================

def load_and_classify(data_file):
    logger.info(f"Loading {os.path.basename(data_file)}")
    df = pd.read_csv(data_file)
    n = len(df)
    logger.info(f"  {n} bars")

    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows  = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)

    body_abs = np.abs(closes - opens)
    avg_body = pd.Series(body_abs).rolling(AVG_BODY_WINDOW).mean().values

    types = []
    for i in range(n):
        ab = avg_body[i] if not np.isnan(avg_body[i]) else 1.0
        types.append(classify_candle(_CandleRow(opens[i], highs[i], lows[i], closes[i]), ab).value)
        if i > 0 and i % 50000 == 0:
            logger.info(f"  Classified {i}/{n}")

    df['rctype'] = types
    logger.info("  Classification done")
    return df


def find_overlap_bar(df):
    if 'timestamp' in df.columns:
        mask = df['timestamp'] >= OVERLAP_TIMESTAMP
        if mask.any():
            return int(mask.values.argmax())
    return max(0, len(df) - 270 * BARS_PER_DAY)


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    return [{'pattern': v['pattern'], 'direction': v['direction'],
             'tp': float(v['tp']), 'sl': float(v['sl'])}
            for v in data['pattern_details'].values()]


# ===================================================================
# SIGNAL & BACKTEST ENGINE
# ===================================================================

def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def _bt_one(idx, is_long, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest single signal, return (eb, xb, pnl) or None."""
    eb = idx + 1
    if eb >= n_bars:
        return None
    entry = opens[eb]
    if entry <= 0:
        return None

    if is_long:
        tpp = entry * (1 + tp_pct / 100)
        slp = entry * (1 - sl_pct / 100)
    else:
        tpp = entry * (1 - tp_pct / 100)
        slp = entry * (1 + sl_pct / 100)

    end = min(idx + 2 + MAX_BARS, n_bars)
    for j in range(idx + 2, end):
        if is_long:
            ht = highs[j] >= tpp
            hs = lows[j] <= slp
        else:
            ht = lows[j] <= tpp
            hs = highs[j] >= slp

        if ht and hs:
            bo = opens[j]
            pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE
            return (eb, j, pnl, idx)
        elif ht:
            return (eb, j, tp_pct * LEVERAGE - FEE, idx)
        elif hs:
            return (eb, j, -sl_pct * LEVERAGE - FEE, idx)
    return None  # timeout → DROP


def generate_baseline_trades(patterns, sig_index, opens, highs, lows, n_bars):
    """All trades with fixed (unscaled) TP/SL. No 1-pos."""
    trades = []
    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue
        is_long = p['direction'] == 'LONG'
        for idx in bars:
            t = _bt_one(idx, is_long, p['tp'], p['sl'], opens, highs, lows, n_bars)
            if t:
                trades.append(t)
    return trades


def generate_scaled_trades(patterns, sig_index, opens, highs, lows, n_bars,
                           atr_ratio, clamp_lo, clamp_hi,
                           scale_tp=True, scale_sl=True):
    """All trades with ATR-scaled TP/SL. No 1-pos."""
    trades = []
    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue
        is_long = p['direction'] == 'LONG'
        base_tp, base_sl = p['tp'], p['sl']

        for idx in bars:
            r = atr_ratio[idx] if idx < len(atr_ratio) and not np.isnan(atr_ratio[idx]) else 1.0
            r = max(clamp_lo, min(clamp_hi, r))

            tp = base_tp * r if scale_tp else base_tp
            sl = base_sl * r if scale_sl else base_sl

            t = _bt_one(idx, is_long, tp, sl, opens, highs, lows, n_bars)
            if t:
                trades.append(t)
    return trades


# ===================================================================
# PORTFOLIO & METRICS
# ===================================================================

def portfolio_1pos(trades):
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


def calc_stats(trades):
    if not trades:
        return dict(trades=0, wr=0, pf=0, add_pnl=0, add_edge=0,
                    add_mdd=0, cmp_pnl=0, cmp_mdd=0)
    pnls = [t[2] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p >= 0]
    losses = [p for p in pnls if p < 0]

    add_pnl = sum(pnls)
    add_edge = add_pnl / n
    cum = peak_a = add_mdd = 0.0
    for p in pnls:
        cum += p
        if cum > peak_a: peak_a = cum
        dd = peak_a - cum
        if dd > add_mdd: add_mdd = dd

    eq = peak_c = 1.0
    cmp_mdd = 0.0
    for p in pnls:
        eq *= (1 + p / 100)
        if eq > peak_c: peak_c = eq
        dd = (peak_c - eq) / peak_c * 100
        if dd > cmp_mdd: cmp_mdd = dd

    wsum = sum(wins) if wins else 0
    lsum = sum(abs(x) for x in losses) if losses else 0
    return dict(
        trades=n, wr=round(len(wins) / n * 100, 1),
        pf=round(wsum / lsum, 2) if lsum > 0 else 999.0,
        add_pnl=round(add_pnl, 1), add_edge=round(add_edge, 3),
        add_mdd=round(add_mdd, 1),
        cmp_pnl=round((eq - 1) * 100, 1), cmp_mdd=round(cmp_mdd, 1),
    )


def filter_by_range(trades, lo, hi):
    return [t for t in trades if lo <= t[0] < hi]


def eval_period(trades, lo, hi):
    return calc_stats(portfolio_1pos(filter_by_range(trades, lo, hi)))


# ===================================================================
# ATR COMPUTATION
# ===================================================================

def compute_atr(highs, lows, closes, period=14):
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_atr_ratio(highs, lows, closes, atr_period=14, window=288):
    """ATR / rolling_median(ATR, window). Backward-looking only."""
    atr = compute_atr(highs, lows, closes, atr_period)
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(len(atr), np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


# ===================================================================
# SCENARIO GRID
# ===================================================================

def build_grid():
    """Build parameter grid for ATR-scaled TP/SL."""
    scenarios = []

    # Standard: scale both TP and SL
    for atr_p in [14, 21]:
        for window in [144, 288, 576]:
            for clo, chi in [(0.5, 2.0), (0.6, 1.7), (0.7, 1.5), (0.8, 1.25)]:
                scenarios.append(dict(
                    name=f'BOTH_a{atr_p}_w{window}_{clo}-{chi}',
                    atr_period=atr_p, window=window,
                    clamp_lo=clo, clamp_hi=chi,
                    scale_tp=True, scale_sl=True,
                ))

    # Asymmetric: scale only TP or only SL (diagnostic, window=288 only)
    for atr_p in [14]:
        for clo, chi in [(0.5, 2.0), (0.7, 1.5)]:
            scenarios.append(dict(
                name=f'TP_ONLY_a{atr_p}_w288_{clo}-{chi}',
                atr_period=atr_p, window=288,
                clamp_lo=clo, clamp_hi=chi,
                scale_tp=True, scale_sl=False,
            ))
            scenarios.append(dict(
                name=f'SL_ONLY_a{atr_p}_w288_{clo}-{chi}',
                atr_period=atr_p, window=288,
                clamp_lo=clo, clamp_hi=chi,
                scale_tp=False, scale_sl=True,
            ))

    return scenarios


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    if not os.path.isfile(DATA_FILE):
        logger.error(f"Data not found: {DATA_FILE}")
        return

    # ---- Load & classify ----
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens  = df['open'].values.astype(np.float64)
    highs  = df['high'].values.astype(np.float64)
    lows   = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    types  = df['rctype'].values

    ov_bar = find_overlap_bar(df)
    ov_len = n_bars - ov_bar
    is_end = ov_bar + int(ov_len * 0.7)

    logger.info(f"Bars: {n_bars}  Overlap: {ov_bar} ({ov_bar // BARS_PER_DAY}d)")
    logger.info(f"Pre-overlap [0, {ov_bar})  IS [{ov_bar}, {is_end})  OOS [{is_end}, {n_bars})")

    patterns = load_patterns()
    sig_index = build_signal_index(types, n_bars)
    logger.info(f"Patterns: {len(patterns)}")

    # ---- Baseline (unscaled) ----
    bl_trades = generate_baseline_trades(patterns, sig_index, opens, highs, lows, n_bars)
    logger.info(f"Baseline raw trades: {len(bl_trades)}")

    bl = {
        'pre':    eval_period(bl_trades, 0, ov_bar),
        'ov_is':  eval_period(bl_trades, ov_bar, is_end),
        'ov_oos': eval_period(bl_trades, is_end, n_bars),
        'full':   eval_period(bl_trades, 0, n_bars),
    }
    logger.info(f"Baseline pre: WR={bl['pre']['wr']}% edge={bl['pre']['add_edge']} "
                f"trades={bl['pre']['trades']}")

    # ---- Pre-compute ATR ratios ----
    logger.info("Computing ATR ratios...")
    atr_cache = {}
    for atr_p in [14, 21]:
        for w in [144, 288, 576]:
            key = (atr_p, w)
            atr_cache[key] = compute_atr_ratio(highs, lows, closes, atr_p, w)
    logger.info(f"ATR ratios cached: {len(atr_cache)}")

    # ---- WF fold definitions ----
    d90  = 90 * BARS_PER_DAY
    d270 = 270 * BARS_PER_DAY
    d450 = 450 * BARS_PER_DAY
    d720 = min(720 * BARS_PER_DAY, n_bars)

    wf_folds = [
        (0, d90, d270),    # Fold 1: IS 0-90d, OOS 90-270d
        (0, d270, d450),   # Fold 2: IS 0-270d, OOS 270-450d
        (0, d450, d720),   # Fold 3: IS 0-450d, OOS 450-720d
    ]

    # WF baselines per fold
    wf_bl = []
    for _, is_e, oos_e in wf_folds:
        wf_bl.append(eval_period(bl_trades, is_e, oos_e))

    # ---- Scenario grid ----
    scenarios = build_grid()
    logger.info(f"Scenarios: {len(scenarios)}")

    # ============================================================
    # EVALUATE ALL SCENARIOS
    # ============================================================
    logger.info("=" * 60)
    logger.info("Evaluating scenarios...")

    results = []
    for sc in scenarios:
        atr_ratio = atr_cache[(sc['atr_period'], sc['window'])]

        # Generate scaled trades
        sc_trades = generate_scaled_trades(
            patterns, sig_index, opens, highs, lows, n_bars,
            atr_ratio, sc['clamp_lo'], sc['clamp_hi'],
            sc['scale_tp'], sc['scale_sl'],
        )

        # Period evaluations
        pre    = eval_period(sc_trades, 0, ov_bar)
        ov_is  = eval_period(sc_trades, ov_bar, is_end)
        ov_oos = eval_period(sc_trades, is_end, n_bars)

        # Walk-Forward
        folds = []
        for fi, (is_s, is_e, oos_e) in enumerate(wf_folds):
            oos_st = eval_period(sc_trades, is_e, oos_e)
            bl_oos = wf_bl[fi]
            ed = round(oos_st['add_edge'] - bl_oos['add_edge'], 3) if bl_oos['trades'] else 0
            folds.append({
                'fold': fi + 1,
                'oos': f"{is_e // BARS_PER_DAY}-{oos_e // BARS_PER_DAY}d",
                'stats': oos_st,
                'bl_edge': bl_oos['add_edge'],
                'edge_d': ed,
                'wr_d': round(oos_st['wr'] - bl_oos['wr'], 1) if bl_oos['trades'] else 0,
            })

        pos_folds = sum(1 for f in folds if f['edge_d'] > 0)
        avg_ed = round(np.mean([f['edge_d'] for f in folds]), 3)
        verdict = 'PASS' if pos_folds >= 2 else 'FAIL'

        r = {
            'name': sc['name'],
            'params': sc,
            'pre': pre,
            'ov_is': ov_is,
            'ov_oos': ov_oos,
            'pre_edge_d': round(pre['add_edge'] - bl['pre']['add_edge'], 3),
            'is_edge_d': round(ov_is['add_edge'] - bl['ov_is']['add_edge'], 3),
            'wf_folds': folds,
            'wf_positive': pos_folds,
            'wf_avg_edge_d': avg_ed,
            'wf_verdict': verdict,
        }
        results.append(r)

        tag = "  " if sc['scale_tp'] and sc['scale_sl'] else "* "
        logger.info(f"{tag}{sc['name']}: pre_ed={pre['add_edge']:.3f} "
                     f"(d={r['pre_edge_d']:+.3f}) "
                     f"WF {pos_folds}/3 {verdict} avg_ed={avg_ed:+.3f}")

    # ============================================================
    # RANK & SUMMARY
    # ============================================================
    logger.info("=" * 60)
    logger.info("RANKING (by pre-overlap edge delta)")

    # Only full-scale (BOTH) scenarios for ranking
    both = [r for r in results if r['params']['scale_tp'] and r['params']['scale_sl']]
    both.sort(key=lambda r: -r['pre_edge_d'])

    logger.info(f"{'Name':<35} {'Pre_ed':>7} {'d':>7} {'WF':>5} {'avg_d':>7}")
    for r in both:
        logger.info(f"  {r['name']:<33} {r['pre']['add_edge']:>7.3f} "
                     f"{r['pre_edge_d']:>+7.3f} "
                     f"{r['wf_positive']}/3  {r['wf_avg_edge_d']:>+7.3f}")

    # Asymmetric results
    asym = [r for r in results if not (r['params']['scale_tp'] and r['params']['scale_sl'])]
    if asym:
        logger.info("\nAsymmetric scaling diagnostic:")
        for r in asym:
            mode = "TP_only" if r['params']['scale_tp'] else "SL_only"
            logger.info(f"  {r['name']:<33} pre_d={r['pre_edge_d']:+.3f} "
                         f"WF {r['wf_positive']}/3 {r['wf_verdict']}")

    # Best WF-passing scenario
    wf_pass = [r for r in both if r['wf_verdict'] == 'PASS']
    wf_pass.sort(key=lambda r: -r['wf_avg_edge_d'])

    best = wf_pass[0] if wf_pass else None
    if best:
        logger.info(f"\nBest WF-PASS: {best['name']}")
        logger.info(f"  Pre-overlap: edge {bl['pre']['add_edge']:.3f} → "
                     f"{best['pre']['add_edge']:.3f} ({best['pre_edge_d']:+.3f}, "
                     f"{best['pre_edge_d']/bl['pre']['add_edge']*100:+.1f}%)")
        logger.info(f"  IS: edge {bl['ov_is']['add_edge']:.3f} → "
                     f"{best['ov_is']['add_edge']:.3f} ({best['is_edge_d']:+.3f})")
        logger.info(f"  WF folds:")
        for f in best['wf_folds']:
            logger.info(f"    Fold {f['fold']} ({f['oos']}): "
                         f"edge {f['stats']['add_edge']:.3f} vs bl {f['bl_edge']:.3f} "
                         f"(d={f['edge_d']:+.3f})")
    else:
        logger.info("\nNo scenario passed WF.")

    # ============================================================
    # CONCLUSION
    # ============================================================
    conclusion = {
        'total_scenarios': len(scenarios),
        'wf_pass_count': len(wf_pass),
        'wf_pass_names': [r['name'] for r in wf_pass],
        'best': best['name'] if best else 'NONE',
        'best_pre_edge_d': best['pre_edge_d'] if best else 0,
        'best_wf_avg_d': best['wf_avg_edge_d'] if best else 0,
    }

    if best:
        # Production recommendation
        p = best['params']
        conclusion['recommendation'] = (
            f"ATR-scaled TP/SL with atr_period={p['atr_period']}, "
            f"window={p['window']}, clamp=[{p['clamp_lo']}, {p['clamp_hi']}] "
            f"improves per-trade edge by {best['pre_edge_d']:+.3f} "
            f"({best['pre_edge_d']/bl['pre']['add_edge']*100:+.1f}%) on true OOS. "
            f"WF {best['wf_positive']}/3 PASS, avg OOS edge delta {best['wf_avg_edge_d']:+.3f}."
        )
    else:
        conclusion['recommendation'] = (
            "ATR-scaled TP/SL does not pass WF validation. "
            "Maintain fixed per-pattern TP/SL."
        )

    logger.info("=" * 60)
    logger.info(f"CONCLUSION: {conclusion['recommendation']}")

    # ============================================================
    # OUTPUT
    # ============================================================
    elapsed = time.time() - t0
    output = {
        'study': 'atr_scaled_tpsl_wf_study',
        'version': '1.0',
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
        'wf_baseline_per_fold': [{'fold': i + 1, **b} for i, b in enumerate(wf_bl)],
        'scenarios_tested': len(scenarios),
        'results_both': [r for r in results if r['params']['scale_tp'] and r['params']['scale_sl']],
        'results_asymmetric': [r for r in results if not (r['params']['scale_tp'] and r['params']['scale_sl'])],
        'conclusion': conclusion,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved: {OUTPUT_FILE}")
    logger.info(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
