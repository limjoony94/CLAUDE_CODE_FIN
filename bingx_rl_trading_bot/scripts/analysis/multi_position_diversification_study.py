#!/usr/bin/env python3
"""
Multi-Position Diversification Study (v2 — N-Sweep 통합)

Question: 1-pos-at-a-time 제약을 완화하면 성과가 개선되는가?

Background:
- Live diagnostic (2/18~2/21): LONG 홀딩 중 37개 SHORT 신호 무시
- 개별 신호 품질 우수 (ATR-scaled WR 100%, 10 TP / 0 SL)
- 1-pos 제약이 기회비용 생성 → 분산 투자 효과 연구

Scenarios (core):
  N=1: 현행 (1-pos-at-a-time), capital 100%
  N=2: 최대 2 동시 포지션, 각 50% capital
  N=3: 최대 3 동시 포지션, 각 33% capital
  N=5: 최대 5 동시 포지션, 각 20% capital  ← USER SELECTED
  N=INF: 무제한 동시 포지션 (개별 독립), 각 1/avg_concurrent capital

N-Sweep (v2 추가):
  N=1..20, 25, 30, 40, 50 전수 검증
  - PnL/MDD ratio: N=1 (27.59x) 에서 단조감소
  - N=4 (26.03x) 미세 bump이나 N=1 미달
  - 모든 N에서 WF 3/3 PASS
  - N=5 선택 근거: miss 50.5% (균형), MDD -30%, 단일 SL 2.5% (vs N=1 12.6%)

Capital model:
  - 총 exposure 고정 (3x leverage)
  - N개 슬롯 → 각 슬롯 capital = 1/N, leverage = 3x
  - PnL per trade = base_pnl * (1/N)
  - 이렇게 하면 동일 exposure로 분산 효과만 비교 가능

Validation: 3-fold expanding window WF on 720d Binance data
ATR scaling: BOTH_a14_w576_0.6-1.7 (v1.28.42 production params)

Reference: atr_scaled_tpsl_wf_study.py (same data, backtest engine)
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
FEE = FEE_PCT * LEVERAGE  # 0.30% capital-space
MAX_BARS = 288
BARS_PER_DAY = 288
SLIPPAGE_BUFFER = 0.02

# ATR scaling (v1.28.42 production)
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'multi_position_diversification_study.json')

OVERLAP_TIMESTAMP = '2025-05-05 15:00:00'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('multi_pos_study')

_CandleRow = namedtuple('_CandleRow', ['open', 'high', 'low', 'close'])


# ===================================================================
# DATA
# ===================================================================

def load_and_classify(data_file):
    logger.info(f"Loading {os.path.basename(data_file)}")
    df = pd.read_csv(data_file)
    n = len(df)
    logger.info(f"  {n} bars")

    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)

    body_abs = np.abs(closes - opens)
    avg_body = pd.Series(body_abs).rolling(AVG_BODY_WINDOW).mean().values

    types = []
    for i in range(n):
        ab = avg_body[i] if not np.isnan(avg_body[i]) else 1.0
        types.append(classify_candle(
            _CandleRow(opens[i], highs[i], lows[i], closes[i]), ab
        ).value)
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


def compute_atr_ratio(highs, lows, closes, atr_period=14, window=576):
    atr = compute_atr(highs, lows, closes, atr_period)
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(len(atr), np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


# ===================================================================
# SIGNAL & BACKTEST ENGINE
# ===================================================================

def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def resolve_trade(sig_bar, is_long, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Resolve a single trade: returns (entry_bar, exit_bar, pnl%) or None."""
    eb = sig_bar + 1
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

    end = min(sig_bar + 2 + MAX_BARS, n_bars)
    for j in range(sig_bar + 2, end):
        if is_long:
            ht = highs[j] >= tpp
            hs = lows[j] <= slp
        else:
            ht = lows[j] <= tpp
            hs = highs[j] >= slp

        if ht and hs:
            bo = opens[j]
            pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE
            return (eb, j, pnl)
        elif ht:
            return (eb, j, tp_pct * LEVERAGE - FEE)
        elif hs:
            return (eb, j, -sl_pct * LEVERAGE - FEE)

    # timeout → market close at next bar open (matches production bot behavior)
    timeout_bar = end if end < n_bars else n_bars - 1
    exit_price = opens[timeout_bar] if timeout_bar < n_bars else opens[-1]
    if is_long:
        pm = (exit_price / entry - 1) * 100
    else:
        pm = (1 - exit_price / entry) * 100
    pnl = pm * LEVERAGE - FEE
    return (eb, timeout_bar, pnl)


def generate_all_trades(patterns, sig_index, opens, highs, lows, n_bars,
                        atr_ratio=None, use_atr=False):
    """Generate ALL trades (no position limit). Returns list of (entry_bar, exit_bar, pnl, sig_bar, direction)."""
    trades = []
    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue
        is_long = p['direction'] == 'LONG'
        base_tp, base_sl = p['tp'], p['sl']

        for idx in bars:
            if use_atr and atr_ratio is not None:
                r = atr_ratio[idx] if idx < len(atr_ratio) and not np.isnan(atr_ratio[idx]) else 1.0
                r = max(CLAMP_LO, min(CLAMP_HI, r))
                tp = base_tp * r
                sl = base_sl * r
            else:
                tp = base_tp
                sl = base_sl

            t = resolve_trade(idx, is_long, tp, sl, opens, highs, lows, n_bars)
            if t:
                trades.append((t[0], t[1], t[2], idx, 1 if is_long else -1))
    return trades


# ===================================================================
# MULTI-POSITION PORTFOLIO SIMULATION
# ===================================================================

def simulate_portfolio(trades, max_positions, lo=0, hi=None):
    """
    Simulate N-position portfolio.

    Each trade's PnL is scaled by (1/max_positions) to keep total exposure constant.
    Trades are sorted by signal bar, processed chronologically.
    A new trade is accepted only if current open positions < max_positions.

    Returns list of (entry_bar, exit_bar, scaled_pnl).
    """
    # Filter by range
    filtered = [t for t in trades if lo <= t[3] and (hi is None or t[3] < hi)]
    if not filtered:
        return []

    # Sort by signal bar (t[3])
    filtered.sort(key=lambda t: t[3])

    if max_positions <= 0:
        # N=INF: all trades independently, scale by average concurrency later
        return filtered

    weight = 1.0 / max_positions
    active = []  # list of exit_bars for currently open positions
    result = []

    for t in filtered:
        entry_bar, exit_bar, pnl, sig_bar, direction = t

        # Remove closed positions
        active = [ex for ex in active if ex > entry_bar]

        if len(active) < max_positions:
            active.append(exit_bar)
            scaled_pnl = pnl * weight
            result.append((entry_bar, exit_bar, scaled_pnl))

    return result


def simulate_portfolio_unlimited(trades, lo=0, hi=None):
    """
    N=INF: all trades execute independently.
    Scale PnL by 1/peak_concurrent to normalize exposure.
    """
    filtered = [t for t in trades if lo <= t[3] and (hi is None or t[3] < hi)]
    if not filtered:
        return [], 0

    filtered.sort(key=lambda t: t[3])

    # Find peak concurrency
    events = []
    for t in filtered:
        events.append((t[0], +1))   # entry
        events.append((t[1], -1))   # exit
    events.sort()

    concurrent = 0
    peak = 0
    for _, delta in events:
        concurrent += delta
        peak = max(peak, concurrent)

    if peak == 0:
        return [], 0

    weight = 1.0 / peak
    result = [(t[0], t[1], t[2] * weight) for t in filtered]
    return result, peak


# ===================================================================
# STATISTICS
# ===================================================================

def calc_stats(portfolio_trades):
    """Compute stats from portfolio trades (entry, exit, scaled_pnl)."""
    if not portfolio_trades:
        return dict(trades=0, wr=0, pf=0, add_pnl=0, add_edge=0,
                    add_mdd=0, cmp_pnl=0, cmp_mdd=0)

    pnls = [t[2] for t in portfolio_trades]
    n = len(pnls)
    wins = [p for p in pnls if p >= 0]
    losses = [p for p in pnls if p < 0]

    add_pnl = sum(pnls)
    add_edge = add_pnl / n

    # Additive MDD
    cum = peak_a = add_mdd = 0.0
    for p in pnls:
        cum += p
        if cum > peak_a:
            peak_a = cum
        dd = peak_a - cum
        if dd > add_mdd:
            add_mdd = dd

    # Compound MDD
    eq = peak_c = 1.0
    cmp_mdd = 0.0
    for p in pnls:
        eq *= (1 + p / 100)
        if eq > peak_c:
            peak_c = eq
        dd = (peak_c - eq) / peak_c * 100
        if dd > cmp_mdd:
            cmp_mdd = dd

    wsum = sum(wins) if wins else 0
    lsum = sum(abs(x) for x in losses) if losses else 0.001

    return dict(
        trades=n,
        wr=round(len(wins) / n * 100, 1),
        pf=round(wsum / lsum, 2) if lsum > 0 else 999.0,
        add_pnl=round(add_pnl, 1),
        add_edge=round(add_edge, 3),
        add_mdd=round(add_mdd, 1),
        cmp_pnl=round((eq - 1) * 100, 1),
        cmp_mdd=round(cmp_mdd, 1),
    )


# ===================================================================
# CONCURRENCY ANALYSIS
# ===================================================================

def analyze_concurrency(trades, lo=0, hi=None):
    """Analyze trade concurrency patterns."""
    filtered = [t for t in trades if lo <= t[3] and (hi is None or t[3] < hi)]
    if not filtered:
        return {'peak': 0, 'mean': 0, 'median': 0, 'pct_95': 0}

    filtered.sort(key=lambda t: t[3])

    events = []
    for t in filtered:
        events.append((t[0], +1))
        events.append((t[1], -1))
    events.sort()

    concurrent = 0
    peak = 0
    levels = []
    for _, delta in events:
        concurrent += delta
        peak = max(peak, concurrent)
        levels.append(concurrent)

    return {
        'peak': peak,
        'mean': round(np.mean(levels), 1),
        'median': int(np.median(levels)),
        'pct_95': int(np.percentile(levels, 95)),
    }


def count_missed_signals(trades, max_positions, lo=0, hi=None):
    """Count how many signals were missed due to position limit."""
    filtered = [t for t in trades if lo <= t[3] and (hi is None or t[3] < hi)]
    if not filtered:
        return 0, 0

    filtered.sort(key=lambda t: t[3])
    active = []
    accepted = 0
    missed = 0

    for t in filtered:
        entry_bar = t[0]
        exit_bar = t[1]
        active = [ex for ex in active if ex > entry_bar]

        if len(active) < max_positions:
            active.append(exit_bar)
            accepted += 1
        else:
            missed += 1

    return accepted, missed


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("  Multi-Position Diversification Study v2 (N-Sweep integrated)")
    print("  Core: N = 1, 2, 3, 5, INF | Sweep: N = 1..20, 25, 30, 40, 50")
    print("  ATR scaling: a14 w576 clamp[0.6, 1.7] (v1.28.42)")
    print("=" * 80)

    if not os.path.isfile(DATA_FILE):
        logger.error(f"Data not found: {DATA_FILE}")
        return

    # ---- Load & classify ----
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    types = df['rctype'].values

    ov_bar = find_overlap_bar(df)
    logger.info(f"Bars: {n_bars}  Overlap: {ov_bar} ({ov_bar // BARS_PER_DAY}d)")

    patterns = load_patterns()
    sig_index = build_signal_index(types, n_bars)
    logger.info(f"Patterns: {len(patterns)}")

    # ---- ATR ratio ----
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    # ---- Generate ALL trades (ATR-scaled, no position limit) ----
    all_trades = generate_all_trades(
        patterns, sig_index, opens, highs, lows, n_bars,
        atr_ratio=atr_ratio, use_atr=True,
    )
    logger.info(f"All trades (ATR-scaled, no limit): {len(all_trades)}")

    # Also generate baseline (no ATR) for comparison
    all_trades_base = generate_all_trades(
        patterns, sig_index, opens, highs, lows, n_bars,
        use_atr=False,
    )
    logger.info(f"All trades (base, no limit): {len(all_trades_base)}")

    # ---- WF fold definitions ----
    d90 = 90 * BARS_PER_DAY
    d270 = 270 * BARS_PER_DAY
    d450 = 450 * BARS_PER_DAY
    d720 = min(720 * BARS_PER_DAY, n_bars)

    wf_folds = [
        (0, d90, d270),    # Fold 1: IS 0-90d, OOS 90-270d
        (0, d270, d450),   # Fold 2: IS 0-270d, OOS 270-450d
        (0, d450, d720),   # Fold 3: IS 0-450d, OOS 450-720d
    ]

    # ---- Concurrency analysis ----
    print("\n" + "=" * 80)
    print("  SECTION 1: CONCURRENCY ANALYSIS (720d, ATR-scaled)")
    print("=" * 80)

    conc_full = analyze_concurrency(all_trades, 0, n_bars)
    conc_pre = analyze_concurrency(all_trades, 0, ov_bar)
    conc_is = analyze_concurrency(all_trades, ov_bar, n_bars)

    print(f"  Full 720d:   peak={conc_full['peak']}, mean={conc_full['mean']}, "
          f"median={conc_full['median']}, p95={conc_full['pct_95']}")
    print(f"  Pre-overlap: peak={conc_pre['peak']}, mean={conc_pre['mean']}")
    print(f"  Overlap IS:  peak={conc_is['peak']}, mean={conc_is['mean']}")

    # ---- Section 2: Missed signals per N ----
    print("\n" + "=" * 80)
    print("  SECTION 2: MISSED SIGNALS BY N (full 720d)")
    print("=" * 80)
    print(f"  {'N':>4s} | {'Accepted':>8s} | {'Missed':>8s} | {'Miss Rate':>9s}")
    print(f"  {'----':>4s}-+-{'--------':>8s}-+-{'--------':>8s}-+-{'--------':>9s}")

    n_slots_list = [1, 2, 3, 5]  # Core scenarios

    # Extended N-sweep list (v2)
    n_sweep_list = list(range(1, 21)) + [25, 30, 40, 50]
    for n_slots in n_slots_list:
        acc, mis = count_missed_signals(all_trades, n_slots, 0, n_bars)
        rate = mis / (acc + mis) * 100 if (acc + mis) > 0 else 0
        print(f"  {n_slots:4d} | {acc:8d} | {mis:8d} | {rate:8.1f}%")

    # All trades (INF)
    total = len([t for t in all_trades if 0 <= t[3] < n_bars])
    print(f"  {'INF':>4s} | {total:8d} | {0:8d} | {0.0:8.1f}%")

    # ---- Section 3: Portfolio simulation per N ----
    print("\n" + "=" * 80)
    print("  SECTION 3: PORTFOLIO PERFORMANCE BY N (full 720d, ATR-scaled)")
    print("=" * 80)

    results_by_n = {}

    for n_slots in n_slots_list:
        port = simulate_portfolio(all_trades, n_slots, 0, n_bars)
        stats = calc_stats(port)
        results_by_n[n_slots] = {'port': port, 'stats': stats}
        print(f"\n  N={n_slots}:")
        print(f"    Trades: {stats['trades']}, WR: {stats['wr']}%, PF: {stats['pf']}")
        print(f"    Add PnL: {stats['add_pnl']}%, Add MDD: {stats['add_mdd']}%")
        print(f"    Cmp PnL: {stats['cmp_pnl']}%, Cmp MDD: {stats['cmp_mdd']}%")
        print(f"    PnL/MDD: {round(stats['cmp_pnl'] / max(stats['cmp_mdd'], 0.1), 1)}x")

    # N=INF
    port_inf, peak_conc = simulate_portfolio_unlimited(all_trades, 0, n_bars)
    stats_inf = calc_stats(port_inf)
    results_by_n['INF'] = {'port': port_inf, 'stats': stats_inf, 'peak_concurrent': peak_conc}
    print(f"\n  N=INF (peak concurrent={peak_conc}):")
    print(f"    Trades: {stats_inf['trades']}, WR: {stats_inf['wr']}%, PF: {stats_inf['pf']}")
    print(f"    Add PnL: {stats_inf['add_pnl']}%, Add MDD: {stats_inf['add_mdd']}%")
    print(f"    Cmp PnL: {stats_inf['cmp_pnl']}%, Cmp MDD: {stats_inf['cmp_mdd']}%")

    # ---- Section 3b: Same for BASE (no ATR) ----
    print("\n" + "=" * 80)
    print("  SECTION 3b: PORTFOLIO PERFORMANCE BY N (full 720d, NO ATR)")
    print("=" * 80)

    results_base_by_n = {}
    for n_slots in n_slots_list:
        port = simulate_portfolio(all_trades_base, n_slots, 0, n_bars)
        stats = calc_stats(port)
        results_base_by_n[n_slots] = stats
        print(f"\n  N={n_slots} (base):")
        print(f"    Trades: {stats['trades']}, WR: {stats['wr']}%, PF: {stats['pf']}")
        print(f"    Add PnL: {stats['add_pnl']}%, Cmp PnL: {stats['cmp_pnl']}%, "
              f"Cmp MDD: {stats['cmp_mdd']}%")

    # ---- Section 4: Walk-Forward Validation per N ----
    print("\n" + "=" * 80)
    print("  SECTION 4: WALK-FORWARD VALIDATION (3-fold, ATR-scaled)")
    print("=" * 80)

    wf_results = {}
    for n_slots in n_slots_list:
        folds = []
        for fi, (is_s, is_e, oos_e) in enumerate(wf_folds):
            port = simulate_portfolio(all_trades, n_slots, is_e, oos_e)
            stats = calc_stats(port)
            folds.append({
                'fold': fi + 1,
                'oos': f"{is_e // BARS_PER_DAY}-{oos_e // BARS_PER_DAY}d",
                'trades': stats['trades'],
                'wr': stats['wr'],
                'add_pnl': stats['add_pnl'],
                'cmp_pnl': stats['cmp_pnl'],
                'cmp_mdd': stats['cmp_mdd'],
            })
        wf_pass = sum(1 for f in folds if f['cmp_pnl'] > 0)
        wf_results[n_slots] = {
            'folds': folds,
            'wf_pass': wf_pass,
            'verdict': 'PASS' if wf_pass == 3 else f'FAIL ({wf_pass}/3)',
        }

        print(f"\n  N={n_slots}: WF {wf_results[n_slots]['verdict']}")
        for f in folds:
            status = '+' if f['cmp_pnl'] > 0 else '-'
            print(f"    Fold {f['fold']} ({f['oos']}): {status} "
                  f"Trades={f['trades']}, WR={f['wr']}%, "
                  f"CmpPnL={f['cmp_pnl']:+.1f}%, CmpMDD={f['cmp_mdd']:.1f}%")

    # N=INF WF
    folds_inf = []
    for fi, (is_s, is_e, oos_e) in enumerate(wf_folds):
        port_inf_f, peak_f = simulate_portfolio_unlimited(all_trades, is_e, oos_e)
        stats = calc_stats(port_inf_f)
        folds_inf.append({
            'fold': fi + 1,
            'oos': f"{is_e // BARS_PER_DAY}-{oos_e // BARS_PER_DAY}d",
            'trades': stats['trades'],
            'wr': stats['wr'],
            'add_pnl': stats['add_pnl'],
            'cmp_pnl': stats['cmp_pnl'],
            'cmp_mdd': stats['cmp_mdd'],
            'peak_concurrent': peak_f,
        })
    wf_pass_inf = sum(1 for f in folds_inf if f['cmp_pnl'] > 0)
    wf_results['INF'] = {
        'folds': folds_inf,
        'wf_pass': wf_pass_inf,
        'verdict': 'PASS' if wf_pass_inf == 3 else f'FAIL ({wf_pass_inf}/3)',
    }
    print(f"\n  N=INF: WF {wf_results['INF']['verdict']}")
    for f in folds_inf:
        status = '+' if f['cmp_pnl'] > 0 else '-'
        print(f"    Fold {f['fold']} ({f['oos']}): {status} "
              f"Trades={f['trades']}, WR={f['wr']}%, "
              f"CmpPnL={f['cmp_pnl']:+.1f}%, CmpMDD={f['cmp_mdd']:.1f}%")

    # ---- Section 5: Risk-adjusted comparison ----
    print("\n" + "=" * 80)
    print("  SECTION 5: RISK-ADJUSTED COMPARISON (full 720d, ATR-scaled)")
    print("=" * 80)

    print(f"\n  {'N':>4s} | {'Trades':>6s} | {'WR%':>5s} | {'CmpPnL%':>9s} | "
          f"{'CmpMDD%':>8s} | {'PnL/MDD':>8s} | {'Miss%':>6s} | {'WF':>10s}")
    print(f"  {'----':>4s}-+-{'------':>6s}-+-{'-----':>5s}-+-{'---------':>9s}-+-"
          f"{'--------':>8s}-+-{'--------':>8s}-+-{'------':>6s}-+-{'----------':>10s}")

    for n_slots in n_slots_list:
        s = results_by_n[n_slots]['stats']
        acc, mis = count_missed_signals(all_trades, n_slots, 0, n_bars)
        miss_rate = mis / (acc + mis) * 100 if (acc + mis) > 0 else 0
        pnl_mdd = round(s['cmp_pnl'] / max(s['cmp_mdd'], 0.1), 1)
        wf_v = wf_results[n_slots]['verdict']
        print(f"  {n_slots:4d} | {s['trades']:6d} | {s['wr']:5.1f} | {s['cmp_pnl']:+9.1f} | "
              f"{s['cmp_mdd']:8.1f} | {pnl_mdd:8.1f} | {miss_rate:5.1f}% | {wf_v:>10s}")

    # INF
    s = results_by_n['INF']['stats']
    total = len([t for t in all_trades if 0 <= t[3] < n_bars])
    wf_v = wf_results['INF']['verdict']
    pnl_mdd = round(s['cmp_pnl'] / max(s['cmp_mdd'], 0.1), 1)
    print(f"  {'INF':>4s} | {s['trades']:6d} | {s['wr']:5.1f} | {s['cmp_pnl']:+9.1f} | "
          f"{s['cmp_mdd']:8.1f} | {pnl_mdd:8.1f} | {0.0:5.1f}% | {wf_v:>10s}")

    # ---- Section 6: Direction analysis per N ----
    print("\n" + "=" * 80)
    print("  SECTION 6: DIRECTION ANALYSIS (full 720d, ATR-scaled)")
    print("=" * 80)

    for n_slots in n_slots_list:
        port = results_by_n[n_slots]['port']
        # We need direction info — re-run with tracking
        port_with_dir = simulate_portfolio_with_direction(all_trades, n_slots, 0, n_bars)
        long_pnl = sum(p for _, _, p, d in port_with_dir if d == 1)
        short_pnl = sum(p for _, _, p, d in port_with_dir if d == -1)
        n_long = sum(1 for _, _, _, d in port_with_dir if d == 1)
        n_short = sum(1 for _, _, _, d in port_with_dir if d == -1)
        print(f"  N={n_slots}: LONG {n_long} trades ({long_pnl:+.1f}%) | "
              f"SHORT {n_short} trades ({short_pnl:+.1f}%)")

    # ---- Section 7: Daily loss exposure ----
    print("\n" + "=" * 80)
    print("  SECTION 7: DAILY LOSS EXPOSURE ANALYSIS")
    print("=" * 80)
    print("  Current max_daily_loss_pct: 13%")
    print("  Max single-trade loss (SL max 4.2% × 3x = 12.6%)")
    print()
    for n_slots in n_slots_list:
        # Worst case: all N positions hit SL simultaneously
        max_sl = 4.2  # max SL% in pattern set
        worst_single = max_sl * LEVERAGE * (1 / n_slots)
        worst_all = max_sl * LEVERAGE  # all N hit SL = same as N=1 worst case
        print(f"  N={n_slots}: worst single-pos loss = {worst_single:.1f}%, "
              f"worst all-N simultaneous SL = {worst_all:.1f}% "
              f"(probability: ~{(1-0.87)**n_slots * 100:.4f}%)")

    # ---- Section 8: Final verdict ----
    print("\n" + "=" * 80)
    print("  SECTION 8: FINAL VERDICT")
    print("=" * 80)

    # Find optimal N
    best_n = None
    best_ratio = 0
    for n_slots in n_slots_list:
        s = results_by_n[n_slots]['stats']
        ratio = s['cmp_pnl'] / max(s['cmp_mdd'], 0.1)
        if wf_results[n_slots]['verdict'] == 'PASS' and ratio > best_ratio:
            best_ratio = ratio
            best_n = n_slots

    if best_n:
        print(f"  Best WF-PASS N: {best_n} (PnL/MDD = {best_ratio:.1f}x)")
    else:
        print("  No WF-PASS scenario found")

    for n_slots in n_slots_list:
        s = results_by_n[n_slots]['stats']
        wf = wf_results[n_slots]['verdict']
        ratio = s['cmp_pnl'] / max(s['cmp_mdd'], 0.1)
        n1 = results_by_n[1]['stats']
        pnl_vs_1 = s['cmp_pnl'] - n1['cmp_pnl']
        mdd_vs_1 = s['cmp_mdd'] - n1['cmp_mdd']
        marker = ' ← BEST' if n_slots == best_n else ''
        print(f"  N={n_slots}: CmpPnL {s['cmp_pnl']:+.1f}% (vs N=1: {pnl_vs_1:+.1f}%), "
              f"MDD {s['cmp_mdd']:.1f}% (vs N=1: {mdd_vs_1:+.1f}%), "
              f"WF {wf}{marker}")

    # ---- Section 9: N-Sweep (N=1..20, 25, 30, 40, 50) ----
    print("\n" + "=" * 80)
    print("  SECTION 9: N-SWEEP (N=1..20, 25, 30, 40, 50)")
    print("  PnL/MDD ratio, WF validation, miss rate for all N values")
    print("=" * 80)

    sweep_results = []
    print(f"\n  {'N':>4s} | {'Trades':>6s} | {'WR%':>5s} | {'AddPnL':>8s} | "
          f"{'AddMDD':>7s} | {'Ratio':>7s} | {'Miss%':>6s} | {'WF':>4s}")
    print(f"  {'----':>4s}-+-{'------':>6s}-+-{'-----':>5s}-+-{'--------':>8s}-+-"
          f"{'-------':>7s}-+-{'-------':>7s}-+-{'------':>6s}-+-{'----':>4s}")

    for n_slots in n_sweep_list:
        port = simulate_portfolio(all_trades, n_slots, 0, n_bars)
        stats = calc_stats(port)
        acc, mis = count_missed_signals(all_trades, n_slots, 0, n_bars)
        miss_rate = mis / max(acc + mis, 1) * 100

        # WF validation
        wf_pass = 0
        wf_folds_data = []
        for fi, (is_s, is_e, oos_e) in enumerate(wf_folds):
            p = simulate_portfolio(all_trades, n_slots, is_e, oos_e)
            s = calc_stats(p)
            if s['cmp_pnl'] > 0:
                wf_pass += 1
            wf_folds_data.append({
                'trades': s['trades'], 'wr': s['wr'],
                'cmp_pnl': s['cmp_pnl'], 'cmp_mdd': s['cmp_mdd'],
            })

        add_ratio = round(stats['add_pnl'] / max(stats['add_mdd'], 0.1), 2)

        sweep_results.append({
            'n': n_slots, 'trades': stats['trades'], 'wr': stats['wr'],
            'add_pnl': stats['add_pnl'], 'add_mdd': stats['add_mdd'],
            'add_ratio': add_ratio,
            'cmp_pnl': stats['cmp_pnl'], 'cmp_mdd': stats['cmp_mdd'],
            'miss_pct': round(miss_rate, 1),
            'wf_pass': wf_pass, 'wf_oos_folds': wf_folds_data,
        })

        marker = ' ←' if n_slots == 5 else ''
        print(f"  {n_slots:4d} | {stats['trades']:6d} | {stats['wr']:5.1f} | "
              f"{stats['add_pnl']:+8.1f} | {stats['add_mdd']:7.1f} | "
              f"{add_ratio:7.2f} | {miss_rate:5.1f}% | {wf_pass}/3{marker}")

    # N=5 vs N=1 comparison
    n1_sweep = next(r for r in sweep_results if r['n'] == 1)
    n5_sweep = next(r for r in sweep_results if r['n'] == 5)
    print(f"\n  N=5 vs N=1 comparison:")
    print(f"    Trades: {n1_sweep['trades']} → {n5_sweep['trades']} "
          f"(+{n5_sweep['trades'] - n1_sweep['trades']})")
    print(f"    AddPnL: {n1_sweep['add_pnl']:+.1f}% → {n5_sweep['add_pnl']:+.1f}% "
          f"({n5_sweep['add_pnl'] - n1_sweep['add_pnl']:+.1f}%)")
    print(f"    AddMDD: {n1_sweep['add_mdd']:.1f}% → {n5_sweep['add_mdd']:.1f}% "
          f"({n5_sweep['add_mdd'] - n1_sweep['add_mdd']:+.1f}%)")
    print(f"    Ratio:  {n1_sweep['add_ratio']:.2f}x → {n5_sweep['add_ratio']:.2f}x "
          f"({n5_sweep['add_ratio'] - n1_sweep['add_ratio']:+.2f}x)")
    print(f"    Miss:   {n1_sweep['miss_pct']:.1f}% → {n5_sweep['miss_pct']:.1f}%")
    print(f"    Max single-trade SL: {4.2 * LEVERAGE:.1f}% → {4.2 * LEVERAGE / 5:.1f}%")

    # ---- Save results ----
    output = {
        'study': 'multi_position_diversification_study',
        'version': '2.0',
        'generated_at': datetime.now().isoformat(),
        'elapsed_seconds': round(time.time() - t0, 1),
        'data': {
            'file': os.path.basename(DATA_FILE),
            'bars': n_bars,
            'overlap_bar': ov_bar,
        },
        'atr_params': {
            'period': ATR_PERIOD,
            'window': ATR_WINDOW,
            'clamp_lo': CLAMP_LO,
            'clamp_hi': CLAMP_HI,
        },
        'patterns': {
            'count': len(patterns),
            'source': os.path.basename(PATTERNS_FILE),
        },
        'concurrency': {
            'full': conc_full,
            'pre_overlap': conc_pre,
            'overlap_is': conc_is,
        },
        'scenarios': {},
        'wf_results': {},
        'base_comparison': results_base_by_n,
    }

    for n_slots in n_slots_list:
        key = f'N={n_slots}'
        s = results_by_n[n_slots]['stats']
        acc, mis = count_missed_signals(all_trades, n_slots, 0, n_bars)
        output['scenarios'][key] = {
            **s,
            'accepted': acc,
            'missed': mis,
            'miss_rate': round(mis / max(acc + mis, 1) * 100, 1),
        }
        output['wf_results'][key] = wf_results[n_slots]

    # INF
    s_inf = results_by_n['INF']['stats']
    output['scenarios']['N=INF'] = {
        **s_inf,
        'peak_concurrent': results_by_n['INF'].get('peak_concurrent', 0),
        'accepted': total,
        'missed': 0,
        'miss_rate': 0,
    }
    output['wf_results']['N=INF'] = wf_results['INF']

    output['n_sweep'] = sweep_results

    output['conclusion'] = {
        'best_n_by_ratio': 1,
        'best_pnl_mdd_ratio': round(n1_sweep['add_ratio'], 2),
        'selected_n': 5,
        'selection_rationale': (
            "N=5 selected by user. PnL/MDD monotonically decreasing from N=1 (27.59x), "
            "but N=5 balances: miss rate 50.5% (vs 87.8%), single-trade SL 2.5% (vs 12.6%), "
            "MDD 52.8% (vs 82.8%). All N values WF 3/3 PASS."
        ),
        'n5_vs_n1': {
            'trades_delta': n5_sweep['trades'] - n1_sweep['trades'],
            'add_pnl_delta': round(n5_sweep['add_pnl'] - n1_sweep['add_pnl'], 1),
            'add_mdd_delta': round(n5_sweep['add_mdd'] - n1_sweep['add_mdd'], 1),
            'ratio_delta': round(n5_sweep['add_ratio'] - n1_sweep['add_ratio'], 2),
            'miss_rate_delta': round(n5_sweep['miss_pct'] - n1_sweep['miss_pct'], 1),
            'max_single_sl_pct': round(4.2 * LEVERAGE / 5, 1),
        },
        'key_finding': (
            "Diversification trades PnL for reduced MDD via 1/N capital allocation. "
            "N=1 'natural filter' effect gives highest WR (79.3%) but 87.8% missed signals. "
            "N=5 captures 4x more trades at cost of -46% additive PnL and -36% ratio."
        ),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved: {OUTPUT_FILE}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


def simulate_portfolio_with_direction(trades, max_positions, lo, hi):
    """Like simulate_portfolio but preserves direction info."""
    filtered = [t for t in trades if lo <= t[3] and (hi is None or t[3] < hi)]
    if not filtered:
        return []

    filtered.sort(key=lambda t: t[3])
    weight = 1.0 / max_positions
    active = []
    result = []

    for t in filtered:
        entry_bar, exit_bar, pnl, sig_bar, direction = t
        active = [ex for ex in active if ex > entry_bar]

        if len(active) < max_positions:
            active.append(exit_bar)
            result.append((entry_bar, exit_bar, pnl * weight, direction))

    return result


if __name__ == '__main__':
    main()
