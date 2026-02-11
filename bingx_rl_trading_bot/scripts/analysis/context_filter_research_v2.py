"""
Context Filter Research v2
==========================
v1.27.0 포트폴리오의 context filter 효과 심층 연구

8-Phase Research:
  Phase 1: Context Feature Tagging — RSI/Vol/Trend 사전 계산 + 분포 검증
  Phase 2: Per-Pattern × Context 교차분석
  Phase 3: 통계 검정 + BH FDR 다중비교 보정
  Phase 4: Filter 후보 선정
  Phase 5: Portfolio Simulation (Filtered vs Unfiltered)
  Phase 6: MC + WF Validation
  Phase 7: Sensitivity Analysis
  Phase 8: Summary & Recommendation

핵심 리스크: 52패턴 × 3차원 × 2~3카테고리 = 수백개 가설 동시 검정 → 과적합 위험.
이미 WR 83.7%로 개선 여지 제한적. FAIL이 가장 유력한 결과.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Install with: pip install scipy")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL, PATTERN_STATS, BOT_VERSION,
    RSI_OVERSOLD_THRESHOLD, RSI_OVERBOUGHT_THRESHOLD,
    VOL_LOW_QUANTILE, VOL_HIGH_QUANTILE,
    TREND_LOOKBACK_BARS,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000

# Context thresholds (base — matching production constants)
RSI_OS_BASE = RSI_OVERSOLD_THRESHOLD   # 30
RSI_OB_BASE = RSI_OVERBOUGHT_THRESHOLD  # 70
VOL_Q_LOW_BASE = VOL_LOW_QUANTILE      # 0.33
VOL_Q_HIGH_BASE = VOL_HIGH_QUANTILE    # 0.66
TREND_LB_BASE = TREND_LOOKBACK_BARS    # 20

# Statistical thresholds
FDR_ALPHA = 0.10
MIN_CELL_TRADES = 10
WR_DIFF_THRESHOLD_PP = 15.0
MIN_TRADE_RETENTION = 0.50

# Context dimensions and their categories
DIMENSIONS = {
    'rsi_zone': ['OS', 'N', 'OB'],
    'vol': ['L', 'M', 'H'],
    'trend': ['UP', 'DN'],
}

t_start = time.time()

print("=" * 70)
print(f"Context Filter Research v2 (from v{BOT_VERSION})")
print("=" * 70)

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"Loaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

print("Done.\n")

# ============================================================
# Build portfolio (v1.27.0)
# ============================================================
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", tp, sl)

ALL_PATTERNS = sorted(PORTFOLIO.keys())
print(f"Portfolio: {len(PORTFOLIO)} patterns "
      f"({len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S)")


# ============================================================
# Reused backtest functions (from uniform_tp_validation.py)
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnls.append((tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT)
                break
            elif ht:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hs:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 5:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def walk_forward_test(indices, direction, tp_pct, sl_pct, n_folds=5):
    si = np.sort(indices)
    fs = len(si) // n_folds
    if fs < 3:
        return 0
    profitable = 0
    for f in range(n_folds):
        s = f * fs
        e = s + fs if f < n_folds - 1 else len(si)
        pnls = bt_fixed(si[s:e], direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


def collect_trades_1pos(pattern_map):
    raw_trades = []
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
        entry_bar = i + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp / 100)
            slp = entry * (1 - sl / 100)
        else:
            tpp = entry * (1 - tp / 100)
            slp = entry * (1 + sl / 100)
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                raw_trades.append((entry_bar, j, pnl))
                break
            elif ht:
                raw_trades.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                raw_trades.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for entry_bar, exit_bar, pnl in raw_trades:
        if entry_bar > last_exit:
            filtered.append((entry_bar, exit_bar, pnl))
            last_exit = exit_bar
    return filtered


def eval_trades_simple(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] for t in trades]
    cum_pnl = 0
    peak_pnl = 0
    max_dd = 0
    wins = 0
    win_pnls = []
    loss_pnls = []
    for p in pnl_list:
        cum_pnl += p
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_pnls.append(p)
        else:
            loss_pnls.append(abs(p))
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    total_win = sum(win_pnls)
    total_loss = sum(loss_pnls)
    wr_pct = wins / len(pnl_list) * 100
    expectancy = (wr_pct / 100 * avg_win) - ((1 - wr_pct / 100) * avg_loss)
    return {
        'pnl': cum_pnl,
        'trades': len(pnl_list),
        'wr': wr_pct,
        'mdd': max_dd,
        'pf': total_win / total_loss if total_loss > 0 else 999,
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'expectancy': float(expectancy),
        'pnl_mdd': cum_pnl / max_dd if max_dd > 0 else 999,
    }


# ============================================================
# New: Context computation
# ============================================================
def compute_context_arrays(rsi_os=RSI_OS_BASE, rsi_ob=RSI_OB_BASE,
                           vol_q_low=VOL_Q_LOW_BASE, vol_q_high=VOL_Q_HIGH_BASE,
                           trend_lookback=TREND_LB_BASE):
    """Pre-compute RSI zone, Vol level, Trend direction for all bars."""
    # RSI-14 (SMA-based, matching production indicators.py)
    delta = pd.Series(closes).diff().values
    delta[0] = 0
    gain = np.where(delta > 0, delta, 0).astype(float)
    loss_arr = np.where(delta < 0, -delta, 0).astype(float)
    avg_gain = pd.Series(gain).rolling(14, min_periods=14).mean().values
    avg_loss = pd.Series(loss_arr).rolling(14, min_periods=14).mean().values
    rsi = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))

    rsi_zones = np.full(n_bars, 'N', dtype='U2')
    valid_rsi = ~np.isnan(rsi)
    rsi_zones[valid_rsi & (rsi < rsi_os)] = 'OS'
    rsi_zones[valid_rsi & (rsi > rsi_ob)] = 'OB'

    # ATR-14% (matching production indicators.py)
    high_low = highs - lows
    high_close = np.abs(highs - np.roll(closes, 1))
    low_close = np.abs(lows - np.roll(closes, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    atr_pct = atr / closes * 100

    # Vol level: rolling 200-bar quantile (matching production signals.py)
    atr_series = pd.Series(atr_pct)
    q_low_arr = atr_series.rolling(200, min_periods=50).quantile(vol_q_low).values
    q_high_arr = atr_series.rolling(200, min_periods=50).quantile(vol_q_high).values

    vol_levels = np.full(n_bars, 'M', dtype='U1')
    valid_vol = ~(np.isnan(q_low_arr) | np.isnan(q_high_arr) | np.isnan(atr_pct))
    vol_levels[valid_vol & (atr_pct < q_low_arr)] = 'L'
    vol_levels[valid_vol & (atr_pct > q_high_arr)] = 'H'

    # Trend: close vs close[lookback bars ago]
    trends = np.full(n_bars, 'N', dtype='U2')
    for i in range(trend_lookback, n_bars):
        trends[i] = 'UP' if closes[i] > closes[i - trend_lookback] else 'DN'

    return rsi_zones, vol_levels, trends, rsi, atr_pct


def bt_fixed_with_context(indices, direction, tp_pct, sl_pct,
                          rsi_zones, vol_levels, trends):
    """Backtest returning list of trade dicts with context tags."""
    results = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        pnl = None
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT
                break
            elif ht:
                pnl = tp_pct * LEVERAGE - FEE_PCT
                break
            elif hs:
                pnl = -sl_pct * LEVERAGE - FEE_PCT
                break
        if pnl is not None:
            results.append({
                'pnl': pnl,
                'win': 1 if pnl > 0 else 0,
                'rsi_zone': rsi_zones[idx],
                'vol': vol_levels[idx],
                'trend': trends[idx],
            })
    return results


def passes_context_filter(ctx, filt):
    """Check if context dict passes a filter specification."""
    for dim, allowed in filt.get('required', {}).items():
        if ctx.get(dim) not in allowed:
            return False
    for dim, blocked in filt.get('excluded', {}).items():
        if ctx.get(dim) in blocked:
            return False
    return True


def collect_trades_1pos_filtered(pattern_map, context_filters,
                                 rsi_zones, vol_levels, trends):
    """Collect 1-position-at-a-time portfolio trades with context filters."""
    raw_trades = []
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue

        # Check context filter
        if pat in context_filters:
            ctx = {'rsi_zone': rsi_zones[i], 'vol': vol_levels[i], 'trend': trends[i]}
            if not passes_context_filter(ctx, context_filters[pat]):
                continue

        entry_bar = i + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp / 100)
            slp = entry * (1 - sl / 100)
        else:
            tpp = entry * (1 - tp / 100)
            slp = entry * (1 + sl / 100)
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                raw_trades.append((entry_bar, j, pnl))
                break
            elif ht:
                raw_trades.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                raw_trades.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for entry_bar, exit_bar, pnl in raw_trades:
        if entry_bar > last_exit:
            filtered.append((entry_bar, exit_bar, pnl))
            last_exit = exit_bar
    return filtered


# ============================================================
# Statistical functions
# ============================================================
def apply_bh_fdr(p_values, alpha=FDR_ALPHA):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    p_arr = np.array(p_values, dtype=float)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]

    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[i] = sorted_p[i] * n / (i + 1)

    # Monotonicity: non-decreasing from right
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    adjusted = np.minimum(adjusted, 1.0)
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result.tolist()


def portfolio_mc_mdd(trades, n_sims=10000):
    """Shuffle trade order → compute MDD for each shuffle."""
    pnl_list = np.array([t[2] for t in trades])
    mdds = []
    for _ in range(n_sims):
        shuffled = np.random.permutation(pnl_list)
        cum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        mdds.append(np.max(dd))
    return np.array(mdds)


def wf_portfolio(trades, n_folds=5):
    """Split trades temporally into n_folds, check each fold profitable."""
    if not trades:
        return 0, []
    sorted_trades = sorted(trades, key=lambda x: x[0])
    fold_size = len(sorted_trades) // n_folds
    if fold_size < 3:
        return 0, []
    results = []
    for f in range(n_folds):
        s = f * fold_size
        e = s + fold_size if f < n_folds - 1 else len(sorted_trades)
        fold_trades = sorted_trades[s:e]
        fold_pnl = sum(t[2] for t in fold_trades)
        fold_wr = sum(1 for t in fold_trades if t[2] > 0) / len(fold_trades) * 100
        results.append({'fold': f + 1, 'trades': len(fold_trades),
                        'pnl': fold_pnl, 'wr': fold_wr})
    profitable_folds = sum(1 for r in results if r['pnl'] > 0)
    return profitable_folds, results


# ============================================================
# Phase 1: Context Feature Tagging
# ============================================================
print("\n" + "=" * 70)
print("Phase 1: Context Feature Tagging")
print("=" * 70)

rsi_zones, vol_levels, trends, rsi_arr, atr_pct_arr = compute_context_arrays()

# Distribution check (excluding first 20 bars with NaN context)
valid_start = max(20, TREND_LB_BASE)  # need both RSI warm-up and trend lookback
valid_mask = np.arange(n_bars) >= valid_start

n_valid = valid_mask.sum()
rsi_dist = {cat: np.sum(rsi_zones[valid_mask] == cat) for cat in DIMENSIONS['rsi_zone']}
vol_dist = {cat: np.sum(vol_levels[valid_mask] == cat) for cat in DIMENSIONS['vol']}
trend_dist = {cat: np.sum(trends[valid_mask] == cat) for cat in DIMENSIONS['trend']}

print(f"\n  Valid bars: {n_valid} (after warm-up period)")
print(f"\n  RSI Zone Distribution:")
for cat, count in rsi_dist.items():
    print(f"    {cat:>3s}: {count:>6d} ({count/n_valid*100:>5.1f}%)")

print(f"\n  Volatility Level Distribution:")
for cat, count in vol_dist.items():
    print(f"    {cat:>3s}: {count:>6d} ({count/n_valid*100:>5.1f}%)")

print(f"\n  Trend Direction Distribution:")
for cat, count in trend_dist.items():
    print(f"    {cat:>3s}: {count:>6d} ({count/n_valid*100:>5.1f}%)")

# Degenerate check
degenerate_dims = []
for dim_name, dist in [('rsi_zone', rsi_dist), ('vol', vol_dist), ('trend', trend_dist)]:
    max_pct = max(dist.values()) / n_valid * 100
    if max_pct > 98:
        degenerate_dims.append(dim_name)
        print(f"\n  WARNING: {dim_name} is degenerate ({max_pct:.1f}% in one category)")

phase1 = {
    'valid_bars': int(n_valid),
    'rsi_dist': {k: int(v) for k, v in rsi_dist.items()},
    'vol_dist': {k: int(v) for k, v in vol_dist.items()},
    'trend_dist': {k: int(v) for k, v in trend_dist.items()},
    'degenerate_dims': degenerate_dims,
}


# ============================================================
# Phase 2: Per-Pattern × Context Cross-Analysis
# ============================================================
print("\n" + "=" * 70)
print("Phase 2: Per-Pattern × Context Cross-Analysis")
print("=" * 70)

# Collect per-pattern trades with context
pattern_context_trades = {}  # pattern -> list of trade dicts
for pat in ALL_PATTERNS:
    direction, tp, sl = PORTFOLIO[pat]
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < 5:
        continue
    trades = bt_fixed_with_context(indices, direction, tp, sl,
                                   rsi_zones, vol_levels, trends)
    pattern_context_trades[pat] = trades

# Cross-analysis: WR by context category
phase2 = {}
notable_effects = []  # (pattern, dimension, best_cat, worst_cat, wr_diff)

print(f"\n  Patterns with WR difference >= {WR_DIFF_THRESHOLD_PP}pp in any dimension:")
print(f"  {'Pattern':<16} {'Dim':<10} {'Cat':>4} {'N':>5} {'WR':>7}  "
      f"{'Cat':>4} {'N':>5} {'WR':>7}  {'Diff':>7}")
print("  " + "-" * 85)

for pat in sorted(pattern_context_trades.keys()):
    trades = pattern_context_trades[pat]
    total_trades = len(trades)
    total_wr = sum(t['win'] for t in trades) / total_trades * 100 if trades else 0

    pat_data = {'total_trades': total_trades, 'total_wr': round(total_wr, 1), 'dims': {}}

    for dim_name, categories in DIMENSIONS.items():
        dim_data = {}
        for cat in categories:
            cat_trades = [t for t in trades if t[dim_name] == cat]
            n_cat = len(cat_trades)
            wr_cat = sum(t['win'] for t in cat_trades) / n_cat * 100 if n_cat > 0 else 0
            dim_data[cat] = {'n': n_cat, 'wr': round(wr_cat, 1),
                             'sufficient': n_cat >= MIN_CELL_TRADES}

        pat_data['dims'][dim_name] = dim_data

        # Find best and worst categories with sufficient data
        sufficient_cats = {c: d for c, d in dim_data.items() if d['sufficient']}
        if len(sufficient_cats) >= 2:
            best_cat = max(sufficient_cats, key=lambda c: sufficient_cats[c]['wr'])
            worst_cat = min(sufficient_cats, key=lambda c: sufficient_cats[c]['wr'])
            wr_diff = sufficient_cats[best_cat]['wr'] - sufficient_cats[worst_cat]['wr']

            if wr_diff >= WR_DIFF_THRESHOLD_PP:
                notable_effects.append((pat, dim_name, best_cat, worst_cat, wr_diff))
                print(f"  {pat:<16} {dim_name:<10} "
                      f"{best_cat:>4} {sufficient_cats[best_cat]['n']:>5d} "
                      f"{sufficient_cats[best_cat]['wr']:>6.1f}%  "
                      f"{worst_cat:>4} {sufficient_cats[worst_cat]['n']:>5d} "
                      f"{sufficient_cats[worst_cat]['wr']:>6.1f}%  "
                      f"{wr_diff:>+6.1f}pp")

    phase2[pat] = pat_data

if not notable_effects:
    print("  (none)")

print(f"\n  Summary: {len(notable_effects)} pattern×dimension pairs with WR diff >= {WR_DIFF_THRESHOLD_PP}pp")
print(f"  Patterns analyzed: {len(pattern_context_trades)}")


# ============================================================
# Phase 3: Statistical Tests + BH FDR
# ============================================================
print("\n" + "=" * 70)
print("Phase 3: Statistical Tests + Benjamini-Hochberg FDR Correction")
print("=" * 70)

# Omnibus test for each pattern × dimension
test_results = []  # list of (pattern, dimension, p_raw, test_type, details)
skipped_count = 0

for pat in sorted(pattern_context_trades.keys()):
    trades = pattern_context_trades[pat]

    for dim_name, categories in DIMENSIONS.items():
        # Build contingency table: wins/losses per category
        table_data = {}
        for cat in categories:
            cat_trades = [t for t in trades if t[dim_name] == cat]
            wins = sum(t['win'] for t in cat_trades)
            losses = len(cat_trades) - wins
            table_data[cat] = (wins, losses)

        # Check if any category has sufficient data
        sufficient_cats = [c for c, (w, l) in table_data.items() if w + l >= MIN_CELL_TRADES]
        if len(sufficient_cats) < 2:
            skipped_count += 1
            test_results.append((pat, dim_name, 1.0, 'skipped', {'reason': 'insufficient'}))
            continue

        # Build contingency table from sufficient categories only
        table = np.array([[table_data[c][0] for c in sufficient_cats],
                          [table_data[c][1] for c in sufficient_cats]])

        if table.shape[1] == 2:
            # 2×2: Fisher exact test
            odds, p = scipy_stats.fisher_exact(table.T)
            test_type = 'fisher_2x2'
            details = {'cats': sufficient_cats, 'table': table.tolist()}
        else:
            # 2×K: chi2 contingency
            try:
                chi2, p, dof, expected = scipy_stats.chi2_contingency(table.T)
                min_exp = expected.min()
                test_type = f'chi2_2x{table.shape[1]}'
                details = {'cats': sufficient_cats, 'chi2': float(chi2),
                           'dof': int(dof), 'min_expected': float(min_exp)}
                if min_exp < 5:
                    details['warning'] = 'low_expected_counts'
            except ValueError:
                p = 1.0
                test_type = 'chi2_error'
                details = {'cats': sufficient_cats}

        test_results.append((pat, dim_name, float(p), test_type, details))

# Apply BH FDR correction
raw_p_values = [r[2] for r in test_results]
adjusted_p_values = apply_bh_fdr(raw_p_values)

# Combine results
phase3 = []
significant_pairs = []  # (pattern, dimension, adj_p, details)

for i, (pat, dim, p_raw, test_type, details) in enumerate(test_results):
    adj_p = adjusted_p_values[i]
    entry = {
        'pattern': pat, 'dimension': dim,
        'p_raw': round(p_raw, 6), 'p_adjusted': round(adj_p, 6),
        'test_type': test_type, 'significant': adj_p < FDR_ALPHA,
    }
    phase3.append(entry)

    if adj_p < FDR_ALPHA and test_type != 'skipped':
        significant_pairs.append((pat, dim, adj_p, details))

n_tested = sum(1 for r in test_results if r[3] != 'skipped')
n_significant = len(significant_pairs)

print(f"\n  Tests performed: {n_tested} (skipped {skipped_count} due to insufficient data)")
print(f"  Total tests for FDR: {len(test_results)}")
print(f"  FDR alpha: {FDR_ALPHA}")
print(f"  Significant after BH correction: {n_significant}")

if significant_pairs:
    print(f"\n  Significant pattern × dimension pairs:")
    print(f"  {'Pattern':<16} {'Dimension':<10} {'p_adj':>8}")
    print("  " + "-" * 40)
    for pat, dim, adj_p, details in significant_pairs:
        print(f"  {pat:<16} {dim:<10} {adj_p:>8.4f}")
else:
    print("\n  No significant effects found after FDR correction.")


# ============================================================
# Phase 4: Filter Candidate Selection
# ============================================================
print("\n" + "=" * 70)
print("Phase 4: Filter Candidate Selection")
print("=" * 70)

filter_candidates = {}  # pattern -> filter spec

if not significant_pairs:
    print("\n  No significant pairs from Phase 3 → no filter candidates.")
    print("  Checking notable effects (Phase 2) as fallback...")

    # Even without statistical significance, check if any large effects exist
    # that might be worth investigating at portfolio level
    if notable_effects:
        print(f"  Found {len(notable_effects)} notable effects (WR diff >= {WR_DIFF_THRESHOLD_PP}pp)")
        print("  These are descriptive only — not statistically significant.")
    else:
        print("  No notable effects either. Context filters provide no value.")

# Process significant pairs into filter candidates
for pat, dim, adj_p, details in significant_pairs:
    trades = pattern_context_trades[pat]
    total_trades = len(trades)

    # Get WR by category for this dimension
    cat_stats = {}
    for cat in DIMENSIONS[dim]:
        cat_trades = [t for t in trades if t[dim] == cat]
        if len(cat_trades) >= MIN_CELL_TRADES:
            wr = sum(t['win'] for t in cat_trades) / len(cat_trades) * 100
            cat_stats[cat] = {'n': len(cat_trades), 'wr': wr}

    if len(cat_stats) < 2:
        continue

    best_cat = max(cat_stats, key=lambda c: cat_stats[c]['wr'])
    worst_cat = min(cat_stats, key=lambda c: cat_stats[c]['wr'])
    wr_diff = cat_stats[best_cat]['wr'] - cat_stats[worst_cat]['wr']

    if wr_diff < WR_DIFF_THRESHOLD_PP:
        continue

    # Evaluate "excluded" filter: block worst category
    excluded_trades = sum(1 for t in trades if t[dim] == worst_cat)
    keep_trades = total_trades - excluded_trades
    keep_ratio = keep_trades / total_trades

    # Evaluate "required" filter: allow only best category
    required_trades = sum(1 for t in trades if t[dim] == best_cat)
    required_ratio = required_trades / total_trades

    # Choose best filter type
    candidate = None

    if keep_ratio >= MIN_TRADE_RETENTION and keep_trades >= MIN_CELL_TRADES:
        # "excluded" filter: remove worst category
        keep_wr = sum(t['win'] for t in trades if t[dim] != worst_cat) / keep_trades * 100
        candidate = {
            'type': 'excluded',
            'dimension': dim,
            'excluded_category': worst_cat,
            'wr_before': sum(t['win'] for t in trades) / total_trades * 100,
            'wr_after': keep_wr,
            'wr_improvement': keep_wr - sum(t['win'] for t in trades) / total_trades * 100,
            'trades_before': total_trades,
            'trades_after': keep_trades,
            'trade_retention': keep_ratio,
            'adj_p': adj_p,
            'filter_spec': {'excluded': {dim: [worst_cat]}},
        }

    if required_ratio >= MIN_TRADE_RETENTION and required_trades >= MIN_CELL_TRADES:
        req_wr = cat_stats[best_cat]['wr']
        orig_wr = sum(t['win'] for t in trades) / total_trades * 100
        req_improvement = req_wr - orig_wr

        if candidate is None or req_improvement > candidate['wr_improvement']:
            candidate = {
                'type': 'required',
                'dimension': dim,
                'required_category': best_cat,
                'wr_before': orig_wr,
                'wr_after': req_wr,
                'wr_improvement': req_improvement,
                'trades_before': total_trades,
                'trades_after': required_trades,
                'trade_retention': required_ratio,
                'adj_p': adj_p,
                'filter_spec': {'required': {dim: [best_cat]}},
            }

    if candidate:
        # Avoid duplicate patterns (keep best improvement)
        if pat not in filter_candidates or candidate['wr_improvement'] > filter_candidates[pat]['wr_improvement']:
            filter_candidates[pat] = candidate

n_candidates = len(filter_candidates)
print(f"\n  Filter candidates: {n_candidates}")

phase4 = {'n_candidates': n_candidates, 'candidates': {}}
if filter_candidates:
    print(f"\n  {'Pattern':<16} {'Type':<10} {'Dim':<10} {'Cat':<5} "
          f"{'WR_before':>9} {'WR_after':>9} {'Improve':>8} {'Retain':>7} {'p_adj':>7}")
    print("  " + "-" * 95)
    for pat, cand in sorted(filter_candidates.items()):
        cat_label = cand.get('excluded_category', cand.get('required_category', '?'))
        print(f"  {pat:<16} {cand['type']:<10} {cand['dimension']:<10} {cat_label:<5} "
              f"{cand['wr_before']:>8.1f}% {cand['wr_after']:>8.1f}% "
              f"{cand['wr_improvement']:>+7.1f}pp {cand['trade_retention']:>6.0%} "
              f"{cand['adj_p']:>7.4f}")
        phase4['candidates'][pat] = {k: v for k, v in cand.items() if k != 'filter_spec'}
else:
    print("  No candidates passed all criteria.")
    print("  Criteria: BH p < 0.10, WR diff >= 15pp, retention >= 50%, keep/remove >= 10 trades")


# ============================================================
# Phase 5: Portfolio Simulation (Filtered vs Unfiltered)
# ============================================================
print("\n" + "=" * 70)
print("Phase 5: Portfolio Simulation — Filtered vs Unfiltered")
print("=" * 70)

# Baseline: unfiltered portfolio
baseline_trades = collect_trades_1pos(PORTFOLIO)
baseline_stats = eval_trades_simple(baseline_trades)

print(f"\n  Baseline (v1.27.0 unfiltered):")
print(f"    PnL: {baseline_stats['pnl']:+.1f}%, Trades: {baseline_stats['trades']}, "
      f"WR: {baseline_stats['wr']:.1f}%, MDD: {baseline_stats['mdd']:.1f}%, "
      f"PF: {baseline_stats['pf']:.2f}, PnL/MDD: {baseline_stats['pnl_mdd']:.1f}x")

phase5 = {
    'baseline': {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in baseline_stats.items()},
}

if not filter_candidates:
    print("\n  No filter candidates → skipping filtered simulation.")
    phase5['filtered'] = None
    phase5['comparison'] = 'skipped'
else:
    # Build context filter map for portfolio simulation
    ctx_filters = {}
    for pat, cand in filter_candidates.items():
        ctx_filters[pat] = cand['filter_spec']

    filtered_trades = collect_trades_1pos_filtered(
        PORTFOLIO, ctx_filters, rsi_zones, vol_levels, trends)
    filtered_stats = eval_trades_simple(filtered_trades)

    print(f"\n  Filtered portfolio:")
    print(f"    PnL: {filtered_stats['pnl']:+.1f}%, Trades: {filtered_stats['trades']}, "
          f"WR: {filtered_stats['wr']:.1f}%, MDD: {filtered_stats['mdd']:.1f}%, "
          f"PF: {filtered_stats['pf']:.2f}, PnL/MDD: {filtered_stats['pnl_mdd']:.1f}x")

    # Comparison
    pnl_change = filtered_stats['pnl'] - baseline_stats['pnl']
    mdd_change = filtered_stats['mdd'] - baseline_stats['mdd']
    pnl_mdd_change = filtered_stats['pnl_mdd'] - baseline_stats['pnl_mdd']

    print(f"\n  Change vs baseline:")
    print(f"    PnL:     {pnl_change:+.1f}%")
    print(f"    MDD:     {mdd_change:+.1f}pp")
    print(f"    PnL/MDD: {pnl_mdd_change:+.1f}x")

    # Minimum improvement criteria
    passes_pnl_mdd = filtered_stats['pnl_mdd'] >= baseline_stats['pnl_mdd']
    passes_pnl = filtered_stats['pnl'] >= baseline_stats['pnl'] * 0.95
    passes_mdd = filtered_stats['mdd'] <= baseline_stats['mdd'] * 1.10

    print(f"\n  Minimum criteria:")
    print(f"    PnL/MDD >= baseline: {'PASS' if passes_pnl_mdd else 'FAIL'} "
          f"({filtered_stats['pnl_mdd']:.1f}x vs {baseline_stats['pnl_mdd']:.1f}x)")
    print(f"    PnL >= 95% baseline: {'PASS' if passes_pnl else 'FAIL'} "
          f"({filtered_stats['pnl']:+.1f}% vs {baseline_stats['pnl']*0.95:+.1f}%)")
    print(f"    MDD <= 110% baseline: {'PASS' if passes_mdd else 'FAIL'} "
          f"({filtered_stats['mdd']:.1f}% vs {baseline_stats['mdd']*1.10:.1f}%)")

    phase5_pass = passes_pnl_mdd and passes_pnl and passes_mdd
    print(f"\n  Phase 5 result: {'PASS' if phase5_pass else 'FAIL'}")

    phase5['filtered'] = {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in filtered_stats.items()}
    phase5['comparison'] = {
        'pnl_change': round(pnl_change, 2),
        'mdd_change': round(mdd_change, 2),
        'pnl_mdd_change': round(pnl_mdd_change, 2),
        'passes_pnl_mdd': passes_pnl_mdd,
        'passes_pnl': passes_pnl,
        'passes_mdd': passes_mdd,
        'phase5_pass': phase5_pass,
    }


# ============================================================
# Phase 6: MC + WF Validation
# ============================================================
print("\n" + "=" * 70)
print("Phase 6: MC + WF Validation")
print("=" * 70)

phase6 = {}

if not filter_candidates or not phase5.get('comparison', {}).get('phase5_pass', False):
    if not filter_candidates:
        print("\n  No filter candidates → skipping validation.")
    else:
        print("\n  Phase 5 FAILED → skipping validation.")
    phase6 = {'skipped': True, 'reason': 'no_candidates_or_phase5_fail'}
else:
    t6 = time.time()

    # 6a: Portfolio MC sign randomization
    pnl_arr = np.array([t[2] for t in filtered_trades])
    real_sum = np.sum(pnl_arr)
    signs = np.random.choice([-1, 1], size=(MC_SIMS, len(pnl_arr)))
    mc_p = float(np.mean(np.sum(pnl_arr * signs, axis=1) >= real_sum))

    print(f"\n  6a. Portfolio MC (sign randomization):")
    print(f"      PnL: {real_sum:+.1f}%, MC p-value: {mc_p:.4f} → "
          f"{'PASS' if mc_p < 0.01 else 'FAIL'}")

    # 6b: Portfolio WF 5-fold temporal
    wf_count, wf_detail = wf_portfolio(filtered_trades)
    print(f"\n  6b. Walk-Forward (5-fold temporal):")
    print(f"      Profitable folds: {wf_count}/5 → "
          f"{'PASS' if wf_count >= 4 else 'FAIL'}")
    for r in wf_detail:
        print(f"      Fold {r['fold']}: {r['trades']:>3d} trades, "
              f"PnL {r['pnl']:>+8.1f}%, WR {r['wr']:>5.1f}%")

    # 6c: MC MDD order shuffle
    mdds = portfolio_mc_mdd(filtered_trades)
    p99_mdd = float(np.percentile(mdds, 99))
    print(f"\n  6c. MC MDD (order shuffle):")
    print(f"      P50: {np.percentile(mdds, 50):.1f}%, P90: {np.percentile(mdds, 90):.1f}%, "
          f"P99: {p99_mdd:.1f}% → {'PASS' if p99_mdd <= 50 else 'FAIL'}")

    # 6d: Per-filter edge preservation
    print(f"\n  6d. Edge preservation (per filtered pattern):")
    edge_failures = []
    for pat, cand in filter_candidates.items():
        direction, tp, sl = PORTFOLIO[pat]
        trades_all = pattern_context_trades[pat]

        # Filtered trades for this pattern
        filt = cand['filter_spec']
        filtered_pat_trades = [t for t in trades_all
                               if passes_context_filter(
                                   {'rsi_zone': t['rsi_zone'], 'vol': t['vol'], 'trend': t['trend']},
                                   filt)]

        if len(filtered_pat_trades) < 5:
            edge_failures.append((pat, 'insufficient_trades'))
            continue

        actual_wr = sum(t['win'] for t in filtered_pat_trades) / len(filtered_pat_trades) * 100
        rw_wr = sl / (tp + sl) * 100
        edge = actual_wr - rw_wr

        status = 'PASS' if edge >= 5 else 'FAIL'
        print(f"      {pat:<16} actual_WR={actual_wr:.1f}%, RW_WR={rw_wr:.1f}%, "
              f"edge={edge:+.1f}pp → {status}")

        if edge < 5:
            edge_failures.append((pat, f'edge={edge:.1f}pp'))

    edge_pass = len(edge_failures) == 0

    # Overall Phase 6
    phase6_pass = mc_p < 0.01 and wf_count >= 4 and p99_mdd <= 50 and edge_pass
    elapsed6 = time.time() - t6

    print(f"\n  Phase 6 result: {'PASS' if phase6_pass else 'FAIL'} ({elapsed6:.1f}s)")
    if not phase6_pass:
        issues = []
        if mc_p >= 0.01:
            issues.append(f"MC p={mc_p:.4f}")
        if wf_count < 4:
            issues.append(f"WF {wf_count}/5")
        if p99_mdd > 50:
            issues.append(f"MDD P99={p99_mdd:.1f}%")
        if not edge_pass:
            issues.append(f"edge failures: {edge_failures}")
        print(f"    Issues: {', '.join(issues)}")

    phase6 = {
        'mc_p': mc_p,
        'wf_count': wf_count,
        'wf_detail': wf_detail,
        'mdd_p99': p99_mdd,
        'mdd_p50': float(np.percentile(mdds, 50)),
        'mdd_p90': float(np.percentile(mdds, 90)),
        'edge_failures': edge_failures,
        'phase6_pass': phase6_pass,
    }


# ============================================================
# Phase 7: Sensitivity Analysis
# ============================================================
print("\n" + "=" * 70)
print("Phase 7: Sensitivity Analysis")
print("=" * 70)

phase7 = {}

if not filter_candidates:
    print("\n  No filter candidates → skipping sensitivity analysis.")
    phase7 = {'skipped': True}
else:
    # Determine which dimensions are used by filter candidates
    used_dims = set()
    for pat, cand in filter_candidates.items():
        for dim_type in ['required', 'excluded']:
            for dim in cand['filter_spec'].get(dim_type, {}):
                used_dims.add(dim)

    print(f"\n  Filter dimensions used: {sorted(used_dims)}")

    ctx_filters = {pat: cand['filter_spec'] for pat, cand in filter_candidates.items()}
    sensitivity_results = {}

    # RSI threshold sensitivity
    if 'rsi_zone' in used_dims:
        print("\n  RSI threshold sensitivity:")
        rsi_variants = [(25, 75), (30, 70), (35, 65)]
        for os_th, ob_th in rsi_variants:
            rz, vl, tr, _, _ = compute_context_arrays(rsi_os=os_th, rsi_ob=ob_th)
            trades = collect_trades_1pos_filtered(PORTFOLIO, ctx_filters, rz, vl, tr)
            stats = eval_trades_simple(trades)
            key = f'rsi_{os_th}_{ob_th}'
            sensitivity_results[key] = stats
            marker = " [base]" if os_th == RSI_OS_BASE else ""
            print(f"    RSI ({os_th}/{ob_th}): PnL {stats['pnl']:+.1f}%, "
                  f"MDD {stats['mdd']:.1f}%, PnL/MDD {stats['pnl_mdd']:.1f}x, "
                  f"Trades {stats['trades']}{marker}")

    # Vol quantile sensitivity
    if 'vol' in used_dims:
        print("\n  Vol quantile sensitivity:")
        vol_variants = [(0.25, 0.75), (0.33, 0.66), (0.40, 0.60)]
        for q_low, q_high in vol_variants:
            rz, vl, tr, _, _ = compute_context_arrays(vol_q_low=q_low, vol_q_high=q_high)
            trades = collect_trades_1pos_filtered(PORTFOLIO, ctx_filters, rz, vl, tr)
            stats = eval_trades_simple(trades)
            key = f'vol_{q_low}_{q_high}'
            sensitivity_results[key] = stats
            marker = " [base]" if q_low == VOL_Q_LOW_BASE else ""
            print(f"    Vol ({q_low:.2f}/{q_high:.2f}): PnL {stats['pnl']:+.1f}%, "
                  f"MDD {stats['mdd']:.1f}%, PnL/MDD {stats['pnl_mdd']:.1f}x, "
                  f"Trades {stats['trades']}{marker}")

    # Trend lookback sensitivity
    if 'trend' in used_dims:
        print("\n  Trend lookback sensitivity:")
        lb_variants = [10, 15, 20, 25, 30]
        for lb in lb_variants:
            rz, vl, tr, _, _ = compute_context_arrays(trend_lookback=lb)
            trades = collect_trades_1pos_filtered(PORTFOLIO, ctx_filters, rz, vl, tr)
            stats = eval_trades_simple(trades)
            key = f'trend_lb_{lb}'
            sensitivity_results[key] = stats
            marker = " [base]" if lb == TREND_LB_BASE else ""
            print(f"    Lookback {lb}: PnL {stats['pnl']:+.1f}%, "
                  f"MDD {stats['mdd']:.1f}%, PnL/MDD {stats['pnl_mdd']:.1f}x, "
                  f"Trades {stats['trades']}{marker}")

    # Leave-one-pattern-out stability
    if len(filter_candidates) > 1:
        print("\n  Leave-one-pattern-out stability:")
        for remove_pat in sorted(filter_candidates.keys()):
            reduced_filters = {p: f for p, f in ctx_filters.items() if p != remove_pat}
            trades = collect_trades_1pos_filtered(PORTFOLIO, reduced_filters,
                                                  rsi_zones, vol_levels, trends)
            stats = eval_trades_simple(trades)
            key = f'loo_{remove_pat}'
            sensitivity_results[key] = stats
            print(f"    Remove {remove_pat}: PnL {stats['pnl']:+.1f}%, "
                  f"PnL/MDD {stats['pnl_mdd']:.1f}x")

    # Robustness check: >= 2/3 of parameter variations improve PnL/MDD vs baseline
    param_variants = {k: v for k, v in sensitivity_results.items()
                      if not k.startswith('loo_')}
    if param_variants:
        improved = sum(1 for s in param_variants.values()
                       if s['pnl_mdd'] > baseline_stats['pnl_mdd'])
        total_v = len(param_variants)
        robust = improved >= total_v * 2 / 3
        print(f"\n  Robustness: {improved}/{total_v} variants improve PnL/MDD "
              f"→ {'ROBUST' if robust else 'NOT ROBUST'} (need >= {total_v*2//3 + 1})")
    else:
        robust = False

    phase7 = {
        'used_dims': sorted(used_dims),
        'sensitivity': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                            for kk, vv in v.items()}
                        for k, v in sensitivity_results.items()},
        'robust': robust,
    }


# ============================================================
# Phase 8: Summary & Recommendation
# ============================================================
print("\n" + "=" * 70)
print("Phase 8: Summary & Recommendation")
print("=" * 70)

elapsed_total = time.time() - t_start

# Determine overall result
if not filter_candidates:
    verdict = 'FAIL'
    reason = 'No statistically significant context effects found'
    detail = (f"52 patterns × 3 dimensions tested. "
              f"{n_significant} significant after BH FDR correction (alpha={FDR_ALPHA}). "
              f"{len(notable_effects)} had WR diff >= {WR_DIFF_THRESHOLD_PP}pp but not significant.")
elif not phase5.get('comparison', {}).get('phase5_pass', False):
    verdict = 'FAIL'
    reason = 'Filter candidates do not improve portfolio'
    detail = f"{n_candidates} candidates found but portfolio metrics worsen."
elif not phase6.get('phase6_pass', False):
    verdict = 'CONDITIONAL'
    reason = 'Filter candidates improve portfolio but fail validation'
    detail = f"MC/WF/edge validation issues."
elif not phase7.get('robust', False):
    verdict = 'CONDITIONAL'
    reason = 'Filter effect is parameter-sensitive'
    detail = f"Not robust across parameter variations."
else:
    verdict = 'PASS'
    reason = 'Filter candidates pass all validation'
    detail = f"{n_candidates} filters improve portfolio and are robust."

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Context Filter Research v2 — Final Verdict                    │
  │                                                                 │
  │  Verdict:  {verdict:<54s}│
  │  Reason:   {reason:<54s}│
  │                                                                 │
  │  Patterns tested:     {len(pattern_context_trades):>4d}                                   │
  │  Dimensions:          {len(DIMENSIONS):>4d} (RSI zone, Vol level, Trend)        │
  │  Total omnibus tests: {len(test_results):>4d}                                   │
  │  Significant (BH):   {n_significant:>4d}                                   │
  │  Filter candidates:   {n_candidates:>4d}                                   │
  │                                                                 │
  │  Baseline portfolio:                                            │
  │    PnL {baseline_stats['pnl']:>+8.1f}%, WR {baseline_stats['wr']:>5.1f}%, MDD {baseline_stats['mdd']:>5.1f}%              │
  │    PF {baseline_stats['pf']:>5.2f}, PnL/MDD {baseline_stats['pnl_mdd']:>5.1f}x                            │
  │                                                                 │
  │  Elapsed: {elapsed_total:>5.1f}s                                            │
  └─────────────────────────────────────────────────────────────────┘
""")

if filter_candidates and phase5.get('filtered'):
    fs = phase5['filtered']
    print(f"  Filtered portfolio (if applied):")
    print(f"    PnL {fs['pnl']:+.1f}%, WR {fs['wr']:.1f}%, MDD {fs['mdd']:.1f}%, "
          f"PF {fs['pf']:.2f}, PnL/MDD {fs['pnl_mdd']:.1f}x")

print(f"\n  {detail}")

if verdict == 'FAIL':
    print(f"\n  RECOMMENDATION: Context filters 불필요 — v1.27.0 그대로 유지")
    print(f"    - 52패턴에서 context별 WR 차이가 통계적으로 유의하지 않음")
    print(f"    - 이미 WR {baseline_stats['wr']:.1f}%로 개선 여지 제한적")
    print(f"    - PATTERN_CONTEXT_FILTERS = {{}} 유지")
elif verdict == 'CONDITIONAL':
    print(f"\n  RECOMMENDATION: 조건부 — 추가 데이터 축적 후 재검증 필요")
    print(f"    - 효과가 있으나 통계적 확신 부족")
    print(f"    - 향후 실거래 데이터로 재검증 권장")
else:
    print(f"\n  RECOMMENDATION: 적용 가능")
    print(f"    - constants.py PATTERN_CONTEXT_FILTERS 업데이트")
    for pat, cand in sorted(filter_candidates.items()):
        print(f"    - {pat}: {cand['filter_spec']}")

phase8 = {
    'verdict': verdict,
    'reason': reason,
    'detail': detail,
    'elapsed_seconds': round(elapsed_total, 1),
}


# ============================================================
# Save results
# ============================================================
results = {
    'meta': {
        'script': 'context_filter_research_v2.py',
        'version': BOT_VERSION,
        'timestamp': datetime.now().isoformat(),
        'patterns': len(PORTFOLIO),
        'elapsed_seconds': round(elapsed_total, 1),
    },
    'phase1_context_distribution': phase1,
    'phase2_cross_analysis': {
        'patterns_analyzed': len(pattern_context_trades),
        'notable_effects': [
            {'pattern': p, 'dimension': d, 'best_cat': bc,
             'worst_cat': wc, 'wr_diff_pp': round(wd, 1)}
            for p, d, bc, wc, wd in notable_effects
        ],
    },
    'phase3_statistical_tests': {
        'n_tested': n_tested,
        'n_skipped': skipped_count,
        'fdr_alpha': FDR_ALPHA,
        'n_significant': n_significant,
        'significant_pairs': [
            {'pattern': p, 'dimension': d, 'p_adjusted': round(ap, 6)}
            for p, d, ap, _ in significant_pairs
        ],
    },
    'phase4_filter_candidates': phase4,
    'phase5_portfolio_simulation': phase5,
    'phase6_validation': phase6,
    'phase7_sensitivity': phase7,
    'phase8_summary': phase8,
}

out_path = Path(__file__).resolve().parent / '../../results/context_filter_research_v2.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved: {out_path}")
