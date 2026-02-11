"""
Overfitting & Forward Decay Research
======================================
패턴 포트폴리오의 과적합 위험과 미래 적용성에 대한 심층 연구.

핵심 질문:
1. 패턴 집합이 버전마다 변했다 → 어느 시점의 패턴이든 "그때의 데이터에 맞춘" 것 아닌가?
2. 270일 과거 데이터로 추정한 통계가 미래에도 유효한가?

Phase 1: Temporal Stability (패턴 edge의 시간적 안정성)
  - 270일을 9개 30일 구간으로 분할
  - 각 구간별 52패턴의 WR/edge 계산
  - 패턴 edge가 시간에 따라 감쇠하는지 (forward decay)?
  - 구간별 "좋은 패턴" 집합의 변동 (Jaccard similarity)

Phase 2: True Out-of-Time Test (순수 미래 예측력)
  - 첫 135일(50%)로만 패턴 선택 + TP/SL 최적화
  - 나머지 135일에서 성과 측정 (한 번도 본 적 없는 데이터)
  - vs 전체 270일로 최적화 후 같은 후반 135일 성과 (look-ahead)
  - Gap = 과적합의 크기

Phase 3: Regime Dependence (시장 국면 의존성)
  - 270일을 변동성(ATR) 기준 상/중/하로 분류
  - 추세(30-bar MA slope) 기준 상승/횡보/하락으로 분류
  - 각 국면별 전략 성과 → 특정 국면에만 작동하면 미래 위험

Phase 4: Edge Decay Curve (edge 수명 추정)
  - 최초 N일로 패턴 선택 → 이후 30일 OOS 성과 측정
  - N = 30, 60, 90, 120, 150, 180, 210, 240
  - 훈련 기간이 길어질수록 OOS edge가 안정화되는지?
  - "전략 수명" 추정: edge가 0에 수렴하는 시점

Phase 5: Pattern Survival Analysis (버전간 패턴 생존)
  - v1.25.4 (23), v1.26.1 (58), v1.26.2 (52), v1.27.0 (52) 패턴 집합
  - 각 버전의 패턴이 다른 기간에서도 유효했는지
  - "항상 좋은" 패턴 vs "한때만 좋았던" 패턴 구분

Phase 6: Bootstrap Confidence Interval
  - 270일 데이터에서 포트폴리오 성과의 부트스트랩 신뢰구간
  - PnL, WR, MDD의 95% CI
  - CI가 0을 포함하면 → edge가 통계적으로 불확실

Standard Research Protocol: next-bar open entry, distance-based exit,
  0.10% fee, 3x leverage, 10k MC sims
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
TP_SCALE = 0.70
N_PERIODS = 5

np.random.seed(42)

print("=" * 70)
print("Overfitting & Forward Decay Research")
print("=" * 70)

# ============================================================
# Data loading & classification
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)
print(f"Data: {n_bars} bars")

print("Classifying candles...")
body_abs = np.abs(closes - opens)
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values
types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20[i] if not pd.isna(avg_body_20[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

# Build 3-candle patterns
patterns_at = [None] * n_bars
for i in range(2, n_bars):
    patterns_at[i] = f"{types[i-2]}-{types[i-1]}-{types[i]}"

# ATR(14)
print("Computing ATR(14)...")
tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)),
                                          np.abs(lows - np.roll(closes, 1))))
tr[0] = highs[0] - lows[0]
atr_14 = pd.Series(tr).rolling(14).mean().values
atr_pct = atr_14 / closes * 100

# 30-bar MA for trend
ma_30 = pd.Series(closes).rolling(30).mean().values

# Portfolio
all_patterns = {}
for p in VALIDATED_LONG_PATTERNS:
    all_patterns[p] = 'LONG'
for p in VALIDATED_SHORT_PATTERNS:
    all_patterns[p] = 'SHORT'
print(f"Portfolio: {len(all_patterns)} patterns")

# ============================================================
# Helper functions
# ============================================================
def simulate_trade(idx, direction, tp_pct, sl_pct, max_bars=MAX_BARS):
    """Simulate a single trade from bar idx+1 open."""
    if idx + 1 >= n_bars:
        return None
    entry = opens[idx + 1]
    if entry <= 0:
        return None

    for j in range(idx + 1, min(idx + 1 + max_bars, n_bars)):
        if direction == 'LONG':
            tp_hit = (highs[j] - entry) / entry * 100 >= tp_pct
            sl_hit = (entry - lows[j]) / entry * 100 >= sl_pct
        else:
            tp_hit = (entry - lows[j]) / entry * 100 >= tp_pct
            sl_hit = (highs[j] - entry) / entry * 100 >= sl_pct

        if tp_hit and sl_hit:
            # Assume SL hit first if both in same bar
            win = False
        elif tp_hit:
            win = True
        elif sl_hit:
            win = False
        else:
            continue

        pnl = (tp_pct if win else -sl_pct) * LEVERAGE - FEE_PCT
        return {'bar': idx, 'entry_bar': idx + 1, 'exit_bar': j,
                'direction': direction, 'win': win, 'pnl': pnl,
                'hold_bars': j - idx - 1}

    # Max bars reached - market close
    exit_price = closes[min(idx + max_bars, n_bars - 1)]
    if direction == 'LONG':
        price_pnl = (exit_price - entry) / entry * 100
    else:
        price_pnl = (entry - exit_price) / entry * 100
    pnl = price_pnl * LEVERAGE - FEE_PCT
    return {'bar': idx, 'entry_bar': idx + 1,
            'exit_bar': min(idx + max_bars, n_bars - 1),
            'direction': direction, 'win': pnl > 0, 'pnl': pnl,
            'hold_bars': MAX_BARS}


def collect_all_signals(start_bar=0, end_bar=None, pattern_set=None):
    """Collect all signals in range."""
    if end_bar is None:
        end_bar = n_bars
    if pattern_set is None:
        pattern_set = all_patterns
    signals = []
    for i in range(max(start_bar, 2), end_bar):
        p = patterns_at[i]
        if p and p in pattern_set:
            direction = pattern_set[p]
            tp_raw, sl = PATTERN_OPTIMAL_TPSL.get(p, (1.0, 2.0))
            tp = tp_raw * TP_SCALE
            signals.append((i, p, direction, tp, sl))
    return signals


def run_1pos(signals):
    """Run 1-position-at-a-time simulation."""
    trades = []
    pos_end = -1
    for (idx, pat, d, tp, sl) in signals:
        if idx <= pos_end:
            continue
        t = simulate_trade(idx, d, tp, sl)
        if t:
            t['pattern'] = pat
            trades.append(t)
            pos_end = t['exit_bar']
    return trades


def eval_trades(trades):
    """Evaluate trade list."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    mdd = float(np.max(dd)) if len(dd) > 0 else 0
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    return {
        'pnl': sum(pnls),
        'trades': len(trades),
        'wr': sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0,
        'mdd': mdd,
        'pf': gross_win / gross_loss,
        'pnl_mdd': sum(pnls) / mdd if mdd > 0 else 999,
    }


def mc_test(trades, n_sims=MC_SIMS):
    """Monte Carlo sign randomization test."""
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t['pnl'] for t in trades])
    real_pnl = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    count = 0
    for _ in range(n_sims):
        signs = np.random.choice([-1, 1], size=len(pnls))
        if np.sum(signs * abs_pnls) >= real_pnl:
            count += 1
    return count / n_sims


results = {}

# ============================================================
# Phase 1: Temporal Stability
# ============================================================
print("\n" + "=" * 70)
print("Phase 1: Temporal Stability (30-day windows)")
print("=" * 70)

bars_per_day = 288  # 5min bars per day
window_days = 30
window_bars = window_days * bars_per_day
n_windows = n_bars // window_bars

print(f"  {n_windows} windows of {window_days} days ({window_bars} bars each)")

window_results = []
pattern_window_wr = defaultdict(list)  # pattern -> [wr per window]
pattern_window_edge = defaultdict(list)

for w in range(n_windows):
    start = w * window_bars
    end = (w + 1) * window_bars
    sigs = collect_all_signals(start, end)
    trades = run_1pos(sigs)
    ev = eval_trades(trades)
    window_results.append({
        'window': w + 1,
        'days': f"{w * window_days}-{(w + 1) * window_days}",
        **ev
    })

    # Per-pattern stats in this window
    pat_trades = defaultdict(list)
    for t in trades:
        pat_trades[t['pattern']].append(t)
    for p in all_patterns:
        pt = pat_trades.get(p, [])
        if len(pt) >= 2:
            wr = sum(1 for t in pt if t['pnl'] > 0) / len(pt) * 100
            # distance-based expected WR
            tp_raw, sl = PATTERN_OPTIMAL_TPSL.get(p, (1.0, 2.0))
            tp = tp_raw * TP_SCALE
            dist_wr = sl / (tp + sl) * 100
            edge = wr - dist_wr
            pattern_window_wr[p].append(wr)
            pattern_window_edge[p].append(edge)
        else:
            pattern_window_wr[p].append(None)
            pattern_window_edge[p].append(None)

print(f"\n  Portfolio performance by 30-day window:")
print(f"  {'Window':>8}  {'Days':>10}  {'PnL':>8}  {'Trades':>7}  {'WR':>6}  {'MDD':>7}  {'PnL/MDD':>8}")
print(f"  {'-'*60}")
for wr in window_results:
    pnl_mdd_str = f"{wr['pnl_mdd']:.1f}x" if wr['mdd'] > 0 else "N/A"
    print(f"  {wr['window']:>8}  {wr['days']:>10}  {wr['pnl']:>+7.1f}%  {wr['trades']:>7}  {wr['wr']:>5.1f}%  {wr['mdd']:>6.1f}%  {pnl_mdd_str:>8}")

# Trend analysis: is there decay?
pnls_by_window = [wr['pnl'] for wr in window_results]
first_half = pnls_by_window[:n_windows // 2]
second_half = pnls_by_window[n_windows // 2:]
print(f"\n  First half avg PnL: {np.mean(first_half):+.1f}%")
print(f"  Second half avg PnL: {np.mean(second_half):+.1f}%")
print(f"  Decay signal: {'YES' if np.mean(second_half) < np.mean(first_half) * 0.5 else 'NO'}")

# Jaccard similarity of "good patterns" across windows
print(f"\n  Pattern set stability (Jaccard between consecutive windows):")
good_sets = []
for w in range(n_windows):
    good = set()
    for p in all_patterns:
        if len(pattern_window_edge[p]) > w and pattern_window_edge[p][w] is not None:
            if pattern_window_edge[p][w] > 0:
                good.add(p)
    good_sets.append(good)

jaccards = []
for w in range(1, len(good_sets)):
    if len(good_sets[w] | good_sets[w-1]) > 0:
        j = len(good_sets[w] & good_sets[w-1]) / len(good_sets[w] | good_sets[w-1])
    else:
        j = 0
    jaccards.append(j)
    print(f"    Window {w}→{w+1}: Jaccard={j:.3f} (overlap {len(good_sets[w] & good_sets[w-1])}/{len(good_sets[w] | good_sets[w-1])})")

avg_jaccard = np.mean(jaccards) if jaccards else 0
print(f"  Average Jaccard: {avg_jaccard:.3f}")
print(f"  Interpretation: {'HIGH stability' if avg_jaccard > 0.5 else 'MODERATE stability' if avg_jaccard > 0.3 else 'LOW stability (overfitting risk)'}")

# Count always-positive patterns
always_positive = 0
mostly_positive = 0
for p in all_patterns:
    edges = [e for e in pattern_window_edge[p] if e is not None]
    if len(edges) >= 3:
        if all(e > 0 for e in edges):
            always_positive += 1
        if sum(1 for e in edges if e > 0) / len(edges) >= 0.7:
            mostly_positive += 1

print(f"\n  Always positive edge (all windows): {always_positive}/{len(all_patterns)}")
print(f"  Mostly positive (>=70% windows): {mostly_positive}/{len(all_patterns)}")

results['phase1_temporal_stability'] = {
    'n_windows': n_windows,
    'window_days': window_days,
    'window_results': window_results,
    'first_half_avg_pnl': float(np.mean(first_half)),
    'second_half_avg_pnl': float(np.mean(second_half)),
    'avg_jaccard': float(avg_jaccard),
    'always_positive': always_positive,
    'mostly_positive': mostly_positive,
    'jaccards': [float(j) for j in jaccards],
}


# ============================================================
# Phase 2: True Out-of-Time Test
# ============================================================
print("\n" + "=" * 70)
print("Phase 2: True Out-of-Time Test")
print("=" * 70)

split_bar = n_bars // 2
print(f"  Train: bars 0-{split_bar} ({split_bar // bars_per_day} days)")
print(f"  Test:  bars {split_bar}-{n_bars} ({(n_bars - split_bar) // bars_per_day} days)")

# Method A: Select patterns using ONLY first half, test on second half
# Step 1: Find which patterns have edge in first half
print(f"\n  2a. Pattern selection from first half only:")
train_pattern_stats = {}
for p, d in all_patterns.items():
    tp_raw, sl = PATTERN_OPTIMAL_TPSL.get(p, (1.0, 2.0))
    tp = tp_raw * TP_SCALE
    sigs = collect_all_signals(0, split_bar, {p: d})
    trades = run_1pos(sigs)
    if len(trades) >= 5:
        ev = eval_trades(trades)
        dist_wr = sl / (tp + sl) * 100
        edge = ev['wr'] - dist_wr
        mc_p = mc_test(trades, n_sims=2000)
        train_pattern_stats[p] = {
            'trades': len(trades), 'wr': ev['wr'], 'edge': edge,
            'mc_p': mc_p, 'pnl': ev['pnl']
        }

# Patterns that pass MC < 0.05 in first half only
train_pass = {p: all_patterns[p] for p, s in train_pattern_stats.items()
              if s['mc_p'] < 0.05 and s['edge'] > 5}
print(f"  Patterns passing MC<0.05 + edge>5pp in first half: {len(train_pass)}/{len(all_patterns)}")

# Test Method A on second half
sigs_a = collect_all_signals(split_bar, n_bars, train_pass)
trades_a = run_1pos(sigs_a)
ev_a = eval_trades(trades_a)
mc_a = mc_test(trades_a, n_sims=5000)

# Method B: Use ALL 52 patterns on second half (current approach)
sigs_b = collect_all_signals(split_bar, n_bars, all_patterns)
trades_b = run_1pos(sigs_b)
ev_b = eval_trades(trades_b)
mc_b = mc_test(trades_b, n_sims=5000)

# Method C: Use ALL 270 days optimized patterns on second half (look-ahead bias)
# This is the same as B since current patterns were selected using full 270d
# The gap between A and B tells us how much look-ahead helps

print(f"\n  Second-half performance comparison:")
print(f"  {'Method':<35}  {'PnL':>8}  {'Trades':>7}  {'WR':>6}  {'MDD':>7}  {'MC':>8}")
print(f"  {'-'*75}")
print(f"  {'A: First-half-only selection':<35}  {ev_a['pnl']:>+7.1f}%  {ev_a['trades']:>7}  {ev_a['wr']:>5.1f}%  {ev_a['mdd']:>6.1f}%  {mc_a:>8.4f}")
print(f"  {'B: Full 270d selection (current)':<35}  {ev_b['pnl']:>+7.1f}%  {ev_b['trades']:>7}  {ev_b['wr']:>5.1f}%  {ev_b['mdd']:>6.1f}%  {mc_b:>8.4f}")
gap = ev_b['pnl'] - ev_a['pnl']
print(f"\n  Look-ahead gap (B - A): {gap:+.1f}%")
print(f"  Interpretation: {'LARGE gap = significant overfitting' if gap > ev_b['pnl'] * 0.5 else 'MODERATE gap' if gap > ev_b['pnl'] * 0.2 else 'SMALL gap = patterns are robust'}")

# Extra: how does Method A's selected subset differ?
overlap = len(set(train_pass.keys()) & set(all_patterns.keys()))
print(f"\n  Pattern overlap: {overlap}/{len(all_patterns)} current patterns were also selected by first-half-only")
print(f"  First-half-only selected: {len(train_pass)} patterns")

results['phase2_out_of_time'] = {
    'split_bar': split_bar,
    'train_days': split_bar // bars_per_day,
    'test_days': (n_bars - split_bar) // bars_per_day,
    'method_a_first_half': {
        'n_patterns': len(train_pass),
        'pnl': ev_a['pnl'], 'trades': ev_a['trades'], 'wr': ev_a['wr'],
        'mdd': ev_a['mdd'], 'mc': mc_a
    },
    'method_b_full_270d': {
        'n_patterns': len(all_patterns),
        'pnl': ev_b['pnl'], 'trades': ev_b['trades'], 'wr': ev_b['wr'],
        'mdd': ev_b['mdd'], 'mc': mc_b
    },
    'look_ahead_gap': gap,
    'pattern_overlap': overlap,
}


# ============================================================
# Phase 3: Regime Dependence
# ============================================================
print("\n" + "=" * 70)
print("Phase 3: Regime Dependence")
print("=" * 70)

# Classify regimes
# Volatility: ATR percentile
atr_median = np.nanmedian(atr_pct)
atr_p33 = np.nanpercentile(atr_pct, 33)
atr_p67 = np.nanpercentile(atr_pct, 67)

# Trend: 30-bar MA slope
ma_slope = np.zeros(n_bars)
for i in range(30, n_bars):
    ma_slope[i] = (ma_30[i] - ma_30[i - 30]) / ma_30[i - 30] * 100 if ma_30[i - 30] > 0 else 0

slope_p33 = np.nanpercentile(ma_slope[30:], 33)
slope_p67 = np.nanpercentile(ma_slope[30:], 67)

# Assign regime to each bar
regimes = []
for i in range(n_bars):
    if np.isnan(atr_pct[i]) or i < 30:
        regimes.append(('UNKNOWN', 'UNKNOWN'))
        continue
    vol = 'LOW' if atr_pct[i] < atr_p33 else ('HIGH' if atr_pct[i] > atr_p67 else 'MED')
    trend = 'DOWN' if ma_slope[i] < slope_p33 else ('UP' if ma_slope[i] > slope_p67 else 'SIDE')
    regimes.append((vol, trend))

# Run strategy per regime
all_sigs = collect_all_signals()
all_trades = run_1pos(all_sigs)

# Assign regime to each trade (at entry bar)
for t in all_trades:
    bar = t['entry_bar']
    t['vol_regime'] = regimes[bar][0] if bar < n_bars else 'UNKNOWN'
    t['trend_regime'] = regimes[bar][1] if bar < n_bars else 'UNKNOWN'

# Volatility regime performance
print(f"\n  3a. Performance by volatility regime:")
print(f"  {'Regime':>10}  {'PnL':>8}  {'Trades':>7}  {'WR':>6}  {'MDD':>7}  {'AvgPnL':>8}")
print(f"  {'-'*55}")
vol_results = {}
for vol in ['LOW', 'MED', 'HIGH']:
    vt = [t for t in all_trades if t['vol_regime'] == vol]
    ev = eval_trades(vt)
    avg_pnl = np.mean([t['pnl'] for t in vt]) if vt else 0
    print(f"  {vol:>10}  {ev['pnl']:>+7.1f}%  {ev['trades']:>7}  {ev['wr']:>5.1f}%  {ev['mdd']:>6.1f}%  {avg_pnl:>+7.2f}%")
    vol_results[vol] = {'pnl': ev['pnl'], 'trades': ev['trades'], 'wr': ev['wr'],
                        'mdd': ev['mdd'], 'avg_pnl': float(avg_pnl)}

# Trend regime performance
print(f"\n  3b. Performance by trend regime:")
print(f"  {'Regime':>10}  {'PnL':>8}  {'Trades':>7}  {'WR':>6}  {'MDD':>7}  {'AvgPnL':>8}")
print(f"  {'-'*55}")
trend_results = {}
for trend in ['DOWN', 'SIDE', 'UP']:
    tt = [t for t in all_trades if t['trend_regime'] == trend]
    ev = eval_trades(tt)
    avg_pnl = np.mean([t['pnl'] for t in tt]) if tt else 0
    print(f"  {trend:>10}  {ev['pnl']:>+7.1f}%  {ev['trades']:>7}  {ev['wr']:>5.1f}%  {ev['mdd']:>6.1f}%  {avg_pnl:>+7.2f}%")
    trend_results[trend] = {'pnl': ev['pnl'], 'trades': ev['trades'], 'wr': ev['wr'],
                            'mdd': ev['mdd'], 'avg_pnl': float(avg_pnl)}

# Cross-regime
print(f"\n  3c. Cross regime (vol x trend):")
print(f"  {'Vol-Trend':>12}  {'PnL':>8}  {'Trades':>7}  {'WR':>6}  {'AvgPnL':>8}")
print(f"  {'-'*50}")
cross_results = {}
for vol in ['LOW', 'MED', 'HIGH']:
    for trend in ['DOWN', 'SIDE', 'UP']:
        ct = [t for t in all_trades if t['vol_regime'] == vol and t['trend_regime'] == trend]
        ev = eval_trades(ct)
        avg_pnl = np.mean([t['pnl'] for t in ct]) if ct else 0
        key = f"{vol}-{trend}"
        print(f"  {key:>12}  {ev['pnl']:>+7.1f}%  {ev['trades']:>7}  {ev['wr']:>5.1f}%  {avg_pnl:>+7.2f}%")
        cross_results[key] = {'pnl': ev['pnl'], 'trades': ev['trades'], 'wr': ev['wr'],
                              'avg_pnl': float(avg_pnl)}

# Check: is there a regime where strategy loses money?
losing_regimes = [k for k, v in cross_results.items() if v['pnl'] < 0 and v['trades'] >= 10]
print(f"\n  Losing regimes (PnL < 0, >= 10 trades): {losing_regimes if losing_regimes else 'NONE'}")
print(f"  Regime concentration risk: {'HIGH' if losing_regimes else 'LOW'}")

results['phase3_regime'] = {
    'volatility': vol_results,
    'trend': trend_results,
    'cross': cross_results,
    'losing_regimes': losing_regimes,
}


# ============================================================
# Phase 4: Edge Decay Curve
# ============================================================
print("\n" + "=" * 70)
print("Phase 4: Edge Decay Curve")
print("=" * 70)

train_days_list = [30, 60, 90, 120, 150, 180, 210, 240]
oos_window = 30 * bars_per_day  # 30-day OOS window

print(f"  Train N days → test next 30 days")
print(f"  {'Train Days':>12}  {'OOS PnL':>8}  {'Trades':>7}  {'WR':>6}  {'MDD':>7}  {'MC':>8}  {'Patterns':>10}")
print(f"  {'-'*70}")

decay_results = []
for train_d in train_days_list:
    train_end = train_d * bars_per_day
    test_end = train_end + oos_window
    if test_end > n_bars:
        break

    # Select patterns from train period
    selected = {}
    for p, d in all_patterns.items():
        tp_raw, sl = PATTERN_OPTIMAL_TPSL.get(p, (1.0, 2.0))
        tp = tp_raw * TP_SCALE
        sigs = collect_all_signals(0, train_end, {p: d})
        trades = run_1pos(sigs)
        if len(trades) >= 3:
            ev = eval_trades(trades)
            dist_wr = sl / (tp + sl) * 100
            edge = ev['wr'] - dist_wr
            if edge > 0 and ev['pnl'] > 0:
                selected[p] = d

    # Test on OOS window
    sigs_oos = collect_all_signals(train_end, test_end, selected)
    trades_oos = run_1pos(sigs_oos)
    ev_oos = eval_trades(trades_oos)
    mc_p = mc_test(trades_oos, n_sims=2000)

    print(f"  {train_d:>12}  {ev_oos['pnl']:>+7.1f}%  {ev_oos['trades']:>7}  {ev_oos['wr']:>5.1f}%  {ev_oos['mdd']:>6.1f}%  {mc_p:>8.4f}  {len(selected):>10}")
    decay_results.append({
        'train_days': train_d,
        'oos_pnl': ev_oos['pnl'],
        'oos_trades': ev_oos['trades'],
        'oos_wr': ev_oos['wr'],
        'oos_mdd': ev_oos['mdd'],
        'mc_p': mc_p,
        'n_selected': len(selected),
    })

# Check trend
if len(decay_results) >= 4:
    oos_pnls = [d['oos_pnl'] for d in decay_results]
    # Simple linear regression
    x = np.array([d['train_days'] for d in decay_results])
    y = np.array(oos_pnls)
    slope = np.polyfit(x, y, 1)[0]
    print(f"\n  OOS PnL trend slope: {slope:+.3f}% per train day")
    print(f"  Interpretation: {'MORE training → BETTER OOS (no decay)' if slope > 0 else 'MORE training → WORSE OOS (DECAY detected)'}")

results['phase4_edge_decay'] = {
    'decay_results': decay_results,
    'trend_slope': float(slope) if len(decay_results) >= 4 else None,
}


# ============================================================
# Phase 5: Pattern Survival Analysis
# ============================================================
print("\n" + "=" * 70)
print("Phase 5: Pattern Survival Analysis (version history)")
print("=" * 70)

# Historical pattern sets from version history
v1254_patterns = {
    # v1.25.4: 23 patterns (6L+17S), WR>=85% filter
    # These were completely removed in v1.26.0
    'quality': 'WR >= 85%',
    'count': 23,
    'note': 'All removed in v1.26.0 migration'
}

# v1.26.1: 58 patterns from T5 leave-one-out
# v1.26.2: 52 patterns (6 removed for MC/edge)
# v1.27.0: same 52 patterns, different TP/SL

# Test: current 52 patterns on different time sub-periods
print(f"\n  5a. Current 52 patterns - edge stability over time:")

# 3 equal periods
period_bars = n_bars // 3
period_results = []
for pi in range(3):
    start = pi * period_bars
    end = (pi + 1) * period_bars
    sigs = collect_all_signals(start, end)
    trades = run_1pos(sigs)
    ev = eval_trades(trades)
    mc_p = mc_test(trades, n_sims=2000)
    period_results.append({
        'period': pi + 1,
        'days': f"{pi * 90}-{(pi + 1) * 90}",
        **ev, 'mc': mc_p
    })
    print(f"  Period {pi+1} (days {pi*90}-{(pi+1)*90}): PnL={ev['pnl']:+.1f}% WR={ev['wr']:.1f}% Trades={ev['trades']} MC={mc_p:.4f}")

# Per-pattern survival: how many patterns are profitable in ALL 3 periods?
print(f"\n  5b. Per-pattern profitability across 3 periods:")
pat_period_pnl = defaultdict(list)
for pi in range(3):
    start = pi * period_bars
    end = (pi + 1) * period_bars
    for p, d in all_patterns.items():
        sigs = collect_all_signals(start, end, {p: d})
        # No 1-pos filter for individual patterns
        trades_p = []
        for (idx, pat, di, tp, sl) in sigs:
            t = simulate_trade(idx, di, tp, sl)
            if t:
                t['pattern'] = pat
                trades_p.append(t)
        ev = eval_trades(trades_p)
        pat_period_pnl[p].append(ev['pnl'])

all3_positive = 0
at_least2_positive = 0
all3_negative = 0
for p in all_patterns:
    pnls = pat_period_pnl[p]
    if all(pnl > 0 for pnl in pnls):
        all3_positive += 1
    if sum(1 for pnl in pnls if pnl > 0) >= 2:
        at_least2_positive += 1
    if all(pnl <= 0 for pnl in pnls):
        all3_negative += 1

print(f"  All 3 periods positive: {all3_positive}/{len(all_patterns)} ({all3_positive/len(all_patterns)*100:.0f}%)")
print(f"  At least 2/3 positive: {at_least2_positive}/{len(all_patterns)} ({at_least2_positive/len(all_patterns)*100:.0f}%)")
print(f"  All 3 periods negative: {all3_negative}/{len(all_patterns)} ({all3_negative/len(all_patterns)*100:.0f}%)")

# Classify pattern robustness
robust_patterns = []
fragile_patterns = []
for p in all_patterns:
    pnls = pat_period_pnl[p]
    if all(pnl > 0 for pnl in pnls):
        robust_patterns.append(p)
    elif any(pnl <= 0 for pnl in pnls):
        fragile_patterns.append(p)

print(f"\n  Robust patterns (positive in all 3 periods): {len(robust_patterns)}")
print(f"  Fragile patterns (negative in at least 1): {len(fragile_patterns)}")

# Test: robust-only portfolio vs full 52
sigs_robust = collect_all_signals(0, n_bars, {p: all_patterns[p] for p in robust_patterns})
trades_robust = run_1pos(sigs_robust)
ev_robust = eval_trades(trades_robust)

sigs_full = collect_all_signals()
trades_full = run_1pos(sigs_full)
ev_full = eval_trades(trades_full)

print(f"\n  5c. Robust-only vs Full portfolio (270d):")
print(f"  {'Portfolio':<25}  {'PnL':>8}  {'Trades':>7}  {'WR':>6}  {'MDD':>7}  {'PnL/MDD':>8}")
print(f"  {'-'*65}")
pnl_mdd_r = f"{ev_robust['pnl_mdd']:.1f}x" if ev_robust.get('pnl_mdd') else "N/A"
pnl_mdd_f = f"{ev_full['pnl_mdd']:.1f}x" if ev_full.get('pnl_mdd') else "N/A"
print(f"  {'Robust only (' + str(len(robust_patterns)) + ')':<25}  {ev_robust['pnl']:>+7.1f}%  {ev_robust['trades']:>7}  {ev_robust['wr']:>5.1f}%  {ev_robust['mdd']:>6.1f}%  {pnl_mdd_r:>8}")
print(f"  {'Full 52':<25}  {ev_full['pnl']:>+7.1f}%  {ev_full['trades']:>7}  {ev_full['wr']:>5.1f}%  {ev_full['mdd']:>6.1f}%  {pnl_mdd_f:>8}")

results['phase5_survival'] = {
    'period_results': period_results,
    'all3_positive': all3_positive,
    'at_least2_positive': at_least2_positive,
    'all3_negative': all3_negative,
    'robust_count': len(robust_patterns),
    'fragile_count': len(fragile_patterns),
    'robust_patterns': robust_patterns,
    'robust_portfolio': {'pnl': ev_robust['pnl'], 'trades': ev_robust['trades'],
                         'wr': ev_robust['wr'], 'mdd': ev_robust['mdd']},
    'full_portfolio': {'pnl': ev_full['pnl'], 'trades': ev_full['trades'],
                       'wr': ev_full['wr'], 'mdd': ev_full['mdd']},
}


# ============================================================
# Phase 6: Bootstrap Confidence Intervals
# ============================================================
print("\n" + "=" * 70)
print("Phase 6: Bootstrap Confidence Intervals")
print("=" * 70)

# Block bootstrap (resample 30-day blocks to preserve autocorrelation)
block_size = 30 * bars_per_day
n_blocks = n_bars // block_size
n_bootstrap = 5000

print(f"  Block bootstrap: {n_blocks} blocks of {block_size} bars, {n_bootstrap} resamples")

boot_pnls = []
boot_wrs = []
boot_mdds = []
boot_trades = []

for b in range(n_bootstrap):
    # Resample blocks with replacement
    sampled_blocks = np.random.choice(n_blocks, size=n_blocks, replace=True)

    # Collect trades from sampled blocks
    block_trades = []
    for bi in sampled_blocks:
        start = bi * block_size
        end = (bi + 1) * block_size
        sigs = collect_all_signals(start, end)
        trades = run_1pos(sigs)
        block_trades.extend(trades)

    ev = eval_trades(block_trades)
    boot_pnls.append(ev['pnl'])
    boot_wrs.append(ev['wr'])
    boot_mdds.append(ev['mdd'])
    boot_trades.append(ev['trades'])

boot_pnls = np.array(boot_pnls)
boot_wrs = np.array(boot_wrs)
boot_mdds = np.array(boot_mdds)

pnl_ci = (float(np.percentile(boot_pnls, 2.5)), float(np.percentile(boot_pnls, 97.5)))
wr_ci = (float(np.percentile(boot_wrs, 2.5)), float(np.percentile(boot_wrs, 97.5)))
mdd_ci = (float(np.percentile(boot_mdds, 2.5)), float(np.percentile(boot_mdds, 97.5)))

print(f"\n  95% Confidence Intervals ({n_bootstrap} bootstrap resamples):")
print(f"  PnL:    [{pnl_ci[0]:+.1f}%, {pnl_ci[1]:+.1f}%]  (observed: {ev_full['pnl']:+.1f}%)")
print(f"  WR:     [{wr_ci[0]:.1f}%, {wr_ci[1]:.1f}%]  (observed: {ev_full['wr']:.1f}%)")
print(f"  MDD:    [{mdd_ci[0]:.1f}%, {mdd_ci[1]:.1f}%]  (observed: {ev_full['mdd']:.1f}%)")

pnl_positive_pct = np.mean(boot_pnls > 0) * 100
print(f"\n  P(PnL > 0): {pnl_positive_pct:.1f}%")
print(f"  P(PnL > 100%): {np.mean(boot_pnls > 100) * 100:.1f}%")
print(f"  P(WR > 60%): {np.mean(boot_wrs > 60) * 100:.1f}%")
print(f"  P(MDD < 30%): {np.mean(boot_mdds < 30) * 100:.1f}%")

# Worst-case scenario
worst_5pct_pnl = float(np.percentile(boot_pnls, 5))
worst_5pct_mdd = float(np.percentile(boot_mdds, 95))
print(f"\n  Worst 5% scenario:")
print(f"    PnL floor (P5): {worst_5pct_pnl:+.1f}%")
print(f"    MDD ceiling (P95): {worst_5pct_mdd:.1f}%")

results['phase6_bootstrap'] = {
    'n_bootstrap': n_bootstrap,
    'block_size_bars': block_size,
    'pnl_ci_95': pnl_ci,
    'wr_ci_95': wr_ci,
    'mdd_ci_95': mdd_ci,
    'pnl_positive_pct': float(pnl_positive_pct),
    'pnl_gt_100_pct': float(np.mean(boot_pnls > 100) * 100),
    'worst_5pct_pnl': worst_5pct_pnl,
    'worst_5pct_mdd': worst_5pct_mdd,
    'boot_pnl_mean': float(np.mean(boot_pnls)),
    'boot_pnl_std': float(np.std(boot_pnls)),
}


# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY: Overfitting Assessment")
print("=" * 70)

# Compile overall assessment
print(f"""
  Phase 1 (Temporal Stability):
    - Window Jaccard: {avg_jaccard:.3f} ({'STABLE' if avg_jaccard > 0.5 else 'MODERATE' if avg_jaccard > 0.3 else 'UNSTABLE'})
    - First half PnL: {np.mean(first_half):+.1f}% vs Second half: {np.mean(second_half):+.1f}%
    - Always positive patterns: {always_positive}/{len(all_patterns)}

  Phase 2 (Out-of-Time):
    - First-half-only selection OOS: {ev_a['pnl']:+.1f}% ({len(train_pass)} patterns)
    - Full-270d selection OOS: {ev_b['pnl']:+.1f}% ({len(all_patterns)} patterns)
    - Look-ahead gap: {gap:+.1f}%

  Phase 3 (Regime):
    - Losing regimes: {losing_regimes if losing_regimes else 'NONE'}

  Phase 4 (Edge Decay):
    - Trend slope: {slope:+.3f}% per train day
    - Direction: {'No decay' if slope >= 0 else 'DECAY detected'}

  Phase 5 (Pattern Survival):
    - Robust (all 3 periods positive): {all3_positive}/{len(all_patterns)} ({all3_positive/len(all_patterns)*100:.0f}%)
    - Robust-only PnL: {ev_robust['pnl']:+.1f}% vs Full: {ev_full['pnl']:+.1f}%

  Phase 6 (Bootstrap CI):
    - PnL 95% CI: [{pnl_ci[0]:+.1f}%, {pnl_ci[1]:+.1f}%]
    - P(PnL > 0): {pnl_positive_pct:.1f}%
    - Worst 5%: PnL {worst_5pct_pnl:+.1f}%, MDD {worst_5pct_mdd:.1f}%
""")

# Overall verdict
overfitting_signals = 0
robustness_signals = 0

if avg_jaccard < 0.3:
    overfitting_signals += 1
else:
    robustness_signals += 1

if gap > ev_b['pnl'] * 0.5:
    overfitting_signals += 1
else:
    robustness_signals += 1

if losing_regimes:
    overfitting_signals += 1
else:
    robustness_signals += 1

if slope < 0:
    overfitting_signals += 1
else:
    robustness_signals += 1

if all3_positive / len(all_patterns) < 0.3:
    overfitting_signals += 1
else:
    robustness_signals += 1

if pnl_positive_pct < 90:
    overfitting_signals += 1
else:
    robustness_signals += 1

print(f"  Overfitting signals: {overfitting_signals}/6")
print(f"  Robustness signals: {robustness_signals}/6")
if overfitting_signals >= 4:
    verdict = "HIGH OVERFITTING RISK — strategy may not work forward"
elif overfitting_signals >= 2:
    verdict = "MODERATE RISK — some concerns but fundamentally sound"
else:
    verdict = "LOW RISK — strategy appears robust"
print(f"  VERDICT: {verdict}")

results['summary'] = {
    'overfitting_signals': overfitting_signals,
    'robustness_signals': robustness_signals,
    'verdict': verdict,
}

# ============================================================
# Save results
# ============================================================
out_path = Path(__file__).resolve().parent / '../../results/overfitting_forward_decay_research.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✅ Saved: {out_path}")
print("=" * 70)
