"""
Portfolio Improvement Research — F, B, D
=========================================
v1.27.0 포트폴리오 52패턴의 개선 가능성 탐색.
MC 필터 제거 방식이 실패했으므로 (WF 검증 완료) 대안 접근법 연구.

Phase 1 (F): Signal Scheduling Optimization
  - 1-position-at-a-time에서 신호 차단(blocking) 빈도 분석
  - Oracle DP: 완벽 사전지식 시 최대 PnL (이론적 상한)
  - Post-exit cooldown 최적화
  - WF validation

Phase 2 (B): SL Reduction
  - High-SL 패턴 식별 및 SL 축소 효과
  - Uniform SL cap (2.5%, 2.0%) vs Per-pattern 축소
  - WF validation

Phase 3 (D): Pattern Correlation & Redundancy
  - 신호 타이밍 동시발생 분석
  - 수익 상관관계 매트릭스
  - 중복 패턴 제거 효과
  - WF validation

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
print("Portfolio Improvement Research — F, B, D")
print("=" * 70)

# ============================================================
# Data loading & classification (shared)
# ============================================================
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", round(tp * TP_SCALE, 2), sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", round(tp * TP_SCALE, 2), sl)

data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)
print(f"Data: {n_bars} bars, Portfolio: {len(PORTFOLIO)} patterns")

print("Classifying candles...")
body_abs = np.abs(closes - opens)
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values
types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20[i] if not pd.isna(avg_body_20[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)
print("Done.\n")


# ============================================================
# Shared helpers
# ============================================================
def simulate_trade(pat, direction, tp, sl, signal_bar):
    """Returns (entry_bar, exit_bar, pnl) or None."""
    eb = signal_bar + 1
    if eb >= n_bars:
        return None
    entry = opens[eb]
    if entry <= 0:
        return None
    if direction == 'LONG':
        tpp = entry * (1 + tp / 100)
        slp = entry * (1 - sl / 100)
    else:
        tpp = entry * (1 - tp / 100)
        slp = entry * (1 + sl / 100)
    for j in range(eb + 1, min(eb + MAX_BARS, n_bars)):
        ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
        hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
        if ht and hs:
            pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
            return (eb, j, pnl)
        elif ht:
            return (eb, j, tp * LEVERAGE - FEE_PCT)
        elif hs:
            return (eb, j, -sl * LEVERAGE - FEE_PCT)
    return None


def collect_raw_trades(port, bar_min=0, bar_max=None):
    """All raw trades with pattern name (no 1-pos filter)."""
    if bar_max is None:
        bar_max = n_bars - 1
    raw = []
    for i in range(max(2, bar_min), min(bar_max + 1, n_bars)):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in port:
            continue
        direction, tp, sl = port[pat]
        trade = simulate_trade(pat, direction, tp, sl, i)
        if trade:
            raw.append((*trade, pat))  # (entry, exit, pnl, pattern)
    raw.sort(key=lambda x: x[0])
    return raw


def filter_1pos(raw_trades):
    """Apply 1-position-at-a-time chronological filter."""
    filtered = []
    last_exit = -1
    for eb, xb, pnl, *rest in raw_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl, *rest))
            last_exit = xb
    return filtered


def filter_1pos_cooldown(raw_trades, cooldown_bars=0):
    """1-pos filter with cooldown after each trade exit."""
    filtered = []
    last_exit = -1
    for eb, xb, pnl, *rest in raw_trades:
        if eb > last_exit + cooldown_bars:
            filtered.append((eb, xb, pnl, *rest))
            last_exit = xb
    return filtered


def eval_trades(trades):
    """Portfolio stats from trade list (tuples with >=3 elements, pnl at index 2)."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0}
    pnls = [t[2] for t in trades]
    cum = peak = max_dd = 0
    win_sum = loss_sum = 0
    wins = 0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_sum += p
        else:
            loss_sum += abs(p)
    return {
        'pnl': round(cum, 2),
        'trades': len(pnls),
        'wr': round(wins / len(pnls) * 100, 1),
        'mdd': round(max_dd, 2),
        'pf': round(win_sum / loss_sum, 2) if loss_sum > 0 else 999,
        'pnl_mdd': round(cum / max_dd, 1) if max_dd > 0 else 999,
    }


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 8:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


# WF periods
period_size = n_bars // N_PERIODS
periods = [(p * period_size, (p + 1) * period_size - 1 if p < N_PERIODS - 1 else n_bars - 1)
           for p in range(N_PERIODS)]


def run_wf(portfolio_fn, label=""):
    """Run expanding-window WF. portfolio_fn(train_range) -> portfolio dict for test."""
    results = []
    for wf_i in range(N_PERIODS - 1):
        test_start = periods[wf_i + 1][0]
        test_end = periods[wf_i + 1][1]
        train_start = periods[0][0]
        train_end = periods[wf_i][1]

        port = portfolio_fn((train_start, train_end))
        raw = collect_raw_trades(port, test_start, test_end)
        trades = filter_1pos(raw)
        results.append(eval_trades(trades))

    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['trades'] * r['wr'] / 100 for r in results)
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    max_mdd = max(r['mdd'] for r in results)
    return {
        'total_pnl': round(total_pnl, 2),
        'total_trades': total_trades,
        'avg_wr': round(avg_wr, 1),
        'max_mdd': round(max_mdd, 2),
        'per_fold': results,
    }


# ============================================================
# Phase 1 (F): Signal Scheduling Optimization
# ============================================================
print("=" * 70)
print("Phase 1 (F): Signal Scheduling Optimization")
print("=" * 70)

# 1a. Signal blocking analysis
raw_all = collect_raw_trades(PORTFOLIO)
taken = filter_1pos(raw_all)
blocked = len(raw_all) - len(taken)

print(f"\n  1a. Signal Blocking Analysis (270d)")
print(f"  Total raw signals: {len(raw_all)}")
print(f"  Taken (1-pos): {len(taken)}")
print(f"  Blocked: {blocked} ({blocked/len(raw_all)*100:.1f}%)")

# Edge of taken vs blocked signals
taken_set = set((t[0], t[1]) for t in taken)
taken_pats = [t[3] for t in taken]
blocked_trades = [t for t in raw_all if (t[0], t[1]) not in taken_set]
blocked_pats = [t[3] for t in blocked_trades]

taken_avg_pnl = np.mean([t[2] for t in taken])
blocked_avg_pnl = np.mean([t[2] for t in blocked_trades]) if blocked_trades else 0

print(f"  Taken avg PnL: {taken_avg_pnl:+.2f}%")
print(f"  Blocked avg PnL: {blocked_avg_pnl:+.2f}%")
print(f"  → Blocked signals are {'better' if blocked_avg_pnl > taken_avg_pnl else 'worse'} on average")

# 1b. Oracle DP (upper bound with perfect foresight)
print(f"\n  1b. Oracle DP (perfect foresight upper bound)")

# Weighted job scheduling DP
def oracle_dp(raw_trades):
    """Max-PnL non-overlapping subset (weighted job scheduling)."""
    if not raw_trades:
        return 0, []
    trades = sorted(raw_trades, key=lambda x: x[1])  # sort by exit
    n = len(trades)
    # Find latest non-overlapping predecessor for each trade
    p = [-1] * n
    exit_bars = [t[1] for t in trades]
    entry_bars = [t[0] for t in trades]
    for i in range(n):
        lo, hi = 0, i - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if exit_bars[mid] < entry_bars[i]:
                lo = mid + 1
            else:
                hi = mid - 1
        p[i] = hi

    dp = [0.0] * n
    dp[0] = max(0.0, trades[0][2])
    for i in range(1, n):
        include = trades[i][2] + (dp[p[i]] if p[i] >= 0 else 0)
        exclude = dp[i - 1]
        dp[i] = max(include, exclude)
    return dp[-1]

oracle_pnl = oracle_dp(raw_all)
current_pnl = sum(t[2] for t in taken)
opportunity_gap = oracle_pnl - current_pnl

print(f"  Current (chronological): {current_pnl:+.1f}%")
print(f"  Oracle (perfect foresight): {oracle_pnl:+.1f}%")
print(f"  Opportunity gap: {opportunity_gap:+.1f}% ({opportunity_gap/current_pnl*100:.1f}% of current)")

# 1c. Cooldown sweep
print(f"\n  1c. Post-Exit Cooldown Sweep (270d)")
print(f"  {'Cooldown':>10} {'PnL':>8} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PnL/MDD':>8}")
print(f"  {'-'*55}")

cooldown_results = {}
for cd in [0, 1, 3, 6, 12, 24, 48]:
    trades_cd = filter_1pos_cooldown(raw_all, cd)
    stats = eval_trades(trades_cd)
    cooldown_results[cd] = stats
    label = f"{cd} bars" if cd > 0 else "0 (current)"
    print(f"  {label:>10} {stats['pnl']:>+8.1f} {stats['trades']:>7} "
          f"{stats['wr']:>5.1f}% {stats['mdd']:>6.1f}% {stats['pnl_mdd']:>7.1f}x")

# 1d. WF validation of best cooldowns
print(f"\n  1d. WF Validation of Cooldown Strategies")

def wf_cooldown(cooldown):
    results = []
    for wf_i in range(N_PERIODS - 1):
        ts = periods[wf_i + 1][0]
        te = periods[wf_i + 1][1]
        raw = collect_raw_trades(PORTFOLIO, ts, te)
        trades = filter_1pos_cooldown(raw, cooldown)
        results.append(eval_trades(trades))
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['trades'] * r['wr'] / 100 for r in results)
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    max_mdd = max(r['mdd'] for r in results)
    return {'total_pnl': round(total_pnl, 2), 'total_trades': total_trades,
            'avg_wr': round(avg_wr, 1), 'max_mdd': round(max_mdd, 2)}

print(f"  {'Cooldown':>10} {'OOS PnL':>8} {'Trades':>7} {'WR':>6} {'MaxMDD':>7}")
print(f"  {'-'*45}")
wf_cd_results = {}
for cd in [0, 3, 6, 12, 24]:
    wf = wf_cooldown(cd)
    wf_cd_results[cd] = wf
    label = f"{cd} bars" if cd > 0 else "0 (current)"
    print(f"  {label:>10} {wf['total_pnl']:>+8.1f} {wf['total_trades']:>7} "
          f"{wf['avg_wr']:>5.1f}% {wf['max_mdd']:>6.1f}%")


# ============================================================
# Phase 2 (B): SL Reduction
# ============================================================
print("\n" + "=" * 70)
print("Phase 2 (B): SL Reduction")
print("=" * 70)

# 2a. Current SL distribution
print(f"\n  2a. Current SL Distribution")
sl_values = []
for pat, (direction, tp, sl) in PORTFOLIO.items():
    sl_values.append((pat, sl, direction))
sl_values.sort(key=lambda x: -x[1])

print(f"  {'SL Range':<15} {'Count':>6} {'Patterns':}")
high_sl = [s for s in sl_values if s[1] >= 3.0]
mid_sl = [s for s in sl_values if 2.0 <= s[1] < 3.0]
low_sl = [s for s in sl_values if s[1] < 2.0]
print(f"  {'SL >= 3.0%':<15} {len(high_sl):>6}  {', '.join(s[0] for s in high_sl[:5])}{'...' if len(high_sl)>5 else ''}")
print(f"  {'2.0-2.9%':<15} {len(mid_sl):>6}  {', '.join(s[0] for s in mid_sl[:5])}{'...' if len(mid_sl)>5 else ''}")
print(f"  {'SL < 2.0%':<15} {len(low_sl):>6}  {', '.join(s[0] for s in low_sl[:5])}{'...' if len(low_sl)>5 else ''}")
print(f"  Max SL: {sl_values[0][0]} = {sl_values[0][1]}%")
print(f"  Min SL: {sl_values[-1][0]} = {sl_values[-1][1]}%")
print(f"  Mean SL: {np.mean([s[1] for s in sl_values]):.2f}%")

# 2b. Portfolio SL reduction strategies
print(f"\n  2b. SL Reduction Strategies (270d)")

def build_portfolio_sl(sl_factor=1.0, sl_cap=None):
    """Build portfolio with modified SL."""
    port = {}
    for pat in VALIDATED_LONG_PATTERNS:
        tp_orig, sl_orig = PATTERN_OPTIMAL_TPSL[pat]
        tp = round(tp_orig * TP_SCALE, 2)
        sl = round(sl_orig * sl_factor, 2)
        if sl_cap:
            sl = min(sl, sl_cap)
        port[pat] = ("LONG", tp, sl)
    for pat in VALIDATED_SHORT_PATTERNS:
        tp_orig, sl_orig = PATTERN_OPTIMAL_TPSL[pat]
        tp = round(tp_orig * TP_SCALE, 2)
        sl = round(sl_orig * sl_factor, 2)
        if sl_cap:
            sl = min(sl, sl_cap)
        port[pat] = ("SHORT", tp, sl)
    return port

print(f"  {'Strategy':<25} {'PnL':>8} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8}")
print(f"  {'-'*73}")

sl_strategies = [
    ("Current (no change)", 1.0, None),
    ("SL × 0.9 (10% cut)", 0.9, None),
    ("SL × 0.8 (20% cut)", 0.8, None),
    ("SL × 0.7 (30% cut)", 0.7, None),
    ("SL cap 2.5%", 1.0, 2.5),
    ("SL cap 2.0%", 1.0, 2.0),
    ("SL cap 1.5%", 1.0, 1.5),
]

sl_full_results = {}
for name, factor, cap in sl_strategies:
    port = build_portfolio_sl(factor, cap)
    raw = collect_raw_trades(port)
    trades = filter_1pos(raw)
    stats = eval_trades(trades)
    sl_full_results[name] = stats
    print(f"  {name:<25} {stats['pnl']:>+8.1f} {stats['trades']:>7} "
          f"{stats['wr']:>5.1f}% {stats['mdd']:>6.1f}% {stats['pf']:>6.2f} {stats['pnl_mdd']:>7.1f}x")

# 2c. WF validation of SL strategies
print(f"\n  2c. WF Validation of SL Strategies")

def wf_sl_strategy(sl_factor, sl_cap):
    results = []
    for wf_i in range(N_PERIODS - 1):
        ts = periods[wf_i + 1][0]
        te = periods[wf_i + 1][1]
        port = build_portfolio_sl(sl_factor, sl_cap)
        raw = collect_raw_trades(port, ts, te)
        trades = filter_1pos(raw)
        results.append(eval_trades(trades))
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['trades'] * r['wr'] / 100 for r in results)
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    max_mdd = max(r['mdd'] for r in results)
    return {'total_pnl': round(total_pnl, 2), 'total_trades': total_trades,
            'avg_wr': round(avg_wr, 1), 'max_mdd': round(max_mdd, 2)}

print(f"  {'Strategy':<25} {'OOS PnL':>8} {'Trades':>7} {'WR':>6} {'MaxMDD':>7}")
print(f"  {'-'*55}")
wf_sl_results = {}
for name, factor, cap in sl_strategies:
    wf = wf_sl_strategy(factor, cap)
    wf_sl_results[name] = wf
    print(f"  {name:<25} {wf['total_pnl']:>+8.1f} {wf['total_trades']:>7} "
          f"{wf['avg_wr']:>5.1f}% {wf['max_mdd']:>6.1f}%")


# ============================================================
# Phase 3 (D): Pattern Correlation & Redundancy
# ============================================================
print("\n" + "=" * 70)
print("Phase 3 (D): Pattern Correlation & Redundancy")
print("=" * 70)

# 3a. Signal co-occurrence analysis
print(f"\n  3a. Signal Co-occurrence (temporal proximity)")

# For each pattern, create binary signal vector
pat_list = sorted(PORTFOLIO.keys())
pat_signals = {}
for pat in pat_list:
    vec = np.zeros(n_bars, dtype=bool)
    for i in range(2, n_bars):
        if f"{types[i-2]}-{types[i-1]}-{types[i]}" == pat:
            vec[i] = True
    pat_signals[pat] = vec

# Compute pairwise Jaccard similarity (within ±6 bar window)
WINDOW = 6
high_corr_pairs = []
print(f"\n  Top signal co-occurrence pairs (Jaccard, ±{WINDOW} bar window):")

jaccard_scores = {}
for i, p1 in enumerate(pat_list):
    for j, p2 in enumerate(pat_list):
        if j <= i:
            continue
        s1 = set(np.where(pat_signals[p1])[0])
        s2 = set(np.where(pat_signals[p2])[0])
        # Expand s2 by window
        s2_expanded = set()
        for idx in s2:
            for w in range(-WINDOW, WINDOW + 1):
                s2_expanded.add(idx + w)
        overlap = len(s1 & s2_expanded)
        union = len(s1 | s2)
        jacc = overlap / union if union > 0 else 0
        jaccard_scores[(p1, p2)] = jacc
        if jacc > 0.15:
            high_corr_pairs.append((p1, p2, jacc))

high_corr_pairs.sort(key=lambda x: -x[2])
for p1, p2, jacc in high_corr_pairs[:15]:
    d1 = PORTFOLIO[p1][0][0]
    d2 = PORTFOLIO[p2][0][0]
    print(f"    {p1} ({d1}) ↔ {p2} ({d2}): Jaccard={jacc:.3f}")

if not high_corr_pairs:
    print(f"    No pairs with Jaccard > 0.15 found")

# 3b. Return correlation (trade-level PnL correlation)
print(f"\n  3b. Trade Return Correlation")

# Build per-pattern trade PnL series (aligned by bar)
pat_pnl_series = {}
for pat, (direction, tp, sl) in PORTFOLIO.items():
    pnl_map = {}
    for i in range(2, n_bars):
        if f"{types[i-2]}-{types[i-1]}-{types[i]}" == pat:
            trade = simulate_trade(pat, direction, tp, sl, i)
            if trade:
                pnl_map[i] = trade[2]
    pat_pnl_series[pat] = pnl_map

# Find patterns that co-trigger AND compute return correlation
print(f"\n  Top return-correlated pairs:")
return_corr_pairs = []
for i, p1 in enumerate(pat_list):
    for j, p2 in enumerate(pat_list):
        if j <= i:
            continue
        s1 = pat_pnl_series[p1]
        s2 = pat_pnl_series[p2]
        # Check temporal proximity of signals (within 3 bars)
        shared_bars = []
        for b1 in s1:
            for b2 in s2:
                if abs(b1 - b2) <= 3:
                    shared_bars.append((b1, b2))
        if len(shared_bars) < 5:
            continue
        pnls1 = [s1[b[0]] for b in shared_bars]
        pnls2 = [s2[b[1]] for b in shared_bars]
        corr = np.corrcoef(pnls1, pnls2)[0, 1]
        if abs(corr) > 0.3:
            return_corr_pairs.append((p1, p2, corr, len(shared_bars)))

return_corr_pairs.sort(key=lambda x: -abs(x[2]))
for p1, p2, corr, n_shared in return_corr_pairs[:15]:
    print(f"    {p1} ↔ {p2}: r={corr:+.3f} (n={n_shared})")

if not return_corr_pairs:
    print(f"    No pairs with |r| > 0.3 found (5+ co-occurring trades)")

# 3c. Mirror pattern analysis (known from v1.26.3)
print(f"\n  3c. Mirror Pattern Detection (reverse sequences)")
mirror_pairs = []
for i, p1 in enumerate(pat_list):
    parts = p1.split('-')
    mirror = '-'.join(reversed(parts))
    if mirror in PORTFOLIO and mirror > p1:
        s1 = pat_pnl_series[p1]
        s2 = pat_pnl_series[mirror]
        pnl1 = sum(s1.values()) if s1 else 0
        pnl2 = sum(s2.values()) if s2 else 0
        n1 = len(s1)
        n2 = len(s2)
        mirror_pairs.append((p1, mirror, pnl1, pnl2, n1, n2))
        print(f"    {p1} ↔ {mirror}: PnL {pnl1:+.1f}% ({n1}t) vs {pnl2:+.1f}% ({n2}t)")

if not mirror_pairs:
    print(f"    No mirror pairs found in portfolio")

# 3d. Redundancy removal test
print(f"\n  3d. Redundancy Removal Test")

# Strategy 1: Remove weaker of high-Jaccard pairs
remove_jaccard = set()
for p1, p2, jacc in high_corr_pairs:
    if jacc < 0.20:
        continue
    pnl1 = sum(pat_pnl_series[p1].values()) if pat_pnl_series[p1] else 0
    pnl2 = sum(pat_pnl_series[p2].values()) if pat_pnl_series[p2] else 0
    weaker = p2 if pnl1 >= pnl2 else p1
    remove_jaccard.add(weaker)

# Strategy 2: Remove weaker of mirror pairs
remove_mirror = set()
for p1, p2, pnl1, pnl2, n1, n2 in mirror_pairs:
    weaker = p2 if pnl1 >= pnl2 else p1
    remove_mirror.add(weaker)

# Strategy 3: Combined removal
remove_combined = remove_jaccard | remove_mirror

print(f"  Jaccard removal (Jacc>0.20): {len(remove_jaccard)} patterns")
if remove_jaccard:
    for p in sorted(remove_jaccard):
        print(f"    - {p}")
print(f"  Mirror removal: {len(remove_mirror)} patterns")
if remove_mirror:
    for p in sorted(remove_mirror):
        print(f"    - {p}")
print(f"  Combined: {len(remove_combined)} patterns")

# Full-sample comparison
for name, remove_set in [("Jaccard removal", remove_jaccard),
                          ("Mirror removal", remove_mirror),
                          ("Combined removal", remove_combined)]:
    if not remove_set:
        continue
    port_filtered = {p: v for p, v in PORTFOLIO.items() if p not in remove_set}
    raw = collect_raw_trades(port_filtered)
    trades = filter_1pos(raw)
    stats = eval_trades(trades)
    raw_base = collect_raw_trades(PORTFOLIO)
    base = eval_trades(filter_1pos(raw_base))
    diff = stats['pnl'] - base['pnl']
    print(f"\n  {name} ({len(port_filtered)} patterns):")
    print(f"    PnL: {stats['pnl']:+.1f}% (vs {base['pnl']:+.1f}%, diff {diff:+.1f}%)")
    print(f"    Trades: {stats['trades']}, WR: {stats['wr']:.1f}%, MDD: {stats['mdd']:.1f}%")

# 3e. WF validation of correlation-based removal
print(f"\n  3e. WF Validation of Redundancy Removal")

if remove_combined:
    def wf_corr_removal(remove_set):
        results = []
        for wf_i in range(N_PERIODS - 1):
            ts = periods[wf_i + 1][0]
            te = periods[wf_i + 1][1]
            port = {p: v for p, v in PORTFOLIO.items() if p not in remove_set}
            raw = collect_raw_trades(port, ts, te)
            trades = filter_1pos(raw)
            results.append(eval_trades(trades))
        total_pnl = sum(r['pnl'] for r in results)
        total_trades = sum(r['trades'] for r in results)
        total_wins = sum(r['trades'] * r['wr'] / 100 for r in results)
        avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        max_mdd = max(r['mdd'] for r in results)
        return {'total_pnl': round(total_pnl, 2), 'total_trades': total_trades,
                'avg_wr': round(avg_wr, 1), 'max_mdd': round(max_mdd, 2)}

    wf_base = wf_cooldown(0)  # reuse from Phase 1
    print(f"  {'Strategy':<25} {'OOS PnL':>8} {'Trades':>7} {'WR':>6} {'MaxMDD':>7}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline (52 all)':<25} {wf_base['total_pnl']:>+8.1f} {wf_base['total_trades']:>7} "
          f"{wf_base['avg_wr']:>5.1f}% {wf_base['max_mdd']:>6.1f}%")

    wf_corr_results = {}
    for name, remove_set in [("Jaccard removal", remove_jaccard),
                              ("Mirror removal", remove_mirror),
                              ("Combined removal", remove_combined)]:
        if not remove_set:
            continue
        wf = wf_corr_removal(remove_set)
        wf_corr_results[name] = wf
        print(f"  {name:<25} {wf['total_pnl']:>+8.1f} {wf['total_trades']:>7} "
              f"{wf['avg_wr']:>5.1f}% {wf['max_mdd']:>6.1f}%")
else:
    print(f"  No patterns to remove — skipping WF")
    wf_corr_results = {}


# ============================================================
# Save results
# ============================================================
results = {
    'meta': {
        'script': 'portfolio_improvement_research.py',
        'timestamp': datetime.now().isoformat(),
        'n_bars': n_bars,
        'tp_scale': TP_SCALE,
        'random_seed': 42,
    },
    'phase1_signal_scheduling': {
        'blocking': {
            'total_raw': len(raw_all),
            'taken': len(taken),
            'blocked': blocked,
            'blocked_pct': round(blocked / len(raw_all) * 100, 1),
            'taken_avg_pnl': round(taken_avg_pnl, 2),
            'blocked_avg_pnl': round(blocked_avg_pnl, 2),
        },
        'oracle_dp': {
            'current_pnl': round(current_pnl, 2),
            'oracle_pnl': round(oracle_pnl, 2),
            'opportunity_gap': round(opportunity_gap, 2),
            'gap_pct': round(opportunity_gap / current_pnl * 100, 1),
        },
        'cooldown_full': {str(k): v for k, v in cooldown_results.items()},
        'cooldown_wf': {str(k): v for k, v in wf_cd_results.items()},
    },
    'phase2_sl_reduction': {
        'sl_distribution': {
            'high_sl_count': len(high_sl),
            'mid_sl_count': len(mid_sl),
            'low_sl_count': len(low_sl),
            'mean_sl': round(np.mean([s[1] for s in sl_values]), 2),
        },
        'full_sample': sl_full_results,
        'wf': wf_sl_results,
    },
    'phase3_correlation': {
        'high_jaccard_pairs': [(p1, p2, round(j, 3)) for p1, p2, j in high_corr_pairs[:20]],
        'return_corr_pairs': [(p1, p2, round(c, 3), n)
                               for p1, p2, c, n in return_corr_pairs[:20]],
        'mirror_pairs': mirror_pairs,
        'removal_sets': {
            'jaccard': sorted(remove_jaccard),
            'mirror': sorted(remove_mirror),
            'combined': sorted(remove_combined),
        },
        'wf': wf_corr_results,
    },
}

out_path = Path(__file__).resolve().parent / '../../results/portfolio_improvement_research.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✅ Saved: {out_path}")


# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY — Improvement Opportunities")
print("=" * 70)

print(f"\n  Phase 1 (F): Signal Scheduling")
print(f"    Blocking rate: {blocked/len(raw_all)*100:.1f}% of signals blocked")
print(f"    Oracle gap: {opportunity_gap:+.1f}% ({opportunity_gap/current_pnl*100:.1f}% of current PnL)")
best_cd = max(wf_cd_results.items(), key=lambda x: x[1]['total_pnl'])
print(f"    Best cooldown (WF): {best_cd[0]} bars → OOS PnL {best_cd[1]['total_pnl']:+.1f}%")

print(f"\n  Phase 2 (B): SL Reduction")
best_sl = max(wf_sl_results.items(), key=lambda x: x[1]['total_pnl'])
print(f"    Best SL strategy (WF): {best_sl[0]} → OOS PnL {best_sl[1]['total_pnl']:+.1f}%")
baseline_sl = wf_sl_results.get("Current (no change)", {}).get('total_pnl', 0)
print(f"    vs Baseline: {best_sl[1]['total_pnl'] - baseline_sl:+.1f}%")

print(f"\n  Phase 3 (D): Pattern Correlation")
print(f"    High-Jaccard pairs (>0.20): {sum(1 for _,_,j in high_corr_pairs if j > 0.20)}")
print(f"    Mirror pairs: {len(mirror_pairs)}")
if wf_corr_results:
    best_corr = max(wf_corr_results.items(), key=lambda x: x[1]['total_pnl'])
    print(f"    Best removal (WF): {best_corr[0]} → OOS PnL {best_corr[1]['total_pnl']:+.1f}%")

print("\n" + "=" * 70)
