"""
Distance vs Edge Decomposition
================================
WR을 두 요소로 분해:
  1. Distance WR: 랜덤워크 기반 TP 선착순 확률 = SL / (TP + SL)
  2. Genuine Edge: 실제 WR - Distance WR (패턴의 진짜 예측력)

핵심 질문: 높은 WR이 "TP가 가까워서"인가, "패턴이 방향을 맞춰서"인가?

추가 분석:
  - R:R별 distance WR vs actual WR 비교
  - 거리 효과 제거 후 진짜 엣지 순위
  - 기대값 분해: distance component vs edge component
  - SL 거래의 가격 경로 심층 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL, PATTERN_STATS, BOT_VERSION,
)

MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000

print("=" * 70)
print(f"Distance vs Edge Decomposition (from v{BOT_VERSION})")
print("=" * 70)

# Build portfolio
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", tp, sl)

# Load data
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"Loaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

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


def bt_with_excursion(indices, direction, tp_pct, sl_pct):
    """Backtest tracking MFE for every trade (used for SL trade path analysis)."""
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

        max_fav = 0.0
        max_adv = 0.0
        outcome = 'TIMEOUT'
        pnl = 0.0

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if direction == 'LONG':
                fav = (highs[j] - entry) / entry * 100
                adv = (entry - lows[j]) / entry * 100
            else:
                fav = (entry - lows[j]) / entry * 100
                adv = (highs[j] - entry) / entry * 100
            max_fav = max(max_fav, fav)
            max_adv = max(max_adv, adv)

            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                if abs(tpp - entry) <= abs(slp - entry):
                    outcome = 'TP'
                    pnl = tp_pct * LEVERAGE - FEE_PCT
                else:
                    outcome = 'SL'
                    pnl = -sl_pct * LEVERAGE - FEE_PCT
                break
            elif ht:
                outcome = 'TP'
                pnl = tp_pct * LEVERAGE - FEE_PCT
                break
            elif hs:
                outcome = 'SL'
                pnl = -sl_pct * LEVERAGE - FEE_PCT
                break

        results.append({
            'outcome': outcome, 'pnl': pnl,
            'mfe': round(max_fav, 4), 'mae': round(max_adv, 4),
            'tp_reach': round(max_fav / tp_pct * 100, 1) if tp_pct > 0 else 0,
        })
    return results


# ============================================================
# Phase 1: Distance vs Edge Decomposition (per pattern)
# ============================================================
print("=" * 70)
print("Phase 1: WR = Distance Component + Edge Component")
print("=" * 70)
print()
print("  Random Walk TP-first probability = SL / (TP + SL)")
print("  Genuine Edge = Actual WR - Random Walk WR")
print()

results = {}

print(f"  {'Pattern':<14} {'Dir':<5} {'TP':>4} {'SL':>4} {'R:R':>5} "
      f"{'RW_WR':>6} {'Act_WR':>6} {'Edge':>6} {'Tr':>4} {'EV':>7}")
print("  " + "-" * 80)

for pat in sorted(PORTFOLIO.keys()):
    direction, tp, sl = PORTFOLIO[pat]
    if pat not in pattern_indices:
        continue
    indices = pattern_indices[pat]

    pnls = bt_fixed(indices, direction, tp, sl)
    if not pnls:
        continue

    actual_wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    random_walk_wr = sl / (tp + sl) * 100  # Distance-based probability
    edge = actual_wr - random_walk_wr
    rr = tp / sl

    # Expected value per trade
    ev = (actual_wr / 100) * tp * LEVERAGE - (1 - actual_wr / 100) * sl * LEVERAGE - FEE_PCT

    # Random walk expected value (should be ~0 or slightly negative due to fees)
    rw_ev = (random_walk_wr / 100) * tp * LEVERAGE - (1 - random_walk_wr / 100) * sl * LEVERAGE - FEE_PCT

    results[pat] = {
        'direction': direction, 'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'trades': len(pnls),
        'random_walk_wr': round(random_walk_wr, 1),
        'actual_wr': round(actual_wr, 1),
        'edge_pp': round(edge, 1),  # percentage points
        'ev_per_trade': round(ev, 3),
        'rw_ev_per_trade': round(rw_ev, 3),
    }

    marker = ""
    if edge < 5:
        marker = " *THIN*"
    print(f"  {pat:<14} {direction:<5} {tp:>3.1f} {sl:>3.1f} {rr:>4.2f} "
          f"{random_walk_wr:>5.1f}% {actual_wr:>5.1f}% {edge:>+5.1f}pp {len(pnls):>4} {ev:>+6.2f}%{marker}")

# Summary stats
edges = [v['edge_pp'] for v in results.values()]
rw_wrs = [v['random_walk_wr'] for v in results.values()]
act_wrs = [v['actual_wr'] for v in results.values()]

print(f"\n  {'='*60}")
print(f"  SUMMARY")
print(f"  {'='*60}")
print(f"  Average Random Walk WR:  {np.mean(rw_wrs):.1f}%")
print(f"  Average Actual WR:       {np.mean(act_wrs):.1f}%")
print(f"  Average Genuine Edge:    {np.mean(edges):+.1f}pp")
print(f"  Median Genuine Edge:     {np.median(edges):+.1f}pp")
print(f"  Min Edge:                {np.min(edges):+.1f}pp ({min(results.items(), key=lambda x: x[1]['edge_pp'])[0]})")
print(f"  Max Edge:                {np.max(edges):+.1f}pp ({max(results.items(), key=lambda x: x[1]['edge_pp'])[0]})")
print(f"  Edge < 5pp:              {sum(1 for e in edges if e < 5)} patterns (thin edge)")
print(f"  Edge >= 10pp:            {sum(1 for e in edges if e >= 10)} patterns (strong edge)")
print(f"  Edge >= 20pp:            {sum(1 for e in edges if e >= 20)} patterns (exceptional)")

# WR contribution breakdown
weighted_rw = sum(v['random_walk_wr'] * v['trades'] for v in results.values()) / sum(v['trades'] for v in results.values())
weighted_act = sum(v['actual_wr'] * v['trades'] for v in results.values()) / sum(v['trades'] for v in results.values())
weighted_edge = weighted_act - weighted_rw

print(f"\n  Trade-Weighted Breakdown:")
print(f"    WR from distance (TP closer): {weighted_rw:.1f}%")
print(f"    WR from pattern edge:         {weighted_edge:+.1f}pp")
print(f"    Total actual WR:              {weighted_act:.1f}%")
print(f"    Edge contribution:            {weighted_edge/weighted_act*100:.1f}% of total WR")


# ============================================================
# Phase 2: R:R regime analysis
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: R:R Regime — Distance vs Edge")
print("=" * 70)

regimes = {
    'R:R < 0.50 (TP very close)': lambda r: r['rr'] < 0.50,
    'R:R 0.50-0.75':              lambda r: 0.50 <= r['rr'] < 0.75,
    'R:R 0.75-1.00':              lambda r: 0.75 <= r['rr'] < 1.00,
    'R:R 1.00-1.50':              lambda r: 1.00 <= r['rr'] < 1.50,
    'R:R >= 1.50 (TP far)':       lambda r: r['rr'] >= 1.50,
}

print(f"\n  {'Regime':<28} {'#':>3} {'Tr':>5} {'RW_WR':>6} {'ActWR':>6} {'Edge':>6} {'AvgEV':>7}")
print("  " + "-" * 70)

regime_results = {}
for label, filt in regimes.items():
    pats = {k: v for k, v in results.items() if filt(v)}
    if not pats:
        continue
    total_trades = sum(v['trades'] for v in pats.values())
    avg_rw = np.mean([v['random_walk_wr'] for v in pats.values()])
    avg_act = np.mean([v['actual_wr'] for v in pats.values()])
    avg_edge = np.mean([v['edge_pp'] for v in pats.values()])
    avg_ev = np.mean([v['ev_per_trade'] for v in pats.values()])

    print(f"  {label:<28} {len(pats):>3} {total_trades:>5} {avg_rw:>5.1f}% {avg_act:>5.1f}% "
          f"{avg_edge:>+5.1f}pp {avg_ev:>+6.2f}%")

    regime_results[label] = {
        'patterns': len(pats), 'trades': total_trades,
        'avg_rw_wr': round(avg_rw, 1), 'avg_actual_wr': round(avg_act, 1),
        'avg_edge': round(avg_edge, 1), 'avg_ev': round(avg_ev, 3),
    }


# ============================================================
# Phase 3: Edge-ranked patterns (genuine predictive power)
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Patterns Ranked by Genuine Edge (distance effect removed)")
print("=" * 70)

# Sort by edge
sorted_by_edge = sorted(results.items(), key=lambda x: x[1]['edge_pp'], reverse=True)

print(f"\n  TOP 15 (strongest genuine edge):")
print(f"  {'#':>3} {'Pattern':<14} {'R:R':>5} {'RW_WR':>6} {'ActWR':>6} {'Edge':>7} {'EV':>7}")
print("  " + "-" * 58)
for i, (pat, info) in enumerate(sorted_by_edge[:15], 1):
    print(f"  {i:>3} {pat:<14} {info['rr']:>4.2f} {info['random_walk_wr']:>5.1f}% "
          f"{info['actual_wr']:>5.1f}% {info['edge_pp']:>+6.1f}pp {info['ev_per_trade']:>+6.2f}%")

print(f"\n  BOTTOM 10 (thinnest edge):")
print(f"  {'#':>3} {'Pattern':<14} {'R:R':>5} {'RW_WR':>6} {'ActWR':>6} {'Edge':>7} {'EV':>7}")
print("  " + "-" * 58)
for i, (pat, info) in enumerate(sorted_by_edge[-10:], len(sorted_by_edge) - 9):
    print(f"  {i:>3} {pat:<14} {info['rr']:>4.2f} {info['random_walk_wr']:>5.1f}% "
          f"{info['actual_wr']:>5.1f}% {info['edge_pp']:>+6.1f}pp {info['ev_per_trade']:>+6.2f}%")


# ============================================================
# Phase 4: "What if random?" — Expected PnL with zero edge
# ============================================================
print(f"\n{'='*70}")
print("Phase 4: Zero-Edge Simulation — PnL if patterns had no predictive power")
print("=" * 70)

print(f"\n  If WR = random walk WR (distance-only, no pattern edge):")

total_rw_pnl = 0
total_actual_pnl = 0
for pat, info in results.items():
    trades = info['trades']
    tp, sl = info['tp'], info['sl']
    # Random walk PnL
    rw_wr = info['random_walk_wr'] / 100
    rw_pnl_per = rw_wr * tp * LEVERAGE - (1 - rw_wr) * sl * LEVERAGE - FEE_PCT
    total_rw_pnl += rw_pnl_per * trades
    # Actual PnL
    total_actual_pnl += info['ev_per_trade'] * trades

print(f"  Total PnL (random walk, no edge): {total_rw_pnl:+.1f}%")
print(f"  Total PnL (actual, with edge):    {total_actual_pnl:+.1f}%")
print(f"  PnL from edge:                    {total_actual_pnl - total_rw_pnl:+.1f}%")
print(f"  Edge contribution to PnL:         {(total_actual_pnl - total_rw_pnl) / total_actual_pnl * 100:.1f}%")

if total_rw_pnl < 0:
    print(f"\n  ** Random walk PnL is NEGATIVE ({total_rw_pnl:+.1f}%)")
    print(f"     Distance advantage alone does NOT generate profit.")
    print(f"     Fees ({FEE_PCT}%) eat into the distance-based edge.")
    print(f"     ALL profit ({total_actual_pnl:+.1f}%) comes from genuine pattern edge.")
else:
    rw_pct = total_rw_pnl / total_actual_pnl * 100
    edge_pct = (total_actual_pnl - total_rw_pnl) / total_actual_pnl * 100
    print(f"\n  PnL Decomposition:")
    print(f"    From distance advantage: {total_rw_pnl:+.1f}% ({rw_pct:.1f}%)")
    print(f"    From pattern edge:       {total_actual_pnl - total_rw_pnl:+.1f}% ({edge_pct:.1f}%)")


# ============================================================
# Phase 5: SL Trade Deep Dive — Are they "wrong direction" or "unlucky"?
# ============================================================
print(f"\n{'='*70}")
print("Phase 5: SL Trade Classification — Wrong Direction vs Unlucky")
print("=" * 70)

print(f"\n  Classifying SL trades by their MFE (favorable excursion before SL):")
print(f"  - MFE < 25% of TP: 'Wrong Direction' — barely moved towards TP")
print(f"  - MFE 25-75% of TP: 'Reversed' — moved partway then reversed")
print(f"  - MFE >= 75% of TP: 'Unlucky' — almost reached TP, then reversed")

wrong_dir = 0
reversed_trades = 0
unlucky = 0
total_sl = 0

pat_classifications = {}

for pat, (direction, tp, sl) in sorted(PORTFOLIO.items()):
    if pat not in pattern_indices:
        continue
    indices = pattern_indices[pat]
    trades = bt_with_excursion(indices, direction, tp, sl)
    sl_trades = [t for t in trades if t['outcome'] == 'SL']

    if not sl_trades:
        continue

    wd = sum(1 for t in sl_trades if t['tp_reach'] < 25)
    rv = sum(1 for t in sl_trades if 25 <= t['tp_reach'] < 75)
    ul = sum(1 for t in sl_trades if t['tp_reach'] >= 75)

    wrong_dir += wd
    reversed_trades += rv
    unlucky += ul
    total_sl += len(sl_trades)

    pat_classifications[pat] = {
        'sl_count': len(sl_trades),
        'wrong_dir': wd, 'reversed': rv, 'unlucky': ul,
        'wrong_dir_pct': round(wd / len(sl_trades) * 100, 1),
    }

print(f"\n  Overall SL Trade Classification ({total_sl} trades):")
print(f"    Wrong Direction (MFE < 25% TP): {wrong_dir:>4} ({wrong_dir/total_sl*100:.1f}%)")
print(f"    Reversed (MFE 25-75% TP):       {reversed_trades:>4} ({reversed_trades/total_sl*100:.1f}%)")
print(f"    Unlucky (MFE >= 75% TP):         {unlucky:>4} ({unlucky/total_sl*100:.1f}%)")

print(f"\n  Key insight:")
if wrong_dir / total_sl > 0.5:
    print(f"    {wrong_dir/total_sl*100:.0f}% of losses are 'wrong direction' — TP 축소로 구제 불가")
    print(f"    TP를 줄여도 이 거래들은 TP 방향으로 가지 않았음")
print(f"    'Unlucky' 거래는 {unlucky}건({unlucky/total_sl*100:.1f}%)뿐 — TP 축소로 구제 가능한 소수")


# ============================================================
# Phase 6: Summary
# ============================================================
print(f"\n{'='*70}")
print("Phase 6: Final Summary")
print("=" * 70)

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  WR Decomposition (거래수 가중 평균)                         │
  │                                                             │
  │  실제 WR:         {weighted_act:.1f}%                               │
  │  ├─ 거리 효과:    {weighted_rw:.1f}% (TP가 가까워서)                │
  │  └─ 패턴 엣지:   {weighted_edge:+.1f}pp (진짜 예측력)                │
  │  엣지 기여율:     {weighted_edge/weighted_act*100:.1f}% of WR                       │
  │                                                             │
  │  PnL Decomposition                                          │
  │  실제 PnL:        {total_actual_pnl:+.1f}%                         │
  │  랜덤워크 PnL:    {total_rw_pnl:+.1f}% (거리만, 엣지 없으면)       │
  │  엣지에서 오는 PnL: {total_actual_pnl - total_rw_pnl:+.1f}%                        │
  │                                                             │
  │  SL 거래 분류                                                │
  │  방향 틀림:       {wrong_dir/total_sl*100:.0f}% (TP 축소 무효)                      │
  │  반전됨:          {reversed_trades/total_sl*100:.0f}% (부분 구제 가능)                │
  │  운 나쁨:          {unlucky/total_sl*100:.0f}% (TP 축소로 구제)                     │
  └─────────────────────────────────────────────────────────────┘
""")

# Save results
output = {
    'metadata': {
        'version': BOT_VERSION,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    },
    'summary': {
        'weighted_rw_wr': round(weighted_rw, 1),
        'weighted_actual_wr': round(weighted_act, 1),
        'weighted_edge_pp': round(weighted_edge, 1),
        'total_rw_pnl': round(total_rw_pnl, 1),
        'total_actual_pnl': round(total_actual_pnl, 1),
        'pnl_from_edge': round(total_actual_pnl - total_rw_pnl, 1),
        'sl_wrong_direction_pct': round(wrong_dir / total_sl * 100, 1),
        'sl_reversed_pct': round(reversed_trades / total_sl * 100, 1),
        'sl_unlucky_pct': round(unlucky / total_sl * 100, 1),
    },
    'per_pattern': {k: v for k, v in results.items()},
    'regime_analysis': regime_results,
    'sl_classifications': pat_classifications,
}

result_path = Path(__file__).resolve().parent / '../../results/distance_edge_decomposition.json'
with open(result_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Results saved: {result_path}")
