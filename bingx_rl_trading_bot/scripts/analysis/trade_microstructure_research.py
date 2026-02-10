"""
Trade Microstructure Research
==============================
v1.26.4 거래의 가격 경로 역학 심층 분석

핵심 질문: TP를 먼저 터치하느냐, SL을 먼저 터치하느냐?
- TP < SL인 패턴(R:R < 1): TP가 가까우므로 먼저 도달할 확률 높음
- TP > SL인 패턴(R:R > 1): SL이 가까우므로 SL에 먼저 도달할 위험
- 핵심: SL에 도달하기 전에 TP를 터치하는 "선착순 경쟁"의 역학

Phase 1: MAE/MFE Analysis (Maximum Adverse/Favorable Excursion)
  - 각 거래의 최대 유리/불리 이동 추적
  - 승리/패배 거래의 MAE/MFE 분포 비교

Phase 2: TP Proximity — SL 거래에서 TP까지 얼마나 접근했나?
  - "Near-miss" 분석: SL 피격 전에 TP의 몇 %까지 도달?
  - 패턴별 near-miss 비율

Phase 3: TP Sensitivity — TP를 줄이면 SL→TP 전환 비율
  - TP를 50/60/70/80/90/100%로 축소 시 승률/PnL 변화
  - 최적 TP 축소 비율 탐색

Phase 4: 최적 TP 배치 — MFE 분포 기반 기대값 최적화
  - "가격이 실제로 도달하는 거리" 기반 TP 배치
  - 패턴별 최적 TP 수준 도출

Phase 5: 선착순 경쟁 시뮬레이션
  - Bar-by-bar 가격 경로 프로파일링
  - TP/SL 방향 "모멘텀" 분석

Phase 6: 종합 권고
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

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000

print("=" * 70)
print(f"Trade Microstructure Research (from v{BOT_VERSION})")
print(f"  Focus: TP vs SL first-touch dynamics")
print(f"  Input: {len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S = "
      f"{len(VALIDATED_LONG_PATTERNS) + len(VALIDATED_SHORT_PATTERNS)} patterns")
print("=" * 70)

# ============================================================
# Build portfolio map
# ============================================================
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", tp, sl)

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

print("Classification done.\n")


# ============================================================
# Microstructure Backtest Engine
# ============================================================
def bt_microstructure(indices, direction, tp_pct, sl_pct):
    """
    Enhanced backtest tracking full price path within each trade.

    Returns list of dicts:
      outcome: 'TP' or 'SL' or 'TIMEOUT'
      pnl: leveraged PnL %
      mfe_pct: Maximum Favorable Excursion (% from entry)
      mae_pct: Maximum Adverse Excursion (% from entry)
      tp_reach_pct: MFE as % of TP distance (100% = hit TP exactly)
      sl_reach_pct: MAE as % of SL distance (100% = hit SL exactly)
      bars_held: number of bars to exit
      mfe_bar: bar offset when MFE occurred
      mae_bar: bar offset when MAE occurred
    """
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

        # Track price path
        max_favorable = 0.0  # Best move towards TP
        max_adverse = 0.0    # Worst move towards SL
        mfe_bar_offset = 0
        mae_bar_offset = 0
        outcome = 'TIMEOUT'
        pnl = 0.0
        exit_bar_offset = 0

        for j_offset, j in enumerate(range(idx + 2, min(idx + 2 + MAX_BARS, n_bars))):
            # Compute favorable/adverse excursion this bar
            if direction == 'LONG':
                favorable = (highs[j] - entry) / entry * 100  # % towards TP
                adverse = (entry - lows[j]) / entry * 100     # % towards SL
            else:
                favorable = (entry - lows[j]) / entry * 100
                adverse = (highs[j] - entry) / entry * 100

            if favorable > max_favorable:
                max_favorable = favorable
                mfe_bar_offset = j_offset
            if adverse > max_adverse:
                max_adverse = adverse
                mae_bar_offset = j_offset

            # Check TP/SL hit
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)

            if ht and hs:
                if abs(tpp - entry) <= abs(slp - entry):
                    outcome = 'TP'
                    pnl = tp_pct * LEVERAGE - FEE_PCT
                else:
                    outcome = 'SL'
                    pnl = -sl_pct * LEVERAGE - FEE_PCT
                exit_bar_offset = j_offset
                break
            elif ht:
                outcome = 'TP'
                pnl = tp_pct * LEVERAGE - FEE_PCT
                exit_bar_offset = j_offset
                break
            elif hs:
                outcome = 'SL'
                pnl = -sl_pct * LEVERAGE - FEE_PCT
                exit_bar_offset = j_offset
                break

        tp_reach = max_favorable / tp_pct * 100 if tp_pct > 0 else 0
        sl_reach = max_adverse / sl_pct * 100 if sl_pct > 0 else 0

        results.append({
            'outcome': outcome,
            'pnl': pnl,
            'mfe_pct': round(max_favorable, 4),
            'mae_pct': round(max_adverse, 4),
            'tp_reach_pct': round(tp_reach, 1),
            'sl_reach_pct': round(sl_reach, 1),
            'bars_held': exit_bar_offset + 1,
            'mfe_bar': mfe_bar_offset,
            'mae_bar': mae_bar_offset,
        })

    return results


# ============================================================
# Phase 1: MAE/MFE Analysis
# ============================================================
def phase1_mae_mfe():
    print("\n" + "=" * 70)
    print("Phase 1: MAE/MFE Analysis (Maximum Adverse/Favorable Excursion)")
    print("=" * 70)

    all_trades = []
    pattern_stats = {}

    for pat, (direction, tp, sl) in sorted(PORTFOLIO.items()):
        if pat not in pattern_indices:
            continue
        indices = pattern_indices[pat]
        results = bt_microstructure(indices, direction, tp, sl)

        tp_trades = [r for r in results if r['outcome'] == 'TP']
        sl_trades = [r for r in results if r['outcome'] == 'SL']

        for r in results:
            r['pattern'] = pat
            r['direction'] = direction
            r['tp_pct'] = tp
            r['sl_pct'] = sl
        all_trades.extend(results)

        if not results:
            continue

        # MAE of winning trades: how far against before winning
        tp_mae = [r['mae_pct'] for r in tp_trades] if tp_trades else [0]
        # MFE of losing trades: how far towards TP before losing
        sl_mfe = [r['mfe_pct'] for r in sl_trades] if sl_trades else [0]

        pattern_stats[pat] = {
            'direction': direction, 'tp': tp, 'sl': sl, 'rr': round(tp / sl, 2),
            'trades': len(results),
            'tp_count': len(tp_trades), 'sl_count': len(sl_trades),
            'wr': round(len(tp_trades) / len(results) * 100, 1),
            # Winning trades: average MAE (how scary was it before winning?)
            'tp_avg_mae': round(np.mean(tp_mae), 3),
            'tp_p90_mae': round(np.percentile(tp_mae, 90), 3) if len(tp_mae) > 2 else 0,
            # Losing trades: average MFE (how close to winning before losing?)
            'sl_avg_mfe': round(np.mean(sl_mfe), 3),
            'sl_p90_mfe': round(np.percentile(sl_mfe, 90), 3) if len(sl_mfe) > 2 else 0,
            # TP reach: for SL trades, how close to TP (% of TP distance)
            'sl_avg_tp_reach': round(np.mean([r['tp_reach_pct'] for r in sl_trades]), 1) if sl_trades else 0,
        }

    # Aggregate stats
    tp_all = [t for t in all_trades if t['outcome'] == 'TP']
    sl_all = [t for t in all_trades if t['outcome'] == 'SL']

    print(f"\n  Total trades: {len(all_trades)} (TP: {len(tp_all)}, SL: {len(sl_all)})")
    print(f"\n  --- Winning trades (TP hit) ---")
    if tp_all:
        mae_vals = [t['mae_pct'] for t in tp_all]
        print(f"  MAE (worst drawdown before winning):")
        print(f"    Mean: {np.mean(mae_vals):.3f}%   Median: {np.median(mae_vals):.3f}%")
        print(f"    P75:  {np.percentile(mae_vals, 75):.3f}%   P90: {np.percentile(mae_vals, 90):.3f}%")
        print(f"    P95:  {np.percentile(mae_vals, 95):.3f}%   Max: {np.max(mae_vals):.3f}%")
        bars_tp = [t['bars_held'] for t in tp_all]
        print(f"  Bars to TP: Mean {np.mean(bars_tp):.0f}, Median {np.median(bars_tp):.0f}, "
              f"P90 {np.percentile(bars_tp, 90):.0f}")

    print(f"\n  --- Losing trades (SL hit) ---")
    if sl_all:
        mfe_vals = [t['mfe_pct'] for t in sl_all]
        tp_reach_vals = [t['tp_reach_pct'] for t in sl_all]
        print(f"  MFE (best point before losing):")
        print(f"    Mean: {np.mean(mfe_vals):.3f}%   Median: {np.median(mfe_vals):.3f}%")
        print(f"    P75:  {np.percentile(mfe_vals, 75):.3f}%   P90: {np.percentile(mfe_vals, 90):.3f}%")
        print(f"  TP reach (how close to TP before SL, % of TP distance):")
        print(f"    Mean: {np.mean(tp_reach_vals):.1f}%   Median: {np.median(tp_reach_vals):.1f}%")
        for thr in [90, 80, 70, 60, 50]:
            count = sum(1 for v in tp_reach_vals if v >= thr)
            print(f"    Reached >= {thr}% of TP: {count}/{len(sl_all)} ({count/len(sl_all)*100:.1f}%)")
        bars_sl = [t['bars_held'] for t in sl_all]
        print(f"  Bars to SL: Mean {np.mean(bars_sl):.0f}, Median {np.median(bars_sl):.0f}, "
              f"P90 {np.percentile(bars_sl, 90):.0f}")

    # Speed comparison
    if tp_all and sl_all:
        print(f"\n  --- Speed: TP vs SL ---")
        avg_bars_tp = np.mean([t['bars_held'] for t in tp_all])
        avg_bars_sl = np.mean([t['bars_held'] for t in sl_all])
        print(f"  Avg bars to TP: {avg_bars_tp:.1f}   Avg bars to SL: {avg_bars_sl:.1f}")
        print(f"  TP is {'faster' if avg_bars_tp < avg_bars_sl else 'slower'} "
              f"(ratio: {avg_bars_tp/avg_bars_sl:.2f}x)")

    return all_trades, pattern_stats


# ============================================================
# Phase 2: TP Proximity (Near-Miss Analysis)
# ============================================================
def phase2_tp_proximity(all_trades, pattern_stats):
    print("\n" + "=" * 70)
    print("Phase 2: TP Proximity — Near-Miss Analysis")
    print("=" * 70)
    print("  Q: SL 피격 거래가 TP의 몇 % 지점까지 도달했는가?")

    sl_trades = [t for t in all_trades if t['outcome'] == 'SL']
    if not sl_trades:
        print("  No SL trades found!")
        return {}

    # Distribution of TP reach for SL trades
    tp_reach_vals = [t['tp_reach_pct'] for t in sl_trades]

    print(f"\n  SL trades: {len(sl_trades)}")
    print(f"\n  TP Reach Distribution (SL trades only):")

    # Histogram
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    hist, _ = np.histogram(tp_reach_vals, bins=bins)
    total = len(tp_reach_vals)
    cumulative = 0
    print(f"  {'Range':<12} {'Count':>6} {'%':>6} {'Cumul%':>8}  Bar")
    for i in range(len(bins) - 1):
        count = hist[i]
        pct = count / total * 100
        cumulative += pct
        bar = "#" * int(pct / 2)
        print(f"  {bins[i]:>3}-{bins[i+1]:>3}%    {count:>4}  {pct:>5.1f}%  {cumulative:>6.1f}%  {bar}")

    # Near-miss categories
    near_miss_90 = sum(1 for v in tp_reach_vals if v >= 90)
    near_miss_80 = sum(1 for v in tp_reach_vals if v >= 80)
    near_miss_70 = sum(1 for v in tp_reach_vals if v >= 70)

    print(f"\n  Near-Miss Summary:")
    print(f"    >= 90% of TP reached: {near_miss_90} trades ({near_miss_90/total*100:.1f}%) — 'Almost won'")
    print(f"    >= 80% of TP reached: {near_miss_80} trades ({near_miss_80/total*100:.1f}%) — 'Close call'")
    print(f"    >= 70% of TP reached: {near_miss_70} trades ({near_miss_70/total*100:.1f}%) — 'Approached'")
    print(f"    <  50% of TP reached: {sum(1 for v in tp_reach_vals if v < 50)} trades — 'Never close'")

    # Per-pattern near-miss analysis
    print(f"\n  Pattern-level Near-Miss (>= 80% TP reach among SL trades):")
    print(f"  {'Pattern':<14} {'Dir':<6} {'TP/SL':<8} {'SL#':>4} {'>=80%':>5} {'NM%':>5} {'AvgReach':>8}")
    print(f"  " + "-" * 60)

    near_miss_by_pat = {}
    for pat in sorted(pattern_stats.keys()):
        ps = pattern_stats[pat]
        pat_sl = [t for t in sl_trades if t['pattern'] == pat]
        if len(pat_sl) < 2:
            continue
        nm80 = sum(1 for t in pat_sl if t['tp_reach_pct'] >= 80)
        nm_pct = nm80 / len(pat_sl) * 100
        avg_reach = np.mean([t['tp_reach_pct'] for t in pat_sl])
        near_miss_by_pat[pat] = {
            'sl_count': len(pat_sl), 'nm80': nm80, 'nm_pct': round(nm_pct, 1),
            'avg_reach': round(avg_reach, 1),
        }
        if nm80 > 0:
            print(f"  {pat:<14} {ps['direction']:<6} {ps['tp']:.1f}/{ps['sl']:.1f} "
                  f"{len(pat_sl):>4} {nm80:>5} {nm_pct:>4.0f}% {avg_reach:>7.1f}%")

    return near_miss_by_pat


# ============================================================
# Phase 3: TP Sensitivity Analysis
# ============================================================
def phase3_tp_sensitivity(all_trades):
    print("\n" + "=" * 70)
    print("Phase 3: TP Sensitivity — TP 축소 시 승률/PnL 변화")
    print("=" * 70)
    print("  Q: TP를 줄이면 SL 거래가 TP 거래로 얼마나 전환되는가?")

    # For each trade, we know the MFE — if MFE >= reduced_tp, it would have been a win
    tp_scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    print(f"\n  {'TP Scale':>9} {'TP Wins':>8} {'SL Loss':>8} {'WR%':>6} {'Total PnL':>10} {'Avg PnL':>8} {'vs Current':>10}")
    print(f"  " + "-" * 72)

    results = {}
    current_pnl = sum(t['pnl'] for t in all_trades if t['outcome'] != 'TIMEOUT')
    current_tp_count = sum(1 for t in all_trades if t['outcome'] == 'TP')
    total_resolved = sum(1 for t in all_trades if t['outcome'] != 'TIMEOUT')

    for scale in tp_scales:
        tp_wins = 0
        sl_losses = 0
        total_pnl = 0.0

        for t in all_trades:
            if t['outcome'] == 'TIMEOUT':
                continue

            tp_pct = t['tp_pct']
            sl_pct = t['sl_pct']
            reduced_tp = tp_pct * scale

            # Check if MFE would have hit the reduced TP
            if t['mfe_pct'] >= reduced_tp:
                # Would have been a TP win at reduced level
                tp_wins += 1
                total_pnl += reduced_tp * LEVERAGE - FEE_PCT
            else:
                # Still an SL loss
                sl_losses += 1
                total_pnl += -sl_pct * LEVERAGE - FEE_PCT

        wr = tp_wins / (tp_wins + sl_losses) * 100 if (tp_wins + sl_losses) > 0 else 0
        avg_pnl = total_pnl / (tp_wins + sl_losses) if (tp_wins + sl_losses) > 0 else 0
        vs_current = total_pnl - current_pnl

        marker = " <-- CURRENT" if scale == 1.0 else ""
        print(f"  {scale:>8.0%}   {tp_wins:>6}   {sl_losses:>6}  {wr:>5.1f}% {total_pnl:>9.1f}% {avg_pnl:>7.2f}% {vs_current:>+9.1f}%{marker}")

        results[f"{int(scale*100)}%"] = {
            'tp_scale': scale,
            'tp_wins': tp_wins, 'sl_losses': sl_losses,
            'wr': round(wr, 1), 'total_pnl': round(total_pnl, 1),
            'avg_pnl': round(avg_pnl, 3), 'vs_current': round(vs_current, 1),
        }

    # Find optimal scale
    best_scale = max(results.items(), key=lambda x: x[1]['total_pnl'])
    print(f"\n  Optimal TP scale: {best_scale[0]} (PnL +{best_scale[1]['total_pnl']:.1f}%)")

    return results


# ============================================================
# Phase 4: Per-Pattern Optimal TP (MFE-based)
# ============================================================
def phase4_optimal_tp_per_pattern(all_trades, pattern_stats):
    print("\n" + "=" * 70)
    print("Phase 4: Per-Pattern Optimal TP Placement")
    print("=" * 70)
    print("  Q: 각 패턴의 MFE 분포 기반 최적 TP는?")

    tp_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    print(f"\n  {'Pattern':<14} {'Dir':<5} {'CurTP':>5} {'CurSL':>5} {'CurWR':>5} {'BestScale':>9} {'NewWR':>5} "
          f"{'CurPnL':>7} {'NewPnL':>7} {'Chg':>6}")
    print(f"  " + "-" * 85)

    for pat in sorted(pattern_stats.keys()):
        ps = pattern_stats[pat]
        pat_trades = [t for t in all_trades if t['pattern'] == pat and t['outcome'] != 'TIMEOUT']

        if len(pat_trades) < 10:
            continue

        tp_pct = ps['tp']
        sl_pct = ps['sl']
        cur_pnl = sum(t['pnl'] for t in pat_trades)
        cur_wr = ps['wr']

        best_scale = 1.0
        best_pnl = cur_pnl
        best_wr = cur_wr

        for scale in tp_scales:
            reduced_tp = tp_pct * scale
            tp_wins = 0
            total_pnl = 0.0

            for t in pat_trades:
                if t['mfe_pct'] >= reduced_tp:
                    tp_wins += 1
                    total_pnl += reduced_tp * LEVERAGE - FEE_PCT
                else:
                    total_pnl += -sl_pct * LEVERAGE - FEE_PCT

            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_scale = scale
                best_wr = tp_wins / len(pat_trades) * 100

        pnl_chg = best_pnl - cur_pnl

        results[pat] = {
            'direction': ps['direction'],
            'current_tp': tp_pct, 'current_sl': sl_pct,
            'current_wr': cur_wr, 'current_pnl': round(cur_pnl, 1),
            'optimal_scale': best_scale,
            'optimal_tp': round(tp_pct * best_scale, 2),
            'new_wr': round(best_wr, 1), 'new_pnl': round(best_pnl, 1),
            'pnl_change': round(pnl_chg, 1),
        }

        marker = ""
        if best_scale < 1.0 and pnl_chg > 5:
            marker = " <<"  # Notable improvement
        print(f"  {pat:<14} {ps['direction']:<5} {tp_pct:>4.1f}% {sl_pct:>4.1f}% {cur_wr:>4.1f}% "
              f"{best_scale:>8.0%}   {best_wr:>4.1f}% {cur_pnl:>6.1f}% {best_pnl:>6.1f}% {pnl_chg:>+5.1f}%{marker}")

    # Summary
    improved = {k: v for k, v in results.items() if v['optimal_scale'] < 1.0 and v['pnl_change'] > 5}
    unchanged = {k: v for k, v in results.items() if v['optimal_scale'] == 1.0}
    print(f"\n  Summary:")
    print(f"    Optimal at current TP (100%): {len(unchanged)} patterns")
    print(f"    Better with reduced TP (PnL > +5%): {len(improved)} patterns")
    if improved:
        total_pnl_gain = sum(v['pnl_change'] for v in improved.values())
        print(f"    Total PnL gain from reductions: +{total_pnl_gain:.1f}%")
        print(f"\n  Top improvements:")
        top_improved = sorted(improved.items(), key=lambda x: x[1]['pnl_change'], reverse=True)[:10]
        for pat, info in top_improved:
            print(f"    {pat}: TP {info['current_tp']}→{info['optimal_tp']} "
                  f"(scale {info['optimal_scale']:.0%}), WR {info['current_wr']}→{info['new_wr']}%, "
                  f"PnL {info['pnl_change']:+.1f}%")

    return results


# ============================================================
# Phase 5: First-Touch Race Dynamics
# ============================================================
def phase5_race_dynamics(all_trades):
    print("\n" + "=" * 70)
    print("Phase 5: TP vs SL First-Touch Race Dynamics")
    print("=" * 70)

    resolved = [t for t in all_trades if t['outcome'] != 'TIMEOUT']
    if not resolved:
        return {}

    # Classify trades by R:R regime
    rr_low = [t for t in resolved if t['tp_pct'] / t['sl_pct'] < 0.75]     # R:R < 0.75
    rr_mid = [t for t in resolved if 0.75 <= t['tp_pct'] / t['sl_pct'] <= 1.25]  # R:R ~ 1.0
    rr_high = [t for t in resolved if t['tp_pct'] / t['sl_pct'] > 1.25]    # R:R > 1.25

    regimes = [
        ("R:R < 0.75 (TP close)", rr_low),
        ("R:R 0.75-1.25 (balanced)", rr_mid),
        ("R:R > 1.25 (TP far)", rr_high),
    ]

    print(f"\n  {'Regime':<28} {'#':>5} {'WR%':>5} {'AvgBar':>6} {'TP bar':>6} {'SL bar':>6} "
          f"{'TP MAE':>6} {'SL MFE':>6} {'SL TPrch':>8}")
    print(f"  " + "-" * 90)

    regime_stats = {}
    for label, trades in regimes:
        if not trades:
            continue
        tp_trades = [t for t in trades if t['outcome'] == 'TP']
        sl_trades = [t for t in trades if t['outcome'] == 'SL']
        wr = len(tp_trades) / len(trades) * 100

        avg_bars = np.mean([t['bars_held'] for t in trades])
        tp_bars = np.mean([t['bars_held'] for t in tp_trades]) if tp_trades else 0
        sl_bars = np.mean([t['bars_held'] for t in sl_trades]) if sl_trades else 0
        tp_mae = np.mean([t['mae_pct'] for t in tp_trades]) if tp_trades else 0
        sl_mfe = np.mean([t['mfe_pct'] for t in sl_trades]) if sl_trades else 0
        sl_tp_reach = np.mean([t['tp_reach_pct'] for t in sl_trades]) if sl_trades else 0

        print(f"  {label:<28} {len(trades):>5} {wr:>4.1f}% {avg_bars:>5.0f} {tp_bars:>5.0f} {sl_bars:>5.0f} "
              f"{tp_mae:>5.3f} {sl_mfe:>5.3f} {sl_tp_reach:>7.1f}%")

        regime_stats[label] = {
            'trades': len(trades), 'wr': round(wr, 1),
            'avg_bars': round(avg_bars, 1),
            'tp_bars': round(tp_bars, 1), 'sl_bars': round(sl_bars, 1),
            'tp_mae_avg': round(tp_mae, 4), 'sl_mfe_avg': round(sl_mfe, 4),
            'sl_tp_reach': round(sl_tp_reach, 1),
        }

    # MFE at time of MAE — do trades "breathe" before resolving?
    print(f"\n  --- Trade 'Breathing' Pattern ---")
    print(f"  For winning trades: MFE bar vs MAE bar (which comes first?)")

    tp_trades = [t for t in resolved if t['outcome'] == 'TP']
    sl_trades = [t for t in resolved if t['outcome'] == 'SL']

    if tp_trades:
        mfe_first = sum(1 for t in tp_trades if t['mfe_bar'] <= t['mae_bar'])
        mae_first = sum(1 for t in tp_trades if t['mae_bar'] < t['mfe_bar'])
        print(f"  TP trades: MFE first {mfe_first} ({mfe_first/len(tp_trades)*100:.1f}%), "
              f"MAE first {mae_first} ({mae_first/len(tp_trades)*100:.1f}%)")
        # Trades where MAE occurs first = "scary dip before winning"
        if mae_first > 0:
            scary_trades = [t for t in tp_trades if t['mae_bar'] < t['mfe_bar']]
            avg_mae_scary = np.mean([t['mae_pct'] for t in scary_trades])
            print(f"    'Scary dip' trades: avg MAE {avg_mae_scary:.3f}% before recovering to TP")

    if sl_trades:
        mfe_first = sum(1 for t in sl_trades if t['mfe_bar'] <= t['mae_bar'])
        mae_first = sum(1 for t in sl_trades if t['mae_bar'] < t['mfe_bar'])
        print(f"  SL trades: MFE first {mfe_first} ({mfe_first/len(sl_trades)*100:.1f}%), "
              f"MAE first {mae_first} ({mae_first/len(sl_trades)*100:.1f}%)")
        # Trades where MFE occurs first = "tease before losing"
        if mfe_first > 0:
            tease_trades = [t for t in sl_trades if t['mfe_bar'] <= t['mae_bar']]
            avg_mfe_tease = np.mean([t['mfe_pct'] for t in tease_trades])
            avg_tp_reach_tease = np.mean([t['tp_reach_pct'] for t in tease_trades])
            print(f"    'Tease' trades: avg MFE {avg_mfe_tease:.3f}% "
                  f"(reached {avg_tp_reach_tease:.1f}% of TP) before reversing to SL")

    return regime_stats


# ============================================================
# Phase 6: Portfolio-Level TP Optimization
# ============================================================
def phase6_portfolio_optimization(all_trades, per_pattern_results):
    print("\n" + "=" * 70)
    print("Phase 6: Portfolio-Level TP Optimization")
    print("=" * 70)

    # Build optimized portfolio maps
    # Strategy A: Current
    # Strategy B: Per-pattern optimal TP (from Phase 4)
    # Strategy C: Uniform 80% TP scale
    # Strategy D: Uniform 70% TP scale
    # Strategy E: Selective — only reduce patterns where gain > 10%

    strategies = {}

    # Strategy A: Current
    cur_map = dict(PORTFOLIO)

    # Strategy B: Per-pattern optimal
    opt_map = {}
    for pat, (direction, tp, sl) in PORTFOLIO.items():
        if pat in per_pattern_results:
            new_tp = per_pattern_results[pat]['optimal_tp']
            opt_map[pat] = (direction, new_tp, sl)
        else:
            opt_map[pat] = (direction, tp, sl)

    # Strategy C: Uniform 80%
    uni80_map = {pat: (d, round(tp * 0.8, 2), sl) for pat, (d, tp, sl) in PORTFOLIO.items()}

    # Strategy D: Uniform 70%
    uni70_map = {pat: (d, round(tp * 0.7, 2), sl) for pat, (d, tp, sl) in PORTFOLIO.items()}

    # Strategy E: Selective (only patterns with > 10% PnL gain)
    sel_map = {}
    for pat, (direction, tp, sl) in PORTFOLIO.items():
        if pat in per_pattern_results and per_pattern_results[pat]['pnl_change'] > 10:
            new_tp = per_pattern_results[pat]['optimal_tp']
            sel_map[pat] = (direction, new_tp, sl)
        else:
            sel_map[pat] = (direction, tp, sl)

    maps = {
        'A_Current': cur_map,
        'B_PerPattern_Optimal': opt_map,
        'C_Uniform_80pct': uni80_map,
        'D_Uniform_70pct': uni70_map,
        'E_Selective_10pct': sel_map,
    }

    print(f"\n  {'Strategy':<24} {'Trades':>6} {'PnL':>8} {'WR%':>6} {'MDD':>6} {'PF':>5} {'PnL/MDD':>8}")
    print(f"  " + "-" * 70)

    for name, pmap in maps.items():
        trades = collect_trades_1pos_simple(pmap)
        stats = eval_trades(trades)
        pnl_mdd = stats['pnl'] / stats['mdd'] if stats['mdd'] > 0 else 999
        marker = " <-- CURRENT" if name == 'A_Current' else ""
        print(f"  {name:<24} {stats['trades']:>6} {stats['pnl']:>+7.1f}% {stats['wr']:>5.1f}% "
              f"{stats['mdd']:>5.1f}% {stats['pf']:>4.2f} {pnl_mdd:>7.1f}x{marker}")
        strategies[name] = {'stats': stats, 'pnl_mdd': round(pnl_mdd, 1)}

    # Changes summary for best non-current strategy
    best_name = max(
        [(n, s) for n, s in strategies.items() if n != 'A_Current'],
        key=lambda x: x[1]['stats']['pnl']
    )
    print(f"\n  Best non-current strategy: {best_name[0]}")
    cur_pnl = strategies['A_Current']['stats']['pnl']
    best_pnl = best_name[1]['stats']['pnl']
    print(f"  PnL change: {cur_pnl:.1f}% → {best_pnl:.1f}% ({best_pnl - cur_pnl:+.1f}%)")

    # Detail the changes in selective strategy
    if 'E_Selective_10pct' in strategies:
        changed_pats = [pat for pat in PORTFOLIO
                        if pat in per_pattern_results and per_pattern_results[pat]['pnl_change'] > 10]
        if changed_pats:
            print(f"\n  Strategy E changes ({len(changed_pats)} patterns):")
            for pat in sorted(changed_pats):
                info = per_pattern_results[pat]
                print(f"    {pat}: TP {info['current_tp']}→{info['optimal_tp']} "
                      f"(WR {info['current_wr']}→{info['new_wr']}%, PnL {info['pnl_change']:+.1f}%)")

    return strategies


def collect_trades_1pos_simple(pattern_map):
    """1-position-at-a-time backtest."""
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


def eval_trades(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    pnl_list = [t[2] for t in trades]
    cum_pnl = 0
    peak_pnl = 0
    max_dd = 0
    wins = 0
    total_win = 0
    total_loss = 0
    for p in pnl_list:
        cum_pnl += p
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            total_win += p
        else:
            total_loss += abs(p)
    return {
        'pnl': cum_pnl,
        'trades': len(pnl_list),
        'wr': wins / len(pnl_list) * 100,
        'mdd': max_dd,
        'pf': total_win / total_loss if total_loss > 0 else 999,
    }


# ============================================================
# Phase 7: Recommendations
# ============================================================
def phase7_recommendations(all_trades, pattern_stats, near_miss, tp_sensitivity,
                           per_pattern, race_dynamics, portfolio_strategies):
    print("\n" + "=" * 70)
    print("Phase 7: Comprehensive Recommendations")
    print("=" * 70)

    recs = []

    # 1. Core finding
    sl_trades = [t for t in all_trades if t['outcome'] == 'SL']
    tp_reach_vals = [t['tp_reach_pct'] for t in sl_trades]
    avg_tp_reach = np.mean(tp_reach_vals) if tp_reach_vals else 0

    recs.append({
        'id': 1, 'category': 'CORE_FINDING',
        'finding': f"SL 거래의 평균 TP 도달률: {avg_tp_reach:.1f}% — 대부분 TP 근처에도 가지 못함",
        'implication': "SL 거래는 '아깝게 놓친' 것이 아니라 애초에 방향이 틀린 거래",
    })

    # 2. Near-miss reality
    nm90 = sum(1 for v in tp_reach_vals if v >= 90) if tp_reach_vals else 0
    nm50 = sum(1 for v in tp_reach_vals if v < 50) if tp_reach_vals else 0
    total_sl = len(sl_trades)
    recs.append({
        'id': 2, 'category': 'NEAR_MISS_REALITY',
        'finding': f"'거의 성공' 거래: {nm90}/{total_sl} ({nm90/total_sl*100:.1f}%)만 TP 90%+ 도달",
        'implication': f"TP 50%에도 못 미치는 거래: {nm50}/{total_sl} ({nm50/total_sl*100:.1f}%) — "
                       "방향 틀린 거래는 TP를 줄여도 구제 불가",
    })

    # 3. TP sensitivity
    best_sens = max(tp_sensitivity.items(), key=lambda x: x[1]['total_pnl'])
    recs.append({
        'id': 3, 'category': 'TP_SENSITIVITY',
        'finding': f"포트폴리오 최적 TP scale: {best_sens[0]} (PnL +{best_sens[1]['total_pnl']:.1f}%)",
        'implication': f"현재 대비 PnL 변화: {best_sens[1]['vs_current']:+.1f}%",
    })

    # 4. Per-pattern TP
    improved_pats = [k for k, v in per_pattern.items() if v['pnl_change'] > 10]
    recs.append({
        'id': 4, 'category': 'PER_PATTERN_TP',
        'finding': f"TP 축소로 PnL > +10% 개선 가능: {len(improved_pats)}개 패턴",
        'implication': ', '.join(f"{p}({per_pattern[p]['optimal_scale']:.0%})" for p in improved_pats[:8]),
    })

    # 5. Race dynamics
    tp_trades = [t for t in all_trades if t['outcome'] == 'TP']
    sl_trades_resolved = [t for t in all_trades if t['outcome'] == 'SL']
    if tp_trades and sl_trades_resolved:
        avg_tp_bars = np.mean([t['bars_held'] for t in tp_trades])
        avg_sl_bars = np.mean([t['bars_held'] for t in sl_trades_resolved])
        recs.append({
            'id': 5, 'category': 'RACE_DYNAMICS',
            'finding': f"TP 평균 {avg_tp_bars:.0f}봉, SL 평균 {avg_sl_bars:.0f}봉에 도달",
            'implication': f"TP가 SL보다 {'빠름' if avg_tp_bars < avg_sl_bars else '느림'} "
                           f"({avg_tp_bars/avg_sl_bars:.2f}x) — "
                           f"{'승리는 빠르고 패배는 느림 (건강한 구조)' if avg_tp_bars < avg_sl_bars else '패배가 더 빠름 (주의)'}",
        })

    # 6. Portfolio strategy
    best_strat = max(
        [(n, s) for n, s in portfolio_strategies.items()],
        key=lambda x: x[1]['stats']['pnl']
    )
    recs.append({
        'id': 6, 'category': 'PORTFOLIO_STRATEGY',
        'finding': f"최적 전략: {best_strat[0]} (PnL +{best_strat[1]['stats']['pnl']:.1f}%)",
        'implication': f"WR {best_strat[1]['stats']['wr']:.1f}%, MDD {best_strat[1]['stats']['mdd']:.1f}%, "
                       f"PF {best_strat[1]['stats']['pf']:.2f}",
    })

    # Print
    for rec in recs:
        print(f"\n  [{rec['id']}] {rec['category']}")
        print(f"      Finding:     {rec['finding']}")
        key = 'implication' if 'implication' in rec else 'detail'
        print(f"      Implication: {rec[key]}")

    return recs


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()

    all_trades, pattern_stats = phase1_mae_mfe()
    near_miss = phase2_tp_proximity(all_trades, pattern_stats)
    tp_sensitivity = phase3_tp_sensitivity(all_trades)
    per_pattern = phase4_optimal_tp_per_pattern(all_trades, pattern_stats)
    race_dynamics = phase5_race_dynamics(all_trades)
    portfolio_strategies = phase6_portfolio_optimization(all_trades, per_pattern)
    recommendations = phase7_recommendations(
        all_trades, pattern_stats, near_miss, tp_sensitivity,
        per_pattern, race_dynamics, portfolio_strategies
    )

    elapsed = time.time() - start_time

    def safe_val(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def safe_dict(d):
        if isinstance(d, dict):
            return {k: safe_val(v) if not isinstance(v, dict) else safe_dict(v)
                    for k, v in d.items()}
        return d

    output = {
        'metadata': {
            'version': BOT_VERSION,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': round(elapsed, 1),
            'total_trades': len(all_trades),
            'parameters': {
                'max_bars': MAX_BARS, 'fee_pct': FEE_PCT,
                'leverage': LEVERAGE, 'mc_sims': MC_SIMS,
            },
        },
        'phase1_pattern_stats': safe_dict(pattern_stats),
        'phase2_near_miss': safe_dict(near_miss),
        'phase3_tp_sensitivity': safe_dict(tp_sensitivity),
        'phase4_per_pattern_optimal': safe_dict(per_pattern),
        'phase5_race_dynamics': safe_dict(race_dynamics),
        'phase6_portfolio_strategies': {
            name: {'stats': safe_dict(s['stats']), 'pnl_mdd': s['pnl_mdd']}
            for name, s in portfolio_strategies.items()
        },
        'phase7_recommendations': recommendations,
    }

    result_path = Path(__file__).resolve().parent / '../../results/trade_microstructure_research.json'
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved: {result_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
