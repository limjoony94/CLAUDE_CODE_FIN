"""
Dual-TP Stability Research
===========================
v1.26.4 포트폴리오의 "초안정" 패턴 발굴

목표: TP1(0.8×TP) + TP2(1.0×TP) 모두 SL 이전 도달하는
"완전 승리(Full Win)" 비율이 높은 패턴 식별 및 최적화

방법:
  Phase 1: 현재 52패턴 Dual-TP 분석 (FWR = Full Win Rate)
  Phase 2: Grid search로 FWR 극대화 TP/SL 최적화
  Phase 3: MC + WF 검증
  Phase 4: Ultra-Stable 포트폴리오 구성 (FWR 임계값별)
  Phase 5: 종합 권고

백테스트 규칙:
  - Entry: signal bar + 1 open
  - TP1: entry × (1 ± 0.8×tp_pct/100), close 50%
  - TP2: entry × (1 ± tp_pct/100), close remaining 50%
  - SL:  entry × (1 ∓ sl_pct/100), close all remaining
  - 1-position-at-a-time, simple returns, distance-based intrabar resolution
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
    TP1_RATIO, TP1_QTY_PCT,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10          # 0.05% * 2 (round trip)
LEVERAGE = 3
MC_SIMS = 10000
TP1_R = TP1_RATIO       # 0.8 (from constants)

print("=" * 70)
print(f"Dual-TP Stability Research (from v{BOT_VERSION})")
print(f"  TP1 Ratio: {TP1_R} (close {TP1_QTY_PCT}% at TP1)")
print(f"  Input: {len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S = "
      f"{len(VALIDATED_LONG_PATTERNS) + len(VALIDATED_SHORT_PATTERNS)} patterns")
print("=" * 70)

# ============================================================
# Build current portfolio map
# ============================================================
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", tp, sl)

print(f"Portfolio map: {len(PORTFOLIO)} patterns")

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
dates = df['timestamp'].values

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
# Backtest Engines
# ============================================================
def bt_dual_tp(indices, direction, tp_pct, sl_pct):
    """
    Dual-TP backtest: TP1(0.8×TP) → TP2(1.0×TP) with SL tracking.

    Outcomes:
      FULL_WIN:    TP1 hit → TP2 hit (both before SL) — "완전 승리"
      PARTIAL_WIN: TP1 hit → SL hit (TP2 not reached) — 50% 이익 + 50% 손실
      SL_LOSS:     SL hit before TP1 — 전량 손실
      TIMEOUT:     MAX_BARS 내 미결제

    PnL:
      FULL_WIN:    (0.5 × tp1_pct + 0.5 × tp_pct) × LEV - FEE
      PARTIAL_WIN: (0.5 × tp1_pct - 0.5 × sl_pct) × LEV - FEE
      SL_LOSS:     -sl_pct × LEV - FEE
    """
    results = []
    tp1_pct = tp_pct * TP1_R

    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue

        if direction == 'LONG':
            tp1p = entry * (1 + tp1_pct / 100)
            tp2p = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tp1p = entry * (1 - tp1_pct / 100)
            tp2p = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        tp1_hit = False
        outcome = 'TIMEOUT'
        pnl = 0.0

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if not tp1_hit:
                # Phase 1: awaiting TP1 or SL
                ht1 = (highs[j] >= tp1p) if direction == 'LONG' else (lows[j] <= tp1p)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)

                if ht1 and hs:
                    # Same bar: distance-based resolution
                    if abs(tp1p - entry) <= abs(slp - entry):
                        tp1_hit = True
                        ht2 = (highs[j] >= tp2p) if direction == 'LONG' else (lows[j] <= tp2p)
                        if ht2:
                            outcome = 'FULL_WIN'
                            pnl = (0.5 * tp1_pct + 0.5 * tp_pct) * LEVERAGE - FEE_PCT
                            break
                    else:
                        outcome = 'SL_LOSS'
                        pnl = -sl_pct * LEVERAGE - FEE_PCT
                        break
                elif ht1:
                    tp1_hit = True
                    ht2 = (highs[j] >= tp2p) if direction == 'LONG' else (lows[j] <= tp2p)
                    if ht2:
                        outcome = 'FULL_WIN'
                        pnl = (0.5 * tp1_pct + 0.5 * tp_pct) * LEVERAGE - FEE_PCT
                        break
                elif hs:
                    outcome = 'SL_LOSS'
                    pnl = -sl_pct * LEVERAGE - FEE_PCT
                    break
            else:
                # Phase 2: TP1 hit, awaiting TP2 or SL (50% remaining)
                ht2 = (highs[j] >= tp2p) if direction == 'LONG' else (lows[j] <= tp2p)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)

                if ht2 and hs:
                    if abs(tp2p - entry) <= abs(slp - entry):
                        outcome = 'FULL_WIN'
                        pnl = (0.5 * tp1_pct + 0.5 * tp_pct) * LEVERAGE - FEE_PCT
                    else:
                        outcome = 'PARTIAL_WIN'
                        pnl = (0.5 * tp1_pct - 0.5 * sl_pct) * LEVERAGE - FEE_PCT
                    break
                elif ht2:
                    outcome = 'FULL_WIN'
                    pnl = (0.5 * tp1_pct + 0.5 * tp_pct) * LEVERAGE - FEE_PCT
                    break
                elif hs:
                    outcome = 'PARTIAL_WIN'
                    pnl = (0.5 * tp1_pct - 0.5 * sl_pct) * LEVERAGE - FEE_PCT
                    break

        results.append({'outcome': outcome, 'pnl': pnl})

    return results


def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Standard single-TP backtest (for MC/WF validation)."""
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
    """1-position-at-a-time portfolio backtest (standard single-TP)."""
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


def collect_trades_dual_tp(pattern_map):
    """1-position-at-a-time portfolio backtest with dual-TP outcome tracking."""
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
        tp1_pct = tp * TP1_R

        if direction == 'LONG':
            tp1p = entry * (1 + tp1_pct / 100)
            tp2p = entry * (1 + tp / 100)
            slp = entry * (1 - sl / 100)
        else:
            tp1p = entry * (1 - tp1_pct / 100)
            tp2p = entry * (1 - tp / 100)
            slp = entry * (1 + sl / 100)

        tp1_hit = False
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            if not tp1_hit:
                ht1 = (highs[j] >= tp1p) if direction == 'LONG' else (lows[j] <= tp1p)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
                if ht1 and hs:
                    if abs(tp1p - entry) <= abs(slp - entry):
                        tp1_hit = True
                        ht2 = (highs[j] >= tp2p) if direction == 'LONG' else (lows[j] <= tp2p)
                        if ht2:
                            pnl = (0.5 * tp1_pct + 0.5 * tp) * LEVERAGE - FEE_PCT
                            raw_trades.append((entry_bar, j, pnl, 'FULL_WIN'))
                            break
                    else:
                        pnl = -sl * LEVERAGE - FEE_PCT
                        raw_trades.append((entry_bar, j, pnl, 'SL_LOSS'))
                        break
                elif ht1:
                    tp1_hit = True
                    ht2 = (highs[j] >= tp2p) if direction == 'LONG' else (lows[j] <= tp2p)
                    if ht2:
                        pnl = (0.5 * tp1_pct + 0.5 * tp) * LEVERAGE - FEE_PCT
                        raw_trades.append((entry_bar, j, pnl, 'FULL_WIN'))
                        break
                elif hs:
                    pnl = -sl * LEVERAGE - FEE_PCT
                    raw_trades.append((entry_bar, j, pnl, 'SL_LOSS'))
                    break
            else:
                ht2 = (highs[j] >= tp2p) if direction == 'LONG' else (lows[j] <= tp2p)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
                if ht2 and hs:
                    if abs(tp2p - entry) <= abs(slp - entry):
                        pnl = (0.5 * tp1_pct + 0.5 * tp) * LEVERAGE - FEE_PCT
                        raw_trades.append((entry_bar, j, pnl, 'FULL_WIN'))
                    else:
                        pnl = (0.5 * tp1_pct - 0.5 * sl) * LEVERAGE - FEE_PCT
                        raw_trades.append((entry_bar, j, pnl, 'PARTIAL_WIN'))
                    break
                elif ht2:
                    pnl = (0.5 * tp1_pct + 0.5 * tp) * LEVERAGE - FEE_PCT
                    raw_trades.append((entry_bar, j, pnl, 'FULL_WIN'))
                    break
                elif hs:
                    pnl = (0.5 * tp1_pct - 0.5 * sl) * LEVERAGE - FEE_PCT
                    raw_trades.append((entry_bar, j, pnl, 'PARTIAL_WIN'))
                    break

    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for t in raw_trades:
        if t[0] > last_exit:
            filtered.append(t)
            last_exit = t[1]
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
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'pnl_mdd': cum_pnl / max_dd if max_dd > 0 else 999,
    }


# ============================================================
# Phase 1: Current Dual-TP Analysis
# ============================================================
def analyze_current_dual_tp():
    print("\n" + "=" * 70)
    print("Phase 1: Current Portfolio Dual-TP Analysis")
    print("=" * 70)

    results = {}
    for pat, (direction, tp, sl) in sorted(PORTFOLIO.items()):
        if pat not in pattern_indices:
            continue
        indices = pattern_indices[pat]
        dual_results = bt_dual_tp(indices, direction, tp, sl)

        if not dual_results:
            continue

        total = len(dual_results)
        full_wins = sum(1 for r in dual_results if r['outcome'] == 'FULL_WIN')
        partial_wins = sum(1 for r in dual_results if r['outcome'] == 'PARTIAL_WIN')
        sl_losses = sum(1 for r in dual_results if r['outcome'] == 'SL_LOSS')
        timeouts = sum(1 for r in dual_results if r['outcome'] == 'TIMEOUT')
        total_pnl = sum(r['pnl'] for r in dual_results)

        results[pat] = {
            'direction': direction,
            'tp': tp, 'sl': sl, 'rr': round(tp / sl, 2),
            'trades': total,
            'full_win': full_wins,
            'partial_win': partial_wins,
            'sl_loss': sl_losses,
            'timeout': timeouts,
            'full_win_rate': round(full_wins / total * 100, 1),
            'any_win_rate': round((full_wins + partial_wins) / total * 100, 1),
            'total_pnl': round(total_pnl, 2),
        }

    # Sort by full_win_rate descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]['full_win_rate'], reverse=True)

    print(f"\n{'Pattern':<14} {'Dir':<6} {'TP/SL':<8} {'Tr':>4} {'FW':>4} {'PW':>4} {'SL':>4} {'FWR%':>6} {'AnyW%':>6} {'PnL':>8}")
    print("-" * 78)
    for pat, info in sorted_results:
        print(f"{pat:<14} {info['direction']:<6} {info['tp']:.1f}/{info['sl']:.1f} "
              f"{info['trades']:>4} {info['full_win']:>4} {info['partial_win']:>4} {info['sl_loss']:>4} "
              f"{info['full_win_rate']:>5.1f}% {info['any_win_rate']:>5.1f}% {info['total_pnl']:>7.1f}%")

    fwr_list = [v['full_win_rate'] for v in results.values()]
    print(f"\nSummary:")
    print(f"  Total patterns:        {len(results)}")
    print(f"  Average Full Win Rate: {np.mean(fwr_list):.1f}%")
    print(f"  Median Full Win Rate:  {np.median(fwr_list):.1f}%")
    print(f"  FWR >= 80%: {sum(1 for f in fwr_list if f >= 80):>3} patterns")
    print(f"  FWR >= 70%: {sum(1 for f in fwr_list if f >= 70):>3} patterns")
    print(f"  FWR >= 60%: {sum(1 for f in fwr_list if f >= 60):>3} patterns")
    print(f"  FWR >= 50%: {sum(1 for f in fwr_list if f >= 50):>3} patterns")
    print(f"  FWR <  50%: {sum(1 for f in fwr_list if f < 50):>3} patterns")

    return dict(sorted_results)


# ============================================================
# Phase 2: Grid Search for Dual-TP Optimized TP/SL
# ============================================================
def grid_search_dual_tp():
    print("\n" + "=" * 70)
    print("Phase 2: Grid Search — FWR-Optimized TP/SL")
    print("=" * 70)

    tp_grid = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    sl_grid = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"  TP grid: {tp_grid}")
    print(f"  SL grid: {sl_grid}")
    print(f"  Total combos per pattern: {len(tp_grid) * len(sl_grid)}")

    results = {}
    total_pats = len(PORTFOLIO)

    for idx, (pat, (direction, cur_tp, cur_sl)) in enumerate(sorted(PORTFOLIO.items())):
        if pat not in pattern_indices:
            continue
        indices = pattern_indices[pat]
        if len(indices) < 10:
            print(f"  [{idx+1}/{total_pats}] {pat}: skip (only {len(indices)} signals)")
            continue

        print(f"  [{idx+1}/{total_pats}] {pat} ({direction}, {len(indices)} sigs)...", end='', flush=True)

        best = None
        for tp in tp_grid:
            for sl in sl_grid:
                dual_res = bt_dual_tp(indices, direction, tp, sl)
                if not dual_res:
                    continue

                total = len(dual_res)
                full_wins = sum(1 for r in dual_res if r['outcome'] == 'FULL_WIN')
                partial_wins = sum(1 for r in dual_res if r['outcome'] == 'PARTIAL_WIN')
                total_pnl = sum(r['pnl'] for r in dual_res)
                fwr = full_wins / total * 100
                awr = (full_wins + partial_wins) / total * 100

                # Must have positive PnL
                if total_pnl <= 0:
                    continue

                # Composite score: FWR (60%) + per-trade PnL efficiency (40%)
                per_trade_pnl = total_pnl / total
                score = fwr * 0.6 + per_trade_pnl * 10 * 0.4

                if best is None or score > best['score']:
                    best = {
                        'tp': tp, 'sl': sl, 'rr': round(tp / sl, 2),
                        'trades': total, 'full_wins': full_wins, 'partial_wins': partial_wins,
                        'fwr': round(fwr, 1), 'awr': round(awr, 1),
                        'pnl': round(total_pnl, 2),
                        'score': round(score, 2),
                    }

        if best is None:
            print(" no viable combo")
            continue

        # Current FWR for comparison
        cur_dual = bt_dual_tp(indices, direction, cur_tp, cur_sl)
        cur_total = len(cur_dual)
        cur_fw = sum(1 for r in cur_dual if r['outcome'] == 'FULL_WIN')
        cur_fwr = cur_fw / cur_total * 100 if cur_total > 0 else 0
        cur_pnl = sum(r['pnl'] for r in cur_dual)

        # "Improved" = FWR goes up AND PnL doesn't drop more than 20%
        improved = best['fwr'] > cur_fwr + 1.0 and best['pnl'] >= cur_pnl * 0.8

        results[pat] = {
            'direction': direction,
            'current': {'tp': cur_tp, 'sl': cur_sl, 'fwr': round(cur_fwr, 1), 'pnl': round(cur_pnl, 2)},
            'optimized': best,
            'improved': improved,
        }

        marker = " [IMPROVED]" if improved else ""
        print(f" FWR {cur_fwr:.0f}%>{best['fwr']:.0f}% TP {cur_tp}/{best['tp']} SL {cur_sl}/{best['sl']}{marker}")

    improved_count = sum(1 for v in results.values() if v['improved'])
    print(f"\nImproved: {improved_count}/{len(results)} patterns")

    return results


# ============================================================
# Phase 3: Validate Optimized Patterns (MC + WF)
# ============================================================
def validate_optimized(grid_results):
    print("\n" + "=" * 70)
    print("Phase 3: MC + WF Validation of Optimized TP/SL")
    print("=" * 70)

    validated = {}
    accepted_count = 0
    rejected_count = 0

    for pat, info in sorted(grid_results.items()):
        direction = info['direction']

        if not info['improved']:
            # Keep current TP/SL
            validated[pat] = {
                'direction': direction,
                'tp': info['current']['tp'], 'sl': info['current']['sl'],
                'source': 'current', 'fwr': info['current']['fwr'],
            }
            continue

        opt = info['optimized']
        tp, sl = opt['tp'], opt['sl']
        indices = pattern_indices[pat]

        # MC test
        pnls = bt_fixed(indices, direction, tp, sl)
        mc_p = mc_test(pnls)

        # WF test
        wf = walk_forward_test(indices, direction, tp, sl)

        if mc_p < 0.01 and wf >= 4:
            validated[pat] = {
                'direction': direction, 'tp': tp, 'sl': sl,
                'source': 'optimized', 'fwr': opt['fwr'],
                'mc': mc_p, 'wf': wf,
            }
            accepted_count += 1
            print(f"  {pat}: ACCEPTED (FWR {opt['fwr']}%, MC {mc_p:.4f}, WF {wf}/5)")
        else:
            validated[pat] = {
                'direction': direction,
                'tp': info['current']['tp'], 'sl': info['current']['sl'],
                'source': 'current_fallback', 'fwr': info['current']['fwr'],
                'rejected_mc': mc_p, 'rejected_wf': wf,
            }
            rejected_count += 1
            reason = []
            if mc_p >= 0.01:
                reason.append(f"MC {mc_p:.4f}")
            if wf < 4:
                reason.append(f"WF {wf}/5")
            print(f"  {pat}: REJECTED ({', '.join(reason)}) > keep current")

    total_improved = sum(1 for v in grid_results.values() if v['improved'])
    print(f"\nValidation: {accepted_count} accepted, {rejected_count} rejected (of {total_improved} candidates)")

    return validated


# ============================================================
# Phase 4: Ultra-Stable Portfolio Construction
# ============================================================
def build_portfolios(validated, phase1_results):
    print("\n" + "=" * 70)
    print("Phase 4: Portfolio Construction & Comparison")
    print("=" * 70)

    # Build optimized portfolio map
    opt_map = {}
    for pat, info in validated.items():
        opt_map[pat] = (info['direction'], info['tp'], info['sl'])

    # Get best FWR per pattern (optimized if accepted, else current from phase1)
    fwr_map = {}
    for pat in validated:
        if validated[pat]['source'] == 'optimized':
            fwr_map[pat] = validated[pat]['fwr']
        elif pat in phase1_results:
            fwr_map[pat] = phase1_results[pat]['full_win_rate']
        else:
            fwr_map[pat] = 0

    portfolios = {}

    # --- A: Current baseline ---
    cur_trades = collect_trades_1pos(PORTFOLIO)
    cur_stats = eval_trades_simple(cur_trades)
    print(f"\n  A_Current ({len(PORTFOLIO)} pats): {cur_stats['trades']} trades, "
          f"PnL +{cur_stats['pnl']:.1f}%, WR {cur_stats['wr']:.1f}%, "
          f"MDD {cur_stats['mdd']:.1f}%, PF {cur_stats['pf']:.2f}")
    portfolios['A_Current'] = {'stats': cur_stats, 'patterns': len(PORTFOLIO)}

    # --- B: Optimized (all patterns, FWR-tuned TP/SL) ---
    opt_trades = collect_trades_1pos(opt_map)
    opt_stats = eval_trades_simple(opt_trades)
    print(f"  B_Optimized ({len(opt_map)} pats): {opt_stats['trades']} trades, "
          f"PnL +{opt_stats['pnl']:.1f}%, WR {opt_stats['wr']:.1f}%, "
          f"MDD {opt_stats['mdd']:.1f}%, PF {opt_stats['pf']:.2f}")
    portfolios['B_Optimized'] = {'stats': opt_stats, 'patterns': len(opt_map)}

    # --- C: FWR-filtered portfolios ---
    thresholds = [50, 55, 60, 65, 70, 75, 80]
    for thr in thresholds:
        filt_map = {pat: val for pat, val in opt_map.items() if fwr_map.get(pat, 0) >= thr}
        if len(filt_map) < 3:
            print(f"  C_FWR{thr} ({len(filt_map)} pats): too few, skip")
            continue

        trades = collect_trades_1pos(filt_map)
        stats = eval_trades_simple(trades)
        label = f"C_FWR{thr}"
        pnl_mdd = stats['pnl'] / stats['mdd'] if stats['mdd'] > 0 else 999
        print(f"  {label} ({len(filt_map)} pats): {stats['trades']} trades, "
              f"PnL +{stats['pnl']:.1f}%, WR {stats['wr']:.1f}%, "
              f"MDD {stats['mdd']:.1f}%, PF {stats['pf']:.2f}, PnL/MDD {pnl_mdd:.1f}x")
        portfolios[label] = {
            'stats': stats, 'patterns': len(filt_map),
            'pattern_list': sorted(filt_map.keys()),
        }

    # --- D: Current portfolio with dual-TP PnL ---
    print(f"\n  === Dual-TP PnL Portfolios ===")
    dtp_trades = collect_trades_dual_tp(PORTFOLIO)
    dtp_stats = eval_trades_simple(dtp_trades)
    total_dtp = len(dtp_trades)
    fw = sum(1 for t in dtp_trades if t[3] == 'FULL_WIN')
    pw = sum(1 for t in dtp_trades if t[3] == 'PARTIAL_WIN')
    sl = sum(1 for t in dtp_trades if t[3] == 'SL_LOSS')
    print(f"  D_DualTP_Current ({len(PORTFOLIO)} pats): {dtp_stats['trades']} trades, "
          f"PnL +{dtp_stats['pnl']:.1f}%, WR {dtp_stats['wr']:.1f}%, MDD {dtp_stats['mdd']:.1f}%")
    if total_dtp > 0:
        print(f"    FullWin: {fw} ({fw/total_dtp*100:.1f}%), "
              f"Partial: {pw} ({pw/total_dtp*100:.1f}%), "
              f"SL: {sl} ({sl/total_dtp*100:.1f}%)")
    portfolios['D_DualTP_Current'] = {
        'stats': dtp_stats, 'patterns': len(PORTFOLIO),
        'full_wins': fw, 'partial_wins': pw, 'sl_losses': sl,
    }

    # --- E: Optimized with dual-TP PnL ---
    dtp_opt = collect_trades_dual_tp(opt_map)
    dtp_opt_stats = eval_trades_simple(dtp_opt)
    total_dtp2 = len(dtp_opt)
    fw2 = sum(1 for t in dtp_opt if t[3] == 'FULL_WIN')
    pw2 = sum(1 for t in dtp_opt if t[3] == 'PARTIAL_WIN')
    sl2 = sum(1 for t in dtp_opt if t[3] == 'SL_LOSS')
    print(f"  E_DualTP_Optimized ({len(opt_map)} pats): {dtp_opt_stats['trades']} trades, "
          f"PnL +{dtp_opt_stats['pnl']:.1f}%, WR {dtp_opt_stats['wr']:.1f}%, MDD {dtp_opt_stats['mdd']:.1f}%")
    if total_dtp2 > 0:
        print(f"    FullWin: {fw2} ({fw2/total_dtp2*100:.1f}%), "
              f"Partial: {pw2} ({pw2/total_dtp2*100:.1f}%), "
              f"SL: {sl2} ({sl2/total_dtp2*100:.1f}%)")
    portfolios['E_DualTP_Optimized'] = {
        'stats': dtp_opt_stats, 'patterns': len(opt_map),
        'full_wins': fw2, 'partial_wins': pw2, 'sl_losses': sl2,
    }

    # --- F: Best FWR-filtered with dual-TP PnL ---
    for thr in [60, 65, 70]:
        filt_map = {pat: val for pat, val in opt_map.items() if fwr_map.get(pat, 0) >= thr}
        if len(filt_map) < 3:
            continue
        dtp_f = collect_trades_dual_tp(filt_map)
        dtp_f_stats = eval_trades_simple(dtp_f)
        total_f = len(dtp_f)
        fw_f = sum(1 for t in dtp_f if t[3] == 'FULL_WIN')
        pw_f = sum(1 for t in dtp_f if t[3] == 'PARTIAL_WIN')
        sl_f = sum(1 for t in dtp_f if t[3] == 'SL_LOSS')
        label = f"F_DualTP_FWR{thr}"
        print(f"  {label} ({len(filt_map)} pats): {dtp_f_stats['trades']} trades, "
              f"PnL +{dtp_f_stats['pnl']:.1f}%, WR {dtp_f_stats['wr']:.1f}%, MDD {dtp_f_stats['mdd']:.1f}%")
        if total_f > 0:
            print(f"    FullWin: {fw_f} ({fw_f/total_f*100:.1f}%), "
                  f"Partial: {pw_f} ({pw_f/total_f*100:.1f}%), "
                  f"SL: {sl_f} ({sl_f/total_f*100:.1f}%)")
        portfolios[label] = {
            'stats': dtp_f_stats, 'patterns': len(filt_map),
            'pattern_list': sorted(filt_map.keys()),
            'full_wins': fw_f, 'partial_wins': pw_f, 'sl_losses': sl_f,
        }

    return portfolios, fwr_map


# ============================================================
# Phase 5: Recommendations
# ============================================================
def generate_recommendations(phase1, grid_results, validated, portfolios, fwr_map):
    print("\n" + "=" * 70)
    print("Phase 5: Recommendations")
    print("=" * 70)

    recs = []

    # 1. FWR distribution
    fwr_values = [v['full_win_rate'] for v in phase1.values()]
    recs.append({
        'id': 1,
        'category': 'FWR_DISTRIBUTION',
        'finding': f"현재 52패턴 평균 Full Win Rate: {np.mean(fwr_values):.1f}% (중앙값 {np.median(fwr_values):.1f}%)",
        'detail': f"FWR>=80%: {sum(1 for f in fwr_values if f >= 80)}, "
                  f"FWR>=70%: {sum(1 for f in fwr_values if f >= 70)}, "
                  f"FWR>=60%: {sum(1 for f in fwr_values if f >= 60)}, "
                  f"FWR<50%: {sum(1 for f in fwr_values if f < 50)}",
    })

    # 2. Optimization results
    improved = sum(1 for v in grid_results.values() if v['improved'])
    accepted = sum(1 for v in validated.values() if v['source'] == 'optimized')
    recs.append({
        'id': 2,
        'category': 'OPTIMIZATION',
        'finding': f"Grid search 개선 후보: {improved}개 → MC/WF 검증 통과: {accepted}개",
        'detail': "FWR > current + 1pp, PnL >= 80% of current, MC < 0.01, WF >= 4/5",
    })

    # 3. Best portfolio (PnL/MDD ratio)
    print(f"\n  === Portfolio Ranking (by PnL/MDD) ===")
    ranked = []
    for name, pf in portfolios.items():
        s = pf.get('stats', {})
        if s.get('mdd', 0) > 0 and s.get('pnl', 0) > 0:
            ratio = s['pnl'] / s['mdd']
            ranked.append((name, ratio, s['pnl'], s['mdd'], s['wr'], s.get('trades', 0), pf.get('patterns', 0)))
    ranked.sort(key=lambda x: x[1], reverse=True)

    for i, (name, ratio, pnl, mdd, wr, tr, npat) in enumerate(ranked[:8], 1):
        marker = " <-- CURRENT" if name == 'A_Current' else ""
        print(f"  {i}. {name:<22} PnL/MDD {ratio:>5.1f}x  PnL +{pnl:>7.1f}%  MDD {mdd:>5.1f}%  WR {wr:>5.1f}%  {tr:>3}t  {npat}p{marker}")

    if ranked:
        best = ranked[0]
        recs.append({
            'id': 3,
            'category': 'BEST_PORTFOLIO',
            'finding': f"PnL/MDD 최고: {best[0]} (PnL/MDD {best[1]:.1f}x)",
            'detail': f"PnL +{best[2]:.1f}%, MDD {best[3]:.1f}%, WR {best[4]:.1f}%, {best[5]} trades, {best[6]} patterns",
        })

    # 4. Ultra-stable patterns
    ultra_stable = [(pat, fwr_map[pat]) for pat in fwr_map if fwr_map[pat] >= 70]
    ultra_stable.sort(key=lambda x: x[1], reverse=True)
    if ultra_stable:
        recs.append({
            'id': 4,
            'category': 'ULTRA_STABLE_PATTERNS',
            'finding': f"FWR >= 70% 초안정 패턴: {len(ultra_stable)}개",
            'detail': ', '.join(f"{p}({f:.0f}%)" for p, f in ultra_stable[:10]),
        })

    # 5. Dual-TP impact
    d_cur = portfolios.get('D_DualTP_Current', {})
    a_cur = portfolios.get('A_Current', {})
    if d_cur.get('stats') and a_cur.get('stats'):
        pnl_diff = d_cur['stats']['pnl'] - a_cur['stats']['pnl']
        recs.append({
            'id': 5,
            'category': 'DUAL_TP_IMPACT',
            'finding': f"Dual-TP PnL: +{d_cur['stats']['pnl']:.1f}% (단일TP 대비 {pnl_diff:+.1f}%)",
            'detail': f"FullWin {d_cur.get('full_wins', 0)}, "
                      f"Partial {d_cur.get('partial_wins', 0)}, "
                      f"SL {d_cur.get('sl_losses', 0)}",
        })

    # Print recs
    print(f"\n  === Recommendations ===")
    for rec in recs:
        print(f"\n  [{rec['id']}] {rec['category']}")
        print(f"      {rec['finding']}")
        print(f"      {rec['detail']}")

    return recs


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()

    phase1 = analyze_current_dual_tp()
    grid_results = grid_search_dual_tp()
    validated = validate_optimized(grid_results)
    portfolios, fwr_map = build_portfolios(validated, phase1)
    recommendations = generate_recommendations(phase1, grid_results, validated, portfolios, fwr_map)

    elapsed = time.time() - start_time

    # Save results
    def safe_val(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    output = {
        'metadata': {
            'version': BOT_VERSION,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': round(elapsed, 1),
            'tp1_ratio': TP1_R,
            'tp1_qty_pct': TP1_QTY_PCT,
            'parameters': {
                'max_bars': MAX_BARS, 'fee_pct': FEE_PCT,
                'leverage': LEVERAGE, 'mc_sims': MC_SIMS,
            },
        },
        'phase1_dual_tp_analysis': {
            k: {kk: safe_val(vv) for kk, vv in v.items()}
            for k, v in phase1.items()
        },
        'phase2_grid_search': {
            pat: {
                'direction': v['direction'],
                'current': v['current'],
                'optimized': {kk: safe_val(vv) for kk, vv in v['optimized'].items()},
                'improved': v['improved'],
            } for pat, v in grid_results.items()
        },
        'phase3_validated': {
            k: {kk: safe_val(vv) for kk, vv in v.items()}
            for k, v in validated.items()
        },
        'phase4_portfolios': {
            name: {
                'patterns': p.get('patterns', 0),
                'stats': {k: round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v
                          for k, v in p.get('stats', {}).items()},
                **{k: safe_val(v) for k, v in p.items() if k not in ('patterns', 'stats')},
            } for name, p in portfolios.items()
        },
        'phase5_recommendations': recommendations,
    }

    result_path = Path(__file__).resolve().parent / '../../results/dual_tp_stability_research.json'
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved: {result_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
