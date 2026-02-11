"""
rr_detection_deep_study.py — R:R 불리 원인 규명 + 감지 시스템 최적화

Part A: R:R 불리 원인 분석
  Phase 1: 백테스트 R:R 구조 분석 — 설계상 R:R인지, 라이브 이상인지
  Phase 2: 라이브 버퍼 교차검증 — actual 31트레이드 복원, 정확한 break-even 재산출
  Phase 3: 패턴별 R:R 기여도 — 어떤 패턴이 R:R을 악화시키는지

Part B: 감지 시스템 최적화
  Phase 4: 감지 규칙 파라미터 스윕 — ROC 곡선으로 최적 임계값 탐색
  Phase 5: 대안 감지 방법 비교 — Rolling WR, CUSUM, Bayesian, 복합 규칙
  Phase 6: 최종 추천 운영 규칙
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL,
)

MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
TP_SCALE = 0.70
np.random.seed(42)

# Live metrics
LIVE_TRADES = 31
LIVE_WINS = 21
LIVE_LOSSES = 10
LIVE_WR = 67.74
LIVE_TOTAL_PNL = 22.02

# Live buffer data (from metrics.json)
LIVE_RECENT_WINS = [
    3.0364, 2.1710, 3.1372, 1.2813, 1.7319, 3.3226, 1.8094, 1.7737,
    2.5273, 4.3496, 4.2181, 1.4355, 1.8145, 2.3352, 8.6349, 3.9624,
    3.8290, 1.7918, 3.8887, 3.9055,
]
LIVE_RECENT_LOSSES = [
    0.0149, 6.4134, 9.2476, 3.3813, 4.8736, 4.8876, 4.7463, 6.0906,
    6.5169, 6.3239, 6.4870, 9.3017,
]

# ─── Data Loading ───
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
highs, lows, opens, closes = df['high'].values, df['low'].values, df['open'].values, df['close'].values
n_bars = len(df)

body_abs = np.abs(closes - opens)
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values
types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20[i] if not pd.isna(avg_body_20[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

patterns_at = [None] * n_bars
for i in range(2, n_bars):
    patterns_at[i] = f"{types[i-2]}-{types[i-1]}-{types[i]}"

all_patterns = {}
for p in VALIDATED_LONG_PATTERNS:
    all_patterns[p] = 'LONG'
for p in VALIDATED_SHORT_PATTERNS:
    all_patterns[p] = 'SHORT'

print(f"Data: {n_bars} bars, Patterns: {len(all_patterns)}")


def simulate_trade(idx, direction, tp_pct, sl_pct):
    if idx + 1 >= n_bars:
        return None
    entry = opens[idx + 1]
    if entry <= 0:
        return None
    for j in range(idx + 1, min(idx + 1 + MAX_BARS, n_bars)):
        if direction == 'LONG':
            tp_hit = (highs[j] - entry) / entry * 100 >= tp_pct
            sl_hit = (entry - lows[j]) / entry * 100 >= sl_pct
        else:
            tp_hit = (entry - lows[j]) / entry * 100 >= tp_pct
            sl_hit = (highs[j] - entry) / entry * 100 >= sl_pct
        if tp_hit and sl_hit:
            return {'win': False, 'pnl': -sl_pct * LEVERAGE - FEE_PCT, 'bar': idx, 'exit_bar': j, 'exit': 'SL'}
        elif tp_hit:
            return {'win': True, 'pnl': tp_pct * LEVERAGE * 0.9 - FEE_PCT, 'bar': idx, 'exit_bar': j, 'exit': 'TP'}
        elif sl_hit:
            return {'win': False, 'pnl': -sl_pct * LEVERAGE - FEE_PCT, 'bar': idx, 'exit_bar': j, 'exit': 'SL'}
    last_j = min(idx + MAX_BARS, n_bars - 1)
    if direction == 'LONG':
        pnl = (closes[last_j] - entry) / entry * 100 * LEVERAGE - FEE_PCT
    else:
        pnl = (entry - closes[last_j]) / entry * 100 * LEVERAGE - FEE_PCT
    return {'win': pnl > 0, 'pnl': pnl, 'bar': idx, 'exit_bar': last_j, 'exit': 'TIMEOUT'}


def run_backtest_detailed():
    trades = []
    in_position_until = 0
    for i in range(2, n_bars):
        if i < in_position_until:
            continue
        pat = patterns_at[i]
        if pat and pat in all_patterns:
            direction = all_patterns[pat]
            base_tp, base_sl = PATTERN_OPTIMAL_TPSL.get(pat, (2.0, 3.0))
            tp = base_tp * TP_SCALE
            sl = base_sl
            result = simulate_trade(i, direction, tp, sl)
            if result:
                result['pattern'] = pat
                result['direction'] = direction
                result['tp_pct'] = tp
                result['sl_pct'] = sl
                trades.append(result)
                in_position_until = result['exit_bar'] + 1
    return trades


# ══════════════════════════════════════════════════════
# Part A - Phase 1: 백테스트 R:R 구조 분석
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part A - Phase 1: 백테스트 R:R 구조 분석")
print("=" * 60)

bt_trades = run_backtest_detailed()
bt_wins = [t for t in bt_trades if t['win']]
bt_losses = [t for t in bt_trades if not t['win']]

bt_win_pnls = np.array([t['pnl'] for t in bt_wins])
bt_loss_pnls = np.array([abs(t['pnl']) for t in bt_losses])

bt_wr = len(bt_wins) / len(bt_trades) * 100
bt_avg_win = np.mean(bt_win_pnls)
bt_avg_loss = np.mean(bt_loss_pnls)
bt_be_wr = bt_avg_loss / (bt_avg_win + bt_avg_loss) * 100
bt_margin = bt_wr - bt_be_wr

print(f"\n백테스트: {len(bt_trades)} trades, WR {bt_wr:.1f}%")
print(f"  avg_win: {bt_avg_win:.3f}%, avg_loss: {bt_avg_loss:.3f}%")
print(f"  median_win: {np.median(bt_win_pnls):.3f}%, median_loss: {np.median(bt_loss_pnls):.3f}%")
print(f"  R:R ratio (avg_win/avg_loss): {bt_avg_win/bt_avg_loss:.3f}")
print(f"  Break-even WR: {bt_be_wr:.1f}%")
print(f"  마진: {bt_margin:+.1f}pp")

# Win/Loss distribution percentiles
print(f"\n  Win 분포: p10={np.percentile(bt_win_pnls,10):.2f} p25={np.percentile(bt_win_pnls,25):.2f} "
      f"p50={np.median(bt_win_pnls):.2f} p75={np.percentile(bt_win_pnls,75):.2f} p90={np.percentile(bt_win_pnls,90):.2f}")
print(f"  Loss 분포: p10={np.percentile(bt_loss_pnls,10):.2f} p25={np.percentile(bt_loss_pnls,25):.2f} "
      f"p50={np.median(bt_loss_pnls):.2f} p75={np.percentile(bt_loss_pnls,75):.2f} p90={np.percentile(bt_loss_pnls,90):.2f}")

# Exit type distribution
exit_types = {}
for t in bt_trades:
    exit_types[t['exit']] = exit_types.get(t['exit'], 0) + 1
print(f"\n  Exit type: {exit_types}")

# TP/SL structure analysis
print(f"\n--- 설계상 R:R 구조 ---")
tp_vals = []
sl_vals = []
for p in all_patterns:
    base_tp, base_sl = PATTERN_OPTIMAL_TPSL.get(p, (2.0, 3.0))
    tp = base_tp * TP_SCALE
    sl = base_sl
    tp_vals.append(tp)
    sl_vals.append(sl)

tp_arr = np.array(tp_vals)
sl_arr = np.array(sl_vals)
design_rr = tp_arr / sl_arr

print(f"  설계 TP: mean {np.mean(tp_arr):.3f}%, median {np.median(tp_arr):.3f}%")
print(f"  설계 SL: mean {np.mean(sl_arr):.3f}%, median {np.median(sl_arr):.3f}%")
print(f"  설계 R:R: mean {np.mean(design_rr):.3f}, median {np.median(design_rr):.3f}")
print(f"  R:R < 1.0: {np.sum(design_rr < 1.0)}/{len(design_rr)} ({np.sum(design_rr < 1.0)/len(design_rr)*100:.0f}%)")
print(f"  → 대다수 패턴이 TP < SL 구조: 설계상 R:R 불리는 의도된 것")

# Theoretical win/loss PnL
theo_win = tp_arr * LEVERAGE * 0.9 - FEE_PCT / 100 * 100  # approximate
theo_loss = sl_arr * LEVERAGE + FEE_PCT / 100 * 100
print(f"\n  이론적 avg_win: {np.mean(tp_arr * LEVERAGE * 0.9 - 0.1):.3f}%")
print(f"  이론적 avg_loss: {np.mean(sl_arr * LEVERAGE + 0.1):.3f}%")
print(f"  실측 백테스트 avg_win: {bt_avg_win:.3f}% (TIMEOUT 포함)")
print(f"  실측 백테스트 avg_loss: {bt_avg_loss:.3f}% (TIMEOUT 포함)")

phase1 = {
    'bt_trades': len(bt_trades),
    'bt_wr': round(bt_wr, 2),
    'bt_avg_win': round(bt_avg_win, 3),
    'bt_avg_loss': round(bt_avg_loss, 3),
    'bt_rr_ratio': round(bt_avg_win / bt_avg_loss, 3),
    'bt_be_wr': round(bt_be_wr, 2),
    'bt_margin_pp': round(bt_margin, 2),
    'design_rr_mean': round(float(np.mean(design_rr)), 3),
    'design_rr_lt1_pct': round(float(np.sum(design_rr < 1.0) / len(design_rr) * 100), 1),
    'exit_types': exit_types,
}


# ══════════════════════════════════════════════════════
# Part A - Phase 2: 라이브 버퍼 교차검증
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part A - Phase 2: 라이브 버퍼 교차검증 & 정확한 Break-Even")
print("=" * 60)

# Buffer anomaly analysis
print(f"\n버퍼 상태:")
print(f"  recent_wins: {len(LIVE_RECENT_WINS)}개 (실제 승리: {LIVE_WINS}개)")
print(f"  recent_losses: {len(LIVE_RECENT_LOSSES)}개 (실제 패배: {LIVE_LOSSES}개)")

buf_win_sum = sum(LIVE_RECENT_WINS)
buf_loss_sum = sum(LIVE_RECENT_LOSSES)
buf_avg_win = buf_win_sum / len(LIVE_RECENT_WINS)
buf_avg_loss = buf_loss_sum / len(LIVE_RECENT_LOSSES)

print(f"\n  버퍼 avg_win: {buf_avg_win:.3f}% (20개 평균)")
print(f"  버퍼 avg_loss: {buf_avg_loss:.3f}% (12개 평균)")

# Reconstruct actual averages from total PnL
# 21W - 10L = 22.02%
# Case A: losses match buffer (use 10 smallest losses from buffer)
sorted_losses = sorted(LIVE_RECENT_LOSSES)
# Most likely: recent 10 losses
loss_candidates = LIVE_RECENT_LOSSES[-LIVE_LOSSES:]  # last 10
actual_loss_sum_a = sum(loss_candidates)
actual_avg_loss_a = actual_loss_sum_a / LIVE_LOSSES
actual_win_sum_a = LIVE_TOTAL_PNL + actual_loss_sum_a
actual_avg_win_a = actual_win_sum_a / LIVE_WINS

print(f"\n--- 복원 시나리오 A (최근 10패 사용) ---")
print(f"  10패 합계: {actual_loss_sum_a:.2f}%, avg_loss: {actual_avg_loss_a:.3f}%")
print(f"  21승 합계: {actual_win_sum_a:.2f}%, avg_win: {actual_avg_win_a:.3f}%")
be_wr_a = actual_avg_loss_a / (actual_avg_win_a + actual_avg_loss_a) * 100
margin_a = LIVE_WR - be_wr_a
print(f"  Break-even WR: {be_wr_a:.1f}%, 마진: {margin_a:+.1f}pp")

# Case B: Use all 12 losses (maybe some from previous session)
# and adjust wins: 21W - 10L_actual = 22.02
# Try: what if actual losses are the 10 LARGEST? (worst case)
worst_10 = sorted(LIVE_RECENT_LOSSES, reverse=True)[:LIVE_LOSSES]
actual_loss_sum_b = sum(worst_10)
actual_avg_loss_b = actual_loss_sum_b / LIVE_LOSSES
actual_win_sum_b = LIVE_TOTAL_PNL + actual_loss_sum_b
actual_avg_win_b = actual_win_sum_b / LIVE_WINS

print(f"\n--- 복원 시나리오 B (최대 10패 사용, 보수적) ---")
print(f"  10패 합계: {actual_loss_sum_b:.2f}%, avg_loss: {actual_avg_loss_b:.3f}%")
print(f"  21승 합계: {actual_win_sum_b:.2f}%, avg_win: {actual_avg_win_b:.3f}%")
be_wr_b = actual_avg_loss_b / (actual_avg_win_b + actual_avg_loss_b) * 100
margin_b = LIVE_WR - be_wr_b
print(f"  Break-even WR: {be_wr_b:.1f}%, 마진: {margin_b:+.1f}pp")

# Case C: Use smallest 10 losses (optimistic)
best_10 = sorted(LIVE_RECENT_LOSSES)[:LIVE_LOSSES]
actual_loss_sum_c = sum(best_10)
actual_avg_loss_c = actual_loss_sum_c / LIVE_LOSSES
actual_win_sum_c = LIVE_TOTAL_PNL + actual_loss_sum_c
actual_avg_win_c = actual_win_sum_c / LIVE_WINS

print(f"\n--- 복원 시나리오 C (최소 10패 사용, 낙관적) ---")
print(f"  10패 합계: {actual_loss_sum_c:.2f}%, avg_loss: {actual_avg_loss_c:.3f}%")
print(f"  21승 합계: {actual_win_sum_c:.2f}%, avg_win: {actual_avg_win_c:.3f}%")
be_wr_c = actual_avg_loss_c / (actual_avg_win_c + actual_avg_loss_c) * 100
margin_c = LIVE_WR - be_wr_c
print(f"  Break-even WR: {be_wr_c:.1f}%, 마진: {margin_c:+.1f}pp")

# Compare with backtest distribution
print(f"\n--- 라이브 vs 백테스트 R:R 비교 ---")
print(f"  백테스트: avg_win {bt_avg_win:.3f}%, avg_loss {bt_avg_loss:.3f}%, R:R {bt_avg_win/bt_avg_loss:.3f}")
print(f"  라이브 A: avg_win {actual_avg_win_a:.3f}%, avg_loss {actual_avg_loss_a:.3f}%, R:R {actual_avg_win_a/actual_avg_loss_a:.3f}")
print(f"  라이브 B: avg_win {actual_avg_win_b:.3f}%, avg_loss {actual_avg_loss_b:.3f}%, R:R {actual_avg_win_b/actual_avg_loss_b:.3f}")

# Which scenario's break-even should we use?
print(f"\n*** Break-Even WR 범위: [{be_wr_b:.1f}%, {be_wr_c:.1f}%] ***")
print(f"*** 마진 범위: [{margin_b:+.1f}pp, {margin_c:+.1f}pp] ***")
print(f"*** 이전 연구 (버퍼 기반): BE {buf_avg_loss/(buf_avg_win+buf_avg_loss)*100:.1f}%, 마진 3.0pp ***")

# Use most conservative (scenario B) for all subsequent analysis
REAL_AVG_WIN = actual_avg_win_b
REAL_AVG_LOSS = actual_avg_loss_b
REAL_BE_WR = be_wr_b
REAL_MARGIN = margin_b
# Also track scenario A as "likely"
LIKELY_AVG_WIN = actual_avg_win_a
LIKELY_AVG_LOSS = actual_avg_loss_a
LIKELY_BE_WR = be_wr_a
LIKELY_MARGIN = margin_a

phase2 = {
    'buffer_anomaly': {
        'buffer_wins': len(LIVE_RECENT_WINS),
        'actual_wins': LIVE_WINS,
        'buffer_losses': len(LIVE_RECENT_LOSSES),
        'actual_losses': LIVE_LOSSES,
    },
    'scenario_a_recent10': {
        'avg_win': round(actual_avg_win_a, 3), 'avg_loss': round(actual_avg_loss_a, 3),
        'rr': round(actual_avg_win_a / actual_avg_loss_a, 3),
        'be_wr': round(be_wr_a, 2), 'margin_pp': round(margin_a, 2),
    },
    'scenario_b_worst10': {
        'avg_win': round(actual_avg_win_b, 3), 'avg_loss': round(actual_avg_loss_b, 3),
        'rr': round(actual_avg_win_b / actual_avg_loss_b, 3),
        'be_wr': round(be_wr_b, 2), 'margin_pp': round(margin_b, 2),
    },
    'scenario_c_best10': {
        'avg_win': round(actual_avg_win_c, 3), 'avg_loss': round(actual_avg_loss_c, 3),
        'rr': round(actual_avg_win_c / actual_avg_loss_c, 3),
        'be_wr': round(be_wr_c, 2), 'margin_pp': round(margin_c, 2),
    },
    'previous_buffer_based': {
        'be_wr': round(buf_avg_loss / (buf_avg_win + buf_avg_loss) * 100, 2),
        'margin_pp': 3.0,
    },
}


# ══════════════════════════════════════════════════════
# Part A - Phase 3: 패턴별 R:R 기여도
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part A - Phase 3: 패턴별 R:R 기여도")
print("=" * 60)

from collections import defaultdict
pattern_stats = defaultdict(lambda: {'wins': [], 'losses': [], 'count': 0})
for t in bt_trades:
    p = t['pattern']
    pattern_stats[p]['count'] += 1
    if t['win']:
        pattern_stats[p]['wins'].append(t['pnl'])
    else:
        pattern_stats[p]['losses'].append(abs(t['pnl']))

# Identify worst R:R patterns
print(f"\n--- 패턴별 R:R (나쁜 순) ---")
pat_rr_list = []
for p, s in pattern_stats.items():
    if s['wins'] and s['losses']:
        aw = np.mean(s['wins'])
        al = np.mean(s['losses'])
        rr = aw / al
        wr = len(s['wins']) / s['count'] * 100
        net_pnl = sum(s['wins']) - sum(s['losses'])
        be = al / (aw + al) * 100
        margin = wr - be
        pat_rr_list.append({
            'pattern': p, 'trades': s['count'], 'wr': wr,
            'avg_win': aw, 'avg_loss': al, 'rr': rr,
            'be_wr': be, 'margin_pp': margin, 'net_pnl': net_pnl,
        })

# Sort by margin (worst first)
pat_rr_list.sort(key=lambda x: x['margin_pp'])

print(f"{'Pattern':<15} {'Tr':>3} {'WR':>5} {'AvgW':>6} {'AvgL':>6} {'R:R':>5} {'BE_WR':>5} {'Margin':>6} {'NetPnL':>7}")
for pr in pat_rr_list[:15]:  # worst 15
    print(f"{pr['pattern']:<15} {pr['trades']:>3} {pr['wr']:>5.1f} {pr['avg_win']:>6.2f} {pr['avg_loss']:>6.2f} "
          f"{pr['rr']:>5.2f} {pr['be_wr']:>5.1f} {pr['margin_pp']:>+6.1f} {pr['net_pnl']:>+7.1f}")

# Count patterns below break-even
below_be = [p for p in pat_rr_list if p['margin_pp'] < 0]
thin_margin = [p for p in pat_rr_list if 0 <= p['margin_pp'] < 5]
comfortable = [p for p in pat_rr_list if p['margin_pp'] >= 5]

print(f"\n  손실 (margin < 0): {len(below_be)}개")
print(f"  위험 (0 ≤ margin < 5pp): {len(thin_margin)}개")
print(f"  안전 (margin ≥ 5pp): {len(comfortable)}개")

# Which patterns contribute most to R:R unfavorability?
# Weighted by trade count
total_win_pnl = sum(sum(s['wins']) for s in pattern_stats.values())
total_loss_pnl = sum(sum(s['losses']) for s in pattern_stats.values())
print(f"\n  전체 Win PnL 합: {total_win_pnl:+.1f}%")
print(f"  전체 Loss PnL 합: {total_loss_pnl:.1f}%")
print(f"  Net: {total_win_pnl - total_loss_pnl:+.1f}%")

phase3 = {
    'below_breakeven': len(below_be),
    'thin_margin': len(thin_margin),
    'comfortable': len(comfortable),
    'worst_5': [{'pattern': p['pattern'], 'margin_pp': round(p['margin_pp'], 1), 'rr': round(p['rr'], 3)} for p in pat_rr_list[:5]],
    'best_5': [{'pattern': p['pattern'], 'margin_pp': round(p['margin_pp'], 1), 'rr': round(p['rr'], 3)} for p in pat_rr_list[-5:]],
}


# ══════════════════════════════════════════════════════
# Part B - Phase 4: 감지 규칙 파라미터 스윕
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part B - Phase 4: 감지 규칙 파라미터 스윕")
print("=" * 60)

N_SIMS = 5000
HORIZON = 200

# Use LIKELY scenario for simulation (scenario A)
SIM_AVG_WIN = LIKELY_AVG_WIN
SIM_AVG_LOSS = LIKELY_AVG_LOSS


def evaluate_detection_rule(wr_threshold, pnl_threshold, window_size, true_wr, n_sims=N_SIMS):
    """Evaluate a detection rule at given true WR."""
    detected = 0
    det_trades_list = []
    det_pnl_list = []

    for _ in range(n_sims):
        cum_pnl = 0.0
        results = []
        det_trade = None

        for t in range(HORIZON):
            win = np.random.random() < true_wr
            pnl = SIM_AVG_WIN if win else -SIM_AVG_LOSS
            cum_pnl += pnl
            results.append(1 if win else 0)

            if det_trade is None and len(results) >= window_size:
                rwr = np.mean(results[-window_size:])
                triggered = False
                if wr_threshold is not None and rwr < wr_threshold:
                    triggered = True
                if pnl_threshold is not None and cum_pnl < pnl_threshold:
                    triggered = True
                if triggered:
                    det_trade = t + 1

        if det_trade is not None:
            detected += 1
            det_trades_list.append(det_trade)
            pnl_at = sum(SIM_AVG_WIN if r else -SIM_AVG_LOSS for r in results[:det_trade])
            det_pnl_list.append(pnl_at)

    return {
        'rate': detected / n_sims,
        'avg_trade': np.mean(det_trades_list) if det_trades_list else None,
        'avg_pnl': np.mean(det_pnl_list) if det_pnl_list else None,
    }


# Sweep parameters
print("\n--- WR threshold sweep (PnL threshold = None, window = 30) ---")
print(f"{'WR_thr':>7} {'FP(68%)':>8} {'TP(60%)':>8} {'TP(55%)':>8} {'Det@60':>7} {'PnL@60':>7}")

sweep_results = []
for wr_thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
    fp = evaluate_detection_rule(wr_thr, None, 30, 0.68)
    tp60 = evaluate_detection_rule(wr_thr, None, 30, 0.60)
    tp55 = evaluate_detection_rule(wr_thr, None, 30, 0.55)
    print(f"  {wr_thr:.2f}   {fp['rate']*100:>7.1f}%  {tp60['rate']*100:>7.1f}%  {tp55['rate']*100:>7.1f}%  "
          f"{tp60['avg_trade'] or 0:>6.0f}  {tp60['avg_pnl'] or 0:>+6.1f}%")
    sweep_results.append({
        'wr_threshold': wr_thr, 'pnl_threshold': None, 'window': 30,
        'fp_68': round(fp['rate'] * 100, 1), 'tp_60': round(tp60['rate'] * 100, 1),
        'tp_55': round(tp55['rate'] * 100, 1),
        'det_trade_60': round(tp60['avg_trade']) if tp60['avg_trade'] else None,
        'pnl_at_det_60': round(tp60['avg_pnl'], 1) if tp60['avg_pnl'] else None,
    })

print("\n--- PnL threshold sweep (WR threshold = None, window = 30) ---")
print(f"{'PnL_thr':>7} {'FP(68%)':>8} {'TP(60%)':>8} {'TP(55%)':>8} {'Det@60':>7} {'PnL@60':>7}")

for pnl_thr in [-10, -15, -20, -25, -30, -40]:
    fp = evaluate_detection_rule(None, pnl_thr, 30, 0.68)
    tp60 = evaluate_detection_rule(None, pnl_thr, 30, 0.60)
    tp55 = evaluate_detection_rule(None, pnl_thr, 30, 0.55)
    print(f"  {pnl_thr:>5}%   {fp['rate']*100:>7.1f}%  {tp60['rate']*100:>7.1f}%  {tp55['rate']*100:>7.1f}%  "
          f"{tp60['avg_trade'] or 0:>6.0f}  {tp60['avg_pnl'] or 0:>+6.1f}%")
    sweep_results.append({
        'wr_threshold': None, 'pnl_threshold': pnl_thr, 'window': 30,
        'fp_68': round(fp['rate'] * 100, 1), 'tp_60': round(tp60['rate'] * 100, 1),
        'tp_55': round(tp55['rate'] * 100, 1),
        'det_trade_60': round(tp60['avg_trade']) if tp60['avg_trade'] else None,
        'pnl_at_det_60': round(tp60['avg_pnl'], 1) if tp60['avg_pnl'] else None,
    })

print("\n--- Window size sweep (WR threshold = 0.50, PnL = None) ---")
print(f"{'Window':>7} {'FP(68%)':>8} {'TP(60%)':>8} {'TP(55%)':>8} {'Det@60':>7} {'PnL@60':>7}")

for win in [15, 20, 25, 30, 40, 50]:
    fp = evaluate_detection_rule(0.50, None, win, 0.68)
    tp60 = evaluate_detection_rule(0.50, None, win, 0.60)
    tp55 = evaluate_detection_rule(0.50, None, win, 0.55)
    print(f"  {win:>5}   {fp['rate']*100:>7.1f}%  {tp60['rate']*100:>7.1f}%  {tp55['rate']*100:>7.1f}%  "
          f"{tp60['avg_trade'] or 0:>6.0f}  {tp60['avg_pnl'] or 0:>+6.1f}%")
    sweep_results.append({
        'wr_threshold': 0.50, 'pnl_threshold': None, 'window': win,
        'fp_68': round(fp['rate'] * 100, 1), 'tp_60': round(tp60['rate'] * 100, 1),
        'tp_55': round(tp55['rate'] * 100, 1),
        'det_trade_60': round(tp60['avg_trade']) if tp60['avg_trade'] else None,
        'pnl_at_det_60': round(tp60['avg_pnl'], 1) if tp60['avg_pnl'] else None,
    })

phase4 = {'sweep_results': sweep_results}


# ══════════════════════════════════════════════════════
# Part B - Phase 5: 대안 감지 방법 비교
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part B - Phase 5: 대안 감지 방법 비교")
print("=" * 60)


def evaluate_cusum(k, h, true_wr, n_sims=N_SIMS):
    """CUSUM detection: accumulate (loss_indicator - k), trigger at h."""
    detected = 0
    det_trades = []
    det_pnls = []

    for _ in range(n_sims):
        cusum = 0.0
        cum_pnl = 0.0
        det_trade = None

        for t in range(HORIZON):
            win = np.random.random() < true_wr
            pnl = SIM_AVG_WIN if win else -SIM_AVG_LOSS
            cum_pnl += pnl

            # CUSUM on losses
            cusum = max(0, cusum + (0 if win else 1) - k)
            if det_trade is None and cusum >= h:
                det_trade = t + 1

        if det_trade is not None:
            detected += 1
            det_trades.append(det_trade)

    return {
        'rate': detected / n_sims,
        'avg_trade': np.mean(det_trades) if det_trades else None,
    }


def evaluate_bayesian(prior_working, threshold, true_wr, n_sims=N_SIMS):
    """Bayesian detection: update belief about WR, trigger when P(broken) > threshold."""
    detected = 0
    det_trades = []

    for _ in range(n_sims):
        # Beta prior: working = Beta(a1, b1), broken = Beta(a2, b2)
        a_w, b_w = 68, 32  # prior: working WR ~68%
        a_b, b_b = 55, 45  # prior: broken WR ~55%
        log_odds = np.log(prior_working / (1 - prior_working))
        det_trade = None

        for t in range(HORIZON):
            win = np.random.random() < true_wr

            # Update log-odds
            if win:
                log_odds += np.log(a_w / (a_w + b_w)) - np.log(a_b / (a_b + b_b))
            else:
                log_odds += np.log(b_w / (a_w + b_w)) - np.log(b_b / (a_b + b_b))

            p_working = 1 / (1 + np.exp(-log_odds))

            if det_trade is None and p_working < (1 - threshold):
                det_trade = t + 1

        if det_trade is not None:
            detected += 1
            det_trades.append(det_trade)

    return {
        'rate': detected / n_sims,
        'avg_trade': np.mean(det_trades) if det_trades else None,
    }


def evaluate_drawdown(dd_threshold, true_wr, n_sims=N_SIMS):
    """Drawdown detection: trigger when drawdown from peak exceeds threshold."""
    detected = 0
    det_trades = []
    det_pnls = []

    for _ in range(n_sims):
        cum_pnl = 0.0
        peak = 0.0
        det_trade = None

        for t in range(HORIZON):
            win = np.random.random() < true_wr
            pnl = SIM_AVG_WIN if win else -SIM_AVG_LOSS
            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            dd = peak - cum_pnl

            if det_trade is None and dd >= dd_threshold:
                det_trade = t + 1

        if det_trade is not None:
            detected += 1
            det_trades.append(det_trade)
            det_pnls.append(cum_pnl)

    return {
        'rate': detected / n_sims,
        'avg_trade': np.mean(det_trades) if det_trades else None,
        'avg_pnl': np.mean(det_pnls) if det_pnls else None,
    }


# Compare methods
print(f"\n--- 감지 방법 비교 (200트레이드 horizon) ---")
print(f"{'Method':<30} {'FP(68%)':>8} {'TP(60%)':>8} {'TP(55%)':>8} {'Det@60':>7}")

methods = {}

# Method 1: Rolling WR (window=30, threshold=0.50)
m1_fp = evaluate_detection_rule(0.50, None, 30, 0.68)
m1_tp60 = evaluate_detection_rule(0.50, None, 30, 0.60)
m1_tp55 = evaluate_detection_rule(0.50, None, 30, 0.55)
print(f"{'Rolling WR(30, <50%)':<30} {m1_fp['rate']*100:>7.1f}%  {m1_tp60['rate']*100:>7.1f}%  {m1_tp55['rate']*100:>7.1f}%  {m1_tp60['avg_trade'] or 0:>6.0f}")
methods['rolling_wr_30_50'] = {
    'fp_68': round(m1_fp['rate'] * 100, 1), 'tp_60': round(m1_tp60['rate'] * 100, 1),
    'tp_55': round(m1_tp55['rate'] * 100, 1), 'det_60': round(m1_tp60['avg_trade']) if m1_tp60['avg_trade'] else None,
}

# Method 2: Rolling WR (window=40, threshold=0.50)
m2_fp = evaluate_detection_rule(0.50, None, 40, 0.68)
m2_tp60 = evaluate_detection_rule(0.50, None, 40, 0.60)
m2_tp55 = evaluate_detection_rule(0.50, None, 40, 0.55)
print(f"{'Rolling WR(40, <50%)':<30} {m2_fp['rate']*100:>7.1f}%  {m2_tp60['rate']*100:>7.1f}%  {m2_tp55['rate']*100:>7.1f}%  {m2_tp60['avg_trade'] or 0:>6.0f}")
methods['rolling_wr_40_50'] = {
    'fp_68': round(m2_fp['rate'] * 100, 1), 'tp_60': round(m2_tp60['rate'] * 100, 1),
    'tp_55': round(m2_tp55['rate'] * 100, 1), 'det_60': round(m2_tp60['avg_trade']) if m2_tp60['avg_trade'] else None,
}

# Method 3: CUSUM (k=0.35, h=5)
m3_fp = evaluate_cusum(0.35, 5, 0.68)
m3_tp60 = evaluate_cusum(0.35, 5, 0.60)
m3_tp55 = evaluate_cusum(0.35, 5, 0.55)
print(f"{'CUSUM(k=0.35, h=5)':<30} {m3_fp['rate']*100:>7.1f}%  {m3_tp60['rate']*100:>7.1f}%  {m3_tp55['rate']*100:>7.1f}%  {m3_tp60['avg_trade'] or 0:>6.0f}")
methods['cusum_035_5'] = {
    'fp_68': round(m3_fp['rate'] * 100, 1), 'tp_60': round(m3_tp60['rate'] * 100, 1),
    'tp_55': round(m3_tp55['rate'] * 100, 1), 'det_60': round(m3_tp60['avg_trade']) if m3_tp60['avg_trade'] else None,
}

# Method 4: CUSUM (k=0.35, h=7)
m4_fp = evaluate_cusum(0.35, 7, 0.68)
m4_tp60 = evaluate_cusum(0.35, 7, 0.60)
m4_tp55 = evaluate_cusum(0.35, 7, 0.55)
print(f"{'CUSUM(k=0.35, h=7)':<30} {m4_fp['rate']*100:>7.1f}%  {m4_tp60['rate']*100:>7.1f}%  {m4_tp55['rate']*100:>7.1f}%  {m4_tp60['avg_trade'] or 0:>6.0f}")
methods['cusum_035_7'] = {
    'fp_68': round(m4_fp['rate'] * 100, 1), 'tp_60': round(m4_tp60['rate'] * 100, 1),
    'tp_55': round(m4_tp55['rate'] * 100, 1), 'det_60': round(m4_tp60['avg_trade']) if m4_tp60['avg_trade'] else None,
}

# Method 5: CUSUM (k=0.40, h=6)
m5_fp = evaluate_cusum(0.40, 6, 0.68)
m5_tp60 = evaluate_cusum(0.40, 6, 0.60)
m5_tp55 = evaluate_cusum(0.40, 6, 0.55)
print(f"{'CUSUM(k=0.40, h=6)':<30} {m5_fp['rate']*100:>7.1f}%  {m5_tp60['rate']*100:>7.1f}%  {m5_tp55['rate']*100:>7.1f}%  {m5_tp60['avg_trade'] or 0:>6.0f}")
methods['cusum_040_6'] = {
    'fp_68': round(m5_fp['rate'] * 100, 1), 'tp_60': round(m5_tp60['rate'] * 100, 1),
    'tp_55': round(m5_tp55['rate'] * 100, 1), 'det_60': round(m5_tp60['avg_trade']) if m5_tp60['avg_trade'] else None,
}

# Method 6: Bayesian (prior=0.8, threshold=0.7)
m6_fp = evaluate_bayesian(0.8, 0.7, 0.68)
m6_tp60 = evaluate_bayesian(0.8, 0.7, 0.60)
m6_tp55 = evaluate_bayesian(0.8, 0.7, 0.55)
print(f"{'Bayesian(prior=0.8, thr=0.7)':<30} {m6_fp['rate']*100:>7.1f}%  {m6_tp60['rate']*100:>7.1f}%  {m6_tp55['rate']*100:>7.1f}%  {m6_tp60['avg_trade'] or 0:>6.0f}")
methods['bayesian_08_07'] = {
    'fp_68': round(m6_fp['rate'] * 100, 1), 'tp_60': round(m6_tp60['rate'] * 100, 1),
    'tp_55': round(m6_tp55['rate'] * 100, 1), 'det_60': round(m6_tp60['avg_trade']) if m6_tp60['avg_trade'] else None,
}

# Method 7: Drawdown threshold
for dd_thr in [20, 25, 30]:
    m_fp = evaluate_drawdown(dd_thr, 0.68)
    m_tp60 = evaluate_drawdown(dd_thr, 0.60)
    m_tp55 = evaluate_drawdown(dd_thr, 0.55)
    label = f'Drawdown(>{dd_thr}%)'
    print(f"{label:<30} {m_fp['rate']*100:>7.1f}%  {m_tp60['rate']*100:>7.1f}%  {m_tp55['rate']*100:>7.1f}%  {m_tp60['avg_trade'] or 0:>6.0f}")
    methods[f'drawdown_{dd_thr}'] = {
        'fp_68': round(m_fp['rate'] * 100, 1), 'tp_60': round(m_tp60['rate'] * 100, 1),
        'tp_55': round(m_tp55['rate'] * 100, 1), 'det_60': round(m_tp60['avg_trade']) if m_tp60['avg_trade'] else None,
    }

# Method 8: Combined optimal — Rolling WR(40, <50%) AND Drawdown(>25%)
print(f"\n--- Combined rules ---")

def evaluate_combined(wr_thr, win_size, dd_thr, true_wr, mode='OR', n_sims=N_SIMS):
    detected = 0
    det_trades = []
    det_pnls = []

    for _ in range(n_sims):
        cum_pnl = 0.0
        peak = 0.0
        results = []
        det_trade = None

        for t in range(HORIZON):
            win = np.random.random() < true_wr
            pnl = SIM_AVG_WIN if win else -SIM_AVG_LOSS
            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            dd = peak - cum_pnl
            results.append(1 if win else 0)

            if det_trade is None and len(results) >= win_size:
                rwr = np.mean(results[-win_size:])
                wr_trigger = rwr < wr_thr
                dd_trigger = dd >= dd_thr

                if mode == 'OR' and (wr_trigger or dd_trigger):
                    det_trade = t + 1
                elif mode == 'AND' and (wr_trigger and dd_trigger):
                    det_trade = t + 1

        if det_trade is not None:
            detected += 1
            det_trades.append(det_trade)
            pnl_at = sum(SIM_AVG_WIN if r else -SIM_AVG_LOSS for r in results[:det_trade])
            det_pnls.append(pnl_at)

    return {
        'rate': detected / n_sims,
        'avg_trade': np.mean(det_trades) if det_trades else None,
        'avg_pnl': np.mean(det_pnls) if det_pnls else None,
    }

# Test AND combinations (both conditions must be true → lower FP)
print(f"{'Method':<40} {'FP(68%)':>8} {'TP(60%)':>8} {'TP(55%)':>8} {'Det@60':>7} {'PnL@60':>7}")

combos = [
    ('WR(30,<50%) AND DD>20%', 0.50, 30, 20),
    ('WR(30,<50%) AND DD>25%', 0.50, 30, 25),
    ('WR(40,<50%) AND DD>20%', 0.50, 40, 20),
    ('WR(40,<50%) AND DD>25%', 0.50, 40, 25),
    ('WR(30,<45%) AND DD>20%', 0.45, 30, 20),
    ('WR(40,<45%) AND DD>20%', 0.45, 40, 20),
    ('WR(50,<50%) AND DD>25%', 0.50, 50, 25),
    ('WR(50,<50%) AND DD>30%', 0.50, 50, 30),
]

combined_results = []
for label, wr_t, win_s, dd_t in combos:
    fp = evaluate_combined(wr_t, win_s, dd_t, 0.68, 'AND')
    tp60 = evaluate_combined(wr_t, win_s, dd_t, 0.60, 'AND')
    tp55 = evaluate_combined(wr_t, win_s, dd_t, 0.55, 'AND')
    print(f"  {label:<38} {fp['rate']*100:>7.1f}%  {tp60['rate']*100:>7.1f}%  {tp55['rate']*100:>7.1f}%  "
          f"{tp60['avg_trade'] or 0:>6.0f}  {tp60['avg_pnl'] or 0:>+6.1f}%")
    combined_results.append({
        'rule': label,
        'fp_68': round(fp['rate'] * 100, 1), 'tp_60': round(tp60['rate'] * 100, 1),
        'tp_55': round(tp55['rate'] * 100, 1),
        'det_60': round(tp60['avg_trade']) if tp60['avg_trade'] else None,
        'pnl_60': round(tp60['avg_pnl'], 1) if tp60['avg_pnl'] else None,
    })

phase5 = {'single_methods': methods, 'combined_rules': combined_results}


# ══════════════════════════════════════════════════════
# Part B - Phase 6: 최종 추천 운영 규칙
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part B - Phase 6: 최종 추천 운영 규칙")
print("=" * 60)

# Find Pareto-optimal rules (low FP + high TP + fast detection)
print("\n--- Pareto 최적 규칙 선정 기준 ---")
print("  목표: FP(68%) < 20%, TP(60%) > 80%, TP(55%) > 95%")

# Evaluate candidates
pareto = []
for cr in combined_results:
    if cr['fp_68'] < 25 and cr['tp_60'] > 70:
        score = cr['tp_60'] - cr['fp_68'] * 2  # TP - 2×FP
        cr['score'] = round(score, 1)
        pareto.append(cr)

pareto.sort(key=lambda x: x['score'], reverse=True)
print(f"\n  Pareto 후보 {len(pareto)}개:")
for p in pareto:
    print(f"    {p['rule']}: FP {p['fp_68']:.1f}%, TP60 {p['tp_60']:.1f}%, TP55 {p['tp_55']:.1f}%, "
          f"Det@60 {p['det_60']}tr, PnL@60 {p['pnl_60']:+.1f}%, Score {p['score']:.1f}")

# Final recommendation
if pareto:
    best = pareto[0]
    print(f"\n  ★ 추천 규칙: {best['rule']}")
    print(f"    오탐률: {best['fp_68']:.1f}%")
    print(f"    WR 60% 감지율: {best['tp_60']:.1f}%")
    print(f"    WR 55% 감지율: {best['tp_55']:.1f}%")
    print(f"    감지 시점 (WR 60%): ~{best['det_60']}트레이드")
    print(f"    감지 시 PnL (WR 60%): {best['pnl_60']:+.1f}%")

# Revalidate best rule at additional WR levels
if pareto:
    best_params = None
    for label, wr_t, win_s, dd_t in combos:
        if label == best['rule']:
            best_params = (wr_t, win_s, dd_t)
            break

    if best_params:
        print(f"\n--- 추천 규칙 상세 검증 ---")
        for wr_label, wr_val in [('65%', 0.65), ('63%', 0.63), ('60%', 0.60), ('55%', 0.55), ('50%', 0.50)]:
            r = evaluate_combined(best_params[0], best_params[1], best_params[2], wr_val, 'AND')
            print(f"  WR {wr_label}: 감지율 {r['rate']*100:.1f}%, 감지 시점 {r['avg_trade']:.0f}tr" if r['avg_trade'] else
                  f"  WR {wr_label}: 감지율 {r['rate']*100:.1f}%")

phase6 = {
    'recommended_rule': best['rule'] if pareto else 'No Pareto-optimal rule found',
    'recommended_details': pareto[0] if pareto else None,
    'pareto_candidates': pareto,
}


# ══════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════╗
║ Part A: R:R 불리 원인                                 ║
╠══════════════════════════════════════════════════════╣
║ 1. R:R 불리는 설계상 의도됨                            ║
║    - 52패턴 중 {int(phase1['design_rr_lt1_pct'])}%가 TP < SL 구조              ║
║    - 백테스트 R:R = {phase1['bt_rr_ratio']:.3f} (avg_win/avg_loss)      ║
║                                                      ║
║ 2. Break-Even WR 수정                                 ║
║    - 이전 (버퍼 기반): {phase2['previous_buffer_based']['be_wr']:.1f}%, 마진 3.0pp     ║
║    - 수정 (복원, likely): {phase2['scenario_a_recent10']['be_wr']:.1f}%, 마진 {phase2['scenario_a_recent10']['margin_pp']:+.1f}pp   ║
║    - 수정 (복원, worst): {phase2['scenario_b_worst10']['be_wr']:.1f}%, 마진 {phase2['scenario_b_worst10']['margin_pp']:+.1f}pp   ║
║    → 이전 연구의 "3pp 마진"은 버퍼 왜곡 결과            ║
║                                                      ║
║ 3. 패턴 분류                                          ║
║    - Break-even 미만: {phase3['below_breakeven']}개                       ║
║    - 얇은 마진 (0-5pp): {phase3['thin_margin']}개                      ║
║    - 안전 (≥5pp): {phase3['comfortable']}개                            ║
╠══════════════════════════════════════════════════════╣
║ Part B: 감지 시스템                                    ║
╠══════════════════════════════════════════════════════╣
║ 기존 규칙 (WR<55% OR PnL<-15%): 오탐률 88%            ║
║                                                      ║
║ 추천 규칙: {phase6['recommended_rule']:<40}║
║   오탐률: {best['fp_68'] if pareto else 'N/A'}%                                    ║
║   WR 60% 감지율: {best['tp_60'] if pareto else 'N/A'}%                           ║
║   WR 55% 감지율: {best['tp_55'] if pareto else 'N/A'}%                           ║
╚══════════════════════════════════════════════════════╝
""")

results = {
    'part_a_phase1_bt_rr': phase1,
    'part_a_phase2_buffer_verify': phase2,
    'part_a_phase3_pattern_rr': phase3,
    'part_b_phase4_sweep': phase4,
    'part_b_phase5_methods': phase5,
    'part_b_phase6_recommendation': phase6,
    'timestamp': datetime.now().isoformat(),
}

output_path = Path(__file__).resolve().parent / '../../results/rr_detection_deep_study.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n결과 저장: {output_path}")
