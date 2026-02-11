"""
future_risk_stress_test.py — 미래 리스크 스트레스 테스트 & 전략 수명 분석

핵심 질문: "미래에 전혀 다른 패턴으로 손실을 입을 수 있는가?"
답: 확률적으로 YES. 문제는 "언제 감지하고 얼마나 잃기 전에 멈출 수 있는가?"

Phase 1: Live vs Backtest Reality Gap — 라이브 31트레이드가 말해주는 진실
Phase 2: Break-Even Margin Analysis — 수익/손실 경계까지의 거리
Phase 3: Strategy Death Detection — 실시간 감지 시스템 설계
Phase 4: Forward Monte Carlo — 현실적 파라미터 기반 미래 시뮬레이션
Phase 5: Edge Decay & Strategy Lifecycle — 전략 수명 추정
Phase 6: Adversarial Stress Tests — 최악 시나리오 테스트

사용 데이터: 라이브 메트릭 + 270일 백테스트 데이터
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

# ─── Parameters ───
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
TP_SCALE = 0.70

np.random.seed(42)

# ─── Live metrics (from pattern_5m_metrics.json) ───
LIVE_TRADES = 31
LIVE_WINS = 21
LIVE_LOSSES = 10
LIVE_WR = 67.74
LIVE_AVG_WIN = 3.094    # leveraged PnL %
LIVE_AVG_LOSS = 5.690   # leveraged PnL %
LIVE_TOTAL_PNL = 22.02

# ─── Data Loading ───
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

# Build candle types
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

# All patterns
all_patterns = {}
for p in VALIDATED_LONG_PATTERNS:
    all_patterns[p] = 'LONG'
for p in VALIDATED_SHORT_PATTERNS:
    all_patterns[p] = 'SHORT'

print(f"Data: {n_bars} bars, Patterns: {len(all_patterns)}")


def simulate_trade(idx, direction, tp_pct, sl_pct, max_bars=MAX_BARS):
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
            pnl = -sl_pct * LEVERAGE - FEE_PCT
            return {'win': False, 'pnl': pnl, 'bar': idx, 'exit_bar': j}
        elif tp_hit:
            pnl = tp_pct * LEVERAGE * 0.9 - FEE_PCT
            return {'win': True, 'pnl': pnl, 'bar': idx, 'exit_bar': j}
        elif sl_hit:
            pnl = -sl_pct * LEVERAGE - FEE_PCT
            return {'win': False, 'pnl': pnl, 'bar': idx, 'exit_bar': j}
    last_j = min(idx + max_bars, n_bars - 1)
    if direction == 'LONG':
        pnl = (closes[last_j] - entry) / entry * 100 * LEVERAGE - FEE_PCT
    else:
        pnl = (entry - closes[last_j]) / entry * 100 * LEVERAGE - FEE_PCT
    return {'win': pnl > 0, 'pnl': pnl, 'bar': idx, 'exit_bar': last_j}


def run_backtest(pattern_set=None, start_bar=0, end_bar=None):
    if pattern_set is None:
        pattern_set = all_patterns
    if end_bar is None:
        end_bar = n_bars
    trades = []
    in_position_until = 0
    for i in range(max(start_bar, 2), end_bar):
        if i < in_position_until:
            continue
        pat = patterns_at[i]
        if pat and pat in pattern_set:
            direction = pattern_set[pat]
            base_tp, base_sl = PATTERN_OPTIMAL_TPSL.get(pat, (2.0, 3.0))
            tp = base_tp * TP_SCALE
            sl = base_sl
            result = simulate_trade(i, direction, tp, sl)
            if result:
                result['pattern'] = pat
                result['direction'] = direction
                trades.append(result)
                in_position_until = result['exit_bar'] + 1
    return trades


# ══════════════════════════════════════════════════════
# Phase 1: Live vs Backtest Reality Gap
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: Live vs Backtest Reality Gap")
print("=" * 60)

bt_trades = run_backtest()
bt_wins = [t for t in bt_trades if t['win']]
bt_losses = [t for t in bt_trades if not t['win']]
bt_wr = len(bt_wins) / len(bt_trades) * 100
bt_avg_win = np.mean([t['pnl'] for t in bt_wins])
bt_avg_loss = np.mean([abs(t['pnl']) for t in bt_losses])
bt_total_pnl = sum(t['pnl'] for t in bt_trades)

print(f"\nBacktest: {len(bt_trades)} trades, WR {bt_wr:.1f}%, avg_win {bt_avg_win:.2f}%, avg_loss {bt_avg_loss:.2f}%, PnL {bt_total_pnl:+.1f}%")
print(f"Live:     {LIVE_TRADES} trades, WR {LIVE_WR:.1f}%, avg_win {LIVE_AVG_WIN:.2f}%, avg_loss {LIVE_AVG_LOSS:.2f}%, PnL {LIVE_TOTAL_PNL:+.1f}%")
print(f"Gap:      WR {bt_wr - LIVE_WR:+.1f}pp")

# Binomial test
binom_p = stats.binom.cdf(LIVE_WINS, LIVE_TRADES, bt_wr / 100)
print(f"\nBinomial test: P(≤{LIVE_WINS} wins | n={LIVE_TRADES}, p={bt_wr / 100:.3f}) = {binom_p:.6f}")
if binom_p < 0.05:
    print("  → SIGNIFICANT (p<0.05): 라이브 WR이 백테스트보다 통계적으로 낮음")
else:
    print("  → 유의하지 않음 (표본 작음), 하지만 gap 주목 필요")

# Wilson CI for live WR (manual calculation)
z = 1.96
p_hat = LIVE_WINS / LIVE_TRADES
denom = 1 + z ** 2 / LIVE_TRADES
center = (p_hat + z ** 2 / (2 * LIVE_TRADES)) / denom
margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * LIVE_TRADES)) / LIVE_TRADES) / denom
live_ci = (center - margin, center + margin)
print(f"Live WR 95% CI: [{live_ci[0] * 100:.1f}%, {live_ci[1] * 100:.1f}%]")

# Sample size needed to distinguish live WR from break-even
be_wr = LIVE_AVG_LOSS / (LIVE_AVG_WIN + LIVE_AVG_LOSS)
diff = LIVE_WR / 100 - be_wr
if diff > 0:
    z_a, z_b = 1.645, 0.842
    p_hat = LIVE_WR / 100
    n_needed = (z_a + z_b) ** 2 * p_hat * (1 - p_hat) / diff ** 2
    bt_trades_per_month = len(bt_trades) / 9
    months_needed = n_needed / bt_trades_per_month
    print(f"\n통계적 검정력 분석:")
    print(f"  H0: WR = {be_wr * 100:.1f}% (break-even) vs H1: WR = {LIVE_WR:.1f}% (live)")
    print(f"  80% power @ 5% significance 필요 표본: {n_needed:.0f} trades")
    print(f"  예상 소요: {months_needed:.0f}개월 ({n_needed:.0f} / {bt_trades_per_month:.0f} trades/month)")
else:
    n_needed = float('inf')
    months_needed = float('inf')

phase1 = {
    'backtest': {'trades': len(bt_trades), 'wr': round(bt_wr, 2), 'avg_win': round(bt_avg_win, 3),
                 'avg_loss': round(bt_avg_loss, 3), 'pnl': round(bt_total_pnl, 1)},
    'live': {'trades': LIVE_TRADES, 'wr': LIVE_WR, 'avg_win': LIVE_AVG_WIN, 'avg_loss': LIVE_AVG_LOSS,
             'pnl': LIVE_TOTAL_PNL},
    'gap_wr_pp': round(bt_wr - LIVE_WR, 2),
    'binom_p': round(binom_p, 6),
    'live_wr_ci_95': [round(live_ci[0] * 100, 1), round(live_ci[1] * 100, 1)],
    'n_trades_for_statistical_power': round(n_needed),
    'months_needed': round(months_needed),
}


# ══════════════════════════════════════════════════════
# Phase 2: Break-Even Margin Analysis
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Break-Even Margin Analysis")
print("=" * 60)

break_even_wr = LIVE_AVG_LOSS / (LIVE_AVG_WIN + LIVE_AVG_LOSS) * 100
margin_live = LIVE_WR - break_even_wr
margin_bt = bt_wr - break_even_wr

print(f"\nBreak-even WR (live R:R 기준): {break_even_wr:.1f}%")
print(f"라이브 WR 마진: {margin_live:+.1f}pp (= {LIVE_WR:.1f}% - {break_even_wr:.1f}%)")
print(f"백테스트 WR 마진: {margin_bt:+.1f}pp")

# PnL per trade at various WR levels
print(f"\n--- WR별 기대 PnL (live R:R = {LIVE_AVG_WIN:.2f}/{LIVE_AVG_LOSS:.2f}) ---")
wr_levels = [45, 50, 55, 60, break_even_wr, LIVE_WR, 70, 75, 80, bt_wr]
scenarios = []
for wr in sorted(set([round(w, 1) for w in wr_levels])):
    exp = LIVE_AVG_WIN * (wr / 100) - LIVE_AVG_LOSS * (1 - wr / 100)
    exp_100 = exp * 100
    label = ""
    if abs(wr - LIVE_WR) < 0.5:
        label = " ← LIVE"
    elif abs(wr - bt_wr) < 0.5:
        label = " ← BACKTEST"
    elif abs(wr - break_even_wr) < 0.5:
        label = " ← BREAK-EVEN"
    print(f"  WR {wr:5.1f}%: {exp:+.3f}%/trade → {exp_100:+.1f}%/100trades{label}")
    scenarios.append({'wr': round(wr, 1), 'pnl_per_trade': round(exp, 4), 'pnl_100': round(exp_100, 1)})

# Consecutive losses to wipe current profit
trades_to_wipe = LIVE_TOTAL_PNL / LIVE_AVG_LOSS
print(f"\n현재 이익 {LIVE_TOTAL_PNL:.1f}% 소진까지: 연속 {trades_to_wipe:.1f}회 손실")

# PnL per trade using actual data (cross-check)
actual_pnl_per_trade = LIVE_TOTAL_PNL / LIVE_TRADES
print(f"\n실측 PnL/trade: {actual_pnl_per_trade:.3f}% (공식 계산: {LIVE_AVG_WIN * LIVE_WR / 100 - LIVE_AVG_LOSS * (1 - LIVE_WR / 100):.3f}%)")
print(f"→ 차이는 recent_wins/losses 버퍼가 전체 31트레이드를 완전히 대표하지 않기 때문")

phase2 = {
    'break_even_wr': round(break_even_wr, 2),
    'live_margin_pp': round(margin_live, 2),
    'bt_margin_pp': round(margin_bt, 2),
    'scenarios': scenarios,
    'trades_to_wipe_profit': round(trades_to_wipe, 1),
    'actual_pnl_per_trade': round(actual_pnl_per_trade, 4),
}


# ══════════════════════════════════════════════════════
# Phase 3: Strategy Death Detection
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 3: Strategy Failure Detection System")
print("=" * 60)

# Detection rule: rolling 20-trade WR < 55% OR cumulative PnL < -15%
DETECT_WR_THRESHOLD = 0.55
DETECT_PNL_THRESHOLD = -15
ROLLING_WINDOW = 20
N_SIMS = 10000
HORIZON = 200

print(f"\n감지 규칙: 최근 {ROLLING_WINDOW}트레이드 WR < {DETECT_WR_THRESHOLD * 100:.0f}% OR 누적 PnL < {DETECT_PNL_THRESHOLD}%")

detection_results = {}
for label, true_wr in [('working_68', 0.68), ('mild_63', 0.63), ('degraded_60', 0.60),
                        ('broken_55', 0.55), ('dead_50', 0.50), ('severe_45', 0.45)]:
    detected = 0
    detection_trades = []
    pnl_at_detection = []
    final_pnls = []

    for _ in range(N_SIMS):
        cum_pnl = 0.0
        results = []
        det_trade = None

        for t in range(HORIZON):
            win = np.random.random() < true_wr
            pnl = LIVE_AVG_WIN if win else -LIVE_AVG_LOSS
            cum_pnl += pnl
            results.append(1 if win else 0)

            if det_trade is None and len(results) >= ROLLING_WINDOW:
                rwr = np.mean(results[-ROLLING_WINDOW:])
                if rwr < DETECT_WR_THRESHOLD or cum_pnl < DETECT_PNL_THRESHOLD:
                    det_trade = t + 1

        if det_trade is not None:
            detected += 1
            detection_trades.append(det_trade)
            # Approximate PnL at detection
            pnl_approx = sum(LIVE_AVG_WIN if r else -LIVE_AVG_LOSS for r in results[:det_trade])
            pnl_at_detection.append(pnl_approx)

        final_pnls.append(cum_pnl)

    det_rate = detected / N_SIMS * 100
    avg_det = np.mean(detection_trades) if detection_trades else None
    avg_pnl_det = np.mean(pnl_at_detection) if pnl_at_detection else None
    mean_final = np.mean(final_pnls)

    print(f"\n  True WR {true_wr * 100:.0f}%:")
    print(f"    감지율 ({HORIZON}트레이드): {det_rate:.1f}%")
    if avg_det:
        print(f"    평균 감지 시점: {avg_det:.0f}번째 트레이드")
    if avg_pnl_det is not None:
        print(f"    감지 시 평균 PnL: {avg_pnl_det:.1f}%")
    print(f"    {HORIZON}트레이드 최종 PnL: {mean_final:+.1f}%")

    detection_results[label] = {
        'true_wr': true_wr * 100,
        'detection_rate': round(det_rate, 1),
        'avg_detection_trade': round(avg_det) if avg_det else None,
        'avg_pnl_at_detection': round(avg_pnl_det, 1) if avg_pnl_det is not None else None,
        'mean_final_pnl': round(mean_final, 1),
    }

# False positive rate (strategy actually works but triggers alarm)
fp_rate = detection_results['working_68']['detection_rate']
print(f"\n⚠ 오탐률 (WR 68%에서 감지): {fp_rate:.1f}%")

phase3 = detection_results


# ══════════════════════════════════════════════════════
# Phase 4: Forward Monte Carlo (Realistic Parameters)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 4: Forward Monte Carlo (Live Parameters)")
print("=" * 60)

N_MC = 10000
horizons = [50, 100, 200, 500]

mc_results = {}
for hz in horizons:
    pnls = []
    mdds = []
    neg_count = 0

    for _ in range(N_MC):
        eq = [0.0]
        peak = 0.0
        mdd = 0.0
        for t in range(hz):
            win = np.random.random() < (LIVE_WR / 100)
            eq.append(eq[-1] + (LIVE_AVG_WIN if win else -LIVE_AVG_LOSS))
            if eq[-1] > peak:
                peak = eq[-1]
            dd = peak - eq[-1]
            if dd > mdd:
                mdd = dd
        pnls.append(eq[-1])
        mdds.append(mdd)
        if eq[-1] < 0:
            neg_count += 1

    pa = np.array(pnls)
    ma = np.array(mdds)

    r = {
        'horizon': hz,
        'mean_pnl': round(float(np.mean(pa)), 1),
        'median_pnl': round(float(np.median(pa)), 1),
        'std_pnl': round(float(np.std(pa)), 1),
        'p5_pnl': round(float(np.percentile(pa, 5)), 1),
        'p95_pnl': round(float(np.percentile(pa, 95)), 1),
        'prob_negative': round(neg_count / N_MC * 100, 1),
        'prob_loss_gt_20': round(float(np.mean(pa < -20)) * 100, 1),
        'prob_loss_gt_50': round(float(np.mean(pa < -50)) * 100, 1),
        'mean_mdd': round(float(np.mean(ma)), 1),
        'p95_mdd': round(float(np.percentile(ma, 95)), 1),
        'worst_pnl': round(float(np.min(pa)), 1),
    }
    mc_results[str(hz)] = r

    print(f"\n  {hz} trades (live WR {LIVE_WR:.1f}%, R:R {LIVE_AVG_WIN:.1f}/{LIVE_AVG_LOSS:.1f}):")
    print(f"    PnL: {r['mean_pnl']:+.1f}% ± {r['std_pnl']:.1f}% (median {r['median_pnl']:+.1f}%)")
    print(f"    5th-95th: [{r['p5_pnl']:+.1f}%, {r['p95_pnl']:+.1f}%]")
    print(f"    P(손실): {r['prob_negative']:.1f}%, P(>20%손실): {r['prob_loss_gt_20']:.1f}%")
    print(f"    MDD 평균: {r['mean_mdd']:.1f}%, p95: {r['p95_mdd']:.1f}%")
    print(f"    최악 경로: {r['worst_pnl']:+.1f}%")

phase4 = mc_results


# ══════════════════════════════════════════════════════
# Phase 5: Edge Decay & Strategy Lifecycle
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 5: Edge Decay & Strategy Lifecycle")
print("=" * 60)

live_edge_per_trade = LIVE_AVG_WIN * (LIVE_WR / 100) - LIVE_AVG_LOSS * (1 - LIVE_WR / 100)
print(f"\n라이브 엣지/트레이드: {live_edge_per_trade:.3f}% (buffer 기반)")
print(f"실측 엣지/트레이드: {actual_pnl_per_trade:.3f}% (total_pnl/trades)")

# Degradation factor
degradation = LIVE_WR / bt_wr
print(f"\nDegradation factor (live_WR/bt_WR): {degradation:.3f}")
print(f"→ 백테스트 대비 {(1 - degradation) * 100:.1f}% 성능 감소 (in-sample → live)")

# Edge decay scenarios
bt_trades_per_month = len(bt_trades) / 9
print(f"\n예상 트레이드 빈도: ~{bt_trades_per_month:.0f} trades/month")
print(f"\n--- WR 감소 시나리오 (live R:R 기준) ---")

decay_scenarios = {}
for decay_pp_month in [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    if decay_pp_month > 0:
        months_to_be = margin_live / decay_pp_month
        # Cumulative PnL until break-even
        cum_pnl = 0
        for m in range(int(months_to_be) + 1):
            wr_m = max(LIVE_WR - decay_pp_month * m, 30) / 100
            monthly_edge = LIVE_AVG_WIN * wr_m - LIVE_AVG_LOSS * (1 - wr_m)
            cum_pnl += monthly_edge * bt_trades_per_month
    else:
        months_to_be = float('inf')
        cum_pnl = live_edge_per_trade * bt_trades_per_month * 24

    print(f"  Decay {decay_pp_month:.1f}pp/month: break-even까지 {months_to_be:.1f}개월" if months_to_be < 100
          else f"  Decay {decay_pp_month:.1f}pp/month: break-even 도달 안 함")

    decay_scenarios[f"{decay_pp_month}pp"] = {
        'decay_pp_per_month': decay_pp_month,
        'months_to_breakeven': round(months_to_be, 1) if months_to_be < 1000 else None,
    }

phase5 = {
    'live_edge_per_trade': round(live_edge_per_trade, 4),
    'actual_edge_per_trade': round(actual_pnl_per_trade, 4),
    'break_even_wr': round(break_even_wr, 2),
    'margin_pp': round(margin_live, 2),
    'degradation_factor': round(degradation, 4),
    'bt_trades_per_month': round(bt_trades_per_month, 0),
    'decay_scenarios': decay_scenarios,
    'n_trades_for_power': round(n_needed),
    'months_for_power': round(months_needed),
}


# ══════════════════════════════════════════════════════
# Phase 6: Adversarial Stress Tests
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 6: Adversarial Stress Tests")
print("=" * 60)

# Test 1: Reverse all directions
print("\nTest 1: 방향 반전 (LONG↔SHORT)")
reverse_patterns = {p: ('SHORT' if d == 'LONG' else 'LONG') for p, d in all_patterns.items()}
rev_trades = run_backtest(reverse_patterns)
rev_pnl = sum(t['pnl'] for t in rev_trades)
rev_wr = len([t for t in rev_trades if t['win']]) / len(rev_trades) * 100
print(f"  원본: {bt_total_pnl:+.1f}%, 반전: {rev_pnl:+.1f}%")
print(f"  → 방향 엣지: {bt_total_pnl - rev_pnl:+.1f}%")

# Test 2: Random direction assignment
print("\nTest 2: 랜덤 방향 (200 trials)")
random_pnls = []
for _ in range(200):
    rp = {p: ('LONG' if np.random.random() < 0.5 else 'SHORT') for p in all_patterns}
    rt = run_backtest(rp)
    random_pnls.append(sum(t['pnl'] for t in rt))

p_random_better = np.mean([r >= bt_total_pnl for r in random_pnls])
print(f"  랜덤 PnL: {np.mean(random_pnls):+.1f}% ± {np.std(random_pnls):.1f}%")
print(f"  P(랜덤 >= 원본): {p_random_better:.4f}")
print(f"  → 방향 선택의 가치: {bt_total_pnl - np.mean(random_pnls):+.1f}%")

# Test 3: Block-shuffled temporal order (50 trials)
print("\nTest 3: 시간 블록 셔플 (30일 블록, 50 trials)")
block_size = 8640  # 30 days
n_blocks = n_bars // block_size
shuffle_pnls = []

for trial in range(50):
    blocks = list(range(n_blocks))
    np.random.shuffle(blocks)

    sh = np.concatenate([highs[b * block_size:(b + 1) * block_size] for b in blocks])
    sl_arr = np.concatenate([lows[b * block_size:(b + 1) * block_size] for b in blocks])
    so = np.concatenate([opens[b * block_size:(b + 1) * block_size] for b in blocks])
    sc = np.concatenate([closes[b * block_size:(b + 1) * block_size] for b in blocks])
    sn = len(sh)

    s_body = np.abs(sc - so)
    s_avg = pd.Series(s_body).rolling(20).mean().values

    s_types = []
    for i in range(sn):
        row = pd.Series({'open': so[i], 'high': sh[i], 'low': sl_arr[i], 'close': sc[i]})
        ab = s_avg[i] if not pd.isna(s_avg[i]) else 1.0
        s_types.append(classify_candle(row, ab).value)

    s_pats = [None] * sn
    for i in range(2, sn):
        s_pats[i] = f"{s_types[i - 2]}-{s_types[i - 1]}-{s_types[i]}"

    # Simulate trades on shuffled data
    s_trades_pnl = []
    in_pos = 0
    for i in range(2, sn):
        if i < in_pos:
            continue
        pat = s_pats[i]
        if pat and pat in all_patterns:
            direction = all_patterns[pat]
            btp, bsl = PATTERN_OPTIMAL_TPSL.get(pat, (2.0, 3.0))
            tp = btp * TP_SCALE
            sl_val = bsl
            if i + 1 >= sn:
                continue
            entry = so[i + 1]
            if entry <= 0:
                continue
            for j in range(i + 1, min(i + 1 + MAX_BARS, sn)):
                if direction == 'LONG':
                    tp_h = (sh[j] - entry) / entry * 100 >= tp
                    sl_h = (entry - sl_arr[j]) / entry * 100 >= sl_val
                else:
                    tp_h = (entry - sl_arr[j]) / entry * 100 >= tp
                    sl_h = (sh[j] - entry) / entry * 100 >= sl_val
                if tp_h and sl_h:
                    s_trades_pnl.append(-sl_val * LEVERAGE - FEE_PCT)
                    in_pos = j + 1
                    break
                elif tp_h:
                    s_trades_pnl.append(tp * LEVERAGE * 0.9 - FEE_PCT)
                    in_pos = j + 1
                    break
                elif sl_h:
                    s_trades_pnl.append(-sl_val * LEVERAGE - FEE_PCT)
                    in_pos = j + 1
                    break

    shuffle_pnls.append(sum(s_trades_pnl))

shuf_mean = np.mean(shuffle_pnls)
shuf_std = np.std(shuffle_pnls)
print(f"  셔플 PnL: {shuf_mean:+.1f}% ± {shuf_std:.1f}%")
print(f"  원본: {bt_total_pnl:+.1f}%")
print(f"  → 시간 순서 의존도: {bt_total_pnl - shuf_mean:+.1f}%")

# Test 4: What if only 60% of patterns work? (random 40% fail)
print("\nTest 4: 40% 패턴 실패 시나리오 (100 trials)")
partial_pnls = []
for _ in range(100):
    keep = np.random.choice(list(all_patterns.keys()),
                            size=int(len(all_patterns) * 0.6), replace=False)
    keep_set = {p: all_patterns[p] for p in keep}
    pt = run_backtest(keep_set)
    partial_pnls.append(sum(t['pnl'] for t in pt))

print(f"  60% 패턴만 사용: {np.mean(partial_pnls):+.1f}% ± {np.std(partial_pnls):.1f}%")
print(f"  P(손실): {np.mean([p < 0 for p in partial_pnls]) * 100:.1f}%")

phase6 = {
    'reversed_directions': {
        'pnl': round(rev_pnl, 1), 'wr': round(rev_wr, 1),
        'direction_edge': round(bt_total_pnl - rev_pnl, 1),
    },
    'random_directions': {
        'mean_pnl': round(np.mean(random_pnls), 1), 'std_pnl': round(np.std(random_pnls), 1),
        'p_better_than_original': round(p_random_better, 4),
        'direction_value': round(bt_total_pnl - np.mean(random_pnls), 1),
    },
    'time_shuffled': {
        'mean_pnl': round(shuf_mean, 1), 'std_pnl': round(shuf_std, 1),
        'temporal_order_impact': round(bt_total_pnl - shuf_mean, 1),
    },
    'partial_failure_60pct': {
        'mean_pnl': round(np.mean(partial_pnls), 1), 'std_pnl': round(np.std(partial_pnls), 1),
        'prob_loss': round(np.mean([p < 0 for p in partial_pnls]) * 100, 1),
    },
}


# ══════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY: 미래 리스크 핵심 수치")
print("=" * 60)

print(f"""
┌──────────────────────────────────────────────────┐
│ 🚨 핵심 발견                                       │
├──────────────────────────────────────────────────┤
│ 1. 라이브 WR: {LIVE_WR:.1f}% (백테스트 {bt_wr:.1f}%)          │
│ 2. Break-even WR: {break_even_wr:.1f}%                      │
│ 3. 안전 마진: {margin_live:.1f}pp                             │
│ 4. 통계적 검증 필요 표본: {n_needed:.0f} trades ({months_needed:.0f}개월)│
│ 5. 현재 이익 소진: 연속 {trades_to_wipe:.0f}회 손실로 가능     │
├──────────────────────────────────────────────────┤
│ 📊 Forward MC (live params, 100 trades)             │
│    평균 PnL: {mc_results['100']['mean_pnl']:+.1f}%, P(손실): {mc_results['100']['prob_negative']:.1f}%           │
│    95% MDD: {mc_results['100']['p95_mdd']:.1f}%, 최악: {mc_results['100']['worst_pnl']:+.1f}%        │
├──────────────────────────────────────────────────┤
│ 🛡️ 전략 실패 감지                                   │
│    WR 60%→{detection_results.get('degraded_60', {}).get('avg_detection_trade', 'N/A')}트레이드, WR 55%→{detection_results.get('broken_55', {}).get('avg_detection_trade', 'N/A')}트레이드│
│    오탐률 (WR 68%): {fp_rate:.1f}%                    │
└──────────────────────────────────────────────────┘
""")

results = {
    'phase1_reality_gap': phase1,
    'phase2_break_even_margin': phase2,
    'phase3_detection': phase3,
    'phase4_forward_mc': phase4,
    'phase5_edge_decay': phase5,
    'phase6_adversarial': phase6,
    'timestamp': datetime.now().isoformat(),
}

output_path = Path(__file__).resolve().parent / '../../results/future_risk_stress_test.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n결과 저장: {output_path}")
