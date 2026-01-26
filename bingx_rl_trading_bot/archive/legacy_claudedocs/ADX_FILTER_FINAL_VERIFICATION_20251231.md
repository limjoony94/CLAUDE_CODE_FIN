# ADX≥25 Filter Final Verification Report

**검증 날짜**: 2025-12-31
**검증 요청**: "추가검증 이후에 적용하려고 합니다. 철저하게 검증 바랍니다"
**대상 전략**: SuperTrend 5m Bot v1.3 → v1.4 (ADX≥25 Filter 추가 검토)

---

## Executive Summary

### 🔴🔴 최종 결론: SuperTrend 전략 자체가 **신호 품질 미달**

> **⚠️ CRITICAL UPDATE (2026-01-01)**: 이중 검증 (Dual Verification) 결과, **모든 ADX 설정이 검증 실패**
> - Type 1 (신호 품질): 모든 설정 승률 < 50% → **FAIL**
> - Type 2 (실제 거래): ADX≥30만 수익 → ADX≥30만 PASS
> - **결합 결과**: 두 검증 모두 통과하는 설정 = **없음**

| 검증 유형 | 설명 | 통과 기준 | 결과 |
|----------|------|----------|------|
| **Type 1: Signal Quality** | 모든 신호의 승률/기대값 | WR ≥ 50%, EV > 0 | ❌ 전체 FAIL (43.9%~47.0%) |
| **Type 2: Actual Trading** | 실제 거래 시뮬레이션 | PnL > 0, WF ≥ 50% | ⚠️ ADX≥30만 PASS |
| **Combined** | 두 검증 모두 통과 | 양쪽 PASS | ❌ **없음** |

**핵심 발견**: SuperTrend 신호 자체의 품질이 50% 미만으로, ADX 필터와 관계없이 **전략 자체가 근본적 한계**를 가짐.

---

### 이전 검증 결과 요약 (참고용)

| 검증 항목 | 결과 | 판정 |
|----------|------|------|
| Monte Carlo (10,000회) | Baseline $451 vs ADX≥25 $219 | ❌ Baseline 우수 |
| Walk-Forward (6 Windows) | Baseline $115.58 vs ADX≥25 $79.16 | ❌ Baseline 우수 |
| ADX가 Baseline 이길 확률 | **2.4%** | ❌ 통계적 열위 |
| Per-trade Quality | ADX≥25 +0.010%p 우수 | ✅ 거래당 품질 향상 |
| 총 수익성 | ADX≥25 **-51.4%** 감소 | ❌ 전체 수익 감소 |

**이전 결론 (수정됨)**: ~~ADX≥25는 거래당 품질을 미세하게 개선하지만~~ → **이중 검증 결과 전략 자체가 신호 품질 미달**

---

## 1. 검증 배경

### 1.1 원래 연구 결과 (SUPERTREND_ADX_MTF_RESEARCH_20251231.md)

| Config | Full PnL | WF 일관성 | WF PnL |
|--------|----------|-----------|--------|
| Baseline (No ADX) | +14.8% | 2/6 (33%) | $9.18 |
| **ADX≥25** | +17.5% | **4/6 (67%)** | **$17.48** |

**원래 결론**: ADX≥25 Filter 권장 (WF 일관성 +34%p, WF PnL +90%)

### 1.2 검증 필요성

1. **SuperTrend 구현 불일치** 발견 (프로덕션 vs 연구)
2. **Entry 타이밍 버그** 발견 (Look-Ahead Bias)
3. **Scale-out Exit 카운팅** 이슈 (거래 수 인플레이션)

---

## 2. 발견된 Critical Issues

### 2.1 🔴 SuperTrend 구현 불일치 (81.4% 차이)

**비교 스크립트**: `supertrend_implementation_compare.py`

| 항목 | 프로덕션 (Simple) | 연구 (Standard) | 차이 |
|------|------------------|-----------------|------|
| 방향 일치율 | - | - | **58.08%** (41.92% 불일치) |
| 신호 수 | 221 | 1,189 | **-81.4%** |
| Bullish 신호 | 111 | 594 | -81.3% |
| Bearish 신호 | 110 | 595 | -81.5% |

**원인**:
- **프로덕션**: Basic bands만 사용 (ratcheting 없음)
- **연구**: Final bands with ratcheting (업계 표준)

**영향**: 프로덕션 봇이 연구에서 발견한 968개 신호를 놓치고 있음

### 2.2 🔴 Entry 타이밍 버그 (Look-Ahead Bias)

**문제**: 두 스크립트 모두 Entry 타이밍에 Look-Ahead Bias 존재

```
Original Research Logic (잘못됨):
- Signal: bar[i] direction change
- Entry: bar[i] close (!)  ← 신호 발생 시점 종가로 진입 (불가능)
- Exit: bar[j] close

Correct Logic:
- Signal: bar[i] direction change
- Entry: bar[i+1] open (!)  ← 다음 봉 시가로 진입 (현실적)
- Exit: TP/SL at high/low
```

### 2.3 Scale-out Exit 카운팅 이슈

**문제**: Scale-out Exit가 거래 수를 인플레이션

| Exit 방식 | 실제 포지션 | 기록된 거래 수 | 인플레이션 |
|----------|------------|---------------|-----------|
| Single Exit | 100 | 100 | 0% |
| Scale-out (50/30/20) | 100 | 300 | **+200%** |

**해결**: Simple Backtest로 Scale-out 없이 검증

---

## 3. 수정된 검증 방법론

### 3.1 검증 스크립트

| 스크립트 | 목적 | 방법론 |
|----------|------|--------|
| `adx_validation_simple_compare.py` | Simple Backtest | Entry@next_open, Single Exit |
| `adx_monte_carlo_validation.py` | 통계적 유의성 | 10,000 Bootstrap 시뮬레이션 |
| `adx_walkforward_final.py` | OOS 일관성 | 30일 Train / 10일 Test, 6 Windows |

### 3.2 수정된 Entry/Exit 로직

```python
# Correct Entry Logic
if i >= 2:
    prev_dir = df.iloc[i-2]['direction']  # 2봉 전
    curr_dir = df.iloc[i-1]['direction']  # 1봉 전 (신호 봉)

    if prev_dir != curr_dir:  # 방향 전환 감지
        entry_price = df.iloc[i]['open']  # 현재 봉 시가로 진입

# Correct Exit Logic
if position:
    if direction == 1:  # LONG
        if row['high'] >= tp_price:  # TP: High로 체크
            exit_price = tp_price
        elif row['low'] <= sl_price:  # SL: Low로 체크
            exit_price = sl_price
```

---

## 4. Monte Carlo 검증 결과 (10,000 Simulations)

### 4.1 평균 PnL 비교

| Config | Mean PnL | Std Dev | 95% CI Lower | 95% CI Upper |
|--------|----------|---------|--------------|--------------|
| **Baseline** | **$451.41** | $156.77 | $339.67 | $579.84 |
| ADX≥15 | $442.22 | $165.41 | $318.35 | $575.11 |
| ADX≥20 | $356.22 | $163.10 | $228.26 | $492.15 |
| ADX≥25 | $219.30 | $159.66 | $85.12 | $357.38 |
| ADX≥30 | $69.01 | $115.97 | -$3.63 | $147.26 |

### 4.2 거래당 수익률

| Config | Trades | Per-Trade Return | Trade Reduction |
|--------|--------|------------------|-----------------|
| **Baseline** | 1,184 | 0.144% | - |
| ADX≥15 | 1,119 | 0.148% | -5.5% |
| ADX≥20 | 979 | 0.139% | -17.3% |
| ADX≥25 | 753 | **0.154%** | **-36.3%** |
| ADX≥30 | 459 | 0.103% | -61.2% |

### 4.3 ADX≥25 vs Baseline 직접 비교

| 메트릭 | 값 |
|--------|-----|
| ADX≥25가 Baseline 이길 확률 | **2.4%** |
| 평균 PnL 차이 | **-$232.11** (-51.4%) |
| Per-trade 차이 | **+0.010%p** |

**해석**: ADX≥25는 거래당 품질이 미세하게 높지만, 거래 빈도 감소(-36.3%)로 인해 전체 수익은 절반으로 감소.

### 4.4 수익 확률

| Config | Profit Probability | Risk Level |
|--------|-------------------|------------|
| Baseline | **100.0%** | Low |
| ADX≥15 | 100.0% | Low |
| ADX≥20 | 100.0% | Low |
| ADX≥25 | 99.9% | Low |
| ADX≥30 | 85.5% | Medium |

---

## 5. Walk-Forward 검증 결과 (6 Windows)

### 5.1 Overall Results

| Config | Winning Windows | WR% | Total OOS PnL | Avg PnL/Window |
|--------|-----------------|-----|---------------|----------------|
| **Baseline** | **6/6** | **100%** | **$115.58** | **$19.26** |
| ADX≥15 | 6/6 | 100% | $116.01 | $19.33 |
| ADX≥20 | 6/6 | 100% | $107.33 | $17.89 |
| ADX≥25 | 6/6 | 100% | $79.16 | $13.19 |
| ADX≥30 | 6/6 | 100% | $39.21 | $6.53 |

### 5.2 Window별 비교 (Baseline vs ADX≥25)

| Window | Test Period | Baseline PnL | ADX≥25 PnL | 승자 |
|--------|-------------|--------------|------------|------|
| W1 | 10/02-10/12 | $17.73 | $10.11 | Baseline |
| W2 | 10/12-10/22 | $14.26 | $8.51 | Baseline |
| W3 | 10/22-11/01 | $23.47 | $19.33 | Baseline |
| W4 | 11/01-11/11 | $18.64 | $12.88 | Baseline |
| W5 | 11/11-11/21 | $22.15 | $15.47 | Baseline |
| W6 | 11/21-12/01 | $19.33 | $12.86 | **Baseline** |

**결과**: ADX≥25가 Baseline을 이긴 윈도우 = **0/6 (0%)**

### 5.3 원래 연구와 비교

| 항목 | 원래 연구 | 수정된 검증 | 차이 |
|------|----------|------------|------|
| Baseline WF 일관성 | 2/6 (33%) | **6/6 (100%)** | +67%p |
| ADX≥25 WF 일관성 | 4/6 (67%) | 6/6 (100%) | +33%p |
| ADX≥25 vs Baseline | ADX 우수 | **Baseline 우수** | 🔄 역전! |
| ADX≥25 WF PnL | $17.48 | **$79.16** | 수정 후 +$61.68 |
| Baseline WF PnL | $9.18 | **$115.58** | 수정 후 +$106.40 |

---

## 6. Simple Backtest 결과 (Scale-out 없음)

### 6.1 Full Period 비교

| Config | Full PnL | Trades | Win Rate | Avg Trade |
|--------|----------|--------|----------|-----------|
| **Baseline** | **+76.7%** | 1,184 | 52.8% | +0.065% |
| ADX≥25 | +50.4% | 753 | 54.7% | +0.067% |
| **차이** | **-26.3%p** | -431 | +1.9%p | +0.002%p |

**결론**: ADX≥25는 승률이 미세하게 높지만, 거래 수 감소로 전체 수익 -26.3%p 감소

---

## 7. 🔴 이중 검증 (Dual Verification) - 2026-01-01 추가

> **검증 요청**: "두 가지 검증을 거쳐야 합니다:
> 1. 신호 발생 시 (포지션 상관없이) 해당 신호 지점에서 진입했을 경우 승률이 높아야 함
> 2. 실제 거래처럼 포지션 있으면 추가 진입 불가 조건으로 두 가지 모두 우수해야 함"

### 7.1 이중 검증 방법론

| 검증 유형 | 설명 | 방법론 | 통과 기준 |
|----------|------|--------|----------|
| **Type 1: Signal Quality** | 신호 자체의 예측력 | 모든 신호에 대해 독립적 평가 (포지션 무시) | WR ≥ 50%, EV > 0 |
| **Type 2: Actual Trading** | 실제 거래 시뮬레이션 | 한 번에 하나의 포지션만 (포지션 있으면 진입 불가) | PnL > 0, WF ≥ 50% |

**핵심 원칙**: 두 검증 모두 우수해야 유효한 전략으로 인정

### 7.2 Type 1: Signal Quality 결과

**모든 신호에 대해 독립적으로 평가** (포지션 상태 무시, Entry@next_open)

| Config | 총 신호 수 | 승률 | 기대값 | Type 1 판정 |
|--------|----------|------|--------|------------|
| **Baseline** | 994 | 43.86% | -0.026% | ❌ FAIL (WR < 50%) |
| ADX≥15 | 985 | 43.76% | -0.031% | ❌ FAIL |
| ADX≥20 | 874 | 44.05% | -0.018% | ❌ FAIL |
| ADX≥25 | 641 | 44.93% | +0.022% | ❌ FAIL (WR < 50%) |
| ADX≥30 | 389 | **47.04%** | **+0.117%** | ❌ FAIL (WR < 50%) |

**핵심 발견**:
- **모든 설정이 Type 1 검증 실패** (승률 50% 미만)
- ADX≥30이 가장 높은 승률 (47.04%)이지만 여전히 50% 미달
- 기대값은 ADX≥25, ADX≥30에서만 양수지만, 승률 미달로 FAIL

### 7.3 Type 2: Actual Trading 결과

**실제 거래처럼 포지션 있으면 추가 진입 불가**

| Config | PnL % | 거래 수 | 승률 | Max DD | WF 일관성 | LONG PnL | SHORT PnL | Type 2 판정 |
|--------|-------|--------|------|--------|----------|----------|-----------|------------|
| Baseline | -14.57% | 90 | 43.33% | 27.0% | 2/6 (33%) | -$19.46 | +$4.89 | ❌ FAIL |
| ADX≥15 | -14.57% | 90 | 43.33% | 27.0% | 2/6 (33%) | -$19.46 | +$4.89 | ❌ FAIL |
| ADX≥20 | -16.57% | 89 | 42.70% | 28.0% | 2/6 (33%) | -$20.33 | +$3.75 | ❌ FAIL |
| ADX≥25 | -0.64% | 85 | 47.06% | 20.4% | 4/6 (67%) | -$10.20 | +$9.56 | ❌ FAIL (PnL < 0) |
| **ADX≥30** | **+32.57%** | 82 | **54.88%** | **10.1%** | **5/6 (83%)** | +$2.17 | +$30.40 | ✅ PASS |

**핵심 발견**:
- **ADX≥30만 Type 2 검증 통과** (PnL +32.57%, WF 5/6, 양방향 수익)
- ADX 임계값이 높을수록 Type 2 성과 향상
- 하지만 Type 1에서 모두 실패하므로 결합 결과는 FAIL

### 7.4 Combined 결과 (최종 판정)

| Config | Type 1 | Type 2 | **Combined** |
|--------|--------|--------|--------------|
| Baseline | ❌ FAIL | ❌ FAIL | ❌ **FAIL** |
| ADX≥15 | ❌ FAIL | ❌ FAIL | ❌ **FAIL** |
| ADX≥20 | ❌ FAIL | ❌ FAIL | ❌ **FAIL** |
| ADX≥25 | ❌ FAIL | ❌ FAIL | ❌ **FAIL** |
| ADX≥30 | ❌ FAIL | ✅ PASS | ❌ **FAIL** |

**최종 결론**: 🔴 **모든 설정이 이중 검증 실패** - SuperTrend 전략 자체의 신호 품질이 미달

### 7.5 분석: Type 2는 PASS지만 Type 1은 FAIL인 이유 (ADX≥30)

| 시나리오 | 의미 |
|----------|------|
| Type 1 FAIL + Type 2 PASS | 전체 신호 중 일부만 좋음, 운 좋게 좋은 신호만 실제 거래됨 |
| **위험** | 시장 상황 변화 시 나쁜 신호가 실제 거래될 수 있음 |
| **결론** | 신뢰할 수 없는 전략 - 신호 자체의 품질 개선 필요 |

### 7.6 검증 스크립트

| 스크립트 | 용도 |
|----------|------|
| `scripts/analysis/adx_dual_verification.py` | 이중 검증 실행 |
| `claudedocs/BACKTEST_VERIFICATION_METHODOLOGY_20251231.md` | 방법론 문서 |
| `results/adx_dual_verification_20260101_*.csv` | 결과 데이터 |

---

## 8. 원래 연구 오류 분석

### 7.1 왜 원래 연구에서 ADX≥25가 우수해 보였나?

1. **Entry 타이밍 버그**: 동일 봉 close 진입 = Look-Ahead
2. **SuperTrend 구현 차이**: 프로덕션과 다른 신호 생성
3. **Scale-out 카운팅**: 거래 수 인플레이션으로 통계 왜곡
4. **Exit 방식**: Close 기반 = 비현실적

### 7.2 수정 후 결과 변화

| 메트릭 | 원래 연구 | 수정 검증 | 변화 |
|--------|----------|----------|------|
| Baseline Full PnL | +14.8% | **+76.7%** | +61.9%p |
| ADX≥25 Full PnL | +17.5% | +50.4% | +32.9%p |
| ADX vs Baseline | ADX +2.7%p | **Baseline +26.3%p** | 🔄 역전 |
| 권장 설정 | ADX≥25 | **Baseline (No ADX)** | 🔄 역전 |

---

## 8. 최종 권장사항

### 8.1 ADX Filter 관련

| 권장사항 | 상세 |
|----------|------|
| ❌ **ADX≥25 Filter 적용 금지** | 전체 수익 51.4% 감소, OOS에서 0/6 승률 |
| ❌ **ADX 임계값 낮추기 (15, 20) 효과 없음** | Baseline과 거의 동일하거나 약간 낮음 |
| ✅ **Baseline (No ADX) 유지** | Monte Carlo $451, WF $115.58 최고 성능 |

### 8.2 SuperTrend 구현 관련

| 권장사항 | 상세 | 우선순위 |
|----------|------|----------|
| 🔴 **프로덕션 SuperTrend 업데이트 필수** | Simple → Standard (ratcheting) 전환 | **Critical** |
| 신호 수 | 221 → 1,189 예상 (+438%) | |
| 예상 효과 | 거래 빈도 5배 증가, 전체 수익 증가 | |

### 8.3 향후 연구 방향

1. **Standard SuperTrend로 재연구**: 프로덕션 호환 구현으로 전체 연구 재수행
2. **다른 필터 탐색**: ADX 대신 Volume, Volatility 필터 검토
3. **Entry 타이밍 최적화**: 다음 봉 open 기반 다양한 전략 테스트

---

## 9. 검증 파일 목록

| 파일 | 용도 |
|------|------|
| `scripts/analysis/adx_validation_simple_compare.py` | Simple Backtest |
| `scripts/analysis/adx_monte_carlo_validation.py` | Monte Carlo 10,000회 |
| `scripts/analysis/adx_walkforward_final.py` | Walk-Forward 6 Windows |
| `scripts/analysis/supertrend_implementation_compare.py` | SuperTrend 구현 비교 |
| `results/adx_monte_carlo_validation_*.csv` | MC 결과 |
| `results/adx_walkforward_final_*.csv` | WF 결과 |

---

## 11. 최종 결론 (2026-01-01 업데이트)

### 11.1 이중 검증 결과 요약

| 검증 유형 | 모든 설정 결과 | 핵심 문제 |
|----------|--------------|----------|
| **Type 1: Signal Quality** | ❌ 전체 FAIL | 승률 43.9%~47.0% (< 50%) |
| **Type 2: Actual Trading** | ADX≥30만 PASS | 나머지 전체 손실 |
| **Combined** | ❌ 전체 FAIL | **두 검증 모두 통과하는 설정 없음** |

### 11.2 ADX Filter 평가 (업데이트)

| 설정 | Type 1 승률 | Type 2 PnL | 종합 판정 |
|------|------------|-----------|----------|
| Baseline | 43.86% | -14.57% | ❌ FAIL |
| ADX≥15 | 43.76% | -14.57% | ❌ FAIL |
| ADX≥20 | 44.05% | -16.57% | ❌ FAIL |
| ADX≥25 | 44.93% | -0.64% | ❌ FAIL |
| ADX≥30 | 47.04% | **+32.57%** | ❌ FAIL (Type 1 미달) |

**최종 판정**: 🔴🔴 **SuperTrend 전략 자체가 신호 품질 미달** - ADX 필터와 관계없이 전략 재검토 필요

### 11.3 즉시 조치 필요 사항

1. 🔴 **SuperTrend 5m Bot 전략 재검토** - 신호 품질 50% 미만으로 근본적 한계
2. ❌ **ADX Filter 적용 금지** - 어떤 ADX 임계값도 이중 검증 통과 못함
3. 🔴 **다른 전략 탐색 권장** - RSI Trend Filter, MACD 등 검증된 전략 검토
4. ⚠️ **ADX≥30 제한적 사용 가능** - Type 2만 통과, 리스크 감수 시에만

### 11.4 교훈 및 향후 연구 방향

#### 이중 검증 방법론의 중요성

| 시나리오 | Type 1 | Type 2 | 해석 |
|----------|--------|--------|------|
| ✅ 이상적 | PASS | PASS | 배포 가능 |
| ⚠️ ADX≥30 사례 | FAIL | PASS | 운 좋은 거래만 수익 (불안정) |
| ❌ 완전 실패 | FAIL | FAIL | 전략 폐기 |

#### 향후 연구 시 필수 적용

1. **모든 전략 연구에 이중 검증 적용**
2. **Type 1 (Signal Quality) 먼저 확인** - 50% 미만이면 조기 폐기
3. **Type 2 (Actual Trading) 후속 검증** - 포지션 관리 포함
4. **두 검증 모두 통과해야 프로덕션 배포**

---

### 참고: 이전 검증 결과 (단일 검증 기준)

| 장점 | 단점 |
|------|------|
| 거래당 품질 +0.010%p | 전체 PnL -51.4% |
| 승률 +1.9%p | 거래 빈도 -36.3% |
| - | OOS 6/6 모두 Baseline에 패배 |
| - | Monte Carlo 2.4% 승률 |

---

**최종 검증 완료**: 2026-01-01
**검증자**: Claude AI Assistant
**검증 방법**:
- 이중 검증 (Type 1 + Type 2) ← **최종 결론 기준**
- Monte Carlo (10,000회)
- Walk-Forward (6 Windows)
- Simple Backtest
