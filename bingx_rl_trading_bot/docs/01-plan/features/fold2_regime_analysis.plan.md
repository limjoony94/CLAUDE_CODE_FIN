# Plan: Fold 2 Regime Analysis — Candidate_C Weakness Investigation

> **Feature**: fold2_regime_analysis
> **Date**: 2026-04-19
> **Phase**: Plan
> **Trigger**: candidate_c_validation STOP 결과의 fold 2 약점 (-9.03pp, 2025-07-11~09-15)
> **Type**: Research-only (diagnostic, no production change)

---

## 1. Background

### Fold 2 약점 관찰

`candidate_c_validation` 결과 slip_med WF 5-fold에서 fold 2만 음수:
```
Fold 1: 2025-05-05 ~ 2025-07-11 | trades=217 | WR 29.5% | PnL +1.19   ✓ barely
Fold 2: 2025-07-11 ~ 2025-09-15 | trades=223 | WR 26.5% | PnL -9.03   ✗ FAIL
Fold 3: 2025-09-15 ~ 2025-11-21 | trades=200 | WR 37.0% | PnL +33.56  ✓
Fold 4: 2025-11-21 ~ 2026-01-26 | trades=205 | WR 28.3% | PnL +12.76  ✓
Fold 5: 2026-01-26 ~ 2026-04-03 | trades=212 | WR 31.6% | PnL +24.59  ✓
```

Fold 2가 candidate_C의 core flag (`wf_slip_pass`) 실패의 직접 원인. 65일 구간 223 trades의 특성이 **다른 fold와 어떻게 다른지** 규명 필요.

### 가능한 원인 가설
1. **낮은 변동성 (Low-volatility regime)**: 좁은 range로 돌파 신호의 drawdown 증가
2. **횡보 상관관계 (Choppy/sideways)**: 돌파 후 즉시 반전(whipsaw) 빈발
3. **체결 질 저하**: 이 기간 거래소 슬리피지 특별히 높음 (BTC 뉴스 영향 등)
4. **고변동 spike**: 채널 전체를 스칩은 후 반전 — SL 터치 후 반등
5. **candidate_C만의 약점**: 넓은 SL(4.0×ATR)이 이 구간 구조와 특수 부적합

### 왜 중요한가
- 후속 PDCA "Regime-conditional candidate_C"의 전제 조건
- Regime detect 기준을 세우려면 fold 2가 왜 약한지 정확히 알아야 함
- 만약 **구조적 약점이 아닌 샘플 특수성**이면 candidate_C가 실제로 강한 edge 가능
- 만약 **구조적 약점**이면 regime filter 설계 방향 결정

---

## 2. Goal

Fold 2 구간(2025-07-11 ~ 2025-09-15)의 **고유 특성**을 정량적으로 규명:
- BTC 시장 레짐 (변동성, 추세성, spike 빈도)
- 전략 거동 (돌파 빈도, SL/TP 비율, 평균 손실)
- Baseline vs Candidate_C **같은 구간**에서의 차이
- **Structural weakness vs sample noise** 판정

---

## 3. Hypotheses

| H | 내용 | 검증 지표 |
|---|------|----------|
| **H1** | Fold 2는 다른 fold 대비 **저변동성** | ATR 평균, close std, daily range |
| **H2** | Fold 2는 **돌파 빈도 낮음** (신호 생성 적음) | 일일 평균 trades, 돌파 %/day |
| **H3** | Fold 2의 **SL exit 비율 높음** (whipsaw) | exit_reason 분포 per fold |
| **H4** | Fold 2에서 **avg loss > avg win** (R:R 악화) | per-fold avg_loss, avg_win |
| **H5** | Baseline도 fold 2에서 약하지만 candidate_C 만큼 심각하지 않음 → **widening SL이 whipsaw 증폭** | baseline vs cand fold 2 PnL/MDD |
| **H6** | Fold 2의 **특정 short sub-window**(예: 2~3주)이 전체 fold 손실 대부분 | 30일 rolling PnL |
| **H7** | Fold 2는 **Regime classifier로 식별 가능** (e.g., ATR/close < threshold) | regime proxy histogram |

---

## 4. Success Criteria (정보 수집 목표)

이 PDCA는 **판정(GO/STOP)이 아닌 진단**이 목적. 성공 기준:

1. Fold 2 구간 시장 레짐 정량 프로파일 확보 (ATR, 변동성, 추세 지표 등)
2. Baseline vs Candidate_C 같은 구간 동일 metrics 비교
3. 가설 H1~H7 각각 **양/음 판정** (증거 기반)
4. **Regime detection 룰 후보 제시** (H7 검증 시)
5. **Structural vs sample** 결론 도출

---

## 5. Methodology

### 5.1 데이터
- 기존 5m → 15m resampled data (332일)
- Fold 2 정확 구간: **2025-07-11 08:45 ~ 2025-09-15 18:45 UTC**
- 15m bars: [6407..12783] (6,376 bars = ~66.4일)
- 5m bars: [19221..38349] (19,128 bars)

### 5.2 레짐 지표 (per 15m bar, 그리고 per-fold aggregated)
| 지표 | 계산 |
|------|------|
| ATR(14) | 기존 계산 재사용 |
| Close returns std (14-bar rolling) | `(close/close[-1]-1).std()` |
| Range pct (high-low)/close | 일별 평균 |
| ADR (avg daily range) | 24h rolling |
| Trend (EMA20-EMA50) | 추세 강도 |
| Sideways index | (max_high - min_low) / ATR 평균 over N bars — 좁을수록 횡보 |
| Breakout freq | entry_baseline() 통과 신호 수 / fold 일수 |

### 5.3 전략 거동 분석 (per fold)
| 지표 | 계산 |
|------|------|
| Trades/day | total_trades / fold_days |
| WR | wins / total |
| Avg win/loss | mean per group |
| R:R | avg_win / avg_loss |
| Exit reason 분포 | SL%, TRAIL_TP%, EMERGENCY%, TIMEOUT% |
| Median bars_held | 포지션 평균 보유 기간 |
| Max consecutive loss streak | 연속 손실 |

### 5.4 비교 대상 모드
- **baseline (3.3, 2.5, 192)** - 5m + slip_med
- **candidate_C (4.0, 2.5, 192)** - 5m + slip_med
- Fold 2 vs Fold 1,3,4,5 평균

### 5.5 구간별 microscopy
Fold 2 내부를 5일 단위 sub-window로 분해:
- 각 window PnL / WR / trade 수
- Worst window 특정 → 해당 구간의 시장 snapshot

### 5.6 Regime classifier 후보
H7 검증 시 다음 룰 제안:
```python
# 예시: 저변동성 필터
if atr_pct < THRESHOLD:  # e.g., ATR/close < 0.5%
    skip_entry()
```
Threshold는 fold 2를 구별해내는 최소값으로 calibration.

---

## 6. Implementation Plan

### 스크립트: `scripts/analysis/fold2_regime_analysis.py` (신규, ~350 lines)

1. Load 5m/15m data, ATR, channel, fractal (기존 engine)
2. Define fold boundaries (trade-based, from candidate_c_validation 결과)
3. Per-fold regime metrics 계산 (ATR, std, range, trend, sideways)
4. Per-fold strategy 거동 (baseline vs cand_C)
5. Sub-window microscopy (5-day rolling)
6. Regime classifier 후보 파라미터 스윕
7. 결과 JSON + 시각화 가능 데이터 저장

### 예상 실행 시간
- Per-fold 통계: ~5초
- Sub-window microscopy: ~10초
- Regime classifier calibration: ~20초
- 총 **1분 이내**

---

## 7. Non-Goals

- candidate_C production 적용 (본 PDCA 이후 regime-conditional PDCA에서 다룸)
- Fold 1이나 다른 fold 약점 연구 (별개)
- 다른 파라미터 조합 재탐색
- LIVE 데이터 수집

---

## 8. Rollback

Research-only. 파일 삭제만 하면 원복. production 변경 없음.

---

## 9. Risks

| Risk | Mitigation |
|------|-----------|
| Fold 2가 **설명 가능한 단일 사건** (e.g., flash crash week)에 의해 설명되면 regime filter 불가능 | 그 경우 "sample noise" 결론 → candidate_C의 조건부 GO 가치↑ |
| 데이터 한계 (5m → 15m, tick 없음) | 가능한 범위에서 진행, tick 필요한 지표는 제외 |
| Regime classifier가 overfit (fold 2만 필터, 다른 fold에서 거래 없앰) | 각 fold 영향 평가, fold 2만 제거하고 나머지 영향 <5% 조건 |

---

## 10. Reference

- candidate_c_validation 결과: `results/candidate_c_validation_20260419_151610.json`
- 선행 메모리: `memory/candidate_c_validation_20260419.md`, `sl_trail_tuning_20260419.md`
- 재사용 엔진: `scripts/analysis/intrabar_trail_impact.py`, `c1_intrabar_parity.py`
- Fold 2 범위: 2025-07-11 ~ 2025-09-15 UTC (bars 6407..12783)
