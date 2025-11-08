# LONG Probability Investigation Report
**Date**: 2025-10-19
**Issue**: Abnormally high LONG probabilities in production (70-98%)

---

## 📋 Investigation Summary

**User Question**: "27.59%/84.0% 달성 모델이 백테스트에서도 동일하게 높은 롱 확률을 출력하였는가?"

**Answer**: ❌ **NO - 백테스트와 프로덕션의 확률 분포가 다릅니다**

---

## 🔍 Key Findings

### 1. LONG Probability Distribution Comparison

```yaml
Backtest (Full Dataset, 31,488 candles):
  Period: ~105 days historical data
  LONG Mean: 23.77%
  LONG >= 0.65: 13.8% of candles
  Status: ✅ Normal distribution

Backtest (Recent 1000 candles):
  Period: Last ~3.5 days of historical data
  LONG Mean: 62.30%
  LONG >= 0.65: 56.4% of candles
  Status: ⚠️ Elevated probabilities

Production (2025-10-19, 78 samples):
  Period: Current live trading
  LONG Mean: 81.84%
  LONG >= 0.65: ~100% of candles
  Status: ❌ Extremely high probabilities
```

### 2. Time-Series Trend Discovery

**LONG Probability Over Time**:
```
전체 기간 (31,488 candles):  23.77%  ← Backtest performance achieved here
           ↓
최근 1000 candles:           62.30%  (+162%)
           ↓
현재 프로덕션 (10/19):       81.84%  (+244%)
```

**Critical Observation**: **시간에 따른 LONG 확률 증가 추세 발견**

### 3. Production Logs Analysis (By Date)

```yaml
2025-10-17:
  Samples: 1,787
  LONG avg: 20.76%
  Status: ✅ Normal (similar to backtest average)

2025-10-18:
  Samples: 435
  LONG avg: 3.12%
  Status: ⚠️ Unusually low

2025-10-19:
  Samples: 78
  LONG avg: 81.84%
  Status: ❌ Abnormally high
```

**Pattern**: 확률이 날짜별로 크게 변동함

---

## ❌ Problem Identified

### Core Issue: **Backtest-Production Mismatch**

**Backtest Performance (27.59%, 84.0%)**는:
- ✅ 전체 기간 (LONG 평균 23.77%)에서 달성됨
- ❌ 현재 시장 (LONG 평균 81.84%)에서는 검증되지 않음
- ❌ 최근 시장 특성이 백테스트 기간과 다름

### Implications

1. **Performance Validity**: 27.59%/84.0% 성능은 현재 시장에서 재현되지 않을 수 있음
2. **Strategy Drift**: 모델이 과거 시장 패턴에 최적화됨, 현재 시장에는 부적합
3. **Risk Exposure**: 거의 모든 캔들에서 LONG 진입 신호 → 과도한 거래 가능성

---

## 🔬 Root Cause Analysis (Hypotheses)

### Hypothesis 1: **Model Overfitting to Historical Data**
- Trade-Outcome 모델이 과거 105일 데이터에 과적합
- 현재 시장 특성과 다른 패턴 학습
- **Evidence**: 최근 데이터로 갈수록 LONG 확률 증가

### Hypothesis 2: **Market Regime Change**
- 최근 시장이 강한 상승 추세 또는 변동성 변화
- 모델이 학습한 패턴과 다른 시장 환경
- **Evidence**: 10/17 (20.76%) → 10/19 (81.84%) 급격한 변화

### Hypothesis 3: **Feature Distribution Shift**
- 기술 지표 (RSI, MACD, Volume 등)의 분포 변화
- StandardScaler가 최근 데이터에 부적합
- **Evidence**: 조사 중 (deep analysis 실행 중)

### Hypothesis 4: **Labeling Bias**
- Trade-Outcome 라벨링이 특정 시장 조건에 편향
- 상승장에서 LONG이 성공적이었기 때문에 상승 패턴만 학습
- **Evidence**: SHORT 확률 매우 낮음 (평균 0.34%)

---

## 📊 Investigation Status

### ✅ Completed Analysis

1. **Model Deployment Verification**:
   - ✅ 백테스트와 프로덕션 모델 동일 확인
   - ✅ 피처 계산 파이프라인 동일 확인
   - ✅ 데이터 전달 방식 동일 확인

2. **Probability Distribution Analysis**:
   - ✅ 백테스트 전체 기간: LONG 평균 23.77%
   - ✅ 백테스트 최근 1000: LONG 평균 62.30%
   - ✅ 프로덕션 현재: LONG 평균 81.84%

3. **Temporal Trend Identification**:
   - ✅ 시간에 따른 LONG 확률 증가 추세 발견
   - ✅ 프로덕션 로그 날짜별 분석 완료

### ⏳ In Progress

4. **Deep Analysis** (Running):
   - ⏳ 시간대별 세부 확률 분포 (10개 기간)
   - ⏳ 각 시기별 백테스트 성능 검증
   - ⏳ 시장 특성 변화 분석 (가격, 거래량, 지표)
   - **ETA**: 3-5 minutes

### 🔜 Pending Investigation

5. **Feature Importance Analysis**:
   - 어떤 피처가 높은 LONG 확률을 유발하는가?
   - 최근 데이터에서 특정 피처 값 변화

6. **Model Calibration Check**:
   - 모델 확률이 실제 성공률과 일치하는가?
   - Platt scaling 필요성 검토

---

## 🎯 Preliminary Conclusions

### 1. **Backtest Performance는 현재 시장과 무관**

27.59%/84.0% 성능은:
- 전체 105일 데이터 (LONG 평균 23.77%)에서 달성
- 현재 시장 (LONG 평균 81.84%)은 백테스트 기간과 **완전히 다른 환경**
- **현재 성능은 예측 불가**

### 2. **모델이 최근 시장에 부적합할 가능성**

증거:
- 거의 모든 캔들에서 LONG >= 0.65
- SHORT 신호 거의 없음 (0.34%)
- 일방적 편향 → 전략 다양성 부족

### 3. **즉각적인 조치 필요**

현재 프로덕션 운영은 **검증되지 않은 상태**:
- 백테스트 성능과 무관한 확률 분포
- 과도한 진입 신호
- 리스크 관리 어려움

---

## 🚨 Immediate Concerns

### Risk Assessment: **HIGH**

**문제점**:
1. **성능 불확실성**: 현재 시장에서 모델 성능 미검증
2. **Overtrading 위험**: 거의 모든 캔들에서 진입 신호
3. **전략 편향**: SHORT 기회 무시, LONG 과도 집중
4. **백테스트 신뢰도**: 과거 성능이 현재와 무관

**영향**:
- 자본 손실 가능성
- 수수료 과다 발생
- 전략 다양성 부족으로 인한 리스크 증가

---

## 📝 Next Steps

### Option 1: **즉시 중단 및 재평가** (권장)
```yaml
Action:
  - 프로덕션 봇 중단
  - 최근 1000 캔들 데이터로 재백테스트
  - 현재 시장 조건에서 성능 검증

Risk: None (안전 우선)
Timeline: 1-2 hours
```

### Option 2: **임계값 상향 조정**
```yaml
Action:
  - LONG threshold: 0.65 → 0.80
  - 진입 빈도 감소
  - Warmup period 유지

Risk: Medium (미봉책, 근본 해결 아님)
Timeline: Immediate
```

### Option 3: **모델 재훈련**
```yaml
Action:
  - 최근 데이터 (10/15~현재) 추가
  - 전체 모델 재훈련
  - 새로운 백테스트 검증

Risk: Low (근본적 해결)
Timeline: 4-6 hours
```

### Option 4: **현재 시장 전용 백테스트**
```yaml
Action:
  - 최근 1000 캔들만 사용한 백테스트
  - 현재 확률 분포에서 예상 성능 계산
  - 성능 검증 후 운영 결정

Risk: Medium (데이터 부족)
Timeline: 30 minutes
```

---

## 🔬 Deep Analysis (In Progress)

### Analysis Script: `temp_deep_analysis.py`

**Investigating**:
1. **시간대별 LONG 확률 추이**
   - 10개 기간으로 분할
   - 각 기간 LONG 평균 계산
   - 추세 시각화

2. **기간별 백테스트 성능**
   - 각 기간별 수익률, 승률 계산
   - 높은 LONG 확률 기간의 실제 성능 검증
   - 성능 상관관계 분석

3. **시장 특성 변화**
   - 초기 3000 vs 최근 3000 캔들 비교
   - 주요 지표 (price, volume, RSI, MACD) 변화
   - 분포 이동 (distribution shift) 확인

**Expected Output**:
- 언제부터 LONG 확률이 증가했는가?
- 높은 LONG 확률 기간의 실제 성능은?
- 어떤 시장 특성이 변화했는가?

---

## 📌 Key Takeaways

1. ✅ **모델 배포는 정확**: 백테스트와 프로덕션 모델/파이프라인 동일
2. ❌ **확률 분포 불일치**: 백테스트 (23.77%) vs 프로덕션 (81.84%)
3. ❌ **백테스트 성능 무효화**: 현재 시장 조건에서 27.59%/84.0% 검증 안 됨
4. ⚠️ **시간 의존성**: 최근으로 올수록 LONG 확률 증가 추세
5. 🚨 **즉각 조치 필요**: 현재 프로덕션 운영은 검증되지 않은 상태

---

## 📚 Supporting Documents

- `MODEL_DEPLOYMENT_VERIFICATION_20251019.md`: 모델 배포 검증 (100% 일치)
- `temp_quick_prob_check.py`: 백테스트 확률 분포 분석 스크립트
- `temp_deep_analysis.py`: 심층 분석 스크립트 (실행 중)
- Production logs: `opportunity_gating_bot_4x_20251017/18/19.log`

---

**Status**: 🔬 **UNDER INVESTIGATION**
**Priority**: 🔴 **HIGH**
**Recommendation**: **프로덕션 중단 및 재평가 권장**

---

*This investigation reveals that the backtest performance (27.59%, 84.0%) was achieved in a different market regime (LONG avg 23.77%) than current production (LONG avg 81.84%). The model may not perform as expected in the current market conditions.*
