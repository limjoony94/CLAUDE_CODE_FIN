# Walk-Forward Decoupled Methodology Analysis

**Date**: 2025-10-29
**Context**: Phase 1 재훈련 후 낮은 거래 빈도 원인 분석

---

## 문제 요약

**기대**: 최근 데이터로 재훈련 → 최근 시장에 적합 → 높은 거래 빈도
**실제**: 재훈련 → 확률 분포 극단적으로 보수적 → 거래 빈도 0.43/day ❌

---

## Walk-Forward Decoupled 방법론 복기

### 1. 훈련 프로세스

```python
for fold in [1, 2, 3, 4, 5]:
    # 1) Validation 기간의 후보 필터링
    val_candidates = filter_candidates(df[val_idx], side)

    # 2) 각 후보에 대해 Trade Outcome 시뮬레이션
    for candidate in val_candidates:
        label = simulate_trade_outcome(
            df, candidate, side, exit_model, exit_scaler
        )
        # label = 1 if leveraged_pnl_pct > 0.02 else 0

    # 3) XGBoost 훈련 (binary classification)
    model.fit(X_train, y_train)

    # 4) Best fold 선택 (highest positive rate)
    if positive_rate > best_positive_rate:
        best_fold = fold
```

### 2. 배포 프로세스

```python
# Entry signal 생성
long_prob = long_entry_model.predict_proba(X)[0][1]
short_prob = short_entry_model.predict_proba(X)[0][1]

# Threshold 적용
if long_prob >= 0.75:
    enter_long()
if short_prob >= 0.75:
    enter_short()
```

---

## 근본 문제 식별

### 문제 1: **Positive Rate vs Probability Threshold 불일치**

**훈련 관점**:
- Positive Rate: 27.96% (LONG), 7.37% (SHORT)
- 의미: "전체 후보 중 27.96%가 2% 이익 달성"

**배포 관점**:
- Threshold: 0.75
- 의미: "이 샘플이 2% 이익 달성할 확률 ≥ 75%"

**불일치**:
```
훈련 시 positive rate가 27.96%라고 해서
모델이 개별 샘플에 대해 75% 확률을 부여하는 것은 아님!

실제 확률 분포:
  평균: 13.13%
  중앙값: 5.92%
  75th percentile: 17.24%

→ 0.75 도달: 0.77%만 해당 (99.2% 필터링)
```

### 문제 2: **XGBoost Probability Calibration 문제**

**XGBoost 특성**:
- Tree-based 앙상블 모델
- 확률 출력이 본질적으로 **poorly calibrated**
- Especially when:
  - Class imbalance 심함 (27.96% vs 72.04%)
  - 복잡한 feature space
  - Limited positive samples

**결과**:
- 모델이 positive class에 대해 under-confident
- 실제 성공 가능성이 높아도 낮은 확률 출력
- 극단적 예: SHORT 평균 확률 1.92% (훈련 positive 7.37%)

### 문제 3: **Best Fold Selection 편향**

```python
# Best fold = highest positive rate
if positive_rate > best_positive_rate:
    best_fold = fold
```

**편향 발생**:
- Positive rate가 높은 fold 선택
- 하지만 이것이 **probability calibration** 보장하지 않음
- Fold 5 선택 (LONG 27.96%, SHORT 7.37%)
- 다른 fold는 더 낮은 positive rate (4-15%)

**결과**:
- Best fold조차 확률 분포가 보수적
- Threshold 0.75는 훈련 positive rate (27.96%)의 2.7배 높음

### 문제 4: **SHORT 모델 근본적 실패**

**SHORT 훈련 결과**:
```yaml
Fold 1: 4.57% positive
Fold 2: 6.18% positive
Fold 3: 3.70% positive
Fold 4: 4.07% positive
Fold 5: 7.37% positive (best)

Best fold조차 7.37%만 positive!
```

**배포 시 확률 분포**:
```yaml
평균: 1.92%
중앙값: 0.21%
75th percentile: 1.00%
95th percentile: 7.65%

≥0.75 도달: 0.02% (1 sample / 4,032)
```

**근본 원인**:
- 최근 시장 (Oct 6-12) 강세장 → SHORT 기회 극소
- 7.37% positive로 학습한 모델 → 1.92% 평균 확률 생성
- **모델이 "SHORT는 거의 성공 안 함"을 학습**

---

## 방법론의 구조적 한계

### 한계 1: **Threshold 고정**

```yaml
문제:
  훈련 시: Positive rate가 fold/side마다 다름 (4-28%)
  배포 시: 모든 side에 동일 threshold (0.75) 적용

결과:
  LONG: 27.96% positive → 0.75 threshold → 간신히 작동
  SHORT: 7.37% positive → 0.75 threshold → 붕괴
```

### 한계 2: **Calibration 무시**

```yaml
현재 접근:
  1) Trade outcome 시뮬레이션 (binary label)
  2) XGBoost 훈련 (확률 출력)
  3) Threshold 적용 (0.75)

누락된 단계:
  - Probability calibration (Platt scaling, Isotonic regression)
  - Calibration 평가 (Brier score, reliability diagram)
  - Threshold 최적화 (positive rate 기반)
```

### 한계 3: **Market Regime Sensitivity**

```yaml
SHORT 모델 사례:
  훈련 기간 (Oct 6-12): 강세장 → 7.37% positive
  테스트 기간 (Oct 13-26): 강세장 지속 → 모델 작동 불가

만약 약세장 도래:
  SHORT 기회 증가 → 하지만 모델은 "SHORT 거의 안됨" 학습
  결과: Opportunity miss
```

---

## 대안 방법론 검토

### 대안 1: **Dynamic Threshold (Percentile-Based)**

```python
# 훈련 시
positive_rate = 27.96%

# 배포 시
threshold = percentile(train_probs, 100 - positive_rate)
# 예: 100 - 27.96 = 72.04th percentile
```

**장점**:
- Training positive rate와 deployment threshold 자동 정렬
- Calibration 문제 우회

**단점**:
- Positive rate가 너무 낮으면 (7.37%) threshold도 낮아짐
- 품질 저하 위험

### 대안 2: **Probability Calibration**

```python
# 훈련 후 calibration
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_model, method='isotonic', cv=5
)
calibrated_model.fit(X_val, y_val)
```

**장점**:
- 확률 출력을 실제 positive rate에 정렬
- Threshold 0.75가 의미 있어짐

**단점**:
- 추가 validation set 필요
- Calibration도 overfitting 가능

### 대안 3: **Threshold Grid Search**

```python
# 백테스트 기반 최적 threshold 탐색
for long_thresh in [0.50, 0.55, ..., 0.80]:
    for short_thresh in [0.50, 0.55, ..., 0.80]:
        backtest_result = run_backtest(long_thresh, short_thresh)
        if backtest_result meets criteria:
            optimal_thresholds = (long_thresh, short_thresh)
```

**장점**:
- 실제 성능 기반 최적화
- 각 side별 독립적 threshold

**단점**:
- Overfitting 위험 (test set에 fitting)
- Holdout period에만 최적

### 대안 4: **Adaptive Threshold**

```python
# Market regime에 따라 threshold 조정
if market_regime == "bullish":
    short_threshold = 0.90  # 더 엄격하게
    long_threshold = 0.65   # 더 느슨하게
elif market_regime == "bearish":
    short_threshold = 0.65
    long_threshold = 0.90
```

**장점**:
- Market regime sensitivity 해결
- 시장 상황에 맞춤

**단점**:
- Regime detection 필요 (또 다른 모델)
- Complexity 증가

### 대안 5: **Return to Fixed Ratio Strategy**

```python
# Walk-Forward 없이 더 긴 기간 훈련
train_period = "2025-07-14 to 2025-10-19"  # 97일
holdout = "2025-10-20 to 2025-10-26"       # 7일

# Threshold 백테스트 최적화 (holdout 사용 안함)
optimal_thresholds = optimize_on_validation_set()
```

**장점**:
- 더 많은 훈련 데이터
- 덜 overfitting (walk-forward overhead 감소)

**단점**:
- Temporal gap 여전히 존재
- Recent market regime 반영 부족

---

## 권고 사항

### 즉시 실행 가능:

**1. Threshold 하향 조정 (Quick Fix)**
```yaml
현재: LONG 0.75, SHORT 0.75
제안: LONG 0.70, SHORT 0.65

기대 효과:
  LONG: 2.21 → 3.64 거래/일 (+65%)
  SHORT: 0.07 → 0.50 거래/일 (+614%)
  Combined: ~4.0 거래/일 ✅
```

**2. Probability Distribution 기반 Threshold**
```yaml
LONG: 75th percentile = 0.1724 (17.24%)
SHORT: 95th percentile = 0.0765 (7.65%)

제안:
  LONG: threshold = 0.65 (약 6.0 거래/일)
  SHORT: threshold = 0.50 (약 2.0 거래/일)
```

### 중기 개선:

**3. Probability Calibration 도입**
- Isotonic regression 적용
- Calibration validation set 분리
- 재훈련 pipeline에 포함

**4. Independent Threshold per Side**
- LONG과 SHORT 독립적 threshold 최적화
- Positive rate 고려한 자동 조정

### 장기 전략:

**5. 방법론 재검토**
- Walk-Forward Decoupled 대신
- Rolling Window + Calibration
- Online Learning 도입

**6. Market Regime Detection**
- Bullish/Bearish/Sideways 자동 감지
- Regime별 threshold 적용

---

## 결론

**Walk-Forward Decoupled 방법론은**:
- ✅ Temporal bias 방지 (look-ahead 없음)
- ✅ 최신 데이터 반영 (recent 90일)
- ❌ **Probability calibration 무시**
- ❌ **Fixed threshold와 불일치**
- ❌ **Market regime sensitivity 높음**

**즉시 조치**:
1. Threshold 하향: LONG 0.70, SHORT 0.65
2. 14일 holdout 재검증
3. 2.0+ 거래/일 달성 확인

**중장기 개선**:
1. Calibration pipeline 추가
2. Dynamic threshold 도입
3. 방법론 전환 검토

---

**작성자**: Claude
**검토 필요**: Threshold 조정 후 성능 검증
