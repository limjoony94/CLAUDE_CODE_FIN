# 근본 원인: 백테스트 방법론의 Gap
**Date**: 2025-10-15 18:32
**발견**: 백테스트 #1과 #2의 심각한 불일치
**Status**: 🚨 **CRITICAL** - EXIT_THRESHOLD 검증 gap

---

## 🔍 발견된 근본 문제

### 문제 요약

**EXIT_THRESHOLD = 0.70의 출처를 추적한 결과**:
- 두 개의 **서로 다른 백테스트**에서 나온 서로 다른 최적값
- 두 백테스트는 **겹치는 범위가 없음**
- **직접 비교 불가능**

```yaml
백테스트 #1 (Exit Model 단독):
  파일: scripts/experiments/backtest_exit_model.py
  테스트 범위: [0.1, 0.2, 0.3, 0.5]
  최적값: 0.2
  성능: 46.67% return, 95.7% win rate, 1.03h hold

백테스트 #2 (종합 파라미터 최적화):
  파일: scripts/analysis/backtest_exit_parameter_optimization.py
  테스트 범위: [0.70, 0.75, 0.80]
  최적값: 0.70
  성능: 47.53% return, 81.9% win rate, 1.53h hold

❌ Gap: 0.3 - 0.70 사이가 테스트되지 않음!
```

---

## 📊 백테스트 #1 vs #2 차이점 분석

### 1. 테스트 데이터 범위

**백테스트 #1** (`backtest_exit_model.py`):
```python
# Rolling window approach
WINDOW_SIZE = 1440  # 5 days
def rolling_window_backtest():
    windows = []
    start_idx = 0
    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx]
        # Test on this window
        start_idx += WINDOW_SIZE
```

**특징**:
- 전체 데이터를 5일 window로 슬라이딩
- 여러 시장 regime 포함
- **더 Robust한 검증**

**백테스트 #2** (`backtest_exit_parameter_optimization.py`):
```python
# Line 111-113
test_size = int(len(df) * 0.2)  # 최근 20%만
test_start = len(df) - test_size
df_test = df.iloc[test_start:].copy()
```

**특징**:
- 최근 20% 데이터만 사용
- 특정 기간에 편향 가능성
- **시간적 일반화 부족**

### 2. Exit 조건 차이

**백테스트 #1**:
```python
# Exit conditions:
1. ML Exit Model (threshold 기반)
2. Emergency Stop Loss: -2%
3. Max Holding: 8 hours

# 단순한 exit 로직
```

**백테스트 #2**:
```python
# Exit conditions:
1. Stop Loss: [1%, 1.5%, 2%] 테스트
2. Take Profit: [2%, 3%, 4%] 테스트
3. Max Holding: [3h, 4h, 6h] 테스트
4. ML Exit Model (threshold 기반)

# 81개 조합 테스트 (3×3×3×3)
```

**차이점**:
- 백테스트 #2는 SL/TP가 ML Exit보다 **먼저 체크됨**
- TP 2%에서 많은 거래가 조기 청산될 수 있음
- ML Exit의 실제 효과가 **희석될 수 있음**

### 3. 테스트된 Threshold 범위

**백테스트 #1**:
```python
# Line 520-546
test_configs = [
    {'name': 'Exit Model @ 0.5', 'params': {'threshold': 0.5}},
    {'name': 'Exit Model @ 0.3', 'params': {'threshold': 0.3}},
    {'name': 'Exit Model @ 0.2', 'params': {'threshold': 0.2}},  # BEST
    {'name': 'Exit Model @ 0.1', 'params': {'threshold': 0.1}},
]
```

**백테스트 #2**:
```python
# Line 120
exit_thresholds = [0.70, 0.75, 0.80]
```

**왜 0.70부터 시작했는가?**
- 백테스트 #1의 결과(0.2)를 **무시**했거나
- EXIT_THRESHOLD를 "높은 신뢰도"로 해석해서 0.7부터 시작했거나
- **근거 없는 임의 선택**

---

## 🚨 핵심 문제점

### 1. 검증 Gap: 0.3 ~ 0.70 사이 미검증

```
[Tested]    [Gap]    [Tested]
  0.1        |          0.70
  0.2    <---Gap--->    0.75
  0.3        |          0.80
  0.5        |
```

**문제**:
- 0.2가 최적인지, 아니면 0.4, 0.5, 0.6에서 더 좋은 값이 있는지 모름
- 0.70이 0.2보다 나은지 **직접 비교 불가능**

### 2. 데이터 범위 편향

**백테스트 #1**:
- 전체 데이터 rolling window → 다양한 시장 조건
- 더 일반화된 결과

**백테스트 #2**:
- 최근 20%만 → 특정 기간 편향 가능
- 만약 최근 20%가 **특별한 시장 조건**이었다면?

**예시**:
```yaml
만약 최근 20%가 "강한 상승장"이었다면:
  - 높은 EXIT_THRESHOLD (0.70): 수익 극대화
  - 낮은 EXIT_THRESHOLD (0.20): 조기 청산 → 수익 놓침

  하지만 다른 시장 조건에서는:
  - 높은 EXIT_THRESHOLD (0.70): 손실 확대
  - 낮은 EXIT_THRESHOLD (0.20): 빠른 손절 → 손실 최소화
```

### 3. Exit 조건 복잡도 차이

**백테스트 #1**:
```yaml
Exit 우선순위:
  1. Emergency SL (-2%)
  2. ML Exit (threshold)
  3. Max Hold (8h)

→ ML Exit이 주도적
```

**백테스트 #2**:
```yaml
Exit 우선순위:
  1. Stop Loss (-1% ~ -2%)
  2. Take Profit (+2% ~ +4%)
  3. Max Hold (3h ~ 6h)
  4. ML Exit (threshold)

→ TP 2%에서 많은 거래가 먼저 청산
→ ML Exit 효과 희석
```

**Line 175-218 (백테스트 #2)**:
```python
# Exit conditions checked in ORDER:
if pnl_pct <= -stop_loss:  # 1. SL 먼저
    should_exit = True
elif pnl_pct >= take_profit:  # 2. TP 다음
    should_exit = True
elif hold_time >= max_hold_candles:  # 3. Max Hold
    should_exit = True
else:  # 4. ML Exit 마지막
    exit_prob = exit_model.predict_proba(...)
    if exit_prob >= exit_thresh:
        should_exit = True
```

**결과**:
```yaml
백테스트 #2 Best Config:
  TP: 2% (조기 이익 실현)
  EXIT_THRESHOLD: 0.70

Exit Breakdown:
  TP Exit: ~30-40% (추정)
  ML Exit (0.70): ~40-50%
  나머지: SL, MaxHold

→ 실제로 ML Exit 0.70의 효과를 제대로 측정했는가?
```

---

## 📈 성능 비교 (주의: 직접 비교 불가)

| Metric | 백테스트 #1 (0.2) | 백테스트 #2 (0.70) | 차이 |
|--------|-------------------|---------------------|------|
| **Return** | 46.67% | 47.53% | +0.86% |
| **Win Rate** | 95.7% | 81.9% | **-13.8%** |
| **Holding Time** | 1.03h (62min) | 1.53h (92min) | +30 min |
| **Data Range** | Rolling window | 최근 20% | 다름 |
| **Exit Logic** | ML 주도 | TP 2% 주도 | 다름 |

**표면적 해석**:
- 백테스트 #2 (0.70)이 return 약간 높음
- 하지만 win rate 13.8% 낮음 (95.7% → 81.9%)
- holding time 50% 길어짐

**문제점**:
1. **데이터가 다름** → 직접 비교 불가
2. **Exit 조건이 다름** → 0.2 vs 0.70 비교가 아님
3. **백테스트 #2의 47.53%는 TP 2%의 효과**일 수 있음

---

## 🔍 근본 원인 분석

### 왜 이런 Gap이 생겼는가?

**1. 백테스트 진행 순서 문제**:
```yaml
Timeline:
  1. 백테스트 #1 실행: Exit Model 단독, 0.1-0.5 테스트
     결과: 0.2 최적

  2. 종합 최적화 결정: "모든 파라미터 최적화하자"

  3. 백테스트 #2 실행:
     - EXIT_THRESHOLD: 0.70부터 시작 (??)
     - 이유: 백테스트 #1 결과 무시?
     - 또는: "높은 신뢰도 = 0.7+"로 오해?
```

**2. 방법론 일관성 부족**:
```yaml
백테스트 #1:
  목적: Exit Model 효과 검증
  접근: Exit Model 중심, 단순한 SL/TP

백테스트 #2:
  목적: 전체 파라미터 최적화
  접근: SL/TP/MaxHold + Exit Model 조합

→ 두 백테스트의 목적이 다름
→ 결과를 직접 비교할 수 없음
```

**3. 테스트 범위 설계 오류**:
```yaml
올바른 접근:
  1. 백테스트 #1에서 0.2 최적 발견
  2. 백테스트 #2에서 0.2 주변 정밀 탐색 (0.1, 0.2, 0.3, 0.4, 0.5)
  3. 동일한 SL/TP 조건으로 비교

실제 진행:
  1. 백테스트 #1에서 0.2 최적 발견
  2. 백테스트 #2에서 0.70부터 시작 (???)
  3. Gap 발생
```

---

## ✅ 근본 해결 방안

### Option A: Gap 백테스트 (즉시 실행 가능)

**목적**: 0.2 vs 0.70 직접 비교

**방법**:
```python
# backtest_exit_threshold_gap_analysis.py

# 같은 조건으로 전체 범위 테스트
exit_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]

# 고정 조건:
STOP_LOSS = 0.01  # 1%
TAKE_PROFIT = 0.02  # 2% (백테스트 #2 최적)
MAX_HOLDING = 4  # hours

# 동일한 데이터 범위:
test_range = df.iloc[-len(df)//5:]  # 최근 20%

# 결과:
# - 0.2 vs 0.70 직접 비교
# - Gap 구간 (0.3-0.6) 성능 확인
```

**예상 시간**: 15-20분 (9개 threshold × 백테스트)

**장점**:
- 근본 문제 해결 (gap 메우기)
- 0.2 vs 0.70 명확한 비교
- 최적 threshold 재확인

**단점**:
- Trade #2 모니터링 중단 필요? (아니면 병렬 실행)
- 즉시 결과 필요

### Option B: V4 Bayesian 결과 대기 (추천)

**현재 상황**:
- V4 Bayesian: 187/220 iterations (85% 완료)
- EXIT_THRESHOLD 탐색 범위: 0.60 - 0.85
- ETA: ~11 minutes

**V4의 장점**:
```yaml
백테스트 방법:
  - 전체 90일 데이터 사용
  - Bayesian optimization (효율적 탐색)
  - SL/TP/MaxHold + EXIT_THRESHOLD 동시 최적화
  - 220 iterations (매우 체계적)

결과:
  - 0.60-0.85 범위에서 최적값 찾음
  - 백테스트 #1의 0.2와 비교 가능
  - 백테스트 #2의 0.70 검증 가능
```

**단점**:
- V4는 0.60부터 시작 (0.2-0.5는 탐색 안 함)
- 여전히 0.2 vs V4 최적값 비교 필요

### Option C: 2단계 검증 (가장 체계적)

**Phase 1: V4 결과 분석 (~15 min)**
```yaml
1. V4 완료 대기 (11분 남음)
2. V4 최적 EXIT_THRESHOLD 확인
3. V4 vs 백테스트 #2 (0.70) 비교
4. V4 범위 내에서 최적값 확정
```

**Phase 2: Gap 백테스트 (~20 min)**
```yaml
5. V4 최적값 vs 백테스트 #1 (0.2) 비교 필요 판단
6. 만약 V4 >> 0.70 성능이면: 0.2 재검증 필요
7. Gap 백테스트 실행 (0.2-0.6 범위)
8. 최종 최적값 확정
```

**Phase 3: 프로덕션 배포**
```yaml
9. 최종 최적 EXIT_THRESHOLD 결정
10. Trade #2 결과 분석과 종합
11. EXPECTED_SIGNAL_RATE + EXIT_THRESHOLD 동시 배포
```

---

## 🎯 즉각적 조치 사항

### 1. V4 Bayesian 완료 대기 (11분)

**이유**:
- 가장 체계적인 최적화
- 전체 90일 데이터
- 220 iterations
- 11분이면 완료

**Trade #2 모니터링**:
- 병렬로 계속 진행
- Exit prob 현재: 0.000 (매우 낮음)
- 아직 충분한 시간 있음

### 2. V4 완료 후 즉시 분석

**분석 항목**:
```yaml
1. V4 최적 EXIT_THRESHOLD 값
2. V4 최적 성능 (return, win rate, holding time)
3. V4 vs 백테스트 #2 (0.70) 비교
4. V4 vs 백테스트 #1 (0.2) 간접 비교

결정:
  If V4 최적 >> 0.70:
    → Gap 백테스트 필요 (0.2 재검증)

  If V4 최적 ≈ 0.70:
    → 0.70 유지, 0.2 재검증은 선택사항

  If V4 최적 < 0.70:
    → Gap 백테스트 필수 (0.2-0.6 탐색)
```

### 3. Gap 백테스트 준비 (스크립트 작성)

**지금 할 수 있는 것**:
- Gap 백테스트 스크립트 준비
- V4 완료 시 즉시 실행 가능하도록

---

## 📊 Trade #2 모니터링 상태

```yaml
Current Time: 18:30
Entry: 18:00 (30분 경과)
Exit Prob: 0.000 (EXIT_THRESHOLD 0.70과 매우 멀리)
Status: OPEN

예상:
  - Exit prob 0.70 도달까지 상당한 시간 필요
  - V4 완료 (11분) + 분석 (10분) = 21분
  - Total: 51분 경과 시점
  - 여전히 Trade #2 OPEN 가능성 높음
```

---

## ✅ 권장 실행 계획

### Timeline (다음 30분)

```yaml
18:32 (현재):
  - ✅ 근본 문제 문서화 완료
  - ⏳ V4 Bayesian 대기 (11분 남음)
  - ⏳ Trade #2 모니터링 계속

18:43 (V4 완료 예정):
  - 📊 V4 결과 즉시 분석
  - 🔍 EXIT_THRESHOLD 최적값 확인
  - 📈 성능 비교 (V4 vs 0.70 vs 0.2)

18:53 (분석 완료):
  - 🎯 Gap 백테스트 필요성 판단
  - If needed: Gap 백테스트 시작 (20분)
  - If not: EXPECTED_SIGNAL_RATE 배포 준비

19:13 (Gap 백테스트 완료, if needed):
  - ✅ 최종 EXIT_THRESHOLD 확정
  - 📋 배포 계획 업데이트

Trade #2 병렬 진행:
  - 5분마다 자동 체크
  - Exit 발생 시 즉시 분석
  - V4 + Gap 백테스트와 무관하게 진행
```

---

## 🎓 핵심 교훈

### 1. 백테스트 방법론 일관성

**문제**:
- 두 백테스트가 서로 다른 방법 사용
- 결과 비교 불가능

**교훈**:
- 백테스트 방법론은 **일관되어야** 함
- 파라미터 탐색 범위는 **연속적**이어야 함
- 이전 결과를 **무시하지 말 것**

### 2. 최적화 순서

**올바른 순서**:
```yaml
1. 단일 파라미터 최적화 (EXIT_THRESHOLD 단독)
   → 최적값 발견 (예: 0.2)

2. 주변 탐색 (0.15, 0.2, 0.25 정밀 탐색)
   → 최적값 재확인

3. 다른 파라미터 추가 (SL/TP/MaxHold)
   → 최적값 주변에서 조합 최적화

4. 전체 재검증
   → 최종 최적값 확정
```

**실제 진행**:
```yaml
1. EXIT_THRESHOLD 0.1-0.5 탐색 → 0.2 최적
2. ??? (Gap)
3. EXIT_THRESHOLD 0.70-0.80 탐색 → 0.70 최적
   (0.2 무시???)

→ 방법론 오류
```

### 3. V4 Bayesian의 가치

**V4가 중요한 이유**:
- **체계적 탐색**: Bayesian optimization
- **넓은 범위**: 11-dimensional space
- **충분한 iterations**: 220번
- **전체 데이터**: 90일

**V4 후 필요 작업**:
- V4가 탐색하지 않은 범위 (0.2-0.6) 검증
- 최종 최적값 확정

---

## 🚨 결론

### 근본 문제

**EXIT_THRESHOLD = 0.70은 불완전한 검증 결과**:
1. 백테스트 #1 (0.2 최적)과 백테스트 #2 (0.70 최적) 사이 gap
2. 두 백테스트 방법론 불일치
3. 0.2 vs 0.70 직접 비교 불가능

### 즉각 조치

**1. V4 Bayesian 완료 대기 (11분)**
- 가장 체계적인 최적화
- 0.60-0.85 범위 탐색

**2. V4 결과 기반 결정**
- V4 최적값 분석
- Gap 백테스트 필요성 판단

**3. Trade #2 모니터링 계속**
- 병렬 진행
- Exit pattern 검증

### 최종 목표

**체계적이고 근거 있는 EXIT_THRESHOLD 확정**:
- 모든 범위 검증 (0.1 - 0.85)
- 일관된 백테스트 방법론
- 명확한 성능 비교

---

**Status**: 🔍 **근본 원인 파악 완료 - V4 대기 중**
**Next**: V4 Bayesian 완료 시 즉시 분석 및 gap 메우기
**Timeline**: 11분 (V4) + 10분 (분석) + 20분 (gap 백테스트 if needed)

---

**Prepared by**: Root Cause Analysis Team
**Methodology**: 분석적, 확인적 사고 → 근본 문제 해결
**Principle**: "단순 증상 제거가 아닌 근본 문제 해결"
