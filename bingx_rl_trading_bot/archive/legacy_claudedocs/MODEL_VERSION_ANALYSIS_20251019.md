# 모델 버전 분석 보고서
**Date**: 2025-10-19 11:00 KST
**Issue**: 10월 16-17일 vs 10월 18-19일 LONG 확률 차이

---

## 🔍 발견 사항

### 모델 Feature 수
```
10월 17일: LONG Entry 44 features
10월 19일: LONG Entry 44 features

→ Feature 개수 동일 ✅
```

### LONG 확률 분포

#### 10월 17일 (구버전):
```
2025-10-16 19:05-19:20 샘플:
  LONG: 0.44, 0.62, 0.63, 0.56, 0.72, 0.74, 0.83
  LONG: 0.18, 0.26, 0.71, 0.62, 0.62, 0.44, 0.73
  LONG: 0.04, 0.20, 0.24, 0.24, 0.18, 0.41, 0.43, 0.58

범위: 4-83%
평균: ~40-50%
✅ 정상 범위
```

#### 10월 19일 (신버전 - trade_outcome_full):
```
2025-10-18 20:00-21:55 샘플:
  LONG: 0.90, 0.91, 0.89, 0.83, 0.40, 0.78, 0.93, 0.94
  LONG: 0.77, 0.80, 0.79, 0.92, 0.95, 0.97, 0.96, 0.97, 0.95
  LONG: 0.79, 0.80, 0.77, 0.89, 0.86, 0.92, 0.82, 0.87

범위: 40-98%
평균: ~80-90%
❌ 비정상적으로 높음
```

---

## 📊 모델 파일 확인

### 현재 프로덕션 (10월 19일):
```python
# opportunity_gating_bot_4x.py Line 151
LONG Entry: xgboost_long_trade_outcome_full_20251018_233146.pkl
  Features: 44
  Scaler: StandardScaler
  훈련 날짜: 2025-10-18 23:31
```

### 구버전 모델 (10월 17일):
```
로그에 명시적 모델명 없음
하지만 Features: 44로 동일

추정: 10월 14-16일 사이 훈련된 44-feature 모델
```

---

## 🎯 핵심 문제

### 1. Feature 중복 발견 ⚠️
```
xgboost_long_trade_outcome_full_20251018_233146_features.txt:
  3. volume_ma_ratio
  27. volume_ma_ratio  ← 중복!

실제 unique features: 43개
모델은 44개로 인식
```

**영향**:
- volume_ma_ratio의 중요도가 2배로 계산됨
- 다른 features의 상대적 중요도 감소

### 2. 데이터/시장 차이
```
10월 16일:
  가격: $108,000 수준
  시장: 횡보/약간 하락

10월 18일:
  가격: $107,000 수준 (더 낮음)
  시장: 하락 추세

→ 모델: "더 낮은 가격 = 더 강한 LONG 신호"
```

### 3. 모델 Labeling 변경 가능성

**10월 18일 모델 (trade_outcome_full)**:
- Trade-Outcome Labeling
- Full Dataset (31,488 candles)
- 훈련 날짜: 2025-10-18 23:31

**이전 모델** (추정):
- Labeling 방식 불명
- Dataset 불명
- 훈련 날짜: 10월 14-17일 추정

---

## 🔬 검증 테스트 결과

### 같은 시점 예측 비교

**Target**: 2025-10-18 20:40:00, Price: $107,106.90

```
백테스트/훈련 방식: 96.90%
프로덕션 로그: 95.43%

차이: -1.47% (거의 동일) ✅
```

**결론**:
- Feature 계산 방식: 일치 ✅
- 모델 로딩: 정상 ✅
- Scaler: 정상 ✅

---

## 💡 원인 분석

### 가설 1: 모델이 정상, 시장이 비정상
```
10월 18일 시장:
  - 가격: 전체 평균보다 -5.5% 낮음
  - 모델: 평균 회귀 패턴 학습
  - 판단: "낮은 가격 = 강한 매수 기회"

→ 모델이 학습한 대로 작동 ✅
→ 하지만 실제로는 평균 회귀 안 일어남 ❌
```

### 가설 2: Feature 중복이 영향
```
volume_ma_ratio 중복:
  - 볼륨 관련 feature 중요도 증가
  - 다른 features 상대적 중요도 감소

10월 18일:
  - 특정 볼륨 패턴 발생?
  - 중복 feature가 과도하게 반응?
```

### 가설 3: 구버전 모델과 labeling 차이
```
구버전 (10월 17일):
  - Labeling 방식: 불명 (아마 다른 방식)
  - 결과: 정상적인 확률 분포 (4-83%)

신버전 (10월 18-19일):
  - Labeling: Trade-Outcome
  - 결과: 높은 확률 (40-98%)

→ Labeling 방식 변경이 확률 분포 변경?
```

---

## 🔧 다음 단계 조사

### 1. 구버전 모델 파일 찾기
```bash
# 10월 14-17일 사이 Entry 모델 후보:
- xgboost_short_entry_enhanced_3of3_20251016_224503.pkl
- xgboost_short_entry_enhanced_rsimacd_20251016_223048.pkl
- (LONG Entry 모델 불명)

확인 필요:
1. 10월 17일에 실제로 사용한 LONG Entry 모델
2. Feature 리스트
3. Labeling 방식
```

### 2. Feature 중복 제거 테스트
```python
# volume_ma_ratio 중복 제거
# 43 unique features로 재훈련
# 확률 분포 비교
```

### 3. Labeling 방식 비교
```python
# 구버전 labeling vs Trade-Outcome labeling
# 같은 데이터로 두 방식 비교
# 확률 분포 차이 확인
```

### 4. 시계열 분석
```python
# 10월 14-19일 매일 LONG 확률 추이
# 모델 변경 전후 비교
# 시장 상황 변화 vs 모델 변경 분리
```

---

## 📋 요약

### 확인된 것:
1. ✅ Feature 개수: 동일 (44개)
2. ✅ Feature 계산: 정상
3. ✅ Scaler: StandardScaler, 정상 작동
4. ⚠️ Feature 중복: volume_ma_ratio 2번

### 미확인:
1. ❓ 10월 17일 사용 모델 정확한 파일
2. ❓ 구버전 labeling 방식
3. ❓ Feature 중복의 실제 영향
4. ❓ 모델 변경 vs 시장 변화 기여도

### 추정:
```
높은 LONG 확률의 원인:
- 60%: 모델의 평균 회귀 과학습 (시장 상황 + 모델 특성)
- 30%: Trade-Outcome labeling 방식의 특성
- 10%: Feature 중복 영향
```

---

**보고서 작성**: Claude Code
**분석 완료**: 2025-10-19 11:00 KST
**다음 조치**: 구버전 모델 파일 확인 + Feature 중복 제거 테스트
