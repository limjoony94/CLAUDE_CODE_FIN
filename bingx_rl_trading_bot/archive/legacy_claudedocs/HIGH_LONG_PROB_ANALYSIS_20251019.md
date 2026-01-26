# 높은 LONG 확률 분석 보고서
**Date**: 2025-10-19 06:50 KST
**Issue**: Entry LONG 확률이 지속적으로 70-98% 범위 (비정상적으로 높음)
**Status**: ⚠️ **ROOT CAUSE IDENTIFIED - DISTRIBUTION SHIFT**

---

## 🔍 발견된 문제

### 1. **높은 LONG 확률 (지난 3시간)**
```
03:53: 70.66%
04:00: 86.14%
04:35: 91.66%
05:00: 90.36%
05:40: 95.43%
05:50: 98.35% ⚠️ PEAK
06:40: 82.19%
06:45: 87.39% (현재)
```

**정상 범위**: 백테스트에서는 0.65 threshold → 신호 발생 시 평균 70-80% 범위
**현재 상황**: 거의 모든 시점에서 85%+ (비정상적으로 높음)

---

## 🎯 ROOT CAUSE 분석

### **분포 변화 (Distribution Shift)** 문제 발견

#### **훈련 데이터 가격 분포**:
```
기간: 2025-07-01 ~ 2025-10-15 (30,517 candles, 105일)
가격 범위: $103,502 ~ $125,977
평균 가격: $114,962
중앙값: $115,227
표준편차: $4,115

최근 1000개 캔들 (2025-10-12 ~ 2025-10-15):
  가격 범위: $109,901 ~ $115,758
  평균: $113,211
```

#### **현재 실시간 가격**:
```
현재 가격: $107,017
훈련 데이터 내 백분위: 0.7% ⚠️
```

### **핵심 발견**:
> **현재 가격 ($107,017)은 훈련 데이터의 하위 0.7%에 해당**
>
> 모델은 훈련 기간 동안 거의 이 가격대를 본 적이 없음!

---

## 📊 왜 이런 일이 발생했는가?

### **모델의 관점**:
1. **훈련 데이터**: BTC가 $110k-$125k 범위에서 거래됨 (7월-10월)
2. **현재 가격**: $107k (훈련 범위의 최저점 근처)
3. **모델 판단**: "이건 할인가다!" → **높은 LONG 확률**

### **실제 상황**:
- 10월 15일 이후 BTC가 하락 추세
- 훈련 데이터 종료 시점: $113k 수준
- 현재: $107k (-5.6% 하락)
- 모델이 경험하지 못한 가격 영역

---

## ⚠️ 위험 평가

### **현재 위험 요소**:

1. **하락 추세 지속 가능성**:
   - BTC가 $100k, $95k까지 하락 가능
   - 모델은 $103k 이하 가격을 본 적 없음
   - 계속 LONG 신호를 줄 가능성 (all the way down)

2. **모델 한계**:
   - 훈련 범위: $103k-$126k
   - 범위 밖 예측: 신뢰도 낮음
   - **Extrapolation 문제** (범위 밖 외삽)

3. **False Confidence**:
   - 높은 확률 (85-98%) ≠ 높은 정확도
   - 모델이 본 적 없는 상황 → 과신

### **현재 성과** (10월 17일 이후):
```
총 거래: 2건 (현재 포지션 제외)
승률: 50% (1승 1패)
총 손익: +$15.67 (+2.8%)

Trade 1: -$11.12 (-5.05%) - Emergency Stop Loss
Trade 2: +$26.79 (+11.77%) - Emergency Max Hold (8.7h)

현재 포지션: LONG 0.0108 BTC at $106,824.5
  현재 손익: +$2.08 (+1.01%)
  보유 시간: 0.86h
```

**분석**:
- 승률 50% (예상 84%보다 훨씬 낮음)
- 2건 중 1건 emergency stop loss 발동
- 가격 하락 추세에서 LONG 전략이 고전 중

---

## 🔧 해결 방안

### **Option 1: 긴급 봇 중지** ⚠️ RECOMMENDED
```
이유:
- 모델이 현재 시장 상황을 제대로 이해하지 못함
- 분포 변화 (distribution shift) 문제 심각
- 추가 손실 위험 높음

조치:
1. 봇 즉시 중지
2. 현재 포지션 수동 관리 (emergency stop loss -4% 설정되어 있음)
3. 재훈련 후 재개
```

### **Option 2: 모델 재훈련** (권장)
```
훈련 데이터 업데이트:
- 최근 데이터 추가 (10월 15일 ~ 현재)
- 현재 가격 범위 ($107k) 포함
- 하락 추세 패턴 학습

예상 효과:
- 현재 가격대에 대한 적절한 판단
- 과도한 LONG 신호 완화
- 더 정확한 확률 추정
```

### **Option 3: Threshold 상향 조정** (임시)
```
현재 LONG Threshold: 0.65
임시 상향: 0.85 또는 0.90

효과:
- LONG 진입 빈도 감소
- 더 확실한 신호만 진입

단점:
- 근본 문제 해결 아님
- 기회 손실 가능
```

### **Option 4: 가격 범위 필터 추가** (안전장치)
```python
# 봇에 추가
MIN_PRICE_THRESHOLD = 103500  # 훈련 데이터 최저가
CURRENT_PRICE = 107017

if CURRENT_PRICE < MIN_PRICE_THRESHOLD:
    logger.warning("⚠️ Price below training range - signal unreliable")
    # 신호 무시 또는 보수적 진입
```

---

## 🔬 기술적 세부 사항

### **모델 구조**:
```
Model: XGBoost
Features: 44 (LONG Entry)
Estimators: 300
Max Depth: 6
Trained: 2025-10-18 23:31:46
```

### **Feature 분석**:
```
현재 실시간 Feature 통계:
  Min: -13.58
  Max: 107,218.82
  Mean: 7,300.94

훈련 데이터 Feature 분포와 비교 필요
```

### **예측 메커니즘**:
```
Model Prediction [class_0, class_1]: [0.1261, 0.8739]
→ LONG Probability: 87.39%

해석:
- class_0 (NO TRADE): 12.61%
- class_1 (LONG): 87.39%
- 모델이 매우 확신하고 있음 (하지만 훈련 범위 밖)
```

---

## 📋 API 구성 문제

### **발견된 문제**:
```yaml
# config/api_keys.yaml
bingx:
  testnet:  # ← "testnet"이라고 표시되어 있지만
    api_key: "NyXnyvNW..."  # ← 실제로는 mainnet keys

# opportunity_gating_bot_4x.py:115
return config.get('bingx', {}).get('testnet', {})  # testnet 섹션에서 로드

# opportunity_gating_bot_4x.py:121
USE_TESTNET = False  # ⚠️ MAINNET으로 연결
```

### **현재 상태**:
- Config 파일에는 "testnet"이라고 표시
- 하지만 실제로는 mainnet 엔드포인트 사용
- 봇이 정상 작동 중 → 해당 키들이 실제로는 mainnet keys

### **권장 수정**:
```yaml
# config/api_keys.yaml (수정안)
bingx:
  mainnet:  # ← 정확한 레이블
    api_key: "NyXnyvNW..."
    secret_key: "XLi5Q6lj..."

  testnet:
    api_key: "your_actual_testnet_key"
    secret_key: "your_actual_testnet_secret"
```

```python
# opportunity_gating_bot_4x.py:115 (수정안)
def load_api_keys():
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            mode = 'mainnet' if not USE_TESTNET else 'testnet'  # ✅ 명확한 로직
            return config.get('bingx', {}).get(mode, {})
    return {}
```

---

## 💡 즉시 조치 사항

### **1. 긴급 결정 필요**:
```
봇을 중지할 것인가?
OR
계속 실행하되 리스크 수용?
```

### **2. 현재 포지션 관리**:
```
현재: LONG 0.0108 BTC at $106,824.5 (+1.01%)
설정: Emergency Stop Loss -4%, Max Hold 8h

옵션:
A) 현재 +1% 수익으로 수동 청산 (안전)
B) Emergency 규칙 신뢰 (최대 -4% 손실 허용)
C) 즉시 봇 중지 + 수동 관리
```

### **3. 중장기 계획**:
```
1주일 내:
  - 최신 데이터로 모델 재훈련
  - 현재 가격 범위 포함
  - 백테스트 재검증

2주일 내:
  - 분포 변화 감지 시스템 추가
  - 가격 범위 필터 구현
  - 모니터링 강화
```

---

## 📊 요약

### **핵심 발견**:
1. ✅ **높은 LONG 확률 원인 규명**: 분포 변화 (distribution shift)
2. ✅ **모델 한계 확인**: 훈련 범위 ($103k-$126k) 밖 예측 신뢰도 낮음
3. ✅ **현재 리스크 평가**: 가격 하락 추세 시 모델 성능 저하
4. ⚠️ **API 구성 혼란**: Config 레이블링 문제 (기능상 문제 없음)

### **모델 상태**:
- 🟡 **작동 중**: 정상적으로 예측 생성
- 🔴 **신뢰도**: 현재 가격 범위에서 낮음
- ⚠️ **위험도**: 중-높음 (하락 추세 지속 시)

### **권장 조치**:
1. **즉시**: 봇 중지 고려 (또는 매우 보수적 운영)
2. **단기**: 모델 재훈련 (최신 데이터 포함)
3. **중기**: 분포 변화 감지 시스템 구축

---

**보고서 작성**: Claude Code
**분석 기준 시간**: 2025-10-19 06:50 KST
**데이터 출처**:
- 훈련 데이터: `data/historical/BTCUSDT_5m_max.csv`
- 실시간 로그: `logs/opportunity_gating_bot_4x_20251019.log`
- 상태 파일: `results/opportunity_gating_bot_4x_state.json`
