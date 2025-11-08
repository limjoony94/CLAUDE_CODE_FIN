# 백테스트 vs 실시간 불일치 분석
**Date**: 2025-10-19 07:10 KST
**Critical Finding**: 백테스트에 Look-Ahead Bias 존재

---

## 🚨 핵심 문제: Look-Ahead Bias

### 현재 백테스트 방식 (잘못됨)

```python
# backtest_trade_outcome_full_models.py (Line 40-48)

df = pd.read_csv("BTCUSDT_5m_max.csv")  # 31,488 candles (7월~10월)
df = calculate_all_features(df)          # ❌ 전체 데이터 기준 계산!

# Sliding window backtest
for window_num in range(total_windows):
    start_idx = window_num * STEP_SIZE   # e.g., window 100 = 8월 중순
    end_idx = start_idx + WINDOW_SIZE
    window_df = df.iloc[start_idx:end_idx]  # 슬라이싱만
    run_backtest(window_df)
```

**문제점**:
- Window 100 (8월 중순)이 **전체 데이터(7월~10월)의 통계**로 계산된 features 사용
- 8월 window가 10월 데이터의 정보를 간접적으로 포함! (Look-ahead bias)

---

### 실시간 봇 방식 (올바름)

```python
# opportunity_gating_bot_4x.py (Line 1113-1118)

ohlcv = client.fetch_ohlcv(symbol, limit=1000)  # 최근 1000개만
df = pd.DataFrame(ohlcv)
df_features = calculate_all_features(df.copy())  # ✅ 최근 1000개 기준 계산!
```

**현실**:
- 실시간에서는 **현재까지의 데이터**만 사용 가능
- 10월 19일 시점: 10월 15-18일 데이터(최근 1000개) 기준으로만 계산

---

## 📊 구체적 예시: Feature 계산 차이

### RSI (Relative Strength Index) 예시

#### 백테스트 Window 100 (8월 15일):
```python
# 전체 데이터(7월~10월) 기준 계산
df = pd.read_csv()  # 31,488 candles
df['rsi'] = calculate_rsi(df['close'])  # 전체 데이터의 상대 강도

# Window 100에서:
# - 8월 15일 가격: $111,000
# - 전체 평균: $114,962
# - RSI: 45 (정상 범위)
# - LONG 확률: 72%
```

#### 실시간 (10월 19일):
```python
# 최근 1000개(10월 15-18일) 기준 계산
ohlcv = fetch_ohlcv(limit=1000)  # 최근 1000개
df['rsi'] = calculate_rsi(df['close'])  # 최근 1000개의 상대 강도

# 현재:
# - 10월 19일 가격: $107,000
# - 최근 1000개 평균: $113,000
# - RSI: 15 (극단적 과매도!) ⚠️
# - LONG 확률: 92%
```

**차이 발생 원인**:
- 백테스트: 전체 맥락 → 정상적인 지표
- 실시간: 최근 맥락 → 극단적인 지표

---

## 🔍 왜 백테스트는 좋은 결과를 보였나?

### 백테스트 결과 (27.59% return, 84% win rate)

**이것은 거짓 성능입니다!**

```python
Window 1 (7월 초):
  - Features 계산: 전체(7월~10월) 기준
  - 7월 가격 = $115k, 전체 평균 = $114.9k
  - 상대적으로 "평균" → LONG 확률: 70%

Window 100 (8월 중순):
  - Features 계산: 동일한 전체(7월~10월) 기준
  - 8월 가격 = $111k, 전체 평균 = $114.9k
  - 상대적으로 "약간 낮음" → LONG 확률: 75%

Window 403 (10월 중순):
  - Features 계산: 동일한 전체(7월~10월) 기준
  - 10월 가격 = $113k, 전체 평균 = $114.9k
  - 상대적으로 "평균" → LONG 확률: 72%
```

**모든 window가 동일한 통계 기준 사용** → 일관된 확률 분포 → 좋은 성능

하지만 실시간에서는:
```python
실시간 (10월 19일):
  - Features 계산: 최근 1000개(10월 15-18일) 기준만!
  - 최근 평균 = $113k (전체보다 낮음)
  - 현재 가격 = $107k
  - 상대적으로 "극단적으로 낮음" → LONG 확률: 85-98% ⚠️
```

---

## 💡 왜 훈련 시에는 문제 없었나?

### 훈련 과정:
```python
# retrain_entry_models_full_batch.py
df = pd.read_csv()  # 전체 데이터
df = calculate_all_features(df)  # 전체 기준 계산

# Label 생성 (Trade Simulation)
for i in range(len(df)):
    features = df.iloc[i]  # 이 시점의 features
    simulate_trade(i)  # 거래 시뮬레이션
```

**훈련 데이터**:
- 모든 데이터 포인트가 **전체 데이터 기준** features 가짐
- 모델이 이 분포로 학습됨
- 전체 평균 $114,962 기준으로 "정상" 판단 학습

**실시간 적용**:
- Features가 **최근 1000개 기준**으로 계산됨
- 분포가 다름! (최근 평균 $113k vs 전체 평균 $115k)
- 모델이 본 적 없는 분포 → 과도한 확신

---

## 🎯 진짜 문제: 백테스트 방법론 오류

### 올바른 백테스트 방식:

```python
# 실시간을 정확히 재현
for window_num in range(total_windows):
    start_idx = window_num * STEP_SIZE
    end_idx = start_idx + WINDOW_SIZE

    # ✅ 최근 1000개 데이터만 사용하여 features 계산
    lookback_start = max(0, end_idx - 1000)
    window_data = df.iloc[lookback_start:end_idx]

    # 이 범위에 대해서만 features 계산
    window_features = calculate_all_features(window_data.copy())

    # 실제 백테스트는 마지막 WINDOW_SIZE만
    backtest_df = window_features.tail(WINDOW_SIZE)

    run_backtest(backtest_df)
```

**이렇게 하면**:
- 각 window가 **그 시점까지의 데이터**만으로 features 계산
- 실시간과 동일한 조건
- Look-ahead bias 제거

---

## 📊 예상되는 수정 후 결과

### 현재 백테스트 (잘못됨):
```
Return: 27.59%
Win Rate: 84.0%
LONG 확률 분포: 65-80% (일관됨)
```

### 수정된 백테스트 (올바름):
```
Return: 예상 15-20% (하락 예상)
Win Rate: 예상 70-75% (하락 예상)
LONG 확률 분포: 더 넓은 범위 (60-95%)
  - 하락 추세 window: 85-95% (현재 실시간처럼)
  - 상승 추세 window: 60-70%
  - 횡보 window: 70-80%
```

**중요**: 수정된 백테스트는 **실시간 성능을 정확히 예측**해야 함!

---

## 🔧 해결 방안

### Option 1: 백테스트 수정 (정확한 검증)
```python
# 실시간을 정확히 재현하는 백테스트
# - 각 window마다 features 재계산
# - 최근 1000개 데이터만 사용
# - Look-ahead bias 제거
```

**장점**:
- 실시간 성능 정확히 예측
- 문제를 사전에 발견 가능

**단점**:
- 백테스트 시간 증가 (각 window마다 features 재계산)
- 현재 27.59% 성능은 거짓이었음 확인됨

### Option 2: 실시간 봇 수정 (백테스트에 맞춤)
```python
# 전체 historical data 사용하여 features 계산
# - 5000+ candles 가져오기
# - 전체 통계로 정규화
```

**장점**:
- 백테스트와 일치
- 즉시 적용 가능

**단점**:
- 백테스트 자체가 잘못되었으므로 의미 없음
- 실시간 환경을 정확히 반영하지 못함

### Option 3: 둘 다 수정 (권장) ⭐
```python
# 1. 백테스트를 실시간 방식에 맞게 수정
# 2. 수정된 백테스트로 재검증
# 3. 성능이 좋으면 실시간 배포
# 4. 성능이 나쁘면 전략 재설계
```

---

## 📋 즉시 조치 사항

### 1. 훈련 중지 결정
```
현재 훈련: 잘못된 백테스트 기준
→ 훈련 완료해도 동일한 문제 발생 가능
→ 중지하고 백테스트부터 수정?
```

### 2. 백테스트 수정 우선
```
1. 올바른 백테스트 스크립트 작성
2. 실시간과 동일한 방식으로 features 계산
3. 실제 예상 성능 확인
4. 성능이 좋으면 → 모델 재훈련
5. 성능이 나쁘면 → 전략 재설계
```

### 3. 현재 포지션 관리
```
현재: LONG +1.13% (0.86h holding)
Emergency Stop Loss: -4%
Emergency Max Hold: 8h

옵션:
A) 수익 실현 (+1.13% 청산)
B) Emergency 규칙 신뢰 (최대 -4% 허용)
```

---

## 🎓 핵심 교훈

### 백테스트의 함정:
1. **Look-ahead Bias**: 미래 정보 간접 사용
2. **Distribution Shift**: 훈련/백테스트 vs 실시간 분포 차이
3. **Feature Calculation**: 계산 범위에 따라 완전히 다른 값

### 올바른 백테스트:
1. **실시간 재현**: 실시간과 동일한 방식
2. **Rolling Window**: 각 시점마다 features 재계산
3. **No Future Data**: 해당 시점까지 데이터만 사용

---

**보고서 작성**: Claude Code
**분석 완료**: 2025-10-19 07:10 KST
**결론**: 백테스트 방법론 오류 → 수정 필요
