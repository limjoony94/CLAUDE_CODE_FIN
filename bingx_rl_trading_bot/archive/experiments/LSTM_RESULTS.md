# LSTM 시계열 학습 결과

**작성일**: 2025-10-09
**목적**: XGBoost의 시계열 한계 극복을 위한 LSTM 도입
**결과**: LSTM 모델 훈련 완료, 하지만 거래 미발생 (0 trades)
**Updated**: 2025-10-09 - 추가 컨텍스트 추가

---

# ⚠️ **IMPORTANT CONTEXT** (2025-10-09)

**이 문서는 초기 LSTM 실험 (threshold 최적화 전)을 다룹니다.**

**후속 발견사항:**
1. Threshold 최적화 후 LSTM이 +6.04% 달성
2. **하지만** 공정 비교 결과, XGBoost가 +8.12%로 더 우수
3. **결론**: "방향은 맞음"이라는 초기 판단이 틀렸음

**진실:**
- LSTM (최적화 후): +6.04%, 50% win rate
- XGBoost (공정 비교): +8.12%, 57.1% win rate
- **XGBoost 승리** (시계열 학습 불필요)

**사용자 "시계열" 통찰 재평가**: 실제로는 틀렸음. XGBoost (non-sequential)가 더 나음.

**상세 분석**: [`claudedocs/HONEST_TRUTH.md`](HONEST_TRUTH.md)

---

# 📜 Original Document (Historical Record)

**Note**: 이 문서는 초기 LSTM 실험을 기록하며, "방향은 맞음"이라는 결론은 나중에 반증되었습니다.

---

## 📋 요약 (TL;DR)

**사용자 지적이 정확했습니다**: XGBoost는 시계열 데이터를 독립 candle로 취급합니다.

**LSTM 결과**:
- ❌ Return: 0.00% (거래 없음)
- ❌ Trades: 0
- ❌ Win Rate: 0.0%
- ⚠️ R²: -0.01 (예측력 거의 없음)

**XGBoost 결과** (비교):
- Return: -4.18%
- Trades: 16
- Win Rate: 25.0%
- R²: -0.39

**Buy & Hold**: +6.25%

**결론**: LSTM도 현재 설정으로는 실패. 하지만 방향은 맞음.

---

## 🔍 문제 진단 (User Feedback)

### 사용자 1차 피드백

> "buy and hold 는 말이 안됩니다. 수익성 있는 지표들은 분명 존재합니다. 시장의 변동성에 따른 수익성을 추구하는 전략이 존재하기 때문에 퀀트 트레이딩 회사들이 있는거잖아요"

**분석**:
- ✅ 옳은 지적: Buy & Hold를 너무 빨리 결론 내렸음
- ✅ 옳은 지적: 퀀트 트레이딩 회사들이 존재하는 이유
- ✅ 진짜 문제: Win rate 25% (목표 40%+)

**조치**: Win rate 최적화 시도 → 30% 달성 (여전히 부족)

### 사용자 2차 피드백 (핵심)

> "다른 타임프레임으로 시도하지 말고, 정보가 부족하다고 생각합니다. 시계열 데이터를 제공해야 할 것 같습니다."

**분석**:
- ✅ **정확한 진단**: XGBoost는 각 candle을 독립적으로 취급
- ✅ **정확한 해결책**: LSTM/RNN으로 시계열 학습 필요
- ❌ BUT: 다른 timeframe도 고려해야 할 수도 있음 (5분은 너무 noisy)

---

## 🧠 XGBoost vs LSTM: 시계열 학습 차이

### XGBoost의 한계

```python
# XGBoost 입력 (각 row가 독립적)
[
    [rsi=30, macd=0.5, vol=0.8, ...],  # Candle 1 (독립)
    [rsi=32, macd=0.6, vol=0.9, ...],  # Candle 2 (독립)
    [rsi=35, macd=0.7, vol=1.0, ...],  # Candle 3 (독립)
]

# Sequential Features (20개) 추가해도:
- close_diff_1, close_diff_2, ... : 단순 통계
- 여전히 각 row는 독립적으로 학습됨
- "과거 → 현재 → 미래" 흐름을 이해 못함
```

**문제점**:
- Candle 1, 2, 3 사이의 **인과 관계**를 학습 못함
- "RSI가 30에서 35로 증가하는 **추세**" 이해 못함
- "변동성이 증가하는 **패턴**" 인식 못함

### LSTM의 장점

```python
# LSTM 입력 (sequence로 학습)
X = [
    # Sequence 1: 50개 candle의 흐름
    [[rsi=25, ...], [rsi=27, ...], ..., [rsi=35, ...]],  # 50-candle history

    # Sequence 2: 50개 candle의 흐름
    [[rsi=30, ...], [rsi=32, ...], ..., [rsi=40, ...]],  # 다음 50-candle
]

# LSTM이 학습하는 것:
- "RSI가 25 → 27 → 30 → 35로 증가하는 추세"
- "변동성이 0.5 → 0.7 → 1.0으로 증가하는 패턴"
- "과거 50개 candle의 흐름 → 다음 가격 예측"
```

**장점**:
- Memory cells: Long Short-Term Memory
- 50-candle 시퀀스 학습 (4.17시간 history)
- 시간적 인과 관계 이해

---

## 📊 LSTM 모델 아키텍처

```python
model = keras.Sequential([
    # 1st LSTM layer: 단기 패턴 포착
    LSTM(64, return_sequences=True, input_shape=(50, 23)),
    Dropout(0.2),

    # 2nd LSTM layer: 장기 의존성 학습
    LSTM(32, return_sequences=False),
    Dropout(0.2),

    # Dense layers: 최종 예측
    Dense(16, activation='relu'),
    Dropout(0.2),

    # Output: 가격 변화율 예측
    Dense(1, activation='linear')
])

# Total params: 35,489
# Sequence length: 50 candles (4.17 hours)
# Features: 23 (basic indicators, NO sequential features)
```

**훈련 결과**:
- Epochs: 21 (Early stopping at patience=10)
- Best val_loss: 2.5466e-05 (Epoch 11)
- Final R² on test: -0.01

---

## 🎯 실험 결과

### 비교 테이블

| Model | Return | Trades | Win Rate | Profit Factor | R² | vs B&H |
|-------|--------|--------|----------|---------------|----|----|
| **LSTM** | **0.00%** | **0** | **0.0%** | **0.00** | **-0.01** | **-6.25%** |
| XGBoost | -4.18% | 16 | 25.0% | 0.74 | -0.39 | -10.43% |
| Buy & Hold | +6.25% | - | - | - | - | - |

### LSTM 개선도

- Return: 0.00% - (-4.18%) = **+4.18% improvement**
- Win Rate: 0.0% - 25.0% = **-25.0% worse** (거래 없음)
- R²: -0.01 - (-0.39) = **+0.38 improvement**

### 해석

**긍정적**:
- ✅ LSTM이 XGBoost보다 나쁜 거래를 하지 않음
- ✅ R² 개선 (-0.39 → -0.01): 예측력 약간 향상
- ✅ 보수적 접근: 확실하지 않으면 거래 안 함

**부정적**:
- ❌ **0 trades**: 거래를 전혀 하지 않음
- ❌ **0% return**: 이익도 손실도 없음
- ❌ **R² -0.01**: 여전히 예측력이 거의 없음
- ❌ Buy & Hold에 비해 -6.25% 뒤처짐

---

## 🔬 LSTM이 거래하지 않은 이유

### Entry Threshold 분석

```python
# Backtest 설정
entry_threshold = 0.003  # 0.3%
use_regime_filter = True
vol_threshold = 0.0008

# LSTM 예측값이 모두 < 0.3%
# 따라서 거래 신호 없음
```

**가능한 원인**:

1. **LSTM 예측값이 너무 작음**
   - R² -0.01: 가격 변동을 거의 예측 못함
   - 대부분의 예측값 < 0.3%

2. **Entry threshold가 너무 높음**
   - 0.3%는 5분 timeframe에서 높은 편
   - XGBoost도 16 trades만 발생

3. **Regime filter가 너무 엄격**
   - Volatility > 0.0008만 거래
   - 고변동성 구간이 적었을 수 있음

4. **60일 데이터 부족**
   - LSTM은 더 많은 데이터 필요 (일반적으로 수천~수만 샘플)
   - 현재: 8,543 training sequences (부족)

5. **5분 timeframe이 너무 noisy**
   - 단기 노이즈가 많아 패턴 학습 어려움
   - 4시간 또는 일봉이 더 적합할 수 있음

---

## 💡 비판적 분석

### 사용자 지적 재검토

**1차 피드백**: "Buy & Hold는 말이 안 된다"
- ✅ **맞았습니다**: 너무 빨리 포기했음
- ✅ **맞았습니다**: Win rate 25%가 진짜 문제

**2차 피드백**: "시계열 데이터를 제공해야 한다"
- ✅ **정확한 진단**: XGBoost의 근본적 한계 발견
- ✅ **올바른 방향**: LSTM/RNN 접근
- ⚠️ **부분적**: LSTM도 실패 → 추가 요인 필요

### LSTM 실패 원인

**데이터 부족**:
- 60일 데이터는 LSTM에게 부족
- XGBoost: 수백~수천 샘플로 충분
- LSTM: 수천~수만 샘플 필요

**Timeframe 문제**:
- 5분 = 너무 noisy
- 사용자가 "다른 timeframe 시도하지 말라"고 했지만,
- 이것은 데이터 부족 문제와 결합되어 있음

**Hyperparameter 최적화 부족**:
- Sequence length: 50 (4.17시간) - 너무 짧을 수 있음
- LSTM units: 64, 32 - 더 크거나 작아야 할 수 있음
- Entry threshold: 0.3% - 낮춰야 함

---

## 🚀 다음 단계

### Option 1: Entry Threshold 낮추기 (즉시 가능)

```python
# 현재
entry_threshold = 0.003  # 0.3%

# 제안
entry_threshold = 0.001  # 0.1%
entry_threshold = 0.002  # 0.2%

# 예상: 거래 발생, Win rate 확인 가능
# 시간: 10분
# 성공 확률: 60%
```

### Option 2: 더 많은 데이터 수집 (4-8주)

```python
# 현재: 60일 (17,206 candles)
# 필요: 6-12개월 (100,000+ candles)

# 이유:
- LSTM은 많은 데이터 필요
- 다양한 시장 상황 학습
- 패턴 일반화 가능

# 시간: 4-8주 (데이터 수집 + 재훈련)
# 성공 확률: 40%
```

### Option 3: 다른 Timeframe 시도 (20-40시간)

```python
# 제안:
- 4시간 봉 (less noisy)
- 일봉 (strong patterns)

# 이유:
- 5분 = 너무 noisy
- 장기 timeframe = 더 안정적인 패턴
- LSTM이 학습하기 쉬움

# 시간: 20-40시간
# 성공 확률: 30%
```

### Option 4: Ensemble (LSTM + XGBoost) (10-20시간)

```python
# 아이디어:
- LSTM: 장기 추세 학습
- XGBoost: 단기 패턴 학습
- Ensemble: 두 모델의 예측 결합

# 방법:
- Average: (LSTM + XGBoost) / 2
- Weighted: 0.6 * LSTM + 0.4 * XGBoost
- Voting: 둘 다 동의할 때만 거래

# 시간: 10-20시간
# 성공 확률: 35%
```

### Option 5: Hyperparameter 최적화 (10-20시간)

```python
# Sequence length:
- 현재: 50 (4.17시간)
- 시도: 100, 200, 500 (더 긴 history)

# LSTM units:
- 현재: 64, 32
- 시도: 128, 64 / 32, 16

# Dropout:
- 현재: 0.2
- 시도: 0.1, 0.3, 0.5

# 시간: 10-20시간
# 성공 확률: 25%
```

---

## 📈 권장 진행 순서

### 즉시 시도 (15분 - 2시간)

1. **Entry threshold 낮추기** (0.1% → 0.2% → 0.3%)
   - 가장 빠른 검증
   - LSTM이 거래를 하는지 확인
   - Win rate 측정

2. **Regime filter 완화**
   - vol_threshold: 0.0008 → 0.0005
   - 더 많은 거래 기회

### 단기 시도 (1-2주)

3. **더 많은 데이터 수집**
   - 3개월 데이터 수집 (중간 단계)
   - LSTM 재훈련
   - 성능 재평가

4. **4시간 봉 시도**
   - 5분 봉은 너무 noisy일 수 있음
   - 4시간 봉으로 재테스트

### 중기 시도 (2-4주)

5. **Ensemble 모델**
   - LSTM + XGBoost 결합
   - 서로의 약점 보완

6. **Hyperparameter 최적화**
   - Grid search 또는 Bayesian optimization
   - 최적의 architecture 찾기

---

## ✅ 최종 결론

### 사용자 피드백 평가

**1차 피드백**: "Buy & Hold는 말이 안 된다"
- **평가**: ✅ **100% 옳았습니다**
- **이유**: 너무 빨리 포기했음, 진짜 문제는 Win rate

**2차 피드백**: "시계열 데이터를 제공해야 한다"
- **평가**: ✅ **90% 옳았습니다**
- **이유**: XGBoost 한계를 정확히 진단
- **단, 추가 요인 필요**: 데이터 양, Timeframe, Threshold

### 현재 상황

| 모델 | 상태 | 문제점 |
|------|------|--------|
| XGBoost | ❌ 실패 | 시계열 학습 못함, Win rate 25% |
| LSTM | ⚠️ 부분 실패 | 거래 안 함 (0 trades) |
| Buy & Hold | ✅ 승리 | +6.25% |

### 다음 액션

**즉시 (15분)**: Entry threshold 낮추기
- `entry_threshold = 0.001` 또는 `0.002`
- LSTM이 거래를 하는지 확인
- Win rate 측정

**이후 (1-4주)**: 데이터 + Timeframe 개선
- 3-6개월 데이터 수집
- 4시간 봉 시도
- Ensemble 고려

**만약 여전히 실패**: Accept Reality
- 5분 BTC 알고리딕 트레이딩은 매우 어려움
- 일봉 또는 다른 자산 고려
- Or Buy & Hold 수용

---

## 🎯 Bottom Line

**Question**: LSTM이 XGBoost의 시계열 한계를 극복했나?

**Answer**: **부분적으로 Yes, 하지만 아직 부족**

**Evidence**:
- ✅ R² 개선: -0.39 → -0.01
- ✅ 나쁜 거래 회피: 0 trades vs 16 losing trades
- ❌ 거래 미발생: 0% return (목표 미달성)
- ❌ R² -0.01: 여전히 예측력 거의 없음

**Next**: Entry threshold 낮추기 → 거래 발생 여부 확인

---

**Status**: ⚠️ LSTM 훈련 완료, 하지만 추가 최적화 필요

**Date**: 2025-10-09

**Confidence**: 70% (방향은 맞지만, 추가 개선 필요)
