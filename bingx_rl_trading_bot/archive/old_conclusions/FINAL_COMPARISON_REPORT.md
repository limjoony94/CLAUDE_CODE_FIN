# 최종 비교 분석 보고서: XGBoost 트레이딩 시스템 (4-Way)

**작성일**: 2025-10-09
**분석 대상**: BUGGY → FIXED → IMPROVED → REGRESSION (4-Way 비교)
**목표**: 비판적 사고 및 개선 과정의 종합 평가

---

## 📊 Executive Summary

### 진행 과정
```
BUGGY (원본)
  ↓ Phase 0: 백테스팅 버그 수정
FIXED (버그 수정)
  ↓ Phase 1B + 2A: lookahead=288 + SMOTE
IMPROVED (개선)
  ↓ Option B: 회귀 문제로 전환
REGRESSION (회귀 모델)
```

### 핵심 성과 비교

| 지표 | BUGGY | FIXED | IMPROVED | REGRESSION | 최종 평가 |
|------|-------|-------|----------|------------|----------|
| **Model Type** | 분류 | 분류 | 분류 | 회귀 | ✅ |
| **lookahead** | 5 (25m) | 60 (5h) | 288 (24h) | 48 (4h) | ✅ |
| **SNR** | 0.95 | 1.37 | 1.26 | **2.31** | 🏆 |
| **Test Return** | -1051.80% | -2.05% | -2.80% | **0.00%** | ⚠️ |
| **Test Trades** | 867 | 1 | 9 | **0** | ❌ |
| **Win Rate** | 2.3% | 0.0% | 66.7% | **0.0%** | ⚠️ |
| **Direction Acc** | - | - | - | **64.6%** | ✅ |
| **Overfitting** | 0.99x | 1.40x | 2.23x | **N/A** | ✅ |

**결론**:
- ✅ 버그 수정 완료, SNR 목표 달성 (2.31)
- ✅ 회귀 모델로 클래스 불균형 근본 해결
- ❌ 임계값 과도하게 보수적 → 거래 0회 발생

---

## 🔍 Part 1: 4-Way 상세 비교

### 1.1 모델 아키텍처 비교

#### BUGGY 버전 (분류)
```python
Model Type: XGBoost Classifier
lookahead = 5 (25분)
threshold = 0.002 (0.2%)
confidence_threshold = 0.55
Class Balancing: 미사용
SNR = 0.2% / 0.2095% = 0.95 ❌

문제점:
- SNR < 1.0 → 신호보다 노이즈가 강함
- threshold < transaction_fee → 구조적으로 손실
- 클래스 불균형 미해결 (HOLD 78%)
- 5개 백테스팅 버그
```

#### FIXED 버전 (분류)
```python
Model Type: XGBoost Classifier
lookahead = 60 (5시간)
threshold = 0.01 (1.0%)
confidence_threshold = 0.65
Class Balancing: scale_pos_weight=7.0
SNR = 1.0% / 0.73% = 1.37 ✅

개선점:
- SNR > 1.0 → 신호가 노이즈보다 강함
- threshold > transaction_fee → 이론적으로 수익 가능
- 모든 백테스팅 버그 수정 완료

한계:
- 여전히 HOLD 86.3% 예측
- 거래 1회만 발생 (너무 보수적)
- scale_pos_weight로는 불균형 완전 해결 불가
```

#### IMPROVED 버전 (분류 + SMOTE)
```python
Model Type: XGBoost Classifier
lookahead = 288 (24시간)
threshold = 0.02 (2.0%)
confidence_threshold = 0.60
Class Balancing: SMOTE (synthetic sampling)
SNR = 2.0% / 1.59% = 1.26 ⚠️

개선점:
- SMOTE로 클래스 불균형 직접 해결
  - Before: {SHORT: 1309, HOLD: 9542, LONG: 990}
  - After: {SHORT: 6679, HOLD: 9542, LONG: 6679}
- Win Rate 66.7% 달성
- 거래 활성화 (1 → 9 trades)

한계:
- 과적합 심화 (Train 96% vs Test 43%, 2.23x)
- SMOTE 합성 샘플에 의존 → 실제 시장과 괴리
- 수익률 오히려 악화 (-2.05% → -2.80%)
```

#### REGRESSION 버전 (회귀) ⭐
```python
Model Type: XGBRegressor (회귀)
lookahead = 48 (4시간)
threshold = ±1.5%
confidence_multiplier = 0.5 (동적 임계값)
Class Balancing: N/A (회귀 문제)
SNR = 1.5% / 0.645% = 2.31 🏆

근본적 개선:
✅ 클래스 불균형 문제 소멸 (분류 → 회귀)
✅ SMOTE 불필요 → 과적합 위험 감소
✅ 실제 수익률 직접 예측
✅ SNR 2.31 목표 달성 (>2.0)
✅ 동적 임계값 조정 가능

치명적 한계:
❌ 거래 0회 발생 (Test Trades: 0)
❌ 임계값이 실제 분포 대비 과도
   - Mean Return: 0.020%
   - Std Return: 0.645%
   - Thresholds: ±1.5% = ±2.3σ
   - P(|return| > 1.5%) ≈ 4%
❌ 모델이 소규모 수익률 예측 학습
   → 임계값 초과하지 못함
```

### 1.2 타겟 분포 및 예측 분포 비교

#### 타겟 분포 (실제 레이블)
```
BUGGY (lookahead=5, threshold=0.2%):
  LONG:  11.3%
  HOLD:  78.0%
  SHORT: 10.7%

FIXED (lookahead=60, threshold=1.0%):
  LONG:  7.0%
  HOLD:  86.3%
  SHORT: 6.7%

IMPROVED (lookahead=288, threshold=2.0%):
  LONG:  10.8%
  HOLD:  78.7%
  SHORT: 10.5%

REGRESSION (lookahead=48, threshold=1.5%):
  Target Type: 연속값 (수익률)
  Mean: 0.020%
  Std: 0.645%
  Range: -8.47% ~ +8.92%
```

#### 예측 분포 (실제 예측)
```
BUGGY: (버그로 측정 불가)
FIXED: HOLD 99%+ (너무 보수적)
IMPROVED: 균형 잡힌 예측 (SMOTE 효과)
REGRESSION:
  실제 신호 분포 (Test):
  - SHORT: 2.1%
  - HOLD: 95.9%
  - LONG: 1.9%

  예측 신호 분포 (Test):
  - HOLD: 100.0% ❌

  → 모든 예측이 ±1.5% 임계값 이내
```

### 1.3 성능 지표 상세 비교

#### Test Return (손익률)
```
BUGGY:     -1,051.80% ❌ (수학적으로 불가능)
FIXED:     -2.05% ⚠️ (손실이지만 정상 범위)
IMPROVED:  -2.80% ⚠️ (FIXED보다 악화)
REGRESSION: 0.00% ⚠️ (거래 없음)
```

#### Test Trades (거래 빈도)
```
BUGGY:     867 trades ❌ (과도한 거래)
FIXED:     1 trade ❌ (너무 보수적)
IMPROVED:  9 trades ⚠️ (개선되었으나 낮음)
REGRESSION: 0 trades ❌ (치명적 실패)
```

#### Test Win Rate (승률)
```
BUGGY:     2.3% ❌ (거의 모든 거래 손실)
FIXED:     0.0% ❌ (거래 1회, 손실)
IMPROVED:  66.7% ✅ (9회 중 6회 승리)
REGRESSION: 0.0% ❌ (거래 없어 측정 불가)
```

#### Direction Accuracy (방향 정확도)
```
REGRESSION: 64.6% ✅ (회귀 모델 특화 지표)
- 예측 수익률의 부호와 실제 부호 일치율
- 64.6%는 양호한 수준
- 하지만 거래로 연결 안됨
```

#### Overfitting (과적합 비율)
```
BUGGY:     Train 78% / Test 79% = 0.99x ✅
FIXED:     Train 87% / Test 62% = 1.40x ⚠️
IMPROVED:  Train 96% / Test 43% = 2.23x ❌
REGRESSION: N/A (회귀는 RMSE 기준)
           Train RMSE 0.68% / Test RMSE 0.69% ≈ 0.99x ✅
```

---

## 📈 Part 2: Phase별 효과 분석

### Phase 0: 백테스팅 버그 수정 (BUGGY → FIXED)

#### 수정 내역
1. **HOLD 의미 복원**: "청산" → "포지션 유지"
2. **강제 청산 추가**: balance < 10% → 거래 중단
3. **수수료 계산 수정**: leverage 이중 적용 제거
4. **PnL 계산 정규화**
5. **포지션 관리 로직 수정**

#### 효과 측정
```
Return: -1051.80% → -2.05% (수학적 정합성 복원) ✅
Trades: 867 → 1 (HOLD 의미 복원 효과) ✅
Win Rate: 2.3% → 0.0% (거래 빈도 감소로 측정 불가)
```

**평가**: **S급** - 모든 버그 완전 수정

### Phase 1B + 2A: lookahead 증가 + SMOTE (FIXED → IMPROVED)

#### 파라미터 변경
```
lookahead: 60 (5시간) → 288 (24시간)
threshold: 1.0% → 2.0%
SMOTE: 미사용 → 활성화
```

#### 실제 효과
```
긍정:
✅ 거래 활성화: 1 → 9 trades
✅ Win Rate: 0.0% → 66.7%
✅ 타겟 분포 균형: HOLD 86.3% → 78.7%

부정:
❌ 수익률 악화: -2.05% → -2.80%
❌ 과적합 심화: Test Accuracy 62.36% → 43.03%
❌ SMOTE 합성 샘플 의존도
```

**평가**: **A급** - Win Rate 극적 개선, 하지만 수익성 악화

### Option B: 회귀 문제로 전환 (IMPROVED → REGRESSION)

#### 근본적 변화
```
분류 문제 (3-class):
  {SHORT, HOLD, LONG} 예측
  클래스 불균형 존재
  SMOTE 필요

    ↓ 전환

회귀 문제 (continuous):
  실제 수익률 예측
  클래스 불균형 개념 소멸
  SMOTE 불필요
```

#### 실제 효과
```
근본적 개선:
✅ 클래스 불균형 문제 완전 소멸
✅ SMOTE 불필요 → 과적합 위험 제거
✅ SNR 2.31 달성 (목표 >2.0 초과)
✅ Direction Accuracy 64.6%
✅ 실제 수익률 직접 예측 가능

치명적 실패:
❌ 거래 0회 발생
❌ 임계값 캘리브레이션 실패
   - Threshold: ±1.5%
   - Actual Std: 0.645%
   - 2.3σ 거리 → 4% 확률만 초과
❌ 동적 임계값 조정 무력화
```

**평가**: **C급** - 이론적으로 우수, 실전에서 무용지물

---

## 🎯 Part 3: 근본 문제 분석

### 3.1 왜 REGRESSION 모델이 실패했는가?

#### 근본 원인: 임계값 캘리브레이션 문제

```python
# 목표 설정
lookahead = 48 (4시간)
long_threshold = 0.015  # 1.5%
short_threshold = -0.015  # -1.5%
SNR = 1.5% / 0.645% = 2.31 ✅ (목표 >2.0 달성)

# 실제 데이터 분포
Target Statistics (Test):
  Mean: 0.020%  # 거의 0
  Std: 0.645%   # 작은 변동성

# 확률 계산
P(return > 1.5%) = P(Z > (1.5-0.02)/0.645)
                 = P(Z > 2.3)
                 ≈ 1.9%  # LONG 신호

P(return < -1.5%) = P(Z < -2.3)
                  ≈ 2.1%  # SHORT 신호

P(-1.5% < return < 1.5%) ≈ 96%  # HOLD

# 모델 학습 결과
모델은 정확히 학습:
  - 96%의 경우 수익률이 작음 (±1.5% 이내)
  - 따라서 작은 수익률 예측
  - 예측값이 모두 임계값 이내
  - 결과: 100% HOLD 예측
```

#### 수학적 분석
```
SNR = 2.31이 의미하는 것:
  신호 = 1.5% (threshold)
  노이즈 = 0.645% (std)
  SNR = 2.31 → 신호가 노이즈보다 2.31배 강함

하지만:
  Mean ≈ 0% → 실제 신호 강도 거의 없음
  신호는 threshold 자체가 아니라 예측 가능한 패턴

결론:
  SNR이 높아도 실제로 threshold 초과하는
  예측 가능한 수익이 없으면 의미 없음
```

### 3.2 IMPROVED vs REGRESSION 비교

#### IMPROVED (분류 + SMOTE)
```
장점:
✅ 거래 활성화 (9 trades)
✅ Win Rate 66.7%
✅ 실제 거래 발생

단점:
❌ 과적합 (2.23x)
❌ SMOTE 합성 데이터 의존
❌ 수익률 -2.80%
```

#### REGRESSION (회귀)
```
장점:
✅ 이론적으로 우수한 접근
✅ 클래스 불균형 소멸
✅ SNR 2.31 달성
✅ 과적합 없음

단점:
❌ 거래 0회
❌ 임계값 과도하게 보수적
❌ 실전 무용지물
```

### 3.3 임계값 트레이드오프 역설

모든 모델이 직면한 근본 문제:

```
Low Threshold (0.5-0.8%):
  → 많은 거래 발생
  → 높은 거래 비용 (0.08% 왕복)
  → 수수료가 수익 잠식
  → 순손실

Medium Threshold (1.0-1.5%):
  → 적절한 거래 빈도 기대
  → 하지만 실제 시장 변동성 작음
  → 임계값 초과 사례 희소
  → 거래 빈도 여전히 낮음

High Threshold (1.5-2.0%):
  → 거래 거의 없음 또는 0
  → 수수료 영향 없음
  → 하지만 수익 기회도 없음
  → 의미 없는 모델
```

**핵심**: 현재 시장 데이터(5분봉 BTC)에서는 수수료(0.08%)를 초과하는 예측 가능한 수익 신호가 매우 약함

---

## 💡 Part 4: 최종 권장사항

### 4.1 REGRESSION 모델 수정 방안 (Priority 1)

#### Option A: 임계값 대폭 하향 조정
```python
# 현재
long_threshold = 0.015  # 1.5%
short_threshold = -0.015  # -1.5%

# 수정
long_threshold = 0.006  # 0.6% (약 1σ)
short_threshold = -0.006  # -0.6%

예상 효과:
- P(|return| > 0.6%) ≈ 35%
- 거래 빈도: 20-50 trades 예상
- SNR: 0.6% / 0.645% = 0.93 (목표 미달)
- 하지만 실제 거래 발생
```

#### Option B: Percentile-Based Thresholds
```python
# 절대값 대신 백분위수
top_20_percent_return = np.percentile(returns, 80)
bottom_20_percent_return = np.percentile(returns, 20)

long_threshold = top_20_percent_return
short_threshold = bottom_20_percent_return

예상 효과:
- 자동으로 시장 분포 반영
- 거래 빈도 보장 (20%)
- Adaptive to market conditions
```

#### Option C: Multi-Threshold Ensemble
```python
models = {
    'aggressive': XGBRegressor(threshold=0.5%),
    'moderate': XGBRegressor(threshold=1.0%),
    'conservative': XGBRegressor(threshold=1.5%)
}

# 투표 시스템
if 2+ models agree → TRADE
if all disagree → HOLD

예상 효과:
- 다양한 리스크 프로파일
- 신뢰도 높은 거래만 실행
- 거래 빈도와 품질 균형
```

### 4.2 데이터 및 피처 개선 (Priority 2)

#### 1. 더 많은 데이터 수집
```
현재: 60일 (17,280 캔들)
목표: 6-12개월 (150,000-300,000 캔들)

효과:
- 과적합 방지
- 다양한 시장 조건 학습
- Threshold 캘리브레이션 정확도 향상
```

#### 2. 피처 엔지니어링 강화
```python
# 현재: 기술적 지표만
features = [RSI, MACD, BB, SMA, EMA, ...]

# 추가: 시장 구조 피처
features += [
    'volume_profile',
    'order_book_imbalance',
    'funding_rate',
    'market_regime',  # BULL/BEAR/SIDEWAYS
    'volatility_regime',  # HIGH/LOW
    'micro_structure'  # bid-ask spread, depth
]
```

#### 3. Multi-Timeframe Features
```python
# 현재: 5분봉만
data_5m = load_data('5m')

# 추가: 다중 시간대
data_1m = load_data('1m')   # 세밀한 패턴
data_15m = load_data('15m')  # 중기 추세
data_1h = load_data('1h')    # 장기 추세

# 융합
features = combine_timeframes(data_1m, data_5m, data_15m, data_1h)
```

### 4.3 대안 접근법 (Priority 3)

#### Option D: Hybrid Classification-Regression
```python
# Stage 1: Classification (거래 여부)
trade_classifier = XGBClassifier(target={TRADE, NO_TRADE})

# Stage 2: Regression (거래 방향 및 크기)
if trade_classifier.predict() == TRADE:
    return_predictor = XGBRegressor()
    predicted_return = return_predictor.predict()

    if predicted_return > 0:
        signal = LONG
    else:
        signal = SHORT

장점:
- 거래 여부와 방향/크기 분리
- 각 단계 최적화 가능
- 거래 빈도 제어 용이
```

#### Option E: Probabilistic Approach
```python
# 회귀 모델로 수익률 분포 예측
mean_return = model.predict(X)
std_return = model.predict_std(X)  # 불확실성

# 기대값 기반 거래
expected_value = (
    mean_return * (1 - transaction_fee)
    - risk_penalty * std_return
)

if expected_value > threshold:
    signal = LONG if mean_return > 0 else SHORT

장점:
- 불확실성 고려
- 리스크-리턴 트레이드오프 명시적
- 기대값 최대화
```

#### Option F: Reinforcement Learning 재검토
```python
# XGBoost는 feature → return 매핑
# RL은 state → action → reward 최적화

class TradingEnv:
    def reward(self, action, result):
        return (
            result.profit
            - result.transaction_cost
            - result.risk_penalty
            + result.sharpe_bonus
        )

agent = PPO(
    env=TradingEnv(),
    policy='MlpPolicy',
    learning_rate=3e-4
)

장점:
- 장기 수익 최적화
- 수수료와 리스크 내재화
- 복잡한 의사결정 가능
```

---

## 📋 Part 5: 최종 결론

### 5.1 4가지 버전 종합 평가

| 버전 | 접근법 | 핵심 성과 | 핵심 한계 | 등급 |
|------|--------|----------|----------|------|
| **BUGGY** | 분류 | 최초 구현 | 백테스팅 버그 5개 | F |
| **FIXED** | 분류 | 버그 완전 수정 | 거래 1회 (보수적) | C |
| **IMPROVED** | 분류+SMOTE | Win Rate 66.7% | 과적합 2.23x | B |
| **REGRESSION** | 회귀 | SNR 2.31, 이론 우수 | 거래 0회 | C |

### 5.2 핵심 학습 내용

#### ✅ 완전 해결된 문제
1. **백테스팅 정합성**: 수학적으로 불가능한 손실 제거
2. **HOLD 의미**: "청산" → "포지션 유지" 복원
3. **클래스 불균형**: 회귀 전환으로 근본 해결 (이론적)
4. **SNR 목표**: 2.31 달성 (목표 >2.0)
5. **Win Rate**: IMPROVED에서 66.7% 달성

#### ⚠️ 부분 해결된 문제
1. **거래 빈도**: 867 → 9 → 0 (조정 필요)
2. **과적합**: SMOTE 제거로 완화, but 거래 없음
3. **수익성**: -1051% → -2.8% → 0% (개선 but 목표 미달)

#### ❌ 미해결 핵심 문제
1. **임계값 캘리브레이션**:
   - 이론적 SNR과 실제 거래 발생 괴리
   - threshold vs 실제 return 분포 불일치

2. **수익성 달성 실패**:
   - 모든 버전에서 양수 수익 미달성
   - 거래 비용 vs 예측 가능 수익 불균형

3. **실전 적용 가능성**:
   - REGRESSION이 이론적으로 최선이나 거래 0회
   - IMPROVED가 실전적이나 과적합 문제

### 5.3 최종 권장사항

#### 즉시 실행 (1주일 내)
```
1. REGRESSION 임계값 조정
   - 현재: ±1.5% → 수정: ±0.6%
   - 또는 Percentile-based thresholds

2. 실험 및 검증
   - Threshold sweep: 0.3%, 0.5%, 0.8%, 1.0%, 1.5%
   - 각 threshold의 거래빈도/수익성 측정
   - 최적 조합 탐색
```

#### 단기 실행 (1개월 내)
```
1. Multi-Threshold Ensemble
   - 3개 모델 조합 (aggressive/moderate/conservative)
   - 투표 시스템으로 신뢰도 향상

2. Hybrid Classification-Regression
   - Stage 1: 거래 여부 분류
   - Stage 2: 방향/크기 회귀

3. 더 많은 데이터 수집
   - 현재 60일 → 목표 6개월
```

#### 중기 실행 (3개월 내)
```
1. 피처 엔지니어링 강화
   - Volume profile, order book, funding rate
   - Multi-timeframe features

2. RL 재검토
   - PPO/SAC with improved reward function
   - 수수료와 리스크 내재화

3. Market Regime Detection
   - BULL/BEAR/SIDEWAYS 분류
   - 상황별 모델 적용
```

### 5.4 성공 기준 재정의

#### 현실적 목표 (3개월)
- [ ] Test Return > 0% (손익분기)
- [ ] Test Trades > 20 (의미 있는 거래)
- [ ] Win Rate > 55% (실전 가능)
- [ ] Sharpe Ratio > 0.5 (리스크 대비 수익)
- [ ] Overfitting < 1.5x (일반화)

#### 이상적 목표 (6개월)
- [ ] Test Return > 5%
- [ ] Sharpe Ratio > 1.0
- [ ] Max Drawdown < 15%
- [ ] Win Rate > 60%
- [ ] Paper Trading 성공

### 5.5 교훈 및 인사이트

#### 핵심 교훈
1. **이론과 실전의 괴리**
   - SNR 2.31 달성했으나 거래 0회
   - 수학적 목표 ≠ 실전 성과

2. **과적합의 역설**
   - SMOTE로 Win Rate 개선
   - 하지만 실제 시장 패턴 학습 실패

3. **임계값 최적화의 중요성**
   - 모델 성능보다 threshold 설정이 더 중요
   - 시장 분포 기반 동적 조정 필수

4. **다단계 검증의 필요성**
   - Train/Val/Test Accuracy만으로 부족
   - 실제 백테스팅 결과가 진실

#### 비판적 사고의 가치
```
초기 가정: "클래스 불균형이 문제다"
→ SMOTE 적용
→ Win Rate 개선, but 과적합

재분석: "분류 자체가 문제다"
→ 회귀 전환
→ 불균형 소멸, but 거래 0회

최종 통찰: "임계값이 진짜 문제다"
→ Threshold 재조정 필요
→ 다음 실험으로
```

---

## 📊 Appendix: 전체 결과 요약표

| 지표 | BUGGY | FIXED | IMPROVED | REGRESSION | 목표 | 최종 |
|------|-------|-------|----------|------------|------|------|
| **아키텍처** |
| Model Type | Classifier | Classifier | Classifier | Regressor | - | ✅ |
| lookahead | 5 (25m) | 60 (5h) | 288 (24h) | 48 (4h) | - | ✅ |
| threshold | 0.2% | 1.0% | 2.0% | 1.5% | - | ⚠️ |
| Class Balance | None | scale_pos | SMOTE | N/A | - | ✅ |
| **신호 품질** |
| SNR | 0.95 | 1.37 | 1.26 | **2.31** | >2.0 | 🏆 |
| Target LONG | 11.3% | 7.0% | 10.8% | - | ~30% | ❌ |
| Target HOLD | 78.0% | 86.3% | 78.7% | - | ~40% | ⚠️ |
| Target SHORT | 10.7% | 6.7% | 10.5% | - | ~30% | ❌ |
| **모델 성능** |
| Train Acc | 78.03% | 87.41% | 96.05% | - | - | - |
| Test Acc | 78.95% | 62.36% | 43.03% | - | >60% | ❌ |
| Direction Acc | - | - | - | 64.6% | >60% | ✅ |
| Overfitting | 0.99x | 1.40x | 2.23x | 0.99x | <1.5x | ⚠️ |
| **백테스팅** |
| Test Return | -1051.80% | -2.05% | -2.80% | **0.00%** | >+5% | ❌ |
| Test Trades | 867 | 1 | 9 | **0** | 50-100 | ❌ |
| Win Rate | 2.3% | 0.0% | 66.7% | **0.0%** | >55% | ⚠️ |
| Liquidated | No | False | False | **False** | False | ✅ |
| **검증** |
| 수학적 정합성 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 버그 수정 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HOLD 의미 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 실전 가능성 | ❌ | ❌ | ⚠️ | ❌ | ✅ | ❌ |

### 최종 등급
```
BUGGY:     F급 (백테스팅 버그로 무효)
FIXED:     C급 (버그 수정 완료, 너무 보수적)
IMPROVED:  B급 (Win Rate 우수, 과적합 문제)
REGRESSION: C급 (이론 우수, 실전 실패)
```

### 다음 버전 목표
```
REGRESSION v2: A급 목표
- Threshold 조정 (1.5% → 0.6%)
- 거래 빈도: >20 trades
- Test Return: >0%
- Win Rate: >55%
- 실전 가능성: ✅
```

---

## 🔬 특별 분석: 왜 회귀 모델이 실패했는가?

### 수학적 분석

```python
# REGRESSION 모델 타겟 통계 (Test Set)
Mean return: 0.020%
Std return: 0.645%
Min return: -8.47%
Max return: +8.92%

# 임계값 설정
long_threshold = 1.5%
short_threshold = -1.5%

# 표준 정규 분포로 근사
Z_score_long = (1.5 - 0.02) / 0.645 = 2.29
Z_score_short = (-1.5 - 0.02) / 0.645 = -2.36

# 확률 계산
P(return > 1.5%) = P(Z > 2.29) ≈ 1.1%
P(return < -1.5%) = P(Z < -2.36) ≈ 0.9%
P(-1.5% < return < 1.5%) ≈ 98%

# 실제 관측 (Test Set)
Actual LONG signals: 1.9% (49/2574)
Actual SHORT signals: 2.1% (54/2574)
Actual HOLD signals: 96.0% (2471/2574)
```

**결론**:
- 임계값 1.5%는 평균에서 2.3σ 거리
- 정규분포에서 2.3σ 초과 확률은 약 2%
- 모델이 정확히 학습: 대부분 수익률은 작음
- 예측값도 작음 → 임계값 초과 불가 → 100% HOLD

### 임계값 민감도 분석

| Threshold | Expected Trade % | Expected Trades (Test) | SNR |
|-----------|------------------|------------------------|-----|
| 0.3% | 68% | 1750 | 0.47 ❌ |
| 0.5% | 44% | 1132 | 0.78 ❌ |
| 0.6% | 35% | 901 | 0.93 ⚠️ |
| 0.8% | 21% | 541 | 1.24 ✅ |
| 1.0% | 12% | 309 | 1.55 ✅ |
| 1.5% | 2% | 51 | 2.31 ✅ |
| 2.0% | 0.5% | 13 | 3.10 ✅ |

**최적 범위**: 0.6-0.8% (거래 빈도와 SNR 균형)

---

**보고서 끝**

*다음 권장사항: REGRESSION Threshold 0.6-0.8%로 재실험*
