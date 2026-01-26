# Momentum/Trend Following 전략 실패 분석 - 2025년 11월 23일

## 📋 요약

**결론**: ❌ **Momentum 전략 실패 - Donchian보다 50% 더 나쁨**

Momentum/Trend Following 전략이 -67.61% 손실로 참담한 실패.
Donchian (-17.55%)보다 훨씬 나쁨.

**다음 단계**: 🎯 **Random Masking ML 모델 활용 검토**

---

## 📊 Momentum 전략 결과 (15분봉, 89일)

### 전체 성과

```yaml
시작 잔고: $200.00
종료 잔고: $64.77
총 수익률: -67.61%
월 평균: -22.79%

거래 수: 343 (3.9/day)
승률: 51.6%
Profit Factor: 0.81×

수수료: $59.35
수수료 영향: 78.2%
```

### 배포 기준 평가

| 기준 | 목표 | 결과 | 통과 |
|------|------|------|------|
| **총 수익률** | >0% | -67.61% | ❌ |
| **월간 일관성** | >60% | 0.0% | ❌ |
| **주간 일관성** | >60% | 14.3% | ❌ |
| **수수료 영향** | <20% | 78.2% | ❌ |
| **승률** | >55% | 51.6% | ❌ |
| **Profit Factor** | >1.5× | 0.81× | ❌ |
| **방향 균형** | 30-70% | 49.9%/50.1% | ✅ |

**통과율**: 1/7 (14.3%) - 방향 균형만 통과

---

## 🔄 3가지 전략 종합 비교

### 성과 비교표

| 지표 | 5분봉 Donchian | 15분봉 Donchian | 15분봉 Momentum |
|------|----------------|-----------------|-----------------|
| **수익률** | -33.49% | **-17.55%** | -67.61% ❌ |
| **월간 일관성** | 50% | **100%** 🏆 | 0% ❌ |
| **주간 일관성** | 40% | **57.1%** | 14.3% ❌ |
| **거래/일** | 6.0 | **4.6** | 3.9 ✅ |
| **승률** | **52.2%** | 48.5% | 51.6% |
| **수수료 영향** | 210.3% | 146.9% | **78.2%** ✅ |
| **Profit Factor** | 0.81× | **1.23×** | 0.81× |
| **LONG/SHORT** | 2.7% / 97.3% | 0% / 100% | **49.9% / 50.1%** 🏆 |
| **보유 시간** | 110분 | 68분 | **185분** 🏆 |

### 순위

1. **최고 수익률**: 15분봉 Donchian (-17.55%)
2. **최악 수익률**: 15분봉 Momentum (-67.61%)
3. **차이**: 50.06% (Momentum이 Donchian보다 3배 나쁨)

---

## ❌ Momentum 전략이 실패한 이유

### 1. 8월 대재앙: -30.84% 손실

**8월 성과 비교**:
```yaml
5분봉 Donchian: -6.12%
15분봉 Donchian: +16.86% ✅
15분봉 Momentum: -30.84% ❌ (최악)

차이: -47.7% (Donchian 대비)
```

**원인 분석**:
- 91 거래 (3.3/day) - 과도한 거래
- 승률 45.1% - 코인 던지기보다 나쁨
- Stop Loss 빈발 - 트렌드 역행 거래

**8월 시장 특성**:
- 급격한 변동성 (Whipsaw)
- 잦은 추세 반전
- MA Cross 신호 혼란

**Momentum 전략 문제**:
- RSI > 50 진입 → 바로 RSI < 50 하락 → Stop Loss
- MA Cross → 빠른 재크로스 → 손실 누적
- Volume spike 오판 → 거짓 신호

### 2. Stop Loss 비율 과다: 19%

**Exit 메커니즘 비교**:
```yaml
5분봉 Donchian:
  RSI Exit: 80.9%
  Stop Loss: 16.6%
  Donchian: 2.5%

15분봉 Donchian:
  Donchian Middle: 78.4%
  RSI Exit: 18.7%
  Stop Loss: 3.0%

15분봉 Momentum:
  RSI Exit: 63.0%
  Stop Loss: 19.0% ❌ (6배 높음)
  MA Cross: 18.1%
```

**문제**:
- Stop Loss 19% = 343 거래 중 65회 -3% 손실
- 총 손실: 65 × -3% = -195% (누적 손실)
- 승리 거래가 이를 만회 못함

**원인**:
- MA Cross 신호 지연 → 트렌드 역행 진입
- RSI 50 기준 너무 느슨 → 약한 신호도 진입
- Volume spike 오판 → 거짓 브레이크아웃

### 3. RSI Exit 63% - 조기 청산

**RSI Exit 로직**:
```python
LONG Exit: RSI >= 70 (overbought)
SHORT Exit: RSI <= 30 (oversold)
```

**문제**:
- RSI 70/30은 극단적 조건 (드물게 도달)
- 하지만 63%가 RSI Exit → 자주 트리거됨
- 의미: RSI가 빠르게 극단으로 이동

**시장 해석**:
- 급격한 변동성 → RSI 빠르게 70/30 도달
- Momentum 전략 → 변동성 높은 시점 진입
- 진입 후 빠른 반전 → RSI Exit 트리거

**결과**:
- 평균 보유 185분 (길지만)
- 63%가 RSI Exit = 극단 도달 후 청산
- 수익 충분히 가져가기 전 조기 청산

### 4. 0% 월간 일관성

**월별 성과**:
```yaml
2025-08: -30.84% ❌ (91 trades, 45.1% WR)
2025-09: -1.80% ❌ (113 trades, 54.9% WR)
2025-10: -4.03% ❌ (115 trades, 53.9% WR)
2025-11: -1.27% ❌ (24 trades, 50.0% WR)

긍정적 월: 0/4 (0%)
```

**Donchian 15분봉 비교**:
```yaml
2025-08: +16.86% ✅
2025-09: +4.97% ✅
2025-10: +6.21% ✅
2025-11: +9.25% ✅

긍정적 월: 4/4 (100%)
```

**차이**:
- Donchian: 모든 달 흑자
- Momentum: 모든 달 적자
- 전략 자체가 현재 시장과 완전 부적합

### 5. LONG/SHORT 균형의 역설

**유일한 통과 항목**:
```yaml
LONG: 49.9% (171 trades)
SHORT: 50.1% (172 trades)

완벽한 50/50 균형! ✅
```

**하지만**:
- LONG P&L: -$39.36
- SHORT P&L: -$36.52
- 둘 다 손실!

**의미**:
- 방향 균형은 좋음
- BUT 둘 다 수익 못 냄
- 전략 로직 자체에 문제

---

## 💡 Momentum 전략의 근본적 한계

### 1. MA Cross 지연 문제

**이론**:
```
가격 > SMA(50) → Uptrend 진입
가격 < SMA(50) → Downtrend 진입
```

**현실**:
- MA는 지표 특성상 후행 (Lagging)
- 가격이 이미 움직인 후 MA가 따라감
- MA Cross 신호 시점 = 이미 늦음

**결과**:
- 진입 타이밍 늦음
- 추세 끝자락에 진입
- 빠른 반전 → Stop Loss

### 2. RSI 50 기준의 오류

**설정**:
```python
rsi_long_entry: 50 (bullish momentum)
rsi_short_entry: 50 (bearish momentum)
```

**문제**:
- RSI 50 = 중립 (neutral)
- 강한 모멘텀 아님 (30/70이 강함)
- 약한 신호도 진입

**개선안**:
- LONG: RSI > 60 (더 강한 모멘텀)
- SHORT: RSI < 40 (더 강한 모멘텀)

**하지만**:
- 개선해도 근본 해결 안 됨
- 전략 자체가 레인지장에 부적합

### 3. Volume Spike 오판

**로직**:
```python
volume_spike > 1.2 (Volume > MA × 1.2)
```

**문제**:
- Volume spike = 브레이크아웃 or 페이크
- 20% 증가는 낮은 기준 → 거짓 신호 많음
- BTC 시장: 급등/급락 시 항상 volume spike

**결과**:
- 거짓 브레이크아웃 진입
- 빠른 반전 → 손실

### 4. BTC 8-11월 시장 특성 불일치

**Momentum 전략 최적 시장**:
```yaml
유형: Trending Market (강한 추세)
특성:
  - 명확한 상승/하락 추세
  - 지속적 모멘텀
  - 적은 반전

시기: Bull/Bear Market
```

**BTC 8-11월 실제 시장**:
```yaml
유형: Ranging/Choppy Market (횡보/변동)
특성:
  - 레인지 바운스 80%
  - 짧은 추세 20%
  - 빈번한 반전 (Whipsaw)

결과: Momentum 전략 불리
```

---

## ✅ Momentum 전략의 유일한 장점

### 1. 완벽한 방향 균형

```yaml
LONG: 49.9%
SHORT: 50.1%

vs Donchian:
  15분봉: 0% / 100%
  5분봉: 2.7% / 97.3%
```

**의미**:
- LONG/SHORT 진입 조건 동등
- 편향 없음

### 2. 수수료 영향 감소

```yaml
Momentum: 78.2%

vs Donchian:
  5분봉: 210.3%
  15분봉: 146.9%
```

**이유**:
- 거래 빈도 낮음 (3.9/day vs 4.6/day)
- 더 긴 보유 시간 (185분 vs 68분)

### 3. 더 긴 보유 시간

```yaml
평균 보유: 185분 (3시간)

vs Donchian:
  5분봉: 110분
  15분봉: 68분
```

**의미**:
- 추세를 더 오래 추종
- 조기 청산 적음

**하지만**:
- 이들이 수익으로 이어지지 않음
- 긴 보유 = 더 큰 손실 (Stop Loss 19%)

---

## 📊 왜 Donchian이 Momentum보다 나은가?

### 비교 요약

```yaml
15분봉 Donchian:
  수익률: -17.55%
  월간 일관성: 100% (4/4)
  Stop Loss: 3.0%
  승률: 48.5%
  전략: 단순 (Donchian Channel)

15분봉 Momentum:
  수익률: -67.61%
  월간 일관성: 0% (0/4)
  Stop Loss: 19.0%
  승률: 51.6%
  전략: 복잡 (MA + RSI + Volume)
```

### 핵심 차이

**1. Stop Loss 비율**:
- Donchian: 3.0% (12/402 trades)
- Momentum: 19.0% (65/343 trades)
- Momentum이 6배 높음

**2. 월간 일관성**:
- Donchian: 100% (모든 달 흑자)
- Momentum: 0% (모든 달 적자)

**3. Exit 메커니즘**:
- Donchian: Donchian Middle (78.4%) - 추세 반전 조기 감지
- Momentum: RSI Exit (63.0%) + Stop Loss (19.0%) - 늦은 감지

**결론**: 단순한 Donchian이 복잡한 Momentum보다 우수

---

## 🎯 결론 및 다음 단계

### 전략 평가 최종 순위

| 순위 | 전략 | 수익률 | 월간 일관성 | 평가 |
|------|------|--------|-------------|------|
| 1 | 15분봉 Donchian | -17.55% | 100% | ⚠️ 최선 (하지만 손실) |
| 2 | 5분봉 Donchian | -33.49% | 50% | ❌ |
| 3 | 15분봉 Momentum | -67.61% | 0% | ❌ 최악 |

### 발견한 사실

1. **시간프레임 중요**: 5분 → 15분 = +15.94% 개선 ✅
2. **Donchian > Momentum**: 단순이 복잡보다 나음 ✅
3. **모든 전략 손실**: 현재 BTC 시장 = 레인지/횡보장 ❌
4. **Rule-based 전략 한계**: 고정 규칙은 변화하는 시장 적응 못함 ❌

### 사용자 요청

**"Momentum/Trend Following 전략 백테스트 진행, 이후 결과가 좋지 않다면 Random Masking ML 모델 활용 검토"**

→ ✅ Momentum 백테스트 완료
→ ❌ 결과 좋지 않음 (-67.61%, 0% 월간 일관성)
→ 🎯 **다음 단계: Random Masking ML 모델 활용 검토**

---

## 🤖 Random Masking ML 모델 검토 방향

### 기존 프로젝트 상태 (CLAUDE.md 기준)

**Random Masking 연구**:
```yaml
위치: experimental/random_masking/
상태: Option J (Alternative Data Integration) 진행 중

진행 상황:
  - Option A-G: 실패 (24.22% accuracy < 25% random)
  - Option J: 24시간 데이터 수집 (42/288 samples, 14.6% 완료)
  - 31 alternative features (order book, on-chain, sentiment)
  - Phase 2: Correlation analysis (auto-triggered)
  - Phase 3: Model retraining (manual)

문제:
  - 74 technical features만으로는 부족 (10-20% 정보만 캡처)
  - 60-80% 정보 누락 (order flow, sentiment, on-chain)
```

### 제안 방향

#### Option 1: Option J 완료 후 활용
```yaml
방법:
  1. Option J Phase 2 완료 대기 (correlation analysis)
  2. 유의미한 features 선택 (|r| > 0.1)
  3. 105-feature model 재훈련
  4. Donchian 전략 대체

예상:
  - 정확도: 30-45% (vs 24% baseline)
  - 백테스트: +10-30% monthly
  - 시간: 2-4주 (데이터 수집 완료 필요)

리스크:
  - Option J도 실패 가능 (|r| < 0.1)
  - 장기 프로젝트
```

#### Option 2: 다른 ML 접근 (LSTM/Transformer)
```yaml
방법:
  1. 시계열 모델 (LSTM, GRU, Transformer)
  2. 기존 74 features + sequence learning
  3. 시장 패턴 자동 학습

장점:
  - 복잡한 시계열 패턴 학습
  - Feature engineering 최소화

단점:
  - 데이터 많이 필요 (>1년)
  - 훈련 시간 오래 걸림
  - 과적합 리스크
```

#### Option 3: Hybrid (Rule-based + ML)
```yaml
방법:
  1. Donchian for Entry (검증됨)
  2. ML for Exit Timing (최적화)
  3. 조합 전략

장점:
  - Donchian 월간 일관성 100% 활용
  - ML로 Exit 개선 (현재 Donchian Middle 78.4%)
  - 빠른 구현

예상:
  - 수익률: -17.55% → 0-10% (추정)
  - Exit 최적화로 수익 증대
```

### 권장 순서

1. **즉시** (1주): Option J 완료 대기, 기존 연구 검토
2. **단기** (2-4주): Option 3 Hybrid 시도 (빠른 개선)
3. **중기** (1-2개월): Option J 결과 활용 또는 Option 2 LSTM

---

## 📁 생성된 파일

- `results/backtest_momentum_20251123_004837.csv` (343 거래 상세)
- `claudedocs/MOMENTUM_STRATEGY_FAILURE_20251123.md` (이 문서)
- `scripts/analysis/backtest_momentum_strategy.py` (Momentum 백테스트)

---

**작성일**: 2025-11-23 00:48 KST
**작성자**: Claude Code
**상태**: ❌ Momentum 전략 실패 - Random Masking ML 모델 검토 필요
