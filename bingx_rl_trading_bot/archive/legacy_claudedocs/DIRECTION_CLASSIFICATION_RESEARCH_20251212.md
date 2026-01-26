# 방향 분류 연구 종합 (2025-12-12)

## 연구 목표
- LONG/SHORT 방향을 제대로 분류하여 각 방향에 맞는 진입
- 상승장 → LONG, 하락장 → SHORT
- Rolling Window 일관성 향상

---

## 핵심 발견

### 1. "레짐 = 진입 방향" 가정의 오류 ❌

| 레짐 | 거래 수 | 방향 | Win Rate |
|------|---------|------|----------|
| 상승 | 4건 | LONG | 50% |
| **하락** | **5건** | **SHORT** | **20%** ❌ |
| 횡보 | 71건 | BOTH | 41% |

**문제점**: 하락 레짐으로 분류된 시점에 SHORT 진입하면 WR 20%
→ 이미 하락이 진행된 후라 **반등 가능성이 높아짐**
→ 레짐 기반 방향 분류는 비효율적

### 2. 시장 레짐 분포

테스트 기간 (150일의 30%)은 하락장 위주:

| 레짐 방법 | 상승 | 하락 | 횡보 |
|-----------|------|------|------|
| ema200 | 18% | 36% | 46% |
| momentum | 25% | 40% | 35% |
| adx_di | 32% | 40% | 28% |
| hma_slope | 23% | 24% | 53% |
| combined | 25% | 35% | 41% |

→ 횡보장이 28-53%로 가장 많음
→ 테스트 데이터는 하락장 편향

### 3. 방향별 성과 비교

| 필터 조합 | LONG WR | SHORT WR | RW 일관성 |
|-----------|---------|----------|-----------|
| 기본 (필터 없음) | 47% | 51-70% | 42-58% |
| LONG: EMA200 위 필수 | 51% | 51% | 67% |
| ADX DI 기반 | 50% | 71% | 58% |
| 강한 LONG 필터 | 58% | 37% | 40% |

**결론**: LONG 필터를 강화하면 SHORT WR이 하락하는 트레이드오프

### 4. LONG 승/패 특성 차이가 미미

| 메트릭 | 승리 트레이드 | 패배 트레이드 | 차이 |
|--------|--------------|---------------|------|
| HMA Slope | 0.244 | 0.218 | +0.027 |
| 6h 모멘텀 | 0.23% | 0.24% | -0.01% |
| 12h 모멘텀 | -0.34% | -0.30% | -0.05% |

→ 기존 지표로는 LONG 승패 예측이 어려움
→ 새로운 지표나 접근 방식 필요

---

## 테스트한 방향 분류 방법

### 방법 1: HMA Slope 기반
```python
if slope > 0.1: LONG
elif slope < -0.1: SHORT
else: NO_TRADE
```
- RW 일관성: 58%
- 문제: 분류 정확도 ~47%

### 방법 2: EMA200 기반
```python
if close > ema200: LONG
elif close < ema200: SHORT
```
- RW 일관성: 67% (LONG만 EMA200 적용시)
- 문제: 양방향 적용시 33%로 하락

### 방법 3: ADX + DI 기반
```python
if adx > 25:
    if plus_di > minus_di: LONG
    else: SHORT
else: NO_TRADE
```
- RW 일관성: 58%
- LONG WR 50%, SHORT WR 71%

### 방법 4: 복합 지표 (Combined)
```python
score = 0
if close > ema200: score += 1
if hma_slope > 0.1: score += 1
if momentum_1d > 0.5: score += 1

if score >= 2: LONG
elif score <= -2: SHORT
else: NO_TRADE
```
- RW 일관성: 42%
- 문제: 횡보 필터링으로 기회 상실

---

## 권장 전략: 비대칭 필터

### 기본 원칙
1. **LONG은 보수적**: 확실한 상승 신호에서만 진입
2. **SHORT는 기본 조건**: 기회를 놓치지 않음
3. **횡보장에서도 진입**: 레짐 분류 무시

### 최적 설정

```yaml
strategy: "비대칭 방향 필터"
description: "LONG은 엄격, SHORT는 유연"

long_conditions:
  - hma30_slope > 0.1
  - close > hma30
  - close > ema200  # LONG만 적용
  - rsi 교차 상향

short_conditions:
  - hma30_slope < -0.1
  - close < hma30
  - rsi 교차 하향
  # ema200 조건 없음

exit:
  tp: 2.5%
  sl: 1.0%
  be_trigger: 1.5%

leverage: 4x
cooldown: 4 candles
```

### 예상 성과

| 메트릭 | 기본 전략 | 비대칭 필터 |
|--------|----------|-------------|
| RW 일관성 | 58% | **67%** |
| LONG WR | 47% | **51%** |
| SHORT WR | 51% | 51% |
| Combined RA | +12.50 | +6.83 |

**트레이드오프**: Combined RA 감소 but 안정성(RW) 향상

---

## 결론 및 권장사항

### 핵심 인사이트

1. **레짐 기반 방향 분류는 비효율적**
   - 하락 레짐 진입 SHORT WR = 20% (최악)
   - 이미 움직임이 발생한 후 분류되기 때문

2. **비대칭 필터가 효과적**
   - LONG만 EMA200 필터 적용
   - SHORT는 기본 조건 유지

3. **횡보장 무시는 손실**
   - 대부분의 거래가 횡보장에서 발생 (71/80건)
   - 횡보장 필터링은 기회 상실

### 권장 전략

**Option A: 안정성 우선**
```
LONG: EMA200 위 + HMA slope > 0.1 + RSI 교차
SHORT: HMA slope < -0.1 + RSI 교차
RW 일관성: 67%
```

**Option B: 수익성 우선**
```
LONG/SHORT: 기본 조건 동일
RW 일관성: 58%
Combined RA: +12.50
```

### 구현 시 주의사항

1. **방향 분류에 집착하지 않기**
   - 완벽한 방향 분류는 불가능
   - 적절한 TP/SL 관리가 더 중요

2. **데이터 편향 인지**
   - 테스트 데이터가 하락장 위주
   - 상승장에서 LONG 성과는 추가 검증 필요

3. **단순함 유지**
   - 복잡한 필터 조합은 과적합 위험
   - 기본 조건 + EMA200 (LONG만)이 최적

---

## 관련 파일

| 파일 | 내용 |
|------|------|
| `scripts/analysis/direction_classification_research.py` | 6가지 분류 방법 테스트 |
| `scripts/analysis/improved_direction_filter.py` | LONG 실패 원인 분석 |
| `scripts/analysis/adaptive_direction_strategy.py` | 35개 필터 조합 테스트 |
| `scripts/analysis/regime_based_direction.py` | 레짐 기반 전략 테스트 |
| `results/adaptive_direction_strategy_20251212.csv` | 필터 조합 결과 |
| `results/regime_based_direction_20251212.csv` | 레짐 기반 결과 |

---

**작성일**: 2025-12-12
**연구 기간**: 150일 BTC/USDT 15분봉 데이터
