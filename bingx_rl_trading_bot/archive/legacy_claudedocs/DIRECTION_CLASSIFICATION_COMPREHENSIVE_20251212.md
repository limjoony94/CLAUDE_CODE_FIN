# 방향 분류 종합 연구 보고서 (2025-12-12)

## 연구 목표
- LONG/SHORT 방향을 제대로 분류하여 적절한 타이밍에 진입
- 상승장 → LONG, 하락장 → SHORT
- Rolling Window 일관성 향상

---

## 연구 방법론

| 연구 영역 | 접근 방식 | 스크립트 |
|-----------|----------|----------|
| 기본 방향 분류 | HMA slope, EMA200, ADX/DI | `direction_classification_research.py` |
| 비대칭 필터 | LONG 엄격 + SHORT 유연 | `adaptive_direction_strategy.py` |
| 레짐 기반 | 시장 상태 분류 후 방향 결정 | `regime_based_direction.py` |
| 역추세 전략 | 하락→LONG, 상승→SHORT | `counter_trend_research.py` |
| ATR% 기반 | 변동성에 따른 방향 선택 | `atr_direction_research.py` |
| 다중 타임프레임 | 상위 TF 추세 참조 | `mtf_direction_research.py` |

---

## 핵심 발견 요약

### 1. "레짐 = 진입 방향" 가정은 틀림 ❌

| 레짐 | 진입 방향 | Win Rate | 문제점 |
|------|----------|----------|--------|
| 상승 | LONG | 50% | 보통 |
| **하락** | **SHORT** | **20%** | ❌ 최악 |
| 횡보 | BOTH | 41% | 대다수 |

→ 하락 레짐 진입 시점은 이미 하락이 끝나고 **반등 구간**

### 2. 전략 유형별 성과

| 전략 | LONG WR | SHORT WR | RW 일관성 | Combined RA |
|------|---------|----------|-----------|-------------|
| 순추세 | 47% | **70%** | 42% | +1.80 |
| 역추세 | 53% | 39% | 50% | +0.57 |
| **역추세+EMA200** | **64%** | 0% | **67%** | **+2.30** |
| 평균회귀 (RSI<25, RSI>70) | 48% | 62% | 58% | +2.20 |

→ **역추세 + EMA200 필터**: LONG WR 64%로 최고

### 3. RSI 임계값 최적화

| RSI 조건 | LONG WR | SHORT WR | RW 일관성 | Combined RA |
|----------|---------|----------|-----------|-------------|
| RSI<25, RSI>70 | 48% | 62% | 58% | **+2.20** |
| RSI<35, RSI>70 | 47% | 62% | 50% | **+2.21** |
| RSI<40, RSI>70 | 39% | 62% | **67%** | +0.85 |

→ **RSI > 70에서 SHORT**가 가장 안정적 (WR 55-62%)

### 4. ATR% 기반 방향 성과

| ATR% + 추세 조합 | 거래 수 | Win Rate |
|-----------------|---------|----------|
| **고변동 + 강한 하락** | 4건 | **75%** |
| 저변동 + 강한 하락 | 3건 | 67% |
| **중변동 (0.17-0.34%)** | 8건 | L **67%** / S **80%** |
| 저변동 + 약한 추세 | 27건 | **30%** ❌ |

→ **중변동 구간**에서 가장 좋은 성과
→ **저변동 + 약한 추세**는 피해야 함

### 5. MTF 필터 효과

| MTF 필터 | LONG WR | SHORT WR | 거래 수 | RW 일관성 |
|----------|---------|----------|---------|-----------|
| 필터 없음 | 47% | 70% | 27 | 42% |
| 비대칭 (LONG 엄격) | 0% | 70% | 10 | 42% |
| 4H 추세 | 0% | 0% | 1 | 50% |

→ 엄격한 MTF 필터는 거래 기회 상실
→ 비대칭 필터가 균형점

---

## 최적 방향 분류 전략

### Option 1: 비대칭 필터 (권장) ⭐

```yaml
strategy: "비대칭 방향 필터"
description: "LONG은 엄격, SHORT는 유연"

long_conditions:
  - HMA30 slope > 0.1
  - Close > HMA30
  - Close > EMA200        # LONG만 적용
  - RSI 교차 상향
  - 1H 추세 = 상승 (선택)

short_conditions:
  - HMA30 slope < -0.1
  - Close < HMA30
  - RSI 교차 하향
  # EMA200/MTF 필터 없음

exit:
  tp: 2.5%
  sl: 1.0%
  be_trigger: 1.5%
```

**예상 성과**: RW 67%, Combined RA +1.5~2.0

### Option 2: RSI Zone + RSI>70 필터

```yaml
strategy: "RSI Zone 강화"
description: "RSI Zone + 더 엄격한 과매수 조건"

long_conditions:
  - RSI < 35 진입 후 반등
  - Close > EMA200

short_conditions:
  - RSI > 70 진입 후 하락    # 65 → 70으로 강화
  - Close < EMA200

exit:
  tp: 2.4%
  sl: 1.4%
  be_trigger: 1.2%
```

**예상 성과**: SHORT WR 62%, RW 58%

### Option 3: ATR% 적응형

```yaml
strategy: "ATR% 기반 적응"
description: "변동성에 따른 방향 선택"

conditions:
  - ATR% > 0.25%: SHORT 위주 (70% 승률)
  - ATR% < 0.25%: BOTH (조심스럽게)
  - 저변동 + 약한 추세: 진입 자제
```

**예상 성과**: 고변동 SHORT WR 70%, RW 50%

---

## 연구 데이터 편향 주의

### 테스트 기간 특성
- **BTC 움직임**: 전반적 하락장 (특히 후반부 -23%)
- **상위 TF 추세**: 54-55% 하락
- **결과**: SHORT 성과가 과대평가될 수 있음

### 상승장 검증 필요
- 현재 연구는 하락장 데이터 위주
- 상승장에서 LONG 성과 별도 검증 필요
- 레짐 전환 시점 감지 메커니즘 고려

---

## 최종 권장사항

### 즉시 적용 가능

1. **RSI Zone v1.3에 RSI > 70 필터 추가**
   - SHORT 진입 조건: RSI > 65 → RSI > 70
   - 예상 효과: SHORT WR 51% → 62%

2. **저변동 + 약한 추세 필터링**
   - ATR% < 0.17% AND |HMA slope| < 0.1 → 진입 자제
   - 예상 효과: WR 30% 거래 제거

### 추가 연구 필요

1. **상승장 데이터 확보 후 LONG 전략 재검증**
2. **레짐 전환 감지 알고리즘 개발**
3. **다양한 시장 조건에서 백테스트**

---

## 결론

| 발견 | 결론 |
|------|------|
| 레짐 기반 방향 분류 | ❌ 비효율적 |
| 역추세 + EMA200 | ✅ LONG WR 64% |
| RSI > 70 SHORT | ✅ WR 55-62% |
| 중변동 구간 | ✅ 최적 (L 67%, S 80%) |
| 저변동 + 약한 추세 | ❌ WR 30%, 피해야 함 |
| MTF 엄격 필터 | ⚠️ 기회 상실 |
| **비대칭 필터** | ⭐ **가장 균형 잡힘** |

**핵심 메시지**:
> "방향을 완벽히 맞추려 하지 말고, LONG은 엄격하게 SHORT는 유연하게"

---

## 관련 파일

| 파일 | 내용 |
|------|------|
| `scripts/analysis/direction_classification_research.py` | 기본 분류 방법 테스트 |
| `scripts/analysis/improved_direction_filter.py` | LONG 실패 원인 분석 |
| `scripts/analysis/adaptive_direction_strategy.py` | 비대칭 필터 연구 |
| `scripts/analysis/regime_based_direction.py` | 레짐 기반 전략 |
| `scripts/analysis/counter_trend_research.py` | 역추세 전략 연구 |
| `scripts/analysis/atr_direction_research.py` | ATR% 기반 연구 |
| `scripts/analysis/mtf_direction_research.py` | 다중 타임프레임 연구 |
| `results/counter_trend_research_20251212.csv` | 역추세 결과 |
| `results/atr_direction_research_20251212.csv` | ATR 결과 |
| `results/mtf_direction_research_20251212.csv` | MTF 결과 |

---

**작성일**: 2025-12-12
**연구 기간**: 150일 BTC/USDT 15분봉 데이터
**총 분석 조합**: 200+ 전략 조합
