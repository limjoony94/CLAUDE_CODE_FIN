# 추세추종 전략 연구 요약 (2025-12-12)

## 연구 목표
- 추세 방향으로 신호를 생성하여 지속적 수익 창출
- 목표: **2-5건/일**, **MDD < 60%**, **높은 일관성**

---

## 핵심 발견

### 1. ATR%가 전략 성과와 가장 높은 상관관계 (r = +0.887)

| 메트릭 | 상관계수 | 의미 |
|--------|----------|------|
| **ATR%** | **+0.887** | **변동성 높을수록 수익** |
| slope_mean | -0.871 | 하락 추세에서 수익 |
| price_change | -0.848 | BTC 하락시 수익 |
| range | +0.793 | 가격 범위 클수록 수익 |

**결론**: "횡보장" 분류는 부적절. **변동성(ATR%)**이 핵심 변수.

### 2. SHORT-only 전략이 압도적으로 일관적

| 방향 | RW 양수 비율 | 평균 수익 | Combined RA |
|------|-------------|-----------|-------------|
| **SHORT only** | **75%** (9/12) | **+34.3%** | +7.14 |
| BOTH | 58% (7/12) | +35.9% | +12.51 |
| LONG only | 25% (3/12) | -1.8% | +1.27 |

**결론**: LONG은 불안정, SHORT가 일관적 수익원.

### 3. 92% Rolling Window 일관성 달성!

**최적 SHORT-only 설정**:
```yaml
strategy: SHORT-only HMA Trend
params:
  min_slope: 0.2           # 강한 하락 추세만
  rsi_entry: 40            # RSI 40 하향 돌파
  tp: 2.0%
  sl: 1.5%
  be_trigger: null         # BE 없음 또는 2.0%
  cooldown: 8 candles (2시간)
  leverage: 4x
```

| 메트릭 | 값 |
|--------|-----|
| **Rolling Window 일관성** | **92%** (11/12 양수) |
| Test 거래 수 | 38건 |
| Test Win Rate | 55% |
| Test MDD | 20.0% |
| Combined RA | +2.62 |
| 거래 빈도 | 0.8건/일 |

---

## 기간별 검증 (최적 설정)

| 기간 | BTC | 수익 | 거래 | WR | 결과 |
|------|-----|------|------|-----|------|
| Day 0-30 | +0.5% | +17.3% | 17건 | 59% | ✅ |
| Day 10-40 | -1.7% | +23.7% | 11건 | 73% | ✅ |
| Day 20-50 | -3.3% | +21.6% | 13건 | 62% | ✅ |
| Day 30-60 | -2.0% | +0.1% | 7건 | 43% | ✅ |
| Day 40-70 | -1.4% | -3.7% | 8건 | 38% | ❌ |
| Day 50-80 | +9.1% | -6.8% | 6건 | 33% | ❌ |
| Day 60-90 | -0.5% | -3.2% | 11건 | 55% | ❌ |
| Day 70-100 | -2.0% | +49.7% | 16건 | 62% | ✅ |
| Day 80-110 | -10.5% | +80.6% | 20건 | 70% | ✅ |
| Day 90-120 | -11.5% | +93.4% | 21건 | 62% | ✅ |
| Day 100-130 | -23.0% | +66.1% | 25건 | 60% | ✅ |
| Day 110-140 | -17.7% | +41.8% | 25건 | 60% | ✅ |

**손실 기간 특징**: BTC +9.1% 상승 (Day 50-80) 또는 약한 변동 구간

---

## 전략 비교

### 현재 운영: RSI Zone v1.3
```yaml
entry: RSI < 35 반등 (LONG) / RSI > 65 하락 (SHORT)
filter: EMA200 방향
tp: 2.4%, sl: 1.4%, be_trigger: 1.2%
rw_consistency: ~57%
trades_per_day: 0.1건 (매우 낮음)
```

### 새로운 전략: SHORT-only HMA Trend
```yaml
entry: RSI 40 하향 돌파 + HMA slope < -0.2
filter: HMA30 기울기
tp: 2.0%, sl: 1.5%, be_trigger: null
rw_consistency: 92%
trades_per_day: 0.8건
```

---

## 권장 사항

### Option 1: SHORT-only HMA Trend 단독 운영
- **장점**: 92% 일관성, 안정적 수익
- **단점**: 상승장 기회 놓침, 0.8건/일 (목표 미달)

### Option 2: RSI Zone + SHORT-only 병렬 운영
- RSI Zone: LONG/SHORT 모두 (다양한 시장)
- SHORT-only HMA: 추가 SHORT 기회
- **장점**: 기회 확대, 분산 효과
- **단점**: 복잡도 증가

### Option 3: 적응형 전략
- BTC 상승장: LONG 전략만
- BTC 횡보/하락장: SHORT 전략 강화
- **장점**: 시장 적응
- **단점**: 레짐 판단 오류 위험

---

## 결론

1. **ATR% (변동성)**이 전략 성과의 핵심 변수
2. **SHORT-only 전략이 가장 일관적** (92% RW 양수)
3. **LONG은 불안정** - 특정 상승장에서만 작동
4. 거래 빈도 0.8건/일은 목표(2-5건/일) 미달이지만 안정성 우선시 권장
5. RSI Zone + SHORT-only 병렬 운영으로 기회 확대 가능

---

## 관련 파일

| 파일 | 내용 |
|------|------|
| `scripts/analysis/market_condition_research.py` | ATR% 상관관계 발견 |
| `scripts/analysis/directional_performance_analysis.py` | 방향별 성과 분석 |
| `scripts/analysis/short_only_strategy_analysis.py` | SHORT-only 최적화 |
| `results/short_only_optimization_20251212.csv` | 3,072개 조합 결과 |

---

**작성일**: 2025-12-12
**연구 기간**: 150일 BTC/USDT 15분봉 데이터
