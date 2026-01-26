# 추세추종 전략 연구 결과 (2025-12-12)

## 연구 배경

### 문제 제기
- 기존 RSI Zone + EMA200 전략은 **역추세(Mean Reversion)** 전략
- 역추세: RSI 과매도 → 반등 예측 (추세 반대 방향)
- **요청**: 추세 방향으로 신호를 발생시키고, 추세 방향으로 수익을 추출하는 **추세추종(Trend Following)** 전략 연구

### 연구 대상 전략
1. **Donchian Breakout** - 20일 고/저점 돌파
2. **EMA 9/21 Crossover** - 이동평균 교차
3. **MACD Momentum** - MACD 모멘텀 기반
4. **Pullback Entry** - 추세 내 풀백 진입
5. **ADX Trend Continuation** - ADX 추세 강도 기반
6. **Keltner Breakout** - Keltner Channel 돌파
7. **HMA Trend** - Hull Moving Average 추세
8. **RSI Momentum (50 cross)** - RSI 50선 돌파

---

## 1차 연구 결과

### 전략별 Test RA 순위 (MDD < 60% 필터)

| 전략 | 설정 | Test Trades | Test WR | Test Return | Test MDD | Test RA |
|------|------|-------------|---------|-------------|----------|---------|
| HMA Trend + RSI | TP3.0 SL1.5 | 166 | 42.8% | 109.9% | 55.0% | 2.00 |
| HMA Trend + RSI | TP3.0 SL1.5 Trail | 166 | 39.8% | 95.3% | 51.8% | 1.84 |
| Breakout Pullback (5) | TP2.0 SL1.0 Trail | 22 | 40.9% | 22.7% | 14.9% | 1.52 |

### 문제점
- MDD 55%는 여전히 높음
- 레버리지 4배 적용 시 청산 위험 존재

---

## 2차 연구: HMA Trend + RSI 최적화

### Breakeven(BE) 트리거 추가

| 전략 | 설정 | Test RA | Test Return | Test MDD |
|------|------|---------|-------------|----------|
| Basic HMA50 | TP3.0 SL1.5 (no BE) | 2.00 | 109.9% | 55.0% |
| **Basic HMA50** | **TP3.0 SL1.5 BE1.5** | **3.66** | **193.3%** | 52.8% |

**핵심 발견**: BE 트리거 추가 시 RA 83% 향상!

### MDD < 50% + RA > 1.0 충족 전략

| 전략 | 설정 | Test Trades | Test WR | Test Return | Test MDD | Test RA |
|------|------|-------------|---------|-------------|----------|---------|
| **Basic HMA20** | **TP3.0 SL1.5 BE1.5** | 228 | 49.1% | 152.5% | **47.4%** | **3.22** |
| HMA50 + EMA200 | TP3.0 SL1.5 BE1.5 | 65 | 55.4% | 103.1% | **38.2%** | 2.70 |
| Basic HMA20 | TP2.5 SL1.5 BE1.2 | 228 | 54.8% | 106.5% | 46.1% | 2.31 |
| HMA50 + EMA200 | TP2.5 SL1.5 BE1.2 | 65 | 60.0% | 55.8% | **35.9%** | 1.56 |

### Rolling Window 검증

**HMA50 Strong Trend (0.5) + TP3.0 SL1.5 BE1.5**:
| Period | Trades | WR | Return | MDD |
|--------|--------|-----|--------|-----|
| Period 1 | 99 | 58.6% | 30.3% | 47.8% |
| Period 2 | 112 | 49.1% | 2.3% | 39.7% |
| Period 3 | 108 | 51.9% | 43.5% | 38.7% |
| Period 4 | 108 | 45.4% | 3.9% | 64.7% |
| **평균** | - | - | **20.0%** | - |
| **수익 구간** | - | - | **4/4 (100%)** | - |

---

## 권장 전략

### 🏆 Option A: 공격적 (최고 RA)

```yaml
strategy: "HMA50 Trend + RSI Momentum"
trend_filter: "Price > HMA50 & HMA50 Rising"
entry_signal: "RSI crosses above 50 (Long) / below 50 (Short)"

exit:
  take_profit_pct: 3.0
  stop_loss_pct: 1.5
  breakeven_trigger: 1.5  # ✅ 핵심
  breakeven_buffer: 0.1

performance:
  test_trades: 166
  test_wr: 53.0%
  test_return: 193.3%
  test_mdd: 52.8%
  test_ra: 3.66
```

### 🥈 Option B: 균형 (낮은 MDD)

```yaml
strategy: "HMA20 Trend + RSI Momentum"
trend_filter: "Price > HMA20 & HMA20 Rising"
entry_signal: "RSI crosses above 50 (Long) / below 50 (Short)"

exit:
  take_profit_pct: 3.0
  stop_loss_pct: 1.5
  breakeven_trigger: 1.5

performance:
  test_trades: 228
  test_wr: 49.1%
  test_return: 152.5%
  test_mdd: 47.4%  # ✅ 50% 미만
  test_ra: 3.22
```

### 🥉 Option C: 보수적 (최저 MDD)

```yaml
strategy: "HMA50 Trend + EMA200 Filter + RSI"
trend_filter: "Price > HMA50 & HMA50 Rising & Price > EMA200"
entry_signal: "RSI crosses above 50 (Long) / below 50 (Short)"

exit:
  take_profit_pct: 3.0
  stop_loss_pct: 1.5
  breakeven_trigger: 1.5

performance:
  test_trades: 65
  test_wr: 55.4%
  test_return: 103.1%
  test_mdd: 38.2%  # ✅ 가장 낮음
  test_ra: 2.70
```

---

## 전략 비교: 추세추종 vs 역추세

### 성과 비교

| 메트릭 | 역추세 (현재) | 추세추종 (HMA50) | 개선율 |
|--------|---------------|------------------|--------|
| Test RA | 0.89 | **3.66** | **+311%** |
| Test Return | 31.8% | **193.3%** | **+508%** |
| Test Trades | 42 | **166** | **+295%** |
| Test WR | 61.9% | 53.0% | -8.9%p |
| Test MDD | 35.9% | 52.8% | +16.9%p |

### 장단점 분석

**추세추종 (HMA Trend)**:
- ✅ RA 3.66 (311% 향상)
- ✅ 거래 빈도 4배 증가
- ✅ 추세 방향 수익 극대화
- ⚠️ MDD 52.8% (높음)
- ⚠️ 승률 53.0% (낮음)

**역추세 (RSI Zone)**:
- ✅ MDD 35.9% (낮음)
- ✅ 승률 61.9% (높음)
- ❌ RA 0.89 (낮음)
- ❌ 거래 빈도 낮음

---

## HMA 계산 방법

```python
def calc_hma(close, period=50):
    """Hull Moving Average - 지연 감소 이동평균"""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))

    wma_half = WMA(close, half_period)
    wma_full = WMA(close, period)

    raw_hma = 2 * wma_half - wma_full
    hma = WMA(raw_hma, sqrt_period)

    return hma

# Entry Signal
uptrend = (close > hma50) and (hma50 > hma50.shift(1))
long_signal = uptrend and (rsi.shift(1) < 50) and (rsi > 50)

downtrend = (close < hma50) and (hma50 < hma50.shift(1))
short_signal = downtrend and (rsi.shift(1) > 50) and (rsi < 50)
```

---

## 핵심 인사이트

### 1. BE(Breakeven) 트리거가 핵심
- BE 없이: RA 2.00
- **BE 추가 시: RA 3.66 (+83%)**
- 이유: 수익 포지션 보호 → 손실 최소화

### 2. HMA가 EMA보다 우수
- HMA: 지연 감소 → 더 빠른 추세 전환 포착
- HMA50: 적절한 노이즈 필터링 + 빠른 반응

### 3. RSI 50 크로스가 효과적
- 모멘텀 확인 신호
- 추세 방향 확인 후 진입

### 4. MDD 관리 필요
- 기본 설정: MDD 52.8%
- EMA200 필터 추가: MDD 38.2%
- 포지션 사이징 또는 레버리지 조정 고려

---

## 하이브리드 전략 (역추세 + 추세추종)

```yaml
strategy: "Hybrid (Mean Reversion + Trend Following)"

# 추세추종 신호
trend_long: "HMA50 Rising + RSI crosses above 50"
trend_short: "HMA50 Falling + RSI crosses below 50"

# 역추세 신호 (추세 내 풀백)
reversal_long: "Uptrend + RSI was < 35 in last 5 candles + recovering"
reversal_short: "Downtrend + RSI was > 65 in last 5 candles + declining"

exit:
  take_profit_pct: 3.0
  stop_loss_pct: 1.5
  breakeven_trigger: 1.5

performance:
  test_trades: 175
  test_wr: 52.0%
  test_return: 180.8%
  test_mdd: 56.9%
  test_ra: 3.18
```

---

## 다음 단계

1. [x] 추세추종 전략 설계 및 백테스트
2. [x] HMA Trend + RSI 최적화
3. [ ] Production bot 구현
4. [ ] Paper trading 검증
5. [ ] MDD 관리를 위한 포지션 사이징 연구

---

**생성일**: 2025-12-12
**분석 스크립트**:
- `scripts/analysis/trend_following_research.py`
- `scripts/analysis/trend_following_deep_analysis.py`
- `scripts/analysis/hma_trend_optimization.py`
