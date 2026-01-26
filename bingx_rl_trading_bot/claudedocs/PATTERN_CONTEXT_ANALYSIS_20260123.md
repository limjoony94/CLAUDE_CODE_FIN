# Pattern Context Analysis Research Report

**Date**: 2026-01-23
**Author**: Claude
**Status**: Completed

---

## Executive Summary

연구 목적: 패턴의 성과가 추세 내 위치(Context)에 따라 어떻게 달라지는지 분석

### 핵심 발견

| 필터 | Trades | Win Rate | Compound | vs Baseline |
|------|--------|----------|----------|-------------|
| Baseline | 57 | 57.9% | +117.9% | - |
| **Counter-trend only** | 30 | **76.7%** | **+223.7%** | **+105.8%** |
| SHORT signals only | 28 | 75.0% | +186.0% | +68.1% |
| Exclude LONG RSI<35 | 51 | 60.8% | +146.0% | +28.1% |

### 가설 검증

**원래 가설**: 반전 패턴은 추세 끝에서, 지속 패턴은 추세 중간에서 효과적

**수정된 결론**:
- ✅ 가설 부분 지지: Context가 성과에 영향을 미침
- ❗ 예상과 다름: **역추세 신호가 추세 추종보다 효과적**
- ❗ LONG 과매도 진입은 오히려 손실

---

## Methodology

### Data
- Source: `btc_5m_90days_validation.csv`
- Period: 90 days (25,920 candles)
- Timeframe: 5-minute

### Context Indicators
1. **EMA Trend**: EMA20 > EMA50 → uptrend, else downtrend
2. **RSI**: 14-period RSI
3. **ADX**: 14-period ADX (trend strength)
4. **EMA Distance**: (close - EMA20) / EMA20 * 100

### Trend State Classification
```
STRONG_UP:    RSI > 70 or EMA_dist > 1.5%
MODERATE_UP:  EMA trend = up, RSI 50-70
WEAK_UP:      EMA trend = up, RSI 40-50
NEUTRAL:      ADX < 20
WEAK_DOWN:    EMA trend = down, RSI 50-60
MODERATE_DOWN: EMA trend = down, RSI 30-50
STRONG_DOWN:  RSI < 30 or EMA_dist < -1.5%
```

### Signal Alignment Classification
```
TREND_FOLLOWING: LONG in uptrend, SHORT in downtrend
COUNTER_TREND:   LONG in downtrend, SHORT in uptrend
NEUTRAL:         ADX < 20
```

---

## Detailed Results

### 1. Alignment × Extremity Analysis

| Alignment | Extremity | Trades | Win Rate | Compound |
|-----------|-----------|--------|----------|----------|
| **counter_trend** | **weak** | 11 | **81.8%** | **+65.6%** |
| neutral | neutral | 7 | 85.7% | +43.1% |
| counter_trend | moderate | 5 | 80.0% | +24.6% |
| counter_trend | extreme | 7 | 57.1% | +9.6% |
| trend_following | weak | 9 | 44.4% | -4.2% |
| trend_following | extreme | 6 | 33.3% | -11.4% |
| trend_following | moderate | 12 | 33.3% | -20.7% |

**결론**: 역추세 신호가 모든 구간에서 추세추종보다 우수

### 2. RSI Zone Analysis

| Signal | RSI Zone | Trades | Win Rate | Compound |
|--------|----------|--------|----------|----------|
| **SHORT** | **overbought (70-100)** | 7 | **85.7%** | **+43.9%** |
| SHORT | high (55-70) | 3 | 100% | +23.9% |
| SHORT | low (30-45) | 10 | 60% | +18.8% |
| SHORT | mid (45-55) | 6 | 66.7% | +17.1% |
| LONG | high (55-70) | 10 | 50% | +2.3% |
| LONG | low (30-45) | 4 | 50% | +0.9% |
| LONG | overbought (70-100) | 8 | 37.5% | -11.0% |
| **LONG** | **oversold (0-30)** | 5 | **20%** | **-17.4%** |

**결론**:
- SHORT + overbought 조합이 최강
- LONG + oversold (역설적으로) 최악

### 3. Signal by Trend State

#### LONG Signals
| Trend State | Trades | Win Rate | Compound |
|-------------|--------|----------|----------|
| neutral | 3 | 100% | +23.2% |
| weak_down | 6 | 67% | +15.9% |
| strong_up | 6 | 33% | -11.4% |
| weak_up | 6 | 33% | -11.4% |
| strong_down | 2 | 0% | -12.2% |
| moderate_up | 5 | 20% | -17.4% |

#### SHORT Signals
| Trend State | Trades | Win Rate | Compound |
|-------------|--------|----------|----------|
| **weak_up** | 5 | **100%** | **+42.8%** |
| **moderate_up** | 4 | **100%** | **+33.0%** |
| strong_up | 5 | 80% | +24.8% |
| neutral | 4 | 75% | +16.2% |
| weak_down | 3 | 67% | +8.2% |
| moderate_down | 7 | 43% | -4.0% |

**결론**: SHORT은 상승 추세에서 압도적으로 좋음

### 4. Pattern × Best Context

| Pattern | Signal | Best Context | Win Rate | Compound |
|---------|--------|--------------|----------|----------|
| IH-DN-DN | SHORT | moderate_up | 100% | +23.9% |
| MU-ST-ST | SHORT | moderate_down | 75% | +16.2% |
| MU-DN-MU | LONG | weak_down | 75% | +15.4% |
| D-ST-U | SHORT | weak_up | 100% | +15.3% |
| DN-MD-BD | LONG | neutral | 100% | +14.9% |

---

## Filter Comparison

| Filter | Trades | Win Rate | Compound | Edge Score |
|--------|--------|----------|----------|------------|
| Baseline | 57 | 57.9% | +117.9% | - |
| **Counter-trend only** | 30 | **76.7%** | **+223.7%** | 2189.6 |
| SHORT signals only | 28 | 75.0% | +186.0% | 1809.2 |
| Exclude LONG RSI<35 | 51 | 60.8% | +146.0% | 933.9 |
| LONG: RSI < 50 | 38 | 63.2% | +123.5% | 607.0 |
| SHORT: uptrend only | 44 | 59.1% | +94.0% | 438.0 |

---

## Recommendations

### Option A: Counter-Trend Filter (Aggressive)

```python
def filter_counter_trend(signal, row):
    ema_trend = row.get('ema_trend', 0)
    adx = row.get('adx', 25)

    if adx < 20:  # 횡보면 허용
        return True

    if signal == "LONG" and ema_trend == 0:  # LONG in downtrend
        return True
    if signal == "SHORT" and ema_trend == 1:  # SHORT in uptrend
        return True

    return False
```

**Expected Performance**:
- Trades: 30 (↓47%)
- Win Rate: 76.7% (↑18.8pp)
- Compound: +223.7% (↑105.8%)

### Option B: Exclude LONG Oversold (Conservative)

```python
def filter_exclude_long_oversold(signal, row):
    if signal == "LONG":
        return row.get('rsi', 50) >= 35
    return True
```

**Expected Performance**:
- Trades: 51 (↓10%)
- Win Rate: 60.8% (↑2.9pp)
- Compound: +146.0% (↑28.1%)

### Option C: SHORT Only Strategy

```python
def filter_short_only(signal, row):
    return signal == "SHORT"
```

**Expected Performance**:
- Trades: 28 (↓51%)
- Win Rate: 75.0% (↑17.1pp)
- Compound: +186.0% (↑68.1%)

---

## Implementation Considerations

### Pros of Context Filter
1. 승률 대폭 향상 (57.9% → 76.7%)
2. 복리 수익 2배 증가 (+117.9% → +223.7%)
3. 손실 거래 감소

### Cons of Context Filter
1. 거래 빈도 감소 (57 → 30)
2. 놓치는 수익 기회 발생
3. 백테스트 기간 제한 (90일)

### Next Steps
1. [ ] 더 긴 기간 데이터로 검증 (6개월+)
2. [ ] Walk-forward 테스트 수행
3. [ ] 프로덕션 적용 전 페이퍼 트레이딩

---

## Files Generated

| File | Description |
|------|-------------|
| `scripts/analysis/pattern_context_analysis.py` | V1 분석 스크립트 |
| `scripts/analysis/pattern_context_v2.py` | V2 분석 스크립트 |
| `scripts/analysis/pattern_context_filter_test.py` | 필터 비교 테스트 |
| `results/pattern_context_trades_*.csv` | 거래 로그 |
| `results/context_filter_comparison_*.csv` | 필터 비교 결과 |

---

## Conclusion

**패턴의 위치가 성과에 중요한 영향을 미친다**는 가설이 지지되었습니다.
특히 **역추세 진입** (상승장에서 SHORT, 하락장에서 LONG)이 추세추종보다 훨씬 효과적입니다.

이는 현재 패턴들이 **추세 반전을 잡는 데** 효과적임을 의미합니다.

권장: **Counter-trend 필터**를 적용하여 승률과 수익을 개선할 것을 권장합니다.
단, 거래 빈도 감소를 감안해야 합니다.
