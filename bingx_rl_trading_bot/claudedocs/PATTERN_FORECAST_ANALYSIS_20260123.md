# Pattern Forecast Analysis Report

**Date**: 2026-01-23
**Data**: 30,232 candles (2025-10-08 ~ 2026-01-21, ~105 days)
**Patterns Analyzed**: 475 unique 3-candle patterns, 65 with sufficient samples (≥30)

---

## Executive Summary

### Key Findings

| Finding | Implication |
|---------|-------------|
| **SNR = 0.037 (avg)** | 가격 변화 크기 예측은 거의 불가능 |
| **Best SNR = 0.117** | 방향 예측도 노이즈 대비 신호가 매우 약함 |
| **Avg \|Mean\| = 0.0082%** | 패턴 후 평균 가격 변화가 매우 작음 |
| **Avg Std = 0.214%** | 변동성이 평균의 26배 (높은 불확실성) |

### Critical Insight

> **패턴은 가격 방향을 약하게 예측할 수 있지만, 크기(magnitude) 예측은 본질적으로 노이즈입니다.**

---

## 1. Signal-to-Noise Ratio (SNR) Analysis

```
SNR = |Mean Return| / Std Return

전체 65 패턴:
  - 평균 SNR: 0.0369
  - 최고 SNR: 0.1170 (BU-U-U)
  - 평균 |Mean|: 0.0082%
  - 평균 Std: 0.2139%
```

### SNR 해석

| SNR Range | Interpretation | Trading Implication |
|-----------|----------------|---------------------|
| > 0.50 | Strong signal | 크기 예측 가능 |
| 0.20 - 0.50 | Moderate signal | 방향 + 약한 크기 예측 |
| 0.05 - 0.20 | Weak signal | 방향만 약하게 예측 |
| **< 0.05** | **Noise dominated** | **방향도 불확실** |

→ 현재 패턴들의 SNR (0.03-0.12)은 "Noise dominated ~ Weak signal" 범위

---

## 2. Multi-Horizon Analysis

### Best Performing Patterns (Consistent across horizons)

| Pattern | T+1 WR | T+3 WR | T+6 WR | T+12 WR | Avg WR | Type |
|---------|--------|--------|--------|---------|--------|------|
| **DN-U-BD** | 55.0% | 55.2% | 55.7% | **57.6%** | **55.9%** | Accumulation |
| DN-BD-DN | 54.4% | 55.6% | 56.9% | 56.7% | 55.8% | Neutral |
| BD-BD-DN | 55.5% | 56.6% | 55.8% | 54.0% | 55.5% | Neutral |
| U-BD-BD | 54.1% | 55.1% | 53.4% | 55.4% | 54.6% | Neutral |
| BD-BU-DN | 54.2% | 54.5% | 54.2% | 54.2% | 54.4% | Neutral |

**Key Pattern: DN-U-BD**
- T+1에서 T+12로 갈수록 WR 상승 (55% → 57.6%)
- 이는 "accumulation" 패턴 - 시간이 지날수록 예측력 증가
- 추천: 더 긴 holding period, 넓은 TP

---

## 3. Pattern Type Classification

### Continuation Patterns (신호 감쇠)
빠른 방향성 후 신호가 약해짐 → **빠른 익절 권장**

| Pattern | T+1 WR | T+12 WR | Decay |
|---------|--------|---------|-------|
| BU-U-BD | 57.3% | 47.1% | -10.2% |
| BD-BD-BU | 55.5% | 49.0% | -6.5% |
| U-BD-BU | 51.7% | 45.4% | -6.3% |

→ **전략**: Tight TP (1.5-2.0%), 빠른 청산

### Reversal/Accumulation Patterns (신호 강화)
초기 신호가 약하지만 시간이 지나며 강해짐 → **홀딩 권장**

| Pattern | T+1 WR | T+12 WR | Improvement |
|---------|--------|---------|-------------|
| U-BD-U | 50.6% | 57.4% | +6.8% |
| BD-BD-U | 48.4% | 54.8% | +6.4% |
| BD-U-U | 46.8% | 53.0% | +6.1% |
| BD-DN-U | 48.6% | 54.3% | +5.7% |

→ **전략**: Wider TP (3.0-4.0%), 인내심 있는 홀딩

---

## 4. Regime Analysis

### High Volatility (ATR > 70th percentile)

| Pattern | Normal WR | High Vol WR | Change |
|---------|-----------|-------------|--------|
| BU-U-BD | 57.3% | 58.6% | +1.3% |
| U-BD-DN | 53.9% | **60.6%** | **+6.7%** |
| BD-BU-DN | 54.2% | 56.5% | +2.3% |

→ U-BD-DN은 고변동성에서 특히 효과적

### Low Volatility (ATR < 30th percentile)

| Pattern | Normal WR | Low Vol WR | Change |
|---------|-----------|------------|--------|
| BD-BD-BD | 54.5% | **64.6%** | **+10.1%** |
| BU-U-BD | 57.3% | 54.5% | -2.8% |

→ BD-BD-BD는 저변동성에서 매우 효과적 (낙폭과대 후 반등)

### Uptrend (EMA slope > 0.1%)

| Pattern | Normal WR | Uptrend WR | Interpretation |
|---------|-----------|------------|----------------|
| BU-U-BD | 57.3% | **67.3%** | 상승추세에서 강력 |
| BD-BD-DN | 55.5% | 62.5% | 조정 후 반등 |

---

## 5. Forecasting Model Recommendations

### 현실적 접근법

```python
# 현재 상태: 단순 패턴 매칭
if pattern in VALIDATED_PATTERNS:
    execute_trade()

# 개선안 1: Confidence-based filtering
confidence = calculate_pattern_confidence(candles)
if pattern in VALIDATED_PATTERNS and confidence > 0.6:
    execute_trade()

# 개선안 2: Regime-aware execution
regime = get_current_regime()  # high_vol, low_vol, uptrend, downtrend
pattern_wr = REGIME_PATTERN_WR[regime][pattern]
if pattern_wr > 0.55:
    execute_trade()

# 개선안 3: Dynamic TP/SL
if pattern_type == 'CONTINUATION':
    tp, sl = 2.0, 1.5  # Quick profit
elif pattern_type == 'REVERSAL':
    tp, sl = 3.5, 2.5  # Hold longer
```

### Why NOT to Build a Complex ML Model

1. **Low SNR**: 데이터 자체가 노이즈 지배적
2. **Sample Size**: 65 패턴 × regime = 희소 데이터
3. **Overfitting Risk**: 복잡한 모델은 과적합 위험 높음
4. **Interpretability**: 블랙박스 모델은 디버깅 어려움

→ **권장**: 간단한 rule-based system + regime awareness

---

## 6. Dynamic TP/SL Configuration

### Recommended Settings

```python
PATTERN_TPSL_CONFIG = {
    # High consistency patterns (Avg WR >= 54%)
    "DN-U-BD": {"tp": 2.4, "sl": 1.9, "type": "ACCUMULATION"},
    "DN-BD-DN": {"tp": 2.8, "sl": 2.2, "type": "NEUTRAL"},
    "BD-BD-DN": {"tp": 3.4, "sl": 2.7, "type": "NEUTRAL"},

    # Continuation patterns (Quick profit)
    "BU-U-BD": {"tp": 2.0, "sl": 1.5, "type": "CONTINUATION"},
    "BD-BD-BU": {"tp": 2.2, "sl": 1.8, "type": "CONTINUATION"},

    # Reversal patterns (Hold longer)
    "U-BD-U": {"tp": 3.5, "sl": 2.5, "type": "REVERSAL"},
    "BD-BD-U": {"tp": 3.2, "sl": 2.3, "type": "REVERSAL"},
}

# Regime multipliers
REGIME_MULTIPLIERS = {
    "high_vol": {"tp": 1.3, "sl": 1.3},
    "low_vol": {"tp": 0.7, "sl": 0.7},
    "uptrend": {"tp": 1.1, "sl": 0.9},
    "downtrend": {"tp": 0.9, "sl": 1.1},
}
```

---

## 7. Implementation Roadmap

### Phase 1: Data Enhancement (1 week)
- [ ] 패턴별 historical distribution 저장
- [ ] Regime 분류 로직 추가 (ATR percentile, EMA slope)
- [ ] Confidence score 계산 함수 구현

### Phase 2: Dynamic TP/SL (1 week)
- [ ] PATTERN_TPSL_CONFIG 적용
- [ ] Regime-based multiplier 적용
- [ ] 백테스트 검증

### Phase 3: Real-time Monitoring (2 weeks)
- [ ] 포지션 보유 중 패턴 변화 모니터링
- [ ] 반전 신호 감지 시 TP/SL 조정
- [ ] Exit signal 로직 추가

### Phase 4: Evaluation (Ongoing)
- [ ] Expected vs Actual 비교
- [ ] Regime별 성과 추적
- [ ] 월간 리밸런싱

---

## 8. Conclusions

### What Works
1. **방향 예측**: 53-58% Win Rate (edge 있음)
2. **Regime awareness**: 특정 조건에서 패턴 효과 증가
3. **Pattern type**: Continuation vs Reversal 구분 유의미

### What Doesn't Work
1. **크기 예측**: SNR 너무 낮음 (0.03-0.12)
2. **복잡한 ML 모델**: 데이터 부족, 과적합 위험
3. **Fixed TP/SL**: 패턴별/regime별 최적값 다름

### Final Recommendation

> **Simple rule-based system with regime awareness > Complex ML model**
>
> 1. 검증된 패턴만 사용 (Avg WR >= 54%)
> 2. Regime에 따른 TP/SL 조정
> 3. Pattern type에 따른 holding strategy
> 4. Continuous monitoring and adaptation

---

## Appendix: Raw Data Files

- `results/forecast_research/pattern_forecast_stats_*.json` - 전체 통계
- `results/forecast_research/pattern_forecast_report_*.md` - 리포트
- `scripts/analysis/pattern_forecast_research.py` - 분석 스크립트
