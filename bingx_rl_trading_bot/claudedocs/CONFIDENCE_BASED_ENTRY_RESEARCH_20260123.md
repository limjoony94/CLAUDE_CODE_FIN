# Confidence-Based Entry Research Report

**Date**: 2026-01-23
**Data**: 30,232 candles (105 days)
**Result**: **PROMISING BUT INSUFFICIENT DATA**

---

## Executive Summary

패턴 신뢰도 점수(Confidence Score)를 기반으로 진입 필터링을 적용한 연구 결과, **효과는 있으나 통계적 신뢰도가 부족**하여 즉시 프로덕션 적용은 권장하지 않습니다.

---

## 1. Confidence Score Definition

### Components (가중치)

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification Clarity | 40% | 각 캔들이 해당 타입에 얼마나 명확히 분류되는지 |
| Pattern Historical WR | 30% | 해당 패턴의 과거 승률 (50%=0, 70%=1 정규화) |
| Regime Alignment | 30% | 현재 시장 상황과 패턴의 최적 regime 일치도 |

### Calculation Formula

```python
confidence = (
    0.40 * avg_candle_classification_confidence +
    0.30 * normalized_historical_wr +
    0.30 * regime_alignment_score
)
```

---

## 2. Confidence Distribution

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Validated Signals | 193 |
| Confidence Mean | 0.640 |
| Confidence Std | 0.098 |
| Confidence Range | 0.438 ~ 0.933 |

### Percentiles

| Percentile | Value |
|------------|-------|
| 25% | 0.568 |
| 50% (Median) | 0.627 |
| 75% | 0.703 |
| 90% | 0.773 |

### By Pattern

| Pattern | Count | Mean Confidence | Std |
|---------|-------|-----------------|-----|
| **DN-MD-BD** | 30 | **0.748** | 0.087 |
| BU-ST-ST | 43 | 0.655 | 0.090 |
| MU-DN-MU | 28 | 0.629 | 0.072 |
| MU-ST-ST | 25 | 0.616 | 0.123 |
| IH-DN-DN | 20 | 0.607 | 0.084 |
| D-ST-U | 47 | 0.591 | 0.055 |

**Insight**: DN-MD-BD 패턴이 가장 높은 평균 신뢰도 (0.748)를 보임

---

## 3. Threshold Performance Analysis

| Threshold | Trades | Win Rate | Compound | Max DD | Sharpe | PF |
|-----------|--------|----------|----------|--------|--------|-----|
| 0.00 | 46 | 50.0% | +5.9% | 16.8% | 0.45 | 1.14 |
| 0.30 | 46 | 50.0% | +5.9% | 16.8% | 0.45 | 1.14 |
| 0.40 | 46 | 50.0% | +5.9% | 16.8% | 0.45 | 1.14 |
| 0.50 | 45 | 48.9% | +3.4% | 16.8% | 0.30 | 1.09 |
| 0.55 | 39 | 48.7% | +2.6% | 12.0% | 0.26 | 1.09 |
| 0.60 | 30 | 50.0% | +3.8% | 10.1% | 0.37 | 1.14 |
| **0.65** | **17** | **58.8%** | **+9.3%** | **6.2%** | **1.02** | **1.63** |
| 0.70 | 10 | 50.0% | +1.3% | 4.2% | 0.21 | 1.14 |
| 0.75 | 5 | 40.0% | -1.6% | 4.2% | -0.30 | 0.76 |
| 0.80 | 5 | 40.0% | -1.6% | 4.2% | -0.30 | 0.76 |

### Key Observation

**Threshold 0.65** shows the best quality metrics:
- Win Rate: 50% → 58.8% (**+8.8%**)
- Max Drawdown: 16.8% → 6.2% (**-10.6%**)
- Sharpe Ratio: 0.45 → 1.02 (**+127%**)
- Profit Factor: 1.14 → 1.63 (**+43%**)

**However**: Trade count drops from 46 to 17 (**-63%**), making statistical reliability questionable.

---

## 4. Walk-Forward Validation

### 6-Fold Results (Baseline = Threshold 0.0)

| Fold | Baseline | Threshold=0.65 |
|------|----------|----------------|
| 1 | +1.0% | N/A (insufficient) |
| 2 | +21.2% | N/A |
| 3 | -5.9% | N/A |
| 4 | -5.5% | N/A |
| 5 | +0.2% | N/A |
| 6 | +5.1% | N/A |

**Baseline WF**: 4/6 profitable folds (66.7%)

Note: Threshold 0.65 has too few trades per fold for reliable WF validation.

---

## 5. Verdict

### Quantitative Assessment

| Criterion | Baseline | Threshold=0.65 | Winner |
|-----------|----------|----------------|--------|
| Win Rate | 50.0% | 58.8% | **Threshold** |
| Compound Return | +5.9% | +9.3% | **Threshold** |
| Max Drawdown | 16.8% | 6.2% | **Threshold** |
| Sharpe Ratio | 0.45 | 1.02 | **Threshold** |
| Trade Count | 46 | 17 | **Baseline** |
| Statistical Reliability | Medium | Low | **Baseline** |

### Final Score

- **Quality Metrics**: Threshold wins (4-0)
- **Reliability Metrics**: Baseline wins (2-0)

### Decision

**DO NOT APPLY YET** - Confidence filter shows promise but lacks statistical backing.

---

## 6. Recommendations

### Immediate Action

```python
# Add confidence logging to production bot (no filtering)
def generate_signal(candles):
    pattern = classify_pattern(candles)
    confidence = calculate_confidence(candles, pattern)

    # Log for future analysis
    logger.info(f"Signal: {pattern}, Confidence: {confidence:.3f}")
    save_confidence_metric(pattern, confidence)

    # Execute trade regardless of confidence (for now)
    if pattern in VALIDATED_PATTERNS:
        return pattern
    return None
```

### Short-Term (1-3 months)

1. **Accumulate Data**: Log confidence scores for all signals
2. **Track Outcomes**: Correlate confidence with actual trade results
3. **Re-evaluate**: After 100+ high-confidence signals

### Medium-Term (3-6 months)

1. **If correlation confirmed**: Implement threshold filter
2. **Suggested threshold**: 0.65 (pending validation)
3. **Expected improvement**: WR +8%, DD -10%

---

## 7. Technical Implementation

### Confidence Calculation Module

```python
def calculate_pattern_confidence(
    candle_types: List[str],
    candle_confs: List[float],
    pattern: str,
    regime: str,
    regime_conf: float
) -> float:
    """
    Calculate overall pattern confidence.

    Args:
        candle_types: 3-candle type sequence
        candle_confs: Classification confidence for each candle
        pattern: Pattern string (e.g., "DN-MD-BD")
        regime: Current market regime
        regime_conf: Regime classification confidence

    Returns:
        confidence: 0.0 ~ 1.0
    """
    # Component 1: Classification clarity (40%)
    clarity = np.mean(candle_confs)

    # Component 2: Historical WR (30%)
    hist_wr = PATTERN_STATS.get(pattern, {}).get("wr", 0.5)
    wr_score = np.clip((hist_wr - 0.50) / 0.20, 0, 1)

    # Component 3: Regime alignment (30%)
    best_regime = PATTERN_STATS.get(pattern, {}).get("best_regime", "neutral")
    if regime == best_regime:
        regime_score = 1.0 * regime_conf
    elif regime == "neutral" or best_regime == "neutral":
        regime_score = 0.6
    else:
        regime_score = 0.3

    return 0.40 * clarity + 0.30 * wr_score + 0.30 * regime_score
```

---

## 8. Files Generated

| File | Description |
|------|-------------|
| `scripts/analysis/confidence_based_entry_research.py` | Research script |
| `results/confidence_research/threshold_analysis.csv` | Threshold performance data |
| `results/confidence_research/summary.json` | Summary metrics |

---

## Conclusion

Confidence-based entry filtering shows **quality improvement potential** (+8.8% WR, -10.6% DD at threshold 0.65) but **reduces trade frequency significantly** (-63%).

**Recommended Path**:
1. Log confidence scores in production (no filtering)
2. Accumulate 3-6 months of data
3. Re-evaluate with sufficient sample size
4. Apply filter only if statistical significance confirmed

**Current Status**: Research complete, awaiting data accumulation.
