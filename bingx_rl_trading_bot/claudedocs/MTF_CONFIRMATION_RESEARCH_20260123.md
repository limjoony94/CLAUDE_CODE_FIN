# Multi-Timeframe (MTF) Confirmation Research Report

**Date**: 2026-01-23
**Data**: 30,232 candles (105 days)
**Result**: **NOT RECOMMENDED**

---

## Executive Summary

15분/1시간 추세와 5분 패턴 신호의 정렬을 테스트한 결과, MTF 필터링이 성과를 **악화**시키는 것으로 나타났습니다.

---

## 1. MTF Configuration

### Trend Detection Method

```python
# EMA-based trend detection
EMA_PERIOD = 20
SLOPE_THRESHOLD = 0.05  # percent

# Classification Rules:
# BULLISH: price > EMA AND ema_slope > 0.05%
# BEARISH: price < EMA AND ema_slope < -0.05%
# NEUTRAL: otherwise

# Look-Ahead Bias Prevention:
# HTF trend is shifted by 1 period before mapping to 5m
```

### Alignment Definitions

| Alignment | Description |
|-----------|-------------|
| **15m Aligned** | LONG + 15m BULLISH, or SHORT + 15m BEARISH |
| **1h Aligned** | LONG + 1h BULLISH, or SHORT + 1h BEARISH |
| **Both Aligned** | 15m AND 1h both aligned |
| **Counter-Trend** | LONG + both BEARISH, or SHORT + both BULLISH |

---

## 2. Trend Distribution

### 15-Minute Trend

| Trend | Count | Percentage |
|-------|-------|------------|
| BULLISH | 2,769 | 27.5% |
| BEARISH | 2,890 | 28.7% |
| NEUTRAL | 4,419 | 43.8% |

### 1-Hour Trend

| Trend | Count | Percentage |
|-------|-------|------------|
| BULLISH | 898 | 35.6% |
| BEARISH | 1,011 | 40.1% |
| NEUTRAL | 611 | 24.2% |

---

## 3. Signal Alignment Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Total Signals | 248 | 100% |
| **15m Aligned** | 65 | 26.2% |
| **1h Aligned** | 91 | 36.7% |
| **Both Aligned** | 50 | 20.2% |
| **Counter-Trend** | 34 | 13.7% |

---

## 4. Backtest Results

| Strategy | Trades | Win Rate | Compound | Max DD | PF |
|----------|--------|----------|----------|--------|-----|
| **Baseline** | **195** | **50.3%** | **+30.4%** | 26.6% | **1.15** |
| 15m Aligned | 53 | 45.3% | -4.5% | 18.5% | 0.95 |
| 1h Aligned | 69 | 47.8% | +1.9% | 20.4% | 1.05 |
| Both Aligned | 38 | 39.5% | -12.4% | 24.0% | 0.75 |
| Counter-Trend | 30 | 50.0% | +3.8% | 11.5% | 1.14 |

### Key Observations

1. **Baseline outperforms ALL filtered strategies**
   - +30.4% vs best filtered +1.9%

2. **Alignment filtering REDUCES win rate**
   - Baseline: 50.3%
   - 15m Aligned: 45.3% (-5.0%)
   - Both Aligned: 39.5% (-10.8%)

3. **Counter-trend performs similarly to baseline**
   - Suggests patterns capture reversals, not continuations

---

## 5. Walk-Forward Validation

| Strategy | Profitable Folds | Success Rate | Avg Compound |
|----------|-----------------|--------------|--------------|
| **Baseline** | **3/6** | **50.0%** | **+1.7%** |
| 15m Aligned | 2/6 | 33.3% | -0.9% |
| 1h Aligned | 2/6 | 33.3% | -0.4% |
| Both Aligned | 2/6 | 33.3% | -1.7% |

**All filtered strategies fail walk-forward validation (<50% success rate)**

---

## 6. Analysis: Why MTF Filter Failed

### 6.1 Pattern Nature

Our validated patterns appear to be **reversal signals**, not **continuation signals**:

| Pattern Type | Examples | Expected Behavior |
|--------------|----------|-------------------|
| **Reversal** | DN-MD-BD, MU-ST-ST | Trade against recent direction |
| **Continuation** | - | Trade with trend |

The 3-candle patterns likely capture:
- Exhaustion moves (MD, MU)
- Consolidation breakouts (ST, ST)
- Direction changes (DN→BD)

### 6.2 Trend-Signal Mismatch

When we filter for trend alignment:
- LONG signals in BULLISH trends already went up → limited upside
- SHORT signals in BEARISH trends already went down → limited downside

When we allow counter-trend:
- LONG signals in BEARISH trends → catching bottoms
- SHORT signals in BULLISH trends → catching tops

### 6.3 Trade Frequency Impact

| Strategy | Trades | Signal Loss |
|----------|--------|-------------|
| Baseline | 195 | - |
| 15m Aligned | 53 | -73% |
| Both Aligned | 38 | -81% |

Severe trade reduction without quality improvement.

---

## 7. Verdict

### Score Summary

| Criterion | Baseline | MTF Aligned | Winner |
|-----------|----------|-------------|--------|
| Win Rate | 50.3% | 39.5-47.8% | **Baseline** |
| Compound Return | +30.4% | -12.4 to +1.9% | **Baseline** |
| Trade Count | 195 | 38-69 | **Baseline** |
| Walk-Forward | 50% | 33% | **Baseline** |
| Profit Factor | 1.15 | 0.75-1.05 | **Baseline** |

**Score: Baseline 5-0 MTF**

### Final Decision

**DO NOT APPLY MTF Confirmation Filter**

Maintain current baseline strategy without trend filtering.

---

## 8. Recommendations

### 8.1 Keep Current Approach
```yaml
# CURRENT (Recommended)
entry_filter: pattern_match_only
mtf_confirmation: disabled
```

### 8.2 Future Research Directions

If MTF enhancement is still desired, consider:

1. **Opposite Approach**: Use MTF as CONTRA-indicator
   - Filter OUT trend-aligned signals
   - Keep only counter-trend signals
   - Hypothesis: Reversal patterns work better counter-trend

2. **Regime Detection Only**: Use MTF for volatility, not direction
   - High volatility → wider TP/SL
   - Low volatility → tighter TP/SL

3. **Pattern-Specific MTF**: Some patterns may benefit from MTF
   - Research per-pattern correlation with trend

---

## 9. Files Generated

| File | Description |
|------|-------------|
| `scripts/analysis/mtf_confirmation_research.py` | Research script |
| `results/mtf_research/mtf_research_*.json` | JSON results |
| `results/mtf_research/mtf_research_*.csv` | CSV results |
| `claudedocs/MTF_CONFIRMATION_RESEARCH_20260123.md` | This report |

---

## Conclusion

MTF Confirmation filtering **degrades strategy performance** by:
1. Reducing win rate by 5-11%
2. Turning positive compound (+30%) into negative (-12%)
3. Failing walk-forward validation

**Root Cause**: The 3-candle patterns are **reversal signals** that work best when catching turning points, not when trading with the trend.

**Action**: Keep baseline strategy without MTF filtering.
