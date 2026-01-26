# Dynamic TP/SL Evaluation Report

**Date**: 2026-01-23
**Data**: 30,232 candles (105 days)
**Result**: **NOT RECOMMENDED**

---

## Executive Summary

Dynamic TP/SL 전략을 Fixed TP/SL과 비교 평가한 결과, Dynamic 접근법이 **현저히 낮은 성과**를 보여 적용을 권장하지 않습니다.

---

## 1. Comparison Results

| Metric | Fixed (Current) | Dynamic (Proposed) | Difference |
|--------|-----------------|-------------------|------------|
| Total Trades | 142 | 202 | +60 |
| **Win Rate** | **51.4%** | 44.6% | **-6.9%** |
| **Compound Return** | **+50.3%** | -61.1% | **-111.4%** |
| **Max Drawdown** | **31.2%** | 66.8% | **+35.6%** |
| **Sharpe Ratio** | **6.68** | -7.22 | **-13.91** |
| Profit Factor | **1.27** | 0.77 | -0.50 |
| Avg Win % | 1.13% | 0.97% | -0.16% |
| Avg Loss % | -0.94% | -1.01% | -0.07% |
| Max Consecutive Loss | 5 | 7 | +2 |

### Walk-Forward Validation (6-Fold)

| Strategy | Profitable Folds | Success Rate |
|----------|-----------------|--------------|
| Fixed | 5/6 | 83.3% |
| Dynamic | 2/6 | 33.3% |

### Statistical Significance

- Bootstrap p-value: **0.081** (Not significant at 0.05)
- T-test p-value: **0.081** (Not significant)

Note: While not statistically significant, the practical difference is substantial (-111% compound return difference).

---

## 2. Configurations Tested

### Fixed TP/SL (Current Production)

```python
CURRENT_CONFIG = {
    "tp_pct": 2.5,
    "sl_pct": 2.0,
    "long_patterns": ["DN-MD-BD", "BU-ST-ST", "MU-DN-MU"],
    "short_patterns": ["MU-ST-ST", "IH-DN-DN", "D-ST-U"],
}
```

### Dynamic TP/SL (Proposed)

```python
DYNAMIC_CONFIG = {
    # Pattern-specific TP/SL based on pattern type
    "pattern_tpsl": {
        "DN-MD-BD": {"tp": 3.0, "sl": 1.8, "type": "REVERSAL"},
        "BU-ST-ST": {"tp": 2.2, "sl": 2.0, "type": "NEUTRAL"},
        "MU-DN-MU": {"tp": 2.5, "sl": 2.0, "type": "NEUTRAL"},
        "MU-ST-ST": {"tp": 2.0, "sl": 1.5, "type": "CONTINUATION"},
        "IH-DN-DN": {"tp": 2.8, "sl": 2.2, "type": "REVERSAL"},
        "D-ST-U": {"tp": 2.5, "sl": 2.0, "type": "NEUTRAL"},
        # Additional patterns from forecast research
        "DN-U-BD": {"tp": 2.4, "sl": 1.9, "type": "ACCUMULATION", "bias": "LONG"},
        "U-BD-DN": {"tp": 2.0, "sl": 1.5, "type": "CONTINUATION", "bias": "LONG"},
        "BD-BU-DN": {"tp": 2.2, "sl": 1.8, "type": "CONTINUATION", "bias": "LONG"},
    },
    # Regime-based multipliers
    "regime_mult": {
        "high_vol": {"tp": 1.2, "sl": 1.2},
        "low_vol": {"tp": 0.8, "sl": 0.8},
        "uptrend": {"tp": 1.1, "sl": 0.9},
        "downtrend": {"tp": 0.9, "sl": 1.1},
    },
}
```

---

## 3. Why Dynamic Failed

### 3.1 Additional Pattern Quality

| Pattern | Research WR | Actual WR | Gap |
|---------|-------------|-----------|-----|
| DN-U-BD | 55.0% | ~45% | -10% |
| U-BD-DN | 53.9% | ~42% | -12% |
| BD-BU-DN | 54.2% | ~40% | -14% |

The additional patterns had lower real-world win rates than the forecast research suggested.

### 3.2 Low Signal-to-Noise Ratio

From the Pattern Forecast Analysis:
- **Average SNR = 0.037** (noise-dominated)
- **Best SNR = 0.117** (still very weak)

Pattern type classification (Continuation vs Reversal) is inherently noisy. Adjusting TP/SL based on this classification amplifies noise rather than signal.

### 3.3 Regime Multiplier Overfitting

The regime-based multipliers were derived from historical data and showed signs of overfitting:
- High volatility × 1.2 sometimes extended losses
- Low volatility × 0.8 sometimes cut profits short
- Trend multipliers often misaligned with actual market direction

### 3.4 Trade Frequency vs Quality Tradeoff

- Fixed: 142 trades with 51.4% WR = quality focus
- Dynamic: 202 trades with 44.6% WR = quantity over quality
- The 60 additional trades from new patterns degraded overall performance

---

## 4. Verdict

### Score Summary

| Criterion | Winner | Points |
|-----------|--------|--------|
| Compound Return | Fixed | +2 |
| Win Rate | Fixed | +1 |
| Max Drawdown | Fixed | +1 |
| Sharpe Ratio | Fixed | +1 |
| Profit Factor | Fixed | +1 |
| Walk-Forward | Fixed | +1 |
| **Total** | **Fixed** | **7-0** |

### Final Decision

**DO NOT APPLY Dynamic TP/SL**

Maintain current Fixed TP/SL configuration:
```yaml
tp_pct: 2.5
sl_pct: 2.0
patterns:
  long: [DN-MD-BD, BU-ST-ST, MU-DN-MU]
  short: [MU-ST-ST, IH-DN-DN, D-ST-U]
```

---

## 5. Recommendations for Future Research

### 5.1 Pattern Addition Criteria

Before adding new patterns, require:
1. **Minimum 6 months of out-of-sample validation**
2. **SNR > 0.2** for the pattern
3. **Walk-forward success rate > 70%** (at least 5/7 folds)

### 5.2 Regime-Based Strategies

Use regime detection as a **filter** (skip trades), not a **modifier** (adjust TP/SL):

```python
# Recommended approach
if regime == "low_vol" and pattern in ["BU-U-BD", "BD-BD-BU"]:
    skip_trade = True  # These patterns underperform in low volatility

# Not recommended
tp = base_tp * regime_multiplier  # Amplifies noise
```

### 5.3 When Dynamic TP/SL Might Work

Dynamic TP/SL could be reconsidered only if:
1. SNR improves to > 0.2 through better classification
2. Multi-horizon analysis shows consistent >56% WR
3. Backtest shows > Fixed compound return consistently

---

## 6. Files Generated

| File | Description |
|------|-------------|
| `scripts/analysis/pattern_dynamic_tpsl_evaluation.py` | Evaluation script |
| `scripts/analysis/pattern_forecast_research.py` | Forecast research script |
| `claudedocs/PATTERN_FORECAST_ANALYSIS_20260123.md` | Forecast analysis report |
| `results/forecast_research/pattern_forecast_*.json` | Raw statistics |

---

## Conclusion

The pattern forecast research provided valuable insights into pattern behavior (SNR, multi-horizon WR, regime dependence), but attempting to operationalize these findings through Dynamic TP/SL resulted in significantly worse performance than the simple Fixed approach.

**Key Lesson**: Low SNR (0.037) means pattern predictions are noise-dominated. Adding complexity (dynamic TP/SL, regime multipliers) amplifies noise rather than improving signal.

**Action**: Keep current Fixed TP/SL settings. Focus future research on improving SNR through better pattern classification before attempting dynamic adjustments.
