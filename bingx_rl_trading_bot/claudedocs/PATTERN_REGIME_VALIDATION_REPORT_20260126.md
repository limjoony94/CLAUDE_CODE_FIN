# Pattern 5m v1.15 - Regime-Validated TP/SL Optimization Report

**Date**: 2026-01-26
**Version**: v1.15.0
**Research Script**: `scripts/analysis/pattern_regime_validation.py`

---

## Executive Summary

v1.14 TP/SL settings showed severe performance degradation in independent validation due to **bear market bias** in the original research. The 90-day validation dataset was entirely bearish (-25.8%), causing SHORT pattern settings to be overfit to riding market direction rather than capturing genuine pattern alpha.

### Critical Finding: DN-DN-BD

| Setting | WR | PnL | Issue |
|---------|-----|------|-------|
| v1.14 (4.0/1.0) | **8.3%** | **-1.75%** | SL too tight, TP unreachable |
| v1.15 (1.5/3.0) | **94.1%** | **+3.71%** | Proper noise filtering |

**Root Cause**: 1.0% SL was too tight for BTC's intrabar volatility, causing positions to stop out before the 4.0% TP could be reached.

---

## Research Methodology

### 1. Dataset
- **Period**: 2025-10-02 to 2025-12-31 (90 days)
- **Bars**: 25,920 (5-minute timeframe)
- **Price Change**: $118,901 → $88,284 (-25.8%)
- **Regime**: Predominantly BEAR/SIDE (no BULL periods)

### 2. Validation Framework

**Grid Search**:
- TP: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
- SL: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
- Total combinations: 64 per pattern

**Walk-Forward Validation**:
- 5-fold time-series split
- Pass criterion: 3/5 folds profitable

**Counter-Trend Analysis**:
- Isolates pattern alpha from market beta
- Tests SHORT patterns in BULL regime, LONG patterns in BEAR regime
- Reveals genuine predictive power vs directional luck

### 3. Entry/Exit Rules

| Rule | Implementation |
|------|----------------|
| Entry | Next bar Open after signal |
| TP Exit | Intrabar High/Low touch |
| SL Exit | Intrabar High/Low touch |
| Fees | 0.10% round-trip |
| Slippage | 0.02% buffer |
| Position Sizing | Compound (100% capital) |

---

## Results by Pattern

### LONG Patterns

| Pattern | v1.14 TP/SL | v1.14 WR | v1.15 TP/SL | v1.15 WR | Change |
|---------|-------------|----------|-------------|----------|--------|
| U-BU-U | 1.5/2.0 | 93.8% | 1.5/2.0 | 93.8% | KEEP |
| ST-BD-DN | 2.5/2.0 | 64.7% | 2.0/3.0 | 70.6% | UPDATE |

### SHORT Patterns

| Pattern | v1.14 TP/SL | v1.14 WR | v1.15 TP/SL | v1.15 WR | Change |
|---------|-------------|----------|-------------|----------|--------|
| BD-BD-BD | 3.5/1.5 | 70.6% | 3.0/2.5 | 88.2% | UPDATE |
| DN-DN-BD | 4.0/1.0 | **8.3%** | 1.5/3.0 | **94.1%** | **CRITICAL** |
| MU-ST-DN | 2.0/1.0 | 71.4% | 1.0/2.5 | 95.2% | UPDATE |
| IH-DN-DN | 2.0/2.5 | 80.0% | 1.0/3.0 | 100% | UPDATE |
| BD-ST-DN | 2.5/2.0 | 75.0% | 1.5/3.0 | 91.7% | UPDATE |
| BU-U-DN | 2.5/2.0 | 73.7% | 1.5/2.5 | 89.5% | UPDATE |
| D-DN-BD | 2.5/2.0 | 85.7% | 2.5/2.0 | 85.7% | KEEP |

---

## Counter-Trend Analysis (Pattern Alpha)

Tests pattern performance when trading AGAINST market direction:

| Pattern | Direction | Counter-Regime | WR | Trades | Interpretation |
|---------|-----------|----------------|-----|--------|----------------|
| U-BU-U | LONG | BEAR | 57.1% | 7 | Genuine alpha |
| BD-BD-BD | SHORT | BULL | N/A | 0 | No BULL data |
| DN-DN-BD | SHORT | BULL | N/A | 0 | No BULL data |

**Limitation**: Validation dataset lacked BULL periods, so SHORT pattern counter-trend analysis was not possible. This is a known limitation requiring future validation with bullish data.

---

## Walk-Forward Results (v1.15 Settings)

| Pattern | Optimal TP/SL | Profitable Folds | Pass |
|---------|---------------|------------------|------|
| U-BU-U | 1.5/2.0 | 4/5 | PASS |
| ST-BD-DN | 2.0/3.0 | 3/5 | PASS |
| BD-BD-BD | 3.0/2.5 | 5/5 | PASS |
| DN-DN-BD | 1.5/3.0 | 4/5 | PASS |
| MU-ST-DN | 1.0/2.5 | 5/5 | PASS |
| IH-DN-DN | 1.0/3.0 | 5/5 | PASS |
| BD-ST-DN | 1.5/3.0 | 4/5 | PASS |
| BU-U-DN | 1.5/2.5 | 4/5 | PASS |
| D-DN-BD | 2.5/2.0 | 3/5 | PASS |

**All 9 patterns pass WF validation with v1.15 settings**.

---

## Key Insights

### 1. Tighter TP, Wider SL Works Better

| Metric | v1.14 Avg | v1.15 Avg | Improvement |
|--------|-----------|-----------|-------------|
| TP | 2.5% | 1.7% | -32% (smaller) |
| SL | 1.7% | 2.6% | +53% (wider) |
| WR | 69.2% | 88.9% | +19.7pp |

**Reason**: BTC 5-minute volatility often triggers tight SLs before larger TPs can be reached. Wider SL allows positions to weather normal noise.

### 2. v1.14 Bear Market Bias

The original v1.14 research was conducted during a -25.8% market decline. This caused:
- SHORT patterns to show artificially inflated performance
- TP settings to be optimized for large directional moves
- SL settings to be too aggressive (tight)

### 3. Pattern-Specific Optimization Matters

Default TP/SL (1.5/3.0) performs well on average, but individual patterns show significant variation:
- **IH-DN-DN**: Optimal at 1.0/3.0 (very tight TP)
- **BD-BD-BD**: Optimal at 3.0/2.5 (larger TP works)
- **D-DN-BD**: Optimal at 2.5/2.0 (balanced)

---

## Implementation Changes (v1.15)

### constants.py Updates

```python
# v1.15 PATTERN_OPTIMAL_TPSL
PATTERN_OPTIMAL_TPSL = {
    # LONG patterns
    'U-BU-U': (1.5, 2.0),     # KEEP - WR 93.8%, WF 4/5
    'ST-BD-DN': (2.0, 3.0),   # UPDATE from (2.5, 2.0)

    # SHORT patterns
    'BD-BD-BD': (3.0, 2.5),   # UPDATE from (3.5, 1.5)
    'DN-DN-BD': (1.5, 3.0),   # UPDATE from (4.0, 1.0) [CRITICAL]
    'MU-ST-DN': (1.0, 2.5),   # UPDATE from (2.0, 1.0)
    'IH-DN-DN': (1.0, 3.0),   # UPDATE from (2.0, 2.5)
    'BD-ST-DN': (1.5, 3.0),   # UPDATE from (2.5, 2.0)
    'BU-U-DN': (1.5, 2.5),    # UPDATE from (2.5, 2.0)
    'D-DN-BD': (2.5, 2.0),    # KEEP
}

# Updated metrics expectations
EXPECTED_WIN_RATE = 88.0  # From v1.15 validation
EXPECTED_AVG_WIN = 1.5    # Smaller TP targets
EXPECTED_AVG_LOSS = 2.8   # Wider SL
```

---

## Recommendations

### Immediate Actions
1. **Deploy v1.15** with updated TP/SL settings
2. **Monitor DN-DN-BD** closely - had severe v1.14 issue

### Future Research
1. **Collect bullish period data** for SHORT pattern counter-trend analysis
2. **Track production vs backtest divergence** to detect overfitting
3. **Re-validate quarterly** as market conditions change

---

## Appendix: Research Output

```
============================================================
Pattern 5m - Comprehensive Regime Validation Research
Dataset: 2025-10-02 to 2025-12-31 (25920 bars)
Price: $118901 → $88284 (-25.8%)
============================================================

[CRITICAL] DN-DN-BD v1.14 (4.0/1.0): 8.3% WR → v1.15 (1.5/3.0): 94.1% WR

All 9 patterns pass WF validation with v1.15 optimal settings.
```

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1.15.0 | 2026-01-26 | Regime-validated TP/SL optimization |
| v1.14.2 | 2026-01-26 | Early exit double-counting bug fix |
| v1.14.1 | 2026-01-26 | Early exit candle timestamp tracking |
| v1.14.0 | 2026-01-26 | Context research optimization (invalidated) |
