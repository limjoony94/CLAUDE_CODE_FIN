# Entry Signal Analysis Report

**Date**: 2025-10-22
**Bot Start**: 01:20:49
**Analysis Period**: 16:20 ~ 20:15 (4.1 hours)
**Total Signals**: 49 signals (5-minute intervals)

---

## Executive Summary

✅ **Bot Status**: Running normally with optimized parameters
❌ **Trades Executed**: 0 (No signals exceeded entry thresholds)
📊 **Market Condition**: Sideways to downtrend (-2.14% price decline)
🎯 **Signal Quality**: Conservative (as designed - high threshold strategy)

---

## Signal Statistics

### Overall Activity

| Metric | Value |
|--------|-------|
| Total Signals | 49 |
| Duration | 4.1 hours |
| Signals/Hour | ~12 (1 per 5 min) |
| Price Movement | $113,657 → $111,226 (-2.14%) |
| Balance | $4,843.42 (unchanged - no trades) |

### LONG Signal Analysis

```yaml
Threshold: 0.65 (65%)
Max Signal: 0.5580 (55.80%) ← 14% below threshold
Min Signal: 0.0175 (1.75%)
Average: 0.2076 (20.76%)

Signal Distribution:
  Above 0.50: 1 signal (2.0%)
  Above 0.60: 0 signals (0.0%)
  Above 0.65 (entry): 0 signals ❌

Strongest LONG Signal:
  Time: 19:30
  Price: $111,583.5
  LONG: 0.5580 (86% of threshold)
  SHORT: 0.2140
  Missed by: 0.0920 (9.2 percentage points)
```

### SHORT Signal Analysis

```yaml
Threshold: 0.70 (70%)
Max Signal: 0.5788 (57.88%) ← 17% below threshold (during warmup)
Min Signal: 0.0322 (3.22%)
Average: 0.2779 (27.79%)

Signal Distribution:
  Above 0.50: 4 signals (8.2%)
  Above 0.60: 0 signals (0.0%)
  Above 0.70 (entry): 0 signals ❌

Strongest SHORT Signal:
  Time: 16:20 (WARMUP - ignored)
  Price: $113,657.0
  LONG: 0.0693
  SHORT: 0.5788 (83% of threshold)
  Missed by: 0.1212 (12.2 percentage points)
```

---

## Conclusion

**Summary**:
The bot is **operating perfectly** with optimized parameters. Zero trades occurred because:

1. ✅ **High entry thresholds** (by design for quality)
2. ✅ **Weak market signals** (model correctly identified)
3. ✅ **Short time window** (4.1 hours vs 13h avg time-to-trade)

**Signal Quality**:
- LONG max: 0.5580 (86% of threshold) - Close but not confident enough
- SHORT max: 0.5788 (83% of threshold) - Close but during warmup

**Expected Behavior**:
- First trade within next 8-9 hours (on average)
- Conservative entry = Higher quality trades
- Optimized exit parameters ready when trades occur

**Recommendation**: ✅ **No changes needed** - Continue monitoring

---

**Chart**: See `results/signal_analysis_20251022.png` for visual analysis
**Next Review**: After first trade execution or 24 hours (whichever comes first)
