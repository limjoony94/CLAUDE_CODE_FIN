# Feature Correlation Analysis - Production Models
**Date**: 2025-10-23
**Status**: ✅ Analysis Complete

---

## LONG Entry Model


================================================================================
LONG Entry Model - Correlation Analysis Report
================================================================================

Total Features: 44
Missing Features: 0
Correlation Threshold: 0.8
High Correlation Pairs Found: 12

================================================================================
High Correlation Pairs (Redundancy Detected)
================================================================================

1. Correlation: 1.0000
   Feature 1: volume_ma_ratio
   Feature 2: volume_ma_ratio
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

2. Correlation: 0.9969
   Feature 1: bb_mid
   Feature 2: bb_low
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

3. Correlation: 0.9969
   Feature 1: bb_high
   Feature 2: bb_mid
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

4. Correlation: 0.9877
   Feature 1: bb_high
   Feature 2: bb_low
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

5. Correlation: 0.9793
   Feature 1: upper_trendline_slope
   Feature 2: lower_trendline_slope
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

6. Correlation: 0.9508
   Feature 1: macd
   Feature 2: macd_signal
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

7. Correlation: 0.9204
   Feature 1: price_vs_upper_trendline_pct
   Feature 2: price_vs_lower_trendline_pct
   ⚠️  HIGH REDUNDANCY (>0.9) - Likely redundant

8. Correlation: 0.8947
   Feature 1: macd_diff
   Feature 2: price_vs_upper_trendline_pct
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

9. Correlation: 0.8890
   Feature 1: macd_diff
   Feature 2: price_vs_lower_trendline_pct
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

10. Correlation: 0.8871
   Feature 1: macd_signal
   Feature 2: lower_trendline_slope
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

11. Correlation: 0.8830
   Feature 1: macd_signal
   Feature 2: upper_trendline_slope
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

12. Correlation: 0.8106
   Feature 1: shooting_star
   Feature 2: strong_selling_pressure
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

================================================================================
Potential Feature Groups (High Mutual Correlation)
================================================================================

Hub Features (3+ high correlations):
  - macd_signal: 3 connections
    Connected to: macd, lower_trendline_slope, upper_trendline_slope

================================================================================
Recommendations
================================================================================

⚠️  Found 12 redundant pairs
   - Severe (>0.95): 6 pairs
   - High (0.9-0.95): 1 pairs
   - Moderate (0.8-0.9): 5 pairs

Recommended Actions:
1. Review severe redundancy pairs first
2. Consider removing one feature from each pair
3. Prioritize keeping features with:
   - Higher importance scores
   - Better interpretability
   - Fewer connections to other features

Potential feature reduction: 44 → 32
Reduction: -12 features (27.3%)

## SHORT Entry Model


================================================================================
SHORT Entry Model - Correlation Analysis Report
================================================================================

Total Features: 38
Missing Features: 0
Correlation Threshold: 0.8
High Correlation Pairs Found: 14

================================================================================
High Correlation Pairs (Redundancy Detected)
================================================================================

1. Correlation: 1.0000
   Feature 1: macd_strength
   Feature 2: macd_divergence_abs
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

2. Correlation: 0.9976
   Feature 1: atr_pct
   Feature 2: atr
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

3. Correlation: 0.9543
   Feature 1: down_candle
   Feature 2: rejection_from_resistance
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

4. Correlation: 0.9104
   Feature 1: volatility
   Feature 2: atr_pct
   ⚠️  HIGH REDUNDANCY (>0.9) - Likely redundant

5. Correlation: 0.9098
   Feature 1: volatility
   Feature 2: atr
   ⚠️  HIGH REDUNDANCY (>0.9) - Likely redundant

6. Correlation: 0.8816
   Feature 1: downside_volatility
   Feature 2: upside_volatility
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

7. Correlation: 0.8597
   Feature 1: atr
   Feature 2: upside_volatility
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

8. Correlation: 0.8574
   Feature 1: atr_pct
   Feature 2: upside_volatility
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

9. Correlation: 0.8192
   Feature 1: rsi_direction
   Feature 2: price_direction_ma20
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

10. Correlation: 0.8122
   Feature 1: atr_pct
   Feature 2: downside_volatility
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

11. Correlation: 0.8107
   Feature 1: atr
   Feature 2: downside_volatility
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

12. Correlation: 0.8072
   Feature 1: volatility
   Feature 2: upside_volatility
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

13. Correlation: 0.8050
   Feature 1: price_distance_ma20
   Feature 2: price_distance_ma50
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

14. Correlation: 0.8008
   Feature 1: down_candle_ratio
   Feature 2: resistance_rejection_count
   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed

================================================================================
Potential Feature Groups (High Mutual Correlation)
================================================================================

Hub Features (3+ high correlations):
  - atr_pct: 4 connections
    Connected to: volatility, upside_volatility, atr, downside_volatility
  - atr: 4 connections
    Connected to: volatility, upside_volatility, downside_volatility, atr_pct
  - upside_volatility: 4 connections
    Connected to: volatility, atr, downside_volatility, atr_pct
  - volatility: 3 connections
    Connected to: upside_volatility, atr, atr_pct
  - downside_volatility: 3 connections
    Connected to: upside_volatility, atr, atr_pct

================================================================================
Recommendations
================================================================================

⚠️  Found 14 redundant pairs
   - Severe (>0.95): 3 pairs
   - High (0.9-0.95): 2 pairs
   - Moderate (0.8-0.9): 9 pairs

Recommended Actions:
1. Review severe redundancy pairs first
2. Consider removing one feature from each pair
3. Prioritize keeping features with:
   - Higher importance scores
   - Better interpretability
   - Fewer connections to other features

Potential feature reduction: 38 → 24
Reduction: -14 features (36.8%)

## Exit Model


================================================================================
Exit Model - Correlation Analysis Report
================================================================================

Total Features: 9
Missing Features: 16
Correlation Threshold: 0.8
High Correlation Pairs Found: 3

================================================================================
High Correlation Pairs (Redundancy Detected)
================================================================================

1. Correlation: 0.9988
   Feature 1: macd
   Feature 2: trend_strength
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

2. Correlation: 0.9508
   Feature 1: macd
   Feature 2: macd_signal
   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one

3. Correlation: 0.9481
   Feature 1: macd_signal
   Feature 2: trend_strength
   ⚠️  HIGH REDUNDANCY (>0.9) - Likely redundant

================================================================================
Potential Feature Groups (High Mutual Correlation)
================================================================================

No hub features found (all features have <3 connections)

================================================================================
Recommendations
================================================================================

⚠️  Found 3 redundant pairs
   - Severe (>0.95): 2 pairs
   - High (0.9-0.95): 1 pairs
   - Moderate (0.8-0.9): 0 pairs

Recommended Actions:
1. Review severe redundancy pairs first
2. Consider removing one feature from each pair
3. Prioritize keeping features with:
   - Higher importance scores
   - Better interpretability
   - Fewer connections to other features

Potential feature reduction: 9 → 6
Reduction: -3 features (33.3%)

## Combined Summary

Total Features: 107
Total High Correlation Pairs: 29
Potential Reduction: 29 features (27.1%)

## Visualizations

- `correlation_matrix_long_entry_model.png`
- `correlation_matrix_short_entry_model.png`
- `correlation_matrix_exit_model.png`
- `correlation_distribution_*.png` (if redundancy found)
