# Code Review: pattern_5m Module

**Date**: 2026-02-01
**Version**: v1.22.0 (12 patterns: 7L+5S, WR 80.3%, PF 3.36)
**Scope**: `bingx_rl_trading_bot/scripts/production/pattern_5m/` (~6,371 lines)

---

## 1. Architecture Overview

The module is well-structured with clear separation of concerns:
- `constants.py` (427L) — Config, patterns, thresholds
- `signals.py` (771L) — Pattern detection, context filtering, confidence
- `indicators.py` (233L) — Candle classification (12 types)
- `bot.py` (512L) — Main trading loop
- `exchange.py` (461L) — BingX API interface
- `orders.py` (506L) — TP/SL order management
- `position_open.py` (575L) — Position opening logic
- `position_close.py` (523L) — Position closing logic
- `position_monitor.py` (328L) — Position monitoring
- `state.py` (264L) — State persistence

## 2. Code Quality Issues

### 2.1 🔴 Critical: Classification Logic Duplication

**indicators.py** and **signals.py** both implement candle classification:
- `indicators.py::classify_candle()` — standalone function
- `signals.py::add_candle_classification()` — duplicates the loop + pattern building

Both have slightly different handling of early bars (NaN avg_body):
- `indicators.py::calculate_indicators()` uses `avg_b = 1.0` default
- `signals.py::add_candle_classification()` falls back to `MED_UP/MED_DOWN`

This means **production** and **backtest** can disagree on classifications for bars 0-19. The `full_270d_revalidation.py` uses a third implementation (vectorized numpy) which also handles early bars differently (`fillna(1.0)`).

**Fix**: Single source of truth for classification. Extract to a shared `classify.py`.

### 2.2 🟡 Medium: Hardcoded Magic Numbers in Revalidation

`full_270d_revalidation.py` has its own hardcoded pattern lists (`CURRENT_LONG`, `CURRENT_SHORT`) that can drift from `constants.py`. There's no shared import.

**Fix**: Import from `constants.py` or a shared config module.

### 2.3 🟡 Medium: No Unit Tests

Zero test files for the pattern_5m module. Classification logic is the core edge — any regression breaks the strategy.

**Fix**: At minimum, add tests for:
- `classify_candle()` with known OHLCV → expected type
- 3-candle pattern string building
- Context filter pass/reject logic

### 2.4 🟡 Medium: `indicators_v2.py` Dead Code

730 lines of `indicators_v2.py` exists but is never imported. Likely a leftover from experimentation.

**Fix**: Remove or archive.

### 2.5 🟢 Minor: Docstring Says "Engulf 5m Bot"

`config.py` and `exchange.py` still have "Engulf 5m Bot" in their module docstrings (renamed from previous version).

### 2.6 🟢 Minor: CSV Confidence Logging in Production

`_save_confidence_to_csv()` does file I/O on every signal. Should be async or batched to avoid blocking the trading loop.

## 3. Performance Improvements

### 3.1 Vectorized Classification (High Impact)

Current production uses row-by-row `for i in range(len(df))` loops. The backtest (`full_270d_revalidation.py`) already uses vectorized numpy. Porting this to production would speed up classification ~10-50x.

```python
# Current (slow):
for i in range(len(df)):
    candle_types.append(classify_candle(df.iloc[i], ...))

# Better (vectorized):
df['body_ratio'] = df['body_abs'] / df['range'].replace(0, 1)
doji_mask = df['body_ratio'] < 0.10
# ... etc
```

### 3.2 Pattern Index Pre-building

For backtests, building `pattern_indices` dict once (O(n)) then looking up (O(1)) is correct. But `check_entry_signal()` in production rechecks `pattern in long_patterns` each call — this is fine for 12 patterns but would matter at 50+. Use a set.

### 3.3 Context Calculation Caching

`calculate_context()` recomputes RSI, ATR from scratch every call. These should be computed once per candle in `calculate_indicators()` and stored as columns.

## 4. New Pattern Exploration Directions

### 4.1 Current State: 12 out of 1,728 possible 3-candle combos

Only 0.7% of the pattern space is used. The revalidation pipeline already explores all 1,728 but filters aggressively (MC < 0.05, WF ≥ 4/5, excess WR > 15%).

### 4.2 Expansion Strategies

1. **Relax Tier Criteria Slightly**: Current Tier 1 requires MC < 0.01. Tier 1.5 allows MC < 0.03. A Tier 2 at MC < 0.05 with WF ≥ 4/5 could yield 5-10 more patterns.

2. **4-Candle Patterns**: 12^4 = 20,736 combinations. Requires more data but could capture more complex setups (e.g., consolidation → breakout → continuation → entry).

3. **Type Grouping**: Group similar types (e.g., {D, DF, GS} → "doji_family", {MU, BU} → "strong_up") to create higher-count aggregate patterns. Trade precision for sample size.

4. **Time-of-Day Filter**: Asian/European/US session as 4th dimension. Some patterns may only work in specific sessions.

5. **Volume Confirmation**: Add volume condition (above/below 20-MA) as filter. Not a new pattern but reduces false signals.

### 4.3 Regime-Conditional Patterns

`REGIME_DETECTION_ENABLED = False` since v1.19.0 (tight TP/SL is regime-independent). But revisiting with per-pattern regime filters could help:
- Pattern X only in BULL regime
- Pattern Y only in SIDEWAYS

This is different from the old approach (regime selects patterns). Instead: pattern is always valid, but regime is an optional filter.

## 5. Context Filter Improvement

### 5.1 Current State

`PATTERN_CONTEXT_FILTERS = {}` — empty since v1.19.0. All old filters were removed when patterns changed.

### 5.2 Research Needed

For each of the 12 current patterns, analyze:
- Win rate by RSI zone (OS/N/OB)
- Win rate by volatility tercile (L/M/H)
- Win rate by trend (UP/DN)
- Win rate by hour-of-day

If any dimension shows >10% WR difference, add as context filter. Use the 270-day dataset for this analysis.

### 5.3 ADX as Context

ADX (Average Directional Index) is a proven trend strength indicator. Adding ADX zones:
- ADX < 20: no trend (range-bound)
- ADX 20-40: trending
- ADX > 40: strong trend

Some patterns may only work in trending or ranging markets.

## 6. Risk/Architecture Recommendations

| Priority | Item | Effort |
|----------|------|--------|
| 🔴 High | Unify classification logic (indicators vs signals vs backtest) | 2h |
| 🔴 High | Add classification unit tests | 3h |
| 🟡 Med | Vectorize production classification | 2h |
| 🟡 Med | Cache RSI/ATR in calculate_indicators | 1h |
| 🟡 Med | Remove indicators_v2.py dead code | 10min |
| 🟢 Low | Fix "Engulf" docstrings | 5min |
| 🟢 Low | Async confidence CSV logging | 1h |

## 7. Summary

The codebase is **production-quality** for its current scope. The 12-pattern system with tight TP/SL is well-validated (MC tests, WF, Holm correction). The main risks are:

1. **Classification drift** between production and backtest implementations
2. **No automated regression tests** for the core classification logic
3. **Untapped pattern space** (99.3% unexplored)

The biggest alpha opportunity is systematic exploration of the remaining pattern combinations with proper validation (MC + WF + period profitability), which is what `discover_new_patterns.py` is designed for.
