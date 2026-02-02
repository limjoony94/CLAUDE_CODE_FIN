# Classification Logic Unification

**Date**: 2026-02-01
**Status**: ✅ Completed (signals.py fixed, analysis scripts TODO'd)

## Problem

Candle classification logic was duplicated in 3 locations with subtle differences:

| Location | Type | Early bars (0-19) handling |
|---|---|---|
| `indicators.py::classify_candle()` | Row-by-row function | `avg_body_20=1.0` → full classify_candle() |
| `signals.py::add_candle_classification()` | Row-by-row wrapper | **BUG**: forced `MED_UP/MED_DOWN` |
| `analysis/full_270d_revalidation.py` | Vectorized numpy | `fillna(1.0)` → equivalent to indicators.py |

### Key Difference (now fixed)

`signals.py` hardcoded early bars to `MED_UP/MED_DOWN`, bypassing `classify_candle()`.
This meant production and backtest could disagree on bars 0-19 — a DOJI at bar 5 would
be classified as `MED_UP` by signals.py but `D` by indicators.py.

### Vectorized vs Row-by-Row

The revalidation script uses vectorized numpy which is ~10-50x faster but has a subtle
ordering difference: vectorized masks are applied in sequence, so later masks can override
earlier ones. The row-by-row `classify_candle()` uses if/elif chains with explicit priority.
Both produce the same results because the mask ordering matches the if/elif priority.

## Changes Made

1. **`signals.py`**: Fixed `add_candle_classification()` to use `classify_candle(row, avg_b)`
   with `avg_b=1.0` fallback for NaN, matching `indicators.py::calculate_indicators()`.

2. **`analysis/full_270d_revalidation.py`**: Added TODO comment to migrate to canonical
   `classify_candle()` import. Not changed because analysis scripts are standalone and
   vectorized classification is performance-critical for backtesting.

3. **`indicators.py`**: Confirmed as canonical implementation. No changes needed.

## Canonical Source of Truth

**`indicators.py::classify_candle(row, avg_body_20)`** is the single source of truth.

All other code should either:
- Import and call `classify_candle()` directly
- Or document why they diverge (e.g., vectorized performance in backtesting)

## Migration Guide for Analysis Scripts

To migrate a script that has inline classification:

```python
# Before (inline):
df['body_ratio'] = df['body_abs'] / df['range'].replace(0, 1)
# ... 30 lines of numpy masks ...

# After (canonical):
import sys; sys.path.insert(0, '/path/to/repo')
from bingx_rl_trading_bot.scripts.production.pattern_5m.indicators import classify_candle
df['body'] = df['close'] - df['open']
df['body_abs'] = df['body'].abs()
df['avg_body_20'] = df['body_abs'].rolling(20).mean().fillna(1.0)
df['candle_type'] = [
    classify_candle(df.iloc[i], df.iloc[i]['avg_body_20']).value
    for i in range(len(df))
]
```

Note: This is slower than vectorized. For backtesting 270d+ data, keep vectorized
but add a unit test that verifies vectorized == row-by-row on a sample.
