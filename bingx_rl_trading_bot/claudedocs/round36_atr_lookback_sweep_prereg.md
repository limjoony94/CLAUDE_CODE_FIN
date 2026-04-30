# Round 36 — ATR Period × Lookback Sweep (5×3 = 15 configs)

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT
**Track**: Untested R26 axes (ATR/lookback)

---

## DISCLOSURE

R33 + R35 = 152 configs on (spacing × levels × trend_exit). All confirmed Pareto
frontier at R26 baseline.

R36 explores 2 untested axes: **ATR period** (filter sensitivity) and
**ranging lookback** (regime detection horizon).

LIVE bot continues running at R26 baseline.

---

## Locked Search Space

```python
GRID_R36 = {
    'atr_period': [10, 14, 20, 30, 40],          # 5 values (R26 baseline = 20)
    'atr_pct_median_lookback_bars': [360, 720, 1440],  # 3 (baseline = 720, 30d)
}
# Total: 5 × 3 = 15 configs
```

Other params LOCKED at R26 baseline:
- grid_spacing_pct: 0.30
- grid_levels_each_side: 5
- trend_exit_distance_pct: 1.5
- max_grid_lifetime_bars: 168

---

## Stability-First Selection (per user 재강조)

1. Stability gate: BS_pos ≥ 0.85 on train
2. Among gated: rank by daily_pct
3. Test winner: must satisfy BS_pos ≥ 0.85 AND daily ≥ baseline + 0.02% AND retention ≥ 60%
4. WF 5-fold: ≥ 4/5 folds positive AND ≥ 4/5 BS_pos ≥ 0.80

Bonferroni-aware: 15 configs → p_threshold 0.0033 (informational).

---

## EV Estimate

- Genuine improvement: 5-10%
- Pareto shallow (no improvement): 70-80% (per R33/R35 pattern)
- Catastrophic overfit: 5-10%

---

## Anti-Adjustment

GRID values, CRITERIA LOCKED. No post-hoc relaxation. If FAIL → R26 baseline
confirmed across all R33/R35/R36 dimensions.

---

## Hash Anchor

Committed BEFORE code.
