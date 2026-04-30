# Round 36 — Grid Spacing × Levels Extended Sweep (6×5 = 30 configs)

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before code)
**Track**: User-prioritized grid count/spacing optimization

---

## DISCLOSURE

R33 (27 configs spacing 0.20-0.50, levels 3-7) + R35 (125 configs same range, finer)
both confirmed R26 baseline (0.30/5) as Pareto frontier. R36 EXTENDS the search space:

- **Tighter spacing** (0.10, 0.15) — finer grid, more cycles
- **Wider spacing** (0.80) — fewer cycles, larger per-cycle profit
- **More levels** (10, 15) — wider grid coverage

LIVE bot continues running at R26 baseline.

---

## Locked Search Space

```python
GRID_R36 = {
    'grid_spacing_pct':       [0.10, 0.15, 0.20, 0.30, 0.50, 0.80],
    'grid_levels_each_side':  [3, 5, 7, 10, 15],
}
# Total: 6 × 5 = 30 configs

LOCKED_FIXED = {
    'capital_usd': 1500,
    'trend_exit_distance_pct': 1.5,    # LOCKED at R26 baseline
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
    'train_test_split': 0.60,
}
```

**Note**: configs where `spacing × levels > trend_exit (1.5%)` mean outermost
levels are beyond trend_exit threshold — those levels rarely fill in practice.
This is naturally measured (not excluded) so empirical effect is captured.

---

## STRICT Switch Criterion (Stability-First, per user 재강조)

Same as R35:
1. Stability gate: train BS_pos ≥ 0.85
2. Among gated: rank by daily_pct
3. Test winner: BS_pos ≥ 0.85 AND daily ≥ baseline + 0.02% AND retention ≥ 60%
4. WF 5-fold: ≥ 4/5 folds positive AND ≥ 4/5 BS_pos ≥ 0.80

If FAIL → R26 baseline (0.30/5) confirmed Pareto-optimal across extended range.

---

## EV Estimate

| Outcome | Probability |
|---------|-------------|
| Genuine improvement (winner switch) | 5-10% (extended range may help) |
| Pareto shallow (no improvement) | 65-75% (R33/R35 pattern) |
| Catastrophic overfit | 5-10% |
| Mixed | 10-20% |

---

## Hash Anchor

Committed BEFORE code.
