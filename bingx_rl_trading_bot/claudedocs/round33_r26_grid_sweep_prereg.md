# Round 33 — R26 Parameter Sweep with Train/Test Split

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before code)
**Track**: R26 Pareto frontier check + 0.20%/day reachability test

---

## DISCLOSURE

R26 (grid trading on ATR-ranging regime, 1× $1500) = 63 rounds 중 유일 positive
1× alpha (0.05%/day, 18.6%/yr, 82.79% bootstrap). User-confirmed market-making
behavior (88.2% anti-FOMO/anti-panic), not retail emotional pattern.

R33: 3 axes × 3 levels = 27 configs systematically vary R26's locked params:
- grid_spacing_pct: 0.20, 0.30 (baseline), 0.50
- grid_levels_each_side: 3, 5 (baseline), 7
- trend_exit_distance_pct: 1.0, 1.5 (baseline), 2.5

Question: Is R26 baseline on Pareto frontier? Can tuning reach user 0.20%/day target?

---

## Procedure (per advisor train/test discipline)

1. Split 720d → TRAIN (60% = first 432d) + TEST (40% = last 288d)
2. Run all 27 configs on TRAIN
3. Rank by train daily_pct
4. Select WINNER = highest train daily
5. Run ONLY WINNER on TEST
6. Report:
   - All 27 train results (full table)
   - Winner test result (single OOS validation)
   - Train vs Test ratio (overfit check)

---

## Locked Grid

```python
GRID = {
    'grid_spacing_pct': [0.20, 0.30, 0.50],
    'grid_levels_each_side': [3, 5, 7],
    'trend_exit_distance_pct': [1.0, 1.5, 2.5],
}
# Total: 27 configs

LOCKED_FIXED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'capital_usd': 1500,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,    # 30d
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,           # 7d
    'train_test_split': 0.60,
}
```

Per-config: per_level_usd = 1500 / (2 × levels_each_side).

---

## Pre-registered Outcomes

| Outcome | Probability (per advisor pattern) |
|---------|----------------------------------|
| Winner test daily ≥ 0.20% AND ≥ 0.5× train | 5-10% (genuine breakthrough) |
| Winner test daily 0.10-0.20% (sub-target) | 15-20% |
| Winner test ≈ baseline 0.05% (Pareto boundary confirmed) | 35-45% |
| Winner test < 0.03% (overfit) | 25-35% |

**Realistic prior**: R26 is likely on Pareto frontier; tuning produces marginal
or negative test changes. R30 (R29 grid) showed catastrophic overfit pattern.
R33 may show similar or robust depending on R26's intrinsic stability.

---

## Anti-Adjustment

GRID values LOCKED. Train/test split 60/40 LOCKED. Winner by train daily_pct
LOCKED (no post-hoc reselection). If WINNER fails on test:
- Result reported as Pareto frontier confirmed at R26 baseline
- No further axis variants without separate pre-reg

---

## Hash Anchor

Committed BEFORE code.
