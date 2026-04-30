# Round 35 — Finer R26 Grid Sweep (5×5×5 = 125 configs) with Strict Switch Criterion

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before sweep code)
**Track**: Finer optimization within R33 frontier area, parallel to LIVE bot

---

## DISCLOSURE

R33 (3×3×3 = 27 configs): all positive train, baseline Pareto frontier confirmed.
Train winner test improvement: only +0.007%/day (negligible).

User epistemology accepted: LIVE 30d ≡ additional BT for future-uncertainty
resolution. So R35 finer sweep parallel to LIVE bot is valid.

LIVE bot continues running at R26 baseline + 4× trading_leverage + per-TP compound.
R35 is BT-only optimization exploration.

---

## Locked Search Space

```python
GRID_R35 = {
    'grid_spacing_pct':        [0.20, 0.30, 0.40, 0.50, 0.60],
    'grid_levels_each_side':   [3, 4, 5, 6, 7],
    'trend_exit_distance_pct': [1.0, 1.25, 1.5, 1.75, 2.0],
}
# Total: 5 × 5 × 5 = 125 configs

LOCKED_FIXED = {
    'capital_usd': 1500,        # BT same as R33 (1× capital, no leverage scaling here)
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
}
```

Note: leverage scaling applied separately if winner switches LIVE config.
R35 measures relative performance at 1× — leverage proportional scaling per R34.

---

## STRICT Switch Criterion — STABILITY-FIRST (per user 재강조 2026-04-30)

User explicit: "소수의 수익 폭발로 인해 (과적합) 오염되지 않고, 어떠한 실측 데이터를 
인풋으로 받아도 통계적으로 안정적인 수익을 낼 수 있는 전략."

**Selection process** (stability gate FIRST, daily SECOND):

1. **Stability gate**: Among 125 train configs, keep only those with
   `train_bs_pos_rate >= 0.85` (vs R26 baseline 0.94)
2. **Among stability-gated**: Rank by train daily_pct, identify winner
3. Test winner: must satisfy ALL:
   - test_bs_pos_rate ≥ 0.85 (stability persists OOS)
   - test_daily ≥ baseline_test + 0.02%
   - test/train retention ≥ 60% (overfit guard)
4. Walk-forward 5-fold on winner:
   - ≥ 4/5 folds positive daily
   - ≥ 4/5 folds bs_pos_rate ≥ 0.80
5. Bonferroni-aware (informational): 125 configs → p_threshold 0.0004

**If ALL criteria met → propose LIVE config switch (user approval required)**
**If ANY criterion fails → R26 baseline confirmed → no LIVE change**

**Rationale**: high daily without stability = profit explosion artifact
(user's overfitting concern). Stability-first selection prevents this.

---

## Procedure

1. Train (60%, first 432d): run all 125 configs
2. Sort by train daily_pct, identify winner
3. Test (40%, last 288d): run only winner config
4. WF 5-fold (1h data): run only winner config
5. Compare to baseline (0.30/5/1.5) on same train/test/WF
6. Apply switch criterion
7. Report all 125 train results + winner test/WF + switch decision

---

## Anti-Adjustment

GRID values, train/test split, switch criteria ALL LOCKED. No post-hoc:
- Re-pick if criterion fails (no relaxation)
- Switch axis (e.g., re-run with different ATR period)
- Threshold modification

If FAIL → R26 baseline confirmed Pareto-optimal; LIVE keeps current config.

---

## EV Estimate

| Outcome | Probability |
|---------|-------------|
| Winner test ≥ baseline + 0.02% AND WF ≥ 4/5 (genuine improvement) | 5-10% |
| Winner test ≈ baseline (overfit / Pareto shallow) | 70-80% |
| Winner test < baseline (overfit) | 10-15% |
| Catastrophic (test sign flip) | 5-10% |

Realistic prior: R33 already confirmed Pareto shallow. R35 most likely shows
similar pattern. Switch unlikely.

---

## Hash Anchor

Committed BEFORE sweep code.
