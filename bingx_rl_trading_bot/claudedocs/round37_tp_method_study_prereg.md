# Round 37 — TP Method Study (4 methods, mechanism modification)

**Date pre-registered**: 2026-05-01
**Status**: PRE-COMMIT
**Track**: Tier 1 mechanism modification (advisor-approved single round)

---

## DISCLOSURE

182 configs (R33+R35+R36) confirmed R26 baseline Pareto-optimal under
stability-first criterion across spacing × levels × trend_exit.

R37 tests **mechanism modification**: TP formula. Distinct from prior pure-parameter
sweeps. Single round, 4 methods, NO follow-up rounds pre-registered (no R38 alongside).

---

## TP Methods (LOCKED)

All methods preserve grid structure: TP cap at 0.50% to prevent overlap with
adjacent grid levels (spacing 0.30%, neighbor at +0.30%, cap at 0.50% = 0.20%
buffer past neighbor).

```python
def tp_distance_pct(method, atr_pct_of_price, level_price):
    if method == 'M1':  # baseline: fixed % equal to spacing
        return 0.30
    elif method == 'M2':  # ATR-modulated, mild
        return min(0.50, 0.5 * atr_pct_of_price)
    elif method == 'M3':  # ATR-driven, full
        return min(0.50, 1.0 * atr_pct_of_price)
    elif method == 'M4':  # hybrid floor
        return max(0.30, min(0.50, 0.5 * atr_pct_of_price))
```

| Method | TP formula (% from level_price) | Rationale |
|--------|-------------------------------|-----------|
| **M1** | 0.30% (= spacing, BASELINE) | Reference R26 |
| **M2** | min(0.50%, 0.5 × ATR%) | Mild ATR scaling, capped |
| **M3** | min(0.50%, 1.0 × ATR%) | Full ATR scaling, capped |
| **M4** | max(0.30%, min(0.50%, 0.5 × ATR%)) | Floor at baseline, cap at 0.50% |

ATR period LOCKED at 14 bars (different from ranging filter's ATR period 20).
ATR_pct_of_price = ATR(14)/close × 100, computed at fill time.

---

## Locked Other Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'capital_usd': 1500,
    'grid_spacing_pct': 0.30,           # R26 baseline
    'grid_levels_each_side': 5,          # R26 baseline
    'trend_exit_distance_pct': 1.5,      # R26 baseline
    'atr_period_for_TP': 14,             # NEW (separate from ranging filter ATR=20)
    'atr_period_for_ranging': 20,
    'atr_pct_median_lookback_bars': 720,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
    'tp_cap_pct': 0.50,                  # ALL methods cap at 0.50%
    'train_test_split': 0.60,
}
```

---

## STRICT Switch Criterion (Stability-First, advisor-aligned)

For winner method to replace M1 (baseline) on LIVE bot:
1. Stability gate: train BS_pos ≥ 0.85
2. Among gated: rank by daily_pct
3. Test winner: BS_pos ≥ 0.85 AND daily ≥ M1_test + 0.02% AND retention ≥ 60%
4. WF 5-fold: ≥ 4/5 folds positive AND ≥ 4/5 BS_pos ≥ 0.80

If FAIL → R26 (M1) confirmed; **NO follow-up rounds**. Mechanism shallow.

---

## Honest EV Estimate (down-weighted per advisor)

| Outcome | Probability |
|---------|-------------|
| Genuine improvement (winner switch) | **10-15%** (down from 25-35%) |
| Pareto-shallow / baseline equivalent | 70-80% |
| Catastrophic overfit | 5-10% |

R33/R35/R36 pattern (all confirm baseline) suggests this likely also confirms
baseline. Mechanism change explored, baseline robust if confirmed.

---

## Anti-Adjustment

LOCKED: 4 methods, ATR period 14 (TP), formulas, cap 0.50%, criteria.
**No post-hoc method addition**. **No R38+ pre-reg** here.

If FAIL → close synthesis: TP formula not binding, ranging filter / spacing are.

---

## Hash Anchor

Committed BEFORE code.
