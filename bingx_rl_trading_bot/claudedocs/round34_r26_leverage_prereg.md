# Round 34 — R26 + Leverage Frontier

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before code)
**Track**: User leverage decision support

---

## DISCLOSURE

User specification (2026-04-30):
- Exchange leverage setting: 10× (max allowed)
- Actual trade leverage: self-computed (to be determined)

R5+leverage frontier (Round 152, 2026-04-30) showed cash-and-carry mechanism:
- 1×/2×: yield-insufficient (0.01%/day)
- 3×+: ALL ruin-bound (drift drawdown × L exceeds capital)

R26 has DIFFERENT leverage profile:
- Drift drawdown 1× = -6.06% / 720d (much smaller than R5's basis swing)
- Trend exit threshold 1.5% protects from large adverse moves
- Liquidation threshold per leverage: (1 - 0.5%) / L

R34 = R26+leverage with empirical scaling.

---

## Locked Parameters

```python
R26_BASELINE = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'capital_usd': 1500,
    'grid_spacing_pct': 0.30,
    'grid_levels_each_side': 5,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'trend_exit_distance_pct': 1.5,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
}

LEVERAGE_LEVELS = [1, 2, 3, 4, 5, 7, 10]
LIQUIDATION_MAINTENANCE_MARGIN_PCT = 0.50  # BingX standard
```

Per-leverage L:
- per_level_notional = $150 × L (kept proportional to L)
- per_level_margin = $150 (fully uses $1500 capital when all 10 fill)
- Position P&L scales with notional → with L
- Friction scales with notional → with L
- Drift drawdown scales with L

Liquidation rule (per position):
- Adverse move × L > (1 − maintenance_margin) = 99.5%
- → liquidation at adverse_pct > 99.5/L

Per leverage:
- 1×: never liquidates
- 2×: 49.75% adverse needed
- 5×: 19.9% adverse
- 10×: 9.95% adverse (intraday bar move possible)

---

## Methodology

1. Re-run R26 simulation with logging
2. For each leverage L:
   - Scale all P&L (harvest, drift, friction) by L
   - Check liquidation on each open position at each bar:
     - Track unrealized adverse % per position
     - If adverse_pct × L > 99.5% → liquidate that position
   - On liquidation: position closes at adverse_pct (full position lost less margin remainder)
3. Compute daily, BS_pos, max_dd per leverage
4. Identify optimal L: max daily where ruin events ≤ 1/year-equivalent

---

## Pre-Registered Outcomes

| Leverage | Expected daily | Liquidation risk | Verdict |
|---------|----------------|------------------|---------|
| 1× | +0.05% | None | Confirmed positive baseline |
| 2× | +0.10% | None (49.75% threshold) | Likely safe + meaningful |
| 3× | +0.15% | Very low (33% threshold) | Probably safe |
| 4× | **+0.20%** | Low (24.9% threshold) | **Target zone** |
| 5× | +0.25% | Low (19.9% threshold) | Above target |
| 7× | +0.35% | Moderate (14.2%) | Possible bar gaps to liquidation |
| 10× | +0.50% | High (9.95%) | BTC ~5-10% gaps possible |

**Ideal outcome**: L=4-5× achieves user 0.20%/day target with ruin_prob/yr < 1%.
**Risk outcome**: L=10× would 10× daily but liquidation events render mean negative.

---

## Decision Criteria (LOCKED)

For each leverage L, classify:
- **DEPLOYABLE**: daily ≥ 0.20% AND liquidation_events ≤ 1/year (in 720d sim, ≤ 2 total)
- **SUB_DEPLOYABLE**: daily ∈ [0.10%, 0.20%) AND liq ≤ 1/yr
- **YIELD_INSUFFICIENT**: daily < 0.10%
- **RUIN_BOUND**: liquidation events > 1/yr (3+ in 720d)

Optimal L = max DEPLOYABLE L. If no DEPLOYABLE: report best SUB_DEPLOYABLE.

---

## Anti-Adjustment

LEVERAGE_LEVELS, maintenance_margin_pct, ruin_threshold (1/yr) ALL LOCKED.
NO post-hoc tuning. If no L meets DEPLOYABLE: that's empirical answer for R26+leverage.

---

## Hash Anchor

Committed BEFORE simulation code.
