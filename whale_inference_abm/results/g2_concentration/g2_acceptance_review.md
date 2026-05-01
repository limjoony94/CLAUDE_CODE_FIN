# G2 Phase 2 Acceptance Review — Wealth-Concentration Validity

**Date**: 2026-05-01
**Phase**: G2 (Wealth-Concentration Validity, design v0.7 Section 2 Gate G2)
**Status**: PENDING (10k bar integration test result)

---

## G2 Pass Criteria (design v0.7 Section 2)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Gini > max(0.5, empirical_BTC_target − 0.1) at T=10000 sim-bars | TBD (filled after run) |
| 2 | Top-5% rank stability ≥ 50% between T=5000 and T=10000 | TBD |
| 3 | Top-5% emerges from initially uniform distribution | Verified by construction (initial wealth = 1000 uniform) |

## G2 Abandon Trigger

| Trigger | Status |
|---------|--------|
| Gini < 0.3 at T=10k → market mechanics fail | TBD |
| Top-5% turnover > 80% → no stable whales | TBD |

---

## Implementation

### `abm/metrics.py` (~140 LOC)

- `gini(wealths)`: standard formula, negative values clipped to 0 (bankrupt MTM handling)
- `top_k_share(wealths, k_pct)`: cumulative wealth share of top-K agents (ranked desc)
- `top_k_overlap(snap_old, snap_new, k_pct)`: rank-stability via set intersection / max-size
- `evaluate_concentration(history, bar_indices_to_compare, ...)`: checkpoint-based eval

### `tests/test_metrics.py` (16 tests)

- 6 gini tests (perfect equality, max inequality, empty, all-zero, negative-clipped, known-value)
- 4 top_k_share tests (uniform, concentrated, empty, min-1-agent)
- 4 top_k_overlap tests (identical, disjoint, partial, empty)
- 1 evaluate_concentration synthetic test
- 1 G2 10k-bar integration (@pytest.mark.slow)

15/15 unit tests PASS in 0.21s. Integration result: TBD.

---

## Empirical BTC Perp Gini Calibration (TODO from architecture v1.1)

**Target**: estimate Gini of actual BTC perp wealth distribution to set realistic threshold.

**Plan options**:
1. On-chain wallet clustering (Glassnode, Chainalysis equivalent — costly/restricted)
2. Whale-tracker datasets (Whalemap, WhaleAlert public statistics — limited granularity)
3. CEX market-maker leaderboards (BingX/Binance public rankings — partial)
4. Academic literature on crypto wealth distributions

**v1 decision (interim)**: use design-default threshold 0.5 per Section 2 Gate G2 pass criterion.
Empirical recalibration deferred to G3/G4 if substrate findings warrant. Document as v1
limitation: G2 passes against synthetic threshold, not empirical BTC perp distribution.

If 10k smoke gini >> 0.5, the synthetic threshold is conservative and absence of empirical
target does not invalidate G2. If gini is borderline (0.45-0.55), empirical calibration
is needed before declaring pass.

---

## Result (TBD)

(Filled after `test_g2_wealth_concentration_at_10k_bars` completes)

---

## Outstanding (G3 prerequisite)

After G2 passes:
1. Frozen-admission window machinery (already in admission.py from G0)
2. explicit_strategies catalog format (per-agent decision function specs serialized)
3. Substrate prereg directory + git-hash workflow
4. Anti-circularity audit module (3-layer: AST + OLS R² + KSG MI)

These are G3 build items, not G2 dependencies.
