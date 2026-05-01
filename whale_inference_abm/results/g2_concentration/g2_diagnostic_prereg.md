# G2 Wealth-Concentration Diagnostic — Pre-Registration

**Date**: 2026-05-01
**Trigger**: G2 abandon trigger fired (Gini 0.191 < 0.3) at first 10k-bar evaluation.
**Status**: Diagnostic experiments designed BEFORE running, per advisor anti-rationalization protocol.

---

## Result that triggered investigation

```
G2 wealth concentration at 10k bars (seed=42):
  Agents alive at T=10k: 978 (15 incumbents + 963 admissions)
  Gini at T=10k: 0.191 (threshold > 0.5 for pass; abandon trigger < 0.3 fired)
  Top-5% share at T=10k: 0.174 (uniform baseline = 0.05)
  Top-5% rank stability T=5k vs T=10k: 0.438 (threshold ≥ 0.5)
```

**Arithmetic context**: 15 incumbents (1000 wealth each = 15,000 total) vs 963 admissions
(100 wealth each = 96,300 total). Incumbents = 14% of pool BEFORE any PnL effects.

---

## Hypothesis (pre-registered)

**Primary**: Open-phase admission dilution overwhelms wealth-weighted sizing's concentration
mechanism. The architecture v1.1 stated `joining wealth = 100 (smaller than initial 1000 to
ensure incumbents have first-mover wealth advantage)` — but at admission rate 1/600s × 10k
bars = 1000 admissions, the entrant volume drowns out the wealth-fraction differential.

**Implication**: G2 may pass under design assumptions only when:
- (a) admission rate is much lower, OR
- (b) Gini is measured in the FROZEN-admission window (G3-relevant), not at T=10k overall

---

## Experiments (pre-registered, designed before results visible)

### E1: Admission rate sweep
**Design**: Run 10k-bar smoke at admission rate_lambda in [1/600 (current), 1/3600, 1/36000, 0].
**Measurement**: Gini at T=10k for each.
**Outcome interpretations**:
- Gini monotonically increases as rate decreases AND hits >0.5 at rate=0:
  → "open-system precludes concentration at meaningful rate"
  → architecture has structural issue with simultaneous open-system + wealth-weighted concentration
- Gini stays <0.3 even at rate=0:
  → wealth-weighted sizing isn't actually concentrating (deeper issue)
  → invalidates architecture v1.1 Section 6.2 mechanism claim
- Gini hits 0.5 at rate=1/3600 (165 admissions over 10k bars):
  → dilution is the issue but bounded; G2 can use lower rate

### E4: Frozen-window-aware Gini (advisor amendment)
**Design**: Run sim with T_open=5000 bars (admissions on), T_extract=5000 bars (frozen).
Compute Gini at end of frozen phase (T=10k), filtering to agents PRESENT AT T_open boundary.
This matches the G3 substrate-extraction window where admissions are by-design disabled.

**Measurement**: Gini[end_of_frozen_among_T_open_population] vs Gini[end_of_frozen_all_agents].

**Outcome interpretations**:
- Gini > 0.5 in frozen window even though overall Gini at T=10k is low:
  → G2 criterion should be redefined as "frozen-window Gini" not "all-T_10k Gini"
  → architecture remains valid, criterion mis-specified
  → unblocks G3 immediately
- Gini still < 0.5 in frozen window:
  → wealth-weighted sizing genuinely insufficient even without dilution
  → architecture-level revision needed

### E2 (optional, if E1+E4 don't resolve): Pareto initial wealth
**Design**: Disable admissions, draw 15 initial wealths from Pareto(α=1.16, x_min=100).
Run 5k bars. Compute Gini trajectory.
**Outcome**: Tests whether wealth-weighted sizing AMPLIFIES initial heterogeneity.

### E3 (optional, if all above fail): Disable wealth-weighted sizing
**Design**: Run with all agents trading fixed-size 0.001 BTC regardless of wealth. 5k bars.
**Outcome**: Tests whether concentration is driven by sizing OR by skill differential.

---

## Decision tree (pre-registered)

| E1 result | E4 result | Decision |
|-----------|-----------|----------|
| Gini ↑ as rate ↓, hits >0.5 at rate=0 | Frozen Gini > 0.5 | **G2 criterion change**: redefine as frozen-window Gini. Update design v0.8. Unblocks G3. |
| Gini ↑ as rate ↓, hits >0.5 at rate=0 | Frozen Gini < 0.5 | **Architecture revision**: open-system + concentration combo doesn't work. Lower admission rate baseline AND restate G2 criterion. Update design v0.8 + new smoke build. |
| Gini stays low even at rate=0 | (any) | **Architecture-level falsification**: wealth-weighted sizing doesn't concentrate. Run E2/E3. Possibly v1 ABM hypothesis genuinely wrong. Surface to user as "ABM v1 hypothesis falsified — research methodology contribution only, no Path B". |

---

## Anti-tuning commitment (per advisor)

This pre-reg is committed BEFORE running E1. Threshold values (Gini > 0.5, abandon < 0.3)
will NOT be relaxed based on results. If results don't pass, the right action is design
v0.8 patch with EXPLICIT rationale (e.g., "G2 criterion redefined to frozen-window scope")
or honest abandonment, NOT silent threshold loosening.

---

## Result (TBD)

Filled after E1 + E4 complete.
