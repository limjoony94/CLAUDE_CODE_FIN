# G2 Falsification Close-out — ABM v1 Hypothesis EMPIRICALLY FALSIFIED

**Date**: 2026-05-01
**Status**: T-G2 CLOSED via empirical falsification (not pass).
**Verdict**: (a) abandonment confirmed via E2-extended trajectory plateau analysis.

---

## Final evidence — E2-extended 20k bar trajectory

| Bar | Gini | Δ from previous |
|-----|------|-----------------|
| 0 (initial) | 0.4389 | — |
| 5000 | 0.4899 | +0.0510 |
| 10000 | 0.5028 | +0.0129 |
| 15000 | 0.5082 | +0.0054 |
| 20000 | 0.5065 | -0.0017 |

**Pattern**: One-time burst from Pareto initial (0.44 → 0.50) over first 5k bars, then complete plateau. Total amplification 0.44 → 0.51 = +0.068 over 20k bars, but +0.051 of that occurred in the first 5k.

**Per advisor decision tree (mechanical application)**:
- Plateau pattern (Gini stalls in 0.50-0.51 range from bar 10k onwards) → **(a) abandonment CONFIRMED**
- Mechanism produces one-time burst then stalls (advisor's exact predicted pattern for plateau case)

---

## Architecture v1.1 Section 6.2 hypothesis status

**Hypothesis**: "Wealth-weighted order sizing mechanism produces emergent whale concentration via PnL feedback amplifying initial wealth differences."

**Evidence accumulated across all G2 experiments**:

| Experiment | Setup | Final Gini | Verdict |
|------------|-------|------------|---------|
| Default G2 (10k bar smoke) | Uniform 1000 + admissions 1/600 | 0.191 | Abandon trigger fired |
| E1 rate=1/3600 | Uniform 1000 + admissions 1/3600 | 0.432 | Reflects bimodal initial heterogeneity |
| E1 rate=0 (no admissions) | Uniform 1000, 10k bars | 0.096 | PnL alone produces +0.096 |
| E4 frozen-window | Same as default but T_open=5000 | 0.288 | Reflects bimodal at T_open boundary |
| E2 (Pareto x10) | Pareto-distributed initial, 5k bars | 0.490 | +0.051 amplification |
| E3 (fixed-size, no wealth-weighting) | Uniform 1000, fixed 0.001 BTC | 0.0007 | Skill alone: zero concentration |
| **E2-extended (Pareto x10, 20k bars)** | **Same as E2 but 4× longer** | **0.507 (PLATEAU)** | **+0.068 total, +0.017 after 5k → STALLS** |

**Decisive finding (E2-extended)**: Even with maximally favorable conditions (seeded Pareto initial heterogeneity, no admission dilution, 4× longer evaluation horizon), the mechanism **plateaus at Gini ~0.50** rather than continuing to grow toward the 0.55 design threshold or beyond.

**Conclusion**: Architecture v1.1 Section 6.2 wealth-weighted-sizing concentration hypothesis is **EMPIRICALLY FALSIFIED**. The mechanism amplifies pre-existing heterogeneity by a fixed amount (~+0.05-0.07 Gini) and then stalls regardless of horizon length.

---

## Root cause (mechanism analysis)

Wealth-weighted sizing produces multiplicative wealth dynamics: `wealth_{t+1} = wealth_t × (1 + r_t)` where `r_t` is the per-bar return fraction. Under a multiplicative random walk:
- Cross-sectional log-wealth variance grows linearly: `Var(log w) ∝ T × σ_r²`
- Cross-sectional log-wealth mean stays approximately constant
- Gini converges to a stationary value determined by `σ_r / mean_r ratio`

For our 5 canonical agents:
- Strategies have similar expected per-bar PnL (`mean_r ≈ 0`, near zero-sum among traders)
- Friction (taker 0.05% / maker 0.02%) acts as a uniform negative `mean_r`
- Strategy-specific volatility `σ_r` differs only modestly across families
- Result: stationary Gini ≈ 0.5 from heterogeneous initial; ≈ 0.1 from uniform initial

This stationary behavior is a **structural property** of the multiplicative wealth dynamics under near-zero-sum trading, not a parameter calibration issue. No tuning of `wealth_fraction`, admission rate, or simulation horizon will overcome it within the current architecture.

---

## Downstream impact

| Phase | Status | Reason |
|-------|--------|--------|
| **T-G2** | ✅ CLOSED via falsification | Architecture hypothesis empirically tested and rejected |
| **T-G3 (substrate discovery)** | ⏸️ STRUCTURALLY BLOCKED | No emergent whales → nothing to extract substrate from |
| **T-G4 (BingX L2 friction-pass)** | ⏸️ BLOCKED | Was always low-probability (advisor Q4(b)+Q5(c) high-prior-of-failure flag) |
| **T-G5 (deploy)** | ⏸️ INDEFINITELY DEFERRED | Was conditional on G4 |

---

## What was completed (Path A: research methodology contribution)

The 6-month research project produced the following durable deliverables in ~2 sessions:

| Deliverable | Status |
|-------------|--------|
| **G0**: Custom ABM (orderbook + scheduler + 5 canonical agents + registry + wealth + admission + simulation + NDJSON logger) | ✅ 14/14 acceptance criteria green |
| **G1**: 3-anchor inverse-recovery MVP (Signature K-means / Parametric LogReg / IRL behavioral cloning) | ✅ All 3 anchors PASS with margins |
| **G2 perf optimization**: Leaderboard cache + orderbook strict toggle (~185× speedup) | ✅ 10k bar smoke: 6956s → 397s |
| **G2 architecture test**: 7-experiment pre-registered diagnostic with mechanical decision tree | ✅ Verdict (a) reached without rationalization |
| **G2 falsification documentation**: Multi-experiment evidence + root cause + plateau confirmation | ✅ This document |

**Test suite final**: 192+ PASSED + 2 deselected (slow). All G0/G1 work intact.

---

## v2 Architecture options (NOT pursued — user-level decision deferred)

If user wishes to pursue Path B (deployable edge) in future, three architecture revisions could be considered:

1. **Skill-differential amplification**: Per-agent "skill" parameter biases PnL distribution; skilled agents win more per trade → wealth divergence over time. Requires empirical anchor for "skill" distribution.
2. **External wealth shock events**: Periodic exogenous wealth injection/withdrawal (e.g., simulated whale OTC purchases, leveraged liquidations). Requires v1 architecture changes to ABM.
3. **Heavy-tail strategy edge**: Replace 5 canonical families with strategies that have HEAVY-TAIL PnL distributions (e.g., breakout-trend-following with rare large wins). May produce concentration through win-magnitude differential.

These are v2 architectural decisions, not parameter tuning. Each requires ~weeks of work and a fresh research project commitment.

---

## Honest scientific outcome

The architecture v1.1 hypothesis was tested rigorously through 7 pre-registered experiments. The mechanism does produce some concentration (+0.05-0.07 Gini one-time burst from heterogeneous initial), but is structurally insufficient for the design's pass criterion (Gini > 0.5 emerging from uniform initial within 10k bars). The plateau pattern in E2-extended confirms this is a structural ceiling, not a tuning issue.

This is a **valid scientific outcome**: the hypothesis was tested and rejected. Path A (research methodology contribution) is complete; Path B (deployable edge) requires architectural revision beyond v1.

T-G2 is marked completed (the work was done; the answer is falsification, not pass). T-G3/G4/G5 are indefinitely deferred pending user decision on whether to pursue v2 architecture.
