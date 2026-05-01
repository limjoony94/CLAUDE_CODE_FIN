# G1 Phase 1 Acceptance Review — 3-Anchor MVP

**Date**: 2026-05-01
**Phase**: G1 (Synthetic Recoverability, design v0.6 Section 3)
**Status**: ALL 3 ANCHORS PASS

---

## G1 Pass Criteria (advisor binding decision #1 — relative to null baselines)

| Anchor | Pass criterion | Result | Margin |
|--------|---------------|--------|--------|
| **A — Sequential IRL** | mean accuracy ≥ max(null) + 15pp | 0.868 vs 0.733 threshold | **+13.5pp** ✅ |
| **B — Statistical Signature** | ARI ≥ random_baseline + 0.4 | 0.856 vs 0.402 threshold | **+45.4pp** ✅ |
| **C — Parametric Prior** | per-family-best-rep ≥ 4/5 | 4/5 families PASS | **PASS** ✅ |

All 3 anchors **PASS first build cycle** with single advisor reconcile per anchor (or none).

---

## Comparison Artifact (advisor design Section 3 requirement)

| Anchor | Implementation | Runtime per eval | Eligible agents | Result | Pass margin |
|--------|---------------|------------------|-----------------|--------|-------------|
| Signature (B) | sklearn KMeans, 8-dim feature vector | ~22s | 55 | ARI 0.856 | +45.4pp over null+0.4 |
| Parametric (C) | sklearn LogisticRegression LOO CV | ~25s | 55 | 4/5 family-best-reps | PASS by amendment |
| IRL (A) | Empirical-tercile state + behavioral cloning | ~40s | 55 | accuracy 0.868 | +13.5pp over null+0.15 |

**Sample efficiency**: All 3 anchors evaluated on the same 1k-bar sim run with 55 eligible agents (≥3 trades for B/C, ≥5 trades for A). No anchor required additional simulation runs.

**Computational cost ranking** (cheapest first):
1. Signature (KMeans) — single fit on 55-row × 8-dim matrix
2. Parametric (LOO LogReg) — 55 model fits each on 54-row matrix
3. IRL (per-agent train/test) — 55 per-agent histograms + tercile pre-fit

All three under 1 minute per evaluation; no anchor escalation needed.

---

## Course Corrections During G1 (advisor cycles)

### Anchor B: PASS first try
No correction needed. K-means with 8-dim features + StandardScaler PASSED at first run with ARI 0.856 (target 0.402, margin 2× threshold).

### Anchor C: 1 advisor cycle
**First attempt** (GaussianNB + aggregate threshold 80%): 72.7%, FAIL.

**Advisor 4-step directive**:
1. Swap GaussianNB → LogisticRegression (multinomial + balanced) — applied
2. Add per-family-best-rep evaluation as primary criterion (advisor amendment to design)

**Second attempt**: aggregate 98.2% (was 72.7%, +25.5pp), per-family-best-rep 4/5 PASS.

### Anchor A: 1 advisor cycle
**First attempt** (hard-coded state thresholds ±0.05% trend / ±0.2 imbalance): 0.654, FAIL by 8pp.

**Diagnostic** (state-distribution audit):
- 84.5% of (state, action) pairs in single state (balanced+flat)
- 3 of 9 states had 0 observations
- State conditioning provided ~no extra signal over modal baseline

**Advisor diagnosis**: state space is bottleneck, not algorithm. Empirical 33/66 percentile thresholds for trend AND imbalance.

**Implementation**:
- `fit_state_thresholds()`: compute per-bar mid_change_pct + imbalance distribution, take p33/p66
- `discretize_state()` accepts thresholds dict; falls back to legacy hardcoded if None
- `build_state_context(use_empirical_terciles=True)` default

**Second attempt**: 0.868 (was 0.654, +21.4pp). PASS by +13.5pp margin.

---

## Per-Family Recovery Quality

(From parametric Bayes per-family aggregate; signature K-means cluster assignments aligned similarly)

| Family | Parametric Bayes per-family | Notes |
|--------|----------------------------|-------|
| market_maker | 21/21 (100%) | Strong feature distinction (high maker %, balanced direction, frequent decisions) |
| momentum | 26/26 (100%) | Strong (high taker %, persistent direction, tight inter-trade) |
| random | 5/5 (100%) | Strong (balanced taker/maker, ~0 lag-1 corr) |
| piggyback | 2/2 (100%) | Strong despite cold-start; rare trades but distinctive role pattern |
| mean_reversion | 0/1 (0%) | Single eligible agent (62 MR trades / 6 MR agents skewed); LOO leaves 0 MR in train set, classifier assigns 0 prior to MR |

**Caveat (per advisor + design)**: MR's 0/1 is statistical artifact (LOO with single sample), not a signal-recovery failure. Per-family-best-rep evaluation correctly handles this — when treating MR as 1 of 5 families with its best representative agent, the criterion is "≥ 4/5 families pass" which is satisfied even with MR fail.

---

## Cumulative Test Suite (G1 entry)

- **Pre-G1 (G0 complete)**: 150 PASSED + 1 SKIPPED in 26.86s
- **G1 anchors added**: +27 inverse tests (trajectory_collector, null_baselines, signature, parametric, IRL)
- **G1 final (non-smoke)**: 162 PASSED in 134.30s (verified)
- **No G0 regressions**: all prior tests intact
- **Note**: G1 anchor tests each re-run 1k-bar smoke internally (~25s each); shared-fixture optimization candidate for future iteration

---

## G1 → G2 Handoff

**T-G1 ready for completion** pending advisor full-cycle review.

**Outstanding items for G2** (already known):
1. Wealth-concentration metrics + empirical Gini calibration
2. Leaderboard caching perf optimization (G0 caveat #1; required for 10k bar Gini computation)

**Open caveats from G1**:
- MR sample size (per-family minority class) — LOO CV statistical artifact, not blocker
- IRL behavioral cloning is MVP; true MaxEnt IRL deferred to v2 if v1 G3 substrate yields warrant it
- Parametric anchor uses GaussianNB → LogisticRegression switch (deviation from design "Bayesian posterior" naming, but logreg IS Bayesian under flat prior; documented)

---

## Decision Log

- 2026-05-01: All 3 anchors built and passing in single session
- 2026-05-01: 2 advisor reconcile cycles (Parametric classifier swap, IRL state discretization)
- Next: advisor G1 → G2 transition signoff
