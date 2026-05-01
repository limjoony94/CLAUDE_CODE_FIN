---
template: plan
version: 1.2 (ABM-customized)
description: Whale Inference ABM Plan — research project, NOT web-app
variables:
  - feature: whale_inference_abm
  - date: 2026-05-01
  - author: 임준영 (with advisor + Claude Opus 4.7)
  - project: CLAUDE_CODE_FIN
  - architecture_ref: bingx_rl_trading_bot/claudedocs/whale_inference_abm_architecture_v1.1.md
---

# whale_inference_abm Planning Document

> **Summary**: Open-system Agent-Based Market simulation with wealth-concentration → whale emergence → 3-anchor inverse-strategy recovery (IRL + signature + parametric prior) → BingX L2 deployment. Goal: discover NEW substrate / mechanism class beyond 28-round failure envelope.
>
> **Project**: CLAUDE_CODE_FIN
> **Version**: ABM v0.1 (initial plan)
> **Author**: 임준영 + advisor + Claude Opus 4.7
> **Date**: 2026-05-01
> **Status**: Plan Draft (Phase 0 spike pending)
> **Architecture reference**: `bingx_rl_trading_bot/claudedocs/whale_inference_abm_architecture_v1.1.md`

---

## 1. Overview

### 1.1 Purpose

Build a synthetic open-system market with heterogeneous strategy agents and wealth-weighted order sizing. Allow whales to emerge from wealth concentration. Apply three independent inverse-recovery anchors (Sequential IRL, Statistical Signature, Parametric Prior) to recover whale strategies. Validate that recovered strategies (a) constitute new substrate not in our 28-round mechanism catalog, and (b) clear BingX friction floor (avg_gross > 0.105% per trade) on real L2 data.

### 1.2 Background

**28-round failure context**: Final envelope synthesis (`bingx_rl_trading_bot/docs/04-report/final_envelope_synthesis_20260429.md`) shows 28 mechanisms × 5+ substrates all converged to friction-floor failure on retail BingX. C1 Breakout (2026-04-27) and R26 (2026-05-01) both shelved with same pattern: BT positive → LIVE negative.

**ABM hypothesis**: ABM with multi-agent interaction may generate emergent mechanisms (interaction-pattern substrate) NOT present in any single-agent backtest of human-conceived strategies. If yes → potential 29th attempt with fundamentally new mechanism class. If no → research methodology contribution + final envelope confirmation.

**Eyes-open warning**: Q4(b) + Q5(c) decisions = high-prior-of-failure territory. ABM finding new substrate that ALSO clears friction = whole game in one bet.

### 1.3 Related Documents

- **Architecture v1.1**: `bingx_rl_trading_bot/claudedocs/whale_inference_abm_architecture_v1.1.md` (5 phase gates, MVP per anchor, anti-circularity protocol, handoff firewall)
- **Final envelope synthesis**: `bingx_rl_trading_bot/docs/04-report/final_envelope_synthesis_20260429.md`
- **R26 postmortem**: `bingx_rl_trading_bot/claudedocs/r26_postmortem_20260501.md`
- **C1 postmortem**: `bingx_rl_trading_bot/docs/04-report/c1_breakout_postmortem_20260427.md`
- **5-gate deploy protocol** (memory): `strategy_deploy_5gate_protocol.md` — applies AT G4 pass, not before

---

## 2. Scope

### 2.1 In Scope

- [ ] **Phase 0 (G0)**: ABM build — continuous double auction orderbook + 5 canonical agents + wealth-weighted sizing + friction model. ABIDES-vs-custom 1-week spike → build → smoke test. Weeks 1-3.
- [ ] **Phase 1 (G1)**: 3-anchor MVP in parallel — Sequential IRL (MaxEnt), Statistical Signature (K-means clustering), Parametric Prior (Bayesian posterior). Null-baseline relative pass criteria. Weeks 4-9.
- [ ] **Phase 2 (G2)**: Wealth-concentration validity — Gini > 0.5, top-5% emergence + rank stability. Empirical BTC perp Gini calibration TODO. Weeks 10-12.
- [ ] **Phase 3 (G3)**: New-substrate discovery — pre-registration enforcement, anti-circularity 6-step protocol, predictive lift ≥ 5% over explicit-strategy baseline. Frozen-admission window for substrate extraction. Weeks 13-18.
- [ ] **Phase 4 (G4)**: Real-data forward-predictive validity — substrate detector applied to BingX Phase 1 L2 collector data, avg_gross ≥ 0.105% on 30-day OOS, WF 4/5 + bootstrap 95% CI excludes friction floor. Weeks 19-24.
- [ ] **Phase 5 (G5)**: 5-gate deploy protocol (inherited) IF G4 passes. Outside this plan's scope until G4 reached.

### 2.2 Out of Scope (v1)

- **Adaptive agents**: Strategies are fixed at instantiation in v1. RL-style adaptation deferred to v2 (architecture v1.1 patch 4).
- **Pure open-system substrate extraction**: Frozen-admission window used during G3 (architecture v1.1 patch 3). Trade-off accepted.
- **6th anchor**: Only IRL + Signature + Parametric. No additional inverse machinery.
- **Multi-asset ABM**: BTC/USDT perp only. Cross-asset deferred.
- **Order types beyond market/limit**: No stop-loss, trailing, OCO in v1 ABM.
- **5-gate deploy protocol enforcement**: Triggered ONLY at G4 pass. Until then, this plan governs.

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | Continuous double auction orderbook with limit + market orders | High | Pending (Phase 0) |
| FR-02 | 5 canonical agents (momentum, mean-rev, market-maker, random, piggyback) with parametric configs | High | Pending (Phase 0) |
| FR-03 | Wealth-weighted order sizing (size ∝ current wealth) | High | Pending (Phase 0) |
| FR-04 | Friction model: taker 0.05%, maker 0.02%, spread/slippage emergent from book | High | Pending (Phase 0) |
| FR-05 | Open-system: agents may join/leave at admission events; frozen-admission window for G3 | High | Pending (Phase 0) |
| FR-06 | Per-trade and per-bar logging schema (agent_id, side, price, size, role, timestamp) | High | Pending (Phase 0) |
| FR-07 | Sequential IRL (MaxEnt) MVP with 9-state × 5-action discretization | High | Pending (Phase 1) |
| FR-08 | Statistical Signature MVP with K-means on (size, timing, aggression, persistence) features | High | Pending (Phase 1) |
| FR-09 | Parametric Prior MVP with PyMC posterior over 5 strategy families | High | Pending (Phase 1) |
| FR-10 | Null-baseline computation harness (modal, last-action, random, uniform-prior) | High | Pending (Phase 1) |
| FR-11 | Wealth-concentration metric: Gini, top-K share, rank stability | Medium | Pending (Phase 2) |
| FR-12 | Substrate pre-registration system (git-committed hash before lift test) | High | Pending (Phase 3) |
| FR-13 | Anti-circularity audit (Step 4 of Section 4 in architecture v1.1) | High | Pending (Phase 3) |
| FR-14 | Substrate detector function (transferable artifact: feature_fn + threshold) | High | Pending (Phase 3) |
| FR-15 | BingX L2 detector application + forward-sim PnL evaluation | High | Pending (Phase 4) |
| FR-16 | WF 5-fold + bootstrap 1000-resample harness on L2 substrate detector | High | Pending (Phase 4) |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| **Reproducibility** | Deterministic ABM runs given seed | Fixed-seed unit test, 3 reruns identical trade tape |
| **Performance** | 10,000 sim-bars × 10 agents × 5 ABM configs in < 4 hours | Wall-clock benchmark |
| **Auditability** | All G3 substrate hypotheses pre-registered with git hash | `git log` verification of prereg files |
| **Statistical rigor** | All G1/G3/G4 metrics have null-baseline + significance test | Pytest-based metric validators |
| **Scientific honesty** | Negative results documented at same fidelity as positive | Per-gate result memo (positive AND negative) |

---

## 4. Success Criteria

### 4.1 Definition of Done (per phase gate)

**G0 — ABM Build**:
- [ ] ABM platform decision finalized (ABIDES-jpmc adopted OR custom built)
- [ ] Continuous double auction operational (FR-01)
- [ ] 5 canonical agents implemented + unit-tested (FR-02)
- [ ] Wealth-weighted sizing operational (FR-03)
- [ ] Friction model integrated (FR-04)
- [ ] Open-system admission events + frozen-admission window mechanism (FR-05)
- [ ] Logging schema (FR-06)
- [ ] Smoke test: 1,000-bar run produces non-trivial price evolution + all 5 agents active + no crashes
- [ ] Reproducibility: fixed-seed identity verified

**G1 — Synthetic Recoverability**:
- [ ] Null-baseline computation harness (FR-10)
- [ ] All 3 anchor MVPs complete (FR-07, FR-08, FR-09)
- [ ] Per-anchor pass/fail vs null baseline + 15pp / 0.4 ARI / 30pp posterior margins
- [ ] Comparison artifact: (anchor, accuracy, sample efficiency, compute cost) table
- [ ] Escalation decision: which anchor(s) proceed to full scale

**G2 — Wealth-Concentration Validity**:
- [ ] Empirical BTC perp Gini estimated from external dataset (TODO from architecture v1.1)
- [ ] Wealth-concentration metrics implemented (FR-11)
- [ ] Gini > max(0.5, empirical_target − 0.1) at T=10,000 sim-bars
- [ ] Top-5% rank stability ≥ 50% between T=5,000 and T=10,000

**G3 — New-Substrate Discovery**:
- [ ] Substrate pre-registration system operational (FR-12)
- [ ] Anti-circularity audit harness (FR-13)
- [ ] ≥ 1 substrate hypothesis passes pre-reg + audit + predictive lift ≥ 5%
- [ ] Substrate detector function written + reviewable (FR-14)
- [ ] Detector deployable on real L2 data (definition check)

**G4 — Real-Data Forward-Predictive Validity**:
- [ ] BingX Phase 1 L2 collector data integration verified
- [ ] Detector applied to OOS 30-day window (FR-15)
- [ ] avg_gross ≥ 0.105% per trade
- [ ] WF 5-fold ≥ 4/5 positive, bootstrap 95% CI excludes 0.07%
- [ ] ≥ 200 trades in OOS window (no vacuous-test BUG#37 repeat)

### 4.2 Quality Criteria (project-wide)

- [ ] All ABM code under pytest (target coverage 80%+)
- [ ] Zero look-ahead bias in inverse machinery (causality test required for each anchor)
- [ ] All G1+ pass claims have null-baseline + significance test reported
- [ ] All G3 substrate claims have pre-registration git hash
- [ ] All negative-result findings documented to same fidelity as positive
- [ ] Architecture v1.1 → v1.x revisions committed with explicit decision-log entries

---

## 5. Risks and Mitigation

(Full register in architecture v1.1 Section 7. Highlights:)

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| ABM never reaches G2 (no whale emergence) | High | Medium | Multiple capital-weighting / heterogeneity configs prepared |
| All 3 anchors fail G1 | Project-killing | Low | Canonical agents are by-construction recoverable |
| G3 yields 0 substrates | Project-pivot | High (advisor flag) | Architecture pivots to (a)-only research methodology contribution |
| G3 yields substrate but G4 fails friction | High | Very High (28-round prior) | Document as 29th-round failure with novel methodology contribution |
| L2 collector data quality insufficient | Medium | Medium | Phase 1 collector running since 2026-04-29; verify data integrity before G3→G4 |
| Self-confirmation trap | High | High | Anti-circularity protocol (architecture Section 4) + pre-registration (patch 2) |
| Phase 0 takes > 5 weeks | Time risk | Medium | ABIDES adoption preferred over custom; abandon custom if 1-week spike inconclusive |
| 6-12 month scope drift | High | High | Phase gates G0-G4 ARE the scope-control mechanism; quarterly G-evaluation enforced |

---

## 6. Architecture Considerations (ABM-customized)

### 6.1 Project Level Selection (NOT web-app)

| Level | Characteristics | Selected |
|-------|-----------------|:--------:|
| Starter | Static sites | ☐ |
| Dynamic | Web apps with backend | ☐ |
| Enterprise | Microservices | ☐ |
| **Research** | Python sim + Jupyter notebooks + reproducibility infrastructure | ✅ |

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| Language | Python / Julia / Rust | Python | Library ecosystem (numpy, pandas, PyMC, gymnasium); team familiarity |
| ABM substrate | ABIDES-jpmc / custom Mesa-based / pure NumPy | **TBD (Phase 0 spike)** | 1-week spike before commit |
| IRL library | irl-imitation / inverse-rl / custom MaxEnt | irl-imitation (default) | Active maintenance; pivot if API issues |
| Bayesian inference | PyMC / Stan / Numpyro | PyMC | Pythonic, posterior visualization built-in |
| Clustering | scikit-learn | scikit-learn | Standard for K-means / DBSCAN |
| Reproducibility | seed + pickle vs nix vs docker | seed + requirements.txt | Lightweight, fits research project |
| Data storage | parquet / HDF5 / SQLite | parquet | Trade tape volume; pandas integration |
| Logging | structlog / python-logging | structlog | JSON-structured, queryable |
| Notebook env | Jupyter / VSCode notebooks | Jupyter | Standard for research |

### 6.3 Folder Structure (proposed)

```
whale_inference_abm/
├── abm/                    # ABM core (FR-01 to FR-06)
│   ├── orderbook.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── momentum.py
│   │   ├── mean_reversion.py
│   │   ├── market_maker.py
│   │   ├── random_agent.py
│   │   └── piggyback.py
│   ├── friction.py
│   ├── wealth.py
│   └── simulation.py
├── inverse/                # 3 anchors (FR-07 to FR-10)
│   ├── irl_maxent.py
│   ├── signature_clustering.py
│   ├── parametric_bayes.py
│   └── null_baselines.py
├── substrate/              # G3 machinery (FR-12, FR-13, FR-14)
│   ├── prereg.py           # git-hash pre-registration
│   ├── audit.py            # anti-circularity 6-step
│   ├── detector.py         # transferable artifact
│   └── lift_test.py
├── deployment/             # G4 (FR-15, FR-16)
│   ├── l2_loader.py        # BingX Phase 1 collector integration
│   ├── forward_sim.py
│   ├── wf_harness.py
│   └── bootstrap.py
├── tests/                  # pytest suite
├── notebooks/              # exploratory + per-gate evaluation
├── results/
│   ├── g0_smoke/
│   ├── g1_recoverability/
│   ├── g2_concentration/
│   ├── g3_substrate/       # includes prereg/ subdir
│   └── g4_realdata/
├── prereg/                 # substrate definitions, git-tracked
└── requirements.txt
```

Location decision: separate top-level directory `whale_inference_abm/` at repo root, NOT inside `bingx_rl_trading_bot/`. ABM is research project distinct from trading bot operational codebase. Shared dependency: BingX Phase 1 L2 collector data (read-only access).

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [✅] `CLAUDE.md` has project-level conventions
- [✅] Memory files document research protocol (`research_protocol_3day_bootstrap.md`, etc.)
- [N/A] ESLint / Prettier (Python project)
- [✅] Existing pytest infrastructure (`bingx_rl_trading_bot/scripts/tests/`)

### 7.2 Conventions to Define for ABM Project

| Category | To Define | Priority |
|----------|-----------|:--------:|
| Naming | snake_case Python conventions; agent IDs format `{family}_{seed}` | High |
| Folder structure | As Section 6.3 above | High |
| Pre-registration format | YAML frontmatter + markdown body, git-committed before lift test | High |
| Negative-result memo format | Same template as `m3_*` postmortem files | High |
| Reproducibility | All ABM runs require seed; results directory includes seed in filename | High |
| Causality testing | Each inverse algorithm has a `test_no_lookahead_*.py` | High |

### 7.3 Environment Variables

| Variable | Purpose | Scope |
|----------|---------|-------|
| `ABM_DATA_DIR` | Trade tape + orderbook snapshots | dev |
| `BINGX_L2_DIR` | Phase 1 collector output (read-only) | dev |
| `ABM_RESULTS_DIR` | Per-gate evaluation outputs | dev |

### 7.4 Pipeline Integration

N/A — research project, not 9-phase pipeline. PDCA Plan → Design → Do → Analyze flow used instead.

---

## 8. Phase Tasks (advisor-aligned: tasks per phase-gate, NOT per implementation step)

| Task | Gate | Phase | Status | Blockers |
|------|------|-------|--------|----------|
| **T-G0** Phase 0: ABM build (ABIDES spike → build → smoke test) | G0 | Weeks 1-3 | Pending | None |
| **T-G1** Phase 1: 3-anchor MVP + null-baseline + comparison | G1 | Weeks 4-9 | Pending | T-G0 |
| **T-G2** Phase 2: Wealth-concentration validity + Gini calibration | G2 | Weeks 10-12 | Pending | T-G0 (parallel-able with T-G1 partially) |
| **T-G3** Phase 3: Substrate prereg + anti-circularity + lift test | G3 | Weeks 13-18 | Pending | T-G1, T-G2 |
| **T-G4** Phase 4: BingX L2 forward-predictive validity | G4 | Weeks 19-24 | Pending | T-G3 + L2 collector data |
| **T-G5** Phase 5: 5-gate deploy protocol | G5 | Post-G4 | Out of scope until G4 pass | T-G4 |

---

## 9. Next Steps

1. [ ] Approve this plan (user)
2. [ ] `/pdca design whale_inference_abm` — design document for Phase 0 (ABM core spec) — first design priority
3. [ ] Phase 0 ABIDES-vs-custom 1-week spike (first real work)
4. [ ] TaskCreate per phase-gate (T-G0 through T-G5) at design-phase entry
5. [ ] /pdca status checkpoints monthly minimum

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-01 | Initial plan from architecture v1.1 | 임준영 + advisor + Claude Opus 4.7 |
