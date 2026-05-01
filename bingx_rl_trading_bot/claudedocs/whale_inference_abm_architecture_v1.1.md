# Whale Inference ABM — Architecture v1.1

**Date**: 2026-05-01
**Status**: Architecture v1.1, advisor-reviewed (v1 + 5 patches applied), ready for `/pdca plan` entry
**v1.1 patches**: (1) Null-baseline G1 thresholds, (2) Pre-registered substrate definitions, (3) Frozen-admission window, (4) Fixed-strategy v1, (5) Phase 0 ABM build added
**Goal class**: Research project → Eventual deployable edge (Q1: a→b phased)
**Scope**: 6~12 month research project (Q2: 3-anchor full parallel accepted)

---

## 1. Project Frame (User decisions)

| Q | Decision | Architectural implication |
|---|----------|---------------------------|
| Q1 | (a) Research → (b) Deployable phased | 5-gate strategy_deploy_5gate_protocol enforced ONLY when entering (b). (a) phase has its own kill gates. |
| Q2 | (b) 3-anchor full parallel | Sequential IRL + Statistical Signature + Parametric Prior all built. NOT 3 fully-built systems Day 1 — MVP per anchor first, escalate ones that show signal. |
| Q3 | A → L2 real data | Synthetic ABM first (ground-truth available), BingX Phase 1 L2 collector second (ground-truth absent, indirect validation). |
| Q4 | (b) New substrate / mechanism class discovery | ABM = hypothesis generator. NOT precision-tuning of existing 28 mechanisms. |
| Q5 | (c) BT friction-pass | Recovered strategy must clear avg_gross > friction × 1.5 margin (= 0.105% on BingX taker 0.07%) on real BT. Most stringent of three options. |

**Eyes-open warning** (advisor flagged):
Q4(b) + Q5(c) combined = high-prior-of-failure. ABM finding genuinely new substrate that ALSO clears friction = whole game in one bet. 28-round evidence is friction floor binding. Architecture must surface this risk continuously, not bury it.

---

## 2. Phase Gates with Explicit Kill Criteria

Five gates (G0-G4) plus inherited G5. Each MUST have numeric pass criterion + abandon trigger. No "we'll evaluate later" gates.

### Gate G0 — ABM Build Complete (v1.1 patch 5)

**Question**: Is the ABM substrate built, smoke-tested, and producing realistic-looking trajectories before any inverse-machinery work begins?

**Pass criterion**:
- ABM platform decision finalized (ABIDES vs custom — 1-week spike in `/pdca plan`)
- Continuous double auction orderbook operational
- 5 canonical agents (momentum, mean-rev, market-maker, random, piggyback) implemented + unit-tested
- Wealth-weighted order sizing operational
- Friction model (taker 0.05% / maker 0.02%, spread+slippage emergent) operational
- Smoke test: 1,000-bar run produces (a) non-trivial price evolution, (b) trade tape with all 5 agent types active, (c) no crashes / divergences
- Logging schema: per-trade record (agent_id, timestamp, side, price, size, role) + per-bar orderbook snapshot

**Abandon trigger**:
- ABM build > 5 weeks elapsed → scope was wrong, simplify or restart with smaller mechanism set
- Smoke test fails 3 consecutive attempts → architecture incompatible, re-evaluate Section 6.1

**Time budget**: 3 weeks (Phase 0 = weeks 1-3 of project).

**Why this gate matters**: G1 assumes ABM exists. v1 underestimated build time. Without G0 budget, G1 becomes a 3-week MVP push that kills at least one anchor unfairly.

### Gate G1 — Synthetic Recoverability

**Question**: Can the inverse machinery recover known agent strategies from a canonical multi-agent ABM where ground truth IS the agent code we wrote?

**Pre-G1 calibration step (MANDATORY before pass evaluation)**:

Compute null-model baselines per metric per canonical agent BEFORE running inverse machinery:
- IRL null baselines: (a) "always predict modal action" accuracy, (b) "copy last action" accuracy, (c) random uniform = 1/|action_space|
- Signature null: ARI of random K-means with K=5 over same feature vectors
- Parametric null: uniform prior posterior (1/5 = 20%)

Pass criterion is RELATIVE to null, not raw threshold. Raw thresholds below are illustrative — replaced by null+margin at calibration.

**Pass criterion (relative)**:
- Sequential IRL anchor: action-prediction accuracy ≥ max(null_baselines) + 15 percentage points on held-out trajectories
- Statistical signature anchor: ARI ≥ random_ARI + 0.4 (where random_ARI typically near 0)
- Parametric prior anchor: posterior mass on correct family ≥ uniform_prior + 30 percentage points (i.e., > 50% if uniform = 20%) for ≥ 4/5 agents

**Abandon trigger (relative)**:
- IRL accuracy ≤ null_baseline + 5pp → no signal above dumbest baseline → kill IRL anchor
- Signature ARI ≤ random_ARI + 0.1 → kill signature anchor
- Parametric posterior ≤ uniform + 10pp on correct family → kill parametric anchor

**Rationale**: 70% raw IRL accuracy is meaningless if "always predict modal" achieves 80% on a momentum agent. Null-baseline relative gates measure actual inverse-recovery signal, not artifact of canonical-agent design.

**If all three abandon**: STOP. Project terminates. Inverse problem ill-posed for this market design.

**Time budget**: weeks 4-9 of project (6 weeks for G1 work, after Phase 0 G0 weeks 1-3). Total to G1 evaluation = 9 weeks. If not at G1 evaluation by week 9, scope was wrong.

### Gate G2 — Wealth-Concentration Validity

**Question**: Does ABM exhibit non-trivial wealth concentration / whale emergence under realistic parameters?

**Pass criterion**:
- After T = 10,000 sim-bars with capital-weighted order sizing, top-5% of agents hold > 40% of total capital (Gini > 0.5)
- Top-5% emerge from initially uniform wealth distribution (NOT seeded as whales)
- Distribution rank-stability: top-5% at T=10,000 overlaps top-5% at T=5,000 by ≥ 50% (not pure noise)

**TODO before G2 evaluation**: empirically estimate BTC perp wealth Gini from on-chain wallet clustering or known whale-tracker datasets. The 0.5 target is currently arbitrary. If real BTC perp Gini is 0.7+, the 0.5 target is too lax and ABM "passes" without realistic concentration. Calibrate before G2 runs.

**Abandon trigger**:
- Gini < 0.3 at T=10,000 → market mechanics don't generate concentration. Architecture broken.
- Top-5% turnover > 80% between checkpoints → no stable whales, only transient leaders. Inverse problem ill-defined target.

**If fails**: Revisit market mechanism (Section 6) before re-attempting G1.

### Gate G3 — New-Substrate Discovery (the hard one)

**Question**: Does the inverse machinery, when applied to ABM-emergent dominant agents, identify mechanisms NOT present in our explicit agent code?

**This is the conjecture-test gate.** ABM by construction only contains mechanisms we programmed. Emergent properties (price impact dynamics, regime feedback, multi-agent coordination) are not strategies but side-effects. The claim "new substrate emerges from interaction" must be testable, not tautological.

**Operational definition**:
1. We catalog the explicit strategies coded into the N agents (e.g., 5 canonical strategies from G1).
2. After ABM runs to wealth-concentrated state, top-5% agents may have ACQUIRED behaviors via adaptation (if adaptation is enabled — see Section 6) or may be USING combinations of explicit strategies that exploit emergent market features.
3. New substrate = inverse machinery extracts a feature/signal from top-5% agent trajectories that:
   - (a) Is NOT a direct readout of any explicit strategy parameter
   - (b) Has predictive power for top-5% next-action that the explicit-strategy decomposition does NOT
   - (c) Can be defined as an observable on real L2/trade tape data

**Pass criterion**:
- ≥ 1 substrate hypothesis meets (a), (b), (c) above
- Predictive lift over best explicit-strategy baseline ≥ 5% in held-out ABM trajectories
- Substrate hypothesis writable as detector function `f(orderbook_state, trade_tape) → signal` deployable on L2 data

**Abandon trigger**:
- 0 substrate hypotheses meet criteria after 3 ABM configurations tested → ABM cannot generate new mechanisms beyond what's coded. Hypothesis-generator value claim falsified.

**If fails**: Project pivots to (a)-only research methodology contribution. Drop (b) deployable phase. Don't proceed to L2.

**Critical anti-circularity check**: A "substrate" that turns out to be a known explicit strategy under disguise (e.g., recovering "momentum" feature when momentum agent is in the code) does NOT count. Cataloging step (1) is enforcement mechanism.

### Gate G4 — Real-Data Forward-Predictive Validity

**Question**: Does the substrate hypothesis from G3, when applied as a detector on BingX L2 / trade tape, predict next-N-bars price/volume better than random?

**Pass criterion**:
- Detector outputs signal on L2 data
- Forward-sim PnL on out-of-sample 30-day window: avg_gross ≥ 0.105% per trade (friction × 1.5)
- Walk-forward 5-fold: ≥ 4/5 folds positive
- Bootstrap 1000-resample: 95% CI on avg_gross excludes friction floor (0.07%)
- Sample size: ≥ 200 trades on test window (no vacuous-test BUG#37 repeat)

**Abandon trigger**:
- avg_gross < 0.07% → 28-round confirmation. Substrate doesn't beat friction.
- avg_gross between [0.07%, 0.105%] → "interesting but not deployable". Document as research finding (a)-phase success, abandon (b)-phase.
- < 100 trades in 30-day window → substrate too rare. Cannot statistically validate. Re-design detector or abandon.

**If passes**: Enter strategy_deploy_5gate_protocol Gates 1-5 (the existing protocol from memory).

### Gate G5 — Deploy Protocol (existing)

Inherited from `strategy_deploy_5gate_protocol.md`. NOT redefined here. Triggered only after G4 pass.

---

## 3. MVP Per Anchor (Phased Commitment Within Research Project)

Advisor pushback: "3-anchor parallel" ≠ 3 fully-built systems Day 1. Build minimum experiment per anchor that produces comparable artifact, escalate based on G1 signal.

### MVP Sequential IRL (Anchor A — primary)

**MVP scope**:
- Algorithm: MaxEnt IRL (Ziebart 2008). Concrete, well-understood, library available (e.g., `irl-imitation`).
- State space: discretized orderbook imbalance (3 bins) × trend regime (3 bins) = 9 states. Small enough for tabular MaxEnt.
- Action space: {buy_aggressive, buy_passive, hold, sell_passive, sell_aggressive} = 5 actions.
- Training data: 1000 trajectories from 1 canonical momentum agent in 3-agent ABM.
- Success: recover reward function whose argmax-policy ≥ 70% action-match on 200 held-out trajectories.

**Time budget**: 2 weeks. Includes data generation + training + evaluation harness.

**Escalation criteria**:
- Hits 70% on momentum agent → expand to 5 canonical agents → continuous state space → GAIL/AIRL
- Stuck below 50% on momentum → diagnose state/action discretization. If still stuck, downgrade to secondary.

### MVP Statistical Signature (Anchor B — secondary)

**MVP scope**:
- Features: (a) trade-size distribution (5 quantile bins), (b) inter-trade arrival time (Poisson rate estimate per regime), (c) aggression ratio (taker/maker fraction), (d) directional persistence (lag-1 autocorrelation of trade direction)
- Method: K-means clustering on feature vector per agent, K=5 (matches canonical agent count for G1).
- Evaluation: Adjusted Rand Index between cluster labels and true agent identity.
- Training data: same 1000 trajectories as IRL MVP.
- Success: ARI ≥ 0.6.

**Time budget**: 1 week. Simpler than IRL.

**Escalation criteria**:
- ARI ≥ 0.6 → expand features to L2 microstructure (queue position, cancel rate, etc.) → DBSCAN for unknown-K
- ARI < 0.3 → fundamental signature insufficient. Downgrade or merge into IRL feature engineering.

### MVP Parametric Prior (Anchor C — tertiary)

**MVP scope**:
- Strategy families: 5 parametric forms covering canonical agents (e.g., momentum: `signal = sign(price[t]-price[t-N])`, mean-rev: `signal = -sign(price[t]-MA[t,N])`, with N as parameter).
- Inference: PyMC Bayesian posterior over (family, parameter) given observed trajectory.
- Evaluation: Posterior probability mass on TRUE family for each test agent.
- Training data: same 1000 trajectories.
- Success: posterior > 50% on correct family for ≥ 4/5 agents.

**Time budget**: 2 weeks. Bayesian inference setup non-trivial.

**Escalation criteria**:
- 4/5 success → expand family library, add parameter priors from real-market estimates
- < 3/5 success → family library too narrow or trajectory too short. Diagnose before downgrade.

### Comparison Artifact (G1 evaluation)

After all three MVPs: comparable table of (anchor, accuracy, sample efficiency, computational cost). Decide which anchor proceeds to full scale based on G1 numbers, not vibes.

**This is what "3-anchor parallel" actually means in (a) phase**: 3 MVPs in parallel weeks 1-3, comparison at week 4, escalation decision at week 5. Total (a)-phase G1 budget: 6 weeks.

---

## 4. New Substrate Validation Pipeline (G3 enforcement)

Operational anti-circularity protocol:

```
Step 1 (cataloging — done before ABM run):
  Record explicit_strategies[] = list of all coded agent decision functions
  Each strategy gets unique ID + parameter signature

Step 2 (ABM run):
  Run wealth-concentration ABM until G2 pass
  Identify top-5% emergent whales

Step 3 (substrate extraction — IRL/signature anchor applied to whales):
  For each whale w:
    feature_vector_w = inverse_machinery(w.trajectory)

Step 4 (anti-circularity audit):
  For each feature in feature_vector_w:
    Check if feature is reducible to readout of any explicit_strategies[i] parameter
    If yes → mark as "explicit-derived"
    If no → mark as "candidate substrate"

Step 4.5 (PRE-REGISTRATION — anti-leakage):
  BEFORE running predictive-lift test, write substrate definitions to
  claudedocs/whale_inference_abm/substrate_prereg_{date}.md with:
    - Substrate ID + human-readable definition
    - Feature function pseudocode
    - Expected mechanism (why this should have predictive lift)
    - Hash committed to git
  Post-hoc-invented substrates (added after Step 5 results visible) are EXCLUDED.
  Mirrors trading-research pre-registration discipline (memory:
  research_protocol_overfit_guards.md).

Step 5 (predictive lift test — ONLY on pre-registered substrates):
  Baseline = best fit of explicit_strategies[] to whale trajectory
  Candidate = baseline + candidate_substrate features (from prereg list)
  Lift = improvement in held-out prediction accuracy
  If lift ≥ 5% → substrate hypothesis CONFIRMED
  Substrates not in prereg → research note only, NOT counted toward G3 pass

Step 6 (deployability check):
  Can candidate_substrate be computed from real L2 / trade tape?
  Define detector function f(orderbook_state, trade_tape) → signal
  If yes → proceed to G4
  If no → substrate not deployable, file as research note only
```

**Trap to avoid**: substrate that backtests on L2 is NOT validation by itself. Circular if ABM's purpose was to find L2-detectable patterns. The non-circular test: substrate must come from ABM-emergent behavior FIRST, then be searched for in L2. Backwards (find pattern in L2, replicate in ABM) is also valuable but is a different pipeline (not what Q4-b commits to).

---

## 5. Synthetic → Real Handoff Artifact

**The artifact that crosses the synthetic/real boundary is NOT the trained inverse model.**

Trained inverse model fits synthetic agent family. Will overfit to synthetic prior. Cannot transfer.

**What transfers**:

| Artifact | Transfers? | Self-confirmation risk |
|----------|------------|------------------------|
| Trained IRL reward function | NO | High — overfits synthetic agents |
| Strategy family priors | PARTIAL | Medium — bias toward synthetic-like strategies |
| Feature definitions (substrate detector functions) | YES | Low — features are observable |
| Detection algorithm (clustering / IRL inference procedure) | YES | Low — algorithm is general |
| Anti-circularity checklist | YES | Zero — methodology |

**Handoff specification**:

```python
# crosses synthetic → real boundary
class SubstrateDetector:
    def __init__(self, hypothesis_id: str, definition: str):
        self.hypothesis_id = hypothesis_id
        self.definition = definition  # human-readable
        self.feature_fn = ...  # f(orderbook, trade_tape) → scalar
        self.threshold = ...   # signal-trigger value

    def detect(self, l2_window: L2Snapshot) -> Signal:
        return self.feature_fn(l2_window) > self.threshold

# does NOT cross boundary
class TrainedIRLModel: ...  # synthetic-only
class CanonicalAgentLibrary: ...  # synthetic-only
```

The L2 phase re-runs detection algorithm + features on REAL data. It does NOT apply the synthetic-trained model to real data. This is the firewall against self-confirmation bias.

---

## 6. ABM Forward Simulation Core Design

(Skeleton — full spec deferred to /pdca design phase. Architecture-level decisions only.)

### 6.1 Market mechanism

**Choice**: Continuous double auction with limit orderbook.

**Rationale**: Most realistic for crypto perp markets. Other options (Kyle's lambda price-impact model, Glosten-Milgrom) abstract away the orderbook microstructure where Q4-b "new substrate" is most likely to emerge.

**Implementation**: Custom Python or use existing (e.g., `abides-jpmc` from JP Morgan, MIT-licensed, designed for ABM market simulation). Existing library reduces 4-8 weeks of infrastructure work.

**Risk**: ABIDES designed for institutional study. Crypto perp specifics (funding rate, liquidations, leverage cascades) may need extension.

### 6.2 Agent population

**Initial canonical set (G1 baseline)**:
1. Momentum (lookback parameter)
2. Mean-reversion (MA-distance parameter)
3. Market-maker (spread + inventory parameter)
4. Random (uniform action distribution)
5. Piggyback (delayed copy of recent best-performing agent)

**Wealth-concentration mechanism**: Each agent's order size proportional to current wealth. Wealth grows/shrinks with PnL. No external capital injection (closed-system within sim, but open in the sense that strategies are heterogeneous). New agents permitted to "join" at random intervals (open-system per user requirement) — fresh agents drawn from a strategy distribution, NOT necessarily the same as initial 5.

**Open-system / fixed-target collision resolution (v1.1 patch 3)**:

Open-system requirement (agents joining mid-stream) and fixed-target inverse problem (recovering whale strategies) collide: if population drifts, top-5% identity drifts, inverse model fits a moving target.

**Decision: option (a) — frozen-admission substrate-extraction window.**

ABM run is partitioned into phases:
- **Open phase** (T = 0 to T_open): full open-system dynamics. Agents join freely. Wealth concentration develops. Used for G2 evaluation.
- **Frozen phase** (T_open to T_open + T_extract): admission frozen. Top-5% identity stable by construction. Inverse machinery applied to whale trajectories from this window only. Used for G3 evaluation.
- **Re-open phase** (optional, post-extraction): admission resumed. Used for stability/robustness checks.

Document this as deliberate deviation from pure-open-system. The user's "open system" requirement is honored at the market-level (heterogeneous strategies, free entry) but suspended during the inverse-extraction window for tractability. Trade-off acknowledged: substrate hypotheses are properties of partially-frozen state, not steady-state open dynamics.

**T_open and T_extract values**: deferred to /pdca design. Initial guess T_open = 7,000 sim-bars, T_extract = 3,000 sim-bars.

**Adaptation decision (v1.1 patch 4)**: **Fixed strategies for v1. Adaptation explicitly OUT OF SCOPE.**

Each agent's decision function is set at instantiation and does not learn / adapt during the ABM run. Only WEALTH and order-size scale. Strategies fixed.

**Rationale**:
- Fixed strategies → "new substrate" can only mean interaction patterns (narrow, sharp claim). G3 test is well-defined.
- Adaptive agents → "new substrate" could be learned behaviors. Anti-circularity verification ("learned ≠ rediscovery of coded primitive") becomes substantially harder. G3 pass criterion shape changes.
- v1 should narrow the claim and sharpen the test. Adaptation can be added in v2 if v1 succeeds.

**G3 implication**: substrate hypotheses in v1 are constrained to emergent interaction patterns only (e.g., orderbook signature of multi-agent coordination, regime-feedback artifacts). Substrate is NOT "the whale learned a new strategy."

### 6.3 Time step

**Choice**: Event-driven (each agent decision is a discrete event), with synthetic timestamps. Aggregation to "bars" (1m / 5m / 15m) for L2 comparison purposes only.

### 6.4 Friction model

**Critical**: Friction must be in synthetic ABM from Day 1, not added later.

- Taker fee: 0.05% (BingX rate)
- Maker fee: 0.02% (BingX rate)
- Spread: emergent from orderbook
- Slippage: emergent from orderbook depth

If friction not modeled in synthetic, G3 substrate hypothesis may be invalidated at G4 — wasted research cycle.

---

## 7. Risk Register

| Risk | Probability | Mitigation |
|------|-------------|------------|
| ABM never reaches G2 (no whale emergence) | Medium | Multiple capital-weighting / heterogeneity configs prepared |
| All 3 anchors fail G1 | Low | Canonical agents are by-construction recoverable |
| G3 yields 0 substrates | High (advisor flag) | Architecture pivots to (a)-only contribution |
| G3 yields substrate but G4 fails friction | Very High (28-round prior) | Document as 29th-round failure with novel methodology contribution |
| L2 collector data quality insufficient | Medium | Phase 1 collector already running; verify before G3→G4 transition |
| 6-12 month research project drift / scope creep | High | Phase gates ARE the scope-control mechanism. Quarterly G-evaluation enforced. |
| Self-confirmation trap (synthetic agents trivially recovered, real data trivially fails) | High | Anti-circularity protocol Section 4. Handoff firewall Section 5. |
| 3-anchor parallel becomes 1-anchor with 2 abandoned | Likely | Acceptable. MVP-first design absorbs this. |

---

## 8. Decision Log

- 2026-05-01: User decisions Q1-Q5 received. Architecture v1 drafted.
- 2026-05-01: Advisor review of v1 → 5 patches identified, no redesign.
- 2026-05-01: v1.1 patches applied:
  - Patch 1 (G1 thresholds): null-baseline relative criterion
  - Patch 2 (G3 anti-circularity): pre-registration of substrate definitions before predictive-lift test
  - Patch 3 (open-system collision): frozen-admission window for substrate extraction (option a)
  - Patch 4 (adaptation): fixed strategies for v1, adaptation out of scope
  - Patch 5 (Phase 0): G0 ABM-build gate added, total to G1 = 9 weeks not 6
  - G2 Gini empirical TODO marked
- Next: `/pdca plan whale_inference_abm` entry. Phase 0 ABIDES-vs-custom 1-week spike is first plan-phase work.

---

## Resolved Open Questions (from v1)

1. ~~Is G3's "new substrate" operational definition tight enough?~~ → Patch 2: pre-registration enforces it
2. ~~Are MVP success thresholds calibrated correctly?~~ → Patch 1: null-baseline relative replaces raw thresholds
3. ~~Should adaptation be in initial ABM?~~ → Patch 4: NO, fixed strategies v1
4. Is ABIDES the right substrate? → DEFERRED to /pdca plan Phase 0 1-week spike
5. ~~Is 6 weeks to G1 achievable?~~ → Patch 5: NO, 9 weeks (3 + 6)

---

**END Architecture v1.1**
