# G2 Final Verdict — Architecture Hypothesis Falsified

**Date**: 2026-05-01
**Status**: T-G2 BLOCKED. User-level decision required between options (a)/(b)/(c) below.

---

## TL;DR

ABM v1's wealth-weighted sizing mechanism produces only **+0.02 to +0.10 of Gini amplification per 5,000-10,000 sim-bars** under any tested admission rate. This is far short of the design's "Gini > 0.5" criterion and is closer to noise than to "whale emergence." The architecture v1.1 Section 6.2 hypothesis ("wealth-weighted order sizing produces emergent concentration") is **empirically falsified**.

---

## Evidence

### E1: admission rate sweep (10k bars each)

| Rate | n_alive | Initial Gini | Measured Gini @ T=10k | PnL amplification |
|------|---------|--------------|----------------------|-------------------|
| 1/600 (default) | 978 | 0.119 | 0.191 | **+0.071** |
| 1/3600 | 184 | 0.389 | 0.432 | **+0.043** |
| 1/36000 | 29 | 0.397 | 0.420 | **+0.023** |
| 0 (no admissions) | 15 | 0.000 | 0.096 | **+0.096** |

**Key finding (advisor amendment)**: The non-monotonic Gini-vs-rate pattern reflects **initial bimodal heterogeneity** between incumbents (1000 wealth) and admissions (100 wealth), NOT emergent concentration. Confirmed by computing initial Gini on `[1000]*15 + [100]*n_admissions` arrays — measured Gini ≈ initial Gini + small PnL noise.

### E4: frozen-window Gini (T_open=5000, T_extract=5000, rate=1/600)

| Metric | Value |
|--------|-------|
| Agents at T_open boundary | 500 (15 incumbents + 485 admissions) |
| Initial Gini at T_open | 0.206 |
| Gini at T=10k (after 5k frozen bars) | 0.288 |
| **PnL amplification over frozen window** | **+0.082** |
| Top-5% rank stability T_open → T_end | 0.72 (good — supports incumbent persistence) |

**Advisor prediction at start of E4**: "Gini at end-of-frozen ≈ 0.50 ± 0.05"
**Actual**: 0.288 — **significantly lower than predicted**, confirming weak amplification.

### Heterogeneity-preservation pattern (across all experiments)

PnL amplification is **bounded at +0.02 to +0.10** of Gini regardless of:
- Admission rate (E1: rates 0 to 1/600)
- Population size (15 to 978 agents)
- Time horizon (5k frozen bars in E4)
- Initial heterogeneity (uniform to strongly bimodal)

This is not noise — it's a structural ceiling on the wealth-weighted-sizing mechanism's ability to drive concentration. The mechanism preserves and weakly amplifies pre-existing inequality but does not generate new inequality at scale.

---

## Diagnostic interpretation

The architecture's mechanism: order_size = wealth_fraction × current_wealth → richer agents trade larger sizes → win/lose proportionally larger PnL → rich get richer (or poorer).

**Why this fails to produce strong concentration in practice:**

1. **Trades are roughly zero-sum in expectation**. A momentum agent placing a larger order doesn't have larger *edge per unit traded* — same expected PnL fraction per dollar traded.
2. **Friction (taker 0.05% / maker 0.02%) erodes wealth uniformly across agents**. Larger trades pay proportionally larger fees, neutralizing the wealth-scaling advantage.
3. **Strategy diversity within families is small**. Momentum N=3, N=5, N=10 don't differ enough in PnL to create separate wealth trajectories.
4. **Wealth volatility ≈ wealth_fraction × edge × sqrt(N_trades)**. With wealth_fraction=0.05 and edge ~bp-level, wealth std after 1000 trades is small fraction of initial. Insufficient time-horizon for compounding.

These are structural properties of the design, not parameter-tuning issues.

---

## What this means for downstream phases

- **T-G2 (current)**: cannot pass with current architecture. Pre-registered abandon trigger fired honestly.
- **T-G3 (substrate discovery)**: assumes whales exist to extract substrate from. With no whales, the inverse machinery from G1 has no concentrated targets to study. **G3 is structurally blocked** until a working concentration mechanism exists.
- **T-G4 (BingX L2 friction-pass)**: was always low-probability per advisor "Q4(b)+Q5(c) high-prior-of-failure". Now further blocked.
- **T-G5 (deploy)**: was always conditional on G4. Now further deferred.

The ABM v1's "open-system + wealth-weighted sizing → whale emergence" hypothesis is the load-bearing claim of the entire research project. With it falsified, Path B (deployable edge) is structurally blocked unless architecture changes.

---

## E2 + E3 results (option (b) executed per advisor 2026-05-01)

| Experiment | Setup | Initial Gini | Final Gini | Δ |
|------------|-------|--------------|------------|---|
| E2 (Pareto initial, wealth-weighted ON) | 15 agents, Pareto x10 (1063-14781), no admissions, 5k bars, 24,236 trades | 0.4389 | 0.4899 | **+0.051** |
| E3 (uniform initial, FIXED-SIZE trades) | 15 agents at 1000, fixed 0.001 BTC trades, no admissions, 5k bars | 0.0000 | 0.0007 | **+0.0007** |

**E2 verification of advisor's prediction**:
- Advisor predicted "~0.5" Gini outcome — actual 0.49 ≈ matches
- Per advisor decision-tree band classification (final < 0.55): **PRESERVES** (not amplifies)
- Mechanism does amplify weakly (+0.05 per 5k bars) but insufficient to cross 0.55 threshold

**E3 verification**:
- ZERO meaningful concentration without wealth-weighting
- Confirms skill differential alone does NOT drive whale emergence
- Wealth-weighted sizing is the load-bearing mechanism (such as it is)

**Advisor decision tree applied** (mechanical):
| E2 | E3 | Decision |
|----|----|----------|
| Preserves (final 0.49 in 0.45-0.55 band) | Doesn't concentrate (0.0007 < 0.1) | **(a) abandonment, v1 mechanisms structurally insufficient** |

**Implementation note**: First E2 attempt with Pareto x_min=100 had random agents below MIN_ORDER_SIZE wealth threshold → 0 trades, sim frozen. Re-ran with x10 scale (x_min=1000) so all agents above tradeable floor. The frozen-Sim case revealed an additional architectural limitation: extreme wealth heterogeneity is incompatible with uniform MIN_ORDER_SIZE — agents below ~$200 wealth can't trade.

**Caveat — (c) "seeded-Pareto v2" sub-variant remains potentially viable**:
- Modest positive amplification (+0.051/5k bars) suggests longer-horizon evaluation with seeded heterogeneity could reach Gini > 0.5
- Linear extrapolation: from 0.44 initial, +0.05/5k → cross 0.55 in ~10k more bars (2.5× longer than current G2 evaluation)
- This is NOT a refutation of mechanical verdict (a) — it's an additional design parameter user could explore in v2

---

## Three options (user-level decision per advisor)

### Option (a): Accept ABM v1 hypothesis falsified

- Document this verdict as Path A research methodology contribution
- T-G3, T-G4, T-G5 deferred indefinitely (acknowledged as v1 architectural limit)
- 6-month research project compresses to ~3 months of completed v1 + 3 months of v2 architecture or alternative direction
- Honest scientific outcome — the hypothesis was tested and failed
- Value: rigorous methodology + falsification evidence + reproducible scaffolding

### Option (b): Confirm with E2/E3 (Pareto initial + disable wealth-weighted sizing)

- E2 (Pareto initial wealth, no admissions, 5k bars): tests whether wealth-weighted sizing AMPLIFIES initial heterogeneity above noise floor
- E3 (uniform initial, fixed-size trades, no wealth-weighted sizing, 5k bars): tests whether ANY mechanism (skill differential alone) drives concentration
- ~14 min wall-clock for both
- Outcomes:
  - E2 amplifies heavily + E3 produces 0: confirms wealth-weighting works only with seeded heterogeneity → option (a) plus suggested v2 design "seeded Pareto"
  - E2 doesn't amplify + E3 produces 0: total falsification of all current mechanisms → option (a) more decisive
  - E3 produces concentration without sizing: skill-driven mechanism is the real driver → option (c) viable
- Strengthens scientific case for whichever conclusion follows

### Option (c): Pivot to v2 skill-driven concentration mechanism

- Add "skill" parameter per agent (drawn from a distribution): biases their PnL probabilistically, making strategy success heterogeneous across agents *within* the same family
- Combined with wealth-weighted sizing: skilled agents win more per trade, win bigger sizes, compound faster → emergent whales
- ~1 day implementation work + re-run G1 to verify anchors still recover families + re-run G2 with new mechanism
- Risk: introduces a parameter that has no obvious empirical anchor (what is "skill" in real markets?)
- Potential reward: working concentration mechanism unblocks G3 with a more realistic dynamics model

---

## Recommendation framing (NOT recommendation)

Per advisor: "deciding between (a)/(b)/(c) is a research-direction decision worth ~6 weeks of work. The user needs to make this call, not the autonomy mandate."

The autonomy mandate covers technical execution. (a)/(b)/(c) is the decision itself.
