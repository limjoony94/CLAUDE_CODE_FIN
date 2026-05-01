# P3a Closure — MAP-Elites Aggregate-Only

**Date**: 2026-05-02 (P3 Day 1)
**Status**: ❌ **0/28 PASS, 1 borderline (M006)**
**Methodology**: Option γ aggregate-only, Choice (a) accept proxy v2 falsification
**Raw**: `experiments/p3/results_raw.json`

---

## Verdict

| Outcome | Count | Mechanisms |
|---------|-------|-----------|
| PASS | 0 | — |
| Borderline (mean>0 AND p_beats>0.55) | 1 | M006 (realistic only) |
| FAIL evaluated | 14 | M004, M006_stress, M007×2, M008×2, M009×2, M010×2, M012×2 |
| VACUOUS_FAIL | 13 (counted twice for both scenarios) | M001×2, M002×2, M003×2, M005×2, M011×2, M013×2, M014×2 |

**0 mechanisms PASS 6/6 at P3_AGGREGATED thresholds**. Friction-floor evidence count: **31 → 45 mechanisms** (P2 4 + P3 14, double-counting OK in evaluation count).

---

## Frequency Scan Results

| Mech | Family | Status | Freq/d | Verdict |
|------|--------|--------|--------|---------|
| M001 | reversion | active | 0.402 | VACUOUS |
| M002 | reversion | active | 0.496 | VACUOUS |
| M003 | reversion | active | 0.402 | VACUOUS (M001 same) |
| M004 | reversion | active | 1.501 | OK |
| M005 | momentum | active | 0.057 | VACUOUS |
| M006 | momentum | active | 1.088 | OK |
| M007 | breakout | control_null | 2.197 | OK |
| M008 | breakout | control_null | 0.858 | OK |
| M009 | pattern | active | 2.161 | OK |
| M010 | pattern | active | 2.298 | OK |
| M011 | regime | active | 0.031 | VACUOUS |
| M012 | regime | active | 3.624 | OK |
| M013 | cross_section | active | 0.255 | VACUOUS |
| M014 | cross_section | active | 0.000 | VACUOUS (dispersion threshold too strict) |

7/14 mechanisms VACUOUS at locked configs. Per anti-fishing rule, NO threshold loosening.

---

## Evaluated 6-Criteria Results (7 mechanisms × 2 scenarios = 14 evals)

### Force-Flow Reversal Family (P2 carryover confirmation)

| Mech | Scenario | Mean (%/d) | p_beats | MaxDD (%) | Sharpe | PASS |
|------|----------|-----------|---------|-----------|--------|------|
| M004 | realistic | -0.167 | 0.015 | -96.4 | -1.76 | 0/6 ❌ |
| M004 | stress | -0.227 | 0.004 | -127.8 | -2.37 | 0/6 ❌ |

### Momentum + Cross-Section

| Mech | Scenario | Mean (%/d) | p_beats | MaxDD (%) | Sharpe | PASS |
|------|----------|-----------|---------|-----------|--------|------|
| M006 | realistic | **+0.168** ✅ | **0.585** ✅ | -109.0 ❌ | 0.72 ❌ | 2/6 (BORDERLINE) |
| M006 | stress | +0.128 ✅ | 0.521 ❌ | -114.1 ❌ | 0.51 ❌ | 1/6 ❌ |

### Pattern Family

| Mech | Scenario | Mean (%/d) | p_beats | MaxDD (%) | Sharpe | PASS |
|------|----------|-----------|---------|-----------|--------|------|
| M009 | realistic | -0.219 | 0.141 | -185.0 | -0.69 | 0/6 ❌ |
| M009 | stress | -0.305 | 0.088 | -199.0 | -0.93 | 0/6 ❌ |
| M010 | realistic | -0.740 | 0.005 | -503.9 | -2.02 | 0/6 ❌ |
| M010 | stress | -0.832 | 0.002 | -545.4 | -2.18 | 0/6 ❌ |

### Regime Family

| Mech | Scenario | Mean (%/d) | p_beats | MaxDD (%) | Sharpe | PASS |
|------|----------|-----------|---------|-----------|--------|------|
| M012 | realistic | -0.405 | 0.024 | -301.9 | -1.42 | 0/6 ❌ |
| M012 | stress | -0.550 | 0.006 | -366.1 | -1.78 | 0/6 ❌ |

### Breakout (Control Null — Expected FAIL)

| Mech | Scenario | Mean (%/d) | p_beats | MaxDD (%) | Sharpe | Validator |
|------|----------|-----------|---------|-----------|--------|-----------|
| M007 | realistic | **-0.357** | 0.000 | -217.9 | -3.84 | ✅ Expected FAIL confirmed |
| M007 | stress | -0.445 | 0.000 | -262.5 | -4.59 | ✅ Expected FAIL |
| M008 | realistic | -0.159 | 0.006 | -88.1 | -3.40 | ✅ Expected FAIL |
| M008 | stress | -0.194 | 0.002 | -106.1 | -3.85 | ✅ Expected FAIL |

**Validator sanity check passed**: control_null mechanisms FAIL as expected, no anomaly escalation needed.

---

## M006 Borderline Analysis

M006 (xs_momentum_30d, multi-asset 1d, top-1 coin long):
- ✅ mean +0.168%/d (> P3 target 0.10%) — only mechanism > 0
- ✅ p_beats 0.585 (just above 0.55 borderline threshold, below 0.70 PASS)
- ❌ max_dd -109% (capital ruin during regime shifts)
- ❌ sharpe 0.72 (BTC has 0.89, M006 is WORSE despite higher mean)
- ❌ stress: p_beats drops to 0.521 (< 0.55) — not borderline at stress

**Interpretation**:
1. Top-1 coin XS momentum captures crypto regime trend (works during bull)
2. Catastrophic drawdown during regime shifts (top coin rotates fast)
3. Volatility absorbs all "edge" — Sharpe < B&H
4. Single-config Option α: no parameter tuning to optimize Sharpe (anti-fishing)
5. Memory/legacy lineage: R13 8-coin equal-weight (advisor 2026-04-30) showed similar friction-floor at multi-coin XS — top-1 likely worse due to concentration

**Per advisor framework**: M006 qualifies for P3b cell-conditional analysis trigger (mean>0 AND p_beats>0.55, realistic only).

---

## Anti-Fishing Compliance

✅ Option γ aggregate-only (no cell breakdown without trigger)
✅ Choice (a) accept proxy v2 falsification (no new variants)
✅ Locked configs only (no parameter sweep)
✅ All 28 evaluations reported (no PASS-only highlight, no selection bias)
✅ Vacuous → FAIL counted (no threshold loosening)
✅ Stress scenario obligatory both reported
✅ Sealed boundary asserted (no leak)
✅ M007/M008 control_null expected FAIL confirmed (validator sanity OK)

---

## Decision Tree (P3a Closure)

Per `experiments/p3/precommit.md`:

| Trigger | Action |
|---------|--------|
| ≥1 mech 6/6 PASS | → Advance P4 |
| 1-3 borderline | → P3b cell-conditional on those |
| 0 PASS + 0 borderline | → Envelope decision tree (A/B/C) |

**This run**: 0 PASS + 1 borderline (M006). Trigger P3b OR escalate.

**M006 P3b consideration**:
- Cell-conditional may improve max_dd via regime gating (low-vol or trending regimes)
- BUT: multi-asset XS momentum is well-studied (Asness, GMO, etc.); regime gating helps but unlikely to lift to 6/6 PASS
- Stress fails p_beats (0.521 < 0.55) — at higher friction, edge collapses

**Recommendation**: P3b on M006 conditional on advisor approval. If skip, proceed to envelope decision.

---

## Friction-Floor Evidence Update

Cumulative across legacy + this project:
- 27 mechanisms × 5 substrates (legacy 2026-04-30) → 0 deployable (R5 carry $49/yr only)
- + P2 4 mechanisms × 2 scenarios = 8 evals → 0 deployable
- + P3 14 mechanisms × 2 scenarios = 28 evals → 0 deployable, 1 borderline (M006)
- **Total: 45 mechanism configurations, 0 6/6 PASS**

This is no longer "prior" — it's an empirical regularity verified across multiple project lineages, substrates, and methodologies.

---

## Honest Closure Posture

Per mandate § 0.7 + § 10:
- Mandate v2 Phase A (target 0.10-0.20%/day) probability adjusted: 30-40% → **10-20%**
- M006 borderline is statistical noise compatible with friction-floor null
- Coinglass paid plan ROI strongly negative (Phase A signal absent)
- Phase B activation timeline: 60-90d forward collection still on track per cron (verified ✅)
- Funding Arb baseline (R5, $49/yr) remains only verified-deployable

**Next**: advisor() escalation for envelope decision (A/B/C) per precommit.

---

## Closure Output

- ✅ `experiments/p3/precommit.md` (lock)
- ✅ `experiments/p3/results_raw.json` (28 eval raw)
- ✅ `experiments/p3/result.md` (this — closure)
- Status: **P3a CLOSED, awaiting advisor on P3b vs envelope decision**
