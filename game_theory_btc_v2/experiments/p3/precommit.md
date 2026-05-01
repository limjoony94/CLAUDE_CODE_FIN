# P3 Pre-Commit — MAP-Elites Aggregate-Only First Pass (Option γ)

**Pre-commit date**: 2026-05-01
**Priority**: P3 (post-P2 closure FAIL)
**Mandate basis**: § P3 + advisor 2026-05-01 budget guidance (Option γ recommended)
**Authority**: Pre-committed BEFORE P3 strategy code. **Mutability**: changes require new amendment.

---

## P2 Inheritance + Choice (a) Accept Proxy Falsification

P2 0/8 PASS. Proxy v2 sum-of-z at threshold 5.5 → vacuous (M001-M003), basis fade catastrophic anti-edge (M004). Per advisor:
- **Choice (a) ACCEPTED**: force-flow proxy v2 design falsified. NO new proxy variants in P3 (anti-fishing).
- M001-M004 included in P3 as PHASE A reference (will likely re-FAIL aggregate; informative consistency).

---

## P3 Selection Rule = Option γ (Aggregate-Only)

Each mechanism evaluated **unconditioned** over full 540d free 1h window:
- 14 active mechanisms × 2 friction scenarios = **28 evaluations** total
- No cell-conditional regime breakdown (Option δ/ε deferred to P3b if any aggregate PASS)
- Single locked config per mechanism (mechanism_catalog.yaml `parameter_space_locked`)

This budget choice avoids 252-eval multiple-comparison explosion (Option ε) at cost of regime-specific niche detection (deferred to P3b conditional only on P3a aggregate PASS).

### P3a / P3b Two-Stage Logic
- **P3a (this priority)**: aggregate-only eval, 28 evaluations
- **P3b (conditional)**: IF ≥1 mechanism shows aggregate p_beats > 0.55 (borderline) AND mean > 0 → cell-conditional analysis on top-3 candidates only
- IF P3a 0 PASS + 0 borderline → close P3, proceed to envelope decision tree

---

## Mechanisms Tested in P3a (14 + 2 control)

### Active (12)
| Mech | Family | Direction | Scale | Forward |
|------|--------|-----------|-------|---------|
| M001 | reversion | long | 1h | 4h |
| M002 | reversion | short | 1h | 1h |
| M003 | reversion | long | 1m fwd | 15m |
| M004 | reversion | bidir | 1h | 4h |
| M005 | momentum | long | 1h | 4h |
| M006 | momentum | long | 1d | 1d |
| M009 | pattern | long | 5m | 24h |
| M010 | pattern | short | 15m | 24h |
| M011 | regime | long | 1h | 24h |
| M012 | regime | bidir | 1h | 8h |
| M013 | cross_section | long | 1d | 1d |
| M014 | cross_section | long | 1d | 1d |

### Control Null (2 — sanity check, expected FAIL)
| Mech | Family | Direction | Notes |
|------|--------|-----------|-------|
| M007 | breakout | bidir | EXPECTED FAIL per status=control_null_expected_fail |
| M008 | breakout | bidir | EXPECTED FAIL per status=control_null_expected_fail |

→ M007/M008 PASS = validator anomaly escalation (advisor sanity check).

---

## 6-Criteria Gate (P3_AGGREGATED priority)

Each mechanism × scenario evaluated against `priority="P3_AGGREGATED"`:
- target_daily ≥ 0.10%/d
- max_dd_floor ≥ -5%
- min_pos_rate ≥ 0.5
- min_p_beats ≥ 0.70 (vs B&H same window)
- min_sharpe ≥ 2.0
- p5 (bs lower CI) ≥ 0
- baseline mandatory (B&H same window)

**PASS** = realistic 6/6 AND stress 6/6.
**PARTIAL** = realistic 6/6 only.
**FAIL** = realistic < 6/6.
**VACUOUS_FAIL** = freq < min_event_freq_gate (per mechanism in catalog).

---

## Frequency Pre-Validation (R38 lesson)

Each mechanism × locked config 별 frequency 측정 first. Vacuous filter:
- M001/M002/M003: re-confirmed VACUOUS from P2 (no rerun needed)
- M004: confirmed not-vacuous (1.5/d)
- M005-M014: must measure

Vacuous → FAIL counted, NO threshold loosening.

---

## Anti-Fishing Locks

1. ❌ **No proxy v2 variants** (Choice (a) accepted)
2. ❌ **No threshold tuning** (catalog `parameter_space_locked` only)
3. ❌ **No M004 sign-flip** (advisor warning — basis_continuation 변형 금지)
4. ❌ **No regime cell breakdown without aggregate PASS** (P3a→P3b gate)
5. ❌ **No silent re-scope** (any added mechanism = new amendment)
6. ✅ **All 28 evaluations reported** (no selection bias, no PASS-only highlight)
7. ✅ **VACUOUS counted as FAIL** (no rerun with looser threshold)
8. ✅ **Stress mandatory**: PASS = realistic AND stress

---

## P3 Stopping Rule + Decision Tree

**P3 hard limit**: 5 days from entry.

**Outcomes**:
| P3a Outcome | Action |
|-------------|--------|
| ≥1 mech 6/6 aggregate PASS | Advance to P4 (Thompson sampling) on PASS subset |
| 1-3 mech borderline (mean>0 AND p_beats>0.55) | P3b cell-conditional analysis on those |
| 0 PASS + 0 borderline | **CLOSE P3**. Advisor escalation to envelope decision tree: (a) accept envelope falsified / (b) wait Phase B / (c) mandate revision |
| Validator anomaly (M007/M008 unexpected PASS) | Halt + advisor escalation |

---

## Envelope Decision Tree (Post-P3 if 0 PASS)

| Choice | Description | Implication |
|--------|-------------|-------------|
| **A. Accept envelope falsified** | Mandate v2 Phase A empty; deploy R5 carry only ($49/yr) | Honest closure per friction-floor evidence |
| **B. Wait Phase B** | Forward collector 60-90d → test H1/H6/H7-full | OI/L-S features may unlock signal |
| **C. Mandate revision** | Escalate user — fundamental rethink (Coinglass paid? new substrate?) | Major scope change |

P3 closure → advisor call → (A/B/C) decision per evidence.

---

## Closure Output (P3 Day-N)

`experiments/p3/result.md` 의무 포함:
- 28 evaluations 6-criteria table (4 already from P2 + 24 new)
- Frequency scan results (10 new mechanisms)
- M007/M008 control_null sanity check
- PASS/PARTIAL/FAIL/VACUOUS_FAIL count
- P3a→P3b decision (if borderline)
- Envelope decision tree triggered if 0 PASS
- Friction-floor evidence count update (currently 31 mechanisms × multiple substrates → 0 deployable except R5)

---

**Pre-commit signed**: Claude Code agent, 2026-05-01.
P3 entry IMMUTABLE. Variation requires amendment doc.
