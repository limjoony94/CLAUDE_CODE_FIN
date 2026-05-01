# Design v0.8 Patch Draft (PENDING E1+E4 RESULTS)

**Status**: DRAFT — to be finalized when E1 + E4 complete and decision tree applied.
**Trigger**: G2 abandon trigger fired (Gini 0.191 < 0.3 at default rate).

---

## Scenario A: Frozen-window concentration works

### v0.8 patch text (if E4 shows Gini > 0.5 among incumbents)

| Version | Date | Changes |
|---------|------|---------|
| 0.8 | 2026-05-01 | **G2 criterion redefined per advisor amendment + E1/E4 diagnostic**: G2 evaluation now uses **FROZEN-WINDOW SCOPE** (Gini at end of T_extract phase among agents present at T_open boundary), NOT all-T_10k scope. Rationale: open-system admissions dilute incumbent wealth at default rate 1/600 (E1: Gini 0.19 at default, vs Gini X.XX at rate=0). Frozen-window scope matches G3 substrate-extraction window where admissions are by-design disabled. |

### Section 2 Gate G2 amendment

```diff
 ### Gate G2 — Wealth-Concentration Validity

 **Question**: Does ABM exhibit non-trivial wealth concentration / whale emergence under realistic parameters?

 **Pass criterion**:
-- After T = 10,000 sim-bars with capital-weighted order sizing, top-5% of agents hold > 40% of total capital (Gini > 0.5)
-- Top-5% emerge from initially uniform wealth distribution (NOT seeded as whales)
-- Distribution rank-stability: top-5% at T=10,000 overlaps top-5% at T=5,000 by ≥ 50% (not pure noise)
+- **AMENDED 2026-05-01 v0.8**: After T_open + T_extract sim-bars (default 5000+5000=10000),
+  among agents present at T_open boundary (= incumbents through frozen extraction window):
+  - Top-5% of incumbents hold > 40% of incumbent capital (Gini > 0.5)
+  - Top-5% emerge from initially uniform wealth distribution (NOT seeded as whales)
+  - Distribution rank-stability: top-5% at T_end overlaps top-5% at T_open by ≥ 50%
+- **Original "all agents" Gini documented as descriptive metric, NOT pass criterion**
+  (admissions dilute by construction; G3 uses frozen-window so G2 should match)
```

### Section 7 Open-System addition

```diff
 **Open-system / fixed-target collision resolution (v1.1 patch 3)**:
 ...
+
+**G2 evaluation scope (v0.8 patch)**:
+G2 wealth-concentration validity is evaluated within the FROZEN-ADMISSION window only,
+matching the G3 substrate-extraction scope. Open-phase admissions provide population
+diversity (heterogeneous strategies, free entry honoring user requirement) but
+statistical concentration analysis filters to incumbents-at-T_open. This avoids the
+arithmetic dilution problem where 1000 admissions × 100 wealth = 96,300 dwarfs
+15 incumbents × 1000 = 15,000 (incumbents 14% of pool by construction before PnL).
```

---

## Scenario B: Even rate=0 doesn't reach 0.5 (architecture issue)

### v0.8 patch text (if E1 rate=0 < 0.5 but E4 still informative)

```
| 0.8 | 2026-05-01 | G2 architecture revision: wealth-weighted sizing alone insufficient
   to produce Gini > 0.5 from initially uniform population (E1: max Gini Y.YY at rate=0).
   Three options under consideration:
   (a) Pareto initial wealth distribution (concentration seeded, then amplified)
   (b) Gini > 0.3 as v1 acceptance threshold (relaxed but documented)
   (c) Skill-driven strategy heterogeneity (e.g., trader winrate seed) instead of
       sizing-driven concentration
   Decision pending user input. |
```

---

## Scenario C: Total falsification (even E2/E3 don't help)

### v0.8 patch text (if E1+E2+E3 all fail)

```
| 0.8 | 2026-05-01 | ABM v1 architecture HYPOTHESIS FALSIFIED. Wealth-weighted sizing
   does not generate concentration under any tested parameter (E1+E2+E3 all <0.3).
   Options for v2: (a) skill-differential mechanism, (b) external wealth shock events,
   (c) abandon as research methodology contribution only (no Path B).
   T-G3, T-G4, T-G5 deferred indefinitely. |
```

---

## Decision flow

E1 + E4 results → script `g2_decision_tree.py` produces verdict → match scenario A/B/C
→ apply corresponding patch → commit as v0.8.
