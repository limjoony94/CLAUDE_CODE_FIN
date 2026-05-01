# P3b Skip Rationale (advisor 2026-05-02)

**Date**: 2026-05-02
**Trigger**: P3a closure 0/28 PASS, M006 borderline only
**Decision**: SKIP P3b. Anti-fishing § 0.7 + advisor lock.

---

## M006 Borderline Analysis (Substance Audit)

M006 (xs_momentum_30d top-1 coin long) realistic:

| Criterion | Threshold | Value | Substance? |
|-----------|-----------|-------|------------|
| mean | ≥ 0.10%/d | +0.168% ✅ | Borderline trigger |
| p_beats | ≥ 0.55 (borderline gate) | 0.585 ✅ | Borderline trigger |
| max_dd | ≥ -5% (P3) | **-109%** ❌ | **CAPITAL RUIN** |
| sharpe | ≥ 2.0 (P3 aggregated) | 0.72 ❌ | Worse than B&H 0.89 |

**Stress scenario**: p_beats drops to 0.521 (< 0.55) — not even borderline at higher friction.

**Key insight**: max_dd -109% additive over 540d means the strategy lost MORE than full capital during regime shifts. Borderline mean+p_beats are technical pass markers but the strategy is ruin-bound.

---

## Why P3b Cell-Conditional Would Be Cherry-Pick

P3b would search for regime cells where M006 max_dd improves to acceptable levels:
- Option (a): Some regime slice has lower max_dd → engineering around result, anti-fishing § 0.1 violation
- Option (b): No regime improves → effort with predetermined outcome

Either way, P3b on M006 fails the discipline test.

**Asness GMO replications + R13 8-coin advisor evidence** (2026-04-30): multi-coin XS momentum has well-studied concentration risk. Top-1 selection makes it worse. No regime gating fixes this fundamentally.

---

## Anti-Fishing Compliance

| Rule | Compliance |
|------|-----------|
| No post-hoc threshold tuning | ✅ Lock M006 config maintained |
| No engineering-around-result | ✅ P3b skipped (advisor lock) |
| No selection-bias retrofitting | ✅ Borderline on tech, FAIL on substance — reported |
| Honest closure | ✅ 0/28 PASS reported |

---

## Decision

**P3b SKIP, proceed to envelope decision tree.**

Per advisor: if user asks "deploy M006 anyway under strict-backtest directive" — answer **NO** with max_dd -109% evidence. Strict-backtest doesn't override basic gate FAIL.

Recorded: 2026-05-02. M006 file closed at "borderline on technicality, FAIL on substance".
