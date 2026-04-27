# Cross-Strategy Lessons (C1 + M1) — INDEX

> **Date**: 2026-04-28
> **Purpose**: 두 negative result에서 cross-cutting 패턴 3가지를 한 페이지로 식별. 새 콘텐츠 X — 기존 lesson/postmortem/report에 link만.
> **Origin**: advisor 권고 (Path B 재계획). M1-A 폐기 직후, 다음 spec 시작 전.

---

## Pattern 1 — BT-LIVE friction gap

**Statement**: BT가 slippage 0 또는 매우 낙관적으로 가정하면 LIVE friction이 strategy edge를 지움. C1은 BT 후 LIVE에서 발견 (-$623 cost), M1은 BT 단계에서 friction inject 후 발견 (capital risk 0).

**Evidence**:
- C1: `docs/04-report/c1_breakout_postmortem_20260427.md` §5 (BT theoretical edge +0.83%/day vs LIVE friction floor -0.5%/trade)
- M1: `claudedocs/m1_friction_model.md` (사전 등록), `docs/04-report/m1_scalping_v1_negative_result_20260427.md` §3.1 (gross -66.85% over 718d 사전 noticed in BT)

**Operational rule**: 모든 BT는 friction 사전 등록 + inject. → `STANDARD_RESEARCH_PROTOCOL.md` Phase 2.5 Gate 3.

---

## Pattern 2 — Edge vs noise (random baseline 의무)

**Statement**: Strategy edge는 random baseline 대비 **반드시** 증명되어야 한다. Filter는 directional alpha를 만들거나 / volatility-rich 시점을 선택하거나 둘 중 하나여야 의미 있음. 둘 다 못하면 noise 변형.

**Evidence**:
- C1: random/B&H baseline이 LIVE 운영 중에도 측정 안 됨 → cohort delta로만 평가 → 폐기 늦어짐. `lessons_distribution_check_20260427.md`
- M1: Phase 2.7 random baseline = M1-A보다 MFE P50 +0.10pp **higher**. Filter가 anti-selective. `docs/04-report/m1_scalping_v1_negative_result_20260427.md` §3.4
- Memory: `lessons_distribution_check_20260427.md`, `lessons_process_audit_20260425.md` (control group 의무)

**Operational rule**: Phase 2.5 Gates 4 (baseline 사전 등록) + Gate 6 (random ≥ candidate면 STOP). BT 후가 아닌 BT 직전. → `STANDARD_RESEARCH_PROTOCOL.md` Phase 2.5 Gates 4, 6.

---

## Pattern 3 — Fix-impulse on negative data (process bug)

**Statement**: 진단 데이터가 negative 나올 때마다 자동으로 "단일 변수 조정으로 해결" 충동 발생. 충동 시점 합리적으로 보이지만 random baseline / EV calc로 검증 시 noise 변형으로 판명. C1 spec-tuning 함정 패턴이 다른 strategy에서도 재현됨.

**Evidence**:
- C1: max_sl 3.3→4.5 변경 시 WF 2/5 통과를 "moderate robust"로 reframe. `lessons_process_audit_20260425.md`
- M1: 단일 세션에서 4회 fix-impulse 발생 (15m role 변형 / Trail K / activation-trail / SL widening). 매번 advisor가 정정. `lessons_fix_impulse_pattern_20260427.md`

**Operational rule**:
- 충동 발견 → stop → advisor 호출 + EV back-of-envelope + random baseline 비교 = fix 코드 전 의무
- Spec evolution log entry는 advisor 검토 후만 추가
- → `STANDARD_RESEARCH_PROTOCOL.md` Phase 2.5 운영 원칙 섹션

---

## How this index is used

새 strategy plan 시작 시:
1. 본 INDEX를 먼저 읽고 plan §1 Background에 "3 patterns 인지" 명시
2. `STANDARD_RESEARCH_PROTOCOL.md` Phase 2.5 Gates 1~6 적용
3. 각 gate fail 시 본 INDEX의 해당 pattern으로 referrer

본 문서는 **INDEX**. 새 lesson 발견 시 별도 memory file에 작성하고 본 INDEX에 link만 추가. 본문 padding 금지.
