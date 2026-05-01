# P0 Pre-Commit Amendment #001 — Friction Stress Variant

**Amendment date**: 2026-05-01
**Original precommit**: `experiments/p0/precommit.md`
**Trigger**: Advisor 권고 (non-blocking item) + R26 LIVE postmortem evidence
**Effect**: P0.3 friction model + P0.4 baseline + 후속 priority 모두 적용

---

## Background

R26 LIVE postmortem (memory: `r26_postmortem_20260501.md`)에서:
- BT-modeled friction 0.10-0.13% RT 가정
- LIVE 14d -12.86% (-0.92%/day) — friction이 BT보다 meaningfully worse
- Catastrophe 회피하지 못함

→ Mandate § 0.4의 "0.13-0.20% RT range" 명시는 단일 median 시나리오로 검증하면 R26 패턴 재발 위험.

---

## Amendment

### A1. Friction model parameterization (D3 deliverable update)

`scripts/validators/friction_model.py`는 다음 3 시나리오를 모두 지원:

| Scenario | Total RT | Taker fee | Slippage | Use case |
|----------|---------|-----------|----------|----------|
| `optimistic` | 0.10% | 0.045% × 2 = 0.09% | 0.005% × 2 = 0.01% | 이상적 maker rebate 시나리오 |
| `realistic` (default) | 0.16% | 0.045% × 2 = 0.09% | 0.035% × 2 = 0.07% | mandate § 0.4 median |
| `stress` | 0.20% | 0.045% × 2 = 0.09% | 0.055% × 2 = 0.11% | mandate § 0.4 high + R26 postmortem evidence |

추가 funding cost는 양 시나리오 모두 8h 단위로 별도 차감/적립.

### A2. Baseline 평가 의무 (D5 deliverable update)

`experiments/p0_baselines/result.md`는 3 시나리오 6-criteria 결과 모두 포함:
- Buy-and-hold × {realistic, stress}
- 1× constant long perp + funding × {realistic, stress}
- Random entry × {realistic, stress}

`optimistic` 시나리오는 reference로만 (PASS 기준 아님).

### A3. 후속 priority 적용

P2-P6 모든 hypothesis test 시:
- Primary 평가: `realistic` (0.16% RT)
- 의무 stress test: `stress` (0.20% RT)
- 둘 다 6-criteria 통과해야 PASS
- Realistic PASS + Stress FAIL → PARTIAL (deploy 금지)
- Stress test 결과 명시 누락 → 자동 FAIL

### A4. 6-Criteria gate 변경 없음

Mandate § 0.5의 6-criteria 자체는 동일. 단지 friction scenario별로 별도 평가.

---

## Anti-Fishing Locks

1. ❌ Stress 시나리오 PASS 못 하는 경우 friction model parameter 변경 금지 (예: slippage 0.04%로 낮추기)
2. ❌ Stress 시나리오 결과 부분 보고 금지 — 모든 priority result에 명시
3. ❌ Stress test를 P5/P6에서만 적용 금지 (P2부터 의무)

---

## Rationale Summary

R26 + C1 두 번 같은 패턴 (BT 양수 → LIVE 음수)의 재발 방지. Mandate § 0.5의 strategy_deploy_5gate_protocol 정신과 정합. Realistic + stress 두 frictions 모두 PASS = 더 conservative deploy gate.

---

**Pre-commit signed (amendment)**: Claude Code agent, 2026-05-01.
원본 precommit.md와 함께 P0 entire scope에 적용. 본 amendment 변경 시 새 amendment doc 작성.
