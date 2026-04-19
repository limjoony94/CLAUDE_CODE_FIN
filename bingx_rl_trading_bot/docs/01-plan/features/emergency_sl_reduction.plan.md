# Plan: Emergency SL Reduction

> **Feature**: emergency_sl_reduction
> **Date**: 2026-04-19
> **Phase**: Plan (compact)
> **Trigger**: Trail/BE 재설계 dead end 확정 후 다음 축 탐색

---

## 1. Background

Emergency SL은 fractal SL의 2차 보호 장치:
- Fractal SL: 구조 기반 first stop (-0.7~-1%)
- Emergency SL: hard stop at -3% (flash crash 대비)

**관찰**: 현재 clean BT에서 emergency 발동 **0회** (fractal SL이 먼저 잡음). 하지만:
- `bt_live_gap_20260419`: 2026-04-12 live LONG -7.89% (fractal SL이 실제 체결 안 됨, 큰 slippage)
- Emergency 3% 축소 시 live에서 tail 단축 가능

---

## 2. Hypotheses

| H | 내용 |
|---|------|
| H1 | Emergency SL 축소(3→2.5→2.0→1.5)로 MDD 감소 | BT에서 |
| H2 | Clean BT에선 0회 trigger 유지 (fractal이 우선) → PnL 변화 없음 | 중립 예상 |
| H3 | Slip_med에서도 emergency trigger 비율 낮음 → 효과 제한적 | 중립 예상 |

---

## 3. 검증 매트릭스

Emergency SL 스윕: 1.5, 2.0, 2.5, 3.0% × 2 combos × 2 modes = 16 runs.

추가: emergency trigger count 측정 (reason 분포)

---

## 4. 예상 결과
- Clean: 변화 거의 없음 (fractal 우선)
- Slip: 미미한 개선 또는 중립
- **Negative result 가능성 높음** — fractal SL이 거의 모든 손실 케이스 처리
- 그러나 live-specific tail risk(slippage beyond fractal) 대비 가치 있을 수도

---

## 5. Success / GO 조건

Simplified 5-flag:
1. wf_clean_pass 5/5
2. wf_slip_pass ≥ 4/5 (borderline tolerable)
3. MDD 감소 ≥ 5%
4. PnL 감소 ≤ 2%
5. rollback_ready

5/5 GO. Core fail 시 STOP.

---

## 6. 구현
스크립트 `scripts/analysis/emergency_sl_reduction_study.py`: intrabar_parity engine 재사용 + emergency_sl_pct monkey patch.
