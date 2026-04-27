# M1 Scalping — Friction Model (Pre-Registered)

> **Date**: 2026-04-27
> **Phase**: 0.3 (사전 등록)
> **Purpose**: BT-LIVE friction gap을 BT 단계에서 처음부터 inject. C1 lesson(BT theoretical slippage 0 가정 위험) 적용.

## 1. Components

| Component | Value | Source |
|-----------|-------|--------|
| Entry MARKET slippage | 0.05% | C1 LIVE 측정 평균 ~0.03% (보수적 +0.02% margin) |
| Exit MARKET slippage  | 0.05% | C1 LIVE TRAIL_TP MARKET ~-0.51% 측정값은 **이상치** (BUG#65 fix 후 재측정 대기). C1 baseline 평균 ~0.03~0.05%. |
| Round-trip taker fee  | 0.10% | BingX taker 0.05% × 2 (entry + exit) |
| **Total friction floor** | **−0.20% / trade** | sum |

## 2. BT 적용 방식

```python
def apply_friction(trade_pnl_pct, direction):
    """trade_pnl_pct (1x, sign-correct) → friction-adjusted return."""
    return trade_pnl_pct - 0.20  # always subtract 0.20% from gross PnL
```

**적용 위치**: `bt_run()` 매 trade 종료 시 gross 1x return에서 일괄 차감.

## 3. Stress 시나리오 (Phase 4 sensitivity 단계에서 측정)

| Scenario | Total floor | 적용 시점 |
|----------|-------------|-----------|
| BASE (위 값)  | −0.20% / trade | Phase 2 main BT |
| MED  | −0.30% / trade | Phase 4 sensitivity |
| HIGH | −0.50% / trade | Phase 4 sensitivity (C1 LIVE 실측 worst) |
| STRESS | −0.80% / trade | Phase 4 sensitivity (extreme adverse) |

## 4. Criterion Gate

거래당 평균 수익이 **friction floor 0.20% 초과** 해야 양수 (criterion 6).

- BT gross avg/trade ≤ 0.20% → 모든 시나리오 음수 → reject
- BT gross avg/trade 0.20~0.50% → BASE 통과, HIGH/STRESS 위험 → 보고
- BT gross avg/trade > 0.50% → BASE/MED/HIGH 모두 양수 (margin 충분)

## 5. Update Rule

LIVE 운영 후 30일 시점에 실측 slippage로 calibrate. 차이가 0.05pp 이상이면 BT re-run.

이 문서는 **사전 등록**. Phase 1+ 모든 BT는 이 friction model 적용. 변경 시 evolution log 추가.
