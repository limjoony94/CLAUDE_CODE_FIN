# Plan: F v3 — LIMIT Close with MARKET Fallback (DRAFT)

> **Feature**: f_v3_limit_close
> **Date**: 2026-04-25
> **Phase**: Plan (DRAFT — 실구현 보류)
> **Scope**: F v2 cycle check_exit의 MARKET close를 LIMIT 주문 + timeout fallback으로 대체하여 execution slippage 축소
> **기반**: F v2 Canary Trade #43 (slippage -0.638%) — execution floor 구조적 발견

---

## 1. Background

### F v2 Canary n=1 관찰 (2026-04-24)

Trade #43 (SHORT 77693.3 → 78189.3):
- F v2 cycle check_exit → TRAIL_TP 정상 발동 ✅ (설계 의도 달성)
- 이론 exit_price: 77693.3 (theoretical break-even)
- 실제 MARKET fill: 78189.3
- **Slippage: -0.638%** (3x 환산 -1.91pp per trade)

### Slippage 분해 (추정)

| 요소 | 추정 기여 | 개선 가능성 |
|---|---|---|
| 이론 exit_price vs bar close actual | ~0.2% | LIMIT으로 해결 (BT 가정가에 주문) |
| Bar close → 주문 지연 (1-2s) | ~0.1% | 구조적 |
| Spread + market impact | ~0.02~0.05% | 구조적 |
| 일시 liquidity thin | ~0.3% | LIMIT으로 완화 가능 (체결 안 되면 wait) |
| **합계** | **~0.6%** | |

### 목표

F v2의 structural timing gap (matched 58%) 해결은 달성됨. 나머지 **execution floor**를 LIMIT으로 축소.

**정량 목표**: Trail exit의 adverse slippage 평균 **0.638% → 0.15% 이내**.

---

## 2. Hypotheses

| H | 가설 | 검증 방법 |
|---|---|---|
| **H1** | Trail trigger 시 LIMIT @ theoretical exit_price 배치 + 60s 대기 시 80%+ fill 성공 | Canary paper test + fill rate 측정 |
| **H2** | Fill 성공 시 slippage < 0.1% (LIMIT price 정확) | 체결가 vs LIMIT price 비교 |
| **H3** | Fill 실패(60s timeout) → MARKET fallback 시 기존 F v2와 동등 slippage (~0.6%) | Timeout 통계 + fallback slippage 측정 |
| **H4** | LIMIT 대기 60s 동안 가격이 추가 adverse 이동 시 MARKET fallback의 추가 손실 < +0.1% per trade | Timeout 구간 가격 이동 측정 |

---

## 3. Success Criteria (GO 조건 7개)

1. **slippage_reduction**: Trail exit slippage 평균 < 0.15% (F v2 0.638% 대비 **75% 축소**)
2. **fill_rate**: LIMIT 60s 내 fill 성공률 ≥ 80%
3. **timeout_penalty**: Timeout fallback의 추가 손실 ≤ +0.1% per trade (F v2 대비)
4. **edge_case_safety**: Fast market move (intrabar ≥ 1%) 시 MARKET fallback으로 손실 확대 ≤ -0.5%
5. **rollback_ready**: config flag `f_v3_limit_close.enabled=false`로 즉시 F v2로 원복
6. **pytest_coverage**: LIMIT 주문 + timeout + fallback 전 path pytest
7. **BT_identity**: BT 변화 없음 (순수 execution layer 개선)

6/7 이상 GO. #1 (slippage reduction) 실패 시 무조건 STOP.

---

## 4. Design

### 4.1 현재 F v2 flow (process_candles → _do_close)

```python
ex = self.signal.check_exit(...)  # returns {'reason': 'TRAIL_TP', 'exit_price': theoretical}
if ex:
    self._do_close(i, ex)  # MARKET close inside

def _do_close(idx, exit_signal):
    # ... bookkeeping ...
    actual_fill = self._exchange_close(d)  # MARKET order
```

### 4.2 F v3 변경안

```python
def _do_close(idx, exit_signal):
    f_v3 = cfg.get('f_v3_limit_close', {}).get('enabled', False)
    if f_v3 and exit_signal['reason'] == 'TRAIL_TP':
        # 1. LIMIT order at theoretical exit_price
        target_price = exit_signal['exit_price']
        limit_id = self._place_limit_close(d, qty, target_price)
        # 2. Poll for fill, up to timeout_s
        timeout_s = cfg['f_v3_limit_close'].get('timeout_s', 60)
        fill_result = self._wait_for_fill(limit_id, timeout_s)
        # 3. If not filled, cancel + MARKET fallback
        if not fill_result['filled']:
            self.exchange.cancel_order(limit_id, symbol)
            actual_fill = self._exchange_close(d)  # MARKET fallback
            logger.warning(f"LIMIT timeout @ ${target_price:.1f} → MARKET fill ${actual_fill:.1f}")
        else:
            actual_fill = fill_result['avg_price']
            logger.info(f"LIMIT fill @ ${actual_fill:.1f} (target ${target_price:.1f})")
    else:
        # F v2 or legacy: MARKET close
        actual_fill = self._exchange_close(d)
    # ... rest of bookkeeping ...
```

### 4.3 LIMIT price 전략

**Option A (conservative)**: LIMIT price = `theoretical exit_price`
- Fill 시 slippage 0 (정확)
- Fast market move 시 fill 실패율 ↑

**Option B (aggressive)**: LIMIT price = `current_close × (1 ± spread_buffer)`
- spread_buffer: 0.05% (1 tick × 10)
- Fill 성공률 ↑, slippage ≤ spread_buffer
- BT와 소량 차이

**Option C (hybrid)**: 첫 30s는 Option A, 그 후 30s는 Option B
- 이중 timeout 복잡도 ↑

**기본**: **Option A** (BT parity 최우선). timeout fallback이 안전망.

### 4.4 Config

```yaml
strategy:
  # ... F v2 유지 ...
  f_v3_limit_close:
    enabled: false
    timeout_s: 60            # LIMIT 대기 시간
    # fill 실패 시 MARKET fallback
```

### 4.5 Partial fill 처리

LIMIT partial fill (예: 50% fill, 50% open) 시:
- timeout 시 나머지만 MARKET fallback
- 두 체결가 weighted average로 PnL 계산

---

## 5. Implementation Plan

### Phase 1: Trade #44 결과 대기
- #44 slippage < 0.3% → **Plan 보류** (reference 유지), F v2 지속
- #44 slippage > 0.3% → **Phase 2 착수** 트리거

### Phase 2: 코드 구현 (3~4일)
1. bot.py `_do_close`에 F v3 branch 추가
2. 신규 helper: `_place_limit_close`, `_wait_for_fill`
3. Partial fill 처리 로직
4. Pytest (test_f_v3_limit_close.py, ~15 케이스)
5. Paper test (testnet 1-2일)

### Phase 3: Canary (14일)
- config enable=true + 봇 재시작
- 2-4 trades 후 GO 조건 평가
- LIMIT fill rate, slippage 분포 측정

---

## 6. Non-Goals

- Entry MARKET을 LIMIT으로 변경 (별도 PDCA 고려)
- SL/Emergency를 LIMIT으로 (SL은 crash safety 우선 → MARKET 유지)
- Spread prediction 모델 (범위 외)

---

## 7. Risks

| 리스크 | 완화 |
|---|---|
| LIMIT fill 실패율 높음 (>30%) | Fallback MARKET으로 F v2와 동등 성과 보장 |
| Fast market move 시 timeout 동안 adverse 확대 | timeout_s 짧게 (60s) + Option B 고려 |
| Partial fill 복잡도 | weighted avg PnL + 잔여 MARKET |
| cancel API 실패 → 중복 주문 | BingX cancel 재시도 로직 (3회) |
| Cycle timing (15m) vs LIMIT timeout 충돌 | 다음 cycle 전 fill/fallback 완료 필수 |

---

## 8. Rollback

- config `f_v3_limit_close.enabled: false` + 봇 재시작 → F v2 MARKET 복귀
- LIMIT 미체결 주문 존재 시 다음 cycle에서 cancel (sync logic 재사용)

---

## 9. Relationship

| PDCA | 관계 |
|---|---|
| F v2 (f_v2_cycle_exit) | F v3는 F v2 위에 execution 개선. F v2 enabled=true 유지. |
| F v1 (activation_gated_trail) | 이미 기각. F v3와 무관. |
| intrabar_parity | 보완 관계. BT model을 현실화하여 F v3 효과 역검증 가능. |

---

## 10. Reference

- F v2 Trade #43 실측: `results/c1_breakout_state.json` trade #43
- F v2 Canary 현황: `logs/c1_breakout.log`
- `bingx_rl_trading_bot/scripts/production/c1_breakout/bot.py:856` `_do_close`
- `bingx_rl_trading_bot/scripts/production/c1_breakout/signals.py:163` check_exit trail
- `memory/post_fix_legacy_gap_20260424.md` "sequencing floor accepted" 개념
- `memory/f_v1_canary_rollback_20260424.md` F v1 실패 교훈

---

## 11. Decision Matrix (Trade #44 결과별)

| #44 slippage | 판단 | 액션 |
|---|---|---|
| < 0.1% | F v2 변동성 범위 — #43 이상치 | F v3 보류, 30일 샘플 확대 |
| 0.1~0.3% | Moderate — 축소 가치 있음 | Phase 2 착수 검토 (우선순위 中) |
| **0.3~1.0%** | **구조적 — 해결 필요** | **Phase 2 즉시 착수** |
| > 1.0% | Severe — 긴급 | Phase 2 즉시 + 긴급 rollback 준비 |
| TRAIL_TP 미발동 (SL만) | F v2 trail 문제 | 별도 원인 조사 |

---

## 12. Open Questions

- BingX LIMIT 주문이 reduceOnly와 호환되는가? (API 스펙 확인 필요)
- Partial fill 시 BingX의 평균 체결가 반환 방식? (`avgPrice` 필드 신뢰성)
- Cycle 15m 이내 LIMIT 완료 보장 가능? (timeout 60s면 14분 여유)
- Fast move (intrabar > 2%) 시 MARKET fallback의 예상 slippage?
