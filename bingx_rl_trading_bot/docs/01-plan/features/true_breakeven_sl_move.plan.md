# Plan: True Breakeven SL Move

> **Feature**: true_breakeven_sl_move
> **Date**: 2026-04-19
> **Phase**: Plan
> **User request**: "손익분기 넘긴 시점부터는 SL을 손익분기점으로 옮기는 방안"
> **Distinct from**: breakeven_trail (BUFFER 방식, 이미 기각) — **완전히 다른 메커니즘**

---

## 1. Background

### 두 메커니즘 비교
| 축 | breakeven_trail (기각) | **true_breakeven_sl_move (본 PDCA)** |
|---|---|---|
| Trail | 발동 차단 (BUFFER 미달 시 hold) | **유지** |
| SL | fractal 그대로 | **Entry 위치로 이동** |
| Tail risk | fractal SL(~-1%) 노출 | **entry + slip(-0.2%)로 제한** |
| Profit upside | trail 여전히 작동 | **trail 여전히 작동** |
| 실증 결과 | ❌ MDD 5.6배, -149pp | ? (본 PDCA) |

**핵심 insight**: SL을 entry로 tighten하면 tail risk는 trail의 `max(0, projected)` clamp와 동등한 수준(-0.2%)으로 제한되면서, trail의 profit 보호는 **그대로 유지**.

### 전통적 기법
Professional trader 사이 널리 알려진 "breakeven stop" 전통 기법. BT 환경에서 C1 전략에 적용 시 효과 검증.

---

## 2. Goal

수익 중(best_pnl > ACTIVATION_PCT) 시 **SL을 entry 가격으로 tighten**하여:
- Fractal SL의 tail risk 제거
- Trail 메커니즘 온전히 보존
- 기대 효과: MDD 감소, WR 유지 또는 상승

**Target 수치 (slip_med 기준)**:
- 낙관: +46.09% → +55% 이상 (+9pp 이상)
- 보수: +46% 유지 + MDD 18.78 → 15 미만

---

## 3. Hypotheses

| H | 내용 | 검증 |
|---|------|------|
| **H1** | BE SL move 적용 시 slip_med MDD 감소 | MDD 감소폭 |
| **H2** | slip_med PnL이 baseline 이상 유지 | PnL |
| **H3** | ACTIVATION_PCT 최적 = 0.3 ~ 0.5% (fee+slip+여유) | 스윕 |
| **H4** | WF 5/5 slip 유지 (fractal SL 대체 후에도) | WF |
| **H5** | Whipsaw 증가 (BE SL에 걸리는 win rate 감소) — 예상 trade-off | WR 감소폭 |
| **H6** | candidate_C + BE SL = 최강 synergy | cand_C 성능 |
| **H7** | fold 2(2025-08 저변동성) 개선 | fold 2 PnL |

---

## 4. Success Criteria (8-flag GO)

### Core (5)
1. **wf_clean_pass** — 5/5
2. **wf_slip_pass** — 5/5
3. **tw_pass** — 3-way 전 양수 (slip)
4. **train_not_degraded** — train ≥ baseline train − 2pp
5. **pnl_improvement** — slip_med PnL ≥ baseline + 3pp OR MDD ≥ -20%

### Non-core (3)
6. **ratio_ok** — PnL/MDD ≥ baseline × 1.05
7. **activation_stable** — ACTIVATION 0.2/0.3/0.4 셋 중 ≥2 baseline 상회
8. **rollback_ready** — config-driven 단일 값

8/8 GO. Core 1개라도 fail 시 STOP.

---

## 5. Methodology

### 5.1 메커니즘
```python
# pos['sl']은 initial fractal SL로 설정됨
# Position update loop에서 매 bar:
if not pos.get('be_activated', False):
    if best_pnl > ACTIVATION_PCT:
        if direction == 'LONG':
            pos['sl'] = max(pos['sl'], pos['ep'])  # entry로 tighten
        else:
            pos['sl'] = min(pos['sl'], pos['ep'])
        pos['be_activated'] = True
# check_exit는 기존과 동일 (updated sl_price 사용)
```

단방향 tightening (한 번 활성화 후 추가 이동 없음).

### 5.2 ACTIVATION_PCT 스윕
| 값 | 의미 |
|----|------|
| 0.10 | fee만 커버 (공격적) |
| 0.20 | fee + entry slip |
| 0.30 | fee + slip + 약간 여유 (권장) |
| 0.50 | 여유 크게 |
| 1.00 | 큰 이익 달성 후만 |

5 values × 2 combos × 2 modes = 20 runs.

### 5.3 재사용
- `intrabar_trail_impact.py` 엔진 + monkey patch
- `c1_intrabar_parity.py` slippage 주입
- `candidate_c_validation.py` 8-flag 평가 패턴

### 5.4 구현
신규 스크립트: `scripts/analysis/true_breakeven_sl_move_study.py` (~350줄)
- Position state에 `be_activated` flag 추가
- Bar loop에서 best_pnl 체크 후 sl tighten
- 나머지 check_exit 경로는 그대로

---

## 6. Expected Outcomes

### 낙관 (기대)
- Tail risk 제거로 slip_med MDD 18.78 → 12 근처
- PnL은 유사 유지 (+46 → +50)
- WF slip 4/5 → 5/5 (fold 2 개선)

### 중립
- Whipsaw 증가로 WR 1~3pp 감소, PnL 변화 미미
- MDD 약간 감소 (18 → 16)

### 비관
- BE SL이 whipsaw로 조기 청산 → PnL 감소
- Trail의 profit trail이 drawdown으로 작동 못 함

---

## 7. Rollback / Production 전환

### GO 시
- Config `breakeven_sl_move.enabled: true`, `activation_pct: 0.30`
- Bot code: position update에 SL tighten 로직 추가
- 봇 재시작 + 30일 LIVE 관찰

### Rollback
- `enabled: false` 즉시 비활성화

---

## 8. Non-Goals

- Trail 메커니즘 변경 (이미 검증됨)
- BE 이후 추가 SL 이동 (continuous SL trail)
- Regime filter 결합

---

## 9. Risks

| Risk | Mitigation |
|------|-----------|
| BE SL whipsaw 증가로 WR 감소 | ACTIVATION 스윕으로 trade-off 지점 탐색 |
| Entry slip으로 BE 체결가 ≠ entry → 실손 -0.2% | 이미 trail max(0) clamp와 동일 수준 |
| 작은 이익에서 early exit | ACTIVATION 값으로 조절 |
| Fractal SL 정보 상실 (언제 활성화돼야?) | Pre-activation은 fractal, post는 entry (monotonic tighten) |

---

## 10. Reference

- 사용자 원 요청: "SL을 손익분기점으로 옮기는 방안"
- 대비 실패 사례: `memory/breakeven_trail_20260419.md` (BUFFER 방식, 7/7 REVERSED)
- BT 신뢰 검증: `memory/lookahead_audit_trail_20260419.md` (NO BIAS)
- 재사용 엔진: `scripts/analysis/c1_intrabar_parity.py`, `intrabar_trail_impact.py`
- 평가 패턴: `scripts/analysis/candidate_c_validation.py`
