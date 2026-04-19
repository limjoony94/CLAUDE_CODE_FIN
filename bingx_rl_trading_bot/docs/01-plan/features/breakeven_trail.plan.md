# Plan: Breakeven Trail — Net-Loss Trail 회피

> **Feature**: breakeven_trail
> **Date**: 2026-04-19
> **Phase**: Plan
> **Core idea**: Trail이 fee+slippage 차감 후 **net-loss가 되는 영역**에서 발동하지 않도록 제어

---

## 1. Background

### 관찰 데이터 (baseline slip_med, 332일)
사전 진단으로 밝혀진 구조적 비효율:

| Trail exit | 건수 | 비율 | 평균 PnL | 합계 |
|------------|------|------|----------|------|
| Net **LOSS** | **630** | **66.0%** | **-0.183%** | **-115.04pp** |
| Net profit | 324 | 34.0% | +0.892% | +289.09pp |
| **Trail net** | 954 | | | +174.05pp |

### 손실 집중의 수학
현재 trail 수식:
```python
realized_pnl = max(0, best_pnl - trail_dist_pct)
exit_price = entry * (1 + realized_pnl/100)
```

Trail이 이론상 "본절 청산(realized=0)"을 해도, 실제 순손실:
- Entry slippage: **0.05%**
- Exit trail slippage: **0.05%**
- Fee RT: **0.10%**
- **Total: -0.20% per trail-at-breakeven trade**

Net loss trail의 p10~p75가 모두 -0.2% 부근에 밀집 (630건 대부분).

### 핵심 문제
Trail trigger 조건(`drawdown >= trail_dist_pct`)이 충족되면 **무조건 발동**:
- best_pnl이 trail_dist_pct보다 작으면 realized_pnl=0으로 clamp
- 하지만 exchange 체결 시 fee+slip으로 실손실 발생
- Trail은 원래 **profit-taking tool**인데 **loss-taking trigger**로도 작동

---

## 2. Goal

Trail 청산이 발생하는 경우 **항상 순이익 이상(net ≥ 0)** 을 보장하는 수식 도입:
```python
if (best_pnl - trail_dist_pct) < BREAKEVEN_BUFFER:
    return None  # hold — fractal SL or emergency가 downside 담당
```

**Target**:
- Loss trail 630건 제거 → 이론 상한 +115pp 개선
- 현실적 목표: baseline clean +170 → +180%+ AND slip_med +46 → +70%+

---

## 3. Hypotheses

| H | 내용 | 검증 지표 |
|---|------|----------|
| **H1** | BUFFER > 0 시 slip_med PnL 개선 | slip_med total PnL |
| **H2** | BUFFER의 최적값은 fee+slip 근사값(약 0.20%) | BUFFER 스윕 peak |
| **H3** | 회피된 loss trail 중 상당수는 SL 터치 전 반등 → profit exit 전환 | Trail→SL 전환 비율 |
| **H4** | WF 5-fold 안정성 유지 (fold 2도 개선) | WF clean + slip 5/5 |
| **H5** | train_not_degraded (train 구간 악화 없음) | train 구간 ±2pp |
| **H6** | Candidate_C + breakeven = 최강 조합 (synergistic) | 두 개선 중첩 효과 |
| **H7** | BUFFER 민감도 낮음 (0.15~0.30 범위에서 유사 성능) | 고원 분석 |

---

## 4. Success Criteria (8-flag GO, `train_not_degraded` 포함)

### Core flags (1개라도 실패 시 STOP)
1. **wf_clean_pass** — WF 5/5 (clean BT)
2. **wf_slip_pass** — WF 5/5 (slip_med BT)
3. **tw_pass** — 3-way train/val/test 모두 양수 (slip_med)
4. **train_not_degraded** — train PnL ≥ baseline train − 2pp
5. **pnl_improvement** — slip_med PnL ≥ baseline slip_med + 5pp

### Non-core
6. **ratio_ok** — PnL/MDD ≥ baseline × 1.0
7. **buffer_stable** — BUFFER 0.15/0.20/0.25 셋 중 ≥2개가 baseline보다 +PnL
8. **rollback_ready** — config-driven 단일 값 변경

8/8 PASS 시 GO (production 변경 검토). 한 개 core 실패 시 STOP.

---

## 5. Methodology

### 5.1 BUFFER 스윕 (5 values)
| BUFFER | 의미 |
|--------|------|
| 0.00 | 기존 (baseline 동작) |
| 0.10 | fee만 커버 |
| 0.20 | fee + round-trip slip (추천) |
| 0.30 | 여유 50% |
| 0.40 | 보수적 상한 |

### 5.2 Combos (2)
- `baseline`: (3.3, 2.5, 192)
- `candidate_C`: (4.0, 2.5, 192)

### 5.3 실행 매트릭스 (20 runs)
2 combos × 5 BUFFER × 2 modes (clean + slip_med) = 20 runs.
Plus WF/3-way on selected top performers.

### 5.4 비교 기준점
기존 baseline slip_med:
- PnL +46.09%, MDD 18.78, WR 30.2%, trail exits 954건 중 630 loss

### 5.5 재사용 인프라
- `scripts/analysis/intrabar_trail_impact.py` (엔진)
- `scripts/analysis/c1_intrabar_parity.py` (slippage 주입)
- `scripts/analysis/candidate_c_validation.py` (9-flag 평가)
- 신규 함수: `check_exit_breakeven()` — trail 로직에 projected < BUFFER 가드 추가

---

## 6. Implementation Plan

### 신규 스크립트: `scripts/analysis/breakeven_trail_study.py` (~350 lines)
- Copy `intrabar_trail_impact.py`의 `_check_exit_5m` 로직
- BUFFER 파라미터 추가
- Trail trigger 전에 `projected < BUFFER` 체크 삽입
- BUFFER 스윕 실행 + 결과 JSON

### Production 적용 시 (GO 판정 후)
- `config/c1_breakout_config.yaml`에 `trail_breakeven_buffer: 0.20` 추가
- `scripts/production/c1_breakout/signals.py::check_exit`에 guard 추가
- 봇 재시작 + 30일 LIVE 관찰

### 예상 실행 시간
- 20 runs × 1초 = 20초
- Top-2 WF/3-way/neighborhood: +30초
- 총 **1분 이내**

---

## 7. Non-Goals

- `trail_K`, `max_sl_atr`, `max_hold_bars` 조정 (이미 sl_trail_tuning/intrabar_parity/candidate_c에서 검증)
- Emergency SL 변경 (별개)
- Intrabar tick modeling (intrabar_parity에서 다룸)

---

## 8. Rollback

Config `trail_breakeven_buffer: 0.20 → 0.0` 단일 변경으로 기존 동작 복원.

---

## 9. Risks

| Risk | Mitigation |
|------|-----------|
| Loss trail 회피가 오히려 **SL 터치까지 hold**로 악화 | BUFFER 스윕으로 0.0~0.4 전 범위 평가 |
| Candidate_C + breakeven의 synergy가 없을 수도 | 2 combos 독립 평가 |
| WF slip fold 2가 여전히 음수 유지 | fold-level 세분 평가 |
| BUFFER calibration을 실측 slip과 분리 | 실측 median 기반 BUFFER=0.20 권장, 민감도 |
| Production 변경 전 strict 검증 | 8/8 GO + 30일 LIVE 재확인 |

---

## 10. Expected Outcomes

### 낙관 시나리오 (loss trail 완전 회피, SL 전환 없음)
- BT clean: +170 → +285%
- slip_med: +46 → +161%
- WF slip: 5/5 (fold 2 -9.03 → 개선 가능)

### 중립 시나리오 (절반 회복, 절반 SL)
- slip_med: +46 → +80% (+34pp)
- WF slip: 4/5 (fold 2 여전 -3~-5)
- Baseline 대비 우위 but synergy 제한

### 비관 시나리오 (hold 후 SL 집중, 악화)
- slip_med: +46 → +20%
- STOP 판정

---

## 11. Reference

- 선행 관찰: baseline slip_med trail loss 630건 × -0.183% 분석
- intrabar_parity: slippage 수치 출처
- candidate_c_validation: 9-flag GO protocol
- fold2_regime_analysis: fold 2 약점은 candidate_C 고유 아님 → breakeven으로도 개선 가능성
- 표준 규칙: `memory/research_protocol_overfit_guards.md`
