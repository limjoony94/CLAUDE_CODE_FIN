# Analysis: Candidate_C Validation

> **Feature**: candidate_c_validation
> **Date**: 2026-04-19
> **Phase**: Check
> **Match Rate**: **95%** (Design ↔ 구현)
> **Outcome**: **STOP** — core flag `wf_slip_pass` fail (4/5) + `mc_pass` borderline fail (p=0.013)
> **Status**: **후속 조건부 관찰 가치 높음** — candidate_C가 모든 slippage 시나리오에서 baseline 압도하지만, fold-2 regime-dependency로 엄격 기준 미달

---

## 1. Executive Summary

Candidate_C `(max_sl_atr=4.0, trail_K=2.5, max_hold_bars=192)`를 baseline `(3.3, 2.5, 192)`와 **단일 파라미터만 변경**(max_sl_atr 3.3→4.0)한 엄격 비교 검증. 9-flag GO protocol 적용.

**결과**: 7/9 PASS, 2/9 FAIL (core 1 + non-core 1). `wf_slip_pass` core flag가 fold 2(2025-08 구간)에서 -9.03pp 음수로 4/5에 그쳐 STOP.

**그러나 Candidate_C의 우위는 명백**:
- Clean BT 5/5 ✓
- 3-way split 전체 양수 ✓
- Neighborhood 6/6 positive ✓ (매우 강한 robustness)
- **3 slippage 시나리오 모두에서 baseline 압도** ✓
- Bootstrap PnL 95% CI [+11.50, +117.67] 하한 양수 ✓

STOP 판정은 **엄격 프로토콜의 타당한 작동**이며, candidate_C의 가치를 부정하지 않음. Report에서 **조건부 재평가 후보**로 영구 기록.

---

## 2. Gap Analysis (Design ↔ 구현)

### Match Rate: 95%

#### ✅ Matched (Critical 요소 전부)
- COMBOS / SLIP_SCENARIOS 3개 수치 완전 일치
- 8 primary runs (2 combos × 4 conditions) 정확 실행
- **Default-arg binding bug fix 반영**: `run_bt_with_slippage(mode, slippage=...)` 명시 전달
- 9 GO flags 수식 Plan §4 / Design §3과 1:1 매핑
- CORE_FLAGS 5개 (wf_clean_pass/wf_slip_pass/tw_pass/train_not_degraded/slip_sensitivity) 정의
- Bootstrap PnL CI, MC direction, Neighborhood 6-axis 모두 정확 구현
- 3-way split 60/20/20 by entry_bar
- WF 5-fold time-partition on adjusted trades
- Verdict() 의 core flag 우선 검사 + 9/9 이중 gate 정확 작동

#### ⚠ Partial Gap (문서 수준, 기능 영향 없음)
1. **함수명 차이**: Design `bootstrap_pnl_on_trades` vs impl `bootstrap_pnl` (기능 동일)
2. **Import 축소**: Design 명시 import 일부 미사용 (자체 helper 구현)
3. **성능 예상치 과대**: Design ~45초 vs 실측 **0.6초** (precompute 캐시 효율)
4. **`max_hold_bars` dead-parameter 관찰**: 144/192/288 전부 동일 결과 — 이는 engine의 기존 구조적 사실(sl_trail_tuning에서 확인). Neighborhood 2개가 "중복 positive"로 +2 과대표시되었을 수 있으나 critical 결과에는 영향 없음

#### ❌ Critical Gap: **0건**

---

## 3. Research Findings

### 3.1 Primary Performance Matrix

| Scenario | baseline PnL/MDD | candidate_C PnL/MDD | Δ PnL | MDD 비교 | Ratio 승자 |
|----------|------------------|---------------------|-------|---------|-----------|
| clean | 169.55 / 5.38 | **192.76 / 5.20** | +23.21 | ▼ 낮음 | **cand** |
| slip_low | 105.84 / 7.10 | **120.95 / 8.10** | +15.11 | ▲ | **cand** (14.93 vs 14.91) |
| slip_med | 46.09 / 18.78 | **63.06 / 14.26** | +16.97 | **▼ 낮음** | **cand** (4.42 vs 2.45) |
| slip_high | -73.39 / 74.67 | **-52.73 / 64.41** | +20.66 | ▼ 낮음 | **cand** (-0.82 vs -0.98) |

**모든 시나리오에서 Candidate_C가 baseline 압도**. 특히 slip_med(실용 범위)에서 ratio 4.42 vs 2.45로 **80% 개선**.

### 3.2 WF Fold 분포 비교

**Clean BT** (candidate_C):
```
Fold 1: +22.40  ✓
Fold 2: +15.95  ✓
Fold 3: +58.08  ✓
Fold 4: +32.63  ✓
Fold 5: +63.69  ✓
→ 5/5 PASS
```

**Slip_med BT** (candidate_C):
```
Fold 1: +1.19   ✓ (barely)
Fold 2: -9.03   ✗ (FAIL — 2025-08 drawdown)
Fold 3: +33.56  ✓
Fold 4: +12.76  ✓
Fold 5: +24.59  ✓
→ 4/5 FAIL
```

**Fold 2 분석**: 2025-08 구간(첫 번째 test fold)에서 strategy가 이미 약한 성과(clean +15.95). Slippage 추가 시 beat fail. 이는 **regime-specific fragility** — 저변동성 or 특정 시장 조건에서 전략이 약한 약점. 단, fold 2만 음수이고 나머지 4개는 양수이므로 **구조적 실패 아닌 샘플 특수성** 가능성 높음.

### 3.3 Bootstrap CI & MC

| 지표 | 값 | 해석 |
|------|-----|------|
| Observed PnL (slip_med) | +63.06% | 평균 이상 성과 |
| Bootstrap 95% CI | [+11.50, +117.67] | 하한 양수, 중앙값 ~63 |
| MC direction p-value | 0.013 | 임계 0.01보다 0.3% 초과 |
| N_sims | 999 | |

**MC p=0.013**은 1.3% 확률로 candidate_C 수준 PnL이 random signal에서 나올 수 있음. Strict 0.01 기준 미달이나, 일반 0.05 기준에서는 강하게 유의. **Borderline**.

### 3.4 Neighborhood Robustness (6 축방향 이웃)

| Neighbor | PnL | MDD |
|----------|-----|-----|
| (3.6, 2.5, 192) | +57.91 | 14.74 |
| (4.5, 2.5, 192) | +65.59 | 12.27 |
| (4.0, 2.2, 192) | +36.43 | 27.61 |
| (4.0, 2.8, 192) | +60.30 | 16.97 |
| (4.0, 2.5, 144) | +63.06 | 14.26 |
| (4.0, 2.5, 288) | +63.06 | 14.26 |

**6/6 positive**, 평균 +57.74% / 평균 MDD 18.35. (4.0, 2.5, 192) 중심이 **parameter plateau의 중간** — 매우 안정된 영역.

단 `max_hold_bars` 이웃(144/288)은 dead parameter로 3개 동일 결과 → 실질 positive neighbors는 **4/4 non-dead** (여전히 100%).

### 3.5 3-way Split 비교

**Clean**:
| Split | baseline | candidate_C | Δ |
|-------|----------|-------------|---|
| Train (60%) | +94.03 | +96.44 | +2.41 |
| Val (20%) | +21.21 | +31.74 | +10.53 |
| Test (20%) | +54.30 | **+64.57** | +10.27 |

**Slip_med**:
| Split | baseline | candidate_C | Δ |
|-------|----------|-------------|---|
| Train | +21.17 | +25.71 | +4.54 |
| Val | +4.01 | +11.08 | +7.07 |
| Test | +20.91 | **+26.27** | +5.36 |

**모든 split에서 candidate_C가 양수 + baseline 대비 우위**. Test 구간 OOS 검증에서도 명확한 개선. 이는 overfit이 아닌 실제 edge일 가능성 시사.

---

## 4. 9-Flag GO Condition 최종 평가

| # | Flag | 결과 | 수치 | 비고 |
|---|------|------|------|------|
| 1 | wf_clean_pass | ✅ PASS | 5/5 | Clean BT 완벽 |
| 2 | **wf_slip_pass** | **❌ FAIL** | **4/5** | **CORE — fold 2 = -9.03** |
| 3 | tw_pass | ✅ PASS | train/val/test 모두 양수 | |
| 4 | test_not_worse | ✅ PASS | clean +64.57 vs req 49.30, slip +26.27 vs req 15.91 | |
| 5 | nbr_pass | ✅ PASS | **6/6** positive | 최고 성능 |
| 6 | mc_pass | ❌ FAIL | p=**0.013** > 0.01 | Borderline, non-core |
| 7 | ci_pass | ✅ PASS | Bootstrap CI [+11.50, +117.67] | 하한 양수 |
| 8 | train_not_degraded | ✅ PASS | +25.71 vs req +19.17 | |
| 9 | slip_sensitivity | ✅ PASS | 3/3 시나리오 cand 우위 | |

**Core flag(#1,#2,#3,#8,#9)**: 4/5 PASS (#2만 실패).
**전체**: **7/9 PASS**.

→ **Verdict: STOP** (core failure).

---

## 5. Interpretation

### 5.1 STOP의 정당성
- Plan §4에서 선언한 "9/9 + core 1개라도 실패 시 STOP" 이중 gate protocol
- `wf_slip_pass` core 실패 → 즉시 STOP. 사후 기준 완화는 selection-after-peek
- `mc_pass` borderline이나 규칙상 fail 처리

### 5.2 Candidate_C의 실제 edge 여부
- Clean WF 5/5, 3-way 전 양수, Bootstrap CI 양수, Neighborhood 6/6 → **강한 edge 증거**
- Slippage 시나리오 3/3 우위 → **현실 환경에서도 개선 신호**
- Fold 2 slip fail은 **특정 regime(2025-08) 약점**이지 구조적 전략 실패 아님
- MC p=0.013은 strict 기준 초과하나 "noise distribution 대비 유의" 해석 가능

### 5.3 왜 sl_trail_tuning에서 val rerank에 탈락했나
- sl_trail_tuning의 val rerank은 **clean BT val PnL 기준**
- candidate_C(4.0, 2.5, 192)의 val PnL +31.74 vs 탈락 기준 val top-3 = +33.88 (candidate_B group)
- 2.14pp 차이로 rerank에서 누락 → 그러나 slip 환경에서는 완전 반대 — baseline 대비 강함
- **clean val PnL 최적화의 함정** 재확인

### 5.4 Fold 2 약점 원인 추정
2025-08 BTC 시장:
- 저변동성 or 횡보 추정 (ATR 평균 낮음)
- 돌파 신호 빈도 낮음 + 체결 시 slippage 상대 비용 큼
- candidate_C의 넓은 SL(4.0×ATR)이 횡보에서 whipsaw 소모 가능

---

## 6. Recommended Action

### 즉시
1. **production 변경 없음** — baseline 유지
2. **Report 작성** 후 memory 영구화
3. **추가 검증 없이 30일 LIVE 관찰 대기**

### 단기 (2주 이내)
1. **Fold 2 원인 파고들기 (별개 연구)**:
   - 2025-08 시장 레짐 분석 (변동성, 돌파 빈도)
   - 해당 구간 단독 backtest로 candidate_C의 약점 정확히 특정
   - Regime filter (저변동성 회피) 추가 가치 검증
2. **30일 fix-후 LIVE 샘플 수집**: slippage median 실측 재calibration

### 중기 (1~2개월)
1. **Regime-conditional candidate_C PDCA**: 고변동성 레짐에서만 4.0 적용, 저변동성에서 3.3 유지하는 adaptive 변종
2. **Slippage-adjusted WF가 표준**: `research_protocol_overfit_guards.md`에 "slippage-BT에서도 WF 5/5 요구" 추가 권장

### 조건부 GO 트리거
다음 조건 충족 시 재검토 가치:
- 30일 LIVE 샘플에서 **WR ≥ 30%, 평균 PnL/trade ≥ baseline**
- 2025-08 fold 약점이 **다른 기간 샘플에 재현되지 않음**
- Slippage 실측이 현 `slip_med` 범위 이하로 확인

---

## 7. 방법론적 교훈

1. **Default-arg binding bug**: Python function signature의 default는 **정의 시점 객체 capture**. module-level 변수 교체로는 반영 안됨. 명시 전달 필수.
2. **Core flag의 가치**: 9 flag 중 5개 core로 지정하여 "비어있는 통과"(7/9)가 GO로 오해되지 않게 설계. 실제로 이번에 정확히 작동.
3. **MC borderline의 위험성**: p=0.013 vs strict 0.01. Plan 선언 시 flexibility 고려(e.g., "p<0.05 with 3+ corroborating flags") 검토 가치.
4. **OOS val rerank 함정 확인**: sl_trail_tuning에서 val PnL 2pp 차이로 탈락한 candidate_C가 slip 환경에서는 baseline보다 우위 → **clean val PnL 최적화는 slippage robustness를 보장하지 않음**.
5. **Fold 민감도 분석 가치**: 전체 PnL 우위이더라도 단일 fold 약점이 core 기준 실패를 초래. 이것이 파라미터 robustness 테스트의 의미.

---

## 8. Files Touched

- `scripts/analysis/candidate_c_validation.py` (NEW, ~400 lines)
- `results/candidate_c_validation_20260419_151610.json` (NEW, 결과)
- `docs/01-plan/features/candidate_c_validation.plan.md`
- `docs/02-design/features/candidate_c_validation.design.md`
- `docs/03-analysis/candidate_c_validation.analysis.md` (본 문서)

Production 변경 **0건**.

---

## 9. Reference

- Plan: `docs/01-plan/features/candidate_c_validation.plan.md`
- Design: `docs/02-design/features/candidate_c_validation.design.md`
- 선행 연구 1: `memory/sl_trail_tuning_20260419.md` (val rerank에서 탈락 경위)
- 선행 연구 2: `memory/intrabar_parity_20260419.md` (slippage 환경에서 #1 발견)
- 표준 규칙: `memory/research_protocol_overfit_guards.md`
- 재사용 엔진: `scripts/analysis/c1_intrabar_parity.py`, `intrabar_trail_impact.py`
