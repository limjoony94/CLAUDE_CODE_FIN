# Plan: Candidate_C (4.0, 2.5, 192) 전용 Validation

> **Feature**: candidate_c_validation
> **Date**: 2026-04-19
> **Phase**: Plan
> **Target**: `(max_sl_atr=4.0, trail_K=2.5, max_hold_bars=192)` vs baseline `(3.3, 2.5, 192)`
> **단일 파라미터 변화**: `max_sl_atr` 3.3 → 4.0

---

## 1. Background

### 발견 경로
| 연구 | Candidate_C 위치 | 비고 |
|------|-----------------|------|
| extended_param_grid (1D) | `max_sl_atr=4.0` PnL **+192.8%** | 1D grid 최대 |
| sl_trail_tuning (3D, clean BT) | train top-9/10 (tied) | val rerank에서 drop → top-3에 부재 |
| intrabar_parity (5m+slippage) | **slip PnL +63.06%, ratio 4.42 → #1** | baseline 2위, candidate_A/B 하위 |

즉 candidate_C는 3D clean BT에서는 **val rerank 필터에 의해 누락**됐지만, slippage-adjusted BT 평가에서는 baseline보다 명확히 우위. sl_trail_tuning에서 채택된 `val rerank`가 slippage 환경에서 역효과였을 수 있음을 시사.

### 왜 재검증인가
- **단일 파라미터 변경**(`max_sl_atr` 3.3→4.0)이므로 **단순하고 설명 가능**
- **Trail/timeout 변경 없음** → sl_trail_tuning 과적합 위험(3D 동시 튜닝)과 성격 다름
- **독립적 정당화 필요**: val top-10 재평가가 아닌 **전용 GO 조건 8개**로 엄격 검증
- 만약 GO라면 **production 교체** — 단, 30일 LIVE fix-후 샘플 확보 후 최종 확정

---

## 2. Goal

Candidate_C `(4.0, 2.5, 192)`가 **slippage-aware 환경에서 통계적으로 baseline을 능가**함을 증명. 실패 시 baseline 유지.

---

## 3. Hypotheses

| H | 내용 | 검증 방법 |
|---|------|----------|
| **H1** | Clean BT WF 5/5 OOS 양수 유지 | expanding WF |
| **H2** | Slippage-adjusted BT (5m+slip)에서도 WF 5/5 양수 | c1_intrabar_parity 엔진 |
| **H3** | 3-way split train/val/test 모두 양수, test PnL ≥ baseline_test − 5pp | 동일 split |
| **H4** | Neighborhood ±1 step (최대 6개)에서 ≥75% positive | `(3.6, 2.2, 144)`~`(4.5, 2.8, 288)` 영역 |
| **H5** | MC direction p < 0.01 (999 sims) | sign randomization |
| **H6** | Bootstrap PnL 95% CI 하한 > 0 | stationary block 1000 sims |
| **H7** | `train_not_degraded`: candidate train ≥ baseline_train − 2pp | slippage BT 기준 |
| **H8** | Slippage sensitivity: low/med/high 세 구간에서 **모두** baseline 대비 PnL/MDD 우위 | ±50% slippage 스윕 |

---

## 4. Success Criteria (GO 조건 9개, H8 포함)

모두 충족해야 GO. 핵심(H1, H2, H3, H7, H8) 중 1개라도 실패 시 STOP.

1. **wf_clean_pass**: H1 5/5
2. **wf_slip_pass**: H2 5/5
3. **tw_pass**: H3 train/val/test 양수
4. **test_not_worse**: H3 test ≥ baseline_test − 5pp (clean), ≥ baseline_test_slip − 5pp (slip)
5. **nbr_pass**: H4 ≥ 75% positive
6. **mc_pass**: H5 p < 0.01
7. **ci_pass**: H6 CI lower > 0
8. **train_not_degraded**: H7 slip 기준
9. **slip_sensitivity**: H8 세 slip 시나리오 모두 우위

9/9 PASS 시 **GO — 30일 LIVE fix-후 재확인 후 production 전환 권장**.

---

## 5. Methodology

### 5.1 데이터
- `btc_5m_270days_reclassified.csv` (332일, 2025-05-05 ~ 2026-04-03)
- Train 60% / Val 20% / Test 20% 분할 (sl_trail_tuning과 동일)

### 5.2 Slippage 시나리오 (H8)
| 시나리오 | entry / exit_sl / exit_trail / exit_emg / exit_timeout |
|---------|--------------------------------------------------------|
| low (best) | 0.025 / 0.075 / 0.025 / 0.15 / 0.025 |
| med (default) | 0.05 / 0.15 / 0.05 / 0.30 / 0.05 |
| high (pre-fix) | 0.10 / 0.30 / 0.10 / 0.60 / 0.10 |

### 5.3 비교 Combos
- `baseline`: (3.3, 2.5, 192)
- `candidate_C`: (4.0, 2.5, 192)
- 둘 다 동일 slippage 적용하여 **차이만 max_sl_atr**에서 발생

### 5.4 재사용 인프라
- `scripts/analysis/c1_refined_validation.py` (WF, 3-way)
- `scripts/analysis/c1_refined_bootstrap_mdd.py` (stationary bootstrap)
- `scripts/analysis/c1_intrabar_parity.py` (slippage engine + 5m traversal)
- `scripts/analysis/intrabar_trail_impact.py` (indicators + backtest engine)

---

## 6. Implementation Plan

### 신규 스크립트: `scripts/analysis/candidate_c_validation.py`
- Baseline vs Candidate_C 동시 평가
- 3 slippage 시나리오 × 2 combos = 6 조건 + clean BT baseline 2개 = 8 run
- 각 조건에 WF + 3-way + bootstrap + MC + neighborhood 수행
- 9-flag GO 판정 자동화

### 예상 실행 시간
- Clean BT 2 run + slip 6 run = 8 × ~1초 = 8초
- WF, MC, bootstrap, neighborhood = 수십 초
- 총 **1~2분**

---

## 7. Rollback / Production 전환 프로토콜

### GO 시
1. `config/c1_breakout_config.yaml` `max_sl_atr: 3.3 → 4.0` **단일 변경**
2. 봇 재시작 (state 유지)
3. CLAUDE.md 업데이트
4. **30일 fix-후 LIVE 샘플** 대기 후 최종 검증 (필요 시 3.3 복귀)

### STOP 시
- baseline 유지. 문서화 후 종료.

### Rollback
- `max_sl_atr: 4.0 → 3.3` 단일 변경. 코드 변경 없음.

---

## 8. Non-Goals

- `trail_K`, `max_hold_bars` 조정 (이미 sl_trail_tuning에서 dead/sharp peak 확정)
- Emergency SL 변경 (별개 PDCA 후보)
- Multi-asset 확장
- Track B LIVE tick polling

---

## 9. Risks

| 리스크 | 완화 |
|--------|------|
| MDD가 clean BT에서 증가 (4.0×ATR이 3.3 대비 손실 확대) | PnL/MDD ratio로 평가 |
| Slippage 시나리오 세팅 자체의 calibration 오류 | 3 시나리오 sensitivity, 모두 우위여야 GO |
| Single-combo 검증의 overfit 위험 | Neighborhood(±1) robustness 추가 |
| LIVE ≠ BT (구조적 #22 MARKET slippage) | GO 시에도 30일 LIVE 재확인 필수 |

---

## 10. Reference

- sl_trail_tuning 결과: `memory/sl_trail_tuning_20260419.md`, `docs/04-report/sl_trail_tuning.report.md`
- intrabar_parity 발견: `memory/intrabar_parity_20260419.md`, `docs/04-report/intrabar_parity.report.md`
- 표준 규칙: `memory/research_protocol_overfit_guards.md` (8-flag 표준)
- 선행 1D grid: `results/extended_param_grid.json`
- Baseline 기준: `results/c1_refined_variants.json`
