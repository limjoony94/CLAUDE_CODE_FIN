# Analysis: Intrabar Parity (Track A / Phase 1)

> **Feature**: intrabar_parity
> **Date**: 2026-04-19
> **Phase**: Check
> **Match Rate**: **92%** (Design ↔ 구현)
> **Research Outcome**: **STOP (Phase 1)** — core flag `intrabar_realism` borderline fail
> **Major Finding**: candidate_C (4.0, 2.5, 192)가 slippage-robust 1위 — sl_trail_tuning clean BT 평가의 blind spot 노출

---

## 1. Executive Summary

C1 Breakout v2.6 BT-LIVE 괴리의 구조적 원인을 **BT 쪽에서 intrabar 해상도 + slippage 모델 주입**으로 정량화하는 연구.

**결과**: 6 GO 조건 중 `rollback_ready` 1개만 PASS (5개 FAIL), 특히 **핵심 core `intrabar_realism` borderline 실패** (0.699 %/day vs 0.5 임계) → STOP. 그러나 예상 외 **큰 가치 있는 부수 발견**: candidate_C(4.0, 2.5, 192)가 slippage 적용 환경에서 PnL/MDD 1위 — sl_trail_tuning 연구가 clean BT로만 평가한 한계 노출.

---

## 2. Gap Analysis (Design ↔ 구현)

### Match Rate: 92% (12 matched + 3 partial + 0 critical)

#### ✅ Matched (12)
- Module 구조: `c1_intrabar_parity.py`가 `intrabar_trail_impact` module-level import
- 5-key SLIPPAGE schema
- `apply_slippage()`: reason→slip 매핑 + emergency priority preservation (SL overflow → EMERGENCY reclass)
- `run_bt_with_slippage()`: entry slip in raw, exit via apply_slippage
- `wf_on_adjusted_trades()`: 5-fold time partition (count, fold_pnls)
- `evaluate_go()`: 6 Phase-1 flags + 2 deferred (None) + diagnostics
- `verdict()`: core[1,2,3,8] + 7/8 rule
- 4 COMBOS params (baseline/A/B/C)
- Output schema, 실행 성공 (0.5s, JSON 생성)
- Critical gaps 4건 전부 해결 (run_wf_on_adj/CLEAN_TRAIN_PNL/emergency priority/Track B thread)

#### ⚠ Partial Gap (3)
1. **Slippage 보정치 하향** (Medium): Design §2.1 0.15/0.30/0.15/0.50 vs 구현 0.05/0.15/0.05/0.30. Fix-후 median 가정으로 절반 축소. Design 문서 미갱신.
2. **`intrabar_realism` 임계 재정의** (Medium): Design §10 `|gap_1x| ≤ 3pp` (window-based) → 구현 `|daily_gap_1x| ≤ 0.5%/day`. 데이터 범위 밖(OOS)으로 window 비교 불가하여 daily-rate fallback.
3. **`5m_slip` 모드 통합** (Low): Design은 4th 독립 모드 vs 구현은 `5m` + `slip` 서브키로 refactor. 같은 정보.

#### ❌ Critical Gap: 0건

---

## 3. Research Findings

### 3.1 Baseline (3.3, 2.5, 192) 성능 분해
| 모드 | PnL | MDD | WR | Trades | 비고 |
|------|-----|-----|-----|--------|------|
| **bar_close clean** | +169.55% | 5.38 | 36.6% | 1028 | sl_trail_tuning 기준 |
| **5m clean** | +165.68% | 5.55 | - | 1074 | **intrabar 효과 -3.87pp** |
| **intrabar clean** | +2.62% | 15.81 | - | 1237 | worst-case path, 비현실적 |
| **5m + slippage** | **+46.09%** | **18.78** | 30.2% | 1028 | fix-후 median slip |

### 3.2 Daily rate 비교
| 주체 | 기간 | 일일 평균 PnL (1x) |
|------|------|-----------|
| BT (5m+slip) | 332.8 days | **+0.139%/day** |
| LIVE (19-trade) | 7 days | **-0.56%/day** |
| **Daily gap** | — | **0.699%/day** (임계 ±0.5 초과) |

**해석**: BT가 여전히 LIVE보다 +0.7%/day 낙관. 단, LIVE 7일 샘플은 단기 음수 streak 가능성 (19 trades로 통계적 확정 불가). Fix-후 30일 샘플 확보 후 재검증 필수.

### 3.3 Combo 순위 변화 (핵심 부수 발견)
| Combo | Clean bar_close | 5m+slip PnL | 5m+slip MDD | Slip ratio |
|-------|-----------------|-------------|-------------|-----------|
| **candidate_C (4.0, 2.5, 192)** | +192.76% | **+63.06%** ⚡ | 14.26 | **4.42** |
| baseline (3.3, 2.5, 192) | +169.55% | +46.09% | 18.78 | 2.45 |
| candidate_B (4.5, 2.2, 144) | +181.92% | +39.48% | 25.96 | 1.52 |
| candidate_A (3.6, 2.2, 144) | +172.30% | +30.63% | 27.71 | 1.11 |

**핵심 통찰**:
- **candidate_C가 slippage-robust 1위** (+63.06%, ratio 4.42) — clean BT의 +192.76%(1위)와 순위 일관
- **candidate_A/B(val-optimized)는 slippage 환경에서 하위** — sl_trail_tuning의 val 재선정이 오히려 slippage 취약 조합 선택
- **baseline이 2위**로 여전히 안정 — STOP 판정이 slippage-robust 평가에서도 타당

### 3.4 WF Fold 분포 (baseline, 5m+slip)
```
Fold 1: -2.08%  (음수)
Fold 2: -11.53% (음수)
Fold 3: +34.79% (양수)
Fold 4: +5.45%  (양수)
Fold 5: +19.47% (양수)
```
3/5 positive. Fold 1-2 early window 음수 구간 있음 — 전략의 regime-dependence 시사.

### 3.5 `intrabar` 모드 해석
Bar low/high를 worst-case로 사용 시 clean PnL도 +2.62% 수준으로 급락. 이는 **매 bar의 worst tick을 항상 거쳐간다**는 가정의 비현실성. 실거래는 tick path 중 일부만 worst를 통과. 따라서 `intrabar` 모드는 **상한 stress test**이며 현실 추정용 아님 — Design §5.3의 "path-assumption risk" 정당.

---

## 4. GO 조건 최종 평가

| # | Flag | 결과 | 수치 | 임계 |
|---|------|------|------|------|
| 1 | **intrabar_realism** | ❌ FAIL | 0.699 %/day | ≤ 0.5 |
| 2 | baseline_preservation | ❌ FAIL | +46.09 | ≥ 150 (88%×170) |
| 3 | wf_pass | ❌ FAIL | 3/5 | 5/5 |
| 4 | ratio_ok | ❌ FAIL | 2.45 | ≥ 26.94 |
| 5 | track_b_cost | ⏭ Phase 2 | None | — |
| 6 | track_b_benefit | ⏭ Phase 2 | None | — |
| 7 | rollback_ready | ✅ PASS | True | — |
| 8 | train_not_degraded | ❌ FAIL | +21.17 | ≥ 90.07 |

**Core flag(#1,#2,#3,#8) 전부 FAIL** → **STOP**.

### 임계 실패 해석
- #1 `intrabar_realism` 0.699 vs 0.5: borderline. 29% 초과이지만 LIVE 샘플(19 trades/7일)의 통계적 신뢰도 낮음
- #2, #4: slippage 0.20%/trade × 1028 trades ≈ -200pp 감소가 기준선을 크게 낮춤
- #3: Fold 1-2 음수로 WF 깨짐
- #8: train 구간도 slippage 누적 영향으로 대폭 감소

---

## 5. Plan 대비 산출물 검증

| Plan 요소 | 상태 |
|-----------|------|
| H1 (intrabar+slip PnL 감소) | 확인 (+170→+46, -124pp) |
| H2 (intrabar-BT가 LIVE 재현) | **부분 확인** — 방향은 맞으나 gap 0.7%/day 잔존 |
| H3 (LIVE tick best_price) | **Phase 2 deferred** |
| H4 (sl_trail_tuning robustness) | **반전 발견** — candidate_C가 slippage-robust 1위 |
| H5 (Track B cost-benefit) | **Phase 2 deferred** |

**H2, H4가 핵심 학습**: Track A가 부분 성공(intrabar 모델 작동), H4 반전은 **후속 PDCA의 trigger**.

---

## 6. 방법론적 한계 & 교훈

### 한계
1. **데이터 OOS**: btc_5m_270days는 2026-04-03에서 끝 — LIVE 창 비교 불가. 최근 3주 5m 데이터 확보 필요.
2. **Slippage calibration**: 0.05~0.30% 는 "fix-후 median" 가정. 실제 측정은 30일 후 가능.
3. **19-trade LIVE 샘플의 소음**: 통계적 가설 검정 어려움. 30+ trades 필요.
4. **intrabar path 가정**: o→h→l→c 단일 경로 — 실제 tick 경로는 더 다양.

### 교훈
- **sl_trail_tuning의 clean BT 편향 실증**: val re-rank로 선정된 candidate_A/B가 slippage 환경에서 하위. **slippage-adjusted BT가 향후 파라미터 연구의 표준**이어야 함.
- **candidate_C 반전**: 1D grid winner가 3D selection protocol 통과자보다 강건. "grid 깊이 ≠ robustness" — 단순 1D 결과도 때로 더 정직.
- **Daily rate 비교의 한계**: 단기 LIVE 샘플은 장기 BT와 직접 비교 어려움. 재평가 기간 30일 이상 권장.
- **intrabar mode stress test**: 극단적 worst-case 가정 시 모든 전략이 붕괴 — path 모델링이 핵심.

---

## 7. Recommended Action

### 즉시
1. **production 변경 없음** — STOP 판정 준수
2. **Track A 스크립트는 연구 자산으로 보존** — 향후 파라미터 연구의 평가 도구
3. Report 작성 (완료 보고)
4. **Candidate_C 후속 PDCA** — 별도 feature로 본격 평가 검토

### 단기 (2~4주)
1. **30일 LIVE 샘플 수집**: fix-후 steady-state slippage median 실측
2. **Slippage 재-calibration**: 실측 기반으로 모델 값 갱신
3. **5m 데이터 확장**: BingX에서 2026-04-03 이후 5m 데이터 fetch → CSV 병합
4. **Candidate_C PDCA**: `(4.0, 2.5, 192)` 전용 sl_trail_tuning-style 재평가 with slippage

### 중기 (1~2개월)
1. **Track B (LIVE tick polling)**: Phase 1 STOP이어도 Track B는 독립 가치. thread 기반 재설계 진행 (Design §3.1 updated)
2. **Slippage-adjusted BT를 표준 파이프라인화**: `research_protocol_overfit_guards.md`에 규칙 추가 — "파라미터 연구 시 clean + slip 이중 평가 필수"
3. **intrabar tick path 모델 연구**: 단일 path 대신 확률적 path 여러 개 평균 (Monte Carlo)

---

## 8. Files Touched

- `scripts/analysis/c1_intrabar_parity.py` (NEW, ~320 lines)
- `results/intrabar_parity_20260419_065541.json` (NEW, 결과)
- `docs/01-plan/features/intrabar_parity.plan.md` (작성)
- `docs/02-design/features/intrabar_parity.design.md` (작성 + iterate)
- `docs/03-analysis/intrabar_parity.analysis.md` (본 문서)

Production 코드 변경 **0건**. Phase 1은 research-only.

---

## 9. Reference

- Plan: `docs/01-plan/features/intrabar_parity.plan.md`
- Design: `docs/02-design/features/intrabar_parity.design.md`
- 구현: `scripts/analysis/c1_intrabar_parity.py`
- 결과: `results/intrabar_parity_20260419_065541.json`
- 재사용: `scripts/analysis/intrabar_trail_impact.py`
- 선행 연구: `memory/sl_trail_tuning_20260419.md` (candidate 정의)
- BT-LIVE 분석: `claudedocs/bt_live_gap_deep_review_20260419.md`
- 정합성 체크: `claudedocs/BACKTEST_LIVE_PARITY.md` (#21, #22)
