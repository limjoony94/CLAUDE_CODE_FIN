# Plan: Body Filter Tuning (진입 Selectivity)

> **Feature**: body_filter_tuning
> **Date**: 2026-04-19
> **Phase**: Plan (compact)
> **Type**: Entry selectivity 개선 연구

---

## 1. Background

Trail/SL/Emergency 3축 dead end 확정 후, 유일한 빠른 검증 가능 축: **진입 selectivity**.

현재 C1: `body_min_ratio = 0.40` — 돌파 봉의 |body|/range ≥ 40% 요구.

**관찰**:
- extended_param_grid (clean BT): 0.25~0.55 전부 양수 (+162~+173)
- 2026-04-19 실거래: body 38.37%로 아슬하게 탈락한 bar 관찰 (사용자 제기)
- pdca_candidate_body_filter 메모리 후보 존재

**가설**: Slippage 환경에서 body ↑ (selective) → whipsaw 감소 → PnL/MDD 개선.

---

## 2. Hypotheses

| H | 내용 |
|---|------|
| H1 | body ↑ 시 slip_med PnL 개선 (selective entry) |
| H2 | body ↑↑ 시 trade 수 급감으로 역효과 (sweet spot 존재) |
| H3 | Clean BT는 평평 (이미 1D grid 확인) |
| H4 | Candidate_C + body 최적 = synergy |

---

## 3. 검증 매트릭스

Body sweep: 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60 × 2 combos × 2 modes = 28 runs.

---

## 4. 간단 GO

1. `pnl_improvement`: slip_med PnL ≥ baseline + 5pp
2. `trade_count_ok`: 거래수 baseline의 ≥ 70%
3. `ratio_ok`: PnL/MDD ≥ baseline
4. `rollback_ready`: config-driven

4/4 PASS GO. 핵심 실패 시 STOP.

---

## 5. 실행
신규 스크립트 `scripts/analysis/body_filter_tuning_study.py`. Monkey patch 불필요 — `set_combo`로 cfg의 body_min_ratio 변경.
