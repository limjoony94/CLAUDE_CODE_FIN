# Regime Filter (Low-Vol) PDCA 완료 보고서

> **Feature**: regime_filter_lowvol
> **Date**: 2026-04-19
> **Outcome**: **STOP (1/3)** — Fold 1/2 구분 불가 한계 실증
> **Conclusion**: **Candidate_C_b0.60 자체 유지가 최선**. Fold 2는 구조적 감수.

---

## 1. Executive Summary

Body_filter_tuning의 유일 실패(wf_slip_pass, fold 2 -5.62)를 rolling ATR% 저변동성 필터로 해결 시도. **15-run grid 결과 Trade-off 확정**:

- Fold 2 양수 전환 가능 (+0.92 at thr0.24_lb192)
- **그러나 Fold 1이 양수→음수** (+2.59 → -3.75)
- Full PnL -15pp (+69 → +54)

**근본 문제**: Fold 1 ATR% 0.248 ≈ Fold 2 ATR% 0.229 (거의 동일). 단일 metric으로 구분 불가.

---

## 2. 실행 결과 (15 runs)

Top 3 (fold 2 양수):
| Combo | slip PnL | Fold 1 | Fold 2 | WF |
|-------|----------|--------|--------|-----|
| thr0.24_lb192 | +54.03 | -3.75 | **+0.92** | 4/5 |
| thr0.26_lb288 | +49.52 | -9.36 | +4.84 | 3/5 |
| thr0.24_lb288 | +48.63 | -7.46 | +0.60 | 3/5 |

**Baseline (no filter)**:
- Slip PnL +69.13, Fold 1 +2.59, Fold 2 -5.62, WF 4/5

---

## 3. 3-flag GO 결과

| Flag | Result |
|------|--------|
| fold2_slip_positive | ✅ |
| overall_not_degraded (±5pp) | ❌ (-15pp) |
| wf_slip_5of5 | ❌ (fold 1이 새 음수) |

**1/3 PASS → STOP**.

---

## 4. 근본 원인: Fold 1 vs Fold 2 미분리

Regime profiles (fold2_regime_analysis):
| Fold | ATR% | returns_std | Trend |
|------|------|-------------|-------|
| 1 | 0.248 | 0.184 | +24% |
| 2 | **0.229** | **0.169** | **-2.6%** |

- ATR%: 0.248 vs 0.229 (8% 차이 — single threshold로 구분 어려움)
- 유일한 명확 차이: **Trend** (+24% vs -2.6%)

단일 ATR% threshold는 **fold 1의 초기 uptrend를 놓친 후 fold 2와 유사해진 시점**을 잡아 fold 1의 양수 구간을 함께 제거.

---

## 5. fold2_regime_analysis 예측 실증

본 PDCA는 fold2_regime_analysis의 **H7 PARTIAL** 결론 (single-metric filter 한계) 실증 확인:
> "Clean filter 1개, fold_1 포함 → 단일 metric 분리 불가, multi-metric ML 필요"

---

## 6. 실제 최선 경로

### Option A: Candidate_C_b0.60 그대로 (추천)
- Fold 2 감수 + 다른 4 folds 강력 이점
- 30일 LIVE에서 fold 2 유형 regime 미발생 시 조건부 GO

### Option B: Multi-metric ML Classifier (별개 대형 PDCA)
- ATR% + trend + range + sideways 등 다변량
- Decision tree or logistic regression
- 본 세션 범위 외

### Option C: Trend filter 추가
- `recent_trend_pct < +5%` 조건
- Fold 1(+24%) 통과, fold 2(-2.6%) 차단
- **본 PDCA 미검증**, 빠른 실험 가치

---

## 7. Production 영향

변경 0건. **Candidate_C_b0.60 (body=0.60)이 여전히 최종 후보**.

---

## 8. 교훈

1. **Single-metric regime filter의 구조적 한계** — 유사 regime 구간들 미분리
2. **Fold-specific 최적화의 비용** — Fold 2 잡으면 Fold 1 잃음 (trade-off)
3. **fold2_regime_analysis 예측의 정확성** — "multi-metric 필요" 예언 실증
4. **Trend filter 미탐색** — Option C가 남은 유일한 빠른 검증 후보

---

## 9. Files Touched

- `scripts/analysis/regime_filter_lowvol_study.py` (NEW, ~270 lines)
- `results/regime_filter_lowvol_20260419_180219.json`
- `docs/01-plan/features/regime_filter_lowvol.plan.md`
- `docs/04-report/regime_filter_lowvol.report.md` (본 문서)

---

## 10. Bottom Line

Fold 2 근본 해결은 **단일 metric 불가 확정**. **Candidate_C_b0.60 (body_min_ratio=0.60)** 이 여전히 세션 최강 후보. 30일 LIVE 대기가 현실적 경로.

남은 가능성: **Trend filter 추가** (fold 1 +24%, fold 2 -2.6% 구분) — 별개 빠른 PDCA 가치.
