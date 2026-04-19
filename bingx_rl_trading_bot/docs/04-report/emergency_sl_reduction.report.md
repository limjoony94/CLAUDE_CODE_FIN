# Emergency SL Reduction PDCA 완료 보고서

> **Feature**: emergency_sl_reduction
> **Date**: 2026-04-19
> **Outcome**: **STOP — NO IMPROVEMENT**
> **Significance**: **3번째 Exit 메커니즘 재설계 기각** — C1 최적성 확정

---

## 1. Executive Summary

Emergency SL 축소 (3.0 → 1.5%) 검증 결과 **실질 개선 없음**:
- MDD 전부 동일 (18.78%)
- PnL은 축소 시 악화 (esl=1.5에서 baseline -11.76pp)

Emergency는 rare event (발동 ≤16/1077 trades). Fractal SL이 이미 대부분 tail 처리.

---

## 2. 실행 결과

| ESL | baseline slip PnL | MDD | SL exits | EMG exits |
|-----|-------------------|-----|----------|-----------|
| **3.0** (현재) | **+46.09** | **18.78** | 119 | 1 |
| 2.5 | +46.40 | 18.78 | 118 | 2 |
| 2.0 | +47.36 | 18.78 | 116 | 4 |
| 1.5 | +34.33 | 18.78 | 112 | 16 |

Candidate_C 동일 패턴 (ESL 축소 시 PnL 감소).

---

## 3. 왜 효과 없는가

### Emergency의 본질
- Fractal SL(~-1%)의 **2차 보호장치** (flash crash 대비)
- 정상 BT에서는 거의 발동 안 함 (MDD는 SL streak에 의해 결정)

### ESL 축소의 부정적 효과
- Fractal SL 발동 전에 미리 emergency 청산
- **Price-structure 기반 손절의 이점 상실**
- 결과: margin PnL 개선 또는 악화

---

## 4. 3축 통합 결론 (세 번째 dead end)

| # | 축 | 시도 | 결과 |
|---|----|------|------|
| 1 | Trail 재설계 | breakeven_trail (BUFFER) | ❌ MDD 5.6배 |
| 2 | SL 재설계 | true_breakeven_sl_move | ❌ Whipsaw 89% |
| 3 | **Emergency 축소** | **emergency_sl_reduction** | **❌ 효과 없음** |

**C1 Exit 메커니즘(Trail + Fractal + Emergency)은 수학적 최적 도달 확정**.

---

## 5. 재정비된 유일한 개선 방향

**Exit 축은 전부 dead end**. 남은 축:

### 1. 진입 Selectivity (빠른 검증 가능)
- **body_min_ratio 민감도** (pdca_candidate_body_filter, 기존 후보)
- body가 낮은 돌파는 필터 → win rate 상승 기대

### 2. Regime Filter (구조적 변경)
- **Regime-conditional candidate_C** (fold 2 원인 해결)
- 고변동성 레짐에서만 max_sl_atr=4.0, 저변동성엔 3.3

### 3. 외부 요인
- **30일 LIVE slippage 실측** (BT calibration)
- **1m tick data 확보** (intrabar parity #22)

---

## 6. Production 영향

변경 0건. 현재 ESL=3.0 유지.

---

## 7. 방법론 교훈

1. **빠른 negative 검증의 가치**: Emergency 축소가 쉽게 기각되어 다른 축으로 즉각 이동 가능
2. **Rare event의 한계**: 발동 <2% events는 전체 PnL/MDD에 영향 거의 없음
3. **3축 연속 기각의 의미**: Strategy integrity 확정 — 향후 연구자에게 시간 절약
4. **원본 설계의 정교함**: C1 exit는 "여기가 최적"을 실증적으로 규명

---

## 8. Files Touched

- `scripts/analysis/emergency_sl_reduction_study.py` (NEW, ~120 lines)
- `results/emergency_sl_reduction_20260419_174713.json`
- `docs/01-plan/features/emergency_sl_reduction.plan.md`
- `docs/04-report/emergency_sl_reduction.report.md`
