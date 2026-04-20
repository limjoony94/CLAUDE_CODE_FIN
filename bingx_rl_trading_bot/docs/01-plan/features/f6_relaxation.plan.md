# Plan: F6 기준 현실화 (Stability Gate Revision)

> **Feature**: f6_relaxation
> **Date**: 2026-04-20
> **Phase**: Plan
> **Trigger**: 4-path exploration 결과 — MR/Scalping/Relaxed 전부 F6 더 심각 실패 (daily 음수+pos% 폭락)
> **Target**: F6("top 5% 제거 후 양수")를 BTC R:R 3:1 trend-following 구조에 맞는 현실적 concentration 기준으로 대체

---

## 1. Background

### 문제 정의
사용자 지정 6-flag stability gate에서 F6(`pnl_ex_top5pct > 0`)가 **모든** 현행/대안 전략을 reject:

| Strategy | F6 값 | Daily | pos% | Sharpe |
|----------|-------|-------|------|--------|
| cand_C_b0.60 + trend (최고) | **-31.42** | +0.213 | 56.4% | +0.309 |
| Mean Reversion 최고 | -200.94 | **-0.356** | 33.7% | -0.432 |
| 5m Scalping 최고 | -385.11 | **-0.689** | 17.2% | -0.868 |

### 핵심 발견
1. **F6는 R:R 3:1 trend-following 구조적 상한**:
   - 대부분 trade의 작은 손실(-1R) + 소수 trade의 큰 이익(+3R 이상)
   - Top 5% 제거 시 "평균 손실 trade만 남음" → 음수는 수학적 필연
2. **대안 전략은 F6 개선이 아니라 **전면 붕괴**** (daily 음수, pos% 10~30%대)
3. **Structural Δ(+11~14pp) 유지**: parameter overfit 의심에도 cand vs base 두 half 일관

### 결론
F6 criterion 자체가 **misspecified**. 전략 교체로 해결 불가 → F6 기준을 "절대 음수 금지"에서 **"과도 집중 금지"**로 재정의 필요.

---

## 2. Hypotheses

| H | 내용 |
|---|------|
| H1 | F6는 R:R 비대칭 전략에 수학적으로 불리 → concentration 기반 기준이 더 적절 |
| H2 | Sharpe retention (ex-top5/full) ≥ 50% → outlier 의존도 허용선 |
| H3 | Top-5% contribution ratio ≤ 70% → "소수 폭발" 경계 유지 |
| H4 | 새 기준(F6')으로 cand_C+trend = 6/6 pass → 기존 5-flag + F6' 통합 |
| H5 | MR/Scalping은 F6' 에서도 reject → discrimination power 유지 (false positive 없음) |

---

## 3. Method

### 3.1 F6 대체 후보 (3 variants)

```python
# V1: Sharpe retention (상대적 안정성)
F6_v1: sharpe_retention_ex5 = sharpe_ex_top5 / sharpe_full >= 0.50

# V2: Contribution ratio (집중도 상한)
F6_v2: top5_contribution = pnl_top5 / total_pnl <= 0.70  # top 5%가 70% 이하 기여

# V3: Daily rate retention (ex-top5 기준 daily)
F6_v3: daily_ex_top5 / daily_full >= 0.30  # 30% 이상 유지
```

### 3.2 Empirical Validation

전체 기간(332일) 12 combos (stability_gate_validation.py 기존 grid)에 대해:
- 각 F6 variant 값 계산
- 기존 F6 실패 combos 중 F6' 통과 비율
- 대안 strategy (MR/Scalping)에도 F6' 적용 시 discrimination 확인

### 3.3 Threshold Calibration

- V1: 0.30, 0.40, 0.50, 0.60, 0.70 스윕
- V2: 0.50, 0.60, 0.70, 0.80, 0.90 스윕
- V3: 0.20, 0.30, 0.40, 0.50 스윕

각 threshold에서:
- True Positive: cand_C+trend 통과 (우리 최적 전략)
- True Negative: MR 4개, Scalping 3개 reject
- False Positive: baseline_b0.40 통과 여부 (우위 소멸 우려)

### 3.4 Combined Metric (권장)

```python
F6_final = (F6_v1 >= thr_v1) AND (F6_v2 <= thr_v2)
# AND 조합 → 두 관점(retention & concentration) 동시 충족
```

### 3.5 Stability Gate 재정의

기존 6-core:
- f1~f5 유지
- f6: `pnl_ex_top5pct > 0` → **`sharpe_retention_ex5 >= 0.5 AND top5_contribution <= 0.70`**

---

## 4. Success Criteria (GO Protocol)

### Core (3-flag, ALL required)
1. **discrimination**: cand_C+trend pass AND MR 4개 + Scalping 3개 전부 reject
2. **baseline_filter**: baseline_b0.40이 pass할 시 cand vs base delta 유지 확인
3. **robustness**: 2-half split에서 F6' 값 일관 (spearman ρ ≥ 0.5 across halves)

### Warnings (log only)
- cand_C+trend의 F6' 값이 threshold 대비 margin 확보 (overfit 위험 체크)
- Baseline combos(b0.40/b0.50) F6' 값

---

## 5. Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Threshold overfit (특정 후보만 통과) | Discrimination 테스트(TP/TN/FP) + margin 확인 |
| F6' 여전히 잘못된 축 측정 | V1/V2/V3 3 관점 동시 고려, combined AND 사용 |
| 기존 accepted 후보 reject | Backward check — 기존 body_filter top candidates 재평가 |
| "완화"로 인식되는 PR 저항 | Design 문서에 "구조적 이유" 명시 (R:R 3:1 수학적 분석) |
| Live 무관 (backtest-only metric) | F6'는 validation gate, live performance와 직교 |

---

## 6. Non-Goals

- 전략 수정 (현재 cand_C+trend 유지)
- F1~F5 변경 (이들은 valid)
- Warning criteria(Sharpe 0.35, p_loss_2pp) 변경
- Production config 변경 (enabled=false 유지)

---

## 7. Deliverables

1. `scripts/analysis/f6_relaxation_study.py` — 3 variant 계산 + threshold sweep
2. `scripts/analysis/stability_gate_v2.py` — F6' 반영한 새 gate
3. `docs/02-design/features/f6_relaxation.design.md` — 최종 기준 선정 근거
4. Memory: `f6_relaxation_20260420.md` — 결정 사항
5. `research_protocol_3day_bootstrap.md` 갱신 (F6 절 수정)

---

## 8. Reference

- 4-path exploration: `memory/four_path_exploration_20260420.md`
- Stability gate spec: `scripts/analysis/stability_gate_validation.py`
- Bootstrap protocol: `memory/research_protocol_3day_bootstrap.md`
- Structural delta 증거: `memory/deep_validation_20260420.md`
