# Body Filter Tuning PDCA 완료 보고서

> **Feature**: body_filter_tuning
> **Date**: 2026-04-19
> **Outcome**: **STOP (8/9) — 세션 최초 Near-GO**, 30일 LIVE 후 재평가 가치 매우 높음
> **Major breakthrough**: Exit 축 3축 모두 dead end 후 **Entry selectivity 축에서 강력 후보 발견**

---

## 1. Executive Summary

Body filter tuning 스윕(0.25~0.60) 결과 **세션 최초 "near-GO" candidate 발견**. 9 flags 중 8 통과, 유일 실패는 `wf_slip_pass` (fold 2 -5.62pp, 여전 음수 but +5.91pp 개선).

**Top candidate: candidate_C_b0.60** (max_sl_atr=4.0, trail_K=2.5, max_hold_bars=192, body_min_ratio=0.60)
- Slip PnL **+69.13%** (baseline +46.09 대비 **+23pp**)
- Slip MDD **10.10** (baseline 18.78 대비 **-46%**)
- **Ratio 6.84** (baseline 2.45 대비 **+179%**)

---

## 2. 7-value 스윕 결과

### Baseline (3.3, 2.5, 192)
| body | Clean PnL | Slip PnL | Slip MDD | Ratio |
|------|-----------|----------|----------|-------|
| 0.25 | +172.70 | +50.63 | 19.63 | 2.58 |
| 0.30 | +172.35 | +50.35 | 18.93 | 2.66 |
| **0.40** (현재) | **+169.55** | **+46.09** | **18.78** | **2.45** |
| 0.50 | +162.30 | +43.34 | 16.26 | 2.67 |
| 0.60 | +170.27 | +48.67 | **11.66** | **4.17** |

### Candidate_C (4.0, 2.5, 192) — **모든 body가 baseline_b0.40 압도**
| body | Clean PnL | Slip PnL | Slip MDD | Ratio |
|------|-----------|----------|----------|-------|
| 0.25 | +200.14 | **+71.89** | 14.31 | 5.02 |
| 0.30 | +199.44 | +71.16 | 13.87 | 5.13 |
| 0.40 | +192.76 | +63.06 | 14.26 | 4.42 |
| **0.60** | +191.07 | +69.13 | **10.10** | **6.84** |

---

## 3. 9-flag Validation (Top 3 candidates)

모든 3 candidate가 **8/9 PASS**:

| Flag | b0.25 | b0.30 | b0.60 |
|------|-------|-------|-------|
| wf_clean_pass | ✅ 5/5 | ✅ 5/5 | ✅ 5/5 |
| **wf_slip_pass** | **❌ 4/5** | **❌ 4/5** | **❌ 4/5** |
| tw_pass | ✅ | ✅ | ✅ |
| test_not_worse | ✅ | ✅ | ✅ |
| mc_pass (p<0.01) | ✅ 0.007 | ✅ 0.005 | ✅ 0.007 |
| ci_pass | ✅ | ✅ | ✅ |
| train_not_degraded | ✅ | ✅ | ✅ |
| pnl_improvement | ✅ (+25.8pp) | ✅ | ✅ |
| ratio_improvement | ✅ (+105%) | ✅ | ✅ (+179%) |

**Strict protocol: STOP** (core wf_slip_pass fail).

---

## 4. Fold-by-Fold Breakdown (slip_med)

| Fold | baseline_b0.40 | b0.25 | b0.30 | **b0.60** | 개선 |
|------|----------------|-------|-------|-----------|------|
| 1 | -2.08 | +0.99 | +0.99 | +2.59 | ↑ |
| **2** | **-11.53** | -8.88 | -8.45 | **-5.62** | **+5.91pp** |
| 3 | +34.79 | +37.27 | +37.37 | +36.29 | ↑ |
| 4 | +5.45 | +16.65 | +15.39 | +8.77 | ↑ |
| 5 | +19.47 | +25.86 | +25.86 | +27.10 | ↑ |

**Fold 2 slip이 여전 음수**이지만 **body=0.60에서 50% 이상 완화**. 2025-08 저변동성 regime 취약성은 구조적이므로 완전 해결 불가.

---

## 5. 세션 통합 인사이트

### Exit 재설계 실패 → Entry Selectivity 성공
| Dimension | 시도 | Top 결과 |
|-----------|------|---------|
| Trail 재설계 | breakeven_trail | ❌ MDD 5.6× |
| SL 재설계 | true_breakeven_sl_move | ❌ Whipsaw 89% |
| Emergency 축소 | emergency_sl_reduction | ❌ 효과 없음 |
| **Entry Selectivity** | **body_filter_tuning** | **✅ 8/9 flags, ratio +179%** |

**Exit는 이미 최적, Entry는 여지가 있었다** — C1 전략 개선의 정확한 축 확정.

### Body Filter의 메커니즘
- Body ↑ = **돌파 확신도 ↑** — 몸통 큰 봉은 방향성 명확
- 약한 돌파 signal 제거 → whipsaw 감소
- 거래 수 1074 → 937 (**-13%**), 그러나 PnL 유지 (+69 vs +46)
- **Per-trade 품질 대폭 상승** → ratio 2.45 → 6.84

---

## 6. 왜 Strict STOP인가

Core flag `wf_slip_pass` 실패:
- Fold 2 slip 여전 음수 (-5.62pp)
- fold2_regime_analysis에서 확인된 **구조적 저변동성 취약성**
- Body tightening은 완화책이지 근본 해결 아님

**하지만 Hit률 8/9 + ratio 2.8× 개선**은 모든 세션 PDCA 중 **최강 결과**.

---

## 7. 30일 LIVE 후 재평가 조건부 GO

### 조건부 GO 트리거 (사전 정의)
다음 **모두** 충족 시 candidate_C_b0.60 production 검토:
1. 30일 LIVE WR ≥ 30% (current 30.5% 근처)
2. 30일 LIVE PnL/trade ≥ baseline live 성과
3. 30일 중 "2025-08 유형 저변동성 regime" (ATR%<0.25, trend<5%) 미발생 OR 발생 시 영향 <3pp
4. Slippage 실측이 slip_med 범위 이하

---

## 8. Production 영향

**즉시 변경 없음**. 단:
- Candidate_C_b0.60이 **최강 production 후보**로 영구화
- 30일 LIVE 완료 후 조건부 재평가 예정
- 만약 GO 시: 단일 config 변경 (`body_min_ratio: 0.40 → 0.60`, `max_sl_atr: 3.3 → 4.0`)

---

## 9. 방법론 교훈

1. **축 선택의 중요성**: Exit 3축 dead end 후 Entry 축에서 대성공. 올바른 축 탐색의 가치.
2. **Body filter의 비선형성**: Clean BT에서 flat(+162~+173)이지만 slip에서 명확 차이. **Slip-adjusted 평가의 필수성 재확인**.
3. **Fold 2 구조적 한계 인정**: 저변동성 regime 약점은 parameter로 완전 해결 불가. Regime filter 필요성 시사.
4. **8/9 vs 9/9 strict protocol**: 1 flag 차이로 STOP이나, **실질 improvement 강도**는 지금까지 최고.
5. **Ratio 중심 평가의 가치**: PnL만 보면 +23pp이지만 ratio +179%가 진정한 품질 지표.

---

## 10. 후속 액션

### 즉시
1. Memory 영구화 (강력 후보 기록)
2. 30일 LIVE 샘플 수집 대기
3. Regime filter PDCA 고려 (fold 2 근본 해결)

### 단기 (2~4주)
1. **Regime-conditional candidate_C_b0.60** — 저변동성 감지 시 b0.40 또는 baseline 복귀
2. 30일 LIVE data로 live-BT parity 재측정

### 중기 (1~2개월)
- candidate_C_b0.60이 30일 live에서 baseline 상회 시 production 전환 검토
- 1m tick data로 intrabar #22 완화

---

## 11. Files Touched

- `scripts/analysis/body_filter_tuning_study.py` (NEW 초기 스윕)
- `scripts/analysis/body_filter_full_validation.py` (NEW 9-flag)
- `results/body_filter_tuning_20260419_175230.json`
- `results/body_filter_full_validation_20260419_175350.json`
- `docs/01-plan/features/body_filter_tuning.plan.md`
- `docs/04-report/body_filter_tuning.report.md` (본 문서)

Production 변경 **0건** (30일 LIVE 후 재평가 조건).

---

## 12. Bottom Line

세션 12건 PDCA 중 **유일한 "near-GO" + 최강 ratio 개선**. Exit 축 전부 dead end 확인 후 **Entry selectivity가 실제 개선 축**임 실증.

**Candidate_C_b0.60 (max_sl_atr=4.0, trail_K=2.5, max_hold_bars=192, body_min_ratio=0.60)** 는 세션 최종 production 후보.

30일 LIVE 후 조건부 GO 트리거 충족 시 실제 production 변경 가치 **매우 높음**.
