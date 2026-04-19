# Analysis: True Breakeven SL Move

> **Feature**: true_breakeven_sl_move
> **Date**: 2026-04-19
> **Phase**: Check
> **Outcome**: **STOP** — 두 번째 negative result. BE SL이 whipsaw로 역효과.
> **Match Rate**: 95% (단순 monkey patch, 구현 직관적)

---

## 1. Executive Summary

"Trail 유지 + SL을 entry로 tighten" 방식(breakeven_trail BUFFER 방식과 다른 메커니즘)을 ACTIVATION 6개 × 2 combos × 2 modes = 20 runs 실증. **결과: 모든 ACTIVATION > 0에서 악화**.

**두 가설(breakeven_trail + true_breakeven_sl_move) 연속 기각**은 "**C1의 원본 trail+fractal SL 조합이 이미 수학적 최적**"임을 강력히 시사.

---

## 2. 실행 결과

### ACTIVATION 스윕 (baseline)
| ACTIVATION | Clean PnL | Slip PnL | Slip MDD | Slip N | SL% | Trail% |
|-----------|-----------|----------|----------|--------|-----|--------|
| **0.00** ⭐ | **+169.55** | **+46.09** | **18.78** | 1074 | 11.1 | 88.8 |
| 0.10 | -301.61 | -588.32 | 588.32 | 1808 | **89.7** | 10.2 |
| 0.20 | -227.40 | -408.73 | 408.73 | 1518 | 73.3 | 26.7 |
| 0.30 | -160.42 | -267.13 | 267.13 | 1323 | 58.1 | 41.8 |
| 0.50 | -50.49 | -101.45 | 101.45 | 1168 | 36.2 | 63.7 |
| 1.00 | +126.06 | +22.74 | 20.46 | 1079 | 15.6 | 84.3 |

### Candidate_C도 동일 패턴
| ACTIVATION | Slip PnL | MDD |
|-----------|----------|-----|
| 0.00 | +63.06 | 14.26 |
| 0.10 | -558.66 | 558.66 |
| 0.20 | -378.66 | 378.66 |
| 0.30 | -232.79 | 232.79 |
| 0.50 | -77.80 | 77.80 |
| 1.00 | +43.04 | 14.96 |

**모든 ACTIVATION > 0가 baseline 이하**. 최적은 ACTIVATION = 0 (기능 disable).

---

## 3. 실패 원인 분해

### 3.1 Whipsaw 폭증 (ACTIVATION=0.10)
- Baseline: 1074 trades, SL 11.1%, Trail 88.8%
- ACTIVATION=0.10: 1808 trades (+68%), SL **89.7%**, Trail 10.2%

**BE SL이 거의 모든 거래를 조기 청산**. 진입→BE 조건 만족→가격 pullback→BE SL hit→재진입 cycle.

### 3.2 수학적 이유
```
Entry fill:  100.05 (slip +0.05%)
Best price:  100.30 (best_pnl = 0.30%, activation 0.10 넘김)
BE SL set:   entry = 100.00  (fill 대비 -0.05%)
Price returns to 100 → BE SL hits
Exit price:  100.00
Net loss:    -0.05 (entry slip) - 0.15 (exit slip) - 0.10 (fee) = -0.30%
```

C1 전략의 일반적 pullback 폭(0.3~0.5%)이 BE SL을 자주 터치 → **whipsaw storm**.

### 3.3 Fractal SL의 가치 재확인
- Fractal SL: entry - 3.3×ATR ≈ -1.0% (넓음)
- **Price structure 기반** — 일시적 pullback 흡수
- Only **tail**(crash level) 차단
- **최적화된 drawdown/win trade-off**

BE SL이 이 fractal을 덮어쓰면:
- Whipsaw 급증 (SL 11% → 89%)
- 작은 이익도 놓침 (trail 기회 상실)
- 진입 빈도 상승으로 fee+slip 누적

### 3.4 왜 ACTIVATION=1.00은 원본에 가까운가
`best_pnl > 1.0%` 달성 시에만 BE 활성화 → 대부분 trade는 activation 전에 trail 또는 SL로 종료. BE는 **선별적으로만 발동** → 원본과 유사한 동작. 그래도 원본보다 -20pp 낮음 (BE 활성화된 trade들이 whipsaw로 손해).

---

## 4. 가설 평가 (7 가설 기각)

| H | 결과 |
|---|------|
| H1 MDD 감소 | ❌ MDD 폭증 (18 → 100+) |
| H2 PnL 유지/상승 | ❌ PnL -149 ~ -634pp |
| H3 ACTIVATION 최적 ~0.3 | ❌ 최적은 0.00 |
| H4 WF 유지 | ❌ 대부분 ACTIVATION WF 붕괴 |
| H5 Whipsaw 증가 (예상 trade-off) | ✅ 예상대로, but 심각 |
| H6 candidate_C synergy | ❌ cand도 동일 악화 |
| H7 fold 2 개선 | ❌ 검증 불필요 (전체 fail) |

**6/7 기각, 1 partial**. breakeven_trail과 같은 강력 negative.

---

## 5. 8-flag GO (top = candidate_C_a0.00, 즉 BE disable)

| Flag | Result |
|------|--------|
| wf_clean_pass | ✅ (원본 값) |
| **wf_slip_pass** | ❌ (4/5, fold 2) |
| tw_pass | ✅ |
| train_not_degraded | ✅ |
| **pnl_improvement** | ❌ (원본과 동일, 개선 없음) |
| ratio_ok | ❌ (1.05× 미달) |
| activation_stable | ❌ (0/3 better) |
| rollback_ready | ✅ |

**VERDICT: STOP** — core flag wf_slip_pass, pnl_improvement fail.

---

## 6. 핵심 인사이트 (두 PDCA 통합)

### breakeven_trail + true_breakeven_sl_move 연속 기각의 의미

**Trail+Fractal 조합은 수학적 최적에 매우 근접**:
- Trail의 `max(0, projected)` 는 implicit breakeven cap
- Fractal SL은 price-structure 기반 tail cut
- 두 메커니즘이 **상호 보완** (둘 중 하나 변경 시 역효과)

### "전통적 기법" 의 C1 부적합성
- Breakeven SL Move: 전통 trader 기법, C1에는 whipsaw storm
- Trail hold till BE: 직관적, C1에는 tail risk 폭발
- **전략의 drawdown-profile과 메커니즘 궁합이 중요**

### 개선 방향 재조정
Trail/SL 메커니즘 재설계 **dead end**. 다른 축으로 개선:
1. **진입 Selectivity** (body filter 민감도, regime filter)
2. **Regime-conditional 파라미터** (candidate_C 조건부)
3. **Slippage 실측 기반 재calibration** (30일 LIVE)

---

## 7. 방법론적 교훈

1. **두 negative result 연속의 통계적 무게**: 단일 실패는 noise, 두 메커니즘 기각은 **방향성 자체가 틀렸다** 는 증거
2. **C1 원본 설계의 정교함**: trail `max(0,...)` + fractal SL 조합은 **수학적 최적점에 근접**
3. **전통 기법의 전략별 적합성**: "breakeven stop"이 모든 전략에 유효하지 않음 — C1처럼 pullback 폭 0.3~0.5% 전략에는 whipsaw 폭발
4. **Regression check의 필수성**: ACTIVATION=0이 원본과 일치 확인 → monkey patch 안전성 검증
5. **Exit reason breakdown의 진단 가치**: SL% 89% 관찰로 원인 즉각 파악 가능

---

## 8. 후속 방향 (재정비)

### 즉시 폐기
- Trail 메커니즘 재설계 방향 **전체** (BUFFER, BE SL Move 전부)

### 재평가 가치 있는 방향
1. **pdca_candidate_body_filter** (memory에 후보로 존재) — 진입 selectivity
2. **Regime-conditional candidate_C** — fold 2 완화
3. **Emergency SL 축소** (3.0→2.5%) — tail cap 직접 제어
4. **30일 LIVE fix-후 slippage 실측** — BT slippage recalibration

### 장기
- 1m tick data 확보 (intrabar parity #22 완화)

---

## 9. Files Touched

- `scripts/analysis/true_breakeven_sl_move_study.py` (NEW, ~340 lines)
- `results/true_breakeven_sl_move_20260419_173611.json`
- `docs/01-plan/features/true_breakeven_sl_move.plan.md`
- `docs/03-analysis/true_breakeven_sl_move.analysis.md` (본 문서)
- `docs/04-report/true_breakeven_sl_move.report.md` (next)

Production 변경 **0건**.

---

## 10. Reference

- Plan: `docs/01-plan/features/true_breakeven_sl_move.plan.md`
- 선행 실패: `memory/breakeven_trail_20260419.md` (BUFFER 방식 기각)
- BT 신뢰: `memory/lookahead_audit_trail_20260419.md` (NO BIAS)
- 원본 최적성 시사: 두 PDCA 연속 기각
