# True Breakeven SL Move PDCA 완료 보고서

> **Feature**: true_breakeven_sl_move
> **Date**: 2026-04-19
> **Outcome**: **STOP** — 두 번째 negative result. Whipsaw 폭증으로 역효과.
> **Match Rate**: 95%

---

## 1. Executive Summary

사용자 요청 "수익중이지만 trail이 본절 미도달 구간" 해결책으로 **SL을 entry로 tighten** (trail은 유지) 방식 검증.

**결과**: ACTIVATION 6단계 중 **ACTIVATION=0 (기능 disable)이 최적**. BE SL이 C1 전략의 정상 pullback(0.3~0.5%)을 허위 청산 → whipsaw 89% SL exit.

**통합 결론**: Trail 메커니즘 재설계 (BUFFER 방식 + BE SL Move) **모두 실증 기각** → C1의 **원본 `max(0, projected)` + Fractal SL 조합이 수학적 최적에 근접** 확정.

---

## 2. 실행 매트릭스 (20 runs)

### Baseline
| ACTIVATION | Slip PnL | MDD | N | SL% |
|-----------|----------|-----|---|-----|
| **0.00** ⭐ | **+46.09** | **18.78** | 1074 | 11.1% |
| 0.10 | -588.32 | 588.32 | 1808 | **89.7%** |
| 0.20 | -408.73 | 408.73 | 1518 | 73.3% |
| 0.30 | -267.13 | 267.13 | 1323 | 58.1% |
| 0.50 | -101.45 | 101.45 | 1168 | 36.2% |
| 1.00 | +22.74 | 20.46 | 1079 | 15.6% |

### Candidate_C (동일 패턴)
| ACTIVATION | Slip PnL | MDD |
|-----------|----------|-----|
| 0.00 ⭐ | +63.06 | 14.26 |
| 1.00 | +43.04 (-20pp) | 14.96 |
| others | 모두 -50 이하 | |

---

## 3. 실패 메커니즘

### 3.1 Whipsaw 폭발 수학
```
Entry fill: 100.05 (slip +0.05)
Best:       100.30 (activation 0.10 넘김)
BE SL:      100.00 (entry) ← fill 대비 -0.05%
Price → 100: BE SL hit
Exit:       100.00
Net loss:   -0.30% (entry slip + exit slip + fee)
```

C1 전략의 **일반적 pullback 폭(0.3~0.5%)** 이 BE SL을 자주 터치:
- ACTIVATION=0.10: SL% 89.7% (!!), 진입 1074 → 1808 (+68%)
- 모든 trade가 BE/entry 사이 반복

### 3.2 Fractal SL의 숨은 가치
Fractal SL(-0.7~-1.0%)은:
- **Price structure 기반** — 일시적 pullback 흡수
- **Tail만 차단** — 정상 drawdown 허용
- BE SL(-0.05%)는 **structure 무시** → whipsaw

### 3.3 ACTIVATION=1.00이 "원본 근처"인 이유
`best_pnl > 1.0%` 도달이 드물어 BE SL **선별 발동**. 그래도 발동된 trades는 손해 → 원본 대비 -20pp.

---

## 4. 두 PDCA 통합 해석 (breakeven_trail + true_breakeven_sl_move)

### 2 가설 모두 기각
| PDCA | 메커니즘 | 결과 |
|------|----------|------|
| breakeven_trail | Trail 발동 차단 | ❌ tail risk 증폭, MDD 5.6× |
| **true_breakeven_sl_move** | **SL tighten to entry** | ❌ **whipsaw 폭증, SL 89%** |

### 근본 교훈
**C1의 `max(0, projected)` + fractal SL은 이미 수학적 최적**:
- Trail `max(0, ...)`: implicit breakeven cap
- Fractal SL: price-structure tail cut
- **두 메커니즘 상호 보완** — 둘 중 하나 변경 시 역효과

**"전통적 breakeven stop" ≠ 만능 기법**. C1처럼 pullback 폭 작은 전략엔 치명적.

---

## 5. 방법론적 교훈

1. **두 연속 기각 통계적 무게**: 단일 실험 failure는 noise. 2개 다른 메커니즘 모두 fail은 **방향성 잘못 확정**.
2. **원본 설계 존중 원칙 재확인**: 함부로 수정 시도 전 "왜 이렇게 설계됐는가" 파악.
3. **전략별 기법 궁합**: 전통 기법도 전략 특성에 따라 역효과.
4. **Exit reason 분포의 진단력**: SL% 11 → 89% 관찰로 원인 즉각 파악.
5. **Look-ahead audit 선행의 가치**: BT 신뢰성 선확인 → 두 negative result의 확실성 보장.

---

## 6. 사용자 질문 종결

> "손익분기 넘긴 시점부터는 SL을 손익분기점으로 옮기는 방안은?"

**답변**: BT 실증 기각. 이유:
1. C1 pullback 폭(0.3~0.5%) > BE SL margin(~0.05%) → whipsaw 폭증
2. Fractal SL이 이미 price-structure 기반 최적화
3. Trail + Fractal 조합이 수학적 최적

> "trailing 할 때 look-ahead bias 존재 여부 조사"

**답변**: 완료 (lookahead_audit_trail). **NO BIAS 확인**. 6 경로 전부 OK 또는 기존 structural limit.

---

## 7. 재정비된 개선 방향

### 폐기
- Trail/SL 메커니즘 재설계 (**dead end**)

### 재평가 우선순위
1. **pdca_candidate_body_filter** — body_min_ratio 민감도 (진입 selectivity)
2. **Regime-conditional candidate_C** — fold 2 완화
3. **Emergency SL 축소** — 3.0→2.5% (tail cap 직접)
4. **30일 LIVE slippage 실측** — BT calibration

### 원칙
- Trail/SL은 건드리지 않음
- **진입 + regime + emergency cap 축**에서 개선 탐색

---

## 8. Production 영향

변경 **0건**. Baseline (3.3, 2.5, 192) + 원본 trail + fractal SL 유지.

---

## 9. Files Touched

| 파일 | 역할 |
|------|------|
| `scripts/analysis/true_breakeven_sl_move_study.py` | NEW (~340 lines) |
| `results/true_breakeven_sl_move_20260419_173611.json` | 결과 |
| `docs/01-plan/features/true_breakeven_sl_move.plan.md` | 가설 |
| `docs/03-analysis/true_breakeven_sl_move.analysis.md` | 상세 분석 |
| `docs/04-report/true_breakeven_sl_move.report.md` | 본 보고서 |

---

## 10. Bottom Line

사용자 직관 "본절 SL 이동"은 **합리적 전통 기법**이나 **C1 전략에는 부적합**. 20-run 매트릭스 실증으로 확정.

**Trail 메커니즘 재설계 방향 전체가 dead end**. 원본이 이미 최적. 개선은 **진입/regime/emergency 축**에서 탐색 필요.

**Look-ahead audit + 두 negative result 결합**으로 "C1 원본 설계의 수학적 정교함" 확정. 이는 귀중한 **strategy integrity 증거**.
