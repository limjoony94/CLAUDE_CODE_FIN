# M1-A MTF Scalping — Phase 0~2 결과: NEGATIVE (사용자 보고)

> **Date**: 2026-04-27
> **Status**: 🛑 **M1-A spec STRUCTURALLY INSUFFICIENT** (data-conclusive)
> **Capital risk**: 0 (BT only, no LIVE deployment)
> **Time invested**: ~1 session (Phase 0 분석 파이프라인 + Phase 1~2 BT)
> **Deliverable**: 방법론 학습 — 첫 spec의 clean negative result

---

## 1. Summary

사용자 사전 등록 success criteria를 plan에 명문화 후, single concrete spec **M1-A** (1h+4h trend filter + 5m RSI pullback + 15m buffered alignment + ATR trail TP + structural SL) BT 수행. 결과:

| Test | Result | Conclusion |
|------|--------|------------|
| Phase 0.1 데이터 정합성 | 4-TF 100% aligned | OK |
| Phase 0.2 entry frequency | 4.52/day (D3 buffered) | criterion 7 PASS |
| Phase 2 full BT (718d, friction-aware) | net -496%, daily -0.69%, **1/5** hard criteria | FAIL |
| Phase 2.5 entry isolation (fixed exit) | gross +1.3 ~ +24.1% | tiny edge |
| Phase 2.6 MFE distribution | MFE P50 +0.29% (vs friction 0.20%) | marginal |
| **Phase 2.7 random baseline** | **Random MFE P50 +0.38% > M1-A +0.29%** | **DECISIVE NEGATIVE** |

---

## 2. Decisive Finding: M1-A entry filter is **anti-selective**

Random entry on the same 1h+4h trend-filtered universe produces:
- **Higher MFE P50** (+0.38% vs M1-A +0.29%) — random captures more favorable excursion
- **More % above friction** (71.1% vs M1-A 62.5%) — random reaches profit threshold more often
- **Same Final close P50** (~0%) — neither has directional edge

**Mechanism**: M1-A의 RSI cross + body > 40% + close > EMA9 조건들은 entry를 momentum **이미 발생한 후** 시점으로 밀어 peak 가까이로 옮김. 결과 MFE 감소.

이는 사용자 spec literal 해석 "RSI cross + body 강한 캔들에서 진입"이 **mean-reversion**보다 **momentum chasing** 패턴이며, 이 패턴이 BTC 5m noise 환경에서 anti-edge를 생성한다는 의미.

---

## 3. Detailed Evidence

### 3.1 Phase 2 Full BT (718 days, friction 0.20%/trade)

| Reason | n | WR | avg gross | avg net | bars |
|--------|---|----|-----------|---------|------|
| SL | 828 | 0% | -0.21% | -0.41% | 3.0 |
| TRAIL_TP | 1276 | 19.2% | +0.07% | -0.13% | 8.4 |
| TIMEOUT | 41 | 82.9% | +0.61% | +0.41% | 24.0 |
| EMERGENCY | 2 | 0% | -1.5% | -1.7% | 1.0 |

- **Gross sum: -66.85%** (전체 718d, 2147 trades)
- Net sum: -496% (1x additive)
- WR: 13.0%
- Daily 1x: -0.69%
- 1/5 hard criteria PASS (≥2 trades/day만 통과)

**Key**: gross -66.85% = friction 이전부터 음수. friction이 edge를 죽인 게 아니라 **edge 자체 부재**.

### 3.2 Phase 2.5 Entry Signal Isolation (advisor 권고)

Same M1-A entries, replace exit with fixed-N-bar timeout (no SL, no trail):

| Horizon | trades | per_day | gross sum | gross avg/trade | gross WR |
|---|---|---|---|---|---|
| 12 bars (1h) | 1813 | 2.52 | +15.27% | +0.0084% | 49.7% |
| 24 bars (2h) | 1565 | 2.18 | **+24.13%** | **+0.0154%** | 49.8% |
| 48 bars (4h) | 1298 | 1.81 | +1.28% | +0.0010% | 49.6% |

**Tiny edge** at 24-bar (+0.0154% gross/trade). Friction 0.20%/trade = **13× larger**. WR ≈ 50% (random expected).

### 3.3 Phase 2.6 MFE/MAE Distribution (24-bar window)

| Metric | M1-A |
|---|---|
| MFE mean / P50 | +0.48% / +0.29% |
| MFE P75 / P90 | +0.61% / +1.12% |
| MAE mean / P50 | -0.45% / -0.29% (symmetric) |
| Final close mean | +0.015% |
| % MFE > friction 0.20% | 62.5% |
| % MFE > 2×friction 0.40% | 39.4% |

Symmetric MFE/MAE는 **noise distribution** 시사. Final close = 0 = no directional edge.

### 3.4 Phase 2.7 Random Baseline (DECISIVE)

Same trend-filtered universe (1h+4h aligned), random entry, N=1, 2-bar cooldown, 5 seeds × 1565 samples each:

| Metric | M1-A | Random (avg) | Diff |
|---|---|---|---|
| MFE P50 | +0.29% | **+0.38%** | M1-A −0.10pp |
| MAE P50 | -0.29% | -0.37% | symmetric |
| Final P50 | 0% | 0% | both ~zero |
| % MFE > 0.20% | 62.5% | **71.1%** | M1-A −8.6pp |

**Random > M1-A on every favorable metric**. M1-A의 entry filter가 directional edge를 만들지 않을 뿐 아니라 **favorable excursion을 줄임**.

---

## 4. C1 Postmortem과의 비교

| 항목 | C1 Breakout v2.6 | M1-A MTF Scalping |
|------|------------------|-------------------|
| BT 단계 평가 | 모든 검증 통과 (WF 5/5, MC p=0.000, etc.) | **BT부터 음수** (gross -66.85%) |
| 발견 시점 | LIVE 14d 후 (-$623) | Phase 2 BT (capital risk 0) |
| 근본 원인 | BT-LIVE friction gap | **Entry signal alpha 부재** (random보다 나쁨) |
| Decisive evidence | rolling 14d distribution: LIVE < BT P0 | random entry MFE > M1-A MFE on filter universe |

**Lesson**: C1과 달리 M1-A는 **BT 단계에서 발견**. distribution check + random baseline이 작동했다는 의미. 사전 등록한 baselines (claudedocs/m1_baseline_definition.md §4 Baseline C)가 결정타를 만듦.

---

## 5. Why Continuing to Tune is the C1 Trap

Trail K, SL multiplier, RSI threshold, body ratio 등을 grid search하면 BT-passing variant 만들기는 가능:
- Phase 2 데이터가 그래도 양수에 borderline → tuning으로 BT positive 가능
- 하지만 random baseline이 더 좋음 = **edge 없는 변수공간**
- C1 함정 정확히 재현

advisor 명시 (2026-04-27 세션 중):
> "MFE numbers don't support activation-trail hypothesis. Don't propose it as next experiment."
> "Tuning exits will produce a BT-passing variant via overfitting, and it will fail in LIVE. Same as C1."

---

## 6. Two Concrete Forward Paths (사용자 결정 영역)

### Path A: Paradigm shift within scalping framework
M1 entry signal 부재 확인 → 다른 entry mechanism 시도. 후보:
- **15m execution** (5m noise 회피, slower 캔들로 noise filtering 자연 발생)
- **Volatility expansion** (squeeze 후 breakout — momentum 이미 발생한 후 진입이 아닌 squeeze 시점 진입)
- **Mean-reversion at extremes** (RSI < 25 / > 75, BB band 외부 등 — 현재 RSI 40 cross는 momentum-following이라 anti-edge)
- **Multi-asset** (BTC 외 다른 asset에서 같은 framework — 단일 asset 5m noise 한계 회피)

### Path B: 방법론 학습 일시 중단 + 재계획
- **Capital risk 0**, **time pressure 0** 상태 유지
- C1 + M1-A 두 negative result에서 공통 lesson 추출:
  - BT-LIVE friction gap은 항상 적용 (M1은 BT에서 미리 inject)
  - Random baseline은 모든 BT의 prerequisite (M1-A에서 결정타)
  - Entry signal "alpha vs noise" 검증이 모든 다른 검증보다 우선
- 다음 시도 전 **사용자가 paradigm/asset/timeframe 결정**

### Path C: Stop & accept (덜 권장)
- Trading 자체 보류. 자본 보존.
- 시장이 변화하거나 더 좋은 idea 생기기 전까지 대기.
- 기존 자본 ($1495.22) 다른 곳에 활용.

---

## 7. What I (assistant) recommend

**Path B**. 이유:
- Capital risk 0 + time pressure 0 = 천천히 결정 가능
- C1, M1-A 두 negative result에서 가장 큰 lesson은 **process** (random baseline, friction inject, 사전 등록 etc.)
- Path A는 같은 process bug 재현 위험. Path A로 가더라도 **사전 등록 + random baseline 의무** 적용.

구체 방법:
1. 본 보고서 + C1 postmortem 사용자 review
2. Spec 결정 (asset / timeframe / signal class) 사용자가 명시
3. 다음 spec 사전 등록 (baseline + criteria + bootstrap protocol)
4. **다른 paradigm**: random baseline 비교를 plan에 사전 포함 — 이번처럼 BT 후 발견이 아니라 사전 검증.

---

## 8. Files Generated (this session)

| File | Purpose |
|------|---------|
| `docs/01-plan/features/mtf_scalping_v1.plan.md` | Plan + spec evolution log |
| `claudedocs/m1_friction_model.md` | Phase 0.3 사전 등록 |
| `claudedocs/m1_baseline_definition.md` | Phase 0.4 사전 등록 (Baseline C가 결정타) |
| `scripts/analysis/m1_data_integrity_20260427.py` | Phase 0.1 |
| `scripts/analysis/m1_entry_frequency_20260427.py` | Phase 0.2 |
| `scripts/analysis/m1_entry_15m_role_compare_20260427.py` | Phase 0.2 보강 |
| `scripts/analysis/m1_bt_framework.py` | Phase 1+2 framework + BT |
| `scripts/analysis/m1_entry_signal_isolation.py` | Phase 2.5 (advisor 권고) |
| `scripts/analysis/m1_mfe_distribution.py` | Phase 2.6 |
| `scripts/analysis/m1_random_entry_baseline.py` | Phase 2.7 (decisive) |
| `results/m1_*.json` | 모든 측정 raw data |
| `docs/04-report/m1_scalping_v1_negative_result_20260427.md` | (이 문서) |

---

## 9. Open Questions for User

1. **Path 선택** (A: paradigm shift / B: 재계획 / C: stop)
2. Path A 시 — asset / timeframe / signal class 선호?
3. 본 negative result에서 process를 추가 강화할 부분 있는지? (예: pre-BT random baseline 의무)

---

## 10. Process Lessons Captured

이번 세션에서 본인 (assistant)이 4번 advisor 호출. 그 중:
- 1번 안 통하고 옵션 C (preemptive 직관 변경) 시도 — advisor 정정 후 옵션 A
- 1번 D3/D4 결정 불일치 — advisor 정정
- 1번 trail K 조정 시도 충동 — advisor "C1 spec-tuning 함정" 경고
- 1번 activation-trail 가설 시도 충동 — advisor "MFE 데이터 미지원, random baseline 먼저"

**Pattern**: 진단 데이터 나오면 자동으로 "fix 시도" 충동. advisor가 매번 "report to user, paradigm shift 가능성"으로 redirect.

→ 이 패턴 자체가 process bug. 다음 strategy session 전에 명문화 (memory feedback로 저장).
