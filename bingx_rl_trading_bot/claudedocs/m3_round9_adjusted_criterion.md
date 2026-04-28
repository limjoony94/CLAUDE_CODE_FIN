# M3-R9 — Adjusted Criterion 사전 등록 + Deep Verify Plan

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 옵션 F (Criterion 조정) + 추가 심층 탐색
> **Anti-pattern declaration**: F는 anti-pattern guard 위반 가능 — 17 mechanisms fail 후 criterion 낮춤. 사용자 explicit instruction 으로 진행. **Retrofit 방지를 위해 본 문서를 deep verify 시작 전 별도 commit**.

---

## 1. Original criterion (사용자 3회 명시)

| # | 항목 | 값 |
|---|------|------|
| 1 | Execution | 5분/15분 캔들 + 1h/4h trend filter |
| 2 | WR | ≥ 50% |
| 3 | R:R | ≥ 1 |
| 4 | TP/SL | dynamic |
| 5 | Daily | ≥ +0.2% (1x) |
| 6 | Per-trade gross | ≥ taker fee |
| 7 | Frequency | ≥ 2 trades/day |
| 8 | 3-day bootstrap | stable |
| 9 | Look-ahead / overfit | none |

**원본 criterion으로 17/17 die — 합격자 0**.

## 2. Adjusted criterion (사용자 옵션 F 적용)

**조정 핵심**: `daily ≥ +0.2%`이 multiplicative gap (~10×)으로 도달 불가능. 다른 5개 항목은 유지.

| # | 항목 | 원본 | **조정** | 사유 |
|---|------|------|---------|------|
| 1 | Execution | 5m/15m + 1h/4h | (유지) | – |
| 2 | WR | ≥50% | **≥40%** | R:R ≥1.5 시 BE_WR=40% — 수학적 정합 |
| 3 | R:R | ≥1 | **≥1.5** | WR 낮춤 보상 |
| 4 | TP/SL | dynamic | (유지) | – |
| 5 | **Daily** | **≥+0.2%** | **≥0.0% net @ 0.10% friction (maker-tier)** | Multiplicative gap 인정. Maker rebate path 가정 (BingX maker 0.02% × 2 = 0.04% RT가 best case이나, 50% maker fill 가정 시 ~0.10% RT 현실적). **양수만 통과 — 여전히 진짜 edge 필요** |
| 6 | Per-trade gross | ≥ taker fee | **≥ 0.10% gross (= maker-RT friction)** | (5)와 정합 |
| 7 | Frequency | ≥2/day | **≥1/day** | Sample 충분 (~365 trades/year) |
| 8 | 3-day bootstrap | stable | (유지) — mean > 0, pos_rate ≥ 50% | – |
| 9 | Look-ahead / overfit | none | (유지) | core integrity |

**Trade-off 명시**:
- Adjusted criterion → BingX **maker-rebate 인프라 가정 필수** (LIMIT entry/exit, partial fill 관리, miss rate 수용). Mathematical path 좁아짐.
- 양수 daily만 요구 (원본 +0.2% 대비 4× 완화). 단 0%는 break-even 아니라 "friction 0.10% 모두 회수 후 양수" 조건이므로 real edge 측정.

## 3. Deep Verify Plan (top 5 candidates)

Selection: 17 mechanisms 중 **C1 strict PASS** 이거나 C3 daily 가장 우호적인 5개.

| Rank | Mechanism | C1 Δp50 | C3 daily @ 0.20 | Selection rationale |
|------|-----------|---------|-----------------|---------------------|
| 1 | **κ** (ι + mid-vol regime) | +0.092 | **-0.039** | 세션 최저 C3 daily |
| 2 | **ι** (α + ETH 24-bar break) | **+0.226** | -0.045 | 최강 entry alpha magnitude |
| 3 | **α** (ETH-lag + 고변동성) | +0.160 | -0.080 | 검증된 baseline |
| 4 | **υ** (volume × cross-asset) | +0.132 | -0.468 | 새 axis (volume + cross-asset) |
| 5 | **σ** (mean-rev at structural break) | +0.124 | -0.487 | 유일한 counter-trend C1 PASS |

**Deep tests per candidate** (각 mechanism마다 6-test suite):

1. **Friction breakdown** (0.04, 0.06, 0.08, 0.10, 0.15, 0.20 %): break-even friction 식별
2. **10-seed strict** (random baseline 10 seeds): measurement variance check
3. **Per-horizon fixed exit** (N=4, 8, 12, 16, 24 bars): 최적 exit 호리즌
4. **Walk-forward 5-fold** (expanding): OOS validation
5. **3-way split** (train/val/test): generalization check
6. **3-day bootstrap** (1000 windows): tail risk

**Pass condition (per candidate)**: 
- friction 0.10% net daily > 0
- WR ≥ 40%
- R:R ≥ 1.5
- WF 3/5 positive
- 3-way split test positive
- bootstrap mean > 0, pos_rate ≥ 50%

## 4. Predictions

| Candidate | Predicted (adjusted) | Rationale |
|-----------|---------------------|-----------|
| **κ** | borderline PASS | Best C3 -0.039 → 0.10% friction extrapolated ~ +0.005~0.02 |
| **ι** | borderline PASS | Magnitude 강함, 가장 가능성 큼 |
| **α** | likely FAIL | -0.080 @ 0.20 → 0.10 ~ -0.04 추정 |
| **υ** | FAIL | -0.468 → 0.10 friction에서도 -0.30 추정 |
| **σ** | FAIL | -0.487 worst |

**가장 가능성 높은 surprise**: ι 또는 κ가 friction 0.10%에서 양수 + WF 3/5 + bootstrap 통과.

**Most likely outcome**: 0~1 PASS. 1 PASS 시 maker-rebate 인프라 path 명확화, 0 PASS 시 framework limitation 확정.

## 5. Anti-pattern guards

- **Pre-registered**: 본 문서 deep verify 시작 전 별도 commit. 결과 후 criterion 추가 조정 금지.
- **Selection-after-peek 회피**: top 5 선정은 R1~R8 결과만 사용 (전 부 commit된 raw data).
- **Parameter optimization 금지**: 17 rounds 베이스 파라미터 그대로. Optimization은 winner 식별 후 별도 PDCA.
- **사용자 picking 금지**: 5 candidates 결과 무관하게 matrix 보고. Winner-label 안 함.

## 6. Stop conditions

- 0 PASS adjusted: framework limitation 확정 → 사용자 옵션 B (maker rebate full infra) 또는 다른 paradigm 결정
- 1 PASS: deep PDCA Plan으로 production path 검토
- 2+ PASS: matrix 보고 + 사용자 picking
