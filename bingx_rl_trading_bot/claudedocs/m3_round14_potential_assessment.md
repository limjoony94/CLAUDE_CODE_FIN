# M3-R14 — Strategy Potential Assessment (사전 등록, 사용자 3-phase methodology)

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 3-phase methodology
>   - Phase 1: 전략 POTENTIAL 평가 (multi-param sweep)
>   - Phase 2: Highest-potential strategy 파라미터 최적화
>   - Phase 3: 실제 거래 적용
> **Origin**: 사용자 지적 — 단일/소수 config로 strategy 단정 성급. R10 grid는 strict OOS criteria로 즉시 reject 했음 — POTENTIAL 평가 누락.

---

## 1. Methodology 변경

### 기존 (R10 등)
- Multi-dim grid → top-K from train → strict OOS pre-reg → 1 fail → drop
- Single-criterion thinking. Selection-after-peek 회피였으나 strategy POTENTIAL 평가 부재.

### 사용자 권장 (R14+)
- Multi-dim grid → **분포 측정** (% positive, robustness, consistency)
- **Strategy POTENTIAL** 정량화 → 가장 높은 family 우선
- Highest-potential within → 파라미터 최적화 후 OOS validation
- 그 결과로 실제 거래

## 2. Potential Metrics (정량 정의)

각 strategy family에 대해, multi-dim grid 결과 distribution으로:

| Metric | 정의 | 의미 |
|--------|------|------|
| **p_train_pos** | % configs with train daily > 0 | Train에서 작동 빈도 |
| **p_test_pos** | % configs with test daily > 0 | OOS에서 작동 빈도 |
| **p_both_pos** | % configs with BOTH train AND test > 0 | Cross-period 안정성 |
| **corr_tt** | Pearson(train_daily, test_daily) | Train-test 일관성 |
| **median_test** | median test daily across configs | "Typical" config 성과 |
| **max_test** | best test daily | Optimum 가능성 |
| **best_config_train_test_diff** | abs(train - test) of best | Stability of best |

### Composite POTENTIAL Score

```
potential = (p_both_pos × 100)        # cross-stable density (most important)
          + (corr_tt × 50)            # signal generalizes
          + (median_test × 100)       # typical config value
```

높을수록 high-potential strategy.

## 3. Strategy families to assess

C1 strict PASS 였거나 borderline (Δp50 ≥ 0.05) 였던 family:

| Family | Base Δp50 | Param dimensions to sweep |
|--------|-----------|---------------------------|
| **α** | +0.160 | eth_thresh × btc_lag × atr_pctile × N_exit |
| **ι** | +0.226 | + eth_break_lookback |
| **κ** | +0.092 | (ι entry + mid-vol regime) |
| **σ** | +0.124 | rsi_thresh × eth_break_lookback × N_exit |
| **υ** | +0.132 | vol_mult × eth_thresh × N_exit |
| **ζ** | +0.079 | eth_accel_thresh × N_exit |

각 family 약 50-200 configs (param × N grid).

Total ~600-1000 configs.

## 4. Compute setup

- **Train/test split**: 60/40
- **Friction**: 0.04% (maker assumption — ω* failed at this level too. Maintained for consistency)
- **Exit**: fixed N timeout, no trail/SL (R9b/R12 finding: trail framework is alpha-killer)
- **Min sample filter**: train_n ≥ 30, test_n ≥ 30 (looser than R10's 50, to capture potential)
- **Pre-registered**: this doc commit before running.

## 5. Pre-registered ranking criterion

### Phase 1 → Phase 2 transition
**Phase 2 진입 조건** (highest-potential strategy 선정):
- p_both_pos ≥ 5% (≥ 5% configs cross-period positive)
- corr_tt > 0 (positive train-test relationship)
- Composite potential ≥ 5

**Phase 2 진행 조건 둘 다 fail** = 모든 family가 noise floor → R10 finding 재확인 → 실제로 directional alpha 부재. 사용자 결정 영역으로 reverting.

### Phase 2: 최적 파라미터 search (선정된 family)
- 선정 family의 grid 결과 중 "robust optimum" 찾기 (p_both_pos × test_daily 최대인 region)
- 그 region 안에서 fine-grained sweep
- OOS validation with stricter pre-reg (R9c level)

## 6. Predictions

| Family | Predicted potential | Rationale |
|--------|---------------------|-----------|
| α | LOW-MED | C1 strong but C3 daily structurally negative |
| ι | MED | Strongest C1 magnitude, but n small at extreme |
| κ | LOW | Mid-vol filter sample 작음 |
| σ | LOW | Counter-trend RR < 1 issue |
| υ | LOW | Volume sample sufficient but C3 -0.47 |
| ζ | LOW | Worst C3 (-0.85) |

**Most likely outcome**: 모든 family p_both_pos < 5%, corr_tt ≈ 0 → no strategy passes potential threshold → R10 finding reinforced → 사용자 결정 (capital decision).

**Most likely surprise**: ι 또는 α가 multi-dim sweep에서 cross-period density >10% 발견 → Phase 2 진행 가능.

## 7. Anti-fix-impulse commitment

- 본 R14 grid 정의대로 한 번 실행. 결과 후 grid 확장 안 함.
- Phase 1 fail 시 그 family는 close. Phase 2 진행 안 함.
- Composite score 계산은 결과 보기 전 정의됨.
