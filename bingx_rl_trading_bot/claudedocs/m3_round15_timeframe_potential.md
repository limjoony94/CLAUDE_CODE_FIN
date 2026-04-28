# M3-R15 — Timeframe Axis Potential Assessment (사전 등록)

> **Date**: 2026-04-28
> **Authority**: 사용자 3-phase methodology 지속 적용
> **Origin**: R14에서 15m timeframe × 6 families × broad sweep = 0/6 eligible. 다른 timeframe은 미테스트.

## 1. 배경

R14 distribution-level evidence: 15m signal generation × 다양한 strategy familiy = 모두 phase-2 eligible 미달. 단 R1~R14 모든 mechanism은 15m signal generation. 진짜 untested axes:

| Axis | Tested? |
|------|---------|
| 15m signal | ✓ (모든 mechanism) |
| **1h signal generation** | ❌ |
| **4h signal generation** | ❌ |
| **1d signal generation** | ❌ |

저빈도 timeframe은 friction-frequency 다른 profile + 다른 noise 구조 → 잠재적 미발견 alpha source.

## 2. R15 Plan (Phase 1 적용 — timeframe axis)

3 timeframes × 2 mechanism families (α-style, mean-rev-style) × broad param sweep:

### 1h signal mechanism
- α concept @ 1h (eth_thresh × btc_lag × atr_pctile × N_exit)
- Filter: 4h + 1d trend
- Friction: 0.04% × 2 = 0.08% RT (single-leg perp)

### 4h signal mechanism
- α concept @ 4h
- Filter: 1d trend
- 같은 grid

### 1d signal mechanism
- 단순 momentum (close > N-day SMA + recent return positive)
- Filter: weekly trend (close > 7-day SMA)
- 720일 = 720 bars, smaller sample

각 timeframe당:
- ~50-200 configs grid
- Train/test 60/40 split
- Min sample: train_n ≥ 20, test_n ≥ 20 (timeframe별 자연 sample)

## 3. Potential Metrics (R14 동일)

- p_both_pos: % configs with BOTH train AND test daily > 0
- corr_tt: Pearson(train_daily, test_daily)
- median_test, max_test
- Composite: p_both_pos + corr_tt × 50 + median_test × 100

**Phase 2 eligible**: p_both_pos ≥ 5% AND corr_tt > 0 AND composite ≥ 5

## 4. Predictions

| Timeframe | Predicted | Confidence | Rationale |
|-----------|-----------|-----------|-----------|
| **1h** | borderline | LOW-MED | 가장 가능성 높음 — 15m noise 회피하면서 sample 충분 |
| **4h** | LOW potential | MED | Sample 작음 (~4320 bars), R:R 잠재 큼 but inference 어려움 |
| **1d** | LOW potential | HIGH | 720 bars only — statistical inference 매우 약함 |

**Most likely outcome**: 모든 timeframe도 phase-2 eligible 미달 → directional alpha 부재 cross-timeframe 확정.

**Most likely surprise**: 1h timeframe에서 family 1+ phase-2 eligible → Phase 2 optimization 진행.

## 5. Stop conditions

- 0/3 timeframe phase-2 eligible: directional candle-data 부재 cross-axes 확정 → 사용자 결정 (data acquisition / paradigm 자체 재정의 / accept)
- 1+ timeframe eligible: Phase 2 optimization → 최적 params + final OOS pre-reg
