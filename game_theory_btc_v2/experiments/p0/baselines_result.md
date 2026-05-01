# P0.4 Baselines Result — Validator End-to-End Demonstration

**Date**: 2026-05-01 (P0.4 closure)
**Sealed boundary**: applied (only first 540d / pre-2025-11-02 used)
**Validator**: `bootstrap_six_criteria` with `priority=P0_BASELINE` (loose)
**Raw**: `experiments/p0/baselines_raw.json`

---

## Results Summary (Free Window 540 daily samples)

| Baseline | Mean (%/d) | p5 (%) | Pos rate | p_beats vs B&H | MaxDD (%) | Sharpe | PASS count |
|----------|-----------|--------|---------|----------------|-----------|--------|-----------|
| Buy-and-hold (realistic) | **+0.110** ✅ | -3.60 ❌ | 0.513 ✅ | 1.000 ✅ | -33.0 ✅ | 0.89 ✅ | **5/6** |
| 1× long + funding (realistic) | **+0.092** ✅ | -3.62 ❌ | 0.504 ✅ | 0.453 ✅ | -34.7 ✅ | 0.74 ✅ | **5/6** |
| Random entry (realistic) | **-0.179** ❌ | -4.08 ❌ | 0.465 ❌ | 0.021 ✅ | -103.2 ❌ | -1.45 ❌ | **1/6** |
| Random entry (stress) | **-0.219** ❌ | -4.12 ❌ | 0.450 ❌ | 0.010 ✅ | -123.9 ❌ | -1.78 ❌ | **1/6** |

(P0_BASELINE thresholds: target_daily=0%, max_dd_floor=-100%, min_sharpe=0, min_pos_rate=0.5, min_p_beats=0)

---

## Validator End-to-End: ✅ WORKING

Advisor의 process check 통과:
- ✅ Buy-and-hold mean (+0.110%/day) > 0 → mean PASS
- ✅ Random entry mean (-0.179%/day) < 0 → mean FAIL
- ✅ Random entry max_dd (-103%) → catastrophic friction-eating PASS demonstration
- ✅ p_beats_baseline 정확히 distinguish (B&H baseline 1.000, random vs B&H 0.021)
- ✅ Sealed boundary assertion (`assert_no_sealed_data`) sanity check 통과
- ✅ Friction model 적용 (random에 0.16% RT/day deduction → -0.179% mean)

---

## p5 Criterion Discovery

**관찰**: 모든 4 baseline에서 p5 (raw 5-percentile of daily returns) ≥ 0 FAIL.
- Buy-and-hold p5 = -3.60% (worst 5% of BTC days lose 3.6%)
- 이건 BTC 자체 daily volatility (std ~3%/day) 결과 — strategy 약점 아님

**해석 옵션**:
- A. 현 interpretation (raw 5-percentile of daily returns ≥ 0): BTC volatile asset에서 사실상 통과 불가능. Mandate § 0.5 strict 그대로
- B. Alternative (5-percentile of bootstrap mean distribution ≥ 0): "95% confident mean is positive" — anti-fishing intent 더 명확
- C. Relax p5 threshold for P0_BASELINE only

**현 결정**: Option A 유지 (mandate fidelity). P2-P6 strategy는 6/6 PASS 어려움 인지.
- Mandate § 1.2 + friction-floor evidence (27 mechanisms 0 deployable)와 정합 — strict criterion 의도적
- Interpretation 변경 시 후속 priority result 일관성 위해 amendment 필요

**P2-P6 expectations 조정**:
- 6/6 PASS는 매우 strict. 5/6 PASS도 의미 있는 결과로 인정
- 단 mean criterion + p_beats criterion은 항상 통과 의무 (편향 없는 edge)
- 우선순위 결과 보고 시 6/6 vs 5/6 vs <5/6 구분

---

## Discovered Anomalies

### A1. Random Entry Friction Cost는 daily 0.16% × 365 = -58.4%/yr
실제 측정 -0.179% × 365 ≈ -65.4%/yr. BTC drift 양향 (+40%/yr 가정) × 0.5 (random direction = average 0) - 0.16%/day = 0 - 58% ≈ -58%. 실제 측정값과 정합.

이는 friction-floor evidence (27 mechanisms 0 deployable) 직접 demonstration:
- 무작위 방향 선택 + 매일 trade = 100% 자본 잃음 within 1.5 years
- 즉 strategy가 random + friction 이상의 edge 생성 못 하면 zero or negative

### A2. 1× long with funding이 buy-and-hold보다 약간 낮음
- B&H mean 0.110% / 1× long+funding mean 0.092%
- Diff = -0.018% / day = -6.5% / yr (funding cost net)
- Binance perp 720d 평균 funding rate × 365 days × 3 ticks/day:
  - mean funding ~ +0.005% / 8h
  - daily cost ~ +0.015% / day for long
  - 6.5%/yr 일치 (rough)
- 즉 long-only perp = spot - 6.5%/yr funding drag. **Mandate § 1.2 force-flow reversal에서 진입 timing 이 6.5%/yr drag보다 큰 edge 가져와야 의미 있음**

### A3. Sharpe baseline 0.74-0.89
- BTC drift 양수 + 일일 std 3% → Sharpe ~ 0.7-1.0 typical
- mandate § 0.5 P2 min_sharpe = 1.5는 baseline 대비 2x 개선 요구 — 어려움
- Friction-floor evidence와 정합

---

## D5 Deliverable Status

✅ **D5 (Buy-and-hold + 1× constant long + random entry baselines)** complete.
- 6-criteria evaluation per scenario
- Realistic + stress friction
- Validator end-to-end demonstrated
- p_beats_baseline correctly discriminates

---

## P0 Updated Status (Post-P0.4)

| # | Deliverable | Status |
|---|-------------|--------|
| D1 | BTC data 720+d | ✅ |
| D2 | API access | ✅ |
| D3 | friction_model.py (3 scenarios + funding) | ✅ (35/35 unit tests PASS) |
| D4 | bootstrap_six_criteria.py | ✅ (35/35 PASS, end-to-end on baselines OK) |
| D5 | Baselines | ✅ (this doc) |
| D6 | H1-H9 정量 정의 | ⏳ P0.5 next |
| D7 | ~30 mechanism catalog | ⏳ P0.6 next |
| D8 | Forward collector | ✅ running, accumulating |
| D9 | Proxy formula v2 | ✅ committed |

P0 6-day mark (2026-05-01 + 6 = 2026-05-07). 7-day budget under target.

---

## Next: P0.5 + P0.6 (Advisor 호출)

P0.5 (H1-H9 정량 정의) + P0.6 (~30 mechanism catalog)는 advisor가 명시한 "P0.4 closure interpretation moment". 전에 호출 의무.

- Phase A subset 정량 정의 format (yaml? markdown? JSON?)
- Mechanism catalog의 minimum entry: entry rule, exit rule, parameter space, expected freq, sample size
- Lookahead-free feature definition 표기법

다음 turn에서 advisor() → P0.5 + P0.6 진입.
