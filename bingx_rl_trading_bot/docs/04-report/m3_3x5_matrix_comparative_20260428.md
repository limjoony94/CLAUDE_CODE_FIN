# M3 — 3×5 Matrix Comparative Analysis (Final Report)

> **Date**: 2026-04-28
> **Scope**: 3 creative mechanisms × 5 critiques (advisor framework). NO winner label.
> **Cumulative**: 4 rounds (R0+R1+R2+R3) + M3 = **30+ cells / 5 critique pipeline applied**

---

## 1. 3×5 Matrix (per-mechanism fail-fast)

| Mechanism | C1 Random | C2 Look-ahead | C3 Friction | C4 Overfit | C5 Bootstrap | Died at |
|-----------|-----------|---------------|-------------|------------|--------------|---------|
| **α** ETH-lag + 고변동성 | **PASS** Δp50 +0.16 (6/10 strict) | **PASS** 0 leaks | **FAIL** all friction negative | skip | skip | **C3** |
| **β** spread × correlation compound | PASS (n=6 small) | PASS | FAIL | skip | skip | C3 |
| **γ** funding × cross-asset | FAIL Δp50 +0.07 < 0.10 | skip | skip | skip | skip | C1 |

**Key per-cell metrics**:
- α C1: Δp50 +0.1604, Δ%>fr +13.53, n_after_seq=173 (강한 sample). 10-seed: avg Δp50 +0.130 ± 0.025
- β C1: Δp50 +0.19, Δ%>fr +40.0 — magnitude 강하나 n=6 (sequencing 후) — **statistical noise 가능성**
- γ C1: Δp50 +0.07 (relaxed PASS but strict FAIL), asym +0.16 favorable

---

## 2. α Deep Verification (advisor "asymmetric call" — 27 cells 중 첫 PASS)

α는 5 critiques 완료 못함. 단 **entry signal alpha는 statistically real**:

### α Robustness (10-seed strict)
- 6/10 seeds STRICT pass (Δp50 ≥ 0.10 AND Δ%>fr ≥ 10)
- 10/10 seeds RELAXED pass (Δp50 ≥ 0.05 AND Δ%>fr ≥ 5)
- avg Δp50 +0.130 ± 0.025

### Entry vs Exit attribution (zero-friction BT)
- α zero-friction daily: -0.020%
- Random entries × same exit: -0.072% (avg 3 seeds)
- **α − random = +0.052%/day** → entry alpha **REAL** preserved through BT

### Friction breakdown
| Friction | Daily net | WR | RR |
|----------|-----------|-----|-----|
| 0.00% | -0.020% | 32.8% | 1.76 |
| 0.10% | -0.080% | 26.3% | 1.59 |
| 0.20% | -0.141% | 23.1% | 1.30 |
| 0.50% | -0.322% | 14.0% | 0.95 |

**α는 friction 0%에서도 negative**. Entry alpha < exit framework drag at every friction level.

### Per-horizon fixed exit (no trail)
| N bars | gross_sum (720d) | gross_avg | gross_WR |
|--------|------------------|-----------|----------|
| 4 | +11.23% | +0.058% | 49.7% |
| 8 | +0.63% | +0.004% | 45.7% |
| 12 | +4.01% | +0.025% | 49.7% |
| **16** | **+12.49%** | **+0.081%** | **51.0%** |
| 24 | +0.03% | +0.000% | 43.1% |

**N=16 fixed exit > trail framework**. Trail K=2.0×ATR이 alpha 일부 잠식.

---

## 3. Cumulative Evidence (5 rounds)

| Round | Cells | PASS (relaxed) | PASS (strict) | Notes |
|-------|-------|----------------|---------------|-------|
| R0 (M1-A) | 1 | 0 | 0 | random > M1-A on filter universe |
| R1 (4 BTC 15m signals) | 4 | 0 | 0 | RSI cross + body anti-edge confirmed |
| R2 (12 cells × 3 dims) | 12 | 0 | 0 | timeframe / NEW signals / no-filter all FAIL |
| R3 (9 cells × 3 families) | 9 | 1 (C.3 artifact) | 0 | C.3 strict re-run failed |
| **M3 (3 mechanisms × 5 critiques)** | 3 | 2 C1 PASS (α, β) | 1 C1 strict (α) | **α entry alpha real, exit drag** |

**29 cells, 1 statistically robust entry signal (α)**, 0 production-grade strategies.

---

## 4. Sketch of "α monetization" question (NOT taken in this round)

α의 entry alpha가 real이지만 magnitude 100× off (사용자 criterion 5: ≥ 0.2%/day):
- α gross +12.49% / 720d (N=16 fixed) ≈ **+0.017%/day**
- 사용자 daily ≥ 0.2% 요구 → **12.5× 격차**

이론적 monetization 후보 (advisor 권고: "ONE structural change with hypothesis"):
- N=16 fixed exit + friction reduction (BingX maker rebate, LIMIT entry) — 단 spec tuning trap 위험
- 이는 **사용자 결정 영역** — 본 보고서는 raw evidence만 제시

---

## 5. Convergent Evidence Pattern (5 rounds)

1. **BTC candle OHLCV alone**: 0/16 PASS across timeframes (5m, 15m, 1h) and signal classes (mean-rev, momentum, breakout, squeeze, volume).
2. **Cross-asset (BTC-ETH)**: 1 strict PASS (α at C1) — entry signal alpha 발견. 단 exit framework + friction이 그 alpha 흡수.
3. **Funding (BingX)**: 1 near-PASS (A.3 sustained extreme, asym +0.18) but γ C1 fail at strict (cross-asset filter dilute).
4. **Volume / OHLCV-내 신호**: 0 near-PASS (4 rounds 일관).

**가장 강한 directional finding**: cross-instrument 정보 (ETH lag, BTC-ETH spread, correlation breakdown) > single-asset OHLCV. 단 magnitude 모두 production criteria 못 도달.

---

## 6. Process Calibration (predictions vs results)

| Round | Predicted PASS | Actual PASS | Calibration |
|-------|----------------|-------------|-------------|
| R0 (M1-A) | uncertain | 0 | confirmed negative |
| R1 (4) | 0~1 marginal | 0 | confirmed |
| R2 (12) | 1~2 marginal positive | 0 | overestimated optimism |
| R3 (9) | 2~3 marginal positive | 1 (C.3 artifact) | C.3 wrong (artifact); A.3 underestimate |
| M3 (3) | 0 expected to survive C5 | 0 surv C5 (α surv C1+C2) | **α C1 strict PASS surprise** |

Calibration insight: **honest pessimism mostly correct**, surprises in:
- R1.V2 squeeze breakout fail (predicted positive)
- R3.A.3 funding sustained extreme strong Δp50 (underestimated)
- **M3.α C1 strict PASS** (most informative surprise — first real entry alpha)

---

## 7. 사용자 결정 영역 (assistant picking 금지)

5 rounds × 30 cells × 5 critiques 종합 → 사용자 옵션:

### Option A — Stop & accept
4 rounds + M3 cumulative evidence: BTC 5m-1h 신호 거의 전부 alpha 부재 또는 magnitude 부족. Capital ($1495) 다른 곳 활용.

### Option B — α 추가 monetization probe
α entry alpha real → exit framework 1 hypothesis (e.g., N=16 fixed exit + maker rebate fee model) → criterion 5 (daily ≥ 0.2%) 가능 여부 검증. **advisor 경고: spec tuning trap, 1회 hypothesis만**.

### Option C — Asset class shift
BTC 데이터 전부 negative. 다른 asset (alt-coins, altcoin pairs)에서 같은 framework 적용. **단 R0~M3 framework는 asset 무관 noise floor 확인 — 다른 asset 동일 결과 가능성 큼**.

### Option D — Different paradigm
Signal-based 단념. Market-making / arbitrage / portfolio-level / DeFi yield 등. **다른 framework 영역 — 본 research arc와 별개**.

### Option E — α monetization probe + Stop fallback
B 1회만 시도. Friction-positive 못 만들면 A로 fallback. **하한 정의된 시도**.

---

## 8. Files Generated

| File | Purpose |
|------|---------|
| `claudedocs/m3_mechanisms_3specs.md` | 3 specs + predictions (사전 등록) |
| `scripts/analysis/m3_critique_pipeline.py` | 5-critique reusable pipeline + 3 specs |
| `scripts/analysis/m3_alpha_deep_verify.py` | α 4-test deep verification |
| `results/m3_3x5_matrix_*.json` | 3×5 raw matrix |
| `results/m3_alpha_deep_*.json` | α deep verify raw data |
| `docs/04-report/m3_3x5_matrix_comparative_20260428.md` | (this report) |

---

## 9. 다음 사용자 메시지에 따른 분기

- A 선택: convergent evidence memo 마무리 + capital decision
- B 선택: α monetization 1-shot probe (N=16 fixed exit + maker fee)
- C 선택: asset universe expansion 1 round (advisor: 3-asset cap)
- D 선택: 다른 conversation (portfolio / market-making 등 — 다른 framework)
- E 선택: B 시도 후 fallback A
