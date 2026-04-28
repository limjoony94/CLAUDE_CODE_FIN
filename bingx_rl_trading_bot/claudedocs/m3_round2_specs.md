# M3-R2 — α′ + ι Specs (사전 등록, 별도 commit)

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 ("둘다")
> **Frame**: 2 specs × 5 critiques = 2×5 matrix. Pipeline 재활용 (m3_critique_pipeline.py).
> **Cap**: advisor cap 2 specs maintained.
> **Origin**: M3 finding — α는 entry alpha REAL but exit framework drag. Test 3 N=16 fixed best.

## Spec α′ (Path A: data-grounded exit change)

**Entry rule**: α 그대로 (ETH-lag + 고변동성 regime).
- ETH 15m return prev bar > +0.3% (LONG) / < -0.3% (SHORT)
- BTC 15m return prev bar < +0.1% (LONG) / > -0.1% (SHORT) — BTC lagging
- 1h+4h trend filter aligned
- BTC 15m ATR(14) > 70th pctile of past 200 bars (high-vol regime)

**Exit rule (CHANGED from α)**:
- ✗ NO trail TP
- ✗ NO structural/ATR SL
- ✓ Emergency SL: -1.5% hard
- ✓ Fixed timeout: **N=16 bars (= 4h)** — exit at bar 16 close

**Hypothesis**: α의 entry alpha (Δp50 +0.16, Δ%>fr +13.5)가 trail 2.0×ATR + SL framework로 잠식. Test 3에서 N=16 fixed exit gross +0.081%/trade (trail framework -0.020%/trade 대비 +0.10pp). N=16 fixed가 alpha 보존.

**Risk**: SL 없이 emergency 1.5%만 — 큰 adverse move 시 손실 클 수 있음. 단 timeout 4h이라 노출 짧음.

## Spec ι (Path B: ETH structural break filter)

**Entry rule** (LONG):
- α 모든 조건 (ETH 15m return > 0.3%, BTC lag, 고변동성 regime, 1h+4h LONG)
- **AND ETH 15m close > prev bar's 24-bar high** (ETH 자체 24-bar level break 직후)

**Entry rule** (SHORT, mirror):
- α SHORT 조건 + ETH 24-bar low 깨짐

**Exit rule**: α 동일 (trail 2.0×ATR + structural SL + emergency + timeout 16 bars).

**Hypothesis**: ETH random move는 BTC lag 예측 무력. ETH 자체 **structural break** (24-bar high/low 돌파) 직후의 BTC lag만 진짜 alpha. α의 average-based entry를 conditional로 narrow.

**Risk**: ETH level break은 rare event → sample 매우 적어질 가능성. criterion 7 (≥2/day) 위반 위험.

## Critical Parameters (sensitivity probe targets, C4)

### α′
| Parameter | Base | Sensitivity ±20% |
|-----------|------|------------------|
| eth_thresh | 0.3 | 0.24 / 0.36 |
| btc_lag_thresh | 0.1 | 0.08 / 0.12 |
| atr_pctile | 70.0 | 60.0 / 80.0 |
| timeout_bars | **16** | 13 / 19 |

### ι
| Parameter | Base | Sensitivity ±20% |
|-----------|------|------------------|
| eth_thresh | 0.3 | 0.24 / 0.36 |
| btc_lag_thresh | 0.1 | 0.08 / 0.12 |
| atr_pctile | 70.0 | 60.0 / 80.0 |
| eth_break_lookback | 24 | 19 / 29 |

## Predictions (사전 등록 — calibration source)

### α′ (Path A)

| Critique | Predicted | Confidence | Rationale |
|----------|-----------|-----------|-----------|
| C1 random baseline | PASS | HIGH | Entry rule unchanged from α (6/10 strict pass) |
| C2 look-ahead | PASS | HIGH | No new indicators |
| C3 friction | **FAIL at MED, borderline at BASE** | MED | N=16 gross +0.017%/day - 0.20% friction × 0.21 trades/day = -0.025%/day. 거의 break-even but negative. C4까지 도달 못 할 가능성. |
| C4 overfitting | n/a (skip if C3 fail) | – | – |
| C5 bootstrap | n/a | – | – |

**Honest scenario**: α′ likely close C3 negative loose thread. **Predict FAIL C3.** 단 BASE에서 양수 가능성 ~20% (entry alpha + tighter exit가 expected savings보다 클 수도).

### ι (Path B)

| Critique | Predicted | Confidence | Rationale |
|----------|-----------|-----------|-----------|
| C1 random baseline | borderline PASS or marginal | MED | ETH break filter로 selectivity 향상 vs sample size 감소 trade-off |
| C2 look-ahead | PASS | HIGH | Backward-looking only |
| C3 friction | **FAIL likely** | MED | criterion 7 (≥2/day) 위반 가능성 우선 큰 issue. BT 대상 sample 너무 작아질 수 있음 |
| C4 overfitting | n/a | – | – |
| C5 bootstrap | n/a | – | – |

**Honest scenario**: ι sample size 너무 작아 criterion 7 fail 가능성. C1에서 "NO_SIGNALS" 또는 borderline. 가장 informative outcome: ETH break 후 BTC lag이 진짜 alpha면 strict PASS — α 위에 진짜 conditional alpha 발견.

### Summary distribution
- α′: 1 HIGH PASS prediction (C1, C2), 1 MED FAIL (C3)
- ι: 1 MED borderline (C1), 1 MED FAIL (C3)

**Most likely surprise**: α′ PASS C3 → first monetizable mechanism. **strong signal trigger advisor call**.

**Most likely outcome**: 둘 다 die at C3. Strong convergent evidence (C1 entry alpha exists in cross-instrument family but magnitude insufficient).

## Stop conditions

- 0/2 PASS C5: convergent evidence 강화 + 사용자 결정 (stop / asset shift / paradigm)
- 1/2 PASS C5: 보고 + 사용자 picking (assistant 자체 picking 금지)
- 2/2 PASS C5: 둘 다 보고 + 사용자 picking

## Anti-pattern guard

- 3번째 spec 추가 충동 → Round 3 candidate list로 deferral
- α′ C3 PASS 시 Phase 3 BT 충동 → advisor 호출 (asymmetric trigger)
- "α′ vs ι 더 좋은 것 picking" 금지 — matrix만 보고
