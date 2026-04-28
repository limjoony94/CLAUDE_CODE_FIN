# M3-R3 — ν + ξ Specs (사전 등록, 별도 commit)

> **Date**: 2026-04-28
> **Authority**: 사용자 자율 모드 명시 ("이후 조건들을 만족하기 위해 사용자에게 묻지 말고 알아서 자동적 진행")
> **Frame**: 2 specs × 5 critiques = 2×5 matrix. Pipeline 재활용.
> **Cap**: 2 specs/round (advisor 권고 유지).
> **Origin**: M3-R2 결과 (α′, ι 모두 C3 die). 4× 연속 C3 die — 구조적 패턴 확인 중.

---

## Spec ν (volatility regime *transition*, not steady-state)

**Hypothesis**: M3 α는 ATR > 70th pctile **steady-state** regime — 이미 vol 높을 때 진입. 그러나 alpha의 진짜 source가 vol *expansion 시작*이라면, transition (low→high crossing) 직후가 더 informative.

**Entry rule (LONG)**:
- ATR(14)[i] > SMA50(ATR)[i] AND ATR(14)[i-1] ≤ SMA50(ATR)[i-1] (= 상향 크로스)
- AND BTC 15m return prev bar > 0 (방향 confirmation)
- AND 1h+4h trend filter LONG

**Entry rule (SHORT)**:
- ATR transition 동일 (vol expanding)
- AND BTC 15m return prev bar < 0
- AND 1h+4h trend filter SHORT

**Exit rule**: α 표준 exit (trail 2.0×ATR + structural SL + emergency 1.5% + timeout 16 bars).

**Distinction from α**:
- α는 "vol 높음" (steady-state) regime gate
- ν는 "vol 막 expansion" (event-trigger) — fundamentally different signal

**Rationale**:
- Cross가 한 시점이므로 sample 적을 가능성 — criterion 7 (≥2/day) 위험
- Vol expansion 자체는 directionless이므로 BTC return + trend로 방향 분리

## Spec ξ (compound: funding extreme × ETH structural break)

**Hypothesis**: 강한 individual signal 두 개 compound으로 narrow conditional alpha 발견:
- A.3 (funding 8sum sustained extreme, asym +0.18 favorable)
- ι (ETH 24-bar high/low break, Δp50 +0.226 — 가장 강한 entry alpha)

**Entry rule (LONG)**:
- funding_8sum[i] ≤ -0.24% (shorts crowded → squeeze potential)
- AND ETH 15m close[i] > prev bar's 24-bar high (ETH breakout)
- AND 1h+4h trend filter LONG

**Entry rule (SHORT)**:
- funding_8sum[i] ≥ +0.24% (longs crowded → flush potential)
- AND ETH 15m close[i] < prev bar's 24-bar low
- AND 1h+4h trend filter SHORT

**Exit rule**: α 표준 exit.

**Distinction from γ**:
- γ는 funding × RSI extreme **counter-trend** (fade)
- ξ는 funding × ETH break **with trend** (continuation)
- 다른 mechanism class

**Risk**:
- 두 rare event 곱 → sample 더 적어질 가능성. criterion 7 위반 risk 매우 큼
- Funding 데이터 166 days만 보유 (R3 제약) — 720일 전체 BT는 funding NaN 구간 자동 skip

---

## Critical Parameters (sensitivity probe targets, C4)

### ν
| Parameter | Base | Sensitivity ±20% |
|-----------|------|------------------|
| atr_sma_period | 50 | 40 / 60 |
| btc_lag_thresh | 0.0 | -0.05 / +0.05 (return sign zone) |
| timeout_bars | 16 | 13 / 19 |

### ξ
| Parameter | Base | Sensitivity ±20% |
|-----------|------|------------------|
| funding_sum_thresh | 0.24 | 0.19 / 0.29 |
| eth_break_lookback | 24 | 19 / 29 |

---

## Predictions (사전 등록 — calibration source)

### ν (volatility transition)

| Critique | Predicted | Confidence | Rationale |
|----------|-----------|-----------|-----------|
| C1 random baseline | borderline PASS or marginal | MED | Vol expansion event는 random보다 informative 가능. 단 sample size 우려. ι 같은 +0.226 magnitude 가능성 낮음 |
| C2 look-ahead | PASS | HIGH | Backward-looking only |
| C3 friction | **FAIL** | HIGH | M3 4× 연속 C3 die 패턴. ν도 entry alpha 찾아도 magnitude 부족 가능성 가장 큼 |
| C4 / C5 | n/a | – | – |

### ξ (funding × ETH break compound)

| Critique | Predicted | Confidence | Rationale |
|----------|-----------|-----------|-----------|
| C1 random baseline | UNCERTAIN | LOW | 두 strong signal compound 시너지 가능 vs sample 폭락으로 noise 우세 가능. 가장 informative critique |
| C2 look-ahead | PASS | HIGH | Backward-looking only |
| C3 friction | **borderline FAIL or marginal PASS** | MED | C1 strict PASS 시 narrow conditional alpha 후보 — most likely surprise scenario |
| C4 / C5 | n/a if C3 fail | – | – |

### Summary distribution
- ν: 1 MED PASS (C1), 1 HIGH FAIL (C3) — likely die at C3
- ξ: 1 LOW UNCERTAIN (C1), 1 MED FAIL (C3) — most informative outcome

**Most likely surprise**: ξ PASS C3 → first compound conditional alpha. 추가 confirmation rounds 필요.

**Most likely outcome**: 둘 다 die (C1 또는 C3) — 5번째 convergent evidence.

## Stop conditions (autonomous mode)

- 0/2 PASS C5: continue R4 queue (μ + π)
- 1/2 PASS C5: deep verify + advisor surprise call
- 2/2 PASS C5: surprise — full verification + advisor

## Anti-pattern guard

- ξ가 sample <30이면 measurement noise 가능성 — C1 통과해도 strict 10-seed 즉시 추가 검증 필요
- "fix-impulse" 회피: 둘 다 fail 시 R4 자동 진행 (parameter 재조정 금지)
