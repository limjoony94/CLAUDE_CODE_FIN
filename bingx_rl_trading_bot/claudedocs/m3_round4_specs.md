# M3-R4 — μ + π Specs (사전 등록)

> **Date**: 2026-04-28
> **Frame**: 2 specs × 5 critiques. Pipeline 재활용.
> **Origin**: M3-R3 둘 다 C1 die (5 mechanisms 누적 dies). 가장 강한 entry alpha (ι Δp50 +0.226) 5× monetization 갭. 새 신호 축 시도.

## Spec μ (funding rate momentum — 1st derivative)

**Hypothesis**: 기존 funding signals (γ, A.3, ξ)는 absolute level. **Funding의 변화율 (acceleration)** 이 더 informative한 새 axis. Funding이 빠르게 상승 = position build-up 가속 = 곧 squeeze 더 가능성. ETH return으로 방향 confirm.

**Entry rule (LONG)**:
- funding_8sum[i] − funding_8sum[i-32] < -0.10% (8h 동안 funding 하락 가속 = shorts 빠르게 build-up)
- AND ETH 15m return prev > 0 (ETH up confirm)
- AND 1h+4h trend LONG

**Entry rule (SHORT)**: mirror (funding 가속 상승 + ETH down).

**Distinction from γ (funding × cross-asset)**: γ는 funding **level**, μ는 funding **change**. 다른 mathematical axis.

**Exit**: α 표준 (trail 2.0×ATR + structural SL + emergency 1.5% + timeout 16).

## Spec π (ETH/BTC ratio trend break)

**Hypothesis**: log(BTC/ETH) ratio가 SMA를 cross하는 시점 = relative strength regime change. ETH가 BTC보다 강해지기 시작 (ratio 하락 cross) = ETH leading → BTC follow가능. 기존 spread mean-rev (β)와 반대 direction (trend follow).

**Entry rule (LONG BTC)**:
- log_ratio[i] crosses ABOVE its SMA20 (BTC가 ETH 대비 강해지기 시작)
- AND BTC return prev > 0
- AND 1h+4h trend LONG

**Entry rule (SHORT BTC)**: ratio crosses BELOW SMA20 (ETH가 BTC 대비 강해지기 시작 → BTC weakness).

**Distinction from β (spread mean-rev)**: β는 z-score extreme **fade**, π는 SMA cross **follow**. 다른 mechanism class.

**Exit**: α 표준.

---

## Critical Parameters (sensitivity)

### μ
| Parameter | Base | ±20% |
|-----------|------|------|
| funding_accel_window | 32 | 26 / 38 |
| funding_accel_thresh | 0.10 | 0.08 / 0.12 |

### π
| Parameter | Base | ±20% |
|-----------|------|------|
| ratio_sma_period | 20 | 16 / 24 |

## Predictions

### μ (funding momentum)
| Critique | Predicted | Confidence |
|----------|-----------|-----------|
| C1 | borderline FAIL | MED — 새 axis but funding 데이터 짧음 (166d) |
| C3 | n/a | – |

### π (ratio trend break)
| Critique | Predicted | Confidence |
|----------|-----------|-----------|
| C1 | borderline FAIL | MED — cross-asset 정보 활용 but SMA cross noisy |
| C3 | n/a | – |

**Most likely outcome**: 둘 다 die (C1) — 7번째 convergent evidence.

**Most likely surprise**: μ가 funding acceleration alpha 발견 → narrow conditional 후보.

## Stop conditions

- 0/2 PASS: 다음 R5 진행 (마지막 시도). 후 7-round 누적 메모.
- 1/2 PASS C5: deep verify + advisor surprise call.

## Anti-pattern guard

- 7-round 결과로 "criterion 5 (daily +0.2%) 도달 불가" 패턴 강해지면 보고 우선.
